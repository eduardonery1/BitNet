#!/usr/bin/env python3
"""Convert a BitNet-MoE checkpoint (HF-style dir, see docs/bitnet-moe-design.md) to a GGUF with arch `bitnet-moe`.

Checkpoint contract (per layer N, everything else uses the dense BitNet names):
    model.layers.N.mlp.router.weight                              [E, d]      -> blk.N.ffn_gate_inp.weight   (F32 always)
    model.layers.N.mlp.experts.{e}.{gate,up,down}_proj.weight     latent FP   -> blk.N.ffn_{gate,up,down}_exps.weight
    model.layers.N.mlp.ffn_sub_norm.weight                        [n_ff]      -> blk.N.ffn_sub_norm.weight   (F32)

Expert tensors are stacked to numpy shape [E, n_out, n_in] (= ggml ne [n_in, n_out, E]).

Weight handling (BitLinear weights = attention q/k/v/o and all expert projections):
    * default: ternarize each matrix with absmean (`BitnetModel.weight_quant`, i.e. bitnet_vl.bitlinear.weight_quant),
      per tensor for attention and per expert for the experts.
    * --outtype f32|f16: attention weights are stored as the dequantized ternary values (as the dense converter does).
      Experts are stored as exact ternary codes {-1,0,1} plus one F32 scale per expert in
      `blk.N.ffn_*_exps.scale` [E] (applied by the graph after mul_mat_id).
    * --outtype i2_s: attention weights become standard 2D I2_S; each expert is packed with `quantize_to_i2_s` and
      the E blobs (packed bytes + 32-byte scale tail each) are concatenated -> 3D I2_S tensor.
    * --expert-type overrides the expert storage (f32, f16, q8_0, i2_s) independently of --outtype.
    * --no-ternarize stores the latent float weights unchanged (f32/f16 only), for float-reference debugging.

Usage:
    python utils/convert-bitnet-moe-to-gguf.py CKPT_DIR --outtype i2_s --outfile ggml-model-i2_s.gguf
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "3rdparty" / "llama.cpp" / "gguf-py"))
sys.path.insert(1, str(Path(__file__).resolve().parent))  # for `convert` imported by the dense converter

import gguf  # noqa: E402

_spec = importlib.util.spec_from_file_location("convert_hf_to_gguf_bitnet",
                                               Path(__file__).resolve().parent / "convert-hf-to-gguf-bitnet.py")
_dense = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dense)
BitnetModel = _dense.BitnetModel
quantize_to_i2_s = _dense.quantize_to_i2_s

logger = logging.getLogger("bitnet-moe-to-gguf")

LLAMA_FTYPE = {"f32": 0, "f16": 1, "i2_s": 41}  # llama_ftype (I2_S = LLAMA_FTYPE_MOSTLY_I2_S = 41, not 40)
QTYPE = gguf.GGMLQuantizationType
ATTN_MAP = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output"}
EXP_MAP = {"gate_proj": "ffn_gate_exps", "up_proj": "ffn_up_exps", "down_proj": "ffn_down_exps"}


def ternary_codes(w: torch.Tensor) -> tuple[np.ndarray, float]:
    """absmean ternarization with the same op order as BitnetModel.weight_quant: returns codes {-1,0,1} and scale."""
    w = w.float()
    inv = 1.0 / w.abs().mean().clamp(min=1e-5)
    q = (w * inv).round().clamp(-1, 1)
    return q.numpy().astype(np.float32), float(1.0 / inv)


class BitnetMoeModel(BitnetModel):
    model_arch = gguf.MODEL_ARCH.BITNET_MOE

    def __init__(self, *a, outtype: str, expert_type: str, ternarize: bool, embd_type: str, **kw):
        self.outtype = outtype
        self.expert_type = expert_type
        self.ternarize = ternarize
        self.embd_type = embd_type
        super().__init__(*a, **kw)

    def set_gguf_parameters(self):
        hp = self.hparams
        w = self.gguf_writer
        w.add_name(self.dir_model.name)
        w.add_block_count(self.block_count)
        if "max_position_embeddings" in hp:
            w.add_context_length(hp["max_position_embeddings"])
        w.add_embedding_length(hp["hidden_size"])
        w.add_feed_forward_length(hp["intermediate_size"])
        w.add_head_count(hp["num_attention_heads"])
        w.add_head_count_kv(hp.get("num_key_value_heads", hp["num_attention_heads"]))
        rope_theta = hp.get("rope_theta", (hp.get("rope_parameters") or {}).get("rope_theta"))
        if rope_theta is None:
            raise ValueError("config.json has no rope_theta")
        w.add_rope_freq_base(rope_theta)
        w.add_layer_norm_rms_eps(hp["rms_norm_eps"])
        w.add_vocab_size(hp["vocab_size"])
        w.add_rope_dimension_count(hp.get("head_dim", hp["hidden_size"] // hp["num_attention_heads"]))
        w.add_expert_count(hp["num_experts"])
        w.add_expert_used_count(hp["num_experts_per_tok"])
        if not hp.get("norm_topk_prob", True):
            raise ValueError("bitnet-moe graph implements norm_topk_prob = true only")
        w.add_file_type(LLAMA_FTYPE[self.outtype])

    # ------------------------------------------------------------------ tensors
    def _float(self, a: np.ndarray, typ: str) -> tuple[np.ndarray, QTYPE]:
        if typ == "f32":
            return a.astype(np.float32), QTYPE.F32
        if typ == "f16":
            return a.astype(np.float16), QTYPE.F16
        raise ValueError(typ)

    def _add(self, name, data, qtype, raw_shape=None):
        shape = raw_shape if raw_shape is not None else data.shape
        logger.info(f"{name:34s} {qtype.name:5s} ne={list(reversed(shape))}")
        self.gguf_writer.add_tensor(name, data, raw_shape=raw_shape, raw_dtype=qtype)

    def _attn(self, name: str, w: torch.Tensor):
        w = w.float()
        if self.outtype == "i2_s":
            if w.shape[1] % 128:
                raise ValueError(f"{name}: K={w.shape[1]} not a multiple of 128, cannot store as I2_S")
            data = quantize_to_i2_s(self.weight_quant(w).numpy())
            self._add(name, data, QTYPE.I2_S, raw_shape=tuple(w.shape))
        else:
            if self.ternarize:
                w = self.weight_quant(w)
            self._add(name, *self._float(w.numpy(), self.outtype))

    def _experts(self, base: str, mats: list[torch.Tensor]):
        E = len(mats)
        n_out, n_in = mats[0].shape
        et = self.expert_type
        if et == "i2_s":
            if n_in % 128:
                raise ValueError(f"{base}: K={n_in} not a multiple of 128, cannot store as I2_S")
            blobs = [quantize_to_i2_s(self.weight_quant(m.float()).numpy()) for m in mats]
            assert all(b.size == n_out * n_in // 4 + 32 for b in blobs)
            self._add(base + ".weight", np.concatenate(blobs), QTYPE.I2_S, raw_shape=(E, n_out, n_in))
            return
        if self.ternarize:
            codes, scales = zip(*(ternary_codes(m) for m in mats))
            stacked = np.stack(codes)
            scale = np.asarray(scales, dtype=np.float32)
        else:
            stacked = np.stack([m.float().numpy() for m in mats])
            scale = None
        if et == "q8_0":
            data = gguf.quants.quantize(stacked.astype(np.float32), QTYPE.Q8_0)
            self._add(base + ".weight", data, QTYPE.Q8_0, raw_shape=stacked.shape)
        else:
            self._add(base + ".weight", *self._float(stacked, et))
        if scale is not None:
            self._add(base + ".scale", scale, QTYPE.F32)

    def write_tensors(self):
        sd = dict(self.get_tensors())
        E = self.hparams["num_experts"]
        used = set()

        def take(k):
            used.add(k)
            return sd[k]

        embd = take("model.embed_tokens.weight").float().numpy()
        self._add("token_embd.weight", *self._float(embd, self.embd_type))
        self._add("output_norm.weight", take("model.norm.weight").float().numpy(), QTYPE.F32)
        if "lm_head.weight" in sd and not self.hparams.get("tie_word_embeddings", False):
            self._add("output.weight", *self._float(take("lm_head.weight").float().numpy(), self.embd_type))
        elif "lm_head.weight" in sd:
            used.add("lm_head.weight")

        for il in range(self.block_count):
            p = f"model.layers.{il}."
            b = f"blk.{il}."
            self._add(b + "attn_norm.weight", take(p + "input_layernorm.weight").float().numpy(), QTYPE.F32)
            self._add(b + "attn_sub_norm.weight", take(p + "self_attn.attn_sub_norm.weight").float().numpy(), QTYPE.F32)
            self._add(b + "ffn_norm.weight", take(p + "post_attention_layernorm.weight").float().numpy(), QTYPE.F32)
            self._add(b + "ffn_sub_norm.weight", take(p + "mlp.ffn_sub_norm.weight").float().numpy(), QTYPE.F32)
            for hf, gg in ATTN_MAP.items():
                self._attn(b + gg + ".weight", take(p + f"self_attn.{hf}.weight"))
            self._add(b + "ffn_gate_inp.weight", take(p + "mlp.router.weight").float().numpy(), QTYPE.F32)
            for hf, gg in EXP_MAP.items():
                mats = [take(p + f"mlp.experts.{e}.{hf}.weight") for e in range(E)]
                self._experts(b + gg, mats)

        unused = sorted(set(sd) - used)
        unused = [k for k in unused if not k.endswith(("rotary_emb.inv_freq",))]
        if unused:
            raise ValueError(f"unconverted checkpoint tensors (contract mismatch?): {unused[:10]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="checkpoint dir (config.json + *.safetensors + tokenizer)")
    ap.add_argument("--outfile", type=Path)
    ap.add_argument("--outtype", choices=["f32", "f16", "i2_s"], default="f32")
    ap.add_argument("--expert-type", choices=["f32", "f16", "q8_0", "i2_s"], default=None,
                    help="expert storage type (default: same as --outtype)")
    ap.add_argument("--embd-type", choices=["f32", "f16"], default=None,
                    help="token embedding / output type (default: f32 for --outtype f32, else f16)")
    ap.add_argument("--no-ternarize", action="store_true", help="store latent float weights (f32/f16 only)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO)

    expert_type = a.expert_type or a.outtype
    if a.no_ternarize and ("i2_s" in (a.outtype, expert_type)):
        ap.error("--no-ternarize is incompatible with I2_S")
    embd_type = a.embd_type or ("f32" if a.outtype == "f32" else "f16")
    cfg = json.load(open(a.model / "config.json"))
    if "num_experts" not in cfg or "BitNetMoEForCausalLM" not in cfg.get("architectures", []):
        ap.error("config.json is not a BitNet-MoE checkpoint (needs architectures=[BitNetMoEForCausalLM], num_experts)")
    outfile = a.outfile or (a.model / f"ggml-model-{a.outtype}.gguf")

    ftype_qt = {"f32": QTYPE.F32, "f16": QTYPE.F16, "i2_s": QTYPE.I2_S}[a.outtype]
    with torch.inference_mode():
        m = BitnetMoeModel(a.model, ftype_qt, outfile, False, False, outtype=a.outtype, expert_type=expert_type,
                           ternarize=not a.no_ternarize, embd_type=embd_type)
        m.set_gguf_parameters()
        m.set_vocab()
        m.write()
    logger.info(f"wrote {outfile}")


if __name__ == "__main__":
    main()
