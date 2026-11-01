#!/usr/bin/env python3
"""Convert a BitNet-VL checkpoint's vision tower + projector to an mmproj GGUF for llama.cpp / mtmd.

Input (checkpoint contract, see docs/bitnet-vl-design.md):
    ckpt/config.json               {"model_type": "bitnet_vl", "pixel_shuffle": 2, "vision_ternary": bool, ...}
    ckpt/vision_tower.safetensors  HF SiglipVisionModel state dict (vision_model.embeddings..., vision_model.encoder.layers.N...)
    ckpt/projector.safetensors     norm.{weight,bias}, fc1.{weight,bias}, fc2.{weight,bias}

Output: a GGUF with general.architecture = "clip" and clip.projector_type = "bitnet_vl" (PROJECTOR_TYPE_BITNET_VL
in tools/mtmd/clip.cpp). The encoder runs through the stock SigLIP graph (clip_graph_siglip).

With --vision-i2s the encoder linears (q, k, v, out, fc1, fc2) are ternarized (absmean, per tensor) and packed as
GGML_TYPE_I2_S; the I2_S matmul kernel then quantizes the activations per token (absmax int8), i.e. it computes the
BitLinear forward used during ternary-encoder training. Everything else stays F16 (2D weights) / F32 (1D, norms, biases).

Usage:
    python multimodal/convert_bitnet_vl_mmproj.py ckpt/ --outfile mmproj-bitnet-vl.gguf [--vision-i2s] [--outtype f16|f32]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "3rdparty" / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

from safetensors import safe_open  # noqa: E402

logger = logging.getLogger("convert-bitnet-vl-mmproj")

EPS = 1e-5          # BitLinear absmean clamp (multimodal/bitnet_vl/bitlinear.py, utils/convert-hf-to-gguf-bitnet.py)
I2S_BLOCK = 128     # I2_S packs 128-value blocks; the x86 kernels require K % 128 == 0

# HF SiglipVisionModel key (after the "vision_model." prefix) -> clip.cpp tensor name
_LAYER_MAP = {
    "self_attn.q_proj":   "attn_q",
    "self_attn.k_proj":   "attn_k",
    "self_attn.v_proj":   "attn_v",
    "self_attn.out_proj": "attn_out",
    "layer_norm1":        "ln1",
    "layer_norm2":        "ln2",
    "mlp.fc1":            "ffn_up",
    "mlp.fc2":            "ffn_down",
}
_TERNARY_LINEARS = {"attn_q", "attn_k", "attn_v", "attn_out", "ffn_up", "ffn_down"}
_GLOBAL_MAP = {
    "embeddings.patch_embedding.weight":    "v.patch_embd.weight",
    "embeddings.patch_embedding.bias":      "v.patch_embd.bias",
    "embeddings.position_embedding.weight": "v.position_embd.weight",
    "post_layernorm.weight":                "v.post_ln.weight",
    "post_layernorm.bias":                  "v.post_ln.bias",
}
_PROJ_MAP = {
    "norm.weight": "mm.input_norm.weight",
    "norm.bias":   "mm.input_norm.bias",
    "fc1.weight":  "mm.1.weight",
    "fc1.bias":    "mm.1.bias",
    "fc2.weight":  "mm.2.weight",
    "fc2.bias":    "mm.2.bias",
}


def _load_i2s_packer():
    """Reuse quantize_to_i2_s from utils/convert-hf-to-gguf-bitnet.py (the packer the BitNet LLM path uses)."""
    path = REPO / "utils" / "convert-hf-to-gguf-bitnet.py"
    sys.path.insert(0, str(REPO / "utils"))
    try:
        spec = importlib.util.spec_from_file_location("_bitnet_hf_converter", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.quantize_to_i2_s
    except Exception as e:  # e.g. sentencepiece missing (imported by utils/convert.py)
        logger.warning(f"could not import quantize_to_i2_s from {path} ({e}); using the vendored copy")
        return _quantize_to_i2_s_vendored
    finally:
        sys.path.pop(0)


def _quantize_to_i2_s_vendored(w: np.ndarray, override_scale: float) -> np.ndarray:
    """Copy of quantize_to_i2_s(w, override_scale) for already-ternary w (values in {-1, 0, 1}).

    Layout: flattened row-major values in 128-value blocks; byte j of a 32-byte block holds elements
    j, 32+j, 64+j, 96+j at bit offsets 6, 4, 2, 0; codes -1->0, 0->1, +1->2; then a 32-byte tail whose first 4 bytes
    are the f32 per-tensor scale (at offset n/4).
    """
    n = w.size
    q = np.ones(n, dtype=np.uint8)
    flat = w.reshape(-1)
    q[flat > 0.5] = 2
    q[flat < -0.5] = 0
    pad = (I2S_BLOCK - n % I2S_BLOCK) % I2S_BLOCK
    if pad:
        q = np.pad(q, (0, pad), constant_values=1)
    q = q.reshape(-1, 4, 32)
    packed = ((q[:, 0] << 6) | (q[:, 1] << 4) | (q[:, 2] << 2) | q[:, 3]).reshape(-1).astype(np.uint8)
    out = np.zeros(n // 4 + 32, dtype=np.uint8)
    out[: n // 4] = packed[: n // 4]
    out[n // 4: n // 4 + 4] = np.frombuffer(np.float32(override_scale).tobytes(), dtype=np.uint8)
    return out


def ternarize(w: np.ndarray) -> tuple[np.ndarray, float]:
    """absmean ternary, same op order as bitnet_vl.bitlinear.ternary_weight / BitnetModel.weight_quant."""
    w = w.astype(np.float32)
    inv = np.float32(1.0) / np.float32(max(float(np.abs(w).mean(dtype=np.float32)), EPS))
    q = np.clip(np.round(w * inv), -1, 1).astype(np.float32)
    return q, float(np.float32(1.0) / inv)


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    out = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k).float().numpy()
    return out


def vision_hparams(cfg: dict, vt: dict[str, np.ndarray], args) -> dict:
    """Derive SigLIP hparams from the weights (config.json only names the HF model), with optional overrides
    from cfg["vision_config"] (HF SiglipVisionConfig dict) if the checkpoint provides it."""
    vc = cfg.get("vision_config", {}) or {}
    pw = vt["embeddings.patch_embedding.weight"]          # [hidden, 3, p, p]
    hidden, _, patch, _ = pw.shape
    n_pos = vt["embeddings.position_embedding.weight"].shape[0]
    side = int(round(math.sqrt(n_pos)))
    assert side * side == n_pos, f"position embedding count {n_pos} is not a square"
    n_layer = 1 + max(int(k.split(".")[2]) for k in vt if k.startswith("encoder.layers."))
    n_ff = vt["encoder.layers.0.mlp.fc1.weight"].shape[0]
    n_head = args.vision_heads or vc.get("num_attention_heads") or hidden // 64
    hp = dict(
        hidden=int(hidden), patch=int(patch), image_size=int(cfg.get("image_size", vc.get("image_size", side * patch))),
        n_layer=int(n_layer), n_ff=int(n_ff), n_head=int(n_head),
        eps=float(vc.get("layer_norm_eps", 1e-6)),
    )
    assert hp["image_size"] == side * patch, \
        f"image_size {hp['image_size']} != sqrt(n_pos)*patch = {side * patch} (clip.cpp needs a fixed-size SigLIP)"
    assert hidden % n_head == 0
    return hp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path, help="BitNet-VL checkpoint directory")
    ap.add_argument("--outfile", type=Path, required=True)
    ap.add_argument("--vision-i2s", action="store_true",
                    help="ternarize + pack encoder linears (q/k/v/out/fc1/fc2) as I2_S")
    ap.add_argument("--outtype", choices=["f16", "f32"], default="f16",
                    help="type for non-I2_S 2D weights (encoder linears without --vision-i2s, patch embed, projector)")
    ap.add_argument("--vision-heads", type=int, default=None, help="override attention head count (default hidden/64)")
    ap.add_argument("--no-post-ln", action="store_true",
                    help="drop vision_model.post_layernorm (use if the PyTorch model feeds hidden_states[-1] "
                         "instead of last_hidden_state to the projector)")
    ap.add_argument("--feature-layer", type=int, default=None,
                    help="HF hidden_states index fed to the projector (e.g. -2). Truncates the encoder and drops "
                         "post_layernorm. Default: cfg['vision_feature_layer'] or last_hidden_state")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    cfg = json.loads((args.ckpt / "config.json").read_text())
    if cfg.get("model_type") != "bitnet_vl":
        logger.warning(f"config.json model_type is {cfg.get('model_type')!r}, expected 'bitnet_vl'")
    if args.vision_i2s and not cfg.get("vision_ternary", False):
        logger.warning("--vision-i2s on a checkpoint with vision_ternary=false: the encoder was not trained with "
                       "BitLinear, expect a large accuracy drop")

    # the C++ graph (tools/mtmd/models/siglip.cpp, PROJECTOR_TYPE_BITNET_VL) hard-codes these choices
    expected = {"projector_act": "gelu", "projector_norm_eps": 1e-5, "pixel_shuffle_order": "idefics3",
                "vision_feature": "last_hidden_state_post_layernorm"}
    for k, v in expected.items():
        if k in cfg and cfg[k] != v and not (k == "vision_feature" and (args.no_post_ln or args.feature_layer is not None)):
            raise SystemExit(f"config.json {k}={cfg[k]!r} is not supported by the bitnet_vl mmproj graph (needs {v!r})")

    vt_raw = read_safetensors(args.ckpt / "vision_tower.safetensors")
    vt = {k.removeprefix("vision_model."): v for k, v in vt_raw.items()}
    proj = read_safetensors(args.ckpt / "projector.safetensors")
    hp = vision_hparams(cfg, vt, args)
    merge = int(cfg.get("pixel_shuffle", 2))

    feature_layer = args.feature_layer if args.feature_layer is not None else cfg.get("vision_feature_layer")
    n_keep = hp["n_layer"]
    use_post_ln = not args.no_post_ln and "post_layernorm.weight" in vt
    if feature_layer is not None:
        # hidden_states[i] (HF): i = 0 is the embedding output, i = k is after layer k-1; none has post_layernorm
        idx = feature_layer if feature_layer >= 0 else hp["n_layer"] + 1 + feature_layer
        assert 1 <= idx <= hp["n_layer"], f"bad feature layer {feature_layer}"
        n_keep, use_post_ln = idx, False
    logger.info(f"vision: {hp}, layers kept {n_keep}, post_ln {use_post_ln}, pixel_shuffle {merge}")

    n_mm_in = proj["fc1.weight"].shape[1]
    n_embd_text = proj["fc2.weight"].shape[0]
    assert n_mm_in == hp["hidden"] * merge * merge, \
        f"projector input {n_mm_in} != hidden*merge^2 = {hp['hidden'] * merge * merge}"
    assert proj["norm.weight"].shape[0] == n_mm_in
    if "llm_hidden" in cfg and cfg["llm_hidden"] != n_embd_text:
        logger.warning(f"llm_hidden {cfg['llm_hidden']} != projector output {n_embd_text}")
    n_tok = (hp["image_size"] // hp["patch"] // merge) ** 2
    if "num_image_tokens" in cfg and cfg["num_image_tokens"] != n_tok:
        logger.warning(f"num_image_tokens {cfg['num_image_tokens']} != computed {n_tok}")

    w = gguf.GGUFWriter(str(args.outfile), arch="clip")
    w.add_type(gguf.GGUFType.MMPROJ)
    w.add_name(cfg.get("name", "BitNet-VL mmproj"))
    w.add_string("general.description", f"BitNet-VL vision tower ({cfg.get('vision_model_name', 'siglip')}) + projector"
                      f"{' [I2_S encoder]' if args.vision_i2s else ''}")
    w.add_file_type(41 if args.vision_i2s else  # 41 = LLAMA_FTYPE_MOSTLY_I2_S (llama.h)
                    (gguf.LlamaFileType.MOSTLY_F16 if args.outtype == "f16" else gguf.LlamaFileType.ALL_F32))
    w.add_clip_has_vision_encoder(True)
    w.add_clip_projector_type("bitnet_vl")
    w.add_vision_projection_dim(int(n_embd_text))
    w.add_vision_image_size(hp["image_size"])
    w.add_vision_patch_size(hp["patch"])
    w.add_vision_embedding_length(hp["hidden"])
    w.add_vision_feed_forward_length(hp["n_ff"])
    w.add_vision_block_count(n_keep)
    w.add_vision_head_count(hp["n_head"])
    w.add_vision_attention_layernorm_eps(hp["eps"])
    w.add_vision_image_mean([float(x) for x in cfg.get("image_mean", [0.5, 0.5, 0.5])])  # SigLIP default 0.5
    w.add_vision_image_std([float(x) for x in cfg.get("image_std", [0.5, 0.5, 0.5])])
    w.add_vision_use_gelu(True)                 # SigLIP MLP: gelu_pytorch_tanh == ggml_gelu
    w.add_vision_projector_scale_factor(merge)

    pack_i2s = _load_i2s_packer() if args.vision_i2s else None
    ftype_np = np.float16 if args.outtype == "f16" else np.float32
    ftype_gg = gguf.GGMLQuantizationType.F16 if args.outtype == "f16" else gguf.GGMLQuantizationType.F32
    stats = {"I2_S": 0, "F16": 0, "F32": 0}

    def add(name: str, data: np.ndarray, ternary_ok: bool = False) -> None:
        if data.ndim == 1:
            arr, qt = data.astype(np.float32), gguf.GGMLQuantizationType.F32
        elif ternary_ok and pack_i2s is not None:
            M, K = data.shape
            if K % I2S_BLOCK != 0:
                logger.warning(f"{name}: K={K} is not a multiple of {I2S_BLOCK}; keeping {args.outtype.upper()}")
                arr, qt = data.astype(ftype_np), ftype_gg
            else:
                q, scale = ternarize(data)
                packed = pack_i2s(q, override_scale=scale)
                assert packed.size == M * K // 4 + 32
                w.add_tensor(name, packed, raw_shape=(M, K), raw_dtype=gguf.GGMLQuantizationType.I2_S)
                stats["I2_S"] += 1
                logger.debug(f"{name:32s} I2_S {(M, K)} scale={scale:.6g} zeros={np.mean(q == 0):.3f}")
                return
        elif name == "v.position_embd.weight":
            arr, qt = data.astype(np.float32), gguf.GGMLQuantizationType.F32
        else:
            arr, qt = data.astype(ftype_np), ftype_gg
        w.add_tensor(name, arr, raw_dtype=qt)
        stats[qt.name] += 1
        logger.debug(f"{name:32s} {qt.name} {data.shape}")

    for hf_name, gg_name in _GLOBAL_MAP.items():
        if hf_name not in vt:
            continue
        if gg_name.startswith("v.post_ln") and not use_post_ln:
            continue
        add(gg_name, vt[hf_name])

    used = set()
    for il in range(hp["n_layer"]):
        for hf_sub, gg_sub in _LAYER_MAP.items():
            for kind in ("weight", "bias"):
                k = f"encoder.layers.{il}.{hf_sub}.{kind}"
                if k not in vt:
                    continue
                used.add(k)
                if il >= n_keep:
                    continue
                add(f"v.blk.{il}.{gg_sub}.{kind}", vt[k], ternary_ok=(kind == "weight" and gg_sub in _TERNARY_LINEARS))

    for hf_name, gg_name in _PROJ_MAP.items():
        add(gg_name, proj[hf_name])  # projector stays full precision by design

    skipped = [k for k in vt if k not in used and k not in _GLOBAL_MAP]
    if skipped:
        logger.info(f"skipped {len(skipped)} vision tensors not used by the mmproj graph (e.g. {skipped[:3]})")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    logger.info(f"wrote {args.outfile}  tensors: {stats}  visual tokens/image: {n_tok}  n_embd out: {n_embd_text}")


if __name__ == "__main__":
    main()
