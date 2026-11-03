"""Tiny random BitNet-MoE checkpoint (contract layout) + reference logits for C++ parity tests.

    cd multimodal && python -m bitnet_vl.moe.make_tiny_moe_ckpt OUT_DIR [--noise 0.3] [--layers 2] [--seed 0]

Dense tiny BitNet (hidden 256, n_ff 512, 4 heads / 2 kv heads, relu2, tied embeddings, Llama-3 BPE tokenizer rebuilt
offline, vocab 128256) with perturbed norms, upcycled to E=4 top-2 with relative expert noise so the experts (and their
ternarized weights) differ and routing matters. Saved in bf16 (``--dtype``); the references are computed in fp32 from
the checkpoint as reloaded from disk.

Outputs in OUT_DIR besides the checkpoint:
  reference.json             prompt, token ids, per mode top-20 ids/values of the last-position logits, per-layer
                             routing (top-k expert ids, renormalized weights, margin p_k - p_{k+1}) for every token,
                             and the max logit change caused by routing (sanity check that experts differ)
  reference_logits.npy       fp32 [vocab] last-position logits, ternary QAT forward (what I2_S should reproduce)
  reference_logits_float.npy fp32 [vocab] last-position logits with float (non-ternarized) weights
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BitNetForCausalLM

from ..make_tiny_ckpt import tiny_llm_config
from ..modeling import _strip_quant_config
from ..tokenizer_utils import build_llama3_tokenizer
from .modeling_moe import BitNetMoEForCausalLM, route
from .upcycle import upcycle

PROMPT = "The quick brown fox jumps over the lazy dog. BitNet experts route tokens"


def build_tiny_dense(seed: int = 0, layers: int = 2, vocab_size: int = 128256, hidden: int = 256,
                     intermediate: int = 512) -> BitNetForCausalLM:
    torch.manual_seed(seed)
    cfg = _strip_quant_config(tiny_llm_config(vocab_size=vocab_size, hidden=hidden, intermediate=intermediate,
                                              layers=layers))
    m = BitNetForCausalLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 1 and "norm" in n:
                p.add_(0.1 * torch.randn_like(p))
    return m.eval()


def build_tiny_moe(seed: int = 0, layers: int = 2, noise: float = 0.3, num_experts: int = 4, top_k: int = 2,
                   vocab_size: int = 128256, hidden: int = 256, intermediate: int = 512,
                   router_std: float = 0.1) -> BitNetMoEForCausalLM:
    dense = build_tiny_dense(seed, layers, vocab_size, hidden, intermediate)
    return upcycle(dense, num_experts, top_k, expert_noise=noise, router_std=router_std, seed=seed + 1).eval()


@torch.no_grad()
def reference(model: BitNetMoEForCausalLM, ids: torch.Tensor, top: int = 20) -> dict:
    out = model(ids)
    logits = out.logits[0, -1].float()
    v, i = torch.topk(logits, top)
    routing = []
    for r in out.router_logits:
        w, idx = route(r, model.config.num_experts_per_tok)
        p = torch.sort(torch.softmax(r.float(), -1), -1, descending=True).values
        k = model.config.num_experts_per_tok
        margin = (p[:, k - 1] - p[:, k]) if k < p.shape[-1] else torch.ones(p.shape[0])
        routing.append({"experts": idx.tolist(), "weights": w.tolist(), "margin": margin.tolist()})
    # how much does routing matter? force every token to experts (0, 1) with weights 1/k and compare
    forced = copy.deepcopy(model)
    for m in forced.moe_layers:
        m.router.weight.zero_()
    lf = forced(ids).logits[0, -1].float()
    return {"logits": logits, "top_ids": i.tolist(), "top_values": v.tolist(), "routing": routing,
            "max_abs_logit_change_vs_uniform_router": float((lf - logits).abs().max())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--noise", type=float, default=0.3, help="relative expert noise (x std(W))")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--num-experts", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    tok = build_llama3_tokenizer()
    model = build_tiny_moe(args.seed, args.layers, args.noise, args.num_experts, args.top_k, vocab_size=len(tok))
    model.save_pretrained(out, tokenizer=tok, dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32)

    # references from the checkpoint as stored on disk
    qat = BitNetMoEForCausalLM.from_pretrained(out, dtype=torch.float32, qat=True).eval()
    flt = BitNetMoEForCausalLM.from_pretrained(out, dtype=torch.float32, qat=False).eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids  # BOS + text
    rq, rf = reference(qat, ids), reference(flt, ids)
    np.save(out / "reference_logits.npy", rq.pop("logits").numpy().astype(np.float32))
    np.save(out / "reference_logits_float.npy", rf.pop("logits").numpy().astype(np.float32))
    ref = {
        "prompt": args.prompt,
        "token_ids": ids[0].tolist(),
        "add_bos": True,
        "position": "last",
        "num_experts": model.config.num_experts,
        "num_experts_per_tok": model.config.num_experts_per_tok,
        "expert_noise": args.noise,
        "saved_dtype": args.dtype,
        "notes": ("ternary = BitLinear QAT forward (absmean ternary weights per tensor, per-token absmax int8 "
                  "activations) == what I2_S should reproduce; float = same latent weights, no quantization. "
                  "Computed in fp32 from the saved checkpoint. routing[layer].experts/weights are per token "
                  "(top-k order, weights renormalized); margin = p_k - p_(k+1) of the full softmax (small margins "
                  "may legitimately flip under quantization noise)."),
        "ternary": {**rq, "logits_file": "reference_logits.npy"},
        "float": {**rf, "logits_file": "reference_logits_float.npy"},
    }
    with open(out / "reference.json", "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=1)
    c = model.config
    print(f"saved tiny BitNet-MoE to {out}: L={c.num_hidden_layers} d={c.hidden_size} n_ff={c.intermediate_size} "
          f"E={c.num_experts} top-{c.num_experts_per_tok} vocab={c.vocab_size} noise={args.noise}; "
          f"{len(ref['token_ids'])} prompt tokens; ternary top1={rq['top_ids'][0]} float top1={rf['top_ids'][0]}; "
          f"routing effect on logits: ternary {rq['max_abs_logit_change_vs_uniform_router']:.3g}, "
          f"float {rf['max_abs_logit_change_vs_uniform_router']:.3g}")


if __name__ == "__main__":
    main()
