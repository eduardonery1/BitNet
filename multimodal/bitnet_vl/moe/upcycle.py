"""Sparse upcycling: dense BitNet b1.58 -> BitNet-MoE (docs/bitnet-moe-design.md).

    cd multimodal && python -m bitnet_vl.moe.upcycle DENSE_DIR OUT_DIR --num-experts 8 --top-k 2 [--expert-noise 0.0]

Every expert e gets a copy of the dense ``mlp.{gate,up,down}_proj`` latent weights; ``mlp.ffn_sub_norm`` is copied
once (shared). The router is random normal with std ``--router-std``. With ``--expert-noise 0`` the MoE model computes
exactly the dense model for ANY router (renormalized top-k weights sum to 1 over identical experts).

``--expert-noise s`` adds ``s * std(W) * N(0, 1)`` independently to each expert's copy of each matrix W (relative
noise; breaks expert symmetry). DENSE_DIR is an HF BitNet dir with latent (bf16/fp32) weights, e.g.
``microsoft/bitnet-b1.58-2B-4T-bf16`` or a BitNet-VL ``language_model/`` dir.
"""

from __future__ import annotations

import argparse

import torch
from transformers import BitNetForCausalLM

from .modeling_moe import BitNetMoEForCausalLM, load_dense, moe_config_from_dense


@torch.no_grad()
def upcycle(dense: BitNetForCausalLM, num_experts: int = 8, top_k: int = 2, expert_noise: float = 0.0,
            router_std: float = 0.02, seed: int = 0, qat: bool = True, **moe_kwargs) -> BitNetMoEForCausalLM:
    cfg = moe_config_from_dense(dense.config, num_experts=num_experts, num_experts_per_tok=top_k, **moe_kwargs)
    dtype = dense.model.embed_tokens.weight.dtype
    moe = BitNetMoEForCausalLM(cfg, qat=qat).to(dtype)
    dsd = dense.state_dict()
    msd = moe.state_dict()
    g = torch.Generator().manual_seed(seed)
    new = {}
    for k in msd:
        if ".mlp." not in k:
            new[k] = dsd[k]
    for n in range(cfg.num_hidden_layers):
        p = f"model.layers.{n}.mlp"
        new[f"{p}.ffn_sub_norm.weight"] = dsd[f"{p}.ffn_sub_norm.weight"]
        r = torch.randn(msd[f"{p}.router.weight"].shape, generator=g) * router_std
        new[f"{p}.router.weight"] = r.to(dtype)
        for e in range(num_experts):
            for x in ("gate", "up", "down"):
                w = dsd[f"{p}.{x}_proj.weight"].float()
                if expert_noise > 0:
                    w = w + expert_noise * w.std() * torch.randn(w.shape, generator=g)
                new[f"{p}.experts.{e}.{x}_proj.weight"] = w.to(dtype)
    missing = set(msd) - set(new)
    assert not missing, missing
    moe.load_state_dict(new, strict=True)
    if moe.config.tie_word_embeddings:
        moe.tie_weights()
    return moe


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dense_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--expert-noise", type=float, default=0.0, help="relative to std(W) per matrix")
    ap.add_argument("--router-std", type=float, default=0.02)
    ap.add_argument("--aux-loss-coef", type=float, default=0.01)
    ap.add_argument("--z-loss-coef", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "f32"], default="bf16", help="saved dtype")
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    dense = load_dense(args.dense_dir, dtype=torch.float32, qat=False)
    moe = upcycle(dense, args.num_experts, args.top_k, args.expert_noise, args.router_std, args.seed,
                  router_aux_loss_coef=args.aux_loss_coef, router_z_loss_coef=args.z_loss_coef)
    try:
        tok = AutoTokenizer.from_pretrained(args.dense_dir)
    except Exception as e:  # noqa: BLE001
        print(f"warning: no tokenizer saved ({e})")
        tok = None
    moe.save_pretrained(args.out_dir, tokenizer=tok, dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32)
    n = sum(p.numel() for p in moe.parameters())
    print(f"upcycled {args.dense_dir} -> {args.out_dir}: E={args.num_experts} top-{args.top_k}, "
          f"noise={args.expert_noise}, {n / 1e6:.1f}M params")


if __name__ == "__main__":
    main()
