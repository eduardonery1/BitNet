# BitNet-MoE: ternary mixture-of-experts BitNet

Status: design + reference implementation (untrained). Date: 2026-11-02.

## Goal

Increase BitNet capacity without increasing per-token compute or bandwidth. Ternary experts are cheap to store at
~2 bits/weight in I2_S, and CPU decode is bandwidth-bound, so only the routed experts' weights are read per token.

## Architecture decisions

| Item | Choice | Why |
|---|---|---|
| Base | `microsoft/bitnet-b1.58-2B-4T-bf16`, sparse upcycling (Komatsuzaki et al. 2022) | Reuses 4T tokens of pretraining. Every expert starts as a copy of the dense FFN, so the upcycled model initially equals the dense model. |
| Experts | E = 8 per layer, top-k = 2 (configurable), every layer | Mixtral-style default. |
| Expert FFN | Identical to BitNet's: `down(sub_norm(relu²(gate(x)) * up(x)))`, BitLinear (W1.58A8) on gate/up/down | Keeps the BitNet block structure. |
| `ffn_sub_norm` | **One per layer, shared by all experts** (RMSNorm weight `[n_ff]`) | Faithful to upcycling (all copies share it) and simple to implement in the ggml graph. |
| Router | `Linear(d_model → E)`, no bias, **full precision** (F32/F16 in GGUF) | Routing is sensitive and tiny. |
| Gating | softmax over all E logits, take top-k, renormalize the k weights to sum to 1 (= softmax over top-k logits) | Mixtral convention. With identical experts the output equals the dense FFN regardless of routing. Maps to llama.cpp softmax gating with `norm_w = true`. |
| Symmetry breaking | Optional small noise on upcycled expert latent weights (`--expert-noise`), plus random router init | Otherwise identical experts get identical gradients. |
| Losses | LM loss + 0.01 · Switch load-balancing loss + 0.001 · router z-loss | Standard. |
| Dims | n_ff must stay a multiple of 128 (I2_S K constraint for `down`) | Rules out fine-grained splits such as 6912/4 = 1728. |

Size at E = 8, top-2 (estimate): FFN params ≈ 8 × 1.59B ≈ 12.7B, total ≈ 13.5B. I2_S on disk ≈ 3.5 GB. About 4B params active per token.

## Checkpoint contract (HF-style dir)

- `config.json`: BitNet config (`model_type: "bitnet"`, `architectures: ["BitNetMoEForCausalLM"]`, top-level `rope_theta`) plus `num_experts`, `num_experts_per_tok`, `router_aux_loss_coef`, `router_z_loss_coef`, `norm_topk_prob: true`.
- Per layer N, replacing the dense `model.layers.N.mlp.{gate,up,down}_proj`:
  - `model.layers.N.mlp.router.weight` `[E, d]`
  - `model.layers.N.mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight` (latent bf16/fp32; ternarized at export)
  - `model.layers.N.mlp.ffn_sub_norm.weight` `[n_ff]`
- All other tensors keep the dense BitNet names. Tokenizer files are saved alongside.

## GGUF contract

- arch `bitnet-moe` (new; keeps `bitnet-b1.58` untouched). Same hparams as `bitnet-b1.58` plus `bitnet-moe.expert_count`, `bitnet-moe.expert_used_count`.
- Tensors: `blk.N.ffn_gate_inp` (router, F32), `blk.N.ffn_{gate,up,down}_exps` (3D `[n_in, n_out, E]`, I2_S or F16/Q8_0), `blk.N.ffn_sub_norm`, and the rest as in `bitnet-b1.58`.
- I2_S for 3D expert tensors requires a per-expert scale. Proposed layout: each expert slice is a standard 2D I2_S blob (packed data + 32-byte scale tail), with slices laid out contiguously. This needs `ggml_nbytes`/row-size handling and `mul_mat_id` support (see docs/bitnet-moe-inference.md).

## Training

Continued pretraining after upcycling, on FineWeb-Edu/DCLM-style text. Upcycling literature uses roughly 5–20% of the original compute budget. At this scale, assume tens of billions of tokens for a meaningful gain (an estimate). LR ≈ 1e-4–3e-4 with cosine decay, and WD 0.1→0, matching BitNet's stage-2 behaviour. Router LR equals or exceeds expert LR.

Evaluation: perplexity vs the dense 2B4T, the 2B4T paper's benchmark suite, CPU decode tok/s and memory in bitnet.cpp, and expert-load histograms.

## Reference implementation

PyTorch code in `multimodal/bitnet_vl/moe/` (run from `multimodal/`):

```bash
python -m bitnet_vl.moe.upcycle DENSE_DIR OUT_DIR --num-experts 8 --top-k 2 [--expert-noise 0.0] [--router-std 0.02]
python -m bitnet_vl.moe.train_moe --init-from OUT_DIR --data corpus/ --output-dir ckpt/moe \
    --seq-len 2048 --batch-size 8 --grad-accum 4 --max-steps 10000 --lr 2e-4 --router-lr 4e-4 [--grad-checkpointing]
python -m bitnet_vl.moe.make_tiny_moe_ckpt OUT_DIR [--noise 0.3]   # tiny E=4 top-2 ckpt + reference logits
python -m pytest bitnet_vl/moe/tests -q
```

- `modeling_moe.py`: `BitNetMoEForCausalLM` subclasses HF `BitNetForCausalLM` and swaps each `mlp` for `BitNetMoEMLP`.
  Expert FFN = HF `BitNetMLP` exactly: `down(ffn_sub_norm(relu(gate(x))**2 * up(x)))`, where `ffn_sub_norm` is
  `BitNetRMSNorm(n_ff, eps=rms_norm_eps)` (`w * h * rsqrt(mean(h**2) + eps)` in fp32; eps 1e-5 for 2B4T), shared by the
  experts of a layer. gate/up/down and q/k/v/o are `BitLinear` (absmean ternary weights, per-token absmax int8
  activations, STE). The router is an FP `nn.Linear` on the post-attention-norm output. Gating is
  `softmax(fp32) → top-k → renormalize`. Dispatch sorts token-expert assignments by expert and uses `index_add_`.
- `upcycle.py`: copies the dense FFN into every expert. `--expert-noise s` adds `s · std(W) · N(0,1)` per expert
  copy (relative). With s = 0 the logits equal the dense model's for any router (tested at atol 1e-5).
- `losses.py`: Switch load-balancing loss `E · Σ_e f_e P_e`, where `f_e` = share of the T·k top-k slots. This gives
  1.0 for uniform routing; HF Mixtral's version gives k. Router z-loss is `mean(logsumexp(logits)²)`. The model
  averages both over layers.
- `make_tiny_moe_ckpt.py` writes `reference.json` (prompt ids, top-20 last-position logits and per-layer
  per-token routing for both the ternary QAT and the float forward), `reference_logits.npy` (ternary, fp32
  `[vocab]`, the I2_S target) and `reference_logits_float.npy`.
