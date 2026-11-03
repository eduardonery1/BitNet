# BitNet-MoE: CPU inference with bitnet.cpp

This doc covers running a BitNet-MoE checkpoint (see `docs/bitnet-moe-design.md`) with bitnet.cpp: the new
`bitnet-moe` architecture in the llama.cpp fork, the converter, the 3D I2_S expert layout, and parity and speed
results. All results below use random or untrained weights.

## Build

Same as the dense model (see `docs/bitnet-vl-inference.md`). The model sources are globbed, so re-run the configure
step once after pulling, because `src/models/bitnet-moe.cpp` is a new file:

```bash
cmake -B build -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_EXAMPLES=ON   # (gcc 15.2 verified)
cmake --build build -j12 --target llama-cli llama-completion llama-bench llama-mtmd-cli
```

## Convert and run

```bash
pip install sentencepiece          # needed by utils/convert.py, which the dense converter imports
python utils/convert-bitnet-moe-to-gguf.py CKPT --outtype i2_s --outfile moe-i2_s.gguf        # deployable
python utils/convert-bitnet-moe-to-gguf.py CKPT --outtype f32  --outfile moe-f32.gguf         # parity / debugging
python utils/convert-bitnet-moe-to-gguf.py CKPT --outtype i2_s --expert-type f16 ...          # I2_S attention, F16 experts
python utils/convert-bitnet-moe-to-gguf.py CKPT --outtype f32 --no-ternarize ...              # latent float weights

build/bin/llama-completion -m moe-i2_s.gguf -p "The quick brown fox" -n 64 --temp 0 -no-cnv -t 12
```
- The converter requires `architectures: ["BitNetMoEForCausalLM"]` and `num_experts` in `config.json`, plus
  `norm_topk_prob: true` (the only gating the graph implements). It fails if any checkpoint tensor goes unconverted.
- `--expert-type {f32,f16,q8_0,i2_s}` sets the expert storage independently of `--outtype`.
- `--embd-type {f32,f16}` sets the token-embedding type. The embedding is also the tied LM head. The default is f32
  for `--outtype f32` and f16 otherwise, the same as the dense converter.
- `general.file_type` is written as 41 (`MOSTLY_I2_S`). The dense converter writes 40, which is a known bug.
- The converter reuses `BitnetModel` (`weight_quant`, vocab/tokenizer writing) and `quantize_to_i2_s` from
  `utils/convert-hf-to-gguf-bitnet.py`. It imports them via importlib and copies nothing.
- Do not use `llama-quantize` to produce I2_S. Its I2_S packing is wrong, see `docs/bitnet-vl-inference.md`.

## GGUF contents (arch `bitnet-moe`)

Hparams are the same as `bitnet-b1.58`, plus `bitnet-moe.expert_count` and `bitnet-moe.expert_used_count`.

| GGUF tensor | checkpoint source | ggml ne | type `--outtype f32` / `f16` / `i2_s` |
|---|---|---|---|
| `token_embd.weight` (also LM head) | `model.embed_tokens` | [d, V] | F32 / F16 / F16 (`--embd-type`) |
| `output.weight` (only if untied) | `lm_head` | [d, V] | as embedding |
| `output_norm`, `blk.N.{attn_norm, attn_sub_norm, ffn_norm}` | RMSNorm weights | [d] | F32 |
| `blk.N.ffn_sub_norm.weight` | `mlp.ffn_sub_norm` (one per layer) | [n_ff] | F32 |
| `blk.N.attn_{q,k,v,output}.weight` | `self_attn.{q,k,v,o}_proj` | 2D | F32 / F16 (ternarized values) / I2_S |
| `blk.N.ffn_gate_inp.weight` | `mlp.router` | [d, E] | F32 always |
| `blk.N.ffn_{gate,up}_exps.weight` | `mlp.experts.{e}.{gate,up}_proj` stacked | [d, n_ff, E] | F32 / F16 codes / I2_S |
| `blk.N.ffn_down_exps.weight` | `mlp.experts.{e}.down_proj` stacked | [n_ff, d, E] | F32 / F16 codes / I2_S |
| `blk.N.ffn_{gate,up,down}_exps.scale` | per-expert absmean scale | [E] | F32, float types only |

For float expert storage, each expert is ternarized with absmean (the same op order as `weight_quant`). The tensor
holds the exact codes {-1, 0, 1}, and the per-expert scale goes into `*_exps.scale`. The graph applies that scale
after `mul_mat_id` (`build_lora_mm_id`'s `w_s`), so F16 experts are exact. With `--no-ternarize`, the latent weights
are stored and no scale tensor is written. Attention weights in f32/f16 are stored as dequantized ternary values,
the same as the dense converter.

## Graph (`3rdparty/llama.cpp/src/models/bitnet-moe.cpp`)

The attention part is unchanged from `bitnet-b1.58`, including the optional per-tensor `*_s` scales and the
`attn_sub_norm`. The FFN is built explicitly. `build_moe_ffn` cannot be used, because it has no hook for a norm
between the activation and `down`. Per token, with x = `ffn_norm(h)`:

```
p   = softmax(W_router x)                        (all E)        ggml_soft_max
S   = top-k(p); w_e = p_e / sum_{j in S} p_j                    ggml_argsort_top_k, get_rows, sum_rows, div
h_e = relu(gate_e x)^2 * (up_e x)                [n_ff, k, T]   mul_mat_id x2, relu, sqr, mul
h_e = rms_norm(h_e, eps = rms_norm_eps) * ffn_sub_norm          one shared weight, per (expert, token) row
out = sum_{e in S} w_e * down_e h_e                             mul_mat_id, mul, views + add
```
The LM head is `output.weight` when present, otherwise the tied `token_embd`.

## 3D I2_S layout and `mul_mat_id`

A 2D I2_S tensor is `ne0*ne1/4` packed bytes (128-element blocks; byte j of a block holds elements j, 32+j, 64+j and
96+j; codes 0/1/2 = -1/0/+1) followed by a 32-byte tail whose first 4 bytes are the f32 scale. A 3D expert tensor
`[n_in, n_out, E]` is E such blobs back to back, and expert e starts at `e * (n_in*n_out/4 + 32)`. Changes:

- `ggml_nbytes` (`ggml/src/ggml.c`, I2_S branch) now returns `(ne0*ne1/4 + 32) * ne2 * ne3`. This is unchanged for
  2D. GGUF offsets, the loader size checks and buffer allocation all go through `ggml_nbytes`, so nothing else in
  the loader needed changing. `nb[2]` keeps the logical int8 stride and is not used for I2_S addressing.
- The CPU `mul_mat_id` dispatches I2_S to `ggml_compute_forward_mul_mat_id_i2s` (`ggml/src/ggml-cpu/ggml-cpu.c`):
  1. It groups the (token, slot) pairs by expert.
  2. It gathers each expert's src1 rows into one contiguous block and quantizes them to int8 with
     `quantize_row_i8_s` (per-row absmax).
  3. For every expert that has tokens, it runs `ggml_gemm_i2_i8_s` on 64-row x 16-token tiles. The expert's scale
     comes from that expert's own tail, and the output is `(dot - sum(q)) * w_scale / act_scale`.
  Threads split the output rows statically, so each selected expert is streamed once per call. An all-zero
  activation row gives 0, not the NaN that the dense path produces. The kernel asserts K % 128 == 0, and the
  converter enforces the same.
- The work-size planner reserves the gathered int8 rows, the scales and sums, and the row maps.
- `gguf-py`: adds `MODEL_ARCH.BITNET_MOE` with its tensor list, and `GGUFReader` now sizes 3D I2_S tensors per matrix.

## Parity (x86_64 AVX2, last-position or all-position logits, rel = ||a-b|| / ||b||)

| GGUF | reference | rel error |
|---|---|---|
| f32, ternarized | ternary weights + float activations (own ckpt, all 16 positions) | 6.1e-4 default ctx; **9e-7** with F32 KV cache + flash-attn off |
| f32, `--no-ternarize` | float forward (own ckpt) | 1.3e-3 default; **1.0e-6** with F32 KV + no FA |
| f16, ternarized (F16 codes + scale) | ternary weights + float acts | 1.2e-3 |
| q8_0 experts | ternary weights + float acts | 1.3e-2 |
| **i2_s**, `--embd-type f32` | **ternary QAT (W1.58A8)**, own ckpt, all positions | **7.6e-4** (last token 3.8e-7) |
| **i2_s**, `--embd-type f32` | ternary QAT, medium ckpt d=1024, n_ff=2048, 48 tokens (multi-tile path) | **1.05e-6** |
| i2_s, default F16 embedding | ternary QAT, own ckpt | 2.5e-2 (dominated by the F16 tied LM head) |
| f32, ternarized | PyTorch agent's `tiny-moe` `reference_logits.npy` (ternary QAT) | 2.5e-2 (expected: no int8 activations) |
| f32, ternarized | the same checkpoint, ternary weights + float acts, recomputed with my reference | 6.9e-7 (F32 KV, no FA) |
| f32 `--no-ternarize` | `tiny-moe` `reference_logits_float.npy` | 8.9e-7 (F32 KV, no FA) |
| **i2_s**, `--embd-type f32` | `tiny-moe` `reference_logits.npy` (ternary QAT) | 1.7e-2, top-1 matches |

A standalone torch reference I wrote for this (not committed; it follows the checkpoint contract) agrees with the PyTorch agent's references to 3e-7 on their
checkpoint. On that checkpoint, the I2_S residual of 1.7e-2 is at the reference's own noise floor. Switching only the
reference's int8 rounding from half-to-even (torch) to half-away-from-zero (`roundf`, what C does) already moves its
logits by 1.6e-2. That checkpoint has bf16 weights and router margins as small as 0.006. On the better-conditioned
fp32 checkpoints, I2_S matches the QAT reference to about 1e-6, and results are bit-identical for 1, 3, 4, 5 and 12
threads. The sequential decode path (`n_tokens = 1`) matches batch prefill.

## Speed (random weights, real shapes, 4 layers)

d=2560, n_ff=6912, 20 heads / 5 KV heads, E=8, top-2, vocab 1024. I shrank the vocab so the LM head does not dominate
a 4-layer model. `llama-bench -p 0 -n 128 -r 5`, x86_64, gcc 15.2. The machine was shared with a training job, so
the 12-thread numbers are noisy. These are **random-weight speed measurements**, not quality measurements.

| model (4 layers) | file | decode t/s, 12 thr | decode t/s, 6 thr | pp128 t/s, 12 thr |
|---|---|---|---|---|
| dense bitnet-b1.58, I2_S | 71 MiB | 269 ± 18 | 321 ± 4 | 1348 ± 497 |
| MoE I2_S (attn + experts) | 426 MiB | 174 ± 14 | 209 ± 3 | 1153 ± 189 |
| MoE, I2_S attn + Q8_0 experts | 1.70 GiB | 61 ± 3 | 73 ± 1 | 326 ± 19 |
| MoE, I2_S attn + F16 experts | 3.18 GiB | 24 ± 15 | 42 ± 0.1 | 213 ± 9 |

The MoE stores 8x the FFN weights but reads 2 experts per token: about 30 MB per layer, versus about 17 MB for the
dense model. Its decode speed is 0.65x dense, close to the bytes ratio (0.57x). F16 experts are 7x slower than
I2_S. In llama-bench output every file shows as "I2_S", because `general.file_type` = 41.

## Limitations / open issues

- **Fixed in this branch (dense):** `bitnet-b1.58` previously ran the FFN with **SiLU** (inherited from the upstream
  `bitnet` graph). HF BitNet / 2B-4T uses `relu2`. On a tiny dense checkpoint, the old graph was off by 49% rel vs
  PyTorch, and with the fix it is off by 8e-7 (`src/models/bitnet.cpp`, now `RELU_SQR` for `LLM_ARCH_BITNET_B158`
  only). This changes the outputs of every `bitnet-b1.58` GGUF, including the BitNet-VL LLM, so they now follow the
  HF model.
- Only the CPU backend implements 3D I2_S `mul_mat_id`. The AVX2 path is verified. The NEON path runs through the
  generic `ggml_gemm_i2_i8_s` fallback and is untested. `QK_I2_S = 64` on some ARM builds may conflict with
  the 128-block packing; this is the same open question as for the dense model.
- 3D I2_S tensors cannot go through `llama-quantize` or the non-`_id` `mul_mat`. TL1/TL2 MoE is not supported.
- Prefill uses the generic I2_S GEMM (no tinyBLAS/VNNI path for experts), so pp is slower per FLOP than dense.
- The defaults (F16 KV cache, auto flash attention, F16 tied embedding) add about 1e-3 to 2.5e-2 relative logit
  error on these random checkpoints. Use `--embd-type f32` if exact parity matters.
- No real-checkpoint quality numbers exist yet. The full 30-layer E=8 model was not built, because RAM is about
  16 GB. Its speed is extrapolated only by bytes.
