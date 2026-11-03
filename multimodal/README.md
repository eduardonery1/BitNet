# BitNet-VL

## Training

The PyTorch reference implementation and training code live in `multimodal/bitnet_vl/` (run everything below from
`multimodal/`). Architecture and recipe: `docs/bitnet-vl-design.md`.

| File | Contents |
|---|---|
| `bitlinear.py` | `BitLinear` (absmean ternary W, per-token absmax int8 A, STE), `replace_linears`, `ternary_weight` export helper |
| `modeling.py` | `BitNetVLConfig`, `BitNetVLForConditionalGeneration` (SigLIP → 2×2 pixel shuffle → FP projector → BitNet LLM), save/load |
| `data.py` | LLaVA-format dataset, SigLIP preprocessing, prompt template, answer-only label masking, collator |
| `train.py` | `--stage {1,2,3}` training loop (cosine LR + warmup, WD schedule, grad accumulation, bf16 autocast, DDP, stage-3 distillation) |
| `make_tiny_ckpt.py` | tiny random checkpoint in the contract layout, for exporter / end-to-end tests |
| `tokenizer_utils.py` | offline Llama-3 tokenizer rebuilt from `3rdparty/llama.cpp/models/ggml-vocab-llama-bpe.gguf` |

Requirements: `torch`, `transformers>=5.0` (native `BitNetForCausalLM` and `SiglipVisionModel`), `safetensors`, `pillow`,
optionally `pytest`. Converting `language_model/` with `utils/convert-hf-to-gguf-bitnet.py` also needs `sentencepiece` installed (the converter imports it).

### Data

The data uses the LLaVA JSON format (`.json` list or `.jsonl`), with image paths relative to `--image-folder`:
```json
{"image": "coco/train2017/000000033471.jpg",
 "conversations": [{"from": "human", "value": "<image>\nWhat is the man doing?"}, {"from": "gpt", "value": "Riding a bike."}]}
```
- Stage 1: `liuhaotian/LLaVA-Pretrain` (`blip_laion_cc_sbu_558k.json` + `images/`).
- Stage 2: `liuhaotian/LLaVA-Instruct-150K` `llava_v1_5_mix665k.json` with the COCO, GQA, OCR-VQA, TextVQA and VG images laid out as in LLaVA-1.5. You can append The Cauldron converted to the same format.
- Stage 3: a subset of the stage-2 mix (≥ 2–5B tokens).

Each sample is rendered as `<|begin_of_text|>User: <image>\n{q}<|eot_id|>Assistant: {a}<|eot_id|>...` (the `<image>`
token is moved to the front of the first question). Loss is computed only on the answer tokens and their `<|eot_id|>`.

### Commands

The global batch is `--batch-size × --grad-accum × #GPUs`, and the recipe uses 256. For example, 8 GPUs × 8 × 4:
```bash
# Stage 1 (align): train the projector. Encoder bf16 frozen, LLM frozen (BitLinear QAT forward). LR 1e-3, no WD.
torchrun --nproc_per_node 8 -m bitnet_vl.train --stage 1 \
  --vision-model google/siglip-base-patch16-384 --llm-model microsoft/bitnet-b1.58-2B-4T-bf16 \
  --data LLaVA-Pretrain/blip_laion_cc_sbu_558k.json --image-folder LLaVA-Pretrain/images \
  --batch-size 8 --grad-accum 4 --epochs 1 --max-length 2048 --grad-checkpointing --output-dir ckpt/stage1

# Stage 2 (instruct): train projector (1e-3) + LLM QAT (3e-4). WD 0.1 -> 0, cosine schedule.
torchrun --nproc_per_node 8 -m bitnet_vl.train --stage 2 --init-from ckpt/stage1 \
  --data llava_v1_5_mix665k.json --image-folder data \
  --batch-size 4 --grad-accum 8 --epochs 1 --grad-checkpointing --output-dir ckpt/stage2

# Stage 3 (ternarize ViT): swap SigLIP q/k/v/out/fc1/fc2 to BitLinear and train only the encoder (1e-4, WD 0.01).
# Loss = LM + 0.1 * mean_l MSE(teacher_l, student_l). The teacher is a frozen bf16 copy of the stage-2 encoder.
torchrun --nproc_per_node 8 -m bitnet_vl.train --stage 3 --init-from ckpt/stage2 \
  --data stage3_subset.json --image-folder data \
  --batch-size 4 --grad-accum 8 --epochs 1 --grad-checkpointing --output-dir ckpt/stage3
# To resume stage 3 from an already-ternary checkpoint, add: --teacher-from ckpt/stage2
```
Useful flags: `--lr-projector/--lr-llm/--lr-vision`, `--weight-decay/--weight-decay-end`, `--warmup-ratio` (0.03),
`--max-steps`, `--save-every N`, `--distill-weight` (0.1), `--save-dtype {bf16,f32}` (dtype of `language_model/`).
Run without `torchrun` for a single process. On CUDA the forward runs under bf16 autocast with fp32 master weights.

Quick interactive check:
```python
from bitnet_vl import BitNetVLForConditionalGeneration
m = BitNetVLForConditionalGeneration.from_pretrained("ckpt/stage3").eval()
print(m.answer("bus.jpg", "What color is the bus?"))
```

### Checkpoint layout

`save_pretrained` writes the layout from the interface contract: `config.json`, `vision_tower.safetensors` (SigLIP
`vision_model.*` keys, latent weights, no attention-pooling head), `projector.safetensors`
(`norm|fc1|fc2 .weight|.bias`) and `language_model/` (HF BitNet + tokenizer with `<image>` added, tied embeddings,
top-level `rope_theta` for the GGUF converter). `config.json` also carries `vision_config` (full SigLIP config),
`image_token_id`, `image_mean/std` and the projector / feature conventions.

Feature conventions the exporter / C++ graph must match:
- Vision features are `last_hidden_state`: the last encoder layer followed by `post_layernorm`.
- Pixel shuffle (s = 2, Idefics3/SmolVLM ordering) on the W×W patch grid:
  `y[i*(W/2)+j, (dy*2+dx)*D + c] = x[(2i+dy)*W + (2j+dx), c]`. See `pixel_shuffle_reference` in `modeling.py`.
- Projector: `LayerNorm(eps 1e-5) → Linear → GELU (exact erf) → Linear`, all with bias.

### Tests and tiny checkpoint

```bash
python -m pytest bitnet_vl/tests -q                 # CPU, ~25 s
python -m bitnet_vl.make_tiny_ckpt /tmp/bitnet-vl-tiny [--vision-ternary]
python ../utils/convert-hf-to-gguf-bitnet.py /tmp/bitnet-vl-tiny/language_model --outtype i2_s
```
The tiny checkpoint has a 64 px image (4 visual tokens), SigLIP d = 128 / mlp 512 with 2 layers, and a BitNet LLM with
d = 256 / ffn 512, 2 layers and the real Llama-3 vocabulary (128257 tokens with `<image>`). All ternary K dimensions
are multiples of 128, so they can be stored as I2_S.

## Inference (bitnet.cpp)

CPU inference goes through `llama-mtmd-cli`. The mmproj uses the `bitnet_vl` projector type in
`3rdparty/llama.cpp/tools/mtmd`. Details, the tensor table and verification results are in
[`docs/bitnet-vl-inference.md`](../docs/bitnet-vl-inference.md).

```bash
cmake -B build -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j12 --target llama-mtmd-cli
export PYTHONPATH=3rdparty/llama.cpp/gguf-py
python utils/convert-hf-to-gguf-bitnet.py ckpt/language_model --outtype i2_s --outfile ggml-model-i2_s.gguf
python multimodal/convert_bitnet_vl_mmproj.py ckpt/ --outfile mmproj-bitnet-vl.gguf [--vision-i2s]
build/bin/llama-mtmd-cli -m ggml-model-i2_s.gguf --mmproj mmproj-bitnet-vl.gguf --image img.jpg \
    -p $'<__media__>\nWhat color is the bus?' --temp 0
```
Use `--vision-i2s` only for a checkpoint trained with a ternary encoder (`vision_ternary: true`, stage 3).
Do not use `llama-quantize` to make I2_S files, because its I2_S packer is broken (see the inference doc).

## Latent reasoning

`multimodal/bitnet_vl/latent/` is a separate experiment that reuses the BitNet QAT pieces above (`BitLinear`, the
HF BitNet loading with `quantization_config` stripped, the offline Llama-3 tokenizer). It tests Coconut-style latent
reasoning with a compressor. The LLM runs over the prompt, the hidden states at layer *i* (default round(2L/3)) are
compressed into K soft tokens, and those tokens are appended as input embeddings with the KV cache reused, so only the
K new positions are computed. This repeats R times before the answer is decoded. The question is whether feeding back
compressed layer-*i* content beats the same extra compute without feedback (learned pause tokens).

| File | Contents |
|---|---|
| `tasks.py` | synthetic variable-tracking programs (`hops` chain length, interleaved distractors, values in [0, 99], trace lines `b = 5`), deterministic per seed, test seeds disjoint from training |
| `compressor.py` | `LatentCompressor` (K learned queries + step embedding, pre-LN cross-attention + MLP over all positions seen so far, Linear → RMS rescaled to the input-embedding RMS) and `PauseTokens`, both full precision |
| `lora.py` | minimal LoRA over `nn.Linear`/`BitLinear` (frozen base, FP LoRA path added to the W1.58A8 output; optional one-time ternarization of frozen weights), save/load of LoRA + compressor only |
| `reasoner.py` | `LatentReasoner` with modes `none` / `latent` / `pause` / `cot`, left-padded batches with explicit position ids, KV-cache reuse, Coconut curriculum, BPTT truncation, `greedy_answer(prompt, R)` |
| `train_latent.py` | CLI and `run(config) -> results` (results.json: accuracy per (eval R, hops), loss curve, throughput); `--ckpt-every N` writes a resumable `ckpt.pt` (trainable weights, optimizer, scaler, data/RNG state), and rerunning the same command continues the same trajectory |
| `run_suite.py` | runs the four modes as a resumable suite: finished modes (with `results.json`) are skipped, the unfinished one resumes from its checkpoint |

Layout: `<BOS>program\n` + R×K thought positions + remaining trace lines + `Answer: <n><|eot_id|>`. The loss covers the
target tokens only, and `Answer:` is in the loss only when explicit trace lines come before it. The curriculum uses
R+1 stages: stage *s* replaces the first *s* trace lines with *s* thought steps, and the last stage is all latent.

Implementation notes:
- transformers 5 BitNet derives `position_ids` from `cache_position` unless they are passed, which is wrong for left
  padding. The reasoner always passes position ids that count the real tokens, plus a 2-D mask over past and new
  positions.
- HF gradient checkpointing drops the KV cache, so `--grad-checkpointing` uses its own per-layer checkpointing
  together with `LatentCache`. Its layers update idempotently (`cache[:start] ++ new`), so recomputation during
  backward does not duplicate cache entries. Tests check that gradients match the non-checkpointed run.
- Per-token int8 activation rounding is discontinuous, so the exact cached-vs-full-forward and padded-vs-unbatched
  tests run with activation quantization off at atol 1e-4. With it on, the tolerance is loose.

```bash
cd multimodal
python -m pytest bitnet_vl/tests/test_latent.py -q        # CPU, ~25 s
# tiny random BitNet, all weights trained (sanity check only)
python -m bitnet_vl.latent.train_latent --model tiny --train-base --lora-rank 0 --mode latent --R 2 --K 2 \
    --hops-train 2-3 --hops-test 2-4 --distractors 1 --steps 1200 --batch-size 32 --out runs/tiny_latent
# 2B4T, frozen base + LoRA + compressor (GPU)
python -m bitnet_vl.latent.train_latent --model microsoft/bitnet-b1.58-2B-4T-bf16 --dtype bf16 --mode latent \
    --curriculum --steps-per-stage 50 --R 4 --K 4 --hops-train 2-4 --hops-test 2-6 --steps 500 \
    --batch-size 4 --grad-accum 4 --grad-checkpointing --eval-R-sweep 0,1,2,4,6,8 --out runs/latent
# all four modes, QUICK preset, resumable (put --run-dir on persistent storage)
python -m bitnet_vl.latent.run_suite --run-dir runs/quick --preset QUICK --modes latent,pause,none,cot
```

Colab: `python multimodal/notebooks/build_colab.py` writes `multimodal/notebooks/bitnet_latent_reasoning.ipynb`. By
default it clones `eduardonery1/BitNet` (branch `feature/bitnet-vl`, no submodules). It also has a fallback cell, skipped
by default, that holds embedded copies of the sources for when the clone is stale (`USE_EMBEDDED_SOURCES = True`).
The notebook runs the tests, a tiny sanity run, and the four modes on 2B4T with the QUICK or FULL preset, then the
eval-R sweep and plots. It mounts Google Drive and keeps results and checkpoints there, so after a reclaimed runtime
re-running the notebook skips finished modes and resumes the unfinished one. Add `--check --workdir DIR` to execute its code cells locally on CPU with the tiny model,
using the fallback sources.
