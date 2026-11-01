# BitNet-VL: a ternary vision-language model for VQA

Status: design + reference implementation (untrained). Date: 2026-10-31.

## Goal

Visual Question Answering on commodity CPUs through bitnet.cpp: image + question → short answer / chat response,
with the LLM **and** the vision encoder running ternary (W1.58A8) I2_S kernels.

## Architecture decisions

| Component | Choice | Why |
|---|---|---|
| LLM | `microsoft/bitnet-b1.58-2B-4T-bf16` (master weights), hidden 2560 | Strongest public native-ternary LLM; already runs in bitnet.cpp as arch `bitnet-b1.58`. Train from bf16 master weights with BitLinear QAT (absmean ternary W, absmax int8 A, STE). |
| Vision encoder | `google/siglip-base-patch16-384` (86M, 12 layers, d=768, mlp=3072, 576 patches) | SigLIP > CLIP for VQA/OCR (TinyLLaVA). Every linear has K ∈ {768, 3072}, a multiple of 128, so **all** encoder linears can be stored I2_S. so400m has `fc2` K=4304, not 128-aligned, so it is offered only as a "quality" variant with fc2 kept F16. CPU prefill cost is ~5× lower than so400m@384. |
| Token compression | 2×2 pixel shuffle (24×24×768 → 12×12×3072): **144 visual tokens** | On CPU, prefill dominates latency; 144 tokens is SmolVLM-style compression while keeping 384px detail for TextVQA. |
| Projector | LayerNorm(3072) → Linear(3072→2560) → GELU → Linear(2560→2560), **full precision** (F16 in GGUF) | Negligible params; literature shows the projector is sensitive to low-bit quantization (EaqVLA) and BitVLA keeps it FP. |
| Encoder quantization | Phase A: bf16 encoder. Phase B: ternary encoder (BitLinear on q/k/v/out/fc1/fc2) trained BitVLA-style with layer-wise hidden-state distillation from the bf16 teacher | BitVLA: ternary SigLIP-L costs ~1.5 avg points; without distillation −8.4. Patch-embed conv, position embeddings and LayerNorms stay FP. |

Structural note: the BitLinear forward in the encoder is "quantize activations per token to int8 (absmax) and weights to ternary (absmean)". The I2_S CPU kernel computes exactly this inside `ggml_mul_mat`, so the stock SigLIP graph in `clip.cpp` can execute a ternary encoder **unchanged** once its weights are stored as I2_S. No extra sub-norms are added to the ViT, which keeps the vanilla SigLIP block structure.

## Training recipe (BitVLA / LLaVA-1.5 derived)

| Stage | Trainable | Frozen | Data | LR | Notes |
|---|---|---|---|---|---|
| 1 Align | projector | encoder (bf16), LLM (BitLinear QAT forward) | LLaVA-Pretrain LCS-558K captions | 1e-3, bs 256, 1 ep, no WD | |
| 2 Instruct | projector + LLM (QAT) | encoder | llava_v1_5_mix665k (VQAv2, GQA, OK-VQA, A-OKVQA, TextVQA, OCR-VQA, VG, RefCOCO + conversations), optional + The Cauldron | 3e-4 LLM / 1e-3 proj, bs 256, WD 0.1→0, cosine | Ternary LLMs need higher LR than FP (BitVLA, 2B4T SFT). Loss on answer tokens only. |
| 3 Ternarize ViT | encoder (BitLinear) | LLM, projector | subset of stage-2 mix (≥2–5B tokens) | 1e-4, WD 0.01 | L = L_LM + 0.1 · mean_l MSE(h_l^bf16, h_l^ternary) over all 12 encoder layers. |

Budget estimate (extrapolated, unverified): stages 1+2 at the LLaVA-1.5 data scale ≈ 40–80 A100-hours; stage 3 at 2–5B tokens ≈ 200–500 A100-hours.
Evaluation: lmms-eval on VQAv2, GQA, TextVQA, ScienceQA-IMG, POPE, MMBench, MMStar; plus bitnet.cpp CPU latency (image encode ms, prefill tok/s, decode tok/s) and GGUF size.

## Prompt format

The BitNet 2B4T chat template, with the image placed before the question:
```
<|begin_of_text|>User: <image>\n{question}<|eot_id|>Assistant: {answer}<|eot_id|>
```
`<image>` is expanded to 144 projected embeddings. In llama.cpp it is the mtmd media marker.

## Interface contract between components

**Training checkpoint** (directory `ckpt/`):
- `config.json`: `{"model_type": "bitnet_vl", "vision_model_name": ..., "llm_model_name": ..., "pixel_shuffle": 2, "vision_ternary": bool, "image_size": 384, "num_image_tokens": 144, "llm_hidden": 2560}`
- `vision_tower.safetensors`: HF `SiglipVisionModel` state-dict keys (`vision_model.embeddings...`, `vision_model.encoder.layers.N...`). These are latent bf16/fp32 weights; ternarization happens at export when `vision_ternary`.
- `projector.safetensors`: `norm.weight`, `norm.bias`, `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias`
- `language_model/`: HF BitNet checkpoint (save_pretrained, bf16 master weights) + tokenizer

**GGUF deployment**:
- LLM: `utils/convert-hf-to-gguf-bitnet.py language_model/ --outtype i2_s` (arch `bitnet-b1.58`).
- mmproj: `multimodal/convert_bitnet_vl_mmproj.py ckpt/ --outfile mmproj-bitnet-vl.gguf [--vision-i2s]`. Encoder 2D linears go to I2_S when ternary; patch embed, pos embed, norms and projector stay F16/F32.
- Run: `build/bin/llama-mtmd-cli -m ggml-model-i2_s.gguf --mmproj mmproj-bitnet-vl.gguf --image img.jpg -p "What color is the bus?"`
