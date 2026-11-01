# BitNet-VL: CPU inference with bitnet.cpp

This doc covers running a BitNet-VL checkpoint (see `docs/bitnet-vl-design.md`) on the CPU through
`llama-mtmd-cli`: how to convert it, how the vision graph is built, and the known issues.
The LLM runs as arch `bitnet-b1.58` (I2_S). The vision encoder runs through the stock SigLIP graph in
`tools/mtmd/clip.cpp`, with its linears optionally stored as I2_S. A small new projector type, `bitnet_vl`, adds
the pixel-shuffle + LayerNorm + MLP head.

## Build

```bash
cmake -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j12 --target llama-mtmd-cli llama-cli
```
I verified this build with gcc 15.2 (`-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++`) on x86_64 with the default
I2_S path (TL1/TL2 off). It needs no codegen, and the tracked `include/bitnet-lut-kernels.h` is used unchanged.

## Convert

```bash
export PYTHONPATH=3rdparty/llama.cpp/gguf-py   # utils/ converter imports `gguf`; also needs sentencepiece
# LLM (ckpt/language_model is a HF BitNet checkpoint with bf16 master weights)
python utils/convert-hf-to-gguf-bitnet.py ckpt/language_model --outtype i2_s --outfile ggml-model-i2_s.gguf
# vision tower + projector
python multimodal/convert_bitnet_vl_mmproj.py ckpt/ --outfile mmproj-bitnet-vl-f16.gguf              # FP encoder (phase A)
python multimodal/convert_bitnet_vl_mmproj.py ckpt/ --outfile mmproj-bitnet-vl-i2s.gguf --vision-i2s # ternary encoder (phase B)
```
Converter options: `--outtype f32` keeps every non-I2_S tensor F32, which is useful for parity debugging.
`--vision-heads N` sets the head count when `config.json` has no `vision_config`.
`--no-post-ln` / `--feature-layer K` are for models that do not feed `last_hidden_state` to the projector.
The converter refuses a `config.json` whose `projector_act`, `projector_norm_eps`, `pixel_shuffle_order` or
`vision_feature` differ from what the C++ graph implements.

Do **not** make I2_S files with `llama-quantize` (see "Known issues"). The Python converters pack I2_S themselves.

## Run

```bash
build/bin/llama-mtmd-cli -m ggml-model-i2_s.gguf --mmproj mmproj-bitnet-vl-i2s.gguf \
    --image img.jpg -p $'<__media__>\nWhat color is the bus?' -n 64 --temp 0
```
- Training puts the image as `User: <image>\n{question}`. `llama-mtmd-cli` *prepends* the marker without a newline
  when the prompt has none, so pass the `<__media__>\n` yourself if you want an exact match.
- The 2B4T GGUF carries its own chat template. A checkpoint whose tokenizer has none (e.g. the tiny test one) needs
  `--jinja --chat-template "{{ bos_token }}{% for message in messages %}{{ message['role'] | capitalize }}: {{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"`.
- The vision tokens are decoded causally, like the PyTorch model.

## mmproj GGUF contents

`general.architecture = clip`, `clip.projector_type = bitnet_vl`, `clip.has_vision_encoder = true`,
`clip.use_gelu = true` (SigLIP MLP = `gelu_pytorch_tanh` = `ggml_gelu`), `clip.vision.{image_size, patch_size,
embedding_length, feed_forward_length, block_count, attention.head_count, attention.layer_norm_epsilon (1e-6),
projection_dim (= LLM n_embd), image_mean/std (0.5), projector.scale_factor (2)}`.

| GGUF tensor | HF source | type (default / `--vision-i2s`) |
|---|---|---|
| `v.patch_embd.weight` / `.bias` | `embeddings.patch_embedding` (conv 16x16) | F16 / F16, bias F32 |
| `v.position_embd.weight` | `embeddings.position_embedding` | F32 |
| `v.blk.N.attn_{q,k,v,out}.weight` | `encoder.layers.N.self_attn.{q,k,v,out}_proj` | F16 / **I2_S** |
| `v.blk.N.ffn_up.weight` / `ffn_down.weight` | `mlp.fc1` / `mlp.fc2` | F16 / **I2_S** |
| `v.blk.N.ln1`, `ln2` (weight, bias), all linear biases | `layer_norm1/2`, `*.bias` | F32 |
| `v.post_ln.{weight,bias}` | `post_layernorm` | F32 |
| `mm.input_norm.{weight,bias}` | `projector.norm` | F32 |
| `mm.1.{weight,bias}`, `mm.2.{weight,bias}` | `projector.fc1`, `projector.fc2` | F16 (+F32 bias), both modes |

For I2_S, each weight is ternarized per tensor with absmean: `inv = 1/max(mean|W|, 1e-5)`, `q = clip(round(W*inv), -1, 1)`,
`scale = 1/inv`. This is the same op order as `bitnet_vl.bitlinear.ternary_weight`. The weight is then packed by
`quantize_to_i2_s` from `utils/convert-hf-to-gguf-bitnet.py` (imported; a byte-identical vendored copy is the fallback).
A tensor whose K is not a multiple of 128 falls back to F16 with a warning. SigLIP-base has K in {768, 3072}, so all
72 encoder linears become I2_S. The encoder mmproj is 202 MB as F16 and 54 MB with I2_S.

## Projector graph (`tools/mtmd/models/siglip.cpp`, branch `PROJECTOR_TYPE_BITNET_VL`)

```
SigLIP ViT (stock build_vit, pre-LN, learned pos-embd, + post_ln)    [576, 768]
-> pixel shuffle s=2 (build_patch_merge_permute, shared with idefics3/LFM2)   [144, 3072]
-> LayerNorm(3072, eps 1e-5) -> Linear(3072->2560)+b -> GELU (erf) -> Linear(2560->2560)+b
```
The pixel-shuffle ordering is HF `Idefics3Connector.pixel_shuffle`. For a W x W patch grid (row-major, W = 24) with
D channels:
```
y[i*(W/s) + j, (dy*s + dx)*D + c] = x[(s*i + dy)*W + (s*j + dx), c]      i, j in [0, W/s), dy, dx in [0, s)
```
This matches `bitnet_vl.modeling.pixel_shuffle`. The C++ code uses the existing `build_patch_merge_permute`
unchanged; I derived its index mapping by hand and it gives the same formula. It is also checked numerically below.

Preprocessing is `mtmd_image_preprocessor_fixed_size` with `RESIZE_ALGO_BICUBIC_PILLOW` and `PAD_NONE`. That is a
Pillow-exact bicubic stretch to `image_size` (no crop, no pad), then `(x/255 - 0.5)/0.5`, the same as
`bitnet_vl.data.preprocess_image`.

Why a new projector type instead of reusing `lfm2` (which has the same shuffle + LN + 2-layer MLP):
- LFM2 uses dynamic-resolution tiling, a 64..256 token limit and position-embedding resizing.
- LFM2 has tanh-GELU and an optional norm.
- BitNet-VL is a fixed 384x384 SigLIP with erf GELU.

## I2_S in the vision encoder: semantics check

`ggml_mul_mat(I2_S weight, f32 x)` (`ggml/src/ggml-cpu/ggml-cpu.c`) runs these steps:
1. It quantizes each row of `x` (one token) with `quantize_row_i8_s`: `s = 127/max|x|`, `q = clamp(roundf(x*s), -128, 127)`.
   It also keeps `sum(q)`.
2. It runs the int8 x 2-bit GEMM. With `GGML_USE_LLAMAFILE` (default) this is `llamafile_sgemm_i2s` (tinyBLAS
   AVX2, any M/N, tiles 4x4 down to 1x1). Without it, it falls back to `ggml_gemm_i2_i8_s` / `gemv`.
3. It computes `y = (dot(code, q) - sum(q)) / s * w_scale`, where code = w+1.

This is exactly the BitLinear forward (per-token absmax int8 activations, per-tensor absmean ternary weights), so the
SigLIP graph needs no change. The large batch (576 patch rows, i.e. 576 columns of src1) goes through the GEMM path.
I tested it with real SigLIP-base shapes, see the parity results below.

## Verification (random weights, x86_64, 12 threads)

These checks used `python -m bitnet_vl.make_tiny_ckpt` (SigLIP d=128, 2 layers, 64px -> 4 tokens, LLM d=256) and a
full-size random checkpoint (SigLIP-base-384, 12 layers, projector 3072->2560, 1-layer d=2560 LLM).
`llama-mtmd-cli` loads both mmproj variants, encodes the image and generates tokens. The output text is garbage, as
expected for random weights.

The embeddings come from the mtmd API (`mtmd_encode_chunk` + `mtmd_get_output_embd`). They are compared with the
PyTorch reference `projector(pixel_shuffle(SiglipVisionModel(px).last_hidden_state))`:

| case | reference | rel. RMS err | min per-token cosine |
|---|---|---|---|
| tiny, F32 mmproj, 64x64 image (no resize) | FP encoder | 6.4e-4 | 1.000000 |
| tiny, F32 mmproj, 200x150 image (resized) | FP encoder | 7.0e-4 | 1.000000 |
| tiny, I2_S mmproj | BitLinear encoder | 7.0e-3 | 0.99996 |
| full, F16 mmproj, 640x480 image | FP encoder | 9.1e-4 | 0.999999 |
| full, I2_S mmproj, 640x480 image | BitLinear encoder | 1.5e-2 | 0.99980 |
| full, I2_S mmproj | FP encoder (ternarization error itself) | 0.88 | 0.54 |

The I2_S residual vs the BitLinear reference comes from int8 rounding ties and accumulation-order differences
(`roundf` rounds half away from zero, `torch.round` rounds half to even). Random weights amplify these through 12
layers. The last row shows that an *untrained* (latent-FP) encoder must not be exported with `--vision-i2s`.
Before the switch to Pillow bicubic, the plain `RESIZE_ALGO_BICUBIC` gave 5% error on resized images.

Image encode time on the full-size random model (`llama-mtmd-cli`, 12 threads, 384px, 576 patches): ~230 ms with
the I2_S encoder vs ~380-520 ms with F16.

## Known issues / bugs found

1. **`llama-quantize ... I2_S` produces corrupt weights.** `quantize_i2_s` in `ggml/src/ggml-cpu/quants.c` packs 4
   *consecutive* values per byte, but the kernels and `dequantize_row_i2_s` expect the 128-block interleaved layout
   (byte j of a block = elements j, 32+j, 64+j, 96+j). `setup_env.py` uses exactly this path for `-q i2_s` (f32 GGUF ->
   `llama-quantize I2_S`). Use `convert-hf-to-gguf-bitnet.py --outtype i2_s` instead. Its scale is also
   "first nonzero |w|" (fine only for already-ternarized input).
2. **`general.file_type` 40 vs 41.** `utils/convert-hf-to-gguf-bitnet.py` writes 40 for I2_S models, which is
   `LLAMA_FTYPE_MOSTLY_Q1_0` in this llama.h. `LLAMA_FTYPE_MOSTLY_I2_S` is 41. The effect is cosmetic (wrong
   ftype name in logs), but the converter should be fixed. The mmproj converter writes 41.
3. gguf-py in the submodule defines `LlamaFileType` twice in `constants.py`. The second definition wins and lacks
   `MOSTLY_I2_S`. `GGUFWriter` also has no `add_description`. The mmproj converter works around both.
4. `quantize_row_i8_s` sets `s = 0` for an all-zero activation row, giving `0/0 = NaN`. PyTorch clamps `max|x|` to
   1e-5 and gives 0. This cannot happen after a LayerNorm with non-zero bias, but it would for zero-padded tokens.
5. tinyBLAS I2S uses `nb = k/128` and silently drops a K tail. The non-llamafile GEMM fallback rounds the row
   range up to a multiple of 4 (out-of-bounds when N % 4 != 0). The converter enforces K % 128. All BitNet-VL N are
   multiples of 4.
6. This was tested on x86_64/AVX2 only. ARM (NEON I2_S path, `QK_I2_S = 64` on some builds) is not verified for
   the vision encoder.
7. No accuracy numbers exist yet. All checks above use random weights, and real-checkpoint VQA quality is untested.
