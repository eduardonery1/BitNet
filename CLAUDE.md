# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

bitnet.cpp: the inference framework for 1-bit / ternary (1.58-bit) LLMs such as BitNet b1.58. It is a thin layer on top of a **fork of llama.cpp** (git submodule at `3rdparty/llama.cpp`, tracking `isHuangXin/llama.cpp` branch `release-bitnet-embedding-0.6b-270m`). This repo holds the BitNet-specific CPU kernels, kernel code generators, model conversion scripts and Python wrappers. Generic inference (model loading, sampling, CLI, server) lives in the submodule. The submodule has its own `CLAUDE.md`/`AGENTS.md` for work inside it.

## Setup and build

Requirements: python>=3.10, cmake>=3.22, clang>=18 (the build requires Clang or GCC; `setup_env.py` forces clang). Clone with `--recursive`, or run `git submodule update --init --recursive`.

```bash
pip install -r requirements.txt      # pulls in llama.cpp's conversion requirements
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s           # local model dir
python setup_env.py --hf-repo tiiuae/Falcon3-7B-Instruct-1.58bit -q i2_s   # download from HF
```

`setup_env.py` runs these steps in order, and each step writes its log to `logs/<step>.log`. Check those logs when something fails, because the console only shows a short error:
1. `pip install 3rdparty/llama.cpp/gguf-py`
2. **Codegen**: runs `utils/codegen_tl1.py` (arm64) or `utils/codegen_tl2.py` (x86_64) with the per-model `--BM/--BK/--bm` tile params, which **overwrites `include/bitnet-lut-kernels.h` and `include/kernel_config.ini`**. With `--use-pretuned`, it copies them from `preset_kernels/<model>/` instead. If the model name isn't one of the cases hard-coded in `gen_code()`, it raises `NotImplementedError`.
3. CMake configure + build into `build/` (binaries in `build/bin/`, or `build/bin/Release/` on Windows).
4. Model prep: convert HF → GGUF with `utils/convert-hf-to-gguf-bitnet.py`. For `i2_s` it converts to f32 and then runs `build/bin/llama-quantize ... I2_S 1`. For `tl1`/`tl2` it converts directly with `--outtype tlN`. Output: `<model_dir>/ggml-model-<quant>.gguf`. The step is skipped if that file already exists and is non-empty.

Manual rebuild after editing kernels or headers (same flags setup_env uses):
```bash
cmake -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_EXAMPLES=ON \
      [-DBITNET_X86_TL2=ON | -DBITNET_ARM_TL1=ON]
cmake --build build --config Release [--target llama-cli]
```
`BITNET_ARM_TL1` / `BITNET_X86_TL2` (default OFF) map to the `GGML_BITNET_ARM_TL1` / `GGML_BITNET_X86_TL2` compile definitions, which turn on the LUT kernel paths.

## Running

```bash
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -p "You are a helpful assistant" -cnv
python run_inference_server.py -m <gguf> --host 127.0.0.1 --port 8080
python utils/e2e_benchmark.py -m <gguf> -n 128 -p 512 -t 4
python utils/generate-dummy-bitnet-model.py models/bitnet_b1_58-large --outfile models/dummy-bitnet-125m.tl1.gguf --outtype tl1 --model-size 125M
python utils/convert-helper-bitnet.py ./models/bitnet-b1.58-2B-4T-bf16   # .safetensors → gguf
```
`run_inference*.py` are thin wrappers that shell out to `build/bin/llama-cli` / `llama-server`.

## Tests and benchmarks

There is no unit-test suite or linter config. Validation is done with scripts:
- `utils/test_gemm_kernel.sh [-i N] [-o out.csv] [-s]`: builds and benchmarks the I2_S GEMM kernel standalone (uses `$CXX`, default g++).
- `utils/test_perplexity.py -m <gguf> [--quick] [-e]`: perplexity check through `llama-perplexity`, optionally comparing embedding quantization types.
- `utils/tune_gemm_config.py`: sweeps `ROW_BLOCK_SIZE`/`COL_BLOCK_SIZE`/`PARALLEL_SIZE` in `include/gemm-config.h` and benchmarks each setting.
- `gpu/test.py`: CUDA kernel performance test (see GPU section below).

## Architecture

**Quantization types / kernels**
- `I2_S`: 2-bit packed ternary weights with 8-bit activations (MAD-style, works on both x86 and ARM). Reference kernels are in `src/ggml-bitnet-mad.cpp`, but no CMake file in the current submodule branch compiles that file. The `ggml_vec_dot_i2_i8_s` that actually gets built lives in `3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c` and is called from `ggml-aarch64.c`, so confirm which copy is compiled before editing. Tiling and parallelism are controlled by the macros in `include/gemm-config.h`, which have separate blocks for AVX, NEON+dotprod and NEON, with or without `ACT_PARALLEL`.
- `TL1` (ARM) / `TL2` (x86): lookup-table kernels (T-MAC methodology). `src/ggml-bitnet-lut.cpp` is the glue, and the actual kernels are the **generated** `include/bitnet-lut-kernels.h`. They are specialized to each model's matmul shapes, so a kernel header generated for one model does not work for another. To change these kernels, edit `utils/codegen_tl1.py`/`codegen_tl2.py` (constraints are documented in `docs/codegen.md`) or the `preset_kernels/` copies, not the generated header.
- Public declarations: `include/ggml-bitnet.h`.

**How the kernels get into the build**: the top-level `CMakeLists.txt` adds `src/` and then `3rdparty/llama.cpp`. The submodule's `ggml/src/ggml-cpu/CMakeLists.txt` refers back to `../../../../src/ggml-bitnet-lut.cpp` and `../../../../include`, and `ggml.c` / `ggml-cpu.c` in the submodule include `ggml-bitnet.h`. Kernel changes therefore often require matching changes inside the submodule fork (type registration, dispatch). Those changes must be committed to the submodule's repo separately.

**Model conversion** (`utils/`): `convert-hf-to-gguf-bitnet.py` is the main HF→GGUF converter (it knows the TL1/TL2/I2_S weight layouts). `convert-ms-to-gguf-bitnet.py` and `convert-helper-bitnet.py` handle Microsoft bf16 safetensors checkpoints. `convert-bitnet-embedding-to-gguf.py` and `quantize_embeddings.py` handle the BitNet embedding models (see `docs/bitnet-embeddings-i2s-guide.md`). Converters import `gguf` from the submodule's `gguf-py`.

**Adding a supported model** means touching `SUPPORTED_HF_MODELS` and the `gen_code()` branches in `setup_env.py` (tile params per arch), optionally adding `preset_kernels/<model>/`, and making sure the converter handles its architecture.

**GPU** (`gpu/`): a standalone W2A8 CUDA GEMV kernel and PyTorch inference path for BitNet-b1.58-2B-4T. It is independent of the CPU/llama.cpp build: build it with `cd gpu/bitnet_kernels && bash compile.sh` and install its own `gpu/requirements.txt`. See `gpu/README.md` for checkpoint conversion and `generate.py`.

## Gotchas

- Running `setup_env.py` changes files tracked by git (`include/bitnet-lut-kernels.h`, `include/kernel_config.ini`). Check `git status` before committing.
- Windows builds must run from a VS2022 Developer Command Prompt/PowerShell with ClangCL (`-T ClangCL`).
- An error building llama.cpp's `log.cpp` around `std::chrono` is a known upstream issue (see README FAQ).
