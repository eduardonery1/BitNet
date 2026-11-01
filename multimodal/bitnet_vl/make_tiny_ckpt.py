"""Build a tiny randomly-initialized BitNet-VL checkpoint in the contract layout (for exporter / e2e tests).

    cd multimodal && python -m bitnet_vl.make_tiny_ckpt OUT_DIR [--vision-ternary] [--seed 0]

Shapes (all linear K dims are multiples of 128 so every ternary matrix can be stored I2_S):
  vision: SigLIP hidden 128, mlp 512, 2 layers, 2 heads, image 64, patch 16 -> 4x4 patches -> 2x2 shuffle -> 4 tokens
  projector: LN(512) -> 512x256 -> GELU -> 256x256
  LLM: BitNet hidden 256, intermediate 512, 2 layers, 4 heads (head_dim 64), 2 kv heads, relu2, tied embeddings,
       Llama-3 tokenizer rebuilt offline from 3rdparty/llama.cpp/models/ggml-vocab-llama-bpe.gguf + `<image>`
       (vocab 128257). `language_model/` converts with `utils/convert-hf-to-gguf-bitnet.py --outtype i2_s`.
"""

from __future__ import annotations

import argparse

import torch

from .modeling import BitNetVLForConditionalGeneration
from .tokenizer_utils import build_llama3_tokenizer


def tiny_vision_config(hidden=128, intermediate=512, layers=2, heads=2, image_size=64, patch=16):
    from transformers import SiglipVisionConfig

    return SiglipVisionConfig(hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=layers,
                              num_attention_heads=heads, image_size=image_size, patch_size=patch,
                              vision_use_head=False)


def tiny_llm_config(vocab_size=128256, hidden=256, intermediate=512, layers=2, heads=4, kv_heads=2):
    from transformers import BitNetConfig

    return BitNetConfig(vocab_size=vocab_size, hidden_size=hidden, intermediate_size=intermediate,
                        num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv_heads,
                        max_position_embeddings=2048, tie_word_embeddings=True, bos_token_id=128000,
                        eos_token_id=128009, pad_token_id=128009)


def build_tiny_model(seed: int = 0, vision_ternary: bool = False, tokenizer=None, llm_hidden=256, vision_hidden=128):
    torch.manual_seed(seed)
    tok = tokenizer if tokenizer is not None else build_llama3_tokenizer()
    vc = tiny_vision_config(hidden=vision_hidden)
    lc = tiny_llm_config(vocab_size=len(tok), hidden=llm_hidden)
    model = BitNetVLForConditionalGeneration.from_configs(
        vc, lc, tok, vision_model_name="tiny-random-siglip", llm_model_name="tiny-random-bitnet",
        vision_ternary=vision_ternary)
    # random init leaves LayerNorm/bias at trivial values; perturb so exporters are tested on non-trivial tensors
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim == 1 and ("norm" in n or "bias" in n):
                p.add_(0.02 * torch.randn_like(p))
    return model.eval()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--vision-ternary", action="store_true", help="mark the vision tower ternary (BitLinear)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm-dtype", choices=["bf16", "f32"], default="bf16")
    args = ap.parse_args(argv)
    model = build_tiny_model(args.seed, args.vision_ternary)
    model.save_pretrained(args.out_dir, llm_dtype=torch.bfloat16 if args.llm_dtype == "bf16" else None)
    c = model.config
    print(f"saved tiny BitNet-VL checkpoint to {args.out_dir}: image {c.image_size}px, {c.num_image_tokens} image "
          f"tokens, vision d={c.vision_hidden}, llm d={c.llm_hidden}, vocab {len(model.tokenizer)}, "
          f"image_token_id {c.image_token_id}, vision_ternary={c.vision_ternary}")


if __name__ == "__main__":
    main()
