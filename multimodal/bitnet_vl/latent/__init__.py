"""Latent-reasoning experiment for BitNet (Coconut-style compressed feedback vs pause tokens vs CoT).

See multimodal/README.md, section "Latent reasoning". Modules:
  tasks.py       synthetic variable-tracking programs
  compressor.py  Perceiver-style LatentCompressor (layer-i hidden states -> K soft tokens)
  lora.py        minimal LoRA over nn.Linear / BitLinear
  reasoner.py    LatentReasoner (modes none / latent / pause / cot), KV-cache reuse, curriculum
  train_latent.py CLI + run(config)
"""
