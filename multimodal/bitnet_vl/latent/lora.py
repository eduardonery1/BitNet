"""Minimal LoRA (no peft) for nn.Linear / BitLinear.

``y = base(x) + (dropout(x) @ A^T @ B^T) * alpha / r``. The LoRA path is full precision and is added to the
(BitLinear W1.58A8) base output; the base weight is frozen. B is zero-initialized so a freshly wrapped model is
exactly the base model.

For a frozen BitLinear base, `LoRALinear.prequantize_base_()` replaces the latent weight by its ternary fake-quant
once (in place, no extra memory) so the forward skips the per-call absmean quantization. This gives the same output
as the BitLinear forward (tested) but the module can no longer be trained or exported as master weights, so it is
only used when the base is frozen.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bitlinear import BitLinear, activation_quant, weight_quant

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_KEY = "lora_"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.prequantized = False
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):  # some HF utilities peek at .weight (dtype/device)
        return self.base.weight

    @torch.no_grad()
    def prequantize_base_(self):
        if isinstance(self.base, BitLinear) and not self.prequantized:
            self.base.weight.data.copy_(weight_quant(self.base.weight.data))
            self.prequantized = True

    def base_forward(self, x):
        if self.prequantized:
            if self.base.quantize_activations:
                x = x + (activation_quant(x) - x).detach()  # STE so grads still reach earlier layers' LoRA
            return F.linear(x, self.base.weight, self.base.bias)
        return self.base(x)

    def forward(self, x):
        y = self.base_forward(x)
        a = self.lora_dropout(x)
        if not torch.is_autocast_enabled(x.device.type):
            a = a.to(self.lora_A.dtype)
        lo = F.linear(F.linear(a, self.lora_A), self.lora_B) * self.scaling
        return y + lo.to(y.dtype)

    def extra_repr(self):
        return f"r={self.r}, alpha={self.alpha}, prequantized={self.prequantized}"


def _default_predicate(targets):
    pat = re.compile(r"(^|\.)(" + "|".join(re.escape(t) for t in targets) + r")$")
    return lambda name, mod: bool(pat.search(name))


def apply_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, dropout: float = 0.0,
               predicate: Callable[[str, nn.Module], bool] | None = None,
               targets=DEFAULT_TARGETS, prequantize: bool = False) -> list[str]:
    """Wrap every nn.Linear (incl. BitLinear) selected by `predicate(name, module)` (default: name ends with one of
    `targets`) in a LoRALinear. Returns the wrapped names. Does NOT freeze the rest of the model (see `freeze_base`)."""
    pred = predicate or _default_predicate(targets)
    names = []
    for name, sub in list(model.named_modules()):
        if isinstance(sub, nn.Linear) and pred(name, sub):
            w = LoRALinear(sub, r=r, alpha=alpha, dropout=dropout)
            if prequantize:
                w.prequantize_base_()
            model.set_submodule(name, w)
            names.append(name)
    return names


def freeze_base(model: nn.Module):
    """requires_grad=False for every parameter except LoRA A/B."""
    for n, p in model.named_parameters():
        p.requires_grad_(LORA_KEY in n.rsplit(".", 1)[-1])


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if LORA_KEY in n.rsplit(".", 1)[-1]]


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    seen, total = set(), 0
    for p in module.parameters():
        if id(p) in seen or (trainable_only and not p.requires_grad):
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if LORA_KEY in k.rsplit(".", 1)[-1]}


def save_trainable(path: str | Path, lm: nn.Module, modules: dict[str, nn.Module] | None = None,
                   include_base: bool = False):
    """Save LoRA weights of `lm` plus the full state of each extra module (e.g. {"compressor": comp}).
    include_base=True also saves the full LM state dict (used with --train-base)."""
    sd = {"lm." + k: v for k, v in (lm.state_dict() if include_base else lora_state_dict(lm)).items()}
    for prefix, m in (modules or {}).items():
        if m is not None:
            sd.update({f"{prefix}.{k}": v.detach().cpu() for k, v in m.state_dict().items()})
    torch.save({k: v.detach().cpu().contiguous() for k, v in sd.items()}, str(path))


def load_trainable(path: str | Path, lm: nn.Module, modules: dict[str, nn.Module] | None = None, strict=True):
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    lm_sd = {k[3:]: v for k, v in sd.items() if k.startswith("lm.")}
    missing, unexpected = lm.load_state_dict(lm_sd, strict=False)
    if strict:
        need = [k for k in lora_state_dict(lm) if k not in lm_sd]
        if need or unexpected:
            raise RuntimeError(f"LoRA keys missing {need[:5]} unexpected {list(unexpected)[:5]}")
    for prefix, m in (modules or {}).items():
        if m is None:
            continue
        sub = {k[len(prefix) + 1:]: v for k, v in sd.items() if k.startswith(prefix + ".")}
        m.load_state_dict(sub, strict=strict)
    return sd
