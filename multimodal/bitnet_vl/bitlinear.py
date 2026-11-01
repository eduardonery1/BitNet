"""BitLinear (W1.58 A8) for quantization-aware training.

Forward semantics (identical to what bitnet.cpp's I2_S `ggml_mul_mat` computes at inference):

* weights: per-tensor absmean ternary. ``s = mean(|W|).clamp(min=1e-5)``; ``W_q = round(W / s).clamp(-1, 1) * s``
  (this is exactly ``BitnetModel.weight_quant`` in ``utils/convert-hf-to-gguf-bitnet.py``).
* activations: per-token absmax int8. ``a = 127 / max(|x|, dim=-1).clamp(min=1e-5)``;
  ``x_q = round(x * a).clamp(-128, 127) / a``.
* gradients: straight-through estimator, ``x + (q(x) - x).detach()``, so the latent full-precision weights receive
  the gradient of the quantized forward.

`BitLinear` subclasses `nn.Linear`, so the parameter names (``weight``, ``bias``) and state-dict keys are unchanged
and a model with BitLinear layers saves/loads exactly like the original HF model.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """Fake-quantize weights to {-s, 0, +s} (no STE)."""
    dtype = w.dtype
    w = w.float()
    s = 1.0 / w.abs().mean().clamp(min=EPS)
    return ((w * s).round().clamp(-1, 1) / s).to(dtype)


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Fake-quantize activations per token (last dim) to int8 levels (no STE)."""
    dtype = x.dtype
    x = x.float()
    s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=EPS)
    return ((x * s).round().clamp(-128, 127) / s).to(dtype)


def ternary_weight(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Export helper: returns (int8 tensor with values in {-1, 0, 1}, scale) with ``w ~= q * scale``.

    ``scale = mean(|w|).clamp(min=1e-5)`` -- the same as the converter's ``weight_quant`` / ``quantize_to_i2_s``.
    """
    w = w.detach().float()
    inv = 1.0 / w.abs().mean().clamp(min=EPS)  # same op order as the converter's weight_quant
    q = (w * inv).round().clamp(-1, 1).to(torch.int8)
    return q, float(1.0 / inv)


class BitLinear(nn.Linear):
    """Drop-in nn.Linear with ternary-weight / int8-activation fake quantization and STE gradients."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None,
                 quantize_activations: bool = True):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.quantize_activations = quantize_activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        w_q = w + (weight_quant(w) - w).detach()
        if self.quantize_activations:
            x = x + (activation_quant(x) - x).detach()
        return F.linear(x, w_q, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "BitLinear":
        new = cls(linear.in_features, linear.out_features, bias=linear.bias is not None,
                  device=linear.weight.device, dtype=linear.weight.dtype)
        # share the parameters: no copy, preserves requires_grad and optimizer identity
        new.weight = linear.weight
        if linear.bias is not None:
            new.bias = linear.bias
        return new

    def extra_repr(self) -> str:
        return super().extra_repr() + ", w=ternary(absmean), a=int8(absmax/token)"


def replace_linears(module: nn.Module, predicate: Callable[[str, nn.Linear], bool] | None = None) -> list[str]:
    """Swap every ``nn.Linear`` submodule (for which ``predicate(qualified_name, linear)`` is true) with a
    `BitLinear` that shares the same weight/bias Parameters. Returns the list of replaced names. Idempotent:
    existing BitLinear modules are left alone.

    Subclasses of nn.Linear that carry their own quantization (e.g. transformers' ``AutoBitLinear``) are also
    replaced; ones with extra buffers such as an offline ``weight_scale`` are rejected because their weights are
    packed, not latent.
    """
    replaced = []
    for name, sub in list(module.named_modules()):
        if not isinstance(sub, nn.Linear) or isinstance(sub, BitLinear):
            continue
        if predicate is not None and not predicate(name, sub):
            continue
        if any(n == "weight_scale" for n, _ in sub.named_buffers(recurse=False)):
            raise ValueError(f"{name}: offline-quantized (packed) linear cannot be used for QAT; "
                             "load the bf16 master-weight checkpoint instead")
        if getattr(sub, "rms_norm", None) is not None:
            raise ValueError(f"{name}: AutoBitLinear with internal rms_norm is not supported")
        module.set_submodule(name, BitLinear.from_linear(sub))
        replaced.append(name)
    return replaced
