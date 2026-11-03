"""BitNet-MoE: ternary mixture-of-experts BitNet b1.58 (PyTorch reference). See docs/bitnet-moe-design.md.

Expert FFN (identical to transformers' ``BitNetMLP``, v5)::

    y_e(x) = down_e( ffn_sub_norm( relu(gate_e(x))**2 * up_e(x) ) )

* ``relu2`` = ``ACT2FN["relu2"]`` = ``torch.square(torch.relu(x))``.
* ``ffn_sub_norm`` = ``BitNetRMSNorm(n_ff, eps=config.rms_norm_eps)`` (1e-5 for 2B4T):
  ``w * (h / sqrt(mean(h**2) + eps))`` computed in fp32. ONE per layer, shared by every expert.
* gate/up/down are `BitLinear` (absmean ternary weights, per-token absmax int8 activations, STE), so the input of
  ``down`` is int8-quantized AFTER the sub-norm, as in the dense model.

Router: FP ``nn.Linear(d, E, bias=False)``, applied to the post-attention-RMSNorm output (the same input as the
experts). Gating: ``p = softmax(logits.float())``; ``w, idx = topk(p, k)``; ``w /= w.sum()`` (== softmax over the
top-k logits); ``out = sum_j w_j * y_{idx_j}(x)``. Ties in topk follow ``torch.topk`` (lower index not guaranteed).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitNetConfig, BitNetForCausalLM
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput
from transformers.models.bitnet.modeling_bitnet import BitNetRMSNorm

from ..bitlinear import BitLinear, replace_linears
from ..modeling import _ensure_rope_theta, _strip_quant_config
from .losses import load_balancing_loss, router_z_loss

ARCHITECTURE = "BitNetMoEForCausalLM"
MOE_CONFIG_DEFAULTS = {
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "router_aux_loss_coef": 0.01,
    "router_z_loss_coef": 0.001,
    "norm_topk_prob": True,
}
# every ternary matrix of the MoE model (router, embeddings, lm_head and norms stay full precision)
MOE_TERNARY_RE = re.compile(
    r"(^|\.)layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.experts\.\d+\.(gate|up|down)_proj)$")


# --------------------------------------------------------------------------------------------------------------------
# config helpers
# --------------------------------------------------------------------------------------------------------------------
def moe_config_from_dense(dense_cfg: BitNetConfig, **moe_kwargs) -> BitNetConfig:
    """Copy a dense BitNetConfig and add the MoE contract keys."""
    d = _strip_quant_config(BitNetConfig.from_dict(dense_cfg.to_dict())).to_dict()
    d.pop("quantization_config", None)
    d.pop("architectures", None)
    d.update({k: moe_kwargs.get(k, v) if moe_kwargs.get(k) is not None else v for k, v in MOE_CONFIG_DEFAULTS.items()})
    unknown = set(moe_kwargs) - set(MOE_CONFIG_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown MoE config keys: {sorted(unknown)}")
    cfg = BitNetConfig.from_dict(d)
    cfg.architectures = [ARCHITECTURE]
    _validate(cfg)
    return cfg


def _validate(cfg):
    for k in MOE_CONFIG_DEFAULTS:
        if not hasattr(cfg, k):
            setattr(cfg, k, MOE_CONFIG_DEFAULTS[k])
    if not cfg.norm_topk_prob:
        raise ValueError("the contract requires norm_topk_prob = true")
    if not 1 <= cfg.num_experts_per_tok <= cfg.num_experts:
        raise ValueError(f"need 1 <= top-k ({cfg.num_experts_per_tok}) <= E ({cfg.num_experts})")
    if cfg.hidden_act != "relu2":
        raise ValueError(f"BitNet expects hidden_act relu2, got {cfg.hidden_act}")


# --------------------------------------------------------------------------------------------------------------------
# MoE MLP
# --------------------------------------------------------------------------------------------------------------------
class BitNetExpert(nn.Module):
    """gate/up/down of one expert. The activation and the (shared) sub-norm live in `BitNetMoEMLP`."""

    def __init__(self, hidden: int, n_ff: int, qat: bool = True):
        super().__init__()
        lin = BitLinear if qat else nn.Linear
        self.gate_proj = lin(hidden, n_ff, bias=False)
        self.up_proj = lin(hidden, n_ff, bias=False)
        self.down_proj = lin(n_ff, hidden, bias=False)


def route(router_logits: torch.Tensor, top_k: int):
    """softmax over E -> top-k -> renormalize. Returns (weights fp32 [T, k], expert ids [T, k])."""
    probs = torch.softmax(router_logits.float(), dim=-1)
    w, idx = torch.topk(probs, top_k, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
    return w, idx


class BitNetMoEMLP(nn.Module):
    def __init__(self, config: BitNetConfig, qat: bool = True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [BitNetExpert(config.hidden_size, config.intermediate_size, qat) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]
        self.ffn_sub_norm = BitNetRMSNorm(config.intermediate_size, eps=config.rms_norm_eps)
        self.last_router_logits: torch.Tensor | None = None  # [T, E] of the latest forward (for aux losses)
        self.last_expert_counts: torch.Tensor | None = None  # [E] top-k assignment counts of the latest forward

    def expert_forward(self, e: int, x: torch.Tensor) -> torch.Tensor:
        ex = self.experts[e]
        return ex.down_proj(self.ffn_sub_norm(self.act_fn(ex.gate_proj(x)) * ex.up_proj(x)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        T = x.shape[0]
        with torch.autocast(device_type=x.device.type, enabled=False):  # router stays full precision under AMP
            logits = self.router(x.to(self.router.weight.dtype))  # [T, E]
        w, idx = route(logits, self.top_k)
        # dispatch: sort the T*k (token, expert) assignments by expert, run each expert on its tokens only
        flat_e = idx.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        tok = torch.arange(T, device=x.device).repeat_interleave(self.top_k)[order]
        wts = w.reshape(-1)[order]
        counts = torch.bincount(flat_e, minlength=self.num_experts)
        out = torch.zeros_like(x)
        start = 0
        for e, c in enumerate(counts.tolist()):
            if c == 0:
                continue
            t = tok[start:start + c]
            y = self.expert_forward(e, x[t]) * wts[start:start + c, None].to(x.dtype)
            out.index_add_(0, t, y.to(out.dtype))
            start += c
        self.last_router_logits = logits
        self.last_expert_counts = counts.detach()
        return out.reshape(shape)

    def forward_reference(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Naive per-token loop (test oracle for the dispatch)."""
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        w, idx = route(self.router(x), self.top_k)
        rows = []
        for t in range(x.shape[0]):
            acc = torch.zeros_like(x[t])
            for j in range(self.top_k):
                acc = acc + w[t, j].to(x.dtype) * self.expert_forward(int(idx[t, j]), x[t:t + 1])[0]
            rows.append(acc)
        return torch.stack(rows).reshape(hidden_states.shape)


# --------------------------------------------------------------------------------------------------------------------
# causal LM
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class BitNetMoECausalLMOutput(ModelOutput):
    loss: torch.Tensor | None = None
    lm_loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    past_key_values: object | None = None
    router_logits: tuple | None = None  # per layer [T, E]
    expert_counts: torch.Tensor | None = None  # [L, E]


class BitNetMoEForCausalLM(BitNetForCausalLM):
    """HF `BitNetForCausalLM` with every layer's ``mlp`` replaced by `BitNetMoEMLP` and (``qat=True``) every
    attention projection swapped for `BitLinear`. Saves/loads with HF ``save_pretrained``/``from_pretrained`` so state
    dict keys are exactly the contract names (``model.layers.N.mlp.router.weight``,
    ``model.layers.N.mlp.experts.{e}.{gate,up,down}_proj.weight``, ``model.layers.N.mlp.ffn_sub_norm.weight``)."""

    def __init__(self, config: BitNetConfig, qat: bool = True):
        _strip_quant_config(config)
        _validate(config)
        super().__init__(config)
        for layer in self.model.layers:
            layer.mlp = BitNetMoEMLP(config, qat=qat)
        self.config.architectures = [ARCHITECTURE]
        self.post_init()  # (re)initializes the new modules
        if qat:
            replace_linears(self, lambda n, m: bool(MOE_TERNARY_RE.search(n)))

    # ---------------------------------------------------------------- helpers
    @property
    def moe_layers(self) -> list[BitNetMoEMLP]:
        return [layer.mlp for layer in self.model.layers]

    def router_parameters(self) -> list[nn.Parameter]:
        return [m.router.weight for m in self.moe_layers]

    def to_float_linears_(self) -> "BitNetMoEForCausalLM":
        """Replace every BitLinear by an nn.Linear sharing its weight (non-ternarized float forward)."""
        for name, sub in list(self.named_modules()):
            if isinstance(sub, BitLinear):
                lin = nn.Linear(sub.in_features, sub.out_features, bias=sub.bias is not None,
                                device=sub.weight.device, dtype=sub.weight.dtype)
                lin.weight = sub.weight
                if sub.bias is not None:
                    lin.bias = sub.bias
                self.set_submodule(name, lin)
        return self

    # ---------------------------------------------------------------- forward
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs) -> BitNetMoECausalLMOutput:
        for m in self.moe_layers:
            m.last_router_logits = None
        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        router_logits = tuple(m.last_router_logits for m in self.moe_layers)
        counts = torch.stack([m.last_expert_counts for m in self.moe_layers])
        mask = None
        if attention_mask is not None and attention_mask.dim() == 2:
            mask = attention_mask.reshape(-1).bool()
        k = self.config.num_experts_per_tok
        aux = torch.stack([load_balancing_loss(r, k, mask) for r in router_logits]).mean()
        z = torch.stack([router_z_loss(r, mask) for r in router_logits]).mean()
        loss = None
        if out.loss is not None:
            loss = out.loss + self.config.router_aux_loss_coef * aux + self.config.router_z_loss_coef * z
        return BitNetMoECausalLMOutput(loss=loss, lm_loss=out.loss, aux_loss=aux, z_loss=z, logits=out.logits,
                                       past_key_values=out.past_key_values, router_logits=router_logits,
                                       expert_counts=counts)

    # ---------------------------------------------------------------- io
    def save_pretrained(self, save_directory, tokenizer=None, dtype: torch.dtype | None = None, **kwargs):
        """Contract checkpoint: HF safetensors + config.json (top-level rope_theta, no quantization_config)."""
        save_directory = Path(save_directory)
        model = self
        if dtype is not None and self.dtype != dtype:
            import copy
            model = copy.deepcopy(self).to(dtype)
        self.config.architectures = [ARCHITECTURE]
        super(BitNetMoEForCausalLM, model).save_pretrained(save_directory, **kwargs)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_directory)
        _ensure_rope_theta(save_directory)
        cfg_path = save_directory / "config.json"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["architectures"] = [ARCHITECTURE]
        cfg["model_type"] = "bitnet"
        for k, v in MOE_CONFIG_DEFAULTS.items():
            cfg[k] = getattr(self.config, k, v)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str | os.PathLike, dtype: torch.dtype = torch.float32, qat: bool = True, **kw):
        cfg = _strip_quant_config(BitNetConfig.from_pretrained(path))
        if getattr(cfg, "num_experts", None) is None:
            raise ValueError(f"{path}: config.json has no num_experts (not a BitNet-MoE checkpoint)")
        return super().from_pretrained(path, config=cfg, dtype=dtype, qat=qat, **kw)


def load_dense(path: str | os.PathLike, dtype: torch.dtype = torch.float32, qat: bool = True) -> BitNetForCausalLM:
    """Dense HF BitNet (bf16 master weights) with our BitLinear QAT (or float when qat=False)."""
    from ..modeling import LLM_LINEAR_RE

    cfg = _strip_quant_config(BitNetConfig.from_pretrained(path))
    m = BitNetForCausalLM.from_pretrained(path, config=cfg, dtype=dtype)
    if qat:
        replace_linears(m, lambda n, _: bool(LLM_LINEAR_RE.search(n)))
    return m


def contract_tensor_names(config) -> set[str]:
    """Expected checkpoint tensor names (contract)."""
    names = {"model.embed_tokens.weight", "model.norm.weight"}
    if not config.tie_word_embeddings:
        names.add("lm_head.weight")
    for n in range(config.num_hidden_layers):
        p = f"model.layers.{n}"
        names |= {f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight",
                  f"{p}.self_attn.attn_sub_norm.weight", f"{p}.mlp.router.weight", f"{p}.mlp.ffn_sub_norm.weight"}
        names |= {f"{p}.self_attn.{x}_proj.weight" for x in "qkvo"}
        names |= {f"{p}.mlp.experts.{e}.{x}_proj.weight" for e in range(config.num_experts)
                  for x in ("gate", "up", "down")}
    return names
