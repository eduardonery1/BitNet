"""LatentCompressor: K learned queries cross-attend (Perceiver-style, pre-LN) over the layer-i hidden states of all
positions seen so far, then Linear -> RMSNorm rescaled to the RMS of the LLM's input-embedding rows.

The output rescaling matters for BitNet: every BitLinear quantizes its input per token with absmax int8, and the
first op on an input embedding is an RMSNorm, but the residual stream keeps the raw embedding, so soft tokens whose
norm is far from that of real token embeddings would be out of distribution for every later layer. The compressor
is always full precision (never BitLinear).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def embedding_rms(emb_weight: torch.Tensor) -> float:
    """Mean over rows of the per-row RMS of an embedding table (the typical norm/sqrt(d) of an input token)."""
    w = emb_weight.detach()
    tot = 0.0
    for i in range(0, w.shape[0], 16384):  # chunked: a 128k x 2560 table in fp32 would need 1.3 GB at once
        tot += float(w[i:i + 16384].float().pow(2).mean(-1).sqrt().sum())
    return tot / w.shape[0]


def rms_rescale(x: torch.Tensor, target_rms: float | torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * target_rms


class CrossAttentionBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        assert d % n_heads == 0
        self.h = n_heads
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        self.ln_mlp = nn.LayerNorm(d)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x, mem, mem_mask=None):
        """x: (B, K, d) latents; mem: (B, L, d); mem_mask: (B, L) 1 = valid."""
        b, k, d = x.shape
        q = self.q(self.ln_q(x)).view(b, k, self.h, d // self.h).transpose(1, 2)
        kv = self.kv(self.ln_kv(mem)).view(b, mem.shape[1], 2, self.h, d // self.h)
        key, val = kv.unbind(2)
        key, val = key.transpose(1, 2), val.transpose(1, 2)
        attn_mask = None
        if mem_mask is not None:
            attn_mask = mem_mask.bool()[:, None, None, :]  # (B, 1, 1, L)
        a = F.scaled_dot_product_attention(q, key, val, attn_mask=attn_mask)
        x = x + self.o(a.transpose(1, 2).reshape(b, k, d))
        return x + self.mlp(self.ln_mlp(x))


class LatentCompressor(nn.Module):
    """``forward(memory, memory_mask, step) -> (B, K, d_model)`` soft tokens with per-token RMS == target_rms.

    Args:
        d_model: LLM hidden size (memory and output dim).
        K: number of soft tokens per latent step.
        n_layers / n_heads: cross-attention blocks over the memory.
        target_rms: RMS of the LLM input embedding rows (`embedding_rms(lm.get_input_embeddings().weight)`).
        max_steps: number of learned step embeddings added to the queries (step index is clipped), so the compressor
            knows which reasoning step it is producing; the memory alone already grows with every step.
    """

    def __init__(self, d_model: int, K: int, n_layers: int = 1, n_heads: int = 8, target_rms: float = 1.0,
                 max_steps: int = 16, mlp_ratio: float = 2.0):
        super().__init__()
        self.d_model, self.K, self.max_steps = d_model, K, max_steps
        self.queries = nn.Parameter(torch.randn(K, d_model) / math.sqrt(d_model))
        self.step_emb = nn.Parameter(torch.zeros(max_steps, d_model))
        self.blocks = nn.ModuleList([CrossAttentionBlock(d_model, n_heads, mlp_ratio) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer("target_rms", torch.tensor(float(target_rms)), persistent=True)

    def forward(self, memory: torch.Tensor, memory_mask: torch.Tensor | None = None, step: int = 0) -> torch.Tensor:
        b = memory.shape[0]
        s = min(int(step), self.max_steps - 1)
        x = (self.queries + self.step_emb[s]).unsqueeze(0).expand(b, -1, -1)
        mem = memory if torch.is_autocast_enabled(memory.device.type) else memory.to(self.queries.dtype)
        for blk in self.blocks:
            x = blk(x, mem, memory_mask)
        y = self.out(self.ln_out(x))
        return rms_rescale(y, self.target_rms)


class PauseTokens(nn.Module):
    """Learned constant pause embeddings (K per step, shared across steps), RMS-matched like the compressor output.
    No dependence on hidden states: the control for the extra compute of latent steps."""

    def __init__(self, d_model: int, K: int, target_rms: float = 1.0, init: torch.Tensor | None = None):
        super().__init__()
        self.K = K
        w = init.detach().float().clone() if init is not None else torch.randn(K, d_model)
        self.emb = nn.Parameter(w)
        self.register_buffer("target_rms", torch.tensor(float(target_rms)), persistent=True)

    def forward(self, batch_size: int, R: int) -> torch.Tensor:
        e = rms_rescale(self.emb, self.target_rms)  # (K, d)
        return e.unsqueeze(0).expand(batch_size, -1, -1).repeat(1, R, 1)  # (B, R*K, d)
