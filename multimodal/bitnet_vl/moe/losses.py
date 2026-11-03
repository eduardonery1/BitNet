"""Router auxiliary losses (per layer; the model averages them over layers).

Switch load-balancing loss with top-k (Fedus et al. 2021; Mixtral-style top-k counting)::

    f_e = (# of top-k slots assigned to expert e) / (T * k)      (non-differentiable)
    P_e = mean_t softmax(logits_t)_e                             (differentiable)
    L_aux = E * sum_e f_e * P_e

Normalized by T*k so that perfectly uniform routing gives exactly 1.0 for any k (HF Mixtral's
``load_balancing_loss_func`` sums over the k slots instead and gives k). Minimum 1.0 when f and P are both uniform;
maximum E when everything collapses onto one expert.

Router z-loss (ST-MoE, Zoph et al. 2022): ``L_z = mean_t logsumexp(logits_t)^2``.

``mask`` (optional, [T] bool) excludes padding tokens from both losses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """mean over dim 0."""
    if mask is None:
        return x.mean(0)
    m = mask.to(x.dtype).reshape(-1, *([1] * (x.dim() - 1)))
    return (x * m).sum(0) / m.sum().clamp(min=1)


def expert_fractions(router_logits: torch.Tensor, top_k: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    """f_e: fraction of the T*k top-k assignments that go to each expert ([E], sums to 1)."""
    E = router_logits.shape[-1]
    idx = torch.topk(router_logits.float(), top_k, dim=-1).indices  # topk(logits) == topk(softmax(logits))
    hits = F.one_hot(idx, E).sum(1).float()  # [T, E]
    return _masked_mean(hits, mask) / top_k


def load_balancing_loss(router_logits: torch.Tensor, top_k: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    logits = router_logits.reshape(-1, router_logits.shape[-1]).float()
    E = logits.shape[-1]
    f = expert_fractions(logits, top_k, mask).detach()
    P = _masked_mean(torch.softmax(logits, dim=-1), mask)
    return E * (f * P).sum()


def router_z_loss(router_logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    logits = router_logits.reshape(-1, router_logits.shape[-1]).float()
    return _masked_mean(torch.logsumexp(logits, dim=-1).pow(2), mask)
