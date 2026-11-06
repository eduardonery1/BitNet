"""CPU backend with the same API as tbp.kernels, for running TBPNet without a GPU.

Weights and counters are kept unpacked (int8 tensors) and matmuls are dense float32: storage savings of the packed GPU
format are not reproduced, only the algorithm. The update picks the top-k per row with torch.topk, so among exactly
equal scores the chosen column may differ from the GPU kernel (which takes the lowest index); otherwise it follows the
same rule (direction -sign(G), movability, counter restart on opposite votes, flip at |counter| >= T).
"""

from __future__ import annotations

import math

import torch


def pack(W: torch.Tensor) -> torch.Tensor:
    return W.to(torch.int8).cpu().contiguous().clone()


def unpack(Wp: torch.Tensor, cols: int, dtype=torch.int8) -> torch.Tensor:
    assert Wp.shape[1] == cols
    return Wp.to(dtype)


def mm_nt(A: torch.Tensor, Wp: torch.Tensor, n_out: int, scale: float = 1.0) -> torch.Tensor:
    return (A.float() @ Wp.float().T) * scale


def mm_nn(A: torch.Tensor, Wp: torch.Tensor, n_cols: int, scale: float = 1.0) -> torch.Tensor:
    return (A.float() @ Wp.float()) * scale


def pack_counters(c: torch.Tensor, bits: int) -> torch.Tensor:
    return c.to(torch.int8).cpu().clone()


def unpack_counters(Cp: torch.Tensor, cols: int, bits: int) -> torch.Tensor:
    return Cp


def sample_krows(k_frac: float, rows: int, fan_in: int, device="cpu", gen: torch.Generator | None = None):
    e = k_frac * fan_in
    base = math.floor(e)
    extra = (torch.rand(rows, generator=gen) < (e - base)).to(torch.int32)
    return (extra + base).to(torch.int32)


def update(G, Wp, cnt, k, T, cache=None, p_inc=1.0, seed=0, counter_bits=None, flips_out=None, krows=None):
    assert p_inc == 1.0, "stochastic counter increments are only implemented in the CUDA kernel"
    G = G.float()
    R, C = G.shape
    d = -torch.sign(G).to(torch.int8)
    movable = ((d > 0) & (Wp < 1)) | ((d < 0) & (Wp > -1))
    score = torch.where(movable & (G != 0), G.abs(), torch.full_like(G, -1.0))
    kmax = int(krows.max()) if krows is not None else int(k)
    kmax = min(kmax, C)
    if kmax <= 0:
        return 0
    vals, idx = score.topk(kmax, dim=1)
    sel = vals > 0
    if krows is not None:
        sel &= torch.arange(kmax).unsqueeze(0) < krows.long().unsqueeze(1)
    rows = torch.arange(R).unsqueeze(1).expand_as(idx)
    r, c = rows[sel], idx[sel]
    dd = d[r, c]
    old = cnt[r, c]
    new = torch.where(torch.sign(old) == dd, old + dd, dd)
    fire = new.abs() >= T
    Wp[r[fire], c[fire]] += torch.sign(new[fire])
    new[fire] = 0
    cnt[r, c] = new
    n = int(fire.sum())
    if flips_out is not None:
        flips_out += n
        return None
    return n
