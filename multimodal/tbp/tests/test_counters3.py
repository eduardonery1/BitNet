"""3-bit packed flip counters and stochastic counter increments."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

if torch.cuda.is_available():
    from tbp import kernels as K

DEV = "cuda"
M32 = 0xFFFFFFFF


def vote_hash(seed, r, c):
    h = ((seed * 0x9E3779B1) & M32) ^ (((r + 0x7F4A7C15) * 0x85EBCA77) & M32) ^ (((c + 0x165667B1) * 0xC2B2AE3D) & M32)
    h ^= h >> 15
    h = (h * 0x2C1B3C6D) & M32
    h ^= h >> 12
    h = (h * 0x297A2D39) & M32
    h ^= h >> 15
    return h


def ref_update(G, W, cnt, k, T, p_inc=1.0, seed=0):
    G, W, cnt = G.cpu().float(), W.cpu().clone().long(), cnt.cpu().clone().long()
    thr = K.acc_threshold(p_inc)
    flips = 0
    for r in range(G.shape[0]):
        cand = []
        for c in range(G.shape[1]):
            v = float(G[r, c])
            if v == 0.0 or math.isnan(v):
                continue
            d = -1 if v > 0 else 1
            w = int(W[r, c])
            if (d > 0 and w < 1) or (d < 0 and w > -1):
                cand.append((-abs(v), c, d))
        cand.sort()
        for _, c, d in cand[:k]:
            old = int(cnt[r, c])
            sc = (old > 0) - (old < 0)
            if sc == d:
                cv = old + d if (vote_hash(seed, r, c) >> 8) < thr else old
            else:
                cv = d
            if abs(cv) >= T:
                W[r, c] += 1 if cv > 0 else -1
                cv = 0
                flips += 1
            cnt[r, c] = cv
    return W.to(torch.int8), cnt.to(torch.int8), flips


@pytest.mark.parametrize("R,C", [(1, 1), (2, 9), (3, 10), (4, 11), (5, 784), (2, 4099)])
def test_pack_counters3_roundtrip(R, C):
    g = torch.Generator().manual_seed(C)
    c = torch.randint(-3, 4, (R, C), generator=g, dtype=torch.int8).to(DEV)
    Cp = K.pack_counters3(c)
    assert Cp.shape == (R, (C + 9) // 10)
    assert torch.equal(K.unpack_counters3(Cp, C), c)
    # layout: field j of word w holds column 10 w + j as code c + 3
    w0 = int(Cp[0, 0])
    for j in range(min(10, C)):
        assert ((w0 >> (3 * j)) & 7) - 3 == int(c[0, j])


def run(G, W, cnt, k, T, bits, cache=None, p_inc=1.0, seed=0):
    Wp = K.pack(W.to(DEV))
    c = cnt.to(DEV)
    cdev = K.pack_counters3(c) if bits == 3 else c.contiguous()
    f = K.update(G.to(DEV), Wp, cdev, k, T, cache=cache, p_inc=p_inc, seed=seed)
    cout = K.unpack_counters3(cdev, W.shape[1]) if bits == 3 else cdev
    return K.unpack(Wp, W.shape[1]).cpu(), cout.cpu(), f


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("R,C,k,T", [(3, 7, 3, 1), (6, 101, 5, 2), (8, 784, 4, 4), (4, 1786, 9, 3), (2, 12000, 60, 4)])
def test_update3_matches_reference(cache, R, C, k, T):
    g = torch.Generator().manual_seed(R * C + T)
    W = torch.randint(-1, 2, (R, C), generator=g, dtype=torch.int8)
    G = torch.randint(-40, 41, (R, C), generator=g).float() / 9
    cnt = torch.randint(-(T - 1), T, (R, C), generator=g, dtype=torch.int8)
    Wr, cr, fr = ref_update(G, W, cnt, k, T)
    Wg, cg, fg = run(G, W, cnt, k, T, 3, cache)
    assert torch.equal(Wg, Wr) and torch.equal(cg, cr) and fg == fr


@pytest.mark.parametrize("T", [1, 2, 3, 4])
def test_3bit_equals_int8_over_many_steps(T):
    g = torch.Generator().manual_seed(T)
    R, C, k = 6, 333, 7
    W = torch.randint(-1, 2, (R, C), generator=g, dtype=torch.int8)
    Wp8, Wp3 = K.pack(W.to(DEV)), K.pack(W.to(DEV))
    c8 = torch.zeros(R, C, dtype=torch.int8, device=DEV)
    c3 = K.pack_counters3(c8)
    for step in range(30):
        G = torch.randn(R, C, generator=g)
        G[:, :50] = 3.0 + torch.rand(50, generator=g)  # persistent votes so counters fill up
        f8 = K.update(G.to(DEV), Wp8, c8, k, T)
        f3 = K.update(G.to(DEV), Wp3, c3, k, T)
        assert f8 == f3
        assert torch.equal(Wp8, Wp3) and torch.equal(K.unpack_counters3(c3, C), c8)


@pytest.mark.parametrize("bits", [8, 3])
def test_stochastic_increments_match_reference(bits):
    g = torch.Generator().manual_seed(11)
    R, C, k, T, p = 5, 640, 12, 4, 3 / 7
    W = torch.randint(-1, 2, (R, C), generator=g, dtype=torch.int8)
    cnt = torch.zeros(R, C, dtype=torch.int8)
    Wr, cr = W.clone(), cnt.clone()
    Wp = K.pack(W.to(DEV))
    cdev = K.pack_counters3(cnt.to(DEV)) if bits == 3 else cnt.to(DEV)
    for step in range(25):
        G = torch.randn(R, C, generator=g)
        G[:, :30] = 2.0 + torch.rand(30, generator=g)
        Wr, cr, fr = ref_update(G, Wr, cr, k, T, p_inc=p, seed=step)
        fg = K.update(G.to(DEV), Wp, cdev, k, T, p_inc=p, seed=step)
        cg = K.unpack_counters3(cdev, C) if bits == 3 else cdev
        assert fg == fr and torch.equal(K.unpack(Wp, C).cpu(), Wr) and torch.equal(cg.cpu(), cr)


def test_stochastic_rate():
    """With p_inc = 3/7 a persistently voted weight needs 1 + 3/p = 8 votes on average to flip (threshold 4)."""
    R, C = 1, 20000
    W = torch.zeros(R, C, dtype=torch.int8, device=DEV)
    Wp = K.pack(W)
    c3 = K.pack_counters3(torch.zeros(R, C, dtype=torch.int8, device=DEV))
    G = torch.full((R, C), 1.0, device=DEV)  # every weight votes "down" every step (k = C)
    first_flip = torch.full((C,), -1)
    for step in range(60):
        K.update(G, Wp, c3, C, 4, p_inc=3 / 7, seed=step)
        Wnow = K.unpack(Wp, C)[0].cpu()
        newly = (Wnow == -1) & (first_flip < 0)
        first_flip[newly] = step + 1
        if bool((first_flip > 0).all()):
            break
    mean_votes = first_flip[first_flip > 0].float().mean().item()
    assert abs(mean_votes - 8.0) < 0.15, mean_votes
