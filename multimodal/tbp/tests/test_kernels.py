"""Correctness tests for the TBP CUDA kernels against plain PyTorch references.

    cd multimodal && runs/.venv-tbp/bin/python -m pytest tbp/tests -q
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

if torch.cuda.is_available():
    from tbp import kernels as K

DEV = "cuda"


def rand_tern(R, C, gen, p0=1 / 3):
    W = torch.randint(-1, 2, (R, C), generator=gen, dtype=torch.int8)
    return W


def ref_pack_word(row, wi):
    word = 0
    for j in range(16):
        c = wi * 16 + j
        code = int(row[c]) + 1 if c < len(row) else 1
        word |= code << (2 * j)
    return word


def ref_update(G, W, cnt, k, T):
    """Reference TBP step on CPU tensors. Returns (W', cnt', flips). Ties broken by lowest column index."""
    G, W, cnt = G.cpu().float(), W.cpu().clone().to(torch.int64), cnt.cpu().clone().to(torch.int64)
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
            cv = int(cnt[r, c])
            sc = (cv > 0) - (cv < 0)
            cv = cv + d if sc == d else d
            if abs(cv) >= T:
                W[r, c] += 1 if cv > 0 else -1
                cv = 0
                flips += 1
            cnt[r, c] = cv
    return W.to(torch.int8), cnt.to(torch.int8), flips


# ----------------------------------------------------------------------------------------------- pack / unpack
@pytest.mark.parametrize("R,C", [(1, 1), (3, 15), (3, 16), (5, 17), (7, 100), (64, 784), (2, 4097)])
def test_pack_unpack_roundtrip(R, C):
    g = torch.Generator().manual_seed(R * 1000 + C)
    W = rand_tern(R, C, g)
    Wp = K.pack(W.to(DEV))
    assert Wp.shape == (R, (C + 15) // 16) and Wp.dtype == torch.int32
    assert torch.equal(K.unpack(Wp, C).cpu(), W)
    assert torch.equal(K.unpack(Wp, C, torch.float32).cpu(), W.float())


def test_pack_bit_layout_and_padding():
    g = torch.Generator().manual_seed(1)
    W = rand_tern(4, 37, g)
    Wp = K.pack(W.to(DEV)).cpu()
    for r in range(4):
        for wi in range(Wp.shape[1]):
            assert int(Wp[r, wi]) & 0xFFFFFFFF == ref_pack_word(W[r].tolist(), wi)


# ----------------------------------------------------------------------------------------------- matmuls
SHAPES = [(1, 1, 1), (3, 5, 7), (32, 32, 32), (33, 31, 65), (64, 10, 784), (257, 300, 129), (5, 1000, 16)]


@pytest.mark.parametrize("M,N,Kd", SHAPES)
def test_mm_nt_exact_on_integers(M, N, Kd):
    g = torch.Generator().manual_seed(M + 7 * N + 13 * Kd)
    W = rand_tern(N, Kd, g)
    A = torch.randint(-8, 9, (M, Kd), generator=g).float()
    Y = K.mm_nt(A.to(DEV), K.pack(W.to(DEV)), N, scale=1.0).cpu()
    assert torch.equal(Y, A @ W.float().T)


@pytest.mark.parametrize("M,N,Kd", SHAPES)
def test_mm_nn_exact_on_integers(M, N, Kd):
    g = torch.Generator().manual_seed(3 * M + N + Kd)
    W = rand_tern(Kd, N, g)  # [K, N]
    A = torch.randint(-8, 9, (M, Kd), generator=g).float()
    Y = K.mm_nn(A.to(DEV), K.pack(W.to(DEV)), N, scale=1.0).cpu()
    assert torch.equal(Y, A @ W.float())


@pytest.mark.parametrize("M,N,Kd", [(64, 256, 784), (256, 9830, 784), (100, 37, 2000)])
def test_mm_float_and_scale(M, N, Kd):
    g = torch.Generator().manual_seed(5)
    W = rand_tern(N, Kd, g)
    A = torch.randn(M, Kd, generator=g)
    s = 1 / math.sqrt(Kd)
    Wp = K.pack(W.to(DEV))
    torch.testing.assert_close(K.mm_nt(A.to(DEV), Wp, N, s).cpu(), (A @ W.float().T) * s, rtol=1e-5, atol=1e-4)
    E = torch.randn(M, N, generator=g)
    torch.testing.assert_close(K.mm_nn(E.to(DEV), Wp, Kd).cpu(), E @ W.float(), rtol=1e-5, atol=1e-3)


# ----------------------------------------------------------------------------------------------- update
def run_update(G, W, cnt, k, T):
    Wp = K.pack(W.to(DEV))
    c = cnt.to(DEV).contiguous()
    flips = K.update(G.to(DEV), Wp, c, k, T)
    return K.unpack(Wp, W.shape[1]).cpu(), c.cpu(), flips, Wp


def check_update(G, W, cnt, k, T):
    Wr, cr, fr = ref_update(G, W, cnt, k, T)
    Wg, cg, fg, Wp = run_update(G, W, cnt, k, T)
    assert torch.equal(Wg, Wr), f"weights differ at {torch.nonzero(Wg != Wr)[:5].tolist()}"
    assert torch.equal(cg, cr), f"counters differ at {torch.nonzero(cg != cr)[:5].tolist()}"
    assert fg == fr
    assert torch.equal(Wp, K.pack(Wg.to(DEV))), "packed padding/bits changed outside the updated weights"
    return Wg, cg, fg


@pytest.mark.parametrize("R,C,k", [(1, 1, 1), (4, 5, 2), (8, 100, 3), (16, 784, 4), (3, 4097, 20), (2, 40000, 200)])
def test_update_distinct_scores_T1(R, C, k):
    g = torch.Generator().manual_seed(R + C + k)
    W = rand_tern(R, C, g)
    G = torch.randn(R, C, generator=g)
    check_update(G, W, torch.zeros(R, C, dtype=torch.int8), k, 1)


def test_update_ties_take_lowest_index():
    g = torch.Generator().manual_seed(2)
    W = torch.zeros(6, 300, dtype=torch.int8)
    G = torch.randint(-3, 4, (6, 300), generator=g).float()  # many exact ties
    for k in (1, 7, 50, 299):
        check_update(G, W, torch.zeros(6, 300, dtype=torch.int8), k, 1)


def test_update_ties_across_thread_chunks():
    # equal scores spread over the whole row, so the tie order must follow the per-thread chunk scan
    C = 1031
    G = torch.ones(2, C)
    G[1] = -1.0
    W = torch.zeros(2, C, dtype=torch.int8)
    check_update(G, W, torch.zeros(2, C, dtype=torch.int8), 517, 1)


def test_update_k_exceeds_eligible_and_saturation():
    W = torch.tensor([[1, 1, 1, 1, 0, -1], [-1, -1, -1, -1, -1, -1]], dtype=torch.int8)
    # row 0: positive G -> step down: all movable except none; row 1: positive G but all at -1 -> nothing movable
    G = torch.tensor([[0.5, 2.0, 0.0, -1.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    Wg, cg, f = check_update(G, W, torch.zeros_like(W), 10, 1)
    assert Wg[1].tolist() == [-1] * 6 and f == 3  # row 0: cols 0, 1, 4 step down; col 3 (+1, wants up) blocked


def test_update_zero_and_nan_are_ignored():
    G = torch.tensor([[0.0, float("nan"), 1.0, -2.0, 0.0]])
    W = torch.zeros(1, 5, dtype=torch.int8)
    Wg, _, f = check_update(G, W, torch.zeros_like(W), 5, 1)
    assert f == 2 and Wg.tolist() == [[0, 0, -1, 1, 0]]


def test_update_close_values_low_bits():
    # scores that share the high bytes, so the radix select has to resolve them in the last passes
    base = 1.2345678
    vals = torch.tensor([base, base * (1 + 1e-7), base * (1 + 2e-7), base * (1 - 1e-7), base * (1 + 3e-7)],
                        dtype=torch.float32)
    G = vals.repeat(3, 1)
    W = torch.zeros(3, 5, dtype=torch.int8)
    for k in range(1, 6):
        check_update(G, W, torch.zeros(3, 5, dtype=torch.int8), k, 1)


@pytest.mark.parametrize("T", [2, 3, 8])
def test_update_counter_sequences(T):
    """Repeated steps with changing signals: counters accumulate, restart on opposite votes, flip at T."""
    g = torch.Generator().manual_seed(T)
    R, C, k = 5, 203, 9
    W = rand_tern(R, C, g)
    cnt = torch.zeros(R, C, dtype=torch.int8)
    Wr, cr = W.clone(), cnt.clone()
    Wp = K.pack(W.to(DEV))
    cg = cnt.to(DEV)
    total = 0
    for step in range(40):
        G = torch.randn(R, C, generator=g)
        if step % 3 == 0:  # persistent direction for a block of columns so counters can reach T
            G[:, :40] = torch.linspace(5, 6, 40)
        Wr, cr, fr = ref_update(G, Wr, cr, k, T)
        fg = K.update(G.to(DEV), Wp, cg, k, T)
        assert fg == fr
        assert torch.equal(K.unpack(Wp, C).cpu(), Wr) and torch.equal(cg.cpu(), cr)
        total += fg
    assert total > 0


def test_update_counter_restart_and_flip_explicit():
    W = torch.zeros(1, 2, dtype=torch.int8)
    cnt = torch.tensor([[2, -1]], dtype=torch.int8)
    # col 0: G<0 -> vote +1, counter 2 -> 3 -> reaches T=3 -> flip 0->+1, reset; col 1: vote +1 opposite -> restart to +1
    G = torch.tensor([[-1.0, -0.5]])
    Wg, cg, f = check_update(G, W, cnt, 2, 3)
    assert Wg.tolist() == [[1, 0]] and cg.tolist() == [[0, 1]] and f == 1


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("R,C,k,T", [(4, 5, 2, 1), (16, 784, 4, 1), (8, 1786, 9, 2), (3, 10240, 51, 1)])
def test_update_shared_cache_and_global_paths(cache, R, C, k, T):
    g = torch.Generator().manual_seed(C + k)
    W = rand_tern(R, C, g)
    G = torch.randint(-50, 51, (R, C), generator=g).float() / 7  # ties and distinct values mixed
    cnt = torch.randint(-1, 2, (R, C), generator=g).to(torch.int8)
    Wr, cr, fr = ref_update(G, W, cnt, k, T)
    Wp = K.pack(W.to(DEV))
    c = cnt.to(DEV).contiguous()
    fg = K.update(G.to(DEV), Wp, c, k, T, cache=cache)
    assert torch.equal(K.unpack(Wp, C).cpu(), Wr) and torch.equal(c.cpu(), cr) and fg == fr
