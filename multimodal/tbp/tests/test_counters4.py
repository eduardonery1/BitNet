"""4-bit packed flip counters (8 per word, |c| <= 7, T <= 8)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

if torch.cuda.is_available():
    from tbp import kernels as K
    from tbp import tbp as TBP
    from tbp.tests.test_counters3 import ref_update

DEV = "cuda"


@pytest.mark.parametrize("R,C", [(1, 1), (2, 7), (3, 8), (4, 9), (5, 784), (2, 4099)])
def test_pack_counters4_roundtrip_and_layout(R, C):
    g = torch.Generator().manual_seed(C)
    c = torch.randint(-7, 8, (R, C), generator=g, dtype=torch.int8).to(DEV)
    Cp = K.pack_counters(c, 4)
    assert Cp.shape == (R, (C + 7) // 8)
    assert torch.equal(K.unpack_counters(Cp, C, 4), c)
    w0 = int(Cp[0, 0]) & 0xFFFFFFFF
    for j in range(min(8, C)):
        assert ((w0 >> (4 * j)) & 15) - 7 == int(c[0, j])


def test_pack_counters4_high_field_sign_bit():
    c = torch.full((1, 8), 7, dtype=torch.int8, device=DEV)  # all codes 14 -> top bit of the word set
    Cp = K.pack_counters(c, 4)
    assert int(Cp[0, 0]) < 0 and torch.equal(K.unpack_counters(Cp, 8, 4), c)


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("R,C,k,T", [(3, 7, 3, 1), (6, 101, 5, 5), (8, 784, 4, 8), (4, 1786, 9, 7), (2, 12000, 60, 8)])
def test_update4_matches_reference(cache, R, C, k, T):
    g = torch.Generator().manual_seed(R * C + T)
    W = torch.randint(-1, 2, (R, C), generator=g, dtype=torch.int8)
    G = torch.randint(-40, 41, (R, C), generator=g).float() / 9
    cnt = torch.randint(-(T - 1), T, (R, C), generator=g, dtype=torch.int8)
    Wr, cr, fr = ref_update(G, W, cnt, k, T)
    Wp = K.pack(W.to(DEV))
    Cp = K.pack_counters(cnt.to(DEV), 4)
    fg = K.update(G.to(DEV), Wp, Cp, k, T, cache=cache, counter_bits=4)
    assert fg == fr and torch.equal(K.unpack(Wp, C).cpu(), Wr) and torch.equal(K.unpack_counters(Cp, C, 4).cpu(), cr)


@pytest.mark.parametrize("T", [1, 4, 8])
def test_4bit_equals_int8_over_many_steps(T):
    g = torch.Generator().manual_seed(T)
    R, C, k = 6, 333, 7
    W = torch.randint(-1, 2, (R, C), generator=g, dtype=torch.int8)
    Wp8, Wp4 = K.pack(W.to(DEV)), K.pack(W.to(DEV))
    c8 = torch.zeros(R, C, dtype=torch.int8, device=DEV)
    c4 = K.pack_counters(c8, 4)
    for step in range(40):
        G = torch.randn(R, C, generator=g)
        G[:, :50] = 3.0 + torch.rand(50, generator=g)
        assert K.update(G.to(DEV), Wp8, c8, k, T) == K.update(G.to(DEV), Wp4, c4, k, T, counter_bits=4)
        assert torch.equal(Wp8, Wp4) and torch.equal(K.unpack_counters(c4, C, 4), c8)


def test_4bit_rejects_T_above_8():
    Wp = K.pack(torch.zeros(1, 8, dtype=torch.int8, device=DEV))
    Cp = K.pack_counters(torch.zeros(1, 8, dtype=torch.int8, device=DEV), 4)
    with pytest.raises(AssertionError):
        K.update(torch.ones(1, 8, device=DEV), Wp, Cp, 1, 9, counter_bits=4)


def test_trainer_4bit_equals_int8_T8():
    torch.manual_seed(6)
    dims = [784, 200, 150, 10]
    Ws = [torch.randint(-1, 2, (dims[i + 1], dims[i]), dtype=torch.int8) for i in range(3)]
    a, b = TBP.TBPNet(Ws, counter_bits=8), TBP.TBPNet(Ws, counter_bits=4)
    for _ in range(25):
        x, y = torch.rand(64, 784).cuda(), torch.randint(0, 10, (64,)).cuda()
        assert a.step(x, y, 0.02, 8) == b.step(x, y, 0.02, 8)
    for l in range(3):
        assert torch.equal(a.Wp[l], b.Wp[l]) and torch.equal(a.cnt[l], K.unpack_counters(b.cnt[l], dims[l], 4))
