"""GPU TBP trainer vs an independent CPU implementation of the same step."""

import math

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

if torch.cuda.is_available():
    from tbp import tbp as TBP
    from tbp.tests.test_kernels import ref_update


def cpu_step(Ws, cnts, x, y, k_frac, T, tau=2.0):
    """Reference TBP step in float64 on the CPU. Returns new (Ws, cnts)."""
    Wf = [W.double() for W in Ws]
    h, hs, pres = x.double(), [x.double()], []
    for W in Wf[:-1]:
        pre = h @ W.T / math.sqrt(W.shape[1])
        pre = (pre - pre.mean(1, keepdim=True)) / pre.std(1, keepdim=True).clamp_min(1e-5)
        h = torch.relu(pre)
        pres.append(pre)
        hs.append(h)
    logits = h @ Wf[-1].T / math.sqrt(Wf[-1].shape[1])
    E = torch.softmax(tau * logits, 1) - F.one_hot(y, logits.shape[1]).double()
    errs = [None] * len(Ws)
    errs[-1] = E
    for l in range(len(Ws) - 2, -1, -1):
        errs[l] = (errs[l + 1] @ Wf[l + 1]) * (pres[l] > 0).double()
    out_W, out_c = [], []
    for l, W in enumerate(Ws):
        G = (errs[l].T @ hs[l]).float()
        n = max(1, int(round(k_frac * W.shape[1])))
        Wn, cn, _ = ref_update(G, W, cnts[l], n, T)
        out_W.append(Wn)
        out_c.append(cn)
    return out_W, out_c


@pytest.mark.parametrize("dims,k_frac,T", [([64, 50, 10], 0.05, 1), ([64, 96, 40, 10], 0.03, 2),
                                           ([100, 300, 10], 0.01, 1)])
def test_steps_match_cpu_reference(dims, k_frac, T):
    torch.manual_seed(0)
    Ws = [torch.randint(-1, 2, (dims[i + 1], dims[i]), dtype=torch.int8) for i in range(len(dims) - 1)]
    cnts = [torch.zeros_like(W) for W in Ws]
    net = TBP.TBPNet(Ws, counter_bits=8)
    mismatches = total = 0
    for step in range(6):
        x = torch.rand(48, dims[0])
        y = torch.randint(0, dims[-1], (48,))
        Ws, cnts = cpu_step(Ws, cnts, x, y, k_frac, T)
        net.step(x.cuda(), y.cuda(), k_frac, T, "softmax")
        for l, Wg in enumerate(net.weights()):
            mismatches += (Wg.cpu() != Ws[l]).sum().item() + (net.cnt[l].cpu() != cnts[l]).sum().item()
            total += 2 * Ws[l].numel()
        # resync to the reference so float-rounding near-ties cannot compound across steps
        net = TBP.TBPNet(Ws, counter_bits=8)
        net.cnt = [c.cuda() for c in cnts]
    # GPU sums are fp32 in a different order: only exact near-ties at the top-k boundary may differ
    assert mismatches / total < 1e-3, f"{mismatches}/{total} entries differ"


def test_packed_and_cublas_forward_agree():
    torch.manual_seed(1)
    dims = [784, 333, 129, 10]
    Ws = [torch.randint(-1, 2, (dims[i + 1], dims[i]), dtype=torch.int8) for i in range(3)]
    x = torch.rand(77, 784).cuda()
    a = TBP.TBPNet(Ws, matmul="packed").forward(x)[2]
    b = TBP.TBPNet(Ws, matmul="cublas").forward(x)[2]
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_learns_digits():
    hist, _ = TBP.run_tbp("digits", depth=1, width=512, epochs=8, bs=64, seed=0, k_frac=0.005, T=4)
    assert hist[-1]["test"] > 0.85, hist[-1]


def test_auto_mode_matches_packed_step():
    torch.manual_seed(3)
    dims = [784, 300, 200, 10]
    Ws = [torch.randint(-1, 2, (dims[i + 1], dims[i]), dtype=torch.int8) for i in range(3)]
    x, y = torch.rand(64, 784).cuda(), torch.randint(0, 10, (64,)).cuda()
    a, b = TBP.TBPNet(Ws, matmul="packed"), TBP.TBPNet(Ws, matmul="auto", batch=64)
    b.fwd_mode, b.bwd_mode = ["cublas", "packed", "cublas"], ["packed", "cublas", "packed"]  # force a mix
    a.step(x, y, 0.02, 1)
    b.step(x, y, 0.02, 1)
    diff = sum((wa != wb).sum().item() for wa, wb in zip(a.weights(), b.weights()))
    assert diff <= 2, diff  # fp32 summation order may flip an exact near-tie


def test_logits_matches_forward():
    torch.manual_seed(4)
    dims = [784, 129, 129, 10]
    net = TBP.TBPNet([torch.randint(-1, 2, (dims[i + 1], dims[i]), dtype=torch.int8) for i in range(3)])
    x = torch.rand(50, 784).cuda()
    torch.testing.assert_close(net.logits(x), net.forward(x)[2])


@pytest.mark.parametrize("T", [2, 4])
def test_trainer_3bit_counters_equal_int8(T):
    torch.manual_seed(5)
    dims = [784, 200, 150, 10]
    Ws = [torch.randint(-1, 2, (dims[i + 1], dims[i]), dtype=torch.int8) for i in range(3)]
    a = TBP.TBPNet(Ws, counter_bits=8)
    b = TBP.TBPNet(Ws, counter_bits=3)
    for _ in range(15):
        x, y = torch.rand(64, 784).cuda(), torch.randint(0, 10, (64,)).cuda()
        assert a.step(x, y, 0.02, T) == b.step(x, y, 0.02, T)
    for l in range(3):
        assert torch.equal(a.Wp[l], b.Wp[l])
        assert torch.equal(a.cnt[l], TBP.K.unpack_counters3(b.cnt[l], dims[l]))
    assert b.state_bytes() < a.state_bytes()


def test_default_is_4bit_and_rejects_large_T():
    Ws = [torch.randint(-1, 2, (20, 30), dtype=torch.int8), torch.randint(-1, 2, (10, 20), dtype=torch.int8)]
    net = TBP.TBPNet(Ws)
    assert net.counter_bits == 4 and net.cnt[0].dtype == torch.int32
    x, y = torch.rand(8, 30).cuda(), torch.randint(0, 10, (8,)).cuda()
    net.step(x, y, 0.1, 8)  # T = 8 fits 4-bit counters
    with pytest.raises(ValueError, match="counter_bits=8"):
        net.step(x, y, 0.1, 16)
    TBP.TBPNet(Ws, counter_bits=8).step(x, y, 0.1, 16)  # int8 counters allow larger T
