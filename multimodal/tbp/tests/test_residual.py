"""Residual TBP: forward vs float reference, one step vs an independent CPU step, and learning."""

import math

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

if torch.cuda.is_available():
    from tbp import tbp as TBP
    from tbp.tests.test_kernels import ref_update


def ln(p):
    return (p - p.mean(1, keepdim=True)) / p.std(1, keepdim=True).clamp_min(1e-5)


def cpu_res_step(Ws, cnts, x, y, k_frac, T, tau=2.0):
    """Independent float64 residual TBP step (same rule as docs: skip + scaled branch error)."""
    Wf = [W.double() for W in Ws]
    L = len(Ws)
    s = lambda l: math.sqrt(Wf[l].shape[1])  # noqa: E731
    raw = x.double() @ Wf[0].T / s(0)
    sig = [raw.std(1, keepdim=True).clamp_min(1e-5)]
    n = [(raw - raw.mean(1, keepdim=True)) / sig[0]]
    h = torch.relu(n[0])
    inputs = [x.double()]
    for l in range(1, L - 1):
        inputs.append(h)
        raw = h @ Wf[l].T / s(l)
        sig.append(raw.std(1, keepdim=True).clamp_min(1e-5))
        n.append((raw - raw.mean(1, keepdim=True)) / sig[-1])
        h = h + torch.relu(n[-1])
    sf = h.std(1, keepdim=True).clamp_min(1e-5)
    z = (h - h.mean(1, keepdim=True)) / sf
    inputs.append(z)
    logits = z @ Wf[-1].T / s(L - 1)
    E = torch.softmax(tau * logits, 1) - F.one_hot(y, logits.shape[1]).double()
    errs = [None] * L
    errs[-1] = E
    e = (E @ Wf[-1]) / s(L - 1) / sf
    for l in range(L - 2, 0, -1):
        loc = e * (n[l] > 0).double()
        errs[l] = loc
        e = e + ((loc / sig[l]) @ Wf[l]) / s(l)
    errs[0] = e * (n[0] > 0).double()
    out_W, out_c = [], []
    for l, W in enumerate(Ws):
        G = (errs[l].T @ inputs[l]).float()
        Wn, cn, _ = ref_update(G, W, cnts[l], max(1, int(round(k_frac * W.shape[1]))), T)
        out_W.append(Wn)
        out_c.append(cn)
    return out_W, out_c, logits


def make(dims, seed):
    torch.manual_seed(seed)
    return [torch.randint(-1, 2, (dims[i + 1], dims[i]), dtype=torch.int8) for i in range(len(dims) - 1)]


def test_residual_forward_matches_reference():
    dims = [784, 256, 256, 256, 10]
    Ws = make(dims, 0)
    x = torch.rand(40, 784)
    net = TBP.TBPNet(Ws, residual=True)
    ref = TBP.residual_forward([W.double() for W in Ws], x.double()).float()
    torch.testing.assert_close(net.logits(x.cuda()).cpu(), ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(net.forward(x.cuda())[2], net.logits(x.cuda()))


@pytest.mark.parametrize("dims,k_frac,T", [([64, 48, 48, 10], 0.05, 1), ([100, 64, 64, 64, 64, 10], 0.03, 2)])
def test_residual_steps_match_cpu_reference(dims, k_frac, T):
    Ws = make(dims, 1)
    cnts = [torch.zeros_like(W) for W in Ws]
    mism = total = 0
    for step in range(6):
        net = TBP.TBPNet(Ws, counter_bits=8, residual=True)
        net.cnt = [c.cuda() for c in cnts]
        x, y = torch.rand(32, dims[0]), torch.randint(0, dims[-1], (32,))
        Ws, cnts, _ = cpu_res_step(Ws, cnts, x, y, k_frac, T)
        net.step(x.cuda(), y.cuda(), k_frac, T)
        for l, Wg in enumerate(net.weights()):
            mism += (Wg.cpu() != Ws[l]).sum().item() + (net.cnt[l].cpu() != cnts[l]).sum().item()
            total += 2 * Ws[l].numel()
    assert mism == 0, f"{mism}/{total}"


def test_residual_requires_equal_widths():
    with pytest.raises(AssertionError):
        TBP.TBPNet(make([64, 32, 48, 10], 2), residual=True)


def test_residual_learns_digits():
    hist, _ = TBP.run_tbp("digits", depth=4, width=256, epochs=8, bs=64, seed=0, k_frac=0.01, T=4, residual=True)
    assert hist[-1]["test"] > 0.85, hist[-1]
