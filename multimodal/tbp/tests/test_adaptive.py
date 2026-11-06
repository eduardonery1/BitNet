"""Per-row k (fractional k), learnable per-layer scales, and the validation-driven k controller."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

if torch.cuda.is_available():
    from tbp import kernels as K
    from tbp.tests.test_counters3 import ref_update

DEV = "cuda"


def ref_update_rows(G, W, cnt, krows, T):
    """ref_update applied row by row with each row's own k."""
    Ws, cs, f = [], [], 0
    for r in range(G.shape[0]):
        w, c, fr = ref_update(G[r:r + 1], W[r:r + 1], cnt[r:r + 1], int(krows[r]), T)
        Ws.append(w)
        cs.append(c)
        f += fr
    return torch.cat(Ws), torch.cat(cs), f


@pytest.mark.parametrize("bits", [8, 4, 3])
@pytest.mark.parametrize("cache", [True, False])
def test_krows_matches_reference(bits, cache):
    g = torch.Generator().manual_seed(bits + 10 * cache)
    R, C, T = 12, 333, 2 if bits == 3 else 4
    W = torch.randint(-1, 2, (R, C), generator=g, dtype=torch.int8)
    G = torch.randint(-30, 31, (R, C), generator=g).float() / 7
    cnt = torch.randint(-(T - 1), T, (R, C), generator=g, dtype=torch.int8)
    krows = torch.tensor([0, 1, 2, 5, 0, 400, 3, 1, 0, 7, 333, 2], dtype=torch.int32)  # 0, huge, mixed
    Wr, cr, fr = ref_update_rows(G, W, cnt, krows, T)
    Wp = K.pack(W.to(DEV))
    cdev = cnt.to(DEV) if bits == 8 else K.pack_counters(cnt.to(DEV), bits)
    fg = K.update(G.to(DEV), Wp, cdev, 1, T, cache=cache, counter_bits=None if bits == 8 else bits,
                  krows=krows.to(DEV))
    cg = cdev if bits == 8 else K.unpack_counters(cdev, C, bits)
    assert fg == fr and torch.equal(K.unpack(Wp, C).cpu(), Wr) and torch.equal(cg.cpu(), cr)


def test_krows_none_unchanged():
    g = torch.Generator().manual_seed(0)
    W = torch.randint(-1, 2, (6, 100), generator=g, dtype=torch.int8)
    G = torch.randn(6, 100, generator=g)
    a, b = K.pack(W.to(DEV)), K.pack(W.to(DEV))
    ca, cb = torch.zeros(6, 100, dtype=torch.int8, device=DEV), torch.zeros(6, 100, dtype=torch.int8, device=DEV)
    K.update(G.to(DEV), a, ca, 4, 1)
    K.update(G.to(DEV), b, cb, 4, 1, krows=torch.full((6,), 4, dtype=torch.int32, device=DEV))
    assert torch.equal(a, b) and torch.equal(ca, cb)


@pytest.mark.parametrize("k_frac,fan_in", [(0.0002, 1024), (0.0005, 2560), (0.003, 784), (0.0101, 100)])
def test_sample_krows_mean(k_frac, fan_in):
    gen = torch.Generator(device=DEV).manual_seed(1)
    kr = torch.stack([K.sample_krows(k_frac, 4096, fan_in, gen=gen) for _ in range(50)]).float()
    e = k_frac * fan_in
    assert abs(kr.mean().item() - e) / e < 0.02
    assert set(kr.unique().tolist()) <= {math.floor(e), math.floor(e) + 1}


# ------------------------------------------------------------------------------------------- learnable scales
import torch.nn.functional as F  # noqa: E402

if torch.cuda.is_available():
    from tbp import tbp as TBP


def make(dims, seed):
    torch.manual_seed(seed)
    return [torch.randint(-1, 2, (dims[i + 1], dims[i]), dtype=torch.int8) for i in range(len(dims) - 1)]


def ln_detached(p):
    """LayerNorm with its statistics treated as constants (the approximation TBP's backward uses)."""
    mu = p.mean(1, keepdim=True).detach()
    sd = p.std(1, keepdim=True).clamp_min(1e-5).detach()
    return (p - mu) / sd


def ref_residual(Ws, x, log_s, ns_out):
    Wf = [W.double() for W in Ws]
    s = log_s.exp()
    n = ln_detached(x @ Wf[0].T / math.sqrt(Wf[0].shape[1])).requires_grad_(True)  # leaf: no grad path above it
    ns_out.append(n)
    h = torch.relu(n) * s[0]
    for l in range(1, len(Wf) - 1):
        n = ln_detached(h @ Wf[l].T / math.sqrt(Wf[l].shape[1]))
        n.retain_grad()
        ns_out.append(n)
        h = h + torch.relu(n) * s[l]
    return (ln_detached(h) @ Wf[-1].T / math.sqrt(Wf[-1].shape[1])) * s[-1]


def ref_plain(Ws, x, log_s):
    Wf = [W.double() for W in Ws]
    s = log_s.exp()
    h = x
    for l in range(len(Wf) - 1):
        p = h @ Wf[l].T / math.sqrt(Wf[l].shape[1])
        p = (p - p.mean(1, keepdim=True)) / p.std(1, keepdim=True).clamp_min(1e-5)  # exact LN
        h = torch.relu(p) * s[l]
    return (h @ Wf[-1].T / math.sqrt(Wf[-1].shape[1])) * s[-1]


def init_scales(net, vals):
    with torch.no_grad():
        net.log_s.copy_(torch.tensor(vals, device=DEV).log())


@pytest.mark.parametrize("depth", [1, 3, 6])
def test_residual_scale_grads_and_errors_match_autograd(depth):
    dims = [64] + [48] * depth + [10]
    Ws = make(dims, depth)
    tau, B = 4.0, 32
    vals = [0.7 + 0.1 * i for i in range(len(Ws))]
    net = TBP.TBPNet(Ws, counter_bits=8, residual=True, learn_scale=True, scale_lr=0.0)
    init_scales(net, vals)
    x, y = torch.rand(B, 64), torch.randint(0, 10, (B,))
    log_s = torch.tensor(vals, dtype=torch.float64).log().requires_grad_(True)
    ns = []
    loss = F.cross_entropy(tau * ref_residual(Ws, x.double(), log_s, ns), y)
    loss.backward()
    pres, hs, logits = net.forward(x.cuda())
    E = torch.softmax(tau * logits, 1) - F.one_hot(y.cuda(), 10).float()
    errs, gs = net._residual_errors(E, pres)
    g_log = (gs * (tau / B) * net.scales()).double().cpu()
    torch.testing.assert_close(g_log, log_s.grad, rtol=1e-4, atol=1e-6)
    for l in range(len(Ws) - 1):  # vote errors = dL/dn_l in units of (p - y)
        torch.testing.assert_close(errs[l].double().cpu(), ns[l].grad * (B / tau), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("depth", [1, 3])
def test_plain_scale_grads_match_autograd(depth):
    dims = [64] + [40] * depth + [10]
    Ws = make(dims, 10 + depth)
    tau, B = 4.0, 32
    vals = [0.8 + 0.15 * i for i in range(len(Ws))]
    net = TBP.TBPNet(Ws, counter_bits=8, learn_scale=True, scale_lr=0.0)
    init_scales(net, vals)
    x, y = torch.rand(B, 64), torch.randint(0, 10, (B,))
    log_s = torch.tensor(vals, dtype=torch.float64).log().requires_grad_(True)
    F.cross_entropy(tau * ref_plain(Ws, x.double(), log_s), y).backward()
    pres, hs, logits = net.forward(x.cuda())
    E = torch.softmax(tau * logits, 1) - F.one_hot(y.cuda(), 10).float()
    _, gs = net._plain_errors(E, pres)
    g_log = (gs * (tau / B) * net.scales() * net.scale_mask).double().cpu()
    want = log_s.grad * net.scale_mask.double().cpu()  # earlier hidden scales are cancelled by the next LN: not trained
    torch.testing.assert_close(g_log, want, rtol=1e-4, atol=1e-6)
    hidden = log_s.grad[:len(Ws) - 2].abs().max().item() if len(Ws) > 2 else 0.0
    assert hidden < 1e-6  # confirms the true gradient of those scales is ~0


def test_learn_scale_off_is_identical_and_on_moves_scales():
    dims = [64, 50, 50, 10]
    Ws = make(dims, 3)
    a = TBP.TBPNet(Ws, counter_bits=8, residual=True)
    b = TBP.TBPNet(Ws, counter_bits=8, residual=True, learn_scale=True, scale_lr=0.05)
    x, y = torch.rand(32, 64).cuda(), torch.randint(0, 10, (32,)).cuda()
    a2 = TBP.TBPNet(Ws, counter_bits=8, residual=True)
    for _ in range(3):
        a.step(x, y, 0.05, 1)
        a2.step(x, y, 0.05, 1)
        b.step(x, y, 0.05, 1)
    assert all(torch.equal(p, q) for p, q in zip(a.Wp, a2.Wp))
    assert (b.scales() - 1).abs().max() > 1e-3


# ------------------------------------------------------------------------------------------- k controller
def test_kcontroller_grow_shrink_bounds():
    c = TBP.KController(0.01, k_min=0.002, k_max=0.015, tol=0.01, patience=2, grow_patience=3)
    ks = [c.update(v) for v in [1.0, 0.9, 0.8]]  # 3 improvements -> grow
    assert ks == [0.01, 0.01, 0.0125]
    assert c.update(0.7) == 0.0125 and c.update(0.6) == 0.0125 and c.update(0.5) == 0.015  # grow again, capped
    assert c.update(0.5) == 0.015  # 1 miss
    assert c.update(0.499) == 0.0075  # 2nd miss (within tol) -> shrink
    for _ in range(10):
        c.update(9.0)
    assert c.k == 0.002  # floor


def test_kcontroller_miss_resets_improvement_streak():
    c = TBP.KController(0.01, tol=0.0, patience=5, grow_patience=3)
    for v in [1.0, 0.9, 0.95, 0.85, 0.8]:  # streak broken by 0.95
        c.update(v)
    assert c.k == 0.01
    c.update(0.7)
    assert c.k == 0.0125


def test_adaptive_and_scale_learn_digits():
    hist, net = TBP.run_tbp("digits", depth=4, width=256, epochs=8, bs=64, seed=0, k_frac=0.01, T=4, residual=True,
                            learn_scale=True, adaptive_k=True, val_split=200)
    assert hist[-1]["test"] > 0.85, hist[-1]
    assert len(hist[-1]["k_trace"]) > 0 and "scales" in hist[-1]


# ------------------------------------------------------------------------------------------- CPU backend
def test_cpu_backend_update_matches_reference():
    from tbp import cpu_backend as C
    g = torch.Generator().manual_seed(5)
    R, Cc, T = 10, 300, 3
    W = torch.randint(-1, 2, (R, Cc), generator=g, dtype=torch.int8)
    G = torch.randn(R, Cc, generator=g)  # distinct scores: top-k order is unambiguous
    cnt = torch.randint(-(T - 1), T, (R, Cc), generator=g, dtype=torch.int8)
    krows = torch.tensor([0, 1, 2, 5, 9, 300, 3, 1, 0, 7], dtype=torch.int32)
    Wr, cr, fr = ref_update_rows(G, W, cnt, krows, T)
    Wc, cc = W.clone(), cnt.clone()
    assert C.update(G, Wc, cc, 1, T, krows=krows) == fr
    assert torch.equal(Wc, Wr) and torch.equal(cc, cr)
    Wr, cr, fr = ref_update(G, W, cnt, 4, T)
    Wc, cc = W.clone(), cnt.clone()
    assert C.update(G, Wc, cc, 4, T) == fr and torch.equal(Wc, Wr) and torch.equal(cc, cr)


@pytest.mark.parametrize("residual,learn", [(False, False), (True, True)])
def test_cpu_and_gpu_nets_agree(residual, learn):
    dims = [64, 48, 48, 10]
    Ws = make(dims, 7)
    a = TBP.TBPNet(Ws, device="cpu", counter_bits=8, residual=residual, learn_scale=learn, scale_lr=0.01)
    b = TBP.TBPNet(Ws, device="cuda", matmul="cublas", counter_bits=8, residual=residual, learn_scale=learn,
                   scale_lr=0.01)
    torch.manual_seed(0)
    diff = total = 0
    for _ in range(5):
        x, y = torch.rand(32, 64), torch.randint(0, 10, (32,))
        a.step(x, y, 0.05, 2)
        b.step(x.cuda(), y.cuda(), 0.05, 2)
        for wa, wb in zip(a.weights(), b.weights()):
            diff += int((wa != wb.cpu()).sum())
            total += wa.numel()
    assert diff / total < 1e-3, f"{diff}/{total}"  # fp32 summation order can flip a near-tie
    if learn:
        torch.testing.assert_close(a.scales(), b.scales().cpu(), rtol=1e-3, atol=1e-4)


# ------------------------------------------------------------------------------------------- stability options
@pytest.mark.parametrize("bits", [8, 4, 3])
def test_counter_reset_zeroes_packed_counters(bits):
    Ws = make([64, 40, 10], 1)
    net = TBP.TBPNet(Ws, counter_bits=bits, counter_reset=3)
    x, y = torch.rand(32, 64).cuda(), torch.randint(0, 10, (32,)).cuda()
    T = 4 if bits == 3 else 8
    for i in range(1, 7):
        net.step(x, y, 0.05, T)
        cnts = [c if bits == 8 else K.unpack_counters(c, net.dims[l], bits) for l, c in enumerate(net.cnt)]
        nonzero = sum(int((c != 0).sum()) for c in cnts)
        if i % 3 == 0:
            assert nonzero == 0, f"step {i}: {nonzero} counters left after reset"
        elif i % 3 == 1:
            assert nonzero > 0


def test_sig_threshold_only_picks_above_mean():
    Ws = make([64, 30, 10], 2)
    a = TBP.TBPNet(Ws, counter_bits=8, sig_z=1.0)
    x, y = torch.rand(32, 64).cuda(), torch.randint(0, 10, (32,)).cuda()
    pres, hs, logits = a.forward(x)
    E = torch.softmax(2 * logits, 1) - F.one_hot(y, 10).float()  # default tau = 2
    errs, _ = a._plain_errors(E, pres)
    G = [(errs[l].T @ hs[l]).cpu() for l in range(2)]
    a.step(x, y, 1.0, 1)  # k = whole row: every eligible weight is picked (T = 1 flips immediately)
    for l in range(2):
        changed = (a.weights()[l].cpu() != Ws[l])
        A = G[l].abs()
        above = A > A.mean(1, keepdim=True)
        assert not (changed & ~above).any()  # nothing below the row mean moved
        assert changed.any()


def test_binary_init_has_no_zeros():
    Ws = TBP.init_weights([64, 100, 10], 0, "binary")
    assert all(set(W.unique().tolist()) == {-1, 1} for W in Ws)
    Wt = TBP.init_weights([64, 100, 10], 0, "ternary")
    assert all(set(W.unique().tolist()) == {-1, 0, 1} for W in Wt)


def test_tcontroller_rules():
    c = TBP.TController(8, t_min=2, t_max=10, tol=0.01)
    assert c.update(1.0) == 8          # first check: no comparison
    assert c.update(0.9) == 8          # decreasing -> keep
    assert c.update(0.95) == 9         # increasing -> T + 1
    assert c.update(0.951) == 8        # stalled (within 1%) -> T - 1
    for _ in range(10):
        c.update(c.prev * 1.5)         # keeps increasing
    assert c.T == 10                   # capped
    for _ in range(20):
        c.update(c.prev)               # stalled
    assert c.T == 2                    # floor


def test_t_schedule_runs_and_changes_T():
    hist, net = TBP.run_tbp("digits", depth=2, width=128, epochs=6, bs=64, seed=0, k_frac=0.01, T=8,
                            t_schedule=True, val_split=200, counter_bits=8, val_every=5, tau=4.0)
    Ts = {t["T"] for t in hist[-1]["k_trace"]}
    assert len(hist[-1]["k_trace"]) > 0 and len(Ts) > 1
    assert hist[-1]["test"] > 0.7


# ------------------------------------------------------------------------------------------- instrumentation
def test_instrument_counts_flips_and_reversals():
    Ws = make([64, 40, 10], 3)
    net = TBP.TBPNet(Ws, counter_bits=8, instrument=True, rev_window=100, agree_every=1)
    plain = TBP.TBPNet(Ws, counter_bits=8)
    x, y = torch.rand(32, 64).cuda(), torch.randint(0, 10, (32,)).cuda()
    total = 0
    for i in range(8):
        y2 = y if i % 2 == 0 else (y + 1) % 10  # alternate targets so some flips get undone
        total += net.step(x, y2, 0.1, 1)
        plain.step(x, y2, 0.1, 1)
    st = net.stats(T=1)
    assert sum(st["flips"]) == total > 0           # tracked flips = kernel-reported flips
    assert 0 < st["reversal_rate_all"] < 1         # alternating targets cause reversals
    assert all(0.0 <= a <= 1.0 for a in st["agreement"])
    assert all(torch.equal(a, b) for a, b in zip(net.Wp, plain.Wp))  # instrumentation does not change training


def test_reversal_rate_definition():
    net = TBP.TBPNet(make([4, 3, 2], 0), counter_bits=8, instrument=True, rev_window=5)
    l = 0
    before = net.K.unpack(net.Wp[l], 4).clone()
    w = before.clone()
    w[0, 0] = 1 if w[0, 0] < 1 else 0
    net.Wp[l].copy_(net.K.pack(w))
    net._track_flips(l, before)       # flip up at step 0
    net.n_steps = 3
    before2 = w.clone()
    w[0, 0] = before[0, 0]
    net.Wp[l].copy_(net.K.pack(w))
    net._track_flips(l, before2)      # flip back at step 3, within the window -> reversal
    st = net.stats()
    assert st["flips"][0] == 2 and st["reversal_rate"][0] == 0.5


# ------------------------------------------------------------------------------------------- step-4 mechanisms
def _run(Ws, **kw):
    net = TBP.TBPNet(Ws, counter_bits=8, **kw)
    torch.manual_seed(0)
    for i in range(12):
        x, y = torch.rand(32, 64).cuda(), torch.randint(0, 10, (32,)).cuda()
        net.step(x, y if i % 2 == 0 else (y + 3) % 10, 0.1, 2)
    return net


def test_defaults_unchanged_by_new_options():
    Ws = make([64, 40, 10], 4)
    a = _run(Ws)
    b = _run(Ws, counter_decay=0, cooldown=0, vote_strength=0.0, stagger=1, flip_every=1)
    assert all(torch.equal(p, q) for p, q in zip(a.Wp, b.Wp))


def test_counter_decay_moves_toward_zero():
    net = TBP.TBPNet(make([8, 6, 4], 0), counter_bits=4, counter_decay=1)
    net._set_counters(0, torch.tensor([[3, -2, 0, 1, -1, 7, -7, 0]] * 6, dtype=torch.int8, device=DEV))
    x, y = torch.zeros(4, 8).cuda(), torch.zeros(4, dtype=torch.long).cuda()  # zero input: no votes on layer 0
    net.step(x, y, 0.1, 8)
    assert net._counters_unpacked(0)[0].tolist() == [2, -1, 0, 0, 0, 6, -6, 0]


def test_cooldown_blocks_quick_reversals():
    Ws = make([64, 40, 10], 5)
    net = _run(Ws, cooldown=1000, instrument=True, rev_window=1000)
    assert net.stats()["reversal_rate_all"] == 0.0  # no weight flipped back within the cooldown
    free = _run(Ws, instrument=True, rev_window=1000)
    assert free.stats()["reversal_rate_all"] > 0.0  # without cooldown the alternating targets do cause reversals


def test_stagger_only_updates_one_group():
    Ws = make([64, 40, 30, 10], 6)
    net = TBP.TBPNet(Ws, counter_bits=8, stagger=3)
    before = [w.clone() for w in net.weights()]
    x, y = torch.rand(32, 64).cuda(), torch.randint(0, 10, (32,)).cuda()
    net.step(x, y, 0.5, 1)  # step 0: only layer 0 ((0 + l) % 3 == 0)
    changed = [bool((a != b).any()) for a, b in zip(net.weights(), before)]
    assert changed[0] and not changed[1] and not changed[2]


def test_flip_every_defers_flips():
    Ws = make([64, 40, 10], 7)
    net = TBP.TBPNet(Ws, counter_bits=8, flip_every=4)
    x, y = torch.rand(32, 64).cuda(), torch.randint(0, 10, (32,)).cuda()
    assert net.step(x, y, 0.2, 1) > 0      # step 0 is a flip round
    assert net.step(x, y, 0.2, 1) == 0     # steps 1-3 only vote
    assert net.step(x, y, 0.2, 1) == 0


def test_vote_strength_adds_votes():
    Ws = make([64, 40, 10], 8)
    a = TBP.TBPNet(Ws, counter_bits=8)
    b = TBP.TBPNet(Ws, counter_bits=8, vote_strength=1.0)
    x, y = torch.rand(32, 64).cuda(), torch.randint(0, 10, (32,)).cuda()
    a.step(x, y, 0.05, 4)
    b.step(x, y, 0.05, 4)
    sa = sum(int(a._counters_unpacked(l).abs().sum()) for l in range(2))
    sb = sum(int(b._counters_unpacked(l).abs().sum()) for l in range(2))
    assert sb > sa


def test_bop_and_ste_baselines_learn_digits():
    hb = TBP.run_bop("digits", 1, 128, 6, 64, 0, tau_flip=0.3, gamma=1e-2, val_split=200, bop_warmup=10, bop_k=0.02)
    assert hb[-1]["test"] > 0.5 and hb[-1]["flips_per_step"] > 0  # smoke test: Bop learns (it is not competitive)
    hs = TBP.run_ste("digits", 1, 128, 6, 64, 0, lr=3e-3, val_split=200)
    assert hs[-1]["test"] > 0.85


def test_rev_controller_rules():
    c = TBP.RevController(8, lo=0.1, hi=0.2, t_min=2, t_max=10, min_flips=100)
    assert c.update(50, 40) == (8, None)            # too few flips: wait
    assert c.update(200, 100)[0] == 9               # 100/200 = 0.5 > hi
    assert c.update(400, 120)[0] == 9               # 20/200 = 0.1: within [lo, hi] -> keep
    T, rate = c.update(700, 125)
    assert abs(rate - 5 / 300) < 1e-9 and T == 8    # < lo -> T - 1
    for i in range(10):
        c.update(c.f0 + 1000, c.r0 + 900)
    assert c.T == 10                                # capped


def test_rev_control_runs():
    hist, _ = TBP.run_tbp("digits", depth=2, width=128, epochs=5, bs=64, seed=0, k_frac=0.02, T=8, counter_bits=8,
                          rev_control=True, val_split=200, val_every=5)
    assert any(t.get("reversal") is not None for t in hist[-1]["k_trace"])
    assert hist[-1]["test"] > 0.7 and "reversal_rate" in hist[-1]


# ------------------------------------------------------------------------------------------- learnable LayerNorm
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("mode", ["gain", "gain_bias", "rms_gain"])
def test_ln_affine_identity_at_init(residual, mode):
    if mode == "rms_gain":
        pytest.skip("RMSNorm is a different normalization, not identical to the LN baseline")
    Ws = make([64, 48, 48, 10], 9)
    a = TBP.TBPNet(Ws, counter_bits=8, residual=residual)
    b = TBP.TBPNet(Ws, counter_bits=8, residual=residual, ln_affine=mode, ln_lr=0.0)
    torch.manual_seed(0)
    for _ in range(3):
        x, y = torch.rand(32, 64).cuda(), torch.randint(0, 10, (32,)).cuda()
        assert a.step(x, y, 0.05, 2) == b.step(x, y, 0.05, 2)
    assert all(torch.equal(p, q) for p, q in zip(a.Wp, b.Wp))


def _ref_aff(Ws, x, gs, bs, residual, rms):
    Wf = [W.double() for W in Ws]

    def norm(p):
        if rms:
            return p / p.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-5).detach()
        return ln_detached(p)
    h = x
    for l in range(len(Wf) - 1):
        y = norm(h @ Wf[l].T / math.sqrt(Wf[l].shape[1])) * gs[l] + bs[l]
        a = torch.relu(y)
        h = h + a if (residual and l > 0) else a
    if residual:
        h = ln_detached(h) * gs[-1] + bs[-1]
    return h @ Wf[-1].T / math.sqrt(Wf[-1].shape[1])


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("mode", ["gain_bias", "rms_gain"])
def test_ln_affine_grads_match_autograd(residual, mode):
    Ws = make([64, 40, 40, 40, 10], 10)
    tau, B = 4.0, 32
    net = TBP.TBPNet(Ws, counter_bits=8, residual=residual, ln_affine=mode, ln_lr=0.0)
    torch.manual_seed(1)
    with torch.no_grad():  # non-trivial parameters
        for t in net.ln_g:
            t.copy_(0.5 + torch.rand_like(t))
        for t in net.ln_b:
            t.copy_(0.2 * torch.randn_like(t))
    x, y = torch.rand(B, 64), torch.randint(0, 10, (B,))
    gs = [t.detach().double().cpu().requires_grad_(True) for t in net.ln_g]
    bs = [t.detach().double().cpu().requires_grad_(True) for t in net.ln_b]
    F.cross_entropy(tau * _ref_aff(Ws, x.double(), gs, bs, residual, mode == "rms_gain"), y).backward()
    logits, c = net._fwd_aff(x.cuda())
    E = torch.softmax(tau * logits, 1) - F.one_hot(y.cuda(), 10).float()
    _, (gg, gb) = net._err_aff(E, c)
    if residual:  # exact (LN statistics constant in both)
        for got, want in zip(gg, gs):
            torch.testing.assert_close((got * tau / B).double().cpu(), want.grad, rtol=1e-3, atol=1e-6)
        for got, want in zip(gb, bs):
            torch.testing.assert_close((got * tau / B).double().cpu(), want.grad, rtol=1e-3, atol=1e-6)
    else:  # plain: votes use the unscaled chain, so gradients match up to a positive per-layer factor; check sign
        for got, want in zip(gg, gs):
            agree = (torch.sign(got.double().cpu()) == torch.sign(want.grad)).float().mean()
            assert agree > 0.9
        want_last = gs[-1].grad  # last hidden layer feeds the output directly: exact up to 1/sqrt(d)
        torch.testing.assert_close((gg[-1] * tau / B / math.sqrt(40)).double().cpu(), want_last, rtol=1e-3, atol=1e-6)


def test_ln_affine_learns_digits():
    hist, net = TBP.run_tbp("digits", depth=3, width=128, epochs=6, bs=64, seed=0, k_frac=0.01, T=4, residual=True,
                            ln_affine="gain_bias", ln_lr=1e-2, val_split=200)
    assert hist[-1]["test"] > 0.85 and hist[-1]["ln_gain"]["std"] > 0


# ------------------------------------------------------------------------------------------- exact backward
@pytest.mark.parametrize("residual", [False, True])
def test_exact_backward_votes_are_true_weight_gradients(residual):
    dims = [64, 40, 40, 10]
    Ws = make(dims, 11)
    net = TBP.TBPNet(Ws, counter_bits=8, residual=residual, exact_backward=True, matmul="cublas")
    x, y = torch.rand(16, 64).cuda(), torch.randint(0, 10, (16,)).cuda()
    hs, errs, _ = net._exact_errors(x, y, 2.0)
    Wt = [W.float().cuda().requires_grad_(True) for W in Ws]
    out = TBP.residual_forward(Wt, x) if residual else TBP._float_forward(Wt, x, False)
    F.cross_entropy(2.0 * out, y, reduction="sum").backward()
    for l in range(len(Ws)):
        G = errs[l].T @ hs[l] / math.sqrt(dims[l])  # dL/dW_l (in units of (p - y), i.e. / tau)
        torch.testing.assert_close(G, Wt[l].grad / 2.0, rtol=1e-4, atol=1e-5)


def test_exact_backward_learns_digits():
    hist, _ = TBP.run_tbp("digits", depth=3, width=128, epochs=6, bs=64, seed=0, k_frac=0.01, T=4, residual=True,
                          exact_backward=True, val_split=200)
    assert hist[-1]["test"] > 0.85
