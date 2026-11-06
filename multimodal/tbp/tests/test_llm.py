"""TBP for BitNet LLMs: TBPLinear vs BitLinear references, chunked LM loss, and a tiny random BitNet end to end."""

import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

if torch.cuda.is_available():
    from bitnet_vl.bitlinear import activation_quant, replace_linears, ternary_weight, weight_quant
    from bitnet_vl.make_tiny_ckpt import tiny_llm_config
    from bitnet_vl.modeling import LLM_LINEAR_RE, _strip_quant_config
    from tbp import kernels as K
    from tbp import llm as L
    from tbp.tests.test_counters3 import ref_update

DEV = "cuda"


def make_linear(out_f, in_f, seed):
    torch.manual_seed(seed)
    lin = nn.Linear(in_f, out_f, bias=False)
    nn.init.normal_(lin.weight, std=0.05)
    return lin


@pytest.mark.parametrize("matmul", ["packed", "cublas"])
@pytest.mark.parametrize("out_f,in_f", [(64, 48), (130, 257), (640, 2560)])
def test_tbplinear_forward_matches_bitlinear(matmul, out_f, in_f):
    lin = make_linear(out_f, in_f, 0)
    q, s = ternary_weight(lin.weight)
    mod = L.TBPLinear(q, s, matmul=matmul)
    x = torch.randn(3, 17, in_f)
    ref = F.linear(activation_quant(x), weight_quant(lin.weight))
    torch.testing.assert_close(mod(x.to(DEV)).cpu(), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("matmul", ["packed", "cublas"])
def test_tbplinear_grad_input_and_in_backward_update(matmul):
    out_f, in_f, T = 96, 200, 1
    lin = make_linear(out_f, in_f, 1)
    q, s = ternary_weight(lin.weight)
    mod = L.TBPLinear(q, s, k_frac=0.05, T=T, matmul=matmul)
    x = torch.randn(4, 9, in_f)
    gy = torch.randn(4, 9, out_f)
    # reference: STE through activation quant, fixed dequantized ternary weights
    xr = x.clone().requires_grad_(True)
    wq = (q.float() * s)
    F.linear(xr + (activation_quant(xr) - xr).detach(), wq).backward(gy)
    xg = x.to(DEV).requires_grad_(True)
    mod(xg).backward(gy.to(DEV))
    torch.testing.assert_close(xg.grad.cpu(), xr.grad, rtol=1e-4, atol=1e-4)
    # the update applied in backward equals the reference rule on G = gy^T x_q
    G = gy.reshape(-1, out_f).T @ activation_quant(x.reshape(-1, in_f))
    Wr, cr, _ = ref_update(G, q, torch.zeros_like(q), mod.n_k, T)
    assert torch.equal(mod.ternary().cpu(), Wr)
    assert torch.equal(K.unpack_counters(mod.cnt, in_f, 4).cpu(), cr)


def test_update_disabled_keeps_weights():
    lin = make_linear(32, 64, 2)
    q, s = ternary_weight(lin.weight)
    mod = L.TBPLinear(q, s, k_frac=0.5, T=1)
    mod.update_enabled = False
    x = torch.randn(5, 64, device=DEV, requires_grad=True)
    mod(x).sum().backward()
    assert torch.equal(mod.ternary().cpu(), q)


@pytest.mark.parametrize("S,V,chunk", [(10, 50, 3), (64, 1000, 64), (37, 333, 16)])
def test_chunked_lm_loss_matches_cross_entropy(S, V, chunk):
    torch.manual_seed(S)
    W = torch.randn(V, 24, device=DEV)
    h = torch.randn(S, 24, device=DEV, requires_grad=True)
    y = torch.randint(0, V, (S,), device=DEV)
    y[::5] = -100
    tot, n = L.lm_loss(h, y, W, chunk)
    h2 = h.detach().clone().requires_grad_(True)
    ref = F.cross_entropy(h2 @ W.T, y, ignore_index=-100)
    ref.backward()
    assert n == int((y != -100).sum())
    assert abs(tot / n - ref.item()) < 1e-4
    torch.testing.assert_close(h.grad, h2.grad, rtol=1e-4, atol=1e-6)


# ------------------------------------------------------------------------------------------------ tiny BitNet
def tiny_lm(seed=0, vocab=512, layers=2):
    from transformers import BitNetForCausalLM

    torch.manual_seed(seed)
    cfg = _strip_quant_config(tiny_llm_config(vocab_size=vocab, hidden=128, intermediate=256, layers=layers,
                                              heads=4, kv_heads=2))
    cfg.bos_token_id = cfg.eos_token_id = cfg.pad_token_id = 0
    lm = BitNetForCausalLM(cfg).float()
    with torch.no_grad():
        for n, p in lm.named_parameters():
            if "norm" in n:
                p.add_(0.05 * torch.randn_like(p))
    return lm.eval()


def test_tiny_converted_logits_match_bitlinear_model():
    lm = tiny_lm()
    ref = copy.deepcopy(lm)
    replace_linears(ref, lambda n, m: bool(LLM_LINEAR_RE.search(n)))
    ids = torch.randint(0, 512, (1, 40))
    with torch.no_grad():
        want = ref(input_ids=ids).logits[0]
    mods = L.convert_bitnet(lm, matmul="packed")
    assert len(mods) == 2 * 7
    with torch.no_grad():
        h = lm.model(inputs_embeds=lm.model.embed_tokens(ids.to(DEV)).float()).last_hidden_state[0]
        got = (h @ L.head_weight(lm).float().T).cpu()
    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)
    assert (got.argmax(1) == want.argmax(1)).float().mean() > 0.97


def test_tiny_tbp_steps_lower_loss():
    lm = tiny_lm(1)
    L.convert_bitnet(lm, k_frac=0.02, T=1)
    torch.manual_seed(0)
    ids = torch.randint(0, 512, (65,), device=DEV)
    losses = []
    for step in range(30):
        tot, n = L.tbp_step(lm, ids, chunk=16, step=step)
        losses.append(tot / n)
    assert losses[-1] < losses[0] - 0.3, losses[::5]
    assert int(lm.tbp_flips.sum()) > 0


def test_tiny_checkpointing_gives_identical_updates():
    a, b = tiny_lm(2), tiny_lm(2)
    L.convert_bitnet(a, k_frac=0.05, T=1)
    L.convert_bitnet(b, k_frac=0.05, T=1)
    b.train()
    b.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    a.train()
    torch.manual_seed(1)
    for step in range(3):
        ids = torch.randint(0, 512, (49,), device=DEV)
        L.tbp_step(a, ids, chunk=16, step=step)
        L.tbp_step(b, ids, chunk=16, step=step)
    for ma, mb in zip(a.tbp_linears, b.tbp_linears):
        assert torch.equal(ma.Wp, mb.Wp) and torch.equal(ma.cnt, mb.cnt), ma.name


def test_random_control_flips_requested_count_roughly():
    lin = make_linear(64, 128, 3)
    q, s = ternary_weight(lin.weight)
    mod = L.TBPLinear(q, s)
    before = mod.ternary().clone()
    L.random_flips(mod, 50, torch.Generator(device=DEV).manual_seed(0))
    changed = int((mod.ternary() != before).sum())
    assert 40 <= changed <= 50
    assert mod.ternary().abs().max() <= 1


def _capture_votes(lm):
    """Disable updates and record each linear's vote matrix G = g^T x_q per backward."""
    store = {}
    for m in lm.tbp_linears:
        m.update_enabled = True

        def rec(g, xq, G=None, m=m):
            store[m.name] = store.get(m.name, 0) + (g.T @ xq).cpu()
        m.tbp_update = rec
    return store


def test_batched_step_matches_separate_sequences():
    torch.manual_seed(3)
    seqs = torch.randint(0, 512, (2, 33), device=DEV)
    lm_b, lm_s = tiny_lm(4), tiny_lm(4)
    L.convert_bitnet(lm_b)
    L.convert_bitnet(lm_s)
    Gb, Gs = _capture_votes(lm_b), _capture_votes(lm_s)
    tot_b, n_b = L.tbp_step(lm_b, seqs, chunk=16)
    tot_s = n_s = 0
    for s in seqs:
        t, n = L.tbp_step(lm_s, s, chunk=16)
        tot_s, n_s = tot_s + t, n_s + n
    assert n_b == n_s == 64
    assert abs(tot_b - tot_s) < 1e-3 * abs(tot_s)
    # votes over the batch = sum of the per-sequence votes (the batch loss is a mean over 2x the tokens: x 1/2)
    for name in Gb:
        torch.testing.assert_close(Gb[name], Gs[name] / 2, rtol=2e-3, atol=1e-5)


def test_batched_steps_lower_loss():
    lm = tiny_lm(5)
    L.convert_bitnet(lm, k_frac=0.02, T=1)
    torch.manual_seed(0)
    ids = torch.randint(0, 512, (2, 65), device=DEV)
    losses = []
    for step in range(30):
        tot, n = L.tbp_step(lm, ids, chunk=16, step=step)
        losses.append(tot / n)
    assert n == 128 and losses[-1] < losses[0] - 0.3, losses[::5]


def test_learnable_scale_gradient_matches_autograd():
    lin = make_linear(48, 96, 12)
    q, s = ternary_weight(lin.weight)
    mod = L.TBPLinear(q, s, matmul="cublas")
    mod.enable_learnable_scale()
    mod.update_enabled = False
    x = torch.randn(5, 7, 96)
    gy = torch.randn(5, 7, 48)
    xg = x.to(DEV)
    mod(xg).backward(gy.to(DEV))
    ls = torch.tensor([math.log(s)], dtype=torch.float64, requires_grad=True)
    y = F.linear(activation_quant(x).double(), q.double()) * ls.exp()
    y.backward(gy.double())
    torch.testing.assert_close(mod.log_scale.grad.double().cpu(), ls.grad, rtol=1e-4, atol=1e-6)


def test_learnable_scale_off_is_unchanged():
    lin = make_linear(32, 64, 13)
    q, s = ternary_weight(lin.weight)
    a, b = L.TBPLinear(q, s, k_frac=0.1, T=1), L.TBPLinear(q, s, k_frac=0.1, T=1)
    x = torch.randn(9, 64, device=DEV)
    a(x.clone().requires_grad_(True)).sum().backward()
    b(x.clone().requires_grad_(True)).sum().backward()
    assert torch.equal(a.Wp, b.Wp)


@pytest.mark.parametrize("rows", [8, 64, 517])
def test_int8_forward_matches_fp32(rows):
    lin = make_linear(640, 256, 14)
    q, s = ternary_weight(lin.weight)
    a, b = L.TBPLinear(q, s, k_frac=0.02, T=1, matmul="cublas"), L.TBPLinear(q, s, k_frac=0.02, T=1, matmul="int8")
    x = torch.randn(rows, 256, device=DEV) * 3
    xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    ya, yb = a(xa), b(xb)
    torch.testing.assert_close(yb, ya, rtol=1e-5, atol=1e-5)
    gy = torch.randn_like(ya)
    ya.backward(gy)
    yb.backward(gy)
    torch.testing.assert_close(xb.grad, xa.grad, rtol=1e-5, atol=1e-5)
    assert (a.ternary() != b.ternary()).sum() <= 2  # same votes up to fp32 near-ties


def test_frac_k_update_matches_reference_rows():
    lin = make_linear(64, 128, 15)
    q, s = ternary_weight(lin.weight)
    mod = L.TBPLinear(q, s, k_frac=0.004, T=1, matmul="cublas")  # 0.51 picks per row on average
    mod.frac_k = True
    torch.manual_seed(0)
    x = torch.randn(40, 128, device=DEV, requires_grad=True)
    mod(x).sum().backward()
    changed = (mod.ternary().cpu() != q).sum(1)
    assert int(changed.max()) <= 1 and 15 <= int((changed > 0).sum()) <= 50  # about half the rows flip one weight
