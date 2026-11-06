"""Ternary Backpropagation for BitNet LLMs (HF BitNetForCausalLM, e.g. microsoft/bitnet-b1.58-2B-4T-bf16).

Every decoder linear (q/k/v/o, gate/up/down) becomes a `TBPLinear`: 2-bit packed ternary weights, 4-bit flip counters
and a fixed per-tensor absmean scale. Its backward computes the input error with the pre-update weights, then votes
G = grad_out^T x_q and applies the TBP top-k / flip-counter update in place. No weight gradient or optimizer state
exists. Error flows through attention, RMSNorm, relu^2 and RoPE with ordinary autograd. Embeddings, LM head and norms
are frozen.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from bitnet_vl.bitlinear import activation_quant, ternary_weight
from bitnet_vl.modeling import LLM_LINEAR_RE

from . import kernels as K


class _TBPLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, log_scale, mod):
        shape = x.shape
        x2 = x.reshape(-1, mod.in_features).float().contiguous()
        if mod.matmul_mode == "int8" and x2.shape[0] > 16:
            # same quantization as activation_quant, kept as integers: int8 x ternary int8 -> exact int32 sums
            a = x2.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
            xi = (x2 / a).round().clamp(-128, 127).to(torch.int8)
            y = torch._int_mm(xi, mod._wi().T).float() * a * mod.current_scale()
            xq = xi.float() * a
        else:
            xq = activation_quant(x2)
            y = mod.matmul(xq)
        ctx.save_for_backward(xq)
        ctx.mod = mod
        return y.reshape(*shape[:-1], mod.out_features).to(x.dtype)

    @staticmethod
    def backward(ctx, gy):
        (xq,) = ctx.saved_tensors
        mod = ctx.mod
        g = gy.reshape(-1, mod.out_features).float().contiguous()
        gx = mod.matmul_back(g)  # error to the input, with this layer's pre-update weights
        g_log = None
        G = None
        if mod.log_scale is not None:  # dL/ds = sum(G * W) with the pre-update weights; d/dlog s = s * dL/ds
            G = g.T @ xq
            g_log = (G * mod._wf()).sum().reshape(1) * mod.current_scale()
        if mod.update_enabled:
            mod.tbp_update(g, xq, G)
        return gx.reshape(*gy.shape[:-1], mod.in_features).to(gy.dtype), g_log, None


class TBPLinear(nn.Module):
    """y = scale * act_quant(x) @ W_t^T with W_t in {-1, 0, 1}, trained by TBP in its own backward."""

    def __init__(self, w_t: torch.Tensor, scale: float, k_frac=0.0005, T=8, counter_bits=4, name="",
                 matmul="packed", flips_buf: torch.Tensor | None = None, index: int = 0):
        super().__init__()
        self.out_features, self.in_features = w_t.shape
        self.Wp = K.pack(w_t.to(torch.int8).cuda())
        z = torch.zeros(w_t.shape, dtype=torch.int8, device=self.Wp.device)
        self.cnt = z if counter_bits == 8 else K.pack_counters(z, counter_bits)
        del z
        self.scale, self.k_frac, self.T, self.counter_bits = float(scale), k_frac, T, counter_bits
        self.name, self.matmul_mode = name, matmul
        self.update_enabled = True
        self.random_control = False  # control: flip `replay` random movable weights instead of voting
        self.replay: int = 0
        self.flips_buf, self.index = flips_buf, index
        self.step_seed = 0
        self.log_scale = None  # set by enable_learnable_scale(): fp32 parameter, scale = exp(log_scale)
        self.frac_k = False  # per-row pick counts sampled with mean k_frac * fan_in (allows < 1 per row)

    def enable_learnable_scale(self):
        self.log_scale = nn.Parameter(torch.tensor([math.log(self.scale)], device=self.Wp.device))

    def current_scale(self) -> float:
        return float(self.log_scale.detach().exp()) if self.log_scale is not None else self.scale

    @property
    def n_k(self) -> int:
        return max(1, int(round(self.k_frac * self.in_features)))

    def _wf(self):
        return K.unpack(self.Wp, self.in_features, torch.float32)

    def _wi(self):
        return K.unpack(self.Wp, self.in_features)  # int8 [out, in]

    def matmul(self, xq):  # fp32 activations (int8 mode falls back here for <= 16 rows)
        s = self.current_scale()
        if self.matmul_mode == "packed":
            return K.mm_nt(xq, self.Wp, self.out_features, s)
        return (xq @ self._wf().T) * s

    def matmul_back(self, g):
        s = self.current_scale()
        if self.matmul_mode == "packed":
            return K.mm_nn(g, self.Wp, self.in_features, s)
        return (g @ self._wf()) * s

    def tbp_update(self, g, xq, G=None):
        out = self.flips_buf[self.index:self.index + 1] if self.flips_buf is not None else None
        if self.random_control:
            random_flips(self, self.replay)
            return
        if G is None:
            G = g.T @ xq  # votes [out, in]
        krows = K.sample_krows(self.k_frac, self.out_features, self.in_features, device=G.device) if self.frac_k \
            else None
        K.update(G, self.Wp, self.cnt, self.n_k, self.T, krows=krows, counter_bits=self.counter_bits,
                 seed=self.step_seed * 1000003 + self.index, flips_out=out)

    def forward(self, x):
        ls = self.log_scale if self.log_scale is not None else torch.zeros(1, device=x.device)
        return _TBPLinearFn.apply(x, ls, self)

    def ternary(self) -> torch.Tensor:
        return K.unpack(self.Wp, self.in_features)

    def extra_repr(self):
        return f"{self.in_features}->{self.out_features}, scale={self.scale:.4g}, k={self.n_k}/row, T={self.T}, " \
               f"counters={self.counter_bits}-bit"


@torch.no_grad()
def enable_learnable_scales(mods: list[TBPLinear], which: str = "od") -> int:
    """Learnable fp32 scale on the linears whose scale is not normalized away: "od" (o_proj, down_proj: each block's
    contribution to the residual stream), "odqk" (plus q/k: attention temperature) or "all". Returns how many."""
    names = {"od": ("o_proj", "down_proj"), "odqk": ("o_proj", "down_proj", "q_proj", "k_proj"),
             "all": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")}[which]
    n = 0
    for m in mods:
        if m.name.rsplit(".", 1)[-1] in names:
            m.enable_learnable_scale()
            n += 1
    return n


def random_flips(mod: TBPLinear, n: int, gen: torch.Generator | None = None):
    """Control: step n random movable weights by one ternary state in a random direction."""
    if n <= 0:
        return
    W = mod.ternary().view(-1)
    idx = torch.randint(0, W.numel(), (n,), device=W.device, generator=gen)
    d = torch.randint(0, 2, (n,), device=W.device, generator=gen, dtype=torch.int8) * 2 - 1
    d = torch.where((W[idx] + d).abs() > 1, -d, d)  # saturated: step the other way
    W.index_add_(0, idx, d)  # duplicates may stack; clamp keeps it ternary
    W.clamp_(-1, 1)
    mod.Wp.copy_(K.pack(W.view(mod.out_features, mod.in_features)))


def convert_bitnet(lm: nn.Module, k_frac=0.0005, T=8, counter_bits=4, matmul="packed",
                   device="cuda", freeze_rest=True) -> list[TBPLinear]:
    """Replace every decoder linear of an HF BitNet model (on CPU, latent bf16/fp32 weights) by a TBPLinear on the
    GPU, layer by layer (the masters are freed as they go). Moves the rest of the model to `device`, keeps embeddings
    in their dtype (frozen) and casts norms to fp32."""
    names = [n for n, m in lm.named_modules() if isinstance(m, nn.Linear) and LLM_LINEAR_RE.search(n)]
    flips_buf = torch.zeros(len(names), dtype=torch.int32, device=device)
    mods = []
    for i, name in enumerate(names):
        parent_name, child = name.rsplit(".", 1)
        parent = lm.get_submodule(parent_name)
        lin = getattr(parent, child)
        assert lin.bias is None, f"{name} has a bias"
        q, s = ternary_weight(lin.weight)
        new = TBPLinear(q, s, k_frac, T, counter_bits, name, matmul, flips_buf, i)
        setattr(parent, child, new)
        del lin, q
        mods.append(new)
    for p in lm.parameters():
        p.requires_grad_(not freeze_rest)  # freeze_rest=False: embedding and norms stay trainable (own optimizer)
    lm.to(device)
    for n, m in lm.named_modules():
        if "norm" in n.split(".")[-1] and hasattr(m, "weight") and m.weight is not None:
            m.float()
    lm.tbp_linears = mods
    lm.tbp_flips = flips_buf
    return mods


def autotune(mods: list[TBPLinear], rows: int, reps: int = 5) -> dict:
    """Pick packed kernels or cuBLAS (unpacked copy) per (out, in) shape at `rows` tokens; sets each module's mode."""
    import time

    choice = {}
    for shape in sorted({(m.out_features, m.in_features) for m in mods}):
        m = next(x for x in mods if (x.out_features, x.in_features) == shape)
        xq = torch.randn(rows, m.in_features, device=m.Wp.device)
        g = torch.randn(rows, m.out_features, device=m.Wp.device)
        ts = {}
        for mode in ("packed", "cublas"):
            m.matmul_mode = mode
            f = lambda: (m.matmul(xq), m.matmul_back(g))  # noqa: E731
            f()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                f()
            torch.cuda.synchronize()
            ts[mode] = (time.perf_counter() - t) / reps
        if rows > 16 and m.in_features % 8 == 0 and m.out_features % 8 == 0:  # int8 forward (+ fp32 backward)
            m.matmul_mode = "int8"
            xin = torch.randn(rows, m.in_features, device=m.Wp.device)

            def f8():
                a = xin.abs().amax(-1, keepdim=True) / 127.0
                xi = (xin / a).round().clamp(-128, 127).to(torch.int8)
                return torch._int_mm(xi, m._wi().T), m.matmul_back(g)
            f8()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                f8()
            torch.cuda.synchronize()
            ts["int8"] = (time.perf_counter() - t) / reps
        choice[shape] = min(ts, key=ts.get)
    for m in mods:
        m.matmul_mode = choice[(m.out_features, m.in_features)]
    return choice


def head_weight(lm: nn.Module) -> torch.Tensor:
    return lm.lm_head.weight


def lm_loss(hidden: torch.Tensor, labels: torch.Tensor, W_head: torch.Tensor, chunk: int = 64,
            backward: bool = True):
    """Mean next-token cross-entropy over labels != -100, computed in chunks against W_head [V, d] (any dtype).
    With backward=True, back-propagates the softmax error into `hidden` (which must require grad) without ever
    holding the full [S, V] logits. Returns (loss sum, number of tokens)."""
    h = hidden.detach()
    S = h.shape[0]
    valid = labels != -100
    n = int(valid.sum())
    grad = torch.zeros_like(h) if backward else None
    total = torch.zeros((), dtype=torch.float64, device=h.device)
    for i in range(0, S, chunk):
        hc, yc, vc = h[i:i + chunk], labels[i:i + chunk], valid[i:i + chunk]
        logits = (hc.to(W_head.dtype) @ W_head.T).float()
        lse = torch.logsumexp(logits, 1)
        tgt = logits.gather(1, yc.clamp_min(0).unsqueeze(1)).squeeze(1)
        total += ((lse - tgt) * vc).sum().double()
        if backward:
            p = torch.softmax(logits, 1)
            p[torch.arange(len(yc), device=p.device), yc.clamp_min(0)] -= 1.0
            p *= vc.unsqueeze(1).float() / max(n, 1)
            grad[i:i + chunk] = (p.to(W_head.dtype) @ W_head).float()
    if backward:
        hidden.backward(grad)
    return float(total), n


def tbp_step(lm: nn.Module, ids: torch.Tensor, chunk: int = 64, step: int = 0) -> tuple[float, int]:
    """One TBP step on token ids [S+1] or a batch [B, S+1] (next-token targets, all sequences the same length).
    Every TBPLinear votes once over all B*S tokens and updates in place during backward. Returns (loss sum, tokens)."""
    for m in lm.tbp_linears:
        m.step_seed = step
    ids = ids if ids.dim() == 2 else ids.unsqueeze(0)
    inp, labels = ids[:, :-1], ids[:, 1:].reshape(-1)
    emb = lm.model.embed_tokens(inp).float().requires_grad_(True)  # leaf, so the backward pass runs
    hidden = lm.model(inputs_embeds=emb, use_cache=False).last_hidden_state
    return lm_loss(hidden.reshape(-1, hidden.shape[-1]), labels, head_weight(lm), chunk, backward=True)


@torch.no_grad()
def eval_ppl(lm: nn.Module, seqs: list[torch.Tensor], chunk: int = 64) -> float:
    tot, n = 0.0, 0
    for ids in seqs:
        emb = lm.model.embed_tokens(ids[:-1].unsqueeze(0)).float()
        hidden = lm.model(inputs_embeds=emb, use_cache=False).last_hidden_state[0]
        t, c = lm_loss(hidden, ids[1:], head_weight(lm), chunk, backward=False)
        tot, n = tot + t, n + c
    return math.exp(tot / max(n, 1))
