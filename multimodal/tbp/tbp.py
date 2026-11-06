"""Ternary Backpropagation (TBP) on the GPU: ternary MLPs trained with the CUDA kernels in tbp.kernels.
Algorithm: docs/ternary-backpropagation.md.

    cd multimodal
    runs/.venv-tbp/bin/python -m tbp.tbp --dataset fmnist --depth 3 --width 1786 --epochs 15 --bs 256
    runs/.venv-tbp/bin/python -m tbp.tbp --dataset digits --depth 1 --width 39321 --seeds 0,1,2 --ste-width 4096

Initialization, data split and batch order follow the CPU prototype exactly (same seeds), so trajectories can be
compared step by step.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import kernels as K

DATA_DIR = Path(__file__).resolve().parents[1] / "runs" / "data"


# ----------------------------------------------------------------------------------------------------------- data
def _idx(path: Path) -> np.ndarray:
    raw = path.read_bytes() if path.exists() else gzip.open(str(path) + ".gz").read()
    ndim = raw[3]
    shape = [int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load(name: str, seed: int):
    """CPU tensors (xtr, xte, ytr, yte). digits: sklearn 80/20 split per seed (as the prototype); fmnist: 60k/10k."""
    if name == "digits":
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        d = load_digits()
        x = torch.tensor(d.data / 16.0, dtype=torch.float32)
        y = torch.tensor(d.target)
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=seed, stratify=y)
        return xtr, xte, ytr, yte
    raw = DATA_DIR / "FashionMNIST" / "raw"
    f = lambda p: torch.from_numpy(_idx(raw / p).copy())  # noqa: E731
    xtr = f("train-images-idx3-ubyte").reshape(60000, -1).float() / 255.0
    xte = f("t10k-images-idx3-ubyte").reshape(10000, -1).float() / 255.0
    return xtr, xte, f("train-labels-idx1-ubyte").long(), f("t10k-labels-idx1-ubyte").long()


# ----------------------------------------------------------------------------------------------------------- model
def layer_norm(p):
    return (p - p.mean(1, keepdim=True)) / p.std(1, keepdim=True).clamp_min(1e-5)


class TBPNet:
    """Ternary MLP din -> [width] * depth -> n_cls. Weights packed 2-bit on the GPU; flip counters 4-bit by default (8 per word, T <= 8), or int8 / 3-bit."""

    def __init__(self, Ws: list[torch.Tensor], device="cuda", matmul="packed", batch=256, counter_bits=4,
                 p_inc=1.0, residual=False, learn_scale=False, scale_lr=1e-2, frac_k=False, counter_reset=0,
                 sig_z=0.0, instrument=False, rev_window=200, agree_every=10, counter_decay=0, cooldown=0,
                 vote_strength=0.0, stagger=1, flip_every=1, ln_affine="none", ln_lr=1e-3, exact_backward=False):
        """matmul: "packed" (ternary kernels), "cublas" (unpacked fp32 copy per call) or "auto" (time both once per
        layer and direction at this batch size and keep the faster)."""
        self.dims = [Ws[0].shape[1]] + [W.shape[0] for W in Ws]
        self.device = torch.device(device)
        if self.device.type == "cpu":
            from . import cpu_backend
            self.K = cpu_backend
            matmul = "cublas"  # dense matmuls on the CPU
        else:
            self.K = K
        self.residual = residual
        if residual:  # h_0 = relu(LN(W_0 x)); h_l = h_{l-1} + relu(LN(W_l h_{l-1})); logits = W_out LN(h)
            assert len(set(self.dims[1:-1])) == 1, "residual blocks need equal hidden widths"
        self.Wp = [self.K.pack(W.to(device)) for W in Ws]
        assert counter_bits in (8, 4, 3)
        self.counter_bits, self.p_inc, self.n_steps = counter_bits, p_inc, 0
        if counter_bits == 8:
            self.cnt = [torch.zeros(W.shape, dtype=torch.int8, device=device) for W in Ws]
        else:  # packed fields: 3-bit (10 per word, T <= 4) or 4-bit (8 per word, T <= 8)
            self.cnt = [self.K.pack_counters(torch.zeros(W.shape, dtype=torch.int8, device=device), counter_bits)
                        for W in Ws]
        n = len(Ws)
        if matmul == "auto":
            self.fwd_mode, self.bwd_mode = self._autotune(batch, device)
        else:
            self.fwd_mode, self.bwd_mode = [matmul] * n, [matmul] * n
        self.matmul = matmul
        # learnable per-layer scales (applied after the LN, see _s); log-parameterized so Adam's step is relative.
        # Plain nets: the next layer's LN cancels every hidden scale except the last, so only the last hidden and the
        # output scales are trained (the others stay 1). Residual nets: all of them change the function.
        self.learn_scale = learn_scale
        self.log_s = torch.zeros(n, dtype=torch.float32, device=device)
        self.scale_mask = torch.ones(n, device=device)
        if not residual and n > 2:
            self.scale_mask[:n - 2] = 0.0
        self.scale_opt = torch.optim.Adam([self.log_s.requires_grad_(True)], lr=scale_lr) if learn_scale else None
        self.frac_k = frac_k  # sample per-row pick counts with mean k * fan_in (allows < 1 per row)
        # stability options: reset every flip counter to 0 every `counter_reset` steps (votes must reach T within that
        # window); with sig_z > 0 only weights with |G| > sig_z * mean_row(|G|) are eligible to be picked
        self.counter_reset, self.sig_z = counter_reset, sig_z
        self._cnt_zero = [c.clone() for c in self.cnt] if counter_reset else None  # packed "all counters = 0"
        # more stability options (all off by default):
        #   counter_decay N: every N steps each counter moves one step toward 0
        #   cooldown W: a weight may not flip back within W steps of its last flip (the counter keeps the votes)
        #   vote_strength z: votes with |G| > z * row mean |G| count twice
        #   stagger G: each step only layers with (step + l) % G == 0 vote and flip
        #   flip_every M: vote every step, flip only every M steps (int8 counters; in between T is set out of reach)
        self.counter_decay, self.cooldown, self.vote_strength = counter_decay, cooldown, vote_strength
        self.stagger, self.flip_every = max(1, stagger), max(1, flip_every)
        if flip_every > 1:
            assert counter_bits == 8, "flip_every needs int8 counters (counter_bits=8)"
        # learnable LayerNorm (ln_affine): y = g * n + b after each hidden norm (and the final norm of residual nets);
        # "gain" (g only), "gain_bias" (g and b), "rms_gain" (RMSNorm without centering, g only, as in BitNet).
        # g = 1, b = 0 at init reproduces the affine-free network exactly. Trained by Adam (ln_lr).
        assert ln_affine in ("none", "gain", "gain_bias", "rms_gain")
        assert not (learn_scale and ln_affine != "none"), "use either learn_scale or ln_affine"
        self.ln_affine = ln_affine
        # exact_backward: the votes come from the true gradient dL/draw_l (autograd through the float network with the
        # ternary weights fixed, LN statistics differentiated), so G_l is exactly the weight gradient of layer l
        self.exact_backward = exact_backward
        assert not (exact_backward and (learn_scale or ln_affine != "none")), "exact_backward: no scales / affine LN"
        if ln_affine != "none":
            n_norm = (n - 1) + (1 if residual else 0)
            widths = self.dims[1:n] + ([self.dims[n - 1]] if residual else [])
            self.ln_g = [torch.ones(w, device=device, requires_grad=True) for w in widths[:n_norm]]
            self.ln_b = [torch.zeros(w, device=device, requires_grad=ln_affine == "gain_bias") for w in widths[:n_norm]]
            params = self.ln_g + (self.ln_b if ln_affine == "gain_bias" else [])
            self.ln_opt = torch.optim.Adam(params, lr=ln_lr)
        # instrumentation (see stats()): per weight, step and direction of its last flip, for the reversal rate
        self.instrument, self.rev_window, self.agree_every = instrument, rev_window, agree_every
        if instrument or cooldown:
            self._last_step = [torch.full(W.shape, -10**9, dtype=torch.int32, device=device) for W in Ws]
            self._last_dir = [torch.zeros(W.shape, dtype=torch.int8, device=device) for W in Ws]
            self.reset_stats()

    def _counters_unpacked(self, l):
        c = self.cnt[l]
        return c if self.counter_bits == 8 else self.K.unpack_counters(c, self.dims[l], self.counter_bits)

    def _set_counters(self, l, cu):
        if self.counter_bits == 8:
            self.cnt[l].copy_(cu)
        else:
            self.cnt[l].copy_(self.K.pack_counters(cu, self.counter_bits))

    def _apply_cooldown(self, l, before):
        """Undo flips that reverse a flip of the same weight made within the last `cooldown` steps; the counter is put
        back to one vote short of the threshold so the flip happens once the cooldown has passed (if votes persist)."""
        after = self.K.unpack(self.Wp[l], self.dims[l])
        changed = after != before
        if not bool(changed.any()):
            return 0
        d = torch.sign(after.to(torch.int16) - before.to(torch.int16)).to(torch.int8)
        blocked = changed & ((self.n_steps - self._last_step[l]) <= self.cooldown) & (self._last_dir[l] == -d)
        if bool(blocked.any()):
            after = torch.where(blocked, before, after)
            self.Wp[l].copy_(self.K.pack(after))
            cu = self._counters_unpacked(l).clone()
            cu[blocked] = (d[blocked] * (self._cur_T - 1)).to(torch.int8)
            self._set_counters(l, cu)
        return int(blocked.sum().item())
        self.last_scale_grad = None

    def reset_stats(self):
        n = len(self.Wp)
        self._st = dict(flips=[0] * n, reversals=[0] * n, agree=[0.0] * n, agree_n=[0] * n)

    def stats(self, T=None, reset=True) -> dict:
        """Per-layer instrumentation since the last reset: flips, reversal rate (flips undoing a flip of the same weight
        within rev_window steps), half-batch vote agreement among the picked weights (0.5 = noise, 1 = signal), and
        counter occupancy (share of counters at |c| >= T - 1, i.e. one vote from flipping)."""
        st = self._st
        out = dict(flips=list(st["flips"]),
                   reversal_rate=[r / f if f else 0.0 for r, f in zip(st["reversals"], st["flips"])],
                   agreement=[a / n if n else float("nan") for a, n in zip(st["agree"], st["agree_n"])])
        if T is not None:
            occ = []
            for l, c in enumerate(self.cnt):
                cu = c if self.counter_bits == 8 else self.K.unpack_counters(c, self.dims[l], self.counter_bits)
                occ.append(float((cu.abs() >= max(T - 1, 1)).float().mean()))
            out["occupancy"] = occ
        tot_f = sum(st["flips"])
        out["reversal_rate_all"] = sum(st["reversals"]) / tot_f if tot_f else 0.0
        if reset:
            self.reset_stats()
        return out

    def _agreement(self, l, errs_l, h_l, G, n_pick):
        """Share of the top picks (by |G| over the full batch) whose two half-batch votes have the same sign."""
        B = errs_l.shape[0] // 2
        if B == 0:
            return
        G1 = errs_l[:B].T @ h_l[:B]
        G2 = errs_l[B:2 * B].T @ h_l[B:2 * B]
        idx = G.abs().topk(min(max(n_pick, 1), G.shape[1]), dim=1).indices
        a, b = G1.gather(1, idx), G2.gather(1, idx)
        valid = (a != 0) & (b != 0)
        same = ((torch.sign(a) == torch.sign(b)) & valid).sum().item()
        self._st["agree"][l] += same
        self._st["agree_n"][l] += int(valid.sum().item())

    def _track_flips(self, l, before):
        after = self.K.unpack(self.Wp[l], self.dims[l])
        changed = after != before
        if not bool(changed.any()):
            return
        d = torch.sign(after.to(torch.int16) - before.to(torch.int16)).to(torch.int8)
        recent = (self.n_steps - self._last_step[l]) <= self.rev_window
        rev = changed & recent & (self._last_dir[l] == -d)
        self._st["flips"][l] += int(changed.sum().item())
        self._st["reversals"][l] += int(rev.sum().item())
        self._last_step[l][changed] = self.n_steps
        self._last_dir[l][changed] = d[changed]

    def _track_last(self, l, before):
        after = self.K.unpack(self.Wp[l], self.dims[l])
        changed = after != before
        if bool(changed.any()):
            d = torch.sign(after.to(torch.int16) - before.to(torch.int16)).to(torch.int8)
            self._last_step[l][changed] = self.n_steps
            self._last_dir[l][changed] = d[changed]

    def scales(self) -> torch.Tensor:
        return self.log_s.detach().exp()

    def _autotune(self, batch, device):
        import time

        def best(fns):
            ts = {}
            for name, f in fns.items():
                f()
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(10):
                    f()
                torch.cuda.synchronize()
                ts[name] = time.perf_counter() - t
            return min(ts, key=ts.get)

        fwd, bwd = [], []
        for l, Wp in enumerate(self.Wp):
            din, dout = self.dims[l], self.dims[l + 1]
            h = torch.rand(batch, din, device=device)
            e = torch.randn(batch, dout, device=device)
            fwd.append(best({"packed": lambda: self.K.mm_nt(h, Wp, dout),
                             "cublas": lambda: h @ self.K.unpack(Wp, din, torch.float32).T}))
            bwd.append(best({"packed": lambda: self.K.mm_nn(e, Wp, din),
                             "cublas": lambda: e @ self.K.unpack(Wp, din, torch.float32)}))
        return fwd, bwd

    def weights(self) -> list[torch.Tensor]:
        return [self.K.unpack(Wp, self.dims[i]) for i, Wp in enumerate(self.Wp)]

    def _mm(self, h, l):
        din, dout = self.dims[l], self.dims[l + 1]
        s = 1.0 / math.sqrt(din)
        if self.fwd_mode[l] == "packed":
            return self.K.mm_nt(h, self.Wp[l], dout, s)
        return (h @ self.K.unpack(self.Wp[l], din, torch.float32).T) * s  # cuBLAS on an unpacked fp32 copy

    def _mm_back(self, e, l):
        """e @ W_l (error sent back through layer l's ternary weights)."""
        if self.bwd_mode[l] == "packed":
            return self.K.mm_nn(e, self.Wp[l], self.dims[l])
        return e @ self.K.unpack(self.Wp[l], self.dims[l], torch.float32)

    def _exact_errors(self, x, y, tau):
        """(inputs[l], errs[l] = dL/draw_l * B, logits) by autograd; raw_l = input_l @ W_l^T / sqrt(d_l)."""
        L = len(self.Wp)
        Wf = [self.K.unpack(self.Wp[l], self.dims[l], torch.float32) for l in range(L)]
        raws, inputs = [], [x]
        with torch.enable_grad():
            h = x
            for l in range(L - 1):
                raw = (inputs[-1] @ Wf[l].T) / math.sqrt(self.dims[l])
                raw.retain_grad() if raw.requires_grad else raw.requires_grad_(True)
                raws.append(raw)
                a = torch.relu(layer_norm(raw))
                h = h + a if (self.residual and l > 0) else a
                inputs.append(h if (not self.residual or l < L - 2) else h)
            z = layer_norm(h) if self.residual else h
            if self.residual:
                inputs[-1] = z.detach()
            raw_out = (z @ Wf[-1].T) / math.sqrt(self.dims[L - 1])
            raw_out.retain_grad()
            raws.append(raw_out)
            loss = F.cross_entropy(tau * raw_out, y, reduction="sum")  # sum: errors in units of (p - y) * tau
            loss.backward()
        errs = [r.grad.detach() / tau for r in raws]  # = (p - y) at the output, like the default signal
        return [t.detach() for t in inputs], errs, raw_out.detach()

    def _norm(self, raw):
        if self.ln_affine == "rms_gain":
            sig = raw.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-5)
            return raw / sig, sig
        sig = raw.std(1, keepdim=True).clamp_min(1e-5)
        return (raw - raw.mean(1, keepdim=True)) / sig, sig

    def _fwd_aff(self, x, keep=True):
        """Forward with learnable norm parameters. Cache: inputs[l] (input of layer l), n[l], y[l] = g n + b, sig[l],
        and for residual nets z (final LN output before its affine) and sig_f."""
        L = len(self.Wp)
        g, b = self.ln_g, self.ln_b
        inputs, ns, ys, sigs = [x], [], [], []
        h = x
        for l in range(L - 1):
            n, sig = self._norm(self._mm(h, l))
            y = n * g[l].detach() + b[l].detach()
            a = torch.relu(y)
            h = h + a if (self.residual and l > 0) else a
            if keep:
                ns.append(n)
                ys.append(y)
                sigs.append(sig)
                inputs.append(h)
        c = dict(n=ns, y=ys, sig=sigs)
        if self.residual:
            sig_f = h.std(1, keepdim=True).clamp_min(1e-5)
            z = (h - h.mean(1, keepdim=True)) / sig_f
            zz = z * g[-1].detach() + b[-1].detach()
            if keep:
                inputs[-1] = zz
                c.update(z=z, sig_f=sig_f)
            out = self._mm(zz, L - 1)
        else:
            out = self._mm(h, L - 1)
        c["inputs"] = inputs
        return out, c

    def _err_aff(self, E, c):
        """Vote errors (w.r.t. each norm output n) and gradients of g and b, in units of E. LN statistics are treated
        as constants. With g = 1, b = 0 this equals the affine-free errors."""
        L = len(self.Wp)
        g = [t.detach() for t in self.ln_g]
        gg = [None] * len(g)
        gb = [None] * len(g)
        errs = [None] * L
        errs[-1] = E
        if self.residual:
            e_zz = self._mm_back(E, L - 1) / math.sqrt(self.dims[L - 1])
            gg[-1] = (e_zz * c["z"]).sum(0)
            gb[-1] = e_zz.sum(0)
            e_h = e_zz * g[-1] / c["sig_f"]
            for l in range(L - 2, -1, -1):
                dy = e_h * (c["y"][l] > 0).float()
                gg[l], gb[l] = (dy * c["n"][l]).sum(0), dy.sum(0)
                errs[l] = dy * g[l]
                if l > 0:
                    e_h = e_h + self._mm_back(errs[l] / c["sig"][l], l) / math.sqrt(self.dims[l])
        else:
            for l in range(L - 2, -1, -1):
                back = self._mm_back(errs[l + 1], l + 1)
                dy = back * (c["y"][l] > 0).float()
                gg[l], gb[l] = (dy * c["n"][l]).sum(0), dy.sum(0)
                errs[l] = dy * g[l]
        return errs, (gg, gb)

    def _forward_res(self, x, keep=True):
        """Residual forward: h_0 = s_0 relu(n_0); h_l = h_{l-1} + s_l relu(n_l); logits = s_out W_out LN(h) / sqrt(d).
        Cache: inputs[l] (input of layer l), n[l] (normalized pre-activation), sig[l] (std of the raw pre-activation,
        the LN scale), sig_f (final LN scale), u (output before s_out). Scales are only applied when learn_scale."""
        L = len(self.Wp)
        sc = self.scales() if self.learn_scale else None
        inputs, ns, sigs = [x], [], []
        raw = self._mm(x, 0)
        sig = raw.std(1, keepdim=True).clamp_min(1e-5)
        n = (raw - raw.mean(1, keepdim=True)) / sig
        h = torch.relu(n) if sc is None else torch.relu(n) * sc[0]
        if keep:
            ns.append(n)
            sigs.append(sig)
        for l in range(1, L - 1):
            if keep:
                inputs.append(h)
            raw = self._mm(h, l)
            sig = raw.std(1, keepdim=True).clamp_min(1e-5)
            n = (raw - raw.mean(1, keepdim=True)) / sig
            h = h + (torch.relu(n) if sc is None else torch.relu(n) * sc[l])
            if keep:
                ns.append(n)
                sigs.append(sig)
        sig_f = h.std(1, keepdim=True).clamp_min(1e-5)
        z = (h - h.mean(1, keepdim=True)) / sig_f
        if keep:
            inputs.append(z)
        u = self._mm(z, L - 1)
        return (u if sc is None else u * sc[-1]), dict(inputs=inputs, n=ns, sig=sigs, sig_f=sig_f, u=u)

    def _residual_errors(self, E, c):
        """Local errors (for the votes) of every layer of the residual net. The error flows unchanged through each skip
        and through each branch via W_l, the ReLU mask, the branch scale s_l and the forward scales 1/sqrt(d) and
        1/sigma (LN statistics treated as constants). Also returns the scale gradients (same units as E)."""
        L = len(self.Wp)
        sc = self.scales() if self.learn_scale else None
        gs = torch.zeros(L, device=E.device) if sc is not None else None
        errs = [None] * L
        errs[-1] = E
        e_h = self._mm_back(E, L - 1) / math.sqrt(self.dims[L - 1]) / c["sig_f"]
        if sc is not None:
            gs[-1] = (E * c["u"]).sum()
            e_h = e_h * sc[-1]
        for l in range(L - 2, 0, -1):
            act = torch.relu(c["n"][l])
            if sc is not None:
                gs[l] = (e_h * act).sum()
            local = e_h * (c["n"][l] > 0).float()
            if sc is not None:
                local = local * sc[l]
            errs[l] = local
            e_h = e_h + self._mm_back(local / c["sig"][l], l) / math.sqrt(self.dims[l])
        if sc is not None:
            gs[0] = (e_h * torch.relu(c["n"][0])).sum()
        errs[0] = e_h * (c["n"][0] > 0).float() * (1.0 if sc is None else sc[0])
        return errs, gs

    def forward(self, x):
        if self.residual:
            logits, c = self._forward_res(x)
            return c, c["inputs"], logits
        sc = self.scales() if self.learn_scale else None
        pres, hs = [], [x]
        h = x
        for l in range(len(self.Wp) - 1):
            pre = layer_norm(self._mm(h, l))
            h = torch.relu(pre) if sc is None else torch.relu(pre) * sc[l]
            pres.append(pre)
            hs.append(h)
        u = self._mm(h, len(self.Wp) - 1)
        self._u = u
        return pres, hs, (u if sc is None else u * sc[-1])

    def _plain_errors(self, E, pres):
        L = len(self.Wp)
        sc = self.scales() if self.learn_scale else None
        errs = [None] * L
        errs[-1] = E
        gs = None
        if sc is not None:
            gs = torch.zeros(L, device=E.device)
            gs[-1] = (E * self._u).sum()
        for l in range(L - 2, -1, -1):  # error sent back through the ternary weights (pre-update)
            back = self._mm_back(errs[l + 1], l + 1)
            if sc is not None:
                if l == L - 2:  # last hidden: logits = s_out W (s_L relu(n_L)) / sqrt(d), no LN after it (exact)
                    gs[l] = (back * sc[-1] / math.sqrt(self.dims[l + 1]) * torch.relu(pres[l])).sum()
                    back = back * sc[-1]
                back = back * sc[l]
            errs[l] = back * (pres[l] > 0).float()
        return errs, gs

    def step(self, x, y, k_frac=0.005, T=8, signal="softmax", tau=2.0, margin=0.5) -> int:
        if self.counter_bits != 8 and T > (1 << (self.counter_bits - 1)):
            raise ValueError(f"T={T} needs counters up to {T - 1}, but {self.counter_bits}-bit counters hold "
                             f"{(1 << (self.counter_bits - 1)) - 1}: use counter_bits=8 (--counter-bits 8)")
        if self.exact_backward:
            hs, errs_exact, logits = self._exact_errors(x, y, tau)
        elif self.ln_affine != "none":
            logits, c_aff = self._fwd_aff(x)
            hs = c_aff["inputs"]
        else:
            pres, hs, logits = self.forward(x)
        if signal == "softmax":
            E = torch.softmax(tau * logits, 1) - F.one_hot(y, logits.shape[1]).float()
        else:
            E = perceptron_signal(logits, y, margin)
        ln_grads = None
        if self.exact_backward:
            errs, gs = errs_exact, None
        elif self.ln_affine != "none":
            errs, ln_grads = self._err_aff(E, c_aff)
            gs = None
        else:
            errs, gs = self._residual_errors(E, pres) if self.residual else self._plain_errors(E, pres)
        flips = 0
        self._cur_T = T
        T_eff = T if (self.n_steps % self.flip_every == 0) else 127  # flip_every: no flips between flip rounds
        for l in range(len(self.Wp)):
            if (self.n_steps + l) % self.stagger != 0:
                continue
            G = errs[l].T @ hs[l]  # votes [out, in] (cuBLAS)
            if self.sig_z > 0:  # a weight votes only if its signal is stronger than sig_z x its row's mean |G|
                A = G.abs()
                G = G * (A > self.sig_z * A.mean(1, keepdim=True))
            if self.instrument or self.cooldown:
                before = self.K.unpack(self.Wp[l], self.dims[l])
            if self.instrument:
                if self.n_steps % self.agree_every == 0:
                    self._agreement(l, errs[l], hs[l], G, max(1, int(round(k_frac * self.dims[l]))))
            if self.frac_k:
                kr = self.K.sample_krows(k_frac, self.dims[l + 1], self.dims[l], device=G.device)
                flips += self.K.update(G, self.Wp[l], self.cnt[l], 1, T, p_inc=self.p_inc,
                                  seed=self.n_steps * 1000003 + l, counter_bits=self.counter_bits, krows=kr)
            else:
                n = max(1, int(round(k_frac * self.dims[l])))
                flips += self.K.update(G, self.Wp[l], self.cnt[l], n, T_eff, p_inc=self.p_inc,
                                  seed=self.n_steps * 1000003 + l, counter_bits=self.counter_bits)
                if self.vote_strength > 0:  # strong votes count twice: a second pass over the strong entries only
                    A = G.abs()
                    Gs = G * (A > self.vote_strength * A.mean(1, keepdim=True))
                    if bool((Gs != 0).any()):
                        flips += self.K.update(Gs, self.Wp[l], self.cnt[l], n, T_eff, p_inc=self.p_inc,
                                               seed=self.n_steps * 1000003 + l + 7919, counter_bits=self.counter_bits)
            if self.cooldown:
                flips -= self._apply_cooldown(l, before)
            if self.instrument:
                self._track_flips(l, before)
            elif self.cooldown:
                self._track_last(l, before)
        if ln_grads is not None:
            f = tau / x.shape[0]  # E is (p - y): the mean cross-entropy gradient is tau / B times that
            for t, gr in zip(self.ln_g, ln_grads[0]):
                t.grad = gr * f
            if self.ln_affine == "gain_bias":
                for t, gr in zip(self.ln_b, ln_grads[1]):
                    t.grad = gr * f
            self.ln_opt.step()
        if gs is not None:
            # E is (p - y); the mean cross-entropy gradient is tau / B times that (softmax signal)
            g_log = gs * (tau / x.shape[0]) * self.scales() * self.scale_mask
            self.last_scale_grad = g_log
            self.log_s.grad = g_log
            self.scale_opt.step()
            with torch.no_grad():
                self.log_s.clamp_(-3.0, 3.0)
        self.n_steps += 1
        if self.counter_decay and self.n_steps % self.counter_decay == 0:
            for l in range(len(self.cnt)):
                cu = self._counters_unpacked(l)
                self._set_counters(l, (cu - torch.sign(cu)).to(torch.int8))
        if self.counter_reset and self.n_steps % self.counter_reset == 0:
            for c, z in zip(self.cnt, self._cnt_zero):
                c.copy_(z)
        return flips

    def state_bytes(self) -> int:
        """Persistent training state: packed weights + counters (+ the scales and their Adam state)."""
        extra = 3 * self.log_s.numel() * 4 if self.learn_scale else 0
        return sum(t.numel() * t.element_size() for t in self.Wp + self.cnt) + extra

    @torch.no_grad()
    def logits(self, x):
        """Forward without keeping per-layer activations (evaluation)."""
        if self.ln_affine != "none":
            return self._fwd_aff(x, keep=False)[0]
        if self.residual:
            return self._forward_res(x, keep=False)[0]
        sc = self.scales() if self.learn_scale else None
        h = x
        for l in range(len(self.Wp) - 1):
            h = torch.relu(layer_norm(self._mm(h, l)))
            if sc is not None:
                h = h * sc[l]
        u = self._mm(h, len(self.Wp) - 1)
        return u if sc is None else u * sc[-1]

    @torch.no_grad()
    def loss(self, x, y, tau=4.0, bs=4096):
        """Mean cross-entropy of tau * logits (the training objective of the softmax signal)."""
        tot = 0.0
        for i in range(0, len(x), bs):
            tot += F.cross_entropy(tau * self.logits(x[i:i + bs]), y[i:i + bs], reduction="sum").item()
        return tot / len(x)

    @torch.no_grad()
    def accuracy(self, x, y, bs=4096):
        ok = 0
        for i in range(0, len(x), bs):
            ok += (self.logits(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
        return ok / len(x)


def perceptron_signal(logits, y, margin):
    B = logits.shape[0]
    ar = torch.arange(B, device=logits.device)
    tgt = logits[ar, y]
    wrong = logits.clone()
    wrong[ar, y] = -float("inf")
    best_wrong = wrong.argmax(1)
    active = (tgt - wrong.max(1).values) < margin * logits.std().clamp_min(1e-6)
    E = torch.zeros_like(logits)
    idx = torch.nonzero(active).squeeze(1)
    E[idx, y[idx]] = -1.0
    E[idx, best_wrong[idx]] = 1.0
    return E


def init_weights(dims, seed, dist="ternary"):
    """Same draw order as the CPU prototype (deep.run_C): manual_seed, data split, then one draw per layer.
    dist: "ternary" = uniform over {-1, 0, 1}; "binary" = uniform over {-1, 1} (no zeros)."""
    if dist == "binary":
        return [torch.randint(0, 2, (dims[i + 1], dims[i])) * 2 - 1 for i in range(len(dims) - 1)]
    assert dist == "ternary", dist
    return [torch.randint(-1, 2, (dims[i + 1], dims[i])) for i in range(len(dims) - 1)]


# ----------------------------------------------------------------------------------------------------------- runs
def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


class KController:
    """Validation-driven k (ReduceLROnPlateau-style with recovery). After each validation check:
    improved on the best by more than `tol` (relative): count an improvement, and after `grow_patience` consecutive
    improvements grow k by `grow` (up to k_max); otherwise count a miss, and after `patience` consecutive misses shrink k
    by `shrink` (down to k_min)."""

    def __init__(self, k, k_min=1e-4, k_max=0.02, tol=0.002, patience=2, grow_patience=3, shrink=0.5, grow=1.25):
        self.k, self.k_min, self.k_max, self.tol = k, k_min, k_max, tol
        self.patience, self.grow_patience, self.shrink, self.grow = patience, grow_patience, shrink, grow
        self.best, self.good, self.bad = float("inf"), 0, 0

    def update(self, val: float) -> float:
        if val < self.best * (1 - self.tol):
            self.best, self.bad = val, 0
            self.good += 1
            if self.good >= self.grow_patience:
                self.k, self.good = min(self.k * self.grow, self.k_max), 0
        else:
            self.bad += 1
            self.good = 0
            if self.bad >= self.patience:
                self.k, self.bad = max(self.k * self.shrink, self.k_min), 0
        return self.k

    def state(self):
        return dict(k=self.k, best=self.best, good=self.good, bad=self.bad)


class TController:
    """Validation-driven flip threshold T. Compares each validation loss with the previous check:
    increasing by more than `tol` (relative) -> T + 1 (flips need more consistent votes);
    decreasing by more than `tol` -> keep T; within +-tol (stalled) -> T - 1 (flips come easier). T in [t_min, t_max]."""

    def __init__(self, T, t_min=2, t_max=16, tol=0.002):
        self.T, self.t_min, self.t_max, self.tol = T, t_min, t_max, tol
        self.prev = None

    def update(self, val: float) -> int:
        if self.prev is not None:
            d = (val - self.prev) / max(abs(self.prev), 1e-12)
            if d > self.tol:
                self.T = min(self.T + 1, self.t_max)
            elif d >= -self.tol:
                self.T = max(self.T - 1, self.t_min)
        self.prev = val
        return self.T


class RevController:
    """Flip threshold T driven by the reversal rate (share of flips that undo a recent flip of the same weight), measured
    over the flips since the previous check: > hi -> T + 1 (oscillating), < lo -> T - 1 (flips stick), else keep."""

    def __init__(self, T, lo=0.08, hi=0.15, t_min=2, t_max=16, min_flips=200):
        self.T, self.lo, self.hi, self.t_min, self.t_max, self.min_flips = T, lo, hi, t_min, t_max, min_flips
        self.f0 = self.r0 = 0

    def update(self, flips_total: int, reversals_total: int):
        f, r = flips_total - self.f0, reversals_total - self.r0
        rate = None
        if f >= self.min_flips:  # too few flips: keep accumulating
            rate = r / f
            if rate > self.hi:
                self.T = min(self.T + 1, self.t_max)
            elif rate < self.lo:
                self.T = max(self.T - 1, self.t_min)
            self.f0, self.r0 = flips_total, reversals_total
        return self.T, rate

    def restart(self):
        self.f0 = self.r0 = 0


def run_tbp(dataset, depth, width, epochs, bs, seed, k_frac=0.005, T=8, signal="softmax", matmul="auto",
            device="cuda", log=None, counter_bits=4, p_inc=1.0, residual=False, learn_scale=False, scale_lr=1e-2,
            adaptive_k=False, val_split=0, val_every=0, k_min=1e-4, k_max=0.02, k_tol=0.002, patience=2,
            grow_patience=3, frac_k=None, counter_reset=0, sig_z=0.0, init_dist="ternary", t_schedule=False,
            t_min=2, t_max=16, t_tol=0.002, instrument=False, tau=2.0, counter_decay=0, cooldown=0,
            vote_strength=0.0, stagger=1, flip_every=1, rev_control=False, rev_lo=0.08, rev_hi=0.15,
            ln_affine="none", ln_lr=1e-3, exact_backward=False):
    """val_split > 0 holds out the last `val_split` training examples (same for every seed) for validation; adaptive_k
    then runs a KController on the validation loss every `val_every` steps (default: a quarter epoch)."""
    torch.manual_seed(seed)
    xtr, xte, ytr, yte = load(dataset, seed)
    if val_split:
        xtr, xva, ytr, yva = xtr[:-val_split], xtr[-val_split:], ytr[:-val_split], ytr[-val_split:]
    assert not (adaptive_k or t_schedule) or val_split, "adaptive k / T schedule need a validation split"
    if rev_control:
        instrument = True
        assert counter_bits == 8 or t_max <= (1 << (counter_bits - 1)), "rev_control with T > 8 needs counter_bits=8"
    if t_schedule and counter_bits != 8:
        assert t_max <= (1 << (counter_bits - 1)), f"t_max={t_max} needs counter_bits=8"
    n_cls = int(ytr.max()) + 1
    dims = [xtr.shape[1]] + [width] * depth + [n_cls]
    Ws = init_weights(dims, seed, init_dist)
    _ = [torch.randint(-1, 2, (n_cls, width)) for _ in range(depth)]  # prototype also draws its DFA matrices here
    frac_k = adaptive_k if frac_k is None else frac_k
    net = TBPNet(Ws, device, matmul, batch=bs, counter_bits=counter_bits, p_inc=p_inc, residual=residual,
                 learn_scale=learn_scale, scale_lr=scale_lr, frac_k=frac_k, counter_reset=counter_reset, sig_z=sig_z,
                 instrument=instrument, counter_decay=counter_decay, cooldown=cooldown, vote_strength=vote_strength,
                 stagger=stagger, flip_every=flip_every, ln_affine=ln_affine, ln_lr=ln_lr,
                 exact_backward=exact_backward)
    xtr_d, ytr_d, xte_d, yte_d = xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)
    if val_split:
        xva_d, yva_d = xva.to(device), yva.to(device)
    steps_per_epoch = (len(xtr) + bs - 1) // bs
    val_every = val_every or max(1, steps_per_epoch // 4)
    ctrl = KController(k_frac, k_min, k_max, k_tol, patience, grow_patience) if adaptive_k else None
    tctrl = TController(T, t_min, t_max, t_tol) if t_schedule else None
    rctrl = RevController(T, rev_lo, rev_hi, t_min, t_max) if rev_control else None
    k_now, k_trace, hist, step = k_frac, [], [], 0
    T_now = T
    sub = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(1234))[:5000]  # fixed train subset
    xsub_d, ysub_d = xtr[sub].to(device), ytr[sub].to(device)
    prev_loss = net.loss(xsub_d, ysub_d, tau=tau) if instrument else None
    for ep in range(epochs):
        perm = torch.randperm(len(xtr))
        _sync(device)
        t = time.time()
        flips = nb = 0
        for i in range(0, len(xtr), bs):
            b = perm[i:i + bs].to(device)
            flips += net.step(xtr_d[b], ytr_d[b], k_now, T_now, signal, tau=tau)
            nb += 1
            step += 1
            if rctrl is not None and step % val_every == 0:
                T_now, rate = rctrl.update(sum(net._st["flips"]), sum(net._st["reversals"]))
                k_trace.append(dict(step=step, reversal=rate, T=T_now, k=k_now))
            if (ctrl is not None or tctrl is not None) and step % val_every == 0:
                v = net.loss(xva_d, yva_d, tau=tau)
                if ctrl is not None:
                    k_now = ctrl.update(v)
                if tctrl is not None:
                    T_now = tctrl.update(v)
                k_trace.append(dict(step=step, val_loss=v, k=k_now, T=T_now))
        _sync(device)
        dt = time.time() - t
        W = net.weights()
        h = dict(epoch=ep + 1, train=net.accuracy(xtr_d, ytr_d), test=net.accuracy(xte_d, yte_d),
                 flips_per_step=flips / nb, epoch_seconds=dt, k=k_now, T=T_now,
                 zeros=float(sum((w == 0).sum().item() for w in W) / sum(w.numel() for w in W)))
        if val_split:
            h.update(val=net.accuracy(xva_d, yva_d), val_loss=net.loss(xva_d, yva_d, tau=tau))
        if learn_scale:
            h["scales"] = [round(v, 4) for v in net.scales().tolist()]
        if ln_affine != "none":
            gcat = torch.cat([t.detach() for t in net.ln_g])
            h["ln_gain"] = dict(mean=float(gcat.mean()), std=float(gcat.std()), min=float(gcat.min()),
                                max=float(gcat.max()))
            if ln_affine == "gain_bias":
                bcat = torch.cat([t.detach() for t in net.ln_b])
                h["ln_bias"] = dict(mean=float(bcat.mean()), std=float(bcat.std()))
        if instrument:
            st = net.stats(T_now)
            if rctrl is not None:
                rctrl.restart()  # the epoch stats were just reset
            cur = net.loss(xsub_d, ysub_d, tau=tau)
            nf = sum(st["flips"])
            h.update(train_loss=cur, productivity=(prev_loss - cur) / nf if nf else 0.0,
                     reversal_rate=st["reversal_rate_all"],
                     agreement=float(np.nanmean(st["agreement"])), occupancy=float(np.mean(st["occupancy"])),
                     layer_stats=st)
            prev_loss = cur
        hist.append(h)
        if log:
            log(h)
    if hist:
        hist[-1]["k_trace"] = k_trace
    return hist, net


def residual_forward(Ws, x):
    """Reference residual forward with float weights (STE baseline and tests)."""
    h = torch.relu(layer_norm(x @ Ws[0].T / math.sqrt(Ws[0].shape[1])))
    for W in Ws[1:-1]:
        h = h + torch.relu(layer_norm(h @ W.T / math.sqrt(W.shape[1])))
    return layer_norm(h) @ Ws[-1].T / math.sqrt(Ws[-1].shape[1])


def ternary_ste(w):
    s = w.abs().mean().clamp_min(1e-5)
    return w + ((w / s).round().clamp(-1, 1) - w).detach()


def _float_forward(Wq, x, residual):
    if residual:
        return residual_forward(Wq, x)
    h = x
    for W in Wq[:-1]:
        h = torch.relu(layer_norm(h @ W.T / math.sqrt(W.shape[1])))
    return h @ Wq[-1].T / math.sqrt(Wq[-1].shape[1])


def run_bop(dataset, depth, width, epochs, bs, seed, tau_flip=1.0, gamma=1e-3, delta=1e-3, device="cuda", log=None,
            residual=False, val_split=0, init_dist="ternary", bop_warmup=100, bop_k=0.0):
    """Ternary Bop baseline (Bop2ndOrder-style): exact backprop gradients g through the ternary network, per weight
    m <- (1-gamma) m + gamma g and v <- (1-delta) v + delta g^2; a weight steps one ternary state toward -sign(m)
    when |m_hat| / sqrt(v_hat) > tau_flip, with Adam-style bias correction of m and v (without it v starts near 0
    and nearly every weight passes the threshold in the first steps). No float master weights, but two fp32 states
    per weight (64 bits)."""
    torch.manual_seed(seed)
    xtr, xte, ytr, yte = load(dataset, seed)
    if val_split:
        xtr, ytr = xtr[:-val_split], ytr[:-val_split]
    n_cls = int(ytr.max()) + 1
    dims = [xtr.shape[1]] + [width] * depth + [n_cls]
    Ws = [W.float().to(device).requires_grad_(True) for W in init_weights(dims, seed, init_dist)]
    ms = [torch.zeros_like(W) for W in Ws]
    vs = [torch.zeros_like(W) for W in Ws]
    xtr_d, ytr_d, xte_d, yte_d = xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)

    def acc(x, y):
        with torch.no_grad():
            return sum((_float_forward(Ws, x[i:i + 4096], residual).argmax(1) == y[i:i + 4096]).sum().item()
                       for i in range(0, len(x), 4096)) / len(x)

    hist, t_step = [], 0
    for ep in range(epochs):
        perm = torch.randperm(len(xtr))
        _sync(device)
        t = time.time()
        flips = nb = 0
        for i in range(0, len(xtr), bs):
            b = perm[i:i + bs].to(device)
            t_step += 1
            bc_m, bc_v = 1 - (1 - gamma) ** t_step, 1 - (1 - delta) ** t_step
            loss = F.cross_entropy(_float_forward(Ws, xtr_d[b], residual) * 4.0, ytr_d[b])
            for W in Ws:
                W.grad = None
            loss.backward()
            with torch.no_grad():
                for W, m, v in zip(Ws, ms, vs):
                    g = W.grad
                    m.mul_(1 - gamma).add_(gamma * g)
                    v.mul_(1 - delta).add_(delta * g * g)
                    d = -torch.sign(m)
                    snr = (m.abs() / bc_m) / (v / bc_v).sqrt().clamp_min(1e-12)
                    go = (snr > tau_flip) & (((d > 0) & (W < 1)) | ((d < 0) & (W > -1)))
                    if bop_k > 0:  # same per-row flip budget as TBP: only the top bop_k fraction by snr may flip
                        n = max(1, int(round(bop_k * W.shape[1])))
                        sc = torch.where(go, snr, torch.full_like(snr, -1.0))
                        vals, idx = sc.topk(min(n, W.shape[1]), dim=1)
                        keep = torch.zeros_like(go)
                        keep.scatter_(1, idx, vals > 0)
                        go = go & keep
                    if t_step <= bop_warmup:  # let the moments settle before any flip
                        go = torch.zeros_like(go)
                    W.add_(torch.where(go, d, torch.zeros_like(d)))
                    m.masked_fill_(go, 0.0)  # a flipped weight starts accumulating again from zero
                    flips += int(go.sum())
            nb += 1
        _sync(device)
        nz = float(sum((W == 0).sum().item() for W in Ws) / sum(W.numel() for W in Ws))
        hist.append(dict(epoch=ep + 1, train=acc(xtr_d, ytr_d), test=acc(xte_d, yte_d), flips_per_step=flips / nb,
                         epoch_seconds=time.time() - t, k=0.0, zeros=nz))
        if log:
            log(hist[-1])
    return hist


def run_ste(dataset, depth, width, epochs, bs, seed, lr=3e-3, device="cuda", log=None, residual=False, val_split=0):
    """STE + Adam baseline on the GPU (float master weights), same forward as TBP (plain or residual)."""
    torch.manual_seed(seed)
    xtr, xte, ytr, yte = load(dataset, seed)
    if val_split:
        xtr, ytr = xtr[:-val_split], ytr[:-val_split]
    n_cls = int(ytr.max()) + 1
    dims = [xtr.shape[1]] + [width] * depth + [n_cls]
    Ws = [torch.nn.Parameter((torch.randn(dims[i + 1], dims[i]) * 0.1).to(device)) for i in range(len(dims) - 1)]
    opt = torch.optim.Adam(Ws, lr=lr)
    xtr_d, ytr_d, xte_d, yte_d = xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)

    def fwd(Wq, x):
        if residual:
            return residual_forward(Wq, x)
        h = x
        for W in Wq[:-1]:
            h = torch.relu(layer_norm(h @ W.T / math.sqrt(W.shape[1])))
        return h @ Wq[-1].T / math.sqrt(Wq[-1].shape[1])

    def acc(x, y):
        with torch.no_grad():
            q = [ternary_ste(W) for W in Ws]
            return sum((fwd(q, x[i:i + 4096]).argmax(1) == y[i:i + 4096]).sum().item()
                       for i in range(0, len(x), 4096)) / len(x)

    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(xtr))
        torch.cuda.synchronize()
        t = time.time()
        for i in range(0, len(xtr), bs):
            b = perm[i:i + bs].to(device)
            loss = F.cross_entropy(fwd([ternary_ste(W) for W in Ws], xtr_d[b]) * 4.0, ytr_d[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        hist.append(dict(epoch=ep + 1, train=acc(xtr_d, ytr_d), test=acc(xte_d, yte_d), epoch_seconds=time.time() - t))
        if log:
            log(hist[-1])
    return hist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["digits", "fmnist"], default="fmnist")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=512, help="batch size (512 beat 256 by +0.19 on both nets, 6 paired seeds, and is faster)")
    ap.add_argument("--k", type=float, default=0.005)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--signal", choices=["softmax", "perceptron"], default="softmax")
    ap.add_argument("--matmul", choices=["auto", "packed", "cublas"], default="auto")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--counter-bits", type=int, choices=[8, 4, 3], default=4,
                    help="4: packed 4-bit counters (T <= 8); 3: packed 3-bit counters (T <= 4)")
    ap.add_argument("--p-inc", type=float, default=1.0,
                    help="same-direction votes increment with this probability (3-bit, T=4, p=3/7 ~ T=8 on average)")
    ap.add_argument("--ste-width", type=int, default=0, help="also run STE+Adam at this width (same depth)")
    ap.add_argument("--residual", action="store_true", help="residual blocks (one ternary layer per block)")
    ap.add_argument("--method", choices=["tbp", "ste", "bop"], default="tbp")
    ap.add_argument("--ste-lr", type=float, default=3e-3)
    ap.add_argument("--bop-tau", type=float, default=1.0, help="Bop flip threshold on |m| / sqrt(v)")
    ap.add_argument("--bop-gamma", type=float, default=1e-2)
    ap.add_argument("--bop-k", type=float, default=0.0, help="per-row flip budget as in TBP (0 = unlimited)")
    ap.add_argument("--learn-scale", action="store_true", help="learnable fp32 scale per layer (after the LN)")
    ap.add_argument("--scale-lr", type=float, default=1e-2)
    ap.add_argument("--adaptive-k", action="store_true", help="validation-driven k (needs --val-split)")
    ap.add_argument("--val-split", type=int, default=0, help="hold out the last N training examples for validation")
    ap.add_argument("--val-every", type=int, default=0, help="steps between validation checks (0: quarter epoch)")
    ap.add_argument("--k-min", type=float, default=1e-4)
    ap.add_argument("--k-max", type=float, default=0.02)
    ap.add_argument("--k-tol", type=float, default=0.002)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--grow-patience", type=int, default=3)
    ap.add_argument("--device", default="cuda", help="cuda (CUDA kernels) or cpu (tbp.cpu_backend)")
    ap.add_argument("--counter-reset", type=int, default=0, help="reset all flip counters every N steps (0 = never)")
    ap.add_argument("--sig-z", type=float, default=0.0, help="vote only if |G| > z * row mean |G| (0 = off)")
    ap.add_argument("--t-schedule", action="store_true",
                    help="validation-driven T: loss up -> T+1, down -> keep, stalled -> T-1 (needs --val-split)")
    ap.add_argument("--t-min", type=int, default=2)
    ap.add_argument("--t-max", type=int, default=16)
    ap.add_argument("--t-tol", type=float, default=0.002, help="relative change counted as up/down vs stalled")
    ap.add_argument("--tau", type=float, default=2.0,
                    help="softmax temperature of the output signal (2 beat 4 on 6 paired seeds, both nets)")
    ap.add_argument("--counter-decay", type=int, default=0, help="every N steps move each counter one step toward 0")
    ap.add_argument("--cooldown", type=int, default=0, help="no flipping back within N steps of a weight's last flip")
    ap.add_argument("--vote-strength", type=float, default=0.0, help="votes with |G| > z * row mean count twice")
    ap.add_argument("--stagger", type=int, default=1, help="update one layer group of G per step")
    ap.add_argument("--flip-every", type=int, default=1, help="vote every step, flip every M steps (int8 counters)")
    ap.add_argument("--ln-affine", choices=["none", "gain", "gain_bias", "rms_gain"], default="none",
                    help="learnable LayerNorm parameters (fp32, Adam)")
    ap.add_argument("--ln-lr", type=float, default=1e-3)
    ap.add_argument("--exact-backward", action="store_true",
                    help="votes from the true weight gradient (autograd, LN statistics differentiated)")
    ap.add_argument("--rev-control", action="store_true", help="reversal-rate controller for T (see RevController)")
    ap.add_argument("--rev-lo", type=float, default=0.08)
    ap.add_argument("--rev-hi", type=float, default=0.15)
    ap.add_argument("--instrument", action="store_true",
                    help="log reversal rate, vote agreement, counter occupancy and flip productivity per epoch")
    ap.add_argument("--init-dist", choices=["ternary", "binary"], default="ternary",
                    help="initial weights uniform over {-1,0,1} (ternary) or {-1,1} (binary, no zeros)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.threads:
        torch.set_num_threads(a.threads)
    res = {}
    arch = "res" if a.residual else "plain"
    if a.method != "tbp":
        name = f"{a.method.upper()} {arch} L={a.depth} W={a.width}" + (f" lr={a.ste_lr:g}" if a.method == "ste" else
                                                                      f" tau={a.bop_tau:g} gamma={a.bop_gamma:g}")
        if a.method == "ste":
            runs = [run_ste(a.dataset, a.depth, a.width, a.epochs, a.bs, s, lr=a.ste_lr, residual=a.residual,
                            val_split=a.val_split, log=lambda h: print(f"  ste {h}", flush=True)) for s in seeds]
        else:
            runs = [run_bop(a.dataset, a.depth, a.width, a.epochs, a.bs, s, tau_flip=a.bop_tau, gamma=a.bop_gamma,
                            bop_k=a.bop_k,
                            residual=a.residual, val_split=a.val_split, init_dist=a.init_dist,
                            log=lambda h: print(f"  bop {h}", flush=True)) for s in seeds]
        for r in runs:
            for h in r:
                h.setdefault("k", 0.0)
                h.setdefault("flips_per_step", 0.0)
        if a.out:
            Path(a.out).write_text(json.dumps({name: runs}, indent=1))
        print(name, "final test", np.mean([r[-1]["test"] for r in runs]))
        return
    name = (f"TBP {arch} L={a.depth} W={a.width} k={a.k} T={a.T} c{a.counter_bits} {a.signal}"
            + (" +adaptive-k" if a.adaptive_k else "") + (" +scale" if a.learn_scale else "")
            + (f" +reset{a.counter_reset}" if a.counter_reset else "") + (f" +sig{a.sig_z:g}" if a.sig_z else "")
            + (" +binary-init" if a.init_dist == "binary" else "") + (" +T-schedule" if a.t_schedule else ""))
    res[name] = [run_tbp(a.dataset, a.depth, a.width, a.epochs, a.bs, s, a.k, a.T, a.signal, a.matmul,
                         log=lambda h: print(f"  tbp {h}", flush=True), counter_bits=a.counter_bits,
                         p_inc=a.p_inc, residual=a.residual, learn_scale=a.learn_scale, scale_lr=a.scale_lr,
                         adaptive_k=a.adaptive_k, val_split=a.val_split, val_every=a.val_every, k_min=a.k_min,
                         k_max=a.k_max, k_tol=a.k_tol, patience=a.patience, grow_patience=a.grow_patience,
                         device=a.device, counter_reset=a.counter_reset, sig_z=a.sig_z, init_dist=a.init_dist,
                         t_schedule=a.t_schedule, t_min=a.t_min, t_max=a.t_max, t_tol=a.t_tol,
                         instrument=a.instrument, tau=a.tau, counter_decay=a.counter_decay, cooldown=a.cooldown,
                         vote_strength=a.vote_strength, stagger=a.stagger, flip_every=a.flip_every,
                         rev_control=a.rev_control, rev_lo=a.rev_lo, rev_hi=a.rev_hi, ln_affine=a.ln_affine,
                         ln_lr=a.ln_lr, exact_backward=a.exact_backward)[0]
                 for s in seeds]
    if a.ste_width:
        res[f"STE+Adam {arch} L={a.depth} W={a.ste_width}"] = [
            run_ste(a.dataset, a.depth, a.ste_width, a.epochs, a.bs, s, log=lambda h: print(f"  ste {h}", flush=True),
                    residual=a.residual)
            for s in seeds]
    print(f"{'method':<52}{'best':>7}{'final':>7}{'train':>7}{'s/epoch':>9}")
    for n, runs in res.items():
        print(f"{n:<52}{np.mean([max(h['test'] for h in r) for r in runs]):>7.3f}"
              f"{np.mean([r[-1]['test'] for r in runs]):>7.3f}{np.mean([r[-1]['train'] for r in runs]):>7.3f}"
              f"{np.mean([np.mean([h['epoch_seconds'] for h in r]) for r in runs]):>9.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
