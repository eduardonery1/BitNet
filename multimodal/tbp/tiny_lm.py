"""TinyStories dress rehearsal: a tiny BitNet LM trained with TBP (ternary linears) vs STE + AdamW.

    cd multimodal
    runs/.venv-tbp/bin/python -m tbp.tiny_lm --method tbp --T 8 --steps 2000
    runs/.venv-tbp/bin/python -m tbp.tiny_lm --method ste --lr 3e-3 --steps 2000

Data: runs/tinystories/{train,val}.pt (int16 token ids, 4,096-token BPE). In TBP runs the ternary decoder linears are
trained by TBP while the embedding (tied LM head) and the norms are trained by Adam; in STE runs every parameter is
trained by AdamW through BitLinear (STE), as in BitNet. Results: runs/tinystories/<run>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bitnet_vl.bitlinear import replace_linears
from bitnet_vl.make_tiny_ckpt import tiny_llm_config
from bitnet_vl.modeling import LLM_LINEAR_RE, _strip_quant_config

from . import llm as L

DATA = Path(__file__).resolve().parents[1] / "runs" / "tinystories"


def build(a, device):
    from transformers import BitNetForCausalLM

    torch.manual_seed(a.seed)
    cfg = _strip_quant_config(tiny_llm_config(vocab_size=4096, hidden=a.hidden, intermediate=a.intermediate,
                                              layers=a.layers, heads=a.heads, kv_heads=a.kv_heads))
    cfg.bos_token_id = cfg.eos_token_id = cfg.pad_token_id = 2
    lm = BitNetForCausalLM(cfg).float()
    if a.method == "tbp":
        mods = L.convert_bitnet(lm, k_frac=a.k, T=a.T, counter_bits=a.counter_bits, matmul="cublas", device=device,
                                freeze_rest=False)
        if a.learn_scale != "none":
            print(f"learnable scales on {L.enable_learnable_scales(mods, a.learn_scale)} linears", flush=True)
        rest = [p for p in lm.parameters() if p.requires_grad]
        opt = torch.optim.Adam(rest, lr=a.lr)
    else:
        replace_linears(lm, lambda n, m: bool(LLM_LINEAR_RE.search(n)))
        lm.to(device)
        mods = []
        decay = [p for n, p in lm.named_parameters() if p.ndim == 2]
        nodecay = [p for n, p in lm.named_parameters() if p.ndim < 2]
        opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=nodecay, weight_decay=0.0)],
                                lr=a.lr, betas=(0.9, 0.95))
    return lm, mods, opt


def batches(ids, seq, bs, gen):
    n = len(ids) - seq - 1
    while True:
        starts = torch.randint(0, n, (bs,), generator=gen)
        yield torch.stack([ids[s:s + seq + 1] for s in starts.tolist()]).long()


@torch.no_grad()
def val_loss(lm, ids, seq, n_batches=40, bs=32, device="cuda"):
    gen = torch.Generator().manual_seed(999)  # same evaluation windows every time
    it = batches(ids, seq, bs, gen)
    tot = 0.0
    for _ in range(n_batches):
        b = next(it).to(device)
        logits = lm(input_ids=b[:, :-1]).logits
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), b[:, 1:].reshape(-1)).item()
    return tot / n_batches


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["tbp", "ste"], default="tbp")
    ap.add_argument("--run", default=None)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--intermediate", type=int, default=768)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--k", type=float, default=0.005)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--learn-scale", choices=["none", "od", "odqk", "all"], default="odqk",
                    help="learnable fp32 scale per ternary linear (o/down, + q/k, or all), trained by the Adam")
    ap.add_argument("--counter-bits", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam lr: embedding + norms (tbp) or everything (ste)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = a.run or (f"{a.method}_T{a.T}_k{a.k:g}" + (f"_sc{a.learn_scale}" if a.learn_scale != "none" else "")
                    + f"_s{a.seed}" if a.method == "tbp" else f"ste_lr{a.lr:g}_s{a.seed}")
    device = "cuda"
    train, val = torch.load(DATA / "train.pt"), torch.load(DATA / "val.pt")
    lm, mods, opt = build(a, device)
    lm.train()
    n_tern = sum(m.in_features * m.out_features for m in mods) if mods else \
        sum(p.numel() for n, p in lm.named_parameters() if LLM_LINEAR_RE.search(n.rsplit(".", 1)[0]))
    print(f"[{run}] ternary linear weights {n_tern / 1e6:.2f}M, other params "
          f"{sum(p.numel() for p in lm.parameters() if p.requires_grad) / 1e6:.2f}M", flush=True)
    gen = torch.Generator().manual_seed(a.seed)
    it = batches(train, a.seq, a.bs, gen)
    hist, t0, run_loss, n = [], time.time(), 0.0, 0
    hist.append(dict(step=0, val_loss=val_loss(lm, val, a.seq)))
    for step in range(1, a.steps + 1):
        f = min(1.0, step / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = a.lr * f
        for m in mods:
            m.step_seed = step
        b = next(it).to(device)
        logits = lm(input_ids=b[:, :-1]).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), b[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()  # TBPLinears update themselves here
        if a.method == "ste":
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        opt.step()
        run_loss, n = run_loss + loss.item(), n + 1
        if step % a.eval_every == 0 or step == a.steps:
            v = val_loss(lm, val, a.seq)
            flips = int(lm.tbp_flips.sum()) if mods else 0
            if mods:
                lm.tbp_flips.zero_()
            hist.append(dict(step=step, train_loss=run_loss / n, val_loss=v, flips_per_step=flips / a.eval_every,
                             tok_per_s=a.eval_every * a.bs * a.seq / (time.time() - t0)))
            print(f"[{run}] step {step} train {run_loss / n:.4f} val {v:.4f} (ppl {math.exp(v):.2f}) "
                  f"flips/step {flips / a.eval_every:.0f} {hist[-1]['tok_per_s']:.0f} tok/s", flush=True)
            t0, run_loss, n = time.time(), 0.0, 0
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    lm.eval()
    with torch.no_grad():
        ids = torch.tensor([tok.encode("Once upon a time").ids], device=device)
        for _ in range(60):
            nxt = lm(input_ids=ids).logits[0, -1].argmax()
            ids = torch.cat([ids, nxt.view(1, 1)], 1)
            if nxt.item() == 2:
                break
    sample = tok.decode(ids[0].tolist())
    print(f"[{run}] sample: {sample}", flush=True)
    (DATA / f"{run}.json").write_text(json.dumps(dict(config=vars(a), hist=hist, sample=sample), indent=1))


if __name__ == "__main__":
    main()
