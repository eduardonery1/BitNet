"""Continued pretraining of BitNet b1.58 2B4T with Ternary Backpropagation on a small GPU.

    cd multimodal
    PY=runs/.venv-tbp/bin/python
    $PY -m tbp.train_lm parity-ref        # CPU fp32 reference logits (HF model + repo BitLinear)
    $PY -m tbp.train_lm parity            # converted GPU model vs the reference
    $PY -m tbp.train_lm data              # stream FineWeb-Edu, tokenize, pack (train + held-out)
    $PY -m tbp.train_lm smoke             # 20 steps: memory and tokens/s
    $PY -m tbp.train_lm train --run tbp_k5e-4 --k 0.0005 --steps 600
    $PY -m tbp.train_lm train --run scratch_emb --init scratch-pretrained-emb --steps 5000
    $PY -m tbp.train_lm train --run scratch_emb_b2 --init scratch-pretrained-emb --batch 2 --steps 2500
    $PY -m tbp.train_lm train --run random --control random --replay runs/tbp_2b4t/tbp_k5e-4/flips.npy

Outputs under runs/tbp_2b4t/<run>/: log.jsonl, flips.npy (per step x per linear), ckpt.pt (packed weights + counters).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from bitnet_vl.modeling import _strip_quant_config

from . import llm as L

MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"
ROOT = Path(__file__).resolve().parents[1] / "runs"
OUT = ROOT / "tbp_2b4t"
DATA = ROOT / "lm_data"
PROMPTS = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
           "Photosynthesis is the process by which plants"]


def load_hf(dtype=torch.bfloat16):
    from transformers import AutoConfig, AutoTokenizer, BitNetForCausalLM

    cfg = _strip_quant_config(AutoConfig.from_pretrained(MODEL))
    lm = BitNetForCausalLM.from_pretrained(MODEL, config=cfg, dtype=dtype)
    return lm.eval(), AutoTokenizer.from_pretrained(MODEL)

INITS = ("pretrained", "scratch", "scratch-pretrained-emb")
# copied from the checkpoint by scratch-pretrained-emb: the tied embedding / LM head, the final norm (its weights,
# mean 0.1, set the logit scale the embedding was trained for) and the norms acting on the residual stream (the
# embedding's channels). Sub-norms act on channels produced by the (random) ternary linears and stay at 1.
PRETRAINED_EMB_KEYS = re.compile(r"^model\.(embed_tokens\.weight|norm\.weight|"
                                 r"layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight)$")


def load_hf_scratch(dtype=torch.bfloat16, seed=0):
    """Randomly initialized 2B4T (HF init, normal(0, 0.02)), built directly in `dtype` (no fp32 copy)."""
    from transformers import AutoConfig, AutoTokenizer, BitNetForCausalLM

    cfg = _strip_quant_config(AutoConfig.from_pretrained(MODEL))
    torch.manual_seed(seed)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        lm = BitNetForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev)
    return lm.eval(), AutoTokenizer.from_pretrained(MODEL)


def copy_pretrained_emb(lm) -> list[str]:
    """Overwrite the embedding, final norm and residual-stream norms of `lm` with the checkpoint's values."""
    import glob

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = glob.glob(f"{snapshot_download(MODEL)}/*.safetensors")
    sd = lm.state_dict()
    copied = []
    for f in path:
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                if PRETRAINED_EMB_KEYS.match(k):
                    with torch.no_grad():
                        sd[k].copy_(fh.get_tensor(k).to(sd[k].dtype))
                    copied.append(k)
    assert "model.embed_tokens.weight" in copied and "model.norm.weight" in copied, copied[:5]
    return copied


def load_tbp(k=0.0005, T=8, counter_bits=4, seq=512, checkpointing=True, init="pretrained", seed=0,
             learn_scale="none"):
    """init: "pretrained" (2B4T weights), "scratch" (all random) or "scratch-pretrained-emb" (random ternary linears,
    pretrained embedding + final and residual-stream norms)."""
    assert init in INITS, init
    if init == "pretrained":
        lm, tok = load_hf(torch.bfloat16)
    else:
        lm, tok = load_hf_scratch(torch.bfloat16, seed)
        if init == "scratch-pretrained-emb":
            n = len(copy_pretrained_emb(lm))
            print(f"copied {n} pretrained tensors (embedding, final norm, residual-stream norms)", flush=True)
    mods = L.convert_bitnet(lm, k_frac=k, T=T, counter_bits=counter_bits)
    if learn_scale != "none":  # learnable fp32 scales, trained by their own Adam (see cmd_train)
        L.enable_learnable_scales(mods, learn_scale)
    choice = L.autotune(mods, seq)
    if checkpointing:
        lm.train()
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return lm, tok, choice


# --------------------------------------------------------------------------------------------------- parity
def cmd_parity_ref(a):
    from bitnet_vl.bitlinear import replace_linears
    from bitnet_vl.modeling import LLM_LINEAR_RE

    lm, tok = load_hf(torch.float32)
    replace_linears(lm, lambda n, m: bool(LLM_LINEAR_RE.search(n)))
    out = []
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids
            out.append(dict(ids=ids[0], logits=lm(input_ids=ids).logits[0].float()))
            print(p[:30], "->", repr(tok.decode(out[-1]["logits"][-1].argmax())), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT / "parity_ref.pt")


def cmd_parity(a):
    ref = torch.load(OUT / "parity_ref.pt")
    lm, tok, choice = load_tbp(checkpointing=False, init="pretrained")  # the reference logits are the pretrained model
    print("matmul modes:", choice)
    for r in ref:
        with torch.no_grad():
            ids = r["ids"].cuda()
            h = lm.model(inputs_embeds=lm.model.embed_tokens(ids[None]).float()).last_hidden_state[0]
            got = (h.to(L.head_weight(lm).dtype) @ L.head_weight(lm).T).float().cpu()
        want = r["logits"]
        rel = ((got - want).norm() / want.norm()).item()
        top1 = (got.argmax(1) == want.argmax(1)).float().mean().item()
        print(f"{tok.decode(r['ids'][:8])!r}: rel err {rel:.2e}, top-1 agreement {top1:.3f}, next "
              f"{tok.decode(got[-1].argmax())!r} vs {tok.decode(want[-1].argmax())!r}")
    print(f"GPU memory after conversion: {torch.cuda.memory_allocated() / 2**20:.0f} MiB")


# --------------------------------------------------------------------------------------------------- data
def cmd_data(a):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    L1 = a.seq + 1
    need_eval, need_train = a.eval_seqs, a.train_tokens // L1
    eval_seqs, train_seqs, buf = [], [], []
    n_docs = 0
    for doc in ds:
        n_docs += 1
        buf.extend(tok(doc["text"]).input_ids + [tok.eos_token_id])
        while len(buf) >= L1 and len(train_seqs) < need_train:
            seq, buf = buf[:L1], buf[L1:]
            if len(eval_seqs) < need_eval:  # held-out = the first documents
                eval_seqs.append(seq)
                if len(eval_seqs) == need_eval:
                    buf = []  # training starts at a fresh document: no sequence spans the boundary
            else:
                train_seqs.append(seq)
        if len(train_seqs) >= need_train:
            break
        if n_docs % 2000 == 0:
            print(f"{n_docs} docs, eval {len(eval_seqs)}, train {len(train_seqs)}", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(eval_seqs, dtype=torch.int32), DATA / "eval.pt")
    torch.save(torch.tensor(train_seqs, dtype=torch.int32), DATA / "train.pt")
    print(f"saved {len(eval_seqs)} held-out and {len(train_seqs)} train sequences of {L1} tokens from {n_docs} docs")


# --------------------------------------------------------------------------------------------------- train
def gpu_mem_used_mb():
    import subprocess

    try:
        return float(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        return float("nan")


def cmd_smoke(a):
    lm, tok, choice = load_tbp(a.k, a.T, a.counter_bits, a.seq, init=a.init, seed=a.seed)
    print("matmul modes:", choice, f"| model on GPU {torch.cuda.memory_allocated() / 2**20:.0f} MiB", flush=True)
    torch.manual_seed(0)
    n = a.steps * a.batch
    seqs = torch.load(DATA / "train.pt")[:n] if (DATA / "train.pt").exists() else \
        torch.randint(0, 128000, (n, a.seq + 1), dtype=torch.int32)
    seqs = seqs[:len(seqs) // a.batch * a.batch].view(-1, a.batch, seqs.shape[1])
    torch.cuda.reset_peak_memory_stats()
    for i, s in enumerate(seqs):
        torch.cuda.synchronize()
        t = time.time()
        tot, n = L.tbp_step(lm, s.long().cuda(), a.chunk, i)
        torch.cuda.synchronize()
        dt = time.time() - t
        print(f"step {i}: loss {tot / n:.3f}  {dt:.2f}s  {n / dt:.0f} tok/s  flips {int(lm.tbp_flips.sum())}  "
              f"peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB  nvidia-smi {gpu_mem_used_mb():.0f} MiB",
              flush=True)


def save_ckpt(lm, path: Path, step: int, init: str):
    tmp = path.with_suffix(".tmp")
    torch.save(dict(step=step, init=init, Wp=[m.Wp.cpu() for m in lm.tbp_linears], cnt=[m.cnt.cpu() for m in lm.tbp_linears],
                    names=[m.name for m in lm.tbp_linears]), tmp)
    tmp.replace(path)


def load_ckpt(lm, path: Path, init: str) -> int:
    ck = torch.load(path)
    if ck.get("init", "unknown") != init:
        raise ValueError(f"{path} was trained with --init {ck.get('init', 'unknown')}, not {init}: use another --run")
    for m, wp, c, n in zip(lm.tbp_linears, ck["Wp"], ck["cnt"], ck["names"]):
        assert m.name == n
        m.Wp.copy_(wp)
        m.cnt.copy_(c)
    return ck["step"]


def cmd_train(a):
    run = OUT / a.run
    run.mkdir(parents=True, exist_ok=True)
    lm, tok, choice = load_tbp(a.k, a.T, a.counter_bits, a.seq, init=a.init, seed=a.seed, learn_scale=a.learn_scale)
    scale_params = [m.log_scale for m in lm.tbp_linears if m.log_scale is not None]
    if a.k_schedule != "const" or a.frac_k:
        for m in lm.tbp_linears:
            m.frac_k = True
    best_val = float("inf")
    scale_opt = torch.optim.Adam(scale_params, lr=a.scale_lr) if scale_params else None
    train = torch.load(DATA / "train.pt")
    ev = torch.load(DATA / "eval.pt")
    if a.split:  # validation (decisions, early stop) and test (reported) come from disjoint held-out sequences
        held = [s.long().cuda() for s in ev[:a.val_seqs]]
        test = [s.long().cuda() for s in ev[len(ev) - a.test_seqs:]]
        assert a.val_seqs + a.test_seqs <= len(ev), "validation and test sequences overlap"
    else:
        held = [s.long().cuda() for s in ev[:a.eval_seqs]]
        test = None
    replay = np.load(a.replay) if a.control == "random" else None
    if replay is not None:
        for m in lm.tbp_linears:
            m.random_control = True
    log = open(run / "log.jsonl", "a")
    start = load_ckpt(lm, run / "ckpt.pt", a.init) if (run / "ckpt.pt").exists() else 0
    flips_hist = list(np.load(run / "flips.npy")) if (run / "flips.npy").exists() and start else []

    def emit(d):
        d.update(step=step, time=time.time())
        log.write(json.dumps(d) + "\n")
        log.flush()
        print(json.dumps(d), flush=True)

    step = start
    if start == 0:
        d0 = dict(eval_ppl=L.eval_ppl(lm, held, a.chunk), config=vars(a), matmul=str(choice))
        if test is not None:
            d0["test_ppl"] = L.eval_ppl(lm, test, a.chunk)
        emit(d0)
    t0, tok_count, loss_sum, loss_n = time.time(), 0, 0.0, 0
    ppl0 = None
    while step < a.steps:
        idx = (torch.arange(a.batch) + step * a.batch) % len(train)  # sequences step*B .. step*B+B-1, wrapping
        ids = train[idx].long().cuda()
        if replay is not None:
            for i, m in enumerate(lm.tbp_linears):
                m.replay = int(replay[step, i]) if step < len(replay) else 0
        if scale_opt is not None:
            scale_opt.zero_grad(set_to_none=True)
        tot, n = L.tbp_step(lm, ids, a.chunk, step)
        if scale_opt is not None:
            scale_opt.step()
        f = lm.tbp_flips.cpu().numpy().copy()
        lm.tbp_flips.zero_()
        if replay is not None:
            f = replay[step] if step < len(replay) else np.zeros_like(f)
        flips_hist.append(f)
        loss_sum, loss_n, tok_count = loss_sum + tot, loss_n + n, tok_count + n
        step += 1
        if a.k_schedule != "const":  # fractional per-row k decaying from --k to --k-end
            prog = step / a.steps
            w = 0.5 * (1 + math.cos(math.pi * prog)) if a.k_schedule == "cosine" else 1 - prog
            kk = a.k_end + (a.k - a.k_end) * w
            for m in lm.tbp_linears:
                m.k_frac = kk
        if a.k_decay:  # optional time-based decay (n_k still floors at one weight per row)
            current_k = a.k * (0.75 - 0.75 * step / a.steps)
            for m in lm.tbp_linears:
                m.k_frac = current_k
        
        if step % a.log_every == 0:
            dt = time.time() - t0
            by_type = {}
            recent = np.sum(flips_hist[-a.log_every:], 0)
            for m, c in zip(lm.tbp_linears, recent):
                key = m.name.rsplit(".", 1)[1]
                by_type[key] = by_type.get(key, 0) + int(c)
            emit(dict(train_loss=loss_sum / loss_n, tok_per_s=tok_count / dt, flips=by_type,
                      peak_mb=torch.cuda.max_memory_allocated() / 2**20, smi_mb=gpu_mem_used_mb()))
            t0, tok_count, loss_sum, loss_n = time.time(), 0, 0.0, 0
        if step % a.eval_every == 0 or step == a.steps:
            ppl = L.eval_ppl(lm, held, a.chunk)
            dd = dict(eval_ppl=ppl)
            if test is not None and (step % a.test_every == 0 or step == a.steps):
                dd["test_ppl"] = L.eval_ppl(lm, test, a.chunk)
            emit(dd)
            np.save(run / "flips.npy", np.array(flips_hist))
            save_ckpt(lm, run / "ckpt.pt", step, a.init)
            if ppl < best_val:  # keep the best validation checkpoint for the final test
                best_val = ppl
                save_ckpt(lm, run / "best.pt", step, a.init)
                emit(dict(best_val=ppl, best_step=step))
            if ppl0 is None:
                first = [json.loads(x) for x in open(run / "log.jsonl") if '"eval_ppl"' in x]
                ppl0 = first[0]["eval_ppl"]
            if ppl > ppl0 * (1 + a.max_ppl_rise):
                emit(dict(stopped=f"held-out perplexity {ppl:.3f} > {1 + a.max_ppl_rise:.2f} x step-0 {ppl0:.3f}"))
                break
            t0 = time.time()
    np.save(run / "flips.npy", np.array(flips_hist))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["parity-ref", "parity", "data", "smoke", "train"])
    ap.add_argument("--run", default="tbp")
    ap.add_argument("--init", choices=INITS, default="pretrained")
    ap.add_argument("--seed", type=int, default=0, help="random init seed (scratch inits)")
    ap.add_argument("--k-decay", action="store_true", help="time-based k decay: k * (0.75 - 0.75 * step / steps)")
    ap.add_argument("--k", type=float, default=0.0005)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--counter-bits", type=int, default=4, choices=[8, 4, 3])
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1, help="sequences per TBP step (2 x 512 fits a 4 GB GPU)")
    ap.add_argument("--chunk", type=int, default=64, help="LM-head loss chunk (tokens)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--eval-seqs", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--k-schedule", choices=["const", "cosine", "linear"], default="const",
                    help="decay k from --k to --k-end over the run (fractional per-row k)")
    ap.add_argument("--k-end", type=float, default=0.00005)
    ap.add_argument("--frac-k", action="store_true", help="fractional per-row k (mean k * fan-in, can be < 1)")
    ap.add_argument("--learn-scale", choices=["none", "od", "odqk", "all"], default="none",
                    help="learnable fp32 scales (o/down, + q/k, or all linears), Adam --scale-lr")
    ap.add_argument("--scale-lr", type=float, default=1e-3)
    ap.add_argument("--split", action="store_true",
                    help="validation (--val-seqs, first) / test (--test-seqs, last) split of the held-out sequences")
    ap.add_argument("--val-seqs", type=int, default=50)
    ap.add_argument("--test-seqs", type=int, default=100)
    ap.add_argument("--test-every", type=int, default=500, help="steps between test evaluations (with --split)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-ppl-rise", type=float, default=0.05)
    ap.add_argument("--control", choices=["none", "random"], default="none")
    ap.add_argument("--replay", default=None, help="flips.npy of a TBP run (random control replays its counts)")
    ap.add_argument("--train-tokens", type=int, default=5_000_000)
    a = ap.parse_args()
    if a.cmd == "data" and a.eval_seqs == 100:
        a.eval_seqs = 200
    {"parity-ref": cmd_parity_ref, "parity": cmd_parity, "data": cmd_data, "smoke": cmd_smoke,
     "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
