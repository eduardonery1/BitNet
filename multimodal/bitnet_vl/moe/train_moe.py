"""Continued pretraining of an upcycled BitNet-MoE on plain text (docs/bitnet-moe-design.md, "Training").

    cd multimodal && python -m bitnet_vl.moe.train_moe --init-from MOE_DIR --data DATA --output-dir OUT \\
        [--seq-len 2048 --batch-size 8 --grad-accum 4 --max-steps 10000 --lr 2e-4 --router-lr 4e-4 ...]

DATA: a directory or file(s) of .jsonl (field ``--text-field``, default "text") / .txt / .md / .py / any text, or
``hf:NAME[:CONFIG]`` for a HF dataset (needs ``datasets``; streamed). Documents are tokenized, joined with EOS and packed
into fixed ``--seq-len`` blocks (no padding). Loss = LM + router_aux_loss_coef * L_aux + router_z_loss_coef * L_z
(coefs from the checkpoint's config.json unless overridden). Linear warmup + cosine decay for all groups; the router
group has its own LR and no weight decay. bf16 autocast on CUDA. Every ``--log-every`` steps prints the per-layer
expert-load histogram (fraction of top-k assignments per expert) and appends it to OUT/train_log.jsonl.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import time
from pathlib import Path

import torch

from ..train import _no_decay, set_schedule
from .modeling_moe import BitNetMoEForCausalLM

TEXT_EXTS = {".txt", ".md", ".py", ".rst", ".json", ".jsonl", ".c", ".cpp", ".h", ".sh", ".toml", ".yaml", ".yml"}


# --------------------------------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------------------------------
def iter_documents(spec: list[str], text_field: str = "text"):
    if len(spec) == 1 and spec[0].startswith("hf:"):
        import datasets  # optional dependency

        name, _, cfg = spec[0][3:].partition(":")
        ds = datasets.load_dataset(name, cfg or None, split="train", streaming=True)
        for row in ds:
            yield row[text_field]
        return
    files = []
    for s in spec:
        p = Path(s)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.is_file() and f.suffix in TEXT_EXTS)
        elif p.exists():
            files.append(p)
        else:
            files += sorted(Path(f) for f in glob.glob(s, recursive=True) if Path(f).is_file())
    if not files:
        raise FileNotFoundError(f"no text files in {spec}")
    for f in files:
        if f.suffix == ".jsonl":
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)[text_field]
        else:
            yield f.read_text(encoding="utf-8", errors="replace")


def pack(docs, tokenizer, seq_len: int):
    """Yield LongTensor blocks of exactly seq_len tokens (docs separated by EOS, BOS at each doc start)."""
    buf: list[int] = []
    eos = tokenizer.eos_token_id
    for d in docs:
        if not d:
            continue
        buf += tokenizer(d, add_special_tokens=True).input_ids + [eos]
        while len(buf) >= seq_len:
            yield torch.tensor(buf[:seq_len], dtype=torch.long)
            buf = buf[seq_len:]


def load_blocks(spec, tokenizer, seq_len, text_field="text", max_blocks=None) -> list[torch.Tensor]:
    return list(itertools.islice(pack(iter_documents(spec, text_field), tokenizer, seq_len), max_blocks))


def batches(blocks: list[torch.Tensor], batch_size: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    epoch = 0
    while True:
        perm = torch.randperm(len(blocks), generator=g)
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            yield torch.stack([blocks[j] for j in perm[i:i + batch_size]]), epoch
        epoch += 1


# --------------------------------------------------------------------------------------------------------------------
# optimizer
# --------------------------------------------------------------------------------------------------------------------
def build_optimizer(model: BitNetMoEForCausalLM, lr: float, router_lr: float, wd: float):
    router_ids = {id(p) for p in model.router_parameters()}
    dec, nodec, router = [], [], []
    seen = set()
    for n, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in router_ids:
            router.append(p)
        elif _no_decay(n, p):
            nodec.append(p)
        else:
            dec.append(p)
    groups = [
        {"params": dec, "lr": lr, "base_lr": lr, "weight_decay": wd, "use_wd": True, "name": "decay"},
        {"params": nodec, "lr": lr, "base_lr": lr, "weight_decay": 0.0, "use_wd": False, "name": "no_decay"},
        {"params": router, "lr": router_lr, "base_lr": router_lr, "weight_decay": 0.0, "use_wd": False,
         "name": "router"},
    ]
    groups = [g for g in groups if g["params"]]
    fused = torch.cuda.is_available()
    kw = dict(betas=(0.9, 0.95), eps=1e-8)
    return torch.optim.AdamW(groups, fused=True, **kw) if fused else torch.optim.AdamW(groups, **kw)


def format_load(frac: torch.Tensor) -> str:
    """frac: [L, E] fractions of top-k assignments."""
    return " | ".join(f"L{i}:" + ",".join(f"{v:.2f}" for v in row.tolist()) for i, row in enumerate(frac))


# --------------------------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init-from", required=True, help="BitNet-MoE checkpoint dir (e.g. from upcycle.py)")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-blocks", type=int, default=None, help="cap on packed blocks loaded (memory)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=1000, help="optimizer steps")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--router-lr", type=float, default=None, help="default: 2 x --lr")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--weight-decay-end", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=None, help="default: 3%% of max-steps")
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--aux-loss-coef", type=float, default=None, help="override config router_aux_loss_coef")
    ap.add_argument("--z-loss-coef", type=float, default=None, help="override config router_z_loss_coef")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--save-dtype", choices=["bf16", "f32"], default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args(argv)


def save(model, tokenizer, out_dir, save_dtype):
    model.save_pretrained(out_dir, tokenizer=tokenizer,
                          dtype=torch.bfloat16 if save_dtype == "bf16" else torch.float32)


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.init_from)
    model = BitNetMoEForCausalLM.from_pretrained(args.init_from, dtype=torch.float32).to(dev)
    if args.aux_loss_coef is not None:
        model.config.router_aux_loss_coef = args.aux_loss_coef
    if args.z_loss_coef is not None:
        model.config.router_z_loss_coef = args.z_loss_coef
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    t0 = time.time()
    blocks = load_blocks(args.data, tok, args.seq_len, args.text_field, args.max_blocks)
    n_tok = len(blocks) * args.seq_len
    print(f"data: {len(blocks)} blocks x {args.seq_len} = {n_tok / 1e6:.2f}M tokens ({time.time() - t0:.1f}s)")
    if len(blocks) < args.batch_size:
        raise ValueError(f"only {len(blocks)} blocks; need >= batch size {args.batch_size}")
    router_lr = args.router_lr if args.router_lr is not None else 2 * args.lr
    opt = build_optimizer(model, args.lr, router_lr, args.weight_decay)
    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, int(0.03 * args.max_steps))
    use_amp = dev.type == "cuda"
    it = batches(blocks, args.batch_size, args.seed)
    L, E = model.config.num_hidden_layers, model.config.num_experts
    load_acc = torch.zeros(L, E)
    acc = {"loss": 0.0, "lm": 0.0, "aux": 0.0, "z": 0.0, "n": 0}
    log_f = open(out / "train_log.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    for step in range(args.max_steps):
        set_schedule(opt, step, args.max_steps, warmup, args.weight_decay, args.weight_decay_end, args.min_lr_ratio)
        for _ in range(args.grad_accum):
            ids, epoch = next(it)
            ids = ids.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                o = model(input_ids=ids, labels=ids)
            (o.loss / args.grad_accum).backward()
            acc["loss"] += o.loss.item()
            acc["lm"] += o.lm_loss.item()
            acc["aux"] += o.aux_loss.item()
            acc["z"] += o.z_loss.item()
            acc["n"] += 1
            load_acc += o.expert_counts.float().cpu()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % args.log_every == 0 or step == args.max_steps - 1:
            n = acc["n"]
            frac = load_acc / load_acc.sum(-1, keepdim=True).clamp(min=1)
            rec = {"step": step + 1, "epoch": epoch, "loss": acc["loss"] / n, "lm_loss": acc["lm"] / n,
                   "aux_loss": acc["aux"] / n, "z_loss": acc["z"] / n, "grad_norm": float(gn),
                   "lr": opt.param_groups[0]["lr"], "router_lr": opt.param_groups[-1]["lr"],
                   "expert_load": frac.tolist(), "max_load": float(frac.max()), "min_load": float(frac.min()),
                   "elapsed_s": time.time() - t0}
            print(f"step {step + 1:5d} loss {rec['loss']:.4f} lm {rec['lm_loss']:.4f} aux {rec['aux_loss']:.4f} "
                  f"z {rec['z_loss']:.3f} gn {rec['grad_norm']:.2f} lr {rec['lr']:.2e} | load {format_load(frac)}",
                  flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            load_acc.zero_()
            acc = {k: 0 for k in acc}
        if args.save_every and (step + 1) % args.save_every == 0:
            save(model, tok, out / f"step{step + 1}", args.save_dtype)
    log_f.close()
    save(model, tok, out, args.save_dtype)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
