"""BitNet-VL training (stages 1-3 of docs/bitnet-vl-design.md).

    python -m bitnet_vl.train --stage 1 --data blip_laion_cc_sbu_558k.json --image-folder LLaVA-Pretrain/images \\
        --vision-model google/siglip-base-patch16-384 --llm-model microsoft/bitnet-b1.58-2B-4T-bf16 --output-dir ckpt/s1
    python -m bitnet_vl.train --stage 2 --init-from ckpt/s1 --data llava_v1_5_mix665k.json --image-folder data ...
    python -m bitnet_vl.train --stage 3 --init-from ckpt/s2 --data subset.json --image-folder data ...

Plain PyTorch loop; multi-GPU via ``torchrun --nproc_per_node N -m bitnet_vl.train ...`` (DDP). bf16 autocast on CUDA.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from .data import Collator, LlavaDataset
from .modeling import BitNetVLForConditionalGeneration

# recipe table defaults: lr per group, weight decay start/end
STAGE_DEFAULTS = {
    1: dict(lr={"projector": 1e-3}, wd=(0.0, 0.0), epochs=1),
    2: dict(lr={"projector": 1e-3, "llm": 3e-4}, wd=(0.1, 0.0), epochs=1),
    3: dict(lr={"vision": 1e-4}, wd=(0.01, 0.01), epochs=1),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--data", required=True, help="LLaVA-format .json (list) or .jsonl")
    ap.add_argument("--image-folder", default=".")
    ap.add_argument("--init-from", help="BitNet-VL checkpoint dir (required for stages 2 and 3)")
    ap.add_argument("--vision-model", default="google/siglip-base-patch16-384")
    ap.add_argument("--llm-model", default="microsoft/bitnet-b1.58-2B-4T-bf16")
    ap.add_argument("--teacher-from", help="stage 3: checkpoint dir whose (non-ternary) vision tower is the teacher; "
                                           "defaults to --init-from when that tower is still full precision")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=-1, help="optimizer steps; overrides --epochs if > 0")
    ap.add_argument("--batch-size", type=int, default=8, help="per-device micro batch")
    ap.add_argument("--grad-accum", type=int, default=1, help="global batch = batch-size * grad-accum * world")
    ap.add_argument("--lr-projector", type=float)
    ap.add_argument("--lr-llm", type=float)
    ap.add_argument("--lr-vision", type=float)
    ap.add_argument("--weight-decay", type=float, help="start WD (default from recipe)")
    ap.add_argument("--weight-decay-end", type=float, help="final WD, linear schedule (default from recipe)")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--min-lr-ratio", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--distill-weight", type=float, default=0.1)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=0, help="save every N optimizer steps (0: only at the end)")
    ap.add_argument("--save-dtype", choices=["bf16", "f32"], default="bf16", help="dtype of language_model/ weights")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------------------------------------------
def load_model(args) -> BitNetVLForConditionalGeneration:
    if args.init_from:
        return BitNetVLForConditionalGeneration.from_pretrained(args.init_from)
    if args.stage != 1:
        raise SystemExit("--init-from is required for stages 2 and 3")
    return BitNetVLForConditionalGeneration.from_components(args.vision_model, args.llm_model)


def build_teacher(model, args):
    """Frozen bf16 copy of the full-precision vision tower (must be taken before ternarization)."""
    if not model.config.vision_ternary:
        teacher = copy.deepcopy(model.vision_tower)
    else:
        src = args.teacher_from
        if not src:
            raise SystemExit("vision tower is already ternary: pass --teacher-from <stage-2 ckpt>")
        teacher = BitNetVLForConditionalGeneration.from_pretrained(src).vision_tower
        if any(type(m).__name__ == "BitLinear" for m in teacher.modules()):
            raise SystemExit("--teacher-from checkpoint has a ternary vision tower")
    teacher = teacher.to(torch.bfloat16).eval().requires_grad_(False)
    return teacher


def _no_decay(name: str, p) -> bool:
    return p.ndim < 2 or "norm" in name or "embed" in name or name.endswith(".bias")


def build_optimizer(model, stage: int, lrs: dict, wd: float):
    groups = model.configure_stage(stage)
    names = {id(p): n for n, p in model.named_parameters()}
    param_groups = []
    for g, params in groups.items():
        if g not in lrs:
            raise ValueError(f"no LR for trainable group {g}")
        dec = [p for p in params if not _no_decay(names[id(p)], p)]
        nodec = [p for p in params if _no_decay(names[id(p)], p)]
        for ps, use_wd in ((dec, True), (nodec, False)):
            if ps:
                param_groups.append({"params": ps, "lr": lrs[g], "base_lr": lrs[g], "weight_decay": wd if use_wd else 0.0,
                                     "use_wd": use_wd, "name": f"{g}{'' if use_wd else '_nodecay'}"})
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8, fused=fused) if fused else \
        torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)


def set_schedule(opt, step: int, total: int, warmup: int, wd_start: float, wd_end: float, min_lr_ratio=0.0):
    """Linear warmup + cosine decay of LR; linear WD schedule wd_start -> wd_end."""
    if step < warmup:
        f = (step + 1) / max(1, warmup)
    else:
        prog = (step - warmup) / max(1, total - warmup)
        f = min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    wd = wd_start + (wd_end - wd_start) * min(1.0, step / max(1, total))
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * f
        if g["use_wd"]:
            g["weight_decay"] = wd


def distill_loss(student_hs, teacher_hs) -> torch.Tensor:
    """mean over encoder layers of MSE(h_teacher, h_student)."""
    assert len(student_hs) == len(teacher_hs)
    return torch.stack([F.mse_loss(s.float(), t.float()) for s, t in zip(student_hs, teacher_hs)]).mean()


def compute_loss(model, batch, stage: int, teacher=None, distill_weight: float = 0.1):
    """Returns (total_loss, logs). `model` may be DDP-wrapped."""
    distill = stage == 3 and teacher is not None and batch.get("pixel_values") is not None
    out = model(input_ids=batch["input_ids"], pixel_values=batch.get("pixel_values"),
                attention_mask=batch["attention_mask"], labels=batch["labels"], output_vision_hidden_states=distill)
    loss = out.loss
    logs = {"lm": loss.detach()}
    if distill:
        with torch.no_grad():
            pv = batch["pixel_values"].to(torch.bfloat16)
            t_hs = teacher(pixel_values=pv, output_hidden_states=True).hidden_states[1:]
        d = distill_loss(out.vision_hidden_states, t_hs)
        loss = loss + distill_weight * d
        logs["distill"] = d.detach()
    return loss, logs


# ----------------------------------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        torch.distributed.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
    else:
        rank, world, local = 0, 1, 0
    device = torch.device(args.device or (f"cuda:{local}" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed + rank)
    is_main = rank == 0

    d = STAGE_DEFAULTS[args.stage]
    lrs = dict(d["lr"])
    for g, v in (("projector", args.lr_projector), ("llm", args.lr_llm), ("vision", args.lr_vision)):
        if v is not None and g in lrs:
            lrs[g] = v
    wd_start = d["wd"][0] if args.weight_decay is None else args.weight_decay
    wd_end = d["wd"][1] if args.weight_decay_end is None else args.weight_decay_end
    epochs = d["epochs"] if args.epochs is None else args.epochs

    model = load_model(args)
    teacher = build_teacher(model, args).to(device) if args.stage == 3 else None
    model.to(device)
    opt = build_optimizer(model, args.stage, lrs, wd_start)  # also ternarizes the ViT for stage 3
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    if is_main:
        n_train = sum(p.numel() for g in opt.param_groups for p in g["params"])
        print(f"stage {args.stage}: trainable params {n_train / 1e6:.1f}M, groups "
              f"{[(g['name'], g['lr']) for g in opt.param_groups]}, wd {wd_start}->{wd_end}", flush=True)

    tok = model.tokenizer
    ds = LlavaDataset(args.data, tok, args.image_folder, model.config.image_size, args.max_length,
                      model.config.num_image_tokens, model.config.image_mean, model.config.image_std)
    sampler = DistributedSampler(ds, shuffle=True, seed=args.seed) if ddp else None
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                    num_workers=args.num_workers, collate_fn=Collator(tok.pad_token_id), drop_last=True,
                    pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    steps_per_epoch = max(1, len(dl) // args.grad_accum)
    total = args.max_steps if args.max_steps > 0 else max(1, int(steps_per_epoch * epochs))
    warmup = int(args.warmup_ratio * total)
    wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if device.type == "cuda" else None) \
        if ddp else model

    use_amp = device.type == "cuda"
    step, micro, t0 = 0, 0, time.time()
    acc_logs: dict[str, float] = {}
    epoch = 0
    done = False
    while not done:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dl:
            if (batch["labels"] != -100).sum() == 0:
                continue
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            set_schedule(opt, step, total, warmup, wd_start, wd_end, args.min_lr_ratio)
            sync = (micro + 1) % args.grad_accum == 0
            ctx = wrapped.no_sync() if (ddp and not sync) else _null()
            with ctx, torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss, logs = compute_loss(wrapped, batch, args.stage, teacher, args.distill_weight)
                (loss / args.grad_accum).backward()
            for k, v in logs.items():
                acc_logs[k] = acc_logs.get(k, 0.0) + float(v) / args.grad_accum
            micro += 1
            if not sync:
                continue
            params = [p for g in opt.param_groups for p in g["params"]]
            gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if is_main and (step % args.log_every == 0 or step == 1):
                lr = opt.param_groups[0]["lr"]
                msg = " ".join(f"{k}={v:.4f}" for k, v in acc_logs.items())
                print(f"step {step}/{total} {msg} gnorm={float(gn):.3f} lr={lr:.2e} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            acc_logs = {}
            if is_main and args.save_every and step % args.save_every == 0:
                save(model, os.path.join(args.output_dir, f"step-{step}"), args.save_dtype)
            if step >= total:
                done = True
                break
        epoch += 1
    if is_main:
        save(model, args.output_dir, args.save_dtype)
        print(f"saved {args.output_dir}", flush=True)
    if ddp:
        torch.distributed.destroy_process_group()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def save(model, out_dir, save_dtype="bf16"):
    model.save_pretrained(out_dir, llm_dtype=torch.bfloat16 if save_dtype == "bf16" else None)


if __name__ == "__main__":
    main()
