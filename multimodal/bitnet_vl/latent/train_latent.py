"""Train / evaluate the latent-reasoning experiment.

    cd multimodal
    # tiny random BitNet on CPU (sanity check; trains all weights)
    python -m bitnet_vl.latent.train_latent --model tiny --train-base --mode latent --R 2 --K 2 \\
        --hops-train 2-3 --hops-test 2-3 --steps 1500 --out runs/tiny_latent
    # real model (GPU), LoRA + compressor
    python -m bitnet_vl.latent.train_latent --model microsoft/bitnet-b1.58-2B-4T-bf16 --dtype bf16 \\
        --mode latent --curriculum --steps-per-stage 300 --R 6 --K 4 --hops-train 2-6 --hops-test 2-8 \\
        --steps 2400 --batch-size 8 --grad-accum 2 --grad-checkpointing --out runs/latent

Importable: ``run(config: dict) -> results dict`` (keys as the CLI flags with dashes -> underscores). Writes
``<out>/results.json`` (accuracy per (eval R, hops), loss curve, periodic evals, config, throughput).

``--ckpt-every N`` writes ``<ckpt-dir or out>/ckpt.pt`` every N steps (trainable weights, optimizer, grad scaler,
data and RNG state, loss curve); rerunning the same command resumes from it, giving the same trajectory as an
uninterrupted run. Point ``--ckpt-dir`` at persistent storage (Google Drive on Colab).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from ..bitlinear import replace_linears
from ..modeling import LLM_LINEAR_RE, _strip_quant_config
from .compressor import LatentCompressor, PauseTokens, embedding_rms
from .lora import apply_lora, count_params, freeze_base, lora_parameters, save_trainable
from .reasoner import LatentReasoner, default_read_layer, enable_checkpointing, num_stages
from .tasks import TaskSampler, make_test_set, parse_hops

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

DEFAULTS = dict(
    model="tiny", mode="latent", R=4, K=4, read_layer=None, hops_train="2-6", hops_test="2-8", distractors=2,
    steps=1000, batch_size=8, grad_accum=1, lr_compressor=1e-3, lr_lora=2e-4, lr_base=1e-3, weight_decay=0.0,
    warmup_steps=50, lora_rank=16, lora_alpha=32.0, lora_dropout=0.0, train_base=False, curriculum=False,
    steps_per_stage=200, bptt=0, dtype="fp32", grad_checkpointing=False, eval_every=0, eval_quick_n=32,
    eval_R_sweep="0,1,2,4,6,8", n_test=100, eval_batch_size=32, compressor_layers=1, compressor_heads=8,
    max_grad_norm=1.0, seed=0, out=None, device=None, log_every=25, save_weights=False,
    tiny_hidden=256, tiny_layers=4, tiny_intermediate=1024, tiny_heads=4, tiny_kv_heads=2, threads=0,
    max_minutes=0.0, ckpt_every=0, ckpt_dir=None, resume=True,
)

# config keys that may differ between a checkpoint and the run resuming it
_RESUME_FREE_KEYS = {"out", "ckpt_dir", "ckpt_every", "resume", "max_minutes", "device", "threads", "log_every",
                     "eval_every", "eval_quick_n", "eval_batch_size", "save_weights", "read_layer"}


# ----------------------------------------------------------------------------------------------------------------
# model construction
# ----------------------------------------------------------------------------------------------------------------
def _fp16_safe_mlp(mlp):
    """relu^2 * up can overflow fp16; compute the product in fp32 (ffn_sub_norm then brings it back to O(1))."""

    def forward(x):
        g = mlp.gate_proj(x).float()
        u = mlp.up_proj(x).float()
        return mlp.down_proj(mlp.ffn_sub_norm(mlp.act_fn(g) * u))

    mlp.forward = forward


_TOK_CACHE = {}


def _tiny_tokenizer():
    """Llama-3 tokenizer for the tiny model: rebuilt offline from llama.cpp's vocab GGUF when it is available (repo
    checkout; override the path with $BITNET_VOCAB_GGUF), else downloaded from the hub ($BITNET_TOKENIZER, default
    the 2B4T repo; same 128256-token vocabulary). Cached: building it from the GGUF takes a few seconds."""
    if "tok" not in _TOK_CACHE:
        from ..tokenizer_utils import DEFAULT_VOCAB_GGUF, build_llama3_tokenizer

        gguf = Path(os.environ.get("BITNET_VOCAB_GGUF", DEFAULT_VOCAB_GGUF))
        if gguf.exists():
            _TOK_CACHE["tok"] = build_llama3_tokenizer(gguf)
        else:
            from transformers import AutoTokenizer

            _TOK_CACHE["tok"] = AutoTokenizer.from_pretrained(
                os.environ.get("BITNET_TOKENIZER", "microsoft/bitnet-b1.58-2B-4T-bf16"))
    return _TOK_CACHE["tok"]


def load_lm(cfg: dict, device):
    """-> (lm with BitLinear decoder linears, tokenizer). Master weights fp32 when the base is trained, else the
    requested dtype (fp16/bf16 halves the memory of the frozen base)."""
    from transformers import AutoConfig, AutoTokenizer, BitNetForCausalLM

    train_base = cfg["train_base"]
    store_dtype = torch.float32 if (train_base or device.type == "cpu" and cfg["dtype"] == "fp16") \
        else DTYPES[cfg["dtype"]]
    if cfg["model"] == "tiny":
        from ..make_tiny_ckpt import tiny_llm_config
        tok = _tiny_tokenizer()
        torch.manual_seed(cfg["seed"])
        lcfg = tiny_llm_config(vocab_size=len(tok), hidden=cfg["tiny_hidden"], intermediate=cfg["tiny_intermediate"],
                               layers=cfg["tiny_layers"], heads=cfg["tiny_heads"], kv_heads=cfg["tiny_kv_heads"])
        lm = BitNetForCausalLM(_strip_quant_config(lcfg)).to(store_dtype)
    else:
        lcfg = _strip_quant_config(AutoConfig.from_pretrained(cfg["model"]))
        lm = BitNetForCausalLM.from_pretrained(cfg["model"], config=lcfg, dtype=store_dtype)
        tok = AutoTokenizer.from_pretrained(cfg["model"])
    lm.config.use_cache = True
    replace_linears(lm, lambda n, m: bool(LLM_LINEAR_RE.search(n)))  # BitLinear QAT forward (W1.58 A8)
    lm.to(device)
    if cfg["lora_rank"] > 0:
        apply_lora(lm, r=cfg["lora_rank"], alpha=cfg["lora_alpha"], dropout=cfg["lora_dropout"],
                   prequantize=not train_base)
    if train_base:
        lm.requires_grad_(True)
    else:
        freeze_base(lm)
    if cfg["dtype"] == "fp16" and device.type == "cuda":
        for layer in lm.model.layers:
            _fp16_safe_mlp(layer.mlp)
    if cfg["grad_checkpointing"]:
        enable_checkpointing(lm, True)
    return lm, tok


def build_reasoner(cfg: dict, lm, tok, device) -> LatentReasoner:
    d = lm.config.hidden_size
    rms = embedding_rms(lm.get_input_embeddings().weight)
    comp = pause = None
    if cfg["mode"] == "latent":
        heads = cfg["compressor_heads"]
        while d % heads:
            heads //= 2
        comp = LatentCompressor(d, cfg["K"], n_layers=cfg["compressor_layers"], n_heads=heads, target_rms=rms,
                                max_steps=max(16, cfg["R"] + 1)).to(device)
    elif cfg["mode"] == "pause":
        g = torch.Generator().manual_seed(cfg["seed"])
        idx = torch.randint(0, lm.get_input_embeddings().num_embeddings, (cfg["K"],), generator=g)
        pause = PauseTokens(d, cfg["K"], target_rms=rms,
                            init=lm.get_input_embeddings().weight[idx.to(device)].float()).to(device)
    rl = cfg["read_layer"] or default_read_layer(lm.config.num_hidden_layers)
    return LatentReasoner(lm, comp, read_layer=rl, K=cfg["K"], mode=cfg["mode"], tokenizer=tok, pause=pause,
                          R=cfg["R"], bptt=cfg["bptt"])


def param_groups(cfg, reasoner):
    groups = []
    aux = [p for m in (reasoner.compressor, reasoner.pause) if m is not None for p in m.parameters()]
    if aux:
        groups.append(dict(params=aux, lr=cfg["lr_compressor"], name="compressor", weight_decay=cfg["weight_decay"]))
    lora = [p for p in lora_parameters(reasoner.lm) if p.requires_grad]
    lora_ids = {id(p) for p in lora}
    if lora:
        groups.append(dict(params=lora, lr=cfg["lr_lora"], name="lora", weight_decay=0.0))
    base = [p for p in reasoner.lm.parameters() if p.requires_grad and id(p) not in lora_ids]
    if base:
        groups.append(dict(params=base, lr=cfg["lr_base"], name="base", weight_decay=cfg["weight_decay"]))
    for g in groups:
        g["base_lr"] = g["lr"]
    return groups


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


# ----------------------------------------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------------------------------------
def evaluate(reasoner, test_set, R, stage=None, batch_size=32, limit=None, autocast=nullcontext):
    was = reasoner.training
    reasoner.eval()
    acc = {}
    samples = {}
    with autocast():
        for h, exs in test_set.items():
            exs = exs[:limit] if limit else exs
            res = []
            for i in range(0, len(exs), batch_size):
                res += reasoner.answer_batch(exs[i:i + batch_size], R=R, stage=stage)
            acc[str(h)] = sum(r["correct"] for r in res) / max(1, len(res))
            samples[str(h)] = res[0]["text"]
    reasoner.train(was)
    return acc, samples


def eval_R_values(cfg) -> list[int]:
    if cfg["mode"] in ("none", "cot"):
        return [0]
    vals = sorted({int(x) for x in str(cfg["eval_R_sweep"]).split(",") if x != ""} | {cfg["R"]})
    return vals


# ----------------------------------------------------------------------------------------------------------------
# checkpointing
# ----------------------------------------------------------------------------------------------------------------
def _ckpt_path(cfg) -> Path | None:
    d = cfg["ckpt_dir"] or cfg["out"]
    return Path(d) / "ckpt.pt" if d else None


def _config_signature(cfg) -> dict:
    return {k: v for k, v in cfg.items() if k not in _RESUME_FREE_KEYS}


def save_checkpoint(path: Path, cfg, reasoner, opt, scaler, sampler, step_done: int, state: dict):
    """Atomically write everything needed to continue training after `step_done` optimizer steps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ck = dict(
        version=1, step=step_done, config=_config_signature(cfg),
        trainable={n: p.detach().cpu().clone() for n, p in reasoner.named_parameters() if p.requires_grad},
        opt=opt.state_dict(), scaler=scaler.state_dict(), sampler_rng=sampler.rng.getstate(),
        py_rng=random.getstate(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, **state,
    )
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ck, str(tmp))
    os.replace(tmp, path)


def load_checkpoint(path: Path, cfg, reasoner, opt, scaler, sampler) -> dict:
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    want, have = _config_signature(cfg), ck["config"]
    diff = {k: (have.get(k), want.get(k)) for k in set(want) | set(have) if have.get(k) != want.get(k)}
    if diff:
        raise ValueError(f"checkpoint {path} was made with a different config (ckpt, now): {diff}")
    params = dict(reasoner.named_parameters())
    missing = [n for n, p in params.items() if p.requires_grad and n not in ck["trainable"]]
    if missing:
        raise RuntimeError(f"checkpoint {path} lacks trainable params {missing[:5]}")
    with torch.no_grad():
        for n, v in ck["trainable"].items():
            params[n].copy_(v.to(params[n].device, params[n].dtype))
    opt.load_state_dict(ck["opt"])
    scaler.load_state_dict(ck["scaler"])
    sampler.rng.setstate(ck["sampler_rng"])
    random.setstate(ck["py_rng"])
    torch.set_rng_state(ck["torch_rng"])
    if ck["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return ck


# ----------------------------------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------------------------------
def run(config: dict) -> dict:
    cfg = dict(DEFAULTS)
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg.update(config)
    if cfg["threads"]:
        torch.set_num_threads(cfg["threads"])
    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    t0 = time.time()
    lm, tok = load_lm(cfg, device)
    reasoner = build_reasoner(cfg, lm, tok, device)
    cfg["read_layer"] = reasoner.read_layer
    n_trainable = count_params(reasoner)
    print(f"[latent] model={cfg['model']} mode={cfg['mode']} R={cfg['R']} K={cfg['K']} read_layer="
          f"{reasoner.read_layer}/{reasoner.num_layers} trainable={n_trainable / 1e6:.2f}M device={device} "
          f"load {time.time() - t0:.1f}s", flush=True)

    amp_dtype = DTYPES[cfg["dtype"]]
    use_amp = amp_dtype != torch.float32
    autocast = (lambda: torch.autocast(device.type, dtype=amp_dtype)) if use_amp else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16 and device.type == "cuda"))
    groups = param_groups(cfg, reasoner)
    if not groups:
        raise ValueError("nothing to train: use --lora-rank > 0 or --train-base")
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8) if groups else None
    trainable = [p for g in groups for p in g["params"]]

    hops_train, hops_test = parse_hops(cfg["hops_train"]), parse_hops(cfg["hops_test"])
    sampler = TaskSampler(hops_train, cfg["distractors"], seed=cfg["seed"])
    test_set = make_test_set(hops_test, cfg["n_test"], cfg["distractors"], seed=cfg["seed"])
    curriculum = cfg["curriculum"] and cfg["mode"] in ("latent", "pause")
    n_st = num_stages(cfg["R"]) if curriculum else 1

    def stage_at(step):
        return min(step // cfg["steps_per_stage"], n_st - 1) if curriculum else None

    loss_curve, evals = [], []
    n_examples = n_positions = 0
    run_loss, run_n = 0.0, 0
    prior_train_s = 0.0
    start = 0
    ckpt = _ckpt_path(cfg)
    if cfg["resume"] and ckpt is not None and ckpt.exists():
        ck = load_checkpoint(ckpt, cfg, reasoner, opt, scaler, sampler)
        start = ck["step"]
        loss_curve, evals = ck["loss_curve"], ck["evals"]
        n_examples, n_positions = ck["n_examples"], ck["n_positions"]
        run_loss, run_n, prior_train_s = ck["run_loss"], ck["run_n"], ck["train_seconds"]
        print(f"[latent] resumed from {ckpt} at step {start}/{cfg['steps']}", flush=True)
        del ck

    def checkpoint(step_done):
        if ckpt is None:
            return
        t = time.time()
        save_checkpoint(ckpt, cfg, reasoner, opt, scaler, sampler, step_done, dict(
            loss_curve=loss_curve, evals=evals, n_examples=n_examples, n_positions=n_positions,
            run_loss=run_loss, run_n=run_n, train_seconds=prior_train_s + time.time() - t_train))
        print(f"[latent] checkpoint step {step_done} -> {ckpt} ({time.time() - t:.1f}s)", flush=True)

    reasoner.train()
    t_train = time.time()
    n_examples_start = n_examples
    steps_done = start
    stopped_early = False
    for step in range(start, cfg["steps"]):
        stage = stage_at(step)
        f = lr_factor(step, cfg["steps"], cfg["warmup_steps"])
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        tot = 0.0
        for _ in range(cfg["grad_accum"]):
            batch = reasoner.build_batch(sampler.batch(cfg["batch_size"]), stage=stage)
            with autocast():
                out = reasoner(batch)
            loss = out["loss"] / cfg["grad_accum"]
            scaler.scale(loss).backward()
            tot += loss.item()
            n_examples += cfg["batch_size"]
            n_positions += cfg["batch_size"] * (batch["prompt_ids"].shape[1] + batch["R"] * cfg["K"] *
                                                (cfg["mode"] in ("latent", "pause")) + batch["target_ids"].shape[1])
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"])
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        run_loss += tot
        run_n += 1
        if (step + 1) % cfg["log_every"] == 0 or step == 0:
            avg = run_loss / run_n
            loss_curve.append(dict(step=step + 1, loss=avg, stage=stage, grad_norm=float(gn)))
            el = time.time() - t_train
            print(f"[latent] step {step + 1}/{cfg['steps']} stage {stage} loss {avg:.4f} gn {float(gn):.2f} "
                  f"{(n_examples - n_examples_start) / el:.1f} ex/s", flush=True)
            run_loss, run_n = 0.0, 0
        if cfg["eval_every"] and (step + 1) % cfg["eval_every"] == 0 and step + 1 < cfg["steps"]:
            acc, smp = evaluate(reasoner, test_set, cfg["R"], stage, cfg["eval_batch_size"], cfg["eval_quick_n"],
                                autocast)
            evals.append(dict(step=step + 1, stage=stage, R=cfg["R"], acc=acc))
            print(f"[latent] eval step {step + 1} stage {stage}: {acc}  e.g. {smp[next(iter(smp))]!r}", flush=True)
        steps_done = step + 1
        if cfg["ckpt_every"] and steps_done % cfg["ckpt_every"] == 0 and steps_done < cfg["steps"]:
            checkpoint(steps_done)
        if cfg["max_minutes"] and (time.time() - t_train) > 60 * cfg["max_minutes"]:
            print(f"[latent] time budget reached at step {step + 1}", flush=True)
            stopped_early = True
            break
    train_time = prior_train_s + time.time() - t_train
    if cfg["ckpt_every"] and steps_done > start:
        checkpoint(steps_done)  # an interrupted final eval then resumes straight into evaluation

    # final evaluation: all-latent setting, eval-R sweep
    t_eval = time.time()
    final, final_samples = {}, {}
    for R in eval_R_values(cfg):
        acc, smp = evaluate(reasoner, test_set, R, None, cfg["eval_batch_size"], None, autocast)
        final[str(R)] = acc
        final_samples[str(R)] = smp
        print(f"[latent] final eval R={R}: {acc}", flush=True)
    eval_time = time.time() - t_eval
    results = dict(
        config=cfg, final_acc=final, final_samples=final_samples, loss_curve=loss_curve, evals=evals,
        throughput=dict(train_seconds=train_time, steps=steps_done, examples_per_s=n_examples / max(train_time, 1e-9),
                        positions_per_s=n_positions / max(train_time, 1e-9), eval_seconds=eval_time,
                        stopped_early=stopped_early),
        n_trainable=n_trainable, device=str(device),
        max_memory_gb=(torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None),
    )
    if cfg["out"]:
        out = Path(cfg["out"])
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "results.json", "w") as fh:
            json.dump(results, fh, indent=1)
        if cfg["save_weights"]:
            save_trainable(out / "trainable.pt", lm, {"compressor": reasoner.compressor, "pause": reasoner.pause},
                           include_base=cfg["train_base"])
        print(f"[latent] wrote {out / 'results.json'}", flush=True)
    results["_reasoner"] = reasoner  # not serialized; handy for notebooks (pop it before json.dump)
    return results


def parse_args(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    D = DEFAULTS
    ap.add_argument("--model", default=D["model"], help="'tiny' (random tiny BitNet) or an HF id / local dir")
    ap.add_argument("--mode", choices=["none", "latent", "pause", "cot"], default=D["mode"])
    ap.add_argument("--R", type=int, default=D["R"], help="thought steps (final curriculum stage)")
    ap.add_argument("--K", type=int, default=D["K"], help="soft tokens per thought step")
    ap.add_argument("--read-layer", type=int, default=None, help="HF hidden_states index (default round(2L/3))")
    ap.add_argument("--hops-train", default=D["hops_train"])
    ap.add_argument("--hops-test", default=D["hops_test"])
    ap.add_argument("--distractors", type=int, default=D["distractors"])
    ap.add_argument("--steps", type=int, default=D["steps"], help="optimizer steps")
    ap.add_argument("--batch-size", type=int, default=D["batch_size"])
    ap.add_argument("--grad-accum", type=int, default=D["grad_accum"])
    ap.add_argument("--lr", type=float, default=None, help="shortcut: sets --lr-lora and --lr-base")
    ap.add_argument("--lr-compressor", type=float, default=D["lr_compressor"], help="compressor / pause embeddings")
    ap.add_argument("--lr-lora", type=float, default=D["lr_lora"])
    ap.add_argument("--lr-base", type=float, default=D["lr_base"], help="with --train-base")
    ap.add_argument("--weight-decay", type=float, default=D["weight_decay"])
    ap.add_argument("--warmup-steps", type=int, default=D["warmup_steps"])
    ap.add_argument("--lora-rank", type=int, default=D["lora_rank"], help="0 disables LoRA")
    ap.add_argument("--lora-alpha", type=float, default=D["lora_alpha"])
    ap.add_argument("--lora-dropout", type=float, default=D["lora_dropout"])
    ap.add_argument("--train-base", action="store_true", help="train ALL LLM weights (needed for --model tiny)")
    ap.add_argument("--curriculum", action="store_true", help="Coconut curriculum (latent / pause modes)")
    ap.add_argument("--steps-per-stage", type=int, default=D["steps_per_stage"])
    ap.add_argument("--bptt", type=int, default=D["bptt"], help="keep grads through the last N thoughts (0 = all)")
    ap.add_argument("--dtype", choices=list(DTYPES), default=D["dtype"])
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--eval-every", type=int, default=D["eval_every"])
    ap.add_argument("--eval-quick-n", type=int, default=D["eval_quick_n"], help="examples per hops in periodic evals")
    ap.add_argument("--eval-R-sweep", default=D["eval_R_sweep"])
    ap.add_argument("--n-test", type=int, default=D["n_test"], help="test examples per hops")
    ap.add_argument("--eval-batch-size", type=int, default=D["eval_batch_size"])
    ap.add_argument("--compressor-layers", type=int, default=D["compressor_layers"])
    ap.add_argument("--compressor-heads", type=int, default=D["compressor_heads"])
    ap.add_argument("--max-grad-norm", type=float, default=D["max_grad_norm"])
    ap.add_argument("--seed", type=int, default=D["seed"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every", type=int, default=D["log_every"])
    ap.add_argument("--save-weights", action="store_true", help="save LoRA + compressor (or full LM with --train-base)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop training after this wall time (0 = off)")
    ap.add_argument("--ckpt-every", type=int, default=0, help="save a resumable checkpoint every N steps (0 = off)")
    ap.add_argument("--ckpt-dir", default=None, help="checkpoint directory (default --out); e.g. on Google Drive")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="ignore an existing checkpoint")
    ap.add_argument("--tiny-hidden", type=int, default=D["tiny_hidden"])
    ap.add_argument("--tiny-layers", type=int, default=D["tiny_layers"])
    ap.add_argument("--tiny-intermediate", type=int, default=D["tiny_intermediate"])
    ap.add_argument("--out", default=None)
    a = vars(ap.parse_args(argv))
    lr = a.pop("lr")
    if lr is not None:
        a["lr_lora"] = a["lr_base"] = lr
    return a


def main(argv=None):
    res = run(parse_args(argv))
    res.pop("_reasoner", None)


if __name__ == "__main__":
    main()
