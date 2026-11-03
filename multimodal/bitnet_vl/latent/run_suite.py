"""Run the latent-reasoning comparison (none / cot / pause / latent) as a resumable suite.

    cd multimodal
    python -m bitnet_vl.latent.run_suite --run-dir /content/drive/MyDrive/bitnet_latent_reasoning/quick \\
        --preset QUICK --modes latent,pause,none,cot

Each mode writes ``<run-dir>/<mode>/results.json`` when it finishes and checkpoints ``<run-dir>/<mode>/ckpt.pt``
every ``ckpt_every`` steps while training. Rerunning the same command after a lost session skips finished modes and
resumes the unfinished one from its last checkpoint, so put ``--run-dir`` on persistent storage (Google Drive on
Colab). ``<run-dir>/all.json`` collects the results of every finished mode.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from .train_latent import run

MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"
MODES = ["none", "cot", "pause", "latent"]

PRESETS = {
    # R thought steps x K soft tokens; curriculum: R+1 stages of `steps_per_stage` (stage 0 = CoT, last = all latent)
    "QUICK": dict(R=4, K=4, hops_train="2-4", hops_test="2-6", distractors=1, steps=500, batch_size=4, grad_accum=4,
                  steps_per_stage=50, n_test=50, eval_every=0, eval_R_sweep="0,1,2,4,6,8", ckpt_every=25),
    "FULL": dict(R=6, K=4, hops_train="2-6", hops_test="2-8", distractors=2, steps=2000, batch_size=8, grad_accum=4,
                 steps_per_stage=150, n_test=200, eval_every=500, eval_R_sweep="0,1,2,4,6,8,10", ckpt_every=50),
}
COMMON_CFG = dict(model=MODEL, lora_rank=16, lora_alpha=32, lr_lora=2e-4, lr_compressor=1e-3, warmup_steps=30,
                  grad_checkpointing=True, log_every=25, eval_batch_size=32, seed=0)


def default_dtype() -> str:
    # bf16 wherever torch supports it (native on L4/A100, emulated on T4). 2B-4T overflows fp16: on a T4 the fp16 run
    # gave NaN loss from step 1 while bf16 trained normally.
    if not torch.cuda.is_available():
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def mode_cfg(mode: str, run_dir: str | Path, preset: str = "QUICK", common: dict | None = None,
             presets: dict | None = None, **overrides) -> dict:
    cfg = dict(common if common is not None else COMMON_CFG, **(presets or PRESETS)[preset], mode=mode,
               out=str(Path(run_dir) / mode))
    cfg["curriculum"] = mode in ("latent", "pause")
    cfg.setdefault("dtype", default_dtype())
    cfg.update(overrides)
    return cfg


def run_suite(modes, run_dir: str | Path, cfg_for=None, **cfg_kw) -> dict:
    """Run `modes` in order, skipping those whose results.json exists. `cfg_for(mode)` builds a mode's config
    (default: mode_cfg(mode, run_dir, **cfg_kw)). Returns {mode: results} for every finished mode."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_for = cfg_for or (lambda m: mode_cfg(m, run_dir, **cfg_kw))
    results = {}
    for mode in modes:
        cfg = cfg_for(mode)
        done = Path(cfg["out"]) / "results.json"
        if done.exists():
            results[mode] = json.loads(done.read_text())
            print(f"===== {mode}: already finished ({done}), skipping", flush=True)
            continue
        print(f"===== {mode} =====", flush=True)
        t0 = time.time()
        res = run(cfg)
        res.pop("_reasoner", None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results[mode] = res
        tp = res["throughput"]
        print(f"===== {mode}: {time.time() - t0:.0f}s this session, {tp['examples_per_s']:.1f} ex/s, "
              f"max mem {res['max_memory_gb']} GB, acc {res['final_acc']}", flush=True)
        (run_dir / "all.json").write_text(json.dumps(results, indent=1))
    (run_dir / "all.json").write_text(json.dumps(results, indent=1))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--preset", choices=list(PRESETS), default="QUICK")
    ap.add_argument("--modes", default=",".join(MODES), help="comma-separated, run in this order")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dtype", default=None, help="default: bf16 if supported, else fp16 (fp32 on CPU)")
    ap.add_argument("--ckpt-every", type=int, default=None, help="override the preset's checkpoint interval")
    ap.add_argument("--steps", type=int, default=None, help="override the preset's step count")
    a = ap.parse_args(argv)
    over = dict(model=a.model)
    for k in ("dtype", "ckpt_every", "steps"):
        if getattr(a, k) is not None:
            over[k] = getattr(a, k)
    run_suite([m.strip() for m in a.modes.split(",") if m.strip()], a.run_dir, preset=a.preset, **over)


if __name__ == "__main__":
    main()
