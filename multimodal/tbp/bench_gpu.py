"""Train STE+Adam vs TBP on the GPU at the same architecture and log GPU usage (nvidia-smi every 0.5 s).

    cd multimodal && runs/.venv-tbp/bin/python -m tbp.bench_gpu --depth 10 --width 1024 --epochs 15
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import tbp as TBP


def sample_gpu(fn, log_path: Path):
    """Run fn() while nvidia-smi logs utilization/memory/power; return (fn result, samples, wall seconds)."""
    proc = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                             "--format=csv,noheader,nounits", "-lms", "500"],
                            stdout=open(log_path, "w"), stderr=subprocess.DEVNULL)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    try:
        out = fn()
        torch.cuda.synchronize()
    finally:
        wall = time.time() - t
        proc.terminate()
        proc.wait()
    rows = []
    for line in log_path.read_text().splitlines():
        try:
            rows.append([float(v) for v in line.split(",")])
        except ValueError:
            pass
    return out, np.array(rows), wall


def summarize(name, hist, s, wall, peak_mb):
    util, mem, power = s[:, 0], s[:, 1], s[:, 2]
    return dict(method=name, final_test=hist[-1]["test"], best_test=max(h["test"] for h in hist),
                final_train=hist[-1]["train"], s_per_epoch=float(np.mean([h["epoch_seconds"] for h in hist])),
                wall_s=wall, gpu_util_mean=float(util.mean()), gpu_util_p90=float(np.percentile(util, 90)),
                mem_used_max_mb=float(mem.max()), torch_peak_mb=peak_mb, power_mean_w=float(power.mean()),
                energy_wh=float(power.mean() * wall / 3600), samples=len(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fmnist")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=float, default=0.005)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--out", default="runs/bench_gpu")
    ap.add_argument("--only", choices=["ste", "tbp"], default=None, help="internal: run one method in this process")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    din = 784 if a.dataset == "fmnist" else 64
    n_w = din * a.width + (a.depth - 1) * a.width ** 2 + a.width * 10
    print(f"{a.depth} hidden layers x {a.width}: {n_w / 1e6:.2f}M ternary weights (both methods)", flush=True)

    if a.only == "ste":
        h, s, wall = sample_gpu(lambda: TBP.run_ste(a.dataset, a.depth, a.width, a.epochs, a.bs, a.seed,
                                                    log=lambda x: print("  ste", x, flush=True)), out / "ste_smi.csv")
        r = summarize("STE+Adam", h, s, wall, torch.cuda.max_memory_allocated() / 2**20)
        (out / "ste.json").write_text(json.dumps(dict(summary=r, hist=h)))
        return
    if a.only == "tbp":
        (h, _), s, wall = sample_gpu(lambda: TBP.run_tbp(a.dataset, a.depth, a.width, a.epochs, a.bs, a.seed, a.k,
                                                         a.T, log=lambda x: print("  tbp", x, flush=True)),
                                     out / "tbp_smi.csv")
        r = summarize("TBP", h, s, wall, torch.cuda.max_memory_allocated() / 2**20)
        (out / "tbp.json").write_text(json.dumps(dict(summary=r, hist=h)))
        return
    # each method in its own process, so memory held by one run cannot show up in the other's numbers
    import sys

    for m in ("ste", "tbp"):
        subprocess.run([sys.executable, "-m", "tbp.bench_gpu", "--only", m] +
                       [f"--{k.replace('_', '-')}={v}" for k, v in vars(a).items() if k != "only"], check=True)
        time.sleep(3)
    res = [json.loads((out / f"{m}.json").read_text())["summary"] for m in ("ste", "tbp")]
    (out / "results.json").write_text(json.dumps(dict(summary=res, n_weights=n_w), indent=1))
    keys = ["final_test", "best_test", "final_train", "s_per_epoch", "gpu_util_mean", "gpu_util_p90",
            "mem_used_max_mb", "torch_peak_mb", "power_mean_w", "energy_wh"]
    print(f"\n{'':<16}" + "".join(f"{r['method']:>12}" for r in res))
    for k in keys:
        print(f"{k:<16}" + "".join(f"{r[k]:>12.3f}" if r[k] < 100 else f"{r[k]:>12.0f}" for r in res))


if __name__ == "__main__":
    main()
