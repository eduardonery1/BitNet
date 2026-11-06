"""2 x 2 ablation of TBP on Fashion-MNIST: fixed vs adaptive k, fixed vs learned per-layer scales.

    cd multimodal && runs/.venv-tbp/bin/python -m tbp.ablate --arch res10x1024,plain3x1786 --seeds 0,1,2

Every run holds out the last 5,000 training images for validation (also the fixed-k baseline, so all variants train on
55,000 images) and reports accuracy on the official 10,000 test images. Each run is a separate process.
Results: runs/ablate/<arch>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "ablate"
ARCHS = {"res10x1024": ["--depth", "10", "--width", "1024", "--residual"],
         "plain3x1786": ["--depth", "3", "--width", "1786"]}
VARIANTS = {"baseline": [], "adaptive_k": ["--adaptive-k"], "scale": ["--learn-scale"],
            "both": ["--adaptive-k", "--learn-scale"],
            # stability study: counters reset every 32 steps / vote only above the row's mean |G| / both
            "reset32": ["--counter-reset", "32"], "sig_mean": ["--sig-z", "1"],
            "reset32_sig": ["--counter-reset", "32", "--sig-z", "1"],
            "binary_init": ["--init-dist", "binary"],
            "t_sched": ["--t-schedule", "--counter-bits", "8"]}


FLAG = {"z": "--sig-z", "T": "--T", "k": "--k", "reset": "--counter-reset", "decay": "--counter-decay",
        "cool": "--cooldown", "strength": "--vote-strength", "stagger": "--stagger",
        "flipevery": "--flip-every", "tau": "--tau", "p": "--p-inc", "cb": "--counter-bits",
        "init": "--init-dist", "method": "--method", "lr": "--ste-lr", "boptau": "--bop-tau",
        "bopgamma": "--bop-gamma", "bopk": "--bop-k", "bs": "--bs", "exact": "--exact-backward", "epochs": "--epochs", "revctl": "--rev-control", "ln": "--ln-affine", "lnlr": "--ln-lr", "revlo": "--rev-lo", "revhi": "--rev-hi"}


def variant_args(variant):
    """Named variant, or 'key=val+key=val' (e.g. 'z=5', 'T=16+k=0.002'). T > 8 switches to int8 counters."""
    if variant in VARIANTS:
        return VARIANTS[variant]
    out = []
    for part in variant.split("+"):
        key, _, val = part.partition("=")
        out += [FLAG[key]] + ([val] if val != "" else [])
        if key == "T" and int(val) > 8 and "cb=" not in variant:
            out += ["--counter-bits", "8"]
    return out


def arch_args(arch):
    """Named arch, or 'res<L>x<W>' / 'plain<L>x<W>'."""
    if arch in ARCHS:
        return ARCHS[arch]
    import re
    m = re.fullmatch(r"(res|plain)(\d+)x(\d+)", arch)
    assert m, arch
    return ["--depth", m.group(2), "--width", m.group(3)] + (["--residual"] if m.group(1) == "res" else [])


def run_one(arch, variant, seed, common):
    out = OUT / f"{arch}_{variant}_s{seed}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        cmd = [sys.executable, "-m", "tbp.tbp", "--dataset", "fmnist", "--seeds", str(seed), "--out", str(out),
               "--val-split", "5000"] + arch_args(arch) + common + variant_args(variant)  # variant flags win
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return next(iter(json.loads(out.read_text()).values()))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="res10x1024,plain3x1786")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out-dir", default=None, help="results directory (default runs/ablate)")
    a = ap.parse_args()
    global OUT
    if a.out_dir:
        OUT = ROOT / a.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    common = ["--epochs", str(a.epochs), "--bs", "512", "--k", "0.005", "--T", "8", "--device", a.device,
              "--threads", str(a.threads)]
    seeds = [int(s) for s in a.seeds.split(",")]
    archs, variants = a.arch.split(","), a.variants.split(",")
    for s in seeds:  # seed-major: a complete one-seed comparison arrives first
        for arch in archs:
            for v in variants:
                h = run_one(arch, v, s, common)
                print(f"  seed {s} {arch} {v}: final test {h[-1]['test']:.4f} (best {max(x['test'] for x in h):.4f}), "
                      f"final k {h[-1]['k']:.5f}, final T {h[-1].get('T')}, flips/step first {h[0]['flips_per_step']:.0f} "
                      f"last {h[-1]['flips_per_step']:.0f}", flush=True)
    for arch in archs:
        res = {v: [run_one(arch, v, s, common) for s in seeds] for v in variants}
        (OUT / f"{arch}.json").write_text(json.dumps(res))
        print(f"\n{arch}: {'variant':<12}{'final test':>18}{'best test':>12}{'train':>8}{'final k':>9}{'s/epoch':>9}")
        for v, runs in res.items():
            fin = [r[-1]["test"] for r in runs]
            best = [max(h["test"] for h in r) for r in runs]
            print(f"{'':<{len(arch) + 2}}{v:<12}{np.mean(fin):>11.4f} ± {np.std(fin, ddof=1) if len(fin) > 1 else 0:.4f}"
                  f"{np.mean(best):>12.4f}{np.mean([r[-1]['train'] for r in runs]):>8.3f}"
                  f"{np.mean([r[-1]['k'] for r in runs]):>9.5f}"
                  f"{np.mean([np.mean([h['epoch_seconds'] for h in r]) for r in runs]):>9.2f}")
            if "scales" in runs[0][-1]:
                sc = np.mean([r[-1]["scales"] for r in runs], 0)
                print(f"{'':<{len(arch) + 14}}scales (mean over seeds): {np.round(sc, 3).tolist()}")


if __name__ == "__main__":
    main()
