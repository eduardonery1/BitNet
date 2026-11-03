"""Generate the self-contained Colab notebook for the latent-reasoning experiment.

    python multimodal/notebooks/build_colab.py            # -> multimodal/notebooks/bitnet_latent_reasoning.ipynb

The default source cell clones the public fork (branch feature/bitnet-vl, no submodules) and cds into
BitNet/multimodal. As a fallback (clone stale or unavailable), every source file the experiment imports
(multimodal/bitnet_vl/{__init__,bitlinear,modeling,tokenizer_utils,make_tiny_ckpt}.py, latent/*.py and
tests/test_latent.py) is also embedded in one fallback cell that writes them only when
``USE_EMBEDDED_SOURCES = True`` or when the clone did not provide them. Re-run this script after editing any of those
files.

``--check`` additionally executes the notebook's code cells locally (CPU, tiny model, skipping pip / nvidia-smi /
Google Drive cells) in a scratch directory: ``python multimodal/notebooks/build_colab.py --check --workdir DIR``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MM = HERE.parent  # multimodal/
OUT = HERE / "bitnet_latent_reasoning.ipynb"
TRANSFORMERS_VERSION = "5.0.0"  # the version the code was verified against

SOURCES = [
    "bitnet_vl/__init__.py",
    "bitnet_vl/bitlinear.py",
    "bitnet_vl/modeling.py",
    "bitnet_vl/tokenizer_utils.py",
    "bitnet_vl/make_tiny_ckpt.py",
    "bitnet_vl/latent/__init__.py",
    "bitnet_vl/latent/tasks.py",
    "bitnet_vl/latent/compressor.py",
    "bitnet_vl/latent/lora.py",
    "bitnet_vl/latent/reasoner.py",
    "bitnet_vl/latent/train_latent.py",
    "bitnet_vl/tests/test_latent.py",
]

INTRO = r"""# BitNet latent reasoning: compressed feedback vs pause tokens vs CoT

**Question.** Coconut-style latent reasoning with a *compressor*: run the BitNet b1.58 LLM over a prompt, read the
hidden states at layer *i* (default ≈ 2/3 depth), compress them with K learned queries (Perceiver-style cross-attention)
into **K soft tokens**, append those as input embeddings (re-using the KV cache, so only the K new positions are
computed), repeat **R** times, then decode the answer. Does feeding back compressed layer-*i* content beat the *same
extra compute without feedback* (learned **pause** tokens)?

**Task.** Synthetic variable tracking (`a = 3`, `b = a + 2`, ..., `print(d)`) with `hops` = chain length to the queried
variable, interleaved distractor assignments, values in [0, 99]. Trained on hops 2–4 (QUICK) / 2–6 (FULL), tested on
longer chains too (hops beyond the training range) and with an **eval-R sweep** (test-time scaling: more thought steps
at test time than in training).

**Modes** (matched settings: same base model, LoRA rank, steps, batch, data):
| mode | what is between the prompt and `Answer:` | loss |
|---|---|---|
| `none` | nothing | answer |
| `cot` | the explicit trace (`b = 5`, `c = 20`, ...) as text | trace + answer |
| `pause` | R×K learned constant pause embeddings (no feedback) | answer (+ remaining trace during curriculum) |
| `latent` | R steps × K compressor soft tokens (feedback of layer-*i* states) | answer (+ remaining trace during curriculum) |

`pause` and `latent` use the Coconut curriculum: stage *s* replaces the first *s* trace lines by *s* thought steps;
the last stage is all-latent with R steps.

The base model (`microsoft/bitnet-b1.58-2B-4T-bf16`, public, MIT, no token needed) is frozen and runs the BitNet
W1.58A8 forward (ternary absmean weights, per-token int8 activations, as in bitnet.cpp); LoRA adapters (full
precision) are added on q/k/v/o/gate/up/down, and the compressor is full precision.

**Runtime.** L4 or A100 with `bf16` is recommended. A T4 (16 GB) works with emulated `bf16` (measured: 6.1 GB peak, ~1.2–1.5 examples/s;
`fp16` overflows and gives NaN loss with 2B-4T) but is several times slower.
Runtimes below are *estimates* (not measured on Colab): QUICK ≈ 1 h on L4 for the four modes, FULL ≈ 5–7 h on L4
(≈ 3 h on A100).
"""


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


SETUP = f"""
!nvidia-smi || echo "no GPU (the tiny sanity run still works on CPU)"
# torch comes with Colab; the code was verified with transformers=={TRANSFORMERS_VERSION} (+ torch 2.9).
!pip install -q "transformers=={TRANSFORMERS_VERSION}" safetensors pytest matplotlib
"""

CLONE = """
# Default: get the sources from the public fork (no --recursive: the latent experiment does not need the submodule).
!test -d BitNet || git clone -b feature/bitnet-vl --depth 1 https://github.com/eduardonery1/BitNet
%cd BitNet/multimodal
"""

FALLBACK_HEAD = """
# FALLBACK (skipped by default): embedded copies of the sources, written only if USE_EMBEDDED_SOURCES = True or the
# clone above did not provide them (e.g. the pushed branch is stale). They are the files this notebook was built from.
import os, sys
USE_EMBEDDED_SOURCES = False
_have_clone = os.path.exists("bitnet_vl/latent/reasoner.py")
"""

FALLBACK_TAIL = """
if USE_EMBEDDED_SOURCES or not _have_clone:
    for _path, _text in EMBEDDED_SOURCES.items():
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        with open(_path, "w", encoding="utf-8") as _f:
            _f.write(_text)
    print(f"wrote {len(EMBEDDED_SOURCES)} embedded source files into {os.getcwd()}")
else:
    print("using the cloned sources in", os.getcwd(), "(set USE_EMBEDDED_SOURCES = True to overwrite them)")
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
"""

TESTS = """
# CPU unit tests (~1-2 min). On Colab the tiny model uses the 2B4T tokenizer from the hub (same Llama-3 vocab).
!python -m pytest bitnet_vl/tests/test_latent.py -q -x
"""

COMMON = """
import gc, json, os, time
import torch
from bitnet_vl.latent.train_latent import run

LOCAL_CHECK = bool(os.environ.get("LATENT_NB_LOCAL_CHECK"))  # set only by build_colab.py --check (CPU smoke run)
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    # bf16 everywhere torch supports it (native on L4/A100, emulated on T4). 2B-4T overflows fp16: on a T4 the
    # fp16 smoke run gave NaN loss from step 1, bf16 trained normally (6.1 GB peak, ~1.2-1.5 ex/s).
    DTYPE = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
else:
    DTYPE = "fp32"
print("device:", torch.cuda.get_device_name() if HAS_CUDA else "cpu", "| dtype:", DTYPE)
RUN_DIR = "runs"
os.makedirs(RUN_DIR, exist_ok=True)


def run_and_free(cfg):
    res = run(cfg)
    res.pop("_reasoner", None)
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
    return res
"""

TINY = """
# Tiny sanity run: random-init tiny BitNet (hidden 256, 4 layers), ALL weights trained (a random LM cannot do anything
# with LoRA only). Only checks that the pipeline trains end to end; accuracy of a tiny model is not a result.
tiny = run_and_free(dict(
    model="tiny", train_base=True, lora_rank=0, mode="latent", R=2, K=2, hops_train="2-3", hops_test="2-3",
    distractors=1, steps=20 if LOCAL_CHECK else 400, batch_size=8 if LOCAL_CHECK else 32, lr_base=1e-3,
    n_test=8 if LOCAL_CHECK else 100, eval_R_sweep="0,2", log_every=10 if LOCAL_CHECK else 50,
    dtype="fp32" if not HAS_CUDA else DTYPE, out=f"{RUN_DIR}/tiny_latent",
    **(dict(tiny_hidden=64, tiny_layers=2, tiny_intermediate=128) if LOCAL_CHECK else {})))
print("loss first/last:", tiny["loss_curve"][0]["loss"], tiny["loss_curve"][-1]["loss"])
print("final acc:", tiny["final_acc"])
"""

PRESETS = """
MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"
PRESET = "QUICK"   # "QUICK" (~1 h on L4, fits a 16 GB T4 in bf16) or "FULL" (~5-7 h on L4); estimates
MODES = ["none", "cot", "pause", "latent"]

PRESETS = {
    # R thought steps x K soft tokens; curriculum: R+1 stages of `steps_per_stage` (stage 0 = CoT, last = all latent)
    "QUICK": dict(R=4, K=4, hops_train="2-4", hops_test="2-6", distractors=1, steps=500, batch_size=4, grad_accum=4,
                  steps_per_stage=50, n_test=50, eval_every=0, eval_R_sweep="0,1,2,4,6,8"),
    "FULL": dict(R=6, K=4, hops_train="2-6", hops_test="2-8", distractors=2, steps=2000, batch_size=8, grad_accum=4,
                 steps_per_stage=150, n_test=200, eval_every=500, eval_R_sweep="0,1,2,4,6,8,10"),
}
COMMON_CFG = dict(model=MODEL, dtype=DTYPE, lora_rank=16, lora_alpha=32, lr_lora=2e-4, lr_compressor=1e-3,
                  warmup_steps=30, grad_checkpointing=True, log_every=25, eval_batch_size=32, seed=0)

if LOCAL_CHECK:  # CPU smoke test of this very cell with the tiny model (used by build_colab.py --check)
    MODEL = "tiny"
    COMMON_CFG.update(model="tiny", train_base=True, lora_rank=0, lr_base=1e-3, dtype="fp32",
                      tiny_hidden=64, tiny_layers=2, tiny_intermediate=128, grad_checkpointing=True)
    PRESETS["QUICK"].update(R=2, K=2, hops_train="2-3", hops_test="2-4", steps=6, batch_size=2, grad_accum=1,
                            steps_per_stage=2, n_test=4, eval_R_sweep="0,1,2,4")


def mode_cfg(mode):
    cfg = dict(COMMON_CFG, **PRESETS[PRESET], mode=mode, out=f"{RUN_DIR}/{PRESET.lower()}_{mode}")
    cfg["curriculum"] = mode in ("latent", "pause")
    return cfg

for m in MODES:
    print(m, mode_cfg(m))
"""

MAIN = """
results = {}
for mode in MODES:
    t0 = time.time()
    print(f"===== {mode} =====")
    results[mode] = run_and_free(mode_cfg(mode))
    tp = results[mode]["throughput"]
    print(f"{mode}: {time.time() - t0:.0f}s, {tp['examples_per_s']:.1f} ex/s, "
          f"max mem {results[mode]['max_memory_gb']} GB, acc {results[mode]['final_acc']}")
json.dump({k: v for k, v in results.items()}, open(f"{RUN_DIR}/{PRESET.lower()}_all.json", "w"), indent=1)
"""

SWEEP_MD = """
### Eval-R sweep (test-time scaling)
`final_acc[R][hops]` is measured with the all-latent setting and R thought steps at test time (R may differ from the
training R; R = 0 means the answer is decoded right after the prompt). `none` / `cot` have a single entry.
"""

SWEEP = """
import pandas as pd

rows = []
for mode, res in results.items():
    for R, by_h in res["final_acc"].items():
        for h, acc in by_h.items():
            rows.append(dict(mode=mode, eval_R=int(R), hops=int(h), acc=acc))
df = pd.DataFrame(rows)
train_R = PRESETS[PRESET]["R"]
print("accuracy at the training R (none/cot: R=0), rows = mode, cols = hops")
main = df[(df.eval_R == train_R) | df["mode"].isin(["none", "cot"])]
display(main.pivot_table(index="mode", columns="hops", values="acc"))
print("eval-R sweep, mean over test hops")
display(df[df["mode"].isin(["latent", "pause"])].pivot_table(index="mode", columns="eval_R", values="acc"))
"""

PLOTS = """
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
for mode, res in results.items():
    R = str(train_R) if str(train_R) in res["final_acc"] else "0"
    hs = sorted(res["final_acc"][R], key=int)
    ax[0].plot([int(h) for h in hs], [res["final_acc"][R][h] for h in hs], marker="o", label=mode)
lo, hi = PRESETS[PRESET]["hops_train"].split("-")
ax[0].axvspan(int(lo), int(hi), color="grey", alpha=0.12, label="train hops")
ax[0].set(xlabel="hops", ylabel="accuracy", title=f"accuracy vs hops (eval R = train R = {train_R})", ylim=(0, 1.02))
ax[0].legend()
for mode in ("latent", "pause"):
    if mode in results:
        fa = results[mode]["final_acc"]
        Rs = sorted(fa, key=int)
        ax[1].plot([int(r) for r in Rs], [sum(fa[r].values()) / len(fa[r]) for r in Rs], marker="o", label=mode)
for mode in ("none", "cot"):
    if mode in results:
        a = results[mode]["final_acc"]["0"]
        ax[1].axhline(sum(a.values()) / len(a), ls="--", lw=1, label=mode, color="C0" if mode == "none" else "C1")
ax[1].axvline(train_R, color="grey", ls=":", lw=1)
ax[1].set(xlabel="eval R (thought steps at test time)", ylabel="mean accuracy over test hops",
          title="test-time scaling", ylim=(0, 1.02))
ax[1].legend()
for mode, res in results.items():
    lc = res["loss_curve"]
    ax[2].plot([x["step"] for x in lc], [x["loss"] for x in lc], label=mode)
ax[2].set(xlabel="optimizer step", ylabel="train loss", yscale="log", title="training loss")
ax[2].legend()
plt.tight_layout()
plt.savefig(f"{RUN_DIR}/{PRESET.lower()}_plots.png", dpi=120)
plt.show()
"""

DRIVE = """
# Optional: copy runs/ (results.json per mode + plots) to Google Drive.
SAVE_TO_DRIVE = False
if SAVE_TO_DRIVE:
    from google.colab import drive
    import shutil
    drive.mount("/content/drive")
    dst = f"/content/drive/MyDrive/bitnet_latent_reasoning/{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copytree(RUN_DIR, dst)
    print("saved to", dst)
"""


def build() -> dict:
    cells = [md(INTRO), code(SETUP),
             md("## Sources\nDefault: clone the public fork `eduardonery1/BitNet` (branch `feature/bitnet-vl`) and work "
                "in `BitNet/multimodal`."), code(CLONE),
             md("### Fallback: embedded sources (skipped by default)\nOnly used when `USE_EMBEDDED_SOURCES = True` or "
                "the clone is missing the latent experiment. Contains `multimodal/bitnet_vl` (the files the latent "
                "experiment imports) and `bitnet_vl/latent/`.")]
    emb = {rel: (MM / rel).read_text(encoding="utf-8") for rel in SOURCES}
    body = "EMBEDDED_SOURCES = {\n" + "".join(f"    {rel!r}: {txt!r},\n" for rel, txt in emb.items()) + "}\n"
    cells.append(code(FALLBACK_HEAD.strip("\n") + "\n" + body + FALLBACK_TAIL))
    cells += [
        md("## Unit tests"), code(TESTS),
        md("## Tiny sanity run"), code(COMMON), code(TINY),
        md("## Main experiment: `microsoft/bitnet-b1.58-2B-4T-bf16`\n"
           "Pick `PRESET`. Each mode reloads the frozen base (from the HF cache after the first download)."),
        code(PRESETS), code(MAIN),
        md(SWEEP_MD), code(SWEEP),
        md("## Plots"), code(PLOTS),
        md("## Save to Google Drive (optional)"), code(DRIVE),
    ]
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def check(nb_path: Path, workdir: Path):
    """Execute code cells locally in `workdir`: '!' lines -> shell (pip / nvidia-smi / git clone skipped), %cd skipped
    (so the embedded-sources fallback cell is exercised),
    Drive cell skipped, display() -> print."""
    nb = json.loads(nb_path.read_text())
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    sys.path.insert(0, str(workdir))
    os.environ["LATENT_NB_LOCAL_CHECK"] = "1"
    repo = MM.parent
    os.environ.setdefault("BITNET_VOCAB_GGUF", str(repo / "3rdparty/llama.cpp/models/ggml-vocab-llama-bpe.gguf"))
    gguf_py = str(repo / "3rdparty/llama.cpp/gguf-py")
    os.environ["PYTHONPATH"] = gguf_py + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, gguf_py)
    import matplotlib

    matplotlib.use("Agg")
    g = {"__name__": "__main__", "display": print}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "google.colab" in src:
            print(f"[check] cell {i}: skip (Drive)")
            continue
        if src.startswith("%%writefile"):
            first, body = src.split("\n", 1)
            path = first.split(None, 1)[1].strip()
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(body if body.endswith("\n") else body + "\n")
            continue
        py_lines = []
        for line in src.splitlines():
            if line.lstrip().startswith("%cd"):
                print(f"[check] cell {i}: skip `{line.strip()}` (local check uses the embedded fallback)")
                continue
            if line.lstrip().startswith("!"):
                cmd = line.lstrip()[1:]
                if "pip install" in cmd or "nvidia-smi" in cmd or "git clone" in cmd:
                    print(f"[check] cell {i}: skip `{cmd.strip()}`")
                    continue
                print(f"[check] cell {i}: $ {cmd.strip()}", flush=True)
                r = subprocess.run(cmd, shell=True, cwd=workdir)
                if r.returncode:
                    raise SystemExit(f"cell {i}: `{cmd}` failed ({r.returncode})")
            else:
                py_lines.append(line)
        if any(x.strip() for x in py_lines):
            print(f"[check] cell {i}: exec python ({len(py_lines)} lines)", flush=True)
            exec(compile("\n".join(py_lines), f"<cell {i}>", "exec"), g)
    print("[check] all code cells ran")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true", help="execute the code cells locally on CPU (tiny model)")
    ap.add_argument("--workdir", default=None, help="scratch dir for --check")
    args = ap.parse_args()
    nb = build()
    out = Path(args.out)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    json.loads(out.read_text())
    try:
        import nbformat

        nbformat.validate(nbformat.read(str(out), as_version=4))
        print(f"wrote {out} ({len(nb['cells'])} cells, nbformat-valid)")
    except ImportError:
        print(f"wrote {out} ({len(nb['cells'])} cells, valid JSON; nbformat not installed)")
    if args.check:
        if not args.workdir:
            raise SystemExit("--check needs --workdir")
        check(out, Path(args.workdir).resolve())


if __name__ == "__main__":
    main()
