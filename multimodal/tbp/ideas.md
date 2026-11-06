# Ternary Backpropagation: findings so far and experiment ideas

Working notes for designing TBP (`docs/ternary-backpropagation.md`, code in `multimodal/tbp/`). The plan: settle the
update rule and its hyperparameters on small models, design controllers based on training statistics, then run the
final experiment on a language model (BitNet b1.58 2B4T).

Priority: **P1** next, **P2** after, **P3** nice to have. Cost assumes the GTX 1650 (4 GB) unless noted.

## 0. Current state (summary)

**Defaults** (MLP trainer `tbp/tbp.py`): k = 0.5% of fan-in per row, T = 8 with 4-bit counters (6 bits/weight of
state), softmax signal with **τ = 2**, **batch 512**, ternary init, LayerNorm after every hidden layer.
LM path (`tbp/llm.py`): **learnable fp32 scales on q/k/o/down** (`--learn-scale odqk`, default in `tiny_lm.py`),
embedding and norms trained by Adam in TinyStories runs, int8 forward matmuls.

**Adopted (passed 6 paired seeds on both nets):** τ = 2 (+0.24 / +0.26 points), batch 512 (+0.19 / +0.19, faster).
On the LM: learnable q/k/o/down scales (TinyStories −0.15 nats).

**Laws found:** T\* ≈ 4 + log₂(width/256) (4-bit counters cover 2B4T's fan-ins except `down_proj`); best late flip
rate ≈ 4 × 10⁻⁵ per weight per step on the MLPs; reversal rate is the instability signal.

**Not adopted:** significance threshold, counter reset / decay, cooldown, vote strength, stagger, flip-every,
learnable LayerNorm (MLP), exact backward, adaptive k, validation-driven T schedule, reversal-rate controller (limit
cycle), binary init, learned per-layer scale (MLP).

**Gap to STE + AdamW at equal size:** ~1.6–1.9 points on Fashion-MNIST (train accuracy 0.90 vs 0.94); 0.75 nats on
TinyStories (2.92 vs 2.17). The gap is in fitting and comes from the update rule (one-state top-k flips gated by
counters), not from the error signal: exact gradients close only 0.1–0.2 points.

**2B4T result:** with the same flip counts per layer per step, TBP lowers held-out test perplexity 16.30 → 15.07
(−7.5%, best at step 300) while random flips raise it to 16.37. The gain peaks early and drifts back later.

**Running now:** 2B4T `final_tbp_sc` (learnable q/k/o/down scales; validation 14.63 → 13.62 at step 50, same as the
fixed-scale run so far), then `final_tbp2` (scales + cosine decay of k from 0.05% to 0.005%, fractional per row).

## 1. What we know so far

| Topic | Finding | Evidence |
|---|---|---|
| Core rule | Votes `G = Eᵀ·input`, top-k per row, one-state steps; the literal "top 5% by average contribution" rule does not learn | digits: ~0.10 → 0.894 |
| Flip counter T | Essential: T = 1 collapses late; T = 2/4/8 → 0.935/0.956/0.965 (digits, k = 0.5%). Wider layers want larger T (T = 16 best at width 39k) | `ctr_T*.json`, `big_T*.json` |
| Signal | Softmax error (`p − onehot`) beats the perceptron signal, mostly in deeper nets (Fashion-MNIST 3 layers: 0.857 → 0.874) | `fmnist_sm.json` |
| Normalization | Hidden-layer LN is required (without it 99% of units die). Per-example error/input normalization helps the perceptron signal but breaks the softmax signal (digits 0.858, Fashion-MNIST 0.68–0.75) | `dg_*.json`, `fmnist_smn.json` |
| k | 5% never stable; 0.2–1% works; 0.5% best on MLPs | digits sweeps |
| Equal memory | Digits: matches STE + Adam at equal training memory. Fashion-MNIST: ~1 point behind STE (0.874 vs 0.884, 3 layers) and fits the training set less well | `fmnist*.json` |
| Depth | Residual connections keep TBP from degrading with depth (20 layers: 0.867 residual vs 0.850 plain) | `runs/res_L*.json` |
| Counters | 4-bit counters are bit-identical to int8 at T ≤ 8 (6 bits/weight of state). 3-bit caps T at 4 (unstable at 10 × 1024: 0.780); stochastic increments p = 3/7 recover most of it (0.852 vs 0.859) | GPU runs |
| GPU | CUDA kernels: 20–29× faster than the CPU prototype; 1.75× faster than STE + Adam per epoch at 10 × 1024, 28% less energy, 14× less training state | `runs/bench_gpu/` |
| Adaptive k (validation) | No gain (residual +0.05, plain −0.05 points, 3 seeds); always hits its floor and freezes the net early; adds seed variance | `runs/ablate/` |
| Learned per-layer scale | Plain net: −0.71 points (it is only a logit temperature there, which rises to ~1.9×). Residual: mixed, scales keep drifting up | `runs/ablate/` |
| Counter reset every 32 steps | Clearly harmful (−8 points on average, two late collapses): rare but consistent votes are real signal | `runs/stability/` |
| Mean threshold (\|G\| > row mean) | Never binds at k = 0.5% (0 of 2.15M picks): a no-op | `runs/stability/` |
| Binary init {−1, +1} | Same accuracy as ternary init within noise; ends much less sparse (13–21% zeros vs 42–46%) and needs 20–40% more flips | `runs/init_study/` |
| T schedule (validation) | Ratchets T up to 14–16 under noisy validation; −0.44/−0.53 points, underfits | `runs/tsched/` |
| 2B4T, pretrained | 600 steps, k = 0.05%: held-out perplexity 14.61 → 13.76 (step 100) → 14.02 (step 600). No random-flip control yet | `runs/tbp_2b4t/tbp_k5e-4/` |
| 2B4T, from scratch | Plateau at loss ~9.27, capped by the frozen random embedding (best single-vector loss with that head: 9.46; unigram 7.74). Flips constant at ~7,800/step | `runs/tbp_2b4t/tbp_scratch_50m/` |
| 2B4T, random linears + pretrained embedding | Starts at loss ~22 (pretrained head, random hidden states); perplexity 7,527 at step 100 (batch 2) | `runs/tbp_2b4t/scratch_emb_b2/` |

Recurring lesson: controllers driven only by noisy validation checks drift one way (k down to its floor, T up to its
ceiling). Fast, local training statistics are the more promising control signal, with validation as a guard.

## 1b. Step-by-step results (in progress)

Fashion-MNIST, 15 epochs, 55k train / 5k validation / 10k test, GPU, 3 seeds unless marked. Differences are final test
accuracy against the same-seed GPU baseline (k = 0.5%, T = 8), in points. Residual = 10 × 1024, plain = 3 × 1786.
Baselines: residual 0.8664 ± 0.0021, plain 0.8702 ± 0.0024.

| Experiment | Result | Verdict |
|---|---|---|
| **Instrumentation** (`--instrument`) | Reversal rate separates unstable settings: baseline 0.10 → 0.17 over training, T = 1 0.50–0.71, k = 5% up to 0.22, counter reset 0.21–0.27. Vote agreement falls 0.84 → 0.70 as learning slows (a signal-to-noise measure, not a stability one). Flip productivity is too noisy per epoch | Reversal rate = control signal for T; agreement = candidate for flip-rate decay |
| **Significance threshold** z × row mean (top-k / mean ratio: median 3.5 residual, 6 plain) | z = 2.5: residual −0.04, plain +0.25 (3/3); z = 3.5: −0.22 / +0.09; z = 5: −1.61 / −0.48 | Blocking weaker votes removes signal. Not adopted |
| **T against width** (1 hidden layer, widths 256/1,024/4,096) | Best T: 4 / 4–8 (tied) / 8. T too low is worst and worsens with width (T = 2 at 4,096: 0.772). One doubling off the best costs 0.5–1 point | **T\* ≈ 4 + log₂(width/256)**. 2B4T (fan-in 2,560–6,912): T ≈ 7–9, so 4-bit counters almost suffice |
| **k sweep** (0.1–2%) | Residual: every k ≠ 0.5% loses 0.2–0.6; plain: 0.2–1% within noise, 0.1% −0.43, 2% −0.73. Optimal late flip rate ≈ 4 × 10⁻⁵ per weight per step; doubling k raises flips only ~1.5× | k = 0.5% stays the default |
| Counter decay (−1 every 200 steps), 2 seeds | −0.07 / −0.17; halves late flips (224 vs 411) | Neutral so far |
| **Cooldown** (no flip back within 200 steps), 2 seeds | +0.14 / +0.07 | Promising, waiting for seed 3 |
| Vote strength (votes > 4 × mean count twice), 2 seeds | −0.72 / −0.38; flips 2.7× | Harmful |
| Stagger (half the layers per step), 2 seeds | −4.4 / −1.5 | Harmful |
| Flip every 4 steps, 2 seeds | −2.8 / −4.3 | Harmful |
| **Softmax temperature τ = 2**, 1 seed | +0.38 (τ = 8: −0.42) | Promising, waiting for more seeds |
| **STE learning-rate sweep** (residual 2 seeds, plain 1–2) | lr 1e-3 is best: residual 0.886 / 0.886, plain 0.882 (lr 3e-3: 0.877–0.878 / 0.859; 1e-2: 0.866–0.878 / 0.852) | **Tuned STE is ~2 points above TBP (residual) and ~1.2 (plain)**: the real gap is larger than the ~1 point measured with lr 3e-3 |
| Bop2ndOrder (τ = 0.5/1/2, γ = 1e-3, δ = 1e-5) | 0.52–0.79: tens of thousands of flips per step early, then almost none | My Bop setup is untuned (γ, δ, τ need a sweep); not a fair baseline yet |
| Reversal-rate T controller | Running | — |
| TinyStories (4.7M ternary weights, 4k BPE, 22.5M tokens) | Pipeline works (`tbp/tiny_lm.py`); full runs queued (TBP T = 4/8, STE lr 1e-3/3e-3) | — |

### Update (6-seed confirmations, learnable LN, controllers, baselines)

| Experiment | Result | Verdict |
|---|---|---|
| **Softmax temperature τ = 2** (6 paired seeds) | Residual +0.24 (6/6 positive, paired t 4.33); plain +0.26 (5/6, t 3.42) | **Adopted as the default** (`--tau 2`, `TBPNet.step(tau=2.0)`) |
| Cooldown 200 steps (6 seeds) | Residual +0.14 (5/6, t 1.54); plain +0.02 (3/6) | Not adopted (only the residual net) |
| τ = 2 + cooldown (6 seeds) | Residual +0.14, plain +0.16: worse than τ = 2 alone | Not adopted |
| Counter decay 200 (3 seeds) | +0.07 mean, halves late flips | Neutral |
| Vote strength 4 / stagger 2 / flip every 4 (3 seeds) | −0.57 / −2.41 / −3.44 | Harmful |
| τ = 8 (3 seeds) | −0.65 (3/3 negative) | Harmful: confirms the direction of τ |
| Reversal-rate T controller (3 seeds) | Residual −0.55, plain −0.84. T oscillates 6 ↔ 16 in a limit cycle: the reversal rate reacts to T with a delay (200-step window) | Not adopted; would need a damped loop (fractional T steps, several consistent readings) |
| **Learnable LayerNorm** gain / gain + bias / RMS gain, Adam lr 1e-3, 3e-3, 1e-2 (3 seeds) | All within ±0.2 points (best: plain gain + bias +0.12 to +0.18 at every lr). At lr 1e-2 the gains move a lot (spread 0.45–0.58, range −0.5 to 4.8) but train accuracy rises only 0.4–0.7 points | Not adopted. Per-channel scale/shift is not what limits TBP |
| **Tuned STE + AdamW** (lr 1e-3, 3 seeds) | Residual 0.8859 ± 0.0002 (train 0.938), plain 0.8862 ± 0.0051 (train 0.945): +1.94 / +1.60 over TBP at equal parameter count | The real gap at equal size; it is in fitting (train accuracy 4 points higher) |
| Bop2ndOrder: plain, bias-corrected with warm-up and momentum reset, and with TBP's per-row budget | 0.62–0.81 at best; flips tens of thousands of weights per step once past the threshold | Not competitive here; dropped |

**Defaults now:** k = 0.5%, T = 8 (4-bit counters), **τ = 2**, **batch 512**, softmax signal, ternary init.

### Update (τ sweep, batch size, exact backward, TinyStories)

| Experiment | Result | Verdict |
|---|---|---|
| τ ∈ {1, 1.5, 3} around the new default (3 seeds) | Residual flat (−0.08 to −0.02); plain τ = 1.5 −0.27, τ = 1 −2.87 (unstable), τ = 3 +0.01 | τ = 2 stays |
| **Batch 512** (6 paired seeds) | Residual +0.19 (4/6, t 1.92), plain +0.19 (5/6, t 2.81); half the steps per epoch, 20% faster on the residual net | **Adopted as the default** (`--bs 512`) |
| Batch 1,024 (3 seeds) | Residual 0.00, plain +0.18 with a quarter of the steps | Same accuracy for far fewer updates |
| **Exact backward** (votes = true weight gradient via autograd; 6 seeds) | Residual 0.00 (2/6), plain +0.15 (6/6, t 5.54); train accuracy +0.6–0.9; 30–55% slower | Not adopted. The approximate error signal costs only ~0.1–0.2 points: the gap to STE is in the update rule, not the signal |
| Exact backward + batch 512 (6 seeds) | Residual +0.38 (5/6, t 3.59), plain +0.11 | Best residual result; option `--exact-backward` |
| **TinyStories** (4.7M ternary weights, 3,000 steps × 8,192 tokens) | TBP T = 4: val perplexity 22.97 (loss 3.13); TBP T = 8: 25.58; **STE lr 1e-3: 9.63 (loss 2.26)** | The width rule holds (T = 4 > T = 8 at fan-in 256–768). **The gap to STE is much larger on a language model (0.87 nats) than on the MLPs** |

**TinyStories sweep** (validation loss, nats/token, 3,000 steps unless noted):

| Run | Val loss | Flips/step |
|---|---|---|
| TBP T = 8 / 6 / 4 / 2 | 3.24 / 3.20 / 3.13 / 3.11 | 430 / 832 / 2,167 / 7,912 |
| TBP T = 4, k = 2% | **3.07** | 8,925 |
| TBP T = 4, 6,000 steps | 2.91 (still falling slowly) | 2,157 |
| STE lr 1e-3 / 3e-3 | 2.26 / **2.17** | — |

More flips help with diminishing returns (4× the flips ≈ −0.06 nats), and doubling the steps reaches 2.91, still
0.74 nats above STE at half the steps. Next candidate: **TBP keeps each layer's scale fixed at its init value**,
while STE's BitLinear recomputes it every step; in BitNet the `o_proj` / `down_proj` scales set each block's
contribution to the residual stream and are not normalized away. Test a learnable fp32 scale on those (and q/k for
attention temperature).

**Learnable scales on the LM** (TinyStories, TBP T = 4, k = 2%): fixed 3.070 → o/down 2.995 → **o/down + q/k 2.917**
(−0.15 nats; 1 seed, steady along the curve). Unlike the MLPs, these scales are not normalized away in BitNet.
Adopted as the `tiny_lm` default (`--learn-scale odqk`); available in `train_lm` as `--learn-scale`.

**2B4T fine-tuning with a random-flip control** (pretrained, batch 2, T = 8, k = 0.05%, 600 steps ≈ 615k tokens,
validation 50 / test 100 held-out sequences, int8 forward at 104 tok/s):

| | step 0 | step 300 | step 600 |
|---|---|---|---|
| TBP test perplexity | 16.30 | **15.07 (−7.5%)** | 15.33 (−6.0%) |
| Random flips (same counts per layer per step) | 16.30 | 16.33 | 16.37 |

**TBP's votes carry the gain**: identical flip counts at random positions slightly degrade the model. TBP's best
validation is at step 150 and it drifts back later (13.60 → 13.99), so the next 2B4T runs should decay the flip rate
(fractional per-row k in `TBPLinear`), keep the best validation checkpoint, and add learnable q/k/o/down scales.
int8 forward matmuls gave +17% tokens/s (the fp32 backward now dominates).

Fitting remains the core limitation: even the best combination trains to 0.908–0.912 on Fashion-MNIST vs ~0.94 for
STE, and the language model shows the gap far more strongly. Next things to understand before 2B4T:
- whether TBP's TinyStories loss is still falling (more steps) or has levelled off;
- the flip rate on the LM: T = 4 makes ~2,170 flips per step on 4.7M weights (≈ 4.6 × 10⁻⁴ per weight per step,
  10× the MLP optimum), so the LM may need a lower k or a higher T than the width rule alone suggests;
- larger LM batches (votes over more tokens), the exact backward, and a k / T sweep on TinyStories.

**What the remaining gap points to:** TBP fits the training set ~4 points worse than tuned STE, and neither per-channel
norm parameters, thresholds, schedules nor controllers changed that. The open candidates are the error signal itself
(an exact LayerNorm backward instead of treating its statistics as constants) and the evidence per update (larger
effective batch).

## 2. Update-rule stability

| Idea | Why | Cost | Priority |
|---|---|---|---|
| **Significance threshold with a real noise reference**: first measure the ratio top-k \|G\| / row mean and median, then sweep z where it starts to bind (z ≈ 5–20) | The row mean is too low a bar; a threshold that binds should make flips self-limiting when votes become noise | 15 min | P1 |
| **Vote agreement between half-batches** as the noise reference: vote only on weights whose two half-batch votes agree | Direct signal-to-noise estimate per weight, no tuning of z | kernel + 30 min | P1 |
| Gentle counter decay: −1 toward 0 every N steps (N in the hundreds), instead of resetting | Forget truly stale votes without discarding rare consistent ones (the reset showed those matter) | 30 min | P2 |
| Hysteresis: a weight flipped within the last W steps needs 2T votes to flip back | Targets oscillation directly; 4-bit counters have spare states | kernel change | P2 |
| Vote strength: +2 to the counter for votes far above the noise level, +1 otherwise | Strong evidence flips sooner, weak repeated noise slower (closer to Bop) | small | P2 |
| Scale tied to the count of nonzero weights (`s ∝ 1/√nnz` per row or tensor) instead of a learned scale | Keeps layer gain constant as sparsity drifts (13% → 46% zeros observed); learned scales drifted instead | small | P2 |
| Staggered layer updates: vote everywhere, flip only a subset of layers per step | Reduce interference from all layers flipping on pre-update errors; targets the depth gap | small | P2 |
| Vote every step, flip every M steps | Counters gather more evidence between flip rounds; fewer interacting flips | small | P3 |
| Flip log + rollback: keep recent flip indices, undo the last interval after a validation spike | Cheap trust region without full checkpoints | medium | P3 |

## 3. Hyperparameter laws (how many bits for T, how large k)

| Idea | Why | Cost | Priority |
|---|---|---|---|
| **T against width sweep**: T ∈ {2, 4, 8, 16, 32} × width {256, 1024, 4096} (× depth), 3 seeds | Test the T ≈ T₀ + log₂(width/width₀) rule; decides 4 vs 5 counter bits for 2B4T (fan-in 2,560–6,912) | 1–2 h GPU | P1 |
| k expressed as a target flip rate (flips per weight per step) and swept | A "learning rate" for ternary weights; measured rates: 4.4e-5 (Fashion-MNIST) vs 3.7e-6 (2B4T scratch) | 1 h | P1 |
| k × T trade-off at a constant flip rate | Larger k + larger T averages more votes per flip | 1 h | P2 |
| Stochastic increments as fractional T (T_eff = 1 + (T−1)/p) at 4 bits | Reach T_eff 12–16 without a 5th bit | 30 min | P2 |
| Softmax temperature τ sweep (only τ = 4 tested) | Untested sensitivity | 30 min | P2 |
| Joint k × T grid with 5 seeds on Fashion-MNIST | Sensitivity analysis with error bars | 2 h | P3 |

## 4. Controllers from training statistics

| Idea | Signal | Lever | Priority |
|---|---|---|---|
| **Reversal-rate controller** | Share of flips that undo a recent flip of the same weight | T up when > ~15%, down when < 5% | P1 |
| Smoothed validation trend | Slope of the last 4–6 validation losses, change only after 2–3 confirmations, tolerance from measured noise | T or k | P2 |
| Flip productivity | Training-loss drop per flip | Flip-rate schedule (decay as it approaches 0) | P2 |
| Agreement-based significance | Half-batch vote agreement per layer | z (which candidates vote) | P2 |
| Adaptive k, fixed | Higher floor (~0.1% or 1 per row), tolerance 0.5%, patience 3–4 | k | P3 |
| Validation as a guard only | Spike detection | Rollback + tighten | P2 |

### Instrumentation needed first (P1)
- Reversal rate (per layer type), vote agreement (half-batches), counter occupancy (share near ±(T−1)), flip
  productivity, flips per step over time. Log all of them in `tbp.py` and `train_lm.py`.

## 5. Baselines and fairness

| Idea | Why | Priority |
|---|---|---|
| **Bop baseline** (ternary, momentum threshold on float accumulators) at equal parameters and equal training memory | The closest prior method; not compared yet | P1 |
| **Learning-rate sweep for STE + Adam** (plain 20-layer STE diverged at the single lr 3e-3) | STE numbers are untuned | P1 |
| More seeds (5) where effects are < 0.5 points; always report mean ± std | Several single-seed "gains" disappeared with 3 seeds | P1 |
| Equal compute / FLOPs comparisons, not only equal memory and wall-clock | Reviewer question | P2 |
| Direct feedback alignment variant on the GPU code | Was 2–4 points behind on digits; no backward pass | P3 |
| Consistent bits/weight convention in the docs (state counted at 10 vs 6 bits; STE counted without the gradient buffer) | Matched-memory claims depend on it | P1 (docs) |

## 6. Benchmarks and datasets

| Dataset | Tests | Fits now? | Priority |
|---|---|---|---|
| **TinyStories** with a 10–20M-parameter BitNet | Attention, norms, vocab softmax: the dress rehearsal for 2B4T | Yes (`tbp/llm.py` works on tiny configs) | P1 |
| **HIGGS** (11M rows, 500k test) | Long-run flip stability, very low-noise test set | Yes (MLP) | P1 |
| **CIFAR-10 / CIFAR-100**, MLP then CNN | Harder fitting, larger method gaps | MLP yes; CNN needs a conv path (im2col + TBPLinear) | P2 |
| enwik8 (bits per character) | Standard small-LM benchmark | Yes | P2 |
| EMNIST-balanced / KMNIST | More classes, quick step up | Yes | P3 |
| ImageNet 32×32 | 1,000 classes, scale | Slow | P3 |
| CIFAR-10-C / MNIST-C | Robustness under distribution shift, TBP vs STE | Yes | P3 |
| Algorithmic: parity, modular addition, ListOps | Credit assignment, unstable late jumps | Yes | P2 |

## 7. The language-model experiment (2B4T)

| Idea | Why / evidence | Priority |
|---|---|---|
| **Random-flip control** (same flips per layer per step at random positions) | Without it, the pretrained perplexity gain cannot be attributed to TBP | P1 |
| **int8 forward matmuls** (`torch._int_mm`): 1.61 ms vs 7.77 ms fp32, bit-exact | Est. ~1.5× faster steps; fp16 is 7× slower on this GPU | P1 |
| Validation / test split for 2B4T (decide on 50 sequences, report on 100 others, evaluate less often) | Evaluation now costs 26% of training time and doubles as the reported number | P1 |
| Batch 2 × 512 (fits, same tokens/s) | Votes average 1,024 tokens: fewer noisy flips | P1 (flag exists) |
| Fine-tuning with lower k / higher T (e.g. k 0.02% via fractional k, or T 16 with int8 counters) | Pretrained run gave back part of its gain after step 300 | P1 |
| Continue `scratch-pretrained-emb` (random linears, pretrained embedding + norms) | Perplexity 7,527 at step 100, already below the all-random plateau | P2 |
| For full scratch: trainable embedding (bf16 + factored optimizer + stochastic rounding), trainable norms, or at least a trainable output bias (alone it restores the unigram level, 7.74) | Frozen random head caps loss at ~9.46 | P2 |
| Learnable scales only where nothing cancels them (q/k: attention temperature; o/down: residual contribution) | v and gate/up feed sub-norms, which cancel their scales | P3 |
| Keep best checkpoint by validation; flip-reversal and k logging | Rollback after spikes like step 4,800 | P2 |
| Streaming data for long scratch runs (only 5M tokens cached) | | P3 |
| Zero-shot capability evaluation (`lm-evaluation-harness`: LAMBADA, HellaSwag, PIQA, WinoGrande, ARC; GSM8K / MMLU) | Perplexity alone does not show what fine-tuning preserves or damages | P2 |
| Partial checkpointing (every other layer), fusing activation quantization into the matmul | Less recomputation; ~7.5% of step time is kernel-launch overhead | P3 |

## 8. Tooling and repo

| Item | Status | Priority |
|---|---|---|
| Fix `run_inference.py`: use `llama-completion` for completion, `-sys` + `--chat-template-file` (2B4T's HF template) for chat, close stdin, update the default model path | `llama-cli` ignores `-no-cnv` and loops on EOF; the GGUF's chat template is wrong (`Human: … BITNETAssistant:` + EOS) | P2 |
| Regenerate the 2B4T GGUF with the llama-bpe pre-tokenizer | Loader warns "missing pre-tokenizer type" | P3 |
| Commit `multimodal/tbp/` and `docs/ternary-backpropagation.md`; fold these results into the doc | Everything is uncommitted | P1 |
| Seed-major ablation runner, per-run JSON (exists: `tbp/ablate.py`); add a summary plot script | | P3 |

## Suggested order
1. ~~Instrumentation + significance threshold z sweep~~ (done: reversal rate validated; threshold not adopted).
2. ~~T against width sweep; k as flip rate~~ (done: T\* ≈ 4 + log₂(width/256); k = 0.5%).
3. ~~Reversal-rate controller; Bop and tuned-STE baselines~~ (done: controller oscillates; Bop not competitive;
   tuned STE +1.6–1.9 points).
4. ~~Mechanism study, 6-seed confirmations, TinyStories~~ (done: τ = 2 and batch 512 adopted; LM gap 0.75 nats;
   learnable q/k/o/down scales −0.15 nats).
5. 2B4T: ~~int8 forward, val/test split, random-flip control~~ (done: +17% speed; TBP −7.5% test perplexity vs
   random +0.5%); learnable scales and k decay runs in progress.

Next after the running 2B4T runs:
- Keep the best validation checkpoint as the reported model (now saved as `best.pt`) and report its test perplexity.
- Learnable scales on all linears (`--learn-scale all`) on TinyStories; a 3-seed TinyStories repeat of the best
  setting; TinyStories with k decay.
- Zero-shot capability evaluation of the fine-tuned 2B4T (`lm-evaluation-harness`) against the base model.
- The fitting gap: candidates that change the update rule itself (multi-state steps for strong evidence, a per-row
  learned threshold, or a hybrid with a small float residual on the most-voted weights).

New ideas from the step-by-step results:
- A τ sweep around 2 (τ ∈ {1, 1.5, 2, 3}) if τ = 2 holds on all seeds.
- Cooldown windows (50 / 200 / 1,000 steps), possibly combined with the reversal-rate controller.
- `--T auto` per layer from the width rule, checked on a second architecture.
- Close the ~2-point gap to tuned STE: the gap is in fitting (train accuracy 0.894 vs ~0.91+), so look at signal
  quality (exact LN backward instead of treating its statistics as constants; larger effective batch).
- **LayerNorm with learnable parameters** (per-channel gain γ and bias β, fp32, trained by Adam or a tiny
  sign-based rule). Today every LN is affine-free and the ternary layers are the only trainable part, so the network
  cannot rescale or shift individual channels. Why it may help:
  - per-channel gain gives each hidden unit its own scale, which ternary weights (one shared magnitude per layer) and
    the per-layer scales (cancelled by the next LN) cannot provide; this may close part of the fitting gap to STE;
  - a bias lets each unit move its ReLU threshold, which ternary weights alone can only do by flipping many weights;
  - BitNet itself has learnable RMSNorm gains (`input_layernorm`, `post_attention_layernorm`, sub-norms), so the MLP
    experiments would match the LLM setting more closely;
  - cost is tiny: 2 × width fp32 values per layer (8 KB per 1,024-wide layer) plus Adam state.

  Things to check: the TBP votes for the layer below must include γ in the error (`e ⊙ γ` before the ReLU mask), so
  skip/branch balance and the vote ranking stay consistent; compare (a) gain only, (b) gain + bias, (c) RMSNorm-style
  gain without centering (as in BitNet); report fitting (train accuracy) as well as test, against the same-seed
  baseline, on the residual and plain nets with 3 seeds; and watch that the gains do not drift the way the per-layer
  scales did (log their mean and spread per epoch). Priority: P1, because it targets the fitting gap directly and makes
  the toy models closer to the LLM.
