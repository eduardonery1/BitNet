"""LatentReasoner: prompt -> R reasoning steps (latent / pause) -> answer, on a HF BitNetForCausalLM.

Sequence layout (all sequences in a batch share one layout; prompts are LEFT-padded so every row ends at the same
cache index):

    [pad..pad BOS program\\n] [thought step 1: K positions] ... [step R] [remaining trace lines] Answer: 23<|eot_id|>
     \\-------- prompt -------/ \\------------- R*K positions -------------/ \\---------- target (right-padded) -------/

Modes:
  none    R = 0, target = "Answer: <a><eot>".
  latent  each step t: soft = compressor(layer-i states of every position seen so far, step=t) (K vectors), fed via
          inputs_embeds while reusing the KV cache (only the K new positions are computed); their layer-i states are
          appended to the memory the compressor reads at step t+1.
  pause   the same R*K extra positions, but the inputs are learned constant pause embeddings (no feedback).
  cot     R = 0, target = all trace lines ("b = 5\\n...") + "Answer: <a><eot>", loss on all of it.

Loss: CE on target tokens only (trace lines, answer digits, eot). The "Answer:" separator is in the loss only when
trace lines precede it (cot / non-final curriculum stages), because then the model must decide when to stop.

Curriculum (Coconut): at stage s < S (S = R), the first min(s, n) trace steps are replaced by s thought steps and
the remaining trace stays explicit (with loss); at stage >= S all trace is latent with R thought steps.

Position ids / masks: transformers 5 BitNet uses ``position_ids = cache_position`` unless given, which is wrong for
left padding, so explicit position_ids = (number of real tokens before this one) are passed for every chunk, together
with a 2-D attention mask over (past + new) positions. cache_position (index into the cache) is left to the model
(= arange(new) + cached length), which is right because the cache layout is shared by all rows.

Gradient checkpointing: HF's GradientCheckpointingLayer drops the KV cache when checkpointing, so we use our own
per-decoder-layer checkpointing (`enable_checkpointing`) together with `LatentCache`, whose layers update
idempotently (keys = cache[:start] ++ new, stored only the first time), so recomputation in backward neither
duplicates entries nor reads later entries.
"""

from __future__ import annotations

import functools
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import Cache, DynamicLayer

MODES = ("none", "latent", "pause", "cot")
SEP = "Answer:"
IGNORE = -100


# ----------------------------------------------------------------------------------------------------------------
# KV cache + checkpointing
# ----------------------------------------------------------------------------------------------------------------
class CheckpointSafeLayer(DynamicLayer):
    """DynamicLayer whose update is a pure function of (cache[:start], new) with start = cache_position[0]."""

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        cur = self.keys.shape[-2] if self.keys.numel() else 0
        cp = (cache_kwargs or {}).get("cache_position")
        start = int(cp[0]) if cp is not None and cp.numel() else cur
        if start == cur:
            if cur == 0:
                self.keys, self.values = key_states, value_states
            else:
                self.keys = torch.cat([self.keys, key_states], dim=-2)
                self.values = torch.cat([self.values, value_states], dim=-2)
            return self.keys, self.values
        if start > cur:
            raise RuntimeError(f"cache gap: write at {start}, cache has {cur}")
        # recomputation (gradient checkpointing): rebuild without touching the stored cache
        return (torch.cat([self.keys[..., :start, :], key_states], dim=-2),
                torch.cat([self.values[..., :start, :], value_states], dim=-2))


class LatentCache(Cache):
    def __init__(self, num_layers: int):
        super().__init__(layers=[CheckpointSafeLayer() for _ in range(num_layers)])


class _CkptCallMixin:
    def __call__(self, *args, **kwargs):
        if getattr(self, "_latent_ckpt", False) and self.training and torch.is_grad_enabled():
            return checkpoint(functools.partial(nn.Module.__call__, self, **kwargs), *args, use_reentrant=False)
        return nn.Module.__call__(self, *args, **kwargs)


def enable_checkpointing(lm, enabled: bool = True):
    """Per-decoder-layer activation checkpointing that keeps the KV cache (see module doc)."""
    for layer in lm.model.layers:
        cls = type(layer)
        if not issubclass(cls, _CkptCallMixin):
            layer.__class__ = type(f"Ckpt{cls.__name__}", (_CkptCallMixin, cls), {})
        layer._latent_ckpt = enabled
        layer.gradient_checkpointing = False  # never use HF's (it drops past_key_values)


# ----------------------------------------------------------------------------------------------------------------
# formatting / curriculum
# ----------------------------------------------------------------------------------------------------------------
class Formatter:
    def __init__(self, tokenizer, sep: str = SEP):
        self.tok = tokenizer
        vocab = tokenizer.get_vocab()
        self.bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else vocab["<|begin_of_text|>"]
        self.eot = vocab["<|eot_id|>"] if "<|eot_id|>" in vocab else tokenizer.eos_token_id
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eot
        self.sep_text = sep
        self.sep = self.enc(sep)

    def enc(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids if text else []

    def prompt(self, text: str) -> list[int]:
        return [self.bos] + self.enc(text)

    def trace(self, lines: list[str]) -> list[int]:
        return self.enc("".join(line + "\n" for line in lines))

    def answer(self, ans) -> list[int]:
        return self.enc(f" {ans}") + [self.eot]


def plan_example(ex: dict, mode: str, R: int, stage: int | None = None) -> tuple[int, list[str]]:
    """-> (number of thought steps, explicit trace lines kept before the answer).

    stage=None means "no curriculum" (final setting). For latent/pause with a curriculum, there are S+1 stages
    0..S with S = R: stage s < S keeps trace[min(s, n):] and uses s thought steps; stage >= S is all-latent."""
    trace = list(ex["trace_steps"])
    if mode == "none":
        return 0, []
    if mode == "cot":
        return 0, trace
    if stage is None or stage >= R:
        return R, []
    return stage, trace[min(stage, len(trace)):]


def num_stages(R: int) -> int:
    return R + 1


def _left_pad(seqs: list[list[int]], pad: int):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, L - len(s):] = 1
    return ids, mask


def _right_pad(seqs: list[list[int]], pad: int, fill=None):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad if fill is None else fill, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return ids, mask


_NUM_RE = re.compile(r"-?\d+")


def parse_answer(text: str, forced_sep: bool) -> int | None:
    if not forced_sep:
        if SEP not in text:
            return None
        text = text.split(SEP, 1)[1]
    m = _NUM_RE.search(text)
    return int(m.group()) if m else None


# ----------------------------------------------------------------------------------------------------------------
# reasoner
# ----------------------------------------------------------------------------------------------------------------
class LatentReasoner(nn.Module):
    """
    Args:
        lm: BitNetForCausalLM (with BitLinear / LoRA already applied).
        compressor: LatentCompressor (mode 'latent'); pause: PauseTokens (mode 'pause').
        read_layer: HF hidden_states index i in [1, num_layers]: output of decoder layer i-1 (i = num_layers is the
            final-norm output, as in HF's hidden_states[-1]).
        K: soft tokens per step. R: default thought steps. bptt: if > 0, when producing thought t the compressor's
            memory entries from thoughts older than the last `bptt` steps are detached (truncated recurrence).
    """

    def __init__(self, lm, compressor=None, read_layer: int | None = None, K: int = 4, mode: str = "latent",
                 tokenizer=None, pause=None, R: int = 4, bptt: int = 0, sep: str = SEP):
        super().__init__()
        assert mode in MODES, mode
        self.lm, self.compressor, self.pause = lm, compressor, pause
        self.mode, self.K, self.R, self.bptt = mode, K, R, bptt
        n = lm.config.num_hidden_layers
        self.num_layers = n
        self.read_layer = read_layer if read_layer is not None else default_read_layer(n)
        assert 1 <= self.read_layer <= n
        if mode == "latent":
            assert compressor is not None and compressor.K == K
        if mode == "pause":
            assert pause is not None and pause.K == K
        self.fmt = Formatter(tokenizer, sep) if tokenizer is not None else None
        self._capture = False
        self._captured = None
        self._hook = None
        if self.read_layer < n:
            self._hook = lm.model.layers[self.read_layer - 1].register_forward_hook(self._hook_fn)
        self.last_thought_inputs = None  # (B, R*K, d) of the last think() call (for tests / analysis)

    def _hook_fn(self, module, args, output):
        if self._capture:
            self._captured = output[0] if isinstance(output, tuple) else output

    @property
    def device(self):
        return self.lm.get_input_embeddings().weight.device

    @property
    def emb_dtype(self):
        return self.lm.get_input_embeddings().weight.dtype

    # ------------------------------------------------------------------------------------------ low level
    def lm_step(self, embeds, attn_mask, position_ids, cache, capture: bool):
        """One forward over new positions. Returns (final-norm hidden (B,n,d), layer-i hidden or None)."""
        self._capture, self._captured = capture, None
        try:
            out = self.lm.model(inputs_embeds=embeds, attention_mask=attn_mask, position_ids=position_ids,
                                past_key_values=cache, use_cache=True)
        finally:
            self._capture = False
        h = out.last_hidden_state
        hi = None
        if capture:
            hi = h if self.read_layer == self.num_layers else self._captured
        self._captured = None
        return h, hi

    def think(self, prompt_ids, prompt_mask, R: int):
        """Prompt pass + R thought steps. Returns a state dict with the cache, the running 2-D mask, the next
        position id per row and the final hidden state of the last position."""
        dev = self.device
        prompt_ids, prompt_mask = prompt_ids.to(dev), prompt_mask.to(dev)
        B = prompt_ids.shape[0]
        cache = LatentCache(self.num_layers)
        pos = (prompt_mask.cumsum(1) - 1).clamp(min=0)
        emb = self.lm.get_input_embeddings()(prompt_ids)
        need_mem = self.mode == "latent" and R > 0
        h, hi = self.lm_step(emb, prompt_mask, pos, cache, capture=need_mem)
        mask = prompt_mask
        next_pos = prompt_mask.sum(1)
        last = h[:, -1]
        thoughts = []
        if R > 0 and self.mode in ("latent", "pause"):
            ar = torch.arange(self.K, device=dev)
            if self.mode == "pause":
                soft = self.pause(B, R).to(self.emb_dtype)
                n = soft.shape[1]
                p = next_pos[:, None] + torch.arange(n, device=dev)
                mask = torch.cat([mask, mask.new_ones(B, n)], 1)
                h, _ = self.lm_step(soft, mask, p, cache, capture=False)
                next_pos = next_pos + n
                last = h[:, -1]
                thoughts.append(soft)
            else:
                mem_h, mem_m = [hi], [prompt_mask]
                for t in range(R):
                    chunks = list(mem_h)
                    if self.bptt > 0:
                        # chunk j (j >= 1) holds thought j-1; keep grads only for the last `bptt` thoughts
                        for j in range(1, len(chunks)):
                            if (j - 1) < t - self.bptt:
                                chunks[j] = chunks[j].detach()
                    memory = torch.cat(chunks, 1)
                    mmask = torch.cat(mem_m, 1)
                    soft = self.compressor(memory, mmask, step=t).to(self.emb_dtype)
                    p = next_pos[:, None] + ar
                    mask = torch.cat([mask, mask.new_ones(B, self.K)], 1)
                    h, hi = self.lm_step(soft, mask, p, cache, capture=True)
                    next_pos = next_pos + self.K
                    mem_h.append(hi)
                    mem_m.append(mask.new_ones(B, self.K))
                    last = h[:, -1]
                    thoughts.append(soft)
        self.last_thought_inputs = torch.cat(thoughts, 1) if thoughts else None
        return dict(cache=cache, mask=mask, next_pos=next_pos, last=last)

    def feed(self, state, ids, ids_mask=None):
        """Append token ids (B, T) to the state; returns final hidden (B, T, d) and updates the state."""
        dev = self.device
        ids = ids.to(dev)
        B, T = ids.shape
        ids_mask = torch.ones_like(ids) if ids_mask is None else ids_mask.to(dev)
        mask = torch.cat([state["mask"], ids_mask], 1)
        p = state["next_pos"][:, None] + torch.arange(T, device=dev)
        h, _ = self.lm_step(self.lm.get_input_embeddings()(ids), mask, p, state["cache"], capture=False)
        state.update(mask=mask, next_pos=state["next_pos"] + T, last=h[:, -1])
        return h

    # ------------------------------------------------------------------------------------------ training
    def build_batch(self, examples: list[dict], R: int | None = None, stage: int | None = None) -> dict:
        R = self.R if R is None else R
        f = self.fmt
        plans = [plan_example(ex, self.mode, R, stage) for ex in examples]
        r = {p[0] for p in plans}
        assert len(r) == 1
        prompts = [f.prompt(ex["prompt"]) for ex in examples]
        tgts, labs = [], []
        for ex, (_, lines) in zip(examples, plans):
            tr = f.trace(lines)
            ans = f.answer(ex["answer"])
            ids = tr + f.sep + ans
            sep_lab = f.sep if lines else [IGNORE] * len(f.sep)
            tgts.append(ids)
            labs.append(tr + sep_lab + ans)
        pid, pmask = _left_pad(prompts, f.pad)
        tid, tmask = _right_pad(tgts, f.pad)
        lab, _ = _right_pad(labs, IGNORE, fill=IGNORE)
        return dict(prompt_ids=pid, prompt_mask=pmask, target_ids=tid, target_mask=tmask, labels=lab,
                    R=r.pop(), stage=stage)

    def forward(self, batch: dict) -> dict:
        """Teacher-forced loss on the target tokens. Returns dict(loss, n_tokens)."""
        state = self.think(batch["prompt_ids"], batch["prompt_mask"], batch["R"])
        prev_last = state["last"]  # final hidden of the last prompt/thought position: predicts target[0]
        h = self.feed(state, batch["target_ids"], batch["target_mask"])
        labels = batch["labels"].to(h.device)
        pred_h = torch.cat([prev_last[:, None], h[:, :-1]], 1)
        sel = labels != IGNORE
        logits = self.lm.lm_head(pred_h[sel]).float()
        return dict(loss=F.cross_entropy(logits, labels[sel]), n_tokens=int(sel.sum()))

    # ------------------------------------------------------------------------------------------ inference
    @torch.no_grad()
    def generate(self, prompt_ids, prompt_mask, R: int, force_sep: bool, max_new_tokens: int):
        state = self.think(prompt_ids, prompt_mask, R)
        B = prompt_ids.shape[0]
        if force_sep:
            sep = torch.tensor([self.fmt.sep] * B, dtype=torch.long)
            self.feed(state, sep)
        out = []
        done = torch.zeros(B, dtype=torch.bool, device=self.device)
        for i in range(max_new_tokens):
            nxt = self.lm.lm_head(state["last"]).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.fmt.eot), nxt)
            out.append(nxt)
            done |= nxt == self.fmt.eot
            if bool(done.all()) or i == max_new_tokens - 1:
                break
            self.feed(state, nxt[:, None])
        return torch.stack(out, 1).cpu()

    @torch.no_grad()
    def answer_batch(self, examples: list[dict], R: int | None = None, stage: int | None = None,
                     max_new_tokens: int | None = None) -> list[dict]:
        """Greedy answers. Free generation (model writes trace + 'Answer:') when the plan keeps explicit trace
        lines (cot / early curriculum stage); otherwise the separator is forced right after the thoughts."""
        R = self.R if R is None else R
        f = self.fmt
        plans = [plan_example(ex, self.mode, R, stage) for ex in examples]
        r_eff = plans[0][0]
        free = any(len(p[1]) > 0 for p in plans) or self.mode == "cot"
        if max_new_tokens is None:
            max_new_tokens = (8 * max(len(ex["trace_steps"]) for ex in examples) + 10) if free else 4
        pid, pmask = _left_pad([f.prompt(ex["prompt"]) for ex in examples], f.pad)
        gen = self.generate(pid, pmask, r_eff, force_sep=not free, max_new_tokens=max_new_tokens)
        res = []
        for ex, g in zip(examples, gen.tolist()):
            if f.eot in g:
                g = g[:g.index(f.eot)]
            text = f.tok.decode(g)
            pred = parse_answer(text, forced_sep=not free)
            res.append(dict(pred=pred, answer=ex["answer"], correct=pred == ex["answer"], text=text))
        return res

    @torch.no_grad()
    def greedy_answer(self, prompt: str | list[str], R: int | None = None, max_new_tokens: int = 4):
        """Answer raw program prompt(s) with R thought steps (R may differ from training R) and a forced
        separator. Returns int | None (or a list for a list input)."""
        single = isinstance(prompt, str)
        prompts = [prompt] if single else list(prompt)
        exs = [dict(prompt=p, answer=None, trace_steps=[]) for p in prompts]
        mode, self.mode = self.mode, ("none" if self.mode == "cot" else self.mode)
        try:
            out = [r["pred"] for r in self.answer_batch(exs, R=0 if self.mode == "none" else R,
                                                         max_new_tokens=max_new_tokens)]
        finally:
            self.mode = mode
        return out[0] if single else out


def default_read_layer(num_layers: int) -> int:
    return max(1, min(num_layers, round(2 * num_layers / 3)))

