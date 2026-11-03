"""CPU tests for the latent-reasoning experiment.  Run:  cd multimodal && python -m pytest bitnet_vl/tests/test_latent.py -q"""

import copy
import random

import pytest
import torch
import torch.nn as nn

from bitnet_vl.bitlinear import BitLinear, replace_linears
from bitnet_vl.latent import train_latent
from bitnet_vl.latent.compressor import LatentCompressor, PauseTokens, embedding_rms
from bitnet_vl.latent.lora import LoRALinear, apply_lora, count_params, freeze_base, load_trainable, save_trainable
from bitnet_vl.latent.reasoner import IGNORE, LatentReasoner, enable_checkpointing, plan_example
from bitnet_vl.latent.tasks import (TaskSampler, exec_program, make_example, make_test_set, parse_hops,
                                    trace_values)
from bitnet_vl.make_tiny_ckpt import tiny_llm_config
from bitnet_vl.modeling import LLM_LINEAR_RE, _strip_quant_config

torch.set_num_threads(4)
D, K = 128, 2


@pytest.fixture(scope="session")
def tok():
    return train_latent._tiny_tokenizer()


def _tiny_lm(tok, seed=0, layers=3):
    from transformers import BitNetForCausalLM

    torch.manual_seed(seed)
    cfg = tiny_llm_config(vocab_size=len(tok), hidden=D, intermediate=256, layers=layers, heads=4, kv_heads=2)
    lm = BitNetForCausalLM(_strip_quant_config(cfg))
    replace_linears(lm, lambda n, m: bool(LLM_LINEAR_RE.search(n)))
    with torch.no_grad():  # non-trivial norms
        for n, p in lm.named_parameters():
            if p.ndim == 1:
                p.add_(0.1 * torch.randn_like(p))
    return lm.eval()


def _reasoner(tok, mode="latent", R=3, lora=True, seed=0, prequantize=True, bptt=0):
    lm = _tiny_lm(tok, seed)
    if lora:
        apply_lora(lm, r=4, alpha=8, prequantize=prequantize)
        freeze_base(lm)
        with torch.no_grad():  # LoRA B is zero-init; make the LoRA path active for the tests
            for n, p in lm.named_parameters():
                if n.endswith("lora_B"):
                    p.normal_(0, 0.02)
    rms = embedding_rms(lm.get_input_embeddings().weight)
    comp = LatentCompressor(D, K, n_layers=1, n_heads=4, target_rms=rms) if mode == "latent" else None
    pause = PauseTokens(D, K, target_rms=rms) if mode == "pause" else None
    return LatentReasoner(lm, comp, read_layer=2, K=K, mode=mode, tokenizer=tok, pause=pause, R=R, bptt=bptt).eval()


def _set_act_quant(r, on: bool):
    """Per-token int8 rounding is discontinuous: a 1e-7 difference between two mathematically equal computations
    (different matmul shapes) can flip one rounding and move logits by ~1e-3. Exact-equivalence tests therefore run
    with activation quantization off, plus a loose check with it on."""
    for m in r.modules():
        if isinstance(m, BitLinear):
            m.quantize_activations = on


def _examples(n=3, seed=0, hops=(2, 3, 4), distractors=2):
    rng = random.Random(seed)
    return [make_example(rng, hops[i % len(hops)], distractors) for i in range(n)]


# ------------------------------------------------------------------------------------------------------- tasks
def test_tasks_match_exec_and_deterministic():
    rng = random.Random(123)
    for _ in range(500):
        h, d = rng.randint(1, 8), rng.randint(0, 6)
        ex = make_example(rng, h, d)
        assert exec_program(ex["prompt"]) == ex["answer"]
        assert 0 <= ex["answer"] <= 99
        assert len(ex["trace_steps"]) == h == ex["hops"]
        assert trace_values(ex["prompt"], ex["trace_steps"])
        assert ex["trace_steps"][-1] == f"{ex['query']} = {ex['answer']}"
        assert ex["prompt"].strip().endswith(f"print({ex['query']})")
        assert len(ex["program_lines"]) == h + 1 + d + 1
    a = TaskSampler([2, 3], 2, seed=5).batch(10)
    b = TaskSampler([2, 3], 2, seed=5).batch(10)
    assert a == b
    assert TaskSampler([2, 3], 2, seed=6).batch(10) != a
    test = make_test_set([2, 8], 20, 2)
    assert set(test) == {2, 8} and all(e["hops"] == 8 for e in test[8])
    assert test == make_test_set([2, 8], 20, 2)
    train_prompts = {e["prompt"] for e in TaskSampler([2], 2, seed=0).batch(200)}
    assert len(train_prompts & {e["prompt"] for e in test[2]}) < 3  # disjoint seed spaces
    assert parse_hops("2-4,7") == [2, 3, 4, 7]


# ------------------------------------------------------------------------------------------------------- compressor
def test_compressor_output_rms_matches_embeddings(tok):
    lm = _tiny_lm(tok)
    rms = embedding_rms(lm.get_input_embeddings().weight)
    comp = LatentCompressor(D, 3, n_layers=2, n_heads=4, target_rms=rms)
    mem = torch.randn(2, 11, D) * 50  # hidden states with a very different scale
    mask = torch.ones(2, 11, dtype=torch.long)
    mask[0, :4] = 0
    out = comp(mem, mask, step=1)
    assert out.shape == (2, 3, D)
    tok_rms = out.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(tok_rms, torch.full_like(tok_rms, rms), rtol=1e-4, atol=1e-5)
    # masked memory entries are ignored
    mem2 = mem.clone()
    mem2[0, :4] = 1e3 * torch.randn(4, D)
    torch.testing.assert_close(comp(mem2, mask, step=1)[0], out[0])
    p = PauseTokens(D, 3, target_rms=rms)(2, 4)
    assert p.shape == (2, 12, D)
    torch.testing.assert_close(p.pow(2).mean(-1).sqrt(), torch.full((2, 12), rms), rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------------------------------------------- KV cache
def _full_forward(r, prompt_ids, prompt_mask, thoughts, target_ids, target_mask):
    """No-cache reference: one forward over [prompt embeds ; thought embeds ; target embeds]."""
    emb = r.lm.get_input_embeddings()
    parts = [emb(prompt_ids)] + ([thoughts] if thoughts is not None else []) + [emb(target_ids)]
    x = torch.cat(parts, 1)
    n_th = 0 if thoughts is None else thoughts.shape[1]
    B = prompt_ids.shape[0]
    mask = torch.cat([prompt_mask, prompt_mask.new_ones(B, n_th), target_mask], 1)
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    out = r.lm(inputs_embeds=x, attention_mask=mask, position_ids=pos, use_cache=False, output_hidden_states=True)
    return out, prompt_ids.shape[1], n_th


@pytest.mark.parametrize("act_quant", [False, True])
@pytest.mark.parametrize("mode", ["latent", "pause"])
def test_kv_cache_loop_matches_full_forward(tok, mode, act_quant):
    torch.manual_seed(0)
    r = _reasoner(tok, mode, R=3)
    _set_act_quant(r, act_quant)
    tol = dict(atol=1e-4, rtol=1e-4) if not act_quant else dict(atol=5e-2, rtol=0)
    batch = r.build_batch(_examples(3))
    with torch.no_grad():
        state = r.think(batch["prompt_ids"], batch["prompt_mask"], 3)
        prev = state["last"]
        h = r.feed(state, batch["target_ids"], batch["target_mask"])
        logits_cached = r.lm.lm_head(torch.cat([prev[:, None], h[:, :-1]], 1))
        thoughts = r.last_thought_inputs
        assert thoughts.shape[1] == 3 * K
        out, P, n_th = _full_forward(r, batch["prompt_ids"], batch["prompt_mask"], thoughts, batch["target_ids"],
                                     batch["target_mask"])
        ref = out.logits[:, P + n_th - 1: P + n_th - 1 + batch["target_ids"].shape[1]]
        valid = batch["target_mask"].bool()
        torch.testing.assert_close(logits_cached[valid], ref[valid], **tol)
        if mode == "latent":
            # the soft tokens are what the compressor makes of the (no-cache) layer-i states of everything before
            hs = out.hidden_states[r.read_layer]
            mmask = torch.cat([batch["prompt_mask"], batch["prompt_mask"].new_ones(3, n_th)], 1)
            for t in range(3):
                L = P + t * K
                soft = r.compressor(hs[:, :L], mmask[:, :L], step=t)
                torch.testing.assert_close(soft, thoughts[:, t * K:(t + 1) * K], **tol)


def test_hidden_capture_matches_output_hidden_states(tok):
    r = _reasoner(tok, "latent", R=1)
    b = r.build_batch(_examples(2))
    with torch.no_grad():
        emb = r.lm.get_input_embeddings()(b["prompt_ids"])
        pos = (b["prompt_mask"].cumsum(1) - 1).clamp(min=0)
        from bitnet_vl.latent.reasoner import LatentCache

        _, hi = r.lm_step(emb, b["prompt_mask"], pos, LatentCache(r.num_layers), capture=True)
        ref = r.lm(inputs_embeds=emb, attention_mask=b["prompt_mask"], position_ids=pos,
                   output_hidden_states=True).hidden_states[r.read_layer]
    torch.testing.assert_close(hi, ref)


@pytest.mark.parametrize("act_quant", [False, True])
@pytest.mark.parametrize("mode", ["latent", "pause", "none"])
def test_left_padded_batch_matches_unbatched(tok, mode, act_quant):
    torch.manual_seed(0)
    r = _reasoner(tok, mode, R=2)
    _set_act_quant(r, act_quant)
    tol = dict(atol=1e-4, rtol=1e-4) if not act_quant else dict(atol=5e-2, rtol=0)
    exs = _examples(3, hops=(2, 5, 3), distractors=3)

    def answer_logits(examples):
        b = r.build_batch(examples)
        with torch.no_grad():
            st = r.think(b["prompt_ids"], b["prompt_mask"], b["R"])
            prev = st["last"]
            h = r.feed(st, b["target_ids"], b["target_mask"])
            lg = r.lm.lm_head(torch.cat([prev[:, None], h[:, :-1]], 1))
        return [lg[i, b["labels"][i] != IGNORE] for i in range(len(examples))]

    batched = answer_logits(exs)
    assert len({len(r.fmt.prompt(e["prompt"])) for e in exs}) > 1  # padding actually happens
    for i, e in enumerate(exs):
        single = answer_logits([e])[0]
        torch.testing.assert_close(batched[i], single, **tol)
    if not act_quant:  # greedy decoding agrees too (with act quant a near-tie argmax of a random model could flip)
        assert [x["text"] for x in r.answer_batch(exs)] == [r.answer_batch([e])[0]["text"] for e in exs]


# ------------------------------------------------------------------------------------------------------- grads
def test_gradients_reach_compressor_and_lora_base_frozen(tok):
    r = _reasoner(tok, "latent", R=2).train()
    out = r(r.build_batch(_examples(2)))
    out["loss"].backward()
    for n, p in r.compressor.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, n
    lora = [(n, p) for n, p in r.lm.named_parameters() if "lora_" in n]
    assert lora and all(p.grad is not None and p.grad.abs().sum() > 0 for n, p in lora)
    base = [(n, p) for n, p in r.lm.named_parameters() if "lora_" not in n]
    assert all(not p.requires_grad and p.grad is None for n, p in base)
    assert count_params(r.lm) == sum(p.numel() for _, p in lora)


def test_checkpointing_gives_same_grads(tok):
    grads = []
    for ckpt in (False, True):
        r = _reasoner(tok, "latent", R=2, seed=1).train()
        if ckpt:
            enable_checkpointing(r.lm, True)
        torch.manual_seed(0)
        out = r(r.build_batch(_examples(2, seed=3)))
        out["loss"].backward()
        grads.append((out["loss"].item(), {n: p.grad.clone() for n, p in r.named_parameters() if p.grad is not None}))
    assert abs(grads[0][0] - grads[1][0]) < 1e-5
    assert grads[0][1].keys() == grads[1][1].keys()
    for n in grads[0][1]:
        torch.testing.assert_close(grads[0][1][n], grads[1][1][n], atol=1e-5, rtol=1e-4, msg=n)


def test_bptt_truncation_cuts_old_thought_paths(tok):
    # with bptt=1, thought 0's memory entry is detached when making thought 2: grads differ from full BPTT but exist
    g = []
    for bptt in (0, 1):
        r = _reasoner(tok, "latent", R=3, seed=2, bptt=bptt).train()
        r(r.build_batch(_examples(2, seed=4)))["loss"].backward()
        g.append(r.compressor.queries.grad.clone())
    assert g[1].abs().sum() > 0 and not torch.allclose(g[0], g[1])


def test_pause_has_no_dependence_on_hidden_states(tok):
    r = _reasoner(tok, "pause", R=2)
    with torch.no_grad():
        b1 = r.build_batch(_examples(2, seed=0))
        r.think(b1["prompt_ids"], b1["prompt_mask"], 2)
        t1 = r.last_thought_inputs.clone()
        b2 = r.build_batch(_examples(2, seed=9))
        r.think(b2["prompt_ids"], b2["prompt_mask"], 2)
        t2 = r.last_thought_inputs.clone()
    torch.testing.assert_close(t1, t2)
    assert t1.shape == (2, 2 * K, D)
    torch.testing.assert_close(t1[:, :K], t1[:, K:])  # same pause block every step
    r.train()
    r(r.build_batch(_examples(2)))["loss"].backward()
    assert r.pause.emb.grad is not None and r.pause.emb.grad.abs().sum() > 0


# ------------------------------------------------------------------------------------------------------- curriculum
def test_curriculum_stage_construction(tok):
    ex = dict(prompt="a = 1\nb = a + 1\nc = b * 2\nprint(c)\n", answer=4, trace_steps=["b = 2", "c = 4"])
    assert plan_example(ex, "none", 3) == (0, [])
    assert plan_example(ex, "cot", 3) == (0, ["b = 2", "c = 4"])
    assert plan_example(ex, "latent", 3) == (3, [])
    assert plan_example(ex, "latent", 3, stage=0) == (0, ["b = 2", "c = 4"])
    assert plan_example(ex, "latent", 3, stage=1) == (1, ["c = 4"])
    assert plan_example(ex, "latent", 3, stage=2) == (2, [])
    assert plan_example(ex, "latent", 3, stage=3) == (3, [])
    assert plan_example(ex, "pause", 3, stage=7) == (3, [])
    r = _reasoner(tok, "latent", R=3)
    f = r.fmt
    b = r.build_batch([ex], stage=1)
    assert b["R"] == 1
    tgt = b["target_ids"][0].tolist()
    lab = b["labels"][0].tolist()
    exp = f.trace(["c = 4"]) + f.sep + f.answer(4)
    assert tgt == exp and lab == exp  # remaining trace + separator + answer all in the loss
    assert f.tok.decode(tgt) == "c = 4\nAnswer: 4<|eot_id|>"
    b = r.build_batch([ex], stage=3)
    assert b["R"] == 3
    lab = b["labels"][0].tolist()
    assert lab[:len(f.sep)] == [IGNORE] * len(f.sep) and lab[len(f.sep):] == f.answer(4)
    assert f.tok.decode(b["prompt_ids"][0]) == "<|begin_of_text|>" + ex["prompt"]


# ------------------------------------------------------------------------------------------------------- lora
def test_lora_prequantize_matches_bitlinear_and_roundtrip(tmp_path):
    torch.manual_seed(0)
    bl = BitLinear(128, 64, bias=False)
    x = torch.randn(3, 5, 128)
    ref = bl(x)
    a, b = LoRALinear(copy.deepcopy(bl), r=4), LoRALinear(copy.deepcopy(bl), r=4)
    b.prequantize_base_()
    torch.testing.assert_close(a(x), ref)
    torch.testing.assert_close(b(x), ref, atol=1e-5, rtol=1e-5)
    m = nn.Sequential()
    m.add_module("q_proj", LoRALinear(nn.Linear(8, 8), r=2))
    nn.init.normal_(m.q_proj.lora_B)
    comp = LatentCompressor(16, 2, n_heads=2)
    save_trainable(tmp_path / "t.pt", m, {"compressor": comp})
    m2 = nn.Sequential()
    m2.add_module("q_proj", LoRALinear(nn.Linear(8, 8), r=2))
    comp2 = LatentCompressor(16, 2, n_heads=2)
    load_trainable(tmp_path / "t.pt", m2, {"compressor": comp2})
    torch.testing.assert_close(m2.q_proj.lora_B, m.q_proj.lora_B)
    torch.testing.assert_close(comp2.queries, comp.queries)


# ------------------------------------------------------------------------------------------------------- train steps
@pytest.mark.parametrize("mode,extra", [("none", {}), ("cot", {}), ("pause", {}),
                                        ("latent", {"curriculum": True, "steps_per_stage": 1}),
                                        ("latent", {"train_base": True, "lora_rank": 0, "grad_checkpointing": True,
                                                    "bptt": 1})])
def test_one_train_step_per_mode(tmp_path, mode, extra):
    cfg = dict(model="tiny", mode=mode, R=2, K=2, hops_train="2-3", hops_test="2,4", distractors=1, steps=3,
               batch_size=2, n_test=2, eval_R_sweep="0,1", tiny_hidden=64, tiny_layers=2, tiny_intermediate=128,
               tiny_heads=2, tiny_kv_heads=1, lora_rank=4, warmup_steps=1, log_every=1, out=str(tmp_path),
               eval_every=2, eval_quick_n=2, save_weights=True)
    cfg.update(extra)
    res = train_latent.run(cfg)
    res.pop("_reasoner")
    assert len(res["loss_curve"]) == 3 and all(torch.isfinite(torch.tensor(x["loss"])) for x in res["loss_curve"])
    exp_R = {"0"} if mode in ("none", "cot") else {"0", "1", "2"}
    assert set(res["final_acc"]) == exp_R and set(res["final_acc"]["0"]) == {"2", "4"}
    assert (tmp_path / "results.json").exists() and (tmp_path / "trainable.pt").exists()
    if extra.get("curriculum"):
        assert [x["stage"] for x in res["loss_curve"]] == [0, 1, 2]


@pytest.mark.parametrize("extra", [{"curriculum": True, "steps_per_stage": 2}, {"lora_rank": 0, "train_base": True}])
def test_resume_after_interrupt_matches_uninterrupted_run(tmp_path, monkeypatch, extra):
    cfg = dict(model="tiny", mode="latent", R=2, K=2, hops_train="2-3", hops_test="2,3", distractors=1, steps=6,
               batch_size=2, n_test=4, eval_R_sweep="0,2", tiny_hidden=64, tiny_layers=2, tiny_intermediate=128,
               tiny_heads=2, tiny_kv_heads=1, lora_rank=4, warmup_steps=1, log_every=1, ckpt_every=3)
    cfg.update(extra)
    ref = train_latent.run(dict(cfg, out=str(tmp_path / "ref")))

    class Interrupted(Exception):
        pass

    orig = train_latent.save_checkpoint

    def save_then_die(path, c, reasoner, opt, scaler, sampler, step_done, state):
        orig(path, c, reasoner, opt, scaler, sampler, step_done, state)
        raise Interrupted

    monkeypatch.setattr(train_latent, "save_checkpoint", save_then_die)
    with pytest.raises(Interrupted):
        train_latent.run(dict(cfg, out=str(tmp_path / "res")))
    monkeypatch.setattr(train_latent, "save_checkpoint", orig)
    assert torch.load(str(tmp_path / "res" / "ckpt.pt"), weights_only=False)["step"] == 3
    res = train_latent.run(dict(cfg, out=str(tmp_path / "res")))

    assert [x["step"] for x in res["loss_curve"]] == [1, 2, 3, 4, 5, 6]
    for a, b in zip(ref["loss_curve"], res["loss_curve"]):
        assert a["stage"] == b["stage"] and a["loss"] == pytest.approx(b["loss"], rel=1e-5, abs=1e-6)
    assert res["final_acc"] == ref["final_acc"]
    sd_ref = {n: p for n, p in ref["_reasoner"].named_parameters() if p.requires_grad}
    for n, p in res["_reasoner"].named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p, sd_ref[n])
    assert res["throughput"]["steps"] == 6

    with pytest.raises(ValueError, match="different config"):
        train_latent.run(dict(cfg, out=str(tmp_path / "res"), lr_lora=1e-2))


def test_run_suite_skips_finished_modes(tmp_path, monkeypatch):
    from bitnet_vl.latent import run_suite as suite

    tiny = dict(model="tiny", R=2, K=2, hops_train="2-3", hops_test="2", distractors=1, steps=2, batch_size=2, n_test=2,
                eval_R_sweep="0", tiny_hidden=64, tiny_layers=2, tiny_intermediate=128, tiny_heads=2, tiny_kv_heads=1,
                lora_rank=4, warmup_steps=1, log_every=1, ckpt_every=1)
    cfg_for = lambda m: dict(tiny, mode=m, out=str(tmp_path / m))  # noqa: E731
    first = suite.run_suite(["none", "pause"], tmp_path, cfg_for=cfg_for)
    calls = []
    monkeypatch.setattr(suite, "run", lambda cfg: calls.append(cfg["mode"]) or {})
    again = suite.run_suite(["none", "pause"], tmp_path, cfg_for=cfg_for)
    assert calls == [] and again["pause"]["final_acc"] == first["pause"]["final_acc"]
    assert (tmp_path / "all.json").exists() and (tmp_path / "pause" / "ckpt.pt").exists()
