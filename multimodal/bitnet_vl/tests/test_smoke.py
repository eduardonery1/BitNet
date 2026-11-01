"""CPU smoke tests for BitNet-VL.  Run:  cd multimodal && python -m pytest bitnet_vl/tests -q"""

import copy
import json

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open

from bitnet_vl.bitlinear import BitLinear, activation_quant, replace_linears, ternary_weight, weight_quant
from bitnet_vl.data import IGNORE_INDEX, Collator, LlavaDataset, build_prompt, tokenize_conversation
from bitnet_vl.make_tiny_ckpt import build_tiny_model
from bitnet_vl.modeling import BitNetVLForConditionalGeneration, bitlinear_names, pixel_shuffle, \
    pixel_shuffle_reference
from bitnet_vl.tokenizer_utils import IMAGE_TOKEN, add_image_token, build_llama3_tokenizer
from bitnet_vl.train import STAGE_DEFAULTS, build_optimizer, build_teacher, compute_loss, set_schedule

torch.set_num_threads(4)


@pytest.fixture(scope="session")
def tokenizer():
    return build_llama3_tokenizer()


@pytest.fixture()
def model(tokenizer):
    # vision hidden 128 / 2 layers / image 64 / patch 16 -> 16 patches -> 4 tokens; LLM hidden 128, 2 layers
    return build_tiny_model(seed=0, tokenizer=copy.deepcopy(tokenizer), llm_hidden=128, vision_hidden=128)


def _records():
    return [
        {"image": "a.png", "conversations": [{"from": "human", "value": "<image>\nWhat color is the bus?"},
                                             {"from": "gpt", "value": "Red."}]},
        {"image": "b.png", "conversations": [{"from": "human", "value": "How many dogs?\n<image>"},
                                             {"from": "gpt", "value": "Two."},
                                             {"from": "human", "value": "Are they big?"},
                                             {"from": "gpt", "value": "No, they are small puppies."}]},
    ]


@pytest.fixture()
def batch(tmp_path, model):
    from PIL import Image

    for i, n in enumerate(["a.png", "b.png"]):
        Image.fromarray((torch.rand(80, 70, 3) * 255).byte().numpy()).save(tmp_path / n)
    ds = LlavaDataset(_records(), model.tokenizer, tmp_path, image_size=model.config.image_size,
                      max_length=256, num_image_tokens=model.config.num_image_tokens)
    return Collator(model.tokenizer.pad_token_id)([ds[0], ds[1]])


# ---------------------------------------------------------------------------------------------------- bitlinear
def test_ternary_values_and_converter_semantics():
    w = torch.randn(64, 128)
    q, s = ternary_weight(w)
    assert q.dtype == torch.int8 and set(q.unique().tolist()) <= {-1, 0, 1}
    assert abs(s - w.abs().mean().item()) < 1e-6
    # identical to utils/convert-hf-to-gguf-bitnet.py::BitnetModel.weight_quant
    inv = 1 / w.abs().mean().clamp(min=1e-5)
    ref = (w * inv).round().clamp(-1, 1) / inv
    torch.testing.assert_close(q.float() * s, ref, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(weight_quant(w), ref)
    a = activation_quant(torch.randn(3, 5, 128))
    scale = 127 / a.abs().amax(-1, keepdim=True)
    lv = a * scale
    assert torch.allclose(lv, lv.round(), atol=1e-3)


def test_bitlinear_ste_grads_flow():
    lin = nn.Linear(128, 32, bias=True)
    bl = BitLinear.from_linear(lin)
    assert bl.weight is lin.weight
    x = torch.randn(4, 128, requires_grad=True)
    y = bl(x)
    y.pow(2).sum().backward()
    assert bl.weight.grad is not None and bl.weight.grad.abs().sum() > 0
    assert bl.bias.grad is not None and x.grad is not None and x.grad.abs().sum() > 0
    # forward equals the quantized matmul
    ref = activation_quant(x.detach()) @ weight_quant(lin.weight.detach()).T + lin.bias.detach()
    torch.testing.assert_close(y.detach(), ref)


def test_replace_linears_preserves_state_dict():
    m = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Sequential(nn.Linear(8, 4, bias=False)))
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    names = replace_linears(m, lambda n, l: n != "0")
    assert names == ["2.0"] and isinstance(m[2][0], BitLinear) and not isinstance(m[0], BitLinear)
    assert replace_linears(m) == ["0"]  # idempotent on already-swapped layers
    assert set(m.state_dict()) == set(sd)
    for k in sd:
        assert torch.equal(sd[k], m.state_dict()[k])


# ---------------------------------------------------------------------------------------------------- pixel shuffle
@pytest.mark.parametrize("hw,d", [(4, 3), (24, 8)])
def test_pixel_shuffle_matches_reference(hw, d):
    x = torch.randn(2, hw * hw, d)
    y = pixel_shuffle(x, 2)
    assert y.shape == (2, hw * hw // 4, 4 * d)
    torch.testing.assert_close(y, pixel_shuffle_reference(x, 2))
    # explicit spot check of the documented formula: y[i*(W/2)+j, (dy*2+dx)*D + c] = x[(2i+dy)*W + 2j+dx, c]
    i, j, dy, dx, c = 1, 0, 1, 1, 2
    assert y[0, i * (hw // 2) + j, (dy * 2 + dx) * d + c] == x[0, (2 * i + dy) * hw + 2 * j + dx, c]


# ---------------------------------------------------------------------------------------------------- data
def test_label_masking(tokenizer):
    tok = copy.deepcopy(tokenizer)
    add_image_token(tok)
    conv = _records()[1]["conversations"]
    ids, labels = tokenize_conversation(tok, conv, has_image=True)
    text = tok.decode(ids)
    assert text == ("<|begin_of_text|>User: <image>\nHow many dogs?<|eot_id|>Assistant: Two.<|eot_id|>"
                    "User: Are they big?<|eot_id|>Assistant: No, they are small puppies.<|eot_id|>")
    supervised = tok.decode(ids[labels != IGNORE_INDEX])
    assert supervised == "Two.<|eot_id|>No, they are small puppies.<|eot_id|>"
    assert (labels[labels != IGNORE_INDEX] == ids[labels != IGNORE_INDEX]).all()
    assert int((ids == tok.convert_tokens_to_ids(IMAGE_TOKEN)).sum()) == 1
    # the prompt part tokenizes exactly like the inference prompt
    p = tok(build_prompt("How many dogs?"), add_special_tokens=False).input_ids
    assert ids[: len(p)].tolist() == p


def test_merge_expands_image_token(model, batch):
    n = model.config.num_image_tokens
    feats = torch.randn(2, n, model.config.llm_hidden)
    e, m, l = model.merge_inputs(batch["input_ids"], feats, batch["attention_mask"], batch["labels"])
    assert e.shape[1] == batch["input_ids"].shape[1] + n - 1
    for b in range(2):
        p = int((batch["input_ids"][b] == model.config.image_token_id).nonzero()[0])
        torch.testing.assert_close(e[b, p:p + n], feats[b])
        assert (l[b, p:p + n] == IGNORE_INDEX).all()
        assert m[b].sum() == batch["attention_mask"][b].sum() + n - 1
        # supervised tokens are preserved in order
        assert torch.equal(l[b][l[b] != IGNORE_INDEX], batch["labels"][b][batch["labels"][b] != IGNORE_INDEX])


# ---------------------------------------------------------------------------------------------------- model
def test_forward_loss_finite_and_quantized(model, batch):
    assert model.config.num_image_tokens == 4
    names = bitlinear_names(model.language_model)
    assert len(names) == 2 * 7 and not any("lm_head" in n for n in names)
    assert bitlinear_names(model.vision_tower) == []
    out = model(**batch)
    assert torch.isfinite(out.loss)
    assert out.logits.shape[1] == batch["input_ids"].shape[1] + model.config.num_image_tokens - 1
    assert len(model.ternarize_vision_()) == 2 * 6


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_one_step_per_stage(model, batch, stage):
    class A:
        teacher_from = None

    teacher = build_teacher(model, A()) if stage == 3 else None
    lrs = STAGE_DEFAULTS[stage]["lr"]
    opt = build_optimizer(model, stage, lrs, STAGE_DEFAULTS[stage]["wd"][0])
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    prefix = {1: ("projector.",), 2: ("projector.", "language_model."), 3: ("vision_tower.",)}[stage]
    assert trainable and all(n.startswith(prefix) for n in trainable)
    if stage == 3:
        assert len(bitlinear_names(model.vision_tower)) == 12
        assert not any(isinstance(m, BitLinear) for m in teacher.modules())
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    set_schedule(opt, 0, 10, 1, *STAGE_DEFAULTS[stage]["wd"])
    model.train()
    loss, logs = compute_loss(model, batch, stage, teacher, 0.1)
    assert torch.isfinite(loss)
    if stage == 3:
        assert "distill" in logs and torch.isfinite(logs["distill"]) and logs["distill"] > 0
    loss.backward()
    opt.step()
    changed = {n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())}
    assert changed and changed <= trainable


def test_save_load_roundtrip(model, batch, tmp_path):
    model.ternarize_vision_()
    model.eval()
    with torch.no_grad():
        ref = model(**batch).logits
    model.save_pretrained(tmp_path / "ckpt")
    d = tmp_path / "ckpt"
    cfg = json.loads((d / "config.json").read_text())
    for k in ("model_type", "vision_model_name", "llm_model_name", "pixel_shuffle", "vision_ternary", "image_size",
              "num_image_tokens", "llm_hidden"):
        assert k in cfg
    assert cfg["model_type"] == "bitnet_vl" and cfg["vision_ternary"] is True and cfg["pixel_shuffle"] == 2
    with safe_open(d / "projector.safetensors", "pt") as f:
        assert sorted(f.keys()) == ["fc1.bias", "fc1.weight", "fc2.bias", "fc2.weight", "norm.bias", "norm.weight"]
    with safe_open(d / "vision_tower.safetensors", "pt") as f:
        keys = list(f.keys())
    assert all(k.startswith("vision_model.") for k in keys)
    assert "vision_model.encoder.layers.1.self_attn.q_proj.weight" in keys
    assert "vision_model.post_layernorm.weight" in keys
    lm = json.loads((d / "language_model" / "config.json").read_text())
    assert lm["architectures"] == ["BitNetForCausalLM"] and lm["tie_word_embeddings"] and "rope_theta" in lm
    assert (d / "language_model" / "tokenizer.json").exists()
    with safe_open(d / "language_model" / "model.safetensors", "pt") as f:
        assert "lm_head.weight" not in f.keys()

    re = BitNetVLForConditionalGeneration.from_pretrained(d).eval()
    assert re.config.image_token_id == model.config.image_token_id
    assert len(bitlinear_names(re.vision_tower)) == 12 and len(bitlinear_names(re.language_model)) == 14
    with torch.no_grad():
        got = re(**batch).logits
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_greedy_answer(model, tmp_path):
    from PIL import Image

    img = Image.fromarray((torch.rand(50, 60, 3) * 255).byte().numpy())
    ans = model.answer(img, "What is this?", max_new_tokens=4)
    assert isinstance(ans, str)
    # cached greedy decoding == uncached argmax
    pv = torch.randn(1, 3, 64, 64)
    ids = model.tokenizer(build_prompt("hi"), add_special_tokens=False, return_tensors="pt").input_ids
    gen = model.generate(ids, pv, max_new_tokens=3, eos_token_ids=set())
    seq = ids
    for t in range(3):
        with torch.no_grad():
            nxt = model(input_ids=seq, pixel_values=pv).logits[:, -1].argmax(-1)
        assert int(nxt) == int(gen[0, t])
        seq = torch.cat([seq, nxt[:, None]], 1)
