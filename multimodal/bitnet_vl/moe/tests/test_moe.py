"""CPU tests for BitNet-MoE.  Run:  cd multimodal && python -m pytest bitnet_vl/moe/tests -q"""

import copy
import json
import math

import numpy as np
import pytest
import torch
from safetensors import safe_open
from transformers import BitNetConfig, BitNetForCausalLM
from transformers.utils import logging as hf_logging

from bitnet_vl.bitlinear import BitLinear, replace_linears
from bitnet_vl.modeling import LLM_LINEAR_RE
from bitnet_vl.moe.losses import expert_fractions, load_balancing_loss, router_z_loss
from bitnet_vl.moe.modeling_moe import BitNetMoEForCausalLM, contract_tensor_names, route
from bitnet_vl.moe.train_moe import pack
from bitnet_vl.moe.upcycle import upcycle

torch.set_num_threads(4)
hf_logging.disable_progress_bar()


def _dense(qat: bool, seed: int = 0, tie: bool = True, layers: int = 2):
    torch.manual_seed(seed)
    cfg = BitNetConfig(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=layers,
                       num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=tie,
                       max_position_embeddings=256)
    m = BitNetForCausalLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim == 1 and "norm" in n:
                p.add_(0.2 * torch.randn_like(p))
    if qat:
        replace_linears(m, lambda n, _: bool(LLM_LINEAR_RE.search(n)))
    return m.eval()


def _ids(b=2, t=24, seed=1):
    return torch.randint(0, 512, (b, t), generator=torch.Generator().manual_seed(seed))


@pytest.fixture()
def moe():
    return upcycle(_dense(True), num_experts=4, top_k=2, expert_noise=0.3, router_std=0.5, seed=3).eval()


# ---------------------------------------------------------------------------------------------- upcycling
@pytest.mark.parametrize("qat", [True, False])
@pytest.mark.parametrize("E,k", [(4, 2), (8, 2), (4, 1), (3, 3)])
def test_upcycle_equals_dense_for_any_router(qat, E, k):
    dense = _dense(qat)
    moe = upcycle(dense, num_experts=E, top_k=k, expert_noise=0.0, qat=qat).eval()
    assert all(isinstance(m.experts[0].gate_proj, BitLinear) == qat for m in moe.moe_layers)
    ids = _ids()
    ref = dense(ids).logits
    for seed in range(3):
        torch.manual_seed(seed)
        with torch.no_grad():
            for r in moe.router_parameters():
                r.normal_(0, 3.0)
        out = moe(ids)
        torch.testing.assert_close(out.logits, ref, atol=1e-5, rtol=0)
        assert out.expert_counts.sum(-1).eq(ids.numel() * k).all()


def test_upcycle_noise_breaks_symmetry():
    moe = upcycle(_dense(True), num_experts=4, top_k=2, expert_noise=0.1)
    w = [e.gate_proj.weight for e in moe.moe_layers[0].experts]
    assert not torch.equal(w[0], w[1])


# ---------------------------------------------------------------------------------------------- routing / dispatch
def test_routing_renormalization():
    logits = torch.randn(50, 8) * 3
    w, idx = route(logits, 2)
    torch.testing.assert_close(w.sum(-1), torch.ones(50))
    # equals softmax over the top-k logits, indices are the top-k
    top = torch.topk(logits, 2, -1)
    assert torch.equal(idx, top.indices)
    torch.testing.assert_close(w, torch.softmax(top.values, -1))
    assert (w[:, 0] >= w[:, 1]).all()


def test_dispatch_matches_naive_loop(moe):
    torch.manual_seed(0)
    for mlp in moe.moe_layers:
        x = torch.randn(3, 17, 128)
        torch.testing.assert_close(mlp(x), mlp.forward_reference(x), atol=1e-5, rtol=1e-5)
    # an expert that receives no tokens: positive inputs, strongly negative router row for expert 3
    mlp = moe.moe_layers[0]
    x = torch.randn(2, 9, 128).abs()
    with torch.no_grad():
        mlp.router.weight.copy_(0.1 * torch.randn(4, 128))
        mlp.router.weight[3] = -torch.ones(128)
    y = mlp(x)
    assert mlp.last_expert_counts[3] == 0
    torch.testing.assert_close(y, mlp.forward_reference(x), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------------------------- aux losses
def test_load_balancing_hand_cases():
    a = 10.0
    # top-1, E=2: everything to expert 0 -> f = [1, 0], P = [s, 1-s] -> 2 * s
    lg = torch.tensor([[a, 0.0], [a, 0.0]])
    s = torch.sigmoid(torch.tensor(a))
    torch.testing.assert_close(load_balancing_loss(lg, 1), 2 * s)
    # balanced top-1 -> f = P = [.5, .5] -> 1
    torch.testing.assert_close(load_balancing_loss(torch.tensor([[a, 0.0], [0.0, a]]), 1), torch.tensor(1.0))
    # top-2 of E=4, tokens pick {0,1} and {2,3} with equal logits -> f = P = 1/4 each -> 1
    lg = torch.tensor([[a, a, 0.0, 0.0], [0.0, 0.0, a, a]])
    torch.testing.assert_close(expert_fractions(lg, 2), torch.full((4,), 0.25))
    torch.testing.assert_close(load_balancing_loss(lg, 2), torch.tensor(1.0))
    # top-2 collapse onto {0,1}: f = [.5,.5,0,0], P ~ [.5,.5,0,0] -> 4 * .5 = 2
    lg = torch.tensor([[a, a, 0.0, 0.0]] * 3)
    p = torch.softmax(lg[0], -1)
    torch.testing.assert_close(load_balancing_loss(lg, 2), 4 * (0.5 * p[0] + 0.5 * p[1]))
    # mask removes the third token
    lg = torch.tensor([[a, 0.0], [0.0, a], [a, 0.0]])
    torch.testing.assert_close(load_balancing_loss(lg, 1, torch.tensor([True, True, False])), torch.tensor(1.0))


def test_z_loss_hand_cases():
    torch.testing.assert_close(router_z_loss(torch.zeros(5, 4)), torch.tensor(math.log(4) ** 2))
    lg = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    exp = (torch.logsumexp(lg, -1) ** 2).mean()
    torch.testing.assert_close(router_z_loss(lg), exp)
    torch.testing.assert_close(router_z_loss(lg, torch.tensor([True, False])), torch.logsumexp(lg[0], -1) ** 2)


def test_aux_loss_gradient_flows_to_router():
    lg = torch.randn(10, 4, requires_grad=True)
    (load_balancing_loss(lg, 2) + router_z_loss(lg)).backward()
    assert lg.grad.abs().sum() > 0


# ---------------------------------------------------------------------------------------------- training
def test_grads_reach_router_and_experts(moe):
    moe.train()
    ids = _ids(2, 32)
    out = moe(input_ids=ids, labels=ids)
    expected = out.lm_loss + 0.01 * out.aux_loss + 0.001 * out.z_loss
    torch.testing.assert_close(out.loss, expected)
    out.loss.backward()
    for li, mlp in enumerate(moe.moe_layers):
        assert mlp.router.weight.grad is not None and mlp.router.weight.grad.abs().sum() > 0
        assert mlp.ffn_sub_norm.weight.grad.abs().sum() > 0
        for e, ex in enumerate(mlp.experts):
            if out.expert_counts[li, e] > 0:
                for p in (ex.gate_proj, ex.up_proj, ex.down_proj):
                    assert p.weight.grad is not None and p.weight.grad.abs().sum() > 0, (li, e)


def test_gradient_checkpointing_same_grads(moe):
    ids = _ids(1, 16)
    m2 = copy.deepcopy(moe)
    moe.train()
    m2.train()
    m2.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for m in (moe, m2):
        m(input_ids=ids, labels=ids).loss.backward()
    for (n, p), (_, q) in zip(moe.named_parameters(), m2.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4, msg=n)


def test_pack_fixed_blocks():
    from bitnet_vl.tokenizer_utils import build_llama3_tokenizer

    tok = build_llama3_tokenizer()
    blocks = list(pack(["hello world " * 20, "", "def f(x):\n    return x\n" * 10], tok, 16))
    assert blocks and all(b.shape == (16,) for b in blocks)
    flat = torch.cat(blocks).tolist()
    assert flat[0] == tok.bos_token_id and tok.eos_token_id in flat


# ---------------------------------------------------------------------------------------------- io / contract
@pytest.mark.parametrize("tie", [True, False])
def test_save_load_roundtrip_and_contract_names(tmp_path, tie):
    moe = upcycle(_dense(True, tie=tie), num_experts=4, top_k=2, expert_noise=0.3, router_std=0.5).eval()
    ids = _ids()
    ref = moe(ids).logits
    moe.save_pretrained(tmp_path, dtype=torch.float32)
    keys = set()
    for f in tmp_path.glob("*.safetensors"):
        with safe_open(str(f), "pt") as sf:
            keys |= set(sf.keys())
    assert keys == contract_tensor_names(moe.config)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["model_type"] == "bitnet" and cfg["architectures"] == ["BitNetMoEForCausalLM"]
    assert cfg["num_experts"] == 4 and cfg["num_experts_per_tok"] == 2 and cfg["norm_topk_prob"] is True
    assert cfg["router_aux_loss_coef"] == 0.01 and cfg["router_z_loss_coef"] == 0.001
    assert "rope_theta" in cfg and "quantization_config" not in cfg
    with safe_open(str(tmp_path / "model.safetensors"), "pt") as sf:
        assert sf.get_tensor("model.layers.0.mlp.router.weight").shape == (4, 128)
        assert sf.get_tensor("model.layers.1.mlp.experts.3.down_proj.weight").shape == (128, 256)
        assert sf.get_tensor("model.layers.1.mlp.ffn_sub_norm.weight").shape == (256,)
    back = BitNetMoEForCausalLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(back(ids).logits, ref, atol=0, rtol=0)
    assert isinstance(back.model.layers[1].mlp.experts[2].down_proj, BitLinear)
    assert isinstance(back.model.layers[1].self_attn.o_proj, BitLinear)
    assert not isinstance(back.model.layers[1].mlp.router, BitLinear) and not isinstance(back.lm_head, BitLinear)


def test_tiny_ckpt_builds(tmp_path):
    from bitnet_vl.moe.make_tiny_moe_ckpt import main

    out = tmp_path / "tiny"
    main([str(out)])
    cfg = json.loads((out / "config.json").read_text())
    assert (cfg["hidden_size"], cfg["intermediate_size"], cfg["num_experts"], cfg["num_experts_per_tok"]) == \
        (256, 512, 4, 2)
    assert cfg["hidden_size"] % 128 == 0 and cfg["intermediate_size"] % 128 == 0
    ref = json.loads((out / "reference.json").read_text())
    lq = np.load(out / "reference_logits.npy")
    lf = np.load(out / "reference_logits_float.npy")
    assert lq.shape == lf.shape == (cfg["vocab_size"],)
    assert ref["ternary"]["top_ids"][0] == int(lq.argmax()) and len(ref["ternary"]["top_ids"]) == 20
    assert ref["ternary"]["max_abs_logit_change_vs_uniform_router"] > 1e-2  # experts differ: routing matters
    m = BitNetMoEForCausalLM.from_pretrained(out).eval()
    with torch.no_grad():
        got = m(torch.tensor([ref["token_ids"]])).logits[0, -1].numpy()
    np.testing.assert_allclose(got, lq, atol=1e-5)
    assert (out / "tokenizer.json").exists()
