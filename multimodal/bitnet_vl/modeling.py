"""BitNet-VL: SigLIP vision tower -> 2x2 pixel shuffle -> FP projector -> BitNet b1.58 LLM.

See docs/bitnet-vl-design.md for the binding decisions. Things the C++ (clip.cpp / mtmd) side must replicate exactly:

1. Vision features = ``SiglipVisionModel(...).last_hidden_state``, i.e. the output of the LAST encoder layer followed
   by ``post_layernorm``. The attention-pooling head is not used (built with ``vision_use_head=False``, so no
   ``vision_model.head.*`` keys are saved).
2. Pixel shuffle (factor s=2, SmolVLM / Idefics3 ordering), see `pixel_shuffle` / `pixel_shuffle_reference`:
   input  x[b, h*W + w, c]          (h, w in the H x W = 24 x 24 patch grid, row-major, c < D = 768)
   output y[b, i*(W/s) + j, (dy*s + dx)*D + c] = x[b, (s*i + dy)*W + (s*j + dx), c]
   for i < H/s, j < W/s, dy, dx < s. So channel block k = dy*s + dx is the patch at row offset dy, column offset dx
   (blocks ordered (0,0), (0,1), (1,0), (1,1)) and output tokens are row-major over the 12 x 12 grid.
3. Projector: LayerNorm(s*s*D, eps=1e-5) -> Linear(s*s*D, H_llm) -> GELU (exact erf form, ``nn.GELU()``) ->
   Linear(H_llm, H_llm). All with bias, full precision.
4. Prompt: ``<|begin_of_text|>User: <image>\n{question}<|eot_id|>Assistant: `` and the single ``<image>`` token is
   replaced by the num_image_tokens projected embeddings (in order).
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .bitlinear import BitLinear, replace_linears
from .tokenizer_utils import EOT_TOKEN, IMAGE_TOKEN, add_image_token

LLM_LINEAR_RE = re.compile(r"(^|\.)layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$")
VISION_LINEAR_RE = re.compile(r"(^|\.)encoder\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|mlp\.fc[12])$")


# --------------------------------------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class BitNetVLConfig:
    # --- contract keys (docs/bitnet-vl-design.md) ---
    model_type: str = "bitnet_vl"
    vision_model_name: str = "google/siglip-base-patch16-384"
    llm_model_name: str = "microsoft/bitnet-b1.58-2B-4T-bf16"
    pixel_shuffle: int = 2
    vision_ternary: bool = False
    image_size: int = 384
    num_image_tokens: int = 144
    llm_hidden: int = 2560
    # --- additional keys (informative for the exporter; never required by the contract) ---
    vision_hidden: int = 768
    patch_size: int = 16
    projector_input: int = 3072
    projector_act: str = "gelu"  # exact (erf) GELU
    projector_norm_eps: float = 1e-5
    vision_feature: str = "last_hidden_state_post_layernorm"
    pixel_shuffle_order: str = "idefics3"  # y[i*(W/s)+j, (dy*s+dx)*D + c] = x[(s*i+dy)*W + s*j+dx, c]
    image_mean: list = field(default_factory=lambda: [0.5, 0.5, 0.5])
    image_std: list = field(default_factory=lambda: [0.5, 0.5, 0.5])
    image_token: str = IMAGE_TOKEN
    image_token_id: int = -1
    llm_qat: bool = True
    vision_config: dict = field(default_factory=dict)  # SiglipVisionConfig.to_dict()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BitNetVLConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: str | os.PathLike):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "BitNetVLConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# --------------------------------------------------------------------------------------------------------------------
# pixel shuffle
# --------------------------------------------------------------------------------------------------------------------
def pixel_shuffle(x: torch.Tensor, s: int = 2) -> torch.Tensor:
    """(B, H*W, D) -> (B, H*W/s^2, D*s^2), Idefics3/SmolVLM ordering (see module docstring)."""
    b, n, d = x.shape
    h = w = int(round(n ** 0.5))
    assert h * w == n and h % s == 0, f"need a square grid divisible by {s}, got {n} tokens"
    x = x.view(b, h, w // s, d * s)              # [b, h, j, dx*D + c]
    x = x.permute(0, 2, 1, 3)                    # [b, j, h, dx*D + c]
    x = x.reshape(b, w // s, h // s, d * s * s)  # [b, j, i, dy*s*D + dx*D + c]
    x = x.permute(0, 2, 1, 3)                    # [b, i, j, ...]
    return x.reshape(b, (h // s) * (w // s), d * s * s)


def pixel_shuffle_reference(x: torch.Tensor, s: int = 2) -> torch.Tensor:
    """Loop reference of `pixel_shuffle`: the formula the C++ side must implement."""
    b, n, d = x.shape
    h = w = int(round(n ** 0.5))
    y = x.new_empty(b, (h // s) * (w // s), d * s * s)
    for i in range(h // s):
        for j in range(w // s):
            for dy in range(s):
                for dx in range(s):
                    k = dy * s + dx
                    y[:, i * (w // s) + j, k * d:(k + 1) * d] = x[:, (s * i + dy) * w + (s * j + dx), :]
    return y


# --------------------------------------------------------------------------------------------------------------------
# projector
# --------------------------------------------------------------------------------------------------------------------
class Projector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=eps)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))


@dataclass
class BitNetVLOutput:
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    vision_hidden_states: tuple | None = None  # per-encoder-layer outputs (len = num_layers), if requested
    labels: torch.Tensor | None = None  # expanded labels aligned with logits


def _strip_quant_config(cfg):
    if getattr(cfg, "quantization_config", None) is not None:
        # HF's bitnet quantizer would swap in AutoBitLinear (torch.compile'd online quant or packed offline
        # weights). We do our own QAT with BitLinear on the plain latent weights instead.
        cfg.quantization_config = None
        try:
            delattr(cfg, "quantization_config")
        except AttributeError:
            pass
    return cfg


def _ensure_rope_theta(llm_dir: Path):
    """transformers 5 stores rope_theta inside rope_parameters; the bitnet GGUF converter reads the top-level key."""
    p = llm_dir / "config.json"
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    rp = cfg.get("rope_parameters") or {}
    if "rope_theta" not in cfg and "rope_theta" in rp:
        cfg["rope_theta"] = rp["rope_theta"]
    cfg.pop("quantization_config", None)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# --------------------------------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------------------------------
class BitNetVLForConditionalGeneration(nn.Module):
    def __init__(self, config: BitNetVLConfig, vision_tower: nn.Module, language_model: nn.Module, tokenizer=None):
        super().__init__()
        self.config = config
        self.vision_tower = vision_tower
        self.language_model = language_model
        self.tokenizer = tokenizer
        vc = vision_tower.config
        s = config.pixel_shuffle
        config.vision_hidden = vc.hidden_size
        config.patch_size = vc.patch_size
        config.image_size = vc.image_size
        config.projector_input = vc.hidden_size * s * s
        config.llm_hidden = language_model.config.hidden_size
        config.num_image_tokens = (vc.image_size // vc.patch_size // s) ** 2
        config.vision_config = vc.to_dict()
        self.projector = Projector(config.projector_input, config.llm_hidden, eps=config.projector_norm_eps)
        if tokenizer is not None:
            tid = add_image_token(tokenizer)
            config.image_token_id = tid
            if language_model.get_input_embeddings().num_embeddings < len(tokenizer):
                language_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        if config.llm_qat:
            self.quantize_llm_()
        if config.vision_ternary:
            self.ternarize_vision_()

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_configs(cls, vision_config, llm_config, tokenizer=None, **cfg_kwargs):
        """Randomly initialized model (tests / tiny checkpoints)."""
        from transformers import BitNetForCausalLM, SiglipVisionModel

        vision_config.vision_use_head = False
        vt = SiglipVisionModel(vision_config)
        llm = BitNetForCausalLM(_strip_quant_config(llm_config))
        return cls(BitNetVLConfig(**cfg_kwargs), vt, llm, tokenizer)

    @classmethod
    def from_components(cls, vision_model_name: str, llm_model_name: str, dtype=torch.float32, **cfg_kwargs):
        """Stage-1 init from pretrained HF parts (SigLIP + BitNet bf16 master weights)."""
        from transformers import AutoConfig, AutoTokenizer, BitNetForCausalLM, SiglipVisionModel

        vt = SiglipVisionModel.from_pretrained(vision_model_name, vision_use_head=False, dtype=dtype)
        lcfg = _strip_quant_config(AutoConfig.from_pretrained(llm_model_name))
        llm = BitNetForCausalLM.from_pretrained(llm_model_name, config=lcfg, dtype=dtype)
        tok = AutoTokenizer.from_pretrained(llm_model_name)
        cfg = BitNetVLConfig(vision_model_name=vision_model_name, llm_model_name=llm_model_name, **cfg_kwargs)
        return cls(cfg, vt, llm, tok)

    @classmethod
    def from_pretrained(cls, ckpt_dir: str | os.PathLike, dtype=None):
        from transformers import AutoConfig, AutoTokenizer, BitNetForCausalLM, SiglipVisionConfig, SiglipVisionModel

        ckpt_dir = Path(ckpt_dir)
        cfg = BitNetVLConfig.load(ckpt_dir / "config.json")
        vcfg = SiglipVisionConfig.from_dict(dict(cfg.vision_config))
        vcfg.vision_use_head = False
        vt = SiglipVisionModel(vcfg)
        sd = load_file(str(ckpt_dir / "vision_tower.safetensors"))
        vt.load_state_dict(sd, strict=True)
        lm_dir = ckpt_dir / "language_model"
        lcfg = _strip_quant_config(AutoConfig.from_pretrained(lm_dir))
        llm = BitNetForCausalLM.from_pretrained(lm_dir, config=lcfg, dtype=dtype or torch.float32)
        tok = AutoTokenizer.from_pretrained(lm_dir)
        vision_ternary = cfg.vision_ternary
        cfg.vision_ternary = False
        model = cls(cfg, vt, llm, tok)
        model.projector.load_state_dict(load_file(str(ckpt_dir / "projector.safetensors")), strict=True)
        if vision_ternary:
            model.ternarize_vision_()
        if dtype is not None:
            model.to(dtype)
        return model

    def save_pretrained(self, out_dir: str | os.PathLike, llm_dtype: torch.dtype | None = None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.config.save(out_dir / "config.json")
        vsd = {k: v.detach().contiguous().cpu() for k, v in self.vision_tower.state_dict().items()}
        save_file(vsd, str(out_dir / "vision_tower.safetensors"), metadata={"format": "pt"})
        psd = {k: v.detach().contiguous().cpu() for k, v in self.projector.state_dict().items()}
        save_file(psd, str(out_dir / "projector.safetensors"), metadata={"format": "pt"})
        lm_dir = out_dir / "language_model"
        llm = self.language_model
        if llm_dtype is not None and llm.dtype != llm_dtype:
            llm = copy.deepcopy(llm).to(llm_dtype)
        llm.save_pretrained(lm_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(lm_dir)
        _ensure_rope_theta(lm_dir)

    # ---------------------------------------------------------------- quantization / freezing
    def quantize_llm_(self) -> list[str]:
        """Make every decoder linear (q/k/v/o/gate/up/down) a BitLinear for QAT. lm_head / embeddings stay FP.

        transformers' `BitNetForCausalLM` (v5) uses plain nn.Linear and only quantizes when a quantization_config
        makes it swap in AutoBitLinear; we strip that config at load time, so there is no double quantization."""
        self.config.llm_qat = True
        return replace_linears(self.language_model, lambda n, m: bool(LLM_LINEAR_RE.search(n)))

    def ternarize_vision_(self) -> list[str]:
        """Swap SigLIP q/k/v/out_proj/fc1/fc2 for BitLinear. Patch embed, pos embed and LayerNorms stay FP."""
        self.config.vision_ternary = True
        return replace_linears(self.vision_tower, lambda n, m: bool(VISION_LINEAR_RE.search(n)))

    def freeze_(self, vision: bool, projector: bool, llm: bool):
        """Arguments say whether each part is TRAINABLE."""
        self.vision_tower.requires_grad_(vision)
        self.projector.requires_grad_(projector)
        self.language_model.requires_grad_(llm)

    def configure_stage(self, stage: int) -> dict[str, list[nn.Parameter]]:
        """Set requires_grad for a recipe stage and return the trainable parameters by group."""
        if stage == 1:
            self.freeze_(vision=False, projector=True, llm=False)
        elif stage == 2:
            self.freeze_(vision=False, projector=True, llm=True)
        elif stage == 3:
            if not self.config.vision_ternary:
                self.ternarize_vision_()
            self.freeze_(vision=True, projector=False, llm=False)
        else:
            raise ValueError(stage)
        groups = {
            "vision": [p for p in self.vision_tower.parameters() if p.requires_grad],
            "projector": [p for p in self.projector.parameters() if p.requires_grad],
            "llm": [p for p in self.language_model.parameters() if p.requires_grad],
        }
        return {k: v for k, v in groups.items() if v}

    def gradient_checkpointing_enable(self):
        kw = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
        self.language_model.gradient_checkpointing_enable(**kw)
        self.vision_tower.gradient_checkpointing_enable(**kw)
        self.language_model.config.use_cache = False

    # ---------------------------------------------------------------- forward
    @property
    def device(self):
        return self.projector.fc1.weight.device

    def encode_images(self, pixel_values: torch.Tensor, output_hidden_states: bool = False):
        vdtype = self.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        out = self.vision_tower(pixel_values=pixel_values.to(vdtype), output_hidden_states=output_hidden_states)
        feats = pixel_shuffle(out.last_hidden_state, self.config.pixel_shuffle)
        feats = self.projector(feats.to(self.projector.fc1.weight.dtype))
        hs = tuple(out.hidden_states[1:]) if output_hidden_states else None
        return feats, hs

    def merge_inputs(self, input_ids, image_features=None, attention_mask=None, labels=None):
        """Replace each sample's single `<image>` token by the image's N projected embeddings.

        The i-th sample that contains an image token uses image_features[i'] where i' counts image-bearing samples.
        Returns (inputs_embeds, attention_mask, labels), right-padded to the longest expanded row."""
        emb = self.language_model.get_input_embeddings()
        tid = self.config.image_token_id
        bsz = input_ids.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        tok_embeds = emb(input_ids)
        rows_e, rows_m, rows_l = [], [], []
        img_i = 0
        for b in range(bsz):
            pos = (input_ids[b] == tid).nonzero(as_tuple=True)[0]
            e, m = tok_embeds[b], attention_mask[b]
            lab = labels[b] if labels is not None else None
            if len(pos) == 0:
                pass
            elif len(pos) == 1:
                if image_features is None or img_i >= image_features.shape[0]:
                    raise ValueError("more <image> tokens than images")
                p = int(pos[0])
                f = image_features[img_i].to(e.dtype)
                img_i += 1
                n = f.shape[0]
                e = torch.cat([e[:p], f, e[p + 1:]], 0)
                m = torch.cat([m[:p], m.new_ones(n), m[p + 1:]], 0)
                if lab is not None:
                    lab = torch.cat([lab[:p], lab.new_full((n,), -100), lab[p + 1:]], 0)
            else:
                raise ValueError(f"sample {b} has {len(pos)} <image> tokens; exactly one is supported")
            rows_e.append(e)
            rows_m.append(m)
            rows_l.append(lab)
        if image_features is not None and img_i != image_features.shape[0]:
            raise ValueError(f"{image_features.shape[0]} images but {img_i} <image> tokens")
        L = max(r.shape[0] for r in rows_e)
        out_e = tok_embeds.new_zeros(bsz, L, tok_embeds.shape[-1])
        out_m = attention_mask.new_zeros(bsz, L)
        out_l = None if labels is None else labels.new_full((bsz, L), -100)
        for b in range(bsz):
            n = rows_e[b].shape[0]
            out_e[b, :n] = rows_e[b]
            out_m[b, :n] = rows_m[b]
            if out_l is not None:
                out_l[b, :n] = rows_l[b]
        return out_e, out_m, out_l

    def forward(self, input_ids, pixel_values=None, attention_mask=None, labels=None,
                output_vision_hidden_states: bool = False, **kwargs) -> BitNetVLOutput:
        feats, vhs = (None, None)
        if pixel_values is not None:
            feats, vhs = self.encode_images(pixel_values, output_hidden_states=output_vision_hidden_states)
        embeds, mask, labels = self.merge_inputs(input_ids, feats, attention_mask, labels)
        out = self.language_model(inputs_embeds=embeds, attention_mask=mask, labels=labels, use_cache=False, **kwargs)
        return BitNetVLOutput(loss=out.loss, logits=out.logits, vision_hidden_states=vhs, labels=labels)

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def generate(self, input_ids, pixel_values=None, attention_mask=None, max_new_tokens: int = 64,
                 eos_token_ids=None) -> torch.Tensor:
        """Greedy decoding with KV cache. Returns generated token ids (B, T) without the prompt."""
        if eos_token_ids is None:
            eos_token_ids = set()
            if self.tokenizer is not None:
                for t in (EOT_TOKEN, self.tokenizer.eos_token):
                    if t is not None and t in self.tokenizer.get_vocab():
                        eos_token_ids.add(self.tokenizer.convert_tokens_to_ids(t))
        feats = self.encode_images(pixel_values)[0] if pixel_values is not None else None
        embeds, mask, _ = self.merge_inputs(input_ids, feats, attention_mask)
        out = self.language_model(inputs_embeds=embeds, attention_mask=mask, use_cache=True)
        past = out.past_key_values
        bsz = input_ids.shape[0]
        done = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)
        generated = []
        nxt = out.logits[:, -1].argmax(-1)
        for _ in range(max_new_tokens):
            generated.append(nxt)
            done |= torch.isin(nxt, torch.tensor(sorted(eos_token_ids), device=nxt.device, dtype=nxt.dtype)) \
                if eos_token_ids else torch.zeros_like(done)
            if bool(done.all()):
                break
            mask = torch.cat([mask, mask.new_ones(bsz, 1)], 1)
            out = self.language_model(input_ids=nxt[:, None], attention_mask=mask, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1)
        return torch.stack(generated, 1)

    @torch.no_grad()
    def answer(self, image, question: str, max_new_tokens: int = 64) -> str:
        """image: PIL image, path, or a preprocessed (3, H, W) tensor."""
        from .data import build_prompt, preprocess_image

        assert self.tokenizer is not None
        pv = image if torch.is_tensor(image) else preprocess_image(image, self.config.image_size,
                                                                  self.config.image_mean, self.config.image_std)
        ids = self.tokenizer(build_prompt(question), add_special_tokens=False, return_tensors="pt").input_ids
        gen = self.generate(ids.to(self.device), pv[None].to(self.device), max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(gen[0], skip_special_tokens=True).strip()


def bitlinear_names(module: nn.Module) -> list[str]:
    return [n for n, m in module.named_modules() if isinstance(m, BitLinear)]
