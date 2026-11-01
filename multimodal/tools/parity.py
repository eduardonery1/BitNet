"""Compare mtmd image embeddings (dump_embd) to the PyTorch BitNet-VL projector output.
usage: python parity.py CKPT IMAGE DUMP_BIN [--ternary]"""
import sys, json, numpy as np, torch
sys.path.insert(0, "/home/nery/Projects/BitNet/multimodal")
from pathlib import Path
from PIL import Image
from safetensors.torch import load_file
from transformers import SiglipVisionConfig, SiglipVisionModel
from bitnet_vl.modeling import BitNetVLConfig, Projector, pixel_shuffle, VISION_LINEAR_RE
from bitnet_vl.bitlinear import replace_linears
from bitnet_vl.data import preprocess_image

ckpt, img, dump = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ternary = "--ternary" in sys.argv
cfg = BitNetVLConfig.load(ckpt / "config.json")
vc = SiglipVisionConfig.from_dict(dict(cfg.vision_config)); vc.vision_use_head = False
vt = SiglipVisionModel(vc); vt.load_state_dict(load_file(str(ckpt / "vision_tower.safetensors")), strict=True)
if ternary:
    replace_linears(vt, lambda n, m: bool(VISION_LINEAR_RE.search(n)))
proj = Projector(cfg.projector_input, cfg.llm_hidden, eps=cfg.projector_norm_eps)
proj.load_state_dict(load_file(str(ckpt / "projector.safetensors")), strict=True)
vt.eval(); proj.eval()
with torch.no_grad():
    px = preprocess_image(Image.open(img), cfg.image_size, cfg.image_mean, cfg.image_std)[None].float()
    ref = proj(pixel_shuffle(vt(pixel_values=px).last_hidden_state, cfg.pixel_shuffle))[0].double().numpy()
raw = np.fromfile(dump, dtype=np.int32, count=2); nt, ne = int(raw[0]), int(raw[1])
got = np.fromfile(dump, dtype=np.float32, offset=8).reshape(nt, ne).astype(np.float64)
assert got.shape == ref.shape, (got.shape, ref.shape)
d = got - ref
cos = (got * ref).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1))
print(f"tokens={nt} n_embd={ne} ref_rms={np.sqrt((ref**2).mean()):.4g} max_abs_err={np.abs(d).max():.4g} "
      f"rel_rms_err={np.sqrt((d**2).mean() / (ref**2).mean()):.4g} min_token_cos={cos.min():.6f}")
