"""LLaVA-format data pipeline for BitNet-VL.

Record format (llava_v1_5_mix665k / LCS-558K style)::

    {"image": "coco/train2017/000000033471.jpg",
     "conversations": [{"from": "human", "value": "<image>\\nWhat is the man doing?"},
                       {"from": "gpt", "value": "Riding a bike."}, ...]}

Prompt template (BitNet 2B4T chat template, image before the first question)::

    <|begin_of_text|>User: <image>\\n{q1}<|eot_id|>Assistant: {a1}<|eot_id|>User: {q2}<|eot_id|>Assistant: {a2}<|eot_id|>

Each segment is tokenized separately (``add_special_tokens=False``) so the prompt part ends exactly like the inference
prompt ``...Assistant: `` that llama.cpp tokenizes. Labels are -100 everywhere except the assistant answer tokens and
their closing ``<|eot_id|>``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer_utils import BOS_TOKEN, EOT_TOKEN, IMAGE_TOKEN

IGNORE_INDEX = -100


def build_prompt(question: str, with_image: bool = True) -> str:
    """Inference prompt for a single question."""
    q = question.replace(IMAGE_TOKEN, "").strip()
    img = f"{IMAGE_TOKEN}\n" if with_image else ""
    return f"{BOS_TOKEN}User: {img}{q}{EOT_TOKEN}Assistant: "


def preprocess_image(image, size: int = 384, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
    """SigLIP preprocessing: RGB, bicubic resize to size x size (no crop), /255, (x - mean) / std. -> (3, size, size)"""
    from PIL import Image

    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    image = image.convert("RGB").resize((size, size), resample=Image.Resampling.BICUBIC)
    x = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).view(size, size, 3)
    x = x.permute(2, 0, 1).float() / 255.0
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (x - m) / s


def tokenize_conversation(tokenizer, conversations: list[dict], has_image: bool, max_length: int | None = None,
                          num_image_tokens: int = 0):
    """Returns (input_ids, labels) as LongTensors. `max_length` counts the expanded image tokens."""
    segments: list[tuple[str, bool]] = [(BOS_TOKEN, False)]
    first_human = True
    for turn in conversations:
        role = turn.get("from", turn.get("role"))
        value = turn.get("value", turn.get("content", ""))
        if role in ("human", "user"):
            q = value.replace(IMAGE_TOKEN, "").strip()
            if first_human and has_image:
                q = f"{IMAGE_TOKEN}\n{q}"
            first_human = False
            segments.append((f"User: {q}{EOT_TOKEN}", False))
        elif role in ("gpt", "assistant"):
            segments.append(("Assistant: ", False))
            segments.append((f"{value.strip()}{EOT_TOKEN}", True))
        elif role == "system":
            segments.append((f"System: {value.strip()}{EOT_TOKEN}", False))
        else:
            raise ValueError(f"unknown role {role!r}")
    ids, labels = [], []
    for text, train in segments:
        t = tokenizer(text, add_special_tokens=False).input_ids
        ids.extend(t)
        labels.extend(t if train else [IGNORE_INDEX] * len(t))
    if max_length is not None:
        budget = max_length - (num_image_tokens - 1 if has_image else 0)
        ids, labels = ids[:budget], labels[:budget]
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class LlavaDataset(Dataset):
    def __init__(self, data_path: str | os.PathLike | list, tokenizer, image_folder: str | os.PathLike = ".",
                 image_size: int = 384, max_length: int = 2048, num_image_tokens: int = 144,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        if isinstance(data_path, list):
            self.records = data_path
        else:
            with open(data_path, encoding="utf-8") as f:
                self.records = json.load(f) if str(data_path).endswith(".json") else [json.loads(l) for l in f if l.strip()]
        self.tokenizer = tokenizer
        self.image_folder = Path(image_folder)
        self.image_size = image_size
        self.max_length = max_length
        self.num_image_tokens = num_image_tokens
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        has_image = rec.get("image") is not None
        input_ids, labels = tokenize_conversation(self.tokenizer, rec["conversations"], has_image,
                                                  self.max_length, self.num_image_tokens)
        item = {"input_ids": input_ids, "labels": labels}
        if has_image:
            item["pixel_values"] = preprocess_image(self.image_folder / rec["image"], self.image_size,
                                                    self.mean, self.std)
        return item


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        L = max(len(b["input_ids"]) for b in batch)
        n = len(batch)
        input_ids = torch.full((n, L), self.pad_token_id, dtype=torch.long)
        labels = torch.full((n, L), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((n, L), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            input_ids[i, :k] = b["input_ids"]
            labels[i, :k] = b["labels"]
            attention_mask[i, :k] = 1
        out = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        pv = [b["pixel_values"] for b in batch if "pixel_values" in b]
        out["pixel_values"] = torch.stack(pv) if pv else None
        return out
