"""Offline tokenizer helpers.

The real BitNet b1.58 2B4T tokenizer is the Llama-3 BPE tokenizer (128256 tokens). For offline / tiny tests we
rebuild an equivalent HF fast tokenizer from the vocab GGUF shipped with llama.cpp
(`3rdparty/llama.cpp/models/ggml-vocab-llama-bpe.gguf`): tokens, merges and token types come from the GGUF, the
pre-tokenizer / post-processor are the ones of Meta-Llama-3's `tokenizer.json`. The result hashes to "llama-bpe" in
`utils/convert-hf-to-gguf-bitnet.py::get_vocab_base_pre`, so a checkpoint saved with it converts with the stock
converter.
"""

from __future__ import annotations

import sys
from pathlib import Path

IMAGE_TOKEN = "<image>"
BOS_TOKEN = "<|begin_of_text|>"
EOT_TOKEN = "<|eot_id|>"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VOCAB_GGUF = REPO_ROOT / "3rdparty" / "llama.cpp" / "models" / "ggml-vocab-llama-bpe.gguf"

# Renders exactly the training format of data.py (BitNet 2B4T style "Role: content<|eot_id|>").
CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{{ (message['role'] | capitalize) + ': ' + (message['content'] | trim) + '<|eot_id|>' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
)

LLAMA3_PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _read_gguf_vocab(path: Path):
    gguf_py = REPO_ROOT / "3rdparty" / "llama.cpp" / "gguf-py"
    if str(gguf_py) not in sys.path:
        sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader  # noqa: E402

    r = GGUFReader(str(path))

    def strarr(key):
        f = r.fields[key]
        return [bytes(f.parts[i]).decode("utf-8") for i in f.data]

    tokens = strarr("tokenizer.ggml.tokens")
    merges = strarr("tokenizer.ggml.merges")
    tt = r.fields["tokenizer.ggml.token_type"]
    types = [int(tt.parts[i][0]) for i in tt.data]
    return tokens, merges, types


def build_llama3_tokenizer(vocab_gguf: str | Path = DEFAULT_VOCAB_GGUF):
    """Build a Llama-3 `PreTrainedTokenizerFast` from llama.cpp's vocab GGUF (no network)."""
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    tokens, merges, types = _read_gguf_vocab(Path(vocab_gguf))
    vocab = {t: i for i, t in enumerate(tokens) if types[i] == 1}
    merge_pairs = [tuple(m.split(" ", 1)) for m in merges]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merge_pairs, ignore_merges=True, fuse_unk=False, byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(LLAMA3_PRETOKENIZE_REGEX), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)
    # special (control) tokens keep their GGUF ids: add them in id order
    specials = [(i, t) for i, t in enumerate(tokens) if types[i] != 1]
    assert [i for i, _ in specials] == list(range(len(vocab), len(tokens))), "special tokens must be contiguous"
    tok.add_special_tokens([AddedToken(t, special=True, normalized=False) for _, t in specials])
    bos_id = tokens.index(BOS_TOKEN)
    tok.post_processor = processors.Sequence(
        [
            processors.ByteLevel(trim_offsets=False),
            processors.TemplateProcessing(
                single=f"{BOS_TOKEN} $A", pair=f"{BOS_TOKEN} $A {BOS_TOKEN} $B", special_tokens=[(BOS_TOKEN, bos_id)]
            ),
        ]
    )
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=BOS_TOKEN,
        eos_token=EOT_TOKEN,
        pad_token=EOT_TOKEN,
        clean_up_tokenization_spaces=True,
        model_max_length=4096,
    )
    hf.chat_template = CHAT_TEMPLATE
    return hf


def add_image_token(tokenizer) -> int:
    """Add the `<image>` special token (idempotent) and return its id."""
    if IMAGE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = EOT_TOKEN if EOT_TOKEN in tokenizer.get_vocab() else tokenizer.eos_token
    return tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
