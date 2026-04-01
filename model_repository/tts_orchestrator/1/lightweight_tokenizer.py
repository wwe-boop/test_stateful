"""
Lightweight text tokenizer for TTS Orchestrator.

Uses the `tokenizers` library (Rust backend, ~5 MB) with special-token
registration from tokenizer_config.json. No torch/transformers required.

Interface: callable(text, return_tensors="pt") -> {"input_ids": np.ndarray int64}.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger("lightweight_tokenizer")


def _parse_merges(merges_path: Path) -> list:
    """Parse merges.txt (GPT2/BPE format: one 'a b' per line)."""
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                merges.append((parts[0], parts[1]))
    return merges


def _load_added_tokens(root: Path) -> List[Any]:
    """
    Parse added_tokens_decoder from tokenizer_config.json and return a list
    of tokenizers.AddedToken objects for special-token registration.
    """
    try:
        from tokenizers import AddedToken
    except ImportError:
        return []

    cfg_path = root / "tokenizer_config.json"
    if not cfg_path.is_file():
        return []

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return []

    decoder = cfg.get("added_tokens_decoder", {})
    if not decoder:
        return []

    tokens = []
    for _tid_str, info in sorted(decoder.items(), key=lambda x: int(x[0])):
        content = info.get("content", "")
        if not content:
            continue
        tokens.append(
            AddedToken(
                content,
                special=info.get("special", False),
                normalized=info.get("normalized", False),
                lstrip=info.get("lstrip", False),
                rstrip=info.get("rstrip", False),
                single_word=info.get("single_word", False),
            )
        )
    return tokens


def load_lightweight_tokenizer(tokenizer_dir: str) -> Optional[Any]:
    """
    Load a BPE tokenizer from tokenizer_dir using only the tokenizers library.

    Strategy:
      1) tokenizer.json (single file, already includes special tokens)
      2) vocab.json + merges.txt, then register special tokens from
         tokenizer_config.json so that <|im_start|>, <|im_end|> etc. are
         recognized as single tokens instead of being split into sub-pieces.

    Returns an object with __call__(text, return_tensors="pt") ->
    {"input_ids": np.ndarray [1, S] int64}, or None if loading fails.
    """
    root = Path(tokenizer_dir)
    if not root.is_dir():
        return None

    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
    except ImportError:
        logger.debug("tokenizers not available, skip lightweight load")
        return None

    # 1) Prefer single-file tokenizer.json (already embeds special tokens)
    tokenizer_json = root / "tokenizer.json"
    if tokenizer_json.is_file():
        try:
            tok = Tokenizer.from_file(str(tokenizer_json))
            return _WrapTokenizer(tok)
        except Exception as e:
            logger.debug("tokenizer.json load failed: %s", e)

    # 2) Build from vocab.json + merges.txt + tokenizer_config.json
    vocab_path = root / "vocab.json"
    merges_path = root / "merges.txt"
    if not vocab_path.is_file() or not merges_path.is_file():
        logger.debug("vocab.json or merges.txt missing")
        return None

    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        merges = _parse_merges(merges_path)
        bpe = BPE(vocab=vocab, merges=merges, unk_token=None)
        tok = Tokenizer(bpe)
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)

        added_tokens = _load_added_tokens(root)
        if added_tokens:
            tok.add_special_tokens(added_tokens)
            logger.info(
                "Registered %d special tokens from tokenizer_config.json",
                len(added_tokens),
            )

        return _WrapTokenizer(tok)
    except Exception as e:
        logger.debug("BPE from vocab+merges failed: %s", e)
        return None


class _WrapTokenizer:
    """Wrapper so that tokenizer(text, return_tensors='pt')['input_ids'] is numpy [1, S] int64."""

    def __init__(self, tokenizer: Any):
        self._tok = tokenizer

    def encode_with_offsets(self, text: str):
        """Return token ids and (char_start, char_end) pairs for each token."""
        enc = self._tok.encode(text)
        return enc.ids, enc.offsets

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Union[list, np.ndarray]]:
        enc = self._tok.encode(text, add_special_tokens=kwargs.get("add_special_tokens", True))
        ids = enc.ids
        if return_tensors == "pt":
            return {"input_ids": np.array([ids], dtype=np.int64)}
        return {"input_ids": ids}
