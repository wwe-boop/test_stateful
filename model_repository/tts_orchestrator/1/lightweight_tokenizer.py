"""
Lightweight text tokenizer for TTS Orchestrator.

Uses only the `tokenizers` library (Rust backend); no torch/transformers.
Interface: callable(text, return_tensors="pt") -> {"input_ids": np.ndarray int64}.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

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
                # merges.txt often uses space as separator; first token might contain \u0120 (space)
                merges.append((parts[0], parts[1]))
    return merges


def load_lightweight_tokenizer(tokenizer_dir: str) -> Optional[Any]:
    """
    Load a BPE tokenizer from tokenizer_dir using only the tokenizers library.

    Tries: 1) tokenizer.json (single file); 2) vocab.json + merges.txt.
    Returns an object with __call__(text, return_tensors="pt") -> {"input_ids": np.ndarray},
    or None if loading fails.
    """
    root = Path(tokenizer_dir)
    if not root.is_dir():
        return None

    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
    except ImportError:
        logger.debug("tokenizers not available, skip lightweight load")
        return None

    # 1) Prefer single-file tokenizer.json (same format tokenizers uses natively)
    tokenizer_json = root / "tokenizer.json"
    if tokenizer_json.is_file():
        try:
            tok = Tokenizer.from_file(str(tokenizer_json))
            return _WrapTokenizer(tok)
        except Exception as e:
            logger.debug("tokenizer.json load failed: %s", e)

    # 2) Build from vocab.json + merges.txt
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
        # GPT2/Qwen often use byte-level; default BPE tokenizer is still correct for encode
        return _WrapTokenizer(tok)
    except Exception as e:
        logger.debug("BPE from vocab+merges failed: %s", e)
        return None


class _WrapTokenizer:
    """Wrapper so that tokenizer(text, return_tensors='pt')['input_ids'] is numpy [1, S] int64."""

    def __init__(self, tokenizer: Any):
        self._tok = tokenizer

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
