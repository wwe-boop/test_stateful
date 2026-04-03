import json
import logging

from pathlib import Path
from typing import Any, Dict, List, Tuple
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class LightQwen3TTSTokenizer:
    def __init__(self, tokenizer_dir: str, logger_level: int = logging.INFO):
        self.tokenizer_dir = Path(tokenizer_dir)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logger_level)
        
        if not self.tokenizer_dir.is_dir():
            raise ValueError(f"Tokenizer directory {self.tokenizer_dir} does not exist")
        
        self.tokenizer_json_path = self.tokenizer_dir / "tokenizer.json"
        # 先看看有没有tokenizer.json文件如果有，那么直接加载就好了
        if self.tokenizer_json_path.is_file():
            try:
                self.tokenizer = Tokenizer.from_file(str(self.tokenizer_json_path))
                self.logger.info(f"Loaded tokenizer from {self.tokenizer_json_path}")
                return
            except Exception as e:
                self.logger.error(f"Failed to load tokenizer from {self.tokenizer_json_path}: {e}")
        
        # 没有tokenizer_config.json文件，那么需要从vocab.json和merges.txt文件中构建tokenizer
        vocab_path = self.tokenizer_dir / "vocab.json"
        merges_path = self.tokenizer_dir / "merges.txt"
        
        if not vocab_path.is_file():
            raise ValueError(f"Vocab file {vocab_path} does not exist")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        
        if not merges_path.is_file():
            raise ValueError(f"Merges file {merges_path} does not exist")
        merges = self._parse_merges(merges_path)

        self.tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token=None))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()

        added_tokens = self._load_added_tokens(self.tokenizer_dir)
        if added_tokens:
            self.tokenizer.add_special_tokens(added_tokens)
            self.logger.info(f"Added {len(added_tokens)} special tokens")

        self.logger.info(f"Loaded tokenizer from {self.tokenizer_dir}")


    @staticmethod
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
    
    @staticmethod
    def _load_added_tokens(root: Path) -> List[Any]:
        """
        Parse added_tokens_decoder from tokenizer_config.json and return a list
        of tokenizers.AddedToken objects for special-token registration.
        """
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
    
    def encode(self, text: str, add_special_tokens: bool = True, **kwargs: Any):
        """Return the raw tokenizers.Encoding object."""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def encode_ids(self, text: str, add_special_tokens: bool = True, **kwargs: Any) -> List[int]:
        """Return token ids only."""
        return self.encode(text, add_special_tokens=add_special_tokens, **kwargs).ids

    def encode_with_offsets(self, text: str, add_special_tokens: bool = True, **kwargs: Any) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Return token ids and the corresponding original text offsets."""
        enc = self.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        return enc.ids, enc.offsets

    def encode_with_tokens(self, text: str, add_special_tokens: bool = True, **kwargs: Any) -> Tuple[List[int], List[str]]:
        """Return token ids and the corresponding original token."""
        enc = self.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        return enc.ids, enc.tokens
    
    def encode_with_text(self, text: str, add_special_tokens: bool = True, **kwargs: Any) -> Tuple[List[int], List[str]]:
        """Return token ids and the corresponding original text spans (via offsets)."""
        enc = self.encode(text, add_special_tokens=add_special_tokens, **kwargs)
        spans = [text[s:e] for s, e in enc.offsets]
        return enc.ids, spans
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

__all__ = ("LightQwen3TTSTokenizer",)