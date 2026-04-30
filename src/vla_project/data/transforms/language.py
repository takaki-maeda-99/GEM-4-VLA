"""Gemma4 prompt tokenizer wrapper.

Tokenizes a single language instruction or a list of instructions to a fixed
length (`prompt_max_len`), right-padding with the tokenizer's pad token.
Returns torch tensors keyed `input_ids`, `attention_mask` so the result drops
straight into the project's internal Batch schema.

The tokenizer is loaded via `AutoTokenizer.from_pretrained` and is **not**
fine-tuned. It is instantiated once per dataset (or process) and reused.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from vla_project.data import constants as C


class GemmaPromptTokenizer:
    def __init__(
        self,
        model_name: str = "google/gemma-4-E2B",
        max_len: int = C.DEFAULT_PROMPT_MAX_LEN,
        _tokenizer=None,
    ) -> None:
        self.max_len = max_len
        if _tokenizer is not None:
            self._tok = _tokenizer
        else:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(model_name)
        if self._tok.pad_token_id is None:
            # Gemma4 tokenizer ships a pad token; fall back to eos defensively.
            self._tok.pad_token = self._tok.eos_token
        # Force right-padding so pad tokens sit at the end of the sequence.
        # Gemma4's tokenizer defaults to padding_side="left"; right-padding is
        # the convention used during supervised finetuning and aligns with the
        # project's attention mask contract (mask[0]==1, mask[-1]==0 for short).
        self._tok.padding_side = "right"

    def __call__(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self._tok(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # AutoTokenizer returns [1, L]; squeeze to [L].
        return {
            "input_ids": enc["input_ids"].squeeze(0).to(torch.long),
            "attention_mask": enc["attention_mask"].squeeze(0).to(torch.long),
        }

    def batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self._tok(
            texts,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].to(torch.long),
            "attention_mask": enc["attention_mask"].to(torch.long),
        }
