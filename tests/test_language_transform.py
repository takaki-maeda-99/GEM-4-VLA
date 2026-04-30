import pytest
import torch

from vla_project.data import constants as C
from vla_project.data.transforms.language import GemmaPromptTokenizer


@pytest.fixture(scope="module")
def tok() -> GemmaPromptTokenizer:
    return GemmaPromptTokenizer(model_name="google/gemma-4-E2B", max_len=C.DEFAULT_PROMPT_MAX_LEN)


def test_short_prompt_padded_right(tok: GemmaPromptTokenizer) -> None:
    out = tok("pick up the red block")
    assert out["input_ids"].shape == (C.DEFAULT_PROMPT_MAX_LEN,)
    assert out["attention_mask"].shape == (C.DEFAULT_PROMPT_MAX_LEN,)
    assert out["input_ids"].dtype == torch.long
    assert out["attention_mask"].dtype == torch.long
    # Padding lives at the right end
    assert out["attention_mask"][0].item() == 1
    assert out["attention_mask"][-1].item() == 0


def test_long_prompt_truncated(tok: GemmaPromptTokenizer) -> None:
    long = " ".join(["block"] * 200)
    out = tok(long)
    assert out["input_ids"].shape == (C.DEFAULT_PROMPT_MAX_LEN,)
    # When truncated, every position is real (mask all ones)
    assert out["attention_mask"].sum().item() == C.DEFAULT_PROMPT_MAX_LEN


def test_batch_call_stacks(tok: GemmaPromptTokenizer) -> None:
    batch = tok.batch(["pick the red block", "stack the blue cube on the green plate"])
    assert batch["input_ids"].shape == (2, C.DEFAULT_PROMPT_MAX_LEN)
    assert batch["attention_mask"].shape == (2, C.DEFAULT_PROMPT_MAX_LEN)
