"""Tests for XVLAAdapterPolicy."""
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from vla_project.data import constants as C
from vla_project.data.normalization import Q99Stats
from vla_project.data.transforms.image import SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.policies.xvla_adapter_policy import XVLAAdapterPolicy
from tests._stubs import _StubGemma, _StubSig


class _StubTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "right"

    def __call__(self, text, **kw):
        L = kw.get("max_length", C.DEFAULT_PROMPT_MAX_LEN)
        if isinstance(text, str):
            ids = torch.zeros(1, L, dtype=torch.long)
            mask = torch.zeros(1, L, dtype=torch.long)
            mask[0, : min(len(text.split()), L)] = 1
            return {"input_ids": ids, "attention_mask": mask}
        out_ids = torch.zeros(len(text), L, dtype=torch.long)
        out_mask = torch.zeros(len(text), L, dtype=torch.long)
        for i, t in enumerate(text):
            out_mask[i, : min(len(t.split()), L)] = 1
        return {"input_ids": out_ids, "attention_mask": out_mask}


def _build_policy() -> tuple[XVLAAdapterPolicy, VLAPolicy]:
    # NOTE: keep num_soft_prompt_tokens / num_action_queries at their default
    # (C.NUM_SOFT_PROMPT_TOKENS=32, C.NUM_ACTION_TOKENS=64). InputPacker
    # currently hardcodes those constants, so a forward pass requires the
    # cfg to match. test_checkpoint_vla_policy.py overrides them to 4/4 but
    # never runs forward — only state_dict round-trip — so it does not hit
    # the InputPacker mismatch.
    model_cfg = VLAPolicyConfig(
        num_domains=1, num_blocks=4, hidden_dim=32,
    )
    model = VLAPolicy(model_cfg, _StubSig(), _StubGemma())
    model.eval()
    stats = Q99Stats(
        q01=torch.full((C.ACTION_DIM,), -1.0, dtype=torch.float32),
        q99=torch.full((C.ACTION_DIM,),  1.0, dtype=torch.float32),
        mask=torch.tensor([True] * (C.ACTION_DIM - 1) + [False], dtype=torch.bool),
    )
    tok = GemmaPromptTokenizer(model_name=None, _tokenizer=_StubTokenizer())
    image_tx = SiglipImageTransform(size=C.SIGLIP_IMAGE_SIZE, training=False)
    p = XVLAAdapterPolicy(
        model=model,
        tokenizer=tok,
        image_transform=image_tx,
        norm_stats=stats,
        action_chunk_len=C.ACTION_CHUNK_LEN,
        domain_id=0,
    )
    return p, model


def _fake_obs() -> Dict[str, Any]:
    return {
        "scene_image": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "proprio":     np.random.randn(C.PROPRIO_DIM).astype(np.float32),
        "language":    "pick the red block",
    }


def test_select_action_returns_correct_shape_and_dtype() -> None:
    p, _ = _build_policy()
    a = p.select_action(_fake_obs())
    assert isinstance(a, np.ndarray)
    assert a.shape == (C.ACTION_DIM,)
    assert a.dtype == np.float32
    assert np.isfinite(a).all()


def test_chunking_drains_buffer_before_calling_model_again(monkeypatch) -> None:
    p, model = _build_policy()
    calls = {"n": 0}
    orig_forward = model.forward

    def counted_forward(batch):
        calls["n"] += 1
        return orig_forward(batch)

    monkeypatch.setattr(model, "forward", counted_forward)

    obs = _fake_obs()
    for _ in range(C.ACTION_CHUNK_LEN):
        p.select_action(obs)
    assert calls["n"] == 1, f"expected 1 forward; got {calls['n']}"

    p.select_action(obs)
    assert calls["n"] == 2


def test_reset_clears_buffer(monkeypatch) -> None:
    p, model = _build_policy()
    calls = {"n": 0}
    orig_forward = model.forward

    def counted_forward(batch):
        calls["n"] += 1
        return orig_forward(batch)

    monkeypatch.setattr(model, "forward", counted_forward)

    obs = _fake_obs()
    p.select_action(obs)
    assert calls["n"] == 1
    p.reset()
    p.select_action(obs)
    assert calls["n"] == 2


def test_action_is_finite_after_full_pipeline() -> None:
    p, _ = _build_policy()
    obs = _fake_obs()
    a = p.select_action(obs)
    assert np.isfinite(a).all()
    assert isinstance(a[-1], np.floating)


def test_from_checkpoint_round_trip(tmp_path: Path) -> None:
    from vla_project.training.checkpoint import save_checkpoint

    p1, model1 = _build_policy()
    with torch.no_grad():
        model1.action_decoder.fc.weight.fill_(0.05)

    out = tmp_path / "step_5"
    save_checkpoint(
        out, model1, step=5, cfg={"smoke": True},
        norm_stats={
            "libero_smoke": {
                "action": {
                    "q01":  p1.norm_stats.q01.tolist(),
                    "q99":  p1.norm_stats.q99.tolist(),
                    "mask": [bool(b) for b in p1.norm_stats.mask.tolist()],
                }
            }
        },
    )

    # NOTE: keep num_soft_prompt_tokens / num_action_queries at their default
    # (C.NUM_SOFT_PROMPT_TOKENS=32, C.NUM_ACTION_TOKENS=64). InputPacker
    # currently hardcodes those constants, so a forward pass requires the
    # cfg to match. test_checkpoint_vla_policy.py overrides them to 4/4 but
    # never runs forward — only state_dict round-trip — so it does not hit
    # the InputPacker mismatch.
    model_cfg = VLAPolicyConfig(
        num_domains=1, num_blocks=4, hidden_dim=32,
    )
    fresh_model = VLAPolicy(model_cfg, _StubSig(), _StubGemma())
    fresh_model.eval()
    tok = GemmaPromptTokenizer(model_name=None, _tokenizer=_StubTokenizer())
    image_tx = SiglipImageTransform(size=C.SIGLIP_IMAGE_SIZE, training=False)

    p2 = XVLAAdapterPolicy.from_checkpoint(
        out,
        model=fresh_model,
        tokenizer=tok,
        image_transform=image_tx,
        unnorm_key="libero_smoke",
        domain_id=0,
    )
    a = p2.select_action(_fake_obs())
    assert a.shape == (C.ACTION_DIM,)
    assert np.isfinite(a).all()
    assert torch.allclose(
        p2.model.action_decoder.fc.weight,
        torch.full_like(p2.model.action_decoder.fc.weight, 0.05),
    )
