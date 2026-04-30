"""Unit tests for LoRA injection.

We do not load real Gemma4. Instead we build a minimal nn.Module that
contains q_proj and v_proj linears (mimicking Gemma4's attention layout)
and verify our LoRA wiring helper:
  - adds lora_A / lora_B linears under each targeted module
  - leaves the base linears' weights frozen
  - leaves untargeted modules untouched
  - sets requires_grad=True only on the LoRA params
"""
from typing import List

import pytest
import torch
import torch.nn as nn

from vla_project.models.language.gemma4_wrapper import _apply_lora


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)


class _MiniModel(nn.Module):
    def __init__(self, n_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b.o_proj(b.v_proj(b.k_proj(b.q_proj(x))))
        return x


def _trainable(m: nn.Module) -> List[str]:
    return [n for n, p in m.named_parameters() if p.requires_grad]


def test_apply_lora_adds_lora_layers() -> None:
    m = _MiniModel(n_blocks=2)
    # Freeze everything first (matches Gemma4Wrapper's Stage 2 prep).
    for p in m.parameters():
        p.requires_grad = False
    _apply_lora(m, {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]})
    # peft replaces the targeted Linear with a wrapper exposing
    # ``lora_A`` and ``lora_B`` ModuleDicts. We just check the names exist.
    names = {n for n, _ in m.named_modules()}
    assert any("lora_A" in n for n in names), f"no lora_A modules in {names!r}"
    assert any("lora_B" in n for n in names), f"no lora_B modules in {names!r}"


def test_apply_lora_only_lora_params_trainable() -> None:
    m = _MiniModel(n_blocks=2)
    for p in m.parameters():
        p.requires_grad = False
    _apply_lora(m, {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]})
    trainable = _trainable(m)
    # All trainable params must contain "lora_" in their name.
    assert trainable, "no trainable params after LoRA injection"
    for n in trainable:
        assert "lora_" in n, f"non-lora param became trainable: {n}"
    # Base linears stay frozen — confirm by checking q_proj.weight is in
    # (renamed) base_layer and not requires_grad.
    base_names = [n for n, p in m.named_parameters() if "lora_" not in n]
    for n, p in m.named_parameters():
        if "lora_" not in n:
            assert not p.requires_grad, f"non-lora param trainable: {n}"
    assert any("q_proj" in n for n in base_names), "q_proj base weight disappeared"


def test_apply_lora_untargeted_modules_unchanged() -> None:
    m = _MiniModel(n_blocks=1)
    for p in m.parameters():
        p.requires_grad = False
    _apply_lora(m, {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]})
    # k_proj and o_proj must NOT have lora_A / lora_B sub-modules.
    for n, _ in m.named_modules():
        if "k_proj" in n or "o_proj" in n:
            assert "lora_" not in n, f"unexpected lora on untargeted module: {n}"


def test_apply_lora_forward_runs() -> None:
    m = _MiniModel(n_blocks=2)
    for p in m.parameters():
        p.requires_grad = False
    _apply_lora(m, {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]})
    x = torch.randn(3, 8)
    y = m(x)
    assert y.shape == (3, 8)
    # backward through LoRA params yields finite grads.
    y.sum().backward()
    grads = [p.grad for n, p in m.named_parameters() if "lora_" in n and p.grad is not None]
    assert grads, "no LoRA grads after backward"
    for g in grads:
        assert torch.isfinite(g).all()


def test_apply_lora_rejects_unknown_target() -> None:
    m = _MiniModel(n_blocks=1)
    for p in m.parameters():
        p.requires_grad = False
    with pytest.raises((ValueError, KeyError, RuntimeError, Exception)):
        _apply_lora(m, {"r": 4, "alpha": 8, "target_modules": ["nonexistent_proj"]})
