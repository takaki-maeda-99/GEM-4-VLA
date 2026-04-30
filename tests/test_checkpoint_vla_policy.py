"""Integration round-trip: save + load on the real VLAPolicy state_dict."""
from pathlib import Path

import torch

from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.training.checkpoint import load_checkpoint, save_checkpoint
from tests._stubs import _StubGemma, _StubSig


def _make_policy() -> VLAPolicy:
    # hidden_dim=32 matches _StubGemma.hidden_dim; num_blocks=4 matches
    # _StubGemma.num_layers so the MLPResNet block count aligns with the stubs.
    cfg = VLAPolicyConfig(
        num_domains=1,
        hidden_dim=32,
        num_blocks=4,
        num_action_queries=4,
        num_soft_prompt_tokens=4,
    )
    return VLAPolicy(cfg, _StubSig(), _StubGemma())


def test_vla_policy_round_trip(tmp_path: Path) -> None:
    p1 = _make_policy()
    # Mutate one parameter so the saved state diverges from a fresh init.
    # DomainAwareLinear stores weights in fc.weight (nn.Embedding rows).
    with torch.no_grad():
        p1.action_decoder.fc.weight.fill_(0.123)

    out = tmp_path / "step_100"
    save_checkpoint(out, p1, step=100, cfg={"smoke": True})

    p2 = _make_policy()
    meta = load_checkpoint(out, p2)

    sd1 = p1.state_dict()
    sd2 = p2.state_dict()
    assert set(sd1.keys()) == set(sd2.keys())
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), f"mismatch at {k}"
    assert meta["step"] == 100
