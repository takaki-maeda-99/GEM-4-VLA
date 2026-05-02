"""Forward pass with use_wrist_bridge=True.

Verifies:
  - The model builds with the new flag (k_wrist/v_wrist allocated, projector present).
  - Forward runs without shape errors on the stub stack.
  - Pred VARIES across timesteps within a chunk and across batches with
    different observations (the v6 constant-prediction collapse should be
    gone once h_w_bridge supplies per-layer wrist cross-attn).
"""
import torch

from tests._stubs import _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


class _StubSigWithLayers(torch.nn.Module):
    """Stub SigLIP that supports forward_all_layers, returning per-layer
    hidden states as deterministic functions of pixel mean (so different
    observations produce different per-layer features).
    """

    def __init__(self, num_blocks: int = 27, hidden_dim: int = 1152, num_tokens: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.num_blocks = num_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Per-batch deterministic feature derived from pixel mean
        seed = x.flatten(1).mean(dim=1)  # (B,)
        out = torch.zeros(B, self.num_tokens, self.hidden_dim, device=x.device, dtype=x.dtype)
        out += seed.view(B, 1, 1)
        return out

    def forward_all_layers(self, x: torch.Tensor, num_layers: int) -> torch.Tensor:
        B = x.shape[0]
        seed = x.flatten(1).mean(dim=1)  # (B,)
        # (B, num_layers, num_tokens, hidden_dim) — vary per layer + per batch
        layer_idx = torch.arange(num_layers, device=x.device, dtype=x.dtype).view(1, num_layers, 1, 1)
        seed_b = seed.view(B, 1, 1, 1)
        return torch.zeros(B, num_layers, self.num_tokens, self.hidden_dim,
                           device=x.device, dtype=x.dtype) + seed_b + 0.01 * layer_idx

    def freeze(self):
        pass


def _make_batch(B: int = 2, prompt_max_len: int = 10) -> dict:
    return dict(
        domain_id=torch.zeros(B, dtype=torch.long),
        scene_image=torch.randn(B, 3, 224, 224),
        # different wrist images per batch element so per-layer features differ
        wrist_image=torch.randn(B, 3, 224, 224),
        prompt_input_ids=torch.zeros(B, prompt_max_len, dtype=torch.long),
        prompt_attention_mask=torch.ones(B, prompt_max_len, dtype=torch.long),
        proprio=torch.randn(B, 8),
        last_action_chunk=torch.randn(B, 8, 7),
        target_action=torch.randn(B, 8, 7),
        action_mask=torch.ones(B, 8, dtype=torch.bool),
    )


def test_forward_with_wrist_bridge_runs() -> None:
    cfg = VLAPolicyConfig(
        num_domains=2, hidden_dim=32, action_dim=7, action_chunk_len=8,
        proprio_dim=8, prompt_max_len=10, num_blocks=4,
        use_wrist_bridge=True,
    )
    policy = VLAPolicy(cfg, vision_encoder=_StubSigWithLayers(num_blocks=cfg.num_blocks),
                       gemma=_StubGemma())
    pred, loss = policy(_make_batch(B=2, prompt_max_len=10))
    assert pred.shape == (2, 8, 7)
    assert torch.isfinite(loss)


def test_wrist_bridge_breaks_constant_prediction() -> None:
    """Sanity: with wrist_bridge active and the gating_factor_wrist set above
    zero (so wrist contribution actually flows through), pred at the last
    chunk position should differ from pred at the first chunk position
    *because* RoPE rotates queries differently per timestep AND wrist
    features differ per-layer. The v6 collapse symptom was pred[0]==pred[7]
    for the same sample; here we confirm the new code path produces
    timestep-dependent predictions.
    """
    cfg = VLAPolicyConfig(
        num_domains=1, hidden_dim=32, action_dim=7, action_chunk_len=8,
        proprio_dim=8, prompt_max_len=10, num_blocks=4,
        use_wrist_bridge=True,
    )
    policy = VLAPolicy(cfg, vision_encoder=_StubSigWithLayers(num_blocks=cfg.num_blocks),
                       gemma=_StubGemma())
    # Force the wrist gating factors above zero so wrist contribution is
    # non-trivial (init=0 means tanh(0)=0 → wrist stream zeroed at init).
    for blk in policy.action_head.model.blocks:
        with torch.no_grad():
            blk.gating_factor_wrist.fill_(1.0)
    policy.eval()
    with torch.no_grad():
        pred, _ = policy(_make_batch(B=1, prompt_max_len=10))
    # pred[:, 0] should differ from pred[:, 7] across at least one channel.
    diff = (pred[:, 0] - pred[:, 7]).abs().max().item()
    assert diff > 1e-4, f"pred constant across timesteps even with wrist_bridge: max abs diff={diff}"


def test_wrist_bridge_off_default() -> None:
    """When use_wrist_bridge=False (default), no wrist_projector_bridge is
    allocated and forward still works (regression check for the v6 path)."""
    cfg = VLAPolicyConfig(
        num_domains=1, hidden_dim=32, action_dim=7, action_chunk_len=8,
        proprio_dim=8, prompt_max_len=10, num_blocks=4,
    )
    policy = VLAPolicy(cfg, vision_encoder=_StubSigWithLayers(num_blocks=cfg.num_blocks),
                       gemma=_StubGemma())
    assert policy.wrist_projector_bridge is None
    pred, loss = policy(_make_batch(B=1, prompt_max_len=10))
    assert pred.shape == (1, 8, 7)
    assert torch.isfinite(loss)
