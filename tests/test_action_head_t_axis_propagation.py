"""Regression tests for the t-axis-collapse bug fixed by the proper-FFN
MLPResNetBlock_Pro (pre-LN + 4× FFN + dual residual).

The legacy block (``return self.ffn(out + x)`` with ``ffn = LN + Linear + ReLU``,
no post-FFN residual) collapsed t-axis variation through a 35-block stack at
inference, producing identical action-chunk anchors regardless of input. These
tests pin the fix so a future refactor cannot regress to the dead-head state.
"""
import torch

from vla_project.models.action_heads.mlp_resnet import MLPResNet
from vla_project.models.action_heads.mlp_resnet_block_pro import MLPResNetBlock_Pro


def _fake_inputs(B, T, D, num_blocks, K_a, K_t, dtype=torch.float32, device="cpu"):
    h_a = torch.randn(B, num_blocks + 1, K_a, D, dtype=dtype, device=device)
    h_t = torch.randn(B, num_blocks + 1, K_t, D, dtype=dtype, device=device)
    p = torch.randn(B, 1, D, dtype=dtype, device=device)
    return h_a, h_t, p


def test_block_has_dual_residual_modules():
    blk = MLPResNetBlock_Pro(dim=64)
    # The fix replaces the legacy ``self.ffn = Sequential(...)`` with explicit
    # pre-LN + ffn_up + ffn_down. Pin the module names so a regression to the
    # legacy attribute layout is caught.
    names = {n for n, _ in blk.named_modules()}
    assert "norm1" in names
    assert "norm2" in names
    assert "ffn_up" in names
    assert "ffn_down" in names
    assert "ffn" not in names, "legacy `self.ffn` must be gone"


def test_single_block_t_axis_survives():
    """A single block fed varying-across-t input must produce varying-across-t
    output (smoke against the obvious failure mode)."""
    torch.manual_seed(0)
    B, T, D = 1, 8, 64
    K_a, K_t = 65, 32
    blk = MLPResNetBlock_Pro(dim=D)
    x = torch.randn(B, T, D)
    h_a, h_t, p = _fake_inputs(B, T, D, num_blocks=1, K_a=K_a - 1, K_t=K_t)
    # __init__ only — pass the layer-1 slice directly.
    out = blk(x, h_a=h_a[:, 0], h_t=h_t[:, 0], p=p)
    assert out.shape == (B, T, D)
    diff = (out[:, -1] - out[:, 0]).abs().max().item()
    assert diff > 1e-3, f"single-block t-axis variation collapsed: max-diff={diff:.2e}"


def test_stacked_blocks_preserve_t_axis_variation():
    """The 35-block (production) stack must NOT exponentially decay or
    saturate t-axis variation. Run with random LAC; require the final
    block's per-t output to differ across t by a meaningful margin AND the
    overall magnitude to remain stable (no >10× collapse from input)."""
    torch.manual_seed(0)
    B, T, D, A = 1, 8, 64, 4
    num_blocks = 35
    K_a, K_t = 65, 32
    head = MLPResNet(
        num_blocks=num_blocks, input_dim=A * D, hidden_dim=D,
        output_dim=D, action_dim=A,
    )
    head.eval()

    # Random across-t input (each row has different feature values, so
    # LayerNorm cannot collapse to zero).
    x = torch.randn(B, T, A * D) * 0.3
    h_a, h_t, p = _fake_inputs(B, T, D, num_blocks=num_blocks, K_a=K_a - 1, K_t=K_t)

    # Hook every block to verify variation propagates monotonically (or at
    # least never decays below a small threshold) through the stack.
    captures = []
    handles = []
    for blk in head.blocks:
        captures.append({})
        cap = captures[-1]
        def make_hook(c):
            def fn(_mod, _inp, out):
                c["out"] = out.detach()
            return fn
        handles.append(blk.register_forward_hook(make_hook(cap)))
    with torch.no_grad():
        out = head(x, h_a=h_a, h_t=h_t, p=p)
    for h in handles:
        h.remove()

    # Acceptance: every block's t-axis variation > input scale × 0.1, AND
    # final-block magnitude is within an order of magnitude of mid-block.
    block_diffs = []
    block_mags = []
    for cap in captures:
        b = cap["out"][0]
        block_diffs.append((b[-1] - b[0]).abs().max().item())
        block_mags.append(b.abs().mean().item())
    min_diff = min(block_diffs)
    last_mag = block_mags[-1]
    mid_mag = block_mags[num_blocks // 2]
    assert min_diff > 0.05, (
        f"some block collapsed t-axis variation: min={min_diff:.2e}, "
        f"per-block diffs={block_diffs}"
    )
    assert last_mag > mid_mag * 0.1, (
        f"final-block magnitude collapsed vs mid-block: last={last_mag:.4f}, "
        f"mid={mid_mag:.4f}"
    )

    # Also: final head output must vary across t.
    out_diff = (out[0, -1] - out[0, 0]).abs().max().item()
    assert out_diff > 1e-3, f"final pred t-axis collapsed: max-diff={out_diff:.2e}"


def test_last_action_chunk_affects_output():
    """Different ``x`` inputs must produce different head outputs (non-trivial
    LAC sensitivity). The legacy block trained models that ignored x at depth."""
    torch.manual_seed(0)
    B, T, D, A = 1, 8, 64, 4
    num_blocks = 35
    K_a, K_t = 65, 32
    head = MLPResNet(
        num_blocks=num_blocks, input_dim=A * D, hidden_dim=D,
        output_dim=D, action_dim=A,
    )
    head.eval()
    h_a, h_t, p = _fake_inputs(B, T, D, num_blocks=num_blocks, K_a=K_a - 1, K_t=K_t)

    x_a = torch.randn(B, T, A * D, generator=torch.Generator().manual_seed(11)) * 0.3
    x_b = torch.randn(B, T, A * D, generator=torch.Generator().manual_seed(22)) * 0.3
    with torch.no_grad():
        out_a = head(x_a, h_a=h_a, h_t=h_t, p=p)
        out_b = head(x_b, h_a=h_a, h_t=h_t, p=p)
    diff = (out_a - out_b).abs().max().item()
    assert diff > 1e-2, f"head ignores last_action_chunk input: max-diff={diff:.2e}"
