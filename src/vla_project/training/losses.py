import torch
import torch.nn.functional as F


def _expand_mask_f32(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Broadcast mask to `like` shape; always float32 for safe accumulation
    even when `pred`/`target` are bf16 (bf16 sums underflow at small norms)."""
    return mask.unsqueeze(-1).expand_as(like).to(torch.float32)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = _expand_mask_f32(mask, pred)
    diff = (pred - target).abs().to(torch.float32) * m
    return diff.sum() / m.sum().clamp_min(1.0)


def masked_huber(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 0.1
) -> torch.Tensor:
    m = _expand_mask_f32(mask, pred)
    diff = F.smooth_l1_loss(pred, target, beta=beta, reduction="none").to(torch.float32) * m
    return diff.sum() / m.sum().clamp_min(1.0)


def masked_l1_per_sample(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-sample masked L1 → returns ``(B,)`` tensor.

    Used by the per-domain logging path in :mod:`vla_policy`. With equal mask
    counts across the batch (v37 ``action_mask`` is all True), the unweighted
    mean of this output equals :func:`masked_l1`. With variable masks, the
    correct weighted equivalent is ``(per_sample * mask_count).sum() / mask_count.sum()``;
    callers that need bit-equivalence under variable masks should use that form.
    """
    m = _expand_mask_f32(mask, pred)                    # (B, T, A)
    diff = (pred - target).abs().to(torch.float32) * m
    diff_sum = diff.flatten(1).sum(dim=1)               # (B,)
    m_sum = m.flatten(1).sum(dim=1).clamp_min(1.0)      # (B,)
    return diff_sum / m_sum                             # (B,)


def masked_huber_per_sample(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 0.1
) -> torch.Tensor:
    """Per-sample masked smooth-L1 (Huber) → ``(B,)`` tensor. See
    :func:`masked_l1_per_sample` for caveats."""
    m = _expand_mask_f32(mask, pred)
    diff = F.smooth_l1_loss(pred, target, beta=beta, reduction="none").to(torch.float32) * m
    diff_sum = diff.flatten(1).sum(dim=1)
    m_sum = m.flatten(1).sum(dim=1).clamp_min(1.0)
    return diff_sum / m_sum
