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
