import torch


def scatter_into_embeds(
    embeds: torch.Tensor,    # [B, L, D]
    idx: torch.Tensor,       # [B, K] long
    new: torch.Tensor,       # [B, K, D]
) -> torch.Tensor:
    """Returns a clone of `embeds` with rows at `idx` replaced by `new`."""
    assert embeds.dim() == 3, embeds.shape
    assert idx.dim() == 2, idx.shape
    assert new.dim() == 3, new.shape
    assert embeds.shape[0] == idx.shape[0] == new.shape[0]
    assert idx.shape[1] == new.shape[1]
    assert embeds.shape[-1] == new.shape[-1]
    out = embeds.clone()
    bs = torch.arange(embeds.shape[0], device=embeds.device).unsqueeze(1).expand_as(idx)
    out[bs, idx] = new
    return out
