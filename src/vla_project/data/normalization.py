from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class NormalizationStats:
    mean: torch.Tensor   # [D]
    std: torch.Tensor    # [D]


def _safe_std(std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return std.clamp_min(eps)


def normalize(x: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return (x - stats.mean) / _safe_std(stats.std)


def denormalize(x: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return x * _safe_std(stats.std) + stats.mean


import json
from pathlib import Path
from typing import Union


@dataclass
class Q99Stats:
    """Per-dim BOUNDS_Q99 stats for action normalization (X-VLA / OpenVLA convention).

    For dims where ``mask[i] == True``, the dim is rescaled to [-1, 1] using
    q01 / q99. Where ``mask[i] == False`` (typically the binary gripper), the
    value is passed through unchanged.
    """

    q01: torch.Tensor   # [A]
    q99: torch.Tensor   # [A]
    mask: torch.Tensor  # [A] bool


def load_q99_stats(path: Union[str, Path], unnorm_key: str) -> Q99Stats:
    """Load BOUNDS_Q99 stats from a `dataset_statistics.json` produced by the
    OpenVLA / VLA-Adapter pipelines.

    The JSON is keyed by dataset name; each entry has an ``action`` block with
    ``q01``, ``q99``, and (optionally) ``mask`` lists of length A.
    """
    payload = json.loads(Path(path).read_text())
    if unnorm_key not in payload:
        raise KeyError(
            f"unnorm_key {unnorm_key!r} not in {path}; available: {list(payload.keys())}"
        )
    action = payload[unnorm_key]["action"]
    q01 = torch.as_tensor(action["q01"], dtype=torch.float32)
    q99 = torch.as_tensor(action["q99"], dtype=torch.float32)
    if "mask" in action:
        mask = torch.as_tensor(action["mask"], dtype=torch.bool)
    else:
        mask = torch.ones_like(q01, dtype=torch.bool)
    if not (q01.shape == q99.shape == mask.shape):
        raise ValueError(
            f"q01/q99/mask shape mismatch: {q01.shape}, {q99.shape}, {mask.shape}"
        )
    return Q99Stats(q01=q01, q99=q99, mask=mask)


def normalize_action_q99(action_raw: torch.Tensor, stats: Q99Stats) -> torch.Tensor:
    """Forward BOUNDS_Q99 normalization. Inverse of the eval-time denormalize.

    For ``mask=True`` dims: rescale (q01, q99) -> (-1, 1) and clip.
    For ``mask=False`` dims: passthrough.

    Args:
        action_raw: [..., A]
        stats: Q99Stats with shape [A]

    Returns:
        Tensor of same shape and dtype as ``action_raw``.
    """
    if action_raw.shape[-1] != stats.q01.shape[0]:
        raise ValueError(
            f"action last dim {action_raw.shape[-1]} != stats dim {stats.q01.shape[0]}"
        )
    q01 = stats.q01.to(action_raw.dtype).to(action_raw.device)
    q99 = stats.q99.to(action_raw.dtype).to(action_raw.device)
    mask = stats.mask.to(action_raw.device)
    denom = (q99 - q01).clamp_min(1e-8)
    norm = (2.0 * (action_raw - q01) / denom - 1.0).clamp(-1.0, 1.0)
    return torch.where(mask, norm, action_raw)


def compute_q99_stats(
    action_arr: "np.ndarray | torch.Tensor",
    mask: "list[bool] | None" = None,
) -> Q99Stats:
    """Compute BOUNDS_Q99 stats from a 2-D array of raw actions.

    Args:
        action_arr: shape ``[N, A]`` (numpy ndarray or torch.Tensor). Each row
            is one raw action vector. Larger N gives more stable quantiles.
        mask: list of ``A`` bools indicating which dims to normalize (``True``)
            vs pass through (``False``). Defaults to all-True. Length must match
            ``action_arr.shape[-1]``.

    Returns:
        Q99Stats with ``q01``, ``q99``, ``mask`` as float32 / bool tensors of
        shape ``[A]``.
    """
    if torch.is_tensor(action_arr):
        arr = action_arr.detach().cpu().numpy()
    else:
        arr = np.asarray(action_arr)
    if arr.ndim != 2:
        raise ValueError(
            f"action_arr must be 2-D [N, A]; got rank {arr.ndim} shape {arr.shape}"
        )
    A = arr.shape[1]
    if mask is None:
        mask = [True] * A
    if len(mask) != A:
        raise ValueError(
            f"mask length {len(mask)} != action dim {A}"
        )
    q01 = np.quantile(arr.astype(np.float32), 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(arr.astype(np.float32), 0.99, axis=0).astype(np.float32)
    return Q99Stats(
        q01=torch.from_numpy(q01),
        q99=torch.from_numpy(q99),
        mask=torch.tensor(mask, dtype=torch.bool),
    )
