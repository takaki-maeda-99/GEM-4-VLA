from typing import Dict, Sequence

import torch


def action_slice(
    abs_traj: torch.Tensor,
    idx_for_delta: Sequence[int] = (),
    idx_for_mask_proprio: Sequence[int] = (),
) -> Dict[str, torch.Tensor]:
    if abs_traj.dim() != 2 or abs_traj.size(0) < 2:
        raise ValueError("abs_traj must be [H+1, D] with H>=1")

    proprio = abs_traj[0].clone()
    action = abs_traj[1:].clone()

    if idx_for_delta:
        idx = torch.as_tensor(list(idx_for_delta), dtype=torch.long)
        action[:, idx] -= proprio[idx]
    if idx_for_mask_proprio:
        idx = torch.as_tensor(list(idx_for_mask_proprio), dtype=torch.long)
        proprio[idx] = 0.0
    return {"proprio": proprio, "action": action}
