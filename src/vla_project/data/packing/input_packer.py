from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from vla_project.data import constants as C


@dataclass
class PackedIDs:
    input_ids: torch.Tensor          # [B, L_total] long
    attention_mask: torch.Tensor     # [B, L_total] long
    idx: Dict[str, torch.Tensor]     # block name -> [B, K]


class InputPacker(nn.Module):
    """Constructs Gemma4 input_ids with placeholders + index dict.

    Layout (per sample):
        [BOS,
         SoftPrompt   x Ks   (range starting at SOFT_PROMPT_BEGIN_IDX),
         Scene        x Ns   (IMAGE_SOFT_TOKEN_ID repeated),
         prompt text  x Lt   (padded with 0),
         Wrist        x Nw   (range starting at WRIST_PLACEHOLDER_BEGIN_IDX),
         ActionQuery  x Q    (range starting at ACTION_TOKEN_BEGIN_IDX),
         EOS]

    No proprio in input_ids — it conditions only the action head.
    """

    def __init__(self, bos_id: int, eos_id: int, prompt_max_len: int) -> None:
        super().__init__()
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.prompt_max_len = prompt_max_len

        soft = torch.arange(C.SOFT_PROMPT_BEGIN_IDX,
                            C.SOFT_PROMPT_BEGIN_IDX + C.NUM_SOFT_PROMPT_TOKENS)
        scene = torch.full((C.NUM_SCENE_TOKENS,), C.IMAGE_SOFT_TOKEN_ID, dtype=torch.long)
        wrist = torch.arange(C.WRIST_PLACEHOLDER_BEGIN_IDX,
                             C.WRIST_PLACEHOLDER_BEGIN_IDX + C.NUM_WRIST_TOKENS)
        action = torch.arange(C.ACTION_TOKEN_BEGIN_IDX,
                              C.ACTION_TOKEN_BEGIN_IDX + C.NUM_ACTION_TOKENS)
        # Cached templates (registered as buffers so they move with .to(device))
        self.register_buffer("_soft", soft, persistent=False)
        self.register_buffer("_scene", scene, persistent=False)
        self.register_buffer("_wrist", wrist, persistent=False)
        self.register_buffer("_action", action, persistent=False)

    def forward(
        self,
        prompt_input_ids: torch.Tensor,        # [B, prompt_max_len] long
        prompt_attention_mask: torch.Tensor,   # [B, prompt_max_len] long
    ) -> PackedIDs:
        B = prompt_input_ids.shape[0]
        device = prompt_input_ids.device
        Lp = self.prompt_max_len
        assert prompt_input_ids.shape == (B, Lp)
        assert prompt_attention_mask.shape == (B, Lp)

        bos = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        eos = torch.full((B, 1), self.eos_id, dtype=torch.long, device=device)
        soft = self._soft.to(device).unsqueeze(0).expand(B, -1)
        scene = self._scene.to(device).unsqueeze(0).expand(B, -1)
        wrist = self._wrist.to(device).unsqueeze(0).expand(B, -1)
        action = self._action.to(device).unsqueeze(0).expand(B, -1)

        ids = torch.cat([bos, soft, scene, prompt_input_ids, wrist, action, eos], dim=1)

        # attention mask: 1 everywhere except prompt-padded positions
        ones = lambda n: torch.ones(B, n, dtype=torch.long, device=device)
        am = torch.cat(
            [
                ones(1),
                ones(soft.shape[1]),
                ones(scene.shape[1]),
                prompt_attention_mask,
                ones(wrist.shape[1]),
                ones(action.shape[1]),
                ones(1),
            ],
            dim=1,
        )

        # Indices
        cur = 1
        soft_idx = torch.arange(cur, cur + soft.shape[1], device=device).expand(B, -1)
        cur += soft.shape[1]
        scene_idx = torch.arange(cur, cur + scene.shape[1], device=device).expand(B, -1)
        cur += scene.shape[1]
        prompt_idx = torch.arange(cur, cur + Lp, device=device).expand(B, -1)
        cur += Lp
        wrist_idx = torch.arange(cur, cur + wrist.shape[1], device=device).expand(B, -1)
        cur += wrist.shape[1]
        action_idx = torch.arange(cur, cur + action.shape[1], device=device).expand(B, -1)
        cur += action.shape[1]
        # EOS not exposed

        idx: Dict[str, torch.Tensor] = {
            "soft": soft_idx,
            "scene": scene_idx,
            "prompt": prompt_idx,
            "wrist": wrist_idx,
            "action": action_idx,
        }

        return PackedIDs(input_ids=ids, attention_mask=am, idx=idx)
