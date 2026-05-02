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
    """Constructs Gemma4 input_ids matching the VLA-Adapter reference layout
    that achieved the 73% LIBERO baseline.

    Layout (per sample):
        [BOS,
         prompt text  x Lt   (padded with 0 / pad_token),
         Scene        x Ns   (IMAGE_SOFT_TOKEN_ID repeated),
         PROPRIO      x 1    (PROPRIO_PLACEHOLDER_IDX),
         ActionQuery  x Q    (range starting at ACTION_TOKEN_BEGIN_IDX),
         EOS]

    Wrist tokens and soft prompts do NOT enter the LLM. They are produced
    independently (wrist_proj on SigLIP wrist features, soft_prompt_hub on
    domain_id) and feed the action head's self-attn pool directly via
    ``h_w`` / ``h_sp``. Reference verifies (modeling_prismatic_gemma4.py:625-630):
    "LLM inputs_embeds への prepend (案 B deviation) は廃止". Keeping
    wrist / soft inside the LLM input distorted RoPE positions for the prompt
    + wasted attention budget on tokens the head was already going to read
    via separate streams.

    The proprio dim is one PROPRIO_PLACEHOLDER token between vision and
    action queries; the LLM's hidden state at that position is unused (the
    head receives proprio via the separate proprio_proj path), but the token
    occupies a position that the trained reference layout includes.
    """

    def __init__(
        self,
        bos_id: int,
        eos_id: int,
        prompt_max_len: int,
        num_scene_tokens: int = C.NUM_SCENE_TOKENS,
        num_action_queries: int = C.NUM_ACTION_TOKENS,
    ) -> None:
        super().__init__()
        if num_scene_tokens <= 0:
            raise ValueError(f"num_scene_tokens must be > 0; got {num_scene_tokens}")
        if num_action_queries <= 0:
            raise ValueError(f"num_action_queries must be > 0; got {num_action_queries}")
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.prompt_max_len = prompt_max_len
        self.num_scene_tokens = num_scene_tokens
        self.num_action_queries = num_action_queries

        scene = torch.full((num_scene_tokens,), C.IMAGE_SOFT_TOKEN_ID, dtype=torch.long)
        action = torch.arange(C.ACTION_TOKEN_BEGIN_IDX,
                              C.ACTION_TOKEN_BEGIN_IDX + num_action_queries)
        self.register_buffer("_scene", scene, persistent=False)
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
        scene = self._scene.to(device).unsqueeze(0).expand(B, -1)
        action = self._action.to(device).unsqueeze(0).expand(B, -1)
        proprio = torch.full((B, 1), C.PROPRIO_PLACEHOLDER_IDX, dtype=torch.long, device=device)

        ids = torch.cat([bos, prompt_input_ids, scene, proprio, action, eos], dim=1)

        ones = lambda n: torch.ones(B, n, dtype=torch.long, device=device)
        am = torch.cat(
            [
                ones(1),
                prompt_attention_mask,
                ones(scene.shape[1]),
                ones(1),
                ones(action.shape[1]),
                ones(1),
            ],
            dim=1,
        )

        cur = 1
        prompt_idx = torch.arange(cur, cur + Lp, device=device).expand(B, -1)
        cur += Lp
        scene_idx = torch.arange(cur, cur + scene.shape[1], device=device).expand(B, -1)
        cur += scene.shape[1]
        proprio_idx = torch.arange(cur, cur + 1, device=device).expand(B, -1)
        cur += 1
        action_idx = torch.arange(cur, cur + action.shape[1], device=device).expand(B, -1)
        cur += action.shape[1]

        idx: Dict[str, torch.Tensor] = {
            "prompt": prompt_idx,
            "scene": scene_idx,
            "proprio": proprio_idx,
            "action": action_idx,
        }

        return PackedIDs(input_ids=ids, attention_mask=am, idx=idx)
