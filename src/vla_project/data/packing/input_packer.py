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
        vision_placeholder_mode: str = "image_token",
        vision_placeholder_begin: int = C.VISION_PLACEHOLDER_BEGIN_IDX,
        prompt_position: str = "before_vision",
        num_soft_prompt_tokens_in_llm: int = 0,
        soft_prompt_placeholder_begin: int = 259462,
        include_proprio_placeholder: bool = True,
        num_wrist_tokens_in_llm: int = 0,
        wrist_placeholder_begin: int = 259494,
    ) -> None:
        super().__init__()
        if num_scene_tokens <= 0:
            raise ValueError(f"num_scene_tokens must be > 0; got {num_scene_tokens}")
        if num_action_queries <= 0:
            raise ValueError(f"num_action_queries must be > 0; got {num_action_queries}")
        if vision_placeholder_mode not in ("image_token", "unused_range"):
            raise ValueError(
                f"vision_placeholder_mode must be 'image_token' or 'unused_range'; "
                f"got {vision_placeholder_mode!r}"
            )
        if prompt_position not in ("before_vision", "after_vision"):
            raise ValueError(
                f"prompt_position must be 'before_vision' or 'after_vision'; "
                f"got {prompt_position!r}"
            )
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.prompt_max_len = prompt_max_len
        self.num_scene_tokens = num_scene_tokens
        self.num_action_queries = num_action_queries
        self.prompt_position = prompt_position
        self.num_soft_prompt_tokens_in_llm = int(num_soft_prompt_tokens_in_llm)
        # ``include_proprio_placeholder``: keep the 1-token PROPRIO_PLACEHOLDER
        # in the LLM input. Default True for v25-v32 backward compat (those
        # ckpts trained with this slot present, even though proprio bypasses
        # the LLM via proprio_proj → action_head — the slot's PLE injection
        # is part of the trained pattern). Set False for v33+ to drop the
        # dead slot and shorten the LLM input by 1 token.
        self.include_proprio_placeholder = bool(include_proprio_placeholder)
        # ``num_wrist_tokens_in_llm > 0``: reserve a wrist slot in the LLM
        # input AFTER the language prompt (π₀-style fixed-slot layout). The
        # caller scatters wrist embeddings (or zeros for masked / missing
        # wrist) into this slot. ``wrist_placeholder_begin`` defaults to
        # 259494 = ``soft_prompt_placeholder_begin`` (259462) + 32 reserved
        # for soft prompts; tweak only when you change either upstream.
        self.num_wrist_tokens_in_llm = int(num_wrist_tokens_in_llm)

        if vision_placeholder_mode == "image_token":
            scene = torch.full((num_scene_tokens,), C.IMAGE_SOFT_TOKEN_ID, dtype=torch.long)
        else:
            scene = torch.arange(
                vision_placeholder_begin,
                vision_placeholder_begin + num_scene_tokens,
                dtype=torch.long,
            )
        action = torch.arange(C.ACTION_TOKEN_BEGIN_IDX,
                              C.ACTION_TOKEN_BEGIN_IDX + num_action_queries)
        self.register_buffer("_scene", scene, persistent=False)
        self.register_buffer("_action", action, persistent=False)
        if self.num_soft_prompt_tokens_in_llm > 0:
            sp = torch.arange(
                soft_prompt_placeholder_begin,
                soft_prompt_placeholder_begin + self.num_soft_prompt_tokens_in_llm,
                dtype=torch.long,
            )
            self.register_buffer("_soft_prompt", sp, persistent=False)
        else:
            self._soft_prompt = None
        if self.num_wrist_tokens_in_llm > 0:
            wrist = torch.arange(
                wrist_placeholder_begin,
                wrist_placeholder_begin + self.num_wrist_tokens_in_llm,
                dtype=torch.long,
            )
            self.register_buffer("_wrist", wrist, persistent=False)
        else:
            self._wrist = None

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

        # Build the variable middle section based on flags. Layout:
        #   v33 (after_vision + soft_prompt): [BOS, soft_prompt(N_sp), scene(N_v), prompt(Lp), proprio(1), action(64), EOS]
        #   default (before_vision):           [BOS, prompt(Lp), scene(N_v), proprio(1), action(64), EOS]
        ones = lambda n: torch.ones(B, n, dtype=torch.long, device=device)
        sections_ids = [bos]
        sections_am = [ones(1)]
        idx_marks: Dict[str, slice] = {}
        cur = 1

        if self.num_soft_prompt_tokens_in_llm > 0:
            sp = self._soft_prompt.to(device).unsqueeze(0).expand(B, -1)
            sections_ids.append(sp)
            sections_am.append(ones(sp.shape[1]))
            idx_marks["soft_prompt"] = slice(cur, cur + sp.shape[1])
            cur += sp.shape[1]

        if self.prompt_position == "before_vision":
            sections_ids.append(prompt_input_ids)
            sections_am.append(prompt_attention_mask)
            idx_marks["prompt"] = slice(cur, cur + Lp)
            cur += Lp
            sections_ids.append(scene)
            sections_am.append(ones(scene.shape[1]))
            idx_marks["scene"] = slice(cur, cur + scene.shape[1])
            cur += scene.shape[1]
        else:  # after_vision
            sections_ids.append(scene)
            sections_am.append(ones(scene.shape[1]))
            idx_marks["scene"] = slice(cur, cur + scene.shape[1])
            cur += scene.shape[1]
            sections_ids.append(prompt_input_ids)
            sections_am.append(prompt_attention_mask)
            idx_marks["prompt"] = slice(cur, cur + Lp)
            cur += Lp

        # v36: wrist slot AFTER prompt (π₀ "fixed slot + mask" convention).
        # The caller (VLAPolicy) scatters wrist features (or zeros for masked
        # / missing wrist) into this slot; the placeholder IDs anchor stable
        # PLE / RoPE positions across train and eval.
        if self.num_wrist_tokens_in_llm > 0:
            wrist = self._wrist.to(device).unsqueeze(0).expand(B, -1)
            sections_ids.append(wrist)
            sections_am.append(ones(wrist.shape[1]))
            idx_marks["wrist"] = slice(cur, cur + wrist.shape[1])
            cur += wrist.shape[1]

        if self.include_proprio_placeholder:
            sections_ids.append(proprio)
            sections_am.append(ones(1))
            idx_marks["proprio"] = slice(cur, cur + 1)
            cur += 1
        sections_ids.append(action)
        sections_am.append(ones(action.shape[1]))
        idx_marks["action"] = slice(cur, cur + action.shape[1])
        cur += action.shape[1]
        sections_ids.append(eos)
        sections_am.append(ones(1))

        ids = torch.cat(sections_ids, dim=1)
        am = torch.cat(sections_am, dim=1)

        idx: Dict[str, torch.Tensor] = {
            name: torch.arange(s.start, s.stop, device=device).expand(B, -1)
            for name, s in idx_marks.items()
        }
        return PackedIDs(input_ids=ids, attention_mask=am, idx=idx)
