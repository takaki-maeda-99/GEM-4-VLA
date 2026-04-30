from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from vla_project.data import constants as C
from vla_project.data.packing.input_packer import InputPacker
from vla_project.models.action_heads.l1_regression_action_head import L1RegressionActionHead
from vla_project.models.language.embed_overwrite import scatter_into_embeds
from vla_project.models.projectors.action_queries import ActionQueryHub
from vla_project.models.projectors.domain_aware_linear import DomainAwareLinear
from vla_project.models.projectors.soft_prompts import SoftPromptHub
from vla_project.training.losses import masked_l1, masked_huber


@dataclass
class VLAPolicyConfig:
    num_domains: int
    hidden_dim: int = C.LLM_HIDDEN_DIM
    siglip_hidden_dim: int = C.SIGLIP_HIDDEN_DIM
    action_dim: int = C.ACTION_DIM
    action_chunk_len: int = C.ACTION_CHUNK_LEN
    proprio_dim: int = C.PROPRIO_DIM
    prompt_max_len: int = C.DEFAULT_PROMPT_MAX_LEN
    num_blocks: int = C.NUM_LLM_LAYERS
    num_soft_prompt_tokens: int = C.NUM_SOFT_PROMPT_TOKENS
    num_action_queries: int = C.NUM_ACTION_TOKENS
    num_scene_tokens: int = C.NUM_SCENE_TOKENS  # NEW
    num_wrist_tokens: int = C.NUM_WRIST_TOKENS  # NEW (raw from SigLIP)
    use_wrist_pool: bool = False                # NEW
    wrist_pool_tokens: int = 49                 # NEW (when use_wrist_pool=True)
    bos_id: int = 2
    eos_id: int = 1
    loss_type: str = "l1"  # or "huber"
    huber_beta: float = 0.1
    use_grad_checkpoint: bool = False


class VLAPolicy(nn.Module):
    def __init__(self, cfg: VLAPolicyConfig, vision_encoder: nn.Module, gemma: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = vision_encoder
        self.gemma = gemma

        D, A = cfg.hidden_dim, cfg.action_dim
        self.scene_proj = DomainAwareLinear(cfg.siglip_hidden_dim, D, cfg.num_domains)
        self.wrist_proj = DomainAwareLinear(cfg.siglip_hidden_dim, D, cfg.num_domains)
        self.proprio_proj = DomainAwareLinear(cfg.proprio_dim, D, cfg.num_domains)
        # NOTE: project last_action [B, T, A] -> [B, T, A*D] directly, so the
        # action-head MLPResNet's fc1 (input_dim = A*D) sees a non-redundant
        # representation per timestep. Earlier draft tiled a [B, T, D] vector
        # along a synthesized A axis, which collapses information.
        self.last_action_proj = DomainAwareLinear(A, A * D, cfg.num_domains)
        self.action_decoder = DomainAwareLinear(D, A, cfg.num_domains)

        self.soft_prompt_hub = SoftPromptHub(cfg.num_domains, cfg.num_soft_prompt_tokens, D)
        self.action_query_hub = ActionQueryHub(cfg.num_action_queries, D)  # shared, not per-domain

        # Resolve effective wrist token count: pooled value when enabled.
        effective_num_wrist = (
            cfg.wrist_pool_tokens if cfg.use_wrist_pool else cfg.num_wrist_tokens
        )
        if cfg.use_wrist_pool and effective_num_wrist <= 0:
            raise ValueError(
                f"wrist_pool_tokens must be > 0 when use_wrist_pool=True; got {cfg.wrist_pool_tokens}"
            )
        self._effective_num_wrist = int(effective_num_wrist)

        self.input_packer = InputPacker(
            cfg.bos_id, cfg.eos_id, cfg.prompt_max_len,
            num_soft_prompt_tokens=cfg.num_soft_prompt_tokens,
            num_scene_tokens=cfg.num_scene_tokens,
            num_wrist_tokens=effective_num_wrist,
            num_action_queries=cfg.num_action_queries,
        )

        self.action_head = L1RegressionActionHead(
            hidden_dim=D,
            action_dim=A,
            num_action_chunks=cfg.action_chunk_len,
            num_blocks=cfg.num_blocks,
            num_task_tokens=cfg.num_scene_tokens + cfg.prompt_max_len + effective_num_wrist,
            use_grad_checkpoint=cfg.use_grad_checkpoint,
        )

    def _pool_wrist(self, wrist_tok: torch.Tensor) -> torch.Tensor:
        """Adaptive-pool a (B, N, D) sequence by reshaping to a square grid.

        Assumes ``N`` is a perfect square (16x16 = 256 for SigLIP@224 / patch14).
        Pools to a grid of side sqrt(self._effective_num_wrist) and flattens back.
        """
        import math
        B, N, D = wrist_tok.shape
        side = int(round(math.sqrt(N)))
        if side * side != N:
            raise ValueError(f"wrist token count {N} is not a perfect square")
        pooled_side = int(round(math.sqrt(self._effective_num_wrist)))
        if pooled_side * pooled_side != self._effective_num_wrist:
            raise ValueError(
                f"wrist_pool_tokens {self._effective_num_wrist} is not a perfect square"
            )
        grid = wrist_tok.transpose(1, 2).reshape(B, D, side, side)
        pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (pooled_side, pooled_side))
        return pooled.reshape(B, D, pooled_side * pooled_side).transpose(1, 2)

    def _build_x(self, last_action: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        """Project last_action [B, T, A] -> [B, T, A*D] directly via DomainAwareLinear.

        The head's MLPResNet.fc1 expects `A*D` per timestep. We project each
        timestep independently with the per-domain weight matrix; no tiling
        and no information collapse along the A axis.
        """
        B, T, A = last_action.shape
        D = self.cfg.hidden_dim
        flat = last_action.reshape(B * T, A)
        dom = domain_id.repeat_interleave(T)
        out = self.last_action_proj(flat, dom)  # [B*T, A*D]
        return out.view(B, T, A * D)

    def forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        domain_id = batch["domain_id"]
        B = domain_id.shape[0]

        # 1. SigLIP encode (shared for both views)
        scene_tok = self.vision_encoder(batch["scene_image"])  # [B, 256, D_vis]
        wrist_tok = self.vision_encoder(batch["wrist_image"])
        if self.cfg.use_wrist_pool:
            wrist_tok = self._pool_wrist(wrist_tok)

        # 2. Project to LLM dim, per domain
        scene_e = self.scene_proj(scene_tok, domain_id)        # [B, 256, D]
        wrist_e = self.wrist_proj(wrist_tok, domain_id)        # [B, 256, D]

        # 3. Soft prompts (per-domain) and action queries (shared, broadcast)
        soft_e = self.soft_prompt_hub(domain_id)
        action_q_e = self.action_query_hub(B)

        # 4. Build input_ids + indices
        packed = self.input_packer(batch["prompt_input_ids"], batch["prompt_attention_mask"])

        # 5. Gemma forward with overwrite. Project module outputs to the LLM's
        # dtype (defensively — when the whole policy is cast to bf16 these are
        # already bf16 and the .to() is a no-op).
        raw_e = self.gemma.embed_tokens(packed.input_ids)
        llm_dtype = raw_e.dtype
        emb = scatter_into_embeds(raw_e, packed.idx["soft"], soft_e.to(llm_dtype))
        emb = scatter_into_embeds(emb, packed.idx["scene"], scene_e.to(llm_dtype))
        emb = scatter_into_embeds(emb, packed.idx["wrist"], wrist_e.to(llm_dtype))
        emb = scatter_into_embeds(emb, packed.idx["action"], action_q_e.to(llm_dtype))

        out = self.gemma(
            input_ids=packed.input_ids,
            attention_mask=packed.attention_mask,
            inputs_embeds=emb,
        )
        hs = out.hidden_states  # [B, layers+1, L, D] in llm_dtype

        # 6. Slice h_t / h_a — keep in llm_dtype (head's Linear weights match
        # when the policy is cast to bf16; for stub-based fp32 tests, hs is fp32
        # so this is also fine).
        task_idx = torch.cat([packed.idx["scene"], packed.idx["prompt"], packed.idx["wrist"]], dim=1)
        bs = torch.arange(B, device=hs.device).view(B, 1, 1)
        layers = torch.arange(hs.shape[1], device=hs.device).view(1, hs.shape[1], 1)
        h_t = hs[bs, layers, task_idx.unsqueeze(1)]                  # [B, layers+1, K_t, D]
        h_a = hs[bs, layers, packed.idx["action"].unsqueeze(1)]      # [B, layers+1, Q, D]

        # 7. x init from LastActionProj (matches policy dtype).
        x_init = self._build_x(batch["last_action_chunk"], domain_id)

        # 8. proprio -> p (matches policy dtype).
        p = self.proprio_proj(batch["proprio"], domain_id).unsqueeze(1)

        # 9. action head (policy dtype throughout).
        head_out = self.action_head(x_init, h_a=h_a, h_t=h_t, p=p)   # [B, T, D]

        # 10. action decoder.
        pred = self.action_decoder(head_out, domain_id)              # [B, T, A]

        # 11. loss
        if cfg.loss_type == "l1":
            loss = masked_l1(pred, batch["target_action"], batch["action_mask"])
        elif cfg.loss_type == "huber":
            loss = masked_huber(pred, batch["target_action"], batch["action_mask"], beta=cfg.huber_beta)
        else:
            raise ValueError(f"unknown loss_type: {cfg.loss_type}")

        return pred, loss
