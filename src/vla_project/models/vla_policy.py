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
from vla_project.training.losses_ee6d import ee6d_loss_components


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
    # NEW (when use_wrist_pool=True). Default 64 (8x8) chosen empirically:
    # SigLIP gives 16x16 patches, so factor=2 -> exact 2x2 mean pool (no
    # adaptive interpolation overhead) AND seq length aligns to a multiple
    # of 8 for friendlier cuBLAS/FlashAttention tiles. 7x7 (=49 from the
    # original X-VLA spec) was 16% slower per-step on A100 due to non-integer
    # downsample factor (16/7≈2.29). See Plan 9 follow-up benchmark.
    wrist_pool_tokens: int = 64
    # Wrist bridge (per-layer SigLIP wrist features feeding each action-head
    # block's 4th cross-attn branch). Mirrors vla-gemma-4 wristb_b16_v2 (73%
    # LIBERO baseline). When True, the action head receives a strong
    # obs-conditioning signal that bypasses the LLM entirely — this is the
    # key fix for v6's constant-prediction collapse (frozen LLM + tiny LoRA
    # couldn't push enough obs info into action positions).
    use_wrist_bridge: bool = False
    # Warm-init gating factors (atanh-space). 0 → tanh(0)=0 cold start;
    # ~0.55 → tanh≈0.5 immediate cross-attn contribution; 1.0 → tanh≈0.76.
    # Default 0 matches reference; use >0 for our smaller-bs runs to avoid
    # the ramp-too-slow collapse observed in v6/v9-step_2500 diagnostics.
    # NOTE: warm-init at 0.5 was found unable to train under bf16 (the ULP
    # at value 0.5 is ~0.0039, larger than typical AdamW updates). Use
    # ``ungated_streams=True`` instead to remove the gating bottleneck.
    gating_init: float = 0.0
    gating_init_wrist: float = 0.0
    # ``ungated_streams=True`` removes the learnable tanh gating from the
    # task and wrist cross-attn streams (fixed scale = 1.0). This avoids
    # the bf16 precision trap that froze warm-init updates in v10. The
    # model can still suppress unhelpful streams via the k_task / k_wrist
    # projection weights.
    ungated_streams: bool = False
    # ``use_soft_prompt=False`` skips the soft_prompt_hub allocation and
    # passes h_sp=None to the action head. The 73% vla-gemma-4 baseline
    # (libero finetune) ran with ``num_pretrain_datasets=0`` so its
    # soft_prompt_library was None and h_sp was never built. Default True
    # for backwards compat with multi-domain configs that need it.
    use_soft_prompt: bool = True
    bos_id: int = 2
    eos_id: int = 1
    loss_type: str = "l1"  # or "huber" or "ee6d"
    huber_beta: float = 0.1
    # EE6D loss only: per-channel weights. Defaults equal so the total roughly
    # matches the units of native L1 (channel-mean-normalized within each group).
    ee6d_w_pos: float = 1.0
    ee6d_w_rot: float = 1.0
    ee6d_w_grip: float = 1.0
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
        # Phase A (Bridge form match): action-head input ``x`` is zeros, matching
        # VLA-Adapter reference (``action_heads.py:71`` ``cond_actions_hidden_states
        # = torch.zeros(...)``). The previous LastAction-projection of
        # ``batch["last_action_chunk"]`` was our extension that conflicted with
        # Bridge dynamics — it gave the action head a strong residual through
        # ``x`` that the cross-attention to image streams could not compete with.
        # ``last_action_chunk`` is still produced by the dataset (kept for
        # potential Phase B reinstatement) but is ignored at the model level.
        self.action_decoder = DomainAwareLinear(D, A, cfg.num_domains)

        if cfg.use_soft_prompt:
            self.soft_prompt_hub = SoftPromptHub(cfg.num_domains, cfg.num_soft_prompt_tokens, D)
        else:
            self.soft_prompt_hub = None
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

        # Reference layout: [BOS][prompt][scene][PROPRIO 1][action][EOS].
        # Soft prompts and wrist do NOT enter the LLM input — they feed the
        # action head's self-attn pool directly via h_w (= wrist_e) and h_sp
        # (= soft_e). Earlier we scattered them into the LLM, which (a)
        # distorted RoPE positions for the prompt and (b) wasted attention
        # budget on tokens the head was already going to consume separately.
        self.input_packer = InputPacker(
            cfg.bos_id, cfg.eos_id, cfg.prompt_max_len,
            num_scene_tokens=cfg.num_scene_tokens,
            num_action_queries=cfg.num_action_queries,
        )

        # h_t feeds the head from scene LLM-positions only (line 207 slice via
        # packed.idx["scene"]). Wrist + soft prompt enter the head via the
        # self-attn pool concat (h_w / h_sp), not as task tokens. So
        # num_task_tokens = num_scene_tokens.
        self.action_head = L1RegressionActionHead(
            hidden_dim=D,
            action_dim=A,
            num_action_chunks=cfg.action_chunk_len,
            num_blocks=cfg.num_blocks,
            num_task_tokens=cfg.num_scene_tokens,
            use_grad_checkpoint=cfg.use_grad_checkpoint,
            use_wrist_bridge=cfg.use_wrist_bridge,
            gating_init=cfg.gating_init,
            gating_init_wrist=cfg.gating_init_wrist,
            ungated_streams=cfg.ungated_streams,
        )

        # Wrist bridge projector: single Linear(siglip_dim → llm_dim) shared
        # across SigLIP layers. Matches vla-gemma-4
        # ``modeling_prismatic_gemma4.py:264``: "vision_projector と同じ
        # 2-layer MLP ではなく、単純な Linear で llm_dim に射影 (層ごとに
        # feature distribution が違うため共有は粗いが、MVP として許容)".
        # h_w_bridge shape after this projector: (B, num_blocks+1, 256, D).
        if cfg.use_wrist_bridge:
            self.wrist_projector_bridge = nn.Linear(cfg.siglip_hidden_dim, D)
        else:
            self.wrist_projector_bridge = None

    def _pool_wrist(self, wrist_tok: torch.Tensor) -> torch.Tensor:
        """Spatially average-pool a (B, N, D) wrist token sequence.

        Assumes ``N`` is a perfect square (16x16 = 256 for SigLIP@224 /
        patch14). Pools to a grid of side ``sqrt(self._effective_num_wrist)``.

        Fast path (integer downsample factor, e.g. 16 -> 8 = factor 2):
        ``view + mean`` over the per-block dims. Avoids transpose+contiguous
        and uses a clean 2x2 (or kxk) mean reduction instead of the
        adaptive_avg_pool2d kernel — measurably faster at bs=1 on A100.

        Slow path (non-integer factor, e.g. 16 -> 7): fall back to
        ``adaptive_avg_pool2d`` with overlapping windows.
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
        if side == pooled_side:
            return wrist_tok  # no-op
        if side % pooled_side == 0:
            # Integer-factor fast path: reshape and reduce.
            f = side // pooled_side
            g = wrist_tok.view(B, pooled_side, f, pooled_side, f, D)
            return g.mean(dim=(2, 4)).reshape(B, pooled_side * pooled_side, D)
        # Non-integer factor: adaptive pool with channel-first layout.
        grid = wrist_tok.transpose(1, 2).reshape(B, D, side, side)
        pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (pooled_side, pooled_side))
        return pooled.reshape(B, D, pooled_side * pooled_side).transpose(1, 2)

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

        # 2b. Wrist bridge: per-layer SigLIP wrist features projected to LLM
        # dim, fed to the action head's 4th cross-attn branch per block.
        # NUM_BRIDGE_LAYERS = num_blocks + 1 (block i sees layer i+1, so
        # we need indices 0..num_blocks). HF SigLIP returns 28 hidden states
        # for so400m-patch14-224 (1 embedding + 27 blocks); we take the
        # first num_blocks+1 of them. The wrist_pool path is incompatible
        # with wrist_bridge (token count mismatch); guard against it.
        h_w_bridge = None
        if self.cfg.use_wrist_bridge:
            if self.cfg.use_wrist_pool:
                raise RuntimeError(
                    "use_wrist_bridge=True is incompatible with use_wrist_pool=True; "
                    "wrist_bridge expects raw 256-token SigLIP per-layer features"
                )
            num_bridge_layers = self.cfg.num_blocks + 1
            wrist_layers = self.vision_encoder.forward_all_layers(
                batch["wrist_image"], num_layers=num_bridge_layers
            )  # [B, num_bridge_layers, 256, D_vis]
            h_w_bridge = self.wrist_projector_bridge(wrist_layers)  # [B, ..., 256, D]

        # 3. Soft prompts (per-domain) and action queries (shared, broadcast)
        soft_e = self.soft_prompt_hub(domain_id) if self.soft_prompt_hub is not None else None
        action_q_e = self.action_query_hub(B)

        # 4. Build input_ids + indices
        packed = self.input_packer(batch["prompt_input_ids"], batch["prompt_attention_mask"])

        # 5. Gemma forward with overwrite. Project module outputs to the LLM's
        # dtype (defensively — when the whole policy is cast to bf16 these are
        # already bf16 and the .to() is a no-op).
        raw_e = self.gemma.embed_tokens(packed.input_ids)
        llm_dtype = raw_e.dtype
        # Only scatter scene + action queries: prompt is real text tokens
        # (already embedded by embed_tokens), proprio is a placeholder whose
        # LLM hidden state is unused, soft / wrist are not in the LLM input.
        emb = scatter_into_embeds(raw_e, packed.idx["scene"], scene_e.to(llm_dtype))
        emb = scatter_into_embeds(emb, packed.idx["action"], action_q_e.to(llm_dtype))

        out = self.gemma(
            input_ids=packed.input_ids,
            attention_mask=packed.attention_mask,
            inputs_embeds=emb,
        )
        hs = out.hidden_states  # [B, layers+1, L, D] in llm_dtype

        # 6. Slice the per-stream hidden states the action head needs.
        # Bridge form (``action_heads.py:133-176``) splits inputs into four
        # streams; semantically:
        #   - h_a (per-layer, action positions)  cross-attn adapter (LLM-processed)
        #   - h_t (per-layer, scene positions)   cross-attn task    (LLM-processed)
        #   - h_w (un-LLM-processed wrist)       self-attn pool concat
        #   - h_sp (un-LLM-processed soft prompt) self-attn pool concat
        # The reference deliberately routes wrist + soft prompt OUTSIDE the
        # LLM (separate ResNet18 wrist encoder + nn.Embedding soft-prompt
        # library) so the head receives a fresh signal that has not been
        # laundered through 35 layers of LLM self-attn. We replicate the
        # semantic by reusing the modules already in this class:
        #   h_w  = self.wrist_proj(SigLIP(wrist_image))   = ``wrist_e``
        #   h_sp = self.soft_prompt_hub(domain_id)        = ``soft_e``
        # Both were already computed for the LLM scatter on lines above, so we
        # just hold a reference. The wrist/soft tokens still appear in the LLM
        # input (so language can attend to them), but the head's self-attn
        # pool concat draws from the pre-LLM source — a faithful Phase A
        # Bridge match that does not require a separate CNN wrist encoder.
        # Verified by code review against
        # vla-gemma-4/.../modeling_prismatic_gemma4.py:541-630.
        bs = torch.arange(B, device=hs.device).view(B, 1, 1)
        layers = torch.arange(hs.shape[1], device=hs.device).view(1, hs.shape[1], 1)
        h_a = hs[bs, layers, packed.idx["action"].unsqueeze(1)]      # [B, layers+1, Q, D]
        h_t = hs[bs, layers, packed.idx["scene"].unsqueeze(1)]       # [B, layers+1, K_scene, D]
        h_w = wrist_e                                                # [B, K_wrist, D]
        h_sp = soft_e                                                # [B, K_soft, D] or None

        # 7. x init = zeros (Bridge match) + train-time gaussian noise.
        # Reference (action_heads.py:14-17, 80-83) adds N(0, 0.02²) noise
        # to the zero-initialized action positions during Training only.
        # Implemented in ref as a fresh nn.Parameter per call (not registered),
        # which is functionally identical to additive iid gaussian noise.
        # Without it, the action positions go through fc1+LN+ReLU as exact
        # zeros, and q/k/v projections at action positions get only weak
        # cross-attn gradient signal in early training (symmetry-breaking
        # has to come purely from RoPE position offsets).
        A = cfg.action_dim
        D = cfg.hidden_dim
        x_init = torch.zeros(
            B, cfg.action_chunk_len, A * D, device=hs.device, dtype=hs.dtype
        )
        if self.training:
            x_init = x_init + 0.02 * torch.randn_like(x_init)

        # 8. proprio -> p (matches policy dtype).
        p = self.proprio_proj(batch["proprio"], domain_id).unsqueeze(1)

        # 9. action head (policy dtype throughout). h_w + h_sp join the
        # self-attn pool (concat to x post-fc1, trimmed back after blocks).
        # When use_wrist_bridge is on, h_w is dropped from the self-attn
        # pool (handled inside MLPResNet.forward) and h_w_bridge supplies
        # per-layer wrist cross-attn instead.
        if h_w_bridge is not None:
            h_w_bridge = h_w_bridge.to(hs.dtype)
        head_out = self.action_head(
            x_init, h_a=h_a, h_t=h_t, p=p,
            h_w=h_w, h_sp=h_sp, h_w_bridge=h_w_bridge,
        )                                                            # [B, T, D]

        # 10. action decoder.
        pred = self.action_decoder(head_out, domain_id)              # [B, T, A]

        # 11. loss
        target_a = batch["target_action"]
        amask = batch["action_mask"]
        if cfg.loss_type == "l1":
            loss = masked_l1(pred, target_a, amask)
        elif cfg.loss_type == "huber":
            loss = masked_huber(pred, target_a, amask, beta=cfg.huber_beta)
        elif cfg.loss_type == "ee6d":
            comps = ee6d_loss_components(pred, target_a, amask)
            loss = (
                cfg.ee6d_w_pos * comps["pos"]
                + cfg.ee6d_w_rot * comps["rot"]
                + cfg.ee6d_w_grip * comps["grip"]
            )
            # Expose per-channel components so the trainer can log them to
            # wandb without changing forward()'s (pred, loss) contract.
            self._last_loss_info = {f"train/loss/{k}": v.detach() for k, v in comps.items()}
        else:
            raise ValueError(f"unknown loss_type: {cfg.loss_type}")

        return pred, loss
