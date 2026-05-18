from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from vla_project.data import constants as C
from vla_project.data.packing.input_packer import InputPacker
from vla_project.models.action_heads.l1_regression_action_head import L1RegressionActionHead
from vla_project.models.language.embed_overwrite import scatter_into_embeds
from vla_project.models.projectors.action_queries import ActionQueryHub
from vla_project.models.projectors.domain_aware_linear import DomainAwareLinear
from vla_project.models.projectors.soft_prompts import SoftPromptHub
from vla_project.training.losses import (
    masked_l1, masked_huber, masked_l1_per_sample, masked_huber_per_sample,
)
from vla_project.training.losses_ee6d import ee6d_loss_components


class PassthroughActionDecoder(nn.Module):
    """Action decoder shim for baseline-compatible heads that already emit actions."""

    def forward(self, x: torch.Tensor, domain_id=None) -> torch.Tensor:
        return x


@dataclass
class VLAPolicyConfig:
    num_domains: int
    compat_profile: str = "x_vla_adapter"
    hidden_dim: int = C.LLM_HIDDEN_DIM
    siglip_hidden_dim: int = C.SIGLIP_HIDDEN_DIM
    action_dim: int = C.ACTION_DIM
    action_chunk_len: int = C.ACTION_CHUNK_LEN
    proprio_dim: int = C.PROPRIO_DIM
    prompt_max_len: int = C.DEFAULT_PROMPT_MAX_LEN
    num_blocks: int = C.NUM_LLM_LAYERS
    # Which Gemma hidden layers each action-head block reads. Hidden state 0
    # is the embedding output; selectable transformer layer ids are 1..35.
    # Baseline uses ``first_n``: block i reads layer i+1 (1..24 for v25).
    # ``even`` keeps the action head shallow while sampling across the full
    # Gemma depth, e.g. 24 blocks over 35 layers.
    action_head_layer_mode: str = "first_n"  # first_n | even | last_n | custom
    action_head_layer_indices: Tuple[int, ...] = field(default_factory=tuple)
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
    # v42 (2026-05-11): when True, MLPResNetBlock_Pro.forward becomes
    # ``x + self.ffn(attn_out + x)`` (proper Pre-LN residual stream) instead
    # of the legacy ``self.ffn(attn_out + x)``. The legacy form is what
    # VLA-Adapter upstream uses (action_heads.py:409) and is the structural
    # root cause of "only the deepest action_head block trains" pattern in
    # v33/v37/v39/v41 — each block fully overwrites x, so shallow block
    # contributions never reach the output. X-VLA upstream uses the proper
    # residual form (transformer.py:279-280). Backward compatible: legacy
    # ckpts load and forward unchanged when flag is False (default).
    use_proper_residual: bool = False
    # v44 (2026-05-11): additional knobs to make ``use_proper_residual`` actually
    # produce a proper Pre-LN residual stream. Codex round 8 found the v42 path
    # (legacy ffn = LN→Linear→ReLU + residual) was structurally biased — ReLU at
    # the end forces non-negative residual contributions, x drifts monotonically
    # → grad_max 221k bursts.
    #   - proper_ffn_mode:
    #       "legacy" (default): LN → Linear → ReLU  (one-sided, v42 bug)
    #       "linear_only":      LN → Linear        (signed, no activation)
    #       "proper_mlp":       LN → Linear → GELU → Linear  (X-VLA upstream style)
    #   - layer_scale_init: when > 0, allocate per-channel γ Parameter initialized
    #     to this value and apply ``x + γ * ffn(...)``. CaiT/timm convention,
    #     1e-4 keeps the branch ~identity at init.
    #   - mlp_ratio: hidden expansion for proper_mlp (mid = dim * mlp_ratio).
    #     1.0 keeps param count minimal, 4.0 matches X-VLA Mlp default.
    proper_ffn_mode: str = "legacy"
    layer_scale_init: float = 0.0
    mlp_ratio: float = 1.0
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
    # Multi-domain training: when True, log ``train/loss_by_domain/<id>`` for
    # each domain present in the batch. Default False (no per-step compute /
    # log overhead). Diagnostic for catching per-domain collapse that
    # aggregate ``train/loss`` would hide. Currently supports L1 / Huber loss
    # types; ee6d falls back to its existing channel split.
    log_per_domain_loss: bool = False
    use_grad_checkpoint: bool = False
    # Swap scene_proj / proprio_proj from DomainAwareLinear to vla-gemma-4
    # baseline-equivalent MLPs. Required to match the 73% wristb_b16_v2
    # baseline's structural capacity:
    #   - scene_proj : 3-MLP 1152→8192→1536→1536 (GELU×2, ~32 M params)
    #   - proprio_proj: 2-MLP 8→1536→1536        (GELU×1)
    # Default DA-Linear is single-Linear w/ no activation (~1.8 M scene params
    # at num_domains=1) which lacks the capacity to learn a strong
    # SigLIP→Gemma alignment. Only valid with num_domains == 1 (the baseline
    # MLPs are not domain-aware).
    use_baseline_projectors: bool = False
    baseline_scene_init_proj_dim: int = 8192
    # Mode B (matches vla-gemma-4 ``training_mode='speed'`` exactly):
    #   - action_query_hub.queries: requires_grad=False (stays zero)
    #   - LLM forward wrapped in ``torch.no_grad()`` so activations are not
    #     stored for backward
    # The baseline runs Mode B at bs=16 on a 40 GB A100 because the no_grad
    # wrap drops Gemma4-E2B's activation memory (~9 GB at bs=16 with grad
    # checkpoint off). Without this, our trainer fits only bs=8 on 40 GB.
    # vision_projector / scene_proj / proprio_projector / wrist_projector_bridge
    # are still trainable; their gradient flows in via the index_put scatter
    # into ``inputs_embeds`` (PyTorch records this as a differentiable op
    # even though the LLM call itself is detached).
    freeze_llm_and_aq: bool = False
    # Granular successors to ``freeze_llm_and_aq``. When ``freeze_llm_and_aq``
    # is True, ``__post_init__`` forces both of these to True for backwards
    # compatibility. Set them directly to mix-and-match (e.g. LoRA-on-LLM with
    # AQ trainable: both False, plus a non-empty ``model.lora`` cfg).
    #   - ``freeze_action_queries=True``: action_query_hub.queries.requires_grad
    #     stays False (zero-init, never updated).
    #   - ``wrap_llm_in_no_grad=True``: the gemma forward call is wrapped in
    #     ``torch.no_grad()`` (saves ~9 GB activations at bs=16, but BLOCKS
    #     all gradient flow through the LLM — incompatible with LoRA).
    freeze_action_queries: bool = False
    wrap_llm_in_no_grad: bool = False
    # Wrist-only SigLIP LoRA: when set, ``VLAPolicy`` builds a SECOND SigLIP
    # encoder (deepcopy of ``vision_encoder``) wrapped with peft LoRA and
    # routes wrist images through it; scene continues through the original
    # frozen ``vision_encoder``. Idea: SigLIP wasn't pretrained on close-up
    # robot wrist views; let LoRA adapt the wrist forward path while keeping
    # the scene path identical to v25/v28 baseline.
    # Schema: ``{"r": int, "alpha": int, "target_modules": list[str], "dropout": float}``.
    # ``target_modules`` matches timm ViT layer names — e.g. ``["qkv"]`` for the
    # combined attention projection, or ``["qkv", "fc1", "fc2"]`` for full LoRA.
    wrist_siglip_lora: Optional[Dict[str, Any]] = None
    # Wrist-only frozen DINOv2 auxiliary stream. This does not enter the LLM;
    # it adds a projected dense wrist residual to ``h_w_bridge`` for the action
    # head. Intended for spatial tasks where DINO geometry may complement
    # SigLIP's language-grounded features while keeping the v25 scene/LLM path
    # intact.
    use_wrist_dinov2: bool = False
    wrist_dinov2_model_name: str = "facebook/dinov2-base"
    wrist_dinov2_hidden_dim: int = 768
    wrist_dinov2_num_tokens: int = C.NUM_SCENE_TOKENS
    wrist_dinov2_gate_init: float = 0.1
    # Next-architecture experiment: encode both scene and wrist with the same
    # frozen DINOv2 encoder, concatenate each stream's SigLIP+DINO patch tokens,
    # project them with a shared projector, and insert [scene; wrist] into the
    # LLM vision-token slots. This deliberately breaks v25 baseline layout.
    use_scene_wrist_dinov2_llm: bool = False
    # Vision placeholder scheme inside the LLM input_ids:
    #   "image_token"  : repeat IMAGE_SOFT_TOKEN_ID (258880) ``num_scene_tokens``
    #                    times. Default for our v15+ training.
    #   "unused_range" : 256 distinct ``<unused>`` IDs (258949..). Required to
    #                    match the vla-gemma-4 baseline ckpt's PLE pattern at
    #                    eval time (Gemma4 PLE injection depends on token IDs
    #                    even when ``inputs_embeds`` is provided).
    vision_placeholder_mode: str = "image_token"
    # v33+ knobs for X-VLA-style multi-domain experiments:
    #   - ``proj_arch``: shape of scene_proj / wrist_proj / proprio_proj /
    #     action_decoder. Mutually exclusive with use_baseline_projectors.
    #     - "da_linear":   1-layer DomainAwareLinear (X-VLA upstream convention)
    #     - "da_2layer":   2-layer DomainAwareTwoLayerMLP with GELU between
    #                      (per-domain × 2-layer, X-VLA style with extra capacity)
    #     - "shared_3mlp": single 3-layer MLP shared across domains
    #                      (= use_baseline_projectors=True path)
    #     - "shared_da_linear": single 1-layer DA-Linear (legacy GEM-4-VLA
    #                          path; default for backward compat)
    #   - ``proj_hidden_dim``: hidden width for da_2layer (default = hidden_dim)
    #   - ``soft_prompt_in_llm``: when True, soft_prompt_hub output is scattered
    #     into the LLM input embeddings at a reserved placeholder block
    #     (prefix-tuning style). When False, the soft prompt feeds the action
    #     head's self-attn pool via h_sp (current default).
    #   - ``prompt_position``: where the natural-language instruction tokens go
    #     relative to the vision placeholder block in the LLM input. v25
    #     baseline = "before_vision"; v33 experiment uses "after_vision".
    #   - ``include_proprio_placeholder``: keep the 1-token PROPRIO_PLACEHOLDER
    #     in the LLM input (for v25-v32 backward compat). v33 sets False to
    #     drop the dead slot (proprio bypasses LLM via proprio_proj).
    proj_arch: str = "shared_da_linear"
    proj_hidden_dim: int = 0  # 0 = use hidden_dim
    soft_prompt_in_llm: bool = False
    prompt_position: str = "before_vision"
    include_proprio_placeholder: bool = True
    # v36+ wrist-into-LLM design (matches π₀ "fixed slot + mask" pattern):
    #   - wrist_in_llm=True: wrist SigLIP features (or zeros when masked) enter
    #     the LLM at a reserved 256-token slot AFTER the language prompt. The
    #     action head's wrist_bridge is dropped (use_wrist_bridge must be False).
    #     Action head's task stream now concatenates scene+wrist hidden states
    #     so the model can attend to wrist features via LLM-encoded context.
    #   - wrist_view_dropout_p: in training, with this prob the wrist slot is
    #     forced to zeros even if a real wrist image is present (modality
    #     dropout, π₀.7 style). Trains the policy to be robust when wrist is
    #     missing at deployment. 0 = disabled.
    # When wrist_in_llm=True, batches must include a ``wrist_mask`` (B,) bool
    # tensor; rows where mask=False have their wrist embeddings zeroed out.
    # This keeps sequence length fixed across train/eval and across datasets
    # with vs without wrist cameras (RoPE positions stay stable).
    wrist_in_llm: bool = False
    wrist_view_dropout_p: float = 0.0
    # ``proprio_in_llm=True`` (v41+): scatter proprio_proj output into the
    # reserved PROPRIO_PLACEHOLDER slot in the LLM input embeddings, and pass
    # ``p=None`` to the action_head (so the adapter bank inside each block is
    # just ``h_a`` — LLM hidden states at action positions, which now also
    # encode proprio). This forces all observation streams (scene, wrist,
    # proprio, prompt, soft_prompt, action_query) to flow through Gemma
    # uniformly, removing the proprio shortcut that allowed v33 to bypass
    # the LLM entirely (which produced 94% on LIBERO single-domain but
    # blocked cross-embodiment learning since LLM never had to encode state).
    # Forces ``include_proprio_placeholder=True`` (the slot must exist in the
    # input layout for the scatter target).
    proprio_in_llm: bool = False
    # v47 (arch_v2): slice the prompt token positions out of Gemma's per-layer
    # hidden states and concat them into ``h_t`` so prompt features feed the
    # action_head's task cross-attn directly. Without this, language only
    # influences the head via Gemma's internal attention mixing it into scene/
    # wrist hidden states — which the v45 occlusion test showed has near-zero
    # effective contribution. Padded prompt positions are gated out via
    # ``h_t_mask`` (built from prompt_attention_mask) inside the block so they
    # don't inject Gemma's pad-position hidden state (undefined/noisy) into
    # k_task / v_task softmax.
    prompt_in_task_stream: bool = False
    # arch v3: slice proprio's per-layer LLM hidden into ``h_t`` (instead of
    # silently discarding it under proprio_in_llm=True). Requires
    # proprio_in_llm=True so the proprio_proj output is actually present in
    # the LLM input at the proprio placeholder slot.
    proprio_in_task_stream: bool = False
    # arch v3: feed soft_prompt's per-layer LLM hidden into a NEW independent
    # cross-attn stream (k_soft_prompt / v_soft_prompt added per block).
    # Mirrors the action_queries pattern (AQ -> Gemma -> h_a -> adapter
    # cross-attn). Replaces the legacy h_sp self-attn-pool concat path.
    # Requires soft_prompt_in_llm=True + use_soft_prompt=True.
    soft_prompt_as_cross_attn_stream: bool = False
    # arch v3 invariant: when False, the legacy ``h_w`` / ``h_sp`` self-attn
    # pool concat path inside MLPResNet is forcibly disabled (vla_policy
    # always passes h_w=None / h_sp=None). Set True (default) for v25/v33/v45
    # backward compat; set False for arch v3 fresh pretrain configs to enforce
    # "self-attn pool = x only" and route all external memory through cross-
    # attn streams (h_a / h_t / h_sp_per_layer).
    legacy_external_in_self_pool: bool = True

    def __post_init__(self) -> None:
        # Backwards-compat: legacy ``freeze_llm_and_aq=True`` activates both
        # new granular flags. New configs should set the granular flags
        # directly; this branch keeps existing v25 / vla_gemma4_baseline configs
        # working unchanged.
        if self.freeze_llm_and_aq:
            self.freeze_action_queries = True
            self.wrap_llm_in_no_grad = True
        if isinstance(self.action_head_layer_indices, list):
            self.action_head_layer_indices = tuple(int(x) for x in self.action_head_layer_indices)
        if self.compat_profile not in ("x_vla_adapter", "vla_gemma4_baseline"):
            raise ValueError(
                "compat_profile must be 'x_vla_adapter' or 'vla_gemma4_baseline'; "
                f"got {self.compat_profile!r}"
            )
        self.resolve_action_head_layer_indices()
        if self.vision_placeholder_mode not in ("image_token", "unused_range"):
            raise ValueError(
                "vision_placeholder_mode must be 'image_token' or 'unused_range'; "
                f"got {self.vision_placeholder_mode!r}"
            )
        if self.use_baseline_projectors and self.num_domains != 1:
            raise ValueError(
                "use_baseline_projectors=True requires num_domains == 1; "
                f"got num_domains={self.num_domains}"
            )
        if self.use_wrist_bridge and self.use_wrist_pool:
            raise ValueError(
                "use_wrist_bridge=True is incompatible with use_wrist_pool=True; "
                "wrist_bridge expects raw per-layer SigLIP tokens"
            )
        if self.use_wrist_dinov2 and not self.use_wrist_bridge:
            raise ValueError("use_wrist_dinov2=True currently requires use_wrist_bridge=True")
        if self.use_wrist_dinov2 and self.wrist_dinov2_num_tokens != self.num_wrist_tokens:
            raise ValueError(
                "use_wrist_dinov2=True requires wrist_dinov2_num_tokens == num_wrist_tokens "
                f"for bridge residual addition; got {self.wrist_dinov2_num_tokens} and "
                f"{self.num_wrist_tokens}"
            )
        if self.use_scene_wrist_dinov2_llm and self.wrist_dinov2_num_tokens != self.num_scene_tokens:
            raise ValueError(
                "use_scene_wrist_dinov2_llm=True requires DINO token count to match "
                f"scene token count; got {self.wrist_dinov2_num_tokens} and {self.num_scene_tokens}"
            )
        if self.proj_arch not in ("shared_3mlp", "shared_da_linear", "da_linear", "da_2layer"):
            raise ValueError(
                "proj_arch must be one of {'shared_3mlp','shared_da_linear','da_linear','da_2layer'}; "
                f"got {self.proj_arch!r}"
            )
        if self.prompt_position not in ("before_vision", "after_vision"):
            raise ValueError(
                f"prompt_position must be 'before_vision' or 'after_vision'; got {self.prompt_position!r}"
            )
        if self.soft_prompt_in_llm and not self.use_soft_prompt:
            raise ValueError(
                "soft_prompt_in_llm=True requires use_soft_prompt=True (the soft_prompt_hub must exist)"
            )
        if self.use_baseline_projectors and self.proj_arch != "shared_3mlp":
            # Backward compat: if use_baseline_projectors=True, force proj_arch='shared_3mlp'
            self.proj_arch = "shared_3mlp"
        if self.wrist_in_llm and self.use_wrist_bridge:
            raise ValueError(
                "wrist_in_llm=True is incompatible with use_wrist_bridge=True; "
                "the wrist features can only enter via one path."
            )
        if self.wrist_in_llm and self.use_scene_wrist_dinov2_llm:
            raise ValueError(
                "wrist_in_llm=True is incompatible with use_scene_wrist_dinov2_llm=True; "
                "use one or the other for wrist→LLM injection."
            )
        if self.proper_ffn_mode not in ("legacy", "linear_only", "proper_mlp"):
            raise ValueError(
                "proper_ffn_mode must be 'legacy' / 'linear_only' / 'proper_mlp'; "
                "got {!r}".format(self.proper_ffn_mode)
            )
        if self.proper_ffn_mode != "legacy" and not self.use_proper_residual:
            raise ValueError(
                "proper_ffn_mode={!r} only makes sense with use_proper_residual=True; "
                "the legacy non-residual path always uses LN→Linear→ReLU".format(self.proper_ffn_mode)
            )
        if self.layer_scale_init < 0.0:
            raise ValueError(
                "layer_scale_init must be >= 0; got {}".format(self.layer_scale_init)
            )
        if self.layer_scale_init > 0.0 and not self.use_proper_residual:
            raise ValueError(
                "layer_scale_init > 0 requires use_proper_residual=True (the legacy "
                "path doesn't apply a residual branch to scale)."
            )
        # arch v3 invariants (per-flag + holistic)
        if self.proprio_in_task_stream and not self.proprio_in_llm:
            raise ValueError(
                "proprio_in_task_stream=True requires proprio_in_llm=True; the "
                "proprio_proj output must be present in the LLM input at the "
                "proprio placeholder slot before its hidden state can be sliced."
            )
        if self.soft_prompt_as_cross_attn_stream and not self.soft_prompt_in_llm:
            raise ValueError(
                "soft_prompt_as_cross_attn_stream=True requires soft_prompt_in_llm=True; "
                "the soft_prompt_hub output must be scattered into the LLM input "
                "for its per-layer hidden to be sliced into h_sp_per_layer."
            )
        if self.soft_prompt_as_cross_attn_stream and not self.use_soft_prompt:
            raise ValueError(
                "soft_prompt_as_cross_attn_stream=True requires use_soft_prompt=True"
            )
        if self.soft_prompt_as_cross_attn_stream and self.num_soft_prompt_tokens <= 0:
            raise ValueError(
                "soft_prompt_as_cross_attn_stream=True requires num_soft_prompt_tokens > 0; "
                f"got {self.num_soft_prompt_tokens}"
            )
        if self.soft_prompt_as_cross_attn_stream and self.wrap_llm_in_no_grad:
            raise ValueError(
                "soft_prompt_as_cross_attn_stream=True with wrap_llm_in_no_grad=True "
                "detaches Gemma hidden states, so soft_prompt_hub (and proprio_proj / "
                "scene_proj / wrist_proj) cannot receive gradient via the cross-attn "
                "memory path. Set wrap_llm_in_no_grad=False for arch v3."
            )
        # arch v3 holistic invariant: when legacy_external_in_self_pool=False
        # (the new "self-attn pool = x only" mode), all memory streams must flow
        # through cross-attn, which requires every external token to be inside
        # the LLM AND sliced back out. Enforce the full coherent set.
        if not self.legacy_external_in_self_pool:
            required = {
                "wrist_in_llm": self.wrist_in_llm,
                "proprio_in_llm": self.proprio_in_llm,
                "soft_prompt_in_llm": self.soft_prompt_in_llm,
                "proprio_in_task_stream": self.proprio_in_task_stream,
                "prompt_in_task_stream": self.prompt_in_task_stream,
                "soft_prompt_as_cross_attn_stream": self.soft_prompt_as_cross_attn_stream,
            }
            forbidden = {
                "wrap_llm_in_no_grad": self.wrap_llm_in_no_grad,
                "use_wrist_bridge": self.use_wrist_bridge,
            }
            mismatched = [n for n, v in required.items() if not v] + [
                n for n, v in forbidden.items() if v
            ]
            if mismatched:
                raise ValueError(
                    "legacy_external_in_self_pool=False (arch v3) requires: "
                    + ", ".join(f"{n}={'True' if n in required else 'False'}"
                                for n in mismatched)
                    + ". This invariant ensures every LLM-scattered token "
                    "(scene/wrist/proprio/soft_prompt/action_queries/prompt) has "
                    "its Gemma hidden routed back into the action_head via "
                    "h_a / h_t / h_sp_per_layer cross-attn streams."
                )

        if self.proprio_in_llm and not self.include_proprio_placeholder:
            raise ValueError(
                "proprio_in_llm=True requires include_proprio_placeholder=True; "
                "the LLM input layout must reserve a slot for the scatter target."
            )
        if not 0.0 <= self.wrist_view_dropout_p <= 1.0:
            raise ValueError(
                f"wrist_view_dropout_p must be in [0,1]; got {self.wrist_view_dropout_p}"
            )
        if self.use_wrist_bridge and self.num_blocks + 1 > C.SIGLIP_NUM_BLOCKS:
            raise ValueError(
                "use_wrist_bridge=True requires num_blocks + 1 <= "
                f"SIGLIP_NUM_BLOCKS ({C.SIGLIP_NUM_BLOCKS}); got num_blocks={self.num_blocks}. "
                "Disable wrist_bridge or add an explicit layer-mapping policy."
            )
        if self.compat_profile == "vla_gemma4_baseline":
            expected = {
                "num_domains": 1,
                "num_blocks": 24,
                "action_dim": C.ACTION_DIM,
                "action_chunk_len": C.ACTION_CHUNK_LEN,
                "prompt_max_len": C.DEFAULT_PROMPT_MAX_LEN,
                "action_head_layer_mode": "first_n",
                "use_baseline_projectors": True,
                "use_wrist_bridge": True,
                "use_wrist_dinov2": False,
                "use_soft_prompt": False,
                "freeze_llm_and_aq": True,
                "vision_placeholder_mode": "unused_range",
            }
            mismatches = [
                f"{name}={getattr(self, name)!r} (expected {value!r})"
                for name, value in expected.items()
                if getattr(self, name) != value
            ]
            if mismatches:
                raise ValueError(
                    "compat_profile='vla_gemma4_baseline' requires: "
                    + "; ".join(mismatches)
                )

    @property
    def action_head_outputs_actions(self) -> bool:
        """Whether the action head emits A-dim actions instead of hidden states.

        True only for the ``shared_3mlp`` baseline (matches vla-gemma-4
        action_head.model.fc2 (7, 1536)). For DA-Linear / DA-2-MLP paths,
        the head outputs hidden_dim and an external action_decoder maps to A.
        """
        return self.proj_arch == "shared_3mlp"

    def resolve_action_head_layer_indices(
        self, total_layers: int = C.NUM_LLM_LAYERS
    ) -> Tuple[int, ...]:
        """Return transformer layer ids consumed by action-head blocks.

        Returned ids are in hidden-state indexing, excluding embedding id 0.
        Length is exactly ``num_blocks``.
        """
        mode = self.action_head_layer_mode
        if mode == "first_n":
            if self.num_blocks > total_layers:
                raise ValueError(
                    f"first_n requires num_blocks <= {total_layers}; got {self.num_blocks}"
                )
            indices = tuple(range(1, self.num_blocks + 1))
        elif mode == "last_n":
            if self.num_blocks > total_layers:
                raise ValueError(
                    f"last_n requires num_blocks <= {total_layers}; got {self.num_blocks}"
                )
            start = total_layers - self.num_blocks + 1
            indices = tuple(range(start, total_layers + 1))
        elif mode == "even":
            if self.num_blocks <= 0:
                raise ValueError(f"num_blocks must be > 0; got {self.num_blocks}")
            if self.num_blocks == 1:
                indices = (total_layers,)
            else:
                indices = tuple(
                    int(round(1 + i * (total_layers - 1) / (self.num_blocks - 1)))
                    for i in range(self.num_blocks)
                )
        elif mode == "custom":
            indices = tuple(int(x) for x in self.action_head_layer_indices)
            if len(indices) != self.num_blocks:
                raise ValueError(
                    "custom action_head_layer_indices length must equal num_blocks; "
                    f"got len={len(indices)} num_blocks={self.num_blocks}"
                )
        else:
            raise ValueError(
                "action_head_layer_mode must be 'first_n', 'even', 'last_n', or 'custom'; "
                f"got {mode!r}"
            )

        bad = [i for i in indices if i < 1 or i > total_layers]
        if bad:
            raise ValueError(
                f"action_head_layer_indices must be in [1, {total_layers}]; got {bad}"
            )
        if len(set(indices)) != len(indices):
            raise ValueError(f"action_head_layer_indices must be unique; got {indices}")
        return indices


class VLAPolicy(nn.Module):
    def __init__(self, cfg: VLAPolicyConfig, vision_encoder: nn.Module, gemma: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = vision_encoder
        self.gemma = gemma
        self._action_head_layer_indices = cfg.resolve_action_head_layer_indices()
        # Wrist-only SigLIP LoRA: deepcopy the scene encoder and inject peft
        # LoRA into the copy. Base weights remain frozen (already done by
        # SigLIPTimmEncoder.freeze() / SigLIPEncoder.freeze() at construction);
        # peft adds new lora_A / lora_B layers with requires_grad=True. Wrist
        # forward then routes through this copy in self.forward.
        if cfg.wrist_siglip_lora is not None:
            from copy import deepcopy
            self.wrist_vision_encoder: Optional[nn.Module] = deepcopy(vision_encoder)
            from peft import LoraConfig, inject_adapter_in_model
            wlora = dict(cfg.wrist_siglip_lora)
            inject_target = self.wrist_vision_encoder
            if hasattr(self.wrist_vision_encoder, "backbone"):
                inject_target = self.wrist_vision_encoder.backbone   # timm path
            elif hasattr(self.wrist_vision_encoder, "model"):
                inject_target = self.wrist_vision_encoder.model       # HF path
            lcfg = LoraConfig(
                r=int(wlora["r"]),
                lora_alpha=int(wlora.get("alpha", 2 * int(wlora["r"]))),
                target_modules=list(wlora.get("target_modules", ["qkv"])),
                lora_dropout=float(wlora.get("dropout", 0.0)),
                bias="none",
            )
            inject_adapter_in_model(lcfg, inject_target)
        else:
            self.wrist_vision_encoder = None

        D, A = cfg.hidden_dim, cfg.action_dim
        proj_hidden = int(cfg.proj_hidden_dim) if cfg.proj_hidden_dim > 0 else D
        if cfg.proj_arch == "shared_3mlp":
            from vla_project.models.projectors.baseline_projectors import (
                BaselineProprioProjector,
                BaselineSceneProjector,
            )
            self.scene_proj = BaselineSceneProjector(
                cfg.siglip_hidden_dim, D, cfg.baseline_scene_init_proj_dim
            )
            self.scene_wrist_dinov2_llm_proj = (
                BaselineSceneProjector(
                    cfg.siglip_hidden_dim + cfg.wrist_dinov2_hidden_dim,
                    D,
                    cfg.baseline_scene_init_proj_dim,
                )
                if cfg.use_scene_wrist_dinov2_llm
                else None
            )
            self.proprio_proj = BaselineProprioProjector(cfg.proprio_dim, D)
            self.wrist_proj = DomainAwareLinear(cfg.siglip_hidden_dim, D, cfg.num_domains)
        elif cfg.proj_arch == "da_2layer":
            from vla_project.models.projectors.domain_aware_linear import DomainAwareTwoLayerMLP
            self.scene_proj = DomainAwareTwoLayerMLP(
                cfg.siglip_hidden_dim, proj_hidden, D, cfg.num_domains
            )
            self.scene_wrist_dinov2_llm_proj = (
                DomainAwareTwoLayerMLP(
                    cfg.siglip_hidden_dim + cfg.wrist_dinov2_hidden_dim,
                    proj_hidden,
                    D,
                    cfg.num_domains,
                )
                if cfg.use_scene_wrist_dinov2_llm
                else None
            )
            self.proprio_proj = DomainAwareTwoLayerMLP(cfg.proprio_dim, proj_hidden, D, cfg.num_domains)
            self.wrist_proj = DomainAwareTwoLayerMLP(cfg.siglip_hidden_dim, proj_hidden, D, cfg.num_domains)
        else:
            # da_linear / shared_da_linear: 1-layer DA-Linear (X-VLA upstream)
            self.scene_proj = DomainAwareLinear(cfg.siglip_hidden_dim, D, cfg.num_domains)
            self.scene_wrist_dinov2_llm_proj = (
                DomainAwareLinear(
                    cfg.siglip_hidden_dim + cfg.wrist_dinov2_hidden_dim,
                    D,
                    cfg.num_domains,
                )
                if cfg.use_scene_wrist_dinov2_llm
                else None
            )
            self.proprio_proj = DomainAwareLinear(cfg.proprio_dim, D, cfg.num_domains)
            self.wrist_proj = DomainAwareLinear(cfg.siglip_hidden_dim, D, cfg.num_domains)
        # Phase A (Bridge form match): action-head input ``x`` is zeros, matching
        # VLA-Adapter reference (``action_heads.py:71`` ``cond_actions_hidden_states
        # = torch.zeros(...)``). The previous LastAction-projection of
        # ``batch["last_action_chunk"]`` was our extension that conflicted with
        # Bridge dynamics — it gave the action head a strong residual through
        # ``x`` that the cross-attention to image streams could not compete with.
        # ``last_action_chunk`` is still produced by the dataset (kept for
        # potential Phase B reinstatement) but is ignored at the model level.
        if cfg.action_head_outputs_actions:
            self.action_decoder = PassthroughActionDecoder()
        elif cfg.proj_arch == "da_2layer":
            from vla_project.models.projectors.domain_aware_linear import DomainAwareTwoLayerMLP
            self.action_decoder = DomainAwareTwoLayerMLP(D, proj_hidden, A, cfg.num_domains)
        else:
            self.action_decoder = DomainAwareLinear(D, A, cfg.num_domains)

        if cfg.use_soft_prompt:
            self.soft_prompt_hub = SoftPromptHub(cfg.num_domains, cfg.num_soft_prompt_tokens, D)
        else:
            self.soft_prompt_hub = None
        self.action_query_hub = ActionQueryHub(cfg.num_action_queries, D)  # shared, not per-domain
        if cfg.freeze_action_queries:
            for p in self.action_query_hub.parameters():
                p.requires_grad = False

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
        # ``llm_vision_tokens`` = number of scene-position tokens scattered
        # into the LLM input. wrist tokens (when ``wrist_in_llm=True``) live
        # in their OWN reserved slot AFTER the prompt, so we don't fold them
        # into the scene block here. ``use_scene_wrist_dinov2_llm`` is the
        # legacy v32 path that fused scene+wrist into one 512-token vision
        # block; that path stays for backward compat but is mutually exclusive
        # with ``wrist_in_llm`` (validated in __post_init__).
        llm_vision_tokens = (
            cfg.num_scene_tokens + cfg.num_wrist_tokens
            if cfg.use_scene_wrist_dinov2_llm
            else cfg.num_scene_tokens
        )
        self._llm_vision_tokens = int(llm_vision_tokens)
        self._llm_wrist_tokens = int(cfg.num_wrist_tokens) if cfg.wrist_in_llm else 0
        self.input_packer = InputPacker(
            cfg.bos_id, cfg.eos_id, cfg.prompt_max_len,
            num_scene_tokens=llm_vision_tokens,
            num_action_queries=cfg.num_action_queries,
            vision_placeholder_mode=cfg.vision_placeholder_mode,
            prompt_position=cfg.prompt_position,
            num_soft_prompt_tokens_in_llm=(
                cfg.num_soft_prompt_tokens if cfg.soft_prompt_in_llm else 0
            ),
            include_proprio_placeholder=cfg.include_proprio_placeholder,
            num_wrist_tokens_in_llm=self._llm_wrist_tokens,
        )

        # h_t feeds the head from LLM-positions of scene (+ optional wrist /
        # proprio / prompt depending on arch-v3 flags). Wrist + soft prompt
        # legacy concat (h_w / h_sp self-attn pool) are still counted only by
        # their cross-attn-stream membership: wrist goes into h_t when
        # wrist_in_llm=True, soft_prompt goes into its own independent
        # cross-attn stream (h_sp_per_layer) under arch v3.
        action_head_task_tokens = llm_vision_tokens + self._llm_wrist_tokens
        if cfg.proprio_in_task_stream:
            action_head_task_tokens += 1                              # arch v3 +proprio
        if cfg.prompt_in_task_stream:
            action_head_task_tokens += cfg.prompt_max_len             # arch v3 +prompt (pad masked)
        self.action_head = L1RegressionActionHead(
            hidden_dim=D,
            action_dim=A,
            num_action_chunks=cfg.action_chunk_len,
            num_blocks=cfg.num_blocks,
            num_task_tokens=action_head_task_tokens,
            use_grad_checkpoint=cfg.use_grad_checkpoint,
            use_wrist_bridge=cfg.use_wrist_bridge,
            gating_init=cfg.gating_init,
            gating_init_wrist=cfg.gating_init_wrist,
            ungated_streams=cfg.ungated_streams,
            use_proper_residual=cfg.use_proper_residual,
            proper_ffn_mode=cfg.proper_ffn_mode,
            layer_scale_init=cfg.layer_scale_init,
            mlp_ratio=cfg.mlp_ratio,
            output_action_dim=cfg.action_head_outputs_actions,
            use_soft_prompt_cross_attn=cfg.soft_prompt_as_cross_attn_stream,
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
        if cfg.use_wrist_dinov2 or cfg.use_scene_wrist_dinov2_llm:
            from vla_project.models.vision.dinov2 import DINOv2Encoder

            self.wrist_dinov2_encoder: Optional[nn.Module] = DINOv2Encoder(
                model_name=cfg.wrist_dinov2_model_name,
                hidden_dim=cfg.wrist_dinov2_hidden_dim,
                num_tokens=cfg.wrist_dinov2_num_tokens,
            )
        else:
            self.wrist_dinov2_encoder = None
        if cfg.use_wrist_dinov2:
            self.wrist_dinov2_projector = nn.Linear(cfg.wrist_dinov2_hidden_dim, D)
            self.wrist_dinov2_gate = nn.Parameter(
                torch.tensor(float(cfg.wrist_dinov2_gate_init), dtype=torch.float32)
            )
        else:
            self.wrist_dinov2_projector = None
            self.wrist_dinov2_gate = None

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

        # 1. SigLIP encode.
        scene_tok = self.vision_encoder(batch["scene_image"])  # [B, 256, D_vis]

        # When ``wrist_siglip_lora`` is set, wrist images go through a
        # separate SigLIP encoder (frozen base + LoRA adapters). Scene path
        # always uses the original ``vision_encoder`` (frozen, no LoRA).
        wrist_encoder = (
            self.wrist_vision_encoder if self.wrist_vision_encoder is not None
            else self.vision_encoder
        )
        wrist_e = None
        wrist_tok_for_llm = None
        if not self.cfg.use_wrist_bridge:
            wrist_tok_raw = wrist_encoder(batch["wrist_image"])
            wrist_tok_for_llm = wrist_tok_raw
            wrist_tok = self._pool_wrist(wrist_tok_raw) if self.cfg.use_wrist_pool else wrist_tok_raw
            wrist_e = self.wrist_proj(wrist_tok, domain_id)     # [B, K_wrist, D]

        # 2. Project vision tokens to LLM dim, per domain.
        if self.cfg.use_scene_wrist_dinov2_llm:
            if "scene_image_dinov2" not in batch or "wrist_image_dinov2" not in batch:
                raise KeyError(
                    "batch must include scene_image_dinov2 and wrist_image_dinov2 "
                    "when use_scene_wrist_dinov2_llm=True"
                )
            dino_scene = self.wrist_dinov2_encoder(batch["scene_image_dinov2"])
            dino_wrist_llm = self.wrist_dinov2_encoder(batch["wrist_image_dinov2"])
            wrist_siglip_for_llm = (
                wrist_tok_for_llm
                if wrist_tok_for_llm is not None
                else wrist_encoder(batch["wrist_image"])
            )
            scene_fused = torch.cat([scene_tok, dino_scene.to(scene_tok.dtype)], dim=-1)
            wrist_fused = torch.cat([wrist_siglip_for_llm, dino_wrist_llm.to(scene_tok.dtype)], dim=-1)
            fused_vis = torch.cat([scene_fused, wrist_fused], dim=1)
            scene_e = self.scene_wrist_dinov2_llm_proj(fused_vis, domain_id)
        else:
            scene_e = self.scene_proj(scene_tok, domain_id)        # [B, 256, D]

        # 2b. Wrist bridge: per-layer SigLIP wrist features projected to LLM
        # dim, fed to the action head's 4th cross-attn branch per block.
        # NUM_BRIDGE_LAYERS = num_blocks + 1 (block i sees layer i+1, so
        # we need indices 0..num_blocks). HF SigLIP returns 28 hidden states
        # for so400m-patch14-224 (1 embedding + 27 blocks); wrappers expose
        # block outputs only, so we take the first num_blocks+1 of them.
        h_w_bridge = None
        if self.cfg.use_wrist_bridge:
            num_bridge_layers = self.cfg.num_blocks + 1
            wrist_layers = wrist_encoder.forward_all_layers(
                batch["wrist_image"], num_layers=num_bridge_layers
            )  # [B, num_bridge_layers, 256, D_vis]
            h_w_bridge = self.wrist_projector_bridge(wrist_layers)  # [B, ..., 256, D]
            if self.wrist_dinov2_encoder is not None:
                if "wrist_image_dinov2" not in batch:
                    raise KeyError(
                        "batch is missing 'wrist_image_dinov2' required by "
                        "use_wrist_dinov2=True"
                    )
                dino_wrist = self.wrist_dinov2_encoder(batch["wrist_image_dinov2"])
                dino_wrist = self.wrist_dinov2_projector(dino_wrist)
                gate = torch.tanh(self.wrist_dinov2_gate).to(dino_wrist.dtype)
                h_w_bridge = h_w_bridge + gate * dino_wrist.unsqueeze(1)

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
        # v41: when proprio_in_llm=True, scatter proprio_proj output into the
        # reserved PROPRIO_PLACEHOLDER slot. The action_head will then receive
        # ``p=None`` (see step 8 below) so the adapter bank inside each block
        # is just ``h_a`` — LLM hidden states must encode proprio for action
        # prediction. Builds proprio_e here (before LLM forward) instead of
        # the legacy step-8 location to keep the scatter targets co-located.
        proprio_e = None
        if self.cfg.proprio_in_llm and "proprio" in packed.idx:
            proprio_e = self.proprio_proj(batch["proprio"], domain_id).unsqueeze(1)
            emb = scatter_into_embeds(emb, packed.idx["proprio"], proprio_e.to(llm_dtype))
        # v33: when soft_prompt_in_llm=True, scatter the soft prompt embeddings
        # into the LLM input at the reserved soft_prompt placeholder block.
        # The soft prompt is no longer concatenated into the action_head's
        # self-attn pool (h_sp is overridden to None below).
        if self.cfg.soft_prompt_in_llm and soft_e is not None and "soft_prompt" in packed.idx:
            emb = scatter_into_embeds(emb, packed.idx["soft_prompt"], soft_e.to(llm_dtype))
        # v36: when wrist_in_llm=True, project wrist features and scatter into
        # the LLM at the wrist slot. ``wrist_mask`` (B,) gates per-sample
        # whether the slot carries real features (mask=True) or zeros
        # (mask=False — π₀ "missing view" convention). View dropout
        # (probability ``wrist_view_dropout_p``) randomly forces mask=False at
        # train-time so the policy is robust to deploy-time wrist drop.
        if self.cfg.wrist_in_llm and "wrist" in packed.idx:
            wrist_e_for_llm = wrist_e if wrist_e is not None else self.wrist_proj(
                wrist_tok_for_llm if wrist_tok_for_llm is not None else wrist_encoder(batch["wrist_image"]),
                domain_id,
            )
            mask = batch.get("wrist_mask", None)
            if mask is None:
                mask = torch.ones(B, dtype=torch.bool, device=wrist_e_for_llm.device)
            else:
                mask = mask.to(device=wrist_e_for_llm.device).bool()
            if self.training and self.cfg.wrist_view_dropout_p > 0.0:
                drop = (
                    torch.rand(B, device=mask.device) < self.cfg.wrist_view_dropout_p
                )
                mask = mask & ~drop
            mask_b = mask.view(B, 1, 1).to(wrist_e_for_llm.dtype)
            wrist_e_masked = wrist_e_for_llm * mask_b
            emb = scatter_into_embeds(emb, packed.idx["wrist"], wrist_e_masked.to(llm_dtype))

        # Match vla-gemma-4 73% baseline: pass all-ones attention_mask. The
        # baseline ignores prompt padding (modeling_prismatic_gemma4.py:634
        # `attention_mask = torch.ones(B, L_total, ...)`). Using the
        # `packed.attention_mask` we built (with 0 for padded prompt
        # positions) changes Gemma4's attention pattern from what its
        # pretrained weights were learned with — we do NOT want that.
        L_total = packed.input_ids.shape[1]
        all_ones_attn = torch.ones(
            B, L_total, dtype=packed.attention_mask.dtype, device=packed.input_ids.device
        )
        # Mode B: wrap LLM forward in no_grad so activations aren't stored for
        # backward. Matches vla-gemma-4 training_mode='speed'. scene_proj /
        # proprio_proj / wrist_projector_bridge still receive gradient via
        # the index_put scatter that builds ``emb`` BEFORE this call.
        if cfg.wrap_llm_in_no_grad:
            with torch.no_grad():
                out = self.gemma(
                    input_ids=packed.input_ids,
                    attention_mask=all_ones_attn,
                    inputs_embeds=emb,
                )
        else:
            out = self.gemma(
                input_ids=packed.input_ids,
                attention_mask=all_ones_attn,
                inputs_embeds=emb,
            )
        hs = out.hidden_states  # [B, layers+1, L, D] in llm_dtype

        # 6. Slice LLM-processed scene/action streams. Wrist and soft prompt,
        # when used, enter the action head outside the LLM path.
        layer_idx = torch.tensor(self._action_head_layer_indices, device=hs.device)
        if int(layer_idx.max()) >= hs.shape[1]:
            raise RuntimeError(
                f"Gemma returned {hs.shape[1] - 1} transformer layers, but "
                f"action head requested layer {int(layer_idx.max())}"
            )
        selected_hs = hs.index_select(1, layer_idx)                  # [B, num_blocks, L, D]
        bs = torch.arange(B, device=hs.device).view(B, 1, 1)
        layers = torch.arange(selected_hs.shape[1], device=hs.device).view(
            1, selected_hs.shape[1], 1
        )
        h_a_selected = selected_hs[bs, layers, packed.idx["action"].unsqueeze(1)]
        h_t_selected = selected_hs[bs, layers, packed.idx["scene"].unsqueeze(1)]
        # Per-piece mask for h_t. scene/wrist/proprio are always-valid tokens;
        # prompt is variable-length with prompt_attention_mask. We collect 1.0
        # masks per piece and concat at the end into h_t_mask (B, K_t). The
        # mask is only used when prompt_in_task_stream=True (the only stream
        # piece with real pad); otherwise we pass h_t_mask=None to the head.
        mask_pieces = [torch.ones(B, h_t_selected.shape[2],
                                  dtype=torch.bool, device=hs.device)]
        # v36: when wrist_in_llm=True, the action head's task stream attends to
        # both scene AND wrist hidden states. Concatenate the slices along the
        # token dim so num_task_tokens matches the head's allocated K_t.
        if self.cfg.wrist_in_llm and "wrist" in packed.idx:
            h_w_selected = selected_hs[bs, layers, packed.idx["wrist"].unsqueeze(1)]
            h_t_selected = torch.cat([h_t_selected, h_w_selected], dim=2)
            mask_pieces.append(torch.ones(B, h_w_selected.shape[2],
                                          dtype=torch.bool, device=hs.device))
        # arch v3: slice proprio LLM hidden into h_t (replaces the legacy
        # discarded path under proprio_in_llm=True).
        if self.cfg.proprio_in_task_stream and "proprio" in packed.idx:
            h_pr_selected = selected_hs[bs, layers, packed.idx["proprio"].unsqueeze(1)]
            h_t_selected = torch.cat([h_t_selected, h_pr_selected], dim=2)
            mask_pieces.append(torch.ones(B, h_pr_selected.shape[2],
                                          dtype=torch.bool, device=hs.device))
        # arch v2/v3: slice prompt LLM hidden into h_t with pad mask via h_t_mask.
        # Padded prompt token positions in Gemma's output hidden are masked OUT
        # at the action_head cross-attn softmax (see MLPResNetBlock_Pro.forward).
        if self.cfg.prompt_in_task_stream and "prompt" in packed.idx:
            h_p_selected = selected_hs[bs, layers, packed.idx["prompt"].unsqueeze(1)]
            h_t_selected = torch.cat([h_t_selected, h_p_selected], dim=2)
            mask_pieces.append(batch["prompt_attention_mask"].bool())
        h_t_mask = torch.cat(mask_pieces, dim=1) if self.cfg.prompt_in_task_stream else None
        # arch v3: slice soft_prompt LLM hidden into independent cross-attn
        # stream h_sp_per_layer (AQ pattern). Requires soft_prompt_in_llm=True
        # (so soft_prompt_hub output is actually scattered into the LLM input).
        if self.cfg.soft_prompt_as_cross_attn_stream and "soft_prompt" in packed.idx:
            h_sp_selected = selected_hs[bs, layers, packed.idx["soft_prompt"].unsqueeze(1)]
            h_sp_per_layer = torch.cat([h_sp_selected[:, :1], h_sp_selected], dim=1)
        else:
            h_sp_per_layer = None
        # MLPResNet indexes h_*[:, i + 1] to match the reference's
        # embedding+layers convention, so prepend an unused slot.
        h_a = torch.cat([h_a_selected[:, :1], h_a_selected], dim=1)   # [B, num_blocks+1, Q, D]
        h_t = torch.cat([h_t_selected[:, :1], h_t_selected], dim=1)   # [B, num_blocks+1, K_t, D]
        # legacy h_w / h_sp self-attn pool concat (v25/v33/v45 path). arch v3
        # disables this by forcing both to None via legacy_external_in_self_pool=False.
        if not self.cfg.legacy_external_in_self_pool:
            h_w = None
            h_sp = None
        else:
            # v36: when wrist_in_llm=True, the wrist features are scattered into
            # the LLM emb already. Don't double-feed via the action_head's
            # self-attn pool — set h_w=None.
            h_w = None if self.cfg.wrist_in_llm else wrist_e         # [B, K_wrist, D] or None
            # v33: when soft_prompt_in_llm=True, the soft prompt is consumed by the
            # LLM (scattered into emb above). Don't double-feed it to the action
            # head's self-attn pool.
            h_sp = None if self.cfg.soft_prompt_in_llm else soft_e    # [B, K_soft, D] or None

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

        # 8. proprio -> p (matches policy dtype). When proprio_in_llm=True,
        # proprio was already scattered into the LLM input embedding (step 5)
        # and the action_head's adapter bank should NOT receive it directly
        # (otherwise the model has both routes and the shortcut is preserved).
        # Reuse the proprio_e computed above when available; otherwise build
        # the legacy ``p`` for the action_head direct path.
        if self.cfg.proprio_in_llm:
            p = None
        elif proprio_e is not None:
            p = proprio_e
        else:
            p = self.proprio_proj(batch["proprio"], domain_id).unsqueeze(1)
        # arch v3: legacy_external_in_self_pool=False forces p=None too. The
        # proprio cross-attn signal flows through h_t via proprio_in_task_stream
        # (sliced from Gemma per-layer hidden), so the legacy concat-to-adapter
        # path is unused.
        if not self.cfg.legacy_external_in_self_pool:
            p = None

        # 9. action head (policy dtype throughout). h_w + h_sp join the
        # self-attn pool (concat to x post-fc1, trimmed back after blocks)
        # only under legacy_external_in_self_pool=True (v25/v33/v45). arch v3
        # forces them None above so the self-attn pool stays x-only.
        # When use_wrist_bridge is on, h_w is dropped from the self-attn
        # pool (handled inside MLPResNet.forward) and h_w_bridge supplies
        # per-layer wrist cross-attn instead.
        if h_w_bridge is not None:
            h_w_bridge = h_w_bridge.to(hs.dtype)
        head_out = self.action_head(
            x_init, h_a=h_a, h_t=h_t, p=p,
            h_w=h_w, h_sp=h_sp, h_w_bridge=h_w_bridge,
            h_sp_per_layer=h_sp_per_layer, h_t_mask=h_t_mask,
        )                                                            # [B, T, D]

        # 10. action decoder.
        pred = self.action_decoder(head_out, domain_id)              # [B, T, A]

        # 11. loss
        target_a = batch["target_action"]
        amask = batch["action_mask"]
        # Build a fresh loss-info dict every forward; trainer.py reads it via
        # ``_last_loss_info`` and logs each entry to wandb. Stale entries from
        # the prior batch (e.g. a domain that was sampled then absent next step)
        # would otherwise persist and mislead — codex round 10 #1.
        loss_info: Dict[str, torch.Tensor] = {}
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
            for k, v in comps.items():
                loss_info[f"train/loss/{k}"] = v.detach()
        else:
            raise ValueError(f"unknown loss_type: {cfg.loss_type}")

        # Per-domain L1/Huber loss decomposition (default off). Adds
        # ``train/loss_by_domain/<id>`` for each domain present in the current
        # batch so per-domain collapse can be detected over many steps.
        # ee6d path keeps its existing channel split; per-domain × per-channel
        # is deferred (codex round 10 F).
        if cfg.log_per_domain_loss and cfg.loss_type in ("l1", "huber"):
            if cfg.loss_type == "l1":
                per_sample = masked_l1_per_sample(pred, target_a, amask)
            else:
                per_sample = masked_huber_per_sample(pred, target_a, amask, beta=cfg.huber_beta)
            domain_id_b = batch["domain_id"]                # (B,) long
            for did in range(int(cfg.num_domains)):
                m = (domain_id_b == did)
                if m.any():
                    loss_info[f"train/loss_by_domain/{did}"] = per_sample[m].mean().detach()

        # Single assignment after building so the trainer always sees a
        # consistent snapshot. Empty dict preserved as-is (no-op for trainer).
        self._last_loss_info = loss_info

        return pred, loss
