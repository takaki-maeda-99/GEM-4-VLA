"""ModelRuntime — Phase 1 implementation.

Loads the model and tokenizer/image_transform inside `from_export` so the
inference server can call ``runtime(batch)`` to get a raw forward pass.

Design choice: the model + tokenizer + image_transform all hang off the
runtime because the predictor needs all three to build a batch, and the
runtime is the natural owner of "everything bound to one ckpt". The
existing `XVLAAdapterPolicy` does the same bundling at eval time; we
duplicate the relevant bits here rather than depend on the policy
class, because the policy carries LIBERO-specific gripper transforms
that don't apply to other contracts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch
from huggingface_hub import snapshot_download

from vla_project.data import constants as C
from vla_project.data.transforms.image import SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer
from vla_project.deployment.post_process_loader import load_post_process
from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper
from vla_project.models.vision.factory import build_vision_encoder
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.training.checkpoint import load_checkpoint


class MetaJsonError(Exception):
    """meta.json missing, malformed, or missing required keys."""


_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"unsupported dtype {name!r} (expected bf16 | fp32)")
    return _DTYPE_MAP[name]


def _resolve_ckpt_dir(ckpt_dir: str | Path) -> Tuple[Path, bool]:
    """Resolve ckpt_dir → (local_path, is_local).

    Accepts:
      - a local directory path (existing) → ``(p, True)``.
      - an HF model repo id ``org/repo`` (no local path) → ``snapshot_download``
        the whole repo to the HF cache and return ``(local, False)``.
      - an HF model repo + subfolder ``org/repo/subfolder`` (e.g.
        ``takaki99/so101-v46/step_2000``) → ``snapshot_download`` with
        ``allow_patterns=[f"{subfolder}/*"]`` and return ``(sub, False)``.

    The HF cache is `~/.cache/huggingface/hub/` by default, so subsequent
    loads are free. ``is_local`` is ``True`` only when the path already existed
    on disk; it is ``False`` for HF-resolved paths.
    """
    p = Path(ckpt_dir)
    if p.exists():
        return p, True
    s = str(ckpt_dir)
    # Reject absolute / relative-with-prefix paths from the HF heuristic
    # (codex round 5): a missing absolute path like "/foo/bar" is clearly
    # a local-path typo, not an HF repo id. HF repo ids never start with
    # "/", ".", or contain "..".
    if s.startswith("/") or s.startswith(".") or ".." in s.split("/"):
        raise FileNotFoundError(f"ckpt_dir {ckpt_dir!r} not found locally")
    parts = s.split("/")
    if len(parts) == 2:
        # bare repo_id like "takaki99/so101-v46" — pull everything
        local = Path(snapshot_download(repo_id=str(ckpt_dir), repo_type="model"))
        return local, False
    if len(parts) == 3:
        # repo_id + subfolder like "takaki99/so101-v46/step_2000"
        repo_id = "/".join(parts[:2])
        subfolder = parts[2]
        local = Path(snapshot_download(
            repo_id=repo_id, repo_type="model",
            allow_patterns=[f"{subfolder}/*"],
        ))
        sub = local / subfolder
        if not sub.is_dir():
            raise FileNotFoundError(
                f"resolved HF repo {repo_id!r} but subfolder {subfolder!r} not present in download"
            )
        return sub, False
    raise FileNotFoundError(
        f"ckpt_dir {ckpt_dir!r} not found locally and not in 'org/repo' or "
        f"'org/repo/subfolder' HF form (got {len(parts)} path components)"
    )


class ModelRuntime:
    def __init__(
        self,
        *,
        step: int,
        cfg: Dict[str, Any],
        norm_stats: Dict[str, Any],
        ckpt_dir: Path,
        model: VLAPolicy | None,
        tokenizer: GemmaPromptTokenizer | None,
        image_transform: SiglipImageTransform | None,
        device: torch.device,
        dtype: torch.dtype,
        is_local: bool,
        post_process_fn: Callable | None,
        post_process_path: str | None,
        meta_raw: dict,
    ) -> None:
        self.step = step
        self.cfg = cfg
        self.norm_stats = norm_stats
        self.ckpt_dir = ckpt_dir
        self.model = model
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.device = device
        self.dtype = dtype
        self.is_local = is_local
        self.post_process_fn = post_process_fn
        self.post_process_path = post_process_path
        self.meta_raw = meta_raw

    @classmethod
    def from_export(
        cls,
        ckpt_dir: str | Path,
        *,
        device: str = "cuda:0",
        dtype: str = "bf16",
        torch_compile: str = "off",
        warmup_iters: int = 1,
        trust_checkpoint_code: bool = False,
    ) -> "ModelRuntime":
        # Accept local path or HF repo id (see _resolve_ckpt_dir). HF
        # downloads are cached at ~/.cache/huggingface/hub/.
        ckpt_dir, is_local = _resolve_ckpt_dir(ckpt_dir)
        meta_path = ckpt_dir / "meta.json"
        if not meta_path.is_file():
            raise MetaJsonError(f"missing meta.json under {ckpt_dir}")
        meta = json.loads(meta_path.read_text())
        for required_key in ("step", "cfg", "norm_stats"):
            if required_key not in meta:
                raise MetaJsonError(f"meta.json missing required key {required_key!r}")

        cfg = meta["cfg"]
        torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        torch_dtype = _resolve_dtype(dtype)

        # Build the model from cfg.model (replicates scripts/eval.py:90-100).
        model_cfg_dict = dict(cfg["model"])
        lora_cfg = model_cfg_dict.pop("lora", None)
        policy_cfg = VLAPolicyConfig(**model_cfg_dict)
        vision = build_vision_encoder(
            vision_type=str(cfg["vision"].get("type", "hf")),
            model_name=cfg["vision"]["model_name"],
        )
        gemma = Gemma4Wrapper(
            model_name=cfg["language"]["model_name"], freeze=True, lora=lora_cfg
        )
        model = VLAPolicy(policy_cfg, vision, gemma).to(torch_device).to(torch_dtype)
        model.eval()

        # Load weights. strict=False allows tolerated drift (e.g. missing
        # baseline wrist_proj weights when use_wrist_bridge=True is dead).
        load_checkpoint(str(ckpt_dir), model, strict=False)

        post_process_fn = load_post_process(
            ckpt_dir, is_local=is_local, trust_checkpoint_code=trust_checkpoint_code,
        )
        post_process_path = str(ckpt_dir / "post_process.py") if post_process_fn is not None else None

        if torch_compile != "off":
            # bs=1 stable-shape inference: reduce-overhead (CUDA graphs)
            # tends to win; default also valid. fullgraph=False allows the
            # HF Gemma graph breaks.
            model = torch.compile(model, mode=torch_compile, fullgraph=False)

        # Tokenizer + image transform are bound to the same ckpt: prompt
        # max_len comes from the model cfg; image is the SigLIP 224 path.
        tokenizer = GemmaPromptTokenizer(
            model_name=cfg["language"]["model_name"],
            max_len=int(policy_cfg.prompt_max_len),
        )
        image_transform = SiglipImageTransform(
            size=C.SIGLIP_IMAGE_SIZE, training=False
        )

        runtime = cls(
            step=int(meta["step"]),
            cfg=cfg,
            norm_stats=meta["norm_stats"],
            ckpt_dir=ckpt_dir,
            model=model,
            tokenizer=tokenizer,
            image_transform=image_transform,
            device=torch_device,
            dtype=torch_dtype,
            is_local=is_local,
            post_process_fn=post_process_fn,
            post_process_path=post_process_path,
            meta_raw=meta,
        )

        # Warmup forward(s) reduce first-call latency (especially with
        # torch.compile, which JITs on first call).
        for _ in range(max(0, int(warmup_iters))):
            runtime._warmup_forward()

        return runtime

    @classmethod
    def from_meta_only(
        cls,
        ckpt_dir: str | Path,
        *,
        trust_checkpoint_code: bool = False,
    ) -> "ModelRuntime":
        """Lightweight load for predictors that don't need the actual model (e.g., HoldPosition smoke).

        Sets model=None; do not use with XVLAAdapter. Loads meta.json and
        post_process.py only — no model weights, no tokenizer, no image_transform.
        """
        ckpt_dir, is_local = _resolve_ckpt_dir(ckpt_dir)
        meta_path = ckpt_dir / "meta.json"
        if not meta_path.is_file():
            raise MetaJsonError(f"missing meta.json under {ckpt_dir}")
        meta = json.loads(meta_path.read_text())
        for required_key in ("step", "cfg", "norm_stats"):
            if required_key not in meta:
                raise MetaJsonError(f"meta.json missing required key {required_key!r}")

        post_process_fn = load_post_process(
            ckpt_dir, is_local=is_local, trust_checkpoint_code=trust_checkpoint_code,
        )
        post_process_path = str(ckpt_dir / "post_process.py") if post_process_fn is not None else None

        return cls(
            step=int(meta["step"]),
            cfg=meta["cfg"],
            norm_stats=meta["norm_stats"],
            ckpt_dir=ckpt_dir,
            model=None,
            tokenizer=None,
            image_transform=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
            is_local=is_local,
            post_process_fn=post_process_fn,
            post_process_path=post_process_path,
            meta_raw=meta,
        )

    def _warmup_forward(self) -> None:
        # action_chunk_len lives under cfg.data in the project's train YAMLs
        # (not cfg.model); fall back to constants if neither is set
        # (codex round 4 partial concern claim 3).
        action_chunk_len = int(
            self.cfg.get("data", {}).get("action_chunk_len")
            or self.cfg["model"].get("action_chunk_len")
            or C.ACTION_CHUNK_LEN
        )
        prompt_max_len = int(self.cfg["model"].get("prompt_max_len", 20))
        action_dim = C.ACTION_DIM
        proprio_dim = C.PROPRIO_DIM
        batch = {
            "domain_id": torch.tensor([0], dtype=torch.long, device=self.device),
            "scene_image": torch.zeros(1, 3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE, device=self.device, dtype=self.dtype),
            "wrist_image": torch.zeros(1, 3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE, device=self.device, dtype=self.dtype),
            "prompt_input_ids": torch.zeros(1, prompt_max_len, dtype=torch.long, device=self.device),
            "prompt_attention_mask": torch.ones(1, prompt_max_len, dtype=torch.long, device=self.device),
            "proprio": torch.zeros(1, proprio_dim, device=self.device, dtype=self.dtype),
            "last_action_chunk": torch.zeros(1, action_chunk_len, action_dim, device=self.device, dtype=self.dtype),
            "target_action": torch.zeros(1, action_chunk_len, action_dim, device=self.device, dtype=self.dtype),
            "action_mask": torch.ones(1, action_chunk_len, dtype=torch.bool, device=self.device),
            "wrist_mask": torch.ones(1, dtype=torch.bool, device=self.device),
        }
        with torch.no_grad():
            self.model(batch)

    def __call__(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Any]:
        """Forward the batch through the model, returning (pred, aux).

        The trainer calls model(batch) which returns (pred, loss_or_aux).
        At inference time we ignore aux and use pred (shape [B, T, A] for
        native action_format).
        """
        with torch.no_grad():
            return self.model(batch)
