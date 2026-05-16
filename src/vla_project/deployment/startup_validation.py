"""Startup-time validation of meta.json against runtime args.

Spec §8 lists the non-contract checks that survived the yaml-less
refactor: domain_id range, unnorm_key in norm_stats, chunk_len/dim
consistency, q01/q99/mask/std/min/max shape agreement, wrist
hard-required derivation, native_action presence (warn only).
"""
from __future__ import annotations

import logging

logger = logging.getLogger("vla_project.deployment.startup_validation")


class HardFailAssertion(Exception):
    """Raised at startup if meta.json ↔ args are inconsistent."""


_STATS_FIELDS = ("q01", "q99", "mask", "mean", "std", "min", "max")


def resolve_unnorm_key(meta: dict, override: str | None) -> str:
    """Pick the unnorm_key for norm_stats lookup.

    Rules:
      - If override is given, must exist in meta.norm_stats.
      - Else if norm_stats has exactly one key, use it.
      - Else fail with HardFailAssertion (require --unnorm-key).
    """
    keys = list(meta["norm_stats"].keys())
    if override is not None:
        if override not in keys:
            raise HardFailAssertion(
                f"--unnorm-key={override!r} not in meta.norm_stats keys {keys}"
            )
        return override
    if len(keys) == 1:
        return keys[0]
    raise HardFailAssertion(
        f"meta.norm_stats has multiple keys {keys}; pass --unnorm-key"
    )


def derive_wrist_hard_required(meta: dict) -> bool:
    """Whether the model architecture requires wrist_image at request time.

    Hard required if any of:
      - use_wrist_bridge True
      - use_scene_wrist_dinov2_llm / wrist_dinov2 True
      - wrist_in_llm True AND wrist_view_dropout_p == 0.0
    """
    m = meta["cfg"].get("model", {})
    bridge_or_dinov2 = bool(
        m.get("use_wrist_bridge", False)
        or m.get("use_scene_wrist_dinov2_llm", False)
        or m.get("wrist_dinov2", False)
    )
    in_llm = bool(m.get("wrist_in_llm", False))
    dropout = float(m.get("wrist_view_dropout_p") or 0.0)
    return bridge_or_dinov2 or (in_llm and dropout == 0.0)


def validate_runtime(
    meta: dict,
    *,
    unnorm_key: str,
    domain_id: int,
    model_action_dim: int,
) -> None:
    """Run all startup checks (§8). Raises HardFailAssertion on first failure."""
    m_model = meta["cfg"].get("model", {})
    m_data = meta["cfg"].get("data", {})

    # (1) domain_id range
    num_domains = int(m_model.get("num_domains", 0))
    if not (0 <= domain_id < num_domains):
        raise HardFailAssertion(
            f"domain_id={domain_id} out of range [0, {num_domains})"
        )

    # (2) unnorm_key in norm_stats
    if unnorm_key not in meta["norm_stats"]:
        raise HardFailAssertion(
            f"unnorm_key={unnorm_key!r} missing from meta.norm_stats"
        )

    # (3) action_chunk_len: cfg.data ↔ cfg.model agree (if model declares)
    data_chunk = m_data.get("action_chunk_len")
    model_chunk = m_model.get("action_chunk_len")
    if data_chunk is not None and model_chunk is not None and data_chunk != model_chunk:
        raise HardFailAssertion(
            f"action_chunk_len mismatch: cfg.data={data_chunk}, cfg.model={model_chunk}"
        )

    # (4) action stats dim ↔ model action dim
    stats = meta["norm_stats"][unnorm_key]
    action_q99 = stats["action"]["q99"]
    if len(action_q99) != model_action_dim:
        raise HardFailAssertion(
            f"norm_stats.action dim={len(action_q99)} != model action_dim={model_action_dim}"
        )

    # (5) proprio stats dim ↔ cfg.model.proprio_dim
    expected_proprio_dim = int(m_model.get("proprio_dim", 0))
    proprio_q99 = stats["proprio"]["q99"]
    if expected_proprio_dim > 0 and len(proprio_q99) != expected_proprio_dim:
        raise HardFailAssertion(
            f"norm_stats.proprio dim={len(proprio_q99)} != cfg.model.proprio_dim={expected_proprio_dim}"
        )

    # (6) all stats fields agree in shape per block
    for block_name in ("action", "proprio"):
        block = stats[block_name]
        ref_len = len(block["q99"])
        for fld in _STATS_FIELDS:
            if fld not in block:
                raise HardFailAssertion(f"norm_stats.{block_name} missing {fld!r}")
            if len(block[fld]) != ref_len:
                raise HardFailAssertion(
                    f"norm_stats.{block_name}.{fld} len={len(block[fld])} != q99 len={ref_len}"
                )

    # (9) native_action absent → WARN only, do not fail
    if "native_action" not in meta:
        logger.warning(
            "native_action metadata absent; clients must know action convention out-of-band"
        )
