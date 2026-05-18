"""Builder for the GET /metadata response. Spec §4."""
from __future__ import annotations


def build_metadata_response(
    meta: dict,
    *,
    unnorm_key: str,
    domain_id: int,
    has_post_process: bool,
    post_process_path: str | None,
) -> dict:
    cfg = meta["cfg"]
    stats = meta["norm_stats"][unnorm_key]
    return {
        "step": int(meta["step"]),
        "model_name": cfg["language"]["model_name"],
        "git_commit": meta.get("git_commit", ""),
        "action_dim": len(stats["action"]["q99"]),
        "proprio_dim": len(stats["proprio"]["q99"]),
        "action_chunk_len": int(cfg["data"]["action_chunk_len"]),
        "domain_id": int(domain_id),
        "num_domains": int(cfg["model"].get("num_domains", 0)),
        "unnorm_key": unnorm_key,
        "native_action": meta.get("native_action"),
        "has_post_process": bool(has_post_process),
        "post_process_module": post_process_path,
    }
