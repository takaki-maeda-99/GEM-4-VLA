"""Upload an GEM-4-VLA checkpoint directory to HuggingFace Hub.

Usage:
  # 1) preview only (lists files + sizes, doesn't upload)
  uv run python tools/push_ckpt_to_hf.py <ckpt_dir> <repo_id> --dry-run

  # 2) basic upload (private, model.pt + meta.json only — no optimizer)
  uv run python tools/push_ckpt_to_hf.py <ckpt_dir> <repo_id>

  # 3) full upload (model + optimizer + meta) for resume_full
  uv run python tools/push_ckpt_to_hf.py <ckpt_dir> <repo_id> --include-optimizer

  # 4) public + custom commit message
  uv run python tools/push_ckpt_to_hf.py <ckpt_dir> <repo_id> --public -m "message"

Examples:
  uv run python tools/push_ckpt_to_hf.py \\
      outputs/cookie_replace_v47_step100k_ft_dl42_2gpu/checkpoints/step_15000 \\
      takaki99/GEM-4-FT-cookie --include-optimizer

  uv run python tools/push_ckpt_to_hf.py \\
      outputs/oxe_pretrain_v47_arch_v3_libero_dl50_bs8/checkpoints/step_100000 \\
      takaki99/Gemma-4-Pretrained-OXE-step100k --include-optimizer

Notes:
  - Auth: reads ~/.cache/huggingface/token (set via `huggingface-cli login`).
  - The HF token must have **write** scope. If you see 403, regenerate via
    https://huggingface.co/settings/tokens with "Write" role.
  - Default skips optimizer.pt to save bandwidth (~1.6 GB). Include only if
    you plan to ``resume_full`` from this ckpt.
  - Repo is created if missing (idempotent, --private by default).
  - Idempotent for re-uploads: HF dedups by content hash, only changed files
    re-transfer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, create_repo, whoami


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _collect_files(ckpt_dir: Path, include_optimizer: bool) -> List[Path]:
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"ckpt dir not found: {ckpt_dir}")
    files: List[Path] = []
    for p in sorted(ckpt_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name == "optimizer.pt" and not include_optimizer:
            continue
        files.append(p)
    if not files:
        raise RuntimeError(f"no uploadable files under {ckpt_dir}")
    return files


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("ckpt_dir", type=Path,
                   help="Checkpoint directory (contains model.pt / meta.json / optimizer.pt).")
    p.add_argument("repo_id",
                   help='Target HF repo id, e.g. "takaki99/GEM-4-FT-cookie".')
    p.add_argument("--include-optimizer", action="store_true",
                   help="Also upload optimizer.pt (needed for resume_full).")
    p.add_argument("--public", action="store_true",
                   help="Create the repo as public (default: private).")
    p.add_argument("-m", "--message", default=None,
                   help="Commit message. Default auto-generated from ckpt dir name.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan and exit without uploading.")
    p.add_argument("--allow-existing-public", action="store_true",
                   help="Suppress warning when uploading to an existing public repo.")
    args = p.parse_args(argv)

    ckpt_dir = args.ckpt_dir.resolve()
    repo_id = args.repo_id

    files = _collect_files(ckpt_dir, include_optimizer=args.include_optimizer)
    total = sum(f.stat().st_size for f in files)

    msg = args.message or f"upload {ckpt_dir.parent.parent.name} / {ckpt_dir.name}"

    print(f"\n[push_ckpt_to_hf]")
    print(f"  source         : {ckpt_dir}")
    print(f"  repo_id        : {repo_id}")
    print(f"  visibility     : {'PUBLIC' if args.public else 'private'}")
    print(f"  commit message : {msg!r}")
    print(f"  files ({len(files)}):")
    for f in files:
        print(f"    {f.name:25s}  {_human_size(f.stat().st_size):>10s}")
    print(f"  total transfer : {_human_size(total)}")
    print()

    if args.dry_run:
        print("[dry-run] would upload the above. exit.")
        return 0

    if total > 15 * 1024**3:
        print(f"[warn] upload is {_human_size(total)}: this will take a while + bandwidth.")

    try:
        info = whoami()
        role = info.get("auth", {}).get("accessToken", {}).get("role")
        print(f"  hf user / role : {info.get('name')} ({role})")
        if role != "write":
            print(
                f"  [error] token role={role!r}, need 'write'. Regenerate via "
                f"https://huggingface.co/settings/tokens with 'Write' role, "
                f"then `huggingface-cli login` (or set HF_TOKEN env).",
                file=sys.stderr,
            )
            return 2
    except Exception as e:
        print(f"  [error] whoami() failed: {e}. Run `huggingface-cli login` first.",
              file=sys.stderr)
        return 2

    print(f"\n[push_ckpt_to_hf] creating / ensuring repo {repo_id} "
          f"({'PUBLIC' if args.public else 'private'})")
    url = create_repo(repo_id=repo_id, repo_type="model",
                      private=(not args.public), exist_ok=True)
    print(f"  url: {url}")

    api = HfApi()

    if args.public and not args.allow_existing_public:
        # Pull repo info to detect if it was already public — the warning helps
        # avoid accidentally leaking unfinished experiments.
        try:
            ri = api.model_info(repo_id)
            if not ri.private:
                # Already public; just informational.
                print(f"  [info] repo already public, proceeding.")
        except Exception:
            pass

    print(f"[push_ckpt_to_hf] uploading {len(files)} files ({_human_size(total)}) ...")
    api.upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=msg,
        allow_patterns=[f.name for f in files],   # skip optimizer.pt when not requested
    )
    print(f"\n[push_ckpt_to_hf] DONE: https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
