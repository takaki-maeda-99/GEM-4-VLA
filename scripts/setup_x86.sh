#!/usr/bin/env bash
# GEM-4-VLA x86_64 training/research environment setup.
# Usage:  bash scripts/setup_x86.sh         (run from repo root)
#
# Targets x86_64 Linux hosts with CUDA driver >= 12.6 (cu128 wheels).
# For Jetson Orin (aarch64), use scripts/setup_jetson.sh instead.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
ENV_DIR="$ROOT/envs/x86"
echo ">> repo root: $ROOT"
echo ">> env dir:   $ENV_DIR"

# 1. uv (manages Python 3.10 via envs/x86/.python-version)
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo ">> uv: $(uv --version)"

# 2. Submodules (VLA-Adapter, X-VLA — used as code references / vendored utils)
echo ">> git submodule update --init --recursive"
git submodule update --init --recursive

# 3. Python deps (resolved from envs/x86/pyproject.toml + envs/x86/uv.lock)
#    - torch/torchvision pinned to cu128 wheels (driver >= 12.6).
#    - transformers>=5.5 override bypasses lerobot's <5.0 cap; required for
#      Gemma4 model_type registration.
#    - dlimp + tensorflow-graphics are bundled in this env for the OXE/RLDS
#      data pipeline. They are NOT in the Jetson env because
#      tensorflow-addons has no Linux aarch64 wheel.
echo ">> uv sync --project $ENV_DIR"
uv sync --project "$ENV_DIR"

# 4. Smoke: torch CUDA + transformers Gemma4 + deployment package
echo ">> smoke check..."
uv run --project "$ENV_DIR" python - <<'PY'
import torch, transformers
print(f"  torch       = {torch.__version__}  cuda={torch.cuda.is_available()}  ndev={torch.cuda.device_count()}")
print(f"  transformers= {transformers.__version__}")
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
gemma4_keys = [k for k in CONFIG_MAPPING_NAMES if k.startswith("gemma4")]
assert gemma4_keys, "Gemma4 model_type not registered (need transformers>=5.0)"
print(f"  Gemma4      = {gemma4_keys}")

# Deployment package: verify fastapi/uvicorn + build_app importable
import fastapi, uvicorn
from vla_project.deployment.inference_server import build_app
print(f"  fastapi     = {fastapi.__version__}  uvicorn={uvicorn.__version__}")
print(f"  deployment  = build_app importable")
PY

# 5. Runtime notes
cat <<EOF

>> setup OK (envs/x86).

Run any subsequent uv command with --project envs/x86 (or cd into envs/x86):
  uv run --project envs/x86 python scripts/train.py configs/train/<config>.yaml
  uv run --project envs/x86 pytest

Runtime notes:
  HF login (one-time)        : uv run --project envs/x86 huggingface-cli login
  Pytest (ROS2 path workaround): PYTHONPATH="" uv run --project envs/x86 pytest -v

Train / eval invocation:
  CUDA_VISIBLE_DEVICES=0 \\
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  uv run --project envs/x86 python scripts/train.py configs/train/<config>.yaml

Host-specific paths used by configs (not installed by this script):
  - vla-gemma-4 sibling repo  (RLDS data + baseline ckpts under data/, outputs/)
  - LIBERO simulator + assets (set MUJOCO_GL=osmesa for headless render)

EOF
