#!/usr/bin/env bash
# X-VLA-Adapter environment setup.
# Usage:  bash scripts/setup.sh         (run from repo root)
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
echo ">> repo root: $ROOT"

# 1. uv (manages Python 3.11 from .python-version)
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo ">> uv: $(uv --version)"

# 2. Submodules (VLA-Adapter, X-VLA — used as code references / vendored utils)
echo ">> git submodule update --init --recursive"
git submodule update --init --recursive

# 3. Python deps
#    - cu128 wheels are pinned in pyproject.toml [tool.uv.sources] for
#      Blackwell sm_120 / driver 12.6+.
#    - transformers>=5.0 override (pyproject [tool.uv]) bypasses lerobot's
#      <5.0 cap; required for Gemma4 model_type registration.
echo ">> uv sync --extra dev"
uv sync --extra dev

# 4. Smoke: torch CUDA + transformers Gemma4 + deployment package
echo ">> smoke check..."
uv run python - <<'PY'
import torch, transformers
print(f"  torch       = {torch.__version__}  cuda={torch.cuda.is_available()}  ndev={torch.cuda.device_count()}")
print(f"  transformers= {transformers.__version__}")
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
gemma4_keys = [k for k in CONFIG_MAPPING_NAMES if k.startswith("gemma4")]
assert gemma4_keys, "Gemma4 model_type not registered (need transformers>=5.0)"
print(f"  Gemma4      = {gemma4_keys}")

# Deployment package: verify fastapi/uvicorn/pyyaml + build_app importable
import fastapi, uvicorn, yaml
from vla_project.deployment.inference_server import build_app
from vla_project.deployment.domain_adapter import load_deploy_config
print(f"  fastapi     = {fastapi.__version__}  uvicorn={uvicorn.__version__}  pyyaml={yaml.__version__}")
cfg = load_deploy_config("configs/deploy/v36_libero_spatial.yaml")
assert cfg.ckpt.expected_action_chunk_len == 8
print(f"  deployment  = build_app importable, v36 deploy yaml loads OK")
PY

# 5. Runtime notes (host-specific paths NOT auto-configured)
cat <<'EOF'

>> setup OK.

Runtime notes:
  HF login (one-time)        : uv run huggingface-cli login
  Pytest (ROS2 path workaround): PYTHONPATH="" uv run pytest -v

Train / eval invocation:
  CUDA_VISIBLE_DEVICES=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scripts/train.py configs/train/<config>.yaml

Inference server (Phase 0 HoldPosition; no GPU / no ckpt required):
  uv run python scripts/serve.py \
    --predictor hold_position \
    --deploy-config configs/deploy/v36_libero_spatial.yaml \
    --domain-id 0 \
    --port 8001
  curl http://127.0.0.1:8001/healthz
  # See src/vla_project/deployment/README.md for XVLAAdapter mode + deploy
  # yaml authoring + Phase 0 acceptance verification.

Host-specific paths used by configs (not installed by this script):
  - vla-gemma-4 sibling repo  (RLDS data + baseline ckpts under data/, outputs/)
  - LIBERO simulator + assets (set MUJOCO_GL=osmesa for headless render)

EOF
