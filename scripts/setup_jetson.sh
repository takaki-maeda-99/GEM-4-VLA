#!/usr/bin/env bash
# X-VLA-Adapter Jetson Orin deploy environment setup.
# Usage:  bash scripts/setup_jetson.sh         (run from repo root)
#
# Targets Jetson Orin (aarch64, sm_87) running JetPack 6.2.x / CUDA 12.6.
# Wheels come from the jetson-ai-lab JetPack 6 / CUDA 12.6 index.
# For x86_64 training/research hosts, use scripts/setup_x86.sh instead.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
ENV_DIR="$ROOT/envs/jetson"
echo ">> repo root: $ROOT"
echo ">> env dir:   $ENV_DIR"

# 1. uv (manages Python 3.10 via envs/jetson/.python-version — Jetson
#    jetson-ai-lab index only publishes cp310 wheels)
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo ">> uv: $(uv --version)"

# 2. Submodules (VLA-Adapter, X-VLA — used as code references / vendored utils)
echo ">> git submodule update --init --recursive"
git submodule update --init --recursive

# 3. Python deps (resolved from envs/jetson/pyproject.toml + envs/jetson/uv.lock)
#    - torch/torchvision pinned to jetson-ai-lab wheels (sm_87 baked in).
#      Upstream cu126/cu128/cu130 wheels are sm_90+ and crash with
#      "no kernel image" at .to(cuda) on Orin.
#    - transformers>=5.5 override bypasses lerobot's <5.0 cap; required for
#      Gemma4 model_type registration.
#    - Triton is excluded from the aarch64 solve (Triton publishes Linux
#      x86_64 wheels only).
#    - dlimp / tensorflow-graphics are NOT installed here — the deploy host
#      doesn't need the OXE/RLDS data pipeline, and tensorflow-addons has
#      no Linux aarch64 wheel. Use envs/x86 for those.
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

>> setup OK (envs/jetson).

Run any subsequent uv command with --project envs/jetson (or cd into envs/jetson):
  uv run --project envs/jetson python scripts/serve.py ...

Runtime notes (Jetson-specific):
  - scripts/serve.py sets PYTORCH_NVML_BASED_CUDA_CHECK=0 before importing
    torch; Tegra does not implement NVML the way PyTorch's caching allocator
    expects, which otherwise breaks model.to('cuda').
  - HF login (one-time): uv run --project envs/jetson huggingface-cli login

Inference server (HoldPosition still needs --checkpoint for meta.json; an
XVLAAdapter ckpt dir or HF id works):
  uv run --project envs/jetson python scripts/serve.py \\
    --predictor hold_position \\
    --checkpoint <ckpt_dir_or_hf_id> \\
    --port 8001
  curl http://127.0.0.1:8001/healthz
  # See src/vla_project/deployment/README.md for XVLAAdapter mode +
  # Phase 0 acceptance verification.

EOF
