#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/environment_new.yml"
CUDA_EXT_REQ="$ROOT_DIR/requirements_cuda_ext.txt"

conda env remove -n saga -y >/dev/null 2>&1 || true
conda env create -n saga -f "$ENV_FILE"
conda run -n saga pip install --no-build-isolation -r "$CUDA_EXT_REQ"

echo "[OK] saga environment is ready."
echo "Use: conda activate saga"
