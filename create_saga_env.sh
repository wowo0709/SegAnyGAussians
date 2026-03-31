#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/environment_new.yml"
CUDA_EXT_REQ="$ROOT_DIR/requirements_cuda_ext.txt"

export FORCE_CUDA="${FORCE_CUDA:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9}"

conda env remove -n saga -y >/dev/null 2>&1 || true
conda env create -n saga -f "$ENV_FILE"
conda run -n saga pip install --upgrade pip setuptools wheel ninja

rm -rf "$ROOT_DIR/submodules/diff-gaussian-rasterization/build"
rm -rf "$ROOT_DIR/submodules/diff-gaussian-rasterization_contrastive_f/build"
rm -rf "$ROOT_DIR/submodules/diff-gaussian-rasterization-depth/build"
rm -rf "$ROOT_DIR/submodules/simple-knn/build"
find "$ROOT_DIR/submodules" -name '*.so' -delete
find "$ROOT_DIR/submodules" -name '*.egg-info' -type d -prune -exec rm -rf {} +

conda run -n saga pip install --no-build-isolation -r "$CUDA_EXT_REQ"

cat <<EOF
[OK] saga environment is ready.
Use: conda activate saga
Check: python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
EOF
