#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 ABLATION_ROOT [--gpu_ids 0 1 2 ...]" >&2
  exit 1
fi

python -u "$SCRIPT_DIR/run_ablation_suite_parallel.py" "$@"
