#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-sefa}"
HOST="${HOST:-0.0.0.0}"
PORT="${1:-${PORT:-8501}}"

if [[ ! -f "$CONDA_HOME/etc/profile.d/conda.sh" ]]; then
  echo "Cannot find conda at: $CONDA_HOME" >&2
  echo "Set CONDA_HOME=/path/to/miniconda3 and try again." >&2
  exit 1
fi

source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$SCRIPT_DIR"

echo "Launching SeFa Streamlit demo"
echo "  env:  $CONDA_ENV"
echo "  url:  http://localhost:$PORT"
echo

exec streamlit run interface.py \
  --server.headless true \
  --server.address "$HOST" \
  --server.port "$PORT"
