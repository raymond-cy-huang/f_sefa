#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/rz/raymond_project/sefa"
CONDA_BIN="/home/rz/miniconda3/bin/conda"

LATENT_DIR="/mnt/d/Ph.D/01_Experiments_GAN_Inv_Log/2026.03.08_All_Experiments_Start/exper_07_e4e_encode2latentcode"
OUTPUT_DIR="${REPO_DIR}/results/exp01_sefa_even_sampling_smoke"

NUM_SAMPLES=4
NUM_PER_GROUP=2
SEMANTIC_INDICES=(0 5 12)
STRENGTHS=(-6 -3 0 3 6)

cd "${REPO_DIR}"

"${CONDA_BIN}" run -n sefa python scripts/exp01_check_e4e_latent_and_sefa_generate.py \
  --latent_dir "${LATENT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_samples "${NUM_SAMPLES}" \
  --num_per_group "${NUM_PER_GROUP}" \
  --sampling_strategy even \
  --semantic_indices "${SEMANTIC_INDICES[@]}" \
  --strengths "${STRENGTHS[@]}" \
  --device cuda \
  --overwrite

echo
echo "Smoke test output: ${OUTPUT_DIR}"
