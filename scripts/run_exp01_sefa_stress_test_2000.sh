#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/rz/raymond_project/sefa"
CONDA_BIN="/home/rz/miniconda3/bin/conda"

LATENT_DIR="/mnt/d/Ph.D/01_Experiments_GAN_Inv_Log/2026.03.08_All_Experiments_Start/exper_07_e4e_encode2latentcode"
OUTPUT_DIR="${LATENT_DIR}/sefa_stress_test_result"

NUM_SAMPLES=140
NUM_PER_GROUP=70
SEMANTIC_INDICES=(0 5 12)
STRENGTHS=(-6 -3 0 3 6)

mkdir -p "${OUTPUT_DIR}"

if pgrep -af "exp01_check_e4e_latent_and_sefa_generate.py" >/dev/null; then
  echo "Another exp01 SeFa generation process is already running:"
  pgrep -af "exp01_check_e4e_latent_and_sefa_generate.py"
  exit 1
fi

female_count=$(find "${LATENT_DIR}/gt_female/per_image_npy" -maxdepth 1 -type f -name "*.npy" | wc -l)
male_count=$(find "${LATENT_DIR}/gt_male/per_image_npy" -maxdepth 1 -type f -name "*.npy" | wc -l)

echo "Repo: ${REPO_DIR}"
echo "Latent dir: ${LATENT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Female per-image latents: ${female_count}"
echo "Male per-image latents: ${male_count}"
echo "Running ${NUM_SAMPLES} samples total (${NUM_PER_GROUP} per gt_female/gt_male group) with semantics: ${SEMANTIC_INDICES[*]}"
echo "Strengths: ${STRENGTHS[*]}"
echo

cd "${REPO_DIR}"

cmd=(
  "${CONDA_BIN}"
  run
  -n
  sefa
  python
  scripts/exp01_check_e4e_latent_and_sefa_generate.py
  --latent_dir "${LATENT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --num_samples "${NUM_SAMPLES}"
  --num_per_group "${NUM_PER_GROUP}"
  --sampling_strategy even
  --semantic_indices "${SEMANTIC_INDICES[@]}"
  --strengths "${STRENGTHS[@]}"
  --device cuda
)

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  cmd+=(--overwrite)
  echo "OVERWRITE=1: existing output directory will be cleared by the Python script."
fi

echo "Command:"
printf ' %q' "${cmd[@]}"
echo
echo

"${cmd[@]}" 2>&1 | tee "${OUTPUT_DIR}/runner_stdout.log"

echo
echo "Done."
echo "Contact sheets should be under: ${OUTPUT_DIR}/gt_*/image_*/contact_sheet_image_*.png"
echo "Main log: ${OUTPUT_DIR}/exp01_log.txt"
