#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-/mnt/d/Ph.D/01_Experiments_GAN_Inv_Log/2026.03.08_All_Experiments_Start/FBE-Protocol_final_output/4.5_Semantic_Editing_Validation/sefa/find_glass_attr}"
OUT="$BASE/contact_sheets"
mkdir -p "$OUT"
for id in 000 001 002; do
  for start in 000 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150 160 170 180 190 200; do
    end=$((10#$start + 9))
    [[ "$start" == 200 ]] && end=200
    files=()
    for n in $(seq -w "$start" "$end"); do
      d="$BASE/eigen_$n"
      [[ -d "$d" ]] || continue
      files+=("$d/${id}_"*.jpg)
    done
    ((${#files[@]} > 0)) || continue
    montage "${files[@]}" -thumbnail 256x256 -tile 5x -geometry 256x256+2+2 \
      -background white -pointsize 18 -fill black \
      "$OUT/identity_${id}_eigen_${start}_${end}.jpg"
  done
done
