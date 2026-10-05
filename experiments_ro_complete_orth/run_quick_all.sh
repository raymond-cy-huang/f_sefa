#!/usr/bin/env bash
set -u

cd /home/rz/raymond_project/f_sefa || exit 1

LOG_PATH=/home/rz/raymond_project/f_sefa/experiments_ro_complete_orth/quick_all.log
PYTHON_PATH=/home/rz/miniconda3/envs/pixel2style2pixel/bin/python
RUN_PATH=/home/rz/raymond_project/f_sefa/experiments_ro_complete_orth/run.py

{
  echo "quick-all started: $(date --iso-8601=seconds)"
  env PATH=/home/rz/miniconda3/envs/pixel2style2pixel/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    "$PYTHON_PATH" "$RUN_PATH" --quick-all --resume
  STATUS=$?
  echo "quick-all finished: $(date --iso-8601=seconds), exit_code=$STATUS"
  exit "$STATUS"
} >"$LOG_PATH" 2>&1
