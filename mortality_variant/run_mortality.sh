#!/usr/bin/env bash
# MORTALITY chain on the clean-v2 data (BIOLOGICAL_LABEL=True in utils/constants.py).
# Same code, pin, forced schedule and selection rules as ml4h_clean_v2; only the label differs.
# Mortality has no incident gate: every metric is FULL COHORT. Margin 0.015 fixed in advance.
BASE_DIR="${BASE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$BASE_DIR" || exit 1
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10} MKL_NUM_THREADS=${MKL_NUM_THREADS:-10} XGB_NTHREAD=${XGB_NTHREAD:-10} PYTHONUNBUFFERED=1 NI_MARGIN=${NI_MARGIN:-0.015}
mkdir -p logs
step () { n=$1; shift; date "+%F %T START $n" >> logs/progress.log
          if "$@" > logs/$n.log 2>&1; then date "+%F %T done  $n" >> logs/progress.log
          else date "+%F %T FAILED $n (chain stopped)" >> logs/progress.log; exit 1; fi; }
: > logs/progress.log
step script2  env PIN_SPLIT="${PIN_SPLIT:-$BASE_DIR/pin_split_v2.json}" FORCE_ADAPT=1 $PY 2_source_model/script2_two_stream_model.py
step script17 $PY 4_adaptation/script17_seed_floor.py
step script30 $PY 4_adaptation/script30_modular_xgboost_clean.py
date "+%F %T ALL DONE" >> logs/progress.log
