#!/usr/bin/env bash
# CLEAN-V2 chain. Changes vs ml4h_clean_full: v2 pin (all three eras have adaptation
# cohorts), FORCE_ADAPT=1, incident-AUROC selection, LR grid 3e-5..3e-3, fatal pin
# check, script17 with C / C_l2sp and patient-level paired bootstrap.
# NI_MARGIN fixed at 0.015 incident AUROC BEFORE results were seen (2026-10-08).
BASE_DIR="${BASE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$BASE_DIR" || exit 1
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10} MKL_NUM_THREADS=${MKL_NUM_THREADS:-10} PYTHONUNBUFFERED=1 NI_MARGIN=${NI_MARGIN:-0.015}
mkdir -p logs
step () { n=$1; shift; date "+%F %T START $n" >> logs/progress.log; "$@" > logs/$n.log 2>&1 && date "+%F %T done  $n" >> logs/progress.log || { date "+%F %T FAILED $n" >> logs/progress.log; return 1; }; }
: > logs/progress.log
step script2 env PIN_SPLIT="${PIN_SPLIT:-$BASE_DIR/pin_split_v2.json}" FORCE_ADAPT=1 $PY 2_source_model/script2_two_stream_model.py || exit 1
step script17 $PY 4_adaptation/script17_seed_floor.py
