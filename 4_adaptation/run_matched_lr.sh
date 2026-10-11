#!/usr/bin/env bash
# matched-LR sensitivity: B and C share one adaptation LR; then explanation stability.
BASE_DIR="${BASE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10} MKL_NUM_THREADS=${MKL_NUM_THREADS:-10} PYTHONUNBUFFERED=1 NI_MARGIN=${NI_MARGIN:-0.015}
mkdir -p logs
run () {  # dir lr tag
  local target_dir="$BASE_DIR/$1"
  [ -d "$target_dir" ] || target_dir="$BASE_DIR"
  ( cd "$target_dir" && mkdir -p logs && \
    S17_ARMS=B,C S17_LR=$2 S17_SUFFIX=_lr$3 S17_SAVE_STATES=1 $PY 4_adaptation/script17_seed_floor.py > logs/script17_lr$3.log 2>&1 && \
    S33_SUFFIX=_lr$3 $PY 5_explainability/script33_nn_explanation_stability.py > logs/script33_lr$3.log 2>&1 ) || { echo "$1 lr$3 FAILED"; return 1; }
  echo "$1 lr$3 done"
}
run ml4h_clean_v2 0.0003 3e-4
run ml4h_clean_v2 0.003 3e-3
run mortality_variant 0.003 3e-3
# reproducibility check: tuned B used 0.003, so B at lr 3e-3 must match it
cd "$BASE_DIR" && $PY - <<'P'
import numpy as np
a = np.load("per_era_predictions_seeds.npz"); b = np.load("per_era_predictions_seeds_lr3e-3.npz")
ks = [k for k in b.files if "__B__" in k]
print("B at lr 3e-3 vs tuned B (same LR): max |diff|", f"{max(float(np.abs(a[k]-b[k]).max()) for k in ks):.2e}")
P
echo ALL_DONE
