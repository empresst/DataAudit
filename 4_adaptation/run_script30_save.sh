#!/usr/bin/env bash
# rerun script30 with model saving; vaso/intub then mortality; compare against v1 predictions
BASE_DIR="${BASE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10} MKL_NUM_THREADS=${MKL_NUM_THREADS:-10} XGB_NTHREAD=${XGB_NTHREAD:-10} PYTHONUNBUFFERED=1 NI_MARGIN=${NI_MARGIN:-0.015}
for d in ml4h_clean_v2 mortality_variant; do
  target_dir="$BASE_DIR/$d"
  [ -d "$target_dir" ] || target_dir="$BASE_DIR"
  cd "$target_dir" || continue
  mkdir -p logs
  $PY 4_adaptation/script30_modular_xgboost_clean.py > logs/script30_save.log 2>&1 && echo "$d script30 OK" || { echo "$d script30 FAILED"; exit 1; }
  $PY - <<'P'
import numpy as np
a=np.load("v1_before_save/per_era_predictions_modxgb.npz"); b=np.load("per_era_predictions_modxgb.npz")
assert set(a.files)==set(b.files), "key sets differ"
w=max(float(np.abs(a[k].astype(float)-b[k].astype(float)).max()) for k in a.files if k.startswith("probs"))
print(f"  max |prob change| vs previous run: {w:.2e}", "IDENTICAL" if w<1e-6 else "DIFFERS")
P
done
echo ALL_DONE
