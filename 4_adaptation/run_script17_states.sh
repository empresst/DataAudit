#!/usr/bin/env bash
BASE_DIR="${BASE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10} MKL_NUM_THREADS=${MKL_NUM_THREADS:-10} PYTHONUNBUFFERED=1 NI_MARGIN=${NI_MARGIN:-0.015}
for d in ml4h_clean_v2 mortality_variant; do
  target_dir="$BASE_DIR/$d"
  [ -d "$target_dir" ] || target_dir="$BASE_DIR"
  cd "$target_dir" || continue
  mkdir -p logs
  S17_ARMS=B,C S17_SUFFIX=_bc_states S17_SAVE_STATES=1 $PY 4_adaptation/script17_seed_floor.py > logs/script17_states.log 2>&1 || { echo "$d FAILED"; exit 1; }
  $PY - <<'P'
import numpy as np
a = np.load("per_era_predictions_seeds.npz", allow_pickle=True); b = np.load("per_era_predictions_seeds_bc_states.npz", allow_pickle=True)
ks = [k for k in b.files if k.startswith("probs") and ("__B__" in k or "__C__" in k)]
w = max(float(np.abs(a[k] - b[k]).max()) for k in ks)
print(f"  {len(ks)} B/C prediction arrays vs original run: max |diff| {w:.2e}", "IDENTICAL" if w < 1e-6 else "DIFFERS")
P
  echo "$d done"
done
echo ALL_DONE
