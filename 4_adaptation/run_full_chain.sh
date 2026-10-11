#!/usr/bin/env bash
# Full sequential chain (shared learning rates, standard drift detector, explanations, retrieval).
# Stops at the first failure. Key lines collected in chain_summary.log.
O="${BASE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
V="${DIR_V:-$O}"
M="${DIR_M:-$O/mortality_variant}"
PY="${PYTHON:-python3}"
PYR="${PYTHON_RAG:-$PY}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10} MKL_NUM_THREADS=${MKL_NUM_THREADS:-10} XGB_NTHREAD=${XGB_NTHREAD:-10} PYTHONUNBUFFERED=1 NI_MARGIN=${NI_MARGIN:-0.015} HF_HUB_OFFLINE=1
S="${SUMMARY_LOG:-$O/logs/chain_runners/chain_summary.log}"
mkdir -p "$(dirname "$S")"
: > "$S"
step () { local name=$1 dir=$2; shift 2; echo "$(date +%T) START $name" >> $S
  ( cd $dir && env "$@" > logs/$name.log 2>&1 ); local rc=$?
  if [ $rc -eq 3 ]; then echo "$(date +%T) STOP $name: drift not detected at some step (see logs/$name.log)" >> $S; exit 3; fi
  if [ $rc -ne 0 ]; then echo "$(date +%T) FAILED $name (exit $rc)" >> $S; tail -5 $dir/logs/$name.log >> $S; exit 1; fi
  echo "$(date +%T) done  $name" >> $S; }
step drift_detector      $V $PY script35_drift_detector.py
step trees_eta0.03       $V S30_ETA=0.03 S30_ARMS=B_xgb,C_xgb S30_SUFFIX=_eta0.03 $PY script30_modular_xgboost_clean.py
step trees_eta0.03       $M S30_ETA=0.03 S30_ARMS=B_xgb,C_xgb S30_SUFFIX=_eta0.03 $PY script30_modular_xgboost_clean.py
step tree_update_expl    $V S31_SUFFIX=_eta0.03 $PY script31_update_explanations.py
step tree_update_expl    $M S31_SUFFIX=_eta0.03 $PY script31_update_explanations.py
step tree_expl_stability $V S31_SUFFIX=_eta0.03 $PY script32_explanation_stability.py
step tree_expl_stability $M S31_SUFFIX=_eta0.03 $PY script32_explanation_stability.py
step nn_update_expl      $V S34_SUFFIX=_lr3e-3 $PY script34_nn_update_explanations.py
step nn_update_expl      $M S34_SUFFIX=_bc_states $PY script34_nn_update_explanations.py
step acc_A_vs_C          $V NN_PRED=per_era_predictions_seeds_lr3e-3.npz TREE_PRED=per_era_predictions_modxgb_eta0.03.npz OUT_TAG=_shared $PY analysis_A_vs_C.py
step acc_A_vs_C          $M NN_PRED=per_era_predictions_seeds.npz TREE_PRED=per_era_predictions_modxgb_eta0.03.npz OUT_TAG=_shared $PY analysis_A_vs_C.py
step acc_pooled          $V NN_PRED_VT=per_era_predictions_seeds_lr3e-3.npz NN_PRED_MORT=per_era_predictions_seeds.npz TREE_PRED=per_era_predictions_modxgb_eta0.03.npz OUT_TAG=_shared $PY analysis_pooled.py
step retrieval           $V TREE_SUFFIX=_eta0.03 NN_SUFFIX=_lr3e-3 $PYR script36_retrieval.py
step retrieval           $M TREE_SUFFIX=_eta0.03 NN_SUFFIX=_bc_states $PYR script36_retrieval.py
# ── compact summary of every step's key lines ──
{ echo; echo "===== KEY RESULTS ====="
  for d in $V $M; do echo "### $(basename $d)"
    for f in drift_detector trees_eta0.03 tree_update_expl tree_expl_stability nn_update_expl acc_A_vs_C acc_pooled retrieval; do
      [ -f $d/logs/$f.log ] || continue; echo "--- $f"
      case $f in
        drift_detector) grep -E "check|calibration|step [0-9]|STOP" $d/logs/$f.log ;;
        trees_eta0.03) grep -E "eta sweep|gain vs A|B_xgb − C_xgb|Saved → modxgb|STOP" $d/logs/$f.log | grep -v "probe" ;;
        tree_update_expl|nn_update_expl) grep -vE "^\[constants\]|Saved|^$" $d/logs/$f.log ;;
        tree_expl_stability) grep -E "check|2020 - 2022 +rho_(all|physio|treat)|floor|undefined" $d/logs/$f.log ;;
        acc_A_vs_C) grep -E "^(neural|trees)" $d/logs/$f.log | cut -c1-150 ;;
        acc_pooled) grep -E "\.both\.(D|R) |R\(intub\)-R\(vaso\)\.both|R\(intub\)-R\(mort\)\.both|subjects" $d/logs/$f.log | head -20 ;;
        retrieval) grep -E "check|Jaccard|Saved" $d/logs/$f.log ;;
      esac; done; done; } >> $S
echo "$(date +%T) ALL DONE" >> $S
