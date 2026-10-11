"""
script37_gated_schedule.py — make the drift detector the explicit GATE of the update schedule,
and verify that the saved forced-schedule results are exactly the gated-schedule results.

Why no retraining is needed
---------------------------
script35 decides each step from (i) feature distributions, (ii) the SOURCE models' predictions
(arc["source"], two_stream_models.pt["source"]) and (iii) the data of earlier adaptation cohorts.
It uses no labels and no updated model, so its decisions cannot depend on whether earlier updates
were run. If every step fires, "update at every era" (FORCE_ADAPT=1) and "update only when the
detector fires" produce the same sequence of models. The saved predictions are therefore the
gated-schedule predictions.

What this script does (trains nothing, runs in seconds)
  1. Re-derives each step's decision from the stored C2ST lower bound with the pre-set rule
     (lower 95% bound > AUC_MIN) and checks it equals the stored `fire` flag.
  2. Checks the detector's eras equal the update schedule's eras (eval_split.json) and that the
     detector's calibration (false-alarm rate <= 5%) passed.
  3. Checks, from script35's source text, that the detector never reads updated-model artefacts
     (a static check; fails if a non-source model is referenced).
  4. Replays the gate over every saved prediction file: for each fired step it selects that
     step's stored arm predictions; fails loudly if a step did NOT fire, because the no-update
     branch (carry the previous model forward) needs predictions that were never saved.
  5. Writes gated_schedule.json (decisions, rule, SHA-256 of every prediction file used).

Also importable by an update chain:
    from script37_gated_schedule import steps_to_update
    for era in steps_to_update(): ...      # eras whose update the detector approved

The no-update branch is NOT exercised by these data (all steps fired); do not claim it was tested.
Run from ml4h_clean_v2/.   Exit code 0 = equivalence verified, 3 = some step did not fire.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

AUC_MIN = 0.60                 # same practical-significance bar as script35
MAX_FALSE_ALARM = 0.05
HERE = Path(".")
OTHER = Path("../ml4h_clean_v2_mortality")
DETECTOR_JSON = "drift_detection_standard.json"
DETECTOR_SRC = "script35_drift_detector.py"

# (directory, prediction file, arms updated at every step, family)
PRED_FILES = [
    (HERE, "per_era_predictions_seeds_lr3e-3.npz", ("B", "C"), "neural vaso/intub"),
    (HERE, "per_era_predictions_modxgb_eta0.03.npz", ("B_xgb", "C_xgb"), "trees vaso/intub"),
    (OTHER, "per_era_predictions_seeds.npz", ("B", "C"), "neural mortality"),
    (OTHER, "per_era_predictions_modxgb_eta0.03.npz", ("B_xgb", "C_xgb"), "trees mortality"),
]


def stop(msg, code=1):
    print(f"STOP: {msg}")
    sys.exit(code)


def load_decisions(det_path=HERE / DETECTOR_JSON):
    """Per-step gate decisions, re-derived from the stored C2ST bound (not trusted from `fire`)."""
    det = json.loads(Path(det_path).read_text())
    if det["false_alarm_rate"] > MAX_FALSE_ALARM:
        stop(f"detector false-alarm rate {det['false_alarm_rate']:.1%} > {MAX_FALSE_ALARM:.0%}")
    out = []
    for s in det["steps"]:
        lo = s["c2st_all"]["lo95"]
        update = bool(lo > AUC_MIN)
        if update != bool(s["fire"]):
            stop(f"stored fire flag disagrees with the rule at {s['era']} (lo95 {lo:.3f})")
        out.append({"step": s["step"], "era": s["era"], "c2st_all_auroc": s["c2st_all"]["auroc"],
                    "c2st_all_lo95": lo, "update": update})
    return det, out


def steps_to_update(det_path=HERE / DETECTOR_JSON):
    """Eras whose update the detector approves, in schedule order (the gate for an update chain)."""
    return [d["era"] for d in load_decisions(det_path)[1] if d["update"]]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    det, dec = load_decisions()

    # 2. schedule eras == detector eras
    sched = json.loads((HERE / "eval_split.json").read_text())["sequential"]["adapt_steps"]
    if [d["era"] for d in dec] != list(sched):
        stop(f"detector eras {[d['era'] for d in dec]} != update schedule {sched}")
    if OTHER.exists():
        s2 = json.loads((OTHER / "eval_split.json").read_text())["sequential"]["adapt_steps"]
        if list(s2) != list(sched):
            stop("mortality run has a different update schedule")

    # 3. static independence check: the detector may use SOURCE models only
    src = (HERE / DETECTOR_SRC).read_text()
    if re.search(r'arc\[\s*["\']models["\']', src) or re.search(r"nn_states|per_era_predictions|seed_floor", src):
        stop("script35 references updated-model artefacts; its decisions may depend on earlier updates")
    if 'arc["source"]' not in src:
        stop("could not confirm script35 reads only source models")
    print("[check] detector reads source models only (no updated-model artefacts, no labels)")

    # 4. gated replay over every saved prediction file
    skipped = [d["era"] for d in dec if not d["update"]]
    if skipped:
        print(f"[gate] detector did NOT fire at: {skipped}")
        stop("the no-update branch needs carry-forward predictions that were never saved; "
             "rerun the update chain with these steps skipped", code=3)
    ekeys = {d["era"]: d["era"].replace(" - ", "_") for d in dec}
    files = []
    for d, fname, arms, fam in PRED_FILES:
        p = d / fname
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        z = np.load(p, allow_pickle=True)
        for d_ in dec:                                   # the gate: fired -> use this step's update
            for arm in arms:
                n = len([k for k in z.files if k.startswith(f"probs__{ekeys[d_['era']]}__{arm}__seed")])
                if n == 0:
                    stop(f"{p}: no stored predictions for arm {arm} at {d_['era']}")
        files.append({"path": str(p), "family": fam, "arms": list(arms), "sha256": sha256(p)})
        print(f"[replay] {fam:18s} all {len(dec)} gated steps present for arms {arms}  ({p.name})")
    if not files:
        stop("no prediction files found")

    # 5. record
    out = {"rule": det["rule"], "auc_min": AUC_MIN, "false_alarm_rate": det["false_alarm_rate"],
           "null_auroc_mean": det["null_auroc_mean"], "decisions": dec,
           "all_steps_fired": True,
           "conclusion": "Gated schedule == forced schedule: every step fired, so the same updates run in "
                         "the same order; saved predictions are the gated-schedule results.",
           "not_exercised": "no-update branch (no step failed to fire)",
           "files": files}
    Path("gated_schedule.json").write_text(json.dumps(out, indent=1))
    print("\nGate decisions:")
    for d_ in dec:
        print(f"  step {d_['step']} {d_['era']}: C2ST AUROC {d_['c2st_all_auroc']:.3f} "
              f"(lo95 {d_['c2st_all_lo95']:.3f} > {AUC_MIN}) -> UPDATE")
    print("\nVerified: forced schedule == gated schedule for all saved results. Saved → gated_schedule.json")


if __name__ == "__main__":
    main()
