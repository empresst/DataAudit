"""
analysis_A_vs_C.py — does the non-inferiority margin mean anything?

The short paper judges the bounded update B against the full update C with a margin
of 0.015 AUROC. The whole gain of a full update over the never-updated model A is
0.006-0.027 AUROC, i.e. of the same size as the margin. If A itself is "non-inferior"
to C at 0.015, the test cannot tell a useful update from no update.

For every powered cell (>= 50 events, test eras only) and model family this computes,
with ONE set of patient-level bootstrap resamples shared by all arms and seed-averaged
AUROC (same estimator as script17's paired_patient_boot):

  gain_C   = C - A                       what a full update buys
  gain_B   = B - A                       what the bounded update buys
  A_vs_C   = A - C, with the one-sided 5% non-inferiority test at the margin
  B_vs_C   = B - C, same test (reproduces the paper's numbers)
  retained = gain_B / gain_C             fraction of the full update's gain kept by B
             (reported with a CI; flagged unstable when the CI of gain_C includes 0)

Reads saved predictions only; trains nothing. Run inside a chain directory
(ml4h_clean_v2 or ml4h_clean_v2_mortality). Writes A_vs_C.json and A_vs_C.csv.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, ".")
from utils.constants import LABEL_COLS, INCIDENT_GATE  # noqa: E402

N_BOOT = 1000
import os
NN_PRED = os.environ.get("NN_PRED", "per_era_predictions_seeds.npz")
TREE_PRED = os.environ.get("TREE_PRED", "per_era_predictions_modxgb.npz")
TAG = os.environ.get("OUT_TAG", "")
MARGIN = json.loads(Path("seed_floor.json").read_text())["ni_margin"]
SEEDS = json.loads(Path("seed_floor.json").read_text())["seeds"]
ERAS = ["2014 - 2016", "2017 - 2019", "2020 - 2022"]
MIN_EVENTS = 50
NAME = {"label_vasopressor": "Vasopressor", "label_intubation": "Intubation",
        "label_mortality": "Mortality"}


def key(e):
    return e.replace(" ", "").replace("-", "_").replace("(", "").replace(")", "")


def auc(y, p):
    """Rank AUROC with average ranks for ties."""
    r = rankdata(p)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def seed_mean_auc(y, P):
    return float(np.mean([auc(y, p) for p in P]))


def pct(x, q):
    return float(np.percentile(x, q))


def analyse(family, arms, A_key, B_key, C_key, S, ref, rng_seed=11):
    """arms: callable(arm, k) -> (n_seeds_or_1, n_all) probabilities for that era."""
    out = []
    for lbl in LABEL_COLS:
        i = LABEL_COLS.index(lbl)
        for era in ERAS:
            k = key(era)
            gate = INCIDENT_GATE[lbl]
            lab_all = ref[f"labels__{k}"][:, i]
            m = (np.ones(len(lab_all), bool) if gate is None
                 else np.asarray(ref[f"{gate}__{k}"]) <= 0)
            y = lab_all[m].astype(int)
            if y.sum() < MIN_EVENTS:
                continue
            P = {a: arms(a, k)[:, m, i] for a in (A_key, B_key, C_key)}   # (seeds, n_incident)
            pt = {a: seed_mean_auc(y, P[a]) for a in P}
            rng = np.random.default_rng(rng_seed)
            n = len(y)
            bs = {a: [] for a in P}
            for _ in range(N_BOOT):
                ix = rng.integers(0, n, n)
                yy = y[ix]
                if not (0 < yy.sum() < n):
                    continue
                for a in P:
                    bs[a].append(seed_mean_auc(yy, P[a][:, ix]))
            bs = {a: np.asarray(v) for a, v in bs.items()}
            gC, gB = bs[C_key] - bs[A_key], bs[B_key] - bs[A_key]
            BmC, AmC = bs[B_key] - bs[C_key], bs[A_key] - bs[C_key]
            ret = gB / gC
            stable = pct(gC, 2.5) > 0
            row = dict(
                family=family, label=NAME[lbl], era=era, events=int(y.sum()), n=n,
                A=pt[A_key], B=pt[B_key], C=pt[C_key],
                gain_C=pt[C_key] - pt[A_key], gain_C_ci95=[pct(gC, 2.5), pct(gC, 97.5)],
                gain_B=pt[B_key] - pt[A_key], gain_B_ci95=[pct(gB, 2.5), pct(gB, 97.5)],
                A_minus_C=pt[A_key] - pt[C_key], A_minus_C_ci90lo=pct(AmC, 5),
                A_vs_C_noninferior=bool(pct(AmC, 5) > -MARGIN),
                B_minus_C=pt[B_key] - pt[C_key], B_minus_C_ci90lo=pct(BmC, 5),
                B_minus_C_ci95=[pct(BmC, 2.5), pct(BmC, 97.5)],
                B_vs_C_noninferior=bool(pct(BmC, 5) > -MARGIN),
                retained=(pt[B_key] - pt[A_key]) / (pt[C_key] - pt[A_key]),
                retained_ci95=[pct(ret, 2.5), pct(ret, 97.5)],
                retained_stable=bool(stable),
                gain_C_over_margin=(pt[C_key] - pt[A_key]) / MARGIN)
            out.append(row)
            print(f"{family:6s} {row['label']:11s} {era}  ev={row['events']:4d}  "
                  f"gain C {row['gain_C']:+.4f} [{row['gain_C_ci95'][0]:+.4f},{row['gain_C_ci95'][1]:+.4f}]  "
                  f"gain B {row['gain_B']:+.4f}  "
                  f"A-C {row['A_minus_C']:+.4f} lo90 {row['A_minus_C_ci90lo']:+.4f} "
                  f"{'A PASSES NI' if row['A_vs_C_noninferior'] else 'A fails NI':11s} | "
                  f"B-C {row['B_minus_C']:+.4f} lo90 {row['B_minus_C_ci90lo']:+.4f} "
                  f"{'B NI' if row['B_vs_C_noninferior'] else 'B --'} | "
                  f"retained {row['retained']:.2f} [{row['retained_ci95'][0]:.2f},{row['retained_ci95'][1]:.2f}]"
                  f"{'' if stable else '  (unstable)'}", flush=True)
    return out


def main():
    ref = np.load("per_era_predictions.npz", allow_pickle=True)
    rows = []

    # ── neural: A from script2, B and C per seed from script17 ────────────────
    S = np.load(NN_PRED, allow_pickle=True)

    def neural(arm, k):
        if not np.array_equal(S[f"stayids__{k}"], ref[f"stayids__{k}"]):
            raise SystemExit(f"stay order differs at {k}")
        if arm == "A":
            return ref[f"probs__{k}__A"][None]                  # (1, n, n_labels)
        return np.stack([S[f"probs__{k}__{arm}__seed{s}"] for s in SEEDS])

    rows += analyse("neural", neural, "A", "B", "C", S, ref)

    # ── modular trees: A_xgb, B_xgb, C_xgb per seed ──────────────────────────
    if Path(TREE_PRED).exists():
        T = np.load(TREE_PRED, allow_pickle=True)

        def trees(arm, k):
            if not np.array_equal(T[f"stayids__{k}"], ref[f"stayids__{k}"]):
                raise SystemExit(f"tree stay order differs at {k}")
            return np.stack([T[f"probs__{k}__{arm}__seed{s}"] for s in SEEDS])

        rows += analyse("trees", trees, "A_xgb", "B_xgb", "C_xgb", T, ref)

    Path(f"A_vs_C{TAG}.json").write_text(json.dumps(rows, indent=2))
    with open(f"A_vs_C{TAG}.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["family", "label", "era", "events", "A", "gain_C", "gain_B", "gain_C_over_margin",
                "A_minus_C", "A_minus_C_ci90lo", "A_vs_C_noninferior",
                "B_minus_C", "B_minus_C_ci90lo", "B_vs_C_noninferior",
                "retained", "retained_stable"]
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if not isinstance(r[c], float) else round(r[c], 4) for c in cols])
    print(f"\nSaved -> A_vs_C{TAG}.json/.csv  (NN {NN_PRED}, trees {TREE_PRED})")


if __name__ == "__main__":
    main()
