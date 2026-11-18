#!/usr/bin/env python3
"""
per_cell_subject_clustered.py

Subject-clustered bootstrap for the audited per-cell B-C table (MIDI Table 3),
using ONLY saved predictions. No model is trained here.

The estimator is the one in script17_seed_floor.paired_patient_boot: the
seed-averaged incident AUROC difference B-C over the 5 seeds. That function
resamples STAYS; here SUBJECTS are resampled with replacement and all of a
sampled subject's stays move together. The stay-level bootstrap is re-run as
a sanity check against the logged intervals.

Inputs (all local, none redistributed):
  NN  npz : per_era_predictions_seeds_lr3e-3.npz   (neural, shared lr 3e-3)
  XGB npz : per_era_predictions_modxgb_eta0.03.npz (trees, shared eta 0.03)
  DATA    : the clean *_final_enriched.parquet files (stay -> subject, gates)

Outputs: table_per_cell_clustered.csv (next to this script) and stdout.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
OPUS = HERE.parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--nn-npz", default=str(OPUS / "ml4h_clean_v2/per_era_predictions_seeds_lr3e-3.npz"))
ap.add_argument("--xgb-npz", default=str(OPUS / "ml4h_clean_v2/per_era_predictions_modxgb_eta0.03.npz"))
ap.add_argument("--data", default=str(OPUS / "ml4h_clean_v2/data"))
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--out", default=str(HERE))
a = ap.parse_args()

SPLITS = ["train", "val", "test"]
ERAS = [("2014 - 2016", "2014_2016"), ("2017 - 2019", "2017_2019"),
        ("2020 - 2022", "2020_2022")]
LABELS = ["vasopressor", "intubation"]
GATE = {"vasopressor": "already_vaso", "intubation": "already_vent"}
SEEDS = [42, 123, 7, 2024, 99]
NI_MARGIN = 0.015


def t0(path, cols):
    return (pl.read_parquet(path, columns=cols + ["hrs_from_admit"])
            .filter(pl.col("hrs_from_admit") == 0).drop("hrs_from_admit")
            .unique("stay_id").to_pandas())


def seed_diff(y, pa, pb, ix):
    yy = y[ix]
    if not (0 < yy.sum() < len(yy)):
        return np.nan
    return float(np.mean([roc_auc_score(yy, p1[ix]) - roc_auc_score(yy, p2[ix])
                          for p1, p2 in zip(pa, pb)]))


def summarise(bs):
    bs = np.asarray([v for v in bs if not np.isnan(v)])
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)),
            float(np.percentile(bs, 5)))


def verdict(lo95, lo90):
    return "sup." if lo95 > 0 else ("NI" if lo90 > -NI_MARGIN else "--")


meta = pd.concat([t0(f"{a.data}/{s}_final_enriched.parquet",
                     ["stay_id", "subject_id", "already_vaso", "already_vent"])
                  for s in SPLITS], ignore_index=True).set_index("stay_id")
nn, xg = np.load(a.nn_npz, allow_pickle=True), np.load(a.xgb_npz, allow_pickle=True)

rows = []
for era, k in ERAS:
    ids = nn[f"stayids__{k}"]
    assert np.array_equal(ids, xg[f"stayids__{k}"]), f"stay order differs in {k}"
    sid = meta["subject_id"].reindex(ids).to_numpy()
    assert not pd.isna(sid).any(), f"unmapped stays in {k}"
    for li, lab in enumerate(LABELS):
        gate = meta[GATE[lab]].reindex(ids).to_numpy()
        m = gate <= 0
        y = nn[f"labels__{k}"][m, li].astype(int)
        s = sid[m]
        uniq = pd.unique(s)
        groups = {u: np.where(s == u)[0] for u in uniq}
        for fam, z, b, c in [("Neural", nn, "B", "C"), ("Trees", xg, "B_xgb", "C_xgb")]:
            pa = np.stack([z[f"probs__{k}__{b}__seed{sd}"][m, li] for sd in SEEDS])
            pb = np.stack([z[f"probs__{k}__{c}__seed{sd}"][m, li] for sd in SEEDS])
            allix = np.arange(len(y))
            est = seed_diff(y, pa, pb, allix)
            # stay-level (as logged): seed 11, stays resampled
            rng = np.random.default_rng(11)
            st = summarise([seed_diff(y, pa, pb, rng.integers(0, len(y), len(y)))
                            for _ in range(a.n_boot)])
            # subject-clustered
            rng = np.random.default_rng(11)
            cl = summarise([seed_diff(y, pa, pb, np.concatenate(
                [groups[u] for u in rng.choice(uniq, len(uniq), replace=True)]))
                for _ in range(a.n_boot)])
            rows.append({"label": lab, "model": fam, "era": era,
                         "n_stays": len(y), "n_events": int(y.sum()), "n_subjects": len(uniq),
                         "est": round(est, 4),
                         "stay_lo95": round(st[0], 4), "stay_hi95": round(st[1], 4),
                         "stay_lo90": round(st[2], 4), "stay_verdict": verdict(st[0], st[2]),
                         "cl_lo95": round(cl[0], 4), "cl_hi95": round(cl[1], 4),
                         "cl_lo90": round(cl[2], 4), "cl_verdict": verdict(cl[0], cl[2])})
            print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv(Path(a.out) / "table_per_cell_clustered.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(df.to_string(index=False))
print(f"\nNI/sup. cells: stay-level {(df.stay_verdict != '--').sum()}/12, "
      f"subject-clustered {(df.cl_verdict != '--').sum()}/12")
