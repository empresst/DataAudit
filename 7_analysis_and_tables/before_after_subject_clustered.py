#!/usr/bin/env python3
"""
before_after_subject_clustered.py

Subject-clustered bootstrap for the before/after audit table (TMLR Table 2),
using ONLY saved predictions. No model is trained here.

Inputs (all local, none redistributed):
  OLD npz : per_era_predictions.npz of the earlier (pre-audit) pipeline
  OLD par : the pre-clean *_final_enriched13Aug.parquet files (stay -> subject)
  NEW par : the clean *_final_enriched.parquet files (audited labels/gates)

For each (era, label) cell it computes B-C under two scoring scenarios
(matching before_after_audit.py exactly):

  Before (submitted)   : old stays, old labels, no filtering
  Data only (audited)  : stays restricted to the clean cohort, labels and
                         incident gates looked up from the clean table

and reports, per scenario:
  - the point estimate (sanity-checked against table_results_before_after.csv),
  - the original stratified-stay bootstrap 95% CI (sanity check),
  - a subject-clustered bootstrap 95% CI (subjects resampled with replacement;
    all stays of a sampled subject move together),
plus a PAIRED subject-clustered CI for the change in B-C between the two
scenarios (same subject draw evaluated under both scorings).

Outputs: table_before_after_clustered.csv (next to this script) and stdout.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
OPUS = HERE.parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--old-npz", default=str(OPUS / "per_era_predictions.npz"))
ap.add_argument("--old-data", default=str(OPUS / "cocacola2/0_data_and_audit/raw_13aug"))
ap.add_argument("--new-data", default=str(OPUS / "ml4h_clean_pilot/data"))
ap.add_argument("--csv-ref", default=str(
    OPUS / "ml4h_clean_v2/short_paper/audit_compare/table_results_before_after.csv"))
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(HERE))
a = ap.parse_args()
OUT = Path(a.out)

SPLITS = ["train", "val", "test"]
ERAS = [("2014 - 2016", "2014_2016"), ("2017 - 2019", "2017_2019"),
        ("2020 - 2022", "2020_2022")]
LABELS = ["vasopressor", "intubation"]
ARMS = ["B", "C"]


def auroc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def t0(path, cols):
    return (pl.read_parquet(path, columns=cols + ["hrs_from_admit"])
            .filter(pl.col("hrs_from_admit") == 0).drop("hrs_from_admit")
            .unique("stay_id").to_pandas())


def strat_boot(y, p1, p2, nb, rng):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    d = []
    for _ in range(nb):
        ix = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(auroc(y[ix], p1[ix]) - auroc(y[ix], p2[ix]))
    d = np.asarray(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def cluster_ci(y, p1, p2, subjects, nb, rng):
    uniq = pd.unique(subjects)
    groups = {s: np.where(subjects == s)[0] for s in uniq}
    d = []
    for _ in range(nb):
        draw = rng.choice(uniq, len(uniq), replace=True)
        ix = np.concatenate([groups[s] for s in draw])
        v = auroc(y[ix], p1[ix]) - auroc(y[ix], p2[ix])
        if not np.isnan(v):
            d.append(v)
    d = np.asarray(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# --------------------------------------------------------------- load data
old = pd.concat([t0(f"{a.old_data}/{s}_final_enriched13Aug.parquet",
                    ["stay_id", "subject_id"]) for s in SPLITS], ignore_index=True)
new = pd.concat([t0(f"{a.new_data}/{s}_final_enriched.parquet",
                    ["stay_id", "subject_id", "label_vasopressor", "label_intubation",
                     "already_vaso", "already_vent"]) for s in SPLITS], ignore_index=True)
old_sid = old.set_index("stay_id")["subject_id"]
new_ix = new.set_index("stay_id")
new_ids = set(new.stay_id)

z = np.load(a.old_npz, allow_pickle=True)
rng = np.random.default_rng(a.seed)

rows = []
for era, key in ERAS:
    ids = z[f"stayids__{key}"]
    sid_all = old_sid.reindex(ids).to_numpy()
    unmapped = int(pd.isna(sid_all).sum())
    probs = {m: z[f"probs__{key}__{m}"] for m in ARMS}
    y_old = z[f"labels__{key}"][:, :2]
    av_old, ae_old = z[f"already_vaso__{key}"], z[f"already_vent__{key}"]

    keep = np.isin(ids, list(new_ids))
    y_new = np.stack([new_ix["label_vasopressor"].reindex(ids).to_numpy(),
                      new_ix["label_intubation"].reindex(ids).to_numpy()], 1)
    av_new = new_ix["already_vaso"].reindex(ids).to_numpy()
    ae_new = new_ix["already_vent"].reindex(ids).to_numpy()

    for li, lab in enumerate(LABELS):
        # full-length aligned arrays over the shared mapped stay universe;
        # each scenario has its own incident/inclusion mask
        scen = {}
        for name, y_mat, av, ae, filt in [
                ("Before (submitted)", y_old, av_old, ae_old, np.ones(len(ids), bool)),
                ("Data only", y_new, av_new, ae_new, keep)]:
            gate = av if lab == "vasopressor" else ae
            # same evaluation set as before_after_audit.py: both labels present
            # and incident only (not already on treatment in hours 0-5)
            inc = filt & ~np.isnan(y_mat).any(1) & (gate == 0)
            yf = y_mat[:, li].astype(float)
            yf[~inc] = np.nan
            scen[name] = (yf, probs["B"][:, li].copy(), probs["C"][:, li].copy(), inc)

        # marginal estimates + CIs per scenario (on that scenario's own stays)
        rec = {"era": era, "label": lab, "unmapped_stays": unmapped}
        for name, (yf, p1f, p2f, inc) in scen.items():
            j = np.where(inc)[0]
            y, p1, p2, sid = yf[j], p1f[j], p2f[j], sid_all[j]
            est = auroc(y, p1) - auroc(y, p2)
            s_lo, s_hi = strat_boot(y, p1, p2, 1000, np.random.default_rng(0))
            c_lo, c_hi = cluster_ci(y, p1, p2, sid, a.n_boot, rng)
            rec[f"{name}|est"] = round(est, 4)
            rec[f"{name}|stay_lo"] = round(s_lo, 4)
            rec[f"{name}|stay_hi"] = round(s_hi, 4)
            rec[f"{name}|cl_lo"] = round(c_lo, 4)
            rec[f"{name}|cl_hi"] = round(c_hi, 4)
            rec[f"{name}|n_stays"] = int(len(y))
            rec[f"{name}|n_events"] = int(y.sum())
            rec[f"{name}|n_subjects"] = int(len(pd.unique(sid)))

        # paired subject-clustered CI for the change (Data only - Before):
        # one subject draw over the shared universe, applied to both scenarios
        uniq = pd.unique(sid_all)
        groups = {s: np.where(sid_all == s)[0] for s in uniq}
        d_delta = []
        for _ in range(a.n_boot):
            draw = rng.choice(uniq, len(uniq), replace=True)
            ix = np.concatenate([groups[s] for s in draw])
            vals = {}
            ok_draw = True
            for name, (yf, p1f, p2f, inc) in scen.items():
                j = ix[inc[ix]]
                v = auroc(yf[j], p1f[j]) - auroc(yf[j], p2f[j])
                if np.isnan(v):
                    ok_draw = False
                    break
                vals[name] = v
            if ok_draw:
                d_delta.append(vals["Data only"] - vals["Before (submitted)"])
        d_delta = np.asarray(d_delta)
        rec["delta|est"] = round(float(np.mean(d_delta)), 4)
        rec["delta|lo"] = round(float(np.percentile(d_delta, 2.5)), 4)
        rec["delta|hi"] = round(float(np.percentile(d_delta, 97.5)), 4)
        rows.append(rec)

df = pd.DataFrame(rows)

# ------------------------------------------------- sanity check vs the CSV
ref = pd.read_csv(a.csv_ref)
ref = ref[ref.scenario.isin(["Before (submitted)", "Before, clean cohort+labels"])]
print("SANITY CHECK (stratified-stay bootstrap, seed 0, 1000 resamples)")
worst = 0.0
for _, r in ref.iterrows():
    if r.era not in [e for e, _ in ERAS]:
        continue
    scen = "Before (submitted)" if r.scenario == "Before (submitted)" else "Data only"
    m = df[(df.era == r.era) & (df.label == r.label)].iloc[0]
    de = abs(m[f"{scen}|est"] - r["B-C"])
    dl = abs(m[f"{scen}|stay_lo"] - r["B-C_lo95"]) if not pd.isna(r["B-C_lo95"]) else 0.0
    dh = abs(m[f"{scen}|stay_hi"] - r["B-C_hi95"]) if not pd.isna(r["B-C_hi95"]) else 0.0
    worst = max(worst, de, dl, dh)
    print(f"  {r.era:12s} {r.label:12s} {scen:20s} "
          f"est {de:.4f}  lo {dl:.4f}  hi {dh:.4f}")
print(f"  worst deviation vs CSV: {worst:.4f} "
      f"({'PASS' if worst < 0.0015 else 'FAIL - INVESTIGATE'})")

df.to_csv(OUT / "table_before_after_clustered.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 60)
print("\nSUBJECT-CLUSTERED RESULTS")
cols = ["era", "label",
        "Before (submitted)|est", "Before (submitted)|cl_lo", "Before (submitted)|cl_hi",
        "Data only|est", "Data only|cl_lo", "Data only|cl_hi",
        "delta|est", "delta|lo", "delta|hi", "Data only|n_subjects"]
print(df[cols].to_string(index=False))
