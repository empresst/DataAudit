#!/usr/bin/env python3
"""
before_after_audit.py

Before-audit (ML4H "Adapting the Wrong Component" submission) vs after-audit
(clean_v2) comparison. Answers three questions with data, then builds a
side-by-side results table.

  Q1  Are label definitions and prediction windows identical?
  Q2  Does the era assignment differ, and does it change the cohorts?
  Q3  Did the OLD pipeline really contain the empty-table stays and the old
      vasopressor ItemIDs (so the audit bundle describes what it claims)?

Inputs (all local, none are redistributed):
  OLD parquets : the pre-clean *_final_enriched13Aug.parquet files
  NEW parquets : the clean *_final_enriched.parquet files
  FLAGS        : stay_flags.parquet (make_flags.py output)
  OLD / NEW npz: per_era_predictions.npz of the submission / the clean run

Outputs (written next to this script): checks.json, table_*.csv, table_*.md
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
OPUS = HERE.parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--old-data", default=str(OPUS / "cocacola2/0_data_and_audit/raw_13aug"))
ap.add_argument("--new-data", default=str(OPUS / "ml4h_clean_pilot/data"))
ap.add_argument("--flags", default=str(OPUS / "cocacola2/data/stay_flags.parquet"))
ap.add_argument("--old-npz", default=str(OPUS / "per_era_predictions.npz"))
ap.add_argument("--new-npz", default=str(OPUS / "ml4h_clean_v2/per_era_predictions.npz"))
ap.add_argument("--old-code", default=str(
    OPUS / "downloadedNow/16_Adapting_the_Wrong_Componen_Supplementary Material/ml4h_supplement"))
ap.add_argument("--new-code", default=str(OPUS / "clean_github_repo"))
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--out", default=str(HERE))
a = ap.parse_args()
OUT = Path(a.out)
SPLITS = ["train", "val", "test"]
ERAS = ["2014 - 2016", "2017 - 2019", "2020 - 2022"]
ARMS = ["A", "B", "B_mirror", "C", "C_l2sp"]
LABELS = ["vasopressor", "intubation"]          # column order of the npz labels
NI_MARGIN = 0.015


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def t0(path, cols):
    return (pl.read_parquet(path, columns=cols + ["hrs_from_admit"])
            .filter(pl.col("hrs_from_admit") == 0).drop("hrs_from_admit")
            .unique("stay_id").to_pandas())


# ---------------------------------------------------------------- load stays
COLS = ["stay_id", "subject_id", "anchor_year_group", "anchor_year_group_raw", "approx_year",
        "label_vasopressor", "label_intubation", "already_vaso", "already_vent"]
old = pd.concat([t0(f"{a.old_data}/{s}_final_enriched13Aug.parquet", COLS).assign(split=s)
                 for s in SPLITS], ignore_index=True)
new = pd.concat([t0(f"{a.new_data}/{s}_final_enriched.parquet", COLS).assign(split=s)
                 for s in SPLITS], ignore_index=True)
flags = pl.read_parquet(a.flags).select(
    "stay_id", "has_input_data", "label_vaso_old", "already_vaso_old",
    "label_vaso_ext", "already_vaso_ext").to_pandas()
old = old.merge(flags, on="stay_id", how="left")
checks = {}

# ================================================================== Q1 labels
j = old.merge(new, on="stay_id", suffixes=("", "_n"))
q1 = {
    "script1_preprocessing_md5_old": md5(Path(a.old_code) / "script1_preprocessing.py"),
    "script1_preprocessing_md5_new": md5(Path(a.new_code) / "0_data_and_audit/upstream_kaggle/script1_preprocessing.py"),
    "stays_old": int(len(old)), "stays_new": int(len(new)), "stays_in_both": int(len(j)),
}
q1["script1_byte_identical"] = q1["script1_preprocessing_md5_old"] == q1["script1_preprocessing_md5_new"]
for c in ["label_intubation", "already_vent", "label_vasopressor", "already_vaso"]:
    q1[f"stays_differing_{c}"] = int((j[c] != j[c + "_n"]).sum())
# old-ID recompute with the audit's make_flags windows (label h8-14 overlap,
# already <h6) must reproduce the OLD parquet exactly -> same windows
q1["make_flags_old_ids_reproduce_old_label_vasopressor"] = float((old.label_vaso_old == old.label_vasopressor).mean())
q1["make_flags_old_ids_reproduce_old_already_vaso"] = float((old.already_vaso_old == old.already_vaso).mean())
# new parquets carry exactly the extended-ID flags
nn = new.merge(flags, on="stay_id", how="left")
q1["new_label_vasopressor_equals_ext_flag"] = float((nn.label_vasopressor == nn.label_vaso_ext).mean())
q1["new_already_vaso_equals_ext_flag"] = float((nn.already_vaso == nn.already_vaso_ext).mean())
checks["Q1_labels_and_windows"] = q1

# ================================================================== Q2 eras
q2 = {"stays_differing_anchor_year_group": int((j.anchor_year_group != j.anchor_year_group_n).sum()),
      "stays_differing_anchor_year_group_raw": int((j.anchor_year_group_raw != j.anchor_year_group_raw_n).sum()),
      "stays_differing_approx_year": int((j.approx_year != j.approx_year_n).sum()),
      "stays_differing_subject_id": int((j.subject_id != j.subject_id_n).sum())}
# the audit's "era assignment" contrast: MIMIC raw anchor-year group vs reconstructed year
test_old = old[old.split == "test"]
q2["share_of_test_stays_whose_raw_group_differs_from_reconstructed_era_OLD"] = float(
    (test_old.anchor_year_group_raw != test_old.anchor_year_group).mean())
q2["old_pipeline_era_is_reconstructed"] = True  # parquet column anchor_year_group == f(approx_year)
bins = pd.cut(old.approx_year, [0, 2010, 2013, 2016, 2019, 3000],
              labels=["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"])
q2["anchor_year_group_equals_binned_approx_year_OLD"] = float((bins.astype(str) == old.anchor_year_group).mean())
# subject-level split: identical role per subject? (subjects present in both)
checks["Q2_era_assignment"] = q2

# ================================================================== Q3 old pipeline artefacts
q3 = {}
old["era"] = old.anchor_year_group
rows = []
for e, g in old.groupby("era"):
    rows.append({"era": e, "stays_old_pipeline": len(g),
                 "no_icu_table_rows": int((g.has_input_data == 0).sum()),
                 "pct_no_icu_table_rows": round(100 * (g.has_input_data == 0).mean(), 1),
                 "vaso_label_rate_old_ids_pct": round(100 * g.label_vasopressor.mean(), 2),
                 "vaso_label_rate_ext_ids_pct": round(100 * g.label_vaso_ext.mean(), 2),
                 "vaso_pos_among_empty": int(g[g.has_input_data == 0].label_vasopressor.sum()),
                 "stays_label_flips_by_ids": int((g.label_vasopressor != g.label_vaso_ext).sum())})
t_q3 = pd.DataFrame(rows)
t_q3.to_csv(OUT / "table_old_pipeline_artifacts.csv", index=False)
q3["old_parquet_label_equals_old_id_recompute"] = q1["make_flags_old_ids_reproduce_old_label_vasopressor"]
q3["old_pipeline_stays_with_no_icu_table_rows"] = int((old.has_input_data == 0).sum())
q3["old_pipeline_stays_with_no_icu_table_rows_and_vaso_positive"] = int(
    ((old.has_input_data == 0) & (old.label_vasopressor == 1)).sum())
q3["removed_by_clean_build"] = int(len(old) - len(new))
q3["removed_equals_empty_table_stays"] = int(len(old) - len(new)) == q3["old_pipeline_stays_with_no_icu_table_rows"]
q3["new_stays_with_no_icu_table_rows"] = int((nn.has_input_data == 0).sum())
checks["Q3_old_pipeline_artifacts"] = q3

# ================================================================== cohort table
npz_o, npz_n = np.load(a.old_npz, allow_pickle=True), np.load(a.new_npz, allow_pickle=True)
ek = {"source (val_calib)": "sourceval_calib", "2014 - 2016": "2014_2016",
      "2017 - 2019": "2017_2019", "2020 - 2022": "2020_2022"}
key = list(ek.items())


def cohort_stats(z, k, labs_override=None):
    ids = z[f"stayids__{k}"]
    y = z[f"labels__{k}"][:, :2] if labs_override is None else labs_override
    av, ae = z[f"already_vaso__{k}"], z[f"already_vent__{k}"]
    return ids, y, av, ae


# ----- helper: map a (stay_id -> value) frame onto npz stays
def lookup(df, ids, col):
    s = df.set_index("stay_id")[col]
    return s.reindex(ids).to_numpy()


def auroc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(p).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


rng = np.random.default_rng(0)


def boot_diff(y, p1, p2, nb):
    n = len(y); d = []
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(nb):
        ix = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(auroc(y[ix], p1[ix]) - auroc(y[ix], p2[ix]))
    d = np.array(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.percentile(d, 5))


def verdict(lo90):
    return "non-inferior" if lo90 > -NI_MARGIN else "inconclusive"


# scenarios: (name, npz, label source, stay filter)
new_ids = set(new.stay_id)
SCEN = [("Before (submitted)", npz_o, "old", False),
        ("Before, clean cohort", npz_o, "old", True),
        ("Before, clean cohort+labels", npz_o, "new", True),
        ("After (re-run)", npz_n, "new", True)]
rows_c, rows_r = [], []
for name, z, labsrc, filt in SCEN:
    for era, k in key:
        ids = z[f"stayids__{k}"]
        keep = np.isin(ids, list(new_ids)) if filt else np.ones(len(ids), bool)
        src_df = old if labsrc == "old" else new
        if labsrc == "old" and name == "Before (submitted)":
            y = z[f"labels__{k}"][:, :2]; av = z[f"already_vaso__{k}"]; ae = z[f"already_vent__{k}"]
        elif labsrc == "old":
            y = z[f"labels__{k}"][:, :2]; av = z[f"already_vaso__{k}"]; ae = z[f"already_vent__{k}"]
        elif name == "After (re-run)":
            y = z[f"labels__{k}"][:, :2]; av = z[f"already_vaso__{k}"]; ae = z[f"already_vent__{k}"]
        else:  # old predictions, new labels looked up by stay id
            y = np.stack([lookup(new, ids, "label_vasopressor"), lookup(new, ids, "label_intubation")], 1)
            av = lookup(new, ids, "already_vaso"); ae = lookup(new, ids, "already_vent")
        ok = keep & ~np.isnan(y).any(1)
        for li, lab in enumerate(LABELS):
            gate = (av if lab == "vasopressor" else ae)
            inc = ok & (gate == 0)
            yy = y[inc, li].astype(int)
            rows_c.append({"scenario": name, "era": era, "label": lab,
                           "eval_stays": int(ok.sum()), "incident_stays": int(inc.sum()),
                           "incident_events": int(yy.sum()),
                           "label_rate_pct": round(100 * float(np.nanmean(y[ok, li])), 2),
                           "already_on_treatment_pct": round(100 * float(np.mean(gate[ok] != 0)), 1)})
            rec = {"scenario": name, "era": era, "label": lab, "events": int(yy.sum())}
            P = {m: z[f"probs__{k}__{m}"][inc, li] for m in ARMS}
            for m in ARMS:
                rec[m] = round(auroc(yy, P[m]), 4)
            for other in ["C", "C_l2sp"]:
                lo, hi, lo90 = boot_diff(yy, P["B"], P[other], a.n_boot) if yy.sum() >= 30 else (np.nan,) * 3
                rec[f"B-{other}"] = round(rec["B"] - rec[other], 4)
                rec[f"B-{other}_lo95"] = round(lo, 4); rec[f"B-{other}_hi95"] = round(hi, 4)
                rec[f"B-{other}_verdict"] = verdict(lo90) if yy.sum() >= 30 else "underpowered"
            rows_r.append(rec)

coh = pd.DataFrame(rows_c); res = pd.DataFrame(rows_r)
coh.to_csv(OUT / "table_cohorts_before_after.csv", index=False)
res.to_csv(OUT / "table_results_before_after.csv", index=False)

# cohort overlap of the evaluation sets (old vs new run)
ov = []
for era, k in key:
    o, n_ = set(npz_o[f"stayids__{k}"].tolist()), set(npz_n[f"stayids__{k}"].tolist())
    ov.append({"era": era, "eval_stays_before": len(o), "eval_stays_after": len(n_),
               "shared": len(o & n_), "only_before": len(o - n_), "only_after": len(n_ - o),
               "only_before_with_no_icu_rows": int(old[old.stay_id.isin(o - n_)].has_input_data.eq(0).sum())})
pd.DataFrame(ov).to_csv(OUT / "table_eval_cohort_overlap.csv", index=False)
checks["eval_cohort_overlap"] = ov
(OUT / "checks.json").write_text(json.dumps(checks, indent=2, default=str))

pd.set_option("display.width", 250, "display.max_columns", 50)
print(json.dumps(checks, indent=2, default=str))
print("\nOLD-PIPELINE ARTEFACTS"); print(t_q3.to_string(index=False))
print("\nEVAL COHORT OVERLAP"); print(pd.DataFrame(ov).to_string(index=False))
print("\nCOHORTS"); print(coh.to_string(index=False))
print("\nRESULTS"); print(res.to_string(index=False))
