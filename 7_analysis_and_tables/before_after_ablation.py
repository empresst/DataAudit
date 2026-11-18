#!/usr/bin/env python3
"""
before_after_ablation.py

Separates the two audit corrections behind the 2020-22 vasopressor reversal
(MIDI Section 4.1), using ONLY saved predictions of the earlier (pre-audit)
pipeline. No model is trained here.

  before            old labels and gates, all stays            (+0.051)
  exclusion only    old labels, stays with ICU input/procedure records only
  relabelling only  extended-ItemID labels and gates, all stays
  both              extended labels, covered stays only      (-0.030)

Point estimates of B-C incident AUROC; events counted after the incident gate.

Inputs (all local, none redistributed):
  OLD npz : per_era_predictions.npz of the earlier pipeline
  FLAGS   : stay_flags.parquet (make_flags.py output: has_input_data,
            label_vaso_ext, already_vaso_ext)

Outputs: table_before_after_ablation.csv (next to this script) and stdout.
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
ap.add_argument("--old-npz", default=str(OPUS / "per_era_predictions.npz"))
ap.add_argument("--flags", default=str(OPUS / "cocacola2/0_data_and_audit/stay_flags.parquet"))
ap.add_argument("--era", default="2020_2022")
ap.add_argument("--out", default=str(HERE))
a = ap.parse_args()

z = np.load(a.old_npz, allow_pickle=True)
f = pl.read_parquet(a.flags).to_pandas().set_index("stay_id")
k = a.era
ids = z[f"stayids__{k}"]
assert not f.label_vaso_ext.reindex(ids).isna().any(), "stays missing from stay_flags"

pB, pC = z[f"probs__{k}__B"][:, 0], z[f"probs__{k}__C"][:, 0]      # column 0 = vasopressor
y_old, g_old = z[f"labels__{k}"][:, 0], z[f"already_vaso__{k}"]
y_ext = f.label_vaso_ext.reindex(ids).to_numpy()
g_ext = f.already_vaso_ext.reindex(ids).to_numpy()
covered = f.has_input_data.reindex(ids).to_numpy() == 1
ok = ~np.isnan(z[f"labels__{k}"][:, :2]).any(1)                     # as before_after_audit.py

rows = []
for name, y, g, keep in [("before", y_old, g_old, ok),
                         ("exclusion only", y_old, g_old, ok & covered),
                         ("relabelling only", y_ext, g_ext, ok),
                         ("both", y_ext, g_ext, ok & covered)]:
    m = keep & (g == 0)
    yy = y[m].astype(int)
    rows.append({"scenario": name, "era": k.replace("_", " - "), "stays": int(m.sum()),
                 "events": int(yy.sum()),
                 "B_minus_C": round(roc_auc_score(yy, pB[m]) - roc_auc_score(yy, pC[m]), 4)})

df = pd.DataFrame(rows)
df.to_csv(Path(a.out) / "table_before_after_ablation.csv", index=False)
print(df.to_string(index=False))
