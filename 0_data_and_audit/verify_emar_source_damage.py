"""
verify_emar_source_damage.py — independent check of the eMAR scaling problem.

Scores the paper's never-updated SOURCE model (two_stream_models15Aug.pt) on each
evaluation era twice, with the ORIGINAL pipeline normalisation:
  as-is   eMAR-derived features as in the data
  zeroed  the 7 eMAR features set to 0 (their value in the training era)
If the near-zero training sd of n_distinct_meds (z up to 735 at test) damages the
model, zeroing should raise AUROC. Inference only; nothing is trained.
"""
import json
from pathlib import Path
import numpy as np, polars as pl, torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN, INCIDENT_GATE
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize, ICUDataset
from models.architectures import TwoStreamModel

EMAR = ["early_steroid", "early_antibiotic", "early_furosemide", "early_anticoagulant",
        "early_antihypertensive", "early_antiarrhythmic", "n_distinct_meds"]
S = json.loads(Path("eval_split.json").read_text())["sequential"]
tr = load_enriched_split(Path("data"), "train", SEQ_FEATURES, TREATMENT_FEATURES)
te = load_enriched_split(Path("data"), "test", SEQ_FEATURES, TREATMENT_FEATURES)
st = calculate_train_stats(tr, sorted(set(SEQ_FEATURES + TREATMENT_FEATURES)))   # original rule
print(f"n_distinct_meds training sd used by the pipeline: {st['n_distinct_meds'][1]:.4f}")
ck = torch.load("two_stream_models.pt", map_location="cpu")
m = TwoStreamModel(len(SEQ_FEATURES), len(TREATMENT_FEATURES), len(LABEL_COLS))
m.load_state_dict(ck["source"]); m.eval()

def score(df):
    ds = ICUDataset(normalize(df, st), SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    with torch.no_grad():
        p = np.vstack([torch.sigmoid(m(a, b)).numpy() for a, b, _ in DataLoader(ds, batch_size=512)])
    return ds, p

print(f"\n{'era':<14}{'label':<13}{'events':>7}{'as-is':>9}{'zeroed':>9}{'Δ':>9}   max z(n_distinct_meds)")
for e in [k for k in S["eval_eras"] if "source" not in k]:
    d = te.filter(pl.col("stay_id").is_in([int(x) for x in S["era_eval_stays"][e]]))
    dz = d.with_columns([pl.lit(0).cast(d[c].dtype).alias(c) for c in EMAR])
    zmax = float(((d["n_distinct_meds"].to_numpy() - st["n_distinct_meds"][0]) / st["n_distinct_meds"][1]).max())
    (ds, p), (_, pz) = score(d), score(dz)
    for i, l in enumerate(LABEL_COLS):
        g = ds.meta[INCIDENT_GATE[l]] <= 0
        y = ds.labels[g, i]
        a, b = roc_auc_score(y, p[g, i]), roc_auc_score(y, pz[g, i])
        print(f"{e:<14}{l.replace('label_',''):<13}{int(y.sum()):>7}{a:>9.4f}{b:>9.4f}{b-a:>+9.4f}   {zmax:.0f}")
