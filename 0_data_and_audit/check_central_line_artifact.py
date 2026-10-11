"""
check_central_line_artifact.py — does the central-line item-ID change matter?

has_central_line_obs uses procedure IDs 224263, 224267, 225199. From 2017,
224267 (Cordis/Introducer) is replaced by 229526 (MAC) and 229516 (Single Lumen),
so the feature undercounts in 2017-22. This script recomputes the feature with
script1's exact window (intime <= starttime < intime + 6 h), first with the
original IDs (must reproduce the parquet exactly) and then with the new IDs added,
and scores the saved models of script2 (source and the median-seed model of each
arm) on the evaluation cohorts both ways. Read-only: nothing is retrained.
"""
import glob
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN, INCIDENT_GATE
from utils.data_utils import load_enriched_split, normalize, ICUDataset
from models.architectures import TwoStreamModel

RAW_PATTERN = os.environ.get("MIMIC_ICU_PATTERN", "./raw_mimic/icu/{}.csv*")
OLD = [224263, 224267, 225199]
NEW = OLD + [229526, 229516]
FEAT = "has_central_line_obs"

pe_matches = glob.glob(RAW_PATTERN.format("procedureevents"))
pe_path = pe_matches[0] if pe_matches else "procedureevents.csv.gz"
icu_matches = glob.glob(RAW_PATTERN.format("icustays"))
icu_path = icu_matches[0] if icu_matches else "icustays.csv.gz"

pe = pl.read_csv(pe_path, columns=["stay_id", "itemid", "starttime"], try_parse_dates=True)
icu = pl.read_csv(icu_path, columns=["stay_id", "intime"], try_parse_dates=True)
win = pe.join(icu, on="stay_id").filter(
    (pl.col("starttime") >= pl.col("intime"))
    & (pl.col("starttime") < pl.col("intime") + pl.duration(hours=6)))
flag = lambda ids: set(win.filter(pl.col("itemid").is_in(ids))["stay_id"].unique().to_list())
S_OLD, S_NEW = flag(OLD), flag(NEW)

te = load_enriched_split(Path("data"), "test", SEQ_FEATURES, TREATMENT_FEATURES)
t0 = te.filter(pl.col("hrs_from_admit") == 0)
rec = t0["stay_id"].is_in(list(S_OLD)).cast(pl.Int64).to_numpy()
if not np.array_equal(rec, t0[FEAT].cast(pl.Int64).to_numpy()):
    raise SystemExit("recomputed feature (original IDs) does not reproduce the parquet — stop")
print("original-ID recomputation reproduces the parquet exactly")

ck = torch.load("two_stream_models.pt", map_location="cpu")
stats = ck["train_stats"]
seq = json.loads(Path("eval_split.json").read_text())["sequential"]
fixed = te.with_columns(pl.col("stay_id").is_in(list(S_NEW)).cast(te.schema[FEAT]).alias(FEAT))


def dataset(df, era):
    ids = [int(x) for x in seq["era_eval_stays"][era]]
    return ICUDataset(normalize(df.filter(pl.col("stay_id").is_in(ids)), stats),
                      SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)


@torch.no_grad()
def probs(state, ds):
    m = TwoStreamModel(ck["seq_dim"], ck["treat_dim"], ck["n_targets"])
    m.load_state_dict(state)
    m.eval()
    return np.vstack([torch.sigmoid(m(xs, xt)).numpy()
                      for xs, xt, _ in DataLoader(ds, batch_size=512, shuffle=False)])


models = {"source": ck["source"], "B": ck["run_b"], "C": ck["run_c"], "C_l2sp": ck["run_c_l2sp"]}
print(f"\n{'era':<13}{'label':<12}{'flipped':>8} | " + "  ".join(f"{a + ' Δ':>10}" for a in models)
      + "   (Δ = incident AUROC corrected − as stored)")
for era in [e for e in seq["eval_eras"] if not e.startswith("source")]:
    d0, d1 = dataset(te, era), dataset(fixed, era)
    assert np.array_equal(np.asarray(d0.stay_ids), np.asarray(d1.stay_ids))
    p0 = {a: probs(s, d0) for a, s in models.items()}
    p1 = {a: probs(s, d1) for a, s in models.items()}
    sid = np.asarray(d0.stay_ids)
    flipped = int(np.isin(sid, list(S_NEW - S_OLD)).sum())
    for i, lbl in enumerate(LABEL_COLS):
        m = np.asarray(d0.meta[INCIDENT_GATE[lbl]]) <= 0
        y = d0.labels[m, i]
        dl = {a: roc_auc_score(y, p1[a][m, i]) - roc_auc_score(y, p0[a][m, i]) for a in models}
        print(f"{era:<13}{lbl.replace('label_', ''):<12}{flipped:>8} | "
              + "  ".join(f"{dl[a]:>+10.4f}" for a in models))
