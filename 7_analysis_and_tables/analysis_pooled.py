"""
analysis_pooled.py — pooled bounded-vs-full contrasts across eras and model families.

Per-cell intervals in the short paper are too wide to say whether the intubation
shortfall is real. This pools the three test eras (and the two model families) and
compares targets directly. Specified after the per-cell results and A_vs_C.py were
seen; report it as a secondary, post-hoc analysis.

Estimands, for label L and family F (neural two-stream; modular trees):
  D_F(L)  = sum_e w_e (B - C)_e / sum_e w_e            pooled AUROC difference
  G_F(L)  = same with (C - A)  and (B - A)              pooled gains
  R_F(L)  = G_B / G_C                                   fraction of the full update's
                                                        gain kept by the bounded update
  w_e     = incident events in era e (equal weights reported as a sensitivity check)
  Combined over families: unweighted mean of the neural and tree values.
Per-cell AUROC is the seed-averaged AUROC (script17's estimator); A_neural is the
single source model, A_xgb is per seed.

Contrasts between targets: R(intub) - R(vaso), R(intub) - R(mort), R(vaso) - R(mort).

Uncertainty: ONE subject-level cluster bootstrap (1,000 resamples) shared by every
label, era, family and arm. Subjects are resampled with replacement from the union
of the three evaluation eras; each drawn subject brings all of its stays in every
era. This respects readmissions within an era (~20% of stays) and the subjects who
appear in more than one era. Percentile 95% CIs; two-sided bootstrap p-values.

Reads saved predictions only. Run from ml4h_clean_v2/; mortality is read from
../ml4h_clean_v2_mortality/. Writes pooled.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

sys.path.insert(0, ".")
from utils.constants import INCIDENT_GATE  # noqa: E402

MORT = Path("../ml4h_clean_v2_mortality")
import os
NN_PRED = {".": os.environ.get("NN_PRED_VT", "per_era_predictions_seeds.npz"),
           str(MORT): os.environ.get("NN_PRED_MORT", "per_era_predictions_seeds.npz")}
TREE_PRED = os.environ.get("TREE_PRED", "per_era_predictions_modxgb.npz")
TAG = os.environ.get("OUT_TAG", "")
N_BOOT = 1000
SEED = 2026
ERAS = ["2014_2016", "2017_2019", "2020_2022"]
SEEDS = json.loads(Path("seed_floor.json").read_text())["seeds"]
ARMS = {"neural": ("A", "B", "C"), "trees": ("A_xgb", "B_xgb", "C_xgb")}


def auc(y, p):
    r = rankdata(p)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


# ── load everything into (label, family, era) -> {y, arm: (seeds, n)} ─────────
def load_dir(d, labels):
    ref = np.load(d / "per_era_predictions.npz", allow_pickle=True)
    S = np.load(d / NN_PRED[str(d)], allow_pickle=True)
    T = np.load(d / TREE_PRED, allow_pickle=True)
    return ref, S, T, labels


vt = load_dir(Path("."), ["label_vasopressor", "label_intubation"])
mo = load_dir(MORT, ["label_mortality"])
REF = vt[0]

subj_map = pl.read_parquet("data/test_final_enriched.parquet",
                           columns=["stay_id", "subject_id"]).unique()
subj_map = dict(zip(subj_map["stay_id"].to_list(), subj_map["subject_id"].to_list()))
all_subj = sorted({subj_map[int(s)] for e in ERAS for s in REF[f"stayids__{e}"]})
code = {s: i for i, s in enumerate(all_subj)}
N_SUBJ = len(all_subj)

CELLS = {}          # (label, family, era) -> dict(y, sc, P)
for ref, S, T, labels in (vt, mo):
    for i, lbl in enumerate(labels):
        for e in ERAS:
            st = ref[f"stayids__{e}"]
            if not np.array_equal(st, REF[f"stayids__{e}"]):
                raise SystemExit(f"stay order differs between runs at {e}")
            for z in (S, T):
                if not np.array_equal(z[f"stayids__{e}"], st):
                    raise SystemExit(f"stay order differs inside a run at {e}")
            gate = INCIDENT_GATE[lbl]
            m = (np.ones(len(st), bool) if gate is None
                 else np.asarray(REF[f"{gate}__{e}"]) <= 0)
            y = ref[f"labels__{e}"][m, i].astype(int)
            sc = np.array([code[subj_map[int(s)]] for s in st[m]])
            for fam, (a, b, c) in ARMS.items():
                src = S if fam == "neural" else T
                P = {}
                for arm in (a, b, c):
                    if arm == "A":
                        P["A"] = ref[f"probs__{e}__A"][m, i][None]
                    else:
                        P[arm[0]] = np.stack([src[f"probs__{e}__{arm}__seed{s}"][m, i]
                                              for s in SEEDS])
                CELLS[(lbl, fam, e)] = dict(y=y, sc=sc, P=P, events=int(y.sum()))

LABELS = ["label_vasopressor", "label_intubation", "label_mortality"]
SHORT = {"label_vasopressor": "vaso", "label_intubation": "intub", "label_mortality": "mort"}


def cell_aucs(c, ix=None):
    y = c["y"] if ix is None else c["y"][ix]
    if not (0 < y.sum() < len(y)):
        return None
    out = {}
    for arm, P in c["P"].items():
        PP = P if ix is None else P[:, ix]
        out[arm] = float(np.mean([auc(y, p) for p in PP]))
    return out


def summarise(per_cell, weights):
    """per_cell: (label, fam, era) -> {A,B,C}. Returns dict of pooled estimands."""
    r = {}
    for lbl in LABELS:
        fam_vals = {}
        for fam in ARMS:
            w = np.array([weights[(lbl, e)] for e in ERAS], float)
            w /= w.sum()
            a = np.array([[per_cell[(lbl, fam, e)][k] for e in ERAS] for k in "ABC"])
            D = float(w @ (a[1] - a[2]))
            GB = float(w @ (a[1] - a[0]))
            GC = float(w @ (a[2] - a[0]))
            fam_vals[fam] = dict(D=D, GB=GB, GC=GC, R=GB / GC if GC != 0 else np.nan)
            for k, v in fam_vals[fam].items():
                r[f"{SHORT[lbl]}.{fam}.{k}"] = v
        for k in ("D", "GB", "GC"):
            r[f"{SHORT[lbl]}.both.{k}"] = float(np.mean([fam_vals[f][k] for f in ARMS]))
        r[f"{SHORT[lbl]}.both.R"] = r[f"{SHORT[lbl]}.both.GB"] / r[f"{SHORT[lbl]}.both.GC"]
    for fam in list(ARMS) + ["both"]:
        for x, y in (("intub", "vaso"), ("intub", "mort"), ("vaso", "mort")):
            r[f"R({x})-R({y}).{fam}"] = r[f"{x}.{fam}.R"] - r[f"{y}.{fam}.R"]
    return r


W_EVENTS = {(lbl, e): CELLS[(lbl, "neural", e)]["events"] for lbl in LABELS for e in ERAS}
W_EQUAL = {k: 1.0 for k in W_EVENTS}

t0 = time.time()
point = {k: cell_aucs(c) for k, c in CELLS.items()}
est = {"events": summarise(point, W_EVENTS), "equal": summarise(point, W_EQUAL)}

rng = np.random.default_rng(SEED)
boots = {"events": [], "equal": []}
for b in range(N_BOOT):
    counts = np.bincount(rng.integers(0, N_SUBJ, N_SUBJ), minlength=N_SUBJ)
    pc, ok = {}, True
    for k, c in CELLS.items():
        ix = np.repeat(np.arange(len(c["y"])), counts[c["sc"]])
        v = cell_aucs(c, ix)
        if v is None:
            ok = False
            break
        pc[k] = v
    if not ok:
        continue
    boots["events"].append(summarise(pc, W_EVENTS))
    boots["equal"].append(summarise(pc, W_EQUAL))
    if (b + 1) % 100 == 0:
        print(f"  bootstrap {b + 1}/{N_BOOT}  [{(time.time() - t0) / 60:.1f} min]", flush=True)

result = {"n_subjects": N_SUBJ, "n_boot_used": len(boots["events"]), "seed": SEED,
          "weights_events": {f"{SHORT[l]}.{e}": W_EVENTS[(l, e)] for l, e in W_EVENTS}}
for scheme in ("events", "equal"):
    res = {}
    for k, v in est[scheme].items():
        bs = np.array([d[k] for d in boots[scheme]])
        bs = bs[np.isfinite(bs)]
        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        res[k] = dict(est=v, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                      p=float(min(1.0, p)))
    result[scheme] = res


def line(scheme, k, fmt="+.4f"):
    r = result[scheme][k]
    return f"{k:28s} {r['est']:{fmt}}  95% [{r['lo']:{fmt}}, {r['hi']:{fmt}}]  p={r['p']:.3f}"


print(f"\nsubjects {N_SUBJ}, bootstrap resamples used {result['n_boot_used']}, "
      f"{(time.time() - t0) / 60:.1f} min")
for scheme in ("events", "equal"):
    print(f"\n=== weights: {scheme} ===")
    for lbl in ("vaso", "intub", "mort"):
        for fam in ("neural", "trees", "both"):
            print("  " + line(scheme, f"{lbl}.{fam}.D") + "   (B - C)")
        for fam in ("neural", "trees", "both"):
            print("  " + line(scheme, f"{lbl}.{fam}.GC") + "   (C - A)")
        for fam in ("neural", "trees", "both"):
            print("  " + line(scheme, f"{lbl}.{fam}.R", "+.2f") + "   (retained)")
        print()
    for fam in ("neural", "trees", "both"):
        for x, y in (("intub", "vaso"), ("intub", "mort"), ("vaso", "mort")):
            print("  " + line(scheme, f"R({x})-R({y}).{fam}", "+.2f"))

Path(f"pooled{TAG}.json").write_text(json.dumps(result, indent=2))
print(f"\nSaved -> pooled{TAG}.json  (NN {NN_PRED}, trees {TREE_PRED})")
