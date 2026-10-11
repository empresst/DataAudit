"""
script35_drift_detector.py — label-free drift detection that DECIDES each update.

Standard method (Rabanser, Guennemann & Lipton, "Failing Loudly", NeurIPS 2019; Lopez-Paz
& Oquab, classifier two-sample tests, ICLR 2017):

  C2ST  classifier two-sample test. A gradient-boosted classifier is trained to tell the
        new era's data from the reference data; out-of-fold scores (5-fold) give an AUROC,
        and a one-sided Mann-Whitney test of new-vs-reference scores gives the p-value.
        Run on all features (decision), and on physiology and treatment separately
        (which stream drifted).
  BBSD  black-box shift detection: two-sample KS test on the source model's predicted
        probabilities, per label, Bonferroni across labels; for the tree and the
        neural source model.

At update step k the reference is what the deployed model was trained on so far (source
training data + adaptation cohorts of earlier steps); the new sample is step k's
adaptation cohort. No labels are used. Both samples are subsampled to N per side so
power does not depend on era size.

DECISION: update at step k if the lower 95% bound (subject-cluster bootstrap) of the
all-features C2ST AUROC, computed with subject-grouped folds, exceeds 0.60.
Calibration: no-shift tests on subject-disjoint halves of the source data; false-alarm rate
must be <= 5% (STOP otherwise). The AUROC between the two source-period year groups is
reported as a scale for ordinary calendar drift. Streams are scored separately (report only).
If any step does not fire, the forced-schedule results are not the drift-triggered
results: the chain stops with exit code 3 so the affected updates can be rerun.
Writes drift_detection_standard.json.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from script30_modular_xgboost_clean import Frame, unpack_model, prob
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize, ICUDataset
from models.architectures import TwoStreamModel

N, N_NULL, SEED = 2000, int(__import__("os").environ.get("S35_NULL", 40)), 2026
AUC_MIN, N_BOOT = 0.60, 300   # practical-significance bar (chosen post hoc, after inspecting null-data behaviour); subject-cluster bootstrap
OTHER = Path("../ml4h_clean_v2_mortality")


def c2st(Xr, Xn, gr, gn, seed):
    """Classifier two-sample test with SUBJECT-grouped folds (readmissions never straddle
    train/test). Returns out-of-fold AUROC and the lower 95% bound of a subject-cluster
    bootstrap of that AUROC (the stays of one patient are not independent)."""
    X = np.vstack([Xr, Xn])
    y = np.r_[np.zeros(len(Xr)), np.ones(len(Xn))]
    g = np.r_[gr, gn]
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    _, inv = np.unique(g, return_inverse=True)
    rng, bs = np.random.default_rng(seed), []
    for _ in range(N_BOOT):
        ix = np.repeat(np.arange(len(y)), np.bincount(rng.integers(0, inv.max() + 1, inv.max() + 1),
                                                      minlength=inv.max() + 1)[inv])
        if 0 < y[ix].sum() < len(ix):
            bs.append(roc_auc_score(y[ix], oof[ix]))
    return float(roc_auc_score(y, oof)), float(np.percentile(bs, 2.5))


def sample(idx, n, rng):
    return idx if len(idx) <= n else rng.choice(idx, n, replace=False)


def main():
    rng = np.random.default_rng(SEED)
    SEQ = json.loads(Path("eval_split.json").read_text())["sequential"]
    steps, COH = SEQ["adapt_steps"], SEQ["cohorts"]
    if OTHER.exists():   # one decision serves both runs only if their cohorts are identical
        S2 = json.loads((OTHER / "eval_split.json").read_text())["sequential"]
        for t in steps:
            for kk in ("adapt_train_stays", "adapt_val_es_stays", "adapt_val_sel_stays"):
                if sorted(COH[t][kk]) != sorted(S2["cohorts"][t][kk]):
                    raise SystemExit(f"STOP: cohorts differ between runs ({t}, {kk})")
        print("[check] adaptation cohorts identical in the vaso/intubation and mortality runs")

    tr = load_enriched_split(Path("./data"), "train", SEQ_FEATURES, TREATMENT_FEATURES)
    te = load_enriched_split(Path("./data"), "test", SEQ_FEATURES, TREATMENT_FEATURES)
    sm = {}
    for sp in ("train", "test"):
        d = pl.read_parquet(f"data/{sp}_final_enriched.parquet", columns=["stay_id", "subject_id", "anchor_year_group"]).unique("stay_id")
        sm.update(zip(d["stay_id"].to_list(), d["subject_id"].to_list()))
    _y = pl.read_parquet("data/train_final_enriched.parquet", columns=["stay_id", "anchor_year_group"]).unique("stay_id")
    yg = dict(zip(_y["stay_id"].to_list(), _y["anchor_year_group"].to_list()))
    subj = lambda ids: np.array([sm[int(i)] for i in ids])
    pick = lambda ids: te.filter(pl.col("stay_id").is_in([int(x) for x in ids]))
    F_src = Frame(tr)
    F_new = {t: Frame(pick(COH[t]["adapt_train_stays"])) for t in steps}
    feats = lambda F: {"all": np.hstack([F.P, F.T]), "physio": F.P, "treat": F.T}

    # source models for BBSD
    arc = pickle.loads(Path("modxgb_models.pkl").read_bytes())
    trees = {l: unpack_model(arc["source"][l]) for l in LABEL_COLS}
    ck = torch.load("two_stream_models.pt", map_location="cpu", weights_only=False)
    st = calculate_train_stats(tr, sorted(set(SEQ_FEATURES + TREATMENT_FEATURES)))
    nn_model = TwoStreamModel(len(SEQ_FEATURES), len(TREATMENT_FEATURES), len(LABEL_COLS))
    nn_model.load_state_dict(ck["source"])
    nn_model.eval()

    def nn_prob(df):
        ds = ICUDataset(normalize(df, st), SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
        xs = torch.from_numpy(np.asarray(ds.seq_data)); xt = torch.from_numpy(np.asarray(ds.treat_data))
        with torch.no_grad():
            p = torch.sigmoid(nn_model(xs, xt)).numpy()
        return dict(zip(ds.stay_ids, p))

    nn_src = nn_prob(tr)
    nn_new = {t: nn_prob(pick(COH[t]["adapt_train_stays"])) for t in steps}

    # ── calibration: no-shift tests (subject-disjoint halves of the source data) and a
    #    SCALE reference: the real shift between the two source-period year groups ────────
    allX, sub_src = feats(F_src)["all"], subj(F_src.ids)
    us, fa, null_auc = np.unique(sub_src), [], []
    for r in range(N_NULL):
        rr = np.random.default_rng(SEED + r)
        half = set(rr.permutation(us)[: len(us) // 2].tolist())
        inA = np.array([x in half for x in sub_src])
        a = rr.choice(np.where(inA)[0], N, replace=False); b = rr.choice(np.where(~inA)[0], N, replace=False)
        auc, lo = c2st(allX[a], allX[b], sub_src[a], sub_src[b], SEED + r)
        null_auc.append(auc); fa.append(lo > AUC_MIN)
    far = float(np.mean(fa))
    print(f"[calibration] {N_NULL} no-shift tests: AUROC mean {np.mean(null_auc):.3f} (max {np.max(null_auc):.3f}); "
          f"false-alarm rate {far:.1%} at rule 'lower 95% bound > {AUC_MIN}'")
    if far > 0.05:
        raise SystemExit("STOP: detector false-alarm rate above 5% — not trustworthy")
    yrs = np.array([yg.get(int(i), "") for i in F_src.ids])
    y1, y2 = [np.where(yrs == k)[0] for k in sorted(set(yrs) - {""})[:2]]
    rr = np.random.default_rng(SEED)
    i1, i2 = rr.choice(y1, min(N, len(y1)), replace=False), rr.choice(y2, min(N, len(y2)), replace=False)
    ref_auc, ref_lo = c2st(allX[i1], allX[i2], sub_src[i1], sub_src[i2], SEED)
    print(f"[scale] adjacent source-period year groups ({sorted(set(yrs) - {''})[:2]}): C2ST AUROC {ref_auc:.3f} "
          f"(lower bound {ref_lo:.3f})  — what ordinary calendar drift looks like")

    # ── decisions ─────────────────────────────────────────────────────────────
    out = {"rule": f"lower 95% subject-bootstrap bound of grouped-CV C2ST AUROC > {AUC_MIN}", "n_per_side": N,
           "false_alarm_rate": far, "null_auroc_mean": float(np.mean(null_auc)),
           "adjacent_source_years_auroc": ref_auc, "steps": []}
    ref_parts = [("src", np.arange(len(F_src.ids)))]
    all_fired = True
    for k, t in enumerate(steps, start=1):
        R = {s: np.vstack([feats(F_src if nm == "src" else F_new[nm])[s][ix] for nm, ix in ref_parts])
             for s in ("all", "physio", "treat")}
        ref_ids = np.concatenate([(F_src if nm == "src" else F_new[nm]).ids[ix] for nm, ix in ref_parts])
        ri = sample(np.arange(len(ref_ids)), N, rng)
        ni = sample(np.arange(len(F_new[t].ids)), N, rng)
        row = {"step": k, "era": t, "n_ref": int(len(ri)), "n_new": int(len(ni))}
        for s in ("all", "physio", "treat"):
            auc, lo = c2st(R[s][ri], feats(F_new[t])[s][ni], subj(ref_ids[ri]), subj(F_new[t].ids[ni]), SEED + k)
            row[f"c2st_{s}"] = {"auroc": auc, "lo95": lo, "fire": bool(lo > AUC_MIN)}
        # BBSD (tree + neural source model), KS per label, Bonferroni
        bb = {}
        look = dict(nn_src)                       # neural source-model probabilities by stay id
        for nm, _ in ref_parts:
            if nm != "src":
                look.update(nn_new[nm])
        for fam in ("tree", "nn"):
            ps = []
            for i, l in enumerate(LABEL_COLS):
                if fam == "tree":
                    pr = np.concatenate([prob(trees[l], F_src if nm == "src" else F_new[nm])[ix]
                                         for nm, ix in ref_parts])[ri]
                    pn = prob(trees[l], F_new[t])[ni]
                else:
                    pr = np.array([look[int(sid)][i] for sid in ref_ids[ri]])
                    pn = np.array([nn_new[t][int(sid)][i] for sid in F_new[t].ids[ni]])
                ps.append(ks_2samp(pr, pn).pvalue)
            bb[fam] = {"p_min_bonf": float(min(1.0, min(ps) * len(ps))), "p_per_label": [float(x) for x in ps]}
        row["bbsd"] = bb
        row["fire"] = row["c2st_all"]["fire"]
        # report-only: which stream moved (does NOT change the schedule)
        row["scope_suggested"] = ("none" if not row["fire"] else
                                  "treatment+fusion only" if row["c2st_treat"]["fire"] and not row["c2st_physio"]["fire"] else
                                  "physiology moved too: freeze premise violated" if row["c2st_physio"]["fire"] else "unclear")
        all_fired &= row["fire"]
        out["steps"].append(row)
        print(f"  step {k} {t}: C2ST AUROC all {row['c2st_all']['auroc']:.3f} (lo {row['c2st_all']['lo95']:.3f}) | "
              f"physio {row['c2st_physio']['auroc']:.3f} (lo {row['c2st_physio']['lo95']:.3f}) | "
              f"treat {row['c2st_treat']['auroc']:.3f} (lo {row['c2st_treat']['lo95']:.3f}) | "
              f"BBSD p tree {bb['tree']['p_min_bonf']:.1e} nn {bb['nn']['p_min_bonf']:.1e} → "
              f"{'UPDATE' if row['fire'] else 'no update'}; {row['scope_suggested']}")
        ref_parts.append((t, np.arange(len(F_new[t].ids))))   # after updating, the model has seen era k

    Path("drift_detection_standard.json").write_text(json.dumps(out, indent=1))
    print("Saved → drift_detection_standard.json")
    if not all_fired:
        print("STOP: at least one step did not fire; the forced-schedule results are not the "
              "drift-triggered results. Rerun the affected updates before using them.")
        sys.exit(3)


if __name__ == "__main__":
    main()
