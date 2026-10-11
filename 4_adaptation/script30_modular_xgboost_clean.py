"""
script30_modular_xgboost_clean.py — the bounded-update question in a modular
gradient-boosted model, on the clean-v2 cohorts.

Port of rebuttal/script25b (updates THROUGH the frozen fusion head), changed only
where the clean pipeline requires it. Design fixed before running; reported as is.

MODEL, per label:  z(x) = b + wP*sP(xP) + wT*sT(xT),  p = sigmoid(z)
  sP  booster on the 51 physiology channels x 6 hours (flattened)
  sT  booster on the 49 treatment-context features
  (b, wP, wT) fusion head: unweighted logistic regression on (sP, sT)

UPDATING branch K through the current head: a new booster on branch-K features
only, base_margin = current fused logit z, trained on adapt-train + replay and
early-stopped on adapt-ES. Its output D is added as sK <- sK + D / wK, so
z <- z + D exactly. Head refit (where the arm adapts it): logistic regression on
adapt-train + replay with CROSS-FITTED scores for every branch updated this step.

ARMS (tree analogues of the neural arms)
  A_xgb         source, never updated                       (neural A)
  B_xgb         treatment branch + head refit; physio frozen (neural B)
  B_mirror_xgb  physiology branch, head frozen; treat frozen (neural B_mirror)
  C_xgb         physiology, then treatment, + head refit     (neural C, full update)

CHANGES vs script25b
  * cohorts, replay and schedule from eval_split.json of the clean-v2 run (3 forced
    updates); eval stays asserted equal to the neural evaluation cohorts
  * every selection on INCIDENT AUROC (source config + seed on val-ES, update eta on
    adapt-SEL at the first step, held thereafter) — as the neural pipeline
  * eta grid widened to 0.3 (script25b's arms sat at its 0.1 edge)
  * C_xgb added; arms not needed for the short paper dropped
  * inference = script17's patient-level paired bootstrap over the seed-averaged
    AUROC (imported, not copied) and script17's non-inferiority margin

Outputs  modular_xgboost_clean.json, per_era_predictions_modxgb.npz
"""
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from utils.constants import (SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN,
                             SOURCE_SEEDS, INCIDENT_GATE, SEED)
from utils.data_utils import load_enriched_split
from utils.eval_utils import to_serialisable
from script17_seed_floor import paired_patient_boot, NI_MARGIN

warnings.filterwarnings("ignore")
SEEDS = list(SOURCE_SEEDS)
N_SEARCH = 12
ETA_GRID = [0.003, 0.01, 0.03, 0.1, 0.3]
MAX_ROUNDS, ES_ROUNDS, POS_CAP, N_FOLDS = 1000, 50, 20.0, 5
NTHREAD = int(os.environ.get("XGB_NTHREAD", 10))
ARMS = ["B_xgb", "B_mirror_xgb", "C_xgb"]
UPDATES = {                        # (branches updated, in order;  head refit)
    "B_xgb": (("T",), True),
    "B_mirror_xgb": (("P",), False),
    "C_xgb": (("P", "T"), True),
}
HEAD_IDX = {"P": 1, "T": 2}
CONTRASTS = [("B_xgb", "C_xgb"), ("B_xgb", "B_mirror_xgb")]
# Shared-LR rerun (fairness): S30_ETA=0.03 S30_ARMS=B_xgb,C_xgb S30_SUFFIX=_eta0.03
SUFFIX = os.environ.get("S30_SUFFIX", "")
if os.environ.get("S30_ETA") or os.environ.get("S30_ARMS"):
    if not SUFFIX:
        raise SystemExit("S30_ETA / S30_ARMS need S30_SUFFIX so main outputs are not overwritten")
    if os.environ.get("S30_ETA"):
        ETA_GRID = [float(os.environ["S30_ETA"])]
    if os.environ.get("S30_ARMS"):
        ARMS = os.environ["S30_ARMS"].split(",")
        CONTRASTS = [(x, z) for x, z in CONTRASTS if x in ARMS and z in ARMS]
TREES_ADDED, HEADS, _CHECKED = {}, {}, []


# ── data ─────────────────────────────────────────────────────────────────────
class Frame:
    def __init__(self, df):
        d = df.sort(["stay_id", "hrs_from_admit"])
        n = d["stay_id"].n_unique()
        assert d.height == n * SEQ_LEN, "expected a dense 6-hour grid"
        self.P = d.select(SEQ_FEATURES).to_numpy().astype(np.float32).reshape(n, -1)
        first = d.gather_every(SEQ_LEN)
        assert first["stay_id"].n_unique() == n
        self.ids = first["stay_id"].to_numpy()
        self.T = first.select(TREATMENT_FEATURES).to_numpy().astype(np.float32)
        self.y = first.select(LABEL_COLS).to_numpy().astype(np.float32)
        self.meta = {c: first[c].to_numpy() for c in ("already_vaso", "already_vent")}

    def X(self, branch):
        return self.P if branch == "P" else self.T


def inc_mask(F, i):
    gate = INCIDENT_GATE[LABEL_COLS[i]]
    if gate is None:             # an outcome, not a treatment state: full cohort
        return np.ones(len(F.y), bool)
    return np.asarray(F.meta[gate]) <= 0


def inc_auc(F, i, score):
    m = inc_mask(F, i)
    return roc_auc_score(F.y[m, i], score[m])


# ── source boosters (script25 definitions) ───────────────────────────────────
def configs(n, seed=SEED):
    r = np.random.default_rng(seed)
    return [{"max_depth": int(r.choice([3, 4, 5, 6])),
             "learning_rate": float(r.choice([0.03, 0.05, 0.1])),
             "subsample": float(r.choice([0.6, 0.8])),
             "colsample_bytree": float(r.choice([0.3, 0.5, 0.8])),
             "min_child_weight": float(r.choice([1, 5, 20])),
             "reg_lambda": float(r.choice([1.0, 5.0, 20.0]))} for _ in range(n)]


def fit_source(X, y, Xv, yv, cfg, seed):
    # AUROC early stopping (script4/25): log-loss stops after one tree under scale_pos_weight
    p = dict(cfg, objective="binary:logistic", eval_metric="auc", tree_method="hist",
             seed=seed, nthread=NTHREAD,
             scale_pos_weight=float(np.clip((len(y) - y.sum()) / (y.sum() + 1e-6), 1, POS_CAP)))
    b = xgb.train(p, xgb.DMatrix(X, label=y), num_boost_round=MAX_ROUNDS,
                  evals=[(xgb.DMatrix(Xv, label=yv), "v")], early_stopping_rounds=ES_ROUNDS,
                  verbose_eval=False)
    return b[: b.best_iteration + 1]


def margin(b, X):
    return b.predict(xgb.DMatrix(X), output_margin=True)


# ── modular model: branch = list of (booster, scale, is_source) ───────────────
def delta(b, X):
    """What a delta booster adds to its base margin (base_margin=0 removes base_score)."""
    return b.predict(xgb.DMatrix(X, base_margin=np.zeros(len(X))), output_margin=True)


def branch_score(terms, X):
    s = np.zeros(len(X))
    for b, scale, is_source in terms:
        s += scale * (margin(b, X) if is_source else delta(b, X))
    return s


def logit(model, F):
    b, wP, wT = model["head"]
    return b + wP * branch_score(model["P"], F.P) + wT * branch_score(model["T"], F.T)


def prob(model, F):
    return 1.0 / (1.0 + np.exp(-logit(model, F)))


def train_delta(X, y, z, Xv, yv, zv, cfg, seed, eta):
    p = dict(cfg, learning_rate=eta, objective="binary:logistic", eval_metric="auc",
             tree_method="hist", seed=seed, nthread=NTHREAD)
    b = xgb.train(p, xgb.DMatrix(X, label=y, base_margin=z), num_boost_round=MAX_ROUNDS,
                  evals=[(xgb.DMatrix(Xv, label=yv, base_margin=zv), "v")],
                  early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
    b = b[: b.best_iteration + 1]
    if not _CHECKED:             # once: the delta is exactly what the booster adds to z
        full = b.predict(xgb.DMatrix(X, base_margin=z), output_margin=True)
        assert np.allclose(full - z, delta(b, X), atol=1e-4), "base_margin handling"
        _CHECKED.append(True)
    return b


def fit_deltas(model, branches, a_tr, a_es, i, cfg, seed, eta, idx=None):
    """One delta booster per branch, in order, each trained through the CURRENT head
    (so a later branch sees the earlier branch's update). idx: training rows."""
    rows = np.arange(len(a_tr.y)) if idx is None else idx
    y, ye = a_tr.y[rows, i], a_es.y[:, i]
    m, added = model, []
    for K in branches:
        w = m["head"][HEAD_IDX[K]]
        assert abs(w) > 1e-3, f"head weight on {K} is ~0; an update cannot be routed"
        b = train_delta(a_tr.X(K)[rows], y, logit(m, a_tr)[rows], a_es.X(K), ye,
                        logit(m, a_es), cfg[K], seed, eta)
        m = dict(m, **{K: m[K] + [(b, 1.0 / w, False)]})
        added.append(b.num_boosted_rounds())
    return m, added


def update(model, arm, frames, i, cfg, seed, eta, record=True):
    a_tr, a_es, _ = frames
    branches, refit = UPDATES[arm]
    m, added = fit_deltas(model, branches, a_tr, a_es, i, cfg, seed, eta)
    if record:
        TREES_ADDED.setdefault(arm, []).extend(added)
    if refit:
        y = a_tr.y[:, i]
        # branches not updated this step keep their current scores; updated ones get
        # out-of-fold scores, so the head never sees in-sample scores of a new delta
        s = {K: branch_score(model[K], a_tr.X(K)) for K in ("P", "T")}
        filled = {K: np.zeros(len(y), bool) for K in branches}
        for trn, hold in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(
                np.zeros(len(y)), y):
            mf, _ = fit_deltas(model, branches, a_tr, a_es, i, cfg, seed, eta, idx=trn)
            for K in branches:
                s[K][hold] = branch_score(mf[K], a_tr.X(K)[hold])
                filled[K][hold] = True
        assert all(f.all() for f in filled.values()), "cross-fitting left rows unscored"
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.column_stack([s["P"], s["T"]]), y)
        m["head"] = (float(lr.intercept_[0]), float(lr.coef_[0][0]), float(lr.coef_[0][1]))
        if record:
            HEADS.setdefault(arm, []).append(m["head"])
    return m


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    split = json.loads(Path("eval_split.json").read_text())
    SEQ = split["sequential"]
    EVAL_ERAS, SRC_KEY = SEQ["eval_eras"], SEQ["source_era_key"]
    STEPS, COH = SEQ["adapt_steps"], SEQ["cohorts"]
    if len(STEPS) != 3 or not all(COH[t].get("adapt_train_stays") for t in STEPS):
        raise SystemExit("expected three adaptation steps with cohorts (clean-v2 forced schedule)")

    tr = load_enriched_split(Path("./data"), "train", SEQ_FEATURES, TREATMENT_FEATURES)
    va = load_enriched_split(Path("./data"), "val", SEQ_FEATURES, TREATMENT_FEATURES)
    te = load_enriched_split(Path("./data"), "test", SEQ_FEATURES, TREATMENT_FEATURES)
    pick = lambda df, ids: df.filter(pl.col("stay_id").is_in([int(x) for x in ids]))
    calib = [int(x) for x in SEQ["era_eval_stays"][SRC_KEY]]
    F_tr = Frame(tr)
    F_ves = Frame(va.filter(~pl.col("stay_id").is_in(calib)))
    F_eval = {e: Frame(pick(va if e == SRC_KEY else te, SEQ["era_eval_stays"][e])) for e in EVAL_ERAS}
    F_step = {t: (Frame(pick(te, COH[t]["adapt_train_stays"] + COH[t].get("replay_stays", []))),
                  Frame(pick(te, COH[t]["adapt_val_es_stays"])),
                  Frame(pick(te, COH[t]["adapt_val_sel_stays"]))) for t in STEPS}

    # ── alignment checks against the pinned split and the neural predictions ──
    ref = np.load("per_era_predictions.npz", allow_pickle=True)
    key = lambda e: e.replace(" ", "").replace("-", "_").replace("(", "").replace(")", "")
    for e in EVAL_ERAS:
        want = sorted(int(x) for x in SEQ["era_eval_stays"][e])
        if sorted(F_eval[e].ids.tolist()) != want:
            raise SystemExit(f"eval cohort {e} does not match eval_split.json")
        if sorted(ref[f"stayids__{key(e)}"].tolist()) != want:
            raise SystemExit(f"eval cohort {e} does not match the neural predictions")
    ev_all = set(int(x) for v in SEQ["era_eval_stays"].values() for x in v)
    for t in STEPS:
        for F in F_step[t]:
            if ev_all & set(F.ids.tolist()):
                raise SystemExit(f"adaptation stays of {t} overlap evaluation stays")
        n_expect = len(set(COH[t]["adapt_train_stays"]) | set(COH[t].get("replay_stays", [])))
        if len(F_step[t][0].ids) != n_expect:
            raise SystemExit(f"adapt-train + replay size mismatch at {t}")

    print("=" * 78)
    print("MODULAR XGBOOST (clean-v2) — bounded vs full update in trees   <<< SHARE THIS BLOCK")
    print("=" * 78)
    for t in STEPS:
        a, es_, se = F_step[t]
        print(f"  cohort {t}: train+replay={len(a.ids)} es={len(es_.ids)} sel={len(se.ids)}")

    # ── SOURCE: config and seed chosen on val-ES incident AUROC ──────────────
    source, best_cfg = {}, {}
    for i, lbl in enumerate(LABEL_COLS):
        ytr, yv = F_tr.y[:, i], F_ves.y[:, i]
        best_cfg[lbl], bo = {}, {}
        for br in ("P", "T"):
            sc = [(inc_auc(F_ves, i, margin(fit_source(F_tr.X(br), ytr, F_ves.X(br), yv, c, SEED),
                                            F_ves.X(br))), j, c) for j, c in enumerate(configs(N_SEARCH))]
            best_cfg[lbl][br] = max(sc, key=lambda z: (z[0], -z[1]))[2]
            fits = [fit_source(F_tr.X(br), ytr, F_ves.X(br), yv, best_cfg[lbl][br], s) for s in SEEDS]
            order = np.argsort([inc_auc(F_ves, i, margin(b, F_ves.X(br))) for b in fits], kind="stable")
            bo[br] = fits[int(order[len(SEEDS) // 2])]
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(
            np.column_stack([margin(bo["P"], F_ves.P), margin(bo["T"], F_ves.T)]), yv)
        source[lbl] = {"P": [(bo["P"], 1.0, True)], "T": [(bo["T"], 1.0, True)],
                       "head": (float(lr.intercept_[0]), float(lr.coef_[0][0]), float(lr.coef_[0][1]))}
        print(f"  source {lbl.replace('label_', ''):<12} val-ES incident AUROC "
              f"{inc_auc(F_ves, i, prob(source[lbl], F_ves)):.4f}   head (b, wP, wT) = "
              f"({source[lbl]['head'][0]:.3f}, {source[lbl]['head'][1]:.3f}, {source[lbl]['head'][2]:.3f})"
              f"   [{(time.time() - t0) / 60:.1f} min]", flush=True)

    # ── ETA: per arm, first step, adapt-SEL incident AUROC (mean over labels) ──
    eta, sweeps, t1 = {}, {}, STEPS[0]
    for arm in ARMS:
        sc = {}
        for e_ in ETA_GRID:
            sc[e_] = float(np.mean([
                inc_auc(F_step[t1][2], i, prob(update(source[lbl], arm, F_step[t1], i, best_cfg[lbl],
                                                      SEEDS[0], e_, record=False), F_step[t1][2]))
                for i, lbl in enumerate(LABEL_COLS)]))
        eta[arm], sweeps[arm] = max(sc, key=sc.get), sc
        edge = " (grid edge)" if eta[arm] in (min(ETA_GRID), max(ETA_GRID)) else ""
        print(f"  {arm:<13} eta sweep {({k: round(v, 4) for k, v in sc.items()})} → {eta[arm]}{edge}"
              f"   [{(time.time() - t0) / 60:.1f} min]", flush=True)

    # ── Walk forward: every arm, seed and label through the same three steps ──
    preds = {a: {s: {} for s in SEEDS} for a in ["A_xgb"] + ARMS}
    for i, lbl in enumerate(LABEL_COLS):
        for s in SEEDS:
            for e in EVAL_ERAS:
                preds["A_xgb"][s].setdefault(e, {})[lbl] = prob(source[lbl], F_eval[e])
        for arm in ARMS:
            for s in SEEDS:
                m = source[lbl]
                hist = [(len(m["P"]), len(m["T"]), m["head"])]      # model after step k = first terms + head k
                for t in STEPS:
                    m = update(m, arm, F_step[t], i, best_cfg[lbl], s, eta[arm])
                    hist.append((len(m["P"]), len(m["T"]), m["head"]))
                SAVED[(lbl, arm, s)] = (m, hist)
                for e in EVAL_ERAS:
                    preds[arm][s].setdefault(e, {})[lbl] = prob(m, F_eval[e])
            print(f"  {lbl.replace('label_', ''):<12} {arm} done [{(time.time() - t0) / 60:.1f} min]",
                  flush=True)

    # ── Report: incident AUROC per seed; contrasts as script17 ────────────────
    print("\n" + "=" * 78 + "\nINCIDENT AUROC, final models, mean ± SD over seeds\n" + "=" * 78)
    out = []
    for i, lbl in enumerate(LABEL_COLS):
        print(f"\n{lbl.replace('label_', '')}")
        for e in EVAL_ERAS:
            F = F_eval[e]
            m = inc_mask(F, i)
            y = F.y[m, i]
            if not (0 < y.sum() < len(y)):
                continue
            powered = "source" not in e.lower() and y.sum() >= 50
            per = {a: np.array([roc_auc_score(y, preds[a][s][e][lbl][m]) for s in SEEDS])
                   for a in ["A_xgb"] + ARMS}
            print(f"  {e:<20} events={int(y.sum()):>4}{'' if powered else '   (probe / not powered)'}")
            for a in ["A_xgb"] + ARMS:
                g = "" if a == "A_xgb" else f"   gain vs A {(per[a] - per['A_xgb']).mean():+.4f}"
                print(f"    {a:<13} {per[a].mean():.4f} ± {per[a].std(ddof=1):.4f}{g}")
            cell = dict(label=lbl, era=e, events=int(y.sum()), powered=bool(powered),
                        per_seed={a: per[a].tolist() for a in per}, contrasts={})
            for x, z_ in CONTRASTS:
                d = per[x] - per[z_]
                pb = paired_patient_boot(y, np.stack([preds[x][s][e][lbl][m] for s in SEEDS]),
                                         np.stack([preds[z_][s][e][lbl][m] for s in SEEDS]))
                sup, ni = pb["ci95"][0] > 0, pb["ci90"][0] > -NI_MARGIN
                tag = "superior" if sup else ("non-inferior" if ni else "inconclusive")
                cell["contrasts"][f"{x}_minus_{z_}"] = dict(
                    paired_mean=float(d.mean()), paired_sd=float(d.std(ddof=1)),
                    sign_wins=int((d > 0).sum()), patient=pb, verdict=tag)
                print(f"    {x + ' − ' + z_:<26} seed {d.mean():+.4f} ± {d.std(ddof=1):.4f} "
                      f"sign {int((d > 0).sum())}/{len(SEEDS)} | patient 95%[{pb['ci95'][0]:+.4f},"
                      f"{pb['ci95'][1]:+.4f}] 90%lo {pb['ci90'][0]:+.4f} → {tag}")
            out.append(cell)

    print("\n  trees added per delta booster (median [min, max]):")
    for a, v in TREES_ADDED.items():
        print(f"    {a:<13} {int(np.median(v))} [{min(v)}, {max(v)}] over {len(v)} boosters")
        if max(v) <= 1:
            print("    ⚠ every update added at most one tree — updates are no-ops")
    print(f"\n  eta per arm: {eta}   NI margin {NI_MARGIN}   wall time {(time.time() - t0) / 60:.1f} min")

    Path(f"modular_xgboost_clean{SUFFIX}.json").write_text(json.dumps(to_serialisable(
        {"arms": ["A_xgb"] + ARMS, "seeds": SEEDS, "ni_margin": NI_MARGIN, "best_cfg": best_cfg,
         "eta": eta, "eta_sweeps": sweeps, "cells": out, "trees_added": TREES_ADDED,
         "heads": {a: [list(h) for h in v] for a, v in HEADS.items()},
         "selection_metric": "incident AUROC"}), indent=2))
    dump = {f"stayids__{key(e)}": F_eval[e].ids for e in EVAL_ERAS}
    for a in preds:
        for s in SEEDS:
            for e in EVAL_ERAS:
                dump[f"probs__{key(e)}__{a}__seed{s}"] = np.column_stack(
                    [preds[a][s][e][l] for l in LABEL_COLS])
    np.savez(f"per_era_predictions_modxgb{SUFFIX}.npz", **dump)
    print(f"\nSaved → modular_xgboost_clean{SUFFIX}.json, per_era_predictions_modxgb{SUFFIX}.npz")
    save_models(source, preds, F_eval, EVAL_ERAS, STEPS)


# ── model archive (added for the update-explanation analysis; changes no result) ──
SAVED = {}


def _pack(terms):
    return [(bytes(b.save_raw(raw_format="ubj")), float(sc), bool(src)) for b, sc, src in terms]


def unpack_model(rec, step=None):
    """Rebuild a saved model, optionally as it was after update step `step` (0 = source)."""
    def boosters(terms):
        out = []
        for raw, sc, src in terms:
            b = xgb.Booster()
            b.load_model(bytearray(raw))
            out.append((b, sc, src))
        return out
    k = len(rec["hist"]) - 1 if step is None else step
    nP, nT, head = rec["hist"][k]
    return {"P": boosters(rec["P"][:nP]), "T": boosters(rec["T"][:nT]), "head": tuple(head)}


def save_models(source, preds, F_eval, eval_eras, steps):
    import pickle
    arc = {"steps": list(steps), "labels": LABEL_COLS, "seeds": SEEDS,
           "feature_names": {"P": [f"{c}@h{h}" for h in range(SEQ_LEN) for c in SEQ_FEATURES],
                             "T": list(TREATMENT_FEATURES)},
           "source": {l: {"P": _pack(m["P"]), "T": _pack(m["T"]), "hist": [(1, 1, m["head"])]}
                      for l, m in source.items()},
           "models": {k: {"P": _pack(m["P"]), "T": _pack(m["T"]), "hist": h}
                      for k, (m, h) in SAVED.items()}}
    Path(f"modxgb_models{SUFFIX}.pkl").write_bytes(pickle.dumps(arc))
    # verify: every reloaded final model reproduces the predictions just saved
    arc = pickle.loads(Path(f"modxgb_models{SUFFIX}.pkl").read_bytes())
    worst = 0.0
    for (lbl, arm, s), rec in arc["models"].items():
        m = unpack_model(rec)
        for e in eval_eras:
            worst = max(worst, float(np.abs(prob(m, F_eval[e]) - preds[arm][s][e][lbl]).max()))
    for lbl, rec in arc["source"].items():
        m = unpack_model(rec)
        for e in eval_eras:
            worst = max(worst, float(np.abs(prob(m, F_eval[e]) - preds["A_xgb"][SEEDS[0]][e][lbl]).max()))
    if worst > 1e-6:
        raise SystemExit(f"STOP: reloaded models differ from saved predictions by {worst:.2e}")
    print(f"Saved → modxgb_models{SUFFIX}.pkl  ({len(arc['models'])} updated models + source; "
          f"reload check max |Δp| = {worst:.1e})")


if __name__ == "__main__":
    main()
