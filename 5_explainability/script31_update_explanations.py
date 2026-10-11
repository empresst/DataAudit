"""
script31_update_explanations.py — exact, per-patient explanation of each tree update.

Model (script30):  z = b + wP*sP + wT*sT,  sK = sum of scale * booster output over branch K.
An update step adds delta boosters and (B, C) refits the head. For every patient the
change in logit between the model before and after step k splits EXACTLY into

  uniform     (b1 - b0) + bias columns of the TreeSHAP decompositions   (same for everyone)
  reweight_K  (wK1 - wK0) * sK_old                                     (head refit; rescales
                                                                         the old branch)
  learn_K     wK1 * sum(scale * new delta booster)                     (what the update learned)

and each non-uniform term splits over features by TreeSHAP (pred_contribs), which is
exact for trees. Physiology attributions are summed over the six hours per channel.

Checks (the run stops if any fails):
  * the step-3 model rebuilt from modxgb_models.pkl reproduces the saved predictions
  * sum of all components == z_after - z_before for every patient  (atol 1e-4)
  * B_xgb: the physiology branch has no new terms and its score is bit-identical

Evaluated on the incident evaluation cohort of the era each step adapts to, i.e. the
data the update was made for. Reads saved models only; trains nothing.
Writes update_explanations.json and a compact log.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from scipy.stats import spearmanr

from script30_modular_xgboost_clean import Frame, inc_mask, unpack_model, prob, logit, branch_score
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS
from utils.data_utils import load_enriched_split

ARMS = ["B_xgb", "C_xgb"]
SUFFIX = os.environ.get("S31_SUFFIX", "")   # which script30 run to read
ALERT = 0.10
TOP = 10
TOL = 1e-4


def contribs(terms, X):
    """Scaled TreeSHAP over a list of (booster, scale, is_source): (n, n_feat) and bias (n,)."""
    phi, bias = np.zeros((len(X), X.shape[1])), np.zeros(len(X))
    for b, sc, src in terms:
        dm = xgb.DMatrix(X) if src else xgb.DMatrix(X, base_margin=np.zeros(len(X)))
        c = b.predict(dm, pred_contribs=True).astype(np.float64)
        phi += sc * c[:, :-1]
        bias += sc * c[:, -1]
    return phi, bias


def channel_sum(phiP):
    """(n, 6*51) hour-major -> (n, 51) summed over hours."""
    return phiP.reshape(len(phiP), -1, len(SEQ_FEATURES)).sum(1)


def decompose(before, after, F, i_mask):
    """Exact split of z_after - z_before over (component, feature) for masked patients."""
    (b0, wP0, wT0), (b1, wP1, wT1) = before["head"], after["head"]
    nP0, nT0 = len(before["P"]), len(before["T"])
    XP, XT = F.P[i_mask], F.T[i_mask]
    oldP, bP0 = contribs(after["P"][:nP0], XP)
    oldT, bT0 = contribs(after["T"][:nT0], XT)
    newP, bP1 = contribs(after["P"][nP0:], XP) if len(after["P"]) > nP0 else (np.zeros_like(oldP), 0.0)
    newT, bT1 = contribs(after["T"][nT0:], XT) if len(after["T"]) > nT0 else (np.zeros_like(oldT), 0.0)
    comp = {
        "reweight_P": (wP1 - wP0) * channel_sum(oldP),
        "reweight_T": (wT1 - wT0) * oldT,
        "learn_P": wP1 * channel_sum(newP),
        "learn_T": wT1 * newT,
    }
    uniform = (b1 - b0) + (wP1 - wP0) * bP0 + (wT1 - wT0) * bT0 + wP1 * bP1 + wT1 * bT1
    dz = logit(after, F)[i_mask] - logit(before, F)[i_mask]
    resid = dz - (uniform + sum(c.sum(1) for c in comp.values()))
    if np.abs(resid).max() > TOL:
        raise SystemExit(f"STOP: decomposition does not add up (max residual {np.abs(resid).max():.2e})")
    return comp, uniform, dz, float(np.abs(resid).max())


def main():
    arc = pickle.loads(Path(f"modxgb_models{SUFFIX}.pkl").read_bytes())
    split = json.loads(Path("eval_split.json").read_text())
    SEQ = split["sequential"]
    steps = SEQ["adapt_steps"]
    assert steps == arc["steps"], "model archive and eval_split.json disagree on the schedule"
    te = load_enriched_split(Path("./data"), "test", SEQ_FEATURES, TREATMENT_FEATURES)
    F = {e: Frame(te.filter(pl.col("stay_id").is_in([int(x) for x in SEQ["era_eval_stays"][e]])))
         for e in steps}
    saved = np.load(f"per_era_predictions_modxgb{SUFFIX}.npz", allow_pickle=True)
    key = lambda e: e.replace(" ", "").replace("-", "_")
    featP, featT = list(SEQ_FEATURES), list(TREATMENT_FEATURES)

    out, worst_resid = [], 0.0
    for i, lbl in enumerate(LABEL_COLS):
        for arm in ARMS:
            for s in arc["seeds"]:
                rec = arc["models"][(lbl, arm, s)]
                # check: rebuilt final model == saved predictions, on every era it was evaluated on
                fin = unpack_model(rec)
                for e in steps:
                    ref_ids = saved[f"stayids__{key(e)}"]
                    ref = saved[f"probs__{key(e)}__{arm}__seed{s}"][:, i]
                    mine = dict(zip(F[e].ids.tolist(), prob(fin, F[e])))
                    if max(abs(mine[int(x)] - p) for x, p in zip(ref_ids, ref)) > 1e-6:
                        raise SystemExit(f"STOP: rebuilt {arm} seed {s} differs from saved predictions at {e}")
                for k, era in enumerate(steps, start=1):
                    before, after = unpack_model(rec, k - 1), unpack_model(rec, k)
                    if arm == "B_xgb":         # the bound: physiology terms untouched, score identical
                        assert len(after["P"]) == len(before["P"]), "B_xgb added a physiology term"
                        assert np.array_equal(branch_score(after["P"], F[era].P),
                                              branch_score(before["P"], F[era].P)), "physiology score moved"
                    m = inc_mask(F[era], i)
                    comp, uni, dz, r = decompose(before, after, F[era], m)
                    worst_resid = max(worst_resid, r)
                    # alert changes at a matched 10% rate (computed on probabilities = monotone in z)
                    z0 = logit(before, F[era])[m]
                    z1 = z0 + dz
                    a0, a1 = z0 >= np.quantile(z0, 1 - ALERT), z1 >= np.quantile(z1, 1 - ALERT)
                    flip = a0 != a1
                    tot = {c: np.abs(v).sum(1) for c, v in comp.items()}
                    allabs = sum(tot.values()) + 1e-12
                    centred = dz - dz.mean()
                    imp_P = np.abs(comp["reweight_P"] + comp["learn_P"]).mean(0)
                    imp_T = np.abs(comp["reweight_T"] + comp["learn_T"]).mean(0)
                    out.append(dict(
                        label=lbl, arm=arm, seed=s, step=k, era=era, n=int(m.sum()),
                        mean_dz=float(dz.mean()), sd_dz=float(dz.std()),
                        uniform_mean=float(np.mean(uni)),
                        share={c: float((tot[c] / allabs).mean()) for c in tot},
                        share_physio=float(((tot["reweight_P"] + tot["learn_P"]) / allabs).mean()),
                        share_learned=float(((tot["learn_P"] + tot["learn_T"]) / allabs).mean()),
                        corr_patient_specific_learnT=float(np.corrcoef(centred, comp["learn_T"].sum(1))[0, 1])
                        if comp["learn_T"].any() else None,
                        alert_flip=float(flip.mean()),
                        flip_share_physio=float(((tot["reweight_P"] + tot["learn_P"])[flip] / allabs[flip]).mean())
                        if flip.any() else None,
                        imp_P=imp_P.tolist(), imp_T=imp_T.tolist()))

    # ── summaries ───────────────────────────────────────────────────────────────
    print(f"[checks] rebuilt models match saved predictions; decomposition exact "
          f"(max residual {worst_resid:.1e}); B_xgb physiology branch unchanged at every step\n")
    res = {"rows": out, "featP": featP, "featT": featT, "max_residual": worst_resid}
    print(f"{'label':6s} {'arm':6s} {'era':12s} {'sd dz':>6s} {'unif':>6s} {'%phys':>6s} {'%learn':>6s} "
          f"{'flip':>5s} {'%phys|flip':>10s}   (mean over 5 seeds)")
    summ = []
    for lbl in LABEL_COLS:
        for arm in ARMS:
            for k, era in enumerate(steps, start=1):
                R = [r for r in out if r["label"] == lbl and r["arm"] == arm and r["step"] == k]
                g = lambda f: float(np.mean([f(r) for r in R]))
                fp = [r["flip_share_physio"] for r in R if r["flip_share_physio"] is not None]
                # seed agreement on WHICH features drive the update (top features by mean |attr|)
                vec = [np.r_[r["imp_P"], r["imp_T"]] for r in R]
                rho = [spearmanr(vec[a], vec[b])[0] for a in range(5) for b in range(a + 1, 5)]
                imp = np.mean(vec, 0)
                names = [f"P:{n}" for n in featP] + [f"T:{n}" for n in featT]
                top = [names[j] for j in np.argsort(-imp)[:TOP]]
                row = dict(label=lbl, arm=arm, era=era, sd_dz=g(lambda r: r["sd_dz"]),
                           uniform=g(lambda r: r["uniform_mean"]), share_physio=g(lambda r: r["share_physio"]),
                           share_learned=g(lambda r: r["share_learned"]), flip=g(lambda r: r["alert_flip"]),
                           flip_share_physio=float(np.mean(fp)) if fp else None,
                           seed_rho_median=float(np.median(rho)), top=top)
                summ.append(row)
                print(f"{lbl[6:12]:6s} {arm[:5]:6s} {era:12s} {row['sd_dz']:6.3f} {row['uniform']:+6.3f} "
                      f"{row['share_physio']:6.1%} {row['share_learned']:6.1%} {row['flip']:5.1%} "
                      f"{(row['flip_share_physio'] or 0):10.1%}   seed-ρ {row['seed_rho_median']:.2f}")
    print("\nTop drivers of each update (final step, mean |attribution| over seeds):")
    for row in summ:
        if row["era"] == steps[-1]:
            print(f"  {row['label'][6:]:11s} {row['arm']:6s} " + ", ".join(row["top"][:6]))
    res["summary"] = summ
    Path(f"update_explanations{SUFFIX}.json").write_text(json.dumps(res, indent=1))
    print("\nSaved → update_explanations.json")


if __name__ == "__main__":
    main()
