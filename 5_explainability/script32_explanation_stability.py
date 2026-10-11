"""
script32_explanation_stability.py — are patients' prediction explanations more stable
under the bounded tree update (B_xgb) than under the full update (C_xgb)?

The arXiv paper's question, on the clean pipeline and the modular tree model. For each
patient, the explanation of the PREDICTION (logit) is TreeSHAP per feature:
    physiology channel c : wP * sum over hours of phi_P(c, h)     (51 values)
    treatment feature f  : wT * phi_T(f)                          (49 values)
computed for the source model A and for each updated model after step k, on the
incident evaluation cohort of era k (k = 3, 2020-22, is primary; it was the arXiv
paper's post-drift era).

Measures per patient (as the arXiv paper): signed Spearman between source and updated
attributions, and top-5 overlap (Jaccard of the top-5 features by |attribution|), for
the physiology stream, the treatment stream and all 100 features.
B - C: per-seed means; mean over seeds; patient bootstrap (1,000) of the seed-averaged
difference. Noise floor: same arm, different seeds (final era).

Check: under B_xgb the physiology attributions are the source's times wP1/wP0 > 0,
so physiology Spearman must be exactly 1 for every patient.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

from script30_modular_xgboost_clean import Frame, inc_mask, unpack_model
from script31_update_explanations import contribs, channel_sum
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS
from utils.data_utils import load_enriched_split

ARMS = ["B_xgb", "C_xgb"]
SUFFIX = os.environ.get("S31_SUFFIX", "")   # which script30 run to read
NP = len(SEQ_FEATURES)
N_BOOT, TOPK = 1000, 5


def explain(model, F, m):
    """(n, 51 + 49) per-patient attribution of the logit."""
    _, wP, wT = model["head"]
    phP, _ = contribs(model["P"], F.P[m])
    phT, _ = contribs(model["T"], F.T[m])
    return np.hstack([wP * channel_sum(phP), wT * phT])


def rowwise_spearman(a, b):
    """Spearman per row; NaN where either row is constant."""
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ra * rb).sum(1) / den, np.nan)


def topk_jaccard(a, b, k=TOPK):
    ta = np.argsort(-np.abs(a), 1)[:, :k]
    tb = np.argsort(-np.abs(b), 1)[:, :k]
    inter = np.array([len(set(x) & set(y)) for x, y in zip(ta, tb)])
    return inter / (2 * k - inter)


STREAMS = {"physio": slice(0, NP), "treat": slice(NP, None), "all": slice(None)}


def measures(src, upd):
    out = {}
    for s, sl in STREAMS.items():
        out[f"rho_{s}"] = rowwise_spearman(src[:, sl], upd[:, sl])
        out[f"top5_{s}"] = topk_jaccard(src[:, sl], upd[:, sl])
    return out


def main():
    arc = pickle.loads(Path(f"modxgb_models{SUFFIX}.pkl").read_bytes())
    SEQ = json.loads(Path("eval_split.json").read_text())["sequential"]
    steps = SEQ["adapt_steps"]
    te = load_enriched_split(Path("./data"), "test", SEQ_FEATURES, TREATMENT_FEATURES)
    F = {e: Frame(te.filter(pl.col("stay_id").is_in([int(x) for x in SEQ["era_eval_stays"][e]])))
         for e in steps}
    seeds = arc["seeds"]
    res, summary = {}, []
    for i, lbl in enumerate(LABEL_COLS):
        src_model = unpack_model(arc["source"][lbl])
        for k, era in enumerate(steps, start=1):
            m = inc_mask(F[era], i)
            src = explain(src_model, F[era], m)
            per = {}                                   # arm -> seed -> measures dict (per patient)
            expl = {}
            for arm in ARMS:
                for s in seeds:
                    rec = arc["models"][(lbl, arm, s)]
                    mod = unpack_model(rec, k)
                    e_ = explain(mod, F[era], m)
                    expl[(arm, s)] = e_
                    per.setdefault(arm, {})[s] = measures(src, e_)
                    if arm == "B_xgb":                 # the guarantee, checked
                        wP0, wP1 = src_model["head"][1], mod["head"][1]
                        assert wP0 > 0 and wP1 > 0, f"physiology head weight changed sign ({wP0}, {wP1})"
                        r = per[arm][s]["rho_physio"]
                        ok = r[~np.isnan(r)]
                        if not np.allclose(ok, 1.0, atol=1e-9):
                            raise SystemExit(f"STOP: B_xgb physiology Spearman != 1 (min {ok.min():.6f})")
            # B - C per measure: seed means, and a patient bootstrap of the seed-averaged difference
            rng = np.random.default_rng(11)
            n = int(m.sum())
            boots = [rng.integers(0, n, n) for _ in range(N_BOOT)]
            row = dict(label=lbl, era=era, step=k, n=n)
            for meas in per["B_xgb"][seeds[0]]:
                B = np.vstack([per["B_xgb"][s][meas] for s in seeds])      # (seeds, n)
                C = np.vstack([per["C_xgb"][s][meas] for s in seeds])
                d_seed = np.nanmean(B, 1) - np.nanmean(C, 1)
                bmean = np.nanmean(B, 0) - np.nanmean(C, 0)                # per patient, seed-averaged
                bs = np.array([np.nanmean(bmean[ix]) for ix in boots])
                row[meas] = dict(B=float(np.nanmean(B)), C=float(np.nanmean(C)),
                                 diff=float(np.nanmean(bmean)), ci95=[float(np.percentile(bs, 2.5)),
                                                                      float(np.percentile(bs, 97.5))],
                                 seeds_B_higher=int((d_seed > 0).sum()),
                                 nan_frac=float(np.isnan(B).mean() + np.isnan(C).mean()) / 2)
            if k == len(steps):                        # noise floor: same arm, different seeds
                for arm in ARMS:
                    pairs = [(a, b) for ai, a in enumerate(seeds) for b in seeds[ai + 1:]]
                    row[f"floor_{arm}"] = {s_: float(np.mean([np.nanmean(measures(expl[(arm, a)], expl[(arm, b)])[s_])
                                                              for a, b in pairs]))
                                           for s_ in ("rho_physio", "rho_treat", "rho_all")}
            summary.append(row)

    print("[check] B_xgb physiology Spearman = 1 for every patient, model and era (linear-head guarantee)\n")
    print("Source-vs-updated explanation agreement, B (bounded) vs C (full); mean over 5 seeds and patients")
    print(f"{'label':6s} {'era':12s} {'measure':12s} {'B':>7s} {'C':>7s} {'B-C':>8s}  {'95% CI':>18s}  B>C seeds")
    for r in summary:
        for meas in ("rho_all", "rho_treat", "top5_all", "top5_treat", "rho_physio", "top5_physio"):
            v = r[meas]
            print(f"{r['label'][6:12]:6s} {r['era']:12s} {meas:12s} {v['B']:7.3f} {v['C']:7.3f} {v['diff']:+8.3f}  "
                  f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  {v['seeds_B_higher']}/5")
        if "floor_B_xgb" in r:
            print(f"{'':6s} {'':12s} seed-to-seed floor (rho phys/treat/all): "
                  f"B {r['floor_B_xgb']['rho_physio']:.3f}/{r['floor_B_xgb']['rho_treat']:.3f}/{r['floor_B_xgb']['rho_all']:.3f}  "
                  f"C {r['floor_C_xgb']['rho_physio']:.3f}/{r['floor_C_xgb']['rho_treat']:.3f}/{r['floor_C_xgb']['rho_all']:.3f}")
        print()
    nanmax = max(r[m_]["nan_frac"] for r in summary for m_ in r if isinstance(r[m_], dict) and "nan_frac" in r[m_])
    print(f"largest share of patients with an undefined Spearman (constant attribution row): {nanmax:.1%}")
    Path(f"explanation_stability{SUFFIX}.json").write_text(json.dumps(summary, indent=1))
    print("Saved → explanation_stability.json")


if __name__ == "__main__":
    main()
