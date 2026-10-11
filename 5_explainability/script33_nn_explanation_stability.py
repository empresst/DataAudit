"""
script33_nn_explanation_stability.py — the arXiv paper's explanation-stability question
for the two-stream neural model, on the clean pipeline; contrast to script32 (trees).

Integrated Gradients as in the arXiv paper (try3/script13Rag_BvsC.py:369): zero baseline
(= training mean after z-scoring), eval mode, attribution of the label's logit, physiology
summed over the six hours. 50 steps instead of 20; the completeness error is reported.
Explanations of the source model and of B / C after update step k, on the incident
evaluation cohort of era k; same measures as script32 (signed Spearman, top-5 Jaccard,
per stream), B - C over five seeds, and a same-arm seed-to-seed floor (final era).

Checks: B's LSTM weights equal the source's at every step; reloaded final models
reproduce the predictions saved by script17 (per_era_predictions_seeds.npz).
Reads nn_states_bc_states.pt (script17, S17_SAVE_STATES=1). Trains nothing.
"""
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch

from models.architectures import TwoStreamModel
from script17_seed_floor import predict, incident_mask, device, BASE_PATH
from script32_explanation_stability import measures, NP
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize, ICUDataset

IG_STEPS = 50
SUFFIX = os.environ.get("S33_SUFFIX", "_bc_states")    # which script17 run to explain
ARMS = ["B", "C"]


def integrated_gradients(model, xs, xt, target, steps=IG_STEPS, bs=1024):
    torch.backends.cudnn.enabled = False
    model.eval()
    out_s, out_t = [], []
    for a in range(0, len(xs), bs):
        s, t = xs[a:a + bs].to(device), xt[a:a + bs].to(device)
        gs, gt = torch.zeros_like(s), torch.zeros_like(t)
        with torch.enable_grad():
            for alpha in np.linspace(0, 1, steps):
                i_s = (alpha * s).requires_grad_(True)
                i_t = (alpha * t).requires_grad_(True)
                g1, g2 = torch.autograd.grad(model(i_s, i_t)[:, target].sum(), [i_s, i_t])
                gs += g1
                gt += g2
        out_s.append((s * gs / steps).cpu().numpy())
        out_t.append((t * gt / steps).cpu().numpy())
    return np.concatenate(out_s), np.concatenate(out_t)


def explain(model, xs, xt, target):
    sa, ta = integrated_gradients(model, xs, xt, target)
    attr = np.hstack([sa.sum(1), ta])                       # (n, 51 + 49)
    with torch.no_grad():
        model.eval()
        f = model(xs.to(device), xt.to(device))[:, target].cpu().numpy()
        f0 = model(torch.zeros_like(xs).to(device), torch.zeros_like(xt).to(device))[:, target].cpu().numpy()
    comp = np.abs(attr.sum(1) - (f - f0))
    return attr, float(np.median(comp)), float(np.median(np.abs(f - f0)))


def main():
    ck = torch.load(f"nn_states{SUFFIX}.pt", map_location="cpu")
    SEQ = json.loads(Path("eval_split.json").read_text())["sequential"]
    steps = SEQ["adapt_steps"]
    assert ck["steps"] == steps, "saved states and eval_split.json disagree on the schedule"
    tr = load_enriched_split(BASE_PATH, "train", SEQ_FEATURES, TREATMENT_FEATURES)
    te = load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES)
    st = calculate_train_stats(tr, sorted(set(SEQ_FEATURES + TREATMENT_FEATURES)))
    te = normalize(te, st)
    ds = {e: ICUDataset(te.filter(pl.col("stay_id").is_in([int(x) for x in SEQ["era_eval_stays"][e]])),
                        SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN) for e in steps}
    seq_dim, treat_dim = len(ds[steps[0]].seq_cols), len(ds[steps[0]].treat_cols)
    assert seq_dim == NP and seq_dim + treat_dim == 100, "feature layout differs from script32"
    seeds = sorted({int(k.split("seed")[1]) for k in ck["states"]}, key=[42, 123, 7, 2024, 99].index)

    def build(state):
        m = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
        m.load_state_dict(state)
        return m

    src = build(ck["source"])
    saved = np.load(f"per_era_predictions_seeds{SUFFIX}.npz", allow_pickle=True)
    key = lambda e: e.replace(" ", "").replace("-", "_")
    # checks: bound exact; final models reproduce script17's saved predictions
    for s in seeds:
        for k in range(len(steps)):
            sb = ck["states"][f"B__seed{s}"][k]
            for name, v in ck["source"].items():
                if name.startswith("physio") and not torch.equal(v, sb[name]):
                    raise SystemExit(f"STOP: B seed {s} step {k + 1}: LSTM parameter {name} moved")
        for arm in ARMS:
            m = build(ck["states"][f"{arm}__seed{s}"][-1])
            for e in steps:
                p = predict(m, ds[e])
                ref = saved[f"probs__{key(e)}__{arm}__seed{s}"]
                assert np.array_equal(np.array(ds[e].stay_ids), saved[f"stayids__{key(e)}"]), "stay order"
                if np.abs(p - ref).max() > 1e-5:
                    raise SystemExit(f"STOP: reloaded {arm} seed {s} differs from saved predictions at {e}")
    print("[checks] B's LSTM identical to the source at every step; reloaded models reproduce saved predictions")

    summary, comp_err = [], []
    for i, lbl in enumerate(LABEL_COLS):
        for k, era in enumerate(steps, start=1):
            d = ds[era]
            m = incident_mask(d, lbl)
            loader = torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
            xs, xt, _ = next(iter(loader))
            xs, xt = xs[torch.from_numpy(m)], xt[torch.from_numpy(m)]
            a_src, ce, scale = explain(src, xs, xt, i)
            comp_err.append((ce, scale))
            per, expl = {}, {}
            for arm in ARMS:
                for s in seeds:
                    a_up, ce, scale = explain(build(ck["states"][f"{arm}__seed{s}"][k - 1]), xs, xt, i)
                    comp_err.append((ce, scale))
                    expl[(arm, s)] = a_up
                    per.setdefault(arm, {})[s] = measures(a_src, a_up)
            row = dict(label=lbl, era=era, step=k, n=int(m.sum()))
            for meas in per["B"][seeds[0]]:
                B = np.vstack([per["B"][s][meas] for s in seeds])
                C = np.vstack([per["C"][s][meas] for s in seeds])
                d_seed = np.nanmean(B, 1) - np.nanmean(C, 1)
                row[meas] = dict(B=float(np.nanmean(B)), C=float(np.nanmean(C)),
                                 diff=float(d_seed.mean()), diff_sd=float(d_seed.std(ddof=1)),
                                 seeds_B_higher=int((d_seed > 0).sum()))
            if k == len(steps):
                pairs = [(a, b) for ai, a in enumerate(seeds) for b in seeds[ai + 1:]]
                for arm in ARMS:
                    row[f"floor_{arm}"] = {s_: float(np.mean([np.nanmean(measures(expl[(arm, a)], expl[(arm, b)])[s_])
                                                              for a, b in pairs]))
                                           for s_ in ("rho_physio", "rho_treat", "rho_all")}
            summary.append(row)

    ce = np.array(comp_err)
    print(f"[IG] completeness error: median {np.median(ce[:, 0]):.4f} vs median |f(x)-f(0)| "
          f"{np.median(ce[:, 1]):.4f}  (ratio {np.median(ce[:, 0] / ce[:, 1]):.1%})\n")
    print("NEURAL: source-vs-updated explanation agreement, B (bounded) vs C (full); mean over 5 seeds")
    print(f"{'label':6s} {'era':12s} {'measure':12s} {'B':>7s} {'C':>7s} {'B-C':>8s} {'seed sd':>8s}  B>C seeds")
    for r in summary:
        for meas in ("rho_all", "rho_physio", "rho_treat", "top5_all", "top5_physio", "top5_treat"):
            v = r[meas]
            print(f"{r['label'][6:12]:6s} {r['era']:12s} {meas:12s} {v['B']:7.3f} {v['C']:7.3f} {v['diff']:+8.3f} "
                  f"{v['diff_sd']:8.3f}  {v['seeds_B_higher']}/5")
        if "floor_B" in r:
            print(f"{'':19s} seed-to-seed floor (rho phys/treat/all): "
                  f"B {r['floor_B']['rho_physio']:.3f}/{r['floor_B']['rho_treat']:.3f}/{r['floor_B']['rho_all']:.3f}  "
                  f"C {r['floor_C']['rho_physio']:.3f}/{r['floor_C']['rho_treat']:.3f}/{r['floor_C']['rho_all']:.3f}")
        print()
    Path(f"nn_explanation_stability{SUFFIX}.json").write_text(json.dumps(summary, indent=1))
    print(f"Saved → nn_explanation_stability{SUFFIX}.json   (LR {ck['best_lr']})")


if __name__ == "__main__":
    main()
