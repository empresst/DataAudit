"""
script34_nn_update_explanations.py — what drove each neural update, per patient.

Neural counterpart of script31. Integrated Gradients is linear in the model, so with a
shared zero baseline the attribution of the update  f_after - f_before  is
    IG(after) - IG(before)                                   (per feature, per patient)
and   f_after(x) - f_before(x) = sum(IG(after) - IG(before)) + [f_after(0) - f_before(0)]
up to IG's integration error: the bracket is the same for every patient ("uniform"),
the sum is the patient-specific part. Physiology is summed over the six hours.
Same summaries as script31: physiology share of the update, alert changes at a matched
10% rate, seed agreement on which features drove the update, top drivers.
Reads nn_states{S34_SUFFIX}.pt. Trains nothing.
"""
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr

from models.architectures import TwoStreamModel
from script17_seed_floor import incident_mask, device, BASE_PATH
from script33_nn_explanation_stability import integrated_gradients
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize, ICUDataset

SUFFIX = os.environ.get("S34_SUFFIX", "_bc_states")
ARMS, ALERT, TOP = ["B", "C"], 0.10, 10


def attr_and_logit(model, xs, xt, i):
    sa, ta = integrated_gradients(model, xs, xt, i)
    with torch.no_grad():
        model.eval()
        f = model(xs.to(device), xt.to(device))[:, i].cpu().numpy()
        f0 = model(torch.zeros_like(xs[:1]).to(device), torch.zeros_like(xt[:1]).to(device))[0, i].item()
    return np.hstack([sa.sum(1), ta]), f, f0


def main():
    ck = torch.load(f"nn_states{SUFFIX}.pt", map_location="cpu")
    SEQ = json.loads(Path("eval_split.json").read_text())["sequential"]
    steps = SEQ["adapt_steps"]
    assert ck["steps"] == steps
    tr = load_enriched_split(BASE_PATH, "train", SEQ_FEATURES, TREATMENT_FEATURES)
    te = normalize(load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES),
                   calculate_train_stats(tr, sorted(set(SEQ_FEATURES + TREATMENT_FEATURES))))
    seeds = [42, 123, 7, 2024, 99]
    nP = len(SEQ_FEATURES)
    names = [f"P:{n}" for n in SEQ_FEATURES] + [f"T:{n}" for n in TREATMENT_FEATURES]

    def build(state):
        m = TwoStreamModel(nP, len(TREATMENT_FEATURES), len(LABEL_COLS)).to(device)
        m.load_state_dict(state)
        return m

    rows, worst = [], []
    for k, era in enumerate(steps, start=1):
        ds = ICUDataset(te.filter(pl.col("stay_id").is_in([int(x) for x in SEQ["era_eval_stays"][era]])),
                        SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
        xs_all, xt_all = torch.from_numpy(ds.seq_data), torch.from_numpy(ds.treat_data)
        for i, lbl in enumerate(LABEL_COLS):
            m = torch.from_numpy(incident_mask(ds, lbl))
            xs, xt = xs_all[m], xt_all[m]
            for arm in ARMS:
                vecs = []
                for s in seeds:
                    st = ck["states"][f"{arm}__seed{s}"]
                    before = build(ck["source"] if k == 1 else st[k - 2])
                    after = build(st[k - 1])
                    a0, f0, z0 = attr_and_logit(before, xs, xt, i)
                    a1, f1, z1 = attr_and_logit(after, xs, xt, i)
                    d = a1 - a0
                    dz = f1 - f0
                    err = np.abs(d.sum(1) + (z1 - z0) - dz)            # IG integration error
                    worst.append((float(np.median(err)), float(np.median(np.abs(dz - dz.mean())) + 1e-12)))
                    ab = np.abs(d)
                    share_P = (ab[:, :nP].sum(1) / (ab.sum(1) + 1e-12)).mean()
                    flip = ((f0 >= np.quantile(f0, 1 - ALERT)) != (f1 >= np.quantile(f1, 1 - ALERT))).mean()
                    vecs.append(ab.mean(0))
                    rows.append(dict(step=k, era=era, label=lbl, arm=arm, seed=s, n=int(m.sum()),
                                     uniform=float(z1 - z0), sd_dz=float(dz.std()),
                                     share_physio=float(share_P), alert_flip=float(flip),
                                     imp=ab.mean(0).tolist()))
                rho = [spearmanr(vecs[a], vecs[b])[0] for a in range(5) for b in range(a + 1, 5)]
                for r in rows[-5:]:
                    r["seed_rho_median"] = float(np.median(rho))

    w = np.array(worst)
    print(f"[IG] median integration error of the update {np.median(w[:, 0]):.4f} vs median patient-specific "
          f"|Δz| {np.median(w[:, 1]):.4f}\n")
    print(f"{'label':6s} {'arm':3s} {'era':12s} {'sd dz':>6s} {'unif':>7s} {'%phys':>6s} {'flip':>5s} {'seed-ρ':>6s}  (mean of 5 seeds)")
    summ = []
    for lbl in LABEL_COLS:
        for arm in ARMS:
            for k, era in enumerate(steps, start=1):
                R = [r for r in rows if r["label"] == lbl and r["arm"] == arm and r["step"] == k]
                g = lambda f: float(np.mean([r[f] for r in R]))
                imp = np.mean([r["imp"] for r in R], 0)
                row = dict(label=lbl, arm=arm, era=era, sd_dz=g("sd_dz"), uniform=g("uniform"),
                           share_physio=g("share_physio"), flip=g("alert_flip"),
                           seed_rho=R[0]["seed_rho_median"], top=[names[j] for j in np.argsort(-imp)[:TOP]])
                summ.append(row)
                print(f"{lbl[6:12]:6s} {arm:3s} {era:12s} {row['sd_dz']:6.3f} {row['uniform']:+7.3f} "
                      f"{row['share_physio']:6.1%} {row['flip']:5.1%} {row['seed_rho']:6.2f}")
    print("\nTop drivers of each update (final step):")
    for r in summ:
        if r["era"] == steps[-1]:
            print(f"  {r['label'][6:]:11s} {r['arm']} " + ", ".join(r["top"][:6]))
    Path(f"nn_update_explanations{SUFFIX}.json").write_text(json.dumps({"rows": rows, "summary": summ}, indent=1))
    print(f"Saved → nn_update_explanations{SUFFIX}.json")


if __name__ == "__main__":
    main()
