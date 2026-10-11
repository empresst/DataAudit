"""
script17_seed_floor.py
══════════════════════
The noise floor for the one claim the study is keeping.

THE PROBLEM. script2 adapts across seeds, selects the median-validation-AUROC
model, and reports that single number. So B vs B_mirror is currently a
difference between two medians with no dispersion attached. The first question
any reviewer asks about a single-number architecture claim is "is +0.0385 larger
than seed-to-seed variation?", and right now the study cannot answer it.

Every other weakness in this pipeline has a control. This one does not, and it
sits under the headline result.

WHAT THIS DOES. Re-runs ONLY Run B and Run B_mirror, replaying script2's exact
schedule from eval_split.json, starting from script2's stored source state, and
keeps EVERY seed instead of collapsing to the median. It reports:

    per-seed AUROC              per arm, era and label — the raw material
    paired per-seed difference  B - B_mirror computed WITHIN each seed, which
                                removes the shared source model and the shared
                                cohort draw, leaving only what the freeze choice
                                did
    mean +/- sd of that paired difference, and a sign count

WHY PAIRED. Both arms start from the same source weights and see the same
adaptation cohorts, so the per-seed difference is a matched comparison. An
unpaired comparison of two 5-number samples throws away that structure and
inflates the apparent noise.

THE BAR. Report the effect as resolved only where |mean paired difference|
exceeds its own standard deviation. That is a deliberately modest bar — it is
not a significance test, and it is not presented as one. It is the dispersion
number that should have been in the paper from the start.

This does NOT re-select a model, re-tune a learning rate, or change any reported
result. It re-runs an existing comparison and reports what was discarded.

Inputs   eval_split.json, two_stream_models.pt
Outputs  seed_floor.json, per_era_predictions_seeds.npz, and a digest block

RUN WITH FAST_MODE=False. At 3 seeds a standard deviation is estimated from
3 points and the interval is not worth printing.
"""
import copy
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from utils.constants import (SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN,
                             BATCH_SIZE, ADAPT_EPOCHS, ADAPT_PATIENCE, LR_ADAPT_GRID,
                             SOURCE_SEEDS, INCIDENT_GATE, L2SP_LAMBDA)
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize, ICUDataset
from utils.train_utils import make_adapt_criterion
from utils.eval_utils import to_serialisable
from models.architectures import TwoStreamModel, count_trainable, l2sp_penalty
import os

BASE_PATH = Path("./data")
ART_DIRS = [Path("."), Path(".")]
SAVE_PATH = Path(".")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLEAN-V2: C and C_l2sp added — the claims under test are B vs B_mirror, B vs C
# (full fine-tune) and B vs C_l2sp (anchored full fine-tune).
ARMS = ["B", "B_mirror", "C", "C_l2sp"]
CONTRASTS = [("B", "B_mirror"), ("B", "C"), ("B", "C_l2sp")]
# CLEAN-V2 lambda sweep hooks (defaults reproduce the main run exactly):
#   S17_ARMS="B,C_l2sp"  S17_L2SP=0.1  S17_SUFFIX=_l2sp0.1
if os.environ.get("S17_ARMS"):
    ARMS = os.environ["S17_ARMS"].split(",")
    CONTRASTS = [(x, z) for x, z in CONTRASTS if x in ARMS and z in ARMS]
L2SP_LAMBDA = float(os.environ.get("S17_L2SP", L2SP_LAMBDA))
OUT_SUFFIX = os.environ.get("S17_SUFFIX", "")
# S17_SAVE_STATES=1 also writes nn_states{suffix}.pt (model after each step, per arm and seed).
# Use with a suffix so the main seed_floor.json / predictions are never overwritten.
SAVE_STATES = os.environ.get("S17_SAVE_STATES") == "1"
if SAVE_STATES and not OUT_SUFFIX:
    raise SystemExit("S17_SAVE_STATES=1 needs S17_SUFFIX so main outputs are not overwritten")
STATES = {}
# S17_LR=0.003 forces one adaptation LR for every arm (matched-LR sensitivity analysis).
if os.environ.get("S17_LR"):
    if not OUT_SUFFIX:
        raise SystemExit("S17_LR needs S17_SUFFIX so main outputs are not overwritten")
    LR_ADAPT_GRID = [float(os.environ["S17_LR"])]
# Non-inferiority margin in incident AUROC. FIX THIS BEFORE LOOKING AT RESULTS;
# changing it afterwards invalidates the non-inferiority statement.
NI_MARGIN = float(os.environ.get("NI_MARGIN", "0.015"))
N_BOOT_PATIENT = 1000


def find(*names):
    for d in ART_DIRS:
        for n in names:
            if (d / n).exists():
                return d / n
    return None


def need(*names):
    """
    find() that refuses to return None.

    Calling `find(...).read_text()` on a missing artifact raises
    AttributeError: 'NoneType' — after the script has already spent time on
    setup, and with a message that names neither the file nor the script that
    should have produced it. Fail here instead, saying exactly what is missing.
    """
    p = find(*names)
    if p is None:
        raise SystemExit(
            f"required artifact not found: {' / '.join(names)}\n"
            f"searched: {[str(d) for d in ART_DIRS]}\n"
            f"run the script that produces it (script2 for eval_split.json, "
            f"per_era_predictions.npz and two_stream_models.pt) before this one.")
    return p


def set_arm(model, arm):
    """B: physio frozen, treat+fusion trainable. B_mirror: the reverse.
    C / C_l2sp: everything trainable (C_l2sp adds the L2-SP anchor in fit)."""
    if arm == "B":
        model.unfreeze_adaptive()
    elif arm == "B_mirror":
        model.unfreeze_mirror()
    elif arm in ("C", "C_l2sp"):
        model.unfreeze_all()
    else:
        raise ValueError(arm)
    return model


def fit(model, tr_ds, es_ds, crit, lr, epochs, patience, l2sp_ref=None, l2sp_lambda=0.0):
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best, bstate, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xs, xt, y in DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True):
            opt.zero_grad()
            loss = crit(model(xs.to(device), xt.to(device)), y.to(device))
            if l2sp_ref is not None and l2sp_lambda > 0:    # as script2: anchor = source
                loss = loss + l2sp_lambda * l2sp_penalty(model, l2sp_ref, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        model.eval()
        tot = n = 0
        with torch.no_grad():
            for xs, xt, y in DataLoader(es_ds, batch_size=512):
                tot += crit(model(xs.to(device), xt.to(device)), y.to(device)).item() * len(y)
                n += len(y)
        vl = tot / max(n, 1)
        sch.step(vl)
        if vl < best:
            best, bstate, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if bstate:
        model.load_state_dict(bstate)
    return model


@torch.no_grad()
def predict(model, ds):
    model.eval()
    return np.vstack([torch.sigmoid(model(xs.to(device), xt.to(device))).cpu().numpy()
                      for xs, xt, _ in DataLoader(ds, batch_size=512, shuffle=False)])


def incident_mask(ds, lbl):
    g = ds.meta.get(INCIDENT_GATE[lbl])
    return (np.asarray(g) <= 0) if g is not None and np.size(g) else np.ones(len(ds.labels), bool)


def auroc_mean(p, ds):
    """Mean INCIDENT AUROC over labels — the reported endpoint (CLEAN-V2; the
    overall AUROC used before was dominated by stays already on treatment)."""
    y, v = ds.labels, []
    for i, lbl in enumerate(LABEL_COLS):
        m = incident_mask(ds, lbl)
        if 0 < y[m, i].sum() < m.sum():
            v.append(roc_auc_score(y[m, i], p[m, i]))
    return float(np.mean(v)) if v else float("nan")


def paired_patient_boot(y, pa, pb, n_boot=N_BOOT_PATIENT, seed=11):
    """Patient-level paired bootstrap of the seed-averaged AUROC difference.
    pa, pb: (n_seeds, n) probabilities. Each resample draws stays, computes
    AUROC per seed for both arms on the SAME draw, and averages over seeds.
    Returns point, 95% CI and 90% CI (the 90% lower bound is the one-sided
    alpha=0.05 non-inferiority test)."""
    rng = np.random.default_rng(seed)
    pt = float(np.mean([roc_auc_score(y, a) - roc_auc_score(y, b) for a, b in zip(pa, pb)]))
    bs, n = [], len(y)
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        yy = y[ix]
        if not (0 < yy.sum() < n):
            continue
        bs.append(np.mean([roc_auc_score(yy, a[ix]) - roc_auc_score(yy, b[ix])
                           for a, b in zip(pa, pb)]))
    bs = np.asarray(bs)
    return dict(point=pt, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                ci90=[float(np.percentile(bs, 5)), float(np.percentile(bs, 95))], n_boot=len(bs))


def main():
    split = json.loads(need("eval_splitR.json", "eval_split.json").read_text())
    ck = torch.load(need("two_stream_modelsR.pt", "two_stream_models.pt"), map_location="cpu")
    SEQ = split["sequential"]
    EVAL_ERAS, SRC_KEY = SEQ["eval_eras"], SEQ["source_era_key"]
    STEPS, COH = SEQ["adapt_steps"], SEQ["cohorts"]
    AL = split["adapt_loss"]
    SEEDS = list(SOURCE_SEEDS)

    if len(SEEDS) < 5:
        print("\n" + "!" * 76)
        print(f"  FAST_MODE is on — {len(SEEDS)} seeds. A standard deviation from"
              f" {len(SEEDS)} points")
        print("  is not a noise floor. Set FAST_MODE=False before quoting anything.")
        print("!" * 76 + "\n")

    train_df = load_enriched_split(BASE_PATH, "train", SEQ_FEATURES, TREATMENT_FEATURES)
    val_df = load_enriched_split(BASE_PATH, "val", SEQ_FEATURES, TREATMENT_FEATURES)
    test_df = load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES)
    st = calculate_train_stats(train_df, sorted(set(SEQ_FEATURES + TREATMENT_FEATURES)))
    train_df, val_df, test_df = (normalize(train_df, st), normalize(val_df, st),
                                 normalize(test_df, st))

    tr_ds = ICUDataset(train_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    seq_dim, treat_dim = len(tr_ds.seq_cols), len(tr_ds.treat_cols)

    def rows(ids, src=False):
        f = val_df if src else test_df
        return f.filter(pl.col("stay_id").is_in([int(x) for x in ids]))

    eval_ds = {e: ICUDataset(rows(SEQ["era_eval_stays"][e], src=(e == SRC_KEY)),
                             SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
               for e in EVAL_ERAS}
    src_state = ck["source"]
    l2sp_kw = lambda arm: ({"l2sp_ref": src_state, "l2sp_lambda": L2SP_LAMBDA}
                           if arm == "C_l2sp" else {})

    print("=" * 78)
    print("SEED FLOOR — Run B vs Run B_mirror, every seed kept   <<< SHARE THIS BLOCK")
    print("=" * 78)
    for a in ARMS:
        probe = set_arm(TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)), a)
        print(f"  {a:<10} adaptable params: {count_trainable(probe):,}")
    print(f"  seeds: {SEEDS}\n")

    # ── ADAPT: one trajectory per (arm, seed), schedule replayed verbatim ──
    preds = {a: {} for a in ARMS}
    best_lr = {}
    # Adaptation cohorts depend on the ERA only, never on the arm or the seed,
    # but were rebuilt inside both loops — up to 45 constructions per arm where 3
    # would do. Dataset construction is the dominant cost in this pipeline, so
    # this is the single largest speed-up available and it changes no number:
    # the objects are read-only and identical on every rebuild.
    STEP_DS = {}
    for t in STEPS:
        c = COH[t]
        if not c.get("adapt_train_stays"):
            continue
        _tr = ICUDataset(rows(c["adapt_train_stays"] + c.get("replay_stays", [])),
                         SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
        _es = ICUDataset(rows(c["adapt_val_es_stays"]),
                         SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
        _sel = ICUDataset(rows(c["adapt_val_sel_stays"]),
                          SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
        STEP_DS[t] = (_tr, _es, _sel,
                      make_adapt_criterion(_tr, max_weight=AL["max_weight"],
                                           gamma=AL["gamma"],
                                           label_smoothing=AL["label_smoothing"]))
        print(f"  cohort {t}: train={len(_tr)} es={len(_es)} sel={len(_sel)}", flush=True)

    sel_auc = {}
    for arm in ARMS:
        for s in SEEDS:
            torch.manual_seed(s)
            np.random.seed(s)
            model = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
            model.load_state_dict(src_state)
            set_arm(model, arm)
            for t in STEPS:
                if t not in STEP_DS:
                    continue
                a_tr, a_es, a_sel, crit = STEP_DS[t]
                if arm not in best_lr:      # each arm sweeps its own LR — §10 fairness
                    cand = {}
                    for lr in LR_ADAPT_GRID:
                        torch.manual_seed(SEEDS[0])
                        m2 = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
                        m2.load_state_dict(src_state)
                        set_arm(m2, arm)
                        m2 = fit(m2, a_tr, a_es, crit, lr, ADAPT_EPOCHS, ADAPT_PATIENCE,
                                 **l2sp_kw(arm))
                        cand[lr] = auroc_mean(predict(m2, a_sel), a_sel)
                    best_lr[arm] = max(cand, key=cand.get)
                    print(f"  {arm}: LR sweep "
                          f"{({f'{k:g}': round(v,4) for k,v in cand.items()})} "
                          f"→ {best_lr[arm]:g}")
                model = fit(model, a_tr, a_es, crit, best_lr[arm],
                            ADAPT_EPOCHS, ADAPT_PATIENCE, **l2sp_kw(arm))
                if SAVE_STATES:             # weights after each step (explanation analysis)
                    STATES.setdefault(f"{arm}__seed{s}", []).append(
                        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            preds[arm][s] = {e: predict(model, eval_ds[e]) for e in EVAL_ERAS}
            # Selection-split score per seed, so downstream median-seed choices never
            # look at evaluation data (audit item 4; correction 2026-10-07).
            _last_sel = STEP_DS[[t for t in STEPS if t in STEP_DS][-1]][2]
            sel_auc.setdefault(arm, {})[s] = auroc_mean(predict(model, _last_sel), _last_sel)
            print(f"  {arm} seed {s} done", flush=True)

    # ── PER-SEED TABLE ──
    key = lambda e: e.replace(" ", "").replace("-", "_").replace("(", "").replace(")", "")
    out, resolved, total = [], 0, 0
    print("\n" + "=" * 78)
    print("PER-SEED AUROC AND PAIRED DIFFERENCE")
    print("=" * 78)
    for i, lbl in enumerate(LABEL_COLS):
        print(f"\n{lbl.replace('label_','')} (incident)")
        for e in EVAL_ERAS:
            k = key(e)
            lab = eval_ds[e].labels
            g = eval_ds[e].meta.get(INCIDENT_GATE[lbl])
            m = (np.asarray(g) <= 0) if g is not None and np.size(g) else np.ones(len(lab), bool)
            y = lab[m, i]
            if not (0 < y.sum() < len(y)):
                continue
            per = {a: np.array([roc_auc_score(y, preds[a][s][e][m, i]) for s in SEEDS])
                   for a in ARMS}
            powered = "source" not in e.lower() and y.sum() >= 50
            print(f"  {e[-9:]:<12} events={int(y.sum()):>5}" + ("" if powered else "   (not powered / probe)"))
            for a in ARMS:
                print(f"    {a:<12} {per[a].mean():.4f} ± {per[a].std(ddof=1):.4f}   "
                      + " ".join(f"{v:.4f}" for v in per[a]))
            cell = dict(label=lbl, era=e, events=int(y.sum()), powered=bool(powered),
                        per_seed={a: per[a].tolist() for a in ARMS},
                        mean={a: float(per[a].mean()) for a in ARMS},
                        sd={a: float(per[a].std(ddof=1)) for a in ARMS}, contrasts={})
            for x, z in CONTRASTS:
                d = per[x] - per[z]                       # paired within seed
                ok = abs(d.mean()) > d.std(ddof=1)
                pb = paired_patient_boot(
                    y, np.stack([preds[x][s][e][m, i] for s in SEEDS]),
                    np.stack([preds[z][s][e][m, i] for s in SEEDS]))
                sup = pb["ci95"][0] > 0                   # superiority, two-sided 95%
                ni = pb["ci90"][0] > -NI_MARGIN           # non-inferiority, one-sided 5%
                tag = "superior" if sup else ("non-inferior" if ni else "inconclusive")
                print(f"    {x + ' − ' + z:<14} seed {d.mean():+.4f} ± {d.std(ddof=1):.4f} "
                      f"sign {int((d > 0).sum())}/{len(d)}{' [resolved]' if ok else ''} | "
                      f"patient {pb['point']:+.4f} 95%[{pb['ci95'][0]:+.4f},{pb['ci95'][1]:+.4f}] "
                      f"90%lo {pb['ci90'][0]:+.4f} → {tag}")
                cell["contrasts"][f"{x}_minus_{z}"] = dict(
                    paired_mean=float(d.mean()), paired_sd=float(d.std(ddof=1)),
                    sign_wins=int((d > 0).sum()), resolved=bool(ok), patient=pb,
                    superior=bool(sup), non_inferior=bool(ni), verdict=tag)
                if powered and (x, z) == ("B", "B_mirror"):
                    total += 1
                    resolved += int(ok)
            out.append(cell)

    print("\n" + "-" * 78)
    print(f"  {resolved}/{total} powered TEST cells exceed their own seed sd.")
    if total and resolved >= max(1, total - 1):
        print("\n  → The B vs B_mirror gap is larger than seed noise in essentially every")
        print("    cell. Quote the per-seed table, not the median. This is the number")
        print("    that was missing.")
    elif total and resolved >= total // 2:
        print("\n  → Mixed. The gap exceeds seed noise in some cells and not others.")
        print("    Report the per-cell table and say so; do not quote a single median.")
    else:
        print("\n  → NOT RESOLVED. The gap is inside seed-to-seed variation. The")
        print("    architecture claim cannot be made on this evidence — report the")
        print("    dispersion and state that the comparison is underpowered.")

    dump = {}
    for e in EVAL_ERAS:
        k = key(e)
        dump[f"labels__{k}"] = eval_ds[e].labels
        dump[f"stayids__{k}"] = np.array(eval_ds[e].stay_ids)
        for a in ARMS:
            for s in SEEDS:
                dump[f"probs__{k}__{a}__seed{s}"] = preds[a][s][e]
    for a in ARMS:
        for s in SEEDS:
            dump[f"selauc__{a}__seed{s}"] = np.array(sel_auc[a][s])
    np.savez(SAVE_PATH / f"per_era_predictions_seeds{OUT_SUFFIX}.npz",
             eval_eras=np.array(EVAL_ERAS, dtype=object), **dump)

    Path(SAVE_PATH / f"seed_floor{OUT_SUFFIX}.json").write_text(json.dumps(to_serialisable({
        "arms": ARMS, "l2sp_lambda": L2SP_LAMBDA,
        "seeds": SEEDS, "best_lr": best_lr, "sel_auc": sel_auc, "resolved": resolved, "total": total,
        "ni_margin": NI_MARGIN, "contrasts": [f"{x}_minus_{z}" for x, z in CONTRASTS],
        "selection_metric": "incident AUROC (mean over labels)",
        "cells": out}), indent=2))
    if SAVE_STATES:
        torch.save({"states": STATES, "steps": [t for t in STEPS if t in STEP_DS],
                    "source": {k: v.cpu() for k, v in src_state.items()}, "best_lr": best_lr},
                   SAVE_PATH / f"nn_states{OUT_SUFFIX}.pt")
        print(f"Saved → {SAVE_PATH/f'nn_states{OUT_SUFFIX}.pt'}")
    print(f"\nSaved → {SAVE_PATH/f'seed_floor{OUT_SUFFIX}.json'}")
    print(f"Saved → {SAVE_PATH/f'per_era_predictions_seeds{OUT_SUFFIX}.npz'}")


if __name__ == "__main__":
    main()
