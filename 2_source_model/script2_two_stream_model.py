"""
script2_two_stream_model.py
═══════════════════════════
Drift detection + adaptation, with the comparisons the two-stream claim needs.

WHAT CHANGED, AND WHY
─────────────────────
1. FORGETTING IS NOW MEASURED. Previously `results["Test-Pre"]` assigned Run A's
   pre-drift metrics to B and C, so the adapted models were never evaluated on
   old-era data at all. "Two-stream keeps performance better after adaptation" is
   a backward-transfer claim, and the number that supports it is adapted-B on
   pre-drift data. That number is now computed for every arm, on both the
   pre-drift test split and the held-out source-era calibration split.

2. THE ARMS ARE MATCHED. Every arm sweeps its own learning rate on adapt_val and
   is reported at its own best setting. Full fine-tuning at a partial-fine-tuning
   LR is the textbook way to manufacture catastrophic forgetting, and a reviewer's
   first sentence would have been "Run C just needed a smaller LR".

3. TWO NEW CONTROL ARMS.
     B-mirror  freeze treat+fusion, adapt physio — the direct test of whether the
               DECOMPOSITION matters or whether freezing any large block helps.
     C-L2SP    full fine-tune anchored to source weights — the test of whether
               Run B is just regularization wearing an architecture costume.
   Without these two, a B > C result is not attributable to the two-stream prior.

4. RUN A IS A REFERENCE LINE, NOT A COMPETITOR. A is frozen and never sees the
   adaptation split; B does. B > A only shows that training on in-domain data
   helps. The defended claims are B vs C, B vs C-L2SP, B vs B-mirror, and B vs D.

5. DRIFT DETECTION IS SPLIT INTO A DEPLOYABLE BRANCH AND AN ORACLE.
   The performance branch needs ground-truth h8-h14 labels, which do not exist at
   deployment time. The PSI/KS branch is the actual detector; the label-based one
   is an upper bound. Which branch fired is recorded and reported.
   The distribution threshold is now calibrated against a null distribution built
   from random splits of the training era, replacing the JUMP_FACTOR magic number.
   The performance reference is a held-out source-era split that was used for
   neither early stopping nor model selection, so selection optimism on `val` no
   longer masquerades as drift.

6. THE EVAL SET IS NEVER TOUCHED BY DETECTION. Detection cohorts are drawn per
   test group BEFORE any detection runs; the final held-out patients are in the
   complement. Previously the changepoint was chosen by looking at AUROC on data
   that included the final evaluation patients.

7. INCIDENT vs PREVALENT REPORTING. Under the ongoing-need framing, a patient on
   norepinephrine from h1 to h20 is a positive, and h0-h5 features identify them
   trivially. Every metric is now reported overall / incident / prevalent, with
   incident as the primary endpoint.
"""
import json
import os
import warnings
import copy
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt

from utils.constants import (
    SEQ_FEATURES, TREATMENT_FEATURES, BINARY_COLS, METADATA_COLS, INCIDENT_GATE,
    LABEL_COLS, SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM, LSTM_LAYERS, BATCH_SIZE,
    DROPOUT, LR_INIT, LR_ADAPT_GRID, EPOCHS, ADAPT_EPOCHS, PATIENCE,
    ADAPT_PATIENCE, BUFFER_SIZE, PSI_THRESH, L2SP_LAMBDA,
    SOURCE_SEEDS, ADAPT_SEEDS, ABLATE_LEAKY_PHYSIO, LEAKY_PHYSIO_CANDIDATES,
    N_BOOT, NULL_DRAWS, FAST_MODE,
)
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize, ICUDataset
from utils.train_utils import FocalBCEWithLogitsLoss, compute_pos_weights, make_adapt_criterion
from utils.eval_utils import (
    bootstrap_ci, paired_bca, metrics_from_probs, stratified_metrics,
    prevalence_audit, retention_table, summarize_arm, to_serialisable,
    backward_transfer, average_accuracy, format_retention_matrix,
    bootstrap_cache, paired_bca_cached,
)
from models.architectures import TwoStreamModel, count_trainable, l2sp_penalty
from utils.drift_utils import (compute_psi, distribution_drift, calibrate_threshold,
                               year_ratio_null)

from utils.drift_explainability import (
    DriftExplainer,
    compute_physio_observed_psi,
    compute_spearman_stability,
)   # compute_rolling_psi_delay intentionally NOT imported — script6 owns delay

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_PATH = Path("./data")
SAVE_PATH = Path(".")
# ── PIN_SPLIT (correction 2026-10-07) ──────────────────────────────────────────
# The cohort assignment below is not deterministic across runs: polars group_by /
# unique return rows in arbitrary order and the seeded tie-breakers are attached by
# row position. PIN_SPLIT=<path to an eval_split.json> reuses that file's cohorts
# exactly (validation partition, global adapt/eval subjects, per-era adapt-train /
# val-ES / val-SEL, replay stays), so a rerun differs from it only by intended
# changes. At the end the written split is checked against the pinned one.
FORCE_ADAPT = os.environ.get("FORCE_ADAPT", "0") == "1"
print(f"FORCE_ADAPT = {FORCE_ADAPT}")
PIN = None
if os.environ.get("PIN_SPLIT"):
    PIN = json.loads(Path(os.environ["PIN_SPLIT"]).read_text())["sequential"]
    print(f"PIN_SPLIT: cohorts pinned to {os.environ['PIN_SPLIT']}")
TRAIN_YEARS = ["2008 - 2010", "2011 - 2013"]

# Distribution branch
MIN_FEAT_DRIFTED = 2
BINARY_DELTA_THR = 0.05
NULL_QUANTILE = 99.0        # empirical floor; see NULL_SD_MULT below
NULL_SD_MULT = 3.0          # threshold = max(empirical p99, mean + 3*sd)

# Stage-2 calibration (drift_utils.year_ratio_null). Without it the random-half
# null treats normal year-to-year variation as drift. See DESIGN_RATIONALE §8.
RATIO_SAFETY = 1.0
MIN_YEAR_N = 60             # a reference year smaller than this cannot be scored

# Only label-free branches may fire an update. The performance branch needs
# h8-h14 outcomes that do not exist at prediction time — it is an oracle, and it
# is measured against the deployed model for reporting only. §8.
GATE_BRANCHES = ("distribution",)          # add "performance" to let the oracle gate

# Performance branch (oracle only — needs labels, never gates)
AUROC_DRIFT_THR = 0.020     # floor; the operative threshold is max(this, 2*SE)
# Never exceed the number of active labels: with a single label (BIOLOGICAL_LABEL
# mode) a fixed threshold of 2 can never be met, so the oracle branch would
# silently report zero degradation and zero missed updates for the whole run —
# a false all-clear rather than an error. Clamped, it stays meaningful at any
# label count while leaving the two-label default unchanged.
MIN_PERF_DROPS = max(1, min(2, len(LABEL_COLS)))

# Guard on the freeze premise. "flag" = observational, cannot alter the primary
# comparison. "unfreeze" = escalate, releasing the freeze while violated. §8.
PHYSIO_DRIFT_ACTION = "flag"               # "flag" | "unfreeze"
MIN_PHYSIO_FEAT_DRIFTED = 3                # stricter than treatment: 3 of ~51

# Post-drift subject partition
FRAC_ADAPT_TRAIN = 0.30
FRAC_ADAPT_VAL_ES = 0.10    # early stopping
FRAC_ADAPT_VAL_SEL = 0.10   # LR + seed selection (kept separate from ES)
# remainder → held-out evaluation

# Source-era val partition
FRAC_VAL_CALIB = 0.35       # never used for early stopping or model selection

LEAKAGE_TIMING_FEATS = ["time_to_first_abx_order_hrs"]
SENTINEL_NO_EARLY_EVENT = float(SEQ_LEN + 1)


torch.manual_seed(SEED)
np.random.seed(SEED)

# ── SPEED ─────────────────────────────────────────────────────────────────────
# EVAL_BATCH only affects how many stays are scored per forward pass — it cannot
# change any number, and evaluation happens far more often than training here
# (every arm on every era after every step).
EVAL_BATCH = 512
torch.backends.cudnn.benchmark = True   # fixed-shape LSTM: safe, picks fast kernels


print(f"\nFeature configuration: {len(SEQ_FEATURES)} physio + {len(TREATMENT_FEATURES)} treatment "
      f"= {len(SEQ_FEATURES) + len(TREATMENT_FEATURES)} channels")
print(f"ABLATE_LEAKY_PHYSIO = {ABLATE_LEAKY_PHYSIO}"
      + (f" (dropped: {LEAKY_PHYSIO_CANDIDATES})" if ABLATE_LEAKY_PHYSIO else
         "  → run again with True and report the ablation"))

# ── DATA ───────────────────────────────────────────────────────────────────────
print("\nLoading data...")
train_df = load_enriched_split(BASE_PATH, "train", SEQ_FEATURES, TREATMENT_FEATURES)
val_df = load_enriched_split(BASE_PATH, "val", SEQ_FEATURES, TREATMENT_FEATURES)
test_df = load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES)

val_groups = set(val_df["anchor_year_group"].unique().to_list())
assert val_groups and val_groups <= set(TRAIN_YEARS), f"Val leakage into non-train years: {val_groups}"
print(f"✅ Val temporal integrity confirmed: {val_groups}")

# Metadata must be present or incident/prevalent reporting is impossible.
HAS_META = all(c in test_df.columns for c in ["already_vaso", "already_vent"])
if not HAS_META:
    print("⚠ already_vaso / already_vent absent — re-run script1. "
          "Falling back to unstratified reporting; the incident endpoint will be unavailable.")

def _clip_timing(df):
    for feat in LEAKAGE_TIMING_FEATS:
        if feat in df.columns:
            df = df.with_columns(
                pl.when(pl.col(feat) > SEQ_LEN).then(SENTINEL_NO_EARLY_EVENT)
                  .otherwise(pl.col(feat)).alias(feat))
    return df


train_df, val_df, test_df = _clip_timing(train_df), _clip_timing(val_df), _clip_timing(test_df)

all_norm_cols = sorted(set(SEQ_FEATURES + TREATMENT_FEATURES))
train_stats = calculate_train_stats(train_df, all_norm_cols)
train_raw, test_raw = train_df.clone(), test_df.clone()

train_df = normalize(train_df, train_stats)
val_df = normalize(val_df, train_stats)
test_df = normalize(test_df, train_stats)

# Metadata columns are NOT dropped any more — they are needed for stratification.
# The guarantee that they never reach the model is structural: ICUDataset only
# selects columns present in SEQ_FEATURES / TREATMENT_FEATURES, and constants.py
# asserts those lists are disjoint from METADATA_COLS.
_leaked = set(METADATA_COLS) & set(SEQ_FEATURES + TREATMENT_FEATURES)
assert not _leaked, f"metadata reached the feature set: {_leaked}"

train_ds = ICUDataset(train_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
val_ds = ICUDataset(val_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)

seq_dim, treat_dim = len(train_ds.seq_cols), len(train_ds.treat_cols)
print(f"seq_dim={seq_dim} treat_dim={treat_dim}")

# ── SOURCE-ERA VAL PARTITION ───────────────────────────────────────────────────
# val is used for early stopping AND for median-source selection, which makes its
# AUROC selection-optimistic. Comparing a test group against that optimistic
# number guarantees an apparent "drop" before any drift exists. val_calib is
# carved out first and touched by neither, so it is a clean source-era reference
# for both the drift detector and the retention measurement.
_val_subj = val_df.filter(pl.col("hrs_from_admit") == 0)["subject_id"].unique().to_list()
_rng_val = np.random.default_rng(SEED)
_rng_val.shuffle(_val_subj)
_n_calib = int(len(_val_subj) * FRAC_VAL_CALIB)
val_calib_subj, val_es_subj = set(_val_subj[:_n_calib]), set(_val_subj[_n_calib:])
if PIN is not None:
    _pc = set(int(x) for x in PIN["era_eval_stays"][PIN["source_era_key"]])
    _vt0 = val_df.filter(pl.col("hrs_from_admit") == 0)
    val_calib_subj = set(_vt0.filter(pl.col("stay_id").is_in(list(_pc)))["subject_id"].to_list())
    val_es_subj = set(_vt0["subject_id"].to_list()) - val_calib_subj

val_calib_df = val_df.filter(pl.col("subject_id").is_in(list(val_calib_subj)))
val_es_df = val_df.filter(pl.col("subject_id").is_in(list(val_es_subj)))
val_es_ds = ICUDataset(val_es_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
val_calib_ds = ICUDataset(val_calib_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
print(f"Source-era val: {len(val_es_ds)} stays for early stopping / selection, "
      f"{len(val_calib_ds)} stays held clean for drift reference + retention")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_es_loader = DataLoader(val_es_ds, batch_size=EVAL_BATCH, shuffle=False)

pos_weights = compute_pos_weights(train_ds, max_weight=20.0)
criterion = FocalBCEWithLogitsLoss(pos_weight=pos_weights, gamma=1.5, label_smoothing=0.02)

# ── TRAIN / EVAL HELPERS ───────────────────────────────────────────────────────
_DS_CACHE = {}


def cached_ds(df, key):
    """eval_on_split used to rebuild the whole dataset on every call, including
    inside the 5-seed x group-pair drift loop. Datasets are immutable here, so
    cache them by an explicit key."""
    if key not in _DS_CACHE:
        _DS_CACHE[key] = ICUDataset(df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    return _DS_CACHE[key]


def train_epoch(model, loader, optimizer, crit, l2sp_ref=None, l2sp_lambda=0.0):
    model.train()
    total_loss, n = 0, 0
    for x_seq, x_treat, y in loader:
        x_seq, x_treat, y = x_seq.to(device), x_treat.to(device), y.to(device)
        loss = crit(model(x_seq, x_treat), y)
        if l2sp_ref is not None and l2sp_lambda > 0:
            loss = loss + l2sp_lambda * l2sp_penalty(model, l2sp_ref, device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    all_logits, all_labels, total_loss, n = [], [], 0, 0
    for x_seq, x_treat, y in loader:
        x_seq, x_treat, y = x_seq.to(device), x_treat.to(device), y.to(device)
        logits = model(x_seq, x_treat)
        total_loss += crit(logits, y).item() * y.size(0)
        n += y.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    metrics = {"loss": total_loss / max(n, 1)}
    metrics.update(metrics_from_probs(labels, probs, LABEL_COLS))
    return metrics, probs, labels


def eval_ds(model, ds, crit):
    return evaluate(model, DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False), crit)


def eval_on_split(model, df, crit, key=None):
    ds = cached_ds(df, key) if key else ICUDataset(
        df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    return evaluate(model, DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False), crit)


def train_model(model, tr_loader, v_loader, crit, lr, epochs, patience, tag="",
                l2sp_ref=None, l2sp_lambda=0.0, verbose=True, weight_decay=1e-5):
    # weight_decay is 1e-5 in BOTH phases, matching script3 so Runs B/C/D are
    # optimised identically.
    #
    # Known, quantified asymmetry: torch Adam couples weight decay into the
    # gradient (L2, not decoupled as in AdamW), so the penalty is divided by
    # sqrt(v_hat) and bites hardest on low-gradient parameters. It also applies
    # only to TRAINABLE weights, so Run B's frozen physio stream never pays it
    # while Run C's does. Magnitude: at a local optimum the loss gradient is
    # zero-mean noise of scale s while wd*w is a constant bias, so Adam's first
    # moment keeps the bias and the drift is ~ T*lr*wd/s. With lr=3e-4, wd=1e-5
    # and s~1e-3 (implied by clipping at norm 1.0 over ~95k params), that is
    # ~0.15% per adaptation phase and ~0.45% over all three steps — and less
    # than that in function space, since LayerNorm after the LSTM absorbs a
    # near-uniform shrinkage. Not large enough to move AUROC.
    #
    # Setting it to 0 was considered and REJECTED: on a few-hundred-stay
    # adaptation set it is real overfitting protection, and it protects the arm
    # with 95k trainable parameters far more than the one with 12k. Removing it
    # would trade a negligible asymmetry for a larger one pointing the same way.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_loss, best_state, wait = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        t_loss = train_epoch(model, tr_loader, optimizer, crit, l2sp_ref, l2sp_lambda)
        v_met, _, _ = evaluate(model, v_loader, crit)
        scheduler.step(v_met["loss"])
        if v_met["loss"] < best_loss:
            best_loss, best_state, wait = v_met["loss"], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if verbose and (ep % 5 == 0 or wait == 0):
            aurocs = [v_met.get(f"{l}_auroc", np.nan) for l in LABEL_COLS]
            print(f"  {tag}Ep {ep:2d} | tL={t_loss:.4f} vL={v_met['loss']:.4f} "
                  f"AUROC={['%.3f' % a for a in aurocs]} {'*' if wait == 0 else ''}")
        if wait >= patience:
            if verbose:
                print(f"  {tag}Early stop at ep {ep}")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


def per_label_mean_auroc(met):
    """Mean over labels, used ONLY for model selection, never reported.
    Reported numbers stay per-label because septic shock is a strict subset of
    vasopressor by construction, so a mean double-counts the vasopressor task.

    CLEAN-V2 (2026-10-08): selects on the INCIDENT AUROC whenever the metrics are
    stratified, because incident AUROC is the reported endpoint. The overall AUROC
    used before was dominated by stays already on treatment (86-93% of positives),
    so every arm's LR and median seed were tuned for an endpoint never reported.
    Falls back to overall only for a label with no incident metric."""
    return float(np.nanmean([met.get(f"incident_{l}_auroc", met.get(f"{l}_auroc", np.nan))
                             for l in LABEL_COLS]))


def sel_auroc(model, ds):
    """Selection score on a dataset: mean incident AUROC (see per_label_mean_auroc)."""
    met, probs, labels = eval_ds(model, ds, criterion)
    met.update(stratified_metrics(ds, labels, probs, LABEL_COLS, INCIDENT_GATE))
    return per_label_mean_auroc(met)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SOURCE ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + f"\nPHASE 1: Source ensemble ({len(SOURCE_SEEDS)} seeds)\n" + "=" * 70)
all_source_states, all_source_calib_mets = [], []
for src_seed in SOURCE_SEEDS:
    print(f"\n--- Source seed {src_seed} ---")
    torch.manual_seed(src_seed)
    np.random.seed(src_seed)
    m_src = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
    m_src = train_model(m_src, train_loader, val_es_loader, criterion,
                        LR_INIT, EPOCHS, PATIENCE, f"SRC-s{src_seed} ")
    all_source_states.append(copy.deepcopy(m_src.state_dict()))
    all_source_calib_mets.append(eval_ds(m_src, val_calib_ds, criterion)[0])

# Median source is chosen on val_es (the selection split). val_calib is left
# clean so it can serve as an unbiased source-era reference for both the drift
# detector and the retention measurement.
_sel_scores = []
for st in all_source_states:
    _m = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
    _m.load_state_dict(st)
    _sel_scores.append(sel_auroc(_m, val_es_ds))
    del _m
_sel_scores = np.array(_sel_scores)

_median_src_idx = int(np.argsort(_sel_scores)[len(_sel_scores) // 2])
source_state = all_source_states[_median_src_idx]
_median_src_seed = SOURCE_SEEDS[_median_src_idx]
print(f"\nCanonical source: seed {_median_src_seed} "
      f"(val_es selection score={_sel_scores[_median_src_idx]:.4f})")

source_model = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
source_model.load_state_dict(source_state)
source_model.freeze_all()

model_A = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model_A.load_state_dict(source_state)
model_A.freeze_all()

# Report adaptable capacity per arm so the comparison is auditable.
_cap = {}
for _arm, _fn in [("B", "unfreeze_adaptive"), ("B_mirror", "unfreeze_mirror"),
                  ("C", "unfreeze_all"), ("C_l2sp", "unfreeze_all")]:
    _m = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
    getattr(_m, _fn)()
    _cap[_arm] = count_trainable(_m)
    del _m
print("\nAdaptable parameters per arm:")
for k, v in _cap.items():
    print(f"  {k:<10} {v:,}")
print("  (Run D in script3 is capacity-matched to Run B via solve_head_hidden)")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2a — COHORT PRE-ASSIGNMENT (before any detection touches the data)
# ══════════════════════════════════════════════════════════════════════════════
raw_all_t0 = pl.concat([train_raw.filter(pl.col("hrs_from_admit") == 0),
                        test_raw.filter(pl.col("hrs_from_admit") == 0)])
ALL_GROUPS_ORDERED = sorted(raw_all_t0["anchor_year_group"].unique().to_list())
TRAIN_GROUP_SET = set(TRAIN_YEARS)
TEST_GROUP_SET = set(ALL_GROUPS_ORDERED) - TRAIN_GROUP_SET

# ── CHRONOLOGICAL COHORT ASSIGNMENT ───────────────────────────────────────────
# Adaptation cases must PRECEDE evaluation cases in time. A random split inside
# an era trains the update on patients who arrived after the ones it is scored
# on, which does not demonstrate that updates use only data available before
# each subsequent prediction — the objection that sinks a deployment claim.
#
# So: within each era, subjects are ordered by year and the EARLIEST
# FRAC_DETECT fraction becomes the adaptation/detection cohort, the LATEST
# becomes the held-out evaluation cohort.
#
# Two constraints have to hold simultaneously:
#   (a) chronological ordering WITHIN each era, and
#   (b) one global role per subject — `approx_year` is per STAY, so a patient
#       with stays years apart appears in two eras, and per-era independent
#       assignment would put the same person on both sides of the wall.
# Both are satisfied by assigning each subject once, in the era of their FIRST
# test-era stay, and carrying that role forward to any later stay.
#
# Resolution limit, state it in the paper: MIMIC-IV date-shifts per patient, so
# `intime` is not comparable across subjects. `approx_year` is the finest
# genuinely comparable ordering available. This is "chronological to year
# resolution", not "strictly chronological", and ties within a year are broken
# by a fixed seed. The residual violation is measured and printed below.
FRAC_DETECT = FRAC_ADAPT_TRAIN + FRAC_ADAPT_VAL_ES + FRAC_ADAPT_VAL_SEL
_test_t0 = test_df.filter(pl.col("hrs_from_admit") == 0)

_subj_first = (_test_t0.group_by("subject_id")
               .agg([pl.col("approx_year").min().alias("first_year"),
                     pl.col("anchor_year_group").sort_by("approx_year").first().alias("first_era")]))
_rng_c = np.random.default_rng(20240)
_subj_first = _subj_first.with_columns(
    pl.Series("tie_break", _rng_c.random(_subj_first.height)))

DETECT_SUBJ, EVAL_SUBJ = set(), set()
_split_years = {}
for g in sorted(TEST_GROUP_SET):
    _g = (_subj_first.filter(pl.col("first_era") == g)
                     .sort(["first_year", "tie_break"]))
    if _g.height == 0:
        continue
    _k = int(_g.height * FRAC_DETECT)
    _d = _g.head(_k)                      # earliest → adaptation
    _e = _g.tail(_g.height - _k)          # latest   → evaluation
    DETECT_SUBJ |= set(_d["subject_id"].to_list())
    EVAL_SUBJ |= set(_e["subject_id"].to_list())
    _split_years[g] = {
        "adapt_year_range": [int(_d["first_year"].min()), int(_d["first_year"].max())] if _d.height else None,
        "eval_year_range": [int(_e["first_year"].min()), int(_e["first_year"].max())] if _e.height else None,
        "boundary_year": int(_d["first_year"].max()) if _d.height else None,
    }
if PIN is not None:
    _s2s = dict(zip(_test_t0["stay_id"].to_list(), _test_t0["subject_id"].to_list()))
    _ev_st = [x for e_, v in PIN["era_eval_stays"].items() if e_ != PIN["source_era_key"] for x in v]
    _ad_st = [x for c in PIN["cohorts"].values()
              for k in ("adapt_train_stays", "adapt_val_es_stays", "adapt_val_sel_stays") for x in c.get(k, [])]
    _old = DETECT_SUBJ | EVAL_SUBJ
    EVAL_SUBJ = {_s2s[int(x)] for x in _ev_st}
    DETECT_SUBJ = {_s2s[int(x)] for x in _ad_st}
    print(f"PIN_SPLIT: {len(DETECT_SUBJ)} adapt / {len(EVAL_SUBJ)} eval subjects "
          f"({len(_old - DETECT_SUBJ - EVAL_SUBJ)} test subjects in neither)")
assert not (DETECT_SUBJ & EVAL_SUBJ), "global subject partition overlaps"

print("\nChronological cohort assignment (earliest → adapt, latest → evaluate):")
for g, r in _split_years.items():
    print(f"  {g:<14} adapt years {r['adapt_year_range']}  →  eval years {r['eval_year_range']}")

detect_subjects, eval_subjects = {}, {}
for g in sorted(TEST_GROUP_SET):
    _g_subj = set(_test_t0.filter(pl.col("anchor_year_group") == g)["subject_id"].unique().to_list())
    detect_subjects[g] = _g_subj & DETECT_SUBJ
    eval_subjects[g] = _g_subj & EVAL_SUBJ
    print(f"  {g}: {len(detect_subjects[g])} detection / {len(eval_subjects[g])} eval subjects")

# How many patients actually span more than one era group — the condition that
# made the per-group split unsafe. Worth reporting: it is also a mild confound
# for any claim that the eras are independent populations.
_subj_groups = (_test_t0.group_by("subject_id")
                        .agg(pl.col("anchor_year_group").n_unique().alias("n_groups")))
_n_multi = int(_subj_groups.filter(pl.col("n_groups") > 1).height)
print(f"  {_n_multi}/{_subj_first.height} test-era subjects have stays in more than one "
      f"era group; each is assigned to a single cohort globally.")

# ── TEMPORAL SEPARATION AUDIT ─────────────────────────────────────────────────
# Year resolution and carried-over multi-era subjects both mean the adapt/eval
# boundary is not perfectly clean. Quantify it rather than claim it is exact:
# for each era, what fraction of evaluation stays occurred in a year at or
# before the last adaptation year. This number belongs in the paper.
temporal_audit = {}
for g in sorted(TEST_GROUP_SET):
    _gt = _test_t0.filter(pl.col("anchor_year_group") == g)
    _a = _gt.filter(pl.col("subject_id").is_in(list(DETECT_SUBJ)))
    _e = _gt.filter(pl.col("subject_id").is_in(list(EVAL_SUBJ)))
    if _a.height == 0 or _e.height == 0:
        continue
    _amax = int(_a["approx_year"].max())
    _viol = int(_e.filter(pl.col("approx_year") < _amax).height)
    temporal_audit[g] = {
        "last_adapt_year": _amax,
        "first_eval_year": int(_e["approx_year"].min()),
        "n_eval_stays": _e.height,
        "n_eval_strictly_before_last_adapt": _viol,
        "frac_violating": _viol / _e.height,
    }
print("\nTemporal separation audit (adaptation must precede evaluation):")
for g, r in temporal_audit.items():
    print(f"  {g:<14} last adapt year={r['last_adapt_year']} | "
          f"{100*r['frac_violating']:.1f}% of eval stays strictly earlier "
          f"({r['n_eval_strictly_before_last_adapt']}/{r['n_eval_stays']})")
print("  Non-zero values come from year-resolution ties and multi-era patients — MIMIC-IV")
print("  date-shifts per patient, so finer-grained ordering is not recoverable. Report")
print("  this table rather than claiming strict chronological separation.")

ALL_DETECT_SUBJ = set().union(*detect_subjects.values()) if detect_subjects else set()
ALL_EVAL_SUBJ = set().union(*eval_subjects.values()) if eval_subjects else set()
assert not (ALL_DETECT_SUBJ & ALL_EVAL_SUBJ), "detection and eval cohorts overlap"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2b — DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nPHASE 2b: Drift detection\n" + "=" * 70)

EXCLUDE_FROM_DRIFT = {"n_distinct_meds", "gender_M"}
TREAT_CONT = [c for c in TREATMENT_FEATURES
              if c not in BINARY_COLS and c != "age" and c not in EXCLUDE_FROM_DRIFT]
BINARY_MONITOR = [c for c in TREATMENT_FEATURES if c in BINARY_COLS and c not in EXCLUDE_FROM_DRIFT]
# ALL non-mask physiology channels (was an arbitrary SEQ_FEATURES[:30]). The
# freeze decision is a bet on this stream being stable, so it must be watched.
PHYSIO_MONITOR = [c for c in SEQ_FEATURES
                  if not c.endswith("_mask") and c in raw_all_t0.columns]
print(f"\nMonitored channels: {len(TREAT_CONT)} treatment continuous + "
      f"{len(BINARY_MONITOR)} treatment binary | {len(PHYSIO_MONITOR)} physiology")


# compute_psi / distribution_drift now live in utils.drift_utils so script6's
# streaming detector scores drift with the identical implementation.

group_t0 = {g: raw_all_t0.filter(pl.col("anchor_year_group") == g) for g in ALL_GROUPS_ORDERED}

def _edges_for(cols):
    out = {}
    for c in cols:
        refs = [group_t0[g][c].to_numpy().astype(float)
                for g in ALL_GROUPS_ORDERED if g in TRAIN_GROUP_SET and c in group_t0[g].columns]
        if refs and len(np.concatenate(refs)) >= 10:
            e = np.percentile(np.concatenate(refs), np.linspace(0, 100, 11))
            e[0] -= 1e-6
            e[-1] += 1e-6
            out[c] = e
    return out


ref_edges = _edges_for(TREAT_CONT)
physio_edges = _edges_for(PHYSIO_MONITOR)

train_era_t0 = raw_all_t0.filter(pl.col("anchor_year_group").is_in(TRAIN_YEARS))
train_year_frames = {int(y): train_era_t0.filter(pl.col("approx_year") == y)
                     for y in sorted(train_era_t0["approx_year"].unique().to_list())}

# ── TWO-STAGE NULL CALIBRATION (DESIGN_RATIONALE §8) ──────────────────────────
# Stage 1: random half-splits of the training era remove SAMPLING noise, rebuilt
#          per batch size because PSI noise scales with 1/n.
# Stage 2: rescale by the largest year-to-year shift inside the training era, so
#          normal annual variation is not read as drift.
# script2 had stage 1 only and script6 had both — which is why they disagreed
# about whether drift occurred at all.
_thr_cache = {}


def threshold_for(n, cont_cols=None, bin_cols=None, edges=None, tag="treat"):
    cont_cols = TREAT_CONT if cont_cols is None else cont_cols
    bin_cols = BINARY_MONITOR if bin_cols is None else bin_cols
    edges = ref_edges if edges is None else edges
    n_eff = int(min(n, train_era_t0.height // 2))
    bucket = max(50, int(round(n_eff / 50.0)) * 50)
    ck = (tag, bucket)
    if ck not in _thr_cache:
        _thr_cache[ck] = calibrate_threshold(
            train_era_t0, cont_cols, bin_cols, batch_size=bucket, n_draws=NULL_DRAWS,
            ref_edges=edges, quantile=NULL_QUANTILE, sd_mult=NULL_SD_MULT,
            psi_thresh=PSI_THRESH, binary_delta_thr=BINARY_DELTA_THR)
    return _thr_cache[ck][0]


treat_ratio_thr, treat_ratio_info = year_ratio_null(
    train_year_frames, TREAT_CONT, BINARY_MONITOR,
    threshold_fn=lambda n: threshold_for(n, tag="treat"),
    ref_edges=ref_edges, psi_thresh=PSI_THRESH, binary_delta_thr=BINARY_DELTA_THR,
    min_year_n=MIN_YEAR_N, safety=RATIO_SAFETY)

physio_ratio_thr, physio_ratio_info = year_ratio_null(
    train_year_frames, PHYSIO_MONITOR, [],
    threshold_fn=lambda n: threshold_for(n, cont_cols=PHYSIO_MONITOR, bin_cols=[],
                                         edges=physio_edges, tag="physio"),
    ref_edges=physio_edges, psi_thresh=PSI_THRESH, binary_delta_thr=BINARY_DELTA_THR,
    min_year_n=MIN_YEAR_N, safety=RATIO_SAFETY)

print(f"\nStage-2 calibration on {treat_ratio_info['n_reference_years']} train-era years "
      f"{treat_ratio_info['years']}:")
for _nm, _t, _i in (("treatment", treat_ratio_thr, treat_ratio_info),
                    ("physiology", physio_ratio_thr, physio_ratio_info)):
    print(f"  {_nm:<11} rolling ratios "
          f"{[round(x, 2) for x in _i['ratios']['rolling']]} → threshold {_t['rolling']:.2f}"
          + ("" if _i["rolling_active"] else "   ⚠ INACTIVE — too few reference years"))
if not treat_ratio_info["rolling_active"]:
    print("  ⚠ The second stage could not be calibrated. Detection reduces to the")
    print("    sampling-noise null, which over-fires on normal annual variation.")
    print("    Report the detector as stage-1 only, or widen TRAIN_YEARS.")

# ── Performance branch reference (ORACLE — needs labels, not deployable) ──
src_calib_auroc, src_calib_se = {}, {}
_calib_labels = val_calib_ds.labels
_calib_probs_per_seed = []
for st in all_source_states:
    _m = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
    _m.load_state_dict(st)
    _m.freeze_all()
    _, _p, _ = eval_ds(_m, val_calib_ds, criterion)
    _calib_probs_per_seed.append(_p)
    del _m
for i, lbl in enumerate(LABEL_COLS):
    vals = [roc_auc_score(_calib_labels[:, i], p[:, i]) for p in _calib_probs_per_seed
            if 0 < _calib_labels[:, i].sum() < len(_calib_labels)]
    src_calib_auroc[lbl] = float(np.mean(vals)) if vals else np.nan
    # Sampling SE of the calibration AUROC — the drop has to clear the noise
    # floor of the reference itself before it means anything.
    if vals:
        _, lo, hi = bootstrap_ci(_calib_labels[:, i], _calib_probs_per_seed[0][:, i],
                                 roc_auc_score, n_boot=400, seed=5)
        src_calib_se[lbl] = float((hi - lo) / 3.92) if hi == hi else np.nan
    else:
        src_calib_se[lbl] = np.nan
perf_thresholds = {lbl: max(AUROC_DRIFT_THR, 2.0 * src_calib_se.get(lbl, 0.0))
                   for lbl in LABEL_COLS}
print("\nPerformance-branch reference (val_calib — clean source-era split):")
for lbl in LABEL_COLS:
    print(f"  {lbl:<24} AUROC={src_calib_auroc[lbl]:.4f} "
          f"SE={src_calib_se[lbl]:.4f} → threshold={perf_thresholds[lbl]:.4f}")
print(f"  MIN_PERF_DROPS={MIN_PERF_DROPS} of {len(LABEL_COLS)} active labels.")
if "label_septic_shock" in LABEL_COLS:
    print("  NOTE: septic shock is a strict subset of vasopressor by construction, so")
    print("  this is closer to 1.5 independent signals than 2.")


def ensemble_perf_drop(detect_df, key, states=None):
    """AUROC drop on `detect_df` vs the clean source-era reference.

    `states` defaults to the SOURCE ensemble; pass the deployed trajectory to ask
    "has the model I am running degraded?" instead. §8.
    """
    states = all_source_states if states is None else states
    per_lbl = {}
    probs_all = []
    for st in states:
        _m = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
        _m.load_state_dict(st)
        _m.freeze_all()
        _, p, y = eval_on_split(_m, detect_df, criterion, key=key)
        probs_all.append(p)
        del _m
    labels = y
    for i, lbl in enumerate(LABEL_COLS):
        if not (0 < labels[:, i].sum() < len(labels)):
            per_lbl[lbl] = np.nan
            continue
        cur = float(np.mean([roc_auc_score(labels[:, i], p[:, i]) for p in probs_all]))
        per_lbl[lbl] = src_calib_auroc[lbl] - cur
    n_sig = sum(1 for lbl in LABEL_COLS
                if not np.isnan(per_lbl[lbl]) and per_lbl[lbl] > perf_thresholds[lbl])
    valid = [v for v in per_lbl.values() if not np.isnan(v)]
    return (float(np.mean(valid)) if valid else 0.0), per_lbl, n_sig


def score_pair(ref_frame, cur_frame, mode="rolling", physio_ref=None):
    """`physio_ref` overrides the reference for the PHYSIOLOGY stream only.

    Necessary because the two streams are fitted to different data. Treat+fusion
    train on every update, so their reference is the latest training mixture. The
    physiology encoder is FROZEN — it is still fitted to the source era — so
    scoring it against the latest mixture lets the reference absorb a
    physiological shift the encoder never learned, and the guard falls silent for
    the wrong reason. Observed: the gate fired at x1.94 on the first era, then
    read x0.65 and x0.68 purely because the reference had moved.
    """
    """Label-free drift of `cur_frame` vs `ref_frame`, both streams.

    Ratio > 1 means the shift exceeds the worst year-to-year shift the model was
    trained through. Compare ratios, never raw scores (raw scale tracks n).

    `mode` must match the comparison shape and the two are NOT interchangeable:
      "rolling"  ref = immediately preceding period (era-pair scan)
      "fixed"    ref = pooled multi-period reference (closed-loop gate)
    A pooled reference averages out year noise, so it is systematically quieter.
    """
    pref = ref_frame if physio_ref is None else physio_ref
    d_s, d_n, d_c, d_b = distribution_drift(
        ref_frame, cur_frame, TREAT_CONT, BINARY_MONITOR, ref_edges=ref_edges,
        psi_thresh=PSI_THRESH, binary_delta_thr=BINARY_DELTA_THR)
    p_s, p_n, p_c, _ = distribution_drift(
        pref, cur_frame, PHYSIO_MONITOR, [], ref_edges=physio_edges,
        psi_thresh=PSI_THRESH, binary_delta_thr=BINARY_DELTA_THR)
    d_t = threshold_for(min(ref_frame.height, cur_frame.height), tag="treat")
    p_t = threshold_for(min(pref.height, cur_frame.height),
                        cont_cols=PHYSIO_MONITOR, bin_cols=[],
                        edges=physio_edges, tag="physio")
    return {
        "dist_score": d_s, "n_drifted": d_n, "drifted_cont": d_c, "drifted_bin": d_b,
        "dist_stage1_thr": d_t,
        "dist_ratio": (d_s / d_t / treat_ratio_thr[mode]) if d_t > 0 else np.nan,
        "physio_score": p_s, "n_physio_drifted": p_n, "drifted_physio": p_c,
        "physio_stage1_thr": p_t,
        "physio_ratio": (p_s / p_t / physio_ratio_thr[mode]) if p_t > 0 else np.nan,
        "n_ref": ref_frame.height, "n_physio_ref": pref.height,
        "n_cur": cur_frame.height, "mode": mode,
    }


# Era-pair scan — REPORTING ONLY. It produces the changepoint for the drift
# diagnostics. The adapt/no-adapt decisions are made in the walk-forward loop.
pair_results = {}
for i in range(1, len(ALL_GROUPS_ORDERED)):
    prev_g, curr_g = ALL_GROUPS_ORDERED[i - 1], ALL_GROUPS_ORDERED[i]
    if group_t0[prev_g].height < 30 or group_t0[curr_g].height < 30:
        continue
    entry = score_pair(group_t0[prev_g], group_t0[curr_g], mode="rolling")
    entry.update({"mean_auroc_drop": np.nan, "per_lbl_drops": {}, "n_perf_sig": 0})

    # Oracle branch only on test-era groups, and only on the DETECTION cohort.
    if curr_g in TEST_GROUP_SET and detect_subjects.get(curr_g):
        det_df = test_df.filter((pl.col("anchor_year_group") == curr_g) &
                                pl.col("subject_id").is_in(list(detect_subjects[curr_g])))
        if det_df.height > 0 and det_df["stay_id"].n_unique() >= 30:
            md, per_lbl, n_sig = ensemble_perf_drop(det_df, key=f"detect::{curr_g}")
            entry.update({"mean_auroc_drop": md, "per_lbl_drops": per_lbl, "n_perf_sig": n_sig})
    pair_results[(prev_g, curr_g)] = entry

# ── Fire the detector ──
drift_group, drift_branch = None, None
test_pairs = [(ALL_GROUPS_ORDERED[ALL_GROUPS_ORDERED.index(g) - 1], g)
              for g in ALL_GROUPS_ORDERED if g in TEST_GROUP_SET and ALL_GROUPS_ORDERED.index(g) > 0]
detection_log = []
for prev_g, curr_g in test_pairs:
    if (prev_g, curr_g) not in pair_results:
        continue
    res = pair_results[(prev_g, curr_g)]
    dist_fired = res["dist_ratio"] > 1.0 and res["n_drifted"] >= MIN_FEAT_DRIFTED
    physio_fired = (res["physio_ratio"] > 1.0
                    and res["n_physio_drifted"] >= MIN_PHYSIO_FEAT_DRIFTED)
    perf_fired = res["n_perf_sig"] >= MIN_PERF_DROPS
    detection_log.append({
        "pair": f"{prev_g} → {curr_g}", "dist_score": res["dist_score"],
        "dist_stage1_thr": res["dist_stage1_thr"], "dist_fired": bool(dist_fired),
        "n_drifted": res["n_drifted"], "physio_score": res["physio_score"],
        "n_physio_drifted": res["n_physio_drifted"],
        "physio_ratio": res["physio_ratio"], "physio_fired": bool(physio_fired),
        "mean_auroc_drop": res["mean_auroc_drop"], "n_perf_sig": res["n_perf_sig"],
        "perf_fired": bool(perf_fired),
        "curr_group": curr_g,
        # Stage-2 severity: multiples of the largest train-era year-to-year shift,
        # so pairs are comparable to each other AND to the variation the model was
        # trained through. Raw scores are not comparable across sample sizes.
        "dist_ratio": res["dist_ratio"],
    })

print("\n--- Detection log (ratios are multiples of the worst train-era year-to-year shift) ---")
for row in detection_log:
    print(f"  {row['pair']:<28} treat x{row['dist_ratio']:>5.2f} "
          f"{'FIRE' if row['dist_fired'] else '  - '} (n={row['n_drifted']:>2}) | "
          f"physio x{row['physio_ratio']:>5.2f} "
          f"{'FIRE' if row['physio_fired'] else '  - '} (n={row['n_physio_drifted']:>2}) | "
          f"oracle ΔAUROC={row['mean_auroc_drop']:+.4f} "
          f"({'drop' if row['perf_fired'] else '  - '}, {row['n_perf_sig']}/3)")
print(f"  Gate = {GATE_BRANCHES}. The oracle branch needs h8-h14 outcomes and is")
print("  reported as a validation of the label-free gate, not as a trigger.")
if any(r["physio_fired"] for r in detection_log):
    print("\n  ⚠ PHYSIOLOGY GATE FIRED. The freeze premise — that the physiological")
    print("    representation is stable enough to hold fixed — is violated at the")
    print(f"    boundaries above. Configured action: {PHYSIO_DRIFT_ACTION}.")
else:
    print("\n  Physiology gate never fired: the freeze premise held across every")
    print("  boundary. This is the guard condition on Run B, and it is what makes")
    print("  the freeze an audited decision rather than an assumption.")

# ── Changepoint selection ─────────────────────────────────────────────────────
# FIRST-CROSSING vs ARGMAX.
#
# First-crossing takes the earliest pair that clears the threshold. Under drift
# that is GRADUAL rather than a single event — which is what MIMIC-IV actually
# looks like, practice moves every year — every boundary clears the threshold and
# first-crossing therefore always returns the earliest test group, by
# construction rather than by evidence. That leaves no pre-drift test period.
#
# Argmax takes the LARGEST divergence among the pairs that clear the threshold.
# This is the standard binary-segmentation formulation of changepoint detection,
# and it is estimating a different quantity: "where is the dominant shift",
# rather than "where does drift first become detectable". Both are defensible;
# argmax is the one that matches the claim "we detect the drift event".
#
# This is NOT threshold tuning. The threshold is untouched and still calibrated
# against the train-era null. Raising the threshold until the changepoint moved
# where you wanted it would tune the detector on the outcome it is meant to
# support, and would additionally corrupt the retention measurement: the
# "pre-drift" period would silently contain drift you had suppressed.
CHANGEPOINT_RULE = "argmax"          # "argmax" | "first"

def gate_fired(row):
    """Did a branch ALLOWED to trigger an update fire? Excludes the oracle
    unless GATE_BRANCHES says otherwise."""
    return ((("distribution" in GATE_BRANCHES) and row["dist_fired"])
            or (("performance" in GATE_BRANCHES) and row["perf_fired"])
            or (("physiology" in GATE_BRANCHES) and row["physio_fired"]))


_fired = [r for r in detection_log if gate_fired(r)]
if _fired:
    if CHANGEPOINT_RULE == "argmax":
        _pick = max(_fired, key=lambda r: (r["dist_ratio"] if r["dist_ratio"] == r["dist_ratio"] else -np.inf))
    else:
        _pick = _fired[0]
    drift_group = _pick["curr_group"]
    drift_branch = "+".join(
        b for b, k in (("distribution", "dist_fired"), ("physiology", "physio_fired"),
                       ("performance", "perf_fired")) if _pick[k]) or "none"

if len(_fired) > 1:
    print(f"\n⚠ {len(_fired)}/{len(detection_log)} boundaries exceeded the threshold — drift in this")
    print("  cohort is GRADUAL, not a single event. That is a finding, not a nuisance:")
    print("  state it in the paper rather than tuning it away. Severity ranking:")
    for r in sorted(_fired, key=lambda r: -r["dist_ratio"]):
        print(f"    {r['pair']:<28} x{r['dist_ratio']:.2f} threshold")
    print(f"  Rule '{CHANGEPOINT_RULE}' selects the "
          f"{'largest' if CHANGEPOINT_RULE == 'argmax' else 'earliest'} one.")

# Agreement between the deployable gate and the oracle, over the reporting scan.
# This is the honest version of "the detector works": it is measured, and the
# disagreements are named. It is not a trigger.
_g = [bool(gate_fired(r)) for r in detection_log]
_o = [bool(r["perf_fired"]) for r in detection_log]
if detection_log:
    print(f"\nGate vs oracle over {len(detection_log)} boundaries: "
          f"agree {sum(a == b for a, b in zip(_g, _o))}, "
          f"gate-only {sum(a and not b for a, b in zip(_g, _o))}, "
          f"oracle-only {sum(b and not a for a, b in zip(_g, _o))}")
    if any(b and not a for a, b in zip(_g, _o)):
        print("  ⚠ The oracle saw degradation the label-free gate missed. Those are")
        print("    MISSED UPDATES under deployment conditions — report them; they are")
        print("    the cost of not having outcome labels at prediction time.")

if drift_group:
    drift_idx = ALL_GROUPS_ORDERED.index(drift_group)
    pre_groups, post_groups = ALL_GROUPS_ORDERED[:drift_idx], ALL_GROUPS_ORDERED[drift_idx:]
    print(f"\n✅ Drift onset (reporting changepoint): {drift_group} | "
          f"branches: {drift_branch} | rule: {CHANGEPOINT_RULE}")
else:
    pre_groups, post_groups = list(ALL_GROUPS_ORDERED), []
    print("\n✅ No drift detected")

# ── GUARD: pre-drift split must be non-empty ──
# TRAIN_YEARS ends at 2013, so the first test pair is 2011-2013 → 2014-2016. If
# the detector fires there, pre_groups contains ONLY train-era groups, test_pre
# is empty, and the replay buffer, the placebo control, the retention measurement
# and the pre-drift row all silently no-op. That failure was invisible before.
test_pre = (test_df.filter(pl.col("anchor_year_group").is_in(pre_groups))
                   .filter(pl.col("anchor_year_group").is_in(list(TEST_GROUP_SET)))
            if pre_groups else pl.DataFrame())
test_post = test_df.filter(pl.col("anchor_year_group").is_in(post_groups)) if post_groups else pl.DataFrame()

if drift_group and test_pre.height == 0:
    print("\n" + "!" * 70)
    print("WARNING: the changepoint landed on the FIRST test-era group, so there is no")
    print("pre-drift TEST split. Consequences, all of them silent until now:")
    print("  - no replay buffer (adaptation trains on post-drift data only)")
    print("  - no placebo adaptation control")
    print("  - retention on pre-drift test data cannot be measured")
    print(f"The changepoint rule is '{CHANGEPOINT_RULE}'. If it is 'first', switch to")
    print("'argmax': under gradual drift, first-crossing always returns the earliest")
    print("test group by construction. If it is already 'argmax', then the dominant")
    print("shift genuinely IS at the first test boundary and no design change moves it —")
    print("in that case use the sequential-adaptation design instead of pre/post.")
    print("Retention will still be measured on val_calib (source era), which is the")
    print("more conservative forgetting probe anyway. To recover a pre-drift test")
    print("split, either move the train/test era boundary later or treat the earliest")
    print("test group as a pre-drift reference by construction.")
    print("!" * 70)

pre_cp_stays = test_pre["stay_id"].unique().to_list() if test_pre.height > 0 else []
post_cp_stays = test_post["stay_id"].unique().to_list() if test_post.height > 0 else []
drift_tag = drift_group if drift_group else "no-drift"

drifted_post = []
for g in post_groups:
    idx = ALL_GROUPS_ORDERED.index(g)
    key = (ALL_GROUPS_ORDERED[idx - 1], g) if idx > 0 else None
    if key and key in pair_results:
        drifted_post.extend([c for c, _, _ in pair_results[key]["drifted_cont"]])
        drifted_post.extend([c for c, _ in pair_results[key]["drifted_bin"]])
drifted_post = sorted(set(drifted_post))

test_pre_raw = test_raw.filter(pl.col("anchor_year_group").is_in(pre_groups)) if pre_groups else pl.DataFrame()
test_post_raw = test_raw.filter(pl.col("anchor_year_group").is_in(post_groups)) if post_groups else pl.DataFrame()
def _safe(label, fn, *a, **kw):
    """Run a diagnostic and keep going if it fails.

    Everything routed through here is supplementary reporting, not a result.
    The split-name keys passed to explain_metric_gains changed in this revision
    ('Test-Post (≥ tag)' rather than '(> tag)'), and drift_explainability is
    maintained separately — a parsing mismatch there must not destroy a run that
    has already spent an hour training 4 arms x 3 LRs x 5 seeds.
    """
    try:
        return fn(*a, **kw)
    except Exception as e:
        print(f"  ⚠ diagnostic '{label}' failed ({type(e).__name__}: {e}) — continuing. "
              f"Core metrics and paired tests are unaffected.")
        return None


explainer = DriftExplainer(
    train_df=train_raw, test_pre_df=test_pre_raw, test_post_df=test_post_raw,
    treat_features=[f for f in TREATMENT_FEATURES if f not in EXCLUDE_FROM_DRIFT],
    binary_cols=BINARY_COLS, label_cols=LABEL_COLS, psi_thresh=PSI_THRESH, save_path=SAVE_PATH)
_safe("explain_drift", explainer.explain_drift)

if test_post_raw.height > 0:
    # Marginal PSI on OBSERVED values only — imputed medians otherwise pin mass
    # to a train-derived constant and make physiology look artificially stable.
    _safe("physio_psi", compute_physio_observed_psi,
          train_raw, test_post_raw, SEQ_FEATURES, SAVE_PATH)
    # The answer to "marginal stability does not establish stability of
    # multivariate relationships or outcome associations". Labels are passed so
    # the outcome half of that objection is actually measured.
    _safe("spearman_stability", compute_spearman_stability,
          train_raw, test_post_raw, SEQ_FEATURES, SAVE_PATH, label_cols=LABEL_COLS)
    # Detection delay lives in script6 (calibrated streaming detector). A second
    # estimator here would be a differently-calibrated duplicate of a number the
    # paper already reports — the failure mode that produced two PSIs and two
    # drift thresholds.

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — SEQUENTIAL ADAPTATION
# ══════════════════════════════════════════════════════════════════════════════
# Drift in this cohort is gradual, so a single pre/post changepoint mis-describes
# it: whichever era is called "pre" has itself already drifted, which corrupts
# the retention measurement the two-stream claim depends on.
#
# Instead the model walks forward through the eras: at each boundary, detect →
# adapt (only if drift fired) → evaluate on EVERY era seen so far. Nothing is
# discarded and no era leaves the evaluation, so forgetting stays visible for the
# whole trajectory. The pre/post table falls out of this for free: retention on
# earlier eras IS the pre-drift number.
#
# This is also the stronger test of the claim. Forgetting COMPOUNDS across
# repeated updates, so the gap between selective freezing and full fine-tuning
# should widen with each step — measured over three adaptation events rather
# than one.
ARMS = {
    "B":        {"setup": "unfreeze_adaptive", "l2sp": 0.0,
                 "desc": "two-stream, physio frozen (main claim)"},
    "B_mirror": {"setup": "unfreeze_mirror", "l2sp": 0.0,
                 "desc": "two-stream, treat+fusion frozen (mirror control)"},
    "C":        {"setup": "unfreeze_all", "l2sp": 0.0,
                 "desc": "two-stream, full fine-tune"},
    "C_l2sp":   {"setup": "unfreeze_all", "l2sp": L2SP_LAMBDA,
                 "desc": "two-stream, full fine-tune + L2-SP anchor to source"},
}

# Sweep the LR once, at the first adaptation step, then hold it fixed for the
# rest of the trajectory. Re-sweeping at every step would be ~3x the compute and
# would also be unrealistic: a deployed system fixes its update hyperparameters
# once, it does not retune them against future data it has not seen yet.
SWEEP_EVERY_STEP = False

TEST_ERAS = [g for g in ALL_GROUPS_ORDERED if g in TEST_GROUP_SET]
SOURCE_ERA_KEY = "source (val_calib)"

# Per-era cohorts, derived from the global subject partition made in Phase 2a.
era_detect_df, era_eval_ds, era_detect_t0_raw = {}, {}, {}
for e in TEST_ERAS:
    _e_rows = test_df.filter(pl.col("anchor_year_group") == e)
    era_detect_df[e] = _e_rows.filter(pl.col("subject_id").is_in(list(DETECT_SUBJ)))
    _ev = _e_rows.filter(pl.col("subject_id").is_in(list(EVAL_SUBJ)))
    if _ev.height > 0:
        era_eval_ds[e] = ICUDataset(_ev, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    # RAW t=0 slice of the same detection cohort. The drift score must be
    # computed on raw units: REF_WINDOW starts as the raw training era, and
    # scoring a z-scored frame against a raw one produces a large PSI that is
    # pure unit mismatch. train_raw/test_raw exist for exactly this reason.
    era_detect_t0_raw[e] = raw_all_t0.filter(
        (pl.col("anchor_year_group") == e) & pl.col("subject_id").is_in(list(DETECT_SUBJ)))

EVAL_ERAS = [SOURCE_ERA_KEY] + [e for e in TEST_ERAS if e in era_eval_ds]
eval_ds_by_era = {SOURCE_ERA_KEY: val_calib_ds, **era_eval_ds}
if PIN is not None:          # fail fast: evaluation cohorts must match the pinned split
    for e_ in EVAL_ERAS:
        _a = sorted(int(x) for x in eval_ds_by_era[e_].stay_ids)
        _b = sorted(int(x) for x in PIN["era_eval_stays"][e_])
        if _a != _b:
            raise SystemExit(f"PIN_SPLIT: eval cohort {e_} differs ({len(_a)} vs {len(_b)} stays)")
    print("PIN_SPLIT: evaluation cohorts match the pinned split")
print(f"\nEvaluation eras (held permanently in the evaluation): {EVAL_ERAS}")
for e in EVAL_ERAS:
    print(f"  {e:<22} {len(eval_ds_by_era[e]):>6} stays")

# ── CLOSED-LOOP DETECTION (DESIGN_RATIONALE §8) ───────────────────────────────
# Decisions are made IN the walk-forward loop, not precomputed against the frozen
# source — an open loop can never be quieted by a successful update.
#   gate    each era scored against REF_WINDOW = the training mixture of the most
#           recent update (replay + that era's adapt-train cohort). Adapting
#           moves the window, so the detector can fall silent.
#           NOT the pooled history: pooling never quiets (§8, §11.14).
#   oracle  measured against the CURRENT trajectory, reporting only.
# One schedule shared by all arms — per-arm schedules would confound the
# comparison with update count (§10). The gate is label-free, so it is unbiased
# between arms.
REF_WINDOW = train_era_t0          # what treat+fusion were last fitted on
ref_window_desc = "source training era"
# The frozen physiology encoder stays fitted to the SOURCE era, so its gate
# reference must not move with the updates. It advances only if an escalated
# update actually unfreezes the stream.
PHYSIO_REF_WINDOW = train_era_t0
physio_ref_desc = "source training era (physio encoder frozen)"

eligible_eras = [e for e in TEST_ERAS
                 if era_detect_df.get(e) is not None
                 and era_detect_df[e]["stay_id"].n_unique() > 50]
print(f"\nEras with a usable detection cohort: {eligible_eras}")
print("Adapt/no-adapt is decided inside the walk-forward loop, against the data")
print("the deployed model has been trained on so far — not precomputed here.")

best_lr, lr_sweep_log = {}, {}
# traj[arm][seed] = running state_dict, carried forward across steps
traj = {a: {s: copy.deepcopy(source_state) for s in ADAPT_SEEDS} for a in ARMS}
# R[step][era][arm] = metrics dict
R = {}
step_probs = {}          # step -> era -> arm -> probs, for paired tests
step_order = []          # EVERY era walked, adapted or not
adapted_eras = []        # subset where an update actually happened
adapt_cohorts = {}
online_log = []          # the closed-loop detection record


def build_arm(arm, state, escalate=False):
    """`escalate` = the physiology gate's safety response: release the freeze
    when the premise justifying it no longer holds. Inert unless
    PHYSIO_DRIFT_ACTION == "unfreeze" AND the gate fired."""
    m = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
    m.load_state_dict(state)
    setup = ARMS[arm]["setup"]
    if escalate and setup == "unfreeze_adaptive":
        setup = "unfreeze_all"
    getattr(m, setup)()
    return m


def era_split(e):
    if PIN is not None:
        _c = PIN["cohorts"][e]
        _t0 = era_detect_df[e].filter(pl.col("hrs_from_admit") == 0)
        _m = dict(zip(_t0["stay_id"].to_list(), _t0["subject_id"].to_list()))
        return tuple(sorted({_m[int(x)] for x in _c[k]}) for k in
                     ("adapt_train_stays", "adapt_val_es_stays", "adapt_val_sel_stays"))
    return _era_split_unpinned(e)


def _era_split_unpinned(e):
    """
    adapt-train / val-ES / val-SEL subjects for era e, CHRONOLOGICALLY ordered.

    Earliest patients train the update, later ones early-stop and select it. A
    random split here would let the update be tuned on patients who arrived
    after the ones it trained on, which is the same temporal violation as a
    random adapt/eval split — just one level down.
    """
    t0 = era_detect_df[e].filter(pl.col("hrs_from_admit") == 0)
    per_subj = (t0.group_by("subject_id")
                  .agg(pl.col("approx_year").min().alias("yr")))
    r = np.random.default_rng(90000 + TEST_ERAS.index(e))
    per_subj = per_subj.with_columns(pl.Series("tb", r.random(per_subj.height))
                                     ).sort(["yr", "tb"])
    subj = per_subj["subject_id"].to_list()
    n = len(subj)
    f_tr = FRAC_ADAPT_TRAIN / FRAC_DETECT
    f_es = FRAC_ADAPT_VAL_ES / FRAC_DETECT
    n_tr, n_es = int(n * f_tr), int(n * f_es)
    return subj[:n_tr], subj[n_tr:n_tr + n_es], subj[n_tr + n_es:]


def replay_buffer(upto_idx):
    if PIN is not None:
        _rs = PIN["cohorts"][TEST_ERAS[upto_idx]].get("replay_stays", [])
        if not _rs:
            return pl.DataFrame()
        _all = pl.concat([era_detect_df[TEST_ERAS[i]] for i in range(upto_idx)])
        return _all.filter(pl.col("stay_id").is_in([int(x) for x in _rs]))
    return _replay_buffer_unpinned(upto_idx)


def _replay_buffer_unpinned(upto_idx):
    """
    Experience replay drawn from the DETECTION cohorts of all earlier eras,
    never from an evaluation cohort. Split evenly so no single era dominates.
    """
    earlier = [TEST_ERAS[i] for i in range(upto_idx)]
    if not earlier:
        return pl.DataFrame()
    per = max(1, BUFFER_SIZE // len(earlier))
    frames = []
    for j, e in enumerate(earlier):
        cand = era_detect_df[e]
        stays = cand.filter(pl.col("hrs_from_admit") == 0)["stay_id"].unique().to_list()
        rr = np.random.default_rng(70000 + j)
        rr.shuffle(stays)
        frames.append(cand.filter(pl.col("stay_id").is_in(stays[:per])))
    frames = [f for f in frames if f.height > 0]
    return pl.concat(frames) if frames else pl.DataFrame()


@torch.no_grad()
def eval_all_eras(state, arm_name):
    """Evaluate one model state on every evaluation era. Returns (metrics, probs)."""
    m = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
    m.load_state_dict(state)
    m.eval()
    mets, prs = {}, {}
    for e in EVAL_ERAS:
        ds = eval_ds_by_era[e]
        met, probs, labels = eval_ds(m, ds, criterion)
        met.update(stratified_metrics(ds, labels, probs, LABEL_COLS, INCIDENT_GATE))
        mets[e], prs[e] = met, probs
    del m
    return mets, prs


# ── Row 0 of the retention matrix: the source model, before any adaptation ─────
print("\n" + "=" * 70 + "\nPHASE 3: Sequential adaptation\n" + "=" * 70)
print("\n--- Step 0: source model, no adaptation (reference row) ---")
_src_mets, _src_probs = eval_all_eras(source_state, "A")
R["source"] = {e: _src_mets[e] for e in EVAL_ERAS}
step_probs["source"] = {e: {"A": _src_probs[e]} for e in EVAL_ERAS}
for e in EVAL_ERAS:
    print(f"  {e:<22} " + "  ".join(
        f"{l.replace('label_','')[:12]}={_src_mets[e].get(f'{l}_auroc', float('nan')):.3f}"
        for l in LABEL_COLS))

# ── Walk forward ──────────────────────────────────────────────────────────────
# Every era is walked. Whether it produces an UPDATE is decided here, from the
# state the deployed system is actually in when that era arrives.
for e in TEST_ERAS:
    t_idx = TEST_ERAS.index(e)
    print("\n" + "=" * 70)
    print(f"STEP {len(step_order) + 1}: era {e}")
    print("=" * 70)

    step_order.append(e)
    cur_t0 = era_detect_t0_raw.get(e, pl.DataFrame())    # RAW units — see above

    # ── DETECT, against what the deployed model has been trained on ──
    if e not in eligible_eras or cur_t0.height < MIN_YEAR_N or REF_WINDOW.height < MIN_YEAR_N:
        gate = {"dist_ratio": np.nan, "physio_ratio": np.nan, "n_drifted": 0,
                "n_physio_drifted": 0}
        dist_fire = physio_fire = False
        reason = ("detection cohort too small to score" if cur_t0.height < MIN_YEAR_N
                  else "reference window too small to score" if REF_WINDOW.height < MIN_YEAR_N
                  else "era not eligible")
    else:
        # "fixed": REF_WINDOW is a pooled training mixture, not the preceding era.
        gate = score_pair(REF_WINDOW, cur_t0, mode="fixed",
                          physio_ref=PHYSIO_REF_WINDOW)
        dist_fire = (gate["dist_ratio"] > 1.0 and gate["n_drifted"] >= MIN_FEAT_DRIFTED)
        physio_fire = (gate["physio_ratio"] > 1.0
                       and gate["n_physio_drifted"] >= MIN_PHYSIO_FEAT_DRIFTED)
        reason = ""

    # "unfreeze" needs an update to attach to, so it necessarily lets the physio
    # gate trigger one. Under "flag" the gate cannot alter the schedule.
    physio_may_gate = ("physiology" in GATE_BRANCHES) or PHYSIO_DRIFT_ACTION == "unfreeze"
    do_adapt = (("distribution" in GATE_BRANCHES and dist_fire)
                or (physio_may_gate and physio_fire))
    # CLEAN-V2: FORCE_ADAPT=1 updates at every era regardless of the gate, so every
    # arm is compared over the same three adaptation steps. The gate is still scored
    # and logged (gate_would_adapt) and is reported as its own result.
    gate_would_adapt = do_adapt
    if FORCE_ADAPT:
        do_adapt = True
    escalate = bool(physio_fire and PHYSIO_DRIFT_ACTION == "unfreeze")

    print(f"  detector vs [{ref_window_desc}]  "
          f"(n_ref={gate.get('n_ref', REF_WINDOW.height)}, n_cur={cur_t0.height})")
    print(f"    treatment  x{gate['dist_ratio']:>5.2f} of the worst train-era year shift "
          f"({gate['n_drifted']} features)  {'FIRE' if dist_fire else 'quiet'}")
    print(f"    physiology x{gate['physio_ratio']:>5.2f} "
          f"({gate['n_physio_drifted']} features)  {'FIRE' if physio_fire else 'quiet'}")
    if reason:
        print(f"    ⚠ {reason}")
    if physio_fire:
        print("    ⚠ FREEZE PREMISE VIOLATED — physiology has moved beyond the")
        print(f"      variation the model was trained through. Action: {PHYSIO_DRIFT_ACTION}"
              + ("  → Run B adapts with physiology UNFROZEN this step." if escalate else
                 "  → logged only; Run B keeps its freeze."))

    # ── Oracle, measured against the CURRENT deployed model, never gating ──
    # Median seed only, and only for the proposed arm. The oracle is a reporting
    # diagnostic — it cannot change the schedule — so ensembling it over every
    # seed and every arm bought nothing and cost |ARMS| x |SEEDS| forward passes
    # per era. ORACLE_ARMS widens it if a reviewer wants the full picture.
    ORACLE_ARMS = ["B"]
    oracle = {}
    if era_detect_df.get(e) is not None and era_detect_df[e]["stay_id"].n_unique() >= 30:
        for _arm in [a for a in ORACLE_ARMS if a in ARMS]:
            _st = traj[_arm].get("_median_seed", ADAPT_SEEDS[0])
            _md, _pl_, _ns = ensemble_perf_drop(
                era_detect_df[e], key=f"online::{e}::{_arm}::{len(adapted_eras)}",
                states=[traj[_arm][_st]])
            oracle[_arm] = {"mean_auroc_drop": _md, "per_lbl": _pl_, "n_sig": _ns}
        print("    oracle ΔAUROC (current deployed weights): "
              + "  ".join(f"{a}={oracle[a]['mean_auroc_drop']:+.4f}"
                          f"({oracle[a]['n_sig']}/{len(LABEL_COLS)})" for a in oracle))
        _missed = (not do_adapt) and any(o["n_sig"] >= MIN_PERF_DROPS for o in oracle.values())
        if _missed:
            print("    ⚠ MISSED UPDATE: the oracle sees degradation the label-free gate")
            print("      did not. This is the cost of having no outcome labels at")
            print("      prediction time — report it, do not gate on the oracle to hide it.")
    else:
        _missed = False

    online_log.append({
        "era": e, "n_detect": int(cur_t0.height),
        "ref_window": ref_window_desc, "n_ref": int(REF_WINDOW.height),
        "dist_ratio": gate["dist_ratio"], "n_drifted": gate["n_drifted"],
        "dist_fired": bool(dist_fire),
        "physio_ratio": gate["physio_ratio"], "n_physio_drifted": gate["n_physio_drifted"],
        "physio_ref": physio_ref_desc,
        "physio_fired": bool(physio_fire), "escalated": escalate,
        "adapted": bool(do_adapt), "gate_would_adapt": bool(gate_would_adapt),
        "forced": bool(FORCE_ADAPT and not gate_would_adapt),
        "oracle_missed_update": bool(_missed),
        "oracle": {a: {"mean_auroc_drop": o["mean_auroc_drop"], "n_sig": o["n_sig"]}
                   for a, o in oracle.items()},
        "skip_reason": reason or None,
    })

    if not do_adapt:
        # A quiet era is a real no-update step, not a gap: re-evaluate unchanged
        # so the retention matrix has a row and the figure shows it holding.
        print("  → NO UPDATE. Re-evaluating the unchanged models for this row.")
        adapt_cohorts[e] = {"adapt_train_stays": [], "adapt_val_es_stays": [],
                            "adapt_val_sel_stays": [], "replay_stays": [],
                            "eval_stays": era_eval_ds[e].stay_ids if e in era_eval_ds else [],
                            "adapted": False}
        R[e], step_probs[e] = {}, {}
        for era in EVAL_ERAS:
            R[e][era], step_probs[e][era] = {}, {}
        for arm in ARMS:
            mets, prs = eval_all_eras(traj[arm][traj[arm].get("_median_seed", ADAPT_SEEDS[0])], arm)
            for era in EVAL_ERAS:
                R[e][era][arm] = mets[era]
                step_probs[e][era][arm] = prs[era]
        for era in EVAL_ERAS:
            R[e][era]["A"] = R["source"][era]
            step_probs[e][era]["A"] = step_probs["source"][era]["A"]
        continue

    adapted_eras.append(e)
    print(f"  → UPDATE {len(adapted_eras)} triggered.")

    tr_subj, es_subj, sel_subj = era_split(e)
    cur = era_detect_df[e]
    tr_df = cur.filter(pl.col("subject_id").is_in(tr_subj))
    es_df = cur.filter(pl.col("subject_id").is_in(es_subj))
    sel_df = cur.filter(pl.col("subject_id").is_in(sel_subj))
    buf_df = replay_buffer(t_idx)
    comb = pl.concat([buf_df, tr_df]) if buf_df.height > 0 else tr_df

    tr_ds = ICUDataset(comb, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    es_ds = ICUDataset(es_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    sel_ds = ICUDataset(sel_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    adapt_cohorts[e] = {
        "adapt_train_stays": tr_df["stay_id"].unique().to_list(),
        "adapt_val_es_stays": es_df["stay_id"].unique().to_list(),
        "adapt_val_sel_stays": sel_df["stay_id"].unique().to_list(),
        "replay_stays": buf_df["stay_id"].unique().to_list() if buf_df.height else [],
        "eval_stays": era_eval_ds[e].stay_ids if e in era_eval_ds else [],
        "adapted": True, "escalated": escalate,
    }
    print(f"  train={len(tr_ds)} (incl. {buf_df['stay_id'].n_unique() if buf_df.height else 0} replay "
          f"from {t_idx} earlier era(s)) | val-ES={len(es_ds)} | val-SEL={len(sel_ds)}")

    # ONE objective, shared by every arm at every step.
    crit_e = make_adapt_criterion(tr_ds, max_weight=20.0, gamma=1.5, label_smoothing=0.02)
    es_loader = DataLoader(es_ds, batch_size=EVAL_BATCH, shuffle=False)

    if not best_lr or SWEEP_EVERY_STEP:
        print(f"\n  LR sweep on val-SEL: {LR_ADAPT_GRID} (each arm at its own best)")
        # Retention is recorded at EVERY learning rate, not just the selected one.
        #
        # The selection rule maximises AUROC on the NEW era. For a full
        # fine-tune a larger LR moves all ~95k weights further, fits the new era
        # better, and therefore wins the sweep — but moving further is also what
        # destroys old-era performance. The rule is identical for every arm, yet
        # its consequences are not: Run B's frozen backbone means its LR choice
        # cannot cost it much retention. So "Run C was handed the LR that forgets
        # most" is a fair objection to the headline number.
        #
        # Changing the rule to select on retention would be worse — that tunes
        # the protocol on the very outcome being reported. Instead the whole
        # grid is logged, so the claim can be stated without any selection rule
        # doing hidden work: if C forgets more than B at EVERY LR in the grid,
        # the result is not an artefact of which LR was picked.
        for arm in ARMS:
            sc, rt = {}, {}
            for lr in LR_ADAPT_GRID:
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                m = build_arm(arm, traj[arm][ADAPT_SEEDS[0]], escalate=escalate)
                m = train_model(m, DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True),
                                es_loader, crit_e, lr, ADAPT_EPOCHS, ADAPT_PATIENCE,
                                l2sp_ref=source_state if ARMS[arm]["l2sp"] > 0 else None,
                                l2sp_lambda=ARMS[arm]["l2sp"], verbose=False)
                sc[lr] = sel_auroc(m, sel_ds)
                rt[lr] = sel_auroc(m, val_calib_ds)
                del m
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            best_lr[arm] = max(sc, key=sc.get)
            lr_sweep_log.setdefault(e, {})[arm] = {
                "new_era_auroc": {f"{k:g}": v for k, v in sc.items()},
                "old_era_auroc": {f"{k:g}": v for k, v in rt.items()},
                "selected_lr": float(best_lr[arm]),
            }
            print(f"    {arm:<10} new-era " + " ".join(f"{k:g}:{v:.4f}" for k, v in sc.items())
                  + f"  → {best_lr[arm]:g}")
            print(f"    {'':<10} old-era " + " ".join(f"{k:g}:{v:.4f}" for k, v in rt.items()))

        # Does the selected LR cost any arm its retention? If C's retention is
        # much better at an LR the rule did not pick, that gap belongs in the
        # paper as a sensitivity note rather than being silently discarded.
        _src_ref = (per_label_mean_auroc(R["source"][SOURCE_ERA_KEY])
                    if "source" in R and SOURCE_ERA_KEY in R["source"] else float("nan"))
        print(f"\n    Retention sensitivity to the LR rule (old-era AUROC; "
              f"source model = {_src_ref:.4f}):")
        for arm in ARMS:
            _rt = lr_sweep_log[e][arm]["old_era_auroc"]
            _sel = _rt[f"{best_lr[arm]:g}"]
            _bestk = max(_rt, key=_rt.get)
            _flag = ("" if _bestk == f"{best_lr[arm]:g}"
                     else f"   ⚠ retention would be {_rt[_bestk] - _sel:+.4f} better at lr={_bestk}")
            print(f"      {arm:<10} selected lr={best_lr[arm]:g} → {_sel:.4f}{_flag}")
        print("      Quote the whole grid, not just the selected column. The claim to")
        print("      defend is 'C retains less than B at every LR', which needs no rule.")

    for arm in ARMS:
        print(f"\n  {arm}: {ARMS[arm]['desc']} @ lr={best_lr[arm]:g}")
        for s in ADAPT_SEEDS:
            torch.manual_seed(s)
            np.random.seed(s)
            # Continue from THIS seed's running state, not from source — that is
            # what makes forgetting compound across steps.
            m = build_arm(arm, traj[arm][s], escalate=escalate)
            m = train_model(m, DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True),
                            es_loader, crit_e, best_lr[arm], ADAPT_EPOCHS, ADAPT_PATIENCE,
                            l2sp_ref=source_state if ARMS[arm]["l2sp"] > 0 else None,
                            l2sp_lambda=ARMS[arm]["l2sp"], verbose=False)
            traj[arm][s] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Canonical seed for this step = median by selection-split score.
        sel_scores = []
        for s in ADAPT_SEEDS:
            mm = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
            mm.load_state_dict(traj[arm][s])
            sel_scores.append(sel_auroc(mm, sel_ds))
            del mm
        med = int(np.argsort(np.array(sel_scores))[len(sel_scores) // 2])
        traj[arm]["_median_seed"] = ADAPT_SEEDS[med]
        print(f"    seeds {ADAPT_SEEDS} → {np.round(sel_scores, 4).tolist()} "
              f"| canonical = seed {ADAPT_SEEDS[med]}")

    # ── CLOSE THE LOOP ──
    # Reference := the exact mixture this update trained on (replay + adapt-train),
    # from the RAW frame so units match. Never the evaluation cohort — the
    # detector must not see the patients the update is later scored on.
    _ref_stays = (adapt_cohorts[e]["adapt_train_stays"] + adapt_cohorts[e]["replay_stays"])
    REF_WINDOW = raw_all_t0.filter(pl.col("stay_id").is_in(_ref_stays))
    ref_window_desc = f"update-{len(adapted_eras)} training mixture ({e} + replay)"
    if escalate:
        # physiology was actually unfrozen this step, so its reference moves too
        PHYSIO_REF_WINDOW = REF_WINDOW
        physio_ref_desc = f"update-{len(adapted_eras)} (physio unfrozen)"
    print(f"  reference window → {ref_window_desc}, n={REF_WINDOW.height} stays")
    if REF_WINDOW.height < MIN_YEAR_N:
        print("  ⚠ reference window is too small to score the next era reliably; "
              "the detector will report NaN and default to no update.")

    # Evaluate every arm on EVERY era seen so far — the retention row.
    R[e], step_probs[e] = {}, {}
    for era in EVAL_ERAS:
        R[e][era], step_probs[e][era] = {}, {}
    for arm in ARMS:
        mets, prs = eval_all_eras(traj[arm][traj[arm]["_median_seed"]], arm)
        for era in EVAL_ERAS:
            R[e][era][arm] = mets[era]
            step_probs[e][era][arm] = prs[era]
    # Run A never adapts, so its row is the source row repeated.
    for era in EVAL_ERAS:
        R[e][era]["A"] = R["source"][era]
        step_probs[e][era]["A"] = step_probs["source"][era]["A"]

    print(f"\n  --- After adapting at {e}: AUROC on every era ---")
    for lbl in LABEL_COLS:
        print(f"  {lbl.replace('label_','')}:")
        for arm in ["A"] + list(ARMS):
            cells = "  ".join(
                f"{era[:9]}={R[e][era][arm].get(f'{lbl}_auroc', float('nan')):.3f}"
                for era in EVAL_ERAS)
            print(f"    {arm:<10} {cells}")

adaptation_performed = len(adapted_eras) > 0
ADAPT_ERAS = list(adapted_eras)        # name kept for the reporting code below

# The "current era" for every downstream summary. step_order now contains eras
# that produced no update, and may contain an era with no held-out evaluation
# cohort, so step_order[-1] is not automatically a valid key into eval_ds_by_era.
# Take the last walked era that has one.
FINAL_ERA = next((e for e in reversed(step_order) if e in eval_ds_by_era), None)
if FINAL_ERA is not None and FINAL_ERA != (step_order[-1] if step_order else None):
    print(f"\n⚠ Last walked era {step_order[-1]} has no evaluation cohort; "
          f"reporting the current era as {FINAL_ERA}.")
step_order = [e for e in step_order if e in R]

print("\n" + "=" * 70)
print("CLOSED-LOOP DETECTION SUMMARY")
print("=" * 70)
print(f"  {'era':<14}{'n':>7}{'treat x':>9}{'physio x':>10}  {'action':<12}{'oracle'}")
for r in online_log:
    _ora = ("missed" if r["oracle_missed_update"] else
            "agrees" if r["adapted"] else "quiet")
    print(f"  {r['era']:<14}{r['n_detect']:>7}{r['dist_ratio']:>9.2f}"
          f"{r['physio_ratio']:>10.2f}  "
          f"{('UPDATE' + (' (esc)' if r['escalated'] else '')) if r['adapted'] else 'no update':<12}"
          f"{_ora}")
print(f"\n  updates: {len(adapted_eras)}/{len(step_order)} eras  → {adapted_eras}")
_gaps = [TEST_ERAS.index(b) - TEST_ERAS.index(a)
         for a, b in zip(adapted_eras, adapted_eras[1:])]
print(f"  cadence: {len(adapted_eras)/max(len(step_order),1):.2f} updates/era"
      + (f", mean gap {np.mean(_gaps):.1f} eras" if _gaps else ""))
_missed_n = sum(1 for r in online_log if r["oracle_missed_update"])
print(f"  missed updates (oracle saw degradation, label-free gate did not): {_missed_n}")
print("  Ratios are multiples of the largest year-to-year shift inside the training")
print("  era, measured against the data the deployed model was trained on. A ratio")
print("  that FALLS after an update is the update working; one that stays high is")
print("  drift the update did not absorb. Both belong in the paper.")
if not adaptation_performed:
    print("\n  ⚠ NO UPDATES FIRED. Every downstream comparison is between models that")
    print("    never adapted, so the adaptation claims cannot be made from this run.")

# Flatten R into the {step: {era: metrics}} shape the summary helpers expect,
# keeping the per-arm nesting available separately.
R_arm = R
R = {t: {era: R_arm[t][era] for era in R_arm[t]} for t in R_arm}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — RETENTION MATRIX, ACC, BWT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\nPHASE 4: Retention matrix and forgetting\n" + "=" * 70)

summary = {}
if adaptation_performed:
    for lbl in LABEL_COLS:
        print(f"\n### {lbl.replace('label_','')} — AUROC by era after each adaptation step")
        for arm in ["A"] + list(ARMS):
            Rm = {"source": {e: R_arm["source"][e] for e in EVAL_ERAS}} if arm == "A" else {}
            Rm = {t: {e: R_arm[t][e][arm] if t != "source" else R_arm["source"][e]
                      for e in EVAL_ERAS} for t in ["source"] + step_order}
            acc = average_accuracy(Rm, step_order, EVAL_ERAS, lbl)
            bwt = backward_transfer(Rm, step_order, EVAL_ERAS, lbl)
            summary.setdefault(lbl, {})[arm] = {"ACC": acc, "BWT": bwt}
            print(f"\n  Run {arm}   ACC={acc:.4f}   BWT={bwt:+.4f}"
                  + ("   [never adapted — reference line]" if arm == "A" else ""))
            print(format_retention_matrix(Rm, step_order, EVAL_ERAS, lbl,
                                          baseline_step="source"))

    print("\n  ACC = mean AUROC across all eras after the final step.")
    print("  BWT = mean change on EARLIER eras between when they were current and now.")
    print("        Negative = forgetting. This is the two-stream claim: B should have")
    print("        BWT closer to zero than C, at comparable ACC.")

    print("\n--- Headline: ACC vs BWT (per label, per arm) ---")
    for lbl in LABEL_COLS:
        print(f"\n  {lbl.replace('label_','')}")
        print(f"    {'arm':<12}{'ACC':>10}{'BWT':>10}")
        for arm in ["A"] + list(ARMS):
            r = summary[lbl][arm]
            print(f"    {arm:<12}{r['ACC']:>10.4f}{r['BWT']:>+10.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — PAIRED TESTS
# ══════════════════════════════════════════════════════════════════════════════
# B_mirror vs C is the decisive contrast once B_mirror turns out to be the best
# arm: it separates "freeze the treatment stream specifically" from "adapt as
# much as possible". B_mirror updates 72k parameters against C's 95k, so if it
# matches or beats C the effect is structural rather than capacity — but that
# only counts with a confidence interval behind it, and the original list never
# compared the two.
CONTRASTS = [("B", "C"), ("B", "C_l2sp"), ("B", "B_mirror"), ("B", "A"),
             ("C", "A"), ("C_l2sp", "C"),
             ("B_mirror", "C"), ("B_mirror", "C_l2sp"), ("B_mirror", "A")]
paired_out = {}

if adaptation_performed:
    print("\n" + "=" * 70 + "\nPHASE 5: Paired bootstrap comparisons\n" + "=" * 70)
    print("Run A never sees adaptation data, so B > A only shows in-domain training")
    print("helps. The defended claims are B vs C, B vs C_l2sp, and B vs B_mirror.\n")

    final = FINAL_ERA
    probs_final = step_probs[final]
    labels_by_era = {e: eval_ds_by_era[e].labels for e in EVAL_ERAS}

    # (a) CURRENT-ERA performance at the final step: did adaptation help?
    # (b) EARLIER-ERA performance at the final step: did it forget?
    for era in EVAL_ERAS:
        role = ("current era" if era == final
                else "source era" if era == SOURCE_ERA_KEY else "earlier era (retention)")
        print(f"--- Era {era}  [{role}] ---")
        ds_e, lab_e = eval_ds_by_era[era], labels_by_era[era]
        paired_out.setdefault(era, {})
        # Loop order is (label, cohort) OUTSIDE and contrast INSIDE, so the
        # bootstrap replicates are computed once per arm and shared by every
        # contrast at that cell. With 9 contrasts over 5 arms that is 5 arm
        # evaluations instead of 18 — bit-identical results, ~4x faster.
        for i, lbl in enumerate(LABEL_COLS):
            y_t = lab_e[:, i]
            if not (0 < y_t.sum() < len(y_t)):
                continue
            cohorts = [("overall", None)]
            if HAS_META and INCIDENT_GATE[lbl] in ds_e.meta:
                g = ds_e.meta[INCIDENT_GATE[lbl]]
                cohorts += [("incident", g <= 0), ("prevalent", g > 0)]
            for cohort, mask in cohorts:
                yy = y_t if mask is None else y_t[mask]
                if not (0 < yy.sum() < len(yy)):
                    continue
                arm_p = {a: (probs_final[era][a][:, i] if mask is None
                             else probs_final[era][a][mask, i]) for a in probs_final[era]}
                cache = bootstrap_cache(yy, arm_p, roc_auc_score,
                                        n_boot=N_BOOT, seed=11)
                for x, y in CONTRASTS:
                    if x not in cache or y not in cache:
                        continue
                    d, lo, hi, pg = paired_bca_cached(cache, x, y)
                    paired_out[era].setdefault(f"{x}_minus_{y}", {}).setdefault(lbl, {})[cohort] = {
                        "dAUROC": d, "lo": lo, "hi": hi, "p_x_gt_y": pg,
                        "n": int(len(yy)), "n_pos": int(yy.sum())}
                    if cohort in ("overall", "incident"):
                        verdict = x if lo > 0 else (y if hi < 0 else "tie")
                        print(f"    {lbl.replace('label_',''):<16} {x:>8} − {y:<8} {cohort:<10} "
                              f"Δ={d:+.4f} [{lo:+.4f}, {hi:+.4f}] → {verdict}")
        print()

    with open(SAVE_PATH / "paired_contrasts.json", "w") as f:
        json.dump(to_serialisable(paired_out), f, indent=2)

# ── Compatibility surface for the rest of the script and for script3 ───────────
# The single-changepoint view is now DERIVED from the sequential run rather than
# being a separate experiment: "post" is the final adapted era, "pre" is every
# earlier era, which is exactly the retention measurement.
arm_post, arm_probs, arm_ret_pre, arm_ret_val = {}, {}, {}, {}
eval_post_stays, labels_ref, eval_post_ds = [], None, None
adapt_train_stays = adapt_val_stays = buf_pre_stays = []
adapt_val_es_df = adapt_val_sel_df = pl.DataFrame()

if adaptation_performed:
    final = FINAL_ERA
    eval_post_ds = eval_ds_by_era[final]
    eval_post_stays = eval_post_ds.stay_ids
    labels_ref = eval_post_ds.labels
    _prev_eras = [e for e in EVAL_ERAS if e != final and e != SOURCE_ERA_KEY]

    for arm in ["A"] + list(ARMS):
        arm_post[arm] = R_arm[final][final][arm]
        arm_probs[arm] = step_probs[final][final][arm]
        arm_ret_val[arm] = retention_table(R_arm["source"][SOURCE_ERA_KEY],
                                           R_arm[final][SOURCE_ERA_KEY][arm], LABEL_COLS)
        if _prev_eras:
            _b = {k: float(np.nanmean([R_arm["source"][e].get(k, np.nan) for e in _prev_eras]))
                  for k in [f"{l}_auroc" for l in LABEL_COLS]}
            _a = {k: float(np.nanmean([R_arm[final][e][arm].get(k, np.nan) for e in _prev_eras]))
                  for k in [f"{l}_auroc" for l in LABEL_COLS]}
            arm_ret_pre[arm] = retention_table(_b, _a, LABEL_COLS)
        else:
            arm_ret_pre[arm] = {}

    _c = adapt_cohorts.get(final, {})
    if not _c.get("adapt_train_stays"):
        # CLEAN-COHORT PILOT PATCH: a quiet final era is a no-update step, so it has no
        # adaptation cohort; fall back to the last era that was actually adapted.
        _adapted = [e_ for e_, c_ in adapt_cohorts.items() if c_.get("adapt_train_stays")]
        _c = adapt_cohorts[_adapted[-1]] if _adapted else {}
    adapt_train_stays = _c.get("adapt_train_stays", [])
    adapt_val_stays = _c.get("adapt_val_es_stays", []) + _c.get("adapt_val_sel_stays", [])
    buf_pre_stays = _c.get("replay_stays", [])

    if HAS_META:
        print("--- Held-out cohort composition, final era ---")
        pa = prevalence_audit(eval_post_ds, labels_ref, LABEL_COLS, INCIDENT_GATE)
        for lbl, r in pa.items():
            print(f"  {lbl.replace('label_',''):<16} incident n={r['n_incident']:>5} | "
                  f"prevalent n={r['n_prevalent']:>5} | "
                  f"{100*r['frac_pos_prevalent']:.0f}% of positives already on treatment at h5")
        with open(SAVE_PATH / "cohort_composition.json", "w") as f:
            json.dump(to_serialisable(pa), f, indent=2)

    for arm in ["A"] + list(ARMS):
        print(summarize_arm(f"Run {arm}" + ("  [reference line]" if arm == "A" else ""),
                            arm_post[arm], arm_ret_pre[arm], arm_ret_val[arm], LABEL_COLS))

    results = {SOURCE_ERA_KEY: {a: R_arm[final][SOURCE_ERA_KEY].get(a, {}) for a in ["A", "B", "C"]}}
    for e in _prev_eras:
        results[f"Test-Pre ({e})"] = {a: R_arm[final][e][a] for a in ["A", "B", "C"]}
    results[f"Test-Post ({final})"] = {a: arm_post[a] for a in ["A", "B", "C"]}
    _safe("explain_metric_gains", explainer.explain_metric_gains, results, LABEL_COLS)

    model_B, model_C = (TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device),
                        TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device))
    model_B.load_state_dict(traj["B"][traj["B"]["_median_seed"]])
    model_C.load_state_dict(traj["C"][traj["C"]["_median_seed"]])
    adapted = {a: {"model": None} for a in ARMS}
    for a in ARMS:
        _m = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
        _m.load_state_dict(traj[a][traj[a]["_median_seed"]])
        adapted[a]["model"] = _m
    adapt_train_df = test_df.filter(pl.col("stay_id").is_in(adapt_train_stays))
    _safe("adaptation_update_RunB", explainer.explain_adaptation_update,
          model_before=source_model, model_after=model_B,
          adapt_train_ds=ICUDataset(adapt_train_df, SEQ_FEATURES, TREATMENT_FEATURES,
                                    LABEL_COLS, SEQ_LEN),
          device=device, run_tag="RunB")

with open(SAVE_PATH / "sequential_results.json", "w") as f:
    json.dump(to_serialisable({
        "eval_eras": EVAL_ERAS,
        # `adapt_steps` is EVERY era walked, so downstream scripts replay the
        # whole trajectory including no-update steps; `adapted_eras` is the
        # subset where weights actually changed. A consumer that trains on
        # adapt_steps without checking `cohorts[e]["adapted"]` will try to train
        # on an empty cohort.
        "adapt_steps": step_order,
        "adapted_eras": adapted_eras,
        "quiet_boundaries": [e for e in step_order if e not in adapted_eras],
        "closed_loop_log": online_log,
        "gate_branches": list(GATE_BRANCHES),
        "physio_drift_action": PHYSIO_DRIFT_ACTION,
        "stage2_thresholds": {"treatment": treat_ratio_thr, "physiology": physio_ratio_thr},
        "stage2_info": {"treatment": treat_ratio_info, "physiology": physio_ratio_info},
        "acc_bwt": summary,
        "retention_matrix": {t: {e: {a: R_arm[t][e] if t == "source" else R_arm[t][e][a]
                                     for a in (["A"] + list(ARMS))}
                                 for e in EVAL_ERAS} for t in ["source"] + step_order},
        "lr_sweep": lr_sweep_log, "best_lr": {k: float(v) for k, v in best_lr.items()},
        # `adapted` / `escalated` are bools, not stay lists — guard the cast.
        "cohorts": {e: {k: ([int(x) for x in v] if isinstance(v, list) else v)
                        for k, v in c.items()}
                    for e, c in adapt_cohorts.items()},
    }), f, indent=2)
# ══════════════════════════════════════════════════════════════════════════════
# PLACEBO CONTROL — adapt where the detector did NOT fire
# ══════════════════════════════════════════════════════════════════════════════
# The sequential-native placebo: run the identical adaptation procedure at a
# boundary the detector called quiet. If Run B's gains are real responses to
# drift, the placebo gain should be ~0. If the placebo gains as much as the real
# adaptation, then "adaptation helps" is just "more training data helps" and the
# drift-detection framing adds nothing.
_quiet = [e for e in step_order if e not in adapted_eras and e in era_detect_df
          and era_detect_df[e]["stay_id"].n_unique() > 50]
_placebo_src = None
if adaptation_performed and _quiet:
    _placebo_src = era_detect_df[_quiet[0]]
    _placebo_tag = f"quiet boundary {_quiet[0]}"
elif adaptation_performed and test_pre.height > 0:
    _placebo_src = test_pre
    _placebo_tag = "pre-changepoint data (no quiet boundary available)"

if _placebo_src is not None:
    print(f"\n--- Placebo adaptation control ({_placebo_tag}, matched size) ---")
    _rng_p = np.random.default_rng(1234)
    _pre_subj = _placebo_src.filter(pl.col("hrs_from_admit") == 0)["subject_id"].unique().to_list()
    _rng_p.shuffle(_pre_subj)
    _n = len(_pre_subj)
    _tr_s = _pre_subj[:int(_n * 0.30)]
    _va_s = _pre_subj[int(_n * 0.30):int(_n * 0.40)]
    _ev_s = _pre_subj[int(_n * 0.40):]
    _tr_df = _placebo_src.filter(pl.col("subject_id").is_in(_tr_s))
    _va_df = _placebo_src.filter(pl.col("subject_id").is_in(_va_s))
    _ev_df = _placebo_src.filter(pl.col("subject_id").is_in(_ev_s))
    _target_n = max(1, len(adapt_train_stays))
    _tr_df = _tr_df.filter(pl.col("stay_id").is_in(_tr_df["stay_id"].unique().to_list()[:_target_n]))
    if _tr_df.height and _va_df.height and _ev_df.height:
        _tr_ds = ICUDataset(_tr_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
        _va_ds = ICUDataset(_va_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
        _ev_ds = ICUDataset(_ev_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
        _crit = make_adapt_criterion(_tr_ds, max_weight=20.0, gamma=1.5, label_smoothing=0.02)
        _placebo = []
        for s in ADAPT_SEEDS:
            torch.manual_seed(s)
            np.random.seed(s)
            _m = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
            _m.load_state_dict(source_state)
            _m.unfreeze_adaptive()
            _m = train_model(_m, DataLoader(_tr_ds, batch_size=BATCH_SIZE, shuffle=True),
                             DataLoader(_va_ds, batch_size=BATCH_SIZE, shuffle=False),
                             _crit, best_lr["B"], ADAPT_EPOCHS, ADAPT_PATIENCE,
                             f"PLACEBO-s{s} ", verbose=False)
            _placebo.append(eval_ds(_m, _ev_ds, criterion)[0])
            del _m
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _base = eval_ds(model_A, _ev_ds, criterion)[0]
        _placebo_json = {}
        for lbl in LABEL_COLS:
            _a = _base.get(f"{lbl}_auroc", np.nan)
            _b = float(np.nanmean([m.get(f"{lbl}_auroc", np.nan) for m in _placebo]))
            _rd = (arm_post["B"].get(f"{lbl}_auroc", np.nan)
                   - arm_post["A"].get(f"{lbl}_auroc", np.nan)) if arm_post else np.nan
            _placebo_json[lbl] = {"runA_baseline": _a, "placebo_B": _b,
                                  "placebo_delta": _b - _a, "real_delta_final_era": _rd,
                                  "source": _placebo_tag}
            print(f"  {lbl.replace('label_',''):<16} placebo Δ={_b - _a:+.4f}  "
                  f"real Δ at final era={_rd:+.4f}"
                  + ("   ⚠ placebo ≈ real: the gain may be data volume, not drift response"
                     if (_b - _a) == (_b - _a) and _rd == _rd and (_b - _a) > 0.6 * _rd else ""))
        with open(SAVE_PATH / "placebo_adaptation.json", "w") as f:
            json.dump(to_serialisable(_placebo_json), f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots...")

if pair_results:
    # Two panels: the era-pair reporting scan, and the closed-loop trace that
    # actually drove the schedule. Plotted as RATIOS, not raw scores — raw PSI is
    # not comparable across sample sizes, and the y=1 line is then a real
    # decision boundary ("larger than any year-to-year shift the model was
    # trained through") rather than a number whose scale nobody can interpret.
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(max(11, len(pair_results) * 1.8), 4.5))

    pair_labels = [f"{a[-7:]}\n→{b[-7:]}" for a, b in pair_results.keys()]
    pair_ratio = [v["dist_ratio"] for v in pair_results.values()]
    pair_phys = [v["physio_ratio"] for v in pair_results.values()]
    pair_nd = [v["n_drifted"] for v in pair_results.values()]
    is_train = [(b in TRAIN_GROUP_SET) for _, b in pair_results.keys()]
    xs = np.arange(len(pair_labels))
    bars = ax0.bar(xs - 0.2, pair_ratio, width=0.4, edgecolor="black", alpha=0.85,
                   color=["#1f77b4" if t else "#d62728" for t in is_train],
                   label="treatment")
    ax0.bar(xs + 0.2, pair_phys, width=0.4, edgecolor="black", alpha=0.55,
            color="#2ca02c", label="physiology")
    for bar, nd in zip(bars, pair_nd):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                 f"n={nd}", ha="center", va="bottom", fontsize=7)
    ax0.axhline(1.0, color="red", ls="--", lw=1.5,
                label="worst train-era year-to-year shift")
    ax0.set_xticks(xs)
    ax0.set_xticklabels(pair_labels, fontsize=8)
    ax0.set_ylabel("drift ratio (stage-2 calibrated)")
    ax0.set_title("Era-pair scan (reporting)")
    ax0.legend(fontsize=7)

    if online_log:
        ox = np.arange(len(online_log))
        ax1.plot(ox, [r["dist_ratio"] for r in online_log], "o-", color="#d62728",
                 label="treatment vs training window")
        ax1.plot(ox, [r["physio_ratio"] for r in online_log], "s-", color="#2ca02c",
                 label="physiology vs training window")
        ax1.axhline(1.0, color="black", ls="--", lw=1.2, label="fire threshold")
        for i, r in enumerate(online_log):
            if r["adapted"]:
                ax1.axvline(i, color="#1f77b4", lw=6, alpha=0.18)
        ax1.set_xticks(ox)
        ax1.set_xticklabels([r["era"][-7:] for r in online_log], fontsize=8)
        ax1.set_ylabel("drift ratio")
        ax1.set_title("Closed loop (shaded = update fired)")
        ax1.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(SAVE_PATH / "drift_changepoint.png", dpi=150, bbox_inches="tight")
    plt.close()

CMAP = {"A": "#d62728", "B": "#2ca02c", "B_mirror": "#8c564b",
        "C": "#1f77b4", "C_l2sp": "#9467bd"}

# ── Retention trajectory: the headline sequential figure ──────────────────────
# One panel per evaluation era. X = adaptation step, Y = AUROC on THAT era. An
# era's curve after its own step is pure forgetting, and forgetting compounds
# left to right — which is the whole reason the sequential design is a stronger
# test than a single adaptation event.
if adaptation_performed and step_order:
    for lbl in LABEL_COLS:
        n = len(EVAL_ERAS)
        fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 4.2), sharey=True)
        axes = np.atleast_1d(axes)
        xs = list(range(len(step_order) + 1))
        xlab = ["source"] + [str(s)[-7:] for s in step_order]
        for j, era in enumerate(EVAL_ERAS):
            ax = axes[j]
            for arm in ["A"] + list(ARMS):
                ys = [R_arm["source"][era].get(f"{lbl}_auroc", np.nan)]
                ys += [R_arm[t][era][arm].get(f"{lbl}_auroc", np.nan) for t in step_order]
                ax.plot(xs, ys, marker="o", ms=5, lw=1.8, color=CMAP[arm],
                        label=arm, ls="--" if arm == "A" else "-")
            if era in step_order:
                ax.axvline(step_order.index(era) + 1, color="black", ls=":", lw=1.2)
                ax.text(step_order.index(era) + 1, 0.995, " adapted here",
                        fontsize=7, va="top", transform=ax.get_xaxis_transform())
            ax.set_xticks(xs)
            ax.set_xticklabels(xlab, fontsize=7, rotation=20)
            ax.set_title(str(era), fontsize=10)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(f"AUROC — {lbl.replace('label_','')}")
                ax.legend(fontsize=7)
        fig.suptitle(f"Retention trajectory — {lbl.replace('label_','')}. "
                     f"Flat lines to the right of the dotted marker = no forgetting.", y=1.02)
        plt.tight_layout()
        plt.savefig(SAVE_PATH / f"retention_trajectory_{lbl.replace('label_','')}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

# ── ACC vs BWT: one point per arm, the summary of the whole trajectory ─────────
if adaptation_performed and summary:
    fig, axes = plt.subplots(1, len(LABEL_COLS), figsize=(5.0 * len(LABEL_COLS), 4.4))
    axes = np.atleast_1d(axes)
    for j, lbl in enumerate(LABEL_COLS):
        ax = axes[j]
        for arm in ["A"] + list(ARMS):
            r = summary[lbl][arm]
            ax.scatter(r["ACC"], r["BWT"], s=140, color=CMAP[arm], edgecolor="black", zorder=3)
            ax.annotate(arm, (r["ACC"], r["BWT"]), textcoords="offset points",
                        xytext=(7, 5), fontsize=8)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("ACC (mean AUROC over all eras)")
        ax.set_ylabel("BWT (negative = forgetting)")
        ax.set_title(lbl.replace("label_", "").replace("_", " ").title(), fontsize=11)
        ax.grid(alpha=0.3)
    fig.suptitle("Upper right is the two-stream claim: adapts well AND forgets least", y=1.02)
    plt.tight_layout()
    plt.savefig(SAVE_PATH / "acc_vs_bwt.png", dpi=150, bbox_inches="tight")
    plt.close()

# Final-era gain vs source-era retention (the single-changepoint view).
if adaptation_performed and arm_post:
    order = ["A"] + list(ARMS)
    fig, axes = plt.subplots(1, len(LABEL_COLS), figsize=(5.2 * len(LABEL_COLS), 4.6))
    axes = np.atleast_1d(axes)
    cmap = {"A": "#d62728", "B": "#2ca02c", "B_mirror": "#8c564b",
            "C": "#1f77b4", "C_l2sp": "#9467bd"}
    for j, lbl in enumerate(LABEL_COLS):
        ax = axes[j]
        for arm in order:
            gain = (arm_post[arm].get(f"{lbl}_auroc", np.nan)
                    - arm_post["A"].get(f"{lbl}_auroc", np.nan))
            ret = arm_ret_val.get(arm, {}).get(lbl, {}).get("retention", np.nan)
            ax.scatter(gain, ret, s=130, color=cmap[arm], edgecolor="black", zorder=3, label=arm)
            ax.annotate(arm, (gain, ret), textcoords="offset points",
                        xytext=(7, 5), fontsize=8)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.axvline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("Post-drift AUROC gain vs Run A")
        ax.set_ylabel("Retention on source era (Δ AUROC)")
        ax.set_title(lbl.replace("label_", "").replace("_", " ").title(), fontsize=11)
        ax.grid(alpha=0.3)
    fig.suptitle("Adaptation gain vs forgetting — upper right is the two-stream claim", y=1.02)
    plt.tight_layout()
    plt.savefig(SAVE_PATH / "gain_vs_retention.png", dpi=150, bbox_inches="tight")
    plt.close()

# Incident vs prevalent AUROC side by side.
if adaptation_performed and HAS_META and arm_post.get("B", {}).get("_stratified"):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width, xs = 0.26, np.arange(len(LABEL_COLS))
    for k, (cohort, off) in enumerate([("", -width), ("incident_", 0), ("prevalent_", width)]):
        vals = [arm_post["B"].get(f"{cohort}{lbl}_auroc", np.nan) for lbl in LABEL_COLS]
        ax.bar(xs + off, vals, width, label=(cohort.rstrip("_") or "overall"), edgecolor="black")
    ax.set_xticks(xs)
    ax.set_xticklabels([l.replace("label_", "") for l in LABEL_COLS])
    ax.axhline(0.5, color="red", ls="--", lw=1)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Run B post-drift: incident (new need) vs prevalent (continuation)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SAVE_PATH / "incident_vs_prevalent.png", dpi=150, bbox_inches="tight")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
_ckpt = {
    "source": source_state,
    "run_a": model_A.state_dict(),
    "seq_dim": seq_dim, "treat_dim": treat_dim, "n_targets": len(LABEL_COLS),
    "train_stats": train_stats,
    "config": {
        "adaptation_performed": adaptation_performed,
        "source_seeds": SOURCE_SEEDS, "median_source_seed": _median_src_seed,
        "best_lr": {k: float(v) for k, v in best_lr.items()},
        "trainable_params": _cap,
        "adapt_loss": {"gamma": 1.5, "label_smoothing": 0.02, "max_weight": 20.0},
        "ablate_leaky_physio": ABLATE_LEAKY_PHYSIO,
    },
}
if adaptation_performed:
    for _a in ARMS:
        _ckpt[f"run_{_a.lower()}"] = adapted[_a]["model"].state_dict()
torch.save(_ckpt, SAVE_PATH / "two_stream_models.pt")

# Per-seed SOURCE states. script10 needs a second, differently-seeded source
# model to compute its seed-noise floor, and without that floor no retrieval
# stability number is interpretable — it refuses to issue a verdict. Saving
# them here costs nothing; regenerating them later costs a full re-run.
torch.save({f"seed_{s}": st for s, st in zip(SOURCE_SEEDS, all_source_states)},
           SAVE_PATH / "two_stream_source_seeds.pt")

if PIN is not None:
    _bad = []
    for e_ in EVAL_ERAS:
        if sorted(int(x) for x in eval_ds_by_era[e_].stay_ids) != sorted(int(x) for x in PIN["era_eval_stays"][e_]):
            _bad.append(f"eval {e_}")
    for e_, c in adapt_cohorts.items():
        for k in ("adapt_train_stays", "adapt_val_es_stays", "adapt_val_sel_stays", "replay_stays"):
            if sorted(int(x) for x in c.get(k, [])) != sorted(int(x) for x in PIN["cohorts"][e_].get(k, [])):
                _bad.append(f"{e_} {k}")
    _bad_eval = [b for b in _bad if b.startswith("eval ")]
    if _bad_eval:
        raise SystemExit(f"PIN_SPLIT mismatch (evaluation cohorts): {_bad_eval}")
    if _bad:
        # CLEAN-V2: fatal again. With a pin that has adaptation cohorts for every era
        # and FORCE_ADAPT=1, every cohort must match; a mismatch means the run used
        # cohorts other than the pinned ones (the clean_full era-3 bug).
        raise SystemExit(f"PIN_SPLIT mismatch (adaptation cohorts): {_bad}")
    else:
        print("PIN_SPLIT: every cohort matches the pinned split exactly")
with open(SAVE_PATH / "eval_split.json", "w") as f:
    json.dump({
        "pre_cp_stays": [int(x) for x in pre_cp_stays],
        "post_cp_stays": [int(x) for x in post_cp_stays],
        "eval_post_stays": [int(x) for x in eval_post_stays],
        "adapt_train_stays": [int(x) for x in adapt_train_stays],
        "adapt_val_stays": [int(x) for x in adapt_val_stays],
        # Last era at which weights actually changed — a no-update era has empty
        # cohorts, and script4 uses these to fit its source-side calibrator.
        "adapt_val_es_stays": [int(x) for x in
                               adapt_cohorts.get(adapted_eras[-1] if adapted_eras else None, {})
                                            .get("adapt_val_es_stays", [])],
        "adapt_val_sel_stays": [int(x) for x in
                                adapt_cohorts.get(adapted_eras[-1] if adapted_eras else None, {})
                                             .get("adapt_val_sel_stays", [])],
        # ── The full sequential schedule, so script3 can replay it exactly ──
        # Run D must walk the SAME boundaries in the SAME order with the SAME
        # cohorts. A Run D that adapts once while B/C adapt three times is not a
        # comparison of architectures, it is a comparison of update counts.
        "sequential": {
            "eval_eras": EVAL_ERAS,
            "source_era_key": SOURCE_ERA_KEY,
            "adapt_steps": list(step_order),
            "adapted_eras": list(adapted_eras),
            "quiet_boundaries": [e for e in step_order if e not in adapted_eras],
            "era_eval_stays": {e: [int(x) for x in eval_ds_by_era[e].stay_ids]
                               for e in EVAL_ERAS},
            "cohorts": {e: {k: ([int(x) for x in v] if isinstance(v, list) else v)
                            for k, v in c.items()}
                        for e, c in adapt_cohorts.items()},
        },
        "buf_pre_stays": [int(x) for x in buf_pre_stays],
        "val_calib_stays": [int(x) for x in val_calib_df["stay_id"].unique().to_list()],
        "drift_tag": str(drift_tag), "drift_branch": drift_branch,
        "seeds": ADAPT_SEEDS,
        # script3 must reproduce these exactly to stay comparable.
        "adapt_loss": {"gamma": 1.5, "label_smoothing": 0.02, "max_weight": 20.0},
        "lr_grid": LR_ADAPT_GRID,
        "best_lr": {k: float(v) for k, v in best_lr.items()},
        "run_b_trainable_params": _cap.get("B"),
    }, f, indent=2)

with open(SAVE_PATH / "drift_detection_log.json", "w") as f:
    json.dump(to_serialisable({
        "pairs": detection_log,
        # The closed-loop trace is the record that matters: it is what actually
        # scheduled the updates. `pairs` is the era-pair reporting scan.
        "closed_loop": online_log,
        "gate_branches": list(GATE_BRANCHES),
        "stage1_thresholds": {f"{t}@{n}": v[0] for (t, n), v in _thr_cache.items()},
        "stage2_thresholds": {"treatment": treat_ratio_thr, "physiology": physio_ratio_thr},
        "stage2_info": {"treatment": treat_ratio_info, "physiology": physio_ratio_info},
        "null_draws": NULL_DRAWS,
        "perf_thresholds": perf_thresholds,
        "src_calib_auroc": src_calib_auroc,
        "physio_monitor": PHYSIO_MONITOR,
        "physio_drift_action": PHYSIO_DRIFT_ACTION,
        "physio_gate_ever_fired": bool(any(r["physio_fired"] for r in online_log)),
        "n_missed_updates": int(sum(1 for r in online_log if r["oracle_missed_update"])),
        "drift_group": drift_group, "drift_branch": drift_branch,
        # The gate is label-free by construction now, so the run is deployable
        # unless the oracle was explicitly allowed to trigger.
        "deployable": "performance" not in GATE_BRANCHES,
        "drifted_features": drifted_post,
        "temporal_separation_audit": temporal_audit,
        "cohort_year_ranges": _split_years,
    }), f, indent=2)

if adaptation_performed and len(eval_post_stays) > 0:
    np.savez(SAVE_PATH / "post_drift_predictions.npz",
             labels=labels_ref,
             **{f"probs_{a.lower()}": arm_probs[a] for a in arm_probs},
             already_vaso=eval_post_ds.meta.get("already_vaso", np.array([])),
             already_vent=eval_post_ds.meta.get("already_vent", np.array([])),
             stay_ids=np.array(eval_post_ds.stay_ids))

    # Per-era predictions at the FINAL step, for every arm. script3 needs these
    # to run paired tests on the retention eras too, not just the current one —
    # forgetting is measured on the earlier eras, so that is where the B vs D
    # contrast actually matters.
    _era_dump = {}
    _final = FINAL_ERA
    for _e in EVAL_ERAS:
        _key = _e.replace(" ", "").replace("-", "_").replace("(", "").replace(")", "")
        _era_dump[f"labels__{_key}"] = eval_ds_by_era[_e].labels
        _era_dump[f"stayids__{_key}"] = np.array(eval_ds_by_era[_e].stay_ids)
        for _a in ["A"] + list(ARMS):
            _era_dump[f"probs__{_key}__{_a}"] = step_probs[_final][_e][_a]
        for _m in ("already_vaso", "already_vent"):
            if _m in eval_ds_by_era[_e].meta:
                _era_dump[f"{_m}__{_key}"] = eval_ds_by_era[_e].meta[_m]
    np.savez(SAVE_PATH / "per_era_predictions.npz",
             eval_eras=np.array(EVAL_ERAS, dtype=object), **_era_dump)

    with open(SAVE_PATH / "arm_results.json", "w") as f:
        json.dump(to_serialisable({
            "post": arm_post, "retention_pre": arm_ret_pre, "retention_val": arm_ret_val,
            "lr_sweep": lr_sweep_log, "best_lr": {k: float(v) for k, v in best_lr.items()},
            "trainable_params": _cap,
        }), f, indent=2)

# ── IG ATTRIBUTION ─────────────────────────────────────────────────────────────
if adaptation_performed and len(eval_post_stays) > 0:
    ds_fig = eval_post_ds
    LABEL_IDX = 0
    fig_patient_i = next((ci for ci in range(len(ds_fig))
                          if ds_fig.labels[ci, LABEL_IDX] == 1), 0)
    seq_t, treat_t, lbl_t = ds_fig[fig_patient_i]

    def integrated_gradients(model, xs, xt, ti, steps=30):
        model.train()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        xs, xt = xs.to(device), xt.to(device)
        bs, bt = torch.zeros_like(xs), torch.zeros_like(xt)
        sg, tg = torch.zeros_like(xs), torch.zeros_like(xt)
        for alpha in np.linspace(0, 1, steps):
            is_ = (bs + alpha * (xs - bs)).requires_grad_(True)
            it_ = (bt + alpha * (xt - bt)).requires_grad_(True)
            out = model(is_, it_)[:, ti].sum()
            g1, g2 = torch.autograd.grad(out, [is_, it_])
            sg += g1
            tg += g2
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = DROPOUT
        model.eval()
        return (((xs - bs) * sg / steps).detach().cpu().numpy()[0].sum(axis=0),
                ((xt - bt) * tg / steps).detach().cpu().numpy()[0])

    xs_in, xt_in = seq_t.unsqueeze(0), treat_t.unsqueeze(0)
    p_src, t_src = integrated_gradients(source_model, xs_in, xt_in, LABEL_IDX)
    p_adp, t_adp = integrated_gradients(adapted["B"]["model"], xs_in, xt_in, LABEL_IDX)

    ig_data = {
        "all_phys_deltas": {n: {"src": float(s), "adp": float(a), "delta": abs(a - s)}
                            for n, s, a in zip(ds_fig.seq_cols, p_src, p_adp)},
        "all_treat_deltas": {n: {"src": float(s), "adp": float(a), "delta": abs(a - s)}
                             for n, s, a in zip(ds_fig.treat_cols, t_src, t_adp)},
        "stay_id": int(ds_fig.stay_ids[fig_patient_i]),
        "true_label": int(lbl_t[LABEL_IDX].item()),
        "already_vaso": float(ds_fig.meta.get("already_vaso", np.zeros(len(ds_fig)))[fig_patient_i]),
    }
    with open(SAVE_PATH / "fig_drift-harm_ig_data.pkl", "wb") as f:
        pickle.dump(ig_data, f)

print("\n" + "=" * 70 + "\nSUMMARY & AUDIT LOG\n" + "=" * 70)
human_text = _safe("human_summary", explainer.generate_human_summary) or "(summary unavailable)"
print(human_text)
with open(SAVE_PATH / "clinical_drift_audit.txt", "w") as f:
    f.write(human_text)

print("\nWhat to check before writing this up:")
print("  1. acc_vs_bwt.png — Run B should sit up and to the RIGHT of C and C_l2sp:")
print("     comparable ACC, BWT closer to zero. That is the two-stream claim in one")
print("     figure. If C_l2sp matches B, the effect is regularization, not structure.")
print("  2. retention_trajectory_*.png — after an era's own adaptation step, a flat")
print("     line means no forgetting and a falling line means forgetting. Run C should")
print("     fall faster with each additional step; that compounding is the mechanism.")
print("  3. B vs B_mirror in paired_contrasts.json, on the EARLIER eras. If the mirror")
print("     ties B, freezing a large block is what helped and the decomposition claim")
print("     does not hold — this is the control a reviewer will look for first.")
print("  4. placebo_adaptation.json — the placebo ran at a boundary the detector called")
print("     quiet. If its gain approaches the real gain, 'adaptation helps' is really")
print("     'more data helps' and the drift framing is not carrying weight.")
print("  5. incident_* columns everywhere — the incident AUROC is the headline number,")
print("     not the overall one. Expect it substantially lower; that is the honest value.")
print("  6. drift_detection_log.json → 'deployable'. If false, only the label-dependent")
print("     oracle fired and the auto-detection claim has to be softened.")
print("\nOn gradual drift: multiple boundaries firing is a RESULT, not a problem. State")
print("it plainly — 'drift in MIMIC-IV is continuous rather than a single event, so we")
print("adapt sequentially' is a stronger and more honest framing than a single")
print("changepoint, and it is why the sequential protocol is used here.")
