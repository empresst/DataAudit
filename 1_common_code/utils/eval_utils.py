"""
eval_utils.py — metrics, cohort stratification, retention, and paired inference.

Three things this module exists to make measurable, none of which the previous
pipeline computed:

1. RETENTION (backward transfer). "Two-stream keeps performance better after
   adaptation" is a claim about what the ADAPTED model does on PRE-drift data.
   That number was never computed — pre-drift rows were filled in with Run A's
   metrics for all arms. retention_table() computes it properly.

2. INCIDENT vs PREVALENT. Under the ongoing-need label framing, a patient on
   norepinephrine from h1 to h20 is a positive, and h0-h5 features such as
   has_central_line_obs identify that patient trivially. That is not temporal
   leakage — the features do precede the label — but it means AUROC is driven by
   treatment continuation rather than prediction of new need. Incident AUROC
   (patients NOT on treatment by h5) is the clinically meaningful endpoint;
   prevalent AUROC is the continuation detector. Report both.

3. PAIRED INFERENCE between arms, on the same patients and the same bootstrap
   resamples. Both the percentile and BCa intervals live here so the two scripts
   cannot drift apart in how they compute them.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


# ── BOOTSTRAP ──────────────────────────────────────────────────────────────────
def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a single model. NOT BCa — see paired_bca."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_prob)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        vals.append(metric_fn(yt, yp))
    if not vals:
        return float(point), float("nan"), float("nan")
    return (float(point),
            float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def paired_bca(y_true, p1, p2, metric_fn, n_boot=1000, seed=11, alpha=0.05, n_blocks=200):
    """BCa CI for the paired difference metric_fn(y, p1) - metric_fn(y, p2)."""
    from scipy.stats import norm as _norm
    rng = np.random.default_rng(seed)
    n = len(y_true)
    theta_hat = metric_fn(y_true, p1) - metric_fn(y_true, p2)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        boots.append(metric_fn(yt, p1[idx]) - metric_fn(yt, p2[idx]))
    boots = np.asarray(boots)
    if boots.size == 0:
        return theta_hat, float("nan"), float("nan"), float("nan")

    prop = np.mean(boots < theta_hat)
    prop = min(max(prop, 1.0 / (2 * boots.size)), 1 - 1.0 / (2 * boots.size))
    z0 = _norm.ppf(prop)

    order = rng.permutation(n)
    blocks = np.array_split(order, n_blocks)
    jack = []
    for b in blocks:
        keep = np.setdiff1d(order, b, assume_unique=False)
        yt = y_true[keep]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        jack.append(metric_fn(yt, p1[keep]) - metric_fn(yt, p2[keep]))
    jack = np.asarray(jack)
    if jack.size < 3:
        a = 0.0
    else:
        d = jack.mean() - jack
        denom = 6.0 * (np.sum(d ** 2) ** 1.5)
        a = float(np.sum(d ** 3) / denom) if denom > 0 else 0.0

    def _adj(z):
        return _norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo = float(np.percentile(boots, 100 * _adj(_norm.ppf(alpha / 2))))
    hi = float(np.percentile(boots, 100 * _adj(_norm.ppf(1 - alpha / 2))))
    p_gt = float((boots > 0).mean())
    return float(theta_hat), lo, hi, p_gt


# ── FAST PAIRED INFERENCE ──────────────────────────────────────────────────────
# paired_bca() recomputes both models' bootstrap replicates on every call. With
# 9 contrasts over 5 arms that is 18 arm-evaluations where 5 suffice, because
# every contrast at a given (era, label, cohort) draws the SAME resample indices
# from the same seed. Computing each arm's replicates once and differencing them
# gives bit-identical results roughly 3-4x faster — an exact optimisation, not an
# approximation. Resamples where the label is degenerate are dropped for ALL arms
# together, exactly as paired_bca drops them.
def bootstrap_cache(y_true, probs_by_arm, metric_fn, n_boot=1000, seed=11, n_blocks=200):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    arms = list(probs_by_arm)

    boot_idx, keep = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        boot_idx.append(idx)
        keep.append(0 < yt.sum() < n)

    order = rng.permutation(n)
    blocks = np.array_split(order, n_blocks)
    jack_keep, jack_idx = [], []
    for b in blocks:
        kept = np.setdiff1d(order, b, assume_unique=False)
        yt = y_true[kept]
        jack_idx.append(kept)
        jack_keep.append(0 < yt.sum() < len(yt))

    cache = {}
    for a in arms:
        p = probs_by_arm[a]
        cache[a] = {
            "point": float(metric_fn(y_true, p)),
            "boot": np.array([metric_fn(y_true[i], p[i])
                              for i, k in zip(boot_idx, keep) if k]),
            "jack": np.array([metric_fn(y_true[i], p[i])
                              for i, k in zip(jack_idx, jack_keep) if k]),
        }
    return cache


def paired_bca_cached(cache, x, y, alpha=0.05):
    """BCa CI for arm x minus arm y, from precomputed replicates."""
    from scipy.stats import norm as _norm
    theta_hat = cache[x]["point"] - cache[y]["point"]
    boots = cache[x]["boot"] - cache[y]["boot"]
    if boots.size == 0:
        return theta_hat, float("nan"), float("nan"), float("nan")

    # Exact tie in every resample: the two arms are the same model on this cell.
    # Falling through would give (boots > 0).mean() == 0, which
    # bootstrap_p_two_sided maps to p = 1/n_boot -- maximal significance for zero
    # evidence. Return NaN instead; holm_reject already drops NaN cells from the
    # family. Only reachable on degenerate cells, so no non-degenerate result moves.
    if np.allclose(boots, 0.0):
        return float(theta_hat), 0.0, 0.0, float("nan")

    prop = np.mean(boots < theta_hat)
    prop = min(max(prop, 1.0 / (2 * boots.size)), 1 - 1.0 / (2 * boots.size))
    z0 = _norm.ppf(prop)

    jack = cache[x]["jack"] - cache[y]["jack"]
    if jack.size < 3:
        a = 0.0
    else:
        d = jack.mean() - jack
        denom = 6.0 * (np.sum(d ** 2) ** 1.5)
        a = float(np.sum(d ** 3) / denom) if denom > 0 else 0.0

    def _adj(z):
        return _norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo = float(np.percentile(boots, 100 * _adj(_norm.ppf(alpha / 2))))
    hi = float(np.percentile(boots, 100 * _adj(_norm.ppf(1 - alpha / 2))))
    return float(theta_hat), lo, hi, float((boots > 0).mean())


# ── MULTIPLICITY ───────────────────────────────────────────────────────────────
# Every contrast is tested at 3 labels x ~4 eras. At alpha=0.05 uncorrected,
# "significant in 2 of 7 cells" is roughly what the NULL produces, so an
# uncorrected tally cannot distinguish a weak real effect from noise — and the
# error runs in both directions: it inflates the proposed method's wins and the
# baseline's wins equally.
#
# Holm-Bonferroni, applied WITHIN a contrast family (one contrast across its
# cells). Holm rather than plain Bonferroni because it is uniformly more powerful
# at the same family-wise error rate, and it needs no independence assumption —
# which matters here, since septic shock is a strict subset of vasopressor and
# the cells are therefore correlated by construction.
#
# The bootstrap gives a two-sided p directly: p = 2*min(P(delta>0), P(delta<0)),
# floored at 1/n_boot because a bootstrap cannot resolve below its resolution.
def bootstrap_p_two_sided(p_x_gt_y, n_boot):
    if p_x_gt_y != p_x_gt_y:
        return float("nan")
    p = 2.0 * min(p_x_gt_y, 1.0 - p_x_gt_y)
    return float(min(1.0, max(p, 1.0 / max(n_boot, 1))))


def holm_reject(pvals, alpha=0.05):
    """
    Holm-Bonferroni step-down. Returns a boolean list aligned with `pvals`.

    NaN p-values (degenerate cells) are never rejected and are excluded from the
    family size — counting an uncomputable cell against the correction would
    penalise the comparison for something that was never tested.
    """
    idx = [i for i, p in enumerate(pvals) if p == p]
    m = len(idx)
    out = [False] * len(pvals)
    if m == 0:
        return out
    order = sorted(idx, key=lambda i: pvals[i])
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break        # step-down: the first failure stops all the rest
    return out


# ── METRICS ────────────────────────────────────────────────────────────────────
def metrics_from_probs(labels, probs, label_cols, row_mask=None, prefix=""):
    """
    Per-label AUROC / AUPRC / Brier over an optional subset of rows.

    Deliberately per-label with no mean. Your own assert in script1 proves
    septic shock is a strict subset of vasopressor by construction, so averaging
    three AUROCs double-counts the vasopressor task.
    """
    out = {}
    for i, lbl in enumerate(label_cols):
        y_true, y_prob = labels[:, i], probs[:, i]
        if row_mask is not None:
            y_true, y_prob = y_true[row_mask], y_prob[row_mask]
        n_pos, n_neg = int(y_true.sum()), int(len(y_true) - y_true.sum())
        k = f"{prefix}{lbl}"
        if n_pos > 0 and n_neg > 0:
            out[f"{k}_auroc"] = float(roc_auc_score(y_true, y_prob))
            out[f"{k}_auprc"] = float(average_precision_score(y_true, y_prob))
            out[f"{k}_brier"] = float(brier_score_loss(y_true, y_prob))
        else:
            out[f"{k}_auroc"] = float("nan")
            out[f"{k}_auprc"] = float("nan")
            out[f"{k}_brier"] = float("nan")
        out[f"{k}_n_pos"] = n_pos
        out[f"{k}_n"] = int(len(y_true))
    return out


def cohort_masks(dataset, label_cols, incident_gate):
    """
    Per-label boolean row masks for the incident and prevalent cohorts.

    Returns {} when the metadata columns are absent, so callers degrade to
    unstratified reporting rather than crashing.
    """
    masks = {}
    for lbl in label_cols:
        gate = incident_gate.get(lbl)
        arr = dataset.meta.get(gate) if gate else None
        if arr is None or np.all(arr < 0):
            continue
        masks[lbl] = {"incident": arr <= 0, "prevalent": arr > 0}
    return masks


def stratified_metrics(dataset, labels, probs, label_cols, incident_gate):
    """
    Overall + incident + prevalent metrics in one dict.

    Primary endpoint for the paper: `incident_<label>_auroc`.
    Expect it to be substantially lower than overall. That is the honest number
    and the one that makes the result a prediction of new need rather than a
    detection of ongoing treatment.
    """
    out = metrics_from_probs(labels, probs, label_cols)
    masks = cohort_masks(dataset, label_cols, incident_gate)
    if not masks:
        out["_stratified"] = False
        return out
    out["_stratified"] = True
    for lbl, m in masks.items():
        i = label_cols.index(lbl)
        for cohort, row_mask in m.items():
            sub = metrics_from_probs(
                labels[:, [i]], probs[:, [i]], [lbl], row_mask=row_mask,
                prefix=f"{cohort}_")
            out.update(sub)
    return out


def prevalence_audit(dataset, labels, label_cols, incident_gate):
    """
    How much of each label's positive mass is prevalent rather than incident.

    A label whose positives are >80% prevalent is a continuation detector no
    matter how good the AUROC looks. Print this next to every headline number.
    """
    rows = {}
    masks = cohort_masks(dataset, label_cols, incident_gate)
    for i, lbl in enumerate(label_cols):
        if lbl not in masks:
            continue
        y = labels[:, i]
        inc, prev = masks[lbl]["incident"], masks[lbl]["prevalent"]
        n_pos = int(y.sum())
        rows[lbl] = {
            "gate": incident_gate[lbl],
            "n_total": int(len(y)),
            "n_incident": int(inc.sum()),
            "n_prevalent": int(prev.sum()),
            "n_pos_total": n_pos,
            "n_pos_incident": int(y[inc].sum()),
            "n_pos_prevalent": int(y[prev].sum()),
            "frac_pos_prevalent": float(y[prev].sum() / n_pos) if n_pos else float("nan"),
            "incident_prevalence": float(y[inc].mean()) if inc.sum() else float("nan"),
            "prevalent_prevalence": float(y[prev].mean()) if prev.sum() else float("nan"),
        }
    return rows


# ── RETENTION / FORGETTING ─────────────────────────────────────────────────────
def retention_table(before, after, label_cols, metric="auroc"):
    """
    Backward transfer: metric(adapted model, old-era data)
                     - metric(source model, old-era data).

    Negative = forgetting. This is the quantity the two-stream claim rests on.
    `before` must be the SOURCE model evaluated on the old-era split, and
    `after` the SAME arm's ADAPTED model on the SAME split.
    """
    out = {}
    for lbl in label_cols:
        b = before.get(f"{lbl}_{metric}", float("nan"))
        a = after.get(f"{lbl}_{metric}", float("nan"))
        out[lbl] = {"before": b, "after": a, "retention": a - b}
    return out


def summarize_arm(name, post, retention_pre, retention_val, label_cols):
    """One row per label: post-drift gain and both retention figures."""
    lines = [f"\n{name}"]
    lines.append(f"  {'label':<24}{'post AUROC':>12}{'inc AUROC':>12}"
                 f"{'prev AUROC':>12}{'Δpre':>10}{'Δval':>10}")
    for lbl in label_cols:
        short = lbl.replace("label_", "")
        lines.append(
            f"  {short:<24}"
            f"{post.get(f'{lbl}_auroc', float('nan')):>12.4f}"
            f"{post.get(f'incident_{lbl}_auroc', float('nan')):>12.4f}"
            f"{post.get(f'prevalent_{lbl}_auroc', float('nan')):>12.4f}"
            f"{retention_pre.get(lbl, {}).get('retention', float('nan')):>+10.4f}"
            f"{retention_val.get(lbl, {}).get('retention', float('nan')):>+10.4f}")
    return "\n".join(lines)


# ── SEQUENTIAL / CONTINUAL EVALUATION ──────────────────────────────────────────
# Under gradual drift there is no single changepoint, so the pre/post framing
# breaks down: whichever era you call "first" will itself have drifted. The
# standard alternative is the continual-learning protocol (Lopez-Paz & Ranzato,
# GEM, 2017): adapt at each boundary in sequence and keep a full matrix
#
#     R[t][e] = performance on era e after adaptation step t
#
# Every era stays in the evaluation forever, which is what makes forgetting
# visible. Two summaries come out of it:
#
#   ACC  mean performance over all eras seen so far, after the final step
#   BWT  mean over earlier eras of (performance now - performance when that era
#        was current). Negative = forgetting. This is the quantity the two-stream
#        claim is about, and it COMPOUNDS over repeated updates — which is why
#        the sequential design is a stronger test of that claim than a single
#        adaptation event.

def backward_transfer(R, step_order, era_order, label, metric="auroc"):
    """
    BWT after the final step: mean over eras e (excluding the last-learned) of
        R[final][e] - R[diag(e)][e]
    where diag(e) is the step at which era e was itself adapted on.
    """
    if not step_order:
        return float("nan")
    final = step_order[-1]
    diffs = []
    for e in era_order:
        if e not in R.get(final, {}):
            continue
        diag_step = e if e in R else None
        if diag_step is None or e == final:
            continue
        now = R[final][e].get(f"{label}_{metric}", float("nan"))
        then = R[diag_step][e].get(f"{label}_{metric}", float("nan"))
        if now == now and then == then:
            diffs.append(now - then)
    return float(np.mean(diffs)) if diffs else float("nan")


def average_accuracy(R, step_order, era_order, label, metric="auroc"):
    """ACC: mean performance across all evaluated eras after the final step."""
    if not step_order:
        return float("nan")
    vals = [R[step_order[-1]][e].get(f"{label}_{metric}", float("nan"))
            for e in era_order if e in R.get(step_order[-1], {})]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def format_retention_matrix(R, step_order, era_order, label, metric="auroc", baseline_step=None):
    """Render R as a text table: rows = adaptation steps, columns = eras."""
    w = max(14, max((len(str(e)) for e in era_order), default=14) + 2)
    lines = [f"  {'after step':<20}" + "".join(f"{str(e):>{w}}" for e in era_order)]
    rows = ([baseline_step] if baseline_step else []) + list(step_order)
    for t in rows:
        if t not in R:
            continue
        cells = ""
        for e in era_order:
            v = R[t].get(e, {}).get(f"{label}_{metric}", float("nan"))
            cells += f"{v:>{w}.4f}" if v == v else f"{'-':>{w}}"
        lines.append(f"  {str(t):<20}" + cells)
    return "\n".join(lines)


def to_serialisable(d):
    """NaN → None, numpy scalars → python, for json.dump."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = to_serialisable(v)
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (float, np.floating)):
            out[k] = None if v != v else float(round(float(v), 6))
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [to_serialisable(x) if isinstance(x, dict) else
                      (None if isinstance(x, float) and x != x else x) for x in v]
        else:
            out[k] = v
    return out
