"""
drift_utils.py — the label-free drift detector, shared by script2 and script6.

One implementation, imported by both, so the era-blocked analysis and the
streaming analysis cannot silently diverge in how they score drift.
"""
import numpy as np
from scipy.stats import ks_2samp


def compute_psi(ref, cur, edges=None, bins=10):
    """Population Stability Index between two samples of one feature."""
    if len(cur) < 10:
        return 0.0
    if edges is None:
        edges = np.percentile(ref, np.linspace(0, 100, bins + 1))
        edges[0] -= 1e-6
        edges[-1] += 1e-6
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    # OUTER EDGES MUST BE INFINITE. Percentile edges span only the reference
    # range, so np.histogram silently DROPS any current value outside it — and
    # out-of-range mass is the strongest evidence of drift there is. With finite
    # edges a cohort lying entirely outside the reference scored PSI ~= 0 (the
    # Laplace prior then dominates a histogram of all zeros) instead of a large
    # value, so the detector was biased against firing on exactly the tail shifts
    # that matter, e.g. crystalloid volume falling 791 -> 255 mL.
    edges = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
    ref_h, _ = np.histogram(ref, bins=edges)
    cur_h, _ = np.histogram(cur, bins=edges)
    ref_p = (ref_h + 1) / (ref_h.sum() + len(edges) - 1)
    cur_p = (cur_h + 1) / (cur_h.sum() + len(edges) - 1)
    return float(np.sum((ref_p - cur_p) * np.log(ref_p / cur_p)))


def distribution_drift(prev_df, curr_df, cont_cols, bin_cols, ref_edges=None,
                       psi_thresh=0.20, binary_delta_thr=0.05, min_n=20):
    """
    Label-free drift score between two cohorts. This is the DEPLOYABLE detector:
    it needs no outcome labels, so it can run at prediction time.

    Returns (score, n_drifted, drifted_continuous, drifted_binary).
    """
    psis, drifted_cont = [], []
    for c in cont_cols:
        if c not in prev_df.columns or c not in curr_df.columns:
            continue
        ref = prev_df[c].to_numpy().astype(float)
        cur = curr_df[c].to_numpy().astype(float)
        if len(ref) < min_n or len(cur) < min_n:
            continue
        psi_val = compute_psi(ref, cur, edges=(ref_edges or {}).get(c))
        _, ks_p = ks_2samp(ref, cur)
        psis.append(psi_val)
        if psi_val > psi_thresh or ks_p < 0.01:
            drifted_cont.append((c, psi_val, float(ks_p)))

    bdeltas, drifted_bin = [], []
    for c in bin_cols:
        if c not in prev_df.columns or c not in curr_df.columns:
            continue
        r_rate = prev_df[c].to_numpy().astype(float).mean()
        c_rate = curr_df[c].to_numpy().astype(float).mean()
        delta = abs(c_rate - r_rate)
        bdeltas.append(delta)
        if delta > binary_delta_thr or (r_rate >= 0.02 and delta / (r_rate + 1e-9) > 0.20):
            drifted_bin.append((c, float(delta)))

    mean_psi = float(np.mean(psis)) if psis else 0.0
    mean_bin = float(np.mean(bdeltas)) if bdeltas else 0.0
    return (mean_psi + 0.5 * mean_bin, len(drifted_cont) + len(drifted_bin),
            drifted_cont, drifted_bin)


def calibrate_threshold(ref_df, cont_cols, bin_cols, batch_size, n_draws=200,
                        ref_edges=None, quantile=99.0, sd_mult=3.0, seed=1000,
                        psi_thresh=0.20, binary_delta_thr=0.05):
    """
    Null distribution of the drift score, built by repeatedly splitting a
    known-stable reference period into two random halves of `batch_size`.

    Threshold = max(empirical quantile, mean + sd_mult*sd). The empirical
    quantile of a finite sample from a right-skewed null sits below the true
    quantile, which inflates the false-alarm rate and does NOT improve with more
    draws — the bias is in the order statistic. The mean+3sd term controls the
    tail properly; the quantile is kept as a floor.

    The threshold depends on batch size (smaller batches are noisier), so it
    must be recalibrated whenever the batch size changes — an era-level
    threshold is far too loose for yearly batches.
    """
    scores = []
    for d in range(n_draws):
        shuf = ref_df.sample(fraction=1.0, shuffle=True, seed=seed + d)
        a, b = shuf.head(batch_size), shuf.slice(batch_size, batch_size)
        if a.height < 20 or b.height < 20:
            continue
        s, _, _, _ = distribution_drift(a, b, cont_cols, bin_cols, ref_edges=ref_edges,
                                        psi_thresh=psi_thresh,
                                        binary_delta_thr=binary_delta_thr)
        scores.append(s)
    if len(scores) < 10:
        raise ValueError(
            f"only {len(scores)} usable null draws at batch_size={batch_size} from "
            f"{ref_df.height} reference rows — cannot calibrate. Use larger batches.")
    scores = np.asarray(scores)
    q = float(np.percentile(scores, quantile))
    p = float(scores.mean() + sd_mult * scores.std(ddof=1))
    return max(q, p), {"n_draws": len(scores), "median": float(np.median(scores)),
                       "quantile": q, "mean_plus_sd": p, "batch_size": batch_size}


# ── SECOND-STAGE CALIBRATION ──────────────────────────────────────────────────
# calibrate_threshold() removes SAMPLING noise: a ratio of 1.0 means "as
# different as two random halves of the same period". But two different YEARS are
# never two random halves — practice moves a little every year even inside a
# period the model was trained through and performs acceptably on. Judging a test
# period against a year-mixed null therefore flags normal annual variation as
# drift; measured train-era alarm rates were 67-100% before this stage existed.
#
# So calibrate a second time on the reference period's OWN year-to-year ratios.
# The operative question becomes "is this shift larger than the year-to-year
# shifts the model was trained through?", which is the question that actually
# bears on whether the model is still valid.
#
# With a handful of reference years the MAX is used rather than a percentile: a
# 95th percentile of 3-4 observations is not an estimate. That is deliberately
# conservative — it raises the bar for firing — and must be stated as such.
#
# IMPORTANT: script2 and script6 must both call this. A batch-level threshold
# alone gave the two scripts opposite answers about whether drift occurred.
def year_ratio_null(year_frames, cont_cols, bin_cols, threshold_fn,
                    ref_edges=None, psi_thresh=0.20, binary_delta_thr=0.05,
                    min_year_n=60, safety=1.0):
    """
    Largest drift ratio observed BETWEEN YEARS inside a known-stable period.

    `year_frames`  {year:int -> frame} for the reference period only.
    `threshold_fn` n -> first-stage threshold at that sample size.

    Returns (thresholds, info) where thresholds has a key per mode:
      rolling  consecutive-year pairs (y-1 vs y)      — detects CHANGE
      fixed    year y vs the rest of the period       — detects DEPARTURE
                 (leave-one-year-out: comparing y against a pool CONTAINING y
                  understates its score, which makes the reference artificially
                  quiet and every downstream threshold too permissive)

    A period with fewer than two usable years yields 1.0, i.e. no second stage —
    the caller must report that the second stage was inactive rather than let it
    pass silently.
    """
    usable = {int(y): f for y, f in year_frames.items() if f.height >= min_year_n}
    years = sorted(usable)
    ratios = {"rolling": [], "fixed": []}

    for i, y in enumerate(years):
        cur = usable[y]
        if i > 0:
            prev = usable[years[i - 1]]
            s, _, _, _ = distribution_drift(prev, cur, cont_cols, bin_cols,
                                            ref_edges=ref_edges, psi_thresh=psi_thresh,
                                            binary_delta_thr=binary_delta_thr)
            t = threshold_fn(min(prev.height, cur.height))
            if t > 0:
                ratios["rolling"].append(s / t)
        others = [usable[o] for o in years if o != y]
        if others:
            pool = others[0]
            for f in others[1:]:
                pool = pool.vstack(f)
            s, _, _, _ = distribution_drift(pool, cur, cont_cols, bin_cols,
                                            ref_edges=ref_edges, psi_thresh=psi_thresh,
                                            binary_delta_thr=binary_delta_thr)
            t = threshold_fn(cur.height)
            if t > 0:
                ratios["fixed"].append(s / t)

    out, info = {}, {"n_reference_years": len(years), "years": years, "ratios": ratios}
    for mode in ("rolling", "fixed"):
        vals = ratios[mode]
        out[mode] = (max(vals) * safety) if vals else 1.0
        info[f"{mode}_active"] = bool(vals)
    return out, info
