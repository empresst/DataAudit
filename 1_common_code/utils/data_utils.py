"""
data_utils.py — split loading, train-only normalization, and datasets.

Changes from the previous version:
  * Hyperparameters are imported from utils.constants, not redeclared here.
  * Dataset construction is O(N) via partition_by instead of O(N^2) via a
    per-stay .filter() inside a Python loop. On ~40k stays this was the single
    largest cost in the pipeline, and eval_on_split rebuilt it on every call.
  * Datasets carry per-stay METADATA (already_vaso / already_vent / stay_id)
    as aligned numpy arrays. These are NOT returned by __getitem__, so the
    model can never see them; they exist so evaluation can be stratified into
    incident vs prevalent cohorts.
  * Short sequences are counted and reported. script1 emits a dense h0..h5 grid,
    so any padding at all means an upstream regression — it is surfaced rather
    than silently zero-filled.
"""
import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from utils.constants import (
    BINARY_COLS, METADATA_COLS, MASKS_AS_CARE_PROCESS, MASK_COLS, _TIME_DELTA_COLS,
    SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM,
    LSTM_LAYERS, BATCH_SIZE, DROPOUT,
    LR_INIT, LR_ADAPT, EPOCHS, ADAPT_EPOCHS,
    PATIENCE, ADAPT_PATIENCE, BUFFER_SIZE, PSI_THRESH,
    LABEL_COLS,
)

# Filename stem of the parquet files written by script1.
SPLIT_SUFFIX = "_final_enriched"


def load_enriched_split(base_path: Path, split_name: str, seq_features: list,
                        treat_features: list, suffix: str = SPLIT_SUFFIX) -> pl.DataFrame:
    """Load a parquet split and ensure all required columns and encodings exist."""
    path = base_path / f"{split_name}{suffix}.parquet"
    if not path.exists():
        # Tolerate versioned dataset stems (…_enriched4.parquet) without silently
        # loading the wrong preprocessing run.
        candidates = sorted(base_path.glob(f"{split_name}{suffix}*.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"No parquet for split '{split_name}' under {base_path} "
                f"(looked for {path.name} and {split_name}{suffix}*.parquet)")
        if len(candidates) > 1:
            raise FileNotFoundError(
                f"Ambiguous parquet stems for split '{split_name}': "
                f"{[c.name for c in candidates]}. Pin one via load_enriched_split(suffix=...).")
        path = candidates[0]
    df = pl.read_parquet(path)

    if "gender" in df.columns:
        df = df.with_columns((pl.col("gender") == "M").cast(pl.Float32).alias("gender_M"))

    # In-hospital mortality is carried through script1 as evaluation metadata.
    # Exposing it as `label_mortality` lets LABEL_COLS select a BIOLOGICAL target
    # without re-extracting anything. Unlike vasopressor and intubation it is not
    # a clinician's decision, so P(y | physiology) is not non-stationary by
    # construction — which is the whole point of running it.
    if "mortality" in df.columns:
        df = df.with_columns(pl.col("mortality").cast(pl.Float32).alias("label_mortality"))

    for eth in ["WHITE", "BLACK", "HISPANIC", "ASIAN"]:
        if "ethnicity" in df.columns:
            df = df.with_columns((pl.col("ethnicity") == eth).cast(pl.Float32).alias(f"eth_{eth}"))

    # Care-process aggregates, derived here so every script that loads a split
    # gets them without re-running script1. Each `x_mask` becomes `x_obs_rate` =
    # fraction of h0..h5 in which x was charted; each `x_time_delta` becomes its
    # per-stay mean. These are stay-level, so they join onto every hour row and
    # the treatment stream reads them from row 0 like any other static feature.
    if MASKS_AS_CARE_PROCESS:
        aggs = ([pl.col(c).cast(pl.Float64).mean().alias(f"{c[:-5]}_obs_rate")
                 for c in MASK_COLS if c in df.columns]
                + [pl.col(c).cast(pl.Float64).mean().alias(f"{c}_mean")
                   for c in _TIME_DELTA_COLS if c in df.columns])
        if aggs:
            df = df.join(df.group_by("stay_id").agg(aggs), on="stay_id", how="left")

    for c in seq_features + treat_features:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(c))

    # Metadata is optional for backwards compatibility with older parquets, but
    # its absence disables incident/prevalent stratification, so say so loudly.
    missing_meta = [c for c in METADATA_COLS if c not in df.columns]
    if missing_meta:
        print(f"  ⚠ {split_name}: missing metadata columns {missing_meta} — "
              f"incident/prevalent stratification will be unavailable. "
              f"Re-run script1 to emit them.")
        for c in missing_meta:
            df = df.with_columns(pl.lit(-1).cast(pl.Int8).alias(c))

    return df


def calculate_train_stats(train_df: pl.DataFrame, norm_cols: list) -> dict:
    """Z-score statistics computed exclusively from the training split."""
    train_stats = {}
    for c in norm_cols:
        if c in train_df.columns:
            vals = train_df[c].cast(pl.Float64)
            mu, sd = vals.mean(), vals.std()
            # Zero-variance rule, extended to NEAR-constant features (correction,
            # 2026-10-07). A feature that takes one value in >= 99.9% of training rows
            # has a spread set by a handful of stays; dividing by it sends later values
            # to hundreds of SDs. n_distinct_meds (eMAR, rolled out in MIMIC-IV during
            # 2011-2013) had sd 0.0367 here and reached the network at z = 136-735 in
            # the test eras. It is the only feature this rule changes.
            nn_ = vals.drop_nulls()
            mode_share = (nn_.value_counts(sort=True)["count"][0] / len(nn_)) if len(nn_) else 1.0
            if sd is None or sd == 0 or sd != sd or mode_share >= 0.999:
                sd = 1.0
            train_stats[c] = (mu if mu is not None else 0.0, sd)
    return train_stats


def normalize(df, stats):
    exprs = []
    for c, (mu, sd) in stats.items():
        if c in df.columns:
            if c.endswith("_mask") or c in BINARY_COLS:
                exprs.append(pl.col(c).cast(pl.Float32))
            else:
                exprs.append(((pl.col(c).cast(pl.Float64) - mu) / sd).cast(pl.Float32).alias(c))
    return df.with_columns(exprs)


# ── DATASETS ───────────────────────────────────────────────────────────────────
def _partition_stays(df: pl.DataFrame):
    """Sorted per-stay partitions. O(N) — replaces the per-stay .filter() loop."""
    stays = df.sort(["stay_id", "hrs_from_admit"])
    return stays.partition_by("stay_id", maintain_order=True)


def _pad_or_trim(seq: np.ndarray, seq_len: int):
    """Returns (seq, was_padded). script1 guarantees a dense grid, so padding
    should never fire; the flag lets the caller assert that."""
    if seq.shape[0] < seq_len:
        pad = np.zeros((seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
        return np.vstack([seq, pad]), True
    return seq[:seq_len], False


def _dense_view(df: pl.DataFrame, seq_len: int):
    """
    Sorted frame plus a one-row-per-stay frame, or None when the grid is ragged.

    script1 emits a dense seq_len grid, and when that holds every per-stay value
    can be read with vectorised column operations instead of a Python loop over
    ~24,000 single-stay DataFrames. That loop, not the GPU, dominates runtime:
    each construction costs on the order of 250,000 polars scalar extractions,
    and the adaptation scripts build datasets once per era PER SEED.

    Ordering is identical to the partition path — both sort by
    (stay_id, hrs_from_admit) first — so the fast path produces bit-identical
    arrays, not merely equivalent ones. It is refused rather than approximated
    if any stay is short or long, and the caller falls back to the loop.
    """
    stays = df.sort(["stay_id", "hrs_from_admit"])
    if stays.height == 0 or stays.height % seq_len != 0:
        return None, None
    counts = stays.group_by("stay_id").len()["len"].to_numpy()
    if not (counts == seq_len).all():
        return None, None
    return stays, stays.gather_every(seq_len)


class _BaseICUDataset(Dataset):
    """Shared metadata extraction for both dataset flavours."""

    def _collect_meta_fast(self, first, label_cols):
        n = first.height
        self.stay_ids = [int(v) for v in first["stay_id"].to_list()]
        self.groups = (first["anchor_year_group"].to_list()
                       if "anchor_year_group" in first.columns else ["UNK"] * n)
        self.subject_ids = (first["subject_id"].to_numpy().astype(np.int64)
                            if "subject_id" in first.columns
                            else np.full(n, -1, dtype=np.int64))
        self.meta = {c: first[c].to_numpy().astype(np.float32)
                     for c in METADATA_COLS if c in first.columns}
        self.labels = first.select(label_cols).to_numpy().astype(np.float32)

    def _collect_meta(self, parts, label_cols):
        self.stay_ids = [int(p["stay_id"][0]) for p in parts]
        self.groups = [
            p["anchor_year_group"][0] if "anchor_year_group" in p.columns else "UNK"
            for p in parts
        ]
        self.subject_ids = np.array(
            [int(p["subject_id"][0]) if "subject_id" in p.columns else -1 for p in parts],
            dtype=np.int64)
        self.meta = {}
        for c in METADATA_COLS:
            if c in parts[0].columns:
                self.meta[c] = np.array([float(p[c][0]) for p in parts], dtype=np.float32)
        self.labels = np.stack(
            [np.array(p.select(label_cols).row(0), dtype=np.float32) for p in parts])

    def _report_padding(self, n_padded, name):
        if n_padded:
            print(f"  ⚠ {name}: {n_padded}/{len(self.stay_ids)} stays shorter than "
                  f"seq_len and were zero-padded. script1 emits a dense grid, so "
                  f"this indicates an upstream regression — the LSTM reads h[-1] "
                  f"and padded steps corrupt it.")

    def __len__(self):
        return len(self.stay_ids)


class ICUDataset(_BaseICUDataset):
    """Two-stream dataset: (sequence, treatment vector, labels)."""

    def __init__(self, df, seq_features, treat_features, label_cols, seq_len=SEQ_LEN):
        self.seq_len, self.label_cols = seq_len, label_cols
        self.seq_cols = [c for c in seq_features if c in df.columns]
        self.treat_cols = [c for c in treat_features if c in df.columns]

        stays, first = _dense_view(df, seq_len)
        if first is not None:                                   # vectorised path
            self.seq_data = (stays.select(self.seq_cols).to_numpy()
                             .astype(np.float32)
                             .reshape(first.height, seq_len, len(self.seq_cols)))
            self.treat_data = first.select(self.treat_cols).to_numpy().astype(np.float32)
            self._collect_meta_fast(first, label_cols)
            self._report_padding(0, "ICUDataset")
            return

        parts = _partition_stays(df)
        seqs, treats, n_padded = [], [], 0
        for p in parts:
            seq, padded = _pad_or_trim(
                p.select(self.seq_cols).to_numpy().astype(np.float32), seq_len)
            n_padded += padded
            seqs.append(seq)
            treats.append(np.array(p.select(self.treat_cols).row(0), dtype=np.float32))

        self.seq_data = np.stack(seqs)
        self.treat_data = np.stack(treats)
        self._collect_meta(parts, label_cols)
        self._report_padding(n_padded, "ICUDataset")

    def __getitem__(self, idx):
        return (torch.from_numpy(self.seq_data[idx]),
                torch.from_numpy(self.treat_data[idx]),
                torch.from_numpy(self.labels[idx]))


class SingleStreamDataset(_BaseICUDataset):
    """
    Monolithic dataset for Run D: treatment features are broadcast across time
    and concatenated onto the physiological sequence, so one LSTM sees all
    (seq_dim + treat_dim) channels at every timestep.
    """

    def __init__(self, df, seq_features, treat_features, label_cols, seq_len=SEQ_LEN):
        self.seq_len, self.label_cols = seq_len, label_cols
        self.seq_cols = [c for c in seq_features if c in df.columns]
        self.treat_cols = [c for c in treat_features if c in df.columns]

        parts = _partition_stays(df)
        combined, n_padded = [], 0
        for p in parts:
            seq, padded = _pad_or_trim(
                p.select(self.seq_cols).to_numpy().astype(np.float32), seq_len)
            n_padded += padded
            treat_vec = np.array(p.select(self.treat_cols).row(0), dtype=np.float32)
            combined.append(np.concatenate([seq, np.tile(treat_vec, (seq_len, 1))], axis=1))

        self.combined_data = np.stack(combined)
        self._collect_meta(parts, label_cols)
        self._report_padding(n_padded, "SingleStreamDataset")

    def __getitem__(self, idx):
        return (torch.from_numpy(self.combined_data[idx]),
                torch.from_numpy(self.labels[idx]))
