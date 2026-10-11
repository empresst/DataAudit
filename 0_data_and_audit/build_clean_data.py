"""
build_clean_data.py — rebuild the clean-v2 parquets (the data behind every result in the paper).

The clean parquets (ml4h_clean_pilot/data/{train,val,test}_final_enriched.parquet) are the
Kaggle-built parquets (*_final_enriched13Aug.parquet, reached through ml4h_corrected_noemar/data)
with two changes, both from the 2026-10-08 audit:

  1. ICU-module availability. Keep only stays that have at least one row in icu/inputevents OR
     icu/procedureevents (a data-availability criterion; no outcome information is used).
     In 2020-22 about half of all ICU stays have no such rows, so their vasopressor label, gate
     and ICU treatment features are structurally zero.
  2. Vasopressor item IDs. From 2017 phenylephrine (and one epinephrine variant) is charted under
     new item IDs. `label_vasopressor` and `already_vaso` are replaced by the versions computed
     with the extended ID set (label_vaso_ext / already_vaso_ext in stay_flags.parquet).

NOT part of the clean data: eMAR features are not rebuilt (constants.py excludes them); the central-line
item-ID change (224267 -> 229526/229516) is NOT corrected, so has_central_line_obs undercounts after 2017.

Inputs
  --base   directory with {train,val,test}_final_enriched.parquet (default: ml4h_corrected_noemar/data)
  --flags  stay_flags.parquet written by make_flags.py (has_input_data, label_vaso_ext, already_vaso_ext)
Output
  --out    directory for the rebuilt parquets (default ./clean_data_rebuilt; never overwrites --reference)
Check
  --reference  existing clean parquets; if present, every column of every row must match exactly
               (exit code 1 otherwise). Pass --reference "" to skip.

Run from anywhere:  python build_clean_data.py [--out DIR]
"""
import argparse
import sys
from pathlib import Path

import polars as pl

import os

BASE_DIR = Path(os.environ.get("BASE_DIR", "."))
SPLITS = ("train", "val", "test")


def same_column(x: pl.Series, y: pl.Series) -> bool:
    """Equal values, nulls matching nulls, NaNs matching NaNs, floats to 1e-9."""
    both_null = x.is_null() & y.is_null()
    if x.dtype in (pl.Float32, pl.Float64):
        ok = ((x - y).abs() < 1e-9).fill_null(False) | both_null | (x.is_nan() & y.is_nan()).fill_null(False)
    else:
        ok = (x == y).fill_null(False) | both_null
    return bool(ok.all())


def build_split(base: pl.DataFrame, flags: pl.DataFrame, name: str) -> pl.DataFrame:
    j = base.join(flags, on="stay_id", how="left")
    n_noflag = j.filter(pl.col("has_input_data").is_null())["stay_id"].n_unique()
    if n_noflag:
        sys.exit(f"STOP: {name}: {n_noflag} stays have no row in stay_flags.parquet "
                 f"(they would be dropped silently); rerun make_flags.py")
    kept = j.filter(pl.col("has_input_data") == 1)
    out = kept.with_columns(
        pl.col("label_vaso_ext").cast(base["label_vasopressor"].dtype).alias("label_vasopressor"),
        pl.col("already_vaso_ext").cast(base["already_vaso"].dtype).alias("already_vaso"),
    ).select(base.columns)
    n0, n1 = base["stay_id"].n_unique(), out["stay_id"].n_unique()
    chg = kept.filter(pl.col("label_vasopressor").cast(pl.Int64) != pl.col("label_vaso_ext"))["stay_id"].n_unique()
    print(f"  {name:5s} stays {n0:6d} -> {n1:6d} (dropped {n0 - n1}); "
          f"vasopressor label changed on {chg} stays")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("PRECLEAN_DATA_DIR", str(BASE_DIR / "data")),
                    help="Directory containing pre-clean enriched parquets")
    ap.add_argument("--flags", default=os.environ.get("STAY_FLAGS_PATH", str(BASE_DIR / "stay_flags.parquet")),
                    help="Path to stay_flags.parquet")
    ap.add_argument("--out", default=os.environ.get("CLEAN_DATA_OUT", "clean_data_rebuilt"),
                    help="Output directory for rebuilt clean parquets")
    ap.add_argument("--reference", default=os.environ.get("REFERENCE_DATA_DIR", None),
                    help="Optional reference directory to verify against")
    a = ap.parse_args()

    out_dir = Path(a.out)
    ref_dir = Path(a.reference) if a.reference else None
    if ref_dir and out_dir.resolve() == ref_dir.resolve():
        sys.exit("STOP: --out must differ from --reference")
    out_dir.mkdir(parents=True, exist_ok=True)

    flags = pl.read_parquet(a.flags).select("stay_id", "label_vaso_ext", "already_vaso_ext", "has_input_data")
    if flags["stay_id"].n_unique() != flags.height:
        sys.exit("STOP: stay_flags.parquet has duplicate stay_id rows")
    print(f"flags: {flags.height} stays; has_input_data share {flags['has_input_data'].mean():.3f}")

    mismatches = []
    for s in SPLITS:
        base = pl.read_parquet(Path(a.base) / f"{s}_final_enriched.parquet")
        built = build_split(base, flags, s)
        built.write_parquet(out_dir / f"{s}_final_enriched.parquet")
        if ref_dir:
            ref = pl.read_parquet(ref_dir / f"{s}_final_enriched.parquet")
            if built.shape != ref.shape or built.columns != ref.columns:
                mismatches.append(f"{s}: shape/columns {built.shape} vs {ref.shape}")
                continue
            keys = ["stay_id", "hrs_from_admit"]
            b, r = built.sort(keys), ref.sort(keys)
            bad = [c for c in ref.columns if not same_column(b[c], r[c])]
            if bad:
                mismatches.append(f"{s}: columns differ from reference: {bad[:8]}")
            else:
                print(f"  {s:5s} identical to reference ({ref.shape[0]} rows x {ref.shape[1]} columns)")
    if mismatches:
        print("\nMISMATCH vs reference:")
        for m in mismatches:
            print("  " + m)
        sys.exit(1)
    print(f"\nSaved -> {out_dir}/" + (f"  (verified identical to {ref_dir})" if ref_dir else "  (not verified)"))


if __name__ == "__main__":
    main()
