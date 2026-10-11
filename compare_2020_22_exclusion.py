#!/usr/bin/env python3
"""
compare_2020_22_exclusion.py

Compares retained vs. excluded cohorts in MIMIC-IV 2020-2022 to assess
potential selection bias across demographics and clinical severity.
"""

import argparse
from pathlib import Path
import numpy as np
import polars as pl


def compute_smd_continuous(mean_1: float, std_1: float, mean_0: float, std_0: float) -> float:
    """Compute Standardized Mean Difference for continuous variables."""
    pooled_sd = np.sqrt((std_1**2 + std_0**2) / 2.0)
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return 0.0
    return float((mean_1 - mean_0) / pooled_sd)


def compute_smd_binary(p_1: float, p_0: float) -> float:
    """Compute Standardized Mean Difference for binary / proportion variables."""
    variance = (p_1 * (1.0 - p_1) + p_0 * (1.0 - p_0)) / 2.0
    if variance <= 0 or np.isnan(variance):
        return 0.0
    return float((p_1 - p_0) / np.sqrt(variance))


def main():
    parser = argparse.ArgumentParser(
        description="Assess 2020-2022 exclusion selection bias."
    )
    parser.add_argument(
        "--base",
        default=Path("."),
        type=Path,
        help="Path to cohort data containing demographics and stays",
    )
    parser.add_argument(
        "--flags",
        required=True,
        type=Path,
        help="Path to stay_flags.parquet containing exclusion indicators",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit base parquet files (overrides --base). Must be the PRE-clean data that "
             "still contains the excluded stays, e.g. the *_final_enriched13Aug.parquet files.",
    )
    parser.add_argument(
        "--out",
        default="exclusion_bias_2020_22.csv",
        type=Path,
        help="Path to save summary CSV",
    )
    args = parser.parse_args()

    # Load stay flags
    print(f"Loading flags from: {args.flags}")
    flags_df = pl.read_parquet(args.flags)

    # Filter to 2020-2022 era using anchor_year_group
    if "anchor_year_group" in flags_df.columns:
        flags_df = flags_df.filter(pl.col("anchor_year_group").cast(pl.Utf8).str.contains("2020"))
    elif "era" in flags_df.columns:
        flags_df = flags_df.filter(pl.col("era").cast(pl.Utf8).str.contains("2020"))
    elif "year" in flags_df.columns:
        flags_df = flags_df.filter(pl.col("year") >= 2020)

    # Determine the exclusion flag column
    if "has_input_data" in flags_df.columns:
        # Explicitly check for 0 to create a proper True/False boolean
        flags_df = flags_df.with_columns(
            (pl.col("has_input_data").cast(pl.Int32) == 0).alias("excluded")
        )
        excl_col = "excluded"
    else:
        excl_col = None
        for cand in ["excluded", "is_excluded", "missing_inputevents"]:
            if cand in flags_df.columns:
                excl_col = cand
                break
        if excl_col is None:
            print(f"Available columns in flags_df: {flags_df.columns}")
            raise ValueError("Could not find an exclusion boolean column in flags parquet.")


    # Load demographic/clinical base table
    base_file = args.base
    if args.files:
        print(f"Loading base features from: {[str(p) for p in args.files]}")
        base_df = pl.concat([pl.read_parquet(p) for p in args.files], how="diagonal_relaxed")
    elif base_file.is_dir():
        parts = [base_file / f"{s}_final_enriched.parquet" for s in ("train", "val", "test")]
        if all(p.exists() for p in parts):
            print(f"Loading base features from: {[str(p) for p in parts]}")
            base_df = pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed")
        else:
            for cand in ["stays.parquet", "demographics.parquet", "cohort.parquet"]:
                if (base_file / cand).exists():
                    base_file = base_file / cand
                    break
            print(f"Loading base features from: {base_file}")
            base_df = pl.read_parquet(base_file)
    else:
        print(f"Loading base features from: {base_file}")
        base_df = pl.read_parquet(base_file)

    # The enriched parquets have one row per stay-hour (6 rows per stay). Reduce to ONE
    # row per stay, otherwise every stay is counted 6 times (N and all statistics).
    if "hrs_from_admit" in base_df.columns:
        base_df = base_df.sort(["stay_id", "hrs_from_admit"]).unique("stay_id", keep="first", maintain_order=True)
    else:
        base_df = base_df.unique("stay_id", keep="first", maintain_order=True)
    assert base_df["stay_id"].n_unique() == base_df.height, "base table is not one row per stay"

    # Join on stay_id / subject_id
    join_key = "stay_id" if "stay_id" in flags_df.columns else "hadm_id"
    df = flags_df.select([join_key, excl_col]).join(
        base_df, on=join_key, how="inner"
    )

    assert df["stay_id"].n_unique() == df.height, "joined table is not one row per stay"
    n_flag = flags_df["stay_id"].n_unique() if "stay_id" in flags_df.columns else -1
    print(f"2020-2022 stays in flags: {n_flag}; found in base table: {df.height}")
    grp = {r[excl_col]: r["len"] for r in df.group_by(excl_col).len().to_dicts()}
    if not grp.get(True) or not grp.get(False):
        raise SystemExit(
            f"STOP: need both excluded and retained stays, got {grp}. The base files do not contain "
            "the excluded stays (they may already be the CLEAN parquets). Pass the pre-clean files "
            "with --files, e.g. the *_final_enriched13Aug.parquet files.")
    total_stays = len(df)
    print(f"\n2020-2022 stays analyzed: {total_stays}")

    # Counts
    counts = df.group_by(excl_col).len()
    print("\nCohort split:")
    print(counts)

    # Candidate columns to inspect
    continuous_candidates = [
        ("age", "Age (years)"),
        ("weight", "Weight (kg)"),
        ("los", "ICU Length of Stay (days)"),
        ("icu_los", "ICU Length of Stay (days)"),
        ("sapsii", "SAPS II Score"),
        ("oasis", "OASIS Score"),
        ("sofa", "SOFA Score"),
    ]

    binary_candidates = [
        ("gender", "Female (%)"),
        ("sex", "Female (%)"),
        ("is_female", "Female (%)"),
        ("mortality", "In-hospital Mortality (%)"),
        ("hospital_expire_flag", "In-hospital Mortality (%)"),
    ]

    results = []
    seen = set()
    n_by_group = {r[excl_col]: r["len"] for r in counts.to_dicts()}

    # 1. Process continuous variables
    for col_name, label in continuous_candidates:
        if col_name in df.columns and label not in seen:
            sub = df.select([excl_col, col_name]).drop_nulls()
            stats = sub.group_by(excl_col).agg([
                pl.col(col_name).mean().alias("mean"),
                pl.col(col_name).std().alias("std"),
                pl.len().alias("count"),
            ])

            excl_row = stats.filter(pl.col(excl_col) == True)
            ret_row = stats.filter(pl.col(excl_col) == False)

            if len(excl_row) > 0 and len(ret_row) > 0:
                m_excl, s_excl = excl_row["mean"][0], excl_row["std"][0]
                m_ret, s_ret = ret_row["mean"][0], ret_row["std"][0]
                smd = compute_smd_continuous(m_excl, s_excl, m_ret, s_ret)
                seen.add(label)

                results.append({
                    "variable": label,
                    "type": "Continuous",
                    "excluded_stat": f"{m_excl:.2f} ± {s_excl:.2f}",
                    "retained_stat": f"{m_ret:.2f} ± {s_ret:.2f}",
                    "smd": round(smd, 4),
                    "balanced_smd_under_0_1": abs(smd) < 0.10,
                })

    # 2. Process binary / categorical variables
    for col_name, label in binary_candidates:
        if col_name in df.columns and label not in seen:
            sub = df.select([excl_col, col_name]).drop_nulls()

            # Handle string flags like 'F'/'M'
            if sub[col_name].dtype == pl.Utf8:
                sub = sub.with_columns(
                    (pl.col(col_name).str.to_uppercase() == "F").alias(col_name)
                )

            stats = sub.group_by(excl_col).agg([
                pl.col(col_name).cast(pl.Float64).mean().alias("prop"),
                pl.len().alias("count"),
            ])

            excl_row = stats.filter(pl.col(excl_col) == True)
            ret_row = stats.filter(pl.col(excl_col) == False)

            if len(excl_row) > 0 and len(ret_row) > 0:
                p_excl = excl_row["prop"][0]
                p_ret = ret_row["prop"][0]
                smd = compute_smd_binary(p_excl, p_ret)
                seen.add(label)

                results.append({
                    "variable": label,
                    "type": "Binary",
                    "excluded_stat": f"{p_excl * 100:.1f}%",
                    "retained_stat": f"{p_ret * 100:.1f}%",
                    "smd": round(smd, 4),
                    "balanced_smd_under_0_1": abs(smd) < 0.10,
                })

    results.insert(0, {"variable": "N (stays)", "type": "Count",
                       "excluded_stat": str(n_by_group.get(True, 0)),
                       "retained_stat": str(n_by_group.get(False, 0)),
                       "smd": None, "balanced_smd_under_0_1": None})
    res_df = pl.DataFrame(results)
    print("\n=== Standardized Mean Differences (SMD) Summary ===")
    print(res_df)

    res_df.write_csv(args.out)
    print(f"\nWrote results to {args.out}")


if __name__ == "__main__":
    main()