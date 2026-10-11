"""verify_audit_numbers.py -- recompute the audit numbers quoted in the paper from saved evidence.

Part A (no data needed): arithmetic on values saved in
  ../raw_check_icu_records_check.log   (reconstructed-year audit of raw icustays/inputevents/procedureevents)
  bigquery_audit_results.md            (BigQuery outputs, MIMIC anchor-year groups)
Part B (needs the clean parquets + stay_flags.parquet, committed on branch `main` under 0_data_and_audit/):
  vasopressor label rates under the old and extended ItemID sets, and the check that the clean data
  use the extended definition.

Usage:  python verify_audit_numbers.py [PARQUET_DIR]
"""
import sys

# ---------------- Part A ----------------
print("== A1. 2020-22 reconstructed-year bin (raw_check_icu_records_check.log)")
bin_stays, bin_neither_share = 16745, 0.531980          # era table, row 2020-22
est = {2020: (2776, 0.198847), 2021: (11124, 0.541802), 2022: (2084, 0.752399)}   # est_year detail rows
n_2023 = (702, 1.0)                                       # est_year 2023 row: 702 stays, 100% neither
in_range = sum(n for n, _ in est.values())
neither_range = sum(n * p for n, p in est.values())
after_2022 = bin_stays - in_range
neither_bin = bin_stays * bin_neither_share
print(f"stays with reconstructed year 2020-2022: {in_range}")
print(f"stays in the bin with reconstructed year after 2022: {after_2022}  (of which 2023: {n_2023[0]}, later years: {after_2022 - n_2023[0]})")
print(f"neither-table stays in the bin: {neither_bin:.0f}; in 2020-2022 proper: {neither_range:.0f}; after 2022: {neither_bin - neither_range:.0f}"
      f"  -> all {after_2022} post-2022 stays are missing: {abs((neither_bin - neither_range) - after_2022) < 1.5}")
print(f"share missing, bin: {neither_bin / bin_stays:.3%}; excluding post-2022: {neither_range / in_range:.3%} ({neither_range:.0f} of {in_range})")
print("share lacking any inputevents (bin):", f"{0.533473:.3%}", f"= {0.533473 * bin_stays:.0f} of {bin_stays}")

print("\n== A2. BigQuery anchor-year group 2020-2022")
miss, cov = 6272, 4525
print(f"stays {miss + cov}; missing {miss} = {miss / (miss + cov):.1%}")
print(f"multi-stay admissions: 400 + 591 + 28 = {400 + 591 + 28}; mixed {28 / 1019:.1%}")
print(f"multi-stay patients: 555 + 807 + 167 = {555 + 807 + 167}; mixed {167 / 1529:.1%}")
units = {"Neuro Intermediate": 58.6, "CVICU": 60.0, "MICU": 57.9, "MICU/SICU": 60.1, "TSICU": 54.9, "CCU": 58.4, "SICU": 60.3, "Neuro SICU": 57.9}
print(f"care units with >=400 stays (8 units): no-data share {min(units.values())}-{max(units.values())}%")
los = [56.5, 61.1, 57.3, 56.3]
print(f"length-of-stay bins: {min(los)}-{max(los)}%")

print("\n== A3. Phenylephrine stays, old ID vs old+229630+229632 (BigQuery Q2)")
for era, a, b in [("2008-2010", 4661, 5038), ("2011-2013", 3777, 4020), ("2014-2016", 3834, 4141), ("2017-2019", 1845, 3495), ("2020-2022", 311, 1147)]:
    print(f"{era}: {a} -> {b}  ({b / a - 1:+.1%})")

# ---------------- Part B ----------------
if len(sys.argv) > 1:
    import polars as pl
    d = sys.argv[1].rstrip("/") + "/"
    f = pl.read_parquet(d + "stay_flags.parquet")
    b = pl.concat([pl.read_parquet(d + f"{s}_final_enriched.parquet", columns=["stay_id", "label_vasopressor", "already_vaso"]).unique("stay_id")
                   for s in ("train", "val", "test")])
    j = b.join(f.select("stay_id", "anchor_year_group", "label_vaso_old", "label_vaso_ext", "already_vaso_old", "already_vaso_ext"), on="stay_id")
    c = lambda x: x.cast(pl.Int64)
    print("\n== B. clean parquets vs flags")
    print("flag stays:", f.height, "| clean-parquet stays:", b.height, "| joined:", j.height)
    print("excluded stays (has_input_data = 0) present in clean parquets:",
          j.join(f.filter(pl.col("has_input_data") == 0).select("stay_id"), on="stay_id").height)
    print("label_vasopressor == extended:", (c(j["label_vasopressor"]) == c(j["label_vaso_ext"])).mean(),
          "| == old:", round((c(j["label_vasopressor"]) == c(j["label_vaso_old"])).mean(), 4))
    print("already_vaso == extended:", (c(j["already_vaso"]) == c(j["already_vaso_ext"])).mean(),
          "| == old:", round((c(j["already_vaso"]) == c(j["already_vaso_old"])).mean(), 4))
    t = j.group_by("anchor_year_group").agg(pl.len().alias("stays"), pl.col("label_vaso_old").mean().alias("rate_old"),
                                            pl.col("label_vaso_ext").mean().alias("rate_ext")).sort("anchor_year_group")
    t = t.with_columns((100 * (pl.col("rate_ext") - pl.col("rate_old"))).round(2).alias("diff_points"))
    print(t)
