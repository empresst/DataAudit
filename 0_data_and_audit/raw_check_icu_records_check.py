"""Raw-table check: share of ICU stays with no inputevents / procedureevents rows, by era.

Raw MIMIC-IV 3.1 icustays + stay-id lists extracted from inputevents, procedureevents
and hosp/emar (hadm-level). Anchor year/group per subject comes from the cohort parquets
(patients.csv is not in the local backup), so stays of subjects outside every cohort
parquet are reported separately as 'no anchor'.
"""
import os
import polars as pl

RAW = os.environ.get("MIMIC_ICUSTAYS_CSV", "./raw_mimic/icu/icustays.csv.gz")
DATA_DIR = os.environ.get("CLEAN_DATA_DIR", "./data")
TRY3_DIR = os.environ.get("RAW_PARQUET_DIR", "./data_preclean")
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")

PARQ = [f"{DATA_DIR}/{s}_final_enriched.parquet" for s in ("train", "val", "test")]
PARQ_TRY3 = [f"{TRY3_DIR}/{s}_final_enriched4.parquet" for s in ("train", "val", "test")]


def ids(path):
    p = os.path.join(SCRATCH_DIR, path) if not os.path.isabs(path) else path
    return pl.read_csv(p, has_header=False, new_columns=["id"], schema={"id": pl.Utf8}) \
             .filter(pl.col("id") != "").with_columns(pl.col("id").cast(pl.Int64))["id"]


inp, proc, emar = ids("input_stays.txt"), ids("proc_stays.txt"), ids("emar_hadm.txt")
icu = pl.read_csv(RAW, try_parse_dates=True)
print(f"icustays rows {icu.height}")

anchor = pl.concat([pl.read_parquet(p, columns=["subject_id", "anchor_year", "anchor_year_group"])
                    for p in PARQ_TRY3]).unique()   # raw anchors only: clean_v2 stores the reconstructed era in anchor_year_group
dup = anchor.group_by("subject_id").len().filter(pl.col("len") > 1).height
print(f"subjects with anchor info {anchor['subject_id'].n_unique()}  (conflicting rows: {dup})")
anchor = anchor.unique("subject_id")

d = (icu.join(anchor, on="subject_id", how="left")
        .with_columns(
            has_input=pl.col("stay_id").is_in(inp.implode()),
            has_proc=pl.col("stay_id").is_in(proc.implode()),
            has_emar=pl.col("hadm_id").is_in(emar.implode()),
            mid=pl.col("anchor_year_group").str.slice(0, 4).cast(pl.Int32) + 1)
        .with_columns(est_year=pl.col("mid") + pl.col("intime").dt.year() - pl.col("anchor_year"))
        .with_columns(era=pl.when(pl.col("est_year").is_null()).then(pl.lit("no anchor"))
                      .when(pl.col("est_year") <= 2010).then(pl.lit("2008-10"))
                      .when(pl.col("est_year") <= 2013).then(pl.lit("2011-13"))
                      .when(pl.col("est_year") <= 2016).then(pl.lit("2014-16"))
                      .when(pl.col("est_year") <= 2019).then(pl.lit("2017-19"))
                      .otherwise(pl.lit("2020-22"))))

agg = [pl.len().alias("stays"),
       (~pl.col("has_input")).mean().alias("no_input"),
       (~pl.col("has_proc")).mean().alias("no_proc"),
       (~pl.col("has_input") & ~pl.col("has_proc")).mean().alias("neither"),
       pl.col("has_emar").mean().alias("hadm_has_emar"),
       (~pl.col("has_input") & ~pl.col("has_proc") & pl.col("has_emar")).sum().alias("neither_but_emar")]
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(160)
print("\nBy reconstructed era (all ICU stays of subjects with anchor info):")
print(d.group_by("era").agg(agg).sort("era"))
print("\nBy raw anchor_year_group:")
print(d.group_by("anchor_year_group").agg(agg).sort("anchor_year_group"))
print("\n2020-22 by first care unit:")
print(d.filter(pl.col("era") == "2020-22").group_by("first_careunit").agg(agg).sort("stays", descending=True))
print("\nStays with neither, by length of stay (2020-22 vs 2017-19):")
print(d.filter(pl.col("era").is_in(["2017-19", "2020-22"]))
        .with_columns(los_bin=pl.col("los").cut([1, 3, 7], labels=["<1d", "1-3d", "3-7d", ">7d"]))
        .group_by("era", "los_bin").agg(agg).sort("era", "los_bin"))
print("\nestimated year (2018-2022) detail:")
print(d.filter(pl.col("est_year").is_between(2017, 2023)).group_by("est_year").agg(agg).sort("est_year"))
