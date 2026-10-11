"""
check_emar_raw.py -- verify the eMAR availability problem directly from raw MIMIC-IV 3.1.

Mirrors script1 exactly (same approx_year reconstruction, same first-6h window,
same event_txt filter, same medication patterns) but needs only 4 tables:
    hosp/patients.csv(.gz), hosp/admissions.csv(.gz), icu/icustays.csv(.gz), hosp/emar.csv(.gz)

Usage:  python check_emar_raw.py /path/to/mimic-iv-3.1
Needs:  pip install duckdb polars
Output: prints the tables below and writes emar_raw_check.csv
"""
import sys
from pathlib import Path
import duckdb
import polars as pl
pl.Config.set_tbl_rows(50); pl.Config.set_tbl_cols(20)

import os
root = Path(_args[0] if _args else os.environ.get("MIMIC_DIR", "./raw_mimic/mimic-iv-3.1"))
OBS_HOURS = 6


def find(rel):
    for p in (root / rel, root / (rel + ".gz")):
        if p.exists():
            return p
    sys.exit(f"missing: {root / rel}[.gz]")


con = duckdb.connect()
for tbl, rel in [("icustays", "icu/icustays.csv"), ("patients", "hosp/patients.csv"),
                 ("admissions", "hosp/admissions.csv"), ("emar", "hosp/emar.csv")]:
    con.execute(f"CREATE OR REPLACE VIEW {tbl} AS SELECT * FROM read_csv_auto('{find(rel)}', sample_size=-1)")

# ── approx_year: identical to script1 Fix 1 ───────────────────────────────────
con.execute("""
CREATE OR REPLACE TABLE stay_year AS
SELECT i.stay_id, i.hadm_id, i.intime,
       LEAST(2022, GREATEST(2008,
           CAST(SUBSTR(pa.anchor_year_group, 1, 4) AS INTEGER) + 1
           + EXTRACT(YEAR FROM i.intime)::INT - pa.anchor_year)) AS approx_year
FROM icustays i
JOIN patients pa  ON i.subject_id = pa.subject_id
JOIN admissions a ON i.hadm_id = a.hadm_id
WHERE pa.anchor_year_group IS NOT NULL AND pa.anchor_year_group != ''
""")

# ── script1's treat_emar, verbatim logic ──────────────────────────────────────
L = "LOWER(e.medication) LIKE"
def any_of(*names):
    return "MAX(CASE WHEN " + " OR ".join(f"{L} '%{n}%'" for n in names) + " THEN 1 ELSE 0 END)"

con.execute(f"""
CREATE OR REPLACE TABLE treat_emar AS
SELECT s.stay_id, s.approx_year,
  {any_of('dexamethasone','hydrocortisone','methylprednisolone','prednisone')} AS early_steroid,
  {any_of('vancomycin','meropenem','piperacillin','cefepime','ceftriaxone','levofloxacin','azithromycin')} AS early_antibiotic,
  {any_of('furosemide','lasix','bumetanide')} AS early_furosemide,
  {any_of('heparin','enoxaparin','lovenox')} AS early_anticoagulant,
  {any_of('labetalol','hydralazine','nicardipine','nitroglycerin')} AS early_antihypertensive,
  {any_of('amiodarone','diltiazem','metoprolol')} AS early_antiarrhythmic,
  COUNT(DISTINCT e.medication) AS n_distinct_meds
FROM stay_year s LEFT JOIN emar e ON e.hadm_id = s.hadm_id
  AND e.charttime >= s.intime AND e.charttime < s.intime + INTERVAL '{OBS_HOURS}' HOUR
  AND e.event_txt NOT IN ('Held','Stopped','Refused','Not Given','Not Given - Loss of Vascular Access')
GROUP BY s.stay_id, s.approx_year
""")

print("=" * 78)
print("1. Stays with any eMAR record in first 6h, and per-flag prevalence, by approx_year")
print("=" * 78)
res = con.execute("""
SELECT approx_year AS year, COUNT(*) AS n_stays,
  ROUND(100*AVG((n_distinct_meds>0)::INT),1) AS pct_any_emar,
  ROUND(AVG(n_distinct_meds),2)              AS mean_n_meds,
  ROUND(100*AVG(early_steroid),1)            AS pct_steroid,
  ROUND(100*AVG(early_antibiotic),1)         AS pct_abx,
  ROUND(100*AVG(early_furosemide),1)         AS pct_furosemide,
  ROUND(100*AVG(early_anticoagulant),1)      AS pct_anticoag,
  ROUND(100*AVG(early_antihypertensive),1)   AS pct_antihtn,
  ROUND(100*AVG(early_antiarrhythmic),1)     AS pct_antiarrhy
FROM treat_emar GROUP BY 1 ORDER BY 1
""").pl()
print(res)
res.write_csv("emar_raw_check.csv")

print("\n" + "=" * 78)
print("2. Pooled: training years (<=2013) vs 2017+")
print("=" * 78)
print(con.execute("""
SELECT CASE WHEN approx_year<=2013 THEN '2008-2013 (train era)'
            WHEN approx_year<=2016 THEN '2014-2016' ELSE '2017-2022' END AS period,
       COUNT(*) AS n_stays, ROUND(100*AVG((n_distinct_meds>0)::INT),2) AS pct_any_emar
FROM treat_emar GROUP BY 1 ORDER BY 1""").pl().__str__())

print("\n" + "=" * 78)
print("3. Control (window-independent): eMAR rows per admission, by admission year")
print("   If this also jumps ~2014-17, it is a source-system change, not the 6h window.")
print("=" * 78)
print(con.execute("""
WITH adm AS (
  SELECT a.hadm_id,
    LEAST(2022, GREATEST(2008, CAST(SUBSTR(pa.anchor_year_group,1,4) AS INTEGER) + 1
        + EXTRACT(YEAR FROM a.admittime)::INT - pa.anchor_year)) AS yr
  FROM admissions a JOIN patients pa USING (subject_id)
  WHERE pa.anchor_year_group IS NOT NULL AND pa.anchor_year_group != ''),
cnt AS (SELECT hadm_id, COUNT(*) AS n FROM emar GROUP BY 1)
SELECT yr AS year, COUNT(*) AS n_admissions,
       ROUND(100*AVG((cnt.n IS NOT NULL)::INT),1) AS pct_adm_with_any_emar,
       ROUND(AVG(COALESCE(cnt.n,0)),1) AS mean_emar_rows
FROM adm LEFT JOIN cnt USING (hadm_id) GROUP BY 1 ORDER BY 1""").pl().__str__())

print("\nCompare section 1 against your audit: expect ~0% through 2012, 1.2% in 2013, ~90-94% from 2017.")
