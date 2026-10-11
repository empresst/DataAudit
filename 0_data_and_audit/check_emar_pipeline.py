"""
check_emar_pipeline.py -- is the eMAR gap a SOURCE property or a PREPROCESSING mistake?

Run after check_emar_raw.py.  Tables: icustays, patients, admissions, emar, prescriptions.
Usage: python check_emar_pipeline.py [/path/to/mimic-iv-3.1]     (needs duckdb, polars)

Each section ends with a VERDICT HINT telling you what result means "bug" vs "source".
"""
import sys
from pathlib import Path
import duckdb
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_str_lengths(40)

import os
root = Path(_args[0] if _args else os.environ.get("MIMIC_DIR", "./raw_mimic/mimic-iv-3.1"))
H = 6
def find(rel):
    for p in (root / rel, root / (rel + ".gz")):
        if p.exists(): return p
    sys.exit(f"missing: {root / rel}[.gz]")

con = duckdb.connect()
for t, rel in [("icustays","icu/icustays.csv"),("patients","hosp/patients.csv"),("admissions","hosp/admissions.csv"),
               ("emar","hosp/emar.csv"),("prescriptions","hosp/prescriptions.csv")]:
    con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_csv_auto('{find(rel)}', sample_size=-1)")
q = lambda s: con.execute(s).pl()
def show(title, df, hint=None):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78); print(df)
    if hint: print("VERDICT HINT:", hint)

# stay -> year (same as script1) + official 3-year group
con.execute("""
CREATE OR REPLACE TABLE sy AS
SELECT i.stay_id, i.hadm_id, i.intime, i.outtime, pa.anchor_year_group AS raw_group,
  LEAST(2022, GREATEST(2008, CAST(SUBSTR(pa.anchor_year_group,1,4) AS INTEGER)+1
        + EXTRACT(YEAR FROM i.intime)::INT - pa.anchor_year)) AS yr
FROM icustays i JOIN patients pa USING (subject_id) JOIN admissions a ON i.hadm_id=a.hadm_id
WHERE pa.anchor_year_group IS NOT NULL AND pa.anchor_year_group != ''""")

# A. integrity of emar rows that script1 silently drops
show("A. Rows script1's join/filter could silently drop (share of ALL emar rows)",
 q("""SELECT COUNT(*) AS n_rows,
      ROUND(100*AVG((hadm_id IS NULL)::INT),2)     AS pct_null_hadm,
      ROUND(100*AVG((event_txt IS NULL)::INT),2)   AS pct_null_event_txt,
      ROUND(100*AVG((medication IS NULL)::INT),2)  AS pct_null_medication,
      ROUND(100*AVG((charttime IS NULL)::INT),2)   AS pct_null_charttime,
      ROUND(100*AVG((event_txt IN ('Held','Stopped','Refused','Not Given','Not Given - Loss of Vascular Access'))::INT),2) AS pct_excluded_by_filter
    FROM emar"""),
 "pct_null_hadm / pct_null_event_txt > a few % => rows lost by the join or by NOT IN (NULL never passes NOT IN). Small => not the cause.")
show("A2. event_txt values", q("SELECT event_txt, COUNT(*) n FROM emar GROUP BY 1 ORDER BY 2 DESC LIMIT 25"))

# B. funnel: where do zeros come from, by year
show("B. Funnel by reconstructed year: where does a stay lose its eMAR signal?",
 q(f"""
 WITH hadm_any AS (SELECT DISTINCT hadm_id FROM emar),
 stay_any AS (SELECT DISTINCT s.stay_id FROM sy s JOIN emar e ON e.hadm_id=s.hadm_id
              WHERE e.charttime >= s.intime AND e.charttime <= s.outtime),
 stay_6h AS (SELECT DISTINCT s.stay_id FROM sy s JOIN emar e ON e.hadm_id=s.hadm_id
              WHERE e.charttime >= s.intime AND e.charttime < s.intime + INTERVAL '{H}' HOUR),
 stay_6h_f AS (SELECT DISTINCT s.stay_id FROM sy s JOIN emar e ON e.hadm_id=s.hadm_id
              WHERE e.charttime >= s.intime AND e.charttime < s.intime + INTERVAL '{H}' HOUR
                AND e.event_txt NOT IN ('Held','Stopped','Refused','Not Given','Not Given - Loss of Vascular Access'))
 SELECT s.yr AS year, COUNT(*) AS n_stays,
   ROUND(100*AVG((h.hadm_id IS NOT NULL)::INT),1)  AS pct_hadm_has_any_emar,
   ROUND(100*AVG((a.stay_id IS NOT NULL)::INT),1)  AS pct_emar_during_icu_stay,
   ROUND(100*AVG((b.stay_id IS NOT NULL)::INT),1)  AS pct_emar_first6h,
   ROUND(100*AVG((c.stay_id IS NOT NULL)::INT),1)  AS pct_first6h_after_filter
 FROM sy s LEFT JOIN hadm_any h ON h.hadm_id=s.hadm_id LEFT JOIN stay_any a ON a.stay_id=s.stay_id
   LEFT JOIN stay_6h b ON b.stay_id=s.stay_id LEFT JOIN stay_6h_f c ON c.stay_id=s.stay_id
 GROUP BY 1 ORDER BY 1"""),
 "If pct_hadm_has_any_emar is already ~0 in early years, the data simply isn't in the table (SOURCE). "
 "If it is high but pct_first6h collapses, the window/join/filter is the culprit (BUG).")

# C. official grouping, independent of my year reconstruction
show("C. Same, by OFFICIAL anchor_year_group (independent of approx_year reconstruction)",
 q(f"""SELECT s.raw_group AS anchor_year_group, COUNT(*) AS n_stays,
   ROUND(100*AVG((EXISTS (SELECT 1 FROM emar e WHERE e.hadm_id=s.hadm_id
        AND e.charttime >= s.intime AND e.charttime < s.intime + INTERVAL '{H}' HOUR))::INT),1) AS pct_emar_first6h
   FROM sy s GROUP BY 1 ORDER BY 1"""),
 "Jump should line up with the later official groups. If it only appears under approx_year, suspect the year reconstruction.")

# D. positive control: prescriptions exist in all years?
show("D. Positive control -- prescriptions table, same window (should NOT have the gap)",
 q(f"""SELECT s.yr AS year, COUNT(*) AS n_stays,
   ROUND(100*AVG((EXISTS (SELECT 1 FROM prescriptions p WHERE p.hadm_id=s.hadm_id
        AND p.starttime >= s.intime AND p.starttime < s.intime + INTERVAL '{H}' HOUR))::INT),1) AS pct_rx_first6h,
   ROUND(100*AVG((EXISTS (SELECT 1 FROM prescriptions p WHERE p.hadm_id=s.hadm_id))::INT),1) AS pct_hadm_has_any_rx
   FROM sy s GROUP BY 1 ORDER BY 1"""),
 "Rx flat across years while eMAR jumps => the window/join code is fine and the gap is eMAR-specific (SOURCE).")

# E. do the medication patterns still match across eras? (name drift)
show("E. Among stays WITH eMAR in first 6h: flag prevalence by era (a flag stuck at 0 => pattern mismatch)",
 q(f"""WITH x AS (SELECT s.stay_id, s.yr, LOWER(e.medication) m FROM sy s JOIN emar e ON e.hadm_id=s.hadm_id
        AND e.charttime >= s.intime AND e.charttime < s.intime + INTERVAL '{H}' HOUR
        AND e.event_txt NOT IN ('Held','Stopped','Refused','Not Given','Not Given - Loss of Vascular Access'))
   SELECT CASE WHEN yr<=2013 THEN '2008-13' WHEN yr<=2016 THEN '2014-16' ELSE '2017-22' END AS era,
     COUNT(DISTINCT stay_id) AS stays_with_emar,
     ROUND(100*COUNT(DISTINCT CASE WHEN m LIKE '%heparin%' OR m LIKE '%enoxaparin%' OR m LIKE '%lovenox%' THEN stay_id END)/COUNT(DISTINCT stay_id),1) AS pct_anticoag,
     ROUND(100*COUNT(DISTINCT CASE WHEN m LIKE '%furosemide%' OR m LIKE '%lasix%' OR m LIKE '%bumetanide%' THEN stay_id END)/COUNT(DISTINCT stay_id),1) AS pct_furosemide,
     ROUND(100*COUNT(DISTINCT CASE WHEN m LIKE '%vancomycin%' OR m LIKE '%meropenem%' OR m LIKE '%piperacillin%' OR m LIKE '%cefepime%' OR m LIKE '%ceftriaxone%' THEN stay_id END)/COUNT(DISTINCT stay_id),1) AS pct_abx,
     ROUND(100*COUNT(DISTINCT CASE WHEN m LIKE '%metoprolol%' OR m LIKE '%amiodarone%' OR m LIKE '%diltiazem%' THEN stay_id END)/COUNT(DISTINCT stay_id),1) AS pct_antiarr
   FROM x GROUP BY 1 ORDER BY 1"""),
 "Similar prevalences across eras => patterns are fine; the 0s come from missing rows, not name mismatch.")

# F. raw emar time coverage (shifted dates)
show("F. eMAR rows by shifted charttime year (raw table, no joins)",
 q("SELECT EXTRACT(YEAR FROM charttime)::INT AS shifted_year, COUNT(*) AS n_rows, COUNT(DISTINCT hadm_id) AS n_hadm FROM emar GROUP BY 1 ORDER BY 1"),
 "Shifted years are per-patient, so this only shows overall scale; use B/C for the real timeline.")

print("""
HOW TO READ THIS
  SOURCE (not your mistake): B shows pct_hadm_has_any_emar ~0 early, D flat, E similar across eras, A small.
  BUG in your pipeline:      B has high pct_hadm_has_any_emar early but ~0 in first6h, or A shows big null hadm_id/event_txt,
                             or E has a flag stuck at 0 in later eras.
  Either way the NN/XGB difference in your audit is explained by the 0 -> ~93% jump, not by prescribing behaviour.
""")
