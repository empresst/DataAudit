"""
check_unlinked_rows.py -- are the 'missing' input/procedure records really absent, or present but
not linked to the ICU stay?

For the 2020-2022 anchor-year stays with no inputevents/procedureevents rows, count rows in each
table that belong to the same patient or admission and fall inside the stay's time window but carry
a NULL or different stay_id. Aggregate counts only.

  MIMIC_DIR=/kaggle/input/datasets/fatematamanna/mimic4/mimic-iv-3.1 python check_unlinked_rows.py
"""
import os
import sys
from pathlib import Path

import duckdb

MIMIC = Path(os.environ.get("MIMIC_DIR", "./data/mimic-iv-3.1"))


def path(rel):
    for ext in ("", ".gz"):
        p = MIMIC / f"{rel}{ext}"
        if p.exists():
            return p
    sys.exit(f"missing {MIMIC}/{rel}(.gz) -- set MIMIC_DIR")


con = duckdb.connect()
con.execute(f"CREATE VIEW icustays AS SELECT * FROM read_csv_auto('{path('icu/icustays.csv')}', header=true)")
con.execute(f"CREATE VIEW patients AS SELECT * FROM read_csv_auto('{path('hosp/patients.csv')}', header=true)")
for t in ("inputevents", "procedureevents"):
    con.execute(f"""CREATE TABLE {t} AS SELECT
        TRY_CAST(subject_id AS BIGINT) subject_id, TRY_CAST(hadm_id AS BIGINT) hadm_id,
        TRY_CAST(stay_id AS BIGINT) stay_id, TRY_CAST(starttime AS TIMESTAMP) starttime
        FROM read_csv('{path(f'icu/{t}.csv')}', header=true, all_varchar=true)""")

con.execute("""
CREATE TABLE miss AS
SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime
FROM icustays i JOIN patients p USING (subject_id)
WHERE p.anchor_year_group = '2020 - 2022'
  AND i.stay_id NOT IN (SELECT stay_id FROM inputevents WHERE stay_id IS NOT NULL)
  AND i.stay_id NOT IN (SELECT stay_id FROM procedureevents WHERE stay_id IS NOT NULL)
""")
n_miss = con.execute("SELECT COUNT(*) FROM miss").fetchone()[0]
print(f"2020-2022 stays with no inputevents/procedureevents rows: {n_miss}  (expected 6272)\n")

def n(v):  # aggregate-only output: hide counts 1-9
    return "<10" if 0 < v < 10 else f"{v:,}"


fails = 0
for t in ("inputevents", "procedureevents"):
    total, null_stay, orphan = con.execute(f"""
        SELECT COUNT(*), COUNT(*) FILTER (WHERE stay_id IS NULL),
               COUNT(*) FILTER (WHERE stay_id IS NOT NULL AND stay_id NOT IN (SELECT stay_id FROM icustays))
        FROM {t}""").fetchone()
    by_subj, rows_subj = con.execute(f"""
        SELECT COUNT(DISTINCT m.stay_id), COUNT(*)
        FROM miss m JOIN {t} r ON r.subject_id = m.subject_id
         AND r.starttime BETWEEN m.intime AND m.outtime""").fetchone()
    by_hadm = con.execute(f"""
        SELECT COUNT(DISTINCT m.stay_id)
        FROM miss m JOIN {t} r ON r.hadm_id = m.hadm_id
         AND r.starttime BETWEEN m.intime AND m.outtime""").fetchone()[0]
    print(f"== {t}")
    print(f"   rows in table: {n(total)}; NULL stay_id: {n(null_stay)}; stay_id not in icustays: {n(orphan)}")
    print(f"   stays without records that have same-patient rows inside their window: "
          f"{n(by_subj)} ({n(rows_subj)} rows)")
    print(f"   stays without records that have same-admission rows inside their window: {n(by_hadm)}")
    share = 1 - by_subj / n_miss
    ok = null_stay == 0 and orphan == 0 and share >= 0.99
    fails += not ok
    print(f"   [{'PASS' if ok else 'FAIL'}] no unlinked {t} rows; {share:.1%} of stays without records have "
          f"no {t} rows of any stay inside their window\n")

print("RESULT:", "no unlinked or mislinked records found" if fails == 0 else "see FAIL lines above")
