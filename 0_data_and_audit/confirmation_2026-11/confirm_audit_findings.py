"""
confirm_audit_findings.py -- re-derive the MIMIC-IV audit claims from the raw CSVs with DuckDB
(same table layout as upstream_kaggle/script1_preprocessing.py) and check them automatically.

Each check prints PASS / FAIL / INFO and the paper claim it tests. The run is written to
confirm_audit_findings.log next to this script.

  MIMIC_DIR=./data/mimic-iv-3.1 python confirm_audit_findings.py

Expected numbers come from bigquery_audit_results.md (BigQuery, anchor_year_group), so a PASS here
also means two independent engines (BigQuery and DuckDB-on-CSV) give the same answer.
"""
import os
import sys
from pathlib import Path

import duckdb

MIMIC = Path(os.environ.get("MIMIC_DIR", "./data/mimic-iv-3.1"))
LOG = Path(__file__).resolve().parent / "confirm_audit_findings.log"
out = open(LOG, "w")
results = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")
    out.flush()


def check(name, ok, detail):
    tag = "PASS" if ok else "FAIL"
    results.append((tag, name))
    say(f"[{tag}] {name}\n       {detail}")


def info(name, detail):
    results.append(("INFO", name))
    say(f"[INFO] {name}\n       {detail}")


def path(rel):
    for ext in ("", ".gz"):
        p = MIMIC / f"{rel}{ext}"
        if p.exists():
            return p
    return None


con = duckdb.connect()
con.execute("SET preserve_insertion_order = false")

# Small tables: auto-detected types. Large tables: read as text, keep only the columns needed.
small = {"icustays": "icu/icustays.csv", "patients": "hosp/patients.csv",
         "admissions": "hosp/admissions.csv", "d_items": "icu/d_items.csv"}
for t, rel in small.items():
    p = path(rel)
    if p is None:
        sys.exit(f"missing {MIMIC}/{rel}(.gz) -- set MIMIC_DIR")
    con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_csv_auto('{p}', header=true)")

big = {  # table -> (file, select list)
    "inputevents": ("icu/inputevents.csv", "TRY_CAST(stay_id AS BIGINT) stay_id, TRY_CAST(itemid AS INT) itemid"),
    "procedureevents": ("icu/procedureevents.csv", "TRY_CAST(stay_id AS BIGINT) stay_id, TRY_CAST(itemid AS INT) itemid"),
    "chartevents": ("icu/chartevents.csv", "TRY_CAST(stay_id AS BIGINT) stay_id, TRY_CAST(itemid AS INT) itemid"),
    "outputevents": ("icu/outputevents.csv", "TRY_CAST(stay_id AS BIGINT) stay_id"),
    "datetimeevents": ("icu/datetimeevents.csv", "TRY_CAST(stay_id AS BIGINT) stay_id"),
    "ingredientevents": ("icu/ingredientevents.csv", "TRY_CAST(stay_id AS BIGINT) stay_id"),
    "emar": ("hosp/emar.csv", "TRY_CAST(hadm_id AS BIGINT) hadm_id, TRY_CAST(charttime AS TIMESTAMP) charttime"),
}
have = set()
for t, (rel, cols) in big.items():
    p = path(rel)
    if p is None:
        say(f"(skipping {t}: {rel} not found)")
        continue
    have.add(t)
    con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT {cols} "
                f"FROM read_csv('{p}', header=true, all_varchar=true)")
for t in ("inputevents", "procedureevents"):
    if t not in have:
        sys.exit(f"{t} is required")

say(f"MIMIC_DIR = {MIMIC}\nBuilding per-stay coverage table (one pass over inputevents/procedureevents) ...")
con.execute("""
CREATE TEMP TABLE s AS
SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime, i.los, i.first_careunit,
       p.anchor_year_group AS grp,
       (ie.stay_id IS NULL AND pe.stay_id IS NULL) AS no_data
FROM icustays i
JOIN patients p USING (subject_id)
LEFT JOIN (SELECT DISTINCT stay_id FROM inputevents) ie USING (stay_id)
LEFT JOIN (SELECT DISTINCT stay_id FROM procedureevents) pe USING (stay_id)
""")
q = lambda sql: con.execute(sql).fetchall()

# ---------------------------------------------------------------------------------------------
say("\n== 1. Reproduce the BigQuery numbers quoted in the paper (anchor-year group 2020 - 2022)")
n, miss = q("SELECT COUNT(*), COUNT(*) FILTER (WHERE no_data) FROM s WHERE grp = '2020 - 2022'")[0]
check("Q5: 10,797 stays, 6,272 with neither table", (n, miss) == (10797, 6272), f"got {n} stays, {miss} missing ({miss / n:.1%})")

r = q("""
WITH x AS (SELECT * FROM s WHERE grp = '2020 - 2022'),
adm AS (SELECT COUNT(*) n, COUNT(*) FILTER (WHERE no_data) m FROM x GROUP BY hadm_id HAVING COUNT(*) > 1),
pat AS (SELECT COUNT(*) n, COUNT(*) FILTER (WHERE no_data) m FROM x GROUP BY subject_id HAVING COUNT(*) > 1)
SELECT 'admission', COUNT(*), COUNT(*) FILTER (WHERE m = 0), COUNT(*) FILTER (WHERE m = n),
       COUNT(*) FILTER (WHERE m BETWEEN 1 AND n - 1) FROM adm
UNION ALL
SELECT 'patient', COUNT(*), COUNT(*) FILTER (WHERE m = 0), COUNT(*) FILTER (WHERE m = n),
       COUNT(*) FILTER (WHERE m BETWEEN 1 AND n - 1) FROM pat
""")
got = {row[0]: tuple(row[1:]) for row in r}
check("Q6: admissions 1019/400/591/28", got.get("admission") == (1019, 400, 591, 28), f"got {got.get('admission')}")
check("Q6: patients 1529/555/807/167", got.get("patient") == (1529, 555, 807, 167), f"got {got.get('patient')}")

# Q4 SQL was never saved; this is a reconstruction. A match recovers the missing query.
r = q("""
SELECT CASE WHEN los < 1 THEN '<1d' WHEN los < 2 THEN '1-2d' WHEN los < 5 THEN '2-5d' ELSE '5d+' END b,
       COUNT(*), ROUND(100 * AVG(CAST(no_data AS INT)), 1)
FROM s WHERE grp = '2020 - 2022' GROUP BY b ORDER BY b
""")
got = {b: (c, p) for b, c, p in r}
exp = {"<1d": (1963, 56.3), "1-2d": (2876, 56.5), "2-5d": (3379, 61.1), "5d+": (2579, 57.3)}
check("Q4 (reconstructed SQL): LOS bins 56.3-61.1%", got == exp, f"got {got}")

r = q("""
SELECT s.grp,
       COUNT(DISTINCT CASE WHEN ie.itemid = 221749 THEN s.stay_id END),
       COUNT(DISTINCT CASE WHEN ie.itemid IN (221749, 229630, 229632) THEN s.stay_id END)
FROM inputevents ie JOIN s USING (stay_id)
WHERE ie.itemid IN (221749, 229630, 229632) GROUP BY s.grp ORDER BY s.grp
""")
got = {g: (a, b) for g, a, b in r}
check("Q2: phenylephrine stays 1845->3495 (2017-19), 311->1147 (2020-22)",
      got.get("2017 - 2019") == (1845, 3495) and got.get("2020 - 2022") == (311, 1147), f"got {got}")

# ---------------------------------------------------------------------------------------------
say("\n== 2. What the ItemIDs are (d_items)")
r = q("SELECT itemid, label, category FROM d_items WHERE itemid IN "
      "(221906,221289,221662,221749,222315,227692,229617,229630,229631,229632) ORDER BY itemid")
lab = {i: l for i, l, _ in r}
for i, l, c in r:
    say(f"       {i}: {l}  [{c}]")
check("229630 and 229632 are phenylephrine",
      all("phenyl" in str(lab.get(i, "")).lower() for i in (229630, 229632)),
      f"229630={lab.get(229630)!r}, 229632={lab.get(229632)!r}")
info("229617 (included in the extended set of make_flags.py)", f"label={lab.get(229617)!r}")

# ---------------------------------------------------------------------------------------------
say("\n== 3. Order of stays with and without records within the same admission / patient")
say("       (dates are shifted per patient but order within a patient is preserved; descriptive only)")
for unit, key in [("admissions", "hadm_id"), ("patients", "subject_id")]:
    for label, where in [("2020 - 2022 anchor group", "WHERE grp = '2020 - 2022'"), ("all anchor groups", "")]:
        r = q(f"""
        WITH x AS (
          SELECT {key}, MAX(intime) FILTER (WHERE NOT no_data) last_cov, MIN(intime) FILTER (WHERE no_data) first_miss
          FROM s {where} GROUP BY {key}
          HAVING COUNT(*) FILTER (WHERE no_data) > 0 AND COUNT(*) FILTER (WHERE NOT no_data) > 0)
        SELECT COUNT(*), COUNT(*) FILTER (WHERE last_cov < first_miss) FROM x
        """)[0]
        info(f"Mixed {unit}, {label}",
             f"{r[1]} of {r[0]} ({r[1] / max(r[0], 1):.1%}) have all stays with records before all stays without")

# ---------------------------------------------------------------------------------------------
say("\n== 4. Do the 'missing' 2020-2022 stays have data in OTHER tables?")
cols, joins = [], []
for t in ("chartevents", "outputevents", "datetimeevents", "ingredientevents"):
    if t in have:
        joins.append(f"LEFT JOIN (SELECT DISTINCT stay_id, TRUE AS has_{t} FROM {t}) USING (stay_id)")
        cols.append(f"COUNT(*) FILTER (WHERE has_{t})")
if "chartevents" in have:
    joins.append("LEFT JOIN (SELECT DISTINCT stay_id, TRUE AS has_vent FROM chartevents "
                 "WHERE itemid IN (224684, 224685)) USING (stay_id)")
    cols.append("COUNT(*) FILTER (WHERE has_vent)")
names = [c.split("has_")[1].rstrip(")") for c in cols]
if "emar" in have:
    con.execute("""CREATE TEMP TABLE em AS
      SELECT x.stay_id, BOOL_OR(e.charttime BETWEEN x.intime AND x.outtime) in_icu
      FROM (SELECT stay_id, hadm_id, intime, outtime FROM s WHERE grp = '2020 - 2022') x
      JOIN emar e USING (hadm_id) GROUP BY x.stay_id""")
    joins.append("LEFT JOIN em USING (stay_id)")
    cols += ["COUNT(*) FILTER (WHERE em.stay_id IS NOT NULL)", "COUNT(*) FILTER (WHERE em.in_icu)"]
    names += ["emar_same_admission", "emar_during_icu_stay"]
r = q(f"""SELECT no_data, COUNT(*), {', '.join(cols)}
          FROM s {' '.join(joins)} WHERE grp = '2020 - 2022' GROUP BY no_data ORDER BY no_data DESC""")
for row in r:
    nd, total, vals = row[0], row[1], row[2:]
    say(f"       {'MISSING' if nd else 'covered'} stays: {total}")
    for nm, v in zip(names, vals):
        say(f"         {nm:22s} {v:6d}  ({v / total:.1%})")
miss_row = next((row for row in r if row[0]), None)
if miss_row and "chartevents" in have:
    k = miss_row[2 + names.index("chartevents")]
    check("Missing stays still have chartevents (i.e. only input/procedure feeds are absent)",
          k > 0, f"{k} of {miss_row[1]} missing stays have chartevents rows")

# ---------------------------------------------------------------------------------------------
say("\n== Summary")
for tag, name in results:
    say(f"  {tag:4s}  {name}")
n_fail = sum(t == "FAIL" for t, _ in results)
say(f"\n{n_fail} FAIL. Log written to {LOG}")
sys.exit(1 if n_fail else 0)
