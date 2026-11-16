# Independent re-check of the MIMIC-IV 3.1 audit (November 2026)

Two scripts that re-derive the audit numbers directly from the raw MIMIC-IV 3.1 CSV files with DuckDB,
independently of the BigQuery queries in `../evidence/bigquery_audit_queries.sql`. Output is aggregate
only; counts between 1 and 9 are shown as `<10`.

```bash
pip install duckdb
MIMIC_DIR=/path/to/mimic-iv-3.1 python confirm_audit_findings.py   # writes confirm_audit_findings.log
MIMIC_DIR=/path/to/mimic-iv-3.1 python check_unlinked_rows.py > check_unlinked_rows.log
```

`MIMIC_DIR` must contain the `icu/` and `hosp/` folders (`.csv` or `.csv.gz`).

## `confirm_audit_findings.py`

| Section | What it checks |
|---|---|
| 1 | Re-computes the BigQuery outputs for the 2020 - 2022 anchor-year group (stays without `inputevents`/`procedureevents` rows, clustering by admission and patient, length-of-stay bins, phenylephrine stays under the original and extended ItemID sets). The length-of-stay query (Q4), whose SQL text was not saved with its output, is reconstructed here. |
| 2 | Labels of the vasoactive ItemIDs in `d_items`. |
| 3 | Order of stays with and without records within the same admission or patient (descriptive). |
| 4 | Coverage of the other ICU tables (`chartevents`, `outputevents`, `datetimeevents`, `ingredientevents`) and eMAR, for stays with and without `inputevents`/`procedureevents` rows. |

## `check_unlinked_rows.py`

For the 2020 - 2022 stays without `inputevents`/`procedureevents` rows, checks whether such rows exist
in those tables with an empty or unknown `stay_id`, or under another stay of the same patient or
admission inside the stay's time window.

## Related public reports

- [MIT-LCP/mimic-code#1839](https://github.com/MIT-LCP/mimic-code/issues/1839): fewer fluid records in
  `inputevents` for the 2020 - 2022 anchor-year group.
- [MIT-LCP/mimic-code#2159](https://github.com/MIT-LCP/mimic-code/issues/2159): phenylephrine
  administrations under the concentration-variant ItemIDs.
