# Evidence for the audit numbers quoted in the paper

| Paper number | Where it is saved | How to check |
|---|---|---|
| 53.3% (8,933 of 16,745) lack inputevents; 53.2% lack both | `../raw_check_icu_records_check.log` (reconstructed-year table) | `verify_audit_numbers.py` part A1 |
| 761 stays with reconstructed year after 2022, all missing; 51.0% (8,147 of 15,984) | derived from the same log by subtraction (the log's year table stops at 2023, so the 59 stays after 2023 are inferred from the bin total) | `verify_audit_numbers.py` part A1 |
| Under 1% missing in each earlier era | `../raw_check_icu_records_check.log` ("neither": 0.28%, 0.19%, 0.28%, 0.64%) | read the log |
| 10,797 stays; 6,272 (58.1%) missing; admissions 1,019 / 400 / 591 / 28; patients 1,529 / 555 / 807 / 167 | `bigquery_audit_results.md` (Q5, Q6) and `bigquery_audit_queries.sql` | rerun the SQL in BigQuery; arithmetic in `verify_audit_numbers.py` A2 |
| 55-60% per care unit, 56-61% per length-of-stay bin | `bigquery_audit_results.md` (Q3, Q4); the Q4 SQL text was not saved | `verify_audit_numbers.py` A2 |
| Mortality 14.0% vs 14.1%, age 63.8 vs 63.9, ICU stay 2.15 vs 2.33 days | `bigquery_audit_results.md` (Q5) | rerun the SQL |
| Phenylephrine 1,845 -> 3,495 (+89%), 311 -> 1,147 (+269%), +6 to +8% before 2017 | `bigquery_audit_results.md` (Q2) | `verify_audit_numbers.py` A3 |
| Vasopressor label rate 20.5 -> 23.9% (2017-19) and 22.5 -> 27.8% (2020-22); clean data use the extended IDs | `verify_audit_numbers.py` part B and `verify_audit_numbers.log`, from `stay_flags.parquet` and the clean parquets on branch `main` | `python verify_audit_numbers.py <dir with the parquets>` |
| Table (appendix) N = 8,124 / 6,962 and SMDs (age 0.0005, weight -0.0191, female 0.0069, mortality 0.0266) | `exclusion_balance_run_try3.log` (transcribed terminal output); the inputs (pre-clean parquets) are not in the repo | rerun `../../compare_2020_22_exclusion.py --files ...` locally |

## Known differences between sources (not errors)

- **9,986 vs 10,797 stays, 58.3% vs 58.1%, for the 2020-2022 anchor-year group.** The raw-table log assigned anchor years only to subjects present in the cohort parquets (60,001 subjects; 5,510 stays had no anchor), whereas the BigQuery query joins all of `patients`. The paper uses the BigQuery counts for anchor-year statements. This explanation is an inference, not verified.
- **Reconstructed-year vs anchor-year groupings.** The saved log reports missingness by reconstructed calendar year (about 50-57% by unit, 51-55% by length-of-stay); the BigQuery tables use MIMIC's raw anchor-year group. The paper states which grouping each number uses.
- **Earlier exclusion runs.** `../../Outpt_compare_2020_22_exclusion.txt` holds two earlier runs (8,272 / 7,091 stays with only age and weight; and 49,632 / 42,546, which counted hourly rows six times). They are superseded by `exclusion_balance_run_try3.log`; 277 of the 15,363 flagged stays are absent from the try3 pre-clean files, which is why N is 15,086.
- **BigQuery Q2 "any vasoactive" columns** use a candidate ItemID list that differs from the pipeline's list in `../make_flags.py`; the paper uses only the phenylephrine columns.
