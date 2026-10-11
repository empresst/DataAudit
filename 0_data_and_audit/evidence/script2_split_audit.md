# Audit of the cohort-split logic in 2_source_model/script2_two_stream_model.py

Read in full: lines 95-260 (config, source-era validation partition) and 436-560 (chronological cohort assignment, temporal separation audit).
Checked against: logs/ml4h_clean_v2/script2.log and the clean parquets committed on branch `main`.

| Property | Result | Evidence |
|---|---|---|
| Train, validation and test subjects are disjoint | True: 0 subjects shared by any pair of splits | computed from the clean parquets (train 18,878 / val 3,596 / test 32,322 subjects) |
| Validation set contains only source-era years | True (asserted in the script) | script2.py line ~150, parquets: val = 2011-2013 only |
| Normalisation statistics come from the training split only | True | `calculate_train_stats(train_df, ...)` then `normalize` on val and test |
| Adaptation and evaluation cohorts are subject-disjoint | True (asserted: `assert not (DETECT_SUBJ & EVAL_SUBJ)`), each subject has one global role | script2.py ~line 495, 16,580 adapt / 15,742 eval subjects in the log |
| Adaptation precedes evaluation in time | **Only at first-year resolution.** 69.8%, 72.4% and 89.6% of evaluation stays occur in a year before the last adaptation year (5,813/8,328; 6,852/9,470; 2,779/3,101) | script2.log, "Temporal separation audit"; the script itself says to report this table |
| Cohorts are reproducible | Pinned with PIN_SPLIT; log confirms "every cohort matches the pinned split exactly" | script2.log |
| 1,464 of 32,322 test-era subjects have stays in more than one era | Each is assigned to one cohort globally | script2.log |

Conclusion: no subject-level leakage found. The paper's description of the split was corrected to state the first-year ordering and the 70-90% overlap, and a Limitation was added.
Not read: lines 260-436 and 560-2155 of script2 (training loops, drift-detector branch used in the old gate, reporting).
