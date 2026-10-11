# BigQuery audit outputs (transcribed from the authors' query results)

Source: query outputs pasted by the authors; not independently rerun. Queries: `bigquery_audit_queries.sql`.
Era = MIMIC raw `anchor_year_group`.

## Q1: stays with each ItemID

| era | itemid | stays |
|---|---|---|
| 2008 - 2010 | 221289 | 921 |
| 2008 - 2010 | 221749 | 4661 |
| 2008 - 2010 | 229617 | 21 |
| 2008 - 2010 | 229630 | 413 |
| 2008 - 2010 | 229632 | 62 |
| 2011 - 2013 | 221289 | 749 |
| 2011 - 2013 | 221749 | 3777 |
| 2011 - 2013 | 229617 | 15 |
| 2011 - 2013 | 229630 | 270 |
| 2011 - 2013 | 229632 | 35 |
| 2014 - 2016 | 221289 | 897 |
| 2014 - 2016 | 221749 | 3834 |
| 2014 - 2016 | 229617 | 10 |
| 2014 - 2016 | 229630 | 345 |
| 2014 - 2016 | 229632 | 43 |
| 2017 - 2019 | 221289 | 935 |
| 2017 - 2019 | 221749 | 1845 |
| 2017 - 2019 | 229617 | 42 |
| 2017 - 2019 | 229630 | 1838 |
| 2017 - 2019 | 229632 | 199 |
| 2020 - 2022 | 221289 | 330 |
| 2020 - 2022 | 221749 | 311 |
| 2020 - 2022 | 229617 | 19 |
| 2020 - 2022 | 229630 | 912 |
| 2020 - 2022 | 229632 | 114 |

(229631 returned no rows.)

## Q2: old versus extended sets

| era | phenyl_old | phenyl_all | anyvaso_old | anyvaso_all |
|---|---|---|---|---|
| 2008 - 2010 | 4661 | 5038 | 8339 | 8487 |
| 2011 - 2013 | 3777 | 4020 | 6080 | 6184 |
| 2014 - 2016 | 3834 | 4141 | 6321 | 6442 |
| 2017 - 2019 | 1845 | 3495 | 4923 | 5768 |
| 2020 - 2022 | 311 | 1147 | 1627 | 2003 |

## Q3: no-data share by care unit

| era | unit | stays | pct_no_data | median_los_days | pct_vaso_among_covered |
|---|---|---|---|---|---|
| 2017 - 2019 | Cardiac Vascular Intensive Care Unit (CVICU) | 2838 | 4.2 | 1.79 | 74.7 |
| 2017 - 2019 | Medical Intensive Care Unit (MICU) | 2694 | 6.5 | 2.14 | 41.8 |
| 2017 - 2019 | Neuro Intermediate | 2370 | 5.4 | 2.91 | 6.4 |
| 2017 - 2019 | Medical/Surgical Intensive Care Unit (MICU/SICU) | 2129 | 8.1 | 1.77 | 33.1 |
| 2017 - 2019 | Trauma SICU (TSICU) | 1685 | 4.5 | 1.93 | 37.0 |
| 2017 - 2019 | Coronary Care Unit (CCU) | 1641 | 7.3 | 2.04 | 38.5 |
| 2017 - 2019 | Surgical Intensive Care Unit (SICU) | 1310 | 5.7 | 2.23 | 37.9 |
| 2017 - 2019 | Neuro Surgical Intensive Care Unit (Neuro SICU) | 713 | 4.6 | 2.19 | 27.1 |
| 2017 - 2019 | Neuro Stepdown | 633 | 0.0 | 2.31 | 5.1 |
| 2017 - 2019 | Surgery/Vascular/Intermediate | 13 | 0.0 | 12.27 | 92.3 |
| 2017 - 2019 | PACU | 12 | 33.3 | 2.69 | 37.5 |
| 2017 - 2019 | Intensive Care Unit (ICU) | 7 | 0.0 | 12.28 | 85.7 |
| 2017 - 2019 | Medicine | 2 | 0.0 | 1.88 | 100.0 |
| 2017 - 2019 | Neurology | 1 | 0.0 | 28.25 | 100.0 |
| 2020 - 2022 | Neuro Intermediate | 2138 | 58.6 | 3.08 | 8.2 |
| 2020 - 2022 | Cardiac Vascular Intensive Care Unit (CVICU) | 2065 | 60.0 | 2.06 | 77.1 |
| 2020 - 2022 | Medical Intensive Care Unit (MICU) | 1525 | 57.9 | 2.35 | 51.1 |
| 2020 - 2022 | Medical/Surgical Intensive Care Unit (MICU/SICU) | 1284 | 60.1 | 1.83 | 44.5 |
| 2020 - 2022 | Trauma SICU (TSICU) | 1190 | 54.9 | 2.05 | 43.8 |
| 2020 - 2022 | Coronary Care Unit (CCU) | 1056 | 58.4 | 2.01 | 42.8 |
| 2020 - 2022 | Surgical Intensive Care Unit (SICU) | 874 | 60.3 | 2.44 | 38.3 |
| 2020 - 2022 | Neuro Surgical Intensive Care Unit (Neuro SICU) | 473 | 57.9 | 2.75 | 41.2 |
| 2020 - 2022 | PACU | 81 | 66.7 | 1.98 | 48.1 |
| 2020 - 2022 | Surgery/Vascular/Intermediate | 78 | 0.0 | 13.01 | 79.5 |
| 2020 - 2022 | Intensive Care Unit (ICU) | 16 | 0.0 | 2.21 | 43.8 |
| 2020 - 2022 | Medicine | 10 | 0.0 | 14.78 | 100.0 |
| 2020 - 2022 | Surgery/Trauma | 6 | 0.0 | 11.96 | 100.0 |
| 2020 - 2022 | Medicine/Cardiology Intermediate | 1 | 0.0 | 2.58 | 100.0 |

## Q4: no-data share by length-of-stay bin (SQL not saved)

| era | los_bin | stays | pct_no_data |
|---|---|---|---|
| 2008 - 2010 | 1-2d | 9155 | 2.9 |
| 2008 - 2010 | 2-5d | 9092 | 3.4 |
| 2008 - 2010 | 5d+ | 4864 | 4.1 |
| 2008 - 2010 | <1d | 6891 | 4.4 |
| 2011 - 2013 | 1-2d | 6149 | 3.3 |
| 2011 - 2013 | 2-5d | 5933 | 3.7 |
| 2011 - 2013 | 5d+ | 3230 | 5.0 |
| 2011 - 2013 | <1d | 4163 | 4.4 |
| 2014 - 2016 | 1-2d | 5609 | 3.7 |
| 2014 - 2016 | 2-5d | 5693 | 4.1 |
| 2014 - 2016 | 5d+ | 3270 | 5.0 |
| 2014 - 2016 | <1d | 3564 | 5.5 |
| 2017 - 2019 | 1-2d | 4703 | 5.5 |
| 2017 - 2019 | 2-5d | 5089 | 4.9 |
| 2017 - 2019 | 5d+ | 3222 | 4.9 |
| 2017 - 2019 | <1d | 3034 | 7.6 |
| 2020 - 2022 | 1-2d | 2876 | 56.5 |
| 2020 - 2022 | 2-5d | 3379 | 61.1 |
| 2020 - 2022 | 5d+ | 2579 | 57.3 |
| 2020 - 2022 | <1d | 1963 | 56.3 |

## Q5: covered versus missing, 2020 - 2022 anchor-year group

| no_data | stays | mortality_pct | mean_age | median_los |
|---|---|---|---|---|
| true | 6272 | 14.1 | 63.9 | 2.33 |
| false | 4525 | 14.0 | 63.8 | 2.15 |

## Q6: clustering of the gap, 2020 - 2022 anchor-year group

| level | multi_stay_units | all_covered | all_missing | mixed |
|---|---|---|---|---|
| patient | 1529 | 555 | 807 | 167 |
| admission | 1019 | 400 | 591 | 28 |
