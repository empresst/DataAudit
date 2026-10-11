-- BigQuery queries run by the authors on physionet-data (MIMIC-IV 3.1) for the audit in the paper.
-- Saved verbatim from the queries the authors ran; outputs are in bigquery_audit_results.md.
-- All "era" values are MIMIC's raw patients.anchor_year_group, not the reconstructed calendar year.

-- Q1. Stays containing each candidate vasoactive ItemID, by anchor-year group
SELECT p.anchor_year_group AS era, ie.itemid, COUNT(DISTINCT ie.stay_id) AS stays
FROM `physionet-data.mimiciv_3_1_icu.inputevents` ie
JOIN `physionet-data.mimiciv_3_1_hosp.patients` p USING (subject_id)
WHERE ie.itemid IN (221749, 221289, 229617, 229630, 229631, 229632)
GROUP BY era, ie.itemid ORDER BY era, ie.itemid;

-- Q2. Old versus extended ItemID sets (phenylephrine; any vasoactive agent) by anchor-year group
-- NOTE: the "anyvaso_old" list below (221653, 221986 included; 227692 absent) differs from the
-- pipeline's list in 0_data_and_audit/make_flags.py (OLD = 221906,221289,221662,221749,222315,227692).
-- Only the phenylephrine columns (221749 / 229630 / 229632) are used in the paper.
WITH x AS (
  SELECT p.anchor_year_group AS era, ie.stay_id, ie.itemid
  FROM `physionet-data.mimiciv_3_1_icu.inputevents` ie
  JOIN `physionet-data.mimiciv_3_1_hosp.patients` p USING (subject_id)
)
SELECT era,
  COUNT(DISTINCT IF(itemid = 221749, stay_id, NULL)) AS phenyl_old,
  COUNT(DISTINCT IF(itemid IN (221749,229630,229632), stay_id, NULL)) AS phenyl_all,
  COUNT(DISTINCT IF(itemid IN (221906,222315,221662,221653,221986,221289,221749), stay_id, NULL)) AS anyvaso_old,
  COUNT(DISTINCT IF(itemid IN (221906,222315,221662,221653,221986,221289,221749,229617,229630,229632), stay_id, NULL)) AS anyvaso_all
FROM x GROUP BY era ORDER BY era;

-- Q3. Share of ICU stays with neither inputevents nor procedureevents, by care unit (2017-2019 vs 2020-2022)
WITH s AS (
  SELECT p.anchor_year_group AS era, i.first_careunit AS unit, i.los,
         (ie.stay_id IS NULL AND pe.stay_id IS NULL) AS no_data,
         (vp.stay_id IS NOT NULL) AS got_vaso
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  JOIN `physionet-data.mimiciv_3_1_hosp.patients` p USING (subject_id)
  LEFT JOIN (SELECT DISTINCT stay_id FROM `physionet-data.mimiciv_3_1_icu.inputevents`) ie USING (stay_id)
  LEFT JOIN (SELECT DISTINCT stay_id FROM `physionet-data.mimiciv_3_1_icu.procedureevents`) pe USING (stay_id)
  LEFT JOIN (SELECT DISTINCT stay_id FROM `physionet-data.mimiciv_3_1_icu.inputevents`
             WHERE itemid IN (221906,222315,221662,221653,221986,221289,221749,229617,229630,229632)) vp USING (stay_id)
  WHERE p.anchor_year_group IN ('2017 - 2019','2020 - 2022')
)
SELECT era, unit,
       COUNT(*) AS stays,
       ROUND(100*AVG(CAST(no_data AS INT64)),1) AS pct_no_data,
       ROUND(APPROX_QUANTILES(los, 100)[OFFSET(50)],2) AS median_los_days,
       ROUND(100*AVG(IF(NOT no_data, CAST(got_vaso AS INT64), NULL)),1) AS pct_vaso_among_covered
FROM s GROUP BY era, unit ORDER BY era, stays DESC;

-- Q4. Share with no data by length-of-stay bin and anchor-year group.
-- The SQL text for this query was not saved with its output (output in bigquery_audit_results.md, table Q4).
-- Bins in the output: <1d, 1-2d, 2-5d, 5d+.

-- Q5. Covered versus missing stays in the 2020-2022 anchor-year group: mortality, age, length of stay
WITH s AS (
  SELECT i.subject_id, i.hadm_id, i.stay_id, a.hospital_expire_flag AS died,
         p.anchor_age AS age, i.los,
         (ie.stay_id IS NULL AND pe.stay_id IS NULL) AS no_data
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  JOIN `physionet-data.mimiciv_3_1_hosp.patients` p USING (subject_id)
  JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a USING (hadm_id)
  LEFT JOIN (SELECT DISTINCT stay_id FROM `physionet-data.mimiciv_3_1_icu.inputevents`) ie USING (stay_id)
  LEFT JOIN (SELECT DISTINCT stay_id FROM `physionet-data.mimiciv_3_1_icu.procedureevents`) pe USING (stay_id)
  WHERE p.anchor_year_group = '2020 - 2022'
)
SELECT no_data, COUNT(*) AS stays,
       ROUND(100*AVG(died),1) AS mortality_pct,
       ROUND(AVG(age),1) AS mean_age,
       ROUND(APPROX_QUANTILES(los,100)[OFFSET(50)],2) AS median_los
FROM s GROUP BY no_data;

-- Q6. Does the missing-table gap follow the admission or the patient? (2020-2022 anchor-year group)
WITH s AS (
  SELECT i.subject_id, i.hadm_id, i.stay_id,
         (ie.stay_id IS NULL AND pe.stay_id IS NULL) AS no_data
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  JOIN `physionet-data.mimiciv_3_1_hosp.patients` p USING (subject_id)
  LEFT JOIN (SELECT DISTINCT stay_id FROM `physionet-data.mimiciv_3_1_icu.inputevents`) ie USING (stay_id)
  LEFT JOIN (SELECT DISTINCT stay_id FROM `physionet-data.mimiciv_3_1_icu.procedureevents`) pe USING (stay_id)
  WHERE p.anchor_year_group = '2020 - 2022'
),
adm AS (
  SELECT hadm_id, COUNT(*) AS n_stays, COUNTIF(no_data) AS n_missing
  FROM s GROUP BY hadm_id HAVING COUNT(*) > 1
),
pat AS (
  SELECT subject_id, COUNT(*) AS n_stays, COUNTIF(no_data) AS n_missing
  FROM s GROUP BY subject_id HAVING COUNT(*) > 1
)
SELECT 'admission' AS level,
       COUNT(*) AS multi_stay_units,
       COUNTIF(n_missing = 0) AS all_covered,
       COUNTIF(n_missing = n_stays) AS all_missing,
       COUNTIF(n_missing BETWEEN 1 AND n_stays - 1) AS mixed
FROM adm
UNION ALL
SELECT 'patient', COUNT(*), COUNTIF(n_missing = 0), COUNTIF(n_missing = n_stays),
       COUNTIF(n_missing BETWEEN 1 AND n_stays - 1)
FROM pat;
