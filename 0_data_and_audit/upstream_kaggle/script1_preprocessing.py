"""
script1_preprocessing_v8.py

═══════════════════════════
"Ongoing Need" framing — the cleanest approach.

Key change: Labels now ask "Will this patient receive vasopressors/ventilation
during h8-h14?" regardless of whether treatment started before h8.
No patient exclusions for already_vaso or already_vent.

This eliminates:
  - The anti-correlation leak (features perfectly predicting label via SQL rules)
  - The constant-feature problem (excluding patients zeroes out feature variance)
  - The inconsistent handling (vaso relabeled FALSE vs vent excluded)

Label semantics:
  label_vasopressor: ANY vasopressor infusion overlapping h8-h14
  label_intubation:  ANY ventilation procedure overlapping h8-h14
                     OR charted vent settings (PEEP/TV) during h8-h14
  label_septic_shock: unchanged (all 3 components within h8-h14)
  label_arrest_proxy: unchanged
"""

import duckdb, json
import numpy as np
import polars as pl
from pathlib import Path

con = duckdb.connect()
data_path = Path("./data/mimic-iv-3.1")

for tbl, path in [
    ("icustays",        "icu/icustays.csv"),
    ("chartevents",     "icu/chartevents.csv"),
    ("patients",        "hosp/patients.csv"),
    ("admissions",      "hosp/admissions.csv"),
    ("labevents",       "hosp/labevents.csv"),
    ("d_labitems",      "hosp/d_labitems.csv"),
    ("inputevents",     "icu/inputevents.csv"),
    ("procedureevents", "icu/procedureevents.csv"),
    ("outputevents",    "icu/outputevents.csv"),
    ("d_items",         "icu/d_items.csv"),
    ("emar",            "hosp/emar.csv"),
    ("prescriptions",   "hosp/prescriptions.csv"),
    ("diagnoses_icd",   "hosp/diagnoses_icd.csv"),
]:
    con.execute(f"CREATE OR REPLACE VIEW {tbl} AS SELECT * FROM '{data_path}/{path}'")
print("All views registered")

OBS_HOURS, GAP_HOURS, PRED_START_H, PRED_END_H = 6, 2, 8, 14
MIN_STAY_HOURS, SEQ_LEN = 14, 6
TRAIN_YEARS = ["2008 - 2010", "2011 - 2013"]
print(f"observe h0-{OBS_HOURS-1} | gap h{OBS_HOURS}-{PRED_START_H-1} | predict h{PRED_START_H}-{PRED_END_H}")
print("Label framing: ONGOING NEED (no patient exclusions)")

# ── DEMOGRAPHICS ───────────────────────────────────────────────────────────────
# Fix 1: anchor_year_group anchors ONLY anchor_year. A stay at anchor_year + k
# occurred ~k years later in real time. We therefore derive a per-stay approximate
# year and re-bin it into the same five era labels. The output column is still
# called anchor_year_group so no downstream code needs to change; the original
# patient-level value is preserved as anchor_year_group_raw for auditing.
demo_df = con.execute("""
    WITH base AS (
        SELECT i.stay_id, i.subject_id, i.intime, i.los,
               pa.anchor_age, pa.anchor_year, pa.anchor_year_group,
               pa.gender, a.race, a.hospital_expire_flag,
               CAST(SUBSTR(pa.anchor_year_group, 1, 4) AS INTEGER) + 1 AS group_mid,
               EXTRACT(YEAR FROM i.intime)::INT - pa.anchor_year        AS year_offset
        FROM icustays i
        JOIN patients pa   ON i.subject_id = pa.subject_id
        JOIN admissions a  ON i.hadm_id    = a.hadm_id
        WHERE pa.anchor_year_group IS NOT NULL AND pa.anchor_year_group != ''
          AND i.subject_id IS NOT NULL AND i.subject_id > 0
    ),
    dated AS (
        SELECT *, LEAST(2022, GREATEST(2008, group_mid + year_offset)) AS approx_year
        FROM base
    )
    SELECT stay_id, subject_id,
           anchor_age + year_offset AS age,
           gender, race AS ethnicity,
           anchor_year, los AS length_of_stay,
           hospital_expire_flag AS mortality,
           anchor_year_group AS anchor_year_group_raw,
           approx_year,
           CASE
             WHEN approx_year <= 2010 THEN '2008 - 2010'
             WHEN approx_year <= 2013 THEN '2011 - 2013'
             WHEN approx_year <= 2016 THEN '2014 - 2016'
             WHEN approx_year <= 2019 THEN '2017 - 2019'
             ELSE                            '2020 - 2022'
           END AS anchor_year_group
    FROM dated
""").pl()

# Audit: how many stays move era once the offset is applied?
_moved = demo_df.filter(pl.col("anchor_year_group") != pl.col("anchor_year_group_raw")).height
print(f"Era reassignment: {_moved}/{demo_df.height} stays "
      f"({100*_moved/demo_df.height:.1f}%) moved to a later era group")
print(demo_df.group_by(["anchor_year_group_raw","anchor_year_group"]).len().sort("anchor_year_group_raw"))

demo_df.write_parquet("./mimiciv_demographics.parquet")
print(f"Demographics → {demo_df.shape}")
print("\nanchor_year_group distribution:")
print(demo_df["anchor_year_group"].value_counts().sort("anchor_year_group"))

# ── LABS ───────────────────────────────────────────────────────────────────────
labs_df = con.execute(f"""
    WITH labs AS (
        SELECT l.hadm_id, l.charttime, l.itemid, l.valuenum FROM labevents l
        WHERE l.itemid IN (50912,50971,50983,50902,50893,51006,50882,51221,51222,
                           51301,51265,50931,50813,51279,50820,50818,50802,50885,51003)
          AND l.valuenum IS NOT NULL AND l.valuenum > -50 AND l.valuenum < 10000
    )
    SELECT i.stay_id,
           GREATEST(0, FLOOR(EXTRACT(EPOCH FROM (l.charttime - i.intime))/3600))::INT AS hrs_from_admit,
           MAX(CASE WHEN l.itemid=50912 THEN l.valuenum END) AS creatinine,
           MAX(CASE WHEN l.itemid=51301 THEN l.valuenum END) AS wbc,
           MAX(CASE WHEN l.itemid=51265 THEN l.valuenum END) AS platelets,
           MAX(CASE WHEN l.itemid=50813 THEN l.valuenum END) AS lactate,
           MAX(CASE WHEN l.itemid=51006 THEN l.valuenum END) AS bun,
           MAX(CASE WHEN l.itemid=51279 THEN l.valuenum END) AS rbc,
           MAX(CASE WHEN l.itemid=50902 THEN l.valuenum END) AS chloride,
           MAX(CASE WHEN l.itemid=50893 THEN l.valuenum END) AS calcium,
           MAX(CASE WHEN l.itemid=50885 THEN l.valuenum END) AS bilirubin_total,
           MAX(CASE WHEN l.itemid=50931 THEN l.valuenum END) AS glucose,
           MAX(CASE WHEN l.itemid=51221 THEN l.valuenum END) AS hematocrit,
           MAX(CASE WHEN l.itemid=50971 THEN l.valuenum END) AS potassium,
           MAX(CASE WHEN l.itemid=50983 THEN l.valuenum END) AS sodium,
           MAX(CASE WHEN l.itemid=51003 THEN l.valuenum END) AS troponin_t,
           MAX(CASE WHEN l.itemid=50820 THEN l.valuenum END) AS ph_venous,
           MAX(CASE WHEN l.itemid=50818 THEN l.valuenum END) AS pco2_venous,
           MAX(CASE WHEN l.itemid=50802 THEN l.valuenum END) AS base_excess
    FROM labs l JOIN icustays i ON l.hadm_id = i.hadm_id
    WHERE l.charttime >= i.intime AND l.charttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
    GROUP BY i.stay_id, hrs_from_admit ORDER BY i.stay_id, hrs_from_admit
""").pl()
labs_df.write_parquet("./mimiciv_labs_hourly.parquet")
print(f"Labs → {labs_df.shape}")

# ── VITALS ─────────────────────────────────────────────────────────────────────
vitals_df = con.execute("""
    WITH v AS (
        SELECT stay_id, charttime, itemid, valuenum, value FROM chartevents
        WHERE itemid IN (
            220045,220179,220180,220050,220051,220052,
            223761,223762,220210,220277,
            220739, 223900, 223901  -- GCS Eye, Verbal, Motor
        )
        AND valuenum IS NOT NULL
    )
    SELECT i.stay_id,
           GREATEST(0, FLOOR(EXTRACT(EPOCH FROM (v.charttime - i.intime))/3600))::INT AS hrs_from_admit,
           MAX(CASE WHEN v.itemid=220045 AND v.valuenum BETWEEN 20  AND 250 THEN v.valuenum END) AS heart_rate,
           MAX(CASE WHEN v.itemid=220179 AND v.valuenum BETWEEN 40  AND 300 THEN v.valuenum END) AS sbp_noninvasive,
           MAX(CASE WHEN v.itemid=220180 AND v.valuenum BETWEEN 20  AND 200 THEN v.valuenum END) AS dbp_noninvasive,
           MAX(CASE WHEN v.itemid=220050 AND v.valuenum BETWEEN 40  AND 300 THEN v.valuenum END) AS sbp_invasive,
           MAX(CASE WHEN v.itemid=220051 AND v.valuenum BETWEEN 20  AND 200 THEN v.valuenum END) AS dbp_invasive,
           MAX(CASE WHEN v.itemid=220052 AND v.valuenum BETWEEN 30  AND 200 THEN v.valuenum END) AS map_invasive,
           MAX(CASE WHEN v.itemid=223762 AND v.valuenum BETWEEN 30  AND 45  THEN v.valuenum
                    WHEN v.itemid=223761 AND v.valuenum BETWEEN 86  AND 113 THEN (v.valuenum-32)*5.0/9.0
               END) AS temperature_c,
           MAX(CASE WHEN v.itemid=220277 AND v.valuenum BETWEEN 50  AND 100 THEN v.valuenum END) AS spo2,
           MAX(CASE WHEN v.itemid=220210 AND v.valuenum BETWEEN 4   AND 60  THEN v.valuenum END) AS resp_rate,
           -- GCS components (valid ranges: eye 1-4, verbal 1-5, motor 1-6).
           -- 'No Response-ETT' rows are an intubation marker, not a verbal score.
           -- Per mimic-code gcs.sql they are held out of gcs_verbal entirely and
           -- surfaced as an explicit flag, so the ETT signal is auditable instead
           -- of being smuggled into the physiology stream via gcs_verbal/its mask.
           MIN(CASE WHEN v.itemid=220739 AND v.valuenum BETWEEN 1 AND 4 THEN v.valuenum END) AS gcs_eye,
           MIN(CASE WHEN v.itemid=223900 AND v.value <> 'No Response-ETT'
                     AND v.valuenum BETWEEN 1 AND 5 THEN v.valuenum END) AS gcs_verbal,
           MIN(CASE WHEN v.itemid=223901 AND v.valuenum BETWEEN 1 AND 6 THEN v.valuenum END) AS gcs_motor,
           MAX(CASE WHEN v.itemid=223900 AND v.value = 'No Response-ETT' THEN 1 ELSE 0 END) AS endotrach_flag
    FROM v JOIN icustays i ON v.stay_id = i.stay_id
    WHERE v.charttime >= i.intime AND v.charttime <= i.outtime
    GROUP BY i.stay_id, hrs_from_admit ORDER BY i.stay_id, hrs_from_admit
""").pl()
vitals_df.write_parquet("./mimiciv_hourly_vitals_clean.parquet")
print(f"Vitals → {vitals_df.shape}")

# ── INTERVENTIONS — metadata only ─────────────────────────────────────────────
interv_df = con.execute(f"""
    WITH vaso AS (
        SELECT stay_id, starttime AS event_time, 1 AS vasopressor_flag, 0 AS ventilation_flag
        -- 221906 norepinephrine | 221289 epinephrine | 221662 dopamine
        -- 221749 phenylephrine  | 222315 vasopressin | 227692 isuprel
        FROM inputevents WHERE itemid IN (221906,221289,221662,221749,222315,227692) AND starttime IS NOT NULL
    ), vent AS (
        -- 225792 Invasive Ventilation | 224385 Intubation
        -- (227194 is EXTUBATION — the end of ventilation, not the start.
        --  224684 is a chartevents itemid and returns nothing here.)
        SELECT stay_id, starttime AS event_time, 0 AS vasopressor_flag, 1 AS ventilation_flag
        FROM procedureevents WHERE itemid IN (225792,224385) AND starttime IS NOT NULL
    ), hourly AS (
        SELECT i.stay_id,
               GREATEST(0, FLOOR(EXTRACT(EPOCH FROM (e.event_time - i.intime))/3600))::INT AS hrs_from_admit,
               MAX(e.vasopressor_flag) AS vasopressor_flag, MAX(e.ventilation_flag) AS ventilation_flag
        FROM icustays i JOIN (SELECT * FROM vaso UNION ALL SELECT * FROM vent) e ON e.stay_id = i.stay_id
        WHERE e.event_time >= i.intime AND e.event_time <= i.outtime
        GROUP BY i.stay_id, hrs_from_admit
    ), grid AS (
        SELECT i.stay_id, g.hr AS hrs_from_admit, 0 AS vasopressor_flag, 0 AS ventilation_flag
        FROM icustays i CROSS JOIN generate_series(0, {PRED_END_H}) g(hr)
    )
    SELECT stay_id, hrs_from_admit, MAX(vasopressor_flag)::INT8 AS vasopressor_flag,
           MAX(ventilation_flag)::INT8 AS ventilation_flag
    FROM (SELECT * FROM hourly UNION ALL
          SELECT g.* FROM grid g WHERE NOT EXISTS (
              SELECT 1 FROM hourly h WHERE h.stay_id=g.stay_id AND h.hrs_from_admit=g.hrs_from_admit))
    GROUP BY stay_id, hrs_from_admit ORDER BY stay_id, hrs_from_admit
""").pl()
interv_df.write_parquet("./mimiciv_interventions.parquet")
print(f"Interventions → {interv_df.shape}")

# ── PREVALENT-TREATMENT FLAGS — evaluation metadata, NEVER model features ─────
# Under the ongoing-need framing a patient on norepinephrine from h1 to h20 is a
# positive, and h0-h5 features (has_central_line_obs, max_fio2_obs,
# has_arterial_line_obs) identify that patient trivially. That is not temporal
# leakage — the features do precede the label — but it means AUROC is driven by
# treatment continuation, not by prediction of new need.
#
# These flags let evaluation be split into:
#   incident  (already_* == 0) → primary endpoint: prediction of NEW need
#   prevalent (already_* == 1) → secondary: continuation detection
# with NO patient excluded from training, so the no-exclusion design and the
# feature variance it protects are both preserved.
#
# CRITICAL — interval OVERLAP, not starttime. interv_df above filters
# `event_time >= i.intime`, which silently misses every infusion or ventilation
# episode that began in the ED, the OR, or a prior ward stay and is still
# running at ICU hour 0. Those are precisely the prevalent patients. The
# predicate below is: episode started before intime+OBS_HOURS AND ended after
# intime, i.e. it overlaps the observation window at any point.
prevalent_df = con.execute(f"""
    WITH vaso_prev AS (
        SELECT DISTINCT i.stay_id
        FROM icustays i JOIN inputevents iv ON iv.stay_id = i.stay_id
        WHERE iv.itemid IN (221906,221289,221662,221749,222315,227692)
          AND iv.starttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
          AND COALESCE(iv.endtime, iv.starttime) > i.intime
    ),
    vent_proc_prev AS (
        SELECT DISTINCT i.stay_id
        FROM icustays i JOIN procedureevents pe ON pe.stay_id = i.stay_id
        WHERE pe.itemid IN (225792, 224385)
          AND pe.starttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
          AND COALESCE(pe.endtime, pe.starttime) > i.intime
    ),
    vent_chart_prev AS (
        -- Set tidal volume (224684) and 224685 are charted only under
        -- positive-pressure ventilation. Mirrors the label definition so the
        -- prevalent flag and the label use the same notion of "ventilated".
        SELECT DISTINCT i.stay_id
        FROM icustays i JOIN chartevents cv ON cv.stay_id = i.stay_id
        WHERE cv.itemid IN (224684, 224685) AND cv.valuenum > 0
          AND cv.charttime >= i.intime
          AND cv.charttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
    ),
    ett_prev AS (
        SELECT DISTINCT i.stay_id
        FROM icustays i JOIN chartevents cv ON cv.stay_id = i.stay_id
        WHERE cv.itemid = 223900 AND cv.value = 'No Response-ETT'
          AND cv.charttime >= i.intime
          AND cv.charttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
    )
    SELECT i.stay_id,
           CASE WHEN vp.stay_id IS NOT NULL THEN 1 ELSE 0 END AS already_vaso,
           CASE WHEN pp.stay_id IS NOT NULL OR cp.stay_id IS NOT NULL
                     OR ep.stay_id IS NOT NULL THEN 1 ELSE 0 END AS already_vent,
           CASE WHEN ep.stay_id IS NOT NULL THEN 1 ELSE 0 END AS endotrach_obs
    FROM icustays i
    LEFT JOIN vaso_prev       vp ON vp.stay_id = i.stay_id
    LEFT JOIN vent_proc_prev  pp ON pp.stay_id = i.stay_id
    LEFT JOIN vent_chart_prev cp ON cp.stay_id = i.stay_id
    LEFT JOIN ett_prev        ep ON ep.stay_id = i.stay_id
""").pl()
prevalent_df.write_parquet("./mimiciv_prevalent_flags.parquet")
print(f"Prevalent flags → {prevalent_df.shape} | "
      f"already_vaso={prevalent_df['already_vaso'].mean():.3f} "
      f"already_vent={prevalent_df['already_vent'].mean():.3f} "
      f"endotrach={prevalent_df['endotrach_obs'].mean():.3f}")

# Contrast with the starttime-only definition to show what overlap recovers.
_starttime_only = (interv_df.filter(pl.col("hrs_from_admit") < OBS_HOURS)
                            .group_by("stay_id")
                            .agg([pl.col("vasopressor_flag").max().alias("st_vaso"),
                                  pl.col("ventilation_flag").max().alias("st_vent")]))
_cmp = prevalent_df.join(_starttime_only, on="stay_id", how="left").fill_null(0)
print(f"  Overlap-vs-starttime: vaso {int(_cmp['already_vaso'].sum())} vs "
      f"{int(_cmp['st_vaso'].sum())}, vent {int(_cmp['already_vent'].sum())} vs "
      f"{int(_cmp['st_vent'].sum())} — the difference is patients whose "
      f"treatment began before ICU admission.")

# ── TREATMENT FEATURES (Stream 2) — ALL features restored ─────────────────────
print("\n--- Extracting treatment features (Stream 2) ---")

treat_input = con.execute(f"""
    SELECT i.stay_id,
        COALESCE(SUM(CASE WHEN ie.itemid IN (225158,225828,225159) THEN ie.amount END), 0)
            AS total_crystalloid_ml
    FROM icustays i LEFT JOIN inputevents ie ON ie.stay_id = i.stay_id
        AND ie.starttime >= i.intime AND ie.starttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
    GROUP BY i.stay_id
""").pl()
print(f"  inputevents → {treat_input.shape}")

treat_emar = con.execute(f"""
    SELECT i.stay_id,
        MAX(CASE WHEN LOWER(e.medication) LIKE '%dexamethasone%' OR LOWER(e.medication) LIKE '%hydrocortisone%'
                   OR LOWER(e.medication) LIKE '%methylprednisolone%' OR LOWER(e.medication) LIKE '%prednisone%'
            THEN 1 ELSE 0 END) AS early_steroid,
        MAX(CASE WHEN LOWER(e.medication) LIKE '%vancomycin%' OR LOWER(e.medication) LIKE '%meropenem%'
                   OR LOWER(e.medication) LIKE '%piperacillin%' OR LOWER(e.medication) LIKE '%cefepime%'
                   OR LOWER(e.medication) LIKE '%ceftriaxone%' OR LOWER(e.medication) LIKE '%levofloxacin%'
                   OR LOWER(e.medication) LIKE '%azithromycin%'
            THEN 1 ELSE 0 END) AS early_antibiotic,
        MAX(CASE WHEN LOWER(e.medication) LIKE '%furosemide%' OR LOWER(e.medication) LIKE '%lasix%'
                   OR LOWER(e.medication) LIKE '%bumetanide%'
            THEN 1 ELSE 0 END) AS early_furosemide,
        MAX(CASE WHEN LOWER(e.medication) LIKE '%heparin%' OR LOWER(e.medication) LIKE '%enoxaparin%'
                   OR LOWER(e.medication) LIKE '%lovenox%'
            THEN 1 ELSE 0 END) AS early_anticoagulant,
        MAX(CASE WHEN LOWER(e.medication) LIKE '%labetalol%' OR LOWER(e.medication) LIKE '%hydralazine%'
                   OR LOWER(e.medication) LIKE '%nicardipine%' OR LOWER(e.medication) LIKE '%nitroglycerin%'
            THEN 1 ELSE 0 END) AS early_antihypertensive,
        MAX(CASE WHEN LOWER(e.medication) LIKE '%amiodarone%' OR LOWER(e.medication) LIKE '%diltiazem%'
                   OR LOWER(e.medication) LIKE '%metoprolol%'
            THEN 1 ELSE 0 END) AS early_antiarrhythmic,
        COUNT(DISTINCT e.medication) AS n_distinct_meds
    FROM icustays i LEFT JOIN emar e ON e.hadm_id = i.hadm_id
        AND e.charttime >= i.intime AND e.charttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
        AND e.event_txt NOT IN ('Held','Stopped','Refused','Not Given','Not Given - Loss of Vascular Access')
    GROUP BY i.stay_id
""").pl()
print(f"  emar → {treat_emar.shape}")

treat_rx = con.execute(f"""
    SELECT i.stay_id,
        MAX(CASE WHEN LOWER(p.drug) LIKE '%dexamethasone%' OR LOWER(p.drug) LIKE '%hydrocortisone%'
                   OR LOWER(p.drug) LIKE '%methylprednisolone%'
            THEN 1 ELSE 0 END) AS steroid_ordered,
        -- Fix 8: Observed values lie in [0, OBS_HOURS). COALESCEing "never ordered" to
        -- OBS_HOURS made it indistinguishable from "ordered at 5.99h". Use a
        -- sentinel outside the observable range instead (matches script2's
        -- SENTINEL_NO_EARLY_EVENT = SEQ_LEN + 1).
        COALESCE(MIN(CASE WHEN LOWER(p.drug) LIKE '%vancomycin%' OR LOWER(p.drug) LIKE '%meropenem%'
                            OR LOWER(p.drug) LIKE '%piperacillin%' OR LOWER(p.drug) LIKE '%cefepime%'
                            OR LOWER(p.drug) LIKE '%ceftriaxone%' OR LOWER(p.drug) LIKE '%levofloxacin%'
            THEN EXTRACT(EPOCH FROM (p.starttime - i.intime))/3600.0 END), {OBS_HOURS + 1})
            AS time_to_first_abx_order_hrs
    FROM icustays i LEFT JOIN prescriptions p ON p.hadm_id = i.hadm_id
        AND p.starttime >= i.intime AND p.starttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
    GROUP BY i.stay_id
""").pl()
print(f"  prescriptions → {treat_rx.shape}")

# Respiratory features including vent indicators (NOT leakage under ongoing-need framing)
treat_resp = con.execute(f"""
    SELECT i.stay_id,
        MAX(CASE WHEN c.itemid = 223835 AND c.valuenum BETWEEN 21 AND 100
            THEN c.valuenum END) AS max_fio2_obs,
        AVG(CASE WHEN c.itemid = 223835 AND c.valuenum BETWEEN 21 AND 100
            THEN c.valuenum END) AS mean_fio2_obs,
        MAX(CASE WHEN c.itemid = 220339 AND c.valuenum BETWEEN 0 AND 30
            THEN c.valuenum END) AS max_peep_obs,
        MAX(CASE WHEN c.itemid IN (224685, 224686) AND c.valuenum BETWEEN 100 AND 1000
            THEN c.valuenum END) AS max_tidal_volume_obs,
        MAX(CASE WHEN c.itemid = 223835 AND c.valuenum > 50 THEN 1 ELSE 0 END) AS high_fio2_flag,
        MAX(CASE WHEN c.itemid = 220339 AND c.valuenum > 0 THEN 1 ELSE 0 END) AS on_peep_flag
    FROM icustays i LEFT JOIN chartevents c ON c.stay_id = i.stay_id
        AND c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
        AND c.itemid IN (223835, 220339, 224685, 224686) AND c.valuenum IS NOT NULL
    GROUP BY i.stay_id
""").pl()
print(f"  respiratory → {treat_resp.shape}")

treat_sed = con.execute(f"""
    SELECT i.stay_id,
        MAX(CASE WHEN ie.itemid IN (222168, 221668) THEN 1 ELSE 0 END) AS has_propofol_midaz_obs,
        COALESCE(SUM(CASE WHEN ie.itemid IN (222168, 221668) THEN ie.amount END), 0)
            AS total_sedation_dose_obs
    FROM icustays i LEFT JOIN inputevents ie ON ie.stay_id = i.stay_id
        AND ie.starttime >= i.intime AND ie.starttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
    GROUP BY i.stay_id
""").pl()
print(f"  sedation → {treat_sed.shape}")


# Blood products (h0-6)
treat_blood = con.execute(f"""
    SELECT i.stay_id,
        MAX(CASE WHEN ie.itemid IN (225168,225170,225171,220970) THEN 1 ELSE 0 END)
            AS has_blood_products_obs,
        COALESCE(SUM(CASE WHEN ie.itemid = 225168 THEN ie.amount END), 0)
            AS total_prbc_ml,
        MAX(CASE WHEN ie.itemid IN (220864, 220862) THEN 1 ELSE 0 END)
            AS has_albumin_obs
    FROM icustays i LEFT JOIN inputevents ie ON ie.stay_id = i.stay_id
        AND ie.starttime >= i.intime
        AND ie.starttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
    GROUP BY i.stay_id
""").pl()
print(f"  blood products → {treat_blood.shape}")

# RRT / dialysis (h0-6)
treat_rrt = con.execute(f"""
    SELECT i.stay_id,
        MAX(CASE WHEN pe.itemid IN (225441,225802,225803,225805,225809,225955)
            THEN 1 ELSE 0 END) AS has_rrt_obs
    FROM icustays i LEFT JOIN procedureevents pe ON pe.stay_id = i.stay_id
        AND pe.starttime >= i.intime
        AND pe.starttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
    GROUP BY i.stay_id
""").pl()
print(f"  rrt → {treat_rrt.shape}")

# Insulin infusion + invasive lines (h0-6)
treat_lines = con.execute(f"""
    SELECT i.stay_id,
        MAX(CASE WHEN ie.itemid = 223258 THEN 1 ELSE 0 END)
            AS has_insulin_infusion_obs,
        MAX(CASE WHEN pe.itemid = 225752 THEN 1 ELSE 0 END)
            AS has_arterial_line_obs,
        MAX(CASE WHEN pe.itemid IN (224263, 224267, 225199) THEN 1 ELSE 0 END)
            AS has_central_line_obs
    FROM icustays i
    LEFT JOIN inputevents ie
        ON ie.stay_id = i.stay_id
        AND ie.starttime >= i.intime
        AND ie.starttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
        AND ie.itemid = 223258
    LEFT JOIN procedureevents pe
        ON pe.stay_id = i.stay_id
        AND pe.starttime >= i.intime
        AND pe.starttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
        AND pe.itemid IN (225752, 224263, 224267, 225199)
    GROUP BY i.stay_id
""").pl()
print(f"  insulin + lines → {treat_lines.shape}")

# Admission type, location, comorbidities (static)
treat_static = con.execute(f"""
    WITH comorbidities AS (
        SELECT d.hadm_id,
            MAX(CASE WHEN (d.icd_version=9  AND d.icd_code LIKE '401%')
                       OR (d.icd_version=10 AND d.icd_code LIKE 'I10%')
                THEN 1 ELSE 0 END) AS has_hypertension,
            MAX(CASE WHEN (d.icd_version=9  AND d.icd_code LIKE '250%')
                       OR (d.icd_version=10 AND (d.icd_code LIKE 'E10%'
                                              OR d.icd_code LIKE 'E11%'))
                THEN 1 ELSE 0 END) AS has_diabetes,
            MAX(CASE WHEN (d.icd_version=9  AND d.icd_code LIKE '428%')
                       OR (d.icd_version=10 AND d.icd_code LIKE 'I50%')
                THEN 1 ELSE 0 END) AS has_chf,
            MAX(CASE WHEN (d.icd_version=9  AND d.icd_code LIKE '585%')
                       OR (d.icd_version=10 AND d.icd_code LIKE 'N18%')
                THEN 1 ELSE 0 END) AS has_ckd,
            MAX(CASE WHEN (d.icd_version=9  AND d.icd_code LIKE '496%')
                       OR (d.icd_version=10 AND d.icd_code LIKE 'J44%')
                THEN 1 ELSE 0 END) AS has_copd,
            MAX(CASE WHEN (d.icd_version=9  AND d.icd_code LIKE '571%')
                       OR (d.icd_version=10 AND d.icd_code LIKE 'K74%')
                THEN 1 ELSE 0 END) AS has_liver_disease,
            MAX(CASE WHEN (d.icd_version=9  AND (
                                CAST(d.icd_code AS VARCHAR) BETWEEN '140' AND '172'
                             OR d.icd_code LIKE '19%'
                             OR d.icd_code LIKE '200%'
                             OR d.icd_code LIKE '208%'))
                       OR (d.icd_version=10 AND d.icd_code LIKE 'C%')
                THEN 1 ELSE 0 END) AS has_malignancy
        FROM diagnoses_icd d
        GROUP BY d.hadm_id
    )
    SELECT
        i.stay_id,
        -- Admission type one-hot (exact strings from your MIMIC-IV 3.1)
        CASE WHEN a.admission_type = 'EW EMER.'  THEN 1 ELSE 0 END
            AS admission_type_ewemer,
        CASE WHEN a.admission_type = 'URGENT'    THEN 1 ELSE 0 END
            AS admission_type_urgent,
        CASE WHEN a.admission_type = 'ELECTIVE'  THEN 1 ELSE 0 END
            AS admission_type_elective,
        -- Admission location one-hot
        CASE WHEN a.admission_location = 'EMERGENCY ROOM'         THEN 1 ELSE 0 END
            AS admission_loc_ed,
        CASE WHEN a.admission_location = 'TRANSFER FROM HOSPITAL' THEN 1 ELSE 0 END
            AS admission_loc_transfer,
        -- Comorbidities (defaulting to 0 if hadm not in diagnoses_icd)
        COALESCE(c.has_hypertension, 0)  AS has_hypertension,
        COALESCE(c.has_diabetes,     0)  AS has_diabetes,
        COALESCE(c.has_chf,          0)  AS has_chf,
        COALESCE(c.has_ckd,          0)  AS has_ckd,
        COALESCE(c.has_copd,         0)  AS has_copd,
        COALESCE(c.has_liver_disease,0)  AS has_liver_disease,
        COALESCE(c.has_malignancy,   0)  AS has_malignancy
    FROM icustays i
    JOIN admissions a ON i.hadm_id = a.hadm_id
    LEFT JOIN comorbidities c ON i.hadm_id = c.hadm_id
""").pl()
print(f"  static (admission + comorbidities) → {treat_static.shape}")


treatment_df = (treat_input.join(treat_emar,   on="stay_id", how="left")
    .join(treat_rx,     on="stay_id", how="left")
    .join(treat_resp,   on="stay_id", how="left")
    .join(treat_sed,    on="stay_id", how="left")
    .join(treat_blood,  on="stay_id", how="left")
    .join(treat_rrt,    on="stay_id", how="left")
    .join(treat_lines,  on="stay_id", how="left")
    .join(treat_static, on="stay_id", how="left")
    .fill_null(0))

TREATMENT_COLS = [
    # Fluid & Resuscitation (h0-h6)
    "total_crystalloid_ml", "has_blood_products_obs", "total_prbc_ml", "has_albumin_obs",
    # Medications & Orders (h0-h6)
    "early_steroid", "early_antibiotic", "early_furosemide", "early_anticoagulant",
    "early_antihypertensive", "early_antiarrhythmic",
    "n_distinct_meds", "steroid_ordered", "time_to_first_abx_order_hrs", "has_insulin_infusion_obs",
    # Organ Support & Invasive Lines (h0-h6)
    "has_rrt_obs", "has_arterial_line_obs", "has_central_line_obs",
    # Oxygenation Therapy (h0-h6)
    "max_fio2_obs", "mean_fio2_obs",
    # Admission Context & Demographics (t=0)
    "admission_type_ewemer", "admission_type_urgent", "admission_type_elective",
    "admission_loc_ed", "admission_loc_transfer",
]

treatment_df.write_parquet("./mimiciv_treatment_features.parquet")
print(f"  Treatment features: {len(TREATMENT_COLS)} cols")


# ── URINE ──────────────────────────────────────────────────────────────────────
urine_df = con.execute("""
    SELECT i.stay_id,
           GREATEST(0, FLOOR(EXTRACT(EPOCH FROM (o.charttime - i.intime))/3600))::INT AS hrs_from_admit,
           GREATEST(0, SUM(
               CASE WHEN LOWER(di.label) LIKE '%gu irrigant%' OR LOWER(di.label) LIKE '%irrigant%'
                    THEN -o.value ELSE o.value END)) AS urine_output_ml
    FROM outputevents o JOIN icustays i ON o.stay_id = i.stay_id JOIN d_items di ON o.itemid = di.itemid
    WHERE o.charttime >= i.intime AND o.charttime <= i.outtime
      AND o.value IS NOT NULL AND o.value > 0 AND o.value < 1500
      AND (LOWER(di.label) LIKE '%urine%' OR LOWER(di.label) LIKE '%foley%'
        OR LOWER(di.label) LIKE '%void%' OR LOWER(di.label) LIKE '%catheter%'
        OR LOWER(di.label) LIKE '%urostomy%' OR LOWER(di.label) LIKE '%ileoconduit%')
      AND LOWER(di.label) NOT LIKE '%irrigant in%' AND LOWER(di.label) NOT LIKE '%bladder irrigation%'
    GROUP BY i.stay_id, hrs_from_admit
    HAVING GREATEST(0, SUM(
               CASE WHEN LOWER(di.label) LIKE '%gu irrigant%' OR LOWER(di.label) LIKE '%irrigant%'
                    THEN -o.value ELSE o.value END)) <= 1500
    ORDER BY i.stay_id, hrs_from_admit
""").pl()
urine_df.write_parquet("./mimiciv_urine_hourly_final.parquet")
print(f"Urine → {urine_df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# LABELS — "ONGOING NEED" framing
# No patient exclusions. Labels check if treatment OVERLAPS with h8-h14.
# ══════════════════════════════════════════════════════════════════════════════

full_labels = con.execute(f"""
    WITH eligible AS (
        SELECT stay_id, subject_id, hadm_id, intime, outtime FROM icustays
        WHERE outtime >= intime + INTERVAL '{MIN_STAY_HOURS}' HOUR
    ),
    septic AS (
        -- The previous version joined `eligible e ON l.hadm_id = e.hadm_id` AND
        -- `icustays i ON l.hadm_id = i.hadm_id`, never using e.stay_id. For an
        -- admission with several ICU stays that produced a cross product which
        -- was then collapsed by DISTINCT. It was not a correctness bug — every
        -- predicate bound to the same `i`, so the h8-h14 window and the
        -- vaso/MAP subqueries all referred to one consistent stay — but it made
        -- the query quadratic in stays-per-admission and the intent unreadable.
        -- `eligible` already carries stay_id and intime, so use them directly.
        SELECT DISTINCT e.stay_id
        FROM labevents l JOIN eligible e ON l.hadm_id = e.hadm_id
        WHERE l.itemid = 50813 AND l.valuenum > 2
          AND l.charttime BETWEEN e.intime + INTERVAL '{PRED_START_H}' HOUR AND e.intime + INTERVAL '{PRED_END_H}' HOUR
          AND EXISTS (SELECT 1 FROM inputevents iv WHERE iv.stay_id = e.stay_id
              AND iv.itemid IN (221906,221289,221662,221749,222315,227692)
              AND iv.starttime < e.intime + INTERVAL '{PRED_END_H}' HOUR
              AND COALESCE(iv.endtime, iv.starttime) > e.intime + INTERVAL '{PRED_START_H}' HOUR)
          -- Fix 7: 220052 = invasive MAP (requires an arterial line), 220181 = non-invasive MAP.
          -- Invasive-only makes arterial-line placement a necessary condition for the
          -- label, which couples it to map_invasive_mask in the physiology stream.
          AND EXISTS (SELECT 1 FROM chartevents cv WHERE cv.stay_id = e.stay_id
              AND cv.itemid IN (220052, 220181) AND cv.valuenum < 65
              AND cv.charttime BETWEEN e.intime + INTERVAL '{PRED_START_H}' HOUR AND e.intime + INTERVAL '{PRED_END_H}' HOUR)
    )
    SELECT e.stay_id, e.subject_id, e.intime,
           -- ONGOING NEED: any vasopressor infusion overlapping h8-h14
           EXISTS (SELECT 1 FROM inputevents v WHERE v.stay_id = e.stay_id
               AND v.itemid IN (221906,221289,221662,221749,222315,227692)
               AND v.starttime < e.intime + INTERVAL '{PRED_END_H}' HOUR
               AND COALESCE(v.endtime, v.starttime) > e.intime + INTERVAL '{PRED_START_H}' HOUR
           ) AS label_vasopressor,
           -- ONGOING NEED, INVASIVE ventilation only.
           -- procedureevents: 225792 Invasive Ventilation, 224385 Intubation.
           -- chartevents: 224684 Tidal Volume (set) and 223849/220339 are set only
           -- under positive-pressure ventilation; 224686 (spontaneous TV) is
           -- excluded because it is also charted under BiPAP/CPAP.
           (EXISTS (SELECT 1 FROM procedureevents ve WHERE ve.stay_id = e.stay_id
               AND ve.itemid IN (225792, 224385)
               AND ve.starttime < e.intime + INTERVAL '{PRED_END_H}' HOUR
               AND COALESCE(ve.endtime, ve.starttime) > e.intime + INTERVAL '{PRED_START_H}' HOUR)
            OR EXISTS (SELECT 1 FROM chartevents cv WHERE cv.stay_id = e.stay_id
               AND cv.itemid IN (224684, 224685)
               AND cv.valuenum > 0
               AND cv.charttime BETWEEN e.intime + INTERVAL '{PRED_START_H}' HOUR
                                                  AND e.intime + INTERVAL '{PRED_END_H}' HOUR)
           ) AS label_intubation,
           -- Diagnostic: how many positives came only from the NIV-ambiguous route?
           EXISTS (SELECT 1 FROM chartevents cv WHERE cv.stay_id = e.stay_id
               AND cv.itemid IN (224686, 220339) AND cv.valuenum > 0
               AND cv.charttime BETWEEN e.intime + INTERVAL '{PRED_START_H}' HOUR
                                                  AND e.intime + INTERVAL '{PRED_END_H}' HOUR)
           AS label_intubation_niv_ambiguous,
           EXISTS (SELECT 1 FROM septic ss WHERE ss.stay_id = e.stay_id) AS label_septic_shock
    FROM eligible e
    ORDER BY e.intime
""").pl()


full_labels = full_labels.join(
    demo_df.select(["stay_id","anchor_year_group","anchor_year"]).unique("stay_id"), on="stay_id", how="left")
full_labels.write_parquet("./mimiciv_full_labels.parquet")
print(f"Labels → {full_labels.shape}")
for col in ["label_vasopressor","label_intubation","label_septic_shock"]:
    print(f"  {col}: {full_labels[col].mean():.4f} ({int(full_labels[col].sum())} positives)")

# Fix 7: label_septic_shock uses the same vasopressor predicate as label_vasopressor,
# so shock is a strict subset of vaso by construction. Quantify and report this.
_nest = full_labels.select([
    (pl.col("label_septic_shock") & ~pl.col("label_vasopressor")).sum().alias("shock_not_vaso"),
    pl.col("label_septic_shock").sum().alias("shock_total"),
    pl.col("label_vasopressor").sum().alias("vaso_total"),
])
print("Label nesting audit:", _nest)
assert _nest["shock_not_vaso"][0] == 0, \
    "septic shock is no longer a strict subset of vasopressor — update the paper text"
    
# ── JOIN TIME-SERIES ───────────────────────────────────────────────────────────
vitals = pl.read_parquet("./mimiciv_hourly_vitals_clean.parquet")
labs   = pl.read_parquet("./mimiciv_labs_hourly.parquet")
interv = pl.read_parquet("./mimiciv_interventions.parquet")
urine  = pl.read_parquet("./mimiciv_urine_hourly_final.parquet")

ts = (vitals.join(labs, on=["stay_id","hrs_from_admit"], how="left")
      .join(interv, on=["stay_id","hrs_from_admit"], how="left")
      .join(urine.rename({"urine_output_ml":"urine_output"}), on=["stay_id","hrs_from_admit"], how="left")
      .join(demo_df, on="stay_id", how="left"))
ts = ts.filter(pl.col("hrs_from_admit") <= OBS_HOURS - 1)

weights = con.execute(f"""
    SELECT i.stay_id, AVG(c.valuenum) AS weight
    FROM chartevents c JOIN icustays i ON c.stay_id = i.stay_id
    WHERE c.itemid IN (226512,224639) AND c.valuenum > 20
      AND c.charttime >= i.intime AND c.charttime < i.intime + INTERVAL '{OBS_HOURS}' HOUR
    GROUP BY i.stay_id
""").pl()
ts = ts.join(weights, on="stay_id", how="left")
ts = ts.with_columns((pl.col("urine_output") / pl.col("weight").fill_null(70.0)).alias("urine_output_ml_kg_hr"))

for col in ["heart_rate","map_invasive","lactate"]:
    ts = ts.sort(["stay_id","hrs_from_admit"]).with_columns(
        (pl.col("hrs_from_admit") -
         pl.when(pl.col(col).is_not_null()).then(pl.col("hrs_from_admit")).otherwise(None)
           .forward_fill().over("stay_id")).fill_null(pl.col("hrs_from_admit")).alias(f"{col}_time_delta"))

feature_cols = [
    "heart_rate","sbp_noninvasive","dbp_noninvasive","sbp_invasive","dbp_invasive",
    "map_invasive","temperature_c","spo2","resp_rate",
    "creatinine","wbc","platelets","lactate","bun","bilirubin_total","glucose",
    "hematocrit","potassium","sodium","troponin_t","ph_venous","pco2_venous",
    "base_excess","rbc","chloride","calcium",
    "urine_output","urine_output_ml_kg_hr","weight",
    "heart_rate_time_delta","map_invasive_time_delta","lactate_time_delta",
    "gcs_eye", "gcs_verbal", "gcs_motor",
]

for col in feature_cols:
    if col in ts.columns:
        ts = ts.with_columns(pl.col(col).is_not_null().cast(pl.Int8).alias(f"{col}_mask"))
ts = ts.with_columns(
    pl.when((pl.col("urine_output") == 0) & (pl.col("urine_output_mask") == 0))
      .then(None).otherwise(pl.col("urine_output")).alias("urine_output"))

ts = ts.join(full_labels.drop("intime"), on="stay_id", how="left")
excl = con.execute(f"""
    SELECT i.stay_id,
           CASE WHEN a.deathtime IS NOT NULL AND a.deathtime <= i.intime + INTERVAL '{PRED_END_H}' HOUR THEN 1 ELSE 0 END AS died_early,
           CASE WHEN i.outtime <= i.intime + INTERVAL '{PRED_END_H}' HOUR THEN 1 ELSE 0 END AS left_early
    FROM icustays i JOIN admissions a ON i.hadm_id = a.hadm_id
""").pl()
ts = (ts.join(excl, on="stay_id", how="left")
        .filter((pl.col("died_early") == 0) & (pl.col("left_early") == 0))
        .drop(["died_early","left_early"]))
ts = ts.filter(
    (pl.col("subject_id") > 0) & pl.col("subject_id").is_not_null() &
    (pl.col("anchor_year_group") != "UNKNOWN") & pl.col("anchor_year_group").is_not_null() &
    pl.col("label_vasopressor").is_not_null())
print(f"After exclusions: {ts['stay_id'].n_unique():,} stays")

# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL SPLIT — computed HERE, before imputation.
#
# Previously the split was computed at the very end, while the imputation
# medians above were taken over `anchor_year_group IN TRAIN_YEARS`. That is the
# train ERA, which contains the validation patients: val is carved out of the
# train eras. So val rows contributed to the medians that were then imputed into
# train rows. The effect is small on a median over ~100k rows, but it is
# avoidable leakage and a reviewer will ask. Splitting first lets the medians be
# computed over train_ids only.
#
# MIMIC-IV shifts dates PER PATIENT, so raw `intime` is not comparable across
# subjects and cannot order stays chronologically. approx_year is the finest
# genuinely comparable resolution (1 year). Within a year, ties break on a fixed
# seed. Describe this as "chronological to year resolution", not "strictly
# chronological".
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Temporal Split (computed pre-imputation) ---")
_rng = np.random.default_rng(42)

_stay_level = ts.select(["stay_id", "subject_id", "anchor_year_group",
                         "approx_year"]).unique(subset="stay_id")

_train_era = _stay_level.filter(pl.col("anchor_year_group").is_in(TRAIN_YEARS))
_train_era = (_train_era
              .with_columns(pl.Series("tie_break", _rng.random(_train_era.height)))
              .sort(["approx_year", "tie_break"]))

_all_train_stays = _train_era["stay_id"].to_list()
_cutoff = int(len(_all_train_stays) * 0.80)
train_ids = set(_all_train_stays[:_cutoff])
val_ids   = set(_all_train_stays[_cutoff:])
test_ids  = set(_stay_level.filter(~pl.col("anchor_year_group").is_in(TRAIN_YEARS))["stay_id"])

print("Train era year distribution:")
print(_train_era.head(_cutoff)["approx_year"].value_counts().sort("approx_year"))
print("Val era year distribution:")
print(_train_era.tail(len(_all_train_stays) - _cutoff)["approx_year"].value_counts().sort("approx_year"))

assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)

# Subject-level purge: no human may appear in more than one split.
train_subjects = set(_stay_level.filter(pl.col("stay_id").is_in(list(train_ids)))["subject_id"])
val_ids  = set(_stay_level.filter(pl.col("stay_id").is_in(list(val_ids))
                                  & ~pl.col("subject_id").is_in(list(train_subjects)))["stay_id"])
val_subjects = set(_stay_level.filter(pl.col("stay_id").is_in(list(val_ids)))["subject_id"])
test_ids = set(_stay_level.filter(pl.col("stay_id").is_in(list(test_ids))
                                  & ~pl.col("subject_id").is_in(list(train_subjects))
                                  & ~pl.col("subject_id").is_in(list(val_subjects)))["stay_id"])
print(f"✅ Zero leakage | train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} stays")

# ── DENSE GRID ─────────────────────────────────────────────────────────────────
eligible = ts.select("stay_id").unique()
ts = eligible.join(pl.DataFrame({"hrs_from_admit": list(range(OBS_HOURS))}), how="cross"
    ).join(ts, on=["stay_id","hrs_from_admit"], how="left")

ts = ts.drop(["age","gender","ethnicity","anchor_year_group","anchor_year","subject_id",
              "approx_year","anchor_year_group_raw"], strict=False)
ts = ts.join(demo_df.select(["stay_id","age","gender","ethnicity","anchor_year",
                             "length_of_stay","mortality","approx_year",
                             "anchor_year_group_raw"]).unique("stay_id"),
             on="stay_id", how="left")
ts = ts.join(full_labels.select(["stay_id","subject_id","anchor_year_group"]).unique("stay_id"),
             on="stay_id", how="left")

# Prevalent-treatment metadata — joined AFTER the dense grid so every hour row of
# a stay carries the stay-level flag. Evaluation strata only; script2/script3
# assert these never enter SEQ_FEATURES or TREATMENT_FEATURES.
ts = ts.join(prevalent_df, on="stay_id", how="left").with_columns(
    [pl.col(c).fill_null(0).cast(pl.Int8)
     for c in ["already_vaso", "already_vent", "endotrach_obs"]])

label_cols = [c for c in full_labels.columns
              if c.startswith("label_") and c != "label_intubation_niv_ambiguous"]
for lc in label_cols:
    if lc not in ts.columns:
        ts = ts.join(full_labels.select(["stay_id", lc]).unique("stay_id"), on="stay_id", how="left")
    ts = ts.sort(["stay_id","hrs_from_admit"]).with_columns(pl.col(lc).forward_fill().over("stay_id"))
    ts = ts.with_columns(pl.col(lc).backward_fill().over("stay_id"))

ts = ts.join(treatment_df, on="stay_id", how="left")
for c in TREATMENT_COLS:
    if c not in ts.columns:
        ts = ts.with_columns(pl.lit(0).alias(c))

# ── IMPUTATION ─────────────────────────────────────────────────────────────────
for col in feature_cols:
    if col in ts.columns:
        ts = ts.with_columns(pl.col(col).is_not_null().cast(pl.Int8).alias(f"{col}_mask"))

# Medians over the TRAIN SPLIT, not the train ERA — the era also contains val.
train_med = (ts.filter(pl.col("stay_id").is_in(list(train_ids)))
               .select([c for c in feature_cols if c in ts.columns]).median())
ts = ts.sort(["stay_id","hrs_from_admit"]).with_columns(
    [pl.col(c).forward_fill().over("stay_id") for c in feature_cols if c in ts.columns]
).with_columns(
    [pl.col(c).fill_null(train_med[c][0])
     for c in feature_cols if c in train_med.columns and train_med[c][0] is not None])

baseline_cols = ["creatinine","lactate","bun","glucose","bilirubin_total"]
for col in baseline_cols:
    base = (ts.filter(pl.col(f"{col}_mask") == 1)
              .group_by("stay_id").agg(pl.col(col).first().alias(f"{col}_baseline")))
    ts = ts.join(base, on="stay_id", how="left")
    fallback = train_med[col][0] if col in train_med.columns else None
    if fallback is not None:
        ts = ts.with_columns(pl.col(f"{col}_baseline").fill_null(fallback))
    ts = ts.with_columns([
        (pl.col(col) - pl.col(f"{col}_baseline")).alias(f"{col}_delta"),
        pl.when(pl.col(f"{col}_baseline") == 0).then(1.0)
          .otherwise(pl.col(col) / pl.col(f"{col}_baseline")).alias(f"{col}_ratio")
    ]).with_columns(
        pl.when(pl.col(f"{col}_ratio").is_infinite() | pl.col(f"{col}_ratio").is_nan())
          .then(1.0).when(pl.col(f"{col}_ratio") > 10).then(10.0)
          .otherwise(pl.col(f"{col}_ratio")).alias(f"{col}_ratio"))
    ts = ts.with_columns([
        pl.col(f"{col}_mask").cast(pl.Int8).alias(f"{col}_{s}_mask") for s in ["baseline","delta","ratio"]])

ts = ts.sort(["stay_id","hrs_from_admit"]).with_columns([
    pl.col("resp_rate").rolling_mean(3, min_samples=1).over("stay_id").alias("resp_rate_rollmean_3"),
    pl.col("resp_rate").rolling_std(3, min_samples=1).over("stay_id").alias("resp_rate_rollstd_3"),
    pl.col("spo2").rolling_mean(6, min_samples=1).over("stay_id").alias("spo2_rollmean_6"),
    pl.col("spo2").rolling_std(4, min_samples=1).over("stay_id").alias("spo2_rollstd_4"),
]).with_columns([
    pl.col("resp_rate_rollmean_3").fill_null(0), pl.col("resp_rate_rollstd_3").fill_null(0),
    pl.col("spo2_rollmean_6").fill_null(train_med["spo2"][0] if "spo2" in train_med.columns else 96.0),
    pl.col("spo2_rollstd_4").fill_null(0)])

train_age_med = ts.filter(pl.col("stay_id").is_in(list(train_ids)))["age"].drop_nulls().median()
urine_med = (ts.filter(pl.col("stay_id").is_in(list(train_ids)) &
                       pl.col("urine_output").is_not_null() & (pl.col("urine_output") > 0))["urine_output"].median())

ts = ts.with_columns(pl.col("ethnicity").str.to_uppercase().alias("eth_tmp")).with_columns(
    pl.when(pl.col("eth_tmp").str.contains("WHITE")).then(pl.lit("WHITE"))
    .when(pl.col("eth_tmp").str.contains("BLACK|AFRICAN")).then(pl.lit("BLACK"))
    .when(pl.col("eth_tmp").str.contains("HISPANIC|LATINO")).then(pl.lit("HISPANIC"))
    .when(pl.col("eth_tmp").str.contains("ASIAN")).then(pl.lit("ASIAN"))
    .otherwise(pl.lit("OTHER/UNKNOWN")).alias("ethnicity")).drop("eth_tmp")

ts = ts.with_columns([
    pl.col("urine_output").fill_null(urine_med), pl.col("age").fill_null(train_age_med),
    pl.col("gender").fill_null("UNKNOWN"), pl.col("ethnicity").fill_null("OTHER/UNKNOWN"),
    pl.col("anchor_year_group").fill_null("UNKNOWN")])

# Fix 9: std() is null when n == 1, so the original condition also removed every stay
# with exactly one charted HR — sparsely monitored patients whose prevalence varies
# by era. Only flag genuine recording artefacts: >= 2 observations, zero variance.
_hr = (ts.filter(pl.col("heart_rate_mask") == 1)
         .group_by("stay_id")
         .agg([pl.col("heart_rate").std().alias("hr_std"),
               pl.col("heart_rate").count().alias("hr_n")]))
zombie_ids = _hr.filter((pl.col("hr_n") >= 2) & (pl.col("hr_std") == 0)).select("stay_id")
_sparse    = _hr.filter(pl.col("hr_n") < 2).height
print(f"Removing {zombie_ids.height} zombie stays (zero HR variance, n>=2); "
      f"retaining {_sparse} stays with a single HR observation")
ts = ts.join(zombie_ids, on="stay_id", how="anti")

ts = ts.filter(pl.col(label_cols[0]).is_not_null())
ts = ts.with_columns([pl.col(c).cast(pl.Int8) for c in label_cols])
ts = ts.with_columns(pl.col("hrs_from_admit").alias("time_idx"))
ts = ts.fill_null(0)

# Re-attach intime for strict chronological sorting in Script 2
intime_df = con.execute("SELECT stay_id, intime FROM icustays").pl()
ts = ts.join(intime_df.unique("stay_id"), on="stay_id", how="left")

ordered = (
    ["stay_id","intime","time_idx","hrs_from_admit"] +
    ["heart_rate","sbp_noninvasive","dbp_noninvasive","sbp_invasive","dbp_invasive",
     "map_invasive","temperature_c","spo2","resp_rate", "gcs_eye", "gcs_verbal", "gcs_motor"] +
    ["creatinine","wbc","platelets","lactate","bun","bilirubin_total","glucose",
     "hematocrit","potassium","sodium","troponin_t","ph_venous","pco2_venous",
     "base_excess","rbc","chloride","calcium"] +
    ["urine_output","urine_output_ml_kg_hr","weight","vasopressor_flag","ventilation_flag",
     # Evaluation metadata — carried through to the parquet, never a model input.
     "already_vaso","already_vent","endotrach_obs",
     "age","gender","ethnicity", "length_of_stay", "mortality"] + TREATMENT_COLS +
    [f"{c}_{s}" for c in baseline_cols for s in ["baseline","delta","ratio"]] +
    ["heart_rate_time_delta","map_invasive_time_delta","lactate_time_delta",
     "resp_rate_rollmean_3","resp_rate_rollstd_3","spo2_rollmean_6","spo2_rollstd_4"] +
    [c for c in ts.columns if c.endswith("_mask")] + label_cols +
    ["anchor_year_group","anchor_year_group_raw","approx_year","anchor_year","subject_id"]
)
final_df = ts.select([c for c in ordered if c in ts.columns])

# ── MATERIALISE THE SPLITS ─────────────────────────────────────────────────────
# train_ids / val_ids / test_ids were computed earlier, before imputation, so the
# medians above saw only training patients. Subject purging already applied.
print("\n--- Materialising splits ---")

train_final = final_df.filter(pl.col("stay_id").is_in(list(train_ids)))
val_final   = final_df.filter(pl.col("stay_id").is_in(list(val_ids)))
test_final  = final_df.filter(pl.col("stay_id").is_in(list(test_ids)))

# Re-assert subject disjointness on the materialised frames (stays were dropped
# by the zombie and label filters after the ids were chosen).
_tr_subj = set(train_final["subject_id"].unique().to_list())
_va_subj = set(val_final["subject_id"].unique().to_list())
_te_subj = set(test_final["subject_id"].unique().to_list())
assert not (_tr_subj & _va_subj), "subject overlap train/val"
assert not (_tr_subj & _te_subj), "subject overlap train/test"
assert not (_va_subj & _te_subj), "subject overlap val/test"
print("✅ Subject-level disjointness confirmed across all three splits")

for name, df in [("Train", train_final), ("Val", val_final), ("Test", test_final)]:
    print(f"{name}: {df['stay_id'].n_unique():,} stays | "
          f"years: {sorted(df['anchor_year_group'].unique().to_list())}")

# ── PREVALENT / INCIDENT AUDIT ─────────────────────────────────────────────────
# The single table that answers "is this model predicting new need, or detecting
# ongoing treatment?". If frac_pos_prevalent is high, the headline AUROC is
# mostly continuation detection and the incident-cohort AUROC is the number that
# belongs in the abstract.
print("\n--- Incident vs prevalent cohort audit (stay level) ---")
_gate = {"label_vasopressor": "already_vaso",
         "label_intubation": "already_vent",
         "label_septic_shock": "already_vaso"}
prevalence_audit = {}
for name, df in [("Train", train_final), ("Val", val_final), ("Test", test_final)]:
    if df.height == 0:
        continue
    # _gate maps three labels onto only TWO distinct gate columns (vasopressor and
    # septic shock share already_vaso), so the naive concatenation asks polars to
    # aggregate already_vaso twice and it raises a duplicate-output-name error.
    _audit_cols = list(dict.fromkeys(list(_gate.values()) + list(_gate.keys())))
    sl = df.group_by("stay_id").agg([pl.col(c).max() for c in _audit_cols])
    prevalence_audit[name] = {}
    print(f"\n  {name}:")
    for lbl, gate in _gate.items():
        inc = sl.filter(pl.col(gate) == 0)
        prev = sl.filter(pl.col(gate) == 1)
        n_pos = int(sl[lbl].sum())
        rec = {
            "gate": gate,
            "n_incident": inc.height, "n_prevalent": prev.height,
            "n_pos_incident": int(inc[lbl].sum()) if inc.height else 0,
            "n_pos_prevalent": int(prev[lbl].sum()) if prev.height else 0,
            "frac_pos_prevalent": (float(prev[lbl].sum()) / n_pos) if n_pos else None,
            "incident_prevalence": float(inc[lbl].mean()) if inc.height else None,
            "prevalent_prevalence": float(prev[lbl].mean()) if prev.height else None,
        }
        prevalence_audit[name][lbl] = rec
        print(f"    {lbl.replace('label_',''):<16} "
              f"incident n={rec['n_incident']:>6} pos={rec['n_pos_incident']:>5} "
              f"({100*(rec['incident_prevalence'] or 0):.1f}%) | "
              f"prevalent n={rec['n_prevalent']:>6} pos={rec['n_pos_prevalent']:>5} "
              f"({100*(rec['prevalent_prevalence'] or 0):.1f}%) | "
              f"{100*(rec['frac_pos_prevalent'] or 0):.1f}% of positives are prevalent")


def validate_and_save(df, name, path, lbl_cols):
    if df.height == 0:
        print(f"⚠ {name} empty"); return
    stay_lab = df.group_by("stay_id").agg([pl.col(c).max() for c in lbl_cols])
    n = stay_lab.height
    print(f"\n{name}: {n} stays", end=" |")
    for col in lbl_cols:
        n_pos = stay_lab.filter(pl.col(col) == 1).height
        print(f" {col.replace('label_','')}={100*n_pos/n:.1f}%", end="")
    print()
    df.write_parquet(path); print(f"  → {path}")


label_cols_final = [c for c in final_df.columns if c.startswith("label_")]
validate_and_save(train_final, "TRAIN", "./train_final_enriched.parquet", label_cols_final)
validate_and_save(val_final,   "VAL",   "./val_final_enriched.parquet",   label_cols_final)
validate_and_save(test_final,  "TEST",  "./test_final_enriched.parquet",  label_cols_final)

meta = {
    "treatment_cols": TREATMENT_COLS,
    "obs_hours": OBS_HOURS,
    "seq_len": SEQ_LEN,
    "gap_hours": GAP_HOURS,
    "pred_window": [PRED_START_H, PRED_END_H],
    "metadata_cols": ["already_vaso", "already_vent", "endotrach_obs",
                      "vasopressor_flag", "ventilation_flag"],
    "prevalence_audit": prevalence_audit,
}
with open("./feature_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"\n✅ Done. Ongoing-need labels. All {len(TREATMENT_COLS)} treatment features retained.")
print("No patient exclusions for already_vaso/already_vent — they are carried as")
print("evaluation metadata so incident and prevalent cohorts can be reported separately.")

