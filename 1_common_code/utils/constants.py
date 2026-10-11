"""
constants.py — single source of truth for feature lists and hyperparameters.

Nothing else in the codebase may redefine SEED / SEQ_LEN / HIDDEN_DIM / ... .
architectures.py, data_utils.py, script2 and script3 all import from here.

Three column families, kept strictly disjoint:

  SEQ_FEATURES       physiology, fed to the LSTM stream (per-hour, h0..h5)
  TREATMENT_FEATURES care-process + admission context, fed to the MLP stream
  METADATA_COLS      NEVER a model input. Used only to stratify evaluation and
                     to audit leakage. `already_vaso` / `already_vent` in
                     particular exist so that "incident" (new need) and
                     "prevalent" (treatment already running at h5) cohorts can
                     be reported separately under the ongoing-need label
                     framing, without excluding anyone from training.
"""

# ── HYPERPARAMETERS ────────────────────────────────────────────────────────────
SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM = 42, 6, 64, 32
LSTM_LAYERS, BATCH_SIZE, DROPOUT = 2, 64, 0.3
LR_INIT = 1e-3
LR_ADAPT = 3e-4                      # default only; each arm sweeps its own
LR_ADAPT_GRID = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]   # CLEAN-V2: widened; C hit the 1e-4 edge and C_l2sp the 1e-3 edge in clean_full
EPOCHS, ADAPT_EPOCHS = 50, 40
PATIENCE, ADAPT_PATIENCE = 8, 8
BUFFER_SIZE, PSI_THRESH = 500, 0.20
L2SP_LAMBDA = 1e-2                   # Run C-L2SP anchor strength

# ── RUN BUDGET ─────────────────────────────────────────────────────────────────
# FAST_MODE trades precision for turnaround. Nothing it changes can flip a
# direction — it only widens intervals and coarsens the seed median. Flip to
# False for the final run that goes in the paper, and say which was used.
#   seeds 5→3     ~40% less training; the median-of-3 is still a median
#   N_BOOT 1000→400  BCa is still valid, CI edges move by ~±0.002 AUROC
#   NULL_DRAWS 200→100  threshold moves in the third decimal
#   LR grid 3→2   drops 1e-4, which never won a sweep in any logged run
FAST_MODE = False

SOURCE_SEEDS = [42, 123, 7] if FAST_MODE else [42, 123, 7, 2024, 99]
ADAPT_SEEDS  = [42, 123, 7] if FAST_MODE else [42, 123, 7, 2024, 99]
N_BOOT       = 400 if FAST_MODE else 1000
NULL_DRAWS   = 100 if FAST_MODE else 200
if FAST_MODE:
    LR_ADAPT_GRID = [3e-4, 1e-3]

# ── LABELS ─────────────────────────────────────────────────────────────────────
# Septic shock is dropped from training and evaluation, for two reasons that were
# both true before any result was seen:
#   1. It is a STRICT SUBSET of vasopressor by construction (asserted in
#      script1), so it is not an independent task — it double-counts.
#   2. ~20 incident events in the final era. Every sensitivity is a multiple of
#      0.05 and no comparison on it is interpretable at any sample size we have.
# It is still computed descriptively in script1 and its event count is reported,
# so the exclusion is visible rather than silent. State the reason in the paper
# BEFORE presenting results on the other two — a label dropped after the fact
# reads as cherry-picking; one dropped on event count does not.
DROP_UNDERPOWERED_LABELS = True

ALL_LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]
UNDERPOWERED_LABELS = ["label_septic_shock"]

# ── BIOLOGICAL-LABEL MODE ─────────────────────────────────────────────────────
# The default targets are INTERVENTIONAL: a clinician decides whether to start a
# vasopressor or intubate, so the label co-moves with the practice drift being
# studied and P(y | physiology) is non-stationary by construction. That works
# against any freeze premise resting on physiological stability, and it is a
# property of the TASK, not of the architecture.
#
# BIOLOGICAL_LABEL swaps in in-hospital mortality, which is an outcome rather
# than a decision. `label_mortality` is derived in data_utils.load_enriched_split
# from a column script1 already writes, so nothing is re-extracted. Two things
# change and both must be reported:
#   * horizon — mortality is whole-stay, not the h8..h14 window, so the task is
#     "predict in-hospital death from the first six hours".
#   * cohort  — script1 already excludes patients who die within h14, so the
#     earliest and most trivially predictable deaths are absent.
# Prevalence is roughly 10% against 1.5-3.7% for the interventional labels, so
# event counts rise about sixfold and a strict chronological split becomes
# affordable for the first time.
BIOLOGICAL_LABEL = False

# Mortality has no "already receiving it" concept, so it is ungated. None is the
# supported sentinel: cohort_masks() short-circuits on a falsy gate, script2
# guards with `INCIDENT_GATE[lbl] in ds.meta`, and the analysis scripts fall back
# to an all-rows mask when the gate array is absent from the npz.
if BIOLOGICAL_LABEL:
    LABEL_COLS = ["label_mortality"]
else:
    LABEL_COLS = ([l for l in ALL_LABEL_COLS if l not in UNDERPOWERED_LABELS]
                  if DROP_UNDERPOWERED_LABELS else ALL_LABEL_COLS)

# ── PHYSIOLOGY STREAM ──────────────────────────────────────────────────────────
_SEQ_CORE = [
    "heart_rate","sbp_noninvasive","dbp_noninvasive","sbp_invasive","dbp_invasive",
    "map_invasive","temperature_c","spo2","resp_rate",
    "creatinine","wbc","platelets","lactate","bun","bilirubin_total","glucose",
    "hematocrit","potassium","sodium","troponin_t","ph_venous","pco2_venous",
    "base_excess","rbc","chloride","calcium",
    "urine_output","urine_output_ml_kg_hr","weight",
    "heart_rate_time_delta","map_invasive_time_delta","lactate_time_delta",
    "creatinine_baseline","creatinine_delta","creatinine_ratio",
    "lactate_baseline","lactate_delta","lactate_ratio",
    "bun_baseline","bun_delta","bun_ratio",
    "glucose_baseline","glucose_delta","glucose_ratio",
    "bilirubin_total_baseline","bilirubin_total_delta","bilirubin_total_ratio",
    "resp_rate_rollmean_3","resp_rate_rollstd_3","spo2_rollmean_6","spo2_rollstd_4",
    "gcs_eye", "gcs_verbal", "gcs_motor",
    "gcs_eye_mask", "gcs_verbal_mask", "gcs_motor_mask",
]

_SEQ_MASKS = [
    "heart_rate_mask","sbp_noninvasive_mask","dbp_noninvasive_mask",
    "sbp_invasive_mask","dbp_invasive_mask","map_invasive_mask",
    "temperature_c_mask","spo2_mask","resp_rate_mask",
    "creatinine_mask","wbc_mask","platelets_mask","lactate_mask","bun_mask",
    "bilirubin_total_mask","glucose_mask","hematocrit_mask","potassium_mask",
    "sodium_mask","troponin_t_mask","ph_venous_mask","pco2_venous_mask",
    "base_excess_mask","rbc_mask","chloride_mask","calcium_mask",
    "urine_output_mask","urine_output_ml_kg_hr_mask","weight_mask",
]

# ── CARE-PROCESS LEAKAGE ABLATION ──────────────────────────────────────────────
# Grouped, not flat. Dropping an observation mask without dropping the value it
# masks does not remove the signal: absent values are median-imputed, so a
# constant-valued channel still reconstructs "no arterial line". Each group is
# therefore value + mask + derived, ablated together or not at all.
LEAKY_PHYSIO_GROUPS = {
    # 'No Response-ETT' is held out of gcs_verbal in script1, which makes
    # gcs_verbal_mask == 0 read partly as "already intubated".
    "gcs_verbal_ett": [
        "gcs_verbal", "gcs_verbal_mask",
    ],
    # Invasive pressures are only charted when an arterial line is in place.
    "arterial_line": [
        "map_invasive", "sbp_invasive", "dbp_invasive",
        "map_invasive_mask", "sbp_invasive_mask", "dbp_invasive_mask",
        "map_invasive_time_delta",
    ],
}

LEAKY_PHYSIO_CANDIDATES = sorted({c for g in LEAKY_PHYSIO_GROUPS.values() for c in g})

# Flip to True to build the ablated feature set. Report BOTH arms in the paper.
ABLATE_LEAKY_PHYSIO = False


# ── WHERE DO OBSERVATION MASKS BELONG? ─────────────────────────────────────────
# A `*_mask` records whether a value was CHARTED — whether a lab was ordered,
# whether an arterial line was placed, whether GCS was assessable. That is a
# clinician's decision, not a physiological measurement. Likewise
# `*_time_delta` is time since someone last took a reading.
#
# Putting them in the physiology stream contradicts the decomposition this work
# is built on: the physiology stream is supposed to hold the part that is stable
# because human biology is stable, and ordering practice is not stable. With the
# masks included, 35 of 86 physiology channels (41%) were care process, and
# freezing that stream froze care-process encoding along with biology.
#
# They are not discarded — they are AGGREGATED to a per-stay observation rate
# (fraction of h0..h5 in which the value was charted) and moved to the treatment
# stream, where measurement decisions belong. The information is preserved and
# the hourly resolution is traded for a correct stream assignment.
#
# Set False to reproduce the original configuration. REPORT BOTH: "masks in
# physiology" vs "masks as care process" is a finding about what makes a
# representation freezable, and it is stronger than either result alone.
MASKS_AS_CARE_PROCESS = True

_TIME_DELTA_COLS = ["heart_rate_time_delta", "map_invasive_time_delta",
                    "lactate_time_delta"]
_GCS_MASKS = ["gcs_eye_mask", "gcs_verbal_mask", "gcs_motor_mask"]

# Every mask column that exists in the parquet.
MASK_COLS = _GCS_MASKS + _SEQ_MASKS
CARE_PROCESS_SOURCE_COLS = MASK_COLS + _TIME_DELTA_COLS

# Aggregated names added to the treatment stream. `x_mask` -> `x_obs_rate`.
CARE_PROCESS_FEATURES = ([f"{c[:-5]}_obs_rate" for c in MASK_COLS]
                         + [f"{c}_mean" for c in _TIME_DELTA_COLS])


def build_seq_features(ablate: bool = ABLATE_LEAKY_PHYSIO,
                       masks_as_care: bool = MASKS_AS_CARE_PROCESS) -> list:
    feats = _SEQ_CORE + _SEQ_MASKS
    if masks_as_care:
        drop = set(CARE_PROCESS_SOURCE_COLS)
        feats = [c for c in feats if c not in drop]
    if ablate:
        feats = [c for c in feats if c not in set(LEAKY_PHYSIO_CANDIDATES)]
    return feats


SEQ_FEATURES = build_seq_features()

# ── TREATMENT STREAM ───────────────────────────────────────────────────────────
TREATMENT_FEATURES = [
    # Fluid & Resuscitation (h0-h5)
    "total_crystalloid_ml",
    "has_blood_products_obs",
    "total_prbc_ml",
    "has_albumin_obs",
    # Medications & Orders (h0-h5)
    "early_steroid",
    "early_antibiotic",
    "early_furosemide",
    "early_anticoagulant",
    "early_antihypertensive",
    "early_antiarrhythmic",
    "n_distinct_meds",
    "steroid_ordered",
    "time_to_first_abx_order_hrs",
    "has_insulin_infusion_obs",
    # Organ Support & Line Placement (h0-h5)
    "has_rrt_obs",
    "has_arterial_line_obs",
    "has_central_line_obs",
    # Oxygen Therapy Intensity (h0-h5)
    "max_fio2_obs",
    "mean_fio2_obs",
    # Admission Context & Demographics (t=0)
    "age",
    "gender_M",
]

# Care-process observation rates join the treatment stream — measurement
# decisions sit with the other clinician decisions.
if MASKS_AS_CARE_PROCESS:
    TREATMENT_FEATURES = TREATMENT_FEATURES + CARE_PROCESS_FEATURES

# ── DROP-eMAR SENSITIVITY (2026-10-07) ────────────────────────────────────────
# The seven hosp/emar features are ~0 in every training-era stay (eMAR rolled out
# after the training years), so the network cannot learn them and they switch on
# only at test time. This copy removes them; everything else matches
# ml4h_corrected (scaling fix, pinned cohorts, submitted LR grid).
EMAR_FEATURES = ["early_steroid", "early_antibiotic", "early_furosemide",
                 "early_anticoagulant", "early_antihypertensive",
                 "early_antiarrhythmic", "n_distinct_meds"]
TREATMENT_FEATURES = [c for c in TREATMENT_FEATURES if c not in EMAR_FEATURES]

BINARY_COLS = {
    "early_steroid",
    "early_antibiotic",
    "early_furosemide",
    "early_anticoagulant",
    "early_antihypertensive",
    "early_antiarrhythmic",
    "steroid_ordered",
    "has_blood_products_obs",
    "has_albumin_obs",
    "has_rrt_obs",
    "has_insulin_infusion_obs",
    "has_arterial_line_obs",
    "has_central_line_obs",
    "gender_M",
}

# ── METADATA — evaluation strata only, never model inputs ──────────────────────
# already_vaso / already_vent are computed in script1 with INTERVAL OVERLAP
# semantics against [intime, intime + OBS_HOURS): an infusion or vent episode
# that began before ICU admission and is still running at h0 counts as
# prevalent. Using starttime alone would miss exactly the population that makes
# the ongoing-need label trivially predictable.
METADATA_COLS = [
    "already_vaso",        # vasopressor running at any point in h0..h5
    "already_vent",        # invasive ventilation running at any point in h0..h5
    "endotrach_obs",       # 'No Response-ETT' charted in h0..h5
    "vasopressor_flag",    # per-hour, kept for auditing only
    "ventilation_flag",
]

# Which metadata column defines the incident cohort for each label.
INCIDENT_GATE = {
    "label_vasopressor":  "already_vaso",
    "label_intubation":   "already_vent",
    "label_septic_shock": "already_vaso",   # shock requires vasopressors
    "label_mortality":    None,             # an outcome, not a treatment state
}

# ── INVARIANTS ─────────────────────────────────────────────────────────────────
_overlap = set(METADATA_COLS) & (set(SEQ_FEATURES) | set(TREATMENT_FEATURES))
assert not _overlap, f"metadata columns leaked into the feature set: {_overlap}"

_dupe = set(SEQ_FEATURES) & set(TREATMENT_FEATURES)
assert not _dupe, f"column appears in both streams: {_dupe}"

assert set(LABEL_COLS) <= set(INCIDENT_GATE), "INCIDENT_GATE must cover every active label"
assert LABEL_COLS, "every label was dropped"
print(f"[constants] FAST_MODE={FAST_MODE} | seeds={len(SOURCE_SEEDS)} N_BOOT={N_BOOT} "
      f"| labels={[l.replace('label_','') for l in LABEL_COLS]}"
      + (f"  (dropped: {[l.replace('label_','') for l in UNDERPOWERED_LABELS]})"
         if DROP_UNDERPOWERED_LABELS else ""))

# CLEAN-FULL: admission descriptors dropped (categories recoded across eras: loc_ed 65% to 0%, ewemer 68% to 45%)
assert not [c for c in ['admission_type_ewemer', 'admission_type_urgent', 'admission_type_elective', 'admission_loc_ed', 'admission_loc_transfer'] if c in TREATMENT_FEATURES]
