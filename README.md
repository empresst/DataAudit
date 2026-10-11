# Audit Before You Update: Recording Artifacts and the Limits of Bounded Model Updates in MIMIC-IV ICU Prediction

This repository contains the complete code, verification audits, aggregate evaluation logs, and manuscript sources for the paper:
> **Audit Before You Update: Recording Artifacts and the Limits of Bounded Model Updates in MIMIC-IV ICU Prediction**  
> *Fatema Ferdous Tamanna, K. M. Merajul Arefin, Md. Abdul Masud*

---

## 1. Overview & Change-Control Pipeline

Clinical predictive models in the Intensive Care Unit (ICU) undergo temporal degradation due to data-generating changes, documentation workflow migrations, and clinical practice evolution. This repository implements an end-to-end change-control pipeline consisting of:
1. **Pre-deployment Temporal Data Auditing (`0_data_and_audit/`)**: Systematic audits for recording artifacts in MIMIC-IV 3.1 (e.g., missing input/procedure tables in 2020–2022, eMAR availability onset during 2011–2013, and vasopressor ItemID migration).
2. **Label-Free Drift Detection (`3_drift_detection/`)**: Classifier two-sample tests (C2ST) with patient-clustered bootstrap calibration to gate model updates without waiting for downstream clinical labels.
3. **Bounded Parameter Updating (`4_adaptation/`)**: Retraining only treatment-context encoders and fusion heads (Arm B) versus full fine-tuning (Arm C) across neural networks (Two-Stream LSTM/MLP) and modular gradient-boosted trees (Modular XGBoost).
4. **Explanation & Literature Retrieval Stability (`5_explainability/`, `6_retrieval/`)**: Measuring whether internal attribution rankings (Integrated Gradients and TreeSHAP) and external knowledge retrieval (MedCPT/PubMed) remain consistent under adaptation.
5. **Matched-Hyperparameter Evaluation (`7_analysis_and_tables/`)**: Pre-specified non-inferiority margins ($\delta = 0.015$ AUROC) under shared vs. separately tuned learning rates.

---

## 2. Data Access & PhysioNet Compliance

> **IMPORTANT NOTICE REGARDING MIMIC-IV DATA:**  
> This study uses the [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) database. In accordance with the PhysioNet Credentialed Health Data Use Agreement (DUA), **no patient-level records, individual time-series parquets, patient embeddings, or model weight checkpoints are distributed in this public repository**.

### Obtaining the Data and Cohort Setup
1. **PhysioNet Credentialing**: Complete CITI training and apply for credentialed access to MIMIC-IV 3.1 on PhysioNet.
2. **Extraction & Feature Pipeline**:
   - Extract raw tables (`icustays`, `patients`, `admissions`, `inputevents`, `procedureevents`, `emar`) into a local directory (`raw_mimic/`).
   - Run the preprocessing pipeline (`0_data_and_audit/upstream_kaggle/script1_preprocessing.py`) to build initial hourly trajectory features.
3. **Running the Audit & Cleaning Protocols**:
   - Run `0_data_and_audit/make_flags.py` to identify stays with complete input/procedure records and extended vasopressor ItemIDs.
   - Run `0_data_and_audit/build_clean_data.py` to filter out unrecorded stays and apply the corrected label mappings.
4. **Cohort Pinning**:
   - Longitudinal split definitions pin patients temporally into Training (2008–2013), and sequential Adaptation/Evaluation eras (2014–2016, 2017–2019, 2020–2022). Due to PhysioNet policy regarding patient/stay identifiers, cohort split definitions (`pin_split_v2.json`) can be generated from the cleaned stay cohorts using reproducible random seeds specified in `utils/constants.py`.

---

## 3. Repository Structure

```text
.
├── 0_data_and_audit/          # Data audit scripts and verification evidence
│   ├── build_clean_data.py    # Applies audit exclusions and extended item mappings
│   ├── check_central_line_artifact.py # Central-line ItemID documentation shift audit
│   ├── check_emar_pipeline.py # DuckDB audit of eMAR availability onset
│   ├── check_emar_raw.py      # Raw inspection of early medication records
│   ├── make_flags.py          # Generates stay-level exclusion flags
│   ├── raw_check_icu_records_check.py # Verification of missing ICU records
│   ├── verify_emar_source_damage.py   # Audits impact of eMAR feature zeroing
│   ├── upstream_kaggle/       # Upstream preprocessing (script1_preprocessing.py)
│   └── evidence/              # Complete BigQuery SQL queries, results, and audit logs
│       ├── bigquery_audit_queries.sql
│       ├── bigquery_audit_results.md
│       ├── citation_check.md
│       ├── exclusion_balance_run_try3.log
│       ├── script2_split_audit.md
│       └── verify_audit_numbers.py
│
├── 1_common_code/             # Shared library modules
│   ├── models/                # TwoStreamModel architecture (architectures.py)
│   └── utils/                 # constants, data_utils, train_utils, eval_utils,
│                              # drift_utils, drift_explainability, rag_retrieval
│
├── 2_source_model/            # Base source model definition & training
│   ├── script2_two_stream_model.py # Source model training on 2008-2013 cohort
│   └── run_v2.sh              # Shell runner for source model baseline
│
├── 3_drift_detection/         # Label-free drift testing and gating
│   ├── script35_drift_detector.py # Classifier two-sample test (C2ST) with bootstrap
│   ├── script37_gated_schedule.py # Verification of update gating (0% false alarm)
│   ├── drift_detection_standard.json # Aggregate drift test statistics
│   └── gated_schedule.json    # Verified update trigger decisions
│
├── 4_adaptation/              # Model adaptation & hyperparameter matching
│   ├── script17_seed_floor.py # Neural adaptation (Arms A, B, C across 5 seeds)
│   ├── script30_modular_xgboost_clean.py # Modular gradient-boosted tree adaptation
│   ├── run_full_chain.sh      # End-to-end execution of sequential adaptation
│   ├── run_matched_lr.sh      # Runner for matched learning rate sensitivity
│   ├── run_script17_states.sh # State-saving runner for neural models
│   └── run_script30_save.sh   # State-saving runner for tree models
│
├── 5_explainability/          # Feature attribution & stability
│   ├── script31_update_explanations.py      # Tree update attribution decomposition
│   ├── script32_explanation_stability.py    # TreeSHAP explanation stability
│   ├── script33_nn_explanation_stability.py # Integrated Gradients stability
│   └── script34_nn_update_explanations.py   # Neural update attribution decomposition
│
├── 6_retrieval/               # Clinical literature retrieval
│   ├── script36_retrieval.py  # MedCPT retrieval stability conditioned on attributions
│   └── upstream_try3/         # Corpus builder (script24_broad_corpus.py)
│
├── 7_analysis_and_tables/     # Statistical analysis & table compilation
│   ├── analysis_A_vs_C.py     # Paired bootstrap differences and non-inferiority
│   ├── analysis_pooled.py     # Post-hoc pooled contrasts across eras
│   └── make_matched_tables.py # Generates camera-ready LaTeX tables & figures
│
├── mortality_variant/         # In-hospital mortality sensitivity analysis
│   ├── constants_mortality.py # Configuration with mortality label
│   └── run_mortality.sh       # Execution runner for mortality variant
│
├── paper/                     # Publication manuscript sources and figures
│   ├── arxiv2.tex             # Complete self-contained LaTeX source for arXiv
│   ├── arxiv2.pdf             # Compiled 11-page preprint PDF
│   ├── Pipeline.png           # Change-control pipeline diagram (Figure 1)
│   ├── Architecture.png       # Bounded vs full model updating schematic (Figure 3)
│   └── fig_auroc_matched.pdf  # Primary discrimination results (Figure 2)
│
├── logs/                      # Aggregate experiment and evaluation logs
│   ├── ml4h_clean_v2/         # Primary task logs (discrimination, calibration, drift)
│   ├── ml4h_clean_v2_mortality/ # Mortality variant evaluation logs
│   └── chain_runners/         # Sequential chain runner summaries
│
├── compare_2020_22_exclusion.py # Evaluates balance of retained vs excluded stays
├── exclusion_bias_2020_22.csv   # Appendix demographic balance table output
├── hp_ablation_from_logs.py     # Parses sensitivity to matched vs tuned rates
├── hp_ablation_summary.csv      # Parsed hyperparameter sensitivity results
├── output_hp_ablation_from_logs.txt # Full console summary of hyperparameter parity
├── LICENSE                    # MIT License
└── requirements.txt           # Python environment dependencies
```

---

## 4. Installation & Environment Setup

Clone the repository and set up a Python virtual environment (Python 3.10+ recommended):

```bash
git clone https://github.com/empresst/cloudTest.git
cd cloudTest

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All runner scripts and Python modules support directory overrides via environment variables:
- `BASE_DIR`: Base project root (default: current repository root).
- `DATA_DIR`: Directory containing cleaned cohort parquets (default: `./data`).
- `MIMIC_DIR`: Path to raw MIMIC-IV 3.1 tables (default: `./raw_mimic`).
- `LOG_DIR`: Path to experiment logs (default: `./logs/ml4h_clean_v2`).

---

## 5. Reproduction & Pipeline Run Order

### Step 1: Pre-Deployment Auditing & Verification
To verify the audit findings documented in Section 3 and the Appendix:
```bash
# 1. Verify missing ICU table clustering in 2020-2022
python 0_data_and_audit/raw_check_icu_records_check.py

# 2. Check eMAR availability onset and pipeline damage
python 0_data_and_audit/check_emar_raw.py
python 0_data_and_audit/check_emar_pipeline.py
python 0_data_and_audit/verify_emar_source_damage.py

# 3. Assess demographic & clinical balance between retained and excluded stays (Table 7)
python compare_2020_22_exclusion.py --flags 0_data_and_audit/stay_flags.parquet

# 4. Verify BigQuery and SQL audit findings
python 0_data_and_audit/evidence/verify_audit_numbers.py
```

### Step 2: Source Model Baseline
```bash
# Train the initial two-stream source model on the 2008-2013 cohort
bash 2_source_model/run_v2.sh
```

### Step 3: Drift Detection & Gated Update Triggers
```bash
# Run classifier two-sample test (C2ST) across sequential evaluation eras
python 3_drift_detection/script35_drift_detector.py

# Check gated update schedule replay (demonstrating 0% false alarms on null data)
python 3_drift_detection/script37_gated_schedule.py
```

### Step 4: Model Adaptation & Full Sequential Chain
Execute the sequential adaptation chain across both neural and tree architectures:
```bash
# Run full adaptation pipeline with shared learning rates
bash 4_adaptation/run_full_chain.sh

# Run matched learning rate sensitivity across eras
bash 4_adaptation/run_matched_lr.sh
```

### Step 5: Statistical Contrasts, Tables & Figures
```bash
# Compile matched tables and generate Figure 2 (fig_auroc_matched.pdf)
python 7_analysis_and_tables/make_matched_tables.py

# Evaluate paired differences and non-inferiority margins
python 7_analysis_and_tables/analysis_A_vs_C.py

# Run post-hoc pooled comparisons across eras
python 7_analysis_and_tables/analysis_pooled.py
```

### Step 6: Hyperparameter Sensitivity & Appendix Parity Analysis
The Appendix reports that when models are evaluated under separately tuned learning rates versus a pre-specified shared rate, the non-inferiority decisions for neural models change in 4 of 6 cells (whereas tree models remain stable in 5 of 6 cells).

You can reproduce this exact comparison from the logs at any time:
```bash
python hp_ablation_from_logs.py
```
This parses `logs/ml4h_clean_v2/script17.log` (separately tuned learning rates) against `logs/ml4h_clean_v2/script17_lr3e-3.log` (shared learning rate), generating `hp_ablation_summary.csv` and printing:
```text
===== Hyper-parameter ablation (B−C) =====

family       label         era  BmC_matched  BmC_unequal  lo90_matched  lo90_unequal verdict_matched verdict_unequal  same_NI
Neural  intubation 2014 - 2016       0.0095      -0.0141       -0.0064       -0.0313    non-inferior    inconclusive    False
Neural  intubation 2017 - 2019       0.0191      -0.0125        0.0072       -0.0251        superior    inconclusive    False
Neural  intubation 2020 - 2022       0.0038      -0.0180       -0.0144       -0.0352    non-inferior    inconclusive    False
Neural vasopressor 2014 - 2016       0.0064      -0.0020       -0.0024       -0.0104    non-inferior    non-inferior     True
Neural vasopressor 2017 - 2019       0.0002      -0.0086       -0.0069       -0.0161    non-inferior    inconclusive    False
Neural vasopressor 2020 - 2022       0.0043       0.0010       -0.0059       -0.0126    non-inferior    non-inferior     True
 Trees  intubation 2014 - 2016      -0.0126      -0.0146       -0.0251       -0.0267    inconclusive    inconclusive     True
 Trees  intubation 2017 - 2019      -0.0046      -0.0083       -0.0147       -0.0194    non-inferior    inconclusive    False
 Trees  intubation 2020 - 2022      -0.0138      -0.0180       -0.0282       -0.0359    inconclusive    inconclusive     True
 Trees vasopressor 2014 - 2016      -0.0002       0.0014       -0.0059       -0.0046    non-inferior    non-inferior     True
 Trees vasopressor 2017 - 2019       0.0014       0.0018       -0.0037       -0.0038    non-inferior    non-inferior     True
 Trees vasopressor 2020 - 2022       0.0044       0.0100       -0.0037        0.0016    non-inferior    non-inferior     True

Cells with identical NI verdict: 7 / 12
Wrote hp_ablation_summary.csv
```

---

## 6. Paper Compilation

The complete self-contained paper manuscript is provided in `paper/arxiv2.tex`. To compile the preprint PDF:
```bash
cd paper
pdflatex arxiv2.tex
pdflatex arxiv2.tex
```
The compiled paper `arxiv2.pdf` includes all embedded figures, audit tables, and non-inferiority evaluation bounds.

---

## 7. Citation & License

If you use this code, the data audit methodologies, or experimental configurations, please cite:

```bibtex
@article{tamanna2026audit,
  title={Audit Before You Update: Recording Artifacts and the Limits of Bounded Model Updates in MIMIC-IV ICU Prediction},
  author={Tamanna, Fatema Ferdous and Arefin, K. M. Merajul and Masud, Md. Abdul},
  journal={arXiv preprint},
  year={2026}
}
```

This repository is licensed under the [MIT License](LICENSE).
