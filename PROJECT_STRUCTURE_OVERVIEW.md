# Comprehensive Project Structure Overview

## Date: November 4, 2025
## Project: Cloudera AI MLOps Lab (Module 1, 2, and 3)

---

## 1. ROOT DIRECTORY FILES AND ORGANIZATION

### Root Level Files
```
/home/cdsw/
├── README.md                          (4.9 KB) - Main project documentation (currently Module 3 README)
├── requirements.txt                   (688 B)  - Project dependencies
├── 1_drift_report_explicit.html        (3.6 MB) - Generated drift report
├── .gitignore                          - Git configuration
├── .project-metadata.yaml              - Project metadata
└── .Rprofile                           - R configuration
```

### Key Directories
```
/home/cdsw/
├── module1/                           - Data ingestion through inference pipeline
├── module2/                           - Model monitoring with degradation detection
├── module3/                           - Proactive MLOps with automated retraining
├── shared_utils/                      - Shared utilities and configuration
├── assets/                            - Documentation and resources
├── outputs/                           - Generated outputs from pipeline
└── data/                              - Module 2 monitoring data
```

---

## 2. MODULE STRUCTURE AND FILE ORGANIZATION

### MODULE 1: Bank Marketing Model - Complete ML Workflow
**Location:** `/home/cdsw/module1/`
**Purpose:** Complete ML pipeline from data ingestion to production inference
**Status:** Fully functional with 6 main steps + utilities

#### Main Script Files (Execution Order)
```
module1/
├── 01_ingest.py                       (7.8 KB)  - Data ingestion
├── 02_eda_notebook.ipynb              (1.04 MB) - Exploratory Data Analysis
├── 03_train_quick.py                  (6.2 KB)  - Quick model training
├── 03_train_extended.py               (9.8 KB)  - Extended training with more models
├── 04_deploy.py                       (11.4 KB) - Model deployment
├── 05.1_inference_data_prep.py        (8.8 KB)  - Feature engineering for inference
├── 05.2_inference_predict.py          (16 KB)   - Generate predictions
├── 05.2_inference_predict_TEST.py     (6.4 KB)  - Test version
└── 06_Inference_101.ipynb             (9.4 KB)  - Interactive inference notebook
```

#### Supporting Files and Directories
```
module1/
├── README.md                          (25.8 KB) - Detailed module documentation
├── shared_utils.py                    (196 B)   - Shared utilities import
├── helpers/                           - Preprocessing and utilities
│   ├── __init__.py
│   ├── preprocessing.py               (10.4 KB) - Feature engineering
│   ├── utils.py                       (5.1 KB)  - Utility functions
│   ├── _training_utils.py             (10 KB)   - Training helpers
│   └── test_runner.py                 (15.2 KB) - Test execution
├── _admin/                            - Administrator tools
│   ├── README.md                      (2.1 KB)  - Admin guide
│   ├── TESTING.md                     (5.2 KB)  - Testing documentation
│   ├── QUICK_START.txt                (4.7 KB)  - Quick reference
│   ├── run_tests.py                   (1.6 KB)  - Automated test runner
│   └── data/                          - Test data directory
├── data/                              - Training data
│   ├── bank-additional-full.csv       (5.8 MB)  - Full training dataset
│   ├── bank-additional.csv            (584 KB)  - Training dataset
│   └── bank-additional-names.txt      (5.5 KB)  - Feature descriptions
├── inference_data/                    - Inference datasets
│   ├── raw_inference_data.csv         (116 KB)  - Raw inference input
│   ├── engineered_inference_data.csv  (150 KB)  - Engineered features
│   ├── predictions.csv                (13.3 KB) - Final predictions
│   ├── prediction_summary.txt         (1.6 KB)  - Summary stats
│   └── feature_engineer.pkl           (59 B)    - Feature encoder
├── outputs/                           - Generated outputs
│   ├── deployment_info.json           (405 B)   - Deployment metadata
│   ├── baseline_results.csv           (554 B)   - Baseline metrics
│   ├── engineered_results.csv         (536 B)   - Engineered metrics
│   └── execution_timing.txt           (324 B)   - Performance metrics
└── .gitignore                         - Git ignore rules
```

**File Naming Convention:** Numbers with underscore separator
- Format: `NN_description.py` or `NN.N_description.py`
- Examples: `01_ingest.py`, `05.1_inference_data_prep.py`

---

### MODULE 2: Model Monitoring with Degradation Detection
**Location:** `/home/cdsw/module2/`
**Purpose:** Monitor model performance and detect accuracy degradation
**Status:** Functional with integrated pipeline pattern

#### Main Script Files
```
module2/
├── 01_create_jobs.ipynb               (26.2 KB) - Job creation guide
├── 02_prepare_artificial_data.py      (12.3 KB) - Create artificial ground truth
├── 03_monitoring_pipeline.py          (21.7 KB) - Integrated monitoring pipeline
└── 04_model_metrics_analysis.ipynb    (124.8 KB) - Metrics analysis notebook
```

#### Supporting Documentation
```
module2/
├── README.md                          (13.9 KB) - Detailed module documentation
└── __pycache__/                       - Python cache
```

**File Naming Convention:** Same as Module 1
- Format: `NN_description.py` with `.ipynb` for notebooks
- Examples: `01_create_jobs.ipynb`, `02_prepare_artificial_data.py`

**Architecture:** Single integrated job (not chained jobs)
- Job 1: Prepare artificial data (one-time setup)
- Job 2: Monitor pipeline (processes all periods sequentially)

---

### MODULE 3: Proactive MLOps - Automated Retraining Loop
**Location:** `/home/cdsw/module3/`
**Purpose:** Detect data drift and automatically retrain/deploy models
**Status:** Functional with event-driven pipeline

#### Main Script Files (Execution Order)
```
module3/
├── 0_simulate_live_data.py            (1.8 KB)  - Generate drifted data
├── 1_check_drift.py                   (3.4 KB)  - Detect drift with Evidently
├── 2_simulate_labeling_job.py         (2.9 KB)  - Simulate label acquisition
├── 3_retrain_model.py                 (5.9 KB)  - Train new model
└── 4_register_and_deploy.py           (9.1 KB)  - Register and deploy model
```

#### Supporting Files
```
module3/
├── README.md                          (5.1 KB)  - Module documentation
├── reporting_launch_app.py            (2.2 KB)  - Streamlit app launcher
├── reporting_main_app.py              (4.9 KB)  - Main app code
├── 1_drift_report_explicit.html       (3.6 MB)  - Drift report (generated)
├── old-1_check_drift-Copy1.py         (2.0 KB)  - Legacy version
├── outputs/                           - Generated outputs
│   ├── deployment_info_v2.json        (405 B)   - v2 deployment info
│   ├── drift_status.json              (132 B)   - Drift detection status
│   └── live_unlabeled_batch.csv       (349 KB)  - Simulated live data
├── static/                            - Web app assets
│   ├── js/
│   │   └── app.js                     (2.5 KB)  - JavaScript
│   └── reports/                       - Report storage
├── templates/                         - HTML templates
│   └── index.html                     (5.4 KB)  - Main template
└── __pycache__/                       - Python cache
```

**File Naming Convention:** Different from Module 1
- Format: Single digit followed by underscore
- Examples: `0_simulate_live_data.py`, `1_check_drift.py`

**Pipeline Flow:**
1. Simulate live data with drift
2. Detect drift (triggers failure)
3. Simulate label acquisition (triggered by failure)
4. Retrain model (triggered by new data)
5. Register and deploy (triggered by training completion)

---

## 3. SHARED UTILITIES

**Location:** `/home/cdsw/shared_utils/`

```
shared_utils/
├── README.md                          (0 B)     - Empty placeholder
├── requirements.txt                   (751 B)   - Dependencies
├── __init__.py                        (1.0 KB)  - Module initialization
├── config.py                          (857 B)   - Configuration
├── data_connection.py                 (1.6 KB)  - Data connectivity
├── cleanup_models.py                  (6.1 KB)  - Model cleanup utility
├── reset.py                           (3.8 KB)  - Reset utility
└── install-dependencies.py            (634 B)   - Dependency installer
```

---

## 4. ASSETS AND DATA STORAGE

### Assets Directory
**Location:** `/home/cdsw/assets/`
```
assets/
└── ML_Evaluation_Metrics_Final.pdf    (165.6 KB) - Reference guide
```

### Root Data and Outputs
**Location:** `/home/cdsw/`
```
/home/cdsw/
├── data/                              - Module 2 monitoring data
│   ├── artificial_ground_truth_data.csv      (168.6 KB)
│   ├── ground_truth_metadata.json            (778 B)
│   ├── check_model_results.json
│   ├── current_period_ground_truth.json
│   ├── period_0_ground_truth.json
│   ├── period_1_ground_truth.json
│   ├── predictions_period_0.json
│   ├── predictions_period_1.json
│   ├── monitoring_results.json               (1.3 KB)
│   └── monitoring_log.txt                    (68.6 KB)
└── outputs/                           - Root level outputs
    ├── deployment_info_v2.json               (408 B)
    ├── drift_status.json                     (132 B)
    ├── execution_timing.txt                  (359 B)
    ├── live_unlabeled_batch.csv              (349 KB)
    ├── new_labeled_batch_01.csv              (359 KB)
    └── retrain_run_info.json                 (214 B)
```

---

## 5. ALL MARKDOWN DOCUMENTATION FILES

### Primary Documentation Files
| File Path | Size | Purpose |
|-----------|------|---------|
| `/home/cdsw/README.md` | 4.9 KB | Root documentation (currently Module 3 README) |
| `/home/cdsw/module1/README.md` | 25.8 KB | Complete Module 1 guide with all 6 steps |
| `/home/cdsw/module2/README.md` | 13.9 KB | Module 2 monitoring pipeline guide |
| `/home/cdsw/module3/README.md` | 5.1 KB | Module 3 drift detection and retraining |
| `/home/cdsw/shared_utils/README.md` | 0 B | Empty placeholder |
| `/home/cdsw/module1/_admin/README.md` | 2.1 KB | Admin tools documentation |

### Supporting Documentation
| File Path | Size | Purpose |
|-----------|------|---------|
| `/home/cdsw/module1/_admin/TESTING.md` | 5.2 KB | Testing guide and procedures |
| `/home/cdsw/module1/_admin/QUICK_START.txt` | 4.7 KB | Quick reference guide |

---

## 6. PDF FILES IN PROJECT

**Location:** `/home/cdsw/assets/`
```
ML_Evaluation_Metrics_Final.pdf        (165.6 KB) - ML metrics reference document
```

---

## 7. FILE NAMING CONVENTIONS ANALYSIS

### Naming Patterns Used

#### Module 1: Numbered Prefix with Underscore
```
Pattern: NN_description.py or NN.N_description.py
Examples:
  01_ingest.py
  02_eda_notebook.ipynb
  03_train_quick.py
  03_train_extended.py
  04_deploy.py
  05.1_inference_data_prep.py          (Dot separator for sub-steps)
  05.2_inference_predict.py
  06_Inference_101.ipynb
  
Supporting files:
  shared_utils.py
  run_tests.py
```

#### Module 2: Numbered Prefix with Underscore (Same as Module 1)
```
Pattern: NN_description.py
Examples:
  01_create_jobs.ipynb
  02_prepare_artificial_data.py
  03_monitoring_pipeline.py
  04_model_metrics_analysis.ipynb
```

#### Module 3: Single Digit Prefix with Underscore (DIFFERENT)
```
Pattern: N_description.py
Examples:
  0_simulate_live_data.py
  1_check_drift.py
  2_simulate_labeling_job.py
  3_retrain_model.py
  4_register_and_deploy.py
  
Supporting files:
  reporting_launch_app.py
  reporting_main_app.py
  old-1_check_drift-Copy1.py            (Legacy with different naming)
```

### INCONSISTENCY IDENTIFIED
- **Module 1 & 2:** Use 2-digit prefixes (01_, 02_, 03_, etc.)
- **Module 3:** Uses single-digit prefixes (0_, 1_, 2_, 3_, 4_)
- **Module 3:** Has files with legacy names using hyphens (old-1_check_drift-Copy1.py)

### Notebook Naming
- Notebooks mixed with scripts: `02_eda_notebook.ipynb`, `01_create_jobs.ipynb`
- Clear naming convention: `NN_description.ipynb` or `N_description.ipynb`

---

## 8. COMPREHENSIVE DIRECTORY TREE

```
/home/cdsw/
│
├── README.md                          # Root documentation
├── requirements.txt                   # Dependencies
├── .gitignore                         # Git configuration
├── .project-metadata.yaml             # Project metadata
├── 1_drift_report_explicit.html       # Generated drift report
│
├── assets/
│   └── ML_Evaluation_Metrics_Final.pdf
│
├── data/                              # Module 2 monitoring outputs
│   ├── artificial_ground_truth_data.csv
│   ├── ground_truth_metadata.json
│   ├── check_model_results.json
│   ├── monitoring_results.json
│   ├── monitoring_log.txt
│   ├── period_*.json
│   └── predictions_period_*.json
│
├── outputs/                           # Root-level pipeline outputs
│   ├── deployment_info_v2.json
│   ├── drift_status.json
│   ├── execution_timing.txt
│   ├── live_unlabeled_batch.csv
│   ├── new_labeled_batch_01.csv
│   └── retrain_run_info.json
│
├── module1/                           # Complete ML Workflow
│   ├── README.md
│   ├── 01_ingest.py
│   ├── 02_eda_notebook.ipynb
│   ├── 03_train_quick.py
│   ├── 03_train_extended.py
│   ├── 04_deploy.py
│   ├── 05.1_inference_data_prep.py
│   ├── 05.2_inference_predict.py
│   ├── 05.2_inference_predict_TEST.py
│   ├── 06_Inference_101.ipynb
│   ├── shared_utils.py
│   ├── .gitignore
│   │
│   ├── _admin/
│   │   ├── README.md
│   │   ├── TESTING.md
│   │   ├── QUICK_START.txt
│   │   ├── run_tests.py
│   │   └── data/
│   │       └── bank-additional/
│   │           ├── bank-additional-full.csv
│   │           ├── bank-additional.csv
│   │           └── bank-additional-names.txt
│   │
│   ├── helpers/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── utils.py
│   │   ├── _training_utils.py
│   │   └── test_runner.py
│   │
│   ├── data/
│   │   └── bank-additional/
│   │       ├── bank-additional-full.csv
│   │       ├── bank-additional.csv
│   │       └── bank-additional-names.txt
│   │
│   ├── inference_data/
│   │   ├── raw_inference_data.csv
│   │   ├── engineered_inference_data.csv
│   │   ├── predictions.csv
│   │   ├── prediction_summary.txt
│   │   └── feature_engineer.pkl
│   │
│   └── outputs/
│       ├── deployment_info.json
│       ├── baseline_results.csv
│       ├── engineered_results.csv
│       └── execution_timing.txt
│
├── module2/                           # Model Monitoring
│   ├── README.md
│   ├── 01_create_jobs.ipynb
│   ├── 02_prepare_artificial_data.py
│   ├── 03_monitoring_pipeline.py
│   └── 04_model_metrics_analysis.ipynb
│
├── module3/                           # Proactive MLOps
│   ├── README.md
│   ├── 0_simulate_live_data.py
│   ├── 1_check_drift.py
│   ├── 2_simulate_labeling_job.py
│   ├── 3_retrain_model.py
│   ├── 4_register_and_deploy.py
│   ├── reporting_launch_app.py
│   ├── reporting_main_app.py
│   ├── old-1_check_drift-Copy1.py
│   ├── 1_drift_report_explicit.html
│   │
│   ├── outputs/
│   │   ├── deployment_info_v2.json
│   │   ├── drift_status.json
│   │   └── live_unlabeled_batch.csv
│   │
│   ├── static/
│   │   ├── js/
│   │   │   └── app.js
│   │   └── reports/
│   │
│   └── templates/
│       └── index.html
│
└── shared_utils/
    ├── README.md
    ├── requirements.txt
    ├── __init__.py
    ├── config.py
    ├── data_connection.py
    ├── cleanup_models.py
    ├── reset.py
    └── install-dependencies.py
```

---

## 9. PYTHON FILES BY MODULE

### Module 1 Python Files (11 total)
```
Main Scripts: 01_ingest.py, 02_eda_notebook.ipynb, 03_train_quick.py,
              03_train_extended.py, 04_deploy.py, 05.1_inference_data_prep.py,
              05.2_inference_predict.py, 05.2_inference_predict_TEST.py
Helpers: preprocessing.py, utils.py, _training_utils.py
Testing: test_runner.py, run_tests.py
Shared: shared_utils.py
```

### Module 2 Python Files (3 total)
```
Main Scripts: 01_create_jobs.ipynb, 02_prepare_artificial_data.py,
              03_monitoring_pipeline.py, 04_model_metrics_analysis.ipynb
```

### Module 3 Python Files (7 total)
```
Pipeline: 0_simulate_live_data.py, 1_check_drift.py, 2_simulate_labeling_job.py,
          3_retrain_model.py, 4_register_and_deploy.py
Apps: reporting_launch_app.py, reporting_main_app.py
Legacy: old-1_check_drift-Copy1.py
```

### Shared Utils (7 total)
```
Utilities: cleanup_models.py, config.py, data_connection.py, reset.py,
           install-dependencies.py
Config: __init__.py, requirements.txt
```

**Total Python/Notebook Files:** 28 files

---

## 10. KEY OBSERVATIONS

### Structure Quality
- Clear hierarchical organization with 3 modules + shared utilities
- Each module is self-contained with clear purpose
- Admin tools properly separated in _admin subdirectory
- Supporting code organized in helpers subdirectory

### Documentation Quality
- Comprehensive README files for each module (25KB+)
- Testing documentation (TESTING.md, QUICK_START.txt)
- Clear execution order documented
- Markdown files provide complete workflow descriptions

### File Naming Inconsistencies
1. **Module 3 uses single-digit prefixes** while Modules 1-2 use two-digit
2. **Module 3 has legacy files** with different naming (hyphens)
3. **Notebooks mixed with scripts** - could be organized separately

### Data Organization
- Training data in module1/data/bank-additional/
- Inference data in module1/inference_data/
- Generated outputs scattered in module outputs/ and root outputs/
- Module 2 monitoring data in root data/ directory

### PDF Resources
- Single PDF reference guide (ML_Evaluation_Metrics_Final.pdf)
- Located in assets/ directory

---

## 11. EXECUTION DEPENDENCIES

### Sequential Module Order
1. **Module 1:** Complete first (data, training, deployment)
2. **Module 2:** Depends on Module 1 inference data
3. **Module 3:** Can run independently or after Module 1

### File Dependencies
- Module 2 requires: module1/inference_data/engineered_inference_data.csv
- Module 3 requires: module1 training data or root banking_train.csv
- Deployment steps require MLflow and CML API access

---

## 12. GIT STATUS SUMMARY

### Deleted Files
- INDEX.md (deleted)
- JOB_PARAMETER_PASSING_FIX.md (deleted)
- QUICK_REFERENCE.md (deleted)

### Modified Files
- module3/4_register_and_deploy.py (modified)

### Untracked Files
- assets/ (new directory with PDF)
- module3/outputs/deployment_info_v2.json (new)

---

## SUMMARY

This is a well-organized, three-module MLOps laboratory for Cloudera AI that demonstrates:
- **Module 1:** Complete ML pipeline (6 steps from ingestion to inference)
- **Module 2:** Reactive monitoring with degradation detection
- **Module 3:** Proactive MLOps with drift detection and automated retraining

The project follows clear naming conventions (with minor inconsistencies in Module 3), has comprehensive documentation, and maintains proper separation of concerns with shared utilities. The only inconsistency to note is the file naming convention change in Module 3 (single digit vs. two digits) which should be standardized.

