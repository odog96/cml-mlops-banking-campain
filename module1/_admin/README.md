# Admin Tools for Module 1

This folder contains administrative tools and documentation for instructors and maintainers. These files are kept separate to avoid cluttering the main module directory for lab participants.

## Contents

### `run_tests.py`
Automated test runner for validating the entire module 1 pipeline.

**Usage:**
```bash
python run_tests.py                  # Run all scripts
python run_tests.py --verbose        # With detailed output
python run_tests.py --skip 04        # Skip deployment
python run_tests.py --only 01,02,03  # Run only specific scripts
python run_tests.py --timeout 900    # Increase timeout to 15 minutes
```

**Output:**
- Console: Real-time execution status
- `test_results.json`: Machine-readable results for CI/CD
- `test_results.log`: Detailed execution log

### `TESTING.md`
Complete testing guide and documentation, including:
- Detailed usage examples
- Understanding test output
- Timeout management
- Troubleshooting guide
- CI/CD integration examples
- Performance considerations

## Quick Commands

### Validate a new environment
```bash
cd _admin
python run_tests.py
```

### Test only data ingestion and EDA (fast)
```bash
python run_tests.py --only 01,02
```

### Test training and inference (skip deployment)
```bash
python run_tests.py --skip 04
```

### Debug with full output
```bash
python run_tests.py --verbose
```

## For Lab Participants

Participants should focus on the numbered scripts (01_*, 02_*, etc.) in the main module directory. The `helpers/` folder contains supporting code they may use but shouldn't need to modify.

This `_admin/` folder is for instructors and course staff to validate the pipeline in new environments.

## Pipeline Execution Order

1. `01_ingest.py` - Data ingestion
2. `02_eda_notebook.ipynb` - EDA analysis
3. `03_train_quick.py` - Quick training
4. `04_deploy.py` - Model deployment
5. `05_1_inference_data_prep.py` - Inference prep
6. `05_2_inference_predict.py` - Predictions

## Troubleshooting

See `TESTING.md` for comprehensive troubleshooting guide.
