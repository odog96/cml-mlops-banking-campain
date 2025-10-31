# Module 1 Testing Guide

## Running the Complete Pipeline

The test runner allows you to execute all scripts sequentially and track their execution status.

### Quick Start

```bash
# Run all scripts
python run_tests.py

# Run with full output (for debugging)
python run_tests.py --verbose

# Run with extended timeout (useful for slow systems)
python run_tests.py --timeout 900  # 15 minutes per script
```

### Selective Testing

```bash
# Skip the deployment script (04_deploy.py)
python run_tests.py --skip 04

# Skip inference pipeline
python run_tests.py --skip 05_1,05_2

# Run only data ingestion and EDA
python run_tests.py --only 01,02

# Run training and deployment only
python run_tests.py --only 03,04
```

### Understanding the Output

The test runner provides:

1. **Console Output**: Real-time execution status with ✅ (passed), ❌ (failed), ⏭️ (skipped)
2. **test_results.json**: Machine-readable results (metrics, timings, errors)
3. **test_results.log**: Detailed execution log with full script output

Example console output:
```
================================================================================
Running: 01 - Data Ingestion
File: 01_ingest.py
================================================================================
✅ PASSED in 12.34s

================================================================================
Running: 02 - EDA Analysis (Jupyter)
File: 02_eda_notebook.ipynb
================================================================================
✅ PASSED in 45.67s

...

================================================================================
TEST EXECUTION SUMMARY
================================================================================

Results:
  Passed:  6/6 ✅
  Failed:  0/6 ❌
  Skipped: 0/6 ⏭️

Timing:
  Total time:  120.45s (2.01m)
  Start time:  2025-10-31 17:30:00
  End time:    2025-10-31 17:32:00

================================================================================
✅ ALL TESTS PASSED!
================================================================================
```

## Script Pipeline

The test runner executes scripts in this order:

1. **01_ingest.py** - Data Ingestion
   - Creates sample inference data
   - Validates data format

2. **02_eda_notebook.ipynb** - EDA Analysis
   - Exploratory data analysis
   - Generates visualizations

3. **03_train_quick.py** - Quick Model Training
   - Trains baseline and engineered feature models
   - Logs metrics to MLflow
   - Duration: ~2-5 minutes

4. **04_deploy.py** - Model Deployment
   - Deploys best model to CML
   - Sets up serving endpoint

5. **05_1_inference_data_prep.py** - Inference Data Preparation
   - Applies feature engineering to inference data
   - Prepares data for predictions

6. **05_2_inference_predict.py** - Inference Predictions
   - Loads trained model
   - Makes predictions on inference data
   - Saves results with row tracking

## Timeout Management

Each script has a default timeout of 600 seconds (10 minutes). If your environment is slower, increase it:

```bash
# 15 minute timeout per script
python run_tests.py --timeout 900
```

For development/testing in constrained environments:

```bash
# Just test data ingestion
python run_tests.py --only 01

# Test training without deployment
python run_tests.py --skip 04
```

## Troubleshooting

### Script Fails with "File not found"
Ensure you're running the test runner from the module1 directory:
```bash
cd module1
python run_tests.py
```

### Jupyter Notebook Fails
If `02_eda_notebook.ipynb` fails, ensure nbconvert is installed:
```bash
pip install nbconvert
```

### Timeout Errors
Increase the timeout value:
```bash
python run_tests.py --timeout 1200  # 20 minutes
```

### Out of Memory
If you get memory errors during training (03_train_quick.py or 03_train_extended.py), you can skip them:
```bash
python run_tests.py --skip 03
```

## Continuous Integration

The test runner is CI/CD friendly. Exit codes indicate success/failure:

```bash
python run_tests.py
if [ $? -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Some tests failed!"
    cat test_results.json  # See details
fi
```

## Advanced Usage

### Check Results Programmatically

```python
from helpers.test_runner import TestRunner

runner = TestRunner(verbose=False, timeout=600)
results = runner.run_all()

for script, result in results.items():
    print(f"{script}: {result['status']}")
    if result['status'] == 'FAILED':
        print(f"  Error: {result['error']}")
```

### Custom Module Path

```bash
python run_tests.py --module-path /path/to/module1
```

## Performance Considerations

- **Full pipeline**: ~5-10 minutes on typical systems
- **Without training** (`--skip 03`): ~2-3 minutes
- **Data only** (`--only 01,02`): ~1 minute
- **Inference only** (`--only 05_1,05_2`): ~1 minute

## Tips

1. **First run**: Use `--verbose` to see what each script is doing
2. **CI/CD**: Check `test_results.json` for machine-readable results
3. **Development**: Use `--only` to test specific components
4. **Debugging**: Run individual scripts directly (e.g., `python 01_ingest.py`) for detailed error output
5. **Environment validation**: Run the full pipeline in a new environment to catch missing dependencies
