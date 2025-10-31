# Module 2: Getting Started

## Quick Overview

You have a complete model monitoring pipeline with **4 main scripts**:

```
00_prepare_artificial_data.py  ← Run ONCE to create artificial data
    ↓
01_get_predictions.py          ← CML Job (processes with tracking)
    ↓
02_load_ground_truth.py        ← CML Job (loads labels per period)
    ↓
03_check_model.py              ← CML Job (validates & orchestrates)
```

## Quick Start (5 minutes)

### Step 1: Prepare Data
```bash
cd /home/cdsw/module2
python 00_prepare_artificial_data.py
```

**What it does**: Creates artificial dataset simulating model degradation over 5 time periods
**Output files**:
- `data/artificial_ground_truth_data.csv`
- `data/ground_truth_metadata.json`

### Step 2: Verify Data Created
```bash
ls -la /home/cdsw/module2/data/
```

You should see:
- `artificial_ground_truth_data.csv` (~5,000 rows)
- `ground_truth_metadata.json` (configuration)

### Step 3: Create CML Jobs (in CML UI)

Create **3 jobs** in your CML project:

#### Job 1: get_predictions
```
Name: get_predictions
Script: /home/cdsw/module2/01_get_predictions.py
Environment: PERIOD=0, BATCH_SIZE=50
Trigger: Manual (or scheduled)
```

#### Job 2: load_ground_truth
```
Name: load_ground_truth
Script: /home/cdsw/module2/02_load_ground_truth.py
Environment: (inherits PERIOD from previous job)
Trigger: When get_predictions completes
```

#### Job 3: check_model
```
Name: check_model
Script: /home/cdsw/module2/03_check_model.py
Environment: ACCURACY_THRESHOLD=0.85
Trigger: When load_ground_truth completes
```

### Step 4: Start Pipeline
Run the first job:
- Click "Run" on the `get_predictions` job in CML UI
- Or use: `cmlapi` to trigger with `PERIOD=0`

The pipeline will:
1. Get and process predictions for Period 0
2. Load ground truth labels for Period 0
3. Validate accuracy (should be ~95%)
4. If not degraded and not last period, continue to Period 1
5. Repeat until completion or degradation detected

## What Happens in Each Job

### 01_get_predictions.py
**Input**: Period metadata & full dataset
**Output**: Predictions with metrics
**Tracks**: `cdsw.track_delayed_metrics()` for each prediction

### 02_load_ground_truth.py
**Input**: Period metadata & artificial data
**Output**: Labels for current period
**Tracks**: Ground truth loading

### 03_check_model.py
**Input**: Predictions & ground truth
**Output**: Accuracy report
**Tracks**: `cdsw.track_aggregate_metrics()` for period

**Decision**:
- Degradation detected? → EXIT (alert)
- Last period? → EXIT (success)
- Otherwise → Continue to next period (trigger 01_get_predictions for Period N+1)

## Expected Results

```
Period 0: ~95% accuracy ✓ PASS
Period 1: ~90% accuracy ✓ PASS  
Period 2: ~85% accuracy ✓ PASS
Period 3: ~80% accuracy ✓ PASS
Period 4: ~75% accuracy ⚠ Might trigger alert
```

Each period shows intentional accuracy degradation to simulate real model drift.

## Key Features

- ✅ Ground truth loaded separately (not embedded)
- ✅ Batch processing with configurable size
- ✅ Cloudera metrics tracking (both individual & aggregate)
- ✅ Automatic job orchestration
- ✅ Degradation detection
- ✅ Complete audit trail

## Customization

### Change Number of Periods
Edit `00_prepare_artificial_data.py`:
```python
num_periods = 10  # Default is 5
```
Then regenerate data.

### Change Batch Size
Edit CML job environment:
```
BATCH_SIZE=100  # Default is 50
```

### Change Accuracy Thresholds
Edit CML job environment:
```
ACCURACY_THRESHOLD=0.90
DEGRADATION_THRESHOLD=0.03
```

## Troubleshooting

**Problem**: Scripts won't run  
**Solution**: Make sure you ran `00_prepare_artificial_data.py` first

**Problem**: Jobs don't chain together  
**Solution**: Check job names match exactly in CML

**Problem**: "FileNotFoundError"  
**Solution**: Ensure `data/` folder exists and has the CSV file

## Files to Review

- **README.md** - Comprehensive documentation
- **IMPLEMENTATION_SUMMARY.md** - Design decisions & API patterns
- **00_prepare_artificial_data.py** - Data generation (read the docstring)
- **01_load_ground_truth.py** - Ground truth loading (read the docstring)
- **02_get_predictions.py** - Prediction tracking (read the docstring)
- **03_check_model.py** - Model validation (read the docstring)

## How It Compares to Examples

This implementation extends the example files:

| Feature | example_get_predictions.py | 02_get_predictions.py |
|---------|---------------------------|----------------------|
| Prediction tracking | ✓ | ✓ (batch-based) |
| Ground truth | Embedded | Separate script |
| Batch size | Fixed | Configurable |
| Job orchestration | None | ✓ Auto-triggered |

| Feature | example_check_model.py | 03_check_model.py |
|---------|----------------------|-------------------|
| Metrics calculation | RMSE | Accuracy metrics |
| Job triggering | Manual | ✓ Auto-triggered |
| Decision logic | Simple | ✓ Period-aware |

---

## Next: Production Integration

Once you verify the pipeline works:

1. **Replace artificial data** with real predictions from your model
2. **Integrate with live API** instead of using stored predictions
3. **Connect real ground truth** from your business systems
4. **Add alerting** (email, Slack, etc.) when degradation detected
5. **Trigger retraining** automatically when needed

---

For detailed documentation, see [README.md](README.md)
