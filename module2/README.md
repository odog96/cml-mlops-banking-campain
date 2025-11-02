# Module 2: Model Monitoring with Degradation Detection

This module implements a comprehensive model monitoring pipeline that tracks predictions over time periods and automatically detects when model accuracy degrades. The pipeline uses artificial ground truth data with intentional accuracy degradation to simulate real-world model drift.

## Overview

The monitoring pipeline consists of 4 scripts that orchestrate together:

```
00_prepare_artificial_data.py (One-time setup)
    ↓
01_get_predictions.py (Period 0 - batch processing)
    ├─ Process batch 0 → Track metrics
    ├─ Process batch 1 → Track metrics
    ├─ Process batch 2 → Track metrics
    └─ ... (all batches) → Trigger 02_load_ground_truth
         ↓
    02_load_ground_truth.py (Period 0 - load labels)
         ↓ Trigger 03_check_model
    03_check_model.py (Period 0 - validate accuracy & decide)
         ↓ (if not degraded and not last period)
    01_get_predictions.py (Period 1 - next period)
         ↓ (repeats for each period until degradation or end of data)
```

**Job Execution Pattern**:
- Number of periods = Total data size / Batch size
- With 1000 samples and batch size 50 = 20 periods
- Each job (01, 02, 03) runs **20 times total** (once per period)
- 01_get_predictions processes ALL batches in a period before triggering 02
- 02_load_ground_truth loads ALL labels for a period before triggering 03
- 03_check_model validates accuracy for a period, then decides: continue to next period or exit

## Key Features

- **Period-based Processing**: Splits data into multiple time periods for sequential monitoring
- **Batch Predictions**: Processes predictions in configurable batch sizes
- **Cloudera Integration**: Tracks metrics using `cdsw.track_delayed_metrics()` and `cdsw.track_aggregate_metrics()`
- **Degradation Detection**: Automatically detects statistically significant accuracy drops
- **Job Orchestration**: Each job triggers the next via `cmlapi` (when configured)
- **Configurable Thresholds**: Adjust accuracy requirements and degradation sensitivity

## Quick Start

1. **Prepare artificial data** (one-time):
   ```bash
   python 00_prepare_artificial_data.py
   ```

2. **Create CML jobs** with the 3 job scripts (01_get_predictions.py, 02_load_ground_truth.py, 03_check_model.py)

3. **Start monitoring**:
   ```bash
   # Run first job (01_get_predictions) manually to start the pipeline
   # Job will auto-trigger subsequent jobs (02_load_ground_truth → 03_check_model)
   # If not degraded and not last period, 03_check_model will trigger 01_get_predictions for next period
   ```

---

## Data Architecture

### Artificial Ground Truth Dataset

The artificial dataset includes:
- **Engineered features** from Module 1 (original features)
- **known_prediction**: Model predictions from Module 1
- **artificial_ground_truth**: Labels that match predictions with degrading accuracy
- **period**: Which time period (0, 1, 2, ...) each sample belongs to
- **Probability scores**: probability_class_0, probability_class_1

**Degradation Pattern** (with 20 periods and 2.5% per-period drop):
- Period 0: 95% accuracy (baseline)
- Period 1: 92.5% accuracy (2.5% drop)
- Period 5: 82.5% accuracy
- Period 10: 72.5% accuracy
- Period 15: 62.5% accuracy
- Period 19: 52.5% accuracy (reaches ~50% by end)

---

## Scripts

### 00_prepare_artificial_data.py

**Status**: One-time setup (run manually before starting pipeline)

**Purpose**: Creates the artificial ground truth dataset with intentional accuracy degradation

**What it does**:
1. Loads engineered inference data from Module 1
2. Loads predictions from Module 1
3. Creates artificial ground truth labels with progressive corruption
4. Splits data into N periods where N = total_samples / batch_size (20 periods for 1000 samples)
5. Saves `artificial_ground_truth_data.csv` and `ground_truth_metadata.json`

**Configuration**:
```python
num_periods = 20                   # = Total samples / Batch size (1000 / 50)
initial_accuracy = 0.95            # 95% match in period 0
degradation_rate = 0.025           # 2.5% drop per period
```

**Output**:
- `data/artificial_ground_truth_data.csv` (full dataset with labels)
- `data/ground_truth_metadata.json` (period boundaries)

**Note**: This data will NOT be available at lab inception. Generate it manually before starting the monitoring pipeline.

**Run**:
```bash
python 00_prepare_artificial_data.py
```

---

### 01_get_predictions.py

**Status**: CML Job (repeatable, one per period)

**Purpose**: Processes predictions for current period with Cloudera tracking

**What it does**:
1. Loads period configuration from metadata
2. Loads full dataset and extracts current PERIOD data
3. Processes all batches for this period:
   - For each batch:
     - For each prediction:
       - Creates tracking record
       - Calls `cdsw.track_delayed_metrics()` with prediction data
       - (metrics tracked to CML for monitoring)
     - After batch completes: Triggers 02_load_ground_truth.py (for this batch)
4. Saves all predictions for the period to JSON
5. This job runs once per period (20 times if 20 periods)

**Environment Variables**:
- `PERIOD`: Current period (default: 0)
- `BATCH_SIZE`: Samples per batch (default: 50)
- `MODEL_NAME`: Deployed model name (default: "banking_campaign_predictor")
- `PROJECT_NAME`: CML project name (default: "CAI Baseline MLOPS")

**Input**:
- `data/artificial_ground_truth_data.csv`

**Output**:
- `data/predictions_period_{PERIOD}.json` (predictions with metadata)
- CML metrics (tracked via `cdsw`)
- Triggers `02_load_ground_truth.py`

**Example usage** (as CML job):
```bash
PERIOD=0 python 01_get_predictions.py
```

---

### 02_load_ground_truth.py

**Status**: CML Job (repeatable, one per period)

**Purpose**: Loads ground truth labels for the current period

**What it does**:
1. Loads period configuration from metadata
2. Reads artificial ground truth dataset
3. Extracts labels for current period
4. Saves period-specific labels to JSON
5. Triggers next job: `03_check_model.py`

**Environment Variables**:
- `PERIOD`: Current period number (0, 1, 2, ...) - inherited from previous job
- `MODEL_NAME`: Deployed model name (default: "banking_campaign_predictor")
- `PROJECT_NAME`: CML project name (default: "CAI Baseline MLOPS")

**Input**:
- `data/ground_truth_metadata.json`
- `data/artificial_ground_truth_data.csv`

**Output**:
- `data/current_period_ground_truth.json` (labels for this period)
- Triggers `03_check_model.py`

---

### 03_check_model.py

**Status**: CML Job (repeatable, one per period)

**Purpose**: Validates model accuracy and orchestrates next action

**What it does**:
1. Loads predictions from `01_get_predictions.py`
2. Loads ground truth from `02_load_ground_truth.py`
3. Calculates accuracy metrics:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
4. Compares to previous period (degradation detection)
5. Decision logic:
   - **If degraded**: Flag alert and EXIT pipeline
   - **If last period**: EXIT pipeline successfully
   - **Otherwise**: Trigger `01_get_predictions.py` for next period

**Environment Variables**:
- `PERIOD`: Current period
- `ACCURACY_THRESHOLD`: Minimum acceptable accuracy (default: 0.85 = 85%)
- `DEGRADATION_THRESHOLD`: Max drop before alert (default: 0.05 = 5%)
- `MODEL_NAME`: Deployed model name (default: "banking_campaign_predictor")
- `PROJECT_NAME`: CML project name (default: "CAI Baseline MLOPS")

**Input**:
- `data/current_period_ground_truth.json`
- `data/predictions_period_{PERIOD}.json`
- `data/check_model_results.json` (previous period, optional)
- `data/ground_truth_metadata.json`

**Output**:
- `data/check_model_results.json` (accuracy report)
- CML metrics (via `cdsw.track_aggregate_metrics()`)
- Job trigger (either continue to next period or exit)

**Decision Matrix**:

| Condition | Action |
|-----------|--------|
| Accuracy degraded > threshold | EXIT (degradation alert) |
| Accuracy < minimum threshold | EXIT (degradation alert) |
| Last period reached | EXIT (success) |
| None of above | Continue to next period |

**Metrics Tracked to CML**:
```python
cdsw.track_aggregate_metrics(
    {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "period": period
    },
    model_deployment_crn=cr_number,
)
```

---

## Setup Instructions

### Step 1: Prepare Artificial Data (One-time)

```bash
cd /home/cdsw/module2
python 00_prepare_artificial_data.py
```

This creates:
- `data/artificial_ground_truth_data.csv` (5000+ rows with 5 periods)
- `data/ground_truth_metadata.json` (configuration)

### Step 2: Create CML Jobs

In CML project, create 3 jobs:

**Job 1: get_predictions**
- Script: `01_get_predictions.py`
- Environment: `PERIOD=0`, `BATCH_SIZE=50`
- Schedule: Manual trigger (or scheduled)

**Job 2: load_ground_truth**
- Script: `02_load_ground_truth.py`
- Environment: (inherits PERIOD from previous job)
- Schedule: Triggered by get_predictions

**Job 3: check_model**
- Script: `03_check_model.py`
- Environment: `ACCURACY_THRESHOLD=0.85`
- Schedule: Triggered by load_ground_truth

### Step 3: Start Monitoring Pipeline

Run the first job manually to start the pipeline:

```bash
# Via CML UI: Run "get_predictions" job with PERIOD=0
# Or via API: cmlapi.create_job_run(project_id=..., job_id=...)
```

The pipeline will then automatically orchestrate:
1. Get predictions (period 0, batch processing)
2. Load ground truth (period 0, load labels)
3. Check model (validates accuracy, decides next action)
4. If not degraded and not last period, repeat starting with get_predictions for periods 1-4

---

## Monitoring Accuracy Over Time

The pipeline tracks accuracy degradation across periods:

**Expected Results** (with 20 periods, 2.5% drop per period, 85% threshold):
- Period 0: ~95% accuracy ✓ PASS
- Period 1-3: ~92-87% accuracy ✓ PASS (gradual decline)
- Period 4: ~85% accuracy ✓ PASS (at threshold)
- Period 5+: <85% accuracy → Pipeline may exit based on degradation logic
- Period 19: ~52% accuracy (final period, if reached)

**Real-world Interpretation**:
- First few periods: Model performs well
- Later periods: Model accuracy degrades as data distribution changes
- Alert triggers: When accuracy drop exceeds threshold
- Pipeline stops: When degradation detected or final period reached

---

## Customization

### Number of Periods

The number of periods is calculated automatically:
```python
num_periods = total_samples / batch_size
# Example: 1000 samples / 50 batch = 20 periods
```

If you want to change it, edit `00_prepare_artificial_data.py`:
```python
num_periods = 20  # This is calculated as total_samples / batch_size
```

**Important**: The periods should match your batch processing pattern to ensure each job runs once per period.

### Adjust Degradation Rate

Edit `00_prepare_artificial_data.py`:
```python
degradation_rate = 0.05  # 5% drop per period
```

### Adjust Accuracy Thresholds

Edit `03_check_model.py` environment or script:
```python
ACCURACY_THRESHOLD = 0.85        # Exit if below 85%
DEGRADATION_THRESHOLD = 0.05     # Exit if drops > 5%
```

### Adjust Batch Size

Edit `02_get_predictions.py` environment:
```
BATCH_SIZE=100  # Process 100 samples per batch
```

---

## Architecture Notes

### Why Separate Scripts?

1. **01_load_ground_truth.py**: Independent ground truth loading
   - Not embedded in prediction script
   - Can be reused for different data sources
   - Keeps concerns separated

2. **02_get_predictions.py**: Batch prediction with tracking
   - Follows example_get_predictions.py API pattern
   - Uses `cdsw.track_delayed_metrics()` for individual predictions
   - Configurable batch size for scalability

3. **03_check_model.py**: Validation and orchestration
   - Follows example_check_model.py API pattern
   - Uses `cdsw.track_aggregate_metrics()` for period-level metrics
   - Controls pipeline flow (continue vs exit)

### API Patterns

The scripts follow Cloudera ML API patterns:

**From example_get_predictions.py**:
- `cdsw.track_delayed_metrics()` for individual prediction tracking
- Batch processing with configurable size
- Job orchestration via `cmlapi`

**From example_check_model.py**:
- `cdsw.track_aggregate_metrics()` for performance metrics
- Model deployment CRN retrieval
- Job triggering with error handling

---

## Files Generated During Execution

```
data/
├── artificial_ground_truth_data.csv      # Initial: Engineered data + labels
├── ground_truth_metadata.json            # Initial: Period configuration
├── current_period_ground_truth.json      # Per period: Labels for current period
├── predictions_period_0.json             # Per period: Predictions with metadata
├── predictions_period_1.json
├── predictions_period_2.json
└── check_model_results.json              # Per period: Accuracy metrics & decisions
```

---

## Troubleshooting

### Missing Artificial Data

**Error**: `FileNotFoundError: Groundtruth data not found`

**Solution**: Run `00_prepare_artificial_data.py` first

```bash
python 00_prepare_artificial_data.py
```

### CML Job Not Triggering Next Job

**Issue**: Jobs don't automatically chain

**Cause**: CML job not found or API key not configured

**Solution**:
1. Check job names match exactly: `load_ground_truth`, `get_predictions`, `check_model`
2. Verify `CDSW_API_URL` and `CDSW_APIV2_KEY` are set in CML
3. Check job logs for `cmlapi` errors
4. Manually trigger next job if automatic triggering fails

### PERIOD Not Passed Correctly

**Issue**: Job runs but uses PERIOD=0 for all jobs

**Cause**: Environment variable not being passed between jobs

**Solution**:
1. Set `PERIOD` in each job's environment
2. Or hardcode period progression in each job

---

## Integration with Module 1

Module 2 uses data generated by Module 1:

```
Module 1 Output:
├── engineered_inference_data.csv    → Used by 00_prepare_artificial_data.py
└── predictions.csv                 → Used by 00_prepare_artificial_data.py

Module 2 Uses:
└── Creates artificial labels based on Module 1 predictions
    └── Simulates model degradation for monitoring demo
```

---

## Next Steps

Once monitoring is working:

1. **Real Deployment**: Replace artificial data with real predictions
2. **Live Models**: Integrate with actual deployed model endpoints
3. **Real Ground Truth**: Load actual labels as they become available
4. **Alerting**: Add email/Slack notifications when degradation detected
5. **Retraining**: Trigger Model retraining when degradation detected

---

## Example CML Job Configuration

### Get Predictions Job

```yaml
Name: get_predictions
Script: module2/01_get_predictions.py
Runtime: Python 3.9+
Environment:
  PERIOD: 0
  BATCH_SIZE: 50
  MODEL_NAME: LSTM-2
  PROJECT_NAME: SDWAN
Trigger: Manual (or scheduled)
```

### Load Ground Truth Job

```yaml
Name: load_ground_truth
Script: module2/02_load_ground_truth.py
Runtime: Python 3.9+
Environment:
  MODEL_NAME: LSTM-2
  PROJECT_NAME: SDWAN
Trigger: By get_predictions job
```

### Check Model Job

```yaml
Name: check_model
Script: module2/03_check_model.py
Runtime: Python 3.9+
Environment:
  ACCURACY_THRESHOLD: 0.85
  DEGRADATION_THRESHOLD: 0.05
  MODEL_NAME: LSTM-2
  PROJECT_NAME: SDWAN
Trigger: By load_ground_truth job
```

---

## Questions?

Refer to the inline documentation in each script for detailed implementation notes.

---

## Step-by-Step Guide (Deprecated - Original Content Below)

### Step 1: Understanding Model Persistence

**Purpose:** Understand where your model's predictions are stored and how to query production data.

**What You'll Learn:**
Every time a prediction request is made to your deployed model, both the request and response are automatically logged to PostgreSQL. This creates a complete audit trail of everything your model has done in production.

**What Gets Logged:**
1. **Request Data** - The features sent to the model
2. **Prediction Output** - The predicted class and probabilities
3. **Metadata** - Timestamp, request ID, API endpoint version
4. **Performance Metadata** - Response time, latency

**What's Stored in PostgreSQL:**

The `inference_logs` table contains:
```
Column                  | Type      | Description
------------------------|-----------|----------------------------------------
request_id              | UUID      | Unique identifier for this prediction
timestamp               | DATETIME  | When the prediction was made
customer_id             | INT       | Which customer (traceable to original data)
model_version           | STRING    | Which model version made the prediction
prediction              | INT       | The prediction (0 or 1)
prediction_probability_0| FLOAT     | Confidence in class 0
prediction_probability_1| FLOAT     | Confidence in class 1
actual_label            | INT NULL  | What actually happened (filled later)
actual_timestamp        | DATETIME  | When we learned the actual outcome
```

**Why This Matters:**
In production, you can't trust metrics from old test data. You need to:
- Track every decision the model makes
- Compare predictions to real outcomes
- Detect when patterns change
- Calculate performance on production data
- Maintain a complete audit trail for compliance

**To Run:**
```bash
cd module2
# Open the notebook in your Cloudera AI project
# Click on: 01_explore_inference_logs.ipynb
# Run all cells to examine the inference logs
```

**What You'll Discover:**
- How many predictions have been made
- Date ranges of production data
- Distribution of predictions (how often does it predict class 0 vs 1)
- Current null values in the actual_label column (records waiting for ground truth)

---

### Step 2: Tracking Ground Truth Updates

**Purpose:** Update prediction records with actual outcomes, enabling performance calculation.

**The Challenge:**
In our bank marketing example, we need to know which customers actually subscribed to term deposits. This information arrives later than the prediction:
- Prediction is made: Day 1 (customer receives marketing)
- Actual outcome happens: Days 1-30 (customer may subscribe during campaign)
- We learn the outcome: Days 31+ (bank's systems record whether they subscribed)

**What This Script Does:**

The `02_1_update_ground_truth.py` script simulates receiving ground truth labels:

1. **Load inference logs** from PostgreSQL
2. **Generate fake ground truth labels** to simulate actual business outcomes
3. **Update PostgreSQL records** with these actual labels
4. **Track which records got updated** and the timing

**To Run:**
```bash
python 02_1_update_ground_truth.py
```

**Expected Output:**
```
==================== GROUND TRUTH UPDATE ====================
Loading inference logs from PostgreSQL...
✓ Found 1,000 prediction records

Simulating ground truth outcomes...
✓ Generated 250 positive labels (25% subscription rate)
✓ Generated 750 negative labels (75% non-subscription rate)

Updating PostgreSQL with actual labels...
✓ Updated 1,000 records in inference_logs table
✓ Update timestamps recorded for audit trail

Performance after update:
- Records with ground truth: 1,000 / 1,000 (100%)
- Average confidence: 0.67
=============================================================
```

**What Gets Changed:**
- `actual_label` column is filled with 0s and 1s
- `actual_timestamp` is recorded
- A new `labeled_date` column tracks when we learned each outcome

**Important:** In a real production system, these would be actual business outcomes from systems like:
- CRM systems (customer took the action or not)
- Transaction databases (account opened or closed)
- Campaign tracking systems (customer engaged with offer)

For this lab, we simulate realistic label distributions to show how performance monitoring works.

**Why This Matters:**
You can't improve a model if you don't know what actually happened:
- Without ground truth, you can't calculate accuracy, precision, recall, F1
- Without ground truth, you can't detect performance degradation
- Without ground truth, you can't make informed decisions about retraining

---

### Step 3: Statistical Testing for Performance Degradation

**Purpose:** Detect when model performance drops significantly, which would trigger a retraining job.

**The Problem:**
Your model's performance doesn't just gradually decline—it can suddenly drop due to:
- Data drift (new customer segments you haven't seen)
- Concept drift (subscription patterns change)
- Seasonality (different behavior in Q4)
- Business changes (new competitor, regulation changes)

You need an automated way to detect these changes quickly.

**What This Script Does:**

The `02_2_performance_degradation_check.py` script:

1. **Segments data by time period**
   - Recent period (last N days): Current model performance
   - Historical baseline: Expected model performance

2. **Calculates performance metrics**
   - Accuracy, Precision, Recall, F1 Score
   - ROC-AUC scores
   - Confusion matrices

3. **Runs statistical tests**
   - Chi-square test for prediction distribution changes
   - T-test for probability score differences
   - Compares recent vs. historical performance

4. **Determines if degradation occurred**
   - Sets significance level (alpha = 0.05)
   - Reports p-values and effect sizes
   - Flags concerning patterns

**To Run:**
```bash
python 02_2_performance_degradation_check.py
```

**Expected Output:**
```
==================== PERFORMANCE DEGRADATION CHECK ====================

Historical Baseline (all data):
  F1 Score: 0.521
  Accuracy: 0.764
  Precision: 0.512
  Recall: 0.531
  ROC-AUC: 0.856

Recent Period (last 7 days):
  F1 Score: 0.489
  Accuracy: 0.751
  Precision: 0.485
  Recall: 0.508
  ROC-AUC: 0.823

Statistical Test Results:
  ✓ Chi-square test (prediction distribution): p=0.089 (no significant change)
  ✓ T-test (probability scores): p=0.156 (no significant change)
  ✓ Prediction distribution drift: MODERATE
    - Baseline: 25% positive predictions
    - Recent: 28% positive predictions
    - Change: +3pp

CONCLUSION: No statistical degradation detected.
  Status: OK - Model performing within expected range

Note: This job runs but does NOT trigger retraining yet.
You'll implement automatic retraining in a future module.
=========================================================================
```

**Understanding Statistical Testing:**

- **P-value > 0.05**: No significant difference detected (model still working well)
- **P-value < 0.05**: Significant change detected (investigate and consider retraining)
- **Effect size**: How big the difference is (not just whether it exists)

**What Doesn't Trigger Retraining Yet:**
- This script detects degradation but doesn't start a retraining job
- In a future module, you'll set up automated retraining when thresholds are crossed

**Why This Matters:**
- Automated detection prevents stale models from hurting business
- Statistical rigor prevents false alarms
- Audit trail shows when degradation was detected
- Historical comparison ensures you're improving, not just changing

---

## Monitoring Dashboards & Grafana

**Purpose:** Visualize model performance and prediction patterns in real-time.

**After completing the steps above, you'll explore Grafana dashboards that show:**

1. **Prediction Volume Over Time**
   - How many predictions per day/hour
   - Identify anomalous patterns
   - Spot when prediction traffic drops

2. **Prediction Distribution**
   - What percentage predict positive vs negative
   - Detect shifts in prediction distribution
   - Monitor for label imbalance drift

3. **Performance Metrics Over Time**
   - Accuracy trend
   - F1 score trend
   - ROC-AUC progression
   - Precision/Recall tradeoff

4. **Confidence Score Patterns**
   - Average prediction confidence
   - Distribution of high-confidence vs low-confidence predictions
   - Identify when model becomes uncertain

5. **Latency Monitoring**
   - Average response time
   - P95 and P99 latencies
   - Identify performance bottlenecks

**To Access Dashboards:**
1. Go to your Cloudera AI project
2. Find the "Monitoring" or "Grafana" section
3. Look for "bank_marketing_model" dashboards
4. Explore different time ranges and metrics

**Key Metrics to Watch:**
- **Recall**: Did we catch all the customers who would subscribe?
- **Precision**: When we predict "will subscribe," are we right?
- **F1 Score**: Overall model health
- **Prediction volume**: Is the model being used?
- **Latency**: Is the model responding quickly?

---

## Production Data Flow

Here's how data flows through a production ML system with monitoring:

```
┌─────────────────────────────────────────────────────────────┐
│              STEP 1: PREDICTIONS IN PRODUCTION              │
│  New Customer Data → API Request → Model → Prediction       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────────┐
        │ PostgreSQL Logs          │
        │ • Request data           │
        │ • Prediction output      │
        │ • Timestamp              │
        │ • actual_label = NULL    │
        └──────────────┬───────────┘
                       │
        ┌──────────────┴─────────────────────┐
        │                                    │
        ↓                                    ↓
┌──────────────────────┐         ┌──────────────────────┐
│  STEP 2: GROUND TRUTH│         │ STEP 3: MONITORING   │
│  Actual outcomes     │         │ Track performance    │
│  arrive from business│         │ Visualize trends     │
│  systems...          │         │ (Grafana)            │
└──────────────────────┘         └──────────────────────┘
        │
        ↓
 ┌─────────────────────┐
 │ Update PostgreSQL   │
 │ actual_label = 0/1  │
 │ actual_timestamp    │
 └──────────┬──────────┘
            │
            ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: PERFORMANCE ANALYSIS                               │
│  Calculate metrics on recent data                           │
│  Run statistical tests                                      │
│  Compare to baseline                                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ↓                     ↓
  Performance OK?        Performance Degraded?
        │                     │
        ↓                     ↓
  Continue running    [Future: Trigger retraining]
  existing model      [Retrain with recent data]
                      [Deploy new model version]
```

**Key Principles:**
- **Continuous logging**: Every prediction is recorded
- **Delayed ground truth**: Actual outcomes arrive later than predictions
- **Automated monitoring**: Statistical tests run on schedule
- **Action triggers**: Bad performance triggers responses (later in curriculum)
- **Audit trail**: Complete history of what the model did and why

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: "PostgreSQL connection refused"

**Solution:**
Ensure PostgreSQL is running and accessible:
```bash
# Check connection
psql -h localhost -U mlops -d inference_logs
```
If it fails, verify:
1. PostgreSQL is running on your system
2. Database credentials are correct
3. Network access is configured

#### Issue: "No inference logs found"

**Solution:**
Make sure you completed Module 1 and made predictions. The logs come from:
1. Running `05_2_inference_predict.py` in Module 1
2. This logs predictions to PostgreSQL

If you skipped inference, run it first:
```bash
cd ../module1
python 05_2_inference_predict.py
```

#### Issue: "actual_label column all NULL"

**Solution:**
This is expected before running the ground truth update script. Run:
```bash
python 02_1_update_ground_truth.py
```
This populates the actual outcomes.

#### Issue: "Statistical test fails with 'insufficient data'"

**Solution:**
Statistical tests need enough records. Make sure:
1. You updated ground truth (Step 2)
2. You have at least 100+ records with labels
3. Recent period has meaningful data

#### Issue: "Grafana dashboard not showing data"

**Solution:**
Dashboards may lag behind data updates. Try:
1. Refresh the browser page
2. Check that time range includes your data
3. Verify data was written to PostgreSQL
4. Wait a few minutes for metric aggregation

#### Issue: Module import errors

**Solution:**
Install required dependencies:
```bash
pip install pandas scikit-learn scipy psycopg2 numpy
```

### Getting Help

If you encounter other issues:
1. Check error messages carefully—they usually describe the problem
2. Review the relevant step above for that section
3. Verify data files exist and are accessible
4. Check database connections and credentials
5. Ask your instructor or consult Cloudera AI documentation

---

## Summary: What You've Accomplished

By completing this module, you've set up production ML operations:

✅ **Explored inference logs** - Understood what gets stored in production
✅ **Updated ground truth** - Collected actual outcomes for evaluation
✅ **Detected performance degradation** - Used statistics to identify problems
✅ **Visualized metrics** - Monitored model health in Grafana
✅ **Set up monitoring** - Established continuous observation

This is how real-world ML systems stay healthy. You're no longer just building models—you're operating them responsibly.

---

## Key Takeaways

1. **Production never stops changing** - Your model's environment is dynamic
2. **You need complete data lineage** - Track predictions, outcomes, and metrics
3. **Automation is critical** - Manual monitoring doesn't scale
4. **Statistics matter** - Test rigorously before taking action
5. **Transparency is essential** - Audit trails protect both users and your team

---

## Next Steps (Future Modules)

- **Module 3**: Automated retraining - Trigger model retraining when performance drops
- **Advanced**: A/B testing different models in production
- **Advanced**: Canary deployments (gradually roll out new models)
- **Advanced**: Handling data drift and concept drift
- **Advanced**: Multi-model serving and versioning strategies

---

**Next: Module 3!** 🚀

Once you're comfortable with monitoring, you'll learn how to automatically retrain and redeploy models based on these signals.

For more information, visit the [Cloudera AI Documentation](https://docs.cloudera.com/cml/).
