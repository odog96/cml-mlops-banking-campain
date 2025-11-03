# Module 2: Model Monitoring with Degradation Detection

This module implements a comprehensive model monitoring pipeline that tracks predictions over time periods and automatically detects when model accuracy degrades. The pipeline uses artificial ground truth data with intentional accuracy degradation to simulate real-world model drift.

## Overview

The monitoring pipeline consists of **2 CML Jobs**:

```
Job 1: 02_prepare_artificial_data.py (One-time setup)
    ↓
Job 2: 03_monitoring_pipeline.py (Integrated Monitoring)
    ├─ Period 0: Get predictions → Load ground truth → Check model
    ├─ Period 1: Get predictions → Load ground truth → Check model
    ├─ Period 2: Get predictions → Load ground truth → Check model
    └─ ... (repeats for each period until degradation or end of data)
```

**Job Execution Pattern**:
- Number of periods = Total data size / Batch size
- With 1000 samples and batch size 250 = 4 periods
- `03_monitoring_pipeline.py` runs **once** and processes **all periods sequentially**
- Single job manages period state internally (no external state files needed)
- Automatically detects degradation and exits gracefully
- All results saved to `data/monitoring_results.json`

## Key Features

- **Self-Contained Pipeline**: Single job manages all periods internally (no external state files or job chaining)
- **Period-based Processing**: Splits data into multiple time periods for sequential monitoring
- **Batch Predictions**: Processes predictions in configurable batch sizes
- **Cloudera Integration**: Tracks metrics using `cml.track_delayed_metrics()` and `cml.track_aggregate_metrics()`
- **Degradation Detection**: Automatically detects statistically significant accuracy drops
- **Configurable Thresholds**: Adjust accuracy requirements and degradation sensitivity
- **Comprehensive Logging**: Logs to both console and file (`data/monitoring_log.txt`)

## Quick Start

1. **Prepare artificial data** (one-time):
   ```bash
   python module2/02_prepare_artificial_data.py
   ```

2. **Create CML jobs** using the `01_create_jobs.ipynb` notebook:
   - Creates Job 1: Prepare Artificial Data
   - Creates Job 2: Monitor Pipeline (integrated)

3. **Start monitoring**:
   ```bash
   # Run Job 1 (Prepare Artificial Data) manually via CML UI or API
   # Job 2 (Monitor Pipeline) processes all periods sequentially
   # Monitor results in data/monitoring_results.json
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

### 02_prepare_artificial_data.py

**Status**: One-time setup (run via CML Job 1)

**Purpose**: Creates the artificial ground truth dataset with intentional accuracy degradation

**What it does**:
1. Reads BATCH_SIZE from top of script (hardcoded)
2. Loads engineered inference data from Module 1
3. Loads predictions from Module 1
4. Calculates num_periods = actual_total_samples / BATCH_SIZE
5. Calculates degradation_rate to spread degradation evenly across all periods
6. Creates artificial ground truth labels with progressive corruption
7. Saves `artificial_ground_truth_data.csv` and `ground_truth_metadata.json`

**Configuration** (hardcoded at top of script):
```python
BATCH_SIZE = 250  # Adjust based on how many periods you want
```

**Output**:
- `data/artificial_ground_truth_data.csv` (full dataset with labels)
- `data/ground_truth_metadata.json` (period boundaries)

**Run** (via CML Job 1):
```bash
python module2/02_prepare_artificial_data.py
```

---

### 03_monitoring_pipeline.py

**Status**: CML Job (Job 2 - the main monitoring pipeline)

**Purpose**: Consolidated monitoring pipeline that processes all periods sequentially

**What it does**:
1. **Phase 0 - Setup**: Initialize CML client, load metadata, load full dataset
2. **Loop through all periods** (Period 0 to Period N):
   - **Phase 1 - Get Predictions**:
     - Extract period data
     - Process predictions in batches
     - Track metrics via `cml.track_delayed_metrics()`
     - Save predictions to JSON
   - **Phase 2 - Load Ground Truth**:
     - Extract period labels
     - Save labels to JSON
   - **Phase 3 - Check Model**:
     - Calculate accuracy metrics (accuracy, precision, recall, F1)
     - Compare to previous period (degradation detection)
     - Check if accuracy below threshold
     - Track metrics via `cml.track_aggregate_metrics()`
     - **Decision**:
       - **If degraded**: Save results and exit gracefully (job completes with status 0)
       - **If last period**: Save results and exit successfully
       - **Otherwise**: Continue to next period

**Environment Variables**:
- `BATCH_SIZE`: Samples per batch (default: 250)
- `MODEL_NAME`: Deployed model name (default: "banking_campaign_predictor")
- `PROJECT_NAME`: CML project name (default: "CAI Baseline MLOPS")
- `ACCURACY_THRESHOLD`: Minimum acceptable accuracy (default: 0.85 = 85%)
- `DEGRADATION_THRESHOLD`: Max drop before alert (default: 0.05 = 5%)

**Command-line Arguments**:
```bash
# Run all periods (default 0 to total)
python 03_monitoring_pipeline.py

# Run specific period range
python 03_monitoring_pipeline.py --start-period 0 --end-period 3

# Run single period
python 03_monitoring_pipeline.py --start-period 2 --end-period 2
```

**Input**:
- `data/artificial_ground_truth_data.csv` (full dataset)
- `data/ground_truth_metadata.json` (period configuration)

**Output**:
- `data/predictions_period_{PERIOD}.json` (predictions per period)
- `data/period_{PERIOD}_ground_truth.json` (labels per period)
- `data/monitoring_results.json` (final summary with status)
- `data/monitoring_log.txt` (detailed execution log)

**Key Advantages**:
- ✅ **Single job** - No job chaining or state file dependencies
- ✅ **Self-contained** - All period state managed internally
- ✅ **Graceful degradation** - Exits with status 0 even when degradation detected
- ✅ **Comprehensive logging** - All events logged to console and file
- ✅ **Flexible execution** - Can run specific period ranges via command-line args

---

## Setup Instructions

### Step 1: Create CML Jobs Using Notebook

Run the `01_create_jobs.ipynb` notebook to create 2 CML jobs:

```bash
cd /home/cdsw/module2

# Open the notebook in your CML project
# Run all cells to create the 2 jobs:
# - Job 1: Prepare Artificial Data (02_prepare_artificial_data.py)
# - Job 2: Monitor Pipeline (03_monitoring_pipeline.py)
```

The notebook will:
1. Authenticate with CML API
2. Query available ML runtimes
3. Create Job 1: `02_prepare_artificial_data.py`
4. Create Job 2: `03_monitoring_pipeline.py`

**Job Configuration:**

**Job 1: Prepare Artificial Data**
- Script: `module2/02_prepare_artificial_data.py`
- CPU: 1 core
- Memory: 2 GB
- Environment: (uses defaults)

**Job 2: Monitor Pipeline**
- Script: `module2/03_monitoring_pipeline.py`
- CPU: 2 cores
- Memory: 4 GB
- Environment: (uses defaults, can be customized)

### Step 2: Configure BATCH_SIZE (Optional)

Edit the `BATCH_SIZE` at the top of `02_prepare_artificial_data.py`:
```python
BATCH_SIZE = 250  # Adjust based on desired number of periods
```

The script will automatically:
1. Load actual data from module1
2. Calculate `num_periods = actual_total_samples / BATCH_SIZE`
3. Determine degradation rate based on number of periods

### Step 3: Start Monitoring Pipeline

Run Job 1 to start the entire monitoring workflow:

```bash
# Via CML UI:
# 1. Go to Jobs tab
# 2. Click on "Mod 2 Job 1: Prepare Artificial Data"
# 3. Click "Run Now"
# 4. Wait for completion

# Then run Job 2:
# 1. Click on "Mod 2 Job 2: Monitor Pipeline"
# 2. Click "Run Now"
# 3. Monitor execution in the job logs
```

Job 2 will automatically:
1. Load prepared data from Job 1
2. Process all periods sequentially
3. Detect degradation (if any)
4. Save results to `data/monitoring_results.json`
5. Log all activity to `data/monitoring_log.txt`

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

### Single Job vs. Multiple Jobs

The original design used **4 separate jobs**:
- Job 1: Prepare Artificial Data
- Job 2: Get Predictions (triggered per period)
- Job 3: Load Ground Truth (triggered per period)
- Job 4: Check Model (triggered per period)

The new design consolidates to **2 jobs**:
- Job 1: Prepare Artificial Data (one-time)
- Job 2: Monitor Pipeline (processes all periods in one execution)

**Why this is better:**
1. ✅ **Simpler**: Single job manages all periods
2. ✅ **Reliable**: No job chaining failures or state passing issues
3. ✅ **Faster**: No overhead from starting multiple jobs
4. ✅ **Cleaner logs**: All output in one execution
5. ✅ **Self-contained**: No external state file dependencies

### API Patterns Used

The scripts follow Cloudera ML API patterns:

**CML Integration**:
- `cml.track_delayed_metrics()` for individual prediction tracking
- `cml.track_aggregate_metrics()` for period-level metrics
- Model deployment CRN retrieval
- Optional job orchestration via `cmlapi`

**Internal State Management**:
- Period tracking via loop counter (no external files)
- Results saved to JSON for inspection
- Comprehensive logging to console and file

---

## Files Generated During Execution

```
data/
├── artificial_ground_truth_data.csv      # Created by Job 1: Engineered data + labels
├── ground_truth_metadata.json            # Created by Job 1: Period configuration
├── predictions_period_0.json             # Created by Job 2: Predictions per period
├── predictions_period_1.json
├── predictions_period_2.json
├── period_0_ground_truth.json            # Created by Job 2: Labels per period
├── period_1_ground_truth.json
├── period_2_ground_truth.json
├── monitoring_results.json               # Created by Job 2: Final summary with all results
└── monitoring_log.txt                    # Created by Job 2: Detailed execution log
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

### Job 1: Prepare Artificial Data

```yaml
Name: Mod 2 Job 1: Prepare Artificial Data
Script: module2/02_prepare_artificial_data.py
Runtime: Python 3.10 (2024.10+)
CPU: 1 core
Memory: 2 GB
Environment:
  (uses defaults from script)
Trigger: Manual (UI or API)
```

### Job 2: Monitor Pipeline

```yaml
Name: Mod 2 Job 2: Monitor Pipeline
Script: module2/03_monitoring_pipeline.py
Runtime: Python 3.10 (2024.10+)
CPU: 2 cores
Memory: 4 GB
Environment:
  BATCH_SIZE: 250 (optional, uses default if not set)
  MODEL_NAME: banking_campaign_predictor (optional)
  PROJECT_NAME: CAI Baseline MLOPS (optional)
  ACCURACY_THRESHOLD: 0.85 (optional)
  DEGRADATION_THRESHOLD: 0.05 (optional)
Trigger: Manual (UI or API)
Arguments: (optional)
  --start-period 0
  --end-period 3
```

**Note**: Environment variables are optional; script has sensible defaults for all of them.

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
