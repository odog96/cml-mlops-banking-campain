# Module 2 Critical Fix: PERIOD Environment Variable Passing

## The Issue (Root Cause)

**Problem:** Jobs 3.1 and 3.2 were NOT explicitly passing the `PERIOD` environment variable when triggering the next job.

**Why This Is Critical:**
Cloudera ML (CML) does NOT automatically inherit parent job's environment variables. Without explicit passing, triggered jobs default to their hardcoded defaults.

**Consequence:**
- Job 3.1 (Period 0) triggers Job 3.2 without PERIOD
- Job 3.2 defaults to PERIOD=0 (from environment default)
- Job 3.2 triggers Job 3.3 without PERIOD
- Job 3.3 defaults to PERIOD=0 (from environment default)
- Job 3.3 decides to continue → triggers Job 3.1 with PERIOD=1
- Job 3.1 now runs with PERIOD=1
- But then the cycle repeats with wrong period boundaries!

**Result:** Unpredictable behavior, data misalignment, or infinite loops

---

## The Solution (Implementation)

### Fix #1: Job 03.1_get_predictions.py
**Changed:** Added explicit environment variable passing in `trigger_load_ground_truth_job()`

**Before:**
```python
job_run_request = cmlapi.CreateJobRunRequest()
job_run = client.create_job_run(job_run_request, ...)  # No env vars!
```

**After:**
```python
job_run_request = cmlapi.CreateJobRunRequest()
job_run_request.environment_variables = {
    "PERIOD": str(PERIOD)  # Explicitly pass current period
}
job_run = client.create_job_run(job_run_request, ...)
```

**Lines Changed:** ~355-365

---

### Fix #2: Job 03.2_load_ground_truth.py
**Changed:** Added explicit environment variable passing in `trigger_next_job()`

**Before:**
```python
job_run_request = cmlapi.CreateJobRunRequest()
job_run = client.create_job_run(job_run_request, ...)  # No env vars!
```

**After:**
```python
job_run_request = cmlapi.CreateJobRunRequest()
job_run_request.environment_variables = {
    "PERIOD": str(PERIOD)  # Explicitly pass current period
}
job_run = client.create_job_run(job_run_request, ...)
```

**Lines Changed:** ~167-174

---

### Fix #3: Enhanced Comments
**Changed:** Added detailed comments in all three jobs explaining:

1. **Job 03.1_get_predictions.py (lines 49-68)**
   - Why BATCH_SIZE must match between Job 02 and this job
   - How period boundaries depend on BATCH_SIZE
   - The critical importance of matching configuration

2. **Job 03.2_load_ground_truth.py (lines 39-61)**
   - PERIOD is explicitly passed from Job 03.1
   - Explains why explicit passing is needed
   - Shows the complete PERIOD flow through pipeline

3. **Job 03.3_check_model.py (lines 48-62)**
   - PERIOD is explicitly passed from Job 03.2
   - This job is unique: it increments PERIOD for next iteration
   - Explains why CML doesn't auto-inherit environment variables

---

## How PERIOD Now Flows Through the Pipeline

### Correct Flow (After Fix):
```
Period 0:
  Job 3.1 receives PERIOD=0 (environment or previous trigger)
    ↓ creates predictions_period_0.json
    ↓ triggers Job 3.2 with environment_variables={"PERIOD": "0"}

  Job 3.2 receives PERIOD=0 (explicitly passed)
    ↓ creates current_period_ground_truth.json
    ↓ triggers Job 3.3 with environment_variables={"PERIOD": "0"}

  Job 3.3 receives PERIOD=0 (explicitly passed)
    ↓ validates accuracy, compares to previous period
    ↓ decision: continue to period 1
    ↓ triggers Job 3.1 with environment_variables={"PERIOD": "1"}

Period 1:
  Job 3.1 receives PERIOD=1 (explicitly passed from Job 3.3)
    ↓ creates predictions_period_1.json
    ↓ triggers Job 3.2 with environment_variables={"PERIOD": "1"}

  Job 3.2 receives PERIOD=1 (explicitly passed)
    ↓ creates current_period_ground_truth.json
    ↓ triggers Job 3.3 with environment_variables={"PERIOD": "1"}

  ... (continues until last period or degradation detected)
```

---

## Key Insight: The Three Different Behaviors

| Job | Passes To | Passes What | Purpose |
|-----|-----------|------------|---------|
| Job 3.1 | Job 3.2 | PERIOD (same) | Horizontal transition: predictions → labels |
| Job 3.2 | Job 3.3 | PERIOD (same) | Horizontal transition: labels → validation |
| Job 3.3 | Job 3.1 | PERIOD+1 (incremented) | Vertical transition: next period loop |

**The Increment:**
Only Job 3.3 increments PERIOD when continuing. This is the orchestration logic that drives the pipeline forward through time periods.

---

## Why This Matters

### Without the Fix:
- Pipeline behavior is undefined/unpredictable
- May infinite loop on period 0
- May have data misalignment between predictions and ground truth
- Difficult to debug because environment variables "disappear" between jobs

### With the Fix:
- Explicit, deterministic pipeline progression
- Each job knows exactly which period it's processing
- Data alignment guaranteed
- Clear audit trail: can see PERIOD in job logs

---

## Testing the Fix

### Manual Test (Before/After):
1. Run Job 3.1 with PERIOD=0
2. Check Job 3.2's logs - what PERIOD does it see?
   - **Before Fix:** PERIOD not set (logs show default "Processing Period: 0" but may be wrong)
   - **After Fix:** PERIOD=0 (explicitly passed, logs show "Processing Period: 0")
3. Check Job 3.3's logs - what PERIOD does it see?
   - **Before Fix:** PERIOD not set properly
   - **After Fix:** PERIOD=0 (explicitly passed)
4. Check if Job 3.1 is triggered with PERIOD=1
   - **Before Fix:** May trigger with PERIOD=0 again (infinite loop)
   - **After Fix:** Triggers with PERIOD=1 (correct progression)

### Automated Test Script:
See `/home/cdsw/test_env_inheritance.py` for a test that demonstrates the CML environment variable inheritance behavior.

---

## Related Files

- `/home/cdsw/module2/03.1_get_predictions.py` - Fixed trigger function (lines 355-365)
- `/home/cdsw/module2/03.2_load_ground_truth.py` - Fixed trigger function (lines 167-174)
- `/home/cdsw/module2/03.3_check_model.py` - Already had fix, enhanced comments (lines 312-318)
- `/home/cdsw/test_env_inheritance.py` - Test script demonstrating the issue

---

## Checklist for CML Job Setup

When creating CML jobs, verify:

- [ ] Job 1 name: **"Get Predictions"** (exact match)
- [ ] Job 2 name: **"Load Ground Truth"** (exact match)
- [ ] Job 3 name: **"Check Model"** (exact match)
- [ ] All jobs have BATCH_SIZE environment variable (matching value)
- [ ] Job 1 initial PERIOD=0 (or set at first run)
- [ ] Job 1 has MODEL_NAME and PROJECT_NAME
- [ ] Job 2 has MODEL_NAME and PROJECT_NAME
- [ ] Job 3 has ACCURACY_THRESHOLD and DEGRADATION_THRESHOLD

**No manual PERIOD passing needed** - the jobs handle it automatically via explicit environment_variables now!

---

## Summary

**Issue:** Jobs didn't pass PERIOD environment variable explicitly
**Root Cause:** Cloudera ML doesn't auto-inherit environment variables
**Fix:** Added explicit `environment_variables={"PERIOD": str(PERIOD)}` to all job triggers
**Result:** Clear, deterministic, debuggable pipeline orchestration
**Impact:** Module 2 pipeline now works reliably across multiple periods

---

*Fix implemented: November 2024*
*Files modified: 03.1_get_predictions.py, 03.2_load_ground_truth.py, 03.3_check_model.py*
