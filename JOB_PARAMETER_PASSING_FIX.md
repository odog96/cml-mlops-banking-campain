# Fix: Job Parameter Passing via Environment Variables

## Problem
Job 3.2 was failing with error: "Period parameter is required". Job 3.1 was unable to pass period parameters to Job 3.2 through CML job orchestration.

## Root Causes Identified

1. **Incorrect API Field Name**: Used `job_config` instead of `environment` in `CreateJobRunRequest`
   - `job_config` is not a valid parameter for `CreateJobRunRequest`
   - The correct parameter is `environment` (type: dict(str, str))

2. **Job Name Mismatch**: Searched for jobs by incomplete names
   - Searched for: "Load Ground Truth"
   - Actual job name: "Mod 2 Job 3: Load Ground Truth"
   - Fixed to use exact job names as created in the notebook

3. **Inefficient Batch Triggering**: Job 3.2 was being triggered for every batch
   - Moved job triggering from `process_batch()` to end of `main()`
   - Now triggers downstream job only once per Job 3.1 execution

## Changes Made

### 1. Fixed Job 3.1 (03.1_get_predictions.py)

**Changed from:**
```python
job_run_request = cmlapi.CreateJobRunRequest()
job_run_request.job_config = {
    "PERIOD_PARAM": period_param,
    "CURRENT_PERIOD": str(period),
    "TOTAL_PERIODS": str(total_periods)
}
```

**Changed to:**
```python
job_run_request = cmlapi.CreateJobRunRequest()
job_run_request.environment = {
    "PERIOD_PARAM": period_param,
    "CURRENT_PERIOD": str(period),
    "TOTAL_PERIODS": str(total_periods)
}
```

**Additional fixes:**
- Updated job name search from "Load Ground Truth" to "Mod 2 Job 3: Load Ground Truth"
- Moved job triggering from `process_batch()` (called for each batch) to `main()` (called once)
- Added debug output showing which environment variables are being set

### 2. Fixed Job 3.2 (03.2_load_ground_truth.py)

**Changed from:**
```python
job_run_request.job_config = {
    "PERIOD_PARAM": period_param,
    "CURRENT_PERIOD": str(period),
    "TOTAL_PERIODS": str(total_periods)
}
```

**Changed to:**
```python
job_run_request.environment = {
    "PERIOD_PARAM": period_param,
    "CURRENT_PERIOD": str(period),
    "TOTAL_PERIODS": str(total_periods)
}
```

**Additional fixes:**
- Updated job name search from "Check Model" to "Mod 2 Job 4: Check Model"
- Added debug output to show received environment variables

### 3. Added Debug Output to Job 3.3 (03.3_check_model.py)

Added debug output at startup to verify parameter reception:
```python
print("DEBUG: Checking for period parameters...")
print(f"DEBUG: PERIOD_PARAM={os.environ.get('PERIOD_PARAM')}")
print(f"DEBUG: CURRENT_PERIOD={os.environ.get('CURRENT_PERIOD')}")
print(f"DEBUG: TOTAL_PERIODS={os.environ.get('TOTAL_PERIODS')}")
```

## CML API Correct Usage

```python
import cmlapi

# Create job run request with environment variables
job_run_request = cmlapi.CreateJobRunRequest()
job_run_request.environment = {
    "VAR_NAME": "value",
    "ANOTHER_VAR": "another_value"
}

# Trigger the job
job_run = client.create_job_run(
    job_run_request,
    project_id=project_id,
    job_id=job_id
)
```

## Testing

Verified Job 3.2 receives environment variables correctly:
```bash
$ PERIOD_PARAM="0,19" CURRENT_PERIOD="0" TOTAL_PERIODS="19" python module2/03.2_load_ground_truth.py
DEBUG: PERIOD_PARAM=0,19
DEBUG: CURRENT_PERIOD=0
DEBUG: TOTAL_PERIOD=19
✓ Period parameters from environment variables: period 0 of 19
```

## Pipeline Flow (Corrected)

1. **Job 3.1 (Get Predictions)** processes period data
   - Sets environment variables: `PERIOD_PARAM`, `CURRENT_PERIOD`, `TOTAL_PERIODS`
   - Calls: `job_run_request.environment = {...}`
   - Triggers: Job 3.2 with these variables

2. **Job 3.2 (Load Ground Truth)** receives environment variables
   - Reads: `os.environ.get("PERIOD_PARAM")`
   - Parses period parameters
   - Sets same environment variables for Job 3.3
   - Triggers: Job 3.3 with these variables

3. **Job 3.3 (Check Model)** receives environment variables
   - Reads: `os.environ.get("PERIOD_PARAM")`
   - Validates model accuracy
   - Can trigger Job 3.1 for next period if needed

## Next Steps

1. Remove debug output once pipeline is confirmed working
2. Test full end-to-end pipeline via CML Jobs UI
3. Monitor for any remaining parameter passing issues

## References

- CML API: `CreateJobRunRequest` parameters
  - `environment`: dict(str, str) - Environment variables to pass to job run
  - `arguments`: str - Command-line arguments (not used for environment variables)
