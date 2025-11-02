# Parameter-Based Job Orchestration in Module 2

## Overview

This document describes the parameter-based state management system implemented across Module 2 Jobs 3.1, 3.2, and 3.3. This approach replaces the earlier (incorrect) environment variable passing method with a robust command-line argument approach that works reliably in CML environments.

## The Problem with Environment Variables

CML job orchestration does NOT inherit parent job environment variables. Each job run is a separate process with its own isolated environment. This means:

```python
# Job 3.1 tries to do this:
job_run_request.environment_variables = {"PERIOD": "0"}
client.create_job_run(job_run_request, ...)

# Job 3.2 receives:
PERIOD = int(os.environ.get("PERIOD", "0"))  # ← Will be "0" (default) NOT from parent!
```

Without explicit parameter passing, Job 3.2 cannot know which period it's processing - it will always default to period 0, causing the pipeline to loop infinitely.

## The Solution: Parameter Tuples via Command-Line Arguments

Instead of relying on environment variable inheritance, we pass parameters as command-line arguments:

```python
# Job 3.1 passes parameters like this:
job_run_request.arguments = ["--period", "0,19"]
client.create_job_run(job_run_request, ...)

# Job 3.2 receives and parses:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--period', type=str, required=True)
args = parser.parse_args()
parts = args.period.split(',')
PERIOD = int(parts[0])
TOTAL_PERIODS = int(parts[1])
```

## Parameter Format

**Tuple Format:** `(current_period, total_periods)`

**String Representation:** `"current_period,total_periods"`

**Examples:**
- `"0,19"` → Period 0 of 19 total periods
- `"5,19"` → Period 5 of 19 total periods
- `"19,19"` → Final period (exit condition)

## The Three-Job Pipeline Flow

### Job 3.1: Get Predictions

**Purpose:** Process predictions for a given period

**Parameter Behavior:**
- **No parameter provided:** Start at period 0, calculate total_periods from metadata
- **Parameter provided:** Use exact period and total_periods from parameter

**Code Implementation:**

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--period', type=str, default=None,
                    help='Period parameter as "current_period,total_periods"')
args = parser.parse_args()

if args.period is None:
    PERIOD = 0
    TOTAL_PERIODS = None  # Will be calculated from metadata
else:
    try:
        parts = args.period.split(',')
        PERIOD = int(parts[0])
        TOTAL_PERIODS = int(parts[1])
    except (ValueError, IndexError):
        print(f"ERROR: Invalid parameter format")
        sys.exit(1)

# Calculate TOTAL_PERIODS from metadata if not provided
if TOTAL_PERIODS is None:
    TOTAL_PERIODS = metadata['num_periods']

# Later, when triggering Job 3.2:
def trigger_load_ground_truth_job(client, proj_id, period, total_periods):
    ...
    job_run_request.arguments = ["--period", f"{period},{total_periods}"]
    client.create_job_run(job_run_request, ...)
```

**Next Step:** Calls Job 3.2 with same period

### Job 3.2: Load Ground Truth

**Purpose:** Load ground truth labels for the current period

**Parameter Behavior:**
- **Always requires** a `--period` parameter (from Job 3.1)
- **Never calculates** total_periods - must be provided by caller

**Code Implementation:**

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--period', type=str, required=True,
                    help='Period parameter as "current_period,total_periods"')
args = parser.parse_args()

try:
    parts = args.period.split(',')
    PERIOD = int(parts[0])
    TOTAL_PERIODS = int(parts[1])
except (ValueError, IndexError):
    print(f"ERROR: Invalid parameter format")
    sys.exit(1)

# Later, when triggering Job 3.3:
def trigger_next_job(period, total_periods):
    ...
    job_run_request.arguments = ["--period", f"{period},{total_periods}"]
    client.create_job_run(job_run_request, ...)
```

**Next Step:** Calls Job 3.3 with same period

### Job 3.3: Check Model

**Purpose:** Validate model accuracy and decide whether to continue to next period

**Parameter Behavior:**
- **Always requires** a `--period` parameter (from Job 3.2)
- **Checks exit condition:** if `PERIOD == TOTAL_PERIODS`, exit cleanly
- **Increments period** when continuing: `PERIOD+1` passed to Job 3.1

**Code Implementation:**

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--period', type=str, required=True,
                    help='Period parameter as "current_period,total_periods"')
args = parser.parse_args()

try:
    parts = args.period.split(',')
    PERIOD = int(parts[0])
    TOTAL_PERIODS = int(parts[1])
except (ValueError, IndexError):
    print(f"ERROR: Invalid parameter format")
    sys.exit(1)

# Check for exit condition
if PERIOD == TOTAL_PERIODS:
    print(f"All periods completed! (Period {PERIOD} = Total {TOTAL_PERIODS})")
    sys.exit(0)

# Later, when continuing to next period:
def trigger_next_period(client, proj_id, period, total_periods):
    ...
    next_period = period + 1
    job_run_request.arguments = ["--period", f"{next_period},{total_periods}"]
    client.create_job_run(job_run_request, ...)
```

**Next Step (if continuing):** Calls Job 3.1 with `PERIOD+1`

## Complete Period Flow Example

For a pipeline with 20 periods (0-19):

```
Job 3.1 (Period 0)
├─ Parses: no --period parameter
├─ Sets: PERIOD=0, TOTAL_PERIODS=20 (from metadata)
├─ Processes predictions for period 0
└─ Triggers Job 3.2 with: ["--period", "0,20"]

Job 3.2 (Period 0)
├─ Parses: "--period" = "0,20"
├─ Sets: PERIOD=0, TOTAL_PERIODS=20
├─ Loads ground truth labels for period 0
└─ Triggers Job 3.3 with: ["--period", "0,20"]

Job 3.3 (Period 0)
├─ Parses: "--period" = "0,20"
├─ Sets: PERIOD=0, TOTAL_PERIODS=20
├─ Checks: PERIOD (0) != TOTAL_PERIODS (20) → continue
├─ Validates model accuracy for period 0
├─ No degradation detected → continue to next period
└─ Triggers Job 3.1 with: ["--period", "1,20"]

Job 3.1 (Period 1)
├─ Parses: "--period" = "1,20"
├─ Sets: PERIOD=1, TOTAL_PERIODS=20
├─ Processes predictions for period 1
└─ Triggers Job 3.2 with: ["--period", "1,20"]

... (continues for periods 2, 3, 4, ... 18) ...

Job 3.1 (Period 19)
├─ Parses: "--period" = "19,20"
├─ Sets: PERIOD=19, TOTAL_PERIODS=20
├─ Processes predictions for period 19
└─ Triggers Job 3.2 with: ["--period", "19,20"]

Job 3.2 (Period 19)
├─ Parses: "--period" = "19,20"
├─ Sets: PERIOD=19, TOTAL_PERIODS=20
├─ Loads ground truth labels for period 19
└─ Triggers Job 3.3 with: ["--period", "19,20"]

Job 3.3 (Period 19)
├─ Parses: "--period" = "19,20"
├─ Sets: PERIOD=19, TOTAL_PERIODS=20
├─ Checks: PERIOD (19) != TOTAL_PERIODS (20) → continue
├─ Validates model accuracy for period 19
├─ No degradation detected → continue to next period
└─ Triggers Job 3.1 with: ["--period", "20,20"]

Job 3.1 (Period 20) - EXIT
├─ Parses: "--period" = "20,20"
├─ Sets: PERIOD=20, TOTAL_PERIODS=20
├─ Checks: PERIOD (20) == TOTAL_PERIODS (20) → EXIT
└─ Exits cleanly (sys.exit(0))
```

## Key Design Principles

### 1. **No Environment Variable Inheritance**
- Parameters are passed via command-line arguments, not environment variables
- Each job explicitly receives its period information from the calling job
- No implicit defaults or environmental assumptions

### 2. **Clear Initialization**
- Job 3.1 is the only job that can start without a parameter
- Job 3.1 calculates TOTAL_PERIODS from metadata when not provided
- All other jobs require explicit parameter passing

### 3. **Explicit State Propagation**
- Each job passes state explicitly to the next job
- No state is lost or corrupted during job transitions
- The parameter tuple `(current_period, total_periods)` is the contract between jobs

### 4. **Clean Exit Condition**
- The exit condition `PERIOD == TOTAL_PERIODS` is unambiguous
- No off-by-one errors or edge cases
- Clear, testable condition for pipeline completion

### 5. **Incrementation Point**
- Only Job 3.3 increments the period
- Jobs 3.1 and 3.2 pass the period unchanged to the next job
- This creates a natural progression: 0 → 1 → 2 → ... → n

## Testing the Implementation

### Manual Testing Steps

1. **Trigger Job 3.1 without parameter:**
   ```bash
   # In CML UI: Run Job "Get Predictions" without any parameters
   ```
   Expected log output:
   ```
   No parameter provided: defaulting to period 0
   Parameter provided: period 0 of 10
   ✓ Triggered job: Load Ground Truth (Period 0/10)
   ```

2. **Check Job 3.2 logs:**
   ```
   Parameter provided: period 0 of 10
   ✓ Extracted period 0 data
   ✓ Triggered next job: Check Model (Period 0/10)
   ```

3. **Check Job 3.3 logs:**
   ```
   Parameter provided: period 0 of 10
   ✓ Period 0 Metrics: Accuracy: 95.00%
   ✓ Triggered next period: Get Predictions (Period 1/10)
   ```

4. **Verify Period 1 is processed correctly:**
   ```
   Parameter provided: period 1 of 10
   ✓ Extracted period 1 data
   ```

### What Success Looks Like

- ✓ Each job prints "Parameter provided: period X of Y"
- ✓ PERIOD increments: 0 → 1 → 2 → ... → n
- ✓ Each period processes the correct data range
- ✓ Pipeline exits cleanly after final period
- ✓ No infinite loops on period 0

### What Indicates the Old Bug

- ✗ All jobs show "PERIOD = 0" (period not changing)
- ✗ Same job IDs appearing in logs repeatedly
- ✗ Data misalignment between jobs
- ✗ Infinite job execution without progress

## Migration Notes

If you have existing CML jobs using environment variables:

1. **Update job configuration:**
   - Remove PERIOD environment variable from job settings
   - Job 3.1 will calculate TOTAL_PERIODS from metadata

2. **Update job calls:**
   - From: `job_run_request.environment_variables = {"PERIOD": str(period)}`
   - To: `job_run_request.arguments = ["--period", f"{period},{total_periods}"]`

3. **Update argument parsing:**
   - Add `import argparse` to each job
   - Add argument parser with `--period` parameter
   - Parse period tuple from arguments

## Summary

This parameter-based implementation provides:
- ✓ Reliable job orchestration in CML
- ✓ Deterministic pipeline behavior
- ✓ Clear data flow between jobs
- ✓ Proper handling of pipeline completion
- ✓ Explicit, testable state management

The approach is robust, scalable, and provides a clear mental model for understanding how periods flow through the pipeline.
