"""
Module 2 - Step 4: Check Model
==============================

This job validates model accuracy for the current period and orchestrates
the monitoring pipeline. This follows the API pattern from example_check_model.py.

PARAMETER-BASED STATE MANAGEMENT:
This job uses a parameter tuple (current_period, total_periods) to track
pipeline state, NOT environment variables.

Workflow:
1. Parse period parameter from command-line arguments
2. Load period predictions from 03.1_get_predictions.py
3. Load period ground truth from 03.2_load_ground_truth.py
4. Calculate accuracy metrics
5. Check if accuracy has degraded (statistically significant)
6. Decision logic:
   - If accuracy degraded significantly → Flag alert and EXIT
   - If this is the last period (PERIOD == TOTAL_PERIODS) → EXIT pipeline
   - Otherwise → Trigger next period via 03.1_get_predictions.py with PERIOD+1

This implements the model monitoring with degradation detection that the user specified.

Parameter Format:
  - Format: (current_period, total_periods)
  - Example: (0, 19) means period 0 of 19 total periods
  - Passed from Job 3.2 via command-line arguments

Input:
  - data/current_period_ground_truth.json (from 03.2_load_ground_truth.py)
  - data/predictions_period_{PERIOD}.json (from 03.1_get_predictions.py)
  - data/ground_truth_metadata.json (configuration)
  - Parameter tuple (from Job 3.2)

Output:
  - data/check_model_results.json (accuracy report)
  - CML metrics (if degradation detected)
  - Job triggers (via cmlapi)

Environment Variables:
  ACCURACY_THRESHOLD: Minimum acceptable accuracy (default: 0.85)
  DEGRADATION_THRESHOLD: Minimum accuracy drop to flag as degraded (default: 0.05)
  MODEL_NAME: Name of deployed model
  PROJECT_NAME: Name of CML project
"""

import pandas as pd
import json
import os
import sys
from datetime import datetime
import numpy as np
import cmlapi
import cml
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuration
# ============================================================
ACCURACY_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.85"))
DEGRADATION_THRESHOLD = float(os.environ.get("DEGRADATION_THRESHOLD", "0.05"))
MODEL_NAME = os.environ.get("MODEL_NAME", "banking_campaign_predictor")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "CAI Baseline MLOPS")

# Parse command-line arguments for period parameter
# ============================================================
parser = argparse.ArgumentParser(description='Check model accuracy for a given period')
parser.add_argument('--period', type=str, default=None,
                    help='Period parameter as (current_period,total_periods), e.g., "0,19"')
# Use parse_known_args to handle Jupyter/IPython arguments that may be present
args, unknown = parser.parse_known_args()

# Extract PERIOD and TOTAL_PERIODS from parameter
# ============================================================
if args.period is None:
    print(f"ERROR: --period parameter is required")
    print(f"  Expected format: --period current_period,total_periods")
    print(f"  Example: --period 0,19")
    print(f"\nThis job should be triggered by Job 3.2 with the period parameter.")
    print(f"Manual testing: Pass --period 0,19 when running this job")
    sys.exit(1)

try:
    parts = args.period.split(',')
    PERIOD = int(parts[0])
    TOTAL_PERIODS = int(parts[1])
    print(f"Parameter provided: period {PERIOD} of {TOTAL_PERIODS}")
except (ValueError, IndexError):
    print(f"ERROR: Invalid parameter format. Expected 'current_period,total_periods'")
    print(f"  Example: --period 0,19")
    sys.exit(1)

# Check for exit condition: if PERIOD == TOTAL_PERIODS, we've completed all periods
# ============================================================
if PERIOD == TOTAL_PERIODS:
    print(f"All periods completed! (Period {PERIOD} = Total {TOTAL_PERIODS})")
    sys.exit(0)

print("=" * 80)
print("Module 2 - Step 4: CHECK MODEL")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Checking Period: {PERIOD}")
print(f"Accuracy Threshold: {ACCURACY_THRESHOLD*100:.1f}%")
print(f"Degradation Threshold: {DEGRADATION_THRESHOLD*100:.1f}%")
print(f"Model: {MODEL_NAME}")
print(f"Project: {PROJECT_NAME}")
print("-" * 80)


def setup_cml_client():
    """Initialize CML API client and get model deployment CRN."""
    try:
        client = cmlapi.default_client(
            url=os.getenv("CDSW_API_URL").replace("/api/v1", ""),
            cml_api_key=os.getenv("CDSW_APIV2_KEY")
        )

        target_model = client.list_all_models(
            search_filter=json.dumps({"name": MODEL_NAME})
        )

        if not target_model.models:
            raise ValueError(f"Model '{MODEL_NAME}' not found")

        proj_id = client.list_projects(
            search_filter=json.dumps({"name": PROJECT_NAME})
        ).projects[0].id

        mod_id = target_model.models[0].id

        build_list = client.list_model_builds(
            project_id=proj_id,
            model_id=mod_id,
            sort='created_at'
        )
        build_id = build_list.model_builds[-1].id

        deployments = client.list_model_deployments(
            project_id=proj_id,
            model_id=mod_id,
            build_id=build_id
        )

        cr_number = deployments.model_deployments[0].crn

        print(f"\n✓ CML client initialized")
        print(f"  Deployment CRN: {cr_number}")

        return client, proj_id, cr_number

    except Exception as e:
        print(f"⚠ Could not initialize CML client: {e}")
        return None, None, None


def load_metadata(metadata_path="data/ground_truth_metadata.json"):
    """Load configuration metadata."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\n✓ Loaded metadata")
    print(f"  Total periods: {metadata['num_periods']}")

    return metadata


def load_period_ground_truth(ground_truth_file="data/current_period_ground_truth.json"):
    """Load ground truth for current period."""
    if not os.path.exists(ground_truth_file):
        raise FileNotFoundError(f"Period ground truth not found at {ground_truth_file}")

    with open(ground_truth_file, 'r') as f:
        period_data = json.load(f)

    print(f"✓ Loaded period {period_data['period']} ground truth")
    print(f"  Samples: {period_data['num_samples']}")

    return period_data


def load_period_predictions(period, predictions_file=None):
    """Load predictions for current period."""
    if predictions_file is None:
        predictions_file = f"data/predictions_period_{period}.json"

    if not os.path.exists(predictions_file):
        raise FileNotFoundError(f"Predictions not found at {predictions_file}")

    with open(predictions_file, 'r') as f:
        predictions_data = json.load(f)

    print(f"✓ Loaded period {period} predictions")
    print(f"  Samples: {predictions_data['num_samples']}")

    return predictions_data


def load_previous_accuracy(results_file="data/check_model_results.json"):
    """Load previous period's accuracy for degradation comparison."""
    if not os.path.exists(results_file):
        print(f"  First period - no previous accuracy to compare")
        return None

    with open(results_file, 'r') as f:
        previous_results = json.load(f)

    if previous_results.get("period") == PERIOD - 1:
        previous_accuracy = previous_results.get("accuracy")
        print(f"  Previous period accuracy: {previous_accuracy*100:.2f}%")
        return previous_accuracy

    return None


def calculate_metrics(ground_truth, predictions):
    """
    Calculate comprehensive accuracy metrics.

    Args:
        ground_truth: List of true labels
        predictions: List of predicted labels

    Returns:
        Dictionary with metrics
    """
    ground_truth_array = np.array(ground_truth)
    predictions_array = np.array(predictions)

    metrics = {
        "accuracy": float(accuracy_score(ground_truth_array, predictions_array)),
        "precision": float(precision_score(ground_truth_array, predictions_array, zero_division=0)),
        "recall": float(recall_score(ground_truth_array, predictions_array, zero_division=0)),
        "f1": float(f1_score(ground_truth_array, predictions_array, zero_division=0)),
    }

    return metrics


def check_degradation(current_accuracy, previous_accuracy=None):
    """
    Check if model accuracy has degraded significantly.

    Args:
        current_accuracy: Current period accuracy
        previous_accuracy: Previous period accuracy (optional)

    Returns:
        Tuple of (is_degraded, reason)
    """
    # First period always passes
    if previous_accuracy is None:
        return False, "First period - no baseline for comparison"

    # Check if accuracy dropped more than threshold
    accuracy_drop = previous_accuracy - current_accuracy

    if accuracy_drop > DEGRADATION_THRESHOLD:
        return True, f"Accuracy dropped {accuracy_drop*100:.2f}% (threshold: {DEGRADATION_THRESHOLD*100:.1f}%)"

    if current_accuracy < ACCURACY_THRESHOLD:
        return True, f"Accuracy {current_accuracy*100:.2f}% below minimum threshold ({ACCURACY_THRESHOLD*100:.1f}%)"

    return False, f"Accuracy acceptable ({current_accuracy*100:.2f}%)"


def save_results(results):
    """Save check_model results to file."""
    output_path = "data/check_model_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


def track_metrics_to_cml(metrics, cr_number, period):
    """Track accuracy metrics to CML."""
    if not cr_number:
        return

    try:
        timestamp_ms = int(datetime.now().timestamp() * 1000)

        cml.track_aggregate_metrics(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "period": period
            },
            start_timestamp_ms=timestamp_ms,
            end_timestamp_ms=timestamp_ms,
            model_deployment_crn=cr_number,
        )

        print(f"✓ Metrics tracked to CML")

    except Exception as e:
        print(f"⚠ Could not track metrics to CML: {e}")


def trigger_next_period(client, proj_id, period, total_periods):
    """
    Trigger Get Predictions for next period.

    This is the ORCHESTRATION DECISION POINT for the entire pipeline.

    PARAMETER-BASED STATE PASSING:
    ==============================
    Passes period state as a parameter tuple: (current_period, total_periods)
    NOT via environment variables (which don't persist between job runs in CML).

    Key difference from trigger functions in Jobs 3.1 and 3.2:
    - Job 3.1 passes PERIOD (current) to Job 3.2
    - Job 3.2 passes PERIOD (current) to Job 3.3
    - Job 3.3 passes PERIOD+1 (next) to Job 3.1  ← THIS JOB, INCREMENTING PERIOD

    This is how the pipeline progresses through periods:
      Period 0 → Period 1 → Period 2 → ... → Last Period → EXIT

    Parameter Format:
    - Format: "current_period,total_periods"
    - Example: "0,19" (period 0 of 19 total periods)
    - Passed to next job via command-line arguments

    Without this increment, the pipeline would loop on the same period forever.
    """
    if not client or not proj_id:
        return

    try:
        next_period = period + 1

        # Search for Get Predictions job by name within the project
        job_response = client.list_jobs(
            proj_id,
            search_filter=json.dumps({"name": "Get Predictions"})
        )

        if not job_response.jobs:
            print(f"⚠ Job 'Get Predictions' not found in project")
            return

        job_id = job_response.jobs[0].id

        # Create job run request for NEXT period (PERIOD + 1)
        # CRITICAL: This is the only place where PERIOD increments!
        # Jobs 3.1 and 3.2 pass current PERIOD, but this job increments it.
        job_run_request = cmlapi.CreateJobRunRequest()
        # Pass period parameter as command-line argument (as a single string)
        period_param = f"{next_period},{total_periods}"
        job_run_request.arguments = f"--period {period_param}"

        job_run = client.create_job_run(
            job_run_request,
            project_id=proj_id,
            job_id=job_id
        )

        print(f"\n✓ Triggered next period: Get Predictions (Period {next_period}/{total_periods})")
        print(f"  Job run ID: {job_run.id}")

    except Exception as e:
        print(f"⚠ Could not trigger Get Predictions for next period: {e}")


def main():
    """Main workflow."""
    try:
        # ==================== SETUP ====================
        print("\nPHASE 1: Setup")
        print("-" * 80)

        client, proj_id, cr_number = setup_cml_client()
        metadata = load_metadata()

        # ==================== LOAD DATA ====================
        print("\nPHASE 2: Load Data")
        print("-" * 80)

        period_data = load_period_ground_truth()
        predictions_data = load_period_predictions(PERIOD)

        # Extract ground truth and predictions
        ground_truth = period_data['labels']
        predictions = [p['prediction'] for p in predictions_data['predictions']]

        # ==================== CALCULATE METRICS ====================
        print("\nPHASE 3: Calculate Metrics")
        print("-" * 80)

        metrics = calculate_metrics(ground_truth, predictions)

        print(f"✓ Period {PERIOD} Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall:    {metrics['recall']*100:.2f}%")
        print(f"  F1 Score:  {metrics['f1']*100:.2f}%")

        # ==================== CHECK DEGRADATION ====================
        print("\nPHASE 4: Check Degradation")
        print("-" * 80)

        previous_accuracy = load_previous_accuracy()
        is_degraded, degradation_reason = check_degradation(
            metrics['accuracy'],
            previous_accuracy
        )

        print(f"\nDegradation Check: {degradation_reason}")

        # ==================== DETERMINE NEXT ACTION ====================
        print("\nPHASE 5: Determine Next Action")
        print("-" * 80)

        is_last_period = PERIOD >= metadata['num_periods'] - 1

        if is_degraded:
            print(f"\n❌ MODEL DEGRADATION DETECTED!")
            print(f"   Reason: {degradation_reason}")
            print(f"   Action: EXITING PIPELINE")
            action = "EXIT_DEGRADATION"

        elif is_last_period:
            print(f"\n✓ Last period completed successfully")
            print(f"   Action: EXITING PIPELINE")
            action = "EXIT_LAST_PERIOD"

        else:
            print(f"\n✓ Period {PERIOD} completed successfully")
            print(f"   Next action: Continue to period {PERIOD + 1}")
            action = "CONTINUE_NEXT_PERIOD"

        # ==================== SAVE RESULTS ====================
        print("\nPHASE 6: Save Results")
        print("-" * 80)

        results = {
            "period": int(PERIOD),
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(ground_truth),
            "metrics": metrics,
            "previous_accuracy": float(previous_accuracy) if previous_accuracy else None,
            "accuracy_drop": float(previous_accuracy - metrics['accuracy']) if previous_accuracy else None,
            "is_degraded": bool(is_degraded),
            "degradation_reason": degradation_reason,
            "is_last_period": bool(is_last_period),
            "next_action": action,
        }

        save_results(results)

        # ==================== TRACK TO CML ====================
        print("\nPHASE 7: Track Metrics")
        print("-" * 80)

        track_metrics_to_cml(metrics, cr_number, PERIOD)

        # ==================== EXECUTE NEXT ACTION ====================
        print("\nPHASE 8: Execute Next Action")
        print("-" * 80)

        if action == "CONTINUE_NEXT_PERIOD":
            trigger_next_period(client, proj_id, PERIOD, TOTAL_PERIODS)

        elif action == "EXIT_DEGRADATION":
            print(f"\n⚠ Pipeline stopped due to model degradation")
            sys.exit(1)

        elif action == "EXIT_LAST_PERIOD":
            print(f"\n✓ All periods completed successfully!")

        # ==================== SUMMARY ====================
        print("\n" + "=" * 80)
        print("CHECK MODEL COMPLETE")
        print("=" * 80)
        print(f"\nSummary for Period {PERIOD}:")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Degraded: {'YES' if is_degraded else 'NO'}")
        print(f"  Last Period: {'YES' if is_last_period else 'NO'}")
        print(f"  Next Action: {action}")
        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n❌ Error in check_model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
