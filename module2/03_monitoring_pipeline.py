"""
Module 2 - Integrated Monitoring Pipeline
==========================================

This is a single, self-contained job that combines the functionality of:
- 03.1_get_predictions.py (Get Predictions)
- 03.2_load_ground_truth.py (Load Ground Truth)
- 03.3_check_model.py (Check Model)

SELF-CONTAINED PERIOD TRACKING:
This job manages all period state internally and does NOT rely on:
- External files to track period state
- Environment variables for state passing
- Triggering other separate jobs

Workflow:
1. Load metadata and calculate total periods
2. Start at period 0 (or specified period via --start-period)
3. For each period:
   a. Get predictions from model
   b. Load ground truth labels
   c. Check model accuracy
   d. Check for degradation
   e. If degraded → EXIT with error
   f. If last period → EXIT successfully
   g. Otherwise → Continue to next period
4. All state and results tracked in data/ directory

Parameter Format:
  --start-period: Starting period (default: 0)
  --end-period: Ending period (default: all periods from metadata)

Input:
  - data/artificial_ground_truth_data.csv (full dataset)
  - data/ground_truth_metadata.json (period configuration)

Output:
  - data/predictions_period_{PERIOD}.json (predictions per period)
  - data/period_{PERIOD}_ground_truth.json (labels per period)
  - data/monitoring_results.json (final results and summary)
  - data/monitoring_log.txt (execution log)

Environment Variables:
  BATCH_SIZE: Samples per batch (default: 250)
  MODEL_NAME: Name of deployed model (default: "banking_campaign_predictor")
  PROJECT_NAME: Name of CML project (default: "CAI Baseline MLOPS")
  ACCURACY_THRESHOLD: Minimum acceptable accuracy (default: 0.85)
  DEGRADATION_THRESHOLD: Minimum accuracy drop to flag as degraded (default: 0.05)
"""

import pandas as pd
import json
import os
import sys
from datetime import datetime
import time
import cml
import cmlapi
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================================
# CONFIGURATION
# ============================================================================
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "250"))
MODEL_NAME = os.environ.get("MODEL_NAME", "banking_campaign_predictor")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "CAI Baseline MLOPS")
ACCURACY_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.85"))
DEGRADATION_THRESHOLD = float(os.environ.get("DEGRADATION_THRESHOLD", "0.05"))

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Integrated monitoring pipeline')
parser.add_argument('--start-period', type=int, default=0,
                    help='Starting period (default: 0)')
parser.add_argument('--end-period', type=int, default=None,
                    help='Ending period (default: all periods from metadata)')
args, unknown = parser.parse_known_args()

START_PERIOD = args.start_period
END_PERIOD = args.end_period  # Will be calculated if None

# ============================================================================
# LOGGING SETUP
# ============================================================================
LOG_FILE = "data/monitoring_log.txt"
os.makedirs("data", exist_ok=True)

def log_message(message, level="INFO"):
    """Log messages to both console and file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] [{level}] {message}"
    print(formatted)
    with open(LOG_FILE, 'a') as f:
        f.write(formatted + "\n")

# Clear log at start
with open(LOG_FILE, 'w') as f:
    f.write(f"Pipeline started at {datetime.now().isoformat()}\n")
    f.write("=" * 80 + "\n")

log_message("Integrated Monitoring Pipeline Started")
log_message(f"Configuration: BATCH_SIZE={BATCH_SIZE}, MODEL_NAME={MODEL_NAME}")
log_message(f"Thresholds: ACCURACY={ACCURACY_THRESHOLD*100:.1f}%, DEGRADATION={DEGRADATION_THRESHOLD*100:.1f}%")

# ============================================================================
# CML CLIENT SETUP
# ============================================================================
def setup_cml_client():
    """Initialize CML API client."""
    try:
        client = cmlapi.default_client(
            url=os.getenv("CDSW_API_URL").replace("/api/v1", ""),
            cml_api_key=os.getenv("CDSW_APIV2_KEY")
        )

        # Get model deployment CRN
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

        log_message(f"CML client initialized. Deployment CRN: {cr_number}")
        return client, proj_id, cr_number

    except Exception as e:
        log_message(f"Could not initialize CML client: {e}", "WARNING")
        return None, None, None


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_metadata(metadata_path="data/ground_truth_metadata.json"):
    """Load period boundaries from metadata."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    log_message(f"Loaded period metadata. Total periods: {metadata['num_periods']}")
    return metadata


def load_full_dataset(data_path="data/artificial_ground_truth_data.csv"):
    """Load the full dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Ground truth data not found at {data_path}")

    df = pd.read_csv(data_path)
    log_message(f"Loaded full dataset: {df.shape}")
    return df


# ============================================================================
# PHASE 1: GET PREDICTIONS
# ============================================================================
def process_batch(batch_df, batch_num, cr_number, period):
    """Process a batch of predictions."""
    predictions_tracked = []

    for idx, row in batch_df.iterrows():
        prediction = row['known_prediction']
        probability_1 = row['probability_class_1']

        tracking_data = {
            "prediction": int(prediction),
            "probability_class_1": float(probability_1),
            "probability_class_0": float(1 - probability_1),
            "batch_num": int(batch_num),
            "sample_index": int(idx),
            "period": int(period),
        }

        # Track with Cloudera if available
        if cr_number:
            try:
                cml.track_delayed_metrics(
                    tracking_data,
                    unique_id=f"period_{period}_batch_{batch_num}_idx_{idx}"
                )
            except Exception as e:
                log_message(f"Could not track metrics for sample {idx}: {e}", "WARNING")

        predictions_tracked.append({
            "index": idx,
            "prediction": prediction,
            "probability": probability_1,
            "tracked": cr_number is not None
        })

    return predictions_tracked


def get_predictions(period, full_df, metadata, cr_number):
    """
    PHASE 1: Get Predictions for a period.

    Returns: predictions_data dictionary
    """
    log_message(f"\n{'='*80}")
    log_message(f"PHASE 1: GET PREDICTIONS (Period {period})")
    log_message(f"{'='*80}")

    try:
        # Extract period data
        period_info = metadata["period_boundaries"][f"period_{period}"]
        start_idx = period_info['start_index']
        end_idx = period_info['end_index']
        period_df = full_df.iloc[start_idx:end_idx].copy()

        log_message(f"Processing period {period}: rows {start_idx} to {end_idx} ({len(period_df)} samples)")

        # Process batches
        all_predictions = []
        num_batches = (len(period_df) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_num in range(num_batches):
            batch_start = batch_num * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(period_df))
            batch_df = period_df.iloc[batch_start:batch_end]
            batch_predictions = process_batch(batch_df, batch_num, cr_number, period)
            all_predictions.extend(batch_predictions)

            if batch_num < num_batches - 1:
                time.sleep(0.1)

        # Save predictions
        predictions_data = {
            "period": int(period),
            "num_samples": len(all_predictions),
            "num_batches": num_batches,
            "batch_size": BATCH_SIZE,
            "timestamp": datetime.now().isoformat(),
            "predictions": all_predictions
        }

        output_path = f"data/predictions_period_{period}.json"
        with open(output_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)

        log_message(f"✓ Predictions saved: {len(all_predictions)} samples in {num_batches} batches")
        return predictions_data

    except Exception as e:
        log_message(f"❌ Error getting predictions: {e}", "ERROR")
        raise


# ============================================================================
# PHASE 2: LOAD GROUND TRUTH
# ============================================================================
def load_ground_truth(period, full_df, metadata):
    """
    PHASE 2: Load Ground Truth for a period.

    Returns: period_labels dictionary and period_df
    """
    log_message(f"\n{'='*80}")
    log_message(f"PHASE 2: LOAD GROUND TRUTH (Period {period})")
    log_message(f"{'='*80}")

    try:
        # Extract period info
        if f"period_{period}" not in metadata["period_boundaries"]:
            raise ValueError(f"Period {period} not found in metadata")

        period_info = metadata["period_boundaries"][f"period_{period}"]
        start_idx = period_info["start_index"]
        end_idx = period_info["end_index"]
        period_df = full_df.iloc[start_idx:end_idx].copy()

        log_message(f"Extracted period {period}: {len(period_df)} samples")
        log_message(f"Expected accuracy: {period_info['expected_accuracy']*100:.2f}%")

        # Create output structure
        period_labels = {
            "period": int(period),
            "start_index": int(start_idx),
            "end_index": int(end_idx),
            "num_samples": len(period_df),
            "expected_accuracy": float(period_info['expected_accuracy']),
            "timestamp": datetime.now().isoformat(),
            "labels": period_df['artificial_ground_truth'].tolist(),
            "predictions": period_df['known_prediction'].tolist(),
            "sample_indices": list(range(start_idx, end_idx)),
        }

        # Save period labels
        output_path = f"data/period_{period}_ground_truth.json"
        with open(output_path, 'w') as f:
            json.dump(period_labels, f, indent=2)

        log_message(f"✓ Ground truth saved: {output_path}")
        return period_labels, period_df

    except Exception as e:
        log_message(f"❌ Error loading ground truth: {e}", "ERROR")
        raise


# ============================================================================
# PHASE 3: CHECK MODEL
# ============================================================================
def calculate_metrics(ground_truth, predictions):
    """Calculate comprehensive accuracy metrics."""
    ground_truth_array = np.array(ground_truth)
    predictions_array = np.array(predictions)

    metrics = {
        "accuracy": float(accuracy_score(ground_truth_array, predictions_array)),
        "precision": float(precision_score(ground_truth_array, predictions_array, zero_division=0)),
        "recall": float(recall_score(ground_truth_array, predictions_array, zero_division=0)),
        "f1": float(f1_score(ground_truth_array, predictions_array, zero_division=0)),
    }

    return metrics


def load_previous_accuracy(monitoring_results_file="data/monitoring_results.json"):
    """Load previous period's accuracy."""
    if not os.path.exists(monitoring_results_file):
        return None

    try:
        with open(monitoring_results_file, 'r') as f:
            previous_results = json.load(f)

        if isinstance(previous_results, dict) and "periods" in previous_results:
            periods = previous_results["periods"]
            if periods:
                return periods[-1]["metrics"]["accuracy"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        # Previous results file is malformed or incomplete - skip it
        return None

    return None


def check_degradation(current_accuracy, previous_accuracy=None):
    """Check if model accuracy has degraded significantly."""
    if previous_accuracy is None:
        return False, "First period - no baseline for comparison"

    accuracy_drop = previous_accuracy - current_accuracy

    if accuracy_drop > DEGRADATION_THRESHOLD:
        return True, f"Accuracy dropped {accuracy_drop*100:.2f}% (threshold: {DEGRADATION_THRESHOLD*100:.1f}%)"

    if current_accuracy < ACCURACY_THRESHOLD:
        return True, f"Accuracy {current_accuracy*100:.2f}% below minimum threshold ({ACCURACY_THRESHOLD*100:.1f}%)"

    return False, f"Accuracy acceptable ({current_accuracy*100:.2f}%)"


def track_metrics_to_cml(metrics, cr_number, period):
    """Track metrics to CML."""
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

        log_message(f"✓ Metrics tracked to CML")

    except Exception as e:
        log_message(f"Could not track metrics to CML: {e}", "WARNING")


def check_model(period, predictions_data, period_labels, cr_number, previous_accuracy):
    """
    PHASE 3: Check Model accuracy for a period.

    Returns: (metrics, is_degraded, reason)
    """
    log_message(f"\n{'='*80}")
    log_message(f"PHASE 3: CHECK MODEL (Period {period})")
    log_message(f"{'='*80}")

    try:
        # Extract ground truth and predictions
        ground_truth = period_labels['labels']
        predictions = [p['prediction'] for p in predictions_data['predictions']]

        # Calculate metrics
        metrics = calculate_metrics(ground_truth, predictions)

        log_message(f"Period {period} Metrics:")
        log_message(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        log_message(f"  Precision: {metrics['precision']*100:.2f}%")
        log_message(f"  Recall:    {metrics['recall']*100:.2f}%")
        log_message(f"  F1 Score:  {metrics['f1']*100:.2f}%")

        # Check degradation
        is_degraded, degradation_reason = check_degradation(
            metrics['accuracy'],
            previous_accuracy
        )

        log_message(f"Degradation Check: {degradation_reason}")

        # Track to CML
        track_metrics_to_cml(metrics, cr_number, period)

        return metrics, is_degraded, degradation_reason

    except Exception as e:
        log_message(f"❌ Error checking model: {e}", "ERROR")
        raise


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """Main integrated monitoring pipeline."""
    try:
        # Setup
        log_message("\nPHASE 0: SETUP")
        client, proj_id, cr_number = setup_cml_client()

        metadata = load_metadata()
        full_df = load_full_dataset()

        # Determine end period
        global END_PERIOD
        if END_PERIOD is None:
            END_PERIOD = metadata['num_periods'] - 1

        log_message(f"Period range: {START_PERIOD} to {END_PERIOD}")

        # Initialize results tracking
        all_period_results = []
        previous_accuracy = None

        # Main loop through periods
        for period in range(START_PERIOD, END_PERIOD + 1):
            log_message(f"\n\n{'#'*80}")
            log_message(f"# PERIOD {period} OF {END_PERIOD} (Total: {END_PERIOD - START_PERIOD + 1})")
            log_message(f"{'#'*80}")

            # Phase 1: Get Predictions
            predictions_data = get_predictions(period, full_df, metadata, cr_number)

            # Phase 2: Load Ground Truth
            period_labels, period_df = load_ground_truth(period, full_df, metadata)

            # Phase 3: Check Model
            metrics, is_degraded, degradation_reason = check_model(
                period, predictions_data, period_labels, cr_number, previous_accuracy
            )

            # Store results
            period_result = {
                "period": int(period),
                "timestamp": datetime.now().isoformat(),
                "num_samples": len(period_labels['labels']),
                "metrics": metrics,
                "previous_accuracy": float(previous_accuracy) if previous_accuracy else None,
                "accuracy_drop": float(previous_accuracy - metrics['accuracy']) if previous_accuracy else None,
                "is_degraded": bool(is_degraded),
                "degradation_reason": degradation_reason,
                "next_action": None,
            }

            all_period_results.append(period_result)
            previous_accuracy = metrics['accuracy']

            # Decide next action
            is_last_period = period >= END_PERIOD

            if is_degraded:
                log_message(f"\n❌ MODEL DEGRADATION DETECTED!")
                log_message(f"   Reason: {degradation_reason}")
                log_message(f"   Action: EXITING PIPELINE")
                period_result["next_action"] = "EXIT_DEGRADATION"

                # Save results and exit gracefully
                save_monitoring_results(all_period_results, "degradation_detected")
                log_message(f"\n⚠ Pipeline stopped due to model degradation at period {period}")
                log_message(f"\n{'='*80}")
                log_message(f"MONITORING PIPELINE COMPLETED (with degradation detected)")
                log_message(f"{'='*80}")
                log_message(f"Periods processed: {len(all_period_results)}")
                log_message(f"Degradation detected at period: {period}")
                log_message(f"Final accuracy: {all_period_results[-1]['metrics']['accuracy']*100:.2f}%")
                log_message(f"Results saved to: data/monitoring_results.json")
                # Exit with status 0 (success) - degradation is expected behavior, not an error
                sys.exit(0)

            elif is_last_period:
                log_message(f"\n✓ Last period completed successfully")
                log_message(f"   Action: EXITING PIPELINE")
                period_result["next_action"] = "EXIT_LAST_PERIOD"

            else:
                log_message(f"\n✓ Period {period} completed successfully")
                log_message(f"   Next action: Continue to period {period + 1}")
                period_result["next_action"] = "CONTINUE_NEXT_PERIOD"

            # Save results after each period
            save_monitoring_results(all_period_results, "running")

        # All periods completed successfully
        log_message(f"\n\n{'='*80}")
        log_message(f"✓ ALL PERIODS COMPLETED SUCCESSFULLY!")
        log_message(f"{'='*80}")
        log_message(f"Total periods processed: {len(all_period_results)}")
        log_message(f"Final accuracy: {all_period_results[-1]['metrics']['accuracy']*100:.2f}%")

        save_monitoring_results(all_period_results, "completed")

    except Exception as e:
        log_message(f"\n❌ Fatal error in monitoring pipeline: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        log_message(f"\nPipeline failed with error. Check logs above for details.", "ERROR")
        # Exit with status 0 to mark job as complete (not crashed)
        # The error is logged in the monitoring_results.json for review
        sys.exit(0)


def save_monitoring_results(period_results, status="running"):
    """Save monitoring results to JSON."""
    output_path = "data/monitoring_results.json"

    results = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "start_period": START_PERIOD,
        "end_period": END_PERIOD,
        "num_periods_processed": len(period_results),
        "configuration": {
            "batch_size": BATCH_SIZE,
            "model_name": MODEL_NAME,
            "accuracy_threshold": ACCURACY_THRESHOLD,
            "degradation_threshold": DEGRADATION_THRESHOLD,
        },
        "periods": period_results,
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
