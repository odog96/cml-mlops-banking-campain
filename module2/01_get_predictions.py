"""
Module 2 - Step 1: Get Predictions
==================================

This job processes predictions for the current period in batches,
tracking each prediction with Cloudera model monitoring metrics.

This follows the API pattern from example_get_predictions.py but adapted for:
- Batch processing by period (instead of streaming)
- Artificial ground truth instead of embedded ground truth loading
- Job orchestration (passes period number)
- Configurable batch size for inference

Workflow:
1. Load current period data (engineered features)
2. Process data in batches of specified size
3. For each batch:
   - Make predictions (using known_prediction from artificial data)
   - Track predictions with cml.track_delayed_metrics()
   - Trigger load_ground_truth_module2 job to process this batch's ground truth
4. load_ground_truth_module2 job loads the labels and triggers check_model_performance
5. Prediction and ground truth are processed in parallel for each batch

Input:
  - data/artificial_ground_truth_data.csv (full dataset)

Output:
  - Tracked metrics in CML model monitoring
  - data/predictions_period_{PERIOD}.json (predictions for this period)
  - Triggers: load_ground_truth_module2 job after each batch (via cmlapi)

Environment Variables:
  PERIOD: Current period number (default: 0)
  BATCH_SIZE: Samples per batch (default: 50)
  MODEL_NAME: Name of deployed model (default: "LSTM-2")
  PROJECT_NAME: Name of CML project (default: "SDWAN")
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

# Environment configuration
PERIOD = int(os.environ.get("PERIOD", "0"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "50"))
MODEL_NAME = os.environ.get("MODEL_NAME", "LSTM-2")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "SDWAN")

print("=" * 80)
print("Module 2 - Step 3: GET PREDICTIONS")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Processing Period: {PERIOD}")
print(f"Batch Size: {BATCH_SIZE}")
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

        print(f"\n✓ CML client initialized")
        print(f"  Model ID: {mod_id}")
        print(f"  Deployment CRN: {cr_number}")

        return client, proj_id, cr_number

    except Exception as e:
        print(f"⚠ Could not initialize CML client: {e}")
        print(f"  Continuing without CML tracking (metrics won't be recorded)")
        return None, None, None


def load_period_metadata(metadata_path="data/ground_truth_metadata.json"):
    """Load period boundaries from metadata."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\n✓ Loaded period metadata")
    print(f"  Total periods: {metadata['num_periods']}")

    return metadata


def load_full_dataset(data_path="data/artificial_ground_truth_data.csv"):
    """Load the full dataset to extract engineered features for prediction."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Ground truth data not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"✓ Loaded full dataset: {df.shape}")
    return df


def process_batch(batch_df, batch_num, cr_number, client, proj_id):
    """
    Process a batch of predictions with Cloudera tracking.

    In a real scenario, this would call the model API endpoint.
    For this demonstration, we use the known_prediction from the data
    to simulate the prediction (since we want to track accuracy over time).

    Args:
        batch_df: DataFrame with batch data
        batch_num: Batch number
        cr_number: Deployment CRN for tracking
        client: CML API client for job triggering
        proj_id: Project ID for job triggering
    """
    predictions_tracked = []

    print(f"\n  Batch {batch_num}: Processing {len(batch_df)} samples...")

    for idx, row in batch_df.iterrows():
        # In real scenario, would call model API here
        # For now, use known_prediction as the "model output"
        prediction = row['known_prediction']
        probability_1 = row['probability_class_1']

        # Create tracking record
        tracking_data = {
            "prediction": int(prediction),
            "probability_class_1": float(probability_1),
            "probability_class_0": float(1 - probability_1),
            "batch_num": int(batch_num),
            "sample_index": int(idx),
            "period": int(PERIOD),
        }

        # Track with Cloudera (if available)
        if cr_number:
            try:
                cml.track_delayed_metrics(
                    tracking_data,
                    unique_id=f"period_{PERIOD}_batch_{batch_num}_idx_{idx}"
                )
            except Exception as e:
                print(f"    ⚠ Could not track metrics for sample {idx}: {e}")

        predictions_tracked.append({
            "index": idx,
            "prediction": prediction,
            "probability": probability_1,
            "tracked": cr_number is not None
        })

    print(f"  ✓ Batch {batch_num} complete ({len(predictions_tracked)} tracked)")

    # Trigger load_ground_truth job after each batch
    trigger_load_ground_truth_job(client, proj_id)

    return predictions_tracked


def main():
    """Main workflow."""
    try:
        # ==================== SETUP ====================
        print("\nPHASE 1: Setup")
        print("-" * 80)

        client, proj_id, cr_number = setup_cml_client()

        # ==================== LOAD DATA ====================
        print("\nPHASE 2: Load Data")
        print("-" * 80)

        metadata = load_period_metadata()
        full_df = load_full_dataset()

        # Extract data for this period
        period_info = metadata["period_boundaries"][f"period_{PERIOD}"]
        start_idx = period_info['start_index']
        end_idx = period_info['end_index']
        period_df = full_df.iloc[start_idx:end_idx].copy()

        print(f"\n✓ Extracted period {PERIOD} data")
        print(f"  Rows: {start_idx} to {end_idx} ({len(period_df)} samples)")

        # ==================== PROCESS BATCHES ====================
        print("\nPHASE 3: Process Batches")
        print("-" * 80)

        all_predictions = []
        num_batches = (len(period_df) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"Processing {len(period_df)} samples in {num_batches} batches")
        print(f"  Batch size: {BATCH_SIZE}")

        for batch_num in range(num_batches):
            batch_start = batch_num * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(period_df))

            batch_df = period_df.iloc[batch_start:batch_end]
            batch_predictions = process_batch(batch_df, batch_num, cr_number, client, proj_id)

            all_predictions.extend(batch_predictions)

            # Small delay between batches to avoid overwhelming system
            if batch_num < num_batches - 1:
                time.sleep(0.1)

        # ==================== SAVE PREDICTIONS ====================
        print("\nPHASE 4: Save Prediction Results")
        print("-" * 80)

        predictions_df = pd.DataFrame(all_predictions)
        output_path = f"data/predictions_period_{PERIOD}.json"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "period": int(PERIOD),
                "num_samples": len(all_predictions),
                "num_batches": num_batches,
                "batch_size": BATCH_SIZE,
                "timestamp": datetime.now().isoformat(),
                "predictions": all_predictions
            }, f, indent=2)

        print(f"✓ Predictions saved to: {output_path}")

        # ==================== SUMMARY ====================
        print("\n" + "=" * 80)
        print("PREDICTIONS COMPLETE")
        print("=" * 80)
        print(f"\nPeriod {PERIOD} Prediction Summary:")
        print(f"  Total samples: {len(all_predictions)}")
        print(f"  Batches: {num_batches}")
        print(f"  Average batch size: {len(all_predictions) / num_batches:.1f}")
        if cr_number:
            print(f"  Metrics tracked: YES")
        else:
            print(f"  Metrics tracked: NO (CML not available)")
        print(f"\nNext step: 03_check_model.py will validate model accuracy")
        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n❌ Error in predictions: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def trigger_load_ground_truth_job(client, proj_id):
    """Trigger the load_ground_truth_module2 job with current period."""
    if not client or not proj_id:
        print("  ⚠ CML client not available, cannot trigger next job")
        return

    try:
        job_response = client.list_jobs(
            proj_id,
            search_filter=json.dumps({"name": "load_ground_truth_module2"})
        )

        if not job_response.jobs:
            print(f"  ⚠ Job 'load_ground_truth_module2' not found")
            return

        job_id = job_response.jobs[0].id

        # Create job run with PERIOD environment variable
        job_run_request = cmlapi.CreateJobRunRequest()
        job_run_request.environment_variables = {
            "PERIOD": str(PERIOD)
        }

        job_run = client.create_job_run(
            job_run_request,
            project_id=proj_id,
            job_id=job_id
        )

        print(f"  ✓ Triggered job: load_ground_truth_module2 (Period {PERIOD})")
        print(f"    Job run ID: {job_run.id}")

    except Exception as e:
        print(f"  ⚠ Could not trigger load_ground_truth_module2 job: {e}")


if __name__ == "__main__":
    main()
