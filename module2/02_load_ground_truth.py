"""
Module 2 - Step 2: Load Ground Truth
====================================

This job loads the ground truth labels for the current period.
It is called by 01_get_predictions.py after predictions are made.

This script:
1. Loads ground truth metadata (period boundaries)
2. Reads artificial ground truth dataset
3. Extracts labels for the current period
4. Saves period-specific ground truth for use by check_model job
5. Triggers the next job (check_model) via cmlapi

Input:
  - data/ground_truth_metadata.json (period configuration)
  - data/artificial_ground_truth_data.csv (full dataset with labels)
  - PERIOD environment variable (current period number, default: 0)

Output:
  - data/current_period_ground_truth.json (labels for current period)
  - Triggers: 03_check_model.py (via cmlapi)

Environment Variables:
  PERIOD: Current period number (0, 1, 2, etc.) - inherited from previous job
  MODEL_NAME: Name of deployed model (default: "LSTM-2")
  PROJECT_NAME: Name of CML project (default: "SDWAN")
  CDSW_API_URL: CML API endpoint (auto-set by CML)
  CDSW_APIV2_KEY: CML API key (auto-set by CML)
"""

import pandas as pd
import json
import os
import sys
from datetime import datetime
import cmlapi

# Environment configuration
PERIOD = int(os.environ.get("PERIOD", "0"))
MODEL_NAME = os.environ.get("MODEL_NAME", "LSTM-2")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "SDWAN")

print("=" * 80)
print("Module 2 - Step 2: LOAD GROUND TRUTH")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Processing Period: {PERIOD}")
print(f"Model: {MODEL_NAME}")
print(f"Project: {PROJECT_NAME}")
print("-" * 80)


def load_metadata(metadata_path="data/ground_truth_metadata.json"):
    """Load period boundaries and configuration from metadata."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"✓ Loaded metadata")
    print(f"  Total periods: {metadata['num_periods']}")
    print(f"  Samples per period: {metadata['samples_per_period']}")
    return metadata


def load_ground_truth_data(data_path="data/artificial_ground_truth_data.csv"):
    """Load the full artificial ground truth dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Ground truth data not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"✓ Loaded ground truth data: {df.shape}")
    return df


def extract_period_labels(ground_truth_df, metadata, period):
    """
    Extract ground truth labels for a specific period.

    Args:
        ground_truth_df: DataFrame with all ground truth data
        metadata: Configuration with period boundaries
        period: Current period number

    Returns:
        Dictionary with period labels and metadata
    """
    if str(f"period_{period}") not in metadata["period_boundaries"]:
        raise ValueError(f"Period {period} not found in metadata. "
                         f"Valid periods: 0-{metadata['num_periods']-1}")

    period_info = metadata["period_boundaries"][f"period_{period}"]
    start_idx = period_info["start_index"]
    end_idx = period_info["end_index"]

    # Extract data for this period
    period_df = ground_truth_df.iloc[start_idx:end_idx].copy()

    print(f"\n✓ Extracted period {period} data")
    print(f"  Rows: {start_idx} to {end_idx} ({len(period_df)} samples)")
    print(f"  Expected accuracy: {period_info['expected_accuracy']*100:.2f}%")

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

    return period_labels, period_df


def save_period_labels(period_labels, output_path="data/current_period_ground_truth.json"):
    """Save ground truth labels for the current period."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(period_labels, f, indent=2)

    print(f"\n✓ Saved period labels to: {output_path}")


def trigger_next_job():
    """
    Trigger the next job in the pipeline (03_check_model.py).

    This assumes the job is configured in CML with name 'check_model_performance'.
    The job will use the current PERIOD environment variable.
    """
    try:
        # Initialize CML API client
        client = cmlapi.default_client(
            url=os.getenv("CDSW_API_URL").replace("/api/v1", ""),
            cml_api_key=os.getenv("CDSW_APIV2_KEY")
        )

        # Get project ID
        proj_response = client.list_projects(
            search_filter=json.dumps({"name": PROJECT_NAME})
        )
        if not proj_response.projects:
            print(f"⚠ Project '{PROJECT_NAME}' not found, skipping job trigger")
            return

        proj_id = proj_response.projects[0].id

        # Find check_model_performance job
        job_response = client.list_jobs(
            proj_id,
            search_filter=json.dumps({"name": "check_model_performance"})
        )

        if not job_response.jobs:
            print(f"⚠ Job 'check_model_performance' not found, skipping trigger")
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

        print(f"\n✓ Triggered next job: check_model_performance")
        print(f"  Job run ID: {job_run.id}")
        print(f"  Period: {PERIOD}")

    except Exception as e:
        print(f"\n⚠ Could not trigger next job: {e}")
        print(f"  This is expected if CML job is not configured")


def main():
    """Main workflow."""
    try:
        # ==================== LOAD DATA ====================
        print("\nPHASE 1: Load Configuration and Data")
        print("-" * 80)

        metadata = load_metadata()
        ground_truth_data = load_ground_truth_data()

        # ==================== EXTRACT PERIOD LABELS ====================
        print("\nPHASE 2: Extract Period Labels")
        print("-" * 80)

        period_labels, period_df = extract_period_labels(
            ground_truth_data,
            metadata,
            PERIOD
        )

        # ==================== SAVE OUTPUTS ====================
        print("\nPHASE 3: Save Outputs")
        print("-" * 80)

        save_period_labels(period_labels)

        # ==================== TRIGGER NEXT JOB ====================
        print("\nPHASE 4: Trigger Next Job")
        print("-" * 80)

        trigger_next_job()

        # ==================== SUMMARY ====================
        print("\n" + "=" * 80)
        print("GROUND TRUTH LOADING COMPLETE")
        print("=" * 80)
        print(f"\nPeriod {PERIOD} Ground Truth Summary:")
        print(f"  Samples: {period_labels['num_samples']}")
        print(f"  Expected accuracy: {period_labels['expected_accuracy']*100:.2f}%")
        print(f"  Labels saved to: data/current_period_ground_truth.json")
        print(f"\nNext step: 02_get_predictions.py will process these labels")
        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n❌ Error in ground truth loading: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
