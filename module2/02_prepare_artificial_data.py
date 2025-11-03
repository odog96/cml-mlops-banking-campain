"""
Module 2 - Step 1: Prepare Artificial Ground Truth Dataset
===========================================================

This STANDALONE script creates an artificial ground truth dataset for model monitoring.
It takes the engineered inference data from Module 1 and the predictions,
then creates a dataset with degrading accuracy over time periods.

This data will NOT be available at lab inception - you'll generate it manually
before starting the monitoring pipeline.

Workflow:
1. Load engineered inference data from module1
2. Load predictions from module1
3. Create artificial ground truth with intentional accuracy degradation
4. Split data into N periods with progressive label corruption
5. Save the complete dataset for use in monitoring pipeline

Output:
  - artificial_ground_truth_data.csv (engineered data + known predictions + artificial labels)
  - ground_truth_metadata.json (period boundaries and configuration)

This dataset is used by:
  - 01_load_ground_truth.py (loads labels for current period)
  - 02_get_predictions.py (makes predictions on batches)
  - 03_check_model.py (validates model performance)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import random

# Configuration
# ============================================================
# CRITICAL: BATCH_SIZE must be synchronized with Job 03.1
# If you change this value, you MUST also change it in:
#   - Job 03.1: "Get Predictions" environment variable
#   - Job 03.2: (inherited from Job 03.1)
#   - Job 03.3: (inherited from Job 03.1)
#
# Example: If BATCH_SIZE=100:
#   - 1,000 samples → 10 periods (1000 / 100)
#   - Each job runs 10 times (once per period)
#
# The num_periods calculation depends on this value:
#   num_periods = total_samples / BATCH_SIZE
# ============================================================
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "250"))


def load_engineered_data(data_path):
    """Load engineered inference data from Module 1."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Engineered data not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"✓ Loaded engineered data: {df.shape}")
    return df


def load_predictions(predictions_path):
    """Load predictions from Module 1."""
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions not found at {predictions_path}")

    df = pd.read_csv(predictions_path)
    print(f"✓ Loaded predictions: {df.shape}")
    return df


def create_ground_truth_with_degradation(predictions_df, num_periods=5,
                                         initial_accuracy=0.95,
                                         degradation_rate=0.05):
    """
    Create artificial ground truth labels with intentional accuracy degradation.

    Args:
        predictions_df: DataFrame with model predictions
        num_periods: Number of time periods to split data into
        initial_accuracy: Starting accuracy for first period (0.95 = 95% match to predictions)
        degradation_rate: How much accuracy decreases each period (0.05 = 5% per period)

    Returns:
        DataFrame with artificial_ground_truth column where accuracy degrades over time
    """
    df = predictions_df.copy()
    total_samples = len(df)
    samples_per_period = total_samples // num_periods

    artificial_labels = []
    period_assignments = []

    print(f"\nCreating artificial ground truth with {num_periods} periods...")
    print(f"  Initial accuracy: {initial_accuracy*100:.1f}%")
    print(f"  Degradation rate: {degradation_rate*100:.1f}% per period")
    print(f"  Samples per period: ~{samples_per_period}")

    for idx in range(total_samples):
        # Determine which period this sample belongs to
        period = min(idx // samples_per_period, num_periods - 1)
        period_assignments.append(period)

        # Calculate accuracy for this period
        current_accuracy = initial_accuracy - (period * degradation_rate)
        current_accuracy = max(0.5, min(current_accuracy, 1.0))  # Clamp between 0.5 and 1.0

        # Decide if we flip the label or keep it
        if random.random() < current_accuracy:
            # Keep the prediction as ground truth
            artificial_labels.append(df['prediction'].iloc[idx])
        else:
            # Flip the label to simulate degradation
            artificial_labels.append(1 - df['prediction'].iloc[idx])

    df['period'] = period_assignments
    df['artificial_ground_truth'] = artificial_labels
    df['known_prediction'] = df['prediction'].copy()

    # Print statistics by period
    print("\nAccuracy degradation by period:")
    for period in range(num_periods):
        period_mask = df['period'] == period
        period_data = df[period_mask]

        # Calculate how well artificial ground truth matches predictions
        matches = (period_data['artificial_ground_truth'] == period_data['known_prediction']).sum()
        accuracy = matches / len(period_data) if len(period_data) > 0 else 0

        print(f"  Period {period}: {accuracy*100:.2f}% match to predictions "
              f"({len(period_data)} samples)")

    return df


def create_metadata(df, num_periods, initial_accuracy, degradation_rate):
    """Create metadata file with period boundaries and configuration."""
    total_samples = len(df)
    samples_per_period = total_samples // num_periods

    metadata = {
        "creation_timestamp": datetime.now().isoformat(),
        "total_samples": int(total_samples),
        "num_periods": int(num_periods),
        "samples_per_period": int(samples_per_period),
        "initial_accuracy": float(initial_accuracy),
        "degradation_rate": float(degradation_rate),
        "period_boundaries": {}
    }

    for period in range(num_periods):
        start_idx = period * samples_per_period
        end_idx = start_idx + samples_per_period if period < num_periods - 1 else total_samples

        metadata["period_boundaries"][f"period_{period}"] = {
            "start_index": int(start_idx),
            "end_index": int(end_idx),
            "num_samples": int(end_idx - start_idx),
            "expected_accuracy": float(max(0.5, initial_accuracy - (period * degradation_rate)))
        }

    return metadata


def main():
    """Main workflow to prepare artificial ground truth data."""
    print("=" * 80)
    print("Module 2 - Step 1: PREPARE ARTIFICIAL GROUND TRUTH DATASET")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script creates an artificial dataset with degrading accuracy over time.")
    print("This simulates model degradation that we'll detect in the monitoring pipeline.")
    print("-" * 80)

    # Configuration (hardcoded at top of script)
    print(f"\nConfiguration:")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # ==================== LOAD DATA ====================
    print("\nPHASE 1: Load Source Data")
    print("-" * 80)

    engineered_data = load_engineered_data("module1/inference_data/engineered_inference_data.csv")
    predictions = load_predictions("module1/inference_data/predictions.csv")

    # Calculate periods from actual data size
    # ==================== IMPORTANT ====================
    # The number of periods is calculated from actual data size and BATCH_SIZE.
    # This determines how many times each job (03.1, 03.2, 03.3) will run.
    #
    # num_periods = total_samples / BATCH_SIZE
    #
    # If this changes (e.g., different dataset), you must verify that
    # Job 03.1's BATCH_SIZE environment variable is still correct!
    # ====================================================
    total_samples = len(engineered_data)
    num_periods = total_samples // BATCH_SIZE

    initial_accuracy = 0.95  # 95% match in period 0
    # Degradation rate: spread degradation evenly across all periods to reach ~50% by last period
    # This ensures: accuracy_period_N = initial_accuracy - (N * degradation_rate)
    degradation_rate = (initial_accuracy - 0.5) / num_periods

    print(f"\n✓ Calculated configuration from data:")
    print(f"  Total samples: {total_samples}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Number of periods: {num_periods} (= {total_samples} / {BATCH_SIZE})")
    print(f"  Initial accuracy: {initial_accuracy*100:.1f}%")
    print(f"  Degradation rate per period: {degradation_rate*100:.2f}%")
    print(f"  Final period accuracy: ~{max(0.5, initial_accuracy - (num_periods * degradation_rate))*100:.1f}%")

    # ==================== CREATE GROUND TRUTH ====================
    print("\nPHASE 2: Create Artificial Ground Truth with Degradation")
    print("-" * 80)

    # Combine data
    combined_df = pd.concat([engineered_data, predictions], axis=1)

    # Create ground truth with degradation
    final_df = create_ground_truth_with_degradation(
        combined_df,
        num_periods=num_periods,
        initial_accuracy=initial_accuracy,
        degradation_rate=degradation_rate
    )

    # ==================== CREATE METADATA ====================
    print("\nPHASE 3: Create Metadata")
    print("-" * 80)

    metadata = create_metadata(final_df, num_periods, initial_accuracy, degradation_rate)

    print(f"✓ Metadata created for {num_periods} periods")
    print(f"  Total samples: {metadata['total_samples']}")
    print(f"  Samples per period: {metadata['samples_per_period']}")

    # ==================== SAVE OUTPUTS ====================
    print("\nPHASE 4: Save Outputs")
    print("-" * 80)

    # Create data directory if needed
    os.makedirs("data", exist_ok=True)

    # Save artificial ground truth dataset
    output_path = "data/artificial_ground_truth_data.csv"
    final_df.to_csv(output_path, index=False)
    print(f"✓ Artificial ground truth dataset saved: {output_path}")
    print(f"  Shape: {final_df.shape}")
    print(f"  Columns: {list(final_df.columns)}")

    # Save metadata
    metadata_path = "data/ground_truth_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")

    # ==================== SUMMARY ====================
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(f"\nDataset structure:")
    print(f"  Original engineered features: {engineered_data.shape[1]} columns")
    print(f"  Prediction columns: {['prediction', 'probability_class_0', 'probability_class_1', 'prediction_label']}")
    print(f"  New columns added:")
    print(f"    - known_prediction: Original model predictions")
    print(f"    - artificial_ground_truth: Artificial labels (degrading accuracy)")
    print(f"    - period: Which period (0-{num_periods-1}) this sample belongs to")

    print(f"\nData usage in monitoring pipeline:")
    print(f"  • 01_load_ground_truth.py uses artificial_ground_truth for current period")
    print(f"  • 02_get_predictions.py processes known_prediction as model output")
    print(f"  • 03_check_model.py compares predictions vs artificial_ground_truth")

    print(f"\nColumn reference:")
    print(f"  Columns 0-{engineered_data.shape[1]-1}: Engineered features from Module 1")
    print(f"  Column {engineered_data.shape[1]}: prediction (model output)")
    print(f"  Columns {engineered_data.shape[1]+1}-{engineered_data.shape[1]+3}: probability scores")
    print(f"  Column {engineered_data.shape[1]+4}: prediction_label")
    print(f"  Column {engineered_data.shape[1]+5}: known_prediction")
    print(f"  Column {engineered_data.shape[1]+6}: artificial_ground_truth")
    print(f"  Column {engineered_data.shape[1]+7}: period")

    print("\n" + "=" * 80)
    print("✅ Artificial ground truth dataset ready!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review the generated data in data/artificial_ground_truth_data.csv")
    print("  2. Adjust configuration (num_periods, degradation_rate) as needed")
    print("  3. Then run the monitoring pipeline:")
    print("     - 01_load_ground_truth.py")
    print("     - 02_get_predictions.py")
    print("     - 03_check_model.py")


if __name__ == "__main__":
    main()
