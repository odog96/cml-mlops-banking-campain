"""
Module 1 - Step 5: Inference Data Preparation
==============================================

This is the FIRST JOB in the inference pipeline.

Takes raw inference data, applies the same feature engineering and preprocessing
transformations that were learned during model training, and outputs engineered
features ready for the prediction job.

Workflow:
1. Load raw inference data from inference_data/raw_inference_data.csv
2. Apply FeatureEngineer (creates engagement_score, age_groups, etc.)
3. Apply PreprocessingPipeline (scaling, one-hot encoding)
4. Save engineered/preprocessed data for the prediction job

Output: engineered_inference_data.csv (ready for model predictions)

Next step: 05.2_inference_predict.py
"""

import os
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime
import pickle
import tempfile

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.preprocessing import FeatureEngineer, PreprocessingPipeline


def load_raw_inference_data(data_path):
    """
    Load raw inference data from CSV.

    Args:
        data_path: Path to raw inference CSV file

    Returns:
        Pandas DataFrame with raw inference data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Raw inference data not found at {data_path}\n"
            "Please run 01_ingest.py first to create sample inference data."
        )

    df = pd.read_csv(data_path, sep=";")
    print(f"✓ Loaded raw inference data: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    return df


def apply_feature_engineering(df):
    """
    Apply feature engineering to raw inference data.

    Uses the same FeatureEngineer logic as training to ensure consistency.

    Args:
        df: Raw inference DataFrame

    Returns:
        Engineered DataFrame with new features
    """
    print("\nApplying feature engineering...")

    feature_engineer = FeatureEngineer()
    df_engineered = feature_engineer.transform(df)

    print(f"✓ Feature engineering complete")
    print(f"  Shape: {df_engineered.shape}")
    print(f"  New features: engagement_score, age_group, emp_var_category, duration_category")

    return df_engineered, feature_engineer


def extract_engineered_features(df_engineered, include_engagement=True):
    """
    Extract engineered features for model input, preserving row_id for tracking.

    Note: This does NOT apply preprocessing (scaling/encoding). That is done
    by the prediction job using the preprocessor that was fit during training.

    Args:
        df_engineered: DataFrame with engineered features
        include_engagement: Whether to include engagement_score

    Returns:
        DataFrame with selected features including row_id
    """
    print("\nExtracting engineered features...")

    # Define numeric and categorical features (same as training)
    numeric_features = [
        'age', 'duration', 'campaign', 'pdays', 'previous',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'
    ]

    categorical_features = [
        'job', 'marital', 'education', 'default',
        'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome',
        'age_group', 'emp_var_category', 'duration_category'
    ]

    if include_engagement:
        numeric_features.append('engagement_score')

    # Extract features and preserve row_id for tracking
    if 'row_id' in df_engineered.columns:
        X = df_engineered[['row_id'] + numeric_features + categorical_features].copy()
        print(f"✓ Features extracted (with row_id for tracking)")
    else:
        X = df_engineered[numeric_features + categorical_features].copy()
        print(f"✓ Features extracted")

    print(f"  Shape: {X.shape}")
    print(f"  Features: {X.shape[1]}")

    return X


def save_engineered_data(df_engineered, output_dir="inference_data"):
    """
    Save engineered (but not yet preprocessed) inference data to CSV.

    This data still needs scaling and encoding, which will be done by the
    prediction job using the preprocessor that was fit during training.

    Args:
        df_engineered: Engineered DataFrame (before preprocessing)
        output_dir: Output directory for engineered data

    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "engineered_inference_data.csv")
    df_engineered.to_csv(output_path, index=False)

    print(f"\n✓ Engineered features saved to: {output_path}")
    print(f"  Shape: {df_engineered.shape}")
    print(f"  Note: Data is engineered but not yet preprocessed (scaling/encoding)")
    print(f"        Preprocessing will be done by prediction job using trained preprocessor")

    return output_path


def save_feature_engineer_artifact(feature_engineer, output_dir="inference_data"):
    """
    Save feature_engineer artifact as a pickle file.

    This artifact will be loaded by the prediction job to apply the same
    feature engineering transformations consistently.

    Args:
        feature_engineer: FeatureEngineer instance
        output_dir: Output directory for artifacts
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save feature engineer
    feature_engineer_path = os.path.join(output_dir, "feature_engineer.pkl")
    with open(feature_engineer_path, 'wb') as f:
        pickle.dump(feature_engineer, f)
    print(f"✓ Feature engineer artifact saved: {feature_engineer_path}")


def main():
    """
    Main inference data preparation pipeline.
    """
    script_start = time.time()

    print("=" * 80)
    print("Module 1 - Step 5: INFERENCE DATA PREPARATION")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis job:")
    print("  1. Loads raw inference data")
    print("  2. Applies feature engineering")
    print("  3. Applies preprocessing (scaling, encoding)")
    print("  4. Saves engineered data for predictions")
    print("-" * 80)

    # ==================== DATA LOADING ====================
    print("\nPHASE 1: Load Raw Inference Data")
    print("-" * 80)

    data_load_start = time.time()
    df_raw = load_raw_inference_data("inference_data/raw_inference_data.csv")
    data_load_time = time.time() - data_load_start
    print(f"Data loading time: {data_load_time:.2f} seconds")

    # ==================== FEATURE ENGINEERING ====================
    print("\nPHASE 2: Feature Engineering")
    print("-" * 80)

    feature_eng_start = time.time()
    df_engineered, feature_engineer = apply_feature_engineering(df_raw)
    feature_eng_time = time.time() - feature_eng_start
    print(f"Feature engineering time: {feature_eng_time:.2f} seconds")

    # ==================== FEATURE EXTRACTION ====================
    print("\nPHASE 3: Extract Features for Model Input")
    print("-" * 80)

    extract_start = time.time()
    df_features = extract_engineered_features(df_engineered, include_engagement=True)
    extract_time = time.time() - extract_start
    print(f"Feature extraction time: {extract_time:.2f} seconds")

    # ==================== SAVE OUTPUTS ====================
    print("\nPHASE 4: Save Outputs")
    print("-" * 80)

    save_start = time.time()
    save_engineered_data(df_features)
    save_feature_engineer_artifact(feature_engineer)
    save_time = time.time() - save_start
    print(f"Save time: {save_time:.2f} seconds")

    # ==================== SUMMARY ====================
    script_elapsed = time.time() - script_start

    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTiming breakdown:")
    print(f"  Data loading:       {data_load_time:7.2f} seconds")
    print(f"  Feature engineering:  {feature_eng_time:7.2f} seconds")
    print(f"  Feature extraction:  {extract_time:7.2f} seconds")
    print(f"  Saving outputs:     {save_time:7.2f} seconds")
    print(f"  ─────────────────────────────")
    print(f"  Total execution:    {script_elapsed:7.2f} seconds ({script_elapsed/60:.2f} minutes)")

    print("\nOutputs created:")
    print("  • inference_data/engineered_inference_data.csv (raw engineered features, not yet scaled/encoded)")
    print("  • inference_data/feature_engineer.pkl (artifact for reproducibility)")

    print("\n" + "=" * 80)
    print("✅ Inference data preparation complete!")
    print("=" * 80)
    print("\n🚀 Next step: python 05.2_inference_predict.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
