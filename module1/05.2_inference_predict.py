"""
Module 1 - Step 6: Inference Predictions
=========================================

This is the SECOND JOB in the inference pipeline.

Takes pre-engineered inference data (output from 05_inference_data_prep.py),
makes batch predictions against the deployed model endpoint, and outputs results.

Workflow:
1. Load pre-engineered inference data and preprocessing artifacts
2. Apply preprocessing (scaling & encoding) using fitted preprocessing pipeline
3. Make batch predictions via REST API calls to deployed model endpoint
4. Save predictions to CSV

Input: engineered_inference_data.csv (from 05_inference_data_prep.py)
Output: predictions.csv with prediction results

Previous step: 05_inference_data_prep.py
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import requests
import time
from datetime import datetime
import pickle

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_engineered_inference_data(data_path):
    """
    Load pre-engineered inference data from CSV.

    Args:
        data_path: Path to engineered inference data

    Returns:
        Pandas DataFrame with engineered features
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Engineered inference data not found at {data_path}\n"
            "Please run 05_inference_data_prep.py first."
        )

    df = pd.read_csv(data_path)
    print(f"✓ Loaded pre-engineered inference data: {df.shape}")
    print(f"  Features: {df.shape[1]}")
    return df


def load_preprocessing_pipeline(data_path="inference_data/feature_engineer.pkl"):
    """
    Load the fitted preprocessing pipeline (FeatureEngineer) from pickle.

    Args:
        data_path: Path to pickled FeatureEngineer

    Returns:
        FeatureEngineer instance (already fitted)
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Feature engineer pickle not found at {data_path}\n"
            "Please run 05_inference_data_prep.py first."
        )

    with open(data_path, 'rb') as f:
        feature_engineer = pickle.load(f)

    print(f"✓ Loaded preprocessing pipeline")
    return feature_engineer




def get_model_endpoint_config():
    """
    Get the deployed model endpoint configuration.

    Returns:
        Tuple of (endpoint_url, access_key)
    """
    MODEL_ENDPOINT = "https://modelservice.ml-dbfc64d1-783.go01-dem.ylcu-atmi.cloudera.site/model"
    ACCESS_KEY = "mbtbh46x9h7wxj4cdkxz9fxl0nzmrefv"

    print(f"✓ Model endpoint: {MODEL_ENDPOINT}")
    return MODEL_ENDPOINT, ACCESS_KEY


def preprocess_for_api(X_engineered, feature_engineer):
    """
    Apply preprocessing (scaling & encoding) to engineered features.

    NOTE: The deployed model is trained with engineered features including engagement_score.

    Args:
        X_engineered: DataFrame with engineered features
        feature_engineer: Loaded FeatureEngineer instance

    Returns:
        DataFrame with preprocessed features ready for API
    """
    from helpers.preprocessing import PreprocessingPipeline

    # Define features (must match DEPLOYED model training setup - WITH engagement_score)
    numeric_features = [
        'age', 'duration', 'campaign', 'pdays', 'previous',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
        'engagement_score'
    ]

    categorical_features = [
        'job', 'marital', 'education', 'default',
        'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome',
        'age_group', 'emp_var_category', 'duration_category'
    ]

    # Load full training data to fit preprocessor on all categorical values
    df_train = pd.read_csv("data/bank-additional/bank-additional-full.csv", sep=";")
    df_train_eng = feature_engineer.transform(df_train)

    # Create and fit preprocessor WITH engagement_score (matches deployed engineered model)
    preprocessor = PreprocessingPipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        include_engagement=True
    )

    X_train_full = df_train_eng[numeric_features + categorical_features].copy()
    preprocessor.fit(X_train_full)

    # Transform inference data
    X_subset = X_engineered[numeric_features + categorical_features].copy()
    X_processed = pd.DataFrame(
        preprocessor.transform(X_subset),
        columns=preprocessor.get_feature_names()
    )

    return X_processed, preprocessor


def make_batch_api_predictions(X_processed, batch_size=100,
                              model_endpoint=None, access_key=None,
                              timeout=30):
    """
    Make batch predictions via REST API calls to the deployed model endpoint.

    Args:
        X_processed: DataFrame with preprocessed features
        batch_size: Number of rows per API call
        model_endpoint: URL of the model endpoint
        access_key: Access key for authentication
        timeout: Request timeout in seconds

    Returns:
        Tuple of (predictions, probabilities) as numpy arrays
    """
    print(f"\nMaking batch API predictions on {len(X_processed)} samples...")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {(len(X_processed) - 1) // batch_size + 1}")

    all_predictions = []
    all_probabilities = []

    # Process in batches
    for batch_start in range(0, len(X_processed), batch_size):
        batch_end = min(batch_start + batch_size, len(X_processed))
        batch_data = X_processed.iloc[batch_start:batch_end]

        # Prepare API payload for this batch
        payload = {
            "accessKey": access_key,
            "request": {
                "dataframe_split": {
                    "columns": list(batch_data.columns),
                    "data": batch_data.values.tolist()
                }
            }
        }

        payload_json = json.dumps(payload)

        try:
            # Send request to model endpoint
            response = requests.post(
                model_endpoint,
                data=payload_json,
                headers={'Content-Type': 'application/json'},
                timeout=timeout
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"API request failed with status {response.status_code}: {response.text}"
                )

            # Parse response
            result = response.json()
            batch_predictions = result['response']['prediction']

            all_predictions.extend(batch_predictions)

            # For now, store predictions (probabilities not returned by API)
            # We'll create synthetic probabilities based on predictions
            for pred in batch_predictions:
                if pred == 0:
                    all_probabilities.append([1.0, 0.0])
                else:
                    all_probabilities.append([0.0, 1.0])

            print(f"  ✓ Batch {batch_start // batch_size + 1}: "
                  f"{len(batch_predictions)} predictions received")

        except Exception as e:
            raise RuntimeError(
                f"Error making predictions for batch {batch_start}-{batch_end}: {e}"
            )

    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)

    print(f"✓ Batch predictions complete")
    print(f"  Total predictions: {len(predictions)}")

    return predictions, probabilities


def create_results_dataframe(predictions, probabilities, row_ids=None):
    """
    Create results DataFrame with predictions and probability scores.

    Args:
        predictions: Binary predictions (0 or 1)
        probabilities: Probability scores for both classes
        row_ids: Optional row IDs for tracking records through the pipeline

    Returns:
        DataFrame with results
    """
    results_data = {
        'prediction': predictions,
        'probability_class_0': probabilities[:, 0],
        'probability_class_1': probabilities[:, 1],
        'prediction_label': ['no' if p == 0 else 'yes' for p in predictions]
    }

    # Add row_id at the beginning if provided
    if row_ids is not None:
        results_data = {'row_id': row_ids, **results_data}

    results_df = pd.DataFrame(results_data)

    print(f"\n✓ Results DataFrame created: {results_df.shape}")
    print(f"\nPrediction Summary:")
    print(f"  Class 0 (no): {(predictions == 0).sum()} samples ({(predictions == 0).sum()/len(predictions)*100:.1f}%)")
    print(f"  Class 1 (yes): {(predictions == 1).sum()} samples ({(predictions == 1).sum()/len(predictions)*100:.1f}%)")
    print(f"\nProbability Statistics:")
    print(f"  Mean probability (class 1): {probabilities[:, 1].mean():.4f}")
    print(f"  Std deviation (class 1): {probabilities[:, 1].std():.4f}")
    print(f"  Min (class 1): {probabilities[:, 1].min():.4f}")
    print(f"  Max (class 1): {probabilities[:, 1].max():.4f}")

    return results_df


def save_predictions(results_df, output_path="inference_data/predictions.csv"):
    """
    Save prediction results to CSV.

    Args:
        results_df: DataFrame with predictions
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")


def create_prediction_summary_report(results_df, output_dir="inference_data"):
    """
    Create a text summary report of prediction results.

    Args:
        results_df: DataFrame with predictions
        output_dir: Output directory for report
    """
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "prediction_summary.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PREDICTION RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total samples predicted: {len(results_df)}\n\n")

        f.write("PREDICTION DISTRIBUTION:\n")
        f.write(f"  Class 0 (no):  {(results_df['prediction'] == 0).sum():6d} ({(results_df['prediction'] == 0).sum()/len(results_df)*100:5.1f}%)\n")
        f.write(f"  Class 1 (yes): {(results_df['prediction'] == 1).sum():6d} ({(results_df['prediction'] == 1).sum()/len(results_df)*100:5.1f}%)\n")

        f.write("\nPROBABILITY STATISTICS (Class 1):\n")
        f.write(f"  Mean:          {results_df['probability_class_1'].mean():.6f}\n")
        f.write(f"  Median:        {results_df['probability_class_1'].median():.6f}\n")
        f.write(f"  Std Deviation: {results_df['probability_class_1'].std():.6f}\n")
        f.write(f"  Min:           {results_df['probability_class_1'].min():.6f}\n")
        f.write(f"  25th Percentile: {results_df['probability_class_1'].quantile(0.25):.6f}\n")
        f.write(f"  Median (50th):   {results_df['probability_class_1'].quantile(0.50):.6f}\n")
        f.write(f"  75th Percentile: {results_df['probability_class_1'].quantile(0.75):.6f}\n")
        f.write(f"  Max:           {results_df['probability_class_1'].max():.6f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Sample predictions (first 10 rows):\n")
        f.write("=" * 80 + "\n")
        f.write(results_df.head(10).to_string(index=False))

    print(f"✓ Prediction summary report saved: {report_path}")


def main():
    """
    Main inference prediction pipeline.
    """
    script_start = time.time()

    print("=" * 80)
    print("Module 1 - Step 6: INFERENCE PREDICTIONS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis job:")
    print("  1. Loads pre-engineered inference data")
    print("  2. Applies preprocessing (scaling & encoding)")
    print("  3. Makes batch API predictions to deployed model")
    print("  4. Saves results to CSV")
    print("-" * 80)

    # ==================== DATA LOADING ====================
    print("\nPHASE 1: Load Pre-engineered Inference Data")
    print("-" * 80)

    data_load_start = time.time()
    df_engineered = load_engineered_inference_data("inference_data/engineered_inference_data.csv")
    data_load_time = time.time() - data_load_start
    print(f"Data loading time: {data_load_time:.2f} seconds")

    # ==================== LOAD PREPROCESSING ====================
    print("\nPHASE 1.5: Load Preprocessing Pipeline")
    print("-" * 80)

    pipeline_load_start = time.time()
    feature_engineer = load_preprocessing_pipeline()
    pipeline_load_time = time.time() - pipeline_load_start
    print(f"Pipeline loading time: {pipeline_load_time:.2f} seconds")

    # ==================== DATA PREPROCESSING ====================
    print("\nPHASE 2: Apply Preprocessing (Scaling & Encoding)")
    print("-" * 80)

    prep_start = time.time()
    X_processed, preprocessor = preprocess_for_api(df_engineered, feature_engineer)
    prep_time = time.time() - prep_start
    print(f"✓ Data preprocessed and ready for API")
    print(f"  Shape: {X_processed.shape}")
    print(f"Preprocessing time: {prep_time:.2f} seconds")

    # ==================== GET MODEL ENDPOINT ====================
    print("\nPHASE 3: Initialize Model Endpoint")
    print("-" * 80)

    endpoint_start = time.time()
    model_endpoint, access_key = get_model_endpoint_config()
    endpoint_time = time.time() - endpoint_start

    # ==================== BATCH PREDICTIONS ====================
    print("\nPHASE 4: Make Batch API Predictions")
    print("-" * 80)

    prediction_start = time.time()
    predictions, probabilities = make_batch_api_predictions(
        X_processed,
        batch_size=100,
        model_endpoint=model_endpoint,
        access_key=access_key,
        timeout=30
    )
    prediction_time = time.time() - prediction_start
    print(f"Prediction time: {prediction_time:.2f} seconds")

    # ==================== RESULTS ====================
    print("\nPHASE 5: Prepare Results")
    print("-" * 80)

    results_start = time.time()
    results_df = create_results_dataframe(predictions, probabilities)
    results_time = time.time() - results_start
    print(f"Results preparation time: {results_time:.2f} seconds")

    # ==================== SAVE OUTPUTS ====================
    print("\nPHASE 6: Save Outputs")
    print("-" * 80)

    save_start = time.time()
    save_predictions(results_df)
    create_prediction_summary_report(results_df)
    save_time = time.time() - save_start
    print(f"Save time: {save_time:.2f} seconds")

    # ==================== SUMMARY ====================
    script_elapsed = time.time() - script_start

    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTiming breakdown:")
    print(f"  Data loading:      {data_load_time:7.2f} seconds")
    print(f"  Pipeline loading:  {pipeline_load_time:7.2f} seconds")
    print(f"  Data preparation:  {prep_time:7.2f} seconds")
    print(f"  Predictions:       {prediction_time:7.2f} seconds")
    print(f"  Results prep:      {results_time:7.2f} seconds")
    print(f"  Saving outputs:    {save_time:7.2f} seconds")
    print(f"  ─────────────────────────────")
    print(f"  Total execution:   {script_elapsed:7.2f} seconds ({script_elapsed/60:.2f} minutes)")

    print("\nOutputs created:")
    print("  • inference_data/predictions.csv")
    print("  • inference_data/prediction_summary.txt")

    print("\n" + "=" * 80)
    print("✅ Inference predictions complete!")
    print("=" * 80)
    print("\nInference pipeline workflow:")
    print("  1. 01_ingest.py - Creates sample inference data")
    print("  2. 05_1_inference_data_prep.py - Engineers and preprocesses data")
    print("  3. 05_2_inference_predict.py - Makes batch API predictions ← You are here")
    print("=" * 80)


if __name__ == "__main__":
    main()
