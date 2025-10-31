"""
Module 1 - Step 6: Inference Predictions
=========================================

This is the SECOND JOB in the inference pipeline.

Takes engineered inference data (output from 05_inference_data_prep.py),
loads a trained model from MLflow, makes predictions, and outputs results.

Workflow:
1. Load engineered inference data and preprocessing artifacts
2. Load best trained model from MLflow
3. Make predictions with probability scores
4. Save predictions to CSV

Input: engineered_inference_data.csv (from 05_inference_data_prep.py)
Output: predictions.csv with prediction results

Previous step: 05_inference_data_prep.py
"""

import os
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime
import pickle
import mlflow
import mlflow.sklearn

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_raw_inference_data(data_path):
    """
    Load raw inference data from CSV.

    Args:
        data_path: Path to raw inference data

    Returns:
        Pandas DataFrame with raw features
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Raw inference data not found at {data_path}\n"
            "Please run 01_ingest.py first to create sample inference data."
        )

    df = pd.read_csv(data_path, sep=";")
    print(f"✓ Loaded raw inference data: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    return df


def load_engineered_inference_data(data_path):
    """
    Load engineered inference data from CSV.

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
    print(f"✓ Loaded engineered inference data: {df.shape}")
    print(f"  Features: {df.shape[1]}")
    return df




def load_model_from_mlflow(experiment_name="bank_marketing_experiments", metric="test_roc_auc"):
    """
    Load the best trained model from MLflow.

    Retrieves the best model based on specified metric.

    Args:
        experiment_name: Name of MLflow experiment
        metric: Metric to use for selecting best model (default: test_roc_auc)

    Returns:
        Tuple of (model, preprocessor, feature_engineer) loaded from MLflow
    """
    print(f"\nLoading model from MLflow experiment: {experiment_name}")

    try:
        model, preprocessor_mlflow, feature_engineer_mlflow = load_best_model(
            experiment_name, metric=metric
        )
        print(f"✓ Loaded best model from MLflow")
        print(f"  Model type: {type(model).__name__}")
        return model, preprocessor_mlflow, feature_engineer_mlflow
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from MLflow: {e}\n"
            "Make sure you have trained models in MLflow using 03_train_quick.py or 03_train_extended.py"
        )


def make_predictions(X_processed, model):
    """
    Make predictions and get probability scores.

    Args:
        X_processed: Preprocessed features (should already be in model-ready format)
        model: Trained sklearn model

    Returns:
        Tuple of (predictions, probabilities)
    """
    print(f"\nMaking predictions on {len(X_processed)} samples...")

    # Convert to numpy array to avoid feature name validation issues
    # This can happen when inference data has different categorical levels than training
    X_array = X_processed.values if hasattr(X_processed, 'values') else X_processed

    # Get predictions
    predictions = model.predict(X_array)

    # Get probability scores
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_array)
    else:
        raise ValueError(f"Model {type(model).__name__} does not support probability predictions")

    print(f"✓ Predictions complete")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Probabilities shape: {probabilities.shape}")

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
    print("  1. Loads engineered inference data")
    print("  2. Loads best trained model from MLflow")
    print("  3. Makes predictions with probability scores")
    print("  4. Saves results to CSV")
    print("-" * 80)

    # ==================== DATA LOADING ====================
    print("\nPHASE 1: Load Raw Inference Data")
    print("-" * 80)

    data_load_start = time.time()

    # Load raw inference data (not engineered - we'll engineer it here with proper preprocessing)
    df_raw = load_raw_inference_data("inference_data/raw_inference_data.csv")

    data_load_time = time.time() - data_load_start
    print(f"Data loading time: {data_load_time:.2f} seconds")

    # ==================== MODEL LOADING ====================
    print("\nPHASE 2: Load Trained Model from MLflow")
    print("-" * 80)

    model_load_start = time.time()
    model, preprocessor_trained, feature_engineer_trained = load_model_from_mlflow()
    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.2f} seconds")

    # ==================== DATA PREPROCESSING ====================
    print("\nPHASE 2.5: Feature Engineering & Preprocessing (Scaling & Encoding)")
    print("-" * 80)

    prep_start = time.time()

    # Import preprocessing classes
    from helpers.preprocessing import PreprocessingPipeline, FeatureEngineer

    # Step 1: Extract row_id if present (for tracking through pipeline)
    row_ids = None
    if 'row_id' in df_raw.columns:
        row_ids = df_raw['row_id'].values
        print(f"✓ Extracted row_ids for tracking ({len(row_ids)} records)")

    # Step 2: Apply feature engineering to raw inference data
    fe = FeatureEngineer()
    df_engineered = fe.transform(df_raw)
    print(f"✓ Feature engineering applied to inference data")

    # Step 3: Load full training data and apply feature engineering for preprocessor fitting
    df_train = pd.read_csv("data/bank-additional/bank-additional-full.csv", sep=";")
    df_train_eng = fe.transform(df_train)

    # Define features
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

    # Step 3: Create and fit preprocessor on full training data
    # This ensures all categorical values are present for proper encoding
    preprocessor = PreprocessingPipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        include_engagement=True
    )

    X_train_full = df_train_eng[numeric_features + categorical_features].copy()
    preprocessor.fit(X_train_full)

    # Step 4: Transform engineered inference data using fitted preprocessor
    X_engineered_subset = df_engineered[numeric_features + categorical_features].copy()
    X_processed = pd.DataFrame(
        preprocessor.transform(X_engineered_subset),
        columns=preprocessor.get_feature_names()
    )

    print(f"✓ Data preprocessed (scaling and one-hot encoding)")
    print(f"  Preprocessor fit on full training data to capture all categorical values")
    prep_time = time.time() - prep_start
    print(f"  Shape: {X_processed.shape}")
    print(f"Preprocessing time: {prep_time:.2f} seconds")

    # ==================== PREDICTIONS ====================
    print("\nPHASE 3: Make Predictions")
    print("-" * 80)

    prediction_start = time.time()
    predictions, probabilities = make_predictions(X_processed, model)
    prediction_time = time.time() - prediction_start
    print(f"Prediction time: {prediction_time:.2f} seconds")

    # ==================== RESULTS ====================
    print("\nPHASE 4: Prepare Results")
    print("-" * 80)

    results_start = time.time()
    results_df = create_results_dataframe(predictions, probabilities, row_ids=row_ids)
    results_time = time.time() - results_start
    print(f"Results preparation time: {results_time:.2f} seconds")

    # ==================== SAVE OUTPUTS ====================
    print("\nPHASE 5: Save Outputs")
    print("-" * 80)

    save_start = time.time()
    save_predictions(results_df)
    create_prediction_summary_report(results_df)
    save_time = time.time() - save_start
    print(f"Save time: {save_time:.2f} seconds")

    total_phases_time = data_load_time + model_load_time + prep_time + prediction_time + results_time + save_time

    # ==================== SUMMARY ====================
    script_elapsed = time.time() - script_start

    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTiming breakdown:")
    print(f"  Data loading:      {data_load_time:7.2f} seconds")
    print(f"  Model loading:     {model_load_time:7.2f} seconds")
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
    print("\nPrediction pipeline workflow:")
    print("  1. 01_ingest.py - Creates sample inference data")
    print("  2. 05_inference_data_prep.py - Prepares/engineers data ← You are here")
    print("  3. 06_inference_predict.py - Makes predictions")
    print("=" * 80)


if __name__ == "__main__":
    main()
