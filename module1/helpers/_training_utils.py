"""
Shared training utilities for MLflow experiment management
"""

import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from shared_utils import MODEL_CONFIG

# MLflow experiment configuration
EXPERIMENT_NAME = "bank_marketing_experiments"


def setup_mlflow():
    """Initialize MLflow experiment - Cloudera AI handles tracking automatically"""
    # In Cloudera AI, just set the experiment name - no tracking URI needed
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"✓ Experiment: {EXPERIMENT_NAME}")
    print(f"✓ Tracking URI: {mlflow.get_tracking_uri()}")
    print("-" * 80)


def load_data():
    """Load the raw dataset for preprocessing and feature engineering"""
    data_path = "/home/cdsw/module1/data/bank-additional/bank-additional-full.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Raw data not found at {data_path}\n"
            "Expected raw banking data file not available!"
        )

    df = pd.read_csv(data_path, sep=";")
    print(f"✓ Loaded raw data: {df.shape}")
    return df


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
    }
    return metrics


def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance"""
    smote = SMOTE(random_state=MODEL_CONFIG["random_state"])
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"  Original class distribution: {np.bincount(y_train)}")
    print(f"  SMOTE class distribution:    {np.bincount(y_train_resampled)}")

    return X_train_resampled, y_train_resampled


def train_model(X_train, X_test, y_train, y_test, model_type, params,
                run_name, use_smote=False, include_engagement=True,
                preprocessor=None, feature_engineer=None):
    """
    Train a single model and log everything to MLflow

    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        model_type: 'logistic', 'random_forest', or 'xgboost'
        params: Model hyperparameters
        run_name: Name for this MLflow run
        use_smote: Whether to apply SMOTE for class balancing
        include_engagement: Whether engagement_score was used
        preprocessor: PreprocessingPipeline instance (to be logged with model)
        feature_engineer: FeatureEngineer instance (to be logged with model)

    Returns:
        Trained model and metrics
    """
    with mlflow.start_run(run_name=run_name):

        # Apply SMOTE if requested
        if use_smote:
            X_train_model, y_train_model = apply_smote(X_train, y_train)
        else:
            X_train_model, y_train_model = X_train, y_train

        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("use_smote", use_smote)
        mlflow.log_param("include_engagement_score", include_engagement)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_train_samples_after_smote", len(X_train_model))
        mlflow.log_param("n_test_samples", len(X_test))

        # Tag for easy filtering in MLflow UI
        mlflow.set_tag("model_family", model_type)
        mlflow.set_tag("smote", "yes" if use_smote else "no")
        mlflow.set_tag("features", "engineered" if include_engagement else "baseline")

        # Train model based on type
        print(f"\nTraining: {run_name}")
        print(f"  Model: {model_type} | SMOTE: {use_smote}")
        print(f"  Parameters: {params}")

        if model_type == 'logistic':
            model = LogisticRegression(**params, random_state=MODEL_CONFIG["random_state"])
        elif model_type == 'random_forest':
            model = RandomForestClassifier(**params, random_state=MODEL_CONFIG["random_state"])
        elif model_type == 'xgboost':
            model = XGBClassifier(**params, random_state=MODEL_CONFIG["random_state"])
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train_model, y_train_model)

        # Predictions
        y_pred_train = model.predict(X_train_model)
        y_pred_test = model.predict(X_test)
        y_pred_proba_train = model.predict_proba(X_train_model)[:, 1]
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        train_metrics = calculate_metrics(y_train_model, y_pred_train, y_pred_proba_train)
        test_metrics = calculate_metrics(y_test, y_pred_test, y_pred_proba_test)

        # Log metrics
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)

        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))

            # Log top 10 features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for idx, (feat, importance) in enumerate(top_features):
                mlflow.log_metric(f"top_{idx+1}_feature_importance", importance)
                mlflow.log_param(f"top_{idx+1}_feature_name", feat)

        # Log model with preprocessing artifacts
        signature = infer_signature(X_train, y_pred_test)

        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )

        # Log preprocessing artifacts separately using pickle
        import pickle
        import tempfile
        if preprocessor is not None:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
                pickle.dump(preprocessor, f)
                mlflow.log_artifact(f.name, artifact_path='preprocessing')
        if feature_engineer is not None:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
                pickle.dump(feature_engineer, f)
                mlflow.log_artifact(f.name, artifact_path='preprocessing')

        # Print results
        print(f"  Train Accuracy: {train_metrics['accuracy']:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test Precision: {test_metrics['precision']:.4f} | Test Recall: {test_metrics['recall']:.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f} | Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

        return model, test_metrics


def save_results(results_baseline, results_engineered, total_elapsed,
                 phase1_elapsed, phase2_elapsed):
    """Save experiment results to CSV and timing summary to text file"""
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    results_baseline.to_csv(f"{output_dir}/baseline_results.csv", index=False)
    results_engineered.to_csv(f"{output_dir}/engineered_results.csv", index=False)

    # Save timing summary to a text file
    timing_summary_path = f"{output_dir}/execution_timing.txt"
    total_experiments = len(results_baseline) + len(results_engineered)

    with open(timing_summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Execution Timing Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Experiments: {total_experiments}\n")
        f.write(f"Total Time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)\n")
        f.write(f"Average per Experiment: {total_elapsed/total_experiments:.2f} seconds\n")
        f.write(f"Phase 1 (Baseline): {phase1_elapsed:.2f} seconds\n")
        f.write(f"Phase 2 (Engineered): {phase2_elapsed:.2f} seconds\n")

    print(f"\n💾 Results saved to {output_dir}/")
    print(f"   • baseline_results.csv")
    print(f"   • engineered_results.csv")
    print(f"   • execution_timing.txt")


def print_summary(results_baseline, results_engineered):
    """Print comprehensive experiment summary"""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY & PERFORMANCE METRICS")
    print("=" * 80)

    # Find best models
    best_baseline = results_baseline.loc[results_baseline['roc_auc'].idxmax()]
    best_engineered = results_engineered.loc[results_engineered['roc_auc'].idxmax()]

    print("\n📊 BEST BASELINE MODEL:")
    print(f"  Model: {best_baseline['model_type']} (Config {best_baseline['config']})")
    print(f"  SMOTE: {best_baseline['smote']}")
    print(f"  Accuracy: {best_baseline['accuracy']:.4f}")
    print(f"  Precision: {best_baseline['precision']:.4f}")
    print(f"  Recall: {best_baseline['recall']:.4f}")
    print(f"  F1-Score: {best_baseline['f1']:.4f}")
    print(f"  ROC-AUC: {best_baseline['roc_auc']:.4f}")

    print("\n📊 BEST ENGINEERED MODEL:")
    print(f"  Model: {best_engineered['model_type']} (Config {best_engineered['config']})")
    print(f"  SMOTE: {best_engineered['smote']}")
    print(f"  Accuracy: {best_engineered['accuracy']:.4f}")
    print(f"  Precision: {best_engineered['precision']:.4f}")
    print(f"  Recall: {best_engineered['recall']:.4f}")
    print(f"  F1-Score: {best_engineered['f1']:.4f}")
    print(f"  ROC-AUC: {best_engineered['roc_auc']:.4f}")

    # Calculate improvement
    auc_improvement = (best_engineered['roc_auc'] - best_baseline['roc_auc']) / best_baseline['roc_auc'] * 100
    precision_improvement = (best_engineered['precision'] - best_baseline['precision']) / best_baseline['precision'] * 100

    print(f"\n📈 IMPROVEMENT WITH FEATURE ENGINEERING:")
    print(f"  ROC-AUC: {auc_improvement:+.2f}%")
    print(f"  Precision: {precision_improvement:+.2f}%")

    # SMOTE impact analysis
    print("\n📊 SMOTE IMPACT ANALYSIS:")

    for feature_type, results in [('Baseline', results_baseline), ('Engineered', results_engineered)]:
        print(f"\n  {feature_type} Features:")

        # Group by model type and SMOTE
        smote_comparison = results.groupby(['model_type', 'smote'])[['precision', 'recall', 'f1']].mean()

        for model_type in results['model_type'].unique():
            if (model_type, False) in smote_comparison.index and (model_type, True) in smote_comparison.index:
                no_smote = smote_comparison.loc[(model_type, False)]
                with_smote = smote_comparison.loc[(model_type, True)]

                precision_change = (with_smote['precision'] - no_smote['precision']) / no_smote['precision'] * 100
                recall_change = (with_smote['recall'] - no_smote['recall']) / no_smote['recall'] * 100

                print(f"    {model_type}:")
                print(f"      Precision: {precision_change:+.2f}% | Recall: {recall_change:+.2f}%")
