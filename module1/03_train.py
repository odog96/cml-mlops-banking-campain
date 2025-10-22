"""
Module 1 - Step 3: Model Training with MLflow
==============================================

This script demonstrates:
1. Experiment tracking with MLflow
2. Training multiple model variants
3. Comparing models with/without engineered features
4. Hyperparameter tuning
5. Model registration

Key Learning Points:
- MLflow tracks ALL experiments for reproducibility
- Compare different approaches systematically
- Feature engineering impact is measurable
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils import MLFLOW_CONFIG, MODEL_CONFIG
from module1.utils import preprocess_for_training, split_data


def setup_mlflow():
    """
    Initialize MLflow experiment
    """
    # Set tracking URI (can be local or remote)
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    
    # Set experiment name
    experiment_name = MLFLOW_CONFIG["experiment_name"]
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {experiment_name}")
    print("-" * 60)


def load_data():
    """
    Load the engineered dataset
    """
    data_path = "data/bank_marketing_engineered.csv"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Engineered data not found at {data_path}\n"
            "Please run 02_eda_feature_engineering.ipynb first!"
        )
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded data: {df.shape}")
    return df


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate comprehensive metrics
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
    }
    
    return metrics


def train_model(X_train, X_test, y_train, y_test, params, run_name, include_engagement=True):
    """
    Train a single model and log everything to MLflow
    
    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        params: Model hyperparameters
        run_name: Name for this MLflow run
        include_engagement: Whether engagement_score was used
        
    Returns:
        Trained model and metrics
    """
    with mlflow.start_run(run_name=run_name):
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("include_engagement_score", include_engagement)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        
        # Train model
        print(f"\nTraining: {run_name}")
        print(f"  Parameters: {params}")
        
        model = XGBClassifier(**params, random_state=MODEL_CONFIG["random_state"])
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_pred_train, y_pred_proba_train)
        test_metrics = calculate_metrics(y_test, y_pred_test, y_pred_proba_test)
        
        # Log metrics
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)
        
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)
        
        # Feature importance
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        
        # Log top 10 features as metrics for easy comparison
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for idx, (feat, importance) in enumerate(top_features):
            mlflow.log_metric(f"top_{idx+1}_feature_importance", importance)
            mlflow.log_param(f"top_{idx+1}_feature_name", feat)
        
        # Log model
        signature = infer_signature(X_train, y_pred_test)
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        # Print results
        print(f"  Train Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
        print(f"  Test F1:        {test_metrics['f1']:.4f}")
        
        return model, test_metrics


def main():
    """
    Main training pipeline
    """
    print("=" * 60)
    print("Module 1 - Step 3: Model Training with MLflow")
    print("=" * 60)
    
    # Setup
    setup_mlflow()
    df = load_data()
    
    # Experiment 1: Baseline model WITHOUT engagement score
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Baseline (without engagement score)")
    print("=" * 60)
    
    X_baseline, y, _, _ = preprocess_for_training(df, include_engagement=False)
    X_train_base, X_test_base, y_train, y_test = split_data(
        X_baseline, y, 
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"]
    )
    
    baseline_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }
    
    model_baseline, metrics_baseline = train_model(
        X_train_base, X_test_base, y_train, y_test,
        baseline_params,
        "baseline_no_engagement",
        include_engagement=False
    )
    
    # Experiment 2: Model WITH engagement score
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: With Engineered Features")
    print("=" * 60)
    
    X_engineered, y, _, _ = preprocess_for_training(df, include_engagement=True)
    X_train_eng, X_test_eng, y_train, y_test = split_data(
        X_engineered, y,
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"]
    )
    
    engineered_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }
    
    model_engineered, metrics_engineered = train_model(
        X_train_eng, X_test_eng, y_train, y_test,
        engineered_params,
        "with_engagement_score",
        include_engagement=True
    )
    
    # Experiment 3: Hyperparameter tuning with engagement score
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Hyperparameter Variants")
    print("=" * 60)
    
    param_variants = [
        {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        {
            'n_estimators': 150,
            'max_depth': 7,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.7,
        },
        {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.15,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
        },
    ]
    
    best_model = None
    best_metrics = None
    best_auc = 0
    
    for idx, params in enumerate(param_variants):
        model, metrics = train_model(
            X_train_eng, X_test_eng, y_train, y_test,
            params,
            f"tuned_variant_{idx+1}",
            include_engagement=True
        )
        
        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_model = model
            best_metrics = metrics
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    print("\nBaseline Model (no engagement score):")
    print(f"  Test Accuracy: {metrics_baseline['accuracy']:.4f}")
    print(f"  Test ROC-AUC:  {metrics_baseline['roc_auc']:.4f}")
    print(f"  Test F1:       {metrics_baseline['f1']:.4f}")
    
    print("\nWith Engagement Score:")
    print(f"  Test Accuracy: {metrics_engineered['accuracy']:.4f}")
    print(f"  Test ROC-AUC:  {metrics_engineered['roc_auc']:.4f}")
    print(f"  Test F1:       {metrics_engineered['f1']:.4f}")
    
    print("\nBest Tuned Model:")
    print(f"  Test Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Test ROC-AUC:  {best_metrics['roc_auc']:.4f}")
    print(f"  Test F1:       {best_metrics['f1']:.4f}")
    
    # Calculate improvement
    auc_improvement = (best_metrics['roc_auc'] - metrics_baseline['roc_auc']) / metrics_baseline['roc_auc'] * 100
    
    print(f"\n📈 Improvement from baseline: {auc_improvement:+.2f}%")
    
    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print("\nView experiments in MLflow UI:")
    print("  Run: mlflow ui")
    print("  Then open: http://localhost:5000")
    print("\nNext step: 04_deploy_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
