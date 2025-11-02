"""
Module 1 - Step 3: Extended Model Training with MLflow (Full Version)
====================================================================

This script demonstrates comprehensive Cloudera AI MLflow capabilities:
1. Experiment tracking across multiple model types
2. Hyperparameter comparison across 3 models with 4 configs each
3. Feature engineering impact (baseline vs engineered)
4. Class imbalance handling (with/without SMOTE)
5. Comprehensive metrics logging

Configuration:
- 3 models (Logistic Regression, Random Forest, XGBoost)
- 4 configs per model
- 2 SMOTE variants (with/without)
- 2 feature sets (baseline and engineered)
- Total: 3 × 4 × 2 × 2 = 48 experiment runs
- Expected runtime: 10-20 minutes (depending on system)

This is the perfect version for:
- Comprehensive model comparison
- Production-grade experimentation
- Understanding model tradeoffs
- Finding optimal hyperparameters

For a quick test run, use: python 03_train_quick.py
"""

import os
import sys
import pandas as pd
import time
from datetime import datetime

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils import MODEL_CONFIG
from helpers.preprocessing import preprocess_for_training, split_data
from helpers._training_utils import (
    setup_mlflow, load_data, train_model, save_results, print_summary
)


def run_extended_experiments(X_train, X_test, y_train, y_test,
                             experiment_name, include_engagement=True,
                             preprocessor=None, feature_engineer=None):
    """
    Run a comprehensive suite of experiments with 3 models and 4 configs each.

    Args:
        X_train, X_test, y_train, y_test: Train/test data splits
        experiment_name: Name of the experiment phase (baseline/engineered)
        include_engagement: Whether engagement score was used in features
        preprocessor: PreprocessingPipeline instance to log with models
        feature_engineer: FeatureEngineer instance to log with models

    Returns:
        DataFrame with all experiment results
    """
    results = []

    print("\n" + "=" * 80)
    print(f"EXTENDED EXPERIMENTS: {experiment_name.upper()}")
    print("=" * 80)

    # ==================== LOGISTIC REGRESSION ====================
    print("\n" + "-" * 80)
    print("LOGISTIC REGRESSION EXPERIMENTS (4 CONFIGS)")
    print("-" * 80)

    lr_configs = [
        {'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs'},
        {'C': 0.1, 'max_iter': 1000, 'solver': 'lbfgs'},
        {'C': 10.0, 'max_iter': 1000, 'solver': 'lbfgs'},
        {'C': 1.0, 'max_iter': 1000, 'solver': 'saga', 'penalty': 'l1'},
    ]

    for idx, params in enumerate(lr_configs, 1):
        # Without SMOTE
        run_name = f"LR_{idx}_{experiment_name}_no_smote"
        model, metrics = train_model(
            X_train, X_test, y_train, y_test,
            'logistic', params, run_name,
            use_smote=False, include_engagement=include_engagement,
            preprocessor=preprocessor, feature_engineer=feature_engineer
        )
        results.append({'model_type': 'Logistic Regression', 'config': idx,
                       'smote': False, **metrics})

        # With SMOTE
        run_name = f"LR_{idx}_{experiment_name}_with_smote"
        model, metrics = train_model(
            X_train, X_test, y_train, y_test,
            'logistic', params, run_name,
            use_smote=True, include_engagement=include_engagement,
            preprocessor=preprocessor, feature_engineer=feature_engineer
        )
        results.append({'model_type': 'Logistic Regression', 'config': idx,
                       'smote': True, **metrics})

    # ==================== RANDOM FOREST ====================
    print("\n" + "-" * 80)
    print("RANDOM FOREST EXPERIMENTS (4 CONFIGS)")
    print("-" * 80)

    rf_configs = [
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5},
        {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 10},
        {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
    ]

    for idx, params in enumerate(rf_configs, 1):
        # Without SMOTE
        run_name = f"RF_{idx}_{experiment_name}_no_smote"
        model, metrics = train_model(
            X_train, X_test, y_train, y_test,
            'random_forest', params, run_name,
            use_smote=False, include_engagement=include_engagement,
            preprocessor=preprocessor, feature_engineer=feature_engineer
        )
        results.append({'model_type': 'Random Forest', 'config': idx,
                       'smote': False, **metrics})

        # With SMOTE
        run_name = f"RF_{idx}_{experiment_name}_with_smote"
        model, metrics = train_model(
            X_train, X_test, y_train, y_test,
            'random_forest', params, run_name,
            use_smote=True, include_engagement=include_engagement,
            preprocessor=preprocessor, feature_engineer=feature_engineer
        )
        results.append({'model_type': 'Random Forest', 'config': idx,
                       'smote': True, **metrics})

    # ==================== XGBOOST ====================
    print("\n" + "-" * 80)
    print("XGBOOST EXPERIMENTS (4 CONFIGS)")
    print("-" * 80)

    xgb_configs = [
        {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1,
         'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
         'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 150, 'max_depth': 7, 'learning_rate': 0.1,
         'subsample': 0.9, 'colsample_bytree': 0.7},
        {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.15,
         'subsample': 0.8, 'colsample_bytree': 0.9},
    ]

    for idx, params in enumerate(xgb_configs, 1):
        # Without SMOTE
        run_name = f"XGB_{idx}_{experiment_name}_no_smote"
        model, metrics = train_model(
            X_train, X_test, y_train, y_test,
            'xgboost', params, run_name,
            use_smote=False, include_engagement=include_engagement,
            preprocessor=preprocessor, feature_engineer=feature_engineer
        )
        results.append({'model_type': 'XGBoost', 'config': idx,
                       'smote': False, **metrics})

        # With SMOTE
        run_name = f"XGB_{idx}_{experiment_name}_with_smote"
        model, metrics = train_model(
            X_train, X_test, y_train, y_test,
            'xgboost', params, run_name,
            use_smote=True, include_engagement=include_engagement,
            preprocessor=preprocessor, feature_engineer=feature_engineer
        )
        results.append({'model_type': 'XGBoost', 'config': idx,
                       'smote': True, **metrics})

    return pd.DataFrame(results)


def main():
    """
    Main extended training pipeline with comprehensive model comparison.
    """
    script_start_time = time.time()

    print("=" * 80)
    print("Module 1 - Step 3: EXTENDED Model Training with MLflow")
    print("MODE: EXTENDED (3 models × 4 configs × 2 SMOTE variants = 24 runs)")
    print("=" * 80)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ==================== SETUP AND DATA LOADING ====================
    setup_mlflow()
    df = load_data()

    # ==================== ENGINEERED EXPERIMENTS ====================
    print("\n" + "=" * 80)
    print("PHASE 1: ENGINEERED FEATURES (WITH ENGAGEMENT SCORE)")
    print("=" * 80)

    phase1_start = time.time()

    X_engineered, y, preprocessor_eng, feature_engineer_eng = preprocess_for_training(df, include_engagement=True)
    X_train_eng, X_test_eng, y_train, y_test = split_data(
        X_engineered, y,
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"]
    )

    results_engineered = run_extended_experiments(
        X_train_eng, X_test_eng, y_train, y_test,
        experiment_name="engineered",
        include_engagement=True,
        preprocessor=preprocessor_eng,
        feature_engineer=feature_engineer_eng
    )

    phase1_elapsed = time.time() - phase1_start

    # ==================== SUMMARY AND RESULTS ====================
    total_elapsed = time.time() - script_start_time
    total_experiments = len(results_engineered)

    print("\n" + "=" * 80)
    print("⏱️  EXECUTION TIMING:")
    print("=" * 80)
    print(f"  Phase 1 (Engineered):     {phase1_elapsed:7.2f} seconds")
    print(f"  Total execution time:     {total_elapsed:7.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"  Average per experiment:   {total_elapsed/total_experiments:.2f} seconds")
    print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Print comprehensive summary
    print_summary(results_engineered)

    # Save results
    save_results(results_engineered, total_elapsed, phase1_elapsed)

    print("\n" + "=" * 80)
    print("✅ EXTENDED TRAINING COMPLETE!")
    print(f"\nTotal experiments run: {total_experiments}")
    print("\n🔍 View experiments in MLflow UI:")
    print("  1. Run: mlflow ui")
    print("  2. Open: http://localhost:5000")
    print("\n💡 Tips for MLflow UI:")
    print("  • Use tags to filter: model_family, smote, features")
    print("  • Compare runs side-by-side")
    print("  • Sort by test_roc_auc to find best models")
    print("  • Check feature_importance for tree-based models")
    print("\n💬 Script Selection:")
    print("  Quick version (4 runs):   python 03_train_quick.py")
    print("  Extended version (24 runs): python 03_train_extended.py")
    print("\nNext step: 04_deploy.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
