"""
Module 1 - Step 3: Quick Model Training with MLflow (Lab Version)
==================================================================

This script demonstrates Cloudera AI MLflow capabilities with a fast setup:
1. Experiment tracking across baseline and engineered features
2. Single model type (Logistic Regression) with 2 configs
3. Class imbalance handling with SMOTE
4. Comprehensive metrics logging

Configuration:
- 1 model (Logistic Regression) × 2 configs × 2 SMOTE variants × 2 feature sets
- Total: 8 experiment runs
- Expected runtime: 2-5 minutes (depending on system)

This is the perfect version for:
- Learning and understanding the workflow
- Quick prototyping
- Lab environments with time constraints
- Testing the pipeline

For a comprehensive model comparison, use: python 03_train_extended.py
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


def run_quick_experiments(X_train, X_test, y_train, y_test,
                          experiment_name, include_engagement=True,
                          preprocessor=None, feature_engineer=None):
    """
    Run a quick suite of Logistic Regression experiments.

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
    print(f"QUICK EXPERIMENTS: {experiment_name.upper()}")
    print("=" * 80)
    print("\nLOGISTIC REGRESSION (2 CONFIGS)")
    print("-" * 80)

    # Quick mode: 2 configs for Logistic Regression
    lr_configs = [
        {'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs'},      # Default balanced
        {'C': 0.1, 'max_iter': 1000, 'solver': 'lbfgs'},      # Strong regularization
    ]

    # Train models with and without SMOTE
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

    return pd.DataFrame(results)


def main():
    """
    Main quick training pipeline.
    """
    script_start_time = time.time()

    print("=" * 80)
    print("Module 1 - Step 3: QUICK Model Training with MLflow")
    print("MODE: QUICK (1 model × 2 configs × 2 SMOTE variants = 4 runs)")
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

    results_engineered = run_quick_experiments(
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
    print("✅ QUICK TRAINING COMPLETE!")
    print(f"\nTotal experiments run: {total_experiments}")
    print("\n🔍 View experiments in MLflow UI:")
    print("  1. Run: mlflow ui")
    print("  2. Open: http://localhost:5000")
    print("\n💡 Tips for MLflow UI:")
    print("  • Use tags to filter: model_family, smote, features")
    print("  • Compare runs side-by-side")
    print("  • Sort by test_roc_auc to find best models")
    print("\n🚀 Ready for more? Try the extended version:")
    print("  python 03_train_extended.py")
    print("\nNext step: 04_deploy.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
