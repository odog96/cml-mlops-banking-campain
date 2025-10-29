"""
Module 1 - Step 3: Model Training with MLflow
==============================================

This script demonstrates Cloudera AI MLflow capabilities:
1. Experiment tracking across multiple model types
2. Hyperparameter comparison (3-4 configs per model)
3. Feature engineering impact (baseline vs engineered)
4. Class imbalance handling (with/without SMOTE)
5. Comprehensive metrics logging

Models: Logistic Regression, Random Forest, XGBoost
Total Experiments: ~24 runs (3 models × 4 configs × 2 SMOTE variants)
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils import MLFLOW_CONFIG, MODEL_CONFIG
from module1.utils import preprocess_for_training, split_data


def setup_mlflow():
    """Initialize MLflow experiment"""
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    experiment_name = MLFLOW_CONFIG["experiment_name"]
    mlflow.set_experiment(experiment_name)
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {experiment_name}")
    print("-" * 80)


def load_data():
    """Load the engineered dataset"""
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
                run_name, use_smote=False, include_engagement=True):
    """
    Train a single model and log everything to MLflow
    
    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        model_type: 'logistic', 'random_forest', or 'xgboost'
        params: Model hyperparameters
        run_name: Name for this MLflow run
        use_smote: Whether to apply SMOTE for class balancing
        include_engagement: Whether engagement_score was used
        
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
        
        # Log model
        signature = infer_signature(X_train, y_pred_test)
        mlflow.sklearn.log_model(
            model, 
            "model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        # OPTIONAL: Log artifacts (confusion matrix, ROC curve, feature importance plot)
        # Commented out for now - uncomment to enable
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {run_name}')
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close()
        
        # Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            fig, ax = plt.subplots(figsize=(10, 8))
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
            ax.set_title(f'Top 15 Feature Importances - {run_name}')
            mlflow.log_figure(fig, "feature_importance.png")
            plt.close()
        """
        
        # Print results
        print(f"  Train Accuracy: {train_metrics['accuracy']:.4f} | Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test Precision: {test_metrics['precision']:.4f} | Test Recall: {test_metrics['recall']:.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f} | Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        
        return model, test_metrics


def run_experiment_suite(X_train, X_test, y_train, y_test, 
                        experiment_name, include_engagement=True):
    """
    Run a complete suite of experiments: 3 models × multiple configs × with/without SMOTE
    
    Returns:
        Dictionary of results
    """
    results = []
    
    print("\n" + "=" * 80)
    print(f"EXPERIMENT SUITE: {experiment_name}")
    print("=" * 80)
    
    # ==================== LOGISTIC REGRESSION ====================
    print("\n" + "-" * 80)
    print("LOGISTIC REGRESSION EXPERIMENTS")
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
            use_smote=False, include_engagement=include_engagement
        )
        results.append({'model_type': 'Logistic Regression', 'config': idx, 
                       'smote': False, **metrics})
        
        # With SMOTE
        run_name = f"LR_{idx}_{experiment_name}_with_smote"
        model, metrics = train_model(
            X_train, X_test, y_train, y_test,
            'logistic', params, run_name,
            use_smote=True, include_engagement=include_engagement
        )
        results.append({'model_type': 'Logistic Regression', 'config': idx, 
                       'smote': True, **metrics})
    
    # ==================== RANDOM FOREST ====================
    print("\n" + "-" * 80)
    print("RANDOM FOREST EXPERIMENTS")
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
            use_smote=False, include_engagement=include_engagement
        )
        results.append({'model_type': 'Random Forest', 'config': idx, 
                       'smote': False, **metrics})
        
        # With SMOTE
        run_name = f"RF_{idx}_{experiment_name}_with_smote"
        model, metrics = train_model(
            X_train, X_test, y_train, y_test,
            'random_forest', params, run_name,
            use_smote=True, include_engagement=include_engagement
        )
        results.append({'model_type': 'Random Forest', 'config': idx, 
                       'smote': True, **metrics})
    
    # ==================== XGBOOST ====================
    print("\n" + "-" * 80)
    print("XGBOOST EXPERIMENTS")
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
            use_smote=False, include_engagement=include_engagement
        )
        results.append({'model_type': 'XGBoost', 'config': idx, 
                       'smote': False, **metrics})
        
        # With SMOTE
        run_name = f"XGB_{idx}_{experiment_name}_with_smote"
        model, metrics = train_model(
            X_train, X_test, y_train, y_test,
            'xgboost', params, run_name,
            use_smote=True, include_engagement=include_engagement
        )
        results.append({'model_type': 'XGBoost', 'config': idx, 
                       'smote': True, **metrics})
    
    return pd.DataFrame(results)


def main():
    """Main training pipeline"""
    print("=" * 80)
    print("Module 1 - Step 3: Comprehensive Model Training with MLflow")
    print("=" * 80)
    
    # Setup
    setup_mlflow()
    df = load_data()
    
    # ==================== BASELINE EXPERIMENTS ====================
    print("\n" + "=" * 80)
    print("PHASE 1: BASELINE FEATURES (WITHOUT ENGAGEMENT SCORE)")
    print("=" * 80)
    
    X_baseline, y, _, _ = preprocess_for_training(df, include_engagement=False)
    X_train_base, X_test_base, y_train, y_test = split_data(
        X_baseline, y, 
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"]
    )
    
    results_baseline = run_experiment_suite(
        X_train_base, X_test_base, y_train, y_test,
        experiment_name="baseline",
        include_engagement=False
    )
    
    # ==================== ENGINEERED EXPERIMENTS ====================
    print("\n" + "=" * 80)
    print("PHASE 2: ENGINEERED FEATURES (WITH ENGAGEMENT SCORE)")
    print("=" * 80)
    
    X_engineered, y, _, _ = preprocess_for_training(df, include_engagement=True)
    X_train_eng, X_test_eng, y_train, y_test = split_data(
        X_engineered, y,
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"]
    )
    
    results_engineered = run_experiment_suite(
        X_train_eng, X_test_eng, y_train, y_test,
        experiment_name="engineered",
        include_engagement=True
    )
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
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
    
    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print(f"\nTotal experiments run: {len(results_baseline) + len(results_engineered)}")
    print("\n🔍 View experiments in MLflow UI:")
    print("  1. Run: mlflow ui")
    print("  2. Open: http://localhost:5000")
    print("\n💡 Tips for MLflow UI:")
    print("  • Use tags to filter: model_family, smote, features")
    print("  • Compare runs side-by-side")
    print("  • Sort by test_roc_auc to find best models")
    print("  • Check feature_importance for tree-based models")
    print("\nNext step: 04_deploy_model.py")
    print("=" * 80)
    
    # Save summary
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    results_baseline.to_csv(f"{output_dir}/baseline_results.csv", index=False)
    results_engineered.to_csv(f"{output_dir}/engineered_results.csv", index=False)
    
    print(f"\n💾 Results saved to {output_dir}/")


if __name__ == "__main__":
    main()




