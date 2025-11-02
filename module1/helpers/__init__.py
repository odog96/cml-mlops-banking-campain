"""
Helpers package for Module 1

Contains utility modules:
- preprocessing: Feature engineering and preprocessing pipelines
- _training_utils: Shared training utilities for MLflow
- utils: General utility functions
- test_runner: Automated test runner for sequential script execution
"""

from .preprocessing import FeatureEngineer, PreprocessingPipeline, preprocess_for_training, split_data
from ._training_utils import setup_mlflow, train_model, calculate_metrics, apply_smote, save_results, print_summary
from .utils import engineer_customer_engagement_score, engineer_features, calculate_feature_importance_summary
from .test_runner import TestRunner

__all__ = [
    'FeatureEngineer',
    'PreprocessingPipeline',
    'preprocess_for_training',
    'split_data',
    'setup_mlflow',
    'train_model',
    'calculate_metrics',
    'apply_smote',
    'save_results',
    'print_summary',
    'engineer_customer_engagement_score',
    'engineer_features',
    'calculate_feature_importance_summary',
    'TestRunner',
]
