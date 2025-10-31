"""
Helpers package for Module 1

Contains utility modules:
- preprocessing: Feature engineering and preprocessing pipelines
- _training_utils: Shared training utilities for MLflow
- utils: General utility functions
- test_runner: Automated test runner for sequential script execution
"""

from .preprocessing import FeatureEngineer, PreprocessingPipeline, preprocess_for_training, split_data
from ._training_utils import setup_mlflow, create_experiment, log_metrics, save_model_artifacts, get_best_run
from .utils import engineer_customer_engagement_score, engineer_features, calculate_feature_importance_summary
from .test_runner import TestRunner

__all__ = [
    'FeatureEngineer',
    'PreprocessingPipeline',
    'preprocess_for_training',
    'split_data',
    'setup_mlflow',
    'create_experiment',
    'log_metrics',
    'save_model_artifacts',
    'get_best_run',
    'engineer_customer_engagement_score',
    'engineer_features',
    'calculate_feature_importance_summary',
    'TestRunner',
]
