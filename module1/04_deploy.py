"""
Module 1 - Step 4: Model Deployment
====================================

This script:
1. Queries MLflow to find the best model
2. Registers the model in CML Model Registry
3. Deploys it as a REST API endpoint

Key Learning Points:
- Automated model selection based on metrics
- Model registry for versioning
- Production deployment patterns
"""

import os
import sys
import json
import mlflow
import cmlapi
from mlflow.tracking import MlflowClient

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils import MLFLOW_CONFIG, API_CONFIG


def get_best_model_from_mlflow(metric="test_roc_auc"):
    """
    Query MLflow to find the best performing model
    
    Args:
        metric: Metric to optimize (default: test_roc_auc)
        
    Returns:
        run_id, model_uri, metrics of best model
    """
    print(f"Searching for best model by {metric}...")
    
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    client = MlflowClient()
    
    # Get experiment
    experiment_name = MLFLOW_CONFIG["experiment_name"]
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found. Run training first!")
    
    # Search for best run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")
    
    best_run = runs[0]
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    
    # Extract metrics
    metrics = {
        'accuracy': best_run.data.metrics.get('test_accuracy', 0),
        'roc_auc': best_run.data.metrics.get('test_roc_auc', 0),
        'f1': best_run.data.metrics.get('test_f1', 0),
        'precision': best_run.data.metrics.get('test_precision', 0),
        'recall': best_run.data.metrics.get('test_recall', 0),
    }
    
    # Extract parameters
    params = best_run.data.params
    
    print(f"\n✓ Best model found!")
    print(f"  Run ID: {run_id}")
    print(f"  