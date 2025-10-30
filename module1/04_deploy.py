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


def calculate_business_score(metrics):
    """
    Calculate a custom composite metric based on business priorities

    This weighted score balances:
    - F1 Score (35%): Overall model balance
    - Precision (30%): Cost efficiency (avoid wasting marketing spend)
    - Recall (20%): Revenue potential (don't miss opportunities)
    - ROC-AUC (15%): Overall discriminative ability

    Args:
        metrics: Dictionary of metrics from MLflow run

    Returns:
        Composite business score (0-1)
    """
    return (
        0.35 * metrics.get('test_f1', 0) +
        0.30 * metrics.get('test_precision', 0) +
        0.20 * metrics.get('test_recall', 0) +
        0.15 * metrics.get('test_roc_auc', 0)
    )


def get_best_model_from_mlflow(metric="test_f1", use_composite=False):
    """
    Query MLflow to find the best performing model

    Args:
        metric: Metric to optimize (default: test_f1)
                Options: test_f1, test_roc_auc, test_precision, test_recall, test_accuracy
        use_composite: If True, use custom composite business score instead

    Returns:
        run_id, model_uri, metrics, params of best model
    """
    if use_composite:
        print("Searching for best model using composite business score...")
    else:
        print(f"Searching for best model by {metric}...")

    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    client = MlflowClient()

    # Get experiment
    experiment_name = MLFLOW_CONFIG["experiment_name"]
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found. Run training first!")

    if use_composite:
        # Get all runs and calculate composite score
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=1000
        )

        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")

        # Calculate composite score for each run
        best_run = None
        best_score = -1

        for run in runs:
            metrics = {
                'test_accuracy': run.data.metrics.get('test_accuracy', 0),
                'test_roc_auc': run.data.metrics.get('test_roc_auc', 0),
                'test_f1': run.data.metrics.get('test_f1', 0),
                'test_precision': run.data.metrics.get('test_precision', 0),
                'test_recall': run.data.metrics.get('test_recall', 0),
            }
            score = calculate_business_score(metrics)

            if score > best_score:
                best_score = score
                best_run = run

        if best_run is None:
            raise ValueError("Could not find best model")

    else:
        # Search for best run by single metric
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )

        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")

        best_run = runs[0]
        best_score = best_run.data.metrics.get(metric, 0)

    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    # Extract all metrics
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
    print(f"  Run Name: {best_run.info.run_name}")
    print(f"  Model Type: {params.get('model_type', 'N/A')}")
    print(f"  SMOTE: {params.get('use_smote', 'N/A')}")
    print(f"  Engineered Features: {params.get('include_engagement_score', 'N/A')}")

    if use_composite:
        print(f"\n  Composite Business Score: {best_score:.4f}")
    else:
        print(f"\n  {metric}: {best_score:.4f}")

    print(f"\n  Performance Metrics:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1 Score:  {metrics['f1']:.4f}")
    print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")

    return run_id, model_uri, metrics, params


def register_model_in_mlflow(model_uri, model_name, run_id, metrics, params):
    """
    Register the model in MLflow Model Registry

    Args:
        model_uri: URI of the model in MLflow
        model_name: Name to register the model under
        run_id: MLflow run ID
        metrics: Model metrics dictionary
        params: Model parameters dictionary

    Returns:
        model_version
    """
    print(f"\nRegistering model in MLflow Model Registry...")

    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])

    # Register model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    # Add description and tags
    client = MlflowClient()
    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=f"Banking campaign prediction model (Run: {run_id})"
    )

    # Set tags with key metrics and params
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="model_type",
        value=params.get('model_type', 'N/A')
    )

    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="f1_score",
        value=f"{metrics['f1']:.4f}"
    )

    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="roc_auc",
        value=f"{metrics['roc_auc']:.4f}"
    )

    print(f"✓ Model registered as '{model_name}' version {model_version.version}")

    return model_version


def deploy_model_to_cml(model_name, model_version):
    """
    Deploy the model as a CML REST API endpoint

    Args:
        model_name: Name of the registered model
        model_version: Model version object

    Returns:
        deployment info
    """
    print(f"\nDeploying model to CML...")

    try:
        # Initialize CML API client
        client = cmlapi.default_client()

        # Get current project
        project_id = os.environ.get("CDSW_PROJECT_ID")
        if not project_id:
            print("⚠ Warning: Not running in CML environment. Skipping CML deployment.")
            print("  Model is registered in MLflow and can be loaded programmatically.")
            return None

        # Create model deployment
        create_model_request = cmlapi.CreateModelRequest(
            project_id=project_id,
            name=API_CONFIG["model_name"],
            description=API_CONFIG["description"],
            registered_model_name=model_name,
            registered_model_version=str(model_version.version)
        )

        model = client.create_model(create_model_request, project_id)

        # Create model build
        create_build_request = cmlapi.CreateModelBuildRequest(
            project_id=project_id,
            model_id=model.id,
            file_path="module1/model_api.py",  # API endpoint script
            function_name="predict",
            kernel="python3",
            cpu=API_CONFIG["cpu"],
            memory=API_CONFIG["memory"]
        )

        build = client.create_model_build(create_build_request, project_id, model.id)

        print(f"✓ Model deployed!")
        print(f"  Model ID: {model.id}")
        print(f"  Build ID: {build.id}")
        print(f"\nAccess the model deployment in the CML UI to:")
        print(f"  1. Monitor build progress")
        print(f"  2. Test the API endpoint")
        print(f"  3. Get the REST API URL")

        return {"model_id": model.id, "build_id": build.id}

    except Exception as e:
        print(f"⚠ Warning: Could not deploy to CML: {e}")
        print(f"  Model is registered in MLflow Model Registry")
        print(f"  You can load it programmatically with:")
        print(f"    model = mlflow.pyfunc.load_model('models:/{model_name}/{model_version.version}')")
        return None


def main():
    """Main deployment pipeline"""
    print("=" * 80)
    print("Module 1 - Step 4: Model Deployment")
    print("=" * 80)

    # Configuration - Change these to adjust model selection
    SELECTION_METHOD = "f1"  # Options: "f1", "composite", "roc_auc", "precision", "recall"
    MODEL_NAME = "banking_campaign_predictor"

    print(f"\nSelection Method: {SELECTION_METHOD}")
    print(f"Model Registry Name: {MODEL_NAME}")

    # Step 1: Find best model
    if SELECTION_METHOD == "composite":
        run_id, model_uri, metrics, params = get_best_model_from_mlflow(use_composite=True)
    else:
        run_id, model_uri, metrics, params = get_best_model_from_mlflow(metric=f"test_{SELECTION_METHOD}")

    # Step 2: Register in MLflow
    model_version = register_model_in_mlflow(model_uri, MODEL_NAME, run_id, metrics, params)

    # Step 3: Deploy to CML (optional)
    deployment = deploy_model_to_cml(MODEL_NAME, model_version)

    print("\n" + "=" * 80)
    print("✅ Deployment complete!")
    print("=" * 80)

    print("\n📊 Model Selection Summary:")
    print(f"  Method: {SELECTION_METHOD}")
    print(f"  Model Type: {params.get('model_type', 'N/A')}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    print("\n💡 Next Steps:")
    print("  1. Test the deployed model with sample data")
    print("  2. Monitor model performance in production")
    print("  3. Set up model monitoring and retraining pipelines")
    print("  4. Consider A/B testing different model versions")

    print("\n📝 To change model selection method, edit SELECTION_METHOD in this script:")
    print("     - 'f1' (default): Balanced precision/recall")
    print("     - 'composite': Custom business score")
    print("     - 'precision': Minimize false positives")
    print("     - 'recall': Minimize false negatives")
    print("     - 'roc_auc': Overall discriminative ability")

    # Save deployment info
    deployment_info = {
        'run_id': run_id,
        'model_name': MODEL_NAME,
        'model_version': model_version.version,
        'selection_method': SELECTION_METHOD,
        'metrics': metrics,
        'params': params,
        'deployment': deployment
    }

    os.makedirs('outputs', exist_ok=True)
    with open('outputs/deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)

    print("\n💾 Deployment info saved to outputs/deployment_info.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
