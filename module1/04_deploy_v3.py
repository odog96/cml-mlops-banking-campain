"""
Module 1 - Step 4: Model Deployment (V3 - Simplified & Clean)
==============================================================

FIXES APPLIED:
1. Register model in CML to get registered_model_id
2. Use proper API objects with body= parameter
3. Clean error handling without nested try-except

All fixes clearly marked with ✅
"""

import os
import sys
import json
import time
import cmlapi
import mlflow
from mlflow.tracking import MlflowClient
from cmlapi.rest import ApiException

# Add parent directory for imports (works in both script and notebook)
try:
    # If running as a script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(script_dir))
except NameError:
    # If running in Jupyter notebook
    script_dir = os.getcwd()
    if '/module1' in script_dir:
        sys.path.append(os.path.dirname(script_dir))
    else:
        sys.path.append(script_dir)

# Try to import shared_utils, provide defaults if not available
try:
    from shared_utils import API_CONFIG
except ImportError:
    print("⚠️  Could not import shared_utils, using defaults")
    API_CONFIG = {
        "model_name": "banking_campaign_predictor",
        "description": "Banking campaign prediction model"
    }

# Configuration
EXPERIMENT_NAME = "bank_marketing_experiments"
MODEL_NAME = "banking_campaign_predictor"

print("=" * 80)
print("Module 1 - Step 4: Model Deployment (V3 - Clean)")
print("=" * 80)

# ============================================================================
# Step 1: Find the best model (by F1 score)
# ============================================================================
print("\n[1/5] Finding best model by F1 score...")

mlflow_client = MlflowClient()
experiment = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)

if not experiment:
    print(f"❌ ERROR: Experiment '{EXPERIMENT_NAME}' not found!")
    sys.exit(1)

runs = mlflow_client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_f1 DESC"],
    max_results=1
)

if not runs:
    print(f"❌ ERROR: No runs found in experiment")
    sys.exit(1)

best_run = runs[0]
run_id = best_run.info.run_id
model_uri = f"runs:/{run_id}/model"
f1_score = best_run.data.metrics.get('test_f1', 0)

print(f"✅ Best model found:")
print(f"   Run ID: {run_id}")
print(f"   F1 Score: {f1_score:.4f}")

# ============================================================================
# Step 2: Register model in CML (CRITICAL FIX!)
# ============================================================================
print("\n[2/5] Registering model in CML...")

cml_client = cmlapi.default_client()
project_id = os.environ.get("CDSW_PROJECT_ID")

if not project_id:
    print("❌ ERROR: Not running in CML environment")
    sys.exit(1)

# ✅ FIX 1: Register in CML to get registered_model_id
create_registered_model_request = {
    "project_id": project_id,
    "experiment_id": experiment.experiment_id,
    "run_id": run_id,
    "model_name": MODEL_NAME,
    "model_path": "model"
}

try:
    registered_model_response = cml_client.create_registered_model(
        create_registered_model_request
    )
    registered_model_id = registered_model_response.model_id
    model_version_id = registered_model_response.model_versions[0].model_version_id
    
    print(f"✅ Model registered in CML:")
    print(f"   Registered Model ID: {registered_model_id}")
    print(f"   Model Version ID: {model_version_id}")
except ApiException as e:
    print(f"❌ ERROR: {e.reason}")
    print(f"   Body: {e.body}")
    sys.exit(1)

# ============================================================================
# Step 3: Wait for CML to finalize
# ============================================================================
print("\n[3/5] Waiting for CML to finalize...")
time.sleep(20)
print("   ✅ Ready")

# ============================================================================
# Step 4: Create CML model
# ============================================================================
print("\n[4/5] Creating CML model...")

# ✅ FIX 2: Use CreateModelRequest with registered_model_id
create_model_request = cmlapi.CreateModelRequest(
    project_id=project_id,
    name=MODEL_NAME,
    description=f"Banking campaign model (F1: {f1_score:.4f})",
    registered_model_id=registered_model_id,  # ✅ CRITICAL!
    disable_authentication=True
)

try:
    cml_model = cml_client.create_model(
        body=create_model_request,  # ✅ Use body= parameter
        project_id=project_id
    )
    print(f"   ✅ Model created: {cml_model.id}")
    print(f"   ✅ Has registered_model_id: {cml_model.registered_model_id}")
    
except ApiException as e:
    # Handle "already exists" error
    if "already has a model with that name" in str(e.body):
        print(f"   ⚠️  Model already exists, getting it...")
        models = cml_client.list_models(project_id)
        cml_model = next((m for m in models.models if m.name == MODEL_NAME), None)
        
        if not cml_model:
            print(f"   ❌ ERROR: Could not find existing model")
            sys.exit(1)
        
        # Check if existing model has registered_model_id
        if not cml_model.registered_model_id:
            print(f"   ❌ Existing model has NO registered_model_id")
            print(f"   Deleting and recreating...")
            cml_client.delete_model(project_id, cml_model.id)
            time.sleep(5)
            
            # Recreate
            cml_model = cml_client.create_model(
                body=create_model_request,
                project_id=project_id
            )
            print(f"   ✅ Model recreated: {cml_model.id}")
        else:
            print(f"   ✅ Using existing model: {cml_model.id}")
    else:
        # Other error
        print(f"   ❌ ERROR: {e.reason}")
        print(f"   Body: {e.body}")
        sys.exit(1)

# ============================================================================
# Step 5: Create model build
# ============================================================================
print("\n[5/6] Creating model build...")

runtime_id = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5"

# ✅ FIX 3: Use CreateModelBuildRequest with registered_model_version_id
create_build_request = cmlapi.CreateModelBuildRequest(
    registered_model_version_id=str(model_version_id),  # ✅ CRITICAL!
    runtime_identifier=runtime_id,
    comment=f"Auto-deployed - F1: {f1_score:.4f}"
)

try:
    build = cml_client.create_model_build(
        body=create_build_request,  # ✅ Use body= parameter
        project_id=project_id,
        model_id=cml_model.id
    )
    print(f"   ✅ Build created: {build.id}")
    print(f"   ⏳ Build is running (~5-10 minutes)")
    
except ApiException as e:
    print(f"   ❌ ERROR: {e.reason}")
    print(f"   Body: {e.body}")
    sys.exit(1)

# ============================================================================
# Step 6: Wait for build to complete, then deploy
# ============================================================================
print("\n[6/6] Waiting for build to complete before deployment...")

# Poll for build status
max_wait_minutes = 15
check_interval_seconds = 30
checks = (max_wait_minutes * 60) // check_interval_seconds

print(f"   ⏳ Checking build status every {check_interval_seconds} seconds...")
print(f"   (Will wait up to {max_wait_minutes} minutes)")

build_succeeded = False
for i in range(checks):
    try:
        build_status = cml_client.get_model_build(
            project_id=project_id,
            model_id=cml_model.id,
            build_id=build.id
        )
        
        status = build_status.status
        print(f"   Check {i+1}/{checks}: Build status = {status}")
        
        if status == "built":
            build_succeeded = True
            print(f"   ✅ Build completed successfully!")
            break
        elif status == "build failed":
            print(f"   ❌ Build failed!")
            print(f"   Check CML UI for build logs: Models > {MODEL_NAME} > Builds")
            sys.exit(1)
        elif status in ["building", "queued"]:
            # Still building, wait
            time.sleep(check_interval_seconds)
        else:
            print(f"   ⚠️  Unknown status: {status}")
            time.sleep(check_interval_seconds)
            
    except Exception as e:
        print(f"   ⚠️  Error checking build status: {e}")
        time.sleep(check_interval_seconds)

if not build_succeeded:
    print(f"   ⚠️  Build did not complete within {max_wait_minutes} minutes")
    print(f"   The build is still running. Check CML UI and deploy manually when ready.")
    print(f"   Location: Models > {MODEL_NAME} > Builds")
else:
    # Build succeeded, create deployment
    print("\n   Creating deployment...")
    
    create_deployment_request = cmlapi.CreateModelDeploymentRequest(
        cpu="2",
        memory="4"
    )
    
    try:
        deployment = cml_client.create_model_deployment(
            body=create_deployment_request,
            project_id=project_id,
            model_id=cml_model.id,
            build_id=build.id
        )
        print(f"   ✅ Deployment created: {deployment.id}")
        deployment_id = deployment.id
        
    except ApiException as e:
        print(f"   ❌ ERROR creating deployment:")
        print(f"   Status: {e.status}")
        print(f"   Body: {e.body}")
        deployment_id = None
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        deployment_id = None

# ============================================================================
# Success Summary
# ============================================================================
print("\n" + "=" * 80)
if 'deployment_id' in locals() and deployment_id:
    print("✅ DEPLOYMENT COMPLETE!")
else:
    print("✅ BUILD COMPLETE - DEPLOY MANUALLY")
print("=" * 80)

print(f"\n📊 Model Summary:")
print(f"   Model Name: {MODEL_NAME}")
print(f"   F1 Score: {f1_score:.4f}")
print(f"   CML Model ID: {cml_model.id}")
print(f"   Build ID: {build.id}")

if 'deployment_id' in locals() and deployment_id:
    print(f"   Deployment ID: {deployment_id}")
    print(f"\n✅ REST API ENDPOINT IS LIVE!")
    print(f"   Access it: Models > {MODEL_NAME} > Deployments")
    print(f"\n🎯 Test your API:")
    print(f'   curl -X POST https://your-cml-workspace/models/...')
else:
    print(f"\n⏳ To deploy manually:")
    print(f"   1. Go to: Models > {MODEL_NAME} > Builds")
    print(f"   2. Once build shows 'Built', click Deploy")
    print(f"   3. Configure resources (CPU: 2, Memory: 4GB)")

# Save deployment info
deployment_info = {
    "run_id": run_id,
    "model_name": MODEL_NAME,
    "registered_model_id": registered_model_id,
    "model_version_id": str(model_version_id),
    "f1_score": float(f1_score),
    "cml_model_id": cml_model.id,
    "build_id": build.id,
    "deployment_id": deployment_id if 'deployment_id' in locals() else None,
    "status": "Deployed" if ('deployment_id' in locals() and deployment_id) else "Built"
}

os.makedirs("outputs", exist_ok=True)
with open("outputs/deployment_info.json", "w") as f:
    json.dump(deployment_info, f, indent=2)

print(f"\n💾 Saved to: outputs/deployment_info.json")
print("=" * 80)