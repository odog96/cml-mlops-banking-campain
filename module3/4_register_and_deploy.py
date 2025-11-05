import os
import sys
import json
import time
import cmlapi
from cmlapi.rest import ApiException

print("=" * 80)
print("Module 3 - Step 4: Register and Deploy (V1 API Pattern)")
print("=" * 80)

# Get the project root directory (CML jobs run from /home/cdsw)
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # In Jupyter notebooks, __file__ is not defined
    BASE_DIR = os.getcwd()

# ============================================================================
# Step 1: Load info from the retraining job
# ============================================================================
print("\n[1/6] Loading info from retraining job...")

RUN_INFO_FILE = os.path.join(BASE_DIR, "outputs", "retrain_run_info.json")

try:
    with open(RUN_INFO_FILE, "r") as f:
        retrain_info = json.load(f)
except FileNotFoundError:
    print(f"❌ ERROR: '{RUN_INFO_FILE}' not found.")
    print("   Did you run '3_retrain_model.py' first?")
    sys.exit(1)

run_id = retrain_info.get("run_id")
experiment_id = retrain_info.get("experiment_id")
#MODEL_NAME = retrain_info.get("model_name", "banking_campaign_predictor_v2")
MODEL_NAME = os.environ.get("MODEL_NAME", "banking_campaign_predictor")
f1_score = retrain_info.get("f1_score", 0.0)

print(f"✅ Loaded info for retrained model:")
print(f"   Run ID: {run_id}")
print(f"   Experiment ID: {experiment_id}")
print(f"   Model Name: {MODEL_NAME}")
print(f"   F1 Score: {f1_score:.4f}")

# ============================================================================
# Step 2: Register model in CML 
# ============================================================================
print("\n[2/6] Registering model in CML...")

cml_client = cmlapi.default_client()
project_id = os.environ.get("CDSW_PROJECT_ID")

if not project_id:
    print("❌ ERROR: Not running in CML environment (CDSW_PROJECT_ID not set)")
    sys.exit(1)

# This is the exact pattern from your example script
create_registered_model_request = {
    "project_id": project_id,
    "experiment_id": experiment_id,
    "run_id": run_id,
    "model_name": MODEL_NAME,
    "model_path": "model" # This is the artifact_path from log_model()
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
print("\n[3/6] Waiting for CML to finalize...")
time.sleep(20) # Wait for registration to propagate
print("   ✅ Ready")

# ============================================================================
# Step 4: Create CML model (the API endpoint)
# ============================================================================
print("\n[4/6] Creating CML model endpoint...")

create_model_request = cmlapi.CreateModelRequest(
    project_id=project_id,
    name=MODEL_NAME,
    description=f"Retrained banking model (F1: {f1_score:.4f})",
    registered_model_id=registered_model_id,
    disable_authentication=True
)

try:
    cml_model = cml_client.create_model(
        body=create_model_request,
        project_id=project_id
    )
    print(f"   ✅ Model endpoint created: {cml_model.id}")
    
except ApiException as e:
    if "already has a model with that name" in str(e.body):
        print(f"   ⚠️  Model endpoint already exists, reusing it...")
        models = cml_client.list_models(project_id)
        cml_model = next((m for m in models.models if m.name == MODEL_NAME), None)

        if not cml_model:
            print(f"   ❌ ERROR: Could not find existing model '{MODEL_NAME}'")
            sys.exit(1)

        print(f"   ✅ Reusing existing model endpoint: {cml_model.id}")
        print(f"   (New model version will be deployed to this endpoint)")
    else:
        print(f"   ❌ ERROR: {e.reason}")
        print(f"   Body: {e.body}")
        sys.exit(1)

# ============================================================================
# Step 5: Create model build
# ============================================================================
print("\n[5/6] Creating model build...")

# Using the same standard runtime from your example
runtime_id = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5"

create_build_request = cmlapi.CreateModelBuildRequest(
    registered_model_version_id=str(model_version_id),
    runtime_identifier=runtime_id,
    comment=f"Retrained build - F1: {f1_score:.4f}"
)

try:
    build = cml_client.create_model_build(
        body=create_build_request,
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

max_wait_minutes = 15
check_interval_seconds = 30
checks = (max_wait_minutes * 60) // check_interval_seconds

print(f"   ⏳ Checking build status every {check_interval_seconds} seconds...")

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
            print(f"   ❌ Build failed! Check CML UI for build logs.")
            sys.exit(1)
        elif status in ["building", "queued"]:
            time.sleep(check_interval_seconds)
        else:
            print(f"   ⚠️  Unknown status: {status}")
            time.sleep(check_interval_seconds)
            
    except Exception as e:
        print(f"   ⚠️  Error checking build status: {e}")
        time.sleep(check_interval_seconds)

if not build_succeeded:
    print(f"   ⚠️  Build did not complete within {max_wait_minutes} minutes")
    print(f"   The build is still running. Deploy manually when ready.")
    deployment_id = None
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
        deployment_id = deployment.id
        print(f"   ✅ Deployment created: {deployment.id}")
        
    except ApiException as e:
        print(f"   ❌ ERROR creating deployment: {e.body}")
        deployment_id = None

# ============================================================================
# Success Summary
# ============================================================================
print("\n" + "=" * 80)
if deployment_id:
    print("✅ MLOPS PIPELINE COMPLETE! (DEPLOYED)")
else:
    print("✅ MLOPS PIPELINE COMPLETE! (BUILT)")
print("=" * 80)
print(f"   Model Name: {MODEL_NAME}")
print(f"   F1 Score: {f1_score:.4f}")
print(f"   CML Model ID: {cml_model.id}")
print(f"   Build ID: {build.id}")

if deployment_id:
    print(f"   Deployment ID: {deployment_id}")
    print(f"\n✅ REST API ENDPOINT IS LIVE!")
    print(f"   Access it: Models > {MODEL_NAME} > Deployments")
else:
    print(f"\n⏳ To deploy manually:")
    print(f"   1. Go to: Models > {MODEL_NAME} > Builds")
    print(f"   2. Once build shows 'Built', click Deploy")

# Save final info
deployment_info = {
    "run_id": run_id,
    "model_name": MODEL_NAME,
    "registered_model_id": registered_model_id,
    "model_version_id": str(model_version_id),
    "f1_score": float(f1_score),
    "cml_model_id": cml_model.id,
    "build_id": build.id,
    "deployment_id": deployment_id,
    "status": "Deployed" if deployment_id else "Built"
}

os.makedirs("outputs", exist_ok=True)
with open(os.path.join("outputs", "deployment_info_v2.json"), "w") as f:
    json.dump(deployment_info, f, indent=2)

print(f"\n💾 Saved to: outputs/deployment_info_v2.json")
print("=" * 80)
