"""
Debug Build Step Only

This script skips registration and model creation, and focuses ONLY on the build step.
Use this to iterate quickly when debugging the build without waiting for earlier steps.

You need to have already run v2_api_deployment.py once to get:
- registered_model_id
- model_version_id
- model_id (cml_model_id)

These will be in outputs/deployment_info.json
"""

import json
import os
import cmlapi
from cmlapi.rest import ApiException
from pprint import pprint

print("=" * 80)
print("Debug Build Step Only")
print("=" * 80)

# ============================================================================
# Load deployment info from previous run
# ============================================================================
print("\n[1/2] Loading deployment info from previous run...")

try:
    with open("outputs/deployment_info.json", "r") as f:
        deployment_info = json.load(f)

    registered_model_id = deployment_info["registered_model_id"]
    model_version_id = deployment_info["model_version_id"]
    model_id = deployment_info["cml_model_id"]
    f1_score = deployment_info["f1_score"]
    model_name = deployment_info["model_name"]

    print(f"✅ Loaded from outputs/deployment_info.json:")
    print(f"   Registered Model ID: {registered_model_id}")
    print(f"   Model Version ID: {model_version_id}")
    print(f"   CML Model ID: {model_id}")
    print(f"   F1 Score: {f1_score:.4f}")
except FileNotFoundError:
    print(f"❌ ERROR: outputs/deployment_info.json not found")
    print(f"   You must run v2_api_deployment.py first!")
    exit(1)
except KeyError as e:
    print(f"❌ ERROR: Missing key in deployment_info.json: {e}")
    exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    exit(1)

# ============================================================================
# Setup CML client and project
# ============================================================================
print("\n[2/2] Creating model build...")

client = cmlapi.default_client()
project_id = os.environ['CDSW_PROJECT_ID']

print(f"   Project ID: {project_id}")

runtime_id = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5"

create_build_request = {
    "registered_model_version_id": str(model_version_id),
    "runtime_identifier": runtime_id,
    "comment": f"Debug build - F1: {f1_score:.4f}"
}

print(f"\n   Build request:")
print(f"     registered_model_version_id: {model_version_id}")
print(f"     runtime_identifier: {runtime_id}")
print(f"     comment: Debug build - F1: {f1_score:.4f}")

print(f"\n   Calling create_model_build...", end="")

try:
    build = client.create_model_build(create_build_request, project_id, model_id)
    build_id = build.id
    print(" ✅")

    print(f"\n✅ Build created successfully!")
    print(f"   Build ID: {build_id}")
    print(f"\nFull response:")
    pprint(vars(build))

    # Update deployment info with build ID
    deployment_info["build_id"] = build_id
    with open("outputs/deployment_info.json", "w") as f:
        json.dump(deployment_info, f, indent=2)
    print(f"\n✅ Updated outputs/deployment_info.json with build ID")

except ApiException as e:
    print(f" ❌")
    print(f"\n❌ ERROR creating build:")
    print(f"   Status: {e.status}")
    print(f"   Reason: {e.reason}")
    print(f"   Body: {e.body}")

    print(f"\n📋 Debug info:")
    print(f"   registered_model_version_id: {model_version_id}")
    print(f"   model_id: {model_id}")
    print(f"   project_id: {project_id}")
    print(f"   runtime_id: {runtime_id}")

except Exception as e:
    print(f" ❌")
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
