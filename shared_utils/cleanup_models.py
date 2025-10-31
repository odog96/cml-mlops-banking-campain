"""
Model Cleanup Utility

Safely delete models and builds from CML via API.
This allows you to clean up and start fresh.

Usage:
    python cleanup_models.py
"""

import os
import cmlapi
from cmlapi.rest import ApiException
from pprint import pprint

print("=" * 80)
print("CML Model Cleanup Utility")
print("=" * 80)

# Initialize CML API client
try:
    client = cmlapi.default_client()
    print("\n✅ CML API client initialized")
except Exception as e:
    print(f"\n❌ ERROR initializing CML client: {e}")
    exit(1)

# Get project ID
project_id = os.environ.get("CDSW_PROJECT_ID")
if not project_id:
    print("\n❌ ERROR: CDSW_PROJECT_ID environment variable not set")
    print("   This script must be run in a CML project")
    exit(1)

print(f"✅ Project ID: {project_id}")

# ============================================================================
# List all models
# ============================================================================
print("\n" + "=" * 80)
print("Listing all models in project...")
print("=" * 80)

try:
    models_response = client.list_models(project_id)
    if not models_response.models:
        print("\n✅ No models found in project")
        exit(0)

    print(f"\nFound {len(models_response.models)} model(s):\n")
    for model in models_response.models:
        print(f"Model: {model.name}")
        print(f"  ID: {model.id}")
        print(f"  Description: {model.description}")

        # List builds for this model
        try:
            builds_response = client.list_model_builds(project_id, model.id)
            # Handle different response formats
            builds = None
            if hasattr(builds_response, 'builds'):
                builds = builds_response.builds
            elif isinstance(builds_response, list):
                builds = builds_response
            elif isinstance(builds_response, dict) and 'builds' in builds_response:
                builds = builds_response['builds']

            if builds:
                print(f"  Builds ({len(builds)}):")
                for build in builds:
                    build_id = build.id if hasattr(build, 'id') else build.get('id') if isinstance(build, dict) else str(build)
                    print(f"    - Build ID: {build_id}")
            else:
                print(f"  Builds: None")
        except Exception as e:
            print(f"  Could not list builds: {e}")

        print()

except ApiException as e:
    print(f"\n❌ Exception when listing models: {e}")
    exit(1)

# ============================================================================
# Delete models and builds
# ============================================================================
print("=" * 80)
print("Deletion Options")
print("=" * 80)

models_list = models_response.models

for idx, model in enumerate(models_list, 1):
    print(f"\n[{idx}] {model.name} (ID: {model.id})")

print(f"\n[0] Exit without deleting")
choice = input("\nEnter the number of the model to delete (or 0 to exit): ").strip()

if choice == "0":
    print("\n✅ Exiting without changes")
    exit(0)

try:
    choice_idx = int(choice) - 1
    if choice_idx < 0 or choice_idx >= len(models_list):
        print("\n❌ Invalid choice")
        exit(1)
except ValueError:
    print("\n❌ Invalid input")
    exit(1)

selected_model = models_list[choice_idx]

print(f"\n⚠️  You selected: {selected_model.name} (ID: {selected_model.id})")
confirm = input("Are you sure you want to delete this model and all its builds? (yes/no): ").strip().lower()

if confirm != "yes":
    print("\n✅ Deletion cancelled")
    exit(0)

# ============================================================================
# Delete builds first
# ============================================================================
print(f"\nDeleting builds for model '{selected_model.name}'...")

try:
    builds_response = client.list_model_builds(project_id, selected_model.id)
    # Handle different response formats
    builds = None
    if hasattr(builds_response, 'builds'):
        builds = builds_response.builds
    elif isinstance(builds_response, list):
        builds = builds_response
    elif isinstance(builds_response, dict) and 'builds' in builds_response:
        builds = builds_response['builds']

    if builds:
        for build in builds:
            build_id = build.id if hasattr(build, 'id') else build.get('id') if isinstance(build, dict) else str(build)
            try:
                print(f"  Deleting build {build_id}...", end=" ")
                client.delete_model_build(project_id, selected_model.id, build_id)
                print("✅")
            except ApiException as e:
                if "not yet implemented" in str(e).lower():
                    print("⚠️  Not supported (API limitation)")
                else:
                    print(f"❌ Error: {e}")
    else:
        print("  No builds to delete")
except Exception as e:
    print(f"  ⚠️  Could not list/delete builds: {e}")

# ============================================================================
# Delete model
# ============================================================================
print(f"\nDeleting model '{selected_model.name}'...", end=" ")

try:
    client.delete_model(project_id, selected_model.id)
    print("✅")
    print(f"\n✅ Model '{selected_model.name}' successfully deleted!")
except ApiException as e:
    if "not yet implemented" in str(e).lower():
        print("⚠️  Not supported in your environment")
        print("\n⚠️  The delete_model API is not yet implemented in your Cloudera AI instance.")
        print("   You must delete the model manually through the CML UI:")
        print(f"   1. Go to Models > {selected_model.name}")
        print("   2. Click the menu icon (⋮) and select 'Delete'")
        print(f"   3. Confirm deletion")
    else:
        print(f"❌ Error: {e}")
        exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

print("\n" + "=" * 80)
print("Cleanup complete!")
print("=" * 80)
