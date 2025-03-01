#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco / Oliver Zarate
#***************************************************************************/
from __future__ import print_function
import cmlapi
from cmlapi.rest import ApiException
from pprint import pprint
import json, secrets, os, time
import mlflow

class ModelReDeployment():
    """
    Class to manage the model redeployment of the xgboost model
    """

    def __init__(self, projectId, username):
        self.client = cmlapi.default_client()
        self.projectId = projectId
        self.username = username

    def createModelBuild(self, projectId, modelVersionId, modelCreationId, runtimeId, cpu, mem, replicas):
        """
        Method to create a Model build
        """
        # Create Model Build
        CreateModelBuildRequest = {
                                    "registered_model_version_id": modelVersionId,
                                    "runtime_identifier": runtimeId,
                                    "comment": "invoking model build",
                                    "model_id": modelCreationId,
                                    "cpu": cpu,
                                    "mem": mem,
                                    "replicas": replicas
                                  }

        try:
            # Create a model build.
            api_response = self.client.create_model_build(CreateModelBuildRequest, projectId, modelCreationId)
            #pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model_build: %s\n" % e)

        return api_response

    def createModelDeployment(self, modelBuildId, projectId, modelCreationId):
        """
        Method to deploy a model build
        """
        CreateModelDeploymentRequest = {
          "cpu" : "2",
          "memory" : "4"
        }

        try:
            # Create a model deployment.
            api_response = self.client.create_model_deployment(CreateModelDeploymentRequest, projectId, modelCreationId, modelBuildId)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model_deployment: %s\n" % e)

        return api_response

    def registerModelFromExperimentRun(self, modelName, experimentId, experimentRunId, modelPath):
        """
        Method to register a model from an Experiment Run
        """
        CreateRegisteredModelRequest = {
                                        "project_id": os.environ['CDSW_PROJECT_ID'],
                                        "experiment_id": experimentId,
                                        "run_id": experimentRunId,
                                        "model_name": modelName,
                                        "model_path": modelPath
                                       }

        try:
            # Register a model.
            api_response = self.client.create_registered_model(CreateRegisteredModelRequest)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_registered_model: %s\n" % e)

        return api_response

    def get_latest_deployment_details(self, model_name):
        """
        Given a APIv2 client object and Model Name, use APIv2 to retrieve details about the latest/current deployment.
        This function only works for models deployed within the current project.
        """
        project_id = os.environ["CDSW_PROJECT_ID"]

        # gather model details
        models = (
            self.client.list_models(project_id=project_id, async_req=True, page_size=50)
            .get()
            .to_dict()
        )
        model_info = [
            model for model in models["models"] if model["name"] == model_name
        ][0]

        model_id = model_info["id"]
        model_crn = model_info["crn"]

        # gather latest build details
        builds = (
            self.client.list_model_builds(
                project_id=project_id, model_id=model_id, async_req=True, page_size=50
            )
            .get()
            .to_dict()
        )
        build_info = builds["model_builds"][-1]  # most recent build

        build_id = build_info["id"]

        # gather latest deployment details
        deployments = (
            self.client.list_model_deployments(
                project_id=project_id,
                model_id=model_id,
                build_id=build_id,
                async_req=True,
                page_size=50
            )
            .get()
            .to_dict()
        )
        deployment_info = deployments["model_deployments"][-1]  # most recent deployment

        model_deployment_crn = deployment_info["crn"]

        return {
            "model_name": model_name,
            "model_id": model_id,
            "model_crn": model_crn,
            "latest_build_id": build_id,
            "latest_deployment_crn": model_deployment_crn,
        }

# Main script execution
USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "BNK_MLOPS_HOL_{}".format(USERNAME)
CONNECTION_NAME = os.environ["CONNECTION_NAME"]
projectId = os.environ['CDSW_PROJECT_ID']

# Set MLflow experiment name for the incremental model
experimentName = f"xgb-bank-marketing-incremental-{USERNAME}"

# Get the run with highest test accuracy
print(f"Fetching experiment: {experimentName}")
experimentId = mlflow.get_experiment_by_name(experimentName).experiment_id
runsDf = mlflow.search_runs(experimentId, run_view_type=1)

if 'metrics.test_accuracy' in runsDf.columns:
    # Sort by test accuracy in descending order and get the top run
    best_run = runsDf.sort_values('metrics.test_accuracy', ascending=False).iloc[0]
    experimentId = best_run['experiment_id']
    experimentRunId = best_run['run_id']
    test_accuracy = best_run['metrics.test_accuracy'] if 'metrics.test_accuracy' in best_run else "unknown"
    print(f"Selected run {experimentRunId} with test accuracy: {test_accuracy}")
else:
    # Fallback to the most recent run if test_accuracy metric doesn't exist
    print("Warning: No 'test_accuracy' metric found. Selecting most recent run instead.")
    experimentId = runsDf.iloc[-1]['experiment_id']
    experimentRunId = runsDf.iloc[-1]['run_id']

modelPath = "artifacts"
modelName = "bank-promo-" + USERNAME

print(f"Setting up redeployment for model: {modelName}")
deployment = ModelReDeployment(projectId, USERNAME)

try:
    # Get existing model deployment details
    print("Fetching existing model deployment details...")
    latestDeploymentDetails = deployment.get_latest_deployment_details(modelName)
    print(f"Found existing model with ID: {latestDeploymentDetails['model_id']}")
    
    # Register new model version from experiment run
    print("Registering new model version from experiment run...")
    registeredModelResponse = deployment.registerModelFromExperimentRun(modelName, experimentId, experimentRunId, modelPath)
    
    modelId = registeredModelResponse.model_id
    modelVersionId = registeredModelResponse.model_versions[0].model_version_id
    print(f"Registered new model version ID: {modelVersionId}")
    
    # Use existing model ID for redeployment
    modelCreationId = latestDeploymentDetails["model_id"]
    
    # Set resource requirements
    cpu = 2
    mem = 4
    replicas = 1
    
    # Specify runtime
    # runtimeId = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-jupyterlab-python3.10-standard:2023.05.1-b4"

    runtimeId = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-workbench-python3.9-standard:2024.10.1-b12" 
    
    # Create model build
    print("Creating new model build...")
    createModelBuildResponse = deployment.createModelBuild(projectId, modelVersionId, modelCreationId, runtimeId, cpu, mem, replicas)
    modelBuildId = createModelBuildResponse.id
    print(f"Created model build with ID: {modelBuildId}")
    
    # Deploy the new model build
    print("Deploying new model build...")
    deploymentResponse = deployment.createModelDeployment(modelBuildId, projectId, modelCreationId)
    print("Model successfully redeployed")
    
except Exception as e:
    print(f"Error in model redeployment: {e}")
