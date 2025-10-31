import numpy as np
import pandas as pd
import json, requests
import time
import cdsw
import cmlapi
import os


if os.environ.get("MODEL_NAME") == "":
    os.environ["MODEL_NAME"] = "lstm"
if os.environ.get("PROJECT_NAME") == "":
    os.environ["PROJECT_NAME"] = "SDWAN"
    
    
# model_name = os.environ["MODEL_NAME"]
# project_name = os.environ["PROJECT_NAME"]


model_name = "LSTM-2"
#os.environ["MODEL_NAME"]
project_name = "SDWAN"

#################################################################
# CML model metrics
client = cmlapi.default_client(url=os.getenv("CDSW_API_URL").replace("/api/v1", ""), cml_api_key=os.getenv("CDSW_APIV2_KEY"))
target_model = client.list_all_models(search_filter=json.dumps({"name": model_name}))
proj_id = client.list_projects(search_filter=json.dumps({"name":project_name })).projects[0].id
mod_id = target_model.models[0].id
build_list = client.list_model_builds(project_id = proj_id, model_id = mod_id,sort='created_at')
build_id = build_list.model_builds[-1].id
model_deployments = client.list_model_deployments(project_id=proj_id,model_id=mod_id,build_id=build_id)
cr_number = model_deployments.model_deployments[0].crn
#################################################################

# read metrics
metrics = cdsw.read_metrics(model_deployment_crn=cr_number,dev=False)

# metrics_df = pd.json_normalize(metrics["metrics"])

performance_metrics = []
core_metrics = []
for item in metrics['metrics']:
    if len(item['metrics'].keys()) < 5:
         performance_metrics.append(item['metrics'])
    else:
        core_metrics.append(item['metrics'])

def check_last_rmse(values):
    if len(values) < 2:
        raise ValueError("The list must contain at least two values.")
    
    rmse_values = [item['rmse'] for item in values]
    last_value = rmse_values[-1]
    prior_values = rmse_values[:-1]
    average_prior = sum(prior_values) / len(prior_values)
    
    if rmse_values[-1] < 5:
        return True
    else:
        return False

job_name = 'retrain_model'
target_job = client.list_jobs(proj_id, search_filter=json.dumps({"name": job_name}))

# todo 

if check_last_rmse(performance_metrics):
    # kick of retrain job
    job_run = client.create_job_run(
            cmlapi.CreateJobRunRequest(),
            project_id = proj_id, 
            job_id = target_job.jobs[0].id
            )