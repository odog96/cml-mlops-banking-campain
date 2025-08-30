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
#  Absent a written agreement with Cloudera, Inc. ("Cloudera") to the
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
# #  Author(s): Paul de Fusco
#***************************************************************************/

import os
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
import cdsw
from flask import Flask, request, jsonify
import json

# SET USER VARIABLES
USERNAME = os.environ["PROJECT_OWNER"]
EXPERIMENT_NAME = "xgb-bank-marketing-{0}".format(USERNAME)

# GET LATEST MODEL FROM MLFLOW
def getLatestExperimentInfo(experimentName):
    experimentId = mlflow.get_experiment_by_name(experimentName).experiment_id
    runsDf = mlflow.search_runs(experimentId, run_view_type=1)
    experimentId = runsDf.iloc[-1]['experiment_id']
    experimentRunId = runsDf.iloc[-1]['run_id']
    return experimentId, experimentRunId

experimentId, experimentRunId = getLatestExperimentInfo(EXPERIMENT_NAME)

# LOAD MODEL FROM MLFLOW
model_uri = f"runs:/{experimentRunId}/model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# DEFINE API ENDPOINT
@cdsw.endpoint
def predict(args):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame(args, index=[0])
        
        # Make prediction
        prediction = loaded_model.predict(input_data)[0]
        probability = loaded_model.predict_proba(input_data)[0][1]
        
        # Return result
        result = {
            "prediction": int(prediction),
            "probability": float(probability),
            "label": "yes" if prediction == 1 else "no"
        }
        
        return result
    
    except Exception as e:
        return {"error": str(e)}

# Example input for testing
test_data = {
    "age": 41,
    "job": "entrepreneur",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 2578,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "duration": 301,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}

# Test the API locally
print("Testing model API with sample data:")
print(json.dumps(test_data, indent=2))
print("\nPrediction result:")
print(json.dumps(predict(test_data), indent=2))
