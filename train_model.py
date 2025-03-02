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

import os, warnings, sys
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import mlflow.sklearn
from xgboost import XGBClassifier
import cml.data_v1 as cmldata

# SET USER VARIABLES
USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "BNK_MLOPS_HOL_"+USERNAME
CONNECTION_NAME = os.environ["CONNECTION_NAME"]

# SET MLFLOW EXPERIMENT NAME
EXPERIMENT_NAME = "xgb-bank-marketing-{0}".format(USERNAME)
mlflow.set_experiment(EXPERIMENT_NAME)

# CREATE SPARK SESSION WITH DATA CONNECTIONS
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# READ BANK MARKETING TABLE
df_spark = spark.sql("SELECT * FROM {}.BANK_MARKETING_{}".format(DBNAME, USERNAME))

# CONVERT TO PANDAS FOR MODELING
df = df_spark.toPandas()

# ENCODE TARGET COLUMN
df['y_encoded'] = df['y'].map({'no': 0, 'yes': 1})

# SELECT FEATURES
features = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
X = pd.get_dummies(df[features])
y = df['y_encoded']

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# MLFLOW EXPERIMENT RUN
with mlflow.start_run():
    # Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Recall: %.2f%%" % (recall * 100.0))
    
    # Log parameters and metrics
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("feature_count", X.shape[1])
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("recall", recall)
    
    # Log model
    mlflow.xgboost.log_model(model, artifact_path="model")

# GET LATEST EXPERIMENT INFO
def getLatestExperimentInfo(experimentName):
    experimentId = mlflow.get_experiment_by_name(experimentName).experiment_id
    runsDf = mlflow.search_runs(experimentId, run_view_type=1)
    experimentId = runsDf.iloc[-1]['experiment_id']
    experimentRunId = runsDf.iloc[-1]['run_id']
    return experimentId, experimentRunId

experimentId, experimentRunId = getLatestExperimentInfo(EXPERIMENT_NAME)
run = mlflow.get_run(experimentRunId)

# DISPLAY RUN INFO
print("\nParameters:")
print(pd.DataFrame(data=[run.data.params], index=["Value"]).T)
print("\nMetrics:")
print(pd.DataFrame(data=[run.data.metrics], index=["Value"]).T)

# LIST ARTIFACTS
client = mlflow.tracking.MlflowClient()
for artifact in client.list_artifacts(run_id=run.info.run_id):
    print(artifact.path)
