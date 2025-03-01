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
# #  Author(s): Paul de Fusco / Oliver Zarate
#***************************************************************************/

import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import mlflow.sklearn
from xgboost import XGBClassifier
from datetime import datetime
import cml.data_v1 as cmldata
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer

# Setup
USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "BNK_MLOPS_HOL_{}".format(USERNAME)
CONNECTION_NAME = os.environ["CONNECTION_NAME"]
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = f"xgb-bank-marketing-incremental-{USERNAME}"

mlflow.set_experiment(EXPERIMENT_NAME)

# Connect to Spark
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# Read Iceberg metadata to get the latest snapshot and its parent
print("Reading Iceberg table metadata...")
table_path = f"{DBNAME}.BANK_MARKETING_{USERNAME}"
snapshots_df = spark.read.format("iceberg").load(f"{table_path}.snapshots")
snapshots_df = snapshots_df.orderBy("committed_at", ascending=False)

# Get the latest snapshot_id and parent_id
latest_snapshot = snapshots_df.first()
snapshot_id = latest_snapshot['snapshot_id']
parent_id = latest_snapshot['parent_id']
committed_at = latest_snapshot['committed_at'].strftime('%Y-%m-%d %H:%M:%S')
print(f"Latest snapshot ID: {snapshot_id}, Parent ID: {parent_id}, Committed at: {committed_at}")

# Load only the incremental data (difference between latest snapshot and parent)
print("Loading incremental data...")
df_spark = spark.read \
    .format("iceberg") \
    .option("start-snapshot-id", parent_id) \
    .option("end-snapshot-id", snapshot_id) \
    .load(table_path)

incremental_count = df_spark.count()
print(f"Incremental data count: {incremental_count} rows")

# Define features
numerical_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
categorical_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

# Cast numerical columns in Spark
for col_name in numerical_cols:
    df_spark = df_spark.withColumn(col_name, col(col_name).cast("double"))

# Index categorical columns and target
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx") for col in categorical_cols]
indexers.append(StringIndexer(inputCol="y", outputCol="y_binary"))
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=indexers)
df_indexed = pipeline.fit(df_spark).transform(df_spark)

# Select features (top ones from your EDA or all for simplicity)
features = numerical_cols + [f"{col}_idx" for col in categorical_cols]
df_selected = df_indexed.select(features + ["y_binary"])

# Convert to Pandas
df = df_selected.toPandas()
X = df[features]
y = df["y_binary"]

# Set MLflow tags for this training run
tags = {
    "iceberg_snapshot_id": str(snapshot_id),
    "iceberg_parent_id": str(parent_id),
    "iceberg_committed_at": committed_at,
    "incremental_row_count": incremental_count
}

# Split the data
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

# Define hyperparameter combinations (reduced for incremental training)
max_depths = [3, 5, 7]
learning_rates = [0.01, 0.1]
hyperparams = [(md, lr) for md in max_depths for lr in learning_rates]

# Train and evaluate models
for max_depth, learning_rate in hyperparams:
    with mlflow.start_run() as run:
        # Add tags to the run
        for key, value in tags.items():
            mlflow.set_tag(key, value)
        
        model = XGBClassifier(
            use_label_encoder=False,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=100,
            eval_metric="logloss",
            early_stopping_rounds=10,
            random_state=42
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]
        
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_prob)
        test_roc_auc = roc_auc_score(y_test, y_test_prob)
        
        print(f"\nRun: max_depth={max_depth}, learning_rate={learning_rate}")
        print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Train Recall: {train_recall:.4f}, Test Recall: {test_recall:.4f}")
        print(f"Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")
        print(f"Train ROC AUC: {train_roc_auc:.4f}, Test ROC AUC: {test_roc_auc:.4f}")
        
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("n_estimators", model.n_estimators)
        
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("train_roc_auc", train_roc_auc)
        mlflow.log_metric("test_roc_auc", test_roc_auc)
        
        mlflow.xgboost.log_model(model, artifact_path="artifacts")

# Get latest experiment info
def getLatestExperimentInfo(experimentName):
    experimentId = mlflow.get_experiment_by_name(experimentName).experiment_id
    runsDf = mlflow.search_runs(experimentId, run_view_type=1)
    experimentId = runsDf.iloc[-1]['experiment_id']
    experimentRunId = runsDf.iloc[-1]['run_id']
    return experimentId, experimentRunId

experimentId, experimentRunId = getLatestExperimentInfo(EXPERIMENT_NAME)
run = mlflow.get_run(experimentRunId)

print("\nLatest Experiment Parameters:")
print(pd.DataFrame(data=[run.data.params], index=["Value"]).T)
print("\nLatest Experiment Metrics:")
print(pd.DataFrame(data=[run.data.metrics], index=["Value"]).T)

client = mlflow.tracking.MlflowClient()
artifacts = client.list_artifacts(run_id=run.info.run_id)
print("\nArtifacts:")
for artifact in artifacts:
    print(artifact.path)

spark.stop()