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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import cml.data_v1 as cmldata
import datetime

# SET USER VARIABLES
USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "BNK_MLOPS_HOL_"+USERNAME
CONNECTION_NAME = os.environ["CONNECTION_NAME"]
EXPERIMENT_NAME = "xgb-bank-marketing-{0}".format(USERNAME)

# CONNECT TO DATA
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# LOAD CURRENT DATA
current_data = spark.sql("SELECT * FROM {}.BANK_MARKETING_{}".format(DBNAME, USERNAME)).toPandas()

# LOAD MODEL PERFORMANCE DATA FROM MLFLOW
def get_mlflow_model_metrics():
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    if experiment is None:
        return pd.DataFrame()
    
    runs = mlflow.search_runs(experiment.experiment_id)
    
    # Extract metrics of interest
    metrics_df = runs[['start_time', 'metrics.accuracy', 'metrics.recall']].copy()
    metrics_df['start_time'] = pd.to_datetime(metrics_df['start_time']).dt.strftime('%Y-%m-%d')
    metrics_df = metrics_df.rename(columns={
        'metrics.accuracy': 'accuracy',
        'metrics.recall': 'recall'
    })
    
    return metrics_df

model_metrics = get_mlflow_model_metrics()

# ANALYZE DATA DISTRIBUTION
def analyze_data_distribution(df):
    # Check target variable distribution
    target_dist = df['y'].value_counts(normalize=True).reset_index()
    target_dist.columns = ['Response', 'Percentage']
    
    # Check numerical features statistics
    numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    num_stats = df[numerical_cols].describe().T
    
    # Check categorical features distribution
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    cat_dists = {}
    for col in categorical_cols:
        cat_dists[col] = df[col].value_counts().reset_index()
        cat_dists[col].columns = [col, 'Count']
    
    return {
        'target_distribution': target_dist,
        'numerical_stats': num_stats,
        'categorical_distributions': cat_dists
    }

data_analysis = analyze_data_distribution(current_data)

# CREATE MONITORING DASHBOARD
def generate_monitoring_report():
    report = {}
    
    # 1. Model Performance Over Time
    if not model_metrics.empty:
        report['model_performance'] = model_metrics
    
    # 2. Data Quality Metrics
    report['data_quality'] = {
        'missing_values': current_data.isnull().sum().sum(),
        'row_count': len(current_data),
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 3. Feature Distributions
    report['data_analysis'] = data_analysis
    
    return report

# GENERATE AND SAVE REPORT
monitoring_report = generate_monitoring_report()

# PRINT SUMMARY
print("\nModel Monitoring Report Generated:")
print(f"Timestamp: {monitoring_report['data_quality']['timestamp']}")
print(f"Data Size: {monitoring_report['data_quality']['row_count']} records")
print(f"Missing Values: {monitoring_report['data_quality']['missing_values']}")

if not model_metrics.empty:
    latest_metrics = model_metrics.iloc[-1]
    print(f"\nLatest Model Performance:")
    print(f"Date: {latest_metrics['start_time']}")
    print(f"Accuracy: {latest_metrics['accuracy']:.4f}")
    print(f"Recall: {latest_metrics['recall']:.4f}")

# Visualize target distribution
target_dist = monitoring_report['data_analysis']['target_distribution']
print(f"\nTarget Distribution:")
print(target_dist)
