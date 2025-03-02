# CML MLOps Banking Marketing Campaign

## Overview

This repository contains a hands-on lab that demonstrates a simplified MLOps (Machine Learning Operations) workflow on Cloudera Machine Learning (CML). The lab uses a banking marketing campaign dataset to predict customer responses, guiding participants through the entire ML lifecycle from data ingestion to model monitoring.

## MLOps on Cloudera

MLOps is a set of practices that combines Machine Learning, DevOps, and Data Engineering to reliably and efficiently deploy and maintain ML models in production. The Cloudera approach to MLOps incorporates:

- **Unified Platform**: Cloudera Machine Learning provides a single platform that handles the entire ML lifecycle, from data exploration to production deployment, eliminating the need for disjointed tools.

- **Enterprise Data Control**: Leveraging Cloudera's SDX (Shared Data Experience), models have secure access to data across any cloud or data center through a single control plane.

- **Streamlined Operations**: CML automates the deployment and management of models, ensuring consistent practices across teams and reducing time-to-production.

- **Scalable Infrastructure**: Built on Kubernetes, CML dynamically provisions resources for both development and production needs, efficiently managing compute resources.

- **Comprehensive Governance**: Tracks model lineage, versions, and performance metrics, providing auditability and compliance throughout the model lifecycle.

- **Iterative Model Management**: Provides capabilities for model monitoring, retraining, and redeployment to address model drift and maintain predictive performance.

## Getting Started

Before working with the lab files (00_download.py through 09_model_monitoring_dashboard.ipynb), you need to set up a CML session with the following specifications:

1. Open the CML workspace
2. Create a new session with:
   - Resource Profile: 2 vCPU / 4 GiB Memory (or larger if needed)
   - Runtime: Python 3.10
   - Edition: Standard
   - Add-on: Spark 3.3

Once the session is running, you can begin working through the lab files sequentially.

## Step-by-Step Guide

### 00_download.py: Data Ingestion
This script downloads the bank marketing dataset from a public UCI repository and stores it locally on the CML session storage. This is the initial step in the data pipeline where we source our raw data. Within the ML process, this represents the data acquisition phase, where we collect the necessary information that will be used for model training and evaluation.

### 01_write_to_dl.py: Data Lake Integration
In this step, we establish a connection to the data lake and write the downloaded dataset to an Iceberg table. Iceberg is a high-performance format for huge analytic tables that provides ACID transactions, schema evolution, and efficient querying. This represents the data storage and organization phase of the ML lifecycle, creating a reliable, versioned data source for our analytics pipeline.

### 02_EDA.ipynb: Exploratory Data Analysis
This notebook guides you through exploratory data analysis of the banking dataset. You'll examine the distribution of features, identify patterns and correlations, and gain insights into customer behavior. This crucial step helps in understanding the data characteristics, identifying potential issues, and informing feature engineering decisions. In the ML lifecycle, this is the data understanding phase that drives better model design decisions.

### 03_train.py: Model Training with MLflow
This script trains classification models on the banking data using different algorithms and hyperparameters, tracking all experiments with MLflow. MLflow is an open-source platform for the machine learning lifecycle that handles experiment tracking, packaging ML code, and model sharing/deployment. The script creates multiple iterations to build an experiment with various modeling approaches. In the ML lifecycle, this represents the model development and experimentation phase.

### 04_api_deployment.py: Model Selection and Deployment
This script selects the best-performing model from the MLflow experiments based on test accuracy, registers it to the Model Registry, and deploys it as a REST API endpoint. Note that while we're using test accuracy as the selection criterion for simplicity, real-world deployments often involve more sophisticated metrics and business considerations. This step represents the model deployment phase of the ML lifecycle, making the model available for consumption by applications.

### 05_newbatch.py, 06_retrain.py, and 07_api_redeployment.py: Automated Model Lifecycle
These scripts work together to demonstrate an automated model update workflow:

1. **05_newbatch.py**: Generates new synthetic banking data and stores it in the data lake, simulating new customer interactions.
2. **06_retrain.py**: Using both historical and new data, retrains the model to incorporate new patterns and improve accuracy.
3. **07_api_redeployment.py**: Selects the best model from the retraining experiments and redeploys it to replace the previous model version.

To set up this workflow:
- Create a CML Job for each script
- Configure dependencies between jobs so when the first job (05_newbatch.py) is triggered, the others automatically follow
- Schedule the first job to run periodically or trigger it manually

This sequence demonstrates the model lifecycle management phase, showing how models evolve and improve as new data becomes available.

### 08_model_simulation.py: Model Monitoring Setup
This script simulates client applications making calls to the deployed model endpoint. It generates synthetic requests to the model, records predictions, and logs "ground truth" outcomes to a PostgreSQL database. This establishes the foundation for monitoring model performance in production by generating the necessary data to track how the model behaves with new inputs. This represents the model serving and logging phase of the ML lifecycle.

### 09_model_monitoring_dashboard.ipynb: Performance Tracking
The final notebook creates a dashboard to monitor model performance metrics like accuracy, precision, and recall over time. It retrieves the logged prediction data and ground truth labels, calculates performance metrics, and visualizes trends. This allows data scientists and stakeholders to identify potential model drift or performance degradation, enabling proactive maintenance. This represents the model monitoring and maintenance phase of the ML lifecycle.

## Additional Resources

- [Cloudera Machine Learning Documentation](https://docs.cloudera.com/machine-learning/cloud/index.html)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Apache Iceberg](https://iceberg.apache.org/)

## Future Enhancements

This lab currently demonstrates a simplified MLOps workflow. Future versions will incorporate:
- Data and concept drift detection
- Advanced model monitoring techniques
- A/B testing for model deployment
- Automated retraining triggers based on model performance
