# MLOps Banking Marketing Campaign: Detailed Guide

## Introduction

This detailed guide walks you through the entire MLOps workflow implemented in this project. By following these steps, you'll learn how to build, deploy, automate, and monitor machine learning models using Cloudera Machine Learning (CML).

## Prerequisites

Before starting, ensure you have:

- Access to a Cloudera Machine Learning (CML) workspace
- Basic understanding of Python, machine learning concepts, and SQL
- User permissions to create projects, sessions, models, and jobs in CML

## Setup

### 1. Create a CML Session

1. Open the CML workspace
2. Create a new session with:
   - **Editor**: JupyterLab
   - **Kernel**: Python 3.10
   - **Edition**: Standard
   - **Version**: 2025.01
   - **Add-on**: Spark 3.3
   - **Resource Profile**: 2 vCPU / 4 GiB Memory (or larger if needed)

### 2. Install Dependencies

Once your session is running, install the required packages:

```python
!pip install -r requirements.txt
```

## Step-by-Step Workflow

### Step 1: Data Acquisition
*File: 00_download.py*

**What it does:**
This script downloads the UCI Bank Marketing dataset, which contains information about bank customers and whether they subscribed to a term deposit during a marketing campaign.

**How to run:**
1. Open `00_download.py` in your CML session
2. Click "Run" to execute the script

**Key components:**
- Downloads a zip file from UCI repository
- Extracts the CSV data
- Stores it locally in the CML session's file system

**MLOps context:**
This represents the data acquisition phase of an ML pipeline where you gather the raw data needed for modeling.

### Step 2: Data Lake Integration
*File: 01_write_to_dl.py*

**What it does:**
This script moves the downloaded dataset to a data lake using Apache Iceberg format, making it accessible for ML workflows.

**How to run:**
1. Open `01_write_to_dl.py` in your CML session
2. Update the `CONNECTION_NAME` variable as instructed by your lab leader
3. Click "Run" to execute the script

**Key components:**
- Uses CML Data Connections to establish a connection to the data lake
- Creates an Iceberg table with the banking data
- Implements proper schema management and data typing

**MLOps context:**
This represents the data storage and organization phase, where raw data is properly stored in a format suitable for analytics and model training, with version tracking capabilities.

### Step 3: Exploratory Data Analysis
*File: 02_EDA.ipynb*

**What it does:**
This notebook guides you through analyzing the banking dataset to understand patterns, distributions, and correlations.

**How to run:**
1. Open `02_EDA.ipynb` in your CML session
2. Execute the cells sequentially

**Key components:**
- Connects to the Iceberg table created in the previous step
- Performs statistical analysis of the features
- Creates visualizations to understand the data better
- Identifies potential feature engineering opportunities

**MLOps context:**
This represents the data understanding phase, where data scientists gain insights that inform model design decisions.

### Step 4: Model Training with MLflow
*File: 03_train.py*

**What it does:**
This script trains various classification models to predict whether a customer will subscribe to a term deposit, tracking all experiments with MLflow.

**How to run:**
1. Open `03_train.py` in your CML session
2. Click "Run" to execute the script

**Key components:**
- Reads data from the Iceberg table
- Performs data preprocessing and feature engineering
- Trains multiple XGBoost models with different hyperparameters
- Logs metrics, parameters, and models in MLflow

**MLOps context:**
This represents the experimentation and model development phase, where multiple approaches are tried and systematically tracked for reproducibility and comparison.

### Step 5: Model Selection and Deployment
*File: 04_api_deployment.py*

**What it does:**
This script selects the best model based on test accuracy, registers it to the Model Registry, and deploys it as a REST API endpoint.

**How to run:**
1. Open `04_api_deployment.py` in your CML session
2. Click "Run" to execute the script

**Key components:**
- Queries the MLflow experiment to find the best-performing model
- Registers the model in the CML Model Registry
- Creates a model deployment with REST API endpoints
- Sets up monitoring for the deployed model

**MLOps context:**
This represents the deployment phase, making the model accessible to other applications and systems through standardized interfaces.

### Step 6: Set Up Automated Model Lifecycle (Steps 6-8)

The next three scripts work together to demonstrate an automated model update workflow. For each script, we'll create a CML Job and configure dependencies between them.

#### Step 6.1: Create New Batch Data
*File: 05_newbatch.py*

**What it does:**
This script generates new synthetic banking data and stores it in the data lake, simulating the arrival of new customer interactions.

**How to run as a job:**
1. Navigate to "Jobs" in the left sidebar of your project
2. Click "New Job"
3. Set the following parameters:
   ```
   Name: New Batch 
   Script: 05_newbatch.py
   Editor: Workbench
   Kernel: Python 3.10
   Spark Add On: Spark 3.3
   Edition: Standard
   Version: 2025.01
   Schedule: Manual
   Resource Profile: 2 vCPU / 4 GiB / 0 GPU
   ```
4. Click "Create Job"

**Key components:**
- Creates synthetic banking transactions data
- Appends new data to the existing Iceberg table
- Simulates real-world data accumulation over time

**MLOps context:**
This represents ongoing data collection in a production system, where new data continues to arrive after a model is deployed.

#### Step 6.2: Model Retraining
*File: 06_retrain.py*

**What it does:**
This script retrains models using both historical and new data, creating a new MLflow experiment.

**How to run as a job:**
1. Navigate to "Jobs" in the left sidebar
2. Click "New Job"
3. Set the following parameters:
   ```
   Name: Retrain Models
   Script: 06_retrain.py
   Editor: Workbench
   Kernel: Python 3.10
   Spark Add On: Spark 3.3
   Edition: Standard
   Version: 2025.01
   Schedule: Dependent on New Batch
   Resource Profile: 2 vCPU / 4 GiB / 0 GPU
   ```
4. Click "Create Job"

**Key components:**
- Loads the expanded dataset with both original and new data
- Retrains models with updated parameters
- Logs new experiment results in MLflow

**MLOps context:**
This represents the model maintenance cycle, where models are periodically retrained on new data to maintain performance.

#### Step 6.3: Model Redeployment
*File: 07_api_redeployment.py*

**What it does:**
This script selects the best model from the retraining experiments and redeploys it to replace the previous model version.

**How to run as a job:**
1. Navigate to "Jobs" in the left sidebar
2. Click "New Job"
3. Set the following parameters:
   ```
   Name: Redeploy Model
   Script: 07_api_redeployment.py
   Editor: Workbench
   Kernel: Python 3.10
   Spark Add On: Spark 3.3
   Edition: Standard
   Version: 2025.01
   Schedule: Dependent on Retrain Models
   Resource Profile: 2 vCPU / 4 GiB / 0 GPU
   ```
4. Click "Create Job"

**Key components:**
- Selects the best-performing model from retraining
- Updates the model in the registry
- Creates a new deployment that replaces the previous version
- Ensures seamless transition for users of the model API

**MLOps context:**
This represents model versioning and updating in production, a critical part of model lifecycle management.

### Step 7: Model Monitoring Setup
*File: 08_model_simulation.py*

**What it does:**
This script simulates client applications making calls to the deployed model endpoint and logs predictions.

**How to run:**
1. Open `08_model_simulation.py` in your CML session
2. Click "Run" to execute the script

**Key components:**
- Generates synthetic requests to the model API
- Records predictions and simulated "ground truth" outcomes
- Creates a foundation for monitoring model performance

**MLOps context:**
This represents the model serving and logging infrastructure needed to monitor model performance in production.

### Step 8: Performance Tracking Dashboard
*File: 09_model_monitoring_dashboard.ipynb*

**What it does:**
This notebook creates a dashboard to monitor model performance metrics over time.

**How to run:**
1. Open `09_model_monitoring_dashboard.ipynb` in your CML session
2. Execute the cells sequentially

**Key components:**
- Retrieves prediction logs and ground truth
- Calculates key performance metrics
- Visualizes performance trends over time
- Creates a dashboard for stakeholders

**MLOps context:**
This represents the monitoring and alerting capabilities needed to ensure models continue to perform well in production.

## The Complete MLOps Workflow

By completing all these steps, you've implemented a comprehensive MLOps workflow that includes:

1. **Data management** - Collection, storage, and versioning
2. **Experimentation** - Multiple model iterations with tracked results
3. **Deployment** - Productionizing models as APIs
4. **Automation** - Scheduled jobs for data processing and model updates
5. **Monitoring** - Tracking model performance in production

This workflow embodies the Cloudera approach to MLOps, using a unified platform with enterprise data controls, streamlined operations, scalable infrastructure, comprehensive governance, and iterative model management.

## Additional Considerations

### Model Selection Criteria

In this lab, we select models based on test accuracy for simplicity. In real-world scenarios, model selection would likely involve:

- Multiple metrics (precision, recall, F1-score)
- Business-specific KPIs
- Fairness and bias considerations
- Model explainability requirements
- Inference speed and resource constraints

### Future Enhancements

The current workflow could be enhanced with:

- Concept and data drift detection
- Feature stores for better feature management
- A/B testing for gradual model rollout
- Model explainability tools
- Trigger-based retraining (rather than scheduled)
- More sophisticated monitoring alerts

## Additional Resources

For more information on the tools and technologies used in this lab, consult:

- [Cloudera Machine Learning Documentation](https://docs.cloudera.com/machine-learning/cloud/index.html)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Apache Iceberg](https://iceberg.apache.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
