# Module 1: Data Science Fundamentals

## Overview
This module introduces the core data science workflow on Cloudera AI, focusing on the fundamentals without the complexity of distributed computing frameworks. You'll work through a complete ML pipeline from data ingestion to model deployment.

You're a data scientist at a Portuguese bank, and your manager just dropped a challenge on your desk: the marketing team is burning through budget on phone campaigns with an 11% success rate. They're calling everyone, and it's not working. Your job? Build a machine learning model that predicts which customers are likely to subscribe to a term deposit before the call is made.
The stakes are real. If your model works, the bank saves money by targeting the right customers. If it doesn't, you'll be explaining to executives why AI failed them. You have 41,188 historical records from past campaigns between 2008-2010. Each record tells a story: customer demographics, their relationship with the bank, past campaign responses, and whether they said "yes" or "no." Your first task is straightforward: explore this data, engineer features that matter, build a model, and deploy it. But here's the catch - this isn't a Kaggle competition. This model needs to work in production, which means you'll need to think beyond accuracy scores.
In future modules, we'll tackle the hard parts of production ML: What happens when customer behavior changes? How do we detect when the model starts failing? When do we retrain? Welcome to the real world of MLOps - where building the model is just the beginning.

**Duration:** ~45-60 minutes

## Learning Objectives
By the end of this module, you will:
- Understand how to ingest data into the data lake
- Perform exploratory data analysis using Pandas
- Apply feature engineering techniques based on domain insights
- Track experiments systematically with MLflow
- Compare model variants to measure improvement
- Deploy models as REST APIs

## Prerequisites
- Cloudera AI workspace access
- Python 3.10 runtime with Standard edition
- Basic Python and ML knowledge

## Setup

### 1. Create CML Session
```
Editor: JupyterLab
Kernel: Python 3.10
Edition: Standard
Version: 2025.01
Add-on: Spark 3.3 (for data lake writes only)
Resource Profile: 2 vCPU / 4 GiB Memory
```

### 2. Install Dependencies
```bash
pip install scikit-learn xgboost mlflow pandas numpy matplotlib seaborn scipy
```

### 3. Set Environment Variables (if needed)
```bash
export DATA_LAKE_NAME="your-datalake-connection"
```

## Module Structure

### Step 1: Data Ingestion (`01_ingest_data.py`)
**What it does:**
- Downloads UCI Bank Marketing dataset
- Performs initial data inspection
- Writes data to Iceberg table in the data lake

**Key concepts:**
- Direct data ingestion patterns
- Using Spark for write operations (leveraging Iceberg benefits)
- Data validation and sanity checks

**Run:**
```bash
python 01_ingest_data.py
```

**Expected output:**
- CSV file downloaded to `data/` directory
- Data written to `default.bank_marketing` table
- Summary statistics displayed

---

### Step 2: EDA & Feature Engineering (`02_eda_feature_engineering.ipynb`)
**What it does:**
- Reads data from lake using Pandas (no Spark for analysis)
- Explores distributions, correlations, and patterns
- Engineers a new feature: **Customer Engagement Score**
- Validates the feature's predictive power

**Key concepts:**
- Data scientists prefer Pandas for interactive analysis
- Feature engineering combines domain knowledge with data insights
- Statistical validation of new features

**The Engagement Score:**
Combines multiple signals into a single metric:
- Current campaign effort (30%)
- Historical responsiveness (30%)
- Contact recency (20%)
- Overall contact history (20%)

**Run:**
Open the notebook in JupyterLab and execute cells sequentially.

**Expected insights:**
- Dataset is imbalanced (~11% positive class)
- Duration is strongest single predictor
- High engagement customers convert 3-4x more than low engagement
- New feature is statistically significant (p < 0.001)

---

### Step 3: Model Training (`03_train_with_mlflow.py`)
**What it does:**
- Trains multiple XGBoost model variants
- Compares baseline vs. engineered features
- Performs hyperparameter tuning
- Tracks everything in MLflow

**Experiments conducted:**
1. **Baseline:** Without engagement score
2. **Engineered:** With engagement score
3. **Tuned:** Multiple hyperparameter variants

**Key concepts:**
- MLflow tracks all experiments automatically
- Systematic comparison reveals feature impact
- Best practices for experiment organization

**Run:**
```bash
python 03_train_with_mlflow.py
```

**View results:**
```bash
mlflow ui
# Open http://localhost:5000
```

**Expected results:**
- 6+ experiment runs logged
- Measurable improvement from feature engineering
- Best model identified by ROC-AUC

---

### Step 4: Model Deployment (`04_deploy_model.py`)
**What it does:**
- Queries MLflow for best model
- Registers model in registry
- Creates API serving code
- (Optionally) Deploys to CML

**Key concepts:**
- Automated model selection
- Model versioning and lineage
- Production deployment patterns

**Run:**
```bash
python 04_deploy_model.py
```

**Manual deployment:**
If automatic deployment fails, use CML UI:
1. Go to Models → New Model
2. Name: `banking-campaign-predictor`
3. File: `module1/model_api.py`
4. Function: `predict`
5. Deploy with 1 CPU / 2 GB RAM

**Test deployment:**
```bash
python test_deployment.py
# Update URL and API key first
```

---

## Key Takeaways

### Feature Engineering Impact
The engagement score provides measurable improvement:
- Captures customer interaction patterns holistically
- More interpretable than raw features
- Shows 3-4x conversion difference across quintiles

### MLflow Benefits
- Complete experiment history
- Easy comparison across runs
- Reproducibility built-in
- Smooth transition to production

### Deployment Patterns
- Models move from experiment to production seamlessly
- API provides standardized interface
- Version control enables rollback

## Common Issues & Solutions

**Issue:** Can't connect to data lake
- **Solution:** Check `DATA_LAKE_NAME` environment variable
- Verify data connection exists in CML

**Issue:** MLflow experiments not showing
- **Solution:** Check `mlflow.db` file exists
- Run `mlflow ui` from project root directory

**Issue:** Model deployment fails
- **Solution:** Deploy manually through CML UI
- Verify runtime and resource settings

**Issue:** Import errors
- **Solution:** Run `pip install` commands
- Ensure you're in correct Python environment

## Next Steps

Once you've completed Module 1:
1. Review your MLflow experiments
2. Compare model performance metrics
3. Test your deployed API
4. Proceed to **Module 2: MLOps Lifecycle**

## Files Summary
```
module1/
├── 01_ingest_data.py              # Data download and ingestion
├── 02_eda_feature_engineering.ipynb  # EDA and feature creation
├── 03_train_with_mlflow.py        # Model training with tracking
├── 04_deploy_model.py             # Model deployment
├── utils.py                       # Helper functions
├── model_api.py                   # Generated API code
└── test_deployment.py             # Generated test script
```

## Questions for Discussion

1. Why did we use Spark for writes but Pandas for analysis?
2. What other features could we engineer from this data?
3. How did the engagement score improve model performance?
4. What metrics matter most for this business problem?
5. How would you explain the model to non-technical stakeholders?
