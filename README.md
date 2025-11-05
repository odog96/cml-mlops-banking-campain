# 🏦 Bank Marketing ML Lab - Complete MLOps Journey

## Welcome!

This hands-on lab teaches you the **complete machine learning operations (MLOps) lifecycle** using a real-world bank marketing prediction scenario. You'll build, deploy, monitor, and automate the retraining of a production ML model.

---

## 📚 Lab Structure: 3 Modules

### **Module 1: Complete ML Workflow** 🚀
**Duration:** 45-90 minutes

[📖 View Module 1](./module1/README.md)

Learn to build an end-to-end ML pipeline from raw data to production inference:
- **Data Ingestion** → Load bank marketing data into the data lake
- **Exploratory Analysis** → Understand patterns and trends
- **Model Training** → Build and compare multiple models with MLflow
- **Model Deployment** → Deploy your best model as a REST API
- **Inference Pipeline** → Generate predictions on new customers

**What you'll accomplish:** A fully trained, deployed model ready to make predictions.

---

### **Module 2: Proactive Monitoring** 📊
**Duration:** 30-45 minutes
**Prerequisites:** Complete Module 1

[📖 View Module 2](./module2/README.md)

Learn to detect when models degrade in production:
- **Track predictions** over time as new data arrives
- **Monitor accuracy** across time periods
- **Detect degradation** automatically when performance drops
- **Alert** when intervention is needed

**What you'll accomplish:** A monitoring pipeline that catches model problems early.

---

### **Module 3: Automated Retraining Loop** 🔄
**Duration:** 60-90 minutes
**Prerequisites:** Complete Modules 1 & 2

[📖 View Module 3](./module3/README.md)

Learn to automate the entire MLOps lifecycle:
- **Detect drift** in production data using Evidently AI
- **Acquire labels** for drifted data
- **Retrain model** on combined historical + new data
- **Deploy automatically** when performance improves

**What you'll accomplish:** A fully automated ML pipeline that improves itself over time.

---

## 📊 Prerequisites: Understanding ML Evaluation Metrics

Before you start, familiarize yourself with the metrics you'll be tracking throughout all three modules:

[📖 View Presentation: ML Evaluation Metrics](./assets/ML_Evaluation_Metrics_Final.pdf)

**Key metrics you'll encounter:**
- **F1 Score** - Balance between precision and recall (important for imbalanced datasets)
- **ROC-AUC** - Model's ability to distinguish between classes at various thresholds
- **Accuracy** - Overall correctness
- **Precision** - Of predicted positives, how many were actually correct
- **Recall** - Of actual positives, how many did we catch

---

## 🎯 Learning Outcomes

By completing all three modules, you'll understand:

✅ **How production ML systems work**
- Complete data-to-predictions pipeline
- Real-world constraints and decisions

✅ **ML Lifecycle Management**
- Experimentation with MLflow
- Model registry and versioning
- Deployment strategies

✅ **Production Monitoring**
- Tracking model performance over time
- Detecting accuracy degradation
- Automated alerting

✅ **Automated MLOps**
- Trigger-based retraining
- Drift detection
- End-to-end automation

✅ **Industry Best Practices**
- Version control for models
- Reproducible pipelines
- Graceful error handling

---

## 🏃 Quick Start

### Step 1: Understand the Business Context
You're building a model that predicts whether a bank customer will subscribe to a term deposit. This is a **binary classification problem** with real business value.

### Step 2: Review Metrics (5 min)
Open and review the [ML Evaluation Metrics presentation](./assets/ML_Evaluation_Metrics_Final.pdf) - you'll reference it throughout the lab.

### Step 3: Start with Module 1
Follow the step-by-step guide in [Module 1 README](./module1/README.md). This is your foundation for everything that follows.

### Step 4: Progress to Module 2 & 3
Once Module 1 is complete, you'll have the data and trained models needed for Modules 2 and 3.

---

## 📋 Lab Environment Requirements

Before you start, ensure you have:
- Access to a Cloudera AI workspace (CML project)
- Python 3.10+ runtime available
- ~500MB disk space for data and models
- Terminal/command line access
- Basic familiarity with Python (helpful but not required)

---

## 🗂️ Project Structure

```
/home/cdsw/
├── README_PROJECT_OVERVIEW.md          ← You are here
├── assets/
│   └── ML_Evaluation_Metrics_Final.pdf ← Essential reading!
├── module1/                             ← START HERE
│   ├── README.md
│   ├── 01_ingest.py
│   ├── 02_eda_notebook.ipynb
│   ├── 03_train_quick.py
│   ├── 04_deploy.py
│   ├── 05.1_inference_data_prep.py
│   ├── 05.2_inference_predict.py
│   ├── 06_Inference_101.ipynb
│   └── helpers/                        ← Shared utilities
├── module2/                             ← AFTER Module 1
│   ├── README.md
│   ├── 02_prepare_artificial_data.py
│   └── 03_monitoring_pipeline.py
└── module3/                             ← AFTER Module 1 & 2
    ├── README.md
    ├── 1_check_drift.py
    ├── 2_simulate_labeling_job.py
    ├── 3_retrain_model.py
    └── 4_register_and_deploy.py
```

---

## 📌 Important Notes

### Execution Locations
- **Module 1 & 3 scripts:** Run from the `module1/` or `module3/` directory
- **Module 2 scripts:** Run from the **PROJECT ROOT** (`/home/cdsw/`) because they reference Module 1 data

### File Naming
Note the **period in filenames**: `05.1_inference_data_prep.py` (not `05_1_`)
```bash
✓ Correct:   python 05.1_inference_data_prep.py
✗ Wrong:     python 05_1_inference_data_prep.py
```

### Data Flow
- **Module 1** creates: training data, trained model, inference data
- **Module 2** uses: Module 1's inference data to test monitoring
- **Module 3** uses: Module 1's trained model to demonstrate drift & retraining

---

## 🆘 Getting Help

### For Module-Specific Issues
See the **Troubleshooting** section at the end of each module's README.

### For General Questions
1. Review the module README carefully - most questions are answered there
2. Check the code comments in the scripts
3. Review error messages - they usually indicate what went wrong
4. Check that you're running from the correct directory

### For Setup Issues
- Verify Python 3.10+ is available: `python --version`
- Verify required packages: `pip list | grep pandas`
- Check that data files exist: `ls -la module1/data/`

---

## 🎓 What You'll Learn at Each Stage

### By end of Module 1:
- How data flows through an ML pipeline
- How to train and compare models
- How to deploy models as API endpoints
- How to generate predictions in production

### By end of Module 2:
- How to monitor model performance over time
- How to detect accuracy degradation
- How to identify when models need retraining
- How to set up automated alerts

### By end of Module 3:
- How to detect data drift (changes in input data)
- How to automatically trigger retraining
- How to deploy improvements without manual intervention
- How to build self-improving systems

---

## 🚀 Next Steps

**Ready to start?** 👇

[**→ Go to Module 1**](./module1/README.md)

---

## 📚 Additional Resources

- **Cloudera AI Documentation:** https://docs.cloudera.com/cml/
- **MLflow Documentation:** https://mlflow.org/docs/latest/
- **Evidently AI (used in Module 3):** https://docs.evidentlyai.com/
- **Bank Marketing Dataset:** https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

---

**Happy Learning!** 🎉

This lab is designed to give you real-world experience with the complete MLOps lifecycle. Enjoy the journey!
