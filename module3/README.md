# Module 2: MLOps - Production Model Lifecycle

**Focus:** Model monitoring, drift detection, and automated retraining in production

---

## 📚 What You'll Learn

By the end of this module, you will:
- Detect when your model's predictions become less reliable (data drift)
- Monitor model performance in production environments
- Implement automated retraining pipelines
- Make data-driven decisions about when and how to retrain
- Version and redeploy models safely

---

## 🎯 Module Overview

**The Challenge:** Your model was trained on data from 2008-2010, but customer behavior changes over time. How do you know when to retrain?

**The Solution:** Monitor for drift → Detect degradation → Retrain automatically → Redeploy safely

### Real-World Scenario
Imagine your bank marketing model is deployed. Over 6 months:
- Customer demographics shift (aging population)
- Contact preferences change (mobile vs. landline)
- Economic conditions evolve
- Campaign timing strategies differ

**Question:** Should you retrain with only new data, or combine old + new?

---

## 📁 Module Files

| File | Purpose | When to Run |
|------|---------|-------------|
| `01_simulate_time_passage.py` | Generate new data with realistic drift | Step 1 |
| `02_drift_detection.py` | Analyze drift using Evidently AI | Step 2 |
| `03_retrain_pipeline.py` | Automated retraining with data mixing | Step 3 |
| `04_redeploy_model.py` | Version and deploy updated model | Step 4 |
| `drift_config.yaml` | Monitoring thresholds and settings | Reference |

---

## 🚀 Step-by-Step Guide

### Step 1: Simulate Production Data (Time Passage)

**Run:** `python module2/01_simulate_time_passage.py`

**What it does:**
- Generates 6 months of new customer data
- Introduces realistic drift patterns:
  - **Age drift:** Customer base ages by ~2 years on average
  - **Contact method drift:** Shift from landline to mobile (20% → 35%)
  - **Economic drift:** Interest rates and employment indicators change
  - **Seasonal drift:** Different campaign timing

**Output:** `data/new_production_data.csv` (~5,000 new records)

**💡 Look for:** Summary statistics showing how distributions changed

---

### Step 2: Detect Drift

**Run:** `python module2/02_drift_detection.py`

**What it does:**
- Compares training data vs. new production data
- Uses **Evidently AI** to detect:
  - Feature drift (did input distributions change?)
  - Target drift (is the outcome rate different?)
  - Model performance degradation
- Generates HTML reports and metrics

**Key Metrics:**
- **Drift Score:** 0-1 scale (>0.3 = significant drift)
- **Feature Stability:** Which features changed most?
- **Prediction Drift:** Is your model's output distribution different?

**Output:** 
- `outputs/drift_report.html` (visual report)
- `outputs/drift_metrics.json` (programmatic metrics)

**💡 Decision Point:** If drift detected, proceed to retraining

---

### Step 3: Retrain the Model

**Run:** `python module2/03_retrain_pipeline.py`

**What it does:**
- Implements **3 retraining strategies** (you choose):

#### Strategy A: New Data Only
```python
# Fastest, but may lose historical patterns
train_data = new_production_data
```
**Pros:** Fast training, adapts quickly  
**Cons:** May forget important historical patterns

#### Strategy B: Combined (Last 6 Months + New) ⭐ Recommended
```python
# Balance of historical knowledge + new patterns
train_data = last_6_months_original + new_production_data
```
**Pros:** Best of both worlds  
**Cons:** Slightly slower training

#### Strategy C: Weighted Sampling
```python
# Emphasize recent data, but keep some history
train_data = sample_with_weights(
    original_data, weight=0.3,
    new_data, weight=0.7
)
```
**Pros:** Fine-grained control  
**Cons:** More complex to tune

**The script will:**
1. Load original training data
2. Load new production data
3. Apply your chosen strategy
4. Train 3 models (LR, RF, XGBoost)
5. Compare performance: Old model vs. New model
6. Log everything to MLflow

**Output:**
- New trained models in MLflow
- Performance comparison report
- Recommendation on whether to deploy

**💡 Minimum Data Requirements:** 
- At least **1,000 new samples** before retraining
- Sufficient representation of both classes

---

### Step 4: Redeploy the Model

**Run:** `python module2/04_redeploy_model.py`

**What it does:**
- Registers new model in MLflow Model Registry
- Implements safe deployment:
  - **Staging:** Test in isolated environment
  - **Canary:** Route 10% traffic to new model
  - **Full rollout:** If metrics improve, route 100%
- Tags with version info and drift metadata

**Deployment Options:**

| Method | Traffic Split | Rollback |
|--------|--------------|----------|
| Blue-Green | 0% → 100% | Instant |
| Canary | 10% → 50% → 100% | Gradual |
| A/B Test | 50/50 for 2 weeks | Compare |

**Output:**
- Model registered as `banking_model_v2`
- Deployment logs
- Rollback commands if needed

---

## ⚙️ Configuration: `drift_config.yaml`

```yaml
# Monitoring thresholds
drift_detection:
  feature_drift_threshold: 0.3  # Trigger retraining
  performance_drop_threshold: 0.05  # 5% AUC drop
  minimum_new_samples: 1000  # Before retraining

# Retraining strategy
retraining:
  strategy: "combined"  # Options: new_only, combined, weighted
  historical_window_months: 6
  new_data_weight: 0.7  # For weighted strategy

# Monitoring frequency
monitoring:
  schedule: "weekly"  # Options: daily, weekly, event_driven
  drift_check_samples: 500  # Minimum for drift detection
```

**How to adjust:**
- **Stricter monitoring:** Lower `feature_drift_threshold` to 0.2
- **More historical data:** Increase `historical_window_months` to 12
- **Faster adaptation:** Use `strategy: "new_only"`

---

## 🎓 Key Concepts

### What is Data Drift?

**Concept Drift:** When the relationship between features and target changes  
*Example:* Economic downturn makes customers less likely to subscribe, even with same demographics

**Feature Drift:** When input data distributions change  
*Example:* Customer age distribution shifts from avg 38 → 42 years

**Prediction Drift:** When model's output distribution changes  
*Example:* Model predicts "yes" 20% of time instead of 11%

### When to Retrain?

| Indicator | Threshold | Action |
|-----------|-----------|--------|
| Drift Score > 0.3 | High drift | Retrain soon |
| AUC drops > 5% | Performance degraded | Retrain immediately |
| 1000+ new samples | Sufficient data | Can retrain |
| < 500 new samples | Insufficient data | Wait for more data |

### Training Data Composition

**Rule of Thumb:** Include enough historical data to preserve patterns, but weight recent data higher.

**Example Decision Tree:**
```
Is drift detected? 
├─ Yes → High drift (>0.5)?
│   ├─ Yes → Use new_only strategy (adapt fast)
│   └─ No → Use combined strategy (6mo + new)
└─ No → No retraining needed
```

---

## 💡 Discussion Questions for Lab

1. **Monitoring Frequency:**
   - Daily: Catches issues fast, but expensive
   - Weekly: Good balance for most use cases
   - Event-driven: Retrain when >1000 samples accumulated
   - *What's right for your use case?*

2. **Data Mixing:**
   - All historical data: Risk of stale patterns
   - Only new data: Risk of forgetting important patterns
   - Last 6 months + new: Balanced approach
   - *How much history should you keep?*

3. **Retraining Triggers:**
   - Scheduled (every Monday)
   - Threshold-based (drift > 0.3)
   - Performance-based (AUC drops 5%)
   - Hybrid (drift OR performance)
   - *Which trigger makes sense?*

---

## 📊 Success Metrics

By the end of Module 2, you should be able to:
- [ ] Generate realistic production data with drift
- [ ] Interpret drift detection reports
- [ ] Choose appropriate retraining strategy
- [ ] Compare old vs. new model performance
- [ ] Safely deploy updated models

**Expected Outcomes:**
- Drift detection report showing 20-40% drift score
- Retrained model with 3-5% improved AUC
- Automated retraining pipeline that runs without manual intervention

---

## 🐛 Troubleshooting

**Issue:** Not enough new data (< 1000 samples)  
**Solution:** Run `01_simulate_time_passage.py` with `--samples 5000`

**Issue:** No drift detected  
**Solution:** Increase drift in simulation or check thresholds in config

**Issue:** Retrained model performs worse  
**Solution:** Try different strategy (combined vs. new_only)

**Issue:** MLflow experiment not found  
**Solution:** Ensure Module 1 completed and experiments exist

---

## 🔗 What's Next?

**Module 3:** Model deployment and serving  
**Module 4:** Advanced monitoring and observability

---

## 📚 Additional Resources

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [MLflow Model Registry Guide](https://mlflow.org/docs/latest/model-registry.html)
- [ML Monitoring Best Practices](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

---

**Lab Duration:** ~2 hours  
**Prerequisites:** Module 1 completed  
**Difficulty:** Intermediate

---

*Questions or issues? Check the troubleshooting section or ask your instructor.*
