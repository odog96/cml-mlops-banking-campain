# Module 3: Proactive MLOps - The Automated Retraining Loop

## 1. Objective

In Module 2, we reacted to poor model performance (accuracy) after we received new labels. This is reactive monitoring, and it's too slow. In the real world, you might not get labels for weeks, by which time a drifting model has already cost you.

This module teaches proactive MLOps. We will build an event-driven pipeline that proactively detects data drift (changes in the input data) and uses that as a trigger to automatically retrain and deploy a new, smarter model. We will simulate the entire MLOps loop, from detection to deployment.

**The Big Idea:** We're moving from a lagging indicator (bad accuracy) to a leading indicator (bad data).

## 2. The MLOps Pipeline We Will Build

We will simulate a 4-step automated pipeline. In a production system, a CML Job's success or failure would automatically trigger the next step. For this lab, we will run the scripts one by one to see the cause-and-effect of each "event."

### Job 1: Detect Drift (1_check_drift.py)

**Action:** Simulates new, unlabeled production data with drift (new age patterns, new job categories).

**Detects:** Uses Evidently AI to compare the new data to our "golden" training set.

**Event:** The job fails intentionally when drift is detected. This "failure" is our trigger!

### Job 2: Acquire Labels (2_simulate_labeling_job.py)

**Trigger:** "Triggered" by the failure of Job 1.

**Action:** Simulates the data engineering work of acquiring labels for the new, drifted data. It also simulates "concept drift" by applying new business rules to create the labels.

**Output:** A new labeled dataset: new_labeled_batch_01.csv.

### Job 3: Retrain Model (3_retrain_model.py)

**Trigger:** "Triggered" by the creation of the new labeled data.

**Action:** Combines the original training data with the new batch.

**Output:** Trains a model_v2 on this combined dataset and logs it to an MLflow experiment, saving the run_id.

### Job 4: Register & Deploy (4_register_and_deploy.py)

**Trigger:** "Triggered" by the successful training of model_v2.

**Action:** Reads the run_id from the previous step.

**Output:** Uses the cmlapi client to register the new model, build a runtime, and deploy it as banking_campaign_predictor_v2, completing the MLOps loop.

## 3. Hands-On Lab Instructions

### Prerequisite

Ensure you have the banking_train.csv file in the root of your project.

### Step 1: Run the Drift Detection Job

This job acts as our sensor. It checks for drift and fails if it finds it.

- Open a CML Workbench session (e.g., a terminal).
- Run the first script:

```bash
python module3/1_check_drift.py
```

**Observe the Output:**

- The script will print `!!! DATA DRIFT DETECTED! !!!`.
- The job will fail (with a `sys.exit(1)`). This is the "event" that would trigger our pipeline.

**Analyze the Report:**

- Go to the file browser on the left.
- Open the `1_drift_report.html` file.
- Inspect the report. You will see clear "DRIFTED" tags on the age and job features that we simulated.

### Step 2: Run the Label Acquisition Job

This job simulates the (often manual) process of getting new labels for the drifted data.

- In the same terminal, run the second script:

```bash
python module3/2_simulate_labeling_job.py
```

**Observe the Output:**

- The script will print that it has created `new_labeled_batch_01.csv`.
- This job finishes successfully, "triggering" the retraining step.

### Step 3: Run the Retraining Job

This job trains a new model on all the data (old + new) and logs it to MLflow.

- In the same terminal, run the third script:

```bash
python module3/3_retrain_model.py
```

**Observe the Output:**

- The script will train a new model and log it to an MLflow experiment called `banking_retraining_pipeline`.
- It saves the model's run_id to `outputs/retrain_run_info.json`.

**Check MLflow (Optional):**

- Go to the MLflow UI.
- You will see the new experiment and the `retrain_on_drift_v2` run with its F1 score.

### Step 4: Run the Register & Deploy Job

This is the final step. It takes the newly trained model from MLflow and deploys it using the CML API.

- In the same terminal, run the final script:

```bash
python module3/4_register_and_deploy.py
```

**Observe the Output:**

- This script will run for several minutes. It is the most complex step.
- It will register the model in the CML Model Registry.
- It will create a model build (compiling the runtime).
- It will wait for the build to complete.
- Once built, it will deploy the model.

## 4. Final Result & Verification

You have successfully automated the entire MLOps loop!

- Go to the Models tab in your CML project.
- You will see your new model: `banking_campaign_predictor_v2`.
- Click on it. You will see that it has been Built and Deployed.

**You now have a brand new, smarter model serving as an API endpoint, all triggered by the initial data drift.**

Congratulations! You've completed the proactive MLOps module.
