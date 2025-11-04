# **Module 3: Proactive MLOps \- The Automated Retraining Loop**

## **1\. Objective**

In Module 2, we *reacted* to poor model performance (accuracy). This module teaches **proactive MLOps**. We will build an event-driven pipeline that *proactively* detects **data drift** and uses that as a trigger to automatically retrain and deploy a new, smarter model.

This is a much more advanced and realistic workflow. We will make the "triggers" and "artifacts" visible at every step.

## **2\. The MLOps Pipeline We Will Build**

We will run a series of scripts that act as a real pipeline. Each script produces an artifact that the next script consumes.

1. **Job 0: 0\_simulate\_live\_data.py**  
   * **Action:** Simulates a batch of new, unlabeled production data with "drift" (new age patterns, new job categories).  
   * **Artifact:** Saves outputs/live\_unlabeled\_batch.csv.  
2. **Job 1: 1\_check\_drift\_explicit.py**  
   * **Action:** Reads both the banking\_train.csv (reference) and the new live\_unlabeled\_batch.csv (current).  
   * **Detects:** Uses an *explicit* Evidently AI Test Suite to check for specific drift (e.g., in age and job).  
   * **Artifacts:**  
     * 1\_drift\_report\_explicit.html: A rich, visual dashboard of the drift.  
     * outputs/drift\_status.json: A simple JSON file with {"status": "FAIL"}. This is our **pipeline trigger**.  
3. **Visualization: app.py**  
   * **Action:** This script is *launched as a CML Application*.  
   * **Result:** It hosts the 1\_drift\_report\_explicit.html file on a permanent, shareable URL for the team to review.  
4. **Job 2: 2\_simulate\_labeling\_job.py**  
   * **Trigger:** This job *first* reads outputs/drift\_status.json.  
   * **Action:** If status \== "FAIL", it proceeds to simulate the data engineering work of acquiring labels for the new, drifted data.  
   * **Artifact:** Saves outputs/new\_labeled\_batch\_01.csv.  
5. **Job 3: 3\_retrain\_model.py**  
   * **Trigger:** "Triggered" by the creation of the new labeled data.  
   * **Action:** Combines the *original* training data with the *new* batch, trains a model\_v2, and logs it to an MLflow experiment.  
   * **Artifact:** Saves the MLflow run\_id to outputs/retrain\_run\_info.json.  
6. **Job 4: 4\_register\_and\_deploy.py**  
   * **Trigger:** Reads the run\_id from outputs/retrain\_run\_info.json.  
   * **Action:** Uses the cmlapi client to register, build, and deploy the new model from that run\_id.  
   * **Artifact:** A new, deployed banking\_campaign\_predictor\_v2 model.

## **3\. Hands-On Lab Instructions**

**Prerequisite:** Ensure you have the banking\_train.csv file in your project. All scripts should be in a module3/ folder.

### **Step 1: Simulate and Detect Drift**

First, we'll run the two jobs that find the problem.

1. **Simulate Live Data:** Open a CML Workbench terminal and run:  
   python module3/0\_simulate\_live\_data.py

   * **Result:** Creates outputs/live\_unlabeled\_batch.csv.  
2. **Run Drift Check:** Now, run the monitoring job:  
   python module3/1\_check\_drift\_explicit.py

   * **Observe:** The job will print **\!\!\! DATA DRIFT DETECTED\! \!\!\!** and fail.  
   * **Artifacts:** This creates 1\_drift\_report\_explicit.html and outputs/drift\_status.json.

### **Step 2: Publish the Visual Dashboard (Option B)**

This is the "pro" MLOps step. Let's publish our report.

1. Go to the **Applications** tab in your CML project.  
2. Click **New Application**.  
3. Fill in the details:  
   * **Name:** Data Drift Dashboard  
   * **Script:** module3/app.py  
   * **Kernel:** Python 3 (or standard)  
   * **Resource Profile:** (Smallest is fine)  
4. Click **Create Application**. After a moment, a URL will appear.  
5. **Launch the URL:** You will see your interactive Evidently report\! This is what you would share with your team.

### **Step 3: Trigger the Retraining Pipeline**

Now we'll run the rest of the pipeline, which is "triggered" by the artifacts from Step 1\.

1. **Run Label Acquisition:**  
   python module3/2\_simulate\_labeling\_job.py

   * **Observe:** It will first print "Found trigger: Status \= FAIL" and then proceed.  
   * **Result:** Creates outputs/new\_labeled\_batch\_01.csv.  
2. **Run Retraining:**  
   python module3/3\_retrain\_model.py

   * **Observe:** This job trains the new model and logs it to MLflow.  
   * **Result:** Creates outputs/retrain\_run\_info.json.  
3. **Run Register & Deploy:**  
   python module3/4\_register\_and\_deploy.py

   * **Observe:** This is the longest step. It will print its progress as it registers, builds, and deploys the new model.

## **4\. Final Result & Verification**

You have successfully simulated an end-to-end, event-driven MLOps pipeline.

1. Go to the **Models** tab in your CML project.  
2. You will see your new model: **banking\_campaign\_predictor\_v2**.  
3. Click on it. You will see that it has been **Built** and **Deployed**.  
4. You now have a new, smarter model serving as an API endpoint, all triggered by a proactive drift detection test\!

**Congratulations\!**