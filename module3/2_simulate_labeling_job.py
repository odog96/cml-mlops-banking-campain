import pandas as pd
import numpy as np
import sys
import os
import json

print("--- Starting Label Acquisition Job (Step 2) ---")

# Get the project root directory (CML jobs run from /home/cdsw)
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # In Jupyter notebooks, __file__ is not defined
    BASE_DIR = os.getcwd()

TRIGGER_FILE = os.path.join(BASE_DIR, "outputs", "drift_status.json")

# --- This is the new "Trigger" check ---
print(f"Checking for trigger file: {TRIGGER_FILE}")
try:
    with open(TRIGGER_FILE, 'r') as f:
        trigger_data = json.load(f)
    
    print(f"Found trigger: Status = {trigger_data.get('status')}")
    if trigger_data.get('status') != "FAIL":
        print("Drift status is not 'FAIL'. No retraining needed.")
        print("--- Label Acquisition Job Finished (Skipped) ---")
        sys.exit(0) # Exit successfully
        
except FileNotFoundError:
    print("Error: Trigger file 'drift_status.json' not found.")
    print("Did you run '1_check_drift_explicit.py' first?")
    sys.exit(1)
# --- End of Trigger check ---

# 1. Load the original data to use as a base
# CML jobs execute from the project root directory (/home/cdsw)
base_data_path = os.path.join(BASE_DIR, "module1", "data", "bank-additional", "bank-additional.csv")
base_data = pd.read_csv(base_data_path, sep=";")
print(f"Loaded {len(base_data)} base records.")

# 2. Re-create the *exact same* drifted features from Job 1
# NOTE: We use the same sample and random_state to get 
# the same data as the monitoring job.
drifted_features = base_data.sample(n=3000, random_state=42).copy()
drifted_features['age'] = drifted_features['age'] + np.random.randint(8, 20, size=len(drifted_features))
drifted_features['job'] = drifted_features['job'].replace({'student': 'gig-worker'})
drifted_features = drifted_features.drop(columns=['y'])

print(f"Re-created {len(drifted_features)} drifted feature records.")

# 3. Simulate the new "Ground Truth" (Concept Drift)
# Our "business rule" for labels has also changed.
new_labels = []
for index, row in drifted_features.iterrows():
    if (row['job'] == 'gig-worker' and row['duration'] > 500):
        new_labels.append('yes')
    elif (row['age'] > 60):
        new_labels.append('yes')
    else:
        # Get the original label as a fallback
        new_labels.append(base_data.loc[index, 'y'])

new_labeled_data = drifted_features.copy()
new_labeled_data['y'] = new_labels

yes_rate = (new_labeled_data['y'] == 'yes').mean() * 100
print(f"Created new labels. New 'yes' rate: {yes_rate:.2f}%")

# 4. Save this as our new "ground truth" batch
output_path = os.path.join(BASE_DIR, "outputs", "new_labeled_batch_01.csv")
new_labeled_data.to_csv(output_path, index=False)

print("--- Label Acquisition Finished ---")
print(f"Saved '{output_path}'. Ready for retraining.")
