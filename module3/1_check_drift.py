import pandas as pd
import numpy as np
import sys
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset

print("--- Starting Monitoring Job (1/4) ---")

# 1. Load the baseline "golden" dataset
try:
    reference_data = pd.read_csv("banking_train.csv")
    print(f"Loaded {len(reference_data)} reference records.")
except FileNotFoundError:
    print("Error: banking_train.csv not found.")
    sys.exit(1)

# 2. Simulate a batch of new, "live" data
# We re-use a sample of the training data but
# introduce drift to simulate a real-world shift.
current_data = reference_data.sample(n=3000, random_state=42).copy()

print(f"Simulating {len(current_data)} new live records...")

# --- SIMULATE DRIFT ---
# a) Numerical Drift: Marketing is targeting older clients
current_data['age'] = current_data['age'] + np.random.randint(8, 20, size=len(current_data))

# b) Categorical Drift: A new 'job' category is appearing
current_data['job'] = current_data['job'].replace({'student': 'gig-worker'})

# c) Drop the label! This is unlabeled production data.
current_data = current_data.drop(columns=['y'])
print("Drift simulated: 'age' shifted, 'student' replaced with 'gig-worker', 'y' label dropped.")


# 3. Run the Evidently Test Suite
data_drift_suite = TestSuite(tests=[
    DataDriftTestPreset(),
])
data_drift_suite.run(current_data=current_data, 
                     reference_data=reference_data, 
                     column_mapping=None)

# 4. Save the report so the engineer can see the drift
data_drift_suite.save_html("1_drift_report.html")
print("Saved 1_drift_report.html for review.")

# 5. The "Trigger"
if not data_drift_suite.is_passed():
    print("!!! DATA DRIFT DETECTED! !!!")
    print("Failing this job to trigger the retraining pipeline.")
    # This non-zero exit code is the "event"
    sys.exit(1)
else:
    print("No data drift detected. All clear.")
    
print("--- Monitoring Job Finished ---")