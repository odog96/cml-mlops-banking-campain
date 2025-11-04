import pandas as pd
import numpy as np
import sys
import json
import os
from evidently.legacy.test_suite import TestSuite
from evidently.legacy.tests import * # Import all standard tests

print("--- Starting Explicit Monitoring Job (Step 1) ---")

# Define our data files
REFERENCE_FILE = "/home/cdsw/module1/data/bank-additional/bank-additional-full.csv"
CURRENT_FILE = os.path.join("outputs", "live_unlabeled_batch.csv")
OUTPUT_DIR = "outputs"
OUTPUT_REPORT_HTML = "1_drift_report_explicit.html"
OUTPUT_TRIGGER_FILE = os.path.join(OUTPUT_DIR, "drift_status.json") # This is our new trigger!

# 1. Load datasets
try:
    reference_data = pd.read_csv(REFERENCE_FILE, sep=';')
    current_data = pd.read_csv(CURRENT_FILE)
    print(f"Loaded {len(reference_data)} reference records.")
    print(f"Loaded {len(current_data)} current records to check.")
except FileNotFoundError as e:
    print(f"Error: Could not load data files. {e}")
    print("Did you run '0_simulate_live_data.py' first?")
    sys.exit(1)

# 2. Build an EXPLICIT Test Suite (The "Ah-ha!" Moment)
# This is far better than a preset for a lab.
# We are manually configuring the tests we care about.
data_drift_suite = TestSuite(tests=[
    # Test 1: A high-level health check.
    # "Fail if more than 30% of my columns show drift."
    TestShareOfDriftedColumns(lte=0.3),

    # Test 2: A specific test for a critical feature.
    # We KNOW 'age' is important. Let's set a tight threshold.
    # (Default test is K-S test, stattest_threshold=0.1 means p-value < 0.1)
    TestColumnDrift(column_name="age", stattest_threshold=0.1),

    # Test 3: A specific test for a categorical feature.
    # (Default test is Chi-Squared)
    TestColumnDrift(column_name="job"),

    # Test 4: Explicitly check for out-of-list values in categorical feature.
    # This will catch 'gig-worker' which is not in the reference data
    TestCatColumnsOutOfListValues(columns=["job"])
])

# 3. Run the tests
print("Running explicit drift tests...")
data_drift_suite.run(current_data=current_data, 
                     reference_data=reference_data, 
                     column_mapping=None)

# 4. Save the visual HTML report for the human
data_drift_suite.save_html(OUTPUT_REPORT_HTML)
print(f"Saved visual report to: {OUTPUT_REPORT_HTML}")

# 5. The "Trigger": Save the status as a JSON artifact
results = data_drift_suite.as_dict()
status = results.get("summary", {}).get("all_passed", False)
status_str = "PASS" if status else "FAIL"

trigger_data = {
    "status": status_str,
    "total_tests": results.get("summary", {}).get("total_tests"),
    "passed_tests": results.get("summary", {}).get("passed_tests"),
    "failed_tests": results.get("summary", {}).get("failed_tests"),
    "timestamp": pd.Timestamp.now().isoformat()
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_TRIGGER_FILE, 'w') as f:
    json.dump(trigger_data, f, indent=2)
    
print(f"Saved trigger artifact to: {OUTPUT_TRIGGER_FILE}")

# 6. Final message based on status
if status_str == "FAIL":
    print("\n!!! DATA DRIFT DETECTED! (Status: FAIL) !!!")
    print("The pipeline will be triggered by the 'drift_status.json' file.")
    # We can still exit(1) to make this job "fail" in the CML UI
    sys.exit(1) 
else:
    print("\nNo significant data drift detected. (Status: PASS)")

print("--- Monitoring Job Finished ---")
