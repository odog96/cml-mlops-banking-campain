import pandas as pd
import numpy as np
import sys
import os

print("--- Starting Live Data Simulator (Step 0) ---")

# Define the output file for our "live" data batch
LIVE_DATA_FILE = "live_unlabeled_batch.csv"
OUTPUT_DIR = "outputs"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, LIVE_DATA_FILE)
REFERENCE_FILE = "/home/cdsw/module1/data/bank-additional/bank-additional-full.csv"

# 1. Load the baseline "golden" dataset
try:
    reference_data = pd.read_csv(REFERENCE_FILE, sep=';')
    print(f"Loaded {len(reference_data)} reference records from banking_train.csv.")
except FileNotFoundError:
    print("Error: banking_train.csv not found.")
    sys.exit(1)

# 2. Simulate a batch of new, "live" data
# We re-use a sample of the training data but
# introduce drift to simulate a real-world shift.
current_data = reference_data.sample(n=3000, random_state=42).copy()

print(f"Simulating a new batch of {len(current_data)} live records...")

# --- SIMULATE DRIFT ---
# a) Numerical Drift: Marketing is targeting older clients
current_data['age'] = current_data['age'] + np.random.randint(8, 20, size=len(current_data))

# b) Categorical Drift: A new 'job' category is appearing
current_data['job'] = current_data['job'].replace({'student': 'gig-worker'})

# c) Drop the label! This is unlabeled production data.
current_data = current_data.drop(columns=['y'])

print("Drift simulated: 'age' shifted, 'student' replaced with 'gig-worker', 'y' label dropped.")

# 3. Save the new data batch to a file
os.makedirs(OUTPUT_DIR, exist_ok=True)
current_data.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Successfully saved live data to: {OUTPUT_PATH}")
print("This file represents a new batch of production data ready for monitoring.")
print("--- Live Data Simulator Finished ---")