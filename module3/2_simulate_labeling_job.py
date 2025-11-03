import pandas as pd
import numpy as np

print("--- Starting Label Acquisition Job (2/4) ---")

# 1. Load the original data to use as a base
base_data = pd.read_csv("banking_train.csv")
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
# Rule: 'gig-worker' now has a high subscription rate,
# and older clients are more receptive.

new_labels = []
for index, row in drifted_features.iterrows():
    if (row['job'] == 'gig-worker' and row['balance'] > 1000):
        new_labels.append('yes')
    elif (row['age'] > 60):
        new_labels.append('yes')
    else:
        # Get the original label as a fallback
        new_labels.append(base_data.loc[index, 'y'])

new_labeled_data = drifted_features.copy()
new_labeled_data['y'] = new_labels

print(f"Created new labels. New 'yes' rate: { (new_labeled_data['y'] == 'yes').mean():.2% }")

# 4. Save this as our new "ground truth" batch
new_labeled_data.to_csv("new_labeled_batch_01.csv", index=False)

print("--- Label Acquisition Finished ---")
print("Saved 'new_labeled_batch_01.csv'. Ready for retraining.")