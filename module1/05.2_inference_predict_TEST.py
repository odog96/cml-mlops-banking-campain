"""
Module 1 - Test Script: HTTP Request/Response Debugging
========================================================

This is a TEST version of 05.2_inference_predict.py that:
- Loads only a small sample (5 rows) for quick testing
- Prints detailed HTTP request and response information
- Helps debug API endpoint communication issues

This is NOT the production script - use 05.2_inference_predict.py for full runs.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import requests
import time
from datetime import datetime
import pickle

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("TEST: HTTP Request/Response Debugging")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ==================== LOAD DATA ====================
print("PHASE 1: Load Pre-engineered Inference Data")
print("-" * 80)

# Load engineered data
df_engineered = pd.read_csv("inference_data/engineered_inference_data.csv")
print(f"✓ Loaded engineered data: {df_engineered.shape}")

# Take only first 5 rows for testing
df_engineered = df_engineered.head(5)
print(f"✓ Using only first 5 rows for testing: {df_engineered.shape}")

# ==================== LOAD PREPROCESSING PIPELINE ====================
print("\nPHASE 1.5: Load Preprocessing Pipeline")
print("-" * 80)

with open("inference_data/feature_engineer.pkl", 'rb') as f:
    feature_engineer = pickle.load(f)
print(f"✓ Loaded preprocessing pipeline")

# ==================== APPLY PREPROCESSING ====================
print("\nPHASE 2: Apply Preprocessing")
print("-" * 80)

from helpers.preprocessing import PreprocessingPipeline

numeric_features = [
    'age', 'duration', 'campaign', 'pdays', 'previous',
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
    'engagement_score'
]

categorical_features = [
    'job', 'marital', 'education', 'default',
    'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome',
    'age_group', 'emp_var_category', 'duration_category'
]

# Load full training data to fit preprocessor on all categorical values
df_train = pd.read_csv("data/bank-additional/bank-additional-full.csv", sep=";")
df_train_eng = feature_engineer.transform(df_train)

# Create and fit preprocessor
preprocessor = PreprocessingPipeline(
    numeric_features=numeric_features,
    categorical_features=categorical_features,
    include_engagement=True
)

X_train_full = df_train_eng[numeric_features + categorical_features].copy()
preprocessor.fit(X_train_full)

# Transform inference data
X_subset = df_engineered[numeric_features + categorical_features].copy()
X_processed = pd.DataFrame(
    preprocessor.transform(X_subset),
    columns=preprocessor.get_feature_names()
)

print(f"✓ Preprocessed data shape: {X_processed.shape}")
print(f"  Features: {X_processed.shape[1]}")

# ==================== GET MODEL ENDPOINT ====================
print("\nPHASE 3: Initialize Model Endpoint")
print("-" * 80)

MODEL_ENDPOINT = "https://modelservice.ml-dbfc64d1-783.go01-dem.ylcu-atmi.cloudera.site/model"
ACCESS_KEY = "mbtbh46x9h7wxj4cdkxz9fxl0nzmrefv"

print(f"✓ Model endpoint: {MODEL_ENDPOINT}")
print(f"✓ Access key: {ACCESS_KEY[:10]}...{ACCESS_KEY[-10:]}")

# ==================== MAKE TEST PREDICTION ====================
print("\nPHASE 4: Make Single Test Prediction with Full HTTP Details")
print("-" * 80)

# Take just the first row
first_row = X_processed.iloc[0:1]

print(f"\nTest data shape: {first_row.shape}")
print(f"Test data columns: {list(first_row.columns)[:5]}... ({len(first_row.columns)} total)")
print(f"First row sample values: {first_row.iloc[0, :3].values}")

# Prepare API payload
payload = {
    "accessKey": ACCESS_KEY,
    "request": {
        "dataframe_split": {
            "columns": list(first_row.columns),
            "data": first_row.values.tolist()
        }
    }
}

payload_json = json.dumps(payload)

print(f"\n📤 REQUEST DETAILS:")
print(f"  URL: {MODEL_ENDPOINT}")
print(f"  Method: POST")
print(f"  Headers: {{'Content-Type': 'application/json'}}")
print(f"  Payload size: {len(payload_json)} bytes")
print(f"  Access Key present: Yes")
print(f"  Data shape in request: {len(first_row)} rows × {len(first_row.columns)} columns")

print(f"\n📨 Payload structure:")
print(f"  {json.dumps(payload, indent=2)[:300]}...")

# Send request
print(f"\n⏳ Sending request...")
request_start = time.time()

try:
    response = requests.post(
        MODEL_ENDPOINT,
        data=payload_json,
        headers={'Content-Type': 'application/json'},
        timeout=30
    )
    request_time = time.time() - request_start

    print(f"✓ Request completed in {request_time:.2f} seconds")

    print(f"\n📥 RESPONSE DETAILS:")
    print(f"  Status Code: {response.status_code}")
    print(f"  Status Text: {response.reason}")
    print(f"  Response Size: {len(response.text)} bytes")
    print(f"  Content-Type: {response.headers.get('Content-Type', 'Not specified')}")

    print(f"\n📄 Response Headers:")
    for key, value in response.headers.items():
        print(f"    {key}: {value}")

    print(f"\n📋 Response Body:")
    if response.status_code == 200:
        try:
            result_json = response.json()
            print(f"  {json.dumps(result_json, indent=2)}")

            # Extract prediction
            if 'response' in result_json and 'prediction' in result_json['response']:
                predictions = result_json['response']['prediction']
                print(f"\n✓ Predictions extracted: {predictions}")
            else:
                print(f"\n⚠️  Response structure doesn't match expected format")
                print(f"   Expected: {{'response': {{'prediction': [...]}}}}")
                print(f"   Got keys: {list(result_json.keys())}")
        except json.JSONDecodeError:
            print(f"  ⚠️  Response is not valid JSON")
            print(f"  Raw text: {response.text[:500]}")
    else:
        print(f"  Status: {response.status_code}")
        print(f"  Body: {response.text}")

except Exception as e:
    print(f"❌ Request failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
