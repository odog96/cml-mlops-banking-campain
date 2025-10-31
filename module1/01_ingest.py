"""
Module 1 - Step 1: Data Ingestion
===================================

This script downloads the UCI Bank Marketing dataset and writes it to the data lake.

Key Learning Points:
- Direct data ingestion without complex transformations
- Using Spark for write operations (leveraging Iceberg format benefits)
- Setting up the foundation for downstream ML workflows

Dataset: UCI Bank Marketing Dataset
https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
"""

import os
import sys
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pyspark.sql import SparkSession

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils import (
    DATALAKE_CONFIG,
    get_spark_session
)


def download_data(url, local_zip_path, local_csv_path):
    """Download and extract the dataset"""
    download_start = time.time()

    print("Downloading dataset...")

    urllib.request.urlretrieve(url, local_zip_path)
    print(f"Downloaded to {local_zip_path}")

    # Extract to data/ directory (not local_csv_path's dirname)
    extract_start = time.time()
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/")  # Extract directly to data/
    extract_elapsed = time.time() - extract_start

    print(f"Extracted to data/ ({extract_elapsed:.2f} seconds)")

    download_elapsed = time.time() - download_start
    print(f"Total download time: {download_elapsed:.2f} seconds")
    

def load_and_inspect_data(csv_path):
    """
    Load CSV into Pandas and perform basic inspection

    Args:
        csv_path: Path to CSV file

    Returns:
        Pandas DataFrame
    """
    load_start = time.time()

    print("\nLoading data with Pandas...")
    print("\n printing csv_path", csv_path)

    read_start = time.time()
    df = pd.read_csv(csv_path, delimiter=';')
    read_elapsed = time.time() - read_start

    print(f"Data loaded in {read_elapsed:.2f} seconds")
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())

    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nTarget variable distribution:")
    print(df['y'].value_counts())

    load_elapsed = time.time() - load_start
    print(f"\nTotal load and inspect time: {load_elapsed:.2f} seconds")

    return df


def write_to_datalake(df, database_name, table_name, username):
    """
    Write Pandas DataFrame to data lake using Spark
    """
    write_start = time.time()

    print(f"\nWriting to data lake with user: {username}")

    # Get connection name from environment (matches yaml)
    CONNECTION_NAME = os.environ.get("CONNECTION_NAME", "se-aws-edl")  #
    print(f"Using connection: {CONNECTION_NAME}")

    # Get Spark from CML data connection
    import cml.data_v1 as cmldata
    conn = cmldata.get_connection(CONNECTION_NAME)
    spark = conn.get_spark_session()

    # Fix Pandas compatibility
    if not hasattr(pd.DataFrame, 'iteritems'):
        pd.DataFrame.iteritems = pd.DataFrame.items

    # Convert to Spark DataFrame
    conversion_start = time.time()
    spark_df = spark.createDataFrame(df)
    conversion_elapsed = time.time() - conversion_start
    print(f"Spark DataFrame conversion: {conversion_elapsed:.2f} seconds")

    # Create unique database and table names
    #full_db = "DEFAULT_ML_WORKSHOP"
    full_db = f"{database_name}_{username}".upper()
    full_table = f"{table_name}_{username}".upper()
    full_path = f"{full_db}.{full_table}"

    # Create database
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {full_db}")

    # Write as Iceberg
    iceberg_start = time.time()
    spark_df.writeTo(full_path).using("iceberg").createOrReplace()
    iceberg_elapsed = time.time() - iceberg_start

    print(f"✓ Data written to {full_path}")
    print(f"Iceberg write time: {iceberg_elapsed:.2f} seconds")

    write_elapsed = time.time() - write_start
    print(f"Total data lake write time: {write_elapsed:.2f} seconds")


def create_sample_inference_data(df):
    """
    Create sample inference data for demonstration and testing purposes.

    Takes a random sample of the raw data (without target variable) and saves
    it to the inference_data folder for use in the inference job pipeline.

    Args:
        df: Full dataset DataFrame
    """
    inference_start = time.time()

    print("\nCreating sample inference data...")

    # Create inference_data directory if it doesn't exist
    os.makedirs("inference_data", exist_ok=True)

    # Take a random sample (1000 records for demo)
    np.random.seed(42)
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)

    # Remove target variable 'y' for inference (in real scenario, we wouldn't have it)
    inference_df = sample_df.drop(columns=['y'])

    # Save as CSV with semicolon delimiter (matching training data format)
    output_path = "inference_data/raw_inference_data.csv"
    inference_df.to_csv(output_path, sep=";", index=False)

    inference_elapsed = time.time() - inference_start

    print(f"✓ Sample inference data created")
    print(f"  Shape: {inference_df.shape}")
    print(f"  Columns: {len(inference_df.columns)}")
    print(f"  Saved to: {output_path}")
    print(f"  Creation time: {inference_elapsed:.2f} seconds")

def main():
    """
    Main execution function with timing instrumentation
    """
    # Start overall timing
    script_start = time.time()

    print("=" * 60)
    print("Module 1 - Step 1: Data Ingestion")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    USERNAME = os.environ["PROJECT_OWNER"]
    print(f"User: {USERNAME}")

    # Configuration
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    LOCAL_ZIP = "data/bank-additional.zip"
    #LOCAL_CSV = "data/bank-additional/bank-additional-full.csv"
    LOCAL_CSV = "data/bank-additional-full.csv"
    read_local_csv = "data/bank-additional/bank-additional-full.csv"

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Step 1: Download data
    print("\n" + "-" * 60)
    print("STEP 1: Download Data")
    print("-" * 60)
    if not os.path.exists(LOCAL_CSV):
        download_data(DATA_URL, LOCAL_ZIP, LOCAL_CSV)
    else:
        print(f"Data already exists at {LOCAL_CSV}")

    # Step 2: Load and inspect
    print("\n" + "-" * 60)
    print("STEP 2: Load and Inspect Data")
    print("-" * 60)
    df = load_and_inspect_data(read_local_csv)

    # Step 3: Write to data lake
    print("\n" + "-" * 60)
    print("STEP 3: Write to Data Lake")
    print("-" * 60)
    write_to_datalake(
        df,
        DATALAKE_CONFIG["database_name"],
        DATALAKE_CONFIG["table_name"],
        USERNAME  # Add username parameter
    )

    # Step 4: Create sample inference data
    print("\n" + "-" * 60)
    print("STEP 4: Create Sample Inference Data")
    print("-" * 60)
    create_sample_inference_data(df)

    # Calculate total execution time
    script_elapsed = time.time() - script_start

    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {script_elapsed:.2f} seconds ({script_elapsed/60:.2f} minutes)")
    print("=" * 60)
    print("✅ Ingestion complete!")
    print("\nNext steps:")
    print("  • Training pipeline: 02_eda_feature_engineering.ipynb → 03_train_quick.py")
    print("  • Inference pipeline: 05_inference_data_prep.py → 06_inference_predict.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
