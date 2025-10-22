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
from pyspark.sql import SparkSession

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils import (
    DATALAKE_CONFIG,
    get_spark_session
)


def download_data(url, local_zip_path, local_csv_path):
    """Download and extract"""
    print("Downloading dataset...")
    
    urllib.request.urlretrieve(url, local_zip_path)
    print(f"Downloaded to {local_zip_path}")
    
    # Extract to data/ directory (not local_csv_path's dirname)
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall("data/")  # Extract directly to data/
    
    print(f"Extracted to data/")
    

def load_and_inspect_data(csv_path):
    """
    Load CSV into Pandas and perform basic inspection
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Pandas DataFrame
    """
    print("\nLoading data with Pandas...")
    print("\n printing csv_path",csv_path)
    df = pd.read_csv(csv_path, delimiter=';')
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nTarget variable distribution:")
    print(df['y'].value_counts())
    
    return df


def write_to_datalake(df, database_name, table_name, username):
    """
    Write Pandas DataFrame to data lake using Spark
    """
    print(f"\nWriting to data lake with user: {username}")
    
    # Get connection name from environment (matches yaml)
    CONNECTION_NAME = os.environ.get("CONNECTION_NAME", "go01-aw-dl")  # Changed default
    #CONNECTION_NAME = "go01-aw-dl"
    print(f"Using connection: {CONNECTION_NAME}")
    
    # Get Spark from CML data connection
    import cml.data_v1 as cmldata
    conn = cmldata.get_connection(CONNECTION_NAME)
    spark = conn.get_spark_session()
    
    # Fix Pandas compatibility
    if not hasattr(pd.DataFrame, 'iteritems'):
        pd.DataFrame.iteritems = pd.DataFrame.items
    
    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(df)
    
    # Create unique database and table names
    #full_db = "DEFAULT_ML_WORKSHOP"
    full_db = f"{database_name}_{username}".upper()
    full_table = f"{table_name}_{username}".upper()
    full_path = f"{full_db}.{full_table}"
    
    # Create database
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {full_db}")
    
    # Write as Iceberg
    spark_df.writeTo(full_path).using("iceberg").createOrReplace()
    
    print(f"\n✓ Data written to {full_path}")

def main():
    """
    Main execution function
    """
    print("="*60)
    print("Module 1 - Step 1: Data Ingestion")
    print("="*60)
    

    USERNAME = os.environ["PROJECT_OWNER"]
    print(f"\nUser: {USERNAME}")

    # Configuration
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    LOCAL_ZIP = "data/bank-additional.zip"
    #LOCAL_CSV = "data/bank-additional/bank-additional-full.csv"
    LOCAL_CSV = "data/bank-additional-full.csv"
    read_local_csv = "data/bank-additional/bank-additional-full.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Step 1: Download data
    if not os.path.exists(LOCAL_CSV):
        download_data(DATA_URL, LOCAL_ZIP, LOCAL_CSV)
    else:
        print(f"Data already exists at {LOCAL_CSV}")
    
    # Step 2: Load and inspect
    df = load_and_inspect_data(read_local_csv)
    
    # Step 3: Write to data lake
    write_to_datalake(
        df, 
        DATALAKE_CONFIG["database_name"], 
        DATALAKE_CONFIG["table_name"],
        USERNAME  # Add username parameter
    )
    
    print("\n" + "="*60)
    print("Ingestion complete! Proceed to 02_eda_feature_engineering.ipynb")
    print("="*60)


if __name__ == "__main__":
    main()
