"""
Reset Workshop Environment
===========================

This script cleans up the workshop environment by:
1. Deleting the local data folder
2. Dropping the Iceberg database and table


Use this to start fresh or clean up after the workshop.
"""

import os
import sys
import shutil
import cml.data_v1 as cmldata

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils import DATALAKE_CONFIG


def delete_local_data():
    """
    Delete the local data folder
    """
    data_folder = "data"
    
    if os.path.exists(data_folder):
        print(f"Deleting local data folder: {data_folder}")
        shutil.rmtree(data_folder)
        print(f"✓ Deleted {data_folder}/")
    else:
        print(f"✓ Data folder doesn't exist: {data_folder}")


def drop_database_and_tables():
    """
    Drop the user's database and all tables
    """
    # Get username
    USERNAME = os.environ.get("PROJECT_OWNER", "default_user")
    
    # Get connection
    CONNECTION_NAME = os.environ.get("CONNECTION_NAME", "go01-aw-dl")
    
    try:
        print(f"\nConnecting to data lake: {CONNECTION_NAME}")
        conn = cmldata.get_connection(CONNECTION_NAME)
        spark = conn.get_spark_session()
        
        # Database name
        #full_db = f"{DATALAKE_CONFIG['database_name']}_{USERNAME}".upper()
        full_db = "DEFAULT_ML_WORKSHOP"
        
        print(f"\nChecking for database: {full_db}")
        
        # Check if database exists
        databases = spark.sql("SHOW DATABASES").collect()
        db_names = [row.namespace for row in databases]
        
        if full_db.lower() in [db.lower() for db in db_names]:
            # Show tables before dropping
            tables = spark.sql(f"SHOW TABLES IN {full_db}").collect()
            if tables:
                print(f"Tables in {full_db}:")
                for table in tables:
                    print(f"  - {table.tableName}")
            
            # Drop database (CASCADE drops all tables)
            print(f"\nDropping database: {full_db}")
            spark.sql(f"DROP DATABASE IF EXISTS {full_db} CASCADE")
            print(f"✓ Dropped database: {full_db}")
        else:
            print(f"✓ Database doesn't exist: {full_db}")
        
        spark.stop()
        
    except Exception as e:
        print(f"✗ Error accessing data lake: {e}")
        print("  Make sure CONNECTION_NAME is set correctly")
        
    except Exception as e:
        print(f"✗ Error accessing data lake: {e}")
        print("  Make sure CONNECTION_NAME is set correctly")

def main():
    """
    Main reset function
    """
    print("=" * 60)
    print("Workshop Environment Reset")
    print("=" * 60)
    
    USERNAME = os.environ.get("PROJECT_OWNER", "default_user")
    print(f"\nUser: {USERNAME}")
    
    # Confirmation
    print("\n⚠️  WARNING: This will delete:")
    print("  - Local data/ folder")
    print(f"  - Database: {DATALAKE_CONFIG['database_name']}_{USERNAME}".upper())
    print("  - All tables in that database")

    
    response = input("\nContinue? (yes/no): ").strip().lower()
    
    if response != "yes":
        print("\nReset cancelled.")
        return
    
    # Step 1: Delete local data
    print("\n" + "-" * 60)
    print("Step 1: Delete Local Data")
    print("-" * 60)
    delete_local_data()
    
    # Step 2: Drop database and tables
    print("\n" + "-" * 60)
    print("Step 2: Drop Database and Tables")
    print("-" * 60)
    drop_database_and_tables()
     
    # Summary
    print("\n" + "=" * 60)
    print("Reset Complete!")
    print("=" * 60)
    print("\nYou can now run the workshop from scratch:")
    print("  python module1/01_ingest_data.py")
    print("=" * 60)


if __name__ == "__main__":
    main()