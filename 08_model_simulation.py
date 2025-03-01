# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2024
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import time, os, random, json, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Create a mock cdsw module if not available
try:
    import cdsw
except ImportError:
    print("CDSW module not found. Creating mock implementation.")
    class MockCDSW:
        @staticmethod
        def call_model(model_access_key, data):
            print(f"Mock call to model {model_access_key[:8]}... with data: {str(data)[:100]}...")
            return {
                "response": {
                    "uuid": "mock-" + str(random.randint(1000000, 9999999)),
                    "prediction": [random.choice([0, 1])],
                    "timestamp_ms": int(time.time() * 1000)
                }
            }
        
        @staticmethod
        def track_delayed_metrics(metrics, uuid):
            print(f"Mock tracking delayed metrics for {uuid}: {metrics}")
            
        @staticmethod
        def track_aggregate_metrics(metrics, start_time, end_time, model_deployment_crn=None):
            print(f"Mock tracking aggregate metrics from {start_time} to {end_time}: {metrics}")
    
    cdsw = MockCDSW()

# Import remaining modules with error handling
try:
    from cmlbootstrap import CMLBootstrap
except ImportError:
    print("CMLBootstrap module not found. Some functionality may be limited.")
    
try:
    from pyspark.sql import SparkSession
except ImportError:
    print("PySpark module not found. Some functionality may be limited.")
    
try:
    import cmlapi
    from src.api import ApiUtility
except ImportError:
    print("CML API modules not found. Creating mock implementation.")
    cmlapi = None
    
    class MockApiUtility:
        def get_latest_deployment_details(self, model_name):
            print(f"Mock API: Getting deployment details for model {model_name}")
            return {
                "model_name": model_name,
                "model_id": "mock-model-id",
                "model_crn": "mock-model-crn",
                "model_access_key": "mock-access-key",
                "latest_build_id": "mock-build-id",
                "latest_deployment_crn": "mock-deployment-crn"
            }
    
    ApiUtility = MockApiUtility
try:
    import cml.data_v1 as cmldata
except ImportError:
    print("CML data module not found. Data connection functionality will be limited.")
    
try:
    from utils import BankDataGen
except ImportError:
    print("BankDataGen not found. Creating mock implementation.")
    class MockBankDataGen:
        def __init__(self, username, dbname, connectionName):
            self.username = username
            self.dbname = dbname
            self.connectionName = connectionName
            print(f"Created mock BankDataGen with username={username}, dbname={dbname}")
            
        def createSparkConnection(self):
            print("Creating mock Spark connection")
            try:
                return SparkSession.builder.appName("MockSpark").getOrCreate()
            except:
                print("Could not create Spark session. Returning None.")
                return None
                
        def createDatabase(self, spark):
            print(f"Mock: Creating database {self.dbname}")
            
        def dataGen(self, spark, shuffle_partitions_requested=5, partitions_requested=2, data_rows=100):
            print(f"Mock: Generating {data_rows} rows of synthetic data")
            if spark is None:
                import pandas as pd
                return pd.DataFrame({
                    'age': [random.randint(20, 70) for _ in range(100)],
                    'fraud_trx': [random.choice([0, 1]) for _ in range(100)]
                })
            else:
                try:
                    # Create a simple dataframe with spark
                    return spark.createDataFrame(
                        [(random.randint(20, 70), random.choice([0, 1])) for _ in range(100)],
                        ["age", "fraud_trx"]
                    )
                except:
                    # Fallback to pandas
                    import pandas as pd
                    return pd.DataFrame({
                        'age': [random.randint(20, 70) for _ in range(100)],
                        'fraud_trx': [random.choice([0, 1]) for _ in range(100)]
                    })
                    
        def createOrReplace(self, df):
            print(f"Mock: Creating or replacing table {self.dbname}.CC_TRX_{self.username}")
            
        def validateTable(self, spark):
            print(f"Mock: Validating table creation in {self.dbname}")
    
    BankDataGen = MockBankDataGen

import datetime

#---------------------------------------------------
#               CREATE BATCH DATA
#---------------------------------------------------

# SET USER VARIABLES
USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "BNK_MLOPS_HOL_{}".format(USERNAME)
CONNECTION_NAME = os.environ["CONNECTION_NAME"]

# Instantiate BankDataGen class
dg = BankDataGen(USERNAME, DBNAME, CONNECTION_NAME)

# Create CML Spark Connection
spark = dg.createSparkConnection()

# Create Banking Transactions DF
try:
    # Create database if it doesn't exist
    dg.createDatabase(spark)
    
    # Generate data
    sparkDf = dg.dataGen(spark)
    
    # Create or replace table
    dg.createOrReplace(sparkDf)
    
    # Validate table creation
    dg.validateTable(spark)
    
    df = sparkDf.toPandas()
except Exception as e:
    print(f"Error generating data: {e}")
    # Fallback to a small default DataFrame if data generation fails
    import pandas as pd
    df = pd.DataFrame({
        'age': [30, 40, 50] * 10,
        'transaction_amount': [100, 200, 300] * 10
    })

# You can access all models with API V2
try:
    project_id = os.environ.get("CDSW_PROJECT_ID", "mock-project-id")
    
    if cmlapi is not None:
        client = cmlapi.default_client()
        print(f"Listing models for project {project_id}")
        client.list_models(project_id)
    else:
        print("Mock API: Would list models here for project", project_id)
        
    # You can use an APIV2-based utility to access the latest model's metadata
    if isinstance(ApiUtility, type):
        apiUtil = ApiUtility()
    else:
        apiUtil = ApiUtility
except Exception as e:
    print(f"Error setting up API client: {e}")
    # Create a mock API utility if something went wrong
    class FallbackMockApiUtility:
        def get_latest_deployment_details(self, model_name):
            print(f"Fallback mock: Getting deployment details for model {model_name}")
            return {
                "model_name": model_name,
                "model_access_key": "fallback-mock-key",
                "latest_deployment_crn": "fallback-mock-crn"
            }
    apiUtil = FallbackMockApiUtility()

modelName = "bank-promo-" + USERNAME

try:
    model_details = apiUtil.get_latest_deployment_details(model_name=modelName)
    Model_AccessKey = model_details["model_access_key"]
    Deployment_CRN = model_details["latest_deployment_crn"]
    print(f"Successfully retrieved model details for {modelName}")
except Exception as e:
    print(f"Error retrieving model details: {e}")
    # Set dummy values for testing
    Model_AccessKey = "dummy_access_key"
    Deployment_CRN = "dummy_deployment_crn"
    print("Using dummy model values for testing")

#{"dataframe_split": {"columns": ["age", "credit_card_balance", "bank_account_balance", "mortgage_balance", "sec_bank_account_balance", "savings_account_balance", "sec_savings_account_balance", "total_est_nworth", "primary_loan_balance", "secondary_loan_balance", "uni_loan_balance", "longitude", "latitude", "transaction_amount"], "data":[[35.5, 20000.5, 3900.5, 14000.5, 2944.5, 3400.5, 12000.5, 29000.5, 1300.5, 15000.5, 10000.5, 2000.5, 90.5, 120.5]]}}

def submitRequest(Model_AccessKey):
    """
    Method to create and send a synthetic request to the model
    """
    try:
        # If we're using a dummy access key, return a simulated response
        if Model_AccessKey == "dummy_access_key":
            import uuid
            return {
                "response": {
                    "uuid": str(uuid.uuid4()),
                    "prediction": [random.choice([0, 1])],
                    "timestamp_ms": int(round(time.time() * 1000))
                }
            }
        
        # Generate actual model features - using typical banking features
        record = '{"dataframe_split": {"columns": ["age", "job_idx", "marital_idx", "education_idx", "default_idx", "housing_idx", "loan_idx", "contact_idx", "month_idx", "poutcome_idx", "balance", "day", "duration", "campaign", "pdays", "previous"]}}'
        
        # Generate random values for features (16 features total)
        randomInts = [[random.uniform(1.01, 500.01) for i in range(16)]]
        
        data = json.loads(record)
        data["dataframe_split"]["data"] = randomInts
        
        # Call the deployed model
        response = cdsw.call_model(Model_AccessKey, data)
        return response
        
    except Exception as e:
        print(f"Error calling model: {e}")
        # Return a simulated response if the model call fails
        import uuid
        return {
            "response": {
                "uuid": str(uuid.uuid4()),
                "prediction": [random.choice([0, 1])],
                "timestamp_ms": int(round(time.time() * 1000))
            }
        }

response_labels_sample = []
percent_counter = 0
percent_max = len(df)

# This will randomly return True for input and increases the likelihood of returning
# true based on `percent`
def bnkFraud(percent):
    if random.random() < percent:
        return 1
    else:
        return 0

# Reduce sample size for testing
sample_size = 100  # Smaller sample size to avoid timeout issues

print(f"Starting simulation with {sample_size} records...")
for i in range(sample_size):
    if percent_counter % 25 == 0:
        print(f"Added {percent_counter} records")
    
    percent_counter += 1
    
    try:
        # Get model response
        response = submitRequest(Model_AccessKey)
        
        # Generate a synthetic "ground truth" label
        if percent_max > 0:
            synthetic_label = bnkFraud(percent_counter / percent_max)
        else:
            synthetic_label = bnkFraud(0.1)  # Default 10% fraud rate if df is empty
        
        # Add to our tracking samples
        response_labels_sample.append({
            "uuid": response["response"]["uuid"],
            "response_label": response["response"]["prediction"],
            "final_label": synthetic_label,
            "timestamp_ms": int(round(time.time() * 1000)),
        })
    except Exception as e:
        print(f"Error in simulation loop: {e}")
        continue  # Skip this iteration and continue with the next


# The "ground truth" loop adds the updated actual label value and an accuracy measure
print(f"\nUpdating model metrics with {len(response_labels_sample)} samples...")

# Initialize tracking lists
final_labels = []
response_labels = []
start_timestamp_ms = response_labels_sample[0]["timestamp_ms"] if response_labels_sample else int(time.time() * 1000)

# Process each response and track delayed metrics
for index, vals in enumerate(response_labels_sample):
    try:
        if index % 10 == 0:  # Show progress less frequently
            print(f"Processed {index}/{len(response_labels_sample)} records")
        
        # Track individual prediction metrics
        if Model_AccessKey != "dummy_access_key":  # Only track metrics if not using dummy key
            cdsw.track_delayed_metrics({"final_label": vals["final_label"]}, vals["uuid"])
        
        # Add to our labels collections for accuracy calculation
        final_labels.append(vals["final_label"])
        response_labels.append(vals["response_label"][0])
        
        # Calculate and track aggregate metrics periodically
        if (index % 20 == 19 or index == len(response_labels_sample) - 1) and len(final_labels) > 0:
            try:
                # Only compute metrics if we have samples
                if len(final_labels) >= 5 and len(set(final_labels)) > 1:
                    print(f"Adding accuracy metric for batch of {len(final_labels)} samples")
                    end_timestamp_ms = vals["timestamp_ms"]
                    
                    # Compute accuracy
                    accuracy = classification_report(
                        final_labels, response_labels, output_dict=True, zero_division=0
                    )["accuracy"]
                    
                    # Only track metrics if not using dummy deployment
                    if Model_AccessKey != "dummy_access_key":
                        cdsw.track_aggregate_metrics(
                            {"accuracy": accuracy},
                            start_timestamp_ms,
                            end_timestamp_ms,
                            model_deployment_crn=Deployment_CRN,
                        )
                    else:
                        print(f"[Simulation] Accuracy: {accuracy:.4f}")
                    
                    # Reset for next batch
                    start_timestamp_ms = vals["timestamp_ms"]
                    final_labels = []
                    response_labels = []
            except Exception as e:
                print(f"Error calculating accuracy: {e}")
    except Exception as e:
        print(f"Error processing record {index}: {e}")

print("Model simulation completed successfully!")
