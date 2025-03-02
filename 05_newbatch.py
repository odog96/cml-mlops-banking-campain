#****************************************************************************
# (C) Cloudera, Inc. 2020-2024
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
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
# #  Author(s): Paul de Fusco / Oliver Zarate
#***************************************************************************/
#****************************************************************************
# Bank Marketing Synthetic Data Generation - PySpark Simple Approach
#***************************************************************************/

import os
import random
from datetime import datetime
import cml.data_v1 as cmldata
from pyspark.sql.functions import col, rand, expr

# Setup
USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = "BNK_MLOPS_HOL_{}".format(USERNAME)
CONNECTION_NAME = os.environ["CONNECTION_NAME"]

# Connect to Spark
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

print("Loading original data from Iceberg table...")
table_name = f"{DBNAME}.BANK_MARKETING_{USERNAME}"
df_spark = spark.table(table_name)

# Store current record count for verification
original_count = df_spark.count()
print(f"Original table record count: {original_count}")

# Define columns for reference
numerical_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
categorical_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
target_col = "y"

# Generate synthetic data - sample with replacement
print("Generating synthetic data by bootstrapping and adding noise...")
num_samples = 10000

# Sample with replacement (bootstrap)
bootstrap_sample = df_spark.sample(withReplacement=True, fraction=num_samples/original_count)

# Add random noise to numerical columns (while keeping them as strings)
synthetic_df = bootstrap_sample
for col_name in numerical_cols:
    # Convert to numeric temporarily
    synthetic_df = synthetic_df.withColumn(
        f"{col_name}_numeric", 
        col(col_name).cast("double")
    )
    
    # Add random noise (±10%)
    synthetic_df = synthetic_df.withColumn(
        f"{col_name}_with_noise", 
        expr(f"{col_name}_numeric * (0.9 + rand() * 0.2)")
    )
    
    # Round and convert back to string
    synthetic_df = synthetic_df.withColumn(
        col_name,
        expr(f"CAST(ROUND({col_name}_with_noise, 0) AS STRING)")
    )
    
    # Drop temporary columns
    synthetic_df = synthetic_df.drop(f"{col_name}_numeric", f"{col_name}_with_noise")

# Limit to exact number of samples
synthetic_df = synthetic_df.limit(num_samples)

# Append to the existing table
print(f"Appending synthetic data to {table_name}...")

# Write as Iceberg table with append mode
synthetic_df.writeTo(table_name) \
    .using("iceberg") \
    .tableProperty("write.format.default", "parquet") \
    .append()

# Verify the data was appended
new_count = spark.table(table_name).count()
print(f"Table record count after append: {new_count}")
print(f"Added {new_count - original_count} records")

print("\nSample of appended data:")
spark.sql(f"SELECT * FROM {table_name} LIMIT 5").show()

print(f"Synthetic data generation and append complete.")

# Close the Spark session
spark.stop()
