# Notebook Cell 1: Setup Spark Connection
import os
import cml.data_v1 as cmldata
from pyspark.sql import SparkSession

import matplotlib.pyplot as plt
import seaborn as sns

# Set environment variables if needed (or read them from your project settings)
USERNAME = os.environ.get("PROJECT_OWNER", "default_user")
DBNAME = "BNK_MLOPS_HOL_" + USERNAME

# Connection name should match your Data Lake connection (S3, etc.)
CONNECTION_NAME = os.environ.get("CONNECTION_NAME", "S3 Object Store")

# Create Spark connection via CML Data Connections
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# Optionally set shuffle partitions or other configs if needed
spark.conf.set("spark.sql.shuffle.partitions", "5")

# Notebook Cell 2: Read from the Iceberg Table
# Replace "CC_TRX_<username>" with your actual table name pattern if needed.
table_name = f"{DBNAME}.CC_TRX_{USERNAME}"
df = spark.table(table_name)

# Basic verification: Count number of rows
print("Total Rows:", df.count())

# Notebook Cell 3: Get Basic Summary Statistics
# Using Spark's describe for numerical columns
df.describe().show()



# Notebook Cell 4: Convert a sample to Pandas for more EDA (if data size permits)
# Limit to a small sample to avoid performance issues
pdf = df.limit(3000).toPandas()

# Example: Plot distribution of 'age'
plt.figure(figsize=(8, 4))
sns.histplot(pdf['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()