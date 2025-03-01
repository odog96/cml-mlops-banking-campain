#****************************************************************************
# (C) Cloudera, Inc. 2020-2023
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
#  Absent a written agreement with Cloudera, Inc. ("Cloudera") to the
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

import os
import sys
import pandas as pd
import requests
import io
import tempfile
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
import cml.data_v1 as cmldata

class BankDataGen:
    '''Class to Generate and Ingest Banking Data'''

    def __init__(self, username, dbname, connectionName):
        self.username = username
        self.dbname = dbname
        self.connectionName = connectionName

    def createSparkConnection(self):
        from pyspark import SparkContext
        SparkContext.setSystemProperty('spark.executor.cores', '2')
        SparkContext.setSystemProperty('spark.executor.memory', '4g')
        import cml.data_v1 as cmldata
        conn = cmldata.get_connection(self.connectionName)
        spark = conn.get_spark_session()
        return spark

    def createDatabase(self, spark):
        spark.sql("CREATE DATABASE IF NOT EXISTS {}".format(self.dbname))
        print("SHOW DATABASES LIKE '{}'".format(self.dbname))
        spark.sql("SHOW DATABASES LIKE '{}'".format(self.dbname)).show()

    def createOrReplace(self, df):
        """
        Method to create or replace the BANK_MARKETING table with a fresh batch
        Ensures the table is overwritten, not appended
        """
        # Define the table name
        table_name = "{0}.BANK_MARKETING_{1}".format(self.dbname, self.username)
        
        # Explicitly drop the table if it exists (optional but ensures a clean slate)
        spark = df.sparkSession
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create or replace with the new data
        df.writeTo(table_name)\
          .using("iceberg")\
          .tableProperty("write.format.default", "parquet")\
          .createOrReplace()
        print(f"Table {table_name} created or replaced with fresh data")

    def validateTable(self, spark):
        print("SHOW TABLES FROM {}".format(self.dbname))
        spark.sql("SHOW TABLES FROM {}".format(self.dbname)).show()

    def ingestRawData(self, spark, local_path="data/bank-full.csv"):
        import os
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"CSV file not found at {local_path}. Please run the download script first.")
        raw_df = spark.read.option("delimiter", ";").option("header", "true").csv(local_path)
        print("Raw Data Schema:")
        raw_df.printSchema()
        print(f"Raw Data Count: {raw_df.count()}")
        self.createOrReplace(raw_df)
        print(f"Ingested raw data into {self.dbname}.BANK_MARKETING_{self.username}")

def main():
    USERNAME = os.environ["PROJECT_OWNER"]
    DBNAME = "BNK_MLOPS_HOL_{}".format(USERNAME)
    CONNECTION_NAME = os.environ["CONNECTION_NAME"]
    
    bank_gen = BankDataGen(USERNAME, DBNAME, CONNECTION_NAME)
    spark = bank_gen.createSparkConnection()
    spark.conf.set("spark.sql.catalog.{}.warehouse".format(DBNAME), "s3://your-bucket/bank-marketing/raw/")
    bank_gen.createDatabase(spark)
    bank_gen.ingestRawData(spark, local_path="data/bank-full.csv")
    bank_gen.validateTable(spark)
    spark.sql(f"SELECT * FROM {DBNAME}.BANK_MARKETING_{USERNAME} LIMIT 5").show()
    spark.stop()

if __name__ == '__main__':
    main()
