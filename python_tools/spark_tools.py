#Description: Converts a Pandas parquet file to a Spark parquet file


#Import Packages
import numpy as np
import pandas as pd
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

#Define Function
def pandas_to_spark_parquet( pandas_file_path, pandas_file_name, spark_file_name):

  #Create Spark session
  sc = SparkContext.getOrCreate()
  spark = SparkSession(sc)

  # Enable Arrow-based columnar data transfers
  spark.conf.set("spark.sql.execution.arrow.enabled", "true")

  #Get file path and name of pandas parquet file
  #pandas_file_path = "..." #Replace with file path of Pandas parquet file
  #pandas_file_name = "..." #Replace with file name of Pandas parquet file

  #New name of file
  #spark_file_name = "..." #Replace with file name of Spark parquet file

  # Generate a Pandas DataFrame, if already a parquet file
  pdf = pd.read_parquet(file_path + file_name)

  # Create a Spark DataFrame from a Pandas DataFrame using Arrow
  df = spark.createDataFrame(pdf)

  #Write File to Domino
  df.write.mode("overwrite").parquet(spark_file_name)
