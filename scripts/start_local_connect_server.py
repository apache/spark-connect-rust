"""Start an embedded Spark Connect server on sc://localhost:15002 for tests.

Runs a regular local Spark driver with the SparkConnectPlugin, which binds the
gRPC server. Stays alive until killed. Uses the installed pyspark 4.0 jars.
"""

import time

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder.master("local[*]")
    .appName("spark-connect-rust-testserver")
    .config("spark.plugins", "org.apache.spark.sql.connect.SparkConnectPlugin")
    .config("spark.connect.grpc.binding.port", "15002")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
print("SPARK_CONNECT_SERVER_READY sc://localhost:15002", flush=True)
while True:
    time.sleep(3600)
