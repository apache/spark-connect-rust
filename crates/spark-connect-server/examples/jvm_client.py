# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Run against the Rust listener embedded in the Spark JVM by run-engine.sh."""

import os

from pyspark.sql import SparkSession

spark = SparkSession.builder.remote(
    os.environ.get("SPARK_REMOTE", "sc://127.0.0.1:15002")
).getOrCreate()
try:
    left = spark.range(5).withColumnRenamed("id", "key")
    right = spark.range(3).withColumnRenamed("id", "key")
    print("JVM DataFrame join:", left.join(right, "key").orderBy("key").collect())
    print(
        "JVM complex types:",
        spark.sql(
            "SELECT CAST('12.34' AS DECIMAL(10,2)) AS amount, array(1,2,3) AS items"
        ).collect(),
    )
finally:
    spark.stop()
