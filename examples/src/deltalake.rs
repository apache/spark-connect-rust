// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// This example demonstrates creating a Spark DataFrame from a CSV with read options
// and then adding transformations for 'select' & 'sort'
// The resulting dataframe is saved in the `delta` format as a `managed` table
// and `spark.sql` queries are run against the delta table
//
// The remote spark session must have the spark package `io.delta:delta-spark_2.13:{DELTA_VERSION}` enabled.
// Where the `DELTA_VERSION` is the specified Delta Lake version.

use spark_connect::{SparkSession, SparkSessionBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spark: SparkSession = SparkSessionBuilder::default()
        .remote("sc://127.0.0.1:15002/")
        .get_or_create()?;

    // path might vary based on where you started your spark cluster
    // the `/datasets/` folder of spark contains dummy data
    let path = "./datasets/people.csv";

    // Load a CSV file from the spark server
    let df = spark
        .read()
        .format("csv")
        .option("header", "True")
        .option("delimiter", ";")
        .option("inferSchema", "True")
        .load(Some(path));

    // write as a delta table and register it as a table
    df.write()
        .format("delta")
        .mode("overwrite")
        .save_as_table("default.people_delta");

    // view the history of the table
    spark
        .sql("DESCRIBE HISTORY default.people_delta")?
        .show(1)?;

    // create another dataframe
    let df = spark.sql("SELECT 'john' as name, 40 as age, 'engineer' as job")?;

    // append to the delta table
    df.write()
        .format("delta")
        .mode("append")
        .save_as_table("default.people_delta");

    // view history
    spark
        .sql("DESCRIBE HISTORY default.people_delta")?
        .show(2)?;

    // Output should show Delta Lake history with operations and metrics

    Ok(())
}
