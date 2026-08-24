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
// printing the results as "show(...)"

use spark_connect::functions as F;
use spark_connect::{lit_double, DataType};
use spark_connect::{SparkSession, SparkSessionBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spark: SparkSession = SparkSessionBuilder::default().get_or_create()?;

    let path = "./datasets/people.csv";

    let df = spark
        .read()
        .format("csv")
        .option("header", "True")
        .option("delimiter", ";")
        .load(Some(path));

    // select columns and perform data manipulations
    let df = df.select(vec![
        F::col("name"),
        F::col("age").cast(DataType::Integer).alias("age_int"),
        (lit_double(3.0) + F::col("age_int")).alias("addition"),
    ]);

    let df = df.sort(vec![F::col("name").desc().expression().clone()]);

    df.show(5)?;

    // print results
    // +-----+-------+--------+
    // |name |age_int|addition|
    // +-----+-------+--------+
    // |Jorge|30     |33.0    |
    // |Bob  |32     |35.0    |
    // +-----+-------+--------+

    Ok(())
}
