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

//! Start the embedded JVM server, then run
//! `cargo run -p spark-connect-server --example rust_client`.

use spark_connect::functions as f;
use spark_connect::{col, SparkSession};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let remote = std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://127.0.0.1:15002".into());
    let spark = SparkSession::builder().remote(&remote).get_or_create()?;

    let totals = spark
        .range(1_000)?
        .agg(vec![f::sum(col("id")).alias("total").expression().clone()]);
    let totals = totals.collect()?;
    assert_eq!(totals.len(), 1);
    assert_eq!(
        totals[0]
            .get_by_name("total")
            .and_then(|value| value.as_i64()),
        Some(499_500)
    );
    let answer = spark.sql("SELECT 1 + 1 AS answer")?.collect()?;
    assert_eq!(answer.len(), 1);
    assert_eq!(
        answer[0]
            .get_by_name("answer")
            .and_then(|value| value.as_i64()),
        Some(2)
    );
    println!("Range aggregate: {totals:?}");
    println!("SQL: {answer:?}");

    spark.stop()?;
    Ok(())
}
