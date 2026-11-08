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

//! WASM UDFs on a live Spark Connect server, whose workers need `pyarrow` and
//! `wasmtime`. Run with `SPARK_REMOTE` set; skipped otherwise.

use spark_connect::column::col;
use spark_connect::row::Value;
use spark_connect::session::SparkSession;
use wasm_udf_tests::udf;

const TABLE: &str =
    "SELECT * FROM VALUES (5L, 2L, 'a', array(1L, 2L)), (7L, NULL, NULL, array(3L)) \
                     AS t(x, y, s, xs) ORDER BY x";

#[test]
fn wasm_udfs_run_on_the_server() {
    let Ok(url) = std::env::var("SPARK_REMOTE") else {
        return;
    };
    let spark = SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .expect("session");
    let rows = spark
        .sql(TABLE)
        .unwrap()
        .select([
            udf::sub_i64(col("x"), col("y")).unwrap(),
            udf::label(col("s"), col("x")).unwrap(),
            udf::sum(col("xs")).unwrap(),
        ])
        .collect()
        .unwrap();
    let rows: Vec<&[Value]> = rows.iter().map(|r| r.values()).collect();
    assert_eq!(
        rows,
        [
            [Value::Long(3), Value::String("a#5".into()), Value::Long(3)],
            [Value::Null, Value::String("#7".into()), Value::Long(3)],
        ]
    );

    spark
        .udf()
        .register("wasm_label", &udf::label_udf())
        .unwrap();
    let rows = spark
        .sql(&format!("SELECT wasm_label(s, x) FROM ({TABLE})"))
        .unwrap()
        .collect()
        .unwrap();
    let labels: Vec<&Value> = rows.iter().map(|r| &r.values()[0]).collect();
    assert_eq!(
        labels,
        [&Value::String("a#5".into()), &Value::String("#7".into())]
    );
}
