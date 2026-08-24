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

// This example runs a Rust UDF on Spark, compiled to WebAssembly. The API
// mirrors `pyspark.sql.functions.udf`.
//
// The user's Rust function:
//
//     #[no_mangle]
//     pub extern "C" fn run(x: i64) -> i64 { x + 1 }
//
// compiled with `cargo build --target wasm32-unknown-unknown --release`,
// produces a `.wasm` module exporting `run`. Here we embed a hand-written
// equivalent so the example is self-contained; in practice load your compiled
// module with `std::fs::read("add_one.wasm")`.
//
// Prerequisites:
//   * client: a Python interpreter with `cloudpickle` + `pyspark`, and the
//     repo's `python/` dir on PYTHONPATH. Point the packer at it, e.g.:
//         export SPARK_CONNECT_WASM_PACKER_PATH=$PWD/python
//   * executors: the `wasmtime` Python package installed.

use spark_connect::functions::col;
use spark_connect::types::DataType;
use spark_connect::wasm_udf::{udf, WasmValType};
use spark_connect::{SparkSession, SparkSessionBuilder};

/// A minimal WebAssembly module exporting `run(i64) -> i64` returning `x + 1`.
/// Equivalent WAT:
///   (module (func (export "run") (param i64) (result i64)
///     local.get 0 i64.const 1 i64.add))
const ADD_ONE_WASM: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic + version
    0x01, 0x06, 0x01, 0x60, 0x01, 0x7e, 0x01, 0x7e, // type: (i64) -> (i64)
    0x03, 0x02, 0x01, 0x00, // function: func0 : type0
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00, // export "run" -> func0
    0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x42, 0x01, 0x7c, 0x0b, // code
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spark: SparkSession = SparkSessionBuilder::default()
        .remote("sc://127.0.0.1:15002/")
        .get_or_create()?;

    // Define the Rust/WASM UDF, mirroring `pyspark.sql.functions.udf`: name
    // "add_one", exported entrypoint "run", signature (i64) -> i64, producing a
    // Spark LongType column.
    let add_one = udf(
        "add_one",
        ADD_ONE_WASM,
        "run",
        vec![WasmValType::I64],
        WasmValType::I64,
        DataType::Long,
    );

    let df = spark.range(5)?;

    // Apply it like any other column expression.
    let result = df.select(vec![
        col("id"),
        add_one.call(vec![col("id")])?.alias("id_plus_one"),
    ]);

    result.show(20)?;

    // +---+-----------+
    // | id|id_plus_one|
    // +---+-----------+
    // |  0|          1|
    // |  1|          2|
    // |  2|          3|
    // |  3|          4|
    // |  4|          5|
    // +---+-----------+

    Ok(())
}
