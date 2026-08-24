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

//! Single-file Rust UDF on Spark: define the function and use it in one file.
//!
//! This file is compiled twice (unavoidable — WASM is what ships to the
//! executors):
//!   * `build.rs` compiles it to a `wasm32` cdylib, keeping only the exported
//!     `#[spark_wasm_udf]` function, and embeds the result; then
//!   * cargo compiles it normally as the host client below.
//!
//! Run (needs the wasm32 target + a Spark Connect server, `wasmtime` on the
//! executors):
//!
//!     rustup target add wasm32-unknown-unknown
//!     export SPARK_CONNECT_WASM_PACKER_PATH=$PWD/python
//!     cargo run -p wasm-udf-inline

use spark_connect_macros::spark_wasm_udf;

/// The UDF — a plain Rust function. `#[spark_wasm_udf]` exports it in the wasm
/// build and generates `add_one_udf(module)` (with the signature inferred) for
/// the host build below.
#[spark_wasm_udf]
pub fn add_one(x: i64) -> i64 {
    x + 1
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use spark_connect::functions::col;
    use spark_connect::{SparkSession, SparkSessionBuilder};

    // This same file, compiled to wasm32 and embedded by build.rs.
    static WASM_MODULE: &[u8] = include_bytes!(env!("WASM_UDFS_MODULE"));

    let spark: SparkSession = SparkSessionBuilder::default()
        .remote("sc://127.0.0.1:15002/")
        .get_or_create()?;

    let add_one = add_one_udf(WASM_MODULE);

    spark
        .range(5)?
        .select(vec![
            col("id"),
            add_one.call(vec![col("id")])?.alias("id_plus_one"),
        ])
        .show(20)?;

    Ok(())
}
