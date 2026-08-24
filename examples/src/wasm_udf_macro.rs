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

// Using UDFs from a reusable crate. The `wasm-udfs` crate defines plain Rust
// functions with `#[spark_wasm_udf]`; its build.rs compiles + embeds the wasm
// module, so the generated `udf::*()` constructors are self-contained and this
// client just calls them.
//
// Run with (needs the wasm32 target + a Spark Connect server, and `wasmtime`
// on the executors):
//
//     rustup target add wasm32-unknown-unknown
//     export SPARK_CONNECT_WASM_PACKER_PATH=$PWD/python
//     cargo run -p examples --bin wasm_udf_macro --features wasm-udf-macro

use spark_connect::functions::col;
use spark_connect::types::DataType;
use spark_connect::{SparkSession, SparkSessionBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spark: SparkSession = SparkSessionBuilder::default()
        .remote("sc://127.0.0.1:15002/")
        .get_or_create()?;

    let string_type = DataType::String {
        collation: "UTF8_BINARY".to_string(),
    };

    // Constructors from the `wasm-udfs` crate, grouped under `udf::`. Signatures
    // are inferred and the wasm module is embedded — nothing to wire up here.
    spark
        .range(5)?
        .select(vec![
            col("id"),
            wasm_udfs::udf::add_one()
                .call(vec![col("id")])?
                .alias("id_plus_one"),
            wasm_udfs::udf::celsius_to_fahrenheit()
                .call(vec![col("id").cast(DataType::Double)])?
                .alias("as_fahrenheit"),
            wasm_udfs::udf::shout()
                .call(vec![col("id").cast(string_type.clone())])?
                .alias("shouted"),
            wasm_udfs::udf::double_or_null()
                .call(vec![col("id")])?
                .alias("doubled"),
        ])
        .show(20)?;

    Ok(())
}
