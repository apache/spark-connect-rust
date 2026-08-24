<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# Apache Spark Connect Client for Rust

A native Rust client for **Apache Spark Connect**: it builds `spark.connect`
protobuf plans, manages the gRPC channel, and decodes Arrow results, exposing a
synchronous DataFrame API that mirrors PySpark's surface.

## Architecture

A Cargo workspace of three library crates:

Crate names are the crates.io package names; the import path (`use spark_connect::…`) is unchanged via `[lib] name`.

| Crate (crates.io) | Path | Responsibility |
|---|---|---|
| `apache-spark-connect-proto` | `crates/spark-connect-proto` | gRPC/protobuf codegen for `spark.connect.*` |
| `apache-spark-connect-core` | `crates/spark-connect-core` | Transport: channel, retries, reattach, artifacts, errors |
| `apache-spark-connect` | `crates/spark-connect` | DataFrame API: session, dataframe, column, functions, plan, group, catalog, window, readwriter, streaming, types |

## Usage

The API is synchronous and mirrors PySpark's surface:

```rust
use spark_connect::session::SparkSession;
use spark_connect::functions as f;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spark = SparkSession::builder()
        .remote("sc://localhost:15002")
        .get_or_create()?;

    let df = spark
        .range(0, 1_000_000)?
        .select(vec![f::col("id") * 2])?
        .filter(f::col("id") % 3 == 0)?;

    println!("Count: {}", df.count()?);
    df.show(10)?;
    Ok(())
}
```

See [`examples/`](examples) for more (SQL, readers/writers, streaming, Delta Lake).

## Optional features

- `datafusion` — `DataFrame::to_datafusion()` converts a collected result into a
  `datafusion::dataframe::DataFrame`.
- `polars` — `DataFrame::to_polars()` converts a collected result into a
  `polars::frame::DataFrame` (bridged via Arrow IPC).

```bash
cargo build -p apache-spark-connect --features datafusion,polars
```

## Rust UDFs on Spark via WebAssembly (experimental)

The `wasm_udf` module runs **Rust** user-defined functions on Spark, distributed
on the executors, without any server-side plugin. A Rust function compiled to
WebAssembly is packaged into a standard Spark `PythonUDF`: the WASM module and
its signature are cloudpickled (**by value**, so nothing needs pre-deploying on
the cluster) into a tiny Python runner that executes the module with `wasmtime`
once per row.

The API mirrors `pyspark.sql.functions.udf` — `udf(...)` returns a
`UserDefinedFunction` you call on columns:

```rust
use spark_connect::functions::col;
use spark_connect::types::DataType;
use spark_connect::wasm_udf::{udf, WasmValType};

// `run(i64) -> i64` compiled from Rust to wasm32.
let wasm = std::fs::read("add_one.wasm")?;
let add_one = udf("add_one", wasm, "run",
    vec![WasmValType::I64], WasmValType::I64, DataType::Long);
df.select(vec![col("id"), add_one.call(vec![col("id")])?.alias("plus_one")])
    .show(20)?;
```

Requirements:

- **Client** (building the command): a Python interpreter with `cloudpickle` and
  `pyspark`, plus the `python/` directory on `PYTHONPATH` so `pyspark_wasm_udf`
  is importable. Configure via `SPARK_CONNECT_PYTHON` /
  `SPARK_CONNECT_WASM_PACKER_PATH`, or `UserDefinedFunction::with_packer`.
- **Executors**: the `wasmtime` Python package installed.

Scope: this prototype supports numeric scalar signatures (`i32`/`i64`/`f32`/`f64`)
and atomic / `StringType` Spark output types. Widening to strings and nested
types via the Arrow-based [`arrow-udf`](https://crates.io/crates/arrow-udf) ABI
is planned follow-up. See `examples/src/wasm_udf.rs`.

## Building & testing

Requires a Rust toolchain and `protobuf-compiler`.

```bash
cargo build            # build the library crates
cargo test             # run unit + golden-proto tests
cargo build -p examples
```

Builder correctness is covered by **golden-proto tests**: plans, expressions, and
all SQL functions are asserted byte-for-byte against captured reference protos
(`tests/golden/`).

## Running a Spark Connect server locally

The repo includes a `docker-compose.yml` that starts a Spark Connect server on
port 15002:

```bash
docker compose up --build -d
```

Or download [Spark 4.2.0](https://spark.apache.org/downloads.html), set
`SPARK_HOME`, and run:

```bash
$SPARK_HOME/sbin/start-connect-server.sh \
  --packages "org.apache.spark:spark-connect_2.13:4.2.0"
```

Sample datasets used by the examples live in `datasets/`.

## Development

```bash
cargo fmt --all
cargo clippy --workspace
```

## License

Apache License 2.0.
