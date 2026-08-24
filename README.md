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

Run **Rust** user-defined functions on Spark, distributed on the executors,
without any server-side plugin. A Rust function compiled to WebAssembly is
packaged into a standard Spark `PythonUDF`: the WASM module and its signature
are cloudpickled (**by value**, so nothing needs pre-deploying on the cluster)
into a tiny Python runner that executes the module with `wasmtime`.

This is **opt-in** — enable the `wasm-udf` feature. Users who don't run Rust
UDFs pull none of the dependencies and need none of the preconditions below.

```toml
apache-spark-connect  = { version = "4.2", features = ["wasm-udf"] }
apache-spark-connect-macros = "4.2"          # the #[spark_wasm_udf] macro
apache-spark-connect-build  = "4.2"          # build.rs helper (build-dependency)
```

### Write plain Rust functions

Annotate a module of functions with `#[spark_wasm_udf]`. For each one it infers
the Spark signature from the Rust types, exports it to WASM, and generates a
self-contained constructor under `udf::<name>()` (the compiled module is
embedded). A one-line `build.rs` compiles the module:

```rust
// src/main.rs
use spark_connect_macros::spark_wasm_udf;

#[spark_wasm_udf]
mod udfs {
    pub fn add_one(x: i64) -> i64 { x + 1 }                         // (Long) -> Long
    pub fn shout(s: String) -> String { format!("{}!", s.to_uppercase()) }
    pub fn sum(xs: Vec<i64>) -> i64 { xs.iter().sum() }             // ArrayType arg
    pub fn double_or_null(x: Option<i64>) -> Option<i64> { x.map(|v| v * 2) }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use spark_connect::functions::col;
    use spark_connect::SparkSessionBuilder;
    let spark = SparkSessionBuilder::default().remote("sc://localhost:15002").get_or_create()?;
    spark.range(5)?
        .select(vec![col("id"), udf::add_one().call(vec![col("id")])?.alias("plus_one")])
        .show(20)?;
    Ok(())
}
```

```rust
// build.rs
fn main() { spark_connect_build::embed_wasm_udf("src/main.rs"); }
```

Run it (the UDFs and client are in one file; `wasm-udf-inline/` is this example):

```bash
rustup target add wasm32-unknown-unknown
cargo run -p wasm-udf-inline
```

Two compiles are unavoidable — WASM is what ships to the executors — but the
`build.rs` helper does the `wasm32` compile of the *same source* and embeds it,
so there is no manual WASM step and no `.wasm` file to load. From another crate
the constructors are `wasm_udfs::udf::add_one()` (see `wasm-udfs/` +
`examples/src/wasm_udf_macro.rs`).

### Supported types

Arguments and results cross the WASM boundary with a length-prefixed binary ABI
(`spark_connect::wasm_udf::AbiType`), inferred from the Rust signature:

| Rust        | Spark SQL type            |
|-------------|---------------------------|
| `i32`       | `IntegerType`             |
| `i64`       | `LongType`                |
| `f32`       | `FloatType`               |
| `f64`       | `DoubleType`              |
| `bool`      | `BooleanType`             |
| `String`    | `StringType`              |
| `Vec<u8>`   | `BinaryType`              |
| `Vec<T>`    | `ArrayType` (of `T`)      |
| `Option<T>` | nullable `T`              |

These nest arbitrarily (e.g. `Vec<Option<String>>` → `ArrayType(StringType, nullable)`).

### Preconditions (only when using Rust UDFs)

Nothing here is needed unless the `wasm-udf` feature is enabled:

- **Build machine**: the `wasm32-unknown-unknown` target
  (`rustup target add wasm32-unknown-unknown`), and `apache-spark-connect-macros`
  + `apache-spark-connect-build` as (build-)dependencies.
- **Client** (building the UDF command): a Python interpreter with `cloudpickle`
  and `pyspark`, plus the repo's `python/` directory importable as
  `pyspark_wasm_udf`. Configure via `SPARK_CONNECT_PYTHON` /
  `SPARK_CONNECT_WASM_PACKER_PATH`, or `UserDefinedFunction::with_packer`.
- **Executors**: the `wasmtime` Python package installed.
- **Spark**: 4.2.0+.

### Lower-level API

`spark_connect::wasm_udf::udf(name, module, entrypoint, arg_types, ret_type)`
builds a `UserDefinedFunction` directly (mirrors `pyspark.sql.functions.udf`)
if you prefer to load a prebuilt module and spell out the `AbiType`s yourself.

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
