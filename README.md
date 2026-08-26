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

# Spark Connect Rust Client

A fast, native **Rust** client for **Apache Spark Connect** - and a drop-in
`pyspark` replacement. It builds `spark.connect` protobuf plans, manages the gRPC
channel, and decodes Arrow results in Rust, speaking the same protocol and
returning the same results as the reference client.

[![PyPI](https://img.shields.io/pypi/v/pyspark-client-rust?color=c2410c&label=pyspark-client-rust)](https://pypi.org/project/pyspark-client-rust/)
![Spark](https://img.shields.io/badge/Apache%20Spark-4.2.0%2B-c2410c)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
<!-- Coverage badges are published by .github/workflows/coverage.yml to the `badges` branch. -->
[![Rust coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/apache/spark-connect-rust/badges/coverage-rust.json)](https://github.com/apache/spark-connect-rust/actions/workflows/coverage.yml)
[![Python coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/apache/spark-connect-rust/badges/coverage-python.json)](https://github.com/apache/spark-connect-rust/actions/workflows/coverage.yml)

## 📖 Documentation

**Full documentation lives at
[apache.github.io/spark-connect-rust](https://apache.github.io/spark-connect-rust/)**
- installation, quickstart, the DataFrame / Columns / SQL / Reading &
Writing / Streaming / Catalog / Types API, [Rust UDFs via
WebAssembly](https://apache.github.io/spark-connect-rust/udfs/), and the
[architecture](https://apache.github.io/spark-connect-rust/architecture/).

## Install

**Python** - a faster, drop-in replacement for the
[`pyspark-client`](https://pypi.org/project/pyspark-client/) PyPI package
(uninstall any existing `pyspark` / `pyspark-client` first):

```bash
pip install pyspark-client-rust
```

Your Spark Connect code then runs unchanged; use it exactly like
[PySpark](https://spark.apache.org/docs/latest/api/python/).

**Rust** - the native crate:

```toml
[dependencies]
apache-spark-connect = "4.2"
```

## Quickstart (Rust)

```rust
use spark_connect::{SparkSession, functions as f, lit};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spark = SparkSession::builder()
        .remote("sc://localhost:15002")
        .get_or_create()?;

    let df = spark
        .range(1_000_000)?
        .select(vec![(f::col("id") * lit(2)).alias("x")])
        .filter((f::col("x") % lit(3)).eq(lit(0)));

    println!("count = {}", df.count()?);
    df.show(20)?;
    Ok(())
}
```

See the [documentation](https://apache.github.io/spark-connect-rust/) for the full
API, running a Spark Connect server, and more.

## API Parity & Drop-in Replacement

The Python drop-in (`pyspark-client-rust`) achieves **100% public-API parity with
PySpark 4.2.0**. The Rust client (crate `apache-spark-connect`) has the same
parity and supports the complete DataFrame, SQL, Streaming, Catalog, and Type
system. The two internal APIs intentionally absent (`Column.to_plan` and
`SparkSession.client`) are implementation details and not part of the public
interface.

## User-Defined Functions (UDFs)

Write UDFs as plain Rust functions and use them right away — define and call
in one file, the way UDFs work in other languages. `#[spark_wasm_udf]` infers
each Spark signature from the Rust types, compiles the module to WebAssembly,
embeds it, and generates a `udf::*()` constructor per function. The runner is
cloudpickled by value, so executors need only the `wasmtime` Python package —
nothing to pre-deploy:

```rust
use spark_connect::functions::col;
use spark_connect::{SparkSession, SparkSessionBuilder};
use spark_connect_macros::spark_wasm_udf;

// The UDFs — plain Rust functions. That's all you write.
#[spark_wasm_udf]
mod udfs {
    pub fn add_one(x: i64) -> i64 { x + 1 }
    pub fn shout(s: String) -> String { format!("{}!", s.to_uppercase()) }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spark: SparkSession =
        SparkSessionBuilder::default().remote("sc://localhost:15002").get_or_create()?;

    // DataFrame API — pass columns straight in (one per argument, arity checked
    // at compile time). No wasm bytes or signatures to spell out by hand.
    spark
        .range(5)?
        .select(vec![col("id"), udf::add_one(col("id"))?.alias("id_plus_one")])
        .show(20)?;

    // ...or register by name and call from SQL (mirrors spark.udf.register):
    spark.udf().register("shout", &udf::shout_udf())?;
    spark.sql("SELECT shout(name) FROM people")?.show(20)?;
    Ok(())
}
```

A one-line `build.rs` compiles the module for `wasm32` and embeds it:

```rust
// build.rs
fn main() { spark_connect_build::embed_wasm_udf("src/main.rs"); }
```

See the [WASM UDF guide](https://apache.github.io/spark-connect-rust/udfs/) for
supported types, non-deterministic UDFs, and the lower-level factory (load a
prebuilt `.wasm` and spell out the types yourself).

## Contributing

Issues are tracked in ASF JIRA under
[SPARK](https://issues.apache.org/jira/browse/SPARK) (GitHub Issues are disabled).
See the [contributing guide](https://apache.github.io/spark-connect-rust/contributing/).

## License

Apache License 2.0.
