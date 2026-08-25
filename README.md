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

A Rust rewrite of the PySpark **Spark Connect** client (`pyspark.sql.connect.*`),
exposed through PyO3 as a drop-in `pyspark` package. It speaks the same Spark
Connect gRPC/protobuf protocol and returns the same results, so existing
Spark Connect code runs unchanged against the same server - with the plan
building, transport, and Arrow decoding done in Rust.

**Supported Spark version:** Apache Spark **4.2.0 and later**. The crate and wheel
version tracks the Spark release it targets (starting at `4.2.0`), so the version
number tells you which Spark it speaks.

> **Status: alpha, work in progress.** The core DataFrame/Column/functions API,
> transport, and Arrow result path work end-to-end against a real Spark Connect
> server. The crates and wheel carry the version **4.2.0** to track the Apache Spark
> release they target — it is a compatibility marker, not a maturity claim; treat the
> client itself as alpha. API parity with the reference client is validated by the
> golden-proto tests (plan building) and the official Apache Spark connect test suite
> (transport + Arrow); see [Continuous integration](#continuous-integration).

## Why

The reference Spark Connect client is pure Python: it builds protobuf plans,
manages the gRPC channel, and decodes Arrow results in Python. This project moves
that work into Rust while keeping a byte-for-byte compatible Python surface, so it
can be a **drop-in replacement** - same imports, same API, same server.

## Architecture

A Cargo workspace of four crates plus a Python skin:

| Component | Path | Responsibility |
|---|---|---|
| `apache-spark-connect-proto` | `crates/spark-connect-proto` | gRPC/protobuf codegen for `spark.connect.*` |
| `apache-spark-connect-core` | `crates/spark-connect-core` | Transport: channel, retries, reattach, artifacts, errors |
| `apache-spark-connect` | `crates/spark-connect` | DataFrame API: session, dataframe, column, functions, plan, group, catalog, window, readwriter, streaming, udf, types |
| `pyspark-rs` | `crates/pyspark-rs` | PyO3 bindings - builds the `_pyspark` extension module |
| Python skin | `python/pyspark` | The drop-in `pyspark` package (+ vendored `pyspark.pandas`, `cloudpickle`, `pyspark.testing`) |

See [`docs/design/ARCHITECTURE.md`](docs/design/ARCHITECTURE.md) for detail.

## Installation

`pyspark-client-rust` is a **complete drop-in** for the reference `pyspark`
Spark Connect client: install it and existing Spark Connect code runs unchanged
— same imports, same API, same server.

```bash
pip install pyspark-client-rust
```

### Build from source

Requires a Rust toolchain, `protobuf-compiler`, and Python 3.9+.

```bash
# Build the wheel (mixed Python/Rust layout via maturin) and install it:
maturin build --release --out dist
pip install dist/pyspark_client_rust-*.whl
```

For local development without maturin, build the extension and copy it into the skin:

```bash
cargo build -p pyspark-rs --release
cp target/release/lib_pyspark.dylib python/pyspark/_pyspark.so   # .so/.dylib per platform
PYTHONPATH=python python -c "from pyspark.sql import SparkSession; ..."
```

## Usage

Identical to the reference Spark Connect client:

```python
from pyspark.sql import SparkSession, functions as sf

spark = SparkSession.builder.remote("sc://localhost:15002").getOrCreate()

df = (
    spark.range(0, 1_000_000)
    .select((sf.col("id") * 2).alias("x"))
    .filter(sf.col("x") % 3 == 0)
)
print(df.count())
df.groupBy((sf.col("x") % 10).alias("k")).agg(sf.sum("x"), sf.avg("x")).show()
```

### Rust usage (native)

The pure-Rust API is synchronous and mirrors PySpark's surface:

```rust
use spark_connect::session::SparkSession;
use spark_connect::functions as f;

fn main() -> Result<()> {
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

### Rust UDFs via WebAssembly

Python UDFs run as normal. In addition, a Rust function can be compiled to
WebAssembly and shipped as a UDF: the `.wasm` bytes are embedded in a cloudpickled
Python shim and executed on the worker via `wasmtime`, so no Rust toolchain is
needed server-side. See `python/pyspark_wasm_udf/` and `examples/src/wasm_udf.rs`.

## Getting started with Spark Connect server

You can run a Spark Connect server locally for development in two ways:

### Option 1: Docker Compose (recommended)

The repo includes a `docker-compose.yml` that starts a Spark Connect server on port 15002:

```bash
docker compose up --build -d
```

### Option 2: Local Spark distribution

1. [Download Spark 4.2.0](https://spark.apache.org/downloads.html) and unzip it
2. Set `SPARK_HOME` to the unzipped directory
3. Start the server with:

```bash
$SPARK_HOME/sbin/start-connect-server.sh \
  --packages "org.apache.spark:spark-connect_2.13:4.2.0,io.delta:delta-spark_2.13:4.2.0" \
  --conf "spark.driver.extraJavaOptions=-Divy.cache.dir=/tmp -Divy.home=/tmp" \
  --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" \
  --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog"
```

The server listens on `sc://localhost:15002` by default.

### Sample data

The repo includes sample datasets in `datasets/` (people.csv, employees.json, kv1.txt, etc.)
mounted at `/opt/spark/work-dir/datasets` in the Docker container for use in tests and examples.

## Continuous integration

- **Build & test** - Rust build/test via `cargo test`
- **Golden-proto tests** - Validate the plan/expression protos this client builds
  (`plan.rs` and friends) byte-for-byte against the reference client. This is what
  guards plan-building correctness.
- **Parity gate** - Runs the official Apache Spark connect test suite against this
  client. The suite drives the standard pyspark public API but routes plan-building
  through *upstream* pyspark (via `rust_transport_plugin.py`), so it exercises our
  **transport and Arrow result paths** end-to-end against a real server — not our
  own `plan.rs` (that is the golden-proto tests' job). Together the two gates cover
  plan-building and transport respectively.
- **Benchmark** - End-to-end performance vs reference client
- **Lint** - `rustfmt --check`, `clippy`, and `ruff` for Python tooling

## Development

```bash
cargo fmt --all                 # format Rust
cargo clippy --workspace        # lint Rust
ruff check scripts/             # lint our Python tooling
ruff format scripts/
```

`ruff` is intentionally scoped to `scripts/`; the `python/pyspark` tree is largely
vendored from Apache Spark and kept byte-compatible with upstream.

Remaining parity work is tracked mechanically in the parity ledger at
`docs/parity/inventory.csv`; the official Apache Spark connect test suite is the
authoritative gate.

## License

Apache License 2.0.
