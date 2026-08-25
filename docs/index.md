# Spark Connect Rust Client

<div class="sc-hero" markdown>

A fast, native **Rust** client for **Apache Spark Connect** - and a drop-in
`pyspark` replacement.

<p class="sc-tagline" markdown>
It builds <code>spark.connect</code> protobuf plans, manages the gRPC channel, and
decodes Arrow results in Rust - speaking the same protocol and returning the same
results as the reference client.
</p>

<div class="sc-badges" markdown>
![PyPI](https://img.shields.io/pypi/v/pyspark-client-rust?color=c2410c&label=pyspark-client-rust)
![Spark](https://img.shields.io/badge/Apache%20Spark-4.2.0%2B-c2410c)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
</div>

[Get started](installation.md){ .md-button .md-button--primary }
[Quickstart](quickstart.md){ .md-button }
[View on GitHub](https://github.com/apache/spark-connect-rust){ .md-button }

</div>

!!! tip "Coming from Python?"
    `pyspark-client-rust` is a **faster, drop-in replacement for the
    [`pyspark-client`](https://pypi.org/project/pyspark-client/) package** on PyPI.
    Uninstall any existing `pyspark` / `pyspark-client`, `pip install
    pyspark-client-rust`, and your Spark Connect code runs **unchanged** - same
    imports, same API, same server (see [Installation](installation.md)). Use it
    exactly like [PySpark](https://spark.apache.org/docs/latest/api/python/); the
    rest of these docs cover the **native Rust API**.

## Why

The reference Spark Connect client is pure Python - it builds protobuf plans,
manages the gRPC channel, and decodes Arrow results in Python. This project moves
that work into **Rust**: a synchronous, PySpark-shaped DataFrame API you can use
directly from Rust, and - through PyO3 - a byte-for-byte compatible `pyspark`
package for the Python world.

<div class="sc-grid" markdown>

<div class="sc-card" markdown>
### Native Rust API
A synchronous `spark_connect` crate that mirrors PySpark's DataFrame, Column,
functions, SQL, streaming, and catalog surface.
</div>

<div class="sc-card" markdown>
### Same protocol & results
Speaks the same `spark.connect` gRPC/protobuf protocol against the same server,
validated against the official Apache Spark test suite.
</div>

<div class="sc-card" markdown>
### Arrow-native results
Results decode through Apache Arrow; optionally convert to
[DataFusion](https://datafusion.apache.org/) or [Polars](https://pola.rs/).
</div>

<div class="sc-card" markdown>
### Rust UDFs via WebAssembly
Compile a Rust function to WASM and run it as a Spark UDF on the executors - no
server-side plugin. See [Rust UDFs](udfs.md).
</div>

<div class="sc-card" markdown>
### Drop-in `pyspark`
A faster replacement for the `pyspark-client` PyPI package - existing Python code
runs unchanged.
</div>

<div class="sc-card" markdown>
### Spark 4.2.0+
The crate and wheel version tracks the Spark release it targets, so the version
number tells you which Spark it speaks.
</div>

</div>

## Hello, Spark Connect (in Rust)

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

## Where to next

- **[Installation](installation.md)** - add the Rust crate, or `pip install` the drop-in.
- **[Quickstart](quickstart.md)** - connect to a server and run your first query.
- **[DataFrames](dataframes.md)** · **[Columns and Functions](columns-and-functions.md)** · **[SQL](sql.md)** - the core API.
- **[Reading and Writing](data-sources.md)** · **[Structured Streaming](streaming.md)** · **[Catalog](catalog.md)**.
- **[Rust UDFs via WebAssembly](udfs.md)** - run Rust functions on the executors.
- **[Architecture](architecture.md)** - how the client is put together.
