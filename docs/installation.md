# Installation

Add the Spark Connect Rust client to your project. The native Rust crate is the primary library; Python support is a drop-in replacement for PySpark.

## Rust (Native Crate)

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
apache-spark-connect = "4.2"
```

### Optional Features

Enable optional features for additional functionality:

```toml
[dependencies]
apache-spark-connect = { version = "4.2", features = ["datafusion", "polars"] }
```

**Requirements:**
- Rust 1.70 or later
- `protobuf-compiler` (e.g., `brew install protobuf` on macOS, `apt-get install protobuf-compiler` on Linux)
- A running Spark Connect server (see [Configuration and Connection](configuration.md))

### Build from Source

To build the native library locally:

```bash
git clone https://github.com/apache/spark-connect-rust
cd spark-connect-rust
cargo build --release
```

**Build requirements:**
- Rust toolchain (`rustup` / `cargo`)
- `protobuf-compiler`

## Python (Drop-in Replacement)

`pyspark-client-rust` is a faster, drop-in replacement for the official `pyspark` Spark Connect client on PyPI. It uses the same `pyspark` import path and public API, but executes plan building, transport, and Arrow decoding in Rust for better performance.

### Installation

If you have an existing `pyspark` or `pyspark-client` installation, uninstall it first to avoid import clashes:

```bash
pip uninstall pyspark pyspark-client -y
pip install pyspark-client-rust
```

**Requirements:**
- Python 3.9 or later
- A running Spark Connect server

### Usage and API Reference

Use it exactly like PySpark. For the complete Python API reference, see the [official PySpark documentation](https://spark.apache.org/docs/latest/api/python/).

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.remote("sc://localhost:15002").getOrCreate()
```

### Build from Source (Python)

To build and install the Python wheel:

```bash
pip install maturin
maturin build --release --out dist
pip install dist/pyspark_client_rust-*.whl
```

**Build requirements:**
- Python 3.9 or later
- Rust toolchain (`rustup` / `cargo`)
- `protobuf-compiler`

## Version and Compatibility

The crate and package versions track Apache Spark: version `4.2.x` supports Spark 4.2.0 and later. The wire protocol is identical to the reference client, so existing Spark Connect code works unchanged.

## Next Steps

- [Quickstart](quickstart.md) - write your first Rust query
- [Configuration and Connection](configuration.md) - connect to a remote or local Spark Connect server
