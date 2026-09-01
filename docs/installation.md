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

All three PyPI packages that ship a Spark Connect client - [`pyspark`](https://pypi.org/project/pyspark/), [`pyspark-client`](https://pypi.org/project/pyspark-client/), and `pyspark-client-rust` - install into the **same `pyspark` import directory**, so only one can be active in an environment at a time. Do **not** install `pyspark-client-rust` on top of another one: `pip` does not remove the previous distribution's files first, so you would be left with a mix of both and a broken `pyspark`. Always uninstall the existing client first:

```bash
pip uninstall -y pyspark pyspark-client
pip install pyspark-client-rust
```

!!! warning "Switching back to the reference client"
    For the same reason, **uninstalling `pyspark-client-rust` is not enough** to restore the reference client - it removes the shared `pyspark` files and leaves the environment with no working `pyspark`. Reinstall PySpark explicitly afterward:

    ```bash
    pip uninstall -y pyspark-client-rust
    pip install pyspark-client   # or `pyspark` for full Apache Spark
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
