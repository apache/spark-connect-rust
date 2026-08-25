<!-- SPDX-License-Identifier: Apache-2.0 -->

# CI/CD Strategy

This document describes the GitHub Actions CI pipeline for the Spark Connect Rust Client, a Rust-backed drop-in replacement for the PySpark Spark Connect client.

## Overview

The CI consists of two main workflows:

1. **Rust CI** (`.github/workflows/rust.yml`) - Build and test the pure-Rust crates
2. **Python Connect Tests** (`.github/workflows/build_python_connect.yml`) - Build the PyO3 wheel and run Spark Connect integration tests

Both workflows run on every push to main/master and on all pull requests.

## Rust CI (`rust.yml`)

**Status**: Expected to be ✅ fully green.

This workflow validates the pure-Rust implementation:

- **Build**: Compiles `spark-connect`, `spark-connect-core`, and `spark-connect-proto` crates (pyspark-rs is excluded from default cargo operations because it requires maturin + Python environment)
- **Test**: Runs `cargo test` on `spark-connect` and `spark-connect-core`
- **No-stub audit**: Runs `scripts/audit_no_stubs.sh` to enforce the "100% coverage, zero stubs" rule (see [Design notes](architecture.md))

### Key steps

1. Checkout
2. Install Rust (stable channel) with caching
3. Install system dependencies (protobuf-compiler)
4. Build workspace (excluding pyspark-rs)
5. Test spark-connect and spark-connect-core crates
6. Audit for placeholder/deferred markers

This job should always pass; failures indicate incomplete implementation or test regressions.

## Python Connect Tests (`build_python_connect.yml`)

**Status**: Expected to start ⚠️ red and improve iteratively as parity features are implemented.

This workflow builds the PyO3 extension and runs the official Apache Spark Connect test suite against our Rust-backed client.

### Strategy

The job is matrix-tested across Python versions (3.9, 3.11) to ensure broad compatibility.

**Expected behavior**:
- Many tests will fail initially - this is intentional and documents the work remaining
- Each failure corresponds to a gap in parity (see `dev/parity/inventory.csv`)
- The workflow remains stable (doesn't flake) because it's deterministic (pinned Spark version)
- As features are implemented and marked `verified` in the parity ledger, test failures will turn green
- **Final goal**: 100% of Connect test suite passing

### Key steps

#### 1. Setup
- Check out the repository
- Set up Python (3.9, 3.11)
- Install Rust (stable)
- Install system dependencies (protobuf-compiler)
- Install build tools (pip, setuptools, wheel, maturin)

#### 2. Build the extension
```bash
maturin build --release --out dist
```
This produces a wheel (`pyspark_client_rust-4.2.0-*.whl`) containing:
- The compiled PyO3 extension module (`_pyspark.so` / `.pyd`)
- The pure-Python drop-in package (`python/pyspark/...`)

#### 3. Install dependencies
```bash
pip install dist/pyspark_client_rust-*.whl
pip install pytest pandas pyarrow numpy grpcio cloudpickle
```

#### 4. Download & start Spark Connect server (4.2.0)
- Downloads official Spark 4.2.0 binary distribution
- Starts the Connect server on `sc://localhost:15002`
- Waits for readiness (nc connectivity check, up to 60 seconds)

#### 5. Clone Apache Spark test suite
- Clones Apache Spark at tag `v4.2.0`
- Test files are discovered from `python/pyspark/**/tests/connect/**/test_*.py`

#### 6. Run Connect test modules
```bash
python scripts/run_official_connect_tests.py \
  --spark <spark-source> \
  --target-pyspark ./python \
  --remote sc://localhost:15002 \
  test_connect_column \
  test_connect_functions \
  test_connect_dataframe \
  test_connect_udf \
  ...
```

The script prepends our package to `PYTHONPATH` and runs pytest against official Spark test files.

#### 7. Artifact upload
- Uploads Connect test log (`/tmp/connect-tests.log`)
- Uploads Connect server log (`/tmp/connect-server.log`)

These logs are available in the workflow run artifacts for debugging failures.

### Expected test modules

The workflow currently runs:
- `test_connect_column` - Column and expression operations
- `test_connect_functions` - SQL functions
- `test_connect_dataframe` - DataFrame operations
- `test_connect_udf` - Python and pandas UDFs

Additional modules can be added as parity improves (e.g., `test_connect_readwriter`, `test_connect_udf_batch`, etc.).

### Handling test failures

Each failing test is a **gap in parity**. The fix process:

1. **Identify the gap**: Read the test failure and corresponding official code
2. **Locate the ledger row**: Find the symbol in `dev/parity/inventory.csv`
3. **Implement the feature**: Code the missing functionality in the Rust crate
4. **Update ledger status**: Mark the row `done` (or `verified` if a golden proto is included)
5. **Verify locally**: Run `scripts/run_official_connect_tests.py` with `--target-pyspark ./python`
6. **Push & verify CI**: The workflow will show the test passing on the next run

## Continuous improvement

The parity ledger (`dev/parity/inventory.csv`) is the source of truth:
- **todo**: Not started
- **wip**: In progress, not testable yet
- **done**: Implemented, test(s) passing but not yet officially verified
- **verified**: Backed by golden-proto match and/or all-green Connect test suite

The CI workflow gates progress:
- Rust CI ensures the core crates are always buildable and testable
- Connect CI documents which official tests are green (and which are red)
- `audit_no_stubs.sh` enforces the "no TODOs" rule before claiming completion

## Running tests locally

To run the Connect test suite locally:

1. **Build and install the wheel**:
   ```bash
   maturin build --release --out dist
   pip install dist/pyspark_client_rust-*.whl
   ```

2. **Start a Spark Connect server** (requires Java + Spark 4.2.0):
   ```bash
   # In one terminal:
   export SPARK_HOME=~/spark-dist/spark-4.2.0-bin-hadoop3
   ${SPARK_HOME}/sbin/start-connect-server.sh
   ```

3. **Run the test suite**:
   ```bash
   pip install pytest
   python scripts/run_official_connect_tests.py \
     --spark /path/to/spark \
     --target-pyspark ./python \
     --remote sc://localhost:15002 \
     test_connect_column
   ```

## References

- [Design notes](architecture.md) - System design, parity strategy, no-stub rule
- [Acceptance criteria](acceptance.md) - Acceptance criteria and definition of done
- [pyproject.toml](https://github.com/apache/spark-connect-rust/blob/master/pyproject.toml) - Maturin build configuration
- [scripts/run_official_connect_tests.py](https://github.com/apache/spark-connect-rust/blob/master/scripts/run_official_connect_tests.py) - Test runner utility
- [scripts/audit_no_stubs.sh](https://github.com/apache/spark-connect-rust/blob/master/scripts/audit_no_stubs.sh) - No-stub audit gate
- [Apache Spark Workflow](https://github.com/apache/spark/blob/master/.github/workflows/build_python_connect.yml) - Reference implementation
