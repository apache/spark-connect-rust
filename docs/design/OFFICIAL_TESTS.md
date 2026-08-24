# Official Spark Connect Tests

This document describes how to run Apache Spark's official PySpark Connect tests against our drop-in package.

## Overview

The **Official Tests gate** validates that our Rust-backed PySpark Connect client achieves **parity** with Apache Spark's reference implementation by running the entire official test suite (371 test modules across 4 test categories).

- **Manifest**: `tests/official/connect_test_modules.txt` lists all 371 official test modules
- **Script**: `scripts/run_official_connect_tests.py` runs tests against a Connect server
- **Requirements**: `requirements-connect-test.txt` specifies pip dependencies

## Test Modules

The manifest (`tests/official/connect_test_modules.txt`) is organized by module category:

1. **pyspark-connect** (96 modules)
   - Core SQL/Connect doctests and unit tests
   - Arrow UDF tests, client tests, parity tests

2. **pyspark-ml-connect** (24 modules)
   - ML/Connect doctests and unit tests
   - Feature engineering, classification, regression, clustering, pipelines

3. **pyspark-pandas-connect** (131 modules)
   - Pandas-on-Spark Connect tests
   - DataFrame operations, type operations, plotting, reshaping, groupby, I/O

4. **pyspark-pandas-slow-connect** (120 modules)
   - Slower Pandas-on-Spark tests (indexes, diff_frames_ops, groupby operations)

**Total: 371 test modules**

## Running Tests Locally

### Prerequisites

1. **Spark source** at `/path/to/spark` (or specify via `--spark`)
2. **Spark Connect server** running on `sc://localhost:15002` (default)
3. **Dependencies installed**: `pip install -r requirements-connect-test.txt`
4. **Our package on PYTHONPATH** (or use `--target-pyspark /path/to/python`)

### Quick Test (Single Module)

```bash
python3 scripts/run_official_connect_tests.py \
    --spark /path/to/spark \
    pyspark.sql.tests.connect.test_connect_basic
```

### Run All Official Tests

```bash
python3 scripts/run_official_connect_tests.py \
    --spark /path/to/spark \
    --modules-file tests/official/connect_test_modules.txt
```

### Against Our Drop-In Package

```bash
python3 scripts/run_official_connect_tests.py \
    --spark /path/to/spark \
    --target-pyspark python/pyspark \
    --modules-file tests/official/connect_test_modules.txt
```

## Running Tests in CI

The CI workflow integrates this test gate following Spark's `build_python_connect.yml` approach:

### CI Setup Steps

1. **Build our wheel** from Python package
2. **Install wheel** into CI environment
3. **Remove bundled pyspark.zip/py4j** to ensure pure-client mode
4. **Point PYTHONPATH** at our installed package
5. **Start Spark 4.2.0 Connect server** (configured as needed)
6. **Run test script** against the manifest

### CI Command (Reference)

```bash
# Install deps
pip install -r requirements-connect-test.txt
pip install dist/pyspark_client_rust-*.whl

# Remove bundled Java libraries to force pure-client mode
rm -rf $(python3 -c "import pyspark; print(pyspark.__path__[0])")/{lib,pyspark.zip}

# Start Connect server (in background)
export PYTHONPATH="<spark>/python"
./sbin/start-connect-server.sh --driver-java-options "..." ...

# Run ALL tests from manifest
python3 scripts/run_official_connect_tests.py \
    --spark /path/to/spark \
    --target-pyspark $(python3 -c "import pyspark; print(pyspark.__path__[0])") \
    --modules-file tests/official/connect_test_modules.txt

# Stop server
./sbin/stop-connect-server.sh
```

## Script Usage

### run_official_connect_tests.py

```
usage: run_official_connect_tests.py [-h] --spark SPARK [--remote REMOTE]
                                      [--target-pyspark TARGET_PYSPARK]
                                      [--modules-file MODULES_FILE]
                                      [names ...]

Arguments:
  --spark SPARK
      Path to Spark source checkout (required)
  
  --remote REMOTE
      Spark Connect server endpoint
      Default: sc://localhost:15002
  
  --target-pyspark TARGET_PYSPARK
      Directory to prepend to PYTHONPATH
      Use for testing our drop-in replacement package
      Optional: if not set, uses Spark's pyspark from PYTHONPATH
  
  --modules-file MODULES_FILE
      File with list of test modules to run (one per line)
      Supports # comments and blank lines
      Example: tests/official/connect_test_modules.txt
  
  names
      Test module names (positional)
      Use either this OR --modules-file, not both
      Example: pyspark.sql.tests.connect.test_connect_basic
```

### Output

The script reports:
- Each module name and file path as it runs
- Pass/fail status via pytest return code
- Summary at the end:
  ```
  ======================================================================
  Test Summary: 371 modules
    Passed:    365
    Failed:    5
    Not found: 1
  ======================================================================
  ```

## Environment Variables

The script sets these for test execution:

| Variable | Value | Purpose |
|----------|-------|---------|
| `SPARK_CONNECT_TESTING_REMOTE` | `sc://localhost:15002` | Connect server endpoint |
| `SPARK_TESTING` | `1` | Enable Spark test mode |
| `SPARK_CONNECT_MODE_ENABLED` | `1` | Force pure-client mode (no local JVM) |
| `PYTHONPATH` | Target + Spark + original | Prioritize our package |

## Troubleshooting

### "test file not found: X"

The module name might not match the actual file. Check:
1. Module is listed in `tests/official/connect_test_modules.txt`
2. Module exists under `<spark>/python/pyspark/**/tests/connect/**/`
3. Module name matches test file (e.g., `test_connect_basic` → `test_connect_basic.py`)

### "Connection refused" to Spark Connect server

The server must be running on the endpoint specified by `--remote`:
1. Verify server is started
2. Check endpoint matches (default: `sc://localhost:15002`)
3. Use `--remote sc://...` if on different host/port

### "ModuleNotFoundError" for our package

Our package must be on PYTHONPATH:
1. Use `--target-pyspark /path/to/python` if not installed
2. Or: `export PYTHONPATH="/path/to/python:$PYTHONPATH"` before running
3. Ensure wheel is installed correctly in CI

### Tests pass locally but fail in CI

Check environment differences:
1. **Python version**: Must match (3.11 in official CI)
2. **Spark version**: Must be 4.2.0+
3. **Dependencies**: All versions in `requirements-connect-test.txt` installed
4. **Connect server**: Same configuration as local testing

## Integration with docs/design/ACCEPTANCE.md

This gate is part of the broader **Acceptance Criteria**:
- **Baseline**: Validates official tests pass with reference client
- **Gating**: Required before shipping our package
- **Cadence**: Runs in CI on every merge

See `docs/design/ACCEPTANCE.md` for the full acceptance criteria framework.

## Files

- `tests/official/connect_test_modules.txt` - Manifest of all 371 test modules
- `scripts/run_official_connect_tests.py` - Script to execute tests
- `requirements-connect-test.txt` - Pip dependencies
- `docs/design/OFFICIAL_TESTS.md` - This document

## References

- Apache Spark: https://github.com/apache/spark
- Reference workflow: `.github/workflows/build_python_connect.yml`
- Module definitions: `dev/sparktestsupport/modules.py`
