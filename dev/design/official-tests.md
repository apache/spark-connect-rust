# Official Spark Connect Tests

This document describes how to run Apache Spark's official PySpark Connect tests against our
Rust-backed client and our drop-in package.

## Overview

The **Official Tests gate** validates that our Rust-backed PySpark Connect client achieves
**parity** with Apache Spark's reference implementation. It runs the official Spark 4.3.0 Connect
test files against a live Spark 4.3.0 Connect server, in one of two modes:

- **Transport mode** (default): the official test files run on the reference `pyspark`, and our Rust
  transport (`scripts/rust_transport_plugin.py`) replaces its gRPC stub. This is the
  `connect-parity` job.
- **Drop-in mode** (`--dropin`): the official test files run against our own `pyspark` drop-in (the
  Rust-backed DataFrame, Column, functions, readers, ...). This is the `connect-dropin` job.

- **Runner**: `scripts/run_official_tests.py`
- **CI**: `.github/workflows/build_python_connect.yml`
- **Known failures**: `scripts/parity_known_failures.txt` (transport mode) and
  `scripts/dropin_known_failures.txt` (drop-in mode)

## Test Modules

The gate discovers test files on disk. It does not read a module list.

- **Transport mode**: every `test_*.py` under the Spark `python/` tree whose path contains
  `/sql/tests/connect/`. CI passes `--exclude` for the three local-cluster-only basenames
  (`test_artifact.py`, `test_artifact_localcluster.py`, `test_parity_resources.py`), and the
  local-cluster step runs those with `--only`.
- **Drop-in mode**: every `test_*.py` under the Connect suites of `pyspark/sql`,
  `pyspark/ml`, `pyspark/errors` and `pyspark/logger`, minus the same three basenames.

The pandas and pipelines suites are not run by this runner. `pandas-parity.yml` and
`pipelines-parity.yml` run their own pytest commands.

`tests/official/connect_test_modules.txt` is a reference list of 371 module names in four
groups: pyspark-connect (96), pyspark-ml-connect (24), pyspark-pandas-connect (131) and
pyspark-pandas-slow-connect (120). Its header names Spark's `dev/sparktestsupport/modules.py`
as the source. No workflow runs it, and `run_official_tests.py` does not read it. Only
`scripts/run_official_connect_tests.py`, which CI does not call, reads it.

## Running Tests Locally

### Prerequisites

1. **Spark source** at `/path/to/spark`: a v4.3.0 checkout with a `python/` directory (`--spark`)
2. **Spark Connect server** running on `sc://localhost:15002` (the default; override with
   `--remote` or `SPARK_CONNECT_TESTING_REMOTE`)
3. **Python dependencies** as the CI job installs them: pytest, `pandas>=2.2,<3.0`, pyarrow,
   numpy, grpcio, grpcio-status, googleapis-common-protos, cloudpickle, zstandard, py4j,
   `plotly<6.0.0` and `graphviz==0.20.3`
4. **Built extension** at `RUST_PYSPARK_SO`, required in both modes:
   ```bash
   cargo build -p pyspark-rs --release
   cp target/release/lib_pyspark.so python/pyspark/_pyspark.so
   export RUST_PYSPARK_SO=$PWD/python/pyspark/_pyspark.so
   ```
   On macOS, copy the `.dylib` that cargo writes instead.

Our package does not need to be installed. Transport mode imports the reference `pyspark`
from `--spark`, and drop-in mode imports an overlay of our `python/` tree.

### Quick Test (Single Module)

```bash
python3 scripts/run_official_tests.py \
    --spark /path/to/spark \
    --only pyspark.sql.tests.connect.test_parity_dataframe
```

`--only` takes dotted module names or `python/`-relative `.py` paths.

### Run All Official Tests

```bash
python3 scripts/run_official_tests.py \
    --spark /path/to/spark \
    --jobs 1 \
    --exclude "test_artifact.py,test_artifact_localcluster.py,test_parity_resources.py"
```

### Against Our Drop-In Package

```bash
python3 scripts/run_official_tests.py \
    --dropin \
    --spark /path/to/spark \
    --jobs 1
```

## Running Tests in CI

The `connect-parity` and `connect-dropin` jobs in `.github/workflows/build_python_connect.yml`
run this gate. Both use Python 3.11 and JDK 17 (Temurin), run files serially (`--jobs 1`), and
test against a Spark 4.3.0 Connect server.

### CI Setup Steps

1. **Build the extension** with `cargo build -p pyspark-rs --release` and copy it to
   `python/pyspark/_pyspark.so`. No wheel is built or installed, because the reference
   `pyspark` and ours both own the `pyspark` namespace.
2. **Install Python dependencies** inline in the workflow. It does not install
   `requirements-connect-test.txt`.
3. **Download Spark 4.3.0** (`spark-4.3.0-bin-hadoop3-connect.tgz` from archive.apache.org,
   cached in `~/spark-dist`) and start its Connect server with `--driver-memory 3g` and
   `spark.sql.session.localRelationSizeLimit=1073741824`. The job waits up to 90 seconds for
   port 15002. The drop-in job starts the server from `SPARK_HOME`, because the ML tests read
   relative data paths.
4. **Clone Apache Spark v4.3.0** (`--depth 1`, cached in `~/spark-source`) for the test files
   and the reference `pyspark`.
5. **Run the runner** with `RUST_PYSPARK_SO` set. The parity job then restarts the server in
   `local-cluster[2, 4, 1024]` mode and runs the resource and artifact files with `--only`
   and `--no-drift-check`.

### CI Command (Reference)

```bash
# Build the extension and place it where the overlay expects it
cargo build -p pyspark-rs --release
cp target/release/lib_pyspark.so python/pyspark/_pyspark.so

# Start the Connect server (Spark 4.3.0 binary distribution)
"${SPARK_HOME}/sbin/start-connect-server.sh" \
    --driver-memory 3g \
    --conf spark.sql.session.localRelationSizeLimit=1073741824

# Transport mode: official files through our Rust client
RUST_PYSPARK_SO=$PWD/python/pyspark/_pyspark.so \
SPARK_CONNECT_TESTING_REMOTE=sc://localhost:15002 \
python3 scripts/run_official_tests.py \
    --spark ~/spark-source --jobs 1 \
    --exclude "test_artifact.py,test_artifact_localcluster.py,test_parity_resources.py"

# Drop-in mode: official files against our drop-in
RUST_PYSPARK_SO=$PWD/python/pyspark/_pyspark.so \
SPARK_CONNECT_TESTING_REMOTE=sc://localhost:15002 \
python3 scripts/run_official_tests.py --dropin --spark ~/spark-source --jobs 1
```

## Script Usage

### run_official_tests.py

```
usage: run_official_tests.py [-h] --spark SPARK [--remote REMOTE] [--dropin]
                             [--manifest MANIFEST] [--jobs JOBS]
                             [--timeout TIMEOUT] [--retries RETRIES]
                             [--no-drift-check] [--only ONLY] [--exclude EXCLUDE]

Arguments:
  --spark SPARK
      Spark source tree that has a python/ directory (required)

  --remote REMOTE
      Spark Connect server endpoint
      Default: $SPARK_CONNECT_TESTING_REMOTE, else sc://localhost:15002

  --dropin
      Run the official sql, ml, errors and logger Connect suites against our drop-in
      (an overlay of our python/ tree) instead of through the transport

  --manifest MANIFEST
      Known-failure manifest
      Default: scripts/parity_known_failures.txt, or scripts/dropin_known_failures.txt
      with --dropin

  --jobs JOBS
      Files to run in parallel. Default: 1 (CI runs serially)

  --timeout TIMEOUT
      Per-file timeout in seconds. Default: 360

  --retries RETRIES
      Extra attempts for the event-timing-driven files only (FLAKY_FILES). Default: 2

  --no-drift-check
      Skip the post-run check that reports manifest entries that now pass

  --only ONLY
      Run only these tests instead of discovering them. Comma-separated dotted module
      names (pyspark.sql.tests.connect.test_parity_dataframe) or python/-relative .py paths

  --exclude EXCLUDE
      Comma-separated test-file basenames to drop from discovery (ignored with --only)
```

The runner needs `RUST_PYSPARK_SO` in the environment. Exit codes are 0 when no file has
unexpected failures, 1 when a file does, and 2 for setup errors: `RUST_PYSPARK_SO` unset, an
`--only` file not found, or no test files discovered.

### Output

The script prints:
- `Discovered N connect test files; manifest skips X tests + Y whole files.`
- One line per file, in discovery order:
  `[done/total] ok|FAIL|skip(env)  <basename>  p=<passed> f=<failed> e=<errors> skip=<skipped>`,
  with optional ` TIMEOUT`, ` rc=<n>` and ` (after <k> attempts)` suffixes
- A drift check, printed only when a manifest entry now passes. It never fails the run.
- `<n> file(s) with unexpected failures out of <N>.` On failure, the node ids of the
  unexpected failures follow.

When `RUST_PARITY_LOG_DIR` is set, each pytest attempt is kept there, including tracebacks,
retries and drift-check runs.

## Selection, Skips and Retries

- **Known-failure manifest**: a line `<relpath>::Class::method  # reason` deselects that test.
  The plugin matches the node-id suffix, and also a parametrized `[param]` variant. A line
  without `::` skips the whole file. The drop-in manifest uses this for the gRPC client
  internals files. Each manifest's header gives the format.
- **Manifest maintenance**: `scripts/gen_parity_skiplist.py` regenerates the parity manifest,
  and it runs the reference client to do so. Only by-design and environmental entries belong
  in the drop-in manifest, per its header.
- **Drift check**: after the run, every manifest test is re-run in select-only mode. Tests
  that now pass are listed, so the manifest can shrink. The check never fails the gate.
- **Retries**: only the basenames in `FLAKY_FILES` are retried, up to `--retries` extra
  attempts. These are the event-timing-driven streaming files, such as
  `test_parity_streaming.py`, `test_parity_listener.py` and `test_parity_observation.py`.
  All other files run once.
- **Timeouts**: `--timeout` (360 seconds) applies to each file. `test_parity_arrow_python_udf.py`
  and `test_parity_udtf.py` get 1200 seconds.
- **Pass or fail per file**: a file fails when pytest exits nonzero, when it times out, or
  when its summary shows failures or errors. Exit code 5 (no tests collected) counts as ok.
- **Drop-in mode**: the runner removes `SPARK_SKIP_CONNECT_COMPAT_TESTS` from the test
  environment, because the official Arrow UDTF suite skips itself when it is set.

## Environment Variables

The runner sets these for each pytest run:

| Variable | Value | Purpose |
|----------|-------|---------|
| `SPARK_CONNECT_TESTING_REMOTE` | `--remote` value | Connect server endpoint |
| `SPARK_TESTING` | `1` | Enable Spark test mode |
| `PYTHONHASHSEED` | `0` | Fixed hash seed, so ML CrossValidator splits are stable |
| `SPARK_SKIP_CONNECT_COMPAT_TESTS` | `1`, removed in drop-in mode | Remote-only session |
| `PYTHONPATH` | Transport: scripts, Spark python; drop-in: overlay, scripts | Plugins first |
| `RUST_PYSPARK_SO` | Set by the caller | Path to the built `_pyspark` extension |
| `RUST_PARITY_DESELECT`, `RUST_PARITY_SELECT_ONLY` | Set by the runner | Deselect plugin input |
| `RUST_PARITY_LOG_DIR` | Set by the caller (optional) | Keeps each pytest attempt's log |

## Troubleshooting

### "--only test file(s) not found"

The runner exits with status 2 before running anything. Check:
1. The module name maps to a file under `<spark>/python/`. For example,
   `pyspark.sql.tests.connect.test_parity_dataframe` maps to
   `python/pyspark/sql/tests/connect/test_parity_dataframe.py`.
2. The Spark source is the v4.3.0 checkout.

Without `--only`, the runner discovers the files itself, so this error does not occur.

### "Connection refused" to Spark Connect server

The runner does not check the endpoint, so a missing server appears as test failures:
1. Verify the server is started (CI waits up to 90 seconds on port 15002)
2. Check the endpoint matches `--remote` (default `sc://localhost:15002`, or
   `SPARK_CONNECT_TESTING_REMOTE`)
3. Use `--remote sc://...` if the server is on a different host or port

### "set RUST_PYSPARK_SO" or the extension is missing

The runner exits with status 2 when `RUST_PYSPARK_SO` is unset. The transport plugin raises
`RuntimeError` when the path does not exist. Build the extension and export the path, as in
Prerequisites.

### Tests pass locally but fail in CI

1. **Python version**: 3.11 in both CI jobs
2. **Spark version**: the v4.3.0 tag for the test files, and the 4.3.0 bin-hadoop3-connect server
3. **Dependencies**: the list in Prerequisites, as the CI job installs it
4. **Server settings**: `--driver-memory 3g` and `localRelationSizeLimit`. CI runs the gate
   serially (`--jobs 1`).
5. **Timing**: only the streaming files are retried. A timing flake elsewhere fails on its first
   attempt, so a single local run can pass where CI fails.

## Integration with acceptance.md

This gate is part of the broader **Acceptance Criteria** in `acceptance.md`:
- **Baseline**: the gate no longer runs the reference client on every file. The reference
  client runs only when `scripts/gen_parity_skiplist.py` regenerates the manifest.
- **Gating**: the `connect-parity` and `connect-dropin` jobs fail on any unexpected failure
- **Cadence**: runs on every push and pull request to `master` and `branch-*`, and on
  manual dispatch

See `acceptance.md` for the full acceptance criteria framework.

## Files

- `tests/official/connect_test_modules.txt` - Reference list of 371 module names (not read by the
  gate)
- `requirements-connect-test.txt` - Pip pins copied from an older Apache workflow (CI does not
  install it)
- `scripts/run_official_tests.py` - Runner for both modes
- `scripts/rust_transport_plugin.py` - Pytest plugin that injects our Rust transport (transport
  mode)
- `scripts/dropin_deselect_plugin.py` - Pytest plugin that applies the manifest (drop-in mode)
- `scripts/parity_known_failures.txt` - Known-failure manifest, transport mode
- `scripts/dropin_known_failures.txt` - Known-failure manifest, drop-in mode
- `scripts/gen_parity_skiplist.py` - Regenerates the transport-mode manifest
- `.github/workflows/build_python_connect.yml` - CI workflow
- `official-tests.md` - This document

## References

- Apache Spark: https://github.com/apache/spark
- Apache Spark's `.github/workflows/build_python_connect.yml`, which this workflow mirrors
- Module source named in the manifest header: `dev/sparktestsupport/modules.py` in apache/spark
