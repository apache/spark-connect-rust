<!-- SPDX-License-Identifier: Apache-2.0 -->

# CI/CD Strategy

This document describes the GitHub Actions CI pipeline for the Spark Connect Rust Client, a
Rust-backed drop-in replacement for the PySpark Spark Connect client.

## Overview

The CI consists of these main workflows:

1. **Rust CI** (`.github/workflows/rust.yml`) - Build and test the pure-Rust crates
2. **Python Connect Parity** (`.github/workflows/build_python_connect.yml`) - Build the Rust
   extension and run the official Spark 4.3.0 Connect test suites through it against a live server
3. **Release wheel build checks** (`.github/workflows/wheel-build-checks.yml`) - Build Linux aarch64
   and macOS universal2 wheels using the release recipe
4. **Coverage** (`.github/workflows/coverage.yml`) - Rust and Python coverage, with regression
   floors

The other workflows are `lint.yml` (rustfmt, clippy, ruff), `generated-drift.yml`,
`vendor-drift.yml`, `docs.yml`, `pandas-parity.yml`, `pipelines-parity.yml` and `release.yml`.

Most workflows run on pushes and pull requests targeting `master` and `branch-*`. Several also
accept manual dispatch. `docs.yml` and `wheel-build-checks.yml` run only when files matching their
path filters change. The release wheel build checks cover changes to packaged Python sources,
protocol definitions, dependencies and build configuration. `release.yml` runs on a schedule and on
manual dispatch.

## Rust CI (`rust.yml`)

**Status**: Expected to be fully green.

This workflow validates the pure-Rust implementation:

- **Build**: `cargo build --release` builds the workspace `default-members`
  (`apache-spark-connect-proto`, `apache-spark-connect-core`, `apache-spark-connect`,
  `apache-spark-connect-macros` and `apache-spark-connect-build`). `pyspark-rs` is not a
  default member, so the Python workflows build it separately.
- **Test**: Runs `cargo test` on `apache-spark-connect` and `apache-spark-connect-core`, then
  builds and tests `apache-spark-connect` with the `datafusion,polars` features
- **No-stub audit**: Runs `scripts/audit_no_stubs.sh` to enforce the "100% coverage, zero stubs"
  rule (see [Design notes](architecture.md))
- **Warnings**: The workflow sets `RUSTFLAGS` to `-D warnings`

### Key steps

1. Checkout
2. Install Rust (stable channel, minimal profile) with caching
3. Install system dependencies (protobuf-compiler)
4. Build the default members
5. Test `apache-spark-connect` and `apache-spark-connect-core`, then build and test with
   `datafusion,polars`
6. Audit for placeholder/deferred markers

The same workflow has three more jobs:

- **Core transport tests (aarch64 Linux)** (`ubuntu-24.04-arm`, 20-minute limit): runs
  `cargo test --locked -p apache-spark-connect-core --release --lib` for `aarch64-unknown-linux-gnu`
- **WASM UDF feature**: installs the `wasm32-unknown-unknown` target and the Python packages
  `cloudpickle`, `pyspark`, `wasmtime` and `pyarrow`. It runs the `wasm-udf` feature tests, builds
  the WASM examples, and runs the ignored packer test and the WASM runner tests.
- **Minimum supported Rust**: `cargo +1.88 check --locked` on the published crates, with
  `RUSTFLAGS` cleared

This job should always pass; failures indicate incomplete implementation or test regressions.

## Python Connect Parity (`build_python_connect.yml`)

**Status**: Gating. The parity and drop-in jobs pass only when every test that runs passes, except
the entries in their known-failures manifests.

This workflow builds the Rust extension (`pyspark-rs`, a PyO3 module) and runs the official Apache
Spark 4.3.0 Connect test suites against it. Its four jobs are:

- `connect-parity` ("Connect parity"): the official `sql/tests/connect` files, run through our Rust
  transport against a Spark 4.3.0 server
- `connect-dropin` ("Connect drop-in"): the official Connect suites of `sql`, `ml`, `errors` and
  `logger`, run against our drop-in package
- `connect-e2e` ("Rust behavioral e2e"): the Rust `e2e_*` tests and the WASM UDF test, against a
  second server with reattach settings
- `connect-benchmark` ("E2E benchmark vs reference"): `scripts/benchmark_e2e.py`, comparing our
  client with the pip-installed `pyspark==4.3.0`

### Strategy

Every job runs Python 3.11 and JDK 17 (Temurin). The parity job is a matrix with that single entry.
The Spark version is pinned by `SPARK_VERSION: "4.3.0"`, and the test files come from the `v4.3.0`
tag.

**Expected behavior**:
- Each failure outside the manifest is a gap in parity or a regression. Fix the client. If the
  failure is environmental, add it to `scripts/parity_known_failures.txt`, regenerated with
  `scripts/gen_parity_skiplist.py`.
- `scripts/dropin_known_failures.txt` holds only by-design and environmental entries
- Only the event-timing-driven streaming files are retried (see
  [official-tests.md](official-tests.md)). A failure elsewhere fails on its first occurrence.
- The parity run reports manifest entries that now pass, so the manifest can shrink

### Key steps

#### 1. Setup
- Check out the repository
- Set up Python 3.11 (pip cache) and JDK 17 (Temurin)
- Install Rust (stable, minimal profile) and cache cargo
- Install system dependencies (protobuf-compiler)

#### 2. Build the extension
```bash
cargo build -p pyspark-rs --release
cp target/release/lib_pyspark.so python/pyspark/_pyspark.so
```
The extension is copied into `python/pyspark/`, so `./python` is a working package. No wheel is
built or installed, because the reference `pyspark` and ours both own the `pyspark` namespace.

#### 3. Install dependencies
```bash
python -m pip install pytest 'pandas>=2.2,<3.0' pyarrow numpy grpcio grpcio-status \
  googleapis-common-protos cloudpickle zstandard py4j 'plotly<6.0.0' graphviz==0.20.3
```
The drop-in job installs the same list. The workflow comments give the reason for the
pandas 2.x pin: Spark 4.2's tests and Python workers are not compatible with pandas 3.x.

#### 4. Download & start Spark Connect server (4.3.0)
- Downloads the official Spark 4.3.0 binary distribution with the Connect server bundled
  (`spark-4.3.0-bin-hadoop3-connect.tgz`) from archive.apache.org into `~/spark-dist`, cached
- Starts the server with `--driver-memory 3g` and
  `spark.sql.session.localRelationSizeLimit=1073741824` on `sc://localhost:15002`
- Waits for readiness (a `/dev/tcp` port check, up to 90 seconds)

#### 5. Clone Apache Spark test suite
- Clones Apache Spark at tag `v4.3.0` (`--depth 1`) into `~/spark-source`, cached

#### 6. Run the parity gate
```bash
RUST_PYSPARK_SO=$PWD/python/pyspark/_pyspark.so \
python scripts/run_official_tests.py \
  --spark ~/spark-source \
  --remote sc://localhost:15002 \
  --jobs 1 \
  --exclude "test_artifact.py,test_artifact_localcluster.py,test_parity_resources.py"
```

The runner injects our Rust transport into the official test files (see
[official-tests.md](official-tests.md)). The same job then:
- runs the drop-in-facing live-server pytest files under `python/tests`
  (`test_arrow_e2e.py`, `test_dropin_e2e.py` and the ML e2e files)
- restarts the server in `local-cluster[2, 4, 1024]` mode and runs the resource and artifact
  files with `--only` and `--no-drift-check`

#### 7. Artifact upload
- Uploads the parity logs (`/tmp/parity-test-logs/`, `/tmp/parity_ours.log`,
  `/tmp/parity_localcluster.log`) and both server logs as the `parity-ours-py3.11` artifact

These logs are available in the workflow run artifacts for debugging failures.

### Other jobs

- **connect-dropin**: the same setup, without the transport. It runs
  `scripts/run_official_tests.py --dropin --jobs 1` against the server started from `SPARK_HOME`.
  Its known failures are in `scripts/dropin_known_failures.txt`. Its logs are uploaded as the
  `dropin-official-py3.11` artifact.
- **connect-e2e**: starts a second server with
  `spark.connect.execute.reattachable.senderMaxStreamSize=1048576`, so that results over
  1 MiB exercise the reattach path. It runs `cargo test -p apache-spark-connect` on the `e2e_*`
  tests and a set of other integration tests with `--test-threads=1`, then the
  `e2e_wasm_udf` test in `tests/wasm-udf`.
- **connect-benchmark**: builds the extension and starts a server with default settings. It runs
  `scripts/benchmark_e2e.py --isolate` for the reference (pip-installed `pyspark==4.3.0`) and for
  ours (`PYTHONPATH=$(pwd)/python`), then `--compare`.

### Test selection

The parity job discovers every `test_*.py` under `<spark>/python` whose path contains
`/sql/tests/connect/`, minus the three local-cluster-only basenames. The drop-in job discovers
the Connect suites of `pyspark/sql`, `pyspark/ml`, `pyspark/errors` and `pyspark/logger`. Apart
from the local-cluster phase's `--only` list, no module is named by hand. See
[official-tests.md](official-tests.md) for the selection, skip and retry rules.

### Handling test failures

Each failing test is a **gap in parity**. The fix process:

1. **Identify the gap**: Read the test failure and corresponding official code. The job log
   lists the unexpected failures at the end of the run.
2. **Locate the ledger row**: Find the symbol in `dev/parity/inventory.csv`
3. **Implement the feature**: Code the missing functionality in the Rust crate
4. **Update ledger status**: Mark the row `done` (or `verified`; see Continuous improvement)
5. **Verify locally**: Run `scripts/run_official_tests.py --only <module>` (see
   [official-tests.md](official-tests.md))
6. **Push & verify CI**: The workflow will show the test passing on the next run

## Release wheel build checks (`wheel-build-checks.yml`)

These checks use the same reusable `build-wheel.yml` recipe as `release.yml`.
An x86_64 Linux runner cross-builds the Linux aarch64 wheel with Maturin and Zig.
A macOS runner builds a universal2 wheel containing both x86_64 and aarch64
slices. These are build-only checks; Rust CI separately executes the native ARM
core tests.

The workflow runs on pushes and pull requests targeting `master` or `branch-*`
when packaged sources under `python/pyspark/`, protocol definitions under
`crates/spark-connect-proto/proto/`, Cargo manifests or the lockfile, build scripts,
packaging/toolchain configuration, or the wheel workflows change.
Ordinary Rust source changes do not trigger these focused checks. The scheduled
`release.yml` dry run builds only `master`, so `branch-*` maintenance branches have
no scheduled backstop for Rust source changes. To validate a maintenance branch,
manually dispatch `release.yml` with `branch` set to that branch, empty
`release-version` and `rc-count` inputs, and `finalize: false`.

Wheel checks and release wheel builds pass `--locked` to Maturin. The selected
source branch must contain an up-to-date `Cargo.lock`; repair missing or stale
lockfiles before dispatching a release. Wheel and sdist recipes install Maturin
within the `>=1.0,<2.0` range declared in `pyproject.toml`. Wheel artifact uploads
fail if the build produces no wheel.

The shared recipe uses no dependency or build caches, so each run is a clean build
matching the release. Checks have a 45-minute timeout;
releases retain their six-hour limit. Check artifacts expire after one day;
release artifacts retain the repository default. Checkout credentials are not
persisted in either build.

## Coverage and the offline Python suite (`coverage.yml`)

This workflow runs on pushes and pull requests targeting `master` and `branch-*`. Its
one job does three things:

- **Rust coverage**: `cargo llvm-cov` over the unit and golden tests of
  `apache-spark-connect` and `apache-spark-connect-core`, plus nine behavioral e2e test
  files (`e2e_functions`, `e2e_columns`, `e2e_dataframe`, `e2e_coverage`, `e2e_integration`,
  `e2e_streaming`, `e2e_new_methods`, `e2e_ops_breadth`, `e2e_session_catalog`) against a live
  Connect server, merged into one report. Instrumented builds use a fresh directory under the
  runner's temporary directory so restored Cargo artifacts cannot contribute stale source
  mappings. These builds do not reuse the ordinary `target` cache; their duration counts toward
  the job's timeout.
- **Offline Python suite**: builds the extension and runs
  `python/tests/test_dropin_offline.py`, `test_arrow_handoff.py` and `test_ml_offline.py`
  under `coverage.py`. No other workflow runs these files, so this step is their gate: a
  failing test fails the job. `-rs` prints every skipped test with its reason. The scope
  of the Python number is the hand-written shim, as `python/.coveragerc` defines it.
- **Coverage floors**: the last step fails the job when a number is below its floor,
  `RUST_COVERAGE_FLOOR` (Rust line coverage) or `PYTHON_COVERAGE_FLOOR` (Python line and
  branch coverage), both set in the `env` block of the workflow. The step runs after the
  badge and the artifacts are published, so a run below a floor still reports the number
  it measured.

The floors are regression floors, set just below what `master` measures; the target
(above 90% on both sides) is in `AGENTS.md`. Raise a floor when coverage has grown and
stayed there. The Rust number moves by about 0.1 between runs of the same commit, since
the e2e tests run against a live server, so keep a floor a few tenths below the measured
value.

On a push to `master` the job also publishes both numbers to the `badges` branch, which
the README badges read.

## Continuous improvement

The parity ledger (`dev/parity/inventory.csv`) is the source of truth. Its status values in use are:
- **todo**: Not started. `scripts/gen_parity_ledger.py` gives new rows this status.
- **done**: Implemented, test(s) passing but not yet officially verified
- **verified**: Backed by golden-proto match and/or all-green Connect test suite
- **n/a**: Not applicable to the client. The notes column gives the reason, for example
  `architecture-satisfied`.

The CI workflows gate progress:
- Rust CI ensures the core crates are always buildable and testable
- The Connect parity and drop-in jobs fail on any official test outside their known-failures
  manifests
- `audit_no_stubs.sh` (run in Rust CI) fails on placeholder markers such as `TODO` or `FIXME` in
  `crates/*/src`

## Running tests locally

To run the Connect test suite locally:

1. **Build the extension**:
   ```bash
   cargo build -p pyspark-rs --release
   cp target/release/lib_pyspark.so python/pyspark/_pyspark.so
   ```

2. **Start a Spark Connect server** (requires Java 17 and the Spark 4.3.0 bin-hadoop3-connect
   distribution):
   ```bash
   # In one terminal:
   export SPARK_HOME=~/spark-dist/spark-4.3.0-bin-hadoop3-connect
   ${SPARK_HOME}/sbin/start-connect-server.sh \
     --driver-memory 3g \
     --conf spark.sql.session.localRelationSizeLimit=1073741824
   ```

3. **Run a test module** (install the Python dependencies listed under Strategy first):
   ```bash
   RUST_PYSPARK_SO=$PWD/python/pyspark/_pyspark.so \
   python scripts/run_official_tests.py \
     --spark /path/to/spark \
     --remote sc://localhost:15002 \
     --only pyspark.sql.tests.connect.test_parity_dataframe
   ```

See [official-tests.md](official-tests.md) for the flags, the output format and the skip rules.

## References

- [Design notes](architecture.md) - System design, parity strategy, no-stub rule
- [Acceptance criteria](acceptance.md) - Acceptance criteria and definition of done
- [pyproject.toml](https://github.com/apache/spark-connect-rust/blob/master/pyproject.toml) - Maturin build configuration
- [Official tests](official-tests.md) - The official-test runner, its modes and its skip rules
- [scripts/run_official_tests.py](https://github.com/apache/spark-connect-rust/blob/master/scripts/run_official_tests.py) - Test runner utility
- [scripts/audit_no_stubs.sh](https://github.com/apache/spark-connect-rust/blob/master/scripts/audit_no_stubs.sh) - No-stub audit gate
- [Apache Spark Workflow](https://github.com/apache/spark/blob/master/.github/workflows/build_python_connect.yml) - Reference implementation
