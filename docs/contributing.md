# Contributing

Contributions are welcome. This is an Apache Software Foundation project, so a few
conventions apply.

The repository holds the Rust client crates under `crates/` and the Python drop-in:
the `pyspark` package under `python/pyspark`, around the `pyspark._pyspark` extension
that the `crates/pyspark-rs` crate builds. [Architecture](architecture.md) shows how
they fit together.

## Issues and discussion

Bugs and feature requests are tracked in **ASF JIRA** under the
[SPARK](https://issues.apache.org/jira/browse/SPARK) project - GitHub Issues are
disabled. Reference the JIRA key in your pull request title, e.g.
`[SPARK-XXXXX][CONNECT] Short summary`, with `[PYTHON]` after `[CONNECT]` for a change
to the Python drop-in.

## Building and testing

Requires a Rust toolchain and `protobuf-compiler`. Non-Windows aarch64 builds
also need the target C tools described in the
[source build requirements](installation.md#build-from-source).

```bash
cargo build                 # build the library crates
cargo test                  # unit + golden-proto tests
cargo build -p examples     # the Rust-native examples
```

Builder correctness is covered by **golden-proto tests**: plans, expressions, and
all SQL functions are asserted byte-for-byte against captured reference protos.
Transport and Arrow paths are exercised by the end-to-end tests below and by the
official Apache Spark Connect test suite in CI - see the CI notes under
[`dev/design/`](https://github.com/apache/spark-connect-rust/tree/master/dev/design).

The end-to-end tests (`crates/spark-connect/tests/e2e_*.rs` and a few others) run only
when `SPARK_REMOTE` names a [Spark Connect server](#a-local-spark-connect-server);
without it they pass without checking anything. They share one server session, so CI
runs them on one thread:

```bash
SPARK_REMOTE=sc://localhost:15002 \
  cargo test -p apache-spark-connect --test e2e_dataframe -- --test-threads=1
```

The `wasm-udf` example crates need the `wasm32` target and are built by manifest
path (they're excluded from the workspace):

```bash
rustup target add wasm32-unknown-unknown
cargo build --manifest-path examples/wasm-udf-inline/Cargo.toml
```

## A local Spark Connect server

The end-to-end tests, the live Python tests and the official test suites run against a
Spark 4.3.0 Connect server. The Spark distribution bundles the server, so no
`--packages` is needed: unpack `spark-4.3.0-bin-hadoop3-connect.tgz` (the one CI uses)
or `spark-4.3.0-bin-hadoop3.tgz` from the
[Apache archive](https://archive.apache.org/dist/spark/spark-4.3.0/), and start it with
Java 17 and CI's settings:

```bash
export SPARK_HOME=/path/to/spark-4.3.0-bin-hadoop3-connect
(cd "$SPARK_HOME" && TZ=UTC SPARK_TESTING=1 PYSPARK_PYTHON="$(command -v python)" \
  ./sbin/start-connect-server.sh \
  --driver-memory 3g --conf spark.sql.session.localRelationSizeLimit=1073741824)
```

It listens on `sc://localhost:15002` and logs to `$SPARK_HOME/logs`; stop it with
`"$SPARK_HOME"/sbin/stop-connect-server.sh`. The settings:

- `TZ=UTC` makes UTC the session time zone, as on CI's runners. The test commands below
  set it too.
- `SPARK_TESTING=1` turns on Spark's test-mode defaults, which some tests rely on.
- `PYSPARK_PYTHON` is the Python that runs UDFs on the server. It needs pandas and
  pyarrow, and the same minor version as the Python that runs the tests, or their UDFs
  fail with `PYTHON_VERSION_MISMATCH`. Start the server with the
  [virtual environment](#set-up-python) active.
- Starting it from `$SPARK_HOME` lets the official tests' relative data paths resolve.

## Python drop-in

`python/pyspark` holds Rust-backed shims over `pyspark._pyspark` and code vendored from
PySpark ([see below](#vendored-and-generated-code)). Its tests assert the behavior of
official PySpark 4.3.0: where the two differ, PySpark is the reference (see
[Compatibility](compatibility.md)).

### Set up Python

The package supports Python 3.9 or later (`requires-python` in `pyproject.toml`); CI
runs Python 3.11. Create a virtual environment with the packages CI installs. pandas
stays below 3.0, which PySpark 4.3 does not fully support yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pytest coverage 'pandas>=2.2,<3.0' pyarrow numpy grpcio grpcio-status \
  googleapis-common-protos cloudpickle zstandard py4j ruff==0.14.0
```

Tests that need an optional package skip without it; for the official suites, CI also
installs `'plotly<6.0.0'` and `graphviz==0.20.3`.

### Build the extension

Build the extension with the Rust requirements above, and copy it into the package.
Repeat this after every change to the Rust code, including the Python sources that the
extension embeds (`crates/pyspark-rs/src/row.py` and `values.py`):

```bash
cargo build -p pyspark-rs --release
cp target/release/lib_pyspark.dylib python/pyspark/_pyspark.so   # macOS
cp target/release/lib_pyspark.so python/pyspark/_pyspark.so      # Linux
```

`python/` is then a working `pyspark` package. CI puts it on `PYTHONPATH` rather than
installing it, which keeps it apart from any other `pyspark` in the environment:

```bash
PYTHONPATH=$(pwd)/python python -c "import pyspark; print(pyspark.__file__)"
```

To build a wheel instead, run `maturin build --release --out dist`, which reads
`[tool.maturin]` in `pyproject.toml` (see
[Installation](installation.md#build-from-source-python)).

### Run the tests

Run the tests from `python/`. The offline suites need no server:

```bash
cd python
TZ=UTC PYTHONPATH=$(pwd) python -m pytest \
  tests/test_dropin_offline.py tests/test_arrow_handoff.py tests/test_ml_offline.py
TZ=UTC PYTHONPATH=$(pwd) python -m pytest tests/test_dropin_offline.py -k pickle   # a subset
```

The live suites, `tests/test_*_e2e.py`, run against the
[local server](#a-local-spark-connect-server) and skip without `SPARK_REMOTE`:

```bash
SPARK_REMOTE=sc://localhost:15002 TZ=UTC PYTHONPATH=$(pwd) python -m pytest tests/test_*_e2e.py
```

CI runs the test files that its workflows name (`coverage.yml` for the offline suites,
`build_python_connect.yml` for the live ones), so add a new test to an existing file, or
add a new file to the workflow. CI also measures the coverage of the hand-written shim
within the scope of `python/.coveragerc`, and fails below the floor that `coverage.yml`
sets:

```bash
TZ=UTC PYTHONPATH=$(pwd) python -m coverage run --rcfile=.coveragerc -m pytest \
  tests/test_dropin_offline.py tests/test_arrow_handoff.py tests/test_ml_offline.py
python -m coverage report --rcfile=.coveragerc
```

### Run the official PySpark tests

On every pull request, CI runs Apache Spark's own Connect test suites twice: against the
drop-in's classes, and with official PySpark sending its requests through the Rust
transport. Run the files that cover your change with `scripts/run_official_tests.py`,
from the repository root. `--spark` is an Apache Spark tree with the tests under
`python/`: CI clones the `v4.3.0` tag, and the unpacked distribution has the same test
files.

```bash
RUST_PYSPARK_SO=$(pwd)/python/pyspark/_pyspark.so TZ=UTC python scripts/run_official_tests.py \
  --dropin --spark "$SPARK_HOME" --remote sc://localhost:15002 \
  --only pyspark/sql/tests/connect/test_parity_column.py
```

Without `--dropin`, the file runs through the transport instead. Without `--only`, the
whole suite runs, which takes about an hour on CI (`build_python_connect.yml` has CI's
exact options). The known failures are listed in `scripts/dropin_known_failures.txt` and
`scripts/parity_known_failures.txt`. Any other failure fails CI, and the runner reports
listed tests that now pass, to be removed from the list.

### Vendored and generated code

Many modules under `python/pyspark` are verbatim copies of PySpark 4.3.0, such as
`pyspark.pandas`, `pyspark.cloudpickle` and much of `pyspark.ml` and `pyspark.errors`.
Do not edit them: they are listed in `scripts/check_vendored_upstream.sh` and in the ruff
excludes of `pyproject.toml`, and CI fails when one that the script lists differs from
the `v4.3.0` tag (the script fetches it from GitHub):

```bash
bash scripts/check_vendored_upstream.sh
```

Fix a behavior in the Rust core (`crates/spark-connect`), in the bindings
(`crates/pyspark-rs`) or in a hand-written shim instead, and not by monkey-patching a
Rust class from Python.

`python/pyspark/sql/functions_generated.py` and
`crates/pyspark-rs/src/dispatch_generated.rs` are generated from
`crates/spark-connect/src/functions.rs`. Regenerate them after changing a function there;
CI runs the same command with `--check`:

```bash
python scripts/gen_fn_dispatch.py --pyspark-src "$SPARK_HOME/python"
```

## Style

```bash
cargo fmt --all                          # format (CI runs it with --check)
cargo clippy --workspace --all-targets   # lint (advisory in CI)
```

Rust CI builds with `RUSTFLAGS="-D warnings"`, so a compiler warning fails it. Clippy
is advisory there: the tree still has findings to clean up (and `pyspark-rs` denies
warnings, so they show as errors), but don't add new ones. CI also fails on deferral
markers such as `TODO`, `FIXME`, `todo!()` or "for now" in `crates/*/src`
(`bash scripts/audit_no_stubs.sh`). Match the surrounding code, keep changes focused,
and add or update golden tests when you change plan or expression building.

Python is checked with ruff 0.14.0, the version CI pins, within the scope and excludes
of `pyproject.toml`. Run it from the repository root without paths, as CI does:

```bash
ruff check
ruff format --check
python python/tests/test_ruff_excludes.py   # the excludes match the vendored list
```

`ruff format` covers `scripts/` and the Row and value sources that the extension
embeds: the shim and the tests are linted but not reformatted, so that open pull
requests don't conflict. When you pass paths, for example from an editor, add
`--force-exclude`: without it, ruff checks and formats every path it is given,
vendored ones included.

## Pull requests

Open PRs against
[`apache/spark-connect-rust`](https://github.com/apache/spark-connect-rust).
Keep the PR description tied to its JIRA. Before you push, make sure `cargo test` and
`cargo fmt --all --check` pass and, for a change to the drop-in, the offline Python
tests and ruff; CI runs these as well as the end-to-end and official suites, coverage
and the vendored-code check.
