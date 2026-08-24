<!-- SPDX-License-Identifier: Apache-2.0 -->

# Acceptance: 100% parity via Spark's official Connect test suite

The definitive gate (per user requirement) is Apache Spark's
`.github/workflows/build_python_connect.yml`: our Rust-backed package, installed
as the pure-Python `pyspark-client` replacement, must pass the **same** tests.

## What the workflow does

1. Builds Spark from source (SBT) and starts a Connect server via
   `./sbin/start-connect-server.sh` (with protobuf + avro jars).
2. `pip install pyspark*client-*.tar.gz` — the pure-Python client package.
   **This is what we replace.** Then installs grpcio 1.76, protobuf 6.33.5,
   pandas 2.3.3, pyarrow (implied), scipy, plotly, mlflow, torch, scikit-learn, …
3. **Deletes `pyspark.zip` and `py4j` zip** so there is *no* JVM/Py4J path — the
   client is pure and talks only gRPC. Our package must likewise need no JVM.
4. Runs, with `SPARK_CONNECT_TESTING_REMOTE=sc://localhost`:
   - `./python/run-tests --parallelism=1 --python-executables=python3 --modules pyspark-connect,pyspark-ml-connect`
   - `./python/run-tests ... --modules pyspark-pandas-connect,pyspark-pandas-slow-connect`
   - local-cluster: `pyspark.resource.tests.test_connect_resources`,
     `pyspark.sql.tests.connect.client.test_artifact*`,
     `pyspark.sql.tests.connect.test_parity_resources`.

## Scope

**371 connect test files** under `python/pyspark/**/tests/connect/**` (SQL,
client, arrow, pandas, streaming, ml, resource). Most are `test_parity_*` — they
run the classic PySpark test body over a Connect session, so passing them means
behavioral parity with classic PySpark, not just API shape.

Implication: our package must expose not only the public API but also the
internal modules these tests import (`pyspark.sql.connect.proto`,
`pyspark.testing.*`, error classes, `pyspark.sql.connect.*` internals used by
parity mixins). Tracked as ledger items in the P10 Python-skin phase.

## Incremental measurement (don't wait until the end)

`scripts/run_official_connect_tests.py` runs a chosen official connect test file
against a *target* `pyspark` package on `PYTHONPATH`, pointed at our live dev
server (`sc://localhost:15002`). As each feature area lands we run its
corresponding official test file (e.g. `test_connect_column`,
`test_connect_functions`, `test_connect_dataframe`) and record pass rate. Full
green across all 371 = done.

Note: our dev server is Spark 4.0 (installed jars); the CI uses a from-source
4.1 server. Protocol is compatible for the core; a from-source server can be
built later for the final certified run.
