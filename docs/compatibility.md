# Compatibility

This client aims for **full API parity** with the reference PySpark Spark Connect
client (`pyspark.sql.connect.*`) and the shared modules it depends on. The goal is
that existing Spark Connect code runs unchanged and gets byte-for-byte identical
results. For how this client relates to the `pyspark` and `pyspark-client` packages
on PyPI - and how to tell which one you have installed - see
[Which client am I using?](which-client.md).

## Supported Spark

Apache Spark **4.2.0 and later**. The crate and wheel version tracks the Spark
release it targets (starting at `4.2.0`), so the version number tells you which
Spark it speaks.

## How parity is verified

Two independent gates in CI cover the two halves of a Spark Connect client - plan
building and the transport/result path:

- **Golden-proto tests** assert that the `spark.connect` protobuf plans this
  client builds (plans, expressions, and every SQL function) match the reference
  client **byte-for-byte**. This guards plan-building correctness.
- **The official Apache Spark Connect test suite** runs the standard PySpark tests
  against this client, exercising the **transport and Arrow result paths**
  end-to-end against a real server. See the official-test-suite notes
  and the CI notes under `dev/design/`.

Together the two gates cover what the client sends and what it does with what
comes back.

## The parity ledger

Coverage is tracked mechanically. `scripts/gen_parity_ledger.py` AST-parses the
reference PySpark source and emits one row per public **class**, **function**, and
**method** into a ledger (`dev/parity/inventory.csv`), so nothing is silently
dropped. The current inventory covers roughly:

| Kind      | Count |
|-----------|-------|
| Methods   | ~1340 |
| Functions | ~620  |
| Classes   | ~300  |

Each row carries a status:

- **done** / **verified** - mirrored by the client (and, where applicable,
  confirmed against the reference).
- **n/a** - *architecture-satisfied*: the equivalent wire behavior is provided by
  the Rust core (`spark-connect` / `-core` / `-proto`), so a reference-private or
  generated symbol is not mirrored one-to-one.

The ledger is a development tracking tool (kept under `dev/parity/`); the official
test suite is the authoritative gate.

## What "drop-in" means

From Python, `pyspark-client-rust` replaces the `pyspark-client` package: same
`import pyspark`, same public API, same server - see [Installation](installation.md).
Use it exactly like [PySpark](https://spark.apache.org/docs/latest/api/python/).
The native [Rust API](dataframes.md) mirrors the same surface.
