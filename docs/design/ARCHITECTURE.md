<!-- SPDX-License-Identifier: Apache-2.0 -->

# Spark Connect Rust Client — Architecture & Plan

A Rust rewrite of the PySpark **Spark Connect** client, exposed to Python as a
drop-in replacement for the `pyspark` package. Goal: **full API parity**
with `pyspark.sql.connect.*` (including Python UDFs and pandas UDFs), with the
hot paths (plan building, gRPC transport, Arrow decode) in Rust.

## Source of truth

- **Protocol**: Spark Connect protobuf `v4.1.0` — vendored under
  `crates/spark-connect-proto/proto/`, SHA pinned in `proto/SPARK_SHA.txt`.
  Developed and validated against Apache Spark 4.2.0 (wire-compatible).
- **Python reference**: `python/pyspark/sql/connect/**` (+ `sql/types.py`,
  `errors/exceptions/**`). This is the code we mirror line-by-line.
- **Prior art (same author)**: `spark-connect-ruby`, `spark-connect-scala3`,
  `pyspark-client-wasm` — proven module decomposition we follow.

## Parity ledger (the "don't miss anything" mechanism)

`scripts/gen_parity_ledger.py` AST-parses the reference source and emits
`docs/parity/inventory.csv`: one row per public class / function / method
(2,255 items across 45 modules), each with a `status` (`todo` → `wip` →
`done` → `verified`) preserved across regenerations. Every implemented symbol
flips its row; parity is reached when all rows are `verified`. Surface size:

| Area | Items | Notes |
|---|---|---|
| `functions/builtin.py` | 576 | SQL functions — mostly mechanical expr builders |
| `plan.py` | 343 (106 cls) | Logical plan → proto `Relation`/`Command` |
| `sql/types.py` | 239 (49 cls) | `DataType` hierarchy + parsing/JSON |
| `dataframe.py` | 147 | The `DataFrame` API |
| `expressions.py` | 123 (25 cls) | `Column` expression tree → proto `Expression` |
| `client/core.py` | 91 | gRPC transport, config, analyze/execute |
| everything else | ~700 | catalog, readwriter, group, window, udf/udtf, streaming, observation, errors |

## Crate layout

```
crates/
  spark-connect-proto   generated spark.connect.* (prost + tonic-prost). DONE.
  spark-connect-core    transport: channel builder, gRPC client, retries,
                        reattach, artifacts, config, error mapping (core.py,
                        retries.py, reattach.py, artifact.py, channel_builder)
  spark-connect         pure-Rust API mirroring pyspark: types, expressions,
                        column, plan, dataframe, functions, session, readwriter,
                        catalog, group, window, udf/udtf, streaming, observation
  pyspark-rs            PyO3 bindings -> native module `_pyspark`
python/
  pyspark/…             thin drop-in package; public modules re-export from
                        `_pyspark` so `from pyspark.sql import SparkSession` works
```

Rationale for the split: `spark-connect` is a usable pure-Rust client in its own
right (like the Ruby/Scala3 clients); `pyspark-rs` + `python/pyspark` is the
thin compatibility skin. Python-only concerns that *must* stay in Python —
cloudpickle UDF serialization, pandas/pyarrow interop — live behind PyO3 calls
into the interpreter, not reimplemented in Rust.

## Runtime & async model

Rust transport is async (`tonic` + `tokio`). Python's API is synchronous, so
`pyspark-rs` owns a shared multi-thread `tokio` runtime and blocks on it at the
FFI boundary, releasing the GIL during network waits (`Python::allow_threads`).
Server-streaming RPCs (`ExecutePlan`) are consumed into Arrow record batches.

## Data path

Results arrive as Arrow IPC streams. `arrow-rs` decodes them; hand-off to Python
uses the **Arrow C Data Interface** (zero-copy to `pyarrow`), which is also how
we build `toPandas()` / `createDataFrame(pandas)` without re-serializing.

## UDF strategy (full compatibility)

Python/pandas UDFs are **serialized, not executed** on the client:
1. `cloudpickle.dumps((func, return_type))` — called via PyO3 into CPython.
2. Wrap into `CommonInlineUserDefinedFunction` with `PythonUDF { command,
   output_type, eval_type, python_ver }`. `eval_type` distinguishes SQL_BATCHED,
   SQL_ARROW_BATCHED, pandas scalar/grouped-map/cogrouped-map, mapInPandas,
   mapInArrow, UDTF, etc. — matching `pyspark.util.PythonEvalType`.
3. The server's Python worker unpickles and runs it. No client-side execution.

This means byte-for-byte parity requires matching the pickle protocol version,
the command layout, and the eval-type enum exactly — tracked as ledger items.

## Definition of done (ENFORCED — 100%, no stubs)

Per explicit user requirement: **100% coverage, zero stubs, no deferred
followups.** A ledger item is complete only when it reaches `verified` — backed
by a passing golden-proto match and/or the official connect test — never merely
`done`. Nothing ships as a stub, including reattach, `UserDefinedType`, TLS/secure
channels, artifacts, and Python/pandas UDFs (all eval types). `scripts/
audit_no_stubs.sh` fails the build on any TODO/`unimplemented!`/stub/"for now"/
"deferred" marker in `crates/*/src`. Final gate: all 371 official connect test
files green.

**Known partials to COMPLETE (not defer):** reattach loop (currently a stub in
`spark-connect-core/src/reattach.rs`), `UserDefinedType` in `spark-connect/src/
types.rs`, TLS/secure channel dialing in `spark-connect-core`. These are tracked
as open ledger `wip` rows and must be finished before their phases count as done.

## Build order (phases)

1. **P0 Foundation** ✅ — workspace, proto codegen (offline), parity ledger.
2. **P1 Transport** — `spark-connect-core`: channel builder (`sc://` URL parse),
   gRPC client, config RPC, analyze/execute, retries, reattach, error mapping.
3. **P2 Types** — `DataType` hierarchy + proto <-> type, DDL/JSON parse.
4. **P3 Expressions & Column** — expression tree → proto `Expression`; operators.
5. **P4 Plan** — logical plan nodes → `Relation`/`Command`.
6. **P5 DataFrame + Session** — end-to-end: `spark.range/sql/createDataFrame`,
   transformations, `collect/show/toPandas`. First runnable vertical slice.
7. **P6 functions/builtin** — the 576 SQL functions.
8. **P7 readwriter / catalog / group / window / observation**.
9. **P8 UDF / UDTF** — cloudpickle path, all eval types.
10. **P9 Streaming** — readwriter, query, listener.
11. **P10 Python skin** — `python/pyspark/**`, import-compat, error classes.
12. **P11 Hardening** — retries/reattach edge cases, artifacts, TLS/auth,
    Databricks `x-databricks-*` headers, parity verification pass.

## Building

The core crates build with a standard `cargo build`. Optional conversion
features are off by default: enable them with
`cargo build -p spark-connect --features datafusion,polars`. `.cargo/config.toml`
sets the macOS link flags needed for the PyO3 extension.
