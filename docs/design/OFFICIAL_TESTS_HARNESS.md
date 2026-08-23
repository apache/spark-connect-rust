# Running the official Spark Connect tests against our client

## The problem (verified)

The official connect test files live inside the reference `pyspark` package tree and
import pyspark *internals* (`pyspark.sql.connect.*`, `pyspark.util.JVM_INT_MAX`,
`functions` as a package, the `pyspark.sql.tests.*` base mixins). When pytest runs
`spark-v4.2.0/python/pyspark/sql/tests/connect/test_x.py`, it inserts that tree's root
at `sys.path[0]`, so `import pyspark` resolves to the reference package regardless of
`PYTHONPATH`. Proven with a non-invasive plugin:

    COLLECTED_PYSPARK: .../spark-v4.2.0/python/pyspark/__init__.py

So pointing `--target-pyspark ./python` did nothing - both sides of the old parity gate
ran the reference client. Overlaying our partial skin onto the tree also fails, because
our internal module layout diverges (single `functions.py` vs `functions/` package,
minimal `util.py` missing `JVM_INT_MAX`, no `pyspark.sql.connect.*`).

Baseline profile to reproduce (reference client, this local 4.2.0 server):
`test_parity_functions` = 144 passed, 17 failed, 4 skipped. The 17 are environmental
(server type support), not client bugs; our client must match this profile.

## The approach: strangler-fig with a Rust transport seam

Our package must present pyspark's full connect-client module layout so the tests
import, run, and pass; then route the hot path (network + Arrow) through Rust.

1. **Vendor** the reference connect client Python into our package: `pyspark/sql/connect/*`
   (60 modules), `pyspark/sql/connect/proto/*` (pb2), `pyspark/sql/pandas/*`, full
   `pyspark/util.py`, `functions/` as a package, `pyspark/errors/*`, `pyspark/resource/*`,
   and the `pyspark/sql/tests/*` + `pyspark/testing/*` infra the tests inherit from.
   Source: the v4.2.0 worktree at `/path/to/spark-v4.2.0/python`.

2. **Rust transport seam.** `SparkConnectClient` funnels every RPC through one gRPC stub
   (`self._stub`). The full method surface it uses (verified):
   `ExecutePlan, AnalyzePlan, Config, Interrupt, ReattachExecute, ReleaseExecute,
   ReleaseSession, GetStatus, CloneSession, FetchErrorDetails`.
   Add a PyO3 stub in `_pyspark` implementing these, bridging
   `pb2 request -> SerializeToString() bytes -> Rust (prost decode, tonic send) ->
   response bytes -> pb2.Response.FromString()`. Our `spark-connect-core::client`
   already implements execute/analyze/config/interrupt/reattach/release; wire prost
   messages (wire-compatible with the pb2 the reference builds).

3. **Inject** the Rust stub as `SparkConnectClient._stub` (or via `toChannel`), run the
   official connect suite with only our package on the path, and match the reference
   pass/skip/fail profile. Iterate on genuine gaps.

Result: reference builds the plan protos (byte-identical), Rust carries them over the
wire and decodes Arrow - so our Rust core is genuinely exercised by the official tests,
and correctness starts high because the protos are the reference's.

## Then parallelize

Once the harness genuinely runs our client, parallelize with `pytest-xdist -n auto`
(within a run) and/or the `parity_baseline_diff.py --jobs N` across files (fresh CI
server handles concurrent sessions).

## What was actually built

Rather than vendoring (M3), we use **runtime injection**: run the official tests from
the reference tree and monkeypatch `SparkConnectClient` so its stub is our Rust transport
(`scripts/rust_transport_plugin.py`), with `_use_reattachable_execute=False`. This tests
our client with zero vendoring. Errors are propagated as `RustRpcError(code,
status_details, message)` and rebuilt into the exact pyspark exception via
`convert_exception` - both for mid-stream and initial-call gRPC errors (the latter needs
`SparkError.grpc_code`/`grpc_details`, added in `error.rs`).

## Milestones

- [x] M1: Rust raw-bytes transport (`RustConnectStub`), verified by a raw round-trip.
- [x] M2: stub methods + Python bridge shim (`rust_transport_plugin.py`), incl. typed
      error propagation.
- [x] M4: inject the stub; modules match the reference profile - `test_parity_functions`
      (144/17/4), `test_parity_catalog` (27/0), `test_parity_collection` (14/1).
- [x] Genuine gate `scripts/rust_parity_diff.py`, wired into CI.
- [ ] M5: full SQL connect suite matches reference; then pandas-on-Spark connect modules
      (needs `pyspark.pandas`); parallelize with `--jobs` / `pytest-xdist`.
