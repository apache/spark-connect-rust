# Shared-plan and shared-expression benchmarks through PySpark

`scripts/benchmark_plan_sharing.py` compares Rust-backed clients before and after
an ownership change. Both use the real native PySpark bindings and the same
running Spark Connect server. It does **not** use `rust_transport_plugin` or the
official-test harness. Official PySpark is an optional third control, not the
baseline for attributing an improvement to a Rust ownership change.

No speedup is established by adding this benchmark. Report actual measurements,
including regressions and noisy action timings, only after running it.

## Change scope and compatibility

DataFrames and their transformations share immutable logical-plan nodes using
`Arc`. Ordinary DataFrame and Python APIs are unchanged. Rust callers constructing
`LogicalPlan` enum variants directly must migrate child fields from
`Box<LogicalPlan>` to `Arc<LogicalPlan>`; this is a Rust source-breaking
representation change. The plan helper functions accept both owned and shared
nodes through `Into<Arc<LogicalPlan>>`.

The expression-sharing follow-up makes `Expression` a cheap, immutable,
`Arc`-backed handle; the former enum is named `ExpressionNode`. Direct Rust enum
construction and pattern matching must migrate to the node/handle API. Ordinary
Column, DataFrame, and Python APIs remain unchanged. These are Rust source
compatibility changes, not changes to the Spark Connect wire protocol.

Sharing avoids copying plan or expression ancestors during construction.
Serialization still materializes fresh protobuf trees, with separate relation
occurrences and plan IDs. This is not zero-copy serialization, Arrow zero-copy,
or server execution deduplication. Recursive serialization/drop limits are not
removed by shared ownership.

## Prepare isolated clients

Prepare two release builds with the same toolchain, build profile, Python version,
and Python dependencies. Keep the baseline checkout/build unchanged and build the
patched checkout separately. Install each wheel in its own virtual environment,
or place each compiled extension in its own `python/pyspark` tree and select that
tree with `PYTHONPATH`. Do not copy both builds over the same extension path.

Choose the baseline to isolate the change: compare the original client with the
plan-sharing client for the plan change; compare the **plan-sharing-only** client
with **plan plus expression sharing** for the expression follow-up. Comparing the
original client with both changes combined cannot isolate expression sharing.

The benchmark does not build or install anything. Use the repository's normal
build process before running it. Record source revisions, local changes, build
commands, and toolchain versions with the experiment. `--build-id` is a
caller-supplied description, not source attestation; the report independently
hashes the actual loaded extension. `--extension-sha256` can pin that binary.

Start one compatible Spark Connect server and leave it running for both clients.
Use the same server configuration, machine, input size, and benchmark arguments.
Avoid concurrent tests or other queries. Server startup, Python import, client
connection, identity checks, and metadata queries are outside the timers.

## Run baseline and patched in separate processes

The following paths are examples; substitute the two prepared environments and
one copy of the new benchmark script. Use **the same script** for both runs.
For wheel installations, unset `PYTHONPATH` to avoid an unintended source tree.

```bash
BENCHMARK=/path/to/patched/scripts/benchmark_plan_sharing.py
BASELINE_PYTHON=/path/to/baseline-venv/bin/python
PATCHED_PYTHON=/path/to/patched-venv/bin/python

env -u PYTHONPATH "$BASELINE_PYTHON" "$BENCHMARK" \
  --label baseline --build-id '<unchanged revision; release build>' \
  --expect-client rust --remote sc://localhost:15002 \
  --workload linear --depth 16 --rows 10000 --branches 4 --partitions 1 \
  --warmups 3 --repetitions 7 --out /tmp/plan-baseline.json

env -u PYTHONPATH "$PATCHED_PYTHON" "$BENCHMARK" \
  --label patched --build-id '<revision plus shared-plan patch; release build>' \
  --expect-client rust --remote sc://localhost:15002 \
  --workload linear --depth 16 --rows 10000 --branches 4 --partitions 1 \
  --warmups 3 --repetitions 7 --out /tmp/plan-patched.json

"$PATCHED_PYTHON" "$BENCHMARK" \
  --compare /tmp/plan-baseline.json /tmp/plan-patched.json
```

Repeat with `--workload branched` and new output paths for the other original
plan workload. Both original workloads retain their construction behavior.

For source-tree builds, replace `env -u PYTHONPATH` with
`env PYTHONPATH=/path/to/the/respective/checkout/python`. The script validates the
actual Session, DataFrame, and Column against the native extension's classes;
matching a version string or module name is not enough. It records imported
paths, extension SHA-256, Python executable/version, dependencies, host, server
version, selected server configs, and script SHA-256. URL credentials are omitted
from the displayed endpoint; its hash still enforces matching connection strings.

For an optional official control, use a third environment containing official
PySpark and the same dependency versions; pass `--expect-client official` and a
distinct label/build ID. Compare separately. Do not label that comparison as the
effect of changing Rust plan ownership.

### Isolate the expression-sharing follow-up

Use the same updated script with two separate native builds, one containing only
plan sharing and the other also containing expression sharing. For example:

```bash
PLAN_ONLY_PYTHON=/path/to/plan-only-venv/bin/python
EXPRESSION_PYTHON=/path/to/plan-and-expression-venv/bin/python

env -u PYTHONPATH "$PLAN_ONLY_PYTHON" "$BENCHMARK" \
  --label plan-only --build-id '<plan-only source and release build>' \
  --expect-client rust --remote sc://localhost:15002 \
  --workload expression_reuse --depth 64 --rows 10000 --branches 4 --partitions 1 \
  --warmups 3 --repetitions 10 --out /tmp/expression-plan-only.json

env -u PYTHONPATH "$EXPRESSION_PYTHON" "$BENCHMARK" \
  --label plan-and-expression --build-id '<expression-sharing source and release build>' \
  --expect-client rust --remote sc://localhost:15002 \
  --workload expression_reuse --depth 64 --rows 10000 --branches 4 --partitions 1 \
  --warmups 3 --repetitions 10 --out /tmp/expression-shared.json

"$EXPRESSION_PYTHON" "$BENCHMARK" \
  --compare /tmp/expression-plan-only.json /tmp/expression-shared.json
```

Repeat for `expression_chain` and depths 16/64, with fresh output paths. Repeat
whole process pairs in reversed order as described below. Do not compare new
reports with older reports produced by a different script revision: the script
hash comparison deliberately rejects that mismatch.

## What is measured

- `linear`: create a range, rename `id` to `value`, then apply `depth` projections
  adding one. Aggregate `sum(value)` and `count(value)` in a single collected row.
- `branched`: reuse the same increment chain in `branches` projections adding
  offsets 1 through `branches`, union them, and collect the same aggregate.
- `expression_chain`: start with `value = col("id")`, apply `value = value + 1`
  `depth` times in Python, then select that one deep Column as `value` and collect
  its sum and count. Unlike `linear`, this creates a deep expression rather than
  a deep sequence of projection plans.
- `expression_reuse`: build the same deep Column once, reuse it in `branches`
  projections of the same source with offsets 1 through `branches`, union those
  projections, then collect the sum and count.

Each name is individually selectable with `--workload`. `--workload all` (the
default) runs all four. Use explicit names to keep follow-up experiments scoped.

The checksum forces evaluation of the projected value, unlike `count()` alone,
which can allow Spark to discard the expressions. Spark may still optimize the
plan; optimizer, execution, transport, and decoding costs remain part of action
time. No UDF, table, cache, or persistent data write is used.

Each fresh query records three wall-clock intervals using `perf_counter`:

- `construction_s`: Python API calls, expression/plan creation, and aggregate
  plan creation, including retaining the intermediate handles.
- `action_s`: `collect()`, including serialization, RPC, server planning and
  execution, and result decoding. This is **not** isolated serialization time.
- `total_s`: construction start through action completion. Report its own median;
  do not add the two phase medians, which may come from different samples.

Every intermediate DataFrame stays alive through the action, keeping destruction
of copied ancestors outside the measurement. The expression workloads also retain
every intermediate Column, including the arithmetic chain, branch additions and
aliases, and aggregate Columns. The original plan workloads retain DataFrames as
before. GC is collected before each sample and cyclic GC disabled during it.
Validation and final destruction are outside the timers. Consequently these are
retained-plan/expression construction/action timings, not a measurement of normal
chaining's intermediate deallocation or cleanup cost. Retaining handles can
amplify baseline memory usage; keep depths modest.

Both warmups and measured iterations check exact 64-bit integer results. With
`n=rows`, `d=depth`, and `b=branches`, the linear checksum is
`n*(n-1)/2 + n*d`; the branched checksum is `b*linear + n*b*(b+1)/2`.
`expression_chain` uses the linear formula; `expression_reuse` uses the branched
formula. Counts must be `n` and `b*n`, respectively. Failures, wrong clients, and
checksum mismatches exit nonzero; no successful JSON report is written. Existing
output paths are not overwritten. JSON retains every warmup/sample plus per-phase
medians.

## Interpreting results

Depth 16 and four branches are deliberately modest: deeply nested plans can hit
protobuf recursion, gRPC message-size, or Spark analyzer limits independently of
construction efficiency. Use identical settings when sweeping depth, and report
failures instead of excluding them. `--workload linear` or `--workload branched`
selects an original plan workload; `--workload expression_chain` or
`--workload expression_reuse` selects an expression workload. All allow separate
processes per workload. Within a process, `--order-seed` controls the reproducible
shuffled workload order for each warmup and measured iteration.

Repeat whole process pairs with alternating or randomized **client** order and
new output names (for example baseline/patched, then patched/baseline). Record that
external schedule. This reduces warm-server and temporal bias; within-process
workload shuffling does not remove baseline-first bias. Keep raw reports.

The comparison rejects differing settings, endpoint hashes, selected server
metadata, Python versions, host/platform, dependency versions, script hashes, or
workload sets. Two Rust runs loading the same extension hash are rejected. These
checks are guardrails, not proof of identical server load or source provenance.
Ratios are baseline median / candidate median; above one favors the candidate.
Medians alone do not establish statistical significance or an end-to-end speedup
for applications dominated by server work.
