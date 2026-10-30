# Arrow input upload benchmarking

## Current decision

Keep sequential cache uploads with 1 MiB artifact frames, accelerated SHA-256,
and shared payload buffers. Generic artifact uploads retain 32 KiB frames.
Keep hash order, CRC validation, acknowledgements, and server-side deduplication.

Two concurrent upload streams did not improve the localhost experiment.
Overlapping serialization and upload reduced latency slightly, but increased
peak client RSS growth from approximately 93 MiB to 145 MiB for a 136 MB input.
Neither concurrency prototype is retained.

The selected framing change reduced the 136 MB creation-plus-query median from
230.58 ms to 138.31 ms; official PySpark measured 307.30 ms in the same experiment.
These are **localhost fixture measurements without Spark query execution**,
using ten fresh-process samples per client. They do not establish remote Spark
performance. A separate live localhost Spark check verified the uploaded values.

## Required before merging: real remote Spark measurements

Remote performance validation is **pending**. Benchmark against an actual Spark
Connect server on another machine over the deployment's normal network and TLS
path. A local JVM, a scripted fixture, or injected latency alone does not satisfy
this requirement. Obtain the remote endpoint and configured authentication from
the operator; keep credentials out of commands, reports, and checked-in files.

Compare all three clients against that same server:

- Official PySpark, matching the server version.
- The previous optimized sequential client with 32 KiB frames, including the
  accelerated hash and shared-buffer changes. Comparing only against the earlier
  software-hash implementation would overstate the framing improvement.
- The proposed sequential client with 1 MiB cache frames.

Use the same Python interpreter and dependency versions. Record exact native
binary hashes and source revisions or patches. Run samples serially, rotating
client order, with at least ten fresh-process samples per client and case.

### Workloads and measurement boundaries

Use eight int64 columns at these sizes:

| Rows | Unencoded input size |
| ---: | ---: |
| 12,500 | 0.8 MB |
| 250,000 | 16 MB |
| 2,125,000 | 136 MB |

Create the input and establish a fresh session outside the measurement. Warm
server execution with an unrelated small query, without uploading the measured
input. Do not reuse a fixed session ID across cold-input samples: a cached hash
would conceal the upload cost.

Measure creation plus a materializing query as the primary comparison, including
all uploads. Aggregate all eight columns, checking row count and column sums
against the generated data, so server execution consumes the complete relation.
Also measure repeated queries on a prepared DataFrame after one untimed query;
report these separately from the initial upload. Never warm up cold-input timing
by creating that same input in the same session.

Keep real server configuration unchanged for the primary comparison. Record
the cache threshold, local relation size limit, chunk row/byte settings, and
upload-batch cap. The fixture used a 64 MiB upload-batch cap; the real server's
default is 1 GiB, so fixture memory and RPC counts cannot be extrapolated directly.

### Report and decision

Record client/server versions and hardware, client/server locations, TLS mode,
small-query round-trip latency, driver heap, and concurrent server activity.
Report raw samples, medians and ranges, client CPU, sampled peak RSS above the
prepared-input baseline, failures, and upload/network counters where available.
Keep server execution time visible rather than attributing the entire query
latency to the client. Record sampling intervals and any unavailable counters.

The 0.8 MB input is a control: the framing change does not optimize its inline
path, which remains slower than PySpark in the localhost results. Preserve this
result alongside larger-input gains.

Reassess frame size if the remote workload regresses. Do not enable parallel
uploads or claim an end-to-end remote speedup solely from the localhost results.
