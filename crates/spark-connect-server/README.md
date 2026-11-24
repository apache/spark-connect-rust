<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# Rust Spark Connect frontend with an embedded JVM engine (WIP)

This prototype embeds a Rust Spark Connect listener inside a Spark JVM through
JNI. Rust owns request decoding, protocol sessions, operation IDs, response
framing and stream lifetimes. The Java adapter calls Spark planning and execution
APIs through an in-process engine interface.

The frontend uses the workspace's `apache-spark-connect-proto` messages. It sends
serialized `Relation` inputs through JNI; Java decodes them with Spark's protobuf
classes and uses `SparkConnectPlanner` to construct Catalyst plans. Spark executes
those plans in the JVM that hosts the Rust listener. This boundary does not pass
Spark Connect service requests, responses or operation registries to Java.

This is a work in progress for local trusted workloads, with a limited protocol
surface. It does not provide authentication, authorization or TLS. Keep its
listener on loopback and read the [security policy](../../SECURITY.md) before use.

## Build requirements and compatibility

- Rust 1.88 or newer, Cargo and `protoc` 3.15 or newer (proto3 optional support).
- JDK 21, a C compiler with C11/pthread support and JNI headers. Set `JAVA_HOME`
  to the JDK directory.
- Python 3.9 or newer for artifact metadata and packaging. Java build scripts use
  `python3`; set `NATIVE_BUILD_PYTHON` to select another interpreter.
- macOS or Linux. Windows is not supported by this prototype.
- A compatible Spark assembly containing the shaded Connect protobuf/gRPC jars.
  Set `SPARK_HOME` to its directory. The scripts accept jars in either
  `$SPARK_HOME/jars` or `$SPARK_HOME/assembly/target/scala-2.13/jars`.

The frontend and JNI adapter passed local checks on macOS ARM64, including the
selected-runtime acceptance described under [Validation](#validation). The
historical performance measurements below have not been reproduced with the
current result-framing frontend.

The workspace currently tracks the Spark 4.3.0 protocol; its pin is recorded in
[PROTO_VERSION.txt](../spark-connect-proto/proto/PROTO_VERSION.txt). The JNI
implementation was developed and validated with Spark `4.2.0-SNAPSHOT`, revision
`250ccca3f2b5f043e20b415adf17bda5d202c60f`, Scala 2.13.17 and JDK 21.0.10. It depends
on internal Spark APIs, including `SessionHolder`, `SparkConnectPlanner`,
`QueryExecution` and Arrow conversion helpers. That validation does not establish
Spark 4.3.0 assembly compatibility or support for every 4.3.0 protocol feature.
Other assemblies may require Java source changes and fresh acceptance checks.

The range-query JNI smoke test uses that unmodified Spark snapshot. The broader
acceptance run used a separate local repair to
`ArrowConverters.InternalRowIteratorFromIPCStream` for Arrow input resource
cleanup at EOF and on failure. That repair is not bundled with this prototype.
The smoke test does not cover Arrow IPC input, and the broader result does not
establish full acceptance on the unmodified snapshot.

The `spark-connect-server` and nested `spark-connect-jni` crates are unpublished,
optional workspace members excluded from `default-members`. Build them explicitly
with `-p`; the server crate provides the frontend library and the JNI crate builds
the JVM-loaded native library. There is no standalone server executable.

All commands below run from the repository root. Cargo places native artifacts in
`target/`; the JVM scripts place generated Java artifacts and acceptance reports
under `crates/spark-connect-server/.build/`.

## Frontend library

The unpublished `spark-connect-server` library uses the published
`apache-spark-connect-proto` crate's optional `server` feature and accepts an
application-provided `Engine` and bound Tokio TCP listener. The feature adds
generated server bindings to the protocol crate's public API; Cargo unifies it
with other uses of that crate in the workspace. `serve_with_engine` uses default
transport options; Rust embedders can call `serve_with_engine_and_options` with
`ServerOptions` to configure Arrow chunking. The embedded JVM launcher currently
uses those defaults. The nested `spark-connect-jni` crate supplies the Spark engine
adapter described in this guide; the frontend library can also be used without it.

Each protocol session receives an `EngineSession` that owns planning, catalog
and configuration state. Its `prepare` method returns a `PreparedQuery` or
command response payloads; the frontend adds protocol IDs and, for requests that
opt into reattachment, `ResultComplete`. The engine decides which plans, commands
and metadata requests it can execute.

## Protocol subset

The frontend routes `ExecutePlan`, `AnalyzePlan` and `Config` to the engine. It
implements `ReleaseExecute` and `ReleaseSession`, including operation cancellation
through `ReleaseAll` and session recreation when `allow_reconnect` permits it.
Completed responses are not retained. Reattachment, artifact upload/status,
`Interrupt`, enriched error details, session cloning and status queries return
`UNIMPLEMENTED`.

Session creation and `ExecutePlan` operation IDs follow JDK 21 `UUID.fromString`
parsing, including uppercase and shortened hyphen-separated groups. Session IDs
retain their supplied spelling: equivalent UUID spellings identify different
sessions, as in Spark. `ExecutePlan` normalizes operation IDs to canonical
36-character lowercase UUIDs and rejects duplicate active IDs within a session.
Completed IDs are forgotten and can be reused, so retrying a completed command
with the same ID can execute it again. Spark remembers recently completed and
abandoned operation IDs and rejects their reuse.
`ReleaseExecute` looks up its operation ID as supplied; clients should use the
normalized ID returned by execution. It ignores the client-observed server
session ID, as Spark does. `ExecutePlan`, `AnalyzePlan` and `Config` check that
observed ID. Client user IDs distinguish session keys but are not authenticated
identities. Release RPCs look up session keys without parsing UUIDs.
`ReleaseSession` for an unknown session is a no-op. `ReleaseExecute` returns
`NOT_FOUND` for a missing session and `FAILED_PRECONDITION` while its
released-session tombstone is retained.

The frontend validates execution operation IDs and plan envelopes before resolving
a session, so invalid execution requests allocate no backend. Spark resolves the
session first; this ordering can change which error is reported when multiple
fields are invalid.

`ResultComplete` is sent only when a request sets `ReattachOptions.reattachable`.
That option supports completion detection on the single live result stream;
results cannot be resumed or replayed because `ReattachExecute` is unimplemented.
Nonreattachable requests complete when their result stream ends.

`ReleaseAll` cancels active operations, including nonreattachable operations.
This is a deliberate experimental deviation: Spark rejects release of a
nonreattachable operation with `INVALID_CURSOR.NOT_REATTACHABLE` and uses
`Interrupt` for cancellation. This frontend uses `ReleaseAll` for cancellation
while `Interrupt` remains unimplemented. `ReleaseUntil` and requests without a
release kind are no-ops, including for live operations; Spark validates
reattachability and the release kind for a live operation. Execution metrics and
Spark structured error classes are also absent; failures use gRPC status codes
and messages.

Protobuf nesting follows the selected Prost features; dependency feature
unification can disable its recursion limit, and the frontend adds no independent
nesting guard. Deep plans can fail before engine planning; when its recursion
limit is enabled, the Rust transport reports a protobuf recursion error as
`INTERNAL`.

## Arrow results

A prepared query declares an Arrow schema before execution. The frontend converts
supported types to the Spark schema representation and checks each batch against
that declaration. Batches may use Spark-equivalent Arrow representations and
nullability different from the declaration. Within one result, emitted IPC
batches must share physical types, field names and nullability, so clients can
assemble them into one Arrow table. Metadata may differ between batches.
Declared field metadata and unsupported types fail explicitly.

Engines may yield Arrow record batches or complete Arrow IPC streams containing
one batch and its actual nonnegative row count. The frontend encodes each record
batch once and forwards existing IPC bytes unchanged, including compressed IPC.
It reads the IPC schema for validation and trusts the producer's batch contents,
record-batch count, EOS marker and declared row count; it does not validate those
parts of the stream. Record-batch encoding materializes string/binary view
arrays, including nested children, to `LargeUtf8`/`LargeBinary`. Existing IPC
retains its physical types. The frontend reports `INTERNAL` before sending a
batch whose output schema differs from earlier batches; engines mixing record
batches with IPC must use consistent output representations.

When the client sets `ResultChunkingOptions.allow_arrow_batch_chunking`, the
frontend divides serialized IPC bytes into chunks and sets `chunk_index` and
`num_chunks_in_batch`. Boundaries need not align with rows or Arrow buffers; the
client concatenates a batch's chunks before decoding or decompressing it. A
preferred chunk payload size between 1 KiB and the configured maximum is accepted;
otherwise the maximum is used. `ServerOptions.max_arrow_chunk_size` defaults to
120,795,955 bytes (about 115.2 MiB; 90% of 128 MiB), matching the JVM server's
default chunk payload limit. It limits IPC payload bytes, excluding the protobuf
envelope. Setting it to zero disables chunking even when the client opts in. An
opted-in client must accept the configured maximum plus its protobuf envelope,
or request a smaller chunk size. A 128 MiB receive limit accommodates the default
maximum; raw tonic's 4 MiB default does not accommodate chunks above 4 MiB.

Without client opt-in, or when chunking is disabled, each IPC batch is sent whole.
Engines remain responsible for bounding batch production, including fitting the
client's gRPC receive limit when batches are sent whole. The frontend applies no
128 MiB batch-production cap. Spark's JVM producer separately uses
`spark.sql.execution.arrow.maxRecordsPerBatch` and a byte estimate of 70% of
`spark.connect.grpc.arrow.maxBatchSize` (128 MiB by default); its estimate can
overshoot by a row. Generic engines must choose their own production bounds.
Schema and command response envelopes have a separate 3 MiB limit and fail with
`RESOURCE_EXHAUSTED` when oversized. Each batch carries a cumulative row offset,
and empty results still include an empty batch.

## Lifetime and trust requirements

Engine methods and streams must cancel their owned work when dropped. The
frontend cancels them when a client disconnects, an operation is released, a
session is released, server shutdown begins or the serving future is dropped.
Results pass through a bounded
channel, so unread responses eventually stop polling the engine. The embedded
JVM engine can continue computing and buffering partitions while polling is
paused; see its [buffering limits](#resource-and-protocol-limits). Blocking Arrow
encoding retains operation ownership until its worker finishes. IPC chunks share
their batch's backing allocation, which remains live while any chunk references
it. Larger batches and chunk limits can therefore increase retained memory even
though the response channel is bounded.

`ReleaseAll` waits up to two seconds for teardown and reports `DEADLINE_EXCEEDED`
if it is still draining. Shutdown cancels sessions and waits up to two seconds
for transport drain; returning from `serve_with_engine` does not prove that an
engine's independently scheduled or blocking work has finished. The caller must
honor the engine's cleanup contract before releasing its resources.

`ReleaseSession(allow_reconnect=true)` permits a new backend session with the
same protocol ID while the previous incarnation still drains. Engines must keep
the two incarnations' resources independent. Commands may commit effects before
frontend response validation fails, for example on an oversized payload; that
failure provides no rollback or safe-retry guarantee.

With panic unwinding enabled, panics in `prepare`, `analyze`, `config` and query
execution are translated to `INTERNAL`. Synchronous session creation and query
schema access are outside that boundary and must not panic. Resource cleanup
must not panic either.

This is an experiment for trusted local clients and trusted engine outputs.
Keep the supplied listener on loopback: this frontend supplies no authentication,
authorization or TLS. Requests can be up to 128 MiB, matching Spark's default
inbound message limit. Larger messages receive tonic's `OUT_OF_RANGE` status.
The frontend registry has no active-session or operation
admission limit or idle expiry; the JNI engine adds the capacities described
below. Released-session tombstones are bounded to the
most recent 4096 distinct IDs. Spark instead uses a size-bounded cache configured
by `spark.connect.session.manager.closedSessionsTombstonesSize`, which defaults
to 1000; this frontend's fixed bound and release-order eviction policy differ.
An evicted ID can create a new session. These limits do not provide isolation or
complete memory and CPU quotas. Execution permissions and resource access are
the engine's responsibility.

## Run the embedded server

With `JAVA_HOME` and `SPARK_HOME` set, build the native library:

```bash
cargo build --release -p spark-connect-jni
```

On macOS:

```bash
export NATIVE_LIBRARY="$PWD/target/release/libspark_connect_jni.dylib"
crates/spark-connect-server/jvm/run-engine.sh "$NATIVE_LIBRARY" 127.0.0.1:15002
```

On Linux, set `NATIVE_LIBRARY` to
`$PWD/target/release/libspark_connect_jni.so` instead. The launcher requires an
absolute library path, compiles the Java adapter and starts `EmbeddedSparkEngine`
with `spark-submit`. `SPARK_MASTER` defaults to `local[2]`.

In another terminal, use the official Rust client example:

```bash
SPARK_REMOTE=sc://127.0.0.1:15002 cargo run -p spark-connect-server --example rust_client
```

It collects a range aggregate and a SQL result. A compatible official PySpark
Connect client can also run:

```bash
SPARK_REMOTE=sc://127.0.0.1:15002 python crates/spark-connect-server/examples/jvm_client.py
```

Stop the launcher with Ctrl-C. Native shutdown cancels and drains owned work,
releases engine sessions and closes the Rust listener before Spark is stopped.
The native library is process-lived; unloading it while the JVM continues running
is unsupported. A failed drain is reported, and resources can remain retained
until teardown succeeds or the process exits.

If `NativeEngineServer.close()` reports a JNI attachment failure, the server keeps
its cleanup handles and admission reservations; a later explicit `close()` can
retry after attachment recovers. Worker detachment must run on the owning OS
thread. A detach failure that persists through its bounded same-thread retry
retains the attachment and makes shutdown fail; a later close on another thread
cannot release it. Restart the hosting JVM to recover. A Java cleanup exception
can occur after Spark or Arrow has released only part of a resource. Such failures
retain their owners and the original error until JVM exit. Repeated `close()` calls
report that error without replaying potentially non-idempotent cleanup or claiming
a successful drain. Restart the hosting JVM to recover from a persistent cleanup
failure.

## Bundle native libraries in JARs

The Java adapter is shared across operating systems and CPU architectures for a
compatible Spark/Scala/JDK combination. Native code is packaged separately for
`linux-x86_64`, `linux-aarch64`, `macos-x86_64` and `macos-aarch64`. Each platform
JAR contains the library and version, ABI and SHA-256 metadata under
`META-INF/native/<platform>/`. The artifact version comes from `jni/Cargo.toml`;
it currently is `0.1.0`, independently of the client workspace's Spark version.

After building the native library and Java adapter, package them with Python 3:

```bash
crates/spark-connect-server/jvm/build.sh
python3 crates/spark-connect-server/jvm/package-native.py \
  --library "$NATIVE_LIBRARY" \
  --adapter-jar crates/spark-connect-server/.build/jvm/spark-connect-native-bridge.jar \
  --output-dir crates/spark-connect-server/.build/jars
```

This produces a shared `spark-connect-native-adapter-0.1.0.jar`, the current
platform's `spark-connect-native-0.1.0-<platform>.jar`, and checksum sidecars.
The packager checks the actual ELF or Mach-O header against the selected
platform. Selecting another platform does not cross-compile the library.
Native JAR entries use ZIP DEFLATE compression at level 9. Archive metadata is
normalized for reproducible packaging with the same toolchain; compressed bytes
can differ between zlib versions.

Every platform, all-platform and macOS universal JAR includes the license and
notice texts for the JNI library's normal, non-proc-macro dependencies across
all four supported Cargo targets. They are stored in `META-INF/LICENSE`,
`META-INF/NOTICE` and `META-INF/native-dependencies.json`. The inventory records
whole-file hashes for the `inputs` in `jvm/native-licenses/dependencies.json`.
Run `python3 scripts/gen_native_binary_licenses.py --check` before packaging or
pushing. If it reports stale assets, run `python3 scripts/gen_native_binary_licenses.py`
and commit the regenerated `jvm/native-licenses/` assets, even if the JNI dependency
closure is unchanged. Packaging checks input and content hashes without running Cargo.

For example, launch the packaged native library on macOS ARM64:

```bash
crates/spark-connect-server/jvm/run-engine.sh --native-jar \
  "$PWD/crates/spark-connect-server/.build/jars/spark-connect-native-0.1.0-macos-aarch64.jar" \
  127.0.0.1:15002
```

For a deployment using already built JARs, pass the platform JAR with
`spark-submit --jars`, use the shared adapter JAR as the application JAR, and
invoke `EmbeddedSparkEngine --bundled [listen-address]`. An existing host can
use `new NativeEngineServer(listenAddress, engine)` with both JARs on its
classpath. The shared adapter JAR must also be available to Spark executors,
where its Java code converts result rows to Arrow. Using it as the application
JAR with `spark-submit` distributes it automatically. A host that creates its
own SparkContext must distribute the adapter through `spark.jars` at context
creation or `SparkContext.addJar` before querying. The host owns this classpath
configuration; the adapter does not change a shared context's submitted JARs.
Executors do not load the native library. The explicit absolute-library-path
constructor remains available.

The loader detects the host platform, checks matching artifact and ABI versions
and the library checksum, then extracts the library into a private temporary
directory before calling `System.load`. Exactly one matching adapter metadata,
platform metadata and library resource must be visible to the adapter's defining
classloader. Mixing duplicate or incompatible native JARs fails explicitly.
Use one defining Java adapter classloader per JVM. The loader's singleton guard
is scoped to that classloader; loading another adapter copy can load a separate
native library and independent runtime state.
The extracted library stays available for the process lifetime and is removed
at normal JVM exit. Abrupt process termination can leave the temporary directory.
If the default temporary filesystem is mounted `noexec`, set `-Djava.io.tmpdir`
(through `spark.driver.extraJavaOptions` for Spark) to a private directory on a
filesystem that permits loading native libraries.

The native CI matrix builds and tests each of the four targets on a runner with
the matching architecture, uploads the platform JARs, and combines them into
`spark-connect-native-0.1.0-all.jar` and a macOS universal JAR. Applications still
need the shared Java adapter JAR. The `--combine` command requires all four
platform artifacts:

```bash
python3 crates/spark-connect-server/jvm/package-native.py \
  --combine /path/to/platform-jars/spark-connect-native-0.1.0-linux-x86_64.jar \
    /path/to/platform-jars/spark-connect-native-0.1.0-linux-aarch64.jar \
    /path/to/platform-jars/spark-connect-native-0.1.0-macos-x86_64.jar \
    /path/to/platform-jars/spark-connect-native-0.1.0-macos-aarch64.jar \
  --output-dir crates/spark-connect-server/.build/jars
```

To bundle both macOS architectures in one JAR:

```bash
python3 crates/spark-connect-server/jvm/package-native.py \
  --combine-macos /path/to/platform-jars/spark-connect-native-0.1.0-macos-x86_64.jar \
    /path/to/platform-jars/spark-connect-native-0.1.0-macos-aarch64.jar \
  --output-dir crates/spark-connect-server/.build/jars
```

This requires exactly the two macOS platform artifacts and produces
`spark-connect-native-0.1.0-macos-universal.jar`. It contains both thin native
libraries; the loader selects the JVM's architecture, including under Rosetta,
and extracts only that library. Pass this JAR with `--native-jar` or
`spark-submit --jars` using the same launch commands above. CI checks loading the
same universal JAR on both macOS architectures.

A Universal2 Mach-O library also contains both architectures' machine code, so
combining them into a fat binary does not itself remove those bytes. ZIP
compression reduces the universal JAR's stored and transfer size; the selected
library is decompressed when extracted.

Linux artifacts are built on Ubuntu 22.04 and require glibc 2.35 or newer; musl
is outside this matrix. macOS artifacts target macOS 11 or newer. These are
development CI artifacts, with no Maven Central publication in this WIP. Native
packaging does not expand the Spark assembly compatibility described above.

## Supported scope

The frontend implements execution, configuration, session and operation release,
and the schema, `isLocal` and `isStreaming` analyses. The engine accepts these
relation messages, including their nested inputs:

- `RANGE`, `LOCAL_RELATION`, `SQL`, `PROJECT`, `FILTER`, `LIMIT`, `OFFSET`, `TAIL`.
- `JOIN`, `SORT`, `AGGREGATE`, `DEDUPLICATE`, `SET_OP`, `SUBQUERY_ALIAS`, `TO_DF`.
- `WITH_COLUMNS_RENAMED`, `WITH_COLUMNS`, `DROP`, `REPARTITION`,
  `REPARTITION_BY_EXPRESSION`, `HINT`, `READ`.

The protobuf `READ` message accepts only its named-table form. This restricts the
message shape, not data access: `SQL` and named-table resolution follow the
hosting Spark session's semantics, including SQL-on-files and built-in reflective
functions when enabled there. Clients can change modifiable session SQL settings,
such as `spark.sql.runSQLOnFiles`, through Config Set. Queries execute with the
hosting process's resource access. This list does not establish support for every
expression or result type inside the accepted messages. Writes, streaming
execution and user-defined functions are outside this prototype's scope.

Spark-owned internal annotations such as `__autoGeneratedAlias` are intentionally
omitted from returned schemas. User field metadata, including column comments,
non-default string collations, user-defined types and CHAR/VARCHAR return
`UNIMPLEMENTED`; Arrow schema conversion cannot preserve their information.
Duplicate field names within nested structs are also rejected, including structs
inside arrays or map keys and values, because Spark's Arrow conversion renames
them. Duplicate top-level column names and case-distinct nested names are preserved.
CHAR/VARCHAR annotations on string-typed table columns
(`__CHAR_VARCHAR_TYPE_STRING`) are also rejected. Ordinary unaliased computed
columns and aggregates remain supported within the other prototype limits.

SQL commands such as `SHOW`, `DESCRIBE`, `SET`, `USE` and `CACHE`, Catalog API
relations and `CACHED_LOCAL_RELATION` are also rejected. Streaming relations are
rejected during planning, including `isStreaming` analysis; that analysis does
not return `true` for unsupported streaming inputs.

Common unsupported client calls include `df.show()`, `df.printSchema()`,
`df.explain()`, `spark.version`, `df.describe()`, `df.summary()`, `df.sample()`,
`df.na.*`, `df.unpivot()`, `spark.read.parquet()` and `df.cache()`.

Configuration is session scoped. Preparation captures analysis and Arrow
metadata, while physical planning starts on the first fetch. These stages do
not share a transactional configuration snapshot; a concurrent configuration
change can affect later physical planning. Config reads apply the hosting
SparkContext's `spark.redaction.regex` to keys and values; a session setting with
the same name does not replace that policy. This follows newer Spark Config
redaction behavior, which is absent from the pinned Spark 4.2 smoke-test assembly.

Silent Config Set applies keys sequentially and is not atomic across keys.
Earlier successful writes remain applied if a later key encounters cancellation,
an admission timeout or another operation-level failure. All keys share the
original RPC's admission deadline; a long list does not reset that budget.

The first fetch starts one daemon execution worker for the query. It opens one
Spark SQL execution and submits all output partitions as one result job.
Partitions can execute concurrently; the cursor returns their batches in
partition order. The execution scope stays open until the job finishes. Closing
the cursor cancels the job and waits for its worker to exit before releasing the
plan. Initializing a zero-partition plan also creates one execution.
Shuffle-file cleanup remains unsupported pending validation: prepare and analysis
reject `spark.sql.classic.shuffleDependency.fileCleanup.enabled=true` with
`UNIMPLEMENTED`, including when the hosting engine inherits that setting.

`SPARK_TESTING` or `spark.testing` can make Spark default this setting to `true`.
In test environments, explicitly set
`spark.sql.classic.shuffleDependency.fileCleanup.enabled=false` in the hosting
Spark configuration or the native session before preparing or analyzing queries.

The frontend's [protocol subset](#protocol-subset) and
[session lifetime limits](#lifetime-and-trust-requirements) also apply. Result
types are limited by the Arrow adapter. This is not full Spark SQL or Spark
Connect parity.

## Resource and protocol limits

Each embedded engine has fixed capacities of 64 initialized JVM sessions and
64 active engine tasks. Session and task permit admission share a two-second
deadline per operation; exhausted capacity returns `RESOURCE_EXHAUSTED`.
These limits are not configurable, and there is no idle session expiry.
A client that exits without releasing its session can leave its session permit
reserved. An open, unread result stream can retain a task permit until it is
completed or cancelled and its cleanup finishes.
Each started query also owns one daemon execution worker.

Call `spark.stop()` on normal client shutdown. Capacity can recover by releasing
known sessions with `ReleaseSession` and allowing their owned work and deferred
cleanup to drain, or by restarting the server. Releasing an operation or closing
its result stream can also release its task capacity; it does not release the
session. Abandoned sessions whose identifiers are lost remain until a restart.
`ReleaseExecute` with `ReleaseAll` waits up to two seconds for frontend operation
teardown, then returns `DEADLINE_EXCEEDED` if that teardown is still pending.
JNI cancellation and deferred cleanup can continue even after a successful
`ReleaseAll`. Their resource owners and capacity remain retained until cleanup
actually finishes; a timeout does not release that capacity early.

A cleanup failure while serving keeps its admission permits reserved and is not
retried automatically. Failed cancellation delivery retains its task handle and
task admission even if the query later closes successfully. That capacity stays
unavailable until an explicit `NativeEngineServer.close()` retries the failure
while stopping the server, or the hosting JVM restarts. A transient JNI
attachment failure can recover during that close. A persistent worker
detach failure or a partial Java cleanup failure remains retained until the
hosting JVM restarts.

Deferred session closes run serially on one process-wide cleanup lane shared by
all embedded engines using the native library. A slow or blocked session close
therefore delays later session cleanup and session-capacity recovery in other
engines too. Session permits remain reserved until the corresponding Java close
completes successfully. Task cancellation has a separate process-wide lane;
a slow cancellation delays other cancellations on that lane. Query cleanup has
its own lane, so session teardown does not directly block task-capacity recovery
from task cancellation or query cleanup. These lanes do not isolate engines from
other shared Spark resources or from a query's own slow cleanup.

Native server close waits up to ten seconds for shutdown. If work still owns
JNI resources, close reports an incomplete shutdown and retains its handle for
a later retry. A serving error is also reported on the first close; once drain
is confirmed, another close can finalize that retained handle.

The frontend limits each decoded gRPC request to 128 MiB. Relation inputs for
planning and metadata cross JNI without a separate 64 MiB cap. Through this
listener, they are bounded by the request limit and JNI's `jsize` (`INT_MAX`)
limit. Arrow IPC payloads returned through JNI and encoded configuration fields
remain capped at 64 MiB. Spark's Arrow converter limits rows per batch using
`spark.sql.execution.arrow.maxRecordsPerBatch` and currently uses a 2 MiB byte
estimate before serialization, independently of frontend chunk sizing.
Nonpositive row limits use a 10,000-row fallback. The JNI Arrow output size check
happens after conversion; these limits do not bound the converter's peak
allocation or guarantee that a large row will fit.

Arrow framing follows the negotiated [byte-chunking behavior](#arrow-results);
the frontend does not row-split or cap Arrow batches. Without client opt-in, a
whole JNI batch can exceed a raw client's gRPC receive limit and fail at the
client. The independent JNI 64 MiB Arrow IPC payload bound applies before
frontend chunking, so a batch rejected by JNI never reaches it. Schema and
eager-command response envelopes retain a separate 3 MiB limit. Execution panics return
`INTERNAL` rather than ending a partial result stream successfully.

Java uses its standard protobuf recursion limit of 100 nested message levels,
in addition to the frontend's [Prost nesting behavior](#protocol-subset). A
DataFrame operation can add several message levels, so this is not a limit of
100 chained operations.

Spark executors use `ArrowConverters.toBatchWithSchemaIterator`, the same helper
as the JVM Connect server. Each batch contains a complete Arrow IPC stream and
its row count. Spark computes partitions concurrently and buffers their completed
output on the driver. The cursor returns batches in partition order, and JNI
copies each Java `byte[]` directly into a Rust-owned `Vec` with one
`GetByteArrayRegion` copy, without an intermediate C payload buffer. Moving that
storage into protobuf `Bytes` and taking byte slices do not copy its payload;
Prost copies it when encoding each outgoing gRPC message. The frontend reads the
IPC schema and preserves the existing IPC bytes, including compressed buffers,
whether it sends a complete batch or negotiated byte chunks. It does not decode,
decompress or re-encode producer IPC to split responses.

With the default `spark.sql.execution.arrow.compression.codec=none`, partition
transfer uses uncompressed Arrow IPC. Nondefault executor/JNI codec
interoperability remains unvalidated. Network volume and the driver heap needed
for partition output can exceed Spark's compressed UnsafeRow transfer. A slow
client or an unfinished earlier partition can leave all completed output,
including the entire query result, buffered on the driver heap. Spark applies
`spark.driver.maxResultSize` to the combined serialized results of the job; this
does not bound retained heap or total query memory. A job can exceed that limit
even when compressed rows would fit; result-size aborts are reported as
`INTERNAL`. Batch bounds do not bound one materialized partition's output.

Arrow conversion uses Spark's allocator and cleanup behavior. Its off-heap
allocations require executor memory overhead, and Spark can retry conversion
failures according to `spark.task.maxFailures`. Executor Arrow allocation
failures are reported as `INTERNAL`, matching the JVM Connect server; the native
IPC size check reports `RESOURCE_EXHAUSTED`.

## Historical performance observations

These historical measurements used an earlier prototype revision with a
different result-framing frontend. They have not been reproduced with the
current byte-chunking path and do not establish its performance.

A November 17, 2026 comparison used native revision
`23b2eb6538b355987341b385cdfccc3222669e4b`, including the direct Java-array-to-Rust
copy, and the unmodified Spark snapshot identified above. The JVM Connect server
and embedded Rust server ran sequentially on the same Linux ARM64 host. This
compares the two servers; it does not isolate the benefit of the copy change.

Each latency column is the geometric mean of four fresh-process fork medians.
The last column is the geometric mean of the four matched Rust/JVM ratios;
positive values mean higher Rust latency.

| Workload | JVM (ms) | Rust (ms) | Rust latency change |
| --- | ---: | ---: | ---: |
| Config Get | 0.837 | 0.429 | -48.8% |
| Range schema analysis | 2.121 | 2.327 | +9.7% |
| Tiny SQL and collect | 33.202 | 47.829 | +44.1% |
| Range/filter aggregate and collect | 49.855 | 78.712 | +57.9% |
| Join aggregate and collect | 72.780 | 92.920 | +27.7% |
| Raw Arrow complete receipt | 246.370 | 382.529 | +55.3% |
| Raw Arrow receipt and table decode | 260.639 | 384.016 | +47.3% |
| Public `DataFrame.toArrow()` | 428.790 | 547.088 | +27.6% |
| Concurrent Config per-call latency | 0.641 | 0.615 | -4.1% |

The public `toArrow()` matched-pair increases ranged from 24.0% to 32.6%.
Concurrent Config was slower in Rust in three of four pairs despite its lower
geometric mean of fork medians; whole-round throughput was 23.3% lower. These
results establish neither a general native speedup nor improved concurrent
capacity.

The four matched pairs ran in AB, BA, BA, AB order, with A = JVM and B = Rust.
Each server and official Python client started in a fresh process. Metadata used
200 warmups and 500 measured calls per fork; queries and Arrow actions used
10 warmups and 30 measured calls; concurrent Config used one warmup and three
measured rounds of 1,000 calls across eight channels. All 13,152 action records
passed their recorded checks, including 9,464 measured records and 3,688 warmups.
Every client/session cleanup stage, server shutdown and listener closure passed.
Arrow inputs had 8 million int64 rows, range/filter inputs 1 million rows, and
join inputs 100,000 rows.

The host had eight visible ARM Neoverse-V2 CPUs and 61.6 GiB visible memory.
Servers used CPUs 2-5 and clients launched with affinity to CPUs 6-7. Both servers
used `local[2]`, a 1 GiB heap, two input/shuffle partitions, UTC, 10,000 Arrow rows
per batch and IPC codec `none`. The runtime was JDK 21.0.13, Python 3.12.3,
official PySpark client 4.2.0, PyArrow 26.0.0 and gRPC 1.84.0. JNI checking was
disabled during timing. Other host services remained present; affinity did not
reserve exclusive CPUs.

Raw receipt drains all protobuf responses through completion, with IPC decoding
outside timing. Public `toArrow()` includes the official client's response
handling, IPC decode, table materialization and schema cast. Validation and
disposal occur outside timing. These are separate sampled actions; subtracting
their times does not measure a pipeline stage. Both servers returned exactly
65,248,000 IPC bytes in 800 record batches per raw Arrow action. Response counts
were 806-810 for JVM and 802 for Rust. Public `collect()` includes Python Row
materialization; tiny SQL also includes `.sql()` construction.

This is one host with local executors and loopback transport. Within-process
calls are correlated; no significance, distributed-network, wide-row, compressed
IPC, GC or retained-memory conclusion follows. The raw checks cover aggregate
count, schema, nulls, sum, minimum and maximum, rather than exact order or
uniqueness. Automatic asynchronous operation release remains best effort;
explicit session/client/server cleanup and listener closure passed separately.

Arrow producer output-sizing optimization and producer/off-heap API redesign
remain follow-up work.

## Validation

### Frontend-only checks

Use the workspace Rust toolchain and a `protoc` compiler with proto3 optional
support. From the repository root:

```bash
cargo test --locked -p spark-connect-server
cargo clippy --locked -p spark-connect-server --all-targets --no-deps -- -D warnings
```

The tests run a real gRPC frontend against deterministic injected engines. They
cover wire framing, schema and IPC handling, backend error propagation,
backpressure, operation ownership, cancellation and session lifetime. Fixture
relations select predefined outputs; they do not implement Spark operators or
establish query semantics or Spark server compatibility.

### JNI and live Spark checks

The deterministic Rust frontend/engine and C bridge tests do not need a
running Spark distribution. JNI compilation still requires `JAVA_HOME`:

```bash
cargo test -p spark-connect-server -p spark-connect-jni
cargo clippy -p spark-connect-server -p spark-connect-jni --all-targets --no-deps -- -D warnings
crates/spark-connect-server/jni/test-c-bridge.sh
crates/spark-connect-server/jni/test-c-bridge.sh --sanitize
python3 -B crates/spark-connect-server/jvm/test_acceptance_guards.py
python3 -B crates/spark-connect-server/jvm/test_package_native.py
crates/spark-connect-server/jvm/test-native-loader.sh
```

The C bridge runner requires macOS or Linux. Sanitizer mode also requires a
compiler with address and undefined-behavior sanitizer support.

With a matching Spark assembly and `NATIVE_LIBRARY` set, run live checks:

```bash
crates/spark-connect-server/jvm/test-engine.sh \
  --library "$NATIVE_LIBRARY" --python /path/to/pyspark-python
```

To include the official Rust client example in the live checks, build it and
pass the optional `--rust-example` argument:

```bash
cargo build --release -p spark-connect-server --example rust_client
crates/spark-connect-server/jvm/test-engine.sh \
  --library "$NATIVE_LIBRARY" --python /path/to/pyspark-python \
  --rust-example "$PWD/target/release/examples/rust_client"
```

This compiles Java tests and checks direct planning, IPC batch transfer,
all-partition scheduling and ordered delivery, native lifecycle, and
cleanup-failure retention with `-Xcheck:jni`, then runs official PySpark and raw
gRPC acceptance against `EmbeddedSparkEngine`. It checks JVM listener ownership,
shutdown and listener closure. The Python interpreter must have a compatible
official PySpark Connect client, gRPC and PyArrow installed. Listener checks
prefer `lsof` when it is installed and otherwise use Linux `/proc`; macOS requires
`lsof`.
`NATIVE_TEST_PYTHON` (default `python3`) selects the interpreter for the standard
library guard tests and orchestration. `--python` selects the official PySpark
client and worker interpreter; it defaults to the orchestration interpreter.
These interoperability tests complement the Rust tests. The default report is
`crates/spark-connect-server/.build/jvm-native/engine-acceptance.json`; pass
`--output /absolute/report.json` to choose another path.

Pass `--native-jar /absolute/platform-or-all.jar` as well as `--library` to run
the client acceptance through classpath extraction. The lifecycle checks still
exercise explicit-path loading. This mode checks the extracted library bytes
against `--library`, and verifies that the extracted file and private directory
are removed after JVM exit.

The native CI workflow runs the deterministic Rust tests, strict native Clippy,
acceptance-runner guards, packaging tests and C ownership checks on its platform
matrix. It builds release libraries, packages platform/all-platform JARs and
tests actual native loading with a JDK-only loader suite. A separate Linux job
restores a Spark assembly cache keyed on the pinned revision above, platform and
JDK, or builds that revision from source on a cache miss. It compiles the
Spark-dependent adapter and runs `jvm/test-jni-smoke.sh` against the real native
library. The smoke test checks a range query's schema and Arrow values,
negotiated 1 KiB byte chunks with reassembly, a whole row larger than 3 MiB without
chunking opt-in, `ResultComplete`, session release, native close, listener rebind
and shared SparkContext survival under `-Xcheck:jni`. It does not exercise Arrow
IPC input or establish compatibility with other Spark revisions. The broader acceptance
suite still requires a selected, compatible assembly and local validation. Its
recorded full run used the separate Arrow input lifecycle repair described under
build requirements; the CI smoke does not apply that repair.

Local validation on macOS ARM64 used Rust 1.100 and covered the frontend and
JNI tests, client release tests and integration-test compilation, acceptance-runner guards,
license generators and packaging. Rustfmt, workspace Clippy, C bridge checks
with and without sanitizers, native loader checks and the real JNI smoke also
passed. Full client acceptance used the separately repaired Spark runtime
described under [Build requirements and compatibility](#build-requirements-and-compatibility).
The smoke passed on the unmodified pinned snapshot. The optional Rust client
example and MSRV 1.88 checks were not rerun for this revision; the hosted frontend
and native CI workflows include Rust 1.88 jobs. These local results do not
establish Linux or x86-64 compatibility; hosted CI and nondefault executor/JNI
IPC compression remain separate validation requirements. Performance was not
remeasured.
