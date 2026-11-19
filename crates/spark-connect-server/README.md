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

# Experimental Spark Connect server frontend

This unpublished library serves a subset of the Spark Connect protocol using the
workspace's published `apache-spark-connect-proto` messages and its optional
`server` feature. This additive feature exposes generated server bindings in the
protocol crate's public API; Cargo unifies it with other uses of that crate in
the workspace. The caller supplies an
`Engine` implementation and a bound Tokio TCP listener to `serve_with_engine`.
Use `serve_with_engine_and_options` with `ServerOptions` to configure Arrow
chunking.
No execution engine or standalone server executable is included.

Rust owns protocol sessions, operation IDs, response framing and result-stream
lifetimes. Each protocol session receives an `EngineSession` that owns its
planning, catalog and configuration state. Its `prepare` method returns a
`PreparedQuery` or command response payloads; the frontend adds protocol IDs and
`ResultComplete` for requests that opt into reattachment. The engine decides
which plans, commands and metadata requests it can execute and reports
unsupported inputs explicitly.

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
120,795,955 bytes (90% of 128 MiB), matching the JVM server's default chunk payload
limit. It limits IPC payload bytes, excluding the protobuf envelope. Setting it
to zero disables chunking even when the client opts in. An opted-in client must
accept the configured maximum plus its protobuf envelope, or request a smaller
chunk size. A 128 MiB receive limit accommodates the default maximum; raw tonic's
4 MiB default does not accommodate chunks above 4 MiB.

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
channel, so unread responses eventually stop polling the engine. Blocking Arrow
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
Active sessions and operations
have no admission limit or idle expiry; only released-session tombstones are
bounded, to the most recent 4096 distinct IDs. Spark instead uses a size-bounded
cache configured by `spark.connect.session.manager.closedSessionsTombstonesSize`,
which defaults to 1000; this frontend's fixed bound and release-order eviction
policy differ. An evicted ID can create a new session. These limits do not
provide isolation or complete memory and CPU quotas.
Execution permissions and resource access are the engine's responsibility.

## Build and test

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
