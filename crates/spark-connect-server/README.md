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

- Rust 1.88 or newer, Cargo and `protoc`.
- JDK 21, a C compiler with C11/pthread support and JNI headers. Set `JAVA_HOME`
  to the JDK directory.
- Python 3.9 or newer for artifact metadata and packaging. Java build scripts use
  `python3`; set `NATIVE_BUILD_PYTHON` to select another interpreter.
- macOS or Linux. Windows is not supported by this prototype.
- A compatible Spark assembly containing the shaded Connect protobuf/gRPC jars.
  Set `SPARK_HOME` to its directory. The scripts accept jars in either
  `$SPARK_HOME/jars` or `$SPARK_HOME/assembly/target/scala-2.13/jars`.

The workspace currently tracks the Spark 4.3.0 protocol; its pin is recorded in
[PROTO_VERSION.txt](../spark-connect-proto/proto/PROTO_VERSION.txt). The JNI
implementation was developed and validated with Spark `4.2.0-SNAPSHOT`, revision
`250ccca3f2b5f043e20b415adf17bda5d202c60f`, Scala 2.13.17 and JDK 21.0.10. It depends
on internal Spark APIs, including `SessionHolder`, `SparkConnectPlanner`,
`QueryExecution` and Arrow conversion helpers. That validation does not establish
Spark 4.3.0 assembly compatibility or support for every 4.3.0 protocol feature.
Other assemblies may require Java source changes and fresh acceptance checks.

The `spark-connect-server` and nested `spark-connect-jni` crates are unpublished,
optional workspace members excluded from `default-members`. Build them explicitly
with `-p`; the server crate provides the frontend library and the JNI crate builds
the JVM-loaded native library. There is no standalone server executable.

All commands below run from the repository root. Cargo places native artifacts in
`target/`; the JVM scripts place generated Java artifacts and acceptance reports
under `crates/spark-connect-server/.build/`.

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
classpath. The explicit absolute-library-path constructor remains available.

The loader detects the host platform, checks matching artifact and ABI versions
and the library checksum, then extracts the library into a private temporary
directory before calling `System.load`. Exactly one matching adapter metadata,
platform metadata and library resource must be visible to the adapter's defining
classloader. Mixing duplicate or incompatible native JARs fails explicitly.
The extracted library stays available for the process lifetime and is removed
at normal JVM exit. Abrupt process termination can leave the temporary directory.
If the default temporary filesystem is mounted `noexec`, set `-Djava.io.tmpdir`
(through `spark.driver.extraJavaOptions` for Spark) to a private directory on a
filesystem that permits loading native libraries.

The native CI matrix builds and tests each of the four targets on a runner with
the matching architecture, uploads the platform JARs, and combines them into
`spark-connect-native-0.1.0-all.jar`. The all-platform JAR includes the native
resources; applications still need the shared Java adapter JAR. To combine
downloaded platform artifacts locally:

```bash
python3 crates/spark-connect-server/jvm/package-native.py \
  --combine /path/to/platform-jars/spark-connect-native-0.1.0-linux-x86_64.jar \
    /path/to/platform-jars/spark-connect-native-0.1.0-linux-aarch64.jar \
    /path/to/platform-jars/spark-connect-native-0.1.0-macos-x86_64.jar \
    /path/to/platform-jars/spark-connect-native-0.1.0-macos-aarch64.jar \
  --output-dir crates/spark-connect-server/.build/jars
```

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

`READ` is restricted to named tables. This list describes the accepted message
forms; it does not establish support for every expression or result type inside
them. Writes, streaming execution and user-defined functions are outside this
prototype's scope.

Spark-owned internal annotations such as `__autoGeneratedAlias` are intentionally
omitted from returned schemas. User field metadata, including column comments,
non-default string collations, user-defined types and CHAR/VARCHAR return
`UNIMPLEMENTED`; Arrow schema conversion cannot preserve their information.
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
change can affect later physical planning.

Each batch fetch creates a separate Spark SQL execution ID. The Spark UI and
SQL metrics therefore record a query as multiple executions rather than one
query-wide execution. The setting
`spark.sql.classic.shuffleDependency.fileCleanup.enabled=true` is unsupported;
prepare and analysis reject it with `UNIMPLEMENTED`, including when the hosting
engine inherits that setting. Per-fetch execution completion could otherwise
delete shuffle files still needed by later fetches.

`SPARK_TESTING` or `spark.testing` can make Spark default this setting to `true`.
In test environments, explicitly set
`spark.sql.classic.shuffleDependency.fileCleanup.enabled=false` in the hosting
Spark configuration or the native session before preparing or analyzing queries.

Reattachment, artifact upload/status, interruption, enriched error details,
session cloning and status queries return `UNIMPLEMENTED`. Result types are also
limited by the Arrow adapter. This is not full Spark SQL or Spark Connect parity.
Closed-session tombstones retain up to 4096 distinct session keys per listener;
older evicted keys can create a new session again.

## Resource and protocol limits

Each embedded engine has fixed capacities of 64 initialized JVM sessions and
64 active engine tasks. Session and task permit admission share a two-second
deadline per operation; exhausted capacity returns `RESOURCE_EXHAUSTED`.
These limits are not configurable, and there is no idle session expiry.
A client that exits without releasing its session can leave its session permit
reserved. An open, unread result stream can retain a task permit until it is
completed or cancelled and its cleanup finishes.

Call `spark.stop()` on normal client shutdown. Capacity can recover by releasing
known sessions with `ReleaseSession` and allowing their owned work and deferred
cleanup to drain, or by restarting the server. Releasing an operation or closing
its result stream can also release its task capacity; it does not release the
session. Abandoned sessions whose identifiers are lost remain until a restart.

The frontend limits each decoded gRPC request to 64 MiB. JNI byte payloads and
the retained Arrow buffer capacities of one imported batch are also capped at
64 MiB. Each Arrow IPC response payload is limited to 3 MiB; larger batches are
split by rows. A single row or schema that cannot fit fails with
`RESOURCE_EXHAUSTED`.

Rust and Java use their standard protobuf recursion limit of 100 nested message
levels. A DataFrame operation can add several message levels, so this is not a
limit of 100 chained operations. Deep plans can fail before engine planning;
the Rust transport currently reports a protobuf recursion error as `INTERNAL`.

Java converts Spark rows into Arrow vectors and leases their immutable value
buffers to Rust through the Arrow C Data Interface. Rust retains the Java owner
until the imported arrays are released. This avoids copying vector data across
the JNI boundary. Spark still converts rows into vectors, descriptor and schema
metadata cross JNI, and Rust serializes Arrow IPC for gRPC. The complete query
path is not zero copy. Batch bounds do not bound Spark's partition materialization
or total query memory.

## Validation

The deterministic Rust frontend/engine tests and C ownership tests do not need a
running Spark distribution. JNI compilation still requires `JAVA_HOME`:

```bash
cargo test -p spark-connect-server -p spark-connect-jni
cargo clippy -p spark-connect-server -p spark-connect-jni --all-targets --no-deps -- -D warnings
crates/spark-connect-server/jni/test-c-export.sh
crates/spark-connect-server/jni/test-c-export.sh --sanitize
python3 -B crates/spark-connect-server/jvm/test_acceptance_guards.py
python3 -B crates/spark-connect-server/jvm/test_package_native.py
crates/spark-connect-server/jvm/test-native-loader.sh
```

The C ownership runner supports macOS and Linux; sanitizer mode requires a
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

This compiles Java tests, checks direct planning, Arrow ownership and native
lifecycle with `-Xcheck:jni`, then runs official PySpark and raw gRPC acceptance
against `EmbeddedSparkEngine`. It checks JVM listener ownership, shutdown and
listener closure. The Python interpreter must have a compatible official PySpark
Connect client, gRPC and PyArrow installed. Listener checks prefer `lsof` when it
is installed and otherwise use Linux `/proc`; macOS requires `lsof`.
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
tests actual native loading with a JDK-only loader suite. It does not download a
Spark snapshot, compile the Spark-dependent Java adapter or run live Spark/JNI
acceptance. Those require the explicitly
selected assembly and must be validated locally. Passing native CI alone does not
establish JVM compatibility.
