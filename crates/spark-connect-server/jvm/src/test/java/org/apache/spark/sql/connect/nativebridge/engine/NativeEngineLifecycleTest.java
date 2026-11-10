// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

package org.apache.spark.sql.connect.nativebridge.engine;

import java.io.ByteArrayInputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.spark.connect.proto.AnalyzePlanRequest;
import org.apache.spark.connect.proto.AnalyzePlanResponse;
import org.apache.spark.connect.proto.ConfigRequest;
import org.apache.spark.connect.proto.ExecutePlanRequest;
import org.apache.spark.connect.proto.ExecutePlanResponse;
import org.apache.spark.connect.proto.Expression;
import org.apache.spark.connect.proto.KeyValue;
import org.apache.spark.connect.proto.Plan;
import org.apache.spark.connect.proto.Project;
import org.apache.spark.connect.proto.Range;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.connect.proto.ReleaseExecuteRequest;
import org.apache.spark.connect.proto.ReleaseSessionRequest;
import org.apache.spark.connect.proto.SQL;
import org.apache.spark.connect.proto.SparkConnectServiceGrpc;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.SparkSessionExtensions;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.catalyst.rules.Rule;
import org.apache.spark.sql.connect.service.SparkConnectService$;
import org.apache.spark.sql.internal.SQLConf;
import org.apache.spark.sql.util.ArrowUtils$;
import org.sparkproject.connect.grpc.ManagedChannel;
import org.sparkproject.connect.grpc.ManagedChannelBuilder;
import org.sparkproject.connect.grpc.Status;
import org.sparkproject.connect.grpc.StatusRuntimeException;

import scala.runtime.AbstractFunction1;
import scala.runtime.BoxedUnit;

/** Native reader/writer ordering, worker context reuse and runtime teardown under -Xcheck:jni. */
public final class NativeEngineLifecycleTest {
  private static final String GATE_KEY = "spark-connect.native.analysis.gate";
  private static final String CONTEXT_KEY = "spark-connect.native.context";
  private static final AtomicBoolean RECORD = new AtomicBoolean();
  private static final Set<Thread> ANALYZER_THREADS = ConcurrentHashMap.newKeySet();
  private static final AtomicReference<AnalysisGate> ANALYSIS_GATE = new AtomicReference<>();
  private static final AtomicReference<ContextProbe> CONTEXT_PROBE = new AtomicReference<>();

  private NativeEngineLifecycleTest() {}

  public static void main(String[] arguments) throws Exception {
    require(arguments.length == 1, "pass the absolute native library path");
    SparkSession spark = SparkSession.builder().appName("Native engine worker lifecycle")
        .config("spark.sql.artifact.isolation.enabled", "true")
        .config("spark.sql.artifact.isolation.alwaysApplyClassloader", "true")
        .withExtensions(new AbstractFunction1<SparkSessionExtensions, BoxedUnit>() {
          @Override
          public BoxedUnit apply(SparkSessionExtensions extensions) {
            extensions.injectResolutionRule(new AbstractFunction1<SparkSession, Rule<LogicalPlan>>() {
              @Override
              public Rule<LogicalPlan> apply(SparkSession ruleSession) {
                return new Rule<LogicalPlan>() {
                  @Override
                  public LogicalPlan apply(LogicalPlan plan) {
                    if (RECORD.get()) ANALYZER_THREADS.add(Thread.currentThread());
                    ContextProbe probe = CONTEXT_PROBE.get();
                    if (probe != null) probe.observe(ruleSession);
                    AnalysisGate gate = ANALYSIS_GATE.get();
                    if (gate != null
                        && gate.marker.equals(ruleSession.conf().get(GATE_KEY, ""))) {
                      gate.enter(ruleSession);
                    }
                    return plan;
                  }
                };
              }
            });
            return BoxedUnit.UNIT;
          }
        }).getOrCreate();
    spark.sparkContext().setLogLevel("ERROR");
    ExecutorService callers = Executors.newFixedThreadPool(8);
    try {
      for (int restart = 0; restart < 3; restart++) {
        int port;
        try (SparkJvmEngine engine = new SparkJvmEngine(spark);
             NativeEngineServer server = new NativeEngineServer(
                 arguments[0], "127.0.0.1:0", engine)) {
          String address = server.address();
          port = Integer.parseInt(address.substring(address.lastIndexOf(':') + 1));
          ManagedChannel channel = ManagedChannelBuilder.forTarget(address).usePlaintext().build();
          try {
            String session = UUID.randomUUID().toString();
            AnalyzePlanRequest request = AnalyzePlanRequest.newBuilder().setSessionId(session)
                .setSchema(AnalyzePlanRequest.Schema.newBuilder().setPlan(Plan.newBuilder()
                    .setRoot(Relation.newBuilder().setRange(Range.newBuilder()
                        .setStart(0).setEnd(10).setStep(1))))).build();
            var stub = SparkConnectServiceGrpc.newBlockingStub(channel);
            for (int warmup = 0; warmup < 32; warmup++) check(analyze(stub, request));
            nativeReadersAndWriter(stub, callers, session, request);
            ANALYZER_THREADS.clear();
            RECORD.set(true);
            try {
              List<Callable<Void>> work = new ArrayList<>();
              for (int worker = 0; worker < 8; worker++) {
                work.add(() -> {
                  for (int call = 0; call < 32; call++) check(analyze(stub, request));
                  return null;
                });
              }
              for (Future<Void> result : callers.invokeAll(work, 60, TimeUnit.SECONDS)) {
                result.get(1, TimeUnit.SECONDS);
              }
            } finally {
              RECORD.set(false);
            }
            // Re-attaching each RPC creates a new Java Thread identity even when
            // Tokio reuses its OS thread. A live attached pool reuses identities.
            require(!ANALYZER_THREADS.isEmpty(), "analyzer observer was not exercised");
            require(ANALYZER_THREADS.size() < 64,
                "native workers did not reuse JVM attachments: " + ANALYZER_THREADS.size());
            String secondSession = UUID.randomUUID().toString();
            twoSessionWorkerReuse(stub, session, secondSession);
            deepArrowDescriptors(stub, session);
            emptyArrowDescriptors(stub, session);
            lossySchemasAreRejected(stub, session, spark);
            for (String released : List.of(session, secondSession)) {
              require(stub.withDeadlineAfter(30, TimeUnit.SECONDS).releaseSession(
                  ReleaseSessionRequest.newBuilder().setSessionId(released).build())
                  .getSessionId().equals(released), "session release identity changed");
            }
          } finally {
            channel.shutdownNow();
            require(channel.awaitTermination(10, TimeUnit.SECONDS), "client channel did not stop");
          }
          require(SparkConnectService$.MODULE$.server() == null, "Java Connect listener appeared");
          server.close();
          // Check immediately after nativeStop, before try-with-resources calls engine.close().
          require(engine.isClosed(),
              "native stop returned before closing its Java engine factory");
        }
        try (ServerSocket rebound = new ServerSocket()) {
          rebound.setReuseAddress(true);
          rebound.bind(new InetSocketAddress(InetAddress.getLoopbackAddress(), port));
          require(rebound.isBound(), "native listener could not rebind after teardown");
        }
        require(!spark.sparkContext().isStopped(), "runtime close stopped shared SparkContext");
      }
      System.out.println("NATIVE_ENGINE_LIFECYCLE_OK: native concurrent metadata/config readers; "
          + "synchronous Java engine factory close before native stop returns; "
          + "Config writer exclusion/readback; two-session worker context reuse; attachment reuse; "
          + "12-level struct/list and all-empty variable-width Arrow descriptor export and values; "
          + "lossy schema/shuffle-cleanup rejection; three runtime restarts; session release; "
          + "listener rebind; shared SparkContext survives");
    } finally {
      RECORD.set(false);
      CONTEXT_PROBE.set(null);
      AnalysisGate gate = ANALYSIS_GATE.get();
      if (gate != null) gate.close();
      callers.shutdownNow();
      require(callers.awaitTermination(10, TimeUnit.SECONDS), "native client workers did not stop");
      spark.stop();
    }
  }

  private static final class AnalysisGate implements AutoCloseable {
    private final String marker = UUID.randomUUID().toString();
    private final String configKey;
    private final CountDownLatch entered = new CountDownLatch(2);
    private final CountDownLatch proceed = new CountDownLatch(1);
    private final Set<Thread> threads = ConcurrentHashMap.newKeySet();

    private AnalysisGate(String configKey) {
      this.configKey = configKey;
    }

    private void enter(SparkSession ruleSession) {
      if (!threads.add(Thread.currentThread())) return;
      entered.countDown();
      try {
        require(proceed.await(30, TimeUnit.SECONDS), "native analysis gate was not released");
        require("before".equals(ruleSession.conf().get(configKey)),
            "native Config writer changed state while metadata readers held it");
      } catch (InterruptedException error) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException("native analysis gate interrupted", error);
      }
    }

    @Override
    public void close() {
      ANALYSIS_GATE.compareAndSet(this, null);
      proceed.countDown();
    }
  }

  private static void nativeReadersAndWriter(
      SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub, ExecutorService callers,
      String session, AnalyzePlanRequest request) throws Exception {
    String key = "spark-connect.native.concurrent.value";
    try (AnalysisGate gate = new AnalysisGate(key)) {
      setConfig(stub, session, GATE_KEY, gate.marker, key, "before");
      require(ANALYSIS_GATE.compareAndSet(null, gate), "another native analysis gate is active");
      Future<AnalyzePlanResponse> first = callers.submit(() -> analyze(stub, request));
      Future<AnalyzePlanResponse> second = callers.submit(() -> analyze(stub, request));
      require(gate.entered.await(10, TimeUnit.SECONDS),
          "Rust/JNI same-session analyses did not enter Spark concurrently");

      Future<String> read = callers.submit(() -> getConfig(stub, session, key));
      require("before".equals(read.get(10, TimeUnit.SECONDS)),
          "native Config read did not overlap metadata readers");
      CountDownLatch writerStarted = new CountDownLatch(1);
      Future<?> write = callers.submit(() -> {
        writerStarted.countDown();
        setConfig(stub, session, key, "after");
        return null;
      });
      require(writerStarted.await(10, TimeUnit.SECONDS), "native Config writer did not start");
      try {
        write.get(200, TimeUnit.MILLISECONDS);
        throw new AssertionError("native Config writer completed while metadata readers held state");
      } catch (TimeoutException expected) {
        // The readers also verify the value after the gate opens, while still inside Spark.
        // Together these test writer exclusion, not exact scheduling of Rust waiter futures.
      }

      gate.close();
      check(first.get(10, TimeUnit.SECONDS));
      check(second.get(10, TimeUnit.SECONDS));
      write.get(10, TimeUnit.SECONDS);
      require("after".equals(getConfig(stub, session, key)),
          "native Config write was not visible after metadata readers drained");
    }
  }

  private static final class ContextProbe {
    private final Map<String, String> zones = Map.of("first", "UTC", "second", "Asia/Tokyo");
    private final Map<String, String> types = Map.of(
        "first", "TIMESTAMP_LTZ", "second", "TIMESTAMP_NTZ");
    private final Map<String, Set<Thread>> threads = new ConcurrentHashMap<>();
    private final Map<String, AtomicInteger> observations = new ConcurrentHashMap<>();

    private void observe(SparkSession session) {
      String context = session.conf().get(CONTEXT_KEY, "");
      require(zones.containsKey(context), "analyzer used an unexpected native session context");
      require(SparkSession.getActiveSession().isDefined()
              && SparkSession.getActiveSession().get() == session,
          "native worker retained another session as its active session");
      require(zones.get(context).equals(SQLConf.get().sessionLocalTimeZone()),
          "native worker used another session's SQL time zone");
      require(types.get(context).equals(SQLConf.get().getConfString("spark.sql.timestampType")),
          "native worker used another session's timestamp type");
      org.apache.spark.sql.classic.SparkSession classic =
          (org.apache.spark.sql.classic.SparkSession) session;
      require(Thread.currentThread().getContextClassLoader() == classic.artifactManager().classloader(),
          "native worker used another session's artifact classloader");
      String jobGroup = session.sparkContext().getLocalProperty("spark.jobGroup.id");
      require(jobGroup != null && jobGroup.startsWith("spark-connect-engine-"),
          "native worker lost its current engine job group");
      threads.computeIfAbsent(context, ignored -> ConcurrentHashMap.newKeySet())
          .add(Thread.currentThread());
      observations.computeIfAbsent(context, ignored -> new AtomicInteger()).incrementAndGet();
    }
  }

  private static void twoSessionWorkerReuse(
      SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub,
      String firstSession, String secondSession) {
    setConfig(stub, firstSession, CONTEXT_KEY, "first", "spark.sql.session.timeZone", "UTC",
        "spark.sql.timestampType", "TIMESTAMP_LTZ");
    setConfig(stub, secondSession, CONTEXT_KEY, "second", "spark.sql.session.timeZone", "Asia/Tokyo",
        "spark.sql.timestampType", "TIMESTAMP_NTZ");
    ContextProbe probe = new ContextProbe();
    require(CONTEXT_PROBE.compareAndSet(null, probe), "another native context probe is active");
    try {
      // ABAB can assign each session to a different worker in a stable two-worker rotation.
      // Unequal same-session bursts exercise context switches without relying on that schedule.
      int[] bursts = {3, 5, 7, 11, 17, 31};
      int call = 0;
      for (int phase = 0; phase < bursts.length; phase++) {
        for (int turn = 0; turn < 2; turn++) {
          boolean first = (phase + turn) % 2 == 0;
          for (int index = 0; index < bursts[phase]; index++) {
            contextCall(stub, first ? firstSession : secondSession, !first, probe, call++);
          }
        }
      }
      Set<Thread> shared = new HashSet<>(probe.threads.getOrDefault("first", Set.of()));
      shared.retainAll(probe.threads.getOrDefault("second", Set.of()));
      require(!shared.isEmpty(), "no attached worker was observed serving both native sessions: "
          + probe.threads);
    } finally {
      CONTEXT_PROBE.compareAndSet(probe, null);
    }
  }

  private static void contextCall(SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub,
      String session, boolean secondSessionContext, ContextProbe probe, int call) {
    String context = secondSessionContext ? "second" : "first";
    AtomicInteger observed =
        probe.observations.computeIfAbsent(context, ignored -> new AtomicInteger());
    int before = observed.get();
    // A distinct alias keeps every RPC's analyzer observation independently testable.
    Relation relation = Relation.newBuilder().setSql(SQL.newBuilder()
        .setQuery("SELECT CAST('2026-01-02 03:04:05' AS TIMESTAMP) AS native_context_" + call)).build();
    checkTimestamp(analyze(stub, schemaRequest(session, relation)), secondSessionContext);
    require(observed.get() > before, "native analyzer context was not checked for RPC " + call);
  }

  private static void deepArrowDescriptors(
      SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub, String session) throws Exception {
    int depth = 12;
    String value = "CAST(7 AS BIGINT)";
    for (int level = 0; level < depth; level++) {
      value = level % 2 == 0 ? "named_struct('level_" + level + "', " + value + ")"
          : "array(" + value + ")";
    }
    Relation relation = Relation.newBuilder().setSql(SQL.newBuilder()
        .setQuery("SELECT " + value + " AS nested")).build();
    String operation = UUID.randomUUID().toString();
    ExecutePlanRequest request = ExecutePlanRequest.newBuilder().setSessionId(session)
        .setOperationId(operation).setPlan(Plan.newBuilder().setRoot(relation)).build();
    long rows = 0;
    int batches = 0;
    try {
      java.util.Iterator<ExecutePlanResponse> stream = stub.withDeadlineAfter(30, TimeUnit.SECONDS)
          .executePlan(request);
      while (stream.hasNext()) {
        ExecutePlanResponse response = stream.next();
        require(response.getSessionId().equals(session)
                && response.getOperationId().equals(operation),
            "deep Arrow execution changed response identity");
        if (!response.hasArrowBatch()) continue;
        require(response.getArrowBatch().getRowCount() == 1, "deep Arrow response row count changed");
        try (BufferAllocator allocator = ArrowUtils$.MODULE$.rootAllocator()
                 .newChildAllocator("native-deep-arrow", 0, Long.MAX_VALUE);
             ArrowStreamReader reader = new ArrowStreamReader(new ByteArrayInputStream(
                 response.getArrowBatch().getData().toByteArray()), allocator)) {
          while (reader.loadNextBatch()) {
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            require(root.getRowCount() == 1 && root.getFieldVectors().size() == 1,
                "deep Arrow batch lost its single row or column");
            Object actual = root.getVector(0).getObject(0);
            // A 12-level struct/list descriptor keeps more than 32 cumulative JNI references
            // live during recursive export. Walk every level to ensure none was flattened/lost.
            for (int level = depth - 1; level >= 0; level--) {
              if (level % 2 == 0) {
                require(actual instanceof Map && ((Map<?, ?>) actual).size() == 1
                        && ((Map<?, ?>) actual).containsKey("level_" + level),
                    "deep Arrow struct changed at level " + level);
                actual = ((Map<?, ?>) actual).get("level_" + level);
              } else {
                require(actual instanceof List && ((List<?>) actual).size() == 1,
                    "deep Arrow list changed at level " + level);
                actual = ((List<?>) actual).get(0);
              }
            }
            require(actual instanceof Number && ((Number) actual).longValue() == 7,
                "deep Arrow leaf value changed");
            rows += root.getRowCount();
            batches++;
          }
        }
      }
      require(rows == 1 && batches == 1, "deep Arrow execution did not return exactly one batch/row");
    } finally {
      var released = stub.withDeadlineAfter(30, TimeUnit.SECONDS).releaseExecute(
          ReleaseExecuteRequest.newBuilder().setSessionId(session).setOperationId(operation)
              .setReleaseAll(ReleaseExecuteRequest.ReleaseAll.newBuilder()).build());
      // A completed, non-reattachable stream has already removed its operation. Release is
      // idempotent: it preserves the session identity and leaves optional operation_id absent.
      require(released.getSessionId().equals(session) && !released.hasOperationId(),
          "completed deep Arrow operation release was not idempotent");
    }
  }

  private static void emptyArrowDescriptors(
      SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub, String session) throws Exception {
    Relation relation = Relation.newBuilder().setSql(SQL.newBuilder().setQuery(
        "SELECT '' AS text, unhex('') AS binary, CAST(array() AS ARRAY<STRING>) AS items, "
            + "array(CAST(array() AS ARRAY<STRING>)) AS nested, array('') AS strings, "
            + "CAST(map() AS MAP<STRING, STRING>) AS mapping, "
            + "CAST(NULL AS STRING) AS missing FROM range(3)")).build();
    String operation = UUID.randomUUID().toString();
    ExecutePlanRequest request = ExecutePlanRequest.newBuilder().setSessionId(session)
        .setOperationId(operation).setPlan(Plan.newBuilder().setRoot(relation)).build();
    long rows = 0;
    boolean complete = false;
    java.util.Iterator<ExecutePlanResponse> stream = stub.withDeadlineAfter(30, TimeUnit.SECONDS)
        .executePlan(request);
    while (stream.hasNext()) {
      ExecutePlanResponse response = stream.next();
      require(response.getSessionId().equals(session)
          && response.getOperationId().equals(operation), "empty Arrow response identity changed");
      require(!complete, "empty Arrow response followed completion");
      if (response.hasResultComplete()) complete = true;
      if (!response.hasArrowBatch()) continue;
      long batchRows = 0;
      try (BufferAllocator allocator = ArrowUtils$.MODULE$.rootAllocator()
               .newChildAllocator("native-empty-arrow", 0, Long.MAX_VALUE);
           ArrowStreamReader reader = new ArrowStreamReader(new ByteArrayInputStream(
               response.getArrowBatch().getData().toByteArray()), allocator)) {
        while (reader.loadNextBatch()) {
          VectorSchemaRoot root = reader.getVectorSchemaRoot();
          require(root.getFieldVectors().size() == 7, "empty Arrow fields changed");
          for (int row = 0; row < root.getRowCount(); row++) {
            require(root.getVector(0).getObject(row).toString().isEmpty(), "empty UTF-8 value changed");
            require(((byte[]) root.getVector(1).getObject(row)).length == 0, "empty binary changed");
            require(((List<?>) root.getVector(2).getObject(row)).isEmpty(), "empty list changed");
            List<?> nested = (List<?>) root.getVector(3).getObject(row);
            require(nested.size() == 1 && ((List<?>) nested.get(0)).isEmpty(),
                "nested empty list changed across the Arrow C import");
            List<?> strings = (List<?>) root.getVector(4).getObject(row);
            require(strings.size() == 1 && strings.get(0).toString().isEmpty(),
                "all-empty string child changed across the Arrow C import");
            require(((List<?>) root.getVector(5).getObject(row)).isEmpty(), "empty map changed");
            require(root.getVector(6).getObject(row) == null, "all-null string changed");
          }
          batchRows += root.getRowCount();
        }
      }
      require(batchRows == response.getArrowBatch().getRowCount(),
          "empty Arrow response row count changed");
      rows += batchRows;
    }
    require(rows == 3 && complete, "empty Arrow result lost rows or completion");
  }

  private static void lossySchemasAreRejected(
      SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub, String session,
      SparkSession spark) throws Exception {
    Relation alias = Relation.newBuilder().setProject(Project.newBuilder()
        .setInput(Relation.newBuilder().setRange(Range.newBuilder().setStart(0).setEnd(1).setStep(1)))
        .addExpressions(Expression.newBuilder().setAlias(Expression.Alias.newBuilder()
            .setExpr(Expression.newBuilder().setUnresolvedAttribute(
                Expression.UnresolvedAttribute.newBuilder().setUnparsedIdentifier("id")))
            .addName("annotated").setMetadata("{\"comment\":\"must not disappear\"}")))).build();
    for (Relation relation : List.of(alias, Relation.newBuilder().setSql(SQL.newBuilder()
        .setQuery("SELECT 'a' COLLATE UTF8_LCASE AS value")).build(),
        Relation.newBuilder().setSql(SQL.newBuilder()
            .setQuery("SELECT array('a' COLLATE UTF8_LCASE) AS value")).build())) {
      unsupportedPlan(stub, session, relation);
    }
    String view = "native_udt_" + UUID.randomUUID().toString().replace("-", "");
    StructType schema = new StructType(new StructField[] {
      DataTypes.createStructField("features", new VectorUDT(), true)
    });
    spark.createDataFrame(java.util.Collections.<Row>emptyList(), schema)
        .createOrReplaceGlobalTempView(view);
    try {
      unsupportedPlan(stub, session, Relation.newBuilder().setSql(SQL.newBuilder()
          .setQuery("SELECT * FROM global_temp." + view)).build());
    } finally {
      spark.catalog().dropGlobalTempView(view);
    }
    String cleanup = "spark.sql.classic.shuffleDependency.fileCleanup.enabled";
    Map<String, String> before = getAllConfig(stub, session);
    setConfig(stub, session, cleanup, "true");
    try {
      unsupportedPlan(stub, session, Relation.newBuilder().setRange(Range.newBuilder()
          .setStart(0).setEnd(1).setStep(1)).build());
    } finally {
      if (before.containsKey(cleanup)) {
        setConfig(stub, session, cleanup, before.get(cleanup));
      } else {
        stub.withDeadlineAfter(30, TimeUnit.SECONDS).config(ConfigRequest.newBuilder()
            .setSessionId(session).setOperation(ConfigRequest.Operation.newBuilder()
                .setUnset(ConfigRequest.Unset.newBuilder().addKeys(cleanup))).build());
      }
      require(before.equals(getAllConfig(stub, session)),
          "cleanup guard test changed explicit configuration state");
    }
  }

  private static void unsupportedPlan(SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub,
      String session, Relation relation) {
    Plan plan = Plan.newBuilder().setRoot(relation).build();
    for (AnalyzePlanRequest request : List.of(schemaRequest(session, relation),
        AnalyzePlanRequest.newBuilder().setSessionId(session)
            .setIsLocal(AnalyzePlanRequest.IsLocal.newBuilder().setPlan(plan)).build(),
        AnalyzePlanRequest.newBuilder().setSessionId(session)
            .setIsStreaming(AnalyzePlanRequest.IsStreaming.newBuilder().setPlan(plan)).build())) {
      expectUnsupported(() -> analyze(stub, request));
    }
    expectUnsupported(() -> {
      java.util.Iterator<ExecutePlanResponse> responses = stub.withDeadlineAfter(30, TimeUnit.SECONDS)
          .executePlan(ExecutePlanRequest.newBuilder().setSessionId(session)
              .setOperationId(UUID.randomUUID().toString()).setPlan(plan).build());
      require(!responses.hasNext(), "unsupported schema produced an execute response");
    });
  }

  private static void expectUnsupported(Runnable operation) {
    try {
      operation.run();
      throw new AssertionError("unsupported plan was accepted");
    } catch (StatusRuntimeException error) {
      require(error.getStatus().getCode() == Status.Code.UNIMPLEMENTED,
          "unsupported plan returned the wrong native status: " + error);
    }
  }

  private static AnalyzePlanRequest schemaRequest(String session, Relation relation) {
    return AnalyzePlanRequest.newBuilder().setSessionId(session)
        .setSchema(AnalyzePlanRequest.Schema.newBuilder().setPlan(Plan.newBuilder().setRoot(relation)))
        .build();
  }

  private static AnalyzePlanResponse analyze(
      SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub, AnalyzePlanRequest request) {
    return stub.withDeadlineAfter(30, TimeUnit.SECONDS).analyzePlan(request);
  }

  private static void setConfig(SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub,
      String session, String... alternating) {
    require(alternating.length % 2 == 0, "native Config fields must be pairs");
    ConfigRequest.Set.Builder values = ConfigRequest.Set.newBuilder();
    for (int index = 0; index < alternating.length; index += 2) {
      values.addPairs(KeyValue.newBuilder().setKey(alternating[index]).setValue(alternating[index + 1]));
    }
    require(stub.withDeadlineAfter(30, TimeUnit.SECONDS).config(ConfigRequest.newBuilder()
        .setSessionId(session).setOperation(ConfigRequest.Operation.newBuilder().setSet(values)).build())
        .getSessionId().equals(session), "native Config Set changed session identity");
  }

  private static String getConfig(SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub,
      String session, String key) {
    var response = stub.withDeadlineAfter(30, TimeUnit.SECONDS).config(ConfigRequest.newBuilder()
        .setSessionId(session).setOperation(ConfigRequest.Operation.newBuilder()
            .setGet(ConfigRequest.Get.newBuilder().addKeys(key))).build());
    require(response.getSessionId().equals(session) && response.getPairsCount() == 1
            && response.getPairs(0).getKey().equals(key) && response.getPairs(0).hasValue(),
        "native Config Get changed response identity or fields");
    return response.getPairs(0).getValue();
  }

  private static Map<String, String> getAllConfig(
      SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub, String session) {
    var response = stub.withDeadlineAfter(30, TimeUnit.SECONDS).config(ConfigRequest.newBuilder()
        .setSessionId(session).setOperation(ConfigRequest.Operation.newBuilder()
            .setGetAll(ConfigRequest.GetAll.newBuilder())).build());
    Map<String, String> values = new java.util.HashMap<>();
    for (KeyValue pair : response.getPairsList()) values.put(pair.getKey(), pair.getValue());
    return values;
  }

  private static void checkTimestamp(AnalyzePlanResponse response, boolean secondSessionContext) {
    check(response);
    var type = response.getSchema().getSchema().getStruct().getFields(0).getDataType();
    require(secondSessionContext ? type.hasTimestampNtz() : type.hasTimestamp(),
        "native analysis used another session's timestamp schema");
  }

  private static void check(AnalyzePlanResponse response) {
    require(response.hasSchema() && response.getSchema().getSchema().hasStruct(),
        "analysis did not return a struct schema");
    require(response.getSchema().getSchema().getStruct().getFieldsCount() == 1,
        "range schema field count changed");
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new AssertionError(message);
  }
}
