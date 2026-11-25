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
import java.lang.reflect.Field;
import java.util.AbstractSet;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.spark.connect.proto.ExecutePlanRequest;
import org.apache.spark.connect.proto.ExecutePlanResponse;
import org.apache.spark.connect.proto.Plan;
import org.apache.spark.connect.proto.Range;
import org.apache.spark.connect.proto.ReattachOptions;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.connect.proto.ReleaseExecuteRequest;
import org.apache.spark.connect.proto.ReleaseSessionRequest;
import org.apache.spark.connect.proto.SparkConnectServiceGrpc;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.util.ArrowUtils$;
import org.sparkproject.connect.grpc.ManagedChannel;
import org.sparkproject.connect.grpc.ManagedChannelBuilder;

/** More than 64 peers reclaim capacity while another real JNI owner is blocked in Java close. */
public final class NativeCleanupIsolationTest {
  private static final int PEERS = 70;
  private static final int RPC_SECONDS = 10;

  private NativeCleanupIsolationTest() {}

  public static void main(String[] arguments) throws Exception {
    require(arguments.length == 1, "pass the absolute native library path");
    SparkSession spark = SparkSession.builder().appName("Native cleanup isolation")
        .config("spark.sql.artifact.isolation.enabled", "true")
        .config("spark.sql.artifact.isolation.alwaysApplyClassloader", "true")
        .config("spark.sql.execution.arrow.compression.codec", "none")
        .getOrCreate();
    spark.sparkContext().setLogLevel("ERROR");
    try {
      isolatedClose(spark, arguments[0], true);
      isolatedClose(spark, arguments[0], false);
      require(!spark.sparkContext().isStopped() && spark.range(1).count() == 1,
          "native shutdown stopped the host SparkContext");
      System.out.println("NATIVE_CLEANUP_ISOLATION_OK: attached blocked query/session closes; "
          + PEERS + " peers each; retained shutdown and successful retry");
    } finally {
      spark.stop();
    }
  }

  private static void isolatedClose(SparkSession spark, String library, boolean query)
      throws Exception {
    SparkJvmEngine engine = new SparkJvmEngine(spark);
    NativeEngineServer server;
    try {
      server = new NativeEngineServer(library, "127.0.0.1:0", engine);
    } catch (RuntimeException | Error error) {
      try { engine.close(); } catch (Throwable cleanup) { error.addSuppressed(cleanup); }
      throw error;
    }
    try (server) {
      String address = server.address();
      ManagedChannel channel = ManagedChannelBuilder.forTarget(address).usePlaintext().build();
      try (AutoCloseable transport = () -> closeChannel(channel)) {
        var stub = SparkConnectServiceGrpc.newBlockingStub(channel);
        String sessionId = UUID.randomUUID().toString();
        execute(stub, sessionId);
        Set<Object> sessions = readSet(engine, "sessions");
        require(sessions.size() == 1, "warmup did not initialize one Java session");
        Object session = sessions.iterator().next();
        Set<Object> queries = readSet(session, "queries");
        Set<Object> tasks = readSet(engine, "tasks");
        await(() -> queries.isEmpty() && tasks.isEmpty(), "warmup cleanup did not finish");
        String kind = query ? "query" : "session";
        try (CloseGate gate = new CloseGate(query ? session : engine,
            query ? "queries" : "sessions", query ? null : session,
            query ? SparkJvmEngine.Query.class : SparkJvmEngine.Session.class)) {
          if (query) execute(stub, sessionId);
          else releaseSession(stub, sessionId);
          require(gate.entered.await(10, TimeUnit.SECONDS), kind + " close did not enter its gate");
          gate.requireBlocked();
          long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(80);
          for (int peer = 0; peer < PEERS; peer++) {
            require(System.nanoTime() < deadline, kind + " peer work exceeded its deadline");
            gate.requireBlocked();
            String peerSession = query ? sessionId : UUID.randomUUID().toString();
            execute(stub, peerSession);
            if (!query) releaseSession(stub, peerSession);
          }
          await(() -> gate.size() == 1 && gate.contains(gate.selected),
              kind + " peers did not leave only the blocked Java owner");
          gate.requireBlocked();
          System.out.println("NATIVE_CLEANUP_PEERS_OK: " + kind + "; peers=" + PEERS
              + "; attachedThread=" + gate.thread.getName());

          closeChannel(channel);
          RuntimeException incomplete = null;
          try { server.close(); } catch (RuntimeException error) { incomplete = error; }
          require(incomplete != null && incomplete.getMessage() != null
                  && (incomplete.getMessage().contains("incomplete")
                      || incomplete.getMessage().contains("retained")),
              "native close did not report its blocked " + kind + " owner: " + incomplete);
          gate.requireBlocked();
          require(server.address().equals(address) && !server.isDrained() && !engine.isClosed(),
              "incomplete native close released its handle or engine factory");
        }
      }
      // The gate has opened and restored its registry before native close retries.
      server.close();
      require(server.isDrained() && !server.isRunning() && engine.isClosed(),
          "native close retry did not drain and close its engine");
    }
  }

  private static void execute(
      SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub, String session) throws Exception {
    String operation = UUID.randomUUID().toString();
    var request = ExecutePlanRequest.newBuilder().setSessionId(session).setOperationId(operation)
        .addRequestOptions(ExecutePlanRequest.RequestOption.newBuilder()
            .setReattachOptions(ReattachOptions.newBuilder().setReattachable(true)))
        .setPlan(Plan.newBuilder().setRoot(Relation.newBuilder().setRange(Range.newBuilder()
            .setStart(0).setEnd(1).setStep(1).setNumPartitions(1)))).build();
    Iterator<ExecutePlanResponse> responses = stub.withDeadlineAfter(RPC_SECONDS, TimeUnit.SECONDS)
        .executePlan(request);
    boolean schema = false;
    boolean complete = false;
    long rows = 0;
    while (responses.hasNext()) {
      ExecutePlanResponse response = responses.next();
      require(!complete && session.equals(response.getSessionId())
              && operation.equals(response.getOperationId()), "response identity or order changed");
      if (response.hasSchema()) {
        require(!schema && response.getSchema().hasStruct()
                && response.getSchema().getStruct().getFieldsCount() == 1
                && response.getSchema().getStruct().getFields(0).getName().equals("id")
                && response.getSchema().getStruct().getFields(0).getDataType().hasLong(),
            "range schema changed");
        schema = true;
      }
      if (response.hasArrowBatch()) {
        require(schema, "Arrow data preceded its schema");
        long batchRows = 0;
        try (BufferAllocator allocator = ArrowUtils$.MODULE$.rootAllocator()
                 .newChildAllocator("native-cleanup-isolation", 0, 1024 * 1024);
             ArrowStreamReader reader = new ArrowStreamReader(new ByteArrayInputStream(
                 response.getArrowBatch().getData().toByteArray()), allocator)) {
          while (reader.loadNextBatch()) {
            var root = reader.getVectorSchemaRoot();
            require(root.getFieldVectors().size() == 1, "range Arrow columns changed");
            for (int row = 0; row < root.getRowCount(); row++) {
              Object value = root.getVector(0).getObject(row);
              require(value instanceof Number && ((Number) value).longValue() == 0,
                  "range Arrow value changed");
              batchRows++;
            }
          }
        }
        require(batchRows == response.getArrowBatch().getRowCount(), "Arrow row count changed");
        rows += batchRows;
      }
      if (response.hasResultComplete()) complete = true;
    }
    require(schema && complete && rows == 1, "range result did not complete correctly");
    var released = stub.withDeadlineAfter(RPC_SECONDS, TimeUnit.SECONDS).releaseExecute(
        ReleaseExecuteRequest.newBuilder().setSessionId(session).setOperationId(operation)
            .setReleaseAll(ReleaseExecuteRequest.ReleaseAll.newBuilder()).build());
    require(session.equals(released.getSessionId())
            && (!released.hasOperationId() || operation.equals(released.getOperationId())),
        "operation release identity changed");
  }

  private static void releaseSession(
      SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub, String session) {
    require(session.equals(stub.withDeadlineAfter(RPC_SECONDS, TimeUnit.SECONDS).releaseSession(
        ReleaseSessionRequest.newBuilder().setSessionId(session).build()).getSessionId()),
        "session release identity changed");
  }

  @SuppressWarnings("unchecked")
  private static Set<Object> readSet(Object owner, String name) throws Exception {
    Field field = owner.getClass().getDeclaredField(name);
    field.setAccessible(true);
    return (Set<Object>) field.get(owner);
  }

  private static void closeChannel(ManagedChannel channel) throws Exception {
    channel.shutdownNow();
    require(channel.awaitTermination(10, TimeUnit.SECONDS), "gRPC channel did not stop");
  }

  private static void await(BooleanSupplier condition, String message) throws Exception {
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
    while (!condition.getAsBoolean()) {
      require(System.nanoTime() < deadline, message);
      Thread.sleep(5);
    }
  }

  // Block inside a real Java close before its JNI checked-free call can return.
  private static final class CloseGate extends AbstractSet<Object> implements AutoCloseable {
    private final Object owner;
    private final Field field;
    private final Set<Object> original;
    private final Object expected;
    private final Class<?> closeClass;
    private final AtomicBoolean claimed = new AtomicBoolean();
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private final CountDownLatch entered = new CountDownLatch(1);
    private final CountDownLatch release = new CountDownLatch(1);
    private final CountDownLatch exited = new CountDownLatch(1);
    private volatile Object selected;
    private volatile Thread thread;
    private volatile boolean closeFrame;

    @SuppressWarnings("unchecked")
    CloseGate(Object owner, String name, Object expected, Class<?> closeClass) throws Exception {
      this.owner = owner;
      this.expected = expected;
      this.closeClass = closeClass;
      field = owner.getClass().getDeclaredField(name);
      field.setAccessible(true);
      original = (Set<Object>) field.get(owner);
      field.set(owner, this);
    }

    @Override public boolean remove(Object value) {
      boolean blocked = (expected == null || expected == value)
          && claimed.compareAndSet(false, true);
      try {
        if (blocked) {
          selected = value;
          thread = Thread.currentThread();
          closeFrame = Arrays.stream(thread.getStackTrace()).anyMatch(frame ->
              frame.getClassName().equals(closeClass.getName())
                  && frame.getMethodName().equals("close"));
          entered.countDown();
          try {
            if (!release.await(180, TimeUnit.SECONDS)) {
              failure.compareAndSet(null, new IllegalStateException("close gate timed out"));
            }
          } catch (InterruptedException error) {
            failure.compareAndSet(null, error);
            thread.interrupt();
          }
        }
        return original.remove(value);
      } finally {
        if (blocked) exited.countDown();
      }
    }

    void requireBlocked() {
      require(failure.get() == null, "close gate failed: " + failure.get());
      require(entered.getCount() == 0 && release.getCount() == 1 && exited.getCount() == 1,
          "Java close did not remain blocked");
      require(closeFrame && thread.isDaemon()
              && thread.getName().matches("spark-connect-engine-worker-[0-9]+"),
          "blocked Java close did not use a persistent JNI daemon attachment");
    }

    @Override public void close() throws Exception {
      release.countDown();
      try {
        require(!claimed.get() || exited.await(10, TimeUnit.SECONDS), "Java close did not resume");
        require(failure.get() == null, "close gate failed: " + failure.get());
      } finally {
        field.set(owner, original);
      }
    }

    @Override public Iterator<Object> iterator() { return original.iterator(); }
    @Override public int size() { return original.size(); }
    @Override public boolean contains(Object value) { return original.contains(value); }
    @Override public boolean add(Object value) { return original.add(value); }
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new AssertionError(message);
  }
}
