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

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.spark.connect.proto.Range;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.Attribute;
import org.apache.spark.sql.execution.SparkPlan;
import org.apache.spark.sql.util.ArrowUtils$;

/** Deliberately failed owners stay retained until this separate test JVM exits. */
public final class CleanupFailureTest {
  private CleanupFailureTest() {}

  public static void main(String[] arguments) throws Exception {
    require(arguments.length == 1, "pass the absolute native library path");
    SparkSession spark = SparkSession.builder().appName("Native cleanup failure retention")
        .config("spark.sql.artifact.isolation.enabled", "true")
        .config("spark.sql.artifact.isolation.alwaysApplyClassloader", "true")
        .getOrCreate();
    spark.sparkContext().setLogLevel("ERROR");
    try {
      taskCancellationFailure(spark);
      querySessionAndFactoryFailure(spark);
      failedBatchConstruction(spark);
      arrowFailure(spark);
      nativeFactoryFailure(spark, arguments[0]);
      require(!spark.sparkContext().isStopped(), "failed cleanup stopped the shared SparkContext");
      System.out.println("CLEANUP_FAILURE_OK: task/query/session/factory/Arrow failures retain "
          + "owners and original errors without replaying partial cleanup; native stop keeps "
          + "its retry handle and reports repeated Java cleanup failure after listener close");
    } finally {
      spark.stop();
    }
  }

  private static void taskCancellationFailure(SparkSession spark) throws Exception {
    SparkJvmEngine engine = new SparkJvmEngine(spark);
    SparkJvmEngine.Task task = engine.newTask("failing cancellation");
    RuntimeException failure = new IllegalStateException("injected task interruption failure");
    AtomicInteger attempts = new AtomicInteger();
    Thread running = new Thread() {
      @Override
      public void interrupt() {
        attempts.incrementAndGet();
        throw failure;
      }
    };
    field(task, "running").set(task, running);
    expectSame(failure, task::cancel);
    // A real interrupted call clears running in Task.run's finally block. Its later close
    // must still report the cancellation failure even after that worker has left.
    field(task, "running").set(task, null);
    expectSame(failure, task::close);
    expectSame(failure, task::close);
    require(registry(engine, "tasks").contains(task), "failed Task cleanup lost its owner");
    expectSame(failure, engine::close);
    expectSame(failure, engine::close);
    require(attempts.get() == 1, "failed task cancellation was replayed");
  }

  private static void querySessionAndFactoryFailure(SparkSession spark) throws Exception {
    FailedQuery fixture = new FailedQuery(spark);
    expectSame(fixture.failure, fixture.query::close);
    expectSame(fixture.failure, fixture.query::close);
    require(registry(fixture.session, "queries").contains(fixture.query),
        "failed Query cleanup lost its owner");
    expectSame(fixture.failure, fixture.session::close);
    expectSame(fixture.failure, fixture.session::close);
    require(registry(fixture.engine, "sessions").contains(fixture.session),
        "failed Session cleanup lost its owner");
    expectSame(fixture.failure, fixture.engine::close);
    expectSame(fixture.failure, fixture.engine::close);
    require(fixture.engine.isClosed(), "failed engine cleanup left admission open");
    require(fixture.plan.attempts.get() == 1, "partial physical-plan cleanup was replayed");
  }

  private static void arrowFailure(SparkSession spark) throws Exception {
    SparkJvmEngine engine = new SparkJvmEngine(spark);
    RuntimeException failure = new IllegalStateException("injected Arrow root cleanup failure");
    AtomicInteger attempts = new AtomicInteger();
    BufferAllocator allocator = ArrowUtils$.MODULE$.rootAllocator()
        .newChildAllocator("failing-batch-cleanup", 0, 1024);
    VectorSchemaRoot root = new VectorSchemaRoot(List.of(), List.of(), 0) {
      @Override
      public void close() {
        attempts.incrementAndGet();
        throw failure;
      }
    };
    @SuppressWarnings("unchecked")
    Set<NativeArrowBatch> owners = (Set<NativeArrowBatch>) field(engine, "batches").get(engine);
    long before = NativeArrowBatch.liveBatchCount();
    NativeArrowBatch batch = new NativeArrowBatch(root, allocator, owners);
    expectSame(failure, batch::close);
    expectSame(failure, batch::close);
    require(owners.contains(batch), "failed Arrow cleanup lost its strong owner");
    require(NativeArrowBatch.liveBatchCount() == before + 1,
        "failed Arrow cleanup falsely released its live count");
    expectSame(failure, engine::close);
    expectSame(failure, engine::close);
    require(attempts.get() == 1, "partial Arrow root cleanup was replayed");
  }

  private static void failedBatchConstruction(SparkSession spark) throws Exception {
    FailedQuery fixture = new FailedQuery(spark);
    RuntimeException queryError = new IllegalArgumentException("injected vector writer failure");
    RuntimeException rootFailure = new IllegalStateException("injected unexported root failure");
    RuntimeException allocatorFailure = new IllegalStateException("injected allocator close failure");
    AtomicInteger rootAttempts = new AtomicInteger();
    AtomicInteger allocatorAttempts = new AtomicInteger();
    VectorSchemaRoot root = new VectorSchemaRoot(List.of(), List.of(), 0) {
      @Override
      public void close() {
        rootAttempts.incrementAndGet();
        super.close();
        throw rootFailure;
      }
    };
    BufferAllocator delegate = ArrowUtils$.MODULE$.rootAllocator()
        .newChildAllocator("failing-unexported-batch", 0, 1024);
    BufferAllocator allocator = (BufferAllocator) Proxy.newProxyInstance(
        BufferAllocator.class.getClassLoader(), new Class<?>[] {BufferAllocator.class},
        (proxy, method, arguments) -> {
          if (method.getName().equals("close")) {
            allocatorAttempts.incrementAndGet();
            delegate.close();
            throw allocatorFailure;
          }
          try {
            return method.invoke(delegate, arguments);
          } catch (InvocationTargetException error) {
            throw error.getCause();
          }
        });
    fixture.query.cleanupFailedBatch(root, allocator, queryError);
    require(queryError.getSuppressed().length == 1
        && queryError.getSuppressed()[0] == rootFailure
        && rootFailure.getSuppressed().length == 1
        && rootFailure.getSuppressed()[0] == allocatorFailure,
        "failed batch construction lost the query error or cleanup failures");
    require(field(fixture.query, "failedBatchRoot").get(fixture.query) == root
        && field(fixture.query, "failedBatchAllocator").get(fixture.query) == allocator,
        "unexported failed batch resources lost their owner");
    expectSame(rootFailure, fixture.query::close);
    expectSame(rootFailure, fixture.query::close);
    expectSame(rootFailure, fixture.session::close);
    expectSame(rootFailure, fixture.engine::close);
    expectSame(rootFailure, fixture.engine::close);
    require(registry(fixture.session, "queries").contains(fixture.query)
        && registry(fixture.engine, "sessions").contains(fixture.session),
        "unexported batch cleanup falsely drained its query or session");
    require(rootAttempts.get() == 1 && allocatorAttempts.get() == 1,
        "unexported batch resource cleanup was replayed");
  }

  private static void nativeFactoryFailure(SparkSession spark, String library) throws Exception {
    FailedQuery fixture = new FailedQuery(spark);
    NativeEngineServer server = new NativeEngineServer(library, "127.0.0.1:0", fixture.engine);
    String address = server.address();
    int port = Integer.parseInt(address.substring(address.lastIndexOf(':') + 1));
    long handle = field(server, "handle").getLong(server);
    for (int attempt = 0; attempt < 2; attempt++) {
      try {
        server.close();
        throw new AssertionError("native stop accepted failed Java factory cleanup");
      } catch (IllegalStateException error) {
        require(error.getMessage().contains("JNI cleanup is incomplete")
            && error.getMessage().contains(fixture.failure.getMessage()),
            "native stop lost the cleanup failure: " + error);
      }
      require(field(server, "handle").getLong(server) == handle,
          "failed native stop discarded its retry handle");
      require(registry(fixture.engine, "sessions").contains(fixture.session),
          "failed native factory cleanup discarded its Java owners");
      require(fixture.plan.attempts.get() == 1, "native retry replayed partial Java cleanup");
    }
    try (ServerSocket rebound = new ServerSocket()) {
      rebound.setReuseAddress(true);
      rebound.bind(new InetSocketAddress("127.0.0.1", port));
      require(rebound.isBound(), "failed factory cleanup left its listener open");
    }
    // server and its failed owners intentionally remain retained until this JVM exits.
  }

  private static final class FailedQuery {
    private final SparkJvmEngine engine;
    private final SparkJvmEngine.Session session;
    private final SparkJvmEngine.Query query;
    private final RuntimeException failure =
        new IllegalStateException("injected physical-plan cleanup failure");
    private final FailingCleanupPlan plan = new FailingCleanupPlan(failure);

    private FailedQuery(SparkSession spark) throws Exception {
      engine = new SparkJvmEngine(spark);
      try (SparkJvmEngine.Task task = engine.newTask("cleanup fixture open")) {
        session = engine.openSession("cleanup-fixture", UUID.randomUUID().toString(), task);
      }
      SparkJvmEngine.Task task = engine.newTask("cleanup fixture prepare");
      query = session.prepare(Relation.newBuilder().setRange(Range.newBuilder()
          .setStart(0).setEnd(1).setStep(1)).build().toByteArray(), task);
      field(query, "plan").set(query, plan);
    }
  }

  private static final class FailingCleanupPlan extends SparkPlan {
    private final RuntimeException failure;
    private final AtomicInteger attempts = new AtomicInteger();

    private FailingCleanupPlan(RuntimeException failure) { this.failure = failure; }

    @Override
    public void cleanupResources() {
      attempts.incrementAndGet();
      throw failure;
    }

    @Override
    public scala.collection.immutable.Seq<Attribute> output() {
      return scala.collection.immutable.List$.MODULE$.empty();
    }

    @Override
    public scala.collection.immutable.Seq<SparkPlan> children() {
      return scala.collection.immutable.List$.MODULE$.empty();
    }

    @Override
    public SparkPlan withNewChildrenInternal(scala.collection.immutable.IndexedSeq<SparkPlan> children) {
      if (!children.isEmpty()) throw new IllegalArgumentException("leaf plan cannot have children");
      return this;
    }

    @Override
    public RDD<InternalRow> doExecute() { throw new AssertionError("cleanup fixture executed"); }

    @Override
    public String simpleStringWithNodeId() { return "FailingCleanupPlan"; }

    @Override
    public String verboseString(int maxFields) { return "FailingCleanupPlan"; }

    @Override
    public int productArity() { return 0; }

    @Override
    public Object productElement(int index) { throw new IndexOutOfBoundsException(index); }

    @Override
    public boolean canEqual(Object other) { return other instanceof FailingCleanupPlan; }
  }

  private static Field field(Object owner, String name) throws ReflectiveOperationException {
    Field field = owner.getClass().getDeclaredField(name);
    field.setAccessible(true);
    return field;
  }

  private static Set<?> registry(Object owner, String name) throws ReflectiveOperationException {
    return (Set<?>) field(owner, name).get(owner);
  }

  private static void expectSame(RuntimeException expected, Runnable close) {
    try {
      close.run();
      throw new AssertionError("failed cleanup was accepted");
    } catch (RuntimeException actual) {
      require(actual == expected, "cleanup replaced the original failure: " + actual);
    }
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new AssertionError(message);
  }
}
