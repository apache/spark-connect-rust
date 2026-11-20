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

import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.cleanup;
import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.field;
import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.registry;
import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.require;

import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.spark.TaskContext;
import org.apache.spark.connect.proto.Range;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.connect.proto.SQL;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.Attribute;
import org.apache.spark.sql.execution.SparkPlan;
import org.apache.spark.sql.types.DataTypes;

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
      cancellationFailureWithLiveWorker(spark);
      querySessionAndFactoryFailure(spark);
      unpublishedSessionFailure(spark);
      nativeFactoryFailure(spark, arguments[0]);
      require(!spark.sparkContext().isStopped(), "failed cleanup stopped the shared SparkContext");
      System.out.println("CLEANUP_FAILURE_OK: cancellation delivery can recover; "
          + "partial query/session/factory cleanup retains owners and original errors "
          + "without replaying cleanup; native stop keeps its retry handle and reports repeated "
          + "Java cleanup failure after listener close");
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
    require(registry(engine, "tasks").contains(task), "failed Task cleanup lost its owner");
    expectSame(failure, engine::close);
    // A real interrupted call clears running in Task.run's finally block. Once that
    // work has left, cancellation and delegated engine close can finish on a retry.
    field(task, "running").set(task, null);
    task.close();
    task.close();
    engine.close();
    engine.close();
    require(!registry(engine, "tasks").contains(task), "recovered Task owner was retained");
    require(attempts.get() == 2, "failed cancellation delivery was not retried");
  }

  private static void cancellationFailureWithLiveWorker(SparkSession spark) throws Exception {
    SparkJvmEngine engine = new SparkJvmEngine(spark);
    SparkJvmEngine.Session session;
    try (SparkJvmEngine.Task opening = engine.newTask("live cleanup fixture open")) {
      session = engine.openSession("live-cleanup-fixture", UUID.randomUUID().toString(), opening);
    }
    SparkJvmEngine.Task task = engine.newTask("live cleanup fixture execute");
    SparkSession sessionSpark = (SparkSession) field(session, "spark").get(session);
    Path directory = Files.createTempDirectory("native-failed-cancel-");
    String function = "cleanup_gate_" + UUID.randomUUID().toString().replace("-", "");
    sessionSpark.udf().register(function, new WaitingPartition(directory), DataTypes.LongType);
    SparkJvmEngine.Query query = session.prepare(Relation.newBuilder().setSql(SQL.newBuilder()
        .setQuery("SELECT " + function + "(id) AS id FROM range(0, 2, 1, 2)"))
        .build().toByteArray(), task);
    RuntimeException failure = new IllegalStateException("injected live worker interruption failure");
    RuntimeException planFailure = new IllegalStateException("injected plan cleanup failure");
    FailingCleanupPlan cleanup = new FailingCleanupPlan(planFailure);
    ExecutorService pool = Executors.newFixedThreadPool(2);
    Thread worker = null;
    SparkPlan originalPlan = null;
    Throwable primaryFailure = null;
    try {
      Future<NativeIpcBatch> pending = pool.submit(query::nextIpcBatch);
      Object results = field(query, "results").get(query);
      long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15);
      for (;;) {
        synchronized (results) {
          worker = (Thread) field(query, "worker").get(query);
          List<?> slots = (List<?>) field(query, "partitionBatches").get(query);
          if (slots != null && slots.get(1) != null
              && Files.exists(directory.resolve("entered"))) break;
        }
        require(!pending.isDone() && System.nanoTime() < deadline,
            "gated query did not buffer its later partition");
        Thread.sleep(5);
      }
      require(worker != null && worker.isAlive() && !pending.isDone(),
          "cleanup fixture did not retain a live worker and waiting pull");
      originalPlan = (SparkPlan) field(query, "plan").get(query);
      field(query, "plan").set(query, cleanup);
      synchronized (task) {
        field(task, "running").set(task, new Thread() {
          @Override
          public void interrupt() { throw failure; }
        });
      }
      // Cursor release must wake admission and cancel jobs even when Task.interrupt fails.
      Future<?> closing = pool.submit(() -> expectSame(failure, query::close));
      closing.get(5, TimeUnit.SECONDS);
      require(cleanup.attempts.get() <= 1
              && field(query, "planCleanupAttempted").getBoolean(query)
                  == (cleanup.attempts.get() == 1),
          "failed close replayed or lost plan cleanup");
      require(cleanup.attempts.get() == 0 || !worker.isAlive(),
          "failed cancellation cleaned the plan while its worker was alive");
      synchronized (results) {
        require(field(query, "partitionBatches").get(query) == null,
            "failed close retained buffered IPC");
      }
      require(registry(engine, "tasks").contains(task)
              && registry(session, "queries").contains(query),
          "failed live-worker close lost its Task or Query owner");
      try {
        pending.get(5, TimeUnit.SECONDS);
        throw new AssertionError("failed close did not reject the waiting pull");
      } catch (ExecutionException error) {
        require(error.getCause() instanceof SparkJvmEngine.EngineException
                && ((SparkJvmEngine.EngineException) error.getCause()).code() == 1,
            "failed close returned the wrong pull error: " + error.getCause());
      }
      worker.join(15000);
      require(!worker.isAlive() && field(query, "workerFinished").getBoolean(query),
          "cursor release did not independently cancel and drain its worker");
      require(!Files.exists(directory.resolve("allow")),
          "fixture opened the partition gate before cancellation drained");
      deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15);
      while (!Files.exists(directory.resolve("exited"))) {
        require(System.nanoTime() < deadline, "cursor release left its gated partition running");
        Thread.sleep(5);
      }
      require(field(query, "partitionBatches").get(query) == null,
          "late callback repopulated the closed cursor");
      expectSame(planFailure, query::close);
      expectSame(planFailure, query::close);
      require(cleanup.attempts.get() == 1,
          "retry did not clean the terminated worker's plan exactly once");
      require(!registry(engine, "tasks").contains(task)
              && registry(session, "queries").contains(query),
          "retry did not separate recovered cancellation from partial plan cleanup");
    } catch (Throwable error) {
      primaryFailure = error;
      throw error;
    } finally {
      Throwable cleanupFailure = cleanup(primaryFailure,
          () -> Files.write(directory.resolve("allow"), new byte[0]));
      Thread ownedWorker = worker;
      cleanupFailure = cleanup(cleanupFailure, () -> {
        if (ownedWorker != null) {
          ownedWorker.join(15000);
          require(!ownedWorker.isAlive(), "cleanup fixture leaked its worker");
        }
      });
      cleanupFailure = cleanup(cleanupFailure, () -> {
        pool.shutdownNow();
        require(pool.awaitTermination(15, TimeUnit.SECONDS), "cleanup fixture pool did not stop");
      });
      SparkPlan ownedPlan = originalPlan;
      cleanupFailure = cleanup(cleanupFailure, () -> {
        if (ownedPlan != null && (ownedWorker == null || !ownedWorker.isAlive())) {
          ownedPlan.cleanupResources();
        }
      });
      cleanupFailure = cleanup(cleanupFailure, () -> {
        try (var paths = Files.list(directory)) {
          for (Path path : paths.toList()) Files.delete(path);
        }
        Files.delete(directory);
      });
      if (primaryFailure == null && cleanupFailure != null) {
        if (cleanupFailure instanceof Error) throw (Error) cleanupFailure;
        throw (Exception) cleanupFailure;
      }
    }
    // This isolated engine and its deliberately failed owners remain until the test JVM exits.
  }

  private static final class WaitingPartition implements UDF1<Long, Long> {
    private static final long serialVersionUID = 1L;
    private final String directory;

    private WaitingPartition(Path directory) { this.directory = directory.toString(); }

    @Override
    public Long call(Long value) throws Exception {
      if (TaskContext.get().partitionId() == 0) {
        Path folder = Path.of(directory);
        Files.write(folder.resolve("entered"), new byte[0]);
        try {
          long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
          while (!Files.exists(folder.resolve("allow"))) {
            if (TaskContext.get().isInterrupted() || Thread.currentThread().isInterrupted()) {
              throw new InterruptedException("cleanup fixture partition cancelled");
            }
            require(System.nanoTime() < deadline, "cleanup fixture gate timed out");
            Thread.sleep(5);
          }
        } finally {
          boolean restoreInterrupt = Thread.interrupted();
          try {
            Files.write(folder.resolve("exited"), new byte[0]);
          } finally {
            if (restoreInterrupt) Thread.currentThread().interrupt();
          }
        }
      }
      return value;
    }
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

  private static void unpublishedSessionFailure(SparkSession spark) throws Exception {
    SparkJvmEngine healthyEngine = new SparkJvmEngine(spark);
    SparkJvmEngine.Session healthy;
    try (SparkJvmEngine.Task task = healthyEngine.newTask("unpublished clean session")) {
      healthy = healthyEngine.openSession("unpublished-fixture", UUID.randomUUID().toString(), task);
    }
    // Reproduce the constructor's ownership state after newSession but before holder assignment.
    field(healthy, "holder").set(healthy, null);
    require(registry(healthyEngine, "sessions").remove(healthy), "clean fixture was not registered");
    RuntimeException construction = new IllegalArgumentException("injected SessionHolder failure");
    healthy.abandonUnpublished(construction);
    require(construction.getSuppressed().length == 0,
        "successful unpublished cleanup added a failure");
    require(!registry(healthyEngine, "sessions").contains(healthy),
        "successfully cleaned unpublished session remained registered");
    require(field(healthy, "closeComplete").getBoolean(healthy),
        "unpublished session cleanup did not finish");
    healthy.close();
    healthyEngine.close();

    FailedQuery failed = new FailedQuery(spark);
    field(failed.session, "holder").set(failed.session, null);
    require(registry(failed.engine, "sessions").remove(failed.session),
        "failed fixture was not registered");
    RuntimeException failedConstruction =
        new IllegalArgumentException("injected SessionHolder and cleanup failure");
    failed.session.abandonUnpublished(failedConstruction);
    require(failedConstruction.getSuppressed().length == 1
            && failedConstruction.getSuppressed()[0] == failed.failure,
        "unpublished session lost its construction or cleanup error");
    require(registry(failed.engine, "sessions").contains(failed.session),
        "failed unpublished cleanup lost its retained session");
    expectSame(failed.failure, failed.engine::close);
    expectSame(failed.failure, failed.engine::close);
    require(failed.plan.attempts.get() == 1, "unpublished partial cleanup was replayed");
    require(!spark.sparkContext().isStopped(), "unpublished cleanup stopped the shared SparkContext");
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

  private static void expectSame(RuntimeException expected, Runnable close) {
    try {
      close.run();
      throw new AssertionError("failed cleanup was accepted");
    } catch (RuntimeException actual) {
      require(actual == expected, "cleanup replaced the original failure: " + actual);
    }
  }
}
