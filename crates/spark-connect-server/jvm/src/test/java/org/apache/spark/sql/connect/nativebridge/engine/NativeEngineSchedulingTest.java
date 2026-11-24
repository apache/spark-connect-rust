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

import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.CheckedAction;
import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.cleanup;
import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.field;
import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.require;
import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.sql;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.spark.SparkConf;
import org.apache.spark.TaskContext;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.scheduler.SparkListener;
import org.apache.spark.scheduler.SparkListenerEvent;
import org.apache.spark.scheduler.SparkListenerJobEnd;
import org.apache.spark.scheduler.SparkListenerJobStart;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionEnd;
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionStart;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.util.ArrowUtils$;
import org.apache.spark.util.TaskCompletionListener;

import scala.Option;

/** Real two-slot Spark jobs: ordered windowed results, waiting pulls and result limits. */
public final class NativeEngineSchedulingTest {
  private static final int TIMEOUT_SECONDS = 15;
  private static final String UPSTREAM_ERROR = "native-scheduling-upstream-failure";

  private NativeEngineSchedulingTest() {}

  public static void main(String[] arguments) throws Exception {
    require(arguments.length == 0 || (arguments.length == 1
        && arguments[0].equals("--result-size-only")), "unknown scheduling test arguments");
    SparkSession spark = SparkSession.builder().appName("Native engine parallel scheduling")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.artifact.isolation.enabled", "true")
        .config("spark.sql.artifact.isolation.alwaysApplyClassloader", "true")
        .config("spark.task.maxFailures", "1")
        .getOrCreate();
    spark.sparkContext().setLogLevel("ERROR");
    try (SparkJvmEngine engine = new SparkJvmEngine(spark)) {
      if (arguments.length == 1) {
        perPartitionResultSize(engine, spark);
        System.out.println("NATIVE_ENGINE_PER_PARTITION_RESULT_SIZE_OK: individually fitting "
            + "partitions keep independent Spark result-size limits in one execution");
      } else {
        laterPartitionCompletesFirst(engine, spark);
        cancelledBeforePull(engine, spark);
        cancelBothRunningPartitions(engine, spark);
        closeWaitingPull(engine, spark);
        closeSessionAndEngine(engine, spark);
        cancelWhilePlanning(engine, spark);
        preInterruptedWaitingPull(engine, spark);
        upstreamFailureWakesPull(engine, spark);
        emptyPartitions(engine);
        fetchWindows(spark);
        defaultWindowFollowsScheduler(spark);
        pausedMultiBatchPartition(spark);
        skewedPartitionSizes(spark);
        cancelBeyondWindow(spark);
        require(!spark.sparkContext().isStopped(), "query cleanup stopped the shared context");
        activeHostStop(engine, spark);
        System.out.println("NATIVE_ENGINE_SCHEDULING_OK: real task overlap, one execution, "
            + "slow-start/byte-target/sequential windows, per-query scheduler defaults, "
            + "paused multi-batch/skewed results, "
            + "cancellation beyond the window, "
            + "ordered retained IPC, pre-cancel, running/planning cancellation, close wakeup, "
            + "session/engine close, waiting caller interrupt, upstream failure, empty partitions, "
            + "session recovery and INTERNAL host-stop classification");
      }
    } finally {
      spark.stop();
    }
  }

  private static void laterPartitionCompletesFirst(SparkJvmEngine engine, SparkSession spark)
      throws Exception {
    try (GatedQuery fixture = new GatedQuery(engine, spark, 4, false, true, -1)) {
      Pull first = fixture.pull();
      fixture.entered(0);
      fixture.entered(1);
      fixture.exited(1);
      require(!first.future.isDone() && !fixture.exists("exited-0"),
          "later partition bypassed the blocked first partition");
      fixture.probe.executionOpen();
      fixture.allow(0);
      NativeIpcBatch retained = first.get();
      byte[] original = retained.data().clone();
      long next = checkValues(retained, 0);
      NativeIpcBatch interrupted;
      Thread.currentThread().interrupt();
      try {
        interrupted = fixture.query.nextIpcBatch();
        require(Thread.currentThread().isInterrupted(), "fetch lost the incoming interrupt flag");
      } finally {
        Thread.interrupted();
      }
      next = checkValues(interrupted, next);
      NativeIpcBatch batch;
      while ((batch = fixture.query.nextIpcBatch()) != null) next = checkValues(batch, next);
      require(next == 8 && fixture.query.nextIpcBatch() == null, "ordered cursor lost rows or EOF");
      require(Arrays.equals(retained.data(), original), "draining later results changed prior IPC");
      fixture.probe.oneExecution(2);
      fixture.query.close();
      assertStopped(fixture.query);
    }
  }

  private static void cancelledBeforePull(SparkJvmEngine engine, SparkSession spark)
      throws Exception {
    try (GatedQuery fixture = new GatedQuery(engine, spark, 2, true, true, -1)) {
      fixture.task.cancel();
      expectCode(1, fixture.query::nextIpcBatch);
      require(!fixture.exists("entered-0") && !fixture.exists("entered-1"),
          "pre-cancelled cursor started executor work");
      require(fixture.probe.jobIds.isEmpty(), "pre-cancelled cursor submitted a job");
      fixture.query.close();
      assertStopped(fixture.query);
      assertRecovery(engine, fixture.session);
    }
  }

  private static void cancelBothRunningPartitions(SparkJvmEngine engine, SparkSession spark)
      throws Exception {
    try (GatedQuery fixture = new GatedQuery(engine, spark, 2, false, false, -1)) {
      Pull pending = fixture.pull();
      fixture.entered(0);
      fixture.entered(1);
      pending.waiting();
      fixture.workers.submit(fixture.task::cancel).get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
      expectCode(1, pending::get);
      fixture.exited(0);
      fixture.exited(1);
      fixture.query.close();
      assertStopped(fixture.query);
      fixture.probe.drained();
      assertRecovery(engine, fixture.session);
    }
  }

  private static void closeWaitingPull(SparkJvmEngine engine, SparkSession spark)
      throws Exception {
    try (GatedQuery fixture = new GatedQuery(engine, spark, 2, true, false, -1)) {
      Pull first = fixture.pull();
      fixture.entered(0);
      fixture.entered(1);
      NativeIpcBatch retained = first.get();
      require(checkValues(retained, 0) == 2, "first partition changed");
      byte[] original = retained.data().clone();
      fixture.exited(0);
      Pull pending = fixture.pull();
      pending.waiting();
      fixture.workers.submit(fixture.query::close).get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
      assertStopped(fixture.query);
      expectCode(1, pending::get);
      fixture.exited(1);
      fixture.probe.drained();
      require(Arrays.equals(retained.data(), original), "close changed already returned IPC");
      assertRecovery(engine, fixture.session);
    }
  }

  private static void upstreamFailureWakesPull(SparkJvmEngine engine, SparkSession spark)
      throws Exception {
    try (GatedQuery fixture = new GatedQuery(engine, spark, 2, false, false, 1)) {
      Pull pending = fixture.pull();
      fixture.entered(0);
      fixture.entered(1);
      pending.waiting();
      fixture.allow(1);
      SparkJvmEngine.EngineException failure = expectCode(13, pending::get);
      require(failure.getMessage() != null && failure.getMessage().contains(UPSTREAM_ERROR),
          "upstream failure lost its cause: " + failure);
      require(expectCode(13, fixture.query::nextIpcBatch) == failure,
          "failed cursor replaced its terminal error");
      fixture.exited(0);
      fixture.exited(1);
      fixture.query.close();
      assertStopped(fixture.query);
      fixture.probe.drained();
      assertRecovery(engine, fixture.session);
    }
  }

  private static void closeSessionAndEngine(SparkJvmEngine engine, SparkSession spark)
      throws Exception {
    for (boolean entireEngine : new boolean[] {false, true}) {
      try (SparkJvmEngine owner = new SparkJvmEngine(spark);
           GatedQuery fixture = new GatedQuery(owner, spark, 2, false, false, -1)) {
        Pull pending = fixture.pull();
        fixture.entered(0);
        fixture.entered(1);
        pending.waiting();
        fixture.workers.submit(() -> {
          if (entireEngine) owner.close();
          else fixture.session.close();
        }).get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        expectCode(1, pending::get);
        assertStopped(fixture.query);
        fixture.exited(0);
        fixture.exited(1);
        fixture.probe.oneExecution(2);
        require(!spark.sparkContext().isStopped(), "owner close stopped the shared context");
        try (SparkJvmEngine.Session peer = open(entireEngine ? engine : owner)) {
          assertRecovery(entireEngine ? engine : owner, peer);
        }
      }
    }
  }

  private static void cancelWhilePlanning(SparkJvmEngine engine, SparkSession spark)
      throws Exception {
    for (boolean closeCursor : new boolean[] {false, true}) {
      try (GatedQuery fixture = new GatedQuery(engine, spark, 2, false, false, -1, true)) {
        Pull pending = fixture.pull();
        fixture.entered(0);
        fixture.entered(1);
        pending.waiting();
        // AQE is still materializing the gated shuffle map stage; no result slots exist yet.
        synchronized (field(fixture.query, "results").get(fixture.query)) {
          require(field(fixture.query, "partitionBatches").get(fixture.query) == null,
              "planning fixture reached the final result job before cancellation");
        }
        fixture.workers.submit(() -> {
          if (closeCursor) fixture.query.close();
          else fixture.task.cancel();
        }).get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        expectCode(1, pending::get);
        fixture.query.close();
        assertStopped(fixture.query);
        fixture.exited(0);
        fixture.exited(1);
        fixture.probe.executionEnded();
        assertRecovery(engine, fixture.session);
      }
    }
  }

  private static void preInterruptedWaitingPull(SparkJvmEngine engine, SparkSession spark)
      throws Exception {
    try (GatedQuery fixture = new GatedQuery(engine, spark, 2, false, false, -1)) {
      Pull pending = new Pull(fixture.workers, fixture.query, true);
      fixture.entered(0);
      fixture.entered(1);
      pending.waiting();
      fixture.allow(0);
      fixture.allow(1);
      long next = checkValues(pending.get(), 0);
      require(pending.interruptedAfter.get(), "waiting fetch lost the incoming interrupt flag");
      NativeIpcBatch batch;
      while ((batch = fixture.query.nextIpcBatch()) != null) next = checkValues(batch, next);
      require(next == 4, "pre-interrupted waiting fetch changed ordered output");
      fixture.probe.oneExecution(2);
    }
  }

  /** A real scheduler shutdown keeps its failure classified as INTERNAL after a parked drain. */
  private static void activeHostStop(SparkJvmEngine engine, SparkSession spark) throws Exception {
    try (GatedQuery fixture = new GatedQuery(engine, spark, 2, false, false, -1)) {
      Pull pending = fixture.pull();
      fixture.entered(0);
      fixture.entered(1);
      pending.waiting();
      Thread worker = (Thread) field(fixture.query, "worker").get(fixture.query);
      await(() -> worker.getState() == Thread.State.TIMED_WAITING
              && Arrays.stream(worker.getStackTrace()).anyMatch(frame ->
                  frame.getClassName().equals(SparkJvmEngine.Query.class.getName())
                      && frame.getMethodName().equals("runJobs")),
          "active query did not park while draining its submitted jobs");
      require(fixture.query.submittedPartitions() == 2,
          "active host-stop probe did not submit both gated partitions");
      fixture.workers.submit(() -> spark.sparkContext().stop())
          .get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
      // A stopped local scheduler can fail its futures before executor code exits.
      // Release the test-owned file gates so those task listeners can finish too.
      fixture.allow(0);
      fixture.allow(1);
      SparkJvmEngine.EngineException failure = expectCode(13, pending::get);
      Throwable schedulerFailure;
      synchronized (field(fixture.query, "results").get(fixture.query)) {
        schedulerFailure = (Throwable) field(fixture.query, "jobFailure").get(fixture.query);
      }
      require(schedulerFailure != null && failure.getCause() == schedulerFailure
              && failure.getMessage() != null
              && failure.getMessage().equals(schedulerFailure.getMessage()),
          "active host stop lost the scheduler's primary failure: " + failure);
      fixture.query.close();
      assertStopped(fixture.query);
      fixture.exited(0);
      fixture.exited(1);
      fixture.probe.drained();
    }
  }

  private static void emptyPartitions(SparkJvmEngine engine) throws Exception {
    try (SparkJvmEngine.Session session = open(engine)) {
      for (long expected : new long[] {0, 2}) {
        try (SparkJvmEngine.Task task = engine.newTask("empty parallel result");
             SparkJvmEngine.Query query = session.prepare(
                 sql("SELECT id FROM range(0, " + expected + ", 1, 4)").toByteArray(), task)) {
          long next = 0;
          NativeIpcBatch batch;
          while ((batch = query.nextIpcBatch()) != null) next = checkValues(batch, next);
          require(next == expected, "empty partitions changed output");
          require(query.nextIpcBatch() == null, "empty partition EOF changed");
        }
      }
      assertRecovery(engine, session);
    }
  }

  /** Eight one-batch partitions: partitions submitted when each one is delivered. */
  private static void expectWindow(SparkSession spark, int maxConcurrent, long maxBytes,
      int[] submittedAtPartition) throws Exception {
    try (SparkJvmEngine engine = new SparkJvmEngine(spark, maxConcurrent, maxBytes)) {
      expectWindow(engine, spark, maxConcurrent, maxBytes, submittedAtPartition);
    }
  }

  private static void expectWindow(SparkJvmEngine engine, SparkSession spark,
      int maxConcurrent, long maxBytes, int[] submittedAtPartition) throws Exception {
    String description = "fetch window " + UUID.randomUUID();
    try (SparkJvmEngine.Session session = open(engine);
         Jobs probe = new Jobs(spark, description);
         SparkJvmEngine.Task task = engine.newTask(description);
         SparkJvmEngine.Query query = session.prepare(
             sql("SELECT id FROM range(0, 80, 1, 8)").toByteArray(), task)) {
      long next = 0;
      for (int partition = 0; partition < 8; partition++) {
        NativeIpcBatch batch = query.nextIpcBatch();
        require(batch != null && batch.rowCount() == 10, "partition was not one complete batch");
        // Delivering this batch has not advanced the cursor past its partition yet, so the
        // window still counts it. Wait for the worker to fill the remaining allowed slots.
        if (partition < submittedAtPartition.length) {
          int expected = submittedAtPartition[partition];
          // A fast first result can reveal its oversized partition before the second
          // admission. Either initial count is valid; later byte-limited counts are exact.
          int minimum = partition == 0 && batch.data().length > maxBytes ? 1 : expected;
          await(() -> query.submittedPartitions() >= minimum, "window " + maxConcurrent + "/"
              + maxBytes + " did not reach " + minimum + " partitions at " + partition);
          awaitProducerIdle(query);
          int submitted = query.submittedPartitions();
          require(submitted >= minimum && submitted <= expected, "window " + maxConcurrent + "/"
              + maxBytes + " had " + submitted + " partitions at " + partition);
        }
        next = checkValues(batch, next);
      }
      require(query.nextIpcBatch() == null && next == 80, "windowed fetch lost rows or EOF");
      probe.oneExecution(8);
    }
  }

  private static void awaitProducerIdle(SparkJvmEngine.Query query) throws Exception {
    Thread worker = (Thread) field(query, "worker").get(query);
    Object results = field(query, "results").get(query);
    await(() -> {
      synchronized (results) {
        try {
          return field(query, "workerFinished").getBoolean(query)
              || worker.getState() == Thread.State.TIMED_WAITING
                  && Arrays.stream(worker.getStackTrace()).anyMatch(frame ->
                      frame.getClassName().equals(SparkJvmEngine.Query.class.getName())
                          && frame.getMethodName().equals("runJobs"));
        } catch (ReflectiveOperationException error) {
          throw new AssertionError(error);
        }
      }
    }, "producer did not finish or park at its admission window");
  }

  private static void fetchWindows(SparkSession spark) throws Exception {
    // Slow start admits up to two partitions before sizes are known, then doubles per delivery.
    expectWindow(spark, 4, Long.MAX_VALUE, new int[] {2, 5, 6, 7, 8});
    // A partition larger than the byte budget falls back to one job at a time.
    expectWindow(spark, 4, 1, new int[] {2, 2, 3, 4});
    // A window of one is the sequential fetch.
    expectWindow(spark, 1, Long.MAX_VALUE, new int[] {1, 2, 3});
  }

  /** Nonpositive limits resolve the current scheduler default for each query, not at startup. */
  private static void defaultWindowFollowsScheduler(SparkSession spark) throws Exception {
    String key = "spark.default.parallelism";
    // getConf() clones; use the host configuration read by the pinned local scheduler.
    SparkConf host = spark.sparkContext().conf();
    Option<String> previous = host.getOption(key);
    try {
      for (int configured : new int[] {0, -1}) {
        host.set(key, "1");
        require(spark.sparkContext().defaultParallelism() == 1,
            "scheduler fixture did not start with parallelism one");
        try (SparkJvmEngine engine = new SparkJvmEngine(spark, configured, Long.MAX_VALUE)) {
          host.set(key, "4");
          require(spark.sparkContext().defaultParallelism() == 4,
              "scheduler fixture did not observe the increased host default");
          expectWindow(engine, spark, configured, Long.MAX_VALUE, new int[] {2, 5, 6, 7, 8});
          host.set(key, "1");
          require(spark.sparkContext().defaultParallelism() == 1,
              "scheduler fixture did not observe the decreased host default");
          expectWindow(engine, spark, configured, Long.MAX_VALUE, new int[] {1, 2, 3});
        }
      }
    } finally {
      if (previous.isDefined()) host.set(key, previous.get());
      else host.remove(key);
    }
  }

  /** A partially delivered partition still occupies its window slot while the client pauses. */
  private static void pausedMultiBatchPartition(SparkSession spark) throws Exception {
    String description = "paused multi-batch window " + UUID.randomUUID();
    try (SparkJvmEngine engine = new SparkJvmEngine(spark, 2, Long.MAX_VALUE);
         SparkJvmEngine.Session session = open(engine);
         Jobs probe = new Jobs(spark, description);
         SparkJvmEngine.Task task = engine.newTask(description)) {
      SparkSession sessionSpark = (SparkSession) field(session, "spark").get(session);
      sessionSpark.conf().set("spark.sql.execution.arrow.maxRecordsPerBatch", "10");
      try (SparkJvmEngine.Query query = session.prepare(
          sql("SELECT id FROM range(0, 800, 1, 8)").toByteArray(), task)) {
        long next = checkValues(query.nextIpcBatch(), 0);
        await(() -> query.submittedPartitions() >= 2, "initial window did not fill");
        await(() -> probe.jobIds.size() >= 2, "initial partition jobs did not start");
        probe.drained();
        Thread worker = (Thread) field(query, "worker").get(query);
        await(() -> worker.getState() == Thread.State.TIMED_WAITING
                && Arrays.stream(worker.getStackTrace()).anyMatch(frame ->
                    frame.getClassName().equals(SparkJvmEngine.Query.class.getName())
                        && frame.getMethodName().equals("runJobs")),
            "producer did not pause at its admission window");
        require(query.submittedPartitions() == 2,
            "paused consumer admitted more than two partitions");
        probe.executionOpen();
        while (next < 100) {
          next = checkValues(query.nextIpcBatch(), next);
          require(query.submittedPartitions() == 2,
              "partially delivered partition stopped counting toward the window");
        }
        NativeIpcBatch batch;
        while ((batch = query.nextIpcBatch()) != null) next = checkValues(batch, next);
        require(next == 800, "paused multi-batch window changed output");
        probe.oneExecution(8);
      }
    }
  }

  /** An oversized later partition can exceed the byte target without breaking ordered drain. */
  private static void skewedPartitionSizes(SparkSession spark) throws Exception {
    String description = "skewed partition window " + UUID.randomUUID();
    try (SparkJvmEngine engine = new SparkJvmEngine(spark, 4, 8192);
         SparkJvmEngine.Session session = open(engine);
         Jobs probe = new Jobs(spark, description);
         SparkJvmEngine.Task task = engine.newTask(description);
         SparkJvmEngine.Query query = session.prepare(sql(
             "SELECT id, repeat('x', CASE WHEN id BETWEEN 20 AND 29 THEN 8192 ELSE 1 END) "
                 + "AS payload FROM range(0, 80, 1, 8)").toByteArray(), task)) {
      long next = 0;
      int shrinkAt = -1;
      NativeIpcBatch batch;
      while ((batch = query.nextIpcBatch()) != null) {
        int partition = (int) (next / 10);
        if (partition == 2) {
          awaitProducerIdle(query);
          shrinkAt = query.submittedPartitions() - 1;
        }
        if (shrinkAt >= 0 && partition >= shrinkAt) {
          // Drain admissions made before the heavy result's size was known. After the
          // last of those, the observed 80 KiB partition limits fetch to one at a time.
          awaitProducerIdle(query);
          require(query.submittedPartitions() - partition == 1,
              "heavy partition did not reduce later admission to one partition");
        }
        require(query.submittedPartitions() - partition <= 4,
            "skewed result exceeded the partition admission bound");
        try (BufferAllocator allocator = ArrowUtils$.MODULE$.rootAllocator()
                .newChildAllocator("skewed-window-test", 0, Long.MAX_VALUE);
             ArrowStreamReader reader = new ArrowStreamReader(
                 new ByteArrayInputStream(batch.data()), allocator)) {
          require(reader.loadNextBatch(), "skewed IPC omitted its record batch");
          var root = reader.getVectorSchemaRoot();
          require(root.getRowCount() == batch.rowCount(), "skewed IPC row count changed");
          for (int row = 0; row < root.getRowCount(); row++) {
            require(((Number) root.getVector(0).getObject(row)).longValue() == next,
                "skewed result lost, duplicated or reordered a row");
            int length = next >= 20 && next < 30 ? 8192 : 1;
            require(root.getVector(1).getObject(row).toString().equals("x".repeat(length)),
                "skewed result changed its payload");
            next++;
          }
          require(!reader.loadNextBatch(), "skewed IPC contains multiple record batches");
        }
      }
      require(next == 80, "skewed window changed the row count");
      probe.oneExecution(8);
    }
  }

  /** Cancelling mid-window drains the submitted jobs and never starts the remaining ones. */
  private static void cancelBeyondWindow(SparkSession spark) throws Exception {
    String description = "cancel beyond window " + UUID.randomUUID();
    ExecutorService workers = Executors.newSingleThreadExecutor();
    try (SparkJvmEngine engine = new SparkJvmEngine(spark, 2, Long.MAX_VALUE);
         SparkJvmEngine.Session session = open(engine);
         Jobs probe = new Jobs(spark, description)) {
      SparkJvmEngine.Task task = engine.newTask(description);
      SparkJvmEngine.Query query = session.prepare(
          sql("SELECT id FROM range(0, 1000000000000, 1, 4)").toByteArray(), task);
      Future<NativeIpcBatch> pending = workers.submit(query::nextIpcBatch);
      await(() -> probe.jobIds.size() == 2, "window of two did not start two jobs");
      require(query.submittedPartitions() == 2, "window of two submitted another partition");
      query.close();
      assertStopped(query);
      try {
        pending.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
        throw new AssertionError("closed query produced a partition");
      } catch (ExecutionException error) {
        require(error.getCause() instanceof SparkJvmEngine.EngineException
            && ((SparkJvmEngine.EngineException) error.getCause()).code() == 1,
            "closed windowed fetch lost CANCELLED status: " + error.getCause());
      }
      probe.drained();
      require(probe.jobIds.size() == 2 && query.submittedPartitions() == 2,
          "closed query submitted a partition beyond its window");
      assertRecovery(engine, session);
    } finally {
      workers.shutdownNow();
      require(workers.awaitTermination(TIMEOUT_SECONDS, TimeUnit.SECONDS),
          "cancelled fetch did not stop");
    }
  }

  private static void perPartitionResultSize(SparkJvmEngine engine, SparkSession spark)
      throws Exception {
    require(spark.sparkContext().getConf().getSizeAsBytes("spark.driver.maxResultSize")
        == 1024 * 1024, "per-partition result-size mode requires spark.driver.maxResultSize=1m");
    try (SparkJvmEngine.Session session = open(engine)) {
      // Each partition is about 0.8 MiB of IPC, so neither alone exceeds the limit.
      try (SparkJvmEngine.Task task = engine.newTask("single result-size control");
           SparkJvmEngine.Query query = session.prepare(
               sql("SELECT id FROM range(0, 100000, 1, 1)").toByteArray(), task)) {
        long next = 0;
        NativeIpcBatch batch;
        while ((batch = query.nextIpcBatch()) != null) next = checkValues(batch, next);
        require(next == 100000, "individually fitting partition control failed");
      }
      String description = "per-partition result-size " + UUID.randomUUID();
      try (Jobs probe = new Jobs(spark, description);
           SparkJvmEngine.Task task = engine.newTask(description);
           SparkJvmEngine.Query query = session.prepare(
               sql("SELECT id FROM range(0, 200000, 1, 2)").toByteArray(), task)) {
        long next = 0;
        NativeIpcBatch batch;
        while ((batch = query.nextIpcBatch()) != null) next = checkValues(batch, next);
        // Each partition is its own job, so the limit applies to each and not to their sum.
        require(next == 200000, "individually fitting partitions shared one result-size limit");
        probe.oneExecution(2);
      }
      assertRecovery(engine, session);
    }
  }

  private static SparkJvmEngine.Session open(SparkJvmEngine engine) {
    try (SparkJvmEngine.Task task = engine.newTask("scheduling session open")) {
      return engine.openSession("scheduling-test", UUID.randomUUID().toString(), task);
    }
  }

  private static long checkValues(NativeIpcBatch batch, long next) throws Exception {
    require(batch != null, "missing IPC batch");
    try (BufferAllocator allocator = ArrowUtils$.MODULE$.rootAllocator()
            .newChildAllocator("scheduling-test", 0, Long.MAX_VALUE);
         ArrowStreamReader reader = new ArrowStreamReader(
             new ByteArrayInputStream(batch.data()), allocator)) {
      require(reader.loadNextBatch(), "IPC omitted its record batch");
      var root = reader.getVectorSchemaRoot();
      require(root.getRowCount() == batch.rowCount(), "IPC row count changed");
      for (int row = 0; row < root.getRowCount(); row++) {
        require(((Number) root.getVector(0).getObject(row)).longValue() == next++,
            "parallel cursor lost, duplicated or reordered a row");
      }
      require(!reader.loadNextBatch(), "IPC contains multiple record batches");
    }
    return next;
  }

  private static void assertRecovery(SparkJvmEngine engine, SparkJvmEngine.Session session)
      throws Exception {
    try (SparkJvmEngine.Task task = engine.newTask("scheduling session recovery");
         SparkJvmEngine.Query query = session.prepare(sql("SELECT 0 AS id").toByteArray(), task)) {
      require(checkValues(query.nextIpcBatch(), 0) == 1 && query.nextIpcBatch() == null,
          "session did not recover after parallel query cleanup");
    }
  }

  private static void assertStopped(SparkJvmEngine.Query query) throws Exception {
    Thread worker = (Thread) field(query, "worker").get(query);
    require(worker == null || !worker.isAlive(), "closed cursor retained a live worker");
    require(field(query, "partitionBatches").get(query) == null,
        "closed cursor retained pending partition results");
  }

  /** File gates work across Spark's driver and task class loaders without shared static state. */
  private static final class GatedIdentity implements UDF1<Long, Long> {
    private static final long serialVersionUID = 1L;
    private final String directory;
    private final int failingPartition;

    private GatedIdentity(Path directory, int failingPartition) {
      this.directory = directory.toString();
      this.failingPartition = failingPartition;
    }

    @Override
    public Long call(Long value) throws Exception {
      Path folder = Path.of(directory);
      TaskContext context = TaskContext.get();
      int partition = context.partitionId();
      try {
        Files.createFile(folder.resolve("entered-" + partition));
        context.addTaskCompletionListener((TaskCompletionListener) ignored -> {
          try {
            Files.createFile(folder.resolve("exited-" + partition));
          } catch (IOException error) {
            throw new UncheckedIOException(error);
          }
        });
      } catch (FileAlreadyExistsException alreadyEntered) {
        // Each fixture disables task retries and has one producing task per partition.
      }
      long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2L * TIMEOUT_SECONDS);
      while (!Files.exists(folder.resolve("allow-" + partition))) {
        if (context.isInterrupted() || Thread.currentThread().isInterrupted()) {
          throw new InterruptedException("gated Spark partition cancelled");
        }
        if (System.nanoTime() >= deadline) {
          throw new IllegalStateException("gated Spark partition timed out: " + partition);
        }
        Thread.sleep(5);
      }
      if (partition == failingPartition) throw new IllegalStateException(UPSTREAM_ERROR);
      return value;
    }
  }

  private static final class Pull {
    private final AtomicReference<Thread> caller = new AtomicReference<>();
    private final AtomicBoolean interruptedAfter = new AtomicBoolean();
    private final Future<NativeIpcBatch> future;

    private Pull(ExecutorService workers, SparkJvmEngine.Query query) {
      this(workers, query, false);
    }

    private Pull(ExecutorService workers, SparkJvmEngine.Query query, boolean preInterrupted) {
      future = workers.submit(() -> {
        caller.set(Thread.currentThread());
        if (preInterrupted) Thread.currentThread().interrupt();
        try {
          return query.nextIpcBatch();
        } finally {
          interruptedAfter.set(Thread.interrupted());
        }
      });
    }

    private NativeIpcBatch get() throws Exception {
      try {
        return future.get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
      } catch (ExecutionException error) {
        Throwable cause = error.getCause();
        if (cause instanceof Exception) throw (Exception) cause;
        throw (Error) cause;
      }
    }

    private void waiting() throws Exception {
      await(() -> {
        Thread thread = caller.get();
        if (thread == null || future.isDone()) return false;
        if (thread.getState() != Thread.State.WAITING) return false;
        return Arrays.stream(thread.getStackTrace()).anyMatch(frame ->
            frame.getClassName().equals(SparkJvmEngine.Query.class.getName())
                && frame.getMethodName().equals("nextIpcBatch"));
      }, "pull did not wait for its pending partition");
    }
  }

  private static final class GatedQuery implements AutoCloseable {
    private final Path directory = Files.createTempDirectory("native-scheduling-");
    private final ExecutorService workers = Executors.newFixedThreadPool(2);
    private final SparkJvmEngine.Session session;
    private final Jobs probe;
    private final SparkJvmEngine.Task task;
    private SparkJvmEngine.Query query;

    private GatedQuery(SparkJvmEngine engine, SparkSession spark, int rowsPerPartition,
        boolean allowFirst, boolean allowSecond, int failingPartition) throws Exception {
      this(engine, spark, rowsPerPartition, allowFirst, allowSecond, failingPartition, false);
    }

    private GatedQuery(SparkJvmEngine engine, SparkSession spark, int rowsPerPartition,
        boolean allowFirst, boolean allowSecond, int failingPartition, boolean shuffle)
        throws Exception {
      session = open(engine);
      String description = "parallel scheduling " + UUID.randomUUID();
      probe = new Jobs(spark, description);
      task = engine.newTask(description);
      try {
        SparkSession sessionSpark = (SparkSession) field(session, "spark").get(session);
        sessionSpark.conf().set("spark.sql.execution.arrow.maxRecordsPerBatch", "2");
        if (shuffle) sessionSpark.conf().set("spark.sql.adaptive.enabled", "true");
        String function = "native_gate_" + UUID.randomUUID().toString().replace("-", "");
        sessionSpark.udf().register(function, new GatedIdentity(directory, failingPartition),
            DataTypes.LongType);
        if (allowFirst) allow(0);
        if (allowSecond) allow(1);
        String rows = "SELECT " + function + "(id) AS id FROM range(0, "
            + (2 * rowsPerPartition) + ", 1, 2)";
        query = session.prepare(sql(shuffle ? "SELECT id FROM (" + rows + " DISTRIBUTE BY id)"
            : rows).toByteArray(), task);
      } catch (Throwable failure) {
        try { close(); } catch (Throwable cleanup) { failure.addSuppressed(cleanup); }
        if (failure instanceof Exception) throw (Exception) failure;
        throw (Error) failure;
      }
    }

    private boolean exists(String name) { return Files.exists(directory.resolve(name)); }
    private void allow(int partition) throws IOException {
      Files.write(directory.resolve("allow-" + partition), new byte[0]);
    }
    private void entered(int partition) throws Exception {
      await(() -> exists("entered-" + partition), "partition did not enter: " + partition);
    }
    private void exited(int partition) throws Exception {
      await(() -> exists("exited-" + partition), "partition did not exit: " + partition);
    }
    private Pull pull() { return new Pull(workers, query); }

    @Override
    public void close() throws Exception {
      Throwable failure = null;
      failure = cleanup(failure, () -> { allow(0); allow(1); });
      failure = cleanup(failure, () -> {
        if (query == null) task.close();
        else { query.close(); assertStopped(query); }
      });
      failure = cleanup(failure, () -> {
        workers.shutdownNow();
        require(workers.awaitTermination(TIMEOUT_SECONDS, TimeUnit.SECONDS),
            "scheduling fetch workers did not stop");
      });
      failure = cleanup(failure, () -> {
        for (int partition = 0; partition < 2; partition++) {
          if (exists("entered-" + partition)) exited(partition);
        }
      });
      failure = cleanup(failure, session::close);
      failure = cleanup(failure, probe::close);
      failure = cleanup(failure, () -> {
        try (var files = Files.list(directory)) {
          for (Path path : files.toList()) Files.delete(path);
        }
        Files.delete(directory);
      });
      if (failure instanceof Exception) throw (Exception) failure;
      if (failure != null) throw (Error) failure;
    }
  }

  private static final class Jobs extends SparkListener implements AutoCloseable {
    private final SparkSession spark;
    private final String description;
    private final Set<Integer> jobIds = ConcurrentHashMap.newKeySet();
    private final Set<Integer> endedJobs = ConcurrentHashMap.newKeySet();
    private final Set<Long> executionIds = ConcurrentHashMap.newKeySet();
    private final Map<Long, AtomicInteger> executionStarts = new ConcurrentHashMap<>();
    private final Set<Long> executionEnds = ConcurrentHashMap.newKeySet();

    private Jobs(SparkSession spark, String description) {
      this.spark = spark;
      this.description = description;
      spark.sparkContext().addSparkListener(this);
    }

    @Override
    public void onJobStart(SparkListenerJobStart event) {
      if (!description.equals(event.properties().getProperty("spark.job.description"))) return;
      jobIds.add(event.jobId());
      String execution = event.properties().getProperty("spark.sql.execution.id");
      if (execution != null) executionIds.add(Long.parseLong(execution));
    }

    @Override
    public void onJobEnd(SparkListenerJobEnd event) { endedJobs.add(event.jobId()); }

    @Override
    public void onOtherEvent(SparkListenerEvent event) {
      if (event instanceof SparkListenerSQLExecutionStart) {
        long id = ((SparkListenerSQLExecutionStart) event).executionId();
        executionStarts.computeIfAbsent(id, ignored -> new AtomicInteger()).incrementAndGet();
      } else if (event instanceof SparkListenerSQLExecutionEnd) {
        executionEnds.add(((SparkListenerSQLExecutionEnd) event).executionId());
      }
    }

    private void drained() throws Exception {
      await(() -> !jobIds.isEmpty() && endedJobs.containsAll(jobIds),
          "owned Spark job did not drain");
    }

    private void executionOpen() throws Exception {
      await(() -> executionIds.size() == 1
              && executionStarts.containsKey(executionIds.iterator().next()),
          "owned SQL execution did not start");
      spark.sparkContext().listenerBus().waitUntilEmpty(TIMEOUT_SECONDS * 1000L);
      require(executionIds.stream().noneMatch(executionEnds::contains),
          "SQL execution ended while the first partition was still running");
    }

    private void oneExecution(int jobs) throws Exception {
      executionEnded();
      require(jobIds.size() == jobs, "windowed result used " + jobIds.size() + " Spark jobs");
    }

    private void executionEnded() throws Exception {
      drained();
      await(() -> !executionIds.isEmpty() && executionEnds.containsAll(executionIds),
          "owned SQL execution did not end");
      require(executionIds.size() == 1, "parallel result used more than one SQL execution");
      long execution = executionIds.iterator().next();
      require(executionStarts.containsKey(execution) && executionStarts.get(execution).get() == 1,
          "SQL execution start was missing or repeated");
    }

    @Override
    public void close() { spark.sparkContext().removeSparkListener(this); }
  }

  private static SparkJvmEngine.EngineException expectCode(int code, CheckedAction action)
      throws Exception {
    try {
      action.run();
      throw new AssertionError("expected engine error " + code);
    } catch (SparkJvmEngine.EngineException error) {
      require(error.code() == code, "unexpected engine status: " + error);
      return error;
    }
  }

  private static void await(BooleanSupplier ready, String message) throws Exception {
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(TIMEOUT_SECONDS);
    while (!ready.getAsBoolean()) {
      require(System.nanoTime() < deadline, message);
      Thread.sleep(5);
    }
  }
}
