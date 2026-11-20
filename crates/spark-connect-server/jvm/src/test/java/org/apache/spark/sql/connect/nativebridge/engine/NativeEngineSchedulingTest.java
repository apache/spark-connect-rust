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
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Field;
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
import org.apache.spark.TaskContext;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.connect.proto.SQL;
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

/** Real two-slot Spark jobs: ordered parallel results, waiting pulls and shared result limits. */
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
        combinedResultSize(engine, spark);
        System.out.println("NATIVE_ENGINE_COMBINED_RESULT_SIZE_OK: individually fitting "
            + "partitions share one Spark result-size budget; failure drains and session recovers");
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
        require(!spark.sparkContext().isStopped(), "query cleanup stopped the shared context");
        System.out.println("NATIVE_ENGINE_SCHEDULING_OK: real task overlap, one job/execution, "
            + "ordered retained IPC, pre-cancel, running/planning cancellation, close wakeup, "
            + "session/engine close, waiting caller interrupt, upstream failure, empty partitions "
            + "and session recovery");
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
      fixture.probe.singleJobAndExecution();
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
        fixture.probe.singleJobAndExecution();
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
      fixture.probe.singleJobAndExecution();
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

  private static void combinedResultSize(SparkJvmEngine engine, SparkSession spark)
      throws Exception {
    require(spark.sparkContext().getConf().getSizeAsBytes("spark.driver.maxResultSize")
        == 1024 * 1024, "combined result-size mode requires spark.driver.maxResultSize=1m");
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
      String description = "combined result-size " + UUID.randomUUID();
      try (Jobs probe = new Jobs(spark, description);
           SparkJvmEngine.Task task = engine.newTask(description);
           SparkJvmEngine.Query query = session.prepare(
               sql("SELECT id FROM range(0, 200000, 1, 2)").toByteArray(), task)) {
        long next = 0;
        SparkJvmEngine.EngineException failure;
        try {
          NativeIpcBatch batch;
          while ((batch = query.nextIpcBatch()) != null) next = checkValues(batch, next);
          throw new AssertionError("combined partitions bypassed the 1 MiB job result-size limit");
        } catch (SparkJvmEngine.EngineException error) {
          failure = error;
        }
        require(failure.code() == 13 && failure.getMessage() != null
                && (failure.getMessage().contains("maxResultSize")
                    || failure.getMessage().contains("serialized results")),
            "combined result-size fixture failed for a different reason: " + failure);
        require(next <= 100000, "result-size abort exposed the rejected partition");
        require(expectCode(13, query::nextIpcBatch) == failure,
            "result-size failure was not sticky");
        probe.singleJobAndExecution();
      }
      assertRecovery(engine, session);
    }
  }

  private static SparkJvmEngine.Session open(SparkJvmEngine engine) {
    try (SparkJvmEngine.Task task = engine.newTask("scheduling session open")) {
      return engine.openSession("scheduling-test", UUID.randomUUID().toString(), task);
    }
  }

  private static Relation sql(String text) {
    return Relation.newBuilder().setSql(SQL.newBuilder().setQuery(text)).build();
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

  private static Field field(Object owner, String name) throws ReflectiveOperationException {
    Field field = owner.getClass().getDeclaredField(name);
    field.setAccessible(true);
    return field;
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
        if (thread.getState() != Thread.State.WAITING
            && thread.getState() != Thread.State.TIMED_WAITING) return false;
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

    private void singleJobAndExecution() throws Exception {
      executionEnded();
      require(jobIds.size() == 1, "parallel result used more than one Spark job");
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

  @FunctionalInterface
  private interface CheckedAction { void run() throws Exception; }

  private static Throwable cleanup(Throwable original, CheckedAction action) {
    try { action.run(); } catch (Throwable error) {
      if (original == null) return error;
      if (error != original) original.addSuppressed(error);
    }
    return original;
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

  private static void require(boolean value, String message) {
    if (!value) throw new AssertionError(message);
  }
}
