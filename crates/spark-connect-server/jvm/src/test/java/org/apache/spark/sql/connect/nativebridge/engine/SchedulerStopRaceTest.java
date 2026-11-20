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

import java.io.ByteArrayInputStream;
import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;
import java.util.Collections;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.spark.SimpleFutureAction;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.TaskContext;
import org.apache.spark.connect.proto.Range;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.rdd.RDD;
import org.apache.spark.scheduler.DAGScheduler;
import org.apache.spark.scheduler.JobWaiter;
import org.apache.spark.scheduler.SparkListener;
import org.apache.spark.scheduler.SparkListenerJobStart;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.util.ArrowUtils$;
import org.apache.spark.util.CallSite;
import org.apache.spark.util.ThreadUtils;

import scala.Function0;
import scala.Function1;
import scala.Function2;
import scala.collection.Iterator;
import scala.collection.immutable.Seq;
import scala.concurrent.CanAwait;
import scala.concurrent.ExecutionContext;
import scala.concurrent.duration.Duration;
import scala.concurrent.duration.FiniteDuration;
import scala.jdk.javaapi.CollectionConverters;
import scala.runtime.AbstractFunction0;
import scala.runtime.AbstractFunction1;
import scala.runtime.AbstractFunction2;
import scala.runtime.BoxedUnit;
import scala.util.Try;

/** Cancellation and host shutdown must stop admission without stranding query cleanup. */
public final class SchedulerStopRaceTest {
  private static final long TIMEOUT_MILLIS = TimeUnit.SECONDS.toMillis(15);

  private enum StopPause {
    CAPTURED_SCHEDULER, BEFORE_SCHEDULER_READ, BEFORE_STOP_ASSERTION, BEFORE_DEFAULT_WINDOW,
    RUNNING_JOB_BEFORE_RETURN
  }

  private SchedulerStopRaceTest() {}

  public static void main(String[] arguments) throws Exception {
    require(arguments.length == 0, "scheduler-stop race test takes no arguments");
    for (StopPause pause : StopPause.values()) executionCrossesCompletedStop(pause);
    drainParkedBeforeStop();
    admissionParkedBeforeStop();
    directTaskCancellationStopsAdmission();
    completedResultSurvivesStop();
    pendingSuccessCallbackSurvivesStop();
    System.out.println("SCHEDULER_STOP_RACE_OK: default resolution or submission crosses completed host stop; "
        + "queued submission is discarded after drain parks; completed jobs at a parked admission "
        + "window release buffered IPC; direct task cancellation stops admission; completed results "
        + "survive host stop and delayed success callbacks; incomplete actions; "
        + "primary job failures retain host-stop context; INTERNAL host-stop failures; "
        + "worker exit and repeated owner cleanup");
  }

  /** Admission must observe Task.cancel even before the scheduler delivers cancellation. */
  private static void directTaskCancellationStopsAdmission() throws Exception {
    CancellingSparkContext context = new CancellingSparkContext();
    SparkSession spark = SparkSession.builder().sparkContext(context).getOrCreate();
    require(spark.sparkContext() == context, "SparkSession did not reuse cancellation context");
    context.setLogLevel("ERROR");
    SparkJvmEngine engine = new SparkJvmEngine(spark, 2, Long.MAX_VALUE);
    SparkJvmEngine.Session session;
    try (SparkJvmEngine.Task opening = engine.newTask("direct-cancel session open")) {
      session = engine.openSession("scheduler-stop-test", UUID.randomUUID().toString(), opening);
    }
    SparkJvmEngine.Task task = engine.newTask("direct task cancellation during admission");
    SparkJvmEngine.Query query = session.prepare(Relation.newBuilder().setRange(
        Range.newBuilder().setStart(0).setEnd(40).setStep(1).setNumPartitions(4))
        .build().toByteArray(), task);
    context.task = task;
    AtomicReference<Throwable> workerFailure = new AtomicReference<>();
    var runWorker = query.getClass().getDeclaredMethod("runWorker");
    runWorker.setAccessible(true);
    Thread worker = thread("native-direct-task-cancel", () -> {
      try {
        runWorker.invoke(query);
      } catch (InvocationTargetException error) {
        workerFailure.set(error.getCause());
      } catch (Throwable error) {
        workerFailure.set(error);
      }
    });
    Throwable failure = null;
    try {
      // Do not start a result recipient: its cancellation check could set pullFailure and
      // independently stop admission, hiding a missing task flag in stopping().
      field(query, "worker").set(query, worker);
      worker.start();
      join(worker);
      throwFailure(workerFailure.get());
      require(context.submissions.get() == 1 && query.submittedPartitions() == 1
              && context.suppressedCancellations.get() == 1,
          "direct task cancellation admitted another partition before scheduler delivery");
      require(field(query, "workerFinished").getBoolean(query)
              && !field(query, "released").getBoolean(query),
          "direct cancellation did not finish without query close");
      try {
        query.nextIpcBatch();
        throw new AssertionError("direct cancellation returned an IPC result");
      } catch (SparkJvmEngine.EngineException error) {
        require(error.code() == 1, "direct task cancellation lost CANCELLED: " + error);
      }
      boundedClose(query, task, session, engine);
      boundedClose(query, task, session, engine);
      require(engine.isClosed() && registry(engine, "tasks").isEmpty()
              && registry(engine, "sessions").isEmpty()
              && registry(session, "queries").isEmpty()
              && registry(session, "activeTasks").isEmpty(),
          "direct cancellation retained an owner after repeated close");
    } catch (Throwable error) {
      failure = error;
    } finally {
      failure = cleanup(failure, () -> boundedClose(query, task, session, engine));
      failure = cleanup(failure, spark::stop);
    }
    throwFailure(failure);
  }

  /** A completed job's buffered result remains valid if the host stops before drain. */
  private static void completedResultSurvivesStop() throws Exception {
    StopAfterCompletionSparkContext context = new StopAfterCompletionSparkContext();
    SparkSession spark = SparkSession.builder().sparkContext(context)
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "20").getOrCreate();
    require(spark.sparkContext() == context, "SparkSession did not reuse completion context");
    context.setLogLevel("ERROR");
    SparkJvmEngine engine = new SparkJvmEngine(spark, 1, Long.MAX_VALUE);
    SparkJvmEngine.Session session;
    try (SparkJvmEngine.Task opening = engine.newTask("complete-result session open")) {
      session = engine.openSession("scheduler-stop-test", UUID.randomUUID().toString(), opening);
    }
    SparkJvmEngine.Task task = engine.newTask("host stop after successful partition completion");
    SparkJvmEngine.Query query = session.prepare(Relation.newBuilder().setRange(
        Range.newBuilder().setStart(0).setEnd(20).setStep(1).setNumPartitions(1))
        .build().toByteArray(), task);
    Throwable failure = null;
    try {
      NativeIpcBatch batch = query.nextIpcBatch();
      require(batch != null, "completed query omitted its IPC result");
      Thread worker = (Thread) field(query, "worker").get(query);
      join(worker);
      require(context.isStopped() && query.submittedPartitions() == 1
              && field(query, "finishedJobs").getInt(query) == 1
              && field(query, "workerFinished").getBoolean(query),
          "probe did not stop after its only Spark action completed");
      long next = 0;
      while (batch != null) {
        next = checkValues(batch, next);
        batch = query.nextIpcBatch();
      }
      require(next == 20 && query.nextIpcBatch() == null,
          "completed query lost rows or stable EOF after host stop");
      boundedClose(query, task, session, engine);
      boundedClose(query, task, session, engine);
      require(engine.isClosed() && registry(engine, "tasks").isEmpty()
              && registry(engine, "sessions").isEmpty()
              && registry(session, "queries").isEmpty()
              && registry(session, "activeTasks").isEmpty(),
          "completed host-stop result retained an owner after repeated close");
    } catch (Throwable error) {
      failure = error;
    } finally {
      failure = cleanup(failure, () -> boundedClose(query, task, session, engine));
      failure = cleanup(failure, spark::stop);
    }
    throwFailure(failure);
  }

  /** A successful action may still be waiting to record its completion under results. */
  private static void pendingSuccessCallbackSurvivesStop() throws Exception {
    GatedCompletionSparkContext context = new GatedCompletionSparkContext();
    SparkSession spark = SparkSession.builder().sparkContext(context)
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "20").getOrCreate();
    require(spark.sparkContext() == context, "SparkSession did not reuse callback context");
    context.setLogLevel("ERROR");
    SparkJvmEngine engine = new SparkJvmEngine(spark, 1, Long.MAX_VALUE);
    SparkJvmEngine.Session session;
    try (SparkJvmEngine.Task opening = engine.newTask("pending-callback session open")) {
      session = engine.openSession("scheduler-stop-test", UUID.randomUUID().toString(), opening);
    }
    SparkJvmEngine.Task task = engine.newTask("host stop before success callback records completion");
    SparkJvmEngine.Query query = session.prepare(Relation.newBuilder().setRange(
        Range.newBuilder().setStart(0).setEnd(20).setStep(1).setNumPartitions(1))
        .build().toByteArray(), task);
    context.results = field(query, "results").get(query);
    AtomicReference<Throwable> stopFailure = new AtomicReference<>();
    Thread stopper = thread("native-pending-callback-host-stop", () -> {
      try { context.stop(); } catch (Throwable error) { stopFailure.set(error); }
    });
    Throwable failure = null;
    try {
      NativeIpcBatch batch = query.nextIpcBatch();
      require(batch != null, "successful action omitted its IPC result");
      context.worker = (Thread) field(query, "worker").get(query);
      require(context.callbackEntered.await(TIMEOUT_MILLIS, TimeUnit.MILLISECONDS),
          "real Spark action did not enter its success callback");
      require(context.action.isCompleted() && context.action.value().isDefined()
              && context.action.value().get().isSuccess()
              && field(query, "finishedJobs").getInt(query) == 0,
          "probe did not delay recording an already successful Spark action");
      stopper.start();
      require(context.workerSawStop.await(TIMEOUT_MILLIS, TimeUnit.MILLISECONDS),
          "query drain did not observe host stop while success callback was pending");
      // The observation is inside results. The delayed callback cannot acquire that
      // monitor until the worker has finished the exact stop check being tested.
      context.releaseCallback.countDown();
      join(stopper);
      throwFailure(stopFailure.get());
      join(context.worker);
      require(context.isStopped() && query.submittedPartitions() == 1
              && field(query, "finishedJobs").getInt(query) == 1
              && field(query, "workerFinished").getBoolean(query),
          "successful callback or worker did not finish across host stop");
      long next = 0;
      while (batch != null) {
        next = checkValues(batch, next);
        batch = query.nextIpcBatch();
      }
      require(next == 20 && query.nextIpcBatch() == null,
          "delayed successful callback lost rows or stable EOF after host stop");
      boundedClose(query, task, session, engine);
      boundedClose(query, task, session, engine);
      require(engine.isClosed() && registry(engine, "tasks").isEmpty()
              && registry(engine, "sessions").isEmpty()
              && registry(session, "queries").isEmpty()
              && registry(session, "activeTasks").isEmpty(),
          "delayed successful callback retained an owner after repeated close");
    } catch (Throwable error) {
      failure = error;
    } finally {
      context.releaseCallback.countDown();
      failure = cleanup(failure, () -> {
        join(stopper);
        throwFailure(stopFailure.get());
      });
      failure = cleanup(failure, () -> boundedClose(query, task, session, engine));
      failure = cleanup(failure, spark::stop);
    }
    throwFailure(failure);
  }

  /** A paused client must not strand a worker whose admitted jobs have already finished. */
  private static void admissionParkedBeforeStop() throws Exception {
    SparkSession spark = SparkSession.builder().appName("Native admission-window host stop")
        .config("spark.sql.artifact.isolation.enabled", "true")
        .config("spark.sql.artifact.isolation.alwaysApplyClassloader", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10")
        .getOrCreate();
    spark.sparkContext().setLogLevel("ERROR");
    SparkJvmEngine engine = new SparkJvmEngine(spark, 2, Long.MAX_VALUE);
    SparkJvmEngine.Session session = null;
    SparkJvmEngine.Task task = null;
    SparkJvmEngine.Query query = null;
    Throwable failure = null;
    try {
      try (SparkJvmEngine.Task opening = engine.newTask("admission-window session open")) {
        session = engine.openSession("scheduler-stop-test", UUID.randomUUID().toString(), opening);
      }
      task = engine.newTask("host stop with a parked admission window");
      query = session.prepare(Relation.newBuilder().setRange(Range.newBuilder()
          .setStart(0).setEnd(800).setStep(1).setNumPartitions(8)).build().toByteArray(), task);
      NativeIpcBatch first = query.nextIpcBatch();
      require(first != null && first.rowCount() == 10, "missing first admission-window batch");
      Thread worker = (Thread) field(query, "worker").get(query);
      Object results = field(query, "results").get(query);
      boolean parked = false;
      long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(TIMEOUT_MILLIS);
      while (System.nanoTime() < deadline) {
        synchronized (results) {
          parked = query.submittedPartitions() == 2
              && field(query, "finishedJobs").getInt(query) == 2
              && field(query, "partitionBatches").get(query) != null
              && worker.getState() == Thread.State.TIMED_WAITING
              && Arrays.stream(worker.getStackTrace()).anyMatch(frame ->
                  frame.getClassName().equals(SparkJvmEngine.Query.class.getName())
                      && frame.getMethodName().equals("runJobs"));
        }
        if (parked) break;
        Thread.sleep(5);
      }
      require(parked, "worker did not park at a full admission window after both jobs finished");
      spark.sparkContext().stop();
      join(worker);
      require(field(query, "workerFinished").getBoolean(query),
          "admission-parked worker did not finish after host stop");
      SparkJvmEngine.EngineException stopped;
      try {
        query.nextIpcBatch();
        throw new AssertionError("host stop returned buffered IPC from an incomplete query");
      } catch (SparkJvmEngine.EngineException error) {
        require(error.code() == 13
                && "hosting SparkContext stopped during execution".equals(error.getMessage()),
            "admission-parked stop lost its INTERNAL host-stop failure: " + error);
        stopped = error;
      }
      try {
        query.nextIpcBatch();
        throw new AssertionError("repeated fetch forgot the host-stop failure");
      } catch (SparkJvmEngine.EngineException repeated) {
        require(repeated == stopped, "repeated fetch replaced its sticky host-stop failure");
      }
      synchronized (results) {
        require(query.submittedPartitions() == 2
                && field(query, "finishedJobs").getInt(query) == 2
                && field(query, "partitionBatches").get(query) == null,
            "host stop admitted more partitions or retained buffered IPC");
      }
      boundedClose(query, task, session, engine);
      boundedClose(query, task, session, engine);
      require(engine.isClosed() && registry(engine, "tasks").isEmpty()
              && registry(engine, "sessions").isEmpty()
              && registry(session, "queries").isEmpty()
              && registry(session, "activeTasks").isEmpty(),
          "admission-window host stop retained an owner after repeated close");
    } catch (Throwable error) {
      failure = error;
    } finally {
      if (query != null) failure = cleanup(failure, query::close);
      if (task != null) failure = cleanup(failure, task::close);
      if (session != null) failure = cleanup(failure, session::close);
      failure = cleanup(failure, engine::close);
      failure = cleanup(failure, spark::stop);
    }
    throwFailure(failure);
  }

  private static void executionCrossesCompletedStop(StopPause pause) throws Exception {
    GatedSparkContext context = new GatedSparkContext(
        new SparkConf().setAppName("Native scheduler-stop execution race " + pause), pause);
    SparkSession spark = SparkSession.builder().sparkContext(context).getOrCreate();
    require(spark.sparkContext() == context, "SparkSession did not reuse the gated SparkContext");
    context.setLogLevel("ERROR");
    SparkJvmEngine engine = new SparkJvmEngine(spark, 0, 1L << 30);
    SparkJvmEngine.Session session;
    try (SparkJvmEngine.Task opening = engine.newTask("scheduler-stop session open")) {
      session = engine.openSession("scheduler-stop-test", UUID.randomUUID().toString(), opening);
    }
    SparkJvmEngine.Task task = engine.newTask("submission overtaken by host stop");
    SparkJvmEngine.Query query = session.prepare(Relation.newBuilder().setRange(
        Range.newBuilder().setStart(0).setEnd(20).setStep(1).setNumPartitions(1))
        .build().toByteArray(), task);
    AtomicReference<Throwable> fetchFailure = new AtomicReference<>();
    AtomicReference<Throwable> stopFailure = new AtomicReference<>();
    Thread fetcher = thread("native-scheduler-stop-fetch", () -> {
      try { query.nextIpcBatch(); } catch (Throwable error) { fetchFailure.set(error); }
    });
    Thread stopper = thread("native-scheduler-stop-host", () -> {
      try { context.stop(); } catch (Throwable error) { stopFailure.set(error); }
    });
    Throwable failure = null;
    try {
      context.defaultWindowArmed = true;
      fetcher.start();
      require(context.entered.await(TIMEOUT_MILLIS, TimeUnit.MILLISECONDS),
          "query worker did not reach the stopped-context boundary");
      int admitted = pause == StopPause.BEFORE_DEFAULT_WINDOW ? 0 : 1;
      require(query.submittedPartitions() == admitted,
          "query admitted the wrong number of partitions before its stopped-context boundary");
      if (pause == StopPause.RUNNING_JOB_BEFORE_RETURN) {
        require(context.orphan != null && !context.orphan.isCompleted(),
            "all-admitted fixture did not retain a real unfinished Spark action");
      }
      stopper.start();
      join(stopper);
      throwFailure(stopFailure.get());
      require(context.isStopped() && context.dagScheduler() == null,
          "host did not finish scheduler shutdown before execution resumed");
      context.release.countDown();
      join(fetcher);
      Thread worker = (Thread) field(query, "worker").get(query);
      require(worker != null, "query did not create its execution worker");
      join(worker);
      Throwable fetchError = fetchFailure.get();
      require(fetchError instanceof SparkJvmEngine.EngineException
              && ((SparkJvmEngine.EngineException) fetchError).code() == 13,
          "stopped execution lost its INTERNAL host-stop failure: " + fetchError);
      require(field(query, "workerFinished").getBoolean(query),
          "query worker did not finish after host stop");
      if (pause == StopPause.RUNNING_JOB_BEFORE_RETURN) {
        require(context.orphan.isCompleted() && context.orphan.value().isDefined()
                && context.orphan.value().get().isFailure(),
            "host stop did not fail the real unfinished Spark action");
        Throwable original = context.orphan.value().get().failed().get();
        require(fetchError.getCause() == original
                && String.valueOf(original.getMessage()).equals(fetchError.getMessage())
                && Arrays.stream(original.getSuppressed()).anyMatch(error ->
                    error instanceof SparkJvmEngine.EngineException
                        && ((SparkJvmEngine.EngineException) error).code() == 13
                        && "hosting SparkContext stopped during execution".equals(error.getMessage())),
            "all-admitted stop replaced its primary job failure or lost suppressed host-stop context");
        try {
          query.nextIpcBatch();
          throw new AssertionError("repeated fetch forgot the failed action");
        } catch (SparkJvmEngine.EngineException repeated) {
          require(repeated == fetchError, "repeated fetch replaced its failed action");
        }
      } else if (pause == StopPause.CAPTURED_SCHEDULER) {
        require(context.orphan != null && !context.orphan.isCompleted(),
            "probe did not retain a real incomplete Spark action");
      } else {
        RuntimeException original = context.boundaryFailure;
        require(original != null && (pause == StopPause.BEFORE_SCHEDULER_READ
                ? original instanceof NullPointerException
                : original instanceof IllegalStateException),
            "execution did not exercise its intended stopped-context boundary: " + original);
        if (pause != StopPause.BEFORE_SCHEDULER_READ) {
          require(String.valueOf(original.getMessage()).startsWith(
                  "Cannot call methods on a stopped SparkContext."),
              "boundary did not reach Spark's stopped-context assertion: " + original);
        }
        require(Arrays.stream(fetchError.getSuppressed()).anyMatch(error -> error == original),
            "host-stop failure did not retain the original boundary exception");
        require(context.orphan == null, "rejected submission unexpectedly created a Spark action");
      }
      if (pause != StopPause.RUNNING_JOB_BEFORE_RETURN) {
        require("hosting SparkContext stopped during execution".equals(fetchError.getMessage()),
            "stopped execution lost its primary host-stop marker: " + fetchError);
      }
      int completed = field(query, "finishedJobs").getInt(query);
      if (pause == StopPause.RUNNING_JOB_BEFORE_RETURN) {
        require(completed == 1 && query.submittedPartitions() == 1,
            "all-admitted fixture did not record its real failed callback");
      } else {
        require(pause == StopPause.BEFORE_DEFAULT_WINDOW
                ? completed == 0 && query.submittedPartitions() == 0
                : completed < query.submittedPartitions(),
            "query fabricated completion at its stopped-context boundary");
      }

      boundedClose(query, task, session, engine);
      boundedClose(query, task, session, engine);
      require(engine.isClosed() && registry(engine, "tasks").isEmpty()
              && registry(engine, "sessions").isEmpty()
              && registry(session, "queries").isEmpty()
              && registry(session, "activeTasks").isEmpty()
              && field(query, "partitionBatches").get(query) == null,
          "scheduler-stop cleanup retained an owner or unread IPC");
    } catch (Throwable error) {
      failure = error;
    } finally {
      // Always release the test hook, including when host stop or a probe assertion failed.
      context.release.countDown();
      failure = cleanup(failure, () -> {
        join(stopper);
        throwFailure(stopFailure.get());
      });
      failure = cleanup(failure, () -> {
        fetcher.interrupt();
        join(fetcher);
      });
      failure = cleanup(failure, () -> boundedClose(query, task, session, engine));
      failure = cleanup(failure, spark::stop);
    }
    throwFailure(failure);
  }

  /** Hold the real scheduler's result callback so the native submission queues behind it. */
  private static void drainParkedBeforeStop() throws Exception {
    SparkSession spark = SparkSession.builder().appName("Native parked-drain shutdown race")
        .getOrCreate();
    SparkContext context = spark.sparkContext();
    context.setLogLevel("ERROR");
    SparkJvmEngine engine = new SparkJvmEngine(spark);
    SparkJvmEngine.Session session;
    try (SparkJvmEngine.Task opening = engine.newTask("parked-drain session open")) {
      session = engine.openSession("scheduler-stop-test", UUID.randomUUID().toString(), opening);
    }
    SparkJvmEngine.Task task = engine.newTask("queued submission with parked drain");
    SparkJvmEngine.Query query = session.prepare(Relation.newBuilder().setRange(
        Range.newBuilder().setStart(0).setEnd(20).setStep(1).setNumPartitions(1))
        .build().toByteArray(), task);
    CountDownLatch entered = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);
    AtomicBoolean handlerInterrupted = new AtomicBoolean();
    AtomicReference<Throwable> fetchFailure = new AtomicReference<>();
    AtomicReference<Throwable> stopFailure = new AtomicReference<>();
    Thread fetcher = thread("native-parked-drain-fetch", () -> {
      try { query.nextIpcBatch(); } catch (Throwable error) { fetchFailure.set(error); }
    });
    Thread stopper = thread("native-parked-drain-host-stop", () -> {
      try { context.stop(); } catch (Throwable error) { stopFailure.set(error); }
    });
    Throwable failure = null;
    try {
      RDD<Long> rows = spark.range(0, 1, 1, 1).toJavaRDD().rdd();
      SimpleFutureAction<BoxedUnit> gate = context.submitJob(rows, new GatePartition(),
          CollectionConverters.asScala(Collections.<Object>singletonList(0)).toSeq(),
          new AbstractFunction2<Object, Integer, BoxedUnit>() {
            @Override
            public BoxedUnit apply(Object ignored, Integer value) {
              entered.countDown();
              try {
                require(release.await(2 * TIMEOUT_MILLIS, TimeUnit.MILLISECONDS),
                    "scheduler result-handler gate timed out");
              } catch (InterruptedException error) {
                handlerInterrupted.set(true);
              }
              return BoxedUnit.UNIT;
            }
          }, new AbstractFunction0<BoxedUnit>() {
            @Override
            public BoxedUnit apply() { return BoxedUnit.UNIT; }
          });
      require(entered.await(TIMEOUT_MILLIS, TimeUnit.MILLISECONDS),
          "test job did not hold the real scheduler result callback");
      fetcher.start();
      Thread worker = null;
      boolean parked = false;
      long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(TIMEOUT_MILLIS);
      while (System.nanoTime() < deadline) {
        synchronized (field(query, "results").get(query)) {
          worker = (Thread) field(query, "worker").get(query);
          parked = worker != null && query.submittedPartitions() == 1
              && field(query, "finishedJobs").getInt(query) == 0;
        }
        parked = parked && worker.getState() == Thread.State.TIMED_WAITING
            && Arrays.stream(worker.getStackTrace()).anyMatch(frame ->
                frame.getClassName().equals(SparkJvmEngine.Query.class.getName())
                    && frame.getMethodName().equals("runJobs"));
        if (parked) break;
        Thread.sleep(5);
      }
      require(parked && !handlerInterrupted.get() && !gate.isCompleted(),
          "query did not park in drain behind the held scheduler callback");
      stopper.start();
      join(stopper);
      throwFailure(stopFailure.get());
      require(handlerInterrupted.get() && gate.isCompleted() && context.dagScheduler() == null,
          "host stop did not interrupt and drain the held scheduler callback");
      join(fetcher);
      join(worker);
      Throwable fetchError = fetchFailure.get();
      require(fetchError instanceof SparkJvmEngine.EngineException
              && ((SparkJvmEngine.EngineException) fetchError).code() == 13
              && "hosting SparkContext stopped during execution".equals(fetchError.getMessage()),
          "parked drain lost its INTERNAL host-stop failure: " + fetchError);
      require(field(query, "workerFinished").getBoolean(query)
              && query.submittedPartitions() == 1 && field(query, "finishedJobs").getInt(query) == 0,
          "parked drain did not exit with its queued submission still incomplete");
      boundedClose(query, task, session, engine);
      boundedClose(query, task, session, engine);
      require(engine.isClosed() && registry(engine, "tasks").isEmpty()
              && registry(engine, "sessions").isEmpty()
              && registry(session, "queries").isEmpty()
              && registry(session, "activeTasks").isEmpty()
              && field(query, "partitionBatches").get(query) == null,
          "parked-drain cleanup retained an owner or unread IPC");
    } catch (Throwable error) {
      failure = error;
    } finally {
      release.countDown();
      failure = cleanup(failure, () -> {
        join(stopper);
        throwFailure(stopFailure.get());
      });
      failure = cleanup(failure, () -> {
        fetcher.interrupt();
        join(fetcher);
      });
      failure = cleanup(failure, () -> boundedClose(query, task, session, engine));
      failure = cleanup(failure, spark::stop);
    }
    throwFailure(failure);
  }

  private static final class GatePartition extends AbstractFunction1<Iterator<Long>, Integer>
      implements java.io.Serializable {
    private static final long serialVersionUID = 1L;

    @Override
    public Integer apply(Iterator<Long> rows) { return 0; }
  }

  private static final class WaitingPartition<T, U> extends AbstractFunction1<Iterator<T>, U>
      implements java.io.Serializable {
    private static final long serialVersionUID = 1L;
    private final Function1<Iterator<T>, U> delegate;

    private WaitingPartition(Function1<Iterator<T>, U> delegate) { this.delegate = delegate; }

    @Override
    public U apply(Iterator<T> rows) {
      try {
        Thread.sleep(2 * TIMEOUT_MILLIS);
      } catch (InterruptedException error) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException("waiting partition interrupted by host stop", error);
      }
      return delegate.apply(rows);
    }
  }

  private static final class CancellingSparkContext extends SparkContext {
    private final AtomicInteger submissions = new AtomicInteger();
    private final AtomicInteger suppressedCancellations = new AtomicInteger();
    private SparkJvmEngine.Task task;

    private CancellingSparkContext() {
      super(new SparkConf().setAppName("Native direct task cancellation"));
    }

    @Override
    public <T, U, R> SimpleFutureAction<R> submitJob(RDD<T> rdd,
        Function1<Iterator<T>, U> processPartition, Seq<Object> partitions,
        Function2<Object, U, BoxedUnit> resultHandler, Function0<R> resultFunc) {
      SimpleFutureAction<R> action = super.submitJob(
          rdd, processPartition, partitions, resultHandler, resultFunc);
      if (submissions.incrementAndGet() == 1) task.cancel();
      return action;
    }

    @Override
    public void cancelJobGroupAndFutureJobs(String group) {
      suppressedCancellations.incrementAndGet();
    }
  }

  private static final class StopAfterCompletionSparkContext extends SparkContext {
    private StopAfterCompletionSparkContext() {
      super(new SparkConf().setAppName("Native completed-result host stop"));
    }

    @Override
    public <T, U, R> SimpleFutureAction<R> submitJob(RDD<T> rdd,
        Function1<Iterator<T>, U> processPartition, Seq<Object> partitions,
        Function2<Object, U, BoxedUnit> resultHandler, Function0<R> resultFunc) {
      SimpleFutureAction<R> action = super.submitJob(
          rdd, processPartition, partitions, resultHandler, resultFunc);
      try {
        ThreadUtils.awaitResult(action, new FiniteDuration(TIMEOUT_MILLIS, TimeUnit.MILLISECONDS));
      } catch (Exception error) {
        throw new IllegalStateException("Spark action did not complete before host stop", error);
      }
      stop();
      return action;
    }
  }

  private static final class GatedCompletionSparkContext extends SparkContext {
    private final CountDownLatch registered = new CountDownLatch(1);
    private final CountDownLatch callbackEntered = new CountDownLatch(1);
    private final CountDownLatch releaseCallback = new CountDownLatch(1);
    private final CountDownLatch workerSawStop = new CountDownLatch(1);
    private volatile SimpleFutureAction<?> action;
    private volatile Thread worker;
    private Object results;

    private GatedCompletionSparkContext() {
      super(new SparkConf().setAppName("Native pending-success-callback host stop"));
    }

    @Override
    public boolean isStopped() {
      boolean stopped = super.isStopped();
      if (stopped && Thread.currentThread() == worker && Thread.holdsLock(results)) {
        StackTraceElement[] stack = Thread.currentThread().getStackTrace();
        boolean inDrain = Arrays.stream(stack).anyMatch(frame ->
            frame.getClassName().equals(SparkJvmEngine.Query.class.getName())
                && frame.getMethodName().equals("runJobs"));
        boolean inAdmission = Arrays.stream(stack).anyMatch(frame ->
            frame.getClassName().equals(SparkJvmEngine.Query.class.getName())
                && frame.getMethodName().equals("stopping"));
        if (inDrain && !inAdmission) workerSawStop.countDown();
      }
      return stopped;
    }

    @Override
    public <T, U, R> SimpleFutureAction<R> submitJob(RDD<T> rdd,
        Function1<Iterator<T>, U> processPartition, Seq<Object> partitions,
        Function2<Object, U, BoxedUnit> resultHandler, Function0<R> resultFunc) {
      SimpleFutureAction<R> submitted = super.submitJob(rdd, processPartition, partitions,
          new AbstractFunction2<Object, U, BoxedUnit>() {
            @Override
            public BoxedUnit apply(Object partition, U value) {
              awaitCallbackGate(registered);
              return resultHandler.apply(partition, value);
            }
          }, resultFunc);
      action = submitted;
      JobWaiter<?> waiter;
      try {
        waiter = (JobWaiter<?>) field(submitted, "jobWaiter").get(submitted);
      } catch (ReflectiveOperationException error) {
        throw new IllegalStateException("cannot inspect real Spark action waiter", error);
      }
      return new SimpleFutureAction<R>(waiter, resultFunc) {
        // Scala's implementation declares Exception, but Awaitable's Java view does not.
        @Override
        public R result(Duration atMost, CanAwait permit) {
          try {
            return super.result(atMost, permit);
          } catch (RuntimeException error) {
            throw error;
          } catch (Exception error) {
            if (error instanceof InterruptedException) Thread.currentThread().interrupt();
            throw new IllegalStateException("Spark action await failed", error);
          }
        }

        @Override
        public <V> void onComplete(Function1<Try<R>, V> callback, ExecutionContext executor) {
          super.onComplete(new AbstractFunction1<Try<R>, V>() {
            @Override
            public V apply(Try<R> outcome) {
              require(outcome.isSuccess(), "gated Spark action failed before host stop");
              callbackEntered.countDown();
              awaitCallbackGate(releaseCallback);
              return callback.apply(outcome);
            }
          }, executor);
          registered.countDown();
        }
      };
    }

    // Host stop interrupts its scheduler. Keep the real callback parked until the
    // test releases it, then restore the scheduler's interrupt for normal shutdown.
    private void awaitCallbackGate(CountDownLatch gate) {
      boolean interrupted = false;
      long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(TIMEOUT_MILLIS);
      try {
        while (gate.getCount() != 0) {
          long remaining = deadline - System.nanoTime();
          require(remaining > 0, "success callback gate timed out");
          try {
            require(gate.await(remaining, TimeUnit.NANOSECONDS), "success callback gate timed out");
          } catch (InterruptedException error) {
            interrupted = true;
          }
        }
      } finally {
        if (interrupted) Thread.currentThread().interrupt();
      }
    }
  }

  private static final class GatedSparkContext extends SparkContext {
    private final StopPause pause;
    private final CountDownLatch entered = new CountDownLatch(1);
    private final CountDownLatch release = new CountDownLatch(1);
    private volatile SimpleFutureAction<?> orphan;
    private volatile RuntimeException boundaryFailure;
    private volatile boolean defaultWindowArmed;

    private GatedSparkContext(SparkConf configuration, StopPause pause) {
      super(configuration);
      this.pause = pause;
    }

    @Override
    public int defaultParallelism() {
      // Spark can also ask during planning; gate only the per-query window resolution.
      if (pause == StopPause.BEFORE_DEFAULT_WINDOW && defaultWindowArmed
          && Arrays.stream(Thread.currentThread().getStackTrace()).anyMatch(frame ->
              frame.getClassName().equals(SparkJvmEngine.Query.class.getName())
                  && frame.getMethodName().equals("runJobs"))) {
        defaultWindowArmed = false;
        awaitGate();
        try {
          return super.defaultParallelism();
        } catch (RuntimeException error) {
          boundaryFailure = error;
          throw error;
        }
      }
      return super.defaultParallelism();
    }

    @Override
    public <T, U, R> SimpleFutureAction<R> submitJob(RDD<T> rdd,
        Function1<Iterator<T>, U> processPartition, Seq<Object> partitions,
        Function2<Object, U, BoxedUnit> resultHandler, Function0<R> resultFunc) {
      try {
        if (pause == StopPause.RUNNING_JOB_BEFORE_RETURN) {
          CountDownLatch started = new CountDownLatch(1);
          String group = getLocalProperty("spark.jobGroup.id");
          SparkListener listener = new SparkListener() {
            @Override
            public void onJobStart(SparkListenerJobStart event) {
              if (group.equals(event.properties().getProperty("spark.jobGroup.id"))) {
                started.countDown();
              }
            }
          };
          addSparkListener(listener);
          try {
            SimpleFutureAction<R> action = super.submitJob(rdd,
                new WaitingPartition<>(processPartition), partitions, resultHandler, resultFunc);
            orphan = action;
            awaitGate(started, "real Spark job did not start before its host-stop gate");
            awaitGate();
            return action;
          } finally {
            removeSparkListener(listener);
          }
        }
        if (pause == StopPause.BEFORE_STOP_ASSERTION) awaitGate();
        if (pause != StopPause.CAPTURED_SCHEDULER && pause != StopPause.BEFORE_SCHEDULER_READ) {
          return super.submitJob(rdd, processPartition, partitions, resultHandler, resultFunc);
        }
        // Only scheduler-reference gates need the pinned SparkContext submission sequence.
        // CAPTURED_SCHEDULER models a pause inside DAGScheduler before its event post.
        assertNotStopped();
        Function1<Iterator<T>, U> cleanFunction = clean(processPartition, true);
        CallSite callSite = getCallSite();
        if (pause == StopPause.BEFORE_SCHEDULER_READ) awaitGate();
        DAGScheduler scheduler = dagScheduler();
        if (pause == StopPause.CAPTURED_SCHEDULER) {
          require(scheduler != null && scheduler.eventProcessLoop().isActive(),
              "submission did not capture its live scheduler");
          awaitGate();
          require(isStopped() && dagScheduler() == null && !scheduler.eventProcessLoop().isActive(),
              "submission resumed before its captured scheduler stopped");
        }
        JobWaiter<U> waiter = scheduler.submitJob(rdd,
            new AbstractFunction2<TaskContext, Iterator<T>, U>() {
              @Override
              public U apply(TaskContext ignored, Iterator<T> rows) {
                return cleanFunction.apply(rows);
              }
            }, partitions, callSite, resultHandler, getLocalProperties());
        SimpleFutureAction<R> action = new SimpleFutureAction<>(waiter, resultFunc);
        orphan = action;
        return action;
      } catch (RuntimeException error) {
        boundaryFailure = error;
        throw error;
      }
    }

    private void awaitGate() {
      entered.countDown();
      awaitGate(release, "host did not release the submission gate");
    }

    private void awaitGate(CountDownLatch gate, String message) {
      try {
        require(gate.await(2 * TIMEOUT_MILLIS, TimeUnit.MILLISECONDS), message);
      } catch (InterruptedException error) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException("submission gate was interrupted", error);
      }
    }
  }

  private static void boundedClose(SparkJvmEngine.Query query, SparkJvmEngine.Task task,
      SparkJvmEngine.Session session, SparkJvmEngine engine) throws Exception {
    AtomicReference<Throwable> failure = new AtomicReference<>();
    Thread closer = thread("native-scheduler-stop-owner-close", () -> {
      Throwable error = cleanup(null, query::close);
      error = cleanup(error, task::close);
      error = cleanup(error, session::close);
      error = cleanup(error, engine::close);
      failure.set(error);
    });
    closer.start();
    join(closer);
    throwFailure(failure.get());
  }

  private static long checkValues(NativeIpcBatch batch, long next) throws Exception {
    try (BufferAllocator allocator = ArrowUtils$.MODULE$.rootAllocator()
            .newChildAllocator("completed-result-stop-test", 0, Long.MAX_VALUE);
         ArrowStreamReader reader = new ArrowStreamReader(
             new ByteArrayInputStream(batch.data()), allocator)) {
      require(reader.loadNextBatch(), "completed IPC omitted its record batch");
      var root = reader.getVectorSchemaRoot();
      require(root.getRowCount() == batch.rowCount(), "completed IPC row count changed");
      for (int row = 0; row < root.getRowCount(); row++) {
        require(((Number) root.getVector(0).getObject(row)).longValue() == next++,
            "completed result lost, duplicated or reordered a row after host stop");
      }
      require(!reader.loadNextBatch(), "completed IPC contains multiple record batches");
    }
    return next;
  }

  private static Thread thread(String name, Runnable action) {
    Thread thread = new Thread(action, name);
    thread.setDaemon(true);
    return thread;
  }

  private static void join(Thread thread) throws InterruptedException {
    thread.join(TIMEOUT_MILLIS);
    require(!thread.isAlive(), "owned thread did not finish: " + thread.getName());
  }

  private static void throwFailure(Throwable error) throws Exception {
    if (error instanceof Exception) throw (Exception) error;
    if (error != null) throw (Error) error;
  }
}
