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
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.nio.file.AccessDeniedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.Logger;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.logging.log4j.core.config.Property;
import org.apache.spark.SparkEnv;
import org.apache.spark.connect.proto.Range;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.scheduler.SparkListener;
import org.apache.spark.scheduler.SparkListenerApplicationEnd;
import org.apache.spark.scheduler.SparkListenerJobEnd;
import org.apache.spark.scheduler.SparkListenerJobStart;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.artifact.ArtifactManager;
import org.apache.spark.sql.util.ArrowUtils$;
import org.apache.spark.util.Utils;

import scala.Option;

/** Unread queries and native factory ownership remain releasable after the host stops Spark. */
public final class StoppedContextCleanupTest {
  private static final long TIMEOUT_MILLIS = TimeUnit.SECONDS.toMillis(15);

  private StoppedContextCleanupTest() {}

  public static void main(String[] arguments) throws Exception {
    require(arguments.length == 1, "pass the absolute native library path");
    run(arguments[0], false, false);
    run(arguments[0], true, false);
    run(arguments[0], false, true);
    System.out.println("STOPPED_CONTEXT_CLEANUP_OK: real unread IPC queries; completed Spark "
        + "jobs and workers; ordered buffered drain after host stop; unused sessions; "
        + "repeated query/session/engine close; "
        + "native factory close; "
        + "artifact/cache/classloader cleanup; permission-denied cleanup when supported; "
        + "listener rebind after completed and in-progress "
        + "host SparkContext.stop");
  }

  private static void run(String library, boolean duringStop, boolean readOnlyArtifacts)
      throws Exception {
    SparkSession spark = SparkSession.builder().appName("Stopped SparkContext cleanup")
        .config("spark.sql.artifact.isolation.enabled", "true")
        .config("spark.sql.artifact.isolation.alwaysApplyClassloader", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "2")
        .getOrCreate();
    spark.sparkContext().setLogLevel(readOnlyArtifacts ? "WARN" : "ERROR");
    Fixture direct = null;
    Fixture nativeOwned = null;
    HostStop hostStop = null;
    DeletionWarnings warnings = null;
    Throwable failure = null;
    try {
      direct = new Fixture(spark, "direct stopped-context cleanup");
      nativeOwned = new Fixture(spark, "native stopped-context cleanup");
      nativeOwned.server = new NativeEngineServer(
          library, "127.0.0.1:0", nativeOwned.engine);
      String address = nativeOwned.server.address();
      int port = Integer.parseInt(address.substring(address.lastIndexOf(':') + 1));
      require(nativeOwned.server.isRunning(), "native listener did not start");
      direct.fetchOneAndAwaitWorker();
      nativeOwned.fetchOneAndAwaitWorker();

      // The Task has submitted real Spark jobs but is not complete until the cursor reaches
      // EOF. Keeping unread output reproduces cancellation after the host has stopped Spark.
      if (duringStop) {
        hostStop = new HostStop(spark);
        hostStop.startAndAwaitGate();
      } else {
        spark.sparkContext().stop();
        require(SparkEnv.get() == null, "completed host stop retained the global SparkEnv");
      }
      require(spark.sparkContext().isStopped(), "host did not stop the shared SparkContext");
      if (readOnlyArtifacts) {
        boolean directDenied = direct.denyArtifactDeletion();
        boolean nativeDenied = nativeOwned.denyArtifactDeletion();
        if (!directDenied || !nativeDenied) {
          direct.restoreArtifactPermissions();
          nativeOwned.restoreArtifactPermissions();
          readOnlyArtifacts = false;
          System.out.println("STOPPED_CONTEXT_READONLY_SKIPPED: current user or filesystem "
              + "cannot deny artifact deletion; running ordinary cleanup instead");
        } else {
          warnings = new DeletionWarnings();
        }
      }
      direct.drainAfterHostStop();
      direct.closeOwners();
      direct.closeOwners();
      if (readOnlyArtifacts) {
        direct.assertReadOnlyResidue();
        require(warnings.count.get() == 1,
            "direct cleanup did not log exactly one directory-deletion warning");
      } else direct.assertCleared();

      // Do not independently close this engine: nativeStop owns its factory and must close
      // the still-open Java Query/Session itself before releasing that factory reference.
      nativeOwned.server.close();
      nativeOwned.server.close();
      require(!nativeOwned.server.isRunning() && nativeOwned.server.isDrained(),
          "native stop did not finish after SparkContext stopped");
      if (readOnlyArtifacts) nativeOwned.assertReadOnlyResidue();
      else nativeOwned.assertCleared();
      nativeOwned.closeOwners();
      nativeOwned.closeOwners();
      if (readOnlyArtifacts) {
        require(warnings.count.get() == 2,
            "native cleanup omitted its warning or repeated close logged another warning");
        direct.restoreArtifactDirectory();
        nativeOwned.restoreArtifactDirectory();
        direct.assertCleared();
        nativeOwned.assertCleared();
      }
      try (ServerSocket rebound = new ServerSocket()) {
        rebound.setReuseAddress(true);
        rebound.bind(new InetSocketAddress("127.0.0.1", port));
        require(rebound.isBound(), "native listener could not rebind after stopped-context close");
      }
    } catch (Throwable error) {
      failure = error;
    } finally {
      // Retried teardown may rethrow the identical retained failure. Java's automatic
      // suppression would replace it with IllegalArgumentException: self-suppression.
      if (nativeOwned != null) failure = cleanup(failure, nativeOwned::close);
      if (direct != null) failure = cleanup(failure, direct::close);
      if (nativeOwned != null) failure = cleanup(failure, nativeOwned::restoreArtifactDirectory);
      if (direct != null) failure = cleanup(failure, direct::restoreArtifactDirectory);
      // Release ApplicationEnd even when an assertion/close failed, and join the host's
      // stop before asking SparkSession to perform its final idempotent cleanup.
      if (hostStop != null) failure = cleanup(failure, hostStop::close);
      failure = cleanup(failure, spark::stop);
      if (warnings != null) failure = cleanup(failure, warnings::close);
    }
    if (failure instanceof Exception) throw (Exception) failure;
    if (failure != null) throw (Error) failure;
  }

  /** Capture the production warning without changing its configured console delivery. */
  private static final class DeletionWarnings extends AbstractAppender implements AutoCloseable {
    private final Logger logger = (Logger) LogManager.getRootLogger();
    private final AtomicInteger count = new AtomicInteger();

    private DeletionWarnings() {
      super("native-artifact-deletion-" + UUID.randomUUID(), null, null, false, Property.EMPTY_ARRAY);
      start();
      logger.addAppender(this);
    }

    @Override
    public void append(LogEvent event) {
      if (SparkJvmEngine.class.getName().equals(event.getLoggerName())
          && event.getLevel() == Level.WARN
          && "Unable to remove native session artifact directory".equals(
              event.getMessage().getFormattedMessage())
          && event.getThrown() instanceof IOException) count.incrementAndGet();
    }

    @Override
    public void close() {
      logger.removeAppender(this);
      stop();
    }
  }

  /** Hold Spark's stop after its stopped flag is set, while its owning environment is live. */
  private static final class HostStop extends SparkListener implements AutoCloseable {
    private final SparkSession spark;
    private final SparkEnv environment;
    private final CountDownLatch entered = new CountDownLatch(1);
    private final CountDownLatch release = new CountDownLatch(1);
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private final Thread thread;

    private HostStop(SparkSession spark) {
      this.spark = spark;
      environment = spark.sparkContext().env();
      thread = new Thread(() -> {
        try { spark.sparkContext().stop(); } catch (Throwable error) { record(error); }
      }, "native-stopped-context-host-stop");
      thread.setDaemon(true);
      spark.sparkContext().addSparkListener(this);
    }

    @Override
    public void onApplicationEnd(SparkListenerApplicationEnd event) {
      entered.countDown();
      try { release.await(); } catch (InterruptedException error) {
        record(error);
        Thread.currentThread().interrupt();
      }
    }

    private void startAndAwaitGate() throws Exception {
      thread.start();
      require(entered.await(TIMEOUT_MILLIS, TimeUnit.MILLISECONDS),
          "host stop did not reach the ApplicationEnd gate");
      require(thread.isAlive() && spark.sparkContext().isStopped()
              && environment != null && SparkEnv.get() == environment,
          "host stop did not pause with its stopped flag and owning SparkEnv still live");
    }

    private void record(Throwable error) {
      if (!failure.compareAndSet(null, error)) {
        Throwable original = failure.get();
        if (original != error) original.addSuppressed(error);
      }
    }

    @Override
    public void close() throws Exception {
      release.countDown();
      boolean interrupted = Thread.interrupted();
      long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(TIMEOUT_MILLIS);
      try {
        while (thread.isAlive()) {
          long remaining = deadline - System.nanoTime();
          require(remaining > 0, "host stop did not finish after its ApplicationEnd gate opened");
          try {
            TimeUnit.NANOSECONDS.timedJoin(thread, remaining);
          } catch (InterruptedException error) {
            interrupted = true;
          }
        }
        Throwable error = failure.get();
        if (error instanceof Exception) throw (Exception) error;
        if (error != null) throw (Error) error;
      } finally {
        spark.sparkContext().removeSparkListener(this);
        if (interrupted) Thread.currentThread().interrupt();
      }
    }
  }

  private static final class Fixture implements AutoCloseable {
    private final SparkSession spark;
    private final SparkJvmEngine engine;
    private final Jobs jobs;
    private SparkJvmEngine.Session session;
    private SparkJvmEngine.Session unusedSession;
    private SparkJvmEngine.Task task;
    private SparkJvmEngine.Query query;
    private NativeEngineServer server;
    private ArtifactManager artifacts;
    private String sessionId;
    private String cachedArtifact;
    private Path artifactPath;
    private Path artifactFile;
    private Path readOnlyResidue;
    private Set<PosixFilePermission> artifactPermissions;
    private long nextValue;

    private Fixture(SparkSession spark, String label) throws Exception {
      this.spark = spark;
      engine = new SparkJvmEngine(spark);
      String description = label + " " + UUID.randomUUID();
      jobs = new Jobs(description);
      spark.sparkContext().addSparkListener(jobs);
      try {
        try (SparkJvmEngine.Task opening = engine.newTask("stopped-context session open")) {
          session = engine.openSession("stopped-context-test", UUID.randomUUID().toString(), opening);
        }
        try (SparkJvmEngine.Task opening = engine.newTask("stopped-context unused session open")) {
          unusedSession = engine.openSession(
              "stopped-context-test", UUID.randomUUID().toString(), opening);
        }
        // Leave this session unused so close cannot depend on prepare/config/analyze having
        // initialized its resources before the host stops the shared SparkContext.
        require(registry(engine, "sessions").size() == 2
                && registry(unusedSession, "queries").isEmpty()
                && registry(unusedSession, "activeTasks").isEmpty(),
            "fixture did not retain its unused session");
        addArtifacts();
        task = engine.newTask(description);
        query = session.prepare(Relation.newBuilder().setRange(Range.newBuilder()
            .setStart(0).setEnd(20).setStep(1).setNumPartitions(1)).build().toByteArray(), task);
      } catch (Throwable failure) {
        try { close(); } catch (Throwable error) {
          if (error != failure) failure.addSuppressed(error);
        }
        if (failure instanceof Exception) throw (Exception) failure;
        throw (Error) failure;
      }
    }

    private void addArtifacts() throws Exception {
      var sessionSpark = (org.apache.spark.sql.classic.SparkSession) field(session, "spark").get(session);
      artifacts = sessionSpark.artifactManager();
      sessionId = sessionSpark.sessionUUID();
      artifactPath = artifacts.artifactPath();
      String name = "stopped-context-" + UUID.randomUUID() + ".txt";
      artifactFile = artifactPath.resolve("files").resolve(name);
      cachedArtifact = "stopped-context-cache-" + UUID.randomUUID();
      addArtifact(Path.of("files", name));
      addArtifact(Path.of("cache", cachedArtifact));
      ClassLoader loader = artifacts.classloader();
      require(Files.isRegularFile(artifactFile)
              && spark.sparkContext().addedFiles().get(sessionId).isDefined()
              && artifacts.getCachedBlockId(cachedArtifact).isDefined()
              && artifacts.cachedClassLoader().isDefined()
              && artifacts.cachedClassLoader().get() == loader,
          "fixture did not materialize session files, cache, registration and classloader");
    }

    private void addArtifact(Path target) throws Exception {
      Path staged = Files.createTempFile("native-stopped-context-", ".txt");
      try {
        Files.writeString(staged, "harmless stopped-context cleanup fixture\n");
        artifacts.addArtifact(target, staged, Option.empty(), true);
      } finally {
        Files.deleteIfExists(staged);
      }
    }

    private boolean denyArtifactDeletion() throws Exception {
      require(spark.sparkContext().isStopped() && SparkEnv.get() == null,
          "read-only artifact fixture requires a completed host stop");
      try {
        artifactPermissions = Files.getPosixFilePermissions(artifactPath);
      } catch (UnsupportedOperationException unsupported) {
        return false;
      }
      readOnlyResidue = artifactPath.resolve("read-only-" + UUID.randomUUID() + ".txt");
      Files.writeString(readOnlyResidue, "harmless owned directory-deletion fixture\n");
      Set<PosixFilePermission> readOnly = EnumSet.copyOf(artifactPermissions);
      readOnly.remove(PosixFilePermission.OWNER_WRITE);
      readOnly.remove(PosixFilePermission.GROUP_WRITE);
      readOnly.remove(PosixFilePermission.OTHERS_WRITE);
      Files.setPosixFilePermissions(artifactPath, readOnly);
      require(Files.getPosixFilePermissions(artifactPath).equals(readOnly),
          "fixture did not apply read-only session artifact permissions");
      if (Files.isWritable(artifactPath)) return false;
      boolean denied = false;
      try {
        Files.delete(readOnlyResidue);
      } catch (AccessDeniedException expected) {
        denied = true;
      }
      if (!denied) return false;
      require(Files.isRegularFile(readOnlyResidue) && Files.isRegularFile(artifactFile)
              && artifacts.getCachedBlockId(cachedArtifact).isDefined()
              && artifacts.cachedClassLoader().isDefined(),
          "permission fixture changed session resources before production cleanup");
      return true;
    }

    private void assertReadOnlyResidue() throws Exception {
      require(engine.isClosed() && field(session, "closeComplete").getBoolean(session)
              && registry(engine, "tasks").isEmpty() && registry(engine, "sessions").isEmpty(),
          "permission-denied artifact deletion prevented owner cleanup");
      require(Files.isRegularFile(readOnlyResidue) && !Files.isWritable(artifactPath),
          "cleanup did not retain the intentionally read-only owned directory residue");
      require(artifacts.getCachedBlockId(cachedArtifact).isEmpty()
              && artifacts.cachedClassLoader().isEmpty(),
          "directory deletion failure skipped cached artifact or classloader cleanup");
    }

    private void restoreArtifactDirectory() throws Exception {
      if (artifactPermissions == null) return;
      if (Files.exists(artifactPath)) {
        restoreArtifactPermissions();
        Utils.deleteRecursively(artifactPath.toFile());
      }
    }

    private void restoreArtifactPermissions() throws Exception {
      if (artifactPermissions != null && Files.exists(artifactPath)) {
        Files.setPosixFilePermissions(artifactPath, artifactPermissions);
      }
    }

    private void fetchOneAndAwaitWorker() throws Exception {
      NativeIpcBatch first = query.nextIpcBatch();
      require(first != null && first.rowCount() == 2, "query did not produce its first IPC batch");
      nextValue = checkValues(first, 0);
      Thread worker = (Thread) field(query, "worker").get(query);
      require(worker != null, "query did not start an execution worker");
      worker.join(TIMEOUT_MILLIS);
      require(!worker.isAlive() && field(query, "workerFinished").getBoolean(query),
          "query worker did not finish while output remained unread");
      spark.sparkContext().listenerBus().waitUntilEmpty(TIMEOUT_MILLIS);
      require(jobs.started.size() == 1 && jobs.ended.containsAll(jobs.started),
          "query did not finish its submitted Spark job");
      require(field(task, "executionStarted").getBoolean(task)
              && !field(task, "completed").getBoolean(task),
          "fixture did not retain a started, incomplete cursor Task");
      List<?> pending = (List<?>) field(query, "partitionBatches").get(query);
      require(pending != null && pending.size() == 1 && pending.get(0) instanceof Object[],
          "query did not retain its unread partition");
      Object[] batches = (Object[]) pending.get(0);
      require(batches.length > 1 && batches[0] == null && batches[1] != null
              && field(query, "nextBatch").getInt(query) == 1,
          "fixture consumed the remaining IPC output");
    }

    private void drainAfterHostStop() throws Exception {
      require(spark.sparkContext().isStopped()
              && field(query, "workerFinished").getBoolean(query)
              && !field(task, "completed").getBoolean(task),
          "buffered drain did not begin after host stop with a completed worker and open cursor");
      int batches = 1;
      NativeIpcBatch batch;
      while ((batch = query.nextIpcBatch()) != null) {
        require(batch.rowCount() == 2, "buffered IPC batch size changed after host stop");
        nextValue = checkValues(batch, nextValue);
        batches++;
      }
      require(nextValue == 20 && batches == 10 && query.nextIpcBatch() == null
              && field(task, "completed").getBoolean(task),
          "buffered drain lost rows, batches, stable EOF or cursor completion after host stop");
    }

    private void closeOwners() {
      query.close();
      task.close();
      session.close();
      engine.close();
    }

    private void assertCleared() throws Exception {
      require(engine.isClosed(), "engine factory did not close");
      require(registry(engine, "tasks").isEmpty() && registry(engine, "sessions").isEmpty()
              && registry(session, "queries").isEmpty()
              && registry(session, "activeTasks").isEmpty()
              && registry(unusedSession, "queries").isEmpty()
              && registry(unusedSession, "activeTasks").isEmpty()
              && field(unusedSession, "closeComplete").getBoolean(unusedSession),
          "stopped-context cleanup retained an owner registry entry");
      require(field(query, "partitionBatches").get(query) == null,
          "stopped-context cleanup retained unread IPC");
      Thread worker = (Thread) field(query, "worker").get(query);
      require(worker == null || !worker.isAlive(), "stopped-context cleanup retained its worker");
      require(!Files.exists(artifactFile) && !Files.exists(artifactPath),
          "stopped-context cleanup retained session artifact files");
      require(spark.sparkContext().addedFiles().get(sessionId).isEmpty()
              && spark.sparkContext().addedArchives().get(sessionId).isEmpty()
              && spark.sparkContext().addedJars().get(sessionId).isEmpty(),
          "stopped-context cleanup retained session resource registrations");
      require(artifacts.getCachedBlockId(cachedArtifact).isEmpty()
              && artifacts.jarsList().isEmpty() && artifacts.pythonIncludeList().isEmpty()
              && artifacts.sparkContextRelativePaths().isEmpty()
              && artifacts.cachedClassLoader().isEmpty(),
          "stopped-context cleanup retained cached artifacts, trackers or classloader");
    }

    @Override
    public void close() throws Exception {
      Throwable failure = null;
      if (server != null) failure = cleanup(failure, server::close);
      // A failed native close still owns the factory; do not bypass its drain contract.
      if (failure == null) {
        if (query != null) failure = cleanup(failure, query::close);
        if (task != null) failure = cleanup(failure, task::close);
        if (session != null) failure = cleanup(failure, session::close);
        failure = cleanup(failure, engine::close);
      }
      failure = cleanup(failure, () -> spark.sparkContext().removeSparkListener(jobs));
      if (failure instanceof Exception) throw (Exception) failure;
      if (failure != null) throw (Error) failure;
    }
  }

  private static final class Jobs extends SparkListener {
    private final String description;
    private final Set<Integer> started = ConcurrentHashMap.newKeySet();
    private final Set<Integer> ended = ConcurrentHashMap.newKeySet();

    private Jobs(String description) { this.description = description; }

    @Override
    public void onJobStart(SparkListenerJobStart event) {
      if (description.equals(event.properties().getProperty("spark.job.description"))) {
        started.add(event.jobId());
      }
    }

    @Override
    public void onJobEnd(SparkListenerJobEnd event) { ended.add(event.jobId()); }
  }

  private static long checkValues(NativeIpcBatch batch, long next) throws Exception {
    try (BufferAllocator allocator = ArrowUtils$.MODULE$.rootAllocator()
            .newChildAllocator("stopped-context-test", 0, Long.MAX_VALUE);
         ArrowStreamReader reader = new ArrowStreamReader(
             new ByteArrayInputStream(batch.data()), allocator)) {
      require(reader.loadNextBatch(), "IPC omitted its record batch");
      var root = reader.getVectorSchemaRoot();
      require(root.getRowCount() == batch.rowCount(), "IPC row count changed after host stop");
      for (int row = 0; row < root.getRowCount(); row++) {
        require(((Number) root.getVector(0).getObject(row)).longValue() == next++,
            "buffered drain lost, duplicated or reordered a row after host stop");
      }
      require(!reader.loadNextBatch(), "IPC contains multiple record batches");
    }
    return next;
  }
}
