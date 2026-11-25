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
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.regex.Pattern;

import org.apache.arrow.memory.OutOfMemoryException;
import org.apache.spark.SimpleFutureAction;
import org.apache.spark.SparkContext;
import org.apache.spark.SparkEnv;
import org.apache.spark.SparkException;
import org.apache.spark.SparkThrowable;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.analysis.UnresolvedExecuteImmediate;
import org.apache.spark.sql.catalyst.plans.logical.Command;
import org.apache.spark.sql.catalyst.plans.logical.InsertIntoDir;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.catalyst.plans.logical.ParsedStatement;
import org.apache.spark.sql.classic.Dataset;
import org.apache.spark.sql.connect.common.InvalidPlanInput;
import org.apache.spark.sql.connect.planner.SparkConnectPlanner;
import org.apache.spark.sql.connect.service.SessionHolder;
import org.apache.spark.sql.errors.QueryExecutionErrors$;
import org.apache.spark.sql.execution.CommandExecutionMode;
import org.apache.spark.sql.execution.QueryExecution;
import org.apache.spark.sql.execution.SQLExecution;
import org.apache.spark.sql.execution.SparkPlan;
import org.apache.spark.sql.execution.arrow.ArrowConverters$;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.CharType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.MapType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.UserDefinedType;
import org.apache.spark.sql.types.VarcharType;
import org.apache.spark.util.ThreadUtils;
import org.sparkproject.connect.protobuf.InvalidProtocolBufferException;
import org.sparkproject.connect.protobuf.Message;

import scala.Option;
import scala.Tuple2;
import scala.collection.immutable.Range$;
import scala.concurrent.duration.Duration;
import scala.reflect.ClassTag;
import scala.reflect.ClassTag$;
import scala.runtime.AbstractFunction0;
import scala.runtime.AbstractFunction1;
import scala.runtime.AbstractFunction2;
import scala.runtime.BoxedUnit;

/**
 * A query engine for the Rust-owned frontend. The JNI boundary passes Catalyst inputs and Arrow
 * batches, never service RPC requests, responses, observers, or operation registries.
 *
 * This intentionally targets the pinned Spark 4.2 assembly. SessionHolder is a planner resource
 * container only: its service-managed close() and SparkSession.close() are never called.
 *
 * Passing this engine to NativeEngineServer transfers its lifetime to that server. Native stop
 * closes it, and a failed native start can close it too; do not share or reuse it afterwards.
 * close() is idempotent, so callers may also close it while cleaning up a failed constructor.
 * The caller retains ownership of the original SparkSession and shared SparkContext.
 */
public final class SparkJvmEngine implements AutoCloseable {
  static final int MAX_IPC_BYTES = 64 * 1024 * 1024;
  private static final int MAX_BATCH_ROWS = 10000;
  private static final String SHUFFLE_FILE_CLEANUP =
      "spark.sql.classic.shuffleDependency.fileCleanup.enabled";
  private static final Set<String> RELATIONS = Set.of(
      "RANGE", "LOCAL_RELATION", "SQL", "PROJECT", "FILTER", "LIMIT", "OFFSET", "TAIL",
      "JOIN", "SORT", "AGGREGATE", "DEDUPLICATE", "SET_OP", "SUBQUERY_ALIAS", "TO_DF",
      "WITH_COLUMNS_RENAMED", "WITH_COLUMNS", "DROP", "REPARTITION",
      "REPARTITION_BY_EXPRESSION", "HINT", "READ");

  private final org.apache.spark.sql.classic.SparkSession base;
  private final SparkContext sparkContext;
  private final Pattern redactionPattern;
  private final SparkEnv sparkEnv;
  private final ClassLoader classLoader;
  private final AtomicBoolean closed = new AtomicBoolean();
  // NativeCleanupIsolationTest reflectively gates session removal here.
  // Keep its field lookup in sync with renames.
  private final Set<Session> sessions = ConcurrentHashMap.newKeySet();
  private final Set<Task> tasks = ConcurrentHashMap.newKeySet();
  private boolean closeComplete;
  private Throwable closeFailure;

  public SparkJvmEngine(SparkSession session) {
    if (!(Objects.requireNonNull(session, "session")
        instanceof org.apache.spark.sql.classic.SparkSession)) {
      throw new IllegalArgumentException("a classic Spark session is required");
    }
    base = (org.apache.spark.sql.classic.SparkSession) session;
    sparkContext = base.sparkContext();
    // SparkContext configuration is fixed after startup. Cache the host policy rather
    // than cloning the whole SparkConf for each Config RPC or reading session overrides.
    redactionPattern = sparkContext.getConf().get(
        org.apache.spark.internal.config.package$.MODULE$.SECRET_REDACTION_PATTERN()).pattern();
    sparkEnv = SparkEnv.get();
    ClassLoader current = Thread.currentThread().getContextClassLoader();
    classLoader = current == null ? getClass().getClassLoader() : current;
  }

  /** Allocate this control before invoking a blocking planner or cursor method. */
  public Task newTask(String description) {
    ensureOpen();
    Task task = new Task(Objects.requireNonNull(description, "description"));
    tasks.add(task);
    if (closed.get()) task.cancel();
    return task;
  }

  public Session openSession(String userId, String sessionId, Task task) {
    Objects.requireNonNull(userId, "userId");
    Objects.requireNonNull(sessionId, "sessionId");
    return task.run(null, () -> {
      ensureOpen();
      Session session = new Session(userId, sessionId);
      sessions.add(session);
      if (closed.get() || task.isCancelled()) {
        session.close();
        task.check();
        ensureOpen();
      }
      task.complete();
      return session;
    });
  }

  boolean isClosed() {
    return closed.get();
  }

  private void ensureOpen() {
    if (closed.get()) throw new EngineException(9, "JVM engine is closed");
  }

  /**
   * Idempotently release this engine's tasks and sessions without stopping the shared
   * SparkContext. Once passed to NativeEngineServer, drain that server before calling this.
   * A partial cleanup failure remains visible on later closes without replaying its side effects.
   */
  @Override
  public synchronized void close() {
    throwFailure(closeFailure);
    if (closeComplete) return;
    closed.set(true);
    Throwable failure = null;
    for (Task task : tasks) {
      try { task.close(); } catch (Throwable error) { failure = recordFailure(failure, error); }
    }
    for (Session session : sessions) {
      try { session.close(); } catch (Throwable error) { failure = recordFailure(failure, error); }
    }
    closeFailure = failure;
    throwFailure(failure);
    closeComplete = true;
  }

  @FunctionalInterface
  private interface Action<T> { T run() throws Exception; }

  /** A small stable error surface for the native adapter's tonic Status mapping. */
  public static final class EngineException extends RuntimeException {
    private static final long serialVersionUID = 1L;
    private final int code;

    public EngineException(int code, String message) {
      super(message);
      this.code = code;
    }

    public EngineException(int code, String message, Throwable cause) {
      super(message, cause);
      this.code = code;
    }

    public int code() { return code; }
  }

  private static EngineException engineError(Exception error) {
    if (error instanceof EngineException) return (EngineException) error;
    if (error instanceof OutOfMemoryException) {
      return arrowAllocationFailure((OutOfMemoryException) error);
    }
    if (error instanceof SparkException) {
      // Executor conversion errors arrive inside Spark's task-failure wrapper. Preserve only
      // resource errors from that cause chain; an arbitrary nested argument error is not a
      // client validation failure.
      Set<Throwable> seen = Collections.newSetFromMap(new IdentityHashMap<>());
      for (Throwable cause = error.getCause(); cause != null && seen.add(cause);
          cause = cause.getCause()) {
        if (cause instanceof EngineException && ((EngineException) cause).code() == 8) {
          return new EngineException(8, cause.getMessage(), error);
        }
      }
    }
    int code = 13;
    // The pinned planner uses InvalidPlanInput for malformed client plans even though its
    // legacy message-only constructor labels them INTERNAL_ERROR. Its concrete type is the
    // input-validation contract; arbitrary Spark internal exceptions still map to INTERNAL.
    if (error instanceof InvalidProtocolBufferException || error instanceof InvalidPlanInput) {
      code = 3;
    } else if (error instanceof SparkThrowable && !((SparkThrowable) error).isInternalError()) {
      if (error instanceof NoSuchElementException) code = 5;
      else if (error instanceof AnalysisException || error instanceof IllegalArgumentException) code = 3;
    }
    return new EngineException(code, error.getMessage(), error);
  }

  private static EngineException interrupted(Throwable error) {
    return new EngineException(1, "JVM engine operation was interrupted", error);
  }

  static EngineException arrowAllocationFailure(OutOfMemoryException error) {
    return new EngineException(8, "JVM Arrow allocation failed: " + error.getMessage(), error);
  }

  /** Cancellation is independent of a query/session lock and remains usable during preparation. */
  public final class Task implements AutoCloseable {
    private final String jobGroup = "spark-connect-engine-" + UUID.randomUUID();
    private final String description;
    // Partition, batch and task-step checks observe cross-thread cancellation.
    // Compound task transitions remain locked.
    private volatile boolean cancelled;
    private boolean completed;
    private boolean executionStarted;
    private Thread running;
    private final Object cancellation = new Object();
    private Throwable cancellationFailure;

    private Task(String description) { this.description = description; }

    public void cancel() {
      synchronized (cancellation) {
        throwFailure(cancellationFailure);
        try {
          boolean cancelJobs;
          synchronized (this) {
            if (completed || cancelled) return;
            cancelled = true;
            if (running != null) running.interrupt();
            cancelJobs = executionStarted || running != null;
          }
          // Active preparation/analysis can submit Spark jobs too (for example, parallel file
          // listing). An idle task that never began fetching has no scheduler work to cancel.
          if (cancelJobs) sparkContext.cancelJobGroupAndFutureJobs(jobGroup);
        } catch (Throwable error) {
          cancellationFailure = error;
          throw error;
        }
      }
    }

    private boolean isCancelled() { return cancelled; }

    private void check() {
      if (cancelled) throw new EngineException(1, "JVM engine operation was cancelled");
    }

    private synchronized void complete() { completed = true; }

    private synchronized void beginExecution() {
      check();
      executionStarted = true;
    }

    private EngineException taskError(Exception error) {
      if (isCancelled() && !(error instanceof EngineException)) {
        return new EngineException(1, "JVM engine operation was cancelled", error);
      }
      return engineError(error);
    }

    private <T> T run(Session session, Action<T> action) {
      Thread thread = Thread.currentThread();
      boolean previousInterrupted;
      synchronized (this) {
        check();
        if (running != null) throw new EngineException(9, "task is already executing");
        previousInterrupted = Thread.interrupted();
        running = thread;
      }
      try {
        // Capture can fail too (for example, a caller's context-class-loader accessor). Keep
        // the running mark covered before any caller/Spark code and until restoration ends.
        Properties previousProperties = sparkContext.getLocalProperties();
        ClassLoader previousLoader = thread.getContextClassLoader();
        SparkEnv previousEnv = SparkEnv.get();
        Option<SparkSession> previousSession = SparkSession.getActiveSession();
        try {
          if (session != null) session.activeTasks.add(this);
          SparkEnv.set(sparkEnv);
          thread.setContextClassLoader(classLoader);
          sparkContext.setLocalProperties(new Properties());
          sparkContext.setJobGroup(jobGroup, description, true);
          SparkSession.setActiveSession(session == null ? base : session.spark);
          check();
          if (session == null) return action.run();
          session.ensureOpen();
          return session.holder.withSession(
              new AbstractFunction1<org.apache.spark.sql.classic.SparkSession, T>() {
                @Override
                public T apply(org.apache.spark.sql.classic.SparkSession ignored) {
                  try {
                    check();
                    return action.run();
                  } catch (InterruptedException error) {
                    throw interrupted(error);
                  } catch (Exception error) {
                    throw taskError(error);
                  }
                }
              });
        } finally {
          if (session != null) session.activeTasks.remove(this);
          if (previousSession.isDefined()) SparkSession.setActiveSession(previousSession.get());
          else SparkSession.clearActiveSession();
          sparkContext.setLocalProperties(previousProperties);
          thread.setContextClassLoader(previousLoader);
          SparkEnv.set(previousEnv);
        }
      } catch (InterruptedException error) {
        throw interrupted(error);
      } catch (Exception error) {
        throw taskError(error);
      } finally {
        synchronized (this) { running = null; }
        Thread.interrupted();
        if (previousInterrupted) thread.interrupt();
      }
    }

    @Override
    public void close() {
      cancel();
      tasks.remove(this);
    }
  }

  /** Independent Catalyst catalog/configuration; identity and wire lifetime remain in Rust. */
  public final class Session implements AutoCloseable {
    private final org.apache.spark.sql.classic.SparkSession spark;
    private final SessionHolder holder;
    private final ReentrantReadWriteLock catalog = new ReentrantReadWriteLock(true);
    private final AtomicBoolean released = new AtomicBoolean();
    // NativeCleanupIsolationTest reflectively gates query removal here.
    // Keep its field lookup in sync with renames.
    private final Set<Query> queries = ConcurrentHashMap.newKeySet();
    private final Set<Task> activeTasks = ConcurrentHashMap.newKeySet();
    private boolean closeComplete;
    private Throwable closeFailure;

    private Session(String userId, String sessionId) {
      spark = base.newSession();
      try {
        holder = new SessionHolder(userId, sessionId, spark);
      } catch (Throwable error) {
        // Never close SparkSession: its clone shares the caller's SparkContext. Use the same
        // owned-resource cleanup as a published session, retaining any partial cleanup failure
        // for engine close even though this constructor cannot return a handle to JNI.
        abandonUnpublished(error);
        throw error;
      }
    }

    // Package-local so the unpublished ownership path can be exercised without breaking a
    // Spark constructor. This cleanup deliberately never accesses the uninitialized holder.
    void abandonUnpublished(Throwable error) {
      sessions.add(this);
      try { close(); } catch (Throwable cleanup) { recordFailure(error, cleanup); }
    }

    private void ensureOpen() {
      SparkJvmEngine.this.ensureOpen();
      if (released.get()) throw new EngineException(9, "JVM engine session is closed");
    }

    public Query prepare(byte[] relationBytes, Task task) {
      Objects.requireNonNull(relationBytes, "relationBytes");
      return task.run(this, () -> {
        // Preparation only analyzes read-only plans and registers a concurrent query handle.
        // Share admission with metadata/config readers; config mutation and close stay exclusive.
        catalog.readLock().lockInterruptibly();
        try {
          Query query = new Query(this, analyzedFrame(relationBytes, task), task);
          queries.add(query);
          if (released.get()) {
            query.close();
            ensureOpen();
          }
          if (task.isCancelled()) {
            query.close();
            task.check();
          }
          return query;
        } finally {
          catalog.readLock().unlock();
        }
      });
    }

    /**
     * Analysis modes 0/1/2 are schema/local/streaming. A strict boolean byte precedes the same
     * schema IPC used by prepare; schema and streaming modes return a zero prefix.
     */
    public byte[] analyze(int opcode, byte[] relationBytes, Task task) {
      Objects.requireNonNull(relationBytes, "relationBytes");
      return task.run(this, () -> {
        if (opcode < 0 || opcode > 2) {
          throw new EngineException(12, "unsupported analysis opcode: " + opcode);
        }
        catalog.readLock().lockInterruptibly();
        try {
          Dataset<Row> frame = analyzedFrame(relationBytes, task);
          PlanMetadata metadata = new PlanMetadata(this, frame);
          byte[] output = new byte[metadata.schemaIpc.length + 1];
          output[0] = (byte) (opcode == 1 && metadata.local ? 1 : 0);
          System.arraycopy(metadata.schemaIpc, 0, output, 1, metadata.schemaIpc.length);
          synchronized (task) {
            task.check();
            ensureOpen();
            task.complete();
          }
          return output;
        } finally {
          catalog.readLock().unlock();
        }
      });
    }

    // The caller holds a catalog read/write lock inside Task.run and its session/artifact scopes.
    private Dataset<Row> analyzedFrame(byte[] relationBytes, Task task) throws IOException {
      task.check();
      ensureOpen();
      if (Boolean.parseBoolean(spark.conf().get(SHUFFLE_FILE_CLEANUP))) {
        throw new EngineException(12, SHUFFLE_FILE_CLEANUP
            + " has not been validated by the JVM engine");
      }
      Relation relation = Relation.parseFrom(relationBytes);
      validateRelations(relation);
      LogicalPlan logical = new SparkConnectPlanner(holder, Option.empty())
          .transformRelation(relation);
      rejectCommands(logical);
      // Dataset.ofRows uses ALL and may eagerly execute a write produced by analysis.
      // Inspect with SKIP first, then keep that same execution through the Dataset constructor.
      QueryExecution execution = spark.sessionState().executePlan(logical, CommandExecutionMode.SKIP());
      execution.assertAnalyzed();
      rejectCommands(execution.analyzed());
      Dataset<Row> frame = new Dataset<>(execution, Encoders.row(execution.analyzed().schema()));
      if (frame.isStreaming()) {
        throw new EngineException(12, "streaming queries are unsupported");
      }
      return frame;
    }

    private void rejectCommands(LogicalPlan logical) {
      if (logical.exists(new AbstractFunction1<LogicalPlan, Object>() {
        @Override
        public Object apply(LogicalPlan node) {
          // InsertIntoStatement extends ParsedStatement, and InsertIntoDir extends UnaryNode;
          // neither is a Command until analysis. SQL scripts and dynamic SQL can execute
          // their statements during analysis itself, before the post-analysis guard runs.
          // CompoundBody already implements Command in the pinned assembly. Use actual types
          // so a Spark internal API rename fails compilation instead of weakening the guard.
          return node instanceof Command || node instanceof ParsedStatement
              || node instanceof InsertIntoDir || node instanceof UnresolvedExecuteImmediate;
        }
      })) {
        throw new EngineException(12, "SQL commands and external writes are unsupported");
      }
    }

    /**
     * Internal typed fields: non-negative big-endian signed 32-bit pair count, then UTF-8 key
     * length/data and nullable value length/data (signed -1 means absent). No Spark Connect
     * request/response envelopes cross here.
     */
    public byte[] config(int opcode, byte[] fields, Task task) {
      List<Pair> input = decodePairs(fields);
      return task.run(this, () -> {
        Lock access = configLock(opcode);
        access.lockInterruptibly();
        try {
          task.check();
          ensureOpen();
          List<Pair> output = new ArrayList<>();
          switch (opcode) {
            case 1:
              for (Pair pair : input) {
                String value = spark.conf().get(pair.key);
                if (isRedacted(redactionPattern, pair.key, value)) {
                  throw QueryExecutionErrors$.MODULE$.sqlConfigNotFoundError(pair.key);
                }
                output.add(new Pair(pair.key, value));
              }
              break;
            case 2:
              for (Pair pair : input) {
                Option<String> stored = spark.conf().getOption(pair.key);
                String value = stored.isDefined() ? stored.get() : null;
                output.add(new Pair(pair.key,
                    isRedacted(redactionPattern, pair.key, value) ? null : value));
              }
              break;
            case 3:
              for (Pair pair : input) {
                String value = spark.conf().get(pair.key, pair.value);
                output.add(new Pair(pair.key,
                    isRedacted(redactionPattern, pair.key, value) ? pair.value : value));
              }
              break;
            case 4:
              if (input.size() > 1) throw new EngineException(3, "getAll accepts one prefix");
              String prefix = input.isEmpty() ? "" : input.get(0).key;
              spark.conf().getAllAsJava().entrySet().stream()
                  .filter(entry -> !isRedacted(redactionPattern, entry.getKey(), entry.getValue()))
                  .filter(entry -> entry.getKey().startsWith(prefix))
                  .sorted(Map.Entry.comparingByKey())
                  .forEach(entry -> output.add(new Pair(
                      entry.getKey().substring(prefix.length()), entry.getValue())));
              break;
            case 5:
              for (Pair pair : input) {
                output.add(new Pair(pair.key, Boolean.toString(spark.conf().isModifiable(pair.key))));
              }
              break;
            case 6:
            case 7:
              applyConfig(opcode, input, task);
              break;
            case 8:
              // A separate result preserves setter-error provenance. JNI/task/admission
              // failures still fail the call rather than becoming silent Set warnings.
              if (input.size() != 1) throw new EngineException(3, "silent set accepts one key");
              String warning = applyConfig(opcode, input, task);
              if (warning != null) output.add(new Pair(input.get(0).key, warning));
              break;
            default:
              throw new EngineException(12, "unsupported config opcode: " + opcode);
          }
          task.complete();
          return encodePairs(output);
        } finally {
          access.unlock();
        }
      });
    }

    private Lock configLock(int opcode) {
      // JNI contract: 1..5 are reads; 6..8 mutate. Unknown opcodes cannot come from Rust;
      // conservatively keep them exclusive until the task's normal validation rejects them.
      switch (opcode) {
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
          return catalog.readLock();
        default:
          return catalog.writeLock();
      }
    }

    private String applyConfig(int opcode, List<Pair> input, Task task) {
      boolean silent = opcode == 8;
      Map<String, String> previous = new LinkedHashMap<>();
      Set<String> attempted = new LinkedHashSet<>();
      Map<String, String> explicit = spark.conf().getAllAsJava();
      for (Pair pair : input) {
        if (opcode == 6 && pair.value == null) throw new EngineException(3, "missing config value");
        if (!previous.containsKey(pair.key)) previous.put(pair.key, explicit.get(pair.key));
      }
      task.check();
      Throwable setterFailure = null;
      try {
        for (Pair pair : input) {
          task.check();
          // A setter can modify state before throwing, so record its key before calling it.
          attempted.add(pair.key);
          // RuntimeConfig accepts unregistered spark.* keys even when isModifiable is false;
          // let its setters reject static/core entries and invalid registered values. Only this
          // input-validation boundary maps an unstructured IllegalArgumentException to a client
          // error; the same exception from arbitrary Spark execution is INTERNAL.
          try {
            if (opcode != 7) spark.conf().set(pair.key, pair.value);
            else spark.conf().unset(pair.key);
          } catch (Throwable error) {
            setterFailure = error;
            if (!silent && error instanceof IllegalArgumentException) {
              if (!(error instanceof SparkThrowable) || !((SparkThrowable) error).isInternalError()) {
                throw new EngineException(3, error.getMessage(), error);
              }
            }
            throw error;
          }
        }
        synchronized (task) {
          task.check();
          task.complete();
        }
      } catch (Throwable error) {
        Map<String, String> current = null;
        boolean restored = true;
        try { current = spark.conf().getAllAsJava(); } catch (Throwable inspection) {
          recordFailure(error, inspection);
          restored = false;
        }
        for (String key : attempted) {
          String value = previous.get(key);
          // Skip rejected static entries and other unchanged keys. If inspection failed,
          // conservatively restore every attempted key; unvisited keys never need rollback.
          if (current != null && Objects.equals(value, current.get(key))) continue;
          try {
            // Restore the accepted snapshot independently of RuntimeConfig restrictions that
            // this batch may have changed, such as rejection of Spark core configuration keys.
            if (value == null) spark.sessionState().conf().unsetConf(key);
            else spark.sessionState().conf().setConfString(key, value);
          } catch (Throwable rollback) {
            if (rollback != error) error.addSuppressed(rollback);
            restored = false;
          }
        }
        if (silent && error == setterFailure && restored) {
          // Match Spark's silent Set diagnostics without retaining rejected values/messages.
          String reason = error instanceof SparkThrowable
              ? ((SparkThrowable) error).getCondition() : null;
          if (reason == null) reason = error.getClass().getName();
          synchronized (task) {
            task.check();
            if (error instanceof InterruptedException || Thread.currentThread().isInterrupted()) {
              throw interrupted(error);
            }
            task.complete();
          }
          return reason;
        }
        throw error;
      }
      return null;
    }

    @Override
    public synchronized void close() {
      throwFailure(closeFailure);
      if (closeComplete) return;
      released.set(true);
      Throwable failure = null;
      for (Task task : activeTasks) {
        try { task.cancel(); } catch (Throwable error) { failure = recordFailure(failure, error); }
      }
      for (Query query : queries) {
        try { query.close(); } catch (Throwable error) { failure = recordFailure(failure, error); }
      }
      // A partial Spark cleanup is not generally replayable (for example, artifact cache
      // cleanup decrements reference counts). Preserve the failure and its owners instead.
      if (failure == null) {
        catalog.writeLock().lock();
        try {
          // These resources belong to this newSession only. Never close SessionHolder or Spark.
          try {
            spark.sessionState().catalog().clearTempTables();
          } catch (Throwable error) {
            failure = recordFailure(failure, error);
          }
          try {
            spark.artifactManager().close();
          } catch (Throwable error) {
            failure = recordFailure(failure, error);
          }
        } finally {
          catalog.writeLock().unlock();
        }
      }
      closeFailure = failure;
      throwFailure(failure);
      sessions.remove(this);
      closeComplete = true;
    }
  }

  /** Shared metadata evaluation order for analysis and executable query preparation. */
  private static final class PlanMetadata {
    private final StructType schema;
    private final String timeZone;
    private final boolean largeVarTypes;
    private final int maxRows;
    private final boolean local;
    private final byte[] schemaIpc;

    private PlanMetadata(Session session, Dataset<Row> frame) {
      schema = frame.schema();
      validateResultType(schema);
      timeZone = session.spark.sessionState().conf().sessionLocalTimeZone();
      largeVarTypes = session.spark.sessionState().conf().arrowUseLargeVarTypes();
      int configured = session.spark.sessionState().conf().arrowMaxRecordsPerBatch();
      maxRows = configured > 0 ? configured : MAX_BATCH_ROWS;
      local = frame.isLocal();
      schemaIpc = createSchemaIpc(schema, timeZone, largeVarTypes);
    }
  }

  private static final class CollectArrowPartition extends AbstractFunction1<
      scala.collection.Iterator<Tuple2<byte[], Long>>, Tuple2<byte[], Long>[]>
      implements java.io.Serializable {
    private static final long serialVersionUID = 1L;

    @Override
    @SuppressWarnings("unchecked")
    public Tuple2<byte[], Long>[] apply(scala.collection.Iterator<Tuple2<byte[], Long>> rows) {
      List<Tuple2<byte[], Long>> batches = new ArrayList<>();
      while (rows.hasNext()) batches.add(rows.next());
      // Spark registers tuple arrays with Kryo, including registrationRequired mode.
      return batches.toArray((Tuple2<byte[], Long>[]) new Tuple2<?, ?>[0]);
    }
  }

  /** An analyzed Catalyst query and a lazy pull cursor. Only one caller may fetch at a time. */
  public final class Query implements AutoCloseable {
    private final Session session;
    private final QueryExecution execution;
    private final ArrowPartitionConverter converter;
    private final Task task;
    private final byte[] schemaIpc;
    private final boolean local;
    private final ReentrantLock cursor = new ReentrantLock();
    // Scheduler callbacks never take the cursor lock or run JNI/client work.
    private final Object results = new Object();
    private boolean released;
    private Throwable closeFailure;
    private boolean ended;
    private Throwable pullFailure;
    private SparkPlan plan;
    private boolean planCleanupAttempted;
    private Thread worker;
    private boolean workerFinished;
    private List<Tuple2<byte[], Long>[]> partitionBatches;
    private int nextPartition;
    private int nextBatch;

    private Query(Session session, Dataset<Row> frame, Task task) {
      this.session = session;
      this.execution = frame.queryExecution();
      PlanMetadata metadata = new PlanMetadata(session, frame);
      this.converter = new ArrowPartitionConverter(
          metadata.schema, metadata.timeZone, metadata.largeVarTypes, metadata.maxRows);
      this.task = task;
      this.local = metadata.local;
      this.schemaIpc = metadata.schemaIpc;
    }

    public byte[] schemaIpc() { return schemaIpc.clone(); }
    public boolean isLocal() { return local; }
    public boolean isStreaming() { return false; }

    private void runWorker() {
      try {
        task.run(session, () -> {
          task.beginExecution();
          // Keep one execution/session scope alive until every submitted partition finishes.
          SQLExecution.withNewExecutionId(execution, Option.apply("spark-connect-engine-arrow"),
              new AbstractFunction0<Object>() {
                @Override
                public Object apply() {
                  plan = execution.executedPlan();
                  ClassTag<Tuple2<byte[], Long>> batchesClass = ClassTag$.MODULE$.apply(Tuple2.class);
                  RDD<Tuple2<byte[], Long>> batches =
                      plan.execute().mapPartitionsInternal(converter, false, batchesClass);
                  int count = batches.getNumPartitions();
                  synchronized (results) {
                    throwFailure(pullFailure);
                    task.check();
                    if (released) throw cursorClosed();
                    partitionBatches = new ArrayList<>(Collections.nCopies(count, null));
                  }
                  if (count == 0) return null;
                  SimpleFutureAction<BoxedUnit> action = sparkContext.submitJob(
                      batches, new CollectArrowPartition(), Range$.MODULE$.apply(0, count),
                      new AbstractFunction2<Object, Tuple2<byte[], Long>[], BoxedUnit>() {
                        @Override
                        public BoxedUnit apply(Object partition, Tuple2<byte[], Long>[] fetched) {
                          synchronized (results) {
                            if (!released && pullFailure == null) {
                              partitionBatches.set((Integer) partition, fetched);
                              results.notifyAll();
                            }
                          }
                          return BoxedUnit.UNIT;
                        }
                      },
                      new AbstractFunction0<BoxedUnit>() {
                        @Override
                        public BoxedUnit apply() { return BoxedUnit.UNIT; }
                      });
                  awaitJob(action);
                  return null;
                }
              });
          return null;
        });
      } catch (Throwable error) {
        synchronized (results) {
          // After release, close retains worker/plan ownership; there is no result recipient.
          if (!released && pullFailure == null) fail(error);
        }
      } finally {
        synchronized (results) {
          workerFinished = true;
          results.notifyAll();
        }
      }
    }

    private void awaitJob(SimpleFutureAction<BoxedUnit> action) {
      EngineException interruption = null;
      // Cancellation interrupts the owner, but scope/plan cleanup must await the job too.
      for (;;) {
        try {
          ThreadUtils.awaitReady(action, Duration.Inf());
          break;
        } catch (Exception error) {
          if (!(error instanceof InterruptedException)) throw task.taskError(error);
          if (interruption == null) interruption = interrupted(error);
          try { action.cancel(); } catch (Throwable cancel) { recordFailure(interruption, cancel); }
        }
      }
      if (interruption != null) throw interruption;
      // Inspect the completed Try to retain the executor's original failure.
      action.value().get().get();
      task.check();
    }

    private EngineException cursorClosed() {
      return new EngineException(9, "query cursor is closed");
    }

    // Caller holds results. The failure guard makes later scheduler callbacks discard data.
    private void fail(Throwable error) {
      pullFailure = error instanceof InterruptedException
          ? interrupted(error)
          : error instanceof Exception ? task.taskError((Exception) error) : error;
      partitionBatches = null;
      results.notifyAll();
    }

    /** Pull a self-contained IPC batch produced by Spark's executor-side converter. */
    public NativeIpcBatch nextIpcBatch() {
      boolean previousInterrupted = Thread.interrupted();
      try {
        return pullIpcBatch();
      } finally {
        if (previousInterrupted) Thread.currentThread().interrupt();
      }
    }

    private NativeIpcBatch pullIpcBatch() {
      try {
        cursor.lockInterruptibly();
      } catch (InterruptedException error) {
        throw interrupted(error);
      }
      try {
        Throwable failure;
        synchronized (results) {
          throwFailure(pullFailure);
          if (released) throw cursorClosed();
          task.check();
          session.ensureOpen();
          if (ended) return null;
          try {
            if (worker == null) {
              worker = new Thread(this::runWorker, task.jobGroup + "-worker");
              worker.setDaemon(true);
              worker.start();
            }
            for (;;) {
              throwFailure(pullFailure);
              task.check();
              if (released) throw cursorClosed();
              if (partitionBatches != null && nextPartition < partitionBatches.size()) {
                Tuple2<byte[], Long>[] batches = partitionBatches.get(nextPartition);
                if (batches != null) {
                  if (nextBatch == batches.length) {
                    partitionBatches.set(nextPartition++, null);
                    nextBatch = 0;
                    continue;
                  }
                  Tuple2<byte[], Long> batch = batches[nextBatch];
                  batches[nextBatch++] = null;
                  return new NativeIpcBatch(batch._1(), batch._2());
                }
              }
              if (workerFinished) {
                if (partitionBatches != null && nextPartition < partitionBatches.size()) {
                  throw new EngineException(13, "Spark job completed without a partition result");
                }
                partitionBatches = null;
                ended = true;
                task.complete();
                return null;
              }
              results.wait();
            }
          } catch (Throwable error) {
            if (pullFailure == null) fail(error);
            failure = pullFailure;
          }
        }
        try { task.cancel(); } catch (Throwable cancel) { recordFailure(failure, cancel); }
        throwFailure(failure);
        return null;
      } finally {
        cursor.unlock();
      }
    }

    @Override
    public void close() {
      Throwable failure = null;
      try { task.close(); } catch (Throwable error) { failure = error; }
      Thread executionWorker;
      synchronized (results) {
        released = true;
        partitionBatches = null;
        executionWorker = worker;
        results.notifyAll();
      }
      boolean restoreInterrupt = Thread.interrupted();
      try {
        // Failed cancellation retains the owner instead of waiting on uncancelled work.
        while (failure == null && executionWorker != null && executionWorker.isAlive()) {
          try { executionWorker.join(); } catch (InterruptedException error) { restoreInterrupt = true; }
        }
        cursor.lock();
        try {
          // The worker has restored its execution/session scope before plan cleanup.
          if ((executionWorker == null || !executionWorker.isAlive()) && !planCleanupAttempted) {
            planCleanupAttempted = true;
            try {
              if (plan != null) plan.cleanupResources();
            } catch (Throwable error) {
              failure = recordFailure(failure, error);
            }
          }
          closeFailure = failure == null ? closeFailure : recordFailure(closeFailure, failure);
          throwFailure(closeFailure);
          session.queries.remove(this);
        } finally {
          cursor.unlock();
        }
      } finally {
        if (restoreInterrupt) Thread.currentThread().interrupt();
      }
    }
  }

  private static byte[] createSchemaIpc(StructType schema, String timeZone, boolean largeVarTypes) {
    return checkedIpc(ArrowConverters$.MODULE$.createEmptyArrowBatch(
        schema, timeZone, false, largeVarTypes));
  }

  // Package-local so tests can cover exact Spark types that earlier Spark planning can normalize.
  static void validateResultType(DataType type) {
    validateResultType(type, false);
  }

  private static void validateResultType(DataType type, boolean nested) {
    if (type instanceof UserDefinedType || type instanceof CharType || type instanceof VarcharType
        || (type instanceof StringType && ((StringType) type).collationId()
            != ((StringType) DataTypes.StringType).collationId())) {
      throw new EngineException(12, "result type cannot be preserved by Arrow schema transport: "
          + type.catalogString());
    }
    if (type instanceof StructType) {
      StructType original = (StructType) type;
      StructField[] fields = org.apache.spark.sql.catalyst.util.package$.MODULE$
          .removeInternalMetadata(original).fields();
      Set<String> names = nested ? new HashSet<>() : Collections.emptySet();
      for (int i = 0; i < fields.length; i++) {
        StructField field = fields[i];
        if (nested && !names.add(field.name())) {
          throw new EngineException(12,
              "duplicate nested field names cannot be preserved by Arrow schema transport: "
                  + field.name());
        }
        // CHAR/VARCHAR's raw type annotation is semantic even though its name is internal.
        if (original.fields()[i].metadata().contains("__CHAR_VARCHAR_TYPE_STRING")) {
          throw new EngineException(12,
              "CHAR/VARCHAR result metadata cannot be preserved by Arrow schema transport: "
                  + field.name());
        }
        if (!field.metadata().isEmpty()) {
          throw new EngineException(12,
              "result field metadata cannot be preserved by Arrow schema transport: " + field.name());
        }
        validateResultType(field.dataType(), true);
      }
    } else if (type instanceof ArrayType) {
      validateResultType(((ArrayType) type).elementType(), true);
    } else if (type instanceof MapType) {
      validateResultType(((MapType) type).keyType(), true);
      validateResultType(((MapType) type).valueType(), true);
    }
  }

  static byte[] checkedIpc(byte[] bytes) {
    if (bytes.length > MAX_IPC_BYTES) throw new EngineException(8, "Arrow batch exceeds 64 MiB");
    return bytes;
  }

  private static Throwable recordFailure(Throwable previous, Throwable error) {
    if (previous == null) return error;
    if (previous != error) previous.addSuppressed(error);
    return previous;
  }

  private static void throwFailure(Throwable failure) {
    if (failure instanceof Error) throw (Error) failure;
    if (failure instanceof RuntimeException) throw (RuntimeException) failure;
    if (failure != null) throw new EngineException(13, "JVM engine cleanup failed", failure);
  }

  private static void validateRelations(Message message) {
    if (message instanceof Relation) {
      Relation relation = (Relation) message;
      String kind = relation.getRelTypeCase().name();
      if (!RELATIONS.contains(kind)) {
        throw new EngineException(12, "unsupported JVM engine relation: " + kind);
      }
      if (relation.hasRead() && !relation.getRead().hasNamedTable()) {
        throw new EngineException(12, "only named-table reads are supported in this prototype");
      }
    }
    String type = message.getDescriptorForType().getFullName();
    if (type.endsWith("CommonInlineUserDefinedFunction")
        || type.endsWith("CommonInlineUserDefinedTableFunction")
        || type.endsWith("TypedAggregateExpression") || type.endsWith("ScalarScalaUDF")) {
      throw new EngineException(12, "user-defined functions are unsupported in this prototype");
    }
    for (Object field : message.getAllFields().values()) {
      if (field instanceof Message) validateRelations((Message) field);
      else if (field instanceof List<?>) {
        for (Object item : (List<?>) field) {
          if (item instanceof Message) validateRelations((Message) item);
        }
      }
    }
  }

  private static final class Pair {
    private final String key;
    private final String value;
    private Pair(String key, String value) { this.key = key; this.value = value; }
  }

  private static boolean isRedacted(Pattern pattern, String key, String value) {
    return pattern.matcher(key).find() || value != null && pattern.matcher(value).find();
  }

  private static List<Pair> decodePairs(byte[] bytes) {
    Objects.requireNonNull(bytes, "fields");
    if (bytes.length > MAX_IPC_BYTES) throw new EngineException(8, "config fields exceed 64 MiB");
    try (DataInputStream input = new DataInputStream(new ByteArrayInputStream(bytes))) {
      int count = input.readInt();
      if (count < 0 || count > input.available() / 8) {
        throw new EngineException(3, "invalid config field count");
      }
      List<Pair> result = new ArrayList<>(count);
      for (int i = 0; i < count; i++) result.add(new Pair(readString(input, false), readString(input, true)));
      if (input.available() != 0) throw new EngineException(3, "trailing config field bytes");
      return result;
    } catch (IOException error) {
      throw new EngineException(3, "invalid config fields", error);
    }
  }

  private static String readString(DataInputStream input, boolean nullable) throws IOException {
    int len = input.readInt();
    if (len == -1 && nullable) return null;
    if (len < 0 || len > input.available()) throw new EngineException(3, "invalid config string length");
    byte[] bytes = input.readNBytes(len);
    try {
      return StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
          .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(bytes)).toString();
    } catch (CharacterCodingException error) {
      throw new EngineException(3, "invalid config UTF-8", error);
    }
  }

  private static byte[] encodePairs(List<Pair> pairs) throws IOException {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    DataOutputStream output = new DataOutputStream(bytes);
    output.writeInt(pairs.size());
    for (Pair pair : pairs) {
      writeString(output, pair.key);
      writeString(output, pair.value);
    }
    return bytes.toByteArray();
  }

  private static void writeString(DataOutputStream output, String value) throws IOException {
    if (value == null) output.writeInt(-1);
    else {
      byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
      output.writeInt(bytes.length);
      output.write(bytes);
    }
  }
}
