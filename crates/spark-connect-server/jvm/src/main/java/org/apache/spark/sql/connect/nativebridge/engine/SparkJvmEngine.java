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
import java.util.EnumSet;
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
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Consumer;
import java.util.regex.Pattern;

import org.apache.arrow.memory.OutOfMemoryException;
import org.apache.spark.SimpleFutureAction;
import org.apache.spark.SparkContext;
import org.apache.spark.SparkException;
import org.apache.spark.SparkThrowable;
import org.apache.spark.connect.proto.CommonInlineUserDefinedFunction;
import org.apache.spark.connect.proto.CommonInlineUserDefinedTableFunction;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.connect.proto.Relation.RelTypeCase;
import org.apache.spark.connect.proto.ScalarScalaUDF;
import org.apache.spark.connect.proto.TypedAggregateExpression;
import org.apache.spark.rdd.RDD;
import org.apache.spark.rpc.RpcEnvFileServer;
import org.apache.spark.scheduler.DAGScheduler;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.artifact.ArtifactManager;
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
import org.apache.spark.util.Utils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sparkproject.connect.protobuf.InvalidProtocolBufferException;
import org.sparkproject.connect.protobuf.Message;

import scala.Option;
import scala.Tuple2;
import scala.jdk.javaapi.CollectionConverters;
import scala.reflect.ClassTag;
import scala.reflect.ClassTag$;
import scala.runtime.AbstractFunction0;
import scala.runtime.AbstractFunction1;
import scala.runtime.AbstractFunction2;
import scala.runtime.BoxedUnit;
import scala.util.Try;

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
  private static final Logger LOG = LoggerFactory.getLogger(SparkJvmEngine.class);
  static final int MAX_IPC_BYTES = 64 * 1024 * 1024;
  private static final int DEFAULT_BATCH_ROWS = 10000;
  // Status values follow the gRPC status-code numbering.
  private static final int STATUS_CANCELLED = 1;
  private static final int STATUS_INVALID_ARGUMENT = 3;
  private static final int STATUS_NOT_FOUND = 5;
  private static final int STATUS_RESOURCE_EXHAUSTED = 8;
  private static final int STATUS_FAILED_PRECONDITION = 9;
  private static final int STATUS_UNIMPLEMENTED = 12;
  private static final int STATUS_INTERNAL = 13;
  private static final int CONFIG_GET = 1;
  private static final int CONFIG_GET_OPTION = 2;
  private static final int CONFIG_GET_WITH_DEFAULT = 3;
  private static final int CONFIG_GET_ALL = 4;
  private static final int CONFIG_IS_MODIFIABLE = 5;
  private static final int CONFIG_SET = 6;
  private static final int CONFIG_UNSET = 7;
  private static final int CONFIG_SET_SILENT = 8;
  private static final long SHUTDOWN_POLL_MILLIS = 100;
  private static final String SHUFFLE_FILE_CLEANUP =
      "spark.sql.classic.shuffleDependency.fileCleanup.enabled";
  // Compiler-generated Cleaner frame name in the pinned Spark build.
  private static final String ARTIFACT_GLOBAL_CLEANUP_METHOD =
      "org$apache$spark$sql$artifact$ArtifactManager$$cleanUpGlobalResources";
  /**
   * Maximum result partitions running or buffered, including the partition being delivered.
   * Nonpositive or unset values use the scheduler's default parallelism for each query.
   */
  static final String FETCH_MAX_CONCURRENT_PARTITIONS =
      "spark.connect.native.fetch.maxConcurrentPartitions";
  /** Target bound on fetched but undelivered IPC bytes; one partition is always admitted. */
  static final String FETCH_MAX_BUFFERED_BYTES = "spark.connect.native.fetch.maxBufferedBytes";
  private static final EnumSet<RelTypeCase> RELATIONS = EnumSet.of(
      RelTypeCase.RANGE, RelTypeCase.LOCAL_RELATION, RelTypeCase.SQL, RelTypeCase.PROJECT,
      RelTypeCase.FILTER, RelTypeCase.LIMIT, RelTypeCase.OFFSET, RelTypeCase.TAIL,
      RelTypeCase.JOIN, RelTypeCase.SORT, RelTypeCase.AGGREGATE, RelTypeCase.DEDUPLICATE,
      RelTypeCase.SET_OP, RelTypeCase.SUBQUERY_ALIAS, RelTypeCase.TO_DF,
      RelTypeCase.WITH_COLUMNS_RENAMED, RelTypeCase.WITH_COLUMNS, RelTypeCase.DROP,
      RelTypeCase.REPARTITION, RelTypeCase.REPARTITION_BY_EXPRESSION,
      RelTypeCase.HINT, RelTypeCase.READ);

  private final org.apache.spark.sql.classic.SparkSession base;
  private final SparkContext sparkContext;
  private final Pattern redactionPattern;
  private final ClassLoader classLoader;
  private final int fetchMaxConcurrentPartitions;
  private final long fetchMaxBufferedBytes;
  private final AtomicBoolean closed = new AtomicBoolean();
  private final Set<Session> sessions = ConcurrentHashMap.newKeySet();
  private final Set<Task> tasks = ConcurrentHashMap.newKeySet();
  private boolean closeComplete;

  public SparkJvmEngine(SparkSession session) {
    // The configured fetch policy comes from the host, not a client session.
    // Use an eighth of the heap as a soft IPC target: Spark's result transfer
    // transiently holds several serialized copies of each partition.
    this(Objects.requireNonNull(session, "session"),
        session.sparkContext().getConf().getInt(FETCH_MAX_CONCURRENT_PARTITIONS, 0),
        session.sparkContext().getConf().getSizeAsBytes(
            FETCH_MAX_BUFFERED_BYTES, Long.toString(Runtime.getRuntime().maxMemory() / 8)));
  }

  // Package-local so tests can exercise fetch windows without restarting SparkContext.
  SparkJvmEngine(SparkSession session, int maxConcurrentPartitions, long maxBufferedBytes) {
    if (!(Objects.requireNonNull(session, "session")
        instanceof org.apache.spark.sql.classic.SparkSession classic)) {
      throw new IllegalArgumentException("a classic Spark session is required");
    }
    base = classic;
    sparkContext = base.sparkContext();
    // SparkContext configuration is fixed after startup. Cache the host policy rather
    // than cloning the whole SparkConf for each configuration access or reading session overrides.
    redactionPattern = sparkContext.getConf().get(
        org.apache.spark.internal.config.package$.MODULE$.SECRET_REDACTION_PATTERN()).pattern();
    ClassLoader current = Thread.currentThread().getContextClassLoader();
    classLoader = current == null ? getClass().getClassLoader() : current;
    fetchMaxConcurrentPartitions = maxConcurrentPartitions;
    fetchMaxBufferedBytes = Math.max(1L, maxBufferedBytes);
  }

  /** Allocate one control per operation before invoking a planner or cursor, then close it. */
  public Task newTask(String description) {
    ensureOpen();
    Task task = new Task(Objects.requireNonNull(description, "description"));
    tasks.add(task);
    if (closed.get()) {
      task.close();
      ensureOpen();
    }
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
    if (closed.get()) throw new EngineException(STATUS_FAILED_PRECONDITION, "JVM engine is closed");
  }

  /**
   * Idempotently release this engine's tasks and sessions without stopping the shared
   * SparkContext. Once passed to NativeEngineServer, drain that server before calling this.
   * Cancellation and worker-drain failures can be retried. A partial resource-teardown failure
   * remains visible on later closes without replaying its side effects.
   */
  @Override
  public synchronized void close() {
    if (closeComplete) return;
    closed.set(true);
    Throwable failure = null;
    for (Task task : tasks) {
      try { task.close(); } catch (Throwable error) { failure = recordFailure(failure, error); }
    }
    for (Session session : sessions) {
      try { session.close(); } catch (Throwable error) { failure = recordFailure(failure, error); }
    }
    // Child owners retain nonreplayable cleanup failures. Cancellation/worker drain
    // can recover, so retry those owners on a later close instead of freezing the error.
    throwFailure(failure);
    closeComplete = true;
  }

  /** A stable error surface using the gRPC status-code numbering. */
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
    if (error instanceof EngineException engine) return engine;
    if (error instanceof OutOfMemoryException allocation) {
      return arrowAllocationFailure(allocation, error);
    }
    if (error instanceof SparkException) {
      // Executor conversion errors arrive inside Spark's task-failure wrapper. Preserve only
      // resource errors from that cause chain; an arbitrary nested argument error is not a
      // client validation failure.
      Set<Throwable> seen = Collections.newSetFromMap(new IdentityHashMap<>());
      for (Throwable cause = error.getCause(); cause != null && seen.add(cause);
          cause = cause.getCause()) {
        if (cause instanceof EngineException resource
            && resource.code() == STATUS_RESOURCE_EXHAUSTED) {
          return new EngineException(STATUS_RESOURCE_EXHAUSTED, resource.getMessage(), error);
        }
        if (cause instanceof OutOfMemoryException allocation) {
          return arrowAllocationFailure(allocation, error);
        }
      }
    }
    int code = STATUS_INTERNAL;
    // The pinned planner uses InvalidPlanInput for malformed client plans even though its
    // legacy message-only constructor labels them INTERNAL_ERROR. Its concrete type is the
    // input-validation contract; arbitrary Spark internal exceptions still map to INTERNAL.
    if (error instanceof InvalidProtocolBufferException || error instanceof InvalidPlanInput) {
      code = STATUS_INVALID_ARGUMENT;
    } else if (error instanceof SparkThrowable structured && !structured.isInternalError()) {
      if (error instanceof NoSuchElementException) code = STATUS_NOT_FOUND;
      else if (error instanceof AnalysisException || error instanceof IllegalArgumentException) {
        code = STATUS_INVALID_ARGUMENT;
      }
    }
    return new EngineException(code, error.getMessage(), error);
  }

  private static EngineException interrupted(Throwable error) {
    return new EngineException(STATUS_CANCELLED, "JVM engine operation was interrupted", error);
  }

  private static EngineException arrowAllocationFailure(
      OutOfMemoryException error, Throwable cause) {
    return new EngineException(STATUS_RESOURCE_EXHAUSTED,
        "JVM Arrow allocation failed: " + error.getMessage(), cause);
  }

  @SuppressWarnings("unchecked")
  static <E extends Exception> RuntimeException executorConversionFailure(Exception error) throws E {
    Set<Throwable> seen = Collections.newSetFromMap(new IdentityHashMap<>());
    for (Throwable cause = error; cause != null && seen.add(cause); cause = cause.getCause()) {
      if (cause instanceof OutOfMemoryException allocation) {
        // Pinned Arrow's exception holds a nonserializable Optional. Preserve its message
        // and stack without retaining the cause so Spark can serialize this typed status.
        EngineException failure = arrowAllocationFailure(allocation, null);
        failure.setStackTrace(allocation.getStackTrace());
        return failure;
      }
    }
    // Scala iterators can throw undeclared checked exceptions; preserve every other failure.
    throw (E) error;
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
    private boolean cancellationDelivered;

    private Task(String description) { this.description = description; }

    public void cancel() {
      synchronized (cancellation) {
        boolean cancelJobs;
        synchronized (this) {
          if (completed || cancellationDelivered) return;
          cancelled = true;
          if (running != null) running.interrupt();
          cancelJobs = executionStarted || running != null;
        }
        // Request and delivery are separate: a failed delivery may be retried. A stopped
        // context accepts no jobs, but query close must still await its execution worker.
        if (cancelJobs && !sparkContext.isStopped()) {
          try {
            sparkContext.cancelJobGroupAndFutureJobs(jobGroup);
          } catch (RuntimeException error) {
            // The context can stop between the check and the scheduler call.
            if (!sparkContext.isStopped()) throw error;
          }
        }
        cancellationDelivered = true;
      }
    }

    private boolean isCancelled() { return cancelled; }

    private void check() {
      if (cancelled) {
        throw new EngineException(STATUS_CANCELLED, "JVM engine operation was cancelled");
      }
    }

    private synchronized void complete() { completed = true; }

    private synchronized void beginExecution() {
      check();
      executionStarted = true;
    }

    private EngineException taskError(Exception error) {
      if (isCancelled() && !(error instanceof EngineException)) {
        return new EngineException(STATUS_CANCELLED, "JVM engine operation was cancelled", error);
      }
      return engineError(error);
    }

    private <T> T run(Session session, Callable<T> action) {
      Thread thread = Thread.currentThread();
      boolean previousInterrupted;
      synchronized (this) {
        check();
        if (completed) {
          throw new EngineException(STATUS_FAILED_PRECONDITION, "task is already complete");
        }
        if (running != null) {
          throw new EngineException(STATUS_FAILED_PRECONDITION, "task is already executing");
        }
        previousInterrupted = Thread.interrupted();
        running = thread;
      }
      try {
        // Capture can fail too (for example, a caller's context-class-loader accessor). Keep
        // the running mark covered before any caller/Spark code and until restoration ends.
        Properties previousProperties = sparkContext.getLocalProperties();
        ClassLoader previousLoader = thread.getContextClassLoader();
        Option<SparkSession> previousSession = SparkSession.getActiveSession();
        try {
          if (session != null) session.activeTasks.add(this);
          thread.setContextClassLoader(classLoader);
          sparkContext.setLocalProperties(new Properties());
          sparkContext.setJobGroup(jobGroup, description, true);
          SparkSession.setActiveSession(session == null ? base : session.spark);
          check();
          if (session == null) return action.call();
          session.ensureOpen();
          return session.holder.withSession(
              new AbstractFunction1<org.apache.spark.sql.classic.SparkSession, T>() {
                @Override
                public T apply(org.apache.spark.sql.classic.SparkSession ignored) {
                  try {
                    check();
                    return action.call();
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
    private final ArtifactManager artifacts;
    private final SessionHolder holder;
    private final ReentrantReadWriteLock catalog = new ReentrantReadWriteLock(true);
    private final AtomicBoolean released = new AtomicBoolean();
    private final Set<Query> queries = ConcurrentHashMap.newKeySet();
    private final Set<Task> activeTasks = ConcurrentHashMap.newKeySet();
    private boolean closeComplete;
    private Throwable closeFailure;

    private Session(String userId, String sessionId) {
      spark = base.newSession();
      // Spark initializes this lazily. Capture it while opening so even a never-used
      // session can close without first constructing resources after the host stops.
      artifacts = spark.artifactManager();
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
      if (released.get()) {
        throw new EngineException(STATUS_FAILED_PRECONDITION, "JVM engine session is closed");
      }
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
          throw new EngineException(STATUS_UNIMPLEMENTED, "unsupported analysis opcode: " + opcode);
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
      checkExecutionSettings();
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
        throw new EngineException(STATUS_UNIMPLEMENTED, "streaming queries are unsupported");
      }
      return frame;
    }

    private void checkExecutionSettings() {
      if (Boolean.parseBoolean(spark.conf().get(SHUFFLE_FILE_CLEANUP).trim())) {
        throw new EngineException(STATUS_UNIMPLEMENTED, SHUFFLE_FILE_CLEANUP
            + " has not been validated by the JVM engine");
      }
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
        throw new EngineException(STATUS_UNIMPLEMENTED,
            "SQL commands and external writes are unsupported");
      }
    }

    /**
     * Internal typed fields: non-negative big-endian signed 32-bit pair count, then UTF-8 key
     * length/data and nullable value length/data (signed -1 means absent). No Spark Connect
     * request/response envelopes cross here.
     * Opcodes are 1=Get, 2=GetOption, 3=GetWithDefault, 4=GetAll, 5=IsModifiable,
     * 6=Set, 7=Unset and 8=SetSilent. Opcodes 1 through 5 read configuration;
     * 6 through 8 mutate it.
     */
    public byte[] config(int opcode, byte[] fields, Task task) {
      List<Pair> input = decodePairs(fields);
      return task.run(this, () -> {
        if (opcode < CONFIG_GET || opcode > CONFIG_SET_SILENT) {
          throw new EngineException(STATUS_UNIMPLEMENTED, "unsupported config opcode: " + opcode);
        }
        Lock access = opcode <= CONFIG_IS_MODIFIABLE ? catalog.readLock() : catalog.writeLock();
        access.lockInterruptibly();
        try {
          task.check();
          ensureOpen();
          List<Pair> output = new ArrayList<>();
          switch (opcode) {
            case CONFIG_GET:
              for (Pair pair : input) {
                String value = spark.conf().get(pair.key);
                if (isRedacted(redactionPattern, pair.key, value)) {
                  throw QueryExecutionErrors$.MODULE$.sqlConfigNotFoundError(pair.key);
                }
                output.add(new Pair(pair.key, value));
              }
              break;
            case CONFIG_GET_OPTION:
              for (Pair pair : input) {
                Option<String> stored = spark.conf().getOption(pair.key);
                String value = stored.isDefined() ? stored.get() : null;
                output.add(new Pair(pair.key,
                    isRedacted(redactionPattern, pair.key, value) ? null : value));
              }
              break;
            case CONFIG_GET_WITH_DEFAULT:
              for (Pair pair : input) {
                String value = spark.conf().get(pair.key, pair.value);
                output.add(new Pair(pair.key,
                    isRedacted(redactionPattern, pair.key, value) ? pair.value : value));
              }
              break;
            case CONFIG_GET_ALL:
              if (input.size() > 1) {
                throw new EngineException(STATUS_INVALID_ARGUMENT, "getAll accepts one prefix");
              }
              String prefix = input.isEmpty() ? "" : input.get(0).key;
              spark.conf().getAllAsJava().entrySet().stream()
                  .filter(entry -> !isRedacted(redactionPattern, entry.getKey(), entry.getValue()))
                  .filter(entry -> entry.getKey().startsWith(prefix))
                  .sorted(Map.Entry.comparingByKey())
                  .forEach(entry -> output.add(new Pair(
                      entry.getKey().substring(prefix.length()), entry.getValue())));
              break;
            case CONFIG_IS_MODIFIABLE:
              for (Pair pair : input) {
                output.add(new Pair(pair.key, Boolean.toString(spark.conf().isModifiable(pair.key))));
              }
              break;
            case CONFIG_SET:
            case CONFIG_UNSET:
              applyConfig(opcode, input, task);
              break;
            case CONFIG_SET_SILENT:
              // A separate result preserves setter-error provenance. Task/admission
              // failures still fail the call rather than becoming silent Set warnings.
              if (input.size() != 1) {
                throw new EngineException(STATUS_INVALID_ARGUMENT, "silent set accepts one key");
              }
              String warning = applyConfig(opcode, input, task);
              if (warning != null) output.add(new Pair(input.get(0).key, warning));
              break;
          }
          if (opcode <= CONFIG_IS_MODIFIABLE) task.complete();
          return encodePairs(output);
        } finally {
          access.unlock();
        }
      });
    }

    private String applyConfig(int opcode, List<Pair> input, Task task) {
      boolean silent = opcode == CONFIG_SET_SILENT;
      Map<String, String> previous = new LinkedHashMap<>();
      Set<String> attempted = new LinkedHashSet<>();
      Map<String, String> explicit = spark.conf().getAllAsJava();
      for (Pair pair : input) {
        if (opcode == CONFIG_SET && pair.value == null) {
          throw new EngineException(STATUS_INVALID_ARGUMENT, "missing config value");
        }
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
            if (opcode != CONFIG_UNSET) spark.conf().set(pair.key, pair.value);
            else spark.conf().unset(pair.key);
            if (SHUFFLE_FILE_CLEANUP.equals(pair.key)) checkExecutionSettings();
          } catch (Throwable error) {
            setterFailure = error;
            if (!silent && error instanceof IllegalArgumentException) {
              if (!(error instanceof SparkThrowable structured) || !structured.isInternalError()) {
                throw new EngineException(STATUS_INVALID_ARGUMENT, error.getMessage(), error);
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
        if (silent && error instanceof Exception && error == setterFailure && restored) {
          // Match Spark's silent Set diagnostics without retaining rejected values/messages.
          String reason = error instanceof SparkThrowable structured
              ? structured.getCondition() : null;
          if (reason == null) reason = error instanceof EngineException
              ? "UNSUPPORTED_EXECUTION_SETTING" : error.getClass().getName();
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
      // Delegated cancellation/drain failures can recover. Only this session's actual
      // resource teardown is nonreplayable (artifact cleanup can decrement reference counts).
      throwFailure(failure);
      catalog.writeLock().lock();
      try {
        // These resources belong to this newSession only. Never close SessionHolder or Spark.
        try {
          spark.sessionState().catalog().clearTempTables();
        } catch (Throwable error) {
          failure = recordFailure(failure, error);
        }
        try {
          closeArtifacts();
        } catch (Throwable error) {
          failure = recordFailure(failure, error);
        }
      } finally {
        catalog.writeLock().unlock();
      }
      closeFailure = failure;
      throwFailure(failure);
      sessions.remove(this);
      closeComplete = true;
    }

    private void closeArtifacts() {
      try {
        artifacts.close();
      } catch (RuntimeException error) {
        if (!sparkContext.isStopped()) throw error;
        boolean globalCleanupFailed = false;
        for (StackTraceElement frame : error.getStackTrace()) {
          if (frame.getClassName().equals("org.apache.spark.sql.artifact.ArtifactManager$")
              && frame.getMethodName().equals(ARTIFACT_GLOBAL_CLEANUP_METHOD)) {
            globalCleanupFailed = true;
            break;
          }
        }
        if (!globalCleanupFailed) throw error;
        // Pinned Spark's one-shot Cleaner can fail on the global SparkEnv or environment
        // update after host stop begins. Its global work is idempotent; local teardown has
        // not started. Finish global work through the owner, then local teardown once.
        // Never restore SparkEnv: it is process-global and belongs to the host.
        Throwable failure = null;
        try {
          RpcEnvFileServer files = sparkContext.env().rpcEnv().fileServer();
          removeArtifacts(sparkContext.addedFiles().remove(spark.sessionUUID()), files::removeFile);
          removeArtifacts(sparkContext.addedArchives().remove(spark.sessionUUID()), files::removeFile);
          removeArtifacts(sparkContext.addedJars().remove(spark.sessionUUID()), files::removeJar);
          try {
            Utils.deleteRecursively(artifacts.artifactPath().toFile());
          } catch (Exception cleanup) {
            // Scala does not declare this checked exception in the Java signature.
            if (!(cleanup instanceof IOException)) throw cleanup;
            // Match pinned ArtifactManager: directory deletion is best effort; failures
            // removing registered resources above must still remain visible to the owner.
            LOG.warn("Unable to remove native session artifact directory", cleanup);
          }
        } catch (Throwable cleanup) {
          failure = cleanup;
        }
        // A directory-delete failure must not skip cache/refcount and classloader cleanup.
        try { artifacts.close(); } catch (Throwable cleanup) {
          failure = recordFailure(failure, cleanup);
        }
        if (failure != null) {
          recordFailure(failure, error);
          throwFailure(failure);
        }
      }
    }

    private void removeArtifacts(Option<scala.collection.concurrent.Map<String, Object>> paths,
        Consumer<String> remove) {
      if (paths.isDefined()) CollectionConverters.asJava(paths.get()).keySet().forEach(remove);
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
      maxRows = configured > 0 ? configured : DEFAULT_BATCH_ROWS;
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
    // Scheduler callbacks never take the cursor lock or run caller work.
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

    // Fetch window, guarded by results. Partitions [nextPartition, submitted) are admitted,
    // including the slot reserved for the current submission. Until sizes are known, start at
    // two and double after each delivered partition while the byte budget allows.
    private int submitted;
    private int finishedJobs;
    private int window;
    private int maxWindow;
    private long largestPartitionBytes;
    private Throwable jobFailure;

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
    /** Always false because preparation rejects streaming plans. */
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
                  runJobs(batches, count);
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

    // Package-local: job-start events are asynchronous, so tests read the window directly.
    int submittedPartitions() {
      synchronized (results) { return submitted; }
    }

    // Caller holds results. Counts the partition being delivered, so a window of one is
    // sequential and a partition larger than the byte budget is still fetched alone.
    private int windowLimit() {
      if (largestPartitionBytes <= 0) return window;
      return (int) Math.max(1, Math.min(window, fetchMaxBufferedBytes / largestPartitionBytes));
    }

    // Caller holds results. Both cancellation paths stop admission before the next submit.
    private boolean stopping() {
      return jobFailure != null || released || pullFailure != null
          || task.isCancelled() || sparkContext.isStopped();
    }

    private EngineException contextStopped() {
      return new EngineException(STATUS_INTERNAL, "hosting SparkContext stopped during execution");
    }

    private SimpleFutureAction<BoxedUnit> submitPartition(
        RDD<Tuple2<byte[], Long>> batches, int partition) {
      // One job per partition keeps Spark's maxResultSize check per partition and lets the
      // window slide. The owner explicitly cancels sibling jobs after a partition fails.
      SimpleFutureAction<BoxedUnit> action = sparkContext.submitJob(
          batches, new CollectArrowPartition(),
          CollectionConverters.asScala(Collections.<Object>singletonList(partition)).toSeq(),
          new AbstractFunction2<Object, Tuple2<byte[], Long>[], BoxedUnit>() {
            @Override
            public BoxedUnit apply(Object ignored, Tuple2<byte[], Long>[] fetched) {
              long bytes = 0;
              for (Tuple2<byte[], Long> batch : fetched) bytes += batch._1().length;
              synchronized (results) {
                largestPartitionBytes = Math.max(largestPartitionBytes, bytes);
                if (!released && pullFailure == null) {
                  partitionBatches.set(partition, fetched);
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
      // Runs after the result callback, on the scheduler thread; it only records and notifies.
      action.onComplete(new AbstractFunction1<Try<BoxedUnit>, BoxedUnit>() {
        @Override
        public BoxedUnit apply(Try<BoxedUnit> outcome) {
          synchronized (results) {
            finishedJobs++;
            if (outcome.isFailure() && jobFailure == null) jobFailure = outcome.failed().get();
            results.notifyAll();
          }
          return BoxedUnit.UNIT;
        }
      }, ThreadUtils.sameThread());
      return action;
    }

    /** Stay in scope until admitted work completes or its owning scheduler shuts down. */
    private void runJobs(RDD<Tuple2<byte[], Long>> batches, int count) {
      DAGScheduler scheduler = sparkContext.dagScheduler();
      if (scheduler == null) throw contextStopped();
      List<SimpleFutureAction<BoxedUnit>> actions = new ArrayList<>();
      EngineException interruption = null;
      Throwable failure = null;
      try {
        // Cluster resources can change after engine startup. Resolve the scheduler default
        // for this query, inside the failure boundary because a concurrent host stop throws.
        int maximum = fetchMaxConcurrentPartitions;
        if (maximum <= 0) {
          maximum = Math.max(1, sparkContext.defaultParallelism());
        }
        synchronized (results) {
          maxWindow = maximum;
          window = Math.min(2, maxWindow);
        }
        for (;;) {
          int partition;
          synchronized (results) {
            while (!stopping() && submitted < count
                && submitted - nextPartition >= windowLimit()) {
              results.wait(SHUTDOWN_POLL_MILLIS);
            }
            if (stopping() || submitted == count) break;
            partition = submitted++;
          }
          actions.add(submitPartition(batches, partition));
        }
      } catch (InterruptedException error) {
        interruption = interrupted(error);
      } catch (Throwable error) {
        failure = error;
        if (error instanceof RuntimeException && sparkContext.isStopped()) {
          failure = contextStopped();
          failure.addSuppressed(error);
        }
      }
      // Cancellation interrupts the owner, but scope/plan cleanup must await every job too.
      boolean cancellationRequested = false;
      for (;;) {
        synchronized (results) {
          if (sparkContext.isStopped() && failure == null && submitted < count) {
            failure = contextStopped();
          }
          if (finishedJobs == actions.size()) break;
          // Spark can discard a queued JobSubmitted during stop without completing its
          // future. Only after stop() has drained the scheduler and its event loop has
          // exited can this scope leave without that callback; never invent completion.
          if (sparkContext.isStopped() && sparkContext.dagScheduler() == null
              && !scheduler.eventProcessLoop().isActive()) {
            if (failure == null) failure = contextStopped();
            break;
          }
          boolean cancel = interruption != null || failure != null || stopping();
          if (!cancel || cancellationRequested) {
            try {
              results.wait(SHUTDOWN_POLL_MILLIS);
            } catch (InterruptedException error) {
              if (interruption == null) interruption = interrupted(error);
              // A new interrupt also retries any earlier failed scheduler cancellation.
              cancellationRequested = false;
            }
            continue;
          }
        }
        // Request every sibling's stop outside the callback monitor before awaiting them.
        for (SimpleFutureAction<BoxedUnit> pending : actions) {
          if (!pending.isCompleted()) {
            try { pending.cancel(); } catch (Throwable error) {
              failure = recordFailure(failure, error);
            }
          }
        }
        cancellationRequested = true;
      }
      if (interruption != null) throw interruption;
      Throwable failed;
      synchronized (results) { failed = jobFailure; }
      if (failed != null && failure == null && sparkContext.isStopped()) failure = contextStopped();
      // Scheduler cancellation errors must not hide the executor's original failure.
      if (failed != null && failure != null) recordFailure(failed, failure);
      if (failed == null) failed = failure;
      // The completed Try's failure retains the executor's original message.
      if (failed instanceof Exception exception) throw task.taskError(exception);
      throwFailure(failed);
      task.check();
    }

    private EngineException cursorClosed() {
      return new EngineException(STATUS_FAILED_PRECONDITION, "query cursor is closed");
    }

    // Caller holds results. The failure guard makes later scheduler callbacks discard data.
    private void fail(Throwable error) {
      pullFailure = error instanceof InterruptedException
          ? interrupted(error)
          : error instanceof Exception exception ? task.taskError(exception) : error;
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
                    window = (int) Math.min(2L * window, maxWindow);
                    results.notifyAll();
                    continue;
                  }
                  Tuple2<byte[], Long> batch = batches[nextBatch];
                  batches[nextBatch++] = null;
                  return new NativeIpcBatch(batch._1(), batch._2());
                }
              }
              if (workerFinished) {
                if (partitionBatches != null && nextPartition < partitionBatches.size()) {
                  throw new EngineException(STATUS_INTERNAL,
                      "Spark job completed without a partition result");
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
              closeFailure = error;
            }
          }
          if (closeFailure != null) failure = recordFailure(failure, closeFailure);
          throwFailure(failure);
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
        || (type instanceof StringType string && string.collationId()
            != ((StringType) DataTypes.StringType).collationId())) {
      throw new EngineException(STATUS_UNIMPLEMENTED,
          "result type cannot be preserved by Arrow schema transport: " + type.catalogString());
    }
    if (type instanceof StructType original) {
      StructField[] fields = org.apache.spark.sql.catalyst.util.package$.MODULE$
          .removeInternalMetadata(original).fields();
      Set<String> names = nested ? new HashSet<>() : Collections.emptySet();
      for (int i = 0; i < fields.length; i++) {
        StructField field = fields[i];
        if (nested && !names.add(field.name())) {
          throw new EngineException(STATUS_UNIMPLEMENTED,
              "duplicate nested field names cannot be preserved by Arrow schema transport: "
                  + field.name());
        }
        // CHAR/VARCHAR's raw type annotation is semantic even though its name is internal.
        if (original.fields()[i].metadata().contains("__CHAR_VARCHAR_TYPE_STRING")) {
          throw new EngineException(STATUS_UNIMPLEMENTED,
              "CHAR/VARCHAR result metadata cannot be preserved by Arrow schema transport: "
                  + field.name());
        }
        if (!field.metadata().isEmpty()) {
          throw new EngineException(STATUS_UNIMPLEMENTED,
              "result field metadata cannot be preserved by Arrow schema transport: " + field.name());
        }
        validateResultType(field.dataType(), true);
      }
    } else if (type instanceof ArrayType array) {
      validateResultType(array.elementType(), true);
    } else if (type instanceof MapType map) {
      validateResultType(map.keyType(), true);
      validateResultType(map.valueType(), true);
    }
  }

  static byte[] checkedIpc(byte[] bytes) {
    if (bytes.length > MAX_IPC_BYTES) {
      throw new EngineException(STATUS_RESOURCE_EXHAUSTED, "Arrow batch exceeds 64 MiB");
    }
    return bytes;
  }

  private static Throwable recordFailure(Throwable previous, Throwable error) {
    if (previous == null) return error;
    if (previous != error) previous.addSuppressed(error);
    return previous;
  }

  private static void throwFailure(Throwable failure) {
    if (failure instanceof Error error) throw error;
    if (failure instanceof RuntimeException exception) throw exception;
    if (failure != null) {
      throw new EngineException(STATUS_INTERNAL, "JVM engine cleanup failed", failure);
    }
  }

  private static void validateRelations(Message message) {
    if (message instanceof Relation relation) {
      RelTypeCase kind = relation.getRelTypeCase();
      if (!RELATIONS.contains(kind)) {
        throw new EngineException(STATUS_UNIMPLEMENTED, "unsupported JVM engine relation: " + kind);
      }
      if (relation.hasRead() && !relation.getRead().hasNamedTable()) {
        throw new EngineException(STATUS_UNIMPLEMENTED, "only named-table reads are supported");
      }
    }
    if (message instanceof CommonInlineUserDefinedFunction
        || message instanceof CommonInlineUserDefinedTableFunction
        || message instanceof TypedAggregateExpression || message instanceof ScalarScalaUDF) {
      throw new EngineException(STATUS_UNIMPLEMENTED,
          "user-defined function inputs are unsupported");
    }
    for (Object field : message.getAllFields().values()) {
      if (field instanceof Message nested) validateRelations(nested);
      else if (field instanceof List<?> values) {
        for (Object item : values) {
          if (item instanceof Message nested) validateRelations(nested);
        }
      }
    }
  }

  private record Pair(String key, String value) {}

  private static boolean isRedacted(Pattern pattern, String key, String value) {
    return pattern.matcher(key).find() || value != null && pattern.matcher(value).find();
  }

  private static List<Pair> decodePairs(byte[] bytes) {
    Objects.requireNonNull(bytes, "fields");
    if (bytes.length > MAX_IPC_BYTES) {
      throw new EngineException(STATUS_RESOURCE_EXHAUSTED, "config fields exceed 64 MiB");
    }
    try (DataInputStream input = new DataInputStream(new ByteArrayInputStream(bytes))) {
      int count = input.readInt();
      if (count < 0 || count > input.available() / 8) {
        throw new EngineException(STATUS_INVALID_ARGUMENT, "invalid config field count");
      }
      List<Pair> result = new ArrayList<>(count);
      for (int i = 0; i < count; i++) result.add(new Pair(readString(input, false), readString(input, true)));
      if (input.available() != 0) {
        throw new EngineException(STATUS_INVALID_ARGUMENT, "trailing config field bytes");
      }
      return result;
    } catch (IOException error) {
      throw new EngineException(STATUS_INVALID_ARGUMENT, "invalid config fields", error);
    }
  }

  private static String readString(DataInputStream input, boolean nullable) throws IOException {
    int len = input.readInt();
    if (len == -1 && nullable) return null;
    if (len < 0 || len > input.available()) {
      throw new EngineException(STATUS_INVALID_ARGUMENT, "invalid config string length");
    }
    byte[] bytes = input.readNBytes(len);
    try {
      return StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
          .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(bytes)).toString();
    } catch (CharacterCodingException error) {
      throw new EngineException(STATUS_INVALID_ARGUMENT, "invalid config UTF-8", error);
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
