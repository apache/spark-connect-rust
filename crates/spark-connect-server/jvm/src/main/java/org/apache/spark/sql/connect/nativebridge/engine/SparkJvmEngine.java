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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.NoSuchElementException;
import java.util.Properties;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.apache.spark.SparkContext;
import org.apache.spark.SparkEnv;
import org.apache.spark.SparkThrowable;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.plans.logical.Command;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.catalyst.plans.logical.InsertIntoDir;
import org.apache.spark.sql.catalyst.plans.logical.ParsedStatement;
import org.apache.spark.sql.classic.Dataset;
import org.apache.spark.sql.connect.planner.SparkConnectPlanner;
import org.apache.spark.sql.connect.service.SessionHolder;
import org.apache.spark.sql.execution.QueryExecution;
import org.apache.spark.sql.execution.CommandExecutionMode;
import org.apache.spark.sql.execution.SQLExecution;
import org.apache.spark.sql.execution.SparkPlan;
import org.apache.spark.sql.execution.arrow.ArrowConverters$;
import org.apache.spark.sql.execution.arrow.ArrowWriter;
import org.apache.spark.sql.util.ArrowUtils$;
import org.apache.spark.sql.catalyst.expressions.UnsafeRow;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.OutOfMemoryException;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.CharType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.MapType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.UserDefinedType;
import org.apache.spark.sql.types.VarcharType;
import org.sparkproject.connect.protobuf.Message;
import org.sparkproject.connect.protobuf.InvalidProtocolBufferException;

import scala.Option;
import scala.collection.Iterator;
import scala.collection.AbstractIterator;
import scala.runtime.AbstractFunction0;
import scala.runtime.AbstractFunction1;

/**
 * A query engine for the Rust-owned frontend. The JNI boundary passes Catalyst inputs and Arrow
 * batches, never service RPC requests, responses, observers, or operation registries.
 *
 * This intentionally targets the cached Spark 4.2 assembly. SessionHolder is a planner resource
 * container only: its service-managed close() and SparkSession.close() are never called.
 *
 * Passing this engine to NativeEngineServer transfers its lifetime to that server. Native stop
 * closes it, and a failed native start can close it too; do not share or reuse it afterwards.
 * close() is idempotent, so callers may also close it while cleaning up a failed constructor.
 * The caller retains ownership of the original SparkSession and shared SparkContext.
 */
public final class SparkJvmEngine implements AutoCloseable {
  private static final int MAX_IPC_BYTES = 64 * 1024 * 1024;
  private static final long MAX_ESTIMATED_BATCH_BYTES = 4 * 1024 * 1024;
  private static final int MAX_BATCH_ROWS = 10000;
  private static final int MAX_CONFIG_FIELDS = 10000;
  private static final String SHUFFLE_FILE_CLEANUP =
      "spark.sql.classic.shuffleDependency.fileCleanup.enabled";
  private static final Set<String> RELATIONS = Set.of(
      "RANGE", "LOCAL_RELATION", "SQL", "PROJECT", "FILTER", "LIMIT", "OFFSET", "TAIL",
      "JOIN", "SORT", "AGGREGATE", "DEDUPLICATE", "SET_OP", "SUBQUERY_ALIAS", "TO_DF",
      "WITH_COLUMNS_RENAMED", "WITH_COLUMNS", "DROP", "REPARTITION",
      "REPARTITION_BY_EXPRESSION", "HINT", "READ");

  private final org.apache.spark.sql.classic.SparkSession base;
  private final SparkContext sparkContext;
  private final SparkEnv sparkEnv;
  private final ClassLoader classLoader;
  private final BufferAllocator arrowAllocator;
  private final AtomicBoolean closed = new AtomicBoolean();
  private final Set<Session> sessions = ConcurrentHashMap.newKeySet();
  private final Set<Task> tasks = ConcurrentHashMap.newKeySet();
  private final Set<NativeArrowBatch> batches = ConcurrentHashMap.newKeySet();
  private boolean closeComplete;
  private Throwable closeFailure;

  public SparkJvmEngine(SparkSession session) {
    this(session, ArrowUtils$.MODULE$.rootAllocator());
  }

  // The caller owns the parent allocator. A bounded parent also permits embedding tests to
  // exercise real allocation failures without exhausting the process's shared Arrow allocator.
  SparkJvmEngine(SparkSession session, BufferAllocator arrowAllocator) {
    if (!(Objects.requireNonNull(session, "session")
        instanceof org.apache.spark.sql.classic.SparkSession)) {
      throw new IllegalArgumentException("a classic Spark session is required");
    }
    base = (org.apache.spark.sql.classic.SparkSession) session;
    sparkContext = base.sparkContext();
    sparkEnv = SparkEnv.get();
    ClassLoader current = Thread.currentThread().getContextClassLoader();
    classLoader = current == null ? getClass().getClassLoader() : current;
    this.arrowAllocator = Objects.requireNonNull(arrowAllocator, "arrowAllocator");
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
    // Healthy leases remain independently owned by their native arrays. Failed batch cleanup
    // must remain visible even if JNI could not acquire its initial global reference.
    for (NativeArrowBatch batch : batches) {
      try { batch.checkCloseFailure(); } catch (Throwable error) {
        failure = recordFailure(failure, error);
      }
    }
    closeFailure = failure;
    throwFailure(failure);
    closeComplete = true;
  }

  @FunctionalInterface
  private interface Action<T> { T run() throws Exception; }

  /** A small stable error surface for the native adapter's tonic Status mapping. */
  public static final class EngineException extends RuntimeException {
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
      return new EngineException(8, "JVM Arrow allocation failed: " + error.getMessage(), error);
    }
    int code = 13;
    if (!(error instanceof SparkThrowable) || !((SparkThrowable) error).isInternalError()) {
      if (error instanceof NoSuchElementException) code = 5;
      else if (error instanceof AnalysisException || error instanceof IllegalArgumentException
          || error instanceof InvalidProtocolBufferException) code = 3;
    }
    return new EngineException(code, error.getMessage(), error);
  }

  /** Cancellation is independent of a query/session lock and remains usable during preparation. */
  public final class Task implements AutoCloseable {
    private final String jobGroup = "spark-connect-engine-" + UUID.randomUUID();
    private final String description;
    private boolean cancelled;
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

    private synchronized boolean isCancelled() { return cancelled; }

    private synchronized void check() {
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
      Properties previousProperties = sparkContext.getLocalProperties();
      ClassLoader previousLoader = thread.getContextClassLoader();
      SparkEnv previousEnv = SparkEnv.get();
      Option<SparkSession> previousSession = SparkSession.getActiveSession();
      if (session != null) session.activeTasks.add(this);
      try {
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
                  throw new EngineException(1, "JVM engine operation was interrupted", error);
                } catch (Exception error) {
                  throw taskError(error);
                }
              }
            });
      } catch (InterruptedException error) {
        throw new EngineException(1, "JVM engine operation was interrupted", error);
      } catch (Exception error) {
        throw taskError(error);
      } finally {
        if (session != null) session.activeTasks.remove(this);
        synchronized (this) { running = null; }
        Thread.interrupted();
        if (previousInterrupted) thread.interrupt();
        if (previousSession.isDefined()) SparkSession.setActiveSession(previousSession.get());
        else SparkSession.clearActiveSession();
        sparkContext.setLocalProperties(previousProperties);
        thread.setContextClassLoader(previousLoader);
        SparkEnv.set(previousEnv);
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
    private final Set<Query> queries = ConcurrentHashMap.newKeySet();
    private final Set<Task> activeTasks = ConcurrentHashMap.newKeySet();
    private boolean closeComplete;
    private Throwable closeFailure;

    private Session(String userId, String sessionId) {
      spark = base.newSession();
      holder = new SessionHolder(userId, sessionId, spark);
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
            + " is unsupported with per-fetch SQL execution scopes");
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
          String name = node.getClass().getSimpleName();
          // InsertIntoStatement extends ParsedStatement, and InsertIntoDir extends UnaryNode;
          // neither is a Command until analysis. SQL scripts and dynamic SQL can execute
          // their statements during analysis itself, before the post-analysis guard runs.
          return node instanceof Command || node instanceof ParsedStatement
              || node instanceof InsertIntoDir || name.equals("CompoundBody")
              || name.equals("ExecuteImmediateQuery") || name.equals("UnresolvedExecuteImmediate");
        }
      })) {
        throw new EngineException(12, "SQL commands and external writes are unsupported");
      }
    }

    /**
     * Internal typed fields: BE u32 pair count, then UTF-8 key length/data and nullable value
     * length/data (signed -1 means absent). No Spark Connect request/response envelopes cross here.
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
              for (Pair pair : input) output.add(new Pair(pair.key, spark.conf().get(pair.key)));
              break;
            case 2:
              for (Pair pair : input) {
                Option<String> value = spark.conf().getOption(pair.key);
                output.add(new Pair(pair.key, value.isDefined() ? value.get() : null));
              }
              break;
            case 3:
              for (Pair pair : input) {
                output.add(new Pair(pair.key, spark.conf().get(pair.key, pair.value)));
              }
              break;
            case 4:
              if (input.size() > 1) throw new EngineException(3, "getAll accepts one prefix");
              String prefix = input.isEmpty() ? "" : input.get(0).key;
              spark.conf().getAllAsJava().entrySet().stream()
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
      // JNI contract: 1..5 are reads; 6..7 mutate. Unknown opcodes cannot come from Rust;
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

    private void applyConfig(int opcode, List<Pair> input, Task task) {
      Map<String, String> previous = new LinkedHashMap<>();
      Map<String, String> explicit = spark.conf().getAllAsJava();
      for (Pair pair : input) {
        if (opcode == 6 && pair.value == null) throw new EngineException(3, "missing config value");
        if (!previous.containsKey(pair.key)) previous.put(pair.key, explicit.get(pair.key));
      }
      synchronized (task) {
        task.check();
        try {
          for (Pair pair : input) {
            // RuntimeConfig accepts unregistered spark.* keys even when isModifiable is false;
            // let its setters reject static/core entries and invalid registered values.
            if (opcode == 6) spark.conf().set(pair.key, pair.value);
            else spark.conf().unset(pair.key);
          }
          task.complete();
        } catch (Throwable error) {
          for (Map.Entry<String, String> entry : previous.entrySet()) {
            try {
              if (entry.getValue() == null) spark.conf().unset(entry.getKey());
              else spark.conf().set(entry.getKey(), entry.getValue());
            } catch (Throwable rollback) {
              if (rollback != error) error.addSuppressed(rollback);
            }
          }
          throw error;
        }
      }
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

  /** An analyzed Catalyst query and a lazy pull cursor. Only one caller may fetch at a time. */
  public final class Query implements AutoCloseable {
    private final Session session;
    private final QueryExecution execution;
    private final StructType schema;
    private final Task task;
    private final String timeZone;
    private final boolean largeVarTypes;
    private final int maxRows;
    private final byte[] schemaIpc;
    private final boolean local;
    private final ReentrantLock cursor = new ReentrantLock();
    private boolean released;
    private Throwable closeFailure;
    // Construction can fail before a NativeArrowBatch owns these resources. Keep partial
    // cleanup failures reachable through this query; their cleanup must not be replayed.
    private VectorSchemaRoot failedBatchRoot;
    private BufferAllocator failedBatchAllocator;
    private boolean ended;
    private SparkPlan plan;
    private Iterator<InternalRow> rows;

    private Query(Session session, Dataset<Row> frame, Task task) {
      this.session = session;
      this.execution = frame.queryExecution();
      PlanMetadata metadata = new PlanMetadata(session, frame);
      this.schema = metadata.schema;
      this.task = task;
      this.timeZone = metadata.timeZone;
      this.largeVarTypes = metadata.largeVarTypes;
      this.maxRows = metadata.maxRows;
      this.local = metadata.local;
      this.schemaIpc = metadata.schemaIpc;
    }

    public byte[] schemaIpc() { return schemaIpc.clone(); }
    public boolean isLocal() { return local; }
    public boolean isStreaming() { return false; }

    private void initializeRows() {
      task.check();
      if (rows == null) {
        plan = execution.executedPlan();
        Iterator<InternalRow> raw = plan.executeToIterator();
        // Spark's iterator can materialize one partition. Arrow conversion below bounds each
        // JNI batch; it does not establish a memory bound for Spark partition execution.
        rows = new AbstractIterator<InternalRow>() {
          @Override
          public boolean hasNext() { task.check(); return raw.hasNext(); }
          @Override
          public InternalRow next() { task.check(); return raw.next(); }
        };
      }
    }

    /** Native zero-copy result path. Only immutable vector descriptors cross JNI. */
    public NativeArrowBatch nextVectorBatch() {
      NativeArrowBatch[] pending = new NativeArrowBatch[1];
      try {
        return task.run(session, () -> {
          cursor.lockInterruptibly();
          try {
            if (released) throw new EngineException(9, "query cursor is closed");
            if (ended) return null;
            task.beginExecution();
            // Each fetch restores its session/job-group scope; workers need not have affinity.
            return SQLExecution.withNewExecutionId(execution,
                Option.apply("spark-connect-engine-nextVectorBatch"),
                new AbstractFunction0<NativeArrowBatch>() {
                  @Override
                  public NativeArrowBatch apply() {
                    pending[0] = pullVectors();
                    return pending[0];
                  }
                });
          } finally {
            cursor.unlock();
          }
        });
      } catch (Throwable error) {
        // A failure while leaving the SQL/task scopes must not leak a batch JNI never received.
        if (pending[0] != null) {
          try { pending[0].close(); } catch (Throwable cleanup) { recordFailure(error, cleanup); }
        }
        throw error;
      }
    }

    private NativeArrowBatch pullVectors() {
      initializeRows();
      if (!rows.hasNext()) {
        ended = true;
        task.complete();
        return null;
      }
      // Exported batches have independent allocators: the next fetch never resets or overwrites
      // vectors still owned by native Arrow arrays, and query/session close cannot free them.
      BufferAllocator allocator = arrowAllocator
          .newChildAllocator("spark-connect-native-arrow-batch", 0, MAX_IPC_BYTES);
      VectorSchemaRoot root = null;
      try {
        root = VectorSchemaRoot.create(ArrowUtils$.MODULE$.toArrowSchema(
            schema, timeZone, false, largeVarTypes), allocator);
        ArrowWriter writer = ArrowWriter.create(root);
        long estimatedBytes = 0;
        int count = 0;
        while (count < maxRows && (count == 0 || estimatedBytes < MAX_ESTIMATED_BATCH_BYTES)
            && rows.hasNext()) {
          task.check();
          InternalRow row = rows.next();
          long size = row instanceof UnsafeRow ? ((UnsafeRow) row).getSizeInBytes()
              : (long) row.numFields() * 16;
          if (size > MAX_IPC_BYTES) {
            throw new EngineException(8, "Arrow row exceeds 64 MiB native batch limit");
          }
          writer.write(row);
          estimatedBytes += size;
          count++;
        }
        writer.finish();
        NativeArrowBatch batch = new NativeArrowBatch(root, allocator, batches);
        root = null; // Ownership is now solely the returned immutable batch lease.
        return batch;
      } catch (Throwable error) {
        cleanupFailedBatch(root, allocator, error);
        throw error;
      }
    }

    // Also covers descriptor construction failures before the batch registers its owner.
    // Package-local for deterministic resource-failure fixtures without exhausting Arrow memory.
    void cleanupFailedBatch(VectorSchemaRoot root, BufferAllocator allocator, Throwable error) {
      cursor.lock();
      try {
        Throwable failure = null;
        if (root != null) {
          try { root.close(); } catch (Throwable cleanup) { failure = cleanup; }
        }
        try { allocator.close(); } catch (Throwable cleanup) {
          failure = recordFailure(failure, cleanup);
        }
        if (failure != null) {
          failedBatchRoot = root;
          failedBatchAllocator = allocator;
          closeFailure = recordFailure(closeFailure, failure);
          released = true;
          recordFailure(error, failure);
        }
      } finally {
        cursor.unlock();
      }
    }

    @Override
    public void close() {
      Throwable failure = null;
      try { task.close(); } catch (Throwable error) { failure = error; }
      cursor.lock();
      try {
        throwFailure(closeFailure);
        if (!released) {
          released = true;
          try {
            rows = null;
            if (plan != null) plan.cleanupResources();
          } catch (Throwable error) {
            failure = recordFailure(failure, error);
          }
        }
        closeFailure = failure;
        throwFailure(failure);
        session.queries.remove(this);
      } finally {
        cursor.unlock();
      }
    }
  }

  private static byte[] createSchemaIpc(StructType schema, String timeZone, boolean largeVarTypes) {
    return checkedIpc(ArrowConverters$.MODULE$.createEmptyArrowBatch(
        schema, timeZone, false, largeVarTypes));
  }

  // Package-local so tests can cover exact Spark types that earlier Spark planning can normalize.
  static void validateResultType(DataType type) {
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
      for (int i = 0; i < fields.length; i++) {
        StructField field = fields[i];
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
        validateResultType(field.dataType());
      }
    } else if (type instanceof ArrayType) {
      validateResultType(((ArrayType) type).elementType());
    } else if (type instanceof MapType) {
      validateResultType(((MapType) type).keyType());
      validateResultType(((MapType) type).valueType());
    }
  }

  private static byte[] checkedIpc(byte[] bytes) {
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
    if (type.endsWith("CommonInlineUserDefinedFunction") || type.endsWith("CommonInlineUserDefinedTableFunction")) {
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

  private static List<Pair> decodePairs(byte[] bytes) {
    Objects.requireNonNull(bytes, "fields");
    if (bytes.length > MAX_IPC_BYTES) throw new EngineException(8, "config fields exceed 64 MiB");
    try (DataInputStream input = new DataInputStream(new ByteArrayInputStream(bytes))) {
      int count = input.readInt();
      if (count < 0 || count > MAX_CONFIG_FIELDS || count > input.available() / 8) {
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
