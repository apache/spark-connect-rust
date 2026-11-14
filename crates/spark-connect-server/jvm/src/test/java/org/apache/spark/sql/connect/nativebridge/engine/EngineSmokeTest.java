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
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.spark.SparkEnv;
import org.apache.spark.connect.proto.CommonInlineUserDefinedFunction;
import org.apache.spark.connect.proto.Expression;
import org.apache.spark.connect.proto.LocalRelation;
import org.apache.spark.connect.proto.Project;
import org.apache.spark.connect.proto.Range;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.connect.proto.SQL;
import org.apache.spark.connect.proto.SubqueryAlias;
import org.apache.spark.connect.proto.WithRelations;
import org.apache.spark.scheduler.SparkListener;
import org.apache.spark.scheduler.SparkListenerJobStart;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSessionExtensions;
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan;
import org.apache.spark.sql.catalyst.rules.Rule;
import org.apache.spark.sql.connect.service.SparkConnectService$;
import org.apache.spark.sql.util.ArrowUtils$;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.MetadataBuilder;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Option;
import scala.runtime.AbstractFunction1;
import scala.runtime.BoxedUnit;

/** Direct Catalyst/Arrow tests. This program does not load the native library. */
public final class EngineSmokeTest {
  private static final AtomicReference<AnalysisGate> ANALYSIS_GATE = new AtomicReference<>();
  private static final AtomicReference<AnalysisJob> ANALYSIS_JOB = new AtomicReference<>();

  private EngineSmokeTest() {}

  public static void main(String[] arguments) throws Exception {
    shutdownHelpers();
    SparkSession spark = SparkSession.builder().appName("Typed JVM Engine Smoke")
        .config("spark.sql.artifact.isolation.enabled", "true")
        .config("spark.sql.artifact.isolation.alwaysApplyClassloader", "true")
        .withExtensions(new AbstractFunction1<SparkSessionExtensions, BoxedUnit>() {
          @Override
          public BoxedUnit apply(SparkSessionExtensions extensions) {
            extensions.injectResolutionRule(new AbstractFunction1<SparkSession, Rule<LogicalPlan>>() {
              @Override
              public Rule<LogicalPlan> apply(SparkSession ruleSession) {
                return new Rule<LogicalPlan>() {
                  @Override
                  public LogicalPlan apply(LogicalPlan plan) {
                    AnalysisGate gate = ANALYSIS_GATE.get();
                    if (gate != null) gate.enter();
                    AnalysisJob job = ANALYSIS_JOB.get();
                    if (job != null) job.enter(ruleSession);
                    return plan;
                  }
                };
              }
            });
            return BoxedUnit.UNIT;
          }
        })
        .getOrCreate();
    spark.sparkContext().setLogLevel("ERROR");
    try (SparkJvmEngine engine = new SparkJvmEngine(spark)) {
      SparkJvmEngine.Session first = open(engine, "session-one");
      SparkJvmEngine.Session second = open(engine, "session-two");
      try {
        metadataOnly(engine, first, second, spark);
        failedControlCallsDoNotCancelJobs(engine, first, spark);
        cancelActiveAnalysisJob(engine, first, spark);
        rangeAndEmpty(engine, first);
        configAtomicityAndIsolation(engine, first, second);
        cancelledBeforePrepare(engine, first);
        commandsCannotMutateCatalog(engine, first);
        earlyCloseKeepsContext(engine, first, second, spark);
        cancelRunningFetch(engine, first, second, spark, true);
        cancelRunningFetch(engine, first, second, spark, false);
        releasedSessionPreparation(engine);
        require(SparkConnectService$.MODULE$.server() == null, "Java gRPC listener was started");
        System.out.println("ENGINE_SMOKE_OK: metadata parity/lifecycle/context/cancellation without jobs; "
            + "Catalyst range and SQL; schema and Arrow batches; "
            + "concurrent metadata/config/preparation readers and exclusive mutations; "
            + "config rollback/isolation; pre-cancel and running fetch cancellation; command/script/"
            + "WITH_RELATIONS rejection; early close and shared-context survival; launcher teardown");
      } finally {
        first.close();
        second.close();
      }
      require(!spark.sparkContext().isStopped(), "session release stopped the shared SparkContext");
    } finally {
      spark.stop();
    }
  }

  private static SparkJvmEngine.Session open(SparkJvmEngine engine, String id) {
    try (SparkJvmEngine.Task task = engine.newTask("open " + id)) {
      return engine.openSession("smoke-user", id, task);
    }
  }

  private static Relation range(long count) {
    return Relation.newBuilder().setRange(Range.newBuilder()
        .setStart(0).setEnd(count).setStep(1).setNumPartitions(2)).build();
  }

  private static Relation sql(String text) {
    return Relation.newBuilder().setSql(SQL.newBuilder().setQuery(text)).build();
  }

  private static SparkJvmEngine.Query query(SparkJvmEngine engine, SparkJvmEngine.Session session,
      Relation relation) {
    SparkJvmEngine.Task task = engine.newTask("prepare smoke query");
    try {
      return session.prepare(relation.toByteArray(), task);
    } catch (Throwable error) {
      task.close();
      throw error;
    }
  }

  private static void rangeAndEmpty(SparkJvmEngine engine, SparkJvmEngine.Session session)
      throws Exception {
    try (SparkJvmEngine.Query query = query(engine, session, range(25))) {
      require(!query.isStreaming(), "range became streaming");
      try (BufferAllocator allocator = allocator();
           ArrowStreamReader reader = new ArrowStreamReader(
               new ByteArrayInputStream(query.schemaIpc()), allocator)) {
        require(reader.getVectorSchemaRoot().getSchema().getFields().get(0).getName().equals("id"),
            "prepared schema is missing the range id");
        require(reader.loadNextBatch(), "schema IPC did not include its empty record batch");
        require(reader.getVectorSchemaRoot().getRowCount() == 0, "prepare executed rows");
      }
      long[] result = consume(query);
      require(Arrays.equals(result, new long[] {25, 300}), "range Arrow values changed");
      require(query.nextVectorBatch() == null, "EOF was not stable");
    }
    try (SparkJvmEngine.Query empty = query(engine, session,
        sql("SELECT CAST(NULL AS BIGINT) AS missing WHERE false"))) {
      require(empty.nextVectorBatch() == null, "empty SQL query returned rows");
    }
    try (SparkJvmEngine.Query join = query(engine, session,
        sql("SELECT sum(a.id + b.id) AS total FROM range(5) a JOIN range(5) b ON a.id = b.id"))) {
      require(Arrays.equals(consume(join), new long[] {1, 20}), "Catalyst SQL join changed");
    }
    try (SparkJvmEngine.Query expression = query(engine, session,
        sql("SELECT id + 1 FROM range(5)"))) {
      require(Arrays.equals(consume(expression), new long[] {5, 15}),
          "internal alias metadata rejected an unaliased computed expression");
    }
    try (SparkJvmEngine.Query aggregate = query(engine, session,
        sql("SELECT sum(id) FROM range(5)"))) {
      require(Arrays.equals(consume(aggregate), new long[] {1, 10}),
          "internal alias metadata rejected an unaliased aggregate");
    }
  }

  private static void failedControlCallsDoNotCancelJobs(SparkJvmEngine engine,
      SparkJvmEngine.Session session, SparkSession spark) throws Exception {
    for (int mode = 0; mode < 3; mode++) {
      SparkJvmEngine.Task task = engine.newTask("failed control cleanup");
      String group = (String) privateField(task, "jobGroup");
      try {
        if (mode == 0) {
          expectCode(3, () -> session.prepare(sql("SELECT FROM").toByteArray(), task));
        } else if (mode == 1) {
          expectCode(3, () -> session.analyze(0, sql("SELECT FROM").toByteArray(), task));
        } else {
          byte[] invalid = encode("spark.sql.shuffle.partitions", "invalid-integer");
          expectCode(3, () -> session.config(6, invalid, task));
        }
        require(!flag(task, "executionStarted"), "failed control call entered result execution");
      } finally {
        task.close();
      }
      // cancelJobGroupAndFutureJobs permanently marks a group in the real DAGScheduler.
      // Reusing this otherwise unused group proves cleanup never submitted that cancellation.
      Properties before = spark.sparkContext().getLocalProperties();
      try {
        spark.sparkContext().setLocalProperties(new Properties());
        spark.sparkContext().setJobGroup(group, "failed control cancellation probe", false);
        require(spark.range(1).repartition(1).collectAsList().size() == 1,
            "failed control cleanup cancelled an unused Spark job group");
      } finally {
        spark.sparkContext().setLocalProperties(before);
      }
    }
  }

  private static final class AnalysisJob {
    private final AtomicBoolean entered = new AtomicBoolean();

    private void enter(SparkSession spark) {
      if (entered.compareAndSet(false, true)) {
        // A real analyzer extension can submit Spark work before result fetching begins.
        // The atomic guard prevents this test's nested query from recursively submitting jobs.
        spark.sql("SELECT sum(id) AS total FROM range(1000000000000)").collectAsList();
      }
    }
  }

  private static void cancelActiveAnalysisJob(SparkJvmEngine engine,
      SparkJvmEngine.Session session, SparkSession spark) throws Exception {
    ExecutorService executor = Executors.newSingleThreadExecutor();
    SparkJvmEngine.Task task = engine.newTask("analysis job cancellation");
    String group = (String) privateField(task, "jobGroup");
    AnalysisJob job = new AnalysisJob();
    require(ANALYSIS_JOB.compareAndSet(null, job), "another analysis job probe is active");
    try {
      Future<byte[]> analysis = executor.submit(
          () -> session.analyze(0, range(1).toByteArray(), task));
      long limit = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
      while (spark.sparkContext().statusTracker().getActiveJobIds().length == 0
          && !analysis.isDone() && System.nanoTime() < limit) Thread.sleep(10);
      require(!analysis.isDone() && spark.sparkContext().statusTracker().getActiveJobIds().length > 0,
          "analysis probe did not submit a real Spark job");
      require(!flag(task, "executionStarted"), "analysis job probe unexpectedly began fetching");
      task.cancel();
      try {
        analysis.get(10, TimeUnit.SECONDS);
        throw new AssertionError("analysis Spark job survived cancellation");
      } catch (ExecutionException expected) {
        require(expected.getCause() instanceof SparkJvmEngine.EngineException
            && ((SparkJvmEngine.EngineException) expected.getCause()).code() == 1,
            "analysis job cancellation returned the wrong status: " + expected.getCause());
      }
      limit = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
      while (spark.sparkContext().statusTracker().getActiveJobIds().length != 0
          && System.nanoTime() < limit) Thread.sleep(10);
      require(spark.sparkContext().statusTracker().getActiveJobIds().length == 0,
          "analysis cancellation left a Spark job running");
      Properties before = spark.sparkContext().getLocalProperties();
      try {
        spark.sparkContext().setLocalProperties(new Properties());
        spark.sparkContext().setJobGroup(group, "cancelled analysis group probe", false);
        try {
          spark.range(1).repartition(1).collectAsList();
          throw new AssertionError("analysis cancellation did not cancel future group submissions");
        } catch (Exception expected) {
          require(expected.getMessage() != null
              && expected.getMessage().toLowerCase(java.util.Locale.ROOT).contains("cancel"),
              "analysis group probe failed for another reason: " + expected);
        }
      } finally {
        spark.sparkContext().setLocalProperties(before);
      }
    } finally {
      ANALYSIS_JOB.compareAndSet(job, null);
      task.close();
      // Ensure a failed assertion cannot leave this deliberately large test job running.
      spark.sparkContext().cancelJobGroupAndFutureJobs(group);
      executor.shutdownNow();
      require(executor.awaitTermination(10, TimeUnit.SECONDS), "analysis job worker did not stop");
    }
  }

  private static void metadataOnly(SparkJvmEngine engine, SparkJvmEngine.Session first,
      SparkJvmEngine.Session second, SparkSession spark) throws Exception {
    AtomicInteger jobs = new AtomicInteger();
    SparkListener listener = new SparkListener() {
      @Override
      public void onJobStart(SparkListenerJobStart event) { jobs.incrementAndGet(); }
    };
    spark.sparkContext().listenerBus().waitUntilEmpty(10000);
    spark.sparkContext().addSparkListener(listener);
    try {
      metadataParity(engine, first);
      metadataFailures(engine, first);
      schemaInformationCannotBeLost(engine, first);
      shuffleCleanupIsRejected(engine, first);
      metadataInvalidModeBoundary(engine, first);
      metadataConfigGetterFailure(engine, first);
      metadataConcurrentReadersAndWriters(engine, first);
      metadataCallerContext(engine, first, second, spark);
      metadataBlocked(engine, first, second, spark, false);
      try (SparkJvmEngine.Session closing = open(engine, "metadata-closing")) {
        metadataBlocked(engine, closing, second, spark, true);
        for (int opcode = 0; opcode < 3; opcode++) {
          int mode = opcode;
          try (SparkJvmEngine.Task task = engine.newTask("closed metadata session")) {
            expectCode(9, () -> closing.analyze(mode, range(1).toByteArray(), task));
          }
        }
      }
      require(registry(first, "queries").isEmpty(), "metadata left a Query registered");
      require(registry(first, "activeTasks").isEmpty(), "metadata left an active Task");
      require(registry(engine, "tasks").isEmpty(), "metadata left a Task registered");
    } finally {
      try {
        spark.sparkContext().listenerBus().waitUntilEmpty(10000);
      } finally {
        spark.sparkContext().removeSparkListener(listener);
      }
    }
    require(jobs.get() == 0, "metadata submitted Spark jobs: " + jobs.get());
  }

  private static void metadataParity(SparkJvmEngine engine, SparkJvmEngine.Session session)
      throws Exception {
    Relation[] relations = {
      range(5),
      Relation.newBuilder().setLocalRelation(LocalRelation.newBuilder().setSchema("id INT")).build(),
      sql("SELECT true AS flag, CAST(1 AS TINYINT) AS tiny, CAST(2 AS SMALLINT) AS small, "
          + "3 AS number, CAST(4 AS BIGINT) AS wide, CAST(1.5 AS FLOAT) AS single, "
          + "CAST(2.5 AS DOUBLE) AS double_value, unhex('00ff') AS binary_value"),
      sql("SELECT 'hello 😀' AS text, CAST(NULL AS STRING) AS missing, NULL AS untyped, "
          + "array(1, CAST(NULL AS INT), 3) AS items, map('x', 7) AS mapping, "
          + "named_struct('a', 1, 'b', 'nested') AS nested"),
      sql("SELECT CAST('12.34' AS DECIMAL(10,2)) AS amount, DATE '2026-01-02' AS day, "
          + "TIMESTAMP '2026-01-02 03:04:05' AS instant, "
          + "CAST('2026-01-02 03:04:05' AS TIMESTAMP_NTZ) AS local_time"),
      sql("SELECT INTERVAL '1-2' YEAR TO MONTH AS months, "
          + "INTERVAL '1 02:03:04' DAY TO SECOND AS duration, make_interval(1) AS calendar"),
      sql("SELECT id FROM range(5) WHERE id < 0")
    };
    boolean[] expectedLocal = {false, true, false, false, false, false, false};
    require(relations.length == expectedLocal.length, "metadata parity expectations are incomplete");
    boolean sawLocal = false;
    boolean sawNonLocal = false;
    for (int index = 0; index < relations.length; index++) {
      Relation relation = relations[index];
      try (SparkJvmEngine.Query prepared = query(engine, session, relation)) {
        boolean local = prepared.isLocal();
        require(local == expectedLocal[index], "prepare IsLocal differs from expected for case " + index);
        sawLocal |= local;
        sawNonLocal |= !local;
        byte[] expected = prepared.schemaIpc();
        for (int opcode = 0; opcode < 3; opcode++) {
          byte[] actual = metadata(engine, session, opcode, relation);
          require(actual[0] == (opcode == 1 && expectedLocal[index] ? 1 : 0),
              "metadata flag differs from expected for case " + index + " mode " + opcode);
          require(Arrays.equals(expected, Arrays.copyOfRange(actual, 1, actual.length)),
              "metadata changed prepare's schema IPC for case " + index + " mode " + opcode);
        }
      }
    }
    require(sawLocal && sawNonLocal, "metadata parity did not cover both IsLocal values");
  }

  private static byte[] metadata(SparkJvmEngine engine, SparkJvmEngine.Session session,
      int opcode, Relation relation) throws Exception {
    int tasksBefore = registry(engine, "tasks").size();
    int queriesBefore = registry(session, "queries").size();
    byte[] result;
    SparkJvmEngine.Task task = engine.newTask("metadata smoke");
    try {
      result = session.analyze(opcode, relation.toByteArray(), task);
      require(result.length > 1 && (result[0] == 0 || result[0] == 1),
          "metadata did not return a strict boolean followed by IPC");
      require(flag(task, "completed"), "successful metadata Task is incomplete");
      task.cancel();
      require(!flag(task, "cancelled"), "completed metadata Task was cancelled");
      require(registry(session, "queries").size() == queriesBefore,
          "metadata registered a Query");
      require(registry(session, "activeTasks").isEmpty(), "metadata Task remained active");
    } finally {
      task.close();
    }
    require(!flag(task, "cancelled"), "closing completed metadata cancelled its job group");
    require(registry(engine, "tasks").size() == tasksBefore, "metadata retained a Task");
    return result;
  }

  private static void schemaInformationCannotBeLost(SparkJvmEngine engine,
      SparkJvmEngine.Session session) throws Exception {
    Metadata tagged = new MetadataBuilder().putString("comment", "must not disappear").build();
    Metadata rawChar = new MetadataBuilder().putString("__autoGeneratedAlias", "true")
        .putString("__CHAR_VARCHAR_TYPE_STRING", "char(4)").build();
    Metadata internalAndUser = new MetadataBuilder().withMetadata(tagged)
        .putString("__autoGeneratedAlias", "true").build();
    DataType[] unsupported = {
      DataTypes.createCharType(4), DataTypes.createVarcharType(4), new VectorUDT(),
      DataTypes.createArrayType(DataTypes.createVarcharType(4)),
      DataTypes.createMapType(DataTypes.StringType, new VectorUDT()),
      new StructType(new StructField[] {
        DataTypes.createStructField("annotated", DataTypes.LongType, true, tagged)
      }),
      DataTypes.createArrayType(new StructType(new StructField[] {
        DataTypes.createStructField("nested", DataTypes.StringType, true, tagged)
      })),
      new StructType(new StructField[] {
        DataTypes.createStructField("raw_char", DataTypes.StringType, true, rawChar)
      }),
      DataTypes.createArrayType(new StructType(new StructField[] {
        DataTypes.createStructField("mixed", DataTypes.StringType, true, internalAndUser)
      }))
    };
    for (DataType type : unsupported) {
      expectCode(12, () -> SparkJvmEngine.validateResultType(type));
    }
    SparkJvmEngine.validateResultType(new StructType(new StructField[] {
      DataTypes.createStructField("plain", DataTypes.createArrayType(
          DataTypes.createMapType(DataTypes.StringType, DataTypes.LongType)), true, Metadata.empty())
    }));
    for (String query : new String[] {
        "SELECT 'a' COLLATE UTF8_LCASE AS value",
        "SELECT array('a' COLLATE UTF8_LCASE) AS value",
        "SELECT map('key', 'a' COLLATE UTF8_LCASE) AS value"}) {
      matchingMetadataFailure(engine, session, sql(query).toByteArray(), 12);
    }
    Relation alias = Relation.newBuilder().setProject(Project.newBuilder().setInput(range(1))
        .addExpressions(Expression.newBuilder().setAlias(Expression.Alias.newBuilder()
            .setExpr(Expression.newBuilder().setUnresolvedAttribute(
                Expression.UnresolvedAttribute.newBuilder().setUnparsedIdentifier("id")))
            .addName("annotated").setMetadata(tagged.json())))).build();
    matchingMetadataFailure(engine, session, alias.toByteArray(), 12);
    SparkSession sessionSpark = (SparkSession) privateField(session, "spark");
    Metadata generated = sessionSpark.sql("SELECT sum(id) FROM range(1)")
        .schema().fields()[0].metadata();
    require(generated.contains("__autoGeneratedAlias")
        && generated.getString("__autoGeneratedAlias").equals("true"),
        "unaliased aggregate did not exercise Spark's generated-alias metadata");
    try (SparkJvmEngine.Query unaliased = query(engine, session,
        sql("SELECT sum(id) FROM range(1)"))) {
      require(unaliased.schemaIpc().length > 0, "internal alias metadata rejected preparation");
    }
    SparkJvmEngine.validateResultType(DataTypes.createArrayType(new StructType(new StructField[] {
      DataTypes.createStructField("internal", DataTypes.StringType, true, generated)
    })));
    require(sessionSpark.sql("SELECT sum(id) AS total FROM range(1)")
        .schema().fields()[0].metadata().isEmpty(),
        "explicit aggregate alias retained unsupported generated metadata");
    String view = "metadata_udt_" + UUID.randomUUID().toString().replace("-", "");
    StructType udtSchema = new StructType(new StructField[] {
      DataTypes.createStructField("features", new VectorUDT(), true)
    });
    sessionSpark.createDataFrame(java.util.Collections.<Row>emptyList(), udtSchema)
        .createOrReplaceTempView(view);
    try {
      matchingMetadataFailure(engine, session, sql("SELECT * FROM " + view).toByteArray(), 12);
    } finally {
      sessionSpark.catalog().dropTempView(view);
    }
  }

  private static void shuffleCleanupIsRejected(SparkJvmEngine engine,
      SparkJvmEngine.Session session) throws Exception {
    String key = "spark.sql.classic.shuffleDependency.fileCleanup.enabled";
    String before = config(engine, session, 4).get(key);
    config(engine, session, 6, key, "true");
    try {
      matchingMetadataFailure(engine, session, range(1).toByteArray(), 12);
    } finally {
      config(engine, session, before == null ? 7 : 6, key, before);
    }
    try (SparkJvmEngine.Query recovered = query(engine, session, range(1))) {
      require(recovered.schemaIpc().length > 0, "shuffle-cleanup rejection poisoned the session");
    }
  }

  private static void releasedSessionPreparation(SparkJvmEngine engine) {
    SparkJvmEngine.Session released = open(engine, "released-prepare");
    released.close();
    try (SparkJvmEngine.Task task = engine.newTask("released preparation")) {
      expectCode(9, () -> released.prepare(range(1).toByteArray(), task));
    }
  }

  private static void metadataFailures(SparkJvmEngine engine, SparkJvmEngine.Session session)
      throws Exception {
    Expression udf = Expression.newBuilder().setCommonInlineUserDefinedFunction(
        CommonInlineUserDefinedFunction.newBuilder().setFunctionName("forbidden_metadata_udf"))
        .build();
    Relation nestedUdf = Relation.newBuilder().setProject(Project.newBuilder().setInput(range(1))
        .addExpressions(Expression.newBuilder().setUnresolvedFunction(
            Expression.UnresolvedFunction.newBuilder().setFunctionName("coalesce")
                .addArguments(udf)))).build();
    Relation references = Relation.newBuilder().setWithRelations(WithRelations.newBuilder()
        .setRoot(sql("CREATE TEMP VIEW metadata_forbidden_refs AS SELECT 1 AS id"))
        .addReferences(Relation.newBuilder().setSubqueryAlias(SubqueryAlias.newBuilder()
            .setAlias("unused_metadata_ref").setInput(range(1))))).build();
    for (Relation relation : new Relation[] {
        sql("CREATE TEMP VIEW metadata_forbidden_command AS SELECT 1 AS id"),
        sql("BEGIN CREATE TEMP VIEW metadata_forbidden_script AS SELECT 1 AS id; END"),
        references, nestedUdf}) {
      matchingMetadataFailure(engine, session, relation.toByteArray(), 12);
    }
    missing(engine, session, "metadata_forbidden_command");
    missing(engine, session, "metadata_forbidden_script");
    missing(engine, session, "metadata_forbidden_refs");
    matchingMetadataFailure(engine, session, sql("SELECT missing FROM range(1)").toByteArray(), 3);
    matchingMetadataFailure(engine, session, sql("SELECT FROM").toByteArray(), 3);
    matchingMetadataFailure(engine, session, new byte[] {0}, 3);
    SparkSession sessionSpark = (SparkSession) privateField(session, "spark");
    sessionSpark.readStream().format("rate").load().createOrReplaceTempView("metadata_stream");
    try {
      matchingMetadataFailure(engine, session, sql("SELECT * FROM metadata_stream").toByteArray(), 12);
    } finally {
      sessionSpark.catalog().dropTempView("metadata_stream");
    }
    for (int opcode = 0; opcode < 3; opcode++) {
      int mode = opcode;
      try (SparkJvmEngine.Task task = engine.newTask("pre-cancel metadata")) {
        task.cancel();
        expectCode(1, () -> session.analyze(mode, range(1).toByteArray(), task));
        require(!flag(task, "completed"), "pre-cancelled metadata completed");
      }
    }
    for (int opcode : new int[] {-1, 3}) {
      try (SparkJvmEngine.Task task = engine.newTask("invalid metadata mode")) {
        expectCode(12, () -> session.analyze(opcode, range(1).toByteArray(), task));
      }
    }
  }

  private static void matchingMetadataFailure(SparkJvmEngine engine,
      SparkJvmEngine.Session session, byte[] relation, int code) throws Exception {
    RuntimeException expected;
    try (SparkJvmEngine.Task task = engine.newTask("prepare expected failure")) {
      try (SparkJvmEngine.Query ignored = session.prepare(relation, task)) {
        throw new AssertionError("prepare unexpectedly accepted invalid metadata input");
      } catch (RuntimeException error) {
        expected = error;
      }
    }
    if (code >= 0) {
      require(expected instanceof SparkJvmEngine.EngineException
          && ((SparkJvmEngine.EngineException) expected).code() == code,
          "prepare produced an unexpected failure: " + expected);
    }
    for (int opcode = 0; opcode < 3; opcode++) {
      try (SparkJvmEngine.Task task = engine.newTask("metadata expected failure")) {
        try {
          session.analyze(opcode, relation, task);
          throw new AssertionError("metadata accepted a plan rejected by prepare");
        } catch (RuntimeException error) {
          require(error.getClass() == expected.getClass(), "metadata changed the failure type");
          if (expected instanceof SparkJvmEngine.EngineException) {
            require(((SparkJvmEngine.EngineException) error).code()
                == ((SparkJvmEngine.EngineException) expected).code(),
                "metadata changed the engine failure code");
          }
        }
        require(!flag(task, "completed"), "failed metadata Task was marked complete");
        require(!flag(task, "executionStarted"), "failed metadata entered query execution");
      }
    }
    require(registry(session, "queries").isEmpty(), "failed metadata retained a Query");
    require(registry(session, "activeTasks").isEmpty(), "failed metadata retained an active Task");
  }

  @SuppressWarnings("unchecked")
  private static void metadataConfigGetterFailure(SparkJvmEngine engine,
      SparkJvmEngine.Session session) throws Exception {
    org.apache.spark.sql.classic.SparkSession sessionSpark =
        (org.apache.spark.sql.classic.SparkSession) privateField(session, "spark");
    Map<String, String> settings = (Map<String, String>) privateField(
        sessionSpark.sessionState().conf(), "settings");
    String key = "spark.sql.execution.arrow.maxRecordsPerBatch";
    // Bypass setter validation only in this test to exercise the shared typed getter boundary.
    String previous = settings.put(key, "invalid-metadata-batch-size");
    try {
      matchingMetadataFailure(engine, session, range(1).toByteArray(), -1);
    } finally {
      if (previous == null) settings.remove(key);
      else settings.put(key, previous);
    }
  }

  private static void metadataCallerContext(SparkJvmEngine engine, SparkJvmEngine.Session first,
      SparkJvmEngine.Session second, SparkSession spark) throws Exception {
    String key = "spark.sql.timestampType";
    String firstBefore = config(engine, first, 4).get(key);
    String secondBefore = config(engine, second, 4).get(key);
    try {
      config(engine, first, 6, key, "TIMESTAMP_LTZ");
      config(engine, second, 6, key, "TIMESTAMP_NTZ");
      Relation relation = sql("SELECT CAST('2026-01-02 03:04:05' AS TIMESTAMP) AS instant");
      byte[] expected;
      try (SparkJvmEngine.Query prepared = query(engine, first, relation)) {
        expected = prepared.schemaIpc();
      }
      SparkSession caller = (SparkSession) privateField(second, "spark");
      for (boolean interrupted : new boolean[] {false, true}) {
        try (SparkJvmEngine.Task task = engine.newTask("metadata caller context")) {
          callerContext(spark, caller, interrupted, () -> {
            byte[] result = first.analyze(0, relation.toByteArray(), task);
            require(Arrays.equals(expected, Arrays.copyOfRange(result, 1, result.length)),
                "metadata used the caller's SQL configuration");
          });
        }
        try (SparkJvmEngine.Task task = engine.newTask("failed metadata caller context")) {
          callerContext(spark, caller, interrupted, () -> expectCode(12, () -> first.analyze(0,
              sql("CREATE TEMP VIEW metadata_context_forbidden AS SELECT 1").toByteArray(), task)));
        }
      }
    } finally {
      config(engine, first, firstBefore == null ? 7 : 6, key, firstBefore);
      config(engine, second, secondBefore == null ? 7 : 6, key, secondBefore);
    }
  }

  private static void callerContext(SparkSession spark, SparkSession caller,
      boolean interrupted, Runnable action) throws Exception {
    Thread thread = Thread.currentThread();
    boolean previousInterrupted = Thread.interrupted();
    ClassLoader previousLoader = thread.getContextClassLoader();
    Properties previousProperties = spark.sparkContext().getLocalProperties();
    SparkEnv previousEnv = SparkEnv.get();
    Option<SparkSession> previousSession = SparkSession.getActiveSession();
    ClassLoader callerLoader = new ClassLoader(previousLoader) {};
    Properties callerProperties = new Properties();
    callerProperties.setProperty("metadata.caller.context", "preserved");
    try (AutoCloseable restore = () -> {
      Thread.interrupted();
      spark.sparkContext().setLocalProperties(previousProperties);
      thread.setContextClassLoader(previousLoader);
      if (previousSession.isDefined()) SparkSession.setActiveSession(previousSession.get());
      else SparkSession.clearActiveSession();
      SparkEnv.set(previousEnv);
      if (previousInterrupted) thread.interrupt();
    }) {
      thread.setContextClassLoader(callerLoader);
      spark.sparkContext().setLocalProperties(callerProperties);
      SparkSession.setActiveSession(caller);
      if (interrupted) thread.interrupt();
      try (AutoCloseable verify = () -> {
        require(thread.getContextClassLoader() == callerLoader, "metadata changed caller classloader");
        require(spark.sparkContext().getLocalProperties().equals(callerProperties),
            "metadata changed caller local properties");
        Option<SparkSession> active = SparkSession.getActiveSession();
        require(active.isDefined() && active.get() == caller, "metadata changed caller active session");
        // SparkEnv is process-global in the pinned Spark assembly. This checks stability of the
        // current global environment, not restoration of a distinct caller environment. Replacing
        // it with a sentinel while Spark background threads run would affect those threads too.
        require(SparkEnv.get() == previousEnv, "metadata changed the current global SparkEnv");
        require(thread.isInterrupted() == interrupted, "metadata changed caller interrupt state");
      }) {
        action.run();
      }
    }
  }

  private static void metadataInvalidModeBoundary(SparkJvmEngine engine,
      SparkJvmEngine.Session session) throws Exception {
    ReentrantReadWriteLock catalog = (ReentrantReadWriteLock) privateField(session, "catalog");
    ExecutorService executor = Executors.newSingleThreadExecutor();
    catalog.writeLock().lock();
    try {
      Future<?> invalid = executor.submit(() -> {
        try (SparkJvmEngine.Task task = engine.newTask("invalid metadata mode")) {
          expectCode(12, () -> session.analyze(3, range(1).toByteArray(), task));
        }
        return null;
      });
      invalid.get(10, TimeUnit.SECONDS);
      try (SparkJvmEngine.Task task = engine.newTask("cancelled invalid metadata mode")) {
        task.cancel();
        expectCode(1, () -> session.analyze(3, range(1).toByteArray(), task));
      }
    } finally {
      catalog.writeLock().unlock();
      executor.shutdownNow();
      require(executor.awaitTermination(10, TimeUnit.SECONDS), "invalid metadata worker did not stop");
    }
  }

  /** Test-only rule barrier: the first analyzer invocation on each thread waits inside Spark. */
  private static final class AnalysisGate implements AutoCloseable {
    private final CountDownLatch entered;
    private final CountDownLatch proceed = new CountDownLatch(1);
    private final Set<Thread> readers = ConcurrentHashMap.newKeySet();
    private final Set<Thread> threads = ConcurrentHashMap.newKeySet();

    private AnalysisGate(int readers) {
      entered = new CountDownLatch(readers);
      require(ANALYSIS_GATE.compareAndSet(null, this), "another analysis gate is active");
    }

    private void enter() {
      Thread thread = Thread.currentThread();
      if (!readers.contains(thread) || !threads.add(thread)) return;
      entered.countDown();
      try {
        require(proceed.await(30, TimeUnit.SECONDS), "analysis gate was not released");
      } catch (InterruptedException error) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException("analysis gate interrupted", error);
      }
    }

    @Override
    public void close() {
      ANALYSIS_GATE.compareAndSet(this, null);
      proceed.countDown();
    }
  }

  private static byte[] concurrentMetadata(SparkJvmEngine engine, SparkJvmEngine.Session session,
      Relation relation) {
    try (SparkJvmEngine.Task task = engine.newTask("concurrent metadata")) {
      return session.analyze(0, relation.toByteArray(), task);
    }
  }

  private static void awaitCatalogWaiter(ReentrantReadWriteLock catalog,
      AtomicReference<Thread> worker, Future<?> result, String operation) {
    long limit = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
    while ((worker.get() == null || !catalog.hasQueuedThread(worker.get()))
        && !result.isDone() && System.nanoTime() < limit) {
      LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(1));
    }
    require(worker.get() != null && catalog.hasQueuedThread(worker.get()) && !result.isDone(),
        operation + " did not wait for active metadata readers");
  }

  private static void metadataConcurrentReadersAndWriters(SparkJvmEngine engine,
      SparkJvmEngine.Session session) throws Exception {
    String key = "spark-connect.concurrent.metadata";
    config(engine, session, 6, key, "before");
    Relation relation = sql("SELECT id FROM range(5)");
    byte[] expected = metadata(engine, session, 0, relation);
    ReentrantReadWriteLock catalog = (ReentrantReadWriteLock) privateField(session, "catalog");
    ExecutorService executor = Executors.newFixedThreadPool(4);
    try (AnalysisGate gate = new AnalysisGate(3)) {
      Future<byte[]> first = executor.submit(() -> {
        gate.readers.add(Thread.currentThread());
        return concurrentMetadata(engine, session, relation);
      });
      Future<byte[]> second = executor.submit(() -> {
        gate.readers.add(Thread.currentThread());
        return concurrentMetadata(engine, session, relation);
      });
      Future<?> prepare = executor.submit(() -> {
        gate.readers.add(Thread.currentThread());
        try (SparkJvmEngine.Query prepared = query(engine, session, relation)) {
          require(prepared.schemaIpc().length > 0, "concurrent prepare lost its schema");
        }
        return null;
      });
      require(gate.entered.await(10, TimeUnit.SECONDS),
          "same-session metadata and preparation did not enter Spark concurrently");

      Future<Map<String, String>> read = executor.submit(
          () -> config(engine, session, 1, key, null));
      require("before".equals(read.get(10, TimeUnit.SECONDS).get(key)),
          "Config read did not overlap metadata or observed a premature mutation");

      AtomicReference<Thread> mutationThread = new AtomicReference<>();
      Future<?> mutation = executor.submit(() -> {
        mutationThread.set(Thread.currentThread());
        config(engine, session, 6, key, "after");
        return null;
      });
      awaitCatalogWaiter(catalog, mutationThread, mutation, "Config mutation");

      gate.close();
      require(Arrays.equals(expected, first.get(10, TimeUnit.SECONDS)),
          "first concurrent metadata changed its schema");
      require(Arrays.equals(expected, second.get(10, TimeUnit.SECONDS)),
          "second concurrent metadata changed its schema");
      mutation.get(10, TimeUnit.SECONDS);
      prepare.get(10, TimeUnit.SECONDS);
      require("after".equals(config(engine, session, 1, key, null).get(key)),
          "Config mutation did not commit after metadata readers drained");
    } finally {
      executor.shutdownNow();
      require(executor.awaitTermination(10, TimeUnit.SECONDS),
          "concurrent metadata workers did not stop");
      config(engine, session, 7, key, null);
    }
    require(registry(session, "activeTasks").isEmpty(), "concurrent metadata retained active Tasks");
    require(registry(session, "queries").isEmpty(), "concurrent prepare retained a Query");
  }

  private static void metadataBlocked(SparkJvmEngine engine, SparkJvmEngine.Session session,
      SparkJvmEngine.Session callerSession, SparkSession spark, boolean closeSession) throws Exception {
    ReentrantReadWriteLock catalog = (ReentrantReadWriteLock) privateField(session, "catalog");
    Set<?> active = registry(session, "activeTasks");
    AtomicBoolean released = (AtomicBoolean) privateField(session, "released");
    SparkSession caller = (SparkSession) privateField(callerSession, "spark");
    AtomicReference<Thread> worker = new AtomicReference<>();
    ExecutorService executor = Executors.newFixedThreadPool(2);
    AtomicReference<Future<?>> close = new AtomicReference<>();
    try (AutoCloseable shutdownExecutor = () -> {
        executor.shutdownNow();
        require(executor.awaitTermination(10, TimeUnit.SECONDS), "metadata workers did not stop");
      };
      AutoCloseable awaitClose = () -> {
        Future<?> closing = close.get();
        if (closing != null) closing.get(10, TimeUnit.SECONDS);
      };
      SparkJvmEngine.Task task = engine.newTask("blocked metadata")) {
      catalog.writeLock().lock();
      try (AutoCloseable unlock = catalog.writeLock()::unlock) {
        Future<?> analysis = executor.submit(() -> {
          worker.set(Thread.currentThread());
          callerContext(spark, caller, false,
              () -> expectCode(1, () -> session.analyze(0, range(1).toByteArray(), task)));
          return null;
        });
        long limit = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while ((worker.get() == null || !active.contains(task)
            || !catalog.hasQueuedThread(worker.get())) && !analysis.isDone()
            && System.nanoTime() < limit) Thread.sleep(5);
        require(worker.get() != null && active.contains(task) && catalog.hasQueuedThread(worker.get()),
            "metadata did not wait on the catalog lock");
        if (closeSession) {
          Future<?> closing = executor.submit(session::close);
          close.set(closing);
          limit = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
          while (!released.get() && !closing.isDone() && System.nanoTime() < limit) Thread.sleep(5);
          require(released.get(), "session close did not publish release");
        } else {
          task.cancel();
        }
        analysis.get(10, TimeUnit.SECONDS);
        require(!flag(task, "completed"), "blocked cancelled metadata completed");
        require(active.isEmpty(), "blocked metadata retained an active Task");
      }
    }
  }

  private static Object privateField(Object object, String name) throws ReflectiveOperationException {
    for (Class<?> type = object.getClass(); type != null; type = type.getSuperclass()) {
      try {
        Field field = type.getDeclaredField(name);
        field.setAccessible(true);
        return field.get(object);
      } catch (NoSuchFieldException missing) {
        // Scala implementation fields can live on a superclass.
      }
    }
    throw new NoSuchFieldException(name);
  }

  private static Set<?> registry(Object object, String name) throws ReflectiveOperationException {
    return (Set<?>) privateField(object, name);
  }

  private static boolean flag(Object object, String name) throws ReflectiveOperationException {
    return (Boolean) privateField(object, name);
  }

  private static long[] consume(SparkJvmEngine.Query query) throws Exception {
    long rows = 0;
    long sum = 0;
    NativeArrowBatch next;
    while ((next = query.nextVectorBatch()) != null) {
      try (NativeArrowBatch batch = next) {
        for (int row = 0; row < batch.descriptor().length(); row++) {
          rows++;
          Object value = batch.valueAt(0, row);
          if (value != null) sum += ((Number) value).longValue();
        }
      }
    }
    return new long[] {rows, sum};
  }

  private static BufferAllocator allocator() {
    return ArrowUtils$.MODULE$.rootAllocator().newChildAllocator("engine-smoke", 0, Long.MAX_VALUE);
  }

  private static Map<String, String> config(SparkJvmEngine engine, SparkJvmEngine.Session session,
      int opcode, String... alternating) throws Exception {
    try (SparkJvmEngine.Task task = engine.newTask("config smoke")) {
      return decode(session.config(opcode, encode(alternating), task));
    }
  }

  private static void configAtomicityAndIsolation(SparkJvmEngine engine,
      SparkJvmEngine.Session first, SparkJvmEngine.Session second) throws Exception {
    config(engine, first, 6, "spark-connect.smoke", "hello 😀");
    require(config(engine, first, 1, "spark-connect.smoke", null).get("spark-connect.smoke").equals("hello 😀"),
        "config UTF-8 changed");
    require(config(engine, second, 2, "spark-connect.smoke", null).get("spark-connect.smoke") == null,
        "configuration leaked between sessions");
    require(config(engine, first, 3, "spark-connect.default", "fallback").get("spark-connect.default").equals("fallback"),
        "config default was ignored");
    String nullableDefault = "spark-connect.null-default." + UUID.randomUUID();
    Map<String, String> nullable = config(engine, first, 3, nullableDefault, null);
    require(nullable.containsKey(nullableDefault) && nullable.get(nullableDefault) == null,
        "missing config with null default did not return null");
    require("hello 😀".equals(config(engine, first, 3, "spark-connect.smoke", null).get("spark-connect.smoke")),
        "null default hid an existing config value");
    String prefix = "spark-connect.prefix." + UUID.randomUUID() + ".";
    String neighboring = prefix.substring(0, prefix.length() - 1) + "-neighbor";
    config(engine, first, 6, prefix + "z", "last", prefix + "a", "first", neighboring, "excluded");
    try {
      Map<String, String> prefixed = config(engine, first, 4, prefix, null);
      require(prefixed.equals(Map.of("a", "first", "z", "last")),
          "GetAll prefix did not filter and strip keys: " + prefixed);
      require(new java.util.ArrayList<>(prefixed.keySet()).equals(java.util.List.of("a", "z")),
          "GetAll prefix keys were not sorted: " + prefixed.keySet());
    } finally {
      config(engine, first, 7, prefix + "z", null, prefix + "a", null, neighboring, null);
    }
    require(config(engine, first, 5, "spark.sql.shuffle.partitions", null)
        .get("spark.sql.shuffle.partitions").equals("true"), "modifiable config is marked static");
    String custom = "spark.connect.native.smoke." + UUID.randomUUID();
    require("false".equals(config(engine, first, 5, custom, null).get(custom)),
        "unregistered key unexpectedly reports a registered SQL config");
    config(engine, first, 6, custom, "unregistered");
    require("unregistered".equals(config(engine, first, 1, custom, null).get(custom)),
        "RuntimeConfig rejected an unregistered spark.* key");
    require(config(engine, second, 2, custom, null).get(custom) == null,
        "unregistered spark.* configuration leaked between sessions");
    Map<String, String> beforeStatic = config(engine, first, 4);
    for (int opcode : new int[] {6, 7}) {
      try {
        config(engine, first, opcode, custom, opcode == 6 ? "changed" : null,
            "spark.sql.warehouse.dir", opcode == 6 ? "/unused-native-smoke" : null);
        throw new AssertionError("static config mutation was accepted");
      } catch (SparkJvmEngine.EngineException expected) {
        require(expected.code() == 3, "static config did not return INVALID_ARGUMENT");
        require(config(engine, first, 4).equals(beforeStatic),
            "failed RuntimeConfig set/unset did not restore an unregistered spark.* key");
      }
    }
    config(engine, first, 7, custom, null);
    require(config(engine, first, 2, custom, null).get(custom) == null,
        "RuntimeConfig could not unset an unregistered spark.* key");
    Map<String, String> before = config(engine, first, 4);
    try {
      config(engine, first, 6, "spark-connect.smoke", "mutated", "spark-connect.rollback.new", "new",
          "spark.sql.shuffle.partitions", "not-a-number");
      throw new AssertionError("invalid config batch was accepted");
    } catch (RuntimeException expected) {
      require(config(engine, first, 4).equals(before), "failed config batch was not rolled back");
    }
    config(engine, first, 7, "spark-connect.smoke", null);
    require(config(engine, first, 2, "spark-connect.smoke", null).get("spark-connect.smoke") == null,
        "config unset failed");
    try {
      config(engine, first, 1, "spark-connect.missing." + UUID.randomUUID(), null);
      throw new AssertionError("missing config key was accepted");
    } catch (SparkJvmEngine.EngineException expected) {
      require(expected.code() == 5, "missing config key did not return NOT_FOUND");
    }
    try {
      config(engine, first, 6, "spark.sql.shuffle.partitions", "invalid-integer");
      throw new AssertionError("invalid config value was accepted");
    } catch (SparkJvmEngine.EngineException expected) {
      require(expected.code() == 3, "invalid config did not return INVALID_ARGUMENT");
    }
  }

  private static void cancelledBeforePrepare(SparkJvmEngine engine, SparkJvmEngine.Session session) {
    try (SparkJvmEngine.Task task = engine.newTask("pre-cancel")) {
      task.cancel();
      expectCode(1, () -> session.prepare(range(1).toByteArray(), task));
    }
  }

  private static void commandsCannotMutateCatalog(SparkJvmEngine engine,
      SparkJvmEngine.Session session) throws Exception {
    reject(engine, session, sql("CREATE TEMP VIEW engine_forbidden_command AS SELECT 1 AS id"));
    missing(engine, session, "engine_forbidden_command");
    reject(engine, session, sql(
        "BEGIN CREATE TEMP VIEW engine_forbidden_script AS SELECT 1 AS id; END"));
    missing(engine, session, "engine_forbidden_script");
    reject(engine, session, Relation.newBuilder().setWithRelations(WithRelations.newBuilder()
        .setRoot(sql("CREATE TEMP VIEW engine_forbidden_refs AS SELECT 1 AS id"))
        .addReferences(Relation.newBuilder().setSubqueryAlias(SubqueryAlias.newBuilder()
            .setAlias("unused_ref").setInput(range(1))))).build());
    missing(engine, session, "engine_forbidden_refs");

    SparkSession sessionSpark = (SparkSession) privateField(session, "spark");
    String table = "engine_write_guard_" + UUID.randomUUID().toString().replace("-", "");
    Path directory = Files.createTempDirectory("engine-write-guard-");
    String location = directory.toUri().toString().replace("'", "''");
    try {
      sessionSpark.sql("CREATE TABLE " + table + " (id BIGINT) USING parquet LOCATION '"
          + location + "'").collect();
      // Prove the target accepts real writes before testing rejection by the adapter.
      sessionSpark.sql("INSERT INTO " + table + " VALUES (7)").collect();
      require(Arrays.equals(tableValues(sessionSpark, table), new long[] {1, 7}),
          "positive INSERT control did not change the target");
      for (String statement : new String[] {
          "INSERT INTO " + table + " VALUES (99)",
          "INSERT OVERWRITE TABLE " + table + " VALUES (99)",
          "WITH source AS (SELECT 99 AS id) INSERT INTO " + table + " SELECT * FROM source",
          "EXECUTE IMMEDIATE 'INSERT INTO " + table + " VALUES (99)'",
          "INSERT OVERWRITE DIRECTORY '" + location + "' USING parquet SELECT 99 AS id",
      }) {
        matchingMetadataFailure(engine, session, sql(statement).toByteArray(), 12);
        // matchingMetadataFailure exercises prepare and every Analyze opcode; opcode zero
        // is also the typed validation used by the native frontend's SqlCommand path.
        require(Arrays.equals(tableValues(sessionSpark, table), new long[] {1, 7}),
            "rejected SQL mutated its target: " + statement);
      }
      System.out.println("ENGINE_WRITE_GUARD_OK: valid INSERT positive control; prepare, schema, "
          + "isLocal, isStreaming and SqlCommand validation leave INSERT/OVERWRITE/dynamic SQL targets unchanged");
    } finally {
      sessionSpark.sql("DROP TABLE IF EXISTS " + table).collect();
      try (var paths = Files.walk(directory)) {
        for (Path path : paths.sorted(java.util.Comparator.reverseOrder()).toList()) {
          Files.deleteIfExists(path);
        }
      }
    }
  }

  private static long[] tableValues(SparkSession spark, String table) {
    Row value = spark.sql("SELECT count(*) AS rows, coalesce(sum(id), 0) AS total FROM "
        + table).collectAsList().get(0);
    return new long[] {value.getLong(0), value.getLong(1)};
  }

  private static void reject(SparkJvmEngine engine, SparkJvmEngine.Session session, Relation relation) {
    try (SparkJvmEngine.Task task = engine.newTask("reject side effects")) {
      expectCode(12, () -> session.prepare(relation.toByteArray(), task));
    }
  }

  private static void missing(SparkJvmEngine engine, SparkJvmEngine.Session session, String table) {
    try (SparkJvmEngine.Task task = engine.newTask("assert missing view")) {
      try (SparkJvmEngine.Query ignored = session.prepare(
          sql("SELECT * FROM " + table).toByteArray(), task)) {
        throw new AssertionError("rejected SQL created view " + table);
      } catch (RuntimeException expected) {
        require(expected.getMessage().contains(table), "unexpected missing-view failure");
      }
    }
  }

  private static void earlyCloseKeepsContext(SparkJvmEngine engine,
      SparkJvmEngine.Session first, SparkJvmEngine.Session second, SparkSession spark)
      throws Exception {
    try (SparkJvmEngine.Query partial = query(engine, first, range(100000))) {
      try (NativeArrowBatch batch = partial.nextVectorBatch()) {
        require(batch != null, "early-close query returned no batch");
      }
    }
    require(!spark.sparkContext().isStopped(), "cursor close stopped the shared SparkContext");
    try (SparkJvmEngine.Query other = query(engine, second, sql("SELECT 7 AS value"))) {
      require(Arrays.equals(consume(other), new long[] {1, 7}), "another session broke after close");
    }
  }

  private static void cancelRunningFetch(SparkJvmEngine engine, SparkJvmEngine.Session first,
      SparkJvmEngine.Session second, SparkSession spark, boolean interrupt) throws Exception {
    ExecutorService executor = Executors.newSingleThreadExecutor();
    SparkJvmEngine.Task task = engine.newTask("running cancellation");
    SparkJvmEngine.Query query = first.prepare(
        sql("SELECT sum(id) AS total FROM range(1000000000000)").toByteArray(), task);
    try {
      Future<NativeArrowBatch> fetch = executor.submit(query::nextVectorBatch);
      long limit = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
      while (spark.sparkContext().statusTracker().getActiveJobIds().length == 0
          && !fetch.isDone() && System.nanoTime() < limit) Thread.sleep(10);
      require(!fetch.isDone(), "long query finished before cancellation");
      require(spark.sparkContext().statusTracker().getActiveJobIds().length != 0,
          "long query did not submit a Spark job");
      if (interrupt) {
        task.cancel();
      } else {
        // Exercise Spark's job-cancellation exception path without a task/thread interrupt.
        // The production cancel() always interrupts too, which would hide this status race.
        Field cancelled = task.getClass().getDeclaredField("cancelled");
        cancelled.setAccessible(true);
        synchronized (task) { cancelled.setBoolean(task, true); }
        spark.sparkContext().cancelJobGroupAndFutureJobs((String) privateField(task, "jobGroup"));
      }
      try {
        try (NativeArrowBatch unexpected = fetch.get(10, TimeUnit.SECONDS)) {
          throw new AssertionError("running fetch survived cancellation");
        }
      } catch (ExecutionException error) {
        require(error.getCause() instanceof SparkJvmEngine.EngineException
            && ((SparkJvmEngine.EngineException) error.getCause()).code() == 1,
            "running cancellation returned the wrong error: " + error.getCause());
      }
    } finally {
      query.close();
      task.close();
      executor.shutdownNow();
      require(executor.awaitTermination(10, TimeUnit.SECONDS), "fetch worker did not stop");
    }
    try (SparkJvmEngine.Query other = query(engine, second, sql("SELECT 3 AS value"))) {
      require(Arrays.equals(consume(other), new long[] {1, 3}), "cancellation broke another session");
    }
  }

  private static void shutdownHelpers() {
    AtomicInteger cleaned = new AtomicInteger();
    CountDownLatch failed = new CountDownLatch(1);
    require(!EmbeddedSparkEngine.shutdown(() -> { throw new IllegalStateException("expected stop failure"); },
        cleaned::incrementAndGet, cleaned::incrementAndGet, failed, false),
        "failed native shutdown was accepted");
    require(failed.getCount() == 0 && cleaned.get() == 0,
        "native failure did not retain resources or release its waiter");
    CountDownLatch startup = new CountDownLatch(1);
    require(EmbeddedSparkEngine.shutdown(() -> {}, cleaned::incrementAndGet,
        cleaned::incrementAndGet, startup, true), "startup cleanup failed");
    require(startup.getCount() == 0 && cleaned.get() == 2, "startup cleanup did not finish");
  }

  private static byte[] encode(String... fields) throws Exception {
    require(fields.length % 2 == 0, "config fields must be pairs");
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    DataOutputStream output = new DataOutputStream(bytes);
    output.writeInt(fields.length / 2);
    for (String value : fields) {
      if (value == null) output.writeInt(-1);
      else {
        byte[] data = value.getBytes(StandardCharsets.UTF_8);
        output.writeInt(data.length);
        output.write(data);
      }
    }
    return bytes.toByteArray();
  }

  private static Map<String, String> decode(byte[] bytes) throws Exception {
    DataInputStream input = new DataInputStream(new ByteArrayInputStream(bytes));
    int count = input.readInt();
    Map<String, String> result = new LinkedHashMap<>();
    for (int i = 0; i < count; i++) result.put(readString(input), readString(input));
    require(input.available() == 0, "trailing config bytes");
    return result;
  }

  private static String readString(DataInputStream input) throws Exception {
    int len = input.readInt();
    return len == -1 ? null : new String(input.readNBytes(len), StandardCharsets.UTF_8);
  }

  private static void expectCode(int code, Runnable action) {
    try {
      action.run();
      throw new AssertionError("operation did not fail with code " + code);
    } catch (SparkJvmEngine.EngineException error) {
      require(error.code() == code, "unexpected engine error: " + error);
    }
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new AssertionError(message);
  }
}
