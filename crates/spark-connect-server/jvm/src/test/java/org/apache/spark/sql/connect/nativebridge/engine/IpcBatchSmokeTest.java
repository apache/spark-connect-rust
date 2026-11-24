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

import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.field;
import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.require;
import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.sql;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Method;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.OutOfMemoryException;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.spark.SparkException;
import org.apache.spark.connect.proto.Range;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.scheduler.SparkListener;
import org.apache.spark.scheduler.SparkListenerJobStart;
import org.apache.spark.scheduler.SparkListenerTaskStart;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.util.ArrowUtils$;

/** Spark IPC interoperability, partition order, errors, cancellation and empty nested values. */
public final class IpcBatchSmokeTest {
  private IpcBatchSmokeTest() {}

  public static void main(String[] arguments) throws Exception {
    require(arguments.length == 0 || (arguments.length == 1
        && arguments[0].equals("--result-size-only")), "unknown IPC test arguments");
    SparkSession spark = SparkSession.builder().appName("Native Arrow IPC Smoke")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.artifact.isolation.enabled", "true")
        .config("spark.sql.artifact.isolation.alwaysApplyClassloader", "true")
        .getOrCreate();
    spark.sparkContext().setLogLevel("ERROR");
    try (SparkJvmEngine engine = new SparkJvmEngine(spark)) {
      if (arguments.length == 1) {
        partitionResultSizeFailure(engine);
        System.out.println("IPC_RESULT_SIZE_SMOKE_OK: result-size abort and session recovery");
        return;
      }
      resourceErrorMapping(engine);
      allPartitions(engine);
      wideRowsUseByteEstimate(engine);
      laterPartitionFailure(engine);
      nestedAndEmpty(engine);
      emptyVariableWidthValues(engine, false);
      emptyVariableWidthValues(engine, true);
      cancelSparkWorker(engine, spark);
      require(!spark.sparkContext().isStopped(), "session close stopped shared SparkContext");
      System.out.println("IPC_BATCH_SMOKE_OK: self-contained IPC, partition order, retained bytes, "
          + "nested/decimal/empty/wide values, resource errors, cancellation and session recovery");
    } finally {
      spark.stop();
    }
  }

  private static SparkJvmEngine.Session open(SparkJvmEngine engine) {
    try (SparkJvmEngine.Task task = engine.newTask("IPC smoke open")) {
      return engine.openSession("ipc-smoke", UUID.randomUUID().toString(), task);
    }
  }

  private static SparkJvmEngine.Query prepare(SparkJvmEngine engine,
      SparkJvmEngine.Session session, Relation relation) {
    SparkJvmEngine.Task task = engine.newTask("IPC smoke prepare");
    try {
      return session.prepare(relation.toByteArray(), task);
    } catch (Throwable failure) {
      task.close();
      throw failure;
    }
  }

  private static void config(SparkJvmEngine engine, SparkJvmEngine.Session session,
      String key, String value) throws Exception {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    DataOutputStream out = new DataOutputStream(bytes);
    out.writeInt(1);
    for (String field : new String[] {key, value}) {
      byte[] encoded = field.getBytes(StandardCharsets.UTF_8);
      out.writeInt(encoded.length);
      out.write(encoded);
    }
    try (SparkJvmEngine.Task task = engine.newTask("IPC smoke configuration")) {
      session.config(6, bytes.toByteArray(), task);
    }
  }

  @FunctionalInterface
  private interface BatchCheck { void run(VectorSchemaRoot root) throws Exception; }

  private static void checkBatch(NativeIpcBatch batch, BatchCheck check) throws Exception {
    require(batch != null, "missing IPC batch");
    try (BufferAllocator allocator = ArrowUtils$.MODULE$.rootAllocator()
            .newChildAllocator("ipc-smoke", 0, Long.MAX_VALUE);
         ArrowStreamReader reader = new ArrowStreamReader(
             new ByteArrayInputStream(batch.data()), allocator)) {
      require(reader.loadNextBatch(), "IPC stream contains no record batch");
      VectorSchemaRoot root = reader.getVectorSchemaRoot();
      require(root.getRowCount() == batch.rowCount(), "IPC row count differs from its header");
      check.run(root);
      require(!reader.loadNextBatch(), "IPC stream contains multiple record batches");
    }
  }

  private static void assertSingle(SparkJvmEngine engine, SparkJvmEngine.Session session,
      int expected) throws Exception {
    try (SparkJvmEngine.Query query = prepare(engine, session,
        sql("SELECT " + expected + " AS v"))) {
      checkBatch(query.nextIpcBatch(), root -> require(root.getRowCount() == 1
          && ((Number) root.getVector(0).getObject(0)).intValue() == expected,
          "session recovery result changed"));
      require(query.nextIpcBatch() == null, "recovery query EOF changed");
    }
  }

  private static void resourceErrorMapping(SparkJvmEngine engine) throws Exception {
    Method engineError = SparkJvmEngine.class.getDeclaredMethod("engineError", Exception.class);
    engineError.setAccessible(true);
    SparkJvmEngine.EngineException original =
        new SparkJvmEngine.EngineException(8, "executor IPC exceeds its size limit");
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (ObjectOutputStream output = new ObjectOutputStream(bytes)) {
      output.writeObject(new SparkException("executor failed", original));
    }
    SparkException transported;
    try (ObjectInputStream input = new ObjectInputStream(
        new ByteArrayInputStream(bytes.toByteArray()))) {
      transported = (SparkException) input.readObject();
    }
    SparkJvmEngine.EngineException mapped =
        (SparkJvmEngine.EngineException) engineError.invoke(null, transported);
    require(mapped.code() == 8 && original.getMessage().equals(mapped.getMessage()),
        "executor serialization or Spark wrapper lost RESOURCE_EXHAUSTED");
    OutOfMemoryException allocation = new OutOfMemoryException("Arrow allocation limit");
    SparkJvmEngine.EngineException portable = (SparkJvmEngine.EngineException)
        SparkJvmEngine.executorConversionFailure(new IllegalStateException(allocation));
    bytes.reset();
    try (ObjectOutputStream output = new ObjectOutputStream(bytes)) {
      output.writeObject(portable);
    }
    try (ObjectInputStream input = new ObjectInputStream(
        new ByteArrayInputStream(bytes.toByteArray()))) {
      SparkJvmEngine.EngineException restored = (SparkJvmEngine.EngineException) input.readObject();
      require(restored.code() == 8 && restored.getCause() == null
              && Arrays.equals(restored.getStackTrace(), allocation.getStackTrace()),
          "portable Arrow allocation failure lost its status or executor stack");
    }
    mapped = (SparkJvmEngine.EngineException) engineError.invoke(null, allocation);
    require(mapped.code() == 8 && mapped.getCause() == allocation,
        "driver Arrow allocation failure lost RESOURCE_EXHAUSTED");
    SparkException executorAllocation = new SparkException("executor Arrow allocation failed",
        new IllegalStateException("task failed", allocation));
    mapped = (SparkJvmEngine.EngineException) engineError.invoke(null, executorAllocation);
    require(mapped.code() == 8 && mapped.getCause() == executorAllocation
            && mapped.getMessage().contains(allocation.getMessage()),
        "wrapped executor Arrow allocation failure lost RESOURCE_EXHAUSTED or Spark context");
    executorAllocationFailure(engine);
    try {
      SparkJvmEngine.checkedIpc(new byte[SparkJvmEngine.MAX_IPC_BYTES + 1]);
      throw new AssertionError("executor IPC check accepted a batch over 64 MiB");
    } catch (SparkJvmEngine.EngineException error) {
      require(error.code() == 8, "oversized IPC batch lost RESOURCE_EXHAUSTED");
    }
  }

  private static void executorAllocationFailure(SparkJvmEngine engine) throws Exception {
    try (SparkJvmEngine.Session session = open(engine)) {
      SparkSession sessionSpark = (SparkSession) field(session, "spark").get(session);
      String function = "arrow_allocation_failure_" + UUID.randomUUID().toString().replace("-", "");
      String message = "injected executor Arrow allocation failure";
      // Throw the allocator's ordinary exception on a real executor without exhausting memory.
      sessionSpark.udf().register(function, (UDF1<Long, Long>) value -> {
        throw new OutOfMemoryException(message);
      }, DataTypes.LongType);
      try (SparkJvmEngine.Query query = prepare(engine, session,
          sql("SELECT " + function + "(id) FROM range(0, 1, 1, 1)"))) {
        SparkJvmEngine.EngineException terminal;
        try {
          query.nextIpcBatch();
          throw new AssertionError("executor Arrow allocation fixture returned IPC");
        } catch (SparkJvmEngine.EngineException error) {
          terminal = error;
          if (error.code() != 8 || !(error.getCause() instanceof SparkException)) {
            throw new AssertionError(
                "real executor Arrow allocation lost RESOURCE_EXHAUSTED or its Spark wrapper: code="
                    + error.code() + ", outer cause="
                    + (error.getCause() == null ? "null" : error.getCause().getClass().getName())
                    + ", mapped=" + error, error);
          }
          Throwable cause = error.getCause();
          while (cause != null && !(cause instanceof SparkJvmEngine.EngineException)) {
            cause = cause.getCause();
          }
          require(cause instanceof SparkJvmEngine.EngineException resource
                  && resource.code() == 8 && resource.getCause() == null
                  && resource.getMessage().contains(message),
              "real executor allocation fixture lost its transportable resource error: " + error);
        }
        try {
          query.nextIpcBatch();
          throw new AssertionError("executor Arrow allocation failure became EOF");
        } catch (SparkJvmEngine.EngineException error) {
          require(error == terminal, "executor Arrow allocation replaced its terminal failure");
        }
      }
      assertSingle(engine, session, 13);
    }
  }

  private static void wideRowsUseByteEstimate(SparkJvmEngine engine) throws Exception {
    try (SparkJvmEngine.Session session = open(engine)) {
      config(engine, session, "spark.sql.execution.arrow.maxRecordsPerBatch", "100000");
      String expected = "x".repeat(500000);
      long rows = 0;
      int batches = 0;
      try (SparkJvmEngine.Query query = prepare(engine, session,
          sql("SELECT repeat('x', 500000) AS value FROM range(20)"))) {
        NativeIpcBatch batch;
        while ((batch = query.nextIpcBatch()) != null) {
          // The producer's 2 MiB estimate can overshoot by one wide row. This
          // fixture's 3 MiB allowance is independent of Connect transport chunking.
          require(batch.data().length < 3 * 1024 * 1024,
              "wide-row producer IPC exceeded this fixture's size-estimate allowance");
          checkBatch(batch, root -> {
            for (int row = 0; row < root.getRowCount(); row++) {
              require(expected.equals(root.getVector(0).getObject(row).toString()),
                  "wide string value changed");
            }
          });
          rows += batch.rowCount();
          batches++;
        }
      }
      require(rows == 20 && batches > 1, "byte estimate did not split complete wide-row output");
    }
  }

  private static void allPartitions(SparkJvmEngine engine) throws Exception {
    NativeIpcBatch retained;
    byte[] expected;
    try (SparkJvmEngine.Session session = open(engine)) {
      config(engine, session, "spark.sql.execution.arrow.maxRecordsPerBatch", "3");
      Relation range = Relation.newBuilder().setRange(Range.newBuilder()
          .setStart(0).setEnd(29).setStep(1).setNumPartitions(3)).build();
      long[] rowsAndSum = {0, 0};
      int batches = 0;
      try (SparkJvmEngine.Query query = prepare(engine, session, range)) {
        retained = query.nextIpcBatch();
        require(retained != null, "range returned no batches");
        expected = retained.data().clone();
        NativeIpcBatch batch = retained;
        while (batch != null) {
          require(batch.rowCount() > 0 && batch.rowCount() <= 3, "configured row limit changed");
          checkBatch(batch, root -> {
            for (int row = 0; row < root.getRowCount(); row++) {
              long value = ((Number) root.getVector(0).getObject(row)).longValue();
              require(value == rowsAndSum[0], "partition cursor reordered or lost rows");
              rowsAndSum[0]++;
              rowsAndSum[1] += value;
            }
          });
          batches++;
          require(Arrays.equals(retained.data(), expected), "fetch overwrote prior IPC bytes");
          batch = query.nextIpcBatch();
        }
        require(query.nextIpcBatch() == null, "partition cursor EOF changed");
      }
      require(rowsAndSum[0] == 29 && rowsAndSum[1] == 406 && batches == 11,
          "multiple partitions did not preserve complete range output");
    }
    require(Arrays.equals(retained.data(), expected), "query/session close changed prior IPC");
    checkBatch(retained, root -> require(
        ((Number) root.getVector(0).getObject(2)).longValue() == 2, "retained value changed"));
  }

  private static void laterPartitionFailure(SparkJvmEngine engine) throws Exception {
    try (SparkJvmEngine.Session session = open(engine)) {
      config(engine, session, "spark.sql.execution.arrow.maxRecordsPerBatch", "2");
      Relation values = sql("SELECT CASE WHEN id >= 4 "
          + "THEN CAST(raise_error('later-partition-ipc-error') AS BIGINT) "
          + "ELSE id END AS value FROM range(0, 12, 1, 3)");
      NativeIpcBatch retained = null;
      try (SparkJvmEngine.Query query = prepare(engine, session, values)) {
        SparkJvmEngine.EngineException terminal;
        long[] next = {0};
        try {
          while (true) {
            NativeIpcBatch batch = query.nextIpcBatch();
            require(batch != null, "later partition error was swallowed");
            if (retained == null) retained = batch;
            checkBatch(batch, root -> {
              for (int row = 0; row < root.getRowCount(); row++) {
                require(((Number) root.getVector(0).getObject(row)).longValue() == next[0]++,
                    "failed query returned an invalid prefix");
              }
            });
            require(next[0] <= 4, "failed partition returned rows");
          }
        } catch (SparkJvmEngine.EngineException error) {
          terminal = error;
          require(error.code() == 13 && error.getMessage().contains("later-partition-ipc-error"),
              "later partition error lost its INTERNAL status or message");
        }
        try {
          query.nextIpcBatch();
          throw new AssertionError("failed query became EOF");
        } catch (SparkJvmEngine.EngineException error) {
          require(error == terminal, "failed query replaced its terminal error");
        }
      }
      if (retained != null) {
        checkBatch(retained, root -> require(
            ((Number) root.getVector(0).getObject(0)).longValue() == 0,
            "failed query invalidated prior bytes"));
      }
      assertSingle(engine, session, 9);
    }
  }

  private static void nestedAndEmpty(SparkJvmEngine engine) throws Exception {
    try (SparkJvmEngine.Session session = open(engine)) {
      Relation values = sql("SELECT 'hello' AS text, CAST(NULL AS INT) AS missing, "
          + "array(1, CAST(NULL AS INT), 3) AS items, map('x', 7) AS mapping, "
          + "named_struct('a', 1, 'b', 'v') AS nested, "
          + "CAST('12.34' AS DECIMAL(10,2)) AS amount, "
          + "array(CAST('56.78' AS DECIMAL(10,2))) AS amounts");
      try (SparkJvmEngine.Query query = prepare(engine, session, values)) {
        checkBatch(query.nextIpcBatch(), root -> {
          require(root.getRowCount() == 1, "nested row missing");
          require(root.getVector(0).getObject(0).toString().equals("hello"), "text changed");
          require(root.getVector(1).getObject(0) == null, "null validity changed");
          List<?> items = (List<?>) root.getVector(2).getObject(0);
          require(items.size() == 3 && items.get(1) == null
              && ((Number) items.get(2)).intValue() == 3, "list values changed");
          List<?> mapping = (List<?>) root.getVector(3).getObject(0);
          Map<?, ?> entry = (Map<?, ?>) mapping.get(0);
          require(entry.get("key").toString().equals("x")
              && ((Number) entry.get("value")).intValue() == 7, "map values changed");
          require(((Number) ((Map<?, ?>) root.getVector(4).getObject(0)).get("a"))
              .intValue() == 1, "struct values changed");
          require(new BigDecimal("12.34").equals(root.getVector(5).getObject(0)),
              "decimal changed");
          require(new BigDecimal("56.78").equals(
              ((List<?>) root.getVector(6).getObject(0)).get(0)), "nested decimal changed");
        });
        require(query.nextIpcBatch() == null, "nested query EOF changed");
      }
      try (SparkJvmEngine.Query empty = prepare(engine, session,
          sql("SELECT CAST(NULL AS BIGINT) AS missing WHERE false"))) {
        require(empty.nextIpcBatch() == null, "empty query returned a record batch");
      }
    }
  }

  private static void emptyVariableWidthValues(SparkJvmEngine engine, boolean largeVarTypes)
      throws Exception {
    try (SparkJvmEngine.Session session = open(engine)) {
      config(engine, session, "spark.sql.execution.arrow.maxRecordsPerBatch", "2");
      config(engine, session, "spark.sql.execution.arrow.useLargeVarTypes",
          Boolean.toString(largeVarTypes));
      Relation values = sql("SELECT '' AS text, unhex('') AS binary, "
          + "CAST(array() AS ARRAY<STRING>) AS items, "
          + "array(CAST(array() AS ARRAY<STRING>), CAST(array() AS ARRAY<STRING>)) AS nested, "
          + "array(array('', CAST(NULL AS STRING))) AS strings, "
          + "CAST(map() AS MAP<STRING, ARRAY<STRING>>) AS mapping, "
          + "CAST(NULL AS STRING) AS missing FROM range(5)");
      long rows = 0;
      try (SparkJvmEngine.Query query = prepare(engine, session, values)) {
        NativeIpcBatch batch;
        while ((batch = query.nextIpcBatch()) != null) {
          checkBatch(batch, root -> {
            require((root.getVector(0) instanceof org.apache.arrow.vector.LargeVarCharVector)
                == largeVarTypes, "large variable-width setting changed");
            for (int row = 0; row < root.getRowCount(); row++) {
              require(root.getVector(0).getObject(row).toString().isEmpty(),
                  "empty string changed");
              require(((byte[]) root.getVector(1).getObject(row)).length == 0,
                  "empty binary changed");
              require(((List<?>) root.getVector(2).getObject(row)).isEmpty(), "empty list changed");
              List<?> nested = (List<?>) root.getVector(3).getObject(row);
              require(nested.size() == 2 && ((List<?>) nested.get(0)).isEmpty()
                  && ((List<?>) nested.get(1)).isEmpty(), "nested empty lists changed");
              List<?> strings = (List<?>) ((List<?>) root.getVector(4).getObject(row)).get(0);
              require(strings.size() == 2 && strings.get(0).toString().isEmpty()
                  && strings.get(1) == null, "nested empty/null strings changed");
              require(((List<?>) root.getVector(5).getObject(row)).isEmpty(), "empty map changed");
              require(root.getVector(6).getObject(row) == null, "all-null string changed");
            }
          });
          rows += batch.rowCount();
        }
      }
      require(rows == 5, "empty variable-width batches lost rows");
    }
  }

  private static void cancelSparkWorker(SparkJvmEngine engine, SparkSession spark)
      throws Exception {
    String description = "cancel worker Arrow conversion " + UUID.randomUUID();
    CountDownLatch started = new CountDownLatch(1);
    Set<Integer> ownStages = ConcurrentHashMap.newKeySet();
    SparkListener listener = new SparkListener() {
      @Override
      public void onJobStart(SparkListenerJobStart event) {
        if (description.equals(event.properties().getProperty("spark.job.description"))) {
          scala.collection.Iterator<Object> stages = event.stageIds().iterator();
          while (stages.hasNext()) ownStages.add(((Number) stages.next()).intValue());
        }
      }
      @Override
      public void onTaskStart(SparkListenerTaskStart event) {
        if (ownStages.contains(event.stageId())) started.countDown();
      }
    };
    spark.sparkContext().addSparkListener(listener);
    ExecutorService worker = Executors.newSingleThreadExecutor();
    try (SparkJvmEngine.Session session = open(engine);
         SparkJvmEngine.Task task = engine.newTask(description);
         SparkJvmEngine.Query query = session.prepare(Relation.newBuilder()
             .setRange(Range.newBuilder()
             .setStart(0).setEnd(1000000000000L).setStep(1).setNumPartitions(1))
             .build().toByteArray(), task)) {
      java.util.concurrent.Future<NativeIpcBatch> fetch = worker.submit(query::nextIpcBatch);
      try {
        require(started.await(15, TimeUnit.SECONDS), "worker conversion task did not start");
        task.cancel();
        try {
          fetch.get(15, TimeUnit.SECONDS);
          throw new AssertionError("cancelled worker produced a complete partition");
        } catch (ExecutionException error) {
          require(error.getCause() instanceof SparkJvmEngine.EngineException
              && ((SparkJvmEngine.EngineException) error.getCause()).code() == 1,
              "cancelled Spark worker lost CANCELLED status: " + error.getCause());
        }
      } finally {
        task.cancel();
      }
      assertSingle(engine, session, 11);
    } finally {
      spark.sparkContext().removeSparkListener(listener);
      worker.shutdownNow();
      require(worker.awaitTermination(15, TimeUnit.SECONDS),
          "cancelled worker fetch did not drain");
    }
  }

  private static void partitionResultSizeFailure(SparkJvmEngine engine) throws Exception {
    try (SparkJvmEngine.Session session = open(engine)) {
      try (SparkJvmEngine.Query query = prepare(engine, session,
          sql("SELECT id FROM range(0, 400000, 1, 1)"))) {
        try {
          query.nextIpcBatch();
          throw new AssertionError("partition unexpectedly fit the 1 MiB Spark result-size limit");
        } catch (SparkJvmEngine.EngineException error) {
          require(error.code() == 13, "Spark result-size abort changed its INTERNAL status");
          require(error.getMessage() != null
              && (error.getMessage().contains("maxResultSize")
                  || error.getMessage().contains("serialized results")),
              "result-size fixture failed for a different reason: " + error);
        }
      }
      assertSingle(engine, session, 5);
    }
  }
}
