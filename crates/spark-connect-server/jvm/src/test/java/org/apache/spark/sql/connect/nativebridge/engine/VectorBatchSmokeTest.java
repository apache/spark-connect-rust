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

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.OutOfMemoryException;
import org.apache.spark.connect.proto.Range;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.connect.proto.SQL;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.util.ArrowUtils$;

/** Java-side export identity, immutable-batch lifetime, and nested vector layout checks. */
public final class VectorBatchSmokeTest {
  private VectorBatchSmokeTest() {}

  public static void main(String[] arguments) throws Exception {
    SparkSession spark = SparkSession.builder().appName("Native Arrow Vector Smoke")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.artifact.isolation.enabled", "true")
        .config("spark.sql.artifact.isolation.alwaysApplyClassloader", "true")
        .getOrCreate();
    spark.sparkContext().setLogLevel("ERROR");
    try (SparkJvmEngine engine = new SparkJvmEngine(spark)) {
      require(NativeArrowBatch.liveBatchCount() == 0, "unexpected live batch before test");
      immutableLeases(engine);
      nestedAndEmpty(engine);
      emptyVariableWidthVectors(engine);
      boundedAllocationFailure(spark);
      require(NativeArrowBatch.liveBatchCount() == 0, "exported batch leak after tests");
      require(!spark.sparkContext().isStopped(), "session close stopped shared SparkContext");
      System.out.println("VECTOR_BATCH_SMOKE_OK: original off-heap address identity, immutable "
          + "multiple batches, retained values after query/session close, cross-thread/idempotent "
          + "release, nested/string/null/map/decimal/timestamp vectors, empty results and "
          + "all-empty variable-width children with valid offset buffers, "
          + "bounded allocator failure returns RESOURCE_EXHAUSTED and recovers without leaks");
    } finally {
      spark.stop();
    }
  }

  private static SparkJvmEngine.Session open(SparkJvmEngine engine) {
    try (SparkJvmEngine.Task task = engine.newTask("vector smoke open")) {
      return engine.openSession("vector-smoke", UUID.randomUUID().toString(), task);
    }
  }

  private static SparkJvmEngine.Query prepare(SparkJvmEngine engine,
      SparkJvmEngine.Session session, Relation relation) {
    SparkJvmEngine.Task task = engine.newTask("vector smoke prepare");
    try {
      return session.prepare(relation.toByteArray(), task);
    } catch (Throwable failure) {
      task.close();
      throw failure;
    }
  }

  private static Relation sql(String value) {
    return Relation.newBuilder().setSql(SQL.newBuilder().setQuery(value)).build();
  }

  private static void batchRows(SparkJvmEngine engine, SparkJvmEngine.Session session,
      String value) throws Exception {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    DataOutputStream out = new DataOutputStream(bytes);
    out.writeInt(1);
    for (String field : new String[] {"spark.sql.execution.arrow.maxRecordsPerBatch", value}) {
      byte[] encoded = field.getBytes(StandardCharsets.UTF_8);
      out.writeInt(encoded.length);
      out.write(encoded);
    }
    try (SparkJvmEngine.Task task = engine.newTask("vector smoke batch size")) {
      session.config(6, bytes.toByteArray(), task);
    }
  }

  private static void immutableLeases(SparkJvmEngine engine) throws Exception {
    SparkJvmEngine.Session session = open(engine);
    NativeArrowBatch first = null;
    NativeArrowBatch second = null;
    try {
      batchRows(engine, session, "3");
      Relation range = Relation.newBuilder().setRange(Range.newBuilder()
          .setStart(0).setEnd(25).setStep(1).setNumPartitions(2)).build();
      try (SparkJvmEngine.Query query = prepare(engine, session, range)) {
        first = query.nextVectorBatch();
        require(first != null && first.descriptor().length() == 3, "first vector batch size");
        NativeArrowBatch.Node field = first.descriptor().children()[0];
        require(field.buffers().length == 2 && field.buffers()[1] != 0, "range data pointer missing");
        require(field.buffers()[1] == first.vectorBufferAddress(0, 1),
            "exported pointer differs from original ArrowBuf address");
        require(field.bufferSizes()[1] >= 3 * Long.BYTES, "range data buffer too small");
        require(((Number) first.valueAt(0, 0)).longValue() == 0, "first range value");
        second = query.nextVectorBatch();
        require(second != null && ((Number) second.valueAt(0, 0)).longValue() == 3,
            "second batch did not advance");
        require(first.descriptor().children()[0].buffers()[1]
            != second.descriptor().children()[0].buffers()[1], "retained batches reused data memory");
        require(((Number) first.valueAt(0, 0)).longValue() == 0,
            "later fetch overwrote an exported batch");
        require(NativeArrowBatch.liveBatchCount() == 2, "batch leases not counted");
      }
      session.close();
      require(((Number) first.valueAt(0, 2)).longValue() == 2,
          "query/session close freed first batch");
      require(((Number) second.valueAt(0, 2)).longValue() == 5,
          "query/session close freed second batch");
      ExecutorService executor = Executors.newSingleThreadExecutor();
      try {
        NativeArrowBatch release = first;
        executor.submit(release::close).get(10, TimeUnit.SECONDS);
      } finally {
        executor.shutdownNow();
      }
      first.close();
      require(NativeArrowBatch.liveBatchCount() == 1, "idempotent cross-thread release changed count");
      second.close();
      require(NativeArrowBatch.liveBatchCount() == 0, "retained batch release leaked");
    } finally {
      if (first != null) first.close();
      if (second != null) second.close();
      session.close();
    }
  }

  private static void nestedAndEmpty(SparkJvmEngine engine) throws Exception {
    try (SparkJvmEngine.Session session = open(engine)) {
      Relation values = sql("SELECT 'hello 😀' AS text, CAST(NULL AS STRING) AS missing, "
          + "array(1, CAST(NULL AS INT), 3) AS items, map('x', 7) AS mapping, "
          + "named_struct('a', 1, 'b', 'nested') AS nested, "
          + "CAST('12.34' AS DECIMAL(10,2)) AS amount, "
          + "TIMESTAMP '2026-01-02 03:04:05' AS instant");
      try (SparkJvmEngine.Query query = prepare(engine, session, values)) {
        try (NativeArrowBatch batch = query.nextVectorBatch()) {
          require(batch != null && batch.descriptor().length() == 1, "nested row missing");
          NativeArrowBatch.Node[] fields = batch.descriptor().children();
          require(fields.length == 7, "nested schema fields");
          require(batch.valueAt(0, 0).toString().equals("hello 😀"), "UTF-8 value changed");
          require(batch.valueAt(1, 0) == null && fields[1].nullCount() == 1, "null validity changed");
          List<?> items = (List<?>) batch.valueAt(2, 0);
          require(items.size() == 3 && items.get(1) == null
              && ((Number) items.get(2)).intValue() == 3, "list values changed");
          require(fields[2].buffers().length == 2 && fields[2].children().length == 1,
              "list C Data layout");
          List<?> mapping = (List<?>) batch.valueAt(3, 0);
          Map<?, ?> entry = (Map<?, ?>) mapping.get(0);
          require(mapping.size() == 1 && entry.get("key").toString().equals("x")
              && ((Number) entry.get("value")).intValue() == 7, "map values changed");
          require(fields[3].buffers().length == 2 && fields[3].children().length == 1
              && fields[3].children()[0].children().length == 2, "map C Data layout");
          Map<?, ?> nested = (Map<?, ?>) batch.valueAt(4, 0);
          require(((Number) nested.get("a")).intValue() == 1, "struct values changed");
          require(fields[4].buffers().length == 1 && fields[4].children().length == 2,
              "struct C Data layout");
          require(new BigDecimal("12.34").equals(batch.valueAt(5, 0)), "decimal value changed");
          long expectedMicros = java.time.Instant.parse("2026-01-02T03:04:05Z")
              .getEpochSecond() * 1000000L;
          require(((Number) batch.valueAt(6, 0)).longValue() == expectedMicros
              && fields[6].buffers().length == 2, "timestamp value changed");
          for (int i = 0; i < fields.length; i++) {
            for (int j = 0; j < fields[i].buffers().length; j++) {
              require(fields[i].buffers()[j] == batch.vectorBufferAddress(i, j),
                  "top-level vector buffer was copied");
            }
          }
        }
        require(query.nextVectorBatch() == null && query.nextVectorBatch() == null, "EOF changed");
      }
      try (SparkJvmEngine.Query empty = prepare(engine, session,
          sql("SELECT CAST(NULL AS BIGINT) AS missing WHERE false"))) {
        require(empty.nextVectorBatch() == null, "empty query allocated a vector batch");
      }
    }
  }

  private static void emptyVariableWidthVectors(SparkJvmEngine engine) throws Exception {
    try (SparkJvmEngine.Session session = open(engine)) {
      batchRows(engine, session, "2");
      Relation values = sql("SELECT '' AS text, unhex('') AS binary, "
          + "CAST(array() AS ARRAY<STRING>) AS items, "
          + "array(CAST(array() AS ARRAY<STRING>), CAST(array() AS ARRAY<STRING>)) AS nested, "
          + "array(array('', CAST(NULL AS STRING))) AS nested_strings, "
          + "CAST(map() AS MAP<STRING, ARRAY<STRING>>) AS mapping, "
          + "CAST(NULL AS STRING) AS missing FROM range(5)");
      long rows = 0;
      int batches = 0;
      try (SparkJvmEngine.Query query = prepare(engine, session, values)) {
        NativeArrowBatch next;
        while ((next = query.nextVectorBatch()) != null) {
          try (NativeArrowBatch batch = next) {
            long count = batch.descriptor().length();
            require(count > 0 && count <= 2, "empty vector batch row limit changed");
            NativeArrowBatch.Node[] fields = batch.descriptor().children();
            for (int field : new int[] {0, 1, 2, 3, 4, 5, 6}) {
              requireOffsets(fields[field], "empty top-level field " + field);
            }
            NativeArrowBatch.Node itemStrings = fields[2].children()[0];
            require(itemStrings.length() == 0, "empty list allocated child values");
            requireOffsets(itemStrings, "zero-length list string child");
            NativeArrowBatch.Node nestedLists = fields[3].children()[0];
            require(nestedLists.length() == 2 * count, "nested empty lists lost their offsets");
            requireOffsets(nestedLists, "nested empty lists");
            require(nestedLists.children()[0].length() == 0,
                "nested empty lists allocated string values");
            requireOffsets(nestedLists.children()[0], "zero-length nested string child");
            NativeArrowBatch.Node stringLists = fields[4].children()[0];
            requireOffsets(stringLists, "nested string lists");
            requireOffsets(stringLists.children()[0], "all-empty/null string values");
            NativeArrowBatch.Node entries = fields[5].children()[0];
            require(entries.length() == 0, "empty map allocated entries");
            requireOffsets(entries.children()[0], "zero-length map keys");
            requireOffsets(entries.children()[1], "zero-length map lists");
            requireOffsets(entries.children()[1].children()[0], "zero-length map string values");
            for (int row = 0; row < count; row++) {
              require(batch.valueAt(0, row).toString().isEmpty(), "empty string changed");
              require(((byte[]) batch.valueAt(1, row)).length == 0, "empty binary changed");
              require(((List<?>) batch.valueAt(2, row)).isEmpty(), "empty list changed");
              List<?> nested = (List<?>) batch.valueAt(3, row);
              require(nested.size() == 2 && ((List<?>) nested.get(0)).isEmpty()
                  && ((List<?>) nested.get(1)).isEmpty(), "nested empty lists changed");
              List<?> strings = (List<?>) ((List<?>) batch.valueAt(4, row)).get(0);
              require(strings.size() == 2 && strings.get(0).toString().isEmpty()
                  && strings.get(1) == null, "nested empty/null strings changed");
              require(((List<?>) batch.valueAt(5, row)).isEmpty(), "empty map changed");
              require(batch.valueAt(6, row) == null, "all-null string changed");
            }
            rows += count;
            batches++;
          }
        }
      }
      require(rows == 5 && batches >= 3, "empty variable-width batches lost rows");
      require(NativeArrowBatch.liveBatchCount() == 0, "empty variable-width batch leaked");
    }
  }

  private static void requireOffsets(NativeArrowBatch.Node node, String description) {
    // Even a zero-length string/list needs its initial offset for an Arrow C Data import.
    // This checks the actual ArrowWriter output, including descendants with no values.
    require(node.buffers().length >= 2 && node.buffers()[1] != 0
        && node.bufferSizes()[1] >= (node.length() + 1) * Integer.BYTES,
        description + " has no usable offset buffer");
  }

  private static void boundedAllocationFailure(SparkSession spark) {
    // Fail a real Arrow allocation with at most 128 KiB retained off heap, rather than
    // exhausting the shared allocator or building a 64 MiB result just to test its status.
    try (BufferAllocator budget = ArrowUtils$.MODULE$.rootAllocator()
            .newChildAllocator("vector-smoke-bounded-parent", 0, 128 * 1024);
         SparkJvmEngine constrained = new SparkJvmEngine(spark, budget);
         SparkJvmEngine.Session session = open(constrained)) {
      try (SparkJvmEngine.Query large = prepare(constrained, session,
          sql("SELECT repeat('x', 262144) AS value"))) {
        try (NativeArrowBatch ignored = large.nextVectorBatch()) {
          throw new AssertionError("large vector unexpectedly fit its parent allocator quota");
        } catch (SparkJvmEngine.EngineException error) {
          require(error.code() == 8, "Arrow allocator failure did not return RESOURCE_EXHAUSTED");
          require(error.getCause() instanceof OutOfMemoryException,
              "quota test did not reach Arrow's actual allocation failure");
        }
      }
      require(budget.getAllocatedMemory() == 0, "failed vector batch retained Arrow allocations");
      require(NativeArrowBatch.liveBatchCount() == 0, "failed vector batch exported a lease");
      try (SparkJvmEngine.Query small = prepare(constrained, session, sql("SELECT 7 AS value"));
           NativeArrowBatch recovered = small.nextVectorBatch()) {
        require(recovered != null && ((Number) recovered.valueAt(0, 0)).intValue() == 7,
            "engine did not recover after its Arrow allocation failed");
      }
      require(budget.getAllocatedMemory() == 0, "recovery vector batch leaked Arrow allocations");
    }
  }

  private static void require(boolean value, String message) {
    if (!value) throw new AssertionError(message);
  }
}
