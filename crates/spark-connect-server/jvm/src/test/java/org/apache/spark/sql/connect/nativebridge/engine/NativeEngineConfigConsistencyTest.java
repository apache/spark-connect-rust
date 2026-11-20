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
import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.require;
import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.sql;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.internal.config.ConfigEntry;
import org.apache.spark.scheduler.SparkListener;
import org.apache.spark.scheduler.SparkListenerJobStart;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.internal.SQLConf;
import org.apache.spark.sql.internal.SQLConf$;
import org.apache.spark.sql.util.ArrowUtils$;

import scala.runtime.AbstractFunction1;

/** Prepared Arrow settings and unsupported execution-setting mutation checks. */
public final class NativeEngineConfigConsistencyTest {
  private static final String CLEANUP = "spark.sql.classic.shuffleDependency.fileCleanup.enabled";
  private static final String TIME_ZONE = "spark.sql.session.timeZone";
  private static final String MAX_ROWS = "spark.sql.execution.arrow.maxRecordsPerBatch";

  private NativeEngineConfigConsistencyTest() {}

  public static void main(String[] arguments) throws Exception {
    require(arguments.length == 0, "unknown configuration test arguments");
    SparkSession spark = SparkSession.builder().appName("Native engine configuration consistency")
        .config(TIME_ZONE, "UTC")
        // Exercise inherited host configuration before repairing it through the public API.
        .config(CLEANUP, "true")
        .config("spark.sql.artifact.isolation.enabled", "true")
        .config("spark.sql.artifact.isolation.alwaysApplyClassloader", "true")
        .getOrCreate();
    spark.sparkContext().setLogLevel("ERROR");
    try (SparkJvmEngine engine = new SparkJvmEngine(spark);
         SparkJvmEngine.Session session = open(engine)) {
      inheritedSettingIsRejected(engine, session, spark);
      preparedArrowSettingsAreFrozen(engine, session);
      mutationIsAtomic(engine, session);
      silentSetMissingValueIsAWarning(engine, session);
      setterErrorsAreNotWarnings(engine, session);
      unsetChecksResolvedDefault(engine, session);
      require(!spark.sparkContext().isStopped(), "query cleanup stopped the shared SparkContext");
      System.out.println("NATIVE_ENGINE_CONFIG_CONSISTENCY_OK: inherited-setting guard; "
          + "atomic Set rollback, default-aware Unset and SilentSet warnings including absent values; "
          + "original setter Error propagation through Set and SilentSet; "
          + "prepared Arrow schema/row limits");
    } finally {
      spark.stop();
    }
  }

  private static SparkJvmEngine.Session open(SparkJvmEngine engine) {
    try (SparkJvmEngine.Task task = engine.newTask("configuration consistency open")) {
      return engine.openSession("config-consistency", UUID.randomUUID().toString(), task);
    }
  }

  private static SparkJvmEngine.Query prepare(SparkJvmEngine engine,
      SparkJvmEngine.Session session, Relation relation) {
    SparkJvmEngine.Task task = engine.newTask("configuration consistency prepare");
    try {
      return session.prepare(relation.toByteArray(), task);
    } catch (Throwable error) {
      task.close();
      throw error;
    }
  }

  private static void inheritedSettingIsRejected(SparkJvmEngine engine,
      SparkJvmEngine.Session session, SparkSession spark) throws Exception {
    require("true".equals(config(engine, session, 1, CLEANUP, null).get(CLEANUP)),
        "fixture did not inherit the unsupported host setting");
    try (JobProbe jobs = new JobProbe(spark)) {
      expectCode(12, () -> {
        try (SparkJvmEngine.Query ignored = prepare(engine, session, sql("SELECT 1"))) {
          throw new AssertionError("unsupported inherited setting was prepared");
        }
      });
      for (int opcode = 0; opcode < 3; opcode++) {
        int mode = opcode;
        try (SparkJvmEngine.Task task = engine.newTask("unsupported inherited analysis")) {
          expectCode(12, () -> session.analyze(mode, sql("SELECT 1").toByteArray(), task));
        }
      }
      jobs.assertNone();
    }
    require(config(engine, session, 6, CLEANUP, "false").isEmpty(),
        "setting the supported cleanup value returned a warning");
    require("false".equals(config(engine, session, 1, CLEANUP, null).get(CLEANUP)),
        "supported cleanup value was not applied");
  }

  private static void preparedArrowSettingsAreFrozen(SparkJvmEngine engine,
      SparkJvmEngine.Session session) throws Exception {
    config(engine, session, 6, TIME_ZONE, "UTC", MAX_ROWS, "2");
    try (SparkJvmEngine.Query query = prepare(engine, session,
        sql("SELECT id, TIMESTAMP '2020-01-02 03:04:05' AS ts FROM range(0, 5, 1, 1)"))) {
      Schema announced;
      try (BufferAllocator allocator = allocator();
           ArrowStreamReader reader = new ArrowStreamReader(
               new ByteArrayInputStream(query.schemaIpc()), allocator)) {
        announced = reader.getVectorSchemaRoot().getSchema();
      }
      config(engine, session, 6, TIME_ZONE, "Asia/Tokyo", MAX_ROWS, "1");
      long next = 0;
      int batches = 0;
      NativeIpcBatch batch;
      while ((batch = query.nextIpcBatch()) != null) {
        require(batch.rowCount() == (batches < 2 ? 2 : 1),
            "a later Config Set changed the prepared Arrow row limit");
        try (BufferAllocator allocator = allocator();
             ArrowStreamReader reader = new ArrowStreamReader(
                 new ByteArrayInputStream(batch.data()), allocator)) {
          require(announced.equals(reader.getVectorSchemaRoot().getSchema()),
              "a later Config Set changed the announced Arrow schema");
          require(reader.loadNextBatch(), "prepared output has no Arrow record batch");
          var root = reader.getVectorSchemaRoot();
          require(root.getRowCount() == batch.rowCount(), "Arrow batch row count changed");
          for (int row = 0; row < root.getRowCount(); row++) {
            require(((Number) root.getVector(0).getObject(row)).longValue() == next++,
                "prepared output reordered or lost rows");
            require(!root.getVector(1).isNull(row), "prepared timestamp value became null");
          }
          require(!reader.loadNextBatch(), "prepared IPC contains multiple record batches");
        }
        batches++;
      }
      require(next == 5 && batches == 3 && query.nextIpcBatch() == null,
          "prepared Arrow settings changed output or EOF");
    }
  }

  private static void mutationIsAtomic(SparkJvmEngine engine,
      SparkJvmEngine.Session session) throws Exception {
    String marker = "spark.connect.native.configConsistency.marker";
    String unvisited = "spark.connect.native.configConsistency.unvisited";
    config(engine, session, 6, marker, "original", CLEANUP, "false");
    Map<String, String> before = config(engine, session, 4);
    for (String value : new String[] {"true", "TRUE", " true", "true ", "\ttrue", " \tTrUe \n"}) {
      expectCode(12, () -> config(engine, session, 6, marker, "changed", TIME_ZONE, "UTC",
          CLEANUP, value, unvisited, "must-not-be-applied"));
      require(before.equals(config(engine, session, 4)),
          "rejected execution setting did not roll back the complete Set batch");
      Map<String, String> warning = config(engine, session, 8, CLEANUP, value);
      require(warning.equals(Map.of(CLEANUP, "UNSUPPORTED_EXECUTION_SETTING")),
          "SilentSet did not return its stable setter-failure warning");
      require(before.equals(config(engine, session, 4)),
          "SilentSet did not restore the rejected execution setting");
    }
    require(config(engine, session, 8, CLEANUP, "false").isEmpty(),
        "SilentSet rejected the supported execution setting");
  }

  private static void silentSetMissingValueIsAWarning(SparkJvmEngine engine,
      SparkJvmEngine.Session session) throws Exception {
    String registered = "spark.sql.shuffle.partitions";
    String unregistered = "spark.review.silentSet.absent." + UUID.randomUUID();
    Map<String, String> initial = config(engine, session, 4);
    try {
      for (String key : new String[] {registered, unregistered}) {
        config(engine, session, 6, key, "17");
        Map<String, String> before = config(engine, session, 4);
        require(config(engine, session, 8, key, null)
                .equals(Map.of(key, "java.lang.IllegalArgumentException")),
            "SilentSet absent value did not preserve its sanitized setter warning");
        require(before.equals(config(engine, session, 4)),
            "SilentSet absent value changed configuration");
        expectCode(3, () -> config(engine, session, 6, key, null));
        require(before.equals(config(engine, session, 4)),
            "Set absent value changed configuration");
        require(config(engine, session, 8, key, "19").isEmpty()
                && "19".equals(config(engine, session, 1, key, null).get(key)),
            "valid SilentSet did not recover after an absent value");
        require(config(engine, session, 6, key, "23").isEmpty()
                && "23".equals(config(engine, session, 1, key, null).get(key)),
            "valid Set did not recover after an absent value");
      }
    } finally {
      for (String key : new String[] {registered, unregistered}) {
        config(engine, session, initial.containsKey(key) ? 6 : 7, key, initial.get(key));
      }
    }
    require(initial.equals(config(engine, session, 4)),
        "absent-value configuration fixture did not restore its initial state");
  }

  private static void setterErrorsAreNotWarnings(SparkJvmEngine engine,
      SparkJvmEngine.Session session) throws Exception {
    String key = "spark.sql.native.configConsistency.error." + UUID.randomUUID();
    String marker = key + ".marker";
    AssertionError expected = new AssertionError("harmless configuration validator fixture");
    ConfigEntry<String> entry = SQLConf$.MODULE$.buildConf(key).stringConf()
        .checkValue(new AbstractFunction1<String, Object>() {
          @Override
          public Object apply(String value) {
            if ("fixture-error".equals(value)) throw expected;
            return true;
          }
        }, "configuration fixture value must be safe").createWithDefault("safe");
    try {
      config(engine, session, 6, marker, "original", key, "safe");
      Map<String, String> before = config(engine, session, 4);
      for (int opcode : new int[] {6, 8}) {
        // Both paths must propagate the exact Error. Only normal Set changes prior state
        // before the validator throws and therefore exercises rollback of an accepted write.
        // SilentSet's one-key validator fails before storing its value.
        String[] changes = opcode == 6
            ? new String[] {marker, "changed", key, "fixture-error"}
            : new String[] {key, "fixture-error"};
        try {
          config(engine, session, opcode, changes);
          throw new AssertionError("configuration setter Error became a successful response");
        } catch (AssertionError actual) {
          require(actual == expected, "configuration setter replaced its original Error");
        }
        require(before.equals(config(engine, session, 4)),
            opcode == 6 ? "normal Set did not roll back accepted writes after setter Error"
                : "SilentSet validator Error changed the prior configuration");
      }
      try (SparkJvmEngine.Query recovered = prepare(engine, session, sql("SELECT 7 AS value"))) {
        NativeIpcBatch batch = recovered.nextIpcBatch();
        require(batch != null && batch.rowCount() == 1, "setter Error poisoned the session");
        try (BufferAllocator allocator = allocator();
             ArrowStreamReader reader = new ArrowStreamReader(
                 new ByteArrayInputStream(batch.data()), allocator)) {
          require(reader.loadNextBatch()
                  && Integer.valueOf(7).equals(reader.getVectorSchemaRoot().getVector(0).getObject(0)),
              "session recovery after setter Error returned different values");
          require(!reader.loadNextBatch(), "session recovery returned extra record batches");
        }
        require(recovered.nextIpcBatch() == null, "session recovery returned extra IPC batches");
      }
    } finally {
      try {
        config(engine, session, 7, marker, null, key, null);
      } finally {
        SQLConf$.MODULE$.unregister(entry);
      }
    }
  }

  private static void unsetChecksResolvedDefault(SparkJvmEngine engine,
      SparkJvmEngine.Session session) throws Exception {
    String marker = "spark.connect.native.configConsistency.unset";
    config(engine, session, 6, marker, "original", CLEANUP, "false");
    boolean defaultEnabled = Boolean.parseBoolean(new SQLConf().getConfString(CLEANUP));
    if (defaultEnabled) {
      Map<String, String> before = config(engine, session, 4);
      expectCode(12, () -> config(engine, session, 7, marker, null, CLEANUP, null));
      require(before.equals(config(engine, session, 4)),
          "Unset to an unsupported default did not restore the complete mutation batch");
    } else {
      config(engine, session, 7, marker, null, CLEANUP, null);
      require("false".equals(config(engine, session, 1, CLEANUP, null).get(CLEANUP)),
          "Unset did not restore the supported registered default");
      Map<String, String> explicit = config(engine, session, 4);
      require(!explicit.containsKey(CLEANUP) && !explicit.containsKey(marker),
          "successful Unset retained explicit configuration values");
    }
    System.out.println("CONFIG_UNSET_DEFAULT: " + defaultEnabled
        + (defaultEnabled ? " (rejection and atomic rollback checked)" : " (supported unset checked)"));
    config(engine, session, 6, CLEANUP, "false");
  }

  private static BufferAllocator allocator() {
    return ArrowUtils$.MODULE$.rootAllocator()
        .newChildAllocator("config-consistency", 0, Long.MAX_VALUE);
  }

  private static final class JobProbe extends SparkListener implements AutoCloseable {
    private final SparkSession spark;
    private final AtomicInteger starts = new AtomicInteger();

    private JobProbe(SparkSession spark) throws Exception {
      this.spark = spark;
      spark.sparkContext().listenerBus().waitUntilEmpty(10000);
      spark.sparkContext().addSparkListener(this);
    }

    @Override
    public void onJobStart(SparkListenerJobStart event) { starts.incrementAndGet(); }

    private void assertNone() throws Exception {
      spark.sparkContext().listenerBus().waitUntilEmpty(10000);
      require(starts.get() == 0, "unsupported execution setting submitted a Spark job");
    }

    @Override
    public void close() { spark.sparkContext().removeSparkListener(this); }
  }

  private static Map<String, String> config(SparkJvmEngine engine,
      SparkJvmEngine.Session session, int opcode, String... fields) throws Exception {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (DataOutputStream output = new DataOutputStream(bytes)) {
      require(fields.length % 2 == 0, "configuration fields must be pairs");
      output.writeInt(fields.length / 2);
      for (String field : fields) {
        if (field == null) output.writeInt(-1);
        else {
          byte[] encoded = field.getBytes(StandardCharsets.UTF_8);
          output.writeInt(encoded.length);
          output.write(encoded);
        }
      }
    }
    try (SparkJvmEngine.Task task = engine.newTask("configuration consistency config");
         DataInputStream input = new DataInputStream(new ByteArrayInputStream(
             session.config(opcode, bytes.toByteArray(), task)))) {
      Map<String, String> result = new LinkedHashMap<>();
      int count = input.readInt();
      for (int i = 0; i < count; i++) result.put(readString(input), readString(input));
      require(input.available() == 0, "trailing configuration response bytes");
      return result;
    }
  }

  private static String readString(DataInputStream input) throws Exception {
    int length = input.readInt();
    return length == -1 ? null : new String(input.readNBytes(length), StandardCharsets.UTF_8);
  }

  private static SparkJvmEngine.EngineException expectCode(int code, CheckedAction action)
      throws Exception {
    try {
      action.run();
      throw new AssertionError("operation did not fail with code " + code);
    } catch (SparkJvmEngine.EngineException error) {
      require(error.code() == code, "unexpected engine error: " + error);
      return error;
    }
  }
}
