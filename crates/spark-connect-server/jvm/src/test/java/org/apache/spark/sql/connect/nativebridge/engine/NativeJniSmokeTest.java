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
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowStreamReader;
import org.apache.spark.connect.proto.ExecutePlanRequest;
import org.apache.spark.connect.proto.ExecutePlanResponse;
import org.apache.spark.connect.proto.Plan;
import org.apache.spark.connect.proto.Range;
import org.apache.spark.connect.proto.ReattachOptions;
import org.apache.spark.connect.proto.Relation;
import org.apache.spark.connect.proto.ReleaseSessionRequest;
import org.apache.spark.connect.proto.ResultChunkingOptions;
import org.apache.spark.connect.proto.SQL;
import org.apache.spark.connect.proto.SparkConnectServiceGrpc;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.connect.service.SparkConnectService$;
import org.apache.spark.sql.util.ArrowUtils$;
import org.sparkproject.connect.grpc.ManagedChannel;
import org.sparkproject.connect.grpc.ManagedChannelBuilder;

/** A minimal real start/query/stop JNI check against the pinned Spark assembly. */
public final class NativeJniSmokeTest {
  private NativeJniSmokeTest() {}

  public static void main(String[] arguments) throws Exception {
    require(arguments.length == 1, "pass the absolute native library path");
    SparkSession spark = SparkSession.builder().appName("Native JNI CI smoke")
        .config("spark.sql.artifact.isolation.enabled", "true")
        .config("spark.sql.execution.arrow.compression.codec", "none")
        .config("spark.sql.artifact.isolation.alwaysApplyClassloader", "true")
        .getOrCreate();
    spark.sparkContext().setLogLevel("ERROR");
    try {
      int port;
      SparkJvmEngine engine = new SparkJvmEngine(spark);
      try (engine;
           NativeEngineServer server = new NativeEngineServer(arguments[0], "127.0.0.1:0", engine)) {
        require(SparkConnectService$.MODULE$.server() == null, "Java Connect listener was started");
        String address = server.address();
        require(server.isRunning() && !server.isDrained(),
            "live native listener reported stopped or drained");
        port = Integer.parseInt(address.substring(address.lastIndexOf(':') + 1));
        ManagedChannel channel = ManagedChannelBuilder.forTarget(address).usePlaintext().build();
        try {
          var stub = SparkConnectServiceGrpc.newBlockingStub(channel)
              .withDeadlineAfter(60, TimeUnit.SECONDS);
          String session = UUID.randomUUID().toString();
          String operation = UUID.randomUUID().toString();
          ExecutePlanRequest request = ExecutePlanRequest.newBuilder().setSessionId(session)
              .setOperationId(operation)
              .addRequestOptions(ExecutePlanRequest.RequestOption.newBuilder()
                  .setReattachOptions(ReattachOptions.newBuilder().setReattachable(true)))
              .setPlan(Plan.newBuilder().setRoot(Relation.newBuilder()
                  .setRange(Range.newBuilder().setStart(0).setEnd(25).setStep(1)
                      .setNumPartitions(2)))).build();
          java.util.Iterator<ExecutePlanResponse> responses = stub.executePlan(request);
          long count = 0;
          long sum = 0;
          boolean schema = false;
          boolean complete = false;
          while (responses.hasNext()) {
            ExecutePlanResponse response = responses.next();
            require(!complete, "response followed ResultComplete");
            require(session.equals(response.getSessionId())
                    && operation.equals(response.getOperationId()), "response identity changed");
            if (response.hasSchema()) {
              require(!schema && response.getSchema().hasStruct()
                      && response.getSchema().getStruct().getFieldsCount() == 1
                      && response.getSchema().getStruct().getFields(0).getName().equals("id")
                      && response.getSchema().getStruct().getFields(0).getDataType().hasLong(),
                  "range schema changed");
              schema = true;
            }
            if (response.hasArrowBatch()) {
              require(schema, "Arrow batch preceded the schema");
              long batchRows = 0;
              try (BufferAllocator allocator = ArrowUtils$.MODULE$.rootAllocator()
                       .newChildAllocator("native-jni-ci-smoke", 0, Long.MAX_VALUE);
                   ArrowStreamReader reader = new ArrowStreamReader(new ByteArrayInputStream(
                       response.getArrowBatch().getData().toByteArray()), allocator)) {
                while (reader.loadNextBatch()) {
                  VectorSchemaRoot root = reader.getVectorSchemaRoot();
                  require(root.getFieldVectors().size() == 1, "range Arrow columns changed");
                  for (int row = 0; row < root.getRowCount(); row++) {
                    Object value = root.getVector(0).getObject(row);
                    require(value instanceof Number && ((Number) value).longValue() == count,
                        "range Arrow value/order changed at row " + count);
                    sum += ((Number) value).longValue();
                    count++;
                    batchRows++;
                  }
                }
              }
              require(batchRows == response.getArrowBatch().getRowCount(),
                  "range Arrow row count changed");
            }
            if (response.hasResultComplete()) complete = true;
          }
          require(schema && complete && count == 25 && sum == 300,
              "range result was incomplete or had incorrect values");
          chunkedRange(stub, session);
          wholeLargeRowWithoutChunking(stub, session);
          require(session.equals(stub.withDeadlineAfter(60, TimeUnit.SECONDS)
              .releaseSession(ReleaseSessionRequest.newBuilder().setSessionId(session).build())
              .getSessionId()), "session release identity changed");
        } finally {
          channel.shutdownNow();
          require(channel.awaitTermination(10, TimeUnit.SECONDS), "gRPC channel did not stop");
        }
        server.close();
        require(engine.isClosed(), "native stop did not close the owned engine");
      }
      require(!spark.sparkContext().isStopped(), "native stop closed the shared SparkContext");
      require(spark.range(1).count() == 1, "shared SparkContext stopped serving work");
      try (ServerSocket rebound = new ServerSocket()) {
        rebound.setReuseAddress(true);
        rebound.bind(new InetSocketAddress(InetAddress.getByName("127.0.0.1"), port));
        require(rebound.isBound(), "native listener could not rebind after stop");
      }
      System.out.println("NATIVE_JNI_SMOKE_OK: real JNI start/query/stop; schema and Arrow values; "
          + "ResultComplete; negotiated byte chunks; whole oversized row; "
          + "session release; listener health; listener rebind; "
          + "shared SparkContext survives");
    } finally {
      spark.stop();
    }
  }

  private static void chunkedRange(
      SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub, String session) throws Exception {
    String operation = UUID.randomUUID().toString();
    ExecutePlanRequest request = ExecutePlanRequest.newBuilder().setSessionId(session)
        .setOperationId(operation)
        .addRequestOptions(ExecutePlanRequest.RequestOption.newBuilder()
            .setReattachOptions(ReattachOptions.newBuilder().setReattachable(true)))
        .setPlan(Plan.newBuilder().setRoot(Relation.newBuilder()
            .setRange(Range.newBuilder().setStart(0).setEnd(10000).setStep(1)
                .setNumPartitions(2))))
        .addRequestOptions(ExecutePlanRequest.RequestOption.newBuilder()
            .setResultChunkingOptions(ResultChunkingOptions.newBuilder()
                .setAllowArrowBatchChunking(true).setPreferredArrowChunkSize(1024)))
        .build();
    var responses = stub.withDeadlineAfter(60, TimeUnit.SECONDS).executePlan(request);
    Set<String> responseIds = new HashSet<>();
    ByteArrayOutputStream pending = null;
    long nextIndex = 0;
    long expectedChunks = 0;
    long expectedRows = 0;
    long rows = 0;
    int chunks = 0;
    int batches = 0;
    boolean schema = false;
    boolean complete = false;
    while (responses.hasNext()) {
      ExecutePlanResponse response = responses.next();
      require(!complete, "chunked range response followed ResultComplete");
      require(session.equals(response.getSessionId())
              && operation.equals(response.getOperationId()), "chunked range identity changed");
      require(!response.getResponseId().isEmpty() && responseIds.add(response.getResponseId()),
          "chunked range reused a response ID");
      if (response.hasSchema()) {
        require(!schema, "chunked range repeated its schema header");
        schema = true;
      }
      if (response.hasArrowBatch()) {
        require(schema, "chunked range data preceded its schema header");
        var chunk = response.getArrowBatch();
        require(chunk.hasChunkIndex() && chunk.hasNumChunksInBatch()
                && chunk.hasStartOffset(), "negotiated range omitted chunk metadata");
        require(chunk.getData().size() <= 1024, "range exceeded preferred chunk payload");
        require(chunk.getStartOffset() == rows, "row offset advanced inside a chunked batch");
        require(chunk.getChunkIndex() == nextIndex, "range chunk order changed");
        if (pending == null) {
          pending = new ByteArrayOutputStream();
          expectedChunks = chunk.getNumChunksInBatch();
          expectedRows = chunk.getRowCount();
          require(expectedChunks > 0 && expectedRows > 0, "invalid range batch metadata");
        }
        require(chunk.getNumChunksInBatch() == expectedChunks
                && chunk.getRowCount() == expectedRows, "range metadata changed within a batch");
        chunk.getData().writeTo(pending);
        chunks++;
        nextIndex++;
        require(nextIndex <= expectedChunks, "range sent too many chunks");
        if (nextIndex == expectedChunks) {
          long decodedRows = 0;
          try (BufferAllocator allocator = ArrowUtils$.MODULE$.rootAllocator()
                   .newChildAllocator("native-jni-chunked-range", 0, Long.MAX_VALUE);
               ArrowStreamReader reader = new ArrowStreamReader(
                   new ByteArrayInputStream(pending.toByteArray()), allocator)) {
            require(reader.loadNextBatch(), "reassembled range omitted its Arrow batch");
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            require(root.getFieldVectors().size() == 1, "chunked range column count changed");
            for (int row = 0; row < root.getRowCount(); row++) {
              Object value = root.getVector(0).getObject(row);
              require(value instanceof Number
                      && ((Number) value).longValue() == rows + decodedRows,
                  "reassembled range value/order changed");
              decodedRows++;
            }
            require(!reader.loadNextBatch(), "range IPC contained multiple record batches");
          }
          require(decodedRows == expectedRows, "range batch row count changed after reassembly");
          rows += decodedRows;
          batches++;
          pending = null;
          nextIndex = 0;
        }
      }
      if (response.hasResultComplete()) {
        require(pending == null, "ResultComplete truncated a chunked Arrow batch");
        complete = true;
      }
    }
    require(schema && complete && pending == null && rows == 10000,
        "chunked range was incomplete");
    require(batches > 0 && chunks > batches, "range did not exercise multi-chunk IPC reassembly");
  }

  private static void wholeLargeRowWithoutChunking(
      SparkConnectServiceGrpc.SparkConnectServiceBlockingStub stub, String session) throws Exception {
    String operation = UUID.randomUUID().toString();
    // Omit chunking opt-in. The IPC exceeds 3 MiB, while the complete protobuf
    // message fits this raw client's default 4 MiB receive limit.
    ExecutePlanRequest request = ExecutePlanRequest.newBuilder().setSessionId(session)
        .setOperationId(operation)
        .addRequestOptions(ExecutePlanRequest.RequestOption.newBuilder()
            .setReattachOptions(ReattachOptions.newBuilder().setReattachable(true)))
        .setPlan(Plan.newBuilder().setRoot(Relation.newBuilder()
            .setSql(SQL.newBuilder().setQuery("SELECT repeat('x', 4000000) AS value"))))
        .build();
    var responses = stub.withDeadlineAfter(60, TimeUnit.SECONDS).executePlan(request);
    boolean schema = false;
    boolean complete = false;
    int batches = 0;
    while (responses.hasNext()) {
      ExecutePlanResponse response = responses.next();
      require(!complete, "large-row response followed ResultComplete");
      require(session.equals(response.getSessionId())
              && operation.equals(response.getOperationId()), "large-row response identity changed");
      if (response.hasSchema()) {
        require(!schema, "large-row response repeated its schema header");
        schema = true;
      }
      if (response.hasArrowBatch()) {
        require(schema, "large-row data preceded its schema header");
        var batch = response.getArrowBatch();
        require(!batch.hasChunkIndex() && !batch.hasNumChunksInBatch(),
            "server chunked a batch without client opt-in");
        require(batch.hasStartOffset() && batch.getStartOffset() == 0 && batch.getRowCount() == 1,
            "large-row offset or count changed");
        require(batch.getData().size() > 3 * 1024 * 1024,
            "large-row probe did not exceed 3 MiB");
        try (BufferAllocator allocator = ArrowUtils$.MODULE$.rootAllocator()
                 .newChildAllocator("native-jni-whole-large-row", 0, Long.MAX_VALUE);
             ArrowStreamReader reader = new ArrowStreamReader(
                 new ByteArrayInputStream(batch.getData().toByteArray()), allocator)) {
          require(reader.loadNextBatch(), "large row omitted its Arrow record batch");
          VectorSchemaRoot root = reader.getVectorSchemaRoot();
          require(root.getRowCount() == 1 && root.getFieldVectors().size() == 1,
              "large-row Arrow shape changed");
          String value = root.getVector(0).getObject(0).toString();
          require(value.length() == 4000000 && value.chars().allMatch(character -> character == 'x'),
              "large-row value changed");
          require(!reader.loadNextBatch(), "large-row IPC contained extra record batches");
        }
        batches++;
      }
      if (response.hasResultComplete()) complete = true;
    }
    require(schema && complete && batches == 1, "large row was split or incomplete");
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new AssertionError(message);
  }
}
