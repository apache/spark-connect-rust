# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""SPARK-60607: exact day-time interval results, without a JVM.

The small gRPC fixture sends real Arrow IPC and just the protobuf envelopes
needed by these tests. It does not depend on the reference PySpark package
(which occupies the same import namespace as this client).

Envelopes and logical types are generated from the vendored Spark Connect protos
using protoc, which is also required to build the Rust client.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import unittest
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

try:
    import grpc
    import pyarrow as pa
except ImportError:
    grpc = pa = None

from pyspark.sql import Row, SparkSession, types as T


def _messages():
    proto_root = Path(__file__).resolve().parents[2] / "crates/spark-connect-proto/proto"
    compiler = os.environ.get("PROTOC") or shutil.which("protoc")
    if not compiler:
        raise RuntimeError("protoc is required for the interval fixture")
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "connect.pb"
        command = [compiler, f"-I{proto_root}", "--include_imports",
                   f"--descriptor_set_out={output}"]
        if os.environ.get("PROTOC_INCLUDE"):
            command.append(f"-I{os.environ['PROTOC_INCLUDE']}")
        subprocess.run(command + ["spark/connect/base.proto"], check=True)
        descriptors = descriptor_pb2.FileDescriptorSet.FromString(output.read_bytes())
    pool = descriptor_pool.DescriptorPool()
    for file in descriptors.file:
        pool.Add(file)
    names = ["DataType", "ExecutePlanRequest", "ExecutePlanResponse",
             "AnalyzePlanRequest", "AnalyzePlanResponse", "ReleaseExecuteRequest",
             "ReleaseExecuteResponse", "ReleaseSessionRequest", "ReleaseSessionResponse"]
    return {name: message_factory.GetMessageClass(
        pool.FindMessageTypeByName(f"spark.connect.{name}")) for name in names}


def _type(data_type, message):
    result = message()
    if pa.types.is_duration(data_type):
        result.day_time_interval.start_field = 0
        result.day_time_interval.end_field = 3
    elif pa.types.is_string(data_type):
        result.string.SetInParent()
    elif pa.types.is_struct(data_type):
        for field in data_type:
            result.struct.fields.add(name=field.name, nullable=field.nullable,
                                     data_type=_type(field.type, message))
    elif pa.types.is_list(data_type):
        result.array.element_type.CopyFrom(_type(data_type.value_type, message))
        result.array.contains_null = True
    elif pa.types.is_map(data_type):
        result.map.key_type.CopyFrom(_type(data_type.key_type, message))
        result.map.value_type.CopyFrom(_type(data_type.item_type, message))
        result.map.value_contains_null = True
    else:
        raise AssertionError(data_type)
    return result


@unittest.skipIf(grpc is None or pa is None, "grpcio and pyarrow are required")
class DayTimeIntervalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.messages = _messages()

    def _normalize(self, value):
        if isinstance(value, Row):
            value = value.asDict()
        if isinstance(value, dict):
            return {k: self._normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._normalize(v) for v in value]
        return value

    def _check_schema(self, actual, expected):
        if pa.types.is_duration(expected):
            self.assertIsInstance(actual, T.DayTimeIntervalType)
            self.assertEqual(actual.json(), T.DayTimeIntervalType(0, 3).json())
        elif pa.types.is_struct(expected):
            self.assertIsInstance(actual, T.StructType)
            for field in expected:
                self._check_schema(actual[field.name].dataType, field.type)
        elif pa.types.is_list(expected):
            self._check_schema(actual.elementType, expected.value_type)
        elif pa.types.is_map(expected):
            self._check_schema(actual.valueType, expected.item_type)

    def _check(self, batches, expected):
        schema = batches[0].schema
        logical_schema = _type(pa.struct(list(schema)), self.messages["DataType"])
        payloads = []
        # Each response contains multiple IPC record batches, including empty ones.
        for group in (batches[:2], batches[2:]):
            if not group:
                continue
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, schema) as writer:
                for batch in group:
                    writer.write_batch(batch)
            payloads.append((sum(b.num_rows for b in group), sink.getvalue().to_pybytes()))

        def execute(raw, context):
            request = self.messages["ExecutePlanRequest"].FromString(raw)
            for i, (count, data) in enumerate(payloads):
                response = self.messages["ExecutePlanResponse"](
                    session_id=request.session_id, server_side_session_id="interval-fixture",
                    operation_id=request.operation_id, response_id=str(i), schema=logical_schema)
                response.arrow_batch.row_count = count
                response.arrow_batch.data = data
                yield response.SerializeToString()
            done = self.messages["ExecutePlanResponse"](
                session_id=request.session_id, server_side_session_id="interval-fixture",
                operation_id=request.operation_id, response_id="complete")
            done.result_complete.SetInParent()
            yield done.SerializeToString()

        def analyze(raw, context):
            request = self.messages["AnalyzePlanRequest"].FromString(raw)
            response = self.messages["AnalyzePlanResponse"](
                session_id=request.session_id, server_side_session_id="interval-fixture")
            response.schema.schema.CopyFrom(logical_schema)
            return response.SerializeToString()

        def release_execute(raw, context):
            request = self.messages["ReleaseExecuteRequest"].FromString(raw)
            return self.messages["ReleaseExecuteResponse"](
                session_id=request.session_id).SerializeToString()

        def release_session(raw, context):
            request = self.messages["ReleaseSessionRequest"].FromString(raw)
            return self.messages["ReleaseSessionResponse"](
                session_id=request.session_id).SerializeToString()

        with ThreadPoolExecutor(max_workers=2) as pool:
            server = grpc.server(pool)
            server.add_generic_rpc_handlers([grpc.method_handlers_generic_handler(
                "spark.connect.SparkConnectService", {
                    "ExecutePlan": grpc.unary_stream_rpc_method_handler(execute),
                    "AnalyzePlan": grpc.unary_unary_rpc_method_handler(analyze),
                    "ReleaseExecute": grpc.unary_unary_rpc_method_handler(release_execute),
                    "ReleaseSession": grpc.unary_unary_rpc_method_handler(release_session),
                }
            )])
            port = server.add_insecure_port("127.0.0.1:0")
            server.start()
            spark = None
            try:
                spark = SparkSession.builder.remote(f"sc://127.0.0.1:{port}").create()
                frame = spark.range(1)
                self._check_schema(frame.schema, pa.struct(list(schema)))
                before = frame.schema.json()
                for rows in (frame.collect(), list(frame.toLocalIterator()),
                             list(frame.toLocalIterator(prefetchPartitions=True))):
                    self.assertEqual(len(rows), len(expected))
                    for row, wanted in zip(rows, expected):
                        actual = self._normalize(row)
                        self.assertEqual(actual, wanted)
                        self._assert_timedeltas(actual, wanted)
                self.assertEqual(frame.schema.json(), before)
                self.assertIn("interval", before)
            finally:
                if spark is not None:
                    spark.stop()
                server.stop(0).wait()

    def _assert_timedeltas(self, actual, expected):
        if isinstance(expected, timedelta):
            self.assertIs(type(actual), timedelta)
        elif isinstance(expected, dict):
            for key, value in expected.items():
                self._assert_timedeltas(actual[key], value)
        elif isinstance(expected, list):
            for a, e in zip(actual, expected):
                self._assert_timedeltas(a, e)

    def test_scalars_boundaries_nulls_slices_and_batches(self):
        values = [-(2**63), -(2**63) + 1, -172800000001, -1000001, -1,
                  0, 1, 1000001, 172800000001, 2**63 - 2, 2**63 - 1, None]
        # Slice away non-null sentinels, checking both value and validity offsets.
        array = pa.array([99] + values + [99], type=pa.duration("us")).slice(1, len(values))
        schema = pa.schema([pa.field("interval", pa.duration("us"))])
        batch = pa.RecordBatch.from_arrays([array], schema=schema)
        empty = batch.slice(0, 0)
        self._check([empty, batch.slice(0, 5), empty, batch.slice(5)], [
            {"interval": None if v is None else timedelta(microseconds=v)} for v in values
        ])

    def test_nested_struct_list_map_and_parent_nulls(self):
        duration = pa.duration("us")
        typ = pa.struct([
            pa.field("leaf", duration),
            pa.field("items", pa.list_(duration)),
            pa.field("mapping", pa.map_(pa.string(), pa.list_(duration))),
        ])
        values = [None, {"leaf": -1, "items": [None, -(2**63), 2**63 - 1],
                         "mapping": [("k", [-172800000001, None])]},
                  {"leaf": 0, "items": [], "mapping": []},
                  {"leaf": None, "items": None, "mapping": None}]
        array = pa.array([None] + values + [None], type=typ).slice(1, len(values))
        batch = pa.RecordBatch.from_arrays([array], names=["nested"])
        self._check([batch.slice(0, 0), batch.slice(0, 2), batch.slice(2)], [
            {"nested": None},
            {"nested": {"leaf": timedelta(microseconds=-1),
                        "items": [None, timedelta(microseconds=-(2**63)),
                                  timedelta(microseconds=2**63 - 1)],
                        "mapping": {"k": [timedelta(microseconds=-172800000001), None]}}},
            {"nested": {"leaf": timedelta(0), "items": [], "mapping": {}}},
            {"nested": {"leaf": None, "items": None, "mapping": None}},
        ])



if __name__ == "__main__":
    unittest.main()
