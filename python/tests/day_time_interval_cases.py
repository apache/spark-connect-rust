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

Envelope fields follow crates/spark-connect-proto/proto/spark/connect/base.proto;
logical type fields follow types.proto in the same directory. Keep the field
numbers in these helpers aligned with those vendored definitions.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import unittest

try:
    import grpc
    import pyarrow as pa
except ImportError:
    grpc = pa = None

from pyspark.sql import Row, SparkSession, types as T


def _varint(value):
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def _bytes(number, value):
    return _varint(number * 8 + 2) + _varint(len(value)) + value


def _int(number, value):
    return _varint(number * 8) + _varint(value)


def _fields(data):
    def read(pos):
        value = shift = 0
        while True:
            byte = data[pos]
            pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value, pos
            shift += 7

    pos = 0
    result = {}
    while pos < len(data):
        tag, pos = read(pos)
        wire = tag & 7
        if wire == 2:
            size, pos = read(pos)
            result[tag >> 3] = data[pos:pos + size]
            pos += size
        elif wire == 0:
            _, pos = read(pos)
        else:
            raise AssertionError(f"Unexpected wire type: {wire}")
    return result


def _type(data_type):
    # Field numbers are from spark/connect/types.proto.
    if pa.types.is_duration(data_type):
        return _bytes(19, _int(1, 0) + _int(2, 3))
    if pa.types.is_string(data_type):
        return _bytes(11, b"")
    if pa.types.is_struct(data_type):
        fields = b"".join(
            _bytes(1, _bytes(1, f.name.encode()) + _bytes(2, _type(f.type))
                   + _int(3, int(f.nullable))) for f in data_type
        )
        return _bytes(21, fields)
    if pa.types.is_list(data_type):
        return _bytes(20, _bytes(1, _type(data_type.value_type)) + _int(2, 1))
    if pa.types.is_map(data_type):
        return _bytes(22, _bytes(1, _type(data_type.key_type))
                      + _bytes(2, _type(data_type.item_type)) + _int(3, 1))
    raise AssertionError(data_type)


@unittest.skipIf(grpc is None or pa is None, "grpcio and pyarrow are required")
class DayTimeIntervalTests(unittest.TestCase):
    def _has_interval(self, value):
        if isinstance(value, timedelta):
            return True
        if isinstance(value, dict):
            return any(self._has_interval(v) for v in value.values())
        if isinstance(value, list):
            return any(self._has_interval(v) for v in value)
        return False

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
        logical_schema = _type(pa.struct(list(schema)))
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
            fields = _fields(raw)
            header = (_bytes(1, fields[1]) + _bytes(15, b"interval-fixture")
                      + _bytes(12, fields.get(6, b"interval-operation")))
            for i, (count, data) in enumerate(payloads):
                yield (header + _bytes(13, str(i).encode()) + _bytes(7, logical_schema)
                       + _bytes(2, _int(1, count) + _bytes(2, data)))
            yield header + _bytes(13, b"complete") + _bytes(14, b"")

        def analyze(raw, context):
            return (_bytes(1, _fields(raw)[1]) + _bytes(15, b"interval-fixture")
                    + _bytes(2, _bytes(1, logical_schema)))

        def release(raw, context):
            return _bytes(1, _fields(raw)[1])

        with ThreadPoolExecutor(max_workers=2) as pool:
            server = grpc.server(pool)
            server.add_generic_rpc_handlers([grpc.method_handlers_generic_handler(
                "spark.connect.SparkConnectService", {
                    "ExecutePlan": grpc.unary_stream_rpc_method_handler(execute),
                    "AnalyzePlan": grpc.unary_unary_rpc_method_handler(analyze),
                    "ReleaseExecute": grpc.unary_unary_rpc_method_handler(release),
                    "ReleaseSession": grpc.unary_unary_rpc_method_handler(release),
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
                        if self._has_interval(wanted):
                            with self.assertRaisesRegex(TypeError, "decode-only"):
                                spark.createDataFrame([row])
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

    def test_all_null_and_empty(self):
        batch = pa.RecordBatch.from_arrays([
            pa.array([None, None], type=pa.duration("us"))
        ], names=["interval"])
        self._check([batch.slice(0, 0)], [])
        self._check([batch.slice(0, 0), batch], [{"interval": None}, {"interval": None}])


if __name__ == "__main__":
    unittest.main()
