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

"""Check integer tuple input and complete IPC values without a Spark server or JVM."""

from concurrent.futures import ThreadPoolExecutor
import json
import unittest

try:
    import grpc
    import pandas as pd
    import pyarrow as pa
except ImportError:
    grpc = pd = pa = None

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


def _fields(data):
    """Read the varint/bytes envelopes used by the fixture's protobuf messages."""
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
            result.setdefault(tag >> 3, []).append(data[pos:pos + size])
            pos += size
        elif wire == 0:
            _, pos = read(pos)
        else:
            raise AssertionError(f"Unexpected protobuf wire type: {wire}")
    return result


@unittest.skipIf(grpc is None or pa is None, "grpcio, pandas and pyarrow are required")
class IntegerTupleTests(unittest.TestCase):
    def setUp(self):
        self.captured = []
        self.config_calls = 0

        def execute(raw, context):
            # ExecutePlanRequest.plan(3) -> Plan.root(1) -> Relation.local_relation(11).
            request = _fields(raw)
            root = _fields(_fields(request[3][0])[1][0])
            self.captured.append(_fields(root[11][0]))
            yield (_bytes(1, request[1][0]) + _bytes(15, b"input-name-fixture")
                   + _bytes(12, request[6][0]) + _bytes(13, b"complete")
                   + _bytes(14, b""))

        def config(raw, context):
            self.config_calls += 1
            request = _fields(raw)
            operation = _fields(request[3][0])
            if 2 in operation:  # ConfigRequest.Get.keys(1).
                keys = _fields(operation[2][0])[1]
            else:
                self.assertEqual(set(operation), {3})  # GetWithDefault.pairs(1).
                keys = [_fields(pair)[1][0] for pair in _fields(operation[3][0])[1]]
            values = {
                b"spark.sql.session.localRelationCacheThreshold": str(1 << 60).encode(),
                b"spark.sql.connect.serverStacktrace.enabled": b"false",
                b"spark.sql.pyspark.jvmStacktrace.enabled": b"false",
            }
            pairs = b"".join(_bytes(2, _bytes(1, key) + _bytes(2, values[key]))
                             for key in keys)
            return _bytes(1, request[1][0]) + _bytes(4, b"input-name-fixture") + pairs

        def release(raw, context):
            return _bytes(1, _fields(raw)[1][0])

        pool = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(pool.shutdown)
        server = grpc.server(pool)
        server.add_generic_rpc_handlers([grpc.method_handlers_generic_handler(
            "spark.connect.SparkConnectService", {
                "ExecutePlan": grpc.unary_stream_rpc_method_handler(execute),
                "Config": grpc.unary_unary_rpc_method_handler(config),
                "ReleaseExecute": grpc.unary_unary_rpc_method_handler(release),
                "ReleaseSession": grpc.unary_unary_rpc_method_handler(release),
            }
        )])
        port = server.add_insecure_port("127.0.0.1:0")
        self.assertNotEqual(port, 0)
        server.start()
        self.addCleanup(lambda: server.stop(0).wait())
        self.spark = SparkSession.builder.remote(f"sc://127.0.0.1:{port}").create()
        self.addCleanup(self.spark.stop)


    def capture(self, data, schema):
        before = len(self.captured)
        config_before = self.config_calls
        frame = self.spark.createDataFrame(data, schema)
        self.assertEqual(len(self.captured), before)  # construction remains offline
        self.assertEqual(self.config_calls, config_before)
        frame.toArrow()
        self.assertEqual(len(self.captured), before + 1)
        self.assertEqual(self.config_calls, config_before)
        local = self.captured[-1]
        self.assertEqual(json.loads(local[2][0]), json.loads(schema.json()))
        if 1 not in local:
            self.assertEqual(len(data), 0)
            return None
        table = pa.ipc.open_stream(local[1][0]).read_all()
        table.validate(full=True)
        return table

    def test_complete_columns_schema_and_metadata(self):
        for width in (1, 3, 32):
            for nullable in (False, True):
                names = [f"雪{j}" for j in range(width)]
                if width == 3:
                    names = ["duplicate", "duplicate", "embedded\x00nul"]
                schema = T.StructType([T.StructField(name, T.LongType(), nullable,
                    {"index": j, "nested": {"v": "raw"}}) for j, name in enumerate(names)])
                data = [tuple(None if nullable and (i+j) % 5 == 0 else
                    (-(1 << 63) if i == 0 else (1 << 63)-1 if i == 1 else i*width+j)
                    for j in range(width)) for i in range(61)]
                with self.subTest(width=width, nullable=nullable):
                    table = self.capture(data, schema)
                    expected = pa.Table.from_arrays([
                        pa.array([row[j] for row in data], type=pa.int64())
                        for j in range(width)], schema=pa.schema([
                            pa.field(name, pa.int64(), nullable=nullable) for name in names]))
                    self.assertEqual(table.schema, expected.schema)
                    self.assertTrue(table.equals(expected))

    def test_empty_and_all_null(self):
        schema = T.StructType([T.StructField("v", T.LongType())])
        self.assertIsNone(self.capture([], schema))
        self.assertEqual(self.capture([(None,), (None,)], schema).column(0).to_pylist(),
                         [None, None])
        # The existing parser treats an empty StructType as an empty names list.
        with self.assertRaisesRegex(ValueError, "can not infer schema from empty dataset"):
            self.spark.createDataFrame([], T.StructType([]))

    def test_mixed_containers_and_subclasses_fall_back(self):
        class IntSubclass(int):
            pass
        class TupleSubclass(tuple):
            pass
        class ListSubclass(list):
            pass
        schema = T.StructType([T.StructField("v", T.LongType())])
        for data in ([(1,), [2], {"v": 3}, Row(v=4)],
                     [(1,), TupleSubclass((2,)), (IntSubclass(3),)],
                     ListSubclass([(1,), (2,), (3,)]),
                     pd.DataFrame({"v": [1, 2, 3]})):
            with self.subTest(kind=type(data).__name__):
                table = self.capture(data, schema)
                self.assertEqual(table.column(0).to_pylist(), list(range(1, len(data)+1)))

    def test_other_schemas_and_inference(self):
        cases = [(T.DoubleType(), [(1.5,), (None,)], pa.float64()),
                 (T.StringType(), [("a",), (None,)], pa.string()),
                 (T.BooleanType(), [(True,), (False,)], pa.bool_()),
                 (T.IntegerType(), [(1,), (None,)], pa.int32())]
        for dtype, rows, arrow_type in cases:
            schema = T.StructType([T.StructField("v", dtype)])
            table = self.capture(rows, schema)
            self.assertEqual(table.column(0).type, arrow_type)
            self.assertEqual(table.column(0).to_pylist(), [r[0] for r in rows])
        for schema in (None, ["v"]):
            frame = self.spark.createDataFrame([(1,), (2,)], schema)
            frame.toArrow()
            table = pa.ipc.open_stream(self.captured[-1][1][0]).read_all()
            self.assertEqual(table.column(0).to_pylist(), [1, 2])

    def test_existing_errors_and_row_indices(self):
        schema = T.StructType([T.StructField("v", T.LongType())])
        for data, message in (([(1,), ()], "Row 1 has 0 values"),
                              ([(1,), (2, 3)], "Row 1 has 2 values")):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, message):
                self.spark.createDataFrame(data, schema)
        # These must keep the generic converter's errors, rather than casting bool
        # to integer, truncating overflow, or silently accepting non-nullable nulls.
        for data in ([(1,), (True,)], [(1,), (1 << 63,)], [(1,), (-(1 << 63)-1,)]):
            with self.subTest(data=data), self.assertRaisesRegex(RuntimeError, "Type mismatch in row data"):
                self.spark.createDataFrame(data, schema)
        required = T.StructType([T.StructField("v", T.LongType(), False)])
        with self.assertRaisesRegex(RuntimeError, "non-nullable"):
            self.spark.createDataFrame([(1,), (None,)], required)
        self.assertEqual(self.captured, [])
        self.assertEqual(self.config_calls, 0)


if __name__ == "__main__":
    unittest.main()
