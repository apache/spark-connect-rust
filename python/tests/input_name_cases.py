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

"""Check input names and complete IPC values without a Spark server or JVM."""

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
class InputNameTests(unittest.TestCase):
    def setUp(self):
        self.captured = []

        def execute(raw, context):
            # ExecutePlanRequest.plan(3) -> Plan.root(1) -> Relation.local_relation(11).
            request = _fields(raw)
            root = _fields(_fields(request[3][0])[1][0])
            self.captured.append(_fields(root[11][0]))
            yield (_bytes(1, request[1][0]) + _bytes(15, b"input-name-fixture")
                   + _bytes(12, request[6][0]) + _bytes(13, b"complete")
                   + _bytes(14, b""))

        def config(raw, context):
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

    def capture(self, frame):
        before = len(self.captured)
        frame.toArrow()
        self.assertEqual(len(self.captured), before + 1)
        local = self.captured[-1]
        if 1 in local:
            table = pa.ipc.open_stream(local[1][0]).read_all()
            return (table.schema.names,
                    [[table.column(j)[i].as_py() for j in range(table.num_columns)]
                     for i in range(table.num_rows)])
        schema = json.loads(local[2][0])
        return [field["name"] for field in schema["fields"]], []

    def test_name_precedence_and_values(self):
        cases = [
            ([(7, 8), (9, 10)], None, ["col0", "col1"]),
            ([Row(z=7, a=8), Row(z=9, a=10)], None, ["z", "a"]),
            ([Row(z=7, a=8), Row(z=9, a=10)], ["x", "y"], ["x", "y"]),
            ([{"z": 8, "a": 7}, {"z": 10, "a": 9}], None, ["a", "z"]),
            ([{"z": 8, "a": 7}, {"z": 10, "a": 9}], ["x", "y"], ["x", "y"]),
            (pd.DataFrame([[7, 8], [9, 10]], columns=["café", "雪"]),
             None, ["café", "雪"]),
            (pd.DataFrame([[7, 8], [9, 10]], columns=["old", "names"]),
             ["x", "y"], ["x", "y"]),
        ]
        for data, schema, names in cases:
            with self.subTest(names=names, source=type(data).__name__):
                self.assertEqual(self.capture(self.spark.createDataFrame(data, schema)),
                                 (names, [[7, 8], [9, 10]]))

    def test_unusual_names_nulls_and_empty_input(self):
        names = ["dup", "dup", "", "café", "雪"]
        data = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10)]
        for schema in (names, T.StructType([
                T.StructField(name, T.LongType(), True) for name in names])):
            self.assertEqual(self.capture(self.spark.createDataFrame(data, schema)),
                             (names, [list(row) for row in data]))
        schema = T.StructType([T.StructField("number", T.LongType(), True),
                               T.StructField("text", T.StringType(), True)])
        data = pd.DataFrame({"old": pd.Series([1, None], dtype=object),
                             "names": ["a", None]})
        before = data.copy(deep=True)
        self.assertEqual(self.capture(self.spark.createDataFrame(data, schema)),
                         (["number", "text"], [[1, "a"], [None, None]]))
        pd.testing.assert_frame_equal(data, before)
        self.assertEqual(self.capture(self.spark.createDataFrame([], schema)),
                         (["number", "text"], []))

    def test_each_call_owns_explicit_and_pandas_names(self):
        names = ["first", "second"]
        first = self.spark.createDataFrame([(1, 2), (3, 4)], names)
        names[:] = ["new", "labels"]
        second = self.spark.createDataFrame([(5, 6)], names)
        self.assertEqual(self.capture(first), (["first", "second"], [[1, 2], [3, 4]]))
        self.assertEqual(self.capture(second), (["new", "labels"], [[5, 6]]))
        self.assertEqual(self.capture(first), (["first", "second"], [[1, 2], [3, 4]]))
        data = pd.DataFrame([[7, 8]], columns=["old", "names"])
        frame = self.spark.createDataFrame(data)
        data.columns = ["changed", "later"]
        self.assertEqual(self.capture(frame), (["old", "names"], [[7, 8]]))

    def test_explicit_names_still_parse_overridden_inputs(self):
        trace = []

        class BrokenKey:
            def __str__(self):
                trace.append("key")
                raise ValueError("key conversion failed")

        with self.assertRaisesRegex(ValueError, "key conversion failed"):
            self.spark.createDataFrame([{BrokenKey(): 1}], ["override"])
        self.assertEqual(trace, ["key"])
        with self.assertRaises(TypeError):
            self.spark.createDataFrame(pd.DataFrame([[1]], columns=[7]), ["override"])
        with self.assertRaisesRegex(TypeError, "unsupported value type.*object"):
            self.spark.createDataFrame([(object(),)], ["x", "y"])


if __name__ == "__main__":
    unittest.main()
