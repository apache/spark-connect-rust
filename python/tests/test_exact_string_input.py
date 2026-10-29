# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Guard exact-string conversion and decode the actual outbound Arrow IPC."""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import pytest
from pyspark.sql import Row, SparkSession


@pytest.mark.parametrize("text", ["", "1.25", "-42", "NaN", "a\x00b", "é雪😀", "e\u0301", "\uffff"])
def test_exact_string_roundtrip(text):
    actual = Row(value=text).value
    assert type(actual) is str
    assert actual == text


@pytest.mark.parametrize("text", ["\ud800", "\udfff", "a\ud800b"])
def test_surrogate_error_preserves_generic_fallback(text):
    with pytest.raises(TypeError, match="unsupported value type for createDataFrame: str"):
        Row(value=text)


def test_string_subclass_protocol_order():
    trace = []

    class IndexString(str):
        def __index__(self):
            trace.append("index")
            return 17
        def __float__(self):
            raise AssertionError("index must take precedence")

    class FloatString(str):
        def __float__(self):
            trace.append("float")
            return 2.5
        def __str__(self):
            raise AssertionError("numeric conversion must take precedence")

    class FallbackString(str):
        def __index__(self):
            trace.append("bad index")
            raise ValueError("index rejected")
        def __float__(self):
            trace.append("bad float")
            raise ValueError("float rejected")
        def __str__(self):
            raise AssertionError("string extraction must retain the underlying text")

    class PlainString(str):
        def __str__(self):
            raise AssertionError("string extraction must not call __str__")

    for value, expected in [(IndexString("x"), 17), (FloatString("x"), 2.5),
                            (FallbackString("original"), "original"),
                            (PlainString("plain"), "plain")]:
        actual = Row(value=value).value
        assert type(actual) is type(expected)
        assert actual == expected
    assert trace == ["index", "float", "bad index", "bad float"]


def test_nested_strings_and_nulls():
    value = Row(items=["雪", None, ""], mapping={"x": "😀", "null": None})
    actual = Row(value=value).value
    assert list(actual.items) == ["雪", None, ""]
    assert actual.mapping == {"x": "😀", "null": None}


def _varint(value):
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def _message(field, value):
    return _varint(field * 8 + 2) + _varint(len(value)) + value


def _fields(data):
    def read(pos):
        value, shift = 0, 0
        while True:
            byte = data[pos]
            pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value, pos
            shift += 7
    result, pos = {}, 0
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
            raise AssertionError(f"unexpected wire type {wire}")
    return result


def _serve(connection):
    import grpc
    import pyarrow as pa

    state = {}

    def analyze(raw, context):
        # AnalyzePlanRequest.schema.plan.root.local_relation.data
        relation = _fields(_fields(_fields(_fields(raw)[4])[1])[1])
        payload = _fields(relation[11])[1]
        table = pa.ipc.open_stream(payload).read_all()
        state["result"] = (table.to_pylist(), str(table.schema))
        # Return an empty StructType; the test inspects the input IPC itself.
        return _message(2, _message(1, _message(21, b"")))

    with ThreadPoolExecutor(max_workers=2) as executor:
        server = grpc.server(executor)
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
            "spark.connect.SparkConnectService", {
                "AnalyzePlan": grpc.unary_unary_rpc_method_handler(analyze)}),))
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        connection.send(port)
        try:
            while True:
                command = connection.recv()
                if command == "stop":
                    break
                assert command == "get"
                connection.send(state["result"])
        finally:
            server.stop(0).wait()


@pytest.fixture
def input_service():
    pytest.importorskip("grpc")
    pytest.importorskip("pyarrow")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_serve, args=(child,))
    process.start()
    try:
        assert parent.poll(20)
        spark = SparkSession.builder.remote(f"sc://127.0.0.1:{parent.recv()}").getOrCreate()
        yield spark, parent
    finally:
        if process.is_alive():
            parent.send("stop")
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        child.close()


@pytest.mark.parametrize("source", ["tuples", "pandas"])
@pytest.mark.parametrize("schema_mode", ["inferred", "names", "explicit"])
def test_string_input_ipc(input_service, source, schema_mode):
    spark, parent = input_service
    from pyspark.sql.types import StructType, StructField, StringType
    import pandas as pd

    rows = [("雪😀", "a\x00b"), ("", None), (None, "1.25"), ("e\u0301", "\uffff")]
    data = pd.DataFrame(rows, columns=["a", "b"]) if source == "pandas" else rows
    schema = (StructType([StructField("a", StringType()), StructField("b", StringType())])
              if schema_mode == "explicit" else ["a", "b"] if schema_mode == "names" else None)
    spark.createDataFrame(data, schema).schema
    parent.send("get")
    assert parent.poll(10)
    actual, arrow_schema = parent.recv()
    names = ["col0", "col1"] if source == "tuples" and schema_mode == "inferred" else ["a", "b"]
    assert actual == [dict(zip(names, row)) for row in rows]
    assert all(f"{name}: string" in arrow_schema for name in names)


def test_nested_string_input_ipc(input_service):
    from pyspark.sql.types import (StructType, StructField, StringType, ArrayType, MapType)
    spark, parent = input_service
    schema = StructType([
        StructField("items", ArrayType(StringType())),
        StructField("mapping", MapType(StringType(), StringType())),
        StructField("nested", StructType([StructField("text", StringType())])),
    ])
    rows = [(["雪", None, ""], {"x": "😀", "null": None}, Row(text="a\x00b")),
            (None, None, None)]
    spark.createDataFrame(rows, schema).schema
    parent.send("get")
    assert parent.poll(10)
    actual, _ = parent.recv()
    # Arrow map columns decode as ordered key/value pairs.
    assert actual == [
        {"items": ["雪", None, ""], "mapping": [("null", None), ("x", "😀")],
         "nested": {"text": "a\x00b"}},
        {"items": None, "mapping": None, "nested": None},
    ]


@pytest.mark.parametrize("source", ["tuples", "pandas"])
def test_large_string_input_ipc(input_service, source):
    from pyspark.sql.types import StructType, StructField, StringType
    import pandas as pd
    spark, parent = input_service
    names = ["a", "b", "c"]
    rows = [(f"雪😀-{i}", str(i), None if i % 7 == 0 else "a\x00b")
            for i in range(10000)]
    data = pd.DataFrame(rows, columns=names) if source == "pandas" else rows
    schema = StructType([StructField(name, StringType()) for name in names])
    spark.createDataFrame(data, schema).schema
    parent.send("get")
    assert parent.poll(10)
    actual, _ = parent.recv()
    assert actual == [dict(zip(names, row)) for row in rows]
