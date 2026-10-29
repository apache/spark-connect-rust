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

"""Exact containers retain recursive values and subclass dispatch semantics."""
from datetime import date, datetime
from decimal import Decimal

import pytest
from pyspark.sql import Row


@pytest.mark.parametrize("container", [list, tuple])
def test_container_values_and_errors(container):
    value = container([1, None, [2, None], (3, 4)])
    assert Row(value=value).value == [1, None, [2, None], [3, 4]]
    assert Row(value=container()).value == []
    assert Row(value=container([1, 2, 255])).value == [1, 2, 255]
    assert Row(value=b"abc").value == b"abc"
    assert Row(value=bytearray(b"abc")).value == b"abc"
    with pytest.raises(TypeError, match="unsupported value type"):
        Row(value=container([object()]))


@pytest.mark.parametrize("container", [list, tuple])
def test_container_subclass_numeric_precedence(container):
    trace = []

    class Indexed(container):
        def __index__(self):
            trace.append("index")
            return 42

    class Floating(container):
        def __float__(self):
            trace.append("float")
            return 1.25

    class Iterated(container):
        def __iter__(self):
            raise AssertionError("existing built-in iterator must remain in use")

    assert Row(value=Indexed([1, 2])).value == 42
    assert Row(value=Floating([1, 2])).value == 1.25
    assert Row(value=Iterated([1, 2])).value == [1, 2]
    assert trace == ["index", "float"]


def test_recursive_leaf_and_row_types():
    row = Row(x=1, y="a")
    values = [True, 42, 1.25, "text", b"abc", Decimal("-12.340"),
              date(2000, 2, 29), datetime(2020, 1, 1), row, None]
    result = Row(value=values).value
    assert result == values
    assert type(result[0]) is bool
    assert type(result[5]) is Decimal
    assert isinstance(result[8], Row)
    assert result[8].__fields__ == ["x", "y"]


def _fields(data):
    """Decode only the small protobuf envelope around a local Arrow relation."""
    def varint(pos):
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
        tag, pos = varint(pos)
        wire = tag & 7
        if wire == 2:
            length, pos = varint(pos)
            result[tag >> 3] = data[pos:pos + length]
            pos += length
        elif wire == 0:
            _, pos = varint(pos)
        elif wire in (1, 5):
            pos += 8 if wire == 1 else 4
        else:
            raise AssertionError(f"unexpected protobuf wire type: {wire}")
    return result


def _capture_server(connection):
    from concurrent.futures import ThreadPoolExecutor
    import grpc

    captured = []

    def execute(request, context):
        # ExecutePlanRequest.plan=3 -> Plan.root=1 ->
        # Relation.local_relation=11 -> LocalRelation.data=1.
        captured.append(_fields(_fields(_fields(_fields(request)[3])[1])[11])[1])
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "captured container input")
        yield b""  # Make this a streaming handler.

    with ThreadPoolExecutor(max_workers=2) as executor:
        server = grpc.server(executor)
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
            "spark.connect.SparkConnectService", {
                "ExecutePlan": grpc.unary_stream_rpc_method_handler(execute),
            }),))
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        connection.send(port)
        try:
            while True:
                command = connection.recv()
                if command == "stop":
                    break
                assert command == "get"
                connection.send(captured.pop(0))
        finally:
            server.stop(0).wait()


@pytest.fixture
def capture_input():
    import multiprocessing
    import sys
    from pathlib import Path

    # importlib-mode pytest does not add the tests namespace to sys.path.
    # Spawned children need it to import this server target; append it so the
    # installed wheel still takes precedence over the source pyspark package.
    test_root = str(Path(__file__).resolve().parents[1])
    if test_root not in sys.path:
        sys.path.append(test_root)
    pytest.importorskip("grpc")
    from pyspark.sql import SparkSession

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_capture_server, args=(child,))
    process.start()
    try:
        assert parent.poll(20), "capture service failed to start"
        spark = SparkSession.builder.remote(f"sc://127.0.0.1:{parent.recv()}").getOrCreate()

        def capture(data, schema):
            import pyarrow as pa
            df = spark.createDataFrame(data, schema=schema)
            with pytest.raises(Exception, match="captured container input"):
                df.collect()
            parent.send("get")
            assert parent.poll(10), "capture service failed to return the IPC payload"
            return pa.ipc.open_stream(parent.recv()).read_all()

        yield capture
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
@pytest.mark.parametrize("schema_kind", ["inferred", "names", "explicit", "ddl"])
@pytest.mark.parametrize("container", [list, tuple])
def test_array_input_ipc(capture_input, source, schema_kind, container):
    import pyarrow as pa
    from pyspark.sql import types as T

    values = [container([1, None, -3]), container(), None, container([255, 256])]
    schema = {"inferred": None, "names": ["items"], "ddl": "items array<bigint>",
              "explicit": T.StructType([T.StructField("items", T.ArrayType(T.LongType()))])}[schema_kind]
    data = [(value,) for value in values]
    if source == "pandas":
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame(data, columns=["items"])
    table = capture_input(data, schema)
    assert table.column(0).type == pa.list_(pa.int64())
    assert table.column(0).to_pylist() == [[1, None, -3], [], None, [255, 256]]


def test_nested_array_struct_binary_ipc(capture_input):
    from pyspark.sql import types as T

    schema = T.StructType([
        T.StructField("nested", T.ArrayType(T.ArrayType(T.LongType()))),
        T.StructField("record", T.StructType([
            T.StructField("x", T.LongType()), T.StructField("y", T.StringType())])),
        T.StructField("blob", T.BinaryType()),
    ])
    table = capture_input([([[1, None], (), None], Row(x=7, y="row"), b"abc")], schema)
    assert table.to_pylist() == [{"nested": [[1, None], [], None],
                                  "record": {"x": 7, "y": "row"}, "blob": b"abc"}]
