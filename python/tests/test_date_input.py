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

"""Date conversion preserves the existing ordinal and subclass semantics."""
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from pyspark.sql import Row


@pytest.mark.parametrize("value", [
    date.min, date.max, date(1, 3, 1), date(4, 2, 29),
    date(1600, 2, 29), date(1900, 2, 28), date(1900, 3, 1),
    date(1969, 12, 31), date(1970, 1, 1), date(1970, 1, 2),
    date(2000, 2, 29), date(2100, 3, 1),
])
def test_exact_dates_roundtrip(value):
    actual = Row(value=value).value
    assert type(actual) is date
    assert actual == value


def test_date_subclasses_keep_protocol_precedence():
    trace = []

    class Indexed(date):
        def __index__(self):
            trace.append("index")
            return 42

        def toordinal(self):
            raise AssertionError("numeric extraction must precede date conversion")

    class Floating(date):
        def __float__(self):
            trace.append("float")
            return 1.25

        def toordinal(self):
            raise AssertionError("numeric extraction must precede date conversion")

    class Ordinal(date):
        def toordinal(self):
            trace.append("ordinal")
            return date(2000, 2, 29).toordinal()

    class Plain(date):
        pass

    assert Row(value=Indexed(2020, 1, 1)).value == 42
    assert Row(value=Floating(2020, 1, 1)).value == 1.25
    assert Row(value=Ordinal(2020, 1, 1)).value == date(2000, 2, 29)
    assert type(Row(value=Plain(2020, 1, 1)).value) is date
    assert trace == ["index", "float", "ordinal"]


def test_datetime_and_timestamp_keep_timestamp_encoding():
    pd = pytest.importorskip("pandas")
    for value in [datetime(2020, 2, 29, 12, 34, 56, 123456),
                  datetime(2020, 2, 29, tzinfo=timezone.utc),
                  pd.Timestamp("2020-02-29 12:34:56.123456"),
                  pd.Timestamp("2020-02-29 12:34:56.123456Z")]:
        actual = Row(value=value).value
        assert type(actual) is datetime
        expected = value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
        assert actual == datetime.fromtimestamp(expected.timestamp())


def test_nested_dates_and_unrelated_values():
    value = Row(items=[date.min, None, date.max],
                mapping={"epoch": date(1970, 1, 1)},
                nested=Row(day=date(2000, 2, 29)),
                amount=Decimal("-12.340"), number=42, flag=True, label="date")
    actual = Row(value=value).value
    assert actual.items == [date.min, None, date.max]
    assert actual.mapping == {"epoch": date(1970, 1, 1)}
    assert actual.nested.day == date(2000, 2, 29)
    assert actual.amount == Decimal("-12.340")
    assert (actual.number, actual.flag, actual.label) == (42, True, "date")


def test_named_impostor_is_not_a_date():
    impostor = type("date", (), {"toordinal": lambda self: 719163})
    with pytest.raises(TypeError, match="unsupported value type"):
        Row(value=impostor())


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
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "captured date input")
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
            with pytest.raises(Exception, match="captured date input"):
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
def test_date_input_ipc_across_calendar_range(capture_input, source, schema_kind):
    import calendar
    import pyarrow as pa
    from pyspark.sql import types as T

    # Every year's endpoints and leap day cover the complete supported calendar
    # range, including non-leap centuries, without relying on the Rust decoder.
    dates = [date.min, date.max, date(1969, 12, 31), date(1970, 1, 1),
             date(1970, 1, 2), None]
    for year in range(1, 10000):
        dates.extend([date(year, 1, 1), date(year, 12, 31)])
        if calendar.isleap(year):
            dates.append(date(year, 2, 29))
    schema = {"inferred": None, "names": ["day"], "ddl": "day date",
              "explicit": T.StructType([T.StructField("day", T.DateType())])}[schema_kind]
    data = [(value,) for value in dates]
    if source == "pandas":
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame(data, columns=["day"])
    table = capture_input(data, schema)
    assert table.column(0).type == pa.date32()
    assert table.column(0).to_pylist() == dates
    epoch = date(1970, 1, 1)
    expected_days = [None if value is None else (value - epoch).days for value in dates]
    assert table.column(0).cast(pa.int32()).to_pylist() == expected_days
