#
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
#
"""Test client scalar conversion through Row without a Spark server."""

from datetime import date
from decimal import Decimal

import pyarrow as pa
import pytest
from pyspark.sql import Row

from test_arrow_handoff import _dataframe


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        -(2**63),
        2**63 - 1,
        1.25,
        "",
        "ascii",
        "a\0b",
        "\u00e9",
        "\u03bb",
        "\U0001f600",
        b"a\0b",
        Decimal("1.25"),
        date(1970, 1, 1),
        date.min,
        date.max,
    ],
)
def test_row_scalar_round_trip_preserves_value_and_type(value):
    actual = Row(value=value).value
    assert actual == value
    assert type(actual) is type(value)


def test_row_extraction_preserves_numeric_coercion_and_builtin_subclasses():
    class IndexValue:
        def __index__(self):
            return 42

    class FloatValue:
        def __float__(self):
            return 1.5

    class IntSubclass(int):
        pass

    class FloatSubclass(float):
        pass

    class StrSubclass(str):
        pass

    for value, expected in [
        (IndexValue(), 42),
        (FloatValue(), 1.5),
        (IntSubclass(7), 7),
        (FloatSubclass(1.25), 1.25),
        (StrSubclass("a\0b"), "a\0b"),
    ]:
        actual = Row(value=value).value
        assert actual == expected
        assert type(actual) is type(expected)


def test_row_binary_values_are_snapshots_but_integer_sequences_stay_lists():
    class BytesSubclass(bytes):
        pass

    class ByteArraySubclass(bytearray):
        pass

    for value in [b"ab", BytesSubclass(b"ab"), bytearray(b"ab"), ByteArraySubclass(b"ab")]:
        row = Row(value=value)
        if isinstance(value, bytearray):
            value[0] = ord("c")
        assert row.value == b"ab"
        assert type(row.value) is bytes

    for value in [[97, 98], (97, 98)]:
        actual = Row(value=value).value
        assert actual == [97, 98]
        assert type(actual) is list


def test_row_rejects_arbitrary_truthiness_stringification_and_buffers():
    class TruthValue:
        def __bool__(self):
            raise AssertionError("truthiness must not be invoked")

    class StringValue:
        def __str__(self):
            raise AssertionError("stringification must not be invoked")

    for value in [TruthValue(), StringValue(), memoryview(b"ab"), "\ud800"]:
        with pytest.raises(TypeError, match="unsupported value type"):
            Row(value=value)


@pytest.mark.parametrize("days", [-719_163, 2_932_897, -(2**31), 2**31 - 1])
def test_collect_rejects_date32_values_outside_python_range(days):
    batch = pa.record_batch([pa.array([days], type=pa.date32())], names=["value"])
    with _dataframe([batch]) as (dataframe, _):
        row = dataframe.collect()[0]

    # Row materialization is lazy, so the exception occurs on field access.
    with pytest.raises((ValueError, OverflowError)) as expected:
        date.fromordinal(days + 719_163)
    with pytest.raises(type(expected.value), match=str(expected.value)):
        _ = row.value
