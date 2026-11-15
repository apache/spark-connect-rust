# SPDX-License-Identifier: Apache-2.0
"""The `(value, DataType)` pairs that `serialize` of `pyspark.ml.connect.serialize` sends as
`LiteralExpression(value, dataType)`, and values near one, of tests/golden/ml_fetch_pairs.json
(scripts/capture_ml_fetch_pairs_golden.py records them of official PySpark).

Each case is a Python expression of a value, which `ChiSquareTest.test` serializes as its dataset
argument: a helper call that serializes its arguments when it is called and sends them when its
DataFrame runs. The outcome of a case is the literal that the call serializes, or its error.
"""

import base64
import datetime
import decimal
import enum
import json
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[2] / "tests/golden/ml_fetch_pairs.json"

CASES = [
    # A value of each type that a literal takes, also of a value that converts to it.
    "(1, IntegerType())",
    "(2**40, LongType())",
    "(7, ShortType())",
    "(True, IntegerType())",
    "(1.5, DoubleType())",
    "(1.5, FloatType())",
    "(True, BooleanType())",
    "(decimal.Decimal('1.5'), DecimalType(10, 2))",
    "('a', StringType())",
    "('a', StringType('UTF8_LCASE'))",
    "(b'ab', BinaryType())",
    "(bytearray(b'ab'), BinaryType())",
    "(datetime.date(2020, 1, 2), DateType())",
    "(datetime.datetime(2020, 1, 2, 3, 4, 5), DateType())",
    "(datetime.datetime(2020, 1, 2, 3, 4, 5), TimestampType())",
    "(datetime.datetime(2020, 1, 2, 3, 4, 5), TimestampNTZType())",
    "(datetime.time(1, 2, 3, 4), TimeType())",
    "(datetime.timedelta(days=1, seconds=2), DayTimeIntervalType())",
    "(None, NullType())",
    "(None, IntegerType())",
    "(Color.RED, IntegerType())",
    "(np.int8(1), ByteType())",
    "(np.float32(1.5), FloatType())",
    "(np.bool_(True), BooleanType())",
    "(np.str_('a'), StringType())",
    # An array, of the element type, also when it is empty.
    "([], ArrayType(StringType()))",
    "(['a', None], ArrayType(StringType()))",
    "([[1, 2], [], None], ArrayType(ArrayType(IntegerType())))",
    "([], ArrayType(MapType(StringType(), StringType())))",
    # A value that is not of the type: the bare AssertionError of the constructor.
    "('x', IntegerType())",
    "(1.0, IntegerType())",
    "(300, ByteType())",
    "(1, DoubleType())",
    "(1.5, DecimalType(10, 2))",
    "(datetime.date(2020, 1, 2), TimestampType())",
    "(1, NullType())",
    "(np.int64(1), IntegerType())",
    "([1, 'x'], ArrayType(IntegerType()))",
    "((1, 2), ArrayType(IntegerType()))",
    "([None], ArrayType(MapType(StringType(), StringType())))",
    # A type that no literal has, also with a None value.
    "(None, StructType())",
    "(None, MapType(StringType(), StringType()))",
    "({'a': 'b'}, MapType(StringType(), StringType()))",
    "('a', CharType(3))",
    "('a', VarcharType(3))",
    "(1, YearMonthIntervalType())",
    "(None, VectorUDT())",
    "(None, StructField('a', IntegerType()))",
    # No pair: the literal of LiteralExpression._from_value, or the error of that conversion.
    "(1, IntegerType(), 2)",
    "[1, IntegerType()]",
    "(IntegerType(), 1)",
    "(1, IntegerType)",
]


class Color(enum.Enum):
    RED = 1


def value_of(case):
    """The value of the expression of a case."""
    import numpy as np
    import pyspark.sql.types as types
    from pyspark.ml.linalg import VectorUDT

    names = {name: getattr(types, name) for name in dir(types) if name.endswith(("Type", "Field"))}
    names.update(datetime=datetime, decimal=decimal, np=np, Color=Color, VectorUDT=VectorUDT)
    return eval(case, names)  # noqa: S307 - the expressions above


def outcome(case, test, relation):
    """The outcome of a case: the literal that `test(value)` serializes as its dataset, base64 of
    its deterministic bytes, or the class and the first line of the message of its error.
    `relation` is the relation of a DataFrame."""
    try:
        result = test(value_of(case))
    except Exception as e:  # noqa: BLE001 - every error is a case
        lines = str(e).strip().splitlines()
        return {"error": {"class": type(e).__name__, "message": lines[0] if lines else ""}}
    (method,) = relation(result).ml_relation.fetch.methods
    literal = method.args[0].param.SerializeToString(deterministic=True)
    return {"sent": base64.b64encode(literal).decode()}


def golden():
    return json.loads(GOLDEN.read_text())
