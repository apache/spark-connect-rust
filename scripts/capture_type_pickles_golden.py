# SPDX-License-Identifier: Apache-2.0
"""Capture official PySpark 4.3.0's pickles of its data types, which the drop-in must load.

PySpark's types are plain classes: pickle saves one as its class and its ``__dict__``, of
``copyreg.__newobj__`` at the protocols 2 and up and of ``copyreg._reconstructor`` below. The
golden holds PySpark's pickle of a list of a type of each family at each protocol, 0 to 5, and
for each type its repr, JSON, simpleString and attributes (the repr of each value of its
``__dict__``), and the pickles of types whose attributes were set after they were made.
``python/tests/test_dropin_types_columns_offline.py`` loads them in the drop-in.

Run with official PySpark 4.3.0 (``pyspark-client``), without a server::

    python scripts/capture_type_pickles_golden.py --out tests/golden/type_pickles.json
"""

import argparse
import base64
import json
import pickle
from pathlib import Path

import pyspark
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import types as T

# The types of the golden, by the expression of `pyspark.sql.types` (and VectorUDT) that makes
# each: the singletons, VariantType, each parameterized type of its parameters, nested types,
# nullable flags that are not bools, and a user-defined type.
CASES = [
    "NullType()",
    "BooleanType()",
    "ByteType()",
    "ShortType()",
    "IntegerType()",
    "LongType()",
    "FloatType()",
    "DoubleType()",
    "BinaryType()",
    "DateType()",
    "TimestampType()",
    "TimestampNTZType()",
    "CalendarIntervalType()",
    "VariantType()",
    "DecimalType()",
    "DecimalType(10, 2)",
    "DecimalType(38, 18)",
    "DecimalType(5, -2)",
    "StringType()",
    "StringType('UTF8_LCASE')",
    "StringType('UNICODE_CI')",
    "CharType(5)",
    "VarcharType(10)",
    "TimeType()",
    "TimeType(3)",
    "TimeType(0)",
    "DayTimeIntervalType()",
    "DayTimeIntervalType(1, 2)",
    "DayTimeIntervalType(3)",
    "YearMonthIntervalType()",
    "YearMonthIntervalType(1)",
    "YearMonthIntervalType(1, 0)",
    "GeometryType(4326)",
    "GeometryType(0)",
    "GeometryType(3857)",
    "GeometryType('ANY')",
    "GeographyType(4326)",
    "GeographyType('ANY')",
    "ArrayType(IntegerType())",
    "ArrayType(StringType('UTF8_LCASE'), False)",
    "ArrayType(IntegerType(), 1)",
    "ArrayType(ArrayType(DecimalType(5, 1), False))",
    "MapType(StringType(), IntegerType())",
    "MapType(IntegerType(), MapType(StringType(), TimeType(3), False), False)",
    "StructField('a', IntegerType())",
    "StructField('b', CharType(3), False, {'k': 'v'})",
    "StructField('c', StringType('UTF8_LCASE'), 'x')",
    "StructType()",
    "StructType([StructField('a', IntegerType()), StructField('b', DateType(), False)])",
    "StructType([StructField('s', StructType([StructField('x', ArrayType(MapType("
    "StringType('UNICODE_CI'), DayTimeIntervalType(1, 2))))])), "
    "StructField('v', VectorUDT())])",
    "VectorUDT()",
    "ArrayType(VectorUDT(), False)",
]


def edited():
    """Types whose attributes were set after they were made, by their names: their pickles keep
    the attributes as they were set, which no constructor makes."""
    decimal = T.DecimalType(10, 2)
    decimal.hasPrecisionInfo = False
    decimal.tag = "x"
    renamed = T.StructType([T.StructField("a", T.IntegerType())])
    renamed.fields[0].name = "z"
    retyped = T.StructType([T.StructField("d", T.DateType())])
    retyped.fields[0].dataType = T.IntegerType()
    scaled = T.DecimalType(10, 2)
    scaled.scale = 3
    moved = T.GeometryType(4326)
    moved.srid = 0
    return {
        # hasPrecisionInfo of another value than True, and an attribute of the user.
        "decimal_attributes": decimal,
        # The names of a struct, which an edit of a field's name does not change.
        "renamed_field": renamed,
        # The _needConversion of a struct, which an edit of a field's type does not change.
        "retyped_field": retyped,
        # A parameter that was set.
        "set_scale": scaled,
        # An SRID that was set, whose _crs stayed that of the constructor.
        "set_srid": moved,
    }


def description(t):
    """The repr, JSON, simpleString and attributes of the type `t`, as PySpark gives them."""
    return {
        "repr": repr(t),
        "json": t.json(),
        "simpleString": t.simpleString(),
        "attributes": {key: repr(value) for key, value in vars(t).items()},
    }


def encode(value, protocol):
    return base64.b64encode(pickle.dumps(value, protocol=protocol)).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("tests/golden/type_pickles.json"))
    args = parser.parse_args()
    names = dict(vars(T), VectorUDT=VectorUDT)
    types = [eval(case, names) for case in CASES]
    edits = edited()
    golden = {
        "pyspark": pyspark.__version__,
        "cases": [dict(type=case, **description(t)) for case, t in zip(CASES, types)],
        "pickles": {str(protocol): encode(types, protocol) for protocol in range(6)},
        "edited": {name: description(t) for name, t in edits.items()},
        "edited_pickles": {str(protocol): encode(edits, protocol) for protocol in range(2, 6)},
    }
    args.out.write_text(json.dumps(golden, indent=1) + "\n")


if __name__ == "__main__":
    main()
