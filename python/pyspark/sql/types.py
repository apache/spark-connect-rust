"""
Minimal pyspark.sql.types module for the Spark Connect client.
Provides DataType classes and conversion functions.
"""

# Import from _pyspark for the actual DataType implementations
try:
    from pyspark._pyspark import (
        DataType, NullType, BooleanType, ByteType, ShortType, IntegerType, LongType,
        FloatType, DoubleType, DecimalType, StringType, BinaryType, DateType,
        TimestampType, TimestampNTZType, ArrayType, MapType, StructField, StructType,
        CharType, VarcharType, TimeType, CalendarIntervalType, YearMonthIntervalType,
        DayTimeIntervalType, VariantType,
    )
except ImportError:  # pragma: no cover - defensive fallback when the extension is absent
    # Fallback: define minimal stubs for testing purposes
    class DataType:
        """Base class for all data types."""
        def __init__(self):
            pass

    class NullType(DataType):
        """Null data type."""
        pass

    class BooleanType(DataType):
        """Boolean data type."""
        pass

    class ByteType(DataType):
        """Byte data type."""
        pass

    class ShortType(DataType):
        """Short data type."""
        pass

    class IntegerType(DataType):
        """Integer data type."""
        pass

    class LongType(DataType):
        """Long data type."""
        pass

    class FloatType(DataType):
        """Float data type."""
        pass

    class DoubleType(DataType):
        """Double data type."""
        pass

    class DecimalType(DataType):
        """Decimal data type."""
        pass

    class StringType(DataType):
        """String data type."""
        pass

    class BinaryType(DataType):
        """Binary data type."""
        pass

    class DateType(DataType):
        """Date data type."""
        pass

    class TimestampType(DataType):
        """Timestamp data type."""
        pass

    class TimestampNTZType(DataType):
        """Timestamp NTZ data type."""
        pass

    class ArrayType(DataType):
        """Array data type."""
        pass

    class MapType(DataType):
        """Map data type."""
        pass

    class StructField(DataType):
        """Struct field."""
        pass

    class StructType(DataType):
        """Struct data type."""
        pass


__all__ = [
    "DataType",
    "NullType",
    "BooleanType",
    "ByteType",
    "ShortType",
    "IntegerType",
    "LongType",
    "FloatType",
    "DoubleType",
    "DecimalType",
    "StringType",
    "BinaryType",
    "DateType",
    "TimestampType",
    "TimestampNTZType",
    "ArrayType",
    "MapType",
    "StructField",
    "StructType",
    "CharType",
    "VarcharType",
    "TimeType",
    "CalendarIntervalType",
    "YearMonthIntervalType",
    "DayTimeIntervalType",
    "VariantType",
    "_parse_datatype_json_string",
    "_parse_datatype_json_value",
]


def _parse_datatype_json_string(json_string):
    """Reconstruct a DataType from its Spark JSON (client-side; UDF workers use the
    official pyspark function of the same name). Handles the full recursive grammar:
    atomic + parameterized string forms and the nested struct/array/map object forms."""
    import json as _json

    return _parse_datatype_json_value(_json.loads(json_string))


def _parse_datatype_json_value(v):
    """Recursively reconstruct a DataType from a parsed Spark-JSON value (a string for
    atomic/parameterized types, or a dict for the nested struct/array/map forms)."""
    import re as _re

    if isinstance(v, str):
        atomic = {
            "void": NullType, "null": NullType, "boolean": BooleanType, "byte": ByteType,
            "short": ShortType, "integer": IntegerType, "long": LongType, "float": FloatType,
            "double": DoubleType, "string": StringType, "binary": BinaryType, "date": DateType,
            "timestamp": TimestampType, "timestamp_ntz": TimestampNTZType, "variant": VariantType,
        }
        if v in atomic:
            return atomic[v]()
        m = _re.match(r"decimal\((\d+),\s*(\d+)\)$", v)
        if m:
            return DecimalType(int(m.group(1)), int(m.group(2)))
        if v == "decimal":
            return DecimalType()
        m = _re.match(r"char\((\d+)\)$", v)
        if m:
            return CharType(int(m.group(1)))
        m = _re.match(r"varchar\((\d+)\)$", v)
        if m:
            return VarcharType(int(m.group(1)))
        m = _re.match(r"time\((\d+)\)$", v)
        if m:
            return TimeType(int(m.group(1)))
        if v == "calendarinterval":
            return CalendarIntervalType()
        _ym = {"year": 0, "month": 1}
        _dt = {"day": 0, "hour": 1, "minute": 2, "second": 3}
        m = _re.match(r"interval (\w+)(?: to (\w+))?$", v)
        if m:
            start, end = m.group(1), m.group(2)
            if start in _ym:
                s = _ym[start]
                return YearMonthIntervalType(s, _ym.get(end, s) if end else s)
            if start in _dt:
                s = _dt[start]
                return DayTimeIntervalType(s, _dt.get(end, s) if end else s)
        raise ValueError(f"cannot parse datatype json: {v!r}")
    elif isinstance(v, dict):
        t = v.get("type")
        if t == "struct":
            fields = [
                StructField(
                    f["name"],
                    _parse_datatype_json_value(f["type"]),
                    f.get("nullable", True),
                    f.get("metadata", {}) or {},
                )
                for f in v["fields"]
            ]
            return StructType(fields)
        elif t == "array":
            return ArrayType(
                _parse_datatype_json_value(v["elementType"]),
                v.get("containsNull", True),
            )
        elif t == "map":
            return MapType(
                _parse_datatype_json_value(v["keyType"]),
                _parse_datatype_json_value(v["valueType"]),
                v.get("valueContainsNull", True),
            )
    raise ValueError(f"cannot parse datatype json: {v!r}")
