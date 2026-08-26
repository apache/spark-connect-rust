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
except ImportError:
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
]


def _parse_datatype_json_string(json_string):
    """Reconstruct a DataType from its Spark JSON (client-side; UDF workers use the
    official pyspark function of the same name). Handles the atomic + decimal forms
    our __reduce__ emits; raises for anything else so it never silently mis-parses."""
    import json as _json
    import re as _re

    v = _json.loads(json_string)
    if isinstance(v, str):
        atomic = {
            "void": NullType, "null": NullType, "boolean": BooleanType, "byte": ByteType,
            "short": ShortType, "integer": IntegerType, "long": LongType, "float": FloatType,
            "double": DoubleType, "string": StringType, "binary": BinaryType, "date": DateType,
            "timestamp": TimestampType, "timestamp_ntz": TimestampNTZType, "variant": VariantType,
        }
        if v in atomic:
            return atomic[v]()
        m = _re.match(r"decimal\((\d+),\s*(\d+)\)", v)
        if m:
            return DecimalType(int(m.group(1)), int(m.group(2)))
        m = _re.match(r"char\((\d+)\)", v)
        if m:
            return CharType(int(m.group(1)))
        m = _re.match(r"varchar\((\d+)\)", v)
        if m:
            return VarcharType(int(m.group(1)))
    raise ValueError(f"cannot parse datatype json: {json_string!r}")
