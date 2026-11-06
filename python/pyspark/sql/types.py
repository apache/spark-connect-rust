"""
Minimal pyspark.sql.types module for the Spark Connect client.
Provides DataType classes and conversion functions.
"""
from typing import cast  # noqa: F401  (upstream leaks typing.cast from this module)

# Import from _pyspark for the actual DataType implementations
try:
    from pyspark._pyspark import (
        DataType, NullType, BooleanType, ByteType, ShortType, IntegerType, LongType,
        FloatType, DoubleType, DecimalType, StringType, BinaryType, DateType,
        TimestampType, TimestampNTZType, ArrayType, MapType, StructField, StructType,
        CharType, VarcharType, TimeType, CalendarIntervalType, YearMonthIntervalType,
        DayTimeIntervalType, VariantType,
        # Abstract base classes of the type hierarchy.
        AtomicType, NumericType, IntegralType, FractionalType, DatetimeType,
        AnyTimeType, AnsiIntervalType, SpatialType,
        GeometryType, GeographyType,
        VariantVal, Geometry, Geography,
        # Spark's nanosecond timestamp types (no PySpark 4.3.0 class; not in __all__).
        TimestampNTZNanosType, TimestampLTZNanosType,
    )
    # Row lives in the Rust core; upstream code imports it from pyspark.sql.types too.
    from pyspark._pyspark import Row
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

        EPOCH_ORDINAL = 719163

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


class UserDefinedType(DataType):
    """User-defined type (UDT).

    .. note:: WARN: Spark Internal Use Only

    A UDT is inherently a Python-level construct: ``jsonValue`` serializes the
    concrete Python class with cloudpickle and ``fromJson`` re-imports it by
    module path, so (like the pickling serializers) it cannot be lowered into
    Rust. It subclasses the Rust-backed :class:`DataType` so ``isinstance`` and
    schema conversion treat it like any other type. Mirrors
    ``pyspark.sql.types.UserDefinedType``.
    """

    @classmethod
    def typeName(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def sqlType(cls) -> "DataType":
        """Underlying SQL storage type for this UDT."""
        from pyspark.errors import PySparkNotImplementedError

        raise PySparkNotImplementedError(
            errorClass="NOT_IMPLEMENTED",
            messageParameters={"feature": "sqlType()"},
        )

    @classmethod
    def module(cls) -> str:
        """The Python module of the UDT."""
        from pyspark.errors import PySparkNotImplementedError

        raise PySparkNotImplementedError(
            errorClass="NOT_IMPLEMENTED",
            messageParameters={"feature": "module()"},
        )

    @classmethod
    def scalaUDT(cls) -> str:
        """The class name of the paired Scala UDT (could be '', if there
        is no corresponding one)."""
        return ""

    def needConversion(self) -> bool:
        return True

    @classmethod
    def _cachedSqlType(cls) -> "DataType":
        """Cache the sqlType() into class, because it's heavily used in `toInternal`."""
        if not hasattr(cls, "_cached_sql_type"):
            cls._cached_sql_type = cls.sqlType()  # type: ignore[attr-defined]
        return cls._cached_sql_type  # type: ignore[attr-defined]

    def toInternal(self, obj):
        if obj is not None:
            return self._cachedSqlType().toInternal(self.serialize(obj))

    def fromInternal(self, obj):
        v = self._cachedSqlType().fromInternal(obj)
        if v is not None:
            return self.deserialize(v)

    def serialize(self, obj):
        """Converts a user-type object into a SQL datum."""
        from pyspark.errors import PySparkNotImplementedError

        raise PySparkNotImplementedError(
            errorClass="NOT_IMPLEMENTED",
            messageParameters={"feature": "toInternal()"},
        )

    def deserialize(self, datum):
        """Converts a SQL datum into a user-type object."""
        from pyspark.errors import PySparkNotImplementedError

        raise PySparkNotImplementedError(
            errorClass="NOT_IMPLEMENTED",
            messageParameters={"feature": "fromInternal()"},
        )

    def simpleString(self) -> str:
        return "udt"

    def json(self) -> str:
        import json as _json

        return _json.dumps(self.jsonValue(), separators=(",", ":"), sort_keys=True)

    def jsonValue(self) -> dict:
        import base64 as _base64
        from pyspark.serializers import CloudPickleSerializer

        if self.scalaUDT():
            assert self.module() != "__main__", "UDT in __main__ cannot work with ScalaUDT"
            schema = {
                "type": "udt",
                "class": self.scalaUDT(),
                "pyClass": "%s.%s" % (self.module(), type(self).__name__),
                "sqlType": self.sqlType().jsonValue(),
            }
        else:
            ser = CloudPickleSerializer()
            b = ser.dumps(type(self))
            schema = {
                "type": "udt",
                "pyClass": "%s.%s" % (self.module(), type(self).__name__),
                "serializedClass": _base64.b64encode(b).decode("utf8"),
                "sqlType": self.sqlType().jsonValue(),
            }
        return schema

    @classmethod
    def fromJson(cls, json: dict) -> "UserDefinedType":
        from pyspark.errors import PySparkValueError, PySparkTypeError

        pyUDT = str(json["pyClass"])  # convert unicode to str
        split = pyUDT.rfind(".")
        pyModule = pyUDT[:split]
        pyClass = pyUDT[split + 1 :]
        m = __import__(pyModule, globals(), locals(), [pyClass])
        if not hasattr(m, pyClass):
            raise PySparkValueError(
                errorClass="UNSUPPORTED_OPERATION",
                messageParameters={"operation": "unpickling user defined types"},
            )
        else:
            UDT = getattr(m, pyClass)
            if not (isinstance(UDT, type) and issubclass(UDT, UserDefinedType)):
                raise PySparkTypeError(
                    errorClass="FIELD_TYPE_MISMATCH",
                    messageParameters={"obj": str(UDT), "data_type": "UserDefinedType"},
                )
        return UDT()


__all__ = [
    "DataType",
    "UserDefinedType",
    "Row",
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
    "VariantVal",
    "GeometryType",
    "GeographyType",
    "Geometry",
    "Geography",
    "_drop_metadata",
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
    atomic/parameterized types, or a dict for the nested struct/array/map forms).

    The string forms (atomics, ``decimal(p,s)``, ``string collate X``, intervals,
    ``geometry(<crs>)``, ``geography(<crs>, <alg>)``, ...) are parsed by the Rust core's
    JSON grammar, so the drop-in keeps one grammar. The nested object forms recurse here,
    because a ``udt`` leaf has to reconstruct its Python class."""
    import json as _json

    if isinstance(v, str):
        from pyspark._pyspark import parse_datatype_json

        try:
            return parse_datatype_json(_json.dumps(v))
        except ValueError:
            raise ValueError(f"cannot parse datatype json: {v!r}") from None
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
        elif t == "udt":
            return UserDefinedType.fromJson(v)
    raise ValueError(f"cannot parse datatype json: {v!r}")


def _drop_metadata(d):
    """Recursively strip StructField metadata (mirrors pyspark.sql.types._drop_metadata)."""
    from typing import cast
    assert isinstance(d, (DataType, StructField))
    if isinstance(d, StructField):
        return StructField(d.name, _drop_metadata(d.dataType), d.nullable, None)
    elif isinstance(d, StructType):
        return StructType([cast(StructField, _drop_metadata(f)) for f in d.fields])
    elif isinstance(d, ArrayType):
        return ArrayType(_drop_metadata(d.elementType), d.containsNull)
    elif isinstance(d, MapType):
        return MapType(_drop_metadata(d.keyType), _drop_metadata(d.valueType), d.valueContainsNull)
    return d

def _has_type(dt, dts):
    """Return whether there are specified types (mirrors pyspark.sql.types._has_type)."""
    if isinstance(dt, dts):
        return True
    elif isinstance(dt, StructType):
        return any(_has_type(f.dataType, dts) for f in dt.fields)
    elif isinstance(dt, ArrayType):
        return _has_type(dt.elementType, dts)
    elif isinstance(dt, MapType):
        return _has_type(dt.keyType, dts) or _has_type(dt.valueType, dts)
    elif isinstance(dt, UserDefinedType):
        return _has_type(dt.sqlType(), dts)
    else:
        return False


def _create_row(fields, values):
    """A Row of ``values`` named by ``fields``, as in PySpark, whose pickled rows call it."""
    row = Row(*values)
    row.__fields__ = fields
    return row
