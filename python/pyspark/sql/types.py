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

    def __reduce__(self):
        # As PySpark's UDT, a Python object, which pickles as its class and its state:
        # cloudpickle sends a class of __main__ by value, and the state keeps the parameters
        # of the instance. DataType.__reduce__ would send its JSON, which names the class by
        # its module and rebuilds the instance without arguments.
        import copyreg

        return copyreg.__newobj__, (type(self),), self.__dict__ or None

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


def _parse_datatype_string(s):
    """The DataType of the DDL string `s`, as the server of the active session parses it, as
    PySpark Connect's `_parse_datatype_string`: a schema of fields, as "a INT, b STRING", or a
    type, as "decimal(10, 2)". It raises NO_ACTIVE_OR_DEFAULT_SESSION without a session, and
    the errors of Spark's parser."""
    from pyspark.sql import SparkSession

    return SparkSession.active()._parse_ddl(s)


def _parse_datatype_json_string(json_string):
    """Reconstruct a DataType from its Spark JSON (client-side; UDF workers use the
    official pyspark function of the same name). Handles the full recursive grammar:
    atomic + parameterized string forms and the nested struct/array/map object forms."""
    import json as _json

    return _parse_datatype_json_value(_json.loads(json_string))


#: The key of the metadata of a struct field of the collations of its strings, as PySpark's.
_COLLATIONS_METADATA_KEY = "__COLLATIONS"

#: The JSON type names to which a collation of a struct field applies, as PySpark's
#: ``_all_mappable_types``: it makes a ``string`` a collated string, and raises
#: ``INVALID_JSON_DATA_TYPE_FOR_COLLATIONS`` for the others.
_MAPPABLE_TYPE_NAMES = frozenset(
    [
        "string", "binary", "boolean", "decimal", "float", "double", "byte", "short",
        "integer", "long", "date", "timestamp", "timestamp_ntz", "void", "variant", "interval",
    ]
)


def _parse_datatype_json_value(json_value, fieldPath="", collationsMap=None):
    """Recursively reconstruct a DataType from a parsed Spark-JSON value (a string for
    atomic/parameterized types, or a dict for the nested struct/array/map forms).

    The string forms (atomics, ``decimal(p,s)``, ``string collate X``, intervals,
    ``geometry(<crs>)``, ``geography(<crs>, <alg>)``, ...) are parsed by the Rust core's
    JSON grammar, so the drop-in keeps one grammar. The nested object forms recurse here,
    because a ``udt`` leaf has to reconstruct its Python class.

    As PySpark's, a string at a ``fieldPath`` of ``collationsMap``, the collations of the
    ``__COLLATIONS`` of the metadata of a struct field by their paths, is of its collation."""
    import json as _json

    if not isinstance(json_value, dict):
        if (
            collationsMap is not None
            and fieldPath in collationsMap
            and json_value in _MAPPABLE_TYPE_NAMES
        ):
            _assert_valid_type_for_collation(fieldPath, json_value, collationsMap)
            return StringType(collationsMap[fieldPath])
        from pyspark._pyspark import parse_datatype_json

        try:
            return parse_datatype_json(_json.dumps(json_value))
        except ValueError:
            # As PySpark's, of a str that is not the JSON value of a data type.
            raise _pyspark_value_error(
                "CANNOT_PARSE_DATATYPE", {"msg": str(json_value)}
            ) from None
    tpe = json_value["type"]
    if tpe in ("array", "map", "struct"):
        if collationsMap is not None and fieldPath in collationsMap:
            _assert_valid_type_for_collation(fieldPath, tpe, collationsMap)
        if tpe == "array":
            return _array_type_from_json(json_value, fieldPath, collationsMap)
        elif tpe == "map":
            return _map_type_from_json(json_value, fieldPath, collationsMap)
        return _struct_type_from_json(json_value)
    elif tpe == "udt":
        return UserDefinedType.fromJson(json_value)
    # As PySpark's, of an object of a type that is not one of a data type.
    raise _pyspark_value_error("UNSUPPORTED_DATA_TYPE", {"data_type": str(tpe)})


def _array_type_from_json(json_value, fieldPath="", collationsMap=None):
    """``ArrayType.fromJson``: its element at ``element`` of ``fieldPath``, as PySpark's."""
    elementType = _parse_datatype_json_value(
        json_value["elementType"],
        "element" if fieldPath == "" else fieldPath + ".element",
        collationsMap,
    )
    return ArrayType(elementType, json_value["containsNull"])


def _map_type_from_json(json_value, fieldPath="", collationsMap=None):
    """``MapType.fromJson``: its key and value at ``key`` and ``value`` of ``fieldPath``, as
    PySpark's."""
    keyType = _parse_datatype_json_value(
        json_value["keyType"], "key" if fieldPath == "" else fieldPath + ".key", collationsMap
    )
    valueType = _parse_datatype_json_value(
        json_value["valueType"],
        "value" if fieldPath == "" else fieldPath + ".value",
        collationsMap,
    )
    return MapType(keyType, valueType, json_value["valueContainsNull"])


def _struct_field_from_json(json_value):
    """``StructField.fromJson``, as PySpark's: the ``__COLLATIONS`` of its metadata,
    ``<provider>.<collation>`` by the paths of its strings, are the collations of those
    strings, and not metadata."""
    metadata = json_value.get("metadata")
    collationsMap = {}
    if metadata and _COLLATIONS_METADATA_KEY in metadata:
        for key, value in metadata[_COLLATIONS_METADATA_KEY].items():
            nameParts = value.split(".")
            assert len(nameParts) == 2
            provider, name = nameParts[0], nameParts[1]
            _assert_valid_collation_provider(provider)
            collationsMap[key] = name
        metadata = {
            key: value for key, value in metadata.items() if key != _COLLATIONS_METADATA_KEY
        }
    return StructField(
        json_value["name"],
        _parse_datatype_json_value(json_value["type"], json_value["name"], collationsMap),
        json_value.get("nullable", True),
        metadata,
    )


def _struct_type_from_json(json_value):
    """``StructType.fromJson``, as PySpark's."""
    return StructType([_struct_field_from_json(f) for f in json_value["fields"]])


def _assert_valid_type_for_collation(fieldPath, fieldType, collationMap):
    """As PySpark's: a collation is of a string."""
    if fieldPath in collationMap and fieldType != "string":
        from pyspark.errors import PySparkTypeError

        raise PySparkTypeError(
            errorClass="INVALID_JSON_DATA_TYPE_FOR_COLLATIONS",
            messageParameters={"jsonType": fieldType},
        )


def _assert_valid_collation_provider(provider):
    """As PySpark's: a collation is of a provider of ``StringType.providers``."""
    if provider.lower() not in StringType.providers:
        raise _pyspark_value_error(
            "COLLATION_INVALID_PROVIDER",
            {"provider": provider, "supportedProviders": ", ".join(StringType.providers)},
        )


def _pyspark_value_error(error_class, parameters):
    from pyspark.errors import PySparkValueError

    return PySparkValueError(errorClass=error_class, messageParameters=parameters)


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


def _from_numpy_type(nt):
    """Convert NumPy type to Spark data type."""
    import numpy as np

    if nt == np.dtype("bool"):
        return BooleanType()
    elif nt == np.dtype("int8"):
        return ByteType()
    elif nt == np.dtype("int16"):
        return ShortType()
    elif nt == np.dtype("int32"):
        return IntegerType()
    elif nt == np.dtype("int64"):
        return LongType()
    elif nt == np.dtype("float32"):
        return FloatType()
    elif nt == np.dtype("float64"):
        return DoubleType()
    elif nt.type == np.dtype("str"):
        return StringType()

    return None


def _create_row(fields, values):
    """A Row of ``values`` named by ``fields``, as in PySpark, whose pickled rows call it."""
    row = Row(*values)
    row.__fields__ = fields
    return row


def _udt_converter(dataType):
    """The function that turns a collected value of ``dataType`` into what PySpark gives for it,
    deserializing the values of its user-defined types, and making the Geometry and Geography
    of its spatial values, as PySpark's ``ArrowTableToRowsConversion`` does, or None for a type
    without either."""
    if isinstance(dataType, (GeometryType, GeographyType)):
        spatial = Geometry if isinstance(dataType, GeometryType) else Geography

        def convert_spatial(value):
            if value is None:
                return None
            return spatial.fromWKB(value["wkb"], value["srid"])

        return convert_spatial
    if isinstance(dataType, UserDefinedType):
        inner = _udt_converter(dataType.sqlType())

        def convert_udt(value):
            if value is None:
                return None
            return dataType.deserialize(value if inner is None else inner(value))

        return convert_udt
    if isinstance(dataType, ArrayType):
        element = _udt_converter(dataType.elementType)
        if element is None:
            return None
        return lambda value: None if value is None else [element(v) for v in value]
    if isinstance(dataType, MapType):
        key, val = _udt_converter(dataType.keyType), _udt_converter(dataType.valueType)
        if key is None and val is None:
            return None
        key = key or (lambda v: v)
        val = val or (lambda v: v)
        return lambda value: None if value is None else {key(k): val(v) for k, v in value.items()}
    if isinstance(dataType, StructType):
        fields = _udt_field_converters(dataType)
        if fields is None:
            return None
        row = Row(*dataType.names)

        def convert_struct(value):
            if value is None:
                return None
            return row(*[v if f is None else f(v) for f, v in zip(fields, value)])

        return convert_struct
    return None


def _udt_field_converters(schema):
    """The converters of the fields of ``schema`` (see ``_udt_converter``), or None when no
    field has a user-defined type. The client applies them to the rows that it collects."""
    converters = [_udt_converter(field.dataType) for field in schema.fields]
    return None if all(c is None for c in converters) else converters


def _to_internal(dataType, obj):
    """The internal SQL value of the Python value ``obj``, as the ``toInternal`` of the PySpark
    type of ``dataType``: the days of a date since the epoch, the microseconds of a timestamp
    and of a day-time interval, the nanoseconds of a time, and the dicts of variants and spatial
    values, also in the elements, keys, values and fields of arrays, maps and structs."""
    import calendar
    import time

    from pyspark.errors import PySparkNotImplementedError

    if isinstance(dataType, StructField):
        return dataType.dataType.toInternal(obj)
    elif isinstance(dataType, DateType):
        if obj is not None:
            return obj.toordinal() - DateType.EPOCH_ORDINAL
    elif isinstance(dataType, TimeType):
        if obj is not None:
            return (
                obj.hour * 3_600_000_000_000
                + obj.minute * 60_000_000_000
                + obj.second * 1_000_000_000
                + obj.microsecond * 1_000
            )
    elif isinstance(dataType, TimestampType):
        if obj is not None:
            seconds = (
                calendar.timegm(obj.utctimetuple()) if obj.tzinfo else time.mktime(obj.timetuple())
            )
            return int(seconds) * 1000000 + obj.microsecond
    elif isinstance(dataType, TimestampNTZType):
        if obj is not None:
            seconds = calendar.timegm(obj.timetuple())
            return int(seconds) * 1000000 + obj.microsecond
    elif isinstance(dataType, DayTimeIntervalType):
        if obj is not None:
            return (((obj.days * 86400) + obj.seconds) * 1_000_000) + obj.microseconds
    elif isinstance(dataType, (YearMonthIntervalType, CalendarIntervalType)):
        raise PySparkNotImplementedError(
            errorClass="NOT_IMPLEMENTED",
            messageParameters={"feature": "%s.toInternal" % type(dataType).__name__},
        )
    elif isinstance(dataType, VariantType):
        if obj is None:
            return None
        assert isinstance(obj, VariantVal)
        return {"value": obj.value, "metadata": obj.metadata}
    elif isinstance(dataType, (GeometryType, GeographyType)):
        if obj is None:
            return None
        assert isinstance(obj, Geometry if isinstance(dataType, GeometryType) else Geography)
        return {"srid": obj.srid, "wkb": obj.wkb}
    elif isinstance(dataType, ArrayType):
        if not dataType.needConversion():
            return obj
        return obj and [dataType.elementType.toInternal(v) for v in obj]
    elif isinstance(dataType, MapType):
        if not dataType.needConversion():
            return obj
        return obj and dict(
            (dataType.keyType.toInternal(k), dataType.valueType.toInternal(v))
            for k, v in obj.items()
        )
    elif isinstance(dataType, StructType):
        return _struct_to_internal(dataType, obj)
    else:
        return obj


def _struct_to_internal(dataType, obj):
    """``StructType.toInternal``, as PySpark's: the tuple of the values of the fields of a
    dict, a tuple, a list, a Row or an object, converted where a field needs it."""
    from pyspark.errors import PySparkValueError

    if obj is None:
        return
    if isinstance(obj, VariantVal):
        raise PySparkValueError("Rows cannot be of type VariantVal")
    names, fields = dataType.names, dataType.fields
    # A Row is a tuple in PySpark.
    if dataType._needSerializeAnyField:
        # Only calling toInternal function for fields that need conversion
        need = dataType._needConversion
        if isinstance(obj, dict):
            return tuple(
                f.toInternal(obj.get(n)) if c else obj.get(n)
                for n, f, c in zip(names, fields, need)
            )
        elif isinstance(obj, (tuple, list, Row)):
            return tuple(f.toInternal(v) if c else v for f, v, c in zip(fields, obj, need))
        elif hasattr(obj, "__dict__"):
            d = obj.__dict__
            return tuple(
                f.toInternal(d.get(n)) if c else d.get(n) for n, f, c in zip(names, fields, need)
            )
    else:
        if isinstance(obj, dict):
            return tuple(obj.get(n) for n in names)
        elif isinstance(obj, (list, tuple, Row)):
            return tuple(obj)
        elif hasattr(obj, "__dict__"):
            d = obj.__dict__
            return tuple(d.get(n) for n in names)
    raise PySparkValueError(
        errorClass="UNEXPECTED_TUPLE_WITH_STRUCT",
        messageParameters={"tuple": str(obj)},
    )


def _from_internal(dataType, obj):
    """The Python value of the internal SQL value ``obj``, as the ``fromInternal`` of the
    PySpark type of ``dataType``: the inverse of ``_to_internal``, of which a struct is a
    Row."""
    import datetime

    from pyspark.errors import PySparkNotImplementedError

    if isinstance(dataType, StructField):
        return dataType.dataType.fromInternal(obj)
    elif isinstance(dataType, DateType):
        if obj is not None:
            return datetime.date.fromordinal(obj + DateType.EPOCH_ORDINAL)
    elif isinstance(dataType, TimeType):
        if obj is not None:
            hours, remainder = divmod(obj, 3_600_000_000_000)
            minutes, remainder = divmod(remainder, 60_000_000_000)
            seconds, remainder = divmod(remainder, 1_000_000_000)
            microseconds = remainder // 1_000
            return datetime.time(hours, minutes, seconds, microseconds)
    elif isinstance(dataType, TimestampType):
        if obj is not None:
            # using int to avoid precision loss in float
            return datetime.datetime.fromtimestamp(obj // 1000000).replace(
                microsecond=obj % 1000000
            )
    elif isinstance(dataType, TimestampNTZType):
        if obj is not None:
            # using int to avoid precision loss in float
            return datetime.datetime.fromtimestamp(obj // 1000000, datetime.timezone.utc).replace(
                microsecond=obj % 1000000, tzinfo=None
            )
    elif isinstance(dataType, DayTimeIntervalType):
        if obj is not None:
            return datetime.timedelta(microseconds=obj)
    elif isinstance(dataType, (YearMonthIntervalType, CalendarIntervalType)):
        raise PySparkNotImplementedError(
            errorClass="NOT_IMPLEMENTED",
            messageParameters={"feature": "%s.fromInternal" % type(dataType).__name__},
        )
    elif isinstance(dataType, VariantType):
        if obj is None or not all(key in obj for key in ["value", "metadata"]):
            return None
        return VariantVal(obj["value"], obj["metadata"])
    elif isinstance(dataType, (GeometryType, GeographyType)):
        if obj is None or not all(key in obj for key in ["srid", "bytes"]):
            return None
        cls = Geometry if isinstance(dataType, GeometryType) else Geography
        return cls(obj["bytes"], obj["srid"])
    elif isinstance(dataType, ArrayType):
        if not dataType.needConversion():
            return obj
        return obj and [dataType.elementType.fromInternal(v) for v in obj]
    elif isinstance(dataType, MapType):
        if not dataType.needConversion():
            return obj
        return obj and dict(
            (dataType.keyType.fromInternal(k), dataType.valueType.fromInternal(v))
            for k, v in obj.items()
        )
    elif isinstance(dataType, StructType):
        if obj is None:
            return
        if isinstance(obj, Row):
            # it's already converted by pickler
            return obj
        if dataType._needSerializeAnyField:
            # Only calling fromInternal function for fields that need conversion
            values = [
                f.fromInternal(v) if c else v
                for f, v, c in zip(dataType.fields, obj, dataType._needConversion)
            ]
        else:
            values = obj
        return _create_row(dataType.names, values)
    else:
        return obj
