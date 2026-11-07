"""
Minimal pyspark.sql.types module for the Spark Connect client.
Provides DataType classes and conversion functions.
"""
import re
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
    # The extension exports the same tuple-backed Row as the public Python imports.
    from pyspark._pyspark import Row
except ImportError:  # pragma: no cover - defensive fallback when the extension is absent
    # Minimal DataType stubs for import-only tests. Row and runtime conversion
    # helpers require the compiled extension.
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
        if isinstance(json_value, str) and json_value.startswith(("geometry", "geography")):
            return _spatial_type_from_json(json_value)
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


# The patterns of the JSON of the spatial types, as PySpark's.
_GEOMETRY = re.compile(r"^geometry$")
_GEOMETRY_CRS = re.compile(r"geometry\(\s*([\w]+:-?[\w]+)\s*\)")
_GEOGRAPHY = re.compile(r"^geography$")
_GEOGRAPHY_CRS = re.compile(r"geography\(\s*([\w]+:-?[\w]+)\s*\)")
_GEOGRAPHY_CRS_ALG = re.compile(r"geography\(\s*([\w]+:-?[\w]+)\s*,\s*(\w+)\s*\)")
_GEOGRAPHY_ALG = re.compile(r"geography\(\s*(\w+)\s*\)")


def _spatial_type_from_json(json_value):
    """The GeometryType or GeographyType of the JSON string ``json_value``, as PySpark's
    ``_parse_datatype_json_value`` parses it, of its patterns: of the default CRS
    ``OGC:CRS84`` and algorithm ``SPHERICAL`` without them, and ``CANNOT_PARSE_DATATYPE`` of
    another form."""
    if _GEOMETRY.match(json_value):
        return _geometry_type_from_crs("OGC:CRS84")
    elif _GEOMETRY_CRS.match(json_value):
        return _geometry_type_from_crs(_GEOMETRY_CRS.match(json_value).group(1))
    elif _GEOGRAPHY.match(json_value):
        return _geography_type_from_crs("OGC:CRS84", "SPHERICAL")
    elif _GEOGRAPHY_CRS.match(json_value):
        return _geography_type_from_crs(_GEOGRAPHY_CRS.match(json_value).group(1), "SPHERICAL")
    elif _GEOGRAPHY_CRS_ALG.match(json_value):
        crs_alg = _GEOGRAPHY_CRS_ALG.match(json_value)
        return _geography_type_from_crs(crs_alg.group(1), crs_alg.group(2))
    elif _GEOGRAPHY_ALG.match(json_value):
        return _geography_type_from_crs("OGC:CRS84", _GEOGRAPHY_ALG.match(json_value).group(1))
    raise _pyspark_value_error("CANNOT_PARSE_DATATYPE", {"msg": str(json_value)})


def _geometry_type_from_crs(crs):
    """``GeometryType._from_crs`` of PySpark: the mixed SRID of the CRS ``SRID:ANY`` of any
    case, or else the SRID of a CRS of ``geo_utils``, and ``ST_INVALID_CRS_VALUE`` of another."""
    from pyspark.errors import IllegalArgumentException
    from pyspark.sql.geo_utils import CartesianSpatialReferenceSystemMapper

    if crs.lower() == "srid:any":
        return GeometryType("ANY")
    srid = CartesianSpatialReferenceSystemMapper.get_srid(crs)
    if srid is None:
        raise IllegalArgumentException(
            errorClass="ST_INVALID_CRS_VALUE", messageParameters={"crs": str(crs)}
        )
    return GeometryType(srid)


def _geography_type_from_crs(crs, alg):
    """``GeographyType._from_crs`` of PySpark: ``ST_INVALID_ALGORITHM_VALUE`` of another
    algorithm than ``SPHERICAL``, the mixed SRID of the CRS ``SRID:ANY`` of any case, or else
    the SRID of a geographic CRS of ``geo_utils``, and ``ST_INVALID_CRS_VALUE`` of another."""
    from pyspark.errors import IllegalArgumentException
    from pyspark.sql.geo_utils import GeographicSpatialReferenceSystemMapper

    if alg != "SPHERICAL":
        raise IllegalArgumentException(
            errorClass="ST_INVALID_ALGORITHM_VALUE", messageParameters={"alg": str(alg)}
        )
    if crs.lower() == "srid:any":
        return GeographyType("ANY")
    srid = GeographicSpatialReferenceSystemMapper.get_srid(crs)
    if srid is None:
        raise IllegalArgumentException(
            errorClass="ST_INVALID_CRS_VALUE", messageParameters={"crs": str(crs)}
        )
    return GeographyType(srid)


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
        field_names = dataType.names

        def convert_struct(value):
            if value is None:
                return None
            return _create_row(field_names, [v if f is None else f(v) for f, v in zip(fields, value)])

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


# The private helpers of PySpark's pyspark.sql.types, as of Apache Spark v4.3.0: the inference,
# merging, conversion and verification of the types of Python objects, and their tables, which
# PySpark Connect's modules, Apache Spark's tests and libraries import. They are PySpark's, as
# the client's types share the classes and attributes of PySpark's, including its tuple Row.
import ctypes  # noqa: E402
import datetime  # noqa: E402
import decimal  # noqa: E402
import sys  # noqa: E402
from array import array  # noqa: E402
from collections.abc import Iterable  # noqa: E402
from functools import reduce  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, overload  # noqa: E402

from pyspark.errors import PySparkTypeError, PySparkValueError  # noqa: E402
from pyspark.util import JVM_INT_MAX  # noqa: E402, F401

_atomic_types: List[Type[DataType]] = [
    StringType,
    CharType,
    VarcharType,
    BinaryType,
    BooleanType,
    DecimalType,
    FloatType,
    DoubleType,
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    DateType,
    TimestampType,
    TimestampNTZType,
    NullType,
    VariantType,
    YearMonthIntervalType,
    DayTimeIntervalType,
]

_complex_types: List[Type[Union[ArrayType, MapType, StructType]]] = [
    ArrayType,
    MapType,
    StructType,
]
_all_complex_types: Dict[str, Type[Union[ArrayType, MapType, StructType]]] = {
    "array": ArrayType,
    "map": MapType,
    "struct": StructType,
}

# Datatypes that can be directly parsed by mapping a json string without regex.
# This dict should be only used in json parsing.
# Note that:
# 1, CharType and VarcharType are not listed here, since they need regex;
# 2, DecimalType can be parsed by both mapping ('decimal') and regex ('decimal(10, 2)');
# 3, CalendarIntervalType is not an atomic type, but can be mapped by 'interval';
_all_mappable_types: Dict[str, Type[DataType]] = {
    "string": StringType,
    "binary": BinaryType,
    "boolean": BooleanType,
    "decimal": DecimalType,
    "float": FloatType,
    "double": DoubleType,
    "byte": ByteType,
    "short": ShortType,
    "integer": IntegerType,
    "long": LongType,
    "date": DateType,
    "timestamp": TimestampType,
    "timestamp_ntz": TimestampNTZType,
    "void": NullType,
    "variant": VariantType,
    "interval": CalendarIntervalType,
}

_type_mappings = {
    type(None): NullType,
    Geometry: GeometryType,
    Geography: GeographyType,
    bool: BooleanType,
    int: LongType,
    float: DoubleType,
    str: StringType,
    bytearray: BinaryType,
    decimal.Decimal: DecimalType,
    datetime.date: DateType,
    datetime.datetime: TimestampType,  # can be TimestampNTZType
    datetime.time: TimeType,
    datetime.timedelta: DayTimeIntervalType,
    bytes: BinaryType,
}

# Mapping Python array types to Spark SQL DataType
# We should be careful here. The size of these types in python depends on C
# implementation. We need to make sure that this conversion does not lose any
# precision. Also, JVM only support signed types, when converting unsigned types,
# keep in mind that it require 1 more bit when stored as signed types.
#
# Reference for C integer size, see:
# ISO/IEC 9899:201x specification, chapter 5.2.4.2.1 Sizes of integer types <limits.h>.
# Reference for python array typecode, see:
# https://docs.python.org/2/library/array.html
# https://docs.python.org/3.6/library/array.html
# Reference for JVM's supported integral types:
# http://docs.oracle.com/javase/specs/jvms/se8/html/jvms-2.html#jvms-2.3.1

_array_signed_int_typecode_ctype_mappings = {
    "b": ctypes.c_byte,
    "h": ctypes.c_short,
    "i": ctypes.c_int,
    "l": ctypes.c_long,
}

_array_unsigned_int_typecode_ctype_mappings = {
    "B": ctypes.c_ubyte,
    "H": ctypes.c_ushort,
    "I": ctypes.c_uint,
    "L": ctypes.c_ulong,
}


def _int_size_to_type(
    size: int,
) -> Optional[Union[Type[ByteType], Type[ShortType], Type[IntegerType], Type[LongType]]]:
    """
    Return the Catalyst datatype from the size of integers.
    """
    if size <= 8:
        return ByteType
    elif size <= 16:
        return ShortType
    elif size <= 32:
        return IntegerType
    elif size <= 64:
        return LongType
    else:
        return None


# The list of all supported array typecodes, is stored here
_array_type_mappings: Dict[str, Type[DataType]] = {
    # Warning: Actual properties for float and double in C is not specified in C.
    # On almost every system supported by both python and JVM, they are IEEE 754
    # single-precision binary floating-point format and IEEE 754 double-precision
    # binary floating-point format. And we do assume the same thing here for now.
    "f": FloatType,
    "d": DoubleType,
}

# compute array typecode mappings for signed integer types
for _typecode in _array_signed_int_typecode_ctype_mappings:
    size = ctypes.sizeof(_array_signed_int_typecode_ctype_mappings[_typecode]) * 8
    dt = _int_size_to_type(size)
    if dt is not None:
        _array_type_mappings[_typecode] = dt

# compute array typecode mappings for unsigned integer types
for _typecode in _array_unsigned_int_typecode_ctype_mappings:
    # JVM does not have unsigned types, so use signed types that is at least 1
    # bit larger to store
    size = ctypes.sizeof(_array_unsigned_int_typecode_ctype_mappings[_typecode]) * 8 + 1
    dt = _int_size_to_type(size)
    if dt is not None:
        _array_type_mappings[_typecode] = dt

# Type code 'u' in Python's array is deprecated since version 3.3, and will be
# removed in version 4.0. See: https://docs.python.org/3/library/array.html
if sys.version_info[0] < 4:
    _array_type_mappings["u"] = StringType


def _infer_type(
    obj: Any,
    infer_dict_as_struct: bool = False,
    infer_array_from_first_element: bool = False,
    infer_map_from_first_pair: bool = False,
    prefer_timestamp_ntz: bool = False,
) -> DataType:
    """Infer the DataType from obj"""
    if obj is None:
        return NullType()

    if hasattr(obj, "__UDT__"):
        return obj.__UDT__

    dataType = _type_mappings.get(type(obj))
    if dataType is GeographyType:
        assert isinstance(obj, Geography)
        return GeographyType(obj.getSrid())
    if dataType is GeometryType:
        assert isinstance(obj, Geometry)
        return GeometryType(obj.getSrid())
    if dataType is DecimalType:
        # the precision and scale of `obj` may be different from row to row.
        return DecimalType(38, 18)
    if dataType is TimestampType and prefer_timestamp_ntz and obj.tzinfo is None:
        return TimestampNTZType()
    if dataType is DayTimeIntervalType:
        return DayTimeIntervalType()
    if dataType is YearMonthIntervalType:
        return YearMonthIntervalType()
    if dataType is CalendarIntervalType:
        return CalendarIntervalType()
    elif dataType is not None:
        return dataType()

    if isinstance(obj, dict):
        if infer_dict_as_struct:
            struct = StructType()
            for key, value in obj.items():
                if key is not None and value is not None:
                    struct.add(
                        key,
                        _infer_type(
                            value,
                            infer_dict_as_struct,
                            infer_array_from_first_element,
                            infer_map_from_first_pair,
                            prefer_timestamp_ntz,
                        ),
                        True,
                    )
            return struct
        elif infer_map_from_first_pair:
            for key, value in obj.items():
                if key is not None and value is not None:
                    return MapType(
                        _infer_type(
                            key,
                            infer_dict_as_struct,
                            infer_array_from_first_element,
                            infer_map_from_first_pair,
                            prefer_timestamp_ntz,
                        ),
                        _infer_type(
                            value,
                            infer_dict_as_struct,
                            infer_array_from_first_element,
                            infer_map_from_first_pair,
                            prefer_timestamp_ntz,
                        ),
                        True,
                    )
            return MapType(NullType(), NullType(), True)
        else:
            key_type: DataType = NullType()
            value_type: DataType = NullType()
            for key, value in obj.items():
                if key is not None:
                    key_type = _merge_type(
                        key_type,
                        _infer_type(
                            key,
                            infer_dict_as_struct,
                            infer_array_from_first_element,
                            infer_map_from_first_pair,
                            prefer_timestamp_ntz,
                        ),
                    )
                if value is not None:
                    value_type = _merge_type(
                        value_type,
                        _infer_type(
                            value,
                            infer_dict_as_struct,
                            infer_array_from_first_element,
                            infer_map_from_first_pair,
                            prefer_timestamp_ntz,
                        ),
                    )

            return MapType(key_type, value_type, True)
    elif isinstance(obj, list):
        if len(obj) > 0:
            if infer_array_from_first_element:
                return ArrayType(
                    _infer_type(
                        obj[0],
                        infer_dict_as_struct,
                        infer_array_from_first_element,
                        prefer_timestamp_ntz,
                    ),
                    True,
                )
            else:
                return ArrayType(
                    reduce(
                        _merge_type,
                        (
                            _infer_type(
                                v,
                                infer_dict_as_struct,
                                infer_array_from_first_element,
                                prefer_timestamp_ntz,
                            )
                            for v in obj
                        ),
                    ),
                    True,
                )
        return ArrayType(NullType(), True)
    elif isinstance(obj, array):
        if obj.typecode in _array_type_mappings:
            return ArrayType(_array_type_mappings[obj.typecode](), False)
        else:
            raise PySparkTypeError(
                errorClass="UNSUPPORTED_DATA_TYPE",
                messageParameters={"data_type": f"array({obj.typecode})"},
            )
    elif isinstance(obj, VariantVal):
        return VariantType()
    else:
        try:
            return _infer_schema(
                obj,
                infer_dict_as_struct=infer_dict_as_struct,
                infer_array_from_first_element=infer_array_from_first_element,
                prefer_timestamp_ntz=prefer_timestamp_ntz,
            )
        except TypeError:
            raise PySparkTypeError(
                errorClass="UNSUPPORTED_DATA_TYPE",
                messageParameters={"data_type": type(obj).__name__},
            )


def _infer_schema(
    row: Any,
    names: Optional[List[str]] = None,
    infer_dict_as_struct: bool = False,
    infer_array_from_first_element: bool = False,
    infer_map_from_first_pair: bool = False,
    prefer_timestamp_ntz: bool = False,
) -> StructType:
    """Infer the schema from dict/namedtuple/object"""
    items: Iterable[Tuple[str, Any]]
    if isinstance(row, dict):
        items = sorted(row.items())

    elif isinstance(row, (tuple, list, Row)):
        if hasattr(row, "__fields__"):  # Row
            items = zip(row.__fields__, tuple(row))
        elif hasattr(row, "_fields"):  # namedtuple
            items = zip(row._fields, tuple(row))
        else:
            if names is None:
                names = ["_%d" % i for i in range(1, len(row) + 1)]
            elif len(names) < len(row):
                names.extend("_%d" % i for i in range(len(names) + 1, len(row) + 1))
            items = zip(names, row)

    elif hasattr(row, "__dict__"):  # object
        items = sorted(row.__dict__.items())

    else:
        raise PySparkTypeError(
            errorClass="CANNOT_INFER_SCHEMA_FOR_TYPE",
            messageParameters={"data_type": type(row).__name__},
        )

    fields = []
    for k, v in items:
        try:
            fields.append(
                StructField(
                    k,
                    _infer_type(
                        v,
                        infer_dict_as_struct,
                        infer_array_from_first_element,
                        infer_map_from_first_pair,
                        prefer_timestamp_ntz,
                    ),
                    True,
                )
            )
        except TypeError:
            raise PySparkTypeError(
                errorClass="CANNOT_INFER_TYPE_FOR_FIELD",
                messageParameters={"field_name": k},
            )
    return StructType(fields)


def _has_nulltype(dt: DataType) -> bool:
    """Return whether there is a NullType in `dt` or not"""
    if isinstance(dt, StructType):
        return any(_has_nulltype(f.dataType) for f in dt.fields)
    elif isinstance(dt, ArrayType):
        return _has_nulltype((dt.elementType))
    elif isinstance(dt, MapType):
        return _has_nulltype(dt.keyType) or _has_nulltype(dt.valueType)
    else:
        return isinstance(dt, NullType)


def _merge_type(a: StructType, b: StructType, name: Optional[str] = None) -> StructType: ...


@overload
def _merge_type(a: ArrayType, b: ArrayType, name: Optional[str] = None) -> ArrayType: ...


@overload
def _merge_type(a: MapType, b: MapType, name: Optional[str] = None) -> MapType: ...


@overload
def _merge_type(a: DataType, b: DataType, name: Optional[str] = None) -> DataType: ...


def _merge_type(
    a: Union[StructType, ArrayType, MapType, DataType],
    b: Union[StructType, ArrayType, MapType, DataType],
    name: Optional[str] = None,
) -> Union[StructType, ArrayType, MapType, DataType]:
    if name is None:

        def new_msg(msg: str) -> str:
            return msg

        def new_name(n: str) -> str:
            return "field %s" % n

    else:

        def new_msg(msg: str) -> str:
            return "%s: %s" % (name, msg)

        def new_name(n: str) -> str:
            return "field %s in %s" % (n, name)

    if isinstance(a, NullType):
        return b
    elif isinstance(b, NullType):
        return a
    elif isinstance(a, TimestampType) and isinstance(b, TimestampNTZType):
        return a
    elif isinstance(a, TimestampNTZType) and isinstance(b, TimestampType):
        return b
    elif isinstance(a, GeometryType) and isinstance(b, GeometryType) and a.srid != b.srid:
        return GeometryType("ANY")
    elif isinstance(a, GeographyType) and isinstance(b, GeographyType) and a.srid != b.srid:
        return GeographyType("ANY")
    elif isinstance(a, AtomicType) and isinstance(b, StringType):
        return b
    elif isinstance(a, StringType) and isinstance(b, AtomicType):
        return a
    elif type(a) is not type(b):
        # TODO: type cast (such as int -> long)
        raise PySparkTypeError(
            errorClass="CANNOT_MERGE_TYPE",
            messageParameters={"data_type1": type(a).__name__, "data_type2": type(b).__name__},
        )

    # same type
    if isinstance(a, StructType):
        nfs = dict((f.name, f.dataType) for f in cast(StructType, b).fields)
        fields = [
            StructField(
                f.name, _merge_type(f.dataType, nfs.get(f.name, NullType()), name=new_name(f.name))
            )
            for f in a.fields
        ]
        names = set([f.name for f in fields])
        for n in nfs:
            if n not in names:
                fields.append(StructField(n, nfs[n]))
        return StructType(fields)

    elif isinstance(a, ArrayType):
        return ArrayType(
            _merge_type(
                a.elementType, cast(ArrayType, b).elementType, name="element in array %s" % name
            ),
            True,
        )

    elif isinstance(a, MapType):
        return MapType(
            _merge_type(a.keyType, cast(MapType, b).keyType, name="key of map %s" % name),
            _merge_type(a.valueType, cast(MapType, b).valueType, name="value of map %s" % name),
            True,
        )
    else:
        return a


def _need_converter(dataType: DataType) -> bool:
    if isinstance(dataType, StructType):
        return True
    elif isinstance(dataType, ArrayType):
        return _need_converter(dataType.elementType)
    elif isinstance(dataType, MapType):
        return _need_converter(dataType.keyType) or _need_converter(dataType.valueType)
    elif isinstance(dataType, NullType):
        return True
    elif isinstance(dataType, StringType):
        # Coercion to StringType is allowed, e.g. dict -> str
        return True
    else:
        return False


def _create_converter(dataType: DataType) -> Callable:
    """Create a converter to drop the names of fields in obj"""
    if not _need_converter(dataType):
        return lambda x: x

    if isinstance(dataType, ArrayType):
        conv = _create_converter(dataType.elementType)
        return lambda row: [conv(v) for v in row] if row is not None else None

    elif isinstance(dataType, MapType):
        kconv = _create_converter(dataType.keyType)
        vconv = _create_converter(dataType.valueType)
        return lambda row: (
            dict((kconv(k), vconv(v)) for k, v in row.items()) if row is not None else None
        )

    elif isinstance(dataType, NullType):
        return lambda x: None

    elif isinstance(dataType, StringType):

        def convert_string(value: Any) -> Any:
            if value is None:
                return None
            else:
                if isinstance(value, bool):
                    # To match the Classic behavior
                    return str(value).lower()
                else:
                    # Coercion to StringType is allowed, e.g. dict -> str
                    return str(value)

        return convert_string

    elif not isinstance(dataType, StructType):
        return lambda x: x

    # dataType must be StructType
    names = [f.name for f in dataType.fields]
    converters = [_create_converter(f.dataType) for f in dataType.fields]
    convert_fields = any(_need_converter(f.dataType) for f in dataType.fields)

    def convert_struct(obj: Any) -> Optional[Tuple]:
        if obj is None:
            return None

        if isinstance(obj, (tuple, list, Row)):
            if convert_fields:
                return tuple(conv(v) for v, conv in zip(obj, converters))
            else:
                return tuple(obj)

        if isinstance(obj, dict):
            d = obj
        elif hasattr(obj, "__dict__"):  # object
            d = obj.__dict__
        else:
            raise PySparkTypeError(
                errorClass="UNSUPPORTED_DATA_TYPE",
                messageParameters={"data_type": type(obj).__name__},
            )

        if convert_fields:
            return tuple([conv(d.get(name)) for name, conv in zip(names, converters)])
        else:
            return tuple([d.get(name) for name in names])

    return convert_struct


_acceptable_types = {
    BooleanType: (bool,),
    ByteType: (int,),
    ShortType: (int,),
    IntegerType: (int,),
    LongType: (int,),
    FloatType: (float,),
    DoubleType: (float,),
    DecimalType: (decimal.Decimal,),
    StringType: (str,),
    CharType: (str,),
    VarcharType: (str,),
    BinaryType: (bytearray, bytes),
    DateType: (datetime.date, datetime.datetime),
    TimeType: (datetime.time,),
    TimestampType: (datetime.datetime,),
    TimestampNTZType: (datetime.datetime,),
    DayTimeIntervalType: (datetime.timedelta,),
    ArrayType: (list, tuple, array),
    MapType: (dict,),
    StructType: (tuple, list, dict, Row),
    GeometryType: (Geometry,),
    GeographyType: (Geography,),
    VariantType: (
        bool,
        int,
        float,
        decimal.Decimal,
        str,
        bytearray,
        bytes,
        datetime.date,
        datetime.datetime,
        datetime.timedelta,
        tuple,
        Row,
        list,
        dict,
        array,
    ),
}


def _make_type_verifier(
    dataType: DataType,
    nullable: bool = True,
    name: Optional[str] = None,
) -> Callable:
    """
    Make a verifier that checks the type of obj against dataType and raises a TypeError if they do
    not match.

    This verifier also checks the value of obj against datatype and raises a ValueError if it's not
    within the allowed range, e.g. using 128 as ByteType will overflow. Note that, Python float is
    not checked, so it will become infinity when cast to Java float, if it overflows.

    Examples
    --------
    >>> _make_type_verifier(StructType([]))(None)
    >>> _make_type_verifier(StringType())("")
    >>> _make_type_verifier(LongType())(0)
    >>> _make_type_verifier(LongType())(1 << 64) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    pyspark.errors.exceptions.base.PySparkValueError:...
    >>> _make_type_verifier(ArrayType(ShortType()))(list(range(3)))
    >>> _make_type_verifier(ArrayType(StringType()))(set()) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    pyspark.errors.exceptions.base.PySparkTypeError:...
    >>> _make_type_verifier(MapType(StringType(), IntegerType()))({})
    >>> _make_type_verifier(StructType([]))(())
    >>> _make_type_verifier(StructType([]))([])
    >>> _make_type_verifier(StructType([]))([1]) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    pyspark.errors.exceptions.base.PySparkValueError:...
    >>> # Check if numeric values are within the allowed range.
    >>> _make_type_verifier(ByteType())(12)
    >>> _make_type_verifier(ByteType())(1234) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    pyspark.errors.exceptions.base.PySparkValueError:...
    >>> _make_type_verifier(ByteType(), False)(None) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    pyspark.errors.exceptions.base.PySparkValueError:...
    >>> _make_type_verifier(
    ...     ArrayType(ShortType(), False))([1, None]) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    pyspark.errors.exceptions.base.PySparkValueError:...
    >>> _make_type_verifier(  # doctest: +IGNORE_EXCEPTION_DETAIL
    ...     MapType(StringType(), IntegerType())
    ...     )({None: 1})
    Traceback (most recent call last):
        ...
    pyspark.errors.exceptions.base.PySparkValueError:...
    >>> schema = StructType().add("a", IntegerType()).add("b", StringType(), False)
    >>> _make_type_verifier(schema)((1, None)) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    pyspark.errors.exceptions.base.PySparkValueError:...
    """

    if name is None:

        def new_msg(msg: str) -> str:
            return msg

        def new_name(n: str) -> str:
            return "field %s" % n

    else:

        def new_msg(msg: str) -> str:
            return "%s: %s" % (name, msg)

        def new_name(n: str) -> str:
            return "field %s in %s" % (n, name)

    def verify_nullability(obj: Any) -> bool:
        if obj is None or (isinstance(obj, decimal.Decimal) and obj.is_nan()):
            # Spark's DecimalType doesn't support NaN,
            # casting DoubleType NaN to DecimalType will return Null.
            if nullable:
                return True
            else:
                if name is not None:
                    raise PySparkValueError(
                        errorClass="FIELD_NOT_NULLABLE_WITH_NAME",
                        messageParameters={
                            "field_name": str(name),
                        },
                    )
                raise PySparkValueError(
                    errorClass="FIELD_NOT_NULLABLE",
                    messageParameters={},
                )
        else:
            return False

    _type = type(dataType)

    def assert_acceptable_types(obj: Any) -> None:
        assert _type in _acceptable_types, new_msg(
            "unknown datatype: %s for object %r" % (dataType, obj)
        )

    def verify_acceptable_types(obj: Any) -> None:
        # subclass of them can not be fromInternal in JVM
        if type(obj) not in _acceptable_types[_type]:
            if name is not None:
                raise PySparkTypeError(
                    errorClass="FIELD_DATA_TYPE_UNACCEPTABLE_WITH_NAME",
                    messageParameters={
                        "field_name": str(name),
                        "data_type": str(dataType),
                        "obj": repr(obj),
                        "obj_type": str(type(obj)),
                    },
                )
            raise PySparkTypeError(
                errorClass="FIELD_DATA_TYPE_UNACCEPTABLE",
                messageParameters={
                    "data_type": str(dataType),
                    "obj": repr(obj),
                    "obj_type": str(type(obj)),
                },
            )

    if isinstance(dataType, (StringType, CharType, VarcharType)):
        # StringType, CharType and VarcharType can work with any types
        def verify_value(obj: Any) -> None:
            pass

    elif isinstance(dataType, UserDefinedType):
        verifier = _make_type_verifier(dataType.sqlType(), name=name)

        def verify_udf(obj: Any) -> None:
            if not (hasattr(obj, "__UDT__") and obj.__UDT__ == dataType):
                if name is not None:
                    raise PySparkValueError(
                        errorClass="FIELD_TYPE_MISMATCH_WITH_NAME",
                        messageParameters={
                            "field_name": str(name),
                            "obj": str(obj),
                            "data_type": str(dataType),
                        },
                    )
                raise PySparkValueError(
                    errorClass="FIELD_TYPE_MISMATCH",
                    messageParameters={
                        "obj": str(obj),
                        "data_type": str(dataType),
                    },
                )
            verifier(dataType.toInternal(obj))

        verify_value = verify_udf

    elif isinstance(dataType, ByteType):

        def verify_byte(obj: Any) -> None:
            assert_acceptable_types(obj)
            verify_acceptable_types(obj)
            lower_bound = -128
            upper_bound = 127
            if obj < lower_bound or obj > upper_bound:
                raise PySparkValueError(
                    errorClass="VALUE_OUT_OF_BOUNDS",
                    messageParameters={
                        "arg_name": "obj",
                        "lower_bound": str(lower_bound),
                        "upper_bound": str(upper_bound),
                        "actual": str(obj),
                    },
                )

        verify_value = verify_byte

    elif isinstance(dataType, ShortType):

        def verify_short(obj: Any) -> None:
            assert_acceptable_types(obj)
            verify_acceptable_types(obj)
            lower_bound = -32768
            upper_bound = 32767
            if obj < lower_bound or obj > upper_bound:
                raise PySparkValueError(
                    errorClass="VALUE_OUT_OF_BOUNDS",
                    messageParameters={
                        "arg_name": "obj",
                        "lower_bound": str(lower_bound),
                        "upper_bound": str(upper_bound),
                        "actual": str(obj),
                    },
                )

        verify_value = verify_short

    elif isinstance(dataType, IntegerType):

        def verify_integer(obj: Any) -> None:
            assert_acceptable_types(obj)
            verify_acceptable_types(obj)
            lower_bound = -2147483648
            upper_bound = 2147483647
            if obj < lower_bound or obj > upper_bound:
                raise PySparkValueError(
                    errorClass="VALUE_OUT_OF_BOUNDS",
                    messageParameters={
                        "arg_name": "obj",
                        "lower_bound": str(lower_bound),
                        "upper_bound": str(upper_bound),
                        "actual": str(obj),
                    },
                )

        verify_value = verify_integer

    elif isinstance(dataType, LongType):

        def verify_long(obj: Any) -> None:
            assert_acceptable_types(obj)
            verify_acceptable_types(obj)
            lower_bound = -9223372036854775808
            upper_bound = 9223372036854775807
            if obj < lower_bound or obj > upper_bound:
                raise PySparkValueError(
                    errorClass="VALUE_OUT_OF_BOUNDS",
                    messageParameters={
                        "arg_name": "obj",
                        "lower_bound": str(lower_bound),
                        "upper_bound": str(upper_bound),
                        "actual": str(obj),
                    },
                )

        verify_value = verify_long

    elif isinstance(dataType, ArrayType):
        element_verifier = _make_type_verifier(
            dataType.elementType, dataType.containsNull, name="element in array %s" % name
        )

        def verify_array(obj: Any) -> None:
            assert_acceptable_types(obj)
            verify_acceptable_types(obj)
            for i in obj:
                element_verifier(i)

        verify_value = verify_array

    elif isinstance(dataType, MapType):
        key_verifier = _make_type_verifier(dataType.keyType, False, name="key of map %s" % name)
        value_verifier = _make_type_verifier(
            dataType.valueType, dataType.valueContainsNull, name="value of map %s" % name
        )

        def verify_map(obj: Any) -> None:
            assert_acceptable_types(obj)
            verify_acceptable_types(obj)
            for k, v in obj.items():
                key_verifier(k)
                value_verifier(v)

        verify_value = verify_map

    elif isinstance(dataType, StructType):
        verifiers = []
        for f in dataType.fields:
            verifier = _make_type_verifier(f.dataType, f.nullable, name=new_name(f.name))
            verifiers.append((f.name, verifier))

        def verify_struct(obj: Any) -> None:
            assert_acceptable_types(obj)

            if isinstance(obj, dict):
                for f, verifier in verifiers:
                    verifier(obj.get(f))
            elif isinstance(obj, (tuple, list, Row)):
                if len(obj) != len(verifiers):
                    if name is not None:
                        raise PySparkValueError(
                            errorClass="FIELD_STRUCT_LENGTH_MISMATCH_WITH_NAME",
                            messageParameters={
                                "field_name": str(name),
                                "object_length": str(len(obj)),
                                "field_length": str(len(verifiers)),
                            },
                        )
                    raise PySparkValueError(
                        errorClass="FIELD_STRUCT_LENGTH_MISMATCH",
                        messageParameters={
                            "object_length": str(len(obj)),
                            "field_length": str(len(verifiers)),
                        },
                    )
                for v, (_, verifier) in zip(obj, verifiers):
                    verifier(v)
            elif hasattr(obj, "__dict__"):
                d = obj.__dict__
                for f, verifier in verifiers:
                    verifier(d.get(f))
            else:
                if name is not None:
                    raise PySparkTypeError(
                        errorClass="FIELD_DATA_TYPE_UNACCEPTABLE_WITH_NAME",
                        messageParameters={
                            "field_name": str(name),
                            "data_type": str(dataType),
                            "obj": repr(obj),
                            "obj_type": str(type(obj)),
                        },
                    )
                raise PySparkTypeError(
                    errorClass="FIELD_DATA_TYPE_UNACCEPTABLE",
                    messageParameters={
                        "data_type": str(dataType),
                        "obj": repr(obj),
                        "obj_type": str(type(obj)),
                    },
                )

        verify_value = verify_struct

    elif isinstance(dataType, VariantType):

        def verify_variant(obj: Any) -> None:
            # The variant data type can take in any type.
            pass

        verify_value = verify_variant

    elif isinstance(dataType, GeometryType):

        def verify_geometry(obj: Any) -> None:
            assert_acceptable_types(obj)
            verify_acceptable_types(obj)
            assert isinstance(obj, Geometry)

        verify_value = verify_geometry

    elif isinstance(dataType, GeographyType):

        def verify_geography(obj: Any) -> None:
            assert_acceptable_types(obj)
            verify_acceptable_types(obj)
            assert isinstance(obj, Geography)

        verify_value = verify_geography

    else:

        def verify_default(obj: Any) -> None:
            assert_acceptable_types(obj)
            verify_acceptable_types(obj)

        verify_value = verify_default

    def verify(obj: Any) -> None:
        if not verify_nullability(obj):
            verify_value(obj)

    return verify


# This is used to unpickle a Row from JVM
def _create_row_inbound_converter(dataType: DataType) -> Callable:
    return lambda *a: dataType.fromInternal(a)


