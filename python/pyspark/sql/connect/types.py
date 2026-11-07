#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""The types of ``pyspark.sql.connect.types`` that are not those of ``pyspark.sql.types``, as
PySpark Connect's: ``UnparsedDataType``, a DDL string of a type that the server parses. The
conversions between the types and their ``spark.connect.DataType`` messages,
``pyspark_types_to_proto_types`` and ``proto_schema_to_pyspark_data_type``, are those of the Rust
transport, of the messages of the generated ``pyspark.sql.connect.proto``, which needs the
``protobuf`` package. ``parse_attr_name``, ``verify_col_name`` and ``verify_numeric_col_name`` are
PySpark Connect's, from Apache Spark v4.3.0."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pyspark.errors import PySparkAssertionError
from pyspark.sql.types import ArrayType, DataType, NumericType, StructType

if TYPE_CHECKING:
    import pyspark.sql.connect.proto as pb2


class UnparsedDataType(DataType):
    """
    Unparsed data type.

    The data type string will be parsed later.

    Parameters
    ----------
    data_type_string : str
        The data type string format equals :class:`DataType.simpleString`,
        except that the top level struct type can omit the ``struct<>``.
        This also supports a schema in a DDL-formatted string and case-insensitive strings.

    Examples
    --------
    >>> from pyspark.sql.connect.types import UnparsedDataType

    >>> UnparsedDataType("int ")
    UnparsedDataType('int ')
    >>> UnparsedDataType("INT ")
    UnparsedDataType('INT ')
    >>> UnparsedDataType("a: byte, b: decimal(  16 , 8   ) ")
    UnparsedDataType('a: byte, b: decimal(  16 , 8   ) ')
    >>> UnparsedDataType("a DOUBLE, b STRING")
    UnparsedDataType('a DOUBLE, b STRING')
    >>> UnparsedDataType("a DOUBLE, b CHAR( 50 )")
    UnparsedDataType('a DOUBLE, b CHAR( 50 )')
    >>> UnparsedDataType("a DOUBLE, b VARCHAR( 50 )")
    UnparsedDataType('a DOUBLE, b VARCHAR( 50 )')
    >>> UnparsedDataType("a: array< short>")
    UnparsedDataType('a: array< short>')
    >>> UnparsedDataType(" map<string , string > ")
    UnparsedDataType(' map<string , string > ')
    """

    def __init__(self, data_type_string: str):
        self.data_type_string = data_type_string

    def simpleString(self) -> str:
        return "unparsed(%s)" % repr(self.data_type_string)

    def __repr__(self) -> str:
        return "UnparsedDataType(%s)" % repr(self.data_type_string)

    def jsonValue(self) -> Dict[str, Any]:
        raise PySparkAssertionError(
            errorClass="INVALID_CALL_ON_UNRESOLVED_OBJECT",
            messageParameters={"func_name": "jsonValue"},
        )

    def needConversion(self) -> bool:
        raise PySparkAssertionError(
            errorClass="INVALID_CALL_ON_UNRESOLVED_OBJECT",
            messageParameters={"func_name": "needConversion"},
        )

    def toInternal(self, obj: Any) -> Any:
        raise PySparkAssertionError(
            errorClass="INVALID_CALL_ON_UNRESOLVED_OBJECT",
            messageParameters={"func_name": "toInternal"},
        )

    def fromInternal(self, obj: Any) -> Any:
        raise PySparkAssertionError(
            errorClass="INVALID_CALL_ON_UNRESOLVED_OBJECT",
            messageParameters={"func_name": "fromInternal"},
        )

    def __reduce__(self) -> Any:
        # The DataType of the drop-in pickles through its JSON, which an unparsed type has not.
        return (UnparsedDataType, (self.data_type_string,))


def pyspark_types_to_proto_types(data_type: DataType) -> "pb2.DataType":
    """The ``spark.connect.DataType`` of ``data_type``, as the client sends the type."""
    import pyspark.sql.connect.proto as pb2
    from pyspark._pyspark import _data_type_proto

    return pb2.DataType.FromString(_data_type_proto(data_type))


def proto_schema_to_pyspark_data_type(schema: "pb2.DataType") -> DataType:
    """The type of ``schema``, a ``spark.connect.DataType``, as the client reads the types of the
    server."""
    from pyspark._pyspark import _data_type_of_proto

    return _data_type_of_proto(schema.SerializeToString())


# The python version of org.apache.spark.sql.catalyst.util.AttributeNameParser
def parse_attr_name(name: str) -> Optional[List[str]]:
    name_parts: List[str] = []
    tmp: str = ""

    in_backtick = False
    i = 0
    n = len(name)
    while i < n:
        char = name[i]
        if in_backtick:
            if char == "`":
                if i + 1 < n and name[i + 1] == "`":
                    tmp += "`"
                    i += 1
                else:
                    in_backtick = False
                    if i + 1 < n and name[i + 1] != ".":
                        return None
            else:
                tmp += char
        else:
            if char == "`":
                if len(tmp) > 0:
                    return None
                in_backtick = True
            elif char == ".":
                if name[i - 1] == "." or i == n - 1:
                    return None
                name_parts.append(tmp)
                tmp = ""
            else:
                tmp += char
        i += 1

    if in_backtick:
        return None

    name_parts.append(tmp)
    return name_parts


# Verify whether the input column name can be resolved with the given schema.
# Note that this method can not 100% match the analyzer behavior, it is designed to
# try the best to eliminate unnecessary validation RPCs.
def verify_col_name(name: str, schema: StructType) -> bool:
    parts = parse_attr_name(name)
    if parts is None or len(parts) == 0:
        return False

    def _quick_verify(parts: List[str], dt: DataType) -> bool:
        if len(parts) == 0:
            return True

        _schema: Optional[StructType] = None
        if isinstance(dt, StructType):
            _schema = dt
        elif isinstance(dt, ArrayType) and isinstance(dt.elementType, StructType):
            _schema = dt.elementType
        else:
            return False

        part = parts[0]
        for field in _schema:
            if field.name == part:
                return _quick_verify(parts[1:], field.dataType)

        return False

    return _quick_verify(parts, schema)


def verify_numeric_col_name(name: str, schema: StructType) -> bool:
    parts = parse_attr_name(name)
    if parts is None or len(parts) == 0:
        return False

    def _quick_verify(parts: List[str], dt: DataType) -> bool:
        if len(parts) == 0 and isinstance(dt, NumericType):
            return True

        _schema: Optional[StructType] = None
        if isinstance(dt, StructType):
            _schema = dt
        elif isinstance(dt, ArrayType) and isinstance(dt.elementType, StructType):
            _schema = dt.elementType
        else:
            return False

        part = parts[0]
        for field in _schema:
            if field.name == part:
                return _quick_verify(parts[1:], field.dataType)

        return False

    return _quick_verify(parts, schema)


__all__ = ["UnparsedDataType"]
