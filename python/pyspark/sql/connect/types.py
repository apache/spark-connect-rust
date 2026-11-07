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
conversions between the types and their protobuf messages of PySpark Connect's module are those
of the Rust transport."""

from typing import Any, Dict

from pyspark.errors import PySparkAssertionError
from pyspark.sql.types import DataType


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


__all__ = ["UnparsedDataType"]
