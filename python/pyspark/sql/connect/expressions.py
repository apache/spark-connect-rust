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
"""Expressions under the connect import path.

The Rust core builds the expressions of the DataFrame API: the expression of a Column, its
``Column._expr``, is of the core. ``Expression`` and ``LiteralExpression`` are PySpark Connect's,
from Apache Spark v4.3.0, for the modules of PySpark that build their own messages, such as the
params of ``pyspark.ml`` (``pyspark.ml.connect.serialize``). The base has no ``alias``: the
``ColumnAlias`` it makes is an expression of PySpark Connect's Column, which this client does not
have. The messages are those of the generated ``pyspark.sql.connect.proto``, which needs the
``protobuf`` package.
"""

from typing import (
    cast,
    Any,
    Callable,
    Sequence,
    Optional,
)

import decimal
import datetime
from threading import Lock

import numpy as np

from pyspark.sql.types import (
    _create_row,
    _from_numpy_type,
    DateType,
    ArrayType,
    NullType,
    BooleanType,
    BinaryType,
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    DecimalType,
    StringType,
    DataType,
    TimeType,
    TimestampType,
    TimestampNTZType,
    DayTimeIntervalType,
    MapType,
    StructType,
)

import pyspark.sql.connect.proto as proto
from pyspark.util import (
    JVM_BYTE_MIN,
    JVM_BYTE_MAX,
    JVM_SHORT_MIN,
    JVM_SHORT_MAX,
    JVM_INT_MIN,
    JVM_INT_MAX,
    JVM_LONG_MIN,
    JVM_LONG_MAX,
)
from pyspark.sql.connect.types import (
    pyspark_types_to_proto_types,
    proto_schema_to_pyspark_data_type,
)
from pyspark.errors import PySparkTypeError, PySparkValueError
from pyspark.errors.utils import current_origin
from pyspark.sql.utils import is_timestamp_ntz_preferred, enum_to_value


class Expression:
    """
    Expression base class.
    """

    def __init__(self) -> None:
        origin = current_origin()
        fragment = origin.fragment
        call_site = origin.call_site
        self.origin = None
        if fragment is not None and call_site is not None:
            self.origin = proto.Origin(
                python_origin=proto.PythonOrigin(
                    fragment=origin.fragment, call_site=origin.call_site
                )
            )

    def to_plan(  # type: ignore[empty-body]
        self, session: Any
    ) -> "proto.Expression": ...

    def __repr__(self) -> str:  # type: ignore[empty-body]
        ...

    def name(self) -> str:  # type: ignore[empty-body]
        ...

    def _create_proto_expression(self) -> "proto.Expression":
        plan = proto.Expression()
        if self.origin is not None:
            plan.common.origin.CopyFrom(self.origin)
        return plan

    @property
    def children(self) -> Sequence["Expression"]:
        return []

    def foreach(self, f: Callable[["Expression"], None]) -> None:
        f(self)
        for c in self.children:
            c.foreach(f)


class LiteralExpression(Expression):
    """A literal expression.

    The Python types are converted best effort into the relevant proto types. On the Spark Connect
    server side, the proto types are converted to the Catalyst equivalents."""

    def __init__(self, value: Any, dataType: DataType) -> None:
        super().__init__()

        assert isinstance(
            dataType,
            (
                NullType,
                BinaryType,
                BooleanType,
                ByteType,
                ShortType,
                IntegerType,
                LongType,
                FloatType,
                DoubleType,
                DecimalType,
                StringType,
                DateType,
                TimeType,
                TimestampType,
                TimestampNTZType,
                DayTimeIntervalType,
                ArrayType,
            ),
        )

        value = enum_to_value(value)
        if isinstance(dataType, NullType):
            assert value is None

        if value is not None:
            if isinstance(dataType, BinaryType):
                assert isinstance(value, (bytes, bytearray))
            elif isinstance(dataType, BooleanType):
                assert isinstance(value, (bool, np.bool_))
                value = bool(value)
            elif isinstance(dataType, ByteType):
                assert isinstance(value, (int, np.int8))
                assert JVM_BYTE_MIN <= int(value) <= JVM_BYTE_MAX
                value = int(value)
            elif isinstance(dataType, ShortType):
                assert isinstance(value, (int, np.int8, np.int16))
                assert JVM_SHORT_MIN <= int(value) <= JVM_SHORT_MAX
                value = int(value)
            elif isinstance(dataType, IntegerType):
                assert isinstance(value, (int, np.int8, np.int16, np.int32))
                assert JVM_INT_MIN <= int(value) <= JVM_INT_MAX
                value = int(value)
            elif isinstance(dataType, LongType):
                assert isinstance(value, (int, np.int8, np.int16, np.int32, np.int64))
                assert JVM_LONG_MIN <= int(value) <= JVM_LONG_MAX
                value = int(value)
            elif isinstance(dataType, FloatType):
                assert isinstance(value, (float, np.float32))
                value = float(value)
            elif isinstance(dataType, DoubleType):
                assert isinstance(value, (float, np.float32, np.float64))
                value = float(value)
            elif isinstance(dataType, DecimalType):
                assert isinstance(value, decimal.Decimal)
            elif isinstance(dataType, StringType):
                assert isinstance(value, (str, np.str_))
                value = str(value)
            elif isinstance(dataType, DateType):
                assert isinstance(value, datetime.date)
                value = DateType().toInternal(value)
            elif isinstance(dataType, TimeType):
                assert isinstance(value, datetime.time)
                value = TimeType().toInternal(value)
            elif isinstance(dataType, TimestampType):
                assert isinstance(value, datetime.datetime)
                value = TimestampType().toInternal(value)
            elif isinstance(dataType, TimestampNTZType):
                assert isinstance(value, datetime.datetime)
                value = TimestampNTZType().toInternal(value)
            elif isinstance(dataType, DayTimeIntervalType):
                assert isinstance(value, datetime.timedelta)
                value = DayTimeIntervalType().toInternal(value)
                assert value is not None
            elif isinstance(dataType, ArrayType):
                assert isinstance(value, list)
            else:
                raise PySparkTypeError(
                    errorClass="UNSUPPORTED_DATA_TYPE",
                    messageParameters={"data_type": str(dataType)},
                )

        self._value = value
        self._dataType = dataType

    @classmethod
    def _infer_type(cls, value: Any) -> DataType:
        value = enum_to_value(value)
        if value is None:
            return NullType()
        elif isinstance(value, (bytes, bytearray)):
            return BinaryType()
        elif isinstance(value, (bool, np.bool_)):
            return BooleanType()
        elif isinstance(value, int):
            if JVM_INT_MIN <= value <= JVM_INT_MAX:
                return IntegerType()
            elif JVM_LONG_MIN <= value <= JVM_LONG_MAX:
                return LongType()
            else:
                raise PySparkValueError(
                    errorClass="VALUE_NOT_BETWEEN",
                    messageParameters={
                        "arg_name": "value",
                        "min": str(JVM_LONG_MIN),
                        "max": str(JVM_SHORT_MAX),
                    },
                )
        elif isinstance(value, float):
            return DoubleType()
        elif isinstance(value, (str, np.str_)):
            return StringType()
        elif isinstance(value, decimal.Decimal):
            return DecimalType()
        elif isinstance(value, datetime.datetime):
            return TimestampNTZType() if is_timestamp_ntz_preferred() else TimestampType()
        elif isinstance(value, datetime.date):
            return DateType()
        elif isinstance(value, datetime.time):
            return TimeType()
        elif isinstance(value, datetime.timedelta):
            return DayTimeIntervalType()
        elif isinstance(value, np.generic):
            dt = _from_numpy_type(value.dtype)
            if dt is not None:
                return dt
        elif isinstance(value, list):
            # follow the 'infer_array_from_first_element' strategy in 'sql.types._infer_type'
            # right now, it's dedicated for pyspark.ml params like array<...>, array<array<...>>
            if len(value) == 0 or value[0] is None:
                raise PySparkTypeError(
                    errorClass="CANNOT_INFER_ARRAY_ELEMENT_TYPE",
                    messageParameters={},
                )
            return ArrayType(LiteralExpression._infer_type(value[0]), True)

        raise PySparkTypeError(
            errorClass="UNSUPPORTED_DATA_TYPE",
            messageParameters={"data_type": type(value).__name__},
        )

    @classmethod
    def _from_value(cls, value: Any) -> "LiteralExpression":
        return LiteralExpression(value=value, dataType=LiteralExpression._infer_type(value))

    @classmethod
    def _to_value(
        cls, literal: "proto.Expression.Literal", dataType: Optional[DataType] = None
    ) -> Any:
        if literal.HasField("null"):
            return None
        elif literal.HasField("binary"):
            assert dataType is None or isinstance(dataType, BinaryType)
            return literal.binary
        elif literal.HasField("boolean"):
            assert dataType is None or isinstance(dataType, BooleanType)
            return literal.boolean
        elif literal.HasField("byte"):
            assert dataType is None or isinstance(dataType, ByteType)
            return literal.byte
        elif literal.HasField("short"):
            assert dataType is None or isinstance(dataType, ShortType)
            return literal.short
        elif literal.HasField("integer"):
            assert dataType is None or isinstance(dataType, IntegerType)
            return literal.integer
        elif literal.HasField("long"):
            assert dataType is None or isinstance(dataType, LongType)
            return literal.long
        elif literal.HasField("float"):
            assert dataType is None or isinstance(dataType, FloatType)
            return literal.float
        elif literal.HasField("double"):
            assert dataType is None or isinstance(dataType, DoubleType)
            return literal.double
        elif literal.HasField("decimal"):
            assert dataType is None or isinstance(dataType, DecimalType)
            return decimal.Decimal(literal.decimal.value)
        elif literal.HasField("string"):
            assert dataType is None or isinstance(dataType, StringType)
            return literal.string
        elif literal.HasField("date"):
            assert dataType is None or isinstance(dataType, DataType)
            return DateType().fromInternal(literal.date)
        elif literal.HasField("time"):
            assert dataType is None or isinstance(dataType, TimeType)
            return TimeType().fromInternal(literal.time.nano)
        elif literal.HasField("timestamp"):
            assert dataType is None or isinstance(dataType, TimestampType)
            return TimestampType().fromInternal(literal.timestamp)
        elif literal.HasField("timestamp_ntz"):
            assert dataType is None or isinstance(dataType, TimestampNTZType)
            return TimestampNTZType().fromInternal(literal.timestamp_ntz)
        elif literal.HasField("day_time_interval"):
            assert dataType is None or isinstance(dataType, DayTimeIntervalType)
            return DayTimeIntervalType().fromInternal(literal.day_time_interval)
        elif literal.HasField("array"):
            elementType = proto_schema_to_pyspark_data_type(literal.array.element_type)
            if dataType is not None:
                assert isinstance(dataType, ArrayType)
                assert elementType == dataType.elementType
            return [LiteralExpression._to_value(v, elementType) for v in literal.array.elements]
        elif literal.HasField("map"):
            keyType = proto_schema_to_pyspark_data_type(literal.map.key_type)
            valueType = proto_schema_to_pyspark_data_type(literal.map.value_type)
            if dataType is not None:
                assert isinstance(dataType, MapType)
                assert keyType == dataType.keyType
                assert valueType == dataType.valueType
            return {
                LiteralExpression._to_value(k, keyType): LiteralExpression._to_value(v, valueType)
                for k, v in zip(literal.map.keys, literal.map.values)
            }
        elif literal.HasField("struct"):
            struct_type = cast(
                StructType, proto_schema_to_pyspark_data_type(literal.struct.struct_type)
            )
            if dataType is not None:
                assert isinstance(dataType, StructType)
                assert struct_type == dataType
            values = [
                LiteralExpression._to_value(v, f.dataType)
                for v, f in zip(literal.struct.elements, struct_type.fields)
            ]
            return _create_row(struct_type.names, values)

        raise PySparkTypeError(
            errorClass="UNSUPPORTED_LITERAL",
            messageParameters={"literal": str(literal)},
        )

    def to_plan(self, session: Any) -> "proto.Expression":
        """Converts the literal expression to the literal in proto."""

        expr = self._create_proto_expression()

        if self._value is None:
            expr.literal.null.CopyFrom(pyspark_types_to_proto_types(self._dataType))
        elif isinstance(self._dataType, BinaryType):
            expr.literal.binary = bytes(self._value)
        elif isinstance(self._dataType, BooleanType):
            expr.literal.boolean = bool(self._value)
        elif isinstance(self._dataType, ByteType):
            expr.literal.byte = int(self._value)
        elif isinstance(self._dataType, ShortType):
            expr.literal.short = int(self._value)
        elif isinstance(self._dataType, IntegerType):
            expr.literal.integer = int(self._value)
        elif isinstance(self._dataType, LongType):
            expr.literal.long = int(self._value)
        elif isinstance(self._dataType, FloatType):
            expr.literal.float = float(self._value)
        elif isinstance(self._dataType, DoubleType):
            expr.literal.double = float(self._value)
        elif isinstance(self._dataType, DecimalType):
            expr.literal.decimal.value = str(self._value)
            expr.literal.decimal.precision = self._dataType.precision
            expr.literal.decimal.scale = self._dataType.scale
        elif isinstance(self._dataType, StringType):
            expr.literal.string = str(self._value)
        elif isinstance(self._dataType, DateType):
            expr.literal.date = int(self._value)
        elif isinstance(self._dataType, TimeType):
            expr.literal.time.precision = self._dataType.precision
            expr.literal.time.nano = int(self._value)
        elif isinstance(self._dataType, TimestampType):
            expr.literal.timestamp = int(self._value)
        elif isinstance(self._dataType, TimestampNTZType):
            expr.literal.timestamp_ntz = int(self._value)
        elif isinstance(self._dataType, DayTimeIntervalType):
            expr.literal.day_time_interval = int(self._value)
        elif isinstance(self._dataType, ArrayType):
            element_type = self._dataType.elementType
            expr.literal.array.element_type.CopyFrom(pyspark_types_to_proto_types(element_type))
            for v in self._value:
                expr.literal.array.elements.append(
                    LiteralExpression(v, element_type).to_plan(session).literal
                )
        else:
            raise PySparkTypeError(
                errorClass="UNSUPPORTED_DATA_TYPE",
                messageParameters={"data_type": str(self._dataType)},
            )

        return expr

    def __repr__(self) -> str:
        if self._value is None:
            return "NULL"
        elif isinstance(self._dataType, DateType):
            dt = DateType().fromInternal(self._value)
            if dt is not None and isinstance(dt, datetime.date):
                return dt.strftime("%Y-%m-%d")
        elif isinstance(self._dataType, TimeType):
            t = TimeType().fromInternal(self._value)
            if t is not None and isinstance(t, datetime.time):
                return t.strftime("%H:%M:%S.%f")
        elif isinstance(self._dataType, TimestampType):
            ts = TimestampType().fromInternal(self._value)
            if ts is not None and isinstance(ts, datetime.datetime):
                return ts.strftime("%Y-%m-%d %H:%M:%S.%f")
        elif isinstance(self._dataType, TimestampNTZType):
            ts = TimestampNTZType().fromInternal(self._value)
            if ts is not None and isinstance(ts, datetime.datetime):
                return ts.strftime("%Y-%m-%d %H:%M:%S.%f")
        elif isinstance(self._dataType, DayTimeIntervalType):
            delta = DayTimeIntervalType().fromInternal(self._value)
            if delta is not None and isinstance(delta, datetime.timedelta):
                import pandas as pd

                # Note: timedelta itself does not provide isoformat method.
                # Both Pandas and java.time.Duration provide it, but the format
                # is sightly different:
                # java.time.Duration only applies HOURS, MINUTES, SECONDS units,
                # while Pandas applies all supported units.
                return pd.Timedelta(delta).isoformat()
        return f"{self._value}"


class UnresolvedNamedLambdaVariable(Expression):
    """A variable of a lambda function, of the name parts of a column reference. The lambda
    functions of ``pyspark.sql.functions`` name their variables with :meth:`fresh_var_name`."""

    _lock: Lock = Lock()
    _nextVarNameId: int = 0

    def __init__(
        self,
        name_parts: Sequence[str],
    ) -> None:
        super().__init__()

        assert (
            isinstance(name_parts, list)
            and len(name_parts) > 0
            and all(isinstance(p, str) for p in name_parts)
        )

        self._name_parts = name_parts

    def to_plan(self, session: Any) -> "proto.Expression":
        expr = self._create_proto_expression()
        expr.unresolved_named_lambda_variable.name_parts.extend(self._name_parts)
        return expr

    def __repr__(self) -> str:
        return ", ".join(self._name_parts)

    @staticmethod
    def fresh_var_name(name: str) -> str:
        assert isinstance(name, str) and name != ""

        with UnresolvedNamedLambdaVariable._lock:
            _id = UnresolvedNamedLambdaVariable._nextVarNameId
            UnresolvedNamedLambdaVariable._nextVarNameId += 1

        return f"{name}_{_id}"


class DistributedSequenceID(Expression):
    """The ``distributed_sequence_id()`` function, which the server evaluates."""

    def to_plan(self, session: Any) -> "proto.Expression":
        # As PySpark's UnresolvedFunction of no argument, which is not distinct.
        expr = self._create_proto_expression()
        expr.unresolved_function.function_name = "distributed_sequence_id"
        expr.unresolved_function.is_distinct = False
        return expr

    def __repr__(self) -> str:
        return "DistributedSequenceID()"
