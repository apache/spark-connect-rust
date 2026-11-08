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

"""
User-defined table function (UDTF) support for the Spark Connect client.
Mirrors pyspark.sql.connect.udtf and pyspark.sql.functions.udtf.
"""

from pyspark._pyspark import SkipRestOfInputTableException

import inspect
import pickle
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Type

from pyspark.errors import PySparkAttributeError, PySparkPicklingError, PySparkTypeError
from pyspark.serializers import CloudPickleSerializer
from pyspark.sql.types import DataType, StructType

__all__ = [
    "AnalyzeArgument",
    "AnalyzeResult",
    "PartitioningColumn",
    "OrderingColumn",
    "SelectedColumn",
    "SkipRestOfInputTableException",
    "UDTFRegistration",
]

# Eval types for Python UDTFs (mirrors pyspark.util.PythonEvalType).
SQL_TABLE_UDF = 300
SQL_ARROW_TABLE_UDF = 301
SQL_ARROW_UDTF = 302


# Verbatim from PySpark (pyspark.sql.udtf): the frozen dataclasses of the analyze method of a
# UDTF, which a handler that the client pickles may capture, and which the Python worker of
# the server unpickles as PySpark's of the same names.
@dataclass(frozen=True)
class AnalyzeArgument:
    """
    The argument for Python UDTF's analyze static method.

    Parameters
    ----------
    dataType : :class:`DataType`
        The argument's data type
    value : any, optional
        The calculated value if the argument is foldable; otherwise None
    isTable : bool
        If True, the argument is a table argument.
    isConstantExpression : bool
        If True, the argument is a constant-foldable scalar expression. Then the 'value' field
        contains None if the argument is a NULL literal, or a non-None value if the argument is a
        non-NULL literal. In this way, we can distinguish between a literal NULL argument and other
        types of arguments such as complex expression trees or table arguments where the 'value'
        field is always None.
    """

    dataType: DataType
    value: Optional[Any]
    isTable: bool
    isConstantExpression: bool


@dataclass(frozen=True)
class PartitioningColumn:
    """
    Represents an expression that the UDTF is specifying for Catalyst to partition the input table
    by. This can be either the name of a single column from the input table (such as "columnA"), or
    a SQL expression based on the column names of the input table (such as "columnA + columnB").

    Parameters
    ----------
    name : str
        The contents of the partitioning column name or expression represented as a SQL string.
    """

    name: str


@dataclass(frozen=True)
class OrderingColumn:
    """
    Represents an expression that the UDTF is specifying for Catalyst to order the input partition
    by. This can be either the name of a single column from the input table (such as "columnA"),
    or a SQL expression based on the column names of the input table (such as "columnA + columnB").

    Parameters
    ----------
    name : str
        The contents of the ordering column name or expression represented as a SQL string.
    ascending : bool, default True
        This is if this expression specifies an ascending sorting order.
    overrideNullsFirst : str, optional
        If this is None, use the default behavior to sort NULL values first when sorting in
        ascending order, or last when sorting in descending order. Otherwise, if this is
        True or False, we override the default behavior accordingly.
    """

    name: str
    ascending: bool = True
    overrideNullsFirst: Optional[bool] = None


@dataclass(frozen=True)
class SelectedColumn:
    """
    Represents an expression that the UDTF is specifying for Catalyst to evaluate against the
    columns in the input TABLE argument. The UDTF then receives one input column for each expression
    in the list, in the order they are listed.

    Parameters
    ----------
    name : str
        The contents of the selected column name or expression represented as a SQL string.
    alias : str, default ''
        If non-empty, this is the alias for the column or expression as visible from the UDTF's
        'eval' method. This is required if the expression is not a simple column reference.
    """

    name: str
    alias: str = ""


@dataclass
class AnalyzeResult:
    """The result of a Python UDTF's ``analyze`` static method.

    As in PySpark, this is a mutable dataclass so a handler can subclass it to
    return custom analysis state, which is passed to the handler's constructor.
    """

    schema: StructType
    withSinglePartition: bool = False
    partitionBy: Sequence[PartitioningColumn] = field(default_factory=tuple)
    orderBy: Sequence[OrderingColumn] = field(default_factory=tuple)
    select: Sequence[SelectedColumn] = field(default_factory=tuple)


# Verbatim from PySpark (pyspark.sql.udtf): validates the handler class of a UDTF.
def _validate_udtf_handler(cls: Any, returnType: Optional[Any]) -> None:
    """Validate the handler class of a UDTF."""

    if not isinstance(cls, type):
        raise PySparkTypeError(
            errorClass="INVALID_UDTF_HANDLER_TYPE", messageParameters={"type": type(cls).__name__}
        )

    if not hasattr(cls, "eval"):
        raise PySparkAttributeError(
            errorClass="INVALID_UDTF_NO_EVAL", messageParameters={"name": cls.__name__}
        )

    has_analyze = hasattr(cls, "analyze")
    has_analyze_staticmethod = has_analyze and isinstance(
        inspect.getattr_static(cls, "analyze"), staticmethod
    )
    if returnType is None and not has_analyze_staticmethod:
        raise PySparkAttributeError(
            errorClass="INVALID_UDTF_RETURN_TYPE", messageParameters={"name": cls.__name__}
        )
    if returnType is not None and has_analyze:
        raise PySparkAttributeError(
            errorClass="INVALID_UDTF_BOTH_RETURN_TYPE_AND_ANALYZE",
            messageParameters={"name": cls.__name__},
        )


# Verbatim from PySpark Connect (pyspark.sql.connect.udtf): validates the handler class of a
# PyArrow UDTF.
def _validate_arrow_udtf_handler(cls: Any, returnType: Optional[Any]) -> None:
    """Validate the handler class of a PyArrow UDTF."""
    # First run standard UDTF validation
    _validate_udtf_handler(cls, returnType)

    # Block analyze method usage in arrow UDTFs
    # TODO(SPARK-53286): Support analyze method for Arrow UDTFs to enable dynamic return types
    has_analyze = hasattr(cls, "analyze")
    if has_analyze:
        raise PySparkAttributeError(
            errorClass="INVALID_ARROW_UDTF_WITH_ANALYZE",
            messageParameters={"name": cls.__name__},
        )


class UserDefinedTableFunction:
    """A user-defined table function: a Python handler class that yields rows.

    Calling it with Column arguments returns a DataFrame — the
    ``CommonInlineUserDefinedTableFunction`` relation. Mirrors
    ``pyspark.sql.connect.udtf.UserDefinedTableFunction``.
    """

    def __init__(
        self,
        func: Type,
        returnType: Optional[Any] = None,
        name: Optional[str] = None,
        evalType: int = SQL_TABLE_UDF,
        deterministic: bool = False,
    ):
        # As PySpark's UserDefinedTableFunction, for udtf(), arrow_udtf() and register().
        _validate_udtf_handler(func, returnType)
        self.func = func
        # As PySpark Connect's: a DDL string is a type that the server parses.
        if isinstance(returnType, str):
            from pyspark.sql.connect.types import UnparsedDataType

            returnType = UnparsedDataType(returnType)
        self.returnType = returnType
        self.evalType = evalType
        self.name = name or (
            func.__name__ if hasattr(func, "__name__") else "udtf"
        )
        self.deterministic = deterministic
        self.python_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    @property
    def command(self) -> bytes:
        """The pickled handler class, as PySpark Connect's ``PythonUDTF.command``: pickled when
        the UDTF is used, and ``UDTF_SERIALIZATION_ERROR`` of a handler that does not pickle.
        Connect sends the schema separately, in ``PythonUDTF.return_type``, so the command is
        only the handler class, unlike that of a scalar UDF."""
        try:
            return CloudPickleSerializer().dumps(self.func)
        except pickle.PicklingError:
            raise PySparkPicklingError(
                errorClass="UDTF_SERIALIZATION_ERROR",
                messageParameters={
                    "name": self.name,
                    "message": "Please check the stack trace and "
                    "make sure the function is serializable.",
                },
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from pyspark import _pyspark
        from pyspark.sql import SparkSession

        # As PySpark's: an argument is ColumnOrName, so a bare string is a column name, of
        # which pyfunc_make_udtf makes a ColumnReference, and each keyword argument is a
        # NamedArgumentExpression, after the positional ones.
        cols = list(args)
        named = list(kwargs.items())
        return _pyspark.functions.pyfunc_make_udtf(
            SparkSession.active(),
            self.name,
            self.returnType,
            self.evalType,
            self.command,
            self.python_ver,
            *cols,
            deterministic=self.deterministic,
            named_args=named,
        )

    def asDeterministic(self) -> "UserDefinedTableFunction":
        """Mark this UDTF as deterministic, and return it. Mirrors
        ``UserDefinedTableFunction.asDeterministic``."""
        self.deterministic = True
        return self


def udtf(
    cls: Optional[Type] = None,
    *,
    returnType: Optional[Any] = None,
    useArrow: Optional[bool] = None,
) -> Any:
    """Create a user-defined table function (UDTF).

    Use as ``@udtf(returnType=...)`` on a handler class, or call
    ``udtf(MyClass, returnType=...)`` directly. As PySpark's ``_create_py_udtf``, the UDTF is
    Arrow-optimized when ``useArrow`` is True, or is not given and
    ``spark.sql.execution.pythonUDTF.arrow.enabled`` of the active session is true.
    """

    def _udtf_decorator(handler: Type) -> UserDefinedTableFunction:
        if useArrow is not None:
            arrow_enabled = useArrow
        else:
            from pyspark.errors import PySparkRuntimeError
            from pyspark.sql import SparkSession

            arrow_enabled = False
            try:
                session = SparkSession.active()
                arrow_enabled = (
                    str(session.conf.get("spark.sql.execution.pythonUDTF.arrow.enabled")).lower()
                    == "true"
                )
            except PySparkRuntimeError as e:
                if e.getCondition() == "NO_ACTIVE_OR_DEFAULT_SESSION":
                    pass  # Just uses the default if no session found.
                else:
                    raise e
        eval_type = SQL_TABLE_UDF
        if arrow_enabled:
            try:
                from pyspark.sql.pandas.utils import (
                    require_minimum_pandas_version,
                    require_minimum_pyarrow_version,
                )

                require_minimum_pandas_version()
                require_minimum_pyarrow_version()
                eval_type = SQL_ARROW_TABLE_UDF
            except ImportError as e:
                import warnings

                warnings.warn(
                    f"Arrow optimization for Python UDTFs cannot be enabled: {str(e)}. "
                    f"Falling back to using regular Python UDTFs.",
                    UserWarning,
                )
        return UserDefinedTableFunction(handler, returnType=returnType, evalType=eval_type)

    if cls is not None:
        return _udtf_decorator(cls)
    return _udtf_decorator


def arrow_udtf(
    cls: Optional[Type] = None,
    *,
    returnType: Optional[Any] = None,
) -> Any:
    """Create an Arrow-based user-defined table function (UDTF).

    Mirrors ``pyspark.sql.functions.arrow_udtf``: a PyArrow-native UDTF (``SQL_ARROW_UDTF``),
    whose ``eval`` takes ``pyarrow`` arrays and returns ``pyarrow`` tables.
    """
    def _arrow_udtf_decorator(handler: Type) -> UserDefinedTableFunction:
        from pyspark.sql.pandas.utils import require_minimum_pyarrow_version

        require_minimum_pyarrow_version()
        _validate_arrow_udtf_handler(handler, returnType)
        return UserDefinedTableFunction(handler, returnType=returnType, evalType=SQL_ARROW_UDTF)

    if cls is not None:
        return _arrow_udtf_decorator(cls)
    return _arrow_udtf_decorator


class UDTFRegistration:
    """
    Wrapper for user-defined table function registration, as ``spark.udtf``. Mirrors
    ``pyspark.sql.connect.udtf.UDTFRegistration``.

    .. versionadded:: 3.5.0
    """

    def __init__(self, sparkSession: Any):
        self.sparkSession = sparkSession

    def register(
        self,
        name: str,
        f: "UserDefinedTableFunction",
    ) -> "UserDefinedTableFunction":
        """Register the UDTF ``f`` as ``name`` on the server, so that SQL calls it, and return
        ``f``. As PySpark's, ``f`` is a UDTF of ``udtf()`` or ``arrow_udtf()``."""
        if not isinstance(f, UserDefinedTableFunction):
            raise PySparkTypeError(
                errorClass="CANNOT_REGISTER_UDTF",
                messageParameters={
                    "name": name,
                },
            )

        if f.evalType not in [SQL_TABLE_UDF, SQL_ARROW_TABLE_UDF, SQL_ARROW_UDTF]:
            raise PySparkTypeError(
                errorClass="INVALID_UDTF_EVAL_TYPE",
                messageParameters={
                    "name": name,
                    "eval_type": "SQL_TABLE_UDF, SQL_ARROW_TABLE_UDF, SQL_ARROW_UDTF",
                },
            )

        self.sparkSession._registerPythonUdtf(
            name, f.returnType, f.evalType, f.command, f.python_ver, f.deterministic
        )
        return f
