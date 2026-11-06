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
User-defined function (UDF) support for Spark Connect client.
Mirrors pyspark.sql.connect.udf and pyspark.sql.functions.udf/pandas_udf.
"""

import sys
from typing import Callable, Optional, Any

from pyspark.errors import PySparkNotImplementedError, PySparkTypeError
from pyspark.sql.pandas.types import to_arrow_type
from pyspark.sql.types import DataType, StringType, StructType
from pyspark.serializers import CloudPickleSerializer
from pyspark.util import PythonEvalType


def _as_concrete_datatype(returnType: Any) -> DataType:
    """Return a concrete, picklable pyspark ``DataType``.

    A DDL string (e.g. ``"double"``, ``"a int, b string"``) is parsed into its
    concrete type with ``DataType.fromDDL``. This matters because the cloudpickled UDF
    command must carry a real ``DataType`` (not a DDL string): the pandas/Arrow
    worker pairs each result column with its spark type to build the Arrow batch,
    so a bare string breaks it ("not enough values to unpack"). Mirrors how the
    reference Connect client pickles the *parsed* output type.
    """
    if isinstance(returnType, str):
        return DataType.fromDDL(returnType)
    return returnType


class UserDefinedFunction:
    """
    Represents a user-defined function (UDF) that can be called with columns.
    Stores the function, return type, eval type, and pickled command.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        returnType: DataType,
        evalType: int = 100,  # SQL_BATCHED_UDF
        name: Optional[str] = None,
        deterministic: bool = True,
    ):
        # The checks of PySpark Connect's UserDefinedFunction.
        if not callable(func):
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "expected_type": "callable",
                    "arg_name": "func",
                    "arg_type": type(func).__name__,
                },
            )

        if not isinstance(returnType, (DataType, str)):
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "expected_type": "DataType or str",
                    "arg_name": "returnType",
                    "arg_type": type(returnType).__name__,
                },
            )

        if not isinstance(evalType, int):
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "expected_type": "int",
                    "arg_name": "evalType",
                    "arg_type": type(evalType).__name__,
                },
            )

        self.func = func
        # Normalize to a concrete, picklable DataType so the cloudpickled command
        # carries a real type (required by the pandas/Arrow worker, not just str).
        self.returnType = _as_concrete_datatype(returnType)
        # As PySpark Connect's UDF checks its return type for its eval type when it is made.
        UserDefinedFunction._check_return_type(self.returnType, evalType)
        self.evalType = evalType
        self.name = name or (
            func.__name__ if hasattr(func, "__name__") else "udf"
        )
        self.deterministic = deterministic
        self.python_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

        # Cloudpickle the function (the command is the pickled (func, returnType) tuple).
        # Use the normalized concrete DataType, not the raw arg, so the worker gets a
        # real type it can turn into an Arrow schema.
        serializer = CloudPickleSerializer()
        self.command = serializer.dumps((func, self.returnType))

    # Verbatim from PySpark (pyspark.sql.udf.UserDefinedFunction).
    @staticmethod
    def _check_return_type(returnType: DataType, evalType: int) -> None:
        if evalType == PythonEvalType.SQL_ARROW_BATCHED_UDF:
            try:
                to_arrow_type(returnType, timezone="UTC")
            except TypeError:
                raise PySparkNotImplementedError(
                    errorClass="NOT_IMPLEMENTED",
                    messageParameters={
                        "feature": f"Invalid return type with Arrow-optimized Python UDF: "
                        f"{returnType}"
                    },
                )
        elif (
            evalType == PythonEvalType.SQL_SCALAR_PANDAS_UDF
            or evalType == PythonEvalType.SQL_SCALAR_PANDAS_ITER_UDF
        ):
            try:
                to_arrow_type(returnType, timezone="UTC")
            except TypeError:
                raise PySparkNotImplementedError(
                    errorClass="NOT_IMPLEMENTED",
                    messageParameters={
                        "feature": f"Invalid return type with scalar Pandas UDFs: {returnType}"
                    },
                )
        elif (
            evalType == PythonEvalType.SQL_SCALAR_ARROW_UDF
            or evalType == PythonEvalType.SQL_SCALAR_ARROW_ITER_UDF
        ):
            try:
                to_arrow_type(returnType, timezone="UTC")
            except TypeError:
                raise PySparkNotImplementedError(
                    errorClass="NOT_IMPLEMENTED",
                    messageParameters={
                        "feature": f"Invalid return type with scalar Arrow UDFs: {returnType}"
                    },
                )
        elif (
            evalType == PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF
            or evalType == PythonEvalType.SQL_GROUPED_MAP_PANDAS_ITER_UDF
            or evalType == PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE
        ):
            if isinstance(returnType, StructType):
                try:
                    to_arrow_type(returnType, timezone="UTC")
                except TypeError:
                    raise PySparkNotImplementedError(
                        errorClass="NOT_IMPLEMENTED",
                        messageParameters={
                            "feature": f"Invalid return type with grouped map Pandas UDFs or "
                            f"at groupby.applyInPandas(WithState): {returnType}"
                        },
                    )
            else:
                raise PySparkTypeError(
                    errorClass="INVALID_RETURN_TYPE_FOR_PANDAS_UDF",
                    messageParameters={
                        "eval_type": "SQL_GROUPED_MAP_PANDAS_UDF or "
                        "SQL_GROUPED_MAP_PANDAS_ITER_UDF or "
                        "SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE",
                        "return_type": str(returnType),
                    },
                )
        elif (
            evalType == PythonEvalType.SQL_MAP_PANDAS_ITER_UDF
            or evalType == PythonEvalType.SQL_MAP_ARROW_ITER_UDF
        ):
            if isinstance(returnType, StructType):
                try:
                    to_arrow_type(returnType, timezone="UTC")
                except TypeError:
                    raise PySparkNotImplementedError(
                        errorClass="NOT_IMPLEMENTED",
                        messageParameters={
                            "feature": f"Invalid return type in mapInPandas: {returnType}"
                        },
                    )
            else:
                raise PySparkTypeError(
                    errorClass="INVALID_RETURN_TYPE_FOR_PANDAS_UDF",
                    messageParameters={
                        "eval_type": "SQL_MAP_PANDAS_ITER_UDF or SQL_MAP_ARROW_ITER_UDF",
                        "return_type": str(returnType),
                    },
                )
        elif (
            evalType == PythonEvalType.SQL_GROUPED_MAP_ARROW_UDF
            or evalType == PythonEvalType.SQL_GROUPED_MAP_ARROW_ITER_UDF
        ):
            if isinstance(returnType, StructType):
                try:
                    to_arrow_type(returnType, timezone="UTC")
                except TypeError:
                    raise PySparkNotImplementedError(
                        errorClass="NOT_IMPLEMENTED",
                        messageParameters={
                            "feature": "Invalid return type with grouped map Arrow UDFs or "
                            f"at groupby.applyInArrow: {returnType}"
                        },
                    )
            else:
                raise PySparkTypeError(
                    errorClass="INVALID_RETURN_TYPE_FOR_ARROW_UDF",
                    messageParameters={
                        "eval_type": "SQL_GROUPED_MAP_ARROW_UDF or SQL_GROUPED_MAP_ARROW_ITER_UDF",
                        "return_type": str(returnType),
                    },
                )
        elif evalType == PythonEvalType.SQL_COGROUPED_MAP_PANDAS_UDF:
            if isinstance(returnType, StructType):
                try:
                    to_arrow_type(returnType, timezone="UTC")
                except TypeError:
                    raise PySparkNotImplementedError(
                        errorClass="NOT_IMPLEMENTED",
                        messageParameters={
                            "feature": f"Invalid return type in cogroup.applyInPandas: {returnType}"
                        },
                    )
            else:
                raise PySparkTypeError(
                    errorClass="INVALID_RETURN_TYPE_FOR_PANDAS_UDF",
                    messageParameters={
                        "eval_type": "SQL_COGROUPED_MAP_PANDAS_UDF",
                        "return_type": str(returnType),
                    },
                )
        elif evalType == PythonEvalType.SQL_COGROUPED_MAP_ARROW_UDF:
            if isinstance(returnType, StructType):
                try:
                    to_arrow_type(returnType, timezone="UTC")
                except TypeError:
                    raise PySparkNotImplementedError(
                        errorClass="NOT_IMPLEMENTED",
                        messageParameters={
                            "feature": f"Invalid return type in cogroup.applyInArrow: {returnType}"
                        },
                    )
            else:
                raise PySparkTypeError(
                    errorClass="INVALID_RETURN_TYPE_FOR_ARROW_UDF",
                    messageParameters={
                        "eval_type": "SQL_COGROUPED_MAP_ARROW_UDF",
                        "return_type": str(returnType),
                    },
                )
        elif evalType == PythonEvalType.SQL_GROUPED_AGG_PANDAS_UDF:
            try:
                # StructType is not yet allowed as a return type, explicitly check here to fail fast
                if isinstance(returnType, StructType):
                    raise PySparkNotImplementedError(
                        errorClass="NOT_IMPLEMENTED",
                        messageParameters={
                            "feature": f"Invalid return type with grouped aggregate Pandas UDFs: "
                            f"{returnType}"
                        },
                    )
                to_arrow_type(returnType, timezone="UTC")
            except TypeError:
                raise PySparkNotImplementedError(
                    errorClass="NOT_IMPLEMENTED",
                    messageParameters={
                        "feature": f"Invalid return type with grouped aggregate Pandas UDFs: "
                        f"{returnType}"
                    },
                )
        elif evalType == PythonEvalType.SQL_GROUPED_AGG_ARROW_UDF:
            try:
                # Different from SQL_GROUPED_AGG_PANDAS_UDF, StructType is allowed here
                to_arrow_type(returnType, timezone="UTC")
            except TypeError:
                raise PySparkNotImplementedError(
                    errorClass="NOT_IMPLEMENTED",
                    messageParameters={
                        "feature": f"Invalid return type with grouped aggregate Arrow UDFs: "
                        f"{returnType}"
                    },
                )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the UDF with Column arguments.
        Returns a Column representing the UDF call.
        """
        from pyspark import _pyspark
        from pyspark.sql.functions import col

        # A UDF argument is ColumnOrName: a bare string is a COLUMN NAME (like PySpark),
        # not a string literal. Coerce str -> col(str) before handing to the Rust layer
        # (whose to_column would otherwise treat a str as a literal).
        cols = [col(a) if isinstance(a, str) else a for a in args]

        # pyfunc_make_udf accepts any DataType object or DDL string as the return type
        # and builds the proto output type; the pickled command carries (func, returnType)
        # for the server-side Python worker.
        return _pyspark.functions.pyfunc_make_udf(
            self.name,
            self.returnType,
            self.evalType,
            self.command,
            self.python_ver,
            *cols,
            deterministic=self.deterministic,
        )

    def asNondeterministic(self) -> "UserDefinedFunction":
        """Return a copy of this UDF marked as non-deterministic. Mirrors
        ``UserDefinedFunction.asNondeterministic`` — the server will not fold/reuse
        results across rows."""
        return UserDefinedFunction(
            self.func, self.returnType, self.evalType, self.name, deterministic=False
        )


def udf(
    f: Optional[Callable[..., Any]] = None,
    returnType: Optional[DataType] = None,
    *,
    useArrow: Optional[bool] = None,
) -> Any:
    """
    Create a user-defined function (UDF).

    Parameters
    ----------
    f : callable, optional
        The Python function to wrap as a UDF.
    returnType : DataType, optional
        The return type of the UDF. Defaults to StringType().
    useArrow : bool, optional
        Whether to use Arrow optimization. Defaults to None (auto-detect).

    Returns
    -------
    UserDefinedFunction or callable
        If f is provided, returns a UDF. If f is None, returns a decorator.

    Examples
    --------
    >>> from pyspark.sql import functions as F
    >>> from pyspark.sql.types import IntegerType
    >>> u = F.udf(lambda x: x + 1, IntegerType())
    >>> result = spark.range(3).select(u(F.col('id')).alias('inc_id'))
    """
    # Decorator form @udf(returnType) / @udf("ddl"): the first positional is the
    # return type (a DataType or DDL string), not the function.
    if f is not None and not callable(f):
        returnType, f = f, None

    if returnType is None:
        returnType = StringType()

    evalType = 100  # SQL_BATCHED_UDF (default)
    if useArrow is True:
        evalType = 101  # SQL_ARROW_BATCHED_UDF

    def _udf_decorator(func):
        return UserDefinedFunction(func, returnType, evalType)

    if f is not None:
        # Direct call: @udf(f, returnType)
        return _udf_decorator(f)
    else:
        # Decorator call: @udf(...) or @udf
        return _udf_decorator


def _create_udf(
    f: Callable[..., Any],
    returnType: Any,
    evalType: int,
    name: Optional[str] = None,
    deterministic: bool = True,
) -> UserDefinedFunction:
    """The UDF of a function, its return type and eval type, as PySpark's ``_create_udf``
    makes it; ``pyspark.sql.pandas.functions`` creates pandas and Arrow UDFs with it."""
    return UserDefinedFunction(
        f, returnType=returnType, evalType=evalType, name=name, deterministic=deterministic
    )


class UDFRegistration:
    """
    Wrapper for user-defined function registration (spark.udf.register).
    """

    def __init__(self, spark_session: Any):
        self.spark_session = spark_session

    def register(
        self,
        name: str,
        f: Callable[..., Any],
        returnType: Optional[DataType] = None,
    ) -> UserDefinedFunction:
        """
        Register a Python UDF with the given name.

        Parameters
        ----------
        name : str
            Name to register the UDF with.
        f : callable
            The Python function to register.
        returnType : DataType, optional
            The return type of the UDF. Defaults to StringType().

        Returns
        -------
        UserDefinedFunction
            The registered UDF that can be called with columns.

        Examples
        --------
        >>> from pyspark.sql.types import IntegerType
        >>> spark.udf.register("inc_id", lambda x: x + 1, IntegerType())
        >>> result = spark.sql("SELECT inc_id(id) AS incremented FROM range(3)")
        """
        # If f is already a UDF (from udf()/pandas_udf()), reuse its packed command;
        # otherwise wrap the plain Python function as a SQL_BATCHED_UDF.
        if hasattr(f, "command") and hasattr(f, "evalType"):
            if returnType is not None:
                from pyspark.errors import PySparkTypeError

                raise PySparkTypeError(
                    errorClass="CANNOT_SPECIFY_RETURN_TYPE_FOR_UDF",
                    messageParameters={"arg_name": "f", "return_type": str(returnType)},
                )
            udf = f
        else:
            if returnType is None:
                returnType = StringType()
            udf = UserDefinedFunction(f, returnType, 100, name)

        # Send the registration to the Connect server so `name` resolves in SQL
        # (mirrors the reference client.register_udf: a RegisterFunction command
        # carrying the cloudpickled PythonUDF with no bound arguments).
        self.spark_session._registerPythonUdf(
            name, udf.returnType, udf.evalType, udf.command, udf.python_ver, udf.deterministic
        )
        return udf

    def registerJavaFunction(
        self,
        name: str,
        javaClassName: str,
        returnType: Optional[DataType] = None,
    ) -> None:
        """Register a Java UDF by fully-qualified class name so it is callable by
        ``name`` in SQL. Mirrors ``UDFRegistration.registerJavaFunction``.

        Parameters
        ----------
        name : str
            Name to register the function with.
        javaClassName : str
            Fully-qualified name of the Java class implementing the UDF.
        returnType : DataType or str, optional
            Return type (a DataType or a DDL string). Defaults to the class's own type.
        """
        rt_ddl = None
        if returnType is not None:
            rt_ddl = returnType if isinstance(returnType, str) else returnType.simpleString()
        self.spark_session._registerJavaFunction(name, javaClassName, rt_ddl, False)

    def registerJavaUDAF(self, name: str, javaClassName: str) -> None:
        """Register a Java user-defined aggregate function (UDAF) by class name.
        Mirrors ``UDFRegistration.registerJavaUDAF``.
        """
        self.spark_session._registerJavaFunction(name, javaClassName, None, True)
