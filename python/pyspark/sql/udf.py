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

import functools
import sys
import warnings
from typing import Callable, Optional, Any

from pyspark.errors import PySparkNotImplementedError, PySparkRuntimeError, PySparkTypeError
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
    User defined function in Python. Mirrors ``pyspark.sql.connect.udf.UserDefinedFunction``.

    Notes
    -----
    The constructor of this class is not supposed to be directly called.
    Use :meth:`pyspark.sql.functions.udf` or :meth:`pyspark.sql.functions.pandas_udf`
    to create this instance.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        returnType: Any = StringType(),
        name: Optional[str] = None,
        evalType: int = PythonEvalType.SQL_BATCHED_UDF,
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
        self._returnType = returnType
        self._returnType_placeholder: Optional[DataType] = None
        self._name = name or (
            func.__name__ if hasattr(func, "__name__") else func.__class__.__name__
        )
        self.evalType = evalType
        self.deterministic = deterministic

    @property
    def returnType(self) -> DataType:
        # As PySpark's, the return type is parsed once, and checked for the eval type each
        # time. It is a concrete DataType, which the pickled command carries for the pandas
        # and Arrow workers; a DDL string is parsed on the client.
        if self._returnType_placeholder is None:
            self._returnType_placeholder = _as_concrete_datatype(self._returnType)

        UserDefinedFunction._check_return_type(self._returnType_placeholder, self.evalType)
        return self._returnType_placeholder

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
        # (whose to_column would otherwise treat a str as a literal). As PySpark's, each
        # keyword argument is a NamedArgumentExpression, after the positional ones.
        cols = [col(a) if isinstance(a, str) else a for a in args]
        named = [(k, col(v) if isinstance(v, str) else v) for k, v in kwargs.items()]

        # As PySpark's PythonUDF, the command is the pickled (func, returnType) for the
        # server-side Python worker, and pyfunc_make_udf builds the proto output type.
        return_type = self.returnType
        return _pyspark.functions.pyfunc_make_udf(
            self._name,
            return_type,
            self.evalType,
            CloudPickleSerializer().dumps((self.func, return_type)),
            "%d.%d" % sys.version_info[:2],
            *cols,
            deterministic=self.deterministic,
            named_args=named,
        )

    # Verbatim from PySpark Connect (pyspark.sql.connect.udf.UserDefinedFunction).
    # This function is for improving the online help system in the interactive interpreter.
    # For example, the built-in help / pydoc.help. It wraps the UDF with the docstring and
    # argument annotation. (See: SPARK-19161)
    def _wrapped(self) -> Any:
        """
        Wrap this udf with a function and attach docstring from func
        """

        # It is possible for a callable instance without __name__ attribute or/and
        # __module__ attribute to be wrapped here. For example, functools.partial. In this case,
        # we should avoid wrapping the attributes from the wrapped function to the wrapper
        # function. So, we take out these attribute names from the default names to set and
        # then manually assign it after being wrapped.
        assignments = tuple(
            a for a in functools.WRAPPER_ASSIGNMENTS if a != "__name__" and a != "__module__"
        )

        @functools.wraps(self.func, assigned=assignments)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self(*args, **kwargs)

        wrapper.__name__ = self._name
        wrapper.__module__ = (
            self.func.__module__
            if hasattr(self.func, "__module__")
            else self.func.__class__.__module__
        )

        wrapper.func = self.func  # type: ignore[attr-defined]
        wrapper.returnType = self.returnType  # type: ignore[attr-defined]
        wrapper.evalType = self.evalType  # type: ignore[attr-defined]
        wrapper.deterministic = self.deterministic  # type: ignore[attr-defined]
        wrapper.asNondeterministic = functools.wraps(  # type: ignore[attr-defined]
            self.asNondeterministic
        )(lambda: self.asNondeterministic()._wrapped())
        wrapper._unwrapped = self  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    def asNondeterministic(self) -> "UserDefinedFunction":
        """
        Updates UserDefinedFunction to nondeterministic.

        .. versionadded:: 3.4.0
        """
        self.deterministic = False
        return self


def udf(
    f: Optional[Any] = None,
    returnType: Any = StringType(),
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
    function or callable
        If f is provided, returns the function wrapper of the UDF, as PySpark does. If f is
        None or a return type, returns a decorator.

    Examples
    --------
    >>> from pyspark.sql import functions as F
    >>> from pyspark.sql.types import IntegerType
    >>> u = F.udf(lambda x: x + 1, IntegerType())
    >>> result = spark.range(3).select(u(F.col('id')).alias('inc_id'))
    """
    # Verbatim from PySpark Connect (pyspark.sql.connect.functions.builtin.udf).
    if f is None or isinstance(f, (str, DataType)):
        # If DataType has been passed as a positional argument
        # for decorator use it as a returnType
        return_type = f or returnType
        return functools.partial(
            _create_py_udf,
            returnType=return_type,
            useArrow=useArrow,
        )
    else:
        return _create_py_udf(f=f, returnType=returnType, useArrow=useArrow)


# Verbatim from PySpark Connect (pyspark.sql.connect.udf).
def _create_py_udf(
    f: Callable[..., Any],
    returnType: Any,
    useArrow: Optional[bool] = None,
) -> Any:
    from pyspark.sql.pandas.utils import (
        require_minimum_pandas_version,
        require_minimum_pyarrow_version,
    )

    is_arrow_enabled = False
    if useArrow is None:
        is_arrow_enabled = False
        try:
            from pyspark.sql import SparkSession

            session = SparkSession.active()
            is_arrow_enabled = (
                str(session.conf.get("spark.sql.execution.pythonUDF.arrow.enabled")).lower()
                == "true"
            )
        except PySparkRuntimeError as e:
            if e.getCondition() == "NO_ACTIVE_OR_DEFAULT_SESSION":
                pass  # Just uses the default if no session found.
            else:
                raise e
    else:
        is_arrow_enabled = useArrow

    if is_arrow_enabled:
        try:
            require_minimum_pandas_version()
            require_minimum_pyarrow_version()
        except ImportError:
            is_arrow_enabled = False
            warnings.warn(
                "Arrow optimization failed to enable because PyArrow or Pandas is not installed. "
                "Falling back to a non-Arrow-optimized UDF.",
                RuntimeWarning,
            )

    eval_type: Optional[int] = None
    if useArrow is None:
        # If the user doesn't explicitly set useArrow
        from pyspark.sql.pandas.typehints import infer_eval_type_for_udf

        try:
            # Try to infer the eval type from type hints
            eval_type = infer_eval_type_for_udf(f)
        except Exception:
            warnings.warn("Cannot infer the eval type from type hints. ", UserWarning)

    if eval_type is None:
        if is_arrow_enabled:
            # Arrow optimized Python UDF
            eval_type = PythonEvalType.SQL_ARROW_BATCHED_UDF
        else:
            # Fallback to Regular Python UDF
            eval_type = PythonEvalType.SQL_BATCHED_UDF

    return _create_udf(f, returnType, eval_type)


# Verbatim from PySpark Connect (pyspark.sql.connect.udf); pyspark.sql.pandas.functions
# creates pandas and Arrow UDFs with it.
def _create_udf(
    f: Callable[..., Any],
    returnType: Any,
    evalType: int,
    name: Optional[str] = None,
    deterministic: bool = True,
) -> Any:
    # Set the name of the UserDefinedFunction object to be the name of function f
    udf_obj = UserDefinedFunction(
        f, returnType=returnType, name=name, evalType=evalType, deterministic=deterministic
    )
    return udf_obj._wrapped()


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
    ) -> Any:
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
        function
            The registered UDF that can be called with columns: ``f`` if it is a UDF,
            or the function wrapper of the UDF of ``f``, as PySpark returns.

        Examples
        --------
        >>> from pyspark.sql.types import IntegerType
        >>> spark.udf.register("inc_id", lambda x: x + 1, IntegerType())
        >>> result = spark.sql("SELECT inc_id(id) AS incremented FROM range(3)")
        """
        # As PySpark Connect's: a UDF (from udf() or pandas_udf()) is registered with its
        # eval type, and a plain Python function as a SQL_BATCHED_UDF of `name`. The
        # registration is a RegisterFunction command carrying the cloudpickled PythonUDF
        # with no bound arguments, as the reference client.register_udf sends, so `name`
        # resolves in SQL.
        if hasattr(f, "asNondeterministic"):
            if returnType is not None:
                raise PySparkTypeError(
                    errorClass="CANNOT_SPECIFY_RETURN_TYPE_FOR_UDF",
                    messageParameters={"arg_name": "f", "return_type": str(returnType)},
                )
            if f.evalType not in [
                PythonEvalType.SQL_BATCHED_UDF,
                PythonEvalType.SQL_ARROW_BATCHED_UDF,
                PythonEvalType.SQL_SCALAR_PANDAS_UDF,
                PythonEvalType.SQL_SCALAR_ARROW_UDF,
                PythonEvalType.SQL_SCALAR_PANDAS_ITER_UDF,
                PythonEvalType.SQL_SCALAR_ARROW_ITER_UDF,
                PythonEvalType.SQL_GROUPED_AGG_PANDAS_UDF,
                PythonEvalType.SQL_GROUPED_AGG_ARROW_UDF,
                PythonEvalType.SQL_GROUPED_AGG_PANDAS_ITER_UDF,
                PythonEvalType.SQL_GROUPED_AGG_ARROW_ITER_UDF,
            ]:
                raise PySparkTypeError(
                    errorClass="INVALID_UDF_EVAL_TYPE",
                    messageParameters={
                        "eval_type": "SQL_BATCHED_UDF, SQL_ARROW_BATCHED_UDF, "
                        "SQL_SCALAR_PANDAS_UDF, SQL_SCALAR_ARROW_UDF, "
                        "SQL_SCALAR_PANDAS_ITER_UDF, SQL_SCALAR_ARROW_ITER_UDF, "
                        "SQL_GROUPED_AGG_PANDAS_UDF, SQL_GROUPED_AGG_ARROW_UDF, "
                        "SQL_GROUPED_AGG_PANDAS_ITER_UDF or SQL_GROUPED_AGG_ARROW_ITER_UDF"
                    },
                )
            self._register_udf(f.func, f.returnType, name, f.evalType, f.deterministic)
            return f
        else:
            if returnType is None:
                returnType = StringType()
            py_udf = _create_udf(
                f, returnType=returnType, evalType=PythonEvalType.SQL_BATCHED_UDF, name=name
            )

            self._register_udf(py_udf.func, py_udf.returnType, name)
            return py_udf

    def _register_udf(
        self,
        function: Callable[..., Any],
        return_type: DataType,
        name: str,
        eval_type: int = PythonEvalType.SQL_BATCHED_UDF,
        deterministic: bool = True,
    ) -> None:
        # As PySpark's SparkConnectClient.register_udf: the PythonUDF command is the pickled
        # (function, return type).
        self.spark_session._registerPythonUdf(
            name,
            return_type,
            eval_type,
            CloudPickleSerializer().dumps((function, return_type)),
            "%d.%d" % sys.version_info[:2],
            deterministic,
        )

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
