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

from pyspark._pyspark import (
    AnalyzeArgument, PartitioningColumn, OrderingColumn, SelectedColumn,
    AnalyzeResult, SkipRestOfInputTableException,
)

import inspect
import sys
from typing import Any, Optional, Type

from pyspark.errors import PySparkAttributeError, PySparkTypeError
from pyspark.serializers import CloudPickleSerializer

# Eval types for Python UDTFs (mirrors pyspark.util.PythonEvalType).
SQL_TABLE_UDF = 300
SQL_ARROW_TABLE_UDF = 301
SQL_ARROW_UDTF = 302


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
        self.returnType = returnType
        self.evalType = evalType
        self.name = name or (
            func.__name__ if hasattr(func, "__name__") else "udtf"
        )
        self.deterministic = deterministic
        self.python_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        # Connect sends the schema separately in PythonUDTF.return_type. The
        # worker command contains only the handler class, unlike scalar UDFs.
        self.command = CloudPickleSerializer().dumps(func)

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
