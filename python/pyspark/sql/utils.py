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
"""Small subset of pyspark.sql.utils needed by the drop-in and vendored modules, and the
exceptions that PySpark's re-exports from ``pyspark.errors``, of their import path before it."""

import functools
import inspect
import os
from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union, cast, overload

# Re-exported, as PySpark's pyspark.sql.utils does: `from pyspark.sql.utils import
# AnalysisException` is their import path before pyspark.errors.
from pyspark.errors import (  # noqa: F401
    AnalysisException,
    IllegalArgumentException,
    ParseException,
    PySparkImportError,
    PySparkNotImplementedError,
    PySparkRuntimeError,
    PythonException,
    QueryExecutionException,
    SparkUpgradeException,
    StreamingQueryException,
    UnknownException,
)
from pyspark.errors.exceptions.captured import CapturedException  # noqa: F401

if TYPE_CHECKING:
    from py4j.java_gateway import JavaObject, JVMView

    from pyspark.pandas._typing import IndexOpsLike, SeriesOrIndex


FuncT = TypeVar("FuncT", bound=Callable[..., Any])


def to_str(value: Any) -> Optional[str]:
    """
    A wrapper over str(), but converts bool values to lower case strings.
    If None is given, just returns None, instead of converting it to string "None".
    """
    if isinstance(value, bool):
        return str(value).lower()
    elif value is None:
        return value
    else:
        return str(value)


def to_scala_map(jvm: "JVMView", dic: dict) -> "JavaObject":
    """
    Convert a dict into a Scala Map.
    """
    assert jvm is not None
    return jvm.PythonUtils.toScalaMap(dic)


def enum_to_value(value: Any) -> Any:
    """Convert an Enum to its value if it is not None."""
    return enum_to_value(value.value) if value is not None and isinstance(value, Enum) else value


def is_remote() -> bool:
    """This client is always a Spark Connect (remote) client."""
    return True


def is_timestamp_ntz_preferred() -> bool:
    """Whether TIMESTAMP_NTZ is the preferred timestamp type.

    Best-effort for the Connect drop-in: honor the active session's
    spark.sql.timestampType when one is running, else False.
    """
    try:
        from pyspark.sql import SparkSession

        session = SparkSession.getActiveSession()
        if session is None:
            return False
        return session.conf.get("spark.sql.timestampType", None) == "TIMESTAMP_NTZ"
    except Exception:
        return False


def get_lit_sql_str(val: str) -> str:
    """Get SQL string literal representation.

    Equivalent to `lit(val)._jc.expr().sql()` for string typed val.
    See `sql` definition in `sql/catalyst/src/main/scala/org/apache/spark/
    sql/catalyst/expressions/literals.scala`
    """
    return "'" + val.replace("\\", "\\\\").replace("'", "\\'") + "'"


def pyspark_column_op(
    func_name: str, left: "IndexOpsLike", right: Any, fillna: Any = None
) -> Union["SeriesOrIndex", None]:
    """Wrapper function for column_op to get proper Column class."""
    from pyspark.pandas.base import column_op
    from pyspark.pandas.data_type_ops.base import _is_extension_dtypes
    from pyspark.sql.column import Column

    result = column_op(getattr(Column, func_name))(left, right)
    # It works as expected on extension dtype, so we don't need to call `fillna` for this case.
    if (fillna is not None) and (_is_extension_dtypes(left) or _is_extension_dtypes(right)):
        fillna = None
    # TODO(SPARK-43877): Fix behavior difference for compare binary functions.
    return result.fillna(fillna) if fillna is not None else result


def require_minimum_plotly_version() -> None:
    """Raise ImportError if plotly is not installed"""
    from pyspark.loose_version import LooseVersion

    minimum_plotly_version = "4.8"

    try:
        import plotly

        have_plotly = True
    except ImportError as error:
        have_plotly = False
        raised_error = error
    if not have_plotly:
        raise PySparkImportError(
            errorClass="PACKAGE_NOT_INSTALLED",
            messageParameters={
                "package_name": "Plotly",
                "minimum_version": str(minimum_plotly_version),
            },
        ) from raised_error
    if LooseVersion(plotly.__version__) < LooseVersion(minimum_plotly_version):
        raise PySparkImportError(
            errorClass="UNSUPPORTED_PACKAGE_VERSION",
            messageParameters={
                "package_name": "Plotly",
                "minimum_version": str(minimum_plotly_version),
                "current_version": str(plotly.__version__),
            },
        )


class NumpyHelper:
    @staticmethod
    def linspace(start: float, stop: float, num: int) -> Sequence[float]:
        if num == 1:
            return [float(start)]
        step = (float(stop) - float(start)) / (num - 1)
        return [start + step * i for i in range(num)]


def try_remote_session_classmethod(f: FuncT) -> FuncT:
    """Mark API supported from Spark Connect, as PySpark's: a classmethod of the classic
    ``pyspark.sql.SparkSession`` runs that of ``pyspark.sql.connect.session.SparkSession``."""

    @functools.wraps(f)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if is_remote() and "PYSPARK_NO_NAMESPACE_SHARE" not in os.environ:
            from pyspark.sql.connect.session import SparkSession

            assert inspect.isclass(args[0])
            return getattr(SparkSession, f.__name__)(*args[1:], **kwargs)
        else:
            return f(*args, **kwargs)

    return cast(FuncT, wrapped)


@overload
def remote_only(func: property) -> property: ...


@overload
def remote_only(func: FuncT) -> FuncT: ...


def remote_only(func: Union[FuncT, property]) -> Union[FuncT, property]:
    """
    Decorator to mark a function or method as only available in Spark Connect.

    This decorator allows for easy identification of Spark Connect-specific APIs.
    """
    if isinstance(func, property):
        # If it's a property, we need to set the attribute on the getter function
        getter_func = func.fget
        getter_func._remote_only = True  # type: ignore[union-attr]
        return property(getter_func)
    else:
        func._remote_only = True  # type: ignore[attr-defined]
        return func
