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
"""Small subset of pyspark.sql.utils needed by the drop-in and vendored modules."""

from typing import Any, Union, TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from pyspark.pandas._typing import IndexOpsLike, SeriesOrIndex


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
    from pyspark.sql.column import Column
    from pyspark.pandas.data_type_ops.base import _is_extension_dtypes

    result = column_op(getattr(Column, func_name))(left, right)
    # It works as expected on extension dtype, so we don't need to call `fillna` for this case.
    if (fillna is not None) and (_is_extension_dtypes(left) or _is_extension_dtypes(right)):
        fillna = None
    # TODO(SPARK-43877): Fix behavior difference for compare binary functions.
    return result.fillna(fillna) if fillna is not None else result


def require_minimum_plotly_version() -> None:
    """Raise ImportError if plotly is not installed"""
    from pyspark.loose_version import LooseVersion
    from pyspark.errors import PySparkImportError

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
