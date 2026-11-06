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
"""Rust-backed pandas UDF entry points under the official import path.

Canonical home of the ``PandasUDFType`` / ``ArrowUDFType`` enums (mirroring reference
pyspark); ``pyspark.sql.functions`` re-exports them from here.
"""
import warnings
from inspect import getfullargspec, signature
from typing import get_type_hints

from pyspark.errors import PySparkValueError
from pyspark.sql.pandas.typehints import infer_eval_type
from pyspark.util import PythonEvalType  # (pyspark.sql.pandas.typehints reads it here too)


class PandasUDFType:
    """Pandas UDF Types. See :func:`pyspark.sql.functions.pandas_udf`."""

    SCALAR = 200
    SCALAR_ITER = 204
    GROUPED_MAP = 201
    GROUPED_AGG = 202
    GROUPED_AGG_ITER = 217
    MAP_ITER = 205
    COGROUPED_MAP = 206


class ArrowUDFType:
    """Arrow UDF Types. See :func:`pyspark.sql.functions.pandas_udf`."""

    SCALAR = 250
    SCALAR_ITER = 251
    GROUPED_AGG = 252
    GROUPED_AGG_ITER = 254



# Verbatim from PySpark: validates the function of a pandas or Arrow UDF for its eval type.
def _validate_vectorized_udf(f, evalType, kind: str = "pandas") -> int:
    assert kind in ["pandas", "arrow"], "kind should be either 'pandas' or 'arrow'"

    argspec = getfullargspec(f)

    # pandas UDF by type hints.
    if evalType in [
        PythonEvalType.SQL_SCALAR_PANDAS_UDF,
        PythonEvalType.SQL_SCALAR_PANDAS_ITER_UDF,
        PythonEvalType.SQL_GROUPED_AGG_PANDAS_UDF,
        PythonEvalType.SQL_GROUPED_AGG_PANDAS_ITER_UDF,
    ]:
        warnings.warn(
            "In Python 3.6+ and Spark 3.0+, it is preferred to specify type hints for "
            "pandas UDF instead of specifying pandas UDF type which will be deprecated "
            "in the future releases. See SPARK-28264 for more details.",
            UserWarning,
        )
    elif evalType in [
        PythonEvalType.SQL_SCALAR_ARROW_UDF,
        PythonEvalType.SQL_SCALAR_ARROW_ITER_UDF,
        PythonEvalType.SQL_GROUPED_AGG_ARROW_UDF,
        PythonEvalType.SQL_GROUPED_AGG_ARROW_ITER_UDF,
    ]:
        warnings.warn(
            "It is preferred to specify type hints for "
            "arrow UDF instead of specifying arrow UDF type.",
            UserWarning,
        )
    elif evalType in [
        PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF,
        PythonEvalType.SQL_GROUPED_MAP_PANDAS_ITER_UDF,
        PythonEvalType.SQL_MAP_PANDAS_ITER_UDF,
        PythonEvalType.SQL_MAP_ARROW_ITER_UDF,
        PythonEvalType.SQL_COGROUPED_MAP_PANDAS_UDF,
        PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE,
        PythonEvalType.SQL_TRANSFORM_WITH_STATE_PANDAS_UDF,
        PythonEvalType.SQL_TRANSFORM_WITH_STATE_PANDAS_INIT_STATE_UDF,
        PythonEvalType.SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_UDF,
        PythonEvalType.SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_INIT_STATE_UDF,
        PythonEvalType.SQL_GROUPED_MAP_ARROW_UDF,
        PythonEvalType.SQL_GROUPED_MAP_ARROW_ITER_UDF,
        PythonEvalType.SQL_COGROUPED_MAP_ARROW_UDF,
        PythonEvalType.SQL_ARROW_BATCHED_UDF,
    ]:
        # In case of 'SQL_GROUPED_MAP_PANDAS_UDF', deprecation warning is being triggered
        # at `apply` instead.
        # In case of 'SQL_MAP_PANDAS_ITER_UDF', 'SQL_MAP_ARROW_ITER_UDF' and
        # 'SQL_COGROUPED_MAP_PANDAS_UDF', the evaluation type will always be set.
        # In case of 'SQL_ARROW_BATCHED_UDF', no deprecation warning is required since it is not
        # exposed to users.
        pass
    elif len(argspec.annotations) > 0:
        try:
            type_hints = get_type_hints(f)
        except NameError:
            type_hints = {}
        evalType = infer_eval_type(signature(f), type_hints, kind)
        assert evalType is not None

    if evalType is None:
        # Set default is scalar UDF.
        if kind == "pandas":
            evalType = PythonEvalType.SQL_SCALAR_PANDAS_UDF
        else:
            evalType = PythonEvalType.SQL_SCALAR_ARROW_UDF

    kind_str = "pandas_udfs" if kind == "pandas" else "arrow_udfs"

    if (
        (
            evalType == PythonEvalType.SQL_SCALAR_PANDAS_UDF
            or evalType == PythonEvalType.SQL_SCALAR_ARROW_UDF
            or evalType == PythonEvalType.SQL_SCALAR_PANDAS_ITER_UDF
            or evalType == PythonEvalType.SQL_SCALAR_ARROW_ITER_UDF
        )
        and len(argspec.args) == 0
        and argspec.varargs is None
    ):
        raise PySparkValueError(
            errorClass="INVALID_PANDAS_UDF",
            messageParameters={
                "detail": f"0-arg {kind_str} are not supported. "
                f"Instead, create a 1-arg {kind_str} and ignore the arg in your function.",
            },
        )

    if evalType == PythonEvalType.SQL_GROUPED_MAP_PANDAS_UDF and len(argspec.args) not in (1, 2):
        raise PySparkValueError(
            errorClass="INVALID_PANDAS_UDF",
            messageParameters={
                "detail": f"{kind_str} with function type GROUPED_MAP or the function in "
                "groupby.applyInPandas must take either one argument (data) or "
                "two arguments (key, data).",
            },
        )

    if evalType == PythonEvalType.SQL_GROUPED_MAP_PANDAS_ITER_UDF and len(argspec.args) not in (
        1,
        2,
    ):
        raise PySparkValueError(
            errorClass="INVALID_PANDAS_UDF",
            messageParameters={
                "detail": "the function in groupby.applyInPandas with iterator API must take "
                "either one argument (batches: Iterator[pandas.DataFrame]) or two arguments "
                "(key, batches: Iterator[pandas.DataFrame]).",
            },
        )

    if evalType == PythonEvalType.SQL_GROUPED_MAP_ARROW_UDF and len(argspec.args) not in (1, 2):
        raise PySparkValueError(
            errorClass="INVALID_PANDAS_UDF",
            messageParameters={
                "detail": "the function in groupby.applyInArrow must take either one argument "
                "(data) or two arguments (key, data).",
            },
        )

    if evalType == PythonEvalType.SQL_COGROUPED_MAP_PANDAS_UDF and len(argspec.args) not in (2, 3):
        raise PySparkValueError(
            errorClass="INVALID_PANDAS_UDF",
            messageParameters={
                "detail": "the function in cogroup.applyInPandas must take either two arguments "
                "(left, right) or three arguments (key, left, right).",
            },
        )

    if evalType == PythonEvalType.SQL_COGROUPED_MAP_ARROW_UDF and len(argspec.args) not in (2, 3):
        raise PySparkValueError(
            errorClass="INVALID_PANDAS_UDF",
            messageParameters={
                "detail": "the function in cogroup.applyInArrow must take either two arguments "
                "(left, right) or three arguments (key, left, right).",
            },
        )

    return evalType

# After the enums: importing this module first imports pyspark.sql.functions, whose
# re-export of the enums then finds them already defined here.
from pyspark.sql.functions import arrow_udf, pandas_udf  # noqa: E402,F401

__all__ = ["arrow_udf", "pandas_udf", "PandasUDFType", "ArrowUDFType"]
