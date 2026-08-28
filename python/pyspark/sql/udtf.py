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

import sys
from typing import Any, Optional, Type

from pyspark.serializers import CloudPickleSerializer

# Eval types for Python UDTFs (mirrors pyspark.util.PythonEvalType).
SQL_TABLE_UDF = 300
SQL_ARROW_TABLE_UDF = 301


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
        deterministic: bool = True,
    ):
        self.func = func
        self.returnType = returnType
        self.evalType = evalType
        self.name = name or (
            func.__name__ if hasattr(func, "__name__") else "udtf"
        )
        self.deterministic = deterministic
        self.python_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        # Cloudpickle the handler class + declared return type for the worker.
        self.command = CloudPickleSerializer().dumps((func, returnType))

    def _active_session(self):
        from pyspark.sql import SparkSession

        session = SparkSession.getActiveSession()
        if session is None:
            raise RuntimeError(
                "No active SparkSession; a UDTF must be called within an active session."
            )
        return session

    def __call__(self, *args: Any) -> Any:
        from pyspark import _pyspark

        return _pyspark.functions.pyfunc_make_udtf(
            self._active_session(),
            self.name,
            self.returnType,
            self.evalType,
            self.command,
            self.python_ver,
            *args,
        )

    def asDeterministic(self) -> "UserDefinedTableFunction":
        """Return a copy of this UDTF marked as deterministic. Mirrors
        ``UserDefinedTableFunction.asDeterministic``."""
        return UserDefinedTableFunction(
            self.func,
            returnType=self.returnType,
            name=self.name,
            evalType=self.evalType,
            deterministic=True,
        )


def udtf(
    cls: Optional[Type] = None,
    *,
    returnType: Optional[Any] = None,
    useArrow: Optional[bool] = None,
) -> Any:
    """Create a user-defined table function (UDTF).

    Use as ``@udtf(returnType=...)`` on a handler class, or call
    ``udtf(MyClass, returnType=...)`` directly.
    """
    eval_type = SQL_ARROW_TABLE_UDF if useArrow is True else SQL_TABLE_UDF

    def _udtf_decorator(handler: Type) -> UserDefinedTableFunction:
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

    Mirrors ``pyspark.sql.functions.arrow_udtf``: like :func:`udtf` but the handler
    operates on Arrow data (``SQL_ARROW_TABLE_UDF``).
    """
    def _arrow_udtf_decorator(handler: Type) -> UserDefinedTableFunction:
        return UserDefinedTableFunction(
            handler, returnType=returnType, evalType=SQL_ARROW_TABLE_UDF
        )

    if cls is not None:
        return _arrow_udtf_decorator(cls)
    return _arrow_udtf_decorator


class UDTFRegistration:
    """Registration accessor mirroring ``SparkSession.udtf``."""

    def register(self, name: str, f: Any, returnType: Optional[Any] = None):
        if isinstance(f, UserDefinedTableFunction):
            f.name = name
            return f
        return UserDefinedTableFunction(f, returnType=returnType, name=name)
