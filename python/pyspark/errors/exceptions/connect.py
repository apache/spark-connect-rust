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
"""Connect-flavored exceptions.

In upstream pyspark this module parses gRPC ``FetchErrorDetails`` into typed
exceptions. In this Rust-backed drop-in the transport already performs that
parsing in Rust and raises the canonical ``pyspark.errors`` classes, so the
Connect exception types are the same classes re-exported under this import path
(what upstream code and tests import as ``pyspark.errors.exceptions.connect.*``).
"""
from typing import Any

from pyspark.errors.exceptions.base import (
    AnalysisException,
    ArithmeticException,
    ArrayIndexOutOfBoundsException,
    DateTimeException,
    IllegalArgumentException,
    NumberFormatException,
    ParseException,
    PySparkException,
    PythonException,
    QueryContext as BaseQueryContext,
    QueryContextType,
    QueryExecutionException,
    SparkNoSuchElementException,
    SparkRuntimeException,
    SparkUpgradeException,
    StreamingPythonRunnerInitializationException,
    StreamingQueryException,
    TempTableAlreadyExistsException,
    UnsupportedOperationException,
)

# `SparkConnectException` is the Connect base; here it aliases the canonical base class.
SparkConnectException = PySparkException
SparkConnectGrpcException = PySparkException


class SparkException(SparkConnectGrpcException):
    """ """


class InvalidPlanInput(SparkConnectGrpcException):
    """Error thrown when a connect plan is not valid."""


def _unsupported(class_name: str, method_name: str) -> "UnsupportedOperationException":
    return UnsupportedOperationException(
        errorClass="UNSUPPORTED_CALL.WITHOUT_SUGGESTION",
        messageParameters={"className": class_name, "methodName": method_name},
    )


class SQLQueryContext(BaseQueryContext):
    """A SQL query context parsed from server error details."""

    def __init__(self, q: Any):
        self._q = q

    def contextType(self) -> QueryContextType:
        return QueryContextType.SQL

    def objectType(self) -> str:
        return str(self._q.object_type)

    def objectName(self) -> str:
        return str(self._q.object_name)

    def startIndex(self) -> int:
        return int(self._q.start_index)

    def stopIndex(self) -> int:
        return int(self._q.stop_index)

    def fragment(self) -> str:
        return str(self._q.fragment)

    def callSite(self) -> str:
        raise _unsupported("SQLQueryContext", "callSite")

    def summary(self) -> str:
        return str(self._q.summary)


class DataFrameQueryContext(BaseQueryContext):
    """A DataFrame query context parsed from server error details."""

    def __init__(self, q: Any):
        self._q = q

    def contextType(self) -> QueryContextType:
        return QueryContextType.DataFrame

    def objectType(self) -> str:
        raise _unsupported("DataFrameQueryContext", "objectType")

    def objectName(self) -> str:
        raise _unsupported("DataFrameQueryContext", "objectName")

    def startIndex(self) -> int:
        raise _unsupported("DataFrameQueryContext", "startIndex")

    def stopIndex(self) -> int:
        raise _unsupported("DataFrameQueryContext", "stopIndex")

    def fragment(self) -> str:
        return str(self._q.fragment)

    def callSite(self) -> str:
        return str(self._q.call_site)

    def summary(self) -> str:
        return str(self._q.summary)


__all__ = [
    "SparkConnectException",
    "SparkConnectGrpcException",
    "SparkException",
    "InvalidPlanInput",
    "SQLQueryContext",
    "DataFrameQueryContext",
    "AnalysisException",
    "ArithmeticException",
    "ArrayIndexOutOfBoundsException",
    "DateTimeException",
    "IllegalArgumentException",
    "NumberFormatException",
    "ParseException",
    "PySparkException",
    "PythonException",
    "QueryExecutionException",
    "SparkNoSuchElementException",
    "SparkRuntimeException",
    "SparkUpgradeException",
    "StreamingPythonRunnerInitializationException",
    "StreamingQueryException",
    "TempTableAlreadyExistsException",
    "UnsupportedOperationException",
]
