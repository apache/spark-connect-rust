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

__all__ = [
    "SparkConnectException",
    "SparkConnectGrpcException",
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
