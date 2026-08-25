#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License") you may not use this file except in compliance with
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
PySpark exceptions - re-exported from the official Apache Spark package.

For Spark Connect-only mode, we re-export the exception classes from the
upstream pyspark package to ensure compatibility with the testing harness.
"""

# Re-export the exception classes from the vendored pyspark.errors.exceptions
# package so Spark Connect code and the testing harness see the same error
# types. Fall back to minimal stubs if they are unavailable.
try:
    from pyspark.errors.exceptions.base import (
        PySparkException,
        AnalysisException,
        SessionNotSameException,
        TempTableAlreadyExistsException,
        ParseException,
        IllegalArgumentException,
        ArithmeticException,
        UnsupportedOperationException,
        ArrayIndexOutOfBoundsException,
        DateTimeException,
        NumberFormatException,
        StreamingQueryException,
        QueryExecutionException,
        PythonException,
        UnknownException,
        SparkRuntimeException,
        SparkUpgradeException,
        SparkNoSuchElementException,
        PySparkTypeError,
        PySparkValueError,
        PySparkImportError,
        PySparkIndexError,
        PySparkAttributeError,
        PySparkRuntimeError,
        PySparkAssertionError,
        PySparkNotImplementedError,
        PySparkPicklingError,
        PySparkKeyError,
        QueryContext,
        QueryContextType,
        StreamingPythonRunnerInitializationException,
        PickleException,
    )
except ImportError as e:
    # If we can't import from upstream, define minimal stubs
    class PySparkException(Exception):
        """Base PySpark exception"""
        pass

    class PySparkAssertionError(PySparkException, AssertionError):
        """PySpark assertion error"""
        pass

    class PySparkTypeError(PySparkException, TypeError):
        """PySpark type error"""
        pass

    class AnalysisException(PySparkException):
        """Analysis exception"""
        pass

    class SessionNotSameException(PySparkException):
        """Session not same exception"""
        pass

    class TempTableAlreadyExistsException(PySparkException):
        """Temp table already exists exception"""
        pass

    class ParseException(PySparkException):
        """Parse exception"""
        pass

    class IllegalArgumentException(PySparkException):
        """Illegal argument exception"""
        pass

    class ArithmeticException(PySparkException):
        """Arithmetic exception"""
        pass

    class UnsupportedOperationException(PySparkException):
        """Unsupported operation exception"""
        pass

    class ArrayIndexOutOfBoundsException(PySparkException):
        """Array index out of bounds exception"""
        pass

    class DateTimeException(PySparkException):
        """Date time exception"""
        pass

    class NumberFormatException(PySparkException):
        """Number format exception"""
        pass

    class StreamingQueryException(PySparkException):
        """Streaming query exception"""
        pass

    class QueryExecutionException(PySparkException):
        """Query execution exception"""
        pass

    class PythonException(PySparkException):
        """Python exception"""
        pass

    class UnknownException(PySparkException):
        """Unknown exception"""
        pass

    class SparkRuntimeException(PySparkException):
        """Spark runtime exception"""
        pass

    class SparkUpgradeException(PySparkException):
        """Spark upgrade exception"""
        pass

    class SparkNoSuchElementException(PySparkException):
        """Spark no such element exception"""
        pass

    class PySparkValueError(PySparkException, ValueError):
        """PySpark value error"""
        pass

    class PySparkImportError(PySparkException, ImportError):
        """PySpark import error"""
        pass

    class PySparkIndexError(PySparkException, IndexError):
        """PySpark index error"""
        pass

    class PySparkAttributeError(PySparkException, AttributeError):
        """PySpark attribute error"""
        pass

    class PySparkRuntimeError(PySparkException, RuntimeError):
        """PySpark runtime error"""
        pass

    class PySparkNotImplementedError(PySparkException, NotImplementedError):
        """PySpark not implemented error"""
        pass

    class PySparkPicklingError(PySparkException):
        """PySpark pickling error"""
        pass

    class PySparkKeyError(PySparkException, KeyError):
        """PySpark key error"""
        pass

    class QueryContextType:
        """Query context type"""
        pass

    class QueryContext:
        """Query context"""
        pass

    class StreamingPythonRunnerInitializationException(PySparkException):
        """Streaming Python runner initialization exception"""
        pass

    class PickleException(PySparkException):
        """Pickle exception"""
        pass

__all__ = [
    "PySparkException",
    "AnalysisException",
    "SessionNotSameException",
    "TempTableAlreadyExistsException",
    "ParseException",
    "IllegalArgumentException",
    "ArithmeticException",
    "UnsupportedOperationException",
    "ArrayIndexOutOfBoundsException",
    "DateTimeException",
    "NumberFormatException",
    "StreamingQueryException",
    "QueryExecutionException",
    "PythonException",
    "UnknownException",
    "SparkRuntimeException",
    "SparkUpgradeException",
    "SparkNoSuchElementException",
    "PySparkTypeError",
    "PySparkValueError",
    "PySparkImportError",
    "PySparkIndexError",
    "PySparkAttributeError",
    "PySparkRuntimeError",
    "PySparkAssertionError",
    "PySparkNotImplementedError",
    "PySparkPicklingError",
    "PySparkKeyError",
    "QueryContext",
    "QueryContextType",
    "StreamingPythonRunnerInitializationException",
    "PickleException",
]
