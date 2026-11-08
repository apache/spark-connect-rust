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
"""Exceptions raised from Spark Connect.

The classes are those of upstream pyspark. Upstream, ``convert_exception`` converts the gRPC
``ErrorInfo`` and ``FetchErrorDetails`` of an error that the server raised into one of them.
In this Rust-backed drop-in, the Rust transport fetches and reads those, and
``_convert_server_error`` picks and builds the exception from what it read, as the rest of
upstream ``_convert_exception`` does. ``grpc`` is optional: without it, ``StatusCode`` is an
enum with the members and values of ``grpc.StatusCode``.
"""
import enum
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

try:
    from grpc import StatusCode
except ImportError:  # pragma: no cover - grpcio is not a dependency of this client

    class StatusCode(enum.Enum):  # type: ignore[no-redef]
        """The status codes of gRPC, as ``grpc.StatusCode`` has them."""

        OK = (0, "ok")
        CANCELLED = (1, "cancelled")
        UNKNOWN = (2, "unknown")
        INVALID_ARGUMENT = (3, "invalid argument")
        DEADLINE_EXCEEDED = (4, "deadline exceeded")
        NOT_FOUND = (5, "not found")
        ALREADY_EXISTS = (6, "already exists")
        PERMISSION_DENIED = (7, "permission denied")
        RESOURCE_EXHAUSTED = (8, "resource exhausted")
        FAILED_PRECONDITION = (9, "failed precondition")
        ABORTED = (10, "aborted")
        OUT_OF_RANGE = (11, "out of range")
        UNIMPLEMENTED = (12, "unimplemented")
        INTERNAL = (13, "internal")
        UNAVAILABLE = (14, "unavailable")
        DATA_LOSS = (15, "data loss")
        UNAUTHENTICATED = (16, "unauthenticated")


from pyspark.errors.exceptions.base import (
    AnalysisException as BaseAnalysisException,
    IllegalArgumentException as BaseIllegalArgumentException,
    ArithmeticException as BaseArithmeticException,
    UnsupportedOperationException as BaseUnsupportedOperationException,
    ArrayIndexOutOfBoundsException as BaseArrayIndexOutOfBoundsException,
    DateTimeException as BaseDateTimeException,
    NumberFormatException as BaseNumberFormatException,
    ParseException as BaseParseException,
    PySparkException,
    PythonException as BasePythonException,
    StreamingQueryException as BaseStreamingQueryException,
    QueryExecutionException as BaseQueryExecutionException,
    SparkRuntimeException as BaseSparkRuntimeException,
    SparkNoSuchElementException as BaseNoSuchElementException,
    SparkUpgradeException as BaseSparkUpgradeException,
    QueryContext as BaseQueryContext,
    QueryContextType,
    StreamingPythonRunnerInitializationException as BaseStreamingPythonRunnerInitException,
    PickleException as BasePickleException,
    UnknownException as BaseUnknownException,
    recover_python_exception,
)

# Exported here too, as this module did before it had the classes of upstream pyspark.
from pyspark.errors.exceptions.base import TempTableAlreadyExistsException  # noqa: F401


class _MultiThreadedRendezvous(Exception):
    """The gRPC error of a call that the server ended with an error, as the ``grpc`` package's
    ``_MultiThreadedRendezvous``, of its status code and details: PySpark Connect builds its
    exception of an error of the server as it handles it, so the DataFrame and SQL query context
    loggers log it as the exception, of its class, its ``str`` and its traceback."""

    def __init__(self, code: Any, details: str) -> None:
        super().__init__(code, details)
        self._code = code
        self._details = details

    def __str__(self) -> str:
        grpc_status = self._code.value[0] if isinstance(self._code.value, tuple) else self._code
        debug = "UNKNOWN:Error received from peer {grpc_message:%s, grpc_status:%s}" % (
            json.dumps(self._details),
            grpc_status,
        )
        return (
            "<_MultiThreadedRendezvous of RPC that terminated with:\n"
            "\tstatus = %s\n"
            '\tdetails = "%s"\n'
            '\tdebug_error_string = "%s"\n'
            ">" % (self._code, self._details, debug)
        )

    __repr__ = __str__


class SparkConnectException(PySparkException):
    """
    Exception thrown from Spark Connect.
    """

    @property
    def operation_id(self) -> Optional[str]:
        """The Spark Connect ExecutePlan operation ID, when available.

        .. versionadded:: 4.3.0
        """
        return getattr(self, "_operation_id", None)


class SparkConnectGrpcException(SparkConnectException):
    """
    Base class to handle the errors from GRPC.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        errorClass: Optional[str] = None,
        messageParameters: Optional[Dict[str, str]] = None,
        reason: Optional[str] = None,
        sql_state: Optional[str] = None,
        server_stacktrace: Optional[str] = None,
        display_server_stacktrace: bool = False,
        contexts: Optional[List[BaseQueryContext]] = None,
        grpc_status_code: StatusCode = StatusCode.UNKNOWN,
        breaking_change_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if contexts is None:
            contexts = []
        self._message = message  # type: ignore[assignment]
        if reason is not None:
            self._message = f"({reason}) {self._message}"

        # PySparkException has the assumption that errorClass and messageParameters are
        # only occurring together. If only one is set, we assume the message to be fully
        # parsed.
        tmp_error_class = errorClass
        tmp_message_parameters = messageParameters
        if errorClass is not None and messageParameters is None:
            tmp_error_class = None
        elif errorClass is None and messageParameters is not None:
            tmp_message_parameters = None

        super().__init__(
            message=self._message,
            errorClass=tmp_error_class,
            messageParameters=tmp_message_parameters,
        )
        self._errorClass = errorClass
        self._sql_state: Optional[str] = sql_state
        self._stacktrace: Optional[str] = server_stacktrace
        self._display_stacktrace: bool = display_server_stacktrace
        self._contexts: List[BaseQueryContext] = contexts
        self._grpc_status_code = grpc_status_code
        self._breaking_change_info: Optional[Dict[str, Any]] = breaking_change_info
        if self._contexts:
            # PySpark Connect builds this exception as it handles the gRPC error of the call,
            # which its query context loggers log as the exception. The Rust transport has no
            # such error in Python, so its stand-in is handled as this logs.
            try:
                raise _MultiThreadedRendezvous(self._grpc_status_code, self._message)
            except _MultiThreadedRendezvous as rpc_error:
                self._log_exception()
                rpc_error.__traceback__ = None
                # As PySpark Connect raises this exception from None, as it handles that error.
                self.__context__ = rpc_error
                self.__suppress_context__ = True

    def getSqlState(self) -> Optional[str]:
        if self._sql_state is not None:
            return self._sql_state
        else:
            return super().getSqlState()

    def getStackTrace(self) -> Optional[str]:
        return self._stacktrace

    def getMessage(self) -> str:
        desc = self._message
        if self._display_stacktrace:
            desc += "\n\nJVM stacktrace:\n%s" % self._stacktrace
        return desc

    def getGrpcStatusCode(self) -> StatusCode:
        return self._grpc_status_code

    def getBreakingChangeInfo(self) -> Optional[Dict[str, Any]]:
        """
        Returns the breaking change info for an error, or None.

        For Spark Connect exceptions, this returns the breaking change info
        received from the server, rather than looking it up from local error files.
        """
        return self._breaking_change_info

    def __str__(self) -> str:
        return self.getMessage()


class UnknownException(SparkConnectGrpcException, BaseUnknownException):
    """
    Exception for unmapped errors in Spark Connect.
    This class is functionally identical to SparkConnectGrpcException but has a different name
    for consistency.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        errorClass: Optional[str] = None,
        messageParameters: Optional[Dict[str, str]] = None,
        reason: Optional[str] = None,
        sql_state: Optional[str] = None,
        server_stacktrace: Optional[str] = None,
        display_server_stacktrace: bool = False,
        contexts: Optional[List[BaseQueryContext]] = None,
        grpc_status_code: StatusCode = StatusCode.UNKNOWN,
        breaking_change_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            errorClass=errorClass,
            messageParameters=messageParameters,
            reason=reason,
            sql_state=sql_state,
            server_stacktrace=server_stacktrace,
            display_server_stacktrace=display_server_stacktrace,
            contexts=contexts,
            grpc_status_code=grpc_status_code,
            breaking_change_info=breaking_change_info,
        )


class AnalysisException(SparkConnectGrpcException, BaseAnalysisException):
    """
    Failed to analyze a SQL query plan, thrown from Spark Connect.
    """


class ParseException(AnalysisException, BaseParseException):
    """
    Failed to parse a SQL command, thrown from Spark Connect.
    """


class IllegalArgumentException(SparkConnectGrpcException, BaseIllegalArgumentException):
    """
    Passed an illegal or inappropriate argument, thrown from Spark Connect.
    """


class StreamingQueryException(SparkConnectGrpcException, BaseStreamingQueryException):
    """
    Exception that stopped a :class:`StreamingQuery` thrown from Spark Connect.
    """


class QueryExecutionException(SparkConnectGrpcException, BaseQueryExecutionException):
    """
    Failed to execute a query, thrown from Spark Connect.
    """


class PythonException(SparkConnectGrpcException, BasePythonException):
    """
    Exceptions thrown from Spark Connect.
    """


class ArithmeticException(SparkConnectGrpcException, BaseArithmeticException):
    """
    Arithmetic exception thrown from Spark Connect.
    """


class UnsupportedOperationException(SparkConnectGrpcException, BaseUnsupportedOperationException):
    """
    Unsupported operation exception thrown from Spark Connect.
    """


class ArrayIndexOutOfBoundsException(SparkConnectGrpcException, BaseArrayIndexOutOfBoundsException):
    """
    Array index out of bounds exception thrown from Spark Connect.
    """


class DateTimeException(SparkConnectGrpcException, BaseDateTimeException):
    """
    Datetime exception thrown from Spark Connect.
    """


class NumberFormatException(IllegalArgumentException, BaseNumberFormatException):
    """
    Number format exception thrown from Spark Connect.
    """


class SparkRuntimeException(SparkConnectGrpcException, BaseSparkRuntimeException):
    """
    Runtime exception thrown from Spark Connect.
    """


class SparkUpgradeException(SparkConnectGrpcException, BaseSparkUpgradeException):
    """
    Exception thrown because of Spark upgrade from Spark Connect.
    """


class SparkException(SparkConnectGrpcException):
    """ """


class SparkNoSuchElementException(SparkConnectGrpcException, BaseNoSuchElementException):
    """
    No such element exception.
    """


class InvalidPlanInput(SparkConnectGrpcException):
    """
    Error thrown when a connect plan is not valid.
    """


class StreamingPythonRunnerInitializationException(
    SparkConnectGrpcException, BaseStreamingPythonRunnerInitException
):
    """
    Failed to initialize a streaming Python runner.
    """


class PickleException(SparkConnectGrpcException, BasePickleException):
    """
    Represents an exception which is failed while pickling from server side
    such as `net.razorvine.pickle.PickleException`. This is different from `PySparkPicklingError`
    which represents an exception failed from Python built-in `pickle.PicklingError`.
    """


# Update EXCEPTION_CLASS_MAPPING here when adding a new exception
EXCEPTION_CLASS_MAPPING = {
    "org.apache.spark.sql.catalyst.parser.ParseException": ParseException,
    "org.apache.spark.sql.AnalysisException": AnalysisException,
    "org.apache.spark.sql.streaming.StreamingQueryException": StreamingQueryException,
    "org.apache.spark.sql.execution.QueryExecutionException": QueryExecutionException,
    "java.lang.NumberFormatException": NumberFormatException,
    "java.lang.IllegalArgumentException": IllegalArgumentException,
    "java.lang.ArithmeticException": ArithmeticException,
    "java.lang.UnsupportedOperationException": UnsupportedOperationException,
    "java.lang.ArrayIndexOutOfBoundsException": ArrayIndexOutOfBoundsException,
    "java.time.DateTimeException": DateTimeException,
    "org.apache.spark.SparkRuntimeException": SparkRuntimeException,
    "org.apache.spark.SparkUpgradeException": SparkUpgradeException,
    "org.apache.spark.api.python.PythonException": PythonException,
    "org.apache.spark.SparkNoSuchElementException": SparkNoSuchElementException,
    "org.apache.spark.SparkException": SparkException,
    "org.apache.spark.sql.connect.common.InvalidPlanInput": InvalidPlanInput,
    "org.apache.spark.api.python.StreamingPythonRunner"
    "$StreamingPythonRunnerInitializationException": StreamingPythonRunnerInitializationException,
}

THIRD_PARTY_EXCEPTION_CLASS_MAPPING = {
    "net.razorvine.pickle.PickleException": PickleException,
}


class SQLQueryContext(BaseQueryContext):
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
        raise UnsupportedOperationException(
            "",
            errorClass="UNSUPPORTED_CALL.WITHOUT_SUGGESTION",
            messageParameters={"className": "SQLQueryContext", "methodName": "callSite"},
            sql_state="0A000",
            server_stacktrace=None,
            display_server_stacktrace=False,
            contexts=[],
        )

    def summary(self) -> str:
        return str(self._q.summary)


class DataFrameQueryContext(BaseQueryContext):
    def __init__(self, q: Any):
        self._q = q

    def contextType(self) -> QueryContextType:
        return QueryContextType.DataFrame

    def objectType(self) -> str:
        raise UnsupportedOperationException(
            "",
            errorClass="UNSUPPORTED_CALL.WITHOUT_SUGGESTION",
            messageParameters={"className": "DataFrameQueryContext", "methodName": "objectType"},
            sql_state="0A000",
            server_stacktrace=None,
            display_server_stacktrace=False,
            contexts=[],
        )

    def objectName(self) -> str:
        raise UnsupportedOperationException(
            "",
            errorClass="UNSUPPORTED_CALL.WITHOUT_SUGGESTION",
            messageParameters={"className": "DataFrameQueryContext", "methodName": "objectName"},
            sql_state="0A000",
            server_stacktrace=None,
            display_server_stacktrace=False,
            contexts=[],
        )

    def startIndex(self) -> int:
        raise UnsupportedOperationException(
            "",
            errorClass="UNSUPPORTED_CALL.WITHOUT_SUGGESTION",
            messageParameters={"className": "DataFrameQueryContext", "methodName": "startIndex"},
            sql_state="0A000",
            server_stacktrace=None,
            display_server_stacktrace=False,
            contexts=[],
        )

    def stopIndex(self) -> int:
        raise UnsupportedOperationException(
            "",
            errorClass="UNSUPPORTED_CALL.WITHOUT_SUGGESTION",
            messageParameters={"className": "DataFrameQueryContext", "methodName": "stopIndex"},
            sql_state="0A000",
            server_stacktrace=None,
            display_server_stacktrace=False,
            contexts=[],
        )

    def fragment(self) -> str:
        return str(self._q.fragment)

    def callSite(self) -> str:
        return str(self._q.call_site)

    def summary(self) -> str:
        return str(self._q.summary)


def _status_code(code: int) -> StatusCode:
    """The ``StatusCode`` of a gRPC status code number, or ``UNKNOWN`` for another number."""
    for status_code in StatusCode:
        if status_code.value[0] == code:
            return status_code
    return StatusCode.UNKNOWN


def _convert_server_error(
    classes: List[str],
    sql_state: Optional[str],
    error_class: Optional[str],
    reason: Optional[str],
    message: str,
    message_parameters: Optional[Dict[str, str]],
    stacktrace: Optional[str],
    display_server_stacktrace: bool,
    contexts: Optional[List[Dict[str, Any]]],
    grpc_status_code: int,
    breaking_change_info: Optional[Dict[str, Any]],
) -> SparkConnectException:
    """The exception for an error that the server raised, from what the Rust transport read of
    its ``ErrorInfo`` and ``FetchErrorDetails``. Picks and builds it as upstream
    ``_convert_exception`` does once it has read them."""
    grpc_status_code = _status_code(grpc_status_code)  # type: ignore[assignment]
    if contexts is not None:
        contexts = [  # type: ignore[misc]
            (SQLQueryContext if c["context_type"] == "SQL" else DataFrameQueryContext)(
                SimpleNamespace(**c)
            )
            for c in contexts
        ]

    if "org.apache.spark.api.python.PythonException" in classes:
        converted: SparkConnectException = PythonException(
            message="\n  An exception was thrown from the Python worker. "
            "Please see the stack trace below.\n%s" % message,
            grpc_status_code=grpc_status_code,
        )
        return recover_python_exception(converted)

    # Return exception based on class mapping
    for error_class_name in classes:
        ExceptionClass = EXCEPTION_CLASS_MAPPING.get(error_class_name)
        if ExceptionClass is SparkException:
            for third_party_exception_class in THIRD_PARTY_EXCEPTION_CLASS_MAPPING:
                ExceptionClass = (
                    THIRD_PARTY_EXCEPTION_CLASS_MAPPING.get(third_party_exception_class)
                    if third_party_exception_class in message
                    else SparkException
                )

        if ExceptionClass:
            converted = ExceptionClass(
                message,
                errorClass=error_class,
                messageParameters=message_parameters,
                sql_state=sql_state,
                server_stacktrace=stacktrace,
                display_server_stacktrace=display_server_stacktrace,
                contexts=contexts,  # type: ignore[arg-type]
                grpc_status_code=grpc_status_code,
                breaking_change_info=breaking_change_info,
            )
            return recover_python_exception(converted)

    # Return UnknownException if there is no matched exception class
    converted = UnknownException(
        message,
        reason=reason,
        messageParameters=message_parameters,
        errorClass=error_class,
        sql_state=sql_state,
        server_stacktrace=stacktrace,
        display_server_stacktrace=display_server_stacktrace,
        contexts=contexts,  # type: ignore[arg-type]
        grpc_status_code=grpc_status_code,
        breaking_change_info=breaking_change_info,
    )
    return recover_python_exception(converted)
