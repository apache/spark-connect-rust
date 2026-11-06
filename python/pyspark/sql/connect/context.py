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
"""Spark Connect SQLContext/HiveContext, ported from the v4.3.0
``pyspark/sql/connect/context.py``. Fork-adapted (not vendored): it wraps the Rust-backed
:class:`SparkSession` and imports from the drop-in's own modules instead of the
gRPC-client internals."""
import warnings
from typing import Any, Callable, ClassVar, List, Optional, Union

from pyspark._globals import _NoValue, _NoValueType
from pyspark.errors import PySparkNotImplementedError
from pyspark.sql.types import BooleanType, DataType, StringType, StructField, StructType

__all__ = ["SQLContext", "HiveContext"]


class SQLContext:
    """The entry point for working with structured data (rows and columns) in Spark, in Spark 1.x.

    As of Spark 2.0, this is replaced by :class:`SparkSession`. This Spark Connect
    implementation wraps a :class:`SparkSession` directly and needs no ``SparkContext``.

    .. deprecated:: 4.3.0
        Use :func:`SparkSession.builder.getOrCreate()` instead.
    """

    _instantiatedContext: ClassVar[Optional["SQLContext"]] = None

    def __init__(self, sparkSession: Any) -> None:
        warnings.warn(
            "Deprecated in 4.3.0. Use SparkSession.builder.getOrCreate() instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.sparkSession = sparkSession
        if type(self)._instantiatedContext is None:
            type(self)._instantiatedContext = self

    @classmethod
    def _from_session(cls, sparkSession: Any) -> "SQLContext":
        """Create a new instance without emitting a deprecation warning."""
        ctx = object.__new__(cls)
        ctx.sparkSession = sparkSession
        return ctx

    @classmethod
    def _get_or_create_from_session(cls, sparkSession: Any) -> "SQLContext":
        """Return the cached instance, or create one when the cached context wraps a
        different session (e.g. the previous one was stopped and a new one started)."""
        if cls._instantiatedContext is None or not _same_session(
            cls._instantiatedContext.sparkSession, sparkSession
        ):
            cls._instantiatedContext = cls._from_session(sparkSession)
        return cls._instantiatedContext

    def newSession(self) -> "SQLContext":
        """Returns a new SQLContext as a new session with empty state."""
        return self._from_session(self.sparkSession.newSession())

    def setConf(self, key: str, value: Union[bool, int, str]) -> None:
        """Sets the given Spark SQL configuration property."""
        self.sparkSession.conf.set(key, value)

    def getConf(
        self, key: str, defaultValue: Union[Optional[str], _NoValueType] = _NoValue
    ) -> Optional[str]:
        """Returns the value of Spark SQL configuration property for the given key."""
        if defaultValue is _NoValue:
            return self.sparkSession.conf.get(key)
        return self.sparkSession.conf.get(key, defaultValue)

    @property
    def udf(self) -> Any:
        """Returns a :class:`UDFRegistration` for UDF registration."""
        return self.sparkSession.udf

    @property
    def udtf(self) -> Any:
        """Returns a :class:`UDTFRegistration` for UDTF registration."""
        return self.sparkSession.udtf

    def range(
        self,
        start: int,
        end: Optional[int] = None,
        step: int = 1,
        numPartitions: Optional[int] = None,
    ) -> Any:
        """Create a :class:`DataFrame` with a single LongType column named ``id``."""
        return self.sparkSession.range(start, end, step, numPartitions)

    def registerFunction(
        self, name: str, f: Callable[..., Any], returnType: Optional[DataType] = None
    ) -> Any:
        """An alias for :func:`spark.udf.register`.

        .. deprecated:: 4.3.0
            Use :func:`spark.udf.register` instead.
        """
        warnings.warn("Deprecated in 4.3.0. Use spark.udf.register instead.", FutureWarning)
        return self.sparkSession.udf.register(name, f, returnType)

    def registerJavaFunction(
        self, name: str, javaClassName: str, returnType: Optional[DataType] = None
    ) -> None:
        """Not supported in Spark Connect."""
        raise PySparkNotImplementedError(
            errorClass="NOT_IMPLEMENTED",
            messageParameters={"feature": "registerJavaFunction"},
        )

    def createDataFrame(
        self,
        data: Any,
        schema: Any = None,
        samplingRatio: Optional[float] = None,
        verifySchema: Optional[bool] = None,
    ) -> Any:
        """Creates a :class:`DataFrame` from local data."""
        return self.sparkSession.createDataFrame(data, schema, samplingRatio, verifySchema)

    def registerDataFrameAsTable(self, df: Any, tableName: str) -> None:
        """Registers the given :class:`DataFrame` as a temporary table in the catalog."""
        df.createOrReplaceTempView(tableName)

    def dropTempTable(self, tableName: str) -> None:
        """Remove the temporary table from catalog."""
        self.sparkSession.catalog.dropTempView(tableName)

    def createExternalTable(
        self,
        tableName: str,
        path: Optional[str] = None,
        source: Optional[str] = None,
        schema: Optional[StructType] = None,
        **options: str,
    ) -> Any:
        """Creates an external table based on the dataset in a data source."""
        return self.sparkSession.catalog.createExternalTable(
            tableName, path, source, schema, **options
        )

    def sql(self, sqlQuery: str) -> Any:
        """Returns a :class:`DataFrame` representing the result of the given query."""
        return self.sparkSession.sql(sqlQuery)

    def table(self, tableName: str) -> Any:
        """Returns the specified table or view as a :class:`DataFrame`."""
        return self.sparkSession.table(tableName)

    def tables(self, dbName: Optional[str] = None) -> Any:
        """Returns a :class:`DataFrame` of (namespace, tableName, isTemporary) for the tables
        in the given (or current) database."""
        schema = StructType(
            [
                StructField("namespace", StringType(), nullable=True),
                StructField("tableName", StringType(), nullable=True),
                StructField("isTemporary", BooleanType(), nullable=False),
            ]
        )
        # catalog.listTables() rather than SHOW TABLES keeps the column names fixed, and the
        # full namespace is joined ("a.b") to match classic SHOW TABLES.
        rows = [
            (".".join(t.namespace) if t.namespace else "", t.name, t.isTemporary)
            for t in self.sparkSession.catalog.listTables(dbName)
        ]
        return self.sparkSession.createDataFrame(rows, schema)

    def tableNames(self, dbName: Optional[str] = None) -> List[str]:
        """Returns a list of names of tables in the database ``dbName``."""
        return [t.name for t in self.sparkSession.catalog.listTables(dbName)]

    def cacheTable(self, tableName: str) -> None:
        """Caches the specified table in-memory."""
        self.sparkSession.catalog.cacheTable(tableName)

    def uncacheTable(self, tableName: str) -> None:
        """Removes the specified table from the in-memory cache."""
        self.sparkSession.catalog.uncacheTable(tableName)

    def clearCache(self) -> None:
        """Removes all cached tables from the in-memory cache."""
        self.sparkSession.catalog.clearCache()

    @property
    def read(self) -> Any:
        """Returns a :class:`DataFrameReader`."""
        return self.sparkSession.read

    @property
    def readStream(self) -> Any:
        """Returns a :class:`DataStreamReader`."""
        return self.sparkSession.readStream

    @property
    def streams(self) -> Any:
        """Returns a :class:`StreamingQueryManager`."""
        return self.sparkSession.streams


class HiveContext(SQLContext):
    """Not supported in Spark Connect.

    .. deprecated:: 4.3.0
        Use SparkSession.builder.enableHiveSupport().getOrCreate().
    """

    # Override to prevent inheriting SQLContext's cached instance, which would cause
    # _get_or_create_from_session to skip _from_session and return the wrong type.
    _instantiatedContext: ClassVar[Optional["SQLContext"]] = None

    def __init__(self, sparkSession: Any) -> None:
        raise PySparkNotImplementedError(
            errorClass="NOT_IMPLEMENTED",
            messageParameters={"feature": "HiveContext"},
        )

    @classmethod
    def _from_session(cls, sparkSession: Any) -> "SQLContext":
        raise PySparkNotImplementedError(
            errorClass="NOT_IMPLEMENTED",
            messageParameters={"feature": "HiveContext"},
        )


def _same_session(a: Any, b: Any) -> bool:
    """Session identity. ``SparkSession.getActiveSession()`` returns a fresh Rust-backed
    wrapper per call, so fall back to comparing session ids when the objects differ."""
    if a is b:
        return True
    sid = getattr(a, "session_id", None)
    return sid is not None and sid == getattr(b, "session_id", None)


def _reset_instantiated_context(session_id: str) -> None:
    """Called from ``SparkSession.stop()``: drop the cached context if it wraps the stopped
    session. Compared by session id since the Rust-backed session objects are wrappers."""
    for cls in (SQLContext, HiveContext):
        ctx = cls.__dict__.get("_instantiatedContext")
        if ctx is not None and getattr(ctx.sparkSession, "session_id", None) == session_id:
            cls._instantiatedContext = None
