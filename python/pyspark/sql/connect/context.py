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
# Spark Connect SQLContext/HiveContext, ported from the v4.3.0
# ``pyspark/sql/connect/context.py``. Fork-adapted (not vendored): it wraps the Rust-backed
# :class:`SparkSession` and imports from the drop-in's own modules instead of the
# gRPC-client internals.
import warnings
from typing import Any, Callable, ClassVar, List, Optional, Union

from pyspark._globals import _NoValue, _NoValueType
from pyspark.errors import PySparkNotImplementedError
from pyspark.sql.types import BooleanType, DataType, StringType, StructField, StructType

__all__ = ["SQLContext", "HiveContext"]


class SQLContext:
    """The entry point for working with structured data (rows and columns) in Spark, in Spark 1.x.

    As of Spark 2.0, this is replaced by :class:`SparkSession`. However, we are keeping the class
    here for backward compatibility.

    This is the Spark Connect-compatible implementation. Unlike the classic implementation,
    it wraps a Connect :class:`SparkSession` directly and does not require a
    :class:`~pyspark.SparkContext`.

    .. deprecated:: 4.3.0
        Use :func:`SparkSession.builder.getOrCreate()` instead.

    Parameters
    ----------
    sparkSession : :class:`SparkSession`
        The Connect :class:`SparkSession` to wrap.
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
        """Returns a new SQLContext as a new session, that has separate SQLConf,
        registered temporary views and UDFs, but shared table cache.

        .. versionadded:: 4.3.0

        Notes
        -----
        The returned session starts with empty state rather than inheriting this session's
        configuration, temporary views, or registered functions, matching the Scala Connect
        ``newSession()`` semantics. Unlike the classic
        :meth:`pyspark.sql.context.SQLContext.newSession`, configurations set through
        ``SparkSession.builder.config(...)`` are not reapplied to the new session.
        """
        return self._from_session(self.sparkSession.newSession())

    def setConf(self, key: str, value: Union[bool, int, str]) -> None:
        """Sets the given Spark SQL configuration property.

        .. versionadded:: 4.3.0
        """
        self.sparkSession.conf.set(key, value)

    def getConf(
        self, key: str, defaultValue: Union[Optional[str], _NoValueType] = _NoValue
    ) -> Optional[str]:
        """Returns the value of Spark SQL configuration property for the given key.

        If the key is not set and defaultValue is set, return
        defaultValue. If the key is not set and defaultValue is not set, return
        the system default value.

        .. versionadded:: 4.3.0
        """
        if defaultValue is _NoValue:
            return self.sparkSession.conf.get(key)
        return self.sparkSession.conf.get(key, defaultValue)

    @property
    def udf(self) -> Any:
        """Returns a :class:`UDFRegistration` for UDF registration.

        .. versionadded:: 4.3.0

        Returns
        -------
        :class:`UDFRegistration`
        """
        return self.sparkSession.udf

    @property
    def udtf(self) -> Any:
        """Returns a :class:`UDTFRegistration` for UDTF registration.

        .. versionadded:: 4.3.0

        Returns
        -------
        :class:`UDTFRegistration`
        """
        return self.sparkSession.udtf

    def range(
        self,
        start: int,
        end: Optional[int] = None,
        step: int = 1,
        numPartitions: Optional[int] = None,
    ) -> Any:
        """Create a :class:`DataFrame` with single :class:`~pyspark.sql.types.LongType` column
        named ``id``, containing elements in a range from ``start`` to ``end`` (exclusive) with
        step value ``step``.

        .. versionadded:: 4.3.0

        Parameters
        ----------
        start : int
            the start value
        end : int, optional
            the end value (exclusive)
        step : int, optional
            the incremental step (default: 1)
        numPartitions : int, optional
            the number of partitions of the DataFrame

        Returns
        -------
        :class:`DataFrame`
        """
        return self.sparkSession.range(start, end, step, numPartitions)

    def registerFunction(
        self, name: str, f: Callable[..., Any], returnType: Optional[DataType] = None
    ) -> Any:
        """An alias for :func:`spark.udf.register`.
        See :meth:`pyspark.sql.UDFRegistration.register`.

        .. versionadded:: 4.3.0

        .. deprecated:: 4.3.0
            Use :func:`spark.udf.register` instead.
        """
        warnings.warn("Deprecated in 4.3.0. Use spark.udf.register instead.", FutureWarning)
        return self.sparkSession.udf.register(name, f, returnType)

    def registerJavaFunction(
        self, name: str, javaClassName: str, returnType: Optional[DataType] = None
    ) -> None:
        """Not supported in Spark Connect.

        .. versionadded:: 4.3.0
        """
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
        """Creates a :class:`DataFrame` from an iterable, a :class:`pandas.DataFrame`,
        or a :class:`pyarrow.Table`.

        .. versionadded:: 4.3.0

        Parameters
        ----------
        data : iterable
            an iterable of any kind of SQL data representation (:class:`Row`,
            :class:`tuple`, ``int``, ``boolean``, etc.), :class:`list`,
            :class:`pandas.DataFrame`, or :class:`pyarrow.Table`.
        schema : :class:`~pyspark.sql.types.DataType`, str or list, optional
            a :class:`~pyspark.sql.types.DataType` or a datatype string or a list/tuple of
            column names.
        samplingRatio : float, optional
            the sample ratio of rows used for inferring the schema.
        verifySchema : bool, optional
            verify data types of every row against schema.

        Returns
        -------
        :class:`DataFrame`
        """
        return self.sparkSession.createDataFrame(data, schema, samplingRatio, verifySchema)

    def registerDataFrameAsTable(self, df: Any, tableName: str) -> None:
        """Registers the given :class:`DataFrame` as a temporary table in the catalog.

        Temporary tables exist only during the lifetime of this instance of :class:`SQLContext`.

        .. versionadded:: 4.3.0
        """
        df.createOrReplaceTempView(tableName)

    def dropTempTable(self, tableName: str) -> None:
        """Remove the temporary table from catalog.

        .. versionadded:: 4.3.0
        """
        self.sparkSession.catalog.dropTempView(tableName)

    def createExternalTable(
        self,
        tableName: str,
        path: Optional[str] = None,
        source: Optional[str] = None,
        schema: Optional[StructType] = None,
        **options: str,
    ) -> Any:
        """Creates an external table based on the dataset in a data source.

        It returns the DataFrame associated with the external table.

        The data source is specified by the ``source`` and a set of ``options``.
        If ``source`` is not specified, the default data source configured by
        ``spark.sql.sources.default`` will be used.

        Optionally, a schema can be provided as the schema of the returned :class:`DataFrame` and
        created external table.

        .. versionadded:: 4.3.0

        Returns
        -------
        :class:`DataFrame`
        """
        return self.sparkSession.catalog.createExternalTable(
            tableName, path, source, schema, **options
        )

    def sql(self, sqlQuery: str) -> Any:
        """Returns a :class:`DataFrame` representing the result of the given query.

        .. versionadded:: 4.3.0

        Returns
        -------
        :class:`DataFrame`
        """
        return self.sparkSession.sql(sqlQuery)

    def table(self, tableName: str) -> Any:
        """Returns the specified table or view as a :class:`DataFrame`.

        .. versionadded:: 4.3.0

        Returns
        -------
        :class:`DataFrame`
        """
        return self.sparkSession.table(tableName)

    def tables(self, dbName: Optional[str] = None) -> Any:
        """Returns a :class:`DataFrame` containing names of tables in the given database.

        If ``dbName`` is not specified, the current database will be used.

        The returned DataFrame has three columns: ``namespace``, ``tableName`` and
        ``isTemporary`` (a column with :class:`~pyspark.sql.types.BooleanType` indicating if a
        table is a temporary one or not).

        .. versionadded:: 4.3.0

        Parameters
        ----------
        dbName: str, optional
            name of the database to use.

        Returns
        -------
        :class:`DataFrame`
        """
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
        """Returns a list of names of tables in the database ``dbName``.

        .. versionadded:: 4.3.0

        Parameters
        ----------
        dbName: str
            name of the database to use. Default to the current database.

        Returns
        -------
        list
            list of table names as strings
        """
        return [t.name for t in self.sparkSession.catalog.listTables(dbName)]

    def cacheTable(self, tableName: str) -> None:
        """Caches the specified table in-memory.

        .. versionadded:: 4.3.0
        """
        self.sparkSession.catalog.cacheTable(tableName)

    def uncacheTable(self, tableName: str) -> None:
        """Removes the specified table from the in-memory cache.

        .. versionadded:: 4.3.0
        """
        self.sparkSession.catalog.uncacheTable(tableName)

    def clearCache(self) -> None:
        """Removes all cached tables from the in-memory cache.

        .. versionadded:: 4.3.0
        """
        self.sparkSession.catalog.clearCache()

    @property
    def read(self) -> Any:
        """Returns a :class:`DataFrameReader` that can be used to read data
        in as a :class:`DataFrame`.

        .. versionadded:: 4.3.0

        Returns
        -------
        :class:`DataFrameReader`
        """
        return self.sparkSession.read

    @property
    def readStream(self) -> Any:
        """Returns a :class:`DataStreamReader` that can be used to read data streams
        as a streaming :class:`DataFrame`.

        .. versionadded:: 4.3.0

        Notes
        -----
        This API is evolving.

        Returns
        -------
        :class:`DataStreamReader`
        """
        return self.sparkSession.readStream

    @property
    def streams(self) -> Any:
        """Returns a :class:`StreamingQueryManager` that allows managing all the
        :class:`~pyspark.sql.streaming.StreamingQuery` instances active on this
        context.

        .. versionadded:: 4.3.0

        Notes
        -----
        This API is evolving.
        """
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
