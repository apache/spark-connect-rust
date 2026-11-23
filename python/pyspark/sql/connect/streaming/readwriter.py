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

# The streaming readers and writers of Spark Connect, as PySpark Connect's
# ``pyspark.sql.connect.streaming.readwriter``.
#
# ``DataStreamReader`` is adapted from PySpark Connect's, of Apache Spark v4.3.0
# (python/pyspark/sql/connect/streaming/readwriter.py), whose state is the ``__dict__`` of the
# reader, as PySpark's: its code is PySpark's, but for the plans of the reads, which are the native
# plans of the Rust core (``_DataSource``, ``_Read`` and ``_RelationChanges``), which check their
# arguments as PySpark's plans do when they are made, and ``_df``, which makes the DataFrame of the
# core of a plan, of the session of the reader's ``_client``. The writer and its ``Trigger`` are
# native. ``DataStreamWriter.foreachBatch`` and ``.foreach`` take a Python callable and cloudpickle
# it inside the extension (with ``pyspark.serializers.CloudPickleSerializer``); ``.foreach`` sends
# the function of ``_construct_foreach_function``, as PySpark does.

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

# As PySpark Connect's module, which imports pyspark.sql.connect.readwriter: the official doctests
# of the stream reader and writer take that module's namespace as their globals.
import pyspark.sql.connect.readwriter  # noqa: F401
from pyspark._pyspark import (
    DataStreamWriter as _DataStreamWriter,
)
from pyspark._pyspark import (
    Trigger as _Trigger,
)
from pyspark._pyspark import _DataSource, _Read, _read_dataframe, _RelationChanges
from pyspark.errors import (
    AnalysisException,
    PySparkAttributeError,
    PySparkRuntimeError,
    PySparkTypeError,
    PySparkValueError,
)
from pyspark.sql.connect.plan import LogicalPlan
from pyspark.sql.connect.readwriter import OptionUtils, to_str
from pyspark.sql.streaming.readwriter import DataStreamReader as PySparkDataStreamReader
from pyspark.sql.types import StructType

if TYPE_CHECKING:
    from pyspark.sql.connect._typing import OptionalPrimitiveType
    from pyspark.sql.connect.dataframe import DataFrame
    from pyspark.sql.connect.session import SparkSession


class DataStreamReader(OptionUtils):
    def __init__(self, client: "SparkSession") -> None:
        self._format: Optional[str] = None
        self._schema = ""
        self._client = client
        self._options: dict[str, str] = {}
        self._source_name: Optional[str] = None

    def _df(self, plan: LogicalPlan) -> "DataFrame":
        # The DataFrame of the plan, of the session of the client, of the Rust core.
        return _read_dataframe(plan, self._client)

    def format(self, source: str) -> "DataStreamReader":
        self._format = source
        return self

    format.__doc__ = PySparkDataStreamReader.format.__doc__

    def schema(self, schema: Union[StructType, str]) -> "DataStreamReader":
        if isinstance(schema, StructType):
            self._schema = schema.json()
        elif isinstance(schema, str):
            self._schema = schema
        else:
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "arg_name": "schema",
                    "expected_type": "str or struct type",
                    "arg_type": type(schema).__name__,
                },
            )
        return self

    schema.__doc__ = PySparkDataStreamReader.schema.__doc__

    def option(self, key: str, value: "OptionalPrimitiveType") -> "DataStreamReader":
        if value is None:
            return self
        self._options[key] = str(value)
        return self

    option.__doc__ = PySparkDataStreamReader.option.__doc__

    def options(self, **options: "OptionalPrimitiveType") -> "DataStreamReader":
        for k in options:
            self.option(k, to_str(options[k]))
        return self

    options.__doc__ = PySparkDataStreamReader.options.__doc__

    def name(self, source_name: str) -> "DataStreamReader":
        if not isinstance(source_name, str):
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "arg_name": "source_name",
                    "expected_type": "str",
                    "arg_type": type(source_name).__name__,
                },
            )

        # Validate that source_name contains only ASCII letters, digits, and underscores
        if not re.match(r"^[a-zA-Z0-9_]+$", source_name):
            raise PySparkValueError(
                errorClass="INVALID_STREAMING_SOURCE_NAME",
                messageParameters={"source_name": source_name},
            )

        self._source_name = source_name
        return self

    name.__doc__ = PySparkDataStreamReader.name.__doc__

    def load(
        self,
        path: Optional[str] = None,
        format: Optional[str] = None,
        schema: Optional[Union[StructType, str]] = None,
        **options: "OptionalPrimitiveType",
    ) -> "DataFrame":
        if format is not None:
            self.format(format)
        if schema is not None:
            self.schema(schema)
        self.options(**options)
        if path is not None and (not isinstance(path, str) or len(path.strip()) == 0):
            raise PySparkValueError(
                errorClass="VALUE_NOT_NON_EMPTY_STR",
                messageParameters={"arg_name": "path", "arg_value": str(path)},
            )

        plan = _DataSource(
            format=self._format,
            schema=self._schema,
            options=self._options,
            paths=[path] if path else None,
            is_streaming=True,
            source_name=self._source_name,
        )

        return self._df(plan)

    load.__doc__ = PySparkDataStreamReader.load.__doc__

    def json(
        self,
        path: str,
        schema: Optional[Union[StructType, str]] = None,
        primitivesAsString: Optional[Union[bool, str]] = None,
        prefersDecimal: Optional[Union[bool, str]] = None,
        allowComments: Optional[Union[bool, str]] = None,
        allowUnquotedFieldNames: Optional[Union[bool, str]] = None,
        allowSingleQuotes: Optional[Union[bool, str]] = None,
        allowNumericLeadingZero: Optional[Union[bool, str]] = None,
        allowBackslashEscapingAnyCharacter: Optional[Union[bool, str]] = None,
        mode: Optional[str] = None,
        columnNameOfCorruptRecord: Optional[str] = None,
        dateFormat: Optional[str] = None,
        timestampFormat: Optional[str] = None,
        multiLine: Optional[Union[bool, str]] = None,
        allowUnquotedControlChars: Optional[Union[bool, str]] = None,
        lineSep: Optional[str] = None,
        locale: Optional[str] = None,
        dropFieldIfAllNull: Optional[Union[bool, str]] = None,
        encoding: Optional[str] = None,
        pathGlobFilter: Optional[Union[bool, str]] = None,
        recursiveFileLookup: Optional[Union[bool, str]] = None,
        allowNonNumericNumbers: Optional[Union[bool, str]] = None,
        useUnsafeRow: Optional[Union[bool, str]] = None,
    ) -> "DataFrame":
        self._set_opts(
            schema=schema,
            primitivesAsString=primitivesAsString,
            prefersDecimal=prefersDecimal,
            allowComments=allowComments,
            allowUnquotedFieldNames=allowUnquotedFieldNames,
            allowSingleQuotes=allowSingleQuotes,
            allowNumericLeadingZero=allowNumericLeadingZero,
            allowBackslashEscapingAnyCharacter=allowBackslashEscapingAnyCharacter,
            mode=mode,
            columnNameOfCorruptRecord=columnNameOfCorruptRecord,
            dateFormat=dateFormat,
            timestampFormat=timestampFormat,
            multiLine=multiLine,
            allowUnquotedControlChars=allowUnquotedControlChars,
            lineSep=lineSep,
            locale=locale,
            dropFieldIfAllNull=dropFieldIfAllNull,
            encoding=encoding,
            pathGlobFilter=pathGlobFilter,
            recursiveFileLookup=recursiveFileLookup,
            allowNonNumericNumbers=allowNonNumericNumbers,
            useUnsafeRow=useUnsafeRow,
        )
        if isinstance(path, str):
            return self.load(path=path, format="json")
        else:
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "arg_name": "path",
                    "expected_type": "str",
                    "arg_type": type(path).__name__,
                },
            )

    json.__doc__ = PySparkDataStreamReader.json.__doc__

    def orc(
        self,
        path: str,
        mergeSchema: Optional[bool] = None,
        pathGlobFilter: Optional[Union[bool, str]] = None,
        recursiveFileLookup: Optional[Union[bool, str]] = None,
    ) -> "DataFrame":
        self._set_opts(
            mergeSchema=mergeSchema,
            pathGlobFilter=pathGlobFilter,
            recursiveFileLookup=recursiveFileLookup,
        )
        if isinstance(path, str):
            return self.load(path=path, format="orc")
        else:
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "arg_name": "path",
                    "expected_type": "str",
                    "arg_type": type(path).__name__,
                },
            )

    orc.__doc__ = PySparkDataStreamReader.orc.__doc__

    def parquet(
        self,
        path: str,
        mergeSchema: Optional[bool] = None,
        pathGlobFilter: Optional[Union[bool, str]] = None,
        recursiveFileLookup: Optional[Union[bool, str]] = None,
        datetimeRebaseMode: Optional[Union[bool, str]] = None,
        int96RebaseMode: Optional[Union[bool, str]] = None,
    ) -> "DataFrame":
        self._set_opts(
            mergeSchema=mergeSchema,
            pathGlobFilter=pathGlobFilter,
            recursiveFileLookup=recursiveFileLookup,
            datetimeRebaseMode=datetimeRebaseMode,
            int96RebaseMode=int96RebaseMode,
        )
        self._set_opts(
            mergeSchema=mergeSchema,
            pathGlobFilter=pathGlobFilter,
            recursiveFileLookup=recursiveFileLookup,
            datetimeRebaseMode=datetimeRebaseMode,
            int96RebaseMode=int96RebaseMode,
        )
        if isinstance(path, str):
            return self.load(path=path, format="parquet")
        else:
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "arg_name": "path",
                    "expected_type": "str",
                    "arg_type": type(path).__name__,
                },
            )

    parquet.__doc__ = PySparkDataStreamReader.parquet.__doc__

    def text(
        self,
        path: str,
        wholetext: bool = False,
        lineSep: Optional[str] = None,
        pathGlobFilter: Optional[Union[bool, str]] = None,
        recursiveFileLookup: Optional[Union[bool, str]] = None,
    ) -> "DataFrame":
        self._set_opts(
            wholetext=wholetext,
            lineSep=lineSep,
            pathGlobFilter=pathGlobFilter,
            recursiveFileLookup=recursiveFileLookup,
        )
        if isinstance(path, str):
            return self.load(path=path, format="text")
        else:
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "arg_name": "path",
                    "expected_type": "str",
                    "arg_type": type(path).__name__,
                },
            )

    text.__doc__ = PySparkDataStreamReader.text.__doc__

    def csv(
        self,
        path: str,
        schema: Optional[Union[StructType, str]] = None,
        sep: Optional[str] = None,
        encoding: Optional[str] = None,
        quote: Optional[str] = None,
        escape: Optional[str] = None,
        comment: Optional[str] = None,
        header: Optional[Union[bool, str]] = None,
        inferSchema: Optional[Union[bool, str]] = None,
        ignoreLeadingWhiteSpace: Optional[Union[bool, str]] = None,
        ignoreTrailingWhiteSpace: Optional[Union[bool, str]] = None,
        nullValue: Optional[str] = None,
        nanValue: Optional[str] = None,
        positiveInf: Optional[str] = None,
        negativeInf: Optional[str] = None,
        dateFormat: Optional[str] = None,
        timestampFormat: Optional[str] = None,
        maxColumns: Optional[Union[int, str]] = None,
        maxCharsPerColumn: Optional[Union[int, str]] = None,
        maxMalformedLogPerPartition: Optional[Union[int, str]] = None,
        mode: Optional[str] = None,
        columnNameOfCorruptRecord: Optional[str] = None,
        multiLine: Optional[Union[bool, str]] = None,
        charToEscapeQuoteEscaping: Optional[Union[bool, str]] = None,
        enforceSchema: Optional[Union[bool, str]] = None,
        emptyValue: Optional[str] = None,
        locale: Optional[str] = None,
        lineSep: Optional[str] = None,
        pathGlobFilter: Optional[Union[bool, str]] = None,
        recursiveFileLookup: Optional[Union[bool, str]] = None,
        unescapedQuoteHandling: Optional[str] = None,
    ) -> "DataFrame":
        self._set_opts(
            schema=schema,
            sep=sep,
            encoding=encoding,
            quote=quote,
            escape=escape,
            comment=comment,
            header=header,
            inferSchema=inferSchema,
            ignoreLeadingWhiteSpace=ignoreLeadingWhiteSpace,
            ignoreTrailingWhiteSpace=ignoreTrailingWhiteSpace,
            nullValue=nullValue,
            nanValue=nanValue,
            positiveInf=positiveInf,
            negativeInf=negativeInf,
            dateFormat=dateFormat,
            timestampFormat=timestampFormat,
            maxColumns=maxColumns,
            maxCharsPerColumn=maxCharsPerColumn,
            maxMalformedLogPerPartition=maxMalformedLogPerPartition,
            mode=mode,
            columnNameOfCorruptRecord=columnNameOfCorruptRecord,
            multiLine=multiLine,
            charToEscapeQuoteEscaping=charToEscapeQuoteEscaping,
            enforceSchema=enforceSchema,
            emptyValue=emptyValue,
            locale=locale,
            lineSep=lineSep,
            pathGlobFilter=pathGlobFilter,
            recursiveFileLookup=recursiveFileLookup,
            unescapedQuoteHandling=unescapedQuoteHandling,
        )
        if isinstance(path, str):
            return self.load(path=path, format="csv")
        else:
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "arg_name": "path",
                    "expected_type": "str",
                    "arg_type": type(path).__name__,
                },
            )

    csv.__doc__ = PySparkDataStreamReader.csv.__doc__

    def xml(
        self,
        path: str,
        rowTag: Optional[str] = None,
        schema: Optional[Union[StructType, str]] = None,
        excludeAttribute: Optional[Union[bool, str]] = None,
        attributePrefix: Optional[str] = None,
        valueTag: Optional[str] = None,
        ignoreSurroundingSpaces: Optional[Union[bool, str]] = None,
        rowValidationXSDPath: Optional[str] = None,
        ignoreNamespace: Optional[Union[bool, str]] = None,
        wildcardColName: Optional[str] = None,
        encoding: Optional[str] = None,
        inferSchema: Optional[Union[bool, str]] = None,
        nullValue: Optional[str] = None,
        dateFormat: Optional[str] = None,
        timestampFormat: Optional[str] = None,
        mode: Optional[str] = None,
        columnNameOfCorruptRecord: Optional[str] = None,
        multiLine: Optional[Union[bool, str]] = None,
        samplingRatio: Optional[Union[float, str]] = None,
        locale: Optional[str] = None,
    ) -> "DataFrame":
        self._set_opts(
            rowTag=rowTag,
            schema=schema,
            excludeAttribute=excludeAttribute,
            attributePrefix=attributePrefix,
            valueTag=valueTag,
            ignoreSurroundingSpaces=ignoreSurroundingSpaces,
            rowValidationXSDPath=rowValidationXSDPath,
            ignoreNamespace=ignoreNamespace,
            wildcardColName=wildcardColName,
            encoding=encoding,
            inferSchema=inferSchema,
            nullValue=nullValue,
            dateFormat=dateFormat,
            timestampFormat=timestampFormat,
            mode=mode,
            columnNameOfCorruptRecord=columnNameOfCorruptRecord,
            multiLine=multiLine,
            samplingRatio=samplingRatio,
            locale=locale,
        )
        if isinstance(path, str):
            return self.load(path=path, format="xml")
        else:
            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "arg_name": "path",
                    "expected_type": "str",
                    "arg_type": type(path).__name__,
                },
            )

    xml.__doc__ = PySparkDataStreamReader.xml.__doc__

    def table(self, tableName: str) -> "DataFrame":
        return self._df(_Read(tableName, self._options, is_streaming=True))

    table.__doc__ = PySparkDataStreamReader.table.__doc__

    def changes(self, tableName: str) -> "DataFrame":
        if self._schema:
            raise AnalysisException(
                "User specified schema not supported with `changes`.",
                errorClass="_LEGACY_ERROR_TEMP_1189",
                messageParameters={"operation": "changes"},
            )
        return self._df(_RelationChanges(tableName, self._options, is_streaming=True))

    changes.__doc__ = PySparkDataStreamReader.changes.__doc__


DataStreamReader.__doc__ = PySparkDataStreamReader.__doc__


DataStreamWriter = _DataStreamWriter
Trigger = _Trigger


def _construct_foreach_function(
    f: Union[Callable[[Any], None], Any],
) -> Callable[[Any, Iterator], Iterator]:
    """
    The function of the partition and the rows of each partition that ``foreach(f)`` sends,
    which the Python worker of the server runs, as PySpark's
    ``DataStreamWriter._construct_foreach_function``: for a function ``f``, one that calls ``f``
    with each row, and for an object, one that calls its ``open(partition_id, epoch_id)``, its
    ``process(row)`` with each row, and its ``close(error)``.

    Ported from PySpark, but for ``TaskContext``, which the function imports on the worker.
    """
    if callable(f):
        # The provided object is a callable function that is supposed to be called on each row.
        # Construct a function that takes an iterator and calls the provided function on each
        # row.
        def func_without_process(_: Any, iterator: Iterator) -> Iterator:
            for x in iterator:
                f(x)
            return iter([])

        return func_without_process

    else:
        # The provided object is not a callable function. Then it is expected to have a
        # 'process(row)' method, and optional 'open(partition_id, epoch_id)' and
        # 'close(error)' methods.

        if not hasattr(f, "process"):
            raise PySparkAttributeError(
                errorClass="ATTRIBUTE_NOT_CALLABLE",
                messageParameters={"attr_name": "process", "obj_name": "f"},
            )

        if not callable(f.process):
            raise PySparkAttributeError(
                errorClass="ATTRIBUTE_NOT_CALLABLE",
                messageParameters={"attr_name": "process", "obj_name": "f"},
            )

        def doesMethodExist(method_name: str) -> bool:
            exists = hasattr(f, method_name)
            if exists and not callable(getattr(f, method_name)):
                raise PySparkAttributeError(
                    errorClass="ATTRIBUTE_NOT_CALLABLE",
                    messageParameters={"attr_name": method_name, "obj_name": "f"},
                )
            return exists

        open_exists = doesMethodExist("open")
        close_exists = doesMethodExist("close")

        def func_with_open_process_close(partition_id: Any, iterator: Iterator) -> Iterator:
            from pyspark.taskcontext import TaskContext

            epoch_id = cast(TaskContext, TaskContext.get()).getLocalProperty(
                "streaming.sql.batchId"
            )
            if epoch_id:
                int_epoch_id = int(epoch_id)
            else:
                raise PySparkRuntimeError(
                    errorClass="CANNOT_GET_BATCH_ID",
                    messageParameters={"obj_name": "TaskContext"},
                )

            # Check if the data should be processed
            should_process = True
            if open_exists:
                should_process = f.open(partition_id, int_epoch_id)

            error = None

            try:
                if should_process:
                    for x in iterator:
                        f.process(x)
            except Exception as ex:
                error = ex
            finally:
                if close_exists:
                    f.close(error)
                if error:
                    raise error

            return iter([])

        return func_with_open_process_close

__all__ = ["DataStreamReader", "DataStreamWriter"]
