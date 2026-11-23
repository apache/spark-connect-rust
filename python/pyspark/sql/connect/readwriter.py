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
# DataFrameReader / DataFrameWriter(V2) of Spark Connect, as PySpark Connect's
# ``pyspark.sql.connect.readwriter``.
#
# ``OptionUtils`` and ``DataFrameReader`` are adapted from PySpark Connect's, of Apache Spark v4.3.0
# (python/pyspark/sql/connect/readwriter.py), whose state is the ``__dict__`` of the reader, as
# PySpark's: their code is PySpark's, but for the plans of the reads, which are the native plans of
# the Rust core (``_DataSource``, ``_Read``, ``_RelationChanges`` and ``_Parse``), which check their
# arguments as PySpark's plans do when they are made, and ``_df``, which makes the DataFrame of the
# core of a plan, of the session of the reader's ``_client``. The writers are native.

from typing import TYPE_CHECKING, Optional, Union, cast, overload

import pyspark.sql.connect.proto as proto
from pyspark._pyspark import (
    DataFrameWriter,
    DataFrameWriterV2,
    _DataSource,
    _Parse,
    _Read,
    _read_dataframe,
    _RelationChanges,
)
from pyspark.errors import (
    AnalysisException,
    PySparkAttributeError,
    PySparkTypeError,
)
from pyspark.sql.connect.plan import LogicalPlan
from pyspark.sql.readwriter import DataFrameReader as PySparkDataFrameReader

# PathOrPaths and TupleOrListOfString: PySpark Connect's aliases of the types of the paths and
# of the lists of column names, which are those of PySpark's classic pyspark.sql.readwriter.
from pyspark.sql.readwriter import PathOrPaths, TupleOrListOfString  # noqa: F401
from pyspark.sql.types import StructType
from pyspark.sql.utils import to_str

if TYPE_CHECKING:
    from pyspark.sql.connect._typing import OptionalPrimitiveType
    from pyspark.sql.connect.dataframe import DataFrame
    from pyspark.sql.connect.session import SparkSession

__all__ = ["DataFrameReader", "DataFrameWriter", "DataFrameWriterV2"]



class OptionUtils:
    def _set_opts(
        self,
        schema: Optional[Union[StructType, str]] = None,
        **options: "OptionalPrimitiveType",
    ) -> None:
        """
        Set named options (filter out those the value is None)
        """
        if schema is not None:
            self.schema(schema)  # type: ignore[attr-defined]
        for k, v in options.items():
            if v is not None:
                self.option(k, v)  # type: ignore[attr-defined]


class DataFrameReader(OptionUtils):
    # TODO(SPARK-40539) Achieve parity with PySpark.

    def __init__(self, client: "SparkSession"):
        self._client = client
        self._format: Optional[str] = None
        self._schema = ""
        self._options: dict[str, str] = {}

    def format(self, source: str) -> "DataFrameReader":
        self._format = source
        return self

    format.__doc__ = PySparkDataFrameReader.format.__doc__

    def schema(self, schema: Union[StructType, str]) -> "DataFrameReader":
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

    schema.__doc__ = PySparkDataFrameReader.schema.__doc__

    def option(self, key: str, value: "OptionalPrimitiveType") -> "DataFrameReader":
        self._options[key] = cast(str, to_str(value))
        return self

    option.__doc__ = PySparkDataFrameReader.option.__doc__

    def options(self, **options: "OptionalPrimitiveType") -> "DataFrameReader":
        for k in options:
            self.option(k, to_str(options[k]))
        return self

    options.__doc__ = PySparkDataFrameReader.options.__doc__

    def load(
        self,
        path: Optional[PathOrPaths] = None,
        format: Optional[str] = None,
        schema: Optional[Union[StructType, str]] = None,
        **options: "OptionalPrimitiveType",
    ) -> "DataFrame":
        if format is not None:
            self.format(format)
        if schema is not None:
            self.schema(schema)
        self.options(**options)

        paths = path
        if isinstance(path, str):
            paths = [path]

        plan = _DataSource(
            format=self._format,
            schema=self._schema,
            options=self._options,
            paths=paths,  # type: ignore[arg-type]
        )
        return self._df(plan)

    load.__doc__ = PySparkDataFrameReader.load.__doc__

    def _df(self, plan: LogicalPlan) -> "DataFrame":
        # The DataFrame of the plan, of the session of the client, of the Rust core.
        return _read_dataframe(plan, self._client)

    def table(self, tableName: str) -> "DataFrame":
        return self._df(_Read(tableName, self._options))

    table.__doc__ = PySparkDataFrameReader.table.__doc__

    def changes(self, tableName: str) -> "DataFrame":
        if self._schema:
            raise AnalysisException(
                "User specified schema not supported with `changes`.",
                errorClass="_LEGACY_ERROR_TEMP_1189",
                messageParameters={"operation": "changes"},
            )
        return self._df(_RelationChanges(tableName, self._options))

    changes.__doc__ = PySparkDataFrameReader.changes.__doc__

    def json(
        self,
        path: Union[PathOrPaths, "DataFrame"],
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
        samplingRatio: Optional[Union[float, str]] = None,
        dropFieldIfAllNull: Optional[Union[bool, str]] = None,
        encoding: Optional[str] = None,
        locale: Optional[str] = None,
        pathGlobFilter: Optional[Union[bool, str]] = None,
        recursiveFileLookup: Optional[Union[bool, str]] = None,
        modifiedBefore: Optional[Union[bool, str]] = None,
        modifiedAfter: Optional[Union[bool, str]] = None,
        allowNonNumericNumbers: Optional[Union[bool, str]] = None,
        useUnsafeRow: Optional[Union[bool, str]] = None,
    ) -> "DataFrame":
        self._set_opts(
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
            samplingRatio=samplingRatio,
            dropFieldIfAllNull=dropFieldIfAllNull,
            encoding=encoding,
            locale=locale,
            pathGlobFilter=pathGlobFilter,
            recursiveFileLookup=recursiveFileLookup,
            modifiedBefore=modifiedBefore,
            modifiedAfter=modifiedAfter,
            allowNonNumericNumbers=allowNonNumericNumbers,
            useUnsafeRow=useUnsafeRow,
        )
        if isinstance(path, str):
            path = [path]
        if isinstance(path, list):
            return self.load(path=path, format="json", schema=schema)

        from pyspark.sql.connect.dataframe import DataFrame

        if isinstance(path, DataFrame):
            # Schema must be set explicitly here because the DataFrame path
            # bypasses load(), which normally calls self.schema(schema).
            if schema is not None:
                self.schema(schema)
            return self._df(
                _Parse(
                    child=path._plan,
                    format=proto.Parse.ParseFormat.PARSE_FORMAT_JSON,
                    schema=self._schema,
                    options=self._options,
                )
            )
        raise PySparkTypeError(
            errorClass="NOT_EXPECTED_TYPE",
            messageParameters={
                "arg_name": "path",
                "expected_type": "str, list, or DataFrame",
                "arg_type": type(path).__name__,
            },
        )

    json.__doc__ = PySparkDataFrameReader.json.__doc__

    def parquet(self, *paths: str, **options: "OptionalPrimitiveType") -> "DataFrame":
        mergeSchema = options.get("mergeSchema", None)
        pathGlobFilter = options.get("pathGlobFilter", None)
        modifiedBefore = options.get("modifiedBefore", None)
        modifiedAfter = options.get("modifiedAfter", None)
        recursiveFileLookup = options.get("recursiveFileLookup", None)
        datetimeRebaseMode = options.get("datetimeRebaseMode", None)
        int96RebaseMode = options.get("int96RebaseMode", None)
        self._set_opts(
            mergeSchema=mergeSchema,
            pathGlobFilter=pathGlobFilter,
            recursiveFileLookup=recursiveFileLookup,
            modifiedBefore=modifiedBefore,
            modifiedAfter=modifiedAfter,
            datetimeRebaseMode=datetimeRebaseMode,
            int96RebaseMode=int96RebaseMode,
        )

        return self.load(path=list(paths), format="parquet")

    parquet.__doc__ = PySparkDataFrameReader.parquet.__doc__

    def text(
        self,
        paths: PathOrPaths,
        wholetext: bool = False,
        lineSep: Optional[str] = None,
        pathGlobFilter: Optional[Union[bool, str]] = None,
        recursiveFileLookup: Optional[Union[bool, str]] = None,
        modifiedBefore: Optional[Union[bool, str]] = None,
        modifiedAfter: Optional[Union[bool, str]] = None,
    ) -> "DataFrame":
        self._set_opts(
            wholetext=wholetext,
            lineSep=lineSep,
            pathGlobFilter=pathGlobFilter,
            recursiveFileLookup=recursiveFileLookup,
            modifiedBefore=modifiedBefore,
            modifiedAfter=modifiedAfter,
        )

        if isinstance(paths, str):
            paths = [paths]
        return self.load(path=paths, format="text")

    text.__doc__ = PySparkDataFrameReader.text.__doc__

    def csv(
        self,
        path: Union[PathOrPaths, "DataFrame"],
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
        charToEscapeQuoteEscaping: Optional[str] = None,
        samplingRatio: Optional[Union[float, str]] = None,
        enforceSchema: Optional[Union[bool, str]] = None,
        emptyValue: Optional[str] = None,
        locale: Optional[str] = None,
        lineSep: Optional[str] = None,
        pathGlobFilter: Optional[Union[bool, str]] = None,
        recursiveFileLookup: Optional[Union[bool, str]] = None,
        modifiedBefore: Optional[Union[bool, str]] = None,
        modifiedAfter: Optional[Union[bool, str]] = None,
        unescapedQuoteHandling: Optional[str] = None,
    ) -> "DataFrame":
        self._set_opts(
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
            samplingRatio=samplingRatio,
            enforceSchema=enforceSchema,
            emptyValue=emptyValue,
            locale=locale,
            lineSep=lineSep,
            pathGlobFilter=pathGlobFilter,
            recursiveFileLookup=recursiveFileLookup,
            modifiedBefore=modifiedBefore,
            modifiedAfter=modifiedAfter,
            unescapedQuoteHandling=unescapedQuoteHandling,
        )
        if isinstance(path, str):
            path = [path]

        from pyspark.sql.connect.dataframe import DataFrame

        if isinstance(path, DataFrame):
            # Schema must be set explicitly here because the DataFrame path
            # bypasses load(), which normally calls self.schema(schema).
            if schema is not None:
                self.schema(schema)
            return self._df(
                _Parse(
                    child=path._plan,
                    format=proto.Parse.ParseFormat.PARSE_FORMAT_CSV,
                    schema=self._schema,
                    options=self._options,
                )
            )
        return self.load(path=path, format="csv", schema=schema)

    csv.__doc__ = PySparkDataFrameReader.csv.__doc__

    def xml(
        self,
        path: Union[PathOrPaths, "DataFrame"],
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
            path = [path]
        if isinstance(path, list):
            return self.load(path=path, format="xml", schema=schema)

        from pyspark.sql.connect.dataframe import DataFrame

        if isinstance(path, DataFrame):
            # Schema must be set explicitly here because the DataFrame path
            # bypasses load(), which normally calls self.schema(schema).
            if schema is not None:
                self.schema(schema)
            return self._df(
                _Parse(
                    child=path._plan,
                    format=proto.Parse.ParseFormat.PARSE_FORMAT_XML,
                    schema=self._schema,
                    options=self._options,
                )
            )
        raise PySparkTypeError(
            errorClass="NOT_EXPECTED_TYPE",
            messageParameters={
                "arg_name": "path",
                "expected_type": "str, list, or DataFrame",
                "arg_type": type(path).__name__,
            },
        )

    xml.__doc__ = PySparkDataFrameReader.xml.__doc__

    def orc(
        self,
        path: PathOrPaths,
        mergeSchema: Optional[bool] = None,
        pathGlobFilter: Optional[Union[bool, str]] = None,
        recursiveFileLookup: Optional[Union[bool, str]] = None,
        modifiedBefore: Optional[Union[bool, str]] = None,
        modifiedAfter: Optional[Union[bool, str]] = None,
    ) -> "DataFrame":
        self._set_opts(
            mergeSchema=mergeSchema,
            pathGlobFilter=pathGlobFilter,
            modifiedBefore=modifiedBefore,
            modifiedAfter=modifiedAfter,
            recursiveFileLookup=recursiveFileLookup,
        )
        if isinstance(path, str):
            path = [path]
        return self.load(path=path, format="orc")

    orc.__doc__ = PySparkDataFrameReader.orc.__doc__

    @overload
    def jdbc(
        self, url: str, table: str, *, properties: Optional[dict[str, str]] = None
    ) -> "DataFrame": ...

    @overload
    def jdbc(
        self,
        url: str,
        table: str,
        column: str,
        lowerBound: Union[int, str],
        upperBound: Union[int, str],
        numPartitions: int,
        *,
        properties: Optional[dict[str, str]] = None,
    ) -> "DataFrame": ...

    @overload
    def jdbc(
        self,
        url: str,
        table: str,
        *,
        predicates: list[str],
        properties: Optional[dict[str, str]] = None,
    ) -> "DataFrame": ...

    def jdbc(
        self,
        url: str,
        table: str,
        column: Optional[str] = None,
        lowerBound: Optional[Union[int, str]] = None,
        upperBound: Optional[Union[int, str]] = None,
        numPartitions: Optional[int] = None,
        predicates: Optional[list[str]] = None,
        properties: Optional[dict[str, str]] = None,
    ) -> "DataFrame":
        if properties is None:
            properties = dict()

        self.format("jdbc")

        if column is not None:
            assert lowerBound is not None, "lowerBound can not be None when ``column`` is specified"
            assert upperBound is not None, "upperBound can not be None when ``column`` is specified"
            assert numPartitions is not None, (
                "numPartitions can not be None when ``column`` is specified"
            )
            self.options(
                partitionColumn=column,
                lowerBound=lowerBound,
                upperBound=upperBound,
                numPartitions=numPartitions,
            )
            self.options(**properties)
            self.options(url=url, dbtable=table)
            return self.load()
        else:
            self.options(**properties)
            self.options(url=url, dbtable=table)
            if predicates is not None:
                plan = _DataSource(
                    format=self._format,
                    schema=self._schema,
                    options=self._options,
                    predicates=predicates,
                )
                return self._df(plan)
            else:
                return self.load()

    jdbc.__doc__ = PySparkDataFrameReader.jdbc.__doc__

    @property
    def _jreader(self) -> None:
        raise PySparkAttributeError(
            errorClass="JVM_ATTRIBUTE_NOT_SUPPORTED", messageParameters={"attr_name": "_jreader"}
        )


DataFrameReader.__doc__ = PySparkDataFrameReader.__doc__
