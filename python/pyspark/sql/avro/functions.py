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
# Avro functions, mirroring ``pyspark.sql.avro.functions``.
"""
A collections of builtin avro functions
"""

from pyspark.errors import PySparkTypeError
from pyspark.sql.column import Column
from pyspark.sql.connect.functions.builtin import (
    _call_function,
    _dict_to_options_column,
    _invoke_function,
    _to_col,
)
from pyspark.sql.connect.functions.builtin import (
    lit as _lit,
)


def _check_data(data):
    if not isinstance(data, (Column, str)):
        raise PySparkTypeError(
            errorClass="INVALID_TYPE",
            messageParameters={"arg_name": "data", "arg_type": "pyspark.sql.Column or str"},
        )


def from_avro(data, jsonFormatSchema, options=None):
    """
    Converts a binary column of Avro format into its corresponding catalyst value.
    The specified schema must match the read data, otherwise the behavior is undefined:
    it may fail or return arbitrary result.
    To deserialize the data with a compatible and evolved schema, the expected Avro schema can be
    set via the option avroSchema.

    .. versionadded:: 3.0.0

    .. versionchanged:: 3.5.0
        Supports Spark Connect.

    Parameters
    ----------
    data : :class:`~pyspark.sql.Column` or str
        the binary column.
    jsonFormatSchema : str
        the avro schema in JSON string format.
    options : dict, optional
        options to control how the Avro record is parsed.

    Notes
    -----
    Avro is built-in but external data source module since Spark 2.4. Please deploy the
    application as per the deployment section of "Apache Avro Data Source Guide".

    Examples
    --------
    >>> from pyspark.sql import Row
    >>> from pyspark.sql.avro.functions import from_avro, to_avro
    >>> data = [(1, Row(age=2, name='Alice'))]
    >>> df = spark.createDataFrame(data, ("key", "value"))
    >>> avroDf = df.select(to_avro(df.value).alias("avro"))
    >>> avroDf.collect()
    [Row(avro=b'\\x00\\x00\\x04\\x00\\nAlice')]

    >>> jsonFormatSchema = '''{"type":"record","name":"topLevelRecord","fields":
    ...     [{"name":"avro","type":[{"type":"record","name":"value","namespace":"topLevelRecord",
    ...     "fields":[{"name":"age","type":["long","null"]},
    ...     {"name":"name","type":["string","null"]}]},"null"]}]}'''
    >>> avroDf.select(from_avro(avroDf.avro, jsonFormatSchema).alias("value")).collect()
    [Row(value=Row(avro=Row(age=2, name='Alice')))]
    """
    _check_data(data)
    if not isinstance(jsonFormatSchema, str):
        raise PySparkTypeError(
            errorClass="INVALID_TYPE",
            messageParameters={"arg_name": "jsonFormatSchema", "arg_type": "str"},
        )
    if options is not None and not isinstance(options, dict):
        raise PySparkTypeError(
            errorClass="INVALID_TYPE",
            messageParameters={"arg_name": "options", "arg_type": "dict, optional"},
        )
    # As PySpark's, the schema is the literal of the str, which a request sets when it builds
    # the plan: one that UTF-8 cannot encode fails each DataFrame of the Column then.
    if options is None:
        return _invoke_function("from_avro", _to_col(data), _lit(jsonFormatSchema))
    return _invoke_function(
        "from_avro", _to_col(data), _lit(jsonFormatSchema), _dict_to_options_column(options)
    )


def to_avro(data, jsonFormatSchema=""):
    """
    Converts a column into binary of avro format.

    .. versionadded:: 3.0.0

    .. versionchanged:: 3.5.0
        Supports Spark Connect.

    Parameters
    ----------
    data : :class:`~pyspark.sql.Column` or str
        the data column.
    jsonFormatSchema : str, optional
        user-specified output avro schema in JSON string format.

    Notes
    -----
    Avro is built-in but external data source module since Spark 2.4. Please deploy the
    application as per the deployment section of "Apache Avro Data Source Guide".

    Examples
    --------
    >>> from pyspark.sql import Row
    >>> from pyspark.sql.avro.functions import to_avro
    >>> data = ['SPADES']
    >>> df = spark.createDataFrame(data, "string")
    >>> df.select(to_avro(df.value).alias("suite")).collect()
    [Row(suite=b'\\x00\\x0cSPADES')]

    >>> jsonFormatSchema = '''["null", {"type": "enum", "name": "value",
    ...     "symbols": ["SPADES", "HEARTS", "DIAMONDS", "CLUBS"]}]'''
    >>> df.select(to_avro(df.value, jsonFormatSchema).alias("suite")).collect()
    [Row(suite=b'\\x02\\x00')]
    """
    _check_data(data)
    if not isinstance(jsonFormatSchema, str):
        raise PySparkTypeError(
            errorClass="INVALID_TYPE",
            messageParameters={"arg_name": "jsonFormatSchema", "arg_type": "str"},
        )
    if jsonFormatSchema == "":
        return _call_function("to_avro", _to_col(data))
    return _invoke_function("to_avro", _to_col(data), _lit(jsonFormatSchema))


__all__ = ["from_avro", "to_avro"]
