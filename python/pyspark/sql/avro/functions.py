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
"""Avro functions, mirroring ``pyspark.sql.avro.functions``."""

from pyspark import _pyspark
from pyspark.errors import PySparkTypeError
from pyspark.sql.column import Column
from pyspark.sql.functions import _call_function, _dict_to_options_column, _to_col

_functions = _pyspark.functions
_pyfunc_from_avro = _functions.pyfunc_from_avro
_pyfunc_from_avro_with_options = _functions.pyfunc_from_avro_with_options
_pyfunc_to_avro_with_schema = _functions.pyfunc_to_avro_with_schema


def _check_data(data):
    if not isinstance(data, (Column, str)):
        raise PySparkTypeError(
            errorClass="INVALID_TYPE",
            messageParameters={"arg_name": "data", "arg_type": "pyspark.sql.Column or str"},
        )


def from_avro(data, jsonFormatSchema, options=None):
    """Converts a binary column of Avro format into its corresponding catalyst value.

    Args:
        data: the binary column (a Column or a column name).
        jsonFormatSchema: the Avro schema in JSON string format.
        options: optional dict of options for the Avro reader.
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
    if options is None:
        return _pyfunc_from_avro(_to_col(data), jsonFormatSchema)
    return _pyfunc_from_avro_with_options(
        _to_col(data), jsonFormatSchema, _dict_to_options_column(options)
    )


def to_avro(data, jsonFormatSchema=""):
    """Converts a column into binary of Avro format.

    Args:
        data: the data column (a Column or a column name).
        jsonFormatSchema: the Avro schema in JSON string format; empty to infer it.
    """
    _check_data(data)
    if not isinstance(jsonFormatSchema, str):
        raise PySparkTypeError(
            errorClass="INVALID_TYPE",
            messageParameters={"arg_name": "jsonFormatSchema", "arg_type": "str"},
        )
    if jsonFormatSchema == "":
        return _call_function("to_avro", _to_col(data))
    return _pyfunc_to_avro_with_schema(_to_col(data), jsonFormatSchema)


__all__ = ["from_avro", "to_avro"]
