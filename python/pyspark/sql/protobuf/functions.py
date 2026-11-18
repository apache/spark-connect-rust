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
"""Protobuf functions, mirroring ``pyspark.sql.protobuf.functions``."""

# As PySpark's `_to_col`, which raises NOT_EXPECTED_TYPE for a `data` that is not a Column or a str.
from pyspark.sql.functions import _column_or_name as _to_col
from pyspark.sql.functions import _dict_to_options_column, _invoke_function, lit


def _descriptor_set(descFilePath, binaryDescriptorSet):
    """The binary descriptor set: the given bytes, else the content of the descriptor file."""
    if binaryDescriptorSet is not None:
        return binaryDescriptorSet
    if descFilePath is not None:
        with open(descFilePath, "rb") as f:
            return f.read()
    return None


def _protobuf_function(name, data, messageName, descriptor, options):
    """The Column of the Protobuf function `name`, as PySpark's: the message name and the
    descriptor set are the literals of the values as they are given, which PySpark does not check,
    and which a request sets when it builds the plan, as the server reads them, so a str that
    UTF-8 cannot encode fails each DataFrame of the Column then, and a message name that is not a
    str, such as 5, bytes or None, is the server's error of the function."""
    arguments = [_to_col(data), lit(messageName)]
    if descriptor is not None:
        arguments.append(lit(descriptor))
    if options is not None:
        arguments.append(_dict_to_options_column(options))
    return _invoke_function(name, *arguments)


def from_protobuf(
    data, messageName, descFilePath=None, options=None, binaryDescriptorSet=None
):
    """Converts a binary column of Protobuf format into its corresponding catalyst value.

    Args:
        data: the binary column (a Column or a column name).
        messageName: the protobuf message name to look for in the descriptor file.
        descFilePath: the protobuf descriptor file path.
        options: optional dict of options for the Protobuf reader.
        binaryDescriptorSet: the protobuf descriptor set as bytes, instead of a file.
    """
    descriptor = _descriptor_set(descFilePath, binaryDescriptorSet)
    return _protobuf_function("from_protobuf", data, messageName, descriptor, options)


def to_protobuf(
    data, messageName, descFilePath=None, options=None, binaryDescriptorSet=None
):
    """Converts a column into binary of Protobuf format.

    Args:
        data: the data column (a Column or a column name).
        messageName: the protobuf message name to look for in the descriptor file.
        descFilePath: the protobuf descriptor file path.
        options: optional dict of options for the Protobuf writer.
        binaryDescriptorSet: the protobuf descriptor set as bytes, instead of a file.
    """
    descriptor = _descriptor_set(descFilePath, binaryDescriptorSet)
    return _protobuf_function("to_protobuf", data, messageName, descriptor, options)


__all__ = ["from_protobuf", "to_protobuf"]
