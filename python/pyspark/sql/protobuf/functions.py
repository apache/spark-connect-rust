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

from pyspark import _pyspark
from pyspark.sql.functions import _dict_to_options_column, _to_col

_functions = _pyspark.functions
_pyfunc_from_protobuf = _functions.pyfunc_from_protobuf
_pyfunc_from_protobuf_with_descriptor = _functions.pyfunc_from_protobuf_with_descriptor
_pyfunc_from_protobuf_with_descriptor_and_options = (
    _functions.pyfunc_from_protobuf_with_descriptor_and_options
)
_pyfunc_from_protobuf_with_options = _functions.pyfunc_from_protobuf_with_options
_pyfunc_to_protobuf = _functions.pyfunc_to_protobuf
_pyfunc_to_protobuf_with_descriptor = _functions.pyfunc_to_protobuf_with_descriptor
_pyfunc_to_protobuf_with_descriptor_and_options = (
    _functions.pyfunc_to_protobuf_with_descriptor_and_options
)
_pyfunc_to_protobuf_with_options = _functions.pyfunc_to_protobuf_with_options


def _descriptor_set(descFilePath, binaryDescriptorSet):
    """The binary descriptor set: the given bytes, else the content of the descriptor file."""
    if binaryDescriptorSet is not None:
        return binaryDescriptorSet
    if descFilePath is not None:
        with open(descFilePath, "rb") as f:
            return f.read()
    return None


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
    column = _to_col(data)
    if descriptor is None:
        if options is None:
            return _pyfunc_from_protobuf(column, messageName)
        return _pyfunc_from_protobuf_with_options(
            column, messageName, _dict_to_options_column(options)
        )
    if options is None:
        return _pyfunc_from_protobuf_with_descriptor(column, messageName, descriptor)
    return _pyfunc_from_protobuf_with_descriptor_and_options(
        column, messageName, descriptor, _dict_to_options_column(options)
    )


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
    column = _to_col(data)
    if descriptor is None:
        if options is None:
            return _pyfunc_to_protobuf(column, messageName)
        return _pyfunc_to_protobuf_with_options(
            column, messageName, _dict_to_options_column(options)
        )
    if options is None:
        return _pyfunc_to_protobuf_with_descriptor(column, messageName, descriptor)
    return _pyfunc_to_protobuf_with_descriptor_and_options(
        column, messageName, descriptor, _dict_to_options_column(options)
    )


__all__ = ["from_protobuf", "to_protobuf"]
