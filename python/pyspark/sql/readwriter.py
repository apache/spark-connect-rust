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
"""Re-export of the Rust-backed class(es) under the official pyspark.sql import path."""
import typing as _typing

from pyspark._pyspark import DataFrameReader, DataFrameWriter, DataFrameWriterV2

__all__ = ["DataFrameReader", "DataFrameWriter", "DataFrameWriterV2"]

# PySpark's aliases of the types of the paths and of the lists of column names, which its
# readers and writers are annotated with, and an application's own code can be: of typing's
# List and Tuple, as PySpark's.
PathOrPaths = _typing.Union[str, _typing.List[str]]  # noqa: UP006
TupleOrListOfString = _typing.Union[_typing.List[str], _typing.Tuple[str, ...]]  # noqa: UP006
