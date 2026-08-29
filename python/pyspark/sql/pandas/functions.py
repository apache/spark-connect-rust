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
"""Rust-backed pandas UDF entry points under the official import path."""
from pyspark.sql.functions import pandas_udf  # noqa: F401
try:
    from pyspark.sql.functions import PandasUDFType  # noqa: F401
except ImportError:  # pragma: no cover
    class PandasUDFType:  # minimal stand-in; enum values set below
        SCALAR = 200
        GROUPED_MAP = 201
        GROUPED_AGG = 202
        SCALAR_ITER = 204
        MAP_ITER = 205
        COGROUPED_MAP = 206
__all__ = ["pandas_udf", "PandasUDFType"]
