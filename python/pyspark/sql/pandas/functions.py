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
"""Rust-backed pandas UDF entry points under the official import path.

Canonical home of the ``PandasUDFType`` / ``ArrowUDFType`` enums (mirroring reference
pyspark); ``pyspark.sql.functions`` re-exports them from here.
"""
from pyspark.sql.functions import pandas_udf  # noqa: F401


class PandasUDFType:
    """Pandas UDF Types. See :func:`pyspark.sql.functions.pandas_udf`."""

    SCALAR = 200
    SCALAR_ITER = 204
    GROUPED_MAP = 201
    GROUPED_AGG = 202
    GROUPED_AGG_ITER = 217
    MAP_ITER = 205
    COGROUPED_MAP = 206


class ArrowUDFType:
    """Arrow UDF Types. See :func:`pyspark.sql.functions.pandas_udf`."""

    SCALAR = 250
    SCALAR_ITER = 251
    GROUPED_AGG = 252
    GROUPED_AGG_ITER = 254


__all__ = ["pandas_udf", "PandasUDFType", "ArrowUDFType"]
