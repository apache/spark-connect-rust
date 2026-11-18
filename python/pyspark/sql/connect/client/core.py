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

"""Thin re-export of the client classes of ``pyspark.sql.connect.client.core``, which official
PySpark code and tests import from there, and of which these classes are, as PySpark's: so a
pickle names them here. The gRPC client itself is the Rust transport: the drop-in does not
provide the rest of ``core`` (the gRPC stub of the client)."""

__all__ = ["ChannelBuilder", "DefaultChannelBuilder", "RpcDeadlines", "SparkConnectClient"]

from typing import Any

# As PySpark's core, which imports the messages of Spark Connect, so that protobuf's default pool
# has their files once this module is imported: a pickle of PySpark's PlanObservedMetrics, which
# imports this module before it makes the Expression.Literal messages of its metrics of their full
# names (protobuf's `_InternalConstructMessage`), loads in a new process, as with PySpark.
import pyspark.sql.connect.proto  # noqa: F401
from pyspark.sql.connect.client import (
    AnalyzeResult,  # noqa: F401
    ChannelBuilder,
    ConfigResult,  # noqa: F401
    DefaultChannelBuilder,
    PlanObservedMetrics,  # noqa: F401
    RpcDeadlines,
    SparkConnectClient,
)

# The names of pyspark.sql.connect.client.retries that PySpark's core imports, and so has too.
_RETRIES_NAMES = (
    "RetryPolicy",
    "Retrying",
    "DefaultPolicy",
    "DEFAULT_MAX_RETRY_EXCEPTION_ELAPSED_TIME",
)


def __getattr__(name: str) -> Any:
    # Of the retries module when they are first read: it imports grpc, which this client does
    # not need otherwise.
    if name in _RETRIES_NAMES:
        from pyspark.sql.connect.client import retries

        return getattr(retries, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
