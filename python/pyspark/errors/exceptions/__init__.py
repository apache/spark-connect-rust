#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License") you may not use this file except in compliance with
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

"""PySpark exception module - re-exported from Apache Spark."""

# Import from the upstream Apache Spark exceptions module
try:
    from pyspark.errors.exceptions.base import *  # noqa: F401, F403
except ImportError:
    # If upstream not available, export base exceptions
    from pyspark.errors import (  # noqa: F401, F403
        PySparkException,
        PySparkAssertionError,
        PySparkTypeError,
        QueryContextType,
        QueryContext,
    )


def _operation_id(self):  # type: ignore[no-untyped-def]
    """The Spark Connect ExecutePlan operation ID, when available.

    .. versionadded:: 4.3.0
    """
    return getattr(self, "_operation_id", None)


# pyspark 4.3.0 adds ``SparkConnectException.operation_id``. In this drop-in the Connect
# exception classes ARE the canonical base classes (``connect.SparkConnectException`` aliases
# ``PySparkException``), so the property lives on that shared base; the Rust transport sets
# ``_operation_id`` on errors raised for an ExecutePlan. ``base.py`` is vendored, hence here.
if not hasattr(PySparkException, "operation_id"):
    PySparkException.operation_id = property(_operation_id)  # type: ignore[attr-defined]

__all__ = [
    "PySparkException",
    "PySparkAssertionError",
    "PySparkTypeError",
    "QueryContextType",
    "QueryContext",
]
