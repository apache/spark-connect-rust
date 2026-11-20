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

# PySpark exception module - re-exported from Apache Spark.

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

# PySpark 4.3.0 has ``operation_id`` on ``SparkConnectException`` only
# (``pyspark.errors.exceptions.connect``), of the ``_operation_id`` that the client sets on any
# error of an ExecutePlan, as the Rust transport sets it: an error of the server, or of a failed
# RPC, is one of the Connect classes, which have the property, and another error, such as a
# ``PySparkTypeError`` of a conversion of the results, keeps its ``_operation_id`` only.

__all__ = [
    "PySparkException",
    "PySparkAssertionError",
    "PySparkTypeError",
    "QueryContextType",
    "QueryContext",
]
