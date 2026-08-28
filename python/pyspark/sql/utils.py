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
"""Small subset of pyspark.sql.utils needed by the drop-in and vendored modules."""


def is_remote() -> bool:
    """This client is always a Spark Connect (remote) client."""
    return True


def is_timestamp_ntz_preferred() -> bool:
    """Whether TIMESTAMP_NTZ is the preferred timestamp type.

    Best-effort for the Connect drop-in: honor the active session's
    spark.sql.timestampType when one is running, else False.
    """
    try:
        from pyspark.sql import SparkSession

        session = SparkSession.getActiveSession()
        if session is None:
            return False
        return session.conf.get("spark.sql.timestampType", None) == "TIMESTAMP_NTZ"
    except Exception:
        return False
