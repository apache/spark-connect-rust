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
"""Rust-backed SparkSession under the connect import path (testing/connectutils imports from here)."""
from typing import Callable, Tuple

from pyspark.sql.session import SparkSession

__all__ = ["SparkSession"]


def _reduce_session(session_id: str) -> Tuple[Callable[[str], "SparkSession"], Tuple[str]]:
    """The ``__reduce__`` of the session of ``session_id``, as PySpark Connect's: a function,
    which cloudpickle pickles by value, that gives back the active session of that ID. It reads
    the class of ``pyspark.sql.connect.session`` when it is called, so that on a Python worker
    of the server, whose PySpark's has no active session, it raises ``NO_ACTIVE_SESSION``."""

    def creator(old_session_id: str) -> "SparkSession":
        from pyspark.sql.connect.session import SparkSession

        return SparkSession._getActiveSessionIfMatches(old_session_id)

    return creator, (session_id,)
