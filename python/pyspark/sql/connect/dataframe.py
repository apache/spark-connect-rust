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
"""Rust-backed DataFrame under the connect import path (testing/connectutils imports from here)."""
from typing import Any, Callable, Tuple

from pyspark._pyspark import DataFrame
from pyspark.sql.dataframe import DataFrameNaFunctions, DataFrameStatFunctions

__all__ = ["DataFrame", "DataFrameNaFunctions", "DataFrameStatFunctions"]


def _reduce_dataframe(session: Any, relation: bytes) -> Tuple[Callable[..., Any], Tuple[Any, ...]]:
    """The ``__reduce__`` of a DataFrame of ``session`` and the bytes of its ``relation``, as
    PySpark Connect's, of its session and its plan: a function, which cloudpickle pickles by
    value, of the session, which unpickles first, and of the relation."""

    def creator(session: Any, relation: bytes) -> "DataFrame":
        from pyspark import _pyspark

        return _pyspark.DataFrame._of_relation(session, relation)

    return creator, (session, relation)
