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
"""Re-export of ``pyspark.sql.functions`` under the Connect import path."""
from typing import TYPE_CHECKING

from pyspark.sql.functions import *  # noqa: F401,F403
from pyspark.sql.functions import col, expr, _invoke_function, _to_col  # noqa: F401

if TYPE_CHECKING:
    from pyspark.sql import Column
    from pyspark.sql.connect._typing import ColumnOrName


def _invoke_function_over_columns(name: str, *cols: "ColumnOrName") -> "Column":
    """Invoke an internal function by name over the given columns.

    Mirrors ``pyspark.sql.connect.functions.builtin._invoke_function_over_columns``:
    builds an ``UnresolvedFunction`` carrying the given name, which the Connect
    server resolves against its function registry (this is how the Pandas-on-Spark
    internal functions -- ``distributed_sequence_id``, ``pandas_product``, etc. --
    are dispatched). No client-side reimplementation of those functions.
    """
    return _invoke_function(name, *[_to_col(c) for c in cols])
