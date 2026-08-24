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

"""
Streaming data readers and writers.

This module re-exports the PyO3-implemented streaming classes that provide
the client-side API for reading and writing streaming data, and extends them
with Python-side implementations of foreachBatch and foreach.
"""

from typing import Callable, Union, TYPE_CHECKING
from pyspark import cloudpickle
from pyspark.sql.connect.utils import get_python_ver
from pyspark.serializers import CPickleSerializer, AutoBatchedSerializer
import pickle
from pyspark.errors import PySparkPicklingError

from pyspark._pyspark import (
    DataStreamReader as _DataStreamReader,
    DataStreamWriter as _DataStreamWriter,
    Trigger as _Trigger,
)

if TYPE_CHECKING:
    from pyspark.sql.connect.dataframe import DataFrame
    from pyspark.sql._typing import SupportsProcess

# Re-export the core classes from the Rust extension
DataStreamReader = _DataStreamReader
Trigger = _Trigger


class DataStreamWriter(_DataStreamWriter):
    """Wrapper around the Rust DataStreamWriter with Python-side methods."""

    def foreachBatch(self, func: Callable[["DataFrame", int], None]) -> "DataStreamWriter":
        """
        Set a foreachBatch function to apply to each batch of streaming data.

        Parameters
        ----------
        func : callable
            A function that takes (batch_df, batch_id) and performs an action.

        Returns
        -------
        DataStreamWriter
            The same DataStreamWriter instance.
        """
        try:
            command = cloudpickle.dumps(func)
        except pickle.PicklingError:
            raise PySparkPicklingError(
                errorClass="STREAMING_CONNECT_SERIALIZATION_ERROR",
                messageParameters={"name": "foreachBatch"},
            )
        import sys
        python_ver = "%d.%d" % sys.version_info[:2]
        return super().foreachBatch(command, python_ver)

    def foreach(self, f: Union[Callable, "SupportsProcess"]) -> "DataStreamWriter":
        """
        Set a foreach function to apply to each row of streaming data.

        Parameters
        ----------
        f : callable or SupportsProcess
            A function that takes a Row or an object with open/process/close methods.

        Returns
        -------
        DataStreamWriter
            The same DataStreamWriter instance.
        """
        # For foreach, wrap the function similar to how PySpark does it
        serializer = AutoBatchedSerializer(CPickleSerializer())
        command = (f, None, serializer, serializer)
        try:
            pickled_command = cloudpickle.dumps(command)
        except pickle.PicklingError:
            raise PySparkPicklingError(
                errorClass="STREAMING_CONNECT_SERIALIZATION_ERROR",
                messageParameters={"name": "foreach"},
            )
        import sys
        python_ver = "%d.%d" % sys.version_info[:2]
        return super().foreach(pickled_command, python_ver)


__all__ = ["DataStreamReader", "DataStreamWriter", "Trigger"]
