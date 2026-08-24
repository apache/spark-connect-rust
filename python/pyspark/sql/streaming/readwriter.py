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
the client-side API for reading and writing streaming data.
"""

from pyspark._pyspark import (
    DataStreamReader as _DataStreamReader,
    DataStreamWriter as _DataStreamWriter,
    Trigger as _Trigger,
)

# Re-export the core classes from the Rust extension
DataStreamReader = _DataStreamReader
DataStreamWriter = _DataStreamWriter
Trigger = _Trigger

__all__ = ["DataStreamReader", "DataStreamWriter", "Trigger"]
