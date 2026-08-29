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
"""Protobuf message definitions under the connect import path.

This module re-exports protobuf types from the generated protocol if available.
These are used in testing Connect functionality and are part of the internal protocol.
This drop-in Spark Connect client does not expose the full protobuf layer, so this
module provides stubs for compatibility with tests that reference the proto layer.
"""

# Try to import protobuf types if available
try:
    from pyspark._pyspark import (  # type: ignore
        Plan,
        Relation,
        Command,
    )
except ImportError:
    # Fallback stubs for when protobuf types are not available
    class Plan:
        """Placeholder for protobuf Plan message."""
        def __init__(self):
            self.root = None
            self.command = None

    class Relation:
        """Placeholder for protobuf Relation message."""
        pass

    class Command:
        """Placeholder for protobuf Command message."""
        pass

__all__ = ["Plan", "Relation", "Command"]
