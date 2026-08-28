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
"""Logical plan builders under the connect import path.

This module re-exports plan builders from the Rust-backed implementation if available.
These are primarily used for testing within connectutils and are not available in the
Spark Connect client library itself (they're part of the internal protocol).
"""

# Try to import plan builders from the Rust extension
try:
    from pyspark._pyspark import (  # type: ignore
        LogicalPlan,
        Read,
        Range,
        SQL,
    )
except ImportError:
    # Fallback stubs for when the plan builders are not available
    class LogicalPlan:
        """Placeholder for LogicalPlan (not available in this Spark Connect client)."""
        pass

    class Read(LogicalPlan):
        """Placeholder for Read plan (not available in this Spark Connect client)."""
        def __init__(self, *args, **kwargs):
            pass

    class Range(LogicalPlan):
        """Placeholder for Range plan (not available in this Spark Connect client)."""
        def __init__(self, *args, **kwargs):
            pass

    class SQL(LogicalPlan):
        """Placeholder for SQL plan (not available in this Spark Connect client)."""
        def __init__(self, *args, **kwargs):
            pass

__all__ = ["LogicalPlan", "Read", "Range", "SQL"]
