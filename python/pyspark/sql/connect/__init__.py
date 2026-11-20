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
"""Spark Connect client"""

# The modules of this package are those of PySpark Connect's import paths, of the drop-in's
# implementation, whose client is the Rust core.
#
# As PySpark's package, it imports check_dependencies, but it does not run it at the import.
# PySpark's call requires pandas, PyArrow, grpcio, grpcio-status, googleapis-common-protos and
# zstandard, which this client does not need: its transport is the Rust core's, and it runs
# without PyArrow and pandas. Of a main module without a file, as of ``python -c``, it also takes
# the import for a run of the doctests and imports pyspark.testing.connectutils, which exits the
# process without SPARK_HOME or the test dependencies. A program that calls check_dependencies
# runs it (see pyspark.sql.connect.utils).
from pyspark.sql.connect.utils import check_dependencies  # noqa: F401
