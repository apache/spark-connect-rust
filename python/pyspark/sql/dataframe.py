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

"""DataFrame API - re-exported from Rust-backed client or upstream."""

# For connect-only mode, re-export DataFrame from the Rust-backed client
try:
    from pyspark._pyspark import DataFrame  # noqa: F401
except ImportError:
    # Fallback to upstream
    from pyspark.sql.dataframe import DataFrame as _DataFrame  # noqa: F401
    DataFrame = _DataFrame

__all__ = ["DataFrame"]
