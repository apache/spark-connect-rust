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
"""The ML class that a live test runs: native in this client, the reference's in official PySpark."""

import importlib


def native_class(module, name, reference_module):
    """The native class `name` of `module` in this client. A class that this client lacks fails the
    test, rather than running it with the vendored pyspark.ml class. Official PySpark, which has no
    pyspark._pyspark, runs the test with its JVM-backed class of `reference_module`, as the
    reference."""
    try:
        import pyspark._pyspark  # noqa: F401
    except ImportError:
        return getattr(importlib.import_module(reference_module), name)
    native = getattr(importlib.import_module(module), name)
    assert native.__module__.startswith("pyspark.ml.connect"), native
    return native
