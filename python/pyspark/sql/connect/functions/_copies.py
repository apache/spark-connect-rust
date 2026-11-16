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
"""The functions of ``pyspark.sql.connect.functions``, as PySpark Connect's: of the modules of
PySpark Connect, distinct from those of ``pyspark.sql.functions``, which PySpark's classic module
holds."""

import types
from typing import Any


def connect_functions(namespace: dict[str, Any], source: Any, module: str) -> None:
    """Put in ``namespace`` the public names of the module ``source``: a copy of each function that
    ``source`` defines (its ``__module__`` is that of ``source``) of the module ``module``, which
    runs as the function does, and each other name as it is. PySpark Connect's functions are of
    the modules of ``pyspark.sql.connect.functions``, so that ``repr``, ``__module__`` and a pickle
    by reference name them there, and a worker of the server imports them from there."""
    for name in dir(source):
        if name.startswith("_"):
            continue
        value = getattr(source, name)
        if isinstance(value, types.FunctionType) and value.__module__ == source.__name__:
            copy = types.FunctionType(
                value.__code__, value.__globals__, value.__name__, value.__defaults__,
                value.__closure__,
            )
            copy.__kwdefaults__ = value.__kwdefaults__
            copy.__dict__.update(value.__dict__)
            copy.__doc__ = value.__doc__
            copy.__annotations__ = dict(value.__annotations__)
            copy.__qualname__ = value.__qualname__
            copy.__module__ = module
            value = copy
        namespace[name] = value
