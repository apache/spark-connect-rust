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
"""The protobuf messages of Spark Connect, as PySpark's ``pyspark.sql.connect.proto``.

The generated modules (``*_pb2.py``) are vendored from Apache Spark v4.3.0. As PySpark's
``__init__``, this package exports their messages, except the gRPC stubs of ``base_pb2_grpc``:
the Rust transport speaks gRPC, so ``grpcio`` is never imported. The modules of PySpark that
build their own messages, such as ``pyspark.ml``, use them.

The generated modules need the ``protobuf`` package. Without it, or of a version older than they
need, this package exports the messages of the drop-in's Rust protobuf runtime instead
(``pyspark.sql.connect.proto._rust``): the same classes, with the API of protobuf's messages, which
the Rust client encodes. ``PYSPARK_CONNECT_PROTO_IMPLEMENTATION=rust`` asks for them even when
``protobuf`` is installed, and ``_implementation`` names the one in use.
"""
import os as _os

_implementation = "protobuf"
if _os.environ.get("PYSPARK_CONNECT_PROTO_IMPLEMENTATION", "").lower() == "rust":
    _implementation = "rust"
else:
    try:
        from pyspark.sql.connect.proto.base_pb2 import *  # noqa: F401,F403
        from pyspark.sql.connect.proto.types_pb2 import *  # noqa: F401,F403
        from pyspark.sql.connect.proto.commands_pb2 import *  # noqa: F401,F403
        from pyspark.sql.connect.proto.expressions_pb2 import *  # noqa: F401,F403
        from pyspark.sql.connect.proto.relations_pb2 import *  # noqa: F401,F403
        from pyspark.sql.connect.proto.catalog_pb2 import *  # noqa: F401,F403
        from pyspark.sql.connect.proto.common_pb2 import *  # noqa: F401,F403
        from pyspark.sql.connect.proto.ml_pb2 import *  # noqa: F401,F403
        from pyspark.sql.connect.proto.ml_common_pb2 import *  # noqa: F401,F403
        from pyspark.sql.connect.proto.pipelines_pb2 import *  # noqa: F401,F403
    except ImportError:
        # No protobuf package, or one older than the generated modules need
        # (google.protobuf.runtime_version.VersionError is an ImportError).
        _implementation = "rust"

if _implementation == "rust":
    from pyspark.sql.connect.proto import _rust

    _rust.install(globals())
    del _rust

del _os
