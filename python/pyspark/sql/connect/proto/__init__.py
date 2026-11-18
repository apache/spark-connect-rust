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

PySpark's package exports the messages of the modules that protoc generates of the protos of
Spark Connect (``base_pb2`` and the others), which need the ``protobuf`` package. This one exports
the same classes of the drop-in's protobuf runtime in Rust (``pyspark.sql.connect.proto._rust``),
with the API, bytes, text and errors of protobuf's messages, and makes each of those modules
(``pyspark.sql.connect.proto.base_pb2``, ...) the one of the runtime, so that ``protobuf`` is not
needed. The modules of PySpark that build their own messages, such as ``pyspark.ml``, use them.
The gRPC stubs of ``base_pb2_grpc`` are not exported: the Rust transport speaks gRPC.

Where ``protobuf`` is installed, the files of the protos are added to its default descriptor
pool too, as the generated modules add them, so that the modules that protoc generates of other
protos that import them (those of a plugin, such as Delta Lake's) build, and those modules get the
classes of the runtime, whose message fields take the messages of this package as they are, as
protobuf's take PySpark's.
"""
from pyspark.sql.connect.proto import _rust

_rust.install(globals())
del _rust
