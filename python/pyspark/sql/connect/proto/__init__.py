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
build their own messages, such as ``pyspark.ml``, use them, which needs the ``protobuf``
package. Without it, or of a version older than the generated modules need, this package
exports only the ``ExecutePlanResponse`` of the progress of an execution, which the Rust
transport builds for the progress handlers, and another message is an ``AttributeError`` that
names the reason.
"""

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
except ImportError as error:
    # No protobuf package, or one older than the generated modules need
    # (google.protobuf.runtime_version.VersionError is an ImportError).
    _import_error = error

    def __getattr__(name: str) -> "object":
        # The messages that the generated modules define, which need the protobuf package: an
        # AttributeError, as of a missing attribute, of the reason that they are not there.
        if name.startswith("_"):
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}: the messages of Spark Connect, "
            "such as those of pyspark.ml, need protobuf >= 6.33.5, which could not be "
            f"imported ({_import_error})"
        )

    class ExecutePlanResponse:  # type: ignore[no-redef]
        """The ``ExecutePlanResponse`` of ``base.proto``, of the progress of an execution: the
        Rust transport reads the responses of the server, of which ``ExecutionProgress`` holds
        the fields that ``pyspark.sql.connect.shell.progress.from_proto`` reads."""

        class ExecutionProgress:
            """``ExecutePlanResponse.ExecutionProgress``: the stages of an execution, and its
            running tasks."""

            class StageInfo:
                """``ExecutionProgress.StageInfo``: the tasks and the bytes read of a stage."""

                def __init__(
                    self,
                    stage_id=0,
                    num_tasks=0,
                    num_completed_tasks=0,
                    input_bytes_read=0,
                    done=False,
                ):
                    self.stage_id = stage_id
                    self.num_tasks = num_tasks
                    self.num_completed_tasks = num_completed_tasks
                    self.input_bytes_read = input_bytes_read
                    self.done = done

            def __init__(self, stages=(), num_inflight_tasks=0):
                self.stages = list(stages)
                self.num_inflight_tasks = num_inflight_tasks
