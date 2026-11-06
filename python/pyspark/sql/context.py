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
"""Legacy SQLContext/HiveContext plus the UDF/UDTF registration re-exports that
pyspark.sql.__init__ imports.

As in pyspark 4.3.0, these are the classic entry points; in Spark Connect mode (always,
for this client) ``getOrCreate()`` routes to the SparkSession-backed counterparts in
:mod:`pyspark.sql.connect.context`. Constructing them from a ``SparkContext`` is not
possible here, since a Connect client has no SparkContext.
"""
import warnings
from typing import Any, ClassVar, Optional

from pyspark.errors import PySparkNotImplementedError
from pyspark.sql.udf import UDFRegistration
from pyspark.sql.udtf import UDTFRegistration

__all__ = ["SQLContext", "HiveContext", "UDFRegistration", "UDTFRegistration"]


class SQLContext:
    """The entry point for working with structured data in Spark 1.x.

    .. deprecated:: 3.0.0
        Use :func:`SparkSession.builder.getOrCreate()` instead.
    """

    _instantiatedContext: ClassVar[Optional[Any]] = None

    def __init__(self, sparkContext: Any, sparkSession: Any = None, jsqlContext: Any = None):
        raise PySparkNotImplementedError(
            errorClass="NOT_IMPLEMENTED",
            messageParameters={
                "feature": f"{type(self).__name__}(sparkContext) in Spark Connect; "
                f"use {type(self).__name__}.getOrCreate()"
            },
        )

    @classmethod
    def getOrCreate(cls, sc: Any = None) -> Any:
        """Get the existing SQLContext or create one for the active SparkSession.

        In Spark Connect mode ``sc`` is not required and the active
        :class:`SparkSession` is used.

        .. deprecated:: 3.0.0
            Use :func:`SparkSession.builder.getOrCreate()` instead.
        """
        from pyspark.sql.utils import is_remote

        warnings.warn(
            "Deprecated in 3.0.0. Use SparkSession.builder.getOrCreate() instead.",
            FutureWarning,
        )
        if not is_remote():  # pragma: no cover - this client is always remote
            raise PySparkNotImplementedError(
                errorClass="NOT_IMPLEMENTED",
                messageParameters={"feature": "SQLContext outside Spark Connect"},
            )
        from pyspark.sql.connect import context as _connect_context

        session = _get_active_session_or_create()
        # Route to the Connect counterpart by name so subclasses (HiveContext) are handled:
        # the Connect HiveContext._from_session raises PySparkNotImplementedError.
        connect_cls = getattr(_connect_context, cls.__name__, None)
        if connect_cls is None:
            # A user-defined subclass has no Connect counterpart; fail loudly rather than
            # return a base context missing the subclass's attributes.
            raise PySparkNotImplementedError(
                errorClass="NOT_IMPLEMENTED",
                messageParameters={"feature": f"{cls.__name__}.getOrCreate in Spark Connect"},
            )
        return connect_cls._get_or_create_from_session(session)


class HiveContext(SQLContext):
    """A variant of Spark SQL that integrates with data stored in Hive.

    .. deprecated:: 2.0.0
        Use SparkSession.builder.enableHiveSupport().getOrCreate().
    """


def _get_active_session_or_create() -> Any:
    """Mirror ``SparkSession._getActiveSessionOrCreate`` for the Rust-backed session: the
    active session, else a new one against ``SPARK_REMOTE``."""
    import os

    from pyspark.sql.session import SparkSession

    session = SparkSession.getActiveSession()
    if session is None:
        session = SparkSession.builder.remote(
            os.environ.get("SPARK_REMOTE", "sc://localhost")
        ).getOrCreate()
    return session
