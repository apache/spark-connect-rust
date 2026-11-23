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
# The SparkSession of Spark Connect, the native class, under PySpark Connect's import path.
#
# As in PySpark, ``pyspark.sql.SparkSession`` is the classic class (``pyspark.sql.session``, of
# Apache Spark), whose builder makes a session of this class, which it is not a subclass of.
from typing import Callable

from pyspark._pyspark import SparkSession, _install_builder
from pyspark.errors import PySparkTypeError as _PySparkTypeError

# As PySpark Connect's module of the session, which binds the conversions of `createDataFrame` and
# the types that they make when it is imported, with the first session: its `createDataFrame`
# calls these by their names here, so a name rebound in `pyspark.sql.types` or
# `pyspark.sql.pandas.types` after this import is not seen here, and one rebound before it, or
# here, is. The conversions of a pandas DataFrame read them here at each call
# (`pyspark.sql._pandas_input`), as PySpark's do; the converters that the conversion of a column
# imports when it is called (`create_arrow_array_from_pandas`) are those of the module of
# `sys.modules` then, in both.
from pyspark.sql.pandas.conversion import create_arrow_table_from_pandas  # noqa: F401
from pyspark.sql.pandas.types import (  # noqa: F401
    _deduplicate_field_names,
    from_arrow_type,
    to_arrow_schema,
)
from pyspark.sql.session import SparkSession as _PySparkSession
from pyspark.sql.session import classproperty as _classproperty
from pyspark.sql.types import (  # noqa: F401
    AtomicType,
    DayTimeIntervalType,
    MapType,
    StringType,
    StructType,
    TimestampType,
)

__all__ = ["SparkSession"]

# As PySpark Connect's `SparkSession.builder`, of its class body: PySpark's classproperty, whose
# each access, of the class or of a session, is a new builder, with the docstring of the classic
# builder. The native class takes it from here (`_install_builder`), as the class of
# pyspark.sql.session cannot be imported while the extension loads.
_builder = _classproperty(lambda cls: cls.Builder())
_builder.__doc__ = _PySparkSession.builder.__doc__
_install_builder(_builder)
del _builder


def _check_schema(schema: object) -> None:
    """PySpark Connect's check of the schema of ``createDataFrame``, of a DDL string as the server
    parsed it, against the ``AtomicType`` and the ``StructType`` of this module, which it reads
    when it is called, as PySpark's reads the names of its module: a struct reads its ``fields``,
    and an object that neither takes, nor a list, a tuple or None, raises ``NOT_EXPECTED_TYPE``.
    The session calls it before it reads any setting, when the names are not the native classes,
    of which its own checks are the same, also of an ``isinstance`` that raises."""
    if isinstance(schema, (AtomicType, StructType)):
        if isinstance(schema, StructType):
            len(schema.fields)  # type: ignore[attr-defined]
    elif not isinstance(schema, (list, tuple)) and schema is not None:
        raise _PySparkTypeError(
            errorClass="NOT_EXPECTED_TYPE",
            messageParameters={
                "arg_name": "schema",
                "expected_type": "list, None or StructType",
                "arg_type": type(schema).__name__,
            },
        )


def _reduce_session(session_id: str) -> tuple[Callable[[str], "SparkSession"], tuple[str]]:
    """The ``__reduce__`` of the session of ``session_id``, as PySpark Connect's: a function,
    which cloudpickle pickles by value, that gives back the active session of that ID. It reads
    the class of ``pyspark.sql.connect.session`` when it is called, so that on a Python worker
    of the server, whose PySpark's has no active session, it raises ``NO_ACTIVE_SESSION``."""

    def creator(old_session_id: str) -> "SparkSession":
        from pyspark.sql.connect.session import SparkSession

        return SparkSession._getActiveSessionIfMatches(old_session_id)

    # Named as PySpark's, so that the error of a plain pickle of a session names the same function.
    creator.__qualname__ = "SparkSession.__reduce__.<locals>.creator"
    return creator, (session_id,)
