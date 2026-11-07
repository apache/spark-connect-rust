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

"""Tuple-backed Row semantics from PySpark's ``pyspark.sql.types.Row``.

Mirrors ``class Row`` in apache/spark v4.3.0, ``python/pyspark/sql/types.py``.
Embedded by the extension so ``pyspark._pyspark.Row`` and the public imports
refer to the same class without importing the partially initialized types shim.
"""


class Row(tuple):
    """A row with optional field names, also usable as a factory of named rows."""

    __module__ = "pyspark.sql.types"

    def __new__(cls, *args, **kwargs):
        if args and kwargs:
            from pyspark.errors import PySparkValueError

            raise PySparkValueError(
                errorClass="CANNOT_SET_TOGETHER",
                messageParameters={"arg_list": "args and kwargs"},
            )
        if kwargs:
            row = tuple.__new__(cls, list(kwargs.values()))
            row.__fields__ = list(kwargs.keys())
            return row
        return tuple.__new__(cls, args)

    def asDict(self, recursive=False):
        """Return field names and values as a dict, optionally converting nested rows."""
        if not hasattr(self, "__fields__"):
            from pyspark.errors import PySparkTypeError

            raise PySparkTypeError(
                errorClass="CANNOT_CONVERT_TYPE",
                messageParameters={"from_type": "Row", "to_type": "dict"},
            )
        if recursive:

            def conv(obj):
                if isinstance(obj, Row):
                    return obj.asDict(True)
                if isinstance(obj, list):
                    return [conv(value) for value in obj]
                if isinstance(obj, dict):
                    return {key: conv(value) for key, value in obj.items()}
                return obj

            return dict(zip(self.__fields__, (conv(value) for value in self)))
        return dict(zip(self.__fields__, self))

    def __contains__(self, item):
        if hasattr(self, "__fields__"):
            return item in self.__fields__
        return super().__contains__(item)

    def __call__(self, *args):
        from pyspark.sql.types import _create_row

        if len(args) > len(self):
            from pyspark.errors import PySparkValueError

            raise PySparkValueError(
                errorClass="TOO_MANY_VALUES",
                messageParameters={
                    "expected": str(len(self)),
                    "item": "fields",
                    "actual": str(len(args)),
                },
            )
        return _create_row(self, args)

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return super().__getitem__(item)
        from pyspark.errors import PySparkKeyError, PySparkValueError

        try:
            index = self.__fields__.index(item)
            return super().__getitem__(index)
        except IndexError:
            raise PySparkKeyError(errorClass="KEY_NOT_EXISTS", messageParameters={"key": str(item)})
        except ValueError:
            raise PySparkValueError(item)

    def __getattr__(self, item):
        from pyspark.errors import PySparkAttributeError

        if item.startswith("__"):
            raise PySparkAttributeError(
                errorClass="ATTRIBUTE_NOT_SUPPORTED", messageParameters={"attr_name": item}
            )
        try:
            index = self.__fields__.index(item)
            return self[index]
        except IndexError:
            raise PySparkAttributeError(
                errorClass="ATTRIBUTE_NOT_SUPPORTED", messageParameters={"attr_name": item}
            )
        except ValueError:
            raise PySparkAttributeError(
                errorClass="ATTRIBUTE_NOT_SUPPORTED", messageParameters={"attr_name": item}
            )

    def __setattr__(self, key, value):
        if key != "__fields__":
            from pyspark.errors import PySparkRuntimeError

            raise PySparkRuntimeError(errorClass="READ_ONLY", messageParameters={"object": "Row"})
        self.__dict__[key] = value

    def __reduce__(self):
        if hasattr(self, "__fields__"):
            from pyspark.sql.types import _create_row

            return _create_row, (self.__fields__, tuple(self))
        return tuple.__reduce__(self)

    def __repr__(self):
        if hasattr(self, "__fields__"):
            return "Row(%s)" % ", ".join(
                "%s=%r" % (key, value) for key, value in zip(self.__fields__, tuple(self))
            )
        return "<Row(%s)>" % ", ".join(repr(value) for value in self)
