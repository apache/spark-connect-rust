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
    """
    A row in :class:`DataFrame`.
    The fields in it can be accessed:

    * like attributes (``row.key``)
    * like dictionary values (``row[key]``)

    ``key in row`` will search through row keys.

    Row can be used to create a row object by using named arguments.
    It is not allowed to omit a named argument to represent that the value is
    None or missing. This should be explicitly set to None in this case.

    .. versionchanged:: 3.0.0
        Rows created from named arguments no longer have
        field names sorted alphabetically and will be ordered in the position as
        entered.

    Examples
    --------
    >>> from pyspark.sql import Row
    >>> row = Row(name="Alice", age=11)
    >>> row
    Row(name='Alice', age=11)
    >>> row['name'], row['age']
    ('Alice', 11)
    >>> row.name, row.age
    ('Alice', 11)
    >>> 'name' in row
    True
    >>> 'wrong_key' in row
    False

    Row also can be used to create another Row like class, then it
    could be used to create Row objects, such as

    >>> Person = Row("name", "age")
    >>> Person
    <Row('name', 'age')>
    >>> 'name' in Person
    True
    >>> 'wrong_key' in Person
    False
    >>> Person("Alice", 11)
    Row(name='Alice', age=11)

    This form can also be used to create rows as tuple values, i.e. with unnamed
    fields.

    >>> row1 = Row("Alice", 11)
    >>> row2 = Row(name="Alice", age=11)
    >>> row1 == row2
    True
    """

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
        """
        Return as a dict

        Parameters
        ----------
        recursive : bool, optional
            turns the nested Rows to dict (default: False).

        Notes
        -----
        If a row contains duplicate field names, e.g., the rows of a join
        between two :class:`DataFrame` that both have the fields of same names,
        one of the duplicate fields will be selected by ``asDict``. ``__getitem__``
        will also return one of the duplicate fields, however returned value might
        be different to ``asDict``.

        Examples
        --------
        >>> from pyspark.sql import Row
        >>> Row(name="Alice", age=11).asDict() == {'name': 'Alice', 'age': 11}
        True
        >>> row = Row(key=1, value=Row(name='a', age=2))
        >>> row.asDict() == {'key': 1, 'value': Row(name='a', age=2)}
        True
        >>> row.asDict(True) == {'key': 1, 'value': {'name': 'a', 'age': 2}}
        True
        """
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
        """create new Row object"""
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
        try:
            index = self.__fields__.index(item)
            return super().__getitem__(index)
        except IndexError:
            from pyspark.errors import PySparkKeyError

            raise PySparkKeyError(errorClass="KEY_NOT_EXISTS", messageParameters={"key": str(item)})
        except ValueError:
            from pyspark.errors import PySparkValueError

            raise PySparkValueError(item)

    def __getattr__(self, item):
        if item.startswith("__"):
            from pyspark.errors import PySparkAttributeError

            raise PySparkAttributeError(
                errorClass="ATTRIBUTE_NOT_SUPPORTED", messageParameters={"attr_name": item}
            )
        try:
            index = self.__fields__.index(item)
            return self[index]
        except IndexError:
            from pyspark.errors import PySparkAttributeError

            raise PySparkAttributeError(
                errorClass="ATTRIBUTE_NOT_SUPPORTED", messageParameters={"attr_name": item}
            )
        except ValueError:
            from pyspark.errors import PySparkAttributeError

            raise PySparkAttributeError(
                errorClass="ATTRIBUTE_NOT_SUPPORTED", messageParameters={"attr_name": item}
            )

    def __setattr__(self, key, value):
        if key != "__fields__":
            from pyspark.errors import PySparkRuntimeError

            raise PySparkRuntimeError(errorClass="READ_ONLY", messageParameters={"object": "Row"})
        self.__dict__[key] = value

    def __reduce__(self):
        """Returns a tuple so Python knows how to pickle Row."""
        if hasattr(self, "__fields__"):
            from pyspark.sql.types import _create_row

            return _create_row, (self.__fields__, tuple(self))
        return tuple.__reduce__(self)

    def __repr__(self):
        """Printable representation of Row used in Python REPL."""
        if hasattr(self, "__fields__"):
            return "Row(%s)" % ", ".join(
                "%s=%r" % (key, value) for key, value in zip(self.__fields__, tuple(self))
            )
        return "<Row(%s)>" % ", ".join(repr(value) for value in self)
