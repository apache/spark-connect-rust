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

"""The plain value classes of PySpark's ``pyspark.sql.types``: ``VariantVal``, ``Geography``
and ``Geometry``.

Mirrors these classes in apache/spark v4.3.0, ``python/pyspark/sql/types.py``, whose
``__init__`` stores its arguments, as they are given, as instance attributes, which pickle as
those of any plain class, also into the server's Python workers. Embedded by the extension so
``pyspark._pyspark`` and the public imports refer to the same classes, which the client builds
when it collects values, without importing the partially initialized types shim.
"""

from typing import Any


class VariantVal:
    """
    A class to represent a Variant value in Python.

    .. versionadded:: 4.0.0

    Parameters
    ----------
    value : bytes
        The bytes representing the value component of the Variant.
    metadata : bytes
        The bytes representing the metadata component of the Variant.

    Methods
    -------
    toPython()
        Convert the VariantVal to a Python data structure.

    Examples
    --------
    >>> from pyspark.sql import functions as sf
    >>> df = spark.createDataFrame([ {'json': '''{ "a" : 1 }'''} ])
    >>> v = df.select(sf.parse_json(df.json).alias("var")).head().var
    >>> v.toPython()
    {'a': 1}
    >>> v.toJson()
    '{"a":1}'
    """

    __module__ = "pyspark.sql.types"

    def __init__(self, value: bytes, metadata: bytes):
        self.value = value
        self.metadata = metadata

    def __str__(self) -> str:
        from pyspark.sql.variant_utils import VariantUtils

        return VariantUtils.to_json(self.value, self.metadata)

    def __repr__(self) -> str:
        return "VariantVal(%r, %r)" % (self.value, self.metadata)

    def toPython(self) -> Any:
        """
        Convert the VariantVal to a Python data structure.

        Returns
        -------
        Any
            A Python object that represents the Variant.
        """
        from pyspark.sql.variant_utils import VariantUtils

        return VariantUtils.to_python(self.value, self.metadata)

    def toJson(self, zone_id: str = "UTC") -> str:
        """
        Convert the VariantVal to a JSON string. The zone ID represents the time zone that the
        timestamp should be printed in. It is defaulted to UTC. The list of valid zone IDs can be
        found by importing the `zoneinfo` module and running :code:`zoneinfo.available_timezones()`.

        Returns
        -------
        str
            A JSON string that represents the Variant.
        """
        from pyspark.sql.variant_utils import VariantUtils

        return VariantUtils.to_json(self.value, self.metadata, zone_id)

    @classmethod
    def parseJson(cls, json_str: str) -> "VariantVal":
        """
        Parse a JSON string and construct a VariantVal.

        Returns
        -------
        VariantVal
            A VariantVal representing the parsed Variant value.
        """
        from pyspark.sql.variant_utils import VariantUtils

        value, metadata = VariantUtils.parse_json(json_str)
        return VariantVal(value, metadata)


class Geography:
    """
    A class to represent a Geography value in Python.

    .. versionadded:: 4.1.0

    Parameters
    ----------
    wkb : bytes
        The bytes representing the WKB of Geography.

    srid : integer
        The integer value representing SRID of Geography.

    Methods
    -------
    getBytes()
        Returns the WKB of Geography.

    getSrid()
        Returns the SRID of Geography.

    Examples
    --------
    >>> g = Geography.fromWKB(bytes.fromhex('010100000000000000000031400000000000001c40'), 4326)
    >>> g.getBytes().hex()
    '010100000000000000000031400000000000001c40'
    >>> g.getSrid()
    4326
    """

    __module__ = "pyspark.sql.types"

    def __init__(self, wkb: bytes, srid: int):
        self.wkb = wkb
        self.srid = srid

    def __str__(self) -> str:
        return "Geography(%r, %d)" % (self.wkb, self.srid)

    def __repr__(self) -> str:
        return "Geography(%r, %d)" % (self.wkb, self.srid)

    def getSrid(self) -> int:
        """
        Returns the SRID of Geography.
        """
        return self.srid

    def getBytes(self) -> bytes:
        """
        Returns the WKB of Geography.
        """
        return self.wkb

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Geography):
            # Don't attempt to compare against unrelated types.
            return NotImplemented

        return self.wkb == other.wkb and self.srid == other.srid

    @classmethod
    def fromWKB(cls, wkb: bytes, srid: int) -> "Geography":
        """
        Construct Python Geography object from WKB.
        :return: Python representation of the Geography type value.
        """
        return Geography(wkb, srid)


class Geometry:
    """
    A class to represent a Geometry value in Python.

    .. versionadded:: 4.1.0

    Parameters
    ----------
    wkb : bytes
        The bytes representing the WKB of Geometry.

    srid : integer
        The integer value representing SRID of Geometry.

    Methods
    -------
    getBytes()
        Returns the WKB of Geometry.

    getSrid()
        Returns the SRID of Geometry.

    Examples
    --------
    >>> g = Geometry.fromWKB(bytes.fromhex('010100000000000000000031400000000000001c40'), 0)
    >>> g.getBytes().hex()
    '010100000000000000000031400000000000001c40'
    >>> g.getSrid()
    0
    """

    __module__ = "pyspark.sql.types"

    def __init__(self, wkb: bytes, srid: int):
        self.wkb = wkb
        self.srid = srid

    def __str__(self) -> str:
        return "Geometry(%r, %d)" % (self.wkb, self.srid)

    def __repr__(self) -> str:
        return "Geometry(%r, %d)" % (self.wkb, self.srid)

    def getSrid(self) -> int:
        """
        Returns the SRID of Geometry.
        """
        return self.srid

    def getBytes(self) -> bytes:
        """
        Returns the WKB of Geometry.
        """
        return self.wkb

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Geometry):
            # Don't attempt to compare against unrelated types.
            return NotImplemented

        return self.wkb == other.wkb and self.srid == other.srid

    @classmethod
    def fromWKB(cls, wkb: bytes, srid: int) -> "Geometry":
        """
        Construct Python Geometry object from WKB.
        :return: Python representation of the Geometry type value.
        """
        return Geometry(wkb, srid)
