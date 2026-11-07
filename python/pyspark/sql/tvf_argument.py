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
"""``TableValuedFunctionArgument`` of PySpark, the base of the arguments of table-valued
functions, of which a ``Column`` and a ``TableArg`` are instances. The drop-in's are Rust-backed
classes, so they are its virtual subclasses."""

import abc


class TableValuedFunctionArgument(abc.ABC):
    """
    Base class for arguments passed to Table-Valued Functions.

    This class is intentionally left empty and serves as a marker to achieve
    parity with the Scala `TableValuedFunctionArgument` trait.
    """

    pass


from pyspark._pyspark import Column as _Column, TableArg as _TableArg  # noqa: E402

TableValuedFunctionArgument.register(_Column)
TableValuedFunctionArgument.register(_TableArg)

__all__ = ["TableValuedFunctionArgument"]
