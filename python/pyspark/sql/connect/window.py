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
# Window / WindowSpec under the connect import path, as PySpark Connect's
# ``pyspark.sql.connect.window``: its ``Window`` is a class of its own, a subclass of PySpark's
# classic ``pyspark.sql.window.Window``, and its ``WindowSpec`` is the native class of the window
# specifications (whose ``_frame`` is a ``WindowFrame``), a virtual subclass of the classic
# ``pyspark.sql.window.WindowSpec``, as PySpark Connect's is its subclass
# (pyspark.sql._classic_bases).
from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

from pyspark import _pyspark
from pyspark._pyspark import WindowSpec
from pyspark.sql.window import Window as _ParentWindow

if TYPE_CHECKING:
    from pyspark.sql.connect._typing import ColumnOrName

__all__ = ["Window", "WindowSpec"]


# Verbatim from PySpark Connect (pyspark.sql.connect.window.WindowFrame): the frame of a window
# specification, which its `_frame` gives.
class WindowFrame:
    def __init__(self, isRowFrame: bool, start: int, end: int) -> None:
        super().__init__()

        assert isinstance(isRowFrame, bool)
        assert isinstance(start, int)
        assert isinstance(end, int)
        self._isRowFrame = isRowFrame
        self._start = start
        self._end = end

    def __repr__(self) -> str:
        if self._isRowFrame:
            return f"WindowFrame(ROW_FRAME, {self._start}, {self._end})"
        else:
            return f"WindowFrame(RANGE_FRAME, {self._start}, {self._end})"


# Verbatim from PySpark Connect (pyspark.sql.connect.window.Window): a subclass of PySpark's classic
# Window, whose static methods dispatch to these, and of `_spec`, the specification of no
# partitioning, ordering and frame, here the native one, of which its builders start.
class Window(_ParentWindow):
    _spec = _pyspark._empty_window_spec()

    @staticmethod
    def partitionBy(*cols: Union["ColumnOrName", Sequence["ColumnOrName"]]) -> "WindowSpec":
        return Window._spec.partitionBy(*cols)

    @staticmethod
    def orderBy(*cols: Union["ColumnOrName", Sequence["ColumnOrName"]]) -> "WindowSpec":
        return Window._spec.orderBy(*cols)

    @staticmethod
    def rowsBetween(start: int, end: int) -> "WindowSpec":
        return Window._spec.rowsBetween(start, end)

    @staticmethod
    def rangeBetween(start: int, end: int) -> "WindowSpec":
        return Window._spec.rangeBetween(start, end)

