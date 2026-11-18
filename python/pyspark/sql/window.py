# SPDX-License-Identifier: Apache-2.0
"""Window API mirroring pyspark.sql.window. Its ``__all__`` is PySpark's; the module has the
native ``FrameBound`` too, the class of the frame bounds that ``rowsBetween`` and
``rangeBetween`` take besides ints."""

from pyspark._pyspark import FrameBound, Window, WindowSpec  # noqa: F401

__all__ = ["Window", "WindowSpec"]
