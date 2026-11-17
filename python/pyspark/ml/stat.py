# SPDX-License-Identifier: Apache-2.0
"""Standard Spark ML statistics backed by the native Rust client."""

# The public names of PySpark's module, which `from pyspark.ml.stat import *` imports, as the module
# has no `__all__`: its classes, and the names of its imports.
import sys  # noqa: F401
from typing import TYPE_CHECKING, Optional, Tuple  # noqa: F401, UP035

from pyspark import since  # noqa: F401
from pyspark._pyspark import (
    ChiSquareTest,  # noqa: F401
    Correlation,  # noqa: F401
    KolmogorovSmirnovTest,  # noqa: F401
    Summarizer,  # noqa: F401
    _register_public_class,
)
from pyspark._pyspark import SummaryBuilder as _SummaryBuilder
from pyspark.ml._native_classes import _del
from pyspark.ml.common import _java2py, _py2java  # noqa: F401
from pyspark.ml.linalg import Matrix, Vector
from pyspark.ml.util import invoke_helper_relation  # noqa: F401
from pyspark.ml.wrapper import JavaWrapper, _jvm  # noqa: F401
from pyspark.sql.column import Column  # noqa: F401
from pyspark.sql.dataframe import DataFrame  # noqa: F401
from pyspark.sql.functions import lit  # noqa: F401
from pyspark.sql.types import ArrayType, DoubleType  # noqa: F401
from pyspark.sql.utils import is_remote  # noqa: F401

if TYPE_CHECKING:
    from py4j.java_gateway import JavaObject


# The public class of the native SummaryBuilder, a JavaWrapper as PySpark's, of PySpark's docstring
# and constructor over Spark Connect: `Summarizer.metrics` makes its objects, of the `_metrics` and
# `_java_obj` of their attributes, which `summary` reads, as PySpark's does.
class SummaryBuilder(_SummaryBuilder, JavaWrapper):
    """
    A builder object that provides summary statistics about a given column.

    Users should not directly create such builders, but instead use one of the methods in
    :py:class:`pyspark.ml.stat.Summarizer`

    .. versionadded:: 2.4.0

    """

    def __init__(self, jSummaryBuilder: "JavaObject"):
        # As PySpark's over Spark Connect, which keeps no Java object: a builder of no `_metrics`
        # raises the AttributeError of its `summary`.
        pass

    __del__ = _del


_register_public_class(_SummaryBuilder, SummaryBuilder)


# PySpark's plain holder of Apache Spark 4.3.0, kept as it is so that its instances, their
# attributes and their pickles are PySpark's own (a holder of the objects that it is given).
class MultivariateGaussian:
    """Represents a (mean, cov) tuple

    .. versionadded:: 3.0.0

    Examples
    --------
    >>> from pyspark.ml.linalg import DenseMatrix, Vectors
    >>> from pyspark.ml.stat import MultivariateGaussian
    >>> m = MultivariateGaussian(Vectors.dense([11,12]), DenseMatrix(2, 2, (1.0, 3.0, 5.0, 2.0)))
    >>> (m.mean, m.cov.toArray())
    (DenseVector([11.0, 12.0]), array([[ 1.,  5.],
           [ 3.,  2.]]))
    """

    def __init__(self, mean: Vector, cov: Matrix):
        self.mean = mean
        self.cov = cov
