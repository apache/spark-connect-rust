# SPDX-License-Identifier: Apache-2.0
"""Standard Spark ML statistics backed by the native Rust client."""

from pyspark._pyspark import (
    ChiSquareTest,
    Correlation,
    KolmogorovSmirnovTest,
    Summarizer,
    SummaryBuilder,
)
from pyspark.ml.linalg import Matrix, Vector

__all__ = [
    "ChiSquareTest",
    "Correlation",
    "KolmogorovSmirnovTest",
    "Summarizer",
    "SummaryBuilder",
    "MultivariateGaussian",
]


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
