# SPDX-License-Identifier: Apache-2.0
"""Standard Spark ML statistics backed by the native Rust client."""

# The public names of PySpark's module, which `from pyspark.ml.stat import *` imports, as the module
# has no `__all__`: its classes, and the names of its imports.
import sys  # noqa: F401
from typing import TYPE_CHECKING, Optional, Tuple  # noqa: F401, UP035

from pyspark import since
from pyspark._pyspark import ChiSquareTest as _ChiSquareTest
from pyspark._pyspark import Correlation as _Correlation
from pyspark._pyspark import KolmogorovSmirnovTest as _KolmogorovSmirnovTest
from pyspark._pyspark import Summarizer as _Summarizer
from pyspark._pyspark import SummaryBuilder as _SummaryBuilder
from pyspark._pyspark import _register_public_class
from pyspark.ml._native_classes import _del_of
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


# The public classes of the native ChiSquareTest, Correlation, KolmogorovSmirnovTest and
# Summarizer, plain classes of static methods as PySpark's, which construct, copy and pickle as the
# native classes do (as `object.__new__` makes PySpark's objects), of PySpark's docstrings: their
# static methods are PySpark's functions of the module, of PySpark's signatures, annotations and
# docstrings, over the native static methods, so that `inspect` and `help` give PySpark's, and a
# call binds as it binds PySpark's function. Summarizer's `_check_param` and `_get_single_metric`
# are PySpark's own code over Spark Connect.
class ChiSquareTest(_ChiSquareTest):
    """
    Conduct Pearson's independence test for every feature against the label. For each feature,
    the (feature, label) pairs are converted into a contingency matrix for which the Chi-squared
    statistic is computed. All label and feature values must be categorical.

    The null hypothesis is that the occurrence of the outcomes is statistically independent.

    .. versionadded:: 2.2.0

    """

    @staticmethod
    def test(
        dataset: DataFrame, featuresCol: str, labelCol: str, flatten: bool = False
    ) -> DataFrame:
        """
        Perform a Pearson's independence test using dataset.

        .. versionadded:: 2.2.0
        .. versionchanged:: 3.1.0
           Added optional ``flatten`` argument.

        Parameters
        ----------
        dataset : :py:class:`pyspark.sql.DataFrame`
            DataFrame of categorical labels and categorical features.
            Real-valued features will be treated as categorical for each distinct value.
        featuresCol : str
            Name of features column in dataset, of type `Vector` (`VectorUDT`).
        labelCol : str
            Name of label column in dataset, of any numerical type.
        flatten : bool, optional
            if True, flattens the returned dataframe.

        Returns
        -------
        :py:class:`pyspark.sql.DataFrame`
            DataFrame containing the test result for every feature against the label.
            If flatten is True, this DataFrame will contain one row per feature with the following
            fields:

            - `featureIndex: int`
            - `pValue: float`
            - `degreesOfFreedom: int`
            - `statistic: float`

            If flatten is False, this DataFrame will contain a single Row with the following fields:

            - `pValues: Vector`
            - `degreesOfFreedom: Array[int]`
            - `statistics: Vector`

            Each of these fields has one value per feature.

        Examples
        --------
        >>> from pyspark.ml.linalg import Vectors
        >>> from pyspark.ml.stat import ChiSquareTest
        >>> dataset = [[0, Vectors.dense([0, 0, 1])],
        ...            [0, Vectors.dense([1, 0, 1])],
        ...            [1, Vectors.dense([2, 1, 1])],
        ...            [1, Vectors.dense([3, 1, 1])]]
        >>> dataset = spark.createDataFrame(dataset, ["label", "features"])
        >>> chiSqResult = ChiSquareTest.test(dataset, 'features', 'label')
        >>> chiSqResult.select("degreesOfFreedom").collect()[0]
        Row(degreesOfFreedom=[3, 1, 0])
        >>> chiSqResult = ChiSquareTest.test(dataset, 'features', 'label', True)
        >>> row = chiSqResult.orderBy("featureIndex").collect()
        >>> row[0].statistic
        4.0
        """
        return _ChiSquareTest.test(dataset, featuresCol, labelCol, flatten)


class Correlation(_Correlation):
    """
    Compute the correlation matrix for the input dataset of Vectors using the specified method.
    Methods currently supported: `pearson` (default), `spearman`.

    .. versionadded:: 2.2.0

    Notes
    -----
    For Spearman, a rank correlation, we need to create an RDD[Double] for each column
    and sort it in order to retrieve the ranks and then join the columns back into an RDD[Vector],
    which is fairly costly. Cache the input Dataset before calling corr with `method = 'spearman'`
    to avoid recomputing the common lineage.
    """

    @staticmethod
    def corr(dataset: DataFrame, column: str, method: str = "pearson") -> DataFrame:
        """
        Compute the correlation matrix with specified method using dataset.

        .. versionadded:: 2.2.0

        Parameters
        ----------
        dataset : :py:class:`pyspark.sql.DataFrame`
            A DataFrame.
        column : str
            The name of the column of vectors for which the correlation coefficient needs
            to be computed. This must be a column of the dataset, and it must contain
            Vector objects.
        method : str, optional
            String specifying the method to use for computing correlation.
            Supported: `pearson` (default), `spearman`.

        Returns
        -------
        A DataFrame that contains the correlation matrix of the column of vectors. This
        DataFrame contains a single row and a single column of name `METHODNAME(COLUMN)`.

        Examples
        --------
        >>> from pyspark.ml.linalg import DenseMatrix, Vectors
        >>> from pyspark.ml.stat import Correlation
        >>> dataset = [[Vectors.dense([1, 0, 0, -2])],
        ...            [Vectors.dense([4, 5, 0, 3])],
        ...            [Vectors.dense([6, 7, 0, 8])],
        ...            [Vectors.dense([9, 0, 0, 1])]]
        >>> dataset = spark.createDataFrame(dataset, ['features'])
        >>> pearsonCorr = Correlation.corr(dataset, 'features', 'pearson').collect()[0][0]
        >>> print(str(pearsonCorr).replace('nan', 'NaN'))
        DenseMatrix([[ 1.        ,  0.0556...,         NaN,  0.4004...],
                     [ 0.0556...,  1.        ,         NaN,  0.9135...],
                     [        NaN,         NaN,  1.        ,         NaN],
                     [ 0.4004...,  0.9135...,         NaN,  1.        ]])
        >>> spearmanCorr = Correlation.corr(dataset, 'features', method='spearman').collect()[0][0]
        >>> print(str(spearmanCorr).replace('nan', 'NaN'))
        DenseMatrix([[ 1.        ,  0.1054...,         NaN,  0.4       ],
                     [ 0.1054...,  1.        ,         NaN,  0.9486... ],
                     [        NaN,         NaN,  1.        ,         NaN],
                     [ 0.4       ,  0.9486... ,         NaN,  1.        ]])
        """
        return _Correlation.corr(dataset, column, method)


class KolmogorovSmirnovTest(_KolmogorovSmirnovTest):
    """
    Conduct the two-sided Kolmogorov Smirnov (KS) test for data sampled from a continuous
    distribution.

    By comparing the largest difference between the empirical cumulative
    distribution of the sample data and the theoretical distribution we can provide a test for the
    the null hypothesis that the sample data comes from that theoretical distribution.

    .. versionadded:: 2.4.0

    """

    @staticmethod
    def test(dataset: DataFrame, sampleCol: str, distName: str, *params: float) -> DataFrame:
        """
        Conduct a one-sample, two-sided Kolmogorov-Smirnov test for probability distribution
        equality. Currently supports the normal distribution, taking as parameters the mean and
        standard deviation.

        .. versionadded:: 2.4.0

        Parameters
        ----------
        dataset : :py:class:`pyspark.sql.DataFrame`
            a Dataset or a DataFrame containing the sample of data to test.
        sampleCol : str
            Name of sample column in dataset, of any numerical type.
        distName : str
            a `string` name for a theoretical distribution, currently only support "norm".
        params : float
            a list of `float` values specifying the parameters to be used for the theoretical
            distribution. For "norm" distribution, the parameters includes mean and variance.

        Returns
        -------
        A DataFrame that contains the Kolmogorov-Smirnov test result for the input sampled data.
        This DataFrame will contain a single Row with the following fields:

        - `pValue: Double`
        - `statistic: Double`

        Examples
        --------
        >>> from pyspark.ml.stat import KolmogorovSmirnovTest
        >>> dataset = [[-1.0], [0.0], [1.0]]
        >>> dataset = spark.createDataFrame(dataset, ['sample'])
        >>> ksResult = KolmogorovSmirnovTest.test(dataset, 'sample', 'norm', 0.0, 1.0).first()
        >>> round(ksResult.pValue, 3)
        1.0
        >>> round(ksResult.statistic, 3)
        0.175
        >>> dataset = [[2.0], [3.0], [4.0]]
        >>> dataset = spark.createDataFrame(dataset, ['sample'])
        >>> ksResult = KolmogorovSmirnovTest.test(dataset, 'sample', 'norm', 3.0, 1.0).first()
        >>> round(ksResult.pValue, 3)
        1.0
        >>> round(ksResult.statistic, 3)
        0.175
        """
        return _KolmogorovSmirnovTest.test(dataset, sampleCol, distName, *params)


class Summarizer(_Summarizer):
    """
    Tools for vectorized statistics on MLlib Vectors.
    The methods in this package provide various statistics for Vectors contained inside DataFrames.
    This class lets users pick the statistics they would like to extract for a given column.

    .. versionadded:: 2.4.0

    Examples
    --------
    >>> from pyspark.ml.stat import Summarizer
    >>> from pyspark.sql import Row
    >>> from pyspark.ml.linalg import Vectors
    >>> summarizer = Summarizer.metrics("mean", "count")
    >>> df = sc.parallelize([Row(weight=1.0, features=Vectors.dense(1.0, 1.0, 1.0)),
    ...                      Row(weight=0.0, features=Vectors.dense(1.0, 2.0, 3.0))]).toDF()
    >>> df.select(summarizer.summary(df.features, df.weight)).show(truncate=False)
    +-----------------------------------+
    |aggregate_metrics(features, weight)|
    +-----------------------------------+
    |{[1.0,1.0,1.0], 1}                 |
    +-----------------------------------+
    >>> df.select(summarizer.summary(df.features)).show(truncate=False)
    +--------------------------------+
    |aggregate_metrics(features, 1.0)|
    +--------------------------------+
    |{[1.0,1.5,2.0], 2}              |
    +--------------------------------+
    >>> df.select(Summarizer.mean(df.features, df.weight)).show(truncate=False)
    +--------------+
    |mean(features)|
    +--------------+
    |[1.0,1.0,1.0] |
    +--------------+
    >>> df.select(Summarizer.mean(df.features)).show(truncate=False)
    +--------------+
    |mean(features)|
    +--------------+
    |[1.0,1.5,2.0] |
    +--------------+
    """

    @staticmethod
    @since("2.4.0")
    def mean(col: Column, weightCol: Optional[Column] = None) -> Column:
        """
        return a column of mean summary
        """
        return _Summarizer.mean(col, weightCol)

    @staticmethod
    @since("3.0.0")
    def sum(col: Column, weightCol: Optional[Column] = None) -> Column:
        """
        return a column of sum summary
        """
        return _Summarizer.sum(col, weightCol)

    @staticmethod
    @since("2.4.0")
    def variance(col: Column, weightCol: Optional[Column] = None) -> Column:
        """
        return a column of variance summary
        """
        return _Summarizer.variance(col, weightCol)

    @staticmethod
    @since("3.0.0")
    def std(col: Column, weightCol: Optional[Column] = None) -> Column:
        """
        return a column of std summary
        """
        return _Summarizer.std(col, weightCol)

    @staticmethod
    @since("2.4.0")
    def count(col: Column, weightCol: Optional[Column] = None) -> Column:
        """
        return a column of count summary
        """
        return _Summarizer.count(col, weightCol)

    @staticmethod
    @since("2.4.0")
    def numNonZeros(col: Column, weightCol: Optional[Column] = None) -> Column:
        """
        return a column of numNonZero summary
        """
        return _Summarizer.numNonZeros(col, weightCol)

    @staticmethod
    @since("2.4.0")
    def max(col: Column, weightCol: Optional[Column] = None) -> Column:
        """
        return a column of max summary
        """
        return _Summarizer.max(col, weightCol)

    @staticmethod
    @since("2.4.0")
    def min(col: Column, weightCol: Optional[Column] = None) -> Column:
        """
        return a column of min summary
        """
        return _Summarizer.min(col, weightCol)

    @staticmethod
    @since("2.4.0")
    def normL1(col: Column, weightCol: Optional[Column] = None) -> Column:
        """
        return a column of normL1 summary
        """
        return _Summarizer.normL1(col, weightCol)

    @staticmethod
    @since("2.4.0")
    def normL2(col: Column, weightCol: Optional[Column] = None) -> Column:
        """
        return a column of normL2 summary
        """
        return _Summarizer.normL2(col, weightCol)

    # Of PySpark's annotation, of `typing.Tuple`.
    @staticmethod
    def _check_param(
        featuresCol: Column, weightCol: Optional[Column]
    ) -> Tuple[Column, Column]:  # noqa: UP006
        if weightCol is None:
            weightCol = lit(1.0)
        if not isinstance(featuresCol, Column) or not isinstance(weightCol, Column):
            raise TypeError("featureCol and weightCol should be a Column")
        return featuresCol, weightCol

    @staticmethod
    def _get_single_metric(col: Column, weightCol: Optional[Column], metric: str) -> Column:
        col, weightCol = Summarizer._check_param(col, weightCol)
        # The alias name maybe different from the one in Spark Classic,
        # because we cannot get the same string representation of the Column object.
        return (
            Summarizer.metrics(metric)
            .summary(col, weightCol)
            .getField(metric)
            .alias(f"{metric}({col._expr})")
        )

    @staticmethod
    def metrics(*metrics: str) -> "SummaryBuilder":
        """
        Given a list of metrics, provides a builder that it turns computes metrics from a column.

        See the documentation of :py:class:`Summarizer` for an example.

        The following metrics are accepted (case sensitive):
         - mean: a vector that contains the coefficient-wise mean.
         - sum: a vector that contains the coefficient-wise sum.
         - variance: a vector that contains the coefficient-wise variance.
         - std: a vector that contains the coefficient-wise standard deviation.
         - count: the count of all vectors seen.
         - numNonzeros: a vector with the number of non-zeros for each coefficients
         - max: the maximum for each coefficient.
         - min: the minimum for each coefficient.
         - normL2: the Euclidean norm for each coefficient.
         - normL1: the L1 norm of each coefficient (sum of the absolute values).

        .. versionadded:: 2.4.0

        Notes
        -----
        Currently, the performance of this interface is about 2x~3x slower than using the RDD
        interface.

        Examples
        --------
        metrics : str
            metrics that can be provided.

        Returns
        -------
        :py:class:`pyspark.ml.stat.SummaryBuilder`
        """
        return _Summarizer.metrics(*metrics)


_register_public_class(_ChiSquareTest, ChiSquareTest)
_register_public_class(_Correlation, Correlation)
_register_public_class(_KolmogorovSmirnovTest, KolmogorovSmirnovTest)
_register_public_class(_Summarizer, Summarizer)


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

    # PySpark's function, of its signature and docstring, over the native method: so a call binds
    # as PySpark's does (`SummaryBuilder.summary() takes from 2 to 3 positional arguments but 4
    # were given`).
    def summary(self, featuresCol: Column, weightCol: Optional[Column] = None) -> Column:
        """
        Returns an aggregate object that contains the summary of the column with the requested
        metrics.

        .. versionadded:: 2.4.0

        Parameters
        ----------
        featuresCol : str
            a column that contains features Vector object.
        weightCol : str, optional
            a column that contains weight value. Default weight is 1.0.

        Returns
        -------
        :py:class:`pyspark.sql.Column`
            an aggregate column that contains the statistics. The exact content of this
            structure is determined during the creation of the builder.
        """
        return _SummaryBuilder.summary(self, featuresCol, weightCol)

    __del__ = _del_of(JavaWrapper)


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
