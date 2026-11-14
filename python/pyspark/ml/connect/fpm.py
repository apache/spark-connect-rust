# SPDX-License-Identifier: Apache-2.0
"""Native Spark Connect frequent itemset and sequence mining."""
from pyspark._pyspark import FPGrowth, FPGrowthModel, PrefixSpan
from pyspark.ml.param import Params

# PrefixSpan is Params only (it mines with no model), so it is not registered with the estimators
# and models of `pyspark.ml.connect`; it is a `Params` of PySpark's, as its class is.
Params.register(PrefixSpan)

__all__ = ["FPGrowth", "FPGrowthModel", "PrefixSpan"]
