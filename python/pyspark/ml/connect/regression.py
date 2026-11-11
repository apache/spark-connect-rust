"""Native regression estimators backed by Spark Connect's JVM ML API."""
from pyspark._pyspark import LinearRegression, RandomForestRegressor

__all__ = ["LinearRegression", "RandomForestRegressor"]
