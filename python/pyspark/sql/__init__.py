#pyspark.sql module exposing the Rust-backed client
#
# Mirrors pyspark.sql.* public API

from pyspark._pyspark import (
    SparkSession,
    SparkSession as SparkSessionBuilder,  # reuse for now
    DataFrame,
    Column,
    Row,
)

import pyspark.sql.functions as functions

__all__ = [
    "SparkSession",
    "DataFrame",
    "Column",
    "Row",
    "functions",
]
