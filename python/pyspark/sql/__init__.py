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


def is_remote() -> bool:
    """This client is always a Spark Connect (remote) client.

    Mirrors ``pyspark.sql.is_remote``; some modules (e.g. ``pyspark.resource``)
    import it to branch on Connect mode.
    """
    return True


__all__ = [
    "SparkSession",
    "DataFrame",
    "Column",
    "Row",
    "functions",
    "is_remote",
]
