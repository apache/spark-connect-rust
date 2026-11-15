# SPDX-License-Identifier: Apache-2.0
"""Native ALS recommendation through Spark Connect."""

from pyspark._pyspark import ALS, ALSModel

__all__ = ["ALS", "ALSModel"]

# The public classes of the native classes (see pyspark.ml._native_classes).
from pyspark.ml._native_classes import publish as _publish  # noqa: E402

_publish(globals(), "pyspark.ml.recommendation")
