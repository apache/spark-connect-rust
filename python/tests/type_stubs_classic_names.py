"""Names of classic PySpark, which test_type_stubs_offline.py type-checks with mypy and pyright and
never runs.

PySpark's ``pyspark/__init__.py`` imports SparkContext, RDD and the other classes of a JVM from
``pyspark.core`` when not ``is_remote_only()``, which holds of neither client at run time and which
type checkers do not evaluate, and the official pyspark-client does not ship ``pyspark.core``: so
mypy takes the names of no type (``Any``), pyright reports the imports of them as of an unknown
symbol and the attributes as ``Unknown``, and the classic SQLContext, which PySpark's module of
pyspark.sql.context declares, is typed. With the official pyspark-client 4.3.0 the diagnostics
are those that the tests expect, and so with the drop-in.
"""

import pyspark
from pyspark import (  # noqa: F401
    RDD,
    Broadcast,
    RDDBarrier,
    SparkContext,
    SparkExecutorInfo,
    SparkFiles,
    SparkJobInfo,
    SparkStageInfo,
    StatusTracker,
)
from pyspark.sql import SQLContext

reveal_type(SparkContext)  # noqa: F821
reveal_type(pyspark.SparkContext)  # noqa: F821
reveal_type(pyspark.RDD)  # noqa: F821
context = SQLContext.getOrCreate()
reveal_type(context)  # noqa: F821
reveal_type(context.sql("SELECT 1"))  # noqa: F821
reveal_type(context.read)  # noqa: F821
