"""A PySpark application, which test_type_stubs_offline.py type-checks with mypy and never runs.

Against the official pyspark-client 4.3.0, mypy reports only the deliberate error of the last
line, and reveals PySpark's types: ``DataFrame``, ``def (col: str) -> Column``, the reader's
signature and the ``Plan`` message of Spark Connect. So does it against the drop-in, whose stubs
(scripts/gen_type_stubs.py) show it PySpark's API. Without them, mypy found no ``desc``,
``row_number``, ``upper``, ``sum`` or ``first`` in pyspark.sql.functions and no message in
pyspark.sql.connect.proto, everything else was ``Any``, and the deliberate error, which an ``Any``
hides, was not reported.
"""

import pyspark.sql.connect.proto as pb2
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

spark = SparkSession.builder.remote("sc://localhost").getOrCreate()
people = spark.read.json("people.json")
by_salary = Window.partitionBy("dept").orderBy(F.desc("salary"))
ranked = people.withColumn("rank", F.row_number().over(by_salary))
totals = ranked.groupBy("dept").agg(F.sum("salary").alias("total"), F.upper(F.first("name")))

reveal_type(totals)  # noqa: F821
reveal_type(F.col)  # noqa: F821
reveal_type(spark.read.json)  # noqa: F821
reveal_type(pb2.Plan(root=pb2.Relation(range=pb2.Range(end=10))))  # noqa: F821

# The deliberate error: count() returns an int.
rows: str = totals.count()
