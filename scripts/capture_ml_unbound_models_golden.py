#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture, on official PySpark, the models of no server-side object of every model class of
`JavaModel(java_model=None)`, which python/tests/_ml_unbound_cases.py records, for
tests/golden/ml_unbound_models.json, which test_ml_api_e2e.py compares the native classes with.

    SPARK_REMOTE=sc://localhost:15013 python scripts/capture_ml_unbound_models_golden.py

Run it with official pyspark-client 4.3.0 against a Spark 4.3.0 Connect server.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "tests"))

import pyspark  # noqa: E402
from _ml_unbound_cases import GOLDEN, MODELS, outcomes  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402


def main():
    spark = SparkSession.builder.remote(os.environ["SPARK_REMOTE"]).getOrCreate()
    try:
        golden = {
            "source": f"official pyspark-client {pyspark.__version__}, Spark {spark.version}",
            "models": {name: outcomes(spark, name) for name in MODELS},
        }
    finally:
        spark.stop()
    GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN}: {len(golden['models'])} models")


if __name__ == "__main__":
    main()
