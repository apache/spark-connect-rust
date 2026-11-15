#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture how official PySpark 4.3.0 serializes a `(value, DataType)` pair as an argument of an ML
call, for tests/golden/ml_fetch_pairs.json.

`serialize` of `pyspark.ml.connect.serialize` sends such a pair, a tuple of two whose second item is
a `DataType`, as `LiteralExpression(value, dataType)`. Its asserts raise a bare AssertionError of a
type of another class than a literal takes, and of a value that is not of the type. Another tuple is
the literal of `LiteralExpression._from_value`, or the error of that conversion. For each case of
python/tests/_ml_fetch_pair_cases.py, the golden records the literal that `ChiSquareTest.test`
serializes of the value as its dataset, or the error of the call. test_ml_offline.py compares the
native classes with it.

Run it with official pyspark-client 4.3.0 against a Spark 4.3.0 Connect server, e.g.
    python scripts/capture_ml_fetch_pairs_golden.py --remote sc://localhost:15002
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "tests"))

import pyspark  # noqa: E402
from _ml_fetch_pair_cases import CASES, GOLDEN, outcome  # noqa: E402
from pyspark.ml.stat import ChiSquareTest  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=GOLDEN)
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    try:
        cases = [
            {
                "value": case,
                **outcome(
                    case,
                    lambda value: ChiSquareTest.test(value, "features", "label"),
                    lambda df: df._plan.plan(spark.client),
                ),
            }
            for case in CASES
        ]
        golden = {
            "source": f"official pyspark-client {pyspark.__version__}, Spark {spark.version}",
            "cases": cases,
        }
    finally:
        spark.stop()
    args.out.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {len(cases)} cases")


if __name__ == "__main__":
    main()
