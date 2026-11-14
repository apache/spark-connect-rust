#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture how official PySpark 4.3.0 serializes the arguments of the pyspark.ml.stat helper calls.

ChiSquareTest.test, Correlation.corr and KolmogorovSmirnovTest.test are helper calls
(`invoke_helper_relation` of `pyspark.ml.util`), which serialize their arguments with `serialize`
of `pyspark.ml.connect.serialize`: a DataFrame as its relation, and another value as a literal.
Of an argument that cannot be a literal (an object, a pandas DataFrame), the call raises the error
of that conversion. Of a literal of another type than the helper takes (a dataset that is not a
DataFrame, a column name that is not a str), the call returns a lazy DataFrame, and the server
rejects the helper call when the DataFrame is collected.

For each case of CASES, the golden records the error of the call, or the arguments that the call
serialized (the relation of the DataFrame as {"input": true}, and another argument as its
literal) and the error of collect(). The tests of python/tests/test_statistics_{offline,e2e}.py
compare the native classes with it.

Run it with official pyspark-client 4.3.0 against a Spark 4.3.0 Connect server, e.g.
    python scripts/capture_statistics_arguments_golden.py --remote sc://localhost:15002
"""

import argparse
import base64
import json
from pathlib import Path

import pyspark
from pyspark.ml import stat
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]

# The arguments of each case, after the operation's name. A str that starts with "$" is a value
# that JSON does not hold: "$frame", a DataFrame; "$object", an object(); and "$pandas", a pandas
# DataFrame. The other values are themselves.
CASES = [
    ("chi", [None, "features", "label"]),
    ("chi", [5, "features", "label"]),
    ("chi", ["table", "features", "label"]),
    ("chi", [[1, 2], "features", "label"]),
    ("chi", [[], "features", "label"]),
    ("chi", ["$pandas", "features", "label"]),
    ("chi", ["$object", "features", "label"]),
    ("chi", ["$frame", 1, "label"]),
    ("chi", ["$frame", None, "label"]),
    ("chi", ["$frame", ["features"], "label"]),
    ("chi", ["$frame", "$object", "label"]),
    ("chi", ["$frame", "features", 1]),
    ("chi", ["$frame", "features", 1.5]),
    ("chi", ["$frame", "features", None]),
    ("chi", ["$frame", "features", "label", 1]),
    ("chi", ["$frame", "features", "label", "yes"]),
    ("chi", ["$frame", "features", "label", None]),
    ("chi", ["$frame", "features", "label", "$object"]),
    ("corr", [None, "features"]),
    ("corr", ["$pandas", "features"]),
    ("corr", ["$frame", 1]),
    ("corr", ["$frame", None]),
    ("corr", ["$frame", "features", 1]),
    ("corr", ["$frame", "features", None]),
    ("corr", ["$frame", "features", "$object"]),
    ("ks", [None, "sample", "norm"]),
    ("ks", ["$pandas", "sample", "norm"]),
    ("ks", ["$frame", 1, "norm"]),
    ("ks", ["$frame", None, "norm"]),
    ("ks", ["$frame", "sample", 1]),
    ("ks", ["$frame", "sample", None]),
    ("ks", ["$frame", "sample", "$object"]),
    ("ks", ["$frame", "sample", {"a": 1}]),
    ("ks", ["$frame", "sample", "norm", "x"]),
    ("ks", ["$frame", "sample", "norm", None]),
]

OPERATIONS = {
    "chi": ("chiSquareTest", lambda *args: stat.ChiSquareTest.test(*args)),
    "corr": ("correlation", lambda *args: stat.Correlation.corr(*args)),
    "ks": ("kolmogorovSmirnovTest", lambda *args: stat.KolmogorovSmirnovTest.test(*args)),
}


def value(spec, frame):
    """The argument of a case's spec: a DataFrame, an object() or a pandas DataFrame of a "$"
    marker, else the spec itself."""
    if spec == "$frame":
        return frame
    if spec == "$object":
        return object()
    if spec == "$pandas":
        import pandas as pd

        return pd.DataFrame({"features": [1.0]})
    return spec


def error(e):
    """The class and the first line of the message of an error."""
    lines = str(e).strip().splitlines()
    return {"class": type(e).__name__, "message": lines[0] if lines else ""}


def encoded(message):
    return base64.b64encode(message.SerializeToString(deterministic=True)).decode()


def capture(spark, op, specs):
    method, call = OPERATIONS[op]
    frame = spark.range(1)
    case = {"op": op, "args": specs, "method": method}
    try:
        result = call(*(value(spec, frame) for spec in specs))
    except Exception as e:  # noqa: BLE001
        case["error"] = error(e)
        return case
    fetch = result._plan.plan(spark.client).ml_relation.fetch
    assert fetch.obj_ref.id == "______ML_CONNECT_HELPER______"
    (call_method,) = fetch.methods
    assert call_method.method == method
    case["sent"] = [
        {"input": True} if arg.HasField("input") else {"literal": encoded(arg.param)}
        for arg in call_method.args
    ]
    try:
        result.collect()
    except Exception as e:  # noqa: BLE001
        case["collect_error"] = error(e)
    else:
        case["collect_error"] = None
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=ROOT / "tests/golden/statistics_arguments.json")
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    try:
        golden = {
            "source": f"official pyspark-client {pyspark.__version__}, Spark {spark.version}",
            "cases": [capture(spark, op, specs) for op, specs in CASES],
        }
    finally:
        spark.stop()
    args.out.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {len(golden['cases'])} cases")


if __name__ == "__main__":
    main()
