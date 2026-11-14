#!/usr/bin/env python3
"""Capture the models, saves and loads of FPGrowth, ALS and the evaluators of official PySpark 4.3.0.

Run with the official pyspark-client on PYTHONPATH and a live Connect server:

    python scripts/capture_fpm_recommendation_evaluation_golden.py --remote sc://localhost:15002

The datasets and configurations are those of tests/golden/frequent_patterns.json, als.json and
evaluators.json, which capture the results of these classes. Here each FPGrowth and ALS case is
fitted, and its model gives the members of ATTRIBUTES and of `arguments`. The estimator, the model
and the first configuration of each evaluator are saved and loaded back, and each loaded model and
evaluator gives those members again. scripts/ml_golden_common.py records the values, errors and
requests.
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_golden_common import Golden  # noqa: E402
from pyspark.ml import evaluation, fpm, recommendation  # noqa: E402
from pyspark.ml.functions import array_to_vector  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FPM = json.loads((ROOT / "tests/golden/frequent_patterns.json").read_text())
ALS = json.loads((ROOT / "tests/golden/als.json").read_text())
EVALUATORS = json.loads((ROOT / "tests/golden/evaluators.json").read_text())

# Members without arguments: a property is read, and a name ending in "()" is called.
ATTRIBUTES = {
    "FPGrowthModel": ["freqItemsets", "associationRules"],
    "ALSModel": ["rank", "userFactors", "itemFactors"],
    "BinaryClassificationEvaluator": ["isLargerBetter()"],
    "RegressionEvaluator": ["isLargerBetter()"],
    "MulticlassClassificationEvaluator": ["isLargerBetter()"],
    "MultilabelClassificationEvaluator": ["isLargerBetter()"],
    "ClusteringEvaluator": ["isLargerBetter()"],
    "RankingEvaluator": ["isLargerBetter()"],
}

# The keyword setter of an estimator's or evaluator's params, which members() would call.
SKIP = {"setParams": "Params setter (client side)"}


def arguments(frames):
    """The calls with arguments of a class, of the frames of its case: held, users and items."""

    def calls(name, frame):
        if name == "FPGrowthModel":
            return [("transform", "transform", (frames["held"],))]
        if name == "ALSModel":
            return [
                ("transform", "transform", (frames["held"],)),
                ("recommendForAllUsers-2", "recommendForAllUsers", (2,)),
                ("recommendForAllItems-2", "recommendForAllItems", (2,)),
                ("recommendForUserSubset-2", "recommendForUserSubset", (frames["users"], 2)),
                ("recommendForItemSubset-2", "recommendForItemSubset", (frames["items"], 2)),
            ]
        return [("evaluate", "evaluate", (frame,))]

    return calls


def configured(cls, values):
    # The constructor takes the values of its signature, and the setters the rest (ALS takes no
    # predictionCol).
    signature = inspect.signature(cls.__init__).parameters
    stage = cls(**{k: v for k, v in values.items() if k in signature})
    for key, value in values.items():
        if key not in signature:
            getattr(stage, "set" + key[0].upper() + key[1:])(value)
    return stage


def register(golden, name, sql, vectors=(), coalesce=None):
    # A frame of the SQL of a fixture, rebuilt by the Rust tests of its spec.
    frame = golden.spark.sql(sql)
    for column in vectors:
        frame = frame.withColumn(column, array_to_vector(column))
    if coalesce is not None:
        frame = frame.coalesce(coalesce)
    spec = {"sql": sql, "vectors": list(vectors)}
    if coalesce is not None:
        spec["coalesce"] = coalesce
    return golden.register(name, frame, spec)


def model_case(golden, key, estimator, train, frames, uid):
    model = golden.fit(key, estimator, train, uid)
    cls_name = type(model).__name__
    calls = arguments(frames)
    golden.record_members(
        golden.models,
        key,
        model,
        cls_name,
        "model",
        attributes=ATTRIBUTES[cls_name],
        arguments=calls,
        frame=train,
    )
    golden.persist(
        key + ".model",
        model,
        cls_name,
        "model",
        attributes=ATTRIBUTES[cls_name],
        arguments=calls,
        frame=train,
    )
    golden.persist(key + ".estimator", estimator, type(estimator).__name__, "estimator")


def capture(golden):
    for case in FPM["fpgrowth"]:
        name = "fpm_" + case["id"].replace("-", "_")
        train = register(golden, name, case["sql"])
        frames = {"held": register(golden, name + "_held", case["held"])}
        estimator = configured(fpm.FPGrowth, case["values"])
        model_case(golden, "FPGrowth." + case["id"], estimator, train, frames, name)

    train = register(golden, "als_train", ALS["train"])
    frames = {
        "held": register(golden, "als_held", ALS["held"]),
        "users": register(golden, "als_users", ALS["users"]),
        "items": register(golden, "als_items", ALS["items"]),
    }
    for case in ALS["cases"]:
        estimator = configured(recommendation.ALS, case["values"])
        uid = "als_" + case["name"].replace("-", "_")
        model_case(golden, "ALS." + case["name"], estimator, train, frames, uid)

    datasets = {
        name: register(golden, "evaluate_" + name, spec["sql"], spec["vectors"], coalesce=1)
        for name, spec in EVALUATORS["datasets"].items()
    }
    seen = set()
    for case in EVALUATORS["cases"]:
        if case["class"] in seen:
            continue
        seen.add(case["class"])
        evaluator = configured(getattr(evaluation, case["class"]), case["values"])
        frame = datasets[case["dataset"]]
        calls = arguments({})
        golden.record_members(
            golden.models,
            case["name"],
            evaluator,
            case["class"],
            "evaluator",
            attributes=ATTRIBUTES[case["class"]],
            arguments=calls,
            frame=frame,
            skip=SKIP,
        )
        golden.persist(
            case["name"] + ".evaluator",
            evaluator,
            case["class"],
            "evaluator",
            attributes=ATTRIBUTES[case["class"]],
            arguments=calls,
            frame=frame,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "tests/golden/fpm_recommendation_evaluation.json"
    )
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    golden = Golden(spark)
    try:
        capture(golden)
        golden.dump(args.out, {"fits": golden.fits, "models": golden.models})
        golden.dump(
            args.out.with_name("fpm_recommendation_evaluation_persistence.json"),
            {"persistence": golden.persistence},
        )
    finally:
        golden.close()
        spark.stop()


if __name__ == "__main__":
    main()
