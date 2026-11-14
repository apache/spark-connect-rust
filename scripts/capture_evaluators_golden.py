# SPDX-License-Identifier: Apache-2.0
"""Capture standard PySpark 4.3.0 evaluator protocols and metrics over Connect.

Run with official PySpark on PYTHONPATH and SPARK_REMOTE pointing at Spark 4.3.0.
"""

import base64
import inspect
import json
import os
from pathlib import Path

import pyspark
from pyspark.ml import evaluation
from pyspark.ml.connect.serialize import serialize_ml_params
from pyspark.ml.functions import array_to_vector
from pyspark.sql import SparkSession

DATASETS = {
    "regression": {
        "sql": "SELECT * FROM VALUES (0, 1.0D, 0.5D, 1.0D), (1, 2.0D, 2.5D, 2.0D), (2, 4.0D, 3.0D, 0.5D), (3, 3.0D, 4.0D, 3.0D), (4, 1.0D, 8.0D, 0.0D) AS t(id, target, guess, w)",
        "vectors": [],
    },
    "binary": {
        "sql": "SELECT *, array(1.0D-score, score) AS scoreVec FROM VALUES (0, 0.0D, 0.1D, 1.0D), (1, 0.0D, 0.4D, 2.0D), (2, 1.0D, 0.35D, 1.0D), (3, 1.0D, 0.8D, 3.0D), (4, 1.0D, 0.4D, 0.5D), (5, 0.0D, 0.9D, 0.0D) AS t(id, target, score, w)",
        "vectors": ["scoreVec"],
    },
    "multiclass": {
        "sql": "SELECT * FROM VALUES (0, 0.0D, 0.0D, array(0.8D,0.1D,0.1D),1.0D), (1,1.0D,0.0D,array(0.6D,0.3D,0.1D),2.0D), (2,2.0D,2.0D,array(0.0D,0.2D,0.8D),1.0D), (3,2.0D,1.0D,array(0.0D,1.0D,0.0D),0.5D), (4,1.0D,1.0D,array(0.1D,0.8D,0.1D),3.0D), (5,0.0D,2.0D,array(0.1D,0.1D,0.8D),1.0D) AS t(id,target,guess,p,w)",
        "vectors": ["p"],
    },
    "multilabel": {
        "sql": "SELECT * FROM VALUES (0,array(0.0D,2.0D),array(0.0D,1.0D)), (1,array(1.0D),array(1.0D,2.0D)), (2,array(0.0D,1.0D,2.0D),array(0.0D,1.0D)), (3,array(2.0D),CAST(array() AS ARRAY<DOUBLE>)), (4,array(0.0D),array(0.0D)) AS t(id,target,guess)",
        "vectors": [],
    },
    "clustering": {
        "sql": "SELECT * FROM VALUES (0,0.0D,array(1.0D,0.0D),1.0D), (1,0.0D,array(0.8D,0.2D),2.0D), (2,1.0D,array(0.0D,1.0D),1.0D), (3,1.0D,array(0.1D,0.9D),3.0D) AS t(id,guess,xs,w)",
        "vectors": ["xs"],
    },
    "ranking": {
        "sql": "SELECT * FROM VALUES (0,array(1.0D,2.0D,3.0D),array(1.0D,6.0D,2.0D,7.0D,3.0D)), (1,array(2.0D,4.0D),array(4.0D,1.0D,2.0D)), (2,array(3.0D),array(1.0D,2.0D)), (3,CAST(array() AS ARRAY<DOUBLE>),array(1.0D,2.0D)), (4,array(1.0D),CAST(array() AS ARRAY<DOUBLE>)) AS t(id,target,guess)",
        "vectors": [],
    },
}

METRICS = {
    "RegressionEvaluator": ["rmse", "mse", "r2", "mae", "var"],
    "BinaryClassificationEvaluator": ["areaUnderROC", "areaUnderPR"],
    "MulticlassClassificationEvaluator": [
        "f1",
        "accuracy",
        "weightedPrecision",
        "weightedRecall",
        "weightedTruePositiveRate",
        "weightedFalsePositiveRate",
        "weightedFMeasure",
        "truePositiveRateByLabel",
        "falsePositiveRateByLabel",
        "precisionByLabel",
        "recallByLabel",
        "fMeasureByLabel",
        "logLoss",
        "hammingLoss",
    ],
    "MultilabelClassificationEvaluator": [
        "subsetAccuracy",
        "accuracy",
        "hammingLoss",
        "precision",
        "recall",
        "f1Measure",
        "precisionByLabel",
        "recallByLabel",
        "f1MeasureByLabel",
        "microPrecision",
        "microRecall",
        "microF1Measure",
    ],
    "ClusteringEvaluator": ["silhouette"],
    "RankingEvaluator": [
        "meanAveragePrecision",
        "meanAveragePrecisionAtK",
        "precisionAtK",
        "ndcgAtK",
        "recallAtK",
    ],
}

BASE = {
    "RegressionEvaluator": dict(
        predictionCol="guess",
        labelCol="target",
        metricName="rmse",
        weightCol="w",
        throughOrigin=False,
    ),
    "BinaryClassificationEvaluator": dict(
        rawPredictionCol="score",
        labelCol="target",
        metricName="areaUnderROC",
        weightCol="w",
        numBins=0,
    ),
    "MulticlassClassificationEvaluator": dict(
        predictionCol="guess",
        labelCol="target",
        metricName="f1",
        weightCol="w",
        metricLabel=1.0,
        beta=2.0,
        probabilityCol="p",
        eps=0.05,
    ),
    "MultilabelClassificationEvaluator": dict(
        predictionCol="guess", labelCol="target", metricName="f1Measure", metricLabel=1.0
    ),
    "ClusteringEvaluator": dict(
        predictionCol="guess",
        featuresCol="xs",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
        weightCol="w",
    ),
    "RankingEvaluator": dict(
        predictionCol="guess", labelCol="target", metricName="meanAveragePrecision", k=2
    ),
}
DATASET_NAMES = dict(zip(BASE, DATASETS))


INPUT_CASES = [
    ("regression-int", "RegressionEvaluator", {}, "SELECT 1 AS prediction, 1.0D AS label"),
    (
        "multiclass-int",
        "MulticlassClassificationEvaluator",
        {},
        "SELECT 1 AS prediction, 1.0D AS label",
    ),
    (
        "binary-float",
        "BinaryClassificationEvaluator",
        {},
        "SELECT 0.5F AS rawPrediction, 1.0D AS label",
    ),
    (
        "binary-array",
        "BinaryClassificationEvaluator",
        {},
        "SELECT array(0.2D,0.8D) AS rawPrediction,1.0D AS label",
    ),
    (
        "ranking-int-array",
        "RankingEvaluator",
        {},
        "SELECT array(1,2) AS prediction,array(1) AS label",
    ),
    (
        "multilabel-int-array",
        "MultilabelClassificationEvaluator",
        {},
        "SELECT array(1,2) AS prediction,array(1) AS label",
    ),
    (
        "clustering-array",
        "ClusteringEvaluator",
        {},
        "SELECT * FROM VALUES (array(1.0D,0.0D),0.0D),(array(0.8D,0.2D),0.0D),(array(0.0D,1.0D),1.0D),(array(0.1D,0.9D),1.0D) AS t(features,prediction)",
    ),
    (
        "column-case",
        "RegressionEvaluator",
        {"labelCol": "LABEL", "predictionCol": "PREDICTION"},
        "SELECT 1.0D AS prediction,1.0D AS label",
    ),
]


def dataset(spark, spec):
    frame = spark.sql(spec["sql"])
    for column in spec["vectors"]:
        frame = frame.withColumn(column, array_to_vector(column))
    return frame


def main():
    spark = SparkSession.builder.remote(os.environ["SPARK_REMOTE"]).getOrCreate()
    output = {
        "spark_version": pyspark.__version__,
        "datasets": DATASETS,
        "classes": {},
        "cases": [],
    }
    frames = {}
    try:
        for name, base in BASE.items():
            cls = getattr(evaluation, name)
            instance = cls()
            signature = inspect.signature(cls)
            info = {
                "signature": "(*, "
                + ", ".join(key + "=" + repr(p.default) for key, p in signature.parameters.items())
                + ")",
                "defaults": {
                    p.name: instance._defaultParamMap[p]
                    for p in instance.params
                    if instance.hasDefault(p)
                },
                "parameters": list(signature.parameters),
                "params": [],
                "directions": [],
            }
            variants = [{}, base, dict.fromkeys(base)]
            integer = (
                "numBins"
                if name == "BinaryClassificationEvaluator"
                else "k"
                if name == "RankingEvaluator"
                else None
            )
            if integer:
                variants += [
                    dict(base, **{integer: value})
                    for value in [2**31 - 1, 2**31, 2**63 - 1, -(2**63)]
                ]
            for values in variants:
                stage = cls(**values)
                message = serialize_ml_params(stage, spark.client)
                info["params"].append(
                    {
                        "values": values,
                        "params_b64": base64.b64encode(
                            message.SerializeToString(deterministic=True)
                        ).decode(),
                    }
                )
            for metric in METRICS[name] + ["invalid", None]:
                stage = cls(metricName=metric)
                info["directions"].append({"metric": metric, "larger": stage.isLargerBetter()})
            output["classes"][name] = info
            frame = dataset(spark, DATASETS[DATASET_NAMES[name]]).cache()
            frame.count()
            frames[name] = frame
            variants = [base]
            if "weightCol" in base:
                variants.append({key: value for key, value in base.items() if key != "weightCol"})
            if name == "ClusteringEvaluator":
                variants += [dict(v, distanceMeasure="cosine") for v in variants[:]]
            if name == "BinaryClassificationEvaluator":
                variants += [dict(base, rawPredictionCol="scoreVec"), dict(base, numBins=2)]
            if name == "RankingEvaluator":
                variants += [dict(base, k=1), dict(base, k=5)]
            if name == "RegressionEvaluator":
                variants += [dict(base, throughOrigin=True)]
            if name == "MulticlassClassificationEvaluator":
                variants += [dict(base, eps=1e-12)]
            for variant, values in enumerate(variants):
                metrics = METRICS[name]
                if name == "MulticlassClassificationEvaluator" and variant == 2:
                    metrics = ["logLoss"]
                for metric in metrics:
                    params = dict(values, metricName=metric)
                    stage = cls(**params)
                    output["cases"].append(
                        {
                            "name": f"{name}-{variant}-{metric}",
                            "class": name,
                            "dataset": DATASET_NAMES[name],
                            "values": params,
                            "value": stage.evaluate(frame),
                            "larger": stage.isLargerBetter(),
                        }
                    )
        output["inputs"] = []
        for name, class_name, values, sql in INPUT_CASES:
            record = {"name": name, "class": class_name, "values": values, "sql": sql}
            try:
                record.update(
                    accepted=True,
                    result=getattr(evaluation, class_name)(**values).evaluate(spark.sql(sql)),
                )
            except Exception as error:
                record.update(
                    accepted=False,
                    type=type(error).__name__,
                    condition=error.getCondition(),
                    message=str(error).splitlines()[0],
                )
            output["inputs"].append(record)
        destination = Path(__file__).resolve().parents[1] / "tests/golden/evaluators.json"
        destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(
            f"Captured {len(output['classes'])} evaluators and {len(output['cases'])} metric cases from official PySpark."
        )
    finally:
        for frame in frames.values():
            frame.unpersist()
        spark.stop()


if __name__ == "__main__":
    main()
