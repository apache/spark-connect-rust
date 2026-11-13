#!/usr/bin/env python3
"""Capture all standard ML feature stages with official PySpark 4.3.0.

Run with the official pyspark-client on PYTHONPATH and a live Connect server.
"""

import argparse
import base64
import inspect
import json
import math
from pathlib import Path

import pyspark
from pyspark.ml import feature
from pyspark.ml.base import Estimator
from pyspark.ml.connect.serialize import serialize_ml_params
from pyspark.ml.functions import array_to_vector
from pyspark.ml.linalg import Vector, Vectors
from pyspark.sql import SparkSession

CASES = [
    {
        "name": "Binarizer",
        "id": "Binarizer",
        "values": {"inputCol": "x", "outputCol": "out", "threshold": 2.5},
    },
    {
        "name": "BucketedRandomProjectionLSH",
        "id": "BucketedRandomProjectionLSH",
        "values": {
            "inputCol": "features",
            "outputCol": "out",
            "numHashTables": 2,
            "bucketLength": 1.5,
            "seed": 42,
        },
    },
    {
        "name": "Bucketizer",
        "id": "Bucketizer",
        "values": {
            "inputCol": "x",
            "outputCol": "out",
            "splits": [{"float": "-inf"}, 2.0, 5.0, {"float": "inf"}],
        },
    },
    {
        "name": "ChiSqSelector",
        "id": "ChiSqSelector",
        "values": {
            "featuresCol": "features",
            "outputCol": "out",
            "labelCol": "label",
            "numTopFeatures": 2,
        },
    },
    {
        "name": "CountVectorizer",
        "id": "CountVectorizer",
        "values": {"inputCol": "words", "outputCol": "out", "vocabSize": 4},
    },
    {"name": "DCT", "id": "DCT", "values": {"inputCol": "features", "outputCol": "out"}},
    {
        "name": "ElementwiseProduct",
        "id": "ElementwiseProduct",
        "values": {
            "inputCol": "features",
            "outputCol": "out",
            "scalingVec": {"vector": [2.0, 1.0, 0.0, -1.0]},
        },
    },
    {
        "name": "FeatureHasher",
        "id": "FeatureHasher",
        "values": {"inputCols": ["x", "y", "category"], "outputCol": "out", "numFeatures": 8},
    },
    {
        "name": "HashingTF",
        "id": "HashingTF",
        "values": {"inputCol": "words", "outputCol": "out", "numFeatures": 8},
    },
    {"name": "IDF", "id": "IDF", "values": {"inputCol": "features", "outputCol": "out"}},
    {
        "name": "Imputer",
        "id": "Imputer",
        "values": {"inputCol": "missing", "outputCol": "out", "missingValue": {"float": "nan"}},
    },
    {
        "name": "IndexToString",
        "id": "IndexToString",
        "values": {"inputCol": "index", "outputCol": "out", "labels": ["a", "b", "c"]},
    },
    {
        "name": "Interaction",
        "id": "Interaction",
        "values": {"inputCols": ["x", "y"], "outputCol": "out"},
    },
    {
        "name": "MaxAbsScaler",
        "id": "MaxAbsScaler",
        "values": {"inputCol": "features", "outputCol": "out"},
    },
    {
        "name": "MinHashLSH",
        "id": "MinHashLSH",
        "values": {"inputCol": "features", "outputCol": "out", "numHashTables": 2, "seed": 42},
    },
    {
        "name": "MinMaxScaler",
        "id": "MinMaxScaler",
        "values": {"inputCol": "features", "outputCol": "out", "min": -1.0, "max": 1.0},
    },
    {"name": "NGram", "id": "NGram", "values": {"inputCol": "words", "outputCol": "out", "n": 2}},
    {
        "name": "Normalizer",
        "id": "Normalizer",
        "values": {"inputCol": "features", "outputCol": "out", "p": 1.0},
    },
    {
        "name": "OneHotEncoder",
        "id": "OneHotEncoder",
        "values": {"inputCol": "index", "outputCol": "out", "dropLast": False},
    },
    {"name": "PCA", "id": "PCA", "values": {"inputCol": "features", "outputCol": "out", "k": 2}},
    {
        "name": "PolynomialExpansion",
        "id": "PolynomialExpansion",
        "values": {"inputCol": "features", "outputCol": "out", "degree": 2},
    },
    {
        "name": "QuantileDiscretizer",
        "id": "QuantileDiscretizer",
        "values": {"inputCol": "x", "outputCol": "out", "numBuckets": 3, "relativeError": 0.0},
    },
    {
        "name": "RobustScaler",
        "id": "RobustScaler",
        "values": {
            "inputCol": "features",
            "outputCol": "out",
            "withCentering": True,
            "relativeError": 0.0,
        },
    },
    {
        "name": "RegexTokenizer",
        "id": "RegexTokenizer",
        "values": {"inputCol": "text", "outputCol": "out", "pattern": "\\s+", "gaps": True},
    },
    {
        "name": "RFormula",
        "id": "RFormula",
        "values": {
            "formula": "label ~ x + category",
            "featuresCol": "out",
            "labelCol": "target",
            "stringIndexerOrderType": "alphabetAsc",
        },
    },
    {
        "name": "SQLTransformer",
        "id": "SQLTransformer",
        "values": {"statement": "SELECT *, x + y AS out FROM __THIS__"},
    },
    {
        "name": "StandardScaler",
        "id": "StandardScaler",
        "values": {"inputCol": "features", "outputCol": "out", "withMean": True, "withStd": True},
    },
    {
        "name": "StopWordsRemover",
        "id": "StopWordsRemover",
        "values": {"inputCol": "words", "outputCol": "out", "stopWords": ["the", "gamma"]},
    },
    {
        "name": "StringIndexer",
        "id": "StringIndexer",
        "values": {"inputCol": "category", "outputCol": "out", "stringOrderType": "alphabetAsc"},
    },
    {
        "name": "TargetEncoder",
        "id": "TargetEncoder",
        "values": {"inputCol": "index", "outputCol": "out", "labelCol": "label", "smoothing": 1.0},
    },
    {"name": "Tokenizer", "id": "Tokenizer", "values": {"inputCol": "text", "outputCol": "out"}},
    {
        "name": "UnivariateFeatureSelector",
        "id": "UnivariateFeatureSelector",
        "values": {
            "featuresCol": "features",
            "outputCol": "out",
            "labelCol": "label",
            "featureType": "categorical",
            "labelType": "categorical",
            "selectionThreshold": 2.0,
        },
    },
    {
        "name": "VarianceThresholdSelector",
        "id": "VarianceThresholdSelector",
        "values": {"featuresCol": "features", "outputCol": "out", "varianceThreshold": 1.0},
    },
    {
        "name": "VectorAssembler",
        "id": "VectorAssembler",
        "values": {"inputCols": ["x", "y"], "outputCol": "out", "handleInvalid": "error"},
    },
    {
        "name": "VectorIndexer",
        "id": "VectorIndexer",
        "values": {"inputCol": "features", "outputCol": "out", "maxCategories": 2},
    },
    {
        "name": "VectorSizeHint",
        "id": "VectorSizeHint",
        "values": {"inputCol": "rawFeatures", "size": 4},
    },
    {
        "name": "VectorSlicer",
        "id": "VectorSlicer",
        "values": {"inputCol": "features", "outputCol": "out", "indices": [0, 2]},
    },
    {
        "name": "Word2Vec",
        "id": "Word2Vec",
        "values": {
            "inputCol": "words",
            "outputCol": "out",
            "vectorSize": 3,
            "minCount": 1,
            "seed": 42,
            "numPartitions": 1,
            "maxIter": 2,
            "windowSize": 2,
        },
    },
    {
        "name": "Binarizer",
        "id": "Binarizer-multi",
        "values": {
            "inputCols": ["x", "y"],
            "outputCols": ["out", "other"],
            "thresholds": [2.0, 4.0],
        },
    },
    {
        "name": "Bucketizer",
        "id": "Bucketizer-multi",
        "values": {
            "inputCols": ["x", "y"],
            "outputCols": ["out", "other"],
            "splitsArray": [
                [{"float": "-inf"}, 3.0, {"float": "inf"}],
                [{"float": "-inf"}, 5.0, {"float": "inf"}],
            ],
        },
    },
    {
        "name": "Imputer",
        "id": "Imputer-multi",
        "values": {
            "inputCols": ["missing", "x"],
            "outputCols": ["out", "other"],
            "strategy": "median",
        },
    },
    {
        "name": "OneHotEncoder",
        "id": "OneHotEncoder-multi",
        "values": {
            "inputCols": ["index", "label"],
            "outputCols": ["out", "other"],
            "dropLast": False,
        },
    },
    {
        "name": "StopWordsRemover",
        "id": "StopWordsRemover-multi",
        "values": {
            "inputCols": ["words", "words2"],
            "outputCols": ["out", "other"],
            "stopWords": ["the", "gamma"],
        },
    },
    {
        "name": "StringIndexer",
        "id": "StringIndexer-multi",
        "values": {
            "inputCols": ["category", "text"],
            "outputCols": ["out", "other"],
            "stringOrderType": "alphabetAsc",
        },
    },
    {
        "name": "TargetEncoder",
        "id": "TargetEncoder-multi",
        "values": {
            "inputCols": ["index", "label"],
            "outputCols": ["out", "other"],
            "labelCol": "x",
            "targetType": "continuous",
        },
    },
    {
        "name": "QuantileDiscretizer",
        "id": "QuantileDiscretizer-multi",
        "values": {
            "inputCols": ["x", "y"],
            "outputCols": ["out", "other"],
            "numBucketsArray": [2, 3],
            "relativeError": 0.0,
        },
    },
    {
        "name": "StandardScaler",
        "id": "StandardScaler-variant",
        "values": {"inputCol": "features", "outputCol": "out", "withMean": False, "withStd": False},
    },
    {
        "name": "RegexTokenizer",
        "id": "RegexTokenizer-variant",
        "values": {
            "inputCol": "text",
            "outputCol": "out",
            "pattern": "[a-z]+",
            "gaps": False,
            "toLowercase": False,
        },
    },
]
SQL = "SELECT *, words AS words2 FROM VALUES (0, 0.0D, 8.0D, 0.0D, 'a', 0.0D, array(0.0D,3.0D,0.0D,1.0D), array('the','alpha','alpha','beta','beta','gamma','delta'), 'the alpha alpha beta beta gamma delta', CAST('NaN' AS DOUBLE)), (1, 1.0D, 7.0D, 1.0D, 'b', 1.0D, array(2.0D,7.0D,1.0D,2.0D), array('alpha','alpha','beta','gamma'), 'alpha alpha beta gamma', CAST(NULL AS DOUBLE)), (2, 2.0D, 6.0D, 0.0D, 'c', 2.0D, array(0.0D,3.0D,2.0D,3.0D), array('alpha','beta','gamma'), 'alpha beta gamma', 2.0D), (3, 3.0D, 5.0D, 1.0D, 'a', 0.0D, array(2.0D,7.0D,0.0D,4.0D), array('alpha','beta'), 'alpha beta', 3.0D), (4, 4.0D, 4.0D, 0.0D, 'b', 1.0D, array(0.0D,3.0D,1.0D,5.0D), array('alpha','beta'), 'alpha beta', 4.0D), (5, 5.0D, 3.0D, 1.0D, 'c', 2.0D, array(2.0D,7.0D,2.0D,6.0D), array('alpha'), 'alpha', 5.0D), (6, 6.0D, 2.0D, 0.0D, 'a', 0.0D, array(0.0D,3.0D,0.0D,7.0D), array('alpha'), 'alpha', 6.0D), (7, 7.0D, 1.0D, 1.0D, 'b', 1.0D, array(2.0D,7.0D,1.0D,8.0D), array('alpha','delta'), 'alpha delta', 7.0D) AS t(id, x, y, label, category, index, xs, words, text, missing)"


def decode(value):
    if isinstance(value, dict):
        if "float" in value:
            return float(value["float"])
        if "vector" in value:
            return Vectors.dense(value["vector"])
        return {key: decode(v) for key, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def clean(value):
    if isinstance(value, Vector):
        return value.toArray().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": str(value)}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, dict):
        return {key: clean(v) for key, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=Path("tests/golden/ml_features.json"))
    args = parser.parse_args()
    assert pyspark.__version__ == "4.3.0"
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    result = {
        "spark_version": pyspark.__version__,
        "sql": SQL,
        "api": {},
        "cases": [],
        "params": [],
    }
    try:
        frame = (
            spark.sql(SQL)
            .withColumn("features", array_to_vector("xs"))
            .withColumn("rawFeatures", array_to_vector("xs"))
            .coalesce(1)
        )
        for name in feature.__all__:
            if name.endswith("Model"):
                continue
            cls = getattr(feature, name)
            instance = cls()
            defaults = {k: clean(v.default) for k, v in inspect.signature(cls).parameters.items()}
            params = {p.name: p.typeConverter.__name__ for p in instance.params}
            model = "Bucketizer" if name == "QuantileDiscretizer" else name + "Model"
            result["api"][name] = {
                "defaults": defaults,
                "params": params,
                "estimator": isinstance(instance, Estimator),
                "model_params": [p.name for p in getattr(feature, model)().params]
                if isinstance(instance, Estimator)
                else [],
            }
            values = {}
            for key, converter in params.items():
                value = defaults.get(key)
                if value is None:
                    value = {
                        "toString": "sample",
                        "toListString": ["x", "y"],
                        "toInt": 42,
                        "toFloat": 0.5,
                        "toListFloat": [0.0, 1.0],
                        "toListInt": [0, 2],
                        "toListListFloat": [[0.0, 1.0], [2.0, 3.0]],
                        "toVector": {"vector": [2.0, -1.0]},
                    }[converter]
                values[key] = value
            variants = [{}, values, {k: None for k in params}]
            for values in variants:
                instance = cls()
                for key, value in decode(values).items():
                    getattr(instance, "set" + key[0].upper() + key[1:])(value)
                wire = serialize_ml_params(instance, spark.client)
                result["params"].append(
                    {
                        "name": name,
                        "values": values,
                        "params_b64": base64.b64encode(
                            wire.SerializeToString(deterministic=True)
                        ).decode(),
                    }
                )
        for case in CASES:
            cls = getattr(feature, case["name"])
            values = decode(case["values"])
            defaults = result["api"][case["name"]]["defaults"]
            stage = cls(**{k: v for k, v in values.items() if k in defaults})
            for key, value in values.items():
                if key not in defaults:
                    getattr(stage, "set" + key[0].upper() + key[1:])(value)
            if isinstance(stage, Estimator):
                stage = stage.fit(frame)
            columns = (
                ["rawFeatures"]
                if case["name"] == "VectorSizeHint"
                else ["out"] + (["other"] if "outputCols" in values else [])
            )
            if case["name"] == "RFormula":
                columns.append("target")
            transformed = stage.transform(frame).orderBy("id").select(*columns)
            result["cases"].append(
                dict(
                    case,
                    columns=columns,
                    rows=clean(transformed.collect()),
                    schema=transformed.schema.jsonValue(),
                )
            )
            print(case["id"], flush=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
