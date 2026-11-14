#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture the classification models and summaries golden from official PySpark 4.3.0.

Each estimator variant is fitted on a fixed one-partition dataset with fixed seeds and pinned
uids. The fitted model's members and the summaries' members are captured by the shared helper
(scripts/ml_golden_common.py), together with the requests the client sends. Models and
estimators are persisted and loaded back.

Run with the official client, for example:
    /tmp/scr-ref-py scripts/capture_classification_models_golden.py \
        --remote sc://localhost:15013 --out tests/golden/classification_models.json

The summaries go to the sibling file classification_summaries.json.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from functools import partial

from ml_golden_common import Golden  # noqa: E402
from pyspark.ml import classification as C  # noqa: E402
from pyspark.ml.linalg import Vectors  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

BINARY_TRAIN = [
    (1.0, 1.0, [0.0, 1.1]),
    (0.0, 2.0, [2.0, 1.0]),
    (1.0, 3.0, [0.5, 2.5]),
    (0.0, 1.0, [3.0, 0.2]),
    (1.0, 1.5, [1.0, 1.9]),
    (0.0, 0.5, [2.5, 3.0]),
    (1.0, 2.0, [0.1, 0.9]),
    (0.0, 1.0, [1.8, 0.4]),
]
BINARY_EVAL = [
    (1.0, 1.0, [0.2, 1.0]),
    (0.0, 1.0, [2.2, 0.6]),
    (1.0, 2.0, [0.8, 2.2]),
    (0.0, 1.0, [2.9, 2.1]),
]
MULTI_TRAIN = [
    (0.0, 1.0, [0.0, 0.5]),
    (1.0, 2.0, [1.0, 0.0]),
    (2.0, 1.0, [2.0, 1.5]),
    (0.0, 1.0, [0.2, 0.9]),
    (1.0, 1.0, [1.1, 0.4]),
    (2.0, 2.0, [2.4, 2.0]),
    (0.0, 0.5, [0.5, 0.1]),
    (1.0, 1.5, [0.9, 0.7]),
    (2.0, 1.0, [1.7, 2.2]),
]
MULTI_EVAL = [
    (0.0, 1.0, [0.1, 0.6]),
    (1.0, 1.0, [1.2, 0.1]),
    (2.0, 1.0, [2.1, 1.8]),
    (1.0, 2.0, [0.6, 0.6]),
]
# Multinomial and complement NaiveBayes need non-negative features; Bernoulli needs 0/1.
NB_COUNTS = [
    (0.0, 1.0, [2.0, 0.0]),
    (1.0, 1.0, [0.0, 3.0]),
    (2.0, 1.0, [1.0, 4.0]),
    (0.0, 1.0, [3.0, 1.0]),
    (1.0, 2.0, [0.0, 2.0]),
    (2.0, 1.0, [2.0, 2.0]),
]
NB_COUNTS_EVAL = [
    (0.0, 1.0, [1.0, 0.0]),
    (1.0, 1.0, [0.0, 2.0]),
    (2.0, 1.0, [1.0, 3.0]),
]
NB_BINARY = [
    (0.0, 1.0, [1.0, 0.0]),
    (1.0, 1.0, [0.0, 1.0]),
    (0.0, 1.0, [1.0, 1.0]),
    (1.0, 2.0, [0.0, 0.0]),
    (0.0, 1.0, [1.0, 0.0]),
    (1.0, 1.0, [0.0, 1.0]),
]
NB_BINARY_EVAL = [
    (0.0, 1.0, [1.0, 1.0]),
    (1.0, 1.0, [0.0, 0.0]),
    (1.0, 1.0, [0.0, 1.0]),
]
DATASETS = [
    ("binary_train", BINARY_TRAIN),
    ("binary_eval", BINARY_EVAL),
    ("multi_train", MULTI_TRAIN),
    ("multi_eval", MULTI_EVAL),
    ("nb_counts", NB_COUNTS),
    ("nb_counts_eval", NB_COUNTS_EVAL),
    ("nb_binary", NB_BINARY),
    ("nb_binary_eval", NB_BINARY_EVAL),
]

VECTORS = [
    ("dense", Vectors.dense([0.5, 1.5])),
    ("dense2", Vectors.dense([3.0, -0.25])),
    ("sparse", Vectors.sparse(2, [1], [2.0])),
]
# Naive Bayes variants need 0/1 (Bernoulli) or non-negative (the others) features; these are both.
NB_VECTORS = [
    ("dense", Vectors.dense([1.0, 0.0])),
    ("dense2", Vectors.dense([0.0, 1.0])),
    ("sparse", Vectors.sparse(2, [1], [1.0])),
]

# Members of the fitted models, as pyspark.ml 4.3.0 reads them. "name()" is a method with no
# arguments; a plain name is a property (or a class attribute, such as supportedImpurities).
ATTRIBUTES = {
    "LogisticRegressionModel": [
        "coefficients",
        "coefficientMatrix",
        "intercept",
        "interceptVector",
        "numClasses",
        "numFeatures",
        "hasSummary",
        "summary",
    ],
    "LinearSVCModel": [
        "coefficients",
        "intercept",
        "numClasses",
        "numFeatures",
        "hasSummary",
        "summary()",
    ],
    "FMClassificationModel": [
        "factors",
        "linear",
        "intercept",
        "numClasses",
        "numFeatures",
        "hasSummary",
        "summary()",
    ],
    "MultilayerPerceptronClassificationModel": [
        "weights",
        "numClasses",
        "numFeatures",
        "hasSummary",
        "summary()",
    ],
    "DecisionTreeClassificationModel": [
        "numNodes",
        "depth",
        "featureImportances",
        "toDebugString",
        "numClasses",
        "numFeatures",
        "supportedImpurities",
    ],
    "RandomForestClassificationModel": [
        "featureImportances",
        "trees",
        "treeWeights",
        "getNumTrees",
        "totalNumNodes",
        "toDebugString",
        "numClasses",
        "numFeatures",
        "hasSummary",
        "summary",
        "supportedFeatureSubsetStrategies",
        "supportedImpurities",
    ],
    "GBTClassificationModel": [
        "featureImportances",
        "trees",
        "treeWeights",
        "getNumTrees",
        "totalNumNodes",
        "toDebugString",
        "numClasses",
        "numFeatures",
        "supportedFeatureSubsetStrategies",
        "supportedImpurities",
        "supportedLossTypes",
    ],
    "NaiveBayesModel": ["pi", "theta", "sigma", "numClasses", "numFeatures"],
    "OneVsRestModel": ["models"],
    "DecisionTreeRegressionModel": [
        "depth",
        "numNodes",
        "featureImportances",
        "toDebugString",
        "numFeatures",
        "supportedImpurities",
    ],
}

# Summary members. The scalars are properties of the summary, and the *ByLabel and weighted*
# ones with a beta are in summary_arguments.
_SCALARS = [
    "accuracy",
    "labels",
    "labelCol",
    "predictionCol",
    "weightCol",
    "weightedPrecision",
    "weightedRecall",
    "weightedTruePositiveRate",
    "weightedFalsePositiveRate",
    "truePositiveRateByLabel",
    "falsePositiveRateByLabel",
    "precisionByLabel",
    "recallByLabel",
    "predictions",
]
_LR = ["featuresCol", "probabilityCol"]
_TRAINING = ["objectiveHistory", "totalIterations"]
_BINARY = [
    "areaUnderROC",
    "scoreCol",
    "roc",
    "pr",
    "fMeasureByThreshold",
    "precisionByThreshold",
    "recallByThreshold",
]
SUMMARY_ATTRIBUTES = {
    "LogisticRegressionSummary": _SCALARS + _LR,
    "LogisticRegressionTrainingSummary": _SCALARS + _LR + _TRAINING,
    "BinaryLogisticRegressionSummary": _SCALARS + _LR + _BINARY,
    "BinaryLogisticRegressionTrainingSummary": _SCALARS + _LR + _TRAINING + _BINARY,
    "LinearSVCSummary": _SCALARS + _BINARY,
    "LinearSVCTrainingSummary": _SCALARS + _TRAINING + _BINARY,
    "FMClassificationSummary": _SCALARS + _BINARY,
    "FMClassificationTrainingSummary": _SCALARS + _TRAINING + _BINARY,
    "MultilayerPerceptronClassificationSummary": _SCALARS,
    "MultilayerPerceptronClassificationTrainingSummary": _SCALARS + _TRAINING,
    "RandomForestClassificationSummary": _SCALARS,
    "RandomForestClassificationTrainingSummary": _SCALARS + _TRAINING,
    "BinaryRandomForestClassificationSummary": _SCALARS + _BINARY,
    "BinaryRandomForestClassificationTrainingSummary": _SCALARS + _TRAINING + _BINARY,
}

# Class methods that build a model client-side (none for the classifiers in this area).
CLASS_METHODS = []

PREDICT = ["predict", "predictRaw", "predictProbability"]
METHODS = {
    "LogisticRegressionModel": PREDICT + ["evaluate", "transform"],
    "LinearSVCModel": ["predict", "predictRaw", "evaluate", "transform"],
    "FMClassificationModel": PREDICT + ["evaluate", "transform"],
    "MultilayerPerceptronClassificationModel": PREDICT + ["evaluate", "transform"],
    "DecisionTreeClassificationModel": PREDICT + ["predictLeaf", "transform"],
    "RandomForestClassificationModel": PREDICT + ["predictLeaf", "evaluate", "transform"],
    "GBTClassificationModel": PREDICT + ["predictLeaf", "evaluateEachIteration", "transform"],
    "NaiveBayesModel": PREDICT + ["transform"],
    "OneVsRestModel": ["transform"],
    "DecisionTreeRegressionModel": ["predict", "predictLeaf", "transform"],
}


def arguments(name, frame):
    """The labelled calls of the model class `name` on `frame` (the evaluation dataset)."""
    calls = []
    vectors = NB_VECTORS if name == "NaiveBayesModel" else VECTORS
    for method in METHODS.get(name, []):
        if method in ("transform", "evaluate", "evaluateEachIteration"):
            calls.append((method, method, (frame,)))
            continue
        for vector_name, vector in vectors:
            calls.append((f"{method}-{vector_name}", method, (vector,)))
    return calls


def sub_arguments(name, frame):
    """The calls of the OneVsRest sub-models: their evaluate takes the internal binary label."""
    return [c for c in arguments(name, frame) if c[0] != "evaluate"]


def summary_arguments(name, frame):
    """The beta-taking methods of the summary classes, with beta 1.0 and 0.5."""
    calls = []
    for method in ("fMeasureByLabel", "weightedFMeasure"):
        for beta in (1.0, 0.5):
            calls.append((f"{method}-{beta}", method, (beta,)))
    return calls


def persist_arguments(name, frame):
    """The copy's calls: the prediction of the dense vector, and the transform."""
    return [c for c in arguments(name, frame) if c[0] in ("predict-dense", "transform")]


def _lr(**kwargs):
    return lambda: C.LogisticRegression(**kwargs)


def _estimators():
    """(key, factory, train dataset, eval dataset, model class, the estimator's uid)."""
    weighted = {"weightCol": "weight"}
    return [
        (
            "LogisticRegression.binomial",
            _lr(maxIter=20, regParam=0.01, family="binomial", **weighted),
            "binary_train",
            "binary_eval",
            "LogisticRegressionModel",
        ),
        (
            "LogisticRegression.multinomial",
            _lr(maxIter=20, regParam=0.01, family="multinomial", **weighted),
            "multi_train",
            "multi_eval",
            "LogisticRegressionModel",
        ),
        (
            "LinearSVC",
            lambda: C.LinearSVC(maxIter=20, regParam=0.01, **weighted),
            "binary_train",
            "binary_eval",
            "LinearSVCModel",
        ),
        (
            "DecisionTreeClassifier",
            lambda: C.DecisionTreeClassifier(maxDepth=3, seed=42, **weighted),
            "multi_train",
            "multi_eval",
            "DecisionTreeClassificationModel",
        ),
        (
            "RandomForestClassifier.binary",
            lambda: C.RandomForestClassifier(numTrees=3, maxDepth=3, seed=42, **weighted),
            "binary_train",
            "binary_eval",
            "RandomForestClassificationModel",
        ),
        (
            "RandomForestClassifier.multiclass",
            lambda: C.RandomForestClassifier(numTrees=3, maxDepth=3, seed=42, **weighted),
            "multi_train",
            "multi_eval",
            "RandomForestClassificationModel",
        ),
        (
            "GBTClassifier",
            lambda: C.GBTClassifier(maxIter=3, maxDepth=2, seed=42, stepSize=0.1, **weighted),
            "binary_train",
            "binary_eval",
            "GBTClassificationModel",
        ),
        (
            "NaiveBayes.multinomial",
            lambda: C.NaiveBayes(modelType="multinomial", **weighted),
            "nb_counts",
            "nb_counts_eval",
            "NaiveBayesModel",
        ),
        (
            "NaiveBayes.complement",
            lambda: C.NaiveBayes(modelType="complement", **weighted),
            "nb_counts",
            "nb_counts_eval",
            "NaiveBayesModel",
        ),
        (
            "NaiveBayes.bernoulli",
            lambda: C.NaiveBayes(modelType="bernoulli", **weighted),
            "nb_binary",
            "nb_binary_eval",
            "NaiveBayesModel",
        ),
        (
            "NaiveBayes.gaussian",
            lambda: C.NaiveBayes(modelType="gaussian", **weighted),
            "binary_train",
            "binary_eval",
            "NaiveBayesModel",
        ),
        (
            "MultilayerPerceptronClassifier",
            lambda: C.MultilayerPerceptronClassifier(
                layers=[2, 4, 3], maxIter=30, seed=42, blockSize=4, stepSize=0.1
            ),
            "multi_train",
            "multi_eval",
            "MultilayerPerceptronClassificationModel",
        ),
        (
            "FMClassifier",
            lambda: C.FMClassifier(factorSize=2, maxIter=20, seed=42, regParam=0.01, stepSize=0.1),
            "binary_train",
            "binary_eval",
            "FMClassificationModel",
        ),
        (
            "OneVsRest",
            lambda: _one_vs_rest(),
            "multi_train",
            "multi_eval",
            "OneVsRestModel",
        ),
    ]


# The training summary of each model that has one: a property, or a no-argument method.
SUMMARY_ACCESS = {
    "LogisticRegressionModel": "summary",
    "LinearSVCModel": "summary()",
    "FMClassificationModel": "summary()",
    "MultilayerPerceptronClassificationModel": "summary()",
    "RandomForestClassificationModel": "summary",
}

# Training-summary access; the evaluation summary comes from evaluate(df) on the eval dataset.
HAS_SUMMARY = set(SUMMARY_ACCESS)


def _one_vs_rest():
    classifier = C.LogisticRegression(maxIter=20, regParam=0.01, weightCol="weight")
    classifier._resetUid("OneVsRest_classifier")
    return C.OneVsRest(classifier=classifier, weightCol="weight")


def _accessor(text):
    return text[:-2] if text.endswith("()") else text


def _training_summary(model, access):
    value = getattr(model, _accessor(access))
    return value() if access.endswith("()") else value


def capture(golden, frames, out_models, out_summaries):
    for key, factory, train, evaluation, model_cls in _estimators():
        uid = key.replace(".", "_")
        estimator = factory()
        model = golden.fit(key, estimator, frames[train], uid)
        if model is None:
            continue
        frame = frames[evaluation]
        golden.record_members(
            out_models,
            key,
            model,
            model_cls,
            "model",
            attributes=ATTRIBUTES[model_cls],
            arguments=arguments,
            frame=frame,
        )
        if model_cls in ("RandomForestClassificationModel", "GBTClassificationModel"):
            tree = model.trees[0]
            tree_cls = type(tree).__name__
            golden.record_members(
                out_models,
                key + ".trees[0]",
                tree,
                tree_cls,
                "model.trees[0]",
                attributes=ATTRIBUTES[tree_cls],
                arguments=arguments,
                frame=frame,
            )
        if model_cls == "OneVsRestModel":
            for i, sub in enumerate(model.models):
                golden.record_members(
                    out_models,
                    f"{key}.models[{i}]",
                    sub,
                    type(sub).__name__,
                    f"model.models[{i}]",
                    attributes=ATTRIBUTES["LogisticRegressionModel"],
                    arguments=sub_arguments,
                    frame=frame,
                    skip={"evaluate": "takes the internal binary label, not the dataset"},
                )
        if model_cls in HAS_SUMMARY:
            access = SUMMARY_ACCESS[model_cls]
            summary, via = golden.call("model." + access, partial(_training_summary, model, access))
            _summary(golden, out_summaries, key + ".training", summary, via)
        if model_cls in (
            "LogisticRegressionModel",
            "LinearSVCModel",
            "FMClassificationModel",
            "MultilayerPerceptronClassificationModel",
            "RandomForestClassificationModel",
        ):
            summary, via = golden.call(
                f"model.evaluate({evaluation})", partial(model.evaluate, frame)
            )
            _summary(golden, out_summaries, key + ".evaluate", summary, via)
        if model_cls == "LogisticRegressionModel" and key == "LogisticRegression.multinomial":
            # A client-side error: the multiclass summary has no binary (roc) summary members.
            summary = model.summary
            out_summaries[key + ".training"]["members"]["roc-on-multiclass"] = golden.call(
                "model.summary.roc", partial(getattr, summary, "roc")
            )[1]
        if model_cls == "NaiveBayesModel" and key == "NaiveBayes.bernoulli":
            # Errors of the family, as official PySpark raises them.
            out_models[key + ".errors"] = {
                "members": {
                    "predict-outside-0-1": golden.call(
                        "model.predict(Vectors.dense([0.5, 1.5]))",
                        partial(model.predict, Vectors.dense([0.5, 1.5])),
                    )[1]
                }
            }
        golden.persist(
            key + ".model",
            model,
            model_cls,
            "model",
            arguments=persist_arguments,
            frame=frame,
        )
        golden.persist(key + ".estimator", estimator, type(estimator).__name__, "estimator")


def _summary(golden, out, key, summary, via):
    cls_name = type(summary).__name__
    record = golden.members(
        summary,
        cls_name,
        "summary",
        attributes=SUMMARY_ATTRIBUTES[cls_name],
        arguments=summary_arguments,
    )
    record["via"] = via
    out[key] = record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=Path("tests/golden/classification_models.json"))
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    golden = Golden(spark)
    try:
        frames = {name: golden.dataset(name, rows) for name, rows in DATASETS}
        models, summaries = {}, {}
        capture(golden, frames, models, summaries)
        golden.models.update(models)
        golden.summaries.update(summaries)
        golden.class_methods(CLASS_METHODS, lambda name: getattr(C, name), frames["binary_eval"])
        golden.dump(
            args.out,
            {
                "fits": golden.fits,
                "models": golden.models,
                "class_methods": golden.class_methods_out,
            },
        )
        golden.dump(
            args.out.with_name("classification_summaries.json"), {"summaries": golden.summaries}
        )
        golden.dump(
            args.out.with_name("classification_persistence.json"),
            {"persistence": golden.persistence},
        )
    finally:
        golden.close()
        spark.stop()


if __name__ == "__main__":
    main()
