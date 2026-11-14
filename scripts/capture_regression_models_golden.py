#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture the regression models and summaries golden from official PySpark 4.3.0.

Each estimator variant is fitted on a fixed one-partition dataset with fixed seeds and pinned
uids. The fitted model's members, its children (the training summary, the summary of evaluate on
the held-out rows, and the first tree of each ensemble) and the summaries' members are captured by
the shared helper (scripts/ml_golden_common.py), together with the requests the client sends.
Models and estimators are persisted and loaded back.

Run with the official client, for example:
    /tmp/scr-ref-py scripts/capture_regression_models_golden.py \
        --remote sc://localhost:15013 --out tests/golden/regression_models.json

The summaries go to the sibling file regression_summaries.json, the copies to
regression_persistence.json.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from functools import partial  # noqa: E402

from ml_golden_common import Golden  # noqa: E402
from pyspark.ml import regression as R  # noqa: E402
from pyspark.ml.linalg import DenseVector, SparseVector, Vectors  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

# Ten rows of three features, of the label 1.5 * x0 - 2 * x1 + 0.5 * x2 + 3 plus a small noise.
# The labels are all positive, so that the gamma family fits them too.
FEATURES = [
    [1.0, 0.0, 2.0],
    [2.0, 1.0, 0.0],
    [3.0, 1.0, 1.0],
    [4.0, 0.0, 3.0],
    [5.0, 2.0, 1.0],
    [6.0, 1.0, 2.0],
    [7.0, 3.0, 0.0],
    [8.0, 2.0, 2.0],
    [9.0, 1.0, 3.0],
    [10.0, 3.0, 1.0],
]
NOISE = [0.1, -0.2, 0.05, 0.0, -0.1, 0.15, -0.05, 0.2, -0.15, 0.1]
LABELS = [1.5 * x0 - 2.0 * x1 + 0.5 * x2 + 3.0 + e for (x0, x1, x2), e in zip(FEATURES, NOISE)]
WEIGHTS = [1.0, 2.0, 1.0, 1.5, 1.0, 0.5, 1.0, 2.0, 1.0, 1.0]
# Counts of the same rows, for the poisson family.
COUNTS = [1.0, 2.0, 2.0, 4.0, 3.0, 5.0, 4.0, 7.0, 9.0, 8.0]
# Rows held out of the fits, for evaluate, transform and evaluateEachIteration.
HELD_OUT = [[2.5, 1.0, 1.5], [6.5, 2.0, 0.5], [9.5, 0.0, 2.5]]
HELD_OUT_LABELS = [6.0, 4.5, 18.0]

# Survival times, censors and two features, after the example of AFTSurvivalRegression.
AFT_ROWS = [
    (1.218, 1.0, [1.560, -0.605]),
    (2.949, 0.0, [0.346, 2.158]),
    (3.627, 0.0, [1.380, 0.231]),
    (0.273, 1.0, [0.520, 1.151]),
    (4.199, 0.0, [0.795, -0.226]),
    (1.5, 1.0, [1.0, 0.5]),
    (2.2, 1.0, [0.3, 1.2]),
    (3.1, 0.0, [1.1, -0.4]),
]
AFT_HELD_OUT = [(2.0, 1.0, [0.9, 0.1]), (1.0, 0.0, [0.2, 1.0])]
QUANTILE_PROBABILITIES = [0.1, 0.5, 0.9]

# Labels that are not monotone in the second feature, so that isotonic and antitonic fits pool.
ISOTONIC_ROWS = [
    (1.0, 1.0, [0.0, 1.0]),
    (2.0, 2.0, [0.0, 2.0]),
    (1.5, 1.0, [0.0, 3.0]),
    (3.0, 1.0, [0.0, 4.0]),
    (2.5, 0.5, [0.0, 5.0]),
    (4.0, 1.0, [0.0, 6.0]),
]
ISOTONIC_HELD_OUT = [(2.0, 1.0, [0.0, 2.5]), (5.0, 1.0, [0.0, 8.0])]

# The single values that predict and the other per-value methods take.
DENSE3 = Vectors.dense([2.5, 1.0, 1.5])
SPARSE3 = Vectors.sparse(3, [0, 2], [2.5, 1.5])
DENSE2 = Vectors.dense([1.0, 0.5])

# "DATASET" stands for the held-out DataFrame of the family, which the capture builds.
DATASET = object()

# The fits: (estimator class, dataset name, label, params). The label names the variant in its
# key, "<estimator>.<label>". Each dataset has a held-out frame, "<dataset>_held_out".
FITS = [
    ("LinearRegression", "rows", "normal", dict(solver="normal", weightCol="weight")),
    ("LinearRegression", "rows", "l-bfgs", dict(solver="l-bfgs", maxIter=20, regParam=0.1)),
    ("LinearRegression", "rows", "huber", dict(loss="huber", maxIter=20)),
    (
        "GeneralizedLinearRegression",
        "rows",
        "gaussian",
        dict(family="gaussian", weightCol="weight"),
    ),
    (
        "GeneralizedLinearRegression",
        "counts",
        "poisson",
        dict(family="poisson", link="log", maxIter=25),
    ),
    ("GeneralizedLinearRegression", "rows", "gamma", dict(family="gamma", link="log", maxIter=25)),
    (
        "GeneralizedLinearRegression",
        "rows",
        "tweedie",
        dict(family="tweedie", variancePower=1.5, linkPower=0.0, maxIter=25),
    ),
    (
        "GeneralizedLinearRegression",
        "rows",
        "no-intercept",
        dict(family="gaussian", fitIntercept=False),
    ),
    (
        "AFTSurvivalRegression",
        "aft",
        "default",
        dict(quantileProbabilities=QUANTILE_PROBABILITIES, quantilesCol="quantiles"),
    ),
    ("IsotonicRegression", "isotonic", "weighted", dict(featureIndex=1, weightCol="weight")),
    ("IsotonicRegression", "isotonic", "antitonic", dict(featureIndex=1, isotonic=False)),
    ("FMRegressor", "rows", "default", dict(factorSize=2, maxIter=20, seed=7)),
    ("DecisionTreeRegressor", "rows", "default", dict(maxDepth=3, seed=1)),
    ("RandomForestRegressor", "rows", "default", dict(numTrees=3, maxDepth=2, seed=1)),
    ("GBTRegressor", "rows", "default", dict(maxIter=3, maxDepth=2, seed=1)),
]

# Members of the fitted models, as pyspark.ml 4.3.0 reads them. "name()" is a method with no
# arguments; a plain name is a property (or a class attribute, such as supportedImpurities).
TREE_ATTRIBUTES = [
    "depth",
    "numNodes",
    "featureImportances",
    "toDebugString",
    "numFeatures",
    "supportedImpurities",
]
ENSEMBLE_ATTRIBUTES = [
    "featureImportances",
    "trees",
    "treeWeights",
    "getNumTrees",
    "totalNumNodes",
    "toDebugString",
    "numFeatures",
    "supportedFeatureSubsetStrategies",
    "supportedImpurities",
]
ATTRIBUTES = {
    "LinearRegressionModel": [
        "coefficients",
        "intercept",
        "scale",
        "numFeatures",
        "hasSummary",
        "summary",
    ],
    "GeneralizedLinearRegressionModel": [
        "coefficients",
        "intercept",
        "numFeatures",
        "hasSummary",
        "summary",
    ],
    "AFTSurvivalRegressionModel": ["coefficients", "intercept", "scale", "numFeatures"],
    "IsotonicRegressionModel": ["boundaries", "predictions", "numFeatures"],
    "FMRegressionModel": ["factors", "linear", "intercept", "numFeatures"],
    "DecisionTreeRegressionModel": TREE_ATTRIBUTES,
    "RandomForestRegressionModel": ENSEMBLE_ATTRIBUTES,
    "GBTRegressionModel": ENSEMBLE_ATTRIBUTES + ["supportedLossTypes"],
}

# Summary members. The properties are listed here; the residuals of the generalized linear
# summaries are a method, and are in CALLS.
_LR_SUMMARY = [
    "coefficientStandardErrors",
    "degreesOfFreedom",
    "devianceResiduals",
    "explainedVariance",
    "featuresCol",
    "labelCol",
    "meanAbsoluteError",
    "meanSquaredError",
    "numInstances",
    "pValues",
    "predictionCol",
    "predictions",
    "r2",
    "r2adj",
    "residuals",
    "rootMeanSquaredError",
    "tValues",
]
_GLR_SUMMARY = [
    "aic",
    "degreesOfFreedom",
    "deviance",
    "dispersion",
    "nullDeviance",
    "numInstances",
    "predictionCol",
    "predictions",
    "rank",
    "residualDegreeOfFreedom",
    "residualDegreeOfFreedomNull",
]
SUMMARY_ATTRIBUTES = {
    "LinearRegressionSummary": _LR_SUMMARY,
    "LinearRegressionTrainingSummary": _LR_SUMMARY + ["objectiveHistory", "totalIterations"],
    "GeneralizedLinearRegressionSummary": _GLR_SUMMARY,
    "GeneralizedLinearRegressionTrainingSummary": _GLR_SUMMARY
    + ["coefficientStandardErrors", "numIterations", "pValues", "solver", "tValues"],
}

# Class methods that build a model client-side (none for the regressors in this area).
CLASS_METHODS = []

RESIDUALS = [(), ("deviance",), ("pearson",), ("working",), ("response",)]
# The arguments of each call of a method, per class. A member that is not listed is a property,
# or a method that takes no argument, which is called once without any. Every model also
# transforms the held-out rows.
CALLS = {
    "LinearRegressionModel": {
        "predict": [(DENSE3,), (SPARSE3,)],
        "evaluate": [(DATASET,)],
        "transform": [(DATASET,)],
    },
    "GeneralizedLinearRegressionModel": {
        "predict": [(DENSE3,)],
        "evaluate": [(DATASET,)],
        "transform": [(DATASET,)],
    },
    "AFTSurvivalRegressionModel": {
        "predict": [(DENSE2,)],
        "predictQuantiles": [(DENSE2,)],
        "transform": [(DATASET,)],
    },
    # Isotonic regression predicts a single feature value: inside, below and above the
    # boundaries.
    "IsotonicRegressionModel": {
        "predict": [(2.5,), (0.0,), (7.0,)],
        "transform": [(DATASET,)],
    },
    "FMRegressionModel": {"predict": [(DENSE3,)], "transform": [(DATASET,)]},
    "DecisionTreeRegressionModel": {
        "predict": [(DENSE3,)],
        "predictLeaf": [(DENSE3,)],
        "transform": [(DATASET,)],
    },
    "RandomForestRegressionModel": {
        "predict": [(DENSE3,)],
        "predictLeaf": [(DENSE3,)],
        "transform": [(DATASET,)],
    },
    "GBTRegressionModel": {
        "predict": [(DENSE3,)],
        "predictLeaf": [(DENSE3,)],
        "evaluateEachIteration": [(DATASET, "squared"), (DATASET, "absolute")],
        "transform": [(DATASET,)],
    },
    "GeneralizedLinearRegressionSummary": {"residuals": RESIDUALS},
    "GeneralizedLinearRegressionTrainingSummary": {"residuals": RESIDUALS},
}

# The objects whose members are captured besides the fitted models: their training summary, the
# summary of evaluate on the held-out rows, and the first tree of each ensemble.
CHILDREN = {
    "LinearRegressionModel": ["summary", ("evaluate", DATASET)],
    "GeneralizedLinearRegressionModel": ["summary", ("evaluate", DATASET)],
    "RandomForestRegressionModel": [("trees", 0)],
    "GBTRegressionModel": [("trees", 0)],
}

# The model class that each estimator fits.
MODEL_CLASS = {
    "LinearRegression": "LinearRegressionModel",
    "GeneralizedLinearRegression": "GeneralizedLinearRegressionModel",
    "AFTSurvivalRegression": "AFTSurvivalRegressionModel",
    "IsotonicRegression": "IsotonicRegressionModel",
    "FMRegressor": "FMRegressionModel",
    "DecisionTreeRegressor": "DecisionTreeRegressionModel",
    "RandomForestRegressor": "RandomForestRegressionModel",
    "GBTRegressor": "GBTRegressionModel",
}

COLUMNS3 = [("label", "double"), ("weight", "double"), ("features", "vector")]
COLUMNS_AFT = [("label", "double"), ("censor", "double"), ("features", "vector")]


def datasets():
    """Each dataset: (columns, rows, held-out rows). Vector columns are dense lists."""
    rows = [(label, weight, list(x)) for label, weight, x in zip(LABELS, WEIGHTS, FEATURES)]
    held = [(label, 1.0, list(x)) for label, x in zip(HELD_OUT_LABELS, HELD_OUT)]
    counts = [(c, weight, list(x)) for c, weight, x in zip(COUNTS, WEIGHTS, FEATURES)]
    counts_held = [(3.0, 1.0, list(x)) for x in HELD_OUT]
    aft = [(label, censor, list(x)) for label, censor, x in AFT_ROWS]
    isotonic = [(label, weight, list(x)) for label, weight, x in ISOTONIC_ROWS]
    return {
        "rows": (COLUMNS3, rows, held),
        "counts": (COLUMNS3, counts, counts_held),
        "aft": (COLUMNS_AFT, aft, AFT_HELD_OUT),
        "isotonic": (COLUMNS3, isotonic, ISOTONIC_HELD_OUT),
    }


def build_frames(golden):
    frames = {}
    for name, (columns, rows, held_out) in datasets().items():
        frames[name] = golden.dataset(name, columns, rows)
        frames[name + "_held_out"] = golden.dataset(name + "_held_out", columns, held_out)
    return frames


def _label(method, args):
    parts = []
    for a in args:
        if a is DATASET:
            continue
        if isinstance(a, DenseVector):
            parts.append("dense")
        elif isinstance(a, SparseVector):
            parts.append("sparse")
        else:
            parts.append(str(a))
    return "-".join([method] + parts)


def arguments(name, frame):
    """The labelled calls of the class `name`; DATASET stands for `frame`."""
    calls, labels = [], set()
    for method, arg_lists in CALLS.get(name, {}).items():
        for args in arg_lists:
            label = _label(method, args)
            if label in labels:
                raise AssertionError(f"{name}: duplicate call label {label}")
            labels.add(label)
            calls.append((label, method, tuple(frame if a is DATASET else a for a in args)))
    return calls


def persist_arguments(name, frame):
    """The copy's calls: the predictions and the transform."""
    return [c for c in arguments(name, frame) if c[1] in ("predict", "transform")]


def _uid(key):
    return re.sub(r"[^A-Za-z0-9_]", "_", key)


def capture(golden, frames, out_models, out_summaries):
    for cls_name, dataset, label, params in FITS:
        key = f"{cls_name}.{label}"
        estimator = getattr(R, cls_name)(**params)
        frame = frames[dataset]
        held_out = frames[dataset + "_held_out"]
        model = golden.fit(key, estimator, frame, _uid(key))
        if model is None:
            continue
        model_cls = MODEL_CLASS[cls_name]
        assert type(model).__name__ == model_cls, type(model).__name__
        golden.record_members(
            out_models,
            key,
            model,
            model_cls,
            "model",
            attributes=ATTRIBUTES[model_cls],
            arguments=arguments,
            frame=held_out,
        )
        for child in CHILDREN.get(model_cls, []):
            _child(golden, out_models, out_summaries, key, model, child, dataset, held_out)
        golden.persist(
            key + ".model",
            model,
            model_cls,
            "model",
            arguments=persist_arguments,
            frame=held_out,
        )
        golden.persist(key + ".estimator", estimator, cls_name, "estimator")


def _child(golden, out_models, out_summaries, key, model, child, dataset, held_out):
    if child == "summary":
        summary, via = golden.call("model.summary", partial(getattr, model, "summary"))
        _summary(golden, out_summaries, key + ".training", summary, via)
    elif child == ("evaluate", DATASET):
        summary, via = golden.call(
            f"model.evaluate({dataset}_held_out)", partial(model.evaluate, held_out)
        )
        _summary(golden, out_summaries, key + ".evaluate", summary, via)
    elif child[0] == "trees":
        i = child[1]
        tree = model.trees[i]
        golden.record_members(
            out_models,
            f"{key}.trees[{i}]",
            tree,
            "DecisionTreeRegressionModel",
            f"model.trees[{i}]",
            attributes=TREE_ATTRIBUTES,
            arguments=arguments,
            frame=held_out,
        )
    else:
        raise AssertionError(f"unknown child {child!r}")


def _summary(golden, out, key, summary, via):
    cls_name = type(summary).__name__
    record = golden.members(
        summary,
        cls_name,
        "summary",
        attributes=SUMMARY_ATTRIBUTES[cls_name],
        arguments=arguments,
    )
    record["via"] = via
    out[key] = record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=Path("tests/golden/regression_models.json"))
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    golden = Golden(spark)
    try:
        frames = build_frames(golden)
        models, summaries = {}, {}
        capture(golden, frames, models, summaries)
        golden.models.update(models)
        golden.summaries.update(summaries)
        golden.class_methods(CLASS_METHODS, lambda name: getattr(R, name), frames["rows_held_out"])
        golden.dump(
            args.out,
            {
                "fits": golden.fits,
                "models": golden.models,
                "class_methods": golden.class_methods_out,
            },
        )
        golden.dump(
            args.out.with_name("regression_summaries.json"), {"summaries": golden.summaries}
        )
        golden.dump(
            args.out.with_name("regression_persistence.json"),
            {"persistence": golden.persistence},
        )
    finally:
        golden.close()
        spark.stop()


if __name__ == "__main__":
    main()
