#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture the clustering models and summaries golden from official PySpark 4.3.0.

Each estimator variant is fitted on a fixed one-partition dataset with fixed seeds and pinned
uids. The fitted model's members, its children (the training summary, and the LocalLDAModel of
toLocal) and the summaries' members are captured by the shared helper
(scripts/ml_golden_common.py), together with the requests the client sends. Models and estimators
are persisted and loaded back. PowerIterationClustering fits no model: its estimator is persisted
and loaded back, and its assignClusters is captured by
scripts/capture_remaining_clustering_golden.py.

Run with the official client, for example:
    /tmp/scr-ref-py scripts/capture_clustering_models_golden.py \
        --remote sc://localhost:15013 --out tests/golden/clustering_models.json

The summaries go to the sibling file clustering_summaries.json, the copies to
clustering_persistence.json.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from functools import partial  # noqa: E402

from ml_golden_common import Golden  # noqa: E402
from pyspark.ml import clustering as C  # noqa: E402
from pyspark.ml.linalg import (  # noqa: E402
    DenseVector,
    SparseVector,
    Vectors,
)
from pyspark.sql import SparkSession  # noqa: E402

# Three blobs of points, none at the origin, so that the cosine distance takes them too.
POINTS = [
    [1.0, 1.0],
    [1.5, 1.2],
    [1.1, 1.6],
    [1.4, 1.4],
    [6.0, 6.0],
    [6.5, 5.8],
    [5.9, 6.6],
    [6.2, 6.3],
    [10.0, 1.5],
    [10.4, 1.1],
    [9.8, 1.9],
]
WEIGHTS = [1.0, 2.0, 1.0, 1.0, 0.5, 1.0, 1.0, 2.0, 1.0, 1.0, 1.5]
# Two blobs of six points each, for the mixture of two gaussians.
MIXTURE_POINTS = [
    [1.0, 1.0],
    [1.6, 1.3],
    [0.8, 1.7],
    [1.3, 0.6],
    [2.0, 1.4],
    [1.2, 2.1],
    [7.0, 7.0],
    [7.7, 6.4],
    [6.6, 7.9],
    [7.4, 7.6],
    [8.1, 6.9],
    [6.9, 6.2],
]
HELD_OUT = [[1.2, 1.3], [6.1, 6.2], [10.1, 1.4], [4.0, 4.0]]

# Two topics of six terms.
DOCUMENTS = [
    [10.0, 8.0, 2.0, 0.0, 0.0, 0.0],
    [8.0, 10.0, 1.0, 0.0, 0.0, 0.0],
    [9.0, 7.0, 3.0, 0.0, 0.0, 0.0],
    [12.0, 6.0, 2.0, 0.0, 0.0, 0.0],
    [7.0, 9.0, 3.0, 0.0, 0.0, 0.0],
    [11.0, 8.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 10.0, 8.0, 2.0],
    [0.0, 0.0, 0.0, 8.0, 10.0, 1.0],
    [0.0, 0.0, 0.0, 9.0, 7.0, 3.0],
    [0.0, 0.0, 0.0, 12.0, 6.0, 2.0],
    [0.0, 0.0, 0.0, 7.0, 9.0, 3.0],
    [0.0, 0.0, 0.0, 11.0, 8.0, 1.0],
]
HELD_OUT_DOCUMENTS = [
    [8.0, 6.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 10.0, 9.0, 2.0],
    [0.0] * 6,
]

# The single values that predict and predictProbability take.
POINT = Vectors.dense([6.1, 6.2])
SPARSE_POINT = Vectors.sparse(2, [1], [1.3])

# "DATASET" stands for the held-out DataFrame of the family, which the capture builds.
DATASET = object()

LDA_PARAMS = dict(k=2, maxIter=10, seed=42, checkpointInterval=-1)

# The fits: (estimator class, dataset name, label, params). The label names the variant in its
# key, "<estimator>.<label>". Each dataset has a held-out frame, "<dataset>_held_out".
# The two LDA optimizers fit the two model classes: online a LocalLDAModel, and em a
# DistributedLDAModel.
FITS = [
    ("KMeans", "points", "default", dict(k=3, seed=5)),
    (
        "KMeans",
        "points",
        "cosine-random",
        dict(k=3, seed=5, distanceMeasure="cosine", initMode="random"),
    ),
    ("KMeans", "points", "weighted", dict(k=3, seed=5, weightCol="weight")),
    ("BisectingKMeans", "points", "default", dict(k=3, seed=5)),
    (
        "BisectingKMeans",
        "points",
        "min-divisible-2",
        dict(k=3, seed=5, minDivisibleClusterSize=2.0),
    ),
    ("BisectingKMeans", "points", "cosine", dict(k=3, seed=5, distanceMeasure="cosine")),
    ("GaussianMixture", "mixture", "default", dict(k=2, seed=11)),
    ("GaussianMixture", "mixture", "weighted", dict(k=2, seed=11, weightCol="weight")),
    ("LDA", "documents", "online", dict(LDA_PARAMS, optimizer="online")),
    (
        "LDA",
        "documents",
        "em",
        dict(LDA_PARAMS, optimizer="em", docConcentration=[2.0, 2.0], topicConcentration=2.0),
    ),
]

# Members of the fitted models, as pyspark.ml 4.3.0 reads them. "name()" is a method with no
# arguments; a plain name is a property (or a class attribute).
_LDA_ATTRIBUTES = [
    "estimatedDocConcentration()",
    "isDistributed()",
    "topicsMatrix()",
    "vocabSize()",
]
ATTRIBUTES = {
    "KMeansModel": ["clusterCenters()", "numFeatures", "hasSummary", "summary"],
    "BisectingKMeansModel": ["clusterCenters()", "numFeatures", "hasSummary", "summary"],
    "GaussianMixtureModel": [
        "gaussians",
        "gaussiansDF",
        "weights",
        "numFeatures",
        "hasSummary",
        "summary",
    ],
    "LocalLDAModel": _LDA_ATTRIBUTES,
    "DistributedLDAModel": _LDA_ATTRIBUTES
    + ["getCheckpointFiles()", "logPrior()", "trainingLogLikelihood()"],
}

# Summary members: all properties.
_CLUSTERING_SUMMARY = [
    "cluster",
    "clusterSizes",
    "featuresCol",
    "k",
    "numIter",
    "predictionCol",
    "predictions",
]
SUMMARY_ATTRIBUTES = {
    "KMeansSummary": _CLUSTERING_SUMMARY + ["trainingCost"],
    "BisectingKMeansSummary": _CLUSTERING_SUMMARY + ["trainingCost"],
    "GaussianMixtureSummary": _CLUSTERING_SUMMARY
    + ["logLikelihood", "probability", "probabilityCol"],
}

# Class methods that build a model client-side (none for the clustering models in this area).
CLASS_METHODS = []

# The arguments of each call of a method, per class. A member that is not listed is a property,
# or a method that takes no argument, which is called once without any. Every model also
# transforms the held-out rows.
_LDA_CALLS = {
    "describeTopics": [(), (3,)],
    "logLikelihood": [(DATASET,)],
    "logPerplexity": [(DATASET,)],
    "transform": [(DATASET,)],
}
CALLS = {
    "KMeansModel": {
        "predict": [(POINT,), (SPARSE_POINT,)],
        "transform": [(DATASET,)],
    },
    "BisectingKMeansModel": {
        "predict": [(POINT,)],
        "computeCost": [(DATASET,)],
        "transform": [(DATASET,)],
    },
    "GaussianMixtureModel": {
        "predict": [(POINT,)],
        "predictProbability": [(POINT,)],
        "transform": [(DATASET,)],
    },
    "LocalLDAModel": _LDA_CALLS,
    "DistributedLDAModel": dict(_LDA_CALLS, toLocal=[()]),
}

# The objects whose members are captured besides the fitted models: the training summaries, and
# the LocalLDAModel of toLocal.
CHILDREN = {
    "KMeansModel": ["summary"],
    "BisectingKMeansModel": ["summary"],
    "GaussianMixtureModel": ["summary"],
    "DistributedLDAModel": [("toLocal",)],
}

# The model class that each fit of an estimator builds (LDA depends on its optimizer).
MODEL_CLASS = {
    "KMeans": "KMeansModel",
    "BisectingKMeans": "BisectingKMeansModel",
    "GaussianMixture": "GaussianMixtureModel",
}

COLUMNS_WEIGHTED = [("weight", "double"), ("features", "vector")]
COLUMNS_DOCUMENTS = [("id", "int"), ("features", "vector")]


def datasets():
    """Each dataset: (columns, rows, held-out rows). Vector columns are dense lists."""
    return {
        "points": (
            COLUMNS_WEIGHTED,
            [(w, list(x)) for w, x in zip(WEIGHTS, POINTS)],
            [(1.0, list(x)) for x in HELD_OUT],
        ),
        "mixture": (
            COLUMNS_WEIGHTED,
            [(w, list(x)) for w, x in zip(WEIGHTS + [1.0], MIXTURE_POINTS)],
            [(1.0, list(x)) for x in HELD_OUT[:2]],
        ),
        "documents": (
            COLUMNS_DOCUMENTS,
            [(i, list(x)) for i, x in enumerate(DOCUMENTS)],
            [(i, list(x)) for i, x in enumerate(HELD_OUT_DOCUMENTS)],
        ),
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
        estimator = getattr(C, cls_name)(**params)
        frame = frames[dataset]
        held_out = frames[dataset + "_held_out"]
        model = golden.fit(key, estimator, frame, _uid(key))
        if model is None:
            continue
        model_cls = type(model).__name__
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
            _child(golden, out_models, out_summaries, key, model, child, held_out)
        golden.persist(
            key + ".model",
            model,
            model_cls,
            "model",
            arguments=persist_arguments,
            frame=held_out,
        )
        golden.persist(key + ".estimator", estimator, cls_name, "estimator")


def _child(golden, out_models, out_summaries, key, model, child, held_out):
    if child == "summary":
        summary, via = golden.call("model.summary", partial(getattr, model, "summary"))
        _summary(golden, out_summaries, key + ".training", summary, via)
    elif child == ("toLocal",):
        local, via = golden.call("model.toLocal()", model.toLocal)
        golden.record_members(
            out_models,
            key + ".toLocal",
            local,
            "LocalLDAModel",
            "model.toLocal()",
            attributes=ATTRIBUTES["LocalLDAModel"],
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
    )
    record["via"] = via
    out[key] = record


def power_iteration(golden, out_models):
    """PowerIterationClustering fits no model: its members and its persisted copy."""
    pic = C.PowerIterationClustering(k=2, maxIter=10, initMode="degree")
    out_models["PowerIterationClustering.estimator"] = golden.members(
        pic,
        "PowerIterationClustering",
        "estimator",
        skip={
            "assignClusters": "covered by capture_remaining_clustering_golden.py",
            "setParams": "Params API (client side): keyword setter of the params",
        },
    )
    golden.persist(
        "PowerIterationClustering.estimator",
        pic,
        "PowerIterationClustering",
        "estimator",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=Path("tests/golden/clustering_models.json"))
    args = parser.parse_args()
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    golden = Golden(spark)
    try:
        frames = build_frames(golden)
        models, summaries = {}, {}
        capture(golden, frames, models, summaries)
        power_iteration(golden, models)
        golden.models.update(models)
        golden.summaries.update(summaries)
        golden.class_methods(
            CLASS_METHODS, lambda name: getattr(C, name), frames["points_held_out"]
        )
        golden.dump(
            args.out,
            {
                "fits": golden.fits,
                "models": golden.models,
                "class_methods": golden.class_methods_out,
            },
        )
        golden.dump(
            args.out.with_name("clustering_summaries.json"), {"summaries": golden.summaries}
        )
        golden.dump(
            args.out.with_name("clustering_persistence.json"),
            {"persistence": golden.persistence},
        )
    finally:
        golden.close()
        spark.stop()


if __name__ == "__main__":
    main()
