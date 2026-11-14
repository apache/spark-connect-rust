# SPDX-License-Identifier: Apache-2.0
"""The members of the fitted feature models that the model golden captures from official PySpark.

Each case of tests/golden/ml_features.json whose stage is an estimator is fitted on that fixture's
dataset with its values. Its model then gives every attribute in ATTRIBUTES, and the result (or the
error) of every call that `arguments` builds against the dataset.
CLASS_METHODS are the model class methods that build a model client-side, as
`CountVectorizerModel.from_vocabulary`, each with its transform of the dataset. The shared helper
(scripts/ml_golden_common.py) renders the values and records the requests.
"""

from pyspark.ml.linalg import Vectors

# Attributes and methods without arguments, as pyspark.ml 4.3.0 reads them: a property is read, and
# a method is called. The model class is the one of official PySpark, which for
# QuantileDiscretizer is the client-built Bucketizer.
ATTRIBUTES = {
    "BucketedRandomProjectionLSHModel": [],
    "Bucketizer": ["getSplits()", "getSplitsArray()"],
    "ChiSqSelectorModel": ["selectedFeatures"],
    "CountVectorizerModel": ["vocabulary"],
    "IDFModel": ["idf", "docFreq", "numDocs"],
    "ImputerModel": ["surrogateDF"],
    "MaxAbsScalerModel": ["maxAbs"],
    "MinHashLSHModel": [],
    "MinMaxScalerModel": ["originalMin", "originalMax"],
    "OneHotEncoderModel": ["categorySizes"],
    "PCAModel": ["pc", "explainedVariance"],
    "RFormulaModel": [],
    "RobustScalerModel": ["median", "range"],
    "StandardScalerModel": ["mean", "std"],
    "StringIndexerModel": ["labels", "labelsArray"],
    "TargetEncoderModel": [],
    "UnivariateFeatureSelectorModel": ["selectedFeatures"],
    "VarianceThresholdSelectorModel": ["selectedFeatures"],
    "VectorIndexerModel": ["numFeatures", "categoryMaps"],
    "Word2VecModel": ["getVectors()"],
}

# Keys of the nearest neighbours of the LSH models, of the dataset's 4-dimensional `features`.
_BRP_KEY = Vectors.dense([1.0, 4.0, 1.0, 2.0])
_MINHASH_KEY = Vectors.sparse(4, [1, 3], [1.0, 1.0])
_SYNONYM_VECTOR = Vectors.dense([0.1, -0.2, 0.3])


def arguments(name, frame, pairs):
    """The calls of the model `name`, a list of (label, method, args), against `frame`.

    The similarity joins take `pairs`, the dataset's id and features, so that the joined rows stay
    small.
    """
    return [("transform", "transform", (frame,))] + _calls(name, frame, pairs)


def persist_arguments(name, frame):
    """The calls of a saved and loaded model: its transform of `frame`."""
    return [("transform", "transform", (frame,))]


def _calls(name, frame, pairs):
    if name == "BucketedRandomProjectionLSHModel":
        return [
            ("nearest-2", "approxNearestNeighbors", (frame, _BRP_KEY, 2)),
            ("nearest-3-dist", "approxNearestNeighbors", (frame, _BRP_KEY, 3, "dist")),
            ("join-3", "approxSimilarityJoin", (pairs, pairs, 3.0)),
            ("join-3-dist", "approxSimilarityJoin", (pairs, pairs, 3.0, "dist")),
        ]
    if name == "MinHashLSHModel":
        return [
            ("nearest-2", "approxNearestNeighbors", (frame, _MINHASH_KEY, 2)),
            ("nearest-3-dist", "approxNearestNeighbors", (frame, _MINHASH_KEY, 3, "dist")),
            ("join-0.6", "approxSimilarityJoin", (pairs, pairs, 0.6)),
            ("join-0.6-dist", "approxSimilarityJoin", (pairs, pairs, 0.6, "dist")),
        ]
    if name == "Word2VecModel":
        return [
            ("synonyms-word", "findSynonyms", ("alpha", 2)),
            ("synonyms-vector", "findSynonyms", (_SYNONYM_VECTOR, 2)),
            ("synonyms-unknown", "findSynonyms", ("zeta", 2)),
            ("synonyms-array-word", "findSynonymsArray", ("alpha", 2)),
            ("synonyms-array-vector", "findSynonymsArray", (_SYNONYM_VECTOR, 2)),
        ]
    return []


# Class methods that build a model client-side, each with the keyword arguments it is given, and
# the transform of the dataset by the model.
CLASS_METHODS = [
    (
        "CountVectorizerModel",
        "from_vocabulary",
        "plain",
        (["alpha", "beta", "zeta"],),
        {"inputCol": "words", "outputCol": "out"},
    ),
    (
        "CountVectorizerModel",
        "from_vocabulary",
        "min-tf-binary",
        (["alpha", "beta", "zeta"],),
        {"inputCol": "words", "outputCol": "out", "minTF": 2.0, "binary": True},
    ),
    (
        "CountVectorizerModel",
        "from_vocabulary",
        "empty",
        ([],),
        {"inputCol": "words", "outputCol": "out"},
    ),
    (
        "StringIndexerModel",
        "from_labels",
        "plain",
        (["c", "a", "b"],),
        {"inputCol": "category", "outputCol": "out"},
    ),
    (
        "StringIndexerModel",
        "from_labels",
        "keep-unseen",
        (["c", "a"],),
        {"inputCol": "category", "outputCol": "out", "handleInvalid": "keep"},
    ),
    (
        "StringIndexerModel",
        "from_arrays_of_labels",
        "plain",
        ([["c", "a", "b"], ["alpha", "alpha beta"]],),
        {
            "inputCols": ["category", "text"],
            "outputCols": ["out", "other"],
            "handleInvalid": "keep",
        },
    ),
]
