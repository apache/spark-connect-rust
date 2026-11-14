#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""The cases of the native feature classes of `pyspark.ml.feature` (the API of PySpark 4.3.0 over
Spark Connect): the transformers' Params API, getters, transform and persistence; the estimators'
Params API, fits and persistence; and the fitted models' members, transforms and persistence.

`offline_cases(name)` needs no server; `live_cases(name, spark, base)` runs against one, with
`base` a scratch directory of the server's file system. The official classes of the same cases
are the goldens of tests/golden/feature_api_cases.json (scripts/capture_feature_api_cases_golden.py).
"""

import re
import shutil

from _ml_api_cases import Recorder, _normalize
from _native_classes import native_class

FEATURE = "pyspark.ml.connect.feature"
REFERENCE = "pyspark.ml.feature"

# The transformers of the case sets, with the keyword arguments of each case.
TRANSFORMERS = [
    "Binarizer",
    "Bucketizer",
    "DCT",
    "ElementwiseProduct",
    "FeatureHasher",
    "HashingTF",
    "IndexToString",
    "Interaction",
    "NGram",
    "Normalizer",
    "PolynomialExpansion",
    "RegexTokenizer",
    "SQLTransformer",
    "StopWordsRemover",
    "Tokenizer",
    "VectorAssembler",
    "VectorSizeHint",
    "VectorSlicer",
]

# The estimators of the case sets, with the keyword arguments of each case. QuantileDiscretizer's
# fit gives a Bucketizer.
ESTIMATORS = [
    "BucketedRandomProjectionLSH",
    "ChiSqSelector",
    "CountVectorizer",
    "IDF",
    "Imputer",
    "MaxAbsScaler",
    "MinHashLSH",
    "MinMaxScaler",
    "OneHotEncoder",
    "PCA",
    "QuantileDiscretizer",
    "RFormula",
    "RobustScaler",
    "StandardScaler",
    "StringIndexer",
    "TargetEncoder",
    "UnivariateFeatureSelector",
    "VarianceThresholdSelector",
    "VectorIndexer",
    "Word2Vec",
]

# The fitted models of the case sets: each is the model of the estimator of its name without
# "Model".
MODELS = [
    "BucketedRandomProjectionLSHModel",
    "ChiSqSelectorModel",
    "CountVectorizerModel",
    "IDFModel",
    "ImputerModel",
    "MaxAbsScalerModel",
    "MinHashLSHModel",
    "MinMaxScalerModel",
    "OneHotEncoderModel",
    "PCAModel",
    "RFormulaModel",
    "RobustScalerModel",
    "StandardScalerModel",
    "StringIndexerModel",
    "TargetEncoderModel",
    "UnivariateFeatureSelectorModel",
    "VarianceThresholdSelectorModel",
    "VectorIndexerModel",
    "Word2VecModel",
]


def _kwargs(name):
    from pyspark.ml.linalg import Vectors

    return {
        "Binarizer": {"inputCol": "x", "threshold": 0.5},
        "Bucketizer": {"inputCol": "x", "splits": [0.0, 0.5, 1.0]},
        "DCT": {"inputCol": "v"},
        "ElementwiseProduct": {"inputCol": "v", "scalingVec": Vectors.dense([1.0, 2.0])},
        "FeatureHasher": {"inputCols": ["a", "b"], "numFeatures": 16},
        "HashingTF": {"inputCol": "tokens", "numFeatures": 16},
        "IndexToString": {"inputCol": "i", "labels": ["zero", "one"]},
        "Interaction": {"inputCols": ["a", "b"]},
        "NGram": {"inputCol": "tokens", "n": 2},
        "Normalizer": {"inputCol": "v", "p": 1.0},
        "PolynomialExpansion": {"inputCol": "v", "degree": 2},
        "RegexTokenizer": {"inputCol": "s", "pattern": "\\W+"},
        "SQLTransformer": {"statement": "SELECT *, a + b AS c FROM __THIS__"},
        "StopWordsRemover": {"inputCol": "tokens", "stopWords": ["big"]},
        "Tokenizer": {"inputCol": "s"},
        "VectorAssembler": {"inputCols": ["a", "b"]},
        "VectorSizeHint": {"inputCol": "v", "size": 2, "handleInvalid": "skip"},
        "VectorSlicer": {"inputCol": "v", "indices": [1]},
        "BucketedRandomProjectionLSH": {
            "inputCol": "v", "outputCol": "h", "seed": 1, "numHashTables": 2, "bucketLength": 2.0,
        },
        "ChiSqSelector": {
            "featuresCol": "features", "labelCol": "label", "outputCol": "sel", "numTopFeatures": 2,
        },
        "CountVectorizer": {"inputCol": "tokens", "outputCol": "cv", "vocabSize": 5},
        "IDF": {"inputCol": "v", "outputCol": "idf"},
        "Imputer": {"inputCols": ["x", "a"], "outputCols": ["x2", "a2"]},
        "MaxAbsScaler": {"inputCol": "v", "outputCol": "ma"},
        "MinHashLSH": {"inputCol": "v", "outputCol": "mh", "seed": 1, "numHashTables": 2},
        "MinMaxScaler": {"inputCol": "v", "outputCol": "mm", "min": 0.0, "max": 2.0},
        "OneHotEncoder": {"inputCol": "idx", "outputCol": "ohe"},
        "PCA": {"inputCol": "v", "outputCol": "pca", "k": 2},
        "QuantileDiscretizer": {"inputCol": "x", "outputCol": "qd", "numBuckets": 3},
        "RFormula": {"formula": "label ~ x + a", "featuresCol": "rf"},
        "RobustScaler": {"inputCol": "v", "outputCol": "rs"},
        "StandardScaler": {"inputCol": "v", "outputCol": "ss", "withMean": True},
        "StringIndexer": {"inputCol": "s", "outputCol": "si"},
        "TargetEncoder": {"inputCols": ["idx"], "outputCols": ["idx_te"], "labelCol": "label"},
        "UnivariateFeatureSelector": {
            "featuresCol": "features",
            "labelCol": "label",
            "outputCol": "uf",
            "selectionMode": "numTopFeatures",
        },
        "VarianceThresholdSelector": {"featuresCol": "v", "outputCol": "vt", "varianceThreshold": 0.1},
        "VectorIndexer": {"inputCol": "v", "outputCol": "vi", "maxCategories": 3},
        "Word2Vec": {
            "inputCol": "tokens",
            "outputCol": "w2v",
            "vectorSize": 2,
            "minCount": 0,
            "seed": 42,
            "numPartitions": 1,
            "maxIter": 2,
        },
    }[name]


def classes(name):
    """The native class of `name`, and PySpark's class of it as the reference."""
    return native_class(FEATURE, name, REFERENCE)


# The params that PySpark sets after the constructor (its constructor does not take them).
_AFTER = {
    "UnivariateFeatureSelector": {"featureType": "categorical", "labelType": "categorical"},
}


def _make(name):
    """An instance of `name` with its keyword arguments, and the params that are set after it."""
    stage = classes(name)(**_kwargs(name))
    for param, value in _AFTER.get(name, {}).items():
        getattr(stage, "set" + param[0].upper() + param[1:])(value)
    return stage


# A uid of a class, which a column name may end with too (`Binarizer_<hex>__output`).
_UID = re.compile(r"([A-Z][A-Za-z]*)_[0-9a-f]{12}")


def _s(value):
    """The text of `value`, with the uids and the class-name hashes normalized (they are random)."""
    return _normalize(_UID.sub(r"\1_<uid>", str(value)))


def _cell(value):
    """A cell of a row or a value of a case, as a Python literal: a vector or matrix as its array,
    a struct as its fields, and a list element by element. A float is rounded to six decimals, so
    that the golden is stable between captures; the cases compare floats within a tolerance
    (see test_feature_api_e2e.py)."""
    if hasattr(value, "toArray"):
        return {"array": value.toArray().round(6).tolist()}
    if hasattr(value, "asDict"):
        return {key: _cell(v) for key, v in value.asDict().items()}
    if isinstance(value, list):
        return [_cell(x) for x in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def _out(value):
    """The value of a case, with the uids of the classes normalized, and a DataFrame as its
    columns and sorted rows (`Recorder` of a DataFrame reads its columns as they are)."""
    if hasattr(value, "collect") and hasattr(value, "columns"):
        return {
            "columns": [_s(c) for c in value.columns],
            "rows": sorted(_s(repr(tuple(_cell(c) for c in r))) for r in value.collect()),
        }
    if isinstance(value, (list, tuple)):
        return [_out(x) for x in value]
    if isinstance(value, str):
        return _s(value)
    if hasattr(value, "toArray") or isinstance(value, float):
        return _cell(value)
    return value


def _recorder(base=None):
    """A `Recorder` of the cases, whose values are `_out` of what they return."""
    rec = Recorder(base)

    def case(tag, f):
        rec(tag, lambda: _out(f()))

    case.out = rec.out
    return case


def frame(spark):
    from pyspark.ml.linalg import Vectors

    rows = [
        (0.7, 1.0, 2.0, "Hello big World", ["hello", "big", "world"], Vectors.dense([3.0, 4.0]), 1.0),
        (0.2, 3.0, 5.0, "a  quick test", ["a", "quick", "test"], Vectors.dense([0.5, -1.0]), 0.0),
    ]
    columns = ["x", "a", "b", "s", "tokens", "v", "i"]
    return spark.createDataFrame(rows, columns).coalesce(1)


def fit_frame(spark):
    """The frame that the estimators fit: a label, a categorical index, and vector, text and token
    columns."""
    from pyspark.ml.linalg import Vectors

    rows = [
        (1.0, 2.0, "a", ["a", "b", "c"], Vectors.dense([1.0, 0.0, 2.0]), 0.0, 0.0),
        (2.0, 1.0, "b", ["b", "c"], Vectors.dense([0.0, 3.0, 1.0]), 1.0, 1.0),
        (3.0, 4.0, "a", ["a", "a", "d"], Vectors.dense([2.0, 1.0, 0.5]), 0.0, 2.0),
        (4.0, 3.0, "c", ["c", "d"], Vectors.dense([1.5, 2.5, 3.0]), 1.0, 0.0),
        (5.0, 6.0, "b", ["b", "a"], Vectors.dense([4.0, 0.5, 1.0]), 0.0, 1.0),
        (6.0, 5.0, "a", ["a", "b", "d"], Vectors.dense([0.5, 1.0, 4.0]), 1.0, 2.0),
    ]
    columns = ["x", "a", "s", "tokens", "v", "label", "idx"]
    df = spark.createDataFrame(rows, columns)
    return df.withColumn("features", df.v).coalesce(1)


def _params(stage):
    return sorted((p.name, _s(repr(v))) for p, v in stage.extractParamMap().items())


def _has_output_col(stage):
    return stage.hasParam("outputCol")


def _getters(stage, rec):
    for p in stage.params:
        getter = "get" + p.name[0].upper() + p.name[1:]
        if hasattr(stage, getter):
            rec(getter, lambda getter=getter: getattr(stage, getter)())


def _stage_cases(rec, stage, cls, name):
    """The Params API cases of a stage (a transformer or an estimator): its class, its params,
    copy, setParams, persistence classes and errors, and its getters."""
    rec("class", lambda: (type(stage).__name__, type(stage).__module__.split(".")[-1]))
    rec("uid_class", lambda: _s(stage.uid).split("_")[0])
    rec("param_names", lambda: [p.name for p in stage.params])
    rec("param_docs", lambda: [p.doc for p in stage.params])
    rec("param_parents", lambda: [_s(p.parent) for p in stage.params])
    rec("explainParams", lambda: _s(stage.explainParams()))
    rec("repr", lambda: _s(repr(stage)))
    rec("extract", lambda: _params(stage))
    rec("copy_extract", lambda: _params(stage.copy()))
    rec("copy_same_uid", lambda: stage.copy().uid == stage.uid)
    rec("setParams_returns_self", lambda: stage.setParams(**_kwargs(name)) is stage)
    rec("write_class", lambda: type(stage.write()).__name__)
    rec("read_class", lambda: type(cls.read()).__name__)
    rec("no_attribute", lambda: stage.fooBarBaz)
    rec("writer_no_attribute", lambda: stage.write().fooBarBaz)
    for p in stage.params:
        rec("isSet_" + p.name, lambda p=p: stage.isSet(p.name))
        rec("isDefined_" + p.name, lambda p=p: stage.isDefined(p))
        rec("hasDefault_" + p.name, lambda p=p: stage.hasDefault(p))
        rec("explainParam_" + p.name, lambda p=p: _s(stage.explainParam(p)))
    _getters(stage, rec)
    rec("getParam_nope", lambda: stage.getParam("nope"))
    rec("set_bad_name", lambda: stage.set("nope", 1))


def offline_cases(name):
    """The cases of the transformer or estimator `name` that need no server: its Params API and
    its members."""
    cls = classes(name)
    stage = _make(name)
    rec = _recorder()
    from pyspark.ml.base import Estimator, Transformer
    from pyspark.ml.param import Params

    if name in ESTIMATORS:
        rec("isinstance_estimator", lambda: isinstance(stage, Estimator))
    else:
        rec("isinstance_transformer", lambda: isinstance(stage, Transformer))
    rec("isinstance_params", lambda: isinstance(stage, Params))
    _stage_cases(rec, stage, cls, name)
    return rec.out


def _transformer_live(name, spark, base):
    """The cases of the transformer `name` against a server: its transform (also with a param
    map, which applies to a copy), and its save and load, of its uid, params and output."""
    cls = classes(name)
    stage = _make(name)
    df = frame(spark)
    rec = _recorder(base)
    rec("transform", lambda: stage.transform(df))
    rec("transform_columns", lambda: stage.transform(df).columns)
    if _has_output_col(stage):
        rec("transform_map", lambda: stage.transform(df, {cls.outputCol: "mapped"}).columns)
        rec("copy_map", lambda: _s(stage.copy({cls.outputCol: "mapped"}).getOutputCol()))
    rec("transform_bad_params", lambda: stage.transform(df, 5))
    shutil.rmtree(base, ignore_errors=True)
    rec("save", lambda: stage.write().overwrite().save(base + "/stage"))
    rec("save_again", lambda: stage.save(base + "/stage"))
    loaded = cls.load(base + "/stage")
    rec("loaded_uid", lambda: loaded.uid == stage.uid)
    rec("loaded_params", lambda: _params(loaded))
    rec("loaded_transform", lambda: loaded.transform(df))
    if _has_output_col(stage):
        # The output of a load keeps the default of the uid that the stage was made of.
        rec("loaded_output_default", lambda: _s(loaded.getOutputCol()))
        rec("loaded_output_is_set", lambda: loaded.isSet("outputCol"))
    rec("load_missing", lambda: cls.load(base + "/missing"))
    shutil.rmtree(base, ignore_errors=True)
    return rec.out


# The estimator's param that a param map sets in the cases: its output column, else its features.
def _map_param(stage):
    return "outputCol" if stage.hasParam("outputCol") else "featuresCol"


def _estimator_live(name, spark, base):
    """The cases of the estimator `name` against a server: its fit (of the model's class, of a
    param map, and of `fitMultiple`), and its save and load."""
    cls = classes(name)
    est = _make(name)
    df = fit_frame(spark)
    rec = _recorder(base)
    model = est.fit(df)
    rec("fit_model_class", lambda: type(model).__name__)
    if name != "QuantileDiscretizer":
        # QuantileDiscretizer's model is a Bucketizer of a new uid.
        rec("fit_model_uid_is_estimator", lambda: model.uid == est.uid)
    rec("fit_transform_columns", lambda: model.transform(df).columns)
    rec("fit_bad_params", lambda: est.fit(df, 5))
    param = _map_param(est)
    rec("fit_map_class", lambda: type(est.fit(df, {getattr(cls, param): "mapped"})).__name__)
    rec("fit_map", lambda: _s(est.fit(df, {getattr(cls, param): "mapped"}).transform(df).columns))
    rec("fit_list", lambda: [type(m).__name__ for m in est.fit(df, [{}, {getattr(cls, param): "m"}])])
    rec(
        "fitMultiple",
        lambda: sorted(
            (i, type(m).__name__) for i, m in est.fitMultiple(df, [{}, {getattr(cls, param): "m"}])
        ),
    )
    rec("copy_map", lambda: _s(est.copy({getattr(cls, param): "mapped"}).getOrDefault(param)))
    shutil.rmtree(base, ignore_errors=True)
    rec("save", lambda: est.write().overwrite().save(base + "/est"))
    rec("save_again", lambda: est.save(base + "/est"))
    loaded = cls.load(base + "/est")
    rec("loaded_uid", lambda: loaded.uid == est.uid)
    rec("loaded_params", lambda: _params(loaded))
    rec("loaded_isSet_" + param, lambda: loaded.isSet(param))
    rec("loaded_fit_model_class", lambda: type(loaded.fit(df)).__name__)
    rec("load_missing", lambda: cls.load(base + "/missing"))
    shutil.rmtree(base, ignore_errors=True)
    return rec.out


# The attributes of PySpark's models (their properties and DataFrames), by model.
MODEL_ATTRIBUTES = {
    "ChiSqSelectorModel": ["selectedFeatures"],
    "CountVectorizerModel": ["vocabulary"],
    "IDFModel": ["idf", "docFreq", "numDocs"],
    "ImputerModel": ["surrogateDF"],
    "MaxAbsScalerModel": ["maxAbs"],
    "MinMaxScalerModel": ["originalMin", "originalMax"],
    "OneHotEncoderModel": ["categorySizes"],
    "PCAModel": ["pc", "explainedVariance"],
    "RobustScalerModel": ["median", "range"],
    "StandardScalerModel": ["mean", "std"],
    "StringIndexerModel": ["labels", "labelsArray"],
    "UnivariateFeatureSelectorModel": ["selectedFeatures"],
    "VarianceThresholdSelectorModel": ["selectedFeatures"],
    "VectorIndexerModel": ["categoryMaps", "numFeatures"],
}


def _model_probes(name, cls, model, df, rec):
    """The members of the fitted model `model` of `name`, each as a case: the attributes, the
    DataFrame methods and the class methods of PySpark's model."""
    from pyspark.ml.linalg import Vectors

    key = Vectors.dense([1.0, 0.0, 2.0])
    if name in ("BucketedRandomProjectionLSHModel", "MinHashLSHModel"):
        # Over Spark Connect PySpark's relation decorator drops every keyword argument of these
        # methods, so a keyword distCol is ignored (the default "distCol" is used), and a call
        # with a keyword argument goes to the server with its positional arguments alone.
        rec("approxNearestNeighbors", lambda: model.approxNearestNeighbors(df, key, 2))
        rec("approxNearestNeighbors_positional", lambda: model.approxNearestNeighbors(df, key, 2, "d"))
        rec("approxNearestNeighbors_keyword_distCol", lambda: model.approxNearestNeighbors(df, key, 2, distCol="d"))
        rec("approxNearestNeighbors_keywords", lambda: model.approxNearestNeighbors(dataset=df, key=key, numNearestNeighbors=2))
        rec("approxSimilarityJoin", lambda: model.approxSimilarityJoin(df, df, 3.0))
        rec("approxSimilarityJoin_positional", lambda: model.approxSimilarityJoin(df, df, 3.0, "d"))
        rec("approxSimilarityJoin_keyword_distCol", lambda: model.approxSimilarityJoin(df, df, 3.0, distCol="d"))
        rec("approxSimilarityJoin_keywords", lambda: model.approxSimilarityJoin(datasetA=df, datasetB=df, threshold=3.0))
    for attribute in MODEL_ATTRIBUTES.get(name, []):
        rec(attribute, lambda attribute=attribute: getattr(model, attribute))
    if name == "Word2VecModel":
        rec("getVectors", lambda: model.getVectors())
        rec("findSynonyms", lambda: model.findSynonyms("a", 2))
        rec("findSynonyms_vector", lambda: model.findSynonyms(Vectors.dense([1.0, 0.0]), 2))
        rec("findSynonymsArray", lambda: model.findSynonymsArray("a", 2))
        rec("findSynonymsArray_vector", lambda: model.findSynonymsArray(Vectors.dense([1.0, 0.0]), 2))
        rec("findSynonyms_missing", lambda: model.findSynonyms("nope", 2))
        rec("findSynonyms_keyword_num", lambda: model.findSynonyms("a", num=2))
        rec("findSynonyms_keywords", lambda: model.findSynonyms(word="a", num=2))
    if name == "CountVectorizerModel":
        rec(
            "from_vocabulary",
            lambda: cls.from_vocabulary(["a", "b"], "tokens", outputCol="cv2", minTF=1.0, binary=True)
            .transform(df).columns,
        )
        rec(
            "from_vocabulary_params",
            lambda: _params(cls.from_vocabulary(["a", "b"], "tokens", minTF=2.0)),
        )
        rec("from_vocabulary_empty", lambda: cls.from_vocabulary([], "tokens"))
    if name == "StringIndexerModel":
        rec(
            "from_labels",
            lambda: _s(cls.from_labels(["a", "b"], "s", outputCol="si2", handleInvalid="keep").getParam("outputCol")),
        )
        rec(
            "from_labels_transform",
            lambda: cls.from_labels(["a", "b"], "s", outputCol="si2", handleInvalid="keep")
            .transform(df).columns,
        )
        rec(
            "from_arrays_of_labels",
            lambda: cls.from_arrays_of_labels([["a", "b"], ["c"]], ["s", "tokens"]).labelsArray,
        )
    if hasattr(model, "setOutputCol"):
        rec("setOutputCol", lambda: model.copy().setOutputCol("zz").getOutputCol())


def _model_live(name, spark, base):
    """The cases of the fitted model `name` of its estimator, against a server: its members, its
    transform, and its save and load."""
    cls = classes(name)
    est_name = name[: -len("Model")]
    est = _make(est_name)
    df = fit_frame(spark)
    model = est.fit(df)
    rec = _recorder(base)
    rec("model_uid_is_estimator_uid", lambda: model.uid == est.uid)
    rec("model_class", lambda: (type(model).__name__, type(model).__module__.split(".")[-1]))
    rec("isinstance_model", lambda: _isinstance_model(model))
    rec("model_no_attribute", lambda: model.fooBarBaz)
    rec("model_params", lambda: _params(model))
    rec("model_explainParams", lambda: _s(model.explainParams()))
    _getters(model, rec)
    rec("transform", lambda: model.transform(df))
    if model.hasParam("outputCol"):
        rec("transform_map", lambda: model.transform(df, {cls.outputCol: "mapped"}).columns)
    rec("copy_transform", lambda: model.copy().transform(df))
    _model_probes(name, cls, model, df, rec)
    shutil.rmtree(base, ignore_errors=True)
    rec("save", lambda: model.write().overwrite().save(base + "/model"))
    rec("save_again", lambda: model.save(base + "/model"))
    loaded = cls.load(base + "/model")
    rec("loaded_uid", lambda: loaded.uid == model.uid)
    rec("loaded_params", lambda: _params(loaded))
    rec("loaded_transform", lambda: loaded.transform(df))
    rec("loaded_model_class", lambda: type(loaded).__name__)
    _model_probes(name, cls, loaded, df, rec)
    shutil.rmtree(base, ignore_errors=True)
    return rec.out


def _isinstance_model(model):
    from pyspark.ml.base import Model, Transformer

    return isinstance(model, Model), isinstance(model, Transformer)


def live_cases(name, spark, base):
    """The cases of `name` against a server: a transformer, an estimator or a fitted model."""
    if name in MODELS:
        return _model_live(name, spark, base)
    if name in ESTIMATORS:
        return _estimator_live(name, spark, base)
    return _transformer_live(name, spark, base)
