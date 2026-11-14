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
"""The cases of the native transformers of `pyspark.ml.feature` (the API of PySpark 4.3.0 over
Spark Connect): their Params API, their getters, their transform, and their persistence.

`offline_cases(name)` needs no server; `live_cases(name, spark, base)` runs against one, with
`base` a scratch directory of the server's file system. The official classes of the same cases
are the goldens of tests/golden/feature_api_cases.json (scripts/capture_feature_api_cases_golden.py).
"""

import re
import shutil

from _ml_api_cases import Recorder
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
    }[name]


def classes(name):
    """The native class of `name`, and PySpark's class of it as the reference."""
    return native_class(FEATURE, name, REFERENCE)


_UID = re.compile(r"([A-Z][A-Za-z]*)_[0-9a-f]{12}")


def _s(value):
    """The text of `value`, with the uids of the classes normalized (they are random)."""
    return _UID.sub(r"\1_<uid>", str(value))


def _out(value):
    """The value of a case, with the uids of the classes normalized, and a DataFrame as its
    columns and sorted rows (`Recorder` of a DataFrame reads its columns as they are)."""
    if hasattr(value, "collect") and hasattr(value, "columns"):
        return {
            "columns": [_s(c) for c in value.columns],
            "rows": sorted(repr(tuple(r)) for r in value.collect()),
        }
    if isinstance(value, (list, tuple)):
        return [_out(x) for x in value]
    if isinstance(value, str):
        return _s(value)
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


def _params(stage):
    return sorted((p.name, _s(repr(v))) for p, v in stage.extractParamMap().items())


def _has_output_col(stage):
    return stage.hasParam("outputCol")


def offline_cases(name):
    """The cases of the transformer `name` that need no server: its Params API and its members."""
    cls = classes(name)
    stage = cls(**_kwargs(name))
    rec = _recorder()
    from pyspark.ml.base import Transformer
    from pyspark.ml.param import Params

    rec("isinstance_transformer", lambda: isinstance(stage, Transformer))
    rec("isinstance_params", lambda: isinstance(stage, Params))
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
        getter = "get" + p.name[0].upper() + p.name[1:]
        rec(getter, lambda getter=getter: _s(getattr(stage, getter)()))
        rec("isSet_" + p.name, lambda p=p: stage.isSet(p.name))
        rec("isDefined_" + p.name, lambda p=p: stage.isDefined(p))
        rec("hasDefault_" + p.name, lambda p=p: stage.hasDefault(p))
        rec("explainParam_" + p.name, lambda p=p: _s(stage.explainParam(p)))
    rec("getParam_nope", lambda: stage.getParam("nope"))
    rec("set_bad_name", lambda: stage.set("nope", 1))
    return rec.out


def live_cases(name, spark, base):
    """The cases of the transformer `name` against a server: its transform (also with a param
    map, which applies to a copy), and its save and load, of its uid, params and output."""
    cls = classes(name)
    stage = cls(**_kwargs(name))
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

