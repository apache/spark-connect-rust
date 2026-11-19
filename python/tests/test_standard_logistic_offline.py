# SPDX-License-Identifier: Apache-2.0
"""The standard import delegates operations while preserving Python ML semantics."""

import copy
import os
import pickle
import subprocess
import sys

import pytest
from pyspark import _pyspark
from pyspark.ml import Estimator
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel


@pytest.mark.parametrize(
    "first", ["pyspark.ml.classification", "pyspark.ml.connect.classification"],
)
@pytest.mark.parametrize("persistence_first", ["read", "write"])
def test_public_classes_are_identical_in_either_import_order(first, persistence_first):
    code = """
import importlib
import pickle
from pyspark import _pyspark
importlib.import_module(FIRST)
public = importlib.import_module("pyspark.ml.classification")
# Exercise persistence before the explicit legacy import. That import used to be necessary
# to register the native reader/writer classes with their Python API (overwrite, option, ...).
lr = public.LogisticRegression(maxIter=3)
getattr(lr, PERSISTENCE_FIRST)()
writer = lr.write()
assert writer.overwrite() is writer and writer.shouldOverwrite
assert writer.option("example", "value") is writer
assert public.LogisticRegression.read()._clazz is public.LogisticRegression
legacy = importlib.import_module("pyspark.ml.connect.classification")
names = (
    "LogisticRegression", "LogisticRegressionModel", "LogisticRegressionSummary",
    "LogisticRegressionTrainingSummary", "BinaryLogisticRegressionSummary",
    "BinaryLogisticRegressionTrainingSummary",
)
for name in names:
    cls = getattr(public, name)
    assert cls is getattr(legacy, name)
    assert cls.__module__ == "pyspark.ml.classification"
    assert issubclass(cls, getattr(_pyspark, name))
    assert pickle.loads(pickle.dumps(cls)) is cls
assert not issubclass(public.LinearSVC, _pyspark.LinearSVC)
"""
    code = code.replace("PERSISTENCE_FIRST", repr(persistence_first)).replace("FIRST", repr(first))
    subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ), check=True, capture_output=True, text=True,
    )


def test_params_copy_and_pickle_keep_the_standard_api():
    lr = LogisticRegression(maxIter=3, regParam=0.1)
    assert isinstance(lr, _pyspark.LogisticRegression)
    assert lr.fit.__func__ is Estimator.fit
    assert lr.setParams(maxIter=5) is lr
    assert lr.getMaxIter() == 5 and lr.getRegParam() == 0.1
    assert lr.isSet(lr.maxIter) and not lr.isSet(lr.featuresCol)
    assert lr.getFeaturesCol() == "features"
    copied = lr.copy({lr.maxIter: 2})
    assert type(copied) is LogisticRegression and copied.uid == lr.uid
    assert copied.getMaxIter() == 2 and lr.getMaxIter() == 5
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(lr, protocol=protocol))
        assert type(restored) is LogisticRegression
        assert restored.uid == lr.uid and restored.getMaxIter() == 5
    assert type(copy.deepcopy(lr)) is LogisticRegression
    assert type(LogisticRegressionModel().copy()) is LogisticRegressionModel


def test_fit_dispatches_to_python_subclass_with_parameter_overrides():
    calls = []
    dataset = object()

    class RecordingLR(LogisticRegression):
        def _fit(self, frame):
            assert frame is dataset
            calls.append((type(self), self.uid, self.getMaxIter()))
            return self.getMaxIter()

    lr = RecordingLR(maxIter=5)
    assert lr.fit(dataset, {lr.maxIter: 2}) == 2
    assert calls == [(RecordingLR, lr.uid, 2)]
    assert lr.getMaxIter() == 5
    assert lr.fit(dataset, [{lr.maxIter: 1}, {lr.maxIter: 3}]) == [1, 3]
    assert type(lr.copy()) is RecordingLR


@pytest.mark.parametrize("kind", ["read", "write"])
@pytest.mark.parametrize("standard_import_first", [False, True])
def test_pickled_readers_and_writers_initialize_in_a_fresh_process(kind, standard_import_first):
    import base64
    import json

    lr = LogisticRegression(maxIter=5)
    value = getattr(lr, kind)()
    payloads = [
        base64.b64encode(pickle.dumps(value, protocol)).decode("ascii")
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
    ]
    code = r"""
import base64
import json
import pickle
import sys
if STANDARD_IMPORT_FIRST:
    from pyspark.ml.classification import LogisticRegression
assert "pyspark.ml.connect" not in sys.modules
for payload in json.loads(sys.argv[1]):
    restored = pickle.loads(base64.b64decode(payload))
    assert restored.session(None) is restored
    if KIND == "write":
        assert restored.overwrite() is restored
        assert restored.option("test", "value") is restored
        assert restored._instance.getMaxIter() == 5
    else:
        from pyspark.ml.classification import LogisticRegression
        assert restored._clazz is LogisticRegression
"""
    code = code.replace("STANDARD_IMPORT_FIRST", repr(standard_import_first)).replace(
        "KIND", repr(kind)
    )
    subprocess.run([sys.executable, "-c", code, json.dumps(payloads)], check=True)
