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


def test_standard_logistic_needs_no_optional_python_dependencies():
    from _standard_logistic_cases import without_optional_dependencies

    without_optional_dependencies("""
import copy
import pickle
from pyspark import _pyspark
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=3)
assert isinstance(lr, _pyspark.LogisticRegression)
lr.setRegParam(0.1)
assert copy.deepcopy(lr).getRegParam() == 0.1
assert pickle.loads(pickle.dumps(lr)).getMaxIter() == 3
assert lr.write().overwrite().shouldOverwrite
assert LogisticRegression.read()._clazz is LogisticRegression
""")


def _assert_native_params_match(instance):
    # Read the native state directly; Python getters would hide a stale Rust snapshot.
    values, defaults = _pyspark._ml_params(instance)
    assert values == {p.name: v for p, v in instance._paramMap.items()}
    assert defaults == {p.name: v for p, v in instance._defaultParamMap.items()}
    assert _pyspark._ml_uid(instance) == instance.uid


def test_constructor_and_parameter_mutations_update_native_estimator_immediately():
    lr = LogisticRegression(maxIter=3, regParam=0.1)
    _assert_native_params_match(lr)
    assert lr.setMaxIter(20) is lr
    _assert_native_params_match(lr)
    assert lr.setParams(maxIter=21, regParam=0.2) is lr
    _assert_native_params_match(lr)
    assert lr.set(lr.maxIter, 22) is None
    _assert_native_params_match(lr)
    assert lr._setDefault(maxIter=23) is lr
    _assert_native_params_match(lr)
    assert lr.clear(lr.maxIter) is None
    _assert_native_params_match(lr)
    assert lr._resetUid("eager_lr") is lr
    _assert_native_params_match(lr)


@pytest.mark.parametrize("cls", [LogisticRegression, LogisticRegressionModel])
def test_model_and_estimator_threshold_setters_update_native_params(cls):
    instance = cls()
    _assert_native_params_match(instance)
    assert instance.setPredictionCol("predicted") is instance
    _assert_native_params_match(instance)
    assert instance.setThreshold(0.3) is instance
    _assert_native_params_match(instance)
    assert instance.setThresholds([0.7, 0.3]) is instance
    _assert_native_params_match(instance)
    # Python Params accepts None; synchronization must retain its deferred validation.
    assert instance.setFeaturesCol(None) is instance
    _assert_native_params_match(instance)


def test_partial_mutation_keeps_native_state_and_original_conversion_error():
    lr = LogisticRegression(maxIter=3, regParam=0.1)
    with pytest.raises(TypeError, match='Invalid param value given for param "regParam"'):
        lr._set(maxIter=9, regParam=object())
    assert lr.getMaxIter() == 9 and lr.getRegParam() == 0.1
    _assert_native_params_match(lr)
    with pytest.raises(TypeError, match='Invalid default param value given for param "regParam"'):
        lr._setDefault(maxIter=8, regParam=object())
    _assert_native_params_match(lr)


@pytest.mark.parametrize("cls", [LogisticRegression, LogisticRegressionModel])
def test_copies_and_restored_state_have_current_native_params(cls):
    instance = cls().setPredictionCol("predicted")
    instance._setDefault(maxIter=8)
    overridden = instance.copy({instance.predictionCol: "copied_prediction"})
    assert overridden.getPredictionCol() == "copied_prediction"
    assert instance.getPredictionCol() == "predicted"
    _assert_native_params_match(overridden)
    for copied in (copy.copy(instance), copy.deepcopy(instance)):
        assert copied is not instance and copied.uid == instance.uid
        _assert_native_params_match(copied)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        _assert_native_params_match(pickle.loads(pickle.dumps(instance, protocol=protocol)))
    _assert_native_params_match(instance)


def test_default_only_copy_values_updates_native_target():
    source = LogisticRegression()._setDefault(maxIter=41)
    assert not source._paramMap
    for target in (LogisticRegression(), LogisticRegressionModel()):
        assert source._copyValues(target) is target
        assert target.getMaxIter() == 41
        _assert_native_params_match(target)
    _assert_native_params_match(source.copy())


def test_subclass_setters_and_fit_overrides_observe_current_native_params():
    class RecordingLR(LogisticRegression):
        def _set(self, **kwargs):
            result = super()._set(**kwargs)
            # During _copyValues, defaults are still being copied one by one.
            assert _pyspark._ml_params(self)[0] == {
                p.name: v for p, v in self._paramMap.items()
            }
            return result

        def _fit(self, dataset):
            _assert_native_params_match(self)
            return self.getMaxIter()

    lr = RecordingLR(maxIter=3)
    assert lr.fit(object(), {lr.maxIter: 7}) == 7
    assert lr.fit(object(), [{lr.maxIter: 2}, {lr.maxIter: 4}]) == [2, 4]
    assert lr.getMaxIter() == 3
    _assert_native_params_match(lr)


def test_other_native_algorithms_keep_operation_time_synchronization():
    from pyspark.ml.connect.classification import LinearSVC

    svc = LinearSVC()
    previous = _pyspark._ml_params(svc)
    svc.setMaxIter(17)
    assert svc.getMaxIter() == 17
    assert _pyspark._ml_params(svc) == previous
    svc.write()
    _assert_native_params_match(svc)


def test_direct_dictionary_mutations_still_sync_before_native_operations():
    lr = LogisticRegression(maxIter=3)
    lr._paramMap[lr.maxIter] = 17
    assert _pyspark._ml_params(lr)[0]["maxIter"] == 3
    lr.write()
    _assert_native_params_match(lr)


def test_eager_native_defaults_do_not_leak_reference_cycles():
    import gc
    import weakref

    lr = LogisticRegression()
    lr.maxIter.typeConverter = lambda value: value
    lr._setDefault(maxIter=lr)
    reference = weakref.ref(lr)
    del lr
    gc.collect()
    assert reference() is None
