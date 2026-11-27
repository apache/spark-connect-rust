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
from pyspark.ml.classification import (
    LinearSVC,
    LinearSVCModel,
    LogisticRegression,
    LogisticRegressionModel,
)
from pyspark.ml.regression import (
    GeneralizedLinearRegression,
    GeneralizedLinearRegressionModel,
    LinearRegression,
    LinearRegressionModel,
)


@pytest.fixture(
    params=[LogisticRegression, LinearSVC, LinearRegression, GeneralizedLinearRegression],
    ids=lambda cls: cls.__name__,
)
def estimator_cls(request):
    return request.param


@pytest.mark.parametrize(
    "first", ["standard", "compatibility"],
)
@pytest.mark.parametrize("persistence_first", ["read", "write"])
def test_public_classes_are_identical_in_either_import_order(first, persistence_first, estimator_cls):
    code = """
import importlib
import pickle
from pyspark import _pyspark
importlib.import_module(FIRST)
public = importlib.import_module("pyspark.ml.classification")
# Exercise persistence before the explicit legacy import. That import used to be necessary
# to register the native reader/writer classes with their Python API (overwrite, option, ...).
estimator = public.LogisticRegression(maxIter=3)
getattr(estimator, PERSISTENCE_FIRST)()
writer = estimator.write()
assert writer.overwrite() is writer and writer.shouldOverwrite
assert writer.option("example", "value") is writer
assert public.LogisticRegression.read()._clazz is public.LogisticRegression
legacy = importlib.import_module("pyspark.ml.connect.classification")
names = (
    "LogisticRegression", "LogisticRegressionModel", "LogisticRegressionSummary",
    "LogisticRegressionTrainingSummary", "BinaryLogisticRegressionSummary",
    "BinaryLogisticRegressionTrainingSummary",
)
from pyspark.ml._upstream import classification as reference
for name in names:
    cls = getattr(public, name)
    assert cls.__doc__ == getattr(reference, name).__doc__
    assert cls is getattr(legacy, name)
    assert cls.__module__ == "pyspark.ml.classification"
    assert issubclass(cls, getattr(_pyspark, name))
    assert pickle.loads(pickle.dumps(cls)) is cls
assert not issubclass(public.NaiveBayes, _pyspark.NaiveBayes)
"""
    if estimator_cls is not LogisticRegression:
        code = code.replace('"BinaryLogisticRegressionSummary",', "").replace(
            '"BinaryLogisticRegressionTrainingSummary",', "",
        )
    code = code.replace("LogisticRegression", estimator_cls.__name__)
    module = estimator_cls.__module__
    code = code.replace("classification", module.rsplit(".", 1)[1])
    if estimator_cls.__module__ == "pyspark.ml.regression":
        code = code.replace("NaiveBayes", "AFTSurvivalRegression")
    first_module = module if first == "standard" else module.replace("pyspark.ml.", "pyspark.ml.connect.")
    code = code.replace("PERSISTENCE_FIRST", repr(persistence_first)).replace("FIRST", repr(first_module))
    subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ), check=True, capture_output=True, text=True,
    )


def test_params_copy_and_pickle_keep_the_standard_api(estimator_cls):
    estimator = estimator_cls(maxIter=3, regParam=0.1)
    assert isinstance(estimator, getattr(_pyspark, estimator_cls.__name__))
    assert estimator.fit.__func__ is Estimator.fit
    assert estimator.setParams(maxIter=5) is estimator
    assert estimator.getMaxIter() == 5 and estimator.getRegParam() == 0.1
    assert estimator.isSet(estimator.maxIter) and not estimator.isSet(estimator.featuresCol)
    assert estimator.getFeaturesCol() == "features"
    copied = estimator.copy({estimator.maxIter: 2})
    assert type(copied) is estimator_cls and copied.uid == estimator.uid
    assert copied.getMaxIter() == 2 and estimator.getMaxIter() == 5
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(estimator, protocol=protocol))
        assert type(restored) is estimator_cls
        assert restored.uid == estimator.uid and restored.getMaxIter() == 5
    assert type(copy.deepcopy(estimator)) is estimator_cls
    model_cls = getattr(sys.modules[estimator_cls.__module__], estimator_cls.__name__ + "Model")
    assert type(model_cls().copy()) is model_cls


def test_fit_dispatches_to_python_subclass_with_parameter_overrides(estimator_cls):
    calls = []
    dataset = object()

    class RecordingEstimator(estimator_cls):
        def _fit(self, frame):
            assert frame is dataset
            calls.append((type(self), self.uid, self.getMaxIter()))
            return self.getMaxIter()

    estimator = RecordingEstimator(maxIter=5)
    assert estimator.fit(dataset, {estimator.maxIter: 2}) == 2
    assert calls == [(RecordingEstimator, estimator.uid, 2)]
    assert estimator.getMaxIter() == 5
    assert estimator.fit(dataset, [{estimator.maxIter: 1}, {estimator.maxIter: 3}]) == [1, 3]
    assert type(estimator.copy()) is RecordingEstimator


@pytest.mark.parametrize("kind", ["read", "write"])
@pytest.mark.parametrize("standard_import_first", [False, True])
def test_pickled_readers_and_writers_initialize_in_a_fresh_process(
    kind, standard_import_first, estimator_cls,
):
    import base64
    import json

    estimator = estimator_cls(maxIter=5)
    value = getattr(estimator, kind)()
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
    code = code.replace("LogisticRegression", estimator_cls.__name__)
    code = code.replace("classification", estimator_cls.__module__.rsplit(".", 1)[1])
    code = code.replace("STANDARD_IMPORT_FIRST", repr(standard_import_first)).replace(
        "KIND", repr(kind)
    )
    subprocess.run([sys.executable, "-c", code, json.dumps(payloads)], check=True)


def test_standard_estimator_needs_no_optional_python_dependencies(estimator_cls):
    from _standard_ml_cases import without_optional_dependencies

    without_optional_dependencies("""
import copy
import pickle
from pyspark import _pyspark
from pyspark.ml.classification import LogisticRegression
estimator = LogisticRegression(maxIter=3)
assert isinstance(estimator, _pyspark.LogisticRegression)
estimator.setRegParam(0.1)
assert copy.deepcopy(estimator).getRegParam() == 0.1
assert pickle.loads(pickle.dumps(estimator)).getMaxIter() == 3
assert estimator.write().overwrite().shouldOverwrite
assert LogisticRegression.read()._clazz is LogisticRegression
""".replace("LogisticRegression", estimator_cls.__name__).replace(
        "classification", estimator_cls.__module__.rsplit(".", 1)[1],
    ))


def _assert_native_params_match(instance):
    # Read the native state directly; Python getters would hide a stale Rust snapshot.
    values, defaults = _pyspark._ml_params(instance)
    assert values == {p.name: v for p, v in instance._paramMap.items()}
    assert defaults == {p.name: v for p, v in instance._defaultParamMap.items()}
    assert _pyspark._ml_uid(instance) == instance.uid


def test_constructor_and_parameter_mutations_update_native_estimator_immediately(estimator_cls):
    estimator = estimator_cls(maxIter=3, regParam=0.1)
    _assert_native_params_match(estimator)
    assert estimator.setMaxIter(20) is estimator
    _assert_native_params_match(estimator)
    assert estimator.setParams(maxIter=21, regParam=0.2) is estimator
    _assert_native_params_match(estimator)
    assert estimator.set(estimator.maxIter, 22) is None
    _assert_native_params_match(estimator)
    assert estimator._setDefault(maxIter=23) is estimator
    _assert_native_params_match(estimator)
    assert estimator.clear(estimator.maxIter) is None
    _assert_native_params_match(estimator)
    assert estimator._resetUid("eager_classifier") is estimator
    _assert_native_params_match(estimator)


@pytest.mark.parametrize(
    "cls", [LogisticRegression, LogisticRegressionModel, LinearSVC, LinearSVCModel],
)
def test_model_and_estimator_threshold_setters_update_native_params(cls):
    instance = cls()
    _assert_native_params_match(instance)
    assert instance.setPredictionCol("predicted") is instance
    _assert_native_params_match(instance)
    assert instance.setThreshold(0.3) is instance
    _assert_native_params_match(instance)
    if cls in (LogisticRegression, LogisticRegressionModel):
        assert instance.setThresholds([0.7, 0.3]) is instance
    else:
        assert instance.setThreshold(-0.3) is instance
    _assert_native_params_match(instance)
    # Python Params accepts None; synchronization must retain its deferred validation.
    assert instance.setFeaturesCol(None) is instance
    _assert_native_params_match(instance)


def test_partial_mutation_keeps_native_state_and_original_conversion_error(estimator_cls):
    estimator = estimator_cls(maxIter=3, regParam=0.1)
    with pytest.raises(TypeError, match='Invalid param value given for param "regParam"'):
        estimator._set(maxIter=9, regParam=object())
    assert estimator.getMaxIter() == 9 and estimator.getRegParam() == 0.1
    _assert_native_params_match(estimator)
    with pytest.raises(TypeError, match='Invalid default param value given for param "regParam"'):
        estimator._setDefault(maxIter=8, regParam=object())
    _assert_native_params_match(estimator)


@pytest.mark.parametrize(
    "cls", [
        LogisticRegression, LogisticRegressionModel, LinearSVC, LinearSVCModel,
        LinearRegression, LinearRegressionModel,
        GeneralizedLinearRegression, GeneralizedLinearRegressionModel,
    ],
)
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


def test_default_only_copy_values_updates_native_target(estimator_cls):
    source = estimator_cls()._setDefault(maxIter=41)
    assert not source._paramMap
    model_cls = getattr(sys.modules[estimator_cls.__module__], estimator_cls.__name__ + "Model")
    for target in (estimator_cls(), model_cls()):
        assert source._copyValues(target) is target
        assert target.getMaxIter() == 41
        _assert_native_params_match(target)
    _assert_native_params_match(source.copy())


def test_subclass_setters_and_fit_overrides_observe_current_native_params(estimator_cls):
    class RecordingEstimator(estimator_cls):
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

    estimator = RecordingEstimator(maxIter=3)
    assert estimator.fit(object(), {estimator.maxIter: 7}) == 7
    assert estimator.fit(object(), [{estimator.maxIter: 2}, {estimator.maxIter: 4}]) == [2, 4]
    assert estimator.getMaxIter() == 3
    _assert_native_params_match(estimator)


def test_other_native_algorithms_keep_operation_time_synchronization():
    from pyspark.ml.connect.classification import FMClassifier

    svc = FMClassifier()
    previous = _pyspark._ml_params(svc)
    svc.setMaxIter(17)
    assert svc.getMaxIter() == 17
    assert _pyspark._ml_params(svc) == previous
    svc.write()
    _assert_native_params_match(svc)


def test_direct_dictionary_mutations_still_sync_before_native_operations(estimator_cls):
    estimator = estimator_cls(maxIter=3)
    estimator._paramMap[estimator.maxIter] = 17
    assert _pyspark._ml_params(estimator)[0]["maxIter"] == 3
    estimator.write()
    _assert_native_params_match(estimator)


def test_eager_native_defaults_do_not_leak_reference_cycles(estimator_cls):
    import gc
    import weakref

    estimator = estimator_cls()
    estimator.maxIter.typeConverter = lambda value: value
    estimator._setDefault(maxIter=estimator)
    reference = weakref.ref(estimator)
    del estimator
    gc.collect()
    assert reference() is None


def test_generalized_regression_setters_update_native_family_link_and_columns():
    estimator = GeneralizedLinearRegression(
        family="poisson", link="log", weightCol="weight", offsetCol="offset",
    )
    _assert_native_params_match(estimator)
    estimator.setFamily("tweedie").setVariancePower(1.5).setLinkPower(0.0)
    _assert_native_params_match(estimator)
    estimator.setParams(family="gamma", link="inverse", linkPredictionCol="eta")
    _assert_native_params_match(estimator)
    estimator.setParams(link=None)
    _assert_native_params_match(estimator)
    model = GeneralizedLinearRegressionModel().setLinkPredictionCol("eta")
    _assert_native_params_match(model)
