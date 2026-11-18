# SPDX-License-Identifier: Apache-2.0
"""Python subclasses of the native ML classes, as of PySpark's classes: their constructors, uids,
copies and dispatchers, and the errors of a call that does not bind. Official PySpark runs them
too, as the reference (`_native_classes.native_class`), and skips the tests of the native classes'
own members."""

import sys

import pytest
from _native_classes import native_class, native_module


def _cls(module, name):
    return native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")


def _unexpected(qualname, name):
    # CPython 3.10+ names the function's qualified name in the message of an unexpected keyword.
    prefix = f"{qualname}" if sys.version_info >= (3, 10) else qualname.rsplit(".", 1)[-1]
    return f"{prefix}() got an unexpected keyword argument '{name}'"


def test_a_subclass_constructs_of_the_keywords_of_its_super_init():
    # PySpark's `__init__` sets the params of its keywords: a subclass's own `__init__` passes them,
    # of any arguments of its own, as `object.__new__` leaves them to it.
    LR = _cls("classification", "LogisticRegression")

    class A(LR):
        pass

    class B(LR):
        def __init__(self, n):
            super().__init__(maxIter=n)

    class C(LR):
        def __init__(self):
            super().__init__(maxIter=7)

    class E(LR):
        def __init__(self, **kwargs):
            kwargs.setdefault("regParam", 0.5)
            super().__init__(**kwargs)
            self.tag = "t"

    assert A().getMaxIter() == 100 and A(maxIter=3).getMaxIter() == 3
    assert B(9).getMaxIter() == 9 and not B(9).isSet("regParam")
    assert C().getMaxIter() == 7
    assert E().getRegParam() == 0.5 and E(regParam=0.2).getRegParam() == 0.2
    assert E().tag == "t"
    # `keyword_only` records the keywords of the call, as PySpark's does.
    assert B(9)._input_kwargs == {"maxIter": 9}
    assert LR(maxIter=4, tol=0.5)._input_kwargs == {"maxIter": 4, "tol": 0.5}


def test_a_subclass_has_a_uid_of_its_own_name_and_the_defaults_of_it():
    # `Identifiable._randomUID` names the uid after the class of the object, and the defaults of
    # the uid (outputCol) and of the class's name (seed) follow it: of the `JavaModel.__init__` of
    # a model too.
    LR = _cls("classification", "LogisticRegression")
    DT = _cls("classification", "DecisionTreeClassifier")
    Binarizer = _cls("feature", "Binarizer")
    DTModel = _cls("classification", "DecisionTreeClassificationModel")
    IndexerModel = _cls("feature", "StringIndexerModel")

    class Sub(LR):
        pass

    class Tree(DT):
        pass

    class Bin(Binarizer):
        pass

    class Fitted(DTModel):
        pass

    class Indexer(IndexerModel):
        pass

    for obj, name in [
        (Sub(), "Sub"),
        (LR(), "LogisticRegression"),
        (Tree(), "Tree"),
        (Fitted(), "Fitted"),
        (DTModel(), "DecisionTreeClassificationModel"),
    ]:
        prefix, suffix = obj.uid.rsplit("_", 1)
        assert prefix == name and len(suffix) == 12
    assert Tree().getSeed() == hash("Tree") and Fitted().getSeed() == hash("Fitted")
    b = Bin(threshold=0.5)
    assert b.uid.startswith("Bin_") and b.getOutputCol() == b.uid + "__output"
    indexer = Indexer()
    assert indexer.uid.startswith("Indexer_") and indexer.getOutputCol() == indexer.uid + "__output"


def test_the_uid_is_assignable_as_pysparks():
    LR = _cls("classification", "LogisticRegression")
    x = LR(maxIter=3)
    x.uid = "LogisticRegression_abcdefabcdef"
    assert x.uid == "LogisticRegression_abcdefabcdef" and repr(x) == x.uid
    assert x.extractParamMap()[x.maxIter] == 3


def test_a_copy_keeps_the_class_and_the_attributes_of_the_object():
    # PySpark's `Params.copy` makes its copy of `copy.copy`: of the object's class (a subclass) and
    # of its attributes, shared.
    LR = _cls("classification", "LogisticRegression")

    class A(LR):
        pass

    a = A(maxIter=4)
    a.tag = [1]
    c = a.copy()
    assert type(c) is A and c.uid == a.uid and c.getMaxIter() == 4 and c.tag is a.tag
    d = a.copy({a.maxIter: 6})
    assert type(d) is A and d.getMaxIter() == 6 and a.getMaxIter() == 4
    plain = LR()
    plain.extra = 5
    assert plain.copy().extra == 5 and type(plain.copy()) is LR


def test_the_dispatchers_are_pysparks_own():
    # `fit`, `fitMultiple`, `transform`, `evaluate`, `save` and `load` are PySpark's functions,
    # which run `_fit`, `_transform`, `_evaluate`, `copy`, `write` and `read` of the object.
    from pyspark.ml import classification
    from pyspark.ml.base import Estimator, Transformer
    from pyspark.ml.evaluation import Evaluator
    from pyspark.ml.util import MLReadable, MLWritable

    LR = _cls("classification", "LogisticRegression")
    LRModel = _cls("classification", "LogisticRegressionModel")
    Evaluation = _cls("evaluation", "BinaryClassificationEvaluator")
    assert LR.fit is Estimator.fit and LR.fitMultiple is Estimator.fitMultiple
    assert LRModel.transform is Transformer.transform
    assert Evaluation.evaluate is Evaluator.evaluate
    assert LR.save is MLWritable.save and LR.load.__func__ is MLReadable.load.__func__
    assert LR.__doc__ == classification.LogisticRegression.__doc__


def test_a_call_that_does_not_bind_raises_pysparks_message():
    LR = _cls("classification", "LogisticRegression")
    LRModel = _cls("classification", "LogisticRegressionModel")
    with pytest.raises(TypeError, match="^Method __init__ forces keyword arguments.$"):
        LR(1)
    with pytest.raises(TypeError) as error:
        LR(foo=1)
    assert str(error.value) == _unexpected("LogisticRegression.__init__", "foo")
    with pytest.raises(TypeError) as error:
        LRModel(None, None)
    assert str(error.value).endswith("__init__() takes from 1 to 2 positional arguments but 3 were given")
    with pytest.raises(AssertionError):
        LRModel(java_model=object())


def test_a_copy_checks_its_extra_before_the_params_walk():
    # `_copyValues` checks `extra` before it reads the params, which raises the AssertionError of a
    # model of no server-side object of cached members (an ensemble's `trees`).
    GBT = _cls("classification", "GBTClassificationModel")
    with pytest.raises(TypeError, match=r"^Expecting a dict, but received an object of type <class 'list'>\.$"):
        GBT().copy([1])
    with pytest.raises(TypeError, match="^Expecting a valid instance of Param, but received: maxDepth$"):
        GBT().copy({"maxDepth": 3})
    with pytest.raises(AssertionError):
        GBT().copy()


class _PicklableSubclass(_cls("classification", "LogisticRegression")):
    """A subclass that `pickle` finds by its module and name."""


def test_the_modules_have_pysparks_readers_and_writers_of_the_python_format():
    # PySpark's own classes, which its modules have and their `__all__` does not name.
    import importlib

    if native_module() is None:
        pytest.skip("of this client's native classes")
    helpers = {
        "pipeline": [
            "PipelineReader",
            "PipelineWriter",
            "PipelineModelReader",
            "PipelineModelWriter",
            "PipelineSharedReadWrite",
        ],
        "tuning": [
            "CrossValidatorReader",
            "CrossValidatorWriter",
            "CrossValidatorModelReader",
            "CrossValidatorModelWriter",
            "TrainValidationSplitReader",
            "TrainValidationSplitWriter",
            "TrainValidationSplitModelReader",
            "TrainValidationSplitModelWriter",
        ],
        "classification": [
            "OneVsRestReader",
            "OneVsRestWriter",
            "OneVsRestModelReader",
            "OneVsRestModelWriter",
        ],
    }
    for module, names in helpers.items():
        ours = importlib.import_module(f"pyspark.ml.connect.{module}")
        pysparks = importlib.import_module(f"pyspark.ml.{module}")
        for name in names:
            assert getattr(ours, name) is getattr(pysparks, name), (module, name)
            assert name not in getattr(ours, "__all__", []), (module, name)


def test_copy_deepcopy_and_pickle_of_an_estimator_an_evaluator_and_a_pipeline():
    # PySpark's objects copy, copy deep and pickle of their attributes: `copy.copy` shares them
    # (and a Pipeline's stages), `copy.deepcopy` and `pickle` copy them.
    import copy
    import pickle

    Evaluator = _cls("evaluation", "BinaryClassificationEvaluator")
    Binarizer = _cls("feature", "Binarizer")
    Pipeline = _cls("pipeline", "Pipeline")
    lr = _PicklableSubclass(maxIter=4)
    lr.extra = [1]
    for copied in (copy.copy(lr), copy.deepcopy(lr), pickle.loads(pickle.dumps(lr))):
        assert type(copied) is _PicklableSubclass and copied.uid == lr.uid
        assert copied.getMaxIter() == 4
        assert copied.extra == [1] and copied.isSet("maxIter") and not copied.isSet("tol")
    assert copy.copy(lr).extra is lr.extra and copy.deepcopy(lr).extra is not lr.extra
    evaluator = Evaluator(metricName="areaUnderPR")
    for copied in (copy.copy(evaluator), copy.deepcopy(evaluator), pickle.loads(pickle.dumps(evaluator))):
        assert copied.getMetricName() == "areaUnderPR" and copied.uid == evaluator.uid
    stage = Binarizer(threshold=0.5, inputCol="x", outputCol="y")
    pipeline = Pipeline(stages=[stage])
    assert copy.copy(pipeline).getStages()[0] is stage
    deep = copy.deepcopy(pipeline).getStages()[0]
    assert deep is not stage and deep.uid == stage.uid and deep.getThreshold() == 0.5
    assert pickle.loads(pickle.dumps(pipeline)).getStages()[0].getThreshold() == 0.5


def test_copy_deepcopy_and_pickle_of_a_model_of_no_server_side_object():
    import copy
    import pickle

    LRModel = _cls("classification", "LogisticRegressionModel")
    model = LRModel()
    for copied in (copy.copy(model), copy.deepcopy(model), pickle.loads(pickle.dumps(model))):
        assert type(copied) is LRModel and copied.uid == model.uid



def test_the_params_take_numpy_values_as_pysparks_type_converters_do():
    # PySpark's converters: a NumPy string stays as given (it is a str), a NumPy int64 becomes an
    # int, and a float32 is no float of `toFloat`; the native object takes the values of the map.
    import numpy as np

    LR = _cls("classification", "LogisticRegression")
    lr = LR(family=np.str_("multinomial"))
    assert type(lr.getFamily()) is np.str_ and lr.getFamily() == "multinomial"
    assert type(lr.setMaxIter(np.int64(3)).getMaxIter()) is int
    message = 'Invalid param value given for param "regParam". Could not convert 0.5 to float'
    with pytest.raises(TypeError, match=f"^{message}$"):
        lr.setRegParam(np.float32(0.5))
    if native_module() is not None:
        lr._native_sync(
            {p.name: v for p, v in lr._paramMap.items()},
            {p.name: v for p, v in lr._defaultParamMap.items()},
        )
        assert lr._native_params()[0]["family"] == "multinomial"
    StringIndexer = _cls("feature", "StringIndexer")
    columns = StringIndexer(inputCols=[np.str_("a"), "b"]).getInputCols()
    assert [type(c) for c in columns] == [np.str_, str]


def test_the_private_params_state_is_pysparks():
    # PySpark's `_paramMap` and `_defaultParamMap` (by Param, in the order that they were set: the
    # defaults of a new object in its constructor's order), its `_set`, `_setDefault`,
    # `_copyValues` and `_input_kwargs`, of the object's attributes.
    LR = _cls("classification", "LogisticRegression")
    lr = LR(maxIter=5, regParam=0.1)
    assert [p.name for p in lr._paramMap] == ["maxIter", "regParam"]
    assert lr._input_kwargs == {"maxIter": 5, "regParam": 0.1}
    names = [p.name for p in lr._defaultParamMap]
    assert names[:4] == ["maxBlockSizeInMB", "threshold", "aggregationDepth", "standardization"]
    assert [p.name for p in lr.extractParamMap()][:4] == names[:4]
    assert lr._set(tol=0.5) is lr and lr.getTol() == 0.5 and lr.isSet("tol")
    lr._setDefault(maxIter=7)
    assert lr._defaultParamMap[lr.maxIter] == 7 and lr.getMaxIter() == 5
    other = LR()
    lr._copyValues(other)
    assert other.getMaxIter() == 5 and other.getTol() == 0.5
    with pytest.raises(TypeError, match="^Expecting a valid instance of Param, but received: x$"):
        lr._copyValues(LR(), {"x": 1})
    # A copy, a fit's model and a load are of the order of PySpark's `_copyValues` (of `params`).
    assert [p.name for p in lr.copy().extractParamMap()][:3] == [
        "aggregationDepth",
        "elasticNetParam",
        "family",
    ]


def test_an_assignment_of_the_uid_keeps_the_params_of_the_old_uid():
    # As PySpark's: an assignment of `uid` keeps the Params of the old uid, which do not belong to
    # the object, so the calls of a Param raise PySpark's ValueError, and those of a name read
    # the map; `_resetUid` gives them the new uid.
    LR = _cls("classification", "LogisticRegression")
    lr = LR(maxIter=7)
    old = lr.uid
    lr.uid = "LogisticRegression_abcdefabcdef"
    assert lr.maxIter.parent == old
    for call in (lr.getMaxIter, lambda: lr.isSet(lr.maxIter), lambda: lr.explainParam("maxIter")):
        with pytest.raises(ValueError, match="does not belong to"):
            call()
    assert lr.isSet("maxIter") is True
    assert [p.parent for p in lr.extractParamMap()][:1] == [old]
    # `_set` reads the Param of the name, of no check of its parent.
    assert lr.setMaxIter(3) is lr and lr.getOrDefault("maxIter") == 3
    assert lr.copy().uid == "LogisticRegression_abcdefabcdef"
    reset = LR(maxIter=7)._resetUid("LogisticRegression_abcdefabcdef")
    assert reset.uid == "LogisticRegression_abcdefabcdef" and reset.maxIter.parent == reset.uid
    assert reset.getMaxIter() == 7 and reset.setMaxIter(3).getMaxIter() == 3


def test_the_params_api_raises_pysparks_messages_of_a_call_that_does_not_bind():
    # PySpark's own functions of the Params API, of their qualified names.
    LR = _cls("classification", "LogisticRegression")
    lr = LR()
    cases = [
        (lambda: lr.getParam(foo=1), "Params.getParam() got an unexpected keyword argument 'foo'"),
        (lambda: lr.isSet(), "Params.isSet() missing 1 required positional argument: 'param'"),
        (lambda: lr.copy(foo=1), "JavaParams.copy() got an unexpected keyword argument 'foo'"),
        (lambda: lr.setMaxIter(), "LogisticRegression.setMaxIter() missing 1 required positional argument: 'value'"),
    ]
    if sys.version_info < (3, 10):
        return
    for call, message in cases:
        with pytest.raises(TypeError) as error:
            call()
        assert str(error.value) == message


def test_a_native_member_takes_the_params_of_the_maps_of_the_objects_that_its_object_holds():
    # The native members of an object read the native objects that it holds (an OneVsRest's
    # classifier, a Pipeline's stages, a validator's estimator, evaluator and grids, a
    # PipelineModel's stages), whose params are of their own maps: they are synced with it.
    if native_module() is None:
        pytest.skip("of this client's native classes")
    LR = _cls("classification", "LogisticRegression")
    OneVsRest = _cls("classification", "OneVsRest")
    Pipeline = _cls("pipeline", "Pipeline")
    PipelineModel = _cls("pipeline", "PipelineModel")
    CrossValidator = _cls("tuning", "CrossValidator")
    Evaluator = _cls("evaluation", "BinaryClassificationEvaluator")
    Binarizer = _cls("feature", "Binarizer")

    def native_set(obj):
        return obj._native_params()[0]

    lr = LR(maxIter=3)
    assert native_set(lr) == {}
    lr.write()
    assert native_set(lr) == {"maxIter": 3}
    classifier = LR(regParam=0.25)
    OneVsRest(classifier=classifier).write()
    assert native_set(classifier) == {"regParam": 0.25}
    stage = Binarizer(threshold=0.5)
    Pipeline(stages=[stage]).write()
    assert native_set(stage) == {"threshold": 0.5}
    held = Binarizer(inputCol="x")
    model = PipelineModel(stages=[held])
    held.setThreshold(0.75)
    model.write()
    assert native_set(held) == {"inputCol": "x", "threshold": 0.75}
    estimator, evaluator, grid = LR(tol=0.5), Evaluator(metricName="areaUnderPR"), LR(maxIter=9)
    ovr = OneVsRest(classifier=estimator)
    CrossValidator(
        estimator=ovr, evaluator=evaluator, estimatorParamMaps=[{ovr.classifier: grid}]
    ).write()
    assert native_set(estimator) == {"tol": 0.5} and native_set(grid) == {"maxIter": 9}
    assert native_set(evaluator) == {"metricName": "areaUnderPR"}


def test_the_native_members_of_a_public_class_have_pysparks_docstrings_and_signatures():
    # PySpark's docstring, signature, module and qualified name of each member: of the class of
    # PySpark's that defines it (`JavaMLWritable.write`, of the wrapper of `try_remote_write`), not
    # of the class of the object.
    import inspect

    import pyspark.ml.classification as reference

    LR = _cls("classification", "LogisticRegression")
    LRModel = _cls("classification", "LogisticRegressionModel")
    for public, pysparks, name, qualname in [
        (LR, reference.LogisticRegression, "write", "JavaMLWritable.write"),
        (LR, reference.LogisticRegression, "_fit", "JavaEstimator._fit"),
        (LR, reference.LogisticRegression, "clear", "JavaParams.clear"),
        (LRModel, reference.LogisticRegressionModel, "predict", "JavaPredictionModel.predict"),
        (LRModel, reference.LogisticRegressionModel, "predictRaw", "_JavaClassificationModel.predictRaw"),
        (LRModel, reference.LogisticRegressionModel, "evaluate", "LogisticRegressionModel.evaluate"),
        (LRModel, reference.LogisticRegressionModel, "_transform", "JavaTransformer._transform"),
    ]:
        member = getattr(public, name)
        assert member.__doc__ == getattr(pysparks, name).__doc__, name
        assert inspect.signature(member) == inspect.signature(getattr(pysparks, name)), name
        assert member.__qualname__ == qualname
        assert member.__module__ == getattr(pysparks, name).__module__, name


def test_the_native_members_bind_a_call_as_pysparks_functions():
    # Python binds a call of a native member as it binds PySpark's function, of its parameters (of
    # the wrapper of a decorator: `try_remote_intercept`'s `JavaParams.clear` takes any positional
    # arguments, and does nothing over Spark Connect), and raises its messages, of its qualified
    # name: before the native member runs, which a model of no server-side object fails.
    from pyspark.ml.linalg import Vectors

    lr = _cls("classification", "LogisticRegression")()
    model = _cls("classification", "LogisticRegressionModel")()
    value = Vectors.dense([1.0])
    assert lr.clear() is None and lr.clear(lr.maxIter, 1) is None
    cases = [
        (lambda: model.predict(value, 1), "JavaPredictionModel.predict() takes 2 positional arguments but 3 were given"),
        (lambda: model.predict(), "JavaPredictionModel.predict() missing 1 required positional argument: 'value'"),
        (lambda: model.predictRaw(value=value, x=1), _unexpected("_JavaClassificationModel.predictRaw", "x")),
        (lambda: model.evaluate(1, 2), "LogisticRegressionModel.evaluate() takes 2 positional arguments but 3 were given"),
        (lambda: model._transform(), "JavaTransformer._transform() missing 1 required positional argument: 'dataset'"),
        (lambda: lr.write(1), "JavaMLWritable.write() takes 1 positional argument but 2 were given"),
        (lambda: lr._fit(1, 2), "JavaEstimator._fit() takes 2 positional arguments but 3 were given"),
        (lambda: lr.clear(param=lr.maxIter), _unexpected("JavaParams.clear", "param")),
    ]
    with pytest.raises(AssertionError):
        model.predict(value)
    # The methods of a summary, of no params to sync, bind as PySpark's too: a relation method of
    # `try_remote_attribute_relation` of any arguments, which then asserts an active session.
    summary = _cls("classification", "LogisticRegressionSummary")()
    with pytest.raises(AssertionError):
        _cls("regression", "GeneralizedLinearRegressionSummary")().residuals(1, 2, x=3)
    cases += [
        (lambda: summary.fMeasureByLabel(1, 2), "_ClassificationSummary.fMeasureByLabel() takes from 1 to 2 positional arguments but 3 were given"),
        (lambda: summary.weightedFMeasure(y=1), _unexpected("_ClassificationSummary.weightedFMeasure", "y")),
    ]
    if sys.version_info < (3, 10):
        return
    for call, message in cases:
        with pytest.raises(TypeError) as error:
            call()
        assert str(error.value) == message


def test_the_class_and_static_methods_are_pysparks_functions():
    # A class method (`read`, of the wrapper of `try_remote_read`, `from_labels`) and a static
    # method (`loadDefaultStopWords`) of a public class are of PySpark's kind, docstring,
    # signature, module and qualified name, and Python binds a call of one as it binds PySpark's
    # function (a class method of its `cls` too). `read` gives the reader of the class of the
    # call, a subclass's. PrefixSpan's `clear` is `JavaParams.clear`, which takes any positional
    # arguments and does nothing over Spark Connect.
    import inspect

    import pyspark.ml.feature as feature_reference
    import pyspark.ml.pipeline as pipeline_reference

    LR = _cls("classification", "LogisticRegression")
    Pipeline = _cls("pipeline", "Pipeline")
    Indexer = _cls("feature", "StringIndexerModel")
    Vectorizer = _cls("feature", "CountVectorizerModel")
    Remover = _cls("feature", "StopWordsRemover")
    import pyspark.ml.classification as classification_reference

    for public, pysparks, name, qualname in [
        (LR, classification_reference.LogisticRegression, "read", "JavaMLReadable.read"),
        (Pipeline, pipeline_reference.Pipeline, "read", "Pipeline.read"),
        (Indexer, feature_reference.StringIndexerModel, "from_labels", "StringIndexerModel.from_labels"),
        (
            Indexer,
            feature_reference.StringIndexerModel,
            "from_arrays_of_labels",
            "StringIndexerModel.from_arrays_of_labels",
        ),
        (Vectorizer, feature_reference.CountVectorizerModel, "from_vocabulary", "CountVectorizerModel.from_vocabulary"),
        (Remover, feature_reference.StopWordsRemover, "loadDefaultStopWords", "StopWordsRemover.loadDefaultStopWords"),
    ]:
        ours, theirs = inspect.getattr_static(public, name), inspect.getattr_static(pysparks, name)
        assert type(ours) is type(theirs), name
        assert ours.__func__.__qualname__ == qualname, name
        assert ours.__func__.__module__ == theirs.__func__.__module__, name
        assert ours.__func__.__doc__ == theirs.__func__.__doc__, name
        assert inspect.signature(getattr(public, name)) == inspect.signature(getattr(pysparks, name)), name
    Sub = type("Sub", (LR,), {})
    assert type(Sub.read()).__name__ == type(LR.read()).__name__ == "RemoteMLReader"
    prefix_span = _cls("fpm", "PrefixSpan")()
    assert prefix_span.clear() is None and prefix_span.clear(1, 2) is None
    assert prefix_span.clear(prefix_span.minSupport) is None
    cases = [
        (lambda: LR.read(1), "JavaMLReadable.read() takes 1 positional argument but 2 were given"),
        (lambda: Pipeline.read(x=1), _unexpected("Pipeline.read", "x")),
        (
            lambda: Indexer.from_labels(1, 2, 3, 4, 5),
            "StringIndexerModel.from_labels() takes from 3 to 5 positional arguments but 6 were given",
        ),
        (lambda: Vectorizer.from_vocabulary(vocab=1), _unexpected("CountVectorizerModel.from_vocabulary", "vocab")),
        (
            lambda: Remover.loadDefaultStopWords(),
            "StopWordsRemover.loadDefaultStopWords() missing 1 required positional argument: 'language'",
        ),
    ]
    if sys.version_info < (3, 10):
        return
    for call, message in cases:
        with pytest.raises(TypeError) as error:
            call()
        assert str(error.value) == message


def test_the_members_of_a_summary_have_pysparks_docstrings_and_signatures():
    # As the native members of a class of params: a member of a summary is a property of a
    # property of PySpark's (`r2`, `predictions`), and a method of a method (`fMeasureByLabel`,
    # `residuals`), of PySpark's docstring and signature.
    import importlib
    import inspect

    if native_module() is None:
        pytest.skip("of this client's native classes")
    for module, names in _SUMMARY_MODULES.items():
        reference = importlib.import_module(f"pyspark.ml.{module}")
        for name in names:
            public, pysparks = _cls(module, name), getattr(reference, name)
            for member in [m for m in dir(pysparks) if not m.startswith("_")]:
                theirs = inspect.getattr_static(pysparks, member)
                if isinstance(theirs, property):
                    ours = inspect.getattr_static(public, member)
                    assert isinstance(ours, property), (name, member)
                    assert ours.__doc__ == theirs.__doc__, (name, member)
                else:
                    ours = getattr(public, member)
                    assert ours.__doc__ == theirs.__doc__, (name, member)
                    assert inspect.signature(ours) == inspect.signature(theirs), (name, member)
                    # Of the class of PySpark's that defines the method, and its module.
                    assert ours.__qualname__ == theirs.__qualname__, (name, member)
                    assert ours.__module__ == theirs.__module__, (name, member)


def test_the_summaries_of_the_models_are_the_native_ones():
    # PySpark's `summary` of a model reads the `_summary` that its fit sets of its server-side
    # object (`super().summary` of LinearSVCModel's own): the native model fetches its own.
    if native_module() is None:
        pytest.skip("of this client's native classes")
    import pyspark.ml.classification as reference

    for name in ["LinearSVCModel", "MultilayerPerceptronClassificationModel", "FMClassificationModel"]:
        public = _cls("classification", name)
        assert public.summary is not getattr(reference, name).summary, name


_SUMMARY_MODULES = {
    "classification": [
        "BinaryLogisticRegressionSummary",
        "BinaryLogisticRegressionTrainingSummary",
        "BinaryRandomForestClassificationSummary",
        "BinaryRandomForestClassificationTrainingSummary",
        "FMClassificationSummary",
        "FMClassificationTrainingSummary",
        "LinearSVCSummary",
        "LinearSVCTrainingSummary",
        "LogisticRegressionSummary",
        "LogisticRegressionTrainingSummary",
        "MultilayerPerceptronClassificationSummary",
        "MultilayerPerceptronClassificationTrainingSummary",
        "RandomForestClassificationSummary",
        "RandomForestClassificationTrainingSummary",
    ],
    "regression": [
        "GeneralizedLinearRegressionSummary",
        "GeneralizedLinearRegressionTrainingSummary",
        "LinearRegressionSummary",
        "LinearRegressionTrainingSummary",
    ],
    "clustering": ["BisectingKMeansSummary", "GaussianMixtureSummary", "KMeansSummary"],
}


def test_the_summaries_are_subclasses_of_the_summaries_of_pysparks_bases_of_their_module():
    # Of PySpark's metaclass (`type`), and a subclass of the public class of each summary of the
    # module that PySpark's class is a subclass of, as a class of its MRO, not a virtual one: a
    # training summary of its summary, and the BinaryLogisticRegressionTrainingSummary of both the
    # BinaryLogisticRegressionSummary and the LogisticRegressionTrainingSummary.
    import importlib

    for module, names in _SUMMARY_MODULES.items():
        reference = importlib.import_module(f"pyspark.ml.{module}")
        for name in names:
            public, pysparks = _cls(module, name), getattr(reference, name)
            assert type(public) is type(pysparks) is type, name
            for base in pysparks.__mro__[1:]:
                if base.__module__ == reference.__name__ and base.__name__ in names:
                    assert _cls(module, base.__name__) in public.__mro__, (name, base.__name__)


class _PicklableSummary(_cls("clustering", "KMeansSummary")):
    """A subclass of a summary that `pickle` finds by its module and name, of the `java_obj` of its
    `super().__init__`."""

    def __init__(self):
        super().__init__("abc.summary")
        self.extra = 1


def test_a_summary_keeps_its_java_obj_and_copies_and_pickles_as_a_plain_object():
    # PySpark's `JavaWrapper.__init__` keeps the `java_obj` in the summary's attributes (of the
    # `super().__init__` of a subclass), which its members read. A summary of no server-side
    # object copies, copies deep and pickles, at every protocol, as a plain object: of its class
    # and of its attributes, shared by a shallow copy.
    import copy
    import pickle

    Summary = _cls("regression", "LinearRegressionSummary")
    assert vars(Summary(java_obj="x")) == {"_java_obj": "x"}
    for given in (None, "abc", "abc.summary"):
        summary = Summary(given)
        assert vars(summary) == {"_java_obj": given}
        summary.tag = [1]
        shallow, deep = copy.copy(summary), copy.deepcopy(summary)
        pickled = [
            pickle.loads(pickle.dumps(summary, protocol))
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
        ]
        for copied in [shallow, deep, *pickled]:
            assert type(copied) is Summary, given
            assert vars(copied) == {"_java_obj": given, "tag": [1]}, given
        assert shallow.tag is summary.tag and deep.tag is not summary.tag
    subclass = _PicklableSummary()
    assert vars(subclass) == {"_java_obj": "abc.summary", "extra": 1}
    for copied in (copy.deepcopy(subclass), pickle.loads(pickle.dumps(subclass))):
        assert type(copied) is _PicklableSummary and vars(copied) == vars(subclass)


def test_a_summary_constructs_as_pysparks_java_wrapper():
    # PySpark's `JavaWrapper.__init__(java_obj=None)`, and its errors of a call that does not bind;
    # of no active session, a member raises PySpark's AssertionError of `try_remote_call`.
    import re

    Summary = _cls("classification", "LinearSVCTrainingSummary")
    assert type(Summary()) is Summary and type(Summary(java_obj=None)) is Summary
    message = "JavaWrapper.__init__() takes from 1 to 2 positional arguments but 3 were given"
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        Summary(None, 1)
    with pytest.raises(TypeError, match=re.escape(_unexpected("JavaWrapper.__init__", "x"))):
        Summary(x=1)
    with pytest.raises(AssertionError):
        Summary().accuracy
    with pytest.raises(AttributeError, match="^'LinearSVCTrainingSummary' object has no attribute 'foo'$"):
        Summary().foo
