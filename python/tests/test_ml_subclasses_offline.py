# SPDX-License-Identifier: Apache-2.0
"""Python subclasses of the native ML classes, as of PySpark's classes: their constructors, uids,
copies and dispatchers, and the errors of a call that does not bind. Official PySpark runs them
too, as the reference (`_native_classes.native_class`)."""

import sys

import pytest
from _native_classes import native_class


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
    # the uid (outputCol) and of the class's name (seed) follow it.
    LR = _cls("classification", "LogisticRegression")
    DT = _cls("classification", "DecisionTreeClassifier")
    Binarizer = _cls("feature", "Binarizer")

    class Sub(LR):
        pass

    class Tree(DT):
        pass

    class Bin(Binarizer):
        pass

    for obj, name in [(Sub(), "Sub"), (LR(), "LogisticRegression"), (Tree(), "Tree")]:
        prefix, suffix = obj.uid.rsplit("_", 1)
        assert prefix == name and len(suffix) == 12
    assert Tree().getSeed() == hash("Tree")
    b = Bin(threshold=0.5)
    assert b.uid.startswith("Bin_") and b.getOutputCol() == b.uid + "__output"


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
