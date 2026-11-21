# SPDX-License-Identifier: Apache-2.0
"""Python subclasses of the native ML classes against a live Connect server: PySpark's `fit`,
`transform` and `evaluate` run the subclass's `_fit`, `_transform` and `_evaluate`, also of a param
map, of a list of maps, of `fitMultiple`, and in the copies that a Pipeline and a CrossValidator
fit. Official PySpark runs them too, as the reference (`_native_classes.native_class`)."""

import os

import pytest
from _native_classes import native_class


def _cls(module, name):
    return native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live ML subclass tests")
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def frame(session):
    from pyspark.ml.linalg import Vectors

    rows = [(Vectors.dense([float(i), float(i % 3)]), float(i % 2), float(i)) for i in range(12)]
    return session.createDataFrame(rows, ["features", "label", "x"]).coalesce(1)


def _counting(base, primitive):
    """A subclass of `base` whose `primitive` (`_fit`, ...) counts its calls, of every copy."""
    calls = []

    def hook(self, dataset):
        calls.append(self.uid)
        return getattr(super(cls, self), primitive)(dataset)

    cls = type("Counting" + base.__name__, (base,), {primitive: hook})
    return cls, calls


def test_fit_runs_the_fit_of_a_subclass(frame):
    LR = _cls("classification", "LogisticRegression")
    Counting, calls = _counting(LR, "_fit")
    estimator = Counting(maxIter=2)
    model = estimator.fit(frame)
    assert calls == [estimator.uid] and type(model).__name__ == "LogisticRegressionModel"
    assert estimator.fit(frame, {estimator.maxIter: 3}).getMaxIter() == 3
    models = estimator.fit(frame, [{}, {estimator.maxIter: 1}])
    assert [m.getMaxIter() for m in models] == [2, 1]
    fitted = sorted(i for i, _ in estimator.fitMultiple(frame, [{}, {}]))
    assert fitted == [0, 1]
    # A fit of a map or of a list fits copies, of the subclass and its uid.
    assert calls == [estimator.uid] * 6


def test_transform_and_evaluate_run_the_primitives_of_a_subclass(frame):
    Binarizer = _cls("feature", "Binarizer")
    Evaluator = _cls("evaluation", "RegressionEvaluator")
    Transforming, transforms = _counting(Binarizer, "_transform")
    Evaluating, evaluations = _counting(Evaluator, "_evaluate")
    binarizer = Transforming(threshold=4.5, inputCol="x", outputCol="b")
    assert binarizer.transform(frame).columns == ["features", "label", "x", "b"]
    assert binarizer.transform(frame, {binarizer.outputCol: "c"}).columns[-1] == "c"
    assert transforms == [binarizer.uid] * 2
    evaluator = Evaluating(predictionCol="x", labelCol="label")
    rmse = evaluator.evaluate(frame)
    assert rmse > 0 and evaluator.evaluate(frame, {evaluator.metricName: "mae"}) > 0
    assert evaluations == [evaluator.uid] * 2


def test_pysparks_readers_and_writers_save_and_load_native_objects(frame, tmp_path):
    # PySpark's own writer and reader of a Pipeline, over the param maps of the native objects.
    import pyspark.ml.connect.pipeline as pipeline

    Binarizer = _cls("feature", "Binarizer")
    stage = Binarizer(threshold=4.5, inputCol="x", outputCol="b")
    path = str(tmp_path) + "/pipeline"
    pipeline.PipelineWriter(pipeline.Pipeline(stages=[stage])).save(path)
    loaded = pipeline.PipelineReader(pipeline.Pipeline).load(path)
    assert [s.getThreshold() for s in loaded.getStages()] == [4.5]
    assert loaded.getStages()[0].uid == stage.uid


def test_a_set_param_of_a_subclass_is_sent_and_refused_by_the_server(frame, tmp_path):
    # PySpark's `serialize_ml_params` sends every param of `_paramMap` of a fit, a transform, an
    # evaluate and a save: the server refuses a param of a Python subclass that is set ("Param
    # myParam does not exist."), and a param of only a default is not sent.
    from pyspark.errors.exceptions.connect import SparkConnectGrpcException
    from pyspark.ml.param import Param, Params

    def extra():
        return Param(Params._dummy(), "myParam", "a param of the subclass")

    class MyLR(_cls("classification", "LogisticRegression")):
        myParam = extra()

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._setDefault(myParam=1)

    class MyTokenizer(_cls("feature", "Tokenizer")):
        myParam = extra()

    class MyEvaluator(_cls("evaluation", "RegressionEvaluator")):
        myParam = extra()

    estimator = MyLR(maxIter=2)
    assert type(estimator.fit(frame)).__name__ == "LogisticRegressionModel"
    estimator._set(myParam=5)
    path = str(tmp_path) + "/estimator"
    tokenizer = MyTokenizer(inputCol="x", outputCol="w")._set(myParam="v")
    evaluator = MyEvaluator(predictionCol="x")._set(myParam=0.5)
    for call in [
        lambda: estimator.fit(frame),
        lambda: estimator.write().save(path),
        lambda: tokenizer.transform(frame).collect(),
        lambda: evaluator.evaluate(frame),
    ]:
        with pytest.raises(SparkConnectGrpcException, match="Param myParam does not exist."):
            call()


def test_the_metadata_of_a_subclass_of_a_meta_algorithm_is_of_its_class(frame, tmp_path):
    # PySpark's `DefaultParamsWriter` writes the class of the object (`instance.__module__ + "." +
    # instance.__class__.__name__`), so of a Python subclass. Its readers of a OneVsRest and a
    # Pipeline load the file as of their own class; those of the validators read the grid of a
    # class of no `pyspark.ml.` as of the JVM's writer, of a JSON string, which fails; and a stage of
    # a pipeline loads of its class, which a class of a function is not an attribute of its module of.
    import glob
    import json

    from pyspark.ml.tuning import ParamGridBuilder

    OneVsRest = _cls("classification", "OneVsRest")
    Pipeline = _cls("pipeline", "Pipeline")
    CrossValidator = _cls("tuning", "CrossValidator")
    TrainValidationSplit = _cls("tuning", "TrainValidationSplit")

    class MyOVR(OneVsRest):
        pass

    class MyPipeline(Pipeline):
        pass

    class MyCV(CrossValidator):
        pass

    class MyTVS(TrainValidationSplit):
        pass

    lr = _cls("classification", "LogisticRegression")(maxIter=2)
    shared = {
        "estimator": lr,
        "estimatorParamMaps": ParamGridBuilder().addGrid(lr.maxIter, [1, 2]).build(),
        "evaluator": _cls("evaluation", "BinaryClassificationEvaluator")(),
    }
    base = str(tmp_path)
    for obj, cls in [
        (MyOVR(classifier=lr), OneVsRest),
        (MyPipeline(stages=[lr]), Pipeline),
        (MyCV(**shared), CrossValidator),
        (MyTVS(**shared), TrainValidationSplit),
    ]:
        name = type(obj).__name__
        path = f"{base}/{name}"
        obj.write().save(path)
        (part,) = glob.glob(os.path.join(path, "metadata", "part-*"))
        with open(part) as file:
            assert json.load(file)["class"] == f"{type(obj).__module__}.{name}"
        if cls in (CrossValidator, TrainValidationSplit):
            with pytest.raises(TypeError, match="^the JSON object must be str, bytes or bytearray"):
                cls.load(path)
        else:
            assert type(cls.load(path)) is cls
        Pipeline(stages=[obj]).write().save(f"{path}-in-pipeline")
        with pytest.raises(AttributeError, match=f"has no attribute '{name}'"):
            Pipeline.load(f"{path}-in-pipeline")


def test_the_metadata_of_a_subclass_has_the_params_of_the_subclass(frame, tmp_path):
    # PySpark's writers write the params of `extractParamMap()` (of a validator and a OneVsRest,
    # but its estimator, evaluator, grid and classifier, which they save of their own) and the
    # defaults of `_defaultParamMap`, of the Params of a Python subclass too; a Pipeline's params
    # are its `stageUids` and `language`, of its defaults.
    import glob
    import json

    from pyspark.ml.param import Param, Params, TypeConverters
    from pyspark.ml.tuning import ParamGridBuilder

    def subclass(base):
        def __init__(self, **kwargs):
            base.__init__(self, **kwargs)
            self._setDefault(myParam=3)

        param = Param(Params._dummy(), "myParam", "a param of the subclass", TypeConverters.toInt)
        return type("My" + base.__name__, (base,), {"myParam": param, "__init__": __init__})

    lr = _cls("classification", "LogisticRegression")(maxIter=2)
    shared = {
        "estimator": lr,
        "estimatorParamMaps": ParamGridBuilder().addGrid(lr.maxIter, [1]).build(),
        "evaluator": _cls("evaluation", "BinaryClassificationEvaluator")(),
    }
    base = str(tmp_path)
    for obj, set_in_params in [
        (subclass(_cls("tuning", "CrossValidator"))(**shared), True),
        (subclass(_cls("tuning", "TrainValidationSplit"))(**shared), True),
        (subclass(_cls("classification", "OneVsRest"))(classifier=lr), True),
        (subclass(_cls("pipeline", "Pipeline"))(stages=[lr]), False),
    ]:
        obj._set(myParam=7)
        path = f"{base}/{type(obj).__name__}"
        obj.write().save(path)
        (part,) = glob.glob(os.path.join(path, "metadata", "part-*"))
        with open(part) as file:
            metadata = json.load(file)
        assert metadata["defaultParamMap"]["myParam"] == 3, type(obj)
        assert metadata["paramMap"].get("myParam") == (7 if set_in_params else None), type(obj)


def test_the_save_of_a_pipeline_of_a_one_shot_iterator_writes_no_stage(frame, tmp_path):
    # PySpark's writers iterate the stages to check them (`validateStages`), then for their uids
    # and then to save them: a generator of stages is consumed by the check, so the metadata has
    # no `stageUids` and no stage is saved, and a fit after it fits none, as of PySpark's.
    import glob
    import json

    Pipeline = _cls("pipeline", "Pipeline")
    PipelineModel = _cls("pipeline", "PipelineModel")

    def stages():
        return [_cls("feature", "Binarizer")(threshold=0.5, inputCol="x", outputCol="b")]

    base = str(tmp_path)
    pipeline = Pipeline(stages=(stage for stage in stages()))
    model = PipelineModel(stages=(stage for stage in Pipeline(stages=stages()).fit(frame).stages))
    for obj, name in [(pipeline, "pipeline"), (model, "model")]:
        path = f"{base}/{name}"
        obj.write().save(path)
        (part,) = glob.glob(os.path.join(path, "metadata", "part-*"))
        with open(part) as file:
            assert json.load(file)["paramMap"]["stageUids"] == [], name
        assert not os.path.exists(os.path.join(path, "stages")), name
    assert Pipeline.load(f"{base}/pipeline").getStages() == []
    assert pipeline.fit(frame).stages == []


def test_the_copies_of_a_pipeline_and_a_validator_fit_the_subclass(frame):
    LR = _cls("classification", "LogisticRegression")
    Pipeline = _cls("pipeline", "Pipeline")
    CrossValidator = _cls("tuning", "CrossValidator")
    Evaluator = _cls("evaluation", "BinaryClassificationEvaluator")
    from pyspark.ml.tuning import ParamGridBuilder

    Counting, calls = _counting(LR, "_fit")
    estimator = Counting(maxIter=2)
    Pipeline(stages=[estimator]).fit(frame)
    assert calls == [estimator.uid]
    grid = ParamGridBuilder().addGrid(estimator.maxIter, [1, 2]).build()
    validator = CrossValidator(
        estimator=estimator, estimatorParamMaps=grid, evaluator=Evaluator(), numFolds=2, seed=1
    )
    model = validator.fit(frame)
    # Two folds of two maps, and the fit of the best map on the whole dataset.
    assert len(calls) == 1 + 2 * 2 + 1 and set(calls) == {estimator.uid}
    assert type(model.bestModel).__name__ == "LogisticRegressionModel"


def _pickle_error(value, protocol):
    # The type and the message of the error of the pickle of value at protocol.
    import pickle

    with pytest.raises(Exception) as error:
        pickle.dumps(value, protocol)
    return type(error.value), str(error.value)


def test_copy_deepcopy_and_pickle_of_a_model_and_its_summary(frame):
    # A model of a server-side object copies (sharing the object), and is not copied deep nor
    # pickled, at any protocol, as PySpark's of the lock of its `RemoteModelRef`, nor are the
    # DataFrames that keep it (its transform, a DataFrame of the summary of its evaluation); so is
    # its summary, which PySpark's does not pickle of the session of its DataFrame, nor a DataFrame
    # of it, and which cloudpickle does not pickle of the lock of its model.
    import copy
    import pickle
    import threading

    from pyspark import cloudpickle

    LR = _cls("classification", "LogisticRegression")
    Pipeline = _cls("pipeline", "Pipeline")
    Binarizer = _cls("feature", "Binarizer")
    model = LR(maxIter=2).fit(frame)
    model.tag = [1]
    shallow = copy.copy(model)
    assert type(shallow) is type(model) and shallow.uid == model.uid and shallow.tag is model.tag
    assert shallow.numFeatures == 2 and shallow.transform(frame).count() == 12
    for call in (lambda: copy.deepcopy(model), lambda: pickle.dumps(model)):
        with pytest.raises(TypeError, match=r"^cannot pickle '_thread.lock' object$"):
            call()
    keeping = (model, model.transform(frame), model.evaluate(frame).predictions)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        expected = _pickle_error(threading.Lock(), protocol)
        assert [_pickle_error(obj, protocol) for obj in keeping] == [expected] * 3, protocol
    summary = model.summary
    assert copy.copy(summary).totalIterations == summary.totalIterations
    for call in (lambda: copy.deepcopy(summary), lambda: cloudpickle.dumps(summary)):
        with pytest.raises(TypeError, match=r"^cannot pickle '_thread.lock' object$"):
            call()
    for of_summary in (summary, summary.predictions):
        with pytest.raises(AttributeError, match="Can't pickle local object"):
            pickle.dumps(of_summary)
    # The pickle of a summary of an evaluation, which keeps its model, or of a training summary of
    # the transform of the model (a KMeans's), raises the TypeError of the model's lock first.
    KMeans = _cls("clustering", "KMeans")
    for other in (model.evaluate(frame), KMeans(k=2, maxIter=2, seed=1).fit(frame).summary):
        for call in (lambda: copy.deepcopy(other), lambda: pickle.dumps(other)):
            with pytest.raises(TypeError, match=r"^cannot pickle '_thread.lock' object$"):
                call()
    # A PipelineModel of transformers copies, copies deep and pickles, of its stages.
    fitted = Pipeline(stages=[Binarizer(threshold=4.5, inputCol="x", outputCol="b")]).fit(frame)
    for copied in (copy.copy(fitted), copy.deepcopy(fitted), pickle.loads(pickle.dumps(fitted))):
        assert len(copied.stages) == 1 and copied.uid == fitted.uid
        assert copied.transform(frame).columns[-1] == "b"


def test_the_models_and_summaries_have_pysparks_java_obj_of_their_call_java(frame, tmp_path):
    # As PySpark's: the `_java_obj` of a model of a fit, a load, a copy, of a tree of a forest and
    # of a class method is the RemoteModelRef of its object, and that of a summary the id of its
    # object and methods, which PySpark's `_call_java` reads.
    from pyspark.ml.util import RemoteModelRef

    LR = _cls("classification", "LogisticRegression")
    LRModel = _cls("classification", "LogisticRegressionModel")
    RF = _cls("classification", "RandomForestClassifier")
    Indexer = _cls("feature", "StringIndexerModel")
    model = LR(maxIter=2).fit(frame)
    path = str(tmp_path) + "/model"
    model.save(path)
    tree = RF(numTrees=2, maxDepth=2, seed=1).fit(frame).trees[0]
    for obj, method, value in [
        (model, "numFeatures", 2),
        (LRModel.load(path), "numFeatures", 2),
        (model.copy(), "numFeatures", 2),
        (tree, "numNodes", tree.numNodes),
        (Indexer.from_labels(["a", "b"], inputCol="x"), "labelsArray", [["a", "b"]]),
    ]:
        assert type(obj._java_obj) is RemoteModelRef, obj
        assert obj._call_java(method) == value, obj
    summary = model.summary
    assert model._summary is summary and summary._java_obj == f"{model._java_obj}.summary"
    assert summary._call_java("totalIterations") == summary.totalIterations
    evaluated = model.evaluate(frame)
    assert evaluated.__source_transformer__ is model
    assert evaluated._call_java("accuracy") == evaluated.accuracy


def test_a_summary_of_no_server_side_object_raises_pysparks_errors(session):
    # As PySpark's `_extract_id_methods` of the `java_obj` of the summary: the AttributeError of its
    # `split` of a value that is not a string, and the server's error of an id that it does not hold.
    Summary = _cls("classification", "LinearSVCTrainingSummary")
    for call in (lambda: Summary().accuracy, lambda: Summary().predictions, lambda: Summary().fMeasureByLabel()):
        with pytest.raises(AttributeError, match="^'NoneType' object has no attribute 'split'$"):
            call()
    with pytest.raises(AttributeError, match="^'int' object has no attribute 'split'$"):
        Summary(123).accuracy
    with pytest.raises(Exception, match="The MLCache key abc is invalid"):
        Summary("abc").accuracy
    KMeansSummary = _cls("clustering", "KMeansSummary")
    with pytest.raises(AttributeError, match="^'NoneType' object has no attribute 'split'$"):
        KMeansSummary().k
    # As PySpark's `invoke_remote_attribute_relation`: a DataFrame member of a summary of the
    # `summary` method first (`"<id>.summary"`) reads its `_summary_dataset`, which PySpark's fit
    # sets, and a summary of another id builds a DataFrame that the server refuses.
    for summary in (Summary("abc.summary"), KMeansSummary("abc.summary")):
        message = f"^'{type(summary).__name__}' object has no attribute '_summary_dataset'$"
        with pytest.raises(AttributeError, match=message):
            summary.predictions
    with pytest.raises(Exception, match="The MLCache key abc is invalid"):
        KMeansSummary("abc").predictions.collect()
    with_dataset = KMeansSummary("abc.summary")
    with_dataset._summary_dataset = session.range(1)
    with pytest.raises(Exception, match="The MLCache key abc is invalid"):
        with_dataset.cluster.collect()
