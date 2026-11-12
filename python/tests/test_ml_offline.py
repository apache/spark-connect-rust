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
"""
pyspark.ml on this client, without a Spark Connect server: the messages that the vendored
pyspark.ml builds, of the literals, the types, the plans and the client of this client, as those
of official PySpark 4.3.0, with a mock server where a command is sent. The expected values are
those of the reference client.
"""

import datetime
import decimal
import enum
import gc
import subprocess
import sys
import textwrap
import weakref

import numpy as np
import pytest
from _mock_connect_server import mock_connect_session
from pyspark.loose_version import LooseVersion

# The generated pyspark.sql.connect.proto, of Apache Spark v4.3.0, needs protobuf >= 6.33.5.
_protobuf = pytest.importorskip("google.protobuf")
if LooseVersion(_protobuf.__version__) < LooseVersion("6.33.5"):
    pytest.skip("pyspark.sql.connect.proto needs protobuf >= 6.33.5", allow_module_level=True)

import pyspark.sql.connect.proto as pb2  # noqa: E402
from google.protobuf import text_format  # noqa: E402
from pyspark.errors import PySparkTypeError, PySparkValueError  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql import types as T  # noqa: E402


@pytest.fixture(scope="module")
def _ml_mock_server():
    pytest.importorskip("grpc")
    with mock_connect_session() as session:
        yield session


@pytest.fixture
def ml_mock(_ml_mock_server):
    # A session of its own of the shared server, which is the active session while the test
    # runs, as pyspark.ml reads it.
    _, service = _ml_mock_server
    service.reset()
    spark = SparkSession.builder.remote(service.remote).getOrCreate()
    yield spark, service
    spark.stop()


@pytest.fixture(autouse=True, scope="module")
def _debugging_read_without_a_session_of_ml():
    # PySpark's is_debugging_enabled reads spark.python.sql.dataFrameDebugging.enabled of the
    # active session at the first expression of a Column method, and keeps it: it is read
    # without a session, as test_dropin_offline reads it, before a mock server is there.
    from pyspark.errors.utils import is_debugging_enabled

    if SparkSession.getActiveSession() is None:
        is_debugging_enabled()


@pytest.fixture(autouse=True)
def _no_session_left_active_of_ml():
    yield
    session = SparkSession.getActiveSession()
    if session is not None:
        session.stop()


def _text(message):
    return text_format.MessageToString(message, as_one_line=True)


def _ml_result(**fields):
    # The ExecutePlanResponse of an MlCommandResult, which the mock server sends of a command.
    return pb2.ExecutePlanResponse(ml_command_result=pb2.MlCommandResult(**fields)).SerializeToString()


def _operator_info(ref_id):
    return _ml_result(
        operator_info=pb2.MlCommandResult.MlOperatorInfo(obj_ref=pb2.ObjectRef(id=ref_id))
    )


def _commands(service):
    # The commands of the ExecutePlanRequests that the mock server received.
    return [pb2.ExecutePlanRequest.FromString(r).plan.command for r in service.execute_requests]


def _requests_but_deletes(service):
    # The commands that the server received, but the Delete of a model that an earlier test
    # released, which can reach the server during a check.
    return sum(not command.ml_command.HasField("delete") for command in _commands(service))


def _commands_but_deletes(service, start=0):
    # The commands of the requests that the server received from the start-th on, but the
    # Deletes of models that earlier tests released: a release is sent without waiting for it,
    # so it can reach the server between a request of a test and the check of that request.
    return [c for c in _commands(service)[start:] if not c.ml_command.HasField("delete")]


def _last_request_but_deletes(service):
    # The last ExecutePlanRequest that the server received, but such a Delete.
    requests = map(pb2.ExecutePlanRequest.FromString, service.execute_requests)
    return [r for r in requests if not r.plan.command.ml_command.HasField("delete")][-1]


def _relation(df):
    # The relation of a DataFrame, as the core sends it: the relation that the plans of pyspark.ml
    # send in place of its placeholder, which the core replaces with it.
    return pb2.Relation.FromString(df._relation_bytes())


class _Color(enum.Enum):
    RED = 1


_UTF8 = 'string { collation: "UTF8_BINARY" }'


def test_ml_cache_queries_are_those_of_pyspark(ml_mock):
    # As PySpark Connect's SparkConnectClient._get_ml_cache_info and _query_model_size: the
    # get_cache_info and get_model_size commands, and the JSON of the cached objects, or the size
    # of the model, of their results.
    spark, service = ml_mock
    client = spark.client
    cached = [pb2.Expression.Literal(string='{"class": "a"}'), pb2.Expression.Literal(string="{}")]
    service.execute_payload = _ml_result(
        param=pb2.Expression.Literal(array=pb2.Expression.Literal.Array(elements=cached))
    )
    assert client._get_ml_cache_info() == ['{"class": "a"}', "{}"]
    (command,) = _commands(service)
    assert command.ml_command.HasField("get_cache_info")

    service.execute_payload = _operator_info("ref-1")
    assert client._get_ml_cache_info() == []

    service.execute_payload = _ml_result(param=pb2.Expression.Literal(long=42))
    assert client._query_model_size("ref-1") == 42
    command = _commands_but_deletes(service)[-1]
    assert command.ml_command.get_model_size.model_ref == pb2.ObjectRef(id="ref-1")


def test_ml_connect_fitted_scalers_transform_the_cached_model(ml_mock):
    # As the fitted models of PySpark Connect: fit sends an MlCommand.Fit of the estimator, and the
    # transform of the fitted model refers to the object that the server cached for the fit, not to
    # a transformer of the estimator's name, which the server rejects.
    from pyspark.ml.connect.feature import MaxAbsScaler, StandardScaler, StringIndexer

    spark, service = ml_mock
    df = spark.range(1).selectExpr("array(1.0D, 2.0D) AS f", "'a' AS s")
    models = []  # Kept alive, so that no model is released during the checks.
    for cls, name, column in [
        (StandardScaler, "org.apache.spark.ml.feature.StandardScaler", "f"),
        (MaxAbsScaler, "org.apache.spark.ml.feature.MaxAbsScaler", "f"),
        (StringIndexer, "org.apache.spark.ml.feature.StringIndexer", "s"),
    ]:
        sent = len(service.execute_requests)
        service.execute_payload = _operator_info("model-1")
        estimator = cls(inputCol=column, outputCol="o")
        models.append(estimator.fit(df))
        fit = _commands_but_deletes(service, sent)[0].ml_command.fit
        assert fit.estimator.name == name
        assert fit.estimator.type == pb2.MlOperator.OPERATOR_TYPE_ESTIMATOR
        assert dict(fit.params.params) == {
            "inputCol": pb2.Expression.Literal(string=column),
            "outputCol": pb2.Expression.Literal(string="o"),
        }
        assert fit.dataset == _relation(df)

        scaled = models[-1].transform(df)
        assert len(service.execute_requests) == sent + 1
        transform = _relation(scaled).ml_relation.transform
        assert transform.obj_ref == pb2.ObjectRef(id="model-1")
        assert not transform.HasField("transformer")
        assert transform.input == _relation(df)
        assert transform.params.params["outputCol"] == pb2.Expression.Literal(string="o")


def test_ml_connect_setters_set_the_params_of_their_receiver_as_pyspark(ml_mock):
    # As PySpark's Params._set: every setter of the native pyspark.ml.connect estimators,
    # transformers and evaluators sets its param on the object itself and returns the object, so
    # the next fit, transform or evaluate sends the value of a setter whose result is dropped,
    # as the object of the constructor of the same params sends it.
    from pyspark.ml.connect import (
        BinaryClassificationEvaluator,
        KMeans,
        LinearRegression,
        LinearSVC,
        LogisticRegression,
        MaxAbsScaler,
        MulticlassClassificationEvaluator,
        RegressionEvaluator,
        StandardScaler,
        StringIndexer,
        VectorAssembler,
    )

    spark, service = ml_mock
    df = spark.range(1)
    models = []  # Kept alive, so that no model is released during the checks.

    def value(literal):
        kind = literal.WhichOneof("literal_type")
        if kind == "array":
            return [value(element) for element in literal.array.elements]
        return getattr(literal, kind)

    def sent_params(obj):
        # The values of the params that the next evaluate, fit or transform of obj sends.
        sent = len(service.execute_requests)
        if hasattr(obj, "evaluate"):
            service.execute_payload = _ml_result(param=pb2.Expression.Literal(double=0.5))
            assert obj.evaluate(df) == 0.5
            params = _commands_but_deletes(service, sent)[0].ml_command.evaluate.params
        elif hasattr(obj, "fit"):
            service.execute_payload = _operator_info("model-1")
            models.append(obj.fit(df))
            params = _commands_but_deletes(service, sent)[0].ml_command.fit.params
        else:
            params = _relation(obj.transform(df)).ml_relation.transform.params
        return {name: value(literal) for name, literal in params.params.items()}

    for cls, values in [
        (StandardScaler, dict(inputCol="in", outputCol="out")),
        (MaxAbsScaler, dict(inputCol="in", outputCol="out")),
        (StringIndexer, dict(inputCol="in", outputCol="out")),
        (VectorAssembler, dict(inputCols=["a", "b"], outputCol="out")),
        (LogisticRegression, dict(featuresCol="f", labelCol="y", predictionCol="p", maxIter=7)),
        (
            LinearSVC,
            dict(
                featuresCol="f", labelCol="y", predictionCol="p", maxIter=7, regParam=0.5,
                tol=0.25, rawPredictionCol="r", fitIntercept=False, standardization=False,
                threshold=0.5, weightCol="w", aggregationDepth=3, maxBlockSizeInMB=1.5,
            ),
        ),
        (
            LinearRegression,
            dict(
                featuresCol="f", labelCol="y", predictionCol="p", maxIter=7, regParam=0.5,
                elasticNetParam=0.25, tol=0.125, fitIntercept=False, standardization=False,
                solver="normal", weightCol="w", aggregationDepth=3, loss="huber", epsilon=1.5,
                maxBlockSizeInMB=1.5,
            ),
        ),
        (
            KMeans,
            dict(
                featuresCol="f", predictionCol="p", k=3, initMode="random", initSteps=3,
                tol=0.5, maxIter=7, seed=5, distanceMeasure="cosine", weightCol="w",
                solver="row", maxBlockSizeInMB=1.5,
            ),
        ),
        (RegressionEvaluator, dict(labelCol="y", predictionCol="p", metricName="r2")),
        (
            BinaryClassificationEvaluator,
            dict(labelCol="y", probabilityCol="p", metricName="areaUnderPR"),
        ),
        (
            MulticlassClassificationEvaluator,
            dict(labelCol="y", predictionCol="p", metricName="accuracy"),
        ),
    ]:
        setters = {"set" + name[0].upper() + name[1:]: name for name in values}
        # Every setter of the class is checked.
        assert sorted(n for n in dir(cls) if n.startswith("set")) == sorted(setters)
        receiver = cls()
        for setter, name in setters.items():
            # By the name of the argument of PySpark's setters, value.
            assert getattr(receiver, setter)(value=values[name]) is receiver, (cls, setter)
        params = sent_params(receiver)
        sent_values = {_sent_name(cls.__name__, name): value for name, value in values.items()}
        assert {name: params.get(name) for name in sent_values} == sent_values, cls
        assert params == sent_params(cls(**values)), cls


def test_ml_pipeline_and_cross_validator_fit_is_not_implemented(ml_mock):
    # The native Pipeline and CrossValidator keep their params as their setters set them, and
    # their fit raises PySpark's NOT_IMPLEMENTED before any request: the server cannot fit the
    # stage names, nor the estimator and evaluator that the CrossValidator does not hold, so no
    # model is returned.
    from pyspark.errors import PySparkNotImplementedError
    from pyspark.ml.connect import CrossValidator, Pipeline

    spark, service = ml_mock
    df = spark.range(1)
    pipeline = Pipeline()
    assert pipeline.setStages(value=["a", "b"]) is pipeline
    cv = CrossValidator()
    assert cv.setNumFolds(value=5) is cv
    assert cv.setParallelism(value=2) is cv
    assert cv.setSeed(value=7) is cv
    for estimator, feature in [
        (pipeline, "Pipeline.fit"),
        (cv, "CrossValidator.fit"),
        (Pipeline(stages=["a"]), "Pipeline.fit"),
        (CrossValidator(numFolds=2, seed=1), "CrossValidator.fit"),
        # The params that PySpark's serialization cannot send, or sends as an empty list, do not
        # raise their errors first: the fit raises NOT_IMPLEMENTED for every param state.
        (Pipeline(stages=[]), "Pipeline.fit"),
        (CrossValidator(seed=2**64), "CrossValidator.fit"),
        (CrossValidator(seed=2**64, numFolds=2), "CrossValidator.fit"),
        (CrossValidator(numFolds=2**70), "CrossValidator.fit"),
    ]:
        sent = _requests_but_deletes(service)
        with pytest.raises(PySparkNotImplementedError) as error:
            estimator.fit(df)
        assert error.value.getCondition() == "NOT_IMPLEMENTED"
        assert error.value.getMessageParameters() == {"feature": feature}
        assert str(error.value) == f"[NOT_IMPLEMENTED] {feature} is not implemented."
        assert _requests_but_deletes(service) == sent


# The params of the native pyspark.ml.connect classes, by the TypeConverters method of each
# param in PySpark. Their constructors take them all but thresholds, which PySpark's
# RandomForestClassifier takes by its setter only.
_NATIVE_PARAMS = {
    "StandardScaler": dict(inputCol="str", outputCol="str"),
    "MaxAbsScaler": dict(inputCol="str", outputCol="str"),
    "StringIndexer": dict(inputCol="str", outputCol="str"),
    "VectorAssembler": dict(inputCols="str_list", outputCol="str"),
    "LogisticRegression": dict(
        featuresCol="str", labelCol="str", predictionCol="str", maxIter="int"
    ),
    "LinearSVC": dict(
        featuresCol="str", labelCol="str", predictionCol="str", maxIter="int", regParam="float",
        tol="float", rawPredictionCol="str", fitIntercept="bool", standardization="bool",
        threshold="float", weightCol="str", aggregationDepth="int", maxBlockSizeInMB="float",
    ),
    "LinearRegression": dict(
        featuresCol="str", labelCol="str", predictionCol="str", maxIter="int", regParam="float",
        elasticNetParam="float", tol="float", fitIntercept="bool", standardization="bool",
        solver="str", weightCol="str", aggregationDepth="int", loss="str", epsilon="float",
        maxBlockSizeInMB="float",
    ),
    "DecisionTreeClassifier": dict(
        featuresCol="str", labelCol="str", predictionCol="str", probabilityCol="str",
        rawPredictionCol="str", maxDepth="int", maxBins="int", minInstancesPerNode="int",
        minInfoGain="float", pruneTree="bool", maxMemoryInMB="int", cacheNodeIds="bool",
        checkpointInterval="int", impurity="str",
        seed="int", leafCol="str", minWeightFractionPerNode="float",
        weightCol="str", thresholds="float_list",
    ),
    "GBTClassifier": dict(
        featuresCol="str", labelCol="str", predictionCol="str",
         maxDepth="int", maxBins="int", minInstancesPerNode="int",
        minInfoGain="float", maxMemoryInMB="int", cacheNodeIds="bool",
        checkpointInterval="int", impurity="str", featureSubsetStrategy="str",
        seed="int", subsamplingRate="float", leafCol="str", minWeightFractionPerNode="float",
        lossType="str", maxIter="int", stepSize="float", validationIndicatorCol="str", weightCol="str", thresholds="float_list",
    ),
    "RandomForestClassifier": dict(
        featuresCol="str", labelCol="str", predictionCol="str", probabilityCol="str",
        rawPredictionCol="str", maxDepth="int", maxBins="int", minInstancesPerNode="int",
        minInfoGain="float", pruneTree="bool", maxMemoryInMB="int", cacheNodeIds="bool",
        checkpointInterval="int", impurity="str", numTrees="int", featureSubsetStrategy="str",
        seed="int", subsamplingRate="float", leafCol="str", minWeightFractionPerNode="float",
        weightCol="str", bootstrap="bool", thresholds="float_list",
    ),
    "DecisionTreeRegressor": dict(
        featuresCol="str", labelCol="str", predictionCol="str", maxDepth="int",
        maxBins="int", minInstancesPerNode="int", minInfoGain="float", maxMemoryInMB="int",
        cacheNodeIds="bool", checkpointInterval="int", impurity="str",
        seed="int", varianceCol="str", leafCol="str",
        minWeightFractionPerNode="float", weightCol="str",
    ),
    "RandomForestRegressor": dict(
        featuresCol="str", labelCol="str", predictionCol="str", maxDepth="int",
        maxBins="int", minInstancesPerNode="int", minInfoGain="float", maxMemoryInMB="int",
        cacheNodeIds="bool", checkpointInterval="int", impurity="str", subsamplingRate="float",
        seed="int", numTrees="int", featureSubsetStrategy="str", leafCol="str",
        minWeightFractionPerNode="float", weightCol="str", bootstrap="bool",
    ),
    "KMeans": dict(
        featuresCol="str", predictionCol="str", k="int", initMode="str", initSteps="int",
        tol="float", maxIter="int", seed="int", distanceMeasure="str", weightCol="str",
        solver="str", maxBlockSizeInMB="float",
    ),
    "RegressionEvaluator": dict(labelCol="str", predictionCol="str", metricName="str"),
    "BinaryClassificationEvaluator": dict(labelCol="str", probabilityCol="str", metricName="str"),
    "MulticlassClassificationEvaluator": dict(
        metricName="str", labelCol="str", predictionCol="str"
    ),
    "Pipeline": dict(stages="str_list"),
    "CrossValidator": dict(numFolds="int", parallelism="int", seed="int"),
}

# Of each kind, values that PySpark's converter takes, with the literal kind and value that
# PySpark sends of them, and values that it rejects, with the end of PySpark's TypeError.
_NATIVE_CONVERTED = {
    "int": [(3.0, ("integer", 3)), (np.int64(4), ("integer", 4)),
            (2**33 + 3, ("long", 8589934595))],
    "float": [(1, ("double", 1.0)), (np.float64(0.5), ("double", 0.5))],
    "str": [(np.str_("a"), ("string", "a"))],
    "bool": [(True, ("boolean", True))],
    "str_list": [(("a", "b"), ("array", ["a", "b"]))],
    "float_list": [((1, 2.5), ("array", [1.0, 2.5]))],
}
_NATIVE_REJECTED = {
    "int": [(3.5, "Could not convert 3.5 to int"), (True, "Could not convert True to int"),
            (np.int32(3), "Could not convert 3 to int"), ("3", "Could not convert 3 to int")],
    "float": [("1", "Could not convert 1 to float"), (True, "Could not convert True to float")],
    "str": [(1, "Could not convert <class 'int'> to string type")],
    "bool": [(1, "Boolean Param requires value of type bool. Found <class 'int'>.")],
    "str_list": [([1], "Could not convert [1] to list of strings"),
                 ("ab", "Could not convert ab to list of strings")],
    "float_list": [(["a"], "Could not convert ['a'] to list of floats")],
}


# The native classes whose fit raises NOT_IMPLEMENTED without a request, so that no request
# carries their params.
_NOT_FITTABLE = ("Pipeline", "CrossValidator")


def _native_request(spark, service, obj, models):
    # The MlOperator and the MlParams that the next evaluate, fit or transform of the native
    # obj sends. The models are kept in models, so that none is released.
    df = spark.range(1)
    sent = len(service.execute_requests)
    if hasattr(obj, "evaluate"):
        service.execute_payload = _ml_result(param=pb2.Expression.Literal(double=0.5))
        assert obj.evaluate(df) == 0.5
        evaluate = _commands_but_deletes(service, sent)[0].ml_command.evaluate
        return evaluate.evaluator, evaluate.params
    if hasattr(obj, "fit"):
        service.execute_payload = _operator_info("model-1")
        models.append(obj.fit(df))
        fit = _commands_but_deletes(service, sent)[0].ml_command.fit
        return fit.estimator, fit.params
    transform = _relation(obj.transform(df)).ml_relation.transform
    return transform.transformer, transform.params


def _literal_value(literal):
    kind = literal.WhichOneof("literal_type")
    if kind == "array":
        return kind, [_literal_value(element)[1] for element in literal.array.elements]
    return kind, getattr(literal, kind)


def _sent_name(name, param):
    # The name of a param in the request: the probability column of PySpark's
    # BinaryClassificationEvaluator is the rawPredictionCol that the JVM evaluator scores.
    if (name, param) == ("BinaryClassificationEvaluator", "probabilityCol"):
        return "rawPredictionCol"
    return param


def _defaults_sent(name):
    # The params that a native class sends unless it is given them: the metric of
    # MulticlassClassificationEvaluator and the probability column of
    # BinaryClassificationEvaluator, which PySpark's constructor sets.
    if name == "MulticlassClassificationEvaluator":
        return {"metricName": ("string", "accuracy")}
    if name == "BinaryClassificationEvaluator":
        return {"rawPredictionCol": ("string", "probability")}
    return {}


def _native_sent(spark, service, obj, models):
    _, params = _native_request(spark, service, obj, models)
    return {name: _literal_value(literal) for name, literal in params.params.items()}


def test_ml_connect_constructors_are_keyword_only_as_pyspark(ml_mock):
    # As PySpark's @keyword_only: the constructor of a native pyspark.ml.connect class raises
    # its TypeError for a positional argument, except MulticlassClassificationEvaluator's, which
    # is not keyword-only in PySpark and takes metricName, labelCol and predictionCol by
    # position. A wrong argument raises Python's TypeError of PySpark's __init__.
    import pyspark.ml.connect as connect

    spark, service = ml_mock
    models = []
    for name in _NATIVE_PARAMS:
        cls = getattr(connect, name)
        with pytest.raises(TypeError) as error:
            cls(bogus=1)
        assert str(error.value) == f"{name}.__init__() got an unexpected keyword argument 'bogus'"
        if name == "MulticlassClassificationEvaluator":
            continue
        for args in [("a",), (1, 2)]:
            with pytest.raises(TypeError) as error:
                cls(*args)
            assert str(error.value) == "Method __init__ forces keyword arguments.", name
    evaluator = connect.MulticlassClassificationEvaluator("accuracy", "y", predictionCol="p")
    assert _native_sent(spark, service, evaluator, models) == {
        "metricName": ("string", "accuracy"),
        "labelCol": ("string", "y"),
        "predictionCol": ("string", "p"),
    }
    with pytest.raises(TypeError) as error:
        connect.MulticlassClassificationEvaluator("a", "b", "c", "d")
    assert str(error.value) == (
        "MulticlassClassificationEvaluator.__init__() takes from 1 to 4 positional arguments "
        "but 5 were given"
    )
    with pytest.raises(TypeError) as error:
        connect.MulticlassClassificationEvaluator("a", metricName="b")
    assert str(error.value) == (
        "MulticlassClassificationEvaluator.__init__() got multiple values for argument "
        "'metricName'"
    )


def test_ml_connect_params_convert_with_the_type_converters_of_pyspark(ml_mock):
    # As PySpark's Params._set with the TypeConverters of each param, in the constructor and
    # the setter of every native class: a value that the converter takes is sent as PySpark
    # sends what it converts it to (an integral float as an int, an int beyond 32 bits as a
    # long), and one that it rejects raises PySpark's TypeError.
    import pyspark.ml.connect as connect

    spark, service = ml_mock
    models = []
    for name, params in _NATIVE_PARAMS.items():
        cls = getattr(connect, name)
        for param, kind in params.items():
            setter = "set" + param[0].upper() + param[1:]
            for value, expected in _NATIVE_CONVERTED[kind]:
                made = [getattr(cls(), setter)(value)]
                if param != "thresholds":
                    made.append(cls(**{param: value}))
                for obj in made:
                    if name in _NOT_FITTABLE:
                        continue  # Their params are sent by no request.
                    sent = _native_sent(spark, service, obj, models)
                    assert sent == {
                        **_defaults_sent(name), _sent_name(name, param): expected
                    }, (name, param, value)
            for value, message in _NATIVE_REJECTED[kind]:
                expected = f'Invalid param value given for param "{param}". {message}'
                with pytest.raises(TypeError) as error:
                    getattr(cls(), setter)(value)
                assert str(error.value) == expected, (name, param, value)
                if param != "thresholds":
                    with pytest.raises(TypeError) as error:
                        cls(**{param: value})
                    assert str(error.value) == expected, (name, param, value)
    with pytest.raises(TypeError) as error:
        connect.RandomForestClassifier(thresholds=[0.5, 0.5])
    assert str(error.value) == (
        "RandomForestClassifier.__init__() got an unexpected keyword argument 'thresholds'"
    )


def test_ml_connect_constructors_send_only_the_params_they_are_given_as_pyspark(ml_mock):
    # As PySpark, which sends the params of its _paramMap, those that the constructor and the
    # setters set: a native class sends none of its defaults, which the server applies.
    import pyspark.ml.connect as connect

    spark, service = ml_mock
    models = []
    for name in _NATIVE_PARAMS:
        if name not in _NOT_FITTABLE:
            sent = _native_sent(spark, service, getattr(connect, name)(), models)
            assert sent == _defaults_sent(name), name
    assert _native_sent(spark, service, connect.KMeans(k=3), models) == {"k": ("integer", 3)}
    forest = connect.RandomForestClassifier(numTrees=3, seed=2**33)
    assert _native_sent(spark, service, forest, models) == {
        "numTrees": ("integer", 3),
        "seed": ("long", 2**33),
    }


def test_ml_connect_evaluators_take_the_signatures_and_defaults_of_pyspark(ml_mock):
    # As PySpark's pyspark.ml.connect evaluators: their signatures, and the metric of
    # MulticlassClassificationEvaluator, "accuracy" unless given, which its constructor sets, where
    # the JVM evaluator's default is "f1".
    import inspect

    import pyspark.ml.connect as connect

    spark, service = ml_mock
    signatures = {
        "RegressionEvaluator": "(*, metricName='rmse', labelCol='label', predictionCol='prediction')",
        "BinaryClassificationEvaluator":
            "(*, metricName='areaUnderROC', labelCol='label', probabilityCol='probability')",
        "MulticlassClassificationEvaluator":
            "(metricName='accuracy', labelCol='label', predictionCol='prediction')",
    }
    for name, signature in signatures.items():
        assert str(inspect.signature(getattr(connect, name))) == signature, name
    models = []
    evaluator = connect.MulticlassClassificationEvaluator
    assert _native_sent(spark, service, evaluator(), models) == {
        "metricName": ("string", "accuracy")
    }
    assert _native_sent(spark, service, evaluator(metricName="f1"), models) == {
        "metricName": ("string", "f1")
    }
    assert _native_sent(spark, service, evaluator("weightedPrecision"), models) == {
        "metricName": ("string", "weightedPrecision")
    }


def test_ml_connect_binary_evaluator_sends_its_probability_col_as_the_jvm_raw_prediction(ml_mock):
    # As PySpark's BinaryClassificationEvaluator, whose probabilityCol defaults to "probability"
    # and which has no rawPredictionCol. The JVM evaluator scores its rawPredictionCol, so the
    # probability column is sent as that param.
    import pyspark.ml.connect as connect

    spark, service = ml_mock
    models = []
    evaluator = connect.BinaryClassificationEvaluator
    assert _native_sent(spark, service, evaluator(), models) == {
        "rawPredictionCol": ("string", "probability")
    }
    assert _native_sent(spark, service, evaluator(probabilityCol="p"), models) == {
        "rawPredictionCol": ("string", "p")
    }
    binary = evaluator(metricName="areaUnderPR", labelCol="y", probabilityCol="p")
    assert _native_sent(spark, service, binary, models) == {
        "metricName": ("string", "areaUnderPR"),
        "labelCol": ("string", "y"),
        "rawPredictionCol": ("string", "p"),
    }
    with pytest.raises(TypeError) as error:
        evaluator(rawPredictionCol="r")
    assert str(error.value) == (
        "BinaryClassificationEvaluator.__init__() got an unexpected keyword argument "
        "'rawPredictionCol'"
    )
    assert not hasattr(evaluator(), "setRawPredictionCol")


def test_ml_connect_unsendable_ints_none_and_uids_as_pyspark(ml_mock):
    # As PySpark: an int beyond a long is taken by the constructor and the setter, and the
    # next fit raises the VALUE_NOT_BETWEEN of PySpark's serialization of the params, before
    # any request, until the param is set again. A CrossValidator raises NOT_IMPLEMENTED instead,
    # see test_ml_pipeline_and_cross_validator_fit_is_not_implemented. None is sent as a null,
    # as PySpark sends it.
    # The uid of every native class is the name of its class and 12 hex digits, as PySpark's.
    import re

    import pyspark.ml.connect as connect

    spark, service = ml_mock
    models = []
    for estimator in [
        connect.KMeans(seed=2**64),
        connect.KMeans().setSeed(-(2**63) - 1),
        connect.RandomForestClassifier().setNumTrees(2**70),
    ]:
        sent = _requests_but_deletes(service)
        with pytest.raises(PySparkValueError) as error:
            estimator.fit(spark.range(1))
        assert error.value.getCondition() == "VALUE_NOT_BETWEEN"
        assert str(error.value) == (
            "[VALUE_NOT_BETWEEN] Value for `value` must be between -9223372036854775808 and "
            "32767."
        )
        assert _requests_but_deletes(service) == sent
    estimator = connect.KMeans(seed=2**64)
    assert estimator.setSeed(5) is estimator
    assert _native_sent(spark, service, estimator, models) == {"seed": ("integer", 5)}
    sent = _native_sent(spark, service, connect.KMeans(seed=2**64, k=2).setSeed(None), models)
    assert sent.keys() == {"k", "seed"} and sent["k"] == ("integer", 2)
    assert sent["seed"][0] == "null"

    for name, params in _NATIVE_PARAMS.items():
        if name in _NOT_FITTABLE:
            continue  # Their params and uids are sent by no request.
        cls = getattr(connect, name)
        for param, kind in params.items():
            setter = "set" + param[0].upper() + param[1:]
            value = _NATIVE_CONVERTED[kind][0][0]
            obj = cls()
            assert getattr(obj, setter)(value) is obj
            assert getattr(obj, setter)(None) is obj
            sent_name = _sent_name(name, param)
            sent = _native_sent(spark, service, obj, models)
            assert sent.keys() == {sent_name} | _defaults_sent(name).keys(), (name, param)
            assert sent[sent_name][0] == "null", (name, param)
            if param != "thresholds":
                sent = _native_sent(spark, service, cls(**{param: None}), models)
                assert sent.keys() == {sent_name} | _defaults_sent(name).keys(), (name, param)
                assert sent[sent_name][0] == "null", (name, param)
        operator, _ = _native_request(spark, service, cls(), models)
        assert re.fullmatch(name + "_[0-9a-f]{12}", operator.uid), operator.uid


def test_ml_connect_setters_run_while_a_fit_or_an_evaluate_waits_as_pyspark(ml_mock):
    # As PySpark, whose fit and evaluate hold no lock of the estimator or the evaluator: a
    # setter that another thread calls while a fit or an evaluate of the same native object
    # waits for the server sets the param, and the waiting call sends the params of when it
    # started. (Pipeline, CrossValidator and VectorAssembler send no request to wait for.)
    import threading

    import pyspark.ml.connect as connect

    spark, service = ml_mock
    df = spark.range(1)
    # Kept alive, so that the release of no model reaches the server after a reset.
    models = []
    for name, params in _NATIVE_PARAMS.items():
        if name in ("Pipeline", "CrossValidator", "VectorAssembler"):
            continue
        param = next(param for param, kind in params.items() if kind == "str")
        setter = "set" + param[0].upper() + param[1:]
        obj = getattr(connect, name)()
        service.reset()
        service.block_execute = True
        if hasattr(obj, "evaluate"):
            service.execute_payload = _ml_result(param=pb2.Expression.Literal(double=0.5))
            call = obj.evaluate
        else:
            service.execute_payload = _operator_info("model-1")
            call = obj.fit
        outcome = []

        def run():
            try:
                outcome.append(call(df))
            except Exception as error:  # noqa: BLE001 - checked below
                outcome.append(error)

        waiting = threading.Thread(target=run)
        waiting.start()
        try:
            assert service.wait_execute_started(1), name
            assert getattr(obj, setter)("changed") is obj
        finally:
            service.release_response()
            waiting.join(10)
        models.append(outcome)
        assert len(outcome) == 1 and not isinstance(outcome[0], Exception), (name, outcome)
        (command,) = _commands(service)
        held = command.ml_command.evaluate if hasattr(obj, "evaluate") else command.ml_command.fit
        assert set(held.params.params) == set(_defaults_sent(name)), name
        assert _native_sent(spark, service, obj, models) == {
            **_defaults_sent(name),
            param: ("string", "changed"),
        }


def test_native_models_released_while_their_session_stops_send_their_delete_as_pyspark(ml_mock):
    # As PySpark's del_remote_cache, which deletes through the active session, whose client is
    # closed only once its ReleaseSession ends: a native model that goes while that release waits
    # for the server sends its Delete, and the session is not stopped yet. One that goes once the
    # session is stopped sends none: stopping the session released its ML cache on the server.
    import threading
    import time

    from pyspark.ml.connect import KMeans

    def deleted():
        # The IDs that the Deletes since the fits name. A Delete of a model that an earlier test
        # released can arrive meanwhile, so the models of this test have IDs of their own.
        return [ref.id for command in _commands(service)[sent:]
                if command.ml_command.HasField("delete")
                for ref in command.ml_command.delete.obj_refs]

    spark, service = ml_mock
    service.execute_payload = _operator_info("model-released-while-stopping")
    during = KMeans(k=2).fit(spark.range(1))
    service.execute_payload = _operator_info("model-released-once-stopped")
    after = KMeans(k=2).fit(spark.range(1))
    sent = len(service.execute_requests)
    service.release_session_blocked = True
    stopper = threading.Thread(target=spark.stop)
    stopper.start()
    try:
        deadline = time.time() + 10
        while service.release_session_calls == 0 and time.time() < deadline:
            time.sleep(0.01)
        assert service.release_session_calls == 1
        assert not spark.is_stopped
        del during
        gc.collect()
        # The release is sent without waiting for it.
        deadline = time.time() + 5
        while "model-released-while-stopping" not in deleted() and time.time() < deadline:
            time.sleep(0.05)
        assert "model-released-while-stopping" in deleted()
    finally:
        service.release_session_release()
        stopper.join(10)
    assert spark.is_stopped
    del after
    gc.collect()
    time.sleep(0.5)
    assert "model-released-once-stopped" not in deleted()


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "literal { null { null { } } }"),
        (1, "literal { integer: 1 }"),
        (2**31, "literal { long: 2147483648 }"),
        (1.5, "literal { double: 1.5 }"),
        ("a", 'literal { string: "a" }'),
        (True, "literal { boolean: true }"),
        (b"ab", 'literal { binary: "ab" }'),
        (decimal.Decimal("1.5"), 'literal { decimal { value: "1.5" precision: 10 scale: 0 } }'),
        (datetime.date(2020, 1, 2), "literal { date: 18263 }"),
        (datetime.time(1, 2, 3, 4), "literal { time { nano: 3723000004000 precision: 6 } }"),
        (datetime.timedelta(days=1, microseconds=3), "literal { day_time_interval: 86400000003 }"),
        (np.int64(3), "literal { long: 3 }"),
        (np.int8(3), "literal { byte: 3 }"),
        (np.float32(1.5), "literal { float: 1.5 }"),
        (np.bool_(True), "literal { boolean: true }"),
        (_Color.RED, "literal { integer: 1 }"),
        (
            [1.0, 2.0],
            "literal { array { element_type { double { } } elements { double: 1.0 } "
            "elements { double: 2.0 } } }",
        ),
        (
            ["a"],
            f'literal {{ array {{ element_type {{ {_UTF8} }} elements {{ string: "a" }} }} }}',
        ),
        (
            [1, None],
            "literal { array { element_type { integer { } } elements { integer: 1 } "
            "elements { null { integer { } } } } }",
        ),
        (
            [[1.0, 2.0], [3.0]],
            "literal { array { element_type { array { element_type { double { } } "
            "contains_null: true } } elements { array { element_type { double { } } "
            "elements { double: 1.0 } elements { double: 2.0 } } } elements { array { "
            "element_type { double { } } elements { double: 3.0 } } } } }",
        ),
    ],
)
def test_ml_literals_of_inferred_types_as_pyspark(value, expected):
    # As official PySpark 4.3.0's LiteralExpression._from_value, of which pyspark.ml sends the
    # values of its params: the type of a value, of a list that of its first element.
    from pyspark.sql.connect.expressions import LiteralExpression

    assert _text(LiteralExpression._from_value(value).to_plan(None)) == expected


@pytest.mark.parametrize(
    "value, data_type, expected",
    [
        (None, T.IntegerType(), "literal { null { integer { } } }"),
        (
            None,
            T.ArrayType(T.StringType()),
            f"literal {{ null {{ array {{ element_type {{ {_UTF8} }} contains_null: true }} }} }}",
        ),
        (1, T.LongType(), "literal { long: 1 }"),
        (1, T.ShortType(), "literal { short: 1 }"),
        (1.0, T.FloatType(), "literal { float: 1.0 }"),
        ([], T.ArrayType(T.StringType()), f"literal {{ array {{ element_type {{ {_UTF8} }} }} }}"),
        (
            [1.0, None],
            T.ArrayType(T.DoubleType()),
            "literal { array { element_type { double { } } elements { double: 1.0 } "
            "elements { null { double { } } } } }",
        ),
        (
            decimal.Decimal("1.25"),
            T.DecimalType(5, 2),
            'literal { decimal { value: "1.25" precision: 5 scale: 2 } }',
        ),
        (
            datetime.datetime(2020, 1, 2),
            T.TimestampNTZType(),
            "literal { timestamp_ntz: 1577923200000000 }",
        ),
        (
            datetime.time(1, 2),
            T.TimeType(),
            "literal { time { nano: 3720000000000 precision: 6 } }",
        ),
        (np.int32(1), T.IntegerType(), "literal { integer: 1 }"),
    ],
)
def test_ml_literals_of_given_types_as_pyspark(value, data_type, expected):
    # As official PySpark 4.3.0's LiteralExpression(value, dataType), of which pyspark.ml sends
    # the arguments of the types it gives, such as an empty list of strings.
    from pyspark.sql.connect.expressions import LiteralExpression

    assert _text(LiteralExpression(value=value, dataType=data_type).to_plan(None)) == expected


def test_ml_literals_raise_the_errors_of_pyspark():
    # As official PySpark 4.3.0: a value of no literal type, an int out of the range of a long,
    # a list of no element type, and a value of another type than the given one.
    from pyspark.sql.connect.expressions import LiteralExpression

    for value, name in [((1, 2), "tuple"), ({"a": 1}, "dict"), (object(), "object")]:
        with pytest.raises(PySparkTypeError) as error:
            LiteralExpression._from_value(value)
        assert error.value.getCondition() == "UNSUPPORTED_DATA_TYPE"
        assert error.value.getMessageParameters() == {"data_type": name}
    with pytest.raises(PySparkValueError) as error:
        LiteralExpression._from_value(2**63)
    assert error.value.getCondition() == "VALUE_NOT_BETWEEN"
    for value in [[], [None]]:
        with pytest.raises(PySparkTypeError) as error:
            LiteralExpression._from_value(value)
        assert error.value.getCondition() == "CANNOT_INFER_ARRAY_ELEMENT_TYPE"
    for value, data_type in [
        (1, T.StringType()),
        ("a", T.IntegerType()),
        (300, T.ByteType()),
        (1, T.BooleanType()),
        ({"a": 1}, T.MapType(T.StringType(), T.IntegerType())),
    ]:
        with pytest.raises(AssertionError):
            LiteralExpression(value, data_type)
    with pytest.raises(AssertionError):
        LiteralExpression._from_value([1, "a"]).to_plan(None)


def test_ml_literal_values_as_pyspark():
    # As official PySpark 4.3.0's LiteralExpression._to_value, of which pyspark.ml reads the
    # values of the results of the server.
    from pyspark.sql.connect.expressions import LiteralExpression

    for value in [
        None,
        1,
        2**40,
        1.5,
        "a",
        True,
        b"ab",
        decimal.Decimal("1.5"),
        datetime.date(2020, 1, 2),
        datetime.time(1, 2, 3, 4),
        datetime.timedelta(days=1, microseconds=3),
        [1.0, 2.0],
        [["a"], ["b", "c"]],
    ]:
        literal = LiteralExpression._from_value(value).to_plan(None).literal
        assert LiteralExpression._to_value(literal) == value
    assert type(LiteralExpression._to_value(pb2.Expression.Literal(binary=b"x"))) is bytes

    struct_type = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())])
    struct = pb2.Expression.Literal()
    struct.struct.struct_type.CopyFrom(_data_type_proto(struct_type))
    struct.struct.elements.append(pb2.Expression.Literal(integer=1))
    struct.struct.elements.append(pb2.Expression.Literal(string="x"))
    row = LiteralExpression._to_value(struct)
    assert row == T.Row(a=1, b="x") and row.asDict() == {"a": 1, "b": "x"}

    mapping = pb2.Expression.Literal()
    mapping.map.key_type.CopyFrom(_data_type_proto(T.StringType()))
    mapping.map.value_type.CopyFrom(_data_type_proto(T.LongType()))
    mapping.map.keys.append(pb2.Expression.Literal(string="k"))
    mapping.map.values.append(pb2.Expression.Literal(long=2))
    assert LiteralExpression._to_value(mapping) == {"k": 2}

    ntz = pb2.Expression.Literal(timestamp_ntz=1577923200000000)
    assert LiteralExpression._to_value(ntz) == datetime.datetime(2020, 1, 2)
    with pytest.raises(PySparkTypeError) as error:
        LiteralExpression._to_value(pb2.Expression.Literal())
    assert error.value.getCondition() == "UNSUPPORTED_LITERAL"


def test_ml_literal_reprs_as_pyspark():
    from pyspark.sql.connect.expressions import Expression, LiteralExpression

    assert repr(LiteralExpression(None, T.IntegerType())) == "NULL"
    assert repr(LiteralExpression(1, T.LongType())) == "1"
    assert repr(LiteralExpression(datetime.date(2020, 1, 2), T.DateType())) == "2020-01-02"
    assert repr(LiteralExpression(datetime.time(1, 2), T.TimeType())) == "01:02:00.000000"
    assert (
        repr(LiteralExpression(datetime.datetime(2020, 1, 2, 3), T.TimestampNTZType()))
        == "2020-01-02 03:00:00.000000"
    )
    assert repr(LiteralExpression(datetime.timedelta(seconds=90), T.DayTimeIntervalType())) == (
        "P0DT0H1M30S"
    )
    literal = LiteralExpression._from_value(1)
    assert isinstance(literal, Expression)
    assert literal.children == [] and literal.origin is None
    seen = []
    literal.foreach(seen.append)
    assert seen == [literal]


def _data_type_proto(data_type):
    from pyspark.sql.connect.types import pyspark_types_to_proto_types

    return pyspark_types_to_proto_types(data_type)


@pytest.mark.parametrize(
    "data_type, expected",
    [
        (T.NullType(), "null { }"),
        (T.StringType(), _UTF8),
        (T.StringType("UTF8_LCASE"), 'string { collation: "UTF8_LCASE" }'),
        (T.CharType(3), "char { length: 3 }"),
        (T.DecimalType(5, 2), "decimal { scale: 2 precision: 5 }"),
        (T.TimeType(3), "time { precision: 3 }"),
        (T.DayTimeIntervalType(0, 1), "day_time_interval { start_field: 0 end_field: 1 }"),
        (
            T.ArrayType(T.StringType(), False),
            f"array {{ element_type {{ {_UTF8} }} }}",
        ),
        (
            T.MapType(T.IntegerType(), T.DoubleType()),
            "map { key_type { integer { } } value_type { double { } } value_contains_null: true }",
        ),
        (
            T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.LongType(), False)]),
            'struct { fields { name: "a" data_type { integer { } } nullable: true } '
            'fields { name: "b" data_type { long { } } } }',
        ),
    ],
)
def test_ml_data_types_are_those_of_pyspark(data_type, expected):
    # As official PySpark 4.3.0's pyspark_types_to_proto_types and
    # proto_schema_to_pyspark_data_type, of which pyspark.ml sends and reads types.
    from pyspark.sql.connect.types import proto_schema_to_pyspark_data_type

    message = _data_type_proto(data_type)
    assert isinstance(message, pb2.DataType)
    assert _text(message) == expected
    assert proto_schema_to_pyspark_data_type(message) == data_type


@pytest.mark.parametrize(
    "build, expected",
    [
        (
            lambda P: P.Read("db.t", {"k": "v"}, True),
            'read { named_table { unparsed_identifier: "db.t" options { key: "k" value: "v" } } '
            "is_streaming: true }",
        ),
        (lambda P: P.Range(0, 3, 1, 2), "range { start: 0 end: 3 step: 1 num_partitions: 2 }"),
        (lambda P: P.Range(5, 9, 2), "range { start: 5 end: 9 step: 2 }"),
        (
            lambda P: P.SQL("SELECT ?", args=[F.lit(1)]),
            'sql { query: "SELECT ?" pos_arguments { literal { integer: 1 } } }',
        ),
        (
            lambda P: P.SQL("SELECT :a", named_args={"a": F.lit("x")}),
            'sql { query: "SELECT :a" named_arguments { key: "a" value { literal { string: "x" } } } }',
        ),
        (
            lambda P: P.SQL("SELECT * FROM v", views=[P.SubqueryAlias(P.Range(0, 1, 1), "v")]),
            'with_relations { root { sql { query: "SELECT * FROM v" } } references { subquery_alias '
            '{ input { range { start: 0 end: 1 step: 1 } } alias: "v" } } }',
        ),
        (
            lambda P: P.SubqueryAlias(P.Read("t"), "x"),
            'subquery_alias { input { read { named_table { unparsed_identifier: "t" } } } alias: "x" }',
        ),
    ],
)
def test_connect_plans_build_the_relations_of_pyspark(build, expected):
    # As official PySpark 4.3.0's plans of pyspark.sql.connect.plan, of which
    # pyspark.testing.connectutils builds DataFrames: their relations, of their own plan IDs.
    import re

    import pyspark.sql.connect.plan as P

    plan = build(P)
    relation = plan.plan(None)
    assert relation.common.plan_id == plan._plan_id
    assert re.sub(r"common \{ plan_id: \d+ \} ?", "", _text(relation)) == expected


def test_connect_sql_plans_as_pyspark():
    # As official PySpark 4.3.0: the command of a SQL plan, its print, and its arguments, which
    # are Columns.
    import pyspark.sql.connect.plan as P

    command = P.SQL("SELECT 1").command(None)
    assert command.sql_command.input.sql.query == "SELECT 1"
    assert P.Read("t").print(2) == "  <Read table_name=t>\n"
    with pytest.raises(AssertionError):
        P.SQL("SELECT ?", args=[1])
    with pytest.raises(AssertionError):
        P.SQL("SELECT 1", views=[P.Range(0, 1, 1)])


def test_column_to_plan_as_pyspark():
    # As PySpark Connect's Column.to_plan: the spark.connect.Expression of the column.
    from pyspark.sql import functions as F

    assert _text(F.col("a").to_plan(None)) == (
        'unresolved_attribute { unparsed_identifier: "a" is_metadata_column: false }'
    )
    column = F.lit(1) + F.col("b")
    assert column.to_plan(None) == column._expr.to_plan(None)
    assert column.to_plan(None).unresolved_function.function_name == "+"


def test_connect_type_helpers_as_pyspark():
    # As official PySpark 4.3.0's helpers of pyspark.sql.connect.types.
    from pyspark.sql.connect.types import (
        parse_attr_name,
        verify_col_name,
        verify_numeric_col_name,
    )

    assert parse_attr_name("a.`b.c`.d") == ["a", "b.c", "d"]
    assert parse_attr_name("`a``b`") == ["a`b"]
    assert parse_attr_name("a..b") is None and parse_attr_name("`a`b") is None
    assert parse_attr_name("a.") is None and parse_attr_name("`a") is None
    schema = T.StructType(
        [
            T.StructField("n", T.IntegerType()),
            T.StructField("s", T.StringType()),
            T.StructField(
                "st", T.StructType([T.StructField("x", T.DoubleType()), T.StructField("y", T.StringType())])
            ),
            T.StructField("arr", T.ArrayType(T.StructType([T.StructField("z", T.LongType())]))),
        ]
    )
    assert verify_col_name("st.x", schema) and verify_col_name("arr.z", schema)
    assert not verify_col_name("st.w", schema) and not verify_col_name("a..b", schema)
    assert verify_numeric_col_name("n", schema) and verify_numeric_col_name("st.x", schema)
    assert verify_numeric_col_name("arr.z", schema)
    assert not verify_numeric_col_name("s", schema) and not verify_numeric_col_name("st.y", schema)


def test_connect_conversions_as_pyspark(ml_mock):
    # As official PySpark 4.3.0's pyspark.sql.connect.conversion: the storage levels of their
    # messages, and the DataFrame of a relation cached on the server, which removes the relation
    # from the server once it is collected.
    from pyspark.sql.connect.conversion import (
        proto_to_remote_cached_dataframe,
        proto_to_storage_level,
        storage_level_to_proto,
    )
    from pyspark.storagelevel import StorageLevel

    spark, service = ml_mock
    level = StorageLevel(True, True, False, False, 2)
    message = storage_level_to_proto(level)
    assert message == pb2.StorageLevel(
        use_disk=True, use_memory=True, use_off_heap=False, deserialized=False, replication=2
    )
    assert proto_to_storage_level(message) == level

    df = proto_to_remote_cached_dataframe(pb2.CachedRemoteRelation(relation_id="r1"))
    assert _relation(df).cached_remote_relation.relation_id == "r1"
    assert service.execute_requests == []
    del df
    gc.collect()
    (command,) = _commands(service)
    assert command.remove_cached_remote_relation_command.relation.relation_id == "r1"


def test_ml_vector_udt_is_sent_as_pyspark():
    from pyspark.ml.linalg import VectorUDT
    from pyspark.sql.connect.types import proto_schema_to_pyspark_data_type

    message = _data_type_proto(VectorUDT())
    assert message.udt.jvm_class == "org.apache.spark.ml.linalg.VectorUDT"
    assert message.udt.python_class == "pyspark.ml.linalg.VectorUDT"
    assert isinstance(proto_schema_to_pyspark_data_type(message), VectorUDT)


def test_user_context_extensions_are_sent_as_pyspark(ml_mock):
    # As PySpark Connect's SparkConnectClient: the user context extensions of the current thread
    # and the global ones, of the thread_local of the client, which the requests of the
    # executions and the analyses of a thread carry, the global ones first.
    import threading

    from google.protobuf import any_pb2

    spark, service = ml_mock
    client = spark.client

    def packed(graph_id):
        extension = any_pb2.Any()
        extension.Pack(pb2.PipelineAnalysisContext(dataflow_graph_id=graph_id))
        return extension

    local = client.add_threadlocal_user_context_extension(packed("local"))
    shared = client.add_global_user_context_extension(packed("global"))
    assert local.startswith("threadlocal_") and shared.startswith("global_")
    assert client.thread_local.user_context_extensions == [(local, packed("local"))]
    assert client.global_user_context_extensions == [(shared, packed("global"))]

    command = pb2.Command()
    command.ml_command.clean_cache.SetInParent()
    client.execute_command(command)
    request = _last_request_but_deletes(service)
    assert list(request.user_context.extensions) == [packed("global"), packed("local")]
    assert spark.range(1).schema is not None
    analysis = pb2.AnalyzePlanRequest.FromString(service.analyze_requests[-1])
    assert list(analysis.user_context.extensions) == [packed("global"), packed("local")]

    # Another thread has the global extensions only.
    seen = []

    def other():
        seen.append(client.thread_local.user_context_extensions)
        client.execute_command(command)

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert seen == [[]]
    request = _last_request_but_deletes(service)
    assert list(request.user_context.extensions) == [packed("global")]

    client.remove_user_context_extension(local)
    assert client.thread_local.user_context_extensions == []
    client.clear_user_context_extensions()
    assert client.global_user_context_extensions == []
    client.execute_command(command)
    request = _last_request_but_deletes(service)
    assert list(request.user_context.extensions) == []


def test_pipeline_analysis_context_is_an_extension_of_its_block_as_pyspark(ml_mock):
    # As official PySpark 4.3.0's add_pipeline_analysis_context: the PipelineAnalysisContext of
    # the requests of its block, of the current thread.
    from pyspark.pipelines.add_pipeline_analysis_context import add_pipeline_analysis_context

    spark, service = ml_mock
    command = pb2.Command()
    command.ml_command.clean_cache.SetInParent()
    with add_pipeline_analysis_context(spark, "graph-1", "flow-1"):
        spark.client.execute_command(command)
    spark.client.execute_command(command)
    inside, after = [pb2.ExecutePlanRequest.FromString(r) for r in service.execute_requests]
    (extension,) = inside.user_context.extensions
    context = pb2.PipelineAnalysisContext()
    assert extension.Unpack(context)
    assert (context.dataflow_graph_id, context.flow_name) == ("graph-1", "flow-1")
    assert list(after.user_context.extensions) == []


def test_ml_session_client_is_one_object_of_the_session(ml_mock):
    # As PySpark Connect's SparkSession.client: one object of each session, of its session ID.
    from pyspark.sql.connect.client import SparkConnectClient

    spark, service = ml_mock
    client = spark.client
    assert isinstance(client, SparkConnectClient)
    assert spark.client is client
    # The server-side session ID is None until a response carries one, as PySpark's is, and
    # it is settable, as PySpark's tests set it.
    assert client._server_session_id is None
    client._server_session_id = "srv-A"
    assert client._server_session_id == "srv-A"
    client._server_session_id = None
    other = SparkSession.builder.remote(service.remote).create()
    try:
        assert other.client is not client
        assert other.client._server_session_id is None
    finally:
        other.stop()


def test_dataframe_session_and_session_client_as_pyspark(ml_mock):
    # As PySpark Connect's DataFrame._session and SparkSession._client, which the modules of
    # PySpark read, such as pyspark.ml, pyspark.sql.plot and pyspark.pandas.
    spark, _ = ml_mock
    session = spark.range(1)._session
    assert isinstance(session, SparkSession)
    assert session.session_id == spark.session_id
    assert spark._client is spark.client


def test_execute_command_converts_the_results_of_pyspark(ml_mock):
    # As PySpark Connect's SparkConnectClient.execute_command: the resources of a
    # GetResourcesCommand, as _resources returns them, the profile ID of a resource profile, the
    # DataFrame of the relation of a checkpoint, and the event of a pipeline.
    from pyspark.resource.information import ResourceInformation

    spark, service = ml_mock
    client = spark.client
    resources = pb2.GetResourcesCommandResult()
    resources.resources["gpu"].name = "gpu"
    resources.resources["gpu"].addresses.extend(["0", "1"])
    service.execute_payload = pb2.ExecutePlanResponse(
        get_resources_command_result=resources
    ).SerializeToString()
    gpu = client._resources()["gpu"]
    assert isinstance(gpu, ResourceInformation)
    assert (gpu.name, gpu.addresses) == ("gpu", ["0", "1"])
    (command,) = _commands(service)
    assert command.HasField("get_resources_command")

    service.execute_payload = pb2.ExecutePlanResponse(
        create_resource_profile_command_result=pb2.CreateResourceProfileCommandResult(profile_id=7)
    ).SerializeToString()
    assert client.execute_command(command)[1] == {"create_resource_profile_command_result": 7}

    service.execute_payload = pb2.ExecutePlanResponse(
        pipeline_event_result=pb2.PipelineEventResult(event=pb2.PipelineEvent(message="m"))
    ).SerializeToString()
    assert client.execute_command(command)[1]["pipeline_event_result"].event.message == "m"

    service.execute_payload = pb2.ExecutePlanResponse(
        checkpoint_command_result=pb2.CheckpointCommandResult(
            relation=pb2.CachedRemoteRelation(relation_id="r9")
        )
    ).SerializeToString()
    df = client.execute_command(command)[1]["checkpoint_command_result"]
    assert _relation(df).cached_remote_relation.relation_id == "r9"
    # The relation is removed from the server once the DataFrame is collected, of a command
    # whose responses are ignored, as PySpark's: so the checkpoint that the mock server answers
    # with makes no other relation to remove.
    del df
    gc.collect()
    removed = [
        command.remove_cached_remote_relation_command.relation.relation_id
        for command in _commands(service)
        if command.HasField("remove_cached_remote_relation_command")
    ]
    assert removed == ["r9"]


def test_ml_execute_command_returns_the_results_of_pyspark(ml_mock):
    # As PySpark Connect's SparkSession.client.execute_command: no data, the results of the
    # responses by their fields, and no execution information.
    spark, service = ml_mock
    command = pb2.Command()
    command.ml_command.clean_cache.SetInParent()
    service.execute_payload = _operator_info("ref-1")
    data, properties, info = spark.client.execute_command(command)
    assert data is None and info is None
    assert list(properties) == ["ml_command_result"]
    assert properties["ml_command_result"].operator_info.obj_ref.id == "ref-1"
    request = _last_request_but_deletes(service)
    assert request.session_id == spark.session_id
    assert request.plan.command == command

    # The result of a SQL command is its relation, as SparkConnectClient's.
    relation = pb2.Relation(sql=pb2.SQL(query="SELECT 1"))
    service.execute_payload = pb2.ExecutePlanResponse(
        sql_command_result=pb2.ExecutePlanResponse.SqlCommandResult(relation=relation)
    ).SerializeToString()
    _, properties, _ = spark.client.execute_command(command)
    assert properties == {"sql_command_result": relation}


def test_ml_cache_commands_are_those_of_pyspark(ml_mock):
    # As PySpark Connect's _delete_ml_cache and _cleanup_ml_cache: the IDs that the server
    # deleted, and none of no IDs or of an error.
    spark, service = ml_mock
    client = spark.client
    service.execute_payload = _operator_info("a,b")
    assert client._delete_ml_cache(["a", "b"], evict_only=True) == ["a", "b"]
    (command,) = _commands(service)
    assert [ref.id for ref in command.ml_command.delete.obj_refs] == ["a", "b"]
    assert command.ml_command.delete.evict_only is True
    assert client._delete_ml_cache([]) == []
    assert len(service.execute_requests) == 1
    client._cleanup_ml_cache()
    assert _commands_but_deletes(service)[-1].ml_command.HasField("clean_cache")
    service.execute_failure = "the server failed"
    assert client._delete_ml_cache(["c"]) == []
    client._cleanup_ml_cache()


def test_ml_fit_and_transform_send_the_messages_of_pyspark(ml_mock):
    # As official PySpark 4.3.0: fit sends an MlCommand.Fit of the estimator, its params and
    # the relation of the dataset, of which the server returns the model, and the transform of
    # the model is an ml_relation of the dataset that later transformations take as input.
    from pyspark.ml.feature import CountVectorizer

    spark, service = ml_mock
    service.execute_payload = _operator_info("model-1")
    df = spark.range(1).selectExpr("array('a', 'b') AS words")
    estimator = CountVectorizer(vocabSize=5, minDF=0.5, inputCol="words", outputCol="counts")
    model = estimator.fit(df)

    (command,) = _commands(service)
    fit = command.ml_command.fit
    assert fit.estimator.name == "org.apache.spark.ml.feature.CountVectorizer"
    assert fit.estimator.uid == estimator.uid
    assert fit.estimator.type == pb2.MlOperator.OPERATOR_TYPE_ESTIMATOR
    assert dict(fit.params.params) == {
        "vocabSize": pb2.Expression.Literal(integer=5),
        "minDF": pb2.Expression.Literal(double=0.5),
        "inputCol": pb2.Expression.Literal(string="words"),
        "outputCol": pb2.Expression.Literal(string="counts"),
    }
    assert fit.dataset == _relation(df)
    assert fit.dataset.HasField("project")

    assert str(model._java_obj) == "model-1"
    assert model.uid == estimator.uid and model.getVocabSize() == 5

    predictions = model.transform(df)
    relation = _relation(predictions)
    transform = relation.ml_relation.transform
    assert transform.obj_ref.id == "model-1"
    assert transform.input == fit.dataset
    assert transform.params.params["vocabSize"] == pb2.Expression.Literal(integer=5)

    selected = _relation(predictions.select("counts"))
    assert selected.project.input == relation
    # A column of the transformed DataFrame refers to the plan ID of its relation.
    reference = predictions["counts"]._expr.to_plan(None).unresolved_attribute
    assert reference.plan_id == relation.common.plan_id
    assert len(service.execute_requests) == 1


def test_ml_transformers_send_the_relation_of_pyspark(ml_mock):
    # As official PySpark 4.3.0: a transformer of no model is an ml_relation of its name, its
    # uid and its params.
    from pyspark.ml.feature import Tokenizer

    spark, service = ml_mock
    df = spark.range(1).selectExpr("'a b' AS text")
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    transform = _relation(tokenizer.transform(df)).ml_relation.transform
    assert transform.transformer.name == "org.apache.spark.ml.feature.Tokenizer"
    assert transform.transformer.uid == tokenizer.uid
    assert transform.transformer.type == pb2.MlOperator.OPERATOR_TYPE_TRANSFORMER
    assert transform.params.params["inputCol"] == pb2.Expression.Literal(string="text")
    assert transform.input == _relation(df)
    assert service.execute_requests == []


def test_ml_dataframes_of_plans_keep_their_plans(ml_mock):
    # As PySpark Connect's DataFrame(plan, session): the relation of the plan, of which the
    # DataFrame and those of it keep the plan, as pyspark.ml keeps its models with the plans
    # of their transforms.
    from pyspark.sql.connect.dataframe import DataFrame
    from pyspark.sql.connect.plan import LogicalPlan

    spark, _ = ml_mock

    class SQLPlan(LogicalPlan):
        def __init__(self):
            super().__init__(None)

        def plan(self, session):
            plan = self._create_proto_relation()
            plan.sql.query = "SELECT 1 AS a"
            return plan

    plan = SQLPlan()
    alive = weakref.ref(plan)
    df = DataFrame(plan=plan, session=spark)
    relation = _relation(df)
    assert relation.sql.query == "SELECT 1 AS a"
    assert df["a"]._expr.to_plan(None).unresolved_attribute.plan_id == relation.common.plan_id
    derived = df.filter("a > 0")
    del plan, df
    gc.collect()
    assert alive() is not None
    assert _relation(derived).filter.input == relation
    del derived
    gc.collect()
    assert alive() is None


def test_public_dataframe_base_factory_and_native_dispatch(ml_mock):
    import inspect
    import pickle

    from pyspark import cloudpickle
    from pyspark.sql import DataFrame
    from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
    from pyspark.sql.connect.plan import LogicalPlan

    spark, service = ml_mock

    class SQLPlan(LogicalPlan):
        def __init__(self):
            super().__init__(None)

        def plan(self, session):
            relation = self._create_proto_relation()
            relation.sql.query = "SELECT 1 AS a"
            return relation

    frame = DataFrame(plan=SQLPlan(), session=spark)
    assert type(frame) is ConnectDataFrame
    assert isinstance(frame, DataFrame)
    assert issubclass(ConnectDataFrame, DataFrame)
    assert DataFrame.__module__ == "pyspark.sql.dataframe"
    assert ConnectDataFrame.__module__ == "pyspark.sql.connect.dataframe"
    assert DataFrame.__doc__ == ConnectDataFrame.__doc__
    assert inspect.signature(DataFrame) == inspect.signature(ConnectDataFrame)
    assert inspect.signature(DataFrame.select) == inspect.signature(ConnectDataFrame.select)
    assert DataFrame.select.__doc__ == ConnectDataFrame.select.__doc__
    assert {
        name for name in dir(ConnectDataFrame)
        if not name.startswith("_") and name != "is_cached"
    } <= set(dir(DataFrame))

    # Public-base unbound methods and properties dispatch to the real native frame;
    # ordinary frame calls keep the native implementation and result type.
    projected = DataFrame.select(frame, "a")
    assert type(projected) is ConnectDataFrame
    assert isinstance(projected, DataFrame)
    assert DataFrame.sparkSession.fget(frame) is spark
    assert DataFrame.sparkSession.fget.__name__ == "sparkSession"
    assert DataFrame.sparkSession.fget.__module__ == "pyspark.sql.dataframe"
    assert DataFrame.sparkSession.fget.__qualname__ == "DataFrame.sparkSession"
    assert repr(DataFrame.__getitem__(frame, "a")) == "Column<'a'>"
    restored = cloudpickle.loads(cloudpickle.dumps(frame))
    assert type(restored) is ConnectDataFrame
    assert _relation(restored).sql.query == "SELECT 1 AS a"
    assert pickle.loads(pickle.dumps(DataFrame)) is DataFrame
    assert pickle.loads(pickle.dumps(ConnectDataFrame)) is ConnectDataFrame

    init_calls = []

    class InheritedFactoryFrame(DataFrame):
        def __init__(self, plan, session):
            init_calls.append(self)

    # As PySpark's public factory: inherited __new__ returns a concrete frame, so
    # Python does not run the independent subclass's initializer.
    factory_frame = InheritedFactoryFrame(plan=SQLPlan(), session=spark)
    assert type(factory_frame) is ConnectDataFrame
    assert not isinstance(factory_frame, InheritedFactoryFrame)
    assert init_calls == []

    class NativeFrame(ConnectDataFrame):
        def __init__(self, plan, session):
            self.marker = "native-subclass-state"

    native_subclass = NativeFrame(plan=SQLPlan(), session=spark)
    assert type(native_subclass) is NativeFrame
    assert isinstance(native_subclass, DataFrame)
    assert native_subclass.marker == "native-subclass-state"
    assert type(native_subclass.select("a")) is ConnectDataFrame
    assert service.execute_requests == []


def test_public_dataframe_extensions_reach_native_frames_and_restore(ml_mock, monkeypatch):
    import inspect

    from pyspark.sql import DataFrame
    from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame

    spark, service = ml_mock
    frame = spark.range(1)
    original_public_count = vars(DataFrame)["count"]
    original_native_count = vars(ConnectDataFrame)["count"]
    original_public_schema = vars(DataFrame)["schema"]
    original_native_schema = vars(ConnectDataFrame)["schema"]
    original_native_session = vars(ConnectDataFrame)["_session"]
    missing = object()

    class LocalFrame(DataFrame):
        def __new__(cls):
            return object.__new__(cls)

    with monkeypatch.context() as patch:
        patch.setattr(DataFrame, "compatibility_helper", lambda self: 42, raising=False)
        assert frame.compatibility_helper() == 42
        assert LocalFrame().compatibility_helper() == 42
        with monkeypatch.context() as local_patch:
            local_patch.setattr(LocalFrame, "compatibility_helper", lambda self: 7)
            assert LocalFrame().compatibility_helper() == 7
            assert frame.compatibility_helper() == 42
        assert LocalFrame().compatibility_helper() == 42
    assert "compatibility_helper" not in vars(DataFrame)
    assert "compatibility_helper" not in vars(ConnectDataFrame)
    assert inspect.getattr_static(frame, "compatibility_helper", missing) is missing

    with monkeypatch.context() as patch:
        patch.setattr(DataFrame, "count", lambda self: 99)
        assert frame.count() == 99
        assert DataFrame.count(frame) == 99
    assert vars(DataFrame)["count"] is original_public_count
    assert vars(ConnectDataFrame)["count"] is original_native_count

    with monkeypatch.context() as patch:
        patch.delattr(DataFrame, "count")
        assert "count" not in vars(DataFrame)
        assert vars(ConnectDataFrame)["count"] is original_native_count
        assert frame.count.__self__ is frame
        assert frame.count.__name__ == "count"
    assert vars(DataFrame)["count"] is original_public_count
    assert vars(ConnectDataFrame)["count"] is original_native_count

    with monkeypatch.context() as patch:
        patch.setattr(DataFrame, "count", lambda self: 99)
        assert frame.count() == 99
        del DataFrame.count
        assert "count" not in vars(DataFrame)
        assert vars(ConnectDataFrame)["count"] is original_native_count
        assert frame.count.__self__ is frame
    assert vars(DataFrame)["count"] is original_public_count
    assert vars(ConnectDataFrame)["count"] is original_native_count

    native_attributes = dict(vars(ConnectDataFrame))
    with monkeypatch.context() as patch:
        patch.setattr(
            DataFrame,
            "__subclasshook__",
            classmethod(lambda cls, subclass: NotImplemented),
        )
        assert dict(vars(ConnectDataFrame)) == native_attributes
    assert dict(vars(ConnectDataFrame)) == native_attributes

    with monkeypatch.context() as patch:
        patch.setattr(DataFrame, "schema", property(lambda self: "custom-schema"))
        assert frame.schema == "custom-schema"
        assert LocalFrame().schema == "custom-schema"
    assert vars(DataFrame)["schema"] is original_public_schema
    assert vars(ConnectDataFrame)["schema"] is original_native_schema

    # Explicit public calls must follow concrete-class patches at call time.
    # Undoing or deleting a public override must preserve a concrete patch that
    # was already in place, rather than reinstating an import-time descriptor.
    with monkeypatch.context() as concrete_patch:
        concrete_count = lambda self: 73
        concrete_schema = property(lambda self: "concrete-schema")
        concrete_patch.setattr(ConnectDataFrame, "count", concrete_count)
        concrete_patch.setattr(ConnectDataFrame, "schema", concrete_schema)
        assert DataFrame.count(frame) == frame.count() == 73
        assert DataFrame.schema.fget(frame) == frame.schema == "concrete-schema"
        with monkeypatch.context() as public_patch:
            public_patch.setattr(DataFrame, "count", lambda self: 99)
            public_patch.setattr(DataFrame, "schema", property(lambda self: "public-schema"))
            assert DataFrame.count(frame) == frame.count() == 99
            assert DataFrame.schema.fget(frame) == frame.schema == "public-schema"
        assert vars(ConnectDataFrame)["count"] is concrete_count
        assert vars(ConnectDataFrame)["schema"] is concrete_schema
        assert DataFrame.count(frame) == 73
        assert DataFrame.schema.fget(frame) == "concrete-schema"
        with monkeypatch.context() as public_patch:
            public_patch.delattr(DataFrame, "count")
            public_patch.delattr(DataFrame, "schema")
            assert frame.count() == 73
            assert frame.schema == "concrete-schema"
        assert DataFrame.count(frame) == 73
        assert DataFrame.schema.fget(frame) == "concrete-schema"
    assert vars(ConnectDataFrame)["count"] is original_native_count
    assert vars(ConnectDataFrame)["schema"] is original_native_schema

    try:
        with monkeypatch.context() as public_patch:
            public_patch.setattr(DataFrame, "count", lambda self: 99)
            concrete_count = lambda self: 74
            # A concrete patch installed after the public override also survives
            # its undo. This assignment is owned by the surrounding finally.
            ConnectDataFrame.count = concrete_count
        assert vars(ConnectDataFrame)["count"] is concrete_count
        assert DataFrame.count(frame) == frame.count() == 74
    finally:
        ConnectDataFrame.count = original_native_count
    assert vars(DataFrame)["count"] is original_public_count

    try:
        with monkeypatch.context() as public_patch:
            public_patch.setattr(DataFrame, "count", lambda self: 99)
            concrete_count = lambda self: 75
            ConnectDataFrame.count = concrete_count
            public_patch.setattr(DataFrame, "count", lambda self: 100)
            assert frame.count() == 100
        assert vars(ConnectDataFrame)["count"] is concrete_count
        assert DataFrame.count(frame) == frame.count() == 75
    finally:
        ConnectDataFrame.count = original_native_count
    assert vars(DataFrame)["count"] is original_public_count

    # Added helpers obey the same ownership rule as initial members, including
    # deletion after a concrete patch and repeated public overrides.
    for repeat_public_override in (False, True):
        try:
            with monkeypatch.context() as public_patch:
                public_patch.setattr(
                    DataFrame, "compatibility_helper", lambda self: 1, raising=False
                )
                concrete_helper = lambda self: 76
                ConnectDataFrame.compatibility_helper = concrete_helper
                if repeat_public_override:
                    public_patch.setattr(DataFrame, "compatibility_helper", lambda self: 2)
                    assert frame.compatibility_helper() == 2
                public_patch.delattr(DataFrame, "compatibility_helper")
                assert "compatibility_helper" not in vars(DataFrame)
                assert vars(ConnectDataFrame)["compatibility_helper"] is concrete_helper
                assert frame.compatibility_helper() == 76
            assert "compatibility_helper" not in vars(DataFrame)
            assert vars(ConnectDataFrame)["compatibility_helper"] is concrete_helper
        finally:
            if "compatibility_helper" in vars(ConnectDataFrame):
                delattr(ConnectDataFrame, "compatibility_helper")

    # A public extension may temporarily shadow a native-only descriptor. Removing
    # that extension must recover the native getter rather than deleting it.
    assert "_session" not in vars(DataFrame)
    with monkeypatch.context() as patch:
        patch.setattr(DataFrame, "_session", property(lambda self: "custom-session"), raising=False)
        assert frame._session == "custom-session"
    assert "_session" not in vars(DataFrame)
    assert vars(ConnectDataFrame)["_session"] is original_native_session
    assert frame._session is spark

    def set_custom_value(self, value):
        self.custom_value = value

    with monkeypatch.context() as patch:
        patch.setattr(
            DataFrame,
            "compatibility_property",
            property(lambda self: self.custom_value, set_custom_value),
            raising=False,
        )
        frame.compatibility_property = 5
        assert frame.compatibility_property == 5
    assert "compatibility_property" not in vars(DataFrame)
    assert "compatibility_property" not in vars(ConnectDataFrame)
    assert service.execute_requests == []


def test_public_dataframe_wrappers_call_the_concrete_attributes_they_captured(
    ml_mock, monkeypatch
):
    # A wrapper of a public-base attribute often calls the attribute it captured before
    # the patch. That reaches the implementation that the wrapper replaced, as on master
    # and in PySpark, and not the wrapper again, which recursed without end.
    from pyspark.sql import DataFrame

    spark, service = ml_mock
    frame = spark.range(3)
    original_limit = vars(DataFrame)["limit"]
    original_session = vars(DataFrame)["sparkSession"]
    original_getattr = vars(DataFrame)["__getattr__"]
    calls = []

    def limit(self, num):
        calls.append(("limit", num))
        return original_limit(self, num)

    def get_attribute(self, name):
        calls.append(("__getattr__", name))
        return original_getattr(self, name)

    with monkeypatch.context() as patch:
        patch.setattr(DataFrame, "limit", limit)
        patch.setattr(
            DataFrame,
            "sparkSession",
            property(lambda self: ("wrapped", original_session.fget(self))),
        )
        patch.setattr(DataFrame, "__getattr__", get_attribute)
        assert isinstance(frame.limit(1), DataFrame)
        assert frame.sparkSession == ("wrapped", spark)
        assert repr(frame.missing_column) == repr(frame["missing_column"])
        # A wrapper of a wrapper reaches the concrete method through both.
        inner = DataFrame.limit
        patch.setattr(DataFrame, "limit", lambda self, num: ("outer", inner(self, num)))
        outer, limited = frame.limit(2)
        assert outer == "outer" and isinstance(limited, DataFrame)
        assert calls == [("limit", 1), ("__getattr__", "missing_column"), ("limit", 2)]
    assert vars(DataFrame)["limit"] is original_limit
    assert frame.sparkSession is spark
    assert service.execute_requests == []


def test_public_dataframe_attribute_dispatch_and_restore(ml_mock, monkeypatch):
    import inspect

    from pyspark.errors import PySparkAttributeError
    from pyspark.sql import DataFrame
    from pyspark.sql.column import Column
    from pyspark.sql.connect.column import Column as ConnectColumn
    from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame

    spark, service = ml_mock
    frame = spark.range(1)
    monkeypatch.delenv("PYSPARK_VALIDATE_COLUMN_NAME_LEGACY", raising=False)
    original_public_getattr = vars(DataFrame)["__getattr__"]
    original_native_lookup = vars(ConnectDataFrame)["__getattribute__"]
    original_native_attributes = dict(vars(ConnectDataFrame))
    native_resolver = vars(ConnectDataFrame)["_get_column_by_attribute"]
    getattr_signature = inspect.signature(DataFrame.__getattr__)
    assert getattr_signature.parameters["name"].annotation is str
    assert getattr_signature.return_annotation is Column
    assert Column is ConnectColumn
    assert "__getattr__" not in vars(ConnectDataFrame)
    assert "__getattr__" in dir(DataFrame)
    assert "_get_column_by_attribute" in dir(ConnectDataFrame)
    assert "_get_column_by_attribute" not in vars(DataFrame)
    assert repr(native_resolver(frame, "count")) == "Column<'count'>"
    assert repr(DataFrame.__getattr__(frame, "a")) == "Column<'a'>"
    # Explicit __getattr__ resolves a column even when a method has that name.
    assert repr(DataFrame.__getattr__(frame, "count")) == "Column<'count'>"
    for name, condition in (
        ("rdd", "JVM_ATTRIBUTE_NOT_SUPPORTED"),
        ("__missing_column__", "ATTRIBUTE_NOT_SUPPORTED"),
    ):
        with pytest.raises(PySparkAttributeError) as error:
            DataFrame.__getattr__(frame, name)
        assert error.value.getCondition() == condition
        assert error.value.getMessageParameters() == {"attr_name": name}
    with monkeypatch.context() as patch:
        patch.setenv("PYSPARK_VALIDATE_COLUMN_NAME_LEGACY", "1")
        with pytest.raises(PySparkAttributeError) as error:
            DataFrame.__getattr__(frame, "missing_column")
        assert error.value.getCondition() == "ATTRIBUTE_NOT_SUPPORTED"
        assert error.value.getMessageParameters() == {"attr_name": "missing_column"}

    delegated_calls = []

    class NativeFrame(ConnectDataFrame):
        def __getattr__(self, name):
            delegated_calls.append(name)
            return DataFrame.__getattr__(self, name)

    native_subclass = NativeFrame(plan=frame._plan, session=spark)
    assert repr(NativeFrame.__getattr__(native_subclass, "count")) == "Column<'count'>"
    assert delegated_calls == ["count"]
    forwarded_calls = []

    def custom_getattr(self, name):
        forwarded_calls.append((self, name))
        return ("custom", name)

    with monkeypatch.context() as patch:
        patch.setattr(DataFrame, "__getattr__", custom_getattr)
        assert vars(ConnectDataFrame)["__getattribute__"] is object.__getattribute__
        assert frame.missing_column == ("custom", "missing_column")
        assert DataFrame.__getattr__(frame, "count") == ("custom", "count")
        assert len(forwarded_calls) == 2
        assert all(item[0] is frame for item in forwarded_calls)
        assert [item[1] for item in forwarded_calls] == ["missing_column", "count"]
        assert frame.count.__self__ is frame
        patch.setattr(DataFrame, "__getattr__", lambda self, name: ("second", name))
        assert frame.missing_column == ("second", "missing_column")
    assert vars(DataFrame)["__getattr__"] is original_public_getattr
    assert "__getattr__" not in vars(ConnectDataFrame)
    assert vars(ConnectDataFrame)["__getattribute__"] is original_native_lookup
    assert dict(vars(ConnectDataFrame)) == original_native_attributes
    assert repr(frame.a) == "Column<'a'>"
    assert frame.count.__self__ is frame
    assert repr(DataFrame.__getattr__(frame, "count")) == "Column<'count'>"
    with pytest.raises(PySparkAttributeError) as error:
        frame.__missing_column__
    assert error.value.getCondition() == "ATTRIBUTE_NOT_SUPPORTED"
    assert error.value.getMessageParameters() == {"attr_name": "__missing_column__"}

    with monkeypatch.context() as patch:
        patch.setattr(DataFrame, "__getattr__", custom_getattr)
        patch.delattr(DataFrame, "__getattr__")
        assert "__getattr__" not in vars(ConnectDataFrame)
        assert vars(ConnectDataFrame)["__getattribute__"] is original_native_lookup
        assert repr(frame.a) == "Column<'a'>"
        assert frame.count.__self__ is frame
    assert vars(DataFrame)["__getattr__"] is original_public_getattr
    assert "__getattr__" not in vars(ConnectDataFrame)
    assert vars(ConnectDataFrame)["__getattribute__"] is original_native_lookup
    assert dict(vars(ConnectDataFrame)) == original_native_attributes
    assert repr(frame.a) == "Column<'a'>"
    assert service.execute_requests == []


@pytest.mark.parametrize("lookup_first", [False, True])
@pytest.mark.parametrize("undo_lookup_first", [False, True])
def test_public_dataframe_lookup_protocol_is_local_during_hook_overrides(
    ml_mock, monkeypatch, lookup_first, undo_lookup_first
):
    from pyspark.errors import PySparkNotImplementedError
    from pyspark.sql import DataFrame
    from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame

    spark, service = ml_mock
    frame = spark.range(1)
    monkeypatch.delenv("PYSPARK_VALIDATE_COLUMN_NAME_LEGACY", raising=False)
    original_native_attributes = dict(vars(ConnectDataFrame))
    original_public_attributes = dict(vars(DataFrame))
    original_native_lookup = vars(ConnectDataFrame)["__getattribute__"]
    local_lookup_calls, hook_calls = [], []

    class BeforeFrame(DataFrame):
        def __new__(cls):
            return object.__new__(cls)

    before_frame = BeforeFrame()

    def local_lookup(self, name):
        if name == "local_marker":
            local_lookup_calls.append(id(self))
            return "local-value"
        return object.__getattribute__(self, name)

    def forwarded_hook(self, name):
        hook_calls.append((id(self), name))
        return ("forwarded", name)

    with monkeypatch.context() as lookup_patch, monkeypatch.context() as hook_patch:
        install_order = ("lookup", "hook") if lookup_first else ("hook", "lookup")
        for protocol in install_order:
            native_before = dict(vars(ConnectDataFrame))
            if protocol == "lookup":
                lookup_patch.setattr(DataFrame, "__getattribute__", local_lookup)
                assert dict(vars(ConnectDataFrame)) == native_before
            else:
                hook_patch.setattr(DataFrame, "__getattr__", forwarded_hook)
                assert vars(ConnectDataFrame)["__getattribute__"] is object.__getattribute__

        class DuringFrame(BeforeFrame):
            pass

        during_frame = DuringFrame()
        for independent in (before_frame, during_frame):
            assert independent.local_marker == "local-value"
            assert independent.missing_value == ("forwarded", "missing_value")
        assert local_lookup_calls == [id(before_frame), id(during_frame)]
        assert frame.local_marker == ("forwarded", "local_marker")
        assert local_lookup_calls == [id(before_frame), id(during_frame)]
        assert frame.missing_value == ("forwarded", "missing_value")
        assert hook_calls[-1] == (id(frame), "missing_value")
        assert frame.count.__self__ is frame

        undo_order = ("lookup", "hook") if undo_lookup_first else ("hook", "lookup")
        for protocol in undo_order:
            native_before = dict(vars(ConnectDataFrame))
            if protocol == "lookup":
                lookup_patch.undo()
                assert dict(vars(ConnectDataFrame)) == native_before
            else:
                hook_patch.undo()
                assert dict(vars(ConnectDataFrame)) == original_native_attributes
                assert vars(ConnectDataFrame)["__getattribute__"] is original_native_lookup
                assert "__getattr__" not in vars(ConnectDataFrame)
                assert repr(frame.a) == "Column<'a'>"
                assert frame.count.__self__ is frame

    assert dict(vars(DataFrame)) == original_public_attributes
    assert dict(vars(ConnectDataFrame)) == original_native_attributes
    assert repr(frame.a) == "Column<'a'>"
    for independent in (before_frame, during_frame):
        with pytest.raises(PySparkNotImplementedError) as error:
            independent.local_marker
        assert error.value.getCondition() == "NOT_IMPLEMENTED"
        assert error.value.getMessageParameters() == {"feature": "DataFrame.__getattr__"}
    assert service.execute_requests == []


def test_public_dataframe_state_protocols_apply_only_to_python_subclasses(ml_mock, monkeypatch):
    from pyspark.sql import DataFrame
    from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame

    spark, service = ml_mock
    frame = spark.range(1)
    original_native_attributes = dict(vars(ConnectDataFrame))
    original_public_attributes = dict(vars(DataFrame))
    set_calls, delete_calls = [], []

    class BeforeFrame(DataFrame):
        def __new__(cls):
            return object.__new__(cls)

    before_frame = BeforeFrame()

    def local_setattr(self, name, value):
        set_calls.append((id(self), name, value))
        object.__setattr__(self, name, value)

    def local_delattr(self, name):
        delete_calls.append((id(self), name))
        object.__delattr__(self, name)

    with monkeypatch.context() as patch:
        patch.setattr(DataFrame, "__setattr__", local_setattr)
        patch.setattr(DataFrame, "__delattr__", local_delattr)
        assert dict(vars(ConnectDataFrame)) == original_native_attributes

        class DuringFrame(BeforeFrame):
            pass

        during_frame = DuringFrame()
        for independent in (before_frame, during_frame):
            independent.value = 8
            assert independent.value == 8
            del independent.value
            assert "value" not in vars(independent)
        assert set_calls == [(id(before_frame), "value", 8), (id(during_frame), "value", 8)]
        assert delete_calls == [(id(before_frame), "value"), (id(during_frame), "value")]
        frame._protocol_state = 9
        assert frame._protocol_state == 9
        del frame._protocol_state
        assert "_protocol_state" not in vars(frame)
        assert set_calls == [(id(before_frame), "value", 8), (id(during_frame), "value", 8)]
        assert delete_calls == [(id(before_frame), "value"), (id(during_frame), "value")]
        assert dict(vars(ConnectDataFrame)) == original_native_attributes

    assert dict(vars(DataFrame)) == original_public_attributes
    assert dict(vars(ConnectDataFrame)) == original_native_attributes
    before_frame.value = 10
    del before_frame.value
    assert len(set_calls) == len(delete_calls) == 2
    assert service.execute_requests == []


def test_python_dataframe_subclasses_can_own_their_state():
    from pyspark.errors import PySparkNotImplementedError
    from pyspark.sql import DataFrame
    from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame

    class MemoryFrame(DataFrame):
        def __new__(cls, values):
            return object.__new__(cls)

        def __init__(self, values):
            self.values = list(values)
            self._session = "own-session"
            self._plan = "own-plan"
            self.is_cached = False

        def collect(self):
            return list(self.values)

        def count(self):
            return len(self.collect())

    class CountedFrame(MemoryFrame):
        def count(self):
            return super().count() + 1

    frame = CountedFrame([1, 2])
    assert isinstance(frame, DataFrame)
    assert not isinstance(frame, ConnectDataFrame)
    assert issubclass(CountedFrame, DataFrame)
    assert frame.collect() == [1, 2]
    assert frame.count() == 3
    assert vars(frame) == {
        "values": [1, 2], "_session": "own-session", "_plan": "own-plan", "is_cached": False
    }
    assert frame._session == "own-session" and frame._plan == "own-plan"
    assert frame.is_cached is False
    frame.is_cached = True
    assert frame.is_cached is True
    assert DataFrame.__annotations__["is_cached"] is bool
    assert "is_cached" not in vars(DataFrame)
    # Official public properties have empty bodies; independent subclasses can
    # override them, and otherwise read None without disrupting hasattr.
    assert frame.sparkSession is None
    assert frame.schema is None
    assert DataFrame.columns.fget(frame) is None
    assert hasattr(frame, "schema")
    # Public-base methods, including repr and dir, use PySpark's dispatch contract:
    # native operations cannot operate on an independent Python data model.
    for operation, name in (
        (lambda: DataFrame.count(frame), "count"),
        (lambda: repr(frame), "__repr__"),
        (lambda: dir(frame), "__dir__"),
        (lambda: frame.missing_attribute, "__getattr__"),
        (lambda: hasattr(frame, "missing_attribute"), "__getattr__"),
    ):
        with pytest.raises(PySparkNotImplementedError) as error:
            operation()
        assert error.value.getCondition() == "NOT_IMPLEMENTED"
        assert error.value.getMessageParameters() == {"feature": f"DataFrame.{name}"}


@pytest.mark.parametrize("name,args,kwargs", [
    ("checkpoint", (False,), {}),
    ("localCheckpoint", (), {"eager": False, "storageLevel": "own-storage"}),
    ("lateralJoin", ("other",), {"on": "condition", "how": "join-type"}),
    (
        "nearestByJoin", ("other", "ranking", 3, "exact", "distance"),
        {"joinType": "left"},
    ),
    ("toPandas", (), {}),
    ("asTable", (), {}),
    ("scalar", (), {}),
    ("exists", (), {}),
])
def test_python_dataframe_undecorated_methods_keep_official_fallbacks(name, args, kwargs):
    from pyspark.sql import DataFrame

    class MemoryFrame(DataFrame):
        def __new__(cls):
            return object.__new__(cls)

    frame = MemoryFrame()
    method = getattr(frame, name)
    assert method(*args, **kwargs) is None
    assert getattr(DataFrame, name)(frame, *args, **kwargs) is None
    if name in {"checkpoint", "localCheckpoint"}:
        assert method() is None
    # Official empty methods enforce arity before returning None, without
    # validating or converting the supplied values.
    with pytest.raises(TypeError):
        method(*args, **kwargs, unexpected=True)
    with pytest.raises(TypeError):
        method(*args, *([None] * 8), **kwargs)
    if name in {"lateralJoin", "nearestByJoin"}:
        with pytest.raises(TypeError):
            method()
    if name == "nearestByJoin":
        with pytest.raises(TypeError):
            method(*args, "left")


def test_python_dataframe_arrow_stream_uses_generic_helper(monkeypatch):
    from pyspark.sql import DataFrame, interchange

    class MemoryFrame(DataFrame):
        def __new__(cls):
            return object.__new__(cls)

    frame = MemoryFrame()
    requested_schema = object()
    stream = object()
    calls = []

    class Streamer:
        def __init__(self, dataframe):
            assert dataframe is frame

        def __arrow_c_stream__(self, schema):
            calls.append(schema)
            return stream

    monkeypatch.setattr(interchange, "SparkArrowCStreamer", Streamer)
    assert frame.__arrow_c_stream__() is stream
    assert frame.__arrow_c_stream__(requested_schema) is stream
    assert DataFrame.__arrow_c_stream__(frame, requested_schema=requested_schema) is stream
    assert calls == [None, requested_schema, requested_schema]
    with pytest.raises(TypeError):
        frame.__arrow_c_stream__(requested_schema, requested_schema)
    with pytest.raises(TypeError):
        frame.__arrow_c_stream__(unexpected=True)
    with pytest.raises(TypeError):
        frame.__arrow_c_stream__(requested_schema, requested_schema=requested_schema)
    assert calls == [None, requested_schema, requested_schema]


def test_public_dataframe_fallback_methods_still_dispatch_to_native(ml_mock, monkeypatch):
    from pyspark.sql import DataFrame
    from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame

    spark, service = ml_mock
    frame = spark.range(1)
    marker = object()
    result = object()
    calls = []

    def implementation(self, value, *, option):
        calls.append((self, value, option))
        return result

    for name in (
        "checkpoint", "localCheckpoint", "lateralJoin", "nearestByJoin", "toPandas",
        "asTable", "scalar", "exists", "__arrow_c_stream__",
    ):
        with monkeypatch.context() as patch:
            patch.setattr(ConnectDataFrame, name, implementation)
            # A native receiver uses the current concrete implementation's
            # signature, even when the independent fallback has another arity.
            assert getattr(DataFrame, name)(frame, marker, option=marker) is result
            assert calls[-1] == (frame, marker, marker)
    assert len(calls) == 9
    assert service.execute_requests == []


def test_local_dataframe_subclasses_cloudpickle_with_importable_metaclass():
    from abc import ABCMeta

    from pyspark import cloudpickle
    from pyspark.sql import DataFrame

    class MemoryFrame(DataFrame):
        def __new__(cls, values):
            return object.__new__(cls)

        def __init__(self, values):
            self.values = list(values)

        def count(self):
            return len(self.values)

    class CountedFrame(MemoryFrame):
        def count(self):
            return super().count() + 1

    def count_values(values):
        return CountedFrame(values).count()

    assert type(MemoryFrame) is ABCMeta
    assert type(CountedFrame) is ABCMeta
    payload = cloudpickle.dumps((CountedFrame, count_values))
    assert b"_DataFrameMeta" not in payload
    restored_class, restored_function = cloudpickle.loads(payload)
    assert issubclass(restored_class, DataFrame)
    assert restored_class([1, 2]).count() == 3
    assert restored_function([1, 2, 3]) == 4

    class UserMeta(type(DataFrame)):
        pass

    class UserFrame(DataFrame, metaclass=UserMeta):
        def __new__(cls):
            return object.__new__(cls)

    assert type(UserFrame) is UserMeta
    assert isinstance(UserFrame(), DataFrame)


def test_python_dataframe_subclass_with_official_mock_pipeline():
    # This is the no-server behavior of official PipelineTestsMixin.test_pipeline,
    # using its actual vendored MockDataset, estimators and transformers.
    from pyspark.errors import PySparkNotImplementedError
    from pyspark.ml import Pipeline
    from pyspark.sql import DataFrame
    from pyspark.testing.mlutils import MockDataset, MockEstimator, MockTransformer

    dataset = MockDataset()
    assert isinstance(dataset, DataFrame)
    with pytest.raises(PySparkNotImplementedError) as error:
        hasattr(dataset, "missing_attribute")
    assert error.value.getCondition() == "NOT_IMPLEMENTED"
    assert error.value.getMessageParameters() == {"feature": "DataFrame.__getattr__"}
    estimator0, transformer1 = MockEstimator(), MockTransformer()
    estimator2, transformer3 = MockEstimator(), MockTransformer()
    model = Pipeline(stages=[estimator0, transformer1, estimator2, transformer3]).fit(
        dataset, {estimator0.fake: 0, transformer1.fake: 1}
    )
    model0, transformer1, model2, transformer3 = model.stages
    assert model0.dataset_index == 0 and model0.getFake() == 0
    assert transformer1.dataset_index == 1 and transformer1.getFake() == 1
    assert dataset.index == 2
    assert model2.dataset_index is None and transformer3.dataset_index is None
    assert model.transform(dataset) is dataset
    assert [stage.dataset_index for stage in model.stages] == [2, 3, 4, 5]
    assert dataset.index == 6


def test_python_dataframe_subclass_with_official_identity_pipeline():
    from pyspark.ml import Pipeline
    from pyspark.testing.mlutils import MockDataset

    dataset = MockDataset()
    assert Pipeline(stages=[]).fit(dataset).transform(dataset) is dataset
    assert dataset.index == 0
    with pytest.raises(KeyError):
        Pipeline().fit(dataset)


def test_ml_logical_plan_ids_as_pyspark():
    # As PySpark Connect's LogicalPlan: a plan ID of each plan, and another of a plan with
    # references, which _plan_id is.
    from pyspark.sql.connect.plan import LogicalPlan

    first, second = LogicalPlan(None), LogicalPlan(None)
    assert second._plan_id == first._plan_id + 1
    assert first._create_proto_relation().common.plan_id == first._plan_id
    assert first.observations == {}
    with_references = LogicalPlan(first, references=[second])
    assert with_references._plan_id != with_references._root_plan_id
    assert with_references._create_proto_relation().common.plan_id == with_references._root_plan_id
    assert first.plan(None) is None and first.command(None) is None

    class Child(LogicalPlan):
        def plan(self, session):
            plan = self._create_proto_relation()
            plan.range.end = 1
            return plan

    child = Child(None)
    assert child._verify(None)
    root = child.plan(None)
    assert child._with_relations(root, None) is root
    parent = LogicalPlan(child, references=[child])
    relations = parent._with_relations(root, None)
    assert relations.common.plan_id == parent._plan_id
    assert relations.with_relations.root == root
    assert list(relations.with_relations.references) == [root]


def test_ml_column_expressions_render_as_pyspark():
    # As PySpark Connect's Column._expr: of which pyspark.ml.stat names the metrics of
    # Summarizer, the expression that Column<'...'> shows.
    from pyspark.ml.stat import Summarizer
    from pyspark.sql import functions as F

    for column in [F.col("a"), F.col("a") + 1, F.lit("x").alias("y"), F.max("b")]:
        assert str(column._expr) == repr(column._expr) == repr(column)[len("Column<'") : -len("'>")]
    assert str((F.col("a") + 1)._expr) == "+(a, 1)"
    assert F.col("a")._expr.to_plan(None).unresolved_attribute.unparsed_identifier == "a"
    assert repr(Summarizer.mean(F.col("features"))) == (
        "Column<'aggregate_metrics(array(mean), features, 1.0)['mean'] AS mean(features)'>"
    )


def test_ml_modules_import_as_pyspark():
    # The modules of pyspark.ml, vendored from Apache Spark v4.3.0, import on this client.
    import importlib

    for name in [
        "pyspark.ml",
        "pyspark.ml.base",
        "pyspark.ml.classification",
        "pyspark.ml.clustering",
        "pyspark.ml.evaluation",
        "pyspark.ml.feature",
        "pyspark.ml.fpm",
        "pyspark.ml.image",
        "pyspark.ml.model_cache",
        "pyspark.ml.pipeline",
        "pyspark.ml.recommendation",
        "pyspark.ml.regression",
        "pyspark.ml.stat",
        "pyspark.ml.tree",
        "pyspark.ml.tuning",
        "pyspark.ml.wrapper",
        "pyspark.ml.connect.proto",
        "pyspark.ml.connect.serialize",
        "pyspark.ml.connect.readwrite",
        "pyspark.ml.connect.summarizer",
        "pyspark.sql.connect.expressions",
        "pyspark.sql.connect.proto.ml_pb2",
    ]:
        assert importlib.import_module(name) is not None
    import pyspark.ml as ml

    assert {"Pipeline", "Estimator", "Transformer", "classification", "feature"} <= set(ml.__all__)


def test_ml_modules_import_without_protobuf():
    # Without the protobuf package, pyspark.ml and the connect modules import, of a progress
    # response of the transport, and a message of the generated modules raises an
    # AttributeError that names protobuf.
    script = textwrap.dedent(
        """
        import importlib.abc, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "google.protobuf" or name.startswith("google.protobuf."):
                    raise ImportError("no protobuf")

        sys.meta_path.insert(0, Block())
        import pyspark.ml
        from pyspark.ml.classification import LogisticRegression
        import pyspark.sql.connect.expressions, pyspark.sql.connect.plan
        import pyspark.sql.connect.proto as proto
        assert LogisticRegression(maxIter=3).getMaxIter() == 3
        assert hasattr(proto, "ExecutePlanResponse") and not hasattr(proto, "Relation")
        try:
            proto.Relation
        except AttributeError as error:
            print(error)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "'Relation'" in result.stdout and "protobuf >= 6.33.5" in result.stdout


def test_session_hooks_rewrite_the_requests_of_executions_as_pyspark(_ml_mock_server):
    # As official PySpark 4.3.0's SparkSession.Hook: a session that the builder creates calls
    # each factory of _registerHook with itself, and the on_execute_plan of each hook with each
    # ExecutePlanRequest before it sends it, which keeps the operation ID of the request.
    _, service = _ml_mock_server
    service.reset()
    created = []

    class TagHook(SparkSession.Hook):
        def __init__(self, session, tag):
            self.session = session
            self.tag = tag
            self.operation_ids = []

        def on_execute_plan(self, request):
            assert isinstance(request, pb2.ExecutePlanRequest)
            self.operation_ids.append(request.operation_id)
            request.tags.append(self.tag)
            request.operation_id = "replaced"
            return request

    def factory(tag):
        def make(session):
            hook = TagHook(session, tag)
            created.append(hook)
            return hook

        return make

    builder = SparkSession.builder.remote(service.remote)
    spark = builder._registerHook(factory("first"))._registerHook(factory("second")).create()
    try:
        assert [hook.session for hook in created] == [spark, spark]
        command = pb2.Command()
        command.ml_command.clean_cache.SetInParent()
        spark.client.execute_command(command)
        request = _last_request_but_deletes(service)
        assert list(request.tags)[-2:] == ["first", "second"]
        assert request.operation_id == created[0].operation_ids[0]
        assert created[1].operation_ids == created[0].operation_ids
        assert request.plan.command == command

        # A session got rather than created, the active or the default one, calls no
        # factories.
        got = SparkSession.builder.remote(service.remote)._registerHook(factory("third"))
        assert not got.getOrCreate().is_stopped and len(created) == 2
    finally:
        spark.stop()

    assert SparkSession.Hook().on_execute_plan(request) is request


def test_session_hooks_raise_their_errors_as_pyspark(_ml_mock_server):
    # As official PySpark 4.3.0: the error of a hook is raised by the execution that calls it,
    # as it is.
    _, service = _ml_mock_server
    service.reset()

    class Failing(SparkSession.Hook):
        def on_execute_plan(self, request):
            raise ValueError("the hook failed")

    spark = SparkSession.builder.remote(service.remote)._registerHook(lambda s: Failing()).create()
    try:
        command = pb2.Command()
        command.ml_command.clean_cache.SetInParent()
        with pytest.raises(ValueError, match="the hook failed"):
            spark.client.execute_command(command)
        assert service.execute_requests == []
    finally:
        spark.stop()


def test_ml_none_params_send_null_and_empty_lists_raise_at_the_command_as_pyspark(ml_mock):
    # As PySpark's Params._set: None is stored and sent as a null literal, and a param that is
    # not given is not sent. A list without elements is accepted by the constructor and the
    # setters, and PySpark raises CANNOT_INFER_ARRAY_ELEMENT_TYPE when its serialization sends
    # the params of a fit, transform or evaluate, before any request.
    import pyspark.ml.connect as connect

    spark, service = ml_mock
    df = spark.range(1)
    models = []

    sent = _native_sent(spark, service, connect.RandomForestClassifier(bootstrap=None), models)
    assert sent.keys() == {"bootstrap"} and sent["bootstrap"][0] == "null"
    assert "bootstrap" not in _native_sent(
        spark, service, connect.RandomForestClassifier(numTrees=3), models
    )

    assembler = connect.VectorAssembler(inputCols=[], outputCol="f")
    before = len(service.execute_requests)
    with pytest.raises(PySparkTypeError, match="CANNOT_INFER_ARRAY_ELEMENT_TYPE") as error:
        assembler.transform(df)
    assert error.value.getCondition() == "CANNOT_INFER_ARRAY_ELEMENT_TYPE"
    with pytest.raises(PySparkTypeError, match="CANNOT_INFER_ARRAY_ELEMENT_TYPE"):
        connect.RandomForestClassifier().setThresholds([]).fit(df)
    assert len(service.execute_requests) == before

    # Elements given again make the list sendable.
    assert assembler.setInputCols(["a", "b"]) is assembler
    _, params = _native_request(spark, service, assembler, models)
    expected = 'array { element_type { %s } elements { string: "a" } elements { string: "b" } }'
    assert params.params["inputCols"] == text_format.Parse(
        expected % _UTF8, pb2.Expression.Literal()
    )


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_random_forest_regressor_native_parameters_match_official_fixture(ml_mock, use_setters):
    import base64
    import inspect
    import json
    from pathlib import Path

    from pyspark.ml.connect import RandomForestRegressor as exported
    from pyspark.ml.connect.regression import RandomForestRegressor

    assert exported is RandomForestRegressor
    defaults = dict(featuresCol="features", labelCol="label", predictionCol="prediction",
                    maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0,
                    maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10,
                    impurity="variance", subsamplingRate=1.0, seed=None, numTrees=20,
                    featureSubsetStrategy="auto", leafCol="", minWeightFractionPerNode=0.0,
                    weightCol=None, bootstrap=True)
    signature = inspect.signature(RandomForestRegressor)
    assert list(signature.parameters) == list(defaults)
    for name, default in defaults.items():
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default == default
    fixture = json.loads((Path(__file__).resolve().parents[2] / "tests/golden/random_forest_regressor.json").read_text())
    spark, service = ml_mock
    models = []
    for case in fixture["cases"]:
        values = dict(featuresCol="input", labelCol="target", predictionCol="value", maxDepth=3,
                      maxBins=16, minInstancesPerNode=2, minInfoGain=0.01, maxMemoryInMB=64,
                      cacheNodeIds=True, checkpointInterval=-1, impurity="variance", numTrees=7,
                      featureSubsetStrategy="all", seed=case["seed"], subsamplingRate=0.75,
                      leafCol="leaves", minWeightFractionPerNode=0.1, weightCol="weight", bootstrap=False)
        if use_setters:
            estimator = RandomForestRegressor()
            for name, value in values.items():
                assert getattr(estimator,"set"+name[0].upper()+name[1:])(value) is estimator
        else:
            estimator = RandomForestRegressor(**values)
        operator, params = _native_request(spark, service, estimator, models)
        assert operator.name == "org.apache.spark.ml.regression.RandomForestRegressor"
        assert params == pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_decision_tree_classifier_native_parameters_match_official_fixture(ml_mock, use_setters):
    import base64
    import inspect
    import json
    from pathlib import Path

    import pyspark._pyspark as native
    from pyspark.ml.connect import DecisionTreeClassifier as exported
    from pyspark.ml.connect.classification import DecisionTreeClassifier

    assert exported is DecisionTreeClassifier is native.DecisionTreeClassifier
    with pytest.raises(TypeError, match="unexpected keyword argument 'thresholds'"):
        DecisionTreeClassifier(thresholds=[0.25, 0.75])
    defaults = dict(featuresCol="features", labelCol="label", predictionCol="prediction",
                    probabilityCol="probability", rawPredictionCol="rawPrediction",
                    maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0,
                    pruneTree=True, maxMemoryInMB=256, cacheNodeIds=False,
                    checkpointInterval=10, impurity="gini", seed=None, weightCol=None,
                    leafCol="", minWeightFractionPerNode=0.0)
    signature = inspect.signature(DecisionTreeClassifier)
    assert list(signature.parameters) == list(defaults)
    for name, default in defaults.items():
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default == default
    fixture = json.loads((Path(__file__).resolve().parents[2] / "tests/golden/decision_tree_classifier.json").read_text())
    spark, service = ml_mock
    models = []
    for case in fixture["cases"]:
        values = dict(featuresCol="input", labelCol="target", predictionCol="class",
                      probabilityCol="probs", rawPredictionCol="counts", maxDepth=3,
                      maxBins=16, minInstancesPerNode=2, minInfoGain=0.01, pruneTree=False,
                      maxMemoryInMB=64, cacheNodeIds=True, checkpointInterval=-1,
                      impurity="entropy", seed=case["seed"], weightCol="weight",
                      leafCol="leaves", minWeightFractionPerNode=0.1)
        if use_setters:
            estimator = DecisionTreeClassifier()
            for name, value in values.items():
                assert getattr(estimator,"set"+name[0].upper()+name[1:])(value) is estimator
        else:
            estimator = DecisionTreeClassifier(**values)
        assert estimator.setThresholds([0.25,0.75]) is estimator
        operator, params = _native_request(spark, service, estimator, models)
        assert operator.name == "org.apache.spark.ml.classification.DecisionTreeClassifier"
        assert params == pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_decision_tree_regressor_native_parameters_match_official_fixture(ml_mock, use_setters):
    import base64
    import inspect
    import json
    from pathlib import Path

    import pyspark._pyspark as native
    from pyspark.ml.connect import DecisionTreeRegressor as exported
    from pyspark.ml.connect.regression import DecisionTreeRegressor

    assert exported is DecisionTreeRegressor is native.DecisionTreeRegressor
    defaults = dict(featuresCol="features", labelCol="label", predictionCol="prediction",
                    maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0,
                    maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10,
                    impurity="variance", seed=None, varianceCol=None, weightCol=None,
                    leafCol="", minWeightFractionPerNode=0.0)
    signature = inspect.signature(DecisionTreeRegressor)
    assert list(signature.parameters) == list(defaults)
    for name, default in defaults.items():
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default == default
    fixture = json.loads((Path(__file__).resolve().parents[2] / "tests/golden/decision_tree_regressor.json").read_text())
    spark, service = ml_mock
    models = []
    for case in fixture["cases"]:
        values = dict(featuresCol="input", labelCol="target", predictionCol="value", maxDepth=3,
                      maxBins=16, minInstancesPerNode=2, minInfoGain=0.01, maxMemoryInMB=64,
                      cacheNodeIds=True, checkpointInterval=-1, impurity="variance",
                      seed=case["seed"], varianceCol="uncertainty", weightCol="weight",
                      leafCol="leaves", minWeightFractionPerNode=0.1)
        if use_setters:
            estimator = DecisionTreeRegressor()
            for name, value in values.items():
                assert getattr(estimator,"set"+name[0].upper()+name[1:])(value) is estimator
        else:
            estimator = DecisionTreeRegressor(**values)
        operator, params = _native_request(spark, service, estimator, models)
        assert operator.name == "org.apache.spark.ml.regression.DecisionTreeRegressor"
        assert params == pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_gbt_classifier_native_parameters_match_official_fixture(ml_mock, use_setters):
    import base64
    import inspect
    import json
    from pathlib import Path

    import pyspark._pyspark as native
    from pyspark.ml.connect import GBTClassifier as exported
    from pyspark.ml.connect.classification import GBTClassifier

    assert exported is GBTClassifier is native.GBTClassifier
    defaults = dict(featuresCol="features", labelCol="label", predictionCol="prediction",
                    maxDepth=5, maxBins=32, minInstancesPerNode=1, minInfoGain=0.0,
                    maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, lossType="logistic",
                    maxIter=20, stepSize=0.1, seed=None, subsamplingRate=1.0, impurity="variance",
                    featureSubsetStrategy="all", validationTol=0.01, validationIndicatorCol=None,
                    leafCol="", minWeightFractionPerNode=0.0, weightCol=None)
    signature = inspect.signature(GBTClassifier)
    assert list(signature.parameters) == list(defaults)
    for name, default in defaults.items():
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default == default
    for name in ("probabilityCol", "rawPredictionCol", "thresholds"):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            GBTClassifier(**{name: None})
    assert not hasattr(GBTClassifier(), "setValidationTol")
    fixture = json.loads((Path(__file__).resolve().parents[2] / "tests/golden/gbt_classifier.json").read_text())
    spark, service = ml_mock
    models = []
    for case in fixture["cases"]:
        values = dict(featuresCol="input", labelCol="target", predictionCol="class", maxDepth=3,
                      maxBins=16, minInstancesPerNode=2, minInfoGain=0.01, maxMemoryInMB=64,
                      cacheNodeIds=True, checkpointInterval=-1, lossType="logistic", maxIter=4,
                      stepSize=0.2, seed=case["seed"], subsamplingRate=0.75, impurity="variance",
                      featureSubsetStrategy="all", validationIndicatorCol="validation",
                      leafCol="leaves", minWeightFractionPerNode=0.1, weightCol="weight")
        if use_setters:
            estimator = GBTClassifier(validationTol=0.05)
            for name, value in values.items():
                assert getattr(estimator,"set"+name[0].upper()+name[1:])(value) is estimator
        else:
            estimator = GBTClassifier(**values, validationTol=0.05)
        for name, value in [("ProbabilityCol","probs"),("RawPredictionCol","margin"),("Thresholds",[0.25,0.75])]:
            assert getattr(estimator,"set"+name)(value) is estimator
        operator, params = _native_request(spark, service, estimator, models)
        assert operator.name == "org.apache.spark.ml.classification.GBTClassifier"
        assert params == pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))
    for value in [np.float64(0.05), 1, None]:
        sent = _native_sent(spark, service, GBTClassifier(validationTol=value), models)
        assert sent["validationTol"][0] == ("null" if value is None else "double")
    with pytest.raises(TypeError, match="Invalid param value"):
        GBTClassifier(validationTol="invalid")


@pytest.mark.parametrize("param", ["probabilityCol", "rawPredictionCol"])
def test_gbt_classifier_setter_only_columns_convert_as_pyspark(ml_mock, param):
    from pyspark.ml.connect.classification import GBTClassifier

    spark, service = ml_mock
    models = []
    setter = "set" + param[0].upper() + param[1:]
    for value, expected in _NATIVE_CONVERTED["str"]:
        estimator = GBTClassifier()
        assert getattr(estimator, setter)(value) is estimator
        assert _native_sent(spark, service, estimator, models) == {param: expected}
    estimator = GBTClassifier()
    assert getattr(estimator, setter)(None) is estimator
    assert _native_sent(spark, service, estimator, models)[param][0] == "null"
    for value, message in _NATIVE_REJECTED["str"]:
        with pytest.raises(TypeError) as error:
            getattr(GBTClassifier(), setter)(value)
        assert str(error.value) == f'Invalid param value given for param "{param}". {message}'
