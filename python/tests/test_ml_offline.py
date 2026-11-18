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
import logging
import subprocess
import sys
import textwrap
import weakref

import _ml_fetch_pair_cases
import numpy as np
import pytest
from _mock_connect_server import mock_connect_session
from _native_classes import native_module

# The text format of protobuf, which the tests read and write the messages of
# pyspark.sql.connect.proto with (the drop-in's Rust protobuf runtime, which needs no protobuf).
pytest.importorskip("google.protobuf")

import pyspark.sql.connect.proto as pb2  # noqa: E402
from google.protobuf import text_format  # noqa: E402
from pyspark.errors import PySparkTypeError, PySparkValueError  # noqa: E402
from pyspark.ml.linalg import DenseMatrix, DenseVector, SparseMatrix, SparseVector  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql import types as T  # noqa: E402

# None in official PySpark, which runs the tests of the modules that import this one's helpers as
# the reference (test_evaluation_api_offline).
_pyspark = native_module()


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
    # A model sends its Delete without waiting for it, so the Delete of a model of this test can
    # reach the server during a later test, which would count it as a request of its own.
    service.forget_session(spark.session_id)
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
    # A message of the Rust protobuf runtime writes text_format's text itself.
    if _pyspark is not None and isinstance(message, _pyspark.ProtoMessage):
        return _pyspark._proto_message_to_string(message, as_one_line=True)
    return text_format.MessageToString(message, as_one_line=True)


def _ml_result(**fields):
    # The ExecutePlanResponse of an MlCommandResult, which the mock server sends of a command.
    return pb2.ExecutePlanResponse(ml_command_result=pb2.MlCommandResult(**fields)).SerializeToString()


def _answer_fit(service, estimator, ref="model-1", fits=1):
    # The responses of the executions of `fits` fits of estimator: each fit's Fit, which returns the
    # model of `ref`; of a LogisticRegression, the numClasses that its training summary reads after
    # it (of a binary model); and of a RandomForestClassifier, its numClasses and then its trees, as
    # PySpark reads them after the Fit. The others are answered by the model of `ref`.
    name = type(estimator).__name__
    numClasses = _ml_result(param=pb2.Expression.Literal(integer=2))
    if name == "LogisticRegression":
        service.execute_payload_queue = [_operator_info(ref), numClasses] * fits
    elif name == "RandomForestClassifier":
        trees = _operator_info("tree-1")
        service.execute_payload_queue = [_operator_info(ref), numClasses, trees] * fits
    else:
        service.execute_payload = _operator_info(ref)


def _operator_info(ref_id):
    return _ml_result(
        operator_info=pb2.MlCommandResult.MlOperatorInfo(obj_ref=pb2.ObjectRef(id=ref_id))
    )




def _label_query_result(obj):
    # The response of the query that the fit of obj makes of its labels: the maximum label of a
    # OneVsRest, which it fits one model of each class of, in the column `max(<labelCol>)` that the
    # server names it, which PySpark's fit reads (`head()["max(" + labelCol + ")"]`). The other
    # native fits make no query.
    import pyarrow as pa

    if type(obj).__name__ == "OneVsRest":
        name = "max(" + obj.getLabelCol() + ")"
        table = pa.table({name: pa.array([1.0], pa.float64())})
        return _arrow_result(table, T.StructType([T.StructField(name, T.DoubleType())]))
    return None


def _arrow_result(table, schema):
    # The ExecutePlanResponse of the rows of the Arrow table, of the Spark schema `schema`, which
    # the mock server sends of an execution.
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    batch = pb2.ExecutePlanResponse.ArrowBatch(
        row_count=table.num_rows, data=sink.getvalue().to_pybytes()
    )
    return pb2.ExecutePlanResponse(
        schema=_data_type_proto(schema), arrow_batch=batch
    ).SerializeToString()


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
    # so it can reach the server between a request of a test and the check of that request. A
    # request of a DataFrame's plan, such as the label query of a fit, has no command.
    return [
        c
        for c in _commands(service)[start:]
        if c.WhichOneof("command_type") and not c.ml_command.HasField("delete")
    ]


def _last_request_but_deletes(service):
    # The last ExecutePlanRequest that the server received, but such a Delete.
    requests = map(pb2.ExecutePlanRequest.FromString, service.execute_requests)
    return [r for r in requests if not r.plan.command.ml_command.HasField("delete")][-1]


def _relation(df):
    # The relation of a DataFrame, as the core sends it: the relation that the plans of pyspark.ml
    # send in place of its placeholder, which the core replaces with it.
    return pb2.Relation.FromString(df._relation_bytes())


def _local_data(message):
    # The Arrow IPC data of the first LocalRelation under message, as createDataFrame sends the
    # local data of a pandas DataFrame, or None when there is none.
    if message.DESCRIPTOR.name == "LocalRelation":
        return message.data
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE or field.message_type.GetOptions().map_entry:
            continue
        for item in value if field.is_repeated else [value]:
            data = _local_data(item)
            if data is not None:
                return data
    return None


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


def test_ml_connect_fitted_string_indexer_transforms_the_cached_model(ml_mock):
    # As the fitted models of PySpark Connect: fit sends an MlCommand.Fit of the estimator, and the
    # transform of the fitted model refers to the object that the server cached for the fit, not to
    # a transformer of the estimator's name, which the server rejects. (The scalers of
    # pyspark.ml.connect are PySpark's own classes, which fit and transform in Python, see
    # test_ml_connect_scalers_are_pyspark_own_classes.)
    from pyspark.ml.connect.feature import StringIndexer

    spark, service = ml_mock
    df = spark.range(1).selectExpr("'a' AS s")
    sent = len(service.execute_requests)
    service.execute_payload = _operator_info("model-1")
    model = StringIndexer(inputCol="s", outputCol="o").fit(df)
    fit = _commands_but_deletes(service, sent)[0].ml_command.fit
    assert fit.estimator.name == "org.apache.spark.ml.feature.StringIndexer"
    assert fit.estimator.type == pb2.MlOperator.OPERATOR_TYPE_ESTIMATOR
    assert dict(fit.params.params) == {
        "inputCol": pb2.Expression.Literal(string="s"),
        "outputCol": pb2.Expression.Literal(string="o"),
    }
    assert fit.dataset == _relation(df)

    scaled = model.transform(df)
    assert len(service.execute_requests) == sent + 1
    transform = _relation(scaled).ml_relation.transform
    assert transform.obj_ref == pb2.ObjectRef(id="model-1")
    assert not transform.HasField("transformer")
    assert transform.input == _relation(df)
    assert transform.params.params["outputCol"] == pb2.Expression.Literal(string="o")


def test_ml_connect_feature_exports_preserve_array_assembler():
    from pyspark import _pyspark
    from pyspark.ml.connect import ArrayAssembler, MaxAbsScaler, StandardScaler, feature

    assert StandardScaler is feature.StandardScaler and issubclass(feature.StandardScaler, _pyspark.StandardScaler)
    assert MaxAbsScaler is feature.MaxAbsScaler and issubclass(feature.MaxAbsScaler, _pyspark.MaxAbsScaler)
    assert ArrayAssembler is feature.ArrayAssembler


def _session_requests(service, spark):
    # The ExecutePlanRequests of the session of spark that the mock server received, but the
    # Deletes of models, which a release sends without waiting for it: only those of this session,
    # so that a late request of another session is not counted.
    requests = map(pb2.ExecutePlanRequest.FromString, service.execute_requests)
    return [
        request
        for request in requests
        if request.session_id == spark.session_id
        and not request.plan.command.ml_command.HasField("delete")
    ]


def _session_analyses(service, spark):
    # The AnalyzePlanRequests of the session of spark that the mock server received.
    requests = map(pb2.AnalyzePlanRequest.FromString, service.analyze_requests)
    return [request for request in requests if request.session_id == spark.session_id]


# The Fetches that the fit of an estimator sends after its Fit, as PySpark's fit does: the
# numClasses that the training summary of a LogisticRegression reads, and the trees of an ensemble,
# which the _resetUid of a random forest or a gradient-boosted tree evaluates as it walks the params
# of the model (its `trees`, a cached property).
_FETCHES_AFTER_FIT = {
    "LogisticRegression": ["numClasses"],
    "RandomForestClassifier": ["numClasses", "trees"],
    "GBTClassifier": ["trees"],
    "RandomForestRegressor": ["trees"],
    "GBTRegressor": ["trees"],
}


def _session_fit(service, spark, estimator, df, models):
    # The Fit of estimator of df, the first request of the session that the fit sends, and then
    # only the Fetches of _FETCHES_AFTER_FIT. The model is kept in models, so that it is not
    # released during the checks.
    sent = len(_session_requests(service, spark))
    _answer_fit(service, estimator)
    models.append(estimator.fit(df))
    request, *fetches = _session_requests(service, spark)[sent:]
    fetched = [r.plan.command.ml_command.fetch.methods[-1].method for r in fetches]
    assert fetched == _FETCHES_AFTER_FIT.get(type(estimator).__name__, [])
    return request.plan.command.ml_command.fit


def test_ml_connect_native_estimators_send_their_dataset_as_it_is_as_pyspark(ml_mock):
    # As pyspark.ml's try_remote_fit and try_remote_transform_relation: the fit of every native
    # estimator sends only its Fit, of the dataset as it is and of the params that it is given (no
    # read of the schema, no query of the labels, no array_to_vector of an array featuresCol), and
    # the transform of its model is the transform of the dataset as it is, which sends no request.
    # The JVM checks the columns: an array featuresCol, a featuresCol of another case than the
    # schema, and an output column that the dataset has already, raise its errors (see the e2e
    # tests). The schema that the mock server would answer has all of them.
    import pyspark.ml.connect as connect
    from _mock_connect_server import ddl_data_type

    spark, service = ml_mock
    service.data_type = ddl_data_type(
        "id bigint, Features array<int>, label double, prediction double, s string"
    )
    models = []
    # Every native estimator of a featuresCol, of its default "features" here.
    featured = [name for name, params in _NATIVE_PARAMS.items() if "featuresCol" in params and not name.endswith("Evaluator")]
    assert {"LogisticRegression", "KMeans"} <= set(featured)
    estimators = [(getattr(connect, name)(), {}) for name in featured]
    estimators += [
        (connect.LogisticRegression(featuresCol="Features"), {"featuresCol": "Features"}),
        (connect.KMeans(featuresCol="s"), {"featuresCol": "s"}),
        (connect.StringIndexer(inputCol="s", outputCol="o"), {"inputCol": "s", "outputCol": "o"}),
    ]
    for estimator, given in estimators:
        df = spark.range(1)
        analyzed = len(_session_analyses(service, spark))
        fit = _session_fit(service, spark, estimator, df, models)
        assert fit.dataset == _relation(df), estimator
        assert {k: v.string for k, v in fit.params.params.items()} == given, estimator

        sent = len(_session_requests(service, spark))
        relation = _relation(models[-1].transform(df))
        assert len(_session_requests(service, spark)) == sent, estimator
        assert len(_session_analyses(service, spark)) == analyzed, estimator
        transform = relation.ml_relation.transform
        assert transform.obj_ref == pb2.ObjectRef(id="model-1"), estimator
        assert transform.input == _relation(df), estimator
        assert transform.params == fit.params, estimator


def test_ml_connect_logistic_regression_takes_the_params_of_pyspark_ml(ml_mock):
    # As pyspark.ml.classification.LogisticRegression, the vendored class of Apache Spark, not the
    # deprecated torch class of pyspark.ml.connect: its keyword-only constructor of 21 params and
    # their defaults, and a setter of each (pyspark.ml's generic `set` and its setParams aside),
    # where the params of the torch training (numTrainWorkers, batchSize, learningRate, momentum
    # and seed) raise the TypeError of an unexpected keyword argument. A fit sends only the params
    # that are given, and the transform of the model sends those of the fit, of the output columns
    # of the JVM model, a rawPrediction and a probability vector and a double prediction, which
    # the native class outputs as they are, as pyspark.ml does.
    import inspect

    from pyspark.ml import classification
    from pyspark.ml.connect import LogisticRegression

    spark, service = ml_mock
    official = classification.LogisticRegression

    def parameters(cls):
        return [(p.name, p.kind, p.default) for p in inspect.signature(cls).parameters.values()]

    assert len(parameters(LogisticRegression)) == 21
    assert parameters(LogisticRegression) == parameters(official)
    setters = {name for name in dir(LogisticRegression) if name.startswith("set")} - {
        "set",
        "setParams",
    }
    assert setters == {n for n in dir(official) if n.startswith("set")} - {"set", "setParams"}
    assert setters == {"set" + name[0].upper() + name[1:] for name, *_ in parameters(official)}
    for param in ("numTrainWorkers", "batchSize", "learningRate", "momentum", "seed"):
        with pytest.raises(TypeError) as error:
            LogisticRegression(**{param: 1})
        assert str(error.value) == (
            f"LogisticRegression.__init__() got an unexpected keyword argument '{param}'"
        )

    df = spark.range(1)
    models = []
    assert _session_fit(service, spark, LogisticRegression(), df, models).params.params == {}
    outputs = dict(predictionCol="p", probabilityCol="q", rawPredictionCol="r")
    fit = _session_fit(service, spark, LogisticRegression(**outputs), df, models)
    expected = {name: pb2.Expression.Literal(string=value) for name, value in outputs.items()}
    assert fit.params.params == expected
    relation = _relation(models[-1].transform(df))
    assert relation.WhichOneof("rel_type") == "ml_relation"
    assert relation.ml_relation.transform.params.params == expected


def _logistic_regression_cases():
    # The constructions of a LogisticRegression that the next test makes of both classes, each with
    # the type of the error that it raises, if any: of the params of each kind, of the threshold
    # and the thresholds, and of the bounds, which are matrices and vectors, with the values that
    # the converters take and those that they reject.
    from pyspark.errors import PySparkTypeError, PySparkValueError
    from pyspark.ml.linalg import DenseMatrix, Matrices, SparseMatrix, Vectors

    dense = Matrices.dense(1, 2, [0.0, -1.0])
    sparse = SparseMatrix(2, 2, [0, 1, 2], [0, 1], [1.0, 2.0], True)
    cases = [
        ("threshold", lambda cls: cls(threshold=0.3), None),
        ("setThresholds", lambda cls: cls(threshold=0.3).setThresholds([0.4, 0.6]), None),
        ("setThreshold", lambda cls: cls(thresholds=[0.4, 0.6]).setThreshold(0.7), None),
        ("both setters", lambda cls: cls().setThresholds([0.3, 0.7]).setThreshold(0.3), None),
        ("no check of setters", lambda cls: cls(thresholds=[0.5]).setThreshold(0.9), None),
        ("empty thresholds", lambda cls: cls().setThresholds([]), PySparkTypeError),
        ("unsendable", lambda cls: cls().setMaxIter(2**70), PySparkValueError),
        ("matrices", lambda cls: cls(
            lowerBoundsOnCoefficients=dense, upperBoundsOnCoefficients=sparse), None),
        ("transposed of ints", lambda cls: cls(
            upperBoundsOnCoefficients=DenseMatrix(2, 2, [1, 2, 3, 4], True)), None),
        ("toSparse", lambda cls: cls(
            lowerBoundsOnCoefficients=Matrices.dense(2, 2, [0.0, 3.0, 0.0, 4.0]).toSparse()), None),
        ("empty sparse", lambda cls: cls(
            lowerBoundsOnCoefficients=SparseMatrix(1, 1, [0, 0], [], [])), None),
        ("vectors", lambda cls: cls(
            lowerBoundsOnIntercepts=Vectors.dense([0.5]),
            upperBoundsOnIntercepts=Vectors.sparse(2, [1], [3.0])), None),
        ("vectors of lists", lambda cls: cls(
            lowerBoundsOnIntercepts=[1, 2.5]).setUpperBoundsOnIntercepts(np.array([3.0])), None),
        ("nones", lambda cls: cls(
            lowerBoundsOnCoefficients=None, upperBoundsOnIntercepts=None), None),
        ("setter nones", lambda cls: cls().setLowerBoundsOnCoefficients(dense)
            .setUpperBoundsOnCoefficients(None).setWeightCol(None), None),
        ("constructor", lambda cls: cls(
            featuresCol="f", labelCol="l", predictionCol="p", maxIter=5.0, regParam=1,
            elasticNetParam=0.2, tol=1e-3, fitIntercept=False, threshold=0.25, thresholds=[3, 1],
            probabilityCol="q", rawPredictionCol="r", standardization=False, weightCol="w",
            aggregationDepth=3, family="multinomial", lowerBoundsOnCoefficients=dense,
            upperBoundsOnCoefficients=sparse, lowerBoundsOnIntercepts=Vectors.dense([0.0]),
            upperBoundsOnIntercepts=[1.0], maxBlockSizeInMB=1), None),
        ("setters", lambda cls: cls().setFeaturesCol("f").setLabelCol("l").setPredictionCol("p")
            .setMaxIter(2**40).setRegParam(0.1).setElasticNetParam(0.2).setTol(1e-3)
            .setFitIntercept(False).setThreshold(0.4).setThresholds([0.6, 0.4])
            .setProbabilityCol("q").setRawPredictionCol("r").setStandardization(False)
            .setWeightCol("w").setAggregationDepth(3).setFamily("binomial")
            .setLowerBoundsOnCoefficients(sparse).setUpperBoundsOnCoefficients(dense)
            .setLowerBoundsOnIntercepts(Vectors.sparse(1, [], [])).setUpperBoundsOnIntercepts([2.0])
            .setMaxBlockSizeInMB(0.5), None),
    ]
    # The constructor checks that the threshold and the thresholds that it is given agree, once it
    # has converted them, as PySpark's _checkThresholdConsistency, which raises a ValueError, or the
    # TypeError or the ZeroDivisionError of its arithmetic of a None or of a zero.
    for given, raises in [
        (dict(threshold=0.3, thresholds=[0.5, 0.5]), ValueError),
        (dict(threshold=0.5, thresholds=[0.5, 0.5]), None),
        (dict(threshold=0.25, thresholds=[3, 1]), None),
        (dict(threshold=0.123456789, thresholds=[1, 2]), ValueError),
        (dict(threshold=0.3, thresholds=[0.2, 0.3, 0.5]), ValueError),
        (dict(threshold=0.5, thresholds=[]), ValueError),
        (dict(threshold=0.5, thresholds=None), TypeError),
        (dict(threshold=None, thresholds=[0.5, 0.5]), TypeError),
        (dict(threshold=None, thresholds=None), TypeError),
        (dict(threshold=1.0, thresholds=[0.5, 0.0]), ZeroDivisionError),
        (dict(threshold=0.5, thresholds=[-1.0, 1.0]), ZeroDivisionError),
        (dict(threshold=1.0, thresholds=[0.0, 0.5]), None),
        (dict(threshold=float("nan"), thresholds=[1, 1]), None),
        (dict(threshold="x", thresholds=[0.2, 0.3, 0.5]), TypeError),
        (dict(threshold=0.3, thresholds=[0.2, 0.3, 0.5], maxIter=1.5), TypeError),
    ]:
        cases.append((f"consistency {given}", lambda cls, given=given: cls(**given), raises))
        # setParams checks them too, after its _set, of the params that either call set.
        first, rest = dict(list(given.items())[:1]), dict(list(given.items())[1:])
        cases.append((f"setParams consistency {given}",
                      lambda cls, given=given: cls().setParams(**given), raises))
        cases.append((f"setParams after {first} consistency {rest}",
                      lambda cls, first=first, rest=rest: cls(**first).setParams(**rest), raises))
    for value in [[[1.0, 2.0]], [1.0, 2.0], "m", 1, (1, 2), Vectors.dense([1.0])]:
        cases.append((f"matrix {value!r}", lambda cls, value=value: cls(
            lowerBoundsOnCoefficients=value), TypeError))
        cases.append((f"setter matrix {value!r}", lambda cls, value=value: cls()
            .setUpperBoundsOnCoefficients(value), TypeError))
    for value in ["v", 1, dense, ["a"]]:
        cases.append((f"vector {value!r}", lambda cls, value=value: cls(
            upperBoundsOnIntercepts=value), TypeError))
        cases.append((f"setter vector {value!r}", lambda cls, value=value: cls()
            .setLowerBoundsOnIntercepts(value), TypeError))
    return cases


def test_ml_connect_logistic_regression_sends_the_params_of_pyspark_ml(ml_mock):
    # Each case, of the native class and of the vendored pyspark.ml class of Apache Spark: the
    # native fit sends the params that pyspark.ml's serialize_ml_params serializes of the same
    # construction, of the bounds as PySpark sends a DenseMatrix, a SparseMatrix (also transposed)
    # and a vector (also of a list), or the construction or the fit raises PySpark's error. As over
    # Spark Connect, where the `clear` of pyspark.ml does nothing, setThreshold leaves the thresholds
    # set, and setThresholds, which is ProbabilisticClassifier's, the threshold: a fit sends both,
    # and the JVM checks that they agree (see the e2e test).
    from pyspark.ml import classification
    from pyspark.ml.connect import LogisticRegression
    from pyspark.ml.connect.serialize import serialize_ml_params
    from pyspark.ml.linalg import Matrices

    spark, service = ml_mock
    models = []

    def outcome(build, cls):
        # The params that the construction of cls sends, as bytes (of which a NaN is equal), or the
        # type and the message of its error.
        try:
            obj = build(cls)
            if cls is LogisticRegression:
                params = _session_fit(service, spark, obj, spark.range(1), models).params
            else:
                params = serialize_ml_params(obj, None)
            return params.SerializeToString(deterministic=True)
        except Exception as error:  # noqa: BLE001 - compared below
            return type(error), str(error)

    for label, build, raises in _logistic_regression_cases():
        ours = outcome(build, LogisticRegression)
        assert ours == outcome(build, classification.LogisticRegression), label
        if raises is None:
            assert isinstance(ours, bytes), (label, ours)
        else:
            assert ours[0] is raises, (label, ours)

    both = LogisticRegression(thresholds=[0.4, 0.6]).setThreshold(0.7)
    sent = _session_fit(service, spark, both, spark.range(1), models).params.params
    assert set(sent) == {"threshold", "thresholds"}
    with pytest.raises(ValueError) as error:
        LogisticRegression(threshold=0.3, thresholds=[0.5, 0.5])
    assert str(error.value) == (
        "Logistic Regression getThreshold found inconsistent values for threshold (0.3) and "
        "thresholds (equivalent to 0.5)"
    )

    # As PySpark's setParams, which checks them after its _set: its ValueError leaves the params
    # that it is given set, which a fit sends.
    def failed(cls):
        obj = cls(threshold=0.3)
        with pytest.raises(ValueError, match="inconsistent values"):
            obj.setParams(maxIter=7, thresholds=[0.5, 0.5])
        return obj

    sent = _session_fit(service, spark, failed(LogisticRegression), spark.range(1), models).params
    official = serialize_ml_params(failed(classification.LogisticRegression), None)
    assert set(sent.params) == {"threshold", "maxIter", "thresholds"}
    assert sent.SerializeToString(deterministic=True) == official.SerializeToString(
        deterministic=True
    )
    # The literal of a matrix of PySpark 4.3.0, which the Rust tests check by its bytes.
    bounded = LogisticRegression(lowerBoundsOnCoefficients=Matrices.dense(1, 2, [0.0, -1.0]))
    sent = _session_fit(service, spark, bounded, spark.range(1), models).params.params
    assert _text(sent["lowerBoundsOnCoefficients"]) == (
        'struct { struct_type { udt { type: "udt" jvm_class: "org.apache.spark.ml.linalg.MatrixUDT"'
        " } } elements { byte: 1 } elements { integer: 1 } elements { integer: 2 } elements { null "
        "{ null { } } } elements { null { null { } } } elements { specialized_array { doubles { "
        "values: 0.0 values: -1.0 } } } elements { boolean: false } }"
    )


def test_native_ml_other_invalid_inputs(ml_mock):
    # As pyspark.ml's try_remote_fit, try_remote_transform_relation and try_remote_evaluate, which
    # read the sparkSession of the dataset first: a fit, a transform or an evaluate of an object
    # that is not a DataFrame raises the AttributeError of sparkSession, before any request, also
    # of an object of the attributes that the deprecated pyspark.ml.connect read. Of an object that
    # has a sparkSession, they read its `client` (a transform asserts a session that is not None
    # first) and then the object's `_plan` (`dataset._plan.plan(client)`), whose AttributeErrors
    # they raise, and an object of both is the DataFrame of its plan. (A pandas DataFrame raises the
    # AttributeError of its own sparkSession, as test_native_ml_rejects_pandas_before_any_requests
    # pins.)
    import pyspark.ml.connect as connect

    spark, service = ml_mock
    models = []
    _session_fit(service, spark, connect.LogisticRegression(), spark.range(1), models)

    class Columns:
        columns = ["features"]
        select = None

    class Session:
        sparkSession = None

    class OfSession:
        sparkSession = spark

    frame = spark.range(1)

    class OfPlan:
        sparkSession = spark
        _plan = frame._plan

    methods = [
        ("fit", connect.LogisticRegression().fit),
        ("fit", connect.KMeans().fit),
        ("fit", connect.StringIndexer().fit),
        ("transform", models[-1].transform),
        ("transform", connect.VectorAssembler(inputCols=["a"]).transform),
        ("evaluate", connect.RegressionEvaluator().evaluate),
    ]
    sent = len(_session_requests(service, spark))
    analyzed = len(_session_analyses(service, spark))
    for name, method in methods:
        for other in (5, None, [1], "x", Columns()):
            with pytest.raises(AttributeError) as error:
                method(other)
            assert str(error.value) == (
                f"'{type(other).__name__}' object has no attribute 'sparkSession'"
            ), (name, other)
        if name == "transform":
            with pytest.raises(AssertionError):
                method(Session())
        else:
            message = "^'NoneType' object has no attribute 'client'$"
            with pytest.raises(AttributeError, match=message):
                method(Session())
        with pytest.raises(AttributeError, match="^'OfSession' object has no attribute '_plan'$"):
            method(OfSession())
    assert len(_session_requests(service, spark)) == sent
    assert len(_session_analyses(service, spark)) == analyzed
    # The transform of an object of a plan and a session is of the relation of the plan (of a plan
    # ID of this client's, as of `DataFrame(plan, session)`).
    transformed = connect.VectorAssembler(inputCols=["id"], outputCol="v").transform(OfPlan())
    expected = connect.VectorAssembler(inputCols=["id"], outputCol="v").transform(frame)
    assert _relation(transformed).ml_relation.transform.input.range == (
        _relation(expected).ml_relation.transform.input.range
    )
    # The helper calls of pyspark.ml read nothing of their dataset, but send it as an argument,
    # which pyspark.ml.connect.serialize sends as a literal unless it is a DataFrame:
    # PowerIterationClustering.assignClusters, PrefixSpan.findFrequentSequentialPatterns and
    # ALSModel.recommendForUserSubset and recommendForItemSubset. A value that cannot be a literal
    # raises the error of LiteralExpression._from_value, a pandas DataFrame too (not the
    # AttributeError of a fit, transform or evaluate), and a (value, DataType) pair whose value is
    # not of the type
    # raises the bare AssertionError of LiteralExpression(value, dataType). PySpark sends the others
    # for the server to reject (a vector and a matrix as they are, and a (value, DataType) pair as a
    # literal of that type), and so do these helper calls, in a lazy DataFrame.
    import numpy as np
    import pandas as pd
    from pyspark.errors import PySparkTypeError, PySparkValueError
    from pyspark.ml.linalg import Matrices, Vectors
    from pyspark.sql.types import IntegerType

    _session_fit(service, spark, connect.ALS(), spark.range(1), models)
    als = models[-1]
    helpers = [
        ("assignClusters", connect.PowerIterationClustering().assignClusters),
        ("findFrequentSequentialPatterns", connect.PrefixSpan().findFrequentSequentialPatterns),
        ("recommendForUserSubset", lambda dataset: als.recommendForUserSubset(dataset, 1)),
        ("recommendForItemSubset", lambda dataset: als.recommendForItemSubset(dataset, 1)),
    ]
    unconvertible = [
        (Columns(), PySparkTypeError, "UNSUPPORTED_DATA_TYPE", {"data_type": "Columns"}),
        (pd.DataFrame({"a": [1]}), PySparkTypeError, "UNSUPPORTED_DATA_TYPE",
         {"data_type": "DataFrame"}),
        (pd.Series([1]), PySparkTypeError, "UNSUPPORTED_DATA_TYPE", {"data_type": "Series"}),
        (np.array([1.0]), PySparkTypeError, "UNSUPPORTED_DATA_TYPE", {"data_type": "ndarray"}),
        ({"a": 1}, PySparkTypeError, "UNSUPPORTED_DATA_TYPE", {"data_type": "dict"}),
        ((1, 2), PySparkTypeError, "UNSUPPORTED_DATA_TYPE", {"data_type": "tuple"}),
        ([], PySparkTypeError, "CANNOT_INFER_ARRAY_ELEMENT_TYPE", None),
        (2**70, PySparkValueError, "VALUE_NOT_BETWEEN", None),
    ]
    sendable = [5, None, [1], "x", Vectors.dense([1.0]), Matrices.dense(1, 1, [1.0])]
    pair = (1, IntegerType())
    sent = len(_session_requests(service, spark))
    for name, helper in helpers:
        for other, error_class, condition, parameters in unconvertible:
            with pytest.raises(error_class) as error:
                helper(other)
            assert error.value.getCondition() == condition, (name, other)
            if parameters is not None:
                assert error.value.getMessageParameters() == parameters, (name, other)
        with pytest.raises(AssertionError):
            helper(("x", IntegerType()))
        if name == "assignClusters":
            for other in sendable:
                # The dataset is the first argument of the helper call, a literal of the value.
                (method,) = _relation(helper(other)).ml_relation.fetch.methods
                assert method.method == "powerIterationClusteringAssignClusters"
                assert method.args[0].HasField("param"), other
            (method,) = _relation(helper(pair)).ml_relation.fetch.methods
            assert method.args[0].param == pb2.Expression.Literal(integer=1)
            continue
        # PrefixSpan's and the subsets' datasets are sent for the server to reject, lazily, as
        # PySpark sends them: a (value, DataType) pair as a literal of its type.
        for other in sendable:
            helper(other)
        (method,) = _relation(helper(pair)).ml_relation.fetch.methods
        assert method.args[0].param == pb2.Expression.Literal(integer=1), name
    assert len(_session_requests(service, spark)) == sent


@pytest.mark.parametrize("case", _ml_fetch_pair_cases.golden()["cases"], ids=lambda c: c["value"])
def test_ml_connect_native_typed_pairs_are_serialized_as_pyspark(ml_mock, case):
    # As serialize of pyspark.ml.connect.serialize sends an argument of an ML call, a
    # (value, DataType) pair is the literal of LiteralExpression(value, dataType), whose asserts
    # raise a bare AssertionError, before any request, of a value that is not of the type and of a
    # type that no literal has, as of official pyspark-client 4.3.0 (tests/golden/ml_fetch_pairs.json).
    from pyspark.ml.stat import ChiSquareTest

    spark, service = ml_mock
    sent = len(_session_requests(service, spark))
    expected = {key: value for key, value in case.items() if key != "value"}
    test = lambda value: ChiSquareTest.test(value, "features", "label")  # noqa: E731
    assert _ml_fetch_pair_cases.outcome(case["value"], test, _relation) == expected
    assert len(_session_requests(service, spark)) == sent


def test_ml_connect_native_column_params_are_sent_as_given_as_pyspark(ml_mock):
    # As pyspark.ml, which sends the column params as they are given: a fit or a transform of a
    # column param of another case than the schema sends it as it is, and the JVM, which reads a
    # column by its exact name, raises its error of it or resolves it, as with pyspark.ml (see the
    # e2e test). The fit reads no schema to respell it.
    import pyspark.ml.connect as connect
    from _mock_connect_server import ddl_data_type

    spark, service = ml_mock
    service.data_type = ddl_data_type("id bigint, features array<double>, label double")
    df = spark.range(1)
    models = []
    analyzed = len(_session_analyses(service, spark))
    kmeans = _session_fit(service, spark, connect.KMeans(featuresCol="FEATURES"), df, models)
    assert kmeans.params.params == {"featuresCol": pb2.Expression.Literal(string="FEATURES")}
    assert kmeans.dataset == _relation(df)
    logistic = connect.LogisticRegression(featuresCol="Features", labelCol="LABEL")
    fit = _session_fit(service, spark, logistic, df, models)
    assert fit.params.params == {
        "featuresCol": pb2.Expression.Literal(string="Features"),
        "labelCol": pb2.Expression.Literal(string="LABEL"),
    }
    transform = _relation(models[-1].transform(df)).ml_relation.transform
    assert transform.params == fit.params
    assert len(_session_analyses(service, spark)) == analyzed

    # Standard evaluators send the column names unchanged for Spark to resolve.
    service.data_type = ddl_data_type("label double, prediction double")
    sent = len(service.execute_requests)
    service.execute_payload = _ml_result(param=pb2.Expression.Literal(double=0.5))
    evaluator = connect.RegressionEvaluator(labelCol="LABEL", predictionCol="PREDICTION")
    assert evaluator.evaluate(spark.range(1)) == 0.5
    evaluate = _commands_but_deletes(service, sent)[0].ml_command.evaluate
    assert evaluate.params.params["labelCol"] == pb2.Expression.Literal(string="LABEL")
    assert evaluate.params.params["predictionCol"] == pb2.Expression.Literal(string="PREDICTION")


def test_ml_connect_setters_set_the_params_of_their_receiver_as_pyspark(ml_mock):
    # As PySpark's Params._set: every setter of the native pyspark.ml.connect estimators,
    # transformers and evaluators sets its param on the object itself and returns the object, so
    # the next fit, transform or evaluate sends the value of a setter whose result is dropped,
    # as the object of the constructor of the same params sends it.
    from pyspark.ml.connect import (
        BinaryClassificationEvaluator,
        GaussianMixture,
        GeneralizedLinearRegression,
        KMeans,
        LinearRegression,
        LinearSVC,
        LogisticRegression,
        MulticlassClassificationEvaluator,
        RegressionEvaluator,
        StringIndexer,
        VectorAssembler,
    )
    from pyspark.ml.connect.serialize import deserialize_param

    spark, service = ml_mock
    df = spark.range(1)
    models = []  # Kept alive, so that no model is released during the checks.

    def value(literal):
        kind = literal.WhichOneof("literal_type")
        if kind == "array":
            return [value(element) for element in literal.array.elements]
        if kind == "struct":
            # A matrix or a vector, as pyspark.ml reads it.
            return deserialize_param(literal)
        return getattr(literal, kind)

    def sent_params(obj):
        # The values of the params that the next evaluate, fit or transform of obj sends.
        sent = len(service.execute_requests)
        if hasattr(obj, "evaluate"):
            service.execute_payload = _ml_result(param=pb2.Expression.Literal(double=0.5))
            assert obj.evaluate(df) == 0.5
            params = _commands_but_deletes(service, sent)[0].ml_command.evaluate.params
        elif hasattr(obj, "fit"):
            _answer_fit(service, obj)
            models.append(obj.fit(df))
            params = _commands_but_deletes(service, sent)[0].ml_command.fit.params
        else:
            params = _relation(obj.transform(df)).ml_relation.transform.params
        return {name: value(literal) for name, literal in params.params.items()}

    for cls, values in [
        (StringIndexer, dict(inputCol="in", outputCol="out", inputCols=["a", "b"], outputCols=["c", "d"], handleInvalid="keep", stringOrderType="alphabetAsc")),
        (VectorAssembler, dict(inputCols=["a", "b"], outputCol="out", handleInvalid="keep")),
        (
            LogisticRegression,
            dict(
                featuresCol="f", labelCol="y", predictionCol="p", maxIter=7, regParam=0.5,
                elasticNetParam=0.25, tol=0.25, fitIntercept=False, threshold=0.25,
                thresholds=[3.0, 1.0], probabilityCol="q", rawPredictionCol="r",
                standardization=False, weightCol="w", aggregationDepth=3, family="binomial",
                lowerBoundsOnCoefficients=DenseMatrix(1, 2, [-1.0, 0.0]),
                upperBoundsOnCoefficients=SparseMatrix(1, 2, [0, 1, 1], [0], [2.0]),
                lowerBoundsOnIntercepts=DenseVector([-1.0]),
                upperBoundsOnIntercepts=SparseVector(1, [0], [1.0]), maxBlockSizeInMB=1.5,
            ),
        ),
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
                solver="row", maxBlockSizeInMB=1.5, intermediateStorageLevel="MEMORY_ONLY",
            ),
        ),
        (
            GaussianMixture,
            dict(
                featuresCol="f", predictionCol="p", k=3, probabilityCol="q", tol=0.5,
                maxIter=7, seed=5, aggregationDepth=3, weightCol="w",
            ),
        ),
        (
            GeneralizedLinearRegression,
            dict(
                labelCol="y", featuresCol="f", predictionCol="p", family="poisson", link="log",
                fitIntercept=False, maxIter=7, tol=0.5, regParam=0.25, weightCol="w",
                solver="irls", linkPredictionCol="lp", variancePower=1.5, linkPower=0.5,
                offsetCol="o", aggregationDepth=3,
            ),
        ),
        (RegressionEvaluator, {'predictionCol': 'guess', 'labelCol': 'target', 'metricName': 'rmse', 'weightCol': 'w', 'throughOrigin': False}),
        (BinaryClassificationEvaluator, {'rawPredictionCol': 'score', 'labelCol': 'target', 'metricName': 'areaUnderROC', 'weightCol': 'w', 'numBins': 0}),
        (MulticlassClassificationEvaluator, {'predictionCol': 'guess', 'labelCol': 'target', 'metricName': 'f1', 'weightCol': 'w', 'metricLabel': 1.0, 'beta': 2.0, 'probabilityCol': 'p', 'eps': 0.05}),
    ]:
        setters = {"set" + name[0].upper() + name[1:]: name for name in values}
        # Every setter of the class is checked, and its setParams, which has a test of its own.
        assert sorted(
            n for n in dir(cls) if n.startswith("set") and n not in ("set", "setParams")
        ) == sorted(setters)
        receiver = cls()
        for setter, name in setters.items():
            if (cls.__name__, name) in _PYSPARKS_BROKEN_SETTERS:
                # PySpark's setter sets a param of another name, which the class does not have.
                with pytest.raises(AttributeError, match=f"has no attribute '{_PYSPARKS_BROKEN_SETTERS[(cls.__name__, name)]}'"):
                    getattr(receiver, setter)(value=values[name])
                receiver._set(**{name: values[name]})
                continue
            # By the name of the argument of PySpark's setters, value. The evaluators' setters are
            # of this client, as PySpark's evaluators have none, with the same argument.
            assert getattr(receiver, setter)(value=values[name]) is receiver, (cls, setter)
        params = sent_params(receiver)
        sent_values = {_sent_name(cls.__name__, name): value for name, value in values.items()}
        assert {name: params.get(name) for name in sent_values} == sent_values, cls
        # A param that PySpark's class has a setter of but its constructor does not take.
        only_set = {"intermediateStorageLevel"} if cls is KMeans else set()
        constructed = cls(**{name: v for name, v in values.items() if name not in only_set})
        for name in only_set:
            params.pop(_sent_name(cls.__name__, name))
        assert params == sent_params(constructed), cls


_ONE_VS_REST_PARAMS = dict(
    featuresCol="str", labelCol="str", predictionCol="str", rawPredictionCol="str",
    weightCol="str", parallelism="int",
)
_NO_PARAM_MAP = (
    "The native ML classes take no param map: set their params with the constructor or "
    "setParams instead."
)


# The setters of PySpark 4.3.0 that set a param of another name, which the class does not have
# (`LinearRegression.setLoss` sets `lossType`), and so raise the AttributeError of its `_set`:
# the public classes run PySpark's setters, of the same error. By the class and param, the name
# that the setter sets.
_PYSPARKS_BROKEN_SETTERS = {("LinearRegression", "loss"): "lossType"}


def _parameters_of(signature):
    """The names, kinds and defaults of the parameters of `signature`, of its annotations."""
    return [(p.name, p.kind, repr(p.default)) for p in signature.parameters.values()]


def _no_param_map(estimator, key):
    """The error of a fit of a param map of the key `key`, which is not a Param: a class of
    PySpark's Params API applies a param map as `Estimator.fit` does (a `copy` of it), whose
    `_copyValues` raises PySpark's error; the other native classes take no param map."""
    if hasattr(estimator, "copy"):
        return f"Expecting a valid instance of Param, but received: {key}"
    return _NO_PARAM_MAP


def _renamed_by_fit(name, param):
    # The fit of a OneVsRest renames its label and prediction columns to the names of its own, which
    # contain its uid, and sends those in place of the params.
    return name == "OneVsRest" and param in ("labelCol", "predictionCol")


def _user_sent(name, sent):
    # The params of a request, without the ones that the fit of a OneVsRest sends in place of its
    # own, as its uid names them.
    if name != "OneVsRest":
        return sent
    return {key: value for key, value in sent.items() if not _renamed_by_fit(name, key)}


def test_ml_connect_fpm_and_als_set_params_and_params_argument_as_pyspark(ml_mock):
    # As pyspark.ml.fpm's and pyspark.ml.recommendation's: FPGrowth, PrefixSpan and ALS have the
    # keyword-only setParams of their constructors, which returns the object, and the fits of
    # FPGrowth and ALS take the params argument; PrefixSpan.findFrequentSequentialPatterns takes
    # none, as in PySpark.
    import inspect

    from pyspark.ml.connect import FPGrowth, PrefixSpan
    from pyspark.ml.connect.recommendation import ALS

    spark, service = ml_mock
    for cls in [FPGrowth, PrefixSpan, ALS]:
        # Of the parameters of the constructor (PySpark's signatures, of their annotations).
        assert _parameters_of(inspect.signature(cls.setParams))[1:] == _parameters_of(
            inspect.signature(cls)
        ), cls
        obj = cls()
        name, value = ("maxIter", 5) if cls is ALS else ("minSupport", 0.5)
        assert obj.setParams(**{name: value}) is obj
        with pytest.raises(TypeError) as error:
            obj.setParams(0.5)
        assert str(error.value) == "Method setParams forces keyword arguments."
        with pytest.raises(TypeError) as error:
            obj.setParams(bogus=1)
        assert str(error.value) == (
            f"{cls.__name__}.setParams() got an unexpected keyword argument 'bogus'"
        )
    assert list(inspect.signature(PrefixSpan.findFrequentSequentialPatterns).parameters) == [
        "self",
        "dataset",
    ]
    df = spark.range(1)
    for estimator in [FPGrowth(), ALS()]:
        for value in [{"minSupport": 0.1}, "x"]:
            sent = _requests_but_deletes(service)
            with pytest.raises(TypeError):
                estimator.fit(df, value)
            assert _requests_but_deletes(service) == sent


def test_ml_connect_set_params_as_pyspark(ml_mock):
    # As the setParams of pyspark.ml, of each class that has one: every class, the evaluators
    # included, but LogisticRegression, whose params are still those of the deprecated
    # pyspark.ml.connect class, which has none. It is keyword-only, with the params of the
    # constructor, sets the params that it is given, in order, and returns the object. A value
    # that its converter rejects raises the TypeError of the constructor.
    import inspect

    import pyspark.ml.connect as connect

    spark, service = ml_mock
    models = []
    tables = dict(_NATIVE_PARAMS)
    tables["OneVsRest"] = _ONE_VS_REST_PARAMS
    for name, params in tables.items():
        cls = getattr(connect, name)

        def make(**given):
            # OneVsRest needs its classifier to fit, as PySpark's does.
            if name == "OneVsRest":
                return cls(classifier=connect.LinearSVC(), **given)
            return cls(**given)

        assert _parameters_of(inspect.signature(cls.setParams))[1:] == _parameters_of(
            inspect.signature(cls)
        ), name
        with pytest.raises(TypeError) as error:
            make().setParams(bogus=1)
        assert str(error.value) == f"{name}.setParams() got an unexpected keyword argument 'bogus'"
        for args in [("a",), (1, 2)]:
            with pytest.raises(TypeError) as error:
                make().setParams(*args)
            assert str(error.value) == "Method setParams forces keyword arguments.", name
        takes = set(inspect.signature(cls).parameters)
        for param, kind in params.items():
            value, expected = _NATIVE_CONVERTED[kind][0]
            if param not in takes:
                # The setter of a param that PySpark's class has none of, and its constructor
                # takes none, so that setParams takes none (as PySpark's setParams).
                with pytest.raises(TypeError) as error:
                    make().setParams(**{param: value})
                assert str(error.value) == (
                    f"{name}.setParams() got an unexpected keyword argument '{param}'"
                ), (name, param)
                continue
            obj = make()
            assert obj.setParams(**{param: value}) is obj, (name, param)
            if not _renamed_by_fit(name, param):
                sent = _user_sent(name, _native_sent(spark, service, obj, models))
                built = _user_sent(
                    name, _native_sent(spark, service, make(**{param: value}), models)
                )
                assert sent == built, (name, param)
                if name != "OneVsRest":
                    assert sent == {
                        **_defaults_sent(name), _sent_name(name, param): expected
                    }, (name, param)
            for bad, message in _NATIVE_REJECTED[kind][:1]:
                expected_error = f'Invalid param value given for param "{param}". {message}'
                with pytest.raises(TypeError) as error:
                    make().setParams(**{param: bad})
                assert str(error.value) == expected_error, (name, param)
                with pytest.raises(TypeError) as error:
                    make(**{param: bad})
                assert str(error.value) == expected_error, (name, param)
        if len(params) > 1:
            # Two params, of which the second is set by setParams: as the constructor with both.
            (first, first_kind), (second, second_kind) = list(params.items())[:2]
            first_value = _NATIVE_CONVERTED[first_kind][0][0]
            second_value = _NATIVE_CONVERTED[second_kind][0][0]
            obj = make(**{first: first_value})
            assert obj.setParams(**{second: second_value}) is obj
            both = make(**{first: first_value, second: second_value})
            assert _user_sent(name, _native_sent(spark, service, obj, models)) == _user_sent(
                name, _native_sent(spark, service, both, models)
            ), name

    # A value that is rejected after another is set leaves that one set, as PySpark's _set does.
    kmeans = connect.KMeans(k=3)
    with pytest.raises(TypeError):
        kmeans.setParams(maxIter=5, seed="x")
    assert _native_sent(spark, service, kmeans, models) == {
        "k": ("integer", 3),
        "maxIter": ("integer", 5),
    }


def test_ml_connect_one_vs_rest_set_params_sets_its_classifier_as_its_constructor(ml_mock):
    # As the constructor, setParams takes the classifier by setClassifier: a classifier of None
    # clears it, and the fit of a OneVsRest with none raises the TypeError of the max label, as
    # PySpark's fit reads the max label before the classifier's first use (on an empty dataset).
    from pyspark.ml.connect.classification import LinearSVC, OneVsRest

    spark, service = ml_mock
    models = []
    base = LinearSVC(maxIter=3)
    estimator = OneVsRest()
    assert estimator.setParams(classifier=base, parallelism=2) is estimator
    assert _user_sent("OneVsRest", _native_sent(spark, service, estimator, models)) == _user_sent(
        "OneVsRest", _native_sent(spark, service, OneVsRest(classifier=base, parallelism=2), models)
    )
    assert estimator.setParams(classifier=None) is estimator
    with pytest.raises(TypeError):
        estimator.fit(spark.range(1))
    with pytest.raises(TypeError) as error:
        OneVsRest().setParams(bogus=1)
    assert str(error.value) == "OneVsRest.setParams() got an unexpected keyword argument 'bogus'"


def test_ml_connect_params_argument_of_fit_transform_and_evaluate_as_pyspark(ml_mock):
    # As PySpark's Estimator.fit, Transformer.transform and Evaluator.evaluate, whose params
    # argument is a param map: None and an empty dict leave the params as they are, so the call
    # sends what it sends without the argument. A non-empty map raises TypeError here, as the
    # native classes take no param map. A list or tuple of maps is fitted one model per map, as
    # pyspark.ml fits it (see test_ml_connect_fit_of_a_list_of_param_maps_as_pyspark). Any other
    # value raises the TypeError that PySpark raises for it, before anything is sent. Every class
    # follows pyspark.ml, as the native classes do.
    import pyspark.ml.connect as connect

    spark, service = ml_mock
    df = spark.range(1)
    models = []
    fit_message = (
        "Params must be either a param map or a list/tuple of param maps, but got {}."
    )
    map_message = "Params must be a param map but got {}."

    def fitted(estimator, *params):
        # The params of the fit request, which the fits of a OneVsRest name with its uid.
        sent = len(service.execute_requests)
        _answer_fit(service, estimator)
        service.query_payload = _label_query_result(estimator)
        models.append(estimator.fit(df, *params))
        service.query_payload = None
        return _commands_but_deletes(service, sent)[0].ml_command.fit.params

    def evaluated(evaluator, *params):
        sent = len(service.execute_requests)
        service.execute_payload = _ml_result(param=pb2.Expression.Literal(double=0.5))
        assert evaluator.evaluate(df, *params) == 0.5
        return _commands_but_deletes(service, sent)[0].ml_command.evaluate

    def transformed(transformer, *params):
        return _relation(transformer.transform(df, *params)).ml_relation.transform

    def refused(call, message):
        sent = _requests_but_deletes(service)
        with pytest.raises(TypeError) as error:
            call()
        assert str(error.value) == message
        assert _requests_but_deletes(service) == sent

    # The fits of the pyspark.ml classes, which take a list or tuple as a fit of many models.
    classifier = connect.LinearSVC()
    for estimator in [
        connect.NaiveBayes(smoothing=2.0),
        connect.KMeans(k=3),
        connect.GBTRegressor(maxIter=3),
        connect.OneVsRest(classifier=classifier),
    ]:
        base = fitted(estimator)
        assert fitted(estimator, None) == base, type(estimator)
        assert fitted(estimator, {}) == base, type(estimator)
        no_map = _no_param_map(estimator, "smoothing")
        refused(lambda: estimator.fit(df, {"smoothing": 2.0}), no_map)
        refused(lambda: estimator.fit(df, [{"smoothing": 2.0}]), no_map)
        for value in ["x", 3]:
            refused(
                lambda: estimator.fit(df, value),
                fit_message.format(type(value)),
            )
    # LogisticRegression, as pyspark.ml's: a list or tuple is a fit of many models.
    logistic = connect.LogisticRegression()
    base = fitted(logistic)
    assert fitted(logistic, None) == base
    assert fitted(logistic, {}) == base
    # As PySpark's, a LogisticRegression, which has the Params API's copy, takes the map as one.
    not_a_param = "Expecting a valid instance of Param, but received: maxIter"
    refused(lambda: logistic.fit(df, {"maxIter": 2}), not_a_param)
    refused(lambda: logistic.fit(df, [{"maxIter": 2}]), not_a_param)
    refused(lambda: logistic.fit(df, "x"), fit_message.format(str))
    assert fitted(logistic, None) == base

    # The check of the params is before the dataset, as in PySpark's fit.
    with pytest.raises(TypeError) as error:
        connect.NaiveBayes().fit(object(), "x")
    assert str(error.value) == fit_message.format(str)

    # The transforms: a transformer and a model.
    assembler = connect.VectorAssembler(inputCols=["id"], outputCol="out")
    base = transformed(assembler)
    assert transformed(assembler, None) == base
    assert transformed(assembler, {}) == base
    model = models[-1]
    model_base = _relation(model.transform(df)).ml_relation.transform
    assert transformed(model, None) == model_base
    assert transformed(model, {}) == model_base
    # A native transformer applies a non-empty map to a copy of itself, as PySpark's transform
    # does (see test_feature_api_offline for its members), and so does a model with the Params
    # API's copy.
    assert transformed(assembler, {assembler.outputCol: "o"}) == _relation(
        assembler.copy({assembler.outputCol: "o"}).transform(df)
    ).ml_relation.transform
    for transformer in [assembler, model]:
        if transformer is model:
            # The model of LogisticRegression has the Params API's copy, which takes the map as
            # PySpark's does, and raises the TypeError of its Param check.
            refused(
                lambda: transformer.transform(df, {"outputCol": "o"}),
                "Expecting a valid instance of Param, but received: outputCol",
            )
        for value in [[], ("a",), "x", 3]:
            refused(lambda: transformer.transform(df, value), map_message.format(type(value)))

    # The evaluates of the evaluators, which take a param map as their transforms do.
    evaluator = connect.RegressionEvaluator()
    base = evaluated(evaluator)
    assert evaluated(evaluator, None) == base
    assert evaluated(evaluator, {}) == base
    # A copy with a param map of Param keys, which the evaluate sends (see test_evaluation_api_offline).
    refused(lambda: evaluator.evaluate(df, {"labelCol": "y"}), _no_param_map(evaluator, "labelCol"))
    for value in [[], "x"]:
        refused(lambda: evaluator.evaluate(df, value), map_message.format(type(value)))


def test_ml_connect_fit_of_a_list_of_param_maps_as_pyspark(ml_mock):
    # As pyspark.ml's Estimator.fit: a list or tuple of param maps fits the estimator once per map,
    # in order, and returns the list of the models. An empty list or tuple fits nothing, and does
    # not read the dataset. An empty map fits as fit(df) does, and a non-empty map raises the
    # TypeError of the native classes. A value that is not a map raises PySpark's TypeError, after
    # the fits of the maps before it. A nested list is fitted as the list is.
    import pandas as pd
    import pyspark.ml.connect as connect
    from pyspark.ml.connect import FPGrowth
    from pyspark.ml.connect.recommendation import ALS

    spark, service = ml_mock
    # The models of the fits stay alive to the end: a Delete of a released model is an execution
    # too, which would take the response of a queued fit (`_answer_fit`).
    kept = []
    df = spark.range(1)
    fit_message = "Params must be either a param map or a list/tuple of param maps, but got {}."

    def fitted(estimator, value):
        # The result of fit(df, value), and the params of the fit requests that it sent, in order.
        sent = len(service.execute_requests)
        fits = len(value) if isinstance(value, (list, tuple)) else 1
        _answer_fit(service, estimator, "model-list", fits)
        service.query_payload = _label_query_result(estimator)
        try:
            models = estimator.fit(df, value)
            kept.append(models)
        finally:
            service.query_payload = None
        return models, [c.ml_command.fit.params for c in _commands_but_deletes(service, sent)]

    def refused(call, message):
        sent = _requests_but_deletes(service)
        with pytest.raises(TypeError) as error:
            call()
        assert str(error.value) == message
        assert _requests_but_deletes(service) == sent

    classifier = connect.LinearSVC()
    for estimator in [
        connect.NaiveBayes(),
        connect.KMeans(),
        connect.OneVsRest(classifier=classifier),
        connect.LogisticRegression(),
        ALS(),
        FPGrowth(),
    ]:
        # One fit sends one command, or one per class of a OneVsRest.
        _, single = fitted(estimator, None)
        assert single, type(estimator)
        for value in [[], ()]:
            sent = _requests_but_deletes(service)
            assert estimator.fit(df, value) == []
            assert _requests_but_deletes(service) == sent

        models, params = fitted(estimator, [{}])
        assert len(models) == 1 and params == single, type(estimator)
        models, params = fitted(estimator, ({}, {}))
        assert len(models) == 2 and params == single * 2, type(estimator)
        models, params = fitted(estimator, [[{}], ({},)])
        assert len(models) == 2 and params == single * 2, type(estimator)
        assert all(isinstance(model, list) and len(model) == 1 for model in models)

        refused(lambda: estimator.fit(df, [{"x": 1}]), _no_param_map(estimator, "x"))
        refused(lambda: estimator.fit(df, [1]), fit_message.format(int))
        refused(lambda: estimator.fit(df, [[1]]), fit_message.format(int))
        # The fit of the first map is sent before the second element raises.
        sent = len(service.execute_requests)
        _answer_fit(service, estimator, "model-list")
        service.query_payload = _label_query_result(estimator)
        with pytest.raises(TypeError) as error:
            estimator.fit(df, [{}, 1])
        service.query_payload = None
        assert str(error.value) == fit_message.format(int)
        sent_params = [c.ml_command.fit.params for c in _commands_but_deletes(service, sent)]
        assert sent_params == single

        # The dataset of an empty list is not read, so a pandas DataFrame is no error for it.
        pdf = pd.DataFrame({"x": [1.0]})
        sent = _requests_but_deletes(service)
        assert estimator.fit(pdf, []) == []
        assert estimator.fit(pdf, ()) == []
        assert _requests_but_deletes(service) == sent
        # A map reads the dataset, which must be a Spark DataFrame, as fit(pdf) reads it: the
        # AttributeError of the pandas DataFrame's own sparkSession. PySpark's own
        # OneVsRest._fit, which fits a OneVsRest of it, raises the KeyError of its pandas
        # aggregate instead.
        if isinstance(estimator, connect.OneVsRest):
            with pytest.raises(KeyError):
                estimator.fit(pdf, [{}])
        else:
            with pytest.raises(AttributeError) as error:
                estimator.fit(pdf, [{}])
            assert str(error.value) == "'DataFrame' object has no attribute 'sparkSession'"



# The params of the native pyspark.ml.connect classes, by the TypeConverters method of each
# param in PySpark. Their constructors take them all but thresholds, which PySpark's
# RandomForestClassifier takes by its setter only.
_NATIVE_PARAMS = {
    "LDA": {'featuresCol': 'str', 'maxIter': 'int', 'seed': 'int', 'checkpointInterval': 'int', 'k': 'int', 'optimizer': 'str', 'learningOffset': 'float', 'learningDecay': 'float', 'subsamplingRate': 'float', 'optimizeDocConcentration': 'bool', 'docConcentration': 'float_list', 'topicConcentration': 'float', 'topicDistributionCol': 'str', 'keepLastCheckpoint': 'bool'},
    'AFTSurvivalRegression': {'featuresCol': 'str', 'labelCol': 'str', 'predictionCol': 'str', 'fitIntercept': 'bool', 'maxIter': 'int', 'tol': 'float', 'censorCol': 'str', 'quantileProbabilities': 'float_list', 'quantilesCol': 'str', 'aggregationDepth': 'int', 'maxBlockSizeInMB': 'float'},
    'IsotonicRegression': {'featuresCol': 'str', 'labelCol': 'str', 'predictionCol': 'str', 'weightCol': 'str', 'isotonic': 'bool', 'featureIndex': 'int'},
    'FMRegressor': {'featuresCol': 'str', 'labelCol': 'str', 'predictionCol': 'str', 'factorSize': 'int', 'fitIntercept': 'bool', 'fitLinear': 'bool', 'regParam': 'float', 'miniBatchFraction': 'float', 'initStd': 'float', 'maxIter': 'int', 'stepSize': 'float', 'tol': 'float', 'solver': 'str', 'seed': 'int'},

    "StringIndexer": dict(inputCol="str", outputCol="str", inputCols="str_list", outputCols="str_list", handleInvalid="str", stringOrderType="str"),
    "VectorAssembler": dict(inputCols="str_list", outputCol="str", handleInvalid="str"),
    "LogisticRegression": dict(
        featuresCol="str", labelCol="str", predictionCol="str", maxIter="int", regParam="float",
        elasticNetParam="float", tol="float", fitIntercept="bool", threshold="float",
        thresholds="float_list", probabilityCol="str", rawPredictionCol="str",
        standardization="bool", weightCol="str", aggregationDepth="int", family="str",
        lowerBoundsOnCoefficients="matrix", upperBoundsOnCoefficients="matrix",
        lowerBoundsOnIntercepts="vector", upperBoundsOnIntercepts="vector",
        maxBlockSizeInMB="float",
    ),
    "FMClassifier": dict(featuresCol="str", labelCol="str", predictionCol="str", probabilityCol="str", rawPredictionCol="str", factorSize="int", fitIntercept="bool", fitLinear="bool", regParam="float", miniBatchFraction="float", initStd="float", maxIter="int", stepSize="float", tol="float", solver="str", thresholds="float_list", seed="int"),
    "MultilayerPerceptronClassifier": dict(featuresCol="str", labelCol="str", predictionCol="str", maxIter="int", tol="float", seed="int", blockSize="int", stepSize="float", solver="str", probabilityCol="str", rawPredictionCol="str", thresholds="float_list"),
    "LinearSVC": dict(
        featuresCol="str", labelCol="str", predictionCol="str", maxIter="int", regParam="float",
        tol="float", rawPredictionCol="str", fitIntercept="bool", standardization="bool",
        threshold="float", weightCol="str", aggregationDepth="int", maxBlockSizeInMB="float",
    ),
    "GeneralizedLinearRegression": dict(labelCol="str", featuresCol="str", predictionCol="str", family="str", link="str", fitIntercept="bool", maxIter="int", tol="float", regParam="float", weightCol="str", solver="str", linkPredictionCol="str", variancePower="float", linkPower="float", offsetCol="str", aggregationDepth="int"),
    "LinearRegression": dict(
        featuresCol="str", labelCol="str", predictionCol="str", maxIter="int", regParam="float",
        elasticNetParam="float", tol="float", fitIntercept="bool", standardization="bool",
        solver="str", weightCol="str", aggregationDepth="int", loss="str", epsilon="float",
        maxBlockSizeInMB="float",
    ),
    "NaiveBayes": dict(
        featuresCol="str", labelCol="str", predictionCol="str", probabilityCol="str",
        rawPredictionCol="str", smoothing="float", modelType="str",
        thresholds="float_list", weightCol="str",
    ),
    "DecisionTreeClassifier": dict(
        featuresCol="str", labelCol="str", predictionCol="str", probabilityCol="str",
        rawPredictionCol="str", maxDepth="int", maxBins="int", minInstancesPerNode="int",
        minInfoGain="float", pruneTree="bool", maxMemoryInMB="int", cacheNodeIds="bool",
        checkpointInterval="int", impurity="str",
        seed="int", leafCol="str", minWeightFractionPerNode="float",
        weightCol="str", thresholds="float_list",
    ),
    "GBTRegressor": dict(
        featuresCol="str", labelCol="str", predictionCol="str",
         maxDepth="int", maxBins="int", minInstancesPerNode="int",
        minInfoGain="float", maxMemoryInMB="int", cacheNodeIds="bool",
        checkpointInterval="int", impurity="str", featureSubsetStrategy="str",
        seed="int", subsamplingRate="float", leafCol="str", minWeightFractionPerNode="float",
        lossType="str", maxIter="int", stepSize="float", validationIndicatorCol="str", weightCol="str",
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
    "BisectingKMeans": dict(
        featuresCol="str", predictionCol="str", maxIter="int", seed="int", k="int",
        minDivisibleClusterSize="float", distanceMeasure="str", weightCol="str",
    ),
    "GaussianMixture": dict(
        featuresCol="str", predictionCol="str", k="int", probabilityCol="str", tol="float",
        maxIter="int", seed="int", aggregationDepth="int", weightCol="str",
    ),
    "KMeans": dict(
        featuresCol="str", predictionCol="str", k="int", initMode="str", initSteps="int",
        tol="float", maxIter="int", seed="int", distanceMeasure="str", weightCol="str",
        solver="str", maxBlockSizeInMB="float",
    ),
    'BinaryClassificationEvaluator': dict(rawPredictionCol='str', labelCol='str', metricName='str', weightCol='str', numBins='int'),
    'RegressionEvaluator': dict(predictionCol='str', labelCol='str', metricName='str', weightCol='str', throughOrigin='bool'),
    'MulticlassClassificationEvaluator': dict(predictionCol='str', labelCol='str', metricName='str', weightCol='str', metricLabel='float', beta='float', probabilityCol='str', eps='float'),
    'MultilabelClassificationEvaluator': dict(predictionCol='str', labelCol='str', metricName='str', metricLabel='float'),
    'ClusteringEvaluator': dict(predictionCol='str', featuresCol='str', metricName='str', distanceMeasure='str', weightCol='str'),
    'RankingEvaluator': dict(predictionCol='str', labelCol='str', metricName='str', k='int'),
}

def _serialized(value):
    # The kind and the text of the literal that PySpark's ML serializer sends of a matrix or of a
    # vector: a struct of its UDT.
    from pyspark.ml.connect.serialize import serialize_param

    return "struct", _text(serialize_param(value, None).struct)


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
    "matrix": [
        (value, _serialized(value))
        for value in [
            DenseMatrix(1, 2, [0, 1]),
            SparseMatrix(2, 2, [0, 1, 2], [1, 0], [3.0, -1.0], True),
        ]
    ],
    "vector": [
        (value, _serialized(DenseVector(value) if isinstance(value, list) else value))
        for value in [[1, 2.5], SparseVector(2, [1], [3.0])]
    ],
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
    "matrix": [([[1.0, 2.0]], "Could not convert [[1.0, 2.0]] to matrix"),
               ("m", "Could not convert m to matrix")],
    "vector": [("v", "Could not convert v to vector"), (["a"], "Could not convert ['a'] to vector")],
}


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
        _answer_fit(service, obj)
        service.query_payload = _label_query_result(obj)
        models.append(obj.fit(df))
        service.execute_payload_queue = None
        service.query_payload = None
        fit = _commands_but_deletes(service, sent)[0].ml_command.fit
        return fit.estimator, fit.params
    transform = _relation(obj.transform(df)).ml_relation.transform
    return transform.transformer, transform.params


def _literal_value(literal):
    kind = literal.WhichOneof("literal_type")
    if kind == "array":
        return kind, [_literal_value(element)[1] for element in literal.array.elements]
    if kind == "struct":
        return kind, _text(literal.struct)
    return kind, getattr(literal, kind)


def _sent_name(name, param):
    return param


def _defaults_sent(name):
    # The native classes send only the params that are set, like standard pyspark.ml.
    return {}


def _native_sent(spark, service, obj, models):
    _, params = _native_request(spark, service, obj, models)
    return {name: _literal_value(literal) for name, literal in params.params.items()}


def test_ml_connect_constructors_are_keyword_only_as_pyspark(ml_mock):
    import pyspark.ml.connect as connect

    for name in _NATIVE_PARAMS:
        cls = getattr(connect, name)
        with pytest.raises(TypeError) as error:
            cls(bogus=1)
        assert str(error.value) == f"{name}.__init__() got an unexpected keyword argument 'bogus'"
        for args in [("a",), (1, 2)]:
            with pytest.raises(TypeError) as error:
                cls(*args)
            assert str(error.value) == "Method __init__ forces keyword arguments.", name


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
            if (name, param) in _PYSPARKS_BROKEN_SETTERS:
                continue
            for value, expected in _NATIVE_CONVERTED[kind]:
                made = [getattr(cls(), setter)(value)]
                if param != "thresholds":
                    made.append(cls(**{param: value}))
                for obj in made:
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
        sent = _native_sent(spark, service, getattr(connect, name)(), models)
        assert sent == _defaults_sent(name), name
    assert _native_sent(spark, service, connect.KMeans(k=3), models) == {"k": ("integer", 3)}
    forest = connect.RandomForestClassifier(numTrees=3, seed=2**33)
    assert _native_sent(spark, service, forest, models) == {
        "numTrees": ("integer", 3),
        "seed": ("long", 2**33),
    }


def test_ml_connect_evaluators_take_the_signatures_and_defaults_of_pyspark(ml_mock):
    import inspect

    import pyspark.ml.connect as connect
    from _evaluators import FIXTURE, signature_text

    spark, service = ml_mock
    for name, info in FIXTURE["classes"].items():
        cls = getattr(connect, name)
        assert signature_text(inspect.signature(cls)) == info["signature"]
        assert _native_sent(spark, service, cls(), []) == {}


def test_ml_connect_binary_evaluator_uses_standard_raw_prediction_parameters(ml_mock):
    import pyspark.ml.connect as connect

    spark, service = ml_mock
    cls = connect.BinaryClassificationEvaluator
    assert _native_sent(spark, service, cls(), []) == {}
    assert _native_sent(spark, service, cls(rawPredictionCol="p"), []) == {
        "rawPredictionCol": ("string", "p")
    }
    evaluator = cls(metricName="areaUnderPR", labelCol="y", rawPredictionCol="p")
    assert _native_sent(spark, service, evaluator, []) == {
        "metricName": ("string", "areaUnderPR"),
        "labelCol": ("string", "y"),
        "rawPredictionCol": ("string", "p"),
    }
    assert evaluator.setRawPredictionCol("q") is evaluator
    with pytest.raises(TypeError, match="unexpected keyword argument 'probabilityCol'"):
        cls(probabilityCol="p")


def test_ml_connect_int_lists_are_literals_of_the_type_of_their_first_element_as_pyspark(ml_mock):
    # As PySpark's serialization of a list of ints (`toListInt`): LiteralExpression._from_value
    # types the list by its first element, an integer when it fits 32 bits, else a long, and each
    # element is a LiteralExpression of that type, whose assert fails of an element out of its
    # range. The constructor, the setters, setParams and a copy take any list of ints, which the
    # getter returns, and the next fit, transform or save raises, before any request, the bare
    # AssertionError of such an element, the VALUE_NOT_BETWEEN of a first element beyond a long,
    # and the CANNOT_INFER_ARRAY_ELEMENT_TYPE of no element: of the first param of `_paramMap` that
    # it cannot send, as official pyspark-client 4.3.0 does of each of these.
    import pyspark.ml.connect as connect

    spark, service = ml_mock
    models = []
    mlp = connect.MultilayerPerceptronClassifier
    slicer = connect.VectorSlicer
    sendable = [
        ([2, 3, 2], "integer"),
        ([-(2**31), 2**31 - 1], "integer"),
        ([2**33, 3, 2], "long"),
        ([2**40, -(2**63), 2**63 - 1], "long"),
    ]
    for values, element in sendable:
        made = [
            ("layers", mlp(layers=values)),
            ("layers", mlp().setLayers(values)),
            ("layers", mlp().setParams(layers=values)),
            ("indices", slicer(indices=values)),
            ("indices", slicer().setIndices(values)),
        ]
        for name, obj in made:
            _, params = _native_request(spark, service, obj, models)
            literal = params.params[name]
            assert literal.array.element_type.WhichOneof("kind") == element, (name, values)
            elements = literal.array.elements
            assert [e.WhichOneof("literal_type") for e in elements] == [element] * len(values)
            assert [getattr(e, element) for e in elements] == values
    unsendable = [
        ([2, 2**32 + 3, 2], AssertionError, None),
        ([2, 2**64, 2], AssertionError, None),
        ([2**40, 2**64], AssertionError, None),
        ([2, -(2**31) - 1], AssertionError, None),
        ([2**64, 2, 2], PySparkValueError, "VALUE_NOT_BETWEEN"),
        ([], PySparkTypeError, "CANNOT_INFER_ARRAY_ELEMENT_TYPE"),
    ]
    df = spark.range(1)
    sent = _requests_but_deletes(service)
    for values, error_class, condition in unsendable:
        made = [
            ("getLayers", mlp(layers=values), "fit"),
            ("getLayers", mlp().setLayers(values), "fit"),
            ("getLayers", mlp().setParams(layers=values).copy(), "fit"),
            ("getIndices", slicer(indices=values), "transform"),
            ("getNumBucketsArray", connect.QuantileDiscretizer(numBucketsArray=values), "fit"),
        ]
        for getter, obj, call in made:
            assert getattr(obj, getter)() == values
            for run in [lambda: getattr(obj, call)(df), lambda: obj.write().save("/unsent/path")]:
                with pytest.raises(error_class) as error:
                    run()
                if condition is None:
                    assert type(error.value) is AssertionError and str(error.value) == ""
                else:
                    assert error.value.getCondition() == condition, (getter, values)
    assert _requests_but_deletes(service) == sent
    # A list that PySpark sends, set again, is sent.
    estimator = mlp(layers=[2, 2**33])
    assert estimator.setLayers([2, 3]) is estimator
    assert _native_sent(spark, service, estimator, models)["layers"] == ("array", [2, 3])
    # The error is that of the first param of `_paramMap` that PySpark cannot send.
    for estimator, expected in [
        (mlp(maxIter=2**64, layers=[2, 2**33]), PySparkValueError),
        (mlp(layers=[2, 2**33], maxIter=2**64), AssertionError),
        (mlp().setMaxIter(2**64).setLayers([2, 2**33]), PySparkValueError),
        (mlp().setLayers([2, 2**33]).setMaxIter(2**64), AssertionError),
    ]:
        with pytest.raises(expected) as error:
            estimator.fit(df)
        assert type(error.value) is expected
    # A model's copy of such a list keeps it, and its transform raises.
    _native_request(spark, service, mlp(layers=[2, 3]), models)
    model = models[-1]
    copied = model.copy({model.layers: [2, 2**32 + 3]})
    assert copied.getLayers() == [2, 2**32 + 3]
    sent = _requests_but_deletes(service)
    with pytest.raises(AssertionError) as error:
        copied.transform(df)
    assert type(error.value) is AssertionError
    assert _requests_but_deletes(service) == sent


def test_ml_connect_params_of_a_python_subclass_are_sent_as_pyspark(ml_mock):
    # As PySpark's `serialize_ml_params`, which serializes every param of `_paramMap`: a Param of a
    # Python subclass that is set is sent as the literal of its value (`serialize_param`) by a fit,
    # a transform, an evaluate and a save, which the server refuses ("Param myParam does not
    # exist.", as official pyspark-client 4.3.0 shows of each), and one of only a default is not
    # sent. A value of no literal raises the error of its serialization, of the first param of
    # `_paramMap` that cannot be sent, before any request.
    import pyspark.ml.connect as connect
    from pyspark.ml.param import Param, Params

    spark, service = ml_mock
    models = []

    def extra():
        return Param(Params._dummy(), "myParam", "a param of the subclass")

    class MyClassifier(connect.LinearSVC):
        myParam = extra()

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._setDefault(myParam=1)

    class MyTokenizer(connect.Tokenizer):
        myParam = extra()

    class MyEvaluator(connect.RegressionEvaluator):
        myParam = extra()

    estimator = MyClassifier(maxIter=2)
    assert _native_sent(spark, service, estimator, models) == {"maxIter": ("integer", 2)}
    estimator._set(myParam=5)
    for obj in [estimator, MyTokenizer(inputCol="t")._set(myParam=5), MyEvaluator()._set(myParam=5)]:
        assert _native_sent(spark, service, obj, models)["myParam"] == ("integer", 5)
        sent = len(service.execute_requests)
        obj.write().save("/saved/path")
        (command,) = _commands_but_deletes(service, sent)
        assert command.ml_command.write.params.params["myParam"] == pb2.Expression.Literal(
            integer=5
        )
    for value, expected in [
        ([1.5], ("array", [1.5])),
        ("a", ("string", "a")),
        (None, ("null", pb2.DataType(null=pb2.DataType.NULL()))),
    ]:
        assert _native_sent(spark, service, MyClassifier()._set(myParam=value), models)[
            "myParam"
        ] == expected, value
    df = spark.range(1)
    sent = _requests_but_deletes(service)
    for value, condition in [((1,), "UNSUPPORTED_DATA_TYPE"), (2**63, "VALUE_NOT_BETWEEN")]:
        for run in [lambda obj: obj.fit(df), lambda obj: obj.write().save("/unsent/path")]:
            with pytest.raises((PySparkTypeError, PySparkValueError)) as error:
                run(MyClassifier()._set(myParam=value))
            assert error.value.getCondition() == condition, value
    # Of the order of `_paramMap`: its first param that cannot be sent raises.
    for first, then, condition in [
        ("myParam", "maxIter", "UNSUPPORTED_DATA_TYPE"),
        ("maxIter", "myParam", "VALUE_NOT_BETWEEN"),
    ]:
        obj = MyClassifier()
        for name in [first, then]:
            obj._set(**{name: (1,) if name == "myParam" else 2**64})
        with pytest.raises((PySparkTypeError, PySparkValueError)) as error:
            obj.fit(df)
        assert error.value.getCondition() == condition, first
    assert _requests_but_deletes(service) == sent


def test_a_writer_and_a_reader_read_their_attributes_at_the_call_as_pyspark(ml_mock):
    # PySpark's `RemoteMLWriter.save` (ml/connect/readwrite.py:39-50) reads the writer's
    # `_instance`, `shouldOverwrite` and `optionMap` at the save, of a copy, a pickle and a writer
    # of the constructor too, which `pb2.MlCommand.Write` takes as its protobuf fields: an int as
    # the bool, None as no overwrite (unset, which the server reads as false) and as no options,
    # pairs as the options, and upb's error of another value, before any request. An object that is not native is saved by PySpark's own
    # `saveInstance`, and a reader's `_clazz` that is not a native class is loaded by its
    # `loadInstance`, with their errors (readwrite.py:190, 366).
    import copy
    import pickle

    from pyspark.ml.connect.classification import LogisticRegression

    spark, service = ml_mock
    lr, other = LogisticRegression(maxIter=3), LogisticRegression(maxIter=9)

    def written(writer):
        sent = len(service.execute_requests)
        writer.save("/saved/path")
        (command,) = _commands_but_deletes(service, sent)
        write = command.ml_command.write
        overwrite = write.should_overwrite if write.HasField("should_overwrite") else None
        return write.operator.uid, overwrite, dict(write.options), write.params.params["maxIter"].integer

    writer = lr.write()
    lr.setMaxIter(4)
    assert written(writer) == (lr.uid, False, {}, 4)
    writer.shouldOverwrite, writer.optionMap = 1, [("Key", "v")]
    assert written(writer) == (lr.uid, True, {"Key": "v"}, 4)
    for copied in (copy.copy(writer), pickle.loads(pickle.dumps(writer)), type(writer)(lr)):
        copied.shouldOverwrite, copied.optionMap = 1, [("Key", "v")]
        assert written(copied) == (lr.uid, True, {"Key": "v"}, 4)
    writer._instance = other
    assert written(writer) == (other.uid, True, {"Key": "v"}, 9)
    writer.shouldOverwrite, writer.optionMap = None, None
    assert written(writer) == (other.uid, None, {}, 9)
    sent = _requests_but_deletes(service)
    for name, value, error, message in [
        ("shouldOverwrite", "x", TypeError, "'str' object cannot be interpreted as an integer"),
        ("optionMap", {"a": 1}, TypeError, "bad argument type for built-in operation"),
        ("optionMap", "ab", ValueError, r"not enough values to unpack \(expected 2, got 1\)"),
        ("_instance", None, NotImplementedError, "Unsupported write for <class 'NoneType'>"),
    ]:
        failing = lr.write()
        setattr(failing, name, value)
        with pytest.raises(error, match=f"^{message}$"):
            failing.save("/unsent/path")
    del writer.optionMap
    with pytest.raises(AttributeError, match="^'RemoteMLWriter' object has no attribute 'optionMap'$"):
        writer.save("/unsent/path")
    reader = LogisticRegression.read()
    for clazz, error, message in [
        (int, RuntimeError, "Unsupported read for <class 'int'>"),
        (None, TypeError, r"issubclass\(\) arg 1 must be a class"),
    ]:
        reader._clazz = clazz
        with pytest.raises(error, match=f"^{message}$"):
            reader.load("/unsent/path")
    assert _requests_but_deletes(service) == sent


def test_ml_connect_unsendable_ints_none_and_uids_as_pyspark(ml_mock):
    # As PySpark: an int beyond a long is taken by the constructor and the setter, and the
    # next fit raises the VALUE_NOT_BETWEEN of PySpark's serialization of the params, before
    # any request, until the param is set again. None is sent as a null, as PySpark sends it.
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
        cls = getattr(connect, name)
        for param, kind in params.items():
            setter = "set" + param[0].upper() + param[1:]
            if (name, param) in _PYSPARKS_BROKEN_SETTERS:
                continue
            value = _NATIVE_CONVERTED[kind][0][0]
            obj = cls()
            assert getattr(obj, setter)(value) is obj
            assert getattr(obj, setter)(None) is obj
            sent_name = _sent_name(name, param)
            sent_names = {sent_name}
            sent = _native_sent(spark, service, obj, models)
            assert sent.keys() == sent_names | _defaults_sent(name).keys(), (name, param)
            if sent_names:
                assert sent[sent_name][0] == "null", (name, param)
            if param != "thresholds":
                sent = _native_sent(spark, service, cls(**{param: None}), models)
                assert sent.keys() == sent_names | _defaults_sent(name).keys(), (name, param)
                if sent_names:
                    assert sent[sent_name][0] == "null", (name, param)
        operator, _ = _native_request(spark, service, cls(), models)
        assert re.fullmatch(name + "_[0-9a-f]{12}", operator.uid), operator.uid


def test_ml_connect_validators_of_a_seed_beyond_a_long_raise_value_not_between_as_pyspark(
    ml_mock,
):
    # As PySpark: the seed param takes an int beyond a long, and the fit raises the
    # VALUE_NOT_BETWEEN of `F.rand(seed)`'s literal (CrossValidator's `_kFold`, and
    # TrainValidationSplit's `_fit`), before any request.
    from pyspark.ml.connect.classification import LogisticRegression
    from pyspark.ml.connect.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.connect.tuning import CrossValidator, TrainValidationSplit
    from pyspark.ml.tuning import ParamGridBuilder

    spark, service = ml_mock
    for seed in (2**63, -(2**63) - 1, 2**64):
        for cls, kwargs in [(CrossValidator, {"numFolds": 2}), (TrainValidationSplit, {})]:
            lr = LogisticRegression(maxIter=1)
            validator = cls(
                estimator=lr,
                estimatorParamMaps=ParamGridBuilder().addGrid(lr.regParam, [0.1]).build(),
                evaluator=BinaryClassificationEvaluator(),
                seed=seed,
                **kwargs,
            )
            assert validator.getSeed() == seed
            sent = _requests_but_deletes(service)
            with pytest.raises(PySparkValueError) as error:
                validator.fit(spark.range(1))
            assert error.value.getCondition() == "VALUE_NOT_BETWEEN", (cls, seed)
            assert str(error.value) == (
                "[VALUE_NOT_BETWEEN] Value for `value` must be between -9223372036854775808 and "
                "32767."
            )
            assert _requests_but_deletes(service) == sent, (cls, seed)


def test_ml_connect_setters_run_while_a_fit_or_an_evaluate_waits_as_pyspark(ml_mock):
    # As PySpark, whose fit and evaluate hold no lock of the estimator or the evaluator: a
    # setter that another thread calls while a fit or an evaluate of the same native object
    # waits for the server sets the param, and the waiting call sends the params of when it
    # started. (VectorAssembler sends no request to wait for.)
    import threading

    import pyspark.ml.connect as connect

    spark, service = ml_mock
    df = spark.range(1)
    # Kept alive, so that the release of no model reaches the server after a reset.
    models = []
    for name, params in _NATIVE_PARAMS.items():
        if name == "VectorAssembler":
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
            _answer_fit(service, obj)
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
        # The request of the fit or the evaluate, and the Fetches that follow a fit (see
        # _FETCHES_AFTER_FIT).
        (command,) = [
            c
            for c in _commands(service)
            if c.WhichOneof("command_type") and c.ml_command.WhichOneof("command") != "fetch"
        ]
        held = command.ml_command.evaluate if hasattr(obj, "evaluate") else command.ml_command.fit
        assert set(held.params.params) == set(_defaults_sent(name)), name
        assert _native_sent(spark, service, obj, models) == {
            **_defaults_sent(name),
            param: ("string", "changed"),
        }


def test_the_models_and_summaries_of_fits_have_pysparks_java_obj_of_their_call_java(ml_mock):
    # As PySpark's `try_remote_fit`: the `_java_obj` of a fitted model is the RemoteModelRef of its
    # object (of its copy too), and its training summary is its `_summary`, of the attributes, in
    # this order, `_java_obj` (`"<id>.summary"`), `_summary_dataset` (the training dataset of a
    # regression) and `_remote_model_obj` (the RemoteModelRef of the model). `evaluate` sets the
    # `_java_obj` of its summary, the id that the server returns, and the model
    # (`__source_transformer__`). PySpark's `_call_java` of each (`try_remote_call`) sends the Fetch
    # of the id and the methods of its `_java_obj`. A model of PySpark's constructor has the
    # `_java_obj` of its `__init__`, None, and no `_summary`.
    import copy

    import pyspark.ml.connect as connect
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.util import RemoteModelRef

    spark, service = ml_mock
    frame = spark.createDataFrame([(1.0, Vectors.dense(0.0, 1.0))], ["label", "features"])
    service.execute_payload = _operator_info("model-of-a-call-java")
    model = connect.regression.LinearRegression().fit(frame)
    svc = connect.classification.LinearSVC().fit(frame)
    service.payload_for_request = {b"evaluate": _ml_result(summary="evaluated-of-a-call-java")}
    evaluated = model.evaluate(frame)
    # A copy shares the model's RemoteModelRef, also of a copy before its first read, as PySpark's
    # `copy.copy` shares the `_java_obj` of its attributes.
    shallow = copy.copy(model)
    assert type(model._java_obj) is RemoteModelRef and shallow._java_obj is model._java_obj
    assert model._java_obj.ref_id == model.copy()._java_obj.ref_id == "model-of-a-call-java"
    summary = model.summary
    # The `summary()` method of a LinearSVCModel, as PySpark's.
    assert model._summary is summary and svc._summary is svc.summary()
    assert list(vars(summary)) == ["_java_obj", "_summary_dataset", "_remote_model_obj"]
    assert summary._java_obj == "model-of-a-call-java.summary"
    assert _relation(summary._summary_dataset) == _relation(frame)
    # The model's own RemoteModelRef, of a reference of the summary too (`add_ref`), and of one of
    # the `JavaParams.copy` above.
    assert summary._remote_model_obj is model._java_obj
    assert svc.summary()._remote_model_obj is svc._java_obj
    assert summary._remote_model_obj._ref_count == 3 and svc._java_obj._ref_count == 2
    assert list(vars(evaluated)) == ["_java_obj", "__source_transformer__"]
    assert evaluated._java_obj == "evaluated-of-a-call-java"
    assert evaluated.__source_transformer__ is model
    service.payload_for_request = {}
    service.execute_payload = _ml_result(param=pb2.Expression.Literal(integer=7))
    sent = len(service.execute_requests)
    assert model._call_java("numFeatures") == 7
    assert summary._call_java("totalIterations") == 7
    assert evaluated._call_java("numInstances") == 7
    fetches = [command.ml_command.fetch for command in _commands_but_deletes(service, sent)]
    assert [(fetch.obj_ref.id, [m.method for m in fetch.methods]) for fetch in fetches] == [
        ("model-of-a-call-java", ["numFeatures"]),
        ("model-of-a-call-java", ["summary", "totalIterations"]),
        ("evaluated-of-a-call-java", ["numInstances"]),
    ]
    unbound = connect.regression.LinearRegressionModel()
    assert unbound._java_obj is None and not unbound.hasSummary
    message = "^'LinearRegressionModel' object has no attribute '_summary'$"
    with pytest.raises(AttributeError, match=message):
        unbound._summary


def test_the_dataframe_members_of_a_summary_of_its_constructor_read_its_summary_dataset(ml_mock):
    # As PySpark's invoke_remote_attribute_relation: of a summary whose `java_obj` starts with the
    # `summary` method (`"<id>.summary"`, as PySpark names the training summary of a fit), a
    # DataFrame member reads the plan of the summary's `_summary_dataset`, which PySpark's fit sets,
    # as the summary dataset of the relation: without one, the AttributeError of the attribute,
    # before any request. The members of a value, of no DataFrame, read none.
    import pyspark.ml.connect as connect

    spark, service = ml_mock
    sent = _requests_but_deletes(service)
    for summary, member in [
        (connect.regression.LinearRegressionSummary("abc.summary"), lambda s: s.predictions),
        (connect.clustering.KMeansSummary("abc.summary.x"), lambda s: s.cluster),
        (
            connect.classification.BinaryLogisticRegressionTrainingSummary("abc.summary"),
            lambda s: s.roc,
        ),
        (
            connect.regression.GeneralizedLinearRegressionSummary("abc.summary"),
            lambda s: s.residuals(),
        ),
    ]:
        with pytest.raises(AttributeError) as error:
            member(summary)
        assert str(error.value) == (
            f"'{type(summary).__name__}' object has no attribute '_summary_dataset'"
        )
    assert _requests_but_deletes(service) == sent
    dataset = spark.range(3)
    summary = connect.regression.LinearRegressionSummary("abc.summary")
    summary._summary_dataset = dataset
    relation = _relation(summary.predictions).ml_relation
    assert relation.fetch.obj_ref.id == "abc"
    assert [method.method for method in relation.fetch.methods] == ["summary", "predictions"]
    assert relation.model_summary_dataset == _relation(dataset)
    for given in ("abc", "abc.x.summary"):
        relation = _relation(connect.regression.LinearRegressionSummary(given).predictions)
        assert not relation.ml_relation.HasField("model_summary_dataset"), given
    summary._summary_dataset = 5
    with pytest.raises(AttributeError, match="^'int' object has no attribute '_plan'$"):
        summary.predictions
    # The member of a value reads the `_java_obj` of the summary as it is at the call.
    summary._java_obj = "def"
    service.execute_payload = _ml_result(param=pb2.Expression.Literal(double=0.5))
    sent = len(service.execute_requests)
    assert summary.r2 == 0.5
    (command,) = _commands_but_deletes(service, sent)
    assert command.ml_command.fetch.obj_ref.id == "def"
    assert [method.method for method in command.ml_command.fetch.methods] == ["r2"]


def _pickle_error(value, protocol):
    # The type and the message of the error of the pickle of value at protocol.
    import pickle

    with pytest.raises(Exception) as error:
        pickle.dumps(value, protocol)
    return type(error.value), str(error.value)


def _lock_pickle_error(protocol):
    # The error of the pickle of a lock at protocol, which PySpark's pickle of a RemoteModelRef
    # raises: the PicklingError of the lookup of its class below protocol 2, a TypeError above.
    import threading

    return _pickle_error(threading.Lock(), protocol)


def _session_pickle_error(protocol):
    # The error of PySpark's pickle of the session of a DataFrame, of the local function of the
    # session's `__reduce__`, at every protocol.
    return AttributeError, "Can't pickle local object 'SparkSession.__reduce__.<locals>.creator'"


def test_a_fitted_model_and_the_dataframes_that_keep_it_are_not_pickled_as_pysparks(ml_mock):
    # As PySpark's pickle of a fitted model and of a DataFrame, which pickles its plan before its
    # session: the error of the lock of the model's RemoteModelRef at the protocol, of the model,
    # of a transform of it (`__source_transformer__`), of a DataFrame that reads one, and of a
    # DataFrame of a method of a model or of the summary of an evaluation, which keeps its model
    # (`__source_instance__`). A DataFrame of a method of a training summary reaches the session of
    # the summary's `_summary_dataset` first, which a plain pickle refuses, and cloudpickle, which
    # pickles it, the lock of the summary's `_remote_model_obj`. `copy.deepcopy` raises the
    # TypeError of the lock, after the requests of the schemas of the DataFrames that PySpark's
    # copy reaches (its lookup of `__deepcopy__` reads the columns of a DataFrame), and
    # `copy.copy` copies, sharing the model.
    import copy
    import pickle

    import pyspark.ml.connect as connect
    from pyspark import cloudpickle
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import functions as F

    spark, service = ml_mock
    frame = spark.createDataFrame([(1.0, Vectors.dense(0.0, 1.0))], ["label", "features"])
    lock = "^cannot pickle '_thread.lock' object$"
    service.execute_payload = _operator_info("model-of-dataframes")
    model = connect.regression.LinearRegression().fit(frame)
    kmeans = connect.clustering.KMeans(k=2).fit(frame)
    service.payload_for_request = {b"evaluate": _ml_result(summary="evaluated-of-dataframes")}
    evaluated = model.evaluate(frame)
    sent = _requests_but_deletes(service)
    for obj, blocked, analyses in [
        (model, _lock_pickle_error, 0),
        (model.transform(frame), _lock_pickle_error, 1),
        (model.transform(frame).select("label"), _lock_pickle_error, 1),
        (frame.join(model.transform(frame), "label"), _lock_pickle_error, 1),
        # A DataFrame that keeps a model only through a subquery of its expressions, as the
        # references of PySpark's plan keep the plans of the subqueries.
        (frame.where(model.transform(frame).exists()), _lock_pickle_error, 1),
        (
            frame.select(model.transform(frame).select(F.max("label")).scalar()),
            _lock_pickle_error,
            1,
        ),
        (
            frame.where(F.col("label").isin(model.transform(frame).select("label"))),
            _lock_pickle_error,
            1,
        ),
        (kmeans.summary.cluster, _lock_pickle_error, 1),
        (evaluated.predictions, _lock_pickle_error, 1),
        # A request of the lookup of `__deepcopy__` of the DataFrame, and two of the copy of the
        # `_summary_dataset` of its training summary, which copies (of the lookups of
        # `__deepcopy__` of the dataset and of `__setstate__` of its copy).
        (model.summary.predictions, _session_pickle_error, 3),
    ]:
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            assert _pickle_error(obj, protocol) == blocked(protocol), (obj, protocol)
        with pytest.raises(TypeError, match=lock):
            cloudpickle.dumps(obj)
        analyzed = len(service.analyze_requests)
        with pytest.raises(TypeError, match=lock):
            copy.deepcopy(obj)
        assert len(service.analyze_requests) - analyzed == analyses, obj
        assert type(copy.copy(obj)) is type(obj)
    plain = frame.select("label")
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        assert _pickle_error(plain, protocol) == _session_pickle_error(protocol)
    assert _relation(cloudpickle.loads(cloudpickle.dumps(plain))) == _relation(plain)
    assert _relation(copy.deepcopy(plain)) == _relation(plain)
    assert _requests_but_deletes(service) == sent


def test_the_dataframes_of_the_models_of_pysparks_own_classes_are_not_pickled_as_pysparks(ml_mock):
    # The plans of PySpark's own classes of `pyspark.ml` (`pyspark.ml.regression.LinearRegression`,
    # where the drop-in's own classes are those of `pyspark.ml.connect`) keep their models and
    # summaries (`__source_transformer__`, `__source_instance__`), which PySpark's pickle of a
    # DataFrame of such a plan (`DataFrame(plan, session)`) reaches before its session: the lock of
    # the model's RemoteModelRef, of a transform, of a DataFrame over one and of a DataFrame of the
    # summary of an evaluation; the session of the `_summary_dataset` of a training summary, which
    # a plain pickle refuses first, and then the lock of its `_remote_model_obj`, which cloudpickle
    # and `copy.deepcopy` reach; and the lock of a summary dataset that is a transform of the model.
    import copy
    import pickle

    from pyspark import cloudpickle
    from pyspark.ml.clustering import KMeansModel, KMeansSummary
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.regression import (
        LinearRegressionModel,
        LinearRegressionSummary,
        LinearRegressionTrainingSummary,
    )
    from pyspark.ml.util import RemoteModelRef

    spark, service = ml_mock
    frame = spark.createDataFrame([(1.0, Vectors.dense(0.0, 1.0))], ["label", "features"])
    lock = "^cannot pickle '_thread.lock' object$"
    # The models and the summaries of PySpark's `try_remote_fit` and `evaluate`.
    model = LinearRegressionModel(RemoteModelRef("pysparks-model"))
    trained = LinearRegressionTrainingSummary("pysparks-model.summary")
    trained._summary_dataset = frame
    trained._remote_model_obj = model._java_obj
    trained._remote_model_obj.add_ref()
    evaluated = LinearRegressionSummary("pysparks-evaluated")
    evaluated.__source_transformer__ = model
    kmeans = KMeansModel(RemoteModelRef("pysparks-kmeans"))
    clustered = KMeansSummary("pysparks-kmeans.summary")
    clustered._summary_dataset = kmeans.transform(frame)
    clustered._remote_model_obj = kmeans._java_obj
    clustered._remote_model_obj.add_ref()
    sent = _requests_but_deletes(service)
    for obj, blocked in [
        (model.transform(frame), _lock_pickle_error),
        (model.transform(frame).select("label"), _lock_pickle_error),
        (evaluated.predictions, _lock_pickle_error),
        (clustered.cluster, _lock_pickle_error),
        (trained.predictions, _session_pickle_error),
    ]:
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            assert _pickle_error(obj, protocol) == blocked(protocol), (obj, protocol)
        with pytest.raises(TypeError, match=lock):
            cloudpickle.dumps(obj)
        with pytest.raises(TypeError, match=lock):
            copy.deepcopy(obj)
    # A transform of a transformer of no model keeps no model, and pickles of its session.
    from pyspark.ml.feature import Binarizer

    binarized = Binarizer(threshold=0.5, inputCol="label", outputCol="b").transform(frame)
    assert _pickle_error(binarized, 2) == _session_pickle_error(2)
    assert _relation(cloudpickle.loads(cloudpickle.dumps(binarized))) == _relation(binarized)
    assert _requests_but_deletes(service) == sent


def test_a_summary_of_a_fit_or_an_evaluation_is_not_pickled_as_pysparks(ml_mock):
    # As PySpark's pickle of a summary of a server-side object, a plain object, of the error of the
    # first of its attributes that does not pickle, at every protocol: the lock of the
    # RemoteModelRef of the model that the summary of an evaluation keeps
    # (`__source_transformer__`), and that the summary dataset of a fit keeps when it reads a
    # transform of a model (`_summary_dataset` of the clustering models, the transform of the
    # training dataset by the model); else the AttributeError of the session of the training
    # DataFrame, which it pickles after its plan, and of cloudpickle, which pickles the session,
    # the lock of the model of the summary (`_remote_model_obj`). `copy.deepcopy` raises the
    # TypeError of the lock, after the requests of the schemas of the summary dataset that
    # PySpark's copy of it sends.
    import copy
    import pickle

    import pyspark.ml.connect as connect
    from pyspark import cloudpickle
    from pyspark.ml.linalg import Vectors

    spark, service = ml_mock
    frame = spark.createDataFrame([(1.0, Vectors.dense(0.0, 1.0))], ["label", "features"])
    lock = "^cannot pickle '_thread.lock' object$"
    service.execute_payload = _operator_info("model-of-a-summary")
    kmeans = connect.clustering.KMeans(k=2).fit(frame)
    regression = connect.regression.LinearRegression().fit(frame)
    # The training dataset of a stage after a model of a Pipeline is a transform of the model.
    after_model = connect.regression.LinearRegression().fit(kmeans.transform(frame).select("*"))
    service.payload_for_request = {b"evaluate": _ml_result(summary="evaluated-summary")}
    evaluated = regression.evaluate(frame)
    assert type(evaluated).__name__ == "LinearRegressionSummary"
    sent = _requests_but_deletes(service)
    for summary, blocked, analyses in [
        (kmeans.summary, _lock_pickle_error, 1),
        (regression.summary, _session_pickle_error, 2),
        (after_model.summary, _lock_pickle_error, 1),
        (evaluated, _lock_pickle_error, 0),
    ]:
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            assert _pickle_error(summary, protocol) == blocked(protocol), (summary, protocol)
        with pytest.raises(TypeError, match=lock):
            cloudpickle.dumps(summary)
        analyzed = len(service.analyze_requests)
        with pytest.raises(TypeError, match=lock):
            copy.deepcopy(summary)
        assert len(service.analyze_requests) - analyzed == analyses, summary
    assert _requests_but_deletes(service) == sent


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


def test_the_deletes_of_the_models_of_an_ended_test_are_not_recorded_in_a_later_test(ml_mock):
    # The fixture forgets the session of a test when the test ends. A model sends its Delete
    # without waiting for it, so the Delete of a model of the session can reach the server while
    # a later test runs: the server answers it and does not record it, so the later test counts
    # only its own requests.
    import time

    from pyspark.ml.connect import KMeans

    spark, service = ml_mock
    ended = SparkSession.builder.remote(service.remote).create()
    try:
        service.execute_payload = _operator_info("model-of-an-ended-test")
        model = KMeans(k=2).fit(ended.range(1))
        service.forget_session(ended.session_id)
        service.reset()
        del model
        gc.collect()
        deadline = time.time() + 5
        # Counted by kind: a late ReleaseExecute of the session can be counted too, so only the
        # ExecutePlan of the Delete is waited for and checked.
        while service.forgotten_count(ended.session_id, "execute") == 0 and time.time() < deadline:
            time.sleep(0.05)
        assert service.forgotten_count(ended.session_id, "execute") == 1
        assert service.execute_requests == []
        # The requests of a session that is not forgotten are still recorded.
        service.execute_payload = _operator_info("a")
        assert spark.client._delete_ml_cache(["a"]) == ["a"]
        (command,) = _commands(service)
        assert [ref.id for ref in command.ml_command.delete.obj_refs] == ["a"]
    finally:
        ended.stop()


def test_the_requests_of_an_ended_test_are_answered_but_not_recorded_in_a_later_test(ml_mock):
    # The fixture forgets the session of a test when the test ends, and the client sends some
    # requests without waiting for them, such as the ReleaseExecute after an execution, which can
    # reach the server during a later test. The server answers a request of a forgotten session
    # of any kind, and records none of them; a request of a session that is not forgotten is
    # recorded.
    import time

    from _mock_connect_server import parse_message

    def sessions(requests):
        return [parse_message(request).get(1, [b""])[0].decode() for request in requests]

    spark, service = ml_mock
    ended = SparkSession.builder.remote(service.remote).create()
    try:
        service.forget_session(ended.session_id)
        service.reset()
        ended.range(1).collect()
        assert ended.range(1).schema.names == ["id"]
        deadline = time.time() + 5
        while service.forgotten_count(ended.session_id, "release_execute") == 0 and time.time() < deadline:
            time.sleep(0.05)
        assert service.forgotten_count(ended.session_id, "release_execute") > 0
        assert service.forgotten_count(ended.session_id, "execute") == 1
        assert service.forgotten_count(ended.session_id, "analyze") == 1
        assert service.execute_requests == []
        assert service.release_execute_requests == []
        assert service.analyze_requests == []
        # Nor is it counted: the plan compression lookup of its first plan, and its ResultComplete.
        assert service.plan_compression_lookups == 0
        assert service.result_complete_operations == []
        spark.range(1).collect()
        deadline = time.time() + 5
        while not service.release_execute_requests and time.time() < deadline:
            time.sleep(0.05)
        assert sessions(service.execute_requests) == [spark.session_id]
        released = sessions(service.release_execute_requests)
        assert released and set(released) == {spark.session_id}
    finally:
        ended.stop()


def test_the_late_deletes_of_an_ended_test_take_no_queued_response_of_a_later_test(ml_mock):
    # The Delete of a model of a forgotten session that reaches the server while a later test runs
    # is answered, but not with a response that the later test queued (`_answer_fit`), which the
    # request of the later test would then miss.
    import time

    from pyspark.ml.connect import KMeans

    spark, service = ml_mock
    ended = SparkSession.builder.remote(service.remote).create()
    try:
        service.execute_payload = _operator_info("model-of-an-ended-test")
        model = KMeans(k=2).fit(ended.range(1))
        service.forget_session(ended.session_id)
        service.reset()
        service.execute_payload_queue = [_operator_info("model-of-the-later-test")]
        del model
        gc.collect()
        deadline = time.time() + 5
        while service.forgotten_count(ended.session_id, "execute") == 0 and time.time() < deadline:
            time.sleep(0.05)
        assert service.forgotten_count(ended.session_id, "execute") == 1
        later = KMeans(k=2).fit(spark.range(1))
        assert later is not None
        assert service.execute_payload_queue == []
    finally:
        ended.stop()


def test_the_late_requests_of_an_ended_test_take_no_queued_failure_or_gate_of_a_later_test(ml_mock):
    # As the queued responses above: the late Delete of a model of a forgotten session takes no
    # failure that the later test queued for its own request (`retry_info_failures`), and the
    # release of a forgotten session does not open the gate that the later test holds its own
    # execution at (`release_session_unblocks_execute`).
    import threading
    import time

    from pyspark.ml.connect import KMeans

    spark, service = ml_mock
    ended = SparkSession.builder.remote(service.remote).create()
    try:
        service.execute_payload = _operator_info("model-of-an-ended-test")
        model = KMeans(k=2).fit(ended.range(1))
        service.forget_session(ended.session_id)
        service.reset()
        service.retry_info_failures = [("RESOURCE_EXHAUSTED", 300)]
        del model
        gc.collect()
        deadline = time.time() + 5
        while service.forgotten_count(ended.session_id, "execute") == 0 and time.time() < deadline:
            time.sleep(0.05)
        assert service.forgotten_count(ended.session_id, "execute") == 1
        assert service.retry_info_failures == [("RESOURCE_EXHAUSTED", 300)]
        start = time.monotonic()
        spark.range(3).count()
        assert time.monotonic() - start >= 0.3
        assert len(service.execute_requests) == 2 and not service.retry_info_failures

        service.reset()
        service.block_execute = True
        service.release_session_unblocks_execute = True
        done = threading.Event()
        held = threading.Thread(target=lambda: (spark.range(1).collect(), done.set()), daemon=True)
        held.start()
        try:
            assert service.wait_execute_started(1)
            ended.stop()
            assert service.forgotten_count(ended.session_id, "release_session") == 1
            assert not done.wait(0.5), "the release of a forgotten session opened the gate"
        finally:
            service.release_response()
            held.join(10)
        assert done.is_set() and service.execute_gate_timeouts == 0
    finally:
        ended.stop()


def test_the_listener_requests_of_a_forgotten_session_are_answered_but_not_counted(ml_mock):
    # The listener counters, which the tests of the listener bus wait on, count the listener
    # requests of the sessions that are not forgotten only: a registration, a removal and the end
    # of the listener stream of a forgotten session, which an ended test can leave behind, are
    # answered, and counted in `forgotten_requests` only.
    import time

    from pyspark.sql.streaming.listener import StreamingQueryListener

    class Plain(StreamingQueryListener):
        def onQueryStarted(self, event):
            pass

        def onQueryProgress(self, event):
            pass

        def onQueryIdle(self, event):
            pass

        def onQueryTerminated(self, event):
            pass

    spark, service = ml_mock
    service.listener_bus_enabled = True
    ended = SparkSession.builder.remote(service.remote).create()
    try:
        service.forget_session(ended.session_id)
        listener = Plain()
        ended.streams.addListener(listener)
        ended.streams.removeListener(listener)
        assert service.forgotten_count(ended.session_id, "execute") == 2
        # The server ends the listener stream within a poll of 0.1 s of the removal.
        time.sleep(0.5)
        assert service.listener_subscriptions == 0
        assert service.listener_removals == 0
        assert service.listener_streams_finished == 0
        # The listener requests of a session that is not forgotten are still counted.
        listener = Plain()
        spark.streams.addListener(listener)
        spark.streams.removeListener(listener)
        assert service.wait_listener_streams_finished(1)
        assert service.listener_subscriptions == 1
        assert service.listener_removals == 1
    finally:
        ended.stop()


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

    from pyspark.sql.connect.proto._rust import Any

    spark, service = ml_mock
    client = spark.client

    def packed(graph_id):
        # The Any of the Rust protobuf runtime of pyspark.sql.connect.proto.
        extension = Any()
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


def test_global_user_context_extensions_are_a_list_that_the_requests_take_as_pyspark(ml_mock):
    # As PySpark Connect's SparkConnectClient, whose global_user_context_extensions is a list
    # attribute of the client, of the IDs and the Any that add_global_user_context_extension
    # appends, which its requests read: a change of the list, or a list that it is set to, is
    # what the requests carry. Its copies and pickles are plain lists.
    import copy

    from pyspark.sql.connect.client import SparkConnectClient
    from pyspark.sql.connect.proto._rust import Any

    spark, service = ml_mock
    client = spark.client

    def packed(graph_id):
        extension = Any()
        extension.Pack(pb2.PipelineAnalysisContext(dataflow_graph_id=graph_id))
        return extension

    def sent():
        command = pb2.Command()
        command.ml_command.clean_cache.SetInParent()
        client.execute_command(command)
        return list(_last_request_but_deletes(service).user_context.extensions)

    assert not hasattr(SparkConnectClient, "global_user_context_extensions")
    assert "global_user_context_extensions" in vars(client)
    first = packed("first")
    first_id = client.add_global_user_context_extension(first)
    extensions = client.global_user_context_extensions
    assert extensions == [(first_id, first)] and extensions[0][1] is first
    assert isinstance(extensions, list) and client.global_user_context_extensions is extensions
    assert sent() == [first]

    extensions.append(("global_appended", packed("appended")))
    assert sent() == [first, packed("appended")]
    del extensions[0]
    assert sent() == [packed("appended")]
    client.global_user_context_extensions = [("global_set", packed("set"))]
    assert sent() == [packed("set")]
    assert client.global_user_context_extensions is not extensions
    assert type(copy.copy(client.global_user_context_extensions)) is list
    copied = copy.deepcopy(client.global_user_context_extensions)
    assert type(copied) is list and copied == [("global_set", packed("set"))]
    # A pickle names a plain list (the Any of the Rust runtime of the list does not pickle).
    assert client.global_user_context_extensions.__reduce_ex__(2)[0] is list

    # Removing a global extension rebinds the list, as PySpark's, which a list read before keeps.
    before = client.global_user_context_extensions
    client.remove_user_context_extension("global_set")
    assert client.global_user_context_extensions == [] and before == [("global_set", packed("set"))]
    assert sent() == []
    client.add_global_user_context_extension(packed("again"))
    client.clear_user_context_extensions()
    assert client.global_user_context_extensions == [] and sent() == []


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
    from pyspark.sql.connect.session import SparkSession

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
    # responses by their fields, and the ExecutionInfo of the operation ID of the request.
    from pyspark.sql.metrics import ExecutionInfo

    spark, service = ml_mock
    command = pb2.Command()
    command.ml_command.clean_cache.SetInParent()
    service.execute_payload = _operator_info("ref-1")
    data, properties, info = spark.client.execute_command(command)
    assert data is None and isinstance(info, ExecutionInfo)
    assert list(properties) == ["ml_command_result"]
    assert properties["ml_command_result"].operator_info.obj_ref.id == "ref-1"
    request = _last_request_but_deletes(service)
    assert request.session_id == spark.session_id
    assert request.plan.command == command
    assert info.operation_id == request.operation_id

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


def _fit_warning_result(ref, warning):
    # The ExecutePlanResponse of a Fit whose model `ref` the server sends with the warning
    # `warning` (MlOperatorInfo.warning_message), as of a tree training that the memory control of
    # its ML cache stopped early.
    return _ml_result(
        operator_info=pb2.MlCommandResult.MlOperatorInfo(
            obj_ref=pb2.ObjectRef(id=ref), warning_message=warning
        )
    )


@pytest.fixture
def fit_warning_mock(_ml_mock_server):
    # A session of its own of the shared server, as of `ml_mock`, whose hook records the request of
    # each execution, but the Delete of a released model (which is sent without waiting, at any
    # time), with the records of the logger pyspark.ml.util, in their order, of any thread:
    # ("request", <a fit or fetch of its class or method, or the plan's kind>) and ("log", record).
    from pyspark.sql.connect.session import SparkSession as ConnectSession

    _, service = _ml_mock_server
    service.reset()
    events = []

    class Recording(ConnectSession.Hook):
        def on_execute_plan(self, request):
            command = request.plan.command.ml_command
            if command.HasField("fit"):
                events.append(("request", "fit " + command.fit.estimator.name.split(".")[-1]))
            elif command.HasField("fetch"):
                methods = ".".join(method.method for method in command.fetch.methods)
                events.append(("request", "fetch " + methods))
            elif not command.HasField("delete"):
                events.append(("request", request.plan.WhichOneof("op_type")))
            return request

    class Logged(logging.Handler):
        def emit(self, record):
            events.append(("log", record))

    logger = logging.getLogger("pyspark.ml.util")
    handler = Logged()
    logger.addHandler(handler)
    builder = ConnectSession.builder.remote(service.remote)
    spark = builder._registerHook(lambda session: Recording()).create()
    try:
        yield spark, service, events
    finally:
        logger.removeHandler(handler)
        service.forget_session(spark.session_id)
        spark.stop()


def test_ml_connect_fits_log_the_warning_of_the_server_as_pyspark(fit_warning_mock):
    # As official PySpark 4.3.0's try_remote_fit: the warning that the server sends with the model
    # of a fit (MlOperatorInfo.warning_message, of a tree training that the memory control of its
    # ML cache stopped early) is logged by its `_logger.warning(warning_msg)`, to the logger
    # pyspark.ml.util at WARNING, of the server's message as it is (of no arguments), from that
    # line of pyspark/ml/util.py, once the result of the Fit has arrived and before the model is
    # made of it (before the fetch of a forest's trees or of a discretizer's splits). So each fit
    # logs its own: of a param map, of each map of a list and of fitMultiple, and of each stage of
    # a Pipeline. A fit whose result has no warning, or an empty one, logs nothing, and so does a
    # fetch whose result has one.
    import inspect
    import threading

    import pyspark.ml.util as util
    from pyspark.ml.connect.feature import QuantileDiscretizer, StringIndexer
    from pyspark.ml.connect.pipeline import Pipeline
    from pyspark.ml.connect.regression import RandomForestRegressor

    spark, service, events = fit_warning_mock
    df = spark.range(1)
    lines, start = inspect.getsourcelines(util.try_remote_fit)
    line = start + next(i for i, text in enumerate(lines) if "_logger.warning(" in text)
    where = (util.try_remote_fit.__code__.co_filename, line, "wrapped")
    forest = (
        "The random forest training stops early because the model size (35016 bytes) exceeds "
        "threshold (24000 bytes)."
    )

    def logged(fit, *payloads):
        # The requests and the logged messages of `fit`, whose executions are answered by
        # `payloads` in turn, in their order; each record is PySpark's.
        service.execute_payload_queue = list(payloads)
        del events[:]
        fit()
        seen = []
        for kind, value in events:
            if kind == "log":
                assert (value.name, value.levelname) == ("pyspark.ml.util", "WARNING")
                assert value.args == () and value.exc_info is None
                assert value.thread == threading.get_ident()
                assert (value.pathname, value.lineno, value.funcName) == where
                value = value.msg
            seen.append((kind, value))
        return seen

    # A forest's fit, whose model reads its trees, of a result of a warning that is no fit's.
    trees = _fit_warning_result("tree-1", "not of a fit")
    fit, read = ("request", "fit RandomForestRegressor"), ("request", "fetch trees")
    first, second = _fit_warning_result("f", forest), _fit_warning_result("s", "second")
    assert logged(lambda: RandomForestRegressor().fit(df), first, trees) == [
        fit, ("log", forest), read
    ]
    estimator = RandomForestRegressor()
    assert logged(lambda: estimator.fit(df, {estimator.maxDepth: 3}), first, trees) == [
        fit, ("log", forest), read
    ]
    expected = [fit, ("log", forest), read, fit, ("log", "second"), read]
    maps = [{}, {estimator.maxDepth: 3}]
    assert logged(lambda: estimator.fit(df, maps), first, trees, second, trees) == expected
    fits = lambda: list(estimator.fitMultiple(df, maps))  # noqa: E731
    assert logged(fits, first, trees, second, trees) == expected

    # A discretizer's fit, whose Bucketizer is made of the splits that it reads of its model.
    doubles = [pb2.Expression.Literal(double=v) for v in [0.0, 0.5, 1.0]]
    splits = pb2.Expression.Literal(array=pb2.Expression.Literal.Array(elements=doubles))
    service.payload_for_request = {b"getSplits": _ml_result(param=splits)}
    discretizer = QuantileDiscretizer(inputCol="id")
    assert logged(lambda: discretizer.fit(df), _fit_warning_result("q", "of splits")) == [
        ("request", "fit QuantileDiscretizer"),
        ("log", "of splits"),
        ("request", "fetch getSplits"),
    ]
    service.payload_for_request = {}

    # The stages of a Pipeline, each of its own warning.
    indexer = StringIndexer(inputCol="x", outputCol="y")
    pipeline = Pipeline(stages=[RandomForestRegressor(predictionCol="p"), indexer])
    of_indexer = _fit_warning_result("i", "of an indexer")
    assert logged(lambda: pipeline.fit(df), first, trees, of_indexer) == [
        fit, ("log", forest), read, ("request", "fit StringIndexer"), ("log", "of an indexer")
    ]

    # No warning, and an empty one.
    for result in [_operator_info("plain"), _fit_warning_result("empty", "")]:
        assert logged(lambda: indexer.fit(df), result) == [("request", "fit StringIndexer")]


def test_ml_connect_one_vs_rest_logs_the_warning_of_each_class_fit_as_pyspark(fit_warning_mock):
    # As official PySpark 4.3.0's OneVsRest._fit, whose `classifier.fit` of each class runs in a
    # thread of its pool, of try_remote_fit: the warning of the Fit of each class is logged, of
    # the logger pyspark.ml.util, on the thread that fits the class (not the caller's), once that
    # Fit has returned and before the next class is fitted.
    import threading

    from pyspark.ml.connect.classification import DecisionTreeClassifier, OneVsRest

    spark, service, events = fit_warning_mock
    estimator = OneVsRest(classifier=DecisionTreeClassifier(), parallelism=1)
    # The labels' maximum is 1.0: two classes, whose Fits are told apart by their binary labels.
    service.query_payload = _label_query_result(estimator)
    service.payload_for_request = {
        b"mc2b$0": _fit_warning_result("class-0", "the warning of class 0"),
        b"mc2b$1": _fit_warning_result("class-1", "the warning of class 1"),
    }
    try:
        estimator.fit(spark.range(1))
    finally:
        service.query_payload = None
    fit = ("request", "fit DecisionTreeClassifier")
    assert [(kind, value if kind == "request" else value.msg) for kind, value in events] == [
        ("request", "root"),
        fit,
        ("log", "the warning of class 0"),
        fit,
        ("log", "the warning of class 1"),
    ]
    for record in [value for kind, value in events if kind == "log"]:
        assert (record.name, record.levelname) == ("pyspark.ml.util", "WARNING")
        assert record.thread != threading.get_ident()


def test_ml_connect_fit_raises_the_exception_of_logging_its_warning_as_pyspark(fit_warning_mock):
    # As official PySpark 4.3.0: the exception of the logging of the warning of a fit, of a filter
    # of the logger that raises, is the exception of the fit, as it is, which makes no model (it
    # reads no trees).
    from pyspark.ml.connect.regression import RandomForestRegressor

    spark, service, events = fit_warning_mock

    class Failing(logging.Filter):
        def filter(self, record):
            raise ValueError("the filter failed")

    logger = logging.getLogger("pyspark.ml.util")
    failing = Failing()
    logger.addFilter(failing)
    service.execute_payload_queue = [
        _fit_warning_result("forest", "stopped early"),
        _operator_info("tree-1"),
    ]
    try:
        with pytest.raises(ValueError, match="the filter failed"):
            RandomForestRegressor().fit(spark.range(1))
    finally:
        logger.removeFilter(failing)
    assert events == [("request", "fit RandomForestRegressor")]


def test_the_logging_of_a_fit_warning_needs_the_logging_call_of_pysparks_function():
    # The warning of a fit is logged from the line of the `_logger.warning(...)` call of PySpark's
    # try_remote_fit (`_native_classes._log_as`), of its nested `wrapped`: a function that makes no
    # such call is a ValueError, rather than a record of another location.
    import pyspark.ml.util as util
    from pyspark.ml import _native_classes

    with pytest.raises(ValueError) as error:
        _native_classes._log_as(util.try_remote_transform_relation, "_logger", "a warning")
    assert str(error.value) == "try_remote_transform_relation calls no _logger.warning"


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


def test_the_native_pipeline_iterates_its_stages_as_pysparks_fit():
    # PySpark's `Pipeline._fit` (pipeline.py:120-142) iterates the stages three times: to check
    # their types, to find the last estimator, and to fit them. So stages of an iterator, such as
    # a generator, which the first iteration consumes, fit a PipelineModel of no stages, as those
    # of PySpark's own Pipeline (vendored, `pyspark.ml.Pipeline`) do, and a list fits each stage.
    from pyspark.ml import Pipeline as PySparksPipeline
    from pyspark.ml.connect import Pipeline, Tokenizer
    from pyspark.testing.mlutils import MockDataset

    dataset = MockDataset()
    stage = Tokenizer(inputCol="a", outputCol="b")
    for cls in (Pipeline, PySparksPipeline):
        assert cls(stages=[stage]).fit(dataset).stages == [stage]
        for stages in (iter([stage]), (s for s in [stage]), map(lambda s: s, [stage])):
            assert cls(stages=stages).fit(dataset).stages == [], cls
        with pytest.raises(TypeError, match=r"^Cannot recognize a pipeline stage of type <class 'int'>\.$"):
            cls(stages=iter([1])).fit(dataset)


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
    # Without the protobuf package, pyspark.sql.connect.proto is of the Rust protobuf runtime:
    # pyspark.ml and the connect modules import, and a message builds and encodes as protobuf's.
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
        from pyspark import _pyspark
        assert issubclass(proto.Relation, _pyspark.ProtoMessage), proto.Relation.__mro__
        assert LogisticRegression(maxIter=3).getMaxIter() == 3
        relation = proto.Relation(range=proto.Range(start=1, end=3))
        assert proto.Relation.FromString(relation.SerializeToString()) == relation
        print(relation.SerializeToString().hex())
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    expected = pb2.Relation(range=pb2.Range(start=1, end=3)).SerializeToString().hex()
    assert result.stdout.strip() == expected


def test_session_hooks_rewrite_the_requests_of_executions_as_pyspark(_ml_mock_server):
    # As official PySpark 4.3.0's SparkSession.Hook: a session that the builder creates calls
    # each factory of _registerHook with itself, and the on_execute_plan of each hook with each
    # ExecutePlanRequest before it sends it, which keeps the operation ID of the request.
    from pyspark.sql.connect.session import SparkSession

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
    from pyspark.sql.connect.session import SparkSession

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

    assert exported is DecisionTreeClassifier and issubclass(DecisionTreeClassifier, native.DecisionTreeClassifier)
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

    assert exported is DecisionTreeRegressor and issubclass(DecisionTreeRegressor, native.DecisionTreeRegressor)
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
def test_gaussian_mixture_native_parameters_match_official_fixture(ml_mock, use_setters):
    import base64
    import inspect
    import json
    from pathlib import Path

    from pyspark import _pyspark as native
    from pyspark.ml.connect import GaussianMixture as exported
    from pyspark.ml.connect.clustering import GaussianMixture

    assert exported is GaussianMixture and issubclass(GaussianMixture, native.GaussianMixture)
    defaults = dict(
        featuresCol="features",
        predictionCol="prediction",
        k=2,
        probabilityCol="probability",
        tol=0.01,
        maxIter=100,
        seed=None,
        aggregationDepth=2,
        weightCol=None,
    )
    signature = inspect.signature(GaussianMixture)
    assert list(signature.parameters) == list(defaults)
    for name, value in defaults.items():
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default == value
    fixture = json.loads(
        (Path(__file__).resolve().parents[2] / "tests/golden/gaussian_mixture.json").read_text()
    )
    spark, service = ml_mock
    models = []
    _, params = _native_request(spark, service, GaussianMixture(), models)
    assert params == pb2.MlParams()
    for case in fixture["cases"]:
        values = dict(
            featuresCol="input",
            predictionCol="cluster",
            k=3,
            probabilityCol="probs",
            tol=1e-5,
            maxIter=50,
            seed=case["seed"],
            aggregationDepth=3,
            weightCol="weight",
        )
        if use_setters:
            estimator = GaussianMixture()
            for name, value in values.items():
                assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
        else:
            estimator = GaussianMixture(**values)
        operator, params = _native_request(spark, service, estimator, models)
        assert operator.name == "org.apache.spark.ml.clustering.GaussianMixture"
        assert params == pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))


def test_the_future_warning_of_compute_cost_is_attributed_as_pysparks():
    # PySpark's `BisectingKMeansModel.computeCost` warns before its `_call_java`, of its
    # `warnings.warn(...)`: of that line of `pyspark/ml/clustering.py` and of the module
    # `pyspark.ml.clustering`, whose filters take it, and of its registry, of which the "default"
    # action shows it once. A model of no server-side object then raises its AssertionError.
    import inspect
    import warnings

    import pyspark.ml.clustering as reference
    from pyspark.ml.connect import clustering

    function = reference.BisectingKMeansModel.computeCost
    lines, start = inspect.getsourcelines(function)
    line = start + next(i for i, text in enumerate(lines) if "warnings.warn(" in text)
    model = clustering.BisectingKMeansModel()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(AssertionError):
            model.computeCost(None)
    (warning,) = caught
    assert warning.category is FutureWarning and str(warning.message).startswith("Deprecated in 3.0.0.")
    assert (warning.filename, warning.lineno) == (function.__code__.co_filename, line)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=FutureWarning, module="pyspark.ml.clustering")
        with pytest.raises(AssertionError):
            model.computeCost(None)
    assert caught == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        warnings.simplefilter("default")
        vars(reference).pop("__warningregistry__", None)
        for _ in range(2):
            with pytest.raises(AssertionError):
                model.computeCost(None)
    assert len(caught) == 1


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_bisecting_kmeans_native_parameters_match_official_fixture(ml_mock, use_setters):
    import base64
    import inspect
    import json
    from pathlib import Path

    from pyspark import _pyspark as native
    from pyspark.ml.connect import BisectingKMeans as exported
    from pyspark.ml.connect.clustering import BisectingKMeans

    assert exported is BisectingKMeans and issubclass(BisectingKMeans, native.BisectingKMeans)
    defaults = dict(
        featuresCol="features",
        predictionCol="prediction",
        maxIter=20,
        seed=None,
        k=4,
        minDivisibleClusterSize=1.0,
        distanceMeasure="euclidean",
        weightCol=None,
    )
    signature = inspect.signature(BisectingKMeans)
    assert list(signature.parameters) == list(defaults)
    for name, value in defaults.items():
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default == value
    fixture = json.loads(
        (Path(__file__).resolve().parents[2] / "tests/golden/bisecting_kmeans.json").read_text()
    )
    spark, service = ml_mock
    models = []
    _, params = _native_request(spark, service, BisectingKMeans(), models)
    assert params == pb2.MlParams()
    for case in fixture["cases"]:
        values = dict(
            featuresCol="input",
            predictionCol="cluster",
            maxIter=50,
            seed=case["seed"],
            k=3,
            minDivisibleClusterSize=0.25,
            distanceMeasure="cosine",
            weightCol="weight",
        )
        if use_setters:
            estimator = BisectingKMeans()
            for name, value in values.items():
                assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
        else:
            estimator = BisectingKMeans(**values)
        operator, params = _native_request(spark, service, estimator, models)
        assert operator.name == "org.apache.spark.ml.clustering.BisectingKMeans"
        assert params == pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_naive_bayes_native_parameters_match_official_fixture(ml_mock, use_setters):
    import base64
    import inspect
    import json
    from pathlib import Path

    from pyspark import _pyspark as native
    from pyspark.ml.connect import NaiveBayes as exported
    from pyspark.ml.connect.classification import NaiveBayes

    assert exported is NaiveBayes and issubclass(NaiveBayes, native.NaiveBayes)
    defaults = dict(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        smoothing=1.0,
        modelType="multinomial",
        thresholds=None,
        weightCol=None,
    )
    signature = inspect.signature(NaiveBayes)
    assert list(signature.parameters) == list(defaults)
    for name, value in defaults.items():
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default == value
    fixture = json.loads(
        (Path(__file__).resolve().parents[2] / "tests/golden/naive_bayes.json").read_text()
    )
    spark, service = ml_mock
    models = []
    _, params = _native_request(spark, service, NaiveBayes(), models)
    assert params == pb2.MlParams()
    for case in fixture["cases"]:
        values = dict(
            featuresCol="input",
            labelCol="target",
            predictionCol="class",
            probabilityCol="probs",
            rawPredictionCol="scores",
            smoothing=0.25,
            modelType=case["model_type"],
            thresholds=[0.25, 0.5, 1.0],
            weightCol="weight",
        )
        if use_setters:
            estimator = NaiveBayes()
            for name, value in values.items():
                assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
        else:
            estimator = NaiveBayes(**values)
        operator, params = _native_request(spark, service, estimator, models)
        assert operator.name == "org.apache.spark.ml.classification.NaiveBayes"
        assert params == pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_gbt_regressor_native_parameters_match_official_fixture(ml_mock, use_setters):
    import base64
    import inspect
    import json
    from pathlib import Path

    import pyspark._pyspark as native
    from pyspark.ml.connect import GBTRegressor as exported
    from pyspark.ml.connect.regression import GBTRegressor

    assert exported is GBTRegressor and issubclass(GBTRegressor, native.GBTRegressor)
    defaults = dict(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        maxDepth=5,
        maxBins=32,
        minInstancesPerNode=1,
        minInfoGain=0.0,
        maxMemoryInMB=256,
        cacheNodeIds=False,
        subsamplingRate=1.0,
        checkpointInterval=10,
        lossType="squared",
        maxIter=20,
        stepSize=0.1,
        seed=None,
        impurity="variance",
        featureSubsetStrategy="all",
        validationTol=0.1,
        validationIndicatorCol=None,
        leafCol="",
        minWeightFractionPerNode=0.0,
        weightCol=None,
    )
    signature = inspect.signature(GBTRegressor)
    assert list(signature.parameters) == list(defaults)
    for name, default in defaults.items():
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default == default
    assert not hasattr(GBTRegressor(), "setValidationTol")
    fixture = json.loads(
        (Path(__file__).resolve().parents[2] / "tests/golden/gbt_regressor.json").read_text()
    )
    assert fixture["omitted_validation_tol"] == 0.01
    assert fixture["explicit_validation_tol"] == 0.1
    spark, service = ml_mock
    models = []
    assert _native_sent(spark, service, GBTRegressor(), models) == {}
    assert _native_sent(spark, service, GBTRegressor(validationTol=0.1), models) == {
        "validationTol": ("double", fixture["explicit_validation_tol"])
    }
    for case in fixture["cases"]:
        values = dict(
            featuresCol="input",
            labelCol="target",
            predictionCol="value",
            maxDepth=3,
            maxBins=16,
            minInstancesPerNode=2,
            minInfoGain=0.01,
            maxMemoryInMB=64,
            cacheNodeIds=True,
            subsamplingRate=0.75,
            checkpointInterval=-1,
            lossType="absolute",
            maxIter=4,
            stepSize=0.2,
            seed=case["seed"],
            impurity="variance",
            featureSubsetStrategy="all",
            validationIndicatorCol="validation",
            leafCol="leaves",
            minWeightFractionPerNode=0.1,
            weightCol="weight",
        )
        if use_setters:
            estimator = GBTRegressor(validationTol=0.05)
            for name, value in values.items():
                assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
        else:
            estimator = GBTRegressor(**values, validationTol=0.05)
        operator, params = _native_request(spark, service, estimator, models)
        assert operator.name == "org.apache.spark.ml.regression.GBTRegressor"
        assert params == pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))
    for value in [np.float64(0.05), 1, None]:
        sent = _native_sent(spark, service, GBTRegressor(validationTol=value), models)
        assert sent["validationTol"][0] == ("null" if value is None else "double")
    with pytest.raises(TypeError, match="Invalid param value"):
        GBTRegressor(validationTol="invalid")


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_gbt_classifier_native_parameters_match_official_fixture(ml_mock, use_setters):
    import base64
    import inspect
    import json
    from pathlib import Path

    import pyspark._pyspark as native
    from pyspark.ml.connect import GBTClassifier as exported
    from pyspark.ml.connect.classification import GBTClassifier

    assert exported is GBTClassifier and issubclass(GBTClassifier, native.GBTClassifier)
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


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_generalized_linear_regression_native_parameters_match_official_fixture(ml_mock, use_setters):
    import base64
    import inspect
    import json
    from pathlib import Path

    from pyspark import _pyspark as native
    from pyspark.ml.connect import GeneralizedLinearRegression as exported
    from pyspark.ml.connect.regression import GeneralizedLinearRegression

    assert exported is GeneralizedLinearRegression and issubclass(GeneralizedLinearRegression, native.GeneralizedLinearRegression)
    defaults = {'labelCol': 'label', 'featuresCol': 'features', 'predictionCol': 'prediction', 'family': 'gaussian', 'link': None, 'fitIntercept': True, 'maxIter': 25, 'tol': 1e-06, 'regParam': 0.0, 'weightCol': None, 'solver': 'irls', 'linkPredictionCol': None, 'variancePower': 0.0, 'linkPower': None, 'offsetCol': None, 'aggregationDepth': 2}
    signature = inspect.signature(GeneralizedLinearRegression)
    assert list(signature.parameters) == list(defaults)
    for name, value in defaults.items():
        assert signature.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default == value
    fixture = json.loads((Path(__file__).resolve().parents[2] / "tests/golden/generalized_linear_regression.json").read_text())
    spark, service = ml_mock
    models = []
    for case in fixture["cases"]:
        if use_setters:
            estimator = GeneralizedLinearRegression()
            for name, value in case["values"].items():
                assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
        else:
            estimator = GeneralizedLinearRegression(**case["values"])
        operator, params = _native_request(spark, service, estimator, models)
        assert operator.name == "org.apache.spark.ml.regression.GeneralizedLinearRegression"
        assert params == pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))


@pytest.mark.parametrize("name", ["FMClassifier", "MultilayerPerceptronClassifier"])
@pytest.mark.parametrize("use_setters", [False, True])
def test_remaining_classifier_native_parameters_match_official_fixture(ml_mock, name, use_setters):
    import base64
    import inspect
    import json
    from pathlib import Path

    from pyspark import _pyspark as native
    from pyspark.ml.connect import classification
    from pyspark.ml.linalg import Vectors

    cls = getattr(classification, name)
    assert issubclass(cls, getattr(native, name))
    defaults = {'FMClassifier': {'featuresCol': 'features', 'labelCol': 'label', 'predictionCol': 'prediction', 'probabilityCol': 'probability', 'rawPredictionCol': 'rawPrediction', 'factorSize': 8, 'fitIntercept': True, 'fitLinear': True, 'regParam': 0.0, 'miniBatchFraction': 1.0, 'initStd': 0.01, 'maxIter': 100, 'stepSize': 1.0, 'tol': 1e-06, 'solver': 'adamW', 'thresholds': None, 'seed': None}, 'MultilayerPerceptronClassifier': {'featuresCol': 'features', 'labelCol': 'label', 'predictionCol': 'prediction', 'maxIter': 100, 'tol': 1e-06, 'seed': None, 'layers': None, 'blockSize': 128, 'stepSize': 0.03, 'solver': 'l-bfgs', 'initialWeights': None, 'probabilityCol': 'probability', 'rawPredictionCol': 'rawPrediction'}, 'OneVsRest': {'featuresCol': 'features', 'labelCol': 'label', 'predictionCol': 'prediction', 'rawPredictionCol': 'rawPrediction', 'classifier': None, 'weightCol': None, 'parallelism': 1}}[name]
    signature = inspect.signature(cls)
    assert list(signature.parameters) == list(defaults)
    for key, value in defaults.items():
        assert signature.parameters[key].default == value
        assert signature.parameters[key].kind == inspect.Parameter.KEYWORD_ONLY
    fixture = json.loads((Path(__file__).resolve().parents[2] / "tests/golden/remaining_classifiers.json").read_text())
    spark, service = ml_mock
    models = []
    for case in fixture["cases"]:
        if case["class"] != name:
            continue
        values = dict(case["values"])
        if "initialWeights" in values:
            vector = values["initialWeights"]
            values["initialWeights"] = Vectors.dense(vector["dense"]) if "dense" in vector else Vectors.sparse(vector["size"], vector["indices"], vector["values"])
        if use_setters:
            estimator = cls()
            for key, value in values.items():
                assert getattr(estimator, "set" + key[0].upper() + key[1:])(value) is estimator
        else:
            thresholds = values.pop("thresholds", None) if name == "MultilayerPerceptronClassifier" else None
            estimator = cls(**values)
            if thresholds is not None:
                estimator.setThresholds(thresholds)
        operator, params = _native_request(spark, service, estimator, models)
        assert operator.name == "org.apache.spark.ml.classification." + name
        assert params == pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))


def test_one_vs_rest_signature_and_native_classifier_validation():
    import inspect

    from pyspark import _pyspark as native
    from pyspark.ml.connect import OneVsRest as exported
    from pyspark.ml.connect.classification import LinearSVC, OneVsRest

    assert OneVsRest is exported and issubclass(exported, native.OneVsRest)
    defaults = {'FMClassifier': {'featuresCol': 'features', 'labelCol': 'label', 'predictionCol': 'prediction', 'probabilityCol': 'probability', 'rawPredictionCol': 'rawPrediction', 'factorSize': 8, 'fitIntercept': True, 'fitLinear': True, 'regParam': 0.0, 'miniBatchFraction': 1.0, 'initStd': 0.01, 'maxIter': 100, 'stepSize': 1.0, 'tol': 1e-06, 'solver': 'adamW', 'thresholds': None, 'seed': None}, 'MultilayerPerceptronClassifier': {'featuresCol': 'features', 'labelCol': 'label', 'predictionCol': 'prediction', 'maxIter': 100, 'tol': 1e-06, 'seed': None, 'layers': None, 'blockSize': 128, 'stepSize': 0.03, 'solver': 'l-bfgs', 'initialWeights': None, 'probabilityCol': 'probability', 'rawPredictionCol': 'rawPrediction'}, 'OneVsRest': {'featuresCol': 'features', 'labelCol': 'label', 'predictionCol': 'prediction', 'rawPredictionCol': 'rawPrediction', 'classifier': None, 'weightCol': None, 'parallelism': 1}}["OneVsRest"]
    signature = inspect.signature(OneVsRest)
    assert list(signature.parameters) == list(defaults)
    for key, value in defaults.items():
        assert signature.parameters[key].default == value
        assert signature.parameters[key].kind == inspect.Parameter.KEYWORD_ONLY
    estimator = OneVsRest()
    assert estimator.setClassifier(LinearSVC()) is estimator
    for key, value in dict(featuresCol="x", labelCol="y", predictionCol="p", rawPredictionCol="r", weightCol="w", parallelism=2).items():
        assert getattr(estimator, "set" + key[0].upper() + key[1:])(value) is estimator
    # PySpark's classifier param takes any object (`TypeConverters.identity`).
    assert type(OneVsRest().setClassifier(object()).getClassifier()) is object
    with pytest.raises(TypeError):
        OneVsRest(LinearSVC())
    with pytest.raises(TypeError):
        OneVsRest(unknown=True)
    # PySpark's fit reads the classifier first, which raises its KeyError when it is unset.
    with pytest.raises(KeyError, match="classifier"):
        OneVsRest().fit(None)


@pytest.mark.parametrize("name,value", [("layers",[2,1.5]),("layers","abc"),("initialWeights","abc")])
def test_multilayer_perceptron_rejects_invalid_parameter_types(name,value):
    from pyspark.ml.connect.classification import MultilayerPerceptronClassifier
    with pytest.raises(TypeError):
        MultilayerPerceptronClassifier(**{name:value})


def test_one_vs_rest_defers_base_param_validation_until_fit():
    from pyspark.ml.connect.classification import MultilayerPerceptronClassifier, OneVsRest

    base = MultilayerPerceptronClassifier(layers=[])
    estimator = OneVsRest(classifier=base)
    assert estimator.setClassifier(base) is estimator
    base.setLayers([2, 2])
    assert estimator.setClassifier(None) is estimator
    # The classifier of None passes, and the fit then reads the dataset's max label, as PySpark's.
    with pytest.raises(AttributeError, match="agg"):
        estimator.fit(None)


@pytest.mark.parametrize("use_setter", [False, True])
@pytest.mark.parametrize("parallelism", [0, -1, None, 2**100, -(2**100)])
def test_one_vs_rest_reads_the_dataset_before_its_parallelism(parallelism, use_setter):
    from pyspark.ml.connect.classification import LinearSVC, OneVsRest

    if use_setter:
        estimator = OneVsRest(classifier=LinearSVC()).setParallelism(parallelism)
    else:
        estimator = OneVsRest(classifier=LinearSVC(), parallelism=parallelism)
    # PySpark's fit reads the max label first, by `dataset.agg`, before its ThreadPool of
    # min(parallelism, numClasses) processes, and never sends the param: an object that is not a
    # DataFrame raises the AttributeError of `agg` before anything is sent, whatever the
    # parallelism, and an int of any size is taken. The live tests check the rest of the order.
    with pytest.raises(AttributeError, match="^'NoneType' object has no attribute 'agg'$"):
        estimator.fit(None)


@pytest.mark.parametrize("value", [2**64, -(2**64), 2**63, -(2**63) - 1, 7])
def test_one_vs_rest_keeps_a_parallelism_of_any_size_as_pyspark(value):
    # PySpark's `parallelism` is an int of any size, which its fit compares in Python and never
    # sends, and which its writer saves as it is: the native object keeps it as it is too (an int
    # beyond a long, of the repr that the core keeps of it), so that its getter, its save and its
    # load are of the int itself.
    from pyspark import _pyspark as native
    from pyspark.ml.connect.classification import OneVsRest

    assert native.OneVsRest(parallelism=value).getParallelism() == value
    assert native.OneVsRest().setParallelism(value).getParallelism() == value
    assert native.OneVsRest().setParams(parallelism=value).getParallelism() == value
    assert OneVsRest(parallelism=value).getParallelism() == value


@pytest.mark.parametrize("name", ["AFTSurvivalRegression", "IsotonicRegression", "FMRegressor"])
@pytest.mark.parametrize("use_setters", [False, True])
def test_remaining_regressor_native_parameters_match_official_fixture(ml_mock, name, use_setters):
    import base64
    import inspect
    import json
    from pathlib import Path

    from pyspark import _pyspark as native
    from pyspark.ml import connect
    from pyspark.ml.connect import regression

    cls = getattr(regression, name)
    assert cls is getattr(connect, name) and issubclass(cls, getattr(native, name))
    fixture = json.loads((Path(__file__).resolve().parents[2] / "tests/golden/remaining_regressors.json").read_text())
    defaults = fixture["defaults"][name]
    signature = inspect.signature(cls)
    assert list(signature.parameters) == list(defaults)
    for key, value in defaults.items():
        assert signature.parameters[key].default == value
        assert signature.parameters[key].kind == inspect.Parameter.KEYWORD_ONLY
    spark, service = ml_mock
    models = []
    for case in fixture["cases"]:
        if case["class"] != name:
            continue
        estimator = cls() if use_setters else cls(**case["values"])
        if use_setters:
            for key, value in case["values"].items():
                assert getattr(estimator, "set" + key[0].upper() + key[1:])(value) is estimator
        operator, params = _native_request(spark, service, estimator, models)
        assert operator.name == "org.apache.spark.ml.regression." + name
        assert params == pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))


@pytest.fixture(params=["arrays", "empty", "vectors", "subclass", "missing-columns"])
def pandas_ml_input(request):
    import pandas as pd
    from pyspark.ml.linalg import Vectors

    if request.param == "empty":
        return pd.DataFrame({"features": pd.Series([], dtype=object)})
    if request.param == "vectors":
        return pd.DataFrame({
            "features": [Vectors.dense([1.0, 0.0]), Vectors.sparse(2, {1: 2.0})],
            "label": [0.0, 1.0],
        })
    if request.param == "missing-columns":
        return pd.DataFrame({0: [object()]})

    class Frame(pd.DataFrame):
        pass

    cls = Frame if request.param == "subclass" else pd.DataFrame
    return cls({"features": [[1.0, 0.0], [0.0, 2.0]], "label": [0.0, 1.0],
                "prediction": [0.0, 1.0]}, index=[7, 9])


def _native_ml_method(name):
    import pyspark.ml.connect as connect

    instance = getattr(connect, name)()
    method = ("assignClusters" if name == "PowerIterationClustering" else
              "evaluate" if name.endswith("Evaluator") else
              "transform" if name == "VectorAssembler" else "fit")
    return getattr(instance, method)


def _assert_rejects_pandas(method, pdf):
    # A fit, transform or evaluate reads the sparkSession of its dataset first, as pyspark.ml does,
    # so a pandas DataFrame raises the AttributeError of its own sparkSession. The helper call of
    # PowerIterationClustering.assignClusters, whose dataset pyspark.ml's serialization sends as a
    # literal, raises the UNSUPPORTED_DATA_TYPE of that conversion, as pyspark.ml's does (see
    # test_native_ml_other_invalid_inputs).
    from pyspark.errors import PySparkTypeError

    if method.__name__ == "assignClusters":
        with pytest.raises(PySparkTypeError) as error:
            method(dataset=pdf)
        assert error.value.getCondition() == "UNSUPPORTED_DATA_TYPE"
        assert error.value.getMessageParameters() == {"data_type": type(pdf).__name__}
        return
    if type(method.__self__).__name__ == "OneVsRest":
        # The fit of a OneVsRest of no classifier is PySpark's own OneVsRest._fit's, as for any
        # dataset that is not a Spark DataFrame: it reads the classifier first, and raises the
        # KeyError of that unset param (test_native_one_vs_rest_fit_of_pandas_is_pysparks has
        # a classifier).
        with pytest.raises(KeyError) as error:
            method(dataset=pdf)
        assert error.value.args[0].name == "classifier"
        return
    with pytest.raises(AttributeError) as error:
        method(dataset=pdf)
    assert str(error.value) == f"'{type(pdf).__name__}' object has no attribute 'sparkSession'"


def test_native_one_vs_rest_fit_of_pandas_is_pysparks(pandas_ml_input):
    # The native OneVsRest fits a dataset that is not a Spark DataFrame by PySpark's own
    # OneVsRest._fit, which reads its max label by the dataset's own agg: a pandas DataFrame
    # raises the error of that expression of PySpark's, before any request.
    import pyspark.ml.connect as connect

    with pytest.raises(Exception) as expected:
        int(pandas_ml_input.agg({"label": "max"}).head()["max(label)"]) + 1
    estimator = connect.OneVsRest(classifier=connect.LinearSVC())
    with pytest.raises(type(expected.value)) as error:
        estimator.fit(pandas_ml_input)
    assert error.value.args == expected.value.args


def test_native_one_vs_rest_fits_a_subclass_of_a_native_classifier_by_pysparks_fit(ml_mock):
    # The native fit takes a classifier of exactly a native class. PySpark's own OneVsRest._fit fits
    # a Python subclass of one, of `classifier.fit(trainingDataset, paramMap)` for each class, so
    # that the subclass's `_fit` runs, of the binary label of each class, and sends no fit of its
    # own. Its model is PySpark's OneVsRestModel of the subclass's models.
    import pyspark.ml.connect as connect
    from pyspark.ml.classification import OneVsRestModel
    from pyspark.ml.connect.classification import LinearSVCModel

    spark, service = ml_mock
    calls = []

    class Overriding(connect.LinearSVC):
        def _fit(self, dataset):
            calls.append(self.getLabelCol())
            return LinearSVCModel()

    estimator = connect.OneVsRest(classifier=Overriding())
    service.query_payload = _label_query_result(estimator)
    try:
        model = estimator.fit(spark.range(1))
    finally:
        service.query_payload = None
    assert sorted(calls) == ["mc2b$0", "mc2b$1"]
    assert type(model) is OneVsRestModel
    assert [type(m) for m in model.models] == [LinearSVCModel] * 2
    assert not any(c.ml_command.HasField("fit") for c in _commands(service))


def test_stop_words_remover_load_default_stop_words_asserts_an_active_session():
    # As PySpark's invoke_helper_attr, which asserts the active session of the ML helper call.
    from pyspark.ml.connect.feature import StopWordsRemover

    assert SparkSession.getActiveSession() is None
    with pytest.raises(AssertionError) as error:
        StopWordsRemover.loadDefaultStopWords("english")
    assert type(error.value) is AssertionError and str(error.value) == ""


@pytest.mark.parametrize("name", [*_NATIVE_PARAMS, "OneVsRest"])
def test_native_ml_rejects_pandas_without_active_session(name, pandas_ml_input):
    assert SparkSession.getActiveSession() is None
    _assert_rejects_pandas(_native_ml_method(name), pandas_ml_input)
    assert SparkSession.getActiveSession() is None


def test_assign_clusters_asserts_an_active_session_before_its_dataset(pandas_ml_input):
    # As PySpark's invoke_helper_relation, which asserts the active session of the helper call
    # before it serializes the arguments: without one, assignClusters raises that AssertionError,
    # also of a pandas DataFrame (with one, the UNSUPPORTED_DATA_TYPE of its conversion).
    assert SparkSession.getActiveSession() is None
    with pytest.raises(AssertionError) as error:
        _native_ml_method("PowerIterationClustering")(dataset=pandas_ml_input)
    assert type(error.value) is AssertionError and str(error.value) == ""
    assert SparkSession.getActiveSession() is None


@pytest.mark.parametrize("estimator_name", ["LogisticRegression", "KMeans"])
def test_native_ml_rejects_pandas_before_any_requests(ml_mock, pandas_ml_input, estimator_name):
    import pyspark.ml.connect as connect

    spark, service = ml_mock
    estimator = getattr(connect, estimator_name)()
    _answer_fit(service, estimator)
    model = estimator.fit(spark.range(1))

    def requests():
        return (_requests_but_deletes(service), len(service.analyze_requests),
                len(service.config_requests))

    sent = requests()
    methods = [_native_ml_method(name) for name in [*_NATIVE_PARAMS, "OneVsRest", "PowerIterationClustering"]]
    methods += [model.transform, connect.OneVsRest(parallelism=0).fit,
                connect.LogisticRegression(maxIter=2**100).fit,
                connect.VectorAssembler(inputCols=[]).transform]
    for method in methods:
        _assert_rejects_pandas(method, pandas_ml_input)
        assert requests() == sent


def test_the_models_of_objects_of_the_server_are_of_their_constructors_as_pysparks(ml_mock):
    # PySpark makes the model of an object that a fetch returns of its class's constructor, of
    # no params (`DecisionTreeClassificationModel(RemoteModelRef(m))` of a tree of an ensemble,
    # `LocalLDAModel(RemoteModelRef(...))` of `toLocal`): its `JavaModel.__init__` makes its uid
    # and its param maps, of the defaults in the order in which its classes set them, which is
    # not of their names (as the maps of a fit or a load are, of `_resetUid`).
    import pyspark.ml.classification as classification_reference
    import pyspark.ml.clustering as clustering_reference
    from pyspark.ml.connect import classification, clustering

    spark, service = ml_mock
    forest = classification.RandomForestClassifier(numTrees=1)
    _answer_fit(service, forest, ref="forest-model")
    (tree,) = forest.fit(spark.range(1)).trees
    service.execute_payload_queue = [_operator_info("lda-model"), _operator_info("local-model")]
    local = clustering.LDA(k=2, optimizer="em").fit(spark.range(1)).toLocal()
    for model, reference in [
        (tree, classification_reference.DecisionTreeClassificationModel),
        (local, clustering_reference.LocalLDAModel),
    ]:
        name = reference.__name__
        assert type(model).__name__ == name and model.uid.startswith(name + "_")
        order = [p.name for p in reference()._defaultParamMap]
        assert order != sorted(order)
        assert [p.name for p in model._defaultParamMap] == order and not model._paramMap
    assert _relation(tree.transform(spark.range(1))).ml_relation.transform.obj_ref.id == "tree-1"
    assert _relation(local.transform(spark.range(1))).ml_relation.transform.obj_ref.id == "local-model"


def test_the_maps_of_the_model_of_a_fit_are_of_pysparks_try_remote_fit(ml_mock):
    # PySpark's `try_remote_fit` (ml/util.py:207-208) makes the maps of the model of a JVM
    # estimator's fit of the model's constructor, then of its `_resetUid`, which rebuilds the
    # defaults of the constructor in the order of the names of the params, and then of the
    # estimator's `_copyValues`, which adds the estimator's defaults that the model's class does
    # not set and sets the estimator's params, each of their names. So an ALSModel's defaults are
    # its constructor's `blockSize` and `predictionCol`, and then ALS's `coldStartStrategy`,
    # `itemCol` and `userCol`, as official pyspark-client 4.3.0 gives them; a model whose
    # constructor sets every default of its estimator (a LogisticRegressionModel's) has them of
    # their names. A copy (`Params.copy`) is of the names.
    import pyspark.ml.recommendation as reference
    from pyspark.ml.connect.classification import LogisticRegression
    from pyspark.ml.connect.recommendation import ALS

    spark, service = ml_mock
    assert [p.name for p in reference.ALSModel()._defaultParamMap] == ["predictionCol", "blockSize"]
    order = ["blockSize", "predictionCol", "coldStartStrategy", "itemCol", "userCol"]
    for kwargs, set_names in [
        ({}, []),
        ({"userCol": "u", "itemCol": "i", "coldStartStrategy": "drop"}, ["coldStartStrategy", "itemCol", "userCol"]),
    ]:
        estimator = ALS(rank=2, **kwargs)
        _answer_fit(service, estimator, ref="als-model")
        model = estimator.fit(spark.range(1))
        assert [p.name for p in model._defaultParamMap] == order
        assert [p.name for p in model.extractParamMap()] == order
        assert [p.name for p in model._paramMap] == set_names
        assert [p.name for p in model.copy()._defaultParamMap] == sorted(order)
    estimator = LogisticRegression(maxIter=2)
    _answer_fit(service, estimator, ref="lr-model")
    names = [p.name for p in estimator.fit(spark.range(1))._defaultParamMap]
    assert names == sorted(names) and "threshold" in names


def test_the_model_of_a_one_vs_rest_fit_is_of_its_constructor_and_warns_as_pysparks(ml_mock):
    # PySpark's `OneVsRest._fit` makes its model of `OneVsRestModel(models)`, whose constructor sets
    # the defaults in the order of its MRO, and then of `_copyValues`, which sets the params of the
    # estimator in the order of their names. Of a classifier of no `weightCol`, it warns of its
    # `warnings.warn(...)` first: of that line of `pyspark/ml/classification.py` and of the module
    # `pyspark.ml.classification`, whose filters take it.
    import inspect
    import warnings

    import pyspark.ml.classification as reference
    from pyspark.ml.connect import classification

    spark, service = ml_mock
    base = classification.MultilayerPerceptronClassifier(layers=[1, 2])
    estimator = classification.OneVsRest(classifier=base, weightCol="w", featuresCol="f")
    function = reference.OneVsRest._fit
    lines, start = inspect.getsourcelines(function)
    line = start + next(i for i, text in enumerate(lines) if "warnings.warn(" in text)
    _answer_fit(service, estimator, ref="class-model")
    service.query_payload = _label_query_result(estimator)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = estimator.fit(spark.range(1))
    (warning,) = caught
    assert warning.category is UserWarning
    assert str(warning.message) == f"weightCol is ignored, as it is not supported by {base.uid} now."
    assert (warning.filename, warning.lineno) == (function.__code__.co_filename, line)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=UserWarning, module="pyspark.ml.classification")
        estimator.fit(spark.range(1))
    service.query_payload = None
    assert caught == []
    order = ["predictionCol", "featuresCol", "labelCol", "rawPredictionCol"]
    assert [p.name for p in reference.OneVsRestModel([])._defaultParamMap] == order
    assert [p.name for p in model._defaultParamMap] == order
    assert [p.name for p in model._paramMap] == ["classifier", "featuresCol", "weightCol"]
    assert [p.name for p in classification.OneVsRestModel(model.models)._defaultParamMap] == order


@pytest.mark.parametrize("name", ["LDA", "PowerIterationClustering"])
def test_remaining_clustering_signatures_and_native_exports(name):
    import inspect
    import json
    from pathlib import Path

    import pyspark.ml.connect as connect
    from pyspark import _pyspark
    from pyspark.ml.connect import clustering

    cls = getattr(clustering, name)
    assert cls is getattr(connect, name) and issubclass(cls, getattr(_pyspark, name))
    fixture = json.loads((Path(__file__).parents[2] / "tests/golden/remaining_clustering.json").read_text())
    parameters = inspect.signature(cls).parameters
    assert list(parameters) == list(fixture["defaults"][name])
    for param, default in fixture["defaults"][name].items():
        assert parameters[param].kind == inspect.Parameter.KEYWORD_ONLY
        assert parameters[param].default == default
        assert hasattr(cls, "set" + param[0].upper() + param[1:])


def test_the_dataframe_methods_of_models_and_summaries_send_their_positional_arguments_only(ml_mock):
    # PySpark's `try_remote_attribute_relation` of `LDAModel.describeTopics`,
    # `Word2VecModel.getVectors` and the `residuals` of a GLR summary sends the positional
    # arguments of a call only: it drops keyword arguments, and checks no arity, in a lazy
    # DataFrame of no request, which the server rejects when it is analyzed of arguments that its
    # method does not take.
    from pyspark.ml.connect import clustering, feature, regression

    spark, service = ml_mock
    frame = spark.range(1)
    service.execute_payload = _operator_info("lda-model")
    lda = clustering.LDA(k=2).fit(frame)
    service.execute_payload = _operator_info("word2vec-model")
    word2vec = feature.Word2Vec(inputCol="text").fit(frame)
    service.execute_payload = _operator_info("glr-model")
    summary = regression.GeneralizedLinearRegression().fit(frame).summary
    sent = _requests_but_deletes(service)
    cases = [
        (lambda: lda.describeTopics(), "lda-model", ["describeTopics"], []),
        (lambda: lda.describeTopics(1), "lda-model", ["describeTopics"], [("integer", 1)]),
        (lambda: lda.describeTopics(maxTermsPerTopic=1), "lda-model", ["describeTopics"], []),
        (lambda: lda.describeTopics(1, 2, x=3), "lda-model", ["describeTopics"], [("integer", 1), ("integer", 2)]),
        (lambda: word2vec.getVectors(x=1), "word2vec-model", ["getVectors"], []),
        (lambda: word2vec.getVectors(1), "word2vec-model", ["getVectors"], [("integer", 1)]),
        (lambda: summary.residuals(), "glr-model", ["summary", "residuals"], []),
        (lambda: summary.residuals(residualsType="pearson"), "glr-model", ["summary", "residuals"], []),
        (
            lambda: summary.residuals("pearson", 1),
            "glr-model",
            ["summary", "residuals"],
            [("string", "pearson"), ("integer", 1)],
        ),
    ]
    for call, ref, methods, args in cases:
        fetch = _relation(call()).ml_relation.fetch
        assert fetch.obj_ref.id == ref
        assert [method.method for method in fetch.methods] == methods
        assert [_literal_value(arg.param) for arg in fetch.methods[-1].args] == args
    assert _requests_but_deletes(service) == sent
    # Of no server-side object, PySpark's AssertionError of any arguments.
    for call in [
        lambda: clustering.LocalLDAModel().describeTopics(1, 2, x=3),
        lambda: feature.Word2VecModel().getVectors(1, x=2),
    ]:
        with pytest.raises(AssertionError):
            call()


def test_pic_helper_request_matches_official_and_is_lazy(ml_mock):
    import base64
    import json
    from pathlib import Path

    from pyspark.ml.connect import PowerIterationClustering

    spark, service = ml_mock
    fixture = json.loads((Path(__file__).parents[2] / "tests/golden/remaining_clustering.json").read_text())
    df = spark.range(1)
    sent = (_requests_but_deletes(service), len(service.analyze_requests), len(service.config_requests))
    for case in fixture["pic_fetch"]:
        result = PowerIterationClustering(**case["values"]).assignClusters(dataset=df)
        actual = _relation(result).ml_relation.fetch
        assert actual.methods[0].args[0].input == _relation(df)
        actual.methods[0].args[0].input.Clear()
        assert actual == pb2.Fetch.FromString(base64.b64decode(case["fetch_b64"]))
    assert (_requests_but_deletes(service), len(service.analyze_requests), len(service.config_requests)) == sent


def test_pic_parameter_converters_and_deferred_overflow(ml_mock):
    from pyspark.ml.connect import PowerIterationClustering

    spark, service = ml_mock
    names = ["k", "maxIter", "initMode", "srcCol", "dstCol", "weightCol"]
    for index, name in enumerate(names):
        setter = "set" + name[0].upper() + name[1:]
        kind = "int" if index < 2 else "str"
        for value, expected in _NATIVE_CONVERTED[kind]:
            original = PowerIterationClustering()
            configured = getattr(original, setter)(value)
            assert configured is original
            for instance in [configured, PowerIterationClustering(**{name: value})]:
                fetch = _relation(instance.assignClusters(spark.range(1))).ml_relation.fetch
                assert _literal_value(fetch.methods[0].args[index + 1].param) == expected
        for value, _ in _NATIVE_REJECTED[kind]:
            with pytest.raises(TypeError):
                getattr(PowerIterationClustering(), setter)(value)
            with pytest.raises(TypeError):
                PowerIterationClustering(**{name: value})
    with pytest.raises(TypeError, match="keyword arguments"):
        PowerIterationClustering(2)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        PowerIterationClustering(seed=42)
    for name in ["k", "maxIter"]:
        with pytest.raises(PySparkValueError) as error:
            PowerIterationClustering(**{name: 2**100}).assignClusters(spark.range(1))
        assert error.value.getCondition() == "VALUE_NOT_BETWEEN"
    assert _requests_but_deletes(service) == 0


def test_native_class_falls_back_only_when_the_native_module_is_absent(monkeypatch):
    # As the live tests' helper: the reference class is run only when pyspark._pyspark is absent.
    # A native module that is present but fails to load, or one of its imports that is missing,
    # raises its own error, which is not the reference.
    import builtins
    import importlib

    from _native_classes import native_class

    reference = importlib.import_module("pyspark.ml.clustering").KMeans
    real_import = builtins.__import__

    def failing_import(error):
        def _import(name, *args, **kwargs):
            if name == "pyspark._pyspark":
                raise error
            return real_import(name, *args, **kwargs)

        return _import

    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "pyspark._pyspark", None)
        found = native_class("pyspark.ml.connect.clustering", "KMeans", "pyspark.ml.clustering")
        assert found is reference
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", failing_import(ImportError("undefined symbol: boom")))
        with pytest.raises(ImportError, match="^undefined symbol: boom$") as error:
            native_class("pyspark.ml.connect.clustering", "KMeans", "pyspark.ml.clustering")
        assert type(error.value) is ImportError
    with monkeypatch.context() as patch:
        missing = ModuleNotFoundError("No module named 'numpy'", name="numpy")
        patch.setattr(builtins, "__import__", failing_import(missing))
        with pytest.raises(ModuleNotFoundError, match="numpy") as error:
            native_class("pyspark.ml.connect.clustering", "KMeans", "pyspark.ml.clustering")
        assert error.value is missing
