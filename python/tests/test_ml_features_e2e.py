# SPDX-License-Identifier: Apache-2.0
"""All feature stages, tested through native bindings unless FEATURE_REFERENCE=1."""
import gc
import os

import pytest
from _ml_features import GOLDEN, assert_values, configured, wire_schema
from pyspark.ml.functions import array_to_vector
from pyspark.sql import SparkSession

REFERENCE = os.environ.get("FEATURE_REFERENCE") == "1"


def algorithm(name):
    if REFERENCE:
        from pyspark.ml import feature
    else:
        from pyspark import _pyspark
        from pyspark.ml.connect import feature
        assert getattr(feature, name) is getattr(_pyspark, name)
    return getattr(feature, name)


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def frame(session):
    return session.sql(GOLDEN["sql"]).withColumn("features", array_to_vector("xs")).withColumn("rawFeatures", array_to_vector("xs")).coalesce(1)


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["id"])
@pytest.mark.parametrize("setters", [False, True], ids=["constructor", "setters"])
def test_features_match_official_outputs_schema_and_lazy_lifetime(frame, case, setters):
    name = case["name"]
    stage = configured(algorithm(name), case["values"], setters)
    model = stage.fit(frame) if GOLDEN["api"][name]["estimator"] else stage
    output = model.transform(frame).orderBy("id").select(*case["columns"])
    # A lazy DataFrame must own both its cached model and the input plan.
    del model, stage
    gc.collect()
    rows = output.collect()
    if name == "PCA":
        # Eigenvector signs may differ across BLAS implementations.
        rows = [[abs(x) for x in row[0]] for row in rows]
        expected = [[abs(x) for x in row[0]] for row in case["rows"]]
        assert_values(rows, expected)
    else:
        assert_values(rows, case["rows"])
    assert wire_schema(output.schema.jsonValue()) == wire_schema(case["schema"])


@pytest.mark.parametrize("name", ["StandardScaler", "MaxAbsScaler", "MinMaxScaler", "RobustScaler"])
def test_scalers_require_vectors_and_reject_output_collisions(frame, name):
    from pyspark.errors import PySparkException
    with pytest.raises(PySparkException):
        algorithm(name)(inputCol="xs", outputCol="out").fit(frame)
    with pytest.raises(PySparkException):
        algorithm(name)(inputCol="features", outputCol="features").fit(frame).transform(frame).collect()


def test_independent_fits_keep_their_training_data(frame):
    from pyspark.sql import functions as F
    estimator = algorithm("StringIndexer")(inputCol="category", outputCol="out", stringOrderType="alphabetAsc")
    first = estimator.fit(frame)
    second = estimator.fit(frame.withColumn("category", F.lit("z")))
    expected = first.transform(frame).select("out")
    del second
    gc.collect()
    assert sorted(r.out for r in expected.collect()) == [0., 0., 0., 1., 1., 1., 2., 2.]


@pytest.mark.parametrize("name", ["StandardScaler", "MaxAbsScaler", "MinMaxScaler", "RobustScaler", "Normalizer", "ElementwiseProduct", "IDF", "VectorSlicer", "VarianceThresholdSelector"])
def test_sparse_vectors_match_dense_reference(frame, session, name):
    from pyspark.ml.linalg import Vectors
    rows = frame.select("id", "features").orderBy("id").collect()
    sparse = session.createDataFrame([(r.id, Vectors.sparse(4, {i: float(v) for i, v in enumerate(r.features) if v != 0})) for r in rows], ["id", "features"])
    case = next(c for c in GOLDEN["cases"] if c["id"] == name)
    stage = configured(algorithm(name), case["values"])
    model = stage.fit(sparse) if GOLDEN["api"][name]["estimator"] else stage
    assert_values(model.transform(sparse).orderBy("id").select("out").collect(), case["rows"])


def test_scaler_transforms_held_out_values_using_fitted_statistics(session):
    from pyspark.ml.linalg import Vectors
    train = session.createDataFrame([(Vectors.dense([1., 0.]),), (Vectors.sparse(2, {0: 3.}),)], ["features"])
    held = session.createDataFrame([(Vectors.dense([6., 4.]),)], ["features"])
    model = algorithm("MaxAbsScaler")(inputCol="features", outputCol="out").fit(train)
    assert_values(model.transform(held).select("out").collect(), [[[2., 4.]]])


def test_stop_words_remover_loads_the_default_stop_words_of_the_server(session):
    # As PySpark's StopWordsRemover.loadDefaultStopWords, which calls the ML helper of the active
    # session, and whose list a remover takes, as with the words added to the default ones.
    from pyspark.errors import IllegalArgumentException

    remover = algorithm("StopWordsRemover")
    english = remover.loadDefaultStopWords("english")
    assert type(english) is list and len(english) == 181
    assert english[:4] == ["i", "me", "my", "myself"]
    assert len(remover.loadDefaultStopWords(language="french")) == 156
    frame = session.createDataFrame([(["I", "like", "spark", "a", "lot"],)], ["words"])
    stage = remover(inputCol="words", outputCol="out", stopWords=english + ["spark"])
    assert stage.transform(frame).collect()[0].out == ["like", "lot"]
    with pytest.raises(IllegalArgumentException, match="klingon is not in the supported language list"):
        remover.loadDefaultStopWords("klingon")
