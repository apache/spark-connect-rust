# SPDX-License-Identifier: Apache-2.0
"""The arguments of the recommendations of ALSModel, against a live Connect server, as PySpark's.

PySpark's recommendForAllUsers, recommendForAllItems, recommendForUserSubset and
recommendForItemSubset are of `try_remote_attribute_relation`, which sends the server's method of
the positional arguments only. So no arity is checked (the DataFrame of a wrong one fails when it
is executed), keyword arguments are dropped, and a model of no server-side object raises an
AssertionError whatever its arguments. Official PySpark runs the tests too, as the reference
(`_native_classes.native_class`).
"""

import os

import pytest
from _native_classes import native_class

# The arguments of each method, of their kinds, and PySpark's names of them.
METHODS = {
    "recommendForAllUsers": (["count"], ["numItems"]),
    "recommendForAllItems": (["count"], ["numUsers"]),
    "recommendForUserSubset": (["users", "count"], ["dataset", "numItems"]),
    "recommendForItemSubset": (["items", "count"], ["dataset", "numUsers"]),
}


def cls(name):
    return native_class("pyspark.ml.connect.recommendation", name, "pyspark.ml.recommendation")


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live ML tests")
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def fitted(session):
    """A fitted model, and the values of the arguments of its methods."""
    rows = [(0, 0, 4.0), (0, 1, 2.0), (1, 1, 3.0), (1, 2, 4.0), (2, 1, 1.0), (2, 2, 5.0)]
    ratings = session.createDataFrame(rows, ["user", "item", "rating"])
    als = cls("ALS")(rank=2, maxIter=2, seed=1, userCol="user", itemCol="item", ratingCol="rating")
    values = {
        "count": 2,
        "users": ratings.select("user").distinct(),
        "items": ratings.select("item").distinct(),
    }
    return als.fit(ratings), values


def arguments(method, values):
    return [values[kind] for kind in METHODS[method][0]]


@pytest.mark.parametrize("method", list(METHODS))
def test_a_model_of_no_server_object_raises_an_assertion_error(method, fitted):
    args = arguments(method, fitted[1])
    for given in ([], args[:-1], args, args + [3]):
        with pytest.raises(AssertionError):
            getattr(cls("ALSModel")(), method)(*given)


@pytest.mark.parametrize("method", list(METHODS))
def test_a_wrong_arity_fails_when_the_dataframe_is_executed(method, fitted):
    model, values = fitted
    args = arguments(method, values)
    for given in (args[:-1], args + [3]):
        # The call sends nothing and checks nothing: its DataFrame fails when it is executed.
        recommendations = getattr(model, method)(*given)
        with pytest.raises(Exception, match="NoSuchMethodException"):
            recommendations.collect()
    assert getattr(model, method)(*args).count() == 3


@pytest.mark.parametrize("method", list(METHODS))
def test_keyword_arguments_are_dropped(method, fitted):
    model, values = fitted
    args = arguments(method, values)
    names = METHODS[method][1]
    with pytest.raises(Exception, match="NoSuchMethodException"):
        getattr(model, method)(**dict(zip(names, args))).collect()
    # A positional argument with a keyword one: only the positional one is sent.
    if len(args) == 2:
        with pytest.raises(Exception, match="NoSuchMethodException"):
            getattr(model, method)(args[0], **{names[1]: args[1]}).collect()
