# SPDX-License-Identifier: Apache-2.0
"""The pyspark.ml.evaluation API of the native evaluators, against a live Connect server: the
evaluate cases (with and without a param map), isLargerBetter and the persistence of each evaluator
(`_evaluation_api_cases.live_cases`) against tests/golden/evaluation_api_cases.json, from official
PySpark (see scripts/capture_evaluation_api_cases_golden.py). Official PySpark runs them too, as the
reference (`_native_classes.native_class`).
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from _evaluation_api_cases import CLASSES, _cls, live_cases
from pyspark.sql import SparkSession

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/evaluation_api_cases.json").read_text()
)


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live evaluation API tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.mark.parametrize("name", CLASSES)
def test_live_cases_are_pyspark_mls(name, session):
    # The saves go to a scratch directory of the local file system, of the server that the tests
    # run against on the same machine, as in CI.
    ours = live_cases(name, session, tempfile.mkdtemp(prefix="evaluation-api-cases-"))
    golden = CASES[name]["live"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)


@pytest.mark.parametrize("name", CLASSES)
def test_objects_of_this_client_are_of_the_classes_of_pyspark_mls(name, session):
    # Every evaluator that this client makes is of the class of its module (of PySpark's on official
    # PySpark), which is a subclass of every class of the MRO of PySpark's class
    # (test_ml_api_offline.py): an evaluator, its copy (of a param map too) and its load.
    def of_its_class(obj):
        assert type(obj) is _cls(name), type(obj)
        return obj

    evaluator = of_its_class(_cls(name)())
    of_its_class(evaluator.copy())
    of_its_class(evaluator.copy({evaluator.metricName: evaluator.getMetricName()}))
    base = tempfile.mkdtemp(prefix="evaluation-api-classes-")
    try:
        evaluator.write().overwrite().save(base + "/evaluator")
        of_its_class(_cls(name).load(base + "/evaluator"))
        of_its_class(_cls(name).read().load(base + "/evaluator"))
    finally:
        shutil.rmtree(base, ignore_errors=True)
