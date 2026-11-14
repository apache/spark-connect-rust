# SPDX-License-Identifier: Apache-2.0
"""The pyspark.ml.evaluation API of the native evaluators, without a server.

- The cases of PySpark's `Params` API, of `evaluate` with a param map that is refused before any
  request, and of `isLargerBetter` (`_evaluation_api_cases.params_api_cases`) against
  tests/golden/evaluation_api_cases.json (from scripts/capture_evaluation_api_cases_golden.py).
- The `evaluate` of a param map: it evaluates a copy of the evaluator, as PySpark's
  `Evaluator.evaluate` does, so that the request sends the params of the copy and the evaluator
  keeps its own.
"""

import json
from pathlib import Path

import pytest
from _evaluation_api_cases import EVALUATORS, MODULE, REFERENCE, params_api_cases
from _native_classes import native_class
from pyspark.sql.connect import proto as pb2
from test_ml_offline import (
    _commands_but_deletes,
    _literal_value,
    _ml_result,
    _native_sent,
)
from test_ml_offline import (
    _ml_mock_server as _ml_mock_server,
)
from test_ml_offline import (
    ml_mock as _ml_fixture,
)

ml_mock = _ml_fixture

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / "tests/golden/evaluation_api_cases.json").read_text())


@pytest.mark.parametrize("name", list(EVALUATORS))
def test_params_api_cases_are_pyspark_mls(name):
    ours = params_api_cases(name)
    golden = CASES[name]["params_api"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)


@pytest.mark.parametrize("name", list(EVALUATORS))
def test_evaluate_of_a_param_map_evaluates_a_copy(ml_mock, name):
    # As PySpark's Evaluator.evaluate: the request of a param map is the one of the copy with that
    # map (of the same uid), and the evaluator itself sends what it sent before.
    spark, service = ml_mock
    cls = native_class(MODULE, name, REFERENCE)
    df = spark.range(1)
    evaluator = cls()
    before = _native_sent(spark, service, evaluator, [])
    alt = {getattr(evaluator, key): value for key, value in EVALUATORS[name]["alt"].items()}

    sent = len(service.execute_requests)
    service.execute_payload = _ml_result(param=pb2.Expression.Literal(double=0.5))
    assert evaluator.evaluate(df, alt) == 0.5
    evaluate = _commands_but_deletes(service, sent)[0].ml_command.evaluate
    assert evaluate.evaluator.uid == evaluator.uid
    assert evaluate.evaluator.name == f"org.apache.spark.ml.evaluation.{name}"
    assert {key: _literal_value(v) for key, v in evaluate.params.params.items()} == _native_sent(
        spark, service, evaluator.copy(alt), []
    )
    assert _native_sent(spark, service, evaluator, []) == before
