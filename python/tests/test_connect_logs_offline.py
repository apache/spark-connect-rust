# SPDX-License-Identifier: Apache-2.0
"""The records of the logger of Spark Connect (`pyspark.sql.connect.logging`), against the mock
server of _mock_connect_server.py: of each case of `_connect_log_cases`, the logger, the level and
the message of each record, in order, against official PySpark's of tests/golden/connect_logs.json
(from scripts/capture_connect_logs_golden.py), normalized of what is of a run; and, of a protobuf
whose descriptors have `label`, the text of each truncated plan, against PySpark's of protobuf
6.33.5 of tests/golden/connect_logs_protobuf6.json.

Official PySpark runs these tests too, as the reference of the goldens.
"""

import json
from pathlib import Path

import pytest
from _connect_log_cases import CASES, TRUNCATED, run_case, truncated_plans
from _mock_connect_server import mock_connect_session

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads((ROOT / "tests/golden/connect_logs.json").read_text())
GOLDEN_PROTOBUF_6 = json.loads((ROOT / "tests/golden/connect_logs_protobuf6.json").read_text())


def _truncates():
    # Whether `_truncate` truncates a message of this environment, of descriptors of `label`.
    import pyspark.sql.connect.proto as pb2

    return hasattr(pb2.Relation.DESCRIPTOR.fields_by_name["sql"], "label")


def _of_truncated_plans(records):
    # The records of truncated plans of the text of the plan left out.
    def of(message):
        truncated = TRUNCATED.match(message)
        return truncated.group(1) + "<plan>" if truncated else message

    return [[name, level, of(message)] for name, level, message in records]


@pytest.fixture(scope="module")
def mock_server():
    pytest.importorskip("grpc")
    with mock_connect_session() as (_, service):
        yield service


def test_the_golden_is_of_pyspark_4_3_0_and_of_every_case():
    assert GOLDEN["pyspark"] == "4.3.0"
    assert sorted(GOLDEN["cases"]) == sorted(CASES)
    assert (GOLDEN_PROTOBUF_6["pyspark"], GOLDEN_PROTOBUF_6["protobuf"]) == ("4.3.0", "6.33.5")
    assert sorted(GOLDEN_PROTOBUF_6["cases"]) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_records_of_the_logger_of_spark_connect_are_pysparks(mock_server, name):
    # As PySpark Connect's client, of the same records, of the same levels and messages, in the
    # same order: the plans, the requests and the responses of each call, of their text of
    # `_proto_to_string`, the settings and the work of the plan compression, and the errors.
    records, returned = run_case(name, mock_server.remote, mock_server)
    expected_records, expected_returned = GOLDEN["cases"][name]
    assert returned == expected_returned
    if _truncates():
        records, expected_records = map(_of_truncated_plans, (records, expected_records))
    assert records == expected_records


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_truncated_plans_are_pysparks_of_a_protobuf_of_labels(mock_server, name):
    # As PySpark's `_proto_to_string(p, True)` of a protobuf whose descriptors have `label`, as
    # 6.33.5, of which `_truncate` cuts each plan to 8 levels and its strings and bytes short: of
    # the text of each truncated plan, which protobuf 7 does not truncate.
    if not _truncates():
        pytest.skip("the descriptors of this protobuf have no `label`")
    records, _ = run_case(name, mock_server.remote, mock_server)
    assert truncated_plans(records) == GOLDEN_PROTOBUF_6["cases"][name]
