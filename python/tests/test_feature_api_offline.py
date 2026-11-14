"""The Params API, the getters and the members of the native feature transformers and estimators,
without a server: their cases (`_feature_api_cases.offline_cases`) against tests/golden/feature_api_cases.json,
from official PySpark (see scripts/capture_feature_api_cases_golden.py). Official PySpark runs them
too, as the reference (see `_native_classes`).

The cases run with an active session, as their capture did: a mock server answers the ML helper
calls that the constructor of StopWordsRemover makes for its defaults (its locale and the English
stop words), as the server of the capture did.
"""

import json
from pathlib import Path

import pytest
from _feature_api_cases import ESTIMATORS, TRANSFORMERS, classes, offline_cases
from _mock_connect_server import mock_connect_session
from pyspark.loose_version import LooseVersion

# The generated pyspark.sql.connect.proto, of Apache Spark v4.3.0, needs protobuf >= 6.33.5.
_protobuf = pytest.importorskip("google.protobuf")
if LooseVersion(_protobuf.__version__) < LooseVersion("6.33.5"):
    pytest.skip("pyspark.sql.connect.proto needs protobuf >= 6.33.5", allow_module_level=True)

from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.connect import proto as pb2  # noqa: E402

# The English stop words that `StopWordsRemover.loadDefaultStopWords("english")` returns on the
# server of the capture.
ENGLISH_STOP_WORDS = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "i'd", "you'd",
    "he'd", "she'd", "we'd", "they'd", "i'm", "you're", "he's", "she's", "it's", "we're",
    "they're", "i've", "we've", "you've", "they've", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "don't", "doesn't", "didn't", "won't", "wouldn't", "shan't",
    "shouldn't", "mustn't", "can't", "couldn't", "cannot", "could", "here's", "how's", "let's",
    "ought", "that's", "there's", "what's", "when's", "where's", "who's", "why's", "would"
]

NAMES = TRANSFORMERS + ESTIMATORS

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/feature_api_cases.json").read_text()
)


def _result(literal):
    # The ExecutePlanResponse of the value of a fetch of the ML helper.
    return pb2.ExecutePlanResponse(
        ml_command_result=pb2.MlCommandResult(param=literal)
    ).SerializeToString()


@pytest.fixture(scope="module")
def mock():
    with mock_connect_session() as pair:
        yield pair


@pytest.fixture
def session(mock):
    _, service = mock
    service.reset()
    # The helper calls of the constructor of StopWordsRemover: its locale, then its stop words.
    stop_words = pb2.Expression.Literal(
        array=pb2.Expression.Literal.Array(
            element_type=pb2.DataType(string=pb2.DataType.String()),
            elements=[pb2.Expression.Literal(string=word) for word in ENGLISH_STOP_WORDS],
        )
    )
    service.payload_for_request = {
        b"stopWordsRemoverGetDefaultOrUS": _result(pb2.Expression.Literal(string="en_US")),
        b"stopWordsRemoverLoadDefaultStopWords": _result(stop_words),
    }
    spark = SparkSession.builder.remote(service.remote).getOrCreate()
    yield spark
    service.forget_session(spark.session_id)
    spark.stop()


@pytest.mark.parametrize("name", NAMES)
def test_offline_cases_are_pyspark_mls(name, session):
    ours = offline_cases(name)
    golden = CASES[name]["offline"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)


def test_stop_words_remover_constructor_needs_an_active_session():
    # PySpark's constructor fetches the defaults of the active session, which its `assert` checks:
    # without one, a bare AssertionError, of the native class and of PySpark's.
    assert SparkSession.getActiveSession() is None
    with pytest.raises(AssertionError) as error:
        classes("StopWordsRemover")()
    assert str(error.value) == ""
