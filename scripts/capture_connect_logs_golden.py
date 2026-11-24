# SPDX-License-Identifier: Apache-2.0
"""Capture official PySpark 4.3.0's records of the logger of Spark Connect
(``pyspark.sql.connect.logging``) of the cases of python/tests/_connect_log_cases.py, against the
mock server of python/tests/_mock_connect_server.py: the logger, the level and the normalized
message of each record of each case, and what the case returned.
python/tests/test_connect_logs_offline.py compares the drop-in's with them.

Run with official PySpark 4.3.0 (``pyspark-client``, of grpcio), without a server, from the
repository::

    python scripts/capture_connect_logs_golden.py --out tests/golden/connect_logs.json

With ``--truncated-plans``, of a protobuf whose descriptors have ``label`` (6.33.5, which
PySpark's ``_truncate`` reads), it captures the messages of the records of truncated plans only,
of each case, which protobuf 7 cannot truncate::

    python scripts/capture_connect_logs_golden.py --truncated-plans \
        --out tests/golden/connect_logs_protobuf6.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--truncated-plans", action="store_true")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "python" / "tests"))
    import pyspark
    from _connect_log_cases import capture, truncated_plans
    from _mock_connect_server import mock_connect_session

    with mock_connect_session() as (_, service):
        cases = capture(service.remote, service)
    if args.truncated_plans:
        import google.protobuf
        import pyspark.sql.connect.proto as pb2

        if not hasattr(pb2.Relation.DESCRIPTOR.fields_by_name["sql"], "label"):
            parser.error("the descriptors of this protobuf have no `label`")
        cases = {name: truncated_plans(records) for name, (records, _) in cases.items()}
        golden = {
            "pyspark": pyspark.__version__,
            "protobuf": google.protobuf.__version__,
            "cases": cases,
        }
        count = sum(len(messages) for messages in cases.values())
    else:
        golden = {"pyspark": pyspark.__version__, "cases": cases}
        count = sum(len(records) for records, _ in cases.values())
    Path(args.out).write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    print(f"{count} records of {len(cases)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
