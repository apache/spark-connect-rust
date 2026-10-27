#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preserve actionable pytest diagnostics without changing parity gate results."""

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "run_official_tests", Path(__file__).resolve().parents[1] / "run_official_tests.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ParityLogTests(unittest.TestCase):
    def test_collection_error_is_logged_and_still_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spark_py = root / "python"
            test_file = spark_py / "pyspark/sql/tests/connect/test_example.py"
            result = subprocess.CompletedProcess(
                [], 2, "collection traceback", "ImportError: dependency"
            )
            with (
                patch.dict(os.environ, {"RUST_PARITY_LOG_DIR": td, "RUST_PYSPARK_SO": "test.so"}),
                patch.object(runner.subprocess, "run", return_value=result),
            ):
                counts, failures = runner.run_ours(test_file, spark_py, "sc://localhost", [], 10)
            self.assertEqual(counts["rc"], 2)
            self.assertEqual(failures, [])
            logs = list(root.rglob("*.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("collection traceback", logs[0].read_text())
            self.assertIn("ImportError: dependency", logs[0].read_text())

    def test_retries_and_drift_logs_are_retained_separately(self):
        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict(os.environ, {"RUST_PARITY_LOG_DIR": td}),
        ):
            spark_py = Path(td) / "python"
            test_file = spark_py / "tests/test_example.py"
            with patch.object(runner.time, "time_ns", side_effect=[1, 2, 3]):
                runner.write_test_log(test_file, spark_py, "first", False)
                runner.write_test_log(test_file, spark_py, "retry", False)
                runner.write_test_log(test_file, spark_py, "drift", True)
            self.assertEqual(
                {p.name: p.read_text() for p in Path(td).rglob("*.log")},
                {
                    "test_example.tests.1.log": "first",
                    "test_example.tests.2.log": "retry",
                    "test_example.drift.3.log": "drift",
                },
            )

    def test_timeout_preserves_partial_byte_output(self):
        with tempfile.TemporaryDirectory() as td:
            spark_py = Path(td) / "python"
            test_file = spark_py / "test_example.py"
            error = subprocess.TimeoutExpired(
                [], 10, output=b"partial output", stderr=b"partial error"
            )
            with (
                patch.dict(os.environ, {"RUST_PARITY_LOG_DIR": td, "RUST_PYSPARK_SO": "test.so"}),
                patch.object(runner.subprocess, "run", side_effect=error),
            ):
                counts, failures = runner.run_ours(test_file, spark_py, "sc://localhost", [], 10)
            self.assertTrue(counts["timeout"])
            self.assertEqual(counts["rc"], -1)
            self.assertEqual(failures, [])
            log = next(Path(td).rglob("*.log")).read_text()
            self.assertIn("partial output", log)
            self.assertIn("partial error", log)
            self.assertIn("TIMEOUT after 10s", log)


if __name__ == "__main__":
    unittest.main()
