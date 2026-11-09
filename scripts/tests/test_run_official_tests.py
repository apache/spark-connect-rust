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


class DropinTests(unittest.TestCase):
    def _touch(self, root: Path, rel: str, text: str = "") -> Path:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_discovery_covers_the_connect_suites_minus_local_cluster_files(self):
        with tempfile.TemporaryDirectory() as td:
            spark_py = Path(td) / "python"
            for rel in (
                "pyspark/sql/tests/connect/test_parity_types.py",
                "pyspark/sql/tests/connect/arrow/test_parity_arrow_map.py",
                "pyspark/sql/tests/connect/client/test_client.py",
                "pyspark/sql/tests/connect/test_parity_resources.py",
                "pyspark/sql/tests/connect/client/test_artifact.py",
                "pyspark/sql/tests/connect/client/test_artifact_localcluster.py",
                "pyspark/ml/tests/connect/test_parity_als.py",
                "pyspark/errors/tests/connect/test_parity_traceback.py",
                "pyspark/logger/tests/connect/test_parity_logger.py",
                # Outside the scope: classic sql, resource, top-level and ml/pandas tests.
                "pyspark/sql/tests/test_dataframe.py",
                "pyspark/resource/tests/test_connect_resources.py",
                "pyspark/tests/test_context.py",
                "pyspark/pandas/tests/connect/test_parity_frame.py",
            ):
                self._touch(spark_py, rel)
            found = [p.relative_to(spark_py).as_posix() for p in runner.discover_dropin(spark_py)]
        self.assertEqual(
            found,
            [
                "pyspark/errors/tests/connect/test_parity_traceback.py",
                "pyspark/logger/tests/connect/test_parity_logger.py",
                "pyspark/ml/tests/connect/test_parity_als.py",
                "pyspark/sql/tests/connect/arrow/test_parity_arrow_map.py",
                "pyspark/sql/tests/connect/client/test_client.py",
                "pyspark/sql/tests/connect/test_parity_types.py",
            ],
        )

    def test_overlay_adds_official_files_and_never_overwrites_ours(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo_python = root / "repo" / "python"
            official = root / "spark" / "python"
            self._touch(repo_python, "pyspark/__init__.py", "ours")
            self._touch(repo_python, "pyspark/sql/tests/connect/test_own.py", "ours")
            self._touch(repo_python, "pyspark/__pycache__/x.pyc", "stale")
            self._touch(repo_python, "pyspark/_pyspark.so", "old")
            self._touch(official, "pyspark/sql/tests/connect/test_own.py", "official")
            self._touch(official, "pyspark/sql/tests/connect/test_official.py", "official")
            self._touch(official, "pyspark/sql/tests/connect/__pycache__/t.pyc", "stale")
            self._touch(official, "pyspark/ml/tests/connect/test_parity_als.py", "official")
            self._touch(official, "pyspark/tests/test_context.py", "official")
            self._touch(official, "pyspark/pandas/tests/test_frame.py", "not copied")
            so = self._touch(root, "built/_pyspark.so", "built")
            overlay = root / "overlay"
            overlay.mkdir()

            dst = runner.build_dropin_overlay(repo_python, official, so, overlay)

            self.assertEqual(dst, overlay / "python")
            self.assertEqual((dst / "pyspark/_pyspark.so").read_text(), "built")
            self.assertEqual((dst / "pyspark/__init__.py").read_text(), "ours")
            self.assertEqual((dst / "pyspark/sql/tests/connect/test_own.py").read_text(), "ours")
            self.assertEqual(
                (dst / "pyspark/sql/tests/connect/test_official.py").read_text(), "official"
            )
            self.assertEqual((dst / "pyspark/ml/tests/connect/test_parity_als.py").exists(), True)
            self.assertTrue((dst / "pyspark/tests/test_context.py").exists())
            self.assertFalse((dst / "pyspark/pandas").exists())
            self.assertFalse((dst / "pyspark/__pycache__").exists())
            self.assertFalse((dst / "pyspark/sql/tests/connect/__pycache__").exists())
            # The repo tree itself is untouched.
            self.assertEqual((repo_python / "pyspark/_pyspark.so").read_text(), "old")
            self.assertFalse((repo_python / "pyspark/sql/tests/connect/test_official.py").exists())

    def test_dropin_runs_without_the_transport_and_with_the_deselect_plugin(self):
        result = subprocess.CompletedProcess([], 0, "1 passed in 0.1s", "")
        with tempfile.TemporaryDirectory() as td:
            spark_py = Path(td) / "overlay" / "python"
            test_file = spark_py / "pyspark/sql/tests/connect/test_example.py"
            with (
                patch.dict(os.environ, {"RUST_PYSPARK_SO": "test.so"}),
                patch.object(runner.subprocess, "run", return_value=result) as run,
            ):
                runner.run_ours(
                    test_file, spark_py, "sc://localhost", ["Cls::test_a"], 10, dropin=True
                )
                runner.run_ours(
                    test_file,
                    spark_py,
                    "sc://localhost",
                    ["Cls::test_b"],
                    10,
                    select_only=True,
                    dropin=True,
                )
        deselect_call, drift_call = run.call_args_list
        argv, kwargs = deselect_call.args[0], deselect_call.kwargs
        env = kwargs["env"]
        self.assertIn("dropin_deselect_plugin", argv)
        self.assertNotIn("rust_transport_plugin", argv)
        self.assertEqual(kwargs["cwd"], str(spark_py))
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(spark_py))
        self.assertNotIn(str(runner.REPO / "python"), env["PYTHONPATH"].split(os.pathsep))
        self.assertEqual(env["RUST_PARITY_DESELECT"], "Cls::test_a")
        self.assertNotIn("RUST_PARITY_SELECT_ONLY", env)
        # The drift check selects only the listed tests.
        self.assertEqual(drift_call.kwargs["env"]["RUST_PARITY_SELECT_ONLY"], "Cls::test_b")

    def test_transport_mode_is_unchanged(self):
        result = subprocess.CompletedProcess([], 0, "1 passed in 0.1s", "")
        with tempfile.TemporaryDirectory() as td:
            spark_py = Path(td) / "python"
            test_file = spark_py / "pyspark/sql/tests/connect/test_example.py"
            with (
                patch.dict(os.environ, {"RUST_PYSPARK_SO": "test.so"}),
                patch.object(runner.subprocess, "run", return_value=result) as run,
            ):
                runner.run_ours(test_file, spark_py, "sc://localhost", [], 10)
        argv, kwargs = run.call_args.args[0], run.call_args.kwargs
        self.assertIn("rust_transport_plugin", argv)
        self.assertNotIn("dropin_deselect_plugin", argv)
        self.assertEqual(kwargs["env"]["RUST_PYSPARK_SO"], "test.so")

    def test_dropin_manifest_lists_no_open_gap(self):
        per_file, whole_file = runner.load_manifest(runner.DROPIN_MANIFEST)
        gaps = (
            "test_connect_session.py",
            "test_connect_context.py",
            "test_connect_error.py",
            "test_connect_column.py",
            "test_connect_cache.py",
            "test_connect_model_offloading.py",
        )
        listed = list(per_file) + list(whole_file)
        for name in gaps:
            self.assertFalse(any(p.endswith(name) for p in listed), name)
        self.assertTrue(all(p.startswith(runner.DROPIN_SCOPE) for p in listed))


def _load_deselect_plugin():
    spec = importlib.util.spec_from_file_location(
        "dropin_deselect_plugin", Path(__file__).resolve().parents[1] / "dropin_deselect_plugin.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DropinDeselectPluginTests(unittest.TestCase):
    class _Item:
        def __init__(self, nodeid):
            self.nodeid = nodeid

    def _apply(self, env, nodeids):
        from unittest.mock import MagicMock

        plugin = _load_deselect_plugin()
        items = [self._Item(n) for n in nodeids]
        config = MagicMock()
        with patch.dict(os.environ, env, clear=False):
            for var in ("RUST_PARITY_DESELECT", "RUST_PARITY_SELECT_ONLY"):
                if var not in env:
                    os.environ.pop(var, None)
            plugin.pytest_collection_modifyitems(config, items)
        return [i.nodeid for i in items], config

    def test_deselects_suffix_and_parametrized_matches_only(self):
        nodeids = [
            "pyspark/x/test_a.py::A::test_one",
            "pyspark/x/test_a.py::A::test_one[param]",
            "pyspark/x/test_a.py::A::test_two",
            "pyspark/x/test_a.py::B::test_one_more",
        ]
        kept, config = self._apply({"RUST_PARITY_DESELECT": "A::test_one\n"}, nodeids)
        self.assertEqual(kept, [nodeids[2], nodeids[3]])
        config.hook.pytest_deselected.assert_called_once()

    def test_select_only_keeps_the_listed_tests(self):
        nodeids = ["pyspark/x/test_a.py::A::test_one", "pyspark/x/test_a.py::A::test_two"]
        kept, _ = self._apply({"RUST_PARITY_SELECT_ONLY": "A::test_two"}, nodeids)
        self.assertEqual(kept, [nodeids[1]])

    def test_no_env_keeps_everything(self):
        nodeids = ["pyspark/x/test_a.py::A::test_one"]
        kept, config = self._apply({}, nodeids)
        self.assertEqual(kept, nodeids)
        config.hook.pytest_deselected.assert_not_called()


if __name__ == "__main__":
    unittest.main()
