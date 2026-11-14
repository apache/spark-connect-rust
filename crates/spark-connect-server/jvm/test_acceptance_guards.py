# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Check acceptance's shutdown and listener ownership evidence; no Spark needed."""

from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_acceptance import (
    SHUTDOWN_COMPLETE, listener_owner_pids, shutdown_error,
)


class ShutdownEvidenceTest(unittest.TestCase):
    def test_completed_cleanup_with_sigterm_is_accepted(self):
        self.assertIsNone(shutdown_error(SHUTDOWN_COMPLETE + "\n", 128 + signal.SIGTERM))

    def test_startup_cleanup_is_not_normal_shutdown(self):
        content = "Embedded Spark Engine startup failed; cleanup complete\n"
        self.assertIsNotNone(shutdown_error(content, 0))

    def test_missing_cleanup_marker_is_rejected(self):
        self.assertIsNotNone(shutdown_error("SparkContext stopped\n", 128 + signal.SIGTERM))

    def test_native_stop_failure_is_rejected_even_with_marker(self):
        content = "JNI stop is incomplete: active calls remain\n" + SHUTDOWN_COMPLETE + "\n"
        self.assertIsNotNone(shutdown_error(content, 128 + signal.SIGTERM))

    def test_launcher_cleanup_failure_is_rejected(self):
        content = "Embedded Spark Engine shutdown failed: native stop incomplete\n"
        self.assertIsNotNone(shutdown_error(content, 128 + signal.SIGTERM))

    def test_resource_cleanup_failure_is_rejected_even_with_marker(self):
        for diagnostic in (
            "JNI cleanup is incomplete; retained cleanup failed: cannot attach",
            "Java engine handle cleanup failed: cannot enter Java engine context",
        ):
            with self.subTest(diagnostic=diagnostic):
                content = diagnostic + "\n" + SHUTDOWN_COMPLETE + "\n"
                self.assertIsNotNone(shutdown_error(content, 128 + signal.SIGTERM))

    def test_successful_per_call_attachment_fallback_is_accepted(self):
        content = ("JVM cleanup lane attachment failed; using per-call attachment: temporary failure\n"
                   + SHUTDOWN_COMPLETE + "\n")
        self.assertIsNone(shutdown_error(content, 128 + signal.SIGTERM))

    def test_abnormal_exit_is_rejected_even_with_marker(self):
        self.assertIsNotNone(shutdown_error(SHUTDOWN_COMPLETE + "\n", 1))

    def test_marker_requires_a_complete_line(self):
        self.assertIsNotNone(shutdown_error(SHUTDOWN_COMPLETE + " (not really)\n", 0))

    def test_other_server_cleanup_marker_is_rejected(self):
        self.assertIsNotNone(shutdown_error("Embedded Spark Connect shutdown complete\n", 0))


class ListenerOwnershipTest(unittest.TestCase):
    @patch("run_acceptance.subprocess.run")
    @patch("run_acceptance.shutil.which", return_value="/usr/sbin/lsof")
    def test_listener_ownership_preserves_distinct_processes(self, which, run):
        run.return_value = subprocess.CompletedProcess([], 0, "p7\nf1\np7\np11\n", "")
        # lsof remains preferred even when /proc is unavailable.
        self.assertEqual(listener_owner_pids(12345, proc_root=Path("/missing-proc")), [7, 11])

    @patch("run_acceptance.subprocess.run")
    @patch("run_acceptance.shutil.which", return_value=None)
    def test_proc_fallback_finds_only_listening_ipv4_and_ipv6_owners(self, which, run):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            (proc / "net").mkdir()
            (proc / "net" / "tcp").write_text(
                "  sl  local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
                "0: 0100007F:3039 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 11111\n"
                "1: 0100007F:3039 00000000:0000 01 00000000:00000000 00:00000000 00000000 1000 0 33333\n"
                "2: 0100007F:303A 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 44444\n")
            (proc / "net" / "tcp6").write_text(
                "  sl  local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
                "0: 00000000000000000000000001000000:3039 00000000000000000000000000000000:0000 "
                "0A 00000000:00000000 00:00000000 00000000 1000 0 22222\n")
            for process, targets in (
                ("11", ("socket:[22222]",)),
                ("7", ("socket:[11111]", "socket:[11111]", "pipe:[11111]")),
                ("22", ("socket:[33333]",)),
                ("33", ("socket:[44444]",)),
                ("self", ("socket:[11111]",)),
            ):
                descriptors = proc / process / "fd"
                descriptors.mkdir(parents=True)
                for number, target in enumerate(targets):
                    (descriptors / str(number)).symlink_to(target)
            # Processes may exit before their descriptors can be inspected.
            (proc / "99").mkdir()

            self.assertEqual(listener_owner_pids(12345, proc_root=proc), [7, 11])
            run.assert_not_called()

    @patch("run_acceptance.shutil.which", return_value=None)
    def test_proc_fallback_accepts_missing_optional_ipv6_table(self, which):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            (proc / "net").mkdir()
            (proc / "net" / "tcp").write_text("sl local_address rem_address st\n")
            self.assertEqual(listener_owner_pids(12345, proc_root=proc), [])

    @patch("run_acceptance.shutil.which", return_value=None)
    def test_listener_ownership_requires_lsof_or_proc(self, which):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "requires lsof or Linux /proc"):
                listener_owner_pids(12345, proc_root=Path(directory))


class LauncherArgumentsTest(unittest.TestCase):
    def launch(self, bundled):
        # Exercise the actual launcher with paths containing spaces. /bin/bash is
        # deliberately used so macOS also checks its system Bash 3.2 behavior.
        with tempfile.TemporaryDirectory(prefix="native launch space ") as directory:
            root = Path(directory)
            project = root / "project"
            jvm = project / "jvm"
            jvm.mkdir(parents=True)
            launcher = jvm / "run-engine.sh"
            launcher.write_bytes(Path(__file__).with_name("run-engine.sh").read_bytes())
            build = jvm / "build.sh"
            build.write_text("#!/bin/bash\nexit 0\n")
            build.chmod(0o755)
            spark = root / "spark home"
            (spark / "bin").mkdir(parents=True)
            submit = spark / "bin" / "spark-submit"
            submit.write_text('#!/bin/bash\nprintf "%s\\0" "${SPARK_REMOTE-unset}" '
                              '"$SPARK_CONNECT_MODE" "$@" > "$NATIVE_LAUNCH_CAPTURE"\n')
            submit.chmod(0o755)
            native = root / ("native artifact.jar" if bundled else "native artifact.so")
            native.touch()
            capture = root / "arguments"
            environment = os.environ.copy()
            environment.update(SPARK_HOME=str(spark), SPARK_REMOTE="sc://must-be-cleared",
                               NATIVE_LAUNCH_CAPTURE=str(capture))
            arguments = ["--native-jar", str(native)] if bundled else [str(native)]
            subprocess.run(["/bin/bash", str(launcher), *arguments, "127.0.0.1:0"],
                           env=environment, check=True, capture_output=True, timeout=10)
            actual = capture.read_bytes().decode().rstrip("\0").split("\0")
            expected = ["unset", "0", "--class",
                        "org.apache.spark.sql.connect.nativebridge.engine.EmbeddedSparkEngine",
                        "--master", environment.get("SPARK_MASTER", "local[2]")]
            if bundled:
                expected += ["--jars", str(native)]
            expected += [str(project / ".build/jvm/spark-connect-native-bridge.jar"),
                         "--bundled" if bundled else str(native), "127.0.0.1:0"]
            self.assertEqual(actual, expected)

    def test_explicit_path_on_system_bash(self):
        self.launch(False)

    def test_packaged_path_on_system_bash(self):
        self.launch(True)


class AcceptanceAssemblyTest(unittest.TestCase):
    def launch(self, option, environment_home):
        with tempfile.TemporaryDirectory(prefix="native acceptance space ") as directory:
            root = Path(directory)
            jvm = root / "project" / "jvm"
            jvm.mkdir(parents=True)
            wrapper = jvm / "test-engine.sh"
            wrapper.write_bytes(Path(__file__).with_name("test-engine.sh").read_bytes())
            (jvm / "test_acceptance_guards.py").write_text("pass\n")
            build = jvm / "build.sh"
            build.write_text('#!/bin/bash\nprintf "%s" "$SPARK_HOME" > "$NATIVE_BUILD_CAPTURE"\n')
            build.chmod(0o755)
            (jvm / "run_acceptance.py").write_text(
                "import json, os, sys\nfrom pathlib import Path\n"
                "Path(os.environ['NATIVE_RUN_CAPTURE']).write_text("
                "json.dumps([os.environ['SPARK_HOME'], sys.argv[1:]]))\n")
            selected = str(root / "selected spark home")
            arguments = ["--library", "native artifact.so"]
            if option == "separate":
                arguments += ["--spark-home", selected]
            elif option == "equals":
                arguments += ["--spark-home=" + selected]
            environment = os.environ.copy()
            environment.pop("SPARK_HOME", None)
            if environment_home:
                environment["SPARK_HOME"] = str(root / "environment spark home")
            environment.update(NATIVE_TEST_PYTHON=sys.executable,
                               NATIVE_BUILD_CAPTURE=str(root / "built"),
                               NATIVE_RUN_CAPTURE=str(root / "ran"))
            subprocess.run(["/bin/bash", str(wrapper), *arguments], env=environment,
                           check=True, capture_output=True, timeout=10)
            expected = selected if option else environment["SPARK_HOME"]
            self.assertEqual((root / "built").read_text(), expected)
            self.assertEqual(json.loads((root / "ran").read_text()), [expected, arguments])

    def test_cli_assembly_overrides_environment_on_system_bash(self):
        self.launch("separate", True)

    def test_equals_assembly_without_environment_on_system_bash(self):
        self.launch("equals", False)

    def test_environment_assembly_on_system_bash(self):
        self.launch(None, True)


if __name__ == "__main__":
    unittest.main()
