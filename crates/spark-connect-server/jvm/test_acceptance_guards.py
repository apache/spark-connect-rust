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

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_acceptance import (
    SHUTDOWN_COMPLETE,
    engine_validation_commands,
    listener_owner_pids,
    record_jni_check,
    runtime_skip_evidence,
    shutdown_error,
)


class RuntimeSkipEvidenceTest(unittest.TestCase):
    def test_readonly_skip_preserves_an_unexpected_reason_and_raw_marker(self):
        reason = "unexpected filesystem capability; running ordinary cleanup instead"
        marker = "STOPPED_CONTEXT_READONLY_SKIPPED: " + reason
        content = "SparkContext stopped\n" + marker + "\nSTOPPED_CONTEXT_CLEANUP_OK\n"
        self.assertEqual(runtime_skip_evidence(content), [
            {"case": "stopped-context-readonly-artifacts", "marker": marker, "reason": reason},
        ])

    def test_only_the_exact_skip_marker_is_reported(self):
        content = ("STOPPED_CONTEXT_CLEANUP_OK\n"
                   "log prefix STOPPED_CONTEXT_READONLY_SKIPPED: not a case marker\n"
                   "STOPPED_CONTEXT_READONLY_SKIPPED_EXTRA: not a case marker\n")
        self.assertEqual(runtime_skip_evidence(content), [])


class EngineValidationCommandsTest(unittest.TestCase):
    def test_engine_validation_commands_preserve_common_guards(self):
        common = ["spark-submit", "--master", "local[2]", "--driver-memory", "1g",
                  "--driver-java-options", "-Xcheck:jni", "--conf", "spark.ui.enabled=false",
                  "--conf", "spark.driver.host=127.0.0.1",
                  "--conf", "spark.driver.bindAddress=127.0.0.1"]
        original = list(common)
        (cluster, result_size, per_partition_size, config_testing,
         global_environment) = engine_validation_commands(common, Path("tests with spaces.jar"))
        self.assertEqual(common, original)
        name, command, marker = cluster
        self.assertEqual(name, "ipc-separate-executor")
        self.assertEqual(marker, "IPC_BATCH_SMOKE_OK")
        for option, value in (("--master", "local-cluster[1,1,1024]"),
                              ("--executor-memory", "512m"), ("--executor-cores", "1")):
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertIn("-Xcheck:jni -XX:ActiveProcessorCount=2", command)
        for setting in ("spark.executor.extraJavaOptions=-Xcheck:jni -XX:ActiveProcessorCount=2",
                        "spark.task.maxFailures=1", "spark.cores.max=1",
                        "spark.serializer=org.apache.spark.serializer.KryoSerializer",
                        "spark.kryo.registrationRequired=true",
                        "spark.rpc.netty.dispatcher.numThreads=2",
                        "spark.shuffle.io.serverThreads=2", "spark.shuffle.io.clientThreads=2"):
            self.assertIn(setting, command)
        name, bounded, marker = result_size
        self.assertEqual(name, "ipc-result-size")
        self.assertEqual(marker, "IPC_RESULT_SIZE_SMOKE_OK")
        self.assertEqual(bounded[bounded.index("--master") + 1], "local[2]")
        self.assertIn("spark.driver.maxResultSize=1m", bounded)
        self.assertEqual(bounded[-1], "--result-size-only")
        name, per_partition, marker = per_partition_size
        self.assertEqual(name, "per-partition-result-size")
        self.assertEqual(marker, "NATIVE_ENGINE_PER_PARTITION_RESULT_SIZE_OK")
        self.assertEqual(per_partition[per_partition.index("--master") + 1], "local[2]")
        self.assertIn("spark.driver.maxResultSize=1m", per_partition)
        self.assertEqual(per_partition[per_partition.index("--class") + 1],
                         "org.apache.spark.sql.connect.nativebridge.engine."
                         "NativeEngineSchedulingTest")
        self.assertEqual(per_partition[-1], "--result-size-only")
        name, testing, marker = config_testing
        self.assertEqual(name, "engine-config-testing")
        self.assertEqual(marker,
                         "CONFIG_UNSET_DEFAULT: true (rejection and atomic rollback checked)")
        self.assertEqual(testing[:2], ["env", "SPARK_TESTING=1"])
        self.assertEqual(testing[testing.index("--class") + 1],
                         "org.apache.spark.sql.connect.nativebridge.engine."
                         "NativeEngineConfigConsistencyTest")
        self.assertEqual(testing[-1], "tests with spaces.jar")
        name, global_env, marker = global_environment
        self.assertEqual(name, "global-environment-lifecycle")
        self.assertEqual(marker, "GLOBAL_ENVIRONMENT_LIFECYCLE_OK")
        self.assertEqual(global_env[global_env.index("--class") + 1],
                         "org.apache.spark.sql.connect.nativebridge.engine."
                         "GlobalEnvironmentLifecycleTest")
        self.assertEqual(global_env[-1], "tests with spaces.jar")
        for command in (command, bounded, per_partition, testing, global_env):
            self.assertIn("spark.ui.enabled=false", command)
            self.assertIn("spark.driver.host=127.0.0.1", command)
            self.assertIn("spark.driver.bindAddress=127.0.0.1", command)
            self.assertIn("tests with spaces.jar", command)
            self.assertNotIn("--worker-only", command)


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
        content = ("JNI cleanup is incomplete; retained cleanup failed: orphaned Java worker context\n"
                   + SHUTDOWN_COMPLETE + "\n")
        self.assertEqual(shutdown_error(content, 128 + signal.SIGTERM),
                         "embedded server reported failed resource cleanup")

    def test_successful_per_call_attachment_fallback_is_accepted(self):
        content = ("JVM cleanup worker attachment failed; using per-call attachment: temporary failure\n"
                   + SHUTDOWN_COMPLETE + "\n")
        self.assertIsNone(shutdown_error(content, 128 + signal.SIGTERM))

    def test_abnormal_exit_is_rejected_even_with_marker(self):
        self.assertIsNotNone(shutdown_error(SHUTDOWN_COMPLETE + "\n", 1))

    def test_marker_requires_a_complete_line(self):
        self.assertIsNotNone(shutdown_error(SHUTDOWN_COMPLETE + " (not really)\n", 0))

    def test_other_server_cleanup_marker_is_rejected(self):
        self.assertIsNotNone(shutdown_error("Embedded Spark Connect shutdown complete\n", 0))


class JniDiagnosticTest(unittest.TestCase):
    def test_warning_preserves_an_earlier_failure(self):
        report = {"error": "client failed before shutdown"}
        record_jni_check(report, "WARNING in native method: expected probe warning\n")
        self.assertFalse(report["jni_check_clean"])
        self.assertEqual(report["error"], "client failed before shutdown")

    def test_warning_is_reported_when_it_is_the_first_failure(self):
        report = {}
        record_jni_check(report, "FATAL ERROR in native method: expected probe warning\n")
        self.assertFalse(report["jni_check_clean"])
        self.assertIn("JNI check warning", report["error"])

    def test_local_reference_capacity_warning_is_rejected(self):
        report = {}
        record_jni_check(report, "WARNING: JNI local refs: 33, exceeds capacity: 32\n")
        self.assertFalse(report["jni_check_clean"])
        self.assertIn("JNI check warning", report["error"])

    def test_smoke_rejects_local_reference_capacity_warning(self):
        # Run the actual smoke validator with a fake Spark command, so the exact
        # -Xcheck:jni marker is rejected without starting a JVM or loading JNI.
        with tempfile.TemporaryDirectory(prefix="native JNI diagnostic ") as directory:
            root = Path(directory)
            jvm = root / "project" / "jvm"
            jvm.mkdir(parents=True)
            smoke = jvm / "test-jni-smoke.sh"
            smoke.write_bytes(Path(__file__).with_name("test-jni-smoke.sh").read_bytes())
            build = jvm / "build.sh"
            build.write_text("#!/bin/bash\nexit 0\n")
            build.chmod(0o755)
            spark = root / "spark home"
            (spark / "bin").mkdir(parents=True)
            submit = spark / "bin" / "spark-submit"
            submit.write_text("#!/bin/bash\nprintf '%s\\n' "
                              "'WARNING: JNI local refs: 33, exceeds capacity: 32' "
                              "'NATIVE_JNI_SMOKE_OK: fake Spark command'\n")
            submit.chmod(0o755)
            library = root / "native artifact.so"
            library.touch()
            environment = os.environ.copy()
            environment.update(SPARK_HOME=str(spark), NATIVE_TEST_PYTHON=sys.executable)
            result = subprocess.run(["/bin/bash", str(smoke), str(library)], env=environment,
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("real JNI smoke triggered a JNI diagnostic", result.stderr)
            reports = list((jvm.parent / ".build" / "jvm-native").glob("*/result.json"))
            self.assertEqual(len(reports), 1)
            self.assertFalse(json.loads(reports[0].read_text())["passed"])

    def test_clean_log_does_not_add_a_failure(self):
        report = {}
        record_jni_check(report, SHUTDOWN_COMPLETE + "\n")
        self.assertTrue(report["jni_check_clean"])
        self.assertNotIn("error", report)


class AssertionModeTest(unittest.TestCase):
    def test_zero_optimization_reaches_input_validation_and_one_is_rejected(self):
        # The deliberately absent library stops the runner before any run directory
        # or Spark process can be created. Getting there proves the zero setting passed.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for setting in ("0", "1"):
                with self.subTest(setting=setting):
                    environment = os.environ.copy()
                    environment["PYTHONOPTIMIZE"] = setting
                    result = subprocess.run(
                        [sys.executable, "-B", str(Path(__file__).with_name("run_acceptance.py")),
                         "--library", str(root / "missing-native-library"),
                         "--spark-home", str(root)],
                        env=environment, capture_output=True, text=True, timeout=10)
                    self.assertNotEqual(result.returncode, 0)
                    if setting == "0":
                        self.assertIn("FileNotFoundError", result.stderr)
                        self.assertNotIn("acceptance checks require Python assertions", result.stderr)
                    else:
                        self.assertIn("acceptance checks require Python assertions", result.stderr)
                        self.assertNotIn("FileNotFoundError", result.stderr)


class AcceptanceEnvironmentTest(unittest.TestCase):
    def test_exported_testing_mode_does_not_reach_default_validation(self):
        # Run the real runner with a fake first Spark command. It records only the
        # testing flag and exits before any JVM or native library can be loaded.
        with tempfile.TemporaryDirectory(prefix="native acceptance environment ") as directory:
            root = Path(directory)
            project = root / "project"
            jvm = project / "jvm"
            jvm.mkdir(parents=True)
            runner = jvm / "run_acceptance.py"
            runner.write_bytes(Path(__file__).with_name("run_acceptance.py").read_bytes())
            artifacts = project / ".build" / "jvm"
            artifacts.mkdir(parents=True)
            for name in ("spark-connect-native-bridge.jar", "bridge-test.jar"):
                (artifacts / name).touch()
            spark = root / "spark home"
            (spark / "bin").mkdir(parents=True)
            submit = spark / "bin" / "spark-submit"
            submit.write_text('#!/bin/bash\nprintf "%s" "${SPARK_TESTING-unset}" '
                              '> "$NATIVE_ENV_CAPTURE"\nexit 19\n')
            submit.chmod(0o755)
            library = root / "native artifact.so"
            library.touch()
            capture = root / "testing mode"
            report = root / "acceptance.json"
            environment = os.environ.copy()
            environment.update(SPARK_TESTING="1", PYTHONOPTIMIZE="0",
                               NATIVE_ENV_CAPTURE=str(capture))
            result = subprocess.run(
                [sys.executable, "-B", str(runner), "--library", str(library),
                 "--spark-home", str(spark), "--output", str(report)],
                env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(capture.read_text(), "unset")
            first = json.loads(report.read_text())["results"][0]
            self.assertEqual((first["name"], first["exit_code"]), ("engine-smoke", 19))


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
    def launch(self, bundled, master=None):
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
            environment.pop("SPARK_MASTER", None)
            if master is not None:
                environment["SPARK_MASTER"] = master
            environment.update(SPARK_HOME=str(spark), SPARK_REMOTE="sc://must-be-cleared",
                               NATIVE_LAUNCH_CAPTURE=str(capture))
            arguments = ["--native-jar", str(native)] if bundled else [str(native)]
            subprocess.run(["/bin/bash", str(launcher), *arguments, "127.0.0.1:0"],
                           env=environment, check=True, capture_output=True, timeout=10)
            actual = capture.read_bytes().decode().rstrip("\0").split("\0")
            expected = ["unset", "0", "--class",
                        "org.apache.spark.sql.connect.nativebridge.engine.EmbeddedSparkEngine",
                        "--master", master or "local[2]"]
            if bundled:
                expected += ["--jars", str(native)]
            expected += [str(project / ".build/jvm/spark-connect-native-bridge.jar"),
                         "--bundled" if bundled else str(native), "127.0.0.1:0"]
            self.assertEqual(actual, expected)

    def test_explicit_path_on_system_bash(self):
        self.launch(False)

    def test_packaged_path_on_system_bash(self):
        self.launch(True)

    def test_empty_ambient_master_on_system_bash(self):
        self.launch(False, master="")

    def test_custom_master_on_system_bash(self):
        self.launch(False, master="local[7]")


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
