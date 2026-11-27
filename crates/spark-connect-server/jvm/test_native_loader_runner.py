#!/usr/bin/env python3
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

"""Exercise loader runner output guards with fake tools, without compiling JNI."""

import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


RUNNER = Path(__file__).with_name("test-native-loader.sh")
MARKERS = {
    "fixture": "NATIVE_LIBRARY_LOADER_OK checks=1\n",
    "packaged": "NATIVE_PACKAGED_LOAD_OK\n",
}


class NativeLoaderRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.java_home = self.directory / "jdk"
        self.tools = self.directory / "tools"
        self.tools.mkdir()
        self.executable(self.java_home / "bin/javac", "raise SystemExit(0)\n")
        self.executable(self.java_home / "bin/java", """
import os
import sys

mode = "PACKAGED" if sys.argv[-1] == "--load-packaged" else "FIXTURE"
sys.stdout.write(os.environ["FAKE_" + mode + "_STDOUT"])
sys.stderr.write(os.environ.get("FAKE_" + mode + "_STDERR", ""))
raise SystemExit(int(os.environ.get("FAKE_" + mode + "_STATUS", "0")))
""")
        self.executable(self.tools / "cc", "raise SystemExit(0)\n")
        self.executable(self.tools / "version", "print('0.1.0')\n")
        self.native_jar = self.directory / "native.jar"
        self.native_jar.touch()

    @staticmethod
    def executable(path, source, interpreter=sys.executable):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexec " + shlex.quote(str(interpreter))
                        + " -c " + shlex.quote(source) + ' "$@"\n', encoding="utf-8")
        path.chmod(0o755)

    def run_loader(self, mode, output, *, stderr="", status=0):
        environment = os.environ.copy()
        environment.update({
            "JAVA_HOME": str(self.java_home),
            "CC": str(self.tools / "cc"),
            "NATIVE_BUILD_PYTHON": str(self.tools / "version"),
            "TMPDIR": str(self.directory),
            "PATH": str(self.tools) + os.pathsep + environment.get("PATH", os.defpath),
        })
        for name, marker in MARKERS.items():
            environment[f"FAKE_{name.upper()}_STDOUT"] = marker
        environment[f"FAKE_{mode.upper()}_STDOUT"] = output
        environment[f"FAKE_{mode.upper()}_STDERR"] = stderr
        environment[f"FAKE_{mode.upper()}_STATUS"] = str(status)
        command = ["bash", str(RUNNER)]
        if mode == "packaged":
            command.extend(["--native-jar", str(self.native_jar)])
        return subprocess.run(command, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=10)

    def test_each_mode_requires_its_own_success_marker(self):
        for mode, marker in MARKERS.items():
            with self.subTest(mode=mode):
                result = self.run_loader(mode, marker)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn(marker, result.stdout)

    def test_python_interpreter_path_with_spaces(self):
        interpreter = self.directory / "python with spaces"
        interpreter.symlink_to(sys.executable)
        self.executable(self.java_home / "bin/javac", "raise SystemExit(0)\n", interpreter)
        result = self.run_loader("fixture", MARKERS["fixture"])
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_jni_diagnostics_fail_even_with_success_marker_and_zero_exit(self):
        for mode, marker in MARKERS.items():
            for diagnostic in ["WARNING in native method", "WARNING: JNI local refs:",
                               "FATAL ERROR in native method"]:
                with self.subTest(mode=mode, diagnostic=diagnostic):
                    result = self.run_loader(mode, marker, stderr=diagnostic + ": probe\n")
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("triggered a JNI diagnostic", result.stdout)
                    self.assertIn(diagnostic, result.stdout)

    def test_empty_output_fails(self):
        for mode in MARKERS:
            with self.subTest(mode=mode):
                result = self.run_loader(mode, "")
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("success marker missing", result.stdout)

    def test_other_modes_marker_does_not_pass(self):
        for mode in MARKERS:
            other = "packaged" if mode == "fixture" else "fixture"
            with self.subTest(mode=mode):
                result = self.run_loader(mode, MARKERS[other])
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("success marker missing", result.stdout)

    def test_incomplete_or_embedded_markers_do_not_pass(self):
        for mode, marker in MARKERS.items():
            outputs = [marker.strip()[:-1] + "\n", "prefix " + marker,
                       marker.rstrip() + " suffix\n"]
            if mode == "fixture":
                outputs.append("NATIVE_LIBRARY_LOADER_OK checks=0\n")
            for output in outputs:
                with self.subTest(mode=mode, output=output):
                    result = self.run_loader(mode, output)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn("success marker missing", result.stdout)

    def test_java_failure_propagates_through_logging_pipeline(self):
        for mode, marker in MARKERS.items():
            with self.subTest(mode=mode):
                result = self.run_loader(mode, marker, status=17)
                self.assertEqual(result.returncode, 17, result.stdout)

    def test_logging_failure_does_not_pass(self):
        self.executable(self.tools / "tee", """
from pathlib import Path
import sys

output = sys.stdin.read()
Path(sys.argv[1]).write_text(output)
sys.stdout.write(output)
raise SystemExit(23)
""")
        result = self.run_loader("fixture", MARKERS["fixture"])
        self.assertEqual(result.returncode, 23, result.stdout)


if __name__ == "__main__":
    unittest.main()
