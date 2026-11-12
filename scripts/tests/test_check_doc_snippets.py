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

"""Check metadata-only compilation with rustc; no Spark server or crate build needed."""

import importlib.util
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "check_doc_snippets.py"
SPEC = importlib.util.spec_from_file_location("check_doc_snippets", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class CheckBlockArgumentsTests(unittest.TestCase):
    def test_rustc_receives_all_dependency_directories(self):
        result = subprocess.CompletedProcess([], 0, "", "")
        deps = [Path("/target/debug/deps"), Path("/target/debug/build/unit/hash/out")]
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(checker.subprocess, "run", return_value=result) as run,
        ):
            ok, error = checker.check_block("", Path(td), 1, Path("/client.rlib"), deps)
        self.assertTrue(ok, error)
        args = run.call_args.args[0]
        self.assertEqual(
            args[args.index("-L") : args.index("--extern")],
            ["-L", f"dependency={deps[0]}", "-L", f"dependency={deps[1]}"],
        )
        self.assertIn("spark_connect=/client.rlib", args)


@unittest.skipUnless(shutil.which("rustc"), "rustc is required")
class CheckDocSnippetsTests(unittest.TestCase):
    # Self-contained snippets isolate compiler mode from the crate's dependencies.
    HARNESS = "fn main() {{{body}}}"

    def test_valid_snippet_emits_metadata_without_an_executable(self):
        with tempfile.TemporaryDirectory() as td, patch.object(checker, "HARNESS", self.HARNESS):
            tmpdir = Path(td)
            # An unresolved foreign symbol type-checks, but cannot be linked.
            code = (
                'extern "C" { fn check_doc_snippets_missing_symbol(); }'
                "unsafe { check_doc_snippets_missing_symbol(); }"
            )
            ok, error = checker.check_block(code, tmpdir, 1)
            self.assertTrue(ok, error)
            self.assertTrue((tmpdir / "snippet_1.rmeta").is_file())
            self.assertEqual(
                {path.name for path in tmpdir.iterdir()},
                {"snippet_1.rs", "snippet_1.rmeta"},
            )

    def test_metadata_emission_rejects_type_and_borrow_errors(self):
        cases = [
            ('let n: i64 = "bad";', "E0308"),
            ("let s = String::new(); let moved = s; drop(s);", "E0382"),
        ]
        for code, diagnostic in cases:
            with (
                self.subTest(diagnostic=diagnostic),
                tempfile.TemporaryDirectory() as td,
                patch.object(checker, "HARNESS", self.HARNESS),
            ):
                ok, error = checker.check_block(code, Path(td), 1)
                self.assertFalse(ok)
                self.assertIn(diagnostic, error)


class SnippetsTests(unittest.TestCase):
    def test_finds_indented_fences_and_strips_their_indentation(self):
        text = "\n".join(
            [
                "```rust",
                "let a = 1;",
                "```",
                "1. In a list item:",
                "   ```rust",
                "   let b = 2;",
                "       let c = 3;",
                "   ```",
                "!!! note",
                "    ```rust",
                "    let d = 4;",
                "    ```",
                "   ```{.rust .no-run}",
                "   let e = 5;",
                "   ```",
                "```python",
                "f = 6",
                "```",
                "",
            ]
        )
        self.assertEqual(
            checker.snippets(text),
            ["let a = 1;", "let b = 2;\n    let c = 3;", "let d = 4;"],
        )


class CrateArtifactsTests(unittest.TestCase):
    def test_failed_build_prints_the_rendered_errors_of_rustc(self):
        # cargo prints the errors of rustc as JSON on stdout with --message-format=json.
        error = {
            "reason": "compiler-message",
            "message": {"level": "error", "rendered": "error[E0425]: cannot find value `x`\n"},
        }
        warning = {
            "reason": "compiler-message",
            "message": {"level": "warning", "rendered": "warning: unused variable\n"},
        }
        stdout = "\n".join([json.dumps(warning), json.dumps(error), "not json"])
        failed = subprocess.CompletedProcess([], 101, stdout=stdout, stderr="error: build failed")
        with (
            patch.object(checker.subprocess, "run", return_value=failed),
            patch("sys.stdout", new_callable=io.StringIO) as out,
        ):
            self.assertEqual(checker.crate_artifacts(), (None, []))
        printed = out.getvalue()
        self.assertIn("error[E0425]: cannot find value `x`", printed)
        self.assertNotIn("unused variable", printed)
        self.assertIn("error: build failed", printed)

    def test_cargo_artifacts_supply_each_dependency_directory(self):
        # Include both layouts, a fresh unit, a proc macro and duplicate paths.
        files = [
            ("arrow", ["/target/debug/deps/libarrow.rlib"]),
            ("macros", ["/target/debug/build/macros/hash/out/libmacros.dylib"]),
            (
                "spark_connect",
                [
                    "/target/debug/build/client/hash/out/libspark_connect.rlib",
                    "/target/debug/build/client/hash/out/libspark_connect.rmeta",
                ],
            ),
        ]
        output = ["cargo diagnostic", json.dumps({"reason": "build-finished"})]
        output.extend(
            json.dumps(
                {
                    "reason": "compiler-artifact",
                    "target": {"name": name},
                    "filenames": names,
                    "fresh": True,
                }
            )
            for name, names in files
        )
        result = subprocess.CompletedProcess([], 0, "\n".join(output), "")
        with patch.object(checker.subprocess, "run", return_value=result):
            rlib, deps = checker.crate_artifacts()
        self.assertEqual(rlib, Path(files[2][1][0]))
        self.assertEqual(deps, [Path(names[0]).parent for _, names in files])


if __name__ == "__main__":
    unittest.main()
