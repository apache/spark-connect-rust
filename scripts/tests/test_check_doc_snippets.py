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

"""Offline doc-checker tests; no Rust toolchain or Spark server required."""

import contextlib
import importlib.util
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "check_doc_snippets.py"
SPEC = importlib.util.spec_from_file_location("check_doc_snippets", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class CheckDocSnippetsTests(unittest.TestCase):
    def test_type_checks_without_linking_a_native_executable(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            result = subprocess.CompletedProcess([], 0, stderr="")
            with patch.object(checker.subprocess, "run", return_value=result) as run:
                self.assertEqual(checker.check_block("let n: i64 = 1;", tmpdir, 7), (True, ""))

            src = tmpdir / "snippet_7.rs"
            self.assertEqual(
                src.read_text(), checker.HARNESS.format(body="        let n: i64 = 1;")
            )
            run.assert_called_once_with(
                [
                    "rustc",
                    "--edition",
                    "2021",
                    "--crate-type",
                    "bin",
                    "--emit=metadata",
                    "-L",
                    f"dependency={checker.DEPS}",
                    "--extern",
                    f"spark_connect={checker.RLIB}",
                    "-o",
                    str(tmpdir / "snippet_7.rmeta"),
                    str(src),
                ],
                capture_output=True,
                text=True,
            )

    def test_compiler_errors_are_preserved(self):
        error = "error[E0308]: mismatched types\nexpected i64, found &str\n"
        result = subprocess.CompletedProcess([], 1, stderr=error)
        with tempfile.TemporaryDirectory() as td:
            with patch.object(checker.subprocess, "run", return_value=result):
                self.assertEqual(
                    checker.check_block('let n: i64 = "bad";', Path(td), 1), (False, error)
                )

    def test_checks_all_rust_snippets_even_after_a_failure(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            docs = repo / "docs"
            nested = docs / "guides"
            nested.mkdir(parents=True)
            rlib = repo / "libspark_connect.rlib"
            rlib.touch()
            (docs / "first.md").write_text(
                '```rust\nlet n: i64 = "bad";\n```\n'
                "```{.rust .no-run}\ndeliberate pseudocode\n```\n"
                "```rust\nlet n: i64 = 1;\n```\n"
            )
            (nested / "second.md").write_text("```rust\nlet n: i64 = 2;\n```\n")
            output = io.StringIO()
            results = [
                (False, "warning: example\nerror[E0308]: mismatched types\nmore detail"),
                (True, ""),
                (True, ""),
            ]
            with (
                patch.object(checker, "REPO", repo),
                patch.object(checker, "RLIB", rlib),
                patch.object(checker, "check_block", side_effect=results) as check,
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(checker.main(), 1)

            self.assertEqual(
                [(call.args[0], call.args[2]) for call in check.call_args_list],
                [('let n: i64 = "bad";\n', 1), ("let n: i64 = 1;\n", 2), ("let n: i64 = 2;\n", 3)],
            )
            self.assertEqual(
                output.getvalue(),
                "Checked 3 rust snippet(s); 1 failed to compile.\n"
                "  docs/first.md: error[E0308]: mismatched types\n",
            )


if __name__ == "__main__":
    unittest.main()
