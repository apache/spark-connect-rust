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
        checked, problems = checker.snippets(text)
        self.assertEqual(
            [(block.line, block.code) for block in checked],
            [(1, "let a = 1;"), (5, "let b = 2;\n    let c = 3;"), (10, "let d = 4;")],
        )
        self.assertEqual(problems, [])

    def test_refuses_a_rust_fence_that_would_be_skipped(self):
        text = "\n".join(
            [
                "```rust,ignore",
                "query.stop()?;",
                "```",
                "",
                "```{.rust}",
                "let a = 1;",
                "```",
                "",
                "```rust no_run",
                "let b = 2;",
                "```",
            ]
        )
        checked, problems = checker.snippets(text)
        self.assertEqual(checked, [])
        self.assertEqual([line for line, _ in problems], [1, 5, 9])
        self.assertIn('"rust,ignore"', problems[0][1])

    def test_no_run_form_is_skipped_without_a_problem(self):
        text = '```{.rust .no-run}\nlet a: i32 = "bad";\n```\n```{.rust  .no-run}\n```\n'
        self.assertEqual(checker.snippets(text), ([], []))

    def test_no_run_marker_before_a_rust_fence_skips_it(self):
        text = (
            '<!-- snippet: no-run -->\n```rust\nlet a: i32 = "bad";\n```\n'
            "```rust\nlet b = 2;\n```\n"
        )
        checked, problems = checker.snippets(text)
        self.assertEqual([(block.line, block.code) for block in checked], [(5, "let b = 2;")])
        self.assertEqual(problems, [])

    def test_no_run_marker_not_directly_before_a_fence_does_nothing(self):
        for text in (
            "<!-- snippet: no-run -->\n\n```rust\nlet a = 1;\n```\n",
            "<!-- snippet: no-run -->\nSome text.\n```rust\nlet a = 1;\n```\n",
        ):
            with self.subTest(text=text):
                checked, problems = checker.snippets(text)
                self.assertEqual([block.line for block in checked], [3])
                self.assertEqual(problems, [])

    def test_plain_and_non_rust_fences_are_not_refused(self):
        text = "```python\nprint(1)\n```\n```\nplain\n```\n```bash\nls\n```\n"
        self.assertEqual(checker.snippets(text), ([], []))

    def test_tilde_fences_are_parsed_as_blocks(self):
        checked, problems = checker.snippets("~~~rust\nlet a = 1;\n~~~\n")
        self.assertEqual([(block.line, block.code) for block in checked], [(1, "let a = 1;")])
        self.assertEqual(problems, [])

    def test_crlf_line_endings_are_normalized(self):
        checked, problems = checker.snippets("```rust\r\nlet a = 1;\r\n```\r\n")
        self.assertEqual([block.code for block in checked], ["let a = 1;"])
        self.assertEqual(problems, [])

    def test_unclosed_fence_is_reported_at_its_line(self):
        checked, problems = checker.snippets("text\n```rust\nlet a = 1;\n")
        self.assertEqual(checked, [])
        self.assertEqual(problems, [(2, "this fence is never closed")])

    def test_fence_opened_inside_an_unclosed_fence_is_not_swallowed(self):
        # The second fence has an info string, so it cannot close the first one. It is
        # reported as a fence line that is not parsed, rather than passed over silently.
        checked, problems = checker.snippets("```rust\nlet a = 1;\n```rust\nlet b = 2;\n```\n")
        self.assertEqual([block.line for block in checked], [1])
        self.assertEqual([line for line, _ in problems], [3])
        self.assertIn("not parsed", problems[0][1])

    def test_fence_in_a_blockquote_is_reported_not_skipped(self):
        checked, problems = checker.snippets("> ```rust\n> let a = 1;\n> ```\n")
        self.assertEqual(checked, [])
        self.assertEqual([line for line, _ in problems], [1])
        self.assertIn("2 fence line(s)", problems[0][1])

    def test_nested_longer_fence_is_reported(self):
        # The three-backtick lines are body of the four-backtick fence, not fences of their
        # own, and the checker says so rather than compiling them as code.
        _, problems = checker.snippets("````rust\n```\nlet a = 1;\n```\n````\n")
        self.assertEqual([line for line, _ in problems], [2])


class MarkdownFilesTests(unittest.TestCase):
    def test_the_readme_is_checked_with_the_docs(self):
        files = checker.markdown_files()
        self.assertEqual(files[0], checker.REPO / "README.md")
        self.assertIn(checker.REPO / "docs" / "udfs.md", files)


class MainReportTests(unittest.TestCase):
    def test_failures_are_reported_at_the_line_of_the_fence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "docs").mkdir()
            doc = root / "docs" / "a.md"
            doc.write_text('Intro.\n\n```rust\nlet n: i64 = "bad";\n```\n\n```rust,ignore\n```\n')
            rlib = root / "libspark_connect.rlib"
            rlib.write_text("")
            rustc_error = "error[E0308]: mismatched types\n --> snippet_1.rs:11:5\n"
            with (
                patch.object(checker, "REPO", root),
                patch.object(checker, "markdown_files", return_value=[doc]),
                patch.object(checker, "crate_artifacts", return_value=(rlib, [])),
                patch.object(checker, "check_block", return_value=(False, rustc_error)),
                patch("sys.stdout", new_callable=io.StringIO) as out,
            ):
                self.assertEqual(checker.main(), 1)
        lines = out.getvalue().splitlines()
        self.assertEqual(
            lines[0], "Checked 1 rust snippet(s); 2 failed to compile or were refused."
        )
        self.assertEqual(lines[1], "  docs/a.md:3: error[E0308]: mismatched types")
        self.assertTrue(lines[2].startswith('  docs/a.md:7: fence info "rust,ignore"'))
        self.assertEqual(len(lines), 3)


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


if __name__ == "__main__":
    unittest.main()
