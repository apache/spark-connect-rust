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
import shutil
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


if __name__ == "__main__":
    unittest.main()
