"""Tests for scripts/gen_binary_licenses.py, on a synthetic `cargo metadata` document."""

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "gen_binary_licenses.py"
SPEC = importlib.util.spec_from_file_location("gen_binary_licenses", SCRIPT)
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)

MIT_TEXT = "Copyright (c) 2024 Example Author\n\nPermission is hereby granted, free of charge.\n"
APACHE_HEAD = (
    "                                 Apache License\n"
    "                           Version 2.0, January 2004\n"
)
DUAL_NOTICE = "Example Notice\nCopyright 2024 Example Authors\n"

ROOT_LICENSE = (
    APACHE_HEAD + "\nTERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n\n"
    "   limitations under the License.\n\n" + "-" * 84 + "\n"
    "This product bundles various third-party components under other open source licenses.\n"
    "This section summarizes those components and their licenses.\n"
)
ROOT_NOTICE = "Apache Spark Connect Rust Client\nCopyright 2023 and onwards\n"


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class ExpressionTests(unittest.TestCase):
    def test_apache_option_permits_apache(self):
        for expr, expected in [
            ("Apache-2.0", True),
            ("MIT OR Apache-2.0", True),
            ("MIT/Apache-2.0", True),
            ("Apache-2.0 OR BSL-1.0", True),
            ("Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT", True),
            ("MIT", False),
            ("Apache-2.0 AND MIT", False),
            ("(MIT OR Apache-2.0) AND Unicode-3.0", False),
            ("Unlicense OR MIT", False),
        ]:
            with self.subTest(expr=expr):
                self.assertEqual(gen.permits_apache(gen._parse_expr(expr)), expected)

    def test_mentions_apache_anywhere(self):
        self.assertTrue(gen.mentions_apache(gen._parse_expr("Apache-2.0 AND MIT")))
        self.assertFalse(gen.mentions_apache(gen._parse_expr("(MIT OR ISC) AND Zlib")))

    def test_malformed_expression_is_rejected(self):
        for expr in ["(MIT OR Apache-2.0", "MIT OR", "AND MIT"]:
            with self.subTest(expr=expr), self.assertRaises(ValueError):
                gen._parse_expr(expr)


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = os.path.join(self.tmp, "registry")
        self.root = os.path.join(self.tmp, "repo")
        os.makedirs(self.root)

    def _crate(self, name, version, files):
        d = os.path.join(self.src, f"{name}-{version}")
        os.makedirs(d)
        for fname, text in files.items():
            with open(os.path.join(d, fname), "w", encoding="utf-8") as f:
                f.write(text)
        return os.path.join(d, "Cargo.toml")

    def _metadata(self):
        registry = "registry+https://github.com/rust-lang/crates.io-index"
        packages = [
            {
                "id": "pyspark-rs",
                "name": "pyspark-rs",
                "version": "4.3.0",
                "source": None,
                "license": "Apache-2.0",
                "targets": [{"kind": ["cdylib"]}],
                "manifest_path": os.path.join(self.root, "crates", "pyspark-rs", "Cargo.toml"),
            },
            {
                "id": "mitlib",
                "name": "mitlib",
                "version": "1.0.0",
                "source": registry,
                "license": "MIT",
                "targets": [{"kind": ["lib"]}],
                "manifest_path": self._crate("mitlib", "1.0.0", {"LICENSE": MIT_TEXT}),
            },
            {
                "id": "duallib",
                "name": "duallib",
                "version": "2.0.0",
                "source": registry,
                "license": "MIT OR Apache-2.0",
                "targets": [{"kind": ["lib"]}],
                "manifest_path": self._crate(
                    "duallib",
                    "2.0.0",
                    {"LICENSE-MIT": MIT_TEXT, "LICENSE-APACHE": APACHE_HEAD, "NOTICE": DUAL_NOTICE},
                ),
            },
            {
                "id": "mymacro",
                "name": "mymacro",
                "version": "0.1.0",
                "source": registry,
                "license": "MIT",
                "targets": [{"kind": ["proc-macro"]}],
                "manifest_path": self._crate("mymacro", "0.1.0", {"LICENSE": MIT_TEXT}),
            },
            {
                "id": "devlib",
                "name": "devlib",
                "version": "0.1.0",
                "source": registry,
                "license": "BSD-3-Clause",
                "targets": [{"kind": ["lib"]}],
                "manifest_path": self._crate("devlib", "0.1.0", {"LICENSE": "BSD"}),
            },
        ]

        def dep(pkg_id, kind=None):
            return {"pkg": pkg_id, "dep_kinds": [{"kind": kind, "target": None}]}

        nodes = [
            {
                "id": "pyspark-rs",
                "deps": [dep("mitlib"), dep("duallib"), dep("mymacro"), dep("devlib", "dev")],
            },
            {"id": "mitlib", "deps": []},
            {"id": "duallib", "deps": []},
            {"id": "mymacro", "deps": []},
            {"id": "devlib", "deps": []},
        ]
        return {"packages": packages, "resolve": {"nodes": nodes}}

    def _outputs(self):
        return gen.build_outputs(self._metadata(), ROOT_LICENSE, ROOT_NOTICE)

    def test_only_linked_non_proc_macro_crates_are_listed(self):
        out = self._outputs()
        license_text = out["LICENSE-binary"]
        self.assertIn("mitlib 1.0.0\n", license_text)
        self.assertIn("duallib 2.0.0 (MIT OR Apache-2.0)\n", license_text)
        self.assertNotIn("mymacro", license_text)
        self.assertNotIn("devlib", license_text)

    def test_non_apache_components_carry_their_license_texts(self):
        license_text = self._outputs()["LICENSE-binary"]
        self.assertIn(
            "The license of mitlib 1.0.0 (MIT), from LICENSE:\n\n" + MIT_TEXT, license_text
        )
        # The dual crate is used under Apache-2.0, so its texts are not bundled.
        self.assertNotIn("duallib 2.0.0 (MIT OR Apache-2.0), from", license_text)
        self.assertNotIn("mymacro", license_text)

    def test_notice_is_bundled_for_apache_components_only(self):
        notice = self._outputs()["NOTICE-binary"]
        self.assertTrue(notice.startswith(ROOT_NOTICE))
        self.assertIn("=== NOTICE FOR duallib 2.0.0 ===", notice)
        self.assertIn(DUAL_NOTICE, notice)
        self.assertNotIn("mitlib", notice)

    def test_check_passes_on_written_files_and_fails_after_an_edit(self):
        outputs = self._outputs()
        gen.write(self.root, outputs)
        self.assertEqual(gen.check(self.root, outputs), [])

        # Dropping a component from the summary makes LICENSE-binary stale.
        license_path = os.path.join(self.root, "LICENSE-binary")
        text = _read(license_path)
        with open(license_path, "w", encoding="utf-8") as f:
            f.write(text.replace("mitlib 1.0.0\n", ""))
        self.assertEqual(gen.check(self.root, outputs), ["LICENSE-binary"])

        # Dropping a component's license text makes it stale too.
        gen.write(self.root, outputs)
        with open(license_path, "w", encoding="utf-8") as f:
            f.write(_read(license_path).replace(MIT_TEXT, ""))
        self.assertEqual(gen.check(self.root, outputs), ["LICENSE-binary"])

        # A missing NOTICE-binary is stale as well.
        gen.write(self.root, outputs)
        os.remove(os.path.join(self.root, "NOTICE-binary"))
        self.assertEqual(gen.check(self.root, outputs), ["NOTICE-binary"])


if __name__ == "__main__":
    unittest.main()
