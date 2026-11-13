"""Tests for scripts/gen_binary_licenses.py, on a synthetic `cargo metadata` document and on
synthetic `cargo tree` outputs, one per wheel target."""

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"
ROOT_TREE = "pyspark-rs v4.3.0 (/repo/crates/pyspark-rs)\n"
# mymacro is a proc-macro that a normal edge reaches. devlib is a dev dependency, so no tree
# lists it. extra is in cargo metadata only.
TREE_LINUX = ROOT_TREE + "mitlib v1.0.0\nduallib v2.0.0 (*)\nmymacro v0.1.0\n"
# winlib is linked for Windows only, and mitlib appears again under another edge.
TREE_WINDOWS = ROOT_TREE + "mitlib v1.0.0 (*)\nwinlib v0.3.0\n"


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


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


class NoticeBlockTests(unittest.TestCase):
    def test_block_holding_the_marker_is_taken_not_the_origin_block(self):
        text = (
            "/* origin: example/s_sin.c */\n"
            "/*\n"
            " * ====================\n"
            " * Copyright (C) 1993 by Sun Microsystems, Inc.\n"
            " * ====================\n"
            " */\n"
            "\n"
            "fn f() {}\n"
        )
        self.assertEqual(
            gen._notice_block(text, "Sun Microsystems"),
            "/*\n * ====================\n"
            " * Copyright (C) 1993 by Sun Microsystems, Inc.\n"
            " * ====================\n */\n",
        )

    def test_line_comment_run_ends_at_a_blank_line(self):
        text = "// Copyright 2024 Example.\n// Permission is granted.\n\n// Not a notice.\n"
        self.assertEqual(
            gen._notice_block(text, "Example"),
            "// Copyright 2024 Example.\n// Permission is granted.\n",
        )

    def test_perl_comments_and_missing_marker(self):
        text = "#!/usr/bin/env perl\n# Copyright 2015 Example.\n# Permission is granted.\n"
        self.assertEqual(
            gen._notice_block(text, "Example"),
            "# Copyright 2015 Example.\n# Permission is granted.\n",
        )
        self.assertIsNone(gen._notice_block(text, "Nobody"))

    def test_comment_styles_of_one_notice_have_one_key(self):
        c_style = "/*\n * Copyright (c) 2014 Example.\n * Permission is granted.\n */\n"
        line_style = "// Copyright (c) 2014 Example.\n// Permission is granted.\n"
        self.assertEqual(gen._notice_key(c_style), gen._notice_key(line_style))
        other_year = c_style.replace("2014", "2015")
        self.assertNotEqual(gen._notice_key(c_style), gen._notice_key(other_year))


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.src = os.path.join(self.tmp, "registry")
        self.root = os.path.join(self.tmp, "repo")
        os.makedirs(self.root)
        # The synthetic crates have no notices of their own; the tests that need some add them.
        patcher = mock.patch.dict(gen.SUPPLEMENTARY_NOTICES, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _crate(self, name, version, files):
        d = os.path.join(self.src, f"{name}-{version}")
        os.makedirs(d, exist_ok=True)
        for fname, text in files.items():
            path = os.path.join(d, fname)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _write(path, text)
        return os.path.join(d, "Cargo.toml")

    def _metadata(self):
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
                "source": REGISTRY,
                "license": "MIT",
                "targets": [{"kind": ["lib"]}],
                "manifest_path": self._crate("mitlib", "1.0.0", {"LICENSE": MIT_TEXT}),
            },
            {
                "id": "duallib",
                "name": "duallib",
                "version": "2.0.0",
                "source": REGISTRY,
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
                "source": REGISTRY,
                "license": "MIT",
                "targets": [{"kind": ["proc-macro"]}],
                "manifest_path": self._crate("mymacro", "0.1.0", {"LICENSE": MIT_TEXT}),
            },
            {
                "id": "devlib",
                "name": "devlib",
                "version": "0.1.0",
                "source": REGISTRY,
                "license": "BSD-3-Clause",
                "targets": [{"kind": ["lib"]}],
                "manifest_path": self._crate("devlib", "0.1.0", {"LICENSE": "BSD"}),
            },
            {
                "id": "extra",
                "name": "extra",
                "version": "0.2.0",
                "source": REGISTRY,
                "license": "MIT",
                "targets": [{"kind": ["lib"]}],
                "manifest_path": self._crate("extra", "0.2.0", {"LICENSE": MIT_TEXT}),
            },
            {
                "id": "winlib",
                "name": "winlib",
                "version": "0.3.0",
                "source": REGISTRY,
                "license": "MIT",
                "targets": [{"kind": ["lib"]}],
                "manifest_path": self._crate("winlib", "0.3.0", {"LICENSE": MIT_TEXT}),
            },
        ]
        return {"packages": packages}

    def _trees(self):
        return [TREE_LINUX, TREE_WINDOWS]

    def _outputs(self):
        return gen.build_outputs(self._metadata(), ROOT_LICENSE, ROOT_NOTICE, self._trees())

    def test_union_of_the_wheel_targets_lists_linked_crates_only(self):
        license_text = self._outputs()["LICENSE-binary"]
        self.assertIn("mitlib 1.0.0\n", license_text)
        self.assertIn("duallib 2.0.0 (MIT OR Apache-2.0)\n", license_text)
        # Linked for Windows only, so it is listed once it is in any target's tree.
        self.assertIn("winlib 0.3.0\n", license_text)
        # A proc-macro, a dev dependency, and a crate that no target lists.
        self.assertNotIn("mymacro", license_text)
        self.assertNotIn("devlib", license_text)
        self.assertNotIn("extra", license_text)

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

    def test_tree_line_that_is_not_a_package_is_rejected(self):
        with self.assertRaises(SystemExit):
            gen.wheel_packages(self._metadata(), ["this is not a tree line\n"])

    def test_tree_line_for_an_unknown_package_is_rejected(self):
        with self.assertRaises(SystemExit):
            gen.wheel_packages(self._metadata(), [ROOT_TREE + "nosuch v1.0.0\n"])

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


class SupplementaryNoticeTests(OutputTests):
    """SUPPLEMENTARY_NOTICES copies the notices of a crate's files into LICENSE-binary."""

    C_NOTICE = (
        "/* origin: example/notice.c\n"
        " * Copyright (c) 2020 Example Corp.\n"
        " * Permission is hereby granted, free of charge.\n"
        " */\n"
        "\n"
        "int notice(void);\n"
    )
    C_NOTICE_BLOCK = (
        "/* origin: example/notice.c\n"
        " * Copyright (c) 2020 Example Corp.\n"
        " * Permission is hereby granted, free of charge.\n"
        " */\n"
    )
    RS_NOTICE = (
        "// origin: example/other.c\n"
        "// Copyright (c) 2020 Example Corp.\n"
        "// Permission is hereby granted, free of charge.\n"
        "\n"
        "fn other() {}\n"
    )

    def _with_files(self):
        self._crate(
            "mitlib",
            "1.0.0",
            {
                "LICENSE": MIT_TEXT,
                "src/notice.c": self.C_NOTICE,
                "src/other.rs": self.RS_NOTICE,
                "src/plain.rs": "fn plain() {}\n",
            },
        )

    def _outputs_with(self, entries):
        table = {("mitlib", "1.0.0"): entries}
        with mock.patch.dict(gen.SUPPLEMENTARY_NOTICES, table, clear=True):
            return self._outputs()

    def test_notice_is_copied_verbatim_under_its_crate(self):
        self._with_files()
        text = self._outputs_with([("src/notice.c", "Example Corp")])["LICENSE-binary"]
        self.assertIn(
            "The notice in the source of mitlib 1.0.0 (MIT), in src/notice.c:\n\n"
            + self.C_NOTICE_BLOCK,
            text,
        )

    def test_one_notice_in_two_comment_styles_is_listed_once(self):
        self._with_files()
        text = self._outputs_with([("src/notice.c", "Example Corp"), ("src/other.rs", "Example")])[
            "LICENSE-binary"
        ]
        self.assertEqual(text.count("The notice in the source of mitlib"), 1)
        self.assertIn("in src/notice.c, src/other.rs:", text)

    def test_glob_skips_files_without_the_marker(self):
        self._with_files()
        text = self._outputs_with([("src/*.rs", "Example Corp")])["LICENSE-binary"]
        self.assertIn("in src/other.rs:", text)
        self.assertNotIn("plain", text)

    def test_plain_path_without_the_marker_is_an_error(self):
        self._with_files()
        with self.assertRaises(SystemExit):
            self._outputs_with([("src/plain.rs", "Example Corp")])

    def test_glob_without_a_match_is_an_error(self):
        self._with_files()
        with self.assertRaises(SystemExit):
            self._outputs_with([("src/*.zig", "Example Corp")])

    def test_whole_file_is_the_notice_without_a_marker(self):
        self._with_files()
        text = self._outputs_with([("src/plain.rs", None)])["LICENSE-binary"]
        self.assertIn("in src/plain.rs:\n\nfn plain() {}\n", text)

    def test_entry_for_a_crate_that_is_not_linked_is_an_error(self):
        self._with_files()
        table = {("nosuch", "9.9.9"): [("src/notice.c", None)]}
        with mock.patch.dict(gen.SUPPLEMENTARY_NOTICES, table, clear=True):
            with self.assertRaises(SystemExit):
                self._outputs()

    def test_check_fails_when_a_noticed_file_changes(self):
        self._with_files()
        entries = [("src/notice.c", "Example Corp")]
        outputs = self._outputs_with(entries)
        gen.write(self.root, outputs)
        self.assertEqual(gen.check(self.root, outputs), [])

        # A crate update that changes its notice makes the committed files stale.
        _write(
            os.path.join(self.src, "mitlib-1.0.0", "src", "notice.c"),
            self.C_NOTICE.replace("Example Corp.", "Example Corp Ltd."),
        )
        outputs = self._outputs_with(entries)
        self.assertEqual(gen.check(self.root, outputs), ["LICENSE-binary"])


if __name__ == "__main__":
    unittest.main()
