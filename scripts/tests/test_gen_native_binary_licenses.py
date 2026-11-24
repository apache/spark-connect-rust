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

"""JNI license generation tests use distinct target closures and real license obligations."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "gen_native_binary_licenses.py"
SPEC = importlib.util.spec_from_file_location("gen_native_binary_licenses", SCRIPT)
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)

MIT = (
    "Copyright (c) 2026 Example Contributors\n\n"
    "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
    "of this software and associated documentation files.\n"
)
NOTICE = "Apache Example\nCopyright 2026 The Apache Software Foundation\n"
APACHE = "Apache License\nVersion 2.0, January 2004\n"
ROOT_LICENSE = APACHE + "\n----\nThis product bundles various third-party components\n"


class NativeLicenseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in gen.SOURCE_FILES:
            self.write(name, "test source\n")
        self.write("LICENSE", ROOT_LICENSE)
        self.write("NOTICE", "Apache Spark Connect Rust Client\n")
        self.meta = {
            "packages": [
                self.package(
                    gen.ROOT_PACKAGE, "0.1.0", None, "crates/spark-connect-server/jni/Cargo.toml"
                ),
                self.package("mitlib", "1.0.0", "MIT"),
                self.package("apachelib", "2.0.0", "Apache-2.0", notice=NOTICE),
                self.package("maclib", "3.0.0", "MIT"),
                self.package("devlib", "4.0.0", "MIT"),
                self.package("buildlib", "5.0.0", "MIT"),
                self.package("procmacro", "6.0.0", "MIT", kind="proc-macro"),
            ]
        }
        self.trees = {}
        for target in gen.TARGETS:
            self.trees[target] = (
                f"{gen.ROOT_PACKAGE} v0.1.0 ({self.root}/crates/spark-connect-server/jni)\n"
                "mitlib v1.0.0\napachelib v2.0.0\nprocmacro v6.0.0\n"
                + ("maclib v3.0.0\n" if "apple" in target else "")
            )

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def package(self, name, version, license, manifest=None, notice=None, kind="lib"):
        if manifest is None:
            directory = f"registry/{name}-{version}"
            manifest = f"{directory}/Cargo.toml"
            self.write(f"{directory}/LICENSE", APACHE if license == "Apache-2.0" else MIT)
            if notice:
                self.write(f"{directory}/NOTICE", notice)
            self.write(manifest, "[package]\n")
        return {
            "name": name,
            "version": version,
            "license": license or "Apache-2.0",
            "source": "registry+test" if license else None,
            "manifest_path": str(self.root / manifest),
            "targets": [{"kind": [kind]}],
        }

    def generate(self):
        return gen.build_outputs(self.root, self.meta, self.trees)

    def store(self):
        outputs = self.generate()
        for name, text in outputs.items():
            self.write(name, text)
        return outputs

    def test_union_preserves_license_texts_notices_and_target_specific_membership(self):
        output = self.generate()
        license_text = output[(gen.OUTPUT / "LICENSE-binary").as_posix()]
        notice_text = output[(gen.OUTPUT / "NOTICE-binary").as_posix()]
        inventory = json.loads(output[(gen.OUTPUT / "dependencies.json").as_posix()])
        self.assertIn("mitlib 1.0.0\n", license_text)
        self.assertIn("maclib 3.0.0\n", license_text)
        self.assertIn(MIT, license_text)
        self.assertIn("apachelib 2.0.0 (Apache-2.0)", license_text)
        self.assertIn(NOTICE, notice_text)
        self.assertNotIn("Python and data", license_text)
        for target, closure in inventory["targets"].items():
            names = {p[0] for p in closure["registry"]}
            self.assertEqual(
                names,
                {"mitlib", "apachelib", "maclib"} if "apple" in target else {"mitlib", "apachelib"},
            )
            self.assertEqual(closure["workspace"], [[gen.ROOT_PACKAGE, "0.1.0"]])
        for excluded in ("devlib", "buildlib", "procmacro"):
            self.assertNotIn(excluded, license_text)

    def test_all_four_targets_and_correct_root_are_required(self):
        del self.trees[gen.TARGETS[0]]
        with self.assertRaisesRegex(ValueError, "four supported"):
            self.generate()
        self.trees[gen.TARGETS[0]] = "pyspark-rs v4.3.0 (/repo)\n"
        with self.assertRaisesRegex(ValueError, "unknown local package|missing spark-connect-jni"):
            self.generate()

    def test_cargo_tree_resolves_native_normal_edges_without_wheel_features(self):
        result = unittest.mock.Mock(returncode=0, stdout=b"tree", stderr=b"")
        with patch.object(gen.subprocess, "run", return_value=result) as run:
            gen.cargo_tree(self.root, gen.TARGETS[1])
        args = run.call_args.args[0]
        self.assertEqual(args[args.index("-p") + 1], gen.ROOT_PACKAGE)
        self.assertEqual(args[args.index("-e") + 1], "normal,no-proc-macro")
        self.assertEqual(args[args.index("--target") + 1], gen.TARGETS[1])
        self.assertNotIn("--features", args)
        self.assertIn("--locked", args)

    def test_supplementary_notices_are_scoped_and_version_changes_require_review(self):
        source_notice = "// Copyright (c) Embedded Author\n// Redistribution is permitted.\n"
        self.write("registry/mitlib-1.0.0/src/lib.rs", source_notice)
        table = {
            ("mitlib", "1.0.0"): [("src/lib.rs", "Embedded Author")],
            ("pyo3", "0.29.3"): [("unused", None)],
        }
        with patch.dict(gen.binary.SUPPLEMENTARY_NOTICES, table, clear=True):
            text = self.generate()[(gen.OUTPUT / "LICENSE-binary").as_posix()]
            self.assertIn(source_notice, text)
            self.assertNotIn("pyo3", text)
            with self.assertRaisesRegex(ValueError, "updated mitlib"):
                gen.scoped_notices(
                    [{"name": "mitlib", "version": "1.0.0"}, {"name": "mitlib", "version": "1.1.0"}]
                )

    def test_packaging_checks_content_hashes_and_cargo_inputs_without_cargo(self):
        self.store()
        with patch.object(gen.subprocess, "run", side_effect=AssertionError("Cargo is forbidden")):
            bundle = gen.load_bundle(self.root)
        self.assertIn(MIT.encode(), bundle["META-INF/LICENSE"])
        self.assertIn(NOTICE.encode(), bundle["META-INF/NOTICE"])
        self.write("Cargo.lock", "changed dependency graph\n")
        with self.assertRaisesRegex(ValueError, "stale native dependency licenses"):
            gen.load_bundle(self.root)

    def test_missing_assets_changed_text_and_inventory_mismatches_fail_closed(self):
        self.store()
        self.write(gen.OUTPUT / "LICENSE-binary", APACHE)
        with self.assertRaisesRegex(ValueError, "does not match"):
            gen.load_bundle(self.root)
        self.store()
        (self.root / gen.OUTPUT / "NOTICE-binary").unlink()
        with self.assertRaises(FileNotFoundError):
            gen.load_bundle(self.root)
        self.store()
        path = self.root / gen.OUTPUT / "dependencies.json"
        manifest = json.loads(path.read_text())
        del manifest["targets"][gen.TARGETS[2]]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "invalid native dependency license inventory"):
            gen.load_bundle(self.root)

    def test_check_detects_dropped_license_text_and_closure_change(self):
        outputs = self.store()
        self.assertEqual(gen.binary.check(str(self.root), outputs), [])
        self.write(gen.OUTPUT / "LICENSE-binary", APACHE)
        self.assertIn(
            (gen.OUTPUT / "LICENSE-binary").as_posix(),
            gen.binary.check(str(self.root), self.generate()),
        )
        self.store()
        self.trees[gen.TARGETS[0]] += "devlib v4.0.0\n"
        stale = gen.binary.check(str(self.root), self.generate())
        self.assertIn((gen.OUTPUT / "LICENSE-binary").as_posix(), stale)
        self.assertIn((gen.OUTPUT / "dependencies.json").as_posix(), stale)

    def test_saved_cargo_outputs_require_matching_manifest_and_lock_fingerprints(self):
        inputs = self.root / "saved-cargo-inputs"
        inputs.mkdir()
        (inputs / "metadata.json").write_text(json.dumps(self.meta))
        for target, tree in self.trees.items():
            (inputs / (target + ".txt")).write_text(tree)
        (inputs / "cargo-inputs.json").write_text(
            json.dumps(gen.input_hashes(self.root, gen.CARGO_INPUTS))
        )
        self.assertEqual(gen.generate(self.root, inputs), self.generate())
        self.write("crates/spark-connect-server/Cargo.toml", "changed dependency features\n")
        with self.assertRaisesRegex(ValueError, "saved native Cargo inputs are stale"):
            gen.generate(self.root, inputs)


if __name__ == "__main__":
    unittest.main()
