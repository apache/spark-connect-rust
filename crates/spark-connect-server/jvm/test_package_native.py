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

"""Artifact checks use synthetic headers; CI separately loads the real release library."""

import hashlib
import importlib.util
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


SPEC = importlib.util.spec_from_file_location("package_native", Path(__file__).with_name("package-native.py"))
packaging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packaging)


def library_header(classifier):
    _, kind, machine = packaging.PLATFORMS[classifier]
    data = bytearray(80)
    if kind == "elf":
        data[:7] = b"\x7fELF\x02\x01\x01"
        struct.pack_into("<HHI", data, 16, 3, machine, 1)
    else:
        data[:4] = b"\xcf\xfa\xed\xfe"
        struct.pack_into("<I", data, 4, machine)
        struct.pack_into("<I", data, 12, 6)
    return bytes(data)


class NativePackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "output"
        self.output.mkdir()
        self.version = packaging.artifact_version()

    def package(self, classifier, payload=b""):
        directory = self.directory / classifier
        directory.mkdir(exist_ok=True)
        library = directory / packaging.PLATFORMS[classifier][0]
        library.write_bytes(library_header(classifier) + payload)
        return packaging.package_library(library, self.output, classifier, self.version)

    def rewrite(self, jar, transform):
        with zipfile.ZipFile(jar) as archive:
            entries = {name: archive.read(name) for name in archive.namelist()}
        transform(entries)
        return packaging.write_jar(jar, entries)

    def test_all_platform_headers_are_checked(self):
        for classifier in packaging.PLATFORMS:
            with self.subTest(classifier=classifier):
                packaging.validate_library(library_header(classifier), classifier)
                for other in packaging.PLATFORMS:
                    if other != classifier:
                        with self.assertRaises(ValueError):
                            packaging.validate_library(library_header(other), classifier)

    def test_executable_and_32_bit_headers_are_rejected(self):
        elf = bytearray(library_header("linux-x86_64"))
        struct.pack_into("<H", elf, 16, 2)
        with self.assertRaisesRegex(ValueError, "shared-library"):
            packaging.validate_library(elf, "linux-x86_64")
        elf[4] = 1
        with self.assertRaisesRegex(ValueError, "64-bit"):
            packaging.validate_library(elf, "linux-x86_64")
        macho = bytearray(library_header("macos-aarch64"))
        struct.pack_into("<I", macho, 12, 2)
        with self.assertRaisesRegex(ValueError, "dynamic-library"):
            packaging.validate_library(macho, "macos-aarch64")

    def test_host_platform_aliases_and_unsupported_host(self):
        for system, machine, expected in [
            ("Linux", "amd64", "linux-x86_64"),
            ("Linux", "aarch64", "linux-aarch64"),
            ("Darwin", "arm64", "macos-aarch64"),
            ("Darwin", "x86_64", "macos-x86_64"),
        ]:
            with patch.object(packaging.platform, "system", return_value=system), \
                    patch.object(packaging.platform, "machine", return_value=machine):
                self.assertEqual(packaging.host_platform(), expected)
        with patch.object(packaging.platform, "system", return_value="Windows"):
            with self.assertRaises(ValueError):
                packaging.host_platform()

    def test_platform_jar_contents_digest_and_reproducibility(self):
        jar = self.package("linux-x86_64")
        first = jar.read_bytes()
        self.package("linux-x86_64")
        self.assertEqual(jar.read_bytes(), first)
        self.assertEqual(stat.S_IMODE(jar.stat().st_mode), 0o644)
        with zipfile.ZipFile(jar) as archive:
            prefix = "META-INF/native/linux-x86_64/"
            metadata = packaging.parse_properties(archive.read(prefix + "native.properties"))
            data = archive.read(prefix + "libspark_connect_jni.so")
            self.assertEqual(metadata["library.sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(metadata["artifact.version"], self.version)
            self.assertEqual(metadata["native.abi.version"], "1")
            self.assertEqual(archive.read("META-INF/LICENSE"), packaging.licenses()["META-INF/LICENSE"])
            self.assertTrue(all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist()))
        self.assertEqual(jar.with_suffix(".jar.sha256").read_text(),
                         f"{hashlib.sha256(first).hexdigest()}  {jar.name}\n")

    def test_invalid_input_path_and_wrong_filename_fail(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            packaging.package_library(Path("relative.so"), self.output, "linux-x86_64", self.version)
        invalid = self.directory / "something.so"
        invalid.write_bytes(library_header("linux-x86_64"))
        with self.assertRaisesRegex(ValueError, "named"):
            packaging.package_library(invalid, self.output, "linux-x86_64", self.version)

    def test_native_jar_compresses_entries_without_changing_bytes_or_digest(self):
        classifier = "macos-aarch64"
        payload = bytes(range(256)) * 2048
        data = library_header(classifier) + payload
        jar = self.package(classifier, payload)
        prefix = f"META-INF/native/{classifier}/"
        with zipfile.ZipFile(jar) as archive:
            self.assertTrue(all(info.compress_type == zipfile.ZIP_DEFLATED
                                for info in archive.infolist()))
            library = archive.getinfo(prefix + "libspark_connect_jni.dylib")
            self.assertLess(library.compress_size, library.file_size // 4)
            self.assertEqual(archive.read(library), data)
            metadata = packaging.parse_properties(archive.read(prefix + "native.properties"))
            self.assertEqual(metadata["library.sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(metadata["artifact.version"], self.version)
            self.assertEqual(metadata["native.abi.version"], "1")
            self.assertEqual(archive.read("META-INF/LICENSE"), packaging.licenses()["META-INF/LICENSE"])

    def test_combine_preserves_four_validated_platforms(self):
        jars = [self.package(classifier) for classifier in packaging.PLATFORMS]
        combined = packaging.combine_jars(jars, self.output, self.version)
        first = combined.read_bytes()
        self.assertEqual(stat.S_IMODE(combined.stat().st_mode), 0o644)
        packaging.combine_jars(reversed(jars), self.output, self.version)
        self.assertEqual(combined.read_bytes(), first)
        with zipfile.ZipFile(combined) as archive:
            for classifier, (filename, _, _) in packaging.PLATFORMS.items():
                self.assertEqual(archive.read(f"META-INF/native/{classifier}/{filename}"),
                                 library_header(classifier))

    def test_combine_rejects_duplicate_platforms(self):
        jar = self.package("macos-aarch64")
        with self.assertRaisesRegex(ValueError, "duplicate platform"):
            packaging.combine_jars([jar, jar], self.output, self.version)

    def test_combine_rejects_subset_before_writing_all_platform_jar(self):
        jars = [self.package(classifier) for classifier in list(packaging.PLATFORMS)[:3]]
        with self.assertRaisesRegex(ValueError, "missing: macos-aarch64"):
            packaging.combine_jars(jars, self.output, self.version)
        self.assertFalse((self.output / f"spark-connect-native-{self.version}-all.jar").exists())

    def test_macos_universal_preserves_both_platforms_and_is_reproducible(self):
        classifiers = ["macos-x86_64", "macos-aarch64"]
        jars = [self.package(classifier) for classifier in classifiers]
        combined = packaging.combine_jars(jars, self.output, self.version, macos_only=True)
        self.assertEqual(combined.name, f"spark-connect-native-{self.version}-macos-universal.jar")
        self.assertEqual(stat.S_IMODE(combined.stat().st_mode), 0o644)
        first = combined.read_bytes()
        packaging.combine_jars(reversed(jars), self.output, self.version, macos_only=True)
        self.assertEqual(combined.read_bytes(), first)
        with zipfile.ZipFile(combined) as archive:
            expected_names = set(packaging.licenses())
            for classifier, jar in zip(classifiers, jars):
                prefix = f"META-INF/native/{classifier}/"
                expected_names.update({prefix + "native.properties",
                                       prefix + "libspark_connect_jni.dylib"})
                with zipfile.ZipFile(jar) as original:
                    for name in original.namelist():
                        self.assertEqual(archive.read(name), original.read(name))
            self.assertEqual(set(archive.namelist()), expected_names)
            self.assertTrue(all(info.compress_type == zipfile.ZIP_DEFLATED
                                for info in archive.infolist()))
        self.assertEqual(combined.with_suffix(".jar.sha256").read_text(),
                         f"{hashlib.sha256(first).hexdigest()}  {combined.name}\n")

    def test_macos_universal_rejects_missing_platform_before_writing(self):
        jars = [self.package(classifier) for classifier in ["macos-x86_64", "macos-aarch64"]]
        combined = self.output / f"spark-connect-native-{self.version}-macos-universal.jar"
        for incomplete in [[], jars[:1], jars[1:]]:
            with self.subTest(inputs=[jar.name for jar in incomplete]):
                with self.assertRaisesRegex(ValueError, "missing"):
                    packaging.combine_jars(incomplete, self.output, self.version, macos_only=True)
                self.assertFalse(combined.exists())

    def test_macos_universal_rejects_duplicate_platform_before_writing(self):
        jar = self.package("macos-aarch64")
        combined = self.output / f"spark-connect-native-{self.version}-macos-universal.jar"
        with self.assertRaisesRegex(ValueError, "duplicate platform"):
            packaging.combine_jars([jar, jar], self.output, self.version, macos_only=True)
        self.assertFalse(combined.exists())

    def test_macos_universal_rejects_linux_inputs_before_writing(self):
        macos = [self.package(classifier) for classifier in ["macos-x86_64", "macos-aarch64"]]
        combined = self.output / f"spark-connect-native-{self.version}-macos-universal.jar"
        for classifier in ["linux-x86_64", "linux-aarch64"]:
            jar = self.package(classifier)
            for inputs in [[jar], macos + [jar]]:
                with self.subTest(inputs=[item.name for item in inputs]):
                    with self.assertRaisesRegex(ValueError, "does not accept platform: linux-"):
                        packaging.combine_jars(inputs, self.output, self.version, macos_only=True)
                    self.assertFalse(combined.exists())

    def test_combine_rejects_version_and_abi_mismatch(self):
        for key, value, message in [
            ("artifact.version", "99.0.0", "artifact versions must match"),
            ("native.abi.version", "99", "unsupported native ABI version"),
        ]:
            jar = self.package("linux-aarch64")
            def alter(entries):
                name = "META-INF/native/linux-aarch64/native.properties"
                values = packaging.parse_properties(entries[name])
                values[key] = value
                entries[name] = packaging.properties(values)
            self.rewrite(jar, alter)
            with self.assertRaisesRegex(ValueError, message):
                packaging.combine_jars([jar], self.output, self.version)

    def test_version_reader_accepts_literal_semver_and_is_scoped_to_package(self):
        for version in ["0.1.0", "0.1.0+build", "1.2.3-rc.1+build.007", "20.10.0"]:
            with self.subTest(version=version):
                manifest = ('[workspace.package]\nversion = "99.0.0"\n'
                            f'[package] # package metadata\nversion = "{version}" # comment\n'
                            '[dependencies]\nversion = "not-a-package-version"\n')
                self.assertEqual(packaging.version_from_manifest(manifest), version)

    def test_version_reader_rejects_invalid_and_ambiguous_forms(self):
        for value in ["1", "1.2", "01.2.3", "0.1.0_foo", "0.1.0+", "0.1.0-01", "../0.1.0"]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "literal SemVer package version"):
                    packaging.version_from_manifest(f'[package]\nversion = "{value}"\n')
        for manifest in [
            '[workspace.package]\nversion = "0.1.0"\n',
            '[package]\nversion.workspace = true\n',
            '[package]\nversion = "0.1.0"\nversion = "0.2.0"\n',
            '[package]\nversion = "0.1.0"\nversion.workspace = true\n',
            '[package]\nversion = "0.1.0"\n[package]\nversion = "0.1.0"\n',
            '[package]\nversion = "0.1.0\\u002bbuild"\n',
        ]:
            with self.subTest(manifest=manifest):
                with self.assertRaisesRegex(ValueError, "literal SemVer package version"):
                    packaging.version_from_manifest(manifest)

    def test_cli_print_version_needs_no_output_directory(self):
        completed = subprocess.run([sys.executable, "-B", str(Path(packaging.__file__)),
                                    "--print-version"], text=True, capture_output=True, check=True)
        self.assertEqual(completed.stdout, self.version + "\n")
        self.assertEqual(completed.stderr, "")

    def test_cli_packaging_requires_output_directory_and_rejects_mixed_version_mode(self):
        script = str(Path(packaging.__file__))
        for mode in ["--library", "--combine", "--combine-macos"]:
            completed = subprocess.run([sys.executable, "-B", script, mode, "/unused"],
                                       text=True, capture_output=True)
            self.assertEqual(completed.returncode, 2)
            self.assertIn("--output-dir is required for packaging", completed.stderr)
        completed = subprocess.run([sys.executable, "-B", script, "--print-version",
                                    "--output-dir", str(self.directory / "unwanted")],
                                   text=True, capture_output=True)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--print-version cannot be combined", completed.stderr)
        self.assertFalse((self.directory / "unwanted").exists())

    def test_cli_macos_combine_creates_valid_universal_jar(self):
        jars = [self.package(classifier) for classifier in ["macos-x86_64", "macos-aarch64"]]
        completed = subprocess.run([sys.executable, "-B", str(Path(packaging.__file__)),
                                    "--combine-macos", *map(str, jars),
                                    "--output-dir", str(self.output)],
                                   text=True, capture_output=True, check=True)
        combined = self.output / f"spark-connect-native-{self.version}-macos-universal.jar"
        self.assertEqual(completed.stdout, str(combined) + "\n")
        self.assertEqual(completed.stderr, "")
        with zipfile.ZipFile(combined) as archive:
            for classifier, jar in zip(["macos-x86_64", "macos-aarch64"], jars):
                prefix = f"META-INF/native/{classifier}/"
                with zipfile.ZipFile(jar) as original:
                    self.assertEqual(archive.read(prefix + "libspark_connect_jni.dylib"),
                                     original.read(prefix + "libspark_connect_jni.dylib"))

    def test_cli_combine_modes_reject_platform_override(self):
        jars = {classifier: self.package(classifier) for classifier in packaging.PLATFORMS}
        for mode, inputs in [("--combine", list(jars.values())),
                             ("--combine-macos", [jars["macos-x86_64"], jars["macos-aarch64"]])]:
            with self.subTest(mode=mode):
                completed = subprocess.run([sys.executable, "-B", str(Path(packaging.__file__)),
                                            mode, *map(str, inputs), "--platform", "macos-aarch64",
                                            "--output-dir", str(self.output)],
                                           text=True, capture_output=True)
                self.assertEqual(completed.returncode, 2)
                self.assertIn("--platform cannot be used", completed.stderr)

    def test_cli_macos_combine_and_print_version_are_mutually_exclusive(self):
        completed = subprocess.run([sys.executable, "-B", str(Path(packaging.__file__)),
                                    "--combine-macos", "/unused", "--print-version"],
                                   text=True, capture_output=True)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("not allowed", completed.stderr)

    def test_combine_rejects_tampered_library(self):
        jar = self.package("linux-x86_64")
        def alter(entries):
            entries["META-INF/native/linux-x86_64/libspark_connect_jni.so"] += b"changed"
        self.rewrite(jar, alter)
        with self.assertRaisesRegex(ValueError, "checksum"):
            packaging.combine_jars([jar], self.output, self.version)

    def test_combine_rejects_unexpected_resource(self):
        jar = self.package("macos-x86_64")
        self.rewrite(jar, lambda entries: entries.update({"../../unwanted": b"x"}))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            packaging.combine_jars([jar], self.output, self.version)

    def test_duplicate_archive_entries_are_rejected(self):
        jar = self.package("linux-x86_64")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(jar, "a") as archive:
                archive.writestr("META-INF/LICENSE", b"duplicate")
        with self.assertRaisesRegex(ValueError, "duplicate JAR"):
            packaging.combine_jars([jar], self.output, self.version)

    def test_properties_reject_ambiguity_and_large_metadata(self):
        for data in [b"key=1\nkey=2\n", b"key =1\n", b"key=\\u0031\n", b"\xff"]:
            with self.assertRaises(ValueError):
                packaging.parse_properties(data)
        with self.assertRaisesRegex(ValueError, "16 KiB"):
            packaging.parse_properties(b"#" * (packaging.MAX_METADATA_SIZE + 1))

    def test_adapter_version_checked_before_copy(self):
        adapter = self.directory / "adapter.jar"
        packaging.write_jar(adapter, {packaging.ADAPTER_METADATA: packaging.properties({
            "artifact.version": self.version, "native.abi.version": "1"}), "Example.class": b"test"})
        copied = packaging.copy_adapter(adapter, self.output, self.version)
        self.assertEqual(adapter.read_bytes(), copied.read_bytes())
        self.rewrite(adapter, lambda entries: entries.update({packaging.ADAPTER_METADATA:
                     packaging.properties({"artifact.version": "other", "native.abi.version": "1"})}))
        with self.assertRaisesRegex(ValueError, "versions"):
            packaging.copy_adapter(adapter, self.output, self.version)


if __name__ == "__main__":
    unittest.main()
