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

"""Package already-built JNI libraries; this command does not cross-compile them."""

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import tempfile
import zipfile


PROJECT = Path(__file__).resolve().parent.parent
REPOSITORY = PROJECT.parent.parent
LICENSE_SPEC = importlib.util.spec_from_file_location(
    "native_binary_licenses", REPOSITORY / "scripts/gen_native_binary_licenses.py"
)
native_licenses = importlib.util.module_from_spec(LICENSE_SPEC)
LICENSE_SPEC.loader.exec_module(native_licenses)
ABI_VERSION = "1"
MAX_LIBRARY_SIZE = 512 * 1024 * 1024
MAX_METADATA_SIZE = 16 * 1024
PLATFORMS = {
    "linux-x86_64": ("libspark_connect_jni.so", "elf", 62),
    "linux-aarch64": ("libspark_connect_jni.so", "elf", 183),
    "macos-x86_64": ("libspark_connect_jni.dylib", "macho", 0x01000007),
    "macos-aarch64": ("libspark_connect_jni.dylib", "macho", 0x0100000C),
}
ADAPTER_METADATA = "META-INF/spark-connect-native-adapter.properties"
SEMVER_NUMBER = r"(?:0|[1-9][0-9]*)"
SEMVER_PRERELEASE = r"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
SEMVER = re.compile(
    rf"{SEMVER_NUMBER}\.{SEMVER_NUMBER}\.{SEMVER_NUMBER}"
    rf"(?:-{SEMVER_PRERELEASE}(?:\.{SEMVER_PRERELEASE})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def version_from_manifest(manifest):
    # This unpublished crate has a literal version; avoid requiring a TOML dependency or
    # Python 3.11 merely to read it. Fail closed if the manifest changes its version form.
    packages = re.findall(
        r"(?ms)^[ \t]*\[package\][ \t]*(?:#[^\n]*)?\n(.*?)(?=^[ \t]*\[|\Z)",
        manifest,
    )
    if len(packages) != 1:
        raise ValueError("JNI crate must declare one literal SemVer package version")
    declarations = re.findall(r"^[ \t]*version(?:[ \t]*=.*|\..*)$", packages[0], re.MULTILINE)
    if len(declarations) != 1:
        raise ValueError("JNI crate must declare one literal SemVer package version")
    literal = re.fullmatch(r'[ \t]*version[ \t]*=[ \t]*"([^"\\\r\n]*)"[ \t]*(?:#[^\r\n]*)?',
                           declarations[0])
    if literal is None or SEMVER.fullmatch(literal.group(1)) is None:
        raise ValueError("JNI crate must declare one literal SemVer package version")
    return literal.group(1)


def artifact_version():
    return version_from_manifest((PROJECT / "jni/Cargo.toml").read_text(encoding="utf-8"))


def host_platform():
    system = {"Linux": "linux", "Darwin": "macos"}.get(platform.system())
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "arm64": "aarch64",
            "aarch64": "aarch64"}.get(platform.machine().lower())
    if system is None or arch is None:
        raise ValueError("native JARs support only Linux/macOS on x86_64 or aarch64")
    return f"{system}-{arch}"


def validate_library(data, classifier):
    """Reject wrong-platform input rather than publishing a mislabeled artifact."""
    if classifier not in PLATFORMS:
        raise ValueError(f"unsupported platform: {classifier}")
    if len(data) > MAX_LIBRARY_SIZE:
        raise ValueError("native library exceeds the 512 MiB packaging limit")
    _, kind, machine = PLATFORMS[classifier]
    if kind == "elf":
        if len(data) < 64 or data[:7] != b"\x7fELF\x02\x01\x01":
            raise ValueError(f"{classifier} requires a 64-bit little-endian ELF library")
        file_type, actual_machine, version = struct.unpack_from("<HHI", data, 16)
        if file_type != 3 or actual_machine != machine or version != 1:
            raise ValueError(f"ELF shared-library architecture does not match {classifier}")
    else:
        if len(data) < 32 or data[:4] != b"\xcf\xfa\xed\xfe":
            raise ValueError(f"{classifier} requires a thin 64-bit little-endian Mach-O library")
        actual_machine = struct.unpack_from("<I", data, 4)[0]
        file_type = struct.unpack_from("<I", data, 12)[0]
        if actual_machine != machine or file_type != 6:
            raise ValueError(f"Mach-O dynamic-library architecture does not match {classifier}")


def properties(values):
    return "".join(f"{key}={value}\n" for key, value in sorted(values.items())).encode("ascii")


def parse_properties(data):
    # We generate only this ASCII subset of Java Properties, without escapes or continuations.
    if len(data) > MAX_METADATA_SIZE:
        raise ValueError("native metadata exceeds the 16 KiB limit")
    try:
        lines = data.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("native metadata must be ASCII") from error
    result = {}
    for line in lines:
        if not line.strip() or line.startswith(("#", "!")):
            continue
        if "=" not in line or "\\" in line:
            raise ValueError("native metadata uses unsupported Properties syntax")
        key, value = line.split("=", 1)
        if key in result or key != key.strip() or value != value.strip():
            raise ValueError("native metadata contains duplicate or padded properties")
        result[key] = value
    return result


def licenses():
    # These assets cover spark-connect-jni's own normal dependency closure on all
    # four targets. Check their Cargo/source and content hashes before packaging.
    return native_licenses.load_bundle(REPOSITORY)


def write_jar(path, entries):
    """Use stable archive metadata and atomically replace only completed output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for name, data in sorted(entries.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    write_checksum(path)
    return path


def write_checksum(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")


def checked_archive(path):
    archive = zipfile.ZipFile(path)
    names = archive.namelist()
    if len(names) != len(set(names)):
        archive.close()
        raise ValueError(f"duplicate JAR resources in {path}")
    return archive


def check_version(metadata, version):
    if metadata.get("artifact.version") != version:
        raise ValueError("native and adapter artifact versions must match the JNI crate")
    if metadata.get("native.abi.version") != ABI_VERSION:
        raise ValueError("unsupported native ABI version")


def package_library(library, output_dir, classifier, version):
    library = Path(library)
    if not library.is_absolute() or not library.is_file():
        raise ValueError("--library must name an existing absolute library path")
    filename = PLATFORMS[classifier][0]
    if library.name != filename:
        raise ValueError(f"{classifier} library must be named {filename}")
    if library.stat().st_size > MAX_LIBRARY_SIZE:
        raise ValueError("native library exceeds the 512 MiB packaging limit")
    data = library.read_bytes()
    validate_library(data, classifier)
    prefix = f"META-INF/native/{classifier}/"
    entries = licenses()
    entries[prefix + filename] = data
    entries[prefix + "native.properties"] = properties({
        "artifact.version": version,
        "native.abi.version": ABI_VERSION,
        "library.name": filename,
        "library.sha256": hashlib.sha256(data).hexdigest(),
    })
    return write_jar(output_dir / f"spark-connect-native-{version}-{classifier}.jar", entries)


def copy_adapter(adapter, output_dir, version):
    adapter = Path(adapter)
    with checked_archive(adapter) as archive:
        try:
            info = archive.getinfo(ADAPTER_METADATA)
        except KeyError as error:
            raise ValueError("adapter JAR is missing generated version metadata") from error
        if info.file_size > MAX_METADATA_SIZE:
            raise ValueError("adapter metadata exceeds the 16 KiB limit")
        check_version(parse_properties(archive.read(info)), version)
    destination = output_dir / f"spark-connect-native-adapter-{version}.jar"
    if adapter.resolve() != destination.resolve():
        shutil.copyfile(adapter, destination)
    write_checksum(destination)
    return destination


def combine_jars(jars, output_dir, version, *, macos_only=False):
    entries = licenses()
    seen = set()
    required = {name for name in PLATFORMS if not macos_only or name.startswith("macos-")}
    artifact = "macos-universal" if macos_only else "all"
    for path in jars:
        with checked_archive(path) as archive:
            metadata_names = [name for name in archive.namelist()
                              if re.fullmatch(r"META-INF/native/[^/]+/native\.properties", name)]
            if len(metadata_names) != 1:
                raise ValueError("combine inputs must each contain exactly one platform")
            metadata_name = metadata_names[0]
            classifier = metadata_name.split("/")[2]
            if classifier not in PLATFORMS or classifier in seen:
                raise ValueError(f"unsupported or duplicate platform in combine inputs: {classifier}")
            if classifier not in required:
                raise ValueError(f"{artifact} JAR does not accept platform: {classifier}")
            info = archive.getinfo(metadata_name)
            if info.file_size > MAX_METADATA_SIZE:
                raise ValueError("native metadata exceeds the 16 KiB limit")
            metadata_data = archive.read(info)
            metadata = parse_properties(metadata_data)
            check_version(metadata, version)
            filename = PLATFORMS[classifier][0]
            library_name = f"META-INF/native/{classifier}/{filename}"
            if metadata.get("library.name") != filename:
                raise ValueError("native metadata contains an unexpected library name")
            expected_names = set(licenses()) | {metadata_name, library_name}
            if set(archive.namelist()) != expected_names:
                raise ValueError("platform JAR contains unexpected or missing resources")
            for name, data in licenses().items():
                if archive.getinfo(name).file_size != len(data) or archive.read(name) != data:
                    raise ValueError("platform JAR license resources do not match this source")
            info = archive.getinfo(library_name)
            if info.file_size > MAX_LIBRARY_SIZE:
                raise ValueError("native library exceeds the 512 MiB packaging limit")
            data = archive.read(info)
            validate_library(data, classifier)
            if hashlib.sha256(data).hexdigest() != metadata.get("library.sha256"):
                raise ValueError("native library checksum does not match metadata")
            entries[metadata_name] = metadata_data
            entries[library_name] = data
            seen.add(classifier)
    missing = required - seen
    if missing:
        label = ("macOS universal JAR requires both macOS platforms" if macos_only
                 else "all-platform JAR requires every supported platform")
        raise ValueError(label + "; missing: "
                         + ", ".join(sorted(missing)))
    return write_jar(output_dir / f"spark-connect-native-{version}-{artifact}.jar", entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-version", action="store_true", help="print the shared JNI artifact version")
    mode.add_argument("--library", type=Path, help="absolute path to an already-built native library")
    mode.add_argument("--combine", type=Path, nargs="+", help="combine platform JARs into an all-platform JAR")
    mode.add_argument("--combine-macos", type=Path, nargs="+",
                      help="combine both macOS platform JARs into a macOS universal JAR")
    parser.add_argument("--platform", choices=sorted(PLATFORMS), help="default: current OS/architecture")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--adapter-jar", type=Path, help="also copy and version-check the compiled adapter")
    args = parser.parse_args()
    try:
        version = artifact_version()
        if args.print_version:
            if args.platform or args.output_dir or args.adapter_jar:
                parser.error("--print-version cannot be combined with packaging options")
            print(version)
            return
        if args.output_dir is None:
            parser.error("--output-dir is required for packaging")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.combine or args.combine_macos:
            if args.platform:
                parser.error("--platform cannot be used with --combine or --combine-macos")
            result = combine_jars(args.combine or args.combine_macos, args.output_dir, version,
                                  macos_only=args.combine_macos is not None)
        else:
            result = package_library(args.library, args.output_dir, args.platform or host_platform(), version)
        print(result)
        if args.adapter_jar:
            print(copy_adapter(args.adapter_jar, args.output_dir, version))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"native packaging failed: {error}\n")


if __name__ == "__main__":
    main()
