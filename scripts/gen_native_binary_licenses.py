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

"""Generate the JNI library's binary licenses from its own four-target dependency closure.

Normal, non-proc-macro Cargo edges identify statically linked dependencies; build and dev
dependencies are excluded. Each supported target is resolved separately, with the native
crate's default features. The bundled texts cover their union, so platform, all-platform
and macOS universal JARs carry the same complete license resources.

The wheel generator supplies SPDX handling, license texts and known source-file notices;
its output and its Python/data notices are independent of this JNI bundle. Regenerate
with `python3 scripts/gen_native_binary_licenses.py`; CI runs it with --check. Packaging
also checks the committed assets' hashes and their Cargo inputs without invoking Cargo.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("crates/spark-connect-server/jvm/native-licenses")
ROOT_PACKAGE = "spark-connect-jni"
TARGETS = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
)
CARGO_INPUTS = (
    "Cargo.toml",
    "Cargo.lock",
    "crates/spark-connect-proto/Cargo.toml",
    "crates/spark-connect-server/Cargo.toml",
    "crates/spark-connect-server/jni/Cargo.toml",
)
SOURCE_FILES = CARGO_INPUTS + (
    "LICENSE",
    "NOTICE",
    "scripts/gen_binary_licenses.py",
    "scripts/gen_native_binary_licenses.py",
)
SPEC = importlib.util.spec_from_file_location(
    "native_license_common", ROOT / "scripts/gen_binary_licenses.py"
)
binary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binary)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def input_hashes(root, names):
    hashes = {}
    for name in sorted(names):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"invalid native license input path: {name}")
        full = root / path
        if not full.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"native license input escapes the source tree: {name}")
        hashes[name] = digest(full.read_bytes())
    return hashes


def local_packages(meta, output):
    packages = {(p["name"], p["version"]): p for p in meta["packages"] if p["source"] is None}
    linked = {}
    for raw in output.splitlines():
        parts = raw.removesuffix(" (*)").split(" ", 2)
        if len(parts) == 3 and parts[2].startswith("("):
            key = (parts[0], parts[1].removeprefix("v"))
            if key not in packages:
                raise ValueError(f"unknown local package in native cargo tree: {raw}")
            linked[key] = packages[key]
    if not any(name == ROOT_PACKAGE for name, _ in linked):
        raise ValueError(f"native cargo tree is missing {ROOT_PACKAGE}")
    return sorted(linked.values(), key=lambda p: (p["name"], p["version"]))


def scoped_notices(packages):
    """Reuse source notices only for linked packages, and require review on version changes."""
    keys = {(p["name"], p["version"]) for p in packages}
    known = binary.SUPPLEMENTARY_NOTICES
    known_names = {name for name, _ in known}
    for key in keys:
        if key[0] in known_names and key not in known:
            raise ValueError(f"review supplementary native license notices for updated {key[0]}")
    return {key: notices for key, notices in known.items() if key in keys}


def build_outputs(root, meta, trees):
    if set(trees) != set(TARGETS):
        raise ValueError("native license generation requires exactly the four supported targets")
    packages = binary.wheel_packages(meta, trees.values())
    components = binary.collect_components(
        meta, list(trees.values()), supplementary_notices=scoped_notices(packages)
    )
    texts = {
        "LICENSE-binary": binary.render_license(
            components,
            binary._apache_text((root / "LICENSE").read_text(encoding="utf-8")),
            include_wheel_sources=False,
        ),
        "NOTICE-binary": binary.render_notice(
            (root / "NOTICE").read_text(encoding="utf-8"), components
        ),
    }
    inputs = set(SOURCE_FILES)
    inventory = {}
    for target in TARGETS:
        local = local_packages(meta, trees[target])
        for package in local:
            inputs.add(
                Path(package["manifest_path"]).resolve().relative_to(root.resolve()).as_posix()
            )
        inventory[target] = {
            "workspace": [[p["name"], p["version"]] for p in local],
            "registry": sorted(
                [p["name"], p["version"], p["license"]]
                for p in binary.wheel_packages(meta, [trees[target]])
            ),
        }
    manifest = {
        "schema_version": 1,
        "root_package": ROOT_PACKAGE,
        "targets": inventory,
        "inputs": input_hashes(root, inputs),
        "outputs": {name: digest(text.encode("utf-8")) for name, text in texts.items()},
    }
    texts["dependencies.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return {(OUTPUT / name).as_posix(): text for name, text in texts.items()}


def load_bundle(root):
    """Check committed assets without Cargo, including source/lock and output fingerprints."""
    inventory_path = root / OUTPUT / "dependencies.json"
    inventory_bytes = inventory_path.read_bytes()
    manifest = json.loads(inventory_bytes)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("root_package") != ROOT_PACKAGE
        or set(manifest.get("targets", {})) != set(TARGETS)
        or set(manifest.get("outputs", {})) != {"LICENSE-binary", "NOTICE-binary"}
        or not set(SOURCE_FILES) <= set(manifest.get("inputs", {}))
    ):
        raise ValueError("invalid native dependency license inventory; regenerate its assets")
    if input_hashes(root, manifest["inputs"]) != manifest["inputs"]:
        raise ValueError(
            "stale native dependency licenses; run scripts/gen_native_binary_licenses.py"
        )
    entries = {"META-INF/native-dependencies.json": inventory_bytes}
    for name, expected in manifest["outputs"].items():
        text = (root / OUTPUT / name).read_bytes()
        if digest(text) != expected:
            raise ValueError(
                f"native dependency license asset does not match its inventory: {name}"
            )
        entries["META-INF/" + name.removesuffix("-binary")] = text
    return entries


def cargo_tree(root, target):
    args = [
        "cargo",
        "tree",
        "--locked",
        "-e",
        "normal,no-proc-macro",
        "-p",
        ROOT_PACKAGE,
        "--target",
        target,
        "--prefix",
        "none",
        "--format",
        "{p}",
    ]
    result = subprocess.run(args, cwd=root, capture_output=True, check=False)
    if result.returncode:
        sys.stderr.write(result.stderr.decode(errors="replace"))
        raise SystemExit(f"native cargo tree failed for {target}")
    return result.stdout.decode("utf-8")


def generate(root, inputs=None):
    before = input_hashes(root, CARGO_INPUTS)
    if inputs is None:
        meta = binary.cargo_metadata(str(root))
        trees = {target: cargo_tree(root, target) for target in TARGETS}
    else:
        recorded = json.loads((inputs / "cargo-inputs.json").read_text(encoding="utf-8"))
        if recorded != before:
            raise ValueError(
                "saved native Cargo inputs are stale; collect metadata and trees again"
            )
        meta = json.loads((inputs / "metadata.json").read_text(encoding="utf-8"))
        trees = {
            target: (inputs / (target + ".txt")).read_text(encoding="utf-8") for target in TARGETS
        }
    outputs = build_outputs(root, meta, trees)
    if input_hashes(root, CARGO_INPUTS) != before:
        raise ValueError("native Cargo inputs changed during generation; rerun the generator")
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail on missing or stale assets")
    parser.add_argument("--inputs-dir", type=Path, help="read saved Cargo metadata/tree outputs")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    outputs = generate(args.root, args.inputs_dir)
    if args.check:
        stale = binary.check(str(args.root), outputs)
        if stale:
            for name in stale:
                print("stale: " + name, file=sys.stderr)
            print(
                "Run `python3 scripts/gen_native_binary_licenses.py` and commit the result.",
                file=sys.stderr,
            )
            return 1
        print("JNI binary licenses and dependency inventory are up to date.")
    else:
        (args.root / OUTPUT).mkdir(parents=True, exist_ok=True)
        binary.write(str(args.root), outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
