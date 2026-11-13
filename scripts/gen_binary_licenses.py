#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Generate LICENSE-binary and NOTICE-binary for the wheels.

The wheels of pyspark-client-rust (built by maturin from crates/pyspark-rs) statically link
the normal dependencies of pyspark-rs. ASF policy asks a binary artifact to carry a LICENSE
that lists every bundled third-party component with its license, the texts of the licenses
that are not Apache's, and a NOTICE with the notices that Apache-2.0 components require.

The set is taken from `cargo metadata --locked`, so it follows Cargo.lock:

- the normal (not dev, not build) edges from pyspark-rs, for every target platform;
- not the workspace's own crates (they are Apache-2.0 and ASF's);
- not the proc-macro crates, which run at compile time and are not linked into the wheel.

A component whose license expression offers Apache-2.0 as one option (`MIT OR Apache-2.0`,
`Apache-2.0 OR BSL-1.0`, ...) is used under Apache-2.0 and is listed with the Apache-2.0
components. Every other component is listed under its license expression, and the license
files of its source package are reproduced verbatim in LICENSE-binary, so that its copyright
notices travel with it. The NOTICE files of the components that mention Apache-2.0 are
reproduced in NOTICE-binary, once per distinct text.

Both files ship in the wheels and the sdist (see [tool.maturin] in pyproject.toml).

Usage (from the repository root):

    python3 scripts/gen_binary_licenses.py          # write the two files
    python3 scripts/gen_binary_licenses.py --check  # fail if they are stale

CI runs the second form. It fails when a dependency is added, removed or re-licensed without
regenerating the files.
"""

import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_PACKAGE = "pyspark-rs"
APACHE = "Apache-2.0"
APACHE_TEXT_MARKER = "Version 2.0, January 2004"
SEPARATOR = "-" * 84

# Crates whose source package ships no license file, mapped to the package that carries the
# license of the same release: (name, version) -> (package name, package version). tonic-prost
# is part of the tonic repository, which ships MIT's LICENSE with tonic 0.14.6.
LICENSE_FALLBACK = {("tonic-prost", "0.14.6"): ("tonic", "0.14.6")}

LICENSE_FILE = re.compile(r"^(LICENSE|LICENCE|COPYING|UNLICENSE)", re.IGNORECASE)
NOTICE_FILE = re.compile(r"^NOTICE", re.IGNORECASE)


class Component:
    """A crate that the wheels link, with what the outputs need from it."""

    def __init__(self, name, version, license_expr, source_dir, elected):
        self.name = name
        self.version = version
        self.license = license_expr
        self.source_dir = source_dir
        self.elected = elected  # used under Apache-2.0, as one of the options
        self.mentions_apache = False
        self.license_files = []  # (file name, text) of the non-Apache license texts
        self.notice = None  # text of the NOTICE file, if it mentions Apache-2.0

    @property
    def coordinates(self):
        return f"{self.name} {self.version}"


# ---- SPDX license expressions ---------------------------------------------------------

_TOKEN = re.compile(r"\(|\)|/|[A-Za-z0-9.+\-]+")


def _parse_expr(expr):
    """Parse an SPDX-style expression into nested tuples: ('id', name), ('and', a, b), ...

    `/` is the old spelling of OR, and `X WITH Y` is taken as X, the license it exceptions.
    """
    tokens = _TOKEN.findall(expr)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError(f"unexpected end of license expression: {expr}")
        tok = tokens[pos]
        pos += 1
        return tok

    def primary():
        tok = take()
        if tok == "(":
            node = or_expr()
            if take() != ")":
                raise ValueError(f"unbalanced parentheses in license expression: {expr}")
            return node
        if tok in ("AND", "OR", "WITH", ")", "/"):
            raise ValueError(f"unexpected {tok!r} in license expression: {expr}")
        if peek() == "WITH":
            take()
            take()  # the exception id
        return ("id", tok)

    def and_expr():
        node = primary()
        while peek() == "AND":
            take()
            node = ("and", node, primary())
        return node

    def or_expr():
        node = and_expr()
        while peek() in ("OR", "/"):
            take()
            node = ("or", node, and_expr())
        return node

    node = or_expr()
    if pos != len(tokens):
        raise ValueError(f"trailing tokens in license expression: {expr}")
    return node


def _ids(node):
    if node[0] == "id":
        return {node[1]}
    return _ids(node[1]) | _ids(node[2])


def permits_apache(node):
    """True when Apache-2.0 alone satisfies the expression (it is one of the options)."""
    if node[0] == "id":
        return node[1] == APACHE
    if node[0] == "and":
        return permits_apache(node[1]) and permits_apache(node[2])
    return permits_apache(node[1]) or permits_apache(node[2])


def mentions_apache(node):
    return APACHE in _ids(node)


# ---- Collecting the components --------------------------------------------------------


def _is_proc_macro(package):
    kinds = {kind for target in package["targets"] for kind in target["kind"]}
    return "proc-macro" in kinds and not kinds & {"lib", "rlib", "dylib", "cdylib", "staticlib"}


def linked_packages(meta):
    """The registry packages that pyspark-rs links, by normal edges, for all platforms."""
    packages = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    roots = [p["id"] for p in meta["packages"] if p["name"] == ROOT_PACKAGE and p["source"] is None]
    if len(roots) != 1:
        raise SystemExit(f"expected one {ROOT_PACKAGE} package in cargo metadata")
    seen = set()
    stack = [roots[0]]
    while stack:
        pkg_id = stack.pop()
        if pkg_id in seen:
            continue
        seen.add(pkg_id)
        if _is_proc_macro(packages[pkg_id]):
            continue  # compile-time only: its dependencies are not linked either
        for dep in nodes[pkg_id]["deps"]:
            if any(kind["kind"] in (None, "normal") for kind in dep["dep_kinds"]):
                stack.append(dep["pkg"])
    return [
        packages[i]
        for i in seen
        if packages[i]["source"] is not None and not _is_proc_macro(packages[i])
    ]


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _is_apache_text(text):
    return APACHE_TEXT_MARKER in text[:400]


def _find_package(meta, key):
    for p in meta["packages"]:
        if (p["name"], p["version"]) == key:
            return p
    raise SystemExit(f"fallback package {key} is not in cargo metadata")


def collect_components(meta):
    """Build a Component for each linked registry package, sorted by name and version."""
    components = []
    for package in sorted(linked_packages(meta), key=lambda p: (p["name"], p["version"])):
        coords = f"{package['name']} {package['version']}"
        expr = package["license"]
        if not expr:
            raise SystemExit(f"{coords} declares no license")
        node = _parse_expr(expr)
        source_dir = os.path.dirname(package["manifest_path"])
        component = Component(
            package["name"], package["version"], expr, source_dir, permits_apache(node)
        )
        component.mentions_apache = mentions_apache(node)
        if not component.elected:
            lic_dir = source_dir
            key = (package["name"], package["version"])
            if key in LICENSE_FALLBACK:
                other = _find_package(meta, LICENSE_FALLBACK[key])
                lic_dir = os.path.dirname(other["manifest_path"])
            texts = [
                (f, _read(os.path.join(lic_dir, f)))
                for f in sorted(os.listdir(lic_dir))
                if LICENSE_FILE.match(f)
            ]
            texts = [(f, t) for f, t in texts if not _is_apache_text(t)]
            if not texts:
                raise SystemExit(f"no license text found for {coords} ({expr})")
            component.license_files = texts
        notices = [f for f in sorted(os.listdir(source_dir)) if NOTICE_FILE.match(f)]
        if notices and component.mentions_apache:
            component.notice = "\n".join(
                _read(os.path.join(source_dir, f)).rstrip() for f in notices
            )
        components.append(component)
    return components


# ---- Rendering -------------------------------------------------------------------------


def _apache_text(license_text):
    """The Apache License 2.0 text at the top of the repository's LICENSE."""
    lines = license_text.split("\n")
    idx = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("This product bundles various third-party components")
    )
    if not lines[idx - 1].startswith("----"):
        raise SystemExit("LICENSE: the third-party section does not follow the Apache text")
    return "\n".join(lines[: idx - 1]).rstrip("\n") + "\n"


def render_license(components, apache_text):
    elected = [c for c in components if c.elected]
    others = [c for c in components if not c.elected]
    out = [apache_text, "\n", SEPARATOR, "\n"]
    out.append(
        "This project bundles some components that are also licensed under the Apache\n"
        "License Version 2.0. Each is listed with its license expression. The components\n"
        "whose expression offers Apache-2.0 as an option are used under Apache-2.0.\n\n"
    )
    for c in elected:
        out.append(f"{c.coordinates} ({c.license})\n")
    out.append("\n" + SEPARATOR + "\n")
    out.append(
        "This product bundles various third-party components under other open source\n"
        "licenses. This section summarizes those components and their licenses. Their\n"
        "license texts follow the summary.\n\n"
    )
    headings = {}
    for c in others:
        headings.setdefault(c.license, []).append(c)
    for expr in sorted(headings):
        out.append(f"{expr}\n{'-' * len(expr)}\n")
        for c in headings[expr]:
            out.append(f"{c.coordinates}\n")
        out.append("\n")
    out.append(SEPARATOR + "\n")
    out.append(
        "The Python and data components of the wheels are not Rust dependencies. They are\n"
        "listed in LICENSE: python/pyspark/loose_version.py (Python Software Foundation\n"
        "License), python/pyspark/cloudpickle/*.py, python/pyspark/_globals.py (BSD\n"
        "3-Clause), python/pyspark/errors/exceptions/tblib.py (BSD 2-Clause), and\n"
        "crates/spark-connect/src/srs_registry.csv (MIT, PROJ).\n"
    )
    for c in others:
        for name, text in c.license_files:
            out.append(
                f"\n{SEPARATOR}\nThe license of {c.coordinates} ({c.license}), "
                f"from {name}:\n\n{text.rstrip()}\n"
            )
    return "".join(out)


def render_notice(root_notice, components):
    by_text = {}
    for c in components:
        if c.notice is not None:
            by_text.setdefault(c.notice, []).append(c)
    out = [root_notice.rstrip("\n") + "\n", "\n", SEPARATOR, "\n"]
    out.append(
        "This product bundles the following components licensed under the Apache License\n"
        "2.0, whose NOTICE files require the notices below. Each notice is reproduced\n"
        "as its component ships it.\n"
    )
    for text in sorted(by_text, key=lambda t: by_text[t][0].name):
        names = ", ".join(c.coordinates for c in by_text[text])
        out.append(f"\n=== NOTICE FOR {names} ===\n{text}\n=== END OF NOTICE FOR {names} ===\n")
    return "".join(out)


def build_outputs(meta, license_text, notice_text):
    """Return {relative path: text} of the two generated files."""
    components = collect_components(meta)
    return {
        "LICENSE-binary": render_license(components, _apache_text(license_text)),
        "NOTICE-binary": render_notice(notice_text, components),
    }


def cargo_metadata(root):
    result = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode(errors="replace"))
        raise SystemExit("cargo metadata failed")
    return json.loads(result.stdout)


def generate(root):
    return build_outputs(
        cargo_metadata(root),
        _read(os.path.join(root, "LICENSE")),
        _read(os.path.join(root, "NOTICE")),
    )


def write(root, outputs):
    for rel, text in outputs.items():
        with open(os.path.join(root, rel), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


def check(root, outputs):
    """Return the generated files that are missing or differ from the generator's output."""
    stale = []
    for rel, text in outputs.items():
        path = os.path.join(root, rel)
        if not os.path.isfile(path) or _read(path) != text:
            stale.append(rel)
    return stale


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if the outputs are not what the generator writes"
    )
    parser.add_argument("--root", default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    outputs = generate(args.root)
    if args.check:
        stale = check(args.root, outputs)
        if stale:
            for rel in stale:
                print(f"stale: {rel}", file=sys.stderr)
            print(
                "Run `python3 scripts/gen_binary_licenses.py` and commit the result.",
                file=sys.stderr,
            )
            return 1
        print("LICENSE-binary and NOTICE-binary are up to date.")
        return 0
    write(args.root, outputs)
    print("Wrote LICENSE-binary and NOTICE-binary.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
