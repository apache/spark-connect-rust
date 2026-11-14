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

The set is the union, over the targets of the wheel jobs, of `cargo tree` for pyspark-rs with
the features that maturin builds it with. cargo tree resolves the features of pyspark-rs alone,
so the crates that only the examples enable (wasm-udf) are not listed. It follows Cargo.lock
(--locked), and it takes the normal (not dev, not build) edges, which excludes the proc-macro
crates, whose dependencies are not linked either. The workspace's own crates are not listed:
they are Apache-2.0 and ASF's. `cargo metadata` gives the license and the source directory of
each crate.

Some crates compile source files with their own notices (for example ring's ISC-licensed C and
Rust files, or the SunPro notices of libm). Those notices are reproduced from the files, as
listed in SUPPLEMENTARY_NOTICES.

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
import glob
import json
import os
import re
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_PACKAGE = "pyspark-rs"
APACHE = "Apache-2.0"
APACHE_TEXT_MARKER = "Version 2.0, January 2004"
SEPARATOR = "-" * 84

# Crates whose source package ships no license file, mapped to the package that carries the
# license of the same release: (name, version) -> (package name, package version). tonic-prost
# is part of the tonic repository, which ships MIT's LICENSE with tonic 0.14.6.
LICENSE_FALLBACK = {("tonic-prost", "0.14.6"): ("tonic", "0.14.6")}

# The targets of the wheel jobs in .github/workflows/release.yml. universal2-apple-darwin is
# built from its two architectures. The features are the ones that [tool.maturin] in
# pyproject.toml builds pyspark-rs with.
WHEEL_TARGETS = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
)
WHEEL_FEATURES = ("pyo3/abi3-py39", "pyo3/extension-module")

# Notices that the source files of a linked crate carry, which its LICENSE files do not
# reproduce: (crate, version) -> [(path, marker)]. The path is relative to the crate's source
# directory and may be a glob. The notice of a file is the comment block that holds a line
# matching the marker regex, or the whole file when the marker is None (a license file). The
# text is copied from the file at generation time, so --check fails when the crate changes.
# A glob skips the files without a matching notice; a plain path must have one. A crate that
# is not linked by the wheels is an error, so the table cannot go stale silently.
#
# Each entry is listed only if the wheels compile the file for one of their targets, checked
# from build.rs and the cfg attributes of the crate; the other files are not listed. A notice
# under a holder of the crate's own (its LICENSE already has it) is not listed either.
SUPPLEMENTARY_NOTICES = {
    # ring: the Google and Intel ISC files of BoringSSL, and the ISC files of ring's authors
    # and contributors. crypto/poly1305/poly1305_arm.c and src/aead/poly1305/ffi_arm_neon.rs
    # are for 32-bit ARM only, and src/aead/chacha/fallback.rs is for 32-bit x86 and tests only.
    ("ring", "0.17.14"): [
        ("crypto/poly1305/poly1305.c", "Google Inc"),
        ("crypto/fipsmodule/aes/aes_nohw.c", "Google Inc"),
        ("crypto/fipsmodule/ec/ecp_nistz.h", "Google Inc"),
        ("crypto/fipsmodule/ec/ecp_nistz.c", "Intel Corporation"),
        ("crypto/fipsmodule/ec/ecp_nistz384.h", "Intel Corporation"),
        ("crypto/fipsmodule/ec/ecp_nistz384.inl", "Intel Corporation"),
        ("src/aead/chacha.rs", "Google Inc"),
        ("src/aead/gcm/fallback.rs", "Google Inc"),
        ("src/aead/poly1305/ffi_fallback.rs", "Google Inc"),
        ("src/debug.rs", "Trent Clarke"),
        ("src/digest/sha1.rs", "Simon Sapin"),
        ("src/limb.rs", "David Judd"),
        ("src/polyfill/once_cell/LICENSE-MIT", None),
    ],
    # zstd-sys: divsufsort.c (Yuta Mori, MIT) is compiled by the zdict_builder feature, and the
    # legacy decoders (zstd/lib/legacy, all of its .c files) by the legacy feature, both defaults.
    # threading.c (Tino Reichardt) is compiled, but its code needs ZSTD_MULTITHREAD (the zstdmt
    # feature), which the wheels do not enable.
    ("zstd-sys", "2.1.3+zstd.1.5.7"): [
        ("zstd/lib/dictBuilder/divsufsort.c", "Yuta Mori"),
        ("zstd/lib/common/threading.c", "Tino Reichardt"),
        ("zstd/lib/legacy/*.c", "Yann Collet"),
    ],
    # rustls-webpki: ISC files of the two authors. The crate's LICENSE is Brian Smith's.
    ("rustls-webpki", "0.103.13"): [
        ("src/subject_name/mod.rs", "Rafael Fern"),
        ("src/crl/mod.rs", "Daniel McCarney"),
    ],
    # libm: the SunPro files (fdlibm), a FreeBSD BSD-2-Clause file, and a file of Alexei
    # Sibidanov's (MIT). The crate's LICENSE.txt lists the holders but not the notices.
    ("libm", "0.2.16"): [
        ("src/**/*.rs", "Sun Microsystems"),
        ("src/**/*.rs", "David Schultz"),
        ("src/**/*.rs", "Alexei Sibidanov"),
    ],
    # lexical-parse-float and lexical-util: fdlibm code under the SunPro notice.
    ("lexical-parse-float", "1.0.6"): [("src/libm.rs", "Sun Microsystems")],
    ("lexical-util", "1.0.7"): [("src/libm.rs", "Sun Microsystems")],
    # futures-channel: the BSD-2-Clause notice of Dmitry Vyukov's queue.
    ("futures-channel", "0.3.32"): [("src/mpsc/queue.rs", "Dmitry Vyukov")],
    # pyo3: the MIT notice of Daniel Grunwald's buffer protocol code (rust-cpython).
    ("pyo3", "0.29.3"): [("src/buffer.rs", "Daniel Grunwald")],
    # sha2-asm: the MIT notices of the assembly (Project Nayuki, Emmanuel Gil Peyrot). Only the
    # aarch64 targets link sha2-asm, so only sha256_aarch64.S is compiled for the wheels. The
    # x86 files are not, and sha256_aarch64_apple.S is in the public domain (no notice).
    ("sha2-asm", "0.6.4"): [
        ("src/sha256_aarch64.S", "Copyright"),
    ],
}

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
        self.source_notices = []  # (files, text) of SUPPLEMENTARY_NOTICES, copied from the files

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


def wheel_packages(meta, tree_outputs):
    """The registry packages that the wheels link, from the `cargo tree` output of each target.

    Each output lists the normal, non-proc-macro dependencies of pyspark-rs for one target. A
    crate is linked when any target lists it. Proc-macro crates are dropped even when a normal
    edge reaches them.
    """
    registry = {(p["name"], p["version"]): p for p in meta["packages"] if p["source"] is not None}
    linked = {}
    for output in tree_outputs:
        for raw in output.splitlines():
            line = raw.strip()
            if line.endswith(" (*)"):
                line = line[: -len(" (*)")]
            if not line or " (" in line:
                continue  # blank, or a workspace path package such as "pyspark-rs v4.3.0 (/...)"
            parts = line.split()
            if len(parts) != 2 or not parts[1].startswith("v"):
                raise SystemExit(f"unexpected cargo tree line: {raw!r}")
            key = (parts[0], parts[1][1:])
            if key not in registry:
                raise SystemExit(f"{key[0]} {key[1]} is not a registry package in cargo metadata")
            linked[key] = registry[key]
    return [p for key, p in linked.items() if not _is_proc_macro(p)]


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _is_apache_text(text):
    return APACHE_TEXT_MARKER in text[:400]


def _comment_blocks(lines):
    """Yield (first, last) line indices of the comment blocks: each /* */ block, and each run of
    // or perl-style # comment lines."""
    i = 0
    while i < len(lines):
        s = lines[i].lstrip()
        if s.startswith("/*"):
            last, rest = i, lines[i].split("/*", 1)[1]
            while "*/" not in rest and last + 1 < len(lines):
                last += 1
                rest = lines[last]
            yield i, last
            i = last + 1
        elif _is_line_comment(s):
            last = i
            while last + 1 < len(lines) and _is_line_comment(lines[last + 1].lstrip()):
                last += 1
            yield i, last
            i = last + 1
        else:
            i += 1


def _is_line_comment(stripped):
    return stripped.startswith("//") or stripped == "#" or stripped.startswith("# ")


def _notice_block(text, marker):
    """The first comment block of the file that has a line matching `marker`, or None."""
    lines = text.split("\n")
    pattern = re.compile(marker)
    for first, last in _comment_blocks(lines):
        if any(pattern.search(line) for line in lines[first : last + 1]):
            return "\n".join(lines[first : last + 1]).rstrip() + "\n"
    return None


def _notice_key(notice):
    """The words of a notice without its comment markers and its `origin:` lines, so that one
    notice in two comment styles, or in files of different origin, is one notice."""
    words = []
    for line in notice.split("\n"):
        if "origin:" in line:
            continue
        line = re.sub(r"\*/\s*$", "", line)
        line = re.sub(r"^\s*(/\*+|\*+|//+!?|#)", "", line)
        words.extend(line.split())
    return " ".join(words)


def _source_notices(source_dir, coords, entries):
    """Copy the notices of a crate's files for SUPPLEMENTARY_NOTICES.

    Returns [(files, text)], one item per distinct notice, with the relative paths of its files
    and the text of the first of them.
    """
    groups = {}  # _notice_key -> [text, [files]]
    for path, marker in entries:
        matches = sorted(glob.glob(os.path.join(source_dir, path), recursive=True))
        if not matches:
            raise SystemExit(f"{coords}: no file matches {path}")
        for full in matches:
            rel = os.path.relpath(full, source_dir).replace(os.sep, "/")
            text = _read(full)
            if marker is None:
                notice = text.rstrip() + "\n"
            else:
                notice = _notice_block(text, marker)
            if notice is None:
                if any(c in path for c in "*?["):
                    continue
                raise SystemExit(f"{coords}: {rel} has no notice that matches {marker!r}")
            groups.setdefault(_notice_key(notice), [notice, []])[1].append(rel)
    return [(files, text) for text, files in sorted(groups.values(), key=lambda g: g[1][0])]


def _find_package(meta, key):
    for p in meta["packages"]:
        if (p["name"], p["version"]) == key:
            return p
    raise SystemExit(f"fallback package {key} is not in cargo metadata")


def collect_components(meta, tree_outputs, *, supplementary_notices=None):
    """Build a Component for each linked registry package, sorted by name and version."""
    if supplementary_notices is None:
        supplementary_notices = SUPPLEMENTARY_NOTICES
    packages = wheel_packages(meta, tree_outputs)
    keys = {(p["name"], p["version"]) for p in packages}
    unlinked = sorted(f"{n} {v}" for n, v in supplementary_notices if (n, v) not in keys)
    if unlinked:
        raise SystemExit(
            "SUPPLEMENTARY_NOTICES names crates that the wheels do not link: " + ", ".join(unlinked)
        )
    components = []
    for package in sorted(packages, key=lambda p: (p["name"], p["version"])):
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
        key = (package["name"], package["version"])
        if key in supplementary_notices:
            component.source_notices = _source_notices(
                source_dir, coords, supplementary_notices[key]
            )
        if not component.elected:
            lic_dir = source_dir
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


def render_license(components, apache_text, *, include_wheel_sources=True):
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
    if include_wheel_sources:
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
    for c in components:
        for files, text in c.source_notices:
            names = textwrap.fill(", ".join(files), width=76, subsequent_indent="  ")
            out.append(
                f"\n{SEPARATOR}\nThe notice in the source of {c.coordinates} ({c.license}), "
                f"in {names}:\n\n{text}"
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


def build_outputs(meta, license_text, notice_text, tree_outputs):
    """Return {relative path: text} of the two generated files."""
    components = collect_components(meta, tree_outputs)
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


def cargo_tree(root, target):
    """The normal, non-proc-macro dependencies of pyspark-rs for one target, as `cargo tree`
    prints them."""
    args = ["cargo", "tree", "--locked", "-e", "normal,no-proc-macro", "-p", ROOT_PACKAGE]
    args += ["--features", ",".join(WHEEL_FEATURES), "--target", target]
    args += ["--prefix", "none", "--format", "{p}"]
    result = subprocess.run(
        args,
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode(errors="replace"))
        raise SystemExit(f"cargo tree failed for {target}")
    return result.stdout.decode(errors="replace")


def generate(root):
    return build_outputs(
        cargo_metadata(root),
        _read(os.path.join(root, "LICENSE")),
        _read(os.path.join(root, "NOTICE")),
        [cargo_tree(root, target) for target in WHEEL_TARGETS],
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
