#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print the `ml_params!` block of each given class, from the PySpark param golden.

Usage:
    python3 scripts/gen_rust_params_table.py [--golden PATH] <Class> [<Class> ...]

Reads `tests/golden/ml_api.json` (captured from official pyspark-client 4.3.0) and prints, for
each class, one `ml_params! { Class { ... } }` block, with one row per param in the golden's
param order. The output is checked-in source: adjust it by hand afterwards.

Converter to Rust type of the setter argument (and of the getter value, see below):
    toString         &str
    toInt            i32   (seed and maxLocalProjDBSize: i64)
    toFloat          f64
    toBoolean        bool
    toListString     impl IntoIterator<Item = impl Into<String>>
    toListFloat      impl IntoIterator<Item = f64>
    toListInt        impl IntoIterator<Item = i32>
    toListListFloat  impl IntoIterator<Item = impl IntoIterator<Item = f64>>
    toVector         impl Into<MlVector>

Rows:
    * A param with a `setX` member in the golden's "members" gets a setter row:
      `set_x(value: T) => "x", put_fn; x -> Ret = |this| body;`.
      Otherwise it gets a getter-only row: `x -> Ret = |this| body => "x";`.
    * A param with a golden default gets a getter returning T; a param without one gets
      Option<T>. The default is the golden's, as the server applies it when the client sends
      no value.
    * `outputCol` with the `uid_suffix` default (`<uid>__output`) uses `output_col_or_default`.
    * `seed` with the `class_name_hash` default uses `java_string_hash` of the JVM class name,
      `org.apache.spark.ml.<module>.<Class>`, from the golden's module `pyspark.ml.<module>`
      (`pyspark.ml.pipeline` maps to `org.apache.spark.ml.<Class>`).

Identity rule: a param whose converter is `identity` (its value is an object, such as an
estimator) has no row. The same holds for `toMatrix`, since there is no `MlMatrix` decoder yet.
The generator prints a `// write by hand` note for each of them, and the class is complete only
once those params are written by hand.
"""

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden" / "ml_api.json"

# Converter -> (setter argument type, put function on `Params`).
SETTERS = {
    "toString": ("&str", "set_param_string"),
    "toInt": ("i32", "set_param_int32"),
    "toFloat": ("f64", "set_param_double"),
    "toBoolean": ("bool", "set_param_bool"),
    "toListString": ("impl IntoIterator<Item = impl Into<String>>", "set_param_string_array"),
    "toListFloat": ("impl IntoIterator<Item = f64>", "set_param_double_array"),
    "toListInt": ("impl IntoIterator<Item = i32>", "set_param_int32_array"),
    "toListListFloat": (
        "impl IntoIterator<Item = impl IntoIterator<Item = f64>>",
        "set_param_double_arrays",
    ),
    "toVector": ("impl Into<MlVector>", "set_param_vector"),
}
# `toInt` params whose JVM type is `Long`.
LONG_PARAMS = {"seed", "maxLocalProjDBSize"}
# Converters that get no row (see the identity rule above).
WRITE_BY_HAND = {"identity", "toMatrix"}
RUST_KEYWORDS = {"box", "impl", "loop", "match", "mod", "move", "ref", "type", "use", "where"}
WIDTH = 100
BLOCK_INDENT = " " * 8
CONTINUATION = " " * 4


def snake(name):
    """`maxBlockSizeInMB` -> `max_block_size_in_mb`."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s).lower()
    return s + "_" if s in RUST_KEYWORDS else s


def setter_name(name):
    return "set" + name[0].upper() + name[1:]


def rust_str(text):
    """A Rust string literal of an ASCII text."""
    assert all(ord(c) < 128 for c in text), text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def rust_f64(text):
    """A Rust `f64` expression of the golden's `repr` of a Python float."""
    if text == "nan":
        return "f64::NAN"
    if text == "inf":
        return "f64::INFINITY"
    if text == "-inf":
        return "f64::NEG_INFINITY"
    return text if re.search(r"[.eE]", text) else text + ".0"


def jvm_class(module, cls):
    """`pyspark.ml.<module>.<Class>` -> `org.apache.spark.ml.<module>.<Class>`."""
    if module == "pyspark.ml.pipeline":
        return "org.apache.spark.ml." + cls
    return ".".join(["org", "apache", "spark", "ml"] + module.split(".")[2:] + [cls])


def default_of(param):
    """(kind, value) of the golden default, or (None, None) when there is none.

    The kind is `plain` (a bool or a str), or the key of the tagged default: `float`, `int`,
    `list`, `uid_suffix` or `class_name_hash`.
    """
    if "default" not in param:
        return None, None
    value = param["default"]
    if isinstance(value, dict):
        return next(iter(value.items()))
    return "plain", value


def getter(cls, module, param):
    """(return type, body) of the getter of a param, from its converter and default."""
    name, conv = param["name"], param["converter"]
    key = rust_str(name)
    dkind, dval = default_of(param)
    if conv == "toString":
        if dkind == "uid_suffix":
            assert dval == "__output", (cls, name, dval)
            return "String", "this.params.output_col_or_default(&this.operator.uid)"
        if dkind == "plain":
            return "&str", f"this.params.string_or({key}, {rust_str(dval)})"
        return "Option<&str>", f"this.params.string_opt({key})"
    if conv == "toInt" and name in LONG_PARAMS:
        if dkind == "class_name_hash":
            jvm = rust_str(jvm_class(module, cls))
            return "i64", f"this.params.long_opt({key}).unwrap_or(java_string_hash({jvm}))"
        if dkind == "int":
            return "i64", f"this.params.long_opt({key}).unwrap_or({dval})"
        return "Option<i64>", f"this.params.long_opt({key})"
    if conv == "toInt":
        if dkind == "int":
            assert -(2**31) <= dval < 2**31, (cls, name, dval)
            return "i32", f"this.params.int_or({key}, {dval})"
        return "Option<i32>", f"this.params.int_opt({key})"
    if conv == "toFloat":
        if dkind == "float":
            return "f64", f"this.params.double_or({key}, {rust_f64(dval)})"
        return "Option<f64>", f"this.params.double_opt({key})"
    if conv == "toBoolean":
        if dkind == "plain":
            return "bool", f"this.params.bool_or({key}, {'true' if dval else 'false'})"
        return "Option<bool>", f"this.params.bool_opt({key})"
    if conv in ("toListString", "toListFloat", "toListInt"):
        elem, opt = {
            "toListString": ("String", "string_array_opt"),
            "toListFloat": ("f64", "double_array_opt"),
            "toListInt": ("i32", "int_array_opt"),
        }[conv]
        if dkind == "list" and not dval:
            return f"Vec<{elem}>", f"this.params.{opt}({key}).unwrap_or_default()"
        if dkind == "list":
            if conv == "toListString":
                items = [f"{rust_str(v)}.to_string()" for v in dval]
            elif conv == "toListFloat":
                items = [rust_f64(v["float"]) for v in dval]
            else:
                items = [str(v["int"]) for v in dval]
            return (
                f"Vec<{elem}>",
                f"this.params.{opt}({key}).unwrap_or_else(|| vec![{', '.join(items)}])",
            )
        return f"Option<Vec<{elem}>>", f"this.params.{opt}({key})"
    if conv == "toListListFloat":
        assert dkind is None, (cls, name)
        return "Option<Vec<Vec<f64>>>", f"this.params.double_arrays_opt({key})"
    if conv == "toVector":
        assert dkind is None, (cls, name)
        return "Option<MlVector>", f"this.params.literal({key}).and_then(MlVector::from_literal)"
    raise ValueError(f"no getter for converter {conv} of {cls}.{name}")


def break_point(line, limit):
    """The cut (the length of the head) of the last safe break at or before `limit`, or None.

    Safe breaks are after `(`, after `,`, after `=>`, and after the lone `=` of a getter, and
    never inside a string literal.
    """
    best, in_str, i = None, False, 0
    while i < len(line) - 1:
        c = line[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif i + 1 <= limit:
            cut = i + 1
            if (
                c in "(,"
                or (c == ">" and line[i - 1] == "=")
                or (c == "=" and line[i - 1] == " " and line[i + 1] == " ")
            ):
                best = cut
        i += 1
    return best


def wrap(line):
    """Break a row onto continuation lines, so that none is over `WIDTH` with the block indent.

    The first line has the block's indentation only; each continuation line has four more. The
    breaks are the safe ones of `break_point`, so the macro sees the same tokens as one line.
    """
    lines, cur, first = [], line, True
    while True:
        prefix = "" if first else CONTINUATION
        limit = WIDTH - len(BLOCK_INDENT) - len(prefix)
        if len(cur) <= limit:
            lines.append(prefix + cur)
            return lines
        cut = break_point(cur, limit)
        if cut is None:
            lines.append(prefix + cur)
            return lines
        lines.append(prefix + cur[:cut].rstrip())
        cur, first = cur[cut:].lstrip(), False


def rows_of(cls, entry):
    """The rows of one class, and the notes of its write-by-hand params."""
    members = entry.get("members", {})
    rows, notes = [], []
    for param in entry.get("params", []):
        name, conv = param["name"], param["converter"]
        if conv in WRITE_BY_HAND:
            notes.append(f"// write by hand: {name} ({conv} converter)")
            continue
        ret, body = getter(cls, entry["module"], param)
        getter_part = f"{snake(name)} -> {ret} = |this| {body}"
        if setter_name(name) in members and conv in SETTERS:
            arg_type, put = SETTERS[conv]
            if name in LONG_PARAMS:
                arg_type, put = "i64", "set_param_int_narrow"
            rows.append(f"set_{snake(name)}(value: {arg_type}) => {rust_str(name)}, {put};")
            rows.append(getter_part + ";")
        else:
            rows.append(getter_part + f" => {rust_str(name)};")
    return rows, notes


def block(cls, entry):
    rows, notes = rows_of(cls, entry)
    out = ["ml_params! {", f"    {cls} {{"]
    for line in notes + rows:
        out.extend(BLOCK_INDENT + part for part in wrap(line))
    out.extend(["    }", "}"])
    for text in out:
        assert len(text) <= WIDTH, f"{cls}: a line of {len(text)} columns: {text}"
    return "\n".join(out)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("classes", nargs="+", metavar="Class")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    args = parser.parse_args(argv)
    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    missing = [cls for cls in args.classes if cls not in golden]
    if missing:
        print(f"error: not in {args.golden}: {', '.join(missing)}", file=sys.stderr)
        return 1
    for cls in args.classes:
        print(block(cls, golden[cls]))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
