#!/usr/bin/env python3
"""
Generator for dispatch and Python wrappers for all 440 Spark SQL functions.
Parses crates/spark-connect/src/functions.rs and generates optimized code.
"""

import argparse
import ast
import dis
import inspect
import re
import sys
import textwrap
import types
from pathlib import Path

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument(
    "--pyspark-src",
    default=None,
    help="Path to the reference PySpark package root (…/python), with the dependencies of "
    "PySpark Connect importable. When given, a function whose PySpark Connect "
    "implementation only marshals its arguments into a function call is generated with "
    "that implementation's signature and body; the others carry the reference's parameter "
    "names and dispatch to the Rust core. Without it, wrappers fall back to the historical "
    "*args form.",
)
_ap.add_argument(
    "--write-dispatch",
    action="store_true",
    help="Also (re)write crates/pyspark-rs/src/dispatch_generated.rs. OFF by default: "
    "that file carries hand-tuned multi-arg arms (e.g. round_scale, "
    "approx_count_distinct_rsd) this generator does not reproduce, so overwriting it "
    "would regress. Only pass this when intentionally regenerating the Rust dispatch.",
)
_args = _ap.parse_args()

# Optionally load the reference pyspark to mirror real function signatures, and its
# PySpark Connect implementations.
_ref_functions = None
_ref_builtin = None
if _args.pyspark_src:
    sys.path.insert(0, _args.pyspark_src)
    import pyspark.sql.functions as _ref_functions  # noqa: E402
    import pyspark.sql.connect.functions.builtin as _ref_builtin  # noqa: E402

# Read the spark-connect functions file
functions_rs = Path("crates/spark-connect/src/functions.rs").read_text()

# Extract all function definitions with their full signatures
# Pattern: pub fn name(params) -> Column {
pattern = r"^pub fn ([a-zA-Z0-9_#]+)\((.*?)\)\s*->\s*Column\s*\{"
matches = re.finditer(pattern, functions_rs, re.MULTILINE | re.DOTALL)

functions: dict[str, dict] = {}
for match in matches:
    full_name = match.group(1)
    name = full_name.replace("r#", "")  # Remove raw string prefix like r#struct
    params_str = match.group(2).strip()

    # Parse parameters
    params = []
    if params_str:
        # Split by comma, but handle nested angle brackets
        param_parts = []
        current = ""
        depth = 0
        for char in params_str:
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                param_parts.append(current.strip())
                current = ""
                continue
            current += char
        if current.strip():
            param_parts.append(current.strip())

        for param in param_parts:
            # Parse each parameter: name: type
            if ":" in param:
                parts = param.split(":")
                pname = parts[0].strip()
                ptype = ":".join(parts[1:]).strip()
                params.append({"name": pname, "type": ptype})

    functions[name] = {
        "full_name": full_name,
        "params": params,
        "param_count": len(params),
    }

print(f"Found {len(functions)} functions")

# Functions excluded from the generic dispatcher AND the Python wrappers because
# their Rust signatures don't fit the generic Vec<Column> -> Column form.
skip_functions = {
    "col",
    "lit",
    "expr",
    "sum",
    "count",
    "avg",
    "max",
    "min",
    "call_function",
    "call_udf",
    "cast",
    "column",
    # Non-standard signatures:
    #   broadcast(DataFrame) -> DataFrame  (a join hint, not a column function)
    #   variant_delete(Column, Vec<Column>) -> Column  (variadic paths)
    "broadcast",
    "variant_delete",
    # variant_strip_nulls(Column, bool) -> Column: bool arg, dedicated pyfunc binding
    # (the generic Vec<Column> dispatch would drop the trailing scalar).
    "variant_strip_nulls",
    # Mixed functions with dedicated pyfunc_* bindings:
    #   sha2(Column, i32) -> Column
    #   window(Column, &str) -> Column
    #   from_avro(Column, &str) -> Column
    #   from_avro_with_options(Column, &str, Column) -> Column
    #   to_avro_with_schema(Column, &str) -> Column
    #   from_protobuf(Column, &str) -> Column (and 3 variants)
    #   to_protobuf(Column, &str) -> Column (and 3 variants)
    "sha2",
    "window",
    # window(Column, windowDuration, slideDuration, startTime): string args, so it is
    # hand-wired via pyfunc_window_with_slide_and_start (functions.py), not the generic
    # Vec<Column> dispatch.
    "window_with_slide_and_start",
    "from_avro",
    "from_avro_with_options",
    "to_avro_with_schema",
    "from_protobuf",
    "from_protobuf_with_descriptor",
    "from_protobuf_with_descriptor_and_options",
    "from_protobuf_with_options",
    "to_protobuf",
    "to_protobuf_with_descriptor",
    "to_protobuf_with_descriptor_and_options",
    "to_protobuf_with_options",
    # listagg_distinct(col, delimiter) and string_agg_distinct: a distinct function call,
    # which the Python wrappers of functions.py dispatch with or without the delimiter.
    "listagg_distinct",
    "string_agg_distinct",
    # count_distinct(col, *cols) and countDistinct: a distinct count of all the columns.
    "count_distinct",
    "countDistinct",
}

# Categorize functions by their parameter types
no_args = []
single_col = []
variadic_cols = []
multiple_cols = []
str_only = []
mixed = []

for name, fn in functions.items():
    if name in skip_functions:
        continue
    if fn["param_count"] == 0:
        no_args.append(name)
    elif fn["param_count"] == 1:
        param_type = fn["params"][0]["type"]
        if "Vec<Column>" in param_type:
            variadic_cols.append(name)
        elif "Column" in param_type:
            single_col.append(name)
        elif "&str" in param_type or "str" in param_type:
            str_only.append(name)
        else:
            mixed.append(name)
    else:
        # Check if all are Columns
        all_cols = all("Column" in p["type"] for p in fn["params"])
        if all_cols:
            multiple_cols.append(name)
        else:
            mixed.append(name)

print(f"  No args: {len(no_args)}")
print(f"  Single Column: {len(single_col)}")
print(f"  Multiple Columns: {len(multiple_cols)}")
print(f"  String only: {len(str_only)}")
print(f"  Mixed/Other: {len(mixed)}")

# Generate Rust dispatch function
rust_arms = []

# No-arg functions
for name in no_args:
    fn = functions[name]
    rust_arms.append(f'            "{name}" => Ok(spark_funcs::{fn["full_name"]}()),')

# Single column functions
for name in single_col:
    fn = functions[name]
    rust_arms.append(f'            "{name}" => {{')
    rust_arms.append(
        '                if args.is_empty() { return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Missing required column argument")); }'
    )
    rust_arms.append(f"                Ok(spark_funcs::{fn['full_name']}(args[0].clone()))")
    rust_arms.append("            },")

# Variadic (Vec<Column>) functions - forward all column args
for name in variadic_cols:
    fn = functions[name]
    rust_arms.append(f'            "{name}" => Ok(spark_funcs::{fn["full_name"]}(args.clone())),')

# String-only functions - convert first arg to string literal
for name in str_only:
    fn = functions[name]
    param_name = fn["params"][0]["name"]
    rust_arms.append(f'            "{name}" => {{')
    rust_arms.append("                // String parameter - use empty string as default")
    rust_arms.append(f'                Ok(spark_funcs::{fn["full_name"]}(""))')
    rust_arms.append("            },")

# Multiple column functions
for name in multiple_cols:
    fn = functions[name]
    col_count = sum(1 for p in fn["params"] if "Column" in p["type"])
    rust_arms.append(f'            "{name}" => {{')
    rust_arms.append(
        f'                if args.len() < {col_count} {{ return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Missing required arguments for {name}: expected at least {col_count}, got {{}}", args.len()))); }}'
    )
    col_args = ", ".join([f"args[{i}].clone()" for i in range(col_count)])
    rust_arms.append(f"                Ok(spark_funcs::{fn['full_name']}({col_args}))")
    rust_arms.append("            },")

# Mixed/other - try to handle generically
for name in mixed:
    fn = functions[name]
    # Special case for window(Column, &str)
    if name == "window":
        rust_arms.append('            "window" => {')
        rust_arms.append(
            '                if args.len() < 1 { return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("window() requires at least 1 column argument")); }'
        )
        rust_arms.append("                // window takes a column and a string duration parameter")
        rust_arms.append('                Ok(spark_funcs::window(args[0].clone(), ""))')
        rust_arms.append("            },")
    else:
        # For other mixed functions, try generic handling
        col_count = fn["param_count"]
        rust_arms.append(f'            "{name}" => {{')
        rust_arms.append("                // Mixed/special case: trying generic handling")
        col_args = ", ".join([f"args[{i}].clone()" for i in range(min(col_count, 1))])
        if col_count <= 1:
            if col_count == 0:
                rust_arms.append(f"                Ok(spark_funcs::{fn['full_name']}())")
            else:
                rust_arms.append(
                    '                if args.is_empty() { return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Missing arguments")); }'
                )
                rust_arms.append(
                    f"                Ok(spark_funcs::{fn['full_name']}(args[0].clone()))"
                )
        else:
            rust_arms.append(f"                // TODO: special handling for {name}")
            rust_arms.append(
                f'                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(format!("Function {name} not yet implemented in dispatch")))'
            )
        rust_arms.append("            },")

# Generate the dispatch code
dispatch_code = (
    """/// Auto-generated dispatch function for Spark SQL functions.
/// Generated by scripts/gen_fn_dispatch.py - do not edit directly.
pub fn call_builtin(name: &str, args: Vec<spark_connect::column::Column>) -> PyResult<spark_connect::column::Column> {
    use spark_connect::functions as spark_funcs;
    match name {
"""
    + "\n".join(rust_arms)
    + """
        _ => Err(PyErr::new::<pyo3::exceptions::PyNameError, _>(
            format!("Unknown function: {}", name)
        )),
    }
}
"""
)

# Write dispatch file (guarded: the committed file is hand-tuned; see --write-dispatch).
dispatch_file = Path("crates/pyspark-rs/src/dispatch_generated.rs")
if _args.write_dispatch:
    dispatch_file.write_text(dispatch_code)
    print(f"\nGenerated dispatch function in {dispatch_file}")
else:
    print(f"\nSkipped {dispatch_file} (pass --write-dispatch to regenerate the Rust dispatch)")

# Generate Python wrapper functions
# (skip_functions is defined above, before categorization.)
#
# When a reference PySpark is available, emit an explicit ``def`` per function carrying
# the reference's real parameter names/order, so keyword calls (``F.first(col,
# ignorenulls=True)``) work. Optional params default to the ``_UNSET`` sentinel and are
# forwarded only when supplied, reproducing the historical ``*args`` dispatch exactly
# (same columns, same order) — see functions.py ``_dispatch`` / ``_UNSET``.


def _emit_def(name, sig):
    """Return source for ``def name(<ref params>): return _dispatch(name, [...])``."""
    sig_parts = []
    build = []
    posonly = []
    star_emitted = False
    for p in sig.parameters.values():
        k = p.kind
        if k is inspect.Parameter.VAR_KEYWORD:
            continue  # **kwargs are not part of the positional dispatch
        if k is inspect.Parameter.POSITIONAL_ONLY:
            posonly.append(p.name)
            sig_parts.append(p.name)
            build.append(f"    _a.append({p.name})")
        elif k is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if p.default is inspect.Parameter.empty:
                sig_parts.append(p.name)
                build.append(f"    _a.append({p.name})")
            else:
                sig_parts.append(f"{p.name}=_UNSET")
                build.append(f"    if {p.name} is not _UNSET: _a.append({p.name})")
        elif k is inspect.Parameter.VAR_POSITIONAL:
            sig_parts.append(f"*{p.name}")
            build.append(f"    _a.extend({p.name})")
            star_emitted = True
        elif k is inspect.Parameter.KEYWORD_ONLY:
            if not star_emitted:
                sig_parts.append("*")
                star_emitted = True
            sig_parts.append(f"{p.name}=_UNSET")
            build.append(f"    if {p.name} is not _UNSET: _a.append({p.name})")
    if posonly:
        # insert the positional-only marker after the last positional-only param
        idx = len(posonly)
        sig_parts.insert(idx, "/")
    header = f"def {name}({', '.join(sig_parts)}):"
    body = ["    _a = []", *build, f'    return _dispatch("{name}", _a)']
    return header + "\n" + "\n".join(body)


# Upstream bodies.
#
# A function whose PySpark Connect implementation only marshals its arguments into a
# function call -- with lit, _to_col and the other helpers of
# pyspark.sql.connect.functions.builtin -- is generated with that implementation's own
# signature and body. So it sends the arguments that PySpark sends: an optional argument
# when it is given, the defaults that PySpark fills in, a NULL for an explicit None where
# PySpark sends one, and PySpark's errors for arguments of other types. The globals that
# such a body may read are those that the generated module provides:
_UPSTREAM_GLOBALS = {
    "ArrayType",
    "Column",
    "DataType",
    "MapType",
    "PySparkTypeError",
    "PySparkValueError",
    "StructType",
    "_enum_to_value",
    "_ensure_column_or_name",
    "_invoke_binary_math_function",
    "_invoke_function",
    "_invoke_function_over_columns",
    "_invoke_higher_order_function",
    "_options_to_col",
    "_to_col",
    "cast",
    "decimal",
    "lit",
    "py_random",
    "sys",
    "warnings",
}
# The helpers of pyspark.sql.connect.functions.builtin that the generated module defines
# with their PySpark Connect bodies. functions.py injects lit, column, _invoke_function and
# _invoke_higher_order_function, which the Rust core implements.
_UPSTREAM_HELPERS = [
    "_to_col",
    "_invoke_function_over_columns",
    "_invoke_binary_math_function",
    "_ensure_column_or_name",
    "_options_to_col",
]
# Functions that functions.py used to write by hand and that are now generated from their
# PySpark Connect implementations, which they must have.
_UPSTREAM_EXTRA = {
    "array",
    "avg",
    "counter_diff",
    "create_map",
    "cume_dist",
    "hmac",
    "instr",
    "json_tuple",
    "lag",
    "lead",
    "map_concat",
    "max",
    "min",
    "regexp_replace",
    "sha2",
    "struct",
    "sum",
    "try_variant_set",
    "variant_delete",
    "variant_set",
    "variant_strip_nulls",
    "vector_norm",
    "vector_normalize",
    "window",
}


def _code_objects(code):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


def _upstream_impl(name):
    """The PySpark Connect implementation of `name`, when it is a plain function."""
    fn = getattr(_ref_builtin, name, None) if _ref_builtin is not None else None
    if not isinstance(fn, types.FunctionType) or inspect.unwrap(fn) is not fn:
        return None  # Absent, or decorated (as col and lit are, with _with_origin).
    return fn


def _upstream_eligible(names):
    """The names whose implementation reads only the provided globals (or other such
    functions) and imports nothing."""
    reads = {}
    for name in names:
        fn = _upstream_impl(name)
        if fn is None:
            continue
        loaded, imports = set(), False
        for code in _code_objects(fn.__code__):
            for ins in dis.get_instructions(code):
                if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
                    loaded.add(ins.argval)
                elif ins.opname == "IMPORT_NAME":
                    imports = True
        if not imports:
            # Builtins that the module does not shadow are not its globals.
            reads[name] = {n for n in loaded if n in fn.__globals__}
    eligible = set(reads)
    while True:
        kept = {n for n in eligible if reads[n] <= _UPSTREAM_GLOBALS | eligible}
        if kept == eligible:
            return eligible
        eligible = kept


def _is_literal(value):
    try:
        return ast.literal_eval(repr(value)) == value
    except (ValueError, SyntaxError):
        return False


def _upstream_def(name, fn):
    """`def name(<signature of fn without annotations>):` and the body of fn, verbatim
    without its docstring."""
    sig = inspect.signature(fn)
    params = []
    for p in sig.parameters.values():
        if p.default is not inspect.Parameter.empty and not _is_literal(p.default):
            raise SystemExit(f"{name}: the default {p.default!r} of {p.name} is not a literal")
        params.append(p.replace(annotation=inspect.Parameter.empty))
    sig = sig.replace(parameters=params, return_annotation=inspect.Signature.empty)
    src = textwrap.dedent(inspect.getsource(fn))
    node = ast.parse(src).body[0]
    body = node.body
    if (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    lines = src.splitlines()[body[0].lineno - 1 : node.end_lineno]
    return f"def {name}{sig}:\n" + "\n".join(lines)


# The wrapper set is the names already in functions_generated.py (the shipped,
# golden-verified set) and _UPSTREAM_EXTRA. Deriving it from the Rust functions.rs regex is
# fragile: generic signatures (e.g. `pub fn concat_ws<C: Into<Column>>(...)`) are silently
# missed.
wrappers_file = Path("python/pyspark/sql/functions_generated.py")
_existing = wrappers_file.read_text() if wrappers_file.exists() else ""
_wrapper_names = re.findall(r"^([A-Za-z][A-Za-z0-9_]*)\s*=\s*_create_wrapper\(", _existing, re.M)
_wrapper_names += re.findall(r"^def ([A-Za-z][A-Za-z0-9_]*)\(", _existing, re.M)
_wrapper_names = sorted(set(_wrapper_names) | _UPSTREAM_EXTRA)
if not _existing:
    # First-time generation: fall back to the Rust-parsed function set.
    _wrapper_names = sorted(
        {n for n in functions if n not in skip_functions} | _UPSTREAM_EXTRA
    )

_upstream = _upstream_eligible(_wrapper_names)
_missing = sorted(_UPSTREAM_EXTRA - _upstream)
if _missing:
    raise SystemExit(
        f"{', '.join(_missing)}: not generated from a PySpark Connect implementation, which "
        "functions.py relies on (pass --pyspark-src)"
    )

python_wrappers = []
generated_names = []
n_typed = n_upstream = 0
for name in _wrapper_names:
    if name in skip_functions and name not in _UPSTREAM_EXTRA:
        continue
    generated_names.append(name)
    if name in _upstream:
        python_wrappers.append(_upstream_def(name, _upstream_impl(name)))
        n_upstream += 1
        continue
    ref_fn = getattr(_ref_functions, name, None) if _ref_functions is not None else None
    sig = None
    if ref_fn is not None and callable(ref_fn):
        try:
            sig = inspect.signature(ref_fn)
        except (TypeError, ValueError):
            sig = None
    if sig is not None:
        python_wrappers.append(_emit_def(name, sig))
        n_typed += 1
    else:
        # Rust-only extra (not in the reference) — keep the historical *args form.
        python_wrappers.append(f'{name} = _create_wrapper("{name}")')

_helpers = [_upstream_def(h, getattr(_ref_builtin, h)) for h in _UPSTREAM_HELPERS]
_MODULE_HEADER = '''"""Auto-generated wrapper functions for Spark SQL functions.
Generated by scripts/gen_fn_dispatch.py - do not edit directly.

A function whose PySpark Connect implementation only marshals its arguments into a
function call has that implementation's signature and body, and the helpers below are
PySpark Connect's, so it sends the arguments, the defaults and the errors that PySpark
does. The other functions dispatch to the Rust core through _dispatch.

Before loading this module, functions.py injects _create_wrapper, _dispatch, _UNSET,
_wrap, _unwrap and _call_function, and lit, column and _invoke_function, which build a
literal, a column reference and an UnresolvedFunction in the Rust core
(_invoke_higher_order_function follows once defined). It copies the names of __all__.
"""
import decimal
import random as py_random
import sys
import warnings
from typing import cast

from pyspark.errors import PySparkTypeError, PySparkValueError
from pyspark.sql.column import Column
from pyspark.sql.types import ArrayType, DataType, MapType, StructType
from pyspark.sql.utils import enum_to_value as _enum_to_value


# Helpers of pyspark.sql.connect.functions.builtin.
'''
python_code = (
    _MODULE_HEADER
    + "\n\n\n".join(_helpers)
    + "\n\n\n# Generated functions.\n"
    + "\n".join(python_wrappers)
    + "\n\n\n__all__ = [\n"
    + "".join(f'    "{n}",\n' for n in generated_names)
    + "]\n"
)

print(f"Generated {len(python_wrappers)} Python wrapper functions")
print(f"  From their PySpark Connect implementations: {n_upstream}")
print(f"  Dispatching to the Rust core with the reference's signatures: {n_typed}")
print("\nSummary:")
print(f"  Total functions: {len(functions)}")
print(f"  Hand-written (skipped): {len(skip_functions)}")
print(f"  Auto-generated: {len(functions) - len(skip_functions)}")

# Write Python wrappers file (to be imported in functions.py)
wrappers_file.write_text(python_code)
print(f"\nGenerated Python wrappers in {wrappers_file}")
