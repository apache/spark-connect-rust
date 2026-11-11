#!/usr/bin/env python3
"""
Generator for the Rust dispatch and the Python wrappers of the Spark SQL functions.

It parses crates/spark-connect/src/functions.rs and generates

  * crates/pyspark-rs/src/dispatch_generated.rs  (``call_builtin``: function name ->
    call into the Rust core), and
  * python/pyspark/sql/functions_generated.py    (the ``pyspark.sql.functions`` wrappers;
    needs ``--pyspark-src``).

Both files are fully generated: every arm or wrapper that does not follow from a
function's Rust signature is declared in one of the tables below, so regenerating
reproduces the checked-in files byte for byte. ``--check`` regenerates in memory and
fails with a diff when a checked-in file differs; CI runs it
(.github/workflows/generated-drift.yml), and scripts/tests/test_gen_fn_dispatch.py runs
the Rust half without a reference PySpark.

Usage, from anywhere:

    python scripts/gen_fn_dispatch.py --pyspark-src <spark>/python            # write
    python scripts/gen_fn_dispatch.py --pyspark-src <spark>/python --check    # verify

``<spark>/python`` is the python/ directory of an Apache Spark checkout at the tracked
tag (SPARK_TAG of scripts/check_vendored_upstream.sh), run with an interpreter in which
the dependencies of PySpark Connect (pandas, pyarrow, grpcio, ...) import. Without
``--pyspark-src`` only the Rust dispatch is generated (or checked).
"""

from __future__ import annotations

import argparse
import ast
import difflib
import dis
import inspect
import re
import sys
import textwrap
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FUNCTIONS_RS = Path("crates/spark-connect/src/functions.rs")
DISPATCH_FILE = Path("crates/pyspark-rs/src/dispatch_generated.rs")
WRAPPERS_FILE = Path("python/pyspark/sql/functions_generated.py")


class GeneratorError(Exception):
    """The Rust source and the tables of this generator disagree."""


# ---------------------------------------------------------------------------------------
# Parsing crates/spark-connect/src/functions.rs
# ---------------------------------------------------------------------------------------


def _matching(src: str, start: int, opening: str, closing: str) -> int:
    """The index of the `closing` that balances the `opening` at `start`."""
    depth = 0
    for i in range(start, len(src)):
        ch = src[i]
        if ch == opening:
            depth += 1
        elif ch == closing and not (closing == ">" and src[i - 1] == "-"):  # not a `->`
            depth -= 1
            if depth == 0:
                return i
    raise GeneratorError(f"unbalanced {opening!r} at offset {start} of {FUNCTIONS_RS}")


def _split_top_level(text: str) -> list[str]:
    """Split on the commas that are outside any bracket."""
    parts, current, depth = [], "", 0
    for i, ch in enumerate(text):
        if ch in "<([":
            depth += 1
        elif ch in ")]" or (ch == ">" and text[i - 1] != "-"):
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def parse_rust_functions(src: str) -> dict[str, dict]:
    """The top-level ``pub fn``s that return a ``Column``, in source order.

    name -> {"full_name": the Rust identifier (``r#struct``), "generics": the text between
    ``<`` and ``>`` after the name, "params": [{"name", "type"}], "order": its rank}.
    """
    functions: dict[str, dict] = {}
    for match in re.finditer(r"^pub fn ((?:r#)?[A-Za-z0-9_]+)", src, re.MULTILINE):
        full_name = match.group(1)
        pos = match.end()
        generics = ""
        if src[pos] == "<":
            end = _matching(src, pos, "<", ">")
            generics = " ".join(src[pos + 1 : end].split())
            pos = end + 1
        if src[pos] != "(":
            raise GeneratorError(f"{full_name}: expected '(' after the function name")
        end = _matching(src, pos, "(", ")")
        params_text = " ".join(src[pos + 1 : end].split())
        returns = re.match(r"\s*->\s*([^\s{]+)", src[end + 1 :])
        if returns is None or returns.group(1) != "Column":
            continue  # e.g. broadcast(DataFrame) -> DataFrame
        params = []
        for part in _split_top_level(params_text):
            pname, _, ptype = part.partition(":")
            params.append({"name": pname.strip(), "type": ptype.strip()})
        name = full_name.replace("r#", "")
        if name in functions:
            raise GeneratorError(f"{name}: defined twice in {FUNCTIONS_RS}")
        functions[name] = {
            "full_name": full_name,
            "generics": generics,
            "params": params,
            "order": len(functions),
        }
    return functions


def _is_column(fn: dict, ptype: str) -> bool:
    return ptype == "Column"


def _is_variadic(fn: dict, ptype: str) -> bool:
    """``Vec<Column>``, or an ``impl IntoIterator`` of ``Column`` or of ``Into<Column>``."""
    if ptype == "Vec<Column>":
        return True
    item = re.fullmatch(r"impl IntoIterator<Item = (\w+)>", ptype)
    if item is None:
        return False
    if item.group(1) == "Column":
        return True
    bounds = dict(
        (g.partition(":")[0].strip(), g.partition(":")[2].strip())
        for g in _split_top_level(fn["generics"])
    )
    return bounds.get(item.group(1)) == "Into<Column>"


# ---------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------

# Functions without a generated Python wrapper: functions.py writes them by hand or binds
# them to a dedicated pyfunc_*. Unless DISPATCH_ONLY lists them, they have no dispatch arm
# either, their Rust signature not being a list of columns.
SKIP_FUNCTIONS = {
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
    # when (a CASE WHEN), and the deprecated partition transforms, which call those of
    # pyspark.sql.functions.partitioning: written in functions.py as PySpark Connect's.
    "when",
    "years",
    "months",
    "days",
    "hours",
    "bucket",
}

# The SKIP_FUNCTIONS that still have a dispatch arm: their Python wrapper is written by
# hand in functions.py (which dispatches count_distinct, listagg_distinct and
# string_agg_distinct by name), but call_builtin resolves them.
DISPATCH_ONLY = {
    "bucket",
    "countDistinct",
    "count_distinct",
    "days",
    "hours",
    "listagg_distinct",
    "months",
    "string_agg_distinct",
    "variant_delete",
    "when",
    "years",
}

# PySpark functions with trailing optional arguments, which the Rust core spells as one
# function per arity: base -> the functions taking one more argument, two more, ... The
# arm of the base picks the function by the number of arguments it gets, and the longer
# functions have no arm (nor Python wrapper) of their own.
OPTIONAL_ARG_VARIANTS = {
    "approx_count_distinct": ["approx_count_distinct_rsd"],
    "bround": ["bround_scale"],
    "ceil": ["ceil_scale"],
    "counter_diff": ["counter_diff_with_start_time"],
    "floor": ["floor_scale"],
    "hmac": ["hmac_with_algorithm"],
    "instr": ["instr_with_start", "instr_with_start_and_occurrence"],
    "listagg_distinct": ["listagg_distinct_with_delimiter"],
    "ltrim": ["ltrim_with"],
    "regexp_replace": ["regexp_replace_with_position"],
    "round": ["round_scale"],
    "rtrim": ["rtrim_with"],
    "string_agg_distinct": ["string_agg_distinct_with_delimiter"],
    "to_binary": ["to_binary_format"],
    "trim": ["trim_with"],
    "try_variant_set": ["try_variant_set_with_create_if_missing"],
    "variant_set": ["variant_set_with_create_if_missing"],
    "vector_norm": ["vector_norm_with_degree"],
    "vector_normalize": ["vector_normalize_with_degree"],
}

# Functions of a column followed by any number of columns: name -> the Rust function
# ``f(Column, impl IntoIterator<..>)`` that the arm calls with ``args[0]`` and the rest.
# A target under another name has no arm (nor Python wrapper) of its own.
HEAD_REST_ARMS = {
    "countDistinct": "count_distinct_columns",
    "count_distinct": "count_distinct_columns",
    "variant_delete": "variant_delete",
}

# Rust functions that are not reachable through call_builtin, and why. (They have no
# generated Python wrapper either.) Listing one here is a decision; a function that fits
# no rule and no table fails the generation instead of being dropped silently.
NOT_DISPATCHED = {
    # Window function without arguments: the Python wrapper is generated from its PySpark
    # Connect implementation (UPSTREAM_EXTRA), which does not go through call_builtin.
    "cume_dist": "generated from its PySpark Connect implementation",
    # An alias of rand() in the Rust core only; the Python `random` is not generated here.
    "random": "Rust-only alias of rand",
    # The seeded forms take the seed as an i64, not as a column.
    "rand_with_seed": "takes an i64 seed",
    "randn_with_seed": "takes an i64 seed",
    "random_with_seed": "takes an i64 seed",
    # Seeded forms of the Rust core. The Python wrappers of their bases are generated from
    # the PySpark Connect implementations, which send the seed themselves.
    "count_min_sketch_with_seed": "the Python wrapper of the base sends the seed",
    "randstr_with_seed": "the Python wrapper of the base sends the seed",
    "shuffle_with_seed": "the Python wrapper of the base sends the seed",
    "uniform_with_seed": "the Python wrapper of the base sends the seed",
    "uuid_with_seed": "the Python wrapper of the base sends the seed",
    # Higher-order functions: they take Rust closures. functions.py builds their lambdas
    # with _invoke_higher_order_function.
    "aggregate": "takes a Rust closure",
    "aggregate_with_finish": "takes a Rust closure",
    "exists": "takes a Rust closure",
    "filter": "takes a Rust closure",
    "forall": "takes a Rust closure",
    "map_filter": "takes a Rust closure",
    "map_zip_with": "takes a Rust closure",
    "reduce": "takes a Rust closure",
    "transform": "takes a Rust closure",
    "transform_idx": "takes a Rust closure",
    "transform_keys": "takes a Rust closure",
    "transform_values": "takes a Rust closure",
    "zip_with": "takes a Rust closure",
}

# The arms come in four sections -- no argument, one column, any number of columns,
# several columns -- and in the order of functions.rs within a section. The two tables
# below record where an arm stands when that is not the place its signature gives it.
# They only keep the generated file stable (the order of the arms of a match on distinct
# string literals has no effect); a new function needs no entry.
SECTIONS = ("no_args", "single", "variadic", "multiple")
# Variadic functions that stand in another section than "variadic".
SECTION_OVERRIDES = {
    "concat_ws": "single",
    "format_string": "single",
    "json_tuple": "multiple",
    "lag": "single",
    "lead": "single",
}
# name -> the function at whose place in functions.rs the arm is ordered.
ORDER_AS = {
    "variant_delete": "variant_strip_nulls",
}


# ---------------------------------------------------------------------------------------
# The Rust dispatch
# ---------------------------------------------------------------------------------------

_TYPE_ERROR = "PyErr::new::<pyo3::exceptions::PyTypeError, _>"
_ARM = " " * 12
_BODY = " " * 16
_NESTED = " " * 20


def _call(full_name: str, count: int) -> str:
    args = ", ".join(f"args[{i}].clone()" for i in range(count))
    return f"spark_funcs::{full_name}({args})"


def _guard(name: str, count: int, section: str) -> str:
    """The line that rejects a call with fewer than the `count` required arguments."""
    if section == "single":
        return (
            f"{_BODY}if args.is_empty() {{ return Err({_TYPE_ERROR}"
            '("Missing required column argument")); }'
        )
    return (
        f"{_BODY}if args.len() < {count} {{ return Err({_TYPE_ERROR}(format!("
        f'"Missing required arguments for {name}: expected at least {count}, got {{}}", '
        "args.len()))); }"
    )


def plan_dispatch(
    functions: dict[str, dict],
    *,
    skip=None,
    dispatch_only=None,
    optional_arg_variants=None,
    head_rest_arms=None,
    not_dispatched=None,
    section_overrides=None,
    order_as=None,
) -> list[dict]:
    """The arms of call_builtin, in the order they are written.

    Each arm is {"name", "section", "kind", ...} with kind one of "plain" (the function,
    with its `count` column arguments), "variadic" (all the arguments), "optional" (the
    function or one of its longer `variants`) and "head_rest" (`target`, with the first
    argument and the rest). Raises GeneratorError when a function of the Rust core is not
    accounted for, or when a table names a function that does not fit it.
    """
    skip = SKIP_FUNCTIONS if skip is None else skip
    dispatch_only = DISPATCH_ONLY if dispatch_only is None else dispatch_only
    variants_of = OPTIONAL_ARG_VARIANTS if optional_arg_variants is None else optional_arg_variants
    head_rest = HEAD_REST_ARMS if head_rest_arms is None else head_rest_arms
    not_dispatched = NOT_DISPATCHED if not_dispatched is None else not_dispatched
    section_overrides = SECTION_OVERRIDES if section_overrides is None else section_overrides
    order_as = ORDER_AS if order_as is None else order_as

    def known(name: str, table: str) -> dict:
        if name not in functions:
            raise GeneratorError(f"{table} names {name}, which {FUNCTIONS_RS} does not define")
        return functions[name]

    def column_count(name: str, table: str) -> int:
        fn = known(name, table)
        if not all(_is_column(fn, p["type"]) for p in fn["params"]):
            raise GeneratorError(f"{table}: {name} does not take only columns")
        return len(fn["params"])

    stale = sorted(dispatch_only - skip)
    if stale:
        raise GeneratorError(f"DISPATCH_ONLY names {', '.join(stale)}, not in SKIP_FUNCTIONS")
    for name in section_overrides:
        known(name, "SECTION_OVERRIDES")
    for name, anchor in order_as.items():
        known(name, "ORDER_AS")
        if anchor not in functions:
            raise GeneratorError(f"ORDER_AS orders {name} as {anchor}, which is not defined")
    for name in not_dispatched:
        known(name, "NOT_DISPATCHED")

    # The functions that another arm calls and that have no arm of their own.
    absorbed: dict[str, str] = {}
    for base, variants in variants_of.items():
        count = column_count(base, "OPTIONAL_ARG_VARIANTS")
        if count == 0 or not variants:
            raise GeneratorError(f"OPTIONAL_ARG_VARIANTS: {base} needs an argument and a variant")
        for extra, variant in enumerate(variants, start=1):
            if column_count(variant, "OPTIONAL_ARG_VARIANTS") != count + extra:
                raise GeneratorError(
                    f"OPTIONAL_ARG_VARIANTS: {variant} must take {count + extra} columns, "
                    f"{extra} more than {base}"
                )
            if absorbed.setdefault(variant, base) != base:
                raise GeneratorError(f"{variant} is a variant of both {absorbed[variant]}, {base}")
    for name, target in head_rest.items():
        known(name, "HEAD_REST_ARMS")
        params = known(target, "HEAD_REST_ARMS")["params"]
        if not (
            len(params) == 2
            and _is_column(functions[target], params[0]["type"])
            and _is_variadic(functions[target], params[1]["type"])
        ):
            raise GeneratorError(f"HEAD_REST_ARMS: {target} is not f(Column, columns)")
        if target != name:
            absorbed.setdefault(target, name)

    excluded = (skip - dispatch_only) | set(not_dispatched) | set(absorbed)
    for name in sorted(excluded & (set(variants_of) | set(head_rest))):
        raise GeneratorError(f"{name} has an arm in a table but is also excluded from dispatch")

    arms = []
    for name, fn in functions.items():
        if name in excluded:
            continue
        types_ = [p["type"] for p in fn["params"]]
        all_columns = all(_is_column(fn, t) for t in types_)
        arm = {"name": name, "full_name": fn["full_name"], "count": len(types_)}
        if name in head_rest:
            arm.update(kind="head_rest", section="single", target=head_rest[name])
            arm["target_full_name"] = functions[head_rest[name]]["full_name"]
        elif name in variants_of:
            arm.update(
                kind="optional",
                section="single" if len(types_) == 1 else "multiple",
                variants=[functions[v]["full_name"] for v in variants_of[name]],
            )
        elif not types_:
            arm.update(kind="plain", section="no_args")
        elif len(types_) == 1 and _is_variadic(fn, types_[0]):
            arm.update(kind="variadic", section="variadic")
        elif all_columns:
            arm.update(kind="plain", section="single" if len(types_) == 1 else "multiple")
        else:
            signature = ", ".join(f"{p['name']}: {p['type']}" for p in fn["params"])
            raise GeneratorError(
                f"{name}({signature}) is not a function of columns: add it to "
                "SKIP_FUNCTIONS (hand-written in functions.py), OPTIONAL_ARG_VARIANTS, "
                "HEAD_REST_ARMS or NOT_DISPATCHED in scripts/gen_fn_dispatch.py"
            )
        if name in section_overrides:
            if arm["kind"] != "variadic":
                raise GeneratorError(f"SECTION_OVERRIDES: {name} is not a variadic function")
            arm["section"] = section_overrides[name]
        arm["order"] = functions[order_as.get(name, name)]["order"]
        arms.append(arm)

    arms.sort(key=lambda a: (SECTIONS.index(a["section"]), a["order"]))
    return arms


def render_arm(arm: dict) -> list[str]:
    name, full_name, count = arm["name"], arm["full_name"], arm["count"]
    if arm["kind"] == "variadic":
        return [f'{_ARM}"{name}" => Ok(spark_funcs::{full_name}(args.clone())),']
    if arm["kind"] == "plain" and count == 0:
        return [f'{_ARM}"{name}" => Ok(spark_funcs::{full_name}()),']
    lines = [f'{_ARM}"{name}" => {{', _guard(name, count, arm["section"])]
    if arm["kind"] == "plain":
        lines.append(f"{_BODY}Ok({_call(full_name, count)})")
    elif arm["kind"] == "head_rest":
        lines.append(
            f"{_BODY}Ok(spark_funcs::{arm['target_full_name']}"
            "(args[0].clone(), args[1..].to_vec()))"
        )
    elif len(arm["variants"]) == 1:
        lines.append(
            f"{_BODY}if args.len() >= {count + 1} "
            f"{{ Ok({_call(arm['variants'][0], count + 1)}) }} "
            f"else {{ Ok({_call(full_name, count)}) }}"
        )
    else:
        lines.append(f"{_BODY}match args.len() {{")
        lines.append(f"{_NESTED}{count} => Ok({_call(full_name, count)}),")
        for extra, variant in enumerate(arm["variants"][:-1], start=1):
            lines.append(f"{_NESTED}{count + extra} => Ok({_call(variant, count + extra)}),")
        last = len(arm["variants"])
        lines.append(f"{_NESTED}_ => Ok({_call(arm['variants'][-1], count + last)}),")
        lines.append(f"{_BODY}}}")
    lines.append(f"{_ARM}}},")
    return lines


_DISPATCH_HEADER = """/// Auto-generated dispatch function for Spark SQL functions.
/// Generated by scripts/gen_fn_dispatch.py - do not edit directly.
pub fn call_builtin(name: &str, args: Vec<spark_connect::column::Column>) -> PyResult<spark_connect::column::Column> {
    use spark_connect::functions as spark_funcs;
    match name {
"""
_DISPATCH_FOOTER = """
        _ => Err(PyErr::new::<pyo3::exceptions::PyNameError, _>(
            format!("Unknown function: {}", name)
        )),
    }
}
"""


def render_dispatch(arms: list[dict]) -> str:
    lines = [line for arm in arms for line in render_arm(arm)]
    return _DISPATCH_HEADER + "\n".join(lines) + _DISPATCH_FOOTER


# ---------------------------------------------------------------------------------------
# The Python wrappers
# ---------------------------------------------------------------------------------------
#
# With the reference PySpark, an explicit ``def`` is emitted per function, carrying the
# reference's real parameter names/order, so keyword calls (``F.first(col,
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
UPSTREAM_GLOBALS = {
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
UPSTREAM_HELPERS = [
    "_to_col",
    "_invoke_function_over_columns",
    "_invoke_binary_math_function",
    "_ensure_column_or_name",
    "_options_to_col",
]
# Functions that functions.py used to write by hand and that are now generated from their
# PySpark Connect implementations, which they must have.
UPSTREAM_EXTRA = {
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


def wrapper_names(arms: list[dict], *, skip=None, upstream_extra=None) -> list[str]:
    """The functions of functions_generated.py: those with a dispatch arm, less the ones
    functions.py writes by hand, and those generated from PySpark Connect."""
    skip = SKIP_FUNCTIONS if skip is None else skip
    upstream_extra = UPSTREAM_EXTRA if upstream_extra is None else upstream_extra
    return sorted({arm["name"] for arm in arms if arm["name"] not in skip} | upstream_extra)


def _code_objects(code):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


def _upstream_impl(ref_builtin, name):
    """The PySpark Connect implementation of `name`, when it is a plain function."""
    fn = getattr(ref_builtin, name, None)
    if not isinstance(fn, types.FunctionType) or inspect.unwrap(fn) is not fn:
        return None  # Absent, or decorated (as col and lit are, with _with_origin).
    return fn


def _upstream_eligible(ref_builtin, names):
    """The names whose implementation reads only the provided globals (or other such
    functions) and imports nothing."""
    reads = {}
    for name in names:
        fn = _upstream_impl(ref_builtin, name)
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
        kept = {n for n in eligible if reads[n] <= UPSTREAM_GLOBALS | eligible}
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
            raise GeneratorError(f"{name}: the default {p.default!r} of {p.name} is not a literal")
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


def load_reference(pyspark_src: str):
    """(pyspark.sql.connect.functions.builtin, pyspark.sql.functions) of the reference
    PySpark whose python/ directory is `pyspark_src`."""
    root = Path(pyspark_src).resolve()
    if not (root / "pyspark" / "sql" / "connect" / "functions" / "builtin.py").is_file():
        raise GeneratorError(f"--pyspark-src {pyspark_src}: not the python/ directory of Spark")
    sys.path.insert(0, str(root))
    import pyspark.sql.connect.functions.builtin as ref_builtin
    import pyspark.sql.functions as ref_functions

    loaded = Path(ref_builtin.__file__).resolve()
    if root not in loaded.parents:
        raise GeneratorError(f"pyspark was imported from {loaded}, not from --pyspark-src")
    return ref_builtin, ref_functions


def render_wrappers(names: list[str], ref_builtin, ref_functions, *, upstream_extra=None):
    """The text of functions_generated.py, and how many functions come from their PySpark
    Connect implementation and how many dispatch with the reference's signature."""
    upstream_extra = UPSTREAM_EXTRA if upstream_extra is None else upstream_extra
    upstream = _upstream_eligible(ref_builtin, names)
    missing = sorted(upstream_extra - upstream)
    if missing:
        raise GeneratorError(
            f"{', '.join(missing)}: not generated from a PySpark Connect implementation, "
            "which functions.py relies on"
        )

    python_wrappers = []
    n_typed = n_upstream = 0
    for name in names:
        if name in upstream:
            python_wrappers.append(_upstream_def(name, _upstream_impl(ref_builtin, name)))
            n_upstream += 1
            continue
        ref_fn = getattr(ref_functions, name, None)
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

    helpers = [_upstream_def(h, getattr(ref_builtin, h)) for h in UPSTREAM_HELPERS]
    code = (
        _MODULE_HEADER
        + "\n\n\n".join(helpers)
        + "\n\n\n# Generated functions.\n"
        + "\n".join(python_wrappers)
        + "\n\n\n__all__ = [\n"
        + "".join(f'    "{n}",\n' for n in names)
        + "]\n"
    )
    return code, n_upstream, n_typed


# ---------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------


def generate(root: Path, pyspark_src: str | None = None, log=print) -> dict[Path, str]:
    """The generated files, by path relative to `root`. functions_generated.py is among
    them only with a reference PySpark."""
    functions = parse_rust_functions((root / FUNCTIONS_RS).read_text())
    arms = plan_dispatch(functions)
    log(f"Found {len(functions)} functions returning a Column in {FUNCTIONS_RS}")
    for section in SECTIONS:
        log(f"  {section}: {sum(1 for a in arms if a['section'] == section)} arms")
    outputs = {DISPATCH_FILE: render_dispatch(arms)}
    if pyspark_src is not None:
        ref_builtin, ref_functions = load_reference(pyspark_src)
        names = wrapper_names(arms)
        code, n_upstream, n_typed = render_wrappers(names, ref_builtin, ref_functions)
        log(f"Generated {len(names)} Python wrapper functions")
        log(f"  From their PySpark Connect implementations: {n_upstream}")
        log(f"  Dispatching to the Rust core with the reference's signatures: {n_typed}")
        outputs[WRAPPERS_FILE] = code
    return outputs


def check(root: Path, outputs: dict[Path, str], log=print) -> bool:
    """Whether the checked-in files equal `outputs`; logs a unified diff of each that
    does not."""
    clean = True
    for path, expected in outputs.items():
        target = root / path
        actual = target.read_text() if target.is_file() else ""
        if actual == expected:
            log(f"OK    {path} matches its generator")
            continue
        clean = False
        log(f"DRIFT {path} differs from the output of scripts/gen_fn_dispatch.py:")
        diff = difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=f"{path} (checked in)",
            tofile=f"{path} (generated)",
        )
        for line in diff:
            log(line.rstrip("\n"))
    return clean


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pyspark-src",
        default=None,
        help="Path to the reference PySpark package root (…/python), with the dependencies of "
        "PySpark Connect importable. A function whose PySpark Connect implementation only "
        "marshals its arguments into a function call is generated with that "
        "implementation's signature and body; the others carry the reference's parameter "
        "names and dispatch to the Rust core. Without it, functions_generated.py is "
        "neither written nor checked.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Write nothing: regenerate in memory, print a diff of every checked-in file "
        "that differs from its generated text, and exit with status 1 if any does.",
    )
    args = parser.parse_args(argv)

    try:
        outputs = generate(REPO_ROOT, args.pyspark_src)
    except GeneratorError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if WRAPPERS_FILE not in outputs:
        print(f"NOTE  {WRAPPERS_FILE} needs --pyspark-src: not generated, not checked")

    if args.check:
        if check(REPO_ROOT, outputs):
            return 0
        print(
            "\nGenerated files have drifted. Do not edit them by hand: change "
            "scripts/gen_fn_dispatch.py (or its tables) and regenerate with\n"
            "  python scripts/gen_fn_dispatch.py --pyspark-src <spark at the tracked tag>/python"
        )
        return 1

    for path, text in outputs.items():
        (REPO_ROOT / path).write_text(text)
        print(f"Generated {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
