#!/usr/bin/env python3
"""Auto-capture golden Expression protos for pyspark.sql.connect.functions.

Introspects the functions module, attempts to call each public function with a
small set of heuristic argument tuples, and for each call that yields a Column,
records the normalized expression protobuf. This seeds a broad golden set for
the Rust `functions` builder to test against, and reports how many of the ~576
functions were captured with simple args (the rest need bespoke args and can be
added by hand later). It also records the calls of OPTIONAL_ARG_CASES, of the optional
arguments of PySpark's functions, each under the name of the Rust function that sends it.

Usage:
    python3 scripts/capture_functions_golden.py --remote sc://localhost:15002 \
        --out tests/golden/functions.jsonl
"""

from __future__ import annotations

import argparse
import base64
import inspect
import json
import sys
from pathlib import Path

# The sibling script is importable when this one runs isolated (`python -I`) too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# reuse the normalizer from the sibling script
from capture_golden import normalize  # type: ignore  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.connect import functions as F  # noqa: E402
from pyspark.sql.connect.column import Column  # noqa: E402

# Each candidate is a list of typed arg specs. Recorded in the golden so the
# Rust side can reconstruct the exact same arguments deterministically.
#   col:NAME  -> F.col(NAME)            (Rust: col(NAME))
#   int:N     -> Python int N           (Rust: integer-literal Column)
#   str:S     -> Python str S           (Rust: string-literal Column)
#   litint:N  -> F.lit(N)               (Rust: integer-literal Column)
# and, in the cases of the optional arguments below,
#   litstr:S  -> F.lit(S)               (Rust: string-literal Column)
#   bool:B    -> Python bool B (true or false)
#   float:F   -> Python float F
#   map:K=V;..-> Python dict {K: V, ...}, in its order (Rust: the (name, value) pairs)
#   kw:P=SPEC -> the keyword argument P of SPEC
CANDIDATE_SPECS = [
    [],
    ["col:a"],
    ["col:a", "col:b"],
    ["col:a", "col:b", "col:c"],
    ["col:a", "col:b", "col:c", "col:d"],
    ["col:a", "int:1"],
    ["col:a", "str:x"],
    ["col:a", "col:b", "int:1"],
    ["int:1"],
    ["str:x"],
    ["col:a", "litint:1"],
]


# The optional arguments of PySpark's functions, each captured under the name of the Rust
# function that sends it: (name, the PySpark function, the args of its call).
OPTIONAL_ARG_CASES = [
    # Sent only when they are given.
    ("any_value_with_ignore_nulls", "any_value", ["col:a", "bool:true"]),
    ("array_join_with_null_replacement", "array_join", ["col:a", "str:,", "str:N"]),
    ("assert_true_with_err_msg", "assert_true", ["col:a", "str:boom"]),
    ("btrim_with", "btrim", ["col:a", "col:b"]),
    ("ceiling_scale", "ceiling", ["col:a", "int:2"]),
    ("current_time_with_precision", "current_time", ["int:3"]),
    ("first_value_with_ignore_nulls", "first_value", ["col:a", "bool:true"]),
    ("hll_sketch_agg_with_lg_config_k", "hll_sketch_agg", ["col:a", "int:12"]),
    ("hll_union_agg_with_allow_different_lg_config_k", "hll_union_agg", ["col:a", "bool:true"]),
    (
        "hll_union_with_allow_different_lg_config_k",
        "hll_union",
        ["col:a", "col:b", "bool:true"],
    ),
    ("ilike_with_escape_char", "ilike", ["col:a", "col:b", "litstr:/"]),
    ("kll_merge_agg_bigint_with_k", "kll_merge_agg_bigint", ["col:a", "int:200"]),
    ("kll_merge_agg_double_with_k", "kll_merge_agg_double", ["col:a", "int:200"]),
    ("kll_merge_agg_float_with_k", "kll_merge_agg_float", ["col:a", "int:200"]),
    ("kll_sketch_agg_bigint_with_k", "kll_sketch_agg_bigint", ["col:a", "int:200"]),
    ("kll_sketch_agg_double_with_k", "kll_sketch_agg_double", ["col:a", "int:200"]),
    ("kll_sketch_agg_float_with_k", "kll_sketch_agg_float", ["col:a", "int:200"]),
    ("last_value_with_ignore_nulls", "last_value", ["col:a", "bool:true"]),
    ("levenshtein_with_threshold", "levenshtein", ["col:a", "col:b", "int:2"]),
    ("like_with_escape_char", "like", ["col:a", "col:b", "litstr:/"]),
    ("listagg_with_delimiter", "listagg", ["col:a", "str:,"]),
    ("max_by_with_k", "max_by", ["col:a", "col:b", "int:2"]),
    ("min_by_with_k", "min_by", ["col:a", "col:b", "int:2"]),
    ("parse_url_with_key", "parse_url", ["col:a", "col:b", "col:c"]),
    ("position_with_start", "position", ["col:a", "col:b", "col:c"]),
    ("regexp_extract_all_with_idx", "regexp_extract_all", ["col:a", "col:b", "int:1"]),
    ("regexp_instr_with_idx", "regexp_instr", ["col:a", "col:b", "int:1"]),
    ("replace_with", "replace", ["col:a", "col:b", "col:c"]),
    ("sequence_with_step", "sequence", ["col:a", "col:b", "col:c"]),
    ("st_asbinary_with_endianness", "st_asbinary", ["col:a", "str:NDR"]),
    ("st_geomfromwkb_with_srid", "st_geomfromwkb", ["col:a", "int:4326"]),
    ("string_agg_with_delimiter", "string_agg", ["col:a", "str:,"]),
    ("substr_with_len", "substr", ["col:a", "col:b", "col:c"]),
    ("theta_sketch_agg_with_lg_nom_entries", "theta_sketch_agg", ["col:a", "int:14"]),
    ("theta_union_agg_with_lg_nom_entries", "theta_union_agg", ["col:a", "int:14"]),
    ("theta_union_with_lg_nom_entries", "theta_union", ["col:a", "col:b", "int:14"]),
    ("time_bucket_with_origin", "time_bucket", ["col:a", "col:b", "col:c"]),
    ("try_parse_url_with_key", "try_parse_url", ["col:a", "col:b", "col:c"]),
    (
        "tuple_intersection_agg_double_with_mode",
        "tuple_intersection_agg_double",
        ["col:a", "str:min"],
    ),
    (
        "tuple_intersection_agg_integer_with_mode",
        "tuple_intersection_agg_integer",
        ["col:a", "str:min"],
    ),
    (
        "tuple_intersection_double_with_mode",
        "tuple_intersection_double",
        ["col:a", "col:b", "str:min"],
    ),
    (
        "tuple_intersection_integer_with_mode",
        "tuple_intersection_integer",
        ["col:a", "col:b", "str:min"],
    ),
    (
        "tuple_intersection_theta_double_with_mode",
        "tuple_intersection_theta_double",
        ["col:a", "col:b", "str:min"],
    ),
    (
        "tuple_intersection_theta_integer_with_mode",
        "tuple_intersection_theta_integer",
        ["col:a", "col:b", "str:min"],
    ),
    ("tuple_sketch_summary_double_with_mode", "tuple_sketch_summary_double", ["col:a", "str:min"]),
    (
        "tuple_sketch_summary_integer_with_mode",
        "tuple_sketch_summary_integer",
        ["col:a", "str:min"],
    ),
    # The formats: to_date, to_timestamp and try_to_date send lit(format), and the others the
    # Column of their format.
    ("to_date_with_format", "to_date", ["col:a", "str:yyyy-MM-dd"]),
    ("to_time_with_format", "to_time", ["col:a", "litstr:HH:mm"]),
    ("to_timestamp_with_format", "to_timestamp", ["col:a", "str:yyyy-MM-dd"]),
    ("to_timestamp_ltz_with_format", "to_timestamp_ltz", ["col:a", "litstr:yyyy-MM-dd"]),
    ("to_timestamp_ntz_with_format", "to_timestamp_ntz", ["col:a", "litstr:yyyy-MM-dd"]),
    ("to_unix_timestamp_with_format", "to_unix_timestamp", ["col:a", "litstr:yyyy-MM-dd"]),
    ("try_to_binary_format", "try_to_binary", ["col:a", "litstr:hex"]),
    ("try_to_date_with_format", "try_to_date", ["col:a", "str:yyyy-MM-dd"]),
    ("try_to_time_with_format", "try_to_time", ["col:a", "litstr:HH:mm"]),
    ("try_to_timestamp_with_format", "try_to_timestamp", ["col:a", "litstr:yyyy-MM-dd"]),
    # unix_timestamp of a timestamp sends its default format.
    ("unix_timestamp_with_timestamp", "unix_timestamp", ["col:a"]),
    ("unix_timestamp_with_format", "unix_timestamp", ["col:a", "str:yyyy-MM-dd"]),
    # In place of the default that PySpark sends.
    ("approx_percentile_with_accuracy", "approx_percentile", ["col:a", "float:0.5", "int:100"]),
    ("first_with_ignore_nulls", "first", ["col:a", "bool:true"]),
    ("from_unixtime_with_format", "from_unixtime", ["col:a", "str:yyyy-MM-dd"]),
    ("last_with_ignore_nulls", "last", ["col:a", "bool:true"]),
    ("locate_with_pos", "locate", ["str:x", "col:a", "int:2"]),
    ("mode_with_deterministic", "mode", ["col:a", "bool:true"]),
    ("months_between_with_round_off", "months_between", ["col:a", "col:b", "bool:false"]),
    ("nth_value_with_ignore_nulls", "nth_value", ["col:a", "int:2", "bool:true"]),
    ("overlay_with_len", "overlay", ["col:a", "col:b", "int:2", "int:3"]),
    ("percentile_approx_with_accuracy", "percentile_approx", ["col:a", "float:0.5", "int:100"]),
    ("percentile_with_frequency", "percentile", ["col:a", "float:0.5", "int:2"]),
    ("sort_array_with_asc", "sort_array", ["col:a", "bool:false"]),
    ("split_with_limit", "split", ["col:a", "str:,", "int:2"]),
    # log of two arguments: the base, then the column.
    ("log_with_base", "log", ["float:2.0", "col:a"]),
    # Trailing arguments that PySpark sends with their defaults.
    ("aes_decrypt_with_mode", "aes_decrypt", ["col:a", "col:b", "col:c"]),
    ("aes_decrypt_with_mode_and_padding", "aes_decrypt", ["col:a", "col:b", "col:c", "col:d"]),
    (
        "aes_decrypt_with_mode_padding_and_aad",
        "aes_decrypt",
        ["col:a", "col:b", "col:c", "col:d", "col:e"],
    ),
    ("aes_encrypt_with_mode", "aes_encrypt", ["col:a", "col:b", "col:c"]),
    ("aes_encrypt_with_mode_and_padding", "aes_encrypt", ["col:a", "col:b", "col:c", "col:d"]),
    (
        "aes_encrypt_with_mode_padding_and_iv",
        "aes_encrypt",
        ["col:a", "col:b", "col:c", "col:d", "col:e"],
    ),
    (
        "aes_encrypt_with_mode_padding_iv_and_aad",
        "aes_encrypt",
        ["col:a", "col:b", "col:c", "col:d", "col:e", "col:f"],
    ),
    ("mask_with_upper_char", "mask", ["col:a", "col:b"]),
    ("mask_with_upper_char_and_lower_char", "mask", ["col:a", "col:b", "col:c"]),
    (
        "mask_with_upper_char_lower_char_and_digit_char",
        "mask",
        ["col:a", "col:b", "col:c", "col:d"],
    ),
    (
        "mask_with_upper_char_lower_char_digit_char_and_other_char",
        "mask",
        ["col:a", "col:b", "col:c", "col:d", "col:e"],
    ),
    ("sentences_with_language", "sentences", ["col:a", "col:b"]),
    ("sentences_with_language_and_country", "sentences", ["col:a", "col:b", "col:c"]),
    ("str_to_map_with_pair_delim", "str_to_map", ["col:a", "col:b"]),
    ("str_to_map_with_pair_delim_and_key_value_delim", "str_to_map", ["col:a", "col:b", "col:c"]),
    ("try_aes_decrypt_with_mode", "try_aes_decrypt", ["col:a", "col:b", "col:c"]),
    (
        "try_aes_decrypt_with_mode_and_padding",
        "try_aes_decrypt",
        ["col:a", "col:b", "col:c", "col:d"],
    ),
    (
        "try_aes_decrypt_with_mode_padding_and_aad",
        "try_aes_decrypt",
        ["col:a", "col:b", "col:c", "col:d", "col:e"],
    ),
    *[
        case
        for kind in ("double", "integer")
        for stem, cols in (
            ("tuple_sketch_agg", ["col:a", "col:b"]),
            ("tuple_union_agg", ["col:a"]),
            ("tuple_union", ["col:a", "col:b"]),
            ("tuple_union_theta", ["col:a", "col:b"]),
        )
        for case in (
            (f"{stem}_{kind}_with_lg_nom_entries", f"{stem}_{kind}", [*cols, "int:14"]),
            (
                f"{stem}_{kind}_with_lg_nom_entries_and_mode",
                f"{stem}_{kind}",
                [*cols, "int:14", "str:max"],
            ),
        )
    ],
    # make_timestamp and the like: the time zone after the parts, or a date and a time.
    *[
        (f"{f}_with_timezone", f, ["col:a", "col:b", "col:c", "col:d", "col:e", "col:f", "col:g"])
        for f in (
            "make_timestamp",
            "make_timestamp_ltz",
            "try_make_timestamp",
            "try_make_timestamp_ltz",
        )
    ],
    *[
        (f"{f}_with_date_and_time", f, ["kw:date=col:a", "kw:time=col:b"])
        for f in (
            "make_timestamp",
            "make_timestamp_ntz",
            "try_make_timestamp",
            "try_make_timestamp_ntz",
        )
    ],
    *[
        (
            f"{f}_with_date_time_and_timezone",
            f,
            ["kw:date=col:a", "kw:time=col:b", "kw:timezone=col:c"],
        )
        for f in ("make_timestamp", "try_make_timestamp")
    ],
    # The interval constructors: lit(0) for a part that is None.
    ("make_dt_interval_with_parts", "make_dt_interval", ["kw:hours=col:a"]),
    ("make_interval_with_parts", "make_interval", ["kw:years=col:a", "kw:secs=col:b"]),
    ("make_ym_interval_with_parts", "make_ym_interval", ["kw:months=col:a"]),
    ("try_make_interval_with_parts", "try_make_interval", ["kw:months=col:a"]),
    # The options: the map of _options_to_col, in their order, and map() of none.
    ("from_csv_with_options", "from_csv", ["col:a", "col:b", "map:sep=|"]),
    (
        "from_json_with_options",
        "from_json",
        ["col:a", "col:b", "map:mode=FAILFAST;allowComments=true"],
    ),
    ("from_xml_with_options", "from_xml", ["col:a", "col:b", "map:rowTag=r"]),
    ("schema_of_csv_with_options", "schema_of_csv", ["col:a", "map:sep=|"]),
    ("schema_of_json_with_options", "schema_of_json", ["col:a", "map:allowComments=true"]),
    ("schema_of_xml_with_options", "schema_of_xml", ["col:a", "map:rowTag=r"]),
    ("to_csv_with_options", "to_csv", ["col:a", "map:"]),
    ("to_json_with_options", "to_json", ["col:a", "map:pretty=true;ignoreNullFields=false"]),
    ("to_xml_with_options", "to_xml", ["col:a", "map:rowTag=r"]),
    # A sliding window, without a start time.
    ("window_with_slide", "window", ["col:a", "str:10 minutes", "str:5 minutes"]),
]


def materialize_one(tok, F):
    kind, _, val = tok.partition(":")
    if kind == "col":
        return F.col(val)
    if kind == "int":
        return int(val)
    if kind == "str":
        return val
    if kind == "litint":
        return F.lit(int(val))
    if kind == "litstr":
        return F.lit(val)
    if kind == "bool":
        return {"true": True, "false": False}[val]
    if kind == "float":
        return float(val)
    if kind == "map":
        return dict(item.split("=", 1) for item in val.split(";") if item)
    raise ValueError(tok)


def materialize(spec, F):
    return [materialize_one(tok, F) for tok in spec]


def materialize_call(spec, F):
    """The positional and the keyword arguments of `spec`."""
    args, kwargs = [], {}
    for tok in spec:
        kind, _, val = tok.partition(":")
        if kind == "kw":
            name, _, arg = val.partition("=")
            kwargs[name] = materialize_one(arg, F)
        else:
            args.append(materialize_one(tok, F))
    return args, kwargs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument("--out", default="tests/golden/functions.jsonl")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()

    names = sorted(
        n
        for n in dir(F)
        if not n.startswith("_") and callable(getattr(F, n)) and not inspect.isclass(getattr(F, n))
    )

    out_records = []
    captured = 0
    skipped = []
    for name in names:
        fn = getattr(F, name)
        got = None
        for spec in CANDIDATE_SPECS:
            try:
                argv = materialize(spec, F)
                res = fn(*argv)
            except Exception:
                continue
            if isinstance(res, Column):
                try:
                    expr = res._expr.to_plan(spark.client)
                except Exception:
                    continue
                normalize(expr)
                got = (spec, expr)
                break
        if got is None:
            skipped.append(name)
            continue
        spec, expr = got
        out_records.append(
            dict(
                name=name,
                kind="function",
                args=spec,
                b64=base64.b64encode(expr.SerializeToString()).decode(),
                text=str(expr),
            )
        )
        captured += 1

    for name, fname, spec in OPTIONAL_ARG_CASES:
        argv, kwargs = materialize_call(spec, F)
        expr = getattr(F, fname)(*argv, **kwargs)._expr.to_plan(spark.client)
        normalize(expr)
        out_records.append(
            dict(
                name=name,
                kind="function",
                function=fname,
                args=spec,
                b64=base64.b64encode(expr.SerializeToString()).decode(),
                text=str(expr),
            )
        )
    names_seen = [r["name"] for r in out_records]
    duplicates = sorted({n for n in names_seen if names_seen.count(n) > 1})
    if duplicates:
        raise SystemExit(f"cases of the same name: {duplicates}")
    out_records.sort(key=lambda r: r["name"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in out_records) + "\n")
    print(f"functions total={len(names)} captured={captured} skipped={len(skipped)}")
    print(f"optional-argument cases={len(OPTIONAL_ARG_CASES)}")
    print(f"wrote {out}")
    print("first 25 skipped (need bespoke args):", ", ".join(skipped[:25]))
    spark.stop()


if __name__ == "__main__":
    main()
