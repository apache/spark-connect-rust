//! Golden parity test for all 440 `functions.*` builders.
//!
//! Each function is constructed with canonical column/string arguments, serialized
//! to protobuf, normalized, and compared byte-for-byte against the reference
//! PySpark client output captured in `tests/golden/functions.jsonl`. Any missing
//! or mismatched case FAILS the test - no silent skips.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect_proto as proto;

/// Random-seed functions whose trailing `long` seed is non-deterministic
/// (PySpark picks a random seed when none is supplied). We zero it on both sides.
const RANDOM_SEED_FUNCS: &[&str] = &["rand", "randn", "randstr", "shuffle", "uniform"];

/// Recursively clear run-to-run noise: `common` (Python origin) everywhere,
/// attribute/regex `plan_id`, and random seeds inside random-seed functions.
fn normalize(e: &mut proto::Expression) {
    use proto::expression::ExprType as T;
    e.common = None;
    if let Some(t) = e.expr_type.as_mut() {
        match t {
            T::UnresolvedAttribute(a) => a.plan_id = None,
            T::UnresolvedRegex(r) => r.plan_id = None,
            T::UnresolvedFunction(f) => {
                let is_random = RANDOM_SEED_FUNCS.contains(&f.function_name.as_str());
                for a in f.arguments.iter_mut() {
                    if is_random {
                        if let Some(T::Literal(lit)) = a.expr_type.as_mut() {
                            if let Some(proto::expression::literal::LiteralType::Long(v)) =
                                lit.literal_type.as_mut()
                            {
                                *v = 0;
                            }
                        }
                    }
                    normalize(a);
                }
            }
            T::Alias(a) => {
                if let Some(x) = a.expr.as_deref_mut() {
                    normalize(x);
                }
            }
            T::Cast(c) => {
                if let Some(x) = c.expr.as_deref_mut() {
                    normalize(x);
                }
            }
            T::SortOrder(s) => {
                if let Some(x) = s.child.as_deref_mut() {
                    normalize(x);
                }
            }
            T::UnresolvedExtractValue(v) => {
                if let Some(x) = v.child.as_deref_mut() {
                    normalize(x);
                }
                if let Some(x) = v.extraction.as_deref_mut() {
                    normalize(x);
                }
            }
            T::CallFunction(cf) => {
                for a in cf.arguments.iter_mut() {
                    normalize(a);
                }
            }
            _ => {}
        }
    }
}

fn load_goldens() -> HashMap<String, proto::Expression> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/functions.jsonl"
    );
    let file = File::open(path).expect("golden file functions.jsonl missing");
    let mut out = HashMap::new();
    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let obj: serde_json::Value = serde_json::from_str(&line).unwrap();
        let name = obj["name"].as_str().unwrap().to_string();
        let b64 = obj["b64"].as_str().unwrap();
        let bytes = STANDARD.decode(b64).unwrap();
        let mut expr = proto::Expression::decode(&bytes[..]).unwrap();
        normalize(&mut expr);
        out.insert(name, expr);
    }
    out
}

#[test]
fn all_440_golden_function_cases_pass() {
    use spark_connect::column::Column;
    use spark_connect::expression::{ColumnReference, Expression};
    use spark_connect::functions::*;

    let goldens = load_goldens();
    let a = || Column::new(Expression::ColumnReference(ColumnReference::new("a")));
    let b = || Column::new(Expression::ColumnReference(ColumnReference::new("b")));
    let c = || Column::new(Expression::ColumnReference(ColumnReference::new("c")));
    let d = || Column::new(Expression::ColumnReference(ColumnReference::new("d")));

    let cases: Vec<(&str, Column)> = vec![
        ("abs", abs(a())),
        ("acos", acos(a())),
        ("acosh", acosh(a())),
        ("add_months", add_months(a(), b())),
        ("aes_decrypt", aes_decrypt(a(), b())),
        ("aes_encrypt", aes_encrypt(a(), b())),
        ("any_value", any_value(a())),
        ("approxCountDistinct", approxCountDistinct(a())),
        ("approx_count_distinct", approx_count_distinct(a())),
        ("approx_percentile", approx_percentile(a(), b())),
        ("array", array()),
        ("array_agg", array_agg(a())),
        ("array_append", array_append(a(), b())),
        ("array_compact", array_compact(a())),
        ("array_contains", array_contains(a(), b())),
        ("array_distinct", array_distinct(a())),
        ("array_except", array_except(a(), b())),
        ("array_insert", array_insert(a(), b(), c())),
        ("array_intersect", array_intersect(a(), b())),
        ("array_join", array_join(a(), b())),
        ("array_max", array_max(a())),
        ("array_min", array_min(a())),
        ("array_position", array_position(a(), b())),
        ("array_prepend", array_prepend(a(), b())),
        ("array_remove", array_remove(a(), b())),
        ("array_repeat", array_repeat(a(), b())),
        ("array_size", array_size(a())),
        ("array_sort", array_sort(a())),
        ("array_union", array_union(a(), b())),
        ("arrays_overlap", arrays_overlap(a(), b())),
        ("arrays_zip", arrays_zip()),
        ("asc", asc(a())),
        ("asc_nulls_first", asc_nulls_first(a())),
        ("asc_nulls_last", asc_nulls_last(a())),
        ("ascii", ascii(a())),
        ("asin", asin(a())),
        ("asinh", asinh(a())),
        ("assert_true", assert_true(a())),
        ("atan", atan(a())),
        ("atan2", atan2(a(), b())),
        ("atanh", atanh(a())),
        ("avg", avg(a())),
        ("base64", base64(a())),
        ("bin", bin(a())),
        ("bit_and", bit_and(a())),
        ("bit_count", bit_count(a())),
        ("bit_get", bit_get(a(), b())),
        ("bit_length", bit_length(a())),
        ("bit_or", bit_or(a())),
        ("bit_xor", bit_xor(a())),
        ("bitmap_bit_position", bitmap_bit_position(a())),
        ("bitmap_bucket_number", bitmap_bucket_number(a())),
        ("bitmap_construct_agg", bitmap_construct_agg(a())),
        ("bitmap_count", bitmap_count(a())),
        ("bitmap_or_agg", bitmap_or_agg(a())),
        ("bitwiseNOT", bitwiseNOT(a())),
        ("bitwise_not", bitwise_not(a())),
        ("bool_and", bool_and(a())),
        ("bool_or", bool_or(a())),
        ("bround", bround(a())),
        ("btrim", btrim(a())),
        ("bucket", bucket(a(), b())),
        ("call_function", call_function("x")),
        ("call_udf", call_udf("x")),
        ("cardinality", cardinality(a())),
        ("cast", cast(a(), b())),
        ("cbrt", cbrt(a())),
        ("ceil", ceil(a())),
        ("ceiling", ceiling(a())),
        ("char", char(a())),
        ("char_length", char_length(a())),
        ("character_length", character_length(a())),
        ("coalesce", coalesce()),
        ("col", col("x")),
        ("collate", collate(a(), b())),
        ("collation", collation(a())),
        ("collect_list", collect_list(a())),
        ("collect_set", collect_set(a())),
        ("column", column("x")),
        ("concat", concat()),
        ("concat_ws", concat_ws(a())),
        ("contains", contains(a(), b())),
        ("conv", conv(a(), b(), c())),
        ("convert_timezone", convert_timezone(a(), b(), c())),
        ("corr", corr(a(), b())),
        ("cos", cos(a())),
        ("cosh", cosh(a())),
        ("cot", cot(a())),
        ("count", count(a())),
        ("countDistinct", countDistinct(a())),
        ("count_distinct", count_distinct(a())),
        ("count_if", count_if(a())),
        ("count_min_sketch", count_min_sketch(a(), b(), c())),
        ("covar_pop", covar_pop(a(), b())),
        ("covar_samp", covar_samp(a(), b())),
        ("crc32", crc32(a())),
        ("create_map", create_map()),
        ("csc", csc(a())),
        ("cume_dist", cume_dist()),
        ("curdate", curdate()),
        ("current_catalog", current_catalog()),
        ("current_database", current_database()),
        ("current_date", current_date()),
        ("current_schema", current_schema()),
        ("current_timestamp", current_timestamp()),
        ("current_timezone", current_timezone()),
        ("current_user", current_user()),
        ("date_add", date_add(a(), b())),
        ("date_diff", date_diff(a(), b())),
        ("date_format", date_format(a(), b())),
        ("date_from_unix_date", date_from_unix_date(a())),
        ("date_part", date_part(a(), b())),
        ("date_sub", date_sub(a(), b())),
        ("date_trunc", date_trunc(a(), b())),
        ("dateadd", dateadd(a(), b())),
        ("datediff", datediff(a(), b())),
        ("datepart", datepart(a(), b())),
        ("day", day(a())),
        ("dayname", dayname(a())),
        ("dayofmonth", dayofmonth(a())),
        ("dayofweek", dayofweek(a())),
        ("dayofyear", dayofyear(a())),
        ("days", days(a())),
        ("decode", decode(a(), b())),
        ("degrees", degrees(a())),
        ("dense_rank", dense_rank()),
        ("desc", desc(a())),
        ("desc_nulls_first", desc_nulls_first(a())),
        ("desc_nulls_last", desc_nulls_last(a())),
        ("e", e()),
        ("element_at", element_at(a(), b())),
        ("elt", elt()),
        ("encode", encode(a(), b())),
        ("endswith", endswith(a(), b())),
        ("equal_null", equal_null(a(), b())),
        ("every", every(a())),
        ("exp", exp(a())),
        ("explode", explode(a())),
        ("explode_outer", explode_outer(a())),
        ("expm1", expm1(a())),
        ("expr", expr("x")),
        ("extract", extract(a(), b())),
        ("factorial", factorial(a())),
        ("find_in_set", find_in_set(a(), b())),
        ("first", first(a())),
        ("first_value", first_value(a())),
        ("flatten", flatten(a())),
        ("floor", floor(a())),
        ("format_number", format_number(a(), b())),
        ("format_string", format_string(a())),
        ("from_csv", from_csv(a(), b())),
        ("from_json", from_json(a(), b())),
        ("from_unixtime", from_unixtime(a())),
        ("from_utc_timestamp", from_utc_timestamp(a(), b())),
        ("from_xml", from_xml(a(), b())),
        ("get", get(a(), b())),
        ("get_json_object", get_json_object(a(), b())),
        ("getbit", getbit(a(), b())),
        ("greatest", greatest(a(), b())),
        ("grouping", grouping(a())),
        ("grouping_id", grouping_id()),
        ("hash", hash()),
        ("hex", hex(a())),
        ("histogram_numeric", histogram_numeric(a(), b())),
        ("hll_sketch_agg", hll_sketch_agg(a())),
        ("hll_sketch_estimate", hll_sketch_estimate(a())),
        ("hll_union", hll_union(a(), b())),
        ("hll_union_agg", hll_union_agg(a())),
        ("hour", hour(a())),
        ("hours", hours(a())),
        ("hypot", hypot(a(), b())),
        ("ifnull", ifnull(a(), b())),
        ("ilike", ilike(a(), b())),
        ("initcap", initcap(a())),
        ("inline", inline(a())),
        ("inline_outer", inline_outer(a())),
        ("input_file_block_length", input_file_block_length()),
        ("input_file_block_start", input_file_block_start()),
        ("input_file_name", input_file_name()),
        ("instr", instr(a(), b())),
        ("is_valid_utf8", is_valid_utf8(a())),
        ("is_variant_null", is_variant_null(a())),
        ("isnan", isnan(a())),
        ("isnotnull", isnotnull(a())),
        ("isnull", isnull(a())),
        ("java_method", java_method()),
        ("json_array_length", json_array_length(a())),
        ("json_object_keys", json_object_keys(a())),
        ("json_tuple", json_tuple(a(), b())),
        ("kurtosis", kurtosis(a())),
        ("lag", lag(a())),
        ("last", last(a())),
        ("last_day", last_day(a())),
        ("last_value", last_value(a())),
        ("lcase", lcase(a())),
        ("lead", lead(a())),
        ("least", least(a(), b())),
        ("left", left(a(), b())),
        ("length", length(a())),
        ("levenshtein", levenshtein(a(), b())),
        ("like", like(a(), b())),
        ("listagg", listagg(a())),
        ("listagg_distinct", listagg_distinct(a())),
        ("lit", lit(a())),
        ("ln", ln(a())),
        ("localtimestamp", localtimestamp()),
        ("locate", locate(a(), b())),
        ("log", log(a())),
        ("log10", log10(a())),
        ("log1p", log1p(a())),
        ("log2", log2(a())),
        ("lower", lower(a())),
        ("lpad", lpad(a(), b(), c())),
        ("ltrim", ltrim(a())),
        ("make_date", make_date(a(), b(), c())),
        ("make_dt_interval", make_dt_interval()),
        ("make_interval", make_interval()),
        ("make_valid_utf8", make_valid_utf8(a())),
        ("make_ym_interval", make_ym_interval()),
        ("map_concat", map_concat()),
        ("map_contains_key", map_contains_key(a(), b())),
        ("map_entries", map_entries(a())),
        ("map_from_arrays", map_from_arrays(a(), b())),
        ("map_from_entries", map_from_entries(a())),
        ("map_keys", map_keys(a())),
        ("map_values", map_values(a())),
        ("mask", mask(a())),
        ("max", max(a())),
        ("max_by", max_by(a(), b())),
        ("md5", md5(a())),
        ("mean", mean(a())),
        ("median", median(a())),
        ("min", min(a())),
        ("min_by", min_by(a(), b())),
        ("minute", minute(a())),
        ("mode", mode(a())),
        ("monotonically_increasing_id", monotonically_increasing_id()),
        ("month", month(a())),
        ("monthname", monthname(a())),
        ("months", months(a())),
        ("months_between", months_between(a(), b())),
        ("named_struct", named_struct()),
        ("nanvl", nanvl(a(), b())),
        ("negate", negate(a())),
        ("negative", negative(a())),
        ("next_day", next_day(a(), b())),
        ("now", now()),
        ("nth_value", nth_value(a(), b())),
        ("ntile", ntile(a())),
        ("nullif", nullif(a(), b())),
        ("nullifzero", nullifzero(a())),
        ("nvl", nvl(a(), b())),
        ("nvl2", nvl2(a(), b(), c())),
        ("octet_length", octet_length(a())),
        ("overlay", overlay(a(), b(), c())),
        ("parse_json", parse_json(a())),
        ("parse_url", parse_url(a(), b())),
        ("percent_rank", percent_rank()),
        ("percentile", percentile(a(), b())),
        ("percentile_approx", percentile_approx(a(), b())),
        ("pi", pi()),
        ("pmod", pmod(a(), b())),
        ("posexplode", posexplode(a())),
        ("posexplode_outer", posexplode_outer(a())),
        ("position", position(a(), b())),
        ("positive", positive(a())),
        ("pow", pow(a(), b())),
        ("power", power(a(), b())),
        ("printf", printf(a())),
        ("product", product(a())),
        ("quarter", quarter(a())),
        ("radians", radians(a())),
        ("raise_error", raise_error(a())),
        ("rand", rand()),
        ("randn", randn()),
        ("randstr", randstr(a())),
        ("rank", rank()),
        ("reflect", reflect()),
        ("regexp", regexp(a(), b())),
        ("regexp_count", regexp_count(a(), b())),
        ("regexp_extract", regexp_extract(a(), b(), c())),
        ("regexp_extract_all", regexp_extract_all(a(), b())),
        ("regexp_instr", regexp_instr(a(), b())),
        ("regexp_like", regexp_like(a(), b())),
        ("regexp_replace", regexp_replace(a(), b(), c())),
        ("regexp_substr", regexp_substr(a(), b())),
        ("regr_avgx", regr_avgx(a(), b())),
        ("regr_avgy", regr_avgy(a(), b())),
        ("regr_count", regr_count(a(), b())),
        ("regr_intercept", regr_intercept(a(), b())),
        ("regr_r2", regr_r2(a(), b())),
        ("regr_slope", regr_slope(a(), b())),
        ("regr_sxx", regr_sxx(a(), b())),
        ("regr_sxy", regr_sxy(a(), b())),
        ("regr_syy", regr_syy(a(), b())),
        ("repeat", repeat(a(), b())),
        ("replace", replace(a(), b())),
        ("reverse", reverse(a())),
        ("right", right(a(), b())),
        ("rint", rint(a())),
        ("rlike", rlike(a(), b())),
        ("round", round(a())),
        ("row_number", row_number()),
        ("rpad", rpad(a(), b(), c())),
        ("rtrim", rtrim(a())),
        ("schema_of_csv", schema_of_csv(a())),
        ("schema_of_json", schema_of_json(a())),
        ("schema_of_variant", schema_of_variant(a())),
        ("schema_of_variant_agg", schema_of_variant_agg(a())),
        ("schema_of_xml", schema_of_xml(a())),
        ("sec", sec(a())),
        ("second", second(a())),
        ("sentences", sentences(a())),
        ("sequence", sequence(a(), b())),
        ("session_user", session_user()),
        ("session_window", session_window(a(), b())),
        ("sha", sha(a())),
        ("sha1", sha1(a())),
        ("shiftLeft", shiftLeft(a(), b())),
        ("shiftRight", shiftRight(a(), b())),
        ("shiftRightUnsigned", shiftRightUnsigned(a(), b())),
        ("shiftleft", shiftleft(a(), b())),
        ("shiftright", shiftright(a(), b())),
        ("shiftrightunsigned", shiftrightunsigned(a(), b())),
        ("shuffle", shuffle(a())),
        ("sign", sign(a())),
        ("signum", signum(a())),
        ("sin", sin(a())),
        ("sinh", sinh(a())),
        ("size", size(a())),
        ("skewness", skewness(a())),
        ("slice", slice(a(), b(), c())),
        ("some", some(a())),
        ("sort_array", sort_array(a())),
        ("soundex", soundex(a())),
        ("spark_partition_id", spark_partition_id()),
        ("split", split(a(), b())),
        ("split_part", split_part(a(), b(), c())),
        ("sqrt", sqrt(a())),
        ("stack", stack()),
        ("startswith", startswith(a(), b())),
        ("std", std(a())),
        ("stddev", stddev(a())),
        ("stddev_pop", stddev_pop(a())),
        ("stddev_samp", stddev_samp(a())),
        ("str_to_map", str_to_map(a())),
        ("string_agg", string_agg(a())),
        ("string_agg_distinct", string_agg_distinct(a())),
        ("struct", r#struct()),
        ("substr", substr(a(), b())),
        ("substring", substring(a(), b(), c())),
        ("substring_index", substring_index(a(), b(), c())),
        ("sum", sum(a())),
        ("sumDistinct", sumDistinct(a())),
        ("sum_distinct", sum_distinct(a())),
        ("tan", tan(a())),
        ("tanh", tanh(a())),
        ("timestamp_add", timestamp_add(a(), b(), c())),
        ("timestamp_diff", timestamp_diff(a(), b(), c())),
        ("timestamp_micros", timestamp_micros(a())),
        ("timestamp_millis", timestamp_millis(a())),
        ("timestamp_seconds", timestamp_seconds(a())),
        ("toDegrees", toDegrees(a())),
        ("toRadians", toRadians(a())),
        ("to_binary", to_binary(a())),
        ("to_char", to_char(a(), b())),
        ("to_csv", to_csv(a())),
        ("to_date", to_date(a())),
        ("to_json", to_json(a())),
        ("to_number", to_number(a(), b())),
        ("to_timestamp", to_timestamp(a())),
        ("to_timestamp_ltz", to_timestamp_ltz(a())),
        ("to_timestamp_ntz", to_timestamp_ntz(a())),
        ("to_unix_timestamp", to_unix_timestamp(a())),
        ("to_utc_timestamp", to_utc_timestamp(a(), b())),
        ("to_varchar", to_varchar(a(), b())),
        ("to_variant_object", to_variant_object(a())),
        ("to_xml", to_xml(a())),
        ("translate", translate(a(), b(), c())),
        ("trim", trim(a())),
        ("trunc", trunc(a(), b())),
        ("try_add", try_add(a(), b())),
        ("try_aes_decrypt", try_aes_decrypt(a(), b())),
        ("try_avg", try_avg(a())),
        ("try_divide", try_divide(a(), b())),
        ("try_element_at", try_element_at(a(), b())),
        ("try_make_interval", try_make_interval()),
        ("try_mod", try_mod(a(), b())),
        ("try_multiply", try_multiply(a(), b())),
        ("try_parse_json", try_parse_json(a())),
        ("try_parse_url", try_parse_url(a(), b())),
        ("try_reflect", try_reflect()),
        ("try_subtract", try_subtract(a(), b())),
        ("try_sum", try_sum(a())),
        ("try_to_binary", try_to_binary(a())),
        ("try_to_number", try_to_number(a(), b())),
        ("try_to_timestamp", try_to_timestamp(a())),
        ("try_url_decode", try_url_decode(a())),
        ("try_validate_utf8", try_validate_utf8(a())),
        ("try_variant_get", try_variant_get(a(), b(), c())),
        ("typeof", r#typeof(a())),
        ("ucase", ucase(a())),
        ("unbase64", unbase64(a())),
        ("unhex", unhex(a())),
        ("uniform", uniform(a(), b())),
        ("unix_date", unix_date(a())),
        ("unix_micros", unix_micros(a())),
        ("unix_millis", unix_millis(a())),
        ("unix_seconds", unix_seconds(a())),
        ("unix_timestamp", unix_timestamp()),
        ("unwrap_udt", unwrap_udt(a())),
        ("upper", upper(a())),
        ("url_decode", url_decode(a())),
        ("url_encode", url_encode(a())),
        ("user", user()),
        ("validate_utf8", validate_utf8(a())),
        ("var_pop", var_pop(a())),
        ("var_samp", var_samp(a())),
        ("variance", variance(a())),
        ("variant_get", variant_get(a(), b(), c())),
        ("version", version()),
        ("weekday", weekday(a())),
        ("weekofyear", weekofyear(a())),
        ("when", when(a(), b())),
        ("width_bucket", width_bucket(a(), b(), c(), d())),
        ("window", window(a(), "x")),
        ("window_time", window_time(a())),
        ("xpath", xpath(a(), b())),
        ("xpath_boolean", xpath_boolean(a(), b())),
        ("xpath_double", xpath_double(a(), b())),
        ("xpath_float", xpath_float(a(), b())),
        ("xpath_int", xpath_int(a(), b())),
        ("xpath_long", xpath_long(a(), b())),
        ("xpath_number", xpath_number(a(), b())),
        ("xpath_short", xpath_short(a(), b())),
        ("xpath_string", xpath_string(a(), b())),
        ("xxhash64", xxhash64()),
        ("year", year(a())),
        ("years", years(a())),
        ("zeroifnull", zeroifnull(a())),
    ];

    let total = cases.len();
    let mut failures: Vec<String> = Vec::new();
    for (name, col) in cases {
        let expected = match goldens.get(name) {
            Some(e) => e.clone(),
            None => {
                failures.push(format!("{name}: MISSING from golden file"));
                continue;
            }
        };
        let mut actual = col.to_proto();
        normalize(&mut actual);
        if actual != expected {
            failures.push(format!(
                "{name}: MISMATCH\n  expected: {expected:?}\n  actual:   {actual:?}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} golden function cases failed:\n{}",
        failures.len(),
        total,
        failures.join("\n")
    );
    assert_eq!(total, 440, "expected exactly 440 cases, got {total}");
    println!("all {total} golden function cases passed");
}
