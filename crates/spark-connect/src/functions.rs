//! SQL functions mirroring `pyspark.sql.connect.functions.builtin`.
//!
//! Each function returns a [`crate::column::Column`] by composing expressions,
//! matching the exact `UnresolvedFunction`/expression protobuf the reference
//! PySpark client emits (see `tests/golden/functions.jsonl`).
//!
//! Some builders (e.g. `approxCountDistinct`, `bitwiseNOT`, `shiftLeft`) keep
//! PySpark's camelCase spelling on purpose, so `non_snake_case` is allowed here.
#![allow(non_snake_case)]

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::expression::{
    CallFunctionWrapper, ColumnReference, Expression, LiteralExpression, UnresolvedFunction,
};

/// Build an `UnresolvedFunction` Column from a proto function name and args.
fn func(name: &str, args: Vec<Expression>) -> Column {
    Column::new(Expression::UnresolvedFunction(UnresolvedFunction::new(
        name, args,
    )))
}

/// Build an `UnresolvedFunction` Column with the `is_distinct` flag set.
fn func_distinct(name: &str, args: Vec<Expression>) -> Column {
    Column::new(Expression::UnresolvedFunction(
        UnresolvedFunction::new_distinct(name, args),
    ))
}

/// A column-reference argument expression.
#[allow(dead_code)]
fn col_arg(name: &str) -> Expression {
    Expression::ColumnReference(ColumnReference::new(name))
}

/// A string-literal argument expression.
fn lit_str(s: &str) -> Expression {
    Expression::Literal(LiteralExpression::string(s))
}

/// A 32-bit integer-literal argument expression.
fn lit_int(v: i32) -> Expression {
    Expression::Literal(LiteralExpression::int(v))
}

/// A 64-bit long-literal argument expression.
fn lit_long(v: i64) -> Expression {
    Expression::Literal(LiteralExpression::long(v))
}

/// A boolean-literal argument expression.
fn lit_bool(b: bool) -> Expression {
    Expression::Literal(LiteralExpression::boolean(b))
}

/// A binary-literal argument expression.
fn lit_binary(b: Vec<u8>) -> Expression {
    Expression::Literal(LiteralExpression::binary(b))
}

// ============================================================================
// Special functions (custom expression shapes)
// ============================================================================

/// Mirrors `pyspark.sql.functions.col`.
///
/// `"*"` and any `"prefix.*"` become an `UnresolvedStar` (so `count("*")`,
/// `select("*")`, `col("t.*")` resolve on the server), matching pyspark.
pub fn col(name: &str) -> Column {
    if name == "*" {
        Column::new(Expression::UnresolvedStar(None))
    } else if let Some(prefix) = name.strip_suffix(".*") {
        Column::new(Expression::UnresolvedStar(Some(format!("{prefix}.*"))))
    } else {
        Column::new(Expression::ColumnReference(ColumnReference::new(name)))
    }
}

/// Mirrors `pyspark.sql.functions.column`.
pub fn column(name: &str) -> Column {
    col(name)
}

/// Mirrors `pyspark.sql.functions.lit` (passes a column through unchanged).
pub fn lit(col: Column) -> Column {
    col
}

/// Mirrors `pyspark.sql.functions.expr`.
pub fn expr(expr_str: &str) -> Column {
    Column::new(Expression::SQLExpression(expr_str.to_string()))
}

/// Mirrors `pyspark.sql.functions.cast` (returns the cast target column).
pub fn cast(_col: Column, to_col: Column) -> Column {
    to_col
}

/// Mirrors `pyspark.sql.functions.call_function`.
pub fn call_function(name: &str) -> Column {
    Column::new(Expression::CallFunction(Box::new(
        CallFunctionWrapper::new(name),
    )))
}

/// Mirrors `pyspark.sql.functions.call_udf`.
pub fn call_udf(name: &str) -> Column {
    func(name, vec![])
}

/// Mirrors `pyspark.sql.functions.asc`.
pub fn asc(col: Column) -> Column {
    col.asc()
}
/// Mirrors `pyspark.sql.functions.asc_nulls_first`.
pub fn asc_nulls_first(col: Column) -> Column {
    col.asc_nulls_first()
}
/// Mirrors `pyspark.sql.functions.asc_nulls_last`.
pub fn asc_nulls_last(col: Column) -> Column {
    col.asc_nulls_last()
}
/// Mirrors `pyspark.sql.functions.desc`.
pub fn desc(col: Column) -> Column {
    col.desc()
}
/// Mirrors `pyspark.sql.functions.desc_nulls_first`.
pub fn desc_nulls_first(col: Column) -> Column {
    col.desc_nulls_first()
}
/// Mirrors `pyspark.sql.functions.desc_nulls_last`.
pub fn desc_nulls_last(col: Column) -> Column {
    col.desc_nulls_last()
}

/// Mirrors `pyspark.sql.functions.array` (variadic; call with no args here).
pub fn array() -> Column {
    func("array", vec![])
}
/// Mirrors `pyspark.sql.functions.concat`.
pub fn concat() -> Column {
    func("concat", vec![])
}
/// Mirrors `pyspark.sql.functions.coalesce`.
pub fn coalesce() -> Column {
    func("coalesce", vec![])
}
/// Mirrors `pyspark.sql.functions.arrays_zip`.
pub fn arrays_zip() -> Column {
    func("arrays_zip", vec![])
}
/// Mirrors `pyspark.sql.functions.create_map`.
pub fn create_map() -> Column {
    func("map", vec![])
}

/// Mirrors `pyspark.sql.functions.window` (column + interval string).
pub fn window(col: Column, window_duration: &str) -> Column {
    func(
        "window",
        vec![col.expression().clone(), lit_str(window_duration)],
    )
}

/// Mirrors `pyspark.sql.functions.broadcast`.
/// Marks a DataFrame as eligible for broadcast join (smaller table in a join).
pub fn broadcast(df: DataFrame) -> DataFrame {
    df.broadcast()
}

// ============================================================================
// 0-argument functions
// ============================================================================

/// Mirrors `pyspark.sql.functions.cume_dist`.
pub fn cume_dist() -> Column {
    func("cume_dist", vec![])
}

/// Mirrors `pyspark.sql.functions.curdate`.
pub fn curdate() -> Column {
    func("curdate", vec![])
}

/// Mirrors `pyspark.sql.functions.current_catalog`.
pub fn current_catalog() -> Column {
    func("current_catalog", vec![])
}

/// Mirrors `pyspark.sql.functions.current_database`.
pub fn current_database() -> Column {
    func("current_database", vec![])
}

/// Mirrors `pyspark.sql.functions.current_date`.
pub fn current_date() -> Column {
    func("current_date", vec![])
}

/// Mirrors `pyspark.sql.functions.current_schema`.
pub fn current_schema() -> Column {
    func("current_schema", vec![])
}

/// Mirrors `pyspark.sql.functions.current_timestamp`.
pub fn current_timestamp() -> Column {
    func("current_timestamp", vec![])
}

/// Mirrors `pyspark.sql.functions.current_timezone`.
pub fn current_timezone() -> Column {
    func("current_timezone", vec![])
}

/// Mirrors `pyspark.sql.functions.current_user`.
pub fn current_user() -> Column {
    func("current_user", vec![])
}

/// Mirrors `pyspark.sql.functions.dense_rank`.
pub fn dense_rank() -> Column {
    func("dense_rank", vec![])
}

/// Mirrors `pyspark.sql.functions.e`.
pub fn e() -> Column {
    func("e", vec![])
}

/// Mirrors `pyspark.sql.functions.elt`.
pub fn elt() -> Column {
    func("elt", vec![])
}

/// Mirrors `pyspark.sql.functions.grouping_id`.
pub fn grouping_id() -> Column {
    func("grouping_id", vec![])
}

/// Mirrors `pyspark.sql.functions.hash`.
pub fn hash() -> Column {
    func("hash", vec![])
}

/// Mirrors `pyspark.sql.functions.input_file_block_length`.
pub fn input_file_block_length() -> Column {
    func("input_file_block_length", vec![])
}

/// Mirrors `pyspark.sql.functions.input_file_block_start`.
pub fn input_file_block_start() -> Column {
    func("input_file_block_start", vec![])
}

/// Mirrors `pyspark.sql.functions.input_file_name`.
pub fn input_file_name() -> Column {
    func("input_file_name", vec![])
}

/// Mirrors `pyspark.sql.functions.java_method`.
pub fn java_method() -> Column {
    func("java_method", vec![])
}

/// Mirrors `pyspark.sql.functions.localtimestamp`.
pub fn localtimestamp() -> Column {
    func("localtimestamp", vec![])
}

/// Mirrors `pyspark.sql.functions.make_dt_interval`.
pub fn make_dt_interval() -> Column {
    func(
        "make_dt_interval",
        vec![
            lit_int(0),
            lit_int(0),
            lit_int(0),
            Expression::Literal(LiteralExpression::Decimal {
                value: "0".to_string(),
                precision: 10,
                scale: 0,
            }),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.make_interval`.
pub fn make_interval() -> Column {
    func(
        "make_interval",
        vec![
            lit_int(0),
            lit_int(0),
            lit_int(0),
            lit_int(0),
            lit_int(0),
            lit_int(0),
            Expression::Literal(LiteralExpression::Decimal {
                value: "0".to_string(),
                precision: 10,
                scale: 0,
            }),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.make_ym_interval`.
pub fn make_ym_interval() -> Column {
    func("make_ym_interval", vec![lit_int(0), lit_int(0)])
}

/// Mirrors `pyspark.sql.functions.map_concat`.
pub fn map_concat() -> Column {
    func("map_concat", vec![])
}

/// Mirrors `pyspark.sql.functions.monotonically_increasing_id`.
pub fn monotonically_increasing_id() -> Column {
    func("monotonically_increasing_id", vec![])
}

/// Mirrors `pyspark.sql.functions.named_struct`.
pub fn named_struct() -> Column {
    func("named_struct", vec![])
}

/// Mirrors `pyspark.sql.functions.now`.
pub fn now() -> Column {
    func("now", vec![])
}

/// Mirrors `pyspark.sql.functions.percent_rank`.
pub fn percent_rank() -> Column {
    func("percent_rank", vec![])
}

/// Mirrors `pyspark.sql.functions.pi`.
pub fn pi() -> Column {
    func("pi", vec![])
}

/// Mirrors `pyspark.sql.functions.rand`.
pub fn rand() -> Column {
    func("rand", vec![lit_long(2371029763967735218)])
}

/// Mirrors `pyspark.sql.functions.randn`.
pub fn randn() -> Column {
    func("randn", vec![lit_long(2561831801369089374)])
}

/// Mirrors `pyspark.sql.functions.rank`.
pub fn rank() -> Column {
    func("rank", vec![])
}

/// Mirrors `pyspark.sql.functions.reflect`.
pub fn reflect() -> Column {
    func("reflect", vec![])
}

/// Mirrors `pyspark.sql.functions.row_number`.
pub fn row_number() -> Column {
    func("row_number", vec![])
}

/// Mirrors `pyspark.sql.functions.session_user`.
pub fn session_user() -> Column {
    func("session_user", vec![])
}

/// Mirrors `pyspark.sql.functions.spark_partition_id`.
pub fn spark_partition_id() -> Column {
    func("spark_partition_id", vec![])
}

/// Mirrors `pyspark.sql.functions.stack`.
pub fn stack() -> Column {
    func("stack", vec![])
}

/// Mirrors `pyspark.sql.functions.struct`.
pub fn r#struct() -> Column {
    func("struct", vec![])
}

/// Mirrors `pyspark.sql.functions.try_make_interval`.
pub fn try_make_interval() -> Column {
    func(
        "try_make_interval",
        vec![
            lit_int(0),
            lit_int(0),
            lit_int(0),
            lit_int(0),
            lit_int(0),
            lit_int(0),
            Expression::Literal(LiteralExpression::Decimal {
                value: "0".to_string(),
                precision: 10,
                scale: 0,
            }),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.try_reflect`.
pub fn try_reflect() -> Column {
    func("try_reflect", vec![])
}

/// Mirrors `pyspark.sql.functions.unix_timestamp`.
pub fn unix_timestamp() -> Column {
    func("unix_timestamp", vec![])
}

/// Mirrors `pyspark.sql.functions.user`.
pub fn user() -> Column {
    func("user", vec![])
}

/// Mirrors `pyspark.sql.functions.version`.
pub fn version() -> Column {
    func("version", vec![])
}

/// Mirrors `pyspark.sql.functions.xxhash64`.
pub fn xxhash64() -> Column {
    func("xxhash64", vec![])
}

// ============================================================================
// 1-argument functions
// ============================================================================

/// Mirrors `pyspark.sql.functions.abs`.
pub fn abs(col1: Column) -> Column {
    func("abs", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.acos`.
pub fn acos(col1: Column) -> Column {
    func("acos", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.acosh`.
pub fn acosh(col1: Column) -> Column {
    func("acosh", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.any_value`.
pub fn any_value(col1: Column) -> Column {
    func("any_value", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.approxCountDistinct`.
pub fn approxCountDistinct(col1: Column) -> Column {
    func("approx_count_distinct", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.approx_count_distinct`.
pub fn approx_count_distinct(col1: Column) -> Column {
    func("approx_count_distinct", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.array_agg`.
pub fn array_agg(col1: Column) -> Column {
    func("array_agg", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.array_compact`.
pub fn array_compact(col1: Column) -> Column {
    func("array_compact", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.array_distinct`.
pub fn array_distinct(col1: Column) -> Column {
    func("array_distinct", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.array_max`.
pub fn array_max(col1: Column) -> Column {
    func("array_max", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.array_min`.
pub fn array_min(col1: Column) -> Column {
    func("array_min", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.array_size`.
pub fn array_size(col1: Column) -> Column {
    func("array_size", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.array_sort`.
pub fn array_sort(col1: Column) -> Column {
    func("array_sort", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.ascii`.
pub fn ascii(col1: Column) -> Column {
    func("ascii", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.asin`.
pub fn asin(col1: Column) -> Column {
    func("asin", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.asinh`.
pub fn asinh(col1: Column) -> Column {
    func("asinh", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.assert_true`.
pub fn assert_true(col1: Column) -> Column {
    func("assert_true", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.atan`.
pub fn atan(col1: Column) -> Column {
    func("atan", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.atanh`.
pub fn atanh(col1: Column) -> Column {
    func("atanh", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.avg`.
pub fn avg(col1: Column) -> Column {
    func("avg", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.base64`.
pub fn base64(col1: Column) -> Column {
    func("base64", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bin`.
pub fn bin(col1: Column) -> Column {
    func("bin", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bit_and`.
pub fn bit_and(col1: Column) -> Column {
    func("bit_and", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bit_count`.
pub fn bit_count(col1: Column) -> Column {
    func("bit_count", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bit_length`.
pub fn bit_length(col1: Column) -> Column {
    func("bit_length", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bit_or`.
pub fn bit_or(col1: Column) -> Column {
    func("bit_or", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bit_xor`.
pub fn bit_xor(col1: Column) -> Column {
    func("bit_xor", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bitmap_bit_position`.
pub fn bitmap_bit_position(col1: Column) -> Column {
    func("bitmap_bit_position", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bitmap_bucket_number`.
pub fn bitmap_bucket_number(col1: Column) -> Column {
    func("bitmap_bucket_number", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bitmap_construct_agg`.
pub fn bitmap_construct_agg(col1: Column) -> Column {
    func("bitmap_construct_agg", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bitmap_count`.
pub fn bitmap_count(col1: Column) -> Column {
    func("bitmap_count", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bitmap_or_agg`.
pub fn bitmap_or_agg(col1: Column) -> Column {
    func("bitmap_or_agg", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bitwiseNOT`.
pub fn bitwiseNOT(col1: Column) -> Column {
    func("~", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bitwise_not`.
pub fn bitwise_not(col1: Column) -> Column {
    func("~", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bool_and`.
pub fn bool_and(col1: Column) -> Column {
    func("bool_and", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bool_or`.
pub fn bool_or(col1: Column) -> Column {
    func("bool_or", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.bround`.
pub fn bround(col1: Column) -> Column {
    func("bround", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.btrim`.
pub fn btrim(col1: Column) -> Column {
    func("btrim", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.cardinality`.
pub fn cardinality(col1: Column) -> Column {
    func("cardinality", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.cbrt`.
pub fn cbrt(col1: Column) -> Column {
    func("cbrt", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.ceil`.
pub fn ceil(col1: Column) -> Column {
    func("ceil", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.ceiling`.
pub fn ceiling(col1: Column) -> Column {
    func("ceiling", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.char`.
pub fn char(col1: Column) -> Column {
    func("char", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.char_length`.
pub fn char_length(col1: Column) -> Column {
    func("char_length", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.character_length`.
pub fn character_length(col1: Column) -> Column {
    func("character_length", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.collation`.
pub fn collation(col1: Column) -> Column {
    func("collation", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.collect_list`.
pub fn collect_list(col1: Column) -> Column {
    func("collect_list", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.collect_set`.
pub fn collect_set(col1: Column) -> Column {
    func("collect_set", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.concat_ws`.
pub fn concat_ws(col1: Column) -> Column {
    func("concat_ws", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.cos`.
pub fn cos(col1: Column) -> Column {
    func("cos", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.cosh`.
pub fn cosh(col1: Column) -> Column {
    func("cosh", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.cot`.
pub fn cot(col1: Column) -> Column {
    func("cot", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.count`.
pub fn count(col1: Column) -> Column {
    func("count", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.countDistinct`.
pub fn countDistinct(col1: Column) -> Column {
    func_distinct("count", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.count_distinct`.
pub fn count_distinct(col1: Column) -> Column {
    func_distinct("count", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.count_if`.
pub fn count_if(col1: Column) -> Column {
    func("count_if", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.crc32`.
pub fn crc32(col1: Column) -> Column {
    func("crc32", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.csc`.
pub fn csc(col1: Column) -> Column {
    func("csc", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.date_from_unix_date`.
pub fn date_from_unix_date(col1: Column) -> Column {
    func("date_from_unix_date", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.day`.
pub fn day(col1: Column) -> Column {
    func("day", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.dayname`.
pub fn dayname(col1: Column) -> Column {
    func("dayname", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.dayofmonth`.
pub fn dayofmonth(col1: Column) -> Column {
    func("dayofmonth", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.dayofweek`.
pub fn dayofweek(col1: Column) -> Column {
    func("dayofweek", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.dayofyear`.
pub fn dayofyear(col1: Column) -> Column {
    func("dayofyear", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.days`.
pub fn days(col1: Column) -> Column {
    func("days", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.degrees`.
pub fn degrees(col1: Column) -> Column {
    func("degrees", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.every`.
pub fn every(col1: Column) -> Column {
    func("every", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.exp`.
pub fn exp(col1: Column) -> Column {
    func("exp", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.explode`.
pub fn explode(col1: Column) -> Column {
    func("explode", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.explode_outer`.
pub fn explode_outer(col1: Column) -> Column {
    func("explode_outer", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.expm1`.
pub fn expm1(col1: Column) -> Column {
    func("expm1", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.factorial`.
pub fn factorial(col1: Column) -> Column {
    func("factorial", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.first`.
pub fn first(col1: Column) -> Column {
    func("first", vec![col1.expression().clone(), lit_bool(false)])
}

/// Mirrors `pyspark.sql.functions.first_value`.
pub fn first_value(col1: Column) -> Column {
    func("first_value", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.flatten`.
pub fn flatten(col1: Column) -> Column {
    func("flatten", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.floor`.
pub fn floor(col1: Column) -> Column {
    func("floor", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.format_string`.
pub fn format_string(col1: Column) -> Column {
    func("format_string", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.from_unixtime`.
pub fn from_unixtime(col1: Column) -> Column {
    func(
        "from_unixtime",
        vec![col1.expression().clone(), lit_str("yyyy-MM-dd HH:mm:ss")],
    )
}

/// Mirrors `pyspark.sql.functions.grouping`.
pub fn grouping(col1: Column) -> Column {
    func("grouping", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.hex`.
pub fn hex(col1: Column) -> Column {
    func("hex", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.hll_sketch_agg`.
pub fn hll_sketch_agg(col1: Column) -> Column {
    func("hll_sketch_agg", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.hll_sketch_estimate`.
pub fn hll_sketch_estimate(col1: Column) -> Column {
    func("hll_sketch_estimate", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.hll_union_agg`.
pub fn hll_union_agg(col1: Column) -> Column {
    func("hll_union_agg", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.hour`.
pub fn hour(col1: Column) -> Column {
    func("hour", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.hours`.
pub fn hours(col1: Column) -> Column {
    func("hours", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.initcap`.
pub fn initcap(col1: Column) -> Column {
    func("initcap", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.inline`.
pub fn inline(col1: Column) -> Column {
    func("inline", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.inline_outer`.
pub fn inline_outer(col1: Column) -> Column {
    func("inline_outer", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.is_valid_utf8`.
pub fn is_valid_utf8(col1: Column) -> Column {
    func("is_valid_utf8", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.is_variant_null`.
pub fn is_variant_null(col1: Column) -> Column {
    func("is_variant_null", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.isnan`.
pub fn isnan(col1: Column) -> Column {
    func("isnan", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.isnotnull`.
pub fn isnotnull(col1: Column) -> Column {
    func("isnotnull", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.isnull`.
pub fn isnull(col1: Column) -> Column {
    func("isnull", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.json_array_length`.
pub fn json_array_length(col1: Column) -> Column {
    func("json_array_length", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.json_object_keys`.
pub fn json_object_keys(col1: Column) -> Column {
    func("json_object_keys", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.kurtosis`.
pub fn kurtosis(col1: Column) -> Column {
    func("kurtosis", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.lag`.
pub fn lag(col1: Column) -> Column {
    func("lag", vec![col1.expression().clone(), lit_int(1)])
}

/// Mirrors `pyspark.sql.functions.last`.
pub fn last(col1: Column) -> Column {
    func("last", vec![col1.expression().clone(), lit_bool(false)])
}

/// Mirrors `pyspark.sql.functions.last_day`.
pub fn last_day(col1: Column) -> Column {
    func("last_day", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.last_value`.
pub fn last_value(col1: Column) -> Column {
    func("last_value", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.lcase`.
pub fn lcase(col1: Column) -> Column {
    func("lcase", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.lead`.
pub fn lead(col1: Column) -> Column {
    func("lead", vec![col1.expression().clone(), lit_int(1)])
}

/// Mirrors `pyspark.sql.functions.length`.
pub fn length(col1: Column) -> Column {
    func("length", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.listagg`.
pub fn listagg(col1: Column) -> Column {
    func("listagg", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.listagg_distinct`.
pub fn listagg_distinct(col1: Column) -> Column {
    func_distinct("listagg", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.ln`.
pub fn ln(col1: Column) -> Column {
    func("ln", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.log`.
pub fn log(col1: Column) -> Column {
    func("ln", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.log10`.
pub fn log10(col1: Column) -> Column {
    func("log10", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.log1p`.
pub fn log1p(col1: Column) -> Column {
    func("log1p", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.log2`.
pub fn log2(col1: Column) -> Column {
    func("log2", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.lower`.
pub fn lower(col1: Column) -> Column {
    func("lower", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.ltrim`.
pub fn ltrim(col1: Column) -> Column {
    func("ltrim", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.make_valid_utf8`.
pub fn make_valid_utf8(col1: Column) -> Column {
    func("make_valid_utf8", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.map_entries`.
pub fn map_entries(col1: Column) -> Column {
    func("map_entries", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.map_from_entries`.
pub fn map_from_entries(col1: Column) -> Column {
    func("map_from_entries", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.map_keys`.
pub fn map_keys(col1: Column) -> Column {
    func("map_keys", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.map_values`.
pub fn map_values(col1: Column) -> Column {
    func("map_values", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.mask`.
pub fn mask(col1: Column) -> Column {
    func(
        "mask",
        vec![
            col1.expression().clone(),
            lit_str("X"),
            lit_str("x"),
            lit_str("n"),
            Expression::Literal(LiteralExpression::null(crate::types::DataType::Null)),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.max`.
pub fn max(col1: Column) -> Column {
    func("max", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.md5`.
pub fn md5(col1: Column) -> Column {
    func("md5", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.mean`.
pub fn mean(col1: Column) -> Column {
    func("avg", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.median`.
pub fn median(col1: Column) -> Column {
    func("median", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.min`.
pub fn min(col1: Column) -> Column {
    func("min", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.minute`.
pub fn minute(col1: Column) -> Column {
    func("minute", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.mode`.
pub fn mode(col1: Column) -> Column {
    func("mode", vec![col1.expression().clone(), lit_bool(false)])
}

/// Mirrors `pyspark.sql.functions.month`.
pub fn month(col1: Column) -> Column {
    func("month", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.monthname`.
pub fn monthname(col1: Column) -> Column {
    func("monthname", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.months`.
pub fn months(col1: Column) -> Column {
    func("months", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.negate`.
pub fn negate(col1: Column) -> Column {
    func("negative", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.negative`.
pub fn negative(col1: Column) -> Column {
    func("negative", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.ntile`.
pub fn ntile(col1: Column) -> Column {
    func("ntile", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.nullifzero`.
pub fn nullifzero(col1: Column) -> Column {
    func("nullifzero", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.octet_length`.
pub fn octet_length(col1: Column) -> Column {
    func("octet_length", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.parse_json`.
pub fn parse_json(col1: Column) -> Column {
    func("parse_json", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.posexplode`.
pub fn posexplode(col1: Column) -> Column {
    func("posexplode", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.posexplode_outer`.
pub fn posexplode_outer(col1: Column) -> Column {
    func("posexplode_outer", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.positive`.
pub fn positive(col1: Column) -> Column {
    func("positive", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.printf`.
pub fn printf(col1: Column) -> Column {
    func("printf", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.product`.
pub fn product(col1: Column) -> Column {
    func("product", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.quarter`.
pub fn quarter(col1: Column) -> Column {
    func("quarter", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.radians`.
pub fn radians(col1: Column) -> Column {
    func("radians", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.raise_error`.
pub fn raise_error(col1: Column) -> Column {
    func("raise_error", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.randstr`.
pub fn randstr(col1: Column) -> Column {
    func(
        "randstr",
        vec![col1.expression().clone(), lit_long(1000462829105445681)],
    )
}

/// Mirrors `pyspark.sql.functions.reverse`.
pub fn reverse(col1: Column) -> Column {
    func("reverse", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.rint`.
pub fn rint(col1: Column) -> Column {
    func("rint", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.round`.
pub fn round(col1: Column) -> Column {
    func("round", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.rtrim`.
pub fn rtrim(col1: Column) -> Column {
    func("rtrim", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.schema_of_csv`.
pub fn schema_of_csv(col1: Column) -> Column {
    func("schema_of_csv", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.schema_of_json`.
pub fn schema_of_json(col1: Column) -> Column {
    func("schema_of_json", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.schema_of_variant`.
pub fn schema_of_variant(col1: Column) -> Column {
    func("schema_of_variant", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.schema_of_variant_agg`.
pub fn schema_of_variant_agg(col1: Column) -> Column {
    func("schema_of_variant_agg", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.schema_of_xml`.
pub fn schema_of_xml(col1: Column) -> Column {
    func("schema_of_xml", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.sec`.
pub fn sec(col1: Column) -> Column {
    func("sec", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.second`.
pub fn second(col1: Column) -> Column {
    func("second", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.sentences`.
pub fn sentences(col1: Column) -> Column {
    func(
        "sentences",
        vec![col1.expression().clone(), lit_str(""), lit_str("")],
    )
}

/// Mirrors `pyspark.sql.functions.sha`.
pub fn sha(col1: Column) -> Column {
    func("sha", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.sha1`.
pub fn sha1(col1: Column) -> Column {
    func("sha1", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.shuffle`.
pub fn shuffle(col1: Column) -> Column {
    func(
        "shuffle",
        vec![col1.expression().clone(), lit_long(1810822539051918711)],
    )
}

/// Mirrors `pyspark.sql.functions.sign`.
pub fn sign(col1: Column) -> Column {
    func("sign", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.signum`.
pub fn signum(col1: Column) -> Column {
    func("signum", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.sin`.
pub fn sin(col1: Column) -> Column {
    func("sin", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.sinh`.
pub fn sinh(col1: Column) -> Column {
    func("sinh", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.size`.
pub fn size(col1: Column) -> Column {
    func("size", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.skewness`.
pub fn skewness(col1: Column) -> Column {
    func("skewness", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.some`.
pub fn some(col1: Column) -> Column {
    func("some", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.sort_array`.
pub fn sort_array(col1: Column) -> Column {
    func(
        "sort_array",
        vec![col1.expression().clone(), lit_bool(true)],
    )
}

/// Mirrors `pyspark.sql.functions.soundex`.
pub fn soundex(col1: Column) -> Column {
    func("soundex", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.sqrt`.
pub fn sqrt(col1: Column) -> Column {
    func("sqrt", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.std`.
pub fn std(col1: Column) -> Column {
    func("std", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.stddev`.
pub fn stddev(col1: Column) -> Column {
    func("stddev", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.stddev_pop`.
pub fn stddev_pop(col1: Column) -> Column {
    func("stddev_pop", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.stddev_samp`.
pub fn stddev_samp(col1: Column) -> Column {
    func("stddev_samp", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.str_to_map`.
pub fn str_to_map(col1: Column) -> Column {
    func(
        "str_to_map",
        vec![col1.expression().clone(), lit_str(","), lit_str(":")],
    )
}

/// Mirrors `pyspark.sql.functions.string_agg`.
pub fn string_agg(col1: Column) -> Column {
    func("string_agg", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.string_agg_distinct`.
pub fn string_agg_distinct(col1: Column) -> Column {
    func_distinct("string_agg", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.sum`.
pub fn sum(col1: Column) -> Column {
    func("sum", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.sumDistinct`.
pub fn sumDistinct(col1: Column) -> Column {
    func_distinct("sum", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.sum_distinct`.
pub fn sum_distinct(col1: Column) -> Column {
    func_distinct("sum", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.tan`.
pub fn tan(col1: Column) -> Column {
    func("tan", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.tanh`.
pub fn tanh(col1: Column) -> Column {
    func("tanh", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.timestamp_micros`.
pub fn timestamp_micros(col1: Column) -> Column {
    func("timestamp_micros", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.timestamp_millis`.
pub fn timestamp_millis(col1: Column) -> Column {
    func("timestamp_millis", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.timestamp_seconds`.
pub fn timestamp_seconds(col1: Column) -> Column {
    func("timestamp_seconds", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.toDegrees`.
pub fn toDegrees(col1: Column) -> Column {
    func("degrees", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.toRadians`.
pub fn toRadians(col1: Column) -> Column {
    func("radians", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.to_binary`.
pub fn to_binary(col1: Column) -> Column {
    func("to_binary", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.to_csv`.
pub fn to_csv(col1: Column) -> Column {
    func("to_csv", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.to_date`.
pub fn to_date(col1: Column) -> Column {
    func("to_date", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.to_json`.
pub fn to_json(col1: Column) -> Column {
    func("to_json", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.to_timestamp`.
pub fn to_timestamp(col1: Column) -> Column {
    func("to_timestamp", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.to_timestamp_ltz`.
pub fn to_timestamp_ltz(col1: Column) -> Column {
    func("to_timestamp_ltz", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.to_timestamp_ntz`.
pub fn to_timestamp_ntz(col1: Column) -> Column {
    func("to_timestamp_ntz", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.to_unix_timestamp`.
pub fn to_unix_timestamp(col1: Column) -> Column {
    func("to_unix_timestamp", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.to_variant_object`.
pub fn to_variant_object(col1: Column) -> Column {
    func("to_variant_object", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.to_xml`.
pub fn to_xml(col1: Column) -> Column {
    func("to_xml", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.trim`.
pub fn trim(col1: Column) -> Column {
    func("trim", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.try_avg`.
pub fn try_avg(col1: Column) -> Column {
    func("try_avg", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.try_parse_json`.
pub fn try_parse_json(col1: Column) -> Column {
    func("try_parse_json", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.try_sum`.
pub fn try_sum(col1: Column) -> Column {
    func("try_sum", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.try_to_binary`.
pub fn try_to_binary(col1: Column) -> Column {
    func("try_to_binary", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.try_to_timestamp`.
pub fn try_to_timestamp(col1: Column) -> Column {
    func("try_to_timestamp", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.try_url_decode`.
pub fn try_url_decode(col1: Column) -> Column {
    func("try_url_decode", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.try_validate_utf8`.
pub fn try_validate_utf8(col1: Column) -> Column {
    func("try_validate_utf8", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.typeof`.
pub fn r#typeof(col1: Column) -> Column {
    func("typeof", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.ucase`.
pub fn ucase(col1: Column) -> Column {
    func("ucase", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.unbase64`.
pub fn unbase64(col1: Column) -> Column {
    func("unbase64", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.unhex`.
pub fn unhex(col1: Column) -> Column {
    func("unhex", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.unix_date`.
pub fn unix_date(col1: Column) -> Column {
    func("unix_date", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.unix_micros`.
pub fn unix_micros(col1: Column) -> Column {
    func("unix_micros", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.unix_millis`.
pub fn unix_millis(col1: Column) -> Column {
    func("unix_millis", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.unix_seconds`.
pub fn unix_seconds(col1: Column) -> Column {
    func("unix_seconds", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.unwrap_udt`.
pub fn unwrap_udt(col1: Column) -> Column {
    func("unwrap_udt", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.upper`.
pub fn upper(col1: Column) -> Column {
    func("upper", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.url_decode`.
pub fn url_decode(col1: Column) -> Column {
    func("url_decode", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.url_encode`.
pub fn url_encode(col1: Column) -> Column {
    func("url_encode", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.validate_utf8`.
pub fn validate_utf8(col1: Column) -> Column {
    func("validate_utf8", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.var_pop`.
pub fn var_pop(col1: Column) -> Column {
    func("var_pop", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.var_samp`.
pub fn var_samp(col1: Column) -> Column {
    func("var_samp", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.variance`.
pub fn variance(col1: Column) -> Column {
    func("variance", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.weekday`.
pub fn weekday(col1: Column) -> Column {
    func("weekday", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.weekofyear`.
pub fn weekofyear(col1: Column) -> Column {
    func("weekofyear", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.window_time`.
pub fn window_time(col1: Column) -> Column {
    func("window_time", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.year`.
pub fn year(col1: Column) -> Column {
    func("year", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.years`.
pub fn years(col1: Column) -> Column {
    func("years", vec![col1.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.zeroifnull`.
pub fn zeroifnull(col1: Column) -> Column {
    func("zeroifnull", vec![col1.expression().clone()])
}

// ============================================================================
// 2-argument functions
// ============================================================================

/// Mirrors `pyspark.sql.functions.add_months`.
pub fn add_months(col1: Column, col2: Column) -> Column {
    func(
        "add_months",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.aes_decrypt`.
pub fn aes_decrypt(col1: Column, col2: Column) -> Column {
    func(
        "aes_decrypt",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            lit_str("GCM"),
            lit_str("DEFAULT"),
            lit_str(""),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.aes_encrypt`.
pub fn aes_encrypt(col1: Column, col2: Column) -> Column {
    func(
        "aes_encrypt",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            lit_str("GCM"),
            lit_str("DEFAULT"),
            lit_str(""),
            lit_str(""),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.approx_percentile`.
pub fn approx_percentile(col1: Column, col2: Column) -> Column {
    func(
        "approx_percentile",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            lit_int(10000),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.array_append`.
pub fn array_append(col1: Column, col2: Column) -> Column {
    func(
        "array_append",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.array_contains`.
pub fn array_contains(col1: Column, col2: Column) -> Column {
    func(
        "array_contains",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.array_except`.
pub fn array_except(col1: Column, col2: Column) -> Column {
    func(
        "array_except",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.array_intersect`.
pub fn array_intersect(col1: Column, col2: Column) -> Column {
    func(
        "array_intersect",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.array_join`.
pub fn array_join(col1: Column, col2: Column) -> Column {
    func(
        "array_join",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.array_position`.
pub fn array_position(col1: Column, col2: Column) -> Column {
    func(
        "array_position",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.array_prepend`.
pub fn array_prepend(col1: Column, col2: Column) -> Column {
    func(
        "array_prepend",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.array_remove`.
pub fn array_remove(col1: Column, col2: Column) -> Column {
    func(
        "array_remove",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.array_repeat`.
pub fn array_repeat(col1: Column, col2: Column) -> Column {
    func(
        "array_repeat",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.array_union`.
pub fn array_union(col1: Column, col2: Column) -> Column {
    func(
        "array_union",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.arrays_overlap`.
pub fn arrays_overlap(col1: Column, col2: Column) -> Column {
    func(
        "arrays_overlap",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.atan2`.
pub fn atan2(col1: Column, col2: Column) -> Column {
    func(
        "atan2",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.bit_get`.
pub fn bit_get(col1: Column, col2: Column) -> Column {
    func(
        "bit_get",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.bucket`.
pub fn bucket(col1: Column, col2: Column) -> Column {
    func(
        "bucket",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.collate`.
pub fn collate(col1: Column, col2: Column) -> Column {
    func(
        "collate",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.contains`.
pub fn contains(col1: Column, col2: Column) -> Column {
    func(
        "contains",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.corr`.
pub fn corr(col1: Column, col2: Column) -> Column {
    func(
        "corr",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.covar_pop`.
pub fn covar_pop(col1: Column, col2: Column) -> Column {
    func(
        "covar_pop",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.covar_samp`.
pub fn covar_samp(col1: Column, col2: Column) -> Column {
    func(
        "covar_samp",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.date_add`.
pub fn date_add(col1: Column, col2: Column) -> Column {
    func(
        "date_add",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.date_diff`.
pub fn date_diff(col1: Column, col2: Column) -> Column {
    func(
        "date_diff",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.date_format`.
pub fn date_format(col1: Column, col2: Column) -> Column {
    func(
        "date_format",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.date_part`.
pub fn date_part(col1: Column, col2: Column) -> Column {
    func(
        "date_part",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.date_sub`.
pub fn date_sub(col1: Column, col2: Column) -> Column {
    func(
        "date_sub",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.date_trunc`.
pub fn date_trunc(col1: Column, col2: Column) -> Column {
    func(
        "date_trunc",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.dateadd`.
pub fn dateadd(col1: Column, col2: Column) -> Column {
    func(
        "dateadd",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.datediff`.
pub fn datediff(col1: Column, col2: Column) -> Column {
    func(
        "datediff",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.datepart`.
pub fn datepart(col1: Column, col2: Column) -> Column {
    func(
        "datepart",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.decode`.
pub fn decode(col1: Column, col2: Column) -> Column {
    func(
        "decode",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.element_at`.
pub fn element_at(col1: Column, col2: Column) -> Column {
    func(
        "element_at",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.encode`.
pub fn encode(col1: Column, col2: Column) -> Column {
    func(
        "encode",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.endswith`.
pub fn endswith(col1: Column, col2: Column) -> Column {
    func(
        "endswith",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.equal_null`.
pub fn equal_null(col1: Column, col2: Column) -> Column {
    func(
        "equal_null",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.extract`.
pub fn extract(col1: Column, col2: Column) -> Column {
    func(
        "extract",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.find_in_set`.
pub fn find_in_set(col1: Column, col2: Column) -> Column {
    func(
        "find_in_set",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.format_number`.
pub fn format_number(col1: Column, col2: Column) -> Column {
    func(
        "format_number",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.from_csv`.
pub fn from_csv(col1: Column, col2: Column) -> Column {
    func(
        "from_csv",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.from_json`.
pub fn from_json(col1: Column, col2: Column) -> Column {
    func(
        "from_json",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.from_utc_timestamp`.
pub fn from_utc_timestamp(col1: Column, col2: Column) -> Column {
    func(
        "from_utc_timestamp",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.from_xml`.
pub fn from_xml(col1: Column, col2: Column) -> Column {
    func(
        "from_xml",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.get`.
pub fn get(col1: Column, col2: Column) -> Column {
    func(
        "get",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.get_json_object`.
pub fn get_json_object(col1: Column, col2: Column) -> Column {
    func(
        "get_json_object",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.getbit`.
pub fn getbit(col1: Column, col2: Column) -> Column {
    func(
        "getbit",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.greatest`.
pub fn greatest(col1: Column, col2: Column) -> Column {
    func(
        "greatest",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.histogram_numeric`.
pub fn histogram_numeric(col1: Column, col2: Column) -> Column {
    func(
        "histogram_numeric",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.hll_union`.
pub fn hll_union(col1: Column, col2: Column) -> Column {
    func(
        "hll_union",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.hypot`.
pub fn hypot(col1: Column, col2: Column) -> Column {
    func(
        "hypot",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.ifnull`.
pub fn ifnull(col1: Column, col2: Column) -> Column {
    func(
        "ifnull",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.ilike`.
pub fn ilike(col1: Column, col2: Column) -> Column {
    func(
        "ilike",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.instr`.
pub fn instr(col1: Column, col2: Column) -> Column {
    func(
        "instr",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.json_tuple`.
pub fn json_tuple(col1: Column, col2: Column) -> Column {
    func(
        "json_tuple",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.least`.
pub fn least(col1: Column, col2: Column) -> Column {
    func(
        "least",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.left`.
pub fn left(col1: Column, col2: Column) -> Column {
    func(
        "left",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.levenshtein`.
pub fn levenshtein(col1: Column, col2: Column) -> Column {
    func(
        "levenshtein",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.like`.
pub fn like(col1: Column, col2: Column) -> Column {
    func(
        "like",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.locate`.
pub fn locate(col1: Column, col2: Column) -> Column {
    func(
        "locate",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            lit_int(1),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.map_contains_key`.
pub fn map_contains_key(col1: Column, col2: Column) -> Column {
    func(
        "map_contains_key",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.map_from_arrays`.
pub fn map_from_arrays(col1: Column, col2: Column) -> Column {
    func(
        "map_from_arrays",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.max_by`.
pub fn max_by(col1: Column, col2: Column) -> Column {
    func(
        "max_by",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.min_by`.
pub fn min_by(col1: Column, col2: Column) -> Column {
    func(
        "min_by",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.months_between`.
pub fn months_between(col1: Column, col2: Column) -> Column {
    func(
        "months_between",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            lit_bool(true),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.nanvl`.
pub fn nanvl(col1: Column, col2: Column) -> Column {
    func(
        "nanvl",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.next_day`.
pub fn next_day(col1: Column, col2: Column) -> Column {
    func(
        "next_day",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.nth_value`.
pub fn nth_value(col1: Column, col2: Column) -> Column {
    func(
        "nth_value",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            lit_bool(false),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.nullif`.
pub fn nullif(col1: Column, col2: Column) -> Column {
    func(
        "nullif",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.nvl`.
pub fn nvl(col1: Column, col2: Column) -> Column {
    func(
        "nvl",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.parse_url`.
pub fn parse_url(col1: Column, col2: Column) -> Column {
    func(
        "parse_url",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.percentile`.
pub fn percentile(col1: Column, col2: Column) -> Column {
    func(
        "percentile",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            lit_int(1),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.percentile_approx`.
pub fn percentile_approx(col1: Column, col2: Column) -> Column {
    func(
        "percentile_approx",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            lit_int(10000),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.pmod`.
pub fn pmod(col1: Column, col2: Column) -> Column {
    func(
        "pmod",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.position`.
pub fn position(col1: Column, col2: Column) -> Column {
    func(
        "position",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.pow`.
pub fn pow(col1: Column, col2: Column) -> Column {
    func(
        "power",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.power`.
pub fn power(col1: Column, col2: Column) -> Column {
    func(
        "power",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regexp`.
pub fn regexp(col1: Column, col2: Column) -> Column {
    func(
        "regexp",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regexp_count`.
pub fn regexp_count(col1: Column, col2: Column) -> Column {
    func(
        "regexp_count",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regexp_extract_all`.
pub fn regexp_extract_all(col1: Column, col2: Column) -> Column {
    func(
        "regexp_extract_all",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regexp_instr`.
pub fn regexp_instr(col1: Column, col2: Column) -> Column {
    func(
        "regexp_instr",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regexp_like`.
pub fn regexp_like(col1: Column, col2: Column) -> Column {
    func(
        "regexp_like",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regexp_substr`.
pub fn regexp_substr(col1: Column, col2: Column) -> Column {
    func(
        "regexp_substr",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regr_avgx`.
pub fn regr_avgx(col1: Column, col2: Column) -> Column {
    func(
        "regr_avgx",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regr_avgy`.
pub fn regr_avgy(col1: Column, col2: Column) -> Column {
    func(
        "regr_avgy",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regr_count`.
pub fn regr_count(col1: Column, col2: Column) -> Column {
    func(
        "regr_count",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regr_intercept`.
pub fn regr_intercept(col1: Column, col2: Column) -> Column {
    func(
        "regr_intercept",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regr_r2`.
pub fn regr_r2(col1: Column, col2: Column) -> Column {
    func(
        "regr_r2",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regr_slope`.
pub fn regr_slope(col1: Column, col2: Column) -> Column {
    func(
        "regr_slope",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regr_sxx`.
pub fn regr_sxx(col1: Column, col2: Column) -> Column {
    func(
        "regr_sxx",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regr_sxy`.
pub fn regr_sxy(col1: Column, col2: Column) -> Column {
    func(
        "regr_sxy",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.regr_syy`.
pub fn regr_syy(col1: Column, col2: Column) -> Column {
    func(
        "regr_syy",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.repeat`.
pub fn repeat(col1: Column, col2: Column) -> Column {
    func(
        "repeat",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.replace`.
pub fn replace(col1: Column, col2: Column) -> Column {
    func(
        "replace",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.right`.
pub fn right(col1: Column, col2: Column) -> Column {
    func(
        "right",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.rlike`.
pub fn rlike(col1: Column, col2: Column) -> Column {
    func(
        "rlike",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.sequence`.
pub fn sequence(col1: Column, col2: Column) -> Column {
    func(
        "sequence",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.session_window`.
pub fn session_window(col1: Column, col2: Column) -> Column {
    func(
        "session_window",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.shiftLeft`.
pub fn shiftLeft(col1: Column, col2: Column) -> Column {
    func(
        "shiftleft",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.shiftRight`.
pub fn shiftRight(col1: Column, col2: Column) -> Column {
    func(
        "shiftright",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.shiftRightUnsigned`.
pub fn shiftRightUnsigned(col1: Column, col2: Column) -> Column {
    func(
        "shiftrightunsigned",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.shiftleft`.
pub fn shiftleft(col1: Column, col2: Column) -> Column {
    func(
        "shiftleft",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.shiftright`.
pub fn shiftright(col1: Column, col2: Column) -> Column {
    func(
        "shiftright",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.shiftrightunsigned`.
pub fn shiftrightunsigned(col1: Column, col2: Column) -> Column {
    func(
        "shiftrightunsigned",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.split`.
pub fn split(col1: Column, col2: Column) -> Column {
    func(
        "split",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            lit_int(-1),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.startswith`.
pub fn startswith(col1: Column, col2: Column) -> Column {
    func(
        "startswith",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.substr`.
pub fn substr(col1: Column, col2: Column) -> Column {
    func(
        "substr",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.to_char`.
pub fn to_char(col1: Column, col2: Column) -> Column {
    func(
        "to_char",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.to_number`.
pub fn to_number(col1: Column, col2: Column) -> Column {
    func(
        "to_number",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.to_utc_timestamp`.
pub fn to_utc_timestamp(col1: Column, col2: Column) -> Column {
    func(
        "to_utc_timestamp",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.to_varchar`.
pub fn to_varchar(col1: Column, col2: Column) -> Column {
    func(
        "to_varchar",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.trunc`.
pub fn trunc(col1: Column, col2: Column) -> Column {
    func(
        "trunc",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.try_add`.
pub fn try_add(col1: Column, col2: Column) -> Column {
    func(
        "try_add",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.try_aes_decrypt`.
pub fn try_aes_decrypt(col1: Column, col2: Column) -> Column {
    func(
        "try_aes_decrypt",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            lit_str("GCM"),
            lit_str("DEFAULT"),
            lit_str(""),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.try_divide`.
pub fn try_divide(col1: Column, col2: Column) -> Column {
    func(
        "try_divide",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.try_element_at`.
pub fn try_element_at(col1: Column, col2: Column) -> Column {
    func(
        "try_element_at",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.try_mod`.
pub fn try_mod(col1: Column, col2: Column) -> Column {
    func(
        "try_mod",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.try_multiply`.
pub fn try_multiply(col1: Column, col2: Column) -> Column {
    func(
        "try_multiply",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.try_parse_url`.
pub fn try_parse_url(col1: Column, col2: Column) -> Column {
    func(
        "try_parse_url",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.try_subtract`.
pub fn try_subtract(col1: Column, col2: Column) -> Column {
    func(
        "try_subtract",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.try_to_number`.
pub fn try_to_number(col1: Column, col2: Column) -> Column {
    func(
        "try_to_number",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.uniform`.
pub fn uniform(col1: Column, col2: Column) -> Column {
    func(
        "uniform",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            lit_long(1779860837174053365),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.when`.
pub fn when(col1: Column, col2: Column) -> Column {
    func(
        "when",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.xpath`.
pub fn xpath(col1: Column, col2: Column) -> Column {
    func(
        "xpath",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.xpath_boolean`.
pub fn xpath_boolean(col1: Column, col2: Column) -> Column {
    func(
        "xpath_boolean",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.xpath_double`.
pub fn xpath_double(col1: Column, col2: Column) -> Column {
    func(
        "xpath_double",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.xpath_float`.
pub fn xpath_float(col1: Column, col2: Column) -> Column {
    func(
        "xpath_float",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.xpath_int`.
pub fn xpath_int(col1: Column, col2: Column) -> Column {
    func(
        "xpath_int",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.xpath_long`.
pub fn xpath_long(col1: Column, col2: Column) -> Column {
    func(
        "xpath_long",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.xpath_number`.
pub fn xpath_number(col1: Column, col2: Column) -> Column {
    func(
        "xpath_number",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.xpath_short`.
pub fn xpath_short(col1: Column, col2: Column) -> Column {
    func(
        "xpath_short",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.xpath_string`.
pub fn xpath_string(col1: Column, col2: Column) -> Column {
    func(
        "xpath_string",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

// ============================================================================
// 3-argument functions
// ============================================================================

/// Mirrors `pyspark.sql.functions.array_insert`.
pub fn array_insert(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "array_insert",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.conv`.
pub fn conv(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "conv",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.convert_timezone`.
pub fn convert_timezone(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "convert_timezone",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.count_min_sketch`.
pub fn count_min_sketch(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "count_min_sketch",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
            lit_long(9151685696388386553),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.lpad`.
pub fn lpad(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "lpad",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.make_date`.
pub fn make_date(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "make_date",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.nvl2`.
pub fn nvl2(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "nvl2",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.overlay`.
pub fn overlay(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "overlay",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
            lit_int(-1),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.regexp_extract`.
pub fn regexp_extract(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "regexp_extract",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.regexp_replace`.
pub fn regexp_replace(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "regexp_replace",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.rpad`.
pub fn rpad(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "rpad",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.slice`.
pub fn slice(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "slice",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.split_part`.
pub fn split_part(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "split_part",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.substring`.
pub fn substring(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "substring",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.substring_index`.
pub fn substring_index(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "substring_index",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.timestamp_add`.
pub fn timestamp_add(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "timestampadd",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.timestamp_diff`.
pub fn timestamp_diff(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "timestampdiff",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.translate`.
pub fn translate(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "translate",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.try_variant_get`.
pub fn try_variant_get(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "try_variant_get",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.variant_get`.
pub fn variant_get(col1: Column, col2: Column, col3: Column) -> Column {
    func(
        "variant_get",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
        ],
    )
}

// ============================================================================
// 4-argument functions
// ============================================================================

/// Mirrors `pyspark.sql.functions.width_bucket`.
pub fn width_bucket(col1: Column, col2: Column, col3: Column, col4: Column) -> Column {
    func(
        "width_bucket",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            col3.expression().clone(),
            col4.expression().clone(),
        ],
    )
}

// ============================================================================
// Higher-Order Functions (HOF) - functions that take lambda expressions
// ============================================================================

use crate::expression::{next_lambda_var_index, LambdaFunction, UnresolvedNamedLambdaVariable};

/// Build a lambda function with fresh variable names from a closure.
/// The closure receives a Vec of fresh Column variables and returns the body expression.
fn make_lambda<F>(num_args: usize, closure: F) -> LambdaFunction
where
    F: FnOnce(Vec<Column>) -> Column,
{
    let mut vars = Vec::new();
    let mut arg_exprs = Vec::new();

    for _ in 0..num_args {
        let idx = next_lambda_var_index();
        let var_name = format!("x_{}", idx);
        let var = UnresolvedNamedLambdaVariable::new(var_name.clone());
        arg_exprs.push(var.clone());
        vars.push(Column::new(Expression::UnresolvedNamedLambdaVariable(var)));
    }

    let body = closure(vars);
    LambdaFunction::new(body.expression().clone(), arg_exprs)
}

/// Mirrors `pyspark.sql.functions.transform` with 1-arg lambda: `transform(col, lambda x: body(x))`.
pub fn transform<F>(col: Column, f: F) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    let lambda = make_lambda(1, f);
    func(
        "transform",
        vec![
            col.expression().clone(),
            Expression::LambdaFunction(Box::new(lambda)),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.transform` with 2-arg lambda: `transform(col, lambda x, i: body(x, i))`.
pub fn transform_idx<F>(col: Column, f: F) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    let lambda = make_lambda(2, f);
    func(
        "transform",
        vec![
            col.expression().clone(),
            Expression::LambdaFunction(Box::new(lambda)),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.filter` with 1-arg lambda: `filter(col, lambda x: body(x))`.
pub fn filter<F>(col: Column, f: F) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    let lambda = make_lambda(1, f);
    func(
        "filter",
        vec![
            col.expression().clone(),
            Expression::LambdaFunction(Box::new(lambda)),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.exists` with 1-arg lambda: `exists(col, lambda x: body(x))`.
pub fn exists<F>(col: Column, f: F) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    let lambda = make_lambda(1, f);
    func(
        "exists",
        vec![
            col.expression().clone(),
            Expression::LambdaFunction(Box::new(lambda)),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.forall` with 1-arg lambda: `forall(col, lambda x: body(x))`.
pub fn forall<F>(col: Column, f: F) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    let lambda = make_lambda(1, f);
    func(
        "forall",
        vec![
            col.expression().clone(),
            Expression::LambdaFunction(Box::new(lambda)),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.aggregate` with 2-arg merge lambda.
/// `aggregate(col, init, lambda (acc, x): merge(acc, x))`
pub fn aggregate<F>(col: Column, init: Column, f: F) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    let lambda = make_lambda(2, f);
    func(
        "aggregate",
        vec![
            col.expression().clone(),
            init.expression().clone(),
            Expression::LambdaFunction(Box::new(lambda)),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.aggregate` with both merge and finish lambdas.
/// `aggregate(col, init, lambda (acc, x): merge(acc, x), lambda acc: finish(acc))`
pub fn aggregate_with_finish<F, G>(col: Column, init: Column, merge_f: F, finish_f: G) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
    G: FnOnce(Vec<Column>) -> Column,
{
    let merge_lambda = make_lambda(2, merge_f);
    let finish_lambda = make_lambda(1, finish_f);
    func(
        "aggregate",
        vec![
            col.expression().clone(),
            init.expression().clone(),
            Expression::LambdaFunction(Box::new(merge_lambda)),
            Expression::LambdaFunction(Box::new(finish_lambda)),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.zip_with` with 2-arg lambda.
/// `zip_with(col1, col2, lambda (x, y): body(x, y))`
pub fn zip_with<F>(col1: Column, col2: Column, f: F) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    let lambda = make_lambda(2, f);
    func(
        "zip_with",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            Expression::LambdaFunction(Box::new(lambda)),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.transform_keys` with 2-arg lambda.
/// `transform_keys(col, lambda (k, v): body(k, v))`
pub fn transform_keys<F>(col: Column, f: F) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    let lambda = make_lambda(2, f);
    func(
        "transform_keys",
        vec![
            col.expression().clone(),
            Expression::LambdaFunction(Box::new(lambda)),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.transform_values` with 2-arg lambda.
/// `transform_values(col, lambda (k, v): body(k, v))`
pub fn transform_values<F>(col: Column, f: F) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    let lambda = make_lambda(2, f);
    func(
        "transform_values",
        vec![
            col.expression().clone(),
            Expression::LambdaFunction(Box::new(lambda)),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.map_filter` with 2-arg lambda.
/// `map_filter(col, lambda (k, v): body(k, v))`
pub fn map_filter<F>(col: Column, f: F) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    let lambda = make_lambda(2, f);
    func(
        "map_filter",
        vec![
            col.expression().clone(),
            Expression::LambdaFunction(Box::new(lambda)),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.map_zip_with` with 3-arg lambda.
/// `map_zip_with(col1, col2, lambda (k, v1, v2): body(k, v1, v2))`
pub fn map_zip_with<F>(col1: Column, col2: Column, f: F) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    let lambda = make_lambda(3, f);
    func(
        "map_zip_with",
        vec![
            col1.expression().clone(),
            col2.expression().clone(),
            Expression::LambdaFunction(Box::new(lambda)),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.reduce` (alias for aggregate).
pub fn reduce<F>(col: Column, init: Column, f: F) -> Column
where
    F: FnOnce(Vec<Column>) -> Column,
{
    aggregate(col, init, f)
}

// ============================================================================
// Protobuf functions
// ============================================================================

/// Mirrors `pyspark.sql.connect.protobuf.functions.from_protobuf`.
/// Decodes a binary protobuf message into a column using the specified message name.
pub fn from_protobuf(data: Column, messageName: &str) -> Column {
    func(
        "from_protobuf",
        vec![data.expression().clone(), lit_str(messageName)],
    )
}

/// Mirrors `pyspark.sql.connect.protobuf.functions.from_protobuf` with descriptor set.
/// Decodes a binary protobuf message into a column using the specified message name and descriptor.
pub fn from_protobuf_with_descriptor(
    data: Column,
    messageName: &str,
    binaryDescriptorSet: Vec<u8>,
) -> Column {
    func(
        "from_protobuf",
        vec![
            data.expression().clone(),
            lit_str(messageName),
            lit_binary(binaryDescriptorSet),
        ],
    )
}

/// Mirrors `pyspark.sql.connect.protobuf.functions.from_protobuf` with descriptor and options.
/// Decodes a binary protobuf message into a column using the specified message name, descriptor, and options.
pub fn from_protobuf_with_descriptor_and_options(
    data: Column,
    messageName: &str,
    binaryDescriptorSet: Vec<u8>,
    options: Column,
) -> Column {
    func(
        "from_protobuf",
        vec![
            data.expression().clone(),
            lit_str(messageName),
            lit_binary(binaryDescriptorSet),
            options.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.connect.protobuf.functions.from_protobuf` with options only.
/// Decodes a binary protobuf message into a column using the specified message name and options.
pub fn from_protobuf_with_options(data: Column, messageName: &str, options: Column) -> Column {
    func(
        "from_protobuf",
        vec![
            data.expression().clone(),
            lit_str(messageName),
            options.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.connect.protobuf.functions.to_protobuf`.
/// Encodes a column into a binary protobuf message using the specified message name.
pub fn to_protobuf(data: Column, messageName: &str) -> Column {
    func(
        "to_protobuf",
        vec![data.expression().clone(), lit_str(messageName)],
    )
}

/// Mirrors `pyspark.sql.connect.protobuf.functions.to_protobuf` with descriptor set.
/// Encodes a column into a binary protobuf message using the specified message name and descriptor.
pub fn to_protobuf_with_descriptor(
    data: Column,
    messageName: &str,
    binaryDescriptorSet: Vec<u8>,
) -> Column {
    func(
        "to_protobuf",
        vec![
            data.expression().clone(),
            lit_str(messageName),
            lit_binary(binaryDescriptorSet),
        ],
    )
}

/// Mirrors `pyspark.sql.connect.protobuf.functions.to_protobuf` with descriptor and options.
/// Encodes a column into a binary protobuf message using the specified message name, descriptor, and options.
pub fn to_protobuf_with_descriptor_and_options(
    data: Column,
    messageName: &str,
    binaryDescriptorSet: Vec<u8>,
    options: Column,
) -> Column {
    func(
        "to_protobuf",
        vec![
            data.expression().clone(),
            lit_str(messageName),
            lit_binary(binaryDescriptorSet),
            options.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.connect.protobuf.functions.to_protobuf` with options only.
/// Encodes a column into a binary protobuf message using the specified message name and options.
pub fn to_protobuf_with_options(data: Column, messageName: &str, options: Column) -> Column {
    func(
        "to_protobuf",
        vec![
            data.expression().clone(),
            lit_str(messageName),
            options.expression().clone(),
        ],
    )
}

// ============================================================================
// Avro functions
// ============================================================================

/// Mirrors `pyspark.sql.connect.avro.functions.from_avro`.
/// Decodes a column of binary avro data into a column using the specified JSON schema.
pub fn from_avro(data: Column, jsonFormatSchema: &str) -> Column {
    func(
        "from_avro",
        vec![data.expression().clone(), lit_str(jsonFormatSchema)],
    )
}

/// Mirrors `pyspark.sql.connect.avro.functions.from_avro` with options.
/// Decodes a column of binary avro data into a column using the specified JSON schema and options.
pub fn from_avro_with_options(data: Column, jsonFormatSchema: &str, options: Column) -> Column {
    func(
        "from_avro",
        vec![
            data.expression().clone(),
            lit_str(jsonFormatSchema),
            options.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.connect.avro.functions.to_avro`.
/// Encodes a column into a column of binary avro data.
pub fn to_avro(data: Column) -> Column {
    func("to_avro", vec![data.expression().clone()])
}

/// Mirrors `pyspark.sql.connect.avro.functions.to_avro` with JSON schema.
/// Encodes a column into a column of binary avro data using the specified JSON schema.
pub fn to_avro_with_schema(data: Column, jsonFormatSchema: &str) -> Column {
    func(
        "to_avro",
        vec![data.expression().clone(), lit_str(jsonFormatSchema)],
    )
}
