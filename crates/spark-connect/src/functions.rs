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

/// Mirrors `pyspark.sql.functions.call_function`: builds a `CallFunction`
/// expression carrying the argument columns.
pub fn call_function<C: Into<Column>>(name: &str, args: impl IntoIterator<Item = C>) -> Column {
    let args: Vec<Column> = args.into_iter().map(Into::into).collect();
    let arg_exprs = args.iter().map(|c| c.expression().clone()).collect();
    Column::new(Expression::CallFunction(Box::new(
        CallFunctionWrapper::new(name, arg_exprs),
    )))
}

/// Mirrors `pyspark.sql.functions.call_udf` = `_invoke_function(udfName, *cols)`
/// (an `UnresolvedFunction` call).
pub fn call_udf<C: Into<Column>>(name: &str, args: impl IntoIterator<Item = C>) -> Column {
    let args: Vec<Column> = args.into_iter().map(Into::into).collect();
    func(name, args.iter().map(|c| c.expression().clone()).collect())
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

/// Mirrors `pyspark.sql.functions.array` (variadic over the given columns).
pub fn array<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "array",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
}
/// Mirrors `pyspark.sql.functions.concat` (variadic over the given columns).
pub fn concat<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "concat",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
}
/// Mirrors `pyspark.sql.functions.coalesce` (variadic over the given columns).
pub fn coalesce<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "coalesce",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
}
/// Mirrors `pyspark.sql.functions.arrays_zip`.
pub fn arrays_zip<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "arrays_zip",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
}
/// Mirrors `pyspark.sql.functions.create_map`.
pub fn create_map<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func("map", cols.iter().map(|c| c.expression().clone()).collect())
}

/// Mirrors `pyspark.sql.functions.window` (column + interval string).
pub fn window(col: Column, window_duration: &str) -> Column {
    func(
        "window",
        vec![col.expression().clone(), lit_str(window_duration)],
    )
}

/// Mirrors `pyspark.sql.functions.window` with an explicit slide duration and
/// start time (a tumbling window is `slide_duration == window_duration`;
/// `start_time` offsets the window boundaries, e.g. `"15 minutes"`).
pub fn window_with_slide_and_start(
    col: Column,
    window_duration: &str,
    slide_duration: &str,
    start_time: &str,
) -> Column {
    func(
        "window",
        vec![
            col.expression().clone(),
            lit_str(window_duration),
            lit_str(slide_duration),
            lit_str(start_time),
        ],
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
pub fn elt<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func("elt", cols.iter().map(|c| c.expression().clone()).collect())
}

/// Mirrors `pyspark.sql.functions.grouping_id`.
pub fn grouping_id<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "grouping_id",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
}

/// Mirrors `pyspark.sql.functions.hash`.
pub fn hash<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "hash",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
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
pub fn java_method<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "java_method",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
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
pub fn map_concat<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "map_concat",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
}

/// Mirrors `pyspark.sql.functions.monotonically_increasing_id`.
pub fn monotonically_increasing_id() -> Column {
    func("monotonically_increasing_id", vec![])
}

/// Mirrors `pyspark.sql.functions.named_struct`.
pub fn named_struct<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "named_struct",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
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
pub fn reflect<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "reflect",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
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
pub fn stack<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "stack",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
}

/// Mirrors `pyspark.sql.functions.struct`.
pub fn r#struct<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "struct",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
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
pub fn try_reflect<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "try_reflect",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
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
pub fn xxhash64<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "xxhash64",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
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

/// 2-arg variant (see the reference optional argument).
pub fn approx_count_distinct_rsd(col1: Column, rsd: Column) -> Column {
    func(
        "approx_count_distinct",
        vec![col1.expression().clone(), rsd.expression().clone()],
    )
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

/// 2-arg variant (see the reference optional argument).
pub fn bround_scale(col1: Column, scale: Column) -> Column {
    func(
        "bround",
        vec![col1.expression().clone(), scale.expression().clone()],
    )
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

/// 2-arg variant (see the reference optional argument).
pub fn ceil_scale(col1: Column, scale: Column) -> Column {
    func(
        "ceil",
        vec![col1.expression().clone(), scale.expression().clone()],
    )
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

/// Mirrors `pyspark.sql.functions.concat_ws(sep, *cols)`: the separator literal
/// followed by a variadic number of columns. Variadic like `concat` so extra
/// columns are not dropped.
pub fn concat_ws<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "concat_ws",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
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

/// 2-arg variant (see the reference optional argument).
pub fn floor_scale(col1: Column, scale: Column) -> Column {
    func(
        "floor",
        vec![col1.expression().clone(), scale.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.format_string(format, *cols)`: the format
/// literal followed by a variadic number of columns. Variadic like `concat` so
/// extra columns are not dropped.
pub fn format_string<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> Column {
    let cols: Vec<Column> = cols.into_iter().map(Into::into).collect();
    func(
        "format_string",
        cols.iter().map(|c| c.expression().clone()).collect(),
    )
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

/// 2-arg variant (see the reference optional argument).
pub fn ltrim_with(col1: Column, trim: Column) -> Column {
    func(
        "ltrim",
        vec![trim.expression().clone(), col1.expression().clone()],
    )
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

/// 2-arg variant (see the reference optional argument).
pub fn round_scale(col1: Column, scale: Column) -> Column {
    func(
        "round",
        vec![col1.expression().clone(), scale.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.rtrim`.
pub fn rtrim(col1: Column) -> Column {
    func("rtrim", vec![col1.expression().clone()])
}

/// 2-arg variant (see the reference optional argument).
pub fn rtrim_with(col1: Column, trim: Column) -> Column {
    func(
        "rtrim",
        vec![trim.expression().clone(), col1.expression().clone()],
    )
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

/// 2-arg variant (see the reference optional argument).
pub fn to_binary_format(col1: Column, format: Column) -> Column {
    func(
        "to_binary",
        vec![col1.expression().clone(), format.expression().clone()],
    )
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

/// 2-arg variant (see the reference optional argument).
pub fn trim_with(col1: Column, trim: Column) -> Column {
    // pyspark emits the trim string first, then the column: TRIM(trimStr FROM col).
    func(
        "trim",
        vec![trim.expression().clone(), col1.expression().clone()],
    )
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

// ===== Batch 1: additional builtin functions (parity) =====
/// Mirrors `pyspark.sql.functions.bitmap_and_agg`.
pub fn bitmap_and_agg(col: Column) -> Column {
    func("bitmap_and_agg", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.chr`.
pub fn chr(n: Column) -> Column {
    func("chr", vec![n.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.counter_diff`.
pub fn counter_diff(value: Column) -> Column {
    func("counter_diff", vec![value.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.current_path`.
pub fn current_path() -> Column {
    func("current_path", vec![])
}

/// Mirrors `pyspark.sql.functions.current_time`.
pub fn current_time() -> Column {
    func("current_time", vec![])
}

/// Mirrors `pyspark.sql.functions.hmac`.
pub fn hmac(key: Column, message: Column) -> Column {
    func(
        "hmac",
        vec![key.expression().clone(), message.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.is_valid_variant`.
pub fn is_valid_variant(v: Column) -> Column {
    func("is_valid_variant", vec![v.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.jaro_winkler_similarity`.
pub fn jaro_winkler_similarity(left: Column, right: Column) -> Column {
    func(
        "jaro_winkler_similarity",
        vec![left.expression().clone(), right.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.kll_merge_agg_bigint`.
pub fn kll_merge_agg_bigint(col: Column) -> Column {
    func("kll_merge_agg_bigint", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.kll_merge_agg_double`.
pub fn kll_merge_agg_double(col: Column) -> Column {
    func("kll_merge_agg_double", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.kll_merge_agg_float`.
pub fn kll_merge_agg_float(col: Column) -> Column {
    func("kll_merge_agg_float", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.kll_sketch_agg_bigint`.
pub fn kll_sketch_agg_bigint(col: Column) -> Column {
    func("kll_sketch_agg_bigint", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.kll_sketch_agg_double`.
pub fn kll_sketch_agg_double(col: Column) -> Column {
    func("kll_sketch_agg_double", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.kll_sketch_agg_float`.
pub fn kll_sketch_agg_float(col: Column) -> Column {
    func("kll_sketch_agg_float", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.kll_sketch_get_n_bigint`.
pub fn kll_sketch_get_n_bigint(col: Column) -> Column {
    func("kll_sketch_get_n_bigint", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.kll_sketch_get_n_double`.
pub fn kll_sketch_get_n_double(col: Column) -> Column {
    func("kll_sketch_get_n_double", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.kll_sketch_get_n_float`.
pub fn kll_sketch_get_n_float(col: Column) -> Column {
    func("kll_sketch_get_n_float", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.kll_sketch_get_quantile_bigint`.
pub fn kll_sketch_get_quantile_bigint(sketch: Column, rank: Column) -> Column {
    func(
        "kll_sketch_get_quantile_bigint",
        vec![sketch.expression().clone(), rank.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.kll_sketch_get_quantile_double`.
pub fn kll_sketch_get_quantile_double(sketch: Column, rank: Column) -> Column {
    func(
        "kll_sketch_get_quantile_double",
        vec![sketch.expression().clone(), rank.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.kll_sketch_get_quantile_float`.
pub fn kll_sketch_get_quantile_float(sketch: Column, rank: Column) -> Column {
    func(
        "kll_sketch_get_quantile_float",
        vec![sketch.expression().clone(), rank.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.kll_sketch_get_rank_bigint`.
pub fn kll_sketch_get_rank_bigint(sketch: Column, quantile: Column) -> Column {
    func(
        "kll_sketch_get_rank_bigint",
        vec![sketch.expression().clone(), quantile.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.kll_sketch_get_rank_double`.
pub fn kll_sketch_get_rank_double(sketch: Column, quantile: Column) -> Column {
    func(
        "kll_sketch_get_rank_double",
        vec![sketch.expression().clone(), quantile.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.kll_sketch_get_rank_float`.
pub fn kll_sketch_get_rank_float(sketch: Column, quantile: Column) -> Column {
    func(
        "kll_sketch_get_rank_float",
        vec![sketch.expression().clone(), quantile.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.kll_sketch_merge_bigint`.
pub fn kll_sketch_merge_bigint(left: Column, right: Column) -> Column {
    func(
        "kll_sketch_merge_bigint",
        vec![left.expression().clone(), right.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.kll_sketch_merge_double`.
pub fn kll_sketch_merge_double(left: Column, right: Column) -> Column {
    func(
        "kll_sketch_merge_double",
        vec![left.expression().clone(), right.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.kll_sketch_merge_float`.
pub fn kll_sketch_merge_float(left: Column, right: Column) -> Column {
    func(
        "kll_sketch_merge_float",
        vec![left.expression().clone(), right.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.kll_sketch_to_string_bigint`.
pub fn kll_sketch_to_string_bigint(col: Column) -> Column {
    func(
        "kll_sketch_to_string_bigint",
        vec![col.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.kll_sketch_to_string_double`.
pub fn kll_sketch_to_string_double(col: Column) -> Column {
    func(
        "kll_sketch_to_string_double",
        vec![col.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.kll_sketch_to_string_float`.
pub fn kll_sketch_to_string_float(col: Column) -> Column {
    func("kll_sketch_to_string_float", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.make_time`.
pub fn make_time(hour: Column, minute: Column, second: Column) -> Column {
    func(
        "make_time",
        vec![
            hour.expression().clone(),
            minute.expression().clone(),
            second.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.make_timestamp`.
pub fn make_timestamp(
    years: Column,
    months: Column,
    days: Column,
    hours: Column,
    mins: Column,
    secs: Column,
) -> Column {
    func(
        "make_timestamp",
        vec![
            years.expression().clone(),
            months.expression().clone(),
            days.expression().clone(),
            hours.expression().clone(),
            mins.expression().clone(),
            secs.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.make_timestamp_ltz`.
pub fn make_timestamp_ltz(
    years: Column,
    months: Column,
    days: Column,
    hours: Column,
    mins: Column,
    secs: Column,
) -> Column {
    func(
        "make_timestamp_ltz",
        vec![
            years.expression().clone(),
            months.expression().clone(),
            days.expression().clone(),
            hours.expression().clone(),
            mins.expression().clone(),
            secs.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.make_timestamp_ntz`.
pub fn make_timestamp_ntz(
    years: Column,
    months: Column,
    days: Column,
    hours: Column,
    mins: Column,
    secs: Column,
) -> Column {
    func(
        "make_timestamp_ntz",
        vec![
            years.expression().clone(),
            months.expression().clone(),
            days.expression().clone(),
            hours.expression().clone(),
            mins.expression().clone(),
            secs.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.quote`.
pub fn quote(col: Column) -> Column {
    func("quote", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.sha2`.
pub fn sha2(col: Column, numbits: i32) -> Column {
    func("sha2", vec![col.expression().clone(), lit_int(numbits)])
}

/// Mirrors `pyspark.sql.functions.st_asbinary`.
pub fn st_asbinary(geo: Column) -> Column {
    func("st_asbinary", vec![geo.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.st_geogfromwkb`.
pub fn st_geogfromwkb(wkb: Column) -> Column {
    func("st_geogfromwkb", vec![wkb.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.st_geomfromwkb`.
pub fn st_geomfromwkb(wkb: Column) -> Column {
    func("st_geomfromwkb", vec![wkb.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.st_setsrid`.
pub fn st_setsrid(geo: Column, srid: Column) -> Column {
    func(
        "st_setsrid",
        vec![geo.expression().clone(), srid.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.st_srid`.
pub fn st_srid(geo: Column) -> Column {
    func("st_srid", vec![geo.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.theta_difference`.
pub fn theta_difference(col1: Column, col2: Column) -> Column {
    func(
        "theta_difference",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.theta_intersection`.
pub fn theta_intersection(col1: Column, col2: Column) -> Column {
    func(
        "theta_intersection",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.theta_intersection_agg`.
pub fn theta_intersection_agg(col: Column) -> Column {
    func("theta_intersection_agg", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.theta_sketch_agg`.
pub fn theta_sketch_agg(col: Column) -> Column {
    func("theta_sketch_agg", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.theta_sketch_estimate`.
pub fn theta_sketch_estimate(col: Column) -> Column {
    func("theta_sketch_estimate", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.theta_union`.
pub fn theta_union(col1: Column, col2: Column) -> Column {
    func(
        "theta_union",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.theta_union_agg`.
pub fn theta_union_agg(col: Column) -> Column {
    func("theta_union_agg", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.time_bucket`.
pub fn time_bucket(bucket_size: Column, ts: Column) -> Column {
    func(
        "time_bucket",
        vec![bucket_size.expression().clone(), ts.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.time_diff`.
pub fn time_diff(unit: Column, start: Column, end: Column) -> Column {
    func(
        "time_diff",
        vec![
            unit.expression().clone(),
            start.expression().clone(),
            end.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.time_from_micros`.
pub fn time_from_micros(col: Column) -> Column {
    func("time_from_micros", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.time_from_millis`.
pub fn time_from_millis(col: Column) -> Column {
    func("time_from_millis", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.time_from_seconds`.
pub fn time_from_seconds(col: Column) -> Column {
    func("time_from_seconds", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.time_to_micros`.
pub fn time_to_micros(col: Column) -> Column {
    func("time_to_micros", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.time_to_millis`.
pub fn time_to_millis(col: Column) -> Column {
    func("time_to_millis", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.time_to_seconds`.
pub fn time_to_seconds(col: Column) -> Column {
    func("time_to_seconds", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.time_trunc`.
pub fn time_trunc(unit: Column, time: Column) -> Column {
    func(
        "time_trunc",
        vec![unit.expression().clone(), time.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.timestamp_nanos`.
pub fn timestamp_nanos(col: Column) -> Column {
    func("timestamp_nanos", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.to_time`.
pub fn to_time(str: Column) -> Column {
    func("to_time", vec![str.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.try_make_timestamp`.
pub fn try_make_timestamp(
    years: Column,
    months: Column,
    days: Column,
    hours: Column,
    mins: Column,
    secs: Column,
) -> Column {
    func(
        "try_make_timestamp",
        vec![
            years.expression().clone(),
            months.expression().clone(),
            days.expression().clone(),
            hours.expression().clone(),
            mins.expression().clone(),
            secs.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.try_make_timestamp_ltz`.
pub fn try_make_timestamp_ltz(
    years: Column,
    months: Column,
    days: Column,
    hours: Column,
    mins: Column,
    secs: Column,
) -> Column {
    func(
        "try_make_timestamp_ltz",
        vec![
            years.expression().clone(),
            months.expression().clone(),
            days.expression().clone(),
            hours.expression().clone(),
            mins.expression().clone(),
            secs.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.try_make_timestamp_ntz`.
pub fn try_make_timestamp_ntz(
    years: Column,
    months: Column,
    days: Column,
    hours: Column,
    mins: Column,
    secs: Column,
) -> Column {
    func(
        "try_make_timestamp_ntz",
        vec![
            years.expression().clone(),
            months.expression().clone(),
            days.expression().clone(),
            hours.expression().clone(),
            mins.expression().clone(),
            secs.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.try_to_date`.
pub fn try_to_date(col: Column) -> Column {
    func("try_to_date", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.try_to_time`.
pub fn try_to_time(str: Column) -> Column {
    func("try_to_time", vec![str.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.try_variant_array_append`.
pub fn try_variant_array_append(v: Column, path: Column, value: Column) -> Column {
    func(
        "try_variant_array_append",
        vec![
            v.expression().clone(),
            path.expression().clone(),
            value.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.try_variant_insert`.
pub fn try_variant_insert(v: Column, path: Column, value: Column) -> Column {
    func(
        "try_variant_insert",
        vec![
            v.expression().clone(),
            path.expression().clone(),
            value.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.try_variant_set`.
pub fn try_variant_set(v: Column, path: Column, value: Column) -> Column {
    func(
        "try_variant_set",
        vec![
            v.expression().clone(),
            path.expression().clone(),
            value.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.unix_nanos`.
pub fn unix_nanos(col: Column) -> Column {
    func("unix_nanos", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.uuid`.
///
/// The reference client seeds the expression with a (random) `long` argument so
/// the result is stable per query plan; we emit a fixed seed literal, matching
/// the `rand`/`randn` builders. The seed is treated as run-to-run noise by the
/// golden test (see `RANDOM_SEED_FUNCS`).
pub fn uuid() -> Column {
    func("uuid", vec![lit_long(1214072022411175128)])
}

/// Mirrors `pyspark.sql.functions.variant_array_append`.
pub fn variant_array_append(v: Column, path: Column, value: Column) -> Column {
    func(
        "variant_array_append",
        vec![
            v.expression().clone(),
            path.expression().clone(),
            value.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.variant_delete`.
pub fn variant_delete<C: Into<Column>>(v: Column, paths: impl IntoIterator<Item = C>) -> Column {
    let mut args = vec![v.expression().clone()];
    args.extend(paths.into_iter().map(|c| c.into().expression().clone()));
    func("variant_delete", args)
}

/// Mirrors `pyspark.sql.functions.variant_insert`.
pub fn variant_insert(v: Column, path: Column, value: Column) -> Column {
    func(
        "variant_insert",
        vec![
            v.expression().clone(),
            path.expression().clone(),
            value.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.variant_set`.
pub fn variant_set(v: Column, path: Column, value: Column) -> Column {
    func(
        "variant_set",
        vec![
            v.expression().clone(),
            path.expression().clone(),
            value.expression().clone(),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.vector_avg`.
pub fn vector_avg(col: Column) -> Column {
    func("vector_avg", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.vector_cosine_similarity`.
pub fn vector_cosine_similarity(left: Column, right: Column) -> Column {
    func(
        "vector_cosine_similarity",
        vec![left.expression().clone(), right.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.vector_inner_product`.
pub fn vector_inner_product(left: Column, right: Column) -> Column {
    func(
        "vector_inner_product",
        vec![left.expression().clone(), right.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.vector_l2_distance`.
pub fn vector_l2_distance(left: Column, right: Column) -> Column {
    func(
        "vector_l2_distance",
        vec![left.expression().clone(), right.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.vector_norm`.
pub fn vector_norm(vector: Column) -> Column {
    func("vector_norm", vec![vector.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.vector_normalize`.
pub fn vector_normalize(vector: Column) -> Column {
    func("vector_normalize", vec![vector.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.vector_sum`.
pub fn vector_sum(col: Column) -> Column {
    func("vector_sum", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.tuple_difference_double`.
pub fn tuple_difference_double(col1: Column, col2: Column) -> Column {
    func(
        "tuple_difference_double",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_difference_integer`.
pub fn tuple_difference_integer(col1: Column, col2: Column) -> Column {
    func(
        "tuple_difference_integer",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_difference_theta_double`.
pub fn tuple_difference_theta_double(col1: Column, col2: Column) -> Column {
    func(
        "tuple_difference_theta_double",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_difference_theta_integer`.
pub fn tuple_difference_theta_integer(col1: Column, col2: Column) -> Column {
    func(
        "tuple_difference_theta_integer",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_intersection_agg_double`.
pub fn tuple_intersection_agg_double(col: Column) -> Column {
    func(
        "tuple_intersection_agg_double",
        vec![col.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_intersection_agg_integer`.
pub fn tuple_intersection_agg_integer(col: Column) -> Column {
    func(
        "tuple_intersection_agg_integer",
        vec![col.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_intersection_double`.
pub fn tuple_intersection_double(col1: Column, col2: Column) -> Column {
    func(
        "tuple_intersection_double",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_intersection_integer`.
pub fn tuple_intersection_integer(col1: Column, col2: Column) -> Column {
    func(
        "tuple_intersection_integer",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_intersection_theta_double`.
pub fn tuple_intersection_theta_double(col1: Column, col2: Column) -> Column {
    func(
        "tuple_intersection_theta_double",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_intersection_theta_integer`.
pub fn tuple_intersection_theta_integer(col1: Column, col2: Column) -> Column {
    func(
        "tuple_intersection_theta_integer",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_sketch_agg_double`.
///
/// The reference client materializes the default `nom_entries` (lgK = 12) and
/// `mode` ("sum") arguments into the call, so we emit them here too for parity.
pub fn tuple_sketch_agg_double(key: Column, summary: Column) -> Column {
    func(
        "tuple_sketch_agg_double",
        vec![
            key.expression().clone(),
            summary.expression().clone(),
            lit_int(12),
            lit_str("sum"),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_sketch_agg_integer`.
///
/// As with the double variant, the reference client materializes the default
/// `nom_entries` (lgK = 12) and `mode` ("sum") arguments.
pub fn tuple_sketch_agg_integer(key: Column, summary: Column) -> Column {
    func(
        "tuple_sketch_agg_integer",
        vec![
            key.expression().clone(),
            summary.expression().clone(),
            lit_int(12),
            lit_str("sum"),
        ],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_sketch_estimate_double`.
pub fn tuple_sketch_estimate_double(col: Column) -> Column {
    func(
        "tuple_sketch_estimate_double",
        vec![col.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_sketch_estimate_integer`.
pub fn tuple_sketch_estimate_integer(col: Column) -> Column {
    func(
        "tuple_sketch_estimate_integer",
        vec![col.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_sketch_summary_double`.
pub fn tuple_sketch_summary_double(col: Column) -> Column {
    func(
        "tuple_sketch_summary_double",
        vec![col.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_sketch_summary_integer`.
pub fn tuple_sketch_summary_integer(col: Column) -> Column {
    func(
        "tuple_sketch_summary_integer",
        vec![col.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_sketch_theta_double`.
pub fn tuple_sketch_theta_double(col: Column) -> Column {
    func("tuple_sketch_theta_double", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.tuple_sketch_theta_integer`.
pub fn tuple_sketch_theta_integer(col: Column) -> Column {
    func("tuple_sketch_theta_integer", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.tuple_union_agg_double`.
pub fn tuple_union_agg_double(col: Column) -> Column {
    func("tuple_union_agg_double", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.tuple_union_agg_integer`.
pub fn tuple_union_agg_integer(col: Column) -> Column {
    func("tuple_union_agg_integer", vec![col.expression().clone()])
}

/// Mirrors `pyspark.sql.functions.tuple_union_double`.
pub fn tuple_union_double(col1: Column, col2: Column) -> Column {
    func(
        "tuple_union_double",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_union_integer`.
pub fn tuple_union_integer(col1: Column, col2: Column) -> Column {
    func(
        "tuple_union_integer",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_union_theta_double`.
pub fn tuple_union_theta_double(col1: Column, col2: Column) -> Column {
    func(
        "tuple_union_theta_double",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

/// Mirrors `pyspark.sql.functions.tuple_union_theta_integer`.
pub fn tuple_union_theta_integer(col1: Column, col2: Column) -> Column {
    func(
        "tuple_union_theta_integer",
        vec![col1.expression().clone(), col2.expression().clone()],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_arg_functions() {
        let _cume = cume_dist();
        let _curdate = curdate();
        let _catalog = current_catalog();
        let _db = current_database();
        let _now = current_date();
        let _schema = current_schema();
        let _ts = current_timestamp();
        let _tz = current_timezone();
        let _user = current_user();
        let _dense = dense_rank();
        let _e_val = e();
        let _input_block = input_file_block_length();
        let _input_start = input_file_block_start();
        let _input_name = input_file_name();
        let _local_ts = localtimestamp();
        let _mono_id = monotonically_increasing_id();
        let _now_fn = now();
        let _pr = percent_rank();
        let _pi_val = pi();
        let _rand_val = rand();
        let _uuid_val = uuid();
    }

    #[test]
    fn test_zero_arg_functions_serialize() {
        let funcs = vec![
            cume_dist(),
            curdate(),
            current_catalog(),
            current_date(),
            dense_rank(),
            e(),
            pi(),
            rand(),
            uuid(),
        ];
        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_interval_builders_serialize() {
        let make_dt = make_dt_interval();
        let _ = make_dt.to_proto();

        let make_i = make_interval();
        let _ = make_i.to_proto();

        let make_ym = make_ym_interval();
        let _ = make_ym.to_proto();
    }

    #[test]
    fn test_math_functions_serialize() {
        let test_col = col("test");
        let funcs = vec![
            abs(test_col.clone()),
            acos(test_col.clone()),
            acosh(test_col.clone()),
            asin(test_col.clone()),
            asinh(test_col.clone()),
            atan(test_col.clone()),
            atanh(test_col.clone()),
            cbrt(test_col.clone()),
            ceil(test_col.clone()),
            cos(test_col.clone()),
            cosh(test_col.clone()),
            exp(test_col.clone()),
            expm1(test_col.clone()),
            floor(test_col.clone()),
            log(test_col.clone()),
            log10(test_col.clone()),
            log1p(test_col.clone()),
            log2(test_col.clone()),
            rint(test_col.clone()),
            round(test_col.clone()),
            signum(test_col.clone()),
            sin(test_col.clone()),
            sinh(test_col.clone()),
            sqrt(test_col.clone()),
            tan(test_col.clone()),
            tanh(test_col.clone()),
        ];
        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_string_functions_serialize() {
        let test_col = col("test");
        let funcs = vec![
            ascii(test_col.clone()),
            bit_count(test_col.clone()),
            char_length(test_col.clone()),
            character_length(test_col.clone()),
            initcap(test_col.clone()),
            lower(test_col.clone()),
            ltrim(test_col.clone()),
            octet_length(test_col.clone()),
            reverse(test_col.clone()),
            rtrim(test_col.clone()),
            size(test_col.clone()),
            soundex(test_col.clone()),
            upper(test_col.clone()),
        ];
        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_col_patterns() {
        let c_col = col("my_column");
        match c_col.expression() {
            Expression::ColumnReference(_) => {}
            _ => panic!("Expected ColumnReference"),
        }

        let star = col("*");
        match star.expression() {
            Expression::UnresolvedStar(None) => {}
            _ => panic!("Expected UnresolvedStar(None)"),
        }

        let table_star = col("t.*");
        match table_star.expression() {
            Expression::UnresolvedStar(Some(_)) => {}
            _ => panic!("Expected UnresolvedStar with prefix"),
        }
    }

    #[test]
    fn test_expr_function_serialize() {
        let e = expr("SELECT 1");
        match e.expression() {
            Expression::SQLExpression(s) => {
                assert_eq!(s, "SELECT 1");
            }
            _ => panic!("Expected SQLExpression"),
        }
        let _ = e.to_proto();
    }

    #[test]
    fn test_variadic_functions_serialize() {
        let c1 = col("a");
        let c2 = col("b");
        let c3 = col("c");

        let arr = array(vec![c1.clone(), c2.clone(), c3.clone()]);
        let _ = arr.to_proto();

        let concat_res = concat(vec![c1.clone(), c2.clone()]);
        let _ = concat_res.to_proto();

        let zip = arrays_zip(vec![c1.clone(), c2.clone()]);
        let _ = zip.to_proto();

        let map_res = create_map(vec![c1, c2, c3]);
        let _ = map_res.to_proto();
    }

    #[test]
    fn test_sort_functions_serialize() {
        let test_col = col("test");
        let funcs = vec![
            asc(test_col.clone()),
            asc_nulls_first(test_col.clone()),
            asc_nulls_last(test_col.clone()),
            desc(test_col.clone()),
            desc_nulls_first(test_col.clone()),
            desc_nulls_last(test_col.clone()),
        ];
        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_window_functions_serialize() {
        let test_col = col("test");
        let w1 = window(test_col.clone(), "1 minute");
        let _ = w1.to_proto();

        let w2 =
            window_with_slide_and_start(test_col.clone(), "10 minutes", "5 minutes", "0 minutes");
        let _ = w2.to_proto();
    }

    #[test]
    fn test_udf_call_serialize() {
        let c1 = col("a");
        let c2 = col("b");

        let udf_res = call_udf("my_udf", vec![c1.clone(), c2.clone()]);
        let _ = udf_res.to_proto();

        let cf = call_function("func_name", vec![c1, c2]);
        let _ = cf.to_proto();
    }

    #[test]
    fn test_two_arg_functions_serialize() {
        let c1 = col("a");
        let c2 = col("b");

        let funcs = vec![
            pow(c1.clone(), c2.clone()),
            hypot(c1.clone(), c2.clone()),
            atan2(c1.clone(), c2.clone()),
            datediff(c1.clone(), c2.clone()),
            from_utc_timestamp(c1.clone(), c2.clone()),
            to_utc_timestamp(c1.clone(), c2.clone()),
            try_add(c1.clone(), c2.clone()),
            try_divide(c1.clone(), c2.clone()),
            try_multiply(c1.clone(), c2.clone()),
            try_subtract(c1.clone(), c2.clone()),
        ];

        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_type_cast_serialize() {
        let test_col = col("test");
        let fmt_col = col("fmt");
        let funcs = vec![
            bitwiseNOT(test_col.clone()),
            bool_and(test_col.clone()),
            bool_or(test_col.clone()),
            to_date(test_col.clone()),
            to_timestamp(test_col.clone()),
            to_char(test_col.clone(), fmt_col.clone()),
            to_varchar(test_col.clone(), fmt_col.clone()),
            to_binary(test_col.clone()),
            unix_date(test_col.clone()),
            unix_seconds(test_col.clone()),
            unix_millis(test_col.clone()),
            unix_micros(test_col.clone()),
        ];

        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_hash_functions_serialize() {
        let test_col = col("test");
        let c1 = col("a");
        let c2 = col("b");

        let funcs = vec![
            md5(test_col.clone()),
            sha1(test_col.clone()),
            sha2(test_col.clone(), 256),
            crc32(test_col.clone()),
        ];

        for col in funcs {
            let _ = col.to_proto();
        }

        let hash_res = hash(vec![c1, c2]);
        let _ = hash_res.to_proto();
    }

    #[test]
    fn test_encoding_serialize() {
        let test_col = col("test");
        let funcs = vec![
            bin(test_col.clone()),
            hex(test_col.clone()),
            unhex(test_col.clone()),
            base64(test_col.clone()),
            unbase64(test_col.clone()),
        ];

        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_datetime_serialize() {
        let test_col = col("test");
        let funcs = vec![
            dayofmonth(test_col.clone()),
            dayofweek(test_col.clone()),
            dayofyear(test_col.clone()),
            month(test_col.clone()),
            quarter(test_col.clone()),
            dayofmonth(test_col.clone()),
            hour(test_col.clone()),
            minute(test_col.clone()),
            second(test_col.clone()),
            year(test_col.clone()),
        ];

        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_array_functions_serialize() {
        let test_col = col("test");
        let other_col = col("other");

        let funcs = vec![
            array_contains(test_col.clone(), other_col.clone()),
            array_intersect(test_col.clone(), other_col.clone()),
            array_max(test_col.clone()),
            array_min(test_col.clone()),
            array_position(test_col.clone(), other_col.clone()),
            array_remove(test_col.clone(), other_col.clone()),
            array_sort(test_col.clone()),
            array_distinct(test_col.clone()),
            array_compact(test_col.clone()),
        ];

        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_map_functions_serialize() {
        let test_col = col("test");
        let other_col = col("other");

        let funcs = vec![
            map_contains_key(test_col.clone(), other_col.clone()),
            map_keys(test_col.clone()),
            map_values(test_col.clone()),
            map_from_arrays(test_col.clone(), other_col.clone()),
            element_at(test_col.clone(), other_col.clone()),
            try_element_at(test_col.clone(), other_col.clone()),
        ];

        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_string_ops_serialize() {
        let test_col = col("test");
        let col1 = col("a");
        let col2 = col("b");
        let col3 = col("c");

        let funcs = vec![
            substring(test_col.clone(), col1.clone(), col2.clone()),
            instr(test_col.clone(), col1.clone()),
            rpad(test_col.clone(), col1.clone(), col2.clone()),
            lpad(test_col.clone(), col1.clone(), col2.clone()),
            split(test_col.clone(), col1.clone()),
            regexp_extract(test_col.clone(), col1.clone(), col3.clone()),
            regexp_like(test_col.clone(), col1.clone()),
        ];

        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_explode_serialize() {
        let test_col = col("test");
        let funcs = vec![
            explode(test_col.clone()),
            explode_outer(test_col.clone()),
            posexplode(test_col.clone()),
            posexplode_outer(test_col.clone()),
        ];

        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_json_serialize() {
        let test_col = col("test");
        let other_col = col("other");
        let funcs = vec![
            get_json_object(test_col.clone(), other_col.clone()),
            json_array_length(test_col.clone()),
            json_object_keys(test_col.clone()),
            parse_json(test_col.clone()),
            from_json(test_col.clone(), other_col.clone()),
            to_json(test_col.clone()),
            schema_of_json(test_col.clone()),
        ];

        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_aggregation_serialize() {
        let test_col = col("test");
        let funcs = vec![
            count_distinct(test_col.clone()),
            approx_count_distinct(test_col.clone()),
            skewness(test_col.clone()),
            kurtosis(test_col.clone()),
            var_pop(test_col.clone()),
            var_samp(test_col.clone()),
            stddev(test_col.clone()),
            stddev_pop(test_col.clone()),
            stddev_samp(test_col.clone()),
        ];

        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_shift_functions_serialize() {
        let test_col = col("test");
        let col_n = col("n");
        let funcs = vec![
            shiftLeft(test_col.clone(), col_n.clone()),
            shiftRight(test_col.clone(), col_n.clone()),
            shiftRightUnsigned(test_col.clone(), col_n.clone()),
        ];

        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_misc_functions_serialize() {
        let test_col = col("test");
        let funcs = vec![
            degrees(test_col.clone()),
            radians(test_col.clone()),
            bround(test_col.clone()),
            url_encode(test_col.clone()),
            url_decode(test_col.clone()),
            try_to_binary(test_col.clone()),
            try_url_decode(test_col.clone()),
        ];

        for col in funcs {
            let _ = col.to_proto();
        }
    }

    #[test]
    fn test_nested_calls() {
        let inner = col("x");
        let outer = sqrt(inner);
        let _ = outer.to_proto();

        let nested = ceil(sqrt(col("y")));
        let _ = nested.to_proto();

        let complex = floor(abs(log(col("z"))));
        let _ = complex.to_proto();
    }

    #[test]
    fn test_comprehensive_coverage() {
        let c1 = col("c1");
        let c2 = col("c2");
        let c3 = col("c3");

        // Test one-argument functions
        let one_arg_funcs = vec![
            abs(c1.clone()),
            acos(c1.clone()),
            acosh(c1.clone()),
            any_value(c1.clone()),
            approxCountDistinct(c1.clone()),
            approx_count_distinct(c1.clone()),
            array_agg(c1.clone()),
            array_compact(c1.clone()),
            array_distinct(c1.clone()),
            array_max(c1.clone()),
            array_min(c1.clone()),
            array_size(c1.clone()),
            array_sort(c1.clone()),
            asc(c1.clone()),
            asc_nulls_first(c1.clone()),
            asc_nulls_last(c1.clone()),
            ascii(c1.clone()),
            asin(c1.clone()),
            asinh(c1.clone()),
            assert_true(c1.clone()),
            atan(c1.clone()),
            atanh(c1.clone()),
            avg(c1.clone()),
            base64(c1.clone()),
            bin(c1.clone()),
            bit_and(c1.clone()),
            bit_count(c1.clone()),
            bit_length(c1.clone()),
            bit_or(c1.clone()),
            bit_xor(c1.clone()),
            bitmap_and_agg(c1.clone()),
            bitmap_bit_position(c1.clone()),
            bitmap_bucket_number(c1.clone()),
            bitmap_construct_agg(c1.clone()),
            bitmap_count(c1.clone()),
            bitmap_or_agg(c1.clone()),
            bitwiseNOT(c1.clone()),
            bitwise_not(c1.clone()),
            bool_and(c1.clone()),
            bool_or(c1.clone()),
            bround(c1.clone()),
            btrim(c1.clone()),
            cardinality(c1.clone()),
            cbrt(c1.clone()),
            ceil(c1.clone()),
            ceiling(c1.clone()),
            char(c1.clone()),
            char_length(c1.clone()),
            character_length(c1.clone()),
            chr(c1.clone()),
            collation(c1.clone()),
            collect_list(c1.clone()),
            collect_set(c1.clone()),
            concat_ws(c1.clone()),
            cos(c1.clone()),
            cosh(c1.clone()),
            cot(c1.clone()),
            count(c1.clone()),
            countDistinct(c1.clone()),
            count_distinct(c1.clone()),
            count_if(c1.clone()),
            counter_diff(c1.clone()),
            crc32(c1.clone()),
            csc(c1.clone()),
            date_from_unix_date(c1.clone()),
            day(c1.clone()),
            dayname(c1.clone()),
            dayofmonth(c1.clone()),
            dayofweek(c1.clone()),
            dayofyear(c1.clone()),
            days(c1.clone()),
            degrees(c1.clone()),
            desc(c1.clone()),
            desc_nulls_first(c1.clone()),
            desc_nulls_last(c1.clone()),
            every(c1.clone()),
            exp(c1.clone()),
            explode(c1.clone()),
            explode_outer(c1.clone()),
            expm1(c1.clone()),
            factorial(c1.clone()),
            first(c1.clone()),
            first_value(c1.clone()),
            flatten(c1.clone()),
            floor(c1.clone()),
            format_string(c1.clone()),
            grouping(c1.clone()),
            hex(c1.clone()),
            hll_sketch_agg(c1.clone()),
            hll_sketch_estimate(c1.clone()),
            hll_union_agg(c1.clone()),
            hour(c1.clone()),
            hours(c1.clone()),
            initcap(c1.clone()),
            inline(c1.clone()),
        ];
        for f in one_arg_funcs {
            let _ = f.to_proto();
        }

        // Test two-argument functions
        let two_arg_funcs = vec![
            add_months(c1.clone(), c2.clone()),
            aes_decrypt(c1.clone(), c2.clone()),
            aes_encrypt(c1.clone(), c2.clone()),
            approx_count_distinct_rsd(c1.clone(), c2.clone()),
            approx_percentile(c1.clone(), c2.clone()),
            array_append(c1.clone(), c2.clone()),
            array_contains(c1.clone(), c2.clone()),
            array_except(c1.clone(), c2.clone()),
            array_intersect(c1.clone(), c2.clone()),
            array_join(c1.clone(), c2.clone()),
            array_position(c1.clone(), c2.clone()),
            array_prepend(c1.clone(), c2.clone()),
            array_remove(c1.clone(), c2.clone()),
            array_repeat(c1.clone(), c2.clone()),
            array_union(c1.clone(), c2.clone()),
            arrays_overlap(c1.clone(), c2.clone()),
            atan2(c1.clone(), c2.clone()),
            bit_get(c1.clone(), c2.clone()),
            bround_scale(c1.clone(), c2.clone()),
            bucket(c1.clone(), c2.clone()),
            cast(c1.clone(), c2.clone()),
            ceil_scale(c1.clone(), c2.clone()),
            collate(c1.clone(), c2.clone()),
            contains(c1.clone(), c2.clone()),
            corr(c1.clone(), c2.clone()),
            covar_pop(c1.clone(), c2.clone()),
            covar_samp(c1.clone(), c2.clone()),
            date_add(c1.clone(), c2.clone()),
            date_diff(c1.clone(), c2.clone()),
            date_format(c1.clone(), c2.clone()),
            date_part(c1.clone(), c2.clone()),
            date_sub(c1.clone(), c2.clone()),
            date_trunc(c1.clone(), c2.clone()),
            dateadd(c1.clone(), c2.clone()),
            datediff(c1.clone(), c2.clone()),
            datepart(c1.clone(), c2.clone()),
            decode(c1.clone(), c2.clone()),
            element_at(c1.clone(), c2.clone()),
            encode(c1.clone(), c2.clone()),
            endswith(c1.clone(), c2.clone()),
            equal_null(c1.clone(), c2.clone()),
            extract(c1.clone(), c2.clone()),
            find_in_set(c1.clone(), c2.clone()),
            floor_scale(c1.clone(), c2.clone()),
            format_number(c1.clone(), c2.clone()),
            from_csv(c1.clone(), c2.clone()),
            from_json(c1.clone(), c2.clone()),
            from_utc_timestamp(c1.clone(), c2.clone()),
            from_xml(c1.clone(), c2.clone()),
            get(c1.clone(), c2.clone()),
            get_json_object(c1.clone(), c2.clone()),
            getbit(c1.clone(), c2.clone()),
            greatest(c1.clone(), c2.clone()),
            histogram_numeric(c1.clone(), c2.clone()),
            hll_union(c1.clone(), c2.clone()),
            hmac(c1.clone(), c2.clone()),
            hypot(c1.clone(), c2.clone()),
            ifnull(c1.clone(), c2.clone()),
            ilike(c1.clone(), c2.clone()),
            instr(c1.clone(), c2.clone()),
            jaro_winkler_similarity(c1.clone(), c2.clone()),
            json_tuple(c1.clone(), c2.clone()),
            kll_sketch_get_quantile_bigint(c1.clone(), c2.clone()),
            kll_sketch_get_quantile_double(c1.clone(), c2.clone()),
            kll_sketch_get_quantile_float(c1.clone(), c2.clone()),
            kll_sketch_get_rank_bigint(c1.clone(), c2.clone()),
            kll_sketch_get_rank_double(c1.clone(), c2.clone()),
            kll_sketch_get_rank_float(c1.clone(), c2.clone()),
            kll_sketch_merge_bigint(c1.clone(), c2.clone()),
            kll_sketch_merge_double(c1.clone(), c2.clone()),
            kll_sketch_merge_float(c1.clone(), c2.clone()),
            least(c1.clone(), c2.clone()),
            left(c1.clone(), c2.clone()),
            levenshtein(c1.clone(), c2.clone()),
            like(c1.clone(), c2.clone()),
            locate(c1.clone(), c2.clone()),
            ltrim_with(c1.clone(), c2.clone()),
            map_contains_key(c1.clone(), c2.clone()),
            map_from_arrays(c1.clone(), c2.clone()),
            max_by(c1.clone(), c2.clone()),
            min_by(c1.clone(), c2.clone()),
            months_between(c1.clone(), c2.clone()),
            nanvl(c1.clone(), c2.clone()),
            next_day(c1.clone(), c2.clone()),
            nth_value(c1.clone(), c2.clone()),
            nullif(c1.clone(), c2.clone()),
            nvl(c1.clone(), c2.clone()),
            parse_url(c1.clone(), c2.clone()),
            percentile(c1.clone(), c2.clone()),
            percentile_approx(c1.clone(), c2.clone()),
            pmod(c1.clone(), c2.clone()),
            position(c1.clone(), c2.clone()),
            pow(c1.clone(), c2.clone()),
            power(c1.clone(), c2.clone()),
            regexp(c1.clone(), c2.clone()),
            regexp_count(c1.clone(), c2.clone()),
            regexp_extract_all(c1.clone(), c2.clone()),
            regexp_instr(c1.clone(), c2.clone()),
        ];
        for f in two_arg_funcs {
            let _ = f.to_proto();
        }

        // Test three-argument functions
        let three_arg_funcs = vec![
            array_insert(c1.clone(), c2.clone(), c3.clone()),
            conv(c1.clone(), c2.clone(), c3.clone()),
            convert_timezone(c1.clone(), c2.clone(), c3.clone()),
            count_min_sketch(c1.clone(), c2.clone(), c3.clone()),
            lpad(c1.clone(), c2.clone(), c3.clone()),
            make_date(c1.clone(), c2.clone(), c3.clone()),
            make_time(c1.clone(), c2.clone(), c3.clone()),
            nvl2(c1.clone(), c2.clone(), c3.clone()),
            overlay(c1.clone(), c2.clone(), c3.clone()),
            regexp_extract(c1.clone(), c2.clone(), c3.clone()),
            regexp_replace(c1.clone(), c2.clone(), c3.clone()),
            rpad(c1.clone(), c2.clone(), c3.clone()),
            slice(c1.clone(), c2.clone(), c3.clone()),
            split_part(c1.clone(), c2.clone(), c3.clone()),
            substring(c1.clone(), c2.clone(), c3.clone()),
            substring_index(c1.clone(), c2.clone(), c3.clone()),
            time_diff(c1.clone(), c2.clone(), c3.clone()),
            timestamp_add(c1.clone(), c2.clone(), c3.clone()),
            timestamp_diff(c1.clone(), c2.clone(), c3.clone()),
            translate(c1.clone(), c2.clone(), c3.clone()),
            try_variant_array_append(c1.clone(), c2.clone(), c3.clone()),
            try_variant_get(c1.clone(), c2.clone(), c3.clone()),
            try_variant_insert(c1.clone(), c2.clone(), c3.clone()),
            try_variant_set(c1.clone(), c2.clone(), c3.clone()),
            variant_array_append(c1.clone(), c2.clone(), c3.clone()),
        ];
        for f in three_arg_funcs {
            let _ = f.to_proto();
        }
    }
}
