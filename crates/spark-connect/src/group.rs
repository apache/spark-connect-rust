//! GroupedData implementation for aggregations.
//!
//! Mirroring `pyspark.sql.GroupedData`.

use std::sync::Arc;

use spark_connect_core::error::{Result, SparkError, SparkErrorKind};

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::expression::Expression;
use crate::plan::{AggregateGroupType, LogicalPlan};
use crate::types::DataType;
use crate::udf::CommonInlineUserDefinedFunctionExpression;

/// Convert a [`crate::row::Value`] into a literal [`Expression`] of a pivot value. Each value
/// keeps its own type, as the literal of Spark's pivot (and of the Scala client's `lit`), so a
/// float is named as the pivot without values names it ("1.1", not the text of its double), and
/// a NULL is an error, as PySpark's `GroupedData.pivot` rejects a None.
fn value_to_lit_expr(v: crate::row::Value) -> Result<Expression> {
    use crate::expression::LiteralExpression as L;
    use crate::row::Value;
    let lit = match v {
        Value::Null => return Err(not_a_pivot_value("NoneType")),
        Value::Bool(b) => L::boolean(b),
        Value::Byte(x) => L::Byte(i32::from(x)),
        Value::Short(x) => L::Short(i32::from(x)),
        Value::Integer(x) => L::int(x),
        Value::Long(x) => L::long(x),
        Value::Float(x) => L::Float(x),
        Value::Double(x) => L::double(x),
        Value::String(s) => L::string(s),
        Value::Binary(b) => L::binary(b),
        Value::Date(d) => L::Date(d),
        Value::Timestamp(t) => L::Timestamp(t),
        Value::TimestampNtz(t) => L::TimestampNtz(t),
        // `TimeType()` of PySpark has a precision of 6.
        Value::Time(nano) => L::Time { nano, precision: 6 },
        Value::DayTimeInterval(v) => L::DayTimeInterval(v),
        Value::YearMonthInterval(months) => L::YearMonthInterval(months),
        Value::Decimal {
            value,
            precision,
            scale,
        } => {
            let (precision, scale) = match (precision, scale) {
                (Some(precision), Some(scale)) => (precision, scale),
                // As a decimal of the text: its digits and fraction, as `Decimal(text)` of Spark.
                _ => decimal_type_of_text(&value),
            };
            L::Decimal {
                value,
                precision,
                scale,
            }
        }
        Value::List(_) => return Err(not_a_pivot_value("list")),
        Value::Map(_) => return Err(not_a_pivot_value("dict")),
        Value::Struct(_) => return Err(not_a_pivot_value("Row")),
        Value::Variant { .. } => return Err(not_a_pivot_value("Variant")),
    };
    Ok(Expression::literal(lit))
}

/// The `NOT_EXPECTED_TYPE` error of a pivot value of the type `arg_type`, named as Python names
/// it, as PySpark's `GroupedData.pivot` raises it. The expected types are those of the values
/// that the Rust API takes, which are more than PySpark's bools, floats, ints and strs.
fn not_a_pivot_value(arg_type: &str) -> SparkError {
    SparkError::classed(
        SparkErrorKind::TypeError,
        "NOT_EXPECTED_TYPE",
        &[
            ("arg_name", "value"),
            ("expected_type", PIVOT_VALUE_TYPES),
            ("arg_type", arg_type),
        ],
    )
}

/// The kinds of the pivot values of the Rust API, as `NOT_EXPECTED_TYPE` names them.
const PIVOT_VALUE_TYPES: &str =
    "a bool, a number, a string, bytes, a date, a timestamp, a time or an interval";

/// The precision and scale of a decimal text, such as `"1.50"` (3 and 2) or `"1E+3"` (1 and -3),
/// as Java's `BigDecimal` of the text has them. Text that is not a decimal gives `(10, 0)`, and
/// the server rejects it.
fn decimal_type_of_text(text: &str) -> (i32, i32) {
    let (mantissa, exponent) = match text.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().ok()),
        None => (text, Some(0)),
    };
    let Some(exponent) = exponent else {
        return (10, 0);
    };
    let unsigned = mantissa.trim_start_matches(['-', '+']);
    let (int_part, frac_part) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    let digits = format!("{int_part}{frac_part}");
    if !digits.bytes().all(|b| b.is_ascii_digit()) || digits.is_empty() {
        return (10, 0);
    }
    let unscaled = digits.trim_start_matches('0');
    let precision = unscaled.len().max(1) as i32;
    (precision, frac_part.len() as i32 - exponent)
}

/// Grouped data for performing aggregations.
///
/// Mirrors `pyspark.sql.GroupedData`.
#[derive(Clone)]
pub struct GroupedData {
    dataframe: DataFrame,
    group_cols: Vec<Column>,
    group_type: AggregateGroupType,
    /// Pivot column, set by [`GroupedData::pivot`].
    pivot_col: Option<Expression>,
    /// Explicit pivot values (literals), set by [`GroupedData::pivot`].
    pivot_values: Vec<Expression>,
    /// Explicit grouping sets, set by [`GroupedData::new_grouping_sets`]. Each inner
    /// Vec is one grouping set. Only meaningful when `group_type == GroupingSets`.
    grouping_sets: Vec<Vec<Column>>,
}

impl GroupedData {
    /// The grouping columns, as PySpark Connect's `GroupedData._grouping_cols`.
    pub fn grouping_columns(&self) -> &[Column] {
        &self.group_cols
    }

    /// The type of the grouping, as PySpark Connect's `GroupedData._group_type`.
    pub fn group_type(&self) -> AggregateGroupType {
        self.group_type
    }

    /// The DataFrame that is grouped, as PySpark Connect's `GroupedData._df`.
    pub fn dataframe(&self) -> &DataFrame {
        &self.dataframe
    }

    /// Create a new GroupedData.
    pub(crate) fn new(
        dataframe: DataFrame,
        group_cols: Vec<Column>,
        group_type: AggregateGroupType,
    ) -> Self {
        GroupedData {
            dataframe,
            group_cols,
            group_type,
            pivot_col: None,
            pivot_values: vec![],
            grouping_sets: vec![],
        }
    }

    /// Create a GroupedData for `GROUPING SETS`. `grouping_sets` holds each explicit
    /// set of grouping columns; the grouping expressions become the (de-duplicated)
    /// union of all columns referenced across the sets.
    pub(crate) fn new_grouping_sets(dataframe: DataFrame, grouping_sets: Vec<Vec<Column>>) -> Self {
        // Union of all columns across the sets, de-duplicated by their proto encoding
        // (Expression is not Eq/Hash), preserving first-seen order.
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let mut group_cols: Vec<Column> = Vec::new();
        for set in &grouping_sets {
            for col in set {
                let key = prost::Message::encode_to_vec(&col.expression().clone().to_proto());
                if !seen.contains(&key) {
                    seen.push(key);
                    group_cols.push(col.clone());
                }
            }
        }
        Self::new_grouping_sets_of(dataframe, grouping_sets, group_cols)
    }

    /// Grouped data of the grouping sets `grouping_sets` and the grouping columns
    /// `group_cols`, the columns of the result besides the aggregations, as PySpark's
    /// `groupingSets(groupingSets, *cols)`.
    pub(crate) fn new_grouping_sets_of(
        dataframe: DataFrame,
        grouping_sets: Vec<Vec<Column>>,
        group_cols: Vec<Column>,
    ) -> Self {
        GroupedData {
            dataframe,
            group_cols,
            group_type: AggregateGroupType::GroupingSets,
            pivot_col: None,
            pivot_values: vec![],
            grouping_sets,
        }
    }

    /// Pivot on a column, optionally with explicit values (`GroupedData.pivot`).
    ///
    /// Mirrors `df.groupBy(...).pivot(col, values)`. When `values` is `None` the
    /// server computes the distinct values; when supplied they are serialized as
    /// pivot-value literals. As PySpark's, only the grouping of `group_by` pivots: that of
    /// a rollup, a cube, grouping sets or a pivot is the error of [`GroupedData::check_pivot`].
    /// A value of the explicit `values` keeps its type as a literal. A NULL, or a list, map or
    /// struct, is `NOT_EXPECTED_TYPE`, as PySpark rejects it.
    pub fn pivot(
        &self,
        pivot_col: Column,
        values: Option<Vec<crate::row::Value>>,
    ) -> Result<GroupedData> {
        self.check_pivot()?;
        let pivot_values = values
            .unwrap_or_default()
            .into_iter()
            .map(value_to_lit_expr)
            .collect::<Result<_>>()?;
        Ok(GroupedData {
            dataframe: self.dataframe.clone(),
            group_cols: self.group_cols.clone(),
            group_type: AggregateGroupType::Pivot,
            pivot_col: Some(pivot_col.expression().clone()),
            pivot_values,
            grouping_sets: vec![],
        })
    }

    /// Whether this grouping pivots, as PySpark's `GroupedData.pivot` checks before its
    /// arguments: only that of `group_by` does. That of a rollup, a cube or grouping sets is
    /// an `UNSUPPORTED_OPERATION` error of "PIVOT after ROLLUP", "PIVOT after CUBE" or
    /// "PIVOT after GROUPING_SETS", and that of a pivot of "Repeated PIVOT operation".
    pub fn check_pivot(&self) -> Result<()> {
        let operation = match self.group_type {
            AggregateGroupType::GroupBy => return Ok(()),
            AggregateGroupType::Pivot => "Repeated PIVOT operation",
            AggregateGroupType::Rollup => "PIVOT after ROLLUP",
            AggregateGroupType::Cube => "PIVOT after CUBE",
            AggregateGroupType::GroupingSets => "PIVOT after GROUPING_SETS",
        };
        Err(SparkError::classed(
            SparkErrorKind::NotImplementedError,
            "UNSUPPORTED_OPERATION",
            &[("operation", operation)],
        ))
    }

    /// Column names of the underlying (pre-grouping) DataFrame. These are passed as the
    /// map/apply UDF's argument columns so the worker's pandas/Arrow input is named.
    pub fn input_columns(&self) -> Result<Vec<String>> {
        self.dataframe.columns()
    }

    /// The session of the grouped DataFrame.
    pub fn spark_session(&self) -> crate::session::SparkSession {
        self.dataframe.spark_session()
    }

    /// Grouping-key expressions for this grouped data.
    fn grouping_expressions(&self) -> Vec<Expression> {
        self.group_cols
            .iter()
            .map(|col| col.expression().clone())
            .collect()
    }

    /// Apply a pandas UDF to each group (`GroupedData.applyInPandas`).
    ///
    /// `func` is built on the Python side (cloudpickled, eval type
    /// `SQL_GROUPED_MAP_PANDAS_UDF`).
    pub fn apply_in_pandas(&self, func: CommonInlineUserDefinedFunctionExpression) -> DataFrame {
        self.group_map(func)
    }

    /// Apply an Arrow UDF to each group (`GroupedData.applyInArrow`).
    pub fn apply_in_arrow(&self, func: CommonInlineUserDefinedFunctionExpression) -> DataFrame {
        self.group_map(func)
    }

    fn group_map(&self, func: CommonInlineUserDefinedFunctionExpression) -> DataFrame {
        let schema = func.python_udf.output_type.clone();
        let plan = LogicalPlan::GroupMap {
            input: Arc::clone(&self.dataframe.plan),
            grouping_expressions: self.grouping_expressions(),
            func,
            sorting_expressions: vec![],
            initial_input: None,
            initial_grouping_expressions: vec![],
            is_map_groups_with_state: None,
            output_mode: None,
            timeout_conf: None,
            state_schema: None,
            transform_with_state_info: None,
        };
        DataFrame::new_with_schema(self.dataframe.session.clone(), plan, Some(schema))
    }

    /// Apply a stateful pandas UDF to each group (`GroupedData.applyInPandasWithState`).
    ///
    /// `func` (built on the Python side with eval type
    /// `SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE` and carrying the output schema as its
    /// return type) is combined with the state schema, output mode, and timeout in an
    /// `ApplyInPandasWithState` relation, as the reference client builds it.
    pub fn apply_in_pandas_with_state(
        &self,
        func: CommonInlineUserDefinedFunctionExpression,
        state_schema: DataType,
        output_mode: &str,
        timeout_conf: &str,
    ) -> DataFrame {
        let plan = LogicalPlan::ApplyInPandasWithState {
            input: Arc::clone(&self.dataframe.plan),
            grouping_expressions: self.grouping_expressions(),
            output_schema: func.python_udf.output_type.clone(),
            func,
            state_schema,
            output_mode: output_mode.to_string(),
            timeout_conf: timeout_conf.to_string(),
        };
        DataFrame::new(self.dataframe.session.clone(), plan)
    }

    /// `GroupedData.transformWithState` (row output). `func` is the cloudpickled
    /// stateful processor (eval type in the 211-214 range). Its return type is sent as the
    /// output schema, which the server reads, as PySpark sends `outputStructType`.
    pub fn transform_with_state(
        &self,
        func: CommonInlineUserDefinedFunctionExpression,
        output_mode: &str,
        time_mode: &str,
        event_time_column_name: Option<&str>,
        initial_state: Option<&GroupedData>,
    ) -> DataFrame {
        let output_schema = func.python_udf.output_type.clone();
        self.transform_with_state_impl(
            func,
            output_mode,
            time_mode,
            event_time_column_name,
            initial_state,
            Some(output_schema),
        )
    }

    /// `GroupedData.transformWithStateInPandas` (carries the output schema in the
    /// `TransformWithStateInfo`).
    pub fn transform_with_state_in_pandas(
        &self,
        func: CommonInlineUserDefinedFunctionExpression,
        output_schema: DataType,
        output_mode: &str,
        time_mode: &str,
        event_time_column_name: Option<&str>,
        initial_state: Option<&GroupedData>,
    ) -> DataFrame {
        self.transform_with_state_impl(
            func,
            output_mode,
            time_mode,
            event_time_column_name,
            initial_state,
            Some(output_schema),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_with_state_impl(
        &self,
        func: CommonInlineUserDefinedFunctionExpression,
        output_mode: &str,
        time_mode: &str,
        event_time_column_name: Option<&str>,
        initial_state: Option<&GroupedData>,
        output_schema: Option<DataType>,
    ) -> DataFrame {
        let (initial_input, initial_grouping_expressions) = match initial_state {
            Some(gd) => (
                Some(Arc::clone(&gd.dataframe.plan)),
                gd.grouping_expressions(),
            ),
            None => (None, vec![]),
        };
        let plan = LogicalPlan::GroupMap {
            input: Arc::clone(&self.dataframe.plan),
            grouping_expressions: self.grouping_expressions(),
            func,
            sorting_expressions: vec![],
            initial_input,
            initial_grouping_expressions,
            is_map_groups_with_state: None,
            output_mode: Some(output_mode.to_string()),
            timeout_conf: None,
            state_schema: None,
            transform_with_state_info: Some(crate::plan::TransformWithStateInfo {
                time_mode: time_mode.to_string(),
                event_time_column_name: event_time_column_name.map(|s| s.to_string()),
                output_schema,
            }),
        };
        DataFrame::new(self.dataframe.session.clone(), plan)
    }

    /// Cogroup this grouped data with another (`GroupedData.cogroup`).
    pub fn cogroup(&self, other: &GroupedData) -> CoGroupedData {
        CoGroupedData {
            left: self.clone(),
            right: other.clone(),
        }
    }

    /// Perform an aggregation.
    pub fn agg(&self, expressions: Vec<Expression>) -> DataFrame {
        // Convert group columns to expressions
        let grouping_expressions = self
            .group_cols
            .iter()
            .map(|col| col.expression().clone())
            .collect();

        // Explicit grouping sets → each set as its own list of grouping expressions.
        let grouping_sets: Vec<Vec<Expression>> = self
            .grouping_sets
            .iter()
            .map(|set| set.iter().map(|c| c.expression().clone()).collect())
            .collect();

        let plan = LogicalPlan::Aggregate {
            input: Arc::clone(&self.dataframe.plan),
            group_type: self.group_type,
            grouping_expressions,
            aggregate_expressions: expressions,
            pivot_col: self.pivot_col.clone(),
            pivot_values: self.pivot_values.clone(),
            grouping_sets,
        };

        DataFrame::new(self.dataframe.session.clone(), plan)
    }

    /// Count rows in each group, in a column `count`, mirroring `GroupedData.count`, which
    /// aggregates `count(lit(1)).alias("count")`.
    pub fn count(&self) -> DataFrame {
        use crate::functions;
        let count_expr = functions::count(Column::new(Expression::literal(
            crate::expression::LiteralExpression::int(1),
        )))
        .alias("count")
        .expression()
        .clone();

        self.agg(vec![count_expr])
    }

    /// Aggregate the numeric `columns` by `function` ("min", "max", "avg" or "sum") in each
    /// group, mirroring `GroupedData._numeric_agg` of `min`, `max`, `avg`, `mean` and `sum`:
    /// every numeric column of the DataFrame without columns, and `NOT_NUMERIC_COLUMNS` for a
    /// column that is not a numeric one of its schema.
    pub fn numeric_agg(&self, function: &str, columns: Vec<&str>) -> Result<DataFrame> {
        let schema = self.dataframe.schema()?;
        let names: Vec<String> = if columns.is_empty() {
            match &schema {
                DataType::Struct { fields } => fields
                    .iter()
                    .filter(|field| is_numeric(&field.data_type))
                    .map(|field| field.name.clone())
                    .collect(),
                _ => Vec::new(),
            }
        } else {
            let mut invalid = Vec::new();
            for column in &columns {
                if !verify_numeric_col_name(column, &schema)? {
                    invalid.push(crate::expression::python_str_repr(column));
                }
            }
            if !invalid.is_empty() {
                return Err(SparkError::classed(
                    SparkErrorKind::TypeError,
                    "NOT_NUMERIC_COLUMNS",
                    &[("invalid_columns", &format!("[{}]", invalid.join(", ")))],
                ));
            }
            columns.iter().map(|column| column.to_string()).collect()
        };
        let expressions = names
            .iter()
            .map(|name| {
                Expression::function(
                    function,
                    vec![crate::column::col(name).expression().clone()],
                )
            })
            .collect();
        Ok(self.agg(expressions))
    }

    /// Sum values in each group, as `GroupedData.sum`: every numeric column without `columns`.
    pub fn sum(&self, columns: Vec<&str>) -> Result<DataFrame> {
        self.numeric_agg("sum", columns)
    }

    /// Average values in each group, as `GroupedData.avg`: every numeric column without `columns`.
    pub fn avg(&self, columns: Vec<&str>) -> Result<DataFrame> {
        self.numeric_agg("avg", columns)
    }

    /// Minimum values in each group, as `GroupedData.min`: every numeric column without `columns`.
    pub fn min(&self, columns: Vec<&str>) -> Result<DataFrame> {
        self.numeric_agg("min", columns)
    }

    /// Maximum values in each group, as `GroupedData.max`: every numeric column without `columns`.
    pub fn max(&self, columns: Vec<&str>) -> Result<DataFrame> {
        self.numeric_agg("max", columns)
    }

    /// Mean (alias for avg).
    pub fn mean(&self, columns: Vec<&str>) -> Result<DataFrame> {
        self.avg(columns)
    }
}

/// Whether `data_type` is a `NumericType` of PySpark.
fn is_numeric(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Float
            | DataType::Double
            | DataType::Decimal { .. }
    )
}

/// The parts of an attribute name, as PySpark's `parse_attr_name`: split by dots, of
/// backtick-quoted parts too; None for a name that it does not parse.
fn parse_attr_name(name: &str) -> Option<Vec<String>> {
    let chars: Vec<char> = name.chars().collect();
    let n = chars.len();
    let (mut parts, mut part, mut in_backtick, mut i) = (Vec::new(), String::new(), false, 0);
    while i < n {
        let c = chars[i];
        if in_backtick {
            if c == '`' {
                if i + 1 < n && chars[i + 1] == '`' {
                    part.push('`');
                    i += 1;
                } else {
                    in_backtick = false;
                    if i + 1 < n && chars[i + 1] != '.' {
                        return None;
                    }
                }
            } else {
                part.push(c);
            }
        } else if c == '`' {
            if !part.is_empty() {
                return None;
            }
            in_backtick = true;
        } else if c == '.' {
            // As Python's `name[i - 1]`, the last character for the first.
            let previous = chars[(i + n - 1) % n];
            if previous == '.' || i == n - 1 {
                return None;
            }
            parts.push(std::mem::take(&mut part));
        } else {
            part.push(c);
        }
        i += 1;
    }
    if in_backtick {
        return None;
    }
    parts.push(part);
    Some(parts)
}

/// Whether `name` is of a numeric column of `schema`, as PySpark's
/// `verify_numeric_col_name`: by its parts, of the fields of structs and of the structs of
/// arrays. As PySpark's, the name of a struct is an `IndexError`.
fn verify_numeric_col_name(name: &str, schema: &DataType) -> Result<bool> {
    fn verify(parts: &[String], data_type: &DataType) -> Result<bool> {
        if parts.is_empty() && is_numeric(data_type) {
            return Ok(true);
        }
        let fields = match data_type {
            DataType::Struct { fields } => fields,
            DataType::Array { element_type, .. } => match element_type.as_ref() {
                DataType::Struct { fields } => fields,
                _ => return Ok(false),
            },
            _ => return Ok(false),
        };
        let Some((part, rest)) = parts.split_first() else {
            let mut err = SparkError::value_msg("list index out of range");
            err.kind = SparkErrorKind::IndexError;
            return Err(err);
        };
        match fields.iter().find(|field| field.name == *part) {
            Some(field) => verify(rest, &field.data_type),
            None => Ok(false),
        }
    }
    match parse_attr_name(name) {
        Some(parts) if !parts.is_empty() => verify(&parts, schema),
        _ => Ok(false),
    }
}

/// Statistical functions for DataFrames.
///
/// Mirrors `pyspark.sql.DataFrameStatFunctions`.
pub struct StatFunctions {
    dataframe: DataFrame,
}

impl StatFunctions {
    /// The session of these statistics, whose executions they run.
    pub fn spark_session(&self) -> crate::session::SparkSession {
        self.dataframe.spark_session()
    }

    /// Create a new StatFunctions.
    pub(crate) fn new(dataframe: DataFrame) -> Self {
        StatFunctions { dataframe }
    }

    /// Compute a cross-tabulation of two columns.
    pub fn crosstab(&self, col1: &str, col2: &str) -> DataFrame {
        let plan = LogicalPlan::StatCrosstab {
            input: Arc::clone(&self.dataframe.plan),
            col1: col1.to_string(),
            col2: col2.to_string(),
        };
        DataFrame::new(self.dataframe.session.clone(), plan)
    }

    /// Find frequent items.
    pub fn freq_items(&self, columns: Vec<&str>, support: f64) -> DataFrame {
        let plan = LogicalPlan::StatFreqItems {
            input: Arc::clone(&self.dataframe.plan),
            columns: columns.iter().map(|s| s.to_string()).collect(),
            support,
        };
        DataFrame::new(self.dataframe.session.clone(), plan)
    }

    /// Compute approximate quantiles.
    pub fn approx_quantile(
        &self,
        columns: Vec<&str>,
        probabilities: Vec<f64>,
        relative_error: f64,
    ) -> DataFrame {
        let plan = LogicalPlan::StatApproxQuantile {
            input: Arc::clone(&self.dataframe.plan),
            columns: columns.iter().map(|s| s.to_string()).collect(),
            probabilities,
            relative_error,
        };
        DataFrame::new(self.dataframe.session.clone(), plan)
    }

    /// Compute correlation between two columns.
    ///
    /// Executes the stat aggregation and returns the resulting `f64`
    /// (NaN when the correlation is undefined, e.g. an empty relation).
    pub fn corr(&self, col1: &str, col2: &str) -> Result<f64> {
        let plan = LogicalPlan::StatCorr {
            input: Arc::clone(&self.dataframe.plan),
            col1: col1.to_string(),
            col2: col2.to_string(),
        };
        let df = DataFrame::new(self.dataframe.session.clone(), plan);
        Ok(df
            .first_value()?
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NAN))
    }

    /// Compute covariance between two columns.
    ///
    /// Executes the stat aggregation and returns the resulting `f64`.
    pub fn cov(&self, col1: &str, col2: &str) -> Result<f64> {
        let plan = LogicalPlan::StatCov {
            input: Arc::clone(&self.dataframe.plan),
            col1: col1.to_string(),
            col2: col2.to_string(),
        };
        let df = DataFrame::new(self.dataframe.session.clone(), plan);
        Ok(df
            .first_value()?
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NAN))
    }

    /// Sample by values in a column, mirroring `DataFrameStatFunctions.sampleBy(col, fractions,
    /// seed)`: each stratum of `fractions` is a literal, of any type, with its fraction. A
    /// stratum that is not a literal, which the server cannot read, is a
    /// `DISALLOWED_TYPE_FOR_CONTAINER` error, as PySpark raises for a key of another type.
    pub fn sample_by(
        &self,
        col: impl Into<Column>,
        fractions: Vec<(Expression, f64)>,
        seed: Option<i64>,
    ) -> Result<DataFrame> {
        if fractions.iter().any(|(stratum, _)| {
            !matches!(
                stratum.node(),
                crate::expression::ExpressionNode::Literal(_)
            )
        }) {
            return Err(SparkError::classed(
                SparkErrorKind::TypeError,
                "DISALLOWED_TYPE_FOR_CONTAINER",
                &[
                    ("arg_name", "fractions"),
                    ("arg_type", "dict"),
                    ("allowed_types", "literal"),
                    ("item_type", "Column"),
                ],
            ));
        }
        let plan = LogicalPlan::StatSampleBy {
            input: Arc::clone(&self.dataframe.plan),
            col: col.into().expression().clone(),
            fractions,
            seed,
        };
        Ok(DataFrame::new(self.dataframe.session.clone(), plan))
    }
}

/// Methods for handling missing data, accessed via [`DataFrame::na`].
///
/// Mirrors `pyspark.sql.DataFrameNaFunctions`.
pub struct NaFunctions {
    dataframe: DataFrame,
}

impl NaFunctions {
    /// Create a new NaFunctions.
    pub(crate) fn new(dataframe: DataFrame) -> Self {
        NaFunctions { dataframe }
    }

    /// Drop rows containing null values. Mirrors `DataFrameNaFunctions.drop`; see
    /// [`DataFrame::dropna`] for `how`.
    pub fn drop(
        &self,
        how: Option<&str>,
        thresh: Option<i32>,
        subset: Option<Vec<&str>>,
    ) -> Result<DataFrame> {
        self.dataframe.dropna(how, thresh, subset)
    }

    /// Fill null values. Mirrors `DataFrameNaFunctions.fill`.
    pub fn fill(&self, value: i64, subset: Option<Vec<&str>>) -> DataFrame {
        self.dataframe.fillna(value, subset)
    }

    /// Replace values. Mirrors `DataFrameNaFunctions.replace`.
    pub fn replace(
        &self,
        to_replace: Vec<(String, String)>,
        subset: Option<Vec<&str>>,
    ) -> DataFrame {
        self.dataframe.replace(to_replace, subset)
    }

    /// Replace values given as literals. Mirrors `DataFrameNaFunctions.replace`, as
    /// [`DataFrame::replace_literals`].
    pub fn replace_literals(
        &self,
        replacements: Vec<(
            crate::expression::LiteralExpression,
            crate::expression::LiteralExpression,
        )>,
        subset: Option<Vec<&str>>,
    ) -> DataFrame {
        self.dataframe.replace_literals(replacements, subset)
    }
}

/// A pair of cogrouped [`GroupedData`], mirroring `pyspark.sql.PandasCogroupedOps`.
///
/// Created via [`GroupedData::cogroup`].
#[derive(Clone)]
pub struct CoGroupedData {
    left: GroupedData,
    right: GroupedData,
}

impl CoGroupedData {
    /// Column names of both sides (left then right), passed as the cogroup UDF's
    /// argument columns so the worker's two pandas/Arrow inputs are named.
    pub fn input_columns(&self) -> Result<Vec<String>> {
        let mut cols = self.left.input_columns()?;
        cols.extend(self.right.input_columns()?);
        Ok(cols)
    }

    /// The session of the cogrouped DataFrames.
    pub fn spark_session(&self) -> crate::session::SparkSession {
        self.left.spark_session()
    }

    /// Apply a pandas UDF to each cogroup (`cogroup(...).applyInPandas`).
    ///
    /// `func` is built on the Python side (cloudpickled, eval type
    /// `SQL_COGROUPED_MAP_PANDAS_UDF`).
    pub fn apply_in_pandas(&self, func: CommonInlineUserDefinedFunctionExpression) -> DataFrame {
        self.cogroup_map(func)
    }

    /// Apply an Arrow UDF to each cogroup (`cogroup(...).applyInArrow`).
    pub fn apply_in_arrow(&self, func: CommonInlineUserDefinedFunctionExpression) -> DataFrame {
        self.cogroup_map(func)
    }

    fn cogroup_map(&self, func: CommonInlineUserDefinedFunctionExpression) -> DataFrame {
        let schema = func.python_udf.output_type.clone();
        let plan = LogicalPlan::CoGroupMap {
            input: Arc::clone(&self.left.dataframe.plan),
            input_grouping_expressions: self.left.grouping_expressions(),
            other: Arc::clone(&self.right.dataframe.plan),
            other_grouping_expressions: self.right.grouping_expressions(),
            func,
        };
        DataFrame::new_with_schema(self.left.dataframe.session.clone(), plan, Some(schema))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SparkSession;

    fn session() -> SparkSession {
        SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("failed to build session")
    }

    #[test]
    fn numeric_column_names_are_verified_as_pyspark() {
        use crate::types::StructField;
        let field = |name: &str, data_type: DataType| StructField {
            name: name.to_string(),
            data_type,
            nullable: true,
            metadata: Default::default(),
        };
        let schema = DataType::Struct {
            fields: vec![
                field("x", DataType::Integer),
                field(
                    "k",
                    DataType::String {
                        collation: "UTF8_BINARY".to_string(),
                    },
                ),
                field(
                    "st",
                    DataType::Struct {
                        fields: vec![field("a", DataType::Double)],
                    },
                ),
                field(
                    "arr",
                    DataType::Array {
                        element_type: Box::new(DataType::Struct {
                            fields: vec![field("c", DataType::Long)],
                        }),
                        contains_null: true,
                    },
                ),
            ],
        };
        // As PySpark's parse_attr_name of these.
        assert_eq!(
            parse_attr_name("a.`b.c`.`d``e`").unwrap(),
            vec!["a", "b.c", "d`e"]
        );
        assert_eq!(parse_attr_name(".a").unwrap(), vec!["", "a"]);
        assert_eq!(parse_attr_name("`a.b`").unwrap(), vec!["a.b"]);
        for unparsed in ["a..b", "a.", "`a`b", "a`b`", "`a", "a.`b.c`.``d``"] {
            assert_eq!(parse_attr_name(unparsed), None, "{unparsed}");
        }
        for (name, numeric) in [
            ("x", true),
            ("k", false),
            ("st.a", true),
            ("`st`.`a`", true),
            ("arr.c", true),
            ("nope", false),
            ("st..a", false),
        ] {
            assert_eq!(
                verify_numeric_col_name(name, &schema).unwrap(),
                numeric,
                "{name}"
            );
        }
        // As PySpark's, of which a struct is an IndexError.
        let err = verify_numeric_col_name("st", &schema).unwrap_err();
        assert_eq!(err.kind, SparkErrorKind::IndexError);
        assert_eq!(err.message(), "list index out of range");
    }

    #[test]
    fn grouped_and_stat_builders_share_the_source_plan() {
        let spark = session();
        let df = spark.range(10).unwrap();
        let grouped = df.group_by(vec![crate::column::col("id")]);
        assert!(Arc::ptr_eq(&grouped.dataframe.plan, &df.plan));
        let cloned = grouped.clone();
        assert!(Arc::ptr_eq(&cloned.dataframe.plan, &df.plan));
        let count = grouped.count();
        let LogicalPlan::Aggregate { input, .. } = count.plan.as_ref() else {
            panic!("expected aggregate");
        };
        assert!(Arc::ptr_eq(input, &df.plan));

        let stat = df.stat();
        assert!(Arc::ptr_eq(&stat.dataframe.plan, &df.plan));
        let result = stat.crosstab("id", "id");
        let LogicalPlan::StatCrosstab { input, .. } = result.plan.as_ref() else {
            panic!("expected crosstab");
        };
        assert!(Arc::ptr_eq(input, &df.plan));
    }

    #[test]
    fn grouped_udfs_reuse_declared_output_schema() {
        use crate::udf::{eval_type, PythonUDFPayload};

        let spark = session();
        let grouped = spark.range(10).unwrap().group_by(Vec::<Column>::new());
        let cogrouped = grouped.cogroup(&grouped);
        let schema = DataType::from_json(&serde_json::json!({
            "type": "struct",
            "fields": [{
                "name": "result",
                "type": "long",
                "nullable": false,
                "metadata": {"description": "declared output"}
            }]
        }))
        .unwrap();
        let udf = |eval_type| {
            CommonInlineUserDefinedFunctionExpression::new(
                "output".to_string(),
                true,
                vec![],
                PythonUDFPayload::new(schema.clone(), eval_type, vec![], "3.12".to_string()),
            )
        };

        // These accesses must work without a server and use the supplied output
        // schema, which differs from the unresolved input's `id` column.
        for result in [
            grouped.apply_in_pandas(udf(eval_type::SQL_GROUPED_MAP_PANDAS_UDF)),
            grouped.apply_in_arrow(udf(eval_type::SQL_GROUPED_MAP_ARROW_UDF)),
            cogrouped.apply_in_pandas(udf(eval_type::SQL_COGROUPED_MAP_PANDAS_UDF)),
            cogrouped.apply_in_arrow(udf(eval_type::SQL_COGROUPED_MAP_ARROW_UDF)),
        ] {
            assert_eq!(result.schema().unwrap(), schema);
        }
    }

    #[test]
    fn apply_in_pandas_with_state_builds_its_relation() {
        use crate::udf::{eval_type, PythonUDFPayload};
        use spark_connect_proto::relation::RelType;

        let spark = session();
        let grouped = spark
            .range(10)
            .unwrap()
            .group_by(vec![crate::column::col("id")]);
        let output = DataType::from_ddl("id long, c long").unwrap();
        let state = DataType::from_ddl("c long").unwrap();
        let udf = CommonInlineUserDefinedFunctionExpression::new(
            "count".to_string(),
            true,
            vec![],
            PythonUDFPayload::new(
                output.clone(),
                eval_type::SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE,
                vec![],
                "3.12".to_string(),
            ),
        );
        let result = grouped.apply_in_pandas_with_state(udf, state.clone(), "update", "NoTimeout");
        // The relation of the reference client, which the server takes for this function,
        // not a GroupMap, which it rejects.
        let Some(RelType::ApplyInPandasWithState(apply)) = result.plan.to_proto().rel_type else {
            panic!("expected an ApplyInPandasWithState relation");
        };
        assert_eq!(apply.output_schema, output.json());
        assert_eq!(apply.state_schema, state.json());
        assert_eq!(apply.output_mode, "update");
        assert_eq!(apply.timeout_conf, "NoTimeout");
        assert_eq!(apply.grouping_expressions.len(), 1);
        assert!(apply.input.is_some());
        assert_eq!(apply.func.unwrap().function_name, "count");
    }

    #[test]
    fn transform_with_state_sends_its_output_schema() {
        use crate::udf::{eval_type, PythonUDFPayload};
        use spark_connect_proto::relation::RelType;

        let spark = session();
        let grouped = spark
            .range(10)
            .unwrap()
            .group_by(vec![crate::column::col("id")]);
        let output = DataType::from_ddl("id long, c long").unwrap();
        let udf = CommonInlineUserDefinedFunctionExpression::new(
            "count".to_string(),
            true,
            vec![],
            PythonUDFPayload::new(
                output.clone(),
                eval_type::SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_UDF,
                vec![],
                "3.12".to_string(),
            ),
        );
        let result = grouped.transform_with_state(udf, "Update", "None", None, None);
        // As PySpark's, the TransformWithStateInfo carries the output schema, which the
        // server reads as the schema of the rows of the processor.
        let Some(RelType::GroupMap(group_map)) = result.plan.to_proto().rel_type else {
            panic!("expected a GroupMap relation");
        };
        let info = group_map.transform_with_state_info.unwrap();
        assert_eq!(info.output_schema, Some(output.to_proto()));
        assert_eq!(info.time_mode, "None");
        assert_eq!(group_map.output_mode, Some("Update".to_string()));
    }

    #[test]
    fn group_count_plan() {
        let spark = session();
        let df = spark.range(10).unwrap();
        let grouped = df.group_by(vec![crate::column::col("id")]);
        let result = grouped.count();
        match result.plan.as_ref() {
            LogicalPlan::Aggregate {
                group_type: AggregateGroupType::GroupBy,
                ..
            } => {
                // Plan structure is correct
            }
            _ => panic!("expected Aggregate plan with GroupBy"),
        }
    }

    #[test]
    fn group_agg_plan() {
        let spark = session();
        let df = spark.range(10).unwrap();
        let grouped = df.group_by(vec![crate::column::col("id")]);
        let exprs = vec![crate::functions::sum(crate::column::col("id"))
            .expression()
            .clone()];
        let result = grouped.agg(exprs);
        match result.plan.as_ref() {
            LogicalPlan::Aggregate {
                group_type: AggregateGroupType::GroupBy,
                ..
            } => {
                // Plan structure is correct
            }
            _ => panic!("expected Aggregate plan"),
        }
    }

    #[test]
    fn pivot_plan() {
        let spark = session();
        let df = spark.range(10).unwrap();
        let grouped = df.group_by(vec![crate::column::col("id")]);
        let pivot_grouped = grouped.pivot(crate::column::col("category"), None).unwrap();
        assert_eq!(pivot_grouped.group_type, AggregateGroupType::Pivot);
        assert!(pivot_grouped.pivot_col.is_some());
    }

    #[test]
    fn only_the_grouping_of_group_by_pivots() {
        let df = session().range(1).unwrap();
        let id = || vec![crate::column::col("id")];
        let pivot = |grouped: GroupedData| grouped.pivot(crate::column::col("id"), None);
        assert!(pivot(df.group_by(id())).is_ok());
        for (grouped, operation) in [
            (df.rollup(id()), "PIVOT after ROLLUP"),
            (df.cube(id()), "PIVOT after CUBE"),
            (df.grouping_sets(vec![id()]), "PIVOT after GROUPING_SETS"),
            (
                pivot(df.group_by(id())).unwrap(),
                "Repeated PIVOT operation",
            ),
        ] {
            let error = pivot(grouped).err().expect("an error");
            assert_eq!(error.kind, SparkErrorKind::NotImplementedError);
            assert_eq!(
                error.get_condition().as_deref(),
                Some("UNSUPPORTED_OPERATION")
            );
            assert_eq!(
                error.get_message_parameters().unwrap()["operation"],
                operation
            );
        }
    }

    #[test]
    fn pivot_day_time_intervals_are_exact_literals() {
        use spark_connect_proto::{expression::literal::LiteralType, relation::RelType};
        let values = [i64::MIN, -1, 0, i64::MAX];
        let frame = session().range(1).unwrap();
        let pivot = frame
            .group_by(vec![] as Vec<Column>)
            .pivot(
                crate::column::col("interval"),
                Some(
                    values
                        .into_iter()
                        .map(crate::row::Value::DayTimeInterval)
                        .collect(),
                ),
            )
            .unwrap()
            .count();
        let Some(RelType::Aggregate(aggregate)) = pivot.plan.to_proto().rel_type else {
            panic!("expected Aggregate");
        };
        let literals = aggregate.pivot.unwrap().values;
        assert_eq!(literals.len(), values.len());
        for (literal, micros) in literals.iter().zip(values) {
            assert_eq!(
                literal.literal_type,
                Some(LiteralType::DayTimeInterval(micros))
            );
        }
    }

    /// As the literals of the Scala client's `lit` and of the pivot without values: a float, a
    /// byte, a short and a year-month interval are sent in their own types, so that a float is
    /// named "1.1" and not by the text of its double, and an interval by its SQL.
    #[test]
    fn pivot_floats_bytes_shorts_and_year_month_intervals_keep_their_types() {
        use crate::row::Value;
        use spark_connect_proto::{expression::literal::LiteralType, relation::RelType};
        let frame = session().range(1).unwrap();
        let pivot = frame
            .group_by(vec![] as Vec<Column>)
            .pivot(
                crate::column::col("value"),
                Some(vec![
                    Value::Float(1.1),
                    Value::Byte(-3),
                    Value::Short(4),
                    Value::YearMonthInterval(-13),
                ]),
            )
            .unwrap()
            .count();
        let Some(RelType::Aggregate(aggregate)) = pivot.plan.to_proto().rel_type else {
            panic!("expected Aggregate");
        };
        let literals: Vec<_> = aggregate
            .pivot
            .unwrap()
            .values
            .into_iter()
            .map(|literal| literal.literal_type)
            .collect();
        assert_eq!(
            literals,
            vec![
                Some(LiteralType::Float(1.1)),
                Some(LiteralType::Byte(-3)),
                Some(LiteralType::Short(4)),
                Some(LiteralType::YearMonthInterval(-13)),
            ]
        );
    }

    #[test]
    fn pivot_values_keep_their_types_as_literals() {
        use crate::row::Value;
        use spark_connect_proto::{expression::literal::LiteralType, relation::RelType};
        let values = vec![
            Value::Date(18262),
            Value::Timestamp(1_577_836_800_000_000),
            Value::TimestampNtz(1_577_836_800_000_000),
            Value::Time(3_600_000_000_000),
            Value::Binary(vec![1, 2]),
            Value::Decimal {
                value: "1.50".to_string(),
                precision: Some(5),
                scale: Some(2),
            },
            // Without a precision and a scale, those of the text: as `BigDecimal` has them.
            Value::Decimal {
                value: "-0.05".to_string(),
                precision: None,
                scale: None,
            },
            Value::Decimal {
                value: "1E+3".to_string(),
                precision: None,
                scale: None,
            },
        ];
        let frame = session().range(1).unwrap();
        let pivot = frame
            .group_by(vec![] as Vec<Column>)
            .pivot(crate::column::col("value"), Some(values))
            .unwrap()
            .count();
        let Some(RelType::Aggregate(aggregate)) = pivot.plan.to_proto().rel_type else {
            panic!("expected Aggregate");
        };
        let literals: Vec<_> = aggregate
            .pivot
            .unwrap()
            .values
            .into_iter()
            .map(|literal| literal.literal_type)
            .collect();
        let decimal = |value: &str, precision, scale| {
            Some(LiteralType::Decimal(
                spark_connect_proto::expression::literal::Decimal {
                    value: value.to_string(),
                    precision: Some(precision),
                    scale: Some(scale),
                },
            ))
        };
        assert_eq!(literals[0], Some(LiteralType::Date(18262)));
        assert_eq!(
            literals[1],
            Some(LiteralType::Timestamp(1_577_836_800_000_000))
        );
        assert_eq!(
            literals[2],
            Some(LiteralType::TimestampNtz(1_577_836_800_000_000))
        );
        assert!(matches!(
            literals[3],
            Some(LiteralType::Time(ref time)) if time.nano == 3_600_000_000_000 && time.precision == Some(6)
        ));
        assert_eq!(literals[4], Some(LiteralType::Binary(vec![1, 2].into())));
        assert_eq!(literals[5], decimal("1.50", 5, 2));
        assert_eq!(literals[6], decimal("-0.05", 1, 2));
        assert_eq!(literals[7], decimal("1E+3", 1, -3));
    }

    #[test]
    fn a_null_or_a_nested_pivot_value_is_not_expected_type() {
        use crate::row::Value;
        let frame = session().range(1).unwrap();
        for (value, arg_type) in [
            (Value::Null, "NoneType"),
            (Value::List(vec![]), "list"),
            (Value::Map(vec![]), "dict"),
        ] {
            let err = frame
                .group_by(vec![] as Vec<Column>)
                .pivot(crate::column::col("value"), Some(vec![value]))
                .err()
                .expect("an error");
            assert_eq!(err.kind, SparkErrorKind::TypeError);
            assert_eq!(err.get_condition().as_deref(), Some("NOT_EXPECTED_TYPE"));
            let parameters = err.get_message_parameters().unwrap();
            assert_eq!(parameters["arg_name"], "value");
            assert_eq!(parameters["arg_type"], arg_type);
            // The types that the Rust API takes, which are more than those of PySpark's.
            assert_eq!(
                err.message(),
                format!(
                    "[NOT_EXPECTED_TYPE] Argument `value` should be a bool, a number, a string, \
                     bytes, a date, a timestamp, a time or an interval, got {arg_type}."
                )
            );
        }
    }

    #[test]
    fn stat_crosstab_plan() {
        let spark = session();
        let df = spark.range(10).unwrap();
        let stats = df.stat();
        let result = stats.crosstab("col1", "col2");
        match result.plan.as_ref() {
            LogicalPlan::StatCrosstab { .. } => {
                // Plan is correct
            }
            _ => panic!("expected StatCrosstab plan"),
        }
    }

    #[test]
    fn stat_freq_items_plan() {
        let spark = session();
        let df = spark.range(10).unwrap();
        let stats = df.stat();
        let result = stats.freq_items(vec!["col1", "col2"], 0.25);
        match result.plan.as_ref() {
            LogicalPlan::StatFreqItems {
                columns, support, ..
            } => {
                assert_eq!(columns.len(), 2);
                assert_eq!(*support, 0.25);
            }
            _ => panic!("expected StatFreqItems plan"),
        }
    }

    #[test]
    fn stat_approx_quantile_plan() {
        let spark = session();
        let df = spark.range(10).unwrap();
        let stats = df.stat();
        let result = stats.approx_quantile(vec!["col1"], vec![0.25, 0.75], 0.05);
        match result.plan.as_ref() {
            LogicalPlan::StatApproxQuantile {
                columns,
                probabilities,
                relative_error,
                ..
            } => {
                assert_eq!(columns.len(), 1);
                assert_eq!(probabilities.len(), 2);
                assert_eq!(*relative_error, 0.05);
            }
            _ => panic!("expected StatApproxQuantile plan"),
        }
    }

    #[test]
    fn stat_sample_by_plan() {
        let spark = session();
        let df = spark.range(10).unwrap();
        let stats = df.stat();
        let fractions = vec![
            (
                Expression::literal(crate::expression::LiteralExpression::string("A")),
                0.5,
            ),
            (
                Expression::literal(crate::expression::LiteralExpression::string("B")),
                0.3,
            ),
        ];
        let result = stats.sample_by("category", fractions, Some(42)).unwrap();
        match result.plan.as_ref() {
            LogicalPlan::StatSampleBy {
                col,
                seed,
                fractions,
                ..
            } => {
                assert_eq!(*col, Expression::column("category"));
                assert_eq!(*seed, Some(42));
                assert_eq!(fractions.len(), 2);
            }
            _ => panic!("expected StatSampleBy plan"),
        }
    }

    /// As PySpark's `sampleBy` rejects a key that is not a float, an int or a str: a stratum
    /// that is not a literal, which the server cannot read, is `DISALLOWED_TYPE_FOR_CONTAINER`.
    #[test]
    fn sample_by_rejects_a_stratum_that_is_not_a_literal_as_pyspark() {
        let df = session().range(10).unwrap();
        let literal = Expression::literal(crate::expression::LiteralExpression::Long(1));
        let error = df
            .stat()
            .sample_by(
                "id",
                vec![(literal, 0.5), (Expression::column("id"), 0.5)],
                Some(1),
            )
            .err()
            .expect("an error");
        assert_eq!(error.kind, SparkErrorKind::TypeError);
        assert_eq!(
            error.message(),
            "[DISALLOWED_TYPE_FOR_CONTAINER] Argument `fractions`(type: dict) should only contain \
             a type in [literal], got Column"
        );
    }
}
