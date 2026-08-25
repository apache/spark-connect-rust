//! GroupedData implementation for aggregations.
//!
//! Mirroring `pyspark.sql.GroupedData`.

use spark_connect_core::error::Result;

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::expression::Expression;
use crate::plan::{AggregateGroupType, LogicalPlan};
use crate::types::DataType;
use crate::udf::CommonInlineUserDefinedFunctionExpression;

/// Convert a [`crate::row::Value`] into a literal [`Expression`] (for pivot values).
fn value_to_lit_expr(v: crate::row::Value) -> Expression {
    use crate::expression::LiteralExpression as L;
    use crate::row::Value;
    let lit = match v {
        Value::Bool(b) => L::boolean(b),
        Value::Byte(x) => L::int(x as i32),
        Value::Short(x) => L::int(x as i32),
        Value::Integer(x) => L::int(x),
        Value::Long(x) => L::long(x),
        Value::Float(x) => L::double(x as f64),
        Value::Double(x) => L::double(x),
        Value::String(s) => L::string(s),
        other => L::string(format!("{:?}", other)),
    };
    Expression::Literal(lit)
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
    /// pivot-value literals (previously they were dropped entirely).
    pub fn pivot(&self, pivot_col: Column, values: Option<Vec<crate::row::Value>>) -> GroupedData {
        let pivot_values = values
            .unwrap_or_default()
            .into_iter()
            .map(value_to_lit_expr)
            .collect();
        GroupedData {
            dataframe: self.dataframe.clone(),
            group_cols: self.group_cols.clone(),
            group_type: AggregateGroupType::Pivot,
            pivot_col: Some(pivot_col.expression().clone()),
            pivot_values,
            grouping_sets: vec![],
        }
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
        let plan = LogicalPlan::GroupMap {
            input: Box::new(self.dataframe.plan.clone()),
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
        DataFrame::new(self.dataframe.session.clone(), plan)
    }

    /// Apply a stateful pandas UDF to each group (`GroupedData.applyInPandasWithState`).
    ///
    /// `func` (built on the Python side with eval type
    /// `SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE` and carrying the output schema as its
    /// return type) is combined with the state schema, output mode, and timeout.
    pub fn apply_in_pandas_with_state(
        &self,
        func: CommonInlineUserDefinedFunctionExpression,
        state_schema: DataType,
        output_mode: &str,
        timeout_conf: &str,
    ) -> DataFrame {
        let plan = LogicalPlan::GroupMap {
            input: Box::new(self.dataframe.plan.clone()),
            grouping_expressions: self.grouping_expressions(),
            func,
            sorting_expressions: vec![],
            initial_input: None,
            initial_grouping_expressions: vec![],
            is_map_groups_with_state: Some(false),
            output_mode: Some(output_mode.to_string()),
            timeout_conf: Some(timeout_conf.to_string()),
            state_schema: Some(state_schema),
            transform_with_state_info: None,
        };
        DataFrame::new(self.dataframe.session.clone(), plan)
    }

    /// `GroupedData.transformWithState` (row output). `func` is the cloudpickled
    /// stateful processor (eval type in the 211-214 range).
    pub fn transform_with_state(
        &self,
        func: CommonInlineUserDefinedFunctionExpression,
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
            None,
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
                Some(Box::new(gd.dataframe.plan.clone())),
                gd.grouping_expressions(),
            ),
            None => (None, vec![]),
        };
        let plan = LogicalPlan::GroupMap {
            input: Box::new(self.dataframe.plan.clone()),
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
            input: Box::new(self.dataframe.plan.clone()),
            group_type: self.group_type,
            grouping_expressions,
            aggregate_expressions: expressions,
            pivot_col: self.pivot_col.clone(),
            pivot_values: self.pivot_values.clone(),
            grouping_sets,
        };

        DataFrame::new(self.dataframe.session.clone(), plan)
    }

    /// Count rows in each group.
    pub fn count(&self) -> DataFrame {
        use crate::functions;
        let count_expr = functions::count(Column::new(Expression::Literal(
            crate::expression::LiteralExpression::int(1),
        )))
        .expression()
        .clone();

        self.agg(vec![count_expr])
    }

    /// Sum values in each group.
    pub fn sum(&self, columns: Vec<&str>) -> DataFrame {
        use crate::functions;
        let expressions: Vec<_> = columns
            .iter()
            .map(|col| functions::sum(crate::column::col(col)).expression().clone())
            .collect();

        self.agg(expressions)
    }

    /// Average values in each group.
    pub fn avg(&self, columns: Vec<&str>) -> DataFrame {
        use crate::functions;
        let expressions: Vec<_> = columns
            .iter()
            .map(|col| functions::avg(crate::column::col(col)).expression().clone())
            .collect();

        self.agg(expressions)
    }

    /// Minimum values in each group.
    pub fn min(&self, columns: Vec<&str>) -> DataFrame {
        use crate::functions;
        let expressions: Vec<_> = columns
            .iter()
            .map(|col| functions::min(crate::column::col(col)).expression().clone())
            .collect();

        self.agg(expressions)
    }

    /// Maximum values in each group.
    pub fn max(&self, columns: Vec<&str>) -> DataFrame {
        use crate::functions;
        let expressions: Vec<_> = columns
            .iter()
            .map(|col| functions::max(crate::column::col(col)).expression().clone())
            .collect();

        self.agg(expressions)
    }

    /// Mean (alias for avg).
    pub fn mean(&self, columns: Vec<&str>) -> DataFrame {
        self.avg(columns)
    }
}

/// Statistical functions for DataFrames.
///
/// Mirrors `pyspark.sql.DataFrameStatFunctions`.
pub struct StatFunctions {
    dataframe: DataFrame,
}

impl StatFunctions {
    /// Create a new StatFunctions.
    pub(crate) fn new(dataframe: DataFrame) -> Self {
        StatFunctions { dataframe }
    }

    /// Compute a cross-tabulation of two columns.
    pub fn crosstab(&self, col1: &str, col2: &str) -> DataFrame {
        let plan = LogicalPlan::StatCrosstab {
            input: Box::new(self.dataframe.plan.clone()),
            col1: col1.to_string(),
            col2: col2.to_string(),
        };
        DataFrame::new(self.dataframe.session.clone(), plan)
    }

    /// Find frequent items.
    pub fn freq_items(&self, columns: Vec<&str>, support: f64) -> DataFrame {
        let plan = LogicalPlan::StatFreqItems {
            input: Box::new(self.dataframe.plan.clone()),
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
            input: Box::new(self.dataframe.plan.clone()),
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
            input: Box::new(self.dataframe.plan.clone()),
            col1: col1.to_string(),
            col2: col2.to_string(),
        };
        let df = DataFrame::new(self.dataframe.session.clone(), plan);
        Ok(df.scalar()?.and_then(|v| v.as_f64()).unwrap_or(f64::NAN))
    }

    /// Compute covariance between two columns.
    ///
    /// Executes the stat aggregation and returns the resulting `f64`.
    pub fn cov(&self, col1: &str, col2: &str) -> Result<f64> {
        let plan = LogicalPlan::StatCov {
            input: Box::new(self.dataframe.plan.clone()),
            col1: col1.to_string(),
            col2: col2.to_string(),
        };
        let df = DataFrame::new(self.dataframe.session.clone(), plan);
        Ok(df.scalar()?.and_then(|v| v.as_f64()).unwrap_or(f64::NAN))
    }

    /// Sample by values in a column.
    pub fn sample_by(
        &self,
        col: &str,
        fractions: Vec<(Expression, f64)>,
        seed: Option<i64>,
    ) -> DataFrame {
        let plan = LogicalPlan::StatSampleBy {
            input: Box::new(self.dataframe.plan.clone()),
            col: col.to_string(),
            fractions,
            seed,
        };
        DataFrame::new(self.dataframe.session.clone(), plan)
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

    /// Drop rows containing null values. Mirrors `DataFrameNaFunctions.drop`.
    pub fn drop(
        &self,
        how: Option<&str>,
        thresh: Option<i32>,
        subset: Option<Vec<&str>>,
    ) -> DataFrame {
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
        let plan = LogicalPlan::CoGroupMap {
            input: Box::new(self.left.dataframe.plan.clone()),
            input_grouping_expressions: self.left.grouping_expressions(),
            other: Box::new(self.right.dataframe.plan.clone()),
            other_grouping_expressions: self.right.grouping_expressions(),
            func,
        };
        DataFrame::new(self.left.dataframe.session.clone(), plan)
    }
}
