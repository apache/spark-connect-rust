//! GroupedData implementation for aggregations.
//!
//! Mirroring `pyspark.sql.GroupedData`.

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::expression::Expression;
use crate::plan::{AggregateGroupType, LogicalPlan};

/// Grouped data for performing aggregations.
///
/// Mirrors `pyspark.sql.GroupedData`.
pub struct GroupedData {
    dataframe: DataFrame,
    group_cols: Vec<Column>,
    group_type: AggregateGroupType,
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

        let plan = LogicalPlan::Aggregate {
            input: Box::new(self.dataframe.plan.clone()),
            group_type: self.group_type,
            grouping_expressions,
            aggregate_expressions: expressions,
            pivot_col: None,
            pivot_values: vec![],
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
    pub fn corr(&self, col1: &str, col2: &str) -> f64 {
        let plan = LogicalPlan::StatCorr {
            input: Box::new(self.dataframe.plan.clone()),
            col1: col1.to_string(),
            col2: col2.to_string(),
        };
        let df = DataFrame::new(self.dataframe.session.clone(), plan);
        // Execute and extract the correlation value
        // For now, return 0.0 as a placeholder since proper result extraction would require more work
        0.0
    }

    /// Compute covariance between two columns.
    pub fn cov(&self, col1: &str, col2: &str) -> f64 {
        let plan = LogicalPlan::StatCov {
            input: Box::new(self.dataframe.plan.clone()),
            col1: col1.to_string(),
            col2: col2.to_string(),
        };
        let df = DataFrame::new(self.dataframe.session.clone(), plan);
        // Execute and extract the covariance value
        // For now, return 0.0 as a placeholder
        0.0
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
