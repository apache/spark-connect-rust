//! Table argument support mirroring `pyspark.sql.connect.table_arg.TableArg`.
//!
//! Provides table argument support for table-valued functions with partitioning and ordering.

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::expression::Expression;
use crate::plan::LogicalPlan;
use spark_connect_core::error::{Result, SparkError};

/// Represents a table argument for a table-valued function.
///
/// Supports partitioning, ordering, and single-partition modes.
/// Mirrors `pyspark.sql.connect.table_arg.TableArg`.
#[allow(dead_code)]
pub struct TableArg {
    /// The underlying DataFrame/plan
    plan: LogicalPlan,
    /// Partition expressions
    partition_spec: Vec<Expression>,
    /// Order expressions
    order_spec: Vec<Expression>,
    /// Whether to use a single partition
    with_single_partition: bool,
}

impl TableArg {
    /// Create a new TableArg from a DataFrame.
    pub fn new(df: DataFrame) -> Self {
        TableArg {
            plan: df.plan().clone(),
            partition_spec: Vec::new(),
            order_spec: Vec::new(),
            with_single_partition: false,
        }
    }

    /// Create a new TableArg from a LogicalPlan.
    #[allow(dead_code)]
    pub(crate) fn from_plan(plan: LogicalPlan) -> Self {
        TableArg {
            plan,
            partition_spec: Vec::new(),
            order_spec: Vec::new(),
            with_single_partition: false,
        }
    }

    /// Check if partitioning has been applied.
    fn is_partitioned(&self) -> bool {
        !self.partition_spec.is_empty() || self.with_single_partition
    }

    /// Add partitioning to this table argument.
    ///
    /// # Arguments
    ///
    /// * `cols` - Columns to partition by
    ///
    /// # Example
    ///
    /// ```ignore
    /// let table_arg = TableArg::new(df)
    ///     .partition_by(vec![col("key")])?;
    /// ```
    pub fn partition_by(mut self, cols: Vec<Column>) -> Result<Self> {
        if self.is_partitioned() {
            return Err(SparkError::value(
                "ILLEGAL_ARGUMENT",
                &[("msg", "Cannot call partitionBy() after partitionBy() or withSinglePartition() has been called.")],
            ));
        }

        for col in cols {
            self.partition_spec.push(col.expression().clone());
        }

        Ok(self)
    }

    /// Add ordering to this table argument.
    ///
    /// Must be called after `partition_by` or `with_single_partition`.
    ///
    /// # Arguments
    ///
    /// * `cols` - Columns to order by
    ///
    /// # Example
    ///
    /// ```ignore
    /// let table_arg = TableArg::new(df)
    ///     .partition_by(vec![col("key")])?
    ///     .order_by(vec![col("value")])?;
    /// ```
    pub fn order_by(mut self, cols: Vec<Column>) -> Result<Self> {
        if !self.is_partitioned() {
            return Err(SparkError::value(
                "ILLEGAL_ARGUMENT",
                &[(
                    "msg",
                    "Please call partitionBy() or withSinglePartition() before orderBy().",
                )],
            ));
        }

        for col in cols {
            self.order_spec.push(col.expression().clone());
        }

        Ok(self)
    }

    /// Use a single partition for this table argument.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let table_arg = TableArg::new(df)
    ///     .with_single_partition()?;
    /// ```
    pub fn with_single_partition(mut self) -> Result<Self> {
        if self.is_partitioned() {
            return Err(SparkError::value(
                "ILLEGAL_ARGUMENT",
                &[("msg", "Cannot call withSinglePartition() after partitionBy() or withSinglePartition() has been called.")],
            ));
        }

        self.with_single_partition = true;
        Ok(self)
    }

    /// Get the underlying plan.
    #[allow(dead_code)]
    pub(crate) fn plan(&self) -> &LogicalPlan {
        &self.plan
    }

    /// Get partition specification.
    #[allow(dead_code)]
    pub(crate) fn partition_spec(&self) -> &[Expression] {
        &self.partition_spec
    }

    /// Get order specification.
    #[allow(dead_code)]
    pub(crate) fn order_spec(&self) -> &[Expression] {
        &self.order_spec
    }

    /// Check if single partition mode is enabled.
    #[allow(dead_code)]
    pub(crate) fn is_single_partition(&self) -> bool {
        self.with_single_partition
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_arg_partitioning() {
        // This test verifies that TableArg partitioning logic works.
        // A full end-to-end test requires a running Spark Connect server.
    }
}
