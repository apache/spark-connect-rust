//! Table argument support mirroring `pyspark.sql.connect.table_arg.TableArg`.
//!
//! Provides table argument support for table-valued functions with partitioning and ordering.

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::expression::{ExpressionNode, SubqueryExpression, SubqueryType};
#[cfg(test)]
use crate::plan::{LogicalPlan, LogicalPlanKind};
use spark_connect_core::error::{Result, SparkError, SparkErrorKind};

/// The `IllegalArgumentException` of `message`, as PySpark's `TableArg` raises for a
/// partitioning or ordering in the wrong order.
fn illegal_argument(message: &str) -> SparkError {
    let mut error = SparkError::value_msg(message);
    error.kind = SparkErrorKind::IllegalArgument;
    error
}

/// Represents a table argument for a table-valued function.
///
/// Supports partitioning, ordering, and single-partition modes. Mirrors
/// `pyspark.sql.connect.table_arg.TableArg`: a `SubqueryExpression` of type `TABLE_ARG` of the
/// plan of the DataFrame, which a UDTF call takes as an argument.
#[derive(Clone)]
pub struct TableArg {
    /// The subquery of the plan, with its partition and order specs.
    subquery: SubqueryExpression,
}

impl TableArg {
    /// Create a new TableArg from a DataFrame.
    pub fn new(df: DataFrame) -> Self {
        TableArg {
            subquery: df.subquery(SubqueryType::TableArg),
        }
    }

    /// Create a new TableArg of a LogicalPlan whose plan ID is `plan_id`.
    #[cfg(test)]
    pub(crate) fn from_plan(plan: LogicalPlan, plan_id: i64) -> Self {
        TableArg {
            subquery: SubqueryExpression::new(
                std::sync::Arc::new(plan),
                plan_id,
                SubqueryType::TableArg,
            ),
        }
    }

    /// Check if partitioning has been applied.
    fn is_partitioned(&self) -> bool {
        !self.subquery.partition_spec.is_empty() || self.is_single_partition()
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
    pub fn partition_by<C: Into<Column>>(
        mut self,
        cols: impl IntoIterator<Item = C>,
    ) -> Result<Self> {
        if self.is_partitioned() {
            return Err(illegal_argument(
                "Cannot call partitionBy() after partitionBy() or withSinglePartition() has \
                 been called.",
            ));
        }

        for col in cols {
            self.subquery
                .partition_spec
                .push(col.into().expression().clone());
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
    pub fn order_by<C: Into<Column>>(mut self, cols: impl IntoIterator<Item = C>) -> Result<Self> {
        if !self.is_partitioned() {
            return Err(illegal_argument(
                "Please call partitionBy() or withSinglePartition() before orderBy().",
            ));
        }

        for col in cols {
            self.subquery
                .order_spec
                .push(col.into().expression().clone());
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
            return Err(illegal_argument(
                "Cannot call withSinglePartition() after partitionBy() or withSinglePartition() \
                 has been called.",
            ));
        }

        self.subquery.with_single_partition = Some(true);
        Ok(self)
    }

    /// Get the underlying plan.
    #[cfg(test)]
    pub(crate) fn plan(&self) -> &LogicalPlan {
        &self.subquery.plan
    }

    /// Check if single partition mode is enabled.
    pub(crate) fn is_single_partition(&self) -> bool {
        self.subquery.with_single_partition == Some(true)
    }

    /// This table argument as a Column, the argument of a UDTF call that passes the rows of
    /// the DataFrame. As the `_subquery_expr` of a PySpark `TableArg`.
    pub fn to_column(&self) -> Column {
        Column::new(ExpressionNode::Subquery(Box::new(self.subquery.clone())).into())
    }
}

impl From<TableArg> for Column {
    fn from(table_arg: TableArg) -> Column {
        table_arg.to_column()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::col;
    use crate::expression::Expression;
    use spark_connect_proto as proto;

    fn base_plan() -> LogicalPlan {
        LogicalPlan::from_builder(
            LogicalPlanKind::Range {
                start: 0,
                end: 10,
                step: 1,
                num_partitions: None,
            },
            [],
        )
    }

    fn arg() -> TableArg {
        TableArg::from_plan(base_plan(), 7)
    }

    fn spec(t: &TableArg) -> (&[Expression], &[Expression]) {
        (&t.subquery.partition_spec, &t.subquery.order_spec)
    }

    /// The `SubqueryExpression` of the table argument.
    fn subquery_proto(t: &TableArg) -> proto::SubqueryExpression {
        let Some(proto::expression::ExprType::SubqueryExpression(subquery)) =
            t.to_column().expression().to_proto().expr_type
        else {
            panic!("expected a SubqueryExpression");
        };
        subquery
    }

    #[test]
    fn fresh_table_arg_is_unpartitioned() {
        let t = arg();
        assert!(!t.is_partitioned());
        assert!(!t.is_single_partition());
        assert!(spec(&t).0.is_empty());
        assert!(spec(&t).1.is_empty());
        assert!(matches!(t.plan().kind(), LogicalPlanKind::Range { .. }));
        // As PySpark, which sets no options for a table argument without any.
        let subquery = subquery_proto(&t);
        assert_eq!(subquery.plan_id, 7);
        assert_eq!(subquery.table_arg_options, None);
    }

    #[test]
    fn partition_by_records_expressions_and_marks_partitioned() {
        let t = arg().partition_by(vec![col("a"), col("b")]).unwrap();
        assert!(t.is_partitioned());
        assert_eq!(spec(&t).0.len(), 2);
    }

    #[test]
    fn with_single_partition_marks_partitioned() {
        let t = arg().with_single_partition().unwrap();
        assert!(t.is_partitioned());
        assert!(t.is_single_partition());
    }

    #[test]
    fn order_by_after_partitioning_records_expressions() {
        let t = arg()
            .partition_by(vec![col("a")])
            .unwrap()
            .order_by(vec![col("b"), col("c")])
            .unwrap();
        assert_eq!(spec(&t).1.len(), 2);
    }

    #[test]
    fn order_by_after_single_partition_is_allowed() {
        let t = arg()
            .with_single_partition()
            .unwrap()
            .order_by(vec![col("b")])
            .unwrap();
        assert_eq!(spec(&t).1.len(), 1);
    }

    /// Assert a `Result<TableArg>` is an `IllegalArgumentException`, as PySpark raises.
    /// (Written as a helper because `unwrap_err` would require `TableArg: Debug`.)
    fn assert_illegal(r: Result<TableArg>) {
        match r {
            Ok(_) => panic!("expected an IllegalArgumentException, got Ok"),
            Err(e) => {
                assert_eq!(e.kind, SparkErrorKind::IllegalArgument, "{e:?}");
                assert!(e.message.contains("() "), "unexpected error: {e:?}");
            }
        }
    }

    #[test]
    fn order_by_before_partitioning_errors() {
        assert_illegal(arg().order_by(vec![col("b")]));
    }

    #[test]
    fn double_partition_by_errors() {
        let t = arg().partition_by(vec![col("a")]).unwrap();
        assert_illegal(t.partition_by(vec![col("b")]));
    }

    #[test]
    fn single_partition_after_partition_by_errors() {
        let t = arg().partition_by(vec![col("a")]).unwrap();
        assert_illegal(t.with_single_partition());
    }

    #[test]
    fn double_single_partition_errors() {
        let t = arg().with_single_partition().unwrap();
        assert_illegal(t.with_single_partition());
    }

    #[test]
    fn to_proto_emits_table_arg_subquery_with_specs() {
        let t = arg()
            .partition_by(vec![col("a")])
            .unwrap()
            .order_by(vec![col("b")])
            .unwrap();
        let subq = subquery_proto(&t);
        assert_eq!(subq.plan_id, 7);
        assert_eq!(
            subq.subquery_type,
            proto::subquery_expression::SubqueryType::TableArg as i32
        );
        let opts = subq.table_arg_options.expect("table_arg_options");
        assert_eq!(opts.partition_spec.len(), 1);
        assert_eq!(opts.order_spec.len(), 1);
        // A bare column ordering defaults to ascending, nulls-first.
        assert_eq!(opts.order_spec[0].direction, 1);
        assert_eq!(opts.order_spec[0].null_ordering, 1);
        // partitionBy path leaves the single-partition flag unset (None).
        assert_eq!(opts.with_single_partition, None);
        assert_eq!(
            t.to_column().expression().render(),
            "SubqueryExpression(plan=<plan 7>, type=table_arg, partition_spec=[a], \
             order_spec=[b])"
        );
    }

    #[test]
    fn to_proto_single_partition_sets_flag_and_preserves_desc_order() {
        let t = arg()
            .with_single_partition()
            .unwrap()
            .order_by(vec![col("b").desc_nulls_last()])
            .unwrap();
        let subq = subquery_proto(&t);
        let opts = subq.table_arg_options.expect("table_arg_options");
        assert_eq!(opts.with_single_partition, Some(true));
        assert!(opts.partition_spec.is_empty());
        // The explicit descending, nulls-last order is preserved (not defaulted).
        assert_eq!(opts.order_spec.len(), 1);
        assert_eq!(opts.order_spec[0].direction, 2);
        assert_eq!(opts.order_spec[0].null_ordering, 2);
    }
}
