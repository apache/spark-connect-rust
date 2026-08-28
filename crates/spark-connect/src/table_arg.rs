//! Table argument support mirroring `pyspark.sql.connect.table_arg.TableArg`.
//!
//! Provides table argument support for table-valued functions with partitioning and ordering.

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::expression::{Expression, SortOrder};
use crate::plan::LogicalPlan;
use spark_connect_core::error::{Result, SparkError};
use spark_connect_proto as proto;

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
    pub fn partition_by<C: Into<Column>>(
        mut self,
        cols: impl IntoIterator<Item = C>,
    ) -> Result<Self> {
        if self.is_partitioned() {
            return Err(SparkError::value(
                "ILLEGAL_ARGUMENT",
                &[("msg", "Cannot call partitionBy() after partitionBy() or withSinglePartition() has been called.")],
            ));
        }

        for col in cols {
            self.partition_spec.push(col.into().expression().clone());
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
            return Err(SparkError::value(
                "ILLEGAL_ARGUMENT",
                &[(
                    "msg",
                    "Please call partitionBy() or withSinglePartition() before orderBy().",
                )],
            ));
        }

        for col in cols {
            self.order_spec.push(col.into().expression().clone());
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

    /// Serialize this table argument to a proto `Expression` - a
    /// `SubqueryExpression` of type `TABLE_ARG` carrying the partition/order
    /// specs and single-partition flag.
    ///
    /// Mirrors `SubqueryExpression.to_plan` in the reference PySpark client:
    /// `plan_id` is the plan-id assigned to the underlying subquery relation,
    /// each partition expression is emitted as-is, each order expression is
    /// emitted as a `SortOrder` (an explicit sort direction is preserved;
    /// otherwise it defaults to ascending, nulls-first), and
    /// `with_single_partition` is only set when true (left unset otherwise, as
    /// the reference leaves it `None`).
    #[allow(dead_code)]
    pub(crate) fn to_proto(&self, plan_id: i64) -> proto::Expression {
        let mut options = proto::subquery_expression::TableArgOptions::default();

        options.partition_spec = self.partition_spec.iter().map(|e| e.to_proto()).collect();

        options.order_spec = self
            .order_spec
            .iter()
            .map(|e| {
                // Preserve an explicit SortOrder; otherwise default to ascending,
                // nulls-first (matching the reference's `_sort_col`).
                let wrapped = match e {
                    Expression::SortOrder(_) => e.to_proto(),
                    other => SortOrder::asc_nulls_first(other.clone()).to_proto(),
                };
                match wrapped.expr_type {
                    Some(proto::expression::ExprType::SortOrder(s)) => *s,
                    // SortOrder::to_proto always yields a SortOrder expr_type.
                    _ => unreachable!("expected a SortOrder expression"),
                }
            })
            .collect();

        if self.with_single_partition {
            options.with_single_partition = Some(true);
        }

        let mut subquery = proto::SubqueryExpression::default();
        subquery.plan_id = plan_id;
        subquery.subquery_type = proto::subquery_expression::SubqueryType::TableArg as i32;
        subquery.table_arg_options = Some(options);

        let mut expr = proto::Expression::default();
        expr.expr_type = Some(proto::expression::ExprType::SubqueryExpression(subquery));
        expr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::col;

    fn base_plan() -> LogicalPlan {
        LogicalPlan::Range {
            start: 0,
            end: 10,
            step: 1,
            num_partitions: None,
        }
    }

    fn arg() -> TableArg {
        TableArg::from_plan(base_plan())
    }

    #[test]
    fn fresh_table_arg_is_unpartitioned() {
        let t = arg();
        assert!(!t.is_partitioned());
        assert!(!t.is_single_partition());
        assert!(t.partition_spec().is_empty());
        assert!(t.order_spec().is_empty());
        assert!(matches!(t.plan(), LogicalPlan::Range { .. }));
    }

    #[test]
    fn partition_by_records_expressions_and_marks_partitioned() {
        let t = arg().partition_by(vec![col("a"), col("b")]).unwrap();
        assert!(t.is_partitioned());
        assert_eq!(t.partition_spec().len(), 2);
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
        assert_eq!(t.order_spec().len(), 2);
    }

    #[test]
    fn order_by_after_single_partition_is_allowed() {
        let t = arg()
            .with_single_partition()
            .unwrap()
            .order_by(vec![col("b")])
            .unwrap();
        assert_eq!(t.order_spec().len(), 1);
    }

    /// Assert a `Result<TableArg>` is an `ILLEGAL_ARGUMENT` error. (Written as a
    /// helper because `unwrap_err` would require `TableArg: Debug`.)
    fn assert_illegal(r: Result<TableArg>) {
        match r {
            Ok(_) => panic!("expected an ILLEGAL_ARGUMENT error, got Ok"),
            Err(e) => assert!(
                format!("{e:?}").contains("ILLEGAL_ARGUMENT"),
                "unexpected error: {e:?}"
            ),
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
        use proto::expression::ExprType;

        let t = arg()
            .partition_by(vec![col("a")])
            .unwrap()
            .order_by(vec![col("b")])
            .unwrap();
        let expr = t.to_proto(7);

        let Some(ExprType::SubqueryExpression(subq)) = expr.expr_type else {
            panic!("expected a SubqueryExpression");
        };
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
    }

    #[test]
    fn to_proto_single_partition_sets_flag_and_preserves_desc_order() {
        let t = arg()
            .with_single_partition()
            .unwrap()
            .order_by(vec![col("b").desc_nulls_last()])
            .unwrap();
        let expr = t.to_proto(3);

        let Some(proto::expression::ExprType::SubqueryExpression(subq)) = expr.expr_type else {
            panic!("expected a SubqueryExpression");
        };
        let opts = subq.table_arg_options.expect("table_arg_options");
        assert_eq!(opts.with_single_partition, Some(true));
        assert!(opts.partition_spec.is_empty());
        // The explicit descending, nulls-last order is preserved (not defaulted).
        assert_eq!(opts.order_spec.len(), 1);
        assert_eq!(opts.order_spec[0].direction, 2);
        assert_eq!(opts.order_spec[0].null_ordering, 2);
    }
}
