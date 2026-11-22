//! Window functions and specifications mirroring PySpark's `pyspark.sql.window`.
//!
//! Provides the `Window` static builder and `WindowSpec` for defining window partitions,
//! ordering, and frame boundaries used with `Column.over()`.

use crate::column::{expressions_of, Column};
use crate::expression::{Expression, ExpressionNode, SortOrder};

/// Frame bound value used in window frame definitions.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameBound {
    /// UNBOUNDED PRECEDING
    UnboundedPreceding,
    /// Specific number of rows PRECEDING
    Preceding(i64),
    /// CURRENT ROW
    CurrentRow,
    /// Specific number of rows FOLLOWING
    Following(i64),
    /// UNBOUNDED FOLLOWING
    UnboundedFollowing,
}

impl FrameBound {
    /// The bound as the int of PySpark's `rowsBetween` and `rangeBetween`, as
    /// [`crate::expression::FrameBoundary::offset`] gives it.
    fn offset(&self) -> i128 {
        match self {
            FrameBound::UnboundedPreceding => i64::MIN.into(),
            FrameBound::Preceding(n) => -i128::from(*n),
            FrameBound::CurrentRow => 0,
            FrameBound::Following(n) => (*n).into(),
            FrameBound::UnboundedFollowing => i64::MAX.into(),
        }
    }

    /// This bound as the start of a frame of PySpark's `rowsBetween` and `rangeBetween`: one at
    /// or before `Window._PRECEDING_THRESHOLD`, `-sys.maxsize`, is unbounded preceding.
    fn into_start(self) -> Self {
        if self.offset() <= -i128::from(i64::MAX) {
            FrameBound::UnboundedPreceding
        } else {
            self
        }
    }

    /// This bound as the end of a frame, as [`FrameBound::into_start`]: one at or after
    /// `Window._FOLLOWING_THRESHOLD`, `sys.maxsize`, is unbounded following.
    fn into_end(self) -> Self {
        if self.offset() >= i128::from(i64::MAX) {
            FrameBound::UnboundedFollowing
        } else {
            self
        }
    }
}

/// Frame type for window specifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// ROWS frame type
    Row,
    /// RANGE frame type
    Range,
}

/// `pyspark.sql.window.WindowSpec`
///
/// Represents a complete window specification with optional partition, order, and frame specs.
/// This is built using the static builder methods on `Window` or by chaining methods on this struct.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowSpec {
    /// Expressions to partition by.
    pub partition_spec: Vec<Expression>,
    /// Sort orders for the window.
    pub order_spec: Vec<SortOrder>,
    /// Optional window frame specification.
    pub frame_spec: Option<(FrameType, FrameBound, FrameBound)>,
}

impl WindowSpec {
    /// As `WindowSpec.__repr__` of PySpark Connect, such as
    /// `WindowSpec(PartitionBy(g), OrderBy(o ASC NULLS FIRST), WindowFrame(ROW_FRAME, -1, 1))`.
    pub fn repr(&self) -> String {
        let boundary = |bound: &FrameBound| match bound {
            FrameBound::UnboundedPreceding => crate::expression::FrameBoundary::UnboundedPreceding,
            FrameBound::Preceding(n) => crate::expression::FrameBoundary::Preceding(*n),
            FrameBound::CurrentRow => crate::expression::FrameBoundary::CurrentRow,
            FrameBound::Following(n) => crate::expression::FrameBoundary::Following(*n),
            FrameBound::UnboundedFollowing => crate::expression::FrameBoundary::UnboundedFollowing,
        };
        let frame = self.frame_spec.as_ref().map(|(frame_type, lower, upper)| {
            let frame_type = match frame_type {
                FrameType::Row => 1u32,
                FrameType::Range => 2u32,
            };
            (frame_type, boundary(lower), boundary(upper))
        });
        crate::expression::window_spec_repr(&self.partition_spec, &self.order_spec, frame.as_ref())
    }

    /// The references of this window specification to the shared parts of its partitioning and
    /// its ordering (see [`crate::gc`]). Not a stable API.
    #[doc(hidden)]
    pub fn held_parts(&self) -> Vec<crate::gc::PartRef<'_>> {
        let orders = self.order_spec.iter().map(|sort| &sort.child);
        self.partition_spec
            .iter()
            .chain(orders)
            .map(crate::gc::PartRef::Expression)
            .collect()
    }

    /// Create a new empty WindowSpec.
    pub fn new() -> Self {
        Self {
            partition_spec: Vec::new(),
            order_spec: Vec::new(),
            frame_spec: None,
        }
    }

    /// The window specification of these partition columns, in place of those of this one, as
    /// PySpark's `WindowSpec.partitionBy` (and Spark's) makes it: a chained call partitions by
    /// the columns of the last one only.
    pub fn partition_by<C: Into<Column>>(mut self, cols: impl IntoIterator<Item = C>) -> Self {
        self.partition_spec = expressions_of(cols);
        self
    }

    /// The window specification of this ordering, in place of that of this one, as PySpark's
    /// `WindowSpec.orderBy` (and Spark's) makes it: a chained call orders by the columns of the
    /// last one only. A sort order, of [`Column::asc`], [`Column::desc`] and their nulls
    /// variants, or a [`SortOrder`], sorts as it says; any other column is sorted ascending,
    /// nulls first, as PySpark's `WindowSpec.orderBy` sorts it.
    pub fn order_by<C: Into<Column>>(mut self, cols: impl IntoIterator<Item = C>) -> Self {
        self.order_spec = expressions_of(cols)
            .into_iter()
            .map(|expr| match expr.node() {
                ExpressionNode::SortOrder(sort) => (**sort).clone(),
                _ => SortOrder::asc_nulls_first(expr),
            })
            .collect();
        self
    }

    /// Set the frame specification for ROWS, as PySpark's `WindowSpec.rowsBetween(start, end)`
    /// of the ints of the bounds: a start at or before `Window._PRECEDING_THRESHOLD`
    /// (`-sys.maxsize`, `Preceding(i64::MAX)`) is unbounded preceding, and an end at or after
    /// `Window._FOLLOWING_THRESHOLD` (`sys.maxsize`) unbounded following. The frame is sent as
    /// PySpark's `WindowExpression` sends it: a bound of 0 is the current row, the unbounded
    /// start and end are unbounded, and another bound is the int literal of its offset. As
    /// PySpark's, a request of an offset beyond an int, such as an unbounded following start,
    /// fails with `VALUE_NOT_BETWEEN` of `start` or `end`.
    pub fn rows_between(mut self, start: FrameBound, end: FrameBound) -> Self {
        self.frame_spec = Some((FrameType::Row, start.into_start(), end.into_end()));
        self
    }

    /// Set the frame specification for RANGE, as PySpark's `WindowSpec.rangeBetween(start, end)`
    /// of the ints of the bounds, which it takes as [`WindowSpec::rows_between`] takes them, and
    /// sends a bound that is not the current row nor unbounded as the long literal of its offset.
    pub fn range_between(mut self, start: FrameBound, end: FrameBound) -> Self {
        self.frame_spec = Some((FrameType::Range, start.into_start(), end.into_end()));
        self
    }
}

impl Default for WindowSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// `pyspark.sql.window.Window`
///
/// Static builder for creating window specifications.
pub struct Window;

impl Window {
    /// UNBOUNDED PRECEDING frame bound.
    pub const fn unbounded_preceding() -> FrameBound {
        FrameBound::UnboundedPreceding
    }

    /// CURRENT ROW frame bound.
    pub const fn current_row() -> FrameBound {
        FrameBound::CurrentRow
    }

    /// UNBOUNDED FOLLOWING frame bound.
    pub const fn unbounded_following() -> FrameBound {
        FrameBound::UnboundedFollowing
    }

    /// Create a WindowSpec partitioned by the given columns.
    pub fn partition_by<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> WindowSpec {
        WindowSpec::new().partition_by(cols)
    }

    /// Create a WindowSpec ordered by the given columns, as [`WindowSpec::order_by`] orders it.
    pub fn order_by<C: Into<Column>>(cols: impl IntoIterator<Item = C>) -> WindowSpec {
        WindowSpec::new().order_by(cols)
    }

    /// Create a WindowSpec with a ROWS frame between the given bounds.
    pub fn rows_between(start: FrameBound, end: FrameBound) -> WindowSpec {
        WindowSpec::new().rows_between(start, end)
    }

    /// Create a WindowSpec with a RANGE frame between the given bounds.
    pub fn range_between(start: FrameBound, end: FrameBound) -> WindowSpec {
        WindowSpec::new().range_between(start, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::{NullOrdering, SortOrder};

    fn col(name: &str) -> Expression {
        Expression::column(name)
    }

    fn sort(name: &str) -> SortOrder {
        SortOrder {
            child: col(name),
            ascending: true,
            null_ordering: NullOrdering::First,
        }
    }

    #[test]
    fn new_and_default_are_empty() {
        let w = WindowSpec::new();
        assert!(w.partition_spec.is_empty());
        assert!(w.order_spec.is_empty());
        assert!(w.frame_spec.is_none());
        assert_eq!(WindowSpec::default(), w);
    }

    #[test]
    fn spec_builders_chain() {
        let w = WindowSpec::new()
            .partition_by(vec![col("a"), col("b")])
            .order_by(vec![sort("c")]);
        assert_eq!(w.partition_spec.len(), 2);
        assert_eq!(w.order_spec.len(), 1);
    }

    #[test]
    fn chained_partition_by_and_order_by_replace_the_specs_as_pyspark() {
        // As PySpark's `WindowSpec.partitionBy` and `orderBy`, which make a spec of the new
        // partitioning or ordering in place of the old one, and keep the other parts.
        let w = WindowSpec::new()
            .partition_by(vec![col("a")])
            .order_by(vec![sort("c")])
            .rows_between(FrameBound::Preceding(1), FrameBound::CurrentRow)
            .partition_by(vec![col("b")])
            .order_by(vec![SortOrder::desc_nulls_last(col("d")), sort("e")]);
        assert_eq!(w.partition_spec, vec![col("b")]);
        assert_eq!(
            w.order_spec,
            vec![SortOrder::desc_nulls_last(col("d")), sort("e")]
        );
        assert_eq!(
            w.frame_spec,
            Some((
                FrameType::Row,
                FrameBound::Preceding(1),
                FrameBound::CurrentRow
            ))
        );
        let unpartitioned = Window::partition_by(vec![col("a")]).partition_by(Vec::<Column>::new());
        assert!(unpartitioned.partition_spec.is_empty());
    }

    #[test]
    fn rows_between_sets_row_frame() {
        let w = WindowSpec::new().rows_between(FrameBound::Preceding(1), FrameBound::Following(2));
        assert_eq!(
            w.frame_spec,
            Some((
                FrameType::Row,
                FrameBound::Preceding(1),
                FrameBound::Following(2)
            ))
        );
    }

    #[test]
    fn range_between_sets_range_frame() {
        let w =
            WindowSpec::new().range_between(FrameBound::UnboundedPreceding, FrameBound::CurrentRow);
        assert_eq!(
            w.frame_spec,
            Some((
                FrameType::Range,
                FrameBound::UnboundedPreceding,
                FrameBound::CurrentRow
            ))
        );
    }

    #[test]
    fn a_later_frame_replaces_an_earlier_one() {
        let w = WindowSpec::new()
            .rows_between(FrameBound::UnboundedPreceding, FrameBound::CurrentRow)
            .range_between(FrameBound::CurrentRow, FrameBound::UnboundedFollowing);
        assert_eq!(
            w.frame_spec,
            Some((
                FrameType::Range,
                FrameBound::CurrentRow,
                FrameBound::UnboundedFollowing
            ))
        );
    }

    #[test]
    fn window_static_constructors_match_spec_builders() {
        assert_eq!(
            Window::partition_by(vec![col("a")]),
            WindowSpec::new().partition_by(vec![col("a")])
        );
        assert_eq!(
            Window::order_by(vec![sort("a")]),
            WindowSpec::new().order_by(vec![sort("a")])
        );
        assert_eq!(
            Window::rows_between(FrameBound::Preceding(1), FrameBound::CurrentRow),
            WindowSpec::new().rows_between(FrameBound::Preceding(1), FrameBound::CurrentRow)
        );
        assert_eq!(
            Window::range_between(FrameBound::CurrentRow, FrameBound::Following(3)),
            WindowSpec::new().range_between(FrameBound::CurrentRow, FrameBound::Following(3))
        );
    }

    /// `partition_by` and `order_by` take Columns, as `DataFrame::select` does, and the
    /// `Vec<Expression>` and `Vec<SortOrder>` that they took before, which they keep as they
    /// kept them. A Column of `order_by` that is not a sort order is ascending, nulls first,
    /// as PySpark's `WindowSpec.orderBy` sorts it.
    #[test]
    fn spec_builders_take_columns_and_expressions_alike() {
        use crate::column::col as column;

        let before = WindowSpec {
            partition_spec: vec![col("a"), col("b")],
            order_spec: vec![SortOrder::desc_nulls_last(col("c")), sort("d")],
            frame_spec: None,
        };
        let of_columns = || {
            WindowSpec::new()
                .partition_by([column("a"), column("b")])
                .order_by([column("c").desc(), column("d")])
        };
        assert_eq!(of_columns(), before);
        assert_eq!(
            WindowSpec::new()
                .partition_by(vec![col("a"), col("b")])
                .order_by(vec![SortOrder::desc_nulls_last(col("c")), sort("d")]),
            before
        );
        assert_eq!(
            Window::partition_by(["a", "b"]).order_by([column("c").desc(), column("d")]),
            before
        );
        assert_eq!(
            Window::order_by([column("c").desc(), column("d")]).partition_by(["a", "b"]),
            before
        );
        // And so are the window expressions over them.
        let over = |spec| crate::functions::sum(column("x")).over(spec).to_proto();
        assert_eq!(over(of_columns()), over(before));
    }

    #[test]
    fn frame_boundaries_are_sent_as_pyspark_sends_them() {
        use spark_connect_proto::expression::literal::LiteralType;
        use spark_connect_proto::expression::window::window_frame::frame_boundary::Boundary;
        use spark_connect_proto::expression::ExprType;

        let literals = |spec: WindowSpec| {
            let column = crate::functions::sum(crate::column::col("id")).over(spec);
            let Some(ExprType::Window(window)) = column.expression().to_proto().expr_type else {
                panic!("expected a window expression");
            };
            let frame = window.frame_spec.expect("a frame");
            [frame.lower.unwrap(), frame.upper.unwrap()].map(|bound| match bound.boundary {
                Some(Boundary::Value(value)) => match value.expr_type {
                    Some(ExprType::Literal(literal)) => literal.literal_type,
                    other => panic!("expected a literal, got {other:?}"),
                },
                other => panic!("expected a value, got {other:?}"),
            })
        };
        // As PySpark's: the long literals of a range frame, of values out of the range of an
        // int too, not truncated.
        assert_eq!(
            literals(
                WindowSpec::new()
                    .range_between(FrameBound::Preceding(1 << 40), FrameBound::Following(1))
            ),
            [
                Some(LiteralType::Long(-(1 << 40))),
                Some(LiteralType::Long(1))
            ]
        );
        // The int literals of a row frame, of which a value out of the range of an int fails the
        // request with PySpark's `VALUE_NOT_BETWEEN` (a serialization that does not fail sends
        // its long).
        let beyond_an_int = WindowSpec::new()
            .rows_between(FrameBound::Preceding(1), FrameBound::Following(1 << 40));
        assert_eq!(
            literals(beyond_an_int.clone()),
            [
                Some(LiteralType::Integer(-1)),
                Some(LiteralType::Long(1 << 40))
            ]
        );
        let not_between = |spec: WindowSpec| {
            crate::functions::sum(crate::column::col("id"))
                .over(spec)
                .try_to_proto()
                .unwrap_err()
                .message()
        };
        assert_eq!(
            not_between(beyond_an_int),
            "[VALUE_NOT_BETWEEN] Value for `end` must be between -2147483648 and 2147483647."
        );
    }

    /// As PySpark's `rowsBetween` and `rangeBetween`, of the ints of the bounds, and its
    /// `WindowExpression`, which sends a bound of 0 as the current row, and the start of
    /// `Window.unboundedPreceding` and the end of `Window.unboundedFollowing` as unbounded: a
    /// start of `-sys.maxsize` or less and an end of `sys.maxsize` or more are unbounded, and an
    /// unbounded bound on the other side is the literal of its int.
    #[test]
    fn frame_boundaries_of_the_current_row_and_unbounded_are_sent_as_pyspark() {
        use crate::expression::LiteralExpression;
        use spark_connect_core::error::SparkErrorKind;
        use spark_connect_proto::expression::window::window_frame::frame_boundary::Boundary;
        use spark_connect_proto::expression::ExprType;
        use FrameBound::*;

        let over = |spec: WindowSpec| crate::functions::sum(crate::column::col("id")).over(spec);
        let boundaries = |spec: WindowSpec| {
            let Some(ExprType::Window(window)) = over(spec).expression().to_proto().expr_type
            else {
                panic!("expected a window expression");
            };
            let frame = window.frame_spec.expect("a frame");
            [frame.lower.unwrap().boundary, frame.upper.unwrap().boundary]
        };
        let long = |n: i64| {
            let literal = Expression::literal(LiteralExpression::long(n)).to_proto();
            Some(Boundary::Value(Box::new(literal)))
        };
        let current = Some(Boundary::CurrentRow(true));
        let unbounded = Some(Boundary::Unbounded(true));
        for spec in [
            WindowSpec::new().rows_between(Preceding(0), Following(0)),
            WindowSpec::new().range_between(Following(0), Preceding(0)),
        ] {
            assert_eq!(boundaries(spec), [current.clone(), current.clone()]);
        }
        for spec in [
            WindowSpec::new().rows_between(Preceding(i64::MAX), Following(i64::MAX)),
            WindowSpec::new().range_between(UnboundedPreceding, UnboundedFollowing),
            WindowSpec::new().range_between(Preceding(i64::MAX), Preceding(i64::MIN)),
        ] {
            assert_eq!(boundaries(spec), [unbounded.clone(), unbounded.clone()]);
        }
        // `rangeBetween(Window.unboundedFollowing, 0)` and `rangeBetween(0,
        // Window.unboundedPreceding)`: the long literals of the unbounded ints.
        assert_eq!(
            boundaries(WindowSpec::new().range_between(UnboundedFollowing, CurrentRow)),
            [long(i64::MAX), current.clone()]
        );
        assert_eq!(
            boundaries(WindowSpec::new().range_between(CurrentRow, UnboundedPreceding)),
            [current.clone(), long(i64::MIN)]
        );
        // Which a row frame refuses, as an int beyond an int32, of the start first.
        let error = |spec: WindowSpec| over(spec).try_to_proto().unwrap_err();
        for (spec, name) in [
            (
                WindowSpec::new().rows_between(UnboundedFollowing, CurrentRow),
                "start",
            ),
            (
                WindowSpec::new().rows_between(CurrentRow, UnboundedPreceding),
                "end",
            ),
            (
                WindowSpec::new().rows_between(Preceding(1 << 31 | 1), Following(1 << 31)),
                "start",
            ),
        ] {
            let error = error(spec);
            assert_eq!(error.kind, SparkErrorKind::ValueError);
            assert_eq!(
                error.message(),
                format!(
                    "[VALUE_NOT_BETWEEN] Value for `{name}` must be between -2147483648 and \
                     2147483647."
                )
            );
        }
        // The start of 2**63 of a range frame, beyond a long: protobuf's `ValueError`.
        let error = error(WindowSpec::new().range_between(Preceding(i64::MIN), CurrentRow));
        assert_eq!(
            (error.kind, error.message()),
            (
                SparkErrorKind::BuiltinValueError,
                "Value out of range: 9223372036854775808".to_string()
            )
        );
        // And the repr of the window is of the ints of its bounds, as PySpark's.
        assert_eq!(
            WindowSpec::new()
                .rows_between(Preceding(i64::MAX), Following(i64::MAX))
                .repr(),
            "WindowSpec(WindowFrame(ROW_FRAME, -9223372036854775808, 9223372036854775807))"
        );
        assert_eq!(
            WindowSpec::new()
                .range_between(UnboundedFollowing, UnboundedPreceding)
                .repr(),
            "WindowSpec(WindowFrame(RANGE_FRAME, 9223372036854775807, -9223372036854775808))"
        );
    }

    #[test]
    fn frame_bound_constants() {
        assert_eq!(
            Window::unbounded_preceding(),
            FrameBound::UnboundedPreceding
        );
        assert_eq!(Window::current_row(), FrameBound::CurrentRow);
        assert_eq!(
            Window::unbounded_following(),
            FrameBound::UnboundedFollowing
        );
    }
}
