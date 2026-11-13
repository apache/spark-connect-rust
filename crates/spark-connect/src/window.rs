//! Window functions and specifications mirroring PySpark's `pyspark.sql.window`.
//!
//! Provides the `Window` static builder and `WindowSpec` for defining window partitions,
//! ordering, and frame boundaries used with `Column.over()`.

use crate::column::{expressions_of, Column};
use crate::expression::{Expression, ExpressionKind, SortOrder};

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

    /// Create a new empty WindowSpec.
    pub fn new() -> Self {
        Self {
            partition_spec: Vec::new(),
            order_spec: Vec::new(),
            frame_spec: None,
        }
    }

    /// Add partition columns to this window specification.
    pub fn partition_by<C: Into<Column>>(mut self, cols: impl IntoIterator<Item = C>) -> Self {
        self.partition_spec.extend(expressions_of(cols));
        self
    }

    /// Set ordering for this window specification. A sort order, of [`Column::asc`],
    /// [`Column::desc`] and their nulls variants, or a [`SortOrder`], sorts as it says; any
    /// other column is sorted ascending, nulls first, as PySpark's `WindowSpec.orderBy` sorts it.
    pub fn order_by<C: Into<Column>>(mut self, cols: impl IntoIterator<Item = C>) -> Self {
        self.order_spec.extend(expressions_of(cols).into_iter().map(
            |expr| match expr.node().kind() {
                ExpressionKind::SortOrder {
                    ascending,
                    null_ordering,
                } => SortOrder {
                    child: expr.node().children()[0].clone(),
                    ascending: *ascending,
                    null_ordering: *null_ordering,
                },
                _ => SortOrder::asc_nulls_first(expr),
            },
        ));
        self
    }

    /// Set the frame specification for ROWS.
    pub fn rows_between(mut self, start: FrameBound, end: FrameBound) -> Self {
        self.frame_spec = Some((FrameType::Row, start, end));
        self
    }

    /// Set the frame specification for RANGE.
    pub fn range_between(mut self, start: FrameBound, end: FrameBound) -> Self {
        self.frame_spec = Some((FrameType::Range, start, end));
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
        // The int literals of a row frame, and a long for a value out of the range of an int,
        // for the server to reject.
        assert_eq!(
            literals(
                WindowSpec::new()
                    .rows_between(FrameBound::Preceding(1), FrameBound::Following(1 << 40))
            ),
            [
                Some(LiteralType::Integer(-1)),
                Some(LiteralType::Long(1 << 40))
            ]
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
