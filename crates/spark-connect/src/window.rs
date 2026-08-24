//! Window functions and specifications mirroring PySpark's `pyspark.sql.window`.
//!
//! Provides the `Window` static builder and `WindowSpec` for defining window partitions,
//! ordering, and frame boundaries used with `Column.over()`.

use crate::expression::{Expression, SortOrder};

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
    /// Create a new empty WindowSpec.
    pub fn new() -> Self {
        Self {
            partition_spec: Vec::new(),
            order_spec: Vec::new(),
            frame_spec: None,
        }
    }

    /// Add partition columns to this window specification.
    pub fn partition_by(mut self, cols: Vec<Expression>) -> Self {
        self.partition_spec.extend(cols);
        self
    }

    /// Set ordering for this window specification.
    pub fn order_by(mut self, sorts: Vec<SortOrder>) -> Self {
        self.order_spec.extend(sorts);
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
    pub fn partition_by(cols: Vec<Expression>) -> WindowSpec {
        WindowSpec::new().partition_by(cols)
    }

    /// Create a WindowSpec ordered by the given sort orders.
    pub fn order_by(sorts: Vec<SortOrder>) -> WindowSpec {
        WindowSpec::new().order_by(sorts)
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
