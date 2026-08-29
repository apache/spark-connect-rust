//! PyO3 wrappers for window functions and specifications.

use pyo3::prelude::*;
use spark_connect::expression::{Expression, NullOrdering, SortOrder};
use spark_connect::window::{FrameBound, WindowSpec};

use crate::functions::to_column;

/// Python wrapper for a window frame bound.
#[pyclass(name = "FrameBound", from_py_object)]
#[derive(Clone)]
pub struct PyFrameBound {
    pub(crate) bound: FrameBound,
}

impl PyFrameBound {
    pub fn new(bound: FrameBound) -> Self {
        PyFrameBound { bound }
    }
}

/// pyspark uses plain integer frame bounds with sentinel extremes; mirror them.
const UNBOUNDED_PRECEDING: i64 = i64::MIN;
const UNBOUNDED_FOLLOWING: i64 = i64::MAX;

/// Resolve a frame-bound argument that is either a plain int (pyspark style, with
/// sentinel extremes and negative=preceding/positive=following) or a `FrameBound`.
fn to_frame_bound(v: &Bound<'_, PyAny>) -> PyResult<FrameBound> {
    if let Ok(fb) = v.extract::<PyFrameBound>() {
        return Ok(fb.bound);
    }
    let n: i64 = v.extract().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>("frame bound must be an int or FrameBound")
    })?;
    Ok(match n {
        UNBOUNDED_PRECEDING => FrameBound::UnboundedPreceding,
        UNBOUNDED_FOLLOWING => FrameBound::UnboundedFollowing,
        0 => FrameBound::CurrentRow,
        n if n < 0 => FrameBound::Preceding(-n),
        n => FrameBound::Following(n),
    })
}

#[pymethods]
impl PyFrameBound {
    fn __repr__(&self) -> String {
        match &self.bound {
            FrameBound::UnboundedPreceding => "FrameBound.unboundedPreceding".to_string(),
            FrameBound::Preceding(n) => format!("FrameBound.preceding({})", n),
            FrameBound::CurrentRow => "FrameBound.currentRow".to_string(),
            FrameBound::Following(n) => format!("FrameBound.following({})", n),
            FrameBound::UnboundedFollowing => "FrameBound.unboundedFollowing".to_string(),
        }
    }
}

/// Python wrapper for a WindowSpec.
#[pyclass(name = "WindowSpec", from_py_object)]
#[derive(Clone)]
pub struct PyWindowSpec {
    pub(crate) spec: WindowSpec,
}

impl PyWindowSpec {
    pub fn new(spec: WindowSpec) -> Self {
        PyWindowSpec { spec }
    }
}

#[pymethods]
impl PyWindowSpec {
    /// Partition the window by the given columns.
    #[pyo3(signature = (*cols))]
    fn partitionBy(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyWindowSpec> {
        let mut expressions = Vec::new();
        for col in cols {
            expressions.push(to_column(&col)?.expression().clone());
        }
        let new_spec = self.spec.clone().partition_by(expressions);
        Ok(PyWindowSpec::new(new_spec))
    }

    /// Order the window by the given columns.
    #[pyo3(signature = (*cols))]
    fn orderBy(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyWindowSpec> {
        let mut sort_orders = Vec::new();
        for col in cols {
            let column = to_column(&col)?;
            // Extract sort order from the column
            if let Expression::SortOrder(sort) = column.expression() {
                sort_orders.push((**sort).clone());
            } else {
                // Default to ascending
                sort_orders.push(SortOrder {
                    child: column.expression().clone(),
                    ascending: true,
                    null_ordering: NullOrdering::Last,
                });
            }
        }
        let new_spec = self.spec.clone().order_by(sort_orders);
        Ok(PyWindowSpec::new(new_spec))
    }

    /// Define a ROWS frame between start and end bounds (ints or FrameBounds).
    fn rowsBetween(
        &self,
        start: &Bound<'_, PyAny>,
        end: &Bound<'_, PyAny>,
    ) -> PyResult<PyWindowSpec> {
        let new_spec = self
            .spec
            .clone()
            .rows_between(to_frame_bound(start)?, to_frame_bound(end)?);
        Ok(PyWindowSpec::new(new_spec))
    }

    /// Define a RANGE frame between start and end bounds (ints or FrameBounds).
    fn rangeBetween(
        &self,
        start: &Bound<'_, PyAny>,
        end: &Bound<'_, PyAny>,
    ) -> PyResult<PyWindowSpec> {
        let new_spec = self
            .spec
            .clone()
            .range_between(to_frame_bound(start)?, to_frame_bound(end)?);
        Ok(PyWindowSpec::new(new_spec))
    }

    fn __repr__(&self) -> String {
        "WindowSpec()".to_string()
    }
}

/// Python wrapper for the Window static builder.
#[pyclass(name = "Window")]
pub struct PyWindow;

#[pymethods]
impl PyWindow {
    /// UNBOUNDED PRECEDING sentinel (matches pyspark's integer value).
    #[classattr]
    fn unboundedPreceding() -> i64 {
        UNBOUNDED_PRECEDING
    }

    /// CURRENT ROW sentinel.
    #[classattr]
    fn currentRow() -> i64 {
        0
    }

    /// UNBOUNDED FOLLOWING sentinel (matches pyspark's integer value).
    #[classattr]
    fn unboundedFollowing() -> i64 {
        UNBOUNDED_FOLLOWING
    }

    /// Create a WindowSpec partitioned by the given columns.
    #[pyo3(signature = (*cols))]
    #[staticmethod]
    fn partitionBy(cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyWindowSpec> {
        let mut expressions = Vec::new();
        for col in cols {
            expressions.push(to_column(&col)?.expression().clone());
        }
        let spec = WindowSpec::new().partition_by(expressions);
        Ok(PyWindowSpec::new(spec))
    }

    /// Create a WindowSpec ordered by the given columns.
    #[pyo3(signature = (*cols))]
    #[staticmethod]
    fn orderBy(cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyWindowSpec> {
        let mut sort_orders = Vec::new();
        for col in cols {
            let column = to_column(&col)?;
            // Extract sort order from the column
            if let Expression::SortOrder(sort) = column.expression() {
                sort_orders.push((**sort).clone());
            } else {
                // Default to ascending
                sort_orders.push(SortOrder {
                    child: column.expression().clone(),
                    ascending: true,
                    null_ordering: NullOrdering::Last,
                });
            }
        }
        let spec = WindowSpec::new().order_by(sort_orders);
        Ok(PyWindowSpec::new(spec))
    }

    /// Create a WindowSpec with a ROWS frame between the given bounds.
    #[staticmethod]
    fn rowsBetween(start: &Bound<'_, PyAny>, end: &Bound<'_, PyAny>) -> PyResult<PyWindowSpec> {
        let spec = WindowSpec::new().rows_between(to_frame_bound(start)?, to_frame_bound(end)?);
        Ok(PyWindowSpec::new(spec))
    }

    /// Create a WindowSpec with a RANGE frame between the given bounds.
    #[staticmethod]
    fn rangeBetween(start: &Bound<'_, PyAny>, end: &Bound<'_, PyAny>) -> PyResult<PyWindowSpec> {
        let spec = WindowSpec::new().range_between(to_frame_bound(start)?, to_frame_bound(end)?);
        Ok(PyWindowSpec::new(spec))
    }

    fn __repr__(&self) -> String {
        "Window".to_string()
    }
}
