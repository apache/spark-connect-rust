//! PyO3 wrappers for window functions and specifications.

use pyo3::prelude::*;
use spark_connect::expression::{Expression, NullOrdering, SortOrder};
use spark_connect::window::{FrameBound, WindowSpec};

use crate::column::PyColumn;
use crate::functions::to_column;

/// Python wrapper for a window frame bound.
#[pyclass(name = "FrameBound")]
#[derive(Clone)]
pub struct PyFrameBound {
    pub(crate) bound: FrameBound,
}

impl PyFrameBound {
    pub fn new(bound: FrameBound) -> Self {
        PyFrameBound { bound }
    }
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
#[pyclass(name = "WindowSpec")]
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

    /// Define a ROWS frame between start and end bounds.
    fn rowsBetween(&self, start: &PyFrameBound, end: &PyFrameBound) -> PyWindowSpec {
        let new_spec = self
            .spec
            .clone()
            .rows_between(start.bound.clone(), end.bound.clone());
        PyWindowSpec::new(new_spec)
    }

    /// Define a RANGE frame between start and end bounds.
    fn rangeBetween(&self, start: &PyFrameBound, end: &PyFrameBound) -> PyWindowSpec {
        let new_spec = self
            .spec
            .clone()
            .range_between(start.bound.clone(), end.bound.clone());
        PyWindowSpec::new(new_spec)
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
    /// UNBOUNDED PRECEDING frame bound.
    #[classattr]
    fn unboundedPreceding() -> PyFrameBound {
        PyFrameBound::new(FrameBound::UnboundedPreceding)
    }

    /// CURRENT ROW frame bound.
    #[classattr]
    fn currentRow() -> PyFrameBound {
        PyFrameBound::new(FrameBound::CurrentRow)
    }

    /// UNBOUNDED FOLLOWING frame bound.
    #[classattr]
    fn unboundedFollowing() -> PyFrameBound {
        PyFrameBound::new(FrameBound::UnboundedFollowing)
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
    fn rowsBetween(start: &PyFrameBound, end: &PyFrameBound) -> PyWindowSpec {
        let spec = WindowSpec::new().rows_between(start.bound.clone(), end.bound.clone());
        PyWindowSpec::new(spec)
    }

    /// Create a WindowSpec with a RANGE frame between the given bounds.
    #[staticmethod]
    fn rangeBetween(start: &PyFrameBound, end: &PyFrameBound) -> PyWindowSpec {
        let spec = WindowSpec::new().range_between(start.bound.clone(), end.bound.clone());
        PyWindowSpec::new(spec)
    }

    fn __repr__(&self) -> String {
        "Window".to_string()
    }
}
