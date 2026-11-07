//! PyO3 wrappers for window functions and specifications.

use pyo3::prelude::*;
use spark_connect::expression::{Expression, ExpressionNode, SortOrder};
use spark_connect::window::{FrameBound, WindowSpec};

/// The columns of `Window.partitionBy` and `orderBy`, as PySpark Connect's `_to_cols`: a single
/// list is the columns, each a Column or a column name (not a literal), else
/// `NOT_EXPECTED_TYPE` of the argument `col`, as its `F._to_col`.
fn window_columns(cols: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<spark_connect::column::Column>> {
    let cols = match cols.as_slice() {
        [only] if only.is_instance_of::<pyo3::types::PyList>() => {
            only.try_iter()?.collect::<PyResult<Vec<_>>>()?
        }
        _ => cols,
    };
    cols.iter()
        .map(|c| {
            if let Ok(column) = c.extract::<PyRef<'_, crate::column::PyColumn>>() {
                return Ok(column.column.clone());
            }
            if c.is_instance_of::<pyo3::types::PyString>() {
                // As PySpark's `F._to_col`, by `col`, of its origin.
                let _origin = crate::origin::with_origin("col");
                return Ok(spark_connect::functions::col(&c.extract::<String>()?));
            }
            let type_name = c.get_type().name()?.to_string();
            Err(crate::column::raise_pyspark(
                c.py(),
                "PySparkTypeError",
                "NOT_EXPECTED_TYPE",
                &[
                    ("expected_type", "Column or str"),
                    ("arg_name", "col"),
                    ("arg_type", &type_name),
                ],
            ))
        })
        .collect()
}

/// Build partition expressions from ColumnOrName args (a bare string is a COLUMN
/// NAME, not a literal — mirrors pyspark Window.partitionBy).
fn to_partition_exprs(cols: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Expression>> {
    Ok(window_columns(cols)?
        .iter()
        .map(|c| c.expression().clone())
        .collect())
}

/// Build sort orders from ColumnOrName args; a bare string becomes an ascending
/// SortOrder over that column (not a string literal). A column that is not a sort order is
/// ascending with nulls first, as `SortOrder(expr)` of PySpark defaults to.
fn to_sort_orders(cols: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<SortOrder>> {
    let mut sort_orders = Vec::new();
    for column in window_columns(cols)? {
        if let ExpressionNode::SortOrder(sort) = column.expression().node() {
            sort_orders.push((**sort).clone());
        } else {
            sort_orders.push(SortOrder::asc_nulls_first(column.expression().clone()));
        }
    }
    Ok(sort_orders)
}

/// Python wrapper for a window frame bound.
#[pyclass(name = "FrameBound", module = "pyspark.sql.window", from_py_object)]
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

/// Resolve a frame-bound argument, the `start` of a frame if `start`, else its end: a plain
/// int (negative=preceding/positive=following) or a `FrameBound`. As PySpark's
/// `rowsBetween` and `rangeBetween`, a start of `-sys.maxsize` or less is unbounded preceding
/// and an end of `sys.maxsize` or more unbounded following, also of a float, such as
/// `float("-inf")` and `float("inf")`, while a start of `Window.unboundedFollowing` or an end of
/// `Window.unboundedPreceding` is a value.
fn to_frame_bound(v: &Bound<'_, PyAny>, start: bool) -> PyResult<FrameBound> {
    if let Ok(fb) = v.extract::<PyFrameBound>() {
        return Ok(fb.bound);
    }
    if !v.is_instance_of::<pyo3::types::PyInt>() {
        if v.is_instance_of::<pyo3::types::PyFloat>() {
            // As PySpark's: a bound past the threshold of its side is unbounded, and
            // WindowFrame asserts that another is an int.
            if start && v.le(-i64::MAX)? {
                return Ok(FrameBound::UnboundedPreceding);
            }
            if !start && v.ge(i64::MAX)? {
                return Ok(FrameBound::UnboundedFollowing);
            }
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "frame bound must be an int or FrameBound",
        ));
    }
    let n: i64 = match v.extract() {
        Ok(n) => n,
        // An int beyond a long is past the thresholds of its side, or else out of range.
        Err(_) if start && v.lt(0)? => return Ok(FrameBound::UnboundedPreceding),
        Err(_) if !start && v.gt(0)? => return Ok(FrameBound::UnboundedFollowing),
        Err(err) => return Err(err),
    };
    Ok(match n {
        n if start && n <= -i64::MAX => FrameBound::UnboundedPreceding,
        n if !start && n == i64::MAX => FrameBound::UnboundedFollowing,
        0 => FrameBound::CurrentRow,
        n if n < 0 => FrameBound::Preceding(n.checked_neg().unwrap_or(i64::MAX)),
        n => FrameBound::Following(n),
    })
}

/// The bounds of a row frame, as PySpark's, whose plan raises `VALUE_NOT_BETWEEN` for a
/// boundary of a value that is not an int.
fn row_frame_bounds(
    py: Python<'_>,
    start: &Bound<'_, PyAny>,
    end: &Bound<'_, PyAny>,
) -> PyResult<(FrameBound, FrameBound)> {
    let bounds = (to_frame_bound(start, true)?, to_frame_bound(end, false)?);
    for (name, bound) in [("start", &bounds.0), ("end", &bounds.1)] {
        let value = match bound {
            FrameBound::Preceding(n) => n.saturating_neg(),
            FrameBound::Following(n) => *n,
            _ => continue,
        };
        if i32::try_from(value).is_err() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "VALUE_NOT_BETWEEN",
                &[
                    ("arg_name", name),
                    ("min", &i32::MIN.to_string()),
                    ("max", &i32::MAX.to_string()),
                ],
            ));
        }
    }
    Ok(bounds)
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
#[pyclass(name = "WindowSpec", module = "pyspark.sql.window", from_py_object)]
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
        let new_spec = self.spec.clone().partition_by(to_partition_exprs(cols)?);
        Ok(PyWindowSpec::new(new_spec))
    }

    /// Order the window by the given columns.
    #[pyo3(signature = (*cols))]
    fn orderBy(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyWindowSpec> {
        let new_spec = self.spec.clone().order_by(to_sort_orders(cols)?);
        Ok(PyWindowSpec::new(new_spec))
    }

    /// Define a ROWS frame between start and end bounds (ints or FrameBounds).
    fn rowsBetween(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyAny>,
        end: &Bound<'_, PyAny>,
    ) -> PyResult<PyWindowSpec> {
        let (start, end) = row_frame_bounds(py, start, end)?;
        let new_spec = self.spec.clone().rows_between(start, end);
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
            .range_between(to_frame_bound(start, true)?, to_frame_bound(end, false)?);
        Ok(PyWindowSpec::new(new_spec))
    }

    /// As `WindowSpec.__repr__` of PySpark Connect.
    fn __repr__(&self) -> String {
        self.spec.repr()
    }
}

/// Python wrapper for the Window static builder.
#[pyclass(name = "Window", module = "pyspark.sql.window")]
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
        let spec = WindowSpec::new().partition_by(to_partition_exprs(cols)?);
        Ok(PyWindowSpec::new(spec))
    }

    /// Create a WindowSpec ordered by the given columns.
    #[pyo3(signature = (*cols))]
    #[staticmethod]
    fn orderBy(cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyWindowSpec> {
        let spec = WindowSpec::new().order_by(to_sort_orders(cols)?);
        Ok(PyWindowSpec::new(spec))
    }

    /// Create a WindowSpec with a ROWS frame between the given bounds.
    #[staticmethod]
    fn rowsBetween(
        py: Python<'_>,
        start: &Bound<'_, PyAny>,
        end: &Bound<'_, PyAny>,
    ) -> PyResult<PyWindowSpec> {
        let (start, end) = row_frame_bounds(py, start, end)?;
        Ok(PyWindowSpec::new(
            WindowSpec::new().rows_between(start, end),
        ))
    }

    /// Create a WindowSpec with a RANGE frame between the given bounds.
    #[staticmethod]
    fn rangeBetween(start: &Bound<'_, PyAny>, end: &Bound<'_, PyAny>) -> PyResult<PyWindowSpec> {
        let spec = WindowSpec::new()
            .range_between(to_frame_bound(start, true)?, to_frame_bound(end, false)?);
        Ok(PyWindowSpec::new(spec))
    }

    fn __repr__(&self) -> String {
        "Window".to_string()
    }
}
