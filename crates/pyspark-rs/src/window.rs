//! PyO3 wrappers for window functions and specifications.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use spark_connect::column::Column;
use spark_connect::expression::{Expression, ExpressionNode, NullOrdering, SortOrder};
use spark_connect::window::{FrameBound, FrameType, WindowSpec};

use crate::column::PyColumn;

/// The columns of `Window.partitionBy` and `orderBy`, and of `TableArg.partitionBy` and
/// `orderBy`, as PySpark Connect's `_to_cols`: a single list is the columns, each a Column or
/// a column name (not a literal), else `NOT_EXPECTED_TYPE` of the argument `col`, as its
/// `F._to_col`.
pub(crate) fn window_columns(
    cols: Vec<Bound<'_, PyAny>>,
) -> PyResult<Vec<spark_connect::column::Column>> {
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
                let _origin = crate::origin::with_origin("col")?;
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

/// The literal fields of the boundary of a value, which PySpark's `WindowExpression` sets: an
/// int32 in a row frame, and a long in a range frame.
const ROW_BOUNDARY_FIELD: &str = "spark.connect.Expression.Literal.integer";
const RANGE_BOUNDARY_FIELD: &str = "spark.connect.Expression.Literal.long";

/// A frame-bound argument once compared with the threshold of its side: a `FrameBound` as it
/// is, the unbounded bound of its side past the threshold, or else the value of the argument.
enum FrameArgument<'a, 'py> {
    Bound(FrameBound),
    Value(&'a Bound<'py, PyAny>),
}

/// The frame-bound argument `v`, the start of a frame if `start`, else its end, compared as
/// PySpark's `rowsBetween` and `rangeBetween` compare it, with Python's `<=` and `>=`: a start of
/// `Window._PRECEDING_THRESHOLD` (`-sys.maxsize`) or less is unbounded preceding and an end of
/// `Window._FOLLOWING_THRESHOLD` (`sys.maxsize`) or more unbounded following, of any type, so
/// also of a float such as `float("inf")`, and a type that does not compare with an int raises
/// the `TypeError` of the comparison. A start of `Window.unboundedFollowing` or an end of
/// `Window.unboundedPreceding` is a value.
fn compared_frame_argument<'a, 'py>(
    v: &'a Bound<'py, PyAny>,
    start: bool,
) -> PyResult<FrameArgument<'a, 'py>> {
    if let Ok(fb) = v.extract::<PyFrameBound>() {
        return Ok(FrameArgument::Bound(fb.bound));
    }
    if start && v.le(-i64::MAX)? {
        return Ok(FrameArgument::Bound(FrameBound::UnboundedPreceding));
    }
    if !start && v.ge(i64::MAX)? {
        return Ok(FrameArgument::Bound(FrameBound::UnboundedFollowing));
    }
    Ok(FrameArgument::Value(v))
}

/// The bound of the offset `n`: a negative one is preceding, and a positive one following. The
/// offset of `Window.unboundedPreceding`, as the end of a range frame, is a following one, as
/// the count of the rows that it precedes is beyond a long.
fn frame_bound_of(n: i64) -> FrameBound {
    match n {
        0 => FrameBound::CurrentRow,
        n if n > 0 => FrameBound::Following(n),
        n => n
            .checked_neg()
            .map_or(FrameBound::Following(n), FrameBound::Preceding),
    }
}

/// PySpark's `VALUE_NOT_BETWEEN` of the boundary `name` of a row frame beyond an int32.
fn boundary_not_between(py: Python<'_>, name: &str) -> PyErr {
    crate::column::raise_pyspark(
        py,
        "PySparkValueError",
        "VALUE_NOT_BETWEEN",
        &[
            ("arg_name", name),
            ("min", &i32::MIN.to_string()),
            ("max", &i32::MAX.to_string()),
        ],
    )
}

/// The bound `name` of a frame argument, as the plan of PySpark's `WindowExpression` converts
/// it, in a row frame if `rows`, else in a range frame: an int of 0 (False too) is the current
/// row; a row frame sets any other int of an int32 in the int32 literal, which protobuf's default
/// (upb) backend refuses of a bool, and raises `VALUE_NOT_BETWEEN` of an int beyond an int32;
/// and a range frame sets it in the long literal, which refuses a bool and an int beyond a long,
/// as [`crate::stat_arguments::proto_int`] does. A `FrameBound` of a row frame is checked as
/// such an int.
fn converted_frame_bound(
    py: Python<'_>,
    argument: FrameArgument<'_, '_>,
    name: &str,
    rows: bool,
) -> PyResult<FrameBound> {
    let value = match argument {
        FrameArgument::Bound(bound) => {
            let offset = match &bound {
                FrameBound::Preceding(n) => Some(n.saturating_neg()),
                FrameBound::Following(n) => Some(*n),
                _ => None,
            };
            if rows && offset.is_some_and(|n| i32::try_from(n).is_err()) {
                return Err(boundary_not_between(py, name));
            }
            return Ok(bound);
        }
        FrameArgument::Value(value) => value,
    };
    let long = value.extract::<i64>().ok();
    if long == Some(0) {
        return Ok(FrameBound::CurrentRow);
    }
    if !rows {
        return crate::stat_arguments::proto_int::<i64>(value, RANGE_BOUNDARY_FIELD)
            .map(frame_bound_of);
    }
    match long.and_then(|n| i32::try_from(n).ok()) {
        Some(_) if value.is_instance_of::<pyo3::types::PyBool>() => Err(
            crate::stat_arguments::bool_not_proto_int(ROW_BOUNDARY_FIELD),
        ),
        Some(n) => Ok(frame_bound_of(n.into())),
        None => Err(boundary_not_between(py, name)),
    }
}

/// The bounds of `rowsBetween(start, end)` if `rows`, else of `rangeBetween(start, end)`, a
/// plain int (negative is preceding, positive following) or a `FrameBound` each, as PySpark's:
/// it compares the start and then the end with the thresholds of their sides, and WindowFrame
/// then asserts that the start and then the end that is not past its threshold is an int (a
/// bool included), a bare `AssertionError` of a NumPy int, a float or a Decimal. The errors of
/// the plan, which converts the start and then the end, are raised here, after those.
fn frame_bounds(
    py: Python<'_>,
    start: &Bound<'_, PyAny>,
    end: &Bound<'_, PyAny>,
    rows: bool,
) -> PyResult<(FrameBound, FrameBound)> {
    let start = compared_frame_argument(start, true)?;
    let end = compared_frame_argument(end, false)?;
    for argument in [&start, &end] {
        if let FrameArgument::Value(value) = argument {
            if !value.is_instance_of::<pyo3::types::PyInt>() {
                return Err(pyo3::exceptions::PyAssertionError::new_err(()));
            }
        }
    }
    Ok((
        converted_frame_bound(py, start, "start", rows)?,
        converted_frame_bound(py, end, "end", rows)?,
    ))
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
#[pyclass(
    name = "WindowSpec",
    module = "pyspark.sql.window",
    from_py_object,
    weakref,
    dict
)]
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
        let (start, end) = frame_bounds(py, start, end, true)?;
        let new_spec = self.spec.clone().rows_between(start, end);
        Ok(PyWindowSpec::new(new_spec))
    }

    /// Define a RANGE frame between start and end bounds (ints or FrameBounds).
    fn rangeBetween(
        &self,
        py: Python<'_>,
        start: &Bound<'_, PyAny>,
        end: &Bound<'_, PyAny>,
    ) -> PyResult<PyWindowSpec> {
        let (start, end) = frame_bounds(py, start, end, false)?;
        let new_spec = self.spec.clone().range_between(start, end);
        Ok(PyWindowSpec::new(new_spec))
    }

    /// As `WindowSpec.__repr__` of PySpark Connect.
    fn __repr__(&self) -> String {
        self.spec.repr()
    }

    /// A WindowSpec pickles as the partition and sort Columns and the frame of it, which the
    /// function `pyspark.sql.connect._window_reduce.window_of` gives back as a WindowSpec built
    /// through `Window`, also on a Python worker of the server, as PySpark's WindowSpec pickles
    /// the Columns it holds. A sort order is given as the Column of its child and the method of
    /// its direction and nulls, as an unpickled Column is not a sort order of its own.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let column = |expr: &Expression| Py::new(py, PyColumn::new(Column::new(expr.clone())));
        let partition = self
            .spec
            .partition_spec
            .iter()
            .map(column)
            .collect::<PyResult<Vec<_>>>()?;
        let order = self
            .spec
            .order_spec
            .iter()
            .map(|sort| {
                let method = match (sort.ascending, sort.null_ordering) {
                    (true, NullOrdering::First) => "asc_nulls_first",
                    (true, NullOrdering::Last) => "asc_nulls_last",
                    (false, NullOrdering::First) => "desc_nulls_first",
                    (false, NullOrdering::Last) => "desc_nulls_last",
                };
                Ok((column(&sort.child)?, method))
            })
            .collect::<PyResult<Vec<_>>>()?;
        // The frame as its ROWS flag and its start and end, as the ints of `Window`.
        let bound = |bound: &FrameBound| match bound {
            FrameBound::UnboundedPreceding => UNBOUNDED_PRECEDING,
            FrameBound::Preceding(n) => -n,
            FrameBound::CurrentRow => 0,
            FrameBound::Following(n) => *n,
            FrameBound::UnboundedFollowing => UNBOUNDED_FOLLOWING,
        };
        let frame = self
            .spec
            .frame_spec
            .as_ref()
            .map(|(frame_type, start, end)| {
                (*frame_type == FrameType::Row, bound(start), bound(end))
            });
        let window_of = py
            .import("pyspark.sql.connect.window")?
            .getattr("_window_reduce")?
            .getattr("window_of")?;
        let args = (partition, order, frame).into_pyobject(py)?;
        Ok(PyTuple::new(py, [window_of, args.into_any()])?.into_any())
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
        let (start, end) = frame_bounds(py, start, end, true)?;
        Ok(PyWindowSpec::new(
            WindowSpec::new().rows_between(start, end),
        ))
    }

    /// Create a WindowSpec with a RANGE frame between the given bounds.
    #[staticmethod]
    fn rangeBetween(
        py: Python<'_>,
        start: &Bound<'_, PyAny>,
        end: &Bound<'_, PyAny>,
    ) -> PyResult<PyWindowSpec> {
        let (start, end) = frame_bounds(py, start, end, false)?;
        Ok(PyWindowSpec::new(
            WindowSpec::new().range_between(start, end),
        ))
    }

    fn __repr__(&self) -> String {
        "Window".to_string()
    }
}
