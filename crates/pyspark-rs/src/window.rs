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
                return crate::plan_args::col(c);
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

/// PySpark's `Window._PRECEDING_THRESHOLD`, `max(-sys.maxsize, JVM_LONG_MIN)`, at or below which
/// the start of a frame is unbounded preceding, and `Window._FOLLOWING_THRESHOLD`,
/// `min(sys.maxsize, JVM_LONG_MAX)`, at or above which its end is unbounded following.
const PRECEDING_THRESHOLD: i64 = {
    let preceding = -(isize::MAX as i64);
    if preceding > UNBOUNDED_PRECEDING {
        preceding
    } else {
        UNBOUNDED_PRECEDING
    }
};
const FOLLOWING_THRESHOLD: i64 = {
    let following = isize::MAX as i64;
    if following < UNBOUNDED_FOLLOWING {
        following
    } else {
        UNBOUNDED_FOLLOWING
    }
};

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
    if start && v.le(PRECEDING_THRESHOLD)? {
        return Ok(FrameArgument::Bound(FrameBound::UnboundedPreceding));
    }
    if !start && v.ge(FOLLOWING_THRESHOLD)? {
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

/// A frame bound as the window of the core holds it, or, of a bound that it does not hold, the
/// error that the plan of PySpark's `WindowExpression` raises of it.
enum ConvertedBound {
    Held(FrameBound),
    Unheld(PyErr),
}

/// The bound `name` of a frame argument, in a row frame if `rows`, else in a range frame, as the
/// window of the core holds it: 0 (False too) is the current row, and another int of a long the
/// offset of its sign, which the core converts when a request builds the plan, as the plan of
/// PySpark's `WindowExpression` converts it then (an int beyond an int32 of a row frame fails
/// with `VALUE_NOT_BETWEEN`). The core does not hold True, which PySpark sets in the int field of
/// the literal, which protobuf's default (upb) backend refuses of a bool, nor an int beyond a
/// long, of `VALUE_NOT_BETWEEN` in a row frame and of the `ValueError` of upb's long field in a
/// range frame: these give the error.
fn converted_frame_bound(
    py: Python<'_>,
    argument: FrameArgument<'_, '_>,
    name: &str,
    rows: bool,
) -> PyResult<ConvertedBound> {
    let value = match argument {
        FrameArgument::Bound(bound) => return Ok(ConvertedBound::Held(bound)),
        FrameArgument::Value(value) => value,
    };
    if value.is_instance_of::<pyo3::types::PyBool>() {
        let field = if rows {
            ROW_BOUNDARY_FIELD
        } else {
            RANGE_BOUNDARY_FIELD
        };
        return Ok(if value.is_truthy()? {
            ConvertedBound::Unheld(crate::stat_arguments::bool_not_proto_int(field))
        } else {
            ConvertedBound::Held(FrameBound::CurrentRow)
        });
    }
    Ok(match value.extract::<i64>() {
        Ok(n) => ConvertedBound::Held(frame_bound_of(n)),
        Err(_) if rows => ConvertedBound::Unheld(boundary_not_between(py, name)),
        Err(_) => ConvertedBound::Unheld(crate::session::value_out_of_range(value)),
    })
}

/// The error that a request raises of the held bound `bound` of a row frame, its start if
/// `lower`, else its end, as the core converts it when it builds the plan
/// (`to_proto_frame_boundary`): an offset beyond an int32 that is neither the current row nor
/// the unbounded bound of its side is `VALUE_NOT_BETWEEN`. A range frame holds no failing bound.
fn held_bound_error(py: Python<'_>, bound: &FrameBound, rows: bool, lower: bool) -> Option<PyErr> {
    let offset = match bound {
        FrameBound::UnboundedPreceding => i128::from(i64::MIN),
        FrameBound::Preceding(n) => -i128::from(*n),
        FrameBound::CurrentRow => 0,
        FrameBound::Following(n) => i128::from(*n),
        FrameBound::UnboundedFollowing => i128::from(i64::MAX),
    };
    let unbounded = i128::from(if lower { i64::MIN } else { i64::MAX });
    let fails = rows && offset != 0 && offset != unbounded && i32::try_from(offset).is_err();
    fails.then(|| boundary_not_between(py, if lower { "start" } else { "end" }))
}

/// The bounds of `rowsBetween(start, end)` if `rows`, else of `rangeBetween(start, end)`, a
/// plain int (negative is preceding, positive following) or a `FrameBound` each, as PySpark's:
/// it compares the start and then the end with the thresholds of their sides, and WindowFrame
/// then asserts that the start and then the end that is not past its threshold is an int (a
/// bool included), a bare `AssertionError` of a NumPy int, a float or a Decimal. The plan converts
/// the start and then the end when a request builds it, as the window of the core does of the
/// bounds that it holds (see [`converted_frame_bound`]). Of a bound that it does not hold, the
/// error that PySpark's build raises first, of the start and then of the end, is raised here.
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
    let start = converted_frame_bound(py, start, "start", rows)?;
    let end = converted_frame_bound(py, end, "end", rows)?;
    match (start, end) {
        (ConvertedBound::Held(start), ConvertedBound::Held(end)) => Ok((start, end)),
        (ConvertedBound::Unheld(error), _) => Err(error),
        (ConvertedBound::Held(start), ConvertedBound::Unheld(error)) => {
            Err(held_bound_error(py, &start, rows, true).unwrap_or(error))
        }
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

// Python wrapper for a WindowSpec.
#[doc = connect_doc!(WindowSpec)]
#[pyclass(
    name = "WindowSpec",
    module = "pyspark.sql.connect.window",
    from_py_object,
    weakref,
    dict
)]
pub struct PyWindowSpec {
    pub(crate) spec: WindowSpec,
    /// The references of the specification to the parts of its partitioning and ordering, which
    /// `__traverse__` reports (see [`crate::gc::Parts`]).
    parts: crate::gc::Parts,
    /// Set once the value is written, as `__traverse__` reads `parts`, which is not valid while
    /// the object is created (see [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

impl PyWindowSpec {
    pub fn new(spec: WindowSpec) -> Self {
        PyWindowSpec {
            parts: Python::attach(|py| crate::gc::Parts::of(py, spec.held_parts())),
            spec,
            gc_ready: crate::gc::GcReady::new(),
        }
    }
}

impl Clone for PyWindowSpec {
    /// A window specification of the same partitioning, ordering and frame, whose value is
    /// written.
    fn clone(&self) -> Self {
        PyWindowSpec::new(self.spec.clone())
    }
}

#[pymethods]
impl PyWindowSpec {
    // Partition the window by the given columns.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(WindowSpec.partitionBy)]
    fn partitionBy(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyWindowSpec> {
        let new_spec = self.spec.clone().partition_by(to_partition_exprs(cols)?);
        Ok(PyWindowSpec::new(new_spec))
    }

    // Order the window by the given columns.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(WindowSpec.orderBy)]
    fn orderBy(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyWindowSpec> {
        let new_spec = self.spec.clone().order_by(to_sort_orders(cols)?);
        Ok(PyWindowSpec::new(new_spec))
    }

    // Define a ROWS frame between start and end bounds (ints or FrameBounds).
    #[doc = connect_doc!(WindowSpec.rowsBetween)]
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

    // Define a RANGE frame between start and end bounds (ints or FrameBounds).
    #[doc = connect_doc!(WindowSpec.rangeBetween)]
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

    /// The expressions of the partitioning, as the `_partitionSpec` of PySpark Connect's
    /// WindowSpec: a list of the `_expr` of each Column, whose `repr` is PySpark's.
    #[getter]
    fn _partitionSpec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        expression_list(py, self.spec.partition_spec.iter().cloned())
    }

    /// The sort orders of the ordering, as the `_orderSpec` of PySpark Connect's WindowSpec: a
    /// list of the `_expr` of each sort order, ascending with nulls first of a Column that is not
    /// one, whose `repr` is PySpark's, such as `a ASC NULLS FIRST`.
    #[getter]
    fn _orderSpec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        expression_list(
            py,
            self.spec
                .order_spec
                .iter()
                .map(|sort| ExpressionNode::SortOrder(Box::new(sort.clone())).into()),
        )
    }

    /// The frame, as the `_frame` of PySpark Connect's WindowSpec: None without one, else the
    /// `WindowFrame` of `pyspark.sql.connect.window` of its ROWS flag and its start and end, as
    /// the ints of `Window`, such as `WindowFrame(ROW_FRAME, -1, 1)`.
    #[getter]
    fn _frame(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let Some((frame_type, start, end)) = &self.spec.frame_spec else {
            return Ok(py.None());
        };
        Ok(py
            .import("pyspark.sql.connect.window")?
            .getattr("WindowFrame")?
            .call1((
                *frame_type == FrameType::Row,
                frame_bound_value(start),
                frame_bound_value(end),
            ))?
            .unbind())
    }

    /// A WindowSpec pickles as the partition and sort Columns and the frame of it, which the
    /// function `pyspark.sql.connect._window_reduce.window_of` gives back as a WindowSpec built
    /// through `Window`, also on a Python worker of the server, as PySpark's WindowSpec pickles
    /// the Columns it holds. A sort order is given as the Column of its child and the method of
    /// its direction and nulls, as an unpickled Column is not a sort order of its own.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.borrow();
        let column = |expr: &Expression| Py::new(py, PyColumn::new(Column::new(expr.clone())));
        let partition = this
            .spec
            .partition_spec
            .iter()
            .map(column)
            .collect::<PyResult<Vec<_>>>()?;
        let order = this
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
        let frame = this
            .spec
            .frame_spec
            .as_ref()
            .map(|(frame_type, start, end)| {
                (
                    *frame_type == FrameType::Row,
                    frame_bound_value(start),
                    frame_bound_value(end),
                )
            });
        let window_of = py
            .import("pyspark.sql.connect.window")?
            .getattr("_window_reduce")?
            .getattr("window_of")?;
        let args = (partition, order, frame).into_pyobject(py)?;
        let reduce = PyTuple::new(py, [window_of, args.into_any()])?.into_any();
        crate::group::reduce_with_state(&reduce, slf.as_any())
    }

    /// As PySpark's window specification, a Python object that holds its partitioning and
    /// ordering columns, which Python's garbage collector sees: the parts of its columns, of the
    /// functions of their UDFs and of the plans of the Python client that their subqueries and
    /// errors keep, are reported (see [`crate::gc`]), so that a reference cycle through them is
    /// collected, as PySpark collects it.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        self.parts.visit(&visit)
    }

    /// Break a reference cycle that the garbage collector found unreachable: the specification
    /// drops its columns, and so the parts that `__traverse__` reported.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            self.parts.release();
            self.spec = WindowSpec::new();
        }
    }
}

/// The int of the frame bound `bound`, as PySpark's `WindowFrame` keeps it: the sentinel of
/// `Window` of an unbounded one, else its offset, negative when it precedes.
fn frame_bound_value(bound: &FrameBound) -> i64 {
    match bound {
        FrameBound::UnboundedPreceding => UNBOUNDED_PRECEDING,
        FrameBound::Preceding(n) => -n,
        FrameBound::CurrentRow => 0,
        FrameBound::Following(n) => *n,
        FrameBound::UnboundedFollowing => UNBOUNDED_FOLLOWING,
    }
}

/// The list of the `_expr` of a Column of each of `expressions`, as PySpark Connect's WindowSpec
/// keeps the expressions of its partitioning and ordering.
fn expression_list<'py>(
    py: Python<'py>,
    expressions: impl Iterator<Item = Expression>,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let items = expressions
        .map(|expression| Bound::new(py, PyColumn::new(Column::new(expression)))?.getattr("_expr"))
        .collect::<PyResult<Vec<_>>>()?;
    pyo3::types::PyList::new(py, items)
}

/// The window specification of no partitioning, ordering and frame, of which the static methods
/// of `pyspark.sql.connect.window.Window` start, as PySpark Connect's `Window._spec`. The classic
/// `pyspark.sql.window.Window` is PySpark's, whose static methods dispatch to those.
#[pyfunction]
#[pyo3(name = "_empty_window_spec")]
pub(crate) fn empty_window_spec() -> PyWindowSpec {
    PyWindowSpec::new(WindowSpec::new())
}
