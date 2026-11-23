//! PyO3 wrappers for window functions and specifications.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use spark_connect::column::Column;
use spark_connect::expression::{ExpressionNode, SortOrder};
use spark_connect::window::{FrameBound, WindowSpec};
use spark_connect_core::error::SparkError;

use crate::column::PyColumn;

/// A column of `_to_cols`: a Column, or the core Column of `F.col` of a column name, which is made
/// at once, of the origin of `col`, as no object of it is read but its expression.
enum WindowColumn<'py> {
    Object(Bound<'py, PyAny>),
    Name(Column),
}

/// The columns of `Window.partitionBy` and `orderBy`, and of `TableArg.partitionBy` and
/// `orderBy`, as PySpark Connect's `_to_cols`: a single list is the columns, each a Column or a
/// column name (not a literal), whose Column is `F.col` of it, of the origin of `col` at the call
/// site of `origin`, else `NOT_EXPECTED_TYPE` of the argument `col`, as its `F._to_col`.
fn window_column_items<'py>(
    cols: Vec<Bound<'py, PyAny>>,
    origin: &crate::origin::CallSiteOrigin,
) -> PyResult<Vec<WindowColumn<'py>>> {
    let cols = match cols.as_slice() {
        [only] if only.is_instance_of::<pyo3::types::PyList>() => {
            only.try_iter()?.collect::<PyResult<Vec<_>>>()?
        }
        _ => cols,
    };
    cols.into_iter()
        .map(|c| {
            if c.is_instance_of::<PyColumn>() {
                return Ok(WindowColumn::Object(c));
            }
            if c.is_instance_of::<pyo3::types::PyString>() {
                // As PySpark's `F._to_col`, by `col`, of its origin.
                let _origin = origin.with("col")?;
                return Ok(WindowColumn::Name(crate::plan_args::col(&c)?));
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

/// The core expressions of the columns of [`window_column_items`], of the `_expr` of a Column.
pub(crate) fn window_columns(cols: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Column>> {
    window_column_items(cols, &crate::origin::CallSiteOrigin::new())?
        .into_iter()
        .map(|column| match column {
            WindowColumn::Object(column) => crate::column::column_of(column.cast::<PyColumn>()?),
            WindowColumn::Name(column) => Ok(column),
        })
        .collect()
}

/// The `_expr` of each Column of `cols` (see [`window_column_items`]), as PySpark's
/// `[c._expr for c in _to_cols(cols)]`: of a column name, the `_ColumnExpression` of its `F.col`.
fn expressions_of<'py>(
    py: Python<'py>,
    cols: Vec<Bound<'py, PyAny>>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    // The `_expr` of the Column of `F._to_col` of a column name is read of the Column, of a patch
    // of the attribute lookup of the class `Column`.
    let unpatched = crate::column::column_class_is_original(py, false);
    window_column_items(cols, &crate::origin::CallSiteOrigin::new())?
        .into_iter()
        .map(|column| match column {
            WindowColumn::Object(column) => crate::column::expression_attr(&column),
            WindowColumn::Name(column) if unpatched => {
                Ok(crate::column::expression_object(py, column)?.into_any())
            }
            WindowColumn::Name(column) => {
                crate::column::expression_attr(crate::column::column_object(py, column)?.as_any())
            }
        })
        .collect()
}

/// The `_expr` of the sort order of each Column of `cols`, as PySpark's
/// `[F._sort_col(c)._expr for c in _to_cols(cols)]`: a Column that is a sort order as it is, and
/// another of its `asc()`, ascending with nulls first. The `asc()` of a column name, and of a
/// Column of the class, which no subclass overrides, is made at once, as the class's `asc` makes it
/// (`asc_nulls_first`, of its origin); that of another Column is the `asc()` that it has.
pub(crate) fn sort_expressions_of<'py>(
    py: Python<'py>,
    cols: Vec<Bound<'py, PyAny>>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let origin = crate::origin::CallSiteOrigin::new();
    // Of the class `Column` as the module made it, of no patch of its `asc` nor of its attribute
    // lookup, which the `asc()` of PySpark's `_sort_col` would call.
    let unpatched = crate::column::column_class_is_original(py, true);
    let ascending = |column: Column| -> PyResult<Bound<'py, PyAny>> {
        let _origin = origin.with("asc_nulls_first")?;
        Ok(crate::column::expression_object(py, column.asc_nulls_first())?.into_any())
    };
    let ascending_of = |column: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        let sorted = column.call_method0(pyo3::intern!(py, "asc"))?;
        crate::column::expression_attr(&sorted)
    };
    window_column_items(cols, &origin)?
        .into_iter()
        .map(|column| match column {
            WindowColumn::Object(column) => {
                let expression = crate::column::expression_attr(&column)?;
                if is_sort_order(&expression) {
                    return Ok(expression);
                }
                if unpatched && column.get_type().is(py.get_type::<PyColumn>()) {
                    return ascending(crate::column::expression_value(&expression)?);
                }
                ascending_of(&column)
            }
            WindowColumn::Name(column) if unpatched => ascending(column),
            // Of the Column of `F._to_col`, whose `_expr` `_sort_col` reads, and then its `asc()`.
            WindowColumn::Name(column) => {
                let column = crate::column::column_object(py, column)?;
                crate::column::expression_attr(column.as_any())?;
                ascending_of(column.as_any())
            }
        })
        .collect()
}

/// Whether `expression` is the expression of a sort order of this client, as PySpark's
/// `isinstance(expression, SortOrder)`, of its class of `pyspark.sql.connect.expressions`, which
/// the drop-in's module does not have, as the sort orders are the core's.
fn is_sort_order(expression: &Bound<'_, PyAny>) -> bool {
    expression
        .cast::<crate::column::PyColumnExpression>()
        .is_ok_and(|expression| {
            matches!(
                expression.borrow().column().expression().node(),
                ExpressionNode::SortOrder(_)
            )
        })
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

// Python wrapper for a WindowSpec. As PySpark's plain class, it keeps its state in its
// `__dict__`, its `_partitionSpec` and `_orderSpec`, lists of expressions, and its `_frame`, a
// `WindowFrame` or None, which its methods read and give the new window specifications that they
// make, and it is subclassed and constructed of PySpark's arguments (see `crate::state`).
#[doc = connect_doc!(WindowSpec)]
#[pyclass(
    name = "WindowSpec",
    module = "pyspark.sql.connect.window",
    subclass,
    weakref,
    dict
)]
pub struct PyWindowSpec {}

/// A new window specification of the class, of `partition`, `order` and `frame`, as PySpark's
/// methods make one of `WindowSpec(partitionSpec=..., orderSpec=..., frame=...)`, whose
/// `__init__` checks them.
fn new_window_spec<'py>(
    py: Python<'py>,
    partition: &Bound<'py, PyAny>,
    order: &Bound<'py, PyAny>,
    frame: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyWindowSpec>> {
    let spec = Bound::new(py, PyWindowSpec {})?;
    PyWindowSpec::__init__(&spec, partition, order, frame)?;
    Ok(spec)
}

/// The `WindowFrame` of `pyspark.sql.connect.window` of `rows` (a ROWS frame, else a RANGE one)
/// and of `start` and `end`, as PySpark's `rowsBetween` and `rangeBetween` make it, whose
/// `__init__` asserts that they are ints.
fn window_frame<'py>(
    py: Python<'py>,
    rows: bool,
    start: &Bound<'py, PyAny>,
    end: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    static WINDOW_FRAME: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
        pyo3::sync::PyOnceLock::new();
    let class = WINDOW_FRAME.import(py, "pyspark.sql.connect.window", "WindowFrame")?;
    // Of the parameters in their order, `isRowFrame`, `start` and `end`, which PySpark passes by
    // name.
    class.call1((rows, start, end))
}

/// The bound `value` of `rowsBetween` or `rangeBetween`, the start of the frame if `start`, else
/// its end, as PySpark compares it with the threshold of its side, with Python's `<=` and `>=`: a
/// start of `Window._PRECEDING_THRESHOLD` (`-sys.maxsize`) or less is `Window.unboundedPreceding`
/// and an end of `Window._FOLLOWING_THRESHOLD` (`sys.maxsize`) or more `Window.unboundedFollowing`,
/// of any type, so also of a float such as `float("inf")`, and a type that does not compare with
/// an int raises the `TypeError` of the comparison. A `FrameBound` is its int.
fn compared_bound<'py>(value: &Bound<'py, PyAny>, start: bool) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    let value = match value.cast::<PyFrameBound>() {
        Ok(bound) => frame_bound_value(&bound.borrow().bound)
            .into_pyobject(py)?
            .into_any(),
        Err(_) => value.clone(),
    };
    // An int of a long, compared at once, as Python compares ints (of i128, as a threshold may be
    // the extreme of a long).
    if value.is_exact_instance_of::<pyo3::types::PyInt>() {
        if let Ok(n) = value.extract::<i64>() {
            if start && i128::from(n) <= i128::from(PRECEDING_THRESHOLD) {
                return Ok(UNBOUNDED_PRECEDING.into_pyobject(py)?.into_any());
            }
            if !start && i128::from(n) >= i128::from(FOLLOWING_THRESHOLD) {
                return Ok(UNBOUNDED_FOLLOWING.into_pyobject(py)?.into_any());
            }
            return Ok(value);
        }
    }
    if start && value.le(PRECEDING_THRESHOLD)? {
        return Ok(UNBOUNDED_PRECEDING.into_pyobject(py)?.into_any());
    }
    if !start && value.ge(FOLLOWING_THRESHOLD)? {
        return Ok(UNBOUNDED_FOLLOWING.into_pyobject(py)?.into_any());
    }
    Ok(value)
}

#[pymethods]
impl PyWindowSpec {
    /// A window specification of no state yet, as PySpark Connect's
    /// `WindowSpec.__new__(cls, partitionSpec, orderSpec, frame)`, which takes the arguments of
    /// its `__init__` and is `object.__new__(cls)`: until `__init__` gives it its state, its
    /// methods raise the `AttributeError` of the first attribute that they read, as PySpark's.
    #[new]
    #[pyo3(
        signature = (partitionSpec, orderSpec, frame),
        text_signature = "(partitionSpec, orderSpec, frame)"
    )]
    #[allow(non_snake_case)]
    fn py_new(
        partitionSpec: &Bound<'_, PyAny>,
        orderSpec: &Bound<'_, PyAny>,
        frame: &Bound<'_, PyAny>,
    ) -> Self {
        let _ = (partitionSpec, orderSpec, frame);
        PyWindowSpec {}
    }

    /// `WindowSpec(partitionSpec, orderSpec, frame)`, as PySpark Connect's `__init__`, which
    /// asserts that the partitioning is a list of expressions, the ordering a list of sort orders
    /// and the frame None or a `WindowFrame`, of a bare `AssertionError`, and keeps them as its
    /// `_partitionSpec`, `_orderSpec` and `_frame`, as they are.
    #[allow(non_snake_case)]
    fn __init__(
        slf: &Bound<'_, Self>,
        partitionSpec: &Bound<'_, PyAny>,
        orderSpec: &Bound<'_, PyAny>,
        frame: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let py = slf.py();
        let assertion = || pyo3::exceptions::PyAssertionError::new_err(());
        let Ok(partition) = partitionSpec.cast::<PyList>() else {
            return Err(assertion());
        };
        for item in partition.iter() {
            if !is_expression(&item)? {
                return Err(assertion());
            }
        }
        let Ok(order) = orderSpec.cast::<PyList>() else {
            return Err(assertion());
        };
        for item in order.iter() {
            if !is_sort_order(&item) {
                return Err(assertion());
            }
        }
        if !frame.is_none() && !is_window_frame(frame)? {
            return Err(assertion());
        }
        crate::state::assign(
            slf.as_any(),
            &[
                (pyo3::intern!(py, "_partitionSpec"), partitionSpec),
                (pyo3::intern!(py, "_orderSpec"), orderSpec),
                (pyo3::intern!(py, "_frame"), frame),
            ],
        )
    }

    /// The arguments of `__new__` of the specification, as PySpark Connect's: its
    /// `_partitionSpec`, `_orderSpec` and `_frame`.
    fn __getnewargs__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        PyTuple::new(
            py,
            [
                crate::state::attr(slf.as_any(), pyo3::intern!(py, "_partitionSpec"))?,
                crate::state::attr(slf.as_any(), pyo3::intern!(py, "_orderSpec"))?,
                crate::state::attr(slf.as_any(), pyo3::intern!(py, "_frame"))?,
            ],
        )
    }

    /// The state of the specification, its `__dict__`, as `object.__getstate__` gives that of
    /// PySpark's plain object, which a copy and a pickle keep (see `state::getstate`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::getstate(slf.as_any())
    }

    /// The reduce of the specification, as PySpark's plain object's from the protocol 2:
    /// `copyreg.__newobj__` of its class and of the arguments of its `__getnewargs__`, and its
    /// `__dict__`, so a copy shares the lists of its partitioning and its ordering and its frame,
    /// and a deep copy and a pickle copy them. Of every protocol, as `__new__` takes the
    /// arguments, which the default reduce below 2 does not give, as copyreg's of a class of a
    /// native base fails.
    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, protocol: i32) -> PyResult<Bound<'py, PyAny>> {
        let _ = protocol;
        let py = slf.py();
        let newobj = py.import("copyreg")?.getattr("__newobj__")?;
        let args = slf.call_method0(pyo3::intern!(py, "__getnewargs__"))?;
        let mut items = vec![slf.get_type().into_any()];
        items.extend(args.try_iter()?.collect::<PyResult<Vec<_>>>()?);
        let state = crate::state::getstate(slf.as_any())?;
        Ok(PyTuple::new(py, [newobj, PyTuple::new(py, items)?.into_any(), state])?.into_any())
    }

    // Partition the window by the given columns, as PySpark's
    // `WindowSpec(partitionSpec=[c._expr for c in _to_cols(cols)], orderSpec=self._orderSpec,
    // frame=self._frame)`, of the ordering and the frame of this one, as they are.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(WindowSpec.partitionBy)]
    fn partitionBy<'py>(
        slf: &Bound<'py, Self>,
        cols: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyWindowSpec>> {
        let py = slf.py();
        let partition = PyList::new(py, expressions_of(py, cols)?)?;
        let order = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_orderSpec"))?;
        let frame = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_frame"))?;
        new_window_spec(py, partition.as_any(), &order, &frame)
    }

    // Order the window by the given columns, as PySpark's
    // `WindowSpec(partitionSpec=self._partitionSpec, orderSpec=[F._sort_col(c)._expr ...],
    // frame=self._frame)`.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(WindowSpec.orderBy)]
    fn orderBy<'py>(
        slf: &Bound<'py, Self>,
        cols: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyWindowSpec>> {
        let py = slf.py();
        let partition = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_partitionSpec"))?;
        let order = PyList::new(py, sort_expressions_of(py, cols)?)?;
        let frame = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_frame"))?;
        new_window_spec(py, &partition, order.as_any(), &frame)
    }

    // Define a ROWS frame between start and end bounds, as PySpark's: of the ints of the bounds,
    // compared with the thresholds of their sides, in a `WindowFrame` of a ROWS frame, of the
    // partitioning and the ordering of this one.
    #[doc = connect_doc!(WindowSpec.rowsBetween)]
    fn rowsBetween<'py>(
        slf: &Bound<'py, Self>,
        start: &Bound<'py, PyAny>,
        end: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyWindowSpec>> {
        Self::between(slf, start, end, true)
    }

    // Define a RANGE frame between start and end bounds, as `rowsBetween` does a ROWS frame.
    #[doc = connect_doc!(WindowSpec.rangeBetween)]
    fn rangeBetween<'py>(
        slf: &Bound<'py, Self>,
        start: &Bound<'py, PyAny>,
        end: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyWindowSpec>> {
        Self::between(slf, start, end, false)
    }

    /// As `WindowSpec.__repr__` of PySpark Connect, of the `str` of each expression of its
    /// partitioning and its ordering, and of its frame.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let joined = |items: &Bound<'_, PyAny>| -> PyResult<Option<String>> {
            if items.len()? == 0 {
                return Ok(None);
            }
            let texts = items
                .try_iter()?
                .map(|item| Ok(item?.str()?.to_cow()?.into_owned()))
                .collect::<PyResult<Vec<String>>>()?;
            Ok(Some(texts.join(", ")))
        };
        let mut parts = Vec::new();
        let partition = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_partitionSpec"))?;
        if let Some(text) = joined(&partition)? {
            parts.push(format!("PartitionBy({text})"));
        }
        let order = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_orderSpec"))?;
        if let Some(text) = joined(&order)? {
            parts.push(format!("OrderBy({text})"));
        }
        let frame = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_frame"))?;
        if !frame.is_none() {
            parts.push(frame.str()?.to_cow()?.into_owned());
        }
        Ok(format!("WindowSpec({})", parts.join(", ")))
    }
}

impl PyWindowSpec {
    /// `rowsBetween(start, end)` if `rows`, else `rangeBetween(start, end)`, as PySpark's: the
    /// start and then the end compared with the thresholds of their sides, and the specification
    /// of the partitioning and the ordering of this one and of the `WindowFrame` of them, which
    /// asserts that each is an int.
    fn between<'py>(
        slf: &Bound<'py, Self>,
        start: &Bound<'py, PyAny>,
        end: &Bound<'py, PyAny>,
        rows: bool,
    ) -> PyResult<Bound<'py, PyWindowSpec>> {
        let py = slf.py();
        let start = compared_bound(start, true)?;
        let end = compared_bound(end, false)?;
        let partition = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_partitionSpec"))?;
        let order = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_orderSpec"))?;
        let frame = window_frame(py, rows, &start, &end)?;
        new_window_spec(py, &partition, &order, &frame)
    }
}

/// Whether `value` is an expression, of this client (a `_ColumnExpression`) or of PySpark's
/// classes, as PySpark's `WindowSpec.__init__` asserts of its partitioning.
fn is_expression(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if value.is_instance_of::<crate::column::PyColumnExpression>() {
        return Ok(true);
    }
    static EXPRESSION: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
        pyo3::sync::PyOnceLock::new();
    let class = EXPRESSION.import(value.py(), "pyspark.sql.connect.expressions", "Expression")?;
    value.is_instance(class)
}

/// Whether `value` is a `WindowFrame` of `pyspark.sql.connect.window`, as PySpark's
/// `WindowSpec.__init__` asserts of its frame.
fn is_window_frame(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    static WINDOW_FRAME: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
        pyo3::sync::PyOnceLock::new();
    let class = WINDOW_FRAME.import(value.py(), "pyspark.sql.connect.window", "WindowFrame")?;
    value.is_instance(class)
}

/// The window of the core of `window`, a window specification, of its `_partitionSpec`, its
/// `_orderSpec` and its `_frame` as they are at the call, as PySpark's `WindowExpression.to_plan`
/// reads them when a request builds its plan, and the error that such a build raises first of a
/// bound of the frame that the window of the core does not hold, in place of which it holds the
/// current row (see [`window_bound`]). The `over` of a Column makes its window expression of them
/// (residue (a) of the design: the expression is a value of the call).
pub(crate) fn window_spec_of(
    window: &Bound<'_, PyAny>,
) -> PyResult<(WindowSpec, Option<SparkError>, bool)> {
    let py = window.py();
    // Of its attribute lookup, as PySpark's reads of `self._windowSpec._partitionSpec`, which no
    // `__getattr__` of the client's class reads again, so of no check of a recursion.
    let partition = window.getattr(pyo3::intern!(py, "_partitionSpec"))?;
    list_of_window(&partition)?;
    let partition_spec = entries_of(&partition, |item| {
        match item.cast::<crate::column::PyColumnExpression>() {
            Ok(expression) => Ok(expression.borrow().column().expression().clone()),
            Err(_) => Ok(expression_value(item)?.expression().clone()),
        }
    })?;
    let order = window.getattr(pyo3::intern!(py, "_orderSpec"))?;
    list_of_window(&order)?;
    let order_spec = entries_of(&order, sort_order_of)?;
    let mut spec = WindowSpec {
        partition_spec,
        order_spec,
        frame_spec: None,
    };
    let frame = window.getattr(pyo3::intern!(py, "_frame"))?;
    if frame.is_none() {
        return Ok((spec, None, true));
    }
    let rows = frame
        .getattr(pyo3::intern!(py, "_isRowFrame"))?
        .is_truthy()?;
    let (start, end) = (
        frame.getattr(pyo3::intern!(py, "_start"))?,
        frame.getattr(pyo3::intern!(py, "_end"))?,
    );
    // The core renders the offset of a bound, which is PySpark's rendering of an int of a long,
    // and not of a bool or of an int of another class.
    let rendered = [&start, &end].iter().all(|bound| {
        bound.is_exact_instance_of::<pyo3::types::PyInt>() && bound.extract::<i64>().is_ok()
    });
    let start = window_bound(&start, true, rows)?;
    let end = window_bound(&end, false, rows)?;
    let mut error = None;
    let mut held = |bound: Result<FrameBound, SparkError>| match bound {
        Ok(bound) => bound,
        Err(failure) => {
            error.get_or_insert(failure);
            FrameBound::CurrentRow
        }
    };
    let (start, end) = (held(start), held(end));
    spec = match rows {
        true => spec.rows_between(start, end),
        false => spec.range_between(start, end),
    };
    Ok((spec, error, rendered))
}

/// The core expression of `item`, an expression of a window specification: of this client, the
/// `_expr` of a Column, or of PySpark's classes, of its `to_plan` (see `column::expression_value`).
fn expression_value(item: &Bound<'_, PyAny>) -> PyResult<Column> {
    crate::column::expression_value(item)
}

/// Check that `entries`, the partitioning or the ordering of a window, is a list, as PySpark's
/// `WindowExpression.children` concatenates them to the list of its window function: of
/// another object, the `TypeError` of that concatenation, which PySpark raises when a plan of the
/// window expression is made (`_collect_references`), and the drop-in when the window expression
/// is made of the window.
fn list_of_window(entries: &Bound<'_, PyAny>) -> PyResult<()> {
    if entries.is_instance_of::<PyList>() {
        return Ok(());
    }
    let type_name = entries.get_type().name()?.to_string();
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "can only concatenate list (not \"{type_name}\") to list"
    )))
}

/// The values of `of` of the entries of `entries`, a list of a window specification, in their
/// order, as PySpark iterates it: of a list, its items at once, and of another iterable, of its
/// iterator.
fn entries_of<T>(
    entries: &Bound<'_, PyAny>,
    of: impl Fn(&Bound<'_, PyAny>) -> PyResult<T>,
) -> PyResult<Vec<T>> {
    if let Ok(list) = entries.cast_exact::<PyList>() {
        return list.iter().map(|item| of(&item)).collect();
    }
    entries.try_iter()?.map(|item| of(&item?)).collect()
}

/// The sort order of the core of `item`, an entry of the ordering of a window specification, as
/// PySpark's `WindowExpression.to_plan` sends `s.to_plan(session).sort_order` of each: of a sort
/// order of this client, its own; of another object, which the ordering holds once it is
/// assigned, such as an `Expression` of the user's class, the sort order of its `to_plan`, of its
/// child as it is sent, rendered as the `str` of its `_child`, as PySpark's `SortOrder.__repr__`
/// renders it. Another expression is sorted ascending with nulls first, as the window of the core
/// holds sort orders.
fn sort_order_of(item: &Bound<'_, PyAny>) -> PyResult<SortOrder> {
    use prost::Message;
    use spark_connect_proto::expression::sort_order::{NullOrdering, SortDirection};
    if let Ok(expression) = item.cast::<crate::column::PyColumnExpression>() {
        let expression = expression.borrow();
        let expression = expression.column().expression();
        return Ok(match expression.node() {
            ExpressionNode::SortOrder(sort) => (**sort).clone(),
            _ => SortOrder::asc_nulls_first(expression.clone()),
        });
    }
    let planned = crate::column::planned_expression(item)?;
    // Decoded, and its child encoded again, then dropped, on a stack that holds its levels.
    let sort = crate::on_grown_stack(|| {
        spark_connect_proto::Expression::decode(planned.as_bytes()).map(|decoded| {
            match decoded.expr_type {
                Some(spark_connect_proto::expression::ExprType::SortOrder(sort)) => Some((
                    sort.child
                        .map(|child| child.encode_to_vec())
                        .unwrap_or_default(),
                    sort.direction != SortDirection::Descending as i32,
                    sort.null_ordering == NullOrdering::SortNullsLast as i32,
                )),
                _ => None,
            }
        })
    })
    .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    // A rendering that fails is the error of the rendering of the window expression, as of a
    // Column (see `column::converted_expression`).
    let rendered_column =
        |bytes: &[u8], text: PyResult<Bound<'_, pyo3::types::PyString>>| match text
            .and_then(|text| text.extract::<String>())
        {
            Ok(text) => crate::column::sent_column(bytes, text),
            Err(error) => crate::column::unrenderable_column(
                crate::column::sent_column(bytes, String::new())?,
                error,
            ),
        };
    let Some((child, ascending, nulls_last)) = sort else {
        let expression = rendered_column(planned.as_bytes(), item.str())?;
        return Ok(SortOrder::asc_nulls_first(expression.expression().clone()));
    };
    let py = item.py();
    let rendered = match item.getattr(pyo3::intern!(py, "_child")) {
        Ok(child) => child.str(),
        Err(_) => item.str(),
    };
    let child = rendered_column(&child, rendered)?;
    Ok(SortOrder {
        child: child.expression().clone(),
        ascending,
        null_ordering: if nulls_last {
            spark_connect::expression::NullOrdering::Last
        } else {
            spark_connect::expression::NullOrdering::First
        },
    })
}

/// The bound `value` of a frame, its start if `lower`, else its end, of a ROWS frame if `rows`,
/// else of a RANGE one, as PySpark's `WindowExpression.to_plan` converts it as it builds the plan
/// of the frame: 0 (False too) is the current row, the long of `Window.unboundedPreceding` as the
/// start and of `Window.unboundedFollowing` as the end is unbounded, and else a ROWS frame sets an
/// int of an int32 in the integer literal, and raises `VALUE_NOT_BETWEEN` of an int beyond, and a
/// RANGE frame any other int in the long literal, as protobuf's default (upb) backend takes a value
/// of an int field: a bool is its `TypeError`, an int beyond a long its `ValueError` (see
/// `stat_arguments::proto_int`). The error is the one of the request that builds the plan.
fn window_bound(
    value: &Bound<'_, PyAny>,
    lower: bool,
    rows: bool,
) -> PyResult<Result<FrameBound, SparkError>> {
    let py = value.py();
    // An int of a long, as `rowsBetween` and `rangeBetween` keep the bounds of ints, compared at
    // once, as Python compares ints, and set as protobuf takes an int in the range of its field.
    if value.is_exact_instance_of::<pyo3::types::PyInt>() {
        if let Ok(n) = value.extract::<i64>() {
            return Ok(int_bound(n, lower, rows));
        }
    }
    if value.eq(0)? {
        return Ok(Ok(FrameBound::CurrentRow));
    }
    if lower && value.eq(UNBOUNDED_PRECEDING)? {
        return Ok(Ok(FrameBound::UnboundedPreceding));
    }
    if !lower && value.eq(UNBOUNDED_FOLLOWING)? {
        return Ok(Ok(FrameBound::UnboundedFollowing));
    }
    let offset = if rows {
        if !(value.ge(i32::MIN)? && value.le(i32::MAX)?) {
            return Ok(Err(not_between(lower)));
        }
        crate::stat_arguments::proto_int::<i32>(value, ROW_BOUNDARY_FIELD).map(i64::from)
    } else {
        crate::stat_arguments::proto_int::<i64>(value, RANGE_BOUNDARY_FIELD)
    };
    Ok(match offset {
        Ok(offset) => Ok(frame_bound_of(offset)),
        Err(error) => Err(crate::dataframe::unbuildable_error(py, error)?),
    })
}

/// The bound of `n`, an int of a long, as [`window_bound`] takes it.
fn int_bound(n: i64, lower: bool, rows: bool) -> Result<FrameBound, SparkError> {
    if n == 0 {
        return Ok(FrameBound::CurrentRow);
    }
    if lower && n == UNBOUNDED_PRECEDING {
        return Ok(FrameBound::UnboundedPreceding);
    }
    if !lower && n == UNBOUNDED_FOLLOWING {
        return Ok(FrameBound::UnboundedFollowing);
    }
    if rows && i32::try_from(n).is_err() {
        return Err(not_between(lower));
    }
    Ok(frame_bound_of(n))
}

/// PySpark's `VALUE_NOT_BETWEEN` of the start of a row frame if `lower`, else of its end, of a
/// bound beyond an int.
fn not_between(lower: bool) -> SparkError {
    let name = if lower { "start" } else { "end" };
    SparkError::value(
        "VALUE_NOT_BETWEEN",
        &[
            ("arg_name", name),
            ("min", &i32::MIN.to_string()),
            ("max", &i32::MAX.to_string()),
        ],
    )
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

/// The window specification of no partitioning, ordering and frame, of which the static methods
/// of `pyspark.sql.connect.window.Window` start, as PySpark Connect's `Window._spec`. The classic
/// `pyspark.sql.window.Window` is PySpark's, whose static methods dispatch to those.
#[pyfunction]
#[pyo3(name = "_empty_window_spec")]
pub(crate) fn empty_window_spec(py: Python<'_>) -> PyResult<Bound<'_, PyWindowSpec>> {
    let none = py.None().into_bound(py);
    new_window_spec(
        py,
        PyList::empty(py).as_any(),
        PyList::empty(py).as_any(),
        &none,
    )
}
