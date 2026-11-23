//! PyO3 wrapper for spark_connect::column::Column.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use spark_connect::column::Column;
use spark_connect::expression::{Alias, ExpressionNode, NullOrdering};

use crate::errors::{not_expected_type, ResultExt};
// Re-exported for the modules that raise through `crate::column::raise_pyspark`.
pub(crate) use crate::errors::raise_pyspark;
use crate::functions::{enum_to_value, to_column, to_value_column};

/// The function `name` of `column` and `other`, a Column or a literal value, as PySpark's
/// `_bin_op`.
fn binary_function(name: &str, column: &Column, other: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
    let other = operand_column(other)?;
    Ok(ColumnOut::new(Column::new(
        spark_connect::expression::Expression::function(
            name,
            vec![column.expression().clone(), other.expression().clone()],
        ),
    )))
}

/// The unparsed identifier of `column` if it is a plain column reference, such as `col("a")`
/// or `df.a`, and None for another expression. `SparkSession.sql` formats such a column in a
/// query as its name, as PySpark's formatter does a `ColumnReference`. A function rather than
/// an attribute, as a Column's attributes are its fields in PySpark.
#[pyfunction]
pub(crate) fn column_reference_name(column: &Bound<'_, PyColumn>) -> PyResult<Option<String>> {
    Ok(match column_of(column)?.expression().node() {
        spark_connect::expression::ExpressionNode::ColumnReference(reference) => {
            Some(reference.name.clone())
        }
        _ => None,
    })
}

// Python wrapper for a Spark Column. As PySpark's plain class, it keeps its expression in its
// `__dict__`, as its `_expr`, which its methods read at the call (see `column_of`), and it is
// subclassed and constructed of PySpark's `Column(expr)`.
#[doc = connect_doc!(Column)]
#[pyclass(
    frozen,
    name = "Column",
    module = "pyspark.sql.connect.column",
    subclass,
    weakref,
    dict
)]
pub struct PyColumn {}

impl PyColumn {
    /// The native part of a Column, which holds nothing: its `__dict__` holds its `_expr`, which
    /// holds the parts of its expression for the garbage collector (see `PyColumnExpression`).
    fn empty() -> Self {
        PyColumn {}
    }
}

/// A new Column of `column`, as PySpark's methods give one of `Column(expr)`: an object of the
/// class whose `_expr` is the `_ColumnExpression` of `column`, made with the GIL when the Column
/// is given back (see [`column_object`]).
pub(crate) struct ColumnOut(Column);

impl ColumnOut {
    pub(crate) fn new(column: Column) -> Self {
        ColumnOut(column)
    }
}

impl<'py> IntoPyObject<'py> for ColumnOut {
    type Target = PyColumn;
    type Output = Bound<'py, PyColumn>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, PyColumn>> {
        column_object(py, self.0)
    }
}

/// The Column of `column`, as PySpark's `Column(expr)`: of the `_ColumnExpression` of `column`,
/// which holds the parts of its expression for the garbage collector, as its `_expr`.
pub(crate) fn column_object(py: Python<'_>, column: Column) -> PyResult<Bound<'_, PyColumn>> {
    column_of_expression(expression_object(py, column)?.as_any())
}

/// The `_ColumnExpression` of `column`, the one holder of the parts of its expression. One of no
/// part of Python objects reaches no Python object, now or later: the collector does not track
/// it, as CPython does not track an object that holds no other, so a Column of it is of two
/// tracked objects (the Column and its `__dict__`), as PySpark's (its Column and its expression).
pub(crate) fn expression_object(
    py: Python<'_>,
    column: Column,
) -> PyResult<Bound<'_, PyColumnExpression>> {
    let expression = Bound::new(py, PyColumnExpression::new_with(py, column))?;
    if expression.borrow().parts.holds_no_shared() {
        // SAFETY: the GIL is held, and the object is a valid object of a type of the collector;
        // its `__traverse__` reports nothing, as it holds no shared part.
        unsafe { pyo3::ffi::PyObject_GC_UnTrack(expression.as_ptr().cast()) };
    }
    Ok(expression)
}

/// The `__getattribute__` and the `asc` of the class `Column` as the module made them, before a
/// patch of them, which [`column_class_is_original`] compares with the class's (see
/// [`remember_column_class`]).
struct ColumnClassOriginals {
    getattribute: Py<PyAny>,
    asc: Py<PyAny>,
}

static COLUMN_CLASS_ORIGINALS: pyo3::sync::PyOnceLock<ColumnClassOriginals> =
    pyo3::sync::PyOnceLock::new();

/// Remember the `__getattribute__` and the `asc` of the class `Column`, as the module makes it,
/// before user code could patch them.
pub(crate) fn remember_column_class(py: Python<'_>) -> PyResult<()> {
    let class = py.get_type::<PyColumn>();
    let originals = ColumnClassOriginals {
        getattribute: class
            .getattr(pyo3::intern!(py, "__getattribute__"))?
            .unbind(),
        asc: class.getattr(pyo3::intern!(py, "asc"))?.unbind(),
    };
    let _ = COLUMN_CLASS_ORIGINALS.set(py, originals);
    Ok(())
}

/// Whether the class `Column` has the `__getattribute__` that the module made it of, and, if
/// `asc`, the `asc` too, each of one lookup of the class: a patch of `Column.__getattribute__` or
/// `Column.asc`, of a library, takes the paths that read and call them as PySpark's do, in place
/// of the paths that make the expressions of a window at once. (A patch of `__getattr__` is called
/// of a Column of no `_expr` only, whose read takes the lookup of its class.)
pub(crate) fn column_class_is_original(py: Python<'_>, asc: bool) -> bool {
    let Some(originals) = COLUMN_CLASS_ORIGINALS.get(py) else {
        return false;
    };
    let class = py.get_type::<PyColumn>();
    let unpatched = |name: &Bound<'_, pyo3::types::PyString>, original: &Py<PyAny>| {
        class
            .getattr(name)
            .is_ok_and(|value| value.is(original.bind(py)))
    };
    unpatched(
        pyo3::intern!(py, "__getattribute__"),
        &originals.getattribute,
    ) && (!asc || unpatched(pyo3::intern!(py, "asc"), &originals.asc))
}

/// The Column of `expression`, as PySpark's `Column(expr)` keeps the object that it is given as
/// its `_expr`: of the class, whose methods read it (see [`column_of`]).
pub(crate) fn column_of_expression<'py>(
    expression: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyColumn>> {
    let py = expression.py();
    let column = Bound::new(py, PyColumn::empty())?;
    crate::state::assign(column.as_any(), &[(pyo3::intern!(py, "_expr"), expression)])?;
    Ok(column)
}

/// The core expression of `column`, a Column of this client (of its class or of a subclass), as
/// PySpark's methods read `self._expr`: of the instance dict, or of a property or a `__getattr__`
/// of a subclass, with its errors, so a Column whose `__init__` never ran raises the
/// `RecursionError` of its `__getattr__`, which reads `_expr` again, as PySpark's does. The value
/// is that of [`expression_value`]: of an expression of PySpark's classes, its conversion of the
/// call, as PySpark's Column builds the message of its expression at each request.
pub(crate) fn column_of(column: &Bound<'_, PyColumn>) -> PyResult<Column> {
    expression_value(&expression_attr(column.as_any())?)
}

/// The `_expr` of `column`, a Column, as PySpark's methods read `self._expr`: of the attribute
/// lookup of its class, of a patch of its `__getattribute__` too (see [`crate::state::attr`]).
pub(crate) fn expression_attr<'py>(column: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    crate::state::attr(column, pyo3::intern!(column.py(), "_expr"))
}

/// Whether `expression` is an expression of PySpark's classes, an instance of the
/// `Expression` of `pyspark.sql.connect.expressions`, which PySpark's `Column(expr)` takes.
fn is_pyspark_expression(expression: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = expression.py();
    static EXPRESSION: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> =
        pyo3::sync::PyOnceLock::new();
    let class = EXPRESSION.import(py, "pyspark.sql.connect.expressions", "Expression")?;
    expression.is_instance(class)
}

/// The argument `expr` of `args` and `kwargs`, as CPython binds them to PySpark's
/// `def __init__(self, expr)` of `Column`: of its errors, in its order, an unexpected keyword
/// argument and a value of `expr` by keyword and by position, then too many positional
/// arguments, then none.
fn bound_argument<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::exceptions::PyTypeError;
    let mut expr = args.get_item(0).ok();
    for (key, value) in kwargs.into_iter().flat_map(|kwargs| kwargs.iter()) {
        let name = key.str()?.to_cow()?.into_owned();
        if name != "expr" {
            return Err(PyTypeError::new_err(format!(
                "Column.__init__() got an unexpected keyword argument '{name}'"
            )));
        }
        if expr.is_some() {
            return Err(PyTypeError::new_err(
                "Column.__init__() got multiple values for argument 'expr'",
            ));
        }
        expr = Some(value);
    }
    if args.len() > 1 {
        return Err(PyTypeError::new_err(format!(
            "Column.__init__() takes 2 positional arguments but {} were given",
            args.len() + 1
        )));
    }
    expr.ok_or_else(|| {
        PyTypeError::new_err("Column.__init__() missing 1 required positional argument: 'expr'")
    })
}

/// The rendering of the window expression of `function` over `window`, as PySpark's
/// `WindowExpression.__repr__` renders it, `WindowExpression(<function>, (<window>))`, of the
/// `str` of the window, as an f-string formats it; None of a function or a window that does not
/// render, whose error the default rendering raises.
fn rendered_over(function: &Column, window: &Bound<'_, PyAny>) -> Option<String> {
    let function = function.expression().try_render().ok()?;
    let window = window.str().ok()?.to_cow().ok()?.into_owned();
    Some(format!("WindowExpression({function}, ({window}))"))
}

/// The check of PySpark's `Column(expr)` of `expression`, as its `__init__` makes it: an
/// expression of this client or of PySpark's classes, else its `NOT_EXPECTED_TYPE`.
fn check_expression(expression: &Bound<'_, PyAny>) -> PyResult<()> {
    if expression.is_instance_of::<PyColumnExpression>() || is_pyspark_expression(expression)? {
        Ok(())
    } else {
        Err(not_an_expression(expression))
    }
}

/// The `NOT_EXPECTED_TYPE` of PySpark's `Column(expr)` of `expression`, which is not an
/// expression.
fn not_an_expression(expression: &Bound<'_, PyAny>) -> PyErr {
    let arg_type = expression
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_default();
    raise_pyspark(
        expression.py(),
        "PySparkTypeError",
        "NOT_EXPECTED_TYPE",
        &[
            ("expected_type", "Expression"),
            ("arg_name", "expr"),
            ("arg_type", &arg_type),
        ],
    )
}

/// The core expression of `expression`, an object that is not an expression of this client, such
/// as one of PySpark's classes: its `spark.connect.Expression`, of its `to_plan` (see
/// [`planned_expression`]), as it is sent, rendered as its `str`, as PySpark's expressions render
/// those that they hold. The error of the `str` (of an `Expression` of a class whose `__repr__`
/// gives None, as PySpark's base does) is that of the rendering of the expression, which a
/// `repr` of a Column of it raises, as PySpark's raises it then, and not of its uses.
fn converted_expression(expression: &Bound<'_, PyAny>) -> PyResult<Column> {
    let planned = planned_expression(expression)?;
    let rendered = expression.str().and_then(|text| text.extract::<String>());
    match rendered {
        Ok(rendered) => sent_column(planned.as_bytes(), rendered),
        Err(error) => unrenderable_column(sent_column(planned.as_bytes(), String::new())?, error),
    }
}

/// `column`, whose rendering raises `error`, the error of the rendering of the object that it was
/// made of, anew at each rendering, as PySpark renders the object at each `repr`.
pub(crate) fn unrenderable_column(column: Column, error: PyErr) -> PyResult<Column> {
    use spark_connect::expression::{Expression, UnbuildableExpression};
    Python::attach(|py| {
        let (error, owner) = match crate::dataframe::unbuildable_error(py, error) {
            Ok(error) => (error, None),
            Err(error) => {
                let (error, owner) = crate::errors::kept_exception_error(py, error)?;
                (error, Some(spark_connect::plan::RelationOwner(owner)))
            }
        };
        Ok(Column::new(Expression::new(ExpressionNode::Unbuildable(
            Box::new(UnbuildableExpression {
                expression: column.expression().clone(),
                before: None,
                after: None,
                unrenderable: Some(error),
                owner,
                rendered: None,
            }),
        ))))
    })
}

/// The `spark.connect.Expression` of `expression`, an object that is not an expression of this
/// client, as PySpark's requests build it: of its `to_plan` of the client of the active session
/// (of None without one), whatever the class of the object, as PySpark calls it, so an object of
/// no `to_plan` raises its `AttributeError`. Serialized, for the core to decode.
pub(crate) fn planned_expression<'py>(
    expression: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let py = expression.py();
    let client = match crate::session::active_session(py) {
        Some(session) => session.bind(py).getattr(pyo3::intern!(py, "client"))?,
        None => py.None().into_bound(py),
    };
    let proto = expression.call_method1(pyo3::intern!(py, "to_plan"), (client,))?;
    Ok(proto
        .call_method0(pyo3::intern!(py, "SerializeToString"))?
        .cast_into::<pyo3::types::PyBytes>()?)
}

/// The Column of `bytes`, a serialized `spark.connect.Expression`, sent as it is and rendered as
/// `rendered`, of the levels of its messages, which bound those of the expression. Of no origin
/// of the call that converts it: the message is sent of its own `common`, as PySpark's object
/// builds it of the origin that it was made with.
pub(crate) fn sent_column(bytes: &[u8], rendered: String) -> PyResult<Column> {
    use prost::Message;
    let levels = spark_connect_proto::response_depth_of("spark.connect.Expression", bytes)
        .map(|depth| depth.levels.max(1) as usize)
        .unwrap_or(spark_connect::expression::MAX_EXPRESSION_DEPTH + 1);
    // Decoded, and then built or dropped, on a stack that holds its levels.
    let column = crate::on_grown_stack(|| {
        spark_connect_proto::Expression::decode(bytes)
            .map(|decoded| Column::sent(decoded, rendered, levels))
    })
    .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
    .to_pyerr()?;
    Ok(Column::new(column.expression().with_origin(None)))
}

/// The core expression of the `_expr` of a Column, read at its first use in a method, where
/// PySpark's method reads `self._expr`: after the checks of its arguments that PySpark's method
/// makes first (see [`column_of`]).
struct Expr<'a, 'py> {
    column: &'a Bound<'py, PyColumn>,
    value: std::cell::OnceCell<Column>,
}

impl<'a, 'py> Expr<'a, 'py> {
    fn of(column: &'a Bound<'py, PyColumn>) -> Self {
        Expr {
            column,
            value: std::cell::OnceCell::new(),
        }
    }

    /// The expression, of the `_expr` of the Column, read at the first call.
    fn get(&self) -> PyResult<&Column> {
        if let Some(value) = self.value.get() {
            return Ok(value);
        }
        let value = column_of(self.column)?;
        Ok(self.value.get_or_init(|| value))
    }
}

/// The reduce `reduce` of `owner` with the entries of its `__dict__` but `key`, whose value the
/// reduce gives back itself, as its state, its third element, which `copy` and `pickle` restore
/// into the `__dict__` of the object that they rebuild, as of a plain object. A reduce of no other
/// entry is left as it is.
pub(crate) fn reduce_with_state_but<'py>(
    reduce: &Bound<'py, PyAny>,
    owner: &Bound<'py, PyAny>,
    key: &Bound<'py, pyo3::types::PyString>,
) -> PyResult<Bound<'py, PyAny>> {
    let state = owner
        .getattr(pyo3::intern!(owner.py(), "__dict__"))?
        .cast_into::<PyDict>()?
        .copy()?;
    if state.contains(key)? {
        state.del_item(key)?;
    }
    if state.is_empty() {
        return Ok(reduce.clone());
    }
    let mut items: Vec<Bound<'py, PyAny>> = reduce.cast::<PyTuple>()?.iter().collect();
    items.push(state.into_any());
    Ok(PyTuple::new(owner.py(), items)?.into_any())
}

/// The reduce `reduce` of `owner`, an object of the class `base` or of a Python subclass of it,
/// of the function that gives back an object of `base` and its arguments: of an object of a
/// subclass, the reduce of `_column_reduce.instance_of`, which makes an object of the subclass of
/// the entries of the `__dict__` of that object (and its native state), as PySpark's pickle of an
/// object of a subclass, a plain object, unpickles as the subclass.
pub(crate) fn subclass_reduce<'py>(
    owner: &Bound<'py, PyAny>,
    base: &Bound<'py, pyo3::types::PyType>,
    reduce: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = owner.py();
    let class = owner.get_type();
    if class.is(base) {
        return Ok(reduce);
    }
    let reduce = reduce.cast_into::<PyTuple>()?;
    let instance_of = py
        .import("pyspark.sql.connect.dataframe")?
        .getattr("_column_reduce")?
        .getattr("instance_of")?;
    let args = PyTuple::new(
        py,
        [class.into_any(), reduce.get_item(0)?, reduce.get_item(1)?],
    )?;
    Ok(PyTuple::new(py, [instance_of, args.into_any()])?.into_any())
}

/// `class.__new__(class, *args)`, of the arguments `args`, a tuple, as `copyreg.__newobj__`
/// makes the copy of a plain object.
fn new_of_class<'py>(
    class: &Bound<'py, pyo3::types::PyType>,
    args: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut items = vec![class.clone().into_any()];
    items.extend(args.try_iter()?.collect::<PyResult<Vec<_>>>()?);
    class.call_method1(
        pyo3::intern!(class.py(), "__new__"),
        PyTuple::new(class.py(), items)?,
    )
}

/// The reference of an object that holds `column` to the parts of its expression.
fn expression_parts(py: Python<'_>, column: &Column) -> crate::gc::Parts {
    crate::gc::Parts::of(
        py,
        [spark_connect::gc::PartRef::Expression(column.expression())],
    )
}

/// Helper to convert Python object to Column.
fn py_obj_to_column(obj: &Bound<'_, PyAny>) -> PyResult<Column> {
    to_column(obj)
}

/// The Column of an operand of a binary operator, as PySpark's `_bin_op` takes it: a Column, or
/// a literal (see `is_literal_operand`). Another object is looked up for its `_expr`, as PySpark
/// does, so that its AttributeError, or a Row's PySparkAttributeError, is the one it raises.
pub(crate) fn operand_column(obj: &Bound<'_, PyAny>) -> PyResult<Column> {
    operand(obj, true)
}

/// The Column of the operand of `__eq__`, as PySpark's `Column.__eq__` takes it: as `_bin_op`,
/// but without timedelta, which `__eq__` does not take as a literal.
fn equality_operand_column(obj: &Bound<'_, PyAny>) -> PyResult<Column> {
    operand(obj, false)
}

fn operand(obj: &Bound<'_, PyAny>, with_timedelta: bool) -> PyResult<Column> {
    // The literal is converted from the object itself, as `to_column` takes an Enum's value.
    let value = enum_to_value(obj)?;
    if obj.is_instance_of::<PyColumn>() || is_literal_operand(&value, with_timedelta)? {
        return to_column(obj);
    }
    // The `_expr` of PySpark's operand: an object that has none raises its lookup's error here.
    value.getattr("_expr")?;
    // An object with an `_expr` is an expression of PySpark's, which this client does not take.
    let type_name = value.get_type().name()?.to_string();
    Err(raise_pyspark(
        value.py(),
        "PySparkTypeError",
        "UNSUPPORTED_DATA_TYPE",
        &[("data_type", &type_name)],
    ))
}

/// Whether `value` is a literal of PySpark's `_bin_op`, by isinstance: None, or an instance of
/// int (a bool too), float (a NumPy float64 too), str, date, time, datetime, timedelta (unless
/// `with_timedelta` is false) or Decimal. A NumPy int64 or bool_ is not one.
fn is_literal_operand(value: &Bound<'_, PyAny>, with_timedelta: bool) -> PyResult<bool> {
    if value.is_none()
        || value.is_instance_of::<pyo3::types::PyInt>()
        || value.is_instance_of::<pyo3::types::PyFloat>()
        || value.is_instance_of::<pyo3::types::PyString>()
    {
        return Ok(true);
    }
    let py = value.py();
    let datetime = py.import("datetime")?;
    let mut types = vec![
        datetime.getattr("date")?,
        datetime.getattr("time")?,
        datetime.getattr("datetime")?,
    ];
    if with_timedelta {
        types.push(datetime.getattr("timedelta")?);
    }
    types.push(py.import("decimal")?.getattr("Decimal")?);
    value.is_instance(pyo3::types::PyTuple::new(py, types)?.as_any())
}

/// The expression of a Column, as PySpark Connect's `Column._expr`: its `str` and `repr` are
/// its rendering, as `Column<'...'>` shows it, and its `to_plan` its `spark.connect.Expression`.
/// The expressions of PySpark Connect are those of `pyspark.sql.connect.expressions`, of which
/// the Rust core builds the expressions of a Column. So is the `_subquery_expr` of a table
/// argument, and each entry of the `_partitionSpec` and `_orderSpec` of a window specification
/// that the client makes. It is the one holder of the parts of its expression: the objects whose
/// `__dict__` holds it hold nothing of it natively.
#[pyclass(name = "_ColumnExpression", module = "pyspark.sql.connect.expressions")]
pub struct PyColumnExpression {
    column: Column,
    /// The reference of the expression to its parts, which `__traverse__` reports (see
    /// [`crate::gc::Parts`]).
    parts: crate::gc::Parts,
    /// Set once the value is written, as `__traverse__` reads `parts` (see
    /// [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

impl PyColumnExpression {
    /// The expression of `column`.
    pub(crate) fn new(column: Column) -> Self {
        Python::attach(|py| Self::new_with(py, column))
    }

    /// The expression of `column`, of the GIL that the caller holds.
    pub(crate) fn new_with(py: Python<'_>, column: Column) -> Self {
        PyColumnExpression {
            parts: expression_parts(py, &column),
            column,
            gc_ready: crate::gc::GcReady::new(),
        }
    }

    /// The Column of the expression of the core.
    pub(crate) fn column(&self) -> &Column {
        &self.column
    }
}

/// The core expression of `expression`, an expression that a Column, a window specification or a
/// table argument holds: of a `_ColumnExpression`, its own; of another object, such as an
/// expression of PySpark's classes, of its `to_plan` (see [`converted_expression`]), at each call,
/// as PySpark builds the message of the expression at each request.
pub(crate) fn expression_value(expression: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(expression) = expression.cast::<PyColumnExpression>() {
        return Ok(expression.borrow().column.clone());
    }
    converted_expression(expression)
}

#[pymethods]
impl PyColumnExpression {
    /// The shallow copy of the expression, as PySpark's `copy.copy` of an expression, a Python
    /// object, which shares its fields: of the same expression, as the copy of a Column shares it.
    fn __copy__(&self) -> PyColumnExpression {
        PyColumnExpression::new(self.column.clone())
    }

    /// The deep copy of the expression, as PySpark's `copy.deepcopy` of an expression, which
    /// copies its fields with `memo`, of the function of each UDF of it and of the plan of a
    /// subquery, as the deep copy of a Column copies them (`deep_copied_picklers`): a function is
    /// the same object, and a callable object, a bound method or a partial is a copy.
    fn __deepcopy__(
        &self,
        py: Python<'_>,
        memo: &Bound<'_, PyAny>,
    ) -> PyResult<PyColumnExpression> {
        let expression = self.column.expression();
        // As the deep copy of a Column of the plan of a model (see `PyColumn::__deepcopy__`).
        let reach = crate::ml::object::expression_pickle_reach(py, expression)?;
        if let Some(error) = crate::ml::object::reach_deepcopy_error(py, &reach, memo)? {
            return Err(error);
        }
        let copies = crate::udf_command::deep_copied_picklers(py, expression.picklers(), memo)?;
        Ok(PyColumnExpression::new(if copies.is_empty() {
            self.column.clone()
        } else {
            Column::new(expression.with_pickler_copies(&copies))
        }))
    }

    /// As PySpark's expression, a Python object whose fields Python's garbage collector sees: the
    /// parts of the expression, of the functions of its UDFs and the plans of the Python client
    /// of the plans of its subqueries and of the errors of its parts (see [`crate::gc`]), which it
    /// alone reports of the Columns, windows and table arguments whose `__dict__` holds it, so
    /// that a reference cycle through one of them and the expression is collected, as PySpark
    /// collects it.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        self.parts.visit(&visit)
    }

    /// Break a reference cycle that the garbage collector found unreachable: the expression is
    /// replaced by one that holds nothing, which drops the parts that `__traverse__` reported.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            self.parts.release();
            self.column = Column::new(spark_connect::expression::Expression::column(""));
        }
    }

    /// An expression pickles, as PySpark's expression, a Python object, pickles its fields: as the
    /// `_expr` of its Column, `getattr(column, "_expr")`, of the pickle of the Column (see
    /// `PyColumn::__reduce__`), so that the unpickled expression is that of the unpickled Column,
    /// also on a Python worker of the server, where it is the expression of PySpark Connect's
    /// Column.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let column = column_object(py, self.column.clone())?;
        let args = PyTuple::new(
            py,
            [
                column.into_any(),
                pyo3::types::PyString::new(py, "_expr").into_any(),
            ],
        )?;
        PyTuple::new(
            py,
            [py.import("builtins")?.getattr("getattr")?, args.into_any()],
        )
    }

    /// As PySpark's `__repr__` of the expression, which raises its error of an expression that it
    /// does not render (a `ColumnAlias` of names that are not all `str`).
    fn __repr__(&self) -> PyResult<String> {
        self.column.expression().try_render().to_pyerr()
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }

    /// The `spark.connect.Expression` of this expression, of `pyspark.sql.connect.proto`. One
    /// too deep to be sent is the `RecursionError` that PySpark's `to_plan`, which recurses,
    /// raises. Built of the session of the client `session` (see `errors::of_client_session`).
    fn to_plan(&self, py: Python<'_>, session: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        crate::errors::of_client_session(session, || expression_message(py, &self.column))
    }
}

/// `Column.__getattr__(column, item)` called as a function of the class: the `__getattr__` of the
/// Column, which PySpark's attribute access of a name that a Column does not have calls.
#[pyfunction]
fn _column_getattr(
    column: &Bound<'_, PyColumn>,
    item: &Bound<'_, pyo3::types::PyString>,
) -> PyResult<ColumnOut> {
    PyColumn::__getattr__(column, column.py(), item)
}

/// Make `__getattr__` a function of the class `Column`, as PySpark Connect's `Column.__getattr__`
/// is, so that `Column.__getattr__(col, "a")` is the field `a` of `col`, as in PySpark: a Python
/// function of that name, which calls the `__getattr__` of the Column. PyO3 makes `__getattr__`
/// only the attribute hook of the type, and that hook, the wrapper `__getattribute__` of the class,
/// is removed, so that an attribute is looked up as of PySpark's class: the class's attributes,
/// and else its `__getattr__`.
pub(crate) fn install_getattr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let globals = PyDict::new(py);
    globals.set_item("__name__", "pyspark.sql.connect.column")?;
    globals.set_item("_getattr", wrap_pyfunction!(_column_getattr, m)?)?;
    py.run(
        c"def __getattr__(self, item):\n    return _getattr(self, item)\n",
        Some(&globals),
        None,
    )?;
    let getattr = globals
        .get_item("__getattr__")?
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("__getattr__ is not defined"))?;
    getattr.setattr("__qualname__", "Column.__getattr__")?;
    let column = py.get_type::<PyColumn>();
    column.setattr("__getattr__", getattr)?;
    column.delattr("__getattribute__")
}

/// The `spark.connect.Expression` of `column`, a message of `pyspark.sql.connect.proto`, as
/// PySpark Connect's `to_plan` of its expression builds it, of any depth (where upb's parse stops
/// at 100 levels). One too deep to be sent is the `RecursionError` that PySpark's `to_plan`, which
/// recurses, raises.
pub(crate) fn expression_message(py: Python<'_>, column: &Column) -> PyResult<Py<PyAny>> {
    use prost::Message;
    let expression = column.try_to_proto().to_pyerr()?;
    let class = py
        .import("pyspark.sql.connect.proto")?
        .getattr("Expression")?;
    Ok(crate::pbmsg::parse_deep(&class, &expression.encode_to_vec())?.unbind())
}

#[pymethods]
impl PyColumn {
    /// A Column of no expression yet, as PySpark Connect's `Column.__new__(cls, *args, **kwargs)`,
    /// which is `object.__new__(cls)`, so that a subclass of another `__init__` is built as it is
    /// in PySpark: until its `__init__` gives it its `_expr`, its methods raise the
    /// `RecursionError` of its `__getattr__`, as PySpark's.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(*args, **kwargs)")]
    fn py_new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyColumn::empty()
    }

    /// `Column(expr)`, as PySpark Connect's `__init__`, which keeps the expression `expr` as its
    /// `_expr`: an expression of this client, the `_expr` of a Column, or one of PySpark's classes
    /// of `pyspark.sql.connect.expressions`, such as a `LiteralExpression`, which the Column
    /// sends as its `to_plan` gives it; another object is PySpark's `NOT_EXPECTED_TYPE`. Of the
    /// arguments as CPython binds them to PySpark's `def __init__(self, expr)`, with its errors.
    #[pyo3(signature = (*args, **kwargs))]
    fn __init__(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let expr = bound_argument(args, kwargs)?;
        check_expression(&expr)?;
        crate::state::assign(slf.as_any(), &[(pyo3::intern!(py, "_expr"), &expr)])
    }

    // Alias the column with each given name, optionally attaching column metadata (a
    // dict). Mirrors `Column.alias(*alias, **kwargs)`, which names a generator's columns in
    // order, takes metadata for a single name only, and sends it as `json.dumps(metadata)`.
    #[pyo3(signature = (*alias, **kwargs))]
    #[doc = connect_doc!(Column.alias)]
    pub(crate) fn alias(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        alias: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        crate::options::check_keyword_keys(kwargs)?;
        let _origin = crate::origin::with_origin("alias")?;
        let mut metadata = None;
        if let Some(kwargs) = kwargs {
            let rest = kwargs.copy()?;
            if let Some(value) = rest.get_item("metadata")? {
                rest.del_item("metadata")?;
                if !value.is_none() {
                    metadata = Some(value);
                }
            }
            if !rest.is_empty() {
                return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
                    "Unexpected kwargs where passed: {}",
                    rest.repr()?
                )));
            }
        }
        if alias.len() > 1 && metadata.is_some() {
            return Err(raise_pyspark(
                py,
                "PySparkValueError",
                "ONLY_ALLOWED_FOR_SINGLE_COLUMN",
                &[("arg_name", "metadata")],
            ));
        }
        let single = alias.len() == 1;
        // A name is of a string field of the plan, which PySpark's `ColumnAlias.to_plan` sets when
        // it builds the plan: a str, a bytes of its UTF-8, and the error of the first other value,
        // which the request of the plan raises before it builds the child. Such a name renders as
        // `crate::plan_args::rendered` gives it: a str that UTF-8 does not encode escaped
        // (PySpark's `__repr__` shows it as it is, which a rendering of UTF-8 cannot hold), and
        // another value as nothing, of a `__repr__` that fails.
        let mut names_args = crate::plan_args::PlanArgs::new();
        let names = names_args.strs(alias.iter())?;
        let mut before = names_args.error().cloned();
        let mut after = None;
        let mut node = Alias::with_names(this.get()?.expression().clone(), names);
        if let Some(metadata) = metadata {
            if single {
                // `json.dumps(metadata)`, of which `to_plan` sets the metadata after it builds the
                // child: its TypeError or ValueError is raised then.
                match py.import("json")?.call_method1("dumps", (&metadata,)) {
                    Ok(json) => node = node.with_metadata(json.extract()?),
                    Err(err) => after = Some(crate::dataframe::unbuildable_error(py, err)?),
                }
            } else if metadata.is_truthy()? {
                // Of no names (several, the check above refuses): `to_plan` raises this before it
                // sets the names.
                before = Some(spark_connect_core::error::SparkError::value(
                    "CANNOT_PROVIDE_METADATA",
                    &[],
                ));
            }
        }
        // PySpark's `ColumnAlias.__repr__` joins the names, which fails of one that is not a str
        // (bytes too), as the `__repr__` of a Column of it raises.
        let unrenderable = match pyo3::types::PyString::new(py, ",").call_method1("join", (alias,))
        {
            Ok(_) => None,
            Err(err) => Some(crate::dataframe::unbuildable_error(py, err)?),
        };
        Ok(ColumnOut::new(Column::new(
            spark_connect::expression::Expression::unbuildable(
                ExpressionNode::Alias(Box::new(node)).into(),
                before,
                after,
                unrenderable,
            ),
        )))
    }

    // Alias for `alias` (pyspark `Column.name`).
    #[pyo3(signature = (*alias, **kwargs))]
    fn name(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        alias: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<ColumnOut> {
        let _origin = crate::origin::with_origin("name")?;
        Self::alias(slf, py, alias, kwargs)
    }

    // Apply a transformation function to this column. Mirrors `Column.transform(f)`,
    // which is simply `f(self)`.
    #[doc = connect_doc!(Column.transform)]
    fn transform<'py>(slf: Bound<'py, Self>, f: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let _origin = crate::origin::with_origin("transform")?;
        f.call1((slf,))
    }

    // Mark this column as an outer reference (for correlated subqueries / lateral
    // joins). Mirrors `Column.outer()`, which returns the same expression — outer
    // resolution is performed server-side via the plan id.
    #[doc = connect_doc!(Column.outer)]
    fn outer<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyColumn>> {
        let _origin = crate::origin::with_origin("outer")?;
        // As PySpark's `Column(self._expr)`, of the same expression, which its `__init__` checks.
        let expression = expression_attr(slf.as_any())?;
        check_expression(&expression)?;
        column_of_expression(&expression)
    }

    // Cast to a different type. Accepts a `DataType` or a DDL type string,
    // matching `pyspark.sql.Column.cast`.
    #[doc = connect_doc!(Column.cast)]
    fn cast(slf: &Bound<'_, Self>, dataType: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("cast")?;
        // As PySpark's, of a DataType or a str.
        if !(dataType.is_instance_of::<pyo3::types::PyString>()
            || dataType.is_instance_of::<crate::types::PyDataType>())
        {
            return Err(not_expected_type(dataType, "dataType", "DataType or str"));
        }
        // A DDL string keeps the unparsed type_str form (reference cast("int"));
        // any DataType object is converted to a typed cast.
        cast_column(this.get()?, dataType, false)
    }

    // Alias for `cast` (pyspark `Column.astype`).
    fn astype(slf: &Bound<'_, Self>, dataType: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
        let _origin = crate::origin::with_origin("astype")?;
        Self::cast(slf, dataType)
    }

    // Try to cast, yielding NULL on failure. Accepts a `DataType` or DDL string.
    // Mirrors `pyspark.sql.Column.try_cast`.
    #[pyo3(name = "try_cast")]
    #[doc = connect_doc!(Column.try_cast)]
    fn try_cast(slf: &Bound<'_, Self>, dataType: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("try_cast")?;
        // As PySpark's: a DataType or a str, else NOT_EXPECTED_TYPE.
        if !dataType.is_instance_of::<crate::types::PyDataType>()
            && !dataType.is_instance_of::<pyo3::types::PyString>()
        {
            return Err(crate::stat_arguments::not_expected_type(
                dataType,
                "dataType",
                "DataType or str",
            ));
        }
        cast_column(this.get()?, dataType, true)
    }

    // Get an item from a list/map by key (pyspark `Column.getItem`), with PySpark's
    // FutureWarning for a Column as the key, of the line of its `warnings.warn` in
    // `pyspark.sql.connect.column` (see `warn_as_pyspark`).
    #[pyo3(name = "getItem")]
    #[doc = connect_doc!(Column.getItem)]
    fn get_item(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
        let _origin = crate::origin::with_origin("getItem")?;
        if key.is_instance_of::<PyColumn>() {
            let py = key.py();
            crate::errors::warn_as_pyspark(
                py,
                "pyspark.sql.connect.column",
                527,
                // PySpark's message, split so that no line reads as a deferral marker of
                // scripts/audit_no_stubs.sh.
                "A column as 'key' in getItem is deprecated as of Spark 3.0, and will not \
                 be supported in the \
                 future release. Use `column[key]` or `column.key` syntax instead.",
                &py.get_type::<pyo3::exceptions::PyFutureWarning>(),
            )?;
        }
        // As PySpark's `self[key]`: its `__getitem__`, of the origin of that, which takes a slice
        // as a `substr`.
        Self::__getitem__(slf, key)
    }

    // Check if NULL.
    #[pyo3(name = "isNull")]
    #[doc = connect_doc!(Column.isNull)]
    fn is_null(slf: &Bound<'_, Self>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("isNull")?;
        Ok(ColumnOut::new(this.get()?.clone().is_null()))
    }

    // Check if NOT NULL.
    #[pyo3(name = "isNotNull")]
    #[doc = connect_doc!(Column.isNotNull)]
    fn is_not_null(slf: &Bound<'_, Self>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("isNotNull")?;
        Ok(ColumnOut::new(this.get()?.clone().is_not_null()))
    }

    // Substring extraction (start and length as columns).
    #[doc = connect_doc!(Column.substr)]
    fn substr(
        slf: &Bound<'_, Self>,
        startPos: &Bound<'_, PyAny>,
        length: &Bound<'_, PyAny>,
    ) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("substr")?;
        // As PySpark: the arguments, as values of an Enum, must be of one type, a Column or
        // an int.
        let py = startPos.py();
        let enum_to_value = py.import("pyspark.sql.utils")?.getattr("enum_to_value")?;
        let start = enum_to_value.call1((startPos,))?;
        let length = enum_to_value.call1((length,))?;
        let type_name = |value: &Bound<'_, PyAny>| -> PyResult<String> {
            Ok(value.get_type().name()?.to_string())
        };
        if !start.get_type().is(length.get_type()) {
            let (start_type, length_type) = (type_name(&start)?, type_name(&length)?);
            return Err(raise_pyspark(
                py,
                "PySparkTypeError",
                "NOT_SAME_TYPE",
                &[
                    ("arg_name1", "startPos"),
                    ("arg_name2", "length"),
                    ("arg_type1", &start_type),
                    ("arg_type2", &length_type),
                ],
            ));
        }
        if !(length.is_instance_of::<PyColumn>() || length.is_instance_of::<pyo3::types::PyInt>()) {
            let length_type = type_name(&length)?;
            return Err(raise_pyspark(
                py,
                "PySparkTypeError",
                "NOT_EXPECTED_TYPE",
                &[
                    ("expected_type", "Column or int"),
                    ("arg_name", "startPos"),
                    ("arg_type", &length_type),
                ],
            ));
        }
        let start_col = py_obj_to_column(&start)?;
        let length_col = py_obj_to_column(&length)?;
        Ok(ColumnOut::new(
            this.get()?.clone().substr(start_col, length_col),
        ))
    }

    // Pattern matching with LIKE, of any value, as PySpark's `_bin_op("like", ...)`.
    #[doc = connect_doc!(Column.like)]
    fn like(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("like")?;
        binary_function("like", this.get()?, other)
    }

    // Pattern matching with RLIKE (regex).
    #[doc = connect_doc!(Column.rlike)]
    fn rlike(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("rlike")?;
        binary_function("rlike", this.get()?, other)
    }

    // Check if contains another column value.
    #[doc = connect_doc!(Column.contains)]
    fn contains(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("contains")?;
        let other_col = operand_column(other)?;
        Ok(ColumnOut::new(this.get()?.clone().contains(other_col)))
    }

    // Check if starts with another column value.
    #[doc = connect_doc!(Column.startswith)]
    fn startswith(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("startswith")?;
        let other_col = operand_column(other)?;
        Ok(ColumnOut::new(this.get()?.clone().startswith(other_col)))
    }

    // Check if ends with another column value.
    #[doc = connect_doc!(Column.endswith)]
    fn endswith(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("endswith")?;
        let other_col = operand_column(other)?;
        Ok(ColumnOut::new(this.get()?.clone().endswith(other_col)))
    }

    // Ascending sort.
    #[doc = connect_doc!(Column.asc)]
    fn asc(slf: &Bound<'_, Self>) -> PyResult<ColumnOut> {
        let _origin = crate::origin::with_origin("asc")?;
        // As PySpark's `asc`, which is its `asc_nulls_first`, of the origin of that.
        Self::asc_nulls_first(slf)
    }

    // Ascending sort with nulls first.
    #[doc = connect_doc!(Column.asc_nulls_first)]
    fn asc_nulls_first(slf: &Bound<'_, Self>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("asc_nulls_first")?;
        Ok(ColumnOut::new(this.get()?.clone().asc_nulls_first()))
    }

    // Ascending sort with nulls last.
    #[doc = connect_doc!(Column.asc_nulls_last)]
    fn asc_nulls_last(slf: &Bound<'_, Self>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("asc_nulls_last")?;
        Ok(ColumnOut::new(this.get()?.clone().asc_nulls_last()))
    }

    // Descending sort.
    #[doc = connect_doc!(Column.desc)]
    fn desc(slf: &Bound<'_, Self>) -> PyResult<ColumnOut> {
        let _origin = crate::origin::with_origin("desc")?;
        // As PySpark's `desc`, which is its `desc_nulls_last`, of the origin of that.
        Self::desc_nulls_last(slf)
    }

    // Descending sort with nulls first.
    #[doc = connect_doc!(Column.desc_nulls_first)]
    fn desc_nulls_first(slf: &Bound<'_, Self>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("desc_nulls_first")?;
        Ok(ColumnOut::new(this.get()?.clone().desc_nulls_first()))
    }

    // Descending sort with nulls last.
    #[doc = connect_doc!(Column.desc_nulls_last)]
    fn desc_nulls_last(slf: &Bound<'_, Self>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("desc_nulls_last")?;
        Ok(ColumnOut::new(this.get()?.clone().desc_nulls_last()))
    }

    // A branch more of a CASE WHEN. Mirrors `Column.when`: the condition is a Column, and
    // the column a `when()` without `otherwise()` (`INVALID_WHEN_USAGE`).
    #[doc = connect_doc!(Column.when)]
    fn when(
        slf: &Bound<'_, Self>,
        condition: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("when")?;
        let py = condition.py();
        let Ok(condition) = condition.cast::<PyColumn>() else {
            let arg_type = condition.get_type().name()?.to_string();
            return Err(raise_pyspark(
                py,
                "PySparkTypeError",
                "NOT_EXPECTED_TYPE",
                &[
                    ("expected_type", "Column"),
                    ("arg_name", "condition"),
                    ("arg_type", &arg_type),
                ],
            ));
        };
        match this.get()?.expression().node() {
            ExpressionNode::CaseWhen(case_when) if case_when.else_expr.is_none() => {}
            _ => {
                return Err(raise_pyspark(
                    py,
                    "PySparkTypeError",
                    "INVALID_WHEN_USAGE",
                    &[],
                ))
            }
        }
        let val_col = to_value_column(value)?;
        Ok(ColumnOut::new(
            this.get()?.clone().when(column_of(condition)?, val_col),
        ))
    }

    // The value of a CASE WHEN when no branch holds. Mirrors `Column.otherwise`, which
    // raises PySparkTypeError for a column that is not a `when()`, or one that has one.
    #[doc = connect_doc!(Column.otherwise)]
    fn otherwise(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("otherwise")?;
        let message = match this.get()?.expression().node() {
            ExpressionNode::CaseWhen(case_when) if case_when.else_expr.is_none() => None,
            ExpressionNode::CaseWhen(_) => Some(
                "otherwise() can only be applied once on a Column previously generated by when()",
            ),
            _ => Some("otherwise() can only be applied on a Column previously generated by when()"),
        };
        if let Some(message) = message {
            let py = value.py();
            let error = py
                .import("pyspark.errors")?
                .getattr("PySparkTypeError")?
                .call1((message,))?;
            return Err(PyErr::from_value(error));
        }
        let val_col = to_value_column(value)?;
        Ok(ColumnOut::new(this.get()?.clone().otherwise(val_col)))
    }

    /// Get item by key/index, or a slice as `substr(start, stop)`. Mirrors
    /// `Column.__getitem__`.
    fn __getitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__getitem__")?;
        if key.is_instance_of::<pyo3::types::PySlice>() {
            if !key.getattr("step")?.is_none() {
                return Err(raise_pyspark(
                    key.py(),
                    "PySparkValueError",
                    "SLICE_WITH_STEP",
                    &[],
                ));
            }
            return Self::substr(slf, &key.getattr("start")?, &key.getattr("stop")?);
        }
        let key_col = to_value_column(key)?;
        Ok(ColumnOut::new(this.get()?.clone().get_item(key_col)))
    }

    /// An attribute that a Column does not have is the field of that name, as
    /// `col["name"]`. Mirrors `Column.__getattr__`, which raises for `_jc` and for a name
    /// that starts with `__`. The name is the str of the attribute as it is: one that UTF-8
    /// cannot encode is the field of a literal that fails the plans of the Column, as `col[name]`.
    fn __getattr__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        item: &Bound<'_, pyo3::types::PyString>,
    ) -> PyResult<ColumnOut> {
        let _origin = crate::origin::with_origin("__getattr__")?;
        if item.as_any().eq("_jc")? {
            return Err(raise_pyspark(
                py,
                "PySparkAttributeError",
                "JVM_ATTRIBUTE_NOT_SUPPORTED",
                &[("attr_name", "_jc")],
            ));
        }
        if item.call_method1("startswith", ("__",))?.is_truthy()? {
            return Err(crate::errors::raise_pyspark_of(
                py,
                "PySparkAttributeError",
                "ATTRIBUTE_NOT_SUPPORTED",
                &[("attr_name", item.as_any())],
            ));
        }
        Self::__getitem__(slf, item.as_any())
    }

    /// Equality (==) - returns a Column condition
    fn __eq__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__eq__")?;
        let other_col = equality_operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().eq(other_col)))
    }

    /// Inequality (!=) - returns a Column condition
    fn __ne__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__ne__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().ne(other_col)))
    }

    /// Greater than (>) - returns a Column condition
    fn __gt__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__gt__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().gt(other_col)))
    }

    /// Less than (<) - returns a Column condition
    fn __lt__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__lt__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().lt(other_col)))
    }

    /// Greater than or equal (>=) - returns a Column condition
    fn __ge__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__ge__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().ge(other_col)))
    }

    /// Less than or equal (<=) - returns a Column condition
    fn __le__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__le__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().le(other_col)))
    }

    // Arithmetic operators
    /// Addition (+)
    fn __add__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__add__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().add(other_col)))
    }

    /// Right addition (for reverse operations)
    fn __radd__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__radd__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(other_col.add(this.get()?.clone())))
    }

    /// Subtraction (-)
    fn __sub__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__sub__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().sub(other_col)))
    }

    /// Right subtraction (for reverse operations)
    fn __rsub__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__rsub__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(other_col.sub(this.get()?.clone())))
    }

    /// Multiplication (*)
    fn __mul__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__mul__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().mul(other_col)))
    }

    /// Right multiplication (for reverse operations)
    fn __rmul__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__rmul__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(other_col.mul(this.get()?.clone())))
    }

    /// Division (/)
    fn __truediv__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__truediv__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().div(other_col)))
    }

    /// Division (/), of the name of Python 2, as PySpark's Column has it, which pandas-on-Spark
    /// calls: as `__truediv__`.
    fn __div__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__div__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().div(other_col)))
    }

    /// Modulo (%)
    fn __mod__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__mod__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().modulo(other_col)))
    }

    // Bitwise operators (logical in Spark context)
    /// Bitwise AND (&) - logical AND in Spark
    fn __and__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__and__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().and(other_col)))
    }

    /// Bitwise OR (|) - logical OR in Spark
    fn __or__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__or__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().or(other_col)))
    }

    /// Bitwise NOT (~) - logical NOT in Spark
    fn __invert__(slf: &Bound<'_, Self>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__invert__")?;
        Ok(ColumnOut::new(this.get()?.clone().not()))
    }

    /// Negation (-)
    fn __neg__(slf: &Bound<'_, Self>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__neg__")?;
        Ok(ColumnOut::new(this.get()?.clone().neg()))
    }

    // Get field by name (for struct/nested types).
    #[pyo3(name = "getField")]
    #[doc = connect_doc!(Column.getField)]
    fn get_field(slf: &Bound<'_, Self>, name: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
        let _origin = crate::origin::with_origin("getField")?;
        // As PySpark's, `self[name]`, with a FutureWarning for a Column, of the line of its
        // `warnings.warn` in `pyspark.sql.connect.column`.
        if name.is_instance_of::<PyColumn>() {
            let py = name.py();
            crate::errors::warn_as_pyspark(
                py,
                "pyspark.sql.connect.column",
                537,
                // PySpark's message, split so that no line reads as a deferral marker of
                // scripts/audit_no_stubs.sh.
                "A column as 'name' in getField is deprecated as of Spark 3.0, and will not \
                 be supported in the \
                 future release. Use `column[name]` or `column.name` syntax instead.",
                &py.get_type::<pyo3::exceptions::PyFutureWarning>(),
            )?;
        }
        Self::__getitem__(slf, name)
    }

    // Apply a window function over a window specification.
    #[doc = connect_doc!(Column.over)]
    fn over(slf: &Bound<'_, Self>, window: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("over")?;
        if !window.is_instance_of::<crate::window::PyWindowSpec>() {
            return Err(not_expected_type(window, "window", "WindowSpec"));
        }
        let function = this.get()?.clone();
        // PySpark's `WindowExpression` holds the window, and builds its message of the state of
        // the window when a request builds the plan: this one is of the state at the call, as the
        // expression is a value of the core (see `window::window_spec_of`). The error of a bound
        // of the frame that the window of the core does not hold is raised by each request, as
        // PySpark's build raises it, after the window function and the specs.
        let (spec, error, rendered) = crate::window::window_spec_of(window)?;
        if error.is_none() && rendered {
            return Ok(ColumnOut::new(function.over(spec)));
        }
        let column = function.clone().over(spec);
        // Rendered as PySpark's `WindowExpression.__repr__` renders the window, of the bounds of
        // its frame as they are, which the window of the core does not hold, or renders of
        // another text (a bool).
        use spark_connect::expression::Expression;
        let expression = column.expression().clone();
        Ok(ColumnOut::new(Column::new(
            match rendered_over(&function, window) {
                Some(rendered) => Expression::unbuildable_rendered(expression, error, rendered),
                None => Expression::unbuildable(expression, None, error, None),
            },
        )))
    }

    /// String representation, mirroring `pyspark.sql.connect.column.Column.__repr__`
    /// (`Column<'<expr>'>`). pandas-on-Spark's `spark_column_equals` relies on this.
    /// Of the error of PySpark's `__repr__` of an expression that it does not render.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        // As PySpark's `"Column<'%s'>" % self._expr.__repr__()`: of a `_ColumnExpression`, a
        // class of no subclass, its own `__repr__` at once.
        let py = slf.py();
        let expression = expression_attr(slf.as_any())?;
        if let Ok(expression) = expression.cast::<PyColumnExpression>() {
            return Ok(format!("Column<'{}'>", expression.borrow().__repr__()?));
        }
        let text = expression.call_method0(pyo3::intern!(py, "__repr__"))?;
        Ok(format!("Column<'{}'>", text.str()?.to_cow()?))
    }

    // The `spark.connect.Expression` of this column, as PySpark Connect's `Column.to_plan`: that
    // of its expression, of `pyspark.sql.connect.proto`.
    fn to_plan<'py>(
        slf: &Bound<'py, Self>,
        session: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // As PySpark's `self._expr.to_plan(session)`.
        let py = slf.py();
        expression_attr(slf.as_any())?.call_method1(pyo3::intern!(py, "to_plan"), (session,))
    }

    /// The shallow copy of a Column, as PySpark's `copy.copy` of a plain object: an object of its
    /// class, of `__new__` of the arguments of its `__getnewargs__`, its `_expr`, and of the
    /// entries of its `__dict__`, so the copy shares the expression of the Column and the values
    /// of its attributes, and the functions of its UDFs are pickled when a request builds a plan of
    /// the copy, as those of the Column, and not when it is copied.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let args = slf.call_method0(pyo3::intern!(py, "__getnewargs__"))?;
        let class = slf.get_type();
        let copy = new_of_class(&class, &args)?;
        let state = crate::group::instance_state(slf.as_any())?;
        if !state.is_none() {
            copy.getattr("__dict__")?.call_method1("update", (state,))?;
        }
        Ok(copy)
    }

    /// The deep copy of a Column, as PySpark's `copy.deepcopy` of a plain object: an object of its
    /// class, of `__new__` of the deep copies of the arguments of its `__getnewargs__`, its
    /// `_expr`, whose copy copies the function of each UDF with `memo` (see
    /// `PyColumnExpression::__deepcopy__`), and then of the deep copies of the entries of its
    /// `__dict__`, with `memo`, which has the copy before them, so the copy's `_expr` is the copy
    /// of the expression, as `copy` reconstructs a plain object.
    fn __deepcopy__<'py>(
        slf: &Bound<'py, Self>,
        memo: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let deepcopy = py.import("copy")?.getattr("deepcopy")?;
        let args = slf.call_method0(pyo3::intern!(py, "__getnewargs__"))?;
        let args = deepcopy.call1((args, memo))?;
        let class = slf.get_type();
        let copy = new_of_class(&class, &args)?;
        memo.set_item(slf.as_ptr() as usize, &copy)?;
        let state = crate::group::instance_state(slf.as_any())?;
        if !state.is_none() {
            let state = deepcopy.call1((state, memo))?;
            copy.getattr("__dict__")?.call_method1("update", (state,))?;
        }
        Ok(copy)
    }

    /// The reduce of a pickle, as PySpark's pickle of a Column, which pickles its expression, and
    /// the plans of its subqueries where they are, and so the objects of `pyspark.ml` that such a
    /// plan keeps (`exists()`, `scalar()`, `isin(df)` of a transform of a model, a DataFrame of a
    /// method of a model or of a summary): the reduce of the objects that PySpark's pickle reaches
    /// first, the functions of its UDFs before them and the lock of the model, or the summary
    /// dataset of a training summary, whose session a plain pickle refuses, and the lock, as the
    /// pickle of a DataFrame of the plan fails (see `PyDataFrame::__reduce_ex__`). Else that of
    /// `__reduce__`.
    fn __reduce_ex__<'py>(
        slf: &Bound<'py, Self>,
        protocol: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = protocol;
        let py = slf.py();
        let column = column_of(slf)?;
        let reach = crate::ml::object::expression_pickle_reach(py, column.expression())?;
        let reduced = slf.call_method0(pyo3::intern!(py, "__reduce__"))?;
        crate::ml::object::reach_reduce(py, &reach, reduced)
    }

    /// A Column pickles, as PySpark Connect's, which pickles its expression: of the bytes of the
    /// `spark.connect.Expression` of this column and of its rendering, of which the function of
    /// `pyspark.sql.connect.column._reduce_column` gives back the Column, also on a Python worker
    /// of the server, such as that of a `foreachBatch` function, as PySpark Connect's Column.
    /// A CASE WHEN also pickles its branches and else value, as PySpark's CaseWhen keeps them, so
    /// that `when` and `otherwise` chain on the unpickled Column. A sort order pickles its child
    /// Column and its sort method, as PySpark's SortOrder keeps its child and direction, so that
    /// the unpickled Column is a sort order of `orderBy`, `Window.orderBy` and the others.
    /// A Column too deep to be sent is the `RecursionError` that PySpark's pickler raises for it.
    /// A user-set attribute of the instance is the reduce's state, as of a plain object, so
    /// that `pickle` keeps it, as it does of PySpark's Column. The expression is of holes in place
    /// of the commands of its UDFs, and the pickle carries their functions, the objects of their
    /// picklers, as PySpark's pickle of the Column carries the functions: a plain pickle pickles a
    /// function by reference and fails at a lambda, and the requests of the unpickled Column
    /// pickle the functions as they are then ([`crate::udf_command::PickleHoles`]). A Column of
    /// subqueries pickles the relations of the plans that they refer to too, of their plan IDs
    /// and their holes, after the expression, as PySpark's pickle keeps the plan objects of its
    /// subqueries: the unpickled Column sends them in the `WithRelations` of the relation that
    /// holds it, as PySpark's plan sends the plans of its subqueries.
    fn __reduce__<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let column_value = column_of(slf)?;
        let (holes, copies) =
            crate::udf_command::PickleHoles::of(py, column_value.expression().picklers());
        let marked = if copies.is_empty() {
            column_value.expression().clone()
        } else {
            column_value.expression().with_pickler_copies(&copies)
        };
        // Of an expression that a request fails to build, as PySpark's Column of an alias of a
        // name that protobuf refuses: its proto of what this client builds, and the errors of
        // a request and of the `__repr__` of it, which the unpickled Column raises as it did.
        let (expression, levels, unbuildable) =
            crate::origin::pickling(|| marked.try_to_proto_with_levels_and_unbuildable())
                .to_pyerr()?;
        let mut build_error = match unbuildable.as_ref() {
            None => None,
            // The error of a plan that raised when it was built, which the pickle carries, as
            // that of a plan of PySpark's unpickled Column raises when it is built again; the
            // exception of another callback is raised.
            Some(error) => match described_error(py, error)? {
                Some(description) => Some(description),
                None => match crate::errors::callback_exception(error) {
                    Some(exception) => return Err(exception),
                    None => Some(error_description(py, error)?),
                },
            },
        };
        // A plan of a subquery that does not build pickles of its error, or of the plan of the
        // Python client whose build raised, which the unpickle builds again, as a DataFrame does,
        // which the requests of the unpickled Column raise; one that does not pickle, or that is
        // too deep, raises its error. A CASE WHEN and a sort order are given back of their parts,
        // which pickle their subqueries.
        let mut subqueries = Vec::new();
        let of_parts = matches!(
            column_value.expression().node(),
            ExpressionNode::CaseWhen(_) | ExpressionNode::SortOrder(_)
        );
        if build_error.is_none() && !of_parts {
            match crate::origin::pickling(|| marked.subquery_relations()) {
                Ok(relations) => {
                    for (relation, levels) in relations {
                        let relation = crate::encode_on_grown_stack(relation);
                        let relation_holes = holes.holes_of(py, &relation)?;
                        let relation = pyo3::types::PyBytes::new(py, &relation);
                        subqueries.push((relation, levels, relation_holes));
                    }
                }
                Err(error) => match described_error(py, &error)? {
                    Some(description) => build_error = Some(description),
                    None => return Err(crate::errors::spark_error_to_py_exception(error)),
                },
            }
        }
        let repr_error = column_value
            .expression()
            .try_render()
            .err()
            .map(|error| error_description(py, &error))
            .transpose()?;
        let expression = crate::encode_on_grown_stack(expression);
        let expression_holes = holes.holes_of(py, &expression)?;
        let expression = pyo3::types::PyBytes::new(py, &expression);
        let column = |expr: &spark_connect::expression::Expression| -> PyResult<Py<PyColumn>> {
            Ok(column_object(py, Column::new(expr.clone()))?.unbind())
        };
        let (branches, else_value) = match column_value.expression().node() {
            ExpressionNode::CaseWhen(case_when) => {
                let branches = case_when
                    .branches
                    .iter()
                    .map(|(condition, value)| Ok((column(condition)?, column(value)?)))
                    .collect::<PyResult<Vec<_>>>()?;
                let else_value = case_when.else_expr.as_ref().map(column).transpose()?;
                (Some(branches), else_value)
            }
            _ => (None, None),
        };
        let sort = match column_value.expression().node() {
            ExpressionNode::SortOrder(sort) => {
                let method = match (sort.ascending, sort.null_ordering) {
                    (true, NullOrdering::First) => "asc_nulls_first",
                    (true, NullOrdering::Last) => "asc_nulls_last",
                    (false, NullOrdering::First) => "desc_nulls_first",
                    (false, NullOrdering::Last) => "desc_nulls_last",
                };
                Some((column(&sort.child)?, method))
            }
            _ => None,
        };
        let reduce = py
            .import("pyspark.sql.connect.column")?
            .getattr("_reduce_column")?
            .call1((
                expression,
                column_value.expression().render(),
                branches,
                else_value,
                sort,
                levels,
                build_error,
                repr_error,
                subqueries,
                expression_holes,
            ))?;
        let reduce = subclass_reduce(slf.as_any(), &py.get_type::<PyColumn>(), reduce)?;
        // The state of the other attributes, as the expression is the reduce's.
        reduce_with_state_but(&reduce, slf.as_any(), pyo3::intern!(py, "_expr"))
    }

    /// The Column of `expression`, the bytes of a `spark.connect.Expression`, and of its
    /// `rendered` rendering, of its `levels`, as a pickled Column unpickles: it sends the
    /// expression as it is, of its `holes` filled with the commands of the functions of its UDFs
    /// at each request (see `udf_command::sent_holes`), and the relations of `subqueries`, the
    /// bytes of the relation of each plan that its subqueries refer to, its levels and its holes,
    /// in the `WithRelations` of the relation that holds it. The CASE WHEN of `branches` and
    /// `else_value`, the Columns of its parts, is built as it was.
    #[staticmethod]
    #[pyo3(signature = (
        expression,
        rendered,
        branches=None,
        else_value=None,
        levels=1,
        build_error=None,
        repr_error=None,
        subqueries=None,
        holes=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn _of_expression(
        py: Python<'_>,
        expression: &[u8],
        rendered: String,
        branches: Option<Vec<(Bound<'_, PyColumn>, Bound<'_, PyColumn>)>>,
        else_value: Option<Bound<'_, PyColumn>>,
        levels: usize,
        build_error: Option<&Bound<'_, PyAny>>,
        repr_error: Option<&Bound<'_, PyAny>>,
        subqueries: Option<Vec<Bound<'_, PyTuple>>>,
        holes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<ColumnOut> {
        // Of the module of the class imported, as PySpark's unpickle of a Column imports the
        // module of its class: an unpickled Column needs no session, whose module imports it
        // otherwise, and cloudpickle pickles a class by reference only when its module is in
        // `sys.modules` (see `pyspark.sql.connect.session`).
        py.import("pyspark.sql.connect.column")?;
        if let Some(branches) = branches {
            let branches = branches
                .iter()
                .map(|(condition, value)| {
                    Ok((
                        column_of(condition)?.expression().clone(),
                        column_of(value)?.expression().clone(),
                    ))
                })
                .collect::<PyResult<Vec<_>>>()?;
            let mut case_when = spark_connect::expression::CaseWhen::new(branches);
            if let Some(else_value) = else_value {
                case_when = case_when.with_else(column_of(&else_value)?.expression().clone());
            }
            // Of the origin that the CASE WHEN was pickled with, as PySpark's unpickled one keeps
            // it, and not of the call that unpickles it.
            return Ok(ColumnOut::new(Column::new(
                spark_connect::expression::Expression::new(ExpressionNode::CaseWhen(Box::new(
                    case_when,
                )))
                .with_origin(spark_connect::origin::origin_of_encoded(expression)),
            )));
        }
        use prost::Message;
        // Each the bytes, the levels and the holes of a relation; a pickle of an earlier client
        // has no holes.
        let subqueries = subqueries
            .iter()
            .flatten()
            .map(|subquery| {
                let relation = subquery.get_item(0)?.cast_into::<pyo3::types::PyBytes>()?;
                let levels: usize = subquery.get_item(1)?.extract()?;
                let holes = match subquery.get_item(2) {
                    Ok(holes) => crate::udf_command::sent_holes(Some(&holes))?,
                    Err(_) => Default::default(),
                };
                Ok((relation, levels, holes))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let holes = crate::udf_command::sent_holes(holes)?;
        // Decoded, and then built or dropped, on a stack that holds its levels.
        // The errors become Python exceptions on the caller's stack, which runs no Python code
        // on the grown one.
        let column = crate::on_grown_stack(|| {
            let relations = subqueries
                .into_iter()
                .map(|(relation, levels, holes)| {
                    spark_connect_proto::Relation::decode(relation.as_bytes())
                        .map(|relation| (relation, levels, holes))
                })
                .collect::<Result<Vec<_>, _>>()?;
            spark_connect_proto::Expression::decode(expression).map(|expression| {
                Column::sent_with_holes(expression, rendered, levels, relations, holes)
            })
        })
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
        .to_pyerr()?;
        // The errors that the pickled Column raised when a request built its plan and when its
        // `__repr__` rendered it.
        let unrenderable = repr_error.map(error_of_description).transpose()?;
        if let Some((plan, _)) = build_error.map(described_plan).transpose()?.flatten() {
            // Built at each request, of the session of the request, as PySpark's Column builds the
            // plan objects of its subqueries.
            let (before, owner) = crate::errors::unpickled_plan_error(plan.py(), &plan)?;
            return Ok(ColumnOut::new(Column::new(
                spark_connect::expression::Expression::unbuildable_owned(
                    column.expression().clone(),
                    before,
                    unrenderable,
                    owner,
                ),
            )));
        }
        let before = build_error.map(error_of_description).transpose()?;
        Ok(ColumnOut::new(Column::new(
            spark_connect::expression::Expression::unbuildable(
                column.expression().clone(),
                before,
                None,
                unrenderable,
            ),
        )))
    }

    /// `column` of the origin of `expression`, the bytes of the expression of a pickled Column, as
    /// PySpark's unpickled expression keeps the origin that it was pickled with: of a Column that
    /// the unpickle makes again of its parts, a sort order of the method of its direction, which
    /// takes the origin of its call otherwise (`_column_reduce.column_of`). Not a stable API.
    #[staticmethod]
    fn _with_origin_of(column: &Bound<'_, PyColumn>, expression: &[u8]) -> PyResult<ColumnOut> {
        Ok(ColumnOut::new(Column::new(
            column_of(column)?
                .expression()
                .with_origin(spark_connect::origin::origin_of_encoded(expression)),
        )))
    }

    /// As PySpark's Column, which has no `__str__` of its own for `_with_origin` to wrap: no
    /// origin, so no read of the debugging conf; the `repr` of the Column, of its class's
    /// `__repr__`, as `object.__str__` gives it.
    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(slf.repr()?.to_cow()?.into_owned())
    }

    // --- reverse arithmetic/logical dunders (operand order swapped) ---
    fn __rtruediv__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: Py<PyAny>,
    ) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__rtruediv__")?;
        let o = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(o.div(this.get()?.clone())))
    }
    /// Reverse division, of the name of Python 2, as PySpark's Column has it: as `__rtruediv__`.
    fn __rdiv__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__rdiv__")?;
        let o = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(o.div(this.get()?.clone())))
    }
    fn __rmod__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__rmod__")?;
        let o = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(o.modulo(this.get()?.clone())))
    }
    fn __rand__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__rand__")?;
        let o = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().and(o)))
    }
    fn __ror__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__ror__")?;
        let o = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().or(o)))
    }

    /// `col ** other`. Mirrors `Column.__pow__` = `power(self, other)`. PySpark's takes no
    /// modulo, so the ternary `pow(col, other, modulo)` raises the TypeError of its call.
    fn __pow__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: Py<PyAny>,
        modulo: Py<PyAny>,
    ) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__pow__")?;
        if !modulo.is_none(py) {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Column.__pow__() takes 2 positional arguments but 3 were given",
            ));
        }
        let o = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(spark_connect::functions::pow(
            this.get()?.clone(),
            o,
        )))
    }

    /// `other ** col`. Mirrors `Column.__rpow__` = `power(other, self)`. The ternary
    /// `pow(other, col, modulo)` is not implemented, so that Python raises its TypeError of the
    /// operand types, as for PySpark's, which takes no modulo.
    fn __rpow__(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: Py<PyAny>,
        modulo: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("__rpow__")?;
        if !modulo.is_none(py) {
            return Ok(py.NotImplemented());
        }
        let o = operand_column(other.bind(py))?;
        let column = ColumnOut::new(spark_connect::functions::pow(o, this.get()?.clone()));
        Ok(column.into_pyobject(py)?.into_any().unbind())
    }

    // --- guard-rails: mirror the reference Column, which raises helpful errors rather
    // than silently doing the wrong thing for these Python protocol hooks. ---

    /// `x in col` is not supported. Mirrors `Column.__contains__`.
    fn __contains__(&self, py: Python<'_>, _item: Py<PyAny>) -> PyResult<()> {
        let _origin = crate::origin::with_origin("__contains__")?;
        Err(raise_pyspark(
            py,
            "PySparkValueError",
            "CANNOT_APPLY_IN_FOR_COLUMN",
            &[],
        ))
    }

    /// A Column is not iterable. Mirrors `Column.__iter__`.
    fn __iter__(&self, py: Python<'_>) -> PyResult<()> {
        Err(raise_pyspark(
            py,
            "PySparkTypeError",
            "NOT_ITERABLE",
            &[("objectName", "Column")],
        ))
    }

    /// `bool(col)` / `if col:` is not supported. Mirrors `Column.__bool__`/`__nonzero__`.
    fn __bool__(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<bool> {
        let this = Expr::of(slf);
        // As PySpark's, of the repr of the expression.
        Err(raise_pyspark(
            py,
            "PySparkValueError",
            "CANNOT_CONVERT_COLUMN_INTO_BOOL",
            &[("column", &this.get()?.expression().render())],
        ))
    }

    /// `__bool__` of the name of Python 2, as PySpark's Column has it.
    fn __nonzero__(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<bool> {
        Self::__bool__(slf, py)
    }

    /// The arguments of `__new__` of this column, as PySpark Connect's: its `_expr`.
    fn __getnewargs__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let expression = expression_attr(slf.as_any())?;
        PyTuple::new(slf.py(), [expression])
    }

    // --- named Column methods (PySpark camelCase) ---
    #[doc = connect_doc!(Column.between)]
    fn between<'py>(
        slf: &Bound<'py, Self>,
        lowerBound: &Bound<'py, PyAny>,
        upperBound: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // As PySpark's `(self >= lowerBound) & (self <= upperBound)`, of Python's operators: each
        // operator and its operand are of the origin of their own dunder, which the origin of
        // `between` does not outlast.
        let _origin = crate::origin::with_origin("between")?;
        let lower = slf
            .as_any()
            .rich_compare(lowerBound, pyo3::basic::CompareOp::Ge)?;
        let upper = slf
            .as_any()
            .rich_compare(upperBound, pyo3::basic::CompareOp::Le)?;
        lower.bitand(upper)
    }
    #[pyo3(name = "bitwiseAND")]
    #[doc = connect_doc!(Column.bitwiseAND)]
    fn bitwise_and(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("bitwiseAND")?;
        let o = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().bitwise_and(o)))
    }
    #[pyo3(name = "bitwiseOR")]
    #[doc = connect_doc!(Column.bitwiseOR)]
    fn bitwise_or(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("bitwiseOR")?;
        let o = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().bitwise_or(o)))
    }
    #[pyo3(name = "bitwiseXOR")]
    #[doc = connect_doc!(Column.bitwiseXOR)]
    fn bitwise_xor(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("bitwiseXOR")?;
        let o = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().bitwise_xor(o)))
    }
    #[doc = connect_doc!(Column.ilike)]
    fn ilike(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("ilike")?;
        binary_function("ilike", this.get()?, other)
    }
    #[pyo3(name = "isNaN")]
    #[doc = connect_doc!(Column.isNaN)]
    fn is_nan(slf: &Bound<'_, Self>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("isNaN")?;
        Ok(ColumnOut::new(this.get()?.clone().is_nan()))
    }
    #[pyo3(name = "eqNullSafe")]
    #[doc = connect_doc!(Column.eqNullSafe)]
    fn eq_null_safe(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        other: Py<PyAny>,
    ) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("eqNullSafe")?;
        let o = operand_column(other.bind(py))?;
        Ok(ColumnOut::new(this.get()?.clone().eq_null_safe(o)))
    }
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(Column.isin)]
    fn isin(slf: &Bound<'_, Self>, py: Python<'_>, cols: Vec<Py<PyAny>>) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("isin")?;
        // As pyspark's, a single DataFrame makes an IN subquery of it.
        if let [only] = cols.as_slice() {
            if let Ok(df) = only.bind(py).cast::<crate::dataframe::PyDataFrame>() {
                let df = crate::dataframe::Frame::of(df)?;
                return Ok(ColumnOut::new(
                    this.get()?.clone().isin_subquery(&df.dataframe),
                ));
            }
        }
        // pyspark's Column.isin unpacks a single list or set argument, but not a tuple, which
        // is a value (UNSUPPORTED_DATA_TYPE of lit): col.isin([1, 2]) == col.isin(1, 2).
        // pandas-on-Spark's Series.isin relies on this (it passes a single list of lit(..)
        // columns).
        let mut items: Vec<Py<PyAny>> = cols;
        if items.len() == 1 {
            let only = items[0].bind(py);
            if only.is_instance_of::<pyo3::types::PyList>()
                || only.is_instance_of::<pyo3::types::PySet>()
            {
                items = only
                    .try_iter()?
                    .map(|r| r.map(|b| b.unbind()))
                    .collect::<PyResult<Vec<_>>>()?;
            }
        }
        let mut vals = Vec::with_capacity(items.len());
        for c in &items {
            vals.push(to_value_column(c.bind(py))?);
        }
        Ok(ColumnOut::new(this.get()?.clone().isin(vals)))
    }
    #[pyo3(name = "withField")]
    #[doc = connect_doc!(Column.withField)]
    fn with_field(
        slf: &Bound<'_, Self>,
        fieldName: &Bound<'_, PyAny>,
        col: &Bound<'_, PyAny>,
    ) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("withField")?;
        // As PySpark's, of a str and a Column.
        if !fieldName.is_instance_of::<pyo3::types::PyString>() {
            return Err(not_expected_type(fieldName, "fieldName", "str"));
        }
        let Ok(value) = col.cast::<PyColumn>() else {
            return Err(not_expected_type(col, "col", "Column"));
        };
        // Its `WithField` sets the name when it builds the plan, after the struct.
        let mut args = crate::plan_args::PlanArgs::new();
        let field_name = args.str(fieldName)?;
        Ok(ColumnOut::new(
            args.column_after(
                this.get()?
                    .clone()
                    .with_field(&field_name, column_of(value)?),
            ),
        ))
    }
    #[pyo3(name = "dropFields", signature = (*fieldNames))]
    #[doc = connect_doc!(Column.dropFields)]
    fn drop_fields(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        fieldNames: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<ColumnOut> {
        let this = Expr::of(slf);
        let _origin = crate::origin::with_origin("dropFields")?;
        // As PySpark's, of strings, at least one, which each `DropField` sets when it builds the
        // plan, after the struct.
        let mut args = crate::plan_args::PlanArgs::new();
        let mut names = Vec::with_capacity(fieldNames.len());
        for name in &fieldNames {
            if !name.is_instance_of::<pyo3::types::PyString>() {
                return Err(not_expected_type(name, "fieldName", "str"));
            }
            names.push(args.str(name)?);
        }
        if names.is_empty() {
            return Err(raise_pyspark(
                py,
                "PySparkValueError",
                "CANNOT_BE_EMPTY",
                &[("item", "dropFields")],
            ));
        }
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        Ok(ColumnOut::new(
            args.column_after(this.get()?.clone().drop_fields(refs)),
        ))
    }
}

/// The cast of `column` to `data_type`, a DataType or a str, as PySpark's `CastExpression`, whose
/// `to_plan` sets the type after its child, when it builds the plan: a str as the unparsed type,
/// of which one that UTF-8 cannot encode fails the requests of the plan with its
/// `UnicodeEncodeError`, and a DataType as the type object, which the cast keeps and reads as
/// PySpark's does, of the object as it is then and of the errors of the reads then: its repr shows
/// `simpleString().upper()` of it, and each request sends `pyspark_types_to_proto_types` of it (see
/// `crate::types::py_to_proto_data_type`), so a DataType that PySpark's conversion does not
/// convert, such as `AtomicType()` or a struct with a `StructField` as a type in it, fails the
/// requests with its `UNSUPPORTED_OPERATION`, and a struct changed in place after the call is
/// shown and sent as it is then. The call reads nothing of the object, as PySpark's. A `try_cast`
/// for `try_`.
fn cast_column(column: &Column, data_type: &Bound<'_, PyAny>, try_: bool) -> PyResult<ColumnOut> {
    if data_type.is_instance_of::<pyo3::types::PyString>() {
        let mut args = crate::plan_args::PlanArgs::new();
        let text = args.str(data_type)?;
        let cast = match try_ {
            false => column.clone().cast_str(&text),
            true => column.clone().try_cast_str(&text),
        };
        return Ok(ColumnOut::new(args.column_after(cast)));
    }
    let type_reader = crate::udf_command::data_type_pickler(data_type.clone().unbind());
    let shown_reader = crate::udf_command::shown_reader(data_type.clone().unbind());
    // The null type stands for the type of a reader that gives none, which no request sends: a
    // request raises the error of the reader, and the repr the error of its text.
    Ok(ColumnOut::new(column.clone().cast_read(
        spark_connect::types::DataType::Null,
        type_reader,
        shown_reader,
        try_,
    )))
}

/// The description of `error`, of an expression that a request does not build or whose
/// `__repr__` fails, as a pickled Column keeps it: `(kind, text, payload)`, of which
/// `pyspark.sql.connect._column_reduce.column_of` raises the error again, also on a Python worker
/// of the server, where it raises it as PySpark's Column of the expression would.
fn error_description<'py>(
    py: Python<'py>,
    error: &spark_connect_core::error::SparkError,
) -> PyResult<Bound<'py, PyTuple>> {
    described_error(py, error)?.ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "cannot pickle a Column whose plan fails with {}",
            error.message
        ))
    })
}

/// The kinds of the errors of PySpark's error classes that a plan or an expression carries, as
/// [`crate::errors::pyspark_class_error`] gives them, by the names of their classes in
/// `pyspark.errors`, which a pickle of the error keeps.
const PYSPARK_ERROR_CLASSES: [(spark_connect_core::error::SparkErrorKind, &str); 10] = {
    use spark_connect_core::error::SparkErrorKind as K;
    [
        (K::ValueError, "PySparkValueError"),
        (K::TypeError, "PySparkTypeError"),
        (K::IndexError, "PySparkIndexError"),
        (K::AttributeError, "PySparkAttributeError"),
        (K::KeyError, "PySparkKeyError"),
        (K::RuntimeError, "PySparkRuntimeError"),
        (K::NotImplementedError, "PySparkNotImplementedError"),
        (K::AssertionError, "PySparkAssertionError"),
        (K::PicklingError, "PySparkPicklingError"),
        (K::ImportError, "PySparkImportError"),
    ]
};

/// The description of `error`, of a plan or an expression that PySpark's client fails to build,
/// as [`error_description`] gives it, of which `pyspark.sql.connect._column_reduce.raise_error`
/// raises the error again: a builtin `TypeError`, `ValueError`, `OverflowError` or
/// `AttributeError` of its message, the error of UTF-8 of its value, or an error of one of
/// PySpark's error classes, of its error class and parameters; and of the error that a plan of
/// the Python client raised when it was built, which a plan keeps, the plan itself
/// ([`plan_description`]), of an object over the DataFrame of the plan. None for the error of
/// another kind, which no pickle keeps.
pub(crate) fn described_error<'py>(
    py: Python<'py>,
    error: &spark_connect_core::error::SparkError,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    described_error_of(py, error, true)
}

/// As [`described_error`], of an object over the DataFrame of the plan of a kept error when
/// `derived`, and else of the DataFrame of the plan itself.
pub(crate) fn described_error_of<'py>(
    py: Python<'py>,
    error: &spark_connect_core::error::SparkError,
    derived: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    use pyo3::types::PyBytes;
    use pyo3::IntoPyObjectExt;
    use spark_connect_core::error::SparkErrorKind as K;
    if let Some(plan) = crate::errors::kept_plan(py, error) {
        return plan_description(&plan, derived).map(Some);
    }
    let none = || py.None().into_bound(py);
    let (kind, text, payload) = match error.kind {
        K::BuiltinTypeError => ("TypeError", error.message.clone(), none()),
        K::BuiltinValueError => ("ValueError", error.message.clone(), none()),
        K::BuiltinOverflowError => ("OverflowError", error.message.clone(), none()),
        K::BuiltinAttributeError => ("AttributeError", error.message.clone(), none()),
        // The str that UTF-8 does not encode, and the bytes that it does not decode.
        K::UnicodeEncode => {
            let utf32le = error.unencodable_value().unwrap_or_default();
            let text = PyBytes::new(py, &utf32le)
                .call_method1("decode", ("utf-32-le", "surrogatepass"))?;
            ("UnicodeEncodeError", String::new(), text)
        }
        K::UnicodeDecode => {
            let bytes = error.invalid_utf8_value().unwrap_or_default();
            (
                "UnicodeDecodeError",
                String::new(),
                PyBytes::new(py, &bytes).into_any(),
            )
        }
        kind => match PYSPARK_ERROR_CLASSES.iter().find(|(of, _)| *of == kind) {
            Some((_, class)) if !error.error_class.is_empty() => (
                *class,
                error.error_class.clone(),
                error.params.clone().into_bound_py_any(py)?,
            ),
            _ => return Ok(None),
        },
    };
    PyTuple::new(
        py,
        [
            kind.into_bound_py_any(py)?,
            text.into_bound_py_any(py)?,
            payload,
        ],
    )
    .map(Some)
}

/// The description of the error of `plan`, a plan of the Python client whose build raised, which
/// a plan keeps ([`crate::errors::kept_error`]): `("plan", "derived" or "", plan)`, as PySpark's
/// pickle carries its plan objects, whose unpickled object builds the plan again and raises a new
/// error of its own, also on a Python worker of the server, of the class, the arguments, the
/// state and the chain that the build gives then, as PySpark's does. The pickler pickles `plan`
/// with the rest of the object, so that its memo of the objects that it pickled takes a plan that
/// reaches the object back, as through an error that holds it. Of an object over the DataFrame of
/// the plan (`derived`), such as a DataFrame over it or a Column of its subquery, whose own plan
/// the pickle does not carry.
fn plan_description<'py>(plan: &Bound<'py, PyAny>, derived: bool) -> PyResult<Bound<'py, PyTuple>> {
    use pyo3::IntoPyObjectExt;
    let py = plan.py();
    PyTuple::new(
        py,
        [
            "plan".into_bound_py_any(py)?,
            if derived { "derived" } else { "" }.into_bound_py_any(py)?,
            plan.clone(),
        ],
    )
}

/// The plan of the Python client of `description`, of [`plan_description`], and whether the
/// object is over the DataFrame of the plan, if it is the description of one.
pub(crate) fn described_plan<'py>(
    description: &Bound<'py, PyAny>,
) -> PyResult<Option<(Bound<'py, PyAny>, bool)>> {
    let (kind, text, payload): (String, String, Bound<'py, PyAny>) = description.extract()?;
    Ok((kind == "plan").then_some((payload, text == "derived")))
}

/// The error of `description`, of [`error_description`].
pub(crate) fn error_of_description(
    description: &Bound<'_, PyAny>,
) -> PyResult<spark_connect_core::error::SparkError> {
    use spark_connect_core::error::{SparkError, SparkErrorKind as K};
    let (kind, text, payload): (String, String, Bound<'_, PyAny>) = description.extract()?;
    let builtin = |kind| SparkError {
        kind,
        ..SparkError::value_msg(text.clone())
    };
    Ok(match kind.as_str() {
        "TypeError" => builtin(K::BuiltinTypeError),
        "ValueError" => builtin(K::BuiltinValueError),
        "OverflowError" => builtin(K::BuiltinOverflowError),
        "AttributeError" => builtin(K::BuiltinAttributeError),
        "UnicodeEncodeError" | "UnicodeDecodeError" => {
            crate::dataframe::unbuildable_str_error(&payload)?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("no {kind} of the value"))
            })?
        }
        class => {
            let Some((of_class, _)) = PYSPARK_ERROR_CLASSES
                .iter()
                .find(|(_, name)| *name == class)
            else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "no error of the kind {class}"
                )));
            };
            let params: std::collections::BTreeMap<String, String> = payload.extract()?;
            let params: Vec<(&str, &str)> = params
                .iter()
                .map(|(key, value)| (key.as_str(), value.as_str()))
                .collect();
            SparkError::classed(*of_class, &text, &params)
        }
    })
}
