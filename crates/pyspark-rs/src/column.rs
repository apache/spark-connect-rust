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
fn binary_function(name: &str, column: &Column, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
    let other = operand_column(other)?;
    Ok(PyColumn::new(Column::new(
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
pub(crate) fn column_reference_name(column: PyRef<'_, PyColumn>) -> Option<String> {
    match column.column.expression().node() {
        spark_connect::expression::ExpressionNode::ColumnReference(reference) => {
            Some(reference.name.clone())
        }
        _ => None,
    }
}

// Python wrapper for a Spark Column.
#[doc = connect_doc!(Column)]
#[pyclass(
    frozen,
    name = "Column",
    module = "pyspark.sql.connect.column",
    weakref,
    dict
)]
pub struct PyColumn {
    pub(crate) column: Column,
    /// Set once the value is written, so that `__traverse__` reads nothing else of an object that
    /// the garbage collector traverses while it is created (see [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

impl PyColumn {
    pub fn new(column: Column) -> Self {
        PyColumn {
            column,
            gc_ready: crate::gc::GcReady::new(),
        }
    }
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
/// argument.
#[pyclass(name = "_ColumnExpression", module = "pyspark.sql.connect.expressions")]
pub struct PyColumnExpression {
    column: Column,
    /// Set once the value is written, as `__traverse__` reads `column` (see
    /// [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

impl PyColumnExpression {
    /// The expression of `column`.
    pub(crate) fn new(column: Column) -> Self {
        PyColumnExpression {
            column,
            gc_ready: crate::gc::GcReady::new(),
        }
    }
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
        let copies = crate::udf_command::deep_copied_picklers(py, expression.picklers(), memo)?;
        Ok(PyColumnExpression::new(if copies.is_empty() {
            self.column.clone()
        } else {
            Column::new(expression.with_pickler_copies(&copies))
        }))
    }

    /// As PySpark's expression, a Python object whose fields Python's garbage collector sees: the
    /// functions of its UDFs, also of the plan of a subquery, that only this expression keeps
    /// alive are reported, as those of a Column, so that a reference cycle through a function and
    /// the expression is collected, as PySpark collects it.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        crate::udf_command::visit_pickler_objects(
            self.column.expression().exclusive_pickler_objects(),
            &visit,
        )
    }

    /// Break a reference cycle that the garbage collector found unreachable: the expression is
    /// replaced by one that holds nothing, which drops the objects that `__traverse__` reported.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            self.column = Column::new(spark_connect::expression::Expression::column(""));
        }
    }

    /// An expression pickles, as PySpark's expression, a Python object, pickles its fields: as the
    /// `_expr` of its Column, `getattr(column, "_expr")`, of the pickle of the Column (see
    /// `PyColumn::__reduce__`), so that the unpickled expression is that of the unpickled Column,
    /// also on a Python worker of the server, where it is the expression of PySpark Connect's
    /// Column.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let column = Py::new(py, PyColumn::new(self.column.clone()))?;
        let args = PyTuple::new(
            py,
            [
                column.into_any().into_bound(py),
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
    /// raises.
    fn to_plan(&self, py: Python<'_>, session: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let _ = session;
        expression_message(py, &self.column)
    }
}

/// `Column.__getattr__(column, item)` called as a function of the class: the `__getattr__` of the
/// Column, which PySpark's attribute access of a name that a Column does not have calls.
#[pyfunction]
fn _column_getattr(
    column: PyRef<'_, PyColumn>,
    item: &Bound<'_, pyo3::types::PyString>,
) -> PyResult<PyColumn> {
    column.__getattr__(column.py(), item)
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
    // Alias the column with each given name, optionally attaching column metadata (a
    // dict). Mirrors `Column.alias(*alias, **kwargs)`, which names a generator's columns in
    // order, takes metadata for a single name only, and sends it as `json.dumps(metadata)`.
    #[pyo3(signature = (*alias, **kwargs))]
    #[doc = connect_doc!(Column.alias)]
    pub(crate) fn alias(
        &self,
        py: Python<'_>,
        alias: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyColumn> {
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
        let mut node = Alias::with_names(self.column.expression().clone(), names);
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
        Ok(PyColumn::new(Column::new(
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
        &self,
        py: Python<'_>,
        alias: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("name")?;
        self.alias(py, alias, kwargs)
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
    fn outer(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("outer")?;
        Ok(PyColumn::new(self.column.clone()))
    }

    // Cast to a different type. Accepts a `DataType` or a DDL type string,
    // matching `pyspark.sql.Column.cast`.
    #[doc = connect_doc!(Column.cast)]
    fn cast(&self, dataType: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("cast")?;
        // As PySpark's, of a DataType or a str.
        if !(dataType.is_instance_of::<pyo3::types::PyString>()
            || dataType.is_instance_of::<crate::types::PyDataType>())
        {
            return Err(not_expected_type(dataType, "dataType", "DataType or str"));
        }
        // A DDL string keeps the unparsed type_str form (reference cast("int"));
        // any DataType object is converted to a typed cast.
        cast_column(&self.column, dataType, false)
    }

    // Alias for `cast` (pyspark `Column.astype`).
    fn astype(&self, dataType: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("astype")?;
        self.cast(dataType)
    }

    // Try to cast, yielding NULL on failure. Accepts a `DataType` or DDL string.
    // Mirrors `pyspark.sql.Column.try_cast`.
    #[pyo3(name = "try_cast")]
    #[doc = connect_doc!(Column.try_cast)]
    fn try_cast(&self, dataType: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
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
        cast_column(&self.column, dataType, true)
    }

    // Get an item from a list/map by key (pyspark `Column.getItem`), with PySpark's
    // FutureWarning for a Column as the key, of the line of its `warnings.warn` in
    // `pyspark.sql.connect.column` (see `warn_as_pyspark`).
    #[pyo3(name = "getItem")]
    #[doc = connect_doc!(Column.getItem)]
    fn get_item(&self, key: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("getItem")?;
        if key.extract::<PyRef<'_, PyColumn>>().is_ok() {
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
        self.__getitem__(key)
    }

    // Check if NULL.
    #[pyo3(name = "isNull")]
    #[doc = connect_doc!(Column.isNull)]
    fn is_null(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("isNull")?;
        Ok(PyColumn::new(self.column.clone().is_null()))
    }

    // Check if NOT NULL.
    #[pyo3(name = "isNotNull")]
    #[doc = connect_doc!(Column.isNotNull)]
    fn is_not_null(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("isNotNull")?;
        Ok(PyColumn::new(self.column.clone().is_not_null()))
    }

    // Substring extraction (start and length as columns).
    #[doc = connect_doc!(Column.substr)]
    fn substr(&self, startPos: &Bound<'_, PyAny>, length: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
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
        Ok(PyColumn::new(
            self.column.clone().substr(start_col, length_col),
        ))
    }

    // Pattern matching with LIKE, of any value, as PySpark's `_bin_op("like", ...)`.
    #[doc = connect_doc!(Column.like)]
    fn like(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("like")?;
        binary_function("like", &self.column, other)
    }

    // Pattern matching with RLIKE (regex).
    #[doc = connect_doc!(Column.rlike)]
    fn rlike(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("rlike")?;
        binary_function("rlike", &self.column, other)
    }

    // Check if contains another column value.
    #[doc = connect_doc!(Column.contains)]
    fn contains(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("contains")?;
        let other_col = operand_column(other)?;
        Ok(PyColumn::new(self.column.clone().contains(other_col)))
    }

    // Check if starts with another column value.
    #[doc = connect_doc!(Column.startswith)]
    fn startswith(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("startswith")?;
        let other_col = operand_column(other)?;
        Ok(PyColumn::new(self.column.clone().startswith(other_col)))
    }

    // Check if ends with another column value.
    #[doc = connect_doc!(Column.endswith)]
    fn endswith(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("endswith")?;
        let other_col = operand_column(other)?;
        Ok(PyColumn::new(self.column.clone().endswith(other_col)))
    }

    // Ascending sort.
    #[doc = connect_doc!(Column.asc)]
    fn asc(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("asc")?;
        // As PySpark's `asc`, which is its `asc_nulls_first`, of the origin of that.
        self.asc_nulls_first()
    }

    // Ascending sort with nulls first.
    #[doc = connect_doc!(Column.asc_nulls_first)]
    fn asc_nulls_first(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("asc_nulls_first")?;
        Ok(PyColumn::new(self.column.clone().asc_nulls_first()))
    }

    // Ascending sort with nulls last.
    #[doc = connect_doc!(Column.asc_nulls_last)]
    fn asc_nulls_last(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("asc_nulls_last")?;
        Ok(PyColumn::new(self.column.clone().asc_nulls_last()))
    }

    // Descending sort.
    #[doc = connect_doc!(Column.desc)]
    fn desc(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("desc")?;
        // As PySpark's `desc`, which is its `desc_nulls_last`, of the origin of that.
        self.desc_nulls_last()
    }

    // Descending sort with nulls first.
    #[doc = connect_doc!(Column.desc_nulls_first)]
    fn desc_nulls_first(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("desc_nulls_first")?;
        Ok(PyColumn::new(self.column.clone().desc_nulls_first()))
    }

    // Descending sort with nulls last.
    #[doc = connect_doc!(Column.desc_nulls_last)]
    fn desc_nulls_last(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("desc_nulls_last")?;
        Ok(PyColumn::new(self.column.clone().desc_nulls_last()))
    }

    // A branch more of a CASE WHEN. Mirrors `Column.when`: the condition is a Column, and
    // the column a `when()` without `otherwise()` (`INVALID_WHEN_USAGE`).
    #[doc = connect_doc!(Column.when)]
    fn when(&self, condition: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("when")?;
        let py = condition.py();
        let Ok(condition) = condition.extract::<PyRef<'_, PyColumn>>() else {
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
        match self.column.expression().node() {
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
        Ok(PyColumn::new(
            self.column.clone().when(condition.column.clone(), val_col),
        ))
    }

    // The value of a CASE WHEN when no branch holds. Mirrors `Column.otherwise`, which
    // raises PySparkTypeError for a column that is not a `when()`, or one that has one.
    #[doc = connect_doc!(Column.otherwise)]
    fn otherwise(&self, value: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("otherwise")?;
        let message = match self.column.expression().node() {
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
        Ok(PyColumn::new(self.column.clone().otherwise(val_col)))
    }

    /// Get item by key/index, or a slice as `substr(start, stop)`. Mirrors
    /// `Column.__getitem__`.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
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
            return self.substr(&key.getattr("start")?, &key.getattr("stop")?);
        }
        let key_col = to_value_column(key)?;
        Ok(PyColumn::new(self.column.clone().get_item(key_col)))
    }

    /// An attribute that a Column does not have is the field of that name, as
    /// `col["name"]`. Mirrors `Column.__getattr__`, which raises for `_jc` and for a name
    /// that starts with `__`. The name is the str of the attribute as it is: one that UTF-8
    /// cannot encode is the field of a literal that fails the plans of the Column, as `col[name]`.
    fn __getattr__(
        &self,
        py: Python<'_>,
        item: &Bound<'_, pyo3::types::PyString>,
    ) -> PyResult<PyColumn> {
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
        self.__getitem__(item.as_any())
    }

    /// Equality (==) - returns a Column condition
    fn __eq__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__eq__")?;
        let other_col = equality_operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().eq(other_col)))
    }

    /// Inequality (!=) - returns a Column condition
    fn __ne__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__ne__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().ne(other_col)))
    }

    /// Greater than (>) - returns a Column condition
    fn __gt__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__gt__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().gt(other_col)))
    }

    /// Less than (<) - returns a Column condition
    fn __lt__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__lt__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().lt(other_col)))
    }

    /// Greater than or equal (>=) - returns a Column condition
    fn __ge__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__ge__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().ge(other_col)))
    }

    /// Less than or equal (<=) - returns a Column condition
    fn __le__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__le__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().le(other_col)))
    }

    // Arithmetic operators
    /// Addition (+)
    fn __add__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__add__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().add(other_col)))
    }

    /// Right addition (for reverse operations)
    fn __radd__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__radd__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(other_col.add(self.column.clone())))
    }

    /// Subtraction (-)
    fn __sub__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__sub__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().sub(other_col)))
    }

    /// Right subtraction (for reverse operations)
    fn __rsub__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__rsub__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(other_col.sub(self.column.clone())))
    }

    /// Multiplication (*)
    fn __mul__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__mul__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().mul(other_col)))
    }

    /// Right multiplication (for reverse operations)
    fn __rmul__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__rmul__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(other_col.mul(self.column.clone())))
    }

    /// Division (/)
    fn __truediv__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__truediv__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().div(other_col)))
    }

    /// Division (/), of the name of Python 2, as PySpark's Column has it, which pandas-on-Spark
    /// calls: as `__truediv__`.
    fn __div__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__div__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().div(other_col)))
    }

    /// Modulo (%)
    fn __mod__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__mod__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().modulo(other_col)))
    }

    // Bitwise operators (logical in Spark context)
    /// Bitwise AND (&) - logical AND in Spark
    fn __and__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__and__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().and(other_col)))
    }

    /// Bitwise OR (|) - logical OR in Spark
    fn __or__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__or__")?;
        let other_col = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().or(other_col)))
    }

    /// Bitwise NOT (~) - logical NOT in Spark
    fn __invert__(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__invert__")?;
        Ok(PyColumn::new(self.column.clone().not()))
    }

    /// Negation (-)
    fn __neg__(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__neg__")?;
        Ok(PyColumn::new(self.column.clone().neg()))
    }

    // Get field by name (for struct/nested types).
    #[pyo3(name = "getField")]
    #[doc = connect_doc!(Column.getField)]
    fn get_field(&self, name: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("getField")?;
        // As PySpark's, `self[name]`, with a FutureWarning for a Column, of the line of its
        // `warnings.warn` in `pyspark.sql.connect.column`.
        if name.extract::<PyRef<'_, PyColumn>>().is_ok() {
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
        self.__getitem__(name)
    }

    // Apply a window function over a window specification.
    #[doc = connect_doc!(Column.over)]
    fn over(&self, window: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("over")?;
        let Ok(window) = window.extract::<PyRef<'_, crate::window::PyWindowSpec>>() else {
            return Err(not_expected_type(window, "window", "WindowSpec"));
        };
        Ok(PyColumn::new(self.column.clone().over(window.spec.clone())))
    }

    /// String representation, mirroring `pyspark.sql.connect.column.Column.__repr__`
    /// (`Column<'<expr>'>`). pandas-on-Spark's `spark_column_equals` relies on this.
    /// Of the error of PySpark's `__repr__` of an expression that it does not render.
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Column<'{}'>",
            self.column.expression().try_render().to_pyerr()?
        ))
    }

    /// The expression of this column, as PySpark Connect's `Column._expr`: its `str` is the
    /// rendering of the expression that `Column<'...'>` shows, of which `pyspark.ml.stat` names
    /// its metrics.
    #[getter]
    fn _expr(&self) -> PyColumnExpression {
        PyColumnExpression::new(self.column.clone())
    }

    // The `spark.connect.Expression` of this column, as PySpark Connect's `Column.to_plan`: that
    // of its expression, of `pyspark.sql.connect.proto`.
    fn to_plan(&self, py: Python<'_>, session: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self._expr().to_plan(py, session)
    }

    /// As a Column of PySpark Connect, a Python object whose expression Python's garbage collector
    /// sees: the functions of its UDFs, which a request pickles when it builds a plan of the
    /// Column, as PySpark's expression keeps them, that only this Column keeps alive are reported
    /// to the collector, so that a reference cycle through a function and the Column is freed, as
    /// PySpark frees it.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        crate::udf_command::visit_pickler_objects(
            self.column.expression().exclusive_pickler_objects(),
            &visit,
        )
    }

    /// The shallow copy of a Column, as PySpark's `copy.copy`, which shares the expression of the
    /// Column and the values of its attributes: so the functions of its UDFs are pickled when a
    /// request builds a plan of the copy, as those of the Column, and not when it is copied.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let copy = Bound::new(slf.py(), PyColumn::new(slf.get().column.clone()))?;
        let state = crate::group::instance_state(slf.as_any())?;
        if !state.is_none() {
            copy.getattr("__dict__")?.call_method1("update", (state,))?;
        }
        Ok(copy)
    }

    /// The deep copy of a Column, as PySpark's `copy.deepcopy`, which deep-copies its expression,
    /// of the function of each UDF with `memo` (`deep_copied_picklers`): a function is the same
    /// object, so the copy pickles it as it is at its requests, and a callable object, a bound
    /// method or a partial is a copy; the expression is shared up to the UDFs of the copies. Then
    /// of deep copies of its attributes, with `memo`, which has the copy before them, as `copy`
    /// reconstructs an object of its arguments.
    fn __deepcopy__<'py>(
        slf: &Bound<'py, Self>,
        memo: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let column = &slf.get().column;
        let copies =
            crate::udf_command::deep_copied_picklers(py, column.expression().picklers(), memo)?;
        let column = if copies.is_empty() {
            column.clone()
        } else {
            Column::new(column.expression().with_pickler_copies(&copies))
        };
        let copy = Bound::new(py, PyColumn::new(column))?;
        let state = crate::group::instance_state(slf.as_any())?;
        if !state.is_none() {
            memo.set_item(slf.as_ptr() as usize, &copy)?;
            let state = py.import("copy")?.call_method1("deepcopy", (state, memo))?;
            copy.getattr("__dict__")?.call_method1("update", (state,))?;
        }
        Ok(copy)
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
    /// that `pickle` keeps it, as it does of PySpark's Column. The function of a UDF of the
    /// Column is pickled now, into the command of the expression, as PySpark's pickle of the
    /// Column pickles the function with it; one that does not pickle raises the error of its
    /// pickling. A Column of subqueries pickles the relations of the plans that they refer to
    /// too, of their plan IDs, after the expression, as PySpark's pickle keeps the plan objects of
    /// its subqueries: the unpickled Column sends them in the `WithRelations` of the relation that
    /// holds it, as PySpark's plan sends the plans of its subqueries.
    fn __reduce__<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let this = slf.borrow();
        // Of an expression that a request fails to build, as PySpark's Column of an alias of a
        // name that protobuf refuses: its proto of what this client builds, and the errors of
        // a request and of the `__repr__` of it, which the unpickled Column raises as it did.
        let (expression, levels, unbuildable) = crate::origin::pickling(|| {
            this.column
                .expression()
                .try_to_proto_with_levels_and_unbuildable()
        })
        .to_pyerr()?;
        if let Some(exception) = unbuildable
            .as_ref()
            .and_then(crate::errors::callback_exception)
        {
            return Err(exception);
        }
        let mut build_error = unbuildable
            .as_ref()
            .map(|error| error_description(py, error))
            .transpose()?;
        // A plan of a subquery that does not build pickles of its error, as a DataFrame does,
        // which the requests of the unpickled Column raise; one that does not pickle, or that is
        // too deep, raises its error. A CASE WHEN and a sort order are given back of their parts,
        // which pickle their subqueries.
        let mut subqueries = Vec::new();
        let of_parts = matches!(
            this.column.expression().node(),
            ExpressionNode::CaseWhen(_) | ExpressionNode::SortOrder(_)
        );
        if build_error.is_none() && !of_parts {
            match crate::origin::pickling(|| this.column.expression().subquery_relations()) {
                Ok(relations) => {
                    for (relation, levels) in relations {
                        let relation =
                            pyo3::types::PyBytes::new(py, &crate::encode_on_grown_stack(relation));
                        subqueries.push((relation, levels));
                    }
                }
                Err(error) => match described_error(py, &error)? {
                    Some(description) => build_error = Some(description),
                    None => return Err(crate::errors::spark_error_to_py_exception(error)),
                },
            }
        }
        let repr_error = this
            .column
            .expression()
            .try_render()
            .err()
            .map(|error| error_description(py, &error))
            .transpose()?;
        let expression = pyo3::types::PyBytes::new(py, &crate::encode_on_grown_stack(expression));
        let column = |expr: &spark_connect::expression::Expression| -> PyResult<Py<PyColumn>> {
            Py::new(py, PyColumn::new(Column::new(expr.clone())))
        };
        let (branches, else_value) = match this.column.expression().node() {
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
        let sort = match this.column.expression().node() {
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
                this.column.expression().render(),
                branches,
                else_value,
                sort,
                levels,
                build_error,
                repr_error,
                subqueries,
            ))?;
        crate::group::reduce_with_state(&reduce, slf.as_any())
    }

    /// The Column of `expression`, the bytes of a `spark.connect.Expression`, and of its
    /// `rendered` rendering, of its `levels`, as a pickled Column unpickles: it sends the
    /// expression as it is, and the relations of `subqueries`, the bytes of the relation of each
    /// plan that its subqueries refer to and its levels, in the `WithRelations` of the relation
    /// that holds it. The CASE WHEN of `branches` and `else_value`, the Columns of its parts, is
    /// built as it was.
    #[staticmethod]
    #[pyo3(signature = (
        expression,
        rendered,
        branches=None,
        else_value=None,
        levels=1,
        build_error=None,
        repr_error=None,
        subqueries=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn _of_expression(
        expression: &[u8],
        rendered: String,
        branches: Option<Vec<(PyRef<'_, PyColumn>, PyRef<'_, PyColumn>)>>,
        else_value: Option<PyRef<'_, PyColumn>>,
        levels: usize,
        build_error: Option<&Bound<'_, PyAny>>,
        repr_error: Option<&Bound<'_, PyAny>>,
        subqueries: Option<Vec<(Bound<'_, pyo3::types::PyBytes>, usize)>>,
    ) -> PyResult<PyColumn> {
        if let Some(branches) = branches {
            let branches = branches
                .iter()
                .map(|(condition, value)| {
                    (
                        condition.column.expression().clone(),
                        value.column.expression().clone(),
                    )
                })
                .collect();
            let mut case_when = spark_connect::expression::CaseWhen::new(branches);
            if let Some(else_value) = else_value {
                case_when = case_when.with_else(else_value.column.expression().clone());
            }
            return Ok(PyColumn::new(Column::new(
                spark_connect::expression::Expression::new(ExpressionNode::CaseWhen(Box::new(
                    case_when,
                ))),
            )));
        }
        use prost::Message;
        let subqueries: Vec<(&[u8], usize)> = subqueries
            .iter()
            .flatten()
            .map(|(relation, levels)| (relation.as_bytes(), *levels))
            .collect();
        // Decoded, and then built or dropped, on a stack that holds its levels.
        // The errors become Python exceptions on the caller's stack, which runs no Python code
        // on the grown one.
        let column = crate::on_grown_stack(|| {
            let relations = subqueries
                .iter()
                .map(|(relation, levels)| {
                    spark_connect_proto::Relation::decode(*relation)
                        .map(|relation| (relation, *levels))
                })
                .collect::<Result<Vec<_>, _>>()?;
            spark_connect_proto::Expression::decode(expression).map(|expression| {
                Column::sent_with_subqueries(expression, rendered, levels, relations)
            })
        })
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
        .to_pyerr()?;
        // The errors that the pickled Column raised when a request built its plan and when its
        // `__repr__` rendered it.
        let before = build_error.map(error_of_description).transpose()?;
        let unrenderable = repr_error.map(error_of_description).transpose()?;
        Ok(PyColumn::new(Column::new(
            spark_connect::expression::Expression::unbuildable(
                column.expression().clone(),
                before,
                None,
                unrenderable,
            ),
        )))
    }

    /// As PySpark's Column, which has no `__str__` of its own for `_with_origin` to wrap: no
    /// origin, so no read of the debugging conf.
    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }

    // --- reverse arithmetic/logical dunders (operand order swapped) ---
    fn __rtruediv__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__rtruediv__")?;
        let o = operand_column(other.bind(py))?;
        Ok(PyColumn::new(o.div(self.column.clone())))
    }
    /// Reverse division, of the name of Python 2, as PySpark's Column has it: as `__rtruediv__`.
    fn __rdiv__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__rdiv__")?;
        let o = operand_column(other.bind(py))?;
        Ok(PyColumn::new(o.div(self.column.clone())))
    }
    fn __rmod__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__rmod__")?;
        let o = operand_column(other.bind(py))?;
        Ok(PyColumn::new(o.modulo(self.column.clone())))
    }
    fn __rand__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__rand__")?;
        let o = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().and(o)))
    }
    fn __ror__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__ror__")?;
        let o = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().or(o)))
    }

    /// `col ** other`. Mirrors `Column.__pow__` = `power(self, other)`. PySpark's takes no
    /// modulo, so the ternary `pow(col, other, modulo)` raises the TypeError of its call.
    fn __pow__(&self, py: Python<'_>, other: Py<PyAny>, modulo: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__pow__")?;
        if !modulo.is_none(py) {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Column.__pow__() takes 2 positional arguments but 3 were given",
            ));
        }
        let o = operand_column(other.bind(py))?;
        Ok(PyColumn::new(spark_connect::functions::pow(
            self.column.clone(),
            o,
        )))
    }

    /// `other ** col`. Mirrors `Column.__rpow__` = `power(other, self)`. The ternary
    /// `pow(other, col, modulo)` is not implemented, so that Python raises its TypeError of the
    /// operand types, as for PySpark's, which takes no modulo.
    fn __rpow__(&self, py: Python<'_>, other: Py<PyAny>, modulo: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let _origin = crate::origin::with_origin("__rpow__")?;
        if !modulo.is_none(py) {
            return Ok(py.NotImplemented());
        }
        let o = operand_column(other.bind(py))?;
        let column = PyColumn::new(spark_connect::functions::pow(o, self.column.clone()));
        Ok(Py::new(py, column)?.into_any())
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
    fn __bool__(&self, py: Python<'_>) -> PyResult<bool> {
        // As PySpark's, of the repr of the expression.
        Err(raise_pyspark(
            py,
            "PySparkValueError",
            "CANNOT_CONVERT_COLUMN_INTO_BOOL",
            &[("column", &self.column.expression().render())],
        ))
    }

    /// `__bool__` of the name of Python 2, as PySpark's Column has it.
    fn __nonzero__(&self, py: Python<'_>) -> PyResult<bool> {
        self.__bool__(py)
    }

    /// The arguments of `__new__` of this column, as PySpark Connect's: its expression.
    fn __getnewargs__(&self) -> (PyColumnExpression,) {
        (self._expr(),)
    }

    // --- named Column methods (PySpark camelCase) ---
    #[doc = connect_doc!(Column.between)]
    fn between(
        &self,
        py: Python<'_>,
        lowerBound: Py<PyAny>,
        upperBound: Py<PyAny>,
    ) -> PyResult<PyColumn> {
        // As PySpark's `(self >= lowerBound) & (self <= upperBound)`: each operator and its operand
        // are of the origin of their own dunder, which the origin of `between` does not outlast.
        let _origin = crate::origin::with_origin("between")?;
        let lower = self.__ge__(py, lowerBound)?;
        let upper = Py::new(py, self.__le__(py, upperBound)?)?.into_any();
        lower.__and__(py, upper)
    }
    #[pyo3(name = "bitwiseAND")]
    #[doc = connect_doc!(Column.bitwiseAND)]
    fn bitwise_and(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("bitwiseAND")?;
        let o = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().bitwise_and(o)))
    }
    #[pyo3(name = "bitwiseOR")]
    #[doc = connect_doc!(Column.bitwiseOR)]
    fn bitwise_or(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("bitwiseOR")?;
        let o = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().bitwise_or(o)))
    }
    #[pyo3(name = "bitwiseXOR")]
    #[doc = connect_doc!(Column.bitwiseXOR)]
    fn bitwise_xor(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("bitwiseXOR")?;
        let o = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().bitwise_xor(o)))
    }
    #[doc = connect_doc!(Column.ilike)]
    fn ilike(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("ilike")?;
        binary_function("ilike", &self.column, other)
    }
    #[pyo3(name = "isNaN")]
    #[doc = connect_doc!(Column.isNaN)]
    fn is_nan(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("isNaN")?;
        Ok(PyColumn::new(self.column.clone().is_nan()))
    }
    #[pyo3(name = "eqNullSafe")]
    #[doc = connect_doc!(Column.eqNullSafe)]
    fn eq_null_safe(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("eqNullSafe")?;
        let o = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().eq_null_safe(o)))
    }
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(Column.isin)]
    fn isin(&self, py: Python<'_>, cols: Vec<Py<PyAny>>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("isin")?;
        // As pyspark's, a single DataFrame makes an IN subquery of it.
        if let [only] = cols.as_slice() {
            if let Ok(df) = only
                .bind(py)
                .extract::<PyRef<'_, crate::dataframe::PyDataFrame>>()
            {
                return Ok(PyColumn::new(
                    self.column.clone().isin_subquery(&df.dataframe),
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
        Ok(PyColumn::new(self.column.clone().isin(vals)))
    }
    #[pyo3(name = "withField")]
    #[doc = connect_doc!(Column.withField)]
    fn with_field(
        &self,
        fieldName: &Bound<'_, PyAny>,
        col: &Bound<'_, PyAny>,
    ) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("withField")?;
        // As PySpark's, of a str and a Column.
        if !fieldName.is_instance_of::<pyo3::types::PyString>() {
            return Err(not_expected_type(fieldName, "fieldName", "str"));
        }
        let Ok(value) = col.extract::<PyRef<'_, PyColumn>>() else {
            return Err(not_expected_type(col, "col", "Column"));
        };
        // Its `WithField` sets the name when it builds the plan, after the struct.
        let mut args = crate::plan_args::PlanArgs::new();
        let field_name = args.str(fieldName)?;
        Ok(PyColumn::new(
            args.column_after(
                self.column
                    .clone()
                    .with_field(&field_name, value.column.clone()),
            ),
        ))
    }
    #[pyo3(name = "dropFields", signature = (*fieldNames))]
    #[doc = connect_doc!(Column.dropFields)]
    fn drop_fields(&self, py: Python<'_>, fieldNames: Vec<Bound<'_, PyAny>>) -> PyResult<PyColumn> {
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
        Ok(PyColumn::new(
            args.column_after(self.column.clone().drop_fields(refs)),
        ))
    }
}

/// The cast of `column` to `data_type`, a DataType or a str, as PySpark's `CastExpression`, whose
/// `to_plan` sets the type after its child, when it builds the plan: a str as the unparsed type,
/// of which one that UTF-8 cannot encode fails the requests of the plan with its
/// `UnicodeEncodeError`, and a DataType as the type. A DataType that PySpark's
/// `pyspark_types_to_proto_types` does not convert, of none of its types
/// ([`crate::plan_args::unsupported_type_error`]) or with a `StructField` as a type in it, fails
/// them with its `UNSUPPORTED_OPERATION`, and renders as its `simpleString()` of PySpark, as
/// PySpark's `__repr__`. A `try_cast` for `try_`.
fn cast_column(column: &Column, data_type: &Bound<'_, PyAny>, try_: bool) -> PyResult<PyColumn> {
    let mut args = crate::plan_args::PlanArgs::new();
    let unparsed = |text: &str| match try_ {
        false => column.clone().cast_str(text),
        true => column.clone().try_cast_str(text),
    };
    if data_type.is_instance_of::<pyo3::types::PyString>() {
        let text = args.str(data_type)?;
        return Ok(PyColumn::new(args.column_after(unparsed(&text))));
    }
    let error = match crate::plan_args::unsupported_type_error(data_type)? {
        Some(error) => error,
        None => {
            let py = data_type.py();
            let parsed = match crate::types::py_to_data_type(data_type) {
                Ok(parsed) => parsed,
                // A type that does not lower now, such as an array of a `UserDefinedType` whose
                // `module()` is not implemented, or a struct of metadata that `json.dumps` refuses,
                // which PySpark's plan raises the error of at each request, as its repr shows its
                // `simpleString()` (see `crate::types::lenient_type`).
                Err(err) if err.is_instance_of::<pyo3::exceptions::PyException>(py) => {
                    args.raised(py, err)?;
                    let text: String = data_type.call_method0("simpleString")?.extract()?;
                    return Ok(PyColumn::new(args.column_after(unparsed(&text))));
                }
                Err(err) => return Err(err),
            };
            let Some(field) = crate::types::unsendable_field_repr(data_type, &parsed)? else {
                // As PySpark's repr, of the `simpleString()` of the type object, which the core's
                // type does not tell of a user-defined type or of a type of a Python subclass.
                return Ok(PyColumn::new(
                    match crate::types::shown_simple_string(data_type, &parsed)? {
                        Some(shown) => column.clone().cast_shown(parsed, &shown, try_),
                        None if try_ => column.clone().try_cast(parsed),
                        None => column.clone().cast(parsed),
                    },
                ));
            };
            spark_connect_core::error::SparkError::classed(
                spark_connect_core::error::SparkErrorKind::ValueError,
                "UNSUPPORTED_OPERATION",
                &[("operation", &format!("data type {field}"))],
            )
        }
    };
    args.record(error);
    // As PySpark's `simpleString`, which is the `typeName` of the type, unless its class has a
    // `simpleString` of its own, as a `StructField` has.
    let base = data_type
        .py()
        .import("pyspark.sql.types")?
        .getattr("DataType")?
        .getattr("simpleString")?;
    let simple_string = data_type.get_type().getattr("simpleString")?;
    let name = if simple_string.is(&base) {
        "typeName"
    } else {
        "simpleString"
    };
    let text: String = data_type.call_method0(name)?.extract()?;
    Ok(PyColumn::new(args.column_after(unparsed(&text))))
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
/// PySpark's error classes, of its error class and parameters. None for the error of another
/// kind, which no pickle keeps.
pub(crate) fn described_error<'py>(
    py: Python<'py>,
    error: &spark_connect_core::error::SparkError,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    use pyo3::types::PyBytes;
    use pyo3::IntoPyObjectExt;
    use spark_connect_core::error::SparkErrorKind as K;
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
