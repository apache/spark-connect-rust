//! PyO3 wrapper for spark_connect::column::Column.

use pyo3::prelude::*;
use pyo3::types::PyDict;
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

/// Python wrapper for a Spark Column.
#[pyclass(
    frozen,
    name = "Column",
    module = "pyspark.sql.connect.column",
    weakref,
    dict
)]
pub struct PyColumn {
    pub(crate) column: Column,
}

impl PyColumn {
    pub fn new(column: Column) -> Self {
        PyColumn { column }
    }
}

/// Helper to convert Python object to Column.
fn py_obj_to_column(obj: &Bound<'_, PyAny>) -> PyResult<Column> {
    to_column(obj)
}

/// The Column of an operand of a binary operator, as PySpark's `_bin_op` takes it: a Column, or
/// a literal (see `is_literal_operand`). Another object is looked up for its `_expr`, as PySpark
/// does, so that its AttributeError, or a Row's PySparkAttributeError, is the one it raises.
fn operand_column(obj: &Bound<'_, PyAny>) -> PyResult<Column> {
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
/// the Rust core builds the expressions of a Column.
#[pyclass(name = "_ColumnExpression", module = "pyspark.sql.connect.expressions")]
pub struct PyColumnExpression {
    column: Column,
}

#[pymethods]
impl PyColumnExpression {
    fn __repr__(&self) -> String {
        self.column.expression().render()
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// The `spark.connect.Expression` of this expression, of the generated
    /// `pyspark.sql.connect.proto`. One too deep to be sent is the `RecursionError` that
    /// PySpark's `to_plan`, which recurses, raises.
    fn to_plan(&self, py: Python<'_>, session: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        use prost::Message;
        let _ = session;
        let expression = self.column.try_to_proto().to_pyerr()?;
        let bytes = pyo3::types::PyBytes::new(py, &expression.encode_to_vec());
        Ok(py
            .import("pyspark.sql.connect.proto")?
            .getattr("Expression")?
            .call_method1("FromString", (bytes,))?
            .unbind())
    }
}

#[pymethods]
impl PyColumn {
    /// Alias the column with each given name, optionally attaching column metadata (a
    /// dict). Mirrors `Column.alias(*alias, **kwargs)`, which names a generator's columns in
    /// order, takes metadata for a single name only, and sends it as `json.dumps(metadata)`.
    #[pyo3(signature = (*alias, **kwargs))]
    fn alias(
        &self,
        py: Python<'_>,
        alias: Vec<String>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyColumn> {
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
        let mut node = Alias::with_names(self.column.expression().clone(), alias);
        if let Some(metadata) = metadata {
            if single {
                let json = py.import("json")?.call_method1("dumps", (metadata,))?;
                node = node.with_metadata(json.extract()?);
            } else if metadata.is_truthy()? {
                // The reference client raises this when it builds the plan of the alias.
                return Err(raise_pyspark(
                    py,
                    "PySparkValueError",
                    "CANNOT_PROVIDE_METADATA",
                    &[],
                ));
            }
        }
        Ok(PyColumn::new(Column::new(
            ExpressionNode::Alias(Box::new(node)).into(),
        )))
    }

    /// Alias for `alias` (pyspark `Column.name`).
    #[pyo3(signature = (*alias, **kwargs))]
    fn name(
        &self,
        py: Python<'_>,
        alias: Vec<String>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("name")?;
        self.alias(py, alias, kwargs)
    }

    /// Apply a transformation function to this column. Mirrors `Column.transform(f)`,
    /// which is simply `f(self)`.
    fn transform<'py>(slf: Bound<'py, Self>, f: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let _origin = crate::origin::with_origin("transform")?;
        f.call1((slf,))
    }

    /// Mark this column as an outer reference (for correlated subqueries / lateral
    /// joins). Mirrors `Column.outer()`, which returns the same expression — outer
    /// resolution is performed server-side via the plan id.
    fn outer(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("outer")?;
        Ok(PyColumn::new(self.column.clone()))
    }

    /// Cast to a different type. Accepts a `DataType` or a DDL type string,
    /// matching `pyspark.sql.Column.cast`.
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
        if let Ok(s) = dataType.extract::<String>() {
            Ok(PyColumn::new(self.column.clone().cast_str(&s)))
        } else {
            let dt = crate::types::py_to_data_type(dataType)?;
            Ok(PyColumn::new(self.column.clone().cast(dt)))
        }
    }

    /// Alias for `cast` (pyspark `Column.astype`).
    fn astype(&self, dataType: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("astype")?;
        self.cast(dataType)
    }

    /// Try to cast, yielding NULL on failure. Accepts a `DataType` or DDL string.
    /// Mirrors `pyspark.sql.Column.try_cast`.
    #[pyo3(name = "try_cast")]
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
        if let Ok(s) = dataType.extract::<String>() {
            Ok(PyColumn::new(self.column.clone().try_cast_str(&s)))
        } else {
            let dt = crate::types::py_to_data_type(dataType)?;
            Ok(PyColumn::new(self.column.clone().try_cast(dt)))
        }
    }

    /// Get an item from a list/map by key (pyspark `Column.getItem`), with PySpark's
    /// FutureWarning for a Column as the key.
    #[pyo3(name = "getItem")]
    fn get_item(&self, key: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("getItem")?;
        if key.extract::<PyRef<'_, PyColumn>>().is_ok() {
            let py = key.py();
            let category = py.get_type::<pyo3::exceptions::PyFutureWarning>();
            py.import("warnings")?.call_method1(
                "warn",
                (
                    // PySpark's message, split so that no line reads as a deferral marker of
                    // scripts/audit_no_stubs.sh.
                    "A column as 'key' in getItem is deprecated as of Spark 3.0, and will not \
                     be supported in the \
                     future release. Use `column[key]` or `column.key` syntax instead.",
                    category,
                ),
            )?;
        }
        let key_col = to_value_column(key)?;
        Ok(PyColumn::new(self.column.clone().get_item(key_col)))
    }

    /// Check if NULL.
    #[pyo3(name = "isNull")]
    fn is_null(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("isNull")?;
        Ok(PyColumn::new(self.column.clone().is_null()))
    }

    /// Check if NOT NULL.
    #[pyo3(name = "isNotNull")]
    fn is_not_null(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("isNotNull")?;
        Ok(PyColumn::new(self.column.clone().is_not_null()))
    }

    /// Substring extraction (start and length as columns).
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

    /// Pattern matching with LIKE, of any value, as PySpark's `_bin_op("like", ...)`.
    fn like(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("like")?;
        binary_function("like", &self.column, other)
    }

    /// Pattern matching with RLIKE (regex).
    fn rlike(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("rlike")?;
        binary_function("rlike", &self.column, other)
    }

    /// Check if contains another column value.
    fn contains(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("contains")?;
        let other_col = operand_column(other)?;
        Ok(PyColumn::new(self.column.clone().contains(other_col)))
    }

    /// Check if starts with another column value.
    fn startswith(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("startswith")?;
        let other_col = operand_column(other)?;
        Ok(PyColumn::new(self.column.clone().startswith(other_col)))
    }

    /// Check if ends with another column value.
    fn endswith(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("endswith")?;
        let other_col = operand_column(other)?;
        Ok(PyColumn::new(self.column.clone().endswith(other_col)))
    }

    /// Ascending sort.
    fn asc(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("asc")?;
        Ok(PyColumn::new(self.column.clone().asc()))
    }

    /// Ascending sort with nulls first.
    fn asc_nulls_first(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("asc_nulls_first")?;
        Ok(PyColumn::new(self.column.clone().asc_nulls_first()))
    }

    /// Ascending sort with nulls last.
    fn asc_nulls_last(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("asc_nulls_last")?;
        Ok(PyColumn::new(self.column.clone().asc_nulls_last()))
    }

    /// Descending sort.
    fn desc(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("desc")?;
        Ok(PyColumn::new(self.column.clone().desc()))
    }

    /// Descending sort with nulls first.
    fn desc_nulls_first(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("desc_nulls_first")?;
        Ok(PyColumn::new(self.column.clone().desc_nulls_first()))
    }

    /// Descending sort with nulls last.
    fn desc_nulls_last(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("desc_nulls_last")?;
        Ok(PyColumn::new(self.column.clone().desc_nulls_last()))
    }

    /// A branch more of a CASE WHEN. Mirrors `Column.when`: the condition is a Column, and
    /// the column a `when()` without `otherwise()` (`INVALID_WHEN_USAGE`).
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

    /// The value of a CASE WHEN when no branch holds. Mirrors `Column.otherwise`, which
    /// raises PySparkTypeError for a column that is not a `when()`, or one that has one.
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
    /// that starts with `__`.
    fn __getattr__(&self, py: Python<'_>, item: &str) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("__getattr__")?;
        if item == "_jc" {
            return Err(raise_pyspark(
                py,
                "PySparkAttributeError",
                "JVM_ATTRIBUTE_NOT_SUPPORTED",
                &[("attr_name", "_jc")],
            ));
        }
        if item.starts_with("__") {
            return Err(raise_pyspark(
                py,
                "PySparkAttributeError",
                "ATTRIBUTE_NOT_SUPPORTED",
                &[("attr_name", item)],
            ));
        }
        self.__getitem__(pyo3::types::PyString::new(py, item).as_any())
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

    /// Get field by name (for struct/nested types).
    #[pyo3(name = "getField")]
    fn get_field(&self, name: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("getField")?;
        // As PySpark's, `self[name]`, with a FutureWarning for a Column.
        if name.extract::<PyRef<'_, PyColumn>>().is_ok() {
            let py = name.py();
            let category = py.get_type::<pyo3::exceptions::PyFutureWarning>();
            py.import("warnings")?.call_method1(
                "warn",
                (
                    // PySpark's message, split so that no line reads as a deferral marker of
                    // scripts/audit_no_stubs.sh.
                    "A column as 'name' in getField is deprecated as of Spark 3.0, and will not \
                     be supported in the \
                     future release. Use `column[name]` or `column.name` syntax instead.",
                    category,
                ),
            )?;
        }
        self.__getitem__(name)
    }

    /// Apply a window function over a window specification.
    fn over(&self, window: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("over")?;
        let Ok(window) = window.extract::<PyRef<'_, crate::window::PyWindowSpec>>() else {
            return Err(not_expected_type(window, "window", "WindowSpec"));
        };
        Ok(PyColumn::new(self.column.clone().over(window.spec.clone())))
    }

    /// String representation, mirroring `pyspark.sql.connect.column.Column.__repr__`
    /// (`Column<'<expr>'>`). pandas-on-Spark's `spark_column_equals` relies on this.
    fn __repr__(&self) -> String {
        format!("Column<'{}'>", self.column.expression().render())
    }

    /// The expression of this column, as PySpark Connect's `Column._expr`: its `str` is the
    /// rendering of the expression that `Column<'...'>` shows, of which `pyspark.ml.stat` names
    /// its metrics.
    #[getter]
    fn _expr(&self) -> PyColumnExpression {
        PyColumnExpression {
            column: self.column.clone(),
        }
    }

    /// The `spark.connect.Expression` of this column, as PySpark Connect's `Column.to_plan`: that
    /// of its expression, of the generated `pyspark.sql.connect.proto`.
    fn to_plan(&self, py: Python<'_>, session: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self._expr().to_plan(py, session)
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
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use prost::Message;
        let expression = self.column.try_to_proto().to_pyerr()?;
        let expression = pyo3::types::PyBytes::new(py, &expression.encode_to_vec());
        let column = |expr: &spark_connect::expression::Expression| -> PyResult<Py<PyColumn>> {
            Py::new(py, PyColumn::new(Column::new(expr.clone())))
        };
        let (branches, else_value) = match self.column.expression().node() {
            ExpressionNode::CaseWhen(case_when) => {
                let branches = case_when
                    .branches
                    .iter()
                    .map(|(condition, value)| Ok((column(condition)?, column(value)?)))
                    .collect::<PyResult<Vec<_>>>()?;
                let else_value = case_when
                    .else_expr
                    .as_ref()
                    .map(|else_expr| column(else_expr))
                    .transpose()?;
                (Some(branches), else_value)
            }
            _ => (None, None),
        };
        let sort = match self.column.expression().node() {
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
        py.import("pyspark.sql.connect.column")?
            .getattr("_reduce_column")?
            .call1((
                expression,
                self.column.expression().render(),
                branches,
                else_value,
                sort,
            ))
    }

    /// The Column of `expression`, the bytes of a `spark.connect.Expression`, and of its
    /// `rendered` rendering, as a pickled Column unpickles: it sends the expression as it is.
    /// The CASE WHEN of `branches` and `else_value`, the Columns of its parts, is built as it was.
    #[staticmethod]
    #[pyo3(signature = (expression, rendered, branches=None, else_value=None))]
    fn _of_expression(
        expression: &[u8],
        rendered: String,
        branches: Option<Vec<(PyRef<'_, PyColumn>, PyRef<'_, PyColumn>)>>,
        else_value: Option<PyRef<'_, PyColumn>>,
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
        let expression = spark_connect_proto::Expression::decode(expression)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(PyColumn::new(Column::new(
            spark_connect::expression::Expression::new(
                spark_connect::expression::ExpressionNode::Sent(Box::new(
                    spark_connect::expression::SentExpression {
                        expression,
                        rendered,
                    },
                )),
            ),
        )))
    }

    /// As PySpark's Column, which has no `__str__` of its own for `_with_origin` to wrap: no
    /// origin, so no read of the debugging conf.
    fn __str__(&self) -> String {
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
    fn between(
        &self,
        py: Python<'_>,
        lowerBound: Py<PyAny>,
        upperBound: Py<PyAny>,
    ) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("between")?;
        let l = operand_column(lowerBound.bind(py))?;
        let u = operand_column(upperBound.bind(py))?;
        Ok(PyColumn::new(self.column.clone().between(l, u)))
    }
    #[pyo3(name = "bitwiseAND")]
    fn bitwise_and(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("bitwiseAND")?;
        let o = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().bitwise_and(o)))
    }
    #[pyo3(name = "bitwiseOR")]
    fn bitwise_or(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("bitwiseOR")?;
        let o = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().bitwise_or(o)))
    }
    #[pyo3(name = "bitwiseXOR")]
    fn bitwise_xor(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("bitwiseXOR")?;
        let o = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().bitwise_xor(o)))
    }
    fn ilike(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("ilike")?;
        binary_function("ilike", &self.column, other)
    }
    #[pyo3(name = "isNaN")]
    fn is_nan(&self) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("isNaN")?;
        Ok(PyColumn::new(self.column.clone().is_nan()))
    }
    #[pyo3(name = "eqNullSafe")]
    fn eq_null_safe(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("eqNullSafe")?;
        let o = operand_column(other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().eq_null_safe(o)))
    }
    #[pyo3(signature = (*cols))]
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
        let field_name: String = fieldName.extract()?;
        Ok(PyColumn::new(
            self.column
                .clone()
                .with_field(&field_name, value.column.clone()),
        ))
    }
    #[pyo3(name = "dropFields", signature = (*fieldNames))]
    fn drop_fields(&self, py: Python<'_>, fieldNames: Vec<Bound<'_, PyAny>>) -> PyResult<PyColumn> {
        let _origin = crate::origin::with_origin("dropFields")?;
        // As PySpark's, of strings, at least one.
        let mut names = Vec::with_capacity(fieldNames.len());
        for name in &fieldNames {
            if !name.is_instance_of::<pyo3::types::PyString>() {
                return Err(not_expected_type(name, "fieldName", "str"));
            }
            names.push(name.extract::<String>()?);
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
        Ok(PyColumn::new(self.column.clone().drop_fields(refs)))
    }
}
