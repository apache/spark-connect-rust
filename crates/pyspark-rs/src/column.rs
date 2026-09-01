//! PyO3 wrapper for spark_connect::column::Column.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_connect::column::Column;

use crate::functions::to_column;

/// Raise a `pyspark.errors` exception (`class_name`) carrying the given `error_class`
/// condition and message parameters, mirroring the reference Column guard-rails.
fn raise_pyspark(
    py: Python<'_>,
    class_name: &str,
    error_class: &str,
    params: &[(&str, &str)],
) -> PyErr {
    use pyo3::types::PyDict;
    let build = || -> PyResult<PyErr> {
        let cls = py.import("pyspark.errors")?.getattr(class_name)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("errorClass", error_class)?;
        let mp = PyDict::new(py);
        for (k, v) in params {
            mp.set_item(*k, *v)?;
        }
        kwargs.set_item("messageParameters", mp)?;
        let exc = cls.call((), Some(&kwargs))?;
        Ok(PyErr::from_value(exc))
    };
    build().unwrap_or_else(|e| e)
}

/// Python wrapper for a Spark Column.
#[pyclass(name = "Column", module = "pyspark.sql.connect.column")]
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

#[pymethods]
impl PyColumn {
    /// Alias the column, optionally attaching column metadata (a dict), mirroring
    /// `Column.alias(name, metadata=...)`.
    #[pyo3(signature = (*alias, **kwargs))]
    fn alias(&self, alias: Vec<String>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyColumn> {
        // Mirrors Column.alias(*alias, **kwargs): the common single-name form aliases the
        // column; `metadata` (a dict) may be passed via kwargs. Extra names beyond the first
        // (multi-output aliasing) are not applied here.
        let metadata: Option<std::collections::HashMap<String, String>> = match kwargs {
            Some(kw) => match kw.get_item("metadata")? {
                Some(v) if !v.is_none() => Some(v.extract()?),
                _ => None,
            },
            None => None,
        };
        let name = match alias.first() {
            Some(n) => n.clone(),
            None => return Ok(PyColumn::new(self.column.clone())),
        };
        Ok(match metadata {
            None => PyColumn::new(self.column.clone().alias(&name)),
            Some(m) => {
                let md: std::collections::BTreeMap<String, String> = m.into_iter().collect();
                PyColumn::new(self.column.clone().alias_with_metadata(&name, md))
            }
        })
    }

    /// Alias for `alias` (pyspark `Column.name`).
    #[pyo3(signature = (*alias, **kwargs))]
    fn name(&self, alias: Vec<String>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<PyColumn> {
        self.alias(alias, kwargs)
    }

    /// Apply a transformation function to this column. Mirrors `Column.transform(f)`,
    /// which is simply `f(self)`.
    fn transform<'py>(slf: Bound<'py, Self>, f: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        f.call1((slf,))
    }

    /// Mark this column as an outer reference (for correlated subqueries / lateral
    /// joins). Mirrors `Column.outer()`, which returns the same expression — outer
    /// resolution is performed server-side via the plan id.
    fn outer(&self) -> PyColumn {
        PyColumn::new(self.column.clone())
    }

    /// Cast to a different type. Accepts a `DataType` or a DDL type string,
    /// matching `pyspark.sql.Column.cast`.
    fn cast(&self, dataType: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
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
        self.cast(dataType)
    }

    /// Try to cast, yielding NULL on failure. Accepts a `DataType` or DDL string.
    /// Mirrors `pyspark.sql.Column.try_cast`.
    #[pyo3(name = "try_cast")]
    fn try_cast(&self, dataType: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        if let Ok(s) = dataType.extract::<String>() {
            Ok(PyColumn::new(self.column.clone().try_cast_str(&s)))
        } else {
            let dt = crate::types::py_to_data_type(dataType)?;
            Ok(PyColumn::new(self.column.clone().try_cast(dt)))
        }
    }

    /// Get an item from a list/map by key (pyspark `Column.getItem`).
    #[pyo3(name = "getItem")]
    fn get_item(&self, key: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let key_col = py_obj_to_column(key)?;
        Ok(PyColumn::new(self.column.clone().get_item(key_col)))
    }

    /// Check if NULL.
    #[pyo3(name = "isNull")]
    fn is_null(&self) -> PyColumn {
        PyColumn::new(self.column.clone().is_null())
    }

    /// Check if NOT NULL.
    #[pyo3(name = "isNotNull")]
    fn is_not_null(&self) -> PyColumn {
        PyColumn::new(self.column.clone().is_not_null())
    }

    /// Substring extraction (start and length as columns).
    fn substr(&self, startPos: &Bound<'_, PyAny>, length: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let start_col = py_obj_to_column(startPos)?;
        let length_col = py_obj_to_column(length)?;
        Ok(PyColumn::new(
            self.column.clone().substr(start_col, length_col),
        ))
    }

    /// Pattern matching with LIKE.
    fn like(&self, other: &str) -> PyColumn {
        PyColumn::new(self.column.clone().like(other))
    }

    /// Pattern matching with RLIKE (regex).
    fn rlike(&self, other: &str) -> PyColumn {
        PyColumn::new(self.column.clone().rlike(other))
    }

    /// Check if contains another column value.
    fn contains(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let other_col = py_obj_to_column(other)?;
        Ok(PyColumn::new(self.column.clone().contains(other_col)))
    }

    /// Check if starts with another column value.
    fn startswith(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let other_col = py_obj_to_column(other)?;
        Ok(PyColumn::new(self.column.clone().startswith(other_col)))
    }

    /// Check if ends with another column value.
    fn endswith(&self, other: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let other_col = py_obj_to_column(other)?;
        Ok(PyColumn::new(self.column.clone().endswith(other_col)))
    }

    /// Ascending sort.
    fn asc(&self) -> PyColumn {
        PyColumn::new(self.column.clone().asc())
    }

    /// Ascending sort with nulls first.
    fn asc_nulls_first(&self) -> PyColumn {
        PyColumn::new(self.column.clone().asc_nulls_first())
    }

    /// Ascending sort with nulls last.
    fn asc_nulls_last(&self) -> PyColumn {
        PyColumn::new(self.column.clone().asc_nulls_last())
    }

    /// Descending sort.
    fn desc(&self) -> PyColumn {
        PyColumn::new(self.column.clone().desc())
    }

    /// Descending sort with nulls first.
    fn desc_nulls_first(&self) -> PyColumn {
        PyColumn::new(self.column.clone().desc_nulls_first())
    }

    /// Descending sort with nulls last.
    fn desc_nulls_last(&self) -> PyColumn {
        PyColumn::new(self.column.clone().desc_nulls_last())
    }

    /// When condition for CASE statements.
    fn when(&self, condition: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let cond_col = py_obj_to_column(condition)?;
        let val_col = py_obj_to_column(value)?;
        Ok(PyColumn::new(self.column.clone().when(cond_col, val_col)))
    }

    /// Otherwise for CASE statements.
    fn otherwise(&self, value: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let val_col = py_obj_to_column(value)?;
        Ok(PyColumn::new(self.column.clone().otherwise(val_col)))
    }

    /// Get item by key/index.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let key_col = py_obj_to_column(key)?;
        Ok(PyColumn::new(self.column.clone().get_item(key_col)))
    }

    /// Equality (==) - returns a Column condition
    fn __eq__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().eq(other_col)))
    }

    /// Inequality (!=) - returns a Column condition
    fn __ne__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().ne(other_col)))
    }

    /// Greater than (>) - returns a Column condition
    fn __gt__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().gt(other_col)))
    }

    /// Less than (<) - returns a Column condition
    fn __lt__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().lt(other_col)))
    }

    /// Greater than or equal (>=) - returns a Column condition
    fn __ge__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().ge(other_col)))
    }

    /// Less than or equal (<=) - returns a Column condition
    fn __le__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().le(other_col)))
    }

    // Arithmetic operators
    /// Addition (+)
    fn __add__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().add(other_col)))
    }

    /// Right addition (for reverse operations)
    fn __radd__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(other_col.add(self.column.clone())))
    }

    /// Subtraction (-)
    fn __sub__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().sub(other_col)))
    }

    /// Right subtraction (for reverse operations)
    fn __rsub__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(other_col.sub(self.column.clone())))
    }

    /// Multiplication (*)
    fn __mul__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().mul(other_col)))
    }

    /// Right multiplication (for reverse operations)
    fn __rmul__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(other_col.mul(self.column.clone())))
    }

    /// Division (/)
    fn __truediv__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().div(other_col)))
    }

    /// Modulo (%)
    fn __mod__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().modulo(other_col)))
    }

    // Bitwise operators (logical in Spark context)
    /// Bitwise AND (&) - logical AND in Spark
    fn __and__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().and(other_col)))
    }

    /// Bitwise OR (|) - logical OR in Spark
    fn __or__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let other_col = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().or(other_col)))
    }

    /// Bitwise NOT (~) - logical NOT in Spark
    fn __invert__(&self) -> PyColumn {
        PyColumn::new(self.column.clone().not())
    }

    /// Negation (-)
    fn __neg__(&self) -> PyColumn {
        PyColumn::new(self.column.clone().neg())
    }

    /// Get field by name (for struct/nested types).
    #[pyo3(name = "getField")]
    fn get_field(&self, name: &str) -> PyColumn {
        PyColumn::new(self.column.clone().get_field(name))
    }

    /// Apply a window function over a window specification.
    fn over(&self, window: &crate::window::PyWindowSpec) -> PyColumn {
        PyColumn::new(self.column.clone().over(window.spec.clone()))
    }

    /// String representation, mirroring `pyspark.sql.connect.column.Column.__repr__`
    /// (`Column<'<expr>'>`). pandas-on-Spark's `spark_column_equals` relies on this.
    fn __repr__(&self) -> String {
        format!("Column<'{}'>", self.column.expression().render())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    // --- reverse arithmetic/logical dunders (operand order swapped) ---
    fn __rtruediv__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let o = to_column(&other.bind(py))?;
        Ok(PyColumn::new(o.div(self.column.clone())))
    }
    fn __rmod__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let o = to_column(&other.bind(py))?;
        Ok(PyColumn::new(o.modulo(self.column.clone())))
    }
    fn __rand__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let o = to_column(&other.bind(py))?;
        Ok(PyColumn::new(o.and(self.column.clone())))
    }
    fn __ror__(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let o = to_column(&other.bind(py))?;
        Ok(PyColumn::new(o.or(self.column.clone())))
    }

    /// `col ** other`. Mirrors `Column.__pow__` = `power(self, other)`.
    /// (`modulo` is part of Python's ternary pow protocol; Spark has no 3-arg pow.)
    fn __pow__(&self, py: Python<'_>, other: Py<PyAny>, _modulo: Py<PyAny>) -> PyResult<PyColumn> {
        let o = to_column(&other.bind(py))?;
        Ok(PyColumn::new(spark_connect::functions::pow(
            self.column.clone(),
            o,
        )))
    }

    /// `other ** col`. Mirrors `Column.__rpow__` = `power(other, self)`.
    fn __rpow__(&self, py: Python<'_>, other: Py<PyAny>, _modulo: Py<PyAny>) -> PyResult<PyColumn> {
        let o = to_column(&other.bind(py))?;
        Ok(PyColumn::new(spark_connect::functions::pow(
            o,
            self.column.clone(),
        )))
    }

    // --- guard-rails: mirror the reference Column, which raises helpful errors rather
    // than silently doing the wrong thing for these Python protocol hooks. ---

    /// `x in col` is not supported. Mirrors `Column.__contains__`.
    fn __contains__(&self, py: Python<'_>, _item: Py<PyAny>) -> PyResult<()> {
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
        Err(raise_pyspark(
            py,
            "PySparkValueError",
            "CANNOT_CONVERT_COLUMN_INTO_BOOL",
            &[],
        ))
    }

    // --- named Column methods (PySpark camelCase) ---
    fn between(
        &self,
        py: Python<'_>,
        lowerBound: Py<PyAny>,
        upperBound: Py<PyAny>,
    ) -> PyResult<PyColumn> {
        let l = to_column(&lowerBound.bind(py))?;
        let u = to_column(&upperBound.bind(py))?;
        Ok(PyColumn::new(self.column.clone().between(l, u)))
    }
    #[pyo3(name = "bitwiseAND")]
    fn bitwise_and(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let o = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().bitwise_and(o)))
    }
    #[pyo3(name = "bitwiseOR")]
    fn bitwise_or(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let o = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().bitwise_or(o)))
    }
    #[pyo3(name = "bitwiseXOR")]
    fn bitwise_xor(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let o = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().bitwise_xor(o)))
    }
    fn ilike(&self, other: &str) -> PyColumn {
        PyColumn::new(self.column.clone().ilike(other))
    }
    #[pyo3(name = "isNaN")]
    fn is_nan(&self) -> PyColumn {
        PyColumn::new(self.column.clone().is_nan())
    }
    #[pyo3(name = "eqNullSafe")]
    fn eq_null_safe(&self, py: Python<'_>, other: Py<PyAny>) -> PyResult<PyColumn> {
        let o = to_column(&other.bind(py))?;
        Ok(PyColumn::new(self.column.clone().eq_null_safe(o)))
    }
    #[pyo3(signature = (*cols))]
    fn isin(&self, py: Python<'_>, cols: Vec<Py<PyAny>>) -> PyResult<PyColumn> {
        // pyspark's Column.isin unpacks a single list/tuple/set argument:
        // col.isin([1, 2]) == col.isin(1, 2). pandas-on-Spark's Series.isin relies on
        // this (it passes a single list of lit(..) columns).
        let mut items: Vec<Py<PyAny>> = cols;
        if items.len() == 1 {
            let only = items[0].bind(py);
            if only.is_instance_of::<pyo3::types::PyList>()
                || only.is_instance_of::<pyo3::types::PyTuple>()
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
            vals.push(to_column(&c.bind(py))?);
        }
        Ok(PyColumn::new(self.column.clone().isin(vals)))
    }
    #[pyo3(name = "withField")]
    fn with_field(&self, py: Python<'_>, fieldName: &str, col: Py<PyAny>) -> PyResult<PyColumn> {
        let v = to_column(&col.bind(py))?;
        Ok(PyColumn::new(self.column.clone().with_field(fieldName, v)))
    }
    #[pyo3(name = "dropFields", signature = (*fieldNames))]
    fn drop_fields(&self, fieldNames: Vec<String>) -> PyColumn {
        let refs: Vec<&str> = fieldNames.iter().map(|s| s.as_str()).collect();
        PyColumn::new(self.column.clone().drop_fields(refs))
    }
}
