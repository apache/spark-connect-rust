//! PyO3 wrapper for spark_connect::column::Column.

use pyo3::prelude::*;
use spark_connect::column::Column;

use crate::functions::to_column;

/// Python wrapper for a Spark Column.
#[pyclass(name = "Column")]
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
    /// Alias the column.
    fn alias(&self, name: &str) -> PyColumn {
        PyColumn::new(self.column.clone().alias(name))
    }

    /// Cast to a different type (using string DDL).
    fn cast(&self, type_name: &str) -> PyColumn {
        PyColumn::new(self.column.clone().cast_str(type_name))
    }

    /// Check if NULL.
    fn is_null(&self) -> PyColumn {
        PyColumn::new(self.column.clone().is_null())
    }

    /// Check if NOT NULL.
    fn is_not_null(&self) -> PyColumn {
        PyColumn::new(self.column.clone().is_not_null())
    }

    /// Substring extraction (start and length as columns).
    fn substr(&self, start: &Bound<'_, PyAny>, length: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        let start_col = py_obj_to_column(start)?;
        let length_col = py_obj_to_column(length)?;
        Ok(PyColumn::new(
            self.column.clone().substr(start_col, length_col),
        ))
    }

    /// Pattern matching with LIKE.
    fn like(&self, pattern: &str) -> PyColumn {
        PyColumn::new(self.column.clone().like(pattern))
    }

    /// Pattern matching with RLIKE (regex).
    fn rlike(&self, pattern: &str) -> PyColumn {
        PyColumn::new(self.column.clone().rlike(pattern))
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
    fn get_field(&self, name: &str) -> PyColumn {
        PyColumn::new(self.column.clone().get_field(name))
    }

    /// Apply a window function over a window specification.
    fn over(&self, window_spec: &crate::window::PyWindowSpec) -> PyColumn {
        PyColumn::new(self.column.clone().over(window_spec.spec.clone()))
    }

    /// String representation.
    fn __repr__(&self) -> String {
        "Column()".to_string()
    }

    fn __str__(&self) -> String {
        "Column()".to_string()
    }
}
