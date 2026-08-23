//! PyO3 wrapper for spark_connect::row::Row, mirroring `pyspark.sql.Row`.

use pyo3::exceptions::{PyAttributeError, PyIndexError, PyKeyError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use spark_connect::row::{Row, Value};

/// Python wrapper for a Spark Row.
#[pyclass(name = "Row")]
pub struct PyRow {
    pub(crate) row: Row,
}

impl PyRow {
    pub fn new(row: Row) -> Self {
        PyRow { row }
    }
}

/// Convert a Spark `Value` into a Python object (PyO3 0.28 `IntoPyObject`).
pub fn value_to_py<'py>(py: Python<'py>, v: &Value) -> Bound<'py, PyAny> {
    match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => b.into_pyobject(py).unwrap().to_owned().into_any(),
        Value::Byte(n) => (*n as i64).into_pyobject(py).unwrap().into_any(),
        Value::Short(n) => (*n as i64).into_pyobject(py).unwrap().into_any(),
        Value::Integer(n) => (*n as i64).into_pyobject(py).unwrap().into_any(),
        Value::Long(n) => (*n).into_pyobject(py).unwrap().into_any(),
        Value::Float(f) => (*f as f64).into_pyobject(py).unwrap().into_any(),
        Value::Double(f) => (*f).into_pyobject(py).unwrap().into_any(),
        Value::String(s) => s.as_str().into_pyobject(py).unwrap().into_any(),
        Value::Binary(b) => PyBytes::new(py, b).into_any(),
        // Date/Timestamp are surfaced as their raw integer encodings (days /
        // microseconds since epoch); pyspark-typed datetime objects are a
        // separate conversion layer.
        Value::Date(d) => (*d as i64).into_pyobject(py).unwrap().into_any(),
        Value::Timestamp(t) => (*t).into_pyobject(py).unwrap().into_any(),
        Value::List(items) => {
            let list = PyList::empty(py);
            for it in items {
                let _ = list.append(value_to_py(py, it));
            }
            list.into_any()
        }
        Value::Map(m) => {
            let dict = PyDict::new(py);
            for (k, val) in m {
                let _ = dict.set_item(k, value_to_py(py, val));
            }
            dict.into_any()
        }
        Value::Struct(fields) => {
            let dict = PyDict::new(py);
            for (k, val) in fields {
                let _ = dict.set_item(k, value_to_py(py, val));
            }
            dict.into_any()
        }
    }
}

#[pymethods]
impl PyRow {
    /// Number of fields.
    fn __len__(&self) -> usize {
        self.row.len()
    }

    /// Index (int, supports negative) or field-name (str) access. Raises
    /// IndexError / KeyError like pyspark so the Row is a proper sequence.
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(mut i) = key.extract::<isize>() {
            let n = self.row.len() as isize;
            if i < 0 {
                i += n;
            }
            if i < 0 || i >= n {
                return Err(PyIndexError::new_err("Row index out of range"));
            }
            return Ok(value_to_py(py, self.row.get(i as usize).unwrap()));
        }
        if let Ok(name) = key.extract::<String>() {
            return match self.row.get_by_name(&name) {
                Some(v) => Ok(value_to_py(py, v)),
                None => Err(PyKeyError::new_err(name)),
            };
        }
        Err(PyKeyError::new_err("Row key must be int or str"))
    }

    /// Attribute access by field name (`row.id`).
    fn __getattr__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        match self.row.get_by_name(name) {
            Some(v) => Ok(value_to_py(py, v)),
            None => Err(PyAttributeError::new_err(name.to_string())),
        }
    }

    /// Field names, in order.
    fn __fields__(&self) -> Vec<String> {
        self.row.fields().to_vec()
    }

    /// As a dict of field name -> value (mirrors `Row.asDict`).
    #[pyo3(name = "asDict")]
    #[pyo3(signature = (recursive=false))]
    fn as_dict<'py>(&self, py: Python<'py>, recursive: bool) -> Bound<'py, PyDict> {
        let _ = recursive; // nested structs are already converted to dicts by value_to_py
        let dict = PyDict::new(py);
        for (name, val) in self.row.fields().iter().zip(self.row.values()) {
            let _ = dict.set_item(name, value_to_py(py, val));
        }
        dict
    }

    fn __repr__(&self) -> String {
        format!("{}", self.row)
    }

    fn __eq__(&self, other: &PyRow) -> bool {
        self.row == other.row
    }
}
