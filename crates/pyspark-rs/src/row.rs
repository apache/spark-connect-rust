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
pub fn value_to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    // Propagate conversion failures as Python exceptions with `?` rather than
    // panicking across the FFI boundary (the conversions are near-infallible,
    // but a panic here would unwind into CPython).
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        Value::Byte(n) => (*n as i64).into_pyobject(py)?.into_any(),
        Value::Short(n) => (*n as i64).into_pyobject(py)?.into_any(),
        Value::Integer(n) => (*n as i64).into_pyobject(py)?.into_any(),
        Value::Long(n) => (*n).into_pyobject(py)?.into_any(),
        Value::Float(f) => (*f as f64).into_pyobject(py)?.into_any(),
        Value::Double(f) => (*f).into_pyobject(py)?.into_any(),
        Value::String(s) => s.as_str().into_pyobject(py)?.into_any(),
        Value::Binary(b) => PyBytes::new(py, b).into_any(),
        Value::Date(d) => {
            // Convert days-since-epoch (i32) to datetime.date
            let date_mod = py.import("datetime")?;
            let date_cls = date_mod.getattr("date")?;
            // 719163 is the ordinal of 1970-01-01
            let ordinal = (*d as i64) + 719163i64;
            date_cls.call_method1("fromordinal", (ordinal,))?.into_any()
        }
        Value::Timestamp(t) => {
            // Convert microseconds-since-epoch (i64) to datetime.datetime
            let datetime_mod = py.import("datetime")?;
            let datetime_cls = datetime_mod.getattr("datetime")?;
            let seconds = *t as f64 / 1_000_000.0;
            datetime_cls
                .call_method1("fromtimestamp", (seconds,))?
                .into_any()
        }
        Value::Decimal { value, .. } => {
            // Convert to Python decimal.Decimal by instantiating the class
            let decimal_mod = py.import("decimal")?;
            let decimal_cls = decimal_mod.getattr("Decimal")?;
            decimal_cls.call1((value.as_str(),))?.into_any()
        }
        Value::List(items) => {
            let list = PyList::empty(py);
            for it in items {
                list.append(value_to_py(py, it)?)?;
            }
            list.into_any()
        }
        Value::Map(m) => {
            let dict = PyDict::new(py);
            for (k, val) in m {
                dict.set_item(k, value_to_py(py, val)?)?;
            }
            dict.into_any()
        }
        Value::Struct(fields) => {
            let dict = PyDict::new(py);
            for (k, val) in fields {
                dict.set_item(k, value_to_py(py, val)?)?;
            }
            dict.into_any()
        }
        Value::Variant { value, metadata } => {
            // A VARIANT materializes as a VariantVal (toJson/toPython decode lazily),
            // matching pyspark, rather than a raw {value, metadata} dict.
            Py::new(
                py,
                crate::values::PyVariantVal::from_parts(value.clone(), metadata.clone()),
            )?
            .into_bound(py)
            .into_any()
        }
    })
}

#[pymethods]
impl PyRow {
    /// Construct a Row, mirroring `pyspark.sql.Row`:
    /// `Row(name=value, ...)` yields named fields; `Row(v1, v2, ...)` yields
    /// positional fields named `_1`, `_2`, ....
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn __new__(
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if let Some(kw) = kwargs {
            if !args.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Row cannot mix positional and keyword arguments",
                ));
            }
            let mut fields = Vec::with_capacity(kw.len());
            let mut values = Vec::with_capacity(kw.len());
            for (k, v) in kw.iter() {
                fields.push(k.extract::<String>()?);
                values.push(crate::session::py_to_value(&v)?);
            }
            Ok(PyRow::new(Row::new(fields, values)))
        } else {
            let mut values = Vec::with_capacity(args.len());
            for v in args.iter() {
                values.push(crate::session::py_to_value(&v)?);
            }
            let fields = (0..values.len()).map(|i| format!("_{}", i + 1)).collect();
            Ok(PyRow::new(Row::new(fields, values)))
        }
    }

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
            return value_to_py(py, self.row.get(i as usize).unwrap());
        }
        if let Ok(name) = key.extract::<String>() {
            return match self.row.get_by_name(&name) {
                Some(v) => value_to_py(py, v),
                None => Err(PyKeyError::new_err(name)),
            };
        }
        Err(PyKeyError::new_err("Row key must be int or str"))
    }

    /// Attribute access by field name (`row.id`).
    fn __getattr__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        match self.row.get_by_name(name) {
            Some(v) => value_to_py(py, v),
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
    fn as_dict<'py>(&self, py: Python<'py>, recursive: bool) -> PyResult<Bound<'py, PyDict>> {
        let _ = recursive; // nested structs are already converted to dicts by value_to_py
        let dict = PyDict::new(py);
        for (name, val) in self.row.fields().iter().zip(self.row.values()) {
            dict.set_item(name, value_to_py(py, val)?)?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("{}", self.row)
    }

    fn __eq__(&self, other: &PyRow) -> bool {
        self.row == other.row
    }

    /// Number of occurrences of `value` among the row's values (tuple.count).
    fn count(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut n = 0;
        for v in self.row.values() {
            if value_to_py(py, v)?.eq(value)? {
                n += 1;
            }
        }
        Ok(n)
    }

    /// Index of the first occurrence of `value` (tuple.index); raises ValueError if
    /// not found. `start`/`stop` bound the search, with Python's negative-index rules.
    #[pyo3(signature = (value, start=0, stop=None))]
    fn index(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
        start: isize,
        stop: Option<isize>,
    ) -> PyResult<usize> {
        let len = self.row.len() as isize;
        let norm = |i: isize| -> isize {
            if i < 0 {
                (len + i).max(0)
            } else {
                i.min(len)
            }
        };
        let lo = norm(start);
        let hi = stop.map(norm).unwrap_or(len);
        let mut i = lo;
        while i < hi {
            if value_to_py(py, self.row.get(i as usize).unwrap())?.eq(value)? {
                return Ok(i as usize);
            }
            i += 1;
        }
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "tuple.index(x): x not in tuple",
        ))
    }
}
