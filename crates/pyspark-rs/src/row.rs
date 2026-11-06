//! PyO3 wrapper for spark_connect::row::Row, mirroring `pyspark.sql.Row`.

use pyo3::basic::CompareOp;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyList, PySlice, PyString, PyTuple};
use spark_connect::row::{Row, Value};

use crate::column::raise_pyspark;

/// Python wrapper for a Spark Row. Behaves as `pyspark.sql.Row`, a `tuple` subclass, does,
/// though it cannot subclass `tuple` itself.
#[pyclass(name = "Row", module = "pyspark.sql.types")]
pub struct PyRow {
    data: RowData,
    /// Whether the row has field names (`__fields__`). As in PySpark, a row made of values
    /// alone, such as `Row(1, 2)` or the row class `Row("name", "age")`, has none; the rest
    /// of the client knows its fields as `_1`, `_2`, and so on.
    named: bool,
}

/// The Python functions that turn the values of the fields of collected rows whose types have
/// user-defined types into what PySpark gives for them, by deserializing those values; None
/// for the other fields.
pub(crate) type FieldConverters = std::sync::Arc<Vec<Option<Py<PyAny>>>>;

enum RowData {
    /// A row that the server returned, whose Spark values become Python values when first
    /// read, and stay those objects, as a PySpark row is a tuple of them: so the row equals
    /// itself, with a NaN or a VARIANT value, and keeps its hash.
    Spark {
        row: Row,
        values: PyOnceLock<Py<PyTuple>>,
        /// The converters of its fields, when one has a user-defined type.
        converters: Option<FieldConverters>,
    },
    /// A row made in Python, which holds its Python values as they are, as PySpark's does.
    Python {
        fields: Vec<String>,
        values: Py<PyTuple>,
    },
}

impl PyRow {
    pub fn new(row: Row) -> Self {
        Self::with_converters(row, None)
    }

    /// A row that the server returned, whose values of user-defined types `converters`
    /// deserializes when they are first read.
    pub(crate) fn with_converters(row: Row, converters: Option<FieldConverters>) -> Self {
        PyRow {
            data: RowData::Spark {
                row,
                values: PyOnceLock::new(),
                converters,
            },
            named: true,
        }
    }

    /// A row made in Python of `values`, named by `fields`.
    fn of_python(fields: Vec<String>, values: Bound<'_, PyTuple>, named: bool) -> Self {
        PyRow {
            data: RowData::Python {
                fields,
                values: values.unbind(),
            },
            named,
        }
    }

    fn fields(&self) -> &[String] {
        match &self.data {
            RowData::Spark { row, .. } => row.fields(),
            RowData::Python { fields, .. } => fields,
        }
    }

    /// The values, as the tuple that a PySpark row is.
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        match &self.data {
            RowData::Spark {
                row,
                values,
                converters,
            } => {
                let values = values.get_or_try_init(py, || -> PyResult<Py<PyTuple>> {
                    let convert = |i: usize| {
                        converters
                            .as_ref()
                            .and_then(|c| c.get(i))
                            .and_then(Option::as_ref)
                    };
                    let values = row
                        .values()
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            let value = value_to_py(py, v)?;
                            match convert(i) {
                                Some(converter) => converter.bind(py).call1((value,)),
                                None => Ok(value),
                            }
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    Ok(PyTuple::new(py, values)?.unbind())
                })?;
                Ok(values.bind(py).clone())
            }
            RowData::Python { values, .. } => Ok(values.bind(py).clone()),
        }
    }

    /// The value of the field at `index`, which is in range.
    fn value<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyAny>> {
        match &self.data {
            RowData::Spark { .. } => self.values(py)?.get_item(index),
            RowData::Python { values, .. } => values.bind(py).get_item(index),
        }
    }

    /// The value of the first field named `name`, if any.
    fn value_by_name<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self.fields().iter().position(|field| field == name) {
            Some(index) => self.value(py, index).map(Some),
            None => Ok(None),
        }
    }

    /// The field names, if the row has them (`__fields__`), and the values, as the tuple
    /// that a PySpark row is, for reading the row as PySpark's schema inference does.
    pub(crate) fn parts<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Option<&[String]>, Bound<'py, PyTuple>)> {
        Ok((self.named.then(|| self.fields()), self.values(py)?))
    }

    /// The row with Spark values, for the rest of the client, which converts the values of a
    /// row made in Python as it converts those of a tuple.
    pub(crate) fn to_row(&self, py: Python<'_>) -> PyResult<Row> {
        match &self.data {
            RowData::Spark { row, .. } => Ok(row.clone()),
            RowData::Python { fields, values } => {
                let values = values
                    .bind(py)
                    .iter()
                    .map(|v| crate::session::py_to_value(&v))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(Row::new(fields.clone(), values))
            }
        }
    }

    /// What PySpark raises for an attribute that a row does not have.
    fn no_attribute(py: Python<'_>, name: &str) -> PyErr {
        raise_pyspark(
            py,
            "PySparkAttributeError",
            "ATTRIBUTE_NOT_SUPPORTED",
            &[("attr_name", name)],
        )
    }

    /// The values of `other` as a tuple, if it is a row or a tuple, as a tuple operation
    /// takes them.
    fn tuple_of<'py>(other: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        if let Ok(row) = other.cast::<PyRow>() {
            return row.borrow().values(other.py()).map(Some);
        }
        Ok(other.cast::<PyTuple>().ok().cloned())
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
        Value::Time(nanos) => {
            // As PySpark's `TimeType.fromInternal`: a datetime.time of the microseconds.
            let (hours, remainder) = (
                nanos.div_euclid(3_600_000_000_000),
                nanos.rem_euclid(3_600_000_000_000),
            );
            let (minutes, remainder) = (remainder / 60_000_000_000, remainder % 60_000_000_000);
            let (seconds, remainder) = (remainder / 1_000_000_000, remainder % 1_000_000_000);
            py.import("datetime")?.getattr("time")?.call1((
                hours,
                minutes,
                seconds,
                remainder / 1_000,
            ))?
        }
        Value::DayTimeInterval(micros) => {
            // Normalize using integer arithmetic: neither floats nor an intermediate
            // nanosecond duration can preserve the full signed i64 microsecond range.
            // The resulting +/-107 million days fit timedelta's +/-999999999 days.
            let days = micros.div_euclid(86_400_000_000);
            let remainder = micros.rem_euclid(86_400_000_000);
            py.import("datetime")?.getattr("timedelta")?.call1((
                days,
                remainder / 1_000_000,
                remainder % 1_000_000,
            ))?
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
            // Keys of their own type, in order; an unhashable key, such as an array, raises
            // TypeError, as in PySpark.
            let dict = PyDict::new(py);
            for (k, val) in m {
                dict.set_item(value_to_py(py, k)?, value_to_py(py, val)?)?;
            }
            dict.into_any()
        }
        Value::Struct(row) => {
            // A struct value materializes as a Row (named fields), matching PySpark,
            // rather than a plain dict. Cloning keeps shared names and independent values.
            Py::new(py, PyRow::new(row.clone().into()))?
                .into_bound(py)
                .into_any()
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
    /// Construct a Row, mirroring `pyspark.sql.Row`: `Row(name=value, ...)` makes a row with
    /// field names, and `Row(v1, v2, ...)` one of values alone, which can also serve as a row
    /// class: `Row("name", "age")("Alice", 11)`.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn __new__(
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let kwargs = kwargs.filter(|kw| !kw.is_empty());
        if let Some(kw) = kwargs {
            if !args.is_empty() {
                return Err(raise_pyspark(
                    py,
                    "PySparkValueError",
                    "CANNOT_SET_TOGETHER",
                    &[("arg_list", "args and kwargs")],
                ));
            }
            let mut fields = Vec::with_capacity(kw.len());
            let mut values = Vec::with_capacity(kw.len());
            for (k, v) in kw.iter() {
                fields.push(k.extract::<String>()?);
                values.push(v);
            }
            Ok(PyRow::of_python(fields, PyTuple::new(py, values)?, true))
        } else {
            let fields = (0..args.len()).map(|i| format!("_{}", i + 1)).collect();
            Ok(PyRow::of_python(fields, args.clone(), false))
        }
    }

    /// Number of fields.
    fn __len__(&self) -> usize {
        self.fields().len()
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
        self.values(py)?.try_iter()
    }

    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("builtins")?
            .getattr("reversed")?
            .call1((self.values(py)?,))
    }

    /// `x in row` tests the field names of a row that has them, and the values otherwise.
    fn __contains__(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        if self.named {
            return PyList::new(py, self.fields())?.contains(item);
        }
        self.values(py)?.contains(item)
    }

    /// An index or a slice, as of a tuple, or a field name.
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if key.cast::<PySlice>().is_ok() {
            return self.values(py)?.as_any().get_item(key);
        }
        if let Ok(mut i) = key.extract::<isize>() {
            let n = self.fields().len() as isize;
            if i < 0 {
                i += n;
            }
            if i < 0 || i >= n {
                return Err(PyIndexError::new_err("tuple index out of range"));
            }
            return self.value(py, i as usize);
        }
        // As PySpark looks the key up in `__fields__`, which a row without names lacks.
        if !self.named {
            return Err(Self::no_attribute(py, "__fields__"));
        }
        if let Ok(name) = key.cast::<PyString>() {
            if let Some(value) = self.value_by_name(py, &name.to_cow()?)? {
                return Ok(value);
            }
        }
        let error = py.import("pyspark.errors")?.getattr("PySparkValueError")?;
        Err(PyErr::from_value(error.call1((key,))?))
    }

    /// Attribute access by field name (`row.id`).
    fn __getattr__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        if !name.starts_with("__") && self.named {
            if let Some(value) = self.value_by_name(py, name)? {
                return Ok(value);
            }
        }
        Err(Self::no_attribute(py, name))
    }

    /// Only `__fields__` can be set, which names the fields, as in PySpark.
    fn __setattr__(
        &mut self,
        py: Python<'_>,
        name: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if name != "__fields__" {
            return Err(raise_pyspark(
                py,
                "PySparkRuntimeError",
                "READ_ONLY",
                &[("object", "Row")],
            ));
        }
        let mut names = value
            .try_iter()?
            .map(|name| name?.str().map(|name| name.to_string()))
            .collect::<PyResult<Vec<_>>>()?;
        // PySpark pairs names with values as `zip` does; fields beyond the names keep theirs.
        let fields = self.fields();
        names.truncate(fields.len());
        names.extend(fields[names.len()..].iter().cloned());
        match &mut self.data {
            RowData::Spark { row, .. } => *row = Row::new(names, row.values().to_vec()),
            RowData::Python { fields, .. } => *fields = names,
        }
        self.named = true;
        Ok(())
    }

    /// Field names, in order: a list attribute, as PySpark's `Row.__fields__` is. A row
    /// without names has none.
    #[getter]
    fn __fields__(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        if !self.named {
            return Err(Self::no_attribute(py, "__fields__"));
        }
        Ok(self.fields().to_vec())
    }

    /// As a dict of field name -> value (mirrors `Row.asDict`), with nested rows, also in
    /// lists and dicts, as dicts if `recursive`.
    #[pyo3(name = "asDict")]
    #[pyo3(signature = (recursive=false))]
    fn as_dict<'py>(&self, py: Python<'py>, recursive: bool) -> PyResult<Bound<'py, PyDict>> {
        fn conv<'py>(obj: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let py = obj.py();
            if let Ok(row) = obj.cast::<PyRow>() {
                return Ok(row.borrow().as_dict(py, true)?.into_any());
            }
            if let Ok(list) = obj.cast::<PyList>() {
                let items = list.iter().map(conv).collect::<PyResult<Vec<_>>>()?;
                return Ok(PyList::new(py, items)?.into_any());
            }
            if let Ok(dict) = obj.cast::<PyDict>() {
                let converted = PyDict::new(py);
                for (k, v) in dict.iter() {
                    converted.set_item(k, conv(v)?)?;
                }
                return Ok(converted.into_any());
            }
            Ok(obj)
        }
        if !self.named {
            return Err(raise_pyspark(
                py,
                "PySparkTypeError",
                "CANNOT_CONVERT_TYPE",
                &[("from_type", "Row"), ("to_type", "dict")],
            ));
        }
        let dict = PyDict::new(py);
        for (name, value) in self.fields().iter().zip(self.values(py)?.iter()) {
            dict.set_item(name, if recursive { conv(value)? } else { value })?;
        }
        Ok(dict)
    }

    /// A row class, such as `Row("name", "age")`, makes a row of `args` named by its values.
    #[pyo3(signature = (*args))]
    fn __call__(&self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<PyRow> {
        if args.len() > self.fields().len() {
            let (expected, actual) = (self.fields().len().to_string(), args.len().to_string());
            return Err(raise_pyspark(
                py,
                "PySparkValueError",
                "TOO_MANY_VALUES",
                &[
                    ("expected", expected.as_str()),
                    ("item", "fields"),
                    ("actual", actual.as_str()),
                ],
            ));
        }
        let fields = self
            .values(py)?
            .iter()
            .take(args.len())
            .map(|name| name.str().map(|name| name.to_string()))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyRow::of_python(fields, args.clone(), true))
    }

    /// Pickles as PySpark's row does: with `_create_row` from the field names and values, or
    /// as `Row(*values)` for a row without names.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let row = slf.borrow();
        let values = row.values(py)?;
        if row.named {
            let create_row = py.import("pyspark.sql.types")?.getattr("_create_row")?;
            let args = (PyList::new(py, row.fields())?, values).into_pyobject(py)?;
            return (create_row, args).into_pyobject(py);
        }
        (slf.get_type(), values).into_pyobject(py)
    }

    /// `Row(name='Alice', age=11)`, with the `repr()` of each value, or `<Row('a', 'b')>` for
    /// a row without names, as PySpark prints a row.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let values = self.values(py)?;
        let mut parts = Vec::with_capacity(values.len());
        if self.named {
            for (name, value) in self.fields().iter().zip(values.iter()) {
                parts.push(format!("{}={}", name, value.repr()?));
            }
            return Ok(format!("Row({})", parts.join(", ")));
        }
        for value in values.iter() {
            parts.push(value.repr()?.to_string());
        }
        Ok(format!("<Row({})>", parts.join(", ")))
    }

    /// Hashes as the tuple of its values does.
    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        self.values(py)?.hash()
    }

    /// Compares as the tuple of its values does, with another row or a tuple.
    fn __richcmp__<'py>(
        &self,
        other: &Bound<'py, PyAny>,
        op: CompareOp,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = other.py();
        match Self::tuple_of(other)? {
            Some(other) => self.values(py)?.rich_compare(other, op),
            None => Ok(py.NotImplemented().into_bound(py)),
        }
    }

    /// Concatenates as a tuple does, into a tuple.
    fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = other.py();
        match Self::tuple_of(other)? {
            Some(other) => self.values(py)?.as_any().add(other),
            None => Ok(py.NotImplemented().into_bound(py)),
        }
    }

    fn __radd__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = other.py();
        match Self::tuple_of(other)? {
            Some(other) => other.as_any().add(self.values(py)?),
            None => Ok(py.NotImplemented().into_bound(py)),
        }
    }

    /// Repeats as a tuple does, into a tuple.
    fn __mul__<'py>(&self, times: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.values(times.py())?.as_any().mul(times)
    }

    fn __rmul__<'py>(&self, times: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.__mul__(times)
    }

    /// Number of occurrences of `value` among the row's values (tuple.count).
    fn count(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.values(py)?
            .as_any()
            .call_method1("count", (value,))?
            .extract()
    }

    /// Index of the first occurrence of `value` among the row's values (tuple.index).
    #[pyo3(signature = (value, *args))]
    fn index(
        &self,
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
    ) -> PyResult<usize> {
        let mut call = vec![value.clone()];
        call.extend(args.iter());
        self.values(py)?
            .as_any()
            .call_method1("index", PyTuple::new(py, call)?)?
            .extract()
    }
}
