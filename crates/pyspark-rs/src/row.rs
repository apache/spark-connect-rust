//! Conversion of native Spark values into tuple-backed Python Rows.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyTuple};
use spark_connect::row::{StructValue, Value};

/// Functions that deserialize user-defined or spatial values in each result field.
pub(crate) type FieldConverters = Arc<Vec<Option<Py<PyAny>>>>;

static CREATE_ROW: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
const ROW_SOURCE: &std::ffi::CStr = pyo3::ffi::c_str!(include_str!("row.py"));

/// Expose the embedded tuple `Row` as `pyspark._pyspark.Row`, re-exported by the Python shims.
/// Its source imports PySpark modules only inside methods, avoiding import cycles
/// while `pyspark.sql.types` is importing this extension.
/// The synthetic filename avoids embedding a build-machine path in distributed wheels;
/// the source is compiled into the extension and need not be installed separately.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let row_module =
        PyModule::from_code(module.py(), ROW_SOURCE, c"pyspark/_row.py", c"pyspark._row")?;
    module.add("Row", row_module.getattr("Row")?)
}

/// One step of a nested schema path. Array elements and map keys/values carry no
/// index, so all of them share the names at their path.
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
enum NestedFieldPath {
    Field(usize),
    ArrayElement,
    MapKey,
    MapValue,
}

/// Mutable names for each nested schema path, shared across the result's Arrow batches.
/// Distinct sibling structs retain independent lists even when their names are identical.
#[derive(Default)]
pub(crate) struct NestedFieldNames {
    path: Vec<NestedFieldPath>,
    fields: HashMap<Vec<NestedFieldPath>, Py<PyList>>,
    /// The schema paths of the maps whose keys have a user-defined type.
    udt_key_maps: HashSet<Vec<NestedFieldPath>>,
}

impl NestedFieldNames {
    /// Release names when an iterator closes, including when materialization fails.
    pub(crate) fn clear(&mut self) {
        self.path.clear();
        self.fields.clear();
        self.udt_key_maps.clear();
    }

    /// Record the maps of `schema` whose keys have a user-defined type. Their keys are
    /// deserialized by the converters, so such a map is passed on as its (key, value) pairs.
    pub(crate) fn note_udt_key_maps(&mut self, schema: &spark_connect::types::DataType) {
        let mut path = Vec::new();
        collect_udt_key_maps(schema, &mut path, &mut self.udt_key_maps);
    }

    /// Convert one schema step below this path, restoring it even if conversion fails.
    fn value_at<'py>(
        &mut self,
        py: Python<'py>,
        value: &Value,
        binary_as_bytes: bool,
        path: NestedFieldPath,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.path.push(path);
        let result = value_to_py(py, value, binary_as_bytes, self);
        self.path.pop();
        result
    }

    /// Get mutable names for this path, using the first struct seen to initialize them.
    fn names_at_path<'py>(
        &mut self,
        py: Python<'py>,
        row: &StructValue,
    ) -> PyResult<Bound<'py, PyList>> {
        if let Some(fields) = self.fields.get(&self.path) {
            return Ok(fields.bind(py).clone());
        }
        let fields = PyList::new(py, row.fields())?;
        self.fields
            .insert(self.path.clone(), fields.clone().unbind());
        Ok(fields)
    }
}

/// Collect the paths of the maps in `data_type` whose keys have a user-defined type, along
/// the same steps that `value_to_py` takes through the values.
fn collect_udt_key_maps(
    data_type: &spark_connect::types::DataType,
    path: &mut Vec<NestedFieldPath>,
    out: &mut HashSet<Vec<NestedFieldPath>>,
) {
    use spark_connect::types::DataType;
    match data_type {
        DataType::Struct { fields } => {
            for (index, field) in fields.iter().enumerate() {
                path.push(NestedFieldPath::Field(index));
                collect_udt_key_maps(&field.data_type, path, out);
                path.pop();
            }
        }
        DataType::Array { element_type, .. } => {
            path.push(NestedFieldPath::ArrayElement);
            collect_udt_key_maps(element_type, path, out);
            path.pop();
        }
        DataType::Map {
            key_type,
            value_type,
            ..
        } => {
            if crate::types::has_udt(key_type) {
                out.insert(path.clone());
            }
            path.push(NestedFieldPath::MapKey);
            collect_udt_key_maps(key_type, path, out);
            path.pop();
            path.push(NestedFieldPath::MapValue);
            collect_udt_key_maps(value_type, path, out);
            path.pop();
        }
        DataType::Udt {
            sql_type: Some(sql_type),
            ..
        } => collect_udt_key_maps(sql_type, path, out),
        _ => {}
    }
}

/// Build a Row from named values, deserializing user-defined fields with `converters`.
/// The caller shares `fields` across result rows and nested names across the execution.
pub(crate) fn row_values_to_py(
    py: Python<'_>,
    fields: &Bound<'_, PyList>,
    values: &[Value],
    converters: Option<&FieldConverters>,
    binary_as_bytes: bool,
    nested_fields: &mut NestedFieldNames,
) -> PyResult<Py<PyAny>> {
    let values = values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let value = nested_fields.value_at(
                py,
                value,
                binary_as_bytes,
                NestedFieldPath::Field(index),
            )?;
            match converters
                .and_then(|converters| converters.get(index))
                .and_then(Option::as_ref)
            {
                Some(converter) => converter.bind(py).call1((value,)),
                None => Ok(value),
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    let create_row = CREATE_ROW.get_or_try_init(py, || {
        py.import("pyspark.sql.types")?
            .getattr("_create_row")
            .map(Bound::unbind)
    })?;
    Ok(create_row
        .bind(py)
        .call1((fields, PyTuple::new(py, values)?))?
        .unbind())
}

/// `datetime.datetime(1970, 1, 1)`, the epoch of Spark's timestamps, without a time zone.
fn naive_epoch(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    py.import("datetime")?
        .getattr("datetime")?
        .call1((1970, 1, 1))
}

/// `datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)`, the epoch of the instants.
fn utc_epoch(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let datetime = py.import("datetime")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("tzinfo", datetime.getattr("timezone")?.getattr("utc")?)?;
    datetime
        .getattr("datetime")?
        .call((1970, 1, 1), Some(&kwargs))
}

/// The `datetime.timedelta` of `micros` microseconds, exact over the whole i64 range: a float
/// of seconds would lose the microseconds of the values far from the epoch.
fn micros_timedelta(py: Python<'_>, micros: i64) -> PyResult<Bound<'_, PyAny>> {
    let days = micros.div_euclid(86_400_000_000);
    let remainder = micros.rem_euclid(86_400_000_000);
    py.import("datetime")?.getattr("timedelta")?.call1((
        days,
        remainder / 1_000_000,
        remainder % 1_000_000,
    ))
}

/// Convert a Spark `Value` into a Python object (PyO3 0.28 `IntoPyObject`), whose binary values
/// are `bytes` if `binary_as_bytes`, or else `bytearray`, also in arrays, maps and structs.
/// `nested_fields` supplies shared struct names. Call through `NestedFieldNames::value_at`
/// so its schema path is set for this value and restored after conversion.
fn value_to_py<'py>(
    py: Python<'py>,
    v: &Value,
    binary_as_bytes: bool,
    nested_fields: &mut NestedFieldNames,
) -> PyResult<Bound<'py, PyAny>> {
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
        Value::Binary(b) if binary_as_bytes => PyBytes::new(py, b).into_any(),
        Value::Binary(b) => PyByteArray::new(py, b).into_any(),
        Value::Date(d) => {
            // Convert days-since-epoch (i32) to datetime.date
            let date_mod = py.import("datetime")?;
            let date_cls = date_mod.getattr("date")?;
            if (-719_162..=2_932_896).contains(d) {
                // 719163 is the ordinal of 1970-01-01
                let ordinal = (*d as i64) + 719163i64;
                date_cls.call_method1("fromordinal", (ordinal,))?.into_any()
            } else {
                // Out of the dates of Python, as PyArrow's `Date32Scalar.as_py`, of
                // `date(1970, 1, 1) + timedelta(days=d)`, which raises its OverflowError.
                let kwargs = PyDict::new(py);
                kwargs.set_item("days", *d)?;
                let delta = date_mod.getattr("timedelta")?.call((), Some(&kwargs))?;
                date_cls.call1((1970, 1, 1))?.add(delta)?.into_any()
            }
        }
        // TIMESTAMP (with a time zone): the instant, shown in the local time zone of the client,
        // as PySpark's Arrow conversion does (`astimezone()` of the UTC datetime, naive).
        Value::Timestamp(t) => {
            let utc = utc_epoch(py)?.add(micros_timedelta(py, *t)?)?;
            let local = utc.call_method0("astimezone")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("tzinfo", py.None())?;
            local.call_method("replace", (), Some(&kwargs))?.into_any()
        }
        // TIMESTAMP_NTZ (without a time zone): the wall clock, which no time zone shifts, as
        // PySpark's `TimestampNTZType.fromInternal` gives it.
        Value::TimestampNtz(t) => naive_epoch(py)?.add(micros_timedelta(py, *t)?)?,
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
        // As PySpark's conversion of the rows, which cannot collect a year-month interval (the
        // collects raise this already from the schema of the result).
        Value::YearMonthInterval(_) => {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkNotImplementedError",
                "NOT_IMPLEMENTED",
                &[(
                    "feature",
                    "Collecting a year-month interval value in Spark Connect",
                )],
            ))
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
                list.append(nested_fields.value_at(
                    py,
                    it,
                    binary_as_bytes,
                    NestedFieldPath::ArrayElement,
                )?)?;
            }
            list.into_any()
        }
        Value::Map(m) if nested_fields.udt_key_maps.contains(&nested_fields.path) => {
            // The keys are deserialized from their user-defined type by the converter of the
            // field, which builds the dict, so the entries are passed on as pairs.
            let pairs = PyList::empty(py);
            for (k, val) in m {
                let key =
                    nested_fields.value_at(py, k, binary_as_bytes, NestedFieldPath::MapKey)?;
                let val =
                    nested_fields.value_at(py, val, binary_as_bytes, NestedFieldPath::MapValue)?;
                pairs.append(PyTuple::new(py, [key, val])?)?;
            }
            pairs.into_any()
        }
        Value::Map(m) => {
            // Keys of their own type, in order; an unhashable key, such as an array, raises
            // TypeError, as in PySpark.
            let dict = PyDict::new(py);
            for (k, val) in m {
                dict.set_item(
                    nested_fields.value_at(py, k, binary_as_bytes, NestedFieldPath::MapKey)?,
                    nested_fields.value_at(py, val, binary_as_bytes, NestedFieldPath::MapValue)?,
                )?;
            }
            dict.into_any()
        }
        Value::Struct(row) => {
            // A struct value materializes as a Row with the mutable names of its schema path.
            let fields = nested_fields.names_at_path(py, row)?;
            row_values_to_py(
                py,
                &fields,
                row.values(),
                None,
                binary_as_bytes,
                nested_fields,
            )?
            .into_bound(py)
        }
        Value::Variant { value, metadata } => {
            // A VARIANT materializes as a VariantVal (toJson/toPython decode lazily),
            // matching pyspark, rather than a raw {value, metadata} dict.
            crate::values::variant_val(py, value, metadata)?
        }
    })
}
