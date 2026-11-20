//! The value of a scalar or enum field, and its conversion from and to Python as protobuf's
//! Python messages (of upb, the implementation of a `protobuf` install) convert it, with their
//! errors:
//! - an integer field takes an int, or any object with `__index__` but a bool, in its range;
//! - a floating point field anything that `float()` takes but a str or bytes, as `PyFloat_AsDouble`
//!   does;
//! - a bool field a bool, or an int (an object with `__index__`) that a C long holds;
//! - a string field a str, or bytes of UTF-8;
//! - a bytes field bytes;
//! - an enum field the number of a value (any int in the range of an int32, as an open enum), or
//!   the name of one of its values.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyInt, PyString};
use pyo3::IntoPyObjectExt;

use super::pool::{pool, Kind};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Scalar {
    F64(f64),
    F32(f32),
    I64(i64),
    U64(u64),
    I32(i32),
    U32(u32),
    Bool(bool),
    Str(String),
    Bytes(Vec<u8>),
    Enum(i32),
}

impl Scalar {
    pub(crate) fn default_of(kind: Kind) -> Scalar {
        match kind {
            Kind::Double => Scalar::F64(0.0),
            Kind::Float => Scalar::F32(0.0),
            Kind::Int64 | Kind::SInt64 | Kind::SFixed64 => Scalar::I64(0),
            Kind::UInt64 | Kind::Fixed64 => Scalar::U64(0),
            Kind::Int32 | Kind::SInt32 | Kind::SFixed32 => Scalar::I32(0),
            Kind::UInt32 | Kind::Fixed32 => Scalar::U32(0),
            Kind::Bool => Scalar::Bool(false),
            Kind::String => Scalar::Str(String::new()),
            Kind::Bytes => Scalar::Bytes(Vec::new()),
            Kind::Enum => Scalar::Enum(0),
            Kind::Message => unreachable!("a message field holds no scalar"),
        }
    }

    /// Whether a field without presence that holds the value is serialized: protobuf skips the
    /// zero of each type, a float only of the bits of +0.0.
    /// Whether `self` and `other` are equal as upb's `==` of two messages compares the values of
    /// a field (`upb_Message_IsEqual`): a float or a double of its bits, so that a NaN equals a
    /// NaN of the same bits and -0.0 is not 0.0, and any other value of its value. (`==` of the
    /// containers and maps compares their values as Python's `==` does.)
    pub(crate) fn eq_as_upb(&self, other: &Scalar) -> bool {
        match (self, other) {
            (Scalar::F64(a), Scalar::F64(b)) => a.to_bits() == b.to_bits(),
            (Scalar::F32(a), Scalar::F32(b)) => a.to_bits() == b.to_bits(),
            _ => self == other,
        }
    }

    pub(crate) fn is_default(&self) -> bool {
        match self {
            Scalar::F64(v) => v.to_bits() == 0,
            Scalar::F32(v) => v.to_bits() == 0,
            Scalar::I64(v) => *v == 0,
            Scalar::U64(v) => *v == 0,
            Scalar::I32(v) => *v == 0,
            Scalar::U32(v) => *v == 0,
            Scalar::Bool(v) => !*v,
            Scalar::Str(v) => v.is_empty(),
            Scalar::Bytes(v) => v.is_empty(),
            Scalar::Enum(v) => *v == 0,
        }
    }

    pub(crate) fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            Scalar::F64(v) => v.into_py_any(py),
            Scalar::F32(v) => (*v as f64).into_py_any(py),
            Scalar::I64(v) => v.into_py_any(py),
            Scalar::U64(v) => v.into_py_any(py),
            Scalar::I32(v) => v.into_py_any(py),
            Scalar::U32(v) => v.into_py_any(py),
            Scalar::Bool(v) => v.into_py_any(py),
            Scalar::Str(v) => v.into_py_any(py),
            Scalar::Bytes(v) => PyBytes::new(py, v).into_py_any(py),
            Scalar::Enum(v) => v.into_py_any(py),
        }
    }

    /// The value of `value` for the field `field` (its full name, which an error names) of
    /// `kind`, of the enum `enum_type` for an enum field.
    pub(crate) fn from_py(
        kind: Kind,
        field: &str,
        enum_type: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Scalar> {
        match kind {
            // `PyFloat_AsDouble`, as protobuf converts: `__float__`, else `__index__`.
            Kind::Double => Ok(Scalar::F64(value.extract::<f64>()?)),
            Kind::Float => Ok(Scalar::F32(value.extract::<f64>()? as f32)),
            Kind::Int64 | Kind::SInt64 | Kind::SFixed64 => {
                let v = int_of(field, value)?;
                i64::try_from(v)
                    .map(Scalar::I64)
                    .map_err(|_| out_of_range(v))
            }
            Kind::UInt64 | Kind::Fixed64 => {
                let v = int_of(field, value)?;
                u64::try_from(v)
                    .map(Scalar::U64)
                    .map_err(|_| out_of_range(v))
            }
            Kind::Int32 | Kind::SInt32 | Kind::SFixed32 => {
                let v = int_of(field, value)?;
                i32::try_from(v)
                    .map(Scalar::I32)
                    .map_err(|_| out_of_range(v))
            }
            Kind::UInt32 | Kind::Fixed32 => {
                let v = int_of(field, value)?;
                u32::try_from(v)
                    .map(Scalar::U32)
                    .map_err(|_| out_of_range(v))
            }
            Kind::Bool => {
                if let Ok(b) = value.cast::<PyBool>() {
                    return Ok(Scalar::Bool(b.is_true()));
                }
                // As upb's: a bool scalar of numpy, which has no `__index__`, by its truth.
                if is_numpy_bool(value)? {
                    return Ok(Scalar::Bool(value.is_truthy()?));
                }
                // As protobuf's: the int of `__index__`, as a C long.
                let index = index_of(value)?;
                let v = unsafe { ffi::PyLong_AsLong(index.as_ptr()) };
                if v == -1 {
                    if let Some(err) = PyErr::take(value.py()) {
                        return Err(err);
                    }
                }
                Ok(Scalar::Bool(v != 0))
            }
            Kind::String => {
                if let Ok(s) = value.cast::<PyString>() {
                    return Ok(Scalar::Str(s.to_cow()?.into_owned()));
                }
                if let Ok(b) = value.cast::<PyBytes>() {
                    return match std::str::from_utf8(b.as_bytes()) {
                        Ok(s) => Ok(Scalar::Str(s.to_string())),
                        // The UnicodeDecodeError of decoding the bytes, as protobuf raises it.
                        Err(_) => Err(value
                            .call_method1("decode", ("utf-8",))
                            .err()
                            .unwrap_or_else(|| PyValueError::new_err("invalid UTF-8"))),
                    };
                }
                Err(PyTypeError::new_err(
                    "bad argument type for built-in operation",
                ))
            }
            Kind::Bytes => {
                if let Ok(b) = value.cast::<PyBytes>() {
                    return Ok(Scalar::Bytes(b.as_bytes().to_vec()));
                }
                Err(bytes_error(value))
            }
            Kind::Enum => {
                if let Ok(name) = value.cast::<PyString>() {
                    let name = name.to_cow()?;
                    return pool()
                        .enumeration(enum_type)
                        .and_then(|e| e.number_of(&name))
                        .map(Scalar::Enum)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!("unknown enum label \"{name}\""))
                        });
                }
                let v = int_of(field, value)?;
                let number = i32::try_from(v).map_err(|_| out_of_range(v))?;
                // As upb's: a closed enum (of proto2, or of the `CLOSED` enum type of editions)
                // takes only the values that it declares.
                if let Some(e) = pool().enumeration(enum_type) {
                    if e.closed && e.name_of(number).is_none() {
                        return Err(PyValueError::new_err(format!(
                            "invalid enumerator {number}"
                        )));
                    }
                }
                Ok(Scalar::Enum(number))
            }
            Kind::Message => unreachable!("a message field holds no scalar"),
        }
    }
}

/// Whether `value` is a bool scalar of numpy (`numpy.bool_`), as upb tells one.
fn is_numpy_bool(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let class = value.get_type();
    if class.getattr("__module__")?.str()?.to_cow()? != "numpy" {
        return Ok(false);
    }
    let name = class.getattr("__name__")?.str()?.to_cow()?.into_owned();
    Ok(name == "bool_" || name == "bool")
}

/// The int of `value`'s `__index__`, as `operator.index` gives it, with its TypeError ("'str'
/// object cannot be interpreted as an integer") for an object without one.
fn index_of<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if value.is_exact_instance_of::<PyInt>() {
        return Ok(value.clone());
    }
    unsafe { Bound::from_owned_ptr_or_err(value.py(), ffi::PyNumber_Index(value.as_ptr())) }
}

/// The integer of `value` for an integer or enum field: the int of its `__index__`, never of a
/// bool, as protobuf takes one.
fn int_of(field: &str, value: &Bound<'_, PyAny>) -> PyResult<i128> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "Field {field}: Expected an int, got a boolean."
        )));
    }
    let index = index_of(value)?;
    index.extract::<i128>().map_err(|_| out_of_range_of(&index))
}

/// The TypeError of protobuf for a value of a bytes field that is not bytes ("expected bytes, str
/// found"), as CPython's `PyBytes_AsStringAndSize` raises it.
fn bytes_error(value: &Bound<'_, PyAny>) -> PyErr {
    let mut data = std::ptr::null_mut();
    let mut len: ffi::Py_ssize_t = 0;
    let status = unsafe { ffi::PyBytes_AsStringAndSize(value.as_ptr(), &mut data, &mut len) };
    match PyErr::take(value.py()) {
        Some(err) if status == -1 => err,
        _ => PyTypeError::new_err("expected bytes"),
    }
}

fn out_of_range(value: i128) -> PyErr {
    PyValueError::new_err(format!("Value out of range: {value}"))
}

fn out_of_range_of(value: &Bound<'_, PyAny>) -> PyErr {
    let repr = value.repr().map(|r| r.to_string()).unwrap_or_default();
    PyValueError::new_err(format!("Value out of range: {repr}"))
}

/// The order of the keys of a map of Python's `sorted`, in which protobuf's `text_format` writes
/// its entries: of strings by their bytes (as by their code points), of numbers by value. (upb's
/// orders, of its text and its serialization, are those of `text.rs`.)
pub(crate) fn key_order(a: &Scalar, b: &Scalar) -> std::cmp::Ordering {
    match (a, b) {
        (Scalar::Str(x), Scalar::Str(y)) => x.cmp(y),
        (Scalar::I64(x), Scalar::I64(y)) => x.cmp(y),
        (Scalar::I32(x), Scalar::I32(y)) => x.cmp(y),
        (Scalar::U64(x), Scalar::U64(y)) => x.cmp(y),
        (Scalar::U32(x), Scalar::U32(y)) => x.cmp(y),
        (Scalar::Bool(x), Scalar::Bool(y)) => x.cmp(y),
        _ => std::cmp::Ordering::Equal,
    }
}
