// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Scalar conversion shared by client row materialization and UDF columns.
//!
//! Exact extraction never invokes a subclass's coercion methods. The separate
//! `extract_*` helpers retain PyO3's conversion semantics; callers choose their order, notably
//! checking booleans before integers and Decimal before numeric coercion. This module does not
//! infer a schema or define nested, decimal, timestamp, or application-specific row types.

use std::borrow::Cow;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyFloat, PyInt, PyString};
use pyo3::IntoPyObjectExt;

/// The ordinal of 1970-01-01 in Python's `datetime.date` calendar.
const EPOCH_ORDINAL: i64 = 719_163;

/// A borrowed scalar, independent of a client's owned value representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar<'a> {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(&'a str),
    /// Materialized as immutable Python `bytes`.
    Binary(&'a [u8]),
    /// Days since 1970-01-01, materialized as `datetime.date`.
    Date(i32),
}

impl Scalar<'_> {
    /// Materializes this scalar in the caller's attached interpreter.
    pub fn into_python<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Self::Null => Ok(py.None().into_bound(py)),
            Self::Bool(value) => value.into_bound_py_any(py),
            Self::Int(value) => value.into_bound_py_any(py),
            Self::Float(value) => value.into_bound_py_any(py),
            Self::String(value) => utf8_into_python(py, value.as_bytes()),
            Self::Binary(value) => Ok(PyBytes::new(py, value).into_any()),
            Self::Date(days) => {
                let fromordinal = py
                    .import("datetime")?
                    .getattr("date")?
                    .getattr("fromordinal")?;
                date_into_python(days, &fromordinal)
            }
        }
    }
}

/// Decodes UTF-8 with Python's strict decoder, preserving its exception on invalid bytes.
///
/// This accepts raw bytes because an Arrow C Data import can contain unvalidated string data;
/// forming a Rust `&str` first would be unsound for such an input.
pub fn utf8_into_python<'py>(py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyString::from_bytes(py, bytes)?.into_any())
}

/// Whether days since 1970-01-01 fall within Python's date range (years 1 through 9999).
pub fn is_date_in_range(days: i32) -> bool {
    (-719_162..=2_932_896).contains(&days)
}

/// Materializes a date using an explicitly supplied, optionally cached `date.fromordinal`.
///
/// The constructor defines out-of-range errors. A caller reproducing another conversion path's
/// errors can check [`is_date_in_range`] before invoking it.
pub fn date_into_python<'py>(
    days: i32,
    fromordinal: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    fromordinal.call1((i64::from(days) + EPOCH_ORDINAL,))
}

/// An exact built-in boolean, excluding truthiness coercion.
pub fn exact_bool(value: &Bound<'_, PyAny>) -> Option<bool> {
    value
        .is_exact_instance_of::<PyBool>()
        .then(|| value.extract().ok())
        .flatten()
}

/// An exact built-in integer fitting in 64 bits, excluding booleans and integer subclasses.
pub fn exact_i64(value: &Bound<'_, PyAny>) -> Option<i64> {
    value
        .is_exact_instance_of::<PyInt>()
        .then(|| value.extract().ok())
        .flatten()
}

/// An exact built-in float, without integer or user-defined numeric coercion.
pub fn exact_f64(value: &Bound<'_, PyAny>) -> Option<f64> {
    value
        .is_exact_instance_of::<PyFloat>()
        .then(|| value.extract().ok())
        .flatten()
}

/// An exact built-in string with a valid UTF-8 representation.
///
/// PyO3 uses a supported ABI conversion and may allocate when the selected limited API cannot
/// borrow CPython's UTF-8 cache. Lone surrogates return `None`, leaving error/fallback policy to
/// the caller, rather than replacing characters or retaining a Python error indicator.
pub fn exact_str<'a>(value: &'a Bound<'_, PyAny>) -> Option<Cow<'a, str>> {
    if !value.is_exact_instance_of::<PyString>() {
        return None;
    }
    value.cast::<PyString>().ok()?.to_cow().ok()
}

/// The borrowed contents of exact built-in `bytes`, excluding subclasses and mutable buffers.
pub fn exact_bytes<'a>(value: &'a Bound<'_, PyAny>) -> Option<&'a [u8]> {
    if !value.is_exact_instance_of::<PyBytes>() {
        return None;
    }
    value.cast::<PyBytes>().ok().map(|bytes| bytes.as_bytes())
}

/// Retains `value` only when it is exact built-in `bytes`.
pub fn into_exact_bytes(value: Bound<'_, PyAny>) -> Option<Bound<'_, PyBytes>> {
    if !value.is_exact_instance_of::<PyBytes>() {
        return None;
    }
    value.cast_into::<PyBytes>().ok()
}

/// Extracts with PyO3's boolean rules, not Python truthiness.
pub fn extract_bool(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    value.extract()
}

/// Extracts with PyO3's integer rules, including accepted integer-like objects.
pub fn extract_i64(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    value.extract()
}

/// Extracts with PyO3's float rules, including numeric coercion.
pub fn extract_f64(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    value.extract()
}

/// Extracts with PyO3's string rules, including string subclasses but not arbitrary `str()`.
pub fn extract_string(value: &Bound<'_, PyAny>) -> PyResult<String> {
    value.extract()
}

/// Copies bytes or bytearray contents, including subclasses, but never a list of integers.
pub fn extract_binary(value: &Bound<'_, PyAny>) -> Option<Vec<u8>> {
    if let Ok(bytes) = value.cast::<PyBytes>() {
        Some(bytes.as_bytes().to_vec())
    } else if let Ok(bytes) = value.cast::<PyByteArray>() {
        Some(bytes.to_vec())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_range_matches_python_calendar_boundaries() {
        assert!(is_date_in_range(-719_162));
        assert!(is_date_in_range(2_932_896));
        assert!(!is_date_in_range(-719_163));
        assert!(!is_date_in_range(2_932_897));
        assert!(!is_date_in_range(i32::MIN));
        assert!(!is_date_in_range(i32::MAX));
    }
}
