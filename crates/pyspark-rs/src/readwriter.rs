//! The conversions of the arguments of the readers of Spark Connect, as PySpark's plans of them
//! convert them when they are sent.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};
use spark_connect::dataframe::DataFrame;
use spark_connect_core::error::SparkError;

use crate::options::utf32le_of;

/// The DataFrame `df`, which fails each request with `error` when it is one: the error of the plan
/// that PySpark raises when it sends the plan of a read, whose arguments it converts there.
pub(crate) fn with_plan_error(df: DataFrame, error: Option<SparkError>) -> DataFrame {
    match error {
        Some(error) => df.unbuildable(error),
        None => df,
    }
}

/// The name of a table, as PySpark's `Read` plan sets it in its string field: a str that UTF-8
/// cannot encode is kept with the error of that plan, and the name is empty. Bytes are decoded as
/// [`crate::str_argument`] decodes them, and bytes that are not UTF-8 are kept with their decode
/// error. Any other object is kept too, and fails the plan with the TypeError of protobuf's string
/// field, as PySpark's `table` does at the action and not at the call.
pub(crate) fn table_name_of(name: &Bound<'_, PyAny>) -> PyResult<(String, Option<SparkError>)> {
    if name.is_instance_of::<PyString>() {
        if let Ok(text) = name.extract::<String>() {
            return Ok((text, None));
        }
        let error = SparkError::unencodable_str(&utf32le_of(name)?);
        return Ok((String::new(), Some(error)));
    }
    if let Ok(bytes) = name.cast::<PyBytes>() {
        return Ok(match crate::str_argument(name) {
            Ok(text) => (text, None),
            Err(_) => (
                String::new(),
                Some(SparkError::invalid_utf8(0, bytes.as_bytes())),
            ),
        });
    }
    Ok((String::new(), Some(crate::options::builtin_type_error())))
}

