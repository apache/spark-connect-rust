//! Error mapping from Rust SparkError to Python exceptions.

use pyo3::exceptions;
use pyo3::prelude::*;
use spark_connect_core::error::SparkError;

/// Maps a SparkError to a Python exception.
pub fn spark_error_to_py_exception(err: SparkError) -> PyErr {
    let msg = err.to_string();
    exceptions::PyRuntimeError::new_err(msg)
}

/// Helper to convert Result<T, SparkError> to PyResult<T>.
pub trait ResultExt<T> {
    fn to_pyerr(self) -> PyResult<T>;
}

impl<T> ResultExt<T> for Result<T, SparkError> {
    fn to_pyerr(self) -> PyResult<T> {
        self.map_err(spark_error_to_py_exception)
    }
}
