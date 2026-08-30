//! `pyspark.util.PythonEvalType` — the evaluation-type constants shared with the JVM
//! (`org.apache.spark.api.python.PythonEvalType`). Rust-backed so the values live in
//! the core extension, matching reference pyspark.

use pyo3::prelude::*;

/// Evaluation-type constants for Python UDFs/UDTFs. Mirrors `pyspark.util.PythonEvalType`.
#[pyclass(name = "PythonEvalType", module = "pyspark.util")]
pub struct PyPythonEvalType;

#[allow(non_upper_case_globals)]
#[pymethods]
impl PyPythonEvalType {
    #[classattr]
    const NON_UDF: i32 = 0;
    #[classattr]
    const SQL_BATCHED_UDF: i32 = 100;
    #[classattr]
    const SQL_ARROW_BATCHED_UDF: i32 = 101;
    #[classattr]
    const SQL_SCALAR_PANDAS_UDF: i32 = 200;
    #[classattr]
    const SQL_GROUPED_MAP_PANDAS_UDF: i32 = 201;
    #[classattr]
    const SQL_GROUPED_AGG_PANDAS_UDF: i32 = 202;
    #[classattr]
    const SQL_WINDOW_AGG_PANDAS_UDF: i32 = 203;
    #[classattr]
    const SQL_SCALAR_PANDAS_ITER_UDF: i32 = 204;
    #[classattr]
    const SQL_MAP_PANDAS_ITER_UDF: i32 = 205;
    #[classattr]
    const SQL_COGROUPED_MAP_PANDAS_UDF: i32 = 206;
    #[classattr]
    const SQL_MAP_ARROW_ITER_UDF: i32 = 207;
    #[classattr]
    const SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE: i32 = 208;
    #[classattr]
    const SQL_GROUPED_MAP_ARROW_UDF: i32 = 209;
    #[classattr]
    const SQL_COGROUPED_MAP_ARROW_UDF: i32 = 210;
    #[classattr]
    const SQL_TRANSFORM_WITH_STATE_PANDAS_UDF: i32 = 211;
    #[classattr]
    const SQL_TRANSFORM_WITH_STATE_PANDAS_INIT_STATE_UDF: i32 = 212;
    #[classattr]
    const SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_UDF: i32 = 213;
    #[classattr]
    const SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_INIT_STATE_UDF: i32 = 214;
    #[classattr]
    const SQL_GROUPED_MAP_ARROW_ITER_UDF: i32 = 215;
    #[classattr]
    const SQL_GROUPED_MAP_PANDAS_ITER_UDF: i32 = 216;
    #[classattr]
    const SQL_GROUPED_AGG_PANDAS_ITER_UDF: i32 = 217;
    #[classattr]
    const SQL_SCALAR_ARROW_UDF: i32 = 250;
    #[classattr]
    const SQL_SCALAR_ARROW_ITER_UDF: i32 = 251;
    #[classattr]
    const SQL_GROUPED_AGG_ARROW_UDF: i32 = 252;
    #[classattr]
    const SQL_WINDOW_AGG_ARROW_UDF: i32 = 253;
    #[classattr]
    const SQL_GROUPED_AGG_ARROW_ITER_UDF: i32 = 254;
    #[classattr]
    const SQL_TABLE_UDF: i32 = 300;
    #[classattr]
    const SQL_ARROW_TABLE_UDF: i32 = 301;
    #[classattr]
    const SQL_ARROW_UDTF: i32 = 302;
}
