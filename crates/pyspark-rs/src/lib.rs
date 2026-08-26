//! PyO3 bindings for the Rust Spark Connect client.

use pyo3::prelude::*;
use pyo3::types::PyBool;

/// Coerce a Python option value the way reference pyspark's `to_str` does:
/// `None` -> `None` (the option is left unset, NOT the literal string "None"), a
/// bool -> lowercase `"true"`/`"false"` (not Python's `"True"`/`"False"`), and
/// everything else -> its `str()`. Used by every reader/writer/conf `option(s)`
/// binding so option handling matches the reference client.
pub(crate) fn coerce_option_value(v: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if v.is_none() {
        return Ok(None);
    }
    if let Ok(b) = v.downcast::<PyBool>() {
        return Ok(Some(if b.is_true() { "true" } else { "false" }.to_string()));
    }
    Ok(Some(v.str()?.to_string()))
}

mod catalog;
mod column;
mod conf;
mod dataframe;
mod datasource;
mod errors;
mod functions;
mod group;
mod profiler;
mod readwriter;
mod resource;
mod row;
mod session;
mod stat;
mod streaming;
mod transport;
mod types;
mod window;

use catalog::PyCatalog;
use column::PyColumn;
use conf::PyRuntimeConf;
use dataframe::{
    PyDataFrame, PyDataFrameNaFunctions, PyDataFrameWriter, PyDataFrameWriterV2,
    PyLocalRowIterator, PyMergeIntoWriter, PyWhenMatched, PyWhenNotMatched,
    PyWhenNotMatchedBySource,
};
use datasource::PyDataSourceRegistration;
use group::{PyCoGroupedData, PyGroupedData};
use profiler::PyProfilerCollector;
use pyo3::prelude::*;
use readwriter::PyDataFrameReader;
use resource::{
    PyExecutorResourceRequests, PyResourceProfile, PyResourceProfileBuilder, PyTaskResourceRequests,
};
use row::PyRow;
use session::{PySparkSession, PySparkSessionBuilder};
use stat::PyStatFunctions;
use streaming::{
    PyDataStreamReader, PyDataStreamWriter, PyListenerEventStream, PyStreamingQuery,
    PyStreamingQueryException, PyStreamingQueryManager, PyStreamingQueryStatus, PyTrigger,
};
use types::{
    PyArrayType, PyBinaryType, PyBooleanType, PyByteType, PyCalendarIntervalType, PyCharType,
    PyDataType, PyDateType, PyDayTimeIntervalType, PyDecimalType, PyDoubleType, PyFloatType,
    PyIntegerType, PyLongType, PyMapType, PyNullType, PyShortType, PyStringType, PyStructField,
    PyStructType, PyTimeType, PyTimestampNTZType, PyTimestampType, PyVarcharType, PyVariantType,
    PyYearMonthIntervalType,
};
use window::{PyFrameBound, PyWindow, PyWindowSpec};

#[pymodule]
fn _pyspark(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySparkSession>()?;
    m.add_class::<PySparkSessionBuilder>()?;
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PyDataFrameWriter>()?;
    m.add_class::<PyDataFrameWriterV2>()?;
    m.add_class::<PyDataFrameNaFunctions>()?;
    m.add_class::<PyMergeIntoWriter>()?;
    m.add_class::<PyWhenMatched>()?;
    m.add_class::<PyWhenNotMatched>()?;
    m.add_class::<PyWhenNotMatchedBySource>()?;
    m.add_class::<PyColumn>()?;
    m.add_class::<PyRow>()?;
    m.add_class::<PyGroupedData>()?;
    m.add_class::<PyCoGroupedData>()?;
    m.add_class::<PyLocalRowIterator>()?;
    m.add_class::<transport::RustConnectStub>()?;
    m.add_class::<transport::ResponseStream>()?;
    m.add("RustRpcError", m.py().get_type::<transport::RustRpcError>())?;
    m.add_class::<PyCatalog>()?;
    m.add_class::<PyRuntimeConf>()?;
    m.add_class::<PyDataFrameReader>()?;
    m.add_class::<PyStatFunctions>()?;

    // Streaming classes
    m.add_class::<PyDataStreamReader>()?;
    m.add_class::<PyDataStreamWriter>()?;
    m.add_class::<PyTrigger>()?;
    m.add_class::<PyStreamingQuery>()?;
    m.add_class::<PyStreamingQueryStatus>()?;
    m.add_class::<PyStreamingQueryException>()?;
    m.add_class::<PyStreamingQueryManager>()?;
    m.add_class::<PyListenerEventStream>()?;

    // Resource profile classes
    m.add_class::<PyExecutorResourceRequests>()?;
    m.add_class::<PyTaskResourceRequests>()?;
    m.add_class::<PyResourceProfileBuilder>()?;
    m.add_class::<PyResourceProfile>()?;

    // Data source and profiler classes
    m.add_class::<PyDataSourceRegistration>()?;
    m.add_class::<PyProfilerCollector>()?;

    // Register DataType classes
    m.add_class::<PyDataType>()?;
    m.add_class::<PyNullType>()?;
    m.add_class::<PyBooleanType>()?;
    m.add_class::<PyByteType>()?;
    m.add_class::<PyShortType>()?;
    m.add_class::<PyIntegerType>()?;
    m.add_class::<PyLongType>()?;
    m.add_class::<PyFloatType>()?;
    m.add_class::<PyDoubleType>()?;
    m.add_class::<PyDecimalType>()?;
    m.add_class::<PyStringType>()?;
    m.add_class::<PyBinaryType>()?;
    m.add_class::<PyDateType>()?;
    m.add_class::<PyTimestampType>()?;
    m.add_class::<PyTimestampNTZType>()?;
    m.add_class::<PyArrayType>()?;
    m.add_class::<PyMapType>()?;
    m.add_class::<PyStructField>()?;
    m.add_class::<PyStructType>()?;
    m.add_class::<PyCharType>()?;
    m.add_class::<PyVarcharType>()?;
    m.add_class::<PyTimeType>()?;
    m.add_class::<PyCalendarIntervalType>()?;
    m.add_class::<PyYearMonthIntervalType>()?;
    m.add_class::<PyDayTimeIntervalType>()?;
    m.add_class::<PyVariantType>()?;

    // Register Window classes
    m.add_class::<PyWindow>()?;
    m.add_class::<PyWindowSpec>()?;
    m.add_class::<PyFrameBound>()?;

    // Register functions as a submodule
    let functions_module = PyModule::new(_py, "functions")?;
    functions::register_functions(_py, &functions_module)?;
    m.add_submodule(&functions_module)?;

    Ok(())
}
