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
mod eval_type;
mod functions;
mod group;
mod ml;
mod observation;
mod pipelines;
mod profiler;
mod readwriter;
mod resource;
mod row;
mod session;
mod stat;
mod storagelevel;
mod streaming;
mod tablearg;
mod transport;
mod tvf;
mod types;
mod udtf_analyze;
mod values;
mod window;

use catalog::{
    PyCatalog, PyCatalogColumn, PyCatalogMetadata, PyDatabase, PyFunction, PyTable,
    PyTablePartition,
};
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
    PyAnsiIntervalType, PyAnyTimeType, PyArrayType, PyAtomicType, PyBinaryType, PyBooleanType,
    PyByteType, PyCalendarIntervalType, PyCharType, PyDataType, PyDateType, PyDatetimeType,
    PyDayTimeIntervalType, PyDecimalType, PyDoubleType, PyFloatType, PyFractionalType,
    PyGeographyType, PyGeometryType, PyIntegerType, PyIntegralType, PyLongType, PyMapType,
    PyNullType, PyNumericType, PyShortType, PySpatialType, PyStringType, PyStructField,
    PyStructType, PyTimeType, PyTimestampNTZType, PyTimestampType, PyVarcharType, PyVariantType,
    PyYearMonthIntervalType,
};
use window::{PyFrameBound, PyWindow, PyWindowSpec};

#[pymodule]
fn _pyspark(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySparkSession>()?;
    m.add_class::<session::PyConnectClientStub>()?;
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
    m.add_class::<PyCatalogMetadata>()?;
    m.add_class::<PyDatabase>()?;
    m.add_class::<PyTable>()?;
    m.add_class::<PyCatalogColumn>()?;
    m.add_class::<PyFunction>()?;
    m.add_class::<PyTablePartition>()?;
    m.add_class::<eval_type::PyPythonEvalType>()?;
    m.add_class::<PyRuntimeConf>()?;
    m.add_class::<PyDataFrameReader>()?;
    m.add_class::<PyStatFunctions>()?;
    m.add_class::<storagelevel::PyStorageLevel>()?;

    // Spark Declarative Pipelines (SDP) command execution
    m.add_class::<pipelines::PyPipelineRunStream>()?;
    m.add_function(wrap_pyfunction!(
        pipelines::pipeline_create_dataflow_graph,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(pipelines::pipeline_define_output, m)?)?;
    m.add_function(wrap_pyfunction!(pipelines::pipeline_define_flow, m)?)?;
    m.add_function(wrap_pyfunction!(
        pipelines::pipeline_define_auto_cdc_flow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        pipelines::pipeline_define_sql_graph_elements,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(pipelines::pipeline_start_run, m)?)?;

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
    // Abstract intermediate base classes (type hierarchy).
    m.add_class::<PyAtomicType>()?;
    m.add_class::<PyNumericType>()?;
    m.add_class::<PyIntegralType>()?;
    m.add_class::<PyFractionalType>()?;
    m.add_class::<PyDatetimeType>()?;
    m.add_class::<PyAnyTimeType>()?;
    m.add_class::<PyAnsiIntervalType>()?;
    m.add_class::<PySpatialType>()?;
    m.add_class::<PyGeometryType>()?;
    m.add_class::<PyGeographyType>()?;
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
    m.add_class::<values::PyVariantVal>()?;
    m.add_class::<values::PyGeography>()?;
    m.add_class::<values::PyGeometry>()?;

    // Register Window classes
    m.add_class::<PyWindow>()?;
    m.add_class::<PyWindowSpec>()?;
    m.add_class::<PyFrameBound>()?;

    // Register TVF + Observation
    m.add_class::<tvf::PyTableValuedFunction>()?;
    m.add_class::<tablearg::PyTableArg>()?;
    m.add_class::<udtf_analyze::PyAnalyzeArgument>()?;
    m.add_class::<udtf_analyze::PyPartitioningColumn>()?;
    m.add_class::<udtf_analyze::PyOrderingColumn>()?;
    m.add_class::<udtf_analyze::PySelectedColumn>()?;
    m.add_class::<udtf_analyze::PyAnalyzeResult>()?;
    m.add(
        "SkipRestOfInputTableException",
        m.py()
            .get_type::<udtf_analyze::SkipRestOfInputTableException>(),
    )?;

    m.add_class::<observation::PyObservation>()?;

    // Register ML classes (pyspark.ml.connect)
    m.add_class::<ml::PyMLModel>()?;
    m.add_class::<ml::PyStandardScaler>()?;
    m.add_class::<ml::PyMaxAbsScaler>()?;
    m.add_class::<ml::PyStringIndexer>()?;
    m.add_class::<ml::PyVectorAssembler>()?;
    m.add_class::<ml::PyLogisticRegression>()?;
    m.add_class::<ml::PyRegressionEvaluator>()?;
    m.add_class::<ml::PyBinaryClassificationEvaluator>()?;
    m.add_class::<ml::PyPipeline>()?;
    m.add_class::<ml::PyMulticlassClassificationEvaluator>()?;
    m.add_class::<ml::PyCrossValidator>()?;

    // Register functions as a submodule
    let functions_module = PyModule::new(_py, "functions")?;
    functions::register_functions(_py, &functions_module)?;
    m.add_submodule(&functions_module)?;

    Ok(())
}
