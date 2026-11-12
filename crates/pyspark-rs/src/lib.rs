//! PyO3 bindings for the Rust Spark Connect client.
//!
//! `non_snake_case` is allowed crate-wide: these classes deliberately expose PySpark's
//! camelCase public API (e.g. `toPandas`, `withColumn`, `containsNull`), so the Rust method,
//! field, and argument names must match the Python names verbatim — renaming them would change
//! the user-facing API.
#![allow(non_snake_case)]

use pyo3::prelude::*;

mod catalog;
mod column;
mod conf;
mod connect_bridge;
mod dataframe;
mod datasource;
mod errors;
mod eval_type;
mod functions;
mod group;
mod infer;
mod instance_dict;
mod interrupt;
mod ml;
mod observation;
mod options;
mod origin;
mod pandas_input;
mod pipelines;
mod profiler;
mod progress;
mod pyarrow_export;
mod pyarrow_import;
mod readwriter;
mod resource;
mod row;
mod session;
mod stat;
mod stat_arguments;
mod storagelevel;
mod streaming;
mod tablearg;
mod transport;
mod tvf;
mod types;
mod udtf_analyze;
mod values;
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
pub(crate) use options::{coerce_option_value, str_text};
use readwriter::PyDataFrameReader;
use resource::{
    PyExecutorResourceRequests, PyResourceProfile, PyResourceProfileBuilder, PyTaskResourceRequests,
};
use session::{PyHook, PySparkSession, PySparkSessionBuilder};
use stat::PyStatFunctions;
use streaming::{
    PyDataStreamReader, PyDataStreamWriter, PyListenerEventStream, PyStreamingQuery,
    PyStreamingQueryManager, PyTrigger,
};
use types::{
    PyAnsiIntervalType, PyAnyTimeType, PyArrayType, PyAtomicType, PyBinaryType, PyBooleanType,
    PyByteType, PyCalendarIntervalType, PyCharType, PyDataType, PyDateType, PyDatetimeType,
    PyDayTimeIntervalType, PyDecimalType, PyDoubleType, PyFloatType, PyFractionalType,
    PyGeographyType, PyGeometryType, PyIntegerType, PyIntegralType, PyLongType, PyMapType,
    PyNullType, PyNumericType, PyShortType, PySpatialType, PyStringType, PyStructField,
    PyStructType, PyTimeType, PyTimestampLTZNanosType, PyTimestampNTZNanosType, PyTimestampNTZType,
    PyTimestampType, PyVarcharType, PyVariantType, PyYearMonthIntervalType,
};
use window::{PyFrameBound, PyWindow, PyWindowSpec};

#[pymodule(gil_used = false)]
fn _pyspark(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySparkSession>()?;
    m.add_class::<PyHook>()?;
    m.add_class::<PySparkSessionBuilder>()?;
    // As PySpark's nested class: `<pyspark.sql.session.SparkSession.Builder object at ...>`.
    m.getattr("Builder")?
        .setattr("__qualname__", "SparkSession.Builder")?;
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PyDataFrameWriter>()?;
    m.add_class::<PyDataFrameWriterV2>()?;
    m.add_class::<PyDataFrameNaFunctions>()?;
    m.add_class::<PyMergeIntoWriter>()?;
    m.add_class::<PyWhenMatched>()?;
    m.add_class::<PyWhenNotMatched>()?;
    m.add_class::<PyWhenNotMatchedBySource>()?;
    m.add_class::<PyColumn>()?;
    row::register(m)?;
    m.add_class::<PyGroupedData>()?;
    m.add_class::<PyCoGroupedData>()?;
    m.add_class::<PyLocalRowIterator>()?;
    m.add_class::<transport::RustConnectStub>()?;
    m.add_class::<transport::ResponseStream>()?;
    m.add("RustRpcError", m.py().get_type::<transport::RustRpcError>())?;
    m.add_class::<PyCatalog>()?;
    m.add_class::<eval_type::PyPythonEvalType>()?;
    m.add_class::<PyRuntimeConf>()?;
    m.add_class::<PyDataFrameReader>()?;
    m.add_class::<PyStatFunctions>()?;
    m.add_class::<storagelevel::PyStorageLevel>()?;

    // The core DataType JSON grammar, which `pyspark.sql.types` parses type JSON with.
    m.add_function(wrap_pyfunction!(types::parse_datatype_json, m)?)?;
    m.add_function(wrap_pyfunction!(connect_bridge::_data_type_proto, m)?)?;
    m.add_function(wrap_pyfunction!(connect_bridge::_data_type_of_proto, m)?)?;
    m.add_function(wrap_pyfunction!(column::column_reference_name, m)?)?;

    m.add_function(wrap_pyfunction!(pandas_input::_pandas_string_ipc, m)?)?;

    // Spark Declarative Pipelines (SDP) command execution
    m.add_class::<pipelines::PyPipelineRunStream>()?;
    m.add_function(wrap_pyfunction!(
        pipelines::pipeline_create_dataflow_graph,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(pipelines::pipeline_define_output, m)?)?;
    m.add_function(wrap_pyfunction!(pipelines::pipeline_define_flow, m)?)?;
    m.add_function(wrap_pyfunction!(
        pipelines::pipeline_enter_flow_function,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(pipelines::pipeline_exit_flow_function, m)?)?;
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
    m.add_class::<PyStreamingQueryManager>()?;
    m.add_class::<PyListenerEventStream>()?;

    // Resource profile classes
    m.add_class::<PyExecutorResourceRequests>()?;
    m.add_class::<PyTaskResourceRequests>()?;
    m.add_class::<PyResourceProfileBuilder>()?;
    m.add_class::<PyResourceProfile>()?;

    // Data source and profiler classes
    m.add_class::<PyDataSourceRegistration>()?;

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
    m.add_class::<PyTimestampNTZNanosType>()?;
    m.add_class::<PyTimestampLTZNanosType>()?;
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
    values::register(m)?;

    // Register Window classes
    m.add_class::<PyWindow>()?;
    m.add_class::<PyWindowSpec>()?;
    m.add_class::<PyFrameBound>()?;

    // Register TVF + Observation
    m.add_class::<tvf::PyTableValuedFunction>()?;
    m.add_class::<tablearg::PyTableArg>()?;
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
    m.add_class::<ml::PyLinearSVC>()?;
    m.add_class::<ml::PyLinearRegression>()?;
    m.add_class::<ml::PyDecisionTreeClassifier>()?;
    m.add_class::<ml::PyGBTClassifier>()?;
    m.add_class::<ml::PyRandomForestClassifier>()?;
    m.add_class::<ml::PyDecisionTreeRegressor>()?;
    m.add_class::<ml::PyRandomForestRegressor>()?;
    m.add_class::<ml::PyKMeans>()?;
    m.add_class::<ml::PyRegressionEvaluator>()?;
    m.add_class::<ml::PyBinaryClassificationEvaluator>()?;
    m.add_class::<ml::PyPipeline>()?;
    m.add_class::<ml::PyMulticlassClassificationEvaluator>()?;
    m.add_class::<ml::PyCrossValidator>()?;

    // Register functions as a submodule
    let functions_module = PyModule::new(_py, "functions")?;
    functions::register_functions(_py, &functions_module)?;
    m.add_submodule(&functions_module)?;

    // The instances of the classes with a `__dict__` release it when they are freed, which
    // PyO3 0.28 does not do (see `instance_dict`).
    instance_dict::release_instance_dicts(_py)?;

    Ok(())
}
