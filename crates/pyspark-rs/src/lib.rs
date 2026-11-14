//! PyO3 bindings for the Rust Spark Connect client.
//!
//! `non_snake_case` is allowed crate-wide: these classes deliberately expose PySpark's
//! camelCase public API (e.g. `toPandas`, `withColumn`, `containsNull`), so the Rust method,
//! field, and argument names must match the Python names verbatim — renaming them would change
//! the user-facing API.
//!
//! Warnings are errors for this crate in every build: CI builds it via several workflows that
//! don't all set `RUSTFLAGS=-D warnings`.
#![deny(warnings)]
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
mod reduce;

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
pub(crate) use options::{coerce_option_value, str_argument, str_text};
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

/// The stack that the encode, the decode and the drop of a protobuf of a pickled Column,
/// DataFrame or hook request take on the thread that unpickles or sends it. Prost recurses once
/// per level of a message, with frames of about 1 KiB in a release build and more in a debug
/// one, and a deep message has at most `MAX_PLAN_DEPTH` levels (the client refuses more), so this
/// holds them with room to spare, on a thread of 512 KiB as well.
const PROTO_STACK: usize = if cfg!(debug_assertions) {
    16 << 20
} else {
    4 << 20
};

/// Run `f`, which encodes, decodes or drops a deep protobuf, on a stack that holds its levels.
pub(crate) fn on_grown_stack<T>(f: impl FnOnce() -> T) -> T {
    stacker::maybe_grow(PROTO_STACK, PROTO_STACK, f)
}

/// The bytes of `message`, encoded and dropped on a stack that holds its levels.
pub(crate) fn encode_on_grown_stack<M: prost::Message + Send>(message: M) -> Vec<u8> {
    on_grown_stack(move || message.encode_to_vec())
}

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
    m.add_class::<ml::object::PyMLWriter>()?;
    m.add_class::<ml::object::PyMLReader>()?;
    m.add_class::<ml::params_api::PyFitMultipleIterator>()?;
    ml::classification_api::register(m)?;
    ml::clustering_api::register(m)?;
    ml::feature::register(m)?;
    ml::stat::register(m)?;
    ml::fpm::register(m)?;
    m.add_class::<ml::PyLogisticRegression>()?;
    m.add_class::<ml::PyLinearSVC>()?;
    m.add_class::<ml::PyFMClassifier>()?;
    m.add_class::<ml::PyMultilayerPerceptronClassifier>()?;
    m.add_class::<ml::PyOneVsRest>()?;

    m.add_class::<ml::PyLinearRegression>()?;
    m.add_class::<ml::PyAFTSurvivalRegression>()?;
    m.add_class::<ml::PyIsotonicRegression>()?;
    m.add_class::<ml::PyFMRegressor>()?;
    m.add_class::<ml::PyGeneralizedLinearRegression>()?;
    m.add_class::<ml::PyDecisionTreeClassifier>()?;
    m.add_class::<ml::PyNaiveBayes>()?;
    m.add_class::<ml::PyGBTClassifier>()?;
    m.add_class::<ml::PyGBTRegressor>()?;
    m.add_class::<ml::PyRandomForestClassifier>()?;
    m.add_class::<ml::PyDecisionTreeRegressor>()?;
    m.add_class::<ml::PyRandomForestRegressor>()?;
    m.add_class::<ml::PyLDA>()?;
    m.add_class::<ml::PyPowerIterationClustering>()?;
    m.add_class::<ml::PyKMeans>()?;
    m.add_class::<ml::PyGaussianMixture>()?;
    m.add_class::<ml::PyBisectingKMeans>()?;
    ml::recommendation::register(m)?;
    m.add_class::<ml::PyRegressionEvaluator>()?;
    m.add_class::<ml::PyBinaryClassificationEvaluator>()?;
    m.add_class::<ml::PyPipeline>()?;
    m.add_class::<ml::PyMulticlassClassificationEvaluator>()?;
    m.add_class::<ml::PyClusteringEvaluator>()?;
    m.add_class::<ml::PyMultilabelClassificationEvaluator>()?;
    m.add_class::<ml::PyRankingEvaluator>()?;
    m.add_class::<ml::PyCrossValidator>()?;

    // Register functions as a submodule
    let functions_module = PyModule::new(_py, "functions")?;
    functions::register_functions(_py, &functions_module)?;
    m.add_submodule(&functions_module)?;

    Ok(())
}
