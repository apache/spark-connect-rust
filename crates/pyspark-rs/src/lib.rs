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
mod pbmsg;
mod pipelines;
mod plan_args;
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
mod udf_command;
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
    origin::listen_to_plan_builds();
    m.add_class::<PySparkSession>()?;
    m.add_function(wrap_pyfunction!(session::_close_sessions_at_exit, m)?)?;
    m.add_function(wrap_pyfunction!(session::_refuse_releases, m)?)?;
    session::register_release_refusal(m)?;
    m.add_function(wrap_pyfunction!(session::_install_builder, m)?)?;
    m.add_class::<PyHook>()?;
    m.add_class::<PySparkSessionBuilder>()?;
    // As PySpark Connect's classes nested in SparkSession, of which they are attributes:
    // `<pyspark.sql.connect.session.SparkSession.Builder ...>`, and a pickle names them there.
    for name in ["Builder", "Hook"] {
        m.getattr(name)?
            .setattr("__qualname__", format!("SparkSession.{name}"))?;
    }
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PyDataFrameWriter>()?;
    m.add_class::<PyDataFrameWriterV2>()?;
    m.add_class::<PyDataFrameNaFunctions>()?;
    m.add_class::<PyMergeIntoWriter>()?;
    m.add_class::<PyWhenMatched>()?;
    m.add_class::<PyWhenNotMatched>()?;
    m.add_class::<PyWhenNotMatchedBySource>()?;
    // As PySpark Connect's classes nested in MergeIntoWriter, of which they are attributes.
    for name in ["WhenMatched", "WhenNotMatched", "WhenNotMatchedBySource"] {
        m.getattr(name)?
            .setattr("__qualname__", format!("MergeIntoWriter.{name}"))?;
    }
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
    pbmsg::register(m)?;
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
    m.add_function(wrap_pyfunction!(ml::object::register_public_class, m)?)?;
    m.add_class::<ml::PyMLModel>()?;
    m.add_class::<ml::object::PyMLWriter>()?;
    m.add_class::<ml::object::PyMLReader>()?;
    m.add_class::<ml::params_api::PyFitMultipleIterator>()?;
    ml::classification_api::register(m)?;
    ml::clustering_api::register(m)?;
    ml::regression_api::register(m)?;
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
    ml::pipeline_api::register(m)?;
    ml::tuning_api::register(m)?;
    m.add_class::<ml::PyMulticlassClassificationEvaluator>()?;
    m.add_class::<ml::PyClusteringEvaluator>()?;
    m.add_class::<ml::PyMultilabelClassificationEvaluator>()?;
    m.add_class::<ml::PyRankingEvaluator>()?;

    // Register functions as a submodule
    let functions_module = PyModule::new(_py, "functions")?;
    functions::register_functions(_py, &functions_module)?;
    m.add_submodule(&functions_module)?;

    bare_type_names(m)?;
    Ok(())
}

/// Give each class of the module, and each native class outside it (`classes_outside_module`),
/// the C name of a Python class, its bare `__name__`, which CPython's messages show: "'Column'
/// object is not iterable", "unsupported operand type(s) for ^: 'bool' and 'Column'", as
/// PySpark's. PyO3 makes it the dotted path of the class's module,
/// "pyspark.sql.connect.column.Column". Setting `__name__` of a heap type sets that C name too;
/// its `__module__` and `__qualname__`, and so its repr and its pickle, stay as they are.
fn bare_type_names(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let module_classes = m
        .dict()
        .values()
        .into_iter()
        .filter_map(|value| value.cast_into::<pyo3::types::PyType>().ok());
    for cls in module_classes.chain(classes_outside_module(py)) {
        let name = cls.getattr(pyo3::intern!(py, "__name__"))?;
        // A type whose name cannot be set, as an immutable type, keeps its C name.
        let _ = cls.setattr(pyo3::intern!(py, "__name__"), name);
    }
    Ok(())
}

/// The native classes that are not attributes of the module, of objects that it makes: the
/// expression of a Column (`Column._expr`), the attribute of a param of an ML class, the exit
/// hook and the active session of the sessions (`SparkSession._active_session`), the tags and the
/// user context extensions that threads share, and the listener bus of a stream manager.
fn classes_outside_module(py: Python<'_>) -> [Bound<'_, pyo3::types::PyType>; 7] {
    [
        py.get_type::<column::PyColumnExpression>(),
        py.get_type::<ml::params_api::PyParamAttribute>(),
        py.get_type::<session::PyExitHook>(),
        py.get_type::<session::PyActiveSession>(),
        py.get_type::<session::PySharedTags>(),
        py.get_type::<session::PySharedUserContextExtensions>(),
        py.get_type::<streaming::PyStreamingQueryListenerBus>(),
    ]
}
