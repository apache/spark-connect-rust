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

// First, as the modules that take its `connect_doc!` in their attributes are those after it, of the
// textual scope of the macro.
#[macro_use]
mod connect_docs_generated;
mod catalog;
mod client;
mod client_log;
mod column;
mod conf;
mod connect_bridge;
mod dataframe;
mod datasource;
mod deliver;
mod errors;
mod functions;
mod gc;
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
mod reader_plans;
mod readwriter;
mod reduce;

mod row;
mod session;
mod stat;
mod stat_arguments;
mod state;
mod storagelevel;
mod streaming;
mod tablearg;
mod transport;
mod tvf;
mod types;
mod udf_command;
mod values;
mod window;

use catalog::PyCatalog;
use column::PyColumn;
use conf::PyRuntimeConf;
use dataframe::{
    PyDataFrame, PyDataFrameNaFunctions, PyLocalRowIterator, PyMergeIntoWriter, PyWhenMatched,
    PyWhenNotMatched, PyWhenNotMatchedBySource,
};
use datasource::PyDataSourceRegistration;
use group::{PyCoGroupedData, PyGroupedData};
pub(crate) use options::str_argument;
use session::{PyHook, PySparkSession, PySparkSessionBuilder};
use stat::PyStatFunctions;
use streaming::{PyListenerEventStream, PyStreamingQuery, PyStreamingQueryManager, PyTrigger};
use types::{
    PyAnsiIntervalType, PyAnyTimeType, PyArrayType, PyAtomicType, PyBinaryType, PyBooleanType,
    PyByteType, PyCalendarIntervalType, PyCharType, PyDataType, PyDateType, PyDatetimeType,
    PyDayTimeIntervalType, PyDecimalType, PyDoubleType, PyFloatType, PyFractionalType,
    PyGeographyType, PyGeometryType, PyIntegerType, PyIntegralType, PyLongType, PyMapType,
    PyNullType, PyNumericType, PyShortType, PySpatialType, PyStringType, PyStructField,
    PyStructType, PyTimeType, PyTimestampLTZNanosType, PyTimestampNTZNanosType, PyTimestampNTZType,
    PyTimestampType, PyVarcharType, PyVariantType, PyYearMonthIntervalType,
};
use window::{PyFrameBound, PyWindowSpec};

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

/// The bundle of root certificates of grpcio, `roots.pem` of its package, which its channels
/// trust by default, if grpcio is installed: found by `importlib.util.find_spec`, which does not
/// import it.
fn grpcio_roots_file(py: Python<'_>) -> Option<std::path::PathBuf> {
    let spec = py
        .import("importlib.util")
        .ok()?
        .call_method1("find_spec", ("grpc",))
        .ok()?;
    let origin: String = spec.getattr("origin").ok()?.extract().ok()?;
    let roots = std::path::Path::new(&origin)
        .parent()?
        .join("_cython")
        .join("_credentials")
        .join("roots.pem");
    roots.is_file().then_some(roots)
}

#[pymodule(gil_used = false)]
fn _pyspark(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // How Python's garbage collector sees the parts of plans and expressions: not at all on a
    // free-threaded build, and of the counts of the calls without the GIL of the forking thread
    // alone in the child of a fork (see `gc`).
    gc::init(_py)?;
    // The check of the clones of the parts that the tests turn on (see `gc::check_clone`).
    m.add_function(wrap_pyfunction!(gc::_check_part_clones, m)?)?;
    m.add_function(wrap_pyfunction!(gc::_part_clones_checked, m)?)?;
    m.add_function(wrap_pyfunction!(gc::_take_part_clone_violations, m)?)?;
    // The watch of the parts that the collector sees, of a test of its own (see
    // `gc::_watch_reported_parts`).
    m.add_function(wrap_pyfunction!(gc::_watch_reported_parts, m)?)?;
    origin::listen_to_plan_builds();
    // As grpcio's fork support, which registers its handlers as it loads: a child of a `fork()`
    // is of a generation of its own, whatever thread used the client first.
    spark_connect_core::fork::watch_forks();
    // As grpcio, the gRPC library of PySpark's client: the texts of the failures of the
    // connections, and the `User-Agent` of the CONNECT request of a proxy.
    spark_connect_core::set_grpcio_dialect(true);
    // As PySpark's `ChannelBuilder.userAgent`, which ends with the version of Python.
    let python_version: String = _py
        .import("platform")?
        .call_method0("python_version")?
        .extract()?;
    spark_connect_core::channel::set_user_agent_language(&format!("python/{python_version}"));
    // The bundle of roots of grpcio, which its channels trust by default, where it is installed,
    // found without importing grpc: trusted besides the roots of the system's store, so that the
    // drop-in trusts what PySpark trusts, and what the system trusts too.
    spark_connect_core::client::set_bundled_ssl_roots_file(grpcio_roots_file(_py));
    client::warn_of_spent_retries();
    client::log_grpc_errors();
    client_log::listen();
    // The requests of the finalizers of a session read the state of its client when they are
    // sent, as PySpark's finalizers build their requests when they run.
    spark_connect::session::set_finalizer_request_hook(
        spark_connect::session::FinalizerRequestHook {
            before: interrupt::before_finalizer_request,
            after: interrupt::after_finalizer_request,
        },
    );
    // So do the requests of the failure of a call, of the details of its error and of the configs
    // of its stack trace, as PySpark's `_handle_error` builds them.
    spark_connect_core::client::set_failure_requests_hook(interrupt::before_failure_requests);
    m.add_function(wrap_pyfunction!(deliver::_stop_deliveries, m)?)?;
    deliver::stop_at_exit(m)?;
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
    column::install_getattr(m)?;
    row::register(m)?;
    m.add_class::<PyGroupedData>()?;
    m.add_class::<PyCoGroupedData>()?;
    m.add_class::<PyLocalRowIterator>()?;
    m.add_class::<client::PyExecutePlanResponses>()?;
    m.add_class::<transport::RustConnectStub>()?;
    m.add_class::<transport::ResponseStream>()?;
    m.add("RustRpcError", m.py().get_type::<transport::RustRpcError>())?;
    m.add_class::<PyCatalog>()?;
    m.add_class::<PyRuntimeConf>()?;
    // The plans of the reads of PySpark's readers (see `reader_plans`).
    m.add_class::<reader_plans::PyReadPlan>()?;
    m.add_function(wrap_pyfunction!(reader_plans::data_source, m)?)?;
    m.add_function(wrap_pyfunction!(reader_plans::read_table, m)?)?;
    m.add_function(wrap_pyfunction!(reader_plans::relation_changes, m)?)?;
    m.add_function(wrap_pyfunction!(reader_plans::parse, m)?)?;
    m.add_function(wrap_pyfunction!(reader_plans::read_dataframe, m)?)?;
    m.add_class::<PyStatFunctions>()?;
    m.add_class::<storagelevel::PyStorageLevel>()?;

    // The core DataType JSON grammar, which `pyspark.sql.types` parses type JSON with.
    pbmsg::register(m)?;
    m.add_function(wrap_pyfunction!(types::parse_datatype_json, m)?)?;
    m.add_function(wrap_pyfunction!(connect_bridge::_data_type_proto, m)?)?;
    m.add_function(wrap_pyfunction!(connect_bridge::_data_type_of_proto, m)?)?;
    m.add_function(wrap_pyfunction!(connect_bridge::_fresh_plan_id, m)?)?;
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
    m.add_function(wrap_pyfunction!(streaming::post_query_started, m)?)?;
    m.add_class::<PyTrigger>()?;
    m.add_class::<PyStreamingQuery>()?;
    m.add_class::<PyStreamingQueryManager>()?;
    m.add_class::<streaming::PyStreamingQueryListenerBus>()?;
    streaming::keep_native_post_to_all(m.py())?;
    m.add_class::<PyListenerEventStream>()?;

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
    // The attributes that PySpark's `jsonValue` reads of the native classes, as they are made,
    // before any patch of them.
    types::keep_native_json_reads(_py)?;
    values::register(m)?;

    // Register Window classes
    m.add_function(wrap_pyfunction!(window::empty_window_spec, m)?)?;
    m.add_class::<PyWindowSpec>()?;
    m.add_class::<PyFrameBound>()?;

    // Register TVF + Observation
    m.add_class::<tvf::PyTableValuedFunction>()?;
    m.add_class::<tablearg::PyTableArg>()?;

    m.add_class::<observation::PyObservation>()?;

    // Register ML classes (pyspark.ml.connect)
    m.add_function(wrap_pyfunction!(ml::object::register_public_class, m)?)?;
    m.add_function(wrap_pyfunction!(ml::object::ml_kind, m)?)?;
    m.add_function(wrap_pyfunction!(ml::object::ml_params, m)?)?;
    m.add_function(wrap_pyfunction!(ml::object::ml_sync, m)?)?;
    m.add_function(wrap_pyfunction!(ml::object::ml_java_obj, m)?)?;
    m.add_function(wrap_pyfunction!(ml::object::ml_uid, m)?)?;
    m.add_function(wrap_pyfunction!(ml::object::ml_jvm_class, m)?)?;
    m.add_function(wrap_pyfunction!(ml::object::ml_finalize, m)?)?;
    m.add_function(wrap_pyfunction!(ml::object::ml_copy, m)?)?;
    m.add_function(wrap_pyfunction!(ml::object::ml_make_java_obj, m)?)?;
    // The watch of the traverses of the models, of a test of its own (see
    // `ml::object::ml_watch_traverses`).
    m.add_function(wrap_pyfunction!(ml::object::ml_watch_traverses, m)?)?;
    ml::object::register_class::<ml::PyMLModel>(m)?;
    ml::object::register_class::<ml::object::PyMLWriter>(m)?;
    ml::object::register_class::<ml::object::PyMLReader>(m)?;
    m.add_function(wrap_pyfunction!(ml::object::new_remote_ml_writer, m)?)?;
    m.add_function(wrap_pyfunction!(ml::object::new_remote_ml_reader, m)?)?;
    ml::object::register_class::<ml::params_api::PyFitMultipleIterator>(m)?;
    ml::classification_api::register(m)?;
    ml::clustering_api::register(m)?;
    ml::regression_api::register(m)?;
    ml::feature::register(m)?;
    ml::stat::register(m)?;
    ml::fpm::register(m)?;
    ml::object::register_class::<ml::PyLogisticRegression>(m)?;
    ml::object::register_class::<ml::PyLinearSVC>(m)?;
    ml::object::register_class::<ml::PyFMClassifier>(m)?;
    ml::object::register_class::<ml::PyMultilayerPerceptronClassifier>(m)?;
    ml::object::register_class::<ml::PyOneVsRest>(m)?;

    ml::object::register_class::<ml::PyLinearRegression>(m)?;
    ml::object::register_class::<ml::PyAFTSurvivalRegression>(m)?;
    ml::object::register_class::<ml::PyIsotonicRegression>(m)?;
    ml::object::register_class::<ml::PyFMRegressor>(m)?;
    ml::object::register_class::<ml::PyGeneralizedLinearRegression>(m)?;
    ml::object::register_class::<ml::PyDecisionTreeClassifier>(m)?;
    ml::object::register_class::<ml::PyNaiveBayes>(m)?;
    ml::object::register_class::<ml::PyGBTClassifier>(m)?;
    ml::object::register_class::<ml::PyGBTRegressor>(m)?;
    ml::object::register_class::<ml::PyRandomForestClassifier>(m)?;
    ml::object::register_class::<ml::PyDecisionTreeRegressor>(m)?;
    ml::object::register_class::<ml::PyRandomForestRegressor>(m)?;
    ml::object::register_class::<ml::PyLDA>(m)?;
    ml::object::register_class::<ml::PyPowerIterationClustering>(m)?;
    ml::object::register_class::<ml::PyKMeans>(m)?;
    ml::object::register_class::<ml::PyGaussianMixture>(m)?;
    ml::object::register_class::<ml::PyBisectingKMeans>(m)?;
    ml::recommendation::register(m)?;
    ml::object::register_class::<ml::PyRegressionEvaluator>(m)?;
    ml::object::register_class::<ml::PyBinaryClassificationEvaluator>(m)?;
    ml::pipeline_api::register(m)?;
    ml::tuning_api::register(m)?;
    ml::object::register_class::<ml::PyMulticlassClassificationEvaluator>(m)?;
    ml::object::register_class::<ml::PyClusteringEvaluator>(m)?;
    ml::object::register_class::<ml::PyMultilabelClassificationEvaluator>(m)?;
    ml::object::register_class::<ml::PyRankingEvaluator>(m)?;

    // Register functions as a submodule
    let functions_module = PyModule::new(_py, "functions")?;
    functions::register_functions(_py, &functions_module)?;
    m.add_submodule(&functions_module)?;

    bare_type_names(m)?;
    for cls in classes_of_no_docstring(m.py()) {
        cls.setattr(pyo3::intern!(m.py(), "__doc__"), m.py().None())?;
    }
    Ok(())
}

/// The native classes of PySpark Connect's classes of no docstring that carry the text signature
/// of PySpark's `__init__` (`text_signature` of their `#[new]`), whose `__doc__` CPython makes the
/// empty str of the rest of that text: None, as PySpark's. Their `__text_signature__`, of the text,
/// stays, which `inspect.signature` reads.
fn classes_of_no_docstring(py: Python<'_>) -> [Bound<'_, pyo3::types::PyType>; 2] {
    [
        py.get_type::<PyStreamingQuery>(),
        py.get_type::<PyStreamingQueryManager>(),
    ]
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
/// hook and the active session of the sessions (`SparkSession._active_session`), the handle of a
/// session that a client holds, and the objects that the garbage collector sees of the parts of
/// plans and expressions and of the states of the writers (see `gc`).
fn classes_outside_module(py: Python<'_>) -> [Bound<'_, pyo3::types::PyType>; 7] {
    [
        py.get_type::<column::PyColumnExpression>(),
        py.get_type::<ml::params_api::PyParamAttribute>(),
        py.get_type::<session::PyExitHook>(),
        py.get_type::<session::PyActiveSession>(),
        py.get_type::<session::PyWeakSession>(),
        py.get_type::<gc::SharedPart>(),
        py.get_type::<dataframe::WriterParts>(),
    ]
}
