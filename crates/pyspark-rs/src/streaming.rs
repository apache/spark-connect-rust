//! PyO3 wrappers for Structured Streaming.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use spark_connect::streaming::{
    DataStreamReader, DataStreamWriter, ListenerEventStream, StreamingQuery,
    StreamingQueryListener, StreamingQueryListenerEvent, StreamingQueryManager, Trigger,
};
use spark_connect::udf::PythonUDFPayload;
use spark_connect_core::error::{Result as SparkResult, SparkError};
use std::collections::HashMap;
use std::sync::Arc;

use crate::dataframe::{PyDataFrame, WriterState};
use crate::errors::spark_error_to_py_exception;
use crate::interrupt::{interruptible, interruptible_outcome};
use crate::options::{set_option_skip_none, set_options_skip_none};

/// Adapts Python listeners to the native bus: deserialize one typed event for the whole
/// snapshot, then dispatch it through Python's callback-exception isolation helper. Decode
/// errors return to the bus, and terminal failures produce the reference handler warning.
struct PyListenerAdapter {
    listener: Py<PyAny>,
}

/// An unisolated dispatch failure. Keep Python errors intact so synchronous starts
/// can re-raise their original type; native listener errors remain typed SparkErrors.
enum ListenerDispatchError {
    Python(PyErr),
    Native(Box<SparkError>),
}

impl From<PyErr> for ListenerDispatchError {
    fn from(error: PyErr) -> Self {
        Self::Python(error)
    }
}

/// Decode before any callback, sharing one Python object across every registration.
/// The same path handles synchronous starts and asynchronous protocol events, including
/// mixed native/Python snapshots. Keep native errors typed until the reporting boundary.
fn post_listener_event<'a>(
    py: Python<'_>,
    listeners: impl IntoIterator<Item = &'a dyn StreamingQueryListener>,
    event: &StreamingQueryListenerEvent,
    started: bool,
) -> std::result::Result<(), ListenerDispatchError> {
    let module = py.import("pyspark.sql.streaming.query")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("started", started)?;
    let decoded = module.call_method(
        "_deserialize_listener_event",
        (event.event_type, event.event_json.as_str()),
        Some(&kwargs),
    )?;
    for listener in listeners {
        match listener
            .as_any()
            .and_then(|any| any.downcast_ref::<PyListenerAdapter>())
        {
            Some(adapter) => {
                module.call_method1(
                    "_dispatch_listener_event",
                    (adapter.listener.bind(py), &decoded),
                )?;
            }
            // Native callbacks keep their usual ability to wait for Python work.
            None if started => py.detach(|| listener.on_event(event)),
            None => py
                .detach(|| listener.try_on_event(event).map_err(Box::new))
                .map_err(ListenerDispatchError::Native)?,
        }
    }
    Ok(())
}

/// Any Python exception not isolated by the helper terminates the subscription: this
/// includes decoding errors, callback BaseException, and warnings promoted to errors.
/// Async reporting keeps only str(), without allocating an unconsumed callback-registry id.
fn async_dispatch_error(py: Python<'_>, error: ListenerDispatchError) -> Box<SparkError> {
    match error {
        ListenerDispatchError::Python(error) => {
            Box::new(SparkError::connect_msg(error.value(py).to_string()))
        }
        ListenerDispatchError::Native(error) => error,
    }
}

impl StreamingQueryListener for PyListenerAdapter {
    fn on_event(&self, event: &StreamingQueryListenerEvent) {
        // This infallible entry point is also used by the core's default synchronous
        // start handler. Python writer calls use post_started_event to propagate errors.
        Python::attach(|py| {
            let _ = post_listener_event(
                py,
                std::iter::once(self as &dyn StreamingQueryListener),
                event,
                true,
            );
        });
    }

    fn try_on_event(&self, event: &StreamingQueryListenerEvent) -> SparkResult<()> {
        Python::attach(|py| {
            post_listener_event(
                py,
                std::iter::once(self as &dyn StreamingQueryListener),
                event,
                false,
            )
            .map_err(|error| async_dispatch_error(py, error))
        })
        .map_err(|error| *error)
    }

    fn try_post_to_all(
        &self,
        listeners: &[Arc<dyn StreamingQueryListener>],
        event: &StreamingQueryListenerEvent,
    ) -> Option<SparkResult<()>> {
        Some(
            Python::attach(|py| {
                post_listener_event(py, listeners.iter().map(Arc::as_ref), event, false)
                    .map_err(|error| async_dispatch_error(py, error))
            })
            .map_err(|error| *error),
        )
    }

    fn on_stream_error(&self, error: &SparkError) -> bool {
        Python::attach(|py| {
            let message = crate::errors::spark_error_to_py_exception(error.clone())
                .value(py)
                .str()
                .and_then(|text| Ok(text.to_cow()?.into_owned()))
                .unwrap_or_else(|_| error.message());
            let _ = warn(
                py,
                format!(
                    "StreamingQueryListenerBus Handler thread received exception, all client side \
                     listeners are removed and handler thread is terminated. The error is: {}",
                    message
                ),
            );
        });
        true
    }

    /// The adapter, of which `StreamingQueryManager._sqlb` gives back the Python listener.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// Parse a synchronous started event once, including when there are no listeners, and
/// dispatch its typed object before start/toTable returns. Preserve the original Python
/// decoding exception across the detached core call.
#[allow(clippy::result_large_err)] // The core started-event handler requires SparkResult.
fn post_started_event(
    manager: &StreamingQueryManager,
    event: &StreamingQueryListenerEvent,
) -> SparkResult<()> {
    Python::attach(|py| {
        let listeners = manager.listeners();
        post_listener_event(py, listeners.iter().map(Arc::as_ref), event, true).map_err(|error| {
            match error {
                ListenerDispatchError::Python(error) => {
                    Box::new(crate::errors::callback_error(error))
                }
                ListenerDispatchError::Native(error) => error,
            }
        })
    })
    .map_err(|error| *error)
}

/// Apply a `schema` argument (a StructType or a DDL string) to a streaming reader, as the
/// batch reader does.
fn apply_stream_schema(r: DataStreamReader, v: &Bound<'_, PyAny>) -> PyResult<DataStreamReader> {
    Ok(r.schema(crate::readwriter::reader_schema(v)?))
}

/// The path of `DataStreamReader.load`, as PySpark checks it: a non-empty str, else
/// `VALUE_NOT_NON_EMPTY_STR` of its `str()`.
fn load_path(path: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(text) = path.cast::<pyo3::types::PyString>() {
        let text = text.to_cow()?;
        if !text.trim().is_empty() {
            return Ok(text.into_owned());
        }
    }
    let shown = path.str()?.to_cow()?.into_owned();
    Err(crate::column::raise_pyspark(
        path.py(),
        "PySparkValueError",
        "VALUE_NOT_NON_EMPTY_STR",
        &[("arg_name", "path"), ("arg_value", &shown)],
    ))
}

/// The path of a format reader of `DataStreamReader`, such as `json`, as PySpark's takes it: a
/// str, else `NOT_EXPECTED_TYPE` of `path`, which it then loads as `load` does.
fn format_path(path: &Bound<'_, PyAny>) -> PyResult<String> {
    if !path.is_instance_of::<pyo3::types::PyString>() {
        return Err(crate::stat_arguments::not_expected_type(
            path, "path", "str",
        ));
    }
    load_path(path)
}

/// Coerce a `partitionBy` argument (a single column name or a list of names) to a
/// column-name vector, mirroring PySpark's `str | list[str]` acceptance.
fn extract_partition_cols(v: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(s) = v.extract::<String>() {
        return Ok(vec![s]);
    }
    v.extract::<Vec<String>>()
}

/// The column names of `partitionBy(*cols)` and `clusterBy(*cols)`, as PySpark's take them:
/// the names, or a list or tuple of them alone.
fn column_names(cols: &Bound<'_, PyTuple>) -> PyResult<Vec<String>> {
    if cols.len() == 1 {
        let only = cols.get_item(0)?;
        if only.is_instance_of::<pyo3::types::PyList>() || only.is_instance_of::<PyTuple>() {
            return only.extract();
        }
    }
    cols.extract()
}

/// The name of a query, as PySpark's `queryName` checks it: a non-empty str, else
/// `VALUE_NOT_NON_EMPTY_STR` of its `str()`.
fn query_name_of(name: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(text) = name.cast::<pyo3::types::PyString>() {
        let text = text.to_cow()?;
        if !text.trim().is_empty() {
            return Ok(text.into_owned());
        }
    }
    let shown = name.str()?.to_cow()?.into_owned();
    Err(crate::column::raise_pyspark(
        name.py(),
        "PySparkValueError",
        "VALUE_NOT_NON_EMPTY_STR",
        &[("arg_name", "queryName"), ("arg_value", &shown)],
    ))
}

/// Python wrapper for DataStreamReader.
///
/// Like PySpark's reader, the setters change this object and return it, and the
/// reader can be used again after a read: the core reader is a consuming builder, so
/// the setters replace the wrapped value in place and the reads work on a clone.
#[pyclass(
    name = "DataStreamReader",
    module = "pyspark.sql.streaming.readwriter",
    weakref,
    dict
)]
pub struct PyDataStreamReader {
    inner: DataStreamReader,
}

impl PyDataStreamReader {
    pub fn new(reader: DataStreamReader) -> Self {
        PyDataStreamReader { inner: reader }
    }
}

#[pymethods]
impl PyDataStreamReader {
    /// Set the schema, if any, and the options of `options` that are not None, through its own
    /// `schema` and `option`, as PySpark's `OptionUtils._set_opts`, which pandas-on-Spark calls.
    #[pyo3(signature = (schema=None, **options))]
    fn _set_opts(
        slf: &Bound<'_, Self>,
        schema: Option<Bound<'_, PyAny>>,
        options: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        crate::options::set_opts(slf.as_any(), schema.as_ref(), options.as_ref())
    }

    /// Set the source format. As PySpark's, None unsets it, so that the server reads its
    /// default source.
    fn format<'py>(
        mut slf: PyRefMut<'py, Self>,
        source: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner = match source {
            Some(source) => slf.inner.clone().format(source),
            None => slf.inner.clone().clear_format(),
        };
        Ok(slf)
    }

    /// Set the schema: a StructType or a DDL string, as in pyspark.
    fn schema<'py>(
        mut slf: PyRefMut<'py, Self>,
        schema: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let schema = crate::readwriter::reader_schema(schema)?;
        slf.inner = slf.inner.clone().schema(&schema);
        Ok(slf)
    }

    fn option<'py>(
        mut slf: PyRefMut<'py, Self>,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        // None -> option left unset; bools -> "true"/"false" (reference `to_str`).
        match crate::coerce_option_value(value)? {
            Some(v) => {
                slf.inner = slf.inner.clone().option(key, &v);
                Ok(slf)
            }
            None => Ok(slf),
        }
    }

    // Mirrors reference `DataStreamReader.options(**options)`: keyword args; None values
    // skipped, booleans lowercased.
    #[pyo3(signature = (**options))]
    fn options<'py>(
        mut slf: PyRefMut<'py, Self>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let opts = crate::options::kwargs_skip_none(options)?;
        slf.inner = slf.inner.clone().options(opts);
        Ok(slf)
    }

    /// The configured source format (PySpark Connect's `DataStreamReader._format`).
    #[getter]
    fn _format(&self) -> Option<String> {
        self.inner.get_format().map(str::to_string)
    }

    /// The configured schema string, empty when unset (`DataStreamReader._schema`).
    #[getter]
    fn _schema(&self) -> String {
        self.inner.get_schema().to_string()
    }

    /// The configured read options (`DataStreamReader._options`); a copy.
    #[getter]
    fn _options(&self) -> HashMap<String, String> {
        self.inner.get_options().clone()
    }

    #[pyo3(signature = (path=None, format=None, schema=None, **options))]
    fn load(
        &mut self,
        path: Option<&Bound<'_, PyAny>>,
        format: Option<&str>,
        schema: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        if let Some(fmt) = format {
            r = r.format(fmt);
        }
        if let Some(v) = schema {
            r = apply_stream_schema(r, v)?;
        }
        if options.is_some() {
            let opts = crate::options::kwargs_skip_none(options)?;
            r = r.options(opts);
        }
        // PySpark's one-shot load sets format/schema/options on the reader itself.
        self.inner = r.clone();
        let path = path.map(load_path).transpose()?;
        Ok(PyDataFrame::new(r.load(path.as_deref())))
    }

    /// Set the source name (for checkpoint stability). Mirrors `DataStreamReader.name`,
    /// which raises `NOT_EXPECTED_TYPE` for a name that is not a str, and
    /// `INVALID_STREAMING_SOURCE_NAME` for one of other characters than ASCII letters, digits
    /// and underscores.
    fn name<'py>(
        mut slf: PyRefMut<'py, Self>,
        source_name: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let Ok(name) = source_name.cast::<pyo3::types::PyString>() else {
            return Err(crate::stat_arguments::not_expected_type(
                source_name,
                "source_name",
                "str",
            ));
        };
        let name = name.to_cow()?;
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(crate::column::raise_pyspark(
                source_name.py(),
                "PySparkValueError",
                "INVALID_STREAMING_SOURCE_NAME",
                &[("source_name", &name)],
            ));
        }
        slf.inner = slf.inner.clone().name(&name);
        Ok(slf)
    }

    /// Read the streaming CDC changes of a named table. Mirrors `DataStreamReader.changes`,
    /// which raises an `AnalysisException` of `_LEGACY_ERROR_TEMP_1189` for a reader of a
    /// schema.
    fn changes(&self, py: Python<'_>, tableName: &str) -> PyResult<PyDataFrame> {
        if !self.inner.get_schema().is_empty() {
            return Err(crate::readwriter::changes_of_schema_error(py));
        }
        Ok(PyDataFrame::new(self.inner.clone().changes(tableName)))
    }

    /// Load an XML streaming source. Mirrors `DataStreamReader.xml(...)`: set each named
    /// option, then `format("xml").load(path)`.
    #[pyo3(signature = (path, rowTag=None, schema=None, excludeAttribute=None, attributePrefix=None, valueTag=None, ignoreSurroundingSpaces=None, rowValidationXSDPath=None, ignoreNamespace=None, wildcardColName=None, encoding=None, inferSchema=None, nullValue=None, dateFormat=None, timestampFormat=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, samplingRatio=None, locale=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn xml(
        &mut self,
        path: &Bound<'_, PyAny>,
        rowTag: Option<&Bound<'_, PyAny>>,
        schema: Option<&Bound<'_, PyAny>>,
        excludeAttribute: Option<&Bound<'_, PyAny>>,
        attributePrefix: Option<&Bound<'_, PyAny>>,
        valueTag: Option<&Bound<'_, PyAny>>,
        ignoreSurroundingSpaces: Option<&Bound<'_, PyAny>>,
        rowValidationXSDPath: Option<&Bound<'_, PyAny>>,
        ignoreNamespace: Option<&Bound<'_, PyAny>>,
        wildcardColName: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        inferSchema: Option<&Bound<'_, PyAny>>,
        nullValue: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        columnNameOfCorruptRecord: Option<&Bound<'_, PyAny>>,
        multiLine: Option<&Bound<'_, PyAny>>,
        samplingRatio: Option<&Bound<'_, PyAny>>,
        locale: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone().format("xml");
        r = set_option_skip_none(r, "rowTag", rowTag)?;
        if let Some(v) = schema {
            r = apply_stream_schema(r, v)?;
        }
        r = set_options_skip_none(
            r,
            &[
                ("excludeAttribute", excludeAttribute),
                ("attributePrefix", attributePrefix),
                ("valueTag", valueTag),
                ("ignoreSurroundingSpaces", ignoreSurroundingSpaces),
                ("rowValidationXSDPath", rowValidationXSDPath),
                ("ignoreNamespace", ignoreNamespace),
                ("wildcardColName", wildcardColName),
                ("encoding", encoding),
                ("inferSchema", inferSchema),
                ("nullValue", nullValue),
                ("dateFormat", dateFormat),
                ("timestampFormat", timestampFormat),
                ("mode", mode),
                ("columnNameOfCorruptRecord", columnNameOfCorruptRecord),
                ("multiLine", multiLine),
                ("samplingRatio", samplingRatio),
                ("locale", locale),
            ],
        )?;
        // Like PySpark, the options and the xml format stay on the reader.
        self.inner = r.clone();
        let path = format_path(path)?;
        Ok(PyDataFrame::new(r.load(Some(&path))))
    }

    fn table(&self, tableName: &str) -> PyResult<PyDataFrame> {
        let df = self.inner.clone().table(tableName);
        Ok(PyDataFrame::new(df))
    }

    #[pyo3(signature = (path, schema=None, primitivesAsString=None, prefersDecimal=None, allowComments=None, allowUnquotedFieldNames=None, allowSingleQuotes=None, allowNumericLeadingZero=None, allowBackslashEscapingAnyCharacter=None, mode=None, columnNameOfCorruptRecord=None, dateFormat=None, timestampFormat=None, multiLine=None, allowUnquotedControlChars=None, lineSep=None, locale=None, dropFieldIfAllNull=None, encoding=None, pathGlobFilter=None, recursiveFileLookup=None, allowNonNumericNumbers=None, useUnsafeRow=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn json(
        &mut self,
        path: &Bound<'_, PyAny>,
        schema: Option<&Bound<'_, PyAny>>,
        primitivesAsString: Option<&Bound<'_, PyAny>>,
        prefersDecimal: Option<&Bound<'_, PyAny>>,
        allowComments: Option<&Bound<'_, PyAny>>,
        allowUnquotedFieldNames: Option<&Bound<'_, PyAny>>,
        allowSingleQuotes: Option<&Bound<'_, PyAny>>,
        allowNumericLeadingZero: Option<&Bound<'_, PyAny>>,
        allowBackslashEscapingAnyCharacter: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        columnNameOfCorruptRecord: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        multiLine: Option<&Bound<'_, PyAny>>,
        allowUnquotedControlChars: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        locale: Option<&Bound<'_, PyAny>>,
        dropFieldIfAllNull: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        allowNonNumericNumbers: Option<&Bound<'_, PyAny>>,
        useUnsafeRow: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        if let Some(v) = schema {
            r = apply_stream_schema(r, v)?;
        }
        r = set_options_skip_none(
            r,
            &[
                ("primitivesAsString", primitivesAsString),
                ("prefersDecimal", prefersDecimal),
                ("allowComments", allowComments),
                ("allowUnquotedFieldNames", allowUnquotedFieldNames),
                ("allowSingleQuotes", allowSingleQuotes),
                ("allowNumericLeadingZero", allowNumericLeadingZero),
                (
                    "allowBackslashEscapingAnyCharacter",
                    allowBackslashEscapingAnyCharacter,
                ),
                ("mode", mode),
                ("columnNameOfCorruptRecord", columnNameOfCorruptRecord),
                ("dateFormat", dateFormat),
                ("timestampFormat", timestampFormat),
                ("multiLine", multiLine),
                ("allowUnquotedControlChars", allowUnquotedControlChars),
                ("lineSep", lineSep),
                ("locale", locale),
                ("dropFieldIfAllNull", dropFieldIfAllNull),
                ("encoding", encoding),
                ("pathGlobFilter", pathGlobFilter),
                ("recursiveFileLookup", recursiveFileLookup),
                ("allowNonNumericNumbers", allowNonNumericNumbers),
                ("useUnsafeRow", useUnsafeRow),
            ],
        )?;
        // Like PySpark, the options and the format stay on the reader.
        self.inner = r.clone().format("json");
        let path = format_path(path)?;
        Ok(PyDataFrame::new(r.json(&path)))
    }

    #[pyo3(signature = (path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None, datetimeRebaseMode=None, int96RebaseMode=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn parquet(
        &mut self,
        path: &Bound<'_, PyAny>,
        mergeSchema: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        datetimeRebaseMode: Option<&Bound<'_, PyAny>>,
        int96RebaseMode: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        r = set_options_skip_none(
            r,
            &[
                ("mergeSchema", mergeSchema),
                ("pathGlobFilter", pathGlobFilter),
                ("recursiveFileLookup", recursiveFileLookup),
                ("datetimeRebaseMode", datetimeRebaseMode),
                ("int96RebaseMode", int96RebaseMode),
            ],
        )?;
        // Like PySpark, the options and the format stay on the reader.
        self.inner = r.clone().format("parquet");
        let path = format_path(path)?;
        Ok(PyDataFrame::new(r.parquet(&path)))
    }

    #[pyo3(signature = (path, schema=None, sep=None, encoding=None, quote=None, escape=None, comment=None, header=None, inferSchema=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, maxMalformedLogPerPartition=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, charToEscapeQuoteEscaping=None, enforceSchema=None, emptyValue=None, locale=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, unescapedQuoteHandling=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn csv(
        &mut self,
        path: &Bound<'_, PyAny>,
        schema: Option<&Bound<'_, PyAny>>,
        sep: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        quote: Option<&Bound<'_, PyAny>>,
        escape: Option<&Bound<'_, PyAny>>,
        comment: Option<&Bound<'_, PyAny>>,
        header: Option<&Bound<'_, PyAny>>,
        inferSchema: Option<&Bound<'_, PyAny>>,
        ignoreLeadingWhiteSpace: Option<&Bound<'_, PyAny>>,
        ignoreTrailingWhiteSpace: Option<&Bound<'_, PyAny>>,
        nullValue: Option<&Bound<'_, PyAny>>,
        nanValue: Option<&Bound<'_, PyAny>>,
        positiveInf: Option<&Bound<'_, PyAny>>,
        negativeInf: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        maxColumns: Option<&Bound<'_, PyAny>>,
        maxCharsPerColumn: Option<&Bound<'_, PyAny>>,
        maxMalformedLogPerPartition: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        columnNameOfCorruptRecord: Option<&Bound<'_, PyAny>>,
        multiLine: Option<&Bound<'_, PyAny>>,
        charToEscapeQuoteEscaping: Option<&Bound<'_, PyAny>>,
        enforceSchema: Option<&Bound<'_, PyAny>>,
        emptyValue: Option<&Bound<'_, PyAny>>,
        locale: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        unescapedQuoteHandling: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        if let Some(v) = schema {
            r = apply_stream_schema(r, v)?;
        }
        r = set_options_skip_none(
            r,
            &[
                ("sep", sep),
                ("encoding", encoding),
                ("quote", quote),
                ("escape", escape),
                ("comment", comment),
                ("header", header),
                ("inferSchema", inferSchema),
                ("ignoreLeadingWhiteSpace", ignoreLeadingWhiteSpace),
                ("ignoreTrailingWhiteSpace", ignoreTrailingWhiteSpace),
                ("nullValue", nullValue),
                ("nanValue", nanValue),
                ("positiveInf", positiveInf),
                ("negativeInf", negativeInf),
                ("dateFormat", dateFormat),
                ("timestampFormat", timestampFormat),
                ("maxColumns", maxColumns),
                ("maxCharsPerColumn", maxCharsPerColumn),
                ("maxMalformedLogPerPartition", maxMalformedLogPerPartition),
                ("mode", mode),
                ("columnNameOfCorruptRecord", columnNameOfCorruptRecord),
                ("multiLine", multiLine),
                ("charToEscapeQuoteEscaping", charToEscapeQuoteEscaping),
                ("enforceSchema", enforceSchema),
                ("emptyValue", emptyValue),
                ("locale", locale),
                ("lineSep", lineSep),
                ("pathGlobFilter", pathGlobFilter),
                ("recursiveFileLookup", recursiveFileLookup),
                ("unescapedQuoteHandling", unescapedQuoteHandling),
            ],
        )?;
        // Like PySpark, the options and the format stay on the reader.
        self.inner = r.clone().format("csv");
        let path = format_path(path)?;
        Ok(PyDataFrame::new(r.csv(&path)))
    }

    #[pyo3(signature = (path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None))]
    #[allow(non_snake_case)]
    fn orc(
        &mut self,
        path: &Bound<'_, PyAny>,
        mergeSchema: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        r = set_options_skip_none(
            r,
            &[
                ("mergeSchema", mergeSchema),
                ("pathGlobFilter", pathGlobFilter),
                ("recursiveFileLookup", recursiveFileLookup),
            ],
        )?;
        // Like PySpark, the options and the format stay on the reader.
        self.inner = r.clone().format("orc");
        let path = format_path(path)?;
        Ok(PyDataFrame::new(r.orc(&path)))
    }

    #[pyo3(signature = (path, wholetext=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None))]
    #[allow(non_snake_case)]
    fn text(
        &mut self,
        path: &Bound<'_, PyAny>,
        wholetext: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        r = set_options_skip_none(
            r,
            &[
                ("wholetext", wholetext),
                ("lineSep", lineSep),
                ("pathGlobFilter", pathGlobFilter),
                ("recursiveFileLookup", recursiveFileLookup),
            ],
        )?;
        // Like PySpark, the options and the format stay on the reader.
        self.inner = r.clone().format("text");
        let path = format_path(path)?;
        Ok(PyDataFrame::new(r.text(&path)))
    }
}

/// Python wrapper for Trigger.
#[pyclass(
    name = "Trigger",
    module = "pyspark.sql.streaming.readwriter",
    from_py_object
)]
#[derive(Clone)]
pub struct PyTrigger {
    inner: Trigger,
}

impl PyTrigger {
    pub fn new(trigger: Trigger) -> Self {
        PyTrigger { inner: trigger }
    }

    pub fn get(&self) -> Trigger {
        self.inner.clone()
    }
}

#[pymethods]
impl PyTrigger {
    #[staticmethod]
    fn processingTime(interval: &str) -> PyTrigger {
        PyTrigger::new(Trigger::ProcessingTime(interval.to_string()))
    }

    #[staticmethod]
    fn once() -> PyTrigger {
        PyTrigger::new(Trigger::Once)
    }

    #[staticmethod]
    fn availableNow() -> PyTrigger {
        PyTrigger::new(Trigger::AvailableNow)
    }

    #[staticmethod]
    fn continuous(interval: &str) -> PyTrigger {
        PyTrigger::new(Trigger::Continuous(interval.to_string()))
    }
}

/// Python wrapper for DataStreamWriter.
#[pyclass(
    name = "DataStreamWriter",
    module = "pyspark.sql.streaming.readwriter",
    weakref,
    dict
)]
pub struct PyDataStreamWriter {
    inner: WriterState<DataStreamWriter>,
}

impl PyDataStreamWriter {
    pub fn new(writer: DataStreamWriter) -> Self {
        PyDataStreamWriter {
            inner: WriterState::new(writer),
        }
    }

    /// Apply the one-shot writer options accepted by `start`/`toTable`
    /// (`format`, `outputMode`, `partitionBy`, `queryName`, `**options`) before
    /// terminating the stream. Mirrors the reference, where these are keyword args.
    fn apply_write_opts(
        &self,
        format: Option<&str>,
        output_mode: Option<&str>,
        partition_by: Option<&Bound<'_, PyAny>>,
        query_name: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<DataStreamWriter> {
        let mut w = self.inner.get();
        if let Some(fmt) = format {
            w = w.format(fmt);
        }
        if let Some(m) = output_mode {
            w = w.output_mode(m);
        }
        if let Some(pb) = partition_by {
            let cols = extract_partition_cols(pb)?;
            let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
            w = w.partition_by(refs);
        }
        if let Some(qn) = query_name {
            w = w.query_name(&query_name_of(qn)?);
        }
        if options.is_some() {
            let opts = crate::options::kwargs_skip_none(options)?;
            w = w.options(opts);
        }
        // PySpark's start/toTable set these on the writer itself before starting.
        self.inner.set(w.clone());
        Ok(w)
    }
}

#[pymethods]
impl PyDataStreamWriter {
    fn outputMode<'py>(slf: &Bound<'py, Self>, outputMode: &str) -> PyResult<Bound<'py, Self>> {
        slf.borrow().inner.update(|w| w.output_mode(outputMode));
        Ok(slf.clone())
    }

    fn format<'py>(slf: &Bound<'py, Self>, source: &str) -> PyResult<Bound<'py, Self>> {
        slf.borrow().inner.update(|w| w.format(source));
        Ok(slf.clone())
    }

    fn option<'py>(
        slf: &Bound<'py, Self>,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        // None -> option left unset; bools -> "true"/"false" (reference `to_str`).
        match crate::coerce_option_value(value)? {
            Some(v) => {
                slf.borrow().inner.update(|w| w.option(key, &v));
                Ok(slf.clone())
            }
            None => Ok(slf.clone()),
        }
    }

    // Mirrors reference `DataStreamWriter.options(**options)`: keyword args; None values
    // skipped, booleans lowercased.
    #[pyo3(signature = (**options))]
    fn options<'py>(
        slf: &Bound<'py, Self>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        let opts = crate::options::kwargs_skip_none(options)?;
        slf.borrow().inner.update(|w| w.options(opts));
        Ok(slf.clone())
    }

    /// Partition by the column names, or a list or tuple of them alone, as PySpark's.
    #[pyo3(signature = (*cols))]
    fn partitionBy<'py>(
        slf: &Bound<'py, Self>,
        cols: &Bound<'_, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        let cols = column_names(cols)?;
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        slf.borrow().inner.update(|w| w.partition_by(col_refs));
        Ok(slf.clone())
    }

    /// Cluster by the column names, or a list or tuple of them alone, as PySpark's.
    #[pyo3(signature = (*cols))]
    fn clusterBy<'py>(
        slf: &Bound<'py, Self>,
        cols: &Bound<'_, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        let cols = column_names(cols)?;
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        slf.borrow().inner.update(|w| w.cluster_by(col_refs));
        Ok(slf.clone())
    }

    /// Name the query, a non-empty str, else `VALUE_NOT_NON_EMPTY_STR`, as PySpark's.
    fn queryName<'py>(
        slf: &Bound<'py, Self>,
        queryName: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let name = query_name_of(queryName)?;
        slf.borrow().inner.update(|w| w.query_name(&name));
        Ok(slf.clone())
    }

    /// `DataStreamWriter.trigger(...)`, as PySpark's: exactly one of `processingTime`, `once`,
    /// `continuous`, `availableNow` and `realTime`, else `ONLY_ALLOW_SINGLE_TRIGGER`; an
    /// interval or duration is a non-empty string, which is stripped, else
    /// `VALUE_NOT_NON_EMPTY_STR`, and `once` and `availableNow` are True, else
    /// `VALUE_NOT_ALLOWED`.
    #[pyo3(signature = (*, processingTime=None, once=None, continuous=None, availableNow=None, realTime=None))]
    fn trigger<'py>(
        slf: &Bound<'py, Self>,
        processingTime: Option<Bound<'py, PyAny>>,
        once: Option<Bound<'py, PyAny>>,
        continuous: Option<Bound<'py, PyAny>>,
        availableNow: Option<Bound<'py, PyAny>>,
        realTime: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        use crate::column::raise_pyspark;
        let py = slf.py();
        let given = [
            &processingTime,
            &once,
            &continuous,
            &availableNow,
            &realTime,
        ]
        .iter()
        .filter(|value| value.is_some())
        .count();
        if given != 1 {
            return Err(raise_pyspark(
                py,
                "PySparkValueError",
                "ONLY_ALLOW_SINGLE_TRIGGER",
                &[],
            ));
        }
        let interval = |value: &Bound<'py, PyAny>, name: &str| -> PyResult<String> {
            let text = match value.cast::<pyo3::types::PyString>() {
                Ok(text) => text.to_cow()?.trim().to_string(),
                Err(_) => String::new(),
            };
            if text.is_empty() {
                return Err(raise_pyspark(
                    py,
                    "PySparkValueError",
                    "VALUE_NOT_NON_EMPTY_STR",
                    &[("arg_name", name), ("arg_value", &value.str()?.to_string())],
                ));
            }
            Ok(text)
        };
        // As PySpark's `once is not True`.
        let true_flag = |value: &Bound<'py, PyAny>, name: &str| -> PyResult<()> {
            if value.is(pyo3::types::PyBool::new(py, true)) {
                Ok(())
            } else {
                Err(raise_pyspark(
                    py,
                    "PySparkValueError",
                    "VALUE_NOT_ALLOWED",
                    &[("arg_name", name), ("allowed_values", "[True]")],
                ))
            }
        };
        let trigger = if let Some(value) = &processingTime {
            Trigger::ProcessingTime(interval(value, "processingTime")?)
        } else if let Some(value) = &once {
            true_flag(value, "once")?;
            Trigger::Once
        } else if let Some(value) = &continuous {
            Trigger::Continuous(interval(value, "continuous")?)
        } else if let Some(value) = &realTime {
            Trigger::RealTime(interval(value, "realTime")?)
        } else {
            if let Some(value) = &availableNow {
                true_flag(value, "availableNow")?;
            }
            Trigger::AvailableNow
        };
        slf.borrow().inner.update(|w| w.trigger(trigger));
        Ok(slf.clone())
    }

    #[pyo3(signature = (path=None, format=None, outputMode=None, partitionBy=None, queryName=None, **options))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn start(
        &self,
        py: Python<'_>,
        path: Option<&str>,
        format: Option<&str>,
        outputMode: Option<&str>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        queryName: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyStreamingQuery> {
        let writer = self.apply_write_opts(format, outputMode, partitionBy, queryName, options)?;
        // Memory/console/foreach sinks take no path; the core treats "" as unset.
        let query = interruptible(py, &writer.spark_session(), || {
            writer
                .start_with_event_handler(path.unwrap_or(""), post_started_event)
                .map_err(Box::new)
        })?;
        Ok(PyStreamingQuery::new(query))
    }

    #[pyo3(signature = (tableName, format=None, outputMode=None, partitionBy=None, queryName=None, **options))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn toTable(
        &self,
        py: Python<'_>,
        tableName: &str,
        format: Option<&str>,
        outputMode: Option<&str>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        queryName: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyStreamingQuery> {
        let writer = self.apply_write_opts(format, outputMode, partitionBy, queryName, options)?;
        let query = interruptible(py, &writer.spark_session(), || {
            writer
                .to_table_with_event_handler(tableName, post_started_event)
                .map_err(Box::new)
        })?;
        Ok(PyStreamingQuery::new(query))
    }

    /// `DataStreamWriter.foreachBatch(func)`: cloudpickle the `(batch_df, batch_id)`
    /// function (with PySpark's `CloudPickleSerializer`) and attach it as the
    /// foreach-batch sink. Pickling happens here so the Python skin can re-export this
    /// class directly rather than subclassing a (non-subclassable) PyO3 type.
    fn foreachBatch<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let command = streaming_function_command(py, func, "foreachBatch")?;
        let payload = PythonUDFPayload::new(
            spark_connect::types::DataType::Struct { fields: vec![] },
            0, // eval type is unused for the streaming foreach sinks
            command,
            crate::dataframe::py_version(py),
        );
        slf.borrow().inner.update(|w| w.foreach_batch(payload));
        Ok(slf.clone())
    }

    /// `DataStreamWriter.foreach(f)`, as PySpark's: the function of the partition and the
    /// rows of each partition of `_construct_foreach_function(f)`, which raises
    /// `ATTRIBUTE_NOT_CALLABLE` for an object without a callable `process`, in the command
    /// `(func, None, serializer, serializer)` of `AutoBatchedSerializer(CPickleSerializer())`,
    /// which the worker of the server deserializes with its own `pyspark.serializers`. Built
    /// here to avoid a Python subclass of the PyO3 class.
    fn foreach<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'_>,
        f: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let func = py
            .import("pyspark.sql.streaming.readwriter")?
            .getattr("_construct_foreach_function")?
            .call1((f,))?;
        let serializers = py.import("pyspark.serializers")?;
        let cpickle = serializers.getattr("CPickleSerializer")?.call0()?;
        let serializer = serializers
            .getattr("AutoBatchedSerializer")?
            .call1((cpickle,))?;
        // (func, return_type=None, input_serializer, output_serializer) — the shape the
        // worker's foreach runner expects; the same serializer instance is used twice.
        let command_tuple = pyo3::types::PyTuple::new(
            py,
            [
                func,
                py.None().into_bound(py),
                serializer.clone(),
                serializer,
            ],
        )?;
        let command = streaming_function_command(py, command_tuple.as_any(), "foreach")?;
        let payload = PythonUDFPayload::new(
            spark_connect::types::DataType::Struct { fields: vec![] },
            0,
            command,
            crate::dataframe::py_version(py),
        );
        slf.borrow().inner.update(|w| w.foreach(payload));
        Ok(slf.clone())
    }
}

/// The progress of a query from its JSON, as PySpark's
/// `StreamingQueryProgress.fromJson(json.loads(progress))`.
fn progress_from_json<'py>(py: Python<'py>, json: &str) -> PyResult<Bound<'py, PyAny>> {
    let loaded = py.import("json")?.call_method1("loads", (json,))?;
    py.import("pyspark.sql.streaming.listener")?
        .getattr("StreamingQueryProgress")?
        .call_method1("fromJson", (loaded,))
}

/// The timeout of `awaitTermination` and `awaitAnyTermination`, in seconds, as PySpark's take
/// it: an int or a float, which must be positive, else `VALUE_NOT_POSITIVE` of the name of its
/// type.
fn positive_timeout(timeout: Option<&Bound<'_, PyAny>>) -> PyResult<Option<f64>> {
    let Some(timeout) = timeout else {
        return Ok(None);
    };
    let number = timeout.is_instance_of::<pyo3::types::PyInt>()
        || timeout.is_instance_of::<pyo3::types::PyFloat>();
    match number.then(|| timeout.extract::<f64>()).transpose()? {
        Some(seconds) if seconds > 0.0 => Ok(Some(seconds)),
        _ => {
            let type_name = timeout.get_type().name()?.to_string();
            Err(crate::column::raise_pyspark(
                timeout.py(),
                "PySparkValueError",
                "VALUE_NOT_POSITIVE",
                &[("arg_name", "timeout"), ("arg_value", &type_name)],
            ))
        }
    }
}

/// A bool argument that PySpark sets on a proto bool field, which takes a bool or an int, and
/// raises the `TypeError` of protobuf for another value.
pub(crate) struct ProtoBool(pub(crate) bool);

impl<'a, 'py> FromPyObject<'a, 'py> for ProtoBool {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<pyo3::types::PyInt>() {
            return Ok(ProtoBool(obj.is_truthy()?));
        }
        let shown = obj.repr()?;
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Cannot set spark.connect.StreamingQueryCommand.ExplainCommand.extended to {shown}: \
             {shown} has type {}, but expected one of: (<class 'bool'>, <class 'int'>)",
            obj.get_type().repr()?
        )))
    }
}

/// The command of the function of `foreach` or `foreachBatch`, as PySpark's: of
/// `CloudPickleSerializer().dumps`, which raises a `pickle.PicklingError` for an object that
/// does not pickle, as `STREAMING_CONNECT_SERIALIZATION_ERROR` of `name`.
fn streaming_function_command(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Vec<u8>> {
    let serializer = py
        .import("pyspark.serializers")?
        .getattr("CloudPickleSerializer")?
        .call0()?;
    match serializer.call_method1("dumps", (function,)) {
        Ok(command) => command.extract(),
        Err(error) if error.matches(py, py.import("pickle")?.getattr("PicklingError")?)? => {
            let raised = crate::column::raise_pyspark(
                py,
                "PySparkPicklingError",
                "STREAMING_CONNECT_SERIALIZATION_ERROR",
                &[("name", name)],
            );
            raised.set_cause(py, Some(error));
            Err(raised)
        }
        Err(error) => Err(error),
    }
}

/// Python wrapper for StreamingQuery.
#[pyclass(
    name = "StreamingQuery",
    module = "pyspark.sql.streaming.query",
    weakref,
    dict
)]
pub struct PyStreamingQuery {
    inner: StreamingQuery,
}

impl PyStreamingQuery {
    pub fn new(query: StreamingQuery) -> Self {
        PyStreamingQuery { inner: query }
    }
}

#[pymethods]
impl PyStreamingQuery {
    #[getter]
    fn id(&self) -> String {
        self.inner.id().to_string()
    }

    #[getter]
    fn runId(&self) -> String {
        self.inner.run_id().to_string()
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name().map(|s| s.to_string())
    }

    #[getter]
    fn isActive(&self, py: Python<'_>) -> PyResult<bool> {
        crate::dataframe::detached(py, || self.inner.is_active().map_err(Box::new))
    }

    /// The status of the query, as PySpark's: a dict of its `message`, `isDataAvailable` and
    /// `isTriggerActive`.
    #[getter]
    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let status = interruptible(py, &self.inner.spark_session(), || {
            self.inner.status().map_err(Box::new)
        })?;
        let dict = PyDict::new(py);
        dict.set_item("message", status.status_message)?;
        dict.set_item("isDataAvailable", status.is_data_available)?;
        dict.set_item("isTriggerActive", status.is_trigger_active)?;
        Ok(dict)
    }

    fn stop(&self, py: Python<'_>) -> PyResult<()> {
        interruptible(py, &self.inner.spark_session(), || {
            self.inner.stop().map_err(Box::new)
        })
    }

    #[pyo3(signature = (timeout=None))]
    fn awaitTermination(
        &self,
        py: Python<'_>,
        timeout: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<bool>> {
        let timeout = positive_timeout(timeout)?;
        interruptible(py, &self.inner.spark_session(), || {
            self.inner.await_termination(timeout).map_err(Box::new)
        })
    }

    /// The last progress of the query, as PySpark's `StreamingQueryProgress`, or None.
    #[getter]
    fn lastProgress<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let progress = interruptible(py, &self.inner.spark_session(), || {
            self.inner.last_progress().map_err(Box::new)
        })?;
        match progress {
            Some(json) => progress_from_json(py, &json).map(Some),
            None => Ok(None),
        }
    }

    /// The recent progress of the query, each as PySpark's `StreamingQueryProgress`.
    #[getter]
    fn recentProgress<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        interruptible(py, &self.inner.spark_session(), || {
            self.inner.recent_progress().map_err(Box::new)
        })?
        .iter()
        .map(|json| progress_from_json(py, json))
        .collect()
    }

    fn processAllAvailable(&self, py: Python<'_>) -> PyResult<()> {
        interruptible(py, &self.inner.spark_session(), || {
            self.inner.process_all_available().map_err(Box::new)
        })
    }

    /// Print the plan of the query with Python's `print`, as PySpark's
    /// `StreamingQuery.explain(extended=False)`.
    #[pyo3(signature = (extended=ProtoBool(false)))]
    fn explain(&self, py: Python<'_>, extended: ProtoBool) -> PyResult<()> {
        let plan = interruptible(py, &self.inner.spark_session(), || {
            self.inner.explain(extended.0).map_err(Box::new)
        })?;
        py.import("builtins")?.call_method1("print", (plan,))?;
        Ok(())
    }

    /// The exception of a failed query, as PySpark's: the `StreamingQueryException` of
    /// `pyspark.errors.exceptions.connect`, of the message of the server without the type of
    /// its Java exception, and with its JVM stack trace, and of the error class as its reason.
    fn exception(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let exception = interruptible(py, &self.inner.spark_session(), || {
            self.inner.exception().map_err(Box::new)
        })?;
        let Some(exception) = exception else {
            return Ok(None);
        };
        // The message is the `toString` of the Java StreamingQueryException: drop its type.
        let mut message = match exception.message.split_once(": ") {
            Some((_, message)) => message.to_string(),
            None => exception.message,
        };
        if let Some(stack_trace) = exception.stack_trace {
            message.push_str(&format!("\n\nJVM stacktrace:\n{stack_trace}"));
        }
        let class = py
            .import("pyspark.errors.exceptions.connect")?
            .getattr("StreamingQueryException")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("reason", exception.error_class)?;
        Ok(Some(class.call((message,), Some(&kwargs))?.unbind()))
    }
}

/// Python wrapper for StreamingQueryManager, of the listener bus of its session.
#[pyclass(
    name = "StreamingQueryManager",
    module = "pyspark.sql.streaming.query",
    weakref,
    dict
)]
pub struct PyStreamingQueryManager {
    inner: StreamingQueryManager,
    /// The session, which the listeners get as their `spark`.
    session: spark_connect::session::SparkSession,
}

impl PyStreamingQueryManager {
    pub fn new(
        manager: StreamingQueryManager,
        session: spark_connect::session::SparkSession,
    ) -> Self {
        PyStreamingQueryManager {
            inner: manager,
            session,
        }
    }
}

/// Warn of `message`, as PySpark Connect's listener bus warns of a failure to add or remove a
/// listener, rather than raising.
fn warn(py: Python<'_>, message: String) -> PyResult<()> {
    py.import("warnings")?.call_method1("warn", (message,))?;
    Ok(())
}

/// `StreamingQueryManager._sqlb`, as PySpark Connect's `StreamingQueryListenerBus`, of its
/// `_listener_bus`: the client-side listeners of the session, in the order that they were added,
/// which Apache Spark's tests read to remove them.
#[pyclass(
    name = "StreamingQueryListenerBus",
    module = "pyspark.sql.streaming.query",
    frozen
)]
pub struct PyStreamingQueryListenerBus {
    manager: StreamingQueryManager,
}

#[pymethods]
impl PyStreamingQueryListenerBus {
    #[getter]
    fn _listener_bus(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.manager
            .listeners()
            .iter()
            .filter_map(|listener| {
                let adapter = listener.as_any()?.downcast_ref::<PyListenerAdapter>()?;
                Some(adapter.listener.clone_ref(py))
            })
            .collect()
    }
}

#[pymethods]
impl PyStreamingQueryManager {
    #[getter]
    fn active(&self, py: Python<'_>) -> PyResult<Vec<PyStreamingQuery>> {
        let queries = interruptible(py, &self.inner.spark_session(), || {
            self.inner.active().map_err(Box::new)
        })?;
        Ok(queries.into_iter().map(PyStreamingQuery::new).collect())
    }

    fn get(&self, py: Python<'_>, id: &str) -> PyResult<Option<PyStreamingQuery>> {
        let query = interruptible(py, &self.inner.spark_session(), || {
            self.inner.get(id).map_err(Box::new)
        })?;
        Ok(query.map(PyStreamingQuery::new))
    }

    #[pyo3(signature = (timeout=None))]
    fn awaitAnyTermination(
        &self,
        py: Python<'_>,
        timeout: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<bool>> {
        let timeout = positive_timeout(timeout)?;
        interruptible(py, &self.inner.spark_session(), || {
            self.inner.await_any_termination(timeout).map_err(Box::new)
        })
    }

    fn resetTerminated(&self, py: Python<'_>) -> PyResult<()> {
        interruptible(py, &self.inner.spark_session(), || {
            self.inner.reset_terminated().map_err(Box::new)
        })
    }

    /// Register a client-side listener object (with onQueryStarted/Progress/Idle/
    /// Terminated callbacks). Mirrors `StreamingQueryManager.addListener`: the listener gets the
    /// session as its `spark`, and the native Rust bus of the session dispatches its events to
    /// it. The id of each registration is appended to the listener's `_rust_listener_ids`, as a
    /// listener added twice is in the bus twice. Should the server fail to add its listener, this
    /// warns, as PySpark Connect's does.
    /// A terminal event-stream end or error clears the registrations; add listeners again
    /// to open a new subscription.
    fn addListener(&self, py: Python<'_>, listener: Py<PyAny>) -> PyResult<()> {
        let session = crate::session::session_object(py, &self.session)?;
        listener
            .bind(py)
            .call_method1("_set_spark_session", (session,))?;
        let adapter = Arc::new(PyListenerAdapter {
            listener: listener.clone_ref(py),
        });
        match interruptible_outcome(py, &self.inner.spark_session(), || {
            self.inner.add_listener(adapter).map_err(Box::new)
        })? {
            Ok(id) => {
                let bound = listener.bind(py);
                match bound.getattr("_rust_listener_ids") {
                    Ok(ids) => ids.call_method1("append", (id,)).map(|_| ()),
                    Err(_) => bound.setattr("_rust_listener_ids", vec![id]),
                }
            }
            Err(error) => warn(
                py,
                format!(
                    "Failed to add the listener because of exception: {}\n\
                     The listener is not added, please add it again.",
                    spark_error_to_py_exception(*error).value(py)
                ),
            ),
        }
    }

    /// Remove a previously-added client-side listener object. Mirrors
    /// `StreamingQueryManager.removeListener`: one registration of the listener is removed from
    /// the bus of the session, the first one that the bus has (as the bus removes the first
    /// occurrence), and the others, which are of other buses, stay. Should the server fail to
    /// remove its listener, this warns, and the listener stays, as PySpark Connect's does.
    /// A callback coalesced with another caller's in-progress last-listener removal or close
    /// keeps its ids so a failed operation can be retried. Only a completed removal consumes
    /// one matching id from this session's bus.
    fn removeListener(&self, py: Python<'_>, listener: Py<PyAny>) -> PyResult<()> {
        let bound = listener.bind(py);
        let Ok(ids) = bound.getattr("_rust_listener_ids") else {
            return Ok(());
        };
        let registered: Vec<String> = ids.extract()?;
        for id in registered {
            match interruptible_outcome(py, &self.inner.spark_session(), || {
                self.inner.remove_listener_found(&id).map_err(Box::new)
            })? {
                Ok(false) => continue,
                Ok(true) => {
                    ids.call_method1("remove", (&id,))?;
                    if ids.len()? == 0 {
                        let _ = bound.delattr("_rust_listener_ids");
                    }
                    return Ok(());
                }
                Err(error) => {
                    return warn(
                        py,
                        format!(
                            "Failed to remove the listener because of exception: {}\n\
                             The listener is not removed, please remove it again.",
                            spark_error_to_py_exception(*error).value(py)
                        ),
                    );
                }
            }
        }
        Ok(())
    }

    /// The listener bus of the session, as PySpark Connect's `_sqlb`.
    #[getter]
    fn _sqlb(&self) -> PyStreamingQueryListenerBus {
        PyStreamingQueryListenerBus {
            manager: self.session.streams(),
        }
    }

    /// Remove all client-side listeners and stop the dispatch thread. Mirrors
    /// `StreamingQueryManager.close`. A server removal failure warns and keeps the
    /// listeners registered, as PySpark Connect's does, so close can be retried.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        if let Err(error) = interruptible_outcome(py, &self.inner.spark_session(), || {
            self.inner.close().map_err(Box::new)
        })? {
            return warn(
                py,
                format!(
                    "Failed to remove the listener because of exception: {}\n\
                     The listener is not removed, please remove it again.",
                    spark_error_to_py_exception(*error).value(py)
                ),
            );
        }
        Ok(())
    }

    fn streamListenerEvents(&self, py: Python<'_>) -> PyResult<PyListenerEventStream> {
        let stream = interruptible(py, &self.inner.spark_session(), || {
            self.inner.listener_event_stream().map_err(Box::new)
        })?;
        Ok(PyListenerEventStream::new(stream))
    }
}

/// Python wrapper for ListenerEventStream.
/// Implements `__iter__` and `__next__` to yield (event_type: i32, event_json: String) tuples.
#[pyclass(name = "ListenerEventStream", module = "pyspark.sql.streaming.query")]
pub struct PyListenerEventStream {
    inner: Option<ListenerEventStream>,
}

impl PyListenerEventStream {
    pub fn new(stream: ListenerEventStream) -> Self {
        PyListenerEventStream {
            inner: Some(stream),
        }
    }
}

#[pymethods]
impl PyListenerEventStream {
    fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<(i32, String)>> {
        let Some(stream) = self.inner.as_mut() else {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "ListenerEventStream already consumed",
            ));
        };
        // The listener bus blocks in `next()` waiting for the server's next event
        // (arbitrarily far apart on a live query), so release the GIL across it —
        // otherwise the daemon thread would freeze the whole interpreter between
        // events. Mirrors `PyLocalRowIterator::__next__`.
        match py.detach(|| stream.next()) {
            Some(Ok((event_type, event_json))) => Ok(Some((event_type, event_json))),
            Some(Err(e)) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error reading listener events: {}",
                e
            ))),
            None => Ok(None),
        }
    }
}
