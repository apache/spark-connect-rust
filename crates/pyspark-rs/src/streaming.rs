//! PyO3 wrappers for Structured Streaming.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use spark_connect::streaming::{
    DataStreamReader, DataStreamWriter, ListenerEventStream, StreamingQuery,
    StreamingQueryListener, StreamingQueryListenerEvent, StreamingQueryManager, Trigger,
};
use spark_connect::udf::PythonUDFPayload;
use spark_connect_core::error::{Result as SparkResult, SparkError};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::dataframe::{PyDataFrame, WriterState};
use crate::errors::spark_error_to_py_exception;
use crate::interrupt::interruptible;
use crate::options::{
    option_key, put_option_skip_none, put_shared_option, reader_option_text, skip_none_shared,
    stream_option_text, utf32le_of, DefaultedOption, OptionKey, OptionText, UnencodableOptions,
};
use crate::readwriter::{
    reader_plan_error, reader_schema, table_name_of, with_plan_error, UnsentFormat,
};

/// One entry of the `_listener_bus` of a `StreamingQueryListenerBus`, registered in the native bus
/// of its session, which keeps the subscription of the session to the events of the server while it
/// has registrations. An event is posted once to the `post_to_all` of each bus of the
/// registrations, which calls the listeners of its `_listener_bus` as they are then, as the handler
/// thread of PySpark's bus posts it: so each entry of the list gets it, one that the bus did not
/// append too, and a listener once for each of its entries. Decode errors return to the native bus,
/// and terminal failures produce the reference handler warning.
struct PyListenerAdapter {
    /// The `StreamingQueryListenerBus` of the entry.
    bus: Py<PyAny>,
    /// The `_listener_bus` of the bus and the listener of the entry, as `append` appended it.
    entry: Option<ListenerEntry>,
    /// The id of this registration in the bus of its session, once the bus has added it. It is
    /// kept here, not on the listener: the listener is the user's object, which may have no
    /// attributes to set (`__slots__`), and may be added by several threads and to several
    /// sessions at once.
    id: std::sync::OnceLock<String>,
    /// Set while the `StreamingQueryListenerBus` itself ends this registration, of a removal, a
    /// close or an append that failed, which changes its list as PySpark's does.
    released: AtomicBool,
}

/// The list of a `StreamingQueryListenerBus`, and the listener of one of its entries.
struct ListenerEntry {
    entries: Py<PyAny>,
    listener: Py<PyAny>,
}

impl PyListenerAdapter {
    /// The adapter of `listener`, a registration of the native bus, if it is one.
    fn of(listener: &dyn StreamingQueryListener) -> Option<&PyListenerAdapter> {
        listener.as_any()?.downcast_ref::<PyListenerAdapter>()
    }

    /// Whether this is the registration of the entry `listener` of the list `entries`, by their
    /// identities.
    fn is_entry(&self, entries: &Bound<'_, PyAny>, listener: &Bound<'_, PyAny>) -> bool {
        self.entry.as_ref().is_some_and(|entry| {
            entry.entries.bind(entries.py()).is(entries)
                && entry.listener.bind(entries.py()).is(listener)
        })
    }
}

impl Drop for PyListenerAdapter {
    /// A registration that the native bus ended, at the end of its subscription or of its session,
    /// takes its entry off the list of its bus, as the handler thread of PySpark's bus clears the
    /// list when its events fail: at once on a thread of Python, else on the delivery thread (see
    /// `deliver::run`), with the exception that is set on the thread kept.
    fn drop(&mut self) {
        if self.released.load(Ordering::Acquire) {
            return;
        }
        if let Some(entry) = self.entry.take() {
            crate::deliver::run(move |py| {
                let _ = remove_identical(entry.entries.bind(py), entry.listener.bind(py));
            });
        }
    }
}

/// Remove the first item of the list `entries` that is `listener`, by its identity.
fn remove_identical(entries: &Bound<'_, PyAny>, listener: &Bound<'_, PyAny>) -> PyResult<()> {
    let entries = entries.cast::<PyList>()?;
    if let Some(index) = entries.iter().position(|item| item.is(listener)) {
        entries.del_item(index)?;
    }
    Ok(())
}

/// Remove the first item of `entries` that is `listener` or equal to it, as PySpark's bus removes a
/// listener of its list: of none, nothing, as another removal took it.
fn remove_equal(entries: &Bound<'_, PyAny>, listener: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = entries.py();
    match entries.call_method1(pyo3::intern!(py, "remove"), (listener,)) {
        Ok(_) => Ok(()),
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyValueError>(py) => Ok(()),
        Err(error) => Err(error),
    }
}

/// Run `f` holding `lock`, of its `__enter__` and `__exit__`, as PySpark's bus runs the block of
/// its `with self._lock:`.
fn with_lock<T>(lock: &Bound<'_, PyAny>, f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    let py = lock.py();
    lock.call_method0(pyo3::intern!(py, "__enter__"))?;
    let result = f();
    let exited = lock.call_method1(
        pyo3::intern!(py, "__exit__"),
        (py.None(), py.None(), py.None()),
    );
    let value = result?;
    exited?;
    Ok(value)
}

/// Clear the bus `bus` as PySpark's handler thread does once its events failed:
/// `with self._lock: self._execution_thread = None; self._listener_bus.clear()`.
fn clear_bus(bus: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = bus.py();
    let lock = crate::state::attr(bus, pyo3::intern!(py, "_lock"))?;
    with_lock(&lock, || {
        bus.setattr(pyo3::intern!(py, "_execution_thread"), py.None())?;
        crate::state::attr(bus, pyo3::intern!(py, "_listener_bus"))?
            .call_method0(pyo3::intern!(py, "clear"))?;
        Ok(())
    })
}

/// Clear the listener bus of the manager `manager`, its `_sqlb`, as [`clear_bus`] does.
pub(crate) fn clear_listeners_of(manager: &Bound<'_, PyAny>) -> PyResult<()> {
    clear_bus(&crate::state::attr(
        manager,
        pyo3::intern!(manager.py(), "_sqlb"),
    )?)
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

/// Decode before any callback, sharing one Python object across every registration, and post it
/// once to each bus of the registrations, of its `post_to_all`, as PySpark's handler thread posts
/// the events of its bus. The same path handles synchronous starts and asynchronous protocol
/// events, including mixed native/Python snapshots. Keep native errors typed until the reporting
/// boundary.
fn post_listener_event<'a>(
    py: Python<'_>,
    listeners: impl IntoIterator<Item = &'a dyn StreamingQueryListener>,
    event: &StreamingQueryListenerEvent,
    started: bool,
) -> std::result::Result<(), ListenerDispatchError> {
    let module = py.import("pyspark.sql.connect.streaming.query")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("started", started)?;
    let decoded = module.call_method(
        "_deserialize_listener_event",
        (event.event_type, event.event_json.as_str()),
        Some(&kwargs),
    )?;
    // The buses that the event was posted to, each once.
    let mut posted: Vec<Bound<'_, PyAny>> = Vec::new();
    for listener in listeners {
        match PyListenerAdapter::of(listener) {
            Some(adapter) => {
                let bus = adapter.bus.bind(py);
                if posted.iter().any(|seen| seen.is(bus)) {
                    continue;
                }
                bus.call_method1(pyo3::intern!(py, "post_to_all"), (&decoded,))?;
                posted.push(bus.clone());
            }
            // Native callbacks keep their usual ability to wait for Python work.
            None if started => crate::gc::detach(py, || listener.on_event(event)),
            None => crate::gc::detach(py, || listener.try_on_event(event).map_err(Box::new))
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
            // Of PySpark's `_query_event_handler`, which then clears its bus.
            let _ = warn(
                py,
                380,
                format!(
                    "StreamingQueryListenerBus Handler thread received exception, all client side \
                     listeners are removed and handler thread is terminated. The error is: {}",
                    message
                ),
            );
            // The registrations of the bus end with the failed subscription: of the list that this
            // clears, so that their ends do not take the listeners that are appended since.
            let bus = self.bus.bind(py);
            if let Ok(manager) = bus
                .cast::<PyStreamingQueryListenerBus>()
                .map_err(PyErr::from)
                .and_then(PyStreamingQueryListenerBus::manager_of)
            {
                for registration in manager.listeners() {
                    if let Some(adapter) = PyListenerAdapter::of(registration.as_ref()) {
                        if adapter.bus.bind(py).is(bus) {
                            adapter.released.store(true, Ordering::Release);
                        }
                    }
                }
            }
            let _ = clear_bus(bus);
        });
        true
    }

    /// The adapter, of which `StreamingQueryListenerBus` finds the registrations of its entries.
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

/// The path of `DataStreamReader.load`, as PySpark checks it: a str that is not empty once
/// `strip()` takes off Python's whitespace, else `VALUE_NOT_NON_EMPTY_STR` of its `str()`. A str
/// that UTF-8 cannot encode is not empty, and is kept with the error of the plan that PySpark
/// builds from it, which is returned with the text of the str where its lone surrogates are
/// replaced: the plan fails with the error, so that text is never sent, and it is not empty, as
/// the core reader requires.
fn load_path(path: &Bound<'_, PyAny>) -> PyResult<(String, Option<SparkError>)> {
    if let Ok(text) = path.cast::<pyo3::types::PyString>() {
        match text.extract::<String>() {
            Ok(text) if !spark_connect::streaming::is_blank_path(&text) => return Ok((text, None)),
            Ok(_) => {}
            Err(_) => {
                let error = SparkError::unencodable_str(&utf32le_of(path)?);
                return Ok((text.to_string_lossy().into_owned(), Some(error)));
            }
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

/// The column names of a stream writer's `partitionBy` or `clusterBy`, which PySpark clears and then
/// extends into its protobuf string field at the call, each one converted as [`crate::str_argument`]
/// does. A list or tuple alone stands for its items. The extension is atomic, so a column that does
/// not convert leaves none, and its error is raised.
fn stream_column_names(cols: &[Bound<'_, PyAny>]) -> PyResult<Vec<String>> {
    crate::readwriter::writer_column_items(cols)?
        .iter()
        .map(crate::str_argument)
        .collect()
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

// Python wrapper for DataStreamReader.
//
// Like PySpark's reader, the setters change this object and return it, and the
// reader can be used again after a read: the core reader is a consuming builder, so
// the setters replace the wrapped value in place and the reads work on a clone.
// As PySpark's plain class, it constructs of a session, `DataStreamReader(spark)`, and is
// subclassed: the setters return the object, of its subclass, and its copies and pickles are
// of its class.
#[doc = connect_doc!(DataStreamReader)]
#[pyclass(
    name = "DataStreamReader",
    module = "pyspark.sql.connect.streaming.readwriter",
    subclass,
    weakref,
    dict
)]
pub struct PyDataStreamReader {
    inner: DataStreamReader,
    /// The options of the reader, which a shallow copy of it shares, as PySpark's `_options` dict
    /// is shared by the copies of a reader. The core reader holds none of them (see `state`).
    options: Arc<WriterState<HashMap<String, String>>>,
    /// The format of `format` when it is not a format of the core reader, which PySpark keeps as it
    /// is, and whose `DataSource` asserts at `load`, after the path ([`stream_reader_format`]).
    unsent_format: Option<UnsentFormat>,
    /// The options that UTF-8 cannot encode, which the copies of this reader share, as they share
    /// `options` ([`UnencodableOptions`]).
    unencodable_options: Arc<WriterState<UnencodableOptions>>,
    /// The Python object of `unencodable_options`, which the copies of this reader share too, and
    /// which reports the keys of its options that are objects to Python's garbage collector, as
    /// the `_options` dict of PySpark's copies holds them (see [`crate::options::SharedObjects`]).
    shared: Option<Py<crate::options::PySharedObjects>>,
    /// The UTF-32LE bytes of the schema, when it is a str that UTF-8 cannot encode, which the core
    /// reader leaves unset.
    unencodable_schema: Option<Vec<u8>>,
    /// Set once the value is written, as `__traverse__` reads `unsent_format`, which is not valid
    /// while the reader object is created (see [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

/// The stream reader with the format `source` of PySpark's `format`, and the unsent format that it
/// keeps: None unsets the format, a str sets it, and any other object is kept for `load`, as the
/// batch reader keeps it.
fn stream_reader_format(
    reader: DataStreamReader,
    source: Option<&Bound<'_, PyAny>>,
) -> (DataStreamReader, Option<UnsentFormat>) {
    match source {
        None => (reader.clear_format(), None),
        Some(source) => match source.extract::<String>() {
            Ok(text) => (reader.format(&text), None),
            Err(_) => (reader.clear_format(), Some(UnsentFormat::of(source))),
        },
    }
}

impl PyDataStreamReader {
    pub fn new(reader: DataStreamReader) -> Self {
        let options = reader.get_options().clone();
        let unencodable_options = Arc::new(WriterState::new(UnencodableOptions::default()));
        PyDataStreamReader {
            inner: without_options(reader, &options),
            options: Arc::new(WriterState::new(options)),
            unsent_format: None,
            shared: crate::options::PySharedObjects::of(unencodable_options.clone()),
            unencodable_options,
            unencodable_schema: None,
            gc_ready: crate::gc::GcReady::new(),
        }
    }

    /// The core reader of this reader, with its options, which the copies of this reader share.
    fn state(&self) -> DataStreamReader {
        self.inner.clone().options(self.options.get())
    }

    /// Keep `reader` as the state of this reader: its options are shared by the copies of this
    /// reader, and its format, schema and source name are this reader's own.
    fn set_state(&mut self, reader: DataStreamReader) {
        let options = reader.get_options().clone();
        self.inner = without_options(reader, &options);
        self.options.set(options);
    }

    /// The path of a format reader, such as `json`, as PySpark's takes it, and the state that
    /// it leaves on this reader, which takes `reader`, with the options of the call. A path that
    /// is not a str is `NOT_EXPECTED_TYPE`, and the format of the reader stays as it was. Else
    /// the reader takes `format`, and the path is checked as `load`, which PySpark's format
    /// readers call with their format, checks it after setting the format. The path is returned
    /// with the error of its plan, when UTF-8 cannot encode it.
    fn format_path(
        &mut self,
        reader: &DataStreamReader,
        format: &str,
        path: &Bound<'_, PyAny>,
    ) -> PyResult<(String, Option<SparkError>)> {
        self.set_state(reader.clone());
        if !path.is_instance_of::<pyo3::types::PyString>() {
            return Err(crate::stat_arguments::not_expected_type(
                path, "path", "str",
            ));
        }
        self.set_state(reader.clone().format(format));
        self.unsent_format = None;
        let checked = load_path(path)?;
        self.check_data_source()?;
        Ok(checked)
    }
}

impl PyDataStreamReader {
    /// `r` with the option `key` set to `value`, or unset when None, as this reader's `option`.
    fn put_option(
        &self,
        r: DataStreamReader,
        key: &OptionKey,
        value: Option<&OptionText>,
    ) -> DataStreamReader {
        put_shared_option(r, &self.unencodable_options, key, value)
    }

    /// `r` with the options of `options` set, **skipping** those that are None, as the format
    /// methods set their keyword arguments, with the values converted as `str()`.
    fn skip_none_options(
        &self,
        r: DataStreamReader,
        options: &[(&str, Option<&Bound<'_, PyAny>>)],
    ) -> PyResult<DataStreamReader> {
        skip_none_shared(r, &self.unencodable_options, options, stream_option_text)
    }

    /// `r` with the `**options` of a call set, **skipping** those that are None, as `options` and
    /// `load` do, with the values converted as [`reader_option_text`] converts them.
    fn skip_none_kwargs(
        &self,
        mut r: DataStreamReader,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<DataStreamReader> {
        crate::options::check_keyword_keys(options)?;
        let mut unencodable = self.unencodable_options.get();
        for (key, value) in options.into_iter().flat_map(|options| options.iter()) {
            let key = option_key(&key)?;
            r = put_option_skip_none(r, &mut unencodable, &key, Some(&value), reader_option_text)?;
        }
        self.unencodable_options.set(unencodable);
        Ok(r)
    }

    /// `r` with the schema `schema` of `Option`, if there is one, as [`Self::set_schema`] sets it.
    fn apply_optional_schema(
        &mut self,
        r: DataStreamReader,
        schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<DataStreamReader> {
        match schema {
            Some(schema) => self.set_schema(r, schema),
            None => Ok(r),
        }
    }

    /// `r` with the schema `schema`, which this reader keeps: a str that UTF-8 cannot encode is
    /// kept as its bytes, and fails the plan, as PySpark keeps it until the plan is built.
    fn set_schema(
        &mut self,
        r: DataStreamReader,
        schema: &Bound<'_, PyAny>,
    ) -> PyResult<DataStreamReader> {
        match reader_schema(schema)? {
            Ok(text) => {
                self.unencodable_schema = None;
                Ok(r.schema(text))
            }
            Err(utf32le) => {
                self.unencodable_schema = Some(utf32le);
                Ok(r.schema(String::new()))
            }
        }
    }

    /// Fail a `DataSource` of this reader, which PySpark builds at its call and asserts that each
    /// key of its options is a str, with the bare `AssertionError` of that assertion.
    fn check_data_source(&self) -> PyResult<()> {
        if self.unencodable_options.get().has_object_key() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        Ok(())
    }

    /// The reader `r` with the options of the bytes keys that are UTF-8, which PySpark's `Read`
    /// plan sends as their text.
    fn with_decodable_keys(&self, r: DataStreamReader) -> DataStreamReader {
        self.unencodable_options
            .get()
            .decodable_pairs()
            .into_iter()
            .fold(r, |r, (key, value)| r.option(&key, &value))
    }

    /// The error of the plan of this reader, as [`reader_plan_error`] gives it.
    fn plan_error(&self) -> Option<SparkError> {
        reader_plan_error(
            self.unsent_format.as_ref(),
            self.unencodable_schema.as_deref(),
            &self.unencodable_options.get(),
        )
    }
}

#[pymethods]
impl PyDataStreamReader {
    /// `DataStreamReader(client)`, as PySpark Connect's, of the session `client`: the reader that
    /// `spark.readStream` gives. As the `object.__new__` of PySpark's plain class, it takes the
    /// arguments of the `__init__` of a subclass too, of which the session is the first that is
    /// a session, and none, as an unpickle gives none (see `session::constructor_session`);
    /// `__init__` checks them.
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(client)")]
    fn py_new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let session = crate::session::constructor_session(args, kwargs, "client")?;
        Ok(PyDataStreamReader::new(session.read_stream()))
    }

    /// As PySpark's `__init__(client)`, which makes the reader of the session anew: of no
    /// format, schema, options or source name.
    fn __init__(slf: &Bound<'_, Self>, client: PyRef<'_, crate::session::PySparkSession>) {
        let reader = PyDataStreamReader::new(client.session.read_stream());
        *slf.borrow_mut() = reader;
    }

    /// The shallow copy of a stream reader, as PySpark's `copy.copy`: the copy shares the
    /// options of this reader, as its `_options` dict is shared, so an option that either one
    /// sets or unsets is seen by both, and has its own format, which is kept as it is. It is of
    /// the class of this reader, as the copy of PySpark's of a subclass is.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let this = slf.borrow();
        let copy = crate::reduce::new_of_class(
            &slf.get_type(),
            PyDataStreamReader {
                inner: this.inner.clone(),
                options: Arc::clone(&this.options),
                unsent_format: this
                    .unsent_format
                    .as_ref()
                    .map(|format| format.clone_ref(py)),
                unencodable_options: Arc::clone(&this.unencodable_options),
                shared: this.shared.as_ref().map(|shared| shared.clone_ref(py)),
                unencodable_schema: this.unencodable_schema.clone(),
                gc_ready: crate::gc::GcReady::new(),
            },
        )?;
        crate::reduce::copy_instance_dict(slf.as_any(), copy.as_any())?;
        Ok(copy)
    }

    /// The reduce of a stream reader, as PySpark's pickle of its session and its state, which
    /// `_of_state` gives back, of the class of the reader (named in the reduce when it is a
    /// subclass), with the instance attributes of the reader.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        use crate::reduce::{optional_object, state_reduce, str_dict};

        let py = slf.py();
        let this = slf.borrow();
        let session = crate::session::session_object(py, &this.inner.spark_session())?;
        let args = PyTuple::new(
            py,
            [
                session.into_bound(py).into_any(),
                this.inner.get_format().into_pyobject(py)?.into_any(),
                this.inner.get_schema().into_pyobject(py)?.into_any(),
                str_dict(py, &this.options.get())?.into_any(),
                this.inner.get_source_name().into_pyobject(py)?.into_any(),
                optional_object(py, this.unsent_format.as_ref().map(|f| &f.object)),
                crate::readwriter::unencodable_state(
                    py,
                    this.unencodable_schema.as_deref(),
                    &this.unencodable_options.get(),
                )?,
            ],
        )?;
        let args = crate::reduce::with_subclass(slf.as_any(), py.get_type::<Self>(), args)?;
        let reduced = state_reduce(py, "stream_reader_reduce", args)?;
        crate::reduce::reduce_with_instance_dict(slf.as_any(), reduced)
    }

    /// The state of a pickle of a stream reader, which pickle sets on the reader that `__new__`
    /// made of no arguments. Of PySpark's pickle, the state is the `__dict__` of its plain object,
    /// which holds `_options`: so the reader is of the session `_client`, and of the format, the
    /// schema, the options and the source name of `_format`, `_schema`, `_options` and
    /// `_source_name`, which it takes as `format`, `schema`, `option` and `name` take them, and
    /// the other entries are its instance attributes. The state of the drop-in's pickle holds the
    /// instance attributes alone (see `__reduce__`), as a reader's `__dict__` cannot hold
    /// `_options`, a property of the reader.
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyDict>) -> PyResult<()> {
        let py = slf.py();
        let state = state.copy()?;
        if state.contains("_options")? {
            let take = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
                let value = state.get_item(key)?;
                if value.is_some() {
                    state.del_item(key)?;
                }
                Ok(value)
            };
            if let Some(client) = take("_client")? {
                let session = client.cast_into::<crate::session::PySparkSession>()?;
                let reader = PyDataStreamReader::new(session.borrow().session.read_stream());
                let replaced = std::mem::replace(&mut *slf.borrow_mut(), reader);
                drop(replaced);
            }
            if let Some(format) = take("_format")? {
                let source = (!format.is_none()).then_some(&format);
                Self::format(slf.borrow_mut(), source)?;
            }
            if let Some(schema) = take("_schema")? {
                Self::schema(slf.borrow_mut(), &schema)?;
            }
            if let Some(options) = take("_options")? {
                for (key, value) in options.cast_into::<PyDict>()?.iter() {
                    Self::option(slf.borrow_mut(), &key, &value)?;
                }
            }
            if let Some(source_name) = take("_source_name")? {
                if !source_name.is_none() {
                    Self::name(slf.borrow_mut(), &source_name)?;
                }
            }
        }
        slf.getattr(pyo3::intern!(py, "__dict__"))?
            .call_method1(pyo3::intern!(py, "update"), (state,))?;
        Ok(())
    }

    /// The source name of `name`, or None, as PySpark's `DataStreamReader._source_name`.
    #[getter]
    fn _source_name(&self) -> Option<String> {
        self.inner.get_source_name().map(str::to_string)
    }

    /// A stream reader of its state, as `__reduce__` gives it, of the class `cls`: of the session
    /// `session`, the format, the schema, the options, the source name, the format that is kept
    /// (`unsent`), and the schema and the options that UTF-8 cannot encode (`unencodable`).
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    fn _of_state<'py>(
        cls: &Bound<'py, pyo3::types::PyType>,
        session: Py<crate::session::PySparkSession>,
        format: Option<String>,
        schema: String,
        options: HashMap<String, String>,
        source_name: Option<String>,
        unsent: Option<Py<PyAny>>,
        unencodable: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyDataStreamReader>> {
        let py = cls.py();
        let mut reader = session.borrow(py).session.read_stream();
        if let Some(format) = format {
            reader = reader.format(&format);
        }
        if !schema.is_empty() {
            reader = reader.schema(schema);
        }
        reader = reader.options(options);
        if let Some(source_name) = source_name {
            reader = reader.name(&source_name);
        }
        let mut result = PyDataStreamReader {
            unsent_format: unsent.map(|object| UnsentFormat::of(object.bind(py))),
            ..PyDataStreamReader::new(reader)
        };
        let (schema, entries) = crate::readwriter::unencodable_entries(unencodable.bind(py))?;
        if let Some(schema) = schema {
            let state = result.set_schema(result.state(), &schema)?;
            result.set_state(state);
        }
        for (key, value) in entries {
            let key = option_key(&key)?;
            let value = stream_option_text(&value)?;
            let state = result.put_option(result.state(), &key, value.as_ref());
            result.set_state(state);
        }
        crate::reduce::new_of_class(cls, result)
    }

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

    // Set the source format. As PySpark's, None unsets it, so that the server reads its
    // default source.
    #[doc = connect_doc!(DataStreamReader.format)]
    fn format<'py>(
        mut slf: PyRefMut<'py, Self>,
        source: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (reader, unsent) = stream_reader_format(slf.state(), source);
        slf.set_state(reader);
        slf.unsent_format = unsent;
        Ok(slf)
    }

    // Set the schema: a StructType or a DDL string, as in pyspark.
    #[doc = connect_doc!(DataStreamReader.schema)]
    fn schema<'py>(
        mut slf: PyRefMut<'py, Self>,
        schema: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let state = slf.state();
        let state = slf.set_schema(state, schema)?;
        slf.set_state(state);
        Ok(slf)
    }

    #[doc = connect_doc!(DataStreamReader.option)]
    fn option<'py>(
        mut slf: PyRefMut<'py, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        // None -> option left unset, before the key is looked at, as PySpark's returns at once;
        // other values as `str()`, so a bool is "True" (reference `DataStreamReader.option`,
        // unlike `options` and `load`).
        if value.is_none() {
            return Ok(slf);
        }
        let key = option_key(key)?;
        let value = stream_option_text(value)?;
        let state = slf.put_option(slf.state(), &key, value.as_ref());
        slf.set_state(state);
        Ok(slf)
    }

    // Mirrors reference `DataStreamReader.options(**options)`: keyword args; None values
    // skipped, booleans lowercased.
    #[pyo3(signature = (**options))]
    #[doc = connect_doc!(DataStreamReader.options)]
    fn options<'py>(
        mut slf: PyRefMut<'py, Self>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let state = slf.skip_none_kwargs(slf.state(), options)?;
        slf.set_state(state);
        Ok(slf)
    }

    /// The format that the reader keeps, a Python object, which Python's garbage collector sees;
    /// nothing of a reader object that is being created.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        if let Some(format) = &self.unsent_format {
            visit.call(&format.object)?;
        }
        if let Some(shared) = &self.shared {
            visit.call(shared)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.unsent_format = None;
        self.shared = None;
    }

    /// The session of the reader, as PySpark Connect's `DataStreamReader._client`, which its
    /// pickle keeps.
    #[getter]
    fn _client(&self, py: Python<'_>) -> PyResult<Py<crate::session::PySparkSession>> {
        crate::session::session_object(py, &self.inner.spark_session())
    }

    /// The configured source format (PySpark Connect's `DataStreamReader._format`), which is the
    /// object itself when it is not a str that the core reader takes, as PySpark keeps it.
    #[getter]
    fn _format(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        match &self.unsent_format {
            Some(format) => Some(format.object.clone_ref(py)),
            None => self
                .inner
                .get_format()
                .map(|format| PyString::new(py, format).into_any().unbind()),
        }
    }

    /// The configured schema string, empty when unset (`DataStreamReader._schema`).
    #[getter]
    fn _schema(&self) -> String {
        self.inner.get_schema().to_string()
    }

    /// The configured read options (`DataStreamReader._options`); a copy.
    #[getter]
    fn _options(&self) -> HashMap<String, String> {
        self.options.get()
    }

    #[pyo3(signature = (path=None, format=None, schema=None, **options))]
    #[doc = connect_doc!(DataStreamReader.load)]
    fn load(
        &mut self,
        path: Option<&Bound<'_, PyAny>>,
        format: Option<&Bound<'_, PyAny>>,
        schema: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's load, which sets the format, the schema and the options on the reader in
        // turn, each one kept when a later one fails.
        crate::options::check_keyword_keys(options)?;
        let mut r = self.state();
        if let Some(fmt) = format {
            let (reader, unsent) = stream_reader_format(r, Some(fmt));
            r = reader;
            self.unsent_format = unsent;
        }
        self.set_state(r.clone());
        if let Some(v) = schema {
            r = self.set_schema(r, v)?;
        }
        self.set_state(r.clone());
        r = self.skip_none_kwargs(r, options)?;
        // Then the path is checked, and the `DataSource` of PySpark asserts the format after that.
        self.set_state(r.clone());
        let (path, path_error) = match path {
            Some(path) => {
                let (text, error) = load_path(path)?;
                (Some(text), error)
            }
            None => (None, None),
        };
        // A format that is not a str fails the assertion, and a str that UTF-8 cannot encode fails
        // the plan, which precedes the path in the plan of PySpark's `DataSource`.
        if self
            .unsent_format
            .as_ref()
            .is_some_and(|format| format.unencodable.is_none())
        {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        self.check_data_source()?;
        let error = self.plan_error().or(path_error);
        Ok(PyDataFrame::new(with_plan_error(
            r.load(path.as_deref()),
            error,
        )))
    }

    // Set the source name (for checkpoint stability). Mirrors `DataStreamReader.name`,
    // which raises `NOT_EXPECTED_TYPE` for a name that is not a str, and
    // `INVALID_STREAMING_SOURCE_NAME`, of the name as it is, unless
    // `re.match(r"^[a-zA-Z0-9_]+$", source_name)` matches it: a name of other characters than
    // ASCII letters, digits and underscores, a lone surrogate too, but one of a final newline,
    // before which `$` matches, which the server refuses when it reads the plan.
    #[doc = connect_doc!(DataStreamReader.name)]
    fn name<'py>(
        mut slf: PyRefMut<'py, Self>,
        source_name: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = source_name.py();
        if !source_name.is_instance_of::<pyo3::types::PyString>() {
            return Err(crate::stat_arguments::not_expected_type(
                source_name,
                "source_name",
                "str",
            ));
        }
        let matched = py
            .import("re")?
            .call_method1("match", (r"^[a-zA-Z0-9_]+$", source_name))?;
        if matched.is_none() {
            return Err(crate::errors::raise_pyspark_of(
                py,
                "PySparkValueError",
                "INVALID_STREAMING_SOURCE_NAME",
                &[("source_name", source_name)],
            ));
        }
        let state = slf.state().name(&source_name.extract::<String>()?);
        slf.set_state(state);
        Ok(slf)
    }

    // Read the streaming CDC changes of a named table. Mirrors `DataStreamReader.changes`,
    // which raises an `AnalysisException` of `_LEGACY_ERROR_TEMP_1189` for a reader of a
    // schema, before it sets the name in its string field, as `table` does.
    #[doc = connect_doc!(DataStreamReader.changes)]
    fn changes(&self, py: Python<'_>, tableName: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        if !self.inner.get_schema().is_empty() || self.unencodable_schema.is_some() {
            return Err(crate::readwriter::changes_of_schema_error(py));
        }
        let (name, error) = table_name_of(tableName)?;
        let error = error.or_else(|| self.unencodable_options.get().error());
        let r = self.with_decodable_keys(self.state());
        Ok(PyDataFrame::new(with_plan_error(r.changes(&name), error)))
    }

    // Load an XML streaming source. Mirrors `DataStreamReader.xml(...)`: set each named
    // option, then `format("xml").load(path)`.
    #[pyo3(signature = (path, rowTag=None, schema=None, excludeAttribute=None, attributePrefix=None, valueTag=None, ignoreSurroundingSpaces=None, rowValidationXSDPath=None, ignoreNamespace=None, wildcardColName=None, encoding=None, inferSchema=None, nullValue=None, dateFormat=None, timestampFormat=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, samplingRatio=None, locale=None, **unexpected), text_signature = "($self, path, rowTag=None, schema=None, excludeAttribute=None, attributePrefix=None, valueTag=None, ignoreSurroundingSpaces=None, rowValidationXSDPath=None, ignoreNamespace=None, wildcardColName=None, encoding=None, inferSchema=None, nullValue=None, dateFormat=None, timestampFormat=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, samplingRatio=None, locale=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(DataStreamReader.xml)]
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
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        crate::options::check_unexpected_keywords("DataStreamReader.xml", unexpected)?;
        let mut r = self.state();
        r = self.skip_none_options(r, &[("rowTag", rowTag)])?;
        r = self.apply_optional_schema(r, schema)?;
        r = self.skip_none_options(
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
        // Like PySpark, the options stay on the reader, and the format of a str path.
        let (path, path_error) = self.format_path(&r, "xml", path)?;
        let error = self.plan_error().or(path_error);
        Ok(PyDataFrame::new(with_plan_error(
            r.format("xml").load(Some(&path)),
            error,
        )))
    }

    // Read a table by name, which PySpark's `Read` plan sets in its string field as it is, as
    // [`crate::str_argument`] converts it.
    #[doc = connect_doc!(DataStreamReader.table)]
    fn table(&self, tableName: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let (name, error) = table_name_of(tableName)?;
        let error = error.or_else(|| self.unencodable_options.get().error());
        let r = self.with_decodable_keys(self.state());
        Ok(PyDataFrame::new(with_plan_error(r.table(&name), error)))
    }

    #[pyo3(signature = (path, schema=None, primitivesAsString=None, prefersDecimal=None, allowComments=None, allowUnquotedFieldNames=None, allowSingleQuotes=None, allowNumericLeadingZero=None, allowBackslashEscapingAnyCharacter=None, mode=None, columnNameOfCorruptRecord=None, dateFormat=None, timestampFormat=None, multiLine=None, allowUnquotedControlChars=None, lineSep=None, locale=None, dropFieldIfAllNull=None, encoding=None, pathGlobFilter=None, recursiveFileLookup=None, allowNonNumericNumbers=None, useUnsafeRow=None, **unexpected), text_signature = "($self, path, schema=None, primitivesAsString=None, prefersDecimal=None, allowComments=None, allowUnquotedFieldNames=None, allowSingleQuotes=None, allowNumericLeadingZero=None, allowBackslashEscapingAnyCharacter=None, mode=None, columnNameOfCorruptRecord=None, dateFormat=None, timestampFormat=None, multiLine=None, allowUnquotedControlChars=None, lineSep=None, locale=None, dropFieldIfAllNull=None, encoding=None, pathGlobFilter=None, recursiveFileLookup=None, allowNonNumericNumbers=None, useUnsafeRow=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(DataStreamReader.json)]
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
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        crate::options::check_unexpected_keywords("DataStreamReader.json", unexpected)?;
        let mut r = self.state();
        r = self.apply_optional_schema(r, schema)?;
        r = self.skip_none_options(
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
        // Like PySpark, the options stay on the reader, and the format of a str path.
        let (path, path_error) = self.format_path(&r, "json", path)?;
        let error = self.plan_error().or(path_error);
        Ok(PyDataFrame::new(with_plan_error(r.json(&path), error)))
    }

    #[pyo3(signature = (path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None, datetimeRebaseMode=None, int96RebaseMode=None, **unexpected), text_signature = "($self, path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None, datetimeRebaseMode=None, int96RebaseMode=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(DataStreamReader.parquet)]
    fn parquet(
        &mut self,
        path: &Bound<'_, PyAny>,
        mergeSchema: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        datetimeRebaseMode: Option<&Bound<'_, PyAny>>,
        int96RebaseMode: Option<&Bound<'_, PyAny>>,
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        crate::options::check_unexpected_keywords("DataStreamReader.parquet", unexpected)?;
        let mut r = self.state();
        r = self.skip_none_options(
            r,
            &[
                ("mergeSchema", mergeSchema),
                ("pathGlobFilter", pathGlobFilter),
                ("recursiveFileLookup", recursiveFileLookup),
                ("datetimeRebaseMode", datetimeRebaseMode),
                ("int96RebaseMode", int96RebaseMode),
            ],
        )?;
        // Like PySpark, the options stay on the reader, and the format of a str path.
        let (path, path_error) = self.format_path(&r, "parquet", path)?;
        let error = self.plan_error().or(path_error);
        Ok(PyDataFrame::new(with_plan_error(r.parquet(&path), error)))
    }

    #[pyo3(signature = (path, schema=None, sep=None, encoding=None, quote=None, escape=None, comment=None, header=None, inferSchema=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, maxMalformedLogPerPartition=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, charToEscapeQuoteEscaping=None, enforceSchema=None, emptyValue=None, locale=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, unescapedQuoteHandling=None, **unexpected), text_signature = "($self, path, schema=None, sep=None, encoding=None, quote=None, escape=None, comment=None, header=None, inferSchema=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, maxMalformedLogPerPartition=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, charToEscapeQuoteEscaping=None, enforceSchema=None, emptyValue=None, locale=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, unescapedQuoteHandling=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(DataStreamReader.csv)]
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
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        crate::options::check_unexpected_keywords("DataStreamReader.csv", unexpected)?;
        let mut r = self.state();
        r = self.apply_optional_schema(r, schema)?;
        r = self.skip_none_options(
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
        // Like PySpark, the options stay on the reader, and the format of a str path.
        let (path, path_error) = self.format_path(&r, "csv", path)?;
        let error = self.plan_error().or(path_error);
        Ok(PyDataFrame::new(with_plan_error(r.csv(&path), error)))
    }

    #[pyo3(signature = (path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None, **unexpected), text_signature = "($self, path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None)")]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(DataStreamReader.orc)]
    fn orc(
        &mut self,
        path: &Bound<'_, PyAny>,
        mergeSchema: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        crate::options::check_unexpected_keywords("DataStreamReader.orc", unexpected)?;
        let mut r = self.state();
        r = self.skip_none_options(
            r,
            &[
                ("mergeSchema", mergeSchema),
                ("pathGlobFilter", pathGlobFilter),
                ("recursiveFileLookup", recursiveFileLookup),
            ],
        )?;
        // Like PySpark, the options stay on the reader, and the format of a str path.
        let (path, path_error) = self.format_path(&r, "orc", path)?;
        let error = self.plan_error().or(path_error);
        Ok(PyDataFrame::new(with_plan_error(r.orc(&path), error)))
    }

    // As PySpark's `wholetext=False`, an omitted `wholetext` sets `False`, also over an earlier
    // `wholetext` option.
    #[pyo3(
        signature = (path, wholetext=DefaultedOption::Omitted, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, **unexpected),
        text_signature = "($self, path, wholetext=False, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None)"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(DataStreamReader.text)]
    fn text(
        &mut self,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        wholetext: DefaultedOption,
        lineSep: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        crate::options::check_unexpected_keywords("DataStreamReader.text", unexpected)?;
        let wholetext = wholetext.or_bool(py, false);
        let mut r = self.state();
        r = self.skip_none_options(
            r,
            &[
                ("wholetext", Some(&wholetext)),
                ("lineSep", lineSep),
                ("pathGlobFilter", pathGlobFilter),
                ("recursiveFileLookup", recursiveFileLookup),
            ],
        )?;
        // Like PySpark, the options stay on the reader, and the format of a str path. The load
        // is of the options set here: the core's `text` would set its own `wholetext`.
        let (path, path_error) = self.format_path(&r, "text", path)?;
        let error = self.plan_error().or(path_error);
        let df = r.format("text").load(Some(&path));
        Ok(PyDataFrame::new(with_plan_error(df, error)))
    }
}

/// Python wrapper for Trigger.
#[pyclass(
    name = "Trigger",
    module = "pyspark.sql.connect.streaming.readwriter",
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

/// The reader without the options of `options`, which the reader keeps apart from its core state.
fn without_options(
    reader: DataStreamReader,
    options: &HashMap<String, String>,
) -> DataStreamReader {
    options
        .keys()
        .fold(reader, |reader, key| reader.unset_option(key))
}

/// The kind and the value of a trigger, which a pickle keeps as a pair of Python values.
fn trigger_pair(trigger: &Trigger) -> (&'static str, Option<String>) {
    match trigger {
        Trigger::ProcessingTime(interval) => ("processingTime", Some(interval.clone())),
        Trigger::Once => ("once", None),
        Trigger::AvailableNow => ("availableNow", None),
        Trigger::Continuous(interval) => ("continuous", Some(interval.clone())),
        Trigger::RealTime(interval) => ("realTime", Some(interval.clone())),
    }
}

/// The trigger of a kind and a value, as `trigger_pair` gives them.
fn trigger_of(kind: &str, value: Option<String>) -> PyResult<Trigger> {
    let value = value.unwrap_or_default();
    match kind {
        "processingTime" => Ok(Trigger::ProcessingTime(value)),
        "once" => Ok(Trigger::Once),
        "availableNow" => Ok(Trigger::AvailableNow),
        "continuous" => Ok(Trigger::Continuous(value)),
        "realTime" => Ok(Trigger::RealTime(value)),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown trigger: {kind}"
        ))),
    }
}

// Python wrapper for DataStreamWriter.
#[pyclass(
    name = "DataStreamWriter",
    module = "pyspark.sql.connect.streaming.readwriter",
    weakref,
    dict
)]
pub struct PyDataStreamWriter {
    /// The state of the writer, which a shallow copy of it shares, as PySpark's `_write_stream`.
    /// It keeps the functions of `foreachBatch` and `foreach` as their pickles, as PySpark's
    /// `WriteStreamOperationStart` does, and not the functions.
    /// It holds the plan that is streamed, and not the DataFrame, as PySpark's writer holds the
    /// plan of its DataFrame.
    inner: Arc<WriterState<DataStreamWriter>>,
    /// The object of the parts of the state, which the shallow copies of this writer share and
    /// report, which reports them (see [`crate::dataframe::WriterParts`]).
    parts: Option<Py<crate::dataframe::WriterParts>>,
    /// Set once the value is written, as `__clear__` replaces `inner`, which is not valid while
    /// the object is created (see [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

impl PyDataStreamWriter {
    pub fn new(writer: DataStreamWriter) -> Self {
        let inner = Arc::new(WriterState::of_parts(writer, DataStreamWriter::held_parts));
        PyDataStreamWriter {
            parts: crate::dataframe::writer_parts(&inner),
            inner,
            gc_ready: crate::gc::GcReady::new(),
        }
    }

    /// The steps of PySpark's `_start_internal` on the writer `w`, in its order: the options, the output
    /// mode, the partitioning columns, the format and the query name. Each step sets its part of the
    /// writer in place, so a step that fails keeps the steps before it, and the writer is returned with
    /// the error of that step. The clear of the columns precedes their extension, which leaves none
    /// when it fails.
    fn start_steps(
        mut w: DataStreamWriter,
        format: Option<&Bound<'_, PyAny>>,
        output_mode: Option<&Bound<'_, PyAny>>,
        partition_by: Option<&Bound<'_, PyAny>>,
        query_name: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> (DataStreamWriter, PyResult<()>) {
        if options.is_some() {
            match crate::options::kwargs_skip_none(options) {
                Ok(opts) => w = w.options(opts),
                Err(error) => return (w, Err(error)),
            }
        }
        if let Some(m) = output_mode {
            match crate::str_argument(m) {
                Ok(mode) => w = w.output_mode(&mode),
                Err(error) => return (w, Err(error)),
            }
        }
        if let Some(pb) = partition_by {
            match stream_column_names(std::slice::from_ref(pb)) {
                Ok(names) => w = w.partition_by(names),
                Err(error) => return (w.partition_by(Vec::<String>::new()), Err(error)),
            }
        }
        if let Some(fmt) = format {
            match crate::str_argument(fmt) {
                Ok(source) => w = w.format(&source),
                Err(error) => return (w, Err(error)),
            }
        }
        if let Some(qn) = query_name {
            match query_name_of(qn) {
                Ok(name) => w = w.query_name(&name),
                Err(error) => return (w, Err(error)),
            }
        }
        (w, Ok(()))
    }

    /// Apply the one-shot writer options accepted by `start`/`toTable`
    /// (`format`, `outputMode`, `partitionBy`, `queryName`, `**options`) before
    /// terminating the stream. Mirrors the reference, where these are keyword args.
    fn apply_write_opts(
        &self,
        format: Option<&Bound<'_, PyAny>>,
        output_mode: Option<&Bound<'_, PyAny>>,
        partition_by: Option<&Bound<'_, PyAny>>,
        query_name: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<DataStreamWriter> {
        crate::options::check_keyword_keys(options)?;
        let (w, applied) = Self::start_steps(
            self.inner.get(),
            format,
            output_mode,
            partition_by,
            query_name,
            options,
        );
        // PySpark's start/toTable set these on the writer itself before starting, and keep each
        // step before a failing one.
        self.inner.set(w.clone());
        applied?;
        Ok(w)
    }
}

#[pymethods]
impl PyDataStreamWriter {
    /// The shallow copy of a stream writer, as PySpark's `copy.copy`: the copy shares the state
    /// of this writer, as its `_write_stream` is the same object, with the pickles of its
    /// functions.
    fn __copy__(slf: &Bound<'_, Self>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let this = slf.borrow();
        let copy = Py::new(
            py,
            PyDataStreamWriter {
                inner: Arc::clone(&this.inner),
                parts: this.parts.as_ref().map(|parts| parts.clone_ref(py)),
                gc_ready: crate::gc::GcReady::new(),
            },
        )?;
        crate::reduce::copy_instance_dict(slf.as_any(), copy.bind(py).as_any())?;
        Ok(copy)
    }

    /// As PySpark's stream writer, a Python object that holds the plan of its DataFrame, which
    /// Python's garbage collector sees: the object of the parts of its state, which a shallow
    /// copy shares, is reported, which reports the parts of the plan (see [`crate::gc`]). The
    /// functions of `foreachBatch` and `foreach` are kept as their pickles, as PySpark keeps them,
    /// so nothing else is held. Only an `Option<Py<_>>` is read, which is `None` while the object
    /// is created.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if let Some(parts) = &self.parts {
            visit.call(parts)?;
        }
        Ok(())
    }

    /// Drop this writer's reference to its state, of which it keeps a cleared copy; a copy of the
    /// writer keeps the state.
    fn __clear__(&mut self) {
        if self.gc_ready.get() {
            self.parts = None;
            self.inner = Arc::new(self.inner.cleared(DataStreamWriter::clear_plan));
        }
    }

    /// The reduce of a stream writer, as PySpark's pickle of its DataFrame and its state, which
    /// `_of_state` gives back, with the pickles of the functions that it applied, which PySpark's
    /// pickle keeps in its `WriteStreamOperationStart`, and its instance attributes.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        use crate::reduce::{state_reduce, str_dict};

        let py = slf.py();
        let this = slf.borrow();
        let writer = this.inner.get();

        let trigger = match writer.get_trigger() {
            Some(trigger) => {
                let (kind, value) = trigger_pair(trigger);
                PyTuple::new(
                    py,
                    [
                        kind.into_pyobject(py)?.into_any(),
                        value.into_pyobject(py)?.into_any(),
                    ],
                )?
                .into_any()
            }
            None => py.None().into_bound(py),
        };
        let commands = PyList::empty(py);
        for (kind, payload) in [
            ("foreachBatch", writer.get_foreach_batch_payload()),
            ("foreach", writer.get_foreach_payload()),
        ] {
            if let Some(payload) = payload {
                let command = pyo3::types::PyBytes::new(py, &payload.command);
                commands.append((kind, command, payload.python_ver.as_str()))?;
            }
        }
        // A DataFrame of the plan, which the pickle keeps as PySpark's keeps the plan.
        let frame = Bound::new(py, PyDataFrame::new(writer.dataframe()))?;
        let args = PyTuple::new(
            py,
            [
                frame.into_any(),
                writer.get_format().into_pyobject(py)?.into_any(),
                writer.get_output_mode().into_pyobject(py)?.into_any(),
                str_dict(py, writer.get_options())?.into_any(),
                PyList::new(py, writer.get_partition_cols())?.into_any(),
                PyList::new(py, writer.get_cluster_cols())?.into_any(),
                writer.get_query_name().into_pyobject(py)?.into_any(),
                trigger,
                commands.into_any(),
            ],
        )?;
        let reduced = state_reduce(py, "stream_writer_reduce", args)?;
        crate::reduce::reduce_with_instance_dict(slf.as_any(), reduced)
    }

    /// A stream writer of its state, as `__reduce__` gives it: of the DataFrame `frame`, the
    /// format, the output mode, the options, the partitioning and clustering columns, the query
    /// name, the trigger (a kind and its value, or None), and the pickles of the functions of
    /// `foreachBatch` and `foreach`, each of its kind and of the version of Python that pickled
    /// it, which the writer keeps as they are, as PySpark's unpickled writer does.
    #[staticmethod]
    fn _of_state<'py>(
        py: Python<'py>,
        frame: Py<PyDataFrame>,
        format: Option<String>,
        output_mode: Option<String>,
        options: HashMap<String, String>,
        partition_cols: Vec<String>,
        cluster_cols: Vec<String>,
        query_name: Option<String>,
        trigger: Option<(String, Option<String>)>,
        commands: Vec<(String, Vec<u8>, String)>,
    ) -> PyResult<Bound<'py, PyDataStreamWriter>> {
        let mut writer = frame.borrow(py).dataframe.write_stream();
        if let Some(format) = format {
            writer = writer.format(&format);
        }
        if let Some(output_mode) = output_mode {
            writer = writer.output_mode(&output_mode);
        }
        writer = writer.options(options);
        if !partition_cols.is_empty() {
            writer = writer.partition_by(partition_cols);
        }
        if !cluster_cols.is_empty() {
            writer = writer.cluster_by(cluster_cols);
        }
        if let Some(query_name) = query_name {
            writer = writer.query_name(&query_name);
        }
        if let Some((kind, value)) = trigger {
            writer = writer.trigger(trigger_of(&kind, value)?);
        }
        for (kind, command, python_ver) in commands {
            let payload = PythonUDFPayload::function(command, python_ver);
            writer = match kind.as_str() {
                "foreachBatch" => writer.foreach_batch(payload),
                _ => writer.foreach(payload),
            };
        }
        Bound::new(py, PyDataStreamWriter::new(writer))
    }

    // The output mode, which PySpark sets in its protobuf string field at once, as
    // [`crate::str_argument`] converts it.
    #[doc = connect_doc!(DataStreamWriter.outputMode)]
    fn outputMode<'py>(
        slf: &Bound<'py, Self>,
        outputMode: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let outputMode = crate::str_argument(outputMode)?;
        slf.borrow().inner.update(|w| w.output_mode(&outputMode));
        Ok(slf.clone())
    }

    // The format, which PySpark sets in its protobuf string field at once, as
    // [`crate::str_argument`] converts it.
    #[doc = connect_doc!(DataStreamWriter.format)]
    fn format<'py>(
        slf: &Bound<'py, Self>,
        source: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let source = crate::str_argument(source)?;
        slf.borrow().inner.update(|w| w.format(&source));
        Ok(slf.clone())
    }

    #[doc = connect_doc!(DataStreamWriter.option)]
    fn option<'py>(
        slf: &Bound<'py, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        // None -> option left unset, before the key is looked at, as PySpark's returns at once;
        // bools -> "true"/"false" (reference `to_str`). PySpark sets the option in the protobuf
        // map of its command at once, with the key as it is.
        if value.is_none() {
            return Ok(slf.clone());
        }
        let key = crate::str_argument(key)?;
        if let Some(v) = crate::coerce_option_value(value)? {
            slf.borrow().inner.update(|w| w.option(&key, &v));
        }
        Ok(slf.clone())
    }

    // Mirrors reference `DataStreamWriter.options(**options)`: keyword args; None values
    // skipped, booleans lowercased.
    #[pyo3(signature = (**options))]
    #[doc = connect_doc!(DataStreamWriter.options)]
    fn options<'py>(
        slf: &Bound<'py, Self>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        let opts = crate::options::kwargs_skip_none(options)?;
        slf.borrow().inner.update(|w| w.options(opts));
        Ok(slf.clone())
    }

    // Partition by the column names, or a list or tuple of them alone, as PySpark's.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataStreamWriter.partitionBy)]
    fn partitionBy<'py>(
        slf: &Bound<'py, Self>,
        cols: &Bound<'_, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        let items: Vec<Bound<'_, PyAny>> = cols.iter().collect();
        let writer = slf.borrow();
        let names = stream_column_names(&items);
        writer
            .inner
            .update(|w| w.partition_by(names.as_deref().unwrap_or_default()));
        names?;
        Ok(slf.clone())
    }

    // Cluster by the column names, or a list or tuple of them alone, as PySpark's.
    #[pyo3(signature = (*cols))]
    #[doc = connect_doc!(DataStreamWriter.clusterBy)]
    fn clusterBy<'py>(
        slf: &Bound<'py, Self>,
        cols: &Bound<'_, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        let items: Vec<Bound<'_, PyAny>> = cols.iter().collect();
        let writer = slf.borrow();
        let names = stream_column_names(&items);
        writer
            .inner
            .update(|w| w.cluster_by(names.as_deref().unwrap_or_default()));
        names?;
        Ok(slf.clone())
    }

    // Name the query, a non-empty str, else `VALUE_NOT_NON_EMPTY_STR`, as PySpark's.
    #[doc = connect_doc!(DataStreamWriter.queryName)]
    fn queryName<'py>(
        slf: &Bound<'py, Self>,
        queryName: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let name = query_name_of(queryName)?;
        slf.borrow().inner.update(|w| w.query_name(&name));
        Ok(slf.clone())
    }

    // `DataStreamWriter.trigger(...)`, as PySpark's: exactly one of `processingTime`, `once`,
    // `continuous`, `availableNow` and `realTime`, else `ONLY_ALLOW_SINGLE_TRIGGER`; an
    // interval or duration is a non-empty string, which is stripped, else
    // `VALUE_NOT_NON_EMPTY_STR`, and `once` and `availableNow` are True, else
    // `VALUE_NOT_ALLOWED`.
    #[pyo3(signature = (*, processingTime=None, once=None, continuous=None, availableNow=None, realTime=None))]
    #[doc = connect_doc!(DataStreamWriter.trigger)]
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
    #[doc = connect_doc!(DataStreamWriter.start)]
    fn start<'py>(
        &self,
        py: Python<'py>,
        path: Option<&Bound<'_, PyAny>>,
        format: Option<&Bound<'_, PyAny>>,
        outputMode: Option<&Bound<'_, PyAny>>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        queryName: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyStreamingQuery>> {
        let writer = self.apply_write_opts(format, outputMode, partitionBy, queryName, options)?;
        // PySpark's start sets these on the writer before it checks the path, which it sets in
        // its string field only `if path:`, so that a false one, such as 0, is no path.
        let path = match path {
            Some(path) if path.is_truthy()? => Some(crate::readwriter::writer_path(path)?),
            _ => None,
        };
        // Memory/console/foreach sinks take no path; the core treats "" as unset. Of the plan and
        // the picklers of the writer, which the command is built of.
        let _in_use = crate::gc::in_use_of_values(py, writer.held_parts());
        let query = interruptible(py, &writer.spark_session(), || {
            writer
                .start_with_event_handler(path.as_deref().unwrap_or(""), post_started_event)
                .map_err(Box::new)
        })?;
        // As PySpark's `StreamingQuery(self._session, ...)`, of the session object of the writer.
        let session = crate::session::session_object(py, &query.spark_session())?;
        PyStreamingQuery::of_query(session.bind(py).as_any(), &query)
    }

    #[pyo3(signature = (tableName, format=None, outputMode=None, partitionBy=None, queryName=None, **options))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(DataStreamWriter.toTable)]
    fn toTable<'py>(
        &self,
        py: Python<'py>,
        tableName: &Bound<'_, PyAny>,
        format: Option<&Bound<'_, PyAny>>,
        outputMode: Option<&Bound<'_, PyAny>>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        queryName: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyStreamingQuery>> {
        let writer = self.apply_write_opts(format, outputMode, partitionBy, queryName, options)?;
        // As the path of `start`, PySpark sets the table in its string field only `if
        // tableName:`, so that a false one, such as "", starts the query as `start()` does.
        let table = if tableName.is_truthy()? {
            Some(crate::str_argument(tableName)?)
        } else {
            None
        };
        // Of the plan and the picklers of the writer, which the command is built of.
        let _in_use = crate::gc::in_use_of_values(py, writer.held_parts());
        let query = interruptible(py, &writer.spark_session(), || {
            match &table {
                Some(table) => writer.to_table_with_event_handler(table, post_started_event),
                None => writer.start_with_event_handler("", post_started_event),
            }
            .map_err(Box::new)
        })?;
        let session = crate::session::session_object(py, &query.spark_session())?;
        PyStreamingQuery::of_query(session.bind(py).as_any(), &query)
    }

    // `DataStreamWriter.foreachBatch(func)`: cloudpickle the `(batch_df, batch_id)`
    // function (with PySpark's `CloudPickleSerializer`) and attach it as the
    // foreach-batch sink. Pickling happens here so the Python skin can re-export this
    // class directly rather than subclassing a (non-subclassable) PyO3 type. The writer keeps
    // the pickle and not the function, as PySpark's `WriteStreamOperationStart` does.
    #[doc = connect_doc!(DataStreamWriter.foreachBatch)]
    fn foreachBatch<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let command = streaming_function_command(py, func, "foreachBatch")?;
        // As PySpark's, the function's command and Python version only.
        let payload = PythonUDFPayload::function(command, crate::dataframe::py_version(py));
        slf.borrow().inner.update(|w| w.foreach_batch(payload));

        Ok(slf.clone())
    }

    // `DataStreamWriter.foreach(f)`, as PySpark's: the function of the partition and the
    // rows of each partition of `_construct_foreach_function(f)`, which raises
    // `ATTRIBUTE_NOT_CALLABLE` for an object without a callable `process`, in the command
    // `(func, None, serializer, serializer)` of `AutoBatchedSerializer(CPickleSerializer())`,
    // which the worker of the server deserializes with its own `pyspark.serializers`. Built
    // here to avoid a Python subclass of the PyO3 class. The writer keeps the pickle and not
    // `f`, as PySpark's `WriteStreamOperationStart` does.
    #[doc = connect_doc!(DataStreamWriter.foreach)]
    fn foreach<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'_>,
        f: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let func = py
            .import("pyspark.sql.connect.streaming.readwriter")?
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
        let payload = PythonUDFPayload::function(command, crate::dataframe::py_version(py));
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

/// The `timeout_ms` of the `timeout` of `awaitTermination` and `awaitAnyTermination`, in seconds,
/// as PySpark's computes it, of Python's own operators: an int or a float that is not `<= 0`,
/// else `VALUE_NOT_POSITIVE` of the name of its type, and then `int(timeout * 1000)`, which the
/// request sets in its int64 field. So a NaN, which is not `<= 0`, raises the `ValueError` of
/// `int`, an infinity its `OverflowError`, and a count beyond a long protobuf's `ValueError`, of
/// the digits of the int; an int is exact, where a float would round it.
fn pyspark_timeout_ms(timeout: Option<&Bound<'_, PyAny>>) -> PyResult<Option<i64>> {
    let Some(timeout) = timeout else {
        return Ok(None);
    };
    let py = timeout.py();
    let number = timeout.is_instance_of::<pyo3::types::PyInt>()
        || timeout.is_instance_of::<pyo3::types::PyFloat>();
    // `not isinstance(timeout, (int, float)) or timeout <= 0`.
    if !number || timeout.le(0)? {
        let type_name = timeout.get_type().name()?.to_string();
        return Err(crate::column::raise_pyspark(
            py,
            "PySparkValueError",
            "VALUE_NOT_POSITIVE",
            &[("arg_name", "timeout"), ("arg_value", &type_name)],
        ));
    }
    let ms = py
        .get_type::<pyo3::types::PyInt>()
        .call1((timeout.mul(1000)?,))?;
    match ms.extract::<i64>() {
        Ok(ms) => Ok(Some(ms)),
        // As protobuf's int64 field, which refuses an int beyond a long.
        Err(_) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Value out of range: {}",
            ms.str()?
        ))),
    }
}

/// A bool argument that PySpark sets in a proto bool field, taken as it is (None when it is
/// omitted, its `default()`) and converted in the call by [`ProtoBool::value`], as PyO3 would
/// prefix the `TypeError` of the conversion of an argument with its name.
#[derive(Default)]
pub(crate) struct ProtoBool(Option<Py<PyAny>>);

impl<'a, 'py> FromPyObject<'a, 'py> for ProtoBool {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(ProtoBool(Some(obj.to_owned().unbind())))
    }
}

impl ProtoBool {
    /// The bool of the argument, false when it is omitted, as [`proto_bool`] converts it.
    pub(crate) fn value(&self, py: Python<'_>) -> PyResult<bool> {
        match &self.0 {
            Some(obj) => proto_bool(obj.bind(py)),
            None => Ok(false),
        }
    }

    /// As [`ProtoBool::value`], of an argument that PySpark's request sets unless it is None,
    /// such as the `blocking` of `unpersist`: None for None, which the request does not set.
    pub(crate) fn value_unless_none(&self, py: Python<'_>) -> PyResult<Option<bool>> {
        match &self.0 {
            Some(obj) if obj.is_none(py) => Ok(None),
            _ => self.value(py).map(Some),
        }
    }
}

/// The bool of `obj`, as protobuf's default (upb) backend converts it for a bool field: a NumPy
/// bool by its truth, a NumPy array is protobuf's `TypeError`, and any other object is read as a
/// C long, true when it is not 0, which raises the `TypeError` of an object that is not an int,
/// and the `OverflowError` of an int that a C long cannot hold.
pub(crate) fn proto_bool(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let typ = obj.get_type();
    let module = typ.module().ok();
    if module.is_some_and(|module| module.to_cow().is_ok_and(|module| module == "numpy")) {
        let name = typ.name()?;
        let name = name.to_cow()?;
        if name == "ndarray" {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{} has type ndarray, but expected one of: bool",
                obj.str()?
            )));
        }
        if name == "bool" {
            return obj.is_truthy();
        }
    }
    Ok(obj.extract::<std::os::raw::c_long>()? != 0)
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

// Python wrapper for StreamingQuery. As PySpark's plain class, its state is its `__dict__`: the
// `_session`, `_query_id`, `_run_id` and `_name` that its `__init__` sets, of any objects, which
// `id`, `runId` and `name` give back and its methods read when they make their requests.
#[pyclass(
    name = "StreamingQuery",
    module = "pyspark.sql.connect.streaming.query",
    subclass,
    weakref,
    dict
)]
pub struct PyStreamingQuery {}

impl PyStreamingQuery {
    /// The query of `query`, of the session object `session`, as PySpark's manager and writer make
    /// it: `StreamingQuery(session, queryId, runId, name)`.
    pub(crate) fn of_query<'py>(
        session: &Bound<'py, PyAny>,
        query: &StreamingQuery,
    ) -> PyResult<Bound<'py, PyStreamingQuery>> {
        let py = session.py();
        let made = Bound::new(py, PyStreamingQuery {})?;
        let name = query.name().into_pyobject(py)?.into_any();
        Self::__init__(
            &made,
            session,
            PyString::new(py, query.id()).as_any(),
            PyString::new(py, query.run_id()).as_any(),
            Some(&name),
        )?;
        Ok(made)
    }

    /// The core query of the state of `slf`, as PySpark's `_execute_streaming_query_cmd` reads it
    /// at each request: its `_query_id` and its `_run_id`, then the client of its `_session`.
    fn query_of(slf: &Bound<'_, Self>) -> PyResult<StreamingQuery> {
        let py = slf.py();
        let query_id: String =
            crate::state::attr(slf.as_any(), pyo3::intern!(py, "_query_id"))?.extract()?;
        let run_id: String =
            crate::state::attr(slf.as_any(), pyo3::intern!(py, "_run_id"))?.extract()?;
        let session = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_session"))?;
        let session = crate::state::session_of_value(&session, pyo3::intern!(py, "client"))?;
        Ok(StreamingQuery::from_parts(session, query_id, run_id, None))
    }
}

#[pymethods]
impl PyStreamingQuery {
    #[new]
    #[pyo3(
        signature = (*_args, **_kwargs),
        text_signature = "(session, queryId, runId, name=None)"
    )]
    fn py_new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyStreamingQuery {}
    }

    /// `StreamingQuery(session, queryId, runId, name=None)`, as PySpark Connect's `__init__`,
    /// which keeps its arguments as they are.
    #[pyo3(signature = (session, queryId, runId, name=None))]
    #[allow(non_snake_case)]
    fn __init__(
        slf: &Bound<'_, Self>,
        session: &Bound<'_, PyAny>,
        queryId: &Bound<'_, PyAny>,
        runId: &Bound<'_, PyAny>,
        name: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let none = py.None().into_bound(py);
        crate::state::assign(
            slf.as_any(),
            &[
                (pyo3::intern!(py, "_session"), session),
                (pyo3::intern!(py, "_query_id"), queryId),
                (pyo3::intern!(py, "_run_id"), runId),
                (pyo3::intern!(py, "_name"), name.unwrap_or(&none)),
            ],
        )
    }

    /// The state of the query, its `__dict__`, as `object.__getstate__` gives that of PySpark's
    /// plain object, which a copy and a pickle keep (see `state::getstate`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::getstate(slf.as_any())
    }

    /// The reduce of the query, as PySpark's plain object's: `copyreg.__newobj__` of its class, of
    /// its `__dict__` (see `state::reduce_ex`).
    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, protocol: i32) -> PyResult<Bound<'py, PyAny>> {
        crate::state::reduce_ex(slf.as_any(), protocol)
    }

    #[getter]
    #[doc = connect_doc!(StreamingQuery.id)]
    fn id<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::attr(slf.as_any(), pyo3::intern!(slf.py(), "_query_id"))
    }

    #[getter]
    #[doc = connect_doc!(StreamingQuery.runId)]
    fn runId<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::attr(slf.as_any(), pyo3::intern!(slf.py(), "_run_id"))
    }

    #[getter]
    #[doc = connect_doc!(StreamingQuery.name)]
    fn name<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::attr(slf.as_any(), pyo3::intern!(slf.py(), "_name"))
    }

    #[getter]
    #[doc = connect_doc!(StreamingQuery.isActive)]
    fn isActive(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let query = Self::query_of(slf)?;
        crate::interrupt::interruptible_without_parts(slf.py(), &query.spark_session(), || {
            query.is_active()
        })
    }

    // The status of the query, as PySpark's: a dict of its `message`, `isDataAvailable` and
    // `isTriggerActive`.
    #[getter]
    #[doc = connect_doc!(StreamingQuery.status)]
    fn status<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let query = Self::query_of(slf)?;
        let status =
            crate::interrupt::interruptible_without_parts(py, &query.spark_session(), || {
                query.status().map_err(Box::new)
            })?;
        let dict = PyDict::new(py);
        dict.set_item("message", status.status_message)?;
        dict.set_item("isDataAvailable", status.is_data_available)?;
        dict.set_item("isTriggerActive", status.is_trigger_active)?;
        Ok(dict)
    }

    #[doc = connect_doc!(StreamingQuery.stop)]
    fn stop(slf: &Bound<'_, Self>) -> PyResult<()> {
        let query = Self::query_of(slf)?;
        crate::interrupt::interruptible_without_parts(slf.py(), &query.spark_session(), || {
            query.stop().map_err(Box::new)
        })
    }

    #[pyo3(signature = (timeout=None))]
    #[doc = connect_doc!(StreamingQuery.awaitTermination)]
    fn awaitTermination(
        slf: &Bound<'_, Self>,
        timeout: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<bool>> {
        let timeout_ms = pyspark_timeout_ms(timeout)?;
        let query = Self::query_of(slf)?;
        crate::interrupt::interruptible_without_parts(slf.py(), &query.spark_session(), || {
            query.await_termination_ms(timeout_ms).map_err(Box::new)
        })
    }

    // The last progress of the query, as PySpark's `StreamingQueryProgress`, or None.
    #[getter]
    #[doc = connect_doc!(StreamingQuery.lastProgress)]
    fn lastProgress<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = slf.py();
        let query = Self::query_of(slf)?;
        let progress =
            crate::interrupt::interruptible_without_parts(py, &query.spark_session(), || {
                query.last_progress().map_err(Box::new)
            })?;
        match progress {
            Some(json) => progress_from_json(py, &json).map(Some),
            None => Ok(None),
        }
    }

    // The recent progress of the query, each as PySpark's `StreamingQueryProgress`.
    #[getter]
    #[doc = connect_doc!(StreamingQuery.recentProgress)]
    fn recentProgress<'py>(slf: &Bound<'py, Self>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let py = slf.py();
        let query = Self::query_of(slf)?;
        crate::interrupt::interruptible_without_parts(py, &query.spark_session(), || {
            query.recent_progress().map_err(Box::new)
        })?
        .iter()
        .map(|json| progress_from_json(py, json))
        .collect()
    }

    #[doc = connect_doc!(StreamingQuery.processAllAvailable)]
    fn processAllAvailable(slf: &Bound<'_, Self>) -> PyResult<()> {
        let query = Self::query_of(slf)?;
        crate::interrupt::interruptible_without_parts(slf.py(), &query.spark_session(), || {
            query.process_all_available().map_err(Box::new)
        })
    }

    // Print the plan of the query with Python's `print`, as PySpark's
    // `StreamingQuery.explain(extended=False)`.
    #[pyo3(signature = (extended=ProtoBool(None)), text_signature = "($self, extended=False)")]
    #[doc = connect_doc!(StreamingQuery.explain)]
    fn explain(slf: &Bound<'_, Self>, extended: ProtoBool) -> PyResult<()> {
        let py = slf.py();
        let extended = extended.value(py)?;
        let query = Self::query_of(slf)?;
        let plan =
            crate::interrupt::interruptible_without_parts(py, &query.spark_session(), || {
                query.explain(extended).map_err(Box::new)
            })?;
        py.import("builtins")?.call_method1("print", (plan,))?;
        Ok(())
    }

    // The exception of a failed query, as PySpark's: the `StreamingQueryException` of
    // `pyspark.errors.exceptions.connect`, of the message of the server without the type of
    // its Java exception, and with its JVM stack trace, and of the error class as its reason.
    #[doc = connect_doc!(StreamingQuery.exception)]
    fn exception(slf: &Bound<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let query = Self::query_of(slf)?;
        let exception =
            crate::interrupt::interruptible_without_parts(py, &query.spark_session(), || {
                query.exception().map_err(Box::new)
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

// Python wrapper for StreamingQueryManager. As PySpark's plain class, its state is its
// `__dict__`: the `_session` that its `__init__` sets, of any object, of whose client its methods
// make their requests, and the `_sqlb` that it makes, its `StreamingQueryListenerBus`, which its
// listeners are added to and removed from.
#[pyclass(
    name = "StreamingQueryManager",
    module = "pyspark.sql.connect.streaming.query",
    subclass,
    weakref,
    dict
)]
pub struct PyStreamingQueryManager {}

impl PyStreamingQueryManager {
    /// The manager of `session`, as `spark.streams` makes it: `StreamingQueryManager(spark)`.
    pub(crate) fn of_session<'py>(
        session: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyStreamingQueryManager>> {
        let manager = Bound::new(session.py(), PyStreamingQueryManager {})?;
        Self::__init__(&manager, session)?;
        Ok(manager)
    }

    /// The `_session` of the manager and the core manager of its client, as PySpark's
    /// `_execute_streaming_query_manager_cmd` reads them at each request.
    fn manager_of<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, StreamingQueryManager)> {
        let py = slf.py();
        let session = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_session"))?;
        let core = crate::state::session_of_value(&session, pyo3::intern!(py, "client"))?;
        Ok((session, core.streams()))
    }
}

/// Warn of `message`, as PySpark Connect's listener bus warns of a failure to add or remove a
/// listener, rather than raising, of the line `line` of that `warnings.warn` in
/// `pyspark.sql.connect.streaming.query` (see `warn_as_pyspark`).
fn warn(py: Python<'_>, line: u32, message: String) -> PyResult<()> {
    crate::errors::warn_as_pyspark(
        py,
        "pyspark.sql.connect.streaming.query",
        line,
        &message,
        &py.get_type::<pyo3::exceptions::PyUserWarning>(),
    )
}

/// The warning of PySpark's bus of a removal that failed of `error`.
fn removal_failed(py: Python<'_>, error: impl std::fmt::Display) -> PyResult<()> {
    // Of PySpark's `StreamingQueryListenerBus.remove`.
    warn(
        py,
        332,
        format!(
            "Failed to remove the listener because of exception: {error}\n\
             The listener is not removed, please remove it again."
        ),
    )
}

#[pymethods]
impl PyStreamingQueryManager {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(session)")]
    fn py_new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyStreamingQueryManager {}
    }

    /// `StreamingQueryManager(session)`, as PySpark Connect's `__init__`, which keeps its argument
    /// as it is and makes its listener bus, `StreamingQueryListenerBus(self)`, of the class of the
    /// module.
    fn __init__(slf: &Bound<'_, Self>, session: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        crate::state::assign(slf.as_any(), &[(pyo3::intern!(py, "_session"), session)])?;
        let bus = py
            .import("pyspark.sql.connect.streaming.query")?
            .getattr(pyo3::intern!(py, "StreamingQueryListenerBus"))?
            .call1((slf,))?;
        crate::state::assign(slf.as_any(), &[(pyo3::intern!(py, "_sqlb"), &bus)])
    }

    /// The state of the manager, its `__dict__`, as `object.__getstate__` gives that of PySpark's
    /// plain object, which a copy and a pickle keep (see `state::getstate`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::getstate(slf.as_any())
    }

    /// The reduce of the manager, as PySpark's plain object's: `copyreg.__newobj__` of its class,
    /// of its `__dict__` (see `state::reduce_ex`).
    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, protocol: i32) -> PyResult<Bound<'py, PyAny>> {
        crate::state::reduce_ex(slf.as_any(), protocol)
    }

    // Remove all client-side listeners of the bus and stop its subscription, as PySpark's
    // `self._sqlb.close()`.
    fn close(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        crate::state::attr(slf.as_any(), pyo3::intern!(py, "_sqlb"))?
            .call_method0(pyo3::intern!(py, "close"))?;
        Ok(())
    }

    #[getter]
    #[doc = connect_doc!(StreamingQueryManager.active)]
    fn active<'py>(slf: &Bound<'py, Self>) -> PyResult<Vec<Bound<'py, PyStreamingQuery>>> {
        let (session, manager) = Self::manager_of(slf)?;
        let queries = crate::interrupt::interruptible_without_parts(
            slf.py(),
            &manager.spark_session(),
            || manager.active().map_err(Box::new),
        )?;
        queries
            .iter()
            .map(|query| PyStreamingQuery::of_query(&session, query))
            .collect()
    }

    #[doc = connect_doc!(StreamingQueryManager.get)]
    fn get<'py>(
        slf: &Bound<'py, Self>,
        id: &str,
    ) -> PyResult<Option<Bound<'py, PyStreamingQuery>>> {
        let (session, manager) = Self::manager_of(slf)?;
        let query = crate::interrupt::interruptible_without_parts(
            slf.py(),
            &manager.spark_session(),
            || manager.get(id).map_err(Box::new),
        )?;
        query
            .map(|query| PyStreamingQuery::of_query(&session, &query))
            .transpose()
    }

    #[pyo3(signature = (timeout=None))]
    #[doc = connect_doc!(StreamingQueryManager.awaitAnyTermination)]
    fn awaitAnyTermination(
        slf: &Bound<'_, Self>,
        timeout: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<bool>> {
        let timeout_ms = pyspark_timeout_ms(timeout)?;
        let (_, manager) = Self::manager_of(slf)?;
        crate::interrupt::interruptible_without_parts(slf.py(), &manager.spark_session(), || {
            manager
                .await_any_termination_ms(timeout_ms)
                .map_err(Box::new)
        })
    }

    #[doc = connect_doc!(StreamingQueryManager.resetTerminated)]
    fn resetTerminated(slf: &Bound<'_, Self>) -> PyResult<()> {
        let (_, manager) = Self::manager_of(slf)?;
        crate::interrupt::interruptible_without_parts(slf.py(), &manager.spark_session(), || {
            manager.reset_terminated().map_err(Box::new)
        })
    }

    // Register a client-side listener object (with onQueryStarted/Progress/Idle/Terminated
    // callbacks), as PySpark's `listener._set_spark_session(self._session)` and
    // `self._sqlb.append(listener)`: of the session of the manager, to its listener bus.
    #[doc = connect_doc!(StreamingQueryManager.addListener)]
    fn addListener(slf: &Bound<'_, Self>, listener: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let session = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_session"))?;
        listener.call_method1(pyo3::intern!(py, "_set_spark_session"), (session,))?;
        crate::state::attr(slf.as_any(), pyo3::intern!(py, "_sqlb"))?
            .call_method1(pyo3::intern!(py, "append"), (listener,))?;
        Ok(())
    }

    // Remove a client-side listener object, as PySpark's `self._sqlb.remove(listener)`.
    #[doc = connect_doc!(StreamingQueryManager.removeListener)]
    fn removeListener(slf: &Bound<'_, Self>, listener: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        crate::state::attr(slf.as_any(), pyo3::intern!(py, "_sqlb"))?
            .call_method1(pyo3::intern!(py, "remove"), (listener,))?;
        Ok(())
    }

    /// The events of the listener bus of the server, an iterator of `(event type, JSON)`, of the
    /// Rust core's listener event stream. A private name, as PySpark's manager has no such
    /// method: its listener bus reads the events itself.
    fn _stream_listener_events(slf: &Bound<'_, Self>) -> PyResult<PyListenerEventStream> {
        let (_, manager) = Self::manager_of(slf)?;
        let session = manager.spark_session();
        let stream = crate::interrupt::interruptible_without_parts(slf.py(), &session, || {
            manager.listener_event_stream().map_err(Box::new)
        })?;
        Ok(PyListenerEventStream::new(stream, session))
    }
}

// Python wrapper for StreamingQueryListenerBus, the listener bus of a StreamingQueryManager, its
// `_sqlb`. As PySpark's plain class, its state is its `__dict__`: the `_sqm` that its `__init__`
// sets, of any object, the `_listener_bus` list of its listeners, which `append` and `remove`
// change and `post_to_all` posts the events to, the `_execution_thread` and the `_lock`, which
// they hold while they read and change the list. The subscription of the session to the events of
// the server is the native bus's, of a registration of each entry that `append` adds, whose thread
// posts the events: `_execution_thread` stays None, as no thread of Python runs it, and the lock is
// not held while the listeners are called, so that a listener may add or remove listeners (see
// `dev/parity/known-differences.md`).
#[pyclass(
    name = "StreamingQueryListenerBus",
    module = "pyspark.sql.connect.streaming.query",
    subclass,
    weakref,
    dict
)]
#[doc = connect_doc!(StreamingQueryListenerBus)]
pub struct PyStreamingQueryListenerBus {}

impl PyStreamingQueryListenerBus {
    /// The core manager of the client of the `_session` of the `_sqm` of the bus, as PySpark's bus
    /// reads them when it makes a request.
    fn manager_of(slf: &Bound<'_, Self>) -> PyResult<StreamingQueryManager> {
        let py = slf.py();
        let sqm = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_sqm"))?;
        let session = crate::state::attr(&sqm, pyo3::intern!(py, "_session"))?;
        Ok(crate::state::session_of_value(&session, pyo3::intern!(py, "client"))?.streams())
    }

    /// The registrations of the entries of `entries`, a list of the bus, in the native bus of
    /// `manager`, in its order.
    fn registrations(
        manager: &StreamingQueryManager,
        entries: &Bound<'_, PyAny>,
    ) -> Vec<Arc<dyn StreamingQueryListener>> {
        manager
            .listeners()
            .into_iter()
            .filter(|listener| {
                PyListenerAdapter::of(listener.as_ref()).is_some_and(|adapter| {
                    adapter
                        .entry
                        .as_ref()
                        .is_some_and(|entry| entry.entries.bind(entries.py()).is(entries))
                })
            })
            .collect()
    }

    /// Mark `registrations`, of [`registrations`](Self::registrations), as ended of the bus itself,
    /// or not.
    fn release(registrations: &[Arc<dyn StreamingQueryListener>], released: bool) {
        for registration in registrations {
            if let Some(adapter) = PyListenerAdapter::of(registration.as_ref()) {
                adapter.released.store(released, Ordering::Release);
            }
        }
    }
}

#[pymethods]
impl PyStreamingQueryListenerBus {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(sqm)")]
    fn py_new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyStreamingQueryListenerBus {}
    }

    /// `StreamingQueryListenerBus(sqm)`, as PySpark Connect's `__init__`: its manager, kept as it
    /// is, an empty list of listeners, no thread, and a new `threading.Lock`.
    fn __init__(slf: &Bound<'_, Self>, sqm: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        static LOCK: pyo3::sync::PyOnceLock<Py<PyAny>> = pyo3::sync::PyOnceLock::new();
        let lock = LOCK.import(py, "threading", "Lock")?.call0()?;
        let none = py.None().into_bound(py);
        crate::state::assign(
            slf.as_any(),
            &[
                (pyo3::intern!(py, "_sqm"), sqm),
                (
                    pyo3::intern!(py, "_listener_bus"),
                    PyList::empty(py).as_any(),
                ),
                (pyo3::intern!(py, "_execution_thread"), &none),
                (pyo3::intern!(py, "_lock"), &lock),
            ],
        )
    }

    /// The state of the bus, its `__dict__`, as `object.__getstate__` gives that of PySpark's
    /// plain object, which a copy and a pickle keep (see `state::getstate`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::getstate(slf.as_any())
    }

    /// The reduce of the bus, as PySpark's plain object's: `copyreg.__newobj__` of its class, of
    /// its `__dict__` (see `state::reduce_ex`).
    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, protocol: i32) -> PyResult<Bound<'py, PyAny>> {
        crate::state::reduce_ex(slf.as_any(), protocol)
    }

    // Remove all the listeners of the bus and end the subscription of the session, as PySpark's
    // `close` removes each listener of its list: a server removal failure warns and keeps the
    // listeners registered, as PySpark Connect's does, so close can be retried. A bus of no
    // listeners does nothing.
    fn close(slf: &Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        let entries = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_listener_bus"))?;
        if !entries.is_truthy()? {
            return Ok(());
        }
        let manager = match Self::manager_of(slf) {
            Ok(manager) => manager,
            Err(error) => return removal_failed(py, error.value(py)),
        };
        // The registrations of the list, which the close ends, as the removal of each of its
        // entries does.
        let registrations = Self::registrations(&manager, &entries);
        Self::release(&registrations, true);
        let closed = crate::interrupt::interruptible_caught_without_parts(
            py,
            &manager.spark_session(),
            || manager.close().map_err(Box::new),
        );
        match closed {
            Ok(Ok(())) => {
                drop(registrations);
                let lock = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_lock"))?;
                with_lock(&lock, || {
                    entries.call_method0(pyo3::intern!(py, "clear"))?;
                    Ok(())
                })
            }
            Ok(Err(error)) => {
                Self::release(&registrations, false);
                removal_failed(py, spark_error_to_py_exception(*error).value(py))
            }
            Err(error) => {
                Self::release(&registrations, false);
                Err(error)
            }
        }
    }

    // Append a listener to the bus, as PySpark's: to its list, and to the native bus of the
    // session, which subscribes the session to the events of the server with its first listener.
    // Should the server fail to add its listener, this warns and takes the listener off the list,
    // as PySpark Connect's does. A signal that ends the wait for the server leaves the listener in
    // the bus, as PySpark's append, which appends it first, leaves it.
    #[doc = connect_doc!(StreamingQueryListenerBus.append)]
    fn append(slf: &Bound<'_, Self>, listener: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let lock = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_lock"))?;
        let entries = with_lock(&lock, || {
            let entries = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_listener_bus"))?;
            entries.call_method1(pyo3::intern!(py, "append"), (listener,))?;
            Ok(entries)
        })?;
        // As PySpark's, whose `except Exception` of the registration warns, and removes the
        // listener of its list.
        let failed = |error: String| -> PyResult<()> {
            with_lock(&lock, || remove_equal(&entries, listener))?;
            // Of PySpark's `StreamingQueryListenerBus.append`.
            warn(
                py,
                297,
                format!(
                    "Failed to add the listener because of exception: {error}\n\
                     The listener is not added, please add it again."
                ),
            )
        };
        let manager = match Self::manager_of(slf) {
            Ok(manager) => manager,
            Err(error) => return failed(error.value(py).to_string()),
        };
        let adapter = Arc::new(PyListenerAdapter {
            bus: slf.clone().into_any().unbind(),
            entry: Some(ListenerEntry {
                entries: entries.clone().unbind(),
                listener: listener.clone().unbind(),
            }),
            id: std::sync::OnceLock::new(),
            released: AtomicBool::new(false),
        });
        let registration: Arc<dyn StreamingQueryListener> = adapter.clone();
        // As PySpark's append, whose `except Exception` also catches the error of an interrupt
        // that the server failed, which replaces the KeyboardInterrupt of a signal.
        let added = crate::interrupt::interruptible_caught_without_parts(
            py,
            &manager.spark_session(),
            || {
                manager
                    .add_listener(registration)
                    .map(|id| {
                        // Set once, by the only call that adds this adapter, as soon as the bus has
                        // the registration: a signal that ends the wait below then leaves it
                        // removable, as PySpark's append leaves the listener in its bus.
                        let _ = adapter.id.set(id);
                    })
                    .map_err(Box::new)
            },
        );
        match added {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => {
                adapter.released.store(true, Ordering::Release);
                // The registration that an add that a signal ended keeps.
                if let Some(id) = adapter.id.get() {
                    manager.discard_listener(id);
                }
                failed(spark_error_to_py_exception(*error).value(py).to_string())
            }
            Err(error) => {
                // A signal that ended the add before the native bus had the registration: the
                // entry goes with it. One that the native bus keeps stays, with its entry.
                if adapter.id.get().is_none() {
                    adapter.released.store(true, Ordering::Release);
                    with_lock(&lock, || remove_identical(&entries, listener))?;
                }
                Err(error)
            }
        }
    }

    // Remove a listener of the bus, as PySpark's: the first entry of its list that is the
    // listener, or equal to it, as PySpark's `in` and `list.remove` find it, and its registration
    // in the native bus of the session, whose last one ends the subscription, as PySpark's
    // removal of its last listener does. Should the server fail to remove its listener, this warns,
    // and the listener stays, as PySpark Connect's does. A registration that is still being added
    // by another thread is after this removal, which then leaves its entry.
    #[doc = connect_doc!(StreamingQueryListenerBus.remove)]
    fn remove(slf: &Bound<'_, Self>, listener: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let lock = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_lock"))?;
        let (entries, entry) = with_lock(&lock, || {
            let entries = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_listener_bus"))?;
            let mut found = None;
            for item in entries.try_iter()? {
                let item = item?;
                if item.is(listener) || item.eq(listener)? {
                    found = Some(item);
                    break;
                }
            }
            Ok((entries, found))
        })?;
        let Some(entry) = entry else {
            return Ok(());
        };
        let manager = match Self::manager_of(slf) {
            Ok(manager) => manager,
            Err(error) => return removal_failed(py, error.value(py)),
        };
        // The registrations that this call did not remove: another call removed them first,
        // or their removal is left to the caller that is closing the bus.
        let mut tried: Vec<String> = Vec::new();
        loop {
            let mut registered = false;
            let mut pending = false;
            let mut first = None;
            for registration in manager.listeners() {
                let Some(adapter) = PyListenerAdapter::of(registration.as_ref()) else {
                    continue;
                };
                if !adapter.is_entry(&entries, &entry) {
                    continue;
                }
                registered = true;
                // A registration without an id is still being added by another thread: this
                // removal is before it.
                let Some(id) = adapter.id.get() else {
                    pending = true;
                    continue;
                };
                if tried.contains(id) {
                    continue;
                }
                first = Some((id.clone(), Arc::clone(&registration)));
                break;
            }
            let Some((id, registration)) = first else {
                // An entry of no registration, such as one that the native bus ended, or that a
                // user put on the list, is taken off it.
                if !registered && !pending {
                    with_lock(&lock, || remove_equal(&entries, listener))?;
                }
                return Ok(());
            };
            let adapter = PyListenerAdapter::of(registration.as_ref()).expect("an adapter");
            adapter.released.store(true, Ordering::Release);
            let removed = crate::interrupt::interruptible_caught_without_parts(
                py,
                &manager.spark_session(),
                || manager.remove_listener_found(&id).map_err(Box::new),
            );
            match removed {
                Ok(Ok(true)) => {
                    drop(registration);
                    let entries =
                        crate::state::attr(slf.as_any(), pyo3::intern!(py, "_listener_bus"))?;
                    return with_lock(&lock, || remove_equal(&entries, listener));
                }
                // The bus is read again once this is tried: a call that removed this
                // registration first leaves the next match to this one, which may have been
                // added since the bus was read.
                Ok(Ok(false)) => {
                    adapter.released.store(false, Ordering::Release);
                    tried.push(id);
                }
                Ok(Err(error)) => {
                    adapter.released.store(false, Ordering::Release);
                    return removal_failed(py, spark_error_to_py_exception(*error).value(py));
                }
                Err(error) => {
                    adapter.released.store(false, Ordering::Release);
                    return Err(error);
                }
            }
        }
    }

    #[staticmethod]
    fn deserialize<'py>(py: Python<'py>, event: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        // As PySpark's, of the `event_type` and the `event_json` of a StreamingQueryListenerEvent.
        let event_type = event.getattr(pyo3::intern!(py, "event_type"))?;
        let event_json = event.getattr(pyo3::intern!(py, "event_json"))?;
        py.import("pyspark.sql.connect.streaming.query")?
            .call_method1("_deserialize_listener_event", (event_type, event_json))
    }

    // Post an event to the listeners of the bus, as PySpark's: each of its list, in its order, of
    // the list as it is then, read under the lock, which the calls of the listeners do not hold.
    // The exception of a listener warns, and the others get the event.
    #[doc = connect_doc!(StreamingQueryListenerBus.post_to_all)]
    fn post_to_all(slf: &Bound<'_, Self>, event: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let lock = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_lock"))?;
        let listeners = with_lock(&lock, || {
            let entries = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_listener_bus"))?;
            entries.try_iter()?.collect::<PyResult<Vec<_>>>()
        })?;
        let module = py.import("pyspark.sql.connect.streaming.query")?;
        for listener in listeners {
            module.call_method1(
                pyo3::intern!(py, "_dispatch_listener_event"),
                (listener, event),
            )?;
        }
        Ok(())
    }
}

/// Python wrapper for ListenerEventStream.
/// Implements `__iter__` and `__next__` to yield (event_type: i32, event_json: String) tuples.
#[pyclass(
    name = "ListenerEventStream",
    module = "pyspark.sql.connect.streaming.query"
)]
pub struct PyListenerEventStream {
    inner: Option<ListenerEventStream>,
    /// The session of the stream, whose waits a signal of the process ends.
    session: spark_connect::session::SparkSession,
}

impl PyListenerEventStream {
    pub fn new(stream: ListenerEventStream, session: spark_connect::session::SparkSession) -> Self {
        PyListenerEventStream {
            inner: Some(stream),
            session,
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
        // (arbitrarily far apart on a live query), so the wait releases the GIL, or a daemon
        // thread would freeze the whole interpreter between events, and a signal of the process
        // ends it with its exception, as it ends the other waits of the client for the server.
        // The events of the stream that started it, of its request.
        let next =
            crate::interrupt::interruptible_continuation_without_parts(py, &self.session, || {
                Ok::<_, Box<SparkError>>(stream.next())
            })?;
        match next {
            Some(Ok((event_type, event_json))) => Ok(Some((event_type, event_json))),
            Some(Err(e)) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error reading listener events: {}",
                e
            ))),
            None => Ok(None),
        }
    }
}
