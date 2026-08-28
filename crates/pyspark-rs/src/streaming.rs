//! PyO3 wrappers for Structured Streaming.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_connect::dataframe::DataFrame;
use spark_connect::session::SparkSession;
use spark_connect::streaming::{
    DataStreamReader, DataStreamWriter, ListenerEventStream, StreamingQuery,
    StreamingQueryException, StreamingQueryListener, StreamingQueryListenerEvent,
    StreamingQueryManager, StreamingQueryStatus, Trigger,
};
use spark_connect::udf::PythonUDFPayload;
use std::collections::HashMap;
use std::sync::Arc;

use crate::dataframe::{py_cloudpickle, py_version, PyDataFrame};
use crate::errors::ResultExt;
use spark_connect_core::runtime::block_on;

/// Adapts a Python `StreamingQueryListener` to the native Rust listener trait: on each
/// event it acquires the GIL and calls the Python-side dispatch helper (which builds the
/// typed event object and invokes the right `onQuery*` callback).
struct PyListenerAdapter {
    listener: Py<PyAny>,
}

impl StreamingQueryListener for PyListenerAdapter {
    fn on_event(&self, event: &StreamingQueryListenerEvent) {
        Python::attach(|py| {
            let dispatch = py
                .import("pyspark.sql.streaming.query")
                .and_then(|m| m.getattr("_dispatch_listener_event"));
            if let Ok(dispatch) = dispatch {
                // Swallow listener/callback errors so one bad listener cannot kill the
                // dispatch thread (reference pyspark also isolates callback exceptions).
                let _ = dispatch.call1((
                    self.listener.bind(py),
                    event.event_type,
                    event.event_json.as_str(),
                ));
            }
        });
    }
}

/// Python wrapper for DataStreamReader.
#[pyclass(name = "DataStreamReader")]
pub struct PyDataStreamReader {
    inner: Option<DataStreamReader>,
}

impl PyDataStreamReader {
    pub fn new(reader: DataStreamReader) -> Self {
        PyDataStreamReader {
            inner: Some(reader),
        }
    }

    fn take(&mut self) -> PyResult<DataStreamReader> {
        self.inner.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("DataStreamReader already consumed")
        })
    }
}

#[pymethods]
impl PyDataStreamReader {
    fn format(&mut self, source: &str) -> PyResult<PyDataStreamReader> {
        Ok(PyDataStreamReader {
            inner: Some(self.take()?.format(source)),
        })
    }

    fn schema(&mut self, schema: &str) -> PyResult<PyDataStreamReader> {
        Ok(PyDataStreamReader {
            inner: Some(self.take()?.schema(schema)),
        })
    }

    fn option(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<PyDataStreamReader> {
        // None -> option left unset; bools -> "true"/"false" (reference `to_str`).
        match crate::coerce_option_value(value)? {
            Some(v) => Ok(PyDataStreamReader {
                inner: Some(self.take()?.option(key, &v)),
            }),
            None => Ok(PyDataStreamReader {
                inner: Some(self.take()?),
            }),
        }
    }

    // Mirrors reference `DataStreamReader.options(**options)`: keyword args; None values
    // skipped, booleans lowercased.
    #[pyo3(signature = (**options))]
    fn options(&mut self, options: Option<&Bound<'_, PyDict>>) -> PyResult<PyDataStreamReader> {
        let mut opts = HashMap::new();
        if let Some(options) = options {
            for (k, v) in options.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    opts.insert(k.str()?.to_string(), val);
                }
            }
        }
        Ok(PyDataStreamReader {
            inner: Some(self.take()?.options(opts)),
        })
    }

    #[pyo3(signature = (path=None))]
    fn load(&mut self, path: Option<&str>) -> PyResult<PyDataFrame> {
        let df = self.take()?.load(path);
        Ok(PyDataFrame::new(df))
    }

    /// Set the source name (for checkpoint stability). Mirrors `DataStreamReader.name`,
    /// which validates the name is a non-empty `[A-Za-z0-9_]+` string.
    fn name(&mut self, source_name: &str) -> PyResult<PyDataStreamReader> {
        if source_name.is_empty()
            || !source_name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid streaming source name: {source_name:?}; only ASCII letters, digits, and underscores are allowed"
            )));
        }
        Ok(PyDataStreamReader {
            inner: Some(self.take()?.name(source_name)),
        })
    }

    /// Read the streaming CDC changes of a named table. Mirrors `DataStreamReader.changes`.
    fn changes(&mut self, tableName: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take()?.changes(tableName)))
    }

    /// Load an XML streaming source. Mirrors `DataStreamReader.xml(path, **options)` =
    /// set the options, then `format("xml").load(path)`.
    #[pyo3(signature = (path, **options))]
    #[allow(non_snake_case)]
    fn xml(&mut self, path: &str, options: Option<&Bound<'_, PyDict>>) -> PyResult<PyDataFrame> {
        let mut opts = HashMap::new();
        if let Some(options) = options {
            for (k, v) in options.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    opts.insert(k.str()?.to_string(), val);
                }
            }
        }
        let df = self.take()?.format("xml").options(opts).load(Some(path));
        Ok(PyDataFrame::new(df))
    }

    fn table(&mut self, table_name: &str) -> PyResult<PyDataFrame> {
        let df = self.take()?.table(table_name);
        Ok(PyDataFrame::new(df))
    }

    fn json(&mut self, path: &str) -> PyResult<PyDataFrame> {
        let df = self.take()?.json(path);
        Ok(PyDataFrame::new(df))
    }

    fn parquet(&mut self, path: &str) -> PyResult<PyDataFrame> {
        let df = self.take()?.parquet(path);
        Ok(PyDataFrame::new(df))
    }

    fn csv(&mut self, path: &str) -> PyResult<PyDataFrame> {
        let df = self.take()?.csv(path);
        Ok(PyDataFrame::new(df))
    }

    fn orc(&mut self, path: &str) -> PyResult<PyDataFrame> {
        let df = self.take()?.orc(path);
        Ok(PyDataFrame::new(df))
    }

    fn text(&mut self, path: &str) -> PyResult<PyDataFrame> {
        let df = self.take()?.text(path);
        Ok(PyDataFrame::new(df))
    }
}

/// Python wrapper for Trigger.
#[pyclass(name = "Trigger")]
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
#[pyclass(name = "DataStreamWriter")]
pub struct PyDataStreamWriter {
    inner: Option<DataStreamWriter>,
}

impl PyDataStreamWriter {
    pub fn new(writer: DataStreamWriter) -> Self {
        PyDataStreamWriter {
            inner: Some(writer),
        }
    }

    fn take(&mut self) -> PyResult<DataStreamWriter> {
        self.inner.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("DataStreamWriter already consumed")
        })
    }
}

#[pymethods]
impl PyDataStreamWriter {
    fn outputMode(&mut self, mode: &str) -> PyResult<PyDataStreamWriter> {
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.output_mode(mode)),
        })
    }

    fn format(&mut self, source: &str) -> PyResult<PyDataStreamWriter> {
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.format(source)),
        })
    }

    fn option(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<PyDataStreamWriter> {
        // None -> option left unset; bools -> "true"/"false" (reference `to_str`).
        match crate::coerce_option_value(value)? {
            Some(v) => Ok(PyDataStreamWriter {
                inner: Some(self.take()?.option(key, &v)),
            }),
            None => Ok(PyDataStreamWriter {
                inner: Some(self.take()?),
            }),
        }
    }

    // Mirrors reference `DataStreamWriter.options(**options)`: keyword args; None values
    // skipped, booleans lowercased.
    #[pyo3(signature = (**options))]
    fn options(&mut self, options: Option<&Bound<'_, PyDict>>) -> PyResult<PyDataStreamWriter> {
        let mut opts = HashMap::new();
        if let Some(options) = options {
            for (k, v) in options.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    opts.insert(k.str()?.to_string(), val);
                }
            }
        }
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.options(opts)),
        })
    }

    #[pyo3(signature = (*cols))]
    fn partitionBy(&mut self, cols: Vec<String>) -> PyResult<PyDataStreamWriter> {
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.partition_by(col_refs)),
        })
    }

    #[pyo3(signature = (*cols))]
    fn clusterBy(&mut self, cols: Vec<String>) -> PyResult<PyDataStreamWriter> {
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.cluster_by(col_refs)),
        })
    }

    fn queryName(&mut self, name: &str) -> PyResult<PyDataStreamWriter> {
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.query_name(name)),
        })
    }

    /// `DataStreamWriter.trigger(...)`: mirrors the reference keyword API — exactly one
    /// of `processingTime` / `once` / `availableNow` / `continuous` is given.
    #[pyo3(signature = (processingTime=None, once=None, availableNow=None, continuous=None))]
    fn trigger(
        &mut self,
        processingTime: Option<&str>,
        once: Option<bool>,
        availableNow: Option<bool>,
        continuous: Option<&str>,
    ) -> PyResult<PyDataStreamWriter> {
        let trigger = if let Some(interval) = processingTime {
            Trigger::ProcessingTime(interval.to_string())
        } else if once == Some(true) {
            Trigger::Once
        } else if availableNow == Some(true) {
            Trigger::AvailableNow
        } else if let Some(interval) = continuous {
            Trigger::Continuous(interval.to_string())
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "trigger() requires exactly one of processingTime, once, availableNow, continuous",
            ));
        };
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.trigger(trigger)),
        })
    }

    #[pyo3(signature = (path=None))]
    fn start(&mut self, path: Option<&str>) -> PyResult<PyStreamingQuery> {
        let writer = self.take()?;
        // Memory/console/foreach sinks take no path; the core treats "" as unset.
        let query = writer.start(path.unwrap_or("")).to_pyerr()?;
        Ok(PyStreamingQuery::new(query))
    }

    fn toTable(&mut self, table_name: &str) -> PyResult<PyStreamingQuery> {
        let writer = self.take()?;
        let query = writer.to_table(table_name).to_pyerr()?;
        Ok(PyStreamingQuery::new(query))
    }

    /// `DataStreamWriter.foreachBatch(func)`: cloudpickle the `(batch_df, batch_id)`
    /// function (via the bundled `pyspark.cloudpickle`) and attach it as the
    /// foreach-batch sink. Pickling happens here so the Python skin can re-export this
    /// class directly rather than subclassing a (non-subclassable) PyO3 type.
    fn foreachBatch(
        &mut self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataStreamWriter> {
        let command = crate::dataframe::py_cloudpickle(py, func)?;
        let payload = PythonUDFPayload::new(
            spark_connect::types::DataType::Struct { fields: vec![] },
            0, // eval type is unused for the streaming foreach sinks
            command,
            crate::dataframe::py_version(py),
        );
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.foreach_batch(payload)),
        })
    }

    /// `DataStreamWriter.foreach(f)`: wrap the row handler as the reference client does
    /// — `(f, None, serializer, serializer)` with `AutoBatchedSerializer(CPickleSerializer())`
    /// — and cloudpickle it so the worker deserializes it against its own
    /// `pyspark.serializers`. Built here to avoid a Python subclass of the PyO3 class.
    fn foreach(&mut self, py: Python<'_>, f: &Bound<'_, PyAny>) -> PyResult<PyDataStreamWriter> {
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
                f.clone(),
                py.None().into_bound(py),
                serializer.clone(),
                serializer,
            ],
        )?;
        let command = crate::dataframe::py_cloudpickle(py, command_tuple.as_any())?;
        let payload = PythonUDFPayload::new(
            spark_connect::types::DataType::Struct { fields: vec![] },
            0,
            command,
            crate::dataframe::py_version(py),
        );
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.foreach(payload)),
        })
    }
}

/// Python wrapper for StreamingQueryStatus.
#[pyclass(name = "StreamingQueryStatus")]
#[derive(Clone)]
pub struct PyStreamingQueryStatus {
    inner: StreamingQueryStatus,
}

impl PyStreamingQueryStatus {
    pub fn new(status: StreamingQueryStatus) -> Self {
        PyStreamingQueryStatus { inner: status }
    }
}

#[pymethods]
impl PyStreamingQueryStatus {
    #[getter]
    fn is_active(&self) -> bool {
        self.inner.is_active
    }

    #[getter]
    fn status_message(&self) -> String {
        self.inner.status_message.clone()
    }

    #[getter]
    fn is_data_available(&self) -> bool {
        self.inner.is_data_available
    }

    #[getter]
    fn is_trigger_active(&self) -> bool {
        self.inner.is_trigger_active
    }
}

/// Python wrapper for StreamingQueryException.
#[pyclass(name = "StreamingQueryException")]
#[derive(Clone)]
pub struct PyStreamingQueryException {
    inner: StreamingQueryException,
}

impl PyStreamingQueryException {
    pub fn new(exc: StreamingQueryException) -> Self {
        PyStreamingQueryException { inner: exc }
    }
}

#[pymethods]
impl PyStreamingQueryException {
    #[getter]
    fn message(&self) -> String {
        self.inner.message.clone()
    }

    #[getter]
    fn error_class(&self) -> String {
        self.inner.error_class.clone()
    }
}

/// Python wrapper for StreamingQuery.
#[pyclass(name = "StreamingQuery")]
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
    fn isActive(&self) -> PyResult<bool> {
        self.inner.is_active().to_pyerr()
    }

    #[getter]
    fn status(&self) -> PyResult<PyStreamingQueryStatus> {
        let status = self.inner.status().to_pyerr()?;
        Ok(PyStreamingQueryStatus::new(status))
    }

    fn stop(&self) -> PyResult<()> {
        self.inner.stop().to_pyerr()
    }

    fn awaitTermination(&self, timeout_sec: Option<f64>) -> PyResult<Option<bool>> {
        self.inner.await_termination(timeout_sec).to_pyerr()
    }

    #[getter]
    fn lastProgress(&self) -> PyResult<Option<String>> {
        self.inner.last_progress().to_pyerr()
    }

    #[getter]
    fn recentProgress(&self) -> PyResult<Vec<String>> {
        self.inner.recent_progress().to_pyerr()
    }

    fn processAllAvailable(&self) -> PyResult<()> {
        self.inner.process_all_available().to_pyerr()
    }

    // Mirrors reference `StreamingQuery.explain(extended=False)` - the arg is optional.
    #[pyo3(signature = (extended=false))]
    fn explain(&self, extended: bool) -> PyResult<String> {
        self.inner.explain(extended).to_pyerr()
    }

    fn exception(&self) -> PyResult<Option<PyStreamingQueryException>> {
        let exc = self.inner.exception().to_pyerr()?;
        Ok(exc.map(PyStreamingQueryException::new))
    }
}

/// Python wrapper for StreamingQueryManager.
#[pyclass(name = "StreamingQueryManager")]
pub struct PyStreamingQueryManager {
    inner: StreamingQueryManager,
}

impl PyStreamingQueryManager {
    pub fn new(manager: StreamingQueryManager) -> Self {
        PyStreamingQueryManager { inner: manager }
    }
}

#[pymethods]
impl PyStreamingQueryManager {
    #[getter]
    fn active(&self) -> PyResult<Vec<PyStreamingQuery>> {
        let queries = self.inner.active().to_pyerr()?;
        Ok(queries.into_iter().map(PyStreamingQuery::new).collect())
    }

    fn get(&self, id: &str) -> PyResult<Option<PyStreamingQuery>> {
        let query = self.inner.get(id).to_pyerr()?;
        Ok(query.map(PyStreamingQuery::new))
    }

    fn awaitAnyTermination(&self, timeout_sec: Option<f64>) -> PyResult<Option<bool>> {
        self.inner.await_any_termination(timeout_sec).to_pyerr()
    }

    fn resetTerminated(&self) -> PyResult<()> {
        self.inner.reset_terminated().to_pyerr()
    }

    /// Register a client-side listener object (with onQueryStarted/Progress/Idle/
    /// Terminated callbacks). Mirrors `StreamingQueryManager.addListener`. The native
    /// Rust bus streams events and dispatches to this listener; the assigned id is
    /// stashed on the listener so `removeListener` can find it.
    fn addListener(&self, py: Python<'_>, listener: Py<PyAny>) -> PyResult<()> {
        let adapter = Arc::new(PyListenerAdapter {
            listener: listener.clone_ref(py),
        });
        let id = self.inner.add_listener(adapter).to_pyerr()?;
        listener.bind(py).setattr("_rust_listener_id", id)?;
        Ok(())
    }

    /// Remove a previously-added client-side listener object. Mirrors
    /// `StreamingQueryManager.removeListener`.
    fn removeListener(&self, py: Python<'_>, listener: Py<PyAny>) -> PyResult<()> {
        let bound = listener.bind(py);
        if let Ok(id_obj) = bound.getattr("_rust_listener_id") {
            let id: String = id_obj.extract()?;
            self.inner.remove_listener(&id).to_pyerr()?;
            let _ = bound.delattr("_rust_listener_id");
        }
        Ok(())
    }

    /// Remove all client-side listeners and stop the dispatch thread. Mirrors
    /// `StreamingQueryManager.close`.
    fn close(&self) -> PyResult<()> {
        self.inner.close().to_pyerr()
    }

    fn streamListenerEvents(&self) -> PyResult<PyListenerEventStream> {
        let stream = self.inner.listener_event_stream().to_pyerr()?;
        Ok(PyListenerEventStream::new(stream))
    }
}

/// Python wrapper for ListenerEventStream.
/// Implements `__iter__` and `__next__` to yield (event_type: i32, event_json: String) tuples.
#[pyclass(name = "ListenerEventStream")]
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
    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
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
