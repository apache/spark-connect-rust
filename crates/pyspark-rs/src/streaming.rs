//! PyO3 wrappers for Structured Streaming.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_connect::dataframe::DataFrame;
use spark_connect::session::SparkSession;
use spark_connect::streaming::{
    DataStreamReader, DataStreamWriter, StreamingQuery, StreamingQueryException,
    StreamingQueryManager, StreamingQueryStatus, Trigger,
};
use spark_connect::udf::PythonUDFPayload;
use std::collections::HashMap;

use crate::dataframe::{py_cloudpickle, py_version, PyDataFrame};
use crate::errors::ResultExt;

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

    fn option(&mut self, key: &str, value: &str) -> PyResult<PyDataStreamReader> {
        Ok(PyDataStreamReader {
            inner: Some(self.take()?.option(key, value)),
        })
    }

    fn options(&mut self, options: &Bound<'_, PyDict>) -> PyResult<PyDataStreamReader> {
        let mut opts = HashMap::new();
        for (k, v) in options.iter() {
            let key: String = k.extract()?;
            let val: String = v.extract()?;
            opts.insert(key, val);
        }
        Ok(PyDataStreamReader {
            inner: Some(self.take()?.options(opts)),
        })
    }

    fn load(&mut self, path: Option<&str>) -> PyResult<PyDataFrame> {
        let df = self.take()?.load(path);
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

    fn option(&mut self, key: &str, value: &str) -> PyResult<PyDataStreamWriter> {
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.option(key, value)),
        })
    }

    fn options(&mut self, options: &Bound<'_, PyDict>) -> PyResult<PyDataStreamWriter> {
        let mut opts = HashMap::new();
        for (k, v) in options.iter() {
            let key: String = k.extract()?;
            let val: String = v.extract()?;
            opts.insert(key, val);
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

    fn trigger(&mut self, trigger: &PyTrigger) -> PyResult<PyDataStreamWriter> {
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.trigger(trigger.get())),
        })
    }

    fn start(&mut self, path: &str) -> PyResult<PyStreamingQuery> {
        let writer = self.take()?;
        let query = writer.start(path).to_pyerr()?;
        Ok(PyStreamingQuery::new(query))
    }

    fn toTable(&mut self, table_name: &str) -> PyResult<PyStreamingQuery> {
        let writer = self.take()?;
        let query = writer.to_table(table_name).to_pyerr()?;
        Ok(PyStreamingQuery::new(query))
    }

    fn foreachBatch(&mut self, command: Vec<u8>, python_ver: &str) -> PyResult<PyDataStreamWriter> {
        let payload = PythonUDFPayload::new(
            spark_connect::types::DataType::Struct { fields: vec![] },
            0, // No specific eval_type for streaming
            command,
            python_ver.to_string(),
        );
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.foreach_batch(payload)),
        })
    }

    fn foreach(&mut self, command: Vec<u8>, python_ver: &str) -> PyResult<PyDataStreamWriter> {
        let payload = PythonUDFPayload::new(
            spark_connect::types::DataType::Struct { fields: vec![] },
            0, // No specific eval_type for streaming
            command,
            python_ver.to_string(),
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

    fn explain(&self, extended: Option<bool>) -> PyResult<String> {
        let ext = extended.unwrap_or(false);
        self.inner.explain(ext).to_pyerr()
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

    fn addListener(&self, listener_payload: Vec<u8>, python_ver: &str) -> PyResult<String> {
        let payload = PythonUDFPayload::new(
            spark_connect::types::DataType::Struct { fields: vec![] },
            0,
            listener_payload,
            python_ver.to_string(),
        );
        self.inner.add_listener(payload).to_pyerr()
    }

    fn removeListener(&self, listener_id: &str) -> PyResult<()> {
        self.inner.remove_listener(listener_id).to_pyerr()
    }

    fn streamListenerEvents(&self) -> PyResult<Vec<(i32, String)>> {
        self.inner.stream_listener_events().to_pyerr()
    }
}
