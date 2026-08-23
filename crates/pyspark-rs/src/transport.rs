//! Raw gRPC transport seam for the strangler-fig harness.
//!
//! This exposes a low-level stub that the vendored reference Spark Connect Python
//! client can use in place of its grpcio stub: it accepts a serialized protobuf
//! request (the pb2 message the reference client built, via `SerializeToString()`),
//! carries it over the wire with our Rust `spark-connect-core` client, and returns
//! serialized response bytes (which the Python side wraps back with `pb2.*.FromString`).
//!
//! Because the reference client builds byte-identical plan protos, this makes our
//! Rust core do the network + Arrow-bearing transport while the official test suite
//! exercises the standard pyspark public API. See docs/design/OFFICIAL_TESTS_HARNESS.md.

use prost::Message;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Notify;

use spark_connect_core::channel::ChannelBuilder;
use spark_connect_core::client::SparkConnectClient;
use spark_connect_core::runtime::block_on;
use spark_connect_proto as proto;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// Convert a SparkError to a Python error: a RustRpcError carrying the raw gRPC status
// details when present (so the Python side rebuilds the exact typed pyspark exception),
// otherwise a plain RuntimeError.
fn spark_err_to_py(py: Python<'_>, e: spark_connect_core::error::SparkError) -> PyErr {
    // Any error that carries a gRPC status code (even a transport error like a DNS
    // failure, which has no status details) should become a RustRpcError so the
    // reference client maps it to the right SparkConnectGrpcException with the code.
    match e.grpc_code() {
        Some(code) => {
            let details = e.grpc_details().unwrap_or(&[]);
            RustRpcError::new_err((code, PyBytes::new(py, details).unbind(), e.message()))
        }
        None => err(e),
    }
}

// Raised when a server RPC returns a gRPC error. Carries (grpc_status_code:int,
// status_details:bytes = serialized google.rpc.Status, message:str) so the Python
// side can rebuild the exact typed pyspark exception via convert_exception().
pyo3::create_exception!(_pyspark, RustRpcError, pyo3::exceptions::PyException);

// Error kinds surfaced from a streaming execute: either a transport-level SparkError
// (e.g. failed to open the stream) or a per-message gRPC status from the server.
enum ExecErr {
    Spark(spark_connect_core::error::SparkError),
    Rpc {
        code: i32,
        details: Vec<u8>,
        message: String,
    },
}

/// A lazy iterator over a server-streaming response (ExecutePlan / ReattachExecute).
///
/// Yields one serialized response at a time (pulling from the gRPC stream on demand),
/// so the reference client's reattachable iterator can drive it correctly - detecting an
/// early stream end (e.g. an interrupted operation) and reattaching, rather than us
/// eagerly collecting the whole stream and reporting partial results as success.
#[pyclass]
pub struct ResponseStream {
    stream: Option<tonic::Streaming<Vec<u8>>>,
    // Cancellation shared with the owning stub: when the client is closed (mirroring
    // grpcio's channel.close() cancelling in-flight RPCs), a reader blocked here is woken
    // and raises CANCELLED. Without this a long-lived stream (e.g. the streaming-query
    // listener bus, consumed on a non-daemon background thread) blocks forever and hangs
    // interpreter shutdown, since the reference client's reattachable iterator keeps
    // re-attaching until it sees an error.
    cancel: Arc<Notify>,
    closed: Arc<AtomicBool>,
}

// gRPC CANCELLED status code, surfaced when the client closes an in-flight stream.
const GRPC_CANCELLED: i32 = 1;

fn cancelled_err(py: Python<'_>) -> PyErr {
    RustRpcError::new_err((
        GRPC_CANCELLED,
        PyBytes::new(py, &[]).unbind(),
        "stream cancelled by client close".to_string(),
    ))
}

#[pymethods]
impl ResponseStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyBytes>>> {
        // If the client was already closed, stop immediately so a caller (e.g. the
        // reattachable iterator's re-attach loop) cannot block again on a dead session.
        if self.closed.load(Ordering::SeqCst) {
            self.stream = None;
            return Err(cancelled_err(py));
        }
        let stream = match &mut self.stream {
            Some(s) => s,
            None => return Ok(None), // Stream already closed
        };
        let cancel = self.cancel.clone();
        // Release the GIL while blocking on the next message: a long-lived stream must not
        // hold the GIL while it waits, or it deadlocks other Python threads (including the
        // main one). Race the read against cancellation so close() can unblock it.
        let outcome = py.detach(|| {
            block_on(async {
                tokio::select! {
                    biased;
                    _ = cancel.notified() => None,
                    m = stream.message() => Some(m),
                }
            })
        });
        match outcome {
            // Cancelled by client close: drop the stream and surface CANCELLED so the
            // reference reattachable iterator errors out instead of re-attaching forever.
            None => {
                self.stream = None;
                Err(cancelled_err(py))
            }
            Some(Ok(Some(bytes))) => Ok(Some(PyBytes::new(py, &bytes).unbind())),
            Some(Ok(None)) => {
                // Stream ended; drop it to release resources promptly.
                self.stream = None;
                Ok(None) // StopIteration
            }
            Some(Err(status)) => {
                // On error, also drop the stream.
                self.stream = None;
                Err(RustRpcError::new_err((
                    status.code() as i32,
                    PyBytes::new(py, status.details()).unbind(),
                    status.message().to_string(),
                )))
            }
        }
    }
}

/// A Rust-backed replacement for `pyspark.sql.connect.client.core`'s gRPC stub.
///
/// Each method mirrors one `SparkConnectServiceStub` RPC. Requests/responses cross
/// the boundary as protobuf bytes so the Python side can keep using its pb2 types.
#[pyclass(name = "RustConnectStub")]
pub struct RustConnectStub {
    client: SparkConnectClient,
    // Shared cancellation, cloned into every ResponseStream this stub hands out. close()
    // flips `closed` and wakes anyone blocked reading, mirroring grpcio channel.close().
    cancel: Arc<Notify>,
    closed: Arc<AtomicBool>,
}

#[pymethods]
impl RustConnectStub {
    #[new]
    fn new(url: &str) -> PyResult<Self> {
        let builder = ChannelBuilder::parse(url).map_err(err)?;
        let client = block_on(SparkConnectClient::connect(&builder)).map_err(err)?;
        Ok(Self {
            client,
            cancel: Arc::new(Notify::new()),
            closed: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Close the stub: mark it closed and wake any in-flight streaming reader so it
    /// raises CANCELLED and returns. Mirrors grpcio's channel.close(), which the
    /// reference client calls from SparkConnectClient.close(); without it a blocked
    /// streaming-query listener thread never terminates and hangs interpreter shutdown.
    fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.cancel.notify_waiters();
    }

    /// ExecutePlan: server-streaming. Returns a lazy ResponseStream so the reference
    /// client's reattachable iterator can drive it (and detect early ends / reattach).
    ///
    /// An initial-call gRPC error is surfaced as the typed pyspark exception; per-message
    /// errors surface as RustRpcError during iteration.
    fn execute_plan(&self, py: Python<'_>, request: &[u8]) -> PyResult<ResponseStream> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(cancelled_err(py));
        }
        let stream = block_on(self.client.execute_plan_raw(request.to_vec()))
            .map_err(|e| spark_err_to_py(py, e))?;
        Ok(ResponseStream {
            stream: Some(stream),
            cancel: self.cancel.clone(),
            closed: self.closed.clone(),
        })
    }

    /// ReattachExecute: server-streaming. Resume a running execution's response stream.
    fn reattach_execute(&self, py: Python<'_>, request: &[u8]) -> PyResult<ResponseStream> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(cancelled_err(py));
        }
        let stream = block_on(self.client.reattach_execute_raw(request.to_vec()))
            .map_err(|e| spark_err_to_py(py, e))?;
        Ok(ResponseStream {
            stream: Some(stream),
            cancel: self.cancel.clone(),
            closed: self.closed.clone(),
        })
    }

    /// ReleaseExecute: unary.
    fn release_execute<'py>(&self, py: Python<'py>, request: &[u8]) -> PyResult<Py<PyBytes>> {
        let req = proto::ReleaseExecuteRequest::decode(request).map_err(err)?;
        let resp =
            block_on(self.client.release_execute(req)).map_err(|e| spark_err_to_py(py, e))?;
        Ok(PyBytes::new(py, &resp.encode_to_vec()).unbind())
    }

    /// AnalyzePlan: unary. Forwards raw request bytes (no prost decode) so deeply
    /// nested plans don't hit prost's recursion limit; returns raw response bytes.
    fn analyze_plan<'py>(&self, py: Python<'py>, request: &[u8]) -> PyResult<Py<PyBytes>> {
        let resp = block_on(self.client.analyze_plan_raw(request.to_vec()))
            .map_err(|e| spark_err_to_py(py, e))?;
        Ok(PyBytes::new(py, &resp).unbind())
    }

    /// Config: unary.
    fn config<'py>(&self, py: Python<'py>, request: &[u8]) -> PyResult<Py<PyBytes>> {
        let req = proto::ConfigRequest::decode(request).map_err(err)?;
        let resp = block_on(self.client.config(req)).map_err(|e| spark_err_to_py(py, e))?;
        Ok(PyBytes::new(py, &resp.encode_to_vec()).unbind())
    }

    /// Interrupt: unary.
    fn interrupt<'py>(&self, py: Python<'py>, request: &[u8]) -> PyResult<Py<PyBytes>> {
        let req = proto::InterruptRequest::decode(request).map_err(err)?;
        let resp = block_on(self.client.interrupt(req)).map_err(|e| spark_err_to_py(py, e))?;
        Ok(PyBytes::new(py, &resp.encode_to_vec()).unbind())
    }

    /// FetchErrorDetails: unary. Enables the reference client to enrich exceptions with
    /// the full server-side stack trace / message (needed for message-regex assertions).
    fn fetch_error_details<'py>(&self, py: Python<'py>, request: &[u8]) -> PyResult<Py<PyBytes>> {
        let req = proto::FetchErrorDetailsRequest::decode(request).map_err(err)?;
        let resp =
            block_on(self.client.fetch_error_details(req)).map_err(|e| spark_err_to_py(py, e))?;
        Ok(PyBytes::new(py, &resp.encode_to_vec()).unbind())
    }
}
