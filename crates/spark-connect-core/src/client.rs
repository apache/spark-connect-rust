//! Spark Connect client implementation.
//!
//! Mirrors `pyspark.sql.connect.client.core.SparkConnectClient`.

use std::collections::BTreeMap;
use std::time::Duration;
use tonic::metadata::{AsciiMetadataKey, AsciiMetadataValue, MetadataMap};
use tonic::transport::{Channel, ClientTlsConfig};
use tonic::{Request, Response, Streaming};
use uuid::Uuid;

use prost::Message;
use spark_connect_proto::plan::compressed_operation::OpType as CompressedOpType;
use spark_connect_proto::plan::{CompressedOperation, OpType as PlanOpType};
use spark_connect_proto::spark_connect_service_client::SparkConnectServiceClient;
use spark_connect_proto::{
    analyze_plan_request::Analyze, AnalyzePlanRequest, AnalyzePlanResponse,
    ArtifactStatusesRequest, CloneSessionRequest, CompressionCodec, ConfigRequest, ConfigResponse,
    ExecutePlanRequest, ExecutePlanResponse, FetchErrorDetailsRequest, FetchErrorDetailsResponse,
    InterruptRequest, InterruptResponse, Plan, ReattachExecuteRequest, ReleaseExecuteRequest,
    ReleaseExecuteResponse, ReleaseSessionRequest, UserContext,
};

use crate::artifact::{
    cache_artifact_requests, check_artifact_upload, take_frame_failure, ArtifactFrames,
    FileArtifact, NamedArtifact,
};
use crate::channel::{is_secret_param, ChannelBuilder, GRPC_MAX_MESSAGE_LENGTH_DEFAULT};
use crate::error::{Result, SparkError, SparkErrorKind};
use crate::reattach::ExecutePlanResponseReattachableIterator;
use crate::retries::{retrying, unknown_operation, Backoff, RetryError, RetryPolicy, Retrying};
use crate::thread_entries::forget_when_thread_ends;

/// How long to wait for the lazy gRPC channel to become ready before surfacing a
/// retriable `UNAVAILABLE`. tonic's lazy channel waits indefinitely for an unreachable
/// or unresolvable host (unlike grpcio, which fails fast), so without this bound a call
/// to a down server would hang forever instead of raising `UNAVAILABLE`.
const READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Apply the keepalive of the connection string, as PySpark's channel options do (SPARK-58094):
/// HTTP2 PINGs detect a connection that a NAT gateway or load balancer dropped silently, so that
/// a blocked call fails with `UNAVAILABLE`, which the client retries or reattaches, instead of
/// waiting on the dead connection forever.
fn with_keepalive(
    endpoint: tonic::transport::Endpoint,
    builder: &ChannelBuilder,
) -> tonic::transport::Endpoint {
    match builder.keepalive() {
        Some(keepalive) => endpoint
            .http2_keep_alive_interval(keepalive.time)
            .keep_alive_timeout(keepalive.timeout)
            .keep_alive_while_idle(keepalive.without_calls),
        None => endpoint,
    }
}

/// gRPC/HTTP header carrying the connection-string `token` as `Bearer <token>`.
/// Named here (rather than as a string literal) so the key is written once.
const AUTHORIZATION_HEADER: &str = "authorization";

/// Wait for `grpc` to become ready, bounded by [`READY_TIMEOUT`], mapping any failure
/// (or the timeout) to a gRPC `UNAVAILABLE` error. This matches how grpcio reports an
/// unreachable server, so the reference client raises `SparkConnectGrpcException` with
/// `UNAVAILABLE` rather than hanging on a channel that never connects.
async fn wait_ready(grpc: &mut tonic::client::Grpc<Channel>) -> Result<()> {
    match tokio::time::timeout(READY_TIMEOUT, grpc.ready()).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(SparkError::from_grpc_status(tonic::Status::unavailable(
            format!("channel not ready: {e}"),
        ))),
        Err(_elapsed) => Err(SparkError::from_grpc_status(tonic::Status::unavailable(
            "channel not ready: connection timed out",
        ))),
    }
}

/// Avoid a blocking-pool round trip for small artifact batches.
/// Keep synchronous work small on async workers, including CPUs without SHA acceleration.
const ARTIFACT_HASH_INLINE_MAX_BYTES: usize = 256 * 1024;
/// Bound per-blob digest and encoding work even when every blob is empty.
const ARTIFACT_HASH_INLINE_MAX_BLOBS: usize = 128;

/// Hash batches inline up to [`ARTIFACT_HASH_INLINE_MAX_BLOBS`] blobs and
/// [`ARTIFACT_HASH_INLINE_MAX_BYTES`] total bytes; offload larger batches.
/// Returns the original buffers for upload framing. If the caller is dropped after
/// blocking hashing starts, the task retains the blob buffers until it finishes.
async fn hash_artifact_blobs(blobs: Vec<Vec<u8>>) -> Result<(Vec<Vec<u8>>, Vec<String>)> {
    use sha2::{Digest, Sha256};

    let hash_inline = blobs.len() <= ARTIFACT_HASH_INLINE_MAX_BLOBS
        && blobs
            .iter()
            .try_fold(ARTIFACT_HASH_INLINE_MAX_BYTES, |remaining, blob| {
                remaining.checked_sub(blob.len())
            })
            .is_some();
    let hash = move || {
        let hashes = blobs
            .iter()
            .map(|blob| format!("{:x}", Sha256::digest(blob)))
            .collect();
        (blobs, hashes)
    };
    if hash_inline {
        return Ok(hash());
    }
    tokio::task::spawn_blocking(hash)
        .await
        .map_err(artifact_hashing_task_error)
}

fn artifact_hashing_task_error(error: tokio::task::JoinError) -> SparkError {
    if error.is_panic() {
        // Preserve inline hashing panic behavior when hashing is offloaded.
        std::panic::resume_unwind(error.into_panic());
    }
    SparkError::connect_msg(format!("Artifact hashing task failed: {error}"))
}

/// Per-RPC deadlines, as PySpark's `RpcDeadlines` of `pyspark.sql.connect.client`: the
/// timeout of each call of a kind, or none. Those of a reattachable `ExecutePlan` and of a
/// `ReattachExecute` are of each stream rather than of the execution: when one passes, the
/// execution goes on on the server, and the client reattaches to resume its results. A
/// non-reattachable `ExecutePlan` has none, as a timeout would end the execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcDeadlines {
    pub reattachable_execute_plan: Option<Duration>,
    pub reattach_execute: Option<Duration>,
    pub analyze_plan: Option<Duration>,
    pub add_artifacts: Option<Duration>,
    pub config: Option<Duration>,
    pub interrupt: Option<Duration>,
    pub release_session: Option<Duration>,
    pub artifact_status: Option<Duration>,
    pub fetch_error_details: Option<Duration>,
    pub clone_session: Option<Duration>,
}

impl Default for RpcDeadlines {
    /// PySpark's: 10 minutes, and an hour for `AnalyzePlan` and `AddArtifacts`.
    fn default() -> Self {
        let minutes = |m: u64| Some(Duration::from_secs(m * 60));
        RpcDeadlines {
            reattachable_execute_plan: minutes(10),
            reattach_execute: minutes(10),
            analyze_plan: minutes(60),
            add_artifacts: minutes(60),
            config: minutes(10),
            interrupt: minutes(10),
            release_session: minutes(10),
            artifact_status: minutes(10),
            fetch_error_details: minutes(10),
            clone_session: minutes(10),
        }
    }
}

impl RpcDeadlines {
    /// No deadline for any call, as `RpcDeadlines.disabled()`.
    pub fn disabled() -> Self {
        RpcDeadlines {
            reattachable_execute_plan: None,
            reattach_execute: None,
            analyze_plan: None,
            add_artifacts: None,
            config: None,
            interrupt: None,
            release_session: None,
            artifact_status: None,
            fetch_error_details: None,
            clone_session: None,
        }
    }
}

/// The longest timeout gRPC encodes in its `grpc-timeout`, 99,999,999 hours. tonic panics on a
/// longer one, and no call reaches this one, so a longer deadline is sent as this one.
const LONGEST_GRPC_TIMEOUT: Duration = Duration::from_secs(99_999_999 * 60 * 60);

/// `request` with the timeout `deadline`, which gRPC sends as its `grpc-timeout`.
fn with_deadline<T>(mut request: Request<T>, deadline: Option<Duration>) -> Request<T> {
    if let Some(deadline) = deadline {
        request.set_timeout(deadline.min(LONGEST_GRPC_TIMEOUT));
    }
    request
}

/// `call` bounded as a whole by `deadline`, as gRPC bounds an RPC: past it, the call fails with
/// `DEADLINE_EXCEEDED`, as PySpark's does, however far it got.
///
/// The `grpc-timeout` of [`with_deadline`] is also enforced by tonic, but only while the
/// response headers are awaited, and an expiry there is a `CANCELLED` "Timeout expired". So
/// the call is bounded here too, and that status is read as the deadline it is. A call stalled
/// after its headers, as for a unary response, is then bounded as well.
async fn within_deadline<T>(
    deadline: Option<Duration>,
    call: impl std::future::Future<Output = std::result::Result<T, tonic::Status>>,
) -> std::result::Result<T, tonic::Status> {
    let Some(deadline) = deadline else {
        return call.await;
    };
    match tokio::time::timeout(deadline, call).await {
        Ok(Err(status))
            if status.code() == tonic::Code::Cancelled && status.message() == "Timeout expired" =>
        {
            Err(tonic::Status::deadline_exceeded("Deadline Exceeded"))
        }
        Ok(result) => result,
        Err(_) => Err(tonic::Status::deadline_exceeded("Deadline Exceeded")),
    }
}

/// The server-side session ID observed by a client, from its shared slot.
fn observed_of(slot: &std::sync::Mutex<Option<String>>) -> Option<String> {
    slot.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// The error of a response that does not match this session, as PySpark's
/// `PySparkAssertionError`, which has no error class.
fn assertion_error(message: String) -> SparkError {
    SparkError {
        kind: SparkErrorKind::AssertionError,
        ..SparkError::connect_msg(message)
    }
}

/// When a stream opened now passes `deadline`, the timeout of the call that opens it. A
/// deadline too far to be a time, as of a duration set to its maximum, is never reached, so
/// it is none, where adding it to the time would panic.
fn stream_deadline_in(deadline: Option<Duration>) -> Option<tokio::time::Instant> {
    deadline.and_then(|deadline| tokio::time::Instant::now().checked_add(deadline))
}

/// The error of a reattach to an operation that the server does not know, once a response was
/// read: `RESPONSE_ALREADY_RECEIVED`, with the error class of the server as `error_type`.
fn response_already_received(error_type: &str) -> SparkError {
    SparkError::classed(
        SparkErrorKind::RuntimeError,
        "RESPONSE_ALREADY_RECEIVED",
        &[("error_type", error_type)],
    )
}

const PLAN_COMPRESSION_THRESHOLD_CONF: &str = "spark.connect.session.planCompression.threshold";
const PLAN_COMPRESSION_ALGORITHM_CONF: &str =
    "spark.connect.session.planCompression.defaultAlgorithm";

/// How a client compresses its plans, as PySpark's client has it from the confs of the server.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum PlanCompression {
    /// Not looked up yet.
    #[default]
    Unknown,
    /// The plans are sent as they are.
    Disabled,
    /// The plans of more than `threshold` bytes are sent zstd-compressed.
    Zstd { threshold: usize },
}

/// The plan compression of the confs of the server, as PySpark reads them: zstd above a threshold
/// that is not negative. Any other algorithm, or a missing or invalid threshold, disables it.
fn plan_compression_of(threshold: Option<&str>, algorithm: Option<&str>) -> PlanCompression {
    let threshold = threshold
        .filter(|threshold| !threshold.is_empty())
        .and_then(|threshold| threshold.parse::<usize>().ok());
    match (threshold, algorithm) {
        (Some(threshold), Some("ZSTD")) => PlanCompression::Zstd { threshold },
        _ => PlanCompression::Disabled,
    }
}

/// The stack that a clone of a request takes for the levels of its plan: prost's derived `Clone`
/// recurses once per level, with about 2 KiB of stack per level in a release build (more in a
/// debug one), while encoding and dropping a plan take a fraction of that. A plan of PySpark's
/// serializer, at most 746 levels, must not overflow a thread of a small stack, such as a Python
/// thread of 512 KiB, that calls the client.
const REQUEST_STACK: usize = if cfg!(debug_assertions) {
    16 << 20
} else {
    4 << 20
};

/// Clone `value`, a request, on a stack that holds its levels (see [`REQUEST_STACK`]).
fn clone_on_grown_stack<T: Clone>(value: &T) -> T {
    stacker::maybe_grow(REQUEST_STACK, REQUEST_STACK, || value.clone())
}

/// Replace the relation or command of `plan` with its zstd-compressed operation, when it is over
/// `threshold` bytes and compression makes it smaller, as PySpark's `_try_compress_and_set_plan`.
fn compress_over_threshold(plan: &mut Plan, threshold: usize) {
    let (op_type, size) = match &plan.op_type {
        Some(PlanOpType::Root(relation)) => (CompressedOpType::Relation, relation.encoded_len()),
        Some(PlanOpType::Command(command)) => (CompressedOpType::Command, command.encoded_len()),
        _ => return,
    };
    // The size of the encoding, without encoding the plans that are not over the threshold.
    if size <= threshold {
        return;
    }
    let message = match &plan.op_type {
        Some(PlanOpType::Root(relation)) => relation.encode_to_vec(),
        Some(PlanOpType::Command(command)) => command.encode_to_vec(),
        _ => return,
    };
    let Ok(data) = zstd::encode_all(message.as_slice(), zstd::DEFAULT_COMPRESSION_LEVEL) else {
        return;
    };
    if data.len() >= message.len() {
        return;
    }
    plan.op_type = Some(PlanOpType::CompressedOperation(CompressedOperation {
        data: data.into(),
        op_type: op_type as i32,
        compression_codec: CompressionCodec::Zstd as i32,
    }));
}

/// The plans of an analysis, which PySpark builds with its plan compression. The plans of
/// the other analyses, as the relations of persist, are not compressed.
fn analyze_plans(analyze: &mut Analyze) -> Vec<&mut Option<Plan>> {
    match analyze {
        Analyze::Schema(request) => vec![&mut request.plan],
        Analyze::Explain(request) => vec![&mut request.plan],
        Analyze::TreeString(request) => vec![&mut request.plan],
        Analyze::IsLocal(request) => vec![&mut request.plan],
        Analyze::IsStreaming(request) => vec![&mut request.plan],
        Analyze::InputFiles(request) => vec![&mut request.plan],
        Analyze::SemanticHash(request) => vec![&mut request.plan],
        Analyze::SameSemantics(request) => vec![&mut request.target_plan, &mut request.other_plan],
        Analyze::SparkVersion(_)
        | Analyze::DdlParse(_)
        | Analyze::Persist(_)
        | Analyze::Unpersist(_)
        | Analyze::GetStorageLevel(_)
        | Analyze::JsonToDdl(_) => Vec::new(),
    }
}

/// A hook of the executions of a client, as PySpark's `SparkSession.Hook.on_execute_plan`: it is
/// called with each `ExecutePlanRequest` before the client sends it, and returns the request to
/// send, which keeps the operation ID of the original.
pub type ExecutePlanHook =
    std::sync::Arc<dyn Fn(ExecutePlanRequest) -> Result<ExecutePlanRequest> + Send + Sync>;

/// Let the server split the Arrow batches too large for one message into chunks, which the
/// stream of the execution joins, as PySpark's client does, without a preferred chunk size.
fn allow_arrow_batch_chunking(request: &mut ExecutePlanRequest) {
    use spark_connect_proto::execute_plan_request::request_option::RequestOption as Kind;
    use spark_connect_proto::execute_plan_request::RequestOption;
    request.request_options.push(RequestOption {
        request_option: Some(Kind::ResultChunkingOptions(
            spark_connect_proto::ResultChunkingOptions {
                allow_arrow_batch_chunking: true,
                preferred_arrow_chunk_size: None,
            },
        )),
    });
}

/// A Spark Connect client for communicating with a remote Spark server.
///
/// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient`.
#[derive(Clone)]
pub struct SparkConnectClient {
    /// The tonic gRPC channel.
    channel: Channel,
    /// The display host of the endpoint of the channel, as `ChannelBuilder::host`.
    host: String,
    /// The generated gRPC client stub.
    stub: SparkConnectServiceClient<Channel>,
    /// Session ID (UUID v4) identifying this session on the server.
    session_id: String,
    /// User context (user_id from the channel builder).
    user_id: Option<String>,
    /// The metadata of the connection string, and the `authorization` of its token.
    metadata: Vec<(String, String)>,
    /// The headers of every request: the user agent, then `metadata`. They are validated
    /// once, by `connect`, so that attaching them to a request cannot fail.
    headers: Vec<(AsciiMetadataKey, AsciiMetadataValue)>,
    /// User-agent header.
    user_agent: String,
    /// The user context extensions of the requests of the client.
    user_context_extensions: std::sync::Arc<std::sync::Mutex<UserContextExtensions>>,
    /// Retry policy applied to RPCs (exponential backoff on transient failures).
    retry_policy: RetryPolicy,
    /// The deadline of each kind of RPC.
    rpc_deadlines: RpcDeadlines,
    /// The hook of the executions, if any, shared by the clients of the clones of the session,
    /// as PySpark's clones keep the hooks of a session.
    execute_plan_hook: std::sync::Arc<std::sync::RwLock<Option<ExecutePlanHook>>>,
    /// Whether the client is closed, shared by its clones: its RPCs then raise
    /// `NO_ACTIVE_SESSION`, as PySpark's `SparkConnectClient._stub` does once it is closed.
    closed: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The server-side session ID that the first response carried, shared by the clones of the
    /// client, as PySpark's `_server_session_id`. Each request sends it as the observed ID, and
    /// each response is checked against it.
    server_session_id: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// Whether the client tracks its server-side session: the observed ID of its requests, the
    /// check of the IDs of its responses, and its close when the server reports that the session
    /// changed. See [`SparkConnectClient::set_session_tracking`].
    session_tracking: bool,
    /// The executions in flight, shared by the clones of the client, in the order they started,
    /// each with the thread that waits for it now, if any. See
    /// [`SparkConnectClient::running_operation_id`].
    inflight_operations: std::sync::Arc<std::sync::Mutex<Vec<InflightEntry>>>,
    /// The plan compression of the client, shared by its clones, looked up on its first plan.
    plan_compression: std::sync::Arc<std::sync::Mutex<PlanCompression>>,
}

/// An execution in flight: its operation ID, and the thread that waits for it now, if any.
#[derive(Debug, Clone, PartialEq)]
struct InflightEntry {
    operation_id: String,
    waiter: Option<std::thread::ThreadId>,
}

/// Keeps an execution in the registry of the in-flight executions of its client until it drops.
struct InflightGuard {
    registry: std::sync::Arc<std::sync::Mutex<Vec<InflightEntry>>>,
    operation_id: String,
    /// The thread that registered the execution, which is the caller that starts it.
    registered_on: std::thread::ThreadId,
}

impl InflightGuard {
    /// Register the execution `operation_id` in `registry` until the guard drops.
    fn register(
        registry: std::sync::Arc<std::sync::Mutex<Vec<InflightEntry>>>,
        operation_id: String,
    ) -> Self {
        registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(InflightEntry {
                operation_id: operation_id.clone(),
                waiter: None,
            });
        InflightGuard {
            registry,
            operation_id,
            registered_on: crate::runtime::calling_thread_id(),
        }
    }

    /// The handle of the execution, which the caller that started it uses to mark its waits.
    fn handle(&self) -> InflightHandle {
        InflightHandle {
            registry: std::sync::Arc::clone(&self.registry),
            operation_id: self.operation_id.clone(),
            registered_on: self.registered_on,
        }
    }

    /// Marks the execution as waited for by the calling thread, until the returned guard drops.
    fn waiting(&self) -> WaitingGuard {
        self.handle().waiting()
    }
}

/// A handle of an execution in flight, to mark the waits of its caller for it: see
/// [`InflightHandle::waiting`]. The handle does not keep the execution in flight.
#[derive(Clone)]
pub struct InflightHandle {
    registry: std::sync::Arc<std::sync::Mutex<Vec<InflightEntry>>>,
    operation_id: String,
    registered_on: std::thread::ThreadId,
}

impl InflightHandle {
    /// Marks the execution as waited for by the calling thread, until the returned guard drops,
    /// so that a signal that the wait polls interrupts it (see
    /// [`SparkConnectClient::running_operation_id`]). Only a wait of the thread that started the
    /// execution is marked: a wait of another thread, such as a background task that reads the
    /// execution for its caller, is not the caller's, and the caller marks its own waits instead.
    pub fn waiting(&self) -> WaitingGuard {
        let caller = crate::runtime::calling_thread_id();
        let marked = caller == self.registered_on;
        if marked {
            set_waiter(&self.registry, &self.operation_id, Some(caller));
        }
        WaitingGuard {
            registry: std::sync::Arc::clone(&self.registry),
            operation_id: self.operation_id.clone(),
            marked,
        }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        let mut inflight = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(position) = inflight
            .iter()
            .rposition(|entry| entry.operation_id == self.operation_id)
        {
            inflight.remove(position);
        }
    }
}

/// Marks an execution as waited for by the calling thread, until it drops. See
/// [`InflightHandle::waiting`].
pub struct WaitingGuard {
    registry: std::sync::Arc<std::sync::Mutex<Vec<InflightEntry>>>,
    operation_id: String,
    marked: bool,
}

impl Drop for WaitingGuard {
    fn drop(&mut self) {
        if self.marked {
            set_waiter(&self.registry, &self.operation_id, None);
        }
    }
}

/// Sets the thread that waits for the execution `operation_id` in `registry`, if it is in flight.
fn set_waiter(
    registry: &std::sync::Mutex<Vec<InflightEntry>>,
    operation_id: &str,
    waiter: Option<std::thread::ThreadId>,
) {
    let mut inflight = registry.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = inflight
        .iter_mut()
        .rev()
        .find(|entry| entry.operation_id == operation_id)
    {
        entry.waiter = waiter;
    }
}

/// The class of the error of a wait that a signal of the process ended (see
/// [`crate::runtime::until_abort`]).
const WAIT_ABORTED: &str = "WAIT_ABORTED";

/// The error of a wait, of an RPC or a retry, that a signal of the process ended: the signal's own
/// exception is what the call raises.
pub fn wait_aborted() -> SparkError {
    SparkError::classed(SparkErrorKind::Connect, WAIT_ABORTED, &[])
}

/// Whether `error` is the error of a wait that a signal of the process ended.
pub fn is_wait_aborted(error: &SparkError) -> bool {
    error.error_class == WAIT_ABORTED
}

/// The user context extensions of a client, as PySpark's `SparkConnectClient` keeps them: those
/// of all the threads, and those of each thread, which the requests of a thread carry, of their
/// IDs.
#[derive(Debug, Default)]
struct UserContextExtensions {
    global: Vec<(String, prost_types::Any)>,
    by_thread: std::collections::HashMap<std::thread::ThreadId, SharedUserContextExtensions>,
}

/// The user context extensions of a thread, shared with the threads that are given them by
/// [`SparkConnectClient::share_threadlocal_user_context_extensions`]. As the threads of PySpark's
/// `inheritable_thread_target` share the copy of the `thread_local` of the client of the thread
/// that decorated their function: an extension that one of them adds is in the list of all, while
/// one that removes or clears its extensions keeps its own list from then on.
#[derive(Clone, Debug, Default)]
pub struct SharedUserContextExtensions(
    std::sync::Arc<std::sync::Mutex<Vec<(String, prost_types::Any)>>>,
);

impl SharedUserContextExtensions {
    fn new(extensions: Vec<(String, prost_types::Any)>) -> Self {
        Self(std::sync::Arc::new(std::sync::Mutex::new(extensions)))
    }

    fn snapshot(&self) -> Vec<(String, prost_types::Any)> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// Forget the extensions of a thread that ends, as PySpark's `thread_local` is freed with its
/// thread.
fn forget_thread_extensions(extensions: &mut UserContextExtensions, thread: std::thread::ThreadId) {
    extensions.by_thread.remove(&thread);
}

impl SparkConnectClient {
    /// Create a new SparkConnectClient from a ChannelBuilder.
    ///
    /// This will dial the server and establish the gRPC channel.
    pub async fn connect(builder: &ChannelBuilder) -> Result<Self> {
        // Generate or reuse session_id.
        let session_id = match builder.session_id()? {
            Some(id) => id,
            None => Uuid::new_v4().to_string(),
        };
        // As PySpark's client, whose channel options read the keepalive parameters next.
        builder.check_keepalive()?;

        // Parse user-agent.
        let user_agent = builder.user_agent()?;

        // Build gRPC channel. We connect *lazily* - the channel is created here but the
        // TCP/TLS connection is established on the first RPC (matching the reference
        // client, where getOrCreate() does not dial the server). This also means the
        // client can be constructed without a live server (e.g. in unit tests that only
        // build request protos).
        let endpoint = builder.endpoint();
        let channel = if builder.secure() {
            // TLS connection with native root certificates
            // `with_native_roots()` is required: tonic >=0.11 does NOT load a trust store
            // from the `tls-native-roots` feature alone, so without this the TLS handshake
            // has no CA anchors, fails, and the retry policy masks it as a hang.
            // The server name of an IPv6 host is its address without the brackets of the URL.
            let domain = builder.host().trim_start_matches('[').trim_end_matches(']');
            let tls_config = ClientTlsConfig::new()
                .domain_name(domain)
                .with_native_roots();

            let tls_endpoint =
                tonic::transport::Channel::from_shared(format!("https://{endpoint}"))
                    .map_err(|e| SparkError::connect_msg(format!("Invalid endpoint: {}", e)))?
                    .tls_config(tls_config)
                    .map_err(|e| {
                        SparkError::connect_msg(format!("Failed to configure TLS: {}", e))
                    })?
                    .connect_timeout(READY_TIMEOUT);
            with_keepalive(tls_endpoint, builder).connect_lazy()
        } else {
            // Plaintext connection (for localhost development)
            let plain_endpoint =
                tonic::transport::Channel::from_shared(format!("http://{endpoint}"))
                    .map_err(|e| SparkError::connect_msg(format!("Invalid endpoint: {}", e)))?
                    .connect_timeout(READY_TIMEOUT);
            with_keepalive(plain_endpoint, builder).connect_lazy()
        };

        // Raise the message-size cap from tonic's 4 MiB default (too small for real Arrow
        // `collect()` results) to 128 MiB, applied to both the stub and the raw Grpc calls
        // below. The reference client sets this unconditionally via GRPC_DEFAULT_OPTIONS
        // (grpc.max_{send,receive}_message_length); it is not a connection-string option.
        let stub = SparkConnectServiceClient::new(channel.clone())
            .max_decoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT)
            .max_encoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT);

        // The headers are checked here, once: one that gRPC cannot send is an error of the
        // connection, rather than a header that every request silently goes without.
        let mut headers = vec![(
            AsciiMetadataKey::from_static("user-agent"),
            user_agent_header(&user_agent)?,
        )];
        let mut metadata: Vec<(String, String)> = builder.metadata();
        for (key, value) in &metadata {
            headers.push((metadata_key(key)?, metadata_value(key, value)?));
        }
        // Spark Connect servers that require token auth expect the connection-string
        // `token` as an `Authorization: Bearer <token>` header. It is a reserved key
        // excluded from `metadata()`, so attach it explicitly here; this then propagates
        // to every request via `_attach_metadata` and the reattach stream.
        if let Some(tok) = builder.token() {
            // The token is sent over TLS, or over plaintext only on `localhost`: the scheme
            // above is `builder.secure()`, which is TLS for a token on any other host, as
            // the reference client does.
            let bearer = format!("Bearer {}", tok);
            headers.push((
                AsciiMetadataKey::from_static(AUTHORIZATION_HEADER),
                token_header(&bearer)?,
            ));
            metadata.push((AUTHORIZATION_HEADER.to_string(), bearer));
        }

        Ok(Self {
            channel,
            host: builder.host().to_string(),
            stub,
            session_id,
            user_id: builder.user_id().map(String::from).or_else(default_user_id),
            metadata,
            headers,
            user_agent,
            user_context_extensions: Default::default(),
            retry_policy: RetryPolicy::default(),
            rpc_deadlines: RpcDeadlines::default(),
            execute_plan_hook: Default::default(),
            closed: Default::default(),
            server_session_id: Default::default(),
            session_tracking: true,
            inflight_operations: Default::default(),
            plan_compression: Default::default(),
        })
    }

    /// Get the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Create a new client that reuses this client's channel/stub/metadata but
    /// starts a brand-new server-side session (fresh session id). Mirrors the
    /// reference client's `newSession()`, which opens a new session over the same
    /// connection.
    pub fn with_new_session_id(&self) -> Self {
        SparkConnectClient {
            channel: self.channel.clone(),
            host: self.host.clone(),
            stub: self.stub.clone(),
            session_id: Uuid::new_v4().to_string(),
            user_id: self.user_id.clone(),
            metadata: self.metadata.clone(),
            headers: self.headers.clone(),
            user_agent: self.user_agent.clone(),
            // As PySpark's client of a new session, of no user context extensions.
            user_context_extensions: Default::default(),
            retry_policy: self.retry_policy.clone(),
            rpc_deadlines: self.rpc_deadlines.clone(),
            execute_plan_hook: std::sync::Arc::clone(&self.execute_plan_hook),
            closed: Default::default(),
            server_session_id: Default::default(),
            session_tracking: self.session_tracking,
            inflight_operations: Default::default(),
            plan_compression: Default::default(),
        }
    }

    /// Close this client, as PySpark's `SparkConnectClient.close` closes its channel: its RPCs
    /// raise `NO_ACTIVE_SESSION` from now on, and its clones with them. `release_session` is
    /// still sent, as `stop()` sends it before it closes the client.
    pub fn close(&self) {
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Whether the client is closed. See [`SparkConnectClient::close`].
    pub fn is_closed(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// The operation ID of the innermost execution that the calling thread is waiting for, if any:
    /// the one that a wait of the calling thread is for, as PySpark interrupts the operation of
    /// the iterator that is interrupted. An execution in flight that nothing waits for, such as
    /// an iterator that is not consumed, is not one.
    pub fn running_operation_id(&self) -> Option<String> {
        let caller = crate::runtime::calling_thread_id();
        let inflight = self
            .inflight_operations
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        inflight
            .iter()
            .rev()
            .find(|entry| entry.waiter == Some(caller))
            .map(|entry| entry.operation_id.clone())
    }

    /// The error of an RPC of a closed client, as PySpark's `SparkConnectClient._stub` raises it.
    #[allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API
    fn ensure_open(&self) -> Result<()> {
        if self.is_closed() {
            return Err(SparkError::classed(
                crate::error::SparkErrorKind::Connect,
                "NO_ACTIVE_SESSION",
                &[],
            ));
        }
        Ok(())
    }

    /// The server-side session ID that the responses have carried, if one has, as PySpark's
    /// `_server_session_id`: None before the first response.
    pub fn observed_server_session_id(&self) -> Option<String> {
        observed_of(&self.server_session_id)
    }

    /// Set the server-side session ID that the requests observe, as PySpark's
    /// `_server_session_id` is settable: a response that carries another one closes the client.
    pub fn set_observed_server_session_id(&self, id: Option<String>) {
        *self
            .server_session_id
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = id;
    }

    /// The display host of the endpoint of the channel, as PySpark's `SparkConnectClient.host`.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Sets the observed server-side session ID of a request, once a response has set it, as
    /// PySpark's requests do.
    fn observe_on(&self, observed: &mut Option<String>) {
        if !self.session_tracking {
            return;
        }
        if let Some(id) = self.observed_server_session_id() {
            *observed = Some(id);
        }
    }

    /// Checks the IDs of a response, as PySpark's `_verify_response_integrity`: the session ID
    /// must be this session's. The first non-empty server-side session ID that a response
    /// carries is observed; a later one that differs closes the client, as the server restarted,
    /// and is an error. An empty ID is neither observed nor checked. PySpark keeps an empty
    /// first ID and then raises for a later non-empty one.
    #[allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API
    fn verify_response(&self, session_id: &str, server_side_session_id: &str) -> Result<()> {
        if !self.session_tracking {
            return Ok(());
        }
        // An empty session ID is none: the server leaves both IDs unset in some responses, such
        // as those of FetchErrorDetails, so only another session's ID is a mismatch.
        if !session_id.is_empty() && session_id != self.session_id {
            return Err(assertion_error(format!(
                "Received incorrect session identifier for request:{session_id} != {}",
                self.session_id
            )));
        }
        let mut observed = self
            .server_session_id
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match observed.clone() {
            None if !server_side_session_id.is_empty() => {
                *observed = Some(server_side_session_id.to_string());
            }
            None => {}
            Some(old) if !server_side_session_id.is_empty() && server_side_session_id != old => {
                drop(observed);
                self.close();
                return Err(assertion_error(format!(
                    "Received incorrect server side session identifier for request. Please \
                     create a new Spark Session to reconnect. ({server_side_session_id} != {old})"
                )));
            }
            Some(_) => {}
        }
        Ok(())
    }

    /// A gRPC failure as a [`SparkError`], which closes the client when the server reports that
    /// the session changed, and disables its plan compression when the server cannot parse a
    /// plan, as PySpark's `_handle_rpc_error` does.
    fn grpc_error(&self, status: tonic::Status) -> SparkError {
        let error = SparkError::from_grpc_status(status);
        let condition = error.get_condition();
        if self.session_tracking && condition.as_deref() == Some("INVALID_HANDLE.SESSION_CHANGED") {
            self.close();
        }
        if condition.as_deref() == Some("CONNECT_INVALID_PLAN.CANNOT_PARSE") {
            *self
                .plan_compression
                .lock()
                .unwrap_or_else(|e| e.into_inner()) = PlanCompression::Disabled;
        }
        error
    }

    /// Set the hook of the executions of this client and of its clones, or clear it.
    pub fn set_execute_plan_hook(&self, hook: Option<ExecutePlanHook>) {
        *self
            .execute_plan_hook
            .write()
            .unwrap_or_else(|e| e.into_inner()) = hook;
    }

    /// `request` as the hook of the executions, if any, returns it, of the operation ID of
    /// `request`, as PySpark's client keeps the operation ID of the request of a hook.
    #[allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API
    fn hooked(&self, request: ExecutePlanRequest) -> Result<ExecutePlanRequest> {
        let hook = self
            .execute_plan_hook
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let Some(hook) = hook else {
            return Ok(request);
        };
        crate::runtime::check_callback_depth()?;
        let operation_id = request.operation_id.clone();
        // The hook may run a query of its own, which runs the hooks too.
        let _callback = crate::runtime::CallbackGuard::enter();
        let mut request = hook(request)?;
        request.operation_id = operation_id;
        Ok(request)
    }

    /// `request` as the hook of the executions returns it, with its plan compressed again when a
    /// hook ran: a hook may put in a plan, such as the drop-in's hook, which resolves the plan of
    /// a DataFrame there, as PySpark's hook would put in the plan of `to_proto`, which compresses
    /// it. A plan that is compressed already is kept.
    async fn hooked_and_compressed(
        &self,
        request: ExecutePlanRequest,
    ) -> Result<ExecutePlanRequest> {
        let hook_set = self
            .execute_plan_hook
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_some();
        let mut request = self.hooked(request)?;
        if hook_set {
            if let Some(plan) = request.plan.as_mut() {
                self.compress_plan(plan).await;
            }
        }
        Ok(request)
    }

    /// Get the user ID, if set.
    pub fn user_id(&self) -> Option<&str> {
        self.user_id.as_deref()
    }

    /// Add `extension` to the user context of the requests that the current thread sends,
    /// and return its ID, as PySpark's `add_threadlocal_user_context_extension`.
    pub fn add_threadlocal_user_context_extension(&self, extension: prost_types::Any) -> String {
        let id = format!("threadlocal_{}", Uuid::new_v4());
        let mut extensions = self
            .user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let thread = std::thread::current().id();
        if !extensions.by_thread.contains_key(&thread) {
            forget_when_thread_ends(&self.user_context_extensions, forget_thread_extensions);
        }
        let own = extensions.by_thread.entry(thread).or_default().clone();
        own.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((id.clone(), extension));
        id
    }

    /// A copy of the user context extensions of the current thread, as of now, which the threads
    /// that are given it with [`SparkConnectClient::share_threadlocal_user_context_extensions`]
    /// share. Mirrors how PySpark's `inheritable_thread_target(session)` deep-copies the
    /// `thread_local` of the client of the thread that decorates a function.
    pub fn copy_threadlocal_user_context_extensions(&self) -> SharedUserContextExtensions {
        SharedUserContextExtensions::new(self.threadlocal_user_context_extensions())
    }

    /// Make `extensions` the user context extensions of the current thread, which it shares with
    /// the other threads given them, as PySpark's `inheritable_thread_target` sets the copy of the
    /// `thread_local` of the client in each thread that runs the function.
    pub fn share_threadlocal_user_context_extensions(
        &self,
        extensions: &SharedUserContextExtensions,
    ) {
        let mut all = self
            .user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let thread = std::thread::current().id();
        if !all.by_thread.contains_key(&thread) {
            forget_when_thread_ends(&self.user_context_extensions, forget_thread_extensions);
        }
        all.by_thread.insert(thread, extensions.clone());
    }

    /// Add `extension` to the user context of the requests of all the threads, and return its
    /// ID, as PySpark's `add_global_user_context_extension`.
    pub fn add_global_user_context_extension(&self, extension: prost_types::Any) -> String {
        let id = format!("global_{}", Uuid::new_v4());
        self.user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .global
            .push((id.clone(), extension));
        id
    }

    /// Remove the extension of `extension_id`, one of the current thread or a global one, as
    /// PySpark's `remove_user_context_extension`.
    pub fn remove_user_context_extension(&self, extension_id: &str) {
        let mut extensions = self
            .user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if extension_id.starts_with("threadlocal_") {
            if let Some(own) = extensions.by_thread.get_mut(&std::thread::current().id()) {
                // As PySpark rebinds the list of the thread, which other threads that share it
                // keep as it was.
                let kept = own
                    .snapshot()
                    .into_iter()
                    .filter(|(id, _)| id != extension_id)
                    .collect();
                *own = SharedUserContextExtensions::new(kept);
            }
        } else if extension_id.starts_with("global_") {
            extensions.global.retain(|(id, _)| id != extension_id);
        }
    }

    /// Remove the extensions of the current thread and the global ones, as PySpark's
    /// `clear_user_context_extensions`.
    pub fn clear_user_context_extensions(&self) {
        let mut extensions = self
            .user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        extensions.by_thread.remove(&std::thread::current().id());
        extensions.global.clear();
    }

    /// The user context extensions of the current thread, of their IDs, as PySpark's
    /// `thread_local.user_context_extensions` of a client.
    pub fn threadlocal_user_context_extensions(&self) -> Vec<(String, prost_types::Any)> {
        self.user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .by_thread
            .get(&std::thread::current().id())
            .map(SharedUserContextExtensions::snapshot)
            .unwrap_or_default()
    }

    /// The global user context extensions, of their IDs, as PySpark's
    /// `global_user_context_extensions` of a client.
    pub fn global_user_context_extensions(&self) -> Vec<(String, prost_types::Any)> {
        self.user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .global
            .clone()
    }

    /// Add the global user context extensions, and then those of the current thread, to
    /// `context`, as PySpark's `_update_request_with_user_context_extensions` does for the
    /// requests of executions, analyses, configurations, error details and interrupts.
    fn extend_user_context(&self, context: &mut Option<UserContext>) {
        let extensions = self
            .user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // The thread that calls the API, also when this future is polled on another thread.
        let own = extensions
            .by_thread
            .get(&crate::runtime::calling_thread_id())
            .map(|own| own.0.lock().unwrap_or_else(|e| e.into_inner()));
        if extensions.global.is_empty() && own.as_ref().is_none_or(|own| own.is_empty()) {
            return;
        }
        let context = context.get_or_insert_with(UserContext::default);
        context.extensions.extend(
            extensions
                .global
                .iter()
                .chain(own.iter().flat_map(|own| own.iter()))
                .map(|(_, extension)| extension.clone()),
        );
    }

    /// Override the retry policy applied to this client's RPCs.
    ///
    /// The transport-injection stub sets this to [`RetryPolicy::no_retries`] so it behaves
    /// like the single-shot grpcio stub the reference client expects (which does its own
    /// retrying on top); see that constructor for why double-retrying is wrong there.
    pub fn set_retry_policy(&mut self, policy: RetryPolicy) {
        self.retry_policy = policy;
    }

    /// Turn the tracking of the server-side session on (the default) or off. With it on, the
    /// requests carry the server-side session ID observed in the responses, each response must
    /// be of this session and of that server-side session, and an
    /// `INVALID_HANDLE.SESSION_CHANGED` error closes the client, as PySpark's
    /// `SparkConnectClient` does.
    ///
    /// The transport-injection stub turns it off. The requests that it sends are of the
    /// reference client's session, not of its own, and that client tracks its session itself,
    /// over the grpcio stub that does no tracking.
    pub fn set_session_tracking(&mut self, enabled: bool) {
        self.session_tracking = enabled;
    }

    /// Override the per-RPC deadlines of this client's calls, as the `rpc_deadlines` of
    /// PySpark's `SparkConnectClient`.
    pub fn set_rpc_deadlines(&mut self, deadlines: RpcDeadlines) {
        self.rpc_deadlines = deadlines;
    }

    /// The per-RPC deadlines of this client's calls.
    pub fn rpc_deadlines(&self) -> &RpcDeadlines {
        &self.rpc_deadlines
    }

    /// The plan compression of this client. The server's confs are read on the first plan, with
    /// one `Config` call, and kept for the client; a failed read disables the compression for it,
    /// as PySpark's `_get_plan_compression_threshold_and_algorithm` does.
    async fn plan_compression(&self) -> PlanCompression {
        let known = *self
            .plan_compression
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if known != PlanCompression::Unknown {
            return known;
        }
        let looked_up = match self
            .get_configs(&[
                PLAN_COMPRESSION_THRESHOLD_CONF,
                PLAN_COMPRESSION_ALGORITHM_CONF,
            ])
            .await
        {
            Ok(values) => match values.as_slice() {
                [threshold, algorithm] => {
                    plan_compression_of(threshold.as_deref(), algorithm.as_deref())
                }
                _ => PlanCompression::Disabled,
            },
            Err(error) => {
                // An aborted lookup is not the answer of the server: nothing is kept, so the
                // lookup is made again by the next plan.
                if is_wait_aborted(&error) {
                    return PlanCompression::Disabled;
                }
                PlanCompression::Disabled
            }
        };
        *self
            .plan_compression
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = looked_up;
        looked_up
    }

    /// Compress `plan` as the plan compression of this client asks.
    async fn compress_plan(&self, plan: &mut Plan) {
        if let PlanCompression::Zstd { threshold } = self.plan_compression().await {
            compress_over_threshold(plan, threshold);
        }
    }

    /// Execute a plan and return a stream of responses.
    ///
    /// The request allows the server to chunk an Arrow batch too large for one message, so a
    /// batch may arrive as several responses, whose `arrow_batch` fields are not decodable on
    /// their own. The chunks are returned as they arrive: see [`ReattachableResponseStream`].
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.execute_plan`.
    pub async fn execute_plan(
        &self,
        mut request: ExecutePlanRequest,
    ) -> Result<Streaming<ExecutePlanResponse>> {
        self.ensure_open()?;
        allow_arrow_batch_chunking(&mut request);
        self.extend_user_context(&mut request.user_context);
        self.fill_user_context(&mut request.user_context);
        self.fill_client_type(&mut request.client_type);
        if let Some(plan) = request.plan.as_mut() {
            self.compress_plan(plan).await;
        }
        // After the lookup of the plan compression of the first plan, whose response gives the
        // server-side session ID, as PySpark builds the plan before the request.
        self.observe_on(&mut request.client_observed_server_side_session_id);
        let request = self.hooked_and_compressed(request).await?;
        // As PySpark reads its `_stub` once the plan is built and the hooks ran: the lookup of
        // the plan compression of the first plan closes the client when it sees that the server
        // restarted, and then the execution raises NO_ACTIVE_SESSION and sends nothing.
        self.ensure_open()?;
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        match self.stub.clone().execute_plan(req).await {
            Ok(response) => Ok(response.into_inner()),
            Err(status) => Err(self.error_from_status(status).await),
        }
    }

    /// Execute a plan, forwarding raw request bytes and returning a raw response stream.
    ///
    /// Byte-exact passthrough (no prost decode/re-encode), so no proto field is dropped
    /// and deep plans don't hit the recursion limit - the server sees exactly the bytes
    /// the reference client built.
    pub async fn execute_plan_raw(&self, request: Vec<u8>) -> Result<Streaming<Vec<u8>>> {
        self.ensure_open()?;
        let mut grpc = tonic::client::Grpc::new(self.channel.clone())
            .max_decoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT)
            .max_encoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT);
        wait_ready(&mut grpc).await?;
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        let path = tonic::codegen::http::uri::PathAndQuery::from_static(
            "/spark.connect.SparkConnectService/ExecutePlan",
        );
        let resp = grpc
            .server_streaming(req, path, crate::bytes_codec::BytesCodec)
            .await
            .map_err(SparkError::from_grpc_status)?;
        Ok(resp.into_inner())
    }

    /// ReattachExecute, byte-exact passthrough returning a raw response stream.
    pub async fn reattach_execute_raw(&self, request: Vec<u8>) -> Result<Streaming<Vec<u8>>> {
        self.ensure_open()?;
        let mut grpc = tonic::client::Grpc::new(self.channel.clone())
            .max_decoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT)
            .max_encoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT);
        wait_ready(&mut grpc).await?;
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        let path = tonic::codegen::http::uri::PathAndQuery::from_static(
            "/spark.connect.SparkConnectService/ReattachExecute",
        );
        let resp = grpc
            .server_streaming(req, path, crate::bytes_codec::BytesCodec)
            .await
            .map_err(SparkError::from_grpc_status)?;
        Ok(resp.into_inner())
    }

    /// Reattach to a running execution and resume its response stream.
    ///
    /// Mirrors the `ReattachExecute` RPC used by the reattachable execute path.
    pub async fn reattach_execute(
        &self,
        mut request: ReattachExecuteRequest,
    ) -> Result<Streaming<ExecutePlanResponse>> {
        self.ensure_open()?;
        self.fill_user_context(&mut request.user_context);
        self.fill_client_type(&mut request.client_type);
        self.observe_on(&mut request.client_observed_server_side_session_id);
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        match self.stub.clone().reattach_execute(req).await {
            Ok(response) => Ok(response.into_inner()),
            Err(status) => Err(self.error_from_status(status).await),
        }
    }

    /// Release a (portion of a) running execution's response stream.
    ///
    /// Mirrors the `ReleaseExecute` RPC used by the reattachable execute path, whose requests
    /// carry no observed server-side session ID in PySpark.
    pub async fn release_execute(
        &self,
        mut request: ReleaseExecuteRequest,
    ) -> Result<ReleaseExecuteResponse> {
        self.ensure_open()?;
        self.fill_user_context(&mut request.user_context);
        self.fill_client_type(&mut request.client_type);
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        match self.stub.clone().release_execute(req).await {
            Ok(response) => {
                let response = response.into_inner();
                self.verify_response(&response.session_id, &response.server_side_session_id)?;
                Ok(response)
            }
            Err(status) => Err(self.error_from_status(status).await),
        }
    }

    /// Analyze a plan, forwarding the raw request bytes without decoding them.
    ///
    /// Decoding the request with prost imposes a recursion limit (100) that a deeply
    /// nested plan (e.g. hundreds of chained withColumn) exceeds - but the server
    /// handles such plans fine. Using a passthrough codec forwards the exact bytes and
    /// returns the raw response bytes, avoiding the client-side limit (matching the
    /// reference client, which never re-decodes the request it built).
    pub async fn analyze_plan_raw(&self, request: Vec<u8>) -> Result<Vec<u8>> {
        self.ensure_open()?;
        let mut grpc = tonic::client::Grpc::new(self.channel.clone())
            .max_decoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT)
            .max_encoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT);
        wait_ready(&mut grpc).await?;
        let deadline = self.rpc_deadlines.analyze_plan;
        let mut req = with_deadline(Request::new(request), deadline);
        self._attach_metadata(&mut req);
        let path = tonic::codegen::http::uri::PathAndQuery::from_static(
            "/spark.connect.SparkConnectService/AnalyzePlan",
        );
        let resp = within_deadline(
            deadline,
            grpc.unary(req, path, crate::bytes_codec::BytesCodec),
        )
        .await
        .map_err(SparkError::from_grpc_status)?;
        Ok(resp.into_inner())
    }

    /// Analyze a plan and return metadata about it.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.analyze_plan`.
    pub async fn analyze_plan(
        &self,
        mut request: AnalyzePlanRequest,
    ) -> Result<AnalyzePlanResponse> {
        self.ensure_open()?;
        self.extend_user_context(&mut request.user_context);
        self.fill_user_context(&mut request.user_context);
        self.fill_client_type(&mut request.client_type);
        if let Some(analyze) = request.analyze.as_mut() {
            for plan in analyze_plans(analyze).into_iter().flatten() {
                self.compress_plan(plan).await;
            }
        }
        // As in `execute_plan`: the lookup of the plan compression may have closed the client.
        self.ensure_open()?;
        let response = self
            .with_retry(|| {
                let deadline = self.rpc_deadlines.analyze_plan;
                let mut request = clone_on_grown_stack(&request);
                self.observe_on(&mut request.client_observed_server_side_session_id);
                let mut req = with_deadline(Request::new(request), deadline);
                self._attach_metadata(&mut req);
                let mut stub = self.stub.clone();
                async move {
                    within_deadline(deadline, stub.analyze_plan(req))
                        .await
                        .map(Response::into_inner)
                }
            })
            .await?;
        self.verify_response(&response.session_id, &response.server_side_session_id)?;
        Ok(response)
    }

    /// Run a unary RPC closure with exponential-backoff retry on transient failures.
    ///
    /// Mirrors `pyspark.sql.connect.client.retries.Retrying` / `RetryPolicy`: retry
    /// `UNAVAILABLE` (and cursor-disconnect) per the policy, sleeping the computed
    /// backoff between attempts, then surface the last error.
    async fn with_retry<T, Fut, F>(&self, mut op: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = std::result::Result<T, tonic::Status>>,
    {
        use futures::FutureExt;
        self.with_attempts(|| {
            // An attempt that is not answered ends with the signal that a wait of the caller
            // takes, as PySpark's blocked call does. A combinator rather than an async block that
            // awaits the call: such a block keeps the future of the call twice, as what it
            // captures and as what it awaits, and that future, of the state of a gRPC call, is
            // large enough that the extra copy overflows the stack of a test thread or of a
            // Python thread of 512 KiB.
            crate::runtime::until_abort(crate::runtime::ABORT_RETRIES, op()).map(|attempt| {
                match attempt {
                    Some(result) => result.map_err(RetryError::Status),
                    None => Err(RetryError::Aborted(wait_aborted())),
                }
            })
        })
        .await
    }

    /// Makes the attempts of a call of this client under its retry policy (see [`retrying`]),
    /// and gives the failed RPC that ends them as the error of the server. An attempt may fail
    /// with an error that is not a failed RPC, which is not retried. The backoff before a retry
    /// ends with a signal that the wait of the caller takes, as the sleep of PySpark's
    /// `Retrying` does.
    async fn with_attempts<T, Fut>(&self, attempt: impl FnMut() -> Fut) -> Result<T>
    where
        Fut: std::future::Future<Output = std::result::Result<T, RetryError<SparkError>>>,
    {
        let retries = Retrying::new(&self.retry_policy).ending_with_abort();
        // As the reference's `_stub`, through which each attempt goes: once the client is closed
        // (its session stopped while the call waited), the retry raises NO_ACTIVE_SESSION and
        // sends nothing.
        let still_open = || self.ensure_open();
        match retrying(retries, still_open, attempt).await {
            Ok(value) => Ok(value),
            Err(RetryError::Status(status)) => Err(self.error_from_status(status).await),
            Err(RetryError::Aborted(error)) => Err(error),
            Err(RetryError::WaitAborted) => Err(wait_aborted()),
        }
    }

    /// Execute a plan as a reattachable stream: retries the initial call and, if the
    /// response stream drops mid-flight (e.g. `UNAVAILABLE` / `INVALID_CURSOR.DISCONNECTED`)
    /// or the server ends it before `ResultComplete`, transparently resumes it with
    /// `ReattachExecute` from the last observed `response_id`. The stream releases the
    /// execution on the server with `ReleaseExecute` (see [`ReattachableResponseStream`]).
    ///
    /// The responses are the raw responses of the execution, chunks included, as the server
    /// sent them: see [`ReattachableResponseStream`] for how a chunked Arrow batch is read.
    /// Mirrors the reference reattachable execute path.
    pub async fn execute_plan_reattachable(
        &self,
        mut request: ExecutePlanRequest,
    ) -> Result<ReattachableResponseStream> {
        self.ensure_open()?;
        allow_arrow_batch_chunking(&mut request);
        self.extend_user_context(&mut request.user_context);
        // Before the iterator: its ReattachExecute and ReleaseExecute requests take the
        // user context and the client type of this request, and the request of the hook, as
        // PySpark's iterator is of the request that the hooks return.
        self.fill_user_context(&mut request.user_context);
        self.fill_client_type(&mut request.client_type);
        if let Some(plan) = request.plan.as_mut() {
            self.compress_plan(plan).await;
        }
        // After the lookup of the plan compression of the first plan, whose response gives the
        // server-side session ID, as PySpark builds the plan before the request.
        self.observe_on(&mut request.client_observed_server_side_session_id);
        let request = self.hooked_and_compressed(request).await?;
        // As in `execute_plan`: the lookup of the plan compression may have closed the client,
        // and PySpark reads its `_stub` before it makes the iterator, so nothing is released.
        self.ensure_open()?;
        let iter = ExecutePlanResponseReattachableIterator::new(request);
        // Before the first RPC, so that an execution that waits for the server is in flight.
        let inflight = InflightGuard::register(
            std::sync::Arc::clone(&self.inflight_operations),
            iter.operation_id().to_string(),
        );
        let deadline = self.rpc_deadlines.reattachable_execute_plan;
        let mut stream_deadline = None;
        let stream = {
            // The first RPC is a wait of the caller for the execution.
            let _waiting = inflight.waiting();
            self.with_retry(|| {
                let request = clone_on_grown_stack(iter.request());
                let mut req = with_deadline(Request::new(request), deadline);
                self._attach_metadata(&mut req);
                stream_deadline = stream_deadline_in(deadline);
                let mut stub = self.stub.clone();
                async move {
                    within_deadline(deadline, stub.execute_plan(req))
                        .await
                        .map(Response::into_inner)
                }
            })
            .await
        };
        let stream = match stream {
            Ok(stream) => Some(stream),
            // As the reference iterator, past its deadline an execution is resumed by
            // reattaching, as it goes on on the server.
            Err(error) if error.grpc_code == Some(tonic::Code::DeadlineExceeded as i32) => None,
            Err(error) => {
                // As the reference iterator releases an execution that fails at once, such
                // as for an analysis error, so the server does not keep it until its
                // detached timeout.
                self.spawn_release(self.execution_metadata(), iter.release_all_request());
                return Err(error.with_operation_id(iter.operation_id()));
            }
        };
        let retry_policy = self.retry_policy.clone();
        Ok(ReattachableResponseStream {
            client: self.clone(),
            retry_policy,
            iter,
            stream,
            stream_deadline,
            execute_again: false,
            done: false,
            released: false,
            _inflight: inflight,
        })
    }

    /// Fetch enriched error details (full exception tree/stack trace) by error id.
    ///
    /// Mirrors `SparkConnectClient._fetch_enriched_error`; used to reconstruct the full
    /// server-side exception message (e.g. wrapped worker errors).
    pub async fn fetch_error_details(
        &self,
        mut request: FetchErrorDetailsRequest,
    ) -> Result<FetchErrorDetailsResponse> {
        self.ensure_open()?;
        self.extend_user_context(&mut request.user_context);
        self.fill_user_context(&mut request.user_context);
        self.fill_client_type(&mut request.client_type);
        self.observe_on(&mut request.client_observed_server_side_session_id);
        let deadline = self.rpc_deadlines.fetch_error_details;
        let mut req = with_deadline(Request::new(request), deadline);
        self._attach_metadata(&mut req);
        let resp = within_deadline(deadline, self.stub.clone().fetch_error_details(req)).await;
        let response = resp
            .map(Response::into_inner)
            .map_err(|status| self.grpc_error(status))?;
        self.verify_response(&response.session_id, &response.server_side_session_id)?;
        Ok(response)
    }

    /// The error for a failed RPC, as `SparkConnectClient._handle_rpc_error` raises it. An
    /// error that the server raised gets the details that `FetchErrorDetails` returns when the
    /// server keeps them, and shows the server's stack trace when
    /// `spark.sql.connect.serverStacktrace.enabled` or `spark.sql.pyspark.jvmStacktrace.enabled`
    /// is true. As the error is raised either way, failing to read those is not.
    pub async fn error_from_status(&self, status: tonic::Status) -> SparkError {
        let error = self.grpc_error(status);
        let Some(server) = &error.server else {
            return error;
        };
        let details = match &server.error_id {
            Some(error_id) => {
                // PySpark fetches the details through `_stub`, which raises NO_ACTIVE_SESSION
                // once the client is closed, as a session change closes it; that replaces the
                // error of the server.
                if let Err(closed) = self.ensure_open() {
                    return closed;
                }
                self.error_details(error_id).await
            }
            None => None,
        };
        let display_stacktrace = self.display_server_stacktrace().await;
        error.with_details(details, display_stacktrace)
    }

    /// The details of the error that the server keeps under `error_id`, or `None` if they
    /// cannot be fetched. Mirrors `SparkConnectClient._fetch_enriched_error`.
    async fn error_details(&self, error_id: &str) -> Option<FetchErrorDetailsResponse> {
        let request = FetchErrorDetailsRequest {
            session_id: self.session_id.clone(),
            error_id: error_id.to_string(),
            client_type: Some(self.user_agent.clone()),
            user_context: self.user_id.as_ref().map(|user_id| UserContext {
                user_id: user_id.clone(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut request = request;
        // As `_fetch_enriched_error`, whose request carries the user context extensions.
        self.extend_user_context(&mut request.user_context);
        self.observe_on(&mut request.client_observed_server_side_session_id);
        let deadline = self.rpc_deadlines.fetch_error_details;
        let mut req = with_deadline(Request::new(request), deadline);
        self._attach_metadata(&mut req);
        let response = within_deadline(deadline, self.stub.clone().fetch_error_details(req))
            .await
            .ok()?
            .into_inner();
        self.verify_response(&response.session_id, &response.server_side_session_id)
            .ok()?;
        Some(response)
    }

    /// Whether the message of an error that the server raised shows the server's stack
    /// trace. Mirrors `SparkConnectClient._display_server_stack_trace`, which shows it when a
    /// config cannot be read.
    async fn display_server_stacktrace(&self) -> bool {
        match self
            .conf_while_raising("spark.sql.connect.serverStacktrace.enabled")
            .await
        {
            Some(enabled) if enabled == "true" => true,
            Some(_) => self
                .conf_while_raising("spark.sql.pyspark.jvmStacktrace.enabled")
                .await
                .is_none_or(|enabled| enabled == "true"),
            None => true,
        }
    }

    /// The value of a config, or `None` if it cannot be read. Read once, while an error is
    /// raised, so that a failure is neither retried nor raised.
    async fn conf_while_raising(&self, key: &str) -> Option<String> {
        use spark_connect_proto::config_request::{operation::OpType, Get, Operation};
        // PySpark reads the config through `_stub`, which raises NO_ACTIVE_SESSION once the
        // client is closed, as a session change closes it, so it sends nothing then.
        if self.is_closed() {
            return None;
        }
        let request = ConfigRequest {
            session_id: self.session_id.clone(),
            user_context: Some(UserContext {
                user_id: self.user_id.clone().unwrap_or_default(),
                ..Default::default()
            }),
            operation: Some(Operation {
                op_type: Some(OpType::Get(Get {
                    keys: vec![key.to_string()],
                })),
            }),
            client_type: Some(self.user_agent.clone()),
            ..Default::default()
        };
        let mut request = request;
        // As the request of PySpark's `RuntimeConf.get`, which carries the user context
        // extensions, as every Config request.
        self.extend_user_context(&mut request.user_context);
        self.observe_on(&mut request.client_observed_server_side_session_id);
        let deadline = self.rpc_deadlines.config;
        let mut req = with_deadline(Request::new(request), deadline);
        self._attach_metadata(&mut req);
        let response = within_deadline(deadline, self.stub.clone().config(req))
            .await
            .ok()?
            .into_inner();
        self.verify_response(&response.session_id, &response.server_side_session_id)
            .ok()?;
        response.pairs.into_iter().next()?.value
    }

    /// Update or fetch configurations.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.config`.
    pub async fn config(&self, mut request: ConfigRequest) -> Result<ConfigResponse> {
        self.ensure_open()?;
        self.extend_user_context(&mut request.user_context);
        self.fill_user_context(&mut request.user_context);
        self.fill_client_type(&mut request.client_type);
        let response = self
            .with_retry(|| {
                let deadline = self.rpc_deadlines.config;
                let mut request = request.clone();
                self.observe_on(&mut request.client_observed_server_side_session_id);
                let mut req = with_deadline(Request::new(request), deadline);
                self._attach_metadata(&mut req);
                let mut stub = self.stub.clone();
                async move {
                    within_deadline(deadline, stub.config(req))
                        .await
                        .map(Response::into_inner)
                }
            })
            .await?;
        self.verify_response(&response.session_id, &response.server_side_session_id)?;
        Ok(response)
    }

    /// Interrupt running operations on this session.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.interrupt`.
    pub async fn interrupt(&self, mut request: InterruptRequest) -> Result<InterruptResponse> {
        self.ensure_open()?;
        self.extend_user_context(&mut request.user_context);
        self.fill_user_context(&mut request.user_context);
        self.fill_client_type(&mut request.client_type);
        let response = self
            .with_retry(|| {
                let deadline = self.rpc_deadlines.interrupt;
                let mut request = request.clone();
                self.observe_on(&mut request.client_observed_server_side_session_id);
                let mut req = with_deadline(Request::new(request), deadline);
                self._attach_metadata(&mut req);
                let mut stub = self.stub.clone();
                async move {
                    within_deadline(deadline, stub.interrupt(req))
                        .await
                        .map(Response::into_inner)
                }
            })
            .await?;
        self.verify_response(&response.session_id, &response.server_side_session_id)?;
        Ok(response)
    }

    /// Get configuration values from the session.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.get_configs`.
    pub async fn get_configs(&self, keys: &[&str]) -> Result<Vec<Option<String>>> {
        let mut request = ConfigRequest::default();
        request.session_id = self.session_id.clone();
        request.user_context = Some(UserContext {
            user_id: self.user_id.clone().unwrap_or_default(),
            ..Default::default()
        });

        // Create the Get operation
        let mut operation = spark_connect_proto::config_request::Operation::default();
        let mut get = spark_connect_proto::config_request::Get::default();
        get.keys = keys.iter().map(|k| k.to_string()).collect();
        operation.op_type = Some(spark_connect_proto::config_request::operation::OpType::Get(
            get,
        ));
        request.operation = Some(operation);

        let response = self.config(request).await?;

        // Build a dict from response pairs
        let mut config_dict = BTreeMap::new();
        for pair in response.pairs {
            if let Some(value) = pair.value {
                config_dict.insert(pair.key, value);
            }
        }

        Ok(keys.iter().map(|k| config_dict.get(*k).cloned()).collect())
    }

    /// Set configuration values for the session, returning the warnings of the server, such
    /// as of a deprecated or unsupported configuration.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.set_config`.
    pub async fn set_config(&self, key: &str, value: &str) -> Result<Vec<String>> {
        self.set_config_values(&[(key, Some(value))], None).await
    }

    /// Set configuration values in one request, ignoring the failures to set them when
    /// `silent`, returning the warnings of the server. Mirrors
    /// `pyspark.sql.connect.conf.RuntimeConf._set_all`, with which the session builder applies
    /// its options.
    pub async fn set_configs(
        &self,
        pairs: &[(String, String)],
        silent: bool,
    ) -> Result<Vec<String>> {
        let pairs: Vec<(&str, Option<&str>)> = pairs
            .iter()
            .map(|(key, value)| (key.as_str(), Some(value.as_str())))
            .collect();
        self.set_config_values(&pairs, Some(silent)).await
    }

    /// Set configuration values in one request, as PySpark's `RuntimeConf.set` and `_set_all`
    /// send them: a pair without a value is sent without one, as PySpark sends None, which the
    /// server rejects, and with `silent` the failures to set them, such as of static
    /// configurations, are ignored. Returns the warnings of the server, such as of a
    /// deprecated or unsupported configuration, which PySpark warns.
    pub async fn set_config_values(
        &self,
        pairs: &[(&str, Option<&str>)],
        silent: Option<bool>,
    ) -> Result<Vec<String>> {
        let set = spark_connect_proto::config_request::Set {
            pairs: pairs
                .iter()
                .map(|(key, value)| spark_connect_proto::KeyValue {
                    key: key.to_string(),
                    value: value.map(str::to_string),
                })
                .collect(),
            silent,
        };
        let request = ConfigRequest {
            session_id: self.session_id.clone(),
            user_context: Some(UserContext {
                user_id: self.user_id.clone().unwrap_or_default(),
                ..Default::default()
            }),
            operation: Some(spark_connect_proto::config_request::Operation {
                op_type: Some(spark_connect_proto::config_request::operation::OpType::Set(
                    set,
                )),
            }),
            ..Default::default()
        };
        Ok(self.config(request).await?.warnings)
    }

    /// Get configuration values with default fallbacks.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.get_config_with_defaults`.
    pub async fn get_config_with_defaults(
        &self,
        pairs: &[(&str, Option<&str>)],
    ) -> Result<Vec<Option<String>>> {
        let mut request = ConfigRequest::default();
        request.session_id = self.session_id.clone();
        request.user_context = Some(UserContext {
            user_id: self.user_id.clone().unwrap_or_default(),
            ..Default::default()
        });

        let mut get_with_default = spark_connect_proto::config_request::GetWithDefault::default();
        get_with_default.pairs = pairs
            .iter()
            .map(|(key, default_value)| spark_connect_proto::KeyValue {
                key: key.to_string(),
                value: default_value.map(|v| v.to_string()),
            })
            .collect();

        let mut operation = spark_connect_proto::config_request::Operation::default();
        operation.op_type = Some(
            spark_connect_proto::config_request::operation::OpType::GetWithDefault(
                get_with_default,
            ),
        );
        request.operation = Some(operation);

        let response = self.config(request).await?;

        let mut config_dict = BTreeMap::new();
        for pair in response.pairs {
            if let Some(value) = pair.value {
                config_dict.insert(pair.key, value);
            }
        }

        Ok(pairs
            .iter()
            .map(|(k, _)| config_dict.get(*k).cloned())
            .collect())
    }

    /// Unset a configuration value, returning the warnings of the server, such as of a
    /// deprecated or unsupported configuration.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.unset_config`.
    pub async fn unset_config(&self, key: &str) -> Result<Vec<String>> {
        let mut request = ConfigRequest::default();
        request.session_id = self.session_id.clone();
        request.user_context = Some(UserContext {
            user_id: self.user_id.clone().unwrap_or_default(),
            ..Default::default()
        });

        let mut unset = spark_connect_proto::config_request::Unset::default();
        unset.keys = vec![key.to_string()];

        let mut operation = spark_connect_proto::config_request::Operation::default();
        operation.op_type =
            Some(spark_connect_proto::config_request::operation::OpType::Unset(unset));
        request.operation = Some(operation);

        Ok(self.config(request).await?.warnings)
    }

    /// Get all configuration values from the session.
    ///
    /// Returns a HashMap of all configuration key-value pairs.
    pub async fn get_configs_all(&self) -> Result<std::collections::HashMap<String, String>> {
        let mut request = ConfigRequest::default();
        request.session_id = self.session_id.clone();
        request.user_context = Some(UserContext {
            user_id: self.user_id.clone().unwrap_or_default(),
            ..Default::default()
        });

        let mut operation = spark_connect_proto::config_request::Operation::default();
        let get_all = spark_connect_proto::config_request::GetAll::default();
        operation.op_type =
            Some(spark_connect_proto::config_request::operation::OpType::GetAll(get_all));
        request.operation = Some(operation);

        let response = self.config(request).await?;

        let mut config_dict = std::collections::HashMap::new();
        for pair in response.pairs {
            if let Some(value) = pair.value {
                config_dict.insert(pair.key, value);
            }
        }

        Ok(config_dict)
    }

    /// Check if a configuration key is modifiable.
    ///
    /// Returns true if the configuration can be changed at runtime.
    pub async fn is_config_modifiable(&self, key: &str) -> Result<bool> {
        let mut request = ConfigRequest::default();
        request.session_id = self.session_id.clone();
        request.user_context = Some(UserContext {
            user_id: self.user_id.clone().unwrap_or_default(),
            ..Default::default()
        });

        let mut is_modifiable = spark_connect_proto::config_request::IsModifiable::default();
        is_modifiable.keys = vec![key.to_string()];

        let mut operation = spark_connect_proto::config_request::Operation::default();
        operation.op_type = Some(
            spark_connect_proto::config_request::operation::OpType::IsModifiable(is_modifiable),
        );
        request.operation = Some(operation);

        let response = self.config(request).await?;

        if let Some(pair) = response.pairs.first() {
            if let Some(value) = &pair.value {
                return Ok(value == "true");
            }
        }

        Ok(false)
    }

    /// Interrupt all running operations in this session.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.interrupt_all`.
    pub async fn interrupt_all(&self) -> Result<Vec<String>> {
        let mut request = InterruptRequest::default();
        request.session_id = self.session_id.clone();
        request.user_context = Some(UserContext {
            user_id: self.user_id.clone().unwrap_or_default(),
            ..Default::default()
        });
        // InterruptType::INTERRUPT_TYPE_ALL = 1
        request.interrupt_type = 1;

        let response = self.interrupt(request).await?;
        Ok(response.interrupted_ids)
    }

    /// Interrupt all running operations with a specific tag.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.interrupt_tag`.
    pub async fn interrupt_tag(&self, tag: &str) -> Result<Vec<String>> {
        let mut request = InterruptRequest::default();
        request.session_id = self.session_id.clone();
        request.user_context = Some(UserContext {
            user_id: self.user_id.clone().unwrap_or_default(),
            ..Default::default()
        });
        // InterruptType::INTERRUPT_TYPE_TAG = 2
        request.interrupt_type = 2;
        request.interrupt =
            Some(spark_connect_proto::interrupt_request::Interrupt::OperationTag(tag.to_string()));

        let response = self.interrupt(request).await?;
        Ok(response.interrupted_ids)
    }

    /// Interrupt a specific operation by ID.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.interrupt_operation`.
    pub async fn interrupt_operation(&self, operation_id: &str) -> Result<Vec<String>> {
        let mut request = InterruptRequest::default();
        request.session_id = self.session_id.clone();
        request.user_context = Some(UserContext {
            user_id: self.user_id.clone().unwrap_or_default(),
            ..Default::default()
        });
        // InterruptType::INTERRUPT_TYPE_OPERATION_ID = 3
        request.interrupt_type = 3;
        request.interrupt = Some(
            spark_connect_proto::interrupt_request::Interrupt::OperationId(
                operation_id.to_string(),
            ),
        );

        let response = self.interrupt(request).await?;
        Ok(response.interrupted_ids)
    }

    /// Release this session and all its resources on the server.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.release_session`.
    /// Clone this client's session on the server, mirroring PySpark's
    /// `SparkConnectClient.clone(new_session_id)`: a `CloneSession` RPC copies the server-side
    /// state of the session, its SQL configurations, temporary views, registered functions
    /// and catalog state, into a new session, of the id `new_session_id` if one is given. The
    /// returned client, of the same channel, is of that session.
    pub async fn clone_session(&self, new_session_id: Option<&str>) -> Result<Self> {
        self.ensure_open()?;
        // As PySpark's request: of the client type "python", not of the user agent, and with
        // no observed server-side session ID.
        let request = CloneSessionRequest {
            session_id: self.session_id.clone(),
            user_context: Some(UserContext {
                user_id: self.user_id.clone().unwrap_or_default(),
                ..Default::default()
            }),
            client_type: Some("python".to_string()),
            new_session_id: new_session_id.map(str::to_string),
            ..Default::default()
        };
        let response = self
            .with_retry(|| {
                let deadline = self.rpc_deadlines.clone_session;
                let mut req = with_deadline(Request::new(request.clone()), deadline);
                self._attach_metadata(&mut req);
                let mut stub = self.stub.clone();
                async move {
                    within_deadline(deadline, stub.clone_session(req))
                        .await
                        .map(Response::into_inner)
                }
            })
            .await?;
        self.verify_response(&response.session_id, &response.server_side_session_id)?;
        if let Some(requested) = new_session_id {
            if response.new_session_id != requested {
                return Err(SparkError::connect_msg(format!(
                    "Returned session ID '{}' does not match requested ID '{requested}'",
                    response.new_session_id
                )));
            }
        }
        let mut cloned = self.with_new_session_id();
        cloned.session_id = response.new_session_id;
        Ok(cloned)
    }

    /// Release this session on the server.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.release_session`, which
    /// retries it under the retry policy: a transient failure at stop does not leave the
    /// session on the server until its timeout.
    pub async fn release_session(&self) -> Result<()> {
        let mut request = ReleaseSessionRequest::default();
        request.session_id = self.session_id.clone();
        request.user_context = Some(UserContext {
            user_id: self.user_id.clone().unwrap_or_default(),
            ..Default::default()
        });
        request.client_type = Some(self.user_agent.clone());

        let response = self
            .with_retry(|| {
                let deadline = self.rpc_deadlines.release_session;
                let mut req = with_deadline(Request::new(request.clone()), deadline);
                self._attach_metadata(&mut req);
                let mut stub = self.stub.clone();
                async move {
                    within_deadline(deadline, stub.release_session(req))
                        .await
                        .map(Response::into_inner)
                }
            })
            .await?;
        self.verify_response(&response.session_id, &response.server_side_session_id)
    }

    /// Add artifacts to this session.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.add_artifacts`, which
    /// uploads each path in an attempt of its own under the retry policy. Handles chunking
    /// large artifacts and batching small ones according to CHUNK_SIZE.
    pub async fn add_artifacts(
        &self,
        paths: &[&str],
        pyfile: bool,
        archive: bool,
        file: bool,
    ) -> Result<()> {
        self.ensure_open()?;
        // Each path is named as PySpark's ArtifactManager names it, by the flags and its file
        // extension (see `parse_artifact`).
        for path in paths {
            let (name, local) = crate::artifact::parse_artifact(path, pyfile, archive, file)
                .map_err(|e| SparkError::connect_msg(e.to_string()))?;
            let artifact = NamedArtifact::new(name, Box::new(FileArtifact::new(local)));
            self.upload_artifact(&artifact).await?;
        }
        Ok(())
    }

    /// Upload a single local file as an artifact under an explicit artifact name
    /// (rather than deriving `files/`/`jars/`… from the extension).
    ///
    /// Used by `SparkSession::copy_from_local_to_fs`, which names the artifact
    /// `forward_to_fs/<dest>` so the server writes it to the target filesystem path.
    pub async fn add_named_artifact(&self, name: &str, local_path: &str) -> Result<()> {
        self.ensure_open()?;
        let artifact =
            NamedArtifact::new(name.to_string(), Box::new(FileArtifact::new(local_path)));
        self.upload_artifact(&artifact).await
    }

    /// Upload session-scoped cache blobs, returning SHA-256 hashes in input order.
    /// Existing artifacts and duplicate blobs are uploaded only once.
    ///
    /// As `SparkConnectClient.cache_artifacts`, the status check and the upload are one
    /// attempt under the retry policy, so a transient failure of either retries both.
    pub async fn cache_artifacts(&self, blobs: Vec<Vec<u8>>) -> Result<Vec<String>> {
        self.ensure_open()?;
        use bytes::Bytes;
        use std::collections::HashSet;

        if blobs.is_empty() {
            return Ok(Vec::new());
        }
        let (blobs, hashes) = hash_artifact_blobs(blobs).await?;
        let names: Vec<String> = hashes.iter().map(|hash| format!("cache/{hash}")).collect();
        let blobs: Vec<Bytes> = blobs.into_iter().map(Bytes::from).collect();
        let unique: Vec<String> = names
            .iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .cloned()
            .collect();
        let (blobs, names, unique) = (&blobs, &names, &unique);
        // As PySpark's ArtifactManager, which does not check the IDs of its responses: a change of
        // the server-side session is raised by the next call of the client.
        let uploaded = self
            .with_retry(|| async move {
                let status = self.artifact_status_rpc(unique).await?;
                let cached: std::collections::HashMap<String, bool> = status
                    .statuses
                    .into_iter()
                    .map(|(name, status)| (name, status.exists))
                    .collect();
                let mut seen = HashSet::new();
                let mut artifacts = Vec::new();
                for (blob, name) in blobs.iter().zip(names) {
                    if !cached.get(name).copied().unwrap_or(false) && seen.insert(name.clone()) {
                        artifacts.push((name.clone(), blob.clone()));
                    }
                }
                if artifacts.is_empty() {
                    return Ok(None);
                }
                let expected: HashSet<String> =
                    artifacts.iter().map(|(name, _)| name.clone()).collect();
                let requests = cache_artifact_requests(self.session_id.clone(), artifacts);
                let response = self.add_artifacts_rpc(requests).await?;
                Ok(Some((response, expected)))
            })
            .await?;
        if let Some((response, expected)) = uploaded {
            check_artifact_upload(response, Some(&expected))?;
        }
        Ok(hashes)
    }

    /// The identity of a typed request that the caller built: adds the user context when it
    /// is missing, and takes a missing user ID from the connection (`user_id` of the
    /// channel), as the reference client sets `user_context.user_id` on the requests it
    /// builds. The server keys a session by the user ID of each request, so without it the
    /// request would reach another session than the RPCs this client builds itself. An
    /// explicit user ID, the user name and the extensions are kept; raw requests are sent
    /// as given.
    fn fill_user_context(&self, context: &mut Option<UserContext>) {
        let context = context.get_or_insert_with(|| self.new_user_context());
        if context.user_id.is_empty() {
            if let Some(user_id) = &self.user_id {
                context.user_id.clone_from(user_id);
            }
        }
    }

    /// The client type of a typed request that the caller built: the user agent, unless it has
    /// one, as PySpark's requests carry `client_type=self._builder.userAgent`.
    fn fill_client_type(&self, client_type: &mut Option<String>) {
        if client_type.is_none() {
            *client_type = Some(self.user_agent.clone());
        }
    }

    /// Build a fresh request context, even when no user ID is configured.
    /// PySpark and Scala both send a present context on artifact requests.
    fn new_user_context(&self) -> UserContext {
        UserContext {
            user_id: self.user_id.clone().unwrap_or_default(),
            ..Default::default()
        }
    }

    /// Upload one artifact, retrying a transient failure of the whole upload under the retry
    /// policy, and validate the response. An attempt sends all the frames again from the start:
    /// a stream is not resumed. The frames are read as the RPC consumes them, so an attempt
    /// holds one chunk of the artifact at a time.
    async fn upload_artifact(&self, artifact: &NamedArtifact) -> Result<()> {
        let response = self
            .with_attempts(|| async move {
                // A data source that cannot be opened is an error before any frame is sent.
                let frames = ArtifactFrames::new(self.session_id.clone(), artifact)
                    .map_err(RetryError::Aborted)?;
                let failure = frames.failure();
                // An RPC that a signal ends is not answered, as an attempt of with_retry.
                let result = crate::runtime::until_abort(
                    crate::runtime::ABORT_RETRIES,
                    self.add_artifacts_rpc(frames),
                )
                .await;
                let result = match result {
                    Some(result) => result,
                    None => return Err(RetryError::Aborted(wait_aborted())),
                };
                // A read that failed ended the frames early, whatever the RPC answered.
                take_frame_failure(&failure).map_err(RetryError::Aborted)?;
                result.map_err(RetryError::Status)
            })
            .await?;
        // The IDs of the response are not checked, as by PySpark's ArtifactManager.
        check_artifact_upload(response, None)
    }

    /// The AddArtifacts call of `requests`, once. Stamps user_context on every frame as it is
    /// sent; all upload paths must go through here. As PySpark's ArtifactManager, the frames
    /// carry no client type and no observed server-side session ID.
    async fn add_artifacts_rpc(
        &self,
        requests: impl Iterator<Item = spark_connect_proto::AddArtifactsRequest> + Send + 'static,
    ) -> std::result::Result<spark_connect_proto::AddArtifactsResponse, tonic::Status> {
        let user_context = self.new_user_context();
        let requests = requests.map(move |mut request| {
            request.user_context = Some(user_context.clone());
            request
        });
        let deadline = self.rpc_deadlines.add_artifacts;
        let mut req = with_deadline(Request::new(futures::stream::iter(requests)), deadline);
        self._attach_metadata(&mut req);
        within_deadline(deadline, self.stub.clone().add_artifacts(req))
            .await
            .map(Response::into_inner)
    }

    /// Check artifact status on the server.
    ///
    /// Mirrors functionality from `pyspark.sql.connect.client.artifact.ArtifactManager.is_cached_artifact`.
    pub async fn artifact_status(
        &self,
        names: &[&str],
    ) -> Result<std::collections::HashMap<String, bool>> {
        self.ensure_open()?;
        let names: Vec<String> = names.iter().map(|n| n.to_string()).collect();
        match self.artifact_status_rpc(&names).await {
            // The IDs of the response are not checked, as by PySpark's ArtifactManager.
            Ok(response) => Ok(response
                .statuses
                .into_iter()
                .map(|(name, status)| (name, status.exists))
                .collect()),
            Err(status) => Err(self.error_from_status(status).await),
        }
    }

    /// The ArtifactStatus call of `names`, once: the response, which says whether each artifact
    /// exists. As PySpark's, the request carries no client type and no observed server-side
    /// session ID.
    async fn artifact_status_rpc(
        &self,
        names: &[String],
    ) -> std::result::Result<spark_connect_proto::ArtifactStatusesResponse, tonic::Status> {
        let mut request = ArtifactStatusesRequest::default();
        request.session_id = self.session_id.clone();
        request.user_context = Some(self.new_user_context());
        request.names = names.to_vec();

        let deadline = self.rpc_deadlines.artifact_status;
        let mut req = with_deadline(Request::new(request), deadline);
        self._attach_metadata(&mut req);
        within_deadline(deadline, self.stub.clone().artifact_status(req))
            .await
            .map(Response::into_inner)
    }

    /// Attach metadata headers to a request.
    ///
    /// Inserts the user agent, the metadata of the connection string and the token (if
    /// present) into the gRPC metadata. They were validated by `connect`.
    fn _attach_metadata<T>(&self, req: &mut Request<T>) {
        let metadata = req.metadata_mut();
        // NOTE: the session id is carried in the request *body* (proto), not in gRPC
        // metadata. The reference client sends no `session_id` header; sending one
        // (with our own client-generated id, which differs from the body's session id
        // when we forward a request built by another client) can make the server
        // associate work with the wrong session. So we do not attach it here.
        // No user_id header: as PySpark's `ChannelBuilder.metadata()`, which leaves the user ID
        // to the user context of the requests.
        for (key, value) in &self.headers {
            metadata.insert(key.clone(), value.clone());
        }
    }
}

/// The key of a metadata parameter of the connection string as a gRPC header key, or the
/// error of grpc-python for it. gRPC keys are of `[0-9a-z_.-]`, which is narrower than the
/// HTTP header names that tonic takes (it would lower-case `X-Hdr`): grpc-python raises
/// `ValueError("metadata was invalid: ...")` for any other key when the call starts. A `-bin`
/// key takes bytes, which a connection string cannot give, so it is the `TypeError` that
/// grpc-python raises for the `str` value of one.
fn metadata_key(key: &str) -> Result<AsciiMetadataKey> {
    let invalid = || {
        SparkError::builtin_value_msg(format!(
            "metadata was invalid: '{}' is not a gRPC metadata key, which is of the \
             characters [0-9a-z_.-]",
            key.escape_debug()
        ))
    };
    let legal = |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit() || b"_.-".contains(&b);
    if key.is_empty() || !key.bytes().all(legal) {
        return Err(invalid());
    }
    if key.ends_with("-bin") {
        let mut error = SparkError::value_msg(format!(
            "Binary metadata key='{key}' expected bytes, got a string"
        ));
        error.kind = SparkErrorKind::TypeError;
        return Err(error);
    }
    AsciiMetadataKey::from_bytes(key.as_bytes()).map_err(|_| invalid())
}

/// The value of the metadata parameter `key` as a gRPC header value. A value is of the
/// printable ASCII characters and the space (not the tab that HTTP allows), as in grpc-python,
/// which raises `ValueError("metadata was invalid: ...")` for any other. The error names the
/// key only: the value may be a secret.
fn metadata_value(key: &str, value: &str) -> Result<AsciiMetadataValue> {
    let invalid = || {
        SparkError::builtin_value_msg(format!(
            "metadata was invalid: the value of '{key}' is not a gRPC metadata value, which \
             is of the printable ASCII characters"
        ))
    };
    let mut header = ascii_header_value(value).ok_or_else(invalid)?;
    header.set_sensitive(is_secret_param(key));
    Ok(header)
}

/// The `Bearer <token>` of the token as the value of the `authorization` header. grpc-python,
/// whose call credentials add the header, fails every call of a token that is not a header
/// value with `UNAVAILABLE: Illegal metadata`; this is that error, which never holds the
/// token. The value is marked sensitive, so that its `Debug` does not show it.
fn token_header(bearer: &str) -> Result<AsciiMetadataValue> {
    let mut header = ascii_header_value(bearer).ok_or_else(|| {
        SparkError::from_grpc_status(tonic::Status::unavailable(
            "Illegal metadata: the token is not a gRPC header value, which is of the \
             printable ASCII characters",
        ))
    })?;
    header.set_sensitive(true);
    Ok(header)
}

/// `value` as a header value of gRPC, if it is of the printable ASCII characters and the space.
fn ascii_header_value(value: &str) -> Option<AsciiMetadataValue> {
    if !value.bytes().all(|b| (0x20..=0x7e).contains(&b)) {
        return None;
    }
    AsciiMetadataValue::try_from(value).ok()
}

/// The user agent as the value of the `user-agent` header. The reference client sets it as a
/// channel option, which gRPC sends as it is, so that its UTF-8 bytes go unchanged. HTTP/2
/// cannot carry a control character, which is an error here rather than a request without
/// the header.
fn user_agent_header(user_agent: &str) -> Result<AsciiMetadataValue> {
    AsciiMetadataValue::try_from(user_agent.as_bytes()).map_err(|_| {
        SparkError::builtin_value_msg(
            "'user_agent' parameter should not contain control characters: it is sent as \
             the user-agent header.",
        )
    })
}

/// A reattachable, retrying response stream for a reattachable ExecutePlan.
///
/// Drives the gRPC response stream and resumes it via `ReattachExecute` from the last
/// observed `response_id` (with backoff) whenever it ends before `ResultComplete`: on a
/// transient mid-stream disconnect, and when the server completes the stream on purpose
/// (after `spark.connect.execute.reattachable.senderMaxStreamDuration` or
/// `senderMaxStreamSize`). It releases each response once it is read, and the whole
/// execution once the result is complete, the stream fails, or it is dropped, so that the
/// server does not keep them until the execution is abandoned. Mirrors
/// `pyspark.sql.connect.client.reattach.ExecutePlanResponseReattachableIterator`.
///
/// # Chunked Arrow batches
///
/// The execution allows the server to split an Arrow batch that is too large for one message.
/// The chunks of a batch come as responses in order, each with an `arrow_batch` whose
/// `chunk_index` is its place among the `num_chunks_in_batch` chunks of the batch. Other
/// responses may come between the chunks. `start_offset`, when the server sets it, is the row
/// where the batch starts, and `row_count` is the number of rows of the whole batch, not of
/// the chunk.
///
/// This stream returns the responses as they arrive, so the `arrow_batch.data` of a chunk
/// is a part of the Arrow IPC stream of its batch, not a batch of its own. A consumer that
/// reads the responses itself must join the `data` of the chunks of a batch, in order, before
/// it decodes the batch, and must check the order and the count of the chunks as PySpark
/// does. The high-level APIs of the `apache-spark-connect` crate, such as `DataFrame::collect`,
/// do this join for their results.
pub struct ReattachableResponseStream {
    /// The client that started the execution, whose stub, headers and session it uses.
    client: SparkConnectClient,
    retry_policy: RetryPolicy,
    iter: ExecutePlanResponseReattachableIterator,
    /// The stream of the execution, or none to reattach before reading.
    stream: Option<Streaming<ExecutePlanResponse>>,
    /// When the deadline of the stream passes, as gRPC's of the call that opened it.
    stream_deadline: Option<tokio::time::Instant>,
    /// Whether the next stream is opened by sending the ExecutePlan again, rather than by
    /// reattaching: the server does not know the execution, which never reached it.
    execute_again: bool,
    done: bool,
    /// Whether `ReleaseAll` was sent, after which nothing else is released.
    released: bool,
    /// The registration of the execution as in flight, which ends with the stream.
    _inflight: InflightGuard,
}

/// The user ID of a connection string that does not set one: `SPARK_USER`, else `USER`, as
/// `SparkConnectClient.__init__` in PySpark takes it.
fn default_user_id() -> Option<String> {
    user_id_from_env(std::env::var("SPARK_USER"), std::env::var("USER"))
}

/// The user ID from the environment, as `os.getenv("SPARK_USER", os.getenv("USER"))`: `USER`
/// is only read when `SPARK_USER` is unset. An empty value is no user ID.
fn user_id_from_env(
    spark_user: std::result::Result<String, std::env::VarError>,
    user: std::result::Result<String, std::env::VarError>,
) -> Option<String> {
    spark_user.or(user).ok().filter(|u| !u.is_empty())
}

impl SparkConnectClient {
    /// The gRPC metadata of every request about an execution.
    fn execution_metadata(&self) -> MetadataMap {
        let mut metadata = MetadataMap::new();
        for (key, value) in &self.headers {
            metadata.insert(key.clone(), value.clone());
        }
        metadata
    }

    /// Sends `request` in the background, so that reading is not delayed, retrying under the
    /// client's retry policy. Like the reference client, a failure is only logged: the server
    /// removes an execution that is never released once it has been detached for
    /// `spark.connect.execute.manager.detachedTimeout`. As the reference's, the request carries
    /// no observed server-side session ID.
    fn spawn_release(&self, metadata: MetadataMap, request: ReleaseExecuteRequest) {
        let stub = self.stub.clone();
        let policy = self.retry_policy.clone();
        let release = async move {
            // As the reference's, which goes through the stub of its iterator rather than the
            // `_stub` of the client: a release is retried whether or not the client was closed.
            let whatever_the_client = || Ok::<(), std::convert::Infallible>(());
            let released = retrying(Retrying::new(&policy), whatever_the_client, || {
                let mut req = Request::new(request.clone());
                *req.metadata_mut() = metadata.clone();
                let mut stub = stub.clone();
                async move {
                    stub.release_execute(req)
                        .await
                        .map(drop)
                        .map_err(RetryError::Status)
                }
            })
            .await;
            match released {
                Ok(()) => {}
                Err(RetryError::Status(status)) => tracing::warn!(
                    "ReleaseExecute of operation {} failed: {}",
                    request.operation_id,
                    status
                ),
                Err(RetryError::Aborted(never)) => match never {},
                // Its backoffs are waited whole.
                Err(RetryError::WaitAborted) => {}
            }
        };
        // A stream is read inside a runtime, but it can be dropped outside of any, for
        // example after the `block_on` that read it.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => drop(handle.spawn(release)),
            Err(_) => drop(crate::runtime::get_runtime().spawn(release)),
        }
    }
}

impl ReattachableResponseStream {
    /// The handle of the execution of this stream, to mark the waits of the caller that starts it
    /// for the execution while the caller reads its responses from another thread (see
    /// [`InflightHandle::waiting`]).
    pub fn inflight_handle(&self) -> InflightHandle {
        self._inflight.handle()
    }

    /// The gRPC metadata of every request about this execution.
    fn metadata_map(&self) -> MetadataMap {
        self.client.execution_metadata()
    }

    fn attach_metadata<T>(&self, req: &mut Request<T>) {
        *req.metadata_mut() = self.metadata_map();
    }

    /// Next response, transparently reattaching whenever the stream ends before
    /// `ResultComplete`.
    pub async fn message(&mut self) -> Result<Option<ExecutePlanResponse>> {
        if self.done {
            return Ok(None);
        }
        // Each wait for the next response is a wait of the caller for the execution.
        let _waiting = self._inflight.waiting();
        // As the reference iterator's `_has_next` retries with a `Retrying` of its own, the
        // reattaches of each response have the attempts and the elapsed time of the policy.
        // The clock of the retries that the reference makes as `RetryException`s, bounded by the
        // policy's elapsed time, starts at the first deadline that passes, the deadline of the
        // initial ExecutePlan included, and the reattaches after a deadline or an unknown
        // execution are its retries.
        let mut retrying = Retrying::new(&self.retry_policy).ending_with_abort();
        if self.stream.is_none() && !retrying.retry_at_once() {
            // The initial ExecutePlan passed its deadline, a RetryException that fails at once
            // when the bound of the retries has passed, as it has when the bound is 0.
            let status = tonic::Status::deadline_exceeded("Deadline Exceeded");
            return Err(self.fail_with_status(status).await);
        }
        loop {
            let Some(stream) = self.stream.as_mut() else {
                self.reattach(&mut retrying).await?;
                continue;
            };
            let deadline = self.stream_deadline;
            let read = async {
                match deadline {
                    Some(deadline) => tokio::time::timeout_at(deadline, stream.message())
                        .await
                        .unwrap_or_else(|_| {
                            Err(tonic::Status::deadline_exceeded("Deadline Exceeded"))
                        }),
                    None => stream.message().await,
                }
            };
            let next = match crate::runtime::until_abort(crate::runtime::ABORT_ALL, read).await {
                Some(next) => next,
                None => return Err(self.fail(wait_aborted())),
            };
            match next {
                Ok(Some(resp)) => {
                    if let Err(error) = self
                        .client
                        .verify_response(&resp.session_id, &resp.server_side_session_id)
                    {
                        return Err(self.fail(error));
                    }
                    self.iter.set_last_response_id(&resp).await;
                    if self.iter.is_completed().await {
                        self.done = true;
                        self.release_all();
                    } else if !resp.response_id.is_empty() {
                        self.release_until(&resp.response_id);
                    }
                    return Ok(Some(resp));
                }
                // The server completes a reattachable stream before `ResultComplete` on
                // purpose, after `senderMaxStreamDuration` or `senderMaxStreamSize`: the rest
                // of the result is read by reattaching.
                Ok(None) => self.reattach(&mut retrying).await?,
                // As the reference iterator: the per-RPC deadline of the stream passed, and the
                // execution goes on on the server, so the iteration reattaches at once, for up
                // to the policy's bound of such retries.
                Err(status)
                    if status.code() == tonic::Code::DeadlineExceeded
                        && !self.iter.is_completed().await =>
                {
                    if !retrying.retry_at_once() {
                        return Err(self.fail_with_status(status).await);
                    }
                    self.stream = None;
                }
                // As the reference iterator: the server does not know the execution (or its
                // session), so the initial ExecutePlan never reached it, and it is sent again,
                // if no response was read yet. Once one was, the reference raises an error.
                Err(status)
                    if unknown_operation(&status).is_some() && !self.iter.is_completed().await =>
                {
                    if self.iter.last_response_id().await.is_some() {
                        let error_type = unknown_operation(&status).unwrap_or_default();
                        return Err(self.fail(response_already_received(error_type)));
                    }
                    if !retrying.retry_at_once() {
                        return Err(self.fail_with_status(status).await);
                    }
                    self.execute_again = true;
                    self.stream = None;
                }
                Err(status) => {
                    // Only reattach a still-running execution on a retriable drop.
                    // As the reference's `_has_next`, which reads the stream and reattaches in
                    // one attempt of its `Retrying`: a stream that fails, also one that a
                    // reattach opened, is an attempt of the policy, which waits its backoff before
                    // reattaching and is the error once the policy's retries are spent. Else a
                    // connection that keeps failing after the headers of each reattach is
                    // reattached at once, without end. As at the other retries, the server's
                    // RetryInfo delay of the failure, if any, is the wait. The backoff ends with a
                    // signal, as the reference's `Retrying` sleep does.
                    if self.iter.is_completed().await {
                        return Err(self.fail_with_status(status).await);
                    }
                    match retrying.backoff(&status).await {
                        Backoff::Retry => {}
                        Backoff::GiveUp => return Err(self.fail_with_status(status).await),
                        Backoff::Aborted => return Err(self.fail(wait_aborted())),
                    }
                    self.reattach(&mut retrying).await?;
                }
            }
        }
    }

    /// Opens the stream of the execution again: by reattaching to it from the last response
    /// read, or, when the server does not know it, by sending the ExecutePlan again. Transient
    /// failures are retried under the policy with their backoff ([`Retrying::backoff`]), and a
    /// deadline or an unknown execution at once, as the reference's `RetryException`s, for up
    /// to their elapsed-time bound ([`Retrying::retry_at_once`]). `retrying` is that of the
    /// response being read, so the failures of its streams and of its reattaches share the
    /// attempts of the policy.
    async fn reattach(&mut self, retrying: &mut Retrying) -> Result<()> {
        loop {
            // As the reference's `_stub`: once the client is closed (its session stopped), the
            // reattach and the ExecutePlan sent again raise NO_ACTIVE_SESSION, and the stream ends.
            if let Err(error) = self.client.ensure_open() {
                return Err(self.fail(error));
            }
            let opened = match crate::runtime::until_abort(crate::runtime::ABORT_RETRIES, async {
                if self.execute_again {
                    self.open_execute_plan_again().await
                } else {
                    self.open_reattach().await
                }
            })
            .await
            {
                Some(opened) => opened,
                None => return Err(self.fail(wait_aborted())),
            };
            let status = match opened {
                Ok(stream) => {
                    self.stream = Some(stream);
                    self.execute_again = false;
                    return Ok(());
                }
                Err(status) => status,
            };
            if let Some(error_type) = unknown_operation(&status) {
                if self.iter.last_response_id().await.is_some() {
                    return Err(self.fail(response_already_received(error_type)));
                }
                if !retrying.retry_at_once() {
                    return Err(self.fail_with_status(status).await);
                }
                self.execute_again = true;
                continue;
            }
            if status.code() == tonic::Code::DeadlineExceeded {
                // As the reference: a call past its deadline, whether it is sent again or
                // reattached, is retried at once, the execution going on on the server.
                if !retrying.retry_at_once() {
                    return Err(self.fail_with_status(status).await);
                }
                self.execute_again = false;
                continue;
            }
            match retrying.backoff(&status).await {
                Backoff::Retry => continue,
                Backoff::GiveUp => return Err(self.fail_with_status(status).await),
                Backoff::Aborted => return Err(self.fail(wait_aborted())),
            }
        }
    }

    /// ReattachExecute from the last response read, bounded by the deadline of the call.
    async fn open_reattach(
        &mut self,
    ) -> std::result::Result<Streaming<ExecutePlanResponse>, tonic::Status> {
        let deadline = self.client.rpc_deadlines.reattach_execute;
        let mut req = Request::new(self.iter.create_reattach_request().await);
        self.attach_metadata(&mut req);
        // After the metadata, which the deadline is a header of.
        let req = with_deadline(req, deadline);
        self.stream_deadline = stream_deadline_in(deadline);
        let response =
            within_deadline(deadline, self.client.stub.clone().reattach_execute(req)).await?;
        Ok(response.into_inner())
    }

    /// ExecutePlan of the request again, as the initial one, bounded by its deadline.
    async fn open_execute_plan_again(
        &mut self,
    ) -> std::result::Result<Streaming<ExecutePlanResponse>, tonic::Status> {
        let deadline = self.client.rpc_deadlines.reattachable_execute_plan;
        let request = clone_on_grown_stack(self.iter.request());
        let mut req = with_deadline(Request::new(request), deadline);
        self.client._attach_metadata(&mut req);
        self.stream_deadline = stream_deadline_in(deadline);
        let response =
            within_deadline(deadline, self.client.stub.clone().execute_plan(req)).await?;
        Ok(response.into_inner())
    }

    /// Ends the stream with the error of `status`, as [`Self::fail`].
    async fn fail_with_status(&mut self, status: tonic::Status) -> SparkError {
        let error = self.client.error_from_status(status).await;
        self.fail(error)
    }

    /// Ends the stream with `error`, releasing the execution as the reference client does
    /// when an execution fails.
    /// The error carries the ExecutePlan operation id (`SparkConnectException.operation_id`).
    fn fail(&mut self, error: SparkError) -> SparkError {
        self.done = true;
        self.release_all();
        error.with_operation_id(self.iter.operation_id())
    }

    /// Releases the responses up to and including `response_id`, which were read.
    fn release_until(&self, response_id: &str) {
        if !self.released {
            self.spawn_release(self.iter.release_until_request(response_id));
        }
    }

    /// Releases the whole execution.
    fn release_all(&mut self) {
        if !self.released {
            self.released = true;
            self.spawn_release(self.iter.release_all_request());
        }
    }

    /// Sends `request` in the background, as [`SparkConnectClient::spawn_release`] does.
    fn spawn_release(&self, request: ReleaseExecuteRequest) {
        self.client
            .spawn_release(self.client.execution_metadata(), request);
    }
}

impl Drop for ReattachableResponseStream {
    /// A stream dropped before its result is complete, for example by a caller that stopped
    /// reading, still holds its execution on the server: release it, as the reference client
    /// does when its iterator is closed.
    fn drop(&mut self) {
        self.release_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_with_busy_blocking_pool(blobs: Vec<Vec<u8>>) -> (bool, Vec<String>) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let (started, ready) = tokio::sync::oneshot::channel();
            let (release, wait) = std::sync::mpsc::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started.send(()).unwrap();
                let _ = wait.recv();
            });
            ready.await.unwrap();

            let allocations: Vec<_> = blobs
                .iter()
                .map(|blob| (blob.as_ptr(), blob.capacity()))
                .collect();
            let mut hashing = std::pin::pin!(hash_artifact_blobs(blobs));
            // Occupying the only blocking thread makes Pending deterministic and
            // catches a regression that hashes large batches inline on the first poll.
            let first_poll = futures::poll!(&mut hashing);
            let pending = first_poll.is_pending();
            // Release the blocking pool before asserting so failures cannot hang shutdown.
            release.send(()).unwrap();
            blocker.await.unwrap();
            let (blobs, hashes) = match first_poll {
                std::task::Poll::Ready(result) => result.unwrap(),
                std::task::Poll::Pending => hashing.await.unwrap(),
            };
            for (blob, allocation) in blobs.iter().zip(allocations) {
                assert_eq!((blob.as_ptr(), blob.capacity()), allocation);
            }
            (pending, hashes)
        })
    }

    #[test]
    fn test_small_artifact_hashing_stays_inline_and_preserves_buffers() {
        // The last digest starts with a zero, which hex encoding must preserve.
        let (pending, hashes) = hash_with_busy_blocking_pool(vec![
            Vec::new(),
            b"abc".to_vec(),
            b"abc".to_vec(),
            b"s".to_vec(),
        ]);
        assert!(
            !pending,
            "small artifact batches must not wait for a blocking thread"
        );
        assert_eq!(
            hashes,
            [
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "043a718774c572bd8a25adbeb1bfcd5c0256ae11cecf9f9c3f925d0e52beaf89",
            ]
        );
    }

    #[test]
    fn test_artifact_hashing_inline_size_boundary() {
        let (pending, hashes) =
            hash_with_busy_blocking_pool(vec![vec![b'a'; ARTIFACT_HASH_INLINE_MAX_BYTES]]);
        assert!(!pending, "the inline size limit is inclusive");
        assert_eq!(
            hashes,
            ["dd3dde87623d9a6b354c68c943d189c89c63652d945e7bbdf0986cae91a49521"]
        );
    }

    #[test]
    fn test_large_artifact_hashing_yields_and_preserves_buffers() {
        // Each blob fits the inline limit, but their combined size exceeds it by one.
        let (pending, hashes) = hash_with_busy_blocking_pool(vec![
            vec![b'a'; ARTIFACT_HASH_INLINE_MAX_BYTES / 2],
            vec![b'a'; ARTIFACT_HASH_INLINE_MAX_BYTES / 2 + 1],
        ]);
        assert!(
            pending,
            "large batches must yield while the blocking pool is busy"
        );
        assert_eq!(
            hashes,
            [
                "b44ffb72fcc259676bd80495fef1b44b808ca8f1ffe1b1706a4d7911b0e31f11",
                "7e009ea4ef882e385b3c0bcbbfa8d009bb0a633bdd764415c09182ee0e75da73",
            ]
        );
    }

    #[test]
    fn test_artifact_hashing_inline_blob_count_boundary() {
        for count in [
            ARTIFACT_HASH_INLINE_MAX_BLOBS,
            ARTIFACT_HASH_INLINE_MAX_BLOBS + 1,
        ] {
            let (pending, hashes) = hash_with_busy_blocking_pool(vec![Vec::new(); count]);
            assert_eq!(
                pending,
                count > ARTIFACT_HASH_INLINE_MAX_BLOBS,
                "blob count: {count}"
            );
            assert_eq!(
                hashes,
                vec!["e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"; count]
            );
        }
    }

    #[tokio::test]
    async fn test_artifact_hashing_resumes_task_panics() {
        let error = tokio::task::spawn_blocking(|| panic!("hashing panic"))
            .await
            .unwrap_err();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            artifact_hashing_task_error(error)
        }))
        .unwrap_err();
        assert_eq!(panic.downcast_ref::<&str>(), Some(&"hashing panic"));
    }

    #[test]
    fn test_artifact_hashing_task_error_reports_queued_task_cancellation() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let (started, ready) = tokio::sync::oneshot::channel();
            let (release, wait) = std::sync::mpsc::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started.send(()).unwrap();
                let _ = wait.recv();
            });
            ready.await.unwrap();

            // Explicit abort models runtime shutdown cancelling queued hashing work;
            // production hashing does not abort its handle.
            let task = tokio::task::spawn_blocking(|| ());
            task.abort();
            // Release the blocking worker before assertions to avoid hanging shutdown.
            release.send(()).unwrap();
            blocker.await.unwrap();

            let cancelled = task.await.unwrap_err();
            assert!(cancelled.is_cancelled());
            let error = artifact_hashing_task_error(cancelled);
            assert!(error.to_string().contains("Artifact hashing task failed:"));
        });
    }

    #[tokio::test]
    async fn test_stream_deadline_too_far_to_be_a_time_is_none() {
        assert_eq!(stream_deadline_in(None), None);
        let before = tokio::time::Instant::now();
        let deadline = stream_deadline_in(Some(Duration::from_secs(600))).expect("a time");
        assert!(deadline >= before + Duration::from_secs(600));
        assert!(deadline <= tokio::time::Instant::now() + Duration::from_secs(600));
        // Adding it to the time would overflow, and panic.
        assert_eq!(stream_deadline_in(Some(Duration::MAX)), None);
    }

    #[test]
    fn test_with_deadline_too_long_for_the_grpc_timeout_is_the_longest_one_sent() {
        let timeout = |request: Request<()>| {
            request
                .metadata()
                .get("grpc-timeout")
                .map(|value| value.to_str().unwrap().to_string())
        };
        assert_eq!(timeout(with_deadline(Request::new(()), None)), None);
        assert_eq!(
            timeout(with_deadline(
                Request::new(()),
                Some(Duration::from_secs(600))
            )),
            Some("600000m".to_string())
        );
        // gRPC encodes at most 99,999,999 hours; tonic panics on a longer timeout, so a longer
        // deadline is sent as that, which no call reaches.
        assert_eq!(
            timeout(with_deadline(Request::new(()), Some(Duration::MAX))),
            Some("99999999H".to_string())
        );
    }

    #[tokio::test]
    async fn test_empty_cache_artifacts_needs_no_server() {
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        // connect() is lazy; the empty upload must return without dialing a server.
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        assert!(client.cache_artifacts(Vec::new()).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_session_id_generation() {
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        let session_id = client.session_id();
        assert!(!session_id.is_empty());
        // Parse to verify it's a valid UUID
        assert!(Uuid::parse_str(session_id).is_ok());
    }

    #[tokio::test]
    async fn test_user_id_passed_through() {
        let builder = ChannelBuilder::parse("sc://localhost/;user_id=test_user").unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        assert_eq!(client.user_id(), Some("test_user"));
    }

    #[tokio::test]
    async fn test_metadata_headers() {
        let builder = ChannelBuilder::parse("sc://localhost/;custom_header=custom_value").unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        assert!(client.user_agent.contains("spark/"));
        // The header of the connection string and the user agent are on a request.
        let mut request = Request::new(());
        client._attach_metadata(&mut request);
        assert_eq!(
            request.metadata().get("custom_header").unwrap(),
            "custom_value"
        );
        assert_eq!(
            request.metadata().get("user-agent").unwrap(),
            client.user_agent.as_str()
        );
    }

    #[test]
    fn test_tls_enabled_channel_builder() {
        // Verify that a ChannelBuilder with use_ssl=true is marked as secure
        let builder = ChannelBuilder::parse("sc://example.com/;use_ssl=true").unwrap();
        assert!(builder.use_ssl());
        assert!(builder.secure());
    }

    #[tokio::test]
    async fn test_token_bearer_in_metadata() {
        // A connection-string token must actually reach the wire as an
        // `Authorization: Bearer` header, not merely be parsed. Build a real request,
        // run it through `_attach_metadata`, and assert on the resulting gRPC metadata
        // so this survives a future change to how the header is stored or inserted.
        let builder =
            ChannelBuilder::parse("sc://localhost/;use_ssl=true;token=my_token_123").unwrap();
        assert!(builder.secure());
        assert_eq!(builder.token(), Some("my_token_123".to_string()));
        let client = SparkConnectClient::connect(&builder).await.unwrap();

        let mut req = Request::new(());
        client._attach_metadata(&mut req);
        assert_eq!(
            req.metadata()
                .get("authorization")
                .map(|v| v.to_str().unwrap()),
            Some("Bearer my_token_123"),
            "expected an `Authorization: Bearer` header on the request, got {:?}",
            req.metadata()
        );
    }

    #[test]
    fn test_running_operation_id_is_the_innermost_execution_the_calling_thread_waits_for() {
        let client = connect_blocking("sc://example.com");
        assert_eq!(client.running_operation_id(), None);
        let registry = || std::sync::Arc::clone(&client.inflight_operations);
        let outer = InflightGuard::register(registry(), "outer".to_string());
        let inner = InflightGuard::register(registry(), "inner".to_string());
        // In flight, but nothing waits for them: no execution is interrupted.
        assert_eq!(client.running_operation_id(), None);
        let wait_outer = outer.waiting();
        assert_eq!(client.running_operation_id().as_deref(), Some("outer"));
        let wait_inner = inner.waiting();
        assert_eq!(client.running_operation_id().as_deref(), Some("inner"));
        drop(wait_inner);
        assert_eq!(client.running_operation_id().as_deref(), Some("outer"));
        drop(wait_outer);
        assert_eq!(client.running_operation_id(), None);
        // The waits of another thread are not those of the calling thread.
        let (waiting_tx, waiting_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let other = std::thread::spawn({
            let registry = registry();
            move || {
                let guard = InflightGuard::register(registry, "other".to_string());
                let _wait = guard.waiting();
                waiting_tx.send(()).unwrap();
                done_rx.recv().unwrap();
            }
        });
        waiting_rx.recv().unwrap();
        assert_eq!(client.running_operation_id(), None);
        done_tx.send(()).unwrap();
        other.join().unwrap();
        // Once the registration of the execution drops, it is not in flight any more.
        let wait_inner = inner.waiting();
        drop(inner);
        assert_eq!(client.running_operation_id(), None);
        drop(wait_inner);
        drop(outer);
        assert_eq!(client.running_operation_id(), None);
    }

    #[test]
    fn test_an_inflight_handle_marks_only_the_waits_of_the_thread_that_started_it() {
        // As a prefetching iterator: the execution is read by a task of its own, and its caller
        // marks its wait for the batches through the handle.
        let client = connect_blocking("sc://example.com");
        let guard = InflightGuard::register(
            std::sync::Arc::clone(&client.inflight_operations),
            "prefetch".to_string(),
        );
        let handle = guard.handle();
        // Not read: the execution is in flight, but no wait of the caller is for it.
        assert_eq!(client.running_operation_id(), None);
        {
            let _waiting = handle.waiting();
            assert_eq!(client.running_operation_id().as_deref(), Some("prefetch"));
        }
        assert_eq!(client.running_operation_id(), None);
        // The reads of a task on another thread do not mark the execution for the caller: the
        // task's wait is not the caller's.
        let (waiting_tx, waiting_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let task = std::thread::spawn({
            let handle = handle.clone();
            move || {
                let _waiting = handle.waiting();
                waiting_tx.send(()).unwrap();
                done_rx.recv().unwrap();
            }
        });
        waiting_rx.recv().unwrap();
        assert_eq!(client.running_operation_id(), None);
        done_tx.send(()).unwrap();
        task.join().unwrap();
        drop(guard);
        assert_eq!(client.running_operation_id(), None);
    }

    /// Connects a client on a runtime of its own, for the tests that set the environment.
    /// The channel is lazy, so no server is dialed.
    fn connect_blocking(url: &str) -> SparkConnectClient {
        let builder = ChannelBuilder::parse(url).unwrap();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime
            .block_on(SparkConnectClient::connect(&builder))
            .unwrap_or_else(|e| panic!("{url} should connect, got: {e}"))
    }

    fn has_bearer(client: &SparkConnectClient, token: &str) -> bool {
        let bearer = format!("Bearer {token}");
        client
            .metadata
            .iter()
            .any(|(k, v)| k == "authorization" && *v == bearer)
    }

    #[test]
    fn test_token_without_ssl_to_remote_host_is_sent_over_tls() {
        // A token over a remote host is not refused: its channel is TLS, which the
        // reference client also uses for a token on any host but localhost.
        let _env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        for url in [
            "sc://example.com/;token=SECRET",
            "sc://10.0.0.1:443/;token=SECRET",
        ] {
            assert!(ChannelBuilder::parse(url).unwrap().secure(), "{url}");
            let client = connect_blocking(url);
            assert!(
                has_bearer(&client, "SECRET"),
                "{url}: {:?}",
                client.metadata
            );
        }
    }

    #[test]
    fn test_token_of_the_environment_to_remote_host_is_sent_over_tls() {
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", Some("ENV"))]);
        assert!(ChannelBuilder::parse("sc://example.com").unwrap().secure());
        let client = connect_blocking("sc://example.com");
        assert!(has_bearer(&client, "ENV"), "{:?}", client.metadata);
        drop(env);
        // Without a token, nothing is attached.
        let _none = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let client = connect_blocking("sc://example.com");
        assert!(!has_bearer(&client, "ENV"), "{:?}", client.metadata);
    }

    #[test]
    fn test_user_id_from_the_environment_as_pyspark() {
        let some = |s: &str| Ok::<String, std::env::VarError>(s.to_string());
        let unset = || Err::<String, std::env::VarError>(std::env::VarError::NotPresent);
        assert_eq!(
            user_id_from_env(some("alice"), some("bob")),
            Some("alice".into())
        );
        assert_eq!(user_id_from_env(unset(), some("bob")), Some("bob".into()));
        // A `SPARK_USER` that is set to an empty string is the user ID, so `USER` is not read.
        assert_eq!(user_id_from_env(some(""), some("bob")), None);
        assert_eq!(user_id_from_env(unset(), some("")), None);
        assert_eq!(user_id_from_env(unset(), unset()), None);
    }

    #[test]
    fn test_user_id_of_the_connection_then_the_environment() {
        // The connection string's user_id, else SPARK_USER, else USER, as PySpark does.
        let user_id = |url: &str, spark_user: Option<&str>, user: Option<&str>| {
            let _env = crate::channel::set_env(&[("SPARK_USER", spark_user), ("USER", user)]);
            let client = connect_blocking(url);
            let id = client.user_id().map(String::from);
            id
        };
        assert_eq!(
            user_id("sc://localhost/;user_id=url", Some("alice"), Some("bob")).as_deref(),
            Some("url")
        );
        assert_eq!(
            user_id("sc://localhost", Some("alice"), Some("bob")).as_deref(),
            Some("alice")
        );
        assert_eq!(
            user_id("sc://localhost", None, Some("bob")).as_deref(),
            Some("bob")
        );
        assert_eq!(user_id("sc://localhost", None, None), None);
    }

    #[tokio::test]
    async fn test_token_without_ssl_to_localhost_is_allowed() {
        // Local development mirrors upstream's `local_channel_credentials()` exception:
        // a token over plaintext `localhost` is permitted and the header is attached.
        for url in [
            "sc://localhost/;token=SECRET",
            "sc://LOCALHOST/;token=SECRET",
        ] {
            let builder = ChannelBuilder::parse(url).unwrap();
            assert!(!builder.secure(), "{url}");
            let client = SparkConnectClient::connect(&builder)
                .await
                .unwrap_or_else(|e| panic!("{url} should be allowed, got: {e}"));
            assert!(
                client
                    .metadata
                    .iter()
                    .any(|(k, v)| k == "authorization" && v == "Bearer SECRET"),
                "{url}: expected the bearer header to be attached, got {:?}",
                client.metadata
            );
        }
    }

    #[tokio::test]
    async fn test_token_to_loopback_ip_literals_is_sent_over_tls() {
        // Only `localhost` is exempt: the IPv4 and IPv6 loopback literals are TLS with a
        // token, as in PySpark, so the channel is https and the header is still attached.
        for url in ["sc://127.0.0.1/;token=SECRET", "sc://[::1]/;token=SECRET"] {
            let builder = ChannelBuilder::parse(url).unwrap();
            assert!(builder.secure(), "{url}");
            let client = SparkConnectClient::connect(&builder).await.unwrap();
            assert!(
                has_bearer(&client, "SECRET"),
                "{url}: {:?}",
                client.metadata
            );
        }
    }

    /// A loopback service of ReleaseSession only: its first `unavailable` calls answer
    /// UNAVAILABLE, the later ones succeed. `calls` counts every call.
    struct ReleaseCalls {
        calls: std::sync::atomic::AtomicUsize,
        unavailable: usize,
    }

    #[derive(Clone)]
    struct ReleaseService(std::sync::Arc<ReleaseCalls>);

    impl tonic::server::NamedService for ReleaseService {
        const NAME: &'static str = "spark.connect.SparkConnectService";
    }

    impl tonic::codegen::Service<tonic::codegen::http::Request<tonic::body::Body>> for ReleaseService {
        type Response = tonic::codegen::http::Response<tonic::body::Body>;
        type Error = std::convert::Infallible;
        type Future = tonic::codegen::BoxFuture<Self::Response, Self::Error>;

        fn poll_ready(
            &mut self,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<std::result::Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn call(
            &mut self,
            request: tonic::codegen::http::Request<tonic::body::Body>,
        ) -> Self::Future {
            let state = self.0.clone();
            Box::pin(async move {
                if request.uri().path() == "/spark.connect.SparkConnectService/ReleaseSession" {
                    let mut grpc = tonic::server::Grpc::new(tonic_prost::ProstCodec::default());
                    return Ok(grpc.unary(ReleaseRpc(state), request).await);
                }
                Ok(tonic::codegen::http::Response::builder()
                    .status(200)
                    .header("grpc-status", "12")
                    .header("content-type", "application/grpc")
                    .body(tonic::body::Body::empty())
                    .unwrap())
            })
        }
    }

    struct ReleaseRpc(std::sync::Arc<ReleaseCalls>);

    impl tonic::server::UnaryService<ReleaseSessionRequest> for ReleaseRpc {
        type Response = spark_connect_proto::ReleaseSessionResponse;
        type Future = tonic::codegen::BoxFuture<tonic::Response<Self::Response>, tonic::Status>;

        fn call(&mut self, request: tonic::Request<ReleaseSessionRequest>) -> Self::Future {
            let state = self.0.clone();
            let request = request.into_inner();
            Box::pin(async move {
                let call = state
                    .calls
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if call < state.unavailable {
                    return Err(tonic::Status::unavailable("the server is restarting"));
                }
                Ok(tonic::Response::new(
                    spark_connect_proto::ReleaseSessionResponse {
                        session_id: request.session_id,
                        ..Default::default()
                    },
                ))
            })
        }
    }

    #[tokio::test]
    async fn release_session_is_retried_on_unavailable_as_the_reference() {
        // A transient UNAVAILABLE at stop is retried under the policy, so the server session is
        // released by the second call; the reference's `release_session` runs in `Retrying`.
        let calls = std::sync::Arc::new(ReleaseCalls {
            calls: std::sync::atomic::AtomicUsize::new(0),
            unavailable: 1,
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let service = ReleaseService(calls.clone());
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        let builder = {
            let _env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
            ChannelBuilder::parse(&format!("sc://127.0.0.1:{}", address.port())).unwrap()
        };
        let mut client = SparkConnectClient::connect(&builder).await.unwrap();
        client.set_retry_policy(RetryPolicy {
            max_retries: Some(2),
            initial_backoff_ms: 1,
            max_backoff_ms: Some(2),
            ..RetryPolicy::default()
        });

        let released = client.release_session().await;
        server.abort();
        released.unwrap();
        assert_eq!(calls.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_add_artifacts_request_structure() {
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();

        // Test that add_artifacts can be called without panicking
        let _result = client.add_artifacts(&[], false, false, false).await;
        assert!(_result.is_ok());
    }

    #[tokio::test]
    async fn test_get_config_with_defaults_request_structure() {
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();

        // Test that get_config_with_defaults returns correctly typed Vec
        let pairs = vec![
            ("spark.sql.shuffle.partitions", Some("200")),
            ("spark.sql.adaptive.enabled", None),
        ];
        let _result = client.get_config_with_defaults(&pairs);
    }

    #[tokio::test]
    async fn threadlocal_user_context_extensions_of_finished_threads_are_forgotten() {
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = std::sync::Arc::new(SparkConnectClient::connect(&builder).await.unwrap());
        for _ in 0..1000 {
            let client = client.clone();
            std::thread::spawn(move || {
                client.add_threadlocal_user_context_extension(prost_types::Any::default());
            })
            .join()
            .unwrap();
        }
        let entries = client
            .user_context_extensions
            .lock()
            .unwrap()
            .by_thread
            .len();
        assert_eq!(entries, 0);
    }

    #[tokio::test]
    async fn threadlocal_user_context_extensions_of_a_living_thread_stay_until_it_ends() {
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = std::sync::Arc::new(SparkConnectClient::connect(&builder).await.unwrap());
        let (added_tx, added_rx) = std::sync::mpsc::channel();
        let (end_tx, end_rx) = std::sync::mpsc::channel::<()>();
        let child = {
            let client = client.clone();
            std::thread::spawn(move || {
                client.add_threadlocal_user_context_extension(prost_types::Any::default());
                added_tx
                    .send(client.threadlocal_user_context_extensions().len())
                    .unwrap();
                end_rx.recv().unwrap();
            })
        };
        assert_eq!(added_rx.recv().unwrap(), 1);
        assert_eq!(
            client
                .user_context_extensions
                .lock()
                .unwrap()
                .by_thread
                .len(),
            1
        );
        end_tx.send(()).unwrap();
        child.join().unwrap();
        assert!(client
            .user_context_extensions
            .lock()
            .unwrap()
            .by_thread
            .is_empty());
    }

    fn any_of(type_url: &str) -> prost_types::Any {
        prost_types::Any {
            type_url: type_url.to_string(),
            value: Vec::new(),
        }
    }

    fn extension_ids(client: &SparkConnectClient) -> Vec<String> {
        client
            .threadlocal_user_context_extensions()
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    #[tokio::test]
    async fn threadlocal_user_context_extensions_of_a_copy_are_shared_as_pyspark() {
        // As PySpark's inheritable_thread_target: the threads given a copy of the extensions of a
        // thread share it, an extension that one of them adds is in the copy of all of them, and
        // one that it removes is removed for it alone. An extension added later is not in the copy.
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = std::sync::Arc::new(SparkConnectClient::connect(&builder).await.unwrap());
        let main = client.add_threadlocal_user_context_extension(any_of("main"));
        let copy = client.copy_threadlocal_user_context_extensions();
        let late = client.add_threadlocal_user_context_extension(any_of("late"));
        let run = |work: fn(&SparkConnectClient) -> Vec<Vec<String>>| {
            let client = client.clone();
            let copy = copy.clone();
            std::thread::spawn(move || {
                client.share_threadlocal_user_context_extensions(&copy);
                work(&client)
            })
            .join()
            .unwrap()
        };
        let first = run(|client| {
            let mut seen = vec![extension_ids(client)];
            client.add_threadlocal_user_context_extension(any_of("added"));
            seen.push(extension_ids(client));
            let main_id = seen[0][0].clone();
            client.remove_user_context_extension(&main_id);
            seen.push(extension_ids(client));
            seen
        });
        assert_eq!(first[0], vec![main.clone()]);
        assert_eq!(first[1].len(), 2);
        assert_eq!(first[1][0], main);
        assert_eq!(first[2].len(), 1);
        assert_ne!(first[2][0], main);
        assert_eq!(
            run(|client| vec![extension_ids(client)]),
            vec![first[1].clone()]
        );
        assert_eq!(
            run(|client| vec![extension_ids(client)]),
            vec![first[1].clone()]
        );
        assert_eq!(extension_ids(&client), vec![main, late]);
    }

    #[test]
    fn host_and_observed_server_session_id_of_the_client() {
        let client = connect_blocking("sc://Other:114");
        assert_eq!(client.host(), "other");
        assert_eq!(client.observed_server_session_id(), None);
        client.set_observed_server_session_id(Some("server-a".to_string()));
        assert_eq!(
            client.observed_server_session_id().as_deref(),
            Some("server-a")
        );
        // The clones of a client share its observed ID, and a client of a new session has none.
        assert_eq!(
            client.clone().observed_server_session_id().as_deref(),
            Some("server-a")
        );
        let other = client.with_new_session_id();
        assert_eq!(other.host(), "other");
        assert_eq!(other.observed_server_session_id(), None);
    }
}

#[cfg(test)]
mod plan_compression_tests {
    use super::*;
    use spark_connect_proto::analyze_plan_request::{SameSemantics, Schema, SparkVersion};
    use spark_connect_proto::command::CommandType;
    use spark_connect_proto::relation::RelType;
    use spark_connect_proto::{Command, LocalRelation, Relation};

    const PAYLOAD: usize = 64 * 1024;

    fn relation_with(data: Vec<u8>) -> Relation {
        Relation {
            rel_type: Some(RelType::LocalRelation(LocalRelation {
                data: Some(data.into()),
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    fn root(relation: &Relation) -> Plan {
        Plan {
            op_type: Some(PlanOpType::Root(relation.clone())),
        }
    }

    /// Bytes of a xorshift sequence, which zstd does not compress.
    fn incompressible(len: usize) -> Vec<u8> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    #[test]
    fn plan_compression_is_zstd_above_the_threshold_of_the_confs() {
        assert_eq!(
            plan_compression_of(Some("10485760"), Some("ZSTD")),
            PlanCompression::Zstd {
                threshold: 10485760
            }
        );
        assert_eq!(
            plan_compression_of(Some("0"), Some("ZSTD")),
            PlanCompression::Zstd { threshold: 0 }
        );
        for (threshold, algorithm) in [
            (Some("-1"), Some("ZSTD")),
            (None, Some("ZSTD")),
            (Some(""), Some("ZSTD")),
            (Some("ten"), Some("ZSTD")),
            (Some("10"), Some("NONE")),
            (Some("10"), Some("LZ4")),
            (Some("10"), None),
        ] {
            assert_eq!(
                plan_compression_of(threshold, algorithm),
                PlanCompression::Disabled,
                "threshold {threshold:?}, algorithm {algorithm:?}"
            );
        }
    }

    #[test]
    fn a_relation_over_the_threshold_is_sent_as_a_zstd_relation_operation() {
        let relation = relation_with(vec![7; PAYLOAD]);
        let encoded = relation.encode_to_vec();
        let mut plan = root(&relation);
        compress_over_threshold(&mut plan, encoded.len() - 1);
        let Some(PlanOpType::CompressedOperation(operation)) = plan.op_type else {
            panic!("the plan over the threshold was not compressed");
        };
        assert_eq!(operation.op_type, CompressedOpType::Relation as i32);
        assert_eq!(operation.compression_codec, CompressionCodec::Zstd as i32);
        assert_eq!(zstd::decode_all(&operation.data[..]).unwrap(), encoded);
    }

    #[test]
    fn a_command_over_the_threshold_is_sent_as_a_zstd_command_operation() {
        let command = Command {
            command_type: Some(CommandType::Extension(prost_types::Any {
                type_url: "example.org/command".into(),
                value: vec![7; PAYLOAD],
            })),
        };
        let encoded = command.encode_to_vec();
        let mut plan = Plan {
            op_type: Some(PlanOpType::Command(command)),
        };
        compress_over_threshold(&mut plan, encoded.len() / 2);
        let Some(PlanOpType::CompressedOperation(operation)) = plan.op_type else {
            panic!("the command over the threshold was not compressed");
        };
        assert_eq!(operation.op_type, CompressedOpType::Command as i32);
        assert_eq!(zstd::decode_all(&operation.data[..]).unwrap(), encoded);
    }

    #[test]
    fn a_plan_is_sent_as_it_is_unless_it_is_over_the_threshold_and_compression_shrinks_it() {
        let relation = relation_with(vec![7; PAYLOAD]);
        let size = relation.encode_to_vec().len();
        let mut at_threshold = root(&relation);
        compress_over_threshold(&mut at_threshold, size);
        assert_eq!(at_threshold, root(&relation));

        let random = relation_with(incompressible(PAYLOAD));
        let mut not_smaller = root(&random);
        compress_over_threshold(&mut not_smaller, 0);
        assert_eq!(not_smaller, root(&random));
    }

    #[test]
    fn the_plans_of_an_analysis_are_those_pyspark_compresses() {
        let relation = relation_with(vec![7; 1]);
        let mut schema = Analyze::Schema(Schema {
            plan: Some(root(&relation)),
        });
        assert_eq!(analyze_plans(&mut schema).len(), 1);

        let mut same_semantics = Analyze::SameSemantics(SameSemantics {
            target_plan: Some(root(&relation)),
            other_plan: Some(root(&relation)),
        });
        assert_eq!(analyze_plans(&mut same_semantics).len(), 2);

        assert!(analyze_plans(&mut Analyze::SparkVersion(SparkVersion {})).is_empty());
    }
}

/// The fields of the requests of each RPC, which a server records.
#[cfg(test)]
mod request_fields_tests {
    use super::*;
    use crate::bytes_codec::BytesCodec;
    use spark_connect_proto::execute_plan_response::{ResponseType, ResultComplete};
    use tonic::codegen::{http, BoxFuture};
    use tonic::server::{
        ClientStreamingService, NamedService, ServerStreamingService, UnaryService,
    };

    const SERVER_ID: &str = "server-1";

    /// The method and the bytes of each request that the server received, in order.
    type Received = std::sync::Arc<std::sync::Mutex<Vec<(String, Vec<u8>)>>>;

    #[derive(Clone)]
    struct Recorder(Received);

    impl NamedService for Recorder {
        const NAME: &'static str = "spark.connect.SparkConnectService";
    }

    impl tonic::codegen::Service<http::Request<tonic::body::Body>> for Recorder {
        type Response = http::Response<tonic::body::Body>;
        type Error = std::convert::Infallible;
        type Future = BoxFuture<Self::Response, Self::Error>;

        fn poll_ready(
            &mut self,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<std::result::Result<(), Self::Error>> {
            std::task::Poll::Ready(Ok(()))
        }

        fn call(&mut self, request: http::Request<tonic::body::Body>) -> Self::Future {
            let method = request
                .uri()
                .path()
                .trim_start_matches("/spark.connect.SparkConnectService/")
                .to_string();
            let rpc = Rpc(self.0.clone(), method.clone());
            Box::pin(async move {
                let mut grpc = tonic::server::Grpc::new(BytesCodec);
                Ok(match method.as_str() {
                    "ExecutePlan" => grpc.server_streaming(rpc, request).await,
                    "AddArtifacts" => grpc.client_streaming(rpc, request).await,
                    _ => grpc.unary(rpc, request).await,
                })
            })
        }
    }

    /// The response of `method`, of the server-side session `SERVER_ID`.
    fn response_of(method: &str) -> Vec<u8> {
        let id = SERVER_ID.to_string();
        match method {
            "ExecutePlan" => ExecutePlanResponse {
                server_side_session_id: id,
                response_id: "complete".into(),
                response_type: Some(ResponseType::ResultComplete(ResultComplete {})),
                ..Default::default()
            }
            .encode_to_vec(),
            "AnalyzePlan" => AnalyzePlanResponse {
                server_side_session_id: id,
                ..Default::default()
            }
            .encode_to_vec(),
            "Config" => ConfigResponse {
                server_side_session_id: id,
                ..Default::default()
            }
            .encode_to_vec(),
            "Interrupt" => InterruptResponse {
                server_side_session_id: id,
                ..Default::default()
            }
            .encode_to_vec(),
            "FetchErrorDetails" => FetchErrorDetailsResponse {
                server_side_session_id: id,
                ..Default::default()
            }
            .encode_to_vec(),
            "ReleaseExecute" => ReleaseExecuteResponse {
                server_side_session_id: id,
                ..Default::default()
            }
            .encode_to_vec(),
            "ReleaseSession" => spark_connect_proto::ReleaseSessionResponse {
                server_side_session_id: id,
                ..Default::default()
            }
            .encode_to_vec(),
            "CloneSession" => spark_connect_proto::CloneSessionResponse {
                server_side_session_id: id,
                ..Default::default()
            }
            .encode_to_vec(),
            "ArtifactStatus" => spark_connect_proto::ArtifactStatusesResponse {
                server_side_session_id: id,
                ..Default::default()
            }
            .encode_to_vec(),
            "AddArtifacts" => spark_connect_proto::AddArtifactsResponse {
                server_side_session_id: id,
                ..Default::default()
            }
            .encode_to_vec(),
            other => panic!("unexpected method {other}"),
        }
    }

    /// An RPC of the recorder, of a method.
    struct Rpc(Received, String);

    impl Rpc {
        fn record(&self, bytes: Vec<u8>) {
            self.0.lock().unwrap().push((self.1.clone(), bytes));
        }
    }

    impl UnaryService<Vec<u8>> for Rpc {
        type Response = Vec<u8>;
        type Future = BoxFuture<tonic::Response<Vec<u8>>, tonic::Status>;

        fn call(&mut self, request: tonic::Request<Vec<u8>>) -> Self::Future {
            self.record(request.into_inner());
            let response = response_of(&self.1);
            Box::pin(async move { Ok(tonic::Response::new(response)) })
        }
    }

    type Frames =
        futures::stream::Iter<std::vec::IntoIter<std::result::Result<Vec<u8>, tonic::Status>>>;

    impl ServerStreamingService<Vec<u8>> for Rpc {
        type Response = Vec<u8>;
        type ResponseStream = Frames;
        type Future = BoxFuture<tonic::Response<Frames>, tonic::Status>;

        fn call(&mut self, request: tonic::Request<Vec<u8>>) -> Self::Future {
            self.record(request.into_inner());
            let frames = futures::stream::iter(vec![Ok(response_of(&self.1))]);
            Box::pin(async move { Ok(tonic::Response::new(frames)) })
        }
    }

    impl ClientStreamingService<Vec<u8>> for Rpc {
        type Response = Vec<u8>;
        type Future = BoxFuture<tonic::Response<Vec<u8>>, tonic::Status>;

        fn call(&mut self, request: tonic::Request<Streaming<Vec<u8>>>) -> Self::Future {
            let rpc = Rpc(self.0.clone(), self.1.clone());
            Box::pin(async move {
                let mut frames = request.into_inner();
                while let Some(frame) = frames.message().await? {
                    rpc.record(frame);
                }
                Ok(tonic::Response::new(response_of(&rpc.1)))
            })
        }
    }

    /// The client type, the observed server-side session ID and the user context extensions of
    /// a request of `method`.
    fn fields_of(
        method: &str,
        bytes: &[u8],
    ) -> (Option<String>, Option<String>, Vec<prost_types::Any>) {
        macro_rules! fields {
            ($request:ty) => {{
                let request = <$request>::decode(bytes).unwrap();
                (
                    request.client_type,
                    request.client_observed_server_side_session_id,
                    request.user_context.unwrap_or_default().extensions,
                )
            }};
        }
        match method {
            "ExecutePlan" => fields!(ExecutePlanRequest),
            "AnalyzePlan" => fields!(AnalyzePlanRequest),
            "Config" => fields!(ConfigRequest),
            "Interrupt" => fields!(InterruptRequest),
            "FetchErrorDetails" => fields!(FetchErrorDetailsRequest),
            "ReleaseExecute" => fields!(ReleaseExecuteRequest),
            "CloneSession" => fields!(CloneSessionRequest),
            "ArtifactStatus" => fields!(ArtifactStatusesRequest),
            "AddArtifacts" => fields!(spark_connect_proto::AddArtifactsRequest),
            // Of no observed server-side session ID.
            "ReleaseSession" => {
                let request = ReleaseSessionRequest::decode(bytes).unwrap();
                (
                    request.client_type,
                    None,
                    request.user_context.unwrap_or_default().extensions,
                )
            }
            other => panic!("unexpected method {other}"),
        }
    }

    /// The status of an error that the server raised and keeps the details of under
    /// `error_id`: a `google.rpc.Status` of an `ErrorInfo` of that `errorId`.
    fn error_kept_under(error_id: &str) -> tonic::Status {
        #[derive(Clone, PartialEq, prost::Message)]
        struct ErrorInfo {
            #[prost(map = "string, string", tag = "3")]
            metadata: std::collections::HashMap<String, String>,
        }
        #[derive(Clone, PartialEq, prost::Message)]
        struct RpcStatus {
            #[prost(message, repeated, tag = "3")]
            details: Vec<prost_types::Any>,
        }
        let info = ErrorInfo {
            metadata: [("errorId".to_string(), error_id.to_string())].into(),
        };
        let status = RpcStatus {
            details: vec![prost_types::Any {
                type_url: "type.googleapis.com/google.rpc.ErrorInfo".into(),
                value: info.encode_to_vec(),
            }],
        };
        tonic::Status::with_details(tonic::Code::Internal, "boom", status.encode_to_vec().into())
    }

    /// As PySpark's hook, which would put in the plan of `to_proto`, which compresses it: the plan
    /// that a hook puts in is compressed, and one that the hook keeps compressed is kept.
    #[tokio::test]
    async fn the_plan_that_a_hook_puts_in_is_compressed_as_pyspark() {
        let received: Received = Default::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let service = Recorder(received.clone());
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        let builder = {
            let _env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
            ChannelBuilder::parse(&format!("sc://127.0.0.1:{}/", address.port())).unwrap()
        };
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        *client.plan_compression.lock().unwrap() = PlanCompression::Zstd { threshold: 1000 };
        let large = spark_connect_proto::Relation {
            rel_type: Some(spark_connect_proto::relation::RelType::LocalRelation(
                spark_connect_proto::LocalRelation {
                    data: Some(vec![0; 64 * 1024].into()),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };
        client.set_execute_plan_hook(Some(std::sync::Arc::new(move |mut request| {
            request.plan = Some(Plan {
                op_type: Some(PlanOpType::Root(large.clone())),
            });
            Ok(request)
        })));
        let small = || ExecutePlanRequest {
            session_id: client.session_id().to_string(),
            plan: Some(Plan {
                op_type: Some(PlanOpType::Root(Default::default())),
            }),
            ..Default::default()
        };
        let mut stream = client.execute_plan(small()).await.unwrap();
        while stream.message().await.unwrap().is_some() {}
        let mut stream = client.execute_plan_reattachable(small()).await.unwrap();
        while stream.message().await.unwrap().is_some() {}
        let executions: Vec<_> = received
            .lock()
            .unwrap()
            .iter()
            .filter(|(method, _)| method == "ExecutePlan")
            .map(|(_, bytes)| ExecutePlanRequest::decode(bytes.as_slice()).unwrap())
            .collect();
        assert_eq!(executions.len(), 2);
        for request in executions {
            let op_type = request.plan.and_then(|plan| plan.op_type);
            assert!(
                matches!(op_type, Some(PlanOpType::CompressedOperation(_))),
                "{op_type:?}"
            );
        }
    }

    #[tokio::test]
    async fn requests_carry_the_client_type_extensions_and_observed_session_of_pyspark() {
        let received: Received = Default::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let service = Recorder(received.clone());
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        let builder = {
            let _env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
            ChannelBuilder::parse(&format!("sc://127.0.0.1:{}/;user_id=u1", address.port()))
                .unwrap()
        };
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        let session_id = client.session_id().to_string();
        let extension = prost_types::Any {
            type_url: "example.org/context".into(),
            value: vec![1, 2, 3],
        };
        client.add_global_user_context_extension(extension.clone());

        // The first request reads the plan compression of the server for the plan of the
        // execution, and its response gives the server-side session ID, which the execution
        // observes, as PySpark builds the plan before the request.
        let mut stream = client
            .execute_plan_reattachable(ExecutePlanRequest {
                session_id: session_id.clone(),
                plan: Some(Plan {
                    op_type: Some(PlanOpType::Root(Default::default())),
                }),
                ..Default::default()
            })
            .await
            .unwrap();
        while stream.message().await.unwrap().is_some() {}
        client
            .config(ConfigRequest {
                session_id: session_id.clone(),
                ..Default::default()
            })
            .await
            .unwrap();
        client
            .analyze_plan(AnalyzePlanRequest {
                session_id: session_id.clone(),
                ..Default::default()
            })
            .await
            .unwrap();
        client
            .interrupt(InterruptRequest {
                session_id: session_id.clone(),
                ..Default::default()
            })
            .await
            .unwrap();
        client
            .fetch_error_details(FetchErrorDetailsRequest {
                session_id: session_id.clone(),
                ..Default::default()
            })
            .await
            .unwrap();
        // An error that the server keeps the details of: raising it fetches them, and reads the
        // conf of the server's stack trace.
        client.error_from_status(error_kept_under("error-1")).await;
        client.artifact_status(&["files/a.txt"]).await.unwrap();
        let file = tempfile::NamedTempFile::new().unwrap();
        client
            .add_named_artifact("files/a.txt", file.path().to_str().unwrap())
            .await
            .unwrap();
        client.clone_session(None).await.unwrap();
        client.release_session().await.unwrap();
        // The release of the execution is sent in the background.
        for _ in 0..200 {
            if received
                .lock()
                .unwrap()
                .iter()
                .any(|(m, _)| m == "ReleaseExecute")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        server.abort();

        let requests = received.lock().unwrap().clone();
        let user_agent = client.user_agent.clone();
        for method in [
            "Config",
            "ExecutePlan",
            "ReleaseExecute",
            "AnalyzePlan",
            "Interrupt",
            "FetchErrorDetails",
            "ArtifactStatus",
            "AddArtifacts",
            "CloneSession",
            "ReleaseSession",
        ] {
            assert!(requests.iter().any(|(m, _)| m == method), "no {method}");
        }
        // The lookup, the explicit Config and the stack trace conf of the error; the explicit
        // FetchErrorDetails and that of the error.
        let count = |method: &str| requests.iter().filter(|(m, _)| m == method).count();
        assert_eq!((count("Config"), count("FetchErrorDetails")), (3, 2));
        for (index, (method, bytes)) in requests.iter().enumerate() {
            // As PySpark: the user agent, but "python" for CloneSession and none for the requests
            // of the ArtifactManager; the observed server-side session ID, once a response gave
            // it, but on ReleaseExecute, ReleaseSession, CloneSession and the artifact requests;
            // and the user context extensions on the requests of
            // `_update_request_with_user_context_extensions`, those that an error sends too, and
            // on the ReleaseExecute of the user context of the execution.
            let client_type = match method.as_str() {
                "CloneSession" => Some("python".to_string()),
                "ArtifactStatus" | "AddArtifacts" => None,
                _ => Some(user_agent.clone()),
            };
            let observed = match method.as_str() {
                _ if index == 0 => None,
                "ReleaseExecute" | "ReleaseSession" | "CloneSession" | "ArtifactStatus"
                | "AddArtifacts" => None,
                _ => Some(SERVER_ID.to_string()),
            };
            let extensions = match method.as_str() {
                "ReleaseSession" | "CloneSession" | "ArtifactStatus" | "AddArtifacts" => vec![],
                _ => vec![extension.clone()],
            };
            assert_eq!(
                fields_of(method, bytes),
                (client_type, observed, extensions),
                "{method}"
            );
        }
    }

    /// A client of the user `u1` of a recording server, what that server received, and the
    /// server.
    async fn recorded_client() -> (SparkConnectClient, Received, tokio::task::JoinHandle<()>) {
        let received: Received = Default::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let service = Recorder(received.clone());
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        let builder = {
            let _env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
            ChannelBuilder::parse(&format!("sc://127.0.0.1:{}/;user_id=u1", address.port()))
                .unwrap()
        };
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        (client, received, server)
    }

    /// The only request that the server received, which is of `method`.
    fn only_request<T: Message + Default>(received: &Received, method: &str) -> T {
        let requests = received.lock().unwrap();
        let methods: Vec<&str> = requests.iter().map(|(m, _)| m.as_str()).collect();
        assert_eq!(methods, [method]);
        T::decode(requests[0].1.as_slice()).unwrap()
    }

    /// The operation of the only request that the server received, a Config of the session and
    /// the user of `client`.
    fn config_operation(
        client: &SparkConnectClient,
        received: &Received,
    ) -> spark_connect_proto::config_request::operation::OpType {
        let request: ConfigRequest = only_request(received, "Config");
        assert_eq!(request.session_id, client.session_id());
        assert_eq!(request.user_context.unwrap().user_id, "u1");
        request.operation.unwrap().op_type.unwrap()
    }

    /// The interrupt type and the interrupt of the only request that the server received, an
    /// Interrupt of the session and the user of `client`.
    fn interruption(
        client: &SparkConnectClient,
        received: &Received,
    ) -> (
        i32,
        Option<spark_connect_proto::interrupt_request::Interrupt>,
    ) {
        let request: InterruptRequest = only_request(received, "Interrupt");
        assert_eq!(request.session_id, client.session_id());
        assert_eq!(request.user_context.unwrap().user_id, "u1");
        (request.interrupt_type, request.interrupt)
    }

    #[tokio::test]
    async fn test_get_configs_request_structure() {
        use spark_connect_proto::config_request::{operation::OpType, Get};
        let (client, received, server) = recorded_client().await;
        let keys = ["spark.sql.shuffle.partitions", "spark.sql.ansi.enabled"];
        // The server answers no pair, so each key is unset.
        assert_eq!(client.get_configs(&keys).await.unwrap(), [None, None]);
        server.abort();
        assert_eq!(
            config_operation(&client, &received),
            OpType::Get(Get {
                keys: keys.map(str::to_string).to_vec(),
            })
        );
    }

    #[tokio::test]
    async fn test_set_config_request_structure() {
        use spark_connect_proto::config_request::{operation::OpType, Set};
        let (client, received, server) = recorded_client().await;
        client
            .set_config("spark.sql.shuffle.partitions", "200")
            .await
            .unwrap();
        server.abort();
        assert_eq!(
            config_operation(&client, &received),
            OpType::Set(Set {
                pairs: vec![spark_connect_proto::KeyValue {
                    key: "spark.sql.shuffle.partitions".into(),
                    value: Some("200".into()),
                }],
                silent: None,
            })
        );
    }

    #[tokio::test]
    async fn test_unset_config_request_structure() {
        use spark_connect_proto::config_request::{operation::OpType, Unset};
        let (client, received, server) = recorded_client().await;
        client
            .unset_config("spark.sql.shuffle.partitions")
            .await
            .unwrap();
        server.abort();
        assert_eq!(
            config_operation(&client, &received),
            OpType::Unset(Unset {
                keys: vec!["spark.sql.shuffle.partitions".into()],
            })
        );
    }

    #[tokio::test]
    async fn test_interrupt_all_request_structure() {
        use spark_connect_proto::interrupt_request::InterruptType;
        let (client, received, server) = recorded_client().await;
        client.interrupt_all().await.unwrap();
        server.abort();
        assert_eq!(
            interruption(&client, &received),
            (InterruptType::All as i32, None)
        );
    }

    #[tokio::test]
    async fn test_interrupt_tag_request_structure() {
        use spark_connect_proto::interrupt_request::{Interrupt, InterruptType};
        let (client, received, server) = recorded_client().await;
        client.interrupt_tag("my-tag").await.unwrap();
        server.abort();
        assert_eq!(
            interruption(&client, &received),
            (
                InterruptType::Tag as i32,
                Some(Interrupt::OperationTag("my-tag".into()))
            )
        );
    }

    #[tokio::test]
    async fn test_interrupt_operation_request_structure() {
        use spark_connect_proto::interrupt_request::{Interrupt, InterruptType};
        let (client, received, server) = recorded_client().await;
        client.interrupt_operation("operation-123").await.unwrap();
        server.abort();
        assert_eq!(
            interruption(&client, &received),
            (
                InterruptType::OperationId as i32,
                Some(Interrupt::OperationId("operation-123".into()))
            )
        );
    }

    #[tokio::test]
    async fn test_release_session_request_structure() {
        let (client, received, server) = recorded_client().await;
        client.release_session().await.unwrap();
        server.abort();
        let request: ReleaseSessionRequest = only_request(&received, "ReleaseSession");
        assert_eq!(request.session_id, client.session_id());
        assert_eq!(request.user_context.unwrap().user_id, "u1");
        assert_eq!(request.client_type, Some(client.user_agent.clone()));
        assert!(!request.allow_reconnect);
    }

    #[tokio::test]
    async fn requests_carry_no_user_id_header_as_pyspark() {
        // PySpark's `ChannelBuilder.metadata()` leaves the user_id parameter out: the user ID is
        // in the user context of the requests.
        let builder = ChannelBuilder::parse("sc://localhost/;user_id=u1;x-hdr=v").unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        let mut request = Request::new(());
        client._attach_metadata(&mut request);
        assert!(request.metadata().get("user_id").is_none());
        assert_eq!(request.metadata().get("x-hdr").unwrap(), "v");
        assert!(client.execution_metadata().get("user_id").is_none());
    }

    #[tokio::test]
    async fn connect_raises_for_a_keepalive_param_that_is_not_an_integer_as_pyspark() {
        // As PySpark's client, whose channel options read the parameter with `int()`.
        let builder = ChannelBuilder::parse("sc://localhost/;grpc_keepalive_time_ms=abc").unwrap();
        let error = SparkConnectClient::connect(&builder)
            .await
            .err()
            .expect("a keepalive time that is not an integer must raise");
        assert_eq!(error.kind, SparkErrorKind::BuiltinValueError);
        assert_eq!(
            error.message(),
            "invalid literal for int() with base 10: 'abc'"
        );
    }
}

#[cfg(test)]
mod metadata_validation_tests {
    use super::*;
    use crate::error::SparkErrorKind;

    /// Connects without dialing (the channel is lazy), with no token in the environment.
    fn connect(url: &str) -> Result<SparkConnectClient> {
        let _env = crate::channel::set_env(&[
            ("SPARK_CONNECT_AUTHENTICATE_TOKEN", None),
            ("SPARK_CONNECT_USER_AGENT", None),
        ]);
        let builder = ChannelBuilder::parse(url)?;
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(SparkConnectClient::connect(&builder))
    }

    fn connect_err(url: &str) -> SparkError {
        match connect(url) {
            Ok(_) => panic!("{url:?} should not connect"),
            Err(e) => e,
        }
    }

    /// Everything of an error that a caller or a log may show.
    fn shown(e: &SparkError) -> String {
        format!("{e} | {e:?} | {}", e.get_message())
    }

    #[test]
    fn token_that_is_not_a_header_value_fails_to_connect() {
        // grpc-python fails every call of such a token with UNAVAILABLE "Illegal metadata"
        // (its call credentials add the header); before, the header was left off silently.
        for url in [
            "sc://localhost/;token=SECRETTOK%0A",
            "sc://localhost/;token=SECRET%0D%0ATOK",
            "sc://localhost/;token=SECRETTOK%C3%A9",
            "sc://localhost/;token=SECRET%7FTOK",
            "sc://localhost/;token=SECRET%09TOK",
            "sc://example.com/;token=SECRETTOK%00",
        ] {
            let e = connect_err(url);
            assert_eq!(
                e.grpc_code(),
                Some(tonic::Code::Unavailable as i32),
                "{url}"
            );
            let shown = shown(&e);
            assert!(shown.contains("Illegal metadata"), "{shown}");
            assert!(shown.contains("token"), "{shown}");
            assert!(
                !shown.contains("SECRET") && !shown.contains("TOK"),
                "{shown}"
            );
        }
    }

    #[test]
    fn token_header_is_checked_whatever_the_source_of_the_token() {
        // The token of the environment (`SPARK_CONNECT_AUTHENTICATE_TOKEN`) takes the path of
        // the one of the connection string, `ChannelBuilder::token`. It is not set here: the
        // other tests of the crate connect without holding the lock of the environment.
        let e = token_header("Bearer SECRET\nTOK").unwrap_err();
        let shown = shown(&e);
        assert!(shown.contains("Illegal metadata"), "{shown}");
        assert!(
            !shown.contains("SECRET") && !shown.contains("TOK"),
            "{shown}"
        );
        let header = token_header("Bearer SECRETTOK").unwrap();
        assert_eq!(header, "Bearer SECRETTOK");
        assert!(header.is_sensitive());
    }

    #[test]
    fn metadata_value_that_is_not_a_header_value_fails_to_connect() {
        // grpc-python raises the builtin ValueError "metadata was invalid: ..." for these.
        for url in [
            "sc://localhost/;x-hdr=VAL%0AUE",
            "sc://localhost/;x-hdr=VALUE%C3%A9",
            "sc://localhost/;x-hdr=VAL%7FUE",
            "sc://localhost/;x-hdr=VAL%09UE",
            "sc://localhost/;ok=1;x-hdr=%00VALUE",
        ] {
            let e = connect_err(url);
            assert_eq!(e.kind, SparkErrorKind::BuiltinValueError, "{url}");
            let shown = shown(&e);
            assert!(shown.contains("metadata was invalid"), "{shown}");
            assert!(shown.contains("x-hdr"), "{shown}");
            assert!(!shown.contains("VAL") && !shown.contains("UE"), "{shown}");
        }
    }

    #[test]
    fn authorization_parameter_that_is_not_a_header_value_is_not_echoed() {
        let e = connect_err("sc://localhost/;authorization=Bearer%20SECRETXYZ%0A");
        assert_eq!(e.kind, SparkErrorKind::BuiltinValueError);
        let shown = shown(&e);
        assert!(shown.contains("authorization"), "{shown}");
        assert!(!shown.contains("SECRETXYZ"), "{shown}");
    }

    #[test]
    fn metadata_key_that_is_not_a_header_key_fails_to_connect() {
        // The keys of gRPC are of `[0-9a-z_.-]`: grpc-python raises the builtin ValueError
        // for any other, an upper-case one included.
        for (url, key) in [
            ("sc://localhost/;x hdr=v", "x hdr"),
            ("sc://localhost/;X-Hdr=v", "X-Hdr"),
            ("sc://localhost/;x%20hdr=v", "x%20hdr"),
            ("sc://localhost/;x:hdr=v", "x:hdr"),
            ("sc://localhost/;h\u{e9}=v", "h\u{e9}"),
            ("sc://localhost/;=v", "''"),
        ] {
            let e = connect_err(url);
            assert_eq!(e.kind, SparkErrorKind::BuiltinValueError, "{url}");
            let shown = shown(&e);
            assert!(shown.contains("metadata was invalid"), "{shown}");
            assert!(shown.contains(key), "{url}: {shown}");
        }
    }

    #[test]
    fn binary_metadata_key_fails_to_connect() {
        // A `-bin` key takes bytes, which a connection string cannot give: grpc-python raises
        // a TypeError for its str value.
        let e = connect_err("sc://localhost/;x-bin=VALUE");
        assert_eq!(e.kind, SparkErrorKind::TypeError);
        let shown = shown(&e);
        assert!(shown.contains("x-bin"), "{shown}");
        assert!(!shown.contains("VALUE"), "{shown}");
    }

    #[test]
    fn user_agent_that_is_not_a_header_value_fails_to_connect() {
        let e = connect_err("sc://localhost/;user_agent=my%0Aagent");
        assert_eq!(e.kind, SparkErrorKind::BuiltinValueError);
        assert!(shown(&e).contains("user_agent"), "{}", shown(&e));
    }

    #[test]
    fn valid_metadata_is_on_every_request() {
        let client =
            connect("sc://localhost/;token=tok-1.2_3~+;x-hdr=a b;x_y.z-0=;user_agent=caf%C3%A9")
                .unwrap();
        let mut request = Request::new(());
        client._attach_metadata(&mut request);
        for metadata in [request.metadata().clone(), client.execution_metadata()] {
            assert_eq!(metadata.get("authorization").unwrap(), "Bearer tok-1.2_3~+");
            assert_eq!(metadata.get("x-hdr").unwrap(), "a b");
            // An empty value is a value, as in grpc-python.
            assert_eq!(metadata.get("x_y.z-0").unwrap(), "");
            // The user agent goes as its UTF-8 bytes, as grpc-python sends it.
            let user_agent = metadata.get("user-agent").unwrap().as_encoded_bytes();
            assert!(user_agent.starts_with("caf\u{e9} spark/".as_bytes()));
            assert_eq!(metadata.len(), 4, "{metadata:?}");
        }
    }

    #[test]
    fn token_parameter_wins_over_an_authorization_parameter() {
        let client = connect("sc://localhost/;authorization=Basic%20abc;token=tok").unwrap();
        let metadata = client.execution_metadata();
        let all: Vec<_> = metadata.get_all("authorization").iter().collect();
        assert_eq!(all, ["Bearer tok"]);
    }

    #[test]
    fn debug_of_the_request_metadata_does_not_show_the_secrets() {
        for url in [
            "sc://localhost/;token=SECRETTOK",
            "sc://localhost/;authorization=Bearer%20SECRETTOK",
        ] {
            let client = connect(url).unwrap();
            let mut request = Request::new(());
            client._attach_metadata(&mut request);
            let shown = format!("{:?} {:?}", request.metadata(), client.execution_metadata());
            assert!(shown.contains("authorization"), "{shown}");
            assert!(!shown.contains("SECRETTOK"), "{shown}");
        }
    }

    #[test]
    fn new_session_keeps_the_metadata() {
        let client = connect("sc://localhost/;token=tok;x-hdr=v").unwrap();
        let other = client.with_new_session_id();
        let metadata = other.execution_metadata();
        assert_eq!(metadata.get("authorization").unwrap(), "Bearer tok");
        assert_eq!(metadata.get("x-hdr").unwrap(), "v");
    }
}
