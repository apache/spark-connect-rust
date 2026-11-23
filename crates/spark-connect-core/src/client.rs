//! Spark Connect client implementation.
//!
//! Mirrors `pyspark.sql.connect.client.core.SparkConnectClient`.

use std::collections::BTreeMap;
use std::time::Duration;
use tonic::metadata::{AsciiMetadataKey, AsciiMetadataValue, MetadataMap};
use tonic::transport::Channel;
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
    GetStatusRequest, GetStatusResponse, InterruptRequest, InterruptResponse, Plan,
    ReattachExecuteRequest, ReleaseExecuteRequest, ReleaseExecuteResponse, ReleaseSessionRequest,
    UserContext,
};

use crate::artifact::{
    cache_artifact_requests, check_artifact_upload, take_frame_failure, ArtifactFrames,
    FileArtifact, NamedArtifact,
};
use crate::channel::{is_secret_param, ChannelBuilder, GRPC_MAX_MESSAGE_LENGTH_DEFAULT};
use crate::error::{GrpcCall, Result, SparkError, SparkErrorKind};
use crate::proxy::ProxyEnv;
use crate::reattach::ExecutePlanResponseReattachableIterator;
use crate::retries::{
    retrying, unknown_operation, BackgroundRequest, Backoff, RetriesExceeded, RetryDecider,
    RetryError, RetryPolicy, Retrying,
};
use crate::thread_entries::forget_when_thread_ends;

/// How long [`wait_ready`] waits for the lazy gRPC channel to take a call before surfacing a
/// retriable `UNAVAILABLE`. The connection itself is bounded by the connector's deadlines (see
/// [`crate::connector`]).
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

/// Names a PEM file of root certificates to trust instead of the default ones, as in grpcio.
const SSL_ROOTS_FILE_VAR: &str = "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH";

/// The PEM roots that `path`, the value of [`SSL_ROOTS_FILE_VAR`], names, as grpcio reads them:
/// `None` (the system's roots) without a value or of an empty file, and of a file that cannot be
/// read, which is logged as grpcio's core logs it (see [`crate::grpc_log`]).
fn ssl_roots_pem(path: Option<std::ffi::OsString>) -> Option<Vec<u8>> {
    let path = path.filter(|path| !path.is_empty())?;
    let path = std::path::Path::new(&path);
    match std::fs::read(path) {
        Ok(pem) => (!pem.is_empty()).then_some(pem),
        Err(error) => {
            crate::grpc_log::error(format!(
                "{SSL_ROOTS_FILE_VAR} names {}, which cannot be read ({error}); using the \
                 default roots",
                path.display()
            ));
            None
        }
    }
}

/// The PEM file of the roots that the TLS channels trust besides the system's store, if any (see
/// [`set_bundled_ssl_roots_file`]).
static BUNDLED_ROOTS_FILE: std::sync::RwLock<Option<std::path::PathBuf>> =
    std::sync::RwLock::new(None);

/// Make the TLS channels trust the roots of the PEM file `path` besides those of the system's
/// store, unless `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH` names a file, whose roots alone they trust: as
/// the Python binding names grpcio's own bundle (`roots.pem` of the `grpc` package) where grpcio
/// is installed, whose roots PySpark's channels trust, so that the drop-in trusts those that
/// PySpark trusts, and those of the system's store too, which an unrelated install of grpcio does
/// not take away. Of the roots read from then on, once per process, when the first TLS channel is
/// made. A file that cannot be read, or is empty, is as none. Not a stable API.
#[doc(hidden)]
pub fn set_bundled_ssl_roots_file(path: Option<std::path::PathBuf>) {
    *BUNDLED_ROOTS_FILE
        .write()
        .unwrap_or_else(|e| e.into_inner()) = path;
}

/// The roots of the bundled roots file, if one is set and can be read.
fn bundled_roots_pem() -> Option<Vec<u8>> {
    let path = BUNDLED_ROOTS_FILE
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()?;
    std::fs::read(path).ok().filter(|pem| !pem.is_empty())
}

/// The roots of the TLS channels, read when the first TLS channel of the process is made, as
/// grpcio computes its default roots once, so that a later change of the variable, of the files or
/// of the system's store is not seen, nor logged again, and every channel of the process trusts the
/// same roots, of which its subchannels are shared (see `crate::connector::SubchannelArgs`): those
/// of [`ssl_roots_pem`] alone, else the system's store (see `crate::connector::system_roots`) and
/// the bundled roots file (see [`set_bundled_ssl_roots_file`]).
fn ssl_roots() -> crate::connector::Roots<'static> {
    struct Read {
        file: Option<Vec<u8>>,
        native: Vec<crate::connector::SystemRoot>,
        bundle: Option<Vec<u8>>,
    }
    static ROOTS: std::sync::OnceLock<Read> = std::sync::OnceLock::new();
    let read = ROOTS.get_or_init(|| {
        let file = ssl_roots_pem(std::env::var_os(SSL_ROOTS_FILE_VAR));
        let (native, bundle) = if file.is_none() {
            (crate::connector::system_roots(), bundled_roots_pem())
        } else {
            (Vec::new(), None)
        };
        Read {
            file,
            native,
            bundle,
        }
    });
    match &read.file {
        Some(file) => crate::connector::Roots::Only(file),
        None => crate::connector::Roots::System {
            native: &read.native,
            bundle: read.bundle.as_deref(),
        },
    }
}

/// The details of the failed call of `status` as grpcio gives them: tonic's message, which is the
/// connector's text of a connection that failed (see [`crate::connector`]), but of a call that the
/// end of its connection failed, `Stream removed (...)` in the dialect of grpcio, and of a message
/// over the size limit of a call (see [`oversized`]), gRPC core's text of it there, `Stream removed
/// (CLIENT: Received message larger than max (<length> vs. <limit>))` of a response and `CLIENT:
/// Sent message larger than max (<length> vs. <limit>)` of a request.
pub(crate) fn status_details(status: &tonic::Status) -> String {
    status_details_in(status, crate::connector::grpcio_dialect())
}

/// As [`status_details`], in the dialect of grpcio if `grpcio`.
fn status_details_in(status: &tonic::Status, grpcio: bool) -> String {
    if grpcio {
        // Of metadata that gRPC's core refuses, as it ends the stream of the call.
        if crate::metadata::is_refused(status) {
            return format!("Stream removed ({})", status.message());
        }
        match oversized(status) {
            Some((Oversized::Received, length, limit)) => {
                return format!(
                    "Stream removed (CLIENT: Received message larger than max ({length} vs. \
                     {limit}))"
                );
            }
            Some((Oversized::Sent, length, limit)) => {
                return format!("CLIENT: Sent message larger than max ({length} vs. {limit})");
            }
            None => {}
        }
    }
    crate::connector::stream_removed_in(status, grpcio)
        .unwrap_or_else(|| status.message().to_string())
}

/// Whether a message over the size limit of a call is a response or a request (see [`oversized`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Oversized {
    /// A response that tonic does not read.
    Received,
    /// A request that the client does not send.
    Sent,
}

/// The message over the size limit of a call that the client's status `status` is of, its
/// direction, length and limit (see [`oversized_of`]), of the limit of the client's calls.
pub(crate) fn oversized(status: &tonic::Status) -> Option<(Oversized, u64, u64)> {
    oversized_of(status, GRPC_MAX_MESSAGE_LENGTH_DEFAULT as u64)
}

/// The message over the size limit `limit` of a call that the client's status `status` is of, its
/// direction, length and limit: of a response, tonic's status of its pinned text, `Error, decoded
/// message length too large: found <length> bytes, the limit is: <limit> bytes`, of the code
/// `OUT_OF_RANGE`, which tonic gives as it reads the header of the message, before any codec sees
/// it, of the limit of the client; of a request, the client's own, which it tags (see
/// [`too_large_to_send`]). A status of a server of the same code and text, of no details, as a
/// server of tonic sends of a message over its own limit, is read so too only if the limit of the
/// server is that of the client. grpc-python's code of either is `RESOURCE_EXHAUSTED`.
fn oversized_of(status: &tonic::Status, limit: u64) -> Option<(Oversized, u64, u64)> {
    if status.code() != tonic::Code::OutOfRange || !status.details().is_empty() {
        return None;
    }
    let sent = std::iter::successors(std::error::Error::source(status), |error| error.source())
        .any(|error| error.is::<RequestTooLarge>());
    let (direction, rest) = if sent {
        (
            Oversized::Sent,
            status
                .message()
                .strip_prefix("Error, encoded message length too large: ")?,
        )
    } else {
        (
            Oversized::Received,
            status
                .message()
                .strip_prefix("Error, decoded message length too large: ")?,
        )
    };
    let (length, of) = rest
        .strip_prefix("found ")?
        .strip_suffix(" bytes")?
        .split_once(" bytes, the limit is: ")?;
    let (length, of) = (length.parse().ok()?, of.parse().ok()?);
    (sent || of == limit).then_some((direction, length, of))
}

/// The source of the client's status of a request over the size limit of a call, which it does
/// not send (see [`too_large_to_send`]).
#[derive(Debug, Clone, Copy)]
struct RequestTooLarge;

impl std::fmt::Display for RequestTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the client does not send a request over the size limit of a call")
    }
}

impl std::error::Error for RequestTooLarge {}

/// The status of a request of `length` bytes over the size limit of a call, as tonic's encoding
/// gives it, tagged as the client's own (see [`oversized`]).
fn too_large_to_send(length: usize) -> tonic::Status {
    let mut status = tonic::Status::out_of_range(format!(
        "Error, encoded message length too large: found {length} bytes, the limit is: \
         {GRPC_MAX_MESSAGE_LENGTH_DEFAULT} bytes"
    ));
    status.set_source(std::sync::Arc::new(RequestTooLarge));
    status
}

/// The error of a request of the bytes `request` over the size limit of a call, if it is.
#[allow(clippy::result_large_err)] // SparkError is large, as in the other entry points
fn sendable_bytes(request: &[u8]) -> Result<()> {
    if request.len() > GRPC_MAX_MESSAGE_LENGTH_DEFAULT {
        return Err(SparkError::from_grpc_status(too_large_to_send(
            request.len(),
        )));
    }
    Ok(())
}

/// `request`, unless its message is over the size limit of a call (see [`Stub`]).
fn sendable<T: prost::Message>(
    request: impl tonic::IntoRequest<T>,
) -> std::result::Result<Request<T>, tonic::Status> {
    let request = request.into_request();
    match request.get_ref().encoded_len() {
        length if length > GRPC_MAX_MESSAGE_LENGTH_DEFAULT => Err(too_large_to_send(length)),
        _ => Ok(request),
    }
}

/// The generated gRPC client of the channel of a client (see [`SparkConnectClient::stub`]), which
/// refuses a request over the size limit of a call before it is sent, of tonic's status of it
/// (see [`oversized`]): tonic checks the length of a request as it encodes it in the body of the
/// call, and its failure there ends the call as a reset of its stream (`h2 protocol error`), of
/// which hyper drops the cause.
#[derive(Clone)]
struct Stub(SparkConnectServiceClient<crate::metadata::Checked>);

/// A call of the generated client whose request is checked first (see [`Stub`]).
macro_rules! checked_calls {
    ($($method:ident($request:ty) -> $response:ty;)*) => {
        impl Stub {
            $(
                async fn $method(
                    &mut self,
                    request: impl tonic::IntoRequest<$request>,
                ) -> std::result::Result<tonic::Response<$response>, tonic::Status> {
                    self.0.$method(sendable(request)?).await
                }
            )*
        }
    };
}

checked_calls! {
    execute_plan(ExecutePlanRequest) -> Streaming<ExecutePlanResponse>;
    reattach_execute(ReattachExecuteRequest) -> Streaming<ExecutePlanResponse>;
    release_execute(ReleaseExecuteRequest) -> ReleaseExecuteResponse;
    analyze_plan(AnalyzePlanRequest) -> AnalyzePlanResponse;
    config(spark_connect_proto::ConfigRequest) -> spark_connect_proto::ConfigResponse;
    interrupt(spark_connect_proto::InterruptRequest) -> spark_connect_proto::InterruptResponse;
    fetch_error_details(spark_connect_proto::FetchErrorDetailsRequest) -> spark_connect_proto::FetchErrorDetailsResponse;
    clone_session(spark_connect_proto::CloneSessionRequest) -> spark_connect_proto::CloneSessionResponse;
    release_session(spark_connect_proto::ReleaseSessionRequest) -> spark_connect_proto::ReleaseSessionResponse;
    get_status(spark_connect_proto::GetStatusRequest) -> GetStatusResponse;
    artifact_status(spark_connect_proto::ArtifactStatusesRequest) -> spark_connect_proto::ArtifactStatusesResponse;
}

impl Stub {
    /// The upload of artifacts, whose requests, chunks of the artifacts, are far smaller than
    /// the size limit of a call.
    async fn add_artifacts(
        &mut self,
        request: impl tonic::IntoStreamingRequest<Message = spark_connect_proto::AddArtifactsRequest>,
    ) -> std::result::Result<
        tonic::Response<spark_connect_proto::AddArtifactsResponse>,
        tonic::Status,
    > {
        self.0.add_artifacts(request).await
    }
}

/// What the channels of a client are made of: the endpoint, of its keepalive and its origin, and
/// what its connections connect to. A client of a new session gets a channel of its own of them
/// (see [`ChannelFactory::channel`]).
#[derive(Clone)]
struct ChannelFactory {
    endpoint: tonic::transport::Endpoint,
    /// The origin of the requests and the keepalive of the endpoint, which the subchannels of the
    /// process are told apart by, with their addresses and the target.
    origin: String,
    keepalive: Option<crate::channel::Keepalive>,
    /// Whether each channel of the factory has call credentials of its own: of a connection string
    /// of a token, whose sessions grpcio gives channels of their own credentials, which share no
    /// connection, as the token of each is its own.
    own_connection: bool,
    target: crate::connector::Target,
}

/// The use of its channel by a client of a session and by its clones: open until the client is
/// closed, or gone, which shuts the channel down, so that it lets go of its subchannels (see
/// [`crate::connector::ClientChannel::shutdown`]).
///
/// A lease is of the generation of the process that made it: in a child of a `fork()`, a lease of
/// the parent, and its channel, are left as they are when it drops, as the subchannels and the
/// connections of the channel are of the runtime of the parent, whose epoll instance the child
/// shares (see [`crate::fork`]).
struct ChannelLease {
    /// The channel of the client, as each client of PySpark has a channel of grpcio of its own:
    /// its resolver, and its `pick_first` over the subchannels of the process, which it shares
    /// with the other channels of the process.
    channel: crate::fork::ForkLocal<std::sync::Arc<crate::connector::ClientChannel>>,
    open: std::sync::atomic::AtomicBool,
}

impl ChannelLease {
    fn new(channel: std::sync::Arc<crate::connector::ClientChannel>) -> Self {
        Self {
            channel: crate::fork::ForkLocal::new(channel),
            open: std::sync::atomic::AtomicBool::new(true),
        }
    }

    fn close(&self) {
        if self.open.swap(false, std::sync::atomic::Ordering::SeqCst) {
            self.channel.shutdown();
        }
    }

    fn is_open(&self) -> bool {
        self.open.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// The channel of a client and of its clones, in the generation of the process that it is of: a
/// client of a parent of a `fork()` takes a channel of the child's in the child, whose threads the
/// parent's channel does not have, as grpcio's fork support resets the channels of a parent there.
/// The lease of each generation is taken without a lock, which a thread of the parent may have
/// held at the fork, and the lease of the parent is left as it is in the child.
struct Transport {
    factory: std::sync::Arc<ChannelFactory>,
    current: crate::fork::PerGeneration<std::sync::Arc<ChannelLease>>,
}

impl Transport {
    fn new(factory: std::sync::Arc<ChannelFactory>) -> Self {
        let transport = Self {
            factory,
            current: crate::fork::PerGeneration::new(),
        };
        transport.lease();
        transport
    }

    /// The lease of the channel of the generation of the process: in a child, of a channel of its
    /// own, closed if the client was closed in the parent.
    fn lease(&self) -> std::sync::Arc<ChannelLease> {
        std::sync::Arc::clone(self.current.get_or_init(|parent| {
            let lease = std::sync::Arc::new(ChannelLease::new(self.factory.channel()));
            if parent.is_some_and(|parent| !parent.is_open()) {
                lease.close();
            }
            lease
        }))
    }
}

impl Drop for ChannelLease {
    fn drop(&mut self) {
        // A lease of a parent is left as it is: its channel is of the parent's runtime.
        if self.channel.is_current() && *self.open.get_mut() {
            self.channel.shutdown();
        }
    }
}

/// The proxy of the environment now that the channel of `builder` connects through, if any, as
/// gRPC's proxy mapper decides it: of a target of gRPC, of the host of its path, and none of a
/// socket of `unix:`; else of the host of the connection string.
fn proxy_of(
    builder: &ChannelBuilder,
) -> std::result::Result<Option<crate::proxy::HttpProxy>, String> {
    // Of a channel of no credentials that sends its token in plaintext, of the rule of
    // `ChannelBuilder::secure` (a token to `localhost`), a loopback host connects directly, as no
    // local credentials keep the token from a proxy (see `crate::proxy`).
    let direct_loopback =
        builder.credentials().is_none() && !builder.secure() && builder.token().is_some();
    match builder.target_names() {
        Some(names) => match &names.proxied {
            Some((host, _)) => ProxyEnv::from_process().proxy_for(host, direct_loopback),
            None => Ok(None),
        },
        None => ProxyEnv::from_process().proxy_for(builder.host(), direct_loopback),
    }
}

impl ChannelFactory {
    /// The factory of the channel of a new session of the same endpoint as of `builder`, through
    /// the proxy of the environment now, as grpcio's channel of a new client reads it, which logs
    /// what grpcio logs of it again. Of the proxy of this factory if that one is not valid, as
    /// the error of the creation of a session of it is of the first session only.
    fn renewed(&self, builder: &ChannelBuilder) -> Self {
        let target = match proxy_of(builder) {
            Ok(proxy) => self.target.renewed(builder.host(), builder.port(), proxy),
            Err(_) => self.target.clone(),
        };
        Self {
            endpoint: self.endpoint.clone(),
            origin: self.origin.clone(),
            keepalive: self.keepalive,
            own_connection: self.own_connection,
            target,
        }
    }

    /// A channel of a client of this factory (see [`crate::connector::ClientChannel`]), of the
    /// subchannels of the process, of its call credentials of its own of a token, whose work in
    /// the background and whose connections run on the shared runtime, as the subchannels are the
    /// process's, which outlive the runtime of the caller.
    fn channel(&self) -> std::sync::Arc<crate::connector::ClientChannel> {
        let keepalive = self
            .keepalive
            .map(|keepalive| (keepalive.time, keepalive.timeout, keepalive.without_calls));
        crate::connector::ClientChannel::new(
            self.target.clone(),
            crate::connector::SubchannelArgs::new(
                &self.target,
                &self.origin,
                keepalive,
                self.own_connection,
            ),
            self.endpoint.clone(),
            crate::connector::Pool::of_the_process(),
            crate::runtime::get_runtime().handle().clone(),
        )
    }
}

/// The flow-control window of a connection, the largest of HTTP/2. h2 gives the window of the
/// connection back as each stream's data is read, so that the streams that a caller does not read
/// on, such as the results of `toLocalIterator()` that a session holds open, would use the window
/// of the connection up, and every other call of the connection, of any session that shares it,
/// would wait. gRPC's core grows the window of its connections with the windows of their streams,
/// so that none starves another: a window this large does so too, and the window of each stream
/// (hyper's, of 2 MiB) still bounds what a stream holds unread.
const CONNECTION_WINDOW: u32 = (1 << 31) - 1;

/// gRPC/HTTP header carrying the connection-string `token` as `Bearer <token>`.
/// Named here (rather than as a string literal) so the key is written once.
const AUTHORIZATION_HEADER: &str = "authorization";

/// The HTTP header of the client library, tonic's own, as grpcio sends its own: never the user
/// agent of the connection string, which the requests carry as their client type.
const USER_AGENT_HEADER: &str = "user-agent";

/// Wait for `grpc` to become ready, bounded by [`READY_TIMEOUT`], mapping any failure
/// (or the timeout) to a gRPC `UNAVAILABLE` error. This matches how grpcio reports an
/// unreachable server, so the reference client raises `SparkConnectGrpcException` with
/// `UNAVAILABLE` rather than hanging on a channel that never connects.
async fn wait_ready(grpc: &mut tonic::client::Grpc<crate::metadata::Checked>) -> Result<()> {
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
    pub get_status: Option<Duration>,
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
            get_status: minutes(10),
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
            get_status: None,
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

/// Add the entries of `metadata` to those of the request `req`: its `grpc-timeout` of a deadline,
/// which [`with_deadline`] set, stays.
fn add_metadata<T>(req: &mut Request<T>, metadata: &MetadataMap) {
    let added = req.metadata_mut();
    for entry in metadata.iter() {
        match entry {
            tonic::metadata::KeyAndValueRef::Ascii(key, value) => {
                added.append(key.clone(), value.clone());
            }
            tonic::metadata::KeyAndValueRef::Binary(key, value) => {
                added.append_bin(key.clone(), value.clone());
            }
        }
    }
}

/// `request`, an execution request, of an operation ID: in the dialect of grpcio, of a new one when
/// it has none, the one that the iterator of the execution would give it, as PySpark's request has
/// one before its hooks run and its stub is called, so that the error of a closed client or
/// channel carries it.
fn with_operation_id_of_grpcio(mut request: ExecutePlanRequest) -> ExecutePlanRequest {
    if crate::connector::grpcio_dialect()
        && request.operation_id.as_deref().is_none_or(str::is_empty)
    {
        request.operation_id = Some(Uuid::new_v4().to_string());
    }
    request
}

/// A failure of a call whose conversion runs for the calling thread (see [`converting_failure`]).
/// Not a stable API.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct Conversion {
    /// grpcio's error of the failure.
    pub failure: crate::error::GrpcioError,
    /// The operation ID of the execution of the failure, of a failure of an execution, which
    /// PySpark's `_handle_error` sets on any exception that goes out of the conversion.
    pub operation_id: Option<String>,
}

/// A mark of [`CONVERTING`]: the conversion, of the thread that its call acts for, of the watch
/// of the waits of that call (see [`crate::runtime::wait_watch_id`]) and of the client of the call
/// (see [`SparkConnectClient::identity`]).
struct ConvertingMark {
    thread: std::thread::ThreadId,
    watch: Option<usize>,
    client: usize,
    conversion: Conversion,
}

/// The failures of the calls whose conversion runs now (see [`converting_failure`]), of each
/// generation of the process (see [`crate::fork`]).
static CONVERTING: crate::fork::PerGeneration<std::sync::Mutex<Vec<ConvertingMark>>> =
    crate::fork::PerGeneration::new();

/// The marks of [`CONVERTING`] of the generation of the process, locked.
fn converting() -> std::sync::MutexGuard<'static, Vec<ConvertingMark>> {
    CONVERTING
        .get_or_init(|_| Default::default())
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Marks the failure of a call as converted for the thread that the call acts for, until it
/// drops (see [`converting_failure`]).
struct Converting {
    thread: std::thread::ThreadId,
}

impl Converting {
    fn start(
        failure: &crate::error::GrpcioError,
        operation_id: Option<String>,
        client: usize,
    ) -> Self {
        let thread = crate::runtime::calling_thread_id();
        converting().push(ConvertingMark {
            thread,
            watch: crate::runtime::wait_watch_id(),
            client,
            conversion: Conversion {
                failure: failure.clone(),
                operation_id,
            },
        });
        Converting { thread }
    }
}

impl Drop for Converting {
    fn drop(&mut self) {
        let mut converting = converting();
        if let Some(position) = converting
            .iter()
            .rposition(|mark| mark.thread == self.thread)
        {
            converting.remove(position);
        }
    }
}

/// The failure of a call whose conversion runs now for the calling thread, the innermost, if any,
/// of the call whose waits the watch of the calling thread polls (see
/// [`crate::runtime::with_wait_watch`]): of its logs, of the request of the details of an error of
/// the server and of the reads of the configs of its stack trace. PySpark's `_handle_error`
/// converts the failure of a call in the `except Exception` clause of the call, outside of the
/// `except KeyboardInterrupt` of an execution: the exception that a signal handler raises
/// meanwhile, as the `KeyboardInterrupt` of Ctrl-C, goes out of the call as it is, of that
/// failure as its context and of the operation ID of its execution, and interrupts nothing, so
/// that a wait that polls the signals ends then rather than interrupt the execution of the
/// failure. A call that runs during the conversion, as of a handler of one of its logs, is not of
/// it, as PySpark's call runs a conversion of its own, if any. Not a stable API.
#[doc(hidden)]
pub fn converting_failure() -> Option<Conversion> {
    let thread = crate::runtime::calling_thread_id();
    let watch = crate::runtime::wait_watch_id();
    converting()
        .iter()
        .rev()
        .find(|mark| mark.thread == thread && mark.watch == watch)
        .map(|mark| mark.conversion.clone())
}

/// The operation IDs of the executions whose failure a call that acts for `thread` converts now,
/// of any call of the thread (see [`converting_failure`]).
fn converted_operation_ids(thread: std::thread::ThreadId) -> Vec<String> {
    converting()
        .iter()
        .filter(|mark| mark.thread == thread)
        .filter_map(|mark| mark.conversion.operation_id.clone())
        .collect()
}

/// Tell the listener of the retries of the process of the failure `failed` of a request in the
/// background, `request` (see [`crate::retries::RetriesListener::background_failure`]): of its
/// spent retries, of a failure that is not retried, or of the error of the decider of its
/// retries.
fn background_failure<E>(request: BackgroundRequest, failed: RetryError<E>) {
    use crate::retries::BackgroundFailure;
    let text = |status: &tonic::Status| crate::error::grpcio_error_text(status, GrpcCall::Unary);
    // A log that no caller waits for, of no error.
    let _ = crate::retries::log(|listener| {
        match failed {
            RetryError::Exceeded(status) => listener.background_failure(
                request,
                Some(RetriesExceeded::Policy),
                BackgroundFailure::Rpc(&text(&status)),
            ),
            RetryError::Status(status) => {
                listener.background_failure(request, None, BackgroundFailure::Rpc(&text(&status)))
            }
            RetryError::Failed(error) => {
                listener.background_failure(request, None, BackgroundFailure::Error(error))
            }
            RetryError::Aborted(_) | RetryError::WaitAborted => {}
        }
        Ok(())
    });
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

/// The details of grpcio's cancel of a call in flight when `close` closes its channel, before the
/// response of the call begins (see [`ChannelClosed`]).
pub(crate) const CLOSED_CHANNEL: &str = "Channel closed!";

/// The details of grpcio's cancel of a call in flight when `close` closes its channel, once the
/// response of the call began, as of a stream whose responses are read.
pub(crate) const STREAM_OF_A_CLOSED_CHANNEL: &str = "Stream removed (Channel closed!)";

/// The next read of `stream`, a stream of responses of an execution, after grpcio's `close` of its
/// channel, which cancels the calls of the channel whose status is not delivered yet: a call whose
/// responses were all read, and which the server ended, ends as the server ended it, of its status;
/// another fails as cancelled, `CANCELLED`, `Stream removed (Channel closed!)`, of none of the
/// responses that it holds, which the cancel discards, and of the trailing metadata of the end of
/// the call, if it arrived, as of the failure that the server ended the stream with, such as the
/// error of an execution of the session that the `stop()` of the client released, whose
/// `google.rpc.Status` then does not match the cancel, as grpc_status raises it of PySpark's call.
fn read_after_close(
    stream: &mut Streaming<ExecutePlanResponse>,
) -> std::result::Result<Option<ExecutePlanResponse>, tonic::Status> {
    read_after_close_of(stream, false)
}

/// As [`read_after_close`], of a stream of which a response was read already after the close, as
/// by a read in flight when the client closed the channel, if `discarded`: grpcio's cancel gives
/// no response after it, so that response is discarded too, and no release of it is sent.
fn read_after_close_of(
    stream: &mut Streaming<ExecutePlanResponse>,
    discarded: bool,
) -> std::result::Result<Option<ExecutePlanResponse>, tonic::Status> {
    use futures::FutureExt;
    const DETAILS: &str = STREAM_OF_A_CLOSED_CHANNEL;
    let mut unread = discarded;
    loop {
        match stream.message().now_or_never() {
            Some(Ok(Some(_))) => unread = true,
            Some(Ok(None)) if !unread => return Ok(None),
            Some(Err(ended)) if !unread => return Err(ended),
            Some(Err(ended)) => {
                return Err(tonic::Status::with_details_and_metadata(
                    tonic::Code::Cancelled,
                    DETAILS,
                    ended.details().to_vec().into(),
                    ended.metadata().clone(),
                ))
            }
            Some(Ok(None)) | None => return Err(tonic::Status::cancelled(DETAILS)),
        }
    }
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

/// What the Rust client makes of the failure of a call that PySpark's client leaves to grpc,
/// whose error the dialect of grpcio raises as grpcio's (see
/// [`SparkConnectClient::result_left_to_grpc`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeftToGrpc {
    /// The error of the server, as of the other calls, with the details that it asks for: of the
    /// uploads of `copy_from_local_to_fs` and of the caching of artifacts.
    Converted,
    /// The error of its status as it is, of nothing more asked of the server, which does not
    /// close the client: of `clone`, as PySpark's `clone` does not call `_handle_error`.
    AsItIs,
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

/// The settings of the plan compression of a client, as PySpark's client keeps them in its
/// `_plan_compression_threshold` and `_plan_compression_algorithm`: the threshold of the conf of
/// the server as an int (-1 of none), and its algorithm, which PySpark's plans compress with when it
/// is `ZSTD`, above a threshold that is not negative. Binding hook; not a stable API.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanCompressionSettings {
    pub threshold: i128,
    pub algorithm: Option<String>,
}

impl PlanCompressionSettings {
    /// The settings that a plan compression of no others stands for: of a failed lookup, of the
    /// threshold -1 and the algorithm `NONE`, as PySpark's lookup and its `CANNOT_PARSE` set them.
    fn of(compression: PlanCompression) -> Option<Self> {
        match compression {
            PlanCompression::Unknown => None,
            PlanCompression::Disabled => Some(Self {
                threshold: -1,
                algorithm: Some("NONE".to_string()),
            }),
            PlanCompression::Zstd { threshold } => Some(Self {
                threshold: i128::try_from(threshold).unwrap_or(i128::MAX),
                algorithm: Some("ZSTD".to_string()),
            }),
        }
    }
}

/// The threshold of the plan compression of the conf `threshold` of the server, as PySpark's
/// `int(plan_compression_threshold_str) if plan_compression_threshold_str else -1` reads it, or
/// None where `int()` raises.
fn plan_compression_threshold(threshold: Option<&str>) -> Option<i128> {
    match threshold {
        None | Some("") => Some(-1),
        Some(threshold) => crate::client_log::python_int(threshold),
    }
}

/// The plan compression of the threshold `threshold`, as PySpark reads it of its conf, and of the
/// algorithm `algorithm`: zstd above a threshold that is not negative.
fn plan_compression_above(threshold: i128, algorithm: Option<&str>) -> PlanCompression {
    match algorithm {
        Some("ZSTD") if threshold >= 0 => PlanCompression::Zstd {
            threshold: usize::try_from(threshold).unwrap_or(usize::MAX),
        },
        _ => PlanCompression::Disabled,
    }
}

/// The plan compression of the confs `threshold` and `algorithm` of the server, as PySpark reads
/// them: zstd above a threshold that is not negative, which any other algorithm, or a missing or
/// invalid threshold, disables; and what PySpark's `_get_plan_compression_threshold_and_algorithm`
/// logs of them: `Plan compression threshold: ..., algorithm: ...`, or, of a threshold that `int()`
/// does not read, its `ValueError` with which it disables the compression.
fn looked_up_plan_compression(
    threshold: Option<&str>,
    algorithm: Option<&str>,
) -> (PlanCompression, crate::client_log::Message) {
    use crate::client_log::Message;
    match plan_compression_threshold(threshold) {
        Some(value) => (
            plan_compression_above(value, algorithm),
            Message::Text(format!(
                "Plan compression threshold: {value}, algorithm: {}",
                algorithm.unwrap_or("None")
            )),
        ),
        None => {
            let text = threshold.unwrap_or_default();
            let error = SparkError::builtin_value_msg(format!(
                "invalid literal for int() with base 10: {}",
                python_str_repr(text)
            ));
            (
                PlanCompression::Disabled,
                Message::WithError {
                    text: PLAN_COMPRESSION_UNSUPPORTED,
                    error: Box::new(error),
                },
            )
        }
    }
}

/// What PySpark's `_get_plan_compression_threshold_and_algorithm` logs, of the exception that it
/// catches as its argument, when it disables the plan compression of a lookup that failed.
const PLAN_COMPRESSION_UNSUPPORTED: &str =
    "Plan compression is disabled because the server does not support it.";

/// The `repr` of the Python string `text`: in single quotes, or in double quotes of a text of
/// single quotes and of no double quotes, with the escapes of Python's `repr` of the backslash,
/// the quote, the whitespace controls and the other characters that are not printable.
fn python_str_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut repr = String::with_capacity(text.len() + 2);
    repr.push(quote);
    for c in text.chars() {
        match c {
            '\\' => repr.push_str("\\\\"),
            '\n' => repr.push_str("\\n"),
            '\r' => repr.push_str("\\r"),
            '\t' => repr.push_str("\\t"),
            c if c == quote => {
                repr.push('\\');
                repr.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                repr.push_str(&format!("\\x{:02x}", c as u32));
            }
            c if c.is_control() => {
                let code = c as u32;
                if code <= 0xff {
                    repr.push_str(&format!("\\x{code:02x}"));
                } else if code <= 0xffff {
                    repr.push_str(&format!("\\u{code:04x}"));
                } else {
                    repr.push_str(&format!("\\U{code:08x}"));
                }
            }
            c => repr.push(c),
        }
    }
    repr.push(quote);
    repr
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
/// `threshold` bytes and compression makes it smaller, as PySpark's `_try_compress_and_set_plan`:
/// what it logs of a compression, its sizes and its time (`Plan compression: ...`), and that it was
/// not effective. The frame is that of the `compress` of python-zstandard, of the same level, which
/// writes the size of the content: the same bytes, of the same version of zstd.
fn compress_over_threshold(plan: &mut Plan, threshold: usize) -> Vec<String> {
    let (op_type, size) = match &plan.op_type {
        Some(PlanOpType::Root(relation)) => (CompressedOpType::Relation, relation.encoded_len()),
        Some(PlanOpType::Command(command)) => (CompressedOpType::Command, command.encoded_len()),
        _ => return Vec::new(),
    };
    // The size of the encoding, without encoding the plans that are not over the threshold.
    if size <= threshold {
        return Vec::new();
    }
    let message = match &plan.op_type {
        Some(PlanOpType::Root(relation)) => relation.encode_to_vec(),
        Some(PlanOpType::Command(command)) => command.encode_to_vec(),
        _ => return Vec::new(),
    };
    let started = std::time::Instant::now();
    let Ok(data) = zstd::bulk::compress(message.as_slice(), zstd::DEFAULT_COMPRESSION_LEVEL) else {
        return Vec::new();
    };
    let duration = started.elapsed().as_secs_f64();
    // As PySpark's f-string of the ratio and the time, of Python's rounding of a float.
    let mut logged = vec![format!(
        "Plan compression: original_size={}, compressed_size={}, saving_ratio={:.2}, \
         duration_s={duration:.1}",
        message.len(),
        data.len(),
        1.0 - data.len() as f64 / message.len() as f64,
    )];
    if data.len() >= message.len() {
        logged.push("Plan compression not effective. Using original plan.".to_string());
        return logged;
    }
    plan.op_type = Some(PlanOpType::CompressedOperation(CompressedOperation {
        data: data.into(),
        op_type: op_type as i32,
        compression_codec: CompressionCodec::Zstd as i32,
    }));
    logged
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

/// An execution whose request is built, as PySpark's client builds it before its `Progress` of
/// the execution starts, and not sent yet: [`SparkConnectClient::build_execution`] builds it and
/// [`SparkConnectClient::execute_built`] sends it. Not a stable API.
#[doc(hidden)]
#[derive(Debug)]
pub struct BuiltExecution {
    request: ExecutePlanRequest,
    log: crate::client_log::ExecutionLog,
}

/// A hook of the executions of a client, as PySpark's `SparkSession.Hook.on_execute_plan`: it is
/// called with each `ExecutePlanRequest` before the client sends it, and returns the request to
/// send, which keeps the operation ID of the original.
pub type ExecutePlanHook =
    std::sync::Arc<dyn Fn(ExecutePlanRequest) -> Result<ExecutePlanRequest> + Send + Sync>;

/// The result chunking options of the executions of a client, as PySpark's `SparkConnectClient`
/// sends them with each `ExecutePlanRequest`: by default, the server may split an Arrow batch too
/// large for one message into chunks, which the stream of the execution joins, without a
/// preferred chunk size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResultChunking {
    /// Whether the server may split a large Arrow batch into chunks.
    pub allow_arrow_batch_chunking: bool,
    /// The preferred size of a chunk, in bytes, which the server uses when it is in its range.
    pub preferred_arrow_chunk_size: Option<i64>,
}

impl Default for ResultChunking {
    /// PySpark's: chunking allowed, without a preferred chunk size.
    fn default() -> Self {
        ResultChunking {
            allow_arrow_batch_chunking: true,
            preferred_arrow_chunk_size: None,
        }
    }
}

/// Append the result chunking options `chunking` to the options of `request`, as PySpark's
/// `_execute_plan_request_with_metadata` appends them.
fn add_result_chunking_option(request: &mut ExecutePlanRequest, chunking: ResultChunking) {
    use spark_connect_proto::execute_plan_request::request_option::RequestOption as Kind;
    use spark_connect_proto::execute_plan_request::RequestOption;
    request.request_options.push(RequestOption {
        request_option: Some(Kind::ResultChunkingOptions(
            spark_connect_proto::ResultChunkingOptions {
                allow_arrow_batch_chunking: chunking.allow_arrow_batch_chunking,
                preferred_arrow_chunk_size: chunking.preferred_arrow_chunk_size,
            },
        )),
    });
}

/// What PySpark's `execute_command` logs of the command of `request` before it sets the command in
/// the plan of its request (`Execute command for command ...`), or its
/// `execute_command_as_iterator`, of the command truncated.
fn logged_command(
    request: &ExecutePlanRequest,
    log: crate::client_log::ExecutionLog,
) -> crate::client_log::Message {
    use crate::client_log::{types, Encoded, ExecutionLog, Message};
    let prefix = match log {
        ExecutionLog::ExecuteCommandAsIterator => "Execute command as iterator for command ",
        _ => "Execute command for command ",
    };
    let message = match request.plan.as_ref().and_then(|plan| plan.op_type.as_ref()) {
        Some(PlanOpType::Command(command)) => Encoded::of(types::COMMAND, command),
        _ => Encoded::of(types::COMMAND, &spark_connect_proto::Command::default()),
    };
    Message::Proto {
        prefix: prefix.into(),
        message,
        truncated: true,
    }
}

/// Log what the method of PySpark's client that the execution of `request` stands for logs as it
/// is called, before it builds the request, as `log` says: the plan of `to_table` and of
/// `to_table_as_iterator`, of the plan as it is sent (`Executing plan ...`), and the command of
/// `execute_command` and of `execute_command_as_iterator`, before it is compressed (`Execute
/// command for command ...`), truncated. The error of a record fails the execution there.
async fn log_execution_call(
    request: &ExecutePlanRequest,
    log: crate::client_log::ExecutionLog,
) -> Result<()> {
    use crate::client_log::{logged, ExecutionLog, Level};
    match log {
        ExecutionLog::ToTable | ExecutionLog::ToTableAsIterator => {
            logged(Level::Debug, || {
                crate::client_log::executing_plan(request.plan.as_ref())
            })
            .await
        }
        ExecutionLog::ExecuteCommand | ExecutionLog::ExecuteCommandAsIterator => {
            logged(Level::Debug, || logged_command(request, log)).await
        }
        _ => Ok(()),
    }
}

/// Log what PySpark's client logs of the execution once its request is built, before its
/// `Progress` of the execution starts, as `log` says: the `ExecuteAndFetch` of
/// `_execute_and_fetch` or the `Execute` of `_execute`. The error of the record fails the execution
/// there, before its request is sent.
async fn log_execution(log: crate::client_log::ExecutionLog) -> Result<()> {
    use crate::client_log::{logged, ExecutionLog, Level, Message};
    let started = match log {
        ExecutionLog::ToTable | ExecutionLog::Fetch | ExecutionLog::ExecuteCommand => {
            Some("ExecuteAndFetch")
        }
        ExecutionLog::Execute => Some("Execute"),
        _ => None,
    };
    if let Some(started) = started {
        logged(Level::Debug, || Message::Text(started.to_string())).await?;
    }
    Ok(())
}

/// Log the request of an execution as `_execute_and_fetch_as_iterator` logs it as it starts, in
/// the `Progress` of the execution, before the hooks of the executions run
/// (`ExecuteAndFetchAsIterator. Request: ...`), of an execution whose responses are logged. The
/// error of the record fails the execution there, before its request is sent.
async fn log_execution_request(
    request: &ExecutePlanRequest,
    log: crate::client_log::ExecutionLog,
) -> Result<()> {
    use crate::client_log::{logged, types, Encoded, Level, Message};
    if log.logs_responses() {
        logged(Level::Debug, || Message::Proto {
            prefix: "ExecuteAndFetchAsIterator. Request: ".into(),
            message: Encoded::of(types::EXECUTE_PLAN_REQUEST, request),
            truncated: false,
        })
        .await?;
    }
    Ok(())
}

/// Log what PySpark's `schema` logs of the plan of the analysis `request` as it starts, as `log`
/// says, of the plan that `to_proto` built, compressed, truncated: `Schema for plan: ...`. The error
/// of the record fails the analysis there, before its request is sent.
async fn log_analysis(
    request: &AnalyzePlanRequest,
    log: crate::client_log::AnalysisLog,
) -> Result<()> {
    use crate::client_log::{logged, types, AnalysisLog, Encoded, Level, Message};
    let plan = match (log, request.analyze.as_ref()) {
        (AnalysisLog::Schema, Some(Analyze::Schema(schema))) => schema.plan.as_ref(),
        _ => return Ok(()),
    };
    logged(Level::Debug, || {
        let message = match plan {
            Some(plan) => Encoded::of(types::PLAN, plan),
            None => Encoded::of(types::PLAN, &Plan::default()),
        };
        Message::Proto {
            prefix: "Schema for plan: ".into(),
            message,
            truncated: true,
        }
    })
    .await
}

/// Log `response`, a response of an execution as the stream reads it, as PySpark's
/// `_execute_and_fetch_as_iterator` logs each response that it reads: the response
/// (`ExecuteAndFetchAsIterator. Response received: ...`), and what it has of the metrics, the
/// observed metrics, the schema, the result of a SQL command and an Arrow batch, or a chunk of one,
/// in that order. The joining of the chunks of a batch logs it once it is whole (`Assembling arrow
/// batch ...`). The error of a record fails the execution there.
pub(crate) async fn log_response(response: &ExecutePlanResponse) -> Result<()> {
    use crate::client_log::{logged, types, Encoded, Level, Message};
    use spark_connect_proto::execute_plan_response::ResponseType;
    if !crate::client_log::enabled(Level::Debug) {
        return Ok(());
    }
    logged(Level::Debug, || Message::Proto {
        prefix: "ExecuteAndFetchAsIterator. Response received: ".into(),
        message: Encoded::of(types::EXECUTE_PLAN_RESPONSE, response),
        truncated: false,
    })
    .await?;
    let text = |text: &str| Message::Text(text.to_string());
    if response.metrics.is_some() {
        logged(Level::Debug, || text("Received metric batch.")).await?;
    }
    if !response.observed_metrics.is_empty() {
        logged(Level::Debug, || text("Received observed metric batch.")).await?;
    }
    if response.schema.is_some() {
        logged(Level::Debug, || text("Received the schema.")).await?;
    }
    match &response.response_type {
        Some(ResponseType::SqlCommandResult(_)) => {
            logged(Level::Debug, || text("Received the SQL command result.")).await?;
        }
        Some(ResponseType::ArrowBatch(batch)) => {
            logged(Level::Debug, || {
                Message::Text(format!(
                    "Received arrow batch rows={} Number of chunks in batch={} Chunk index={} \
                     size={}",
                    batch.row_count,
                    batch.num_chunks_in_batch.unwrap_or(0),
                    batch.chunk_index.unwrap_or(0),
                    batch.data.len()
                ))
            })
            .await?;
        }
        _ => {}
    }
    Ok(())
}

/// A Spark Connect client for communicating with a remote Spark server.
///
/// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient`.
#[derive(Clone)]
pub struct SparkConnectClient {
    /// The channel of the client and its clones, of the subchannels of the process (see
    /// [`ChannelLease`]), of each generation of the process (see [`Transport`]).
    transport: std::sync::Arc<Transport>,
    /// What the channel was made of, of which a client of a new session gets its own.
    channel_factory: std::sync::Arc<ChannelFactory>,
    /// The display host of the endpoint of the channel, as `ChannelBuilder::host`.
    host: String,
    /// Session ID (UUID v4) identifying this session on the server.
    session_id: String,
    /// The session ID that the requests carry and the responses are checked against, shared by
    /// the clones of the client: `session_id` until a caller sets another, as PySpark's
    /// `client._session_id` is an attribute that its requests read (see
    /// [`SparkConnectClient::set_client_session_id`]). Those of the artifacts carry
    /// `session_id`, as PySpark's ArtifactManager keeps the ID of the client when it was built.
    client_session_id: std::sync::Arc<crate::fork::Shared<String>>,
    /// The channel builder of the connection, as PySpark's `client._builder`: of no session ID
    /// for the client of a new session, and of its own for that of a clone.
    builder: crate::channel::ChannelBuilder,
    /// User context (user_id from the channel builder), which the requests of the artifacts
    /// carry, as PySpark's ArtifactManager keeps the user of the client when it was built.
    user_id: Option<String>,
    /// The user that the other requests carry, shared by the clones of the client: `user_id`
    /// until a caller sets another, as PySpark's `client._user_id` is an attribute that its
    /// requests read (see [`SparkConnectClient::set_request_user_id`]).
    request_user_id: std::sync::Arc<crate::fork::Shared<Option<String>>>,
    /// The metadata of the connection string, and the `authorization` of its token.
    metadata: Vec<(String, String)>,
    /// The headers of every request: `metadata`, but a `user-agent` parameter (see `connect`).
    /// They are validated once, by `connect`, so that attaching them to a request cannot fail.
    headers: Vec<(AsciiMetadataKey, AsciiMetadataValue)>,
    /// The `authorization` of the token among `headers`, which the requests of a source of the
    /// metadata carry with the metadata of the source, as grpcio's call credentials of PySpark
    /// add it to the metadata of each call.
    call_credentials: Vec<(AsciiMetadataKey, AsciiMetadataValue)>,
    /// What gives the metadata of each call, and of each request of the artifacts, in place of
    /// `headers`, if anything, shared by the clones of the client (see
    /// [`SparkConnectClient::set_metadata_source`]).
    metadata_source: std::sync::Arc<crate::fork::Shared<MetadataSources>>,
    /// The user agent, which the requests that PySpark builds with `ChannelBuilder.userAgent`
    /// carry as their client type, or the error of that property, which each of them raises
    /// (see [`SparkConnectClient::client_type`]).
    user_agent: std::sync::Arc<crate::fork::Shared<Result<String>>>,
    /// The user context extensions of the requests of the client.
    user_context_extensions: std::sync::Arc<std::sync::Mutex<UserContextExtensions>>,
    /// Retry policy applied to RPCs (exponential backoff on transient failures), shared by the
    /// clones of the client, as PySpark's `set_retry_policies` replaces the policies of a client
    /// in use: each call reads the policy when it starts its retries.
    retry_policy: std::sync::Arc<crate::fork::Shared<RetryPolicy>>,
    /// What decides the retries in place of the policy, if anything, shared by the clones of the
    /// client as the policy is (see [`SparkConnectClient::set_retry_decider`]).
    retry_decider: std::sync::Arc<crate::fork::Shared<Option<std::sync::Arc<dyn RetryDecider>>>>,
    /// The deadline of each kind of RPC.
    rpc_deadlines: RpcDeadlines,
    /// The `token` parameter of the connection string, if any, which PySpark's
    /// `SparkConnectClient.token` reads before the `SPARK_CONNECT_AUTHENTICATE_TOKEN` variable.
    token: Option<String>,
    /// The result chunking options of the executions, as the `allow_arrow_batch_chunking` and
    /// `preferred_arrow_chunk_size` of PySpark's `SparkConnectClient`.
    result_chunking: ResultChunking,
    /// Whether the executions are reattachable, shared by the clones of the client, as PySpark's
    /// `use_reattachable_execute`, which `enable_reattachable_execute` and
    /// `disable_reattachable_execute` set on a client in use.
    reattachable: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The hook of the executions, if any, shared by the clients of the clones of the session,
    /// as PySpark's clones keep the hooks of a session.
    execute_plan_hook: std::sync::Arc<std::sync::RwLock<Option<ExecutePlanHook>>>,
    /// Whether the client is closed, shared by its clones: its RPCs then raise
    /// `NO_ACTIVE_SESSION`, as PySpark's `SparkConnectClient._stub` does once it is closed.
    closed: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Whether [`SparkConnectClient::close`] closed the channel of the client, shared by its
    /// clones, as PySpark's `close` closes its gRPC channel: once the client is open again (see
    /// [`SparkConnectClient::set_closed`]), its RPCs raise the `ValueError` of grpc's closed
    /// channel, and so do those of the artifacts, which PySpark's ArtifactManager sends whether
    /// the client is closed or not. Its calls in flight at the close end then (see
    /// [`ChannelClosed`]).
    channel_closed: std::sync::Arc<ChannelClosed>,
    /// The server-side session ID that the first response carried, shared by the clones of the
    /// client, as PySpark's `_server_session_id`. Each request sends it as the observed ID, and
    /// each response is checked against it.
    server_session_id: std::sync::Arc<crate::fork::Shared<Option<String>>>,
    /// Whether the client tracks its session: the session ID and the observed server-side session
    /// ID of the requests that its callers build, the check of the IDs of its responses, and its
    /// close when the server reports that the session changed. See
    /// [`SparkConnectClient::set_session_tracking`].
    session_tracking: bool,
    /// The executions in flight, shared by the clones of the client, in the order they started,
    /// each with the thread that waits for it now, if any. See
    /// [`SparkConnectClient::running_operation_id`].
    inflight_operations: std::sync::Arc<Inflight>,
    /// The releases in the background of the executions of the client, shared by its clones (see
    /// [`SparkConnectClient::wait_for_releases`]).
    releases: std::sync::Arc<Releases>,
    /// The plan compression of the client, shared by its clones, looked up on its first plan.
    plan_compression: std::sync::Arc<std::sync::Mutex<PlanCompression>>,
    /// The settings of the plan compression of the client as PySpark's client keeps them, of its
    /// lookup or of a binding, shared by its clones (see [`PlanCompressionSettings`]).
    plan_compression_settings: std::sync::Arc<std::sync::Mutex<Option<PlanCompressionSettings>>>,
}

/// Whether [`SparkConnectClient::close`] closed the channel of a client, which its clones share,
/// and the wake-up of their calls in flight then, which end, as grpcio's `close` cancels the calls
/// of its channel (see `crate::metadata::Checked`).
#[derive(Default)]
pub(crate) struct ChannelClosed {
    closed: std::sync::atomic::AtomicBool,
    /// Of each generation of the process (see [`crate::fork`]), as the calls that wait for it are
    /// of the threads of its generation.
    wake_up: crate::fork::PerGeneration<tokio::sync::Notify>,
}

impl ChannelClosed {
    /// Whether the channel is closed.
    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Mark the channel closed: the calls that start from now on see it closed.
    fn mark_closed(&self) {
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Wake the calls that wait for the close of the channel, once it is marked closed.
    fn wake_up(&self) {
        self.notify().notify_waiters();
    }

    fn notify(&self) -> &tokio::sync::Notify {
        self.wake_up.get_or_init(|_| tokio::sync::Notify::new())
    }

    /// Ends once the channel is closed: at once if it is.
    pub(crate) async fn closed(&self) {
        // Made before the mark is read, so that a close after the read wakes it.
        let woken = self.notify().notified();
        if self.is_closed() {
            return;
        }
        woken.await;
    }
}

/// An execution in flight: its operation ID, and the thread that waits for it now, if any, of the
/// watch of the waits of the call of that thread (see [`crate::runtime::wait_watch_id`]).
#[derive(Debug, Clone, PartialEq)]
struct InflightEntry {
    operation_id: String,
    waiter: Option<std::thread::ThreadId>,
    watch: Option<usize>,
}

/// The executions in flight of a client, of each generation of the process (see
/// [`crate::fork`]): in a child of a `fork()`, the executions of the parent are the parent's, which
/// the child neither waits for nor interrupts, and the lock of its own is one that no thread of
/// the parent held at the fork.
type Inflight = crate::fork::PerGeneration<std::sync::Mutex<Vec<InflightEntry>>>;

/// The executions in flight of `registry`, of the generation of the process, locked.
fn inflight_of(registry: &Inflight) -> std::sync::MutexGuard<'_, Vec<InflightEntry>> {
    registry
        .get_or_init(|_| Default::default())
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// How long the close of a session waits for the releases of its executions, as PySpark's
/// `close` waits for its release futures (`concurrent.futures.wait(..., timeout=10)`).
const RELEASES_WAIT: Duration = Duration::from_secs(10);

/// The releases in the background of the executions of a client (see
/// [`SparkConnectClient::spawn_release`]), of each generation of the process (see
/// [`crate::fork`]): a child waits for none of its parent's.
type Releases = crate::fork::PerGeneration<std::sync::Mutex<Vec<PendingRelease>>>;

/// The releases of `releases` of the generation of the process, locked, of those that ended
/// dropped.
fn releases_of(releases: &Releases) -> std::sync::MutexGuard<'_, Vec<PendingRelease>> {
    let mut pending = releases
        .get_or_init(|_| Default::default())
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    pending.retain(|release| !release.done.is_done());
    pending
}

/// A release in the background of an execution.
struct PendingRelease {
    /// Whether the stream of the execution ended, as PySpark's client keeps the release futures
    /// of an iterator once the iterator ended (`_release_futures`), which its `close` waits for.
    stream_ended: std::sync::Arc<std::sync::atomic::AtomicBool>,
    done: std::sync::Arc<ReleaseDone>,
}

/// Whether a release in the background ended, and the wake-up of its waits.
#[derive(Default)]
struct ReleaseDone {
    done: std::sync::atomic::AtomicBool,
    woken: tokio::sync::Notify,
}

impl ReleaseDone {
    fn is_done(&self) -> bool {
        self.done.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Ends once the release ended: at once if it did.
    async fn ended(&self) {
        // Made before the flag is read, so that an end after the read wakes it.
        let woken = self.woken.notified();
        if self.is_done() {
            return;
        }
        woken.await;
    }
}

/// Ends its release when it drops, with the task of the release, also unfinished, as of a
/// runtime that drops its tasks.
struct ReleaseEnds(std::sync::Arc<ReleaseDone>);

impl Drop for ReleaseEnds {
    fn drop(&mut self) {
        self.0.done.store(true, std::sync::atomic::Ordering::SeqCst);
        self.0.woken.notify_waiters();
    }
}

/// Keeps an execution in the registry of the in-flight executions of its client until it drops.
struct InflightGuard {
    registry: std::sync::Arc<Inflight>,
    operation_id: String,
    /// The thread that registered the execution, which is the caller that starts it.
    registered_on: std::thread::ThreadId,
}

impl InflightGuard {
    /// Register the execution `operation_id` in `registry` until the guard drops.
    fn register(registry: std::sync::Arc<Inflight>, operation_id: String) -> Self {
        inflight_of(&registry).push(InflightEntry {
            operation_id: operation_id.clone(),
            waiter: None,
            watch: None,
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
    registry: std::sync::Arc<Inflight>,
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
        let mut inflight = inflight_of(&self.registry);
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
    registry: std::sync::Arc<Inflight>,
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

/// Sets the thread that waits for the execution `operation_id` in `registry`, if it is in flight,
/// of the watch of the waits of the calling thread.
fn set_waiter(registry: &Inflight, operation_id: &str, waiter: Option<std::thread::ThreadId>) {
    let mut inflight = inflight_of(registry);
    if let Some(entry) = inflight
        .iter_mut()
        .rev()
        .find(|entry| entry.operation_id == operation_id)
    {
        entry.waiter = waiter;
        entry.watch = waiter.and_then(|_| crate::runtime::wait_watch_id());
    }
}

/// The hook of [`set_failure_requests_hook`].
type FailureRequestsHook = fn(&SparkConnectClient, &SparkError) -> Result<bool>;

/// The hook of [`set_failure_requests_hook`].
static FAILURE_REQUESTS_HOOK: std::sync::OnceLock<FailureRequestsHook> = std::sync::OnceLock::new();

/// Set, once for the process, the function that a client calls on the thread that converts the
/// failure of a call, an error that the server raised, which it is given, before it builds the
/// requests that it sends for it, of the details of the error and of the configs of its stack
/// trace: a binding brings the context of the requests of that thread up to date then, of the
/// attributes of its client, as PySpark builds those requests of them, and returns whether to send
/// them. When it does not, the
/// error of the server is the error of the call as it is, of no details and of the stack trace
/// that its message has; an error of the hook is the error of the call in place of it, of no
/// requests either, as PySpark raises the error of the build of the request of the details. A
/// later call does nothing. Not a stable API.
#[doc(hidden)]
pub fn set_failure_requests_hook(hook: FailureRequestsHook) {
    let _ = FAILURE_REQUESTS_HOOK.set(hook);
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
    /// The session ID, the user and the global extensions of the requests of each thread that a
    /// binding gave them (see [`SparkConnectClient::set_thread_request_context`]), which take the
    /// place of those of the client for the requests of that thread.
    request_by_thread: std::collections::HashMap<std::thread::ThreadId, ThreadRequestContext>,
    /// The call stack of the user's code of the call of each thread that a binding gave it (see
    /// [`SparkConnectClient::set_thread_call_stack`]), which the requests of its executions, its
    /// analyses and its configurations carry last of their user context extensions.
    call_stack_by_thread: std::collections::HashMap<std::thread::ThreadId, prost_types::Any>,
    /// What a binding keeps to bring the extensions up to date before the calls of a thread send
    /// requests (see [`SparkConnectClient::set_user_context_refresher`]).
    refresher: Option<UserContextRefresher>,
    /// The deadlines and the result chunking that a binding gave a thread last (see
    /// [`SparkConnectClient::set_thread_call_settings`]), which the calls of a thread that it gave
    /// no request context take, as a thread of the transport that sends the requests of a call,
    /// such as one of the fits of the classes of a OneVsRest.
    latest_call_settings: Option<(Option<RpcDeadlines>, Option<ResultChunking>)>,
}

/// The session ID, the user and the global user context extensions of the requests of a thread,
/// of the client of a binding that the thread called last (see
/// [`SparkConnectClient::set_thread_request_context`]).
#[derive(Clone, Debug)]
struct ThreadRequestContext {
    session_id: String,
    user_id: Option<String>,
    global: Vec<(String, prost_types::Any)>,
    /// The deadlines of the RPCs of the thread, in the place of the client's, of a binding (see
    /// [`SparkConnectClient::set_thread_call_settings`]).
    rpc_deadlines: Option<RpcDeadlines>,
    /// The result chunking options of the executions of the thread, in the place of the
    /// client's, of a binding (see [`SparkConnectClient::set_thread_call_settings`]).
    result_chunking: Option<ResultChunking>,
}

/// The context of the requests of a thread as it was, of
/// [`SparkConnectClient::thread_request_state`], which
/// [`SparkConnectClient::restore_thread_request_state`] gives the thread again. Not a stable API.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct ThreadRequestState {
    thread: std::thread::ThreadId,
    request: Option<ThreadRequestContext>,
    own: Option<SharedUserContextExtensions>,
    call_stack: Option<prost_types::Any>,
}

/// The opaque refresher of the user context extensions of a binding.
#[derive(Clone)]
struct UserContextRefresher(std::sync::Arc<dyn std::any::Any + Send + Sync>);

impl std::fmt::Debug for UserContextRefresher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("UserContextRefresher")
    }
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
    /// A list of the extensions `extensions`, of their IDs, which no thread has yet (see
    /// [`SparkConnectClient::share_threadlocal_user_context_extensions`]), as the list that a
    /// binding keeps of PySpark's `thread_local.user_context_extensions`. Not a stable API.
    #[doc(hidden)]
    pub fn new(extensions: Vec<(String, prost_types::Any)>) -> Self {
        Self(std::sync::Arc::new(std::sync::Mutex::new(extensions)))
    }

    /// The extensions of this list, of their IDs, in their order. Not a stable API.
    #[doc(hidden)]
    pub fn snapshot(&self) -> Vec<(String, prost_types::Any)> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Replace the extensions of this list with `extensions`, which the requests of the threads
    /// that have it carry from now on, as a change of the list of PySpark's
    /// `thread_local.user_context_extensions`. Not a stable API.
    #[doc(hidden)]
    pub fn replace(&self, extensions: Vec<(String, prost_types::Any)>) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = extensions;
    }
}

/// Forget the extensions of a thread that ends, as PySpark's `thread_local` is freed with its
/// thread.
fn forget_thread_extensions(extensions: &mut UserContextExtensions, thread: std::thread::ThreadId) {
    extensions.by_thread.remove(&thread);
    extensions.request_by_thread.remove(&thread);
    extensions.call_stack_by_thread.remove(&thread);
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

        // Not an error of the connection: PySpark reads its `userAgent` when it builds a
        // request, so the error of the user agent is that of each request that carries it.
        let user_agent = builder.user_agent();

        // Build gRPC channel. We connect *lazily* - the channel is created here but the
        // TCP/TLS connection is established on the first RPC (matching the reference
        // client, where getOrCreate() does not dial the server). This also means the
        // client can be constructed without a live server (e.g. in unit tests that only
        // build request protos).
        // The `:authority` of the calls: of a target of gRPC, gRPC's default authority of it, of
        // the host and the port of the connection string, which stand for it, where HTTP cannot
        // carry it (of a `%` but in brackets, or of more than one `:`, as of a Unix socket or of a
        // list of addresses); else the host and the port of the connection string.
        let endpoint = builder
            .target_names()
            .map(|names| names.authority.clone())
            .filter(|authority| {
                !authority.is_empty()
                    && authority
                        .parse::<tonic::codegen::http::uri::Authority>()
                        .is_ok()
            })
            .unwrap_or_else(|| builder.endpoint());
        // The connector makes the TLS session itself (see `crate::connector`), so the endpoint is
        // of plaintext, and the requests have the `https` scheme of its origin.
        let mut tonic_endpoint = Channel::from_shared(format!("http://{endpoint}"))
            .map_err(|e| SparkError::connect_msg(format!("Invalid endpoint: {}", e)))?;
        let tls = if builder.secure() {
            // TLS with the roots of GRPC_DEFAULT_SSL_ROOTS_FILE_PATH alone, as grpcio trusts them,
            // else those of the system's store and of the bundled roots file, grpcio's bundle of
            // the Python binding (see `docs/configuration.md`). The server name of an IPv6 host
            // is its address without the brackets of the URL; of a target of gRPC, it is the
            // name that gRPC's TLS verifies of its authority, never the host that stands for it.
            let tls = match builder.target_names() {
                Some(names) => crate::connector::Tls::of_target_name(names.tls_name(), ssl_roots()),
                None => crate::connector::Tls::new(
                    builder.host().trim_start_matches('[').trim_end_matches(']'),
                    ssl_roots(),
                ),
            }
            .map_err(|e| SparkError::connect_msg(format!("Failed to configure TLS: {}", e)))?;
            let origin = format!("https://{endpoint}")
                .parse()
                .map_err(|e| SparkError::connect_msg(format!("Invalid endpoint: {}", e)))?;
            tonic_endpoint = tonic_endpoint.origin(origin);
            Some(tls)
        } else {
            None
        };
        let tonic_endpoint = with_keepalive(tonic_endpoint, builder)
            .initial_connection_window_size(CONNECTION_WINDOW)
            // Far past gRPC's limits of the metadata of a response, which the channel checks.
            .http2_max_header_list_size(crate::metadata::MAX_HEADER_LIST_SIZE);
        // Through the proxy of the environment, as grpcio's channel of PySpark is.
        let proxy = proxy_of(builder).map_err(SparkError::connect_msg)?;
        let origin = format!(
            "{}://{endpoint}",
            if tls.is_some() { "https" } else { "http" }
        );
        let factory = std::sync::Arc::new(ChannelFactory {
            endpoint: tonic_endpoint,
            origin,
            keepalive: builder.keepalive(),
            own_connection: builder.token().is_some(),
            target: crate::connector::Target::new(builder.host(), builder.port(), proxy, tls)
                .of_local_credentials(
                    builder.credentials() == Some(crate::channel::Credentials::Local),
                )
                .of_channel_target(builder.target().cloned(), builder.target_names().cloned()),
        });
        let transport = std::sync::Arc::new(Transport::new(std::sync::Arc::clone(&factory)));

        // The headers are checked here, once: one that gRPC cannot send is an error of the
        // connection, rather than a header that every request silently goes without. The
        // `user-agent` header is tonic's own, as grpcio sends its own rather than the user agent,
        // which goes as the client type of the requests. grpcio replaces a `user-agent` of the
        // metadata with its own too, so a `user-agent` parameter is checked as metadata and not
        // sent, rather than put before tonic's.
        let mut headers = Vec::new();
        let mut metadata: Vec<(String, String)> = builder.metadata();
        for (key, value) in &metadata {
            let header = (metadata_key(key)?, metadata_value(key, value)?);
            // Neither the `user-agent`, of grpcio's own, nor a key that gRPC's core sets itself.
            if key != USER_AGENT_HEADER && crate::metadata::sent_of_an_application(key) {
                headers.push(header);
            }
        }
        // Spark Connect servers that require token auth expect the connection-string
        // `token` as an `Authorization: Bearer <token>` header. It is a reserved key
        // excluded from `metadata()`, so attach it explicitly here; this then propagates
        // to every request via `_attach_metadata` and the reattach stream.
        if let Some(tok) = builder.token() {
            // The token is sent over TLS, or over plaintext only on `localhost`: the scheme
            // above is `builder.secure()`, which is TLS for a token on any other host, as
            // the reference client does. Of local credentials, it is sent over a connection of
            // the loopback only (`Credentials::Local`).
            let bearer = format!("Bearer {}", tok);
            headers.push((
                AsciiMetadataKey::from_static(AUTHORIZATION_HEADER),
                token_header(&bearer)?,
            ));
            metadata.push((AUTHORIZATION_HEADER.to_string(), bearer));
        }
        let call_credentials = match builder.token() {
            Some(_) => headers.last().cloned().into_iter().collect(),
            None => Vec::new(),
        };

        let user_id = builder.user_id().map(String::from).or_else(default_user_id);
        Ok(Self {
            transport,
            channel_factory: factory,
            host: builder.host().to_string(),
            client_session_id: std::sync::Arc::new(crate::fork::Shared::new(session_id.clone())),
            session_id,
            builder: builder.clone(),
            request_user_id: std::sync::Arc::new(crate::fork::Shared::new(user_id.clone())),
            user_id,
            metadata,
            headers,
            call_credentials,
            metadata_source: Default::default(),
            user_agent: std::sync::Arc::new(crate::fork::Shared::new(user_agent)),
            user_context_extensions: Default::default(),
            retry_policy: Default::default(),
            retry_decider: Default::default(),
            rpc_deadlines: RpcDeadlines::default(),
            token: builder.get(crate::channel::PARAM_TOKEN).map(String::from),
            result_chunking: ResultChunking::default(),
            reattachable: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
            execute_plan_hook: Default::default(),
            closed: Default::default(),
            channel_closed: Default::default(),
            server_session_id: Default::default(),
            session_tracking: true,
            inflight_operations: Default::default(),
            releases: Default::default(),
            plan_compression: Default::default(),
            plan_compression_settings: Default::default(),
        })
    }

    /// Get the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The session ID that the requests of this client and of its clones carry, as PySpark's
    /// `client._session_id`: the session ID, unless
    /// [`SparkConnectClient::set_client_session_id`] set another, or the one that a binding gave
    /// the requests of the calling thread ([`SparkConnectClient::set_thread_request_context`]).
    /// Not a stable API.
    #[doc(hidden)]
    pub fn client_session_id(&self) -> String {
        if let Some(session_id) = self.thread_request(|request| request.session_id.clone()) {
            return session_id;
        }
        self.client_session_id.get()
    }

    /// `of` of the request context that a binding gave the calling thread, if it gave it one (see
    /// [`SparkConnectClient::set_thread_request_context`]).
    fn thread_request<T>(&self, of: impl FnOnce(&ThreadRequestContext) -> T) -> Option<T> {
        self.user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .request_by_thread
            .get(&crate::runtime::calling_thread_id())
            .map(of)
    }

    /// Make the requests of the calling thread ([`crate::runtime::calling_thread_id`], whose
    /// requests read it) carry the session ID `session_id`, the user `user_id` and the global user
    /// context extensions `global`, of their IDs, in their order, in the place of those of the
    /// client, until the thread ends or is given others: as a binding gives them of the client
    /// that the thread calls, which PySpark's requests read when they are built, so that the calls
    /// of two clients of the session, such as a client and its copy, carry their own, also at once
    /// on two threads. Not a stable API.
    #[doc(hidden)]
    pub fn set_thread_request_context(
        &self,
        session_id: String,
        user_id: Option<String>,
        global: Vec<(String, prost_types::Any)>,
    ) {
        let mut extensions = self
            .user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let thread = crate::runtime::calling_thread_id();
        if !extensions.by_thread.contains_key(&thread)
            && !extensions.request_by_thread.contains_key(&thread)
        {
            forget_when_thread_ends(&self.user_context_extensions, forget_thread_extensions);
        }
        // The settings of the calls of the thread stay those that the binding gave it (see
        // `set_thread_call_settings`).
        let (rpc_deadlines, result_chunking) = extensions
            .request_by_thread
            .get(&thread)
            .map(|request| (request.rpc_deadlines.clone(), request.result_chunking))
            .unwrap_or_default();
        extensions.request_by_thread.insert(
            thread,
            ThreadRequestContext {
                session_id,
                user_id,
                global,
                rpc_deadlines,
                result_chunking,
            },
        );
    }

    /// Give the calls of the calling thread the deadlines `rpc_deadlines` of their RPCs, and their
    /// executions the result chunking options `result_chunking`, in the place of those of the
    /// client, or the client's again of None, until the thread is given others: as a binding
    /// gives them of the client that the thread calls, which PySpark's requests read of its
    /// `_rpc_deadlines`, `_allow_arrow_batch_chunking` and `_preferred_arrow_chunk_size` when they
    /// are sent. An execution keeps those that it started with for its reattaches and its
    /// release, as PySpark's iterator of the execution keeps those of its client then; the
    /// requests of the artifacts keep the client's deadlines, as PySpark's `ArtifactManager` keeps
    /// those that its client was made with. A thread of no request context of a binding (see
    /// [`SparkConnectClient::set_thread_request_context`]) is given one of the session ID, the
    /// user and the global user context extensions of the client. Binding hook; not a stable API.
    #[doc(hidden)]
    pub fn set_thread_call_settings(
        &self,
        rpc_deadlines: Option<RpcDeadlines>,
        result_chunking: Option<ResultChunking>,
    ) {
        let mut extensions = self
            .user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        extensions.latest_call_settings = Some((rpc_deadlines.clone(), result_chunking));
        let thread = crate::runtime::calling_thread_id();
        if let Some(request) = extensions.request_by_thread.get_mut(&thread) {
            request.rpc_deadlines = rpc_deadlines;
            request.result_chunking = result_chunking;
            return;
        }
        if rpc_deadlines.is_none() && result_chunking.is_none() {
            return;
        }
        if !extensions.by_thread.contains_key(&thread) {
            forget_when_thread_ends(&self.user_context_extensions, forget_thread_extensions);
        }
        let global = extensions.global.clone();
        extensions.request_by_thread.insert(
            thread,
            ThreadRequestContext {
                session_id: self.client_session_id.get(),
                user_id: self.request_user_id.get(),
                global,
                rpc_deadlines,
                result_chunking,
            },
        );
    }

    /// The deadlines of the RPCs of the calling thread now: those that a binding gave it (see
    /// [`SparkConnectClient::set_thread_call_settings`]), or, of a thread of no request context of
    /// a binding, those that it gave a thread last; else the client's.
    fn call_rpc_deadlines(&self) -> RpcDeadlines {
        self.call_settings_of_thread(|(deadlines, _)| deadlines.clone())
            .unwrap_or_else(|| self.rpc_deadlines.clone())
    }

    /// Of the settings of the calls of the calling thread (see
    /// [`SparkConnectClient::call_rpc_deadlines`]), the one that `of` reads, if a binding gave it.
    fn call_settings_of_thread<T>(
        &self,
        of: impl FnOnce(&(Option<RpcDeadlines>, Option<ResultChunking>)) -> Option<T>,
    ) -> Option<T> {
        let extensions = self
            .user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match extensions
            .request_by_thread
            .get(&crate::runtime::calling_thread_id())
        {
            Some(request) => of(&(request.rpc_deadlines.clone(), request.result_chunking)),
            None => extensions.latest_call_settings.as_ref().and_then(of),
        }
    }

    /// The result chunking options of the executions of the calling thread now: those that a
    /// binding gave it (see [`SparkConnectClient::set_thread_call_settings`]), else the client's.
    fn call_result_chunking(&self) -> ResultChunking {
        self.call_settings_of_thread(|(_, chunking)| *chunking)
            .unwrap_or(self.result_chunking)
    }

    /// A clone of this client of the settings of the calls of the calling thread now, its
    /// deadlines and its result chunking (see [`SparkConnectClient::set_thread_call_settings`]),
    /// for the requests that a call sends once it returned, as the reattaches and the release of
    /// an execution, on threads that are not the caller's.
    fn of_call_settings(&self) -> Self {
        let mut client = self.clone();
        client.rpc_deadlines = self.call_rpc_deadlines();
        client.result_chunking = self.call_result_chunking();
        client
    }

    /// What the requests of the calling thread carry of the context that a binding gave it, as of
    /// now: the session ID, the user and the global user context extensions of
    /// [`SparkConnectClient::set_thread_request_context`], and the user context extensions of the
    /// thread, which [`SparkConnectClient::restore_thread_request_state`] gives the thread again.
    /// Not a stable API.
    #[doc(hidden)]
    pub fn thread_request_state(&self) -> ThreadRequestState {
        let extensions = self
            .user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let thread = crate::runtime::calling_thread_id();
        ThreadRequestState {
            thread,
            request: extensions.request_by_thread.get(&thread).cloned(),
            own: extensions.by_thread.get(&thread).cloned(),
            call_stack: extensions.call_stack_by_thread.get(&thread).cloned(),
        }
    }

    /// Give the thread of `state` the context of its requests that `state` kept (see
    /// [`SparkConnectClient::thread_request_state`]), in the place of the one that it has: as a
    /// binding gives the requests of a call back their context once a call that ran during it,
    /// such as one of a progress handler or of a finalizer, gave the thread its own. Not a stable
    /// API.
    #[doc(hidden)]
    pub fn restore_thread_request_state(&self, state: ThreadRequestState) {
        let mut extensions = self
            .user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let thread = state.thread;
        // A thread that had an entry is registered still: its registration ends with it.
        match state.request {
            Some(request) => {
                extensions.request_by_thread.insert(thread, request);
            }
            None => {
                extensions.request_by_thread.remove(&thread);
            }
        }
        match state.own {
            Some(own) => {
                extensions.by_thread.insert(thread, own);
            }
            None => {
                extensions.by_thread.remove(&thread);
            }
        }
        match state.call_stack {
            Some(call_stack) => {
                extensions.call_stack_by_thread.insert(thread, call_stack);
            }
            None => {
                extensions.call_stack_by_thread.remove(&thread);
            }
        }
    }

    /// Make the requests of the executions, the analyses and the configurations of the calling
    /// thread carry `call_stack`, the call stack of the user's code of the call that the thread
    /// runs, last of their user context extensions, or none, until the thread ends or is given
    /// another: as PySpark's requests carry the call stack of their build when
    /// `SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK` is set (`_build_call_stack_trace`), which a
    /// binding reads as each call starts. Not a stable API.
    #[doc(hidden)]
    pub fn set_thread_call_stack(&self, call_stack: Option<prost_types::Any>) {
        let mut extensions = self
            .user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let thread = crate::runtime::calling_thread_id();
        let Some(call_stack) = call_stack else {
            extensions.call_stack_by_thread.remove(&thread);
            return;
        };
        if !extensions.by_thread.contains_key(&thread)
            && !extensions.request_by_thread.contains_key(&thread)
            && !extensions.call_stack_by_thread.contains_key(&thread)
        {
            forget_when_thread_ends(&self.user_context_extensions, forget_thread_extensions);
        }
        extensions.call_stack_by_thread.insert(thread, call_stack);
    }

    /// Append the call stack of the calling thread (see
    /// [`SparkConnectClient::set_thread_call_stack`]), if it has one, to the user context
    /// extensions of a request, after those of the client and of the thread, as PySpark's
    /// `_execute_plan_request_with_metadata`, `_analyze_plan_request_with_metadata` and
    /// `_config_request_with_metadata` append the call stack that they build.
    fn add_call_stack(&self, context: &mut Option<UserContext>) {
        let extensions = self
            .user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(call_stack) = extensions
            .call_stack_by_thread
            .get(&crate::runtime::calling_thread_id())
        {
            context
                .get_or_insert_with(UserContext::default)
                .extensions
                .push(call_stack.clone());
        }
    }

    /// Make the requests of this client and of its clones carry `session_id`, against which
    /// their responses are checked, as a set of PySpark's `client._session_id` does: the server
    /// then answers of the session of that ID, and a response of another server-side session
    /// than the one that the client observed closes the client. [`SparkConnectClient::session_id`]
    /// stays the ID of the session, as the `session_id` of PySpark's session does, and the
    /// requests of the artifacts carry it, as PySpark's ArtifactManager keeps the ID of the
    /// client when it was built. A client that does not track its session (see
    /// [`SparkConnectClient::set_session_tracking`]) sends the requests that its callers build
    /// with their own session ID. Not a stable API.
    #[doc(hidden)]
    pub fn set_client_session_id(&self, session_id: String) {
        self.client_session_id.set(session_id);
    }

    /// The channel builder of the connection of this client, as PySpark's `client._builder`: of
    /// no session ID for the client of a new session, and of its own for that of a clone. Not a
    /// stable API.
    #[doc(hidden)]
    pub fn channel_builder(&self) -> &crate::channel::ChannelBuilder {
        &self.builder
    }

    /// Create a new client of this client's endpoint, metadata and proxy, of a brand-new
    /// server-side session (fresh session id). Mirrors the reference client's `newSession()`:
    /// the new client shares the connection of the process to its target, as PySpark's new client
    /// makes a channel of the subchannels of the process, reading the proxy of the environment
    /// again, and keeps the deadlines, the hook and whether the executions are reattachable, and
    /// has the default retry policy and result chunking, as PySpark's `newSession()` and `clone()`
    /// build their client of these only. Its user is the `user_id` of the connection, else the
    /// user of the requests of this client (see [`SparkConnectClient::set_request_user_id`]), as
    /// PySpark's new client takes the `_user_id` of the client that makes it.
    pub fn with_new_session_id(&self) -> Self {
        let user_id = self
            .builder
            .user_id()
            .map(String::from)
            .or_else(|| self.request_user_id());
        self.with_new_session_id_of_user(user_id)
    }

    /// As [`SparkConnectClient::with_new_session_id`], of the user `user_id` as it is given, whom
    /// the requests of the new client and those of its artifacts carry, also when the connection
    /// has a `user_id`, which PySpark's `SparkConnectClient(connection, user_id=...)` and
    /// `with_new_session_id` take in its place: the caller decides the user, as a binding does of
    /// the attributes of PySpark's client. Not a stable API.
    #[doc(hidden)]
    pub fn with_new_session_id_of_user(&self, user_id: Option<String>) -> Self {
        let session_id = Uuid::new_v4().to_string();
        // As PySpark's `newSession`, which pops the session ID of a copy of the builder.
        let mut builder = self.builder.clone();
        builder.remove(crate::channel::PARAM_SESSION_ID);
        let channel_factory = std::sync::Arc::new(self.channel_factory.renewed(&self.builder));
        let transport =
            std::sync::Arc::new(Transport::new(std::sync::Arc::clone(&channel_factory)));
        SparkConnectClient {
            transport,
            channel_factory,
            host: self.host.clone(),
            client_session_id: std::sync::Arc::new(crate::fork::Shared::new(session_id.clone())),
            session_id,
            builder,
            request_user_id: std::sync::Arc::new(crate::fork::Shared::new(user_id.clone())),
            user_id,
            metadata: self.metadata.clone(),
            headers: self.headers.clone(),
            call_credentials: self.call_credentials.clone(),
            // A client of a new session reads the metadata of its own builder, if anything.
            metadata_source: Default::default(),
            user_agent: std::sync::Arc::new(crate::fork::Shared::new(self.user_agent.get())),
            // As PySpark's client of a new session, of no user context extensions.
            user_context_extensions: Default::default(),
            retry_policy: Default::default(),
            retry_decider: Default::default(),
            rpc_deadlines: self.call_rpc_deadlines(),
            token: self.token.clone(),
            result_chunking: ResultChunking::default(),
            reattachable: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                self.reattachable_execute(),
            )),
            execute_plan_hook: std::sync::Arc::clone(&self.execute_plan_hook),
            closed: Default::default(),
            channel_closed: Default::default(),
            server_session_id: Default::default(),
            session_tracking: self.session_tracking,
            inflight_operations: Default::default(),
            releases: Default::default(),
            plan_compression: Default::default(),
            plan_compression_settings: Default::default(),
        }
    }

    /// The channel of tonic of the connection of the client, of the generation of the process (see
    /// [`Transport`]), of the tests, which call it without [`crate::metadata::Checked`].
    #[cfg(test)]
    async fn channel(&self) -> Channel {
        match self.transport.lease().channel.pick().await {
            Ok(channel) => channel,
            Err(_) => panic!("the client did not connect"),
        }
    }

    /// The channel of the client for its calls, of the generation of the process (see
    /// [`Transport`]), whose metadata is checked as gRPC's core checks it, whose resolver and
    /// `pick_first` pick the subchannel whose connection each call takes, and which the close of the
    /// client ends while they are in flight (see [`crate::metadata::Checked`]).
    fn checked_channel(&self) -> crate::metadata::Checked {
        let lease = self.transport.lease();
        crate::metadata::Checked {
            channel: std::sync::Arc::clone(&lease.channel),
            closed: std::sync::Arc::clone(&self.channel_closed),
            refused: matches!(
                self.builder.target(),
                Some(crate::channel::ChannelTarget::Refused { .. })
            ),
        }
    }

    /// The generated gRPC client stub of the channel of the client. Its message-size cap is
    /// raised from tonic's 4 MiB default (too small for real Arrow `collect()` results) to 128 MiB,
    /// as the raw Grpc calls' too: the reference client sets this unconditionally via
    /// GRPC_DEFAULT_OPTIONS (grpc.max_{send,receive}_message_length); it is not a connection-string
    /// option.
    fn stub(&self) -> Stub {
        Stub(
            SparkConnectServiceClient::new(self.checked_channel())
                .max_decoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT)
                .max_encoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT),
        )
    }

    /// Close this client, as PySpark's `SparkConnectClient.close` closes its channel: its RPCs
    /// raise `NO_ACTIVE_SESSION` from now on, and its clones with them, but those of the
    /// artifacts, which raise the `ValueError` of grpc's closed channel, as PySpark's
    /// ArtifactManager does of its own stub, and the reattaches, the retries and the releases of
    /// a reattachable execution in flight, which raise it too, as PySpark's iterator sends them
    /// through the stub that it was made with; once the client is open again
    /// ([`SparkConnectClient::set_closed`]), all of them raise that `ValueError`, as PySpark's do
    /// then. Its calls in flight, and those of its clones, end now, as grpcio's `close` cancels
    /// the calls of its channel: as cancelled, `CANCELLED`, `Channel closed!` of a call whose
    /// response has not begun, and `Stream removed (Channel closed!)` of one whose response is
    /// read (see `crate::metadata::Checked`), the releases of executions in the background too,
    /// which the close of a session waits for first (see
    /// [`SparkConnectClient::wait_for_releases`]).
    pub fn close(&self) {
        self.channel_closed.mark_closed();
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
        self.channel_closed.wake_up();
        // As grpcio's `close` of the channel: it lets go of its subchannels, whose connections end
        // once no other channel holds them.
        self.transport.lease().close();
    }

    /// Wait for the releases in the background of the executions of the client, and of its
    /// clones, whose streams ended, at most 10 s, as PySpark's `close` waits for the release
    /// futures that its iterators left it (`concurrent.futures.wait(self._release_futures,
    /// timeout=10)`) before it closes its channel, which would cancel them (see
    /// [`SparkConnectClient::close`]): a release of an execution whose stream is still read is
    /// not waited for, as PySpark's iterator keeps its own until it ends. Not a stable API.
    #[doc(hidden)]
    pub async fn wait_for_releases(&self) {
        let pending = self.releases_to_wait_for();
        if pending.is_empty() {
            return;
        }
        let ended = futures::future::join_all(pending.iter().map(|done| done.ended()));
        let _ = tokio::time::timeout(RELEASES_WAIT, ended).await;
    }

    /// Whether [`SparkConnectClient::wait_for_releases`] currently has releases to wait for. Not a
    /// stable API.
    #[doc(hidden)]
    pub fn has_releases_to_wait_for(&self) -> bool {
        !self.releases_to_wait_for().is_empty()
    }

    /// The releases in the background, not ended, of the streams of the client that ended.
    fn releases_to_wait_for(&self) -> Vec<std::sync::Arc<ReleaseDone>> {
        releases_of(&self.releases)
            .iter()
            .filter(|release| {
                release
                    .stream_ended
                    .load(std::sync::atomic::Ordering::SeqCst)
            })
            .map(|release| std::sync::Arc::clone(&release.done))
            .collect()
    }

    /// Mark this client closed, or open, as a set of PySpark's `client._closed`, without closing
    /// its channel: a closed client raises `NO_ACTIVE_SESSION` for its RPCs but those of the
    /// artifacts, which PySpark's ArtifactManager sends of its own stub, and an open one sends
    /// them, unless [`SparkConnectClient::close`] closed its channel. Not a stable API.
    #[doc(hidden)]
    pub fn set_closed(&self, closed: bool) {
        self.closed
            .store(closed, std::sync::atomic::Ordering::SeqCst);
    }

    /// Open this client again, as `set_closed(false)`. Not a stable API.
    #[doc(hidden)]
    pub fn reopen(&self) {
        self.set_closed(false);
    }

    /// The user that the requests of this client and of its clones carry, but those of the
    /// artifacts, as PySpark's `client._user_id`: the user of the connection, unless
    /// [`SparkConnectClient::set_request_user_id`] set another, or the one that a binding gave the
    /// requests of the calling thread ([`SparkConnectClient::set_thread_request_context`]). Not a
    /// stable API.
    #[doc(hidden)]
    pub fn request_user_id(&self) -> Option<String> {
        if let Some(user_id) = self.thread_request(|request| request.user_id.clone()) {
            return user_id;
        }
        self.request_user_id.get()
    }

    /// Make the requests of this client and of its clones carry the user `user_id`, or none, as
    /// a set of PySpark's `client._user_id` does, which its requests read. The requests of the
    /// artifacts carry the user of the client when it was made, as PySpark's ArtifactManager
    /// keeps it. Not a stable API.
    #[doc(hidden)]
    pub fn set_request_user_id(&self, user_id: Option<String>) {
        self.request_user_id.set(user_id);
    }

    /// Whether the client is closed. See [`SparkConnectClient::close`].
    pub fn is_closed(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// The operation ID of the innermost execution that the calling thread is waiting for, if any:
    /// the one that a wait of the calling thread is for, as PySpark interrupts the operation of
    /// the iterator that is interrupted. An execution in flight that nothing waits for, such as
    /// an iterator that is not consumed, is not one, nor is one whose failure the thread converts
    /// now (see [`converting_failure`]), as for a call that a handler of a log of the conversion
    /// makes: PySpark converts the failure in the `except Exception` clause of the execution, so
    /// the exception of a signal during that call goes out of the conversion and interrupts none
    /// of it.
    pub fn running_operation_id(&self) -> Option<String> {
        let caller = crate::runtime::calling_thread_id();
        let converted = converted_operation_ids(caller);
        let inflight = inflight_of(&self.inflight_operations);
        inflight
            .iter()
            .rev()
            .find(|entry| entry.waiter == Some(caller) && !converted.contains(&entry.operation_id))
            .map(|entry| entry.operation_id.clone())
    }

    /// The identity of the client and of its clones, which share its state, as PySpark's session
    /// has one client: not of the client of another session (see
    /// [`SparkConnectClient::with_new_session_id`]).
    fn identity(&self) -> usize {
        std::sync::Arc::as_ptr(&self.inflight_operations) as *const () as usize
    }

    /// Whether the calling thread converts the failure of a call of this client now, as the
    /// `inside_error_handling` of the `thread_local` of PySpark's client tells it while its
    /// `_handle_error` runs: the failure of a call of the client that runs meanwhile, as of a
    /// handler of a log of the conversion, is not converted, and the error of such a call has no
    /// operation ID, as that `_handle_error` raises the error of the call as it is. Not a stable
    /// API.
    ///
    /// A call runs during a conversion when it runs within a poll of the conversion (see
    /// [`crate::runtime::polling_conversion`]), as a call of a handler of one of its records, or,
    /// of a conversion of a call that a watch waits for (see [`crate::runtime::with_wait_watch`]),
    /// which blocks its thread until it ends, of the Python binding, when it runs on the thread at
    /// all, as a call of a signal handler that a poll of the watch runs. A call of another task
    /// of an async caller of the client, which the thread polls between two polls of the
    /// conversion, does not.
    #[doc(hidden)]
    pub fn inside_error_handling(&self) -> bool {
        let client = self.identity();
        if crate::runtime::polls_conversion_of(client) {
            return true;
        }
        let thread = crate::runtime::calling_thread_id();
        converting()
            .iter()
            .any(|mark| mark.thread == thread && mark.client == client && mark.watch.is_some())
    }

    /// The operation ID of the innermost execution that the calling thread is waiting for within
    /// the call whose watch it runs (see [`crate::runtime::wait_watch_id`]), if any: not of the
    /// execution of another call of the thread that a call during it, as of a handler of a log,
    /// waits within, as PySpark's `_handle_error` of a call has the operation ID of that call.
    fn operation_id_of_this_call(&self) -> Option<String> {
        let caller = crate::runtime::calling_thread_id();
        let watch = crate::runtime::wait_watch_id();
        let inflight = inflight_of(&self.inflight_operations);
        inflight
            .iter()
            .rev()
            .find(|entry| entry.waiter == Some(caller) && entry.watch == watch)
            .map(|entry| entry.operation_id.clone())
    }

    /// The error of a unary RPC of a closed client, as PySpark's `SparkConnectClient._stub` raises
    /// it, or of one whose channel is closed (see [`SparkConnectClient::close`]).
    #[allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API
    fn ensure_open(&self) -> Result<()> {
        if self.is_closed() {
            return Err(SparkError::classed(
                crate::error::SparkErrorKind::Connect,
                "NO_ACTIVE_SESSION",
                &[],
            ));
        }
        self.ensure_channel_open()
    }

    /// Whether [`SparkConnectClient::close`] closed the channel of this client.
    fn channel_is_closed(&self) -> bool {
        self.channel_closed.is_closed()
    }

    /// The `ValueError` of a server-streaming RPC on a closed channel, as grpc raises it once
    /// PySpark's `close` closed it, whether the client is closed or not: the reattaches and the
    /// retries of a reattachable execution check it only, as PySpark's iterator sends them through
    /// the stub that it was made with rather than the `_stub` of the client.
    #[allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API
    fn ensure_channel_open_for_stream(&self) -> Result<()> {
        if self.channel_is_closed() {
            return Err(SparkError::builtin_value_msg(
                "Cannot invoke RPC: Channel closed!",
            ));
        }
        Ok(())
    }

    /// As [`SparkConnectClient::ensure_open`], for a server-streaming RPC, `ExecutePlan` and
    /// `ReattachExecute`, whose call grpc refuses of another message on a closed channel.
    #[allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API
    fn ensure_open_for_stream(&self) -> Result<()> {
        if self.channel_is_closed() && !self.is_closed() {
            return Err(SparkError::builtin_value_msg(
                "Cannot invoke RPC: Channel closed!",
            ));
        }
        self.ensure_open()
    }

    /// The `ValueError` of a unary or a client-streaming RPC on a closed channel, as grpc raises it
    /// once PySpark's `close` closed it, whether the client is closed or not: the RPCs of the
    /// artifacts check it only, as PySpark's ArtifactManager calls its own stub.
    #[allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API
    fn ensure_channel_open(&self) -> Result<()> {
        if self.channel_is_closed() {
            return Err(SparkError::builtin_value_msg(
                "Cannot invoke RPC on closed channel!",
            ));
        }
        Ok(())
    }

    /// The server-side session ID that the responses have carried, if one has, as PySpark's
    /// `_server_session_id`: None before the first response.
    pub fn observed_server_session_id(&self) -> Option<String> {
        self.server_session_id.get()
    }

    /// Set the server-side session ID that the requests observe, as PySpark's
    /// `_server_session_id` is settable: a response that carries another one closes the client.
    pub fn set_observed_server_session_id(&self, id: Option<String>) {
        self.server_session_id.set(id);
    }

    /// The display host of the endpoint of the channel, as PySpark's `SparkConnectClient.host`.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Sets the session ID of a request that the caller built to the client's, as PySpark's
    /// requests read `client._session_id`. The requests of a client that does not track its
    /// session, those of the transport-injection stub, are of the reference client's session and
    /// keep its ID, as they keep their observed one (see `observe_on`).
    fn session_on(&self, session_id: &mut String) {
        if self.session_tracking {
            *session_id = self.client_session_id();
        }
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
    /// carries is observed; a later one that differs marks the client closed, as the server
    /// restarted, and is an error. As PySpark's `_closed = True` of then, the channel stays open,
    /// so the artifacts are sent and the client sends its requests again once it is open (see
    /// [`SparkConnectClient::set_closed`]). An empty ID is neither observed nor checked. PySpark
    /// keeps an empty first ID and then raises for a later non-empty one.
    #[allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API
    fn verify_response(&self, session_id: &str, server_side_session_id: &str) -> Result<()> {
        if !self.session_tracking {
            return Ok(());
        }
        // An empty session ID is none: the server leaves both IDs unset in some responses, such
        // as those of FetchErrorDetails, so only another session's ID is a mismatch.
        let expected = self.client_session_id();
        if !session_id.is_empty() && session_id != expected {
            return Err(assertion_error(format!(
                "Received incorrect session identifier for request:{session_id} != {expected}"
            )));
        }
        if server_side_session_id.is_empty() {
            return Ok(());
        }
        // Read without a lock, as each response is, and set by the first response that carries one.
        let observed = |observed: &Option<String>| observed.clone();
        let old = match self.server_session_id.read(observed) {
            None => {
                let first = Some(server_side_session_id.to_string());
                if self
                    .server_session_id
                    .update(|observed| observed.is_none().then(|| first.clone()))
                {
                    return Ok(());
                }
                // Another response was first.
                self.server_session_id.read(observed)
            }
            old => old,
        };
        match old {
            Some(old) if server_side_session_id != old => {
                self.set_closed(true);
                Err(assertion_error(format!(
                    "Received incorrect server side session identifier for request. Please \
                     create a new Spark Session to reconnect. ({server_side_session_id} != {old})"
                )))
            }
            _ => Ok(()),
        }
    }

    /// The gRPC failure of a call of the kind `call` as a [`SparkError`], which marks the client
    /// closed when the server reports that the session changed, of an open channel (see
    /// [`SparkConnectClient::verify_response`]), and disables its plan compression when the server
    /// cannot parse a plan, as PySpark's `_handle_rpc_error` does.
    fn grpc_error(&self, status: tonic::Status, call: GrpcCall) -> SparkError {
        let read = crate::connector::grpcio_dialect()
            .then(|| crate::rpc_status::Details::of(status.details()));
        self.grpc_error_read(status, call, read)
    }

    /// [`SparkConnectClient::grpc_error`], of `read`, the details of `status` as upb reads them in
    /// the dialect of grpcio, which the other readers of the failure share, or none out of it.
    fn grpc_error_read(
        &self,
        status: tonic::Status,
        call: GrpcCall,
        read: Option<crate::rpc_status::Details>,
    ) -> SparkError {
        let error = self.session_checked_grpc_error_read(status, call, read);
        if error.get_condition().as_deref() == Some("CONNECT_INVALID_PLAN.CANNOT_PARSE") {
            self.keep_plan_compression(PlanCompression::Disabled);
        }
        error
    }

    /// The gRPC failure of a call of the kind `call` as a [`SparkError`], of `read`, the details
    /// of `status` as upb reads them in the dialect of grpcio, or none out of it, which marks the
    /// client closed when the server reports that the session changed (see
    /// [`SparkConnectClient::grpc_error`]).
    fn session_checked_grpc_error_read(
        &self,
        status: tonic::Status,
        call: GrpcCall,
        read: Option<crate::rpc_status::Details>,
    ) -> SparkError {
        let error = SparkError::from_grpc_status_read(status, call, read);
        if self.session_tracking
            && error.get_condition().as_deref() == Some("INVALID_HANDLE.SESSION_CHANGED")
        {
            self.set_closed(true);
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
        let thread = crate::runtime::calling_thread_id();
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
        let thread = crate::runtime::calling_thread_id();
        if !all.by_thread.contains_key(&thread) {
            forget_when_thread_ends(&self.user_context_extensions, forget_thread_extensions);
        }
        all.by_thread.insert(thread, extensions.clone());
    }

    /// Keep `refresher`, or none, for this client and its clones: what a binding keeps to bring
    /// the user context extensions up to date before a call of a thread sends requests, as the
    /// Python client re-serializes the `google.protobuf.Any` messages of its lists, which PySpark
    /// serializes when it builds each request. The core never calls it. Not a stable API.
    #[doc(hidden)]
    pub fn set_user_context_refresher(
        &self,
        refresher: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
    ) {
        self.user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .refresher = refresher.map(UserContextRefresher);
    }

    /// The refresher of [`SparkConnectClient::set_user_context_refresher`], if any. Not a stable
    /// API.
    #[doc(hidden)]
    pub fn user_context_refresher(
        &self,
    ) -> Option<std::sync::Arc<dyn std::any::Any + Send + Sync>> {
        self.user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .refresher
            .clone()
            .map(|refresher| refresher.0)
    }

    /// Forget the user context extensions of the current thread, which has none from now on,
    /// as PySpark's `thread_local` once its `user_context_extensions` is deleted. Not a stable
    /// API.
    #[doc(hidden)]
    pub fn clear_threadlocal_user_context_extensions(&self) {
        self.user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .by_thread
            .remove(&crate::runtime::calling_thread_id());
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

    /// Replace the global user context extensions with `extensions`, of their IDs, in their order,
    /// as PySpark's requests take the list of `global_user_context_extensions` of a client, which
    /// its caller may change or rebind.
    pub fn set_global_user_context_extensions(&self, extensions: Vec<(String, prost_types::Any)>) {
        self.user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .global = extensions;
    }

    /// Remove the extension of `extension_id`, one of the current thread or a global one, as
    /// PySpark's `remove_user_context_extension`.
    pub fn remove_user_context_extension(&self, extension_id: &str) {
        let mut extensions = self
            .user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if extension_id.starts_with("threadlocal_") {
            if let Some(own) = extensions
                .by_thread
                .get_mut(&crate::runtime::calling_thread_id())
            {
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
        extensions
            .by_thread
            .remove(&crate::runtime::calling_thread_id());
        extensions.global.clear();
    }

    /// The user context extensions of the current thread, of their IDs, as PySpark's
    /// `thread_local.user_context_extensions` of a client.
    pub fn threadlocal_user_context_extensions(&self) -> Vec<(String, prost_types::Any)> {
        self.user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .by_thread
            .get(&crate::runtime::calling_thread_id())
            .map(SharedUserContextExtensions::snapshot)
            .unwrap_or_default()
    }

    /// The global user context extensions, of their IDs, as PySpark's
    /// `global_user_context_extensions` of a client: those that the requests of the calling
    /// thread carry, of a binding (see [`SparkConnectClient::set_thread_request_context`]), else
    /// those of the client.
    pub fn global_user_context_extensions(&self) -> Vec<(String, prost_types::Any)> {
        let extensions = self
            .user_context_extensions
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        extensions
            .request_by_thread
            .get(&crate::runtime::calling_thread_id())
            .map_or(&extensions.global, |request| &request.global)
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
        let caller = crate::runtime::calling_thread_id();
        let own = extensions
            .by_thread
            .get(&caller)
            .map(|own| own.0.lock().unwrap_or_else(|e| e.into_inner()));
        // The global extensions that a binding gave the thread, else those of the client.
        let global = extensions
            .request_by_thread
            .get(&caller)
            .map_or(&extensions.global, |request| &request.global);
        if global.is_empty() && own.as_ref().is_none_or(|own| own.is_empty()) {
            return;
        }
        let context = context.get_or_insert_with(UserContext::default);
        context.extensions.extend(
            global
                .iter()
                .chain(own.iter().flat_map(|own| own.iter()))
                .map(|(_, extension)| extension.clone()),
        );
    }

    /// Replace the retry policy of this client's RPCs, which its clones share: a clone of the
    /// client (`Clone`) retries under the policy that this sets, and this client under the one
    /// that a clone sets, from their next calls on (see
    /// [`SparkConnectClient::replace_retry_policy`]). A client of a new session
    /// ([`SparkConnectClient::with_new_session_id`], `clone_session`) starts with the default
    /// policy, as PySpark's `newSession` and `clone` do, and so does not share it. To give one
    /// client a policy of its own, connect it ([`SparkConnectClient::connect`]) rather than
    /// cloning another.
    ///
    /// The transport-injection stub sets this to [`RetryPolicy::no_retries`] so it behaves
    /// like the single-shot grpcio stub the reference client expects (which does its own
    /// retrying on top); see that constructor for why double-retrying is wrong there. The stub
    /// connects a client of its own, so the policy is not shared with any other.
    pub fn set_retry_policy(&mut self, policy: RetryPolicy) {
        self.replace_retry_policy(policy);
    }

    /// Replace the retry policy of this client and of its clones while they are in use, as
    /// PySpark's `SparkConnectClient.set_retry_policies`: each call reads the policy when it
    /// starts its retries, so the calls that start from now on retry under `policy`.
    pub fn replace_retry_policy(&self, policy: RetryPolicy) {
        self.retry_policy.set(policy);
    }

    /// The retry policy of this client's RPCs.
    pub fn retry_policy(&self) -> RetryPolicy {
        self.retry_policy.get()
    }

    /// Make `decider` decide the retries of the calls of this client and of its clones that start
    /// from now on, in place of its retry policy, or the policy again with none: as the Python
    /// binding does of PySpark's retry policies of methods of their own, whose `Retrying` it runs
    /// (see [`RetryDecider`]). The calls that wait for their retries and the requests sent in the
    /// background have it alike, as PySpark's `Retrying` of them is of the policies of the
    /// client. A client of a new session has none. Not a stable API.
    #[doc(hidden)]
    pub fn set_retry_decider(&self, decider: Option<std::sync::Arc<dyn RetryDecider>>) {
        self.retry_decider.set(decider);
    }

    /// Give the calls of this client and of its clones, from now on, the metadata of `calls`, and
    /// the requests of the artifacts that of `artifacts`, in place of the metadata of the builder,
    /// with the `authorization` of the token, or that of the builder again with none: as the
    /// Python binding does of a channel builder whose `metadata()` is its own, which PySpark's
    /// client calls for each attempt of a call, and once as it is made, of which its
    /// ArtifactManager sends its requests. The requests of a reattachable execution, and the
    /// release of its results, carry the metadata of `calls` as the execution started, as
    /// PySpark's iterator is made of a `metadata()`, and a request sent in the background, of no
    /// caller, the metadata of the builder. A client of a new session has none. Not a stable API.
    #[doc(hidden)]
    pub fn set_metadata_source(
        &self,
        calls: Option<std::sync::Arc<dyn crate::channel::MetadataSource>>,
        artifacts: Option<std::sync::Arc<dyn crate::channel::MetadataSource>>,
    ) {
        self.metadata_source
            .set(MetadataSources { calls, artifacts });
    }

    /// The retries of a call of the kind `call` that starts now: of the decider of the client, if
    /// it has one, else of its policy.
    fn retries_of(&self, call: GrpcCall) -> Retrying {
        let decider = self.retry_decider.get();
        Retrying::new(&self.retry_policy()).of_call(call, decider.as_ref())
    }

    /// The `token` parameter of the connection string, if it has one. PySpark's
    /// `SparkConnectClient.token` is this parameter, else the `SPARK_CONNECT_AUTHENTICATE_TOKEN`
    /// variable of the environment as it is when the token is read.
    pub fn token_parameter(&self) -> Option<&str> {
        self.token.as_deref()
    }

    /// Set the result chunking options of this client's executions, as the
    /// `allow_arrow_batch_chunking` and `preferred_arrow_chunk_size` of PySpark's
    /// `SparkConnectClient`.
    pub fn set_result_chunking(&mut self, chunking: ResultChunking) {
        self.result_chunking = chunking;
    }

    /// The result chunking options of this client's executions.
    pub fn result_chunking(&self) -> ResultChunking {
        self.result_chunking
    }

    /// Make the executions of this client and of its clones reattachable or not, from the next
    /// one on, as PySpark's `enable_reattachable_execute` and `disable_reattachable_execute`. A
    /// reattachable execution, the default, asks the server to keep its results until they are
    /// released, so that a dropped stream is resumed with `ReattachExecute`; one that is not is a
    /// single `ExecutePlan`, sent again from the start when its stream fails in a way that the
    /// retry policy retries, and never released.
    pub fn set_reattachable_execute(&self, reattachable: bool) {
        self.reattachable
            .store(reattachable, std::sync::atomic::Ordering::SeqCst);
    }

    /// Whether the executions of this client are reattachable. See
    /// [`SparkConnectClient::set_reattachable_execute`].
    pub fn reattachable_execute(&self) -> bool {
        self.reattachable.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Turn the tracking of the session on (the default) or off. With it on, the requests that
    /// the callers build carry the session ID of the client (see
    /// [`SparkConnectClient::client_session_id`]) and the server-side session ID observed in the
    /// responses, each response must be of that session and of that server-side session, and an
    /// `INVALID_HANDLE.SESSION_CHANGED` error closes the client, as PySpark's
    /// `SparkConnectClient` does.
    ///
    /// The transport-injection stub turns it off. The requests that it sends are of the
    /// reference client's session, not of its own, and keep their IDs, and that client tracks
    /// its session itself, over the grpcio stub that does no tracking.
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
    /// as PySpark's `_get_plan_compression_threshold_and_algorithm` does. The error is one that
    /// its `except Exception` does not catch (see [`crate::retries::RetriesListener::caught`]).
    async fn plan_compression(&self) -> Result<PlanCompression> {
        let known = *self
            .plan_compression
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if known != PlanCompression::Unknown {
            return Ok(known);
        }
        // As PySpark's `except Exception` of the lookup, which catches the exception that a signal
        // handler raises while the lookup waits, unless it is a `BaseException` only, such as the
        // KeyboardInterrupt of Ctrl-C: then the settings are those of a failed lookup, kept for the
        // later plans, and the call goes on (see `runtime::catching`).
        // The settings as PySpark's lookup sets them, of the threshold as an int and the algorithm
        // as it is, of a lookup whose threshold is one.
        let mut read = None;
        let looked_up = match crate::runtime::catching(self.get_configs(&[
            PLAN_COMPRESSION_THRESHOLD_CONF,
            PLAN_COMPRESSION_ALGORITHM_CONF,
        ]))
        .await
        {
            None => PlanCompression::Disabled,
            Some(Ok(values)) => match values.as_slice() {
                [threshold, algorithm] => {
                    read = plan_compression_threshold(threshold.as_deref()).map(|threshold| {
                        PlanCompressionSettings {
                            threshold,
                            algorithm: algorithm.clone(),
                        }
                    });
                    // As PySpark's lookup, which logs the settings that it read, or the error of a
                    // threshold that is not an integer, which disables the compression.
                    let (looked_up, logged) =
                        looked_up_plan_compression(threshold.as_deref(), algorithm.as_deref());
                    let of_error = match &logged {
                        crate::client_log::Message::WithError { error, .. } => {
                            Some(error.as_ref().clone())
                        }
                        _ => None,
                    };
                    match crate::client_log::logged(crate::client_log::Level::Debug, || logged)
                        .await
                    {
                        Ok(()) => looked_up,
                        Err(error) => match &of_error {
                            // The log of the error of the threshold is in PySpark's `except
                            // Exception as e`: its error is raised, of the compression disabled,
                            // of the error of the threshold as its context.
                            Some(caught) => {
                                self.keep_plan_compression(PlanCompression::Disabled);
                                return Err(crate::retries::raised_in_handling(
                                    error,
                                    crate::retries::Handled::Error(caught),
                                ));
                            }
                            // The log of the settings is in its `try`, whose `except Exception`
                            // catches the error of a handler of the log, as a failed lookup.
                            None => {
                                return self
                                    .plan_compression_after_error(error, Some((looked_up, read)))
                                    .await
                            }
                        },
                    }
                }
                _ => PlanCompression::Disabled,
            },
            Some(Err(error)) => {
                // An aborted lookup is not the answer of the server: nothing is kept, so the
                // lookup is made again by the next plan.
                if is_wait_aborted(&error) {
                    if let Some(error) = crate::retries::caught(error) {
                        return Err(error);
                    }
                    return Ok(PlanCompression::Disabled);
                }
                return self.plan_compression_after_error(error, None).await;
            }
        };
        self.keep_plan_compression_of(looked_up, read);
        Ok(looked_up)
    }

    /// Keep `looked_up` as the plan compression of this client, for its later plans, of the
    /// settings that it stands for (see [`PlanCompressionSettings::of`]).
    fn keep_plan_compression(&self, looked_up: PlanCompression) {
        self.keep_plan_compression_of(looked_up, None);
    }

    /// Keep `looked_up` as the plan compression of this client, of the settings `read` of its
    /// lookup, else of those that it stands for.
    fn keep_plan_compression_of(
        &self,
        looked_up: PlanCompression,
        read: Option<PlanCompressionSettings>,
    ) {
        *self
            .plan_compression
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = looked_up;
        *self
            .plan_compression_settings
            .lock()
            .unwrap_or_else(|e| e.into_inner()) =
            read.or_else(|| PlanCompressionSettings::of(looked_up));
    }

    /// The settings of the plan compression of this client, as PySpark's client keeps them (see
    /// [`PlanCompressionSettings`]): of its lookup, of the first plan, or of
    /// [`SparkConnectClient::set_plan_compression_settings`], and None before. Binding hook; not
    /// a stable API.
    #[doc(hidden)]
    pub fn plan_compression_settings(&self) -> Option<PlanCompressionSettings> {
        self.plan_compression_settings
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Make `settings` the plan compression of this client and of its clones, as a binding sets
    /// PySpark's `_plan_compression_threshold` and `_plan_compression_algorithm` of its client,
    /// which its plans read: of a threshold that is not negative and the algorithm `ZSTD`, the
    /// plans of more bytes than the threshold are compressed, and of other settings none is; of
    /// None, or of no algorithm, the settings are looked up again with the next plan, as
    /// PySpark's client looks them up while either is None. Binding hook; not a stable API.
    #[doc(hidden)]
    pub fn set_plan_compression_settings(&self, settings: Option<PlanCompressionSettings>) {
        let compression = match &settings {
            Some(PlanCompressionSettings {
                threshold,
                algorithm: Some(algorithm),
            }) => plan_compression_above(*threshold, Some(algorithm.as_str())),
            _ => PlanCompression::Unknown,
        };
        *self
            .plan_compression
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = compression;
        *self
            .plan_compression_settings
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = settings;
    }

    /// The plan compression after `error`, of the lookup of its settings or of the log of them, as
    /// the `except Exception` of PySpark's `_get_plan_compression_threshold_and_algorithm` catches
    /// it: the error of a lookup, the exception of the warning of its spent retries or of a policy,
    /// or that of a handler of the log of the settings, `read` the settings that it logged. The
    /// compression is disabled, and the exception that it caught is logged, as the argument of a
    /// text of no placeholder for it, whose error is raised, as the log is in the `except` clause.
    /// An error that is not caught, of a `BaseException` only, as the `KeyboardInterrupt` of a
    /// policy's Python or of a handler, is raised: of the settings that the log read, which
    /// PySpark has set before it logs them, else of none kept, so that the next plan looks them up.
    async fn plan_compression_after_error(
        &self,
        error: SparkError,
        read: Option<(PlanCompression, Option<PlanCompressionSettings>)>,
    ) -> Result<PlanCompression> {
        // Of the error as it is, before it is caught, of which the listener logs only one of an
        // `Exception`. The error of the log has the error that it logs as its context, as the log
        // is in the `except Exception as e` of PySpark's lookup.
        let logged = crate::client_log::logged(crate::client_log::Level::Debug, || {
            crate::client_log::Message::WithError {
                text: PLAN_COMPRESSION_UNSUPPORTED,
                error: Box::new(error.clone()),
            }
        })
        .await
        .map_err(|raised| {
            crate::retries::raised_in_handling(raised, crate::retries::Handled::Error(&error))
        });
        if let Some(error) = crate::retries::caught(error) {
            if let Some((read, settings)) = read {
                self.keep_plan_compression_of(read, settings);
            }
            return Err(error);
        }
        self.keep_plan_compression(PlanCompression::Disabled);
        logged?;
        Ok(PlanCompression::Disabled)
    }

    /// Read the server's plan compression settings, unless this client has them, as PySpark's
    /// client reads them when it builds its first plan (`to_proto`), before it sends the plan.
    /// The plans that this client sends afterwards are compressed with them without another
    /// read. A caller that reads settings of its own between building a plan and sending it, as
    /// PySpark's `toPandas` does, reads these first; any other execution reads them as it sends
    /// its plan.
    /// The error is one that the `except Exception` of PySpark's lookup does not catch, as the
    /// KeyboardInterrupt of the Python of a retry policy.
    pub async fn read_plan_compression(&self) -> Result<()> {
        self.plan_compression().await.map(drop)
    }

    /// The plan compression of this client once its settings are read, of the first plan, and
    /// None before: the threshold over which the plans of relations and of commands are
    /// compressed, the size of their encoding, or None when the compression is disabled. A caller
    /// that encodes a plan outside the core tells with it that the core would send the plan as it
    /// is (see [`SparkConnectClient::prepare_plan`]). Not a stable API.
    #[doc(hidden)]
    pub fn known_plan_compression(&self) -> Option<Option<usize>> {
        match *self
            .plan_compression
            .lock()
            .unwrap_or_else(|e| e.into_inner())
        {
            PlanCompression::Unknown => None,
            PlanCompression::Disabled => Some(None),
            PlanCompression::Zstd { threshold } => Some(Some(threshold)),
        }
    }

    /// Compress `plan` as the plan compression of this client asks, after the lookup of its
    /// settings of the first plan, as PySpark's `to_proto` builds a plan, with what it logs of
    /// them. The compression is tried once: a plan prepared so is sent as it is, compressed or
    /// not, with [`SparkConnectClient::execute_plan_reattachable_as_is`]. A caller that reads
    /// settings of its own between building a plan and sending it, as PySpark's `toPandas` and
    /// `toLocalIterator` do, prepares its plan first. The error is that of
    /// [`SparkConnectClient::read_plan_compression`], or of a record of the compression. Not a
    /// stable API.
    #[doc(hidden)]
    pub async fn prepare_plan(&self, plan: &mut Plan) -> Result<()> {
        self.compress_plan(plan).await
    }

    /// Compress `plan` as the plan compression of this client asks, with what PySpark's
    /// `_try_compress_and_set_plan` logs of it, whose error fails the call there.
    async fn compress_plan(&self, plan: &mut Plan) -> Result<()> {
        if let PlanCompression::Zstd { threshold } = self.plan_compression().await? {
            for logged in compress_over_threshold(plan, threshold) {
                crate::client_log::logged(crate::client_log::Level::Debug, || {
                    crate::client_log::Message::Text(logged)
                })
                .await?;
            }
        }
        Ok(())
    }

    /// `request`, the request of an execution, as PySpark's client sends it (see
    /// [`SparkConnectClient::unstarted_execution_request`] and
    /// [`SparkConnectClient::started_execution_request`]).
    async fn execution_request(
        &self,
        request: ExecutePlanRequest,
        compressed: bool,
        log: crate::client_log::ExecutionLog,
    ) -> Result<ExecutePlanRequest> {
        let request = self
            .unstarted_execution_request(request, compressed, log)
            .await?;
        self.started_execution_request(request, log).await
    }

    /// `request`, the request of an execution, as PySpark's client builds it before its
    /// `Progress` of the execution starts: of its client type, its result chunking options, its
    /// user context, the server-side session ID that it observes, and its plan compressed as the
    /// plan compression of this client asks when `compressed`.
    ///
    /// PySpark builds the plan of a relation, with the lookup of the plan compression of its
    /// first plan, before the request (`to_proto`), which then observes the server-side session
    /// ID that the response of the lookup gives. It builds the request of a command before it sets
    /// the command in its plan (`execute_command`), so the request of the first command of a
    /// session observes none. A plan that is not `compressed` is set as it is, without the lookup,
    /// as the registrations of PySpark's `_execute` set their command, and as PySpark's client
    /// sends the plan that it is given, which `to_proto` built and compressed.
    ///
    /// It logs what PySpark's client logs of the execution then, as `log` says (see
    /// [`ExecutionLog`](crate::client_log::ExecutionLog)): a command before the request is built
    /// and the command compressed, as `execute_command` logs it before it builds the request and
    /// sets the command in its plan; a plan of a relation once it is compressed, before the request
    /// is built, as `to_table` logs the plan that `to_proto` built; and the `ExecuteAndFetch` of
    /// `_execute_and_fetch`. The error of a record fails the execution there. A closed client
    /// raises once the records are written, when the request is sent (see
    /// [`SparkConnectClient::started_execution_request`]), as PySpark's client reads its `_stub`
    /// only then: a compression of the plan, of the first plan of the client, looks its settings up
    /// first, which fails of the closed client as PySpark's lookup does, and disables it.
    async fn unstarted_execution_request(
        &self,
        mut request: ExecutePlanRequest,
        compressed: bool,
        log: crate::client_log::ExecutionLog,
    ) -> Result<ExecutePlanRequest> {
        use crate::client_log::{ExecutionLog, Level};
        let of_a_relation = matches!(
            request.plan.as_ref().and_then(|plan| plan.op_type.as_ref()),
            Some(PlanOpType::Root(_))
        );
        // As PySpark's `to_proto`, which compresses the plan of a relation before the method of
        // the client is called with it.
        if compressed && of_a_relation {
            if let Some(plan) = request.plan.as_mut() {
                self.compress_plan(plan).await?;
            }
        }
        // Boxed, of an execution that is logged only: the future of the request holds its pointer.
        if log != ExecutionLog::Silent && crate::client_log::enabled(Level::Debug) {
            Box::pin(log_execution_call(&request, log)).await?;
        }
        self.fill_client_type(&mut request.client_type)?;
        add_result_chunking_option(&mut request, self.call_result_chunking());
        self.session_on(&mut request.session_id);
        // The caller has extended the user context with the extensions of its thread.
        self.fill_user_context(&mut request.user_context);
        // Of the server-side session ID that the lookup of the plan compression of the first plan
        // of a relation gave, and before that of a command, which PySpark builds the request of
        // before it sets the command in its plan.
        self.observe_on(&mut request.client_observed_server_side_session_id);
        if compressed && !of_a_relation {
            if let Some(plan) = request.plan.as_mut() {
                self.compress_plan(plan).await?;
            }
        }
        if log != ExecutionLog::Silent && crate::client_log::enabled(Level::Debug) {
            Box::pin(log_execution(log)).await?;
        }
        Ok(request)
    }

    /// `request`, an execution request that [`SparkConnectClient::unstarted_execution_request`]
    /// built, as PySpark's `_execute_and_fetch_as_iterator` sends it, in the `Progress` of the
    /// execution: it logs the request, as `log` says, and the hook of the executions, if any, gives
    /// the request to send, as it is, as PySpark sends the request of its hooks.
    async fn started_execution_request(
        &self,
        request: ExecutePlanRequest,
        log: crate::client_log::ExecutionLog,
    ) -> Result<ExecutePlanRequest> {
        use crate::client_log::Level;
        if log.logs_responses() && crate::client_log::enabled(Level::Debug) {
            Box::pin(log_execution_request(&request, log)).await?;
        }
        let request = self.hooked(request)?;
        // As PySpark reads its `_stub` once the plan is built, the records are written and the
        // hooks ran: a closed client or channel raises its error and sends nothing, as of the
        // lookup of the plan compression of the first plan, which closes the client when it sees
        // that the server restarted. Its error carries the operation ID of the request, as the
        // `_handle_error` of PySpark's execution sets it, of an operation ID that is not empty.
        if let Err(closed) = self.ensure_open_for_stream() {
            return Err(
                match request.operation_id.as_deref().filter(|id| !id.is_empty()) {
                    Some(operation_id) => closed.with_operation_id(operation_id),
                    None => closed,
                },
            );
        }
        Ok(request)
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
        self.extend_user_context(&mut request.user_context);
        self.add_call_stack(&mut request.user_context);
        let request = self
            .execution_request(request, true, crate::client_log::ExecutionLog::Silent)
            .await?;
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        match self.stub().execute_plan(req).await {
            Ok(response) => Ok(response.into_inner()),
            Err(status) => Err(self
                .error_from_status_of(status, GrpcCall::ServerStreaming)
                .await),
        }
    }

    /// Execute a plan, forwarding raw request bytes and returning a raw response stream.
    ///
    /// Byte-exact passthrough (no prost decode/re-encode), so no proto field is dropped
    /// and deep plans don't hit the recursion limit - the server sees exactly the bytes
    /// the reference client built.
    pub async fn execute_plan_raw(&self, request: Vec<u8>) -> Result<Streaming<Vec<u8>>> {
        self.ensure_open_for_stream()?;
        sendable_bytes(&request)?;
        let mut grpc = tonic::client::Grpc::new(self.checked_channel())
            .max_decoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT)
            .max_encoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT);
        wait_ready(&mut grpc).await?;
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        let path = tonic::codegen::http::uri::PathAndQuery::from_static(
            "/spark.connect.SparkConnectService/ExecutePlan",
        );
        let codec = crate::bytes_codec::ClientBytesCodec::<ExecutePlanResponse>::default();
        let resp = grpc
            .server_streaming(req, path, codec)
            .await
            .map_err(SparkError::from_grpc_status)?;
        Ok(resp.into_inner())
    }

    /// ReattachExecute, byte-exact passthrough returning a raw response stream.
    pub async fn reattach_execute_raw(&self, request: Vec<u8>) -> Result<Streaming<Vec<u8>>> {
        self.ensure_open_for_stream()?;
        sendable_bytes(&request)?;
        let mut grpc = tonic::client::Grpc::new(self.checked_channel())
            .max_decoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT)
            .max_encoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT);
        wait_ready(&mut grpc).await?;
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        let path = tonic::codegen::http::uri::PathAndQuery::from_static(
            "/spark.connect.SparkConnectService/ReattachExecute",
        );
        let codec = crate::bytes_codec::ClientBytesCodec::<ExecutePlanResponse>::default();
        let resp = grpc
            .server_streaming(req, path, codec)
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
        self.fill_client_type(&mut request.client_type)?;
        self.ensure_open_for_stream()?;
        self.fill_user_context(&mut request.user_context);
        self.observe_on(&mut request.client_observed_server_side_session_id);
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        match self.stub().reattach_execute(req).await {
            Ok(response) => Ok(response.into_inner()),
            Err(status) => Err(self
                .error_from_status_of(status, GrpcCall::ServerStreaming)
                .await),
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
        self.fill_client_type(&mut request.client_type)?;
        self.ensure_open()?;
        self.fill_user_context(&mut request.user_context);
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        match self.stub().release_execute(req).await {
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
        sendable_bytes(&request)?;
        let mut grpc = tonic::client::Grpc::new(self.checked_channel())
            .max_decoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT)
            .max_encoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT);
        wait_ready(&mut grpc).await?;
        let deadline = self.call_rpc_deadlines().analyze_plan;
        let mut req = with_deadline(Request::new(request), deadline);
        self._attach_metadata(&mut req);
        let path = tonic::codegen::http::uri::PathAndQuery::from_static(
            "/spark.connect.SparkConnectService/AnalyzePlan",
        );
        let codec = crate::bytes_codec::ClientBytesCodec::<AnalyzePlanResponse>::default();
        let resp = within_deadline(deadline, grpc.unary(req, path, codec))
            .await
            .map_err(SparkError::from_grpc_status)?;
        Ok(resp.into_inner())
    }

    /// Analyze a plan and return metadata about it.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.analyze_plan`.
    pub async fn analyze_plan(&self, request: AnalyzePlanRequest) -> Result<AnalyzePlanResponse> {
        self.analyze_plan_logged(request, crate::client_log::AnalysisLog::Silent)
            .await
    }

    /// As [`SparkConnectClient::analyze_plan`], with what the method of PySpark's client that the
    /// analysis stands for logs of its plan as it is sent, compressed, as `log` says: `Schema for
    /// plan: ...` of `schema`. Not a stable API.
    #[doc(hidden)]
    pub async fn analyze_plan_logged(
        &self,
        mut request: AnalyzePlanRequest,
        log: crate::client_log::AnalysisLog,
    ) -> Result<AnalyzePlanResponse> {
        // As PySpark's `to_proto`, which builds the plans of an analysis, of the lookup of the plan
        // compression of its first plan, before the method of the analysis logs its plan and
        // `_analyze` builds the request. A closed client raises when the request is sent, once the
        // plan is logged, as PySpark's `_analyze` reads its `_stub` only then; the lookup of the
        // first plan fails of it, as PySpark's does, and disables the compression.
        if let Some(analyze) = request.analyze.as_mut() {
            for plan in analyze_plans(analyze).into_iter().flatten() {
                self.compress_plan(plan).await?;
            }
        }
        self.analyze_plan_as_is(request, log).await
    }

    /// As [`SparkConnectClient::analyze_plan_logged`], for an analysis whose plans are sent as
    /// they are, neither compressed nor with the lookup of the plan compression of the client: a
    /// plan that [`SparkConnectClient::prepare_plan`] prepared, and the plans that PySpark's
    /// `SparkConnectClient.schema`, `explain_string`, `same_semantics`, `semantic_hash` and
    /// `_analyze` are given, which they send as they are. Not a stable API.
    #[doc(hidden)]
    pub async fn analyze_plan_as_is(
        &self,
        mut request: AnalyzePlanRequest,
        log: crate::client_log::AnalysisLog,
    ) -> Result<AnalyzePlanResponse> {
        // As the method of PySpark's client, which logs the plan before `_analyze` builds the
        // request: the error of the record fails the analysis there.
        if log != crate::client_log::AnalysisLog::Silent
            && crate::client_log::enabled(crate::client_log::Level::Debug)
        {
            Box::pin(log_analysis(&request, log)).await?;
        }
        self.session_on(&mut request.session_id);
        self.fill_client_type(&mut request.client_type)?;
        self.ensure_open()?;
        self.extend_user_context(&mut request.user_context);
        self.add_call_stack(&mut request.user_context);
        self.fill_user_context(&mut request.user_context);
        let response = self
            .with_retry(|| {
                let deadline = self.call_rpc_deadlines().analyze_plan;
                let mut request = clone_on_grown_stack(&request);
                self.observe_on(&mut request.client_observed_server_side_session_id);
                let mut req = with_deadline(Request::new(request), deadline);
                self._attach_metadata(&mut req);
                let mut stub = self.stub();
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
    async fn with_retry<T, Fut, F>(&self, op: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = std::result::Result<T, tonic::Status>>,
    {
        let mut retries = self.retries_of(GrpcCall::Unary).ending_with_abort();
        self.with_retry_in(&mut retries, GrpcCall::Unary, op).await
    }

    /// As [`Self::with_retry`], of a call that PySpark's client leaves to grpc, without converting
    /// its failure, which the Rust client makes as `rust` says (see
    /// [`Self::result_left_to_grpc`]).
    async fn with_retry_left_to_grpc<T, Fut, F>(&self, rust: LeftToGrpc, op: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = std::result::Result<T, tonic::Status>>,
    {
        let mut retries = self.retries_of(GrpcCall::Unary).ending_with_abort();
        let attempts = self.retried(&mut retries, op).await;
        self.result_left_to_grpc(attempts, rust).await
    }

    /// As [`Self::with_retry`], of a call of the kind `call`, whose attempts are those of
    /// `retries`, which the caller keeps for the wait that follows the call (see
    /// [`ReattachableResponseStream`]).
    async fn with_retry_in<T, Fut, F>(
        &self,
        retries: &mut Retrying,
        call: GrpcCall,
        op: F,
    ) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = std::result::Result<T, tonic::Status>>,
    {
        let attempts = self.retried(retries, op).await;
        self.result_of_attempts(attempts, call).await
    }

    /// The attempts of the RPC closure `op` under `retries`, as [`Self::with_retry_in`] makes
    /// them, which end with the failure as it is.
    async fn retried<T, Fut, F>(
        &self,
        retries: &mut Retrying,
        op: F,
    ) -> std::result::Result<T, RetryError<SparkError>>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = std::result::Result<T, tonic::Status>>,
    {
        // As the reference's `_stub`, through which each attempt goes: once the client is closed
        // (its session stopped while the call waited), the retry raises NO_ACTIVE_SESSION and
        // sends nothing.
        self.retried_checking(retries, || self.ensure_open(), op)
            .await
    }

    /// As [`Self::retried`], of the attempts that `still_open` checks before each retry, whose
    /// error ends them.
    async fn retried_checking<T, Fut, F>(
        &self,
        retries: &mut Retrying,
        still_open: impl FnMut() -> Result<()>,
        mut op: F,
    ) -> std::result::Result<T, RetryError<SparkError>>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = std::result::Result<T, tonic::Status>>,
    {
        use futures::FutureExt;
        retrying(retries, still_open, || {
            // An attempt that is not answered ends with the signal that a wait of the caller
            // takes, as PySpark's blocked call does. A combinator rather than an async block that
            // awaits the call: such a block keeps the future of the call twice, as what it
            // captures and as what it awaits, and that future, of the state of a gRPC call, is
            // large enough that the extra copy overflows the stack of a test thread or of a
            // Python thread of 512 KiB. Boxed, so that the layers of the retries above it hold a
            // pointer to it rather than the future itself, each of which a debug build reserves
            // again in the frame that polls it.
            Box::pin(
                crate::runtime::until_abort(crate::runtime::ABORT_RETRIES, op()).map(|attempt| {
                    match attempt {
                        Some(result) => result.map_err(RetryError::Status),
                        None => Err(RetryError::Aborted(wait_aborted())),
                    }
                }),
            )
        })
        .await
    }

    /// The attempts of a call of this client under `retries` (see [`retrying`]), which end with
    /// the failure as it is.
    async fn attempts_in<T, Fut>(
        &self,
        retries: &mut Retrying,
        attempt: impl FnMut() -> Fut,
    ) -> std::result::Result<T, RetryError<SparkError>>
    where
        Fut: std::future::Future<Output = std::result::Result<T, RetryError<SparkError>>>,
    {
        // As the reference's `_stub`, through which each attempt goes: once the client is closed
        // (its session stopped while the call waited), the retry raises NO_ACTIVE_SESSION and
        // sends nothing.
        let still_open = || self.ensure_open();
        retrying(retries, still_open, attempt).await
    }

    /// The result of the attempts of a call of the kind `call`: the failed RPC that ends them is
    /// the error of the server (see [`Self::exceeded_error`] for the failure of retries that are
    /// spent).
    async fn result_of_attempts<T>(
        &self,
        attempts: std::result::Result<T, RetryError<SparkError>>,
        call: GrpcCall,
    ) -> Result<T> {
        match attempts {
            Ok(value) => Ok(value),
            Err(RetryError::Status(status)) => Err(self.error_from_status_of(status, call).await),
            Err(RetryError::Exceeded(status)) => Err(self
                .exceeded_error(status, RetriesExceeded::Policy, call)
                .await),
            Err(RetryError::Aborted(error)) => Err(error),
            Err(RetryError::Failed(error)) => Err(*error),
            Err(RetryError::WaitAborted) => Err(wait_aborted()),
        }
    }

    /// The result of the attempts of a call of one response that PySpark's client leaves to grpc,
    /// without converting its failure (`clone`, and the uploads of `copy_from_local_to_fs` and
    /// of the caching of artifacts), in the dialect of grpcio: the failed RPC that ends them is
    /// the error of its status, of grpcio's error of it, which the Python binding raises, and of
    /// no details of the server nor logs, once PySpark's `Retrying` warned of its spent retries.
    /// Of the Rust client, as `rust` says: the result of the attempts as of any call
    /// ([`Self::result_of_attempts`]), or the error of the status as it is, as of the grpcio
    /// dialect but without grpcio's error.
    async fn result_left_to_grpc<T>(
        &self,
        attempts: std::result::Result<T, RetryError<SparkError>>,
        rust: LeftToGrpc,
    ) -> Result<T> {
        let grpcio = crate::connector::grpcio_dialect();
        if !grpcio && rust == LeftToGrpc::Converted {
            return self.result_of_attempts(attempts, GrpcCall::Unary).await;
        }
        // Neither the details of the error nor the configs of its stack trace are asked for, and
        // the client is not closed of it, as PySpark's `_handle_error` is not called.
        let left_to_grpc = |status: tonic::Status| {
            let grpcio =
                grpcio.then(|| Box::new(crate::error::grpcio_error(&status, GrpcCall::Unary)));
            let mut error = SparkError::from_grpc_status(status);
            error.grpcio = grpcio;
            error
        };
        match attempts {
            Ok(value) => Ok(value),
            Err(RetryError::Status(status)) => Err(left_to_grpc(status)),
            Err(RetryError::Exceeded(status)) => {
                crate::retries::retries_exceeded(RetriesExceeded::Policy)?;
                Err(left_to_grpc(status))
            }
            Err(RetryError::Aborted(error)) => Err(error),
            Err(RetryError::Failed(error)) => Err(*error),
            Err(RetryError::WaitAborted) => Err(wait_aborted()),
        }
    }

    /// The error of a call of the kind `call` whose retries, which `exceeded` tells, are spent with
    /// the failure `status`, as PySpark's `Retrying._wait` raises it: it warns that they are first,
    /// as the listener of the retries of the process is told (see
    /// [`crate::retries::RetriesListener`]). The error of the listener, if any, is the error in
    /// place of the failure, which is not converted then: PySpark raises the exception of the
    /// warning (of a filter of `"error"`), which its `_handle_error` raises as it is.
    async fn exceeded_error(
        &self,
        status: tonic::Status,
        exceeded: RetriesExceeded,
        call: GrpcCall,
    ) -> SparkError {
        match crate::retries::retries_exceeded(exceeded) {
            Ok(()) => self.error_from_status_of(status, call).await,
            Err(error) => error,
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
    ///
    /// Once [`SparkConnectClient::set_reattachable_execute`] turned the reattachable executions
    /// off, the execution is not reattachable, as PySpark's with `use_reattachable_execute`
    /// False: its request has no reattach option, its `ExecutePlan` has no deadline, the stream
    /// is never reattached or released, and a failure that the retry policy retries, of the call
    /// or of its stream, sends the `ExecutePlan` again, under the retries of the whole execution.
    ///
    /// It logs what PySpark's `to_table` logs of a plan of a relation, and what its
    /// `execute_command` logs of a command (see [`crate::client_log`]).
    pub async fn execute_plan_reattachable(
        &self,
        request: ExecutePlanRequest,
    ) -> Result<ReattachableResponseStream> {
        let log = match request.plan.as_ref().and_then(|plan| plan.op_type.as_ref()) {
            Some(PlanOpType::Root(_)) => crate::client_log::ExecutionLog::ToTable,
            _ => crate::client_log::ExecutionLog::ExecuteCommand,
        };
        self.execute_reattachable(request, true, log).await
    }

    /// As [`SparkConnectClient::execute_plan_reattachable`], with the logs of the method of
    /// PySpark's client that the execution stands for, as `log` says. Not a stable API.
    #[doc(hidden)]
    pub async fn execute_plan_reattachable_logged(
        &self,
        request: ExecutePlanRequest,
        log: crate::client_log::ExecutionLog,
    ) -> Result<ReattachableResponseStream> {
        self.execute_reattachable(request, true, log).await
    }

    /// As [`SparkConnectClient::execute_plan_reattachable_logged`], for a request whose plan is
    /// sent as it is, neither compressed nor with the lookup of the plan compression of the
    /// client: a plan that [`SparkConnectClient::prepare_plan`] prepared, whose compression was
    /// tried once, as PySpark's `to_proto` tries it, and the plan that PySpark's
    /// `SparkConnectClient.to_table`, `to_table_as_iterator` and `to_pandas` are given, which they
    /// send as it is. Not a stable API.
    #[doc(hidden)]
    pub async fn execute_plan_reattachable_as_is(
        &self,
        request: ExecutePlanRequest,
        log: crate::client_log::ExecutionLog,
    ) -> Result<ReattachableResponseStream> {
        self.execute_reattachable(request, false, log).await
    }

    /// As [`SparkConnectClient::execute_plan_reattachable`], for a request whose plan is sent as
    /// it is, neither compressed nor with the lookup of the plan compression of the client, as
    /// PySpark's registrations of functions, table functions, data sources and Java functions
    /// (`register_udf`, `register_udtf`, `register_data_source` and `register_java`) set their
    /// command in the plan of their request, whose `_execute` logs `Execute`.
    pub async fn execute_plan_reattachable_uncompressed(
        &self,
        request: ExecutePlanRequest,
    ) -> Result<ReattachableResponseStream> {
        self.execute_reattachable(request, false, crate::client_log::ExecutionLog::Execute)
            .await
    }

    /// The execution of `request` as a reattachable stream, of its plan compressed when
    /// `compressed` (see [`SparkConnectClient::execute_plan_reattachable`]), and of the logs that
    /// `log` says.
    async fn execute_reattachable(
        &self,
        request: ExecutePlanRequest,
        compressed: bool,
        log: crate::client_log::ExecutionLog,
    ) -> Result<ReattachableResponseStream> {
        let built = self.build_execution(request, compressed, log).await?;
        self.execute_built(built).await
    }

    /// The first part of the execution of `request` that
    /// [`SparkConnectClient::execute_plan_reattachable_logged`] makes, of its plan compressed when
    /// `compressed`, as PySpark's client builds an execution before its `Progress` of the
    /// execution starts: the request of the user context extensions of the calling thread, of the
    /// call stack of its call, and of the client, its plan compressed, and the records that
    /// PySpark's client logs then (see [`SparkConnectClient::unstarted_execution_request`]). An
    /// error of it, as of one of the records, fails the execution before its `Progress` starts:
    /// [`SparkConnectClient::execute_built`] sends it. Not a stable API.
    #[doc(hidden)]
    pub async fn build_execution(
        &self,
        mut request: ExecutePlanRequest,
        compressed: bool,
        log: crate::client_log::ExecutionLog,
    ) -> Result<BuiltExecution> {
        self.extend_user_context(&mut request.user_context);
        self.add_call_stack(&mut request.user_context);
        // In the dialect of grpcio, the operation ID that the iterator would give the request is
        // given it first (see `with_operation_id_of_grpcio`).
        let request = with_operation_id_of_grpcio(request);
        let request = self
            .unstarted_execution_request(request, compressed, log)
            .await?;
        Ok(BuiltExecution { request, log })
    }

    /// Send `built`, an execution that [`SparkConnectClient::build_execution`] built, as
    /// PySpark's `_execute_and_fetch_as_iterator` sends it in the `Progress` of the execution: the
    /// record of the request, the hook of the executions, and the stream of the responses, as
    /// [`SparkConnectClient::execute_plan_reattachable`] reads them. Not a stable API.
    #[doc(hidden)]
    pub async fn execute_built(&self, built: BuiltExecution) -> Result<ReattachableResponseStream> {
        let request = self
            .started_execution_request(built.request, built.log)
            .await?;
        self.execute_plan_reattachable_of(request, built.log).await
    }

    /// Run `request` as [`SparkConnectClient::execute_plan_reattachable`] runs it, without
    /// waiting for it, as PySpark's `RemoteModelRef` deletes a model from the ML cache of the
    /// server when its last reference goes: one `execute_command` of the thread that drops it,
    /// whose responses are read to the end and then released.
    ///
    /// The request takes the user context extensions of the calling thread now, as PySpark's
    /// request is of the thread of its finalizer; the caller gives it the tags of that thread. It
    /// is then sent on the shared runtime, so that a drop never blocks on it, and its errors are
    /// ignored, as PySpark's `_delete_ml_cache` ignores them. Not a stable API.
    #[doc(hidden)]
    pub fn spawn_execute_plan_reattachable(&self, mut request: ExecutePlanRequest) {
        self.extend_user_context(&mut request.user_context);
        self.add_call_stack(&mut request.user_context);
        let client = self.of_call_settings();
        drop(crate::runtime::get_runtime().spawn(async move {
            if let Ok(mut stream) = client
                .execute_plan_reattachable_in_context(
                    request,
                    true,
                    crate::client_log::ExecutionLog::ExecuteCommand,
                )
                .await
            {
                while let Ok(Some(_)) = stream.message().await {}
            }
        }));
    }

    /// [`SparkConnectClient::execute_plan_reattachable`] of a request whose user context has the
    /// user context extensions of its thread already.
    async fn execute_plan_reattachable_in_context(
        &self,
        request: ExecutePlanRequest,
        compressed: bool,
        log: crate::client_log::ExecutionLog,
    ) -> Result<ReattachableResponseStream> {
        // Before the iterator: its ReattachExecute and ReleaseExecute requests take the
        // user context and the client type of this request, and the request of the hook, as
        // PySpark's iterator is of the request that the hooks return. PySpark reads its `_stub`
        // before it makes the iterator, so nothing is released of a closed client. In the dialect
        // of grpcio, the operation ID that the iterator would give the request is given it first
        // (see `with_operation_id_of_grpcio`).
        let request = with_operation_id_of_grpcio(request);
        let request = self.execution_request(request, compressed, log).await?;
        self.execute_plan_reattachable_of(request, log).await
    }

    /// The stream of the execution of `request`, the request to send, which
    /// [`SparkConnectClient::execution_request`] built, of the logs of `log`.
    async fn execute_plan_reattachable_of(
        &self,
        request: ExecutePlanRequest,
        log: crate::client_log::ExecutionLog,
    ) -> Result<ReattachableResponseStream> {
        let reattachable = self.reattachable_execute();
        // As PySpark's: the metadata of a reattachable execution is read once, of which its
        // iterator sends each of its requests; each attempt of one that is not reads its own.
        let execution_metadata = if reattachable {
            Some(self.metadata_of_execution().await?)
        } else {
            None
        };
        let iter = if reattachable {
            ExecutePlanResponseReattachableIterator::new(request)
        } else {
            ExecutePlanResponseReattachableIterator::not_reattachable(request)
        };
        // Before the first RPC, so that an execution that waits for the server is in flight.
        let inflight = InflightGuard::register(
            std::sync::Arc::clone(&self.inflight_operations),
            iter.operation_id().to_string(),
        );
        // As PySpark's: an execution that is not reattachable has no deadline, which would end it.
        let deadline = if reattachable {
            self.call_rpc_deadlines().reattachable_execute_plan
        } else {
            None
        };
        let mut stream_deadline = None;
        // The attempts of the first RPC are those of the first response too, as the reference's
        // first `_has_next` retries both (see `first_response_retries`).
        let mut first_response_retries = self
            .retries_of(GrpcCall::ServerStreaming)
            .ending_with_abort()
            .reading_status_first(reattachable);
        let stream = {
            // The first RPC is a wait of the caller for the execution.
            let _waiting = inflight.waiting();
            let op = || {
                let request = clone_on_grown_stack(iter.request());
                let mut req = with_deadline(Request::new(request), deadline);
                let refused = match &execution_metadata {
                    Some(Ok(metadata)) => {
                        add_metadata(&mut req, metadata);
                        None
                    }
                    Some(Err(status)) => Some(status.clone()),
                    None => {
                        self._attach_metadata(&mut req);
                        None
                    }
                };
                stream_deadline = stream_deadline_in(deadline);
                let mut stub = self.stub();
                async move {
                    if let Some(status) = refused {
                        return Err(status);
                    }
                    within_deadline(deadline, stub.execute_plan(req))
                        .await
                        .map(Response::into_inner)
                }
            };
            // As the reference's iterator, which retries through the stub that it was made with:
            // once the channel of the client is closed, as by the `stop()` of another thread, a
            // retry raises grpc's `ValueError` of a closed channel; of an execution that is not
            // reattachable, as its `_stub`, which each attempt reads.
            let attempts = if reattachable {
                self.retried_checking(
                    &mut first_response_retries,
                    || self.ensure_channel_open_for_stream(),
                    op,
                )
                .await
            } else {
                self.retried(&mut first_response_retries, op).await
            };
            self.result_of_attempts(attempts, GrpcCall::ServerStreaming)
                .await
        };
        let stream = match stream {
            Ok(stream) => Some(stream),
            // As the reference iterator, past its deadline an execution is resumed by
            // reattaching, as it goes on on the server.
            Err(error)
                if reattachable
                    && error.grpc_code == Some(tonic::Code::DeadlineExceeded as i32) =>
            {
                None
            }
            Err(error) => {
                // As the reference iterator releases an execution that fails at once, such
                // as for an analysis error, so the server does not keep it until its
                // detached timeout. Once releases are refused, that release raises instead.
                // An execution that is not reattachable is not kept by the server.
                if reattachable {
                    if let Some(refused) = release_refused() {
                        return Err(refused);
                    }
                    // Of an execution whose iterator ended with the failure. Of metadata that
                    // gRPC does not send, the release fails too, unsent.
                    if let Some(Ok(metadata)) = &execution_metadata {
                        self.spawn_release(
                            metadata.clone(),
                            iter.release_all_request(),
                            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
                        );
                    }
                }
                return Err(error.with_operation_id(iter.operation_id()));
            }
        };
        Ok(ReattachableResponseStream {
            metadata: execution_metadata
                .and_then(std::result::Result::ok)
                .unwrap_or_else(|| self.execution_metadata()),
            client: self.of_call_settings(),
            log,
            reattachable,
            first_response_retries: Some(first_response_retries),
            iter,
            stream: stream.map(crate::fork::ForkLocal::new),
            stream_deadline,
            execute_again: false,
            done: false,
            stream_ended: Default::default(),
            // Nothing is released of an execution that is not reattachable.
            released: !reattachable,
            _inflight: inflight,
            generation: crate::fork::generation(),
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
        self.session_on(&mut request.session_id);
        self.fill_client_type(&mut request.client_type)?;
        self.ensure_open()?;
        self.extend_user_context(&mut request.user_context);
        self.fill_user_context(&mut request.user_context);
        self.observe_on(&mut request.client_observed_server_side_session_id);
        let deadline = self.call_rpc_deadlines().fetch_error_details;
        let mut req = with_deadline(Request::new(request), deadline);
        self._attach_metadata(&mut req);
        let resp = within_deadline(deadline, self.stub().fetch_error_details(req)).await;
        let response = resp
            .map(Response::into_inner)
            .map_err(|status| self.grpc_error(status, GrpcCall::Unary))?;
        self.verify_response(&response.session_id, &response.server_side_session_id)?;
        Ok(response)
    }

    /// The error for a failed RPC of one response, as `SparkConnectClient._handle_rpc_error`
    /// raises it. An error that the server raised gets the details that `FetchErrorDetails`
    /// returns when the server keeps them, and shows the server's stack trace when
    /// `spark.sql.connect.serverStacktrace.enabled` or `spark.sql.pyspark.jvmStacktrace.enabled`
    /// is true. As the error is raised either way, failing to read those is not.
    pub async fn error_from_status(&self, status: tonic::Status) -> SparkError {
        self.error_from_status_of(status, GrpcCall::Unary).await
    }

    /// As [`SparkConnectClient::error_from_status`], for a failure of a call of the kind `call`,
    /// which names the class of grpcio's error in the message of a status that the server did not
    /// raise an error of (see [`SparkError::from_grpc_status_of`]).
    async fn error_from_status_of(&self, status: tonic::Status, call: GrpcCall) -> SparkError {
        // The error of the source of the metadata of the call, which PySpark raises as it is.
        if let Some(error) = crate::metadata::metadata_failure(&status) {
            return error;
        }
        let failure = crate::error::grpcio_error(&status, call);
        // As PySpark's `_handle_error`, which raises the error of a call as grpc gives it while the
        // thread handles another failure of the client (`inside_error_handling`), as of a call
        // that a handler of a log of that conversion makes, so that the handling does not recurse:
        // of no records, no details of the server, no configs of its stack trace, and the error of
        // the status as it is, of grpcio's error, which the Python binding raises.
        if self.inside_error_handling() {
            let mut error = SparkError::from_grpc_status(status);
            error.grpcio = crate::connector::grpcio_dialect().then(|| Box::new(failure));
            return error;
        }
        // As PySpark's `_handle_error`, which converts the failure in the `except Exception`
        // clause of the call, outside of the `except KeyboardInterrupt` of an execution: a signal
        // while it runs interrupts nothing (see [`converting_failure`]).
        let _converting =
            Converting::start(&failure, self.operation_id_of_this_call(), self.identity());
        // Of whose polls a call of the client is of the handling of this failure (see
        // [`SparkConnectClient::inside_error_handling`]). Boxed, as the future of every call that
        // converts its failure would otherwise hold the conversion.
        crate::runtime::polling_conversion(
            self.identity(),
            Box::pin(self.converted_error_of(status, call, failure)),
        )
        .await
    }

    /// The error of the failed RPC of `status`, of grpcio's error `failure`, as
    /// [`SparkConnectClient::error_from_status_of`] converts it.
    async fn converted_error_of(
        &self,
        status: tonic::Status,
        call: GrpcCall,
        failure: crate::error::GrpcioError,
    ) -> SparkError {
        // As PySpark's `_handle_rpc_error`, which logs the error of grpc before it reads it, and
        // then the `ErrorInfo` of a status of the server, but of a call past its deadline, which
        // it raises at once. It reads the `google.rpc.Status` of the call with grpc_status, which
        // raises its `ValueError` of one that does not match the call, or protobuf's `DecodeError`
        // of one that upb does not read, before the `ErrorInfo` is read, and so does the unpacking
        // of its first `ErrorInfo`, before it is logged.
        let deadline = crate::retries::grpc_python_code(&status) == tonic::Code::DeadlineExceeded;
        // The details as upb reads them, in the dialect of grpcio, read once for the readers of the
        // failure. Boxed, as they are kept across the wait for the logs, and so is the mismatch.
        let read = crate::connector::grpcio_dialect()
            .then(|| Box::new(crate::rpc_status::Details::of(status.details())));
        let mismatch = match (&read, deadline) {
            (Some(read), false) => crate::error::from_call_error_of(
                &status,
                read,
                call,
                crate::error::StatusRead::ErrorInfo,
            )
            .map(Box::new),
            _ => None,
        };
        // The logs of a call off the thread of its caller, as of a stream that is read in the
        // background, precede what the call does next with its error, as PySpark's do. The error
        // of a log, of a handler or a filter of the logger that raised, is the error of the call,
        // of the failure as its context, as the exception of PySpark's log goes out of
        // `_handle_rpc_error`, before what follows the log.
        if let Err(error) =
            crate::retries::log_delivered(|listener| listener.rpc_error(&failure)).await
        {
            return crate::retries::raised_in_handling(
                error,
                crate::retries::Handled::Rpc(&failure),
            );
        }
        if !deadline && mismatch.is_none() {
            if let Some(text) = crate::error::error_info_text(status.details(), read.as_deref()) {
                if let Err(error) =
                    crate::retries::log_delivered(|listener| listener.error_info(&text)).await
                {
                    return crate::retries::raised_in_handling(
                        error,
                        crate::retries::Handled::Rpc(&failure),
                    );
                }
            }
        }
        if let Some(mismatch) = mismatch {
            return *mismatch;
        }
        let error = self.session_checked_grpc_error_read(status, call, read.map(|read| *read));
        if error.get_condition().as_deref() == Some("CONNECT_INVALID_PLAN.CANNOT_PARSE") {
            // As PySpark's `_handle_rpc_error`, which logs that it disables the plan compression
            // before it disables it and asks for the details of the error.
            if let Err(logged) = crate::client_log::logged(crate::client_log::Level::Info, || {
                crate::client_log::Message::Text(
                    "Disabling plan compression for the session due to \
                     CONNECT_INVALID_PLAN.CANNOT_PARSE error."
                        .to_string(),
                )
            })
            .await
            {
                return crate::retries::raised_in_handling(
                    logged,
                    crate::retries::Handled::Rpc(&failure),
                );
            }
            self.keep_plan_compression(PlanCompression::Disabled);
        }
        if error.server.is_none() {
            return error;
        }
        // Boxed: the fetch of the details and the reads of the confs, which retry, would
        // otherwise make the future of every call that converts its failure much larger, in a
        // debug build past the stack of a thread that polls it.
        Box::pin(self.server_error_with_details(error, &failure)).await
    }

    /// `error`, an error that the server raised, with its details and its stack trace, as
    /// [`SparkConnectClient::error_from_status`] completes it, of the failed RPC of grpcio's error
    /// `failure`.
    async fn server_error_with_details(
        &self,
        error: SparkError,
        failure: &crate::error::GrpcioError,
    ) -> SparkError {
        let Some(server) = &error.server else {
            return error;
        };
        // The requests of the failure are of the context of the calling thread as it is now, as
        // PySpark builds them of the attributes of its client then, and none is sent when the
        // binding cannot build them.
        if let Some(hook) = FAILURE_REQUESTS_HOOK.get() {
            match hook(self, &error) {
                Ok(true) => {}
                Ok(false) => return error,
                Err(unbuilt) => return unbuilt,
            }
        }
        let details = match &server.error_id {
            Some(error_id) => {
                // PySpark builds the request of the details with its user agent, and fetches
                // them through `_stub`, which raises NO_ACTIVE_SESSION once the client is closed,
                // as a session change closes it: the error of either replaces the error of the
                // server, the user agent's first.
                let client_type = match self.client_type() {
                    Ok(client_type) => client_type,
                    Err(error) => return error,
                };
                if let Err(closed) = self.ensure_open() {
                    return closed;
                }
                self.error_details(error_id, client_type).await
            }
            None => None,
        };
        match self.display_server_stacktrace().await {
            Ok(display_stacktrace) => error.with_details(details, display_stacktrace),
            // As PySpark's `_display_server_stack_trace`, whose `except Exception` lets out a
            // `BaseException` only, which replaces the error that is raised, as it is raised while
            // `_handle_rpc_error` handles the failure.
            Err(uncaught) => {
                crate::retries::raised_in_handling(uncaught, crate::retries::Handled::Rpc(failure))
            }
        }
    }

    /// The details of the error that the server keeps under `error_id`, or `None` if they
    /// cannot be fetched, or once a signal interrupted the wait of the caller, with a request of
    /// the client type `client_type`. Mirrors `SparkConnectClient._fetch_enriched_error`.
    async fn error_details(
        &self,
        error_id: &str,
        client_type: String,
    ) -> Option<FetchErrorDetailsResponse> {
        let request = FetchErrorDetailsRequest {
            session_id: self.client_session_id(),
            error_id: error_id.to_string(),
            client_type: Some(client_type),
            user_context: self.request_user_id().map(|user_id| UserContext {
                user_id,
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut request = request;
        // As `_fetch_enriched_error`, whose request carries the user context extensions.
        self.extend_user_context(&mut request.user_context);
        self.observe_on(&mut request.client_observed_server_side_session_id);
        let deadline = self.call_rpc_deadlines().fetch_error_details;
        let mut req = with_deadline(Request::new(request), deadline);
        self._attach_metadata(&mut req);
        // Not sent once a signal interrupted the wait of the caller, which raises the exception of
        // the signal: as PySpark's KeyboardInterrupt leaves the failure of the execution that it
        // interrupts unread, such as the server's cancellation of it. The reads of the configs of
        // the stack trace end then too (see `conf_while_raising`).
        let response = crate::runtime::until_abort(
            crate::runtime::ABORT_RETRIES,
            within_deadline(deadline, self.stub().fetch_error_details(req)),
        )
        .await?
        .ok()?
        .into_inner();
        self.verify_response(&response.session_id, &response.server_side_session_id)
            .ok()?;
        Some(response)
    }

    /// Whether the message of an error that the server raised shows the server's stack
    /// trace. Mirrors `SparkConnectClient._display_server_stack_trace`, which shows it when a
    /// config cannot be read. The error is one that its `except Exception` does not catch.
    async fn display_server_stacktrace(&self) -> Result<bool> {
        Ok(
            match self
                .conf_while_raising("spark.sql.connect.serverStacktrace.enabled")
                .await?
            {
                Some(enabled) if enabled == "true" => true,
                Some(_) => self
                    .conf_while_raising("spark.sql.pyspark.jvmStacktrace.enabled")
                    .await?
                    .is_none_or(|enabled| enabled == "true"),
                None => true,
            },
        )
    }

    /// The value of a config, or `None` if it cannot be read, as PySpark's
    /// `_display_server_stack_trace` reads it while an error is raised: with `RuntimeConf.get`,
    /// whose `config` retries it under the retry policy, and whose failure its `except Exception`
    /// catches, with the exception of the warning of the retries once they are spent. A failure
    /// is not converted, as PySpark's `_handle_error` raises an error of the handling of another
    /// as it is. The error is one that the `except Exception` does not catch.
    async fn conf_while_raising(&self, key: &str) -> Result<Option<String>> {
        use spark_connect_proto::config_request::{operation::OpType, Get, Operation};
        // PySpark reads the config through `_stub`, which raises NO_ACTIVE_SESSION once the
        // client is closed, as a session change closes it, so it sends nothing then.
        if self.is_closed() {
            return Ok(None);
        }
        let mut request = ConfigRequest {
            session_id: self.client_session_id(),
            user_context: Some(UserContext {
                user_id: self.request_user_id().unwrap_or_default(),
                ..Default::default()
            }),
            operation: Some(Operation {
                op_type: Some(OpType::Get(Get {
                    keys: vec![key.to_string()],
                })),
            }),
            // As the conf of PySpark's `_display_server_stack_trace`, which catches the error of
            // its user agent as any other: the stack trace is then shown.
            client_type: match self.client_type() {
                Ok(client_type) => Some(client_type),
                Err(_) => return Ok(None),
            },
            ..Default::default()
        };
        // As the request of PySpark's `RuntimeConf.get`, which carries the user context
        // extensions and the call stack of the user's code, as every Config request, of
        // `_config_request_with_metadata`, and which `config` builds before its retries.
        self.extend_user_context(&mut request.user_context);
        self.add_call_stack(&mut request.user_context);
        self.observe_on(&mut request.client_observed_server_side_session_id);
        let request = &request;
        let mut retries = self.retries_of(GrpcCall::Unary).ending_with_abort();
        // As the `except Exception` of `_display_server_stack_trace`, which catches the exception
        // that a signal handler raises while the config is read, unless it is a `BaseException`
        // only: the config is then not read (see `runtime::catching`).
        let Some(attempts) = crate::runtime::catching(self.retried(&mut retries, || {
            let deadline = self.call_rpc_deadlines().config;
            let mut req = with_deadline(Request::new(request.clone()), deadline);
            self._attach_metadata(&mut req);
            let mut stub = self.stub();
            async move {
                within_deadline(deadline, stub.config(req))
                    .await
                    .map(Response::into_inner)
            }
        }))
        .await
        else {
            return Ok(None);
        };
        let response = match attempts {
            Ok(response) => response,
            Err(RetryError::Exceeded(_)) => {
                if let Err(error) = crate::retries::retries_exceeded(RetriesExceeded::Policy) {
                    if let Some(error) = crate::retries::caught(error) {
                        return Err(error);
                    }
                }
                return Ok(None);
            }
            // As PySpark's `except Exception`, which catches the exception of a decider too, but
            // not a `BaseException` only.
            Err(RetryError::Failed(error)) => {
                if let Some(error) = crate::retries::caught(*error) {
                    return Err(error);
                }
                return Ok(None);
            }
            Err(_) => return Ok(None),
        };
        if self
            .verify_response(&response.session_id, &response.server_side_session_id)
            .is_err()
        {
            return Ok(None);
        }
        Ok(response
            .pairs
            .into_iter()
            .next()
            .and_then(|pair| pair.value))
    }

    /// Update or fetch configurations.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.config`.
    pub async fn config(&self, mut request: ConfigRequest) -> Result<ConfigResponse> {
        self.session_on(&mut request.session_id);
        self.fill_client_type(&mut request.client_type)?;
        self.ensure_open()?;
        self.extend_user_context(&mut request.user_context);
        self.add_call_stack(&mut request.user_context);
        self.fill_user_context(&mut request.user_context);
        let response = self
            .with_retry(|| {
                let deadline = self.call_rpc_deadlines().config;
                let mut request = request.clone();
                self.observe_on(&mut request.client_observed_server_side_session_id);
                let mut req = with_deadline(Request::new(request), deadline);
                self._attach_metadata(&mut req);
                let mut stub = self.stub();
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
        self.session_on(&mut request.session_id);
        self.fill_client_type(&mut request.client_type)?;
        self.ensure_open()?;
        self.extend_user_context(&mut request.user_context);
        self.fill_user_context(&mut request.user_context);
        let response = self
            .with_retry(|| {
                let deadline = self.call_rpc_deadlines().interrupt;
                let mut request = request.clone();
                self.observe_on(&mut request.client_observed_server_side_session_id);
                let mut req = with_deadline(Request::new(request), deadline);
                self._attach_metadata(&mut req);
                let mut stub = self.stub();
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
        request.session_id = self.client_session_id();
        request.user_context = Some(UserContext {
            user_id: self.request_user_id().unwrap_or_default(),
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
            session_id: self.client_session_id(),
            user_context: Some(UserContext {
                user_id: self.request_user_id().unwrap_or_default(),
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
        request.session_id = self.client_session_id();
        request.user_context = Some(UserContext {
            user_id: self.request_user_id().unwrap_or_default(),
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
        request.session_id = self.client_session_id();
        request.user_context = Some(UserContext {
            user_id: self.request_user_id().unwrap_or_default(),
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
        request.session_id = self.client_session_id();
        request.user_context = Some(UserContext {
            user_id: self.request_user_id().unwrap_or_default(),
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
        request.session_id = self.client_session_id();
        request.user_context = Some(UserContext {
            user_id: self.request_user_id().unwrap_or_default(),
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
        request.session_id = self.client_session_id();
        request.user_context = Some(UserContext {
            user_id: self.request_user_id().unwrap_or_default(),
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
        request.session_id = self.client_session_id();
        request.user_context = Some(UserContext {
            user_id: self.request_user_id().unwrap_or_default(),
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
        request.session_id = self.client_session_id();
        request.user_context = Some(UserContext {
            user_id: self.request_user_id().unwrap_or_default(),
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
    /// returned client, of the connection of the process to the same endpoint, is of that
    /// session. A failure of the RPC is the error of its status as it is, of grpcio's error of it
    /// in the dialect of grpcio, as PySpark's `clone` leaves it to grpc: neither the details of the
    /// error nor the stack trace of the server are requested for it. As PySpark's, the response is
    /// not checked against the session of this client: a session ID or a server-side session ID
    /// of the response that is not that of this client is not an error, and does not close it.
    pub async fn clone_session(&self, new_session_id: Option<&str>) -> Result<Self> {
        let user_id = self
            .builder
            .user_id()
            .map(String::from)
            .or_else(|| self.request_user_id());
        self.clone_session_of_user(new_session_id, user_id).await
    }

    /// As [`SparkConnectClient::clone_session`], of a client of the user `user_id` as it is given,
    /// whom its requests and those of its artifacts carry, also when the connection has a
    /// `user_id`, which `clone_session` takes in its place, as PySpark's `clone` makes its client
    /// of the connection and `user_id=self._user_id` (see
    /// [`SparkConnectClient::with_new_session_id_of_user`]). Not a stable API.
    #[doc(hidden)]
    pub async fn clone_session_of_user(
        &self,
        new_session_id: Option<&str>,
        user_id: Option<String>,
    ) -> Result<Self> {
        self.ensure_open()?;
        // As PySpark's request: of the client type "python", not of the user agent, and with
        // no observed server-side session ID.
        let request = CloneSessionRequest {
            session_id: self.client_session_id(),
            user_context: Some(UserContext {
                user_id: self.request_user_id().unwrap_or_default(),
                ..Default::default()
            }),
            client_type: Some("python".to_string()),
            new_session_id: new_session_id.map(str::to_string),
            ..Default::default()
        };
        // As PySpark's `clone`, which leaves the failure to grpc: nothing else is asked of the
        // server for it, of the Rust client too.
        let response = self
            .with_retry_left_to_grpc(LeftToGrpc::AsItIs, || {
                let deadline = self.call_rpc_deadlines().clone_session;
                let mut req = with_deadline(Request::new(request.clone()), deadline);
                self._attach_metadata(&mut req);
                let mut stub = self.stub();
                async move {
                    within_deadline(deadline, stub.clone_session(req))
                        .await
                        .map(Response::into_inner)
                }
            })
            .await?;
        // As PySpark's `clone`, which does not check the IDs of its response: neither its session
        // ID nor its server-side session ID, which the client does not observe either.
        if let Some(requested) = new_session_id {
            if response.new_session_id != requested {
                return Err(SparkError::connect_msg(format!(
                    "Returned session ID '{}' does not match requested ID '{requested}'",
                    response.new_session_id
                )));
            }
        }
        let mut cloned = self.with_new_session_id_of_user(user_id);
        // As PySpark's `clone`, which sets the new session ID in a copy of the builder.
        cloned
            .builder
            .set(crate::channel::PARAM_SESSION_ID, &response.new_session_id);
        cloned.client_session_id =
            std::sync::Arc::new(crate::fork::Shared::new(response.new_session_id.clone()));
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
        request.session_id = self.client_session_id();
        request.user_context = Some(UserContext {
            user_id: self.request_user_id().unwrap_or_default(),
            ..Default::default()
        });
        // The error of the user agent is that of the release, which PySpark's `stop()` logs.
        request.client_type = Some(self.client_type()?);
        // As PySpark's `_stub`, read once the request is built: NO_ACTIVE_SESSION once the client is
        // closed, and the error of a closed channel once it is open again.
        self.ensure_open()?;

        let response = self
            .with_retry(|| {
                let deadline = self.call_rpc_deadlines().release_session;
                let mut req = with_deadline(Request::new(request.clone()), deadline);
                self._attach_metadata(&mut req);
                let mut stub = self.stub();
                async move {
                    within_deadline(deadline, stub.release_session(req))
                        .await
                        .map(Response::into_inner)
                }
            })
            .await?;
        self.verify_response(&response.session_id, &response.server_side_session_id)
    }

    /// The statuses of operations of this session, as PySpark's `_get_operation_statuses`: the
    /// GetStatus RPC of `request`, of the operations and the extensions that the caller asks of,
    /// which the client gives the IDs of its session, its user agent as the client type, its user
    /// and the server-side session ID that it observed, as PySpark builds it, under the retry
    /// policy and the deadline of `get_status`. Its response is checked as the others are.
    pub async fn get_status_proto(
        &self,
        mut request: GetStatusRequest,
    ) -> Result<GetStatusResponse> {
        self.ensure_open()?;
        request.session_id = self.client_session_id();
        request.client_type = Some(self.client_type()?);
        if let Some(user_id) = self.request_user_id().filter(|user_id| !user_id.is_empty()) {
            request.user_context = Some(UserContext {
                user_id,
                ..Default::default()
            });
        }
        if let Some(observed) = self.observed_server_session_id() {
            request.client_observed_server_side_session_id = Some(observed);
        }
        let response = self
            .with_retry(|| {
                let deadline = self.call_rpc_deadlines().get_status;
                let mut req = with_deadline(Request::new(request.clone()), deadline);
                self._attach_metadata(&mut req);
                let mut stub = self.stub();
                async move {
                    within_deadline(deadline, stub.get_status(req))
                        .await
                        .map(Response::into_inner)
                }
            })
            .await?;
        self.verify_response(&response.session_id, &response.server_side_session_id)?;
        Ok(response)
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
        self.ensure_channel_open()?;
        // Each path is named as PySpark's ArtifactManager names it, by the flags and its file
        // extension (see `parse_artifact`).
        for path in paths {
            let (name, local) = crate::artifact::parse_artifact(path, pyfile, archive, file)
                .map_err(|e| SparkError::connect_msg(e.to_string()))?;
            let artifact = NamedArtifact::new(name, Box::new(FileArtifact::new(local)));
            self.upload_artifact(&artifact, false).await?;
        }
        Ok(())
    }

    /// Upload a single local file as an artifact under an explicit artifact name
    /// (rather than deriving `files/`/`jars/`… from the extension).
    ///
    /// Used by `SparkSession::copy_from_local_to_fs`, which names the artifact
    /// `forward_to_fs/<dest>` so the server writes it to the target filesystem path.
    pub async fn add_named_artifact(&self, name: &str, local_path: &str) -> Result<()> {
        self.ensure_channel_open()?;
        let artifact =
            NamedArtifact::new(name.to_string(), Box::new(FileArtifact::new(local_path)));
        // As PySpark's `copy_from_local_to_fs`, which leaves the failure to grpc.
        self.upload_artifact(&artifact, true).await
    }

    /// Upload session-scoped cache blobs, returning SHA-256 hashes in input order.
    /// Existing artifacts and duplicate blobs are uploaded only once.
    ///
    /// As `SparkConnectClient.cache_artifacts`, the status check and the upload are one
    /// attempt under the retry policy, so a transient failure of either retries both.
    pub async fn cache_artifacts(&self, blobs: Vec<Vec<u8>>) -> Result<Vec<String>> {
        self.ensure_channel_open()?;
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
        // As PySpark's `cache_artifact`, which leaves the failure to grpc.
        let uploaded = self
            .with_retry_left_to_grpc(LeftToGrpc::Converted, || async move {
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
    /// is missing, and takes a missing user ID from the user of the requests (the `user_id` of
    /// the channel, unless [`SparkConnectClient::set_request_user_id`] set another), as the
    /// reference client sets `user_context.user_id` on the requests it builds of its `_user_id`.
    /// The server keys a session by the user ID of each request, so without it the request
    /// would reach another session than the RPCs this client builds itself. An explicit user
    /// ID, the user name and the extensions are kept; raw requests are sent as given.
    fn fill_user_context(&self, context: &mut Option<UserContext>) {
        let context = context.get_or_insert_with(UserContext::default);
        if context.user_id.is_empty() {
            if let Some(user_id) = self.request_user_id() {
                context.user_id = user_id;
            }
        }
    }

    /// The client type of a typed request that the caller built: the user agent, unless it has
    /// one, as PySpark's requests carry `client_type=self._builder.userAgent`. Its error is that
    /// of the request. A request fills it before anything else, as PySpark builds a request
    /// before it reads `_stub`, which raises `NO_ACTIVE_SESSION` once the client is closed.
    fn fill_client_type(&self, client_type: &mut Option<String>) -> Result<()> {
        if client_type.is_none() {
            *client_type = Some(self.client_type()?);
        }
        Ok(())
    }

    /// The user agent as the client type of a request, as PySpark's `ChannelBuilder.userAgent`,
    /// or the error that the property raises when a request is built: `SparkConnectException`
    /// for a user agent longer than 2048 characters after URL escaping, or `UnicodeEncodeError`
    /// for a `SPARK_CONNECT_USER_AGENT` that is not UTF-8 (see [`ChannelBuilder::user_agent`]).
    /// The session is made all the same, and the requests that PySpark builds without the user
    /// agent, those of the artifacts and `CloneSession`, are sent.
    fn client_type(&self) -> Result<String> {
        self.user_agent.get()
    }

    /// Read the user agent of the requests of this client and of its clones again, of the
    /// connection string, else of `SPARK_CONNECT_USER_AGENT` in the environment of the process
    /// now (see [`ChannelBuilder::user_agent`]): as the Python binding reads it as each of its
    /// calls starts, while it holds the GIL, under which Python changes the environment, as
    /// PySpark's requests read `os.environ` when each is built. Not a stable API.
    #[doc(hidden)]
    pub fn refresh_user_agent(&self) {
        let user_agent = self.builder.user_agent();
        // Read first, without a lock: the user agent of a call is that of the one before it but
        // when the environment changed.
        let unchanged = self
            .user_agent
            .read(|current| match (current, &user_agent) {
                (Ok(current), Ok(user_agent)) => current == user_agent,
                _ => false,
            });
        if !unchanged {
            self.user_agent.set(user_agent);
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
    /// holds one chunk of the artifact at a time. With `left_to_grpc`, of an upload that PySpark's
    /// client leaves to grpc, its failure is not converted (see [`Self::result_left_to_grpc`]).
    async fn upload_artifact(&self, artifact: &NamedArtifact, left_to_grpc: bool) -> Result<()> {
        let mut retries = self.retries_of(GrpcCall::Unary).ending_with_abort();
        let attempts = self
            .attempts_in(&mut retries, || async move {
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
            .await;
        let response = if left_to_grpc {
            self.result_left_to_grpc(attempts, LeftToGrpc::Converted)
                .await?
        } else {
            self.result_of_attempts(attempts, GrpcCall::Unary).await?
        };
        // The IDs of the response are not checked, as by PySpark's ArtifactManager.
        check_artifact_upload(response, None)
    }

    /// The AddArtifacts call of `requests`, once. Stamps user_context on every frame as it is
    /// sent; all upload paths must go through here. As PySpark's ArtifactManager, the frames
    /// carry no client type and no observed server-side session ID, and the ID of the session,
    /// which a set of the client's session ID does not change, as the manager keeps the ID of
    /// the client when it was built.
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
        self.attach_artifact_metadata(&mut req);
        within_deadline(deadline, self.stub().add_artifacts(req))
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
        self.ensure_channel_open()?;
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
    /// session ID, and the ID of the session, as the frames of `add_artifacts_rpc`.
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
        self.attach_artifact_metadata(&mut req);
        within_deadline(deadline, self.stub().artifact_status(req))
            .await
            .map(Response::into_inner)
    }

    /// Attach metadata headers to a request.
    ///
    /// Inserts the metadata of the connection string and the token (if present) into the gRPC
    /// metadata. They were validated by `connect`. The `user-agent` header is tonic's own. Of a
    /// source of the metadata ([`SparkConnectClient::set_metadata_source`]), the `authorization`
    /// of the token only, and the mark of the source, which gives the rest as the request is sent
    /// (see `crate::metadata::Checked`).
    fn _attach_metadata<T>(&self, req: &mut Request<T>) {
        match self.metadata_source.get().calls {
            Some(source) => self.attach_pending_metadata(req, source),
            None => self.attach_static_metadata(req),
        }
    }

    /// Attach the metadata of a request of the artifacts, as PySpark's ArtifactManager sends the
    /// `metadata()` that its client read when it was made: of the source of the artifacts, if
    /// any (see [`SparkConnectClient::set_metadata_source`]), else of the builder.
    fn attach_artifact_metadata<T>(&self, req: &mut Request<T>) {
        match self.metadata_source.get().artifacts {
            Some(source) => self.attach_pending_metadata(req, source),
            None => self.attach_static_metadata(req),
        }
    }

    /// Attach the `authorization` of the token to a request, and the mark of `source`, which
    /// gives the rest of its metadata as it is sent (see `crate::metadata::Checked`).
    fn attach_pending_metadata<T>(
        &self,
        req: &mut Request<T>,
        source: std::sync::Arc<dyn crate::channel::MetadataSource>,
    ) {
        // After the metadata of the source, as grpcio's call credentials (see
        // `crate::metadata::append_metadata`).
        req.extensions_mut()
            .insert(crate::metadata::PendingMetadata {
                source,
                after: self.call_credentials.clone(),
            });
    }

    /// Attach the metadata of the builder, and the token, to a request.
    fn attach_static_metadata<T>(&self, req: &mut Request<T>) {
        let metadata = req.metadata_mut();
        // NOTE: the session id is carried in the request *body* (proto), not in gRPC
        // metadata. The reference client sends no `session_id` header; sending one
        // (with our own client-generated id, which differs from the body's session id
        // when we forward a request built by another client) can make the server
        // associate work with the wrong session. So we do not attach it here.
        // No user_id header: as PySpark's `ChannelBuilder.metadata()`, which leaves the user ID
        // to the user context of the requests.
        // Of every value of a key, in their order, the token's `authorization` last, as grpcio
        // sends the metadata of a call and then its call credentials.
        for (key, value) in &self.headers {
            metadata.append(key.clone(), value.clone());
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

/// The sources of the metadata of a client (see [`SparkConnectClient::set_metadata_source`]): of
/// its calls, and of the requests of its artifacts.
#[derive(Clone, Default)]
struct MetadataSources {
    calls: Option<std::sync::Arc<dyn crate::channel::MetadataSource>>,
    artifacts: Option<std::sync::Arc<dyn crate::channel::MetadataSource>>,
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
    /// The client that started the execution, whose stub, headers, session and retry policy it
    /// uses.
    client: SparkConnectClient,
    /// The gRPC metadata of the requests of the execution, as it started (see
    /// `SparkConnectClient::metadata_of_execution`).
    metadata: MetadataMap,
    /// What PySpark's client logs of the execution, of its responses as they are read.
    log: crate::client_log::ExecutionLog,
    /// Whether the execution is reattachable, as
    /// [`SparkConnectClient::set_reattachable_execute`] sets the executions of a client.
    reattachable: bool,
    /// The retries of the first wait for a response, which carry on the attempts of the first RPC
    /// that started the execution: as the reference's first `_has_next` retries the initial
    /// ExecutePlan together with the reads of its first response, the policy's attempts are shared
    /// by both. Taken by that wait; the later waits have their own. Those of an execution that is
    /// not reattachable are of the whole execution, as PySpark's one `Retrying` around its stream.
    first_response_retries: Option<Retrying>,
    iter: ExecutePlanResponseReattachableIterator,
    /// The stream of the execution, or none to reattach before reading, of the connection of the
    /// generation of the process that read it (see [`crate::fork`]).
    stream: Option<crate::fork::ForkLocal<Streaming<ExecutePlanResponse>>>,
    /// When the deadline of the stream passes, as gRPC's of the call that opened it.
    stream_deadline: Option<tokio::time::Instant>,
    /// Whether the next stream is opened by sending the ExecutePlan again, rather than by
    /// reattaching: the server does not know the execution, which never reached it.
    execute_again: bool,
    done: bool,
    /// Whether the stream ended, done or dropped, which its releases in the background share
    /// (see [`SparkConnectClient::wait_for_releases`]).
    stream_ended: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Whether `ReleaseAll` was sent, after which nothing else is released.
    released: bool,
    /// The registration of the execution as in flight, which ends with the stream.
    _inflight: InflightGuard,
    /// The generation of the process that started the execution (see [`crate::fork`]).
    generation: u64,
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

/// Whether the releases of executions are refused. The reference client sends its releases
/// through a `ThreadPoolExecutor`, whose `submit` refuses new work once the interpreter has
/// begun to shut down. The Python binding sets this at that point, with
/// [`refuse_releases`]. The Rust API never sets it.
static RELEASES_REFUSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Refuse the releases of executions for the rest of the process, as the reference client's
/// thread pool refuses them once the interpreter has begun to shut down. Each reattachable
/// execution then fails where it would schedule a release, after its first response or at its
/// failure, with that pool's `RuntimeError`, and sends no ReleaseExecute. For the exit of the
/// Python binding; not a stable API.
#[doc(hidden)]
pub fn refuse_releases() {
    RELEASES_REFUSED.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// The error of a release once releases are refused, as `ThreadPoolExecutor.submit` raises it
/// after the interpreter has begun to shut down; none before.
fn release_refused() -> Option<SparkError> {
    RELEASES_REFUSED
        .load(std::sync::atomic::Ordering::SeqCst)
        .then(|| {
            SparkError::builtin_runtime_msg(
                "cannot schedule new futures after interpreter shutdown",
            )
        })
}

impl SparkConnectClient {
    /// The gRPC metadata of every request of a reattachable execution that starts now, as
    /// PySpark's iterator of the execution is made of a `metadata()`: of the source of the
    /// metadata of the calls, read once (see [`SparkConnectClient::set_metadata_source`]), with
    /// the `authorization` of the token, or the `INTERNAL` of metadata that gRPC does not send,
    /// which fails the requests; else of the builder. The error is that of the source.
    async fn metadata_of_execution(
        &self,
    ) -> Result<std::result::Result<MetadataMap, tonic::Status>> {
        let Some(source) = self.metadata_source.get().calls else {
            return Ok(Ok(self.execution_metadata()));
        };
        let pairs = source.metadata().await?;
        Ok(crate::metadata::call_metadata(&pairs).map(|mut metadata| {
            // After the metadata of the source, as grpcio's call credentials.
            for (key, value) in &self.call_credentials {
                metadata.append(key.clone(), value.clone());
            }
            metadata
        }))
    }

    /// The gRPC metadata of every request about an execution.
    fn execution_metadata(&self) -> MetadataMap {
        let mut metadata = MetadataMap::new();
        for (key, value) in &self.headers {
            metadata.append(key.clone(), value.clone());
        }
        metadata
    }

    /// Sends `request` in the background, so that reading is not delayed, retrying under the
    /// client's retry policy. Like the reference client, a failure is only logged: the server
    /// removes an execution that is never released once it has been detached for
    /// `spark.connect.execute.manager.detachedTimeout`. As the reference's, the request carries
    /// no observed server-side session ID.
    fn spawn_release(
        &self,
        metadata: MetadataMap,
        request: ReleaseExecuteRequest,
        stream_ended: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) {
        let stub = self.stub();
        let mut retries = self.retries_of(GrpcCall::Unary);
        let channel_closed = std::sync::Arc::clone(&self.channel_closed);
        let done = std::sync::Arc::<ReleaseDone>::default();
        releases_of(&self.releases).push(PendingRelease {
            stream_ended,
            done: std::sync::Arc::clone(&done),
        });
        let ends = ReleaseEnds(done);
        let release = async move {
            let _ends = ends;
            // As the reference's, which goes through the stub of its iterator rather than the
            // `_stub` of the client: a release is retried whether or not the client was closed,
            // but grpc refuses each of its attempts once the channel is closed, as by the `stop()`
            // of another thread, of its `ValueError` of a closed channel, which is not retried.
            let channel_open = || {
                if channel_closed.is_closed() {
                    return Err(SparkError::builtin_value_msg(
                        "Cannot invoke RPC on closed channel!",
                    ));
                }
                Ok(())
            };
            if let Err(refused) = channel_open() {
                background_failure(
                    BackgroundRequest::ReleaseExecute,
                    RetryError::<SparkError>::Failed(Box::new(refused)),
                );
                return;
            }
            let released = retrying(&mut retries, channel_open, || {
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
                // Its backoffs are waited whole.
                Err(RetryError::WaitAborted) => {}
                // Of a channel closed during its backoff.
                Err(RetryError::Aborted(refused)) => background_failure(
                    BackgroundRequest::ReleaseExecute,
                    RetryError::<SparkError>::Failed(Box::new(refused)),
                ),
                Err(failed) => {
                    tracing::warn!(
                        "ReleaseExecute of operation {} failed: {failed:?}",
                        request.operation_id
                    );
                    background_failure(BackgroundRequest::ReleaseExecute, failed);
                }
            }
        };
        // A stream is read inside a runtime, but it can be dropped outside of any, for
        // example after the `block_on` that read it.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => drop(handle.spawn(release)),
            Err(_) => drop(crate::runtime::get_runtime().spawn(release)),
        }
    }

    /// Send `request`, the execution of a finalizer, as PySpark sends the removal of a cached
    /// remote relation in `CachedRemoteRelation.__del__`: through the channel itself, as a
    /// `unary_unary` call of ExecutePlan, so neither reattachable, with no ReleaseExecute after
    /// it, nor passed to the hooks of the executions. Its request is that of
    /// `_execute_plan_request_with_metadata`, whose only option is the result chunking of the
    /// client, and the call is retried under the retry policy of the client, as `_retrying()`
    /// retries it.
    ///
    /// The request is completed on the calling thread, whose user context extensions it takes,
    /// as PySpark's is of the thread of its finalizer. It is sent on the shared runtime without
    /// waiting for the server, so that a drop never blocks on it: a drop may run on a thread of a
    /// runtime, or in a Python finalizer that holds the GIL, which another thread may need to
    /// read the responses of the same connection, whose flow control would hold this response
    /// back meanwhile. A failure is only logged, as PySpark's, and nothing is sent once the
    /// client is closed. Not a stable API.
    #[doc(hidden)]
    pub fn spawn_unary_execute_plan(&self, mut request: ExecutePlanRequest) {
        // Neither is sent of a closed channel, whose call PySpark's finalizer would fail.
        if self.is_closed() || self.channel_is_closed() {
            return;
        }
        add_result_chunking_option(&mut request, self.call_result_chunking());
        self.extend_user_context(&mut request.user_context);
        self.add_call_stack(&mut request.user_context);
        self.fill_user_context(&mut request.user_context);
        // A user agent that PySpark refuses raises as its finalizer builds the request, before
        // anything is sent, and the error of a finalizer is only reported.
        if let Err(error) = self.fill_client_type(&mut request.client_type) {
            tracing::warn!("ExecutePlan of a finalizer was not sent: {error}");
            return;
        }
        self.observe_on(&mut request.client_observed_server_side_session_id);
        let operation_id = request
            .operation_id
            .get_or_insert_with(|| Uuid::new_v4().to_string())
            .clone();
        let stub = self.stub();
        let metadata = self.execution_metadata();
        // As PySpark's finalizer, whose call reads the `metadata()` of a builder of its own for
        // each attempt: of the source of the metadata of the calls, if any, of which the first
        // attempt is read now, on the thread of the finalizer, where it can be.
        let source = self.metadata_source.get().calls;
        let after = self.call_credentials.clone();
        let mut first = match source.as_ref().and_then(|source| source.metadata_now()) {
            Some(Ok(pairs)) => match crate::metadata::call_metadata(&pairs) {
                Ok(mut read) => {
                    for (key, value) in &after {
                        read.append(key.clone(), value.clone());
                    }
                    Some(read)
                }
                Err(status) => {
                    tracing::warn!("ExecutePlan of a finalizer was not sent: {status}");
                    return;
                }
            },
            // As PySpark's finalizer, whose `except Exception` takes the error of its call.
            Some(Err(error)) => {
                tracing::warn!("ExecutePlan of a finalizer was not sent: {error}");
                return;
            }
            None => None,
        };
        let mut retries = self.retries_of(GrpcCall::Unary);
        let closed = std::sync::Arc::clone(&self.closed);
        let send = async move {
            // A client closed meanwhile makes no more attempts, as the channel of a closed
            // PySpark client refuses its calls.
            let still_open = || {
                if closed.load(std::sync::atomic::Ordering::SeqCst) {
                    Err(())
                } else {
                    Ok(())
                }
            };
            let sent = retrying(&mut retries, still_open, || {
                let mut req = Request::new(request.clone());
                match (first.take(), &source) {
                    (Some(read), _) => *req.metadata_mut() = read,
                    (None, Some(source)) => {
                        req.extensions_mut()
                            .insert(crate::metadata::PendingMetadata {
                                source: std::sync::Arc::clone(source),
                                after: after.clone(),
                            });
                    }
                    (None, None) => *req.metadata_mut() = metadata.clone(),
                }
                let mut stub = stub.clone();
                async move {
                    let mut responses = stub
                        .execute_plan(req)
                        .await
                        .map_err(RetryError::Status)?
                        .into_inner();
                    // The call ends with the responses of the execution, as the unary call
                    // of the reference reads them.
                    while responses
                        .message()
                        .await
                        .map_err(RetryError::Status)?
                        .is_some()
                    {}
                    Ok(())
                }
            })
            .await;
            match sent {
                Ok(()) | Err(RetryError::Aborted(())) | Err(RetryError::WaitAborted) => {}
                Err(failed) => {
                    tracing::warn!("ExecutePlan of operation {operation_id} failed: {failed:?}");
                    background_failure(BackgroundRequest::RemoveCachedRemoteRelation, failed);
                }
            }
        };
        drop(crate::runtime::get_runtime().spawn(send));
    }
}

impl ReattachableResponseStream {
    /// Whether the responses of the execution are logged as PySpark's client logs them (see
    /// [`crate::client_log::ExecutionLog`]): a reader that joins the chunks of an Arrow batch logs
    /// the batch once it is whole, as PySpark's `_execute_and_fetch_as_iterator` does
    /// (`Assembling arrow batch ...`). Not a stable API.
    #[doc(hidden)]
    pub fn logs_responses(&self) -> bool {
        self.log.logs_responses()
    }

    /// The handle of the execution of this stream, to mark the waits of the caller that starts it
    /// for the execution while the caller reads its responses from another thread (see
    /// [`InflightHandle::waiting`]).
    pub fn inflight_handle(&self) -> InflightHandle {
        self._inflight.handle()
    }

    /// The gRPC metadata of every request about this execution.
    fn metadata_map(&self) -> MetadataMap {
        self.metadata.clone()
    }

    fn attach_metadata<T>(&self, req: &mut Request<T>) {
        add_metadata(req, &self.metadata);
    }

    /// Next response, transparently reattaching whenever the stream ends before
    /// `ResultComplete`.
    pub async fn message(&mut self) -> Result<Option<ExecutePlanResponse>> {
        if self.done {
            return Ok(None);
        }
        // Each wait for the next response is a wait of the caller for the execution.
        let _waiting = self._inflight.waiting();
        if !self.reattachable {
            let mut retrying = match self.first_response_retries.take() {
                Some(retries) => retries,
                None => self
                    .client
                    .retries_of(GrpcCall::ServerStreaming)
                    .ending_with_abort(),
            };
            let next = self.next_of_one_execution(&mut retrying).await;
            self.first_response_retries = Some(retrying);
            return next;
        }
        // As the reference iterator's `_has_next` retries with a `Retrying` of its own, the
        // reattaches of each response have the attempts and the elapsed time of the policy.
        // The clock of the retries that the reference makes as `RetryException`s, bounded by the
        // policy's elapsed time, starts at the first deadline that passes, the deadline of the
        // initial ExecutePlan included, and the reattaches after a deadline or an unknown
        // execution are its retries.
        let mut retrying = match self.first_response_retries.take() {
            Some(retries) => retries,
            None => self
                .client
                .retries_of(GrpcCall::ServerStreaming)
                .ending_with_abort()
                .reading_status_first(true),
        };
        if self.stream.is_none() {
            // The initial ExecutePlan passed its deadline, which PySpark's iterator logs as it
            // reads it, then a RetryException that fails at once when the bound of the retries
            // has passed, as it has when the bound is 0.
            if let Err(error) = self.log_deadline().await {
                return Err(self.fail(error));
            }
            let retried = match retrying.retry_at_once() {
                Ok(retried) => retried,
                Err(error) => return Err(self.fail(error)),
            };
            if !retried {
                let status = tonic::Status::deadline_exceeded("Deadline Exceeded");
                return Err(self
                    .fail_exceeded(status, RetriesExceeded::RetryException)
                    .await);
            }
        }
        loop {
            let Some(stream) = self.stream.as_mut() else {
                self.reattach(&mut retrying).await?;
                continue;
            };
            let next = if self.client.channel_is_closed() {
                // As grpcio's `close` of the channel, which cancels its calls: the stream of the
                // execution of a client whose channel was closed since, as by `stop()`, fails at
                // its next read, of the responses that it holds too, unless it ended already (see
                // `read_after_close`).
                read_after_close(stream)
            } else {
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
                match crate::runtime::until_abort(crate::runtime::ABORT_ALL, read).await {
                    // A response that a read in flight gives once the client closed the channel,
                    // which grpcio's cancel would not give (see `read_after_close_of`).
                    Some(Ok(Some(_))) if self.client.channel_is_closed() => {
                        read_after_close_of(stream, true)
                    }
                    Some(next) => next,
                    None => return Err(self.fail(wait_aborted())),
                }
            };
            // As the reference's `_call_iter`, which reads the status of each failure first.
            if let Err(status) = &next {
                if let Some(mismatch) = self.mismatch(status) {
                    return Err(self.fail(mismatch));
                }
            }
            match next {
                Ok(Some(resp)) => {
                    if let Err(error) = self
                        .client
                        .verify_response(&resp.session_id, &resp.server_side_session_id)
                    {
                        return Err(self.fail(error));
                    }
                    self.iter.set_last_response_id(&resp).await;
                    // As the reference's `send`, whose release raises before it returns the
                    // response once releases are refused: the stream ends with that error.
                    let released = if self.iter.is_completed().await {
                        self.end();
                        self.release_all()
                    } else if !resp.response_id.is_empty() {
                        self.release_until(&resp.response_id)
                    } else {
                        Ok(())
                    };
                    if let Err(refused) = released {
                        self.end();
                        return Err(refused);
                    }
                    if let Err(error) = self.log_response(&resp).await {
                        return Err(self.fail(error));
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
                    if let Err(error) = self.log_deadline().await {
                        return Err(self.fail(error));
                    }
                    let retried = match retrying.retry_at_once() {
                        Ok(retried) => retried,
                        Err(error) => return Err(self.fail(error)),
                    };
                    if !retried {
                        return Err(self
                            .fail_exceeded(status, RetriesExceeded::RetryException)
                            .await);
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
                    let retried = match retrying.retry_at_once() {
                        Ok(retried) => retried,
                        Err(error) => return Err(self.fail(error)),
                    };
                    if !retried {
                        return Err(self
                            .fail_exceeded(status, RetriesExceeded::RetryException)
                            .await);
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
                        Backoff::Exceeded => {
                            return Err(self.fail_exceeded(status, RetriesExceeded::Policy).await)
                        }
                        Backoff::Aborted => return Err(self.fail(wait_aborted())),
                        Backoff::Failed(error) => return Err(self.fail(*error)),
                    }
                    self.reattach(&mut retrying).await?;
                }
            }
        }
    }

    /// The next response of an execution that is not reattachable, as PySpark's client reads its
    /// one `ExecutePlan` inside a `Retrying`: the stream ends with the end of the call, and a
    /// failure that the policy retries, of the call or of its stream, sends the `ExecutePlan` again
    /// after its backoff, whose responses follow. `retrying` is that of the whole execution.
    async fn next_of_one_execution(
        &mut self,
        retrying: &mut Retrying,
    ) -> Result<Option<ExecutePlanResponse>> {
        loop {
            let Some(stream) = self.stream.as_mut() else {
                // As PySpark's `_stub`, which each attempt reads: once the client is closed, the
                // ExecutePlan sent again raises NO_ACTIVE_SESSION.
                if let Err(error) = self.client.ensure_open() {
                    return Err(self.fail(error));
                }
                let opened = match crate::runtime::until_abort(
                    crate::runtime::ABORT_RETRIES,
                    self.open_execute_plan_again(),
                )
                .await
                {
                    Some(opened) => opened,
                    None => return Err(self.fail(wait_aborted())),
                };
                match opened {
                    Ok(stream) => self.stream = Some(crate::fork::ForkLocal::new(stream)),
                    Err(status) => match retrying.backoff(&status).await {
                        Backoff::Retry => {}
                        Backoff::GiveUp => return Err(self.fail_with_status(status).await),
                        Backoff::Exceeded => {
                            return Err(self.fail_exceeded(status, RetriesExceeded::Policy).await)
                        }
                        Backoff::Aborted => return Err(self.fail(wait_aborted())),
                        Backoff::Failed(error) => return Err(self.fail(*error)),
                    },
                }
                continue;
            };
            let next = if self.client.channel_is_closed() {
                // As grpcio's `close` of the channel, which cancels its calls (see
                // `read_after_close`).
                read_after_close(stream)
            } else {
                match crate::runtime::until_abort(crate::runtime::ABORT_ALL, stream.message()).await
                {
                    // As in the reattachable stream (see `read_after_close_of`).
                    Some(Ok(Some(_))) if self.client.channel_is_closed() => {
                        read_after_close_of(stream, true)
                    }
                    Some(next) => next,
                    None => return Err(self.fail(wait_aborted())),
                }
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
                    if let Err(error) = self.log_response(&resp).await {
                        return Err(self.fail(error));
                    }
                    return Ok(Some(resp));
                }
                Ok(None) => {
                    self.end();
                    return Ok(None);
                }
                Err(status) => match retrying.backoff(&status).await {
                    Backoff::Retry => self.stream = None,
                    Backoff::GiveUp => return Err(self.fail_with_status(status).await),
                    Backoff::Exceeded => {
                        return Err(self.fail_exceeded(status, RetriesExceeded::Policy).await)
                    }
                    Backoff::Aborted => return Err(self.fail(wait_aborted())),
                    Backoff::Failed(error) => return Err(self.fail(*error)),
                },
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
            // As the reference's iterator, which sends its reattaches and its ExecutePlans again
            // through the stub that it was made with: once the channel of the client is closed, as
            // by the `stop()` of another thread, they raise grpc's `ValueError` of a closed
            // channel, and the stream ends; a client closed of its session alone, as of a session
            // change, still sends them.
            if let Err(error) = self.client.ensure_channel_open_for_stream() {
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
                    self.stream = Some(crate::fork::ForkLocal::new(stream));
                    self.execute_again = false;
                    return Ok(());
                }
                Err(status) => status,
            };
            if let Some(mismatch) = self.mismatch(&status) {
                return Err(self.fail(mismatch));
            }
            if let Some(error_type) = unknown_operation(&status) {
                if self.iter.last_response_id().await.is_some() {
                    return Err(self.fail(response_already_received(error_type)));
                }
                let retried = match retrying.retry_at_once() {
                    Ok(retried) => retried,
                    Err(error) => return Err(self.fail(error)),
                };
                if !retried {
                    return Err(self
                        .fail_exceeded(status, RetriesExceeded::RetryException)
                        .await);
                }
                self.execute_again = true;
                continue;
            }
            if status.code() == tonic::Code::DeadlineExceeded {
                // As the reference: a call past its deadline, whether it is sent again or
                // reattached, is retried at once, the execution going on on the server.
                if let Err(error) = self.log_deadline().await {
                    return Err(self.fail(error));
                }
                let retried = match retrying.retry_at_once() {
                    Ok(retried) => retried,
                    Err(error) => return Err(self.fail(error)),
                };
                if !retried {
                    return Err(self
                        .fail_exceeded(status, RetriesExceeded::RetryException)
                        .await);
                }
                self.execute_again = false;
                continue;
            }
            match retrying.backoff(&status).await {
                Backoff::Retry => continue,
                Backoff::GiveUp => return Err(self.fail_with_status(status).await),
                Backoff::Exceeded => {
                    return Err(self.fail_exceeded(status, RetriesExceeded::Policy).await)
                }
                Backoff::Aborted => return Err(self.fail(wait_aborted())),
                Backoff::Failed(error) => return Err(self.fail(*error)),
            }
        }
    }

    /// Log `response`, which the stream returns, as PySpark's `_execute_and_fetch_as_iterator`
    /// logs each response that it reads, of an execution whose responses are logged (see
    /// [`log_response`]): the error of a record, which ends the stream.
    async fn log_response(&self, response: &ExecutePlanResponse) -> Result<()> {
        // Boxed, of a response that is logged only: the stream's future holds its pointer.
        if self.log.logs_responses() && crate::client_log::enabled(crate::client_log::Level::Debug)
        {
            return Box::pin(log_response(response)).await;
        }
        Ok(())
    }

    /// Log that the deadline of the stream passed, as the `_call_iter` of PySpark's iterator logs
    /// it before the iterator reattaches (`Deadline exceeded on stream for operation ...`): the
    /// error of the record, which ends the stream in place of the reattach.
    async fn log_deadline(&self) -> Result<()> {
        use crate::client_log::{Level, Message};
        if self.log == crate::client_log::ExecutionLog::Silent
            || !crate::client_log::enabled(Level::Debug)
        {
            return Ok(());
        }
        Box::pin(async {
            let last = self.iter.last_response_id().await;
            crate::client_log::logged(Level::Debug, || {
                Message::Text(format!(
                    "Deadline exceeded on stream for operation {}; will reattach. (last \
                     response: {})",
                    self.iter.operation_id(),
                    last.as_deref().unwrap_or("None")
                ))
            })
            .await
        })
        .await
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
        let response = within_deadline(deadline, self.client.stub().reattach_execute(req)).await?;
        Ok(response.into_inner())
    }

    /// ExecutePlan of the request again, as the initial one, bounded by its deadline, which an
    /// execution that is not reattachable has none of.
    async fn open_execute_plan_again(
        &mut self,
    ) -> std::result::Result<Streaming<ExecutePlanResponse>, tonic::Status> {
        let deadline = if self.reattachable {
            self.client.rpc_deadlines.reattachable_execute_plan
        } else {
            None
        };
        let request = clone_on_grown_stack(self.iter.request());
        let mut req = with_deadline(Request::new(request), deadline);
        self.attach_metadata(&mut req);
        self.stream_deadline = stream_deadline_in(deadline);
        let response = within_deadline(deadline, self.client.stub().execute_plan(req)).await?;
        Ok(response.into_inner())
    }

    /// The `ValueError` of a failure `status` of the stream of a reattachable execution whose
    /// `google.rpc.Status` does not match it, as the `_call_iter` of PySpark's iterator raises it
    /// of `rpc_status.from_call` before its retries see the failure, which is not logged (see
    /// [`crate::error::from_call_error`]).
    fn mismatch(&self, status: &tonic::Status) -> Option<SparkError> {
        if !self.reattachable {
            return None;
        }
        crate::error::from_call_error(
            status,
            GrpcCall::ServerStreaming,
            crate::error::StatusRead::Status,
        )
    }

    /// Ends the stream with the error of `status`, as [`Self::fail`].
    async fn fail_with_status(&mut self, status: tonic::Status) -> SparkError {
        let error = self
            .client
            .error_from_status_of(status, GrpcCall::ServerStreaming)
            .await;
        self.fail(error)
    }

    /// Ends the stream with the failure `status` of retries that are spent, which `exceeded`
    /// tells, as [`Self::fail`]: with the error that [`SparkConnectClient::exceeded_error`] makes
    /// of it, as PySpark's `Retrying` warns that they are before it raises the failure.
    async fn fail_exceeded(
        &mut self,
        status: tonic::Status,
        exceeded: RetriesExceeded,
    ) -> SparkError {
        let error = self
            .client
            .exceeded_error(status, exceeded, GrpcCall::ServerStreaming)
            .await;
        self.fail(error)
    }

    /// Ends the stream with `error`, releasing the execution as the reference client does
    /// when an execution fails.
    /// The error carries the ExecutePlan operation id (`SparkConnectException.operation_id`).
    /// Once releases are refused, the error is that of the refused release, as the reference's
    /// `_has_next` raises the error of its release of the failed execution.
    fn fail(&mut self, error: SparkError) -> SparkError {
        self.end();
        match self.release_all() {
            Ok(()) => error.with_operation_id(self.iter.operation_id()),
            Err(refused) => refused,
        }
    }

    /// Releases the responses up to and including `response_id`, which were read. Once
    /// releases are refused, nothing is sent, and the refusal is the error.
    fn release_until(&self, response_id: &str) -> Result<()> {
        if !self.released {
            if let Some(refused) = release_refused() {
                return Err(refused);
            }
            self.spawn_release(self.iter.release_until_request(response_id));
        }
        Ok(())
    }

    /// Releases the whole execution. Once releases are refused, nothing is sent, and the
    /// refusal is the error.
    fn release_all(&mut self) -> Result<()> {
        if !self.released {
            if let Some(refused) = release_refused() {
                return Err(refused);
            }
            self.released = true;
            self.spawn_release(self.iter.release_all_request());
        }
        Ok(())
    }

    /// Sends `request` in the background, as [`SparkConnectClient::spawn_release`] does.
    fn spawn_release(&self, request: ReleaseExecuteRequest) {
        self.client.spawn_release(
            self.metadata_map(),
            request,
            std::sync::Arc::clone(&self.stream_ended),
        );
    }

    /// Ends the stream: nothing more is read of it, and its releases are those of an ended
    /// stream, which the close of the session waits for.
    fn end(&mut self) {
        self.done = true;
        self.stream_ended
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Drop for ReattachableResponseStream {
    /// A stream dropped before its result is complete, for example by a caller that stopped
    /// reading, still holds its execution on the server: release it, as the reference client
    /// does when its iterator is closed. A release that is refused sends nothing. In a child of a
    /// `fork()`, the execution of a stream of the parent is the parent's, which reads and releases
    /// it: the child releases nothing of it, and leaves its stream, of the parent's connection,
    /// as it is (see [`crate::fork`]).
    fn drop(&mut self) {
        self.stream_ended
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if self.generation == crate::fork::generation() {
            let _ = self.release_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssl_roots_follow_grpc_default_ssl_roots_file_path() {
        let dir = tempfile::tempdir().unwrap();
        let pem = dir.path().join("roots.pem");
        std::fs::write(&pem, b"-----BEGIN CERTIFICATE-----\n...").unwrap();
        let empty = dir.path().join("empty.pem");
        std::fs::write(&empty, b"").unwrap();
        // As grpcio: a readable file replaces the default roots.
        assert_eq!(
            ssl_roots_pem(Some(pem.clone().into_os_string())),
            Some(std::fs::read(&pem).unwrap())
        );
        // No variable, an empty one, an empty file or one that cannot be read: the defaults.
        for path in [
            None,
            Some(std::ffi::OsString::new()),
            Some(empty.into_os_string()),
            Some(dir.path().join("missing.pem").into_os_string()),
            Some(dir.path().as_os_str().to_owned()),
        ] {
            assert_eq!(ssl_roots_pem(path), None);
        }
    }

    #[test]
    fn the_roots_of_the_tls_channels_are_read_once_per_process() {
        // As grpcio computes its default roots once: the system's store too is read as the first
        // TLS channel of the process is made, and never again, so that every channel of the process
        // trusts the same roots, of which the subchannels are shared (see
        // `crate::connector::SubchannelArgs`).
        let read = |roots: crate::connector::Roots<'static>| match roots {
            crate::connector::Roots::Only(file) => (file.as_ptr(), None),
            crate::connector::Roots::System { native, bundle } => {
                (native.as_ptr().cast::<u8>(), bundle.map(<[u8]>::as_ptr))
            }
        };
        assert_eq!(read(ssl_roots()), read(ssl_roots()));
    }

    #[test]
    fn only_a_roots_file_that_cannot_be_read_is_logged() {
        // As grpcio, which logs the error of a file that it cannot load and takes an empty one as
        // no file, silently.
        let dir = tempfile::tempdir().unwrap();
        let empty = dir.path().join("empty.pem");
        std::fs::write(&empty, b"").unwrap();
        let missing = dir.path().join("missing.pem");
        let (roots, logged) = crate::grpc_log::logged_by(|| {
            [
                None,
                Some(empty.into_os_string()),
                Some(missing.clone().into_os_string()),
            ]
            .map(ssl_roots_pem)
        });
        assert_eq!(roots, [None, None, None]);
        assert_eq!(logged.len(), 1, "{logged:?}");
        let message = &logged[0].message;
        assert!(
            message.starts_with(&format!(
                "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH names {}, which cannot be read (",
                missing.display()
            )),
            "{message}"
        );
        assert!(message.ends_with("); using the default roots"), "{message}");
    }

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
        assert!(client.client_type().unwrap().contains("spark/"));
        // The header of the connection string is on a request. The user agent is not: it is
        // the client type of the requests, and tonic sends its own `user-agent` header.
        let mut request = Request::new(());
        client._attach_metadata(&mut request);
        assert_eq!(
            request.metadata().get("custom_header").unwrap(),
            "custom_value"
        );
        assert!(request.metadata().get("user-agent").is_none());
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
    fn test_an_execution_whose_failure_the_thread_converts_is_not_running() {
        // As PySpark's `_handle_error`, which converts the failure of an execution in its `except
        // Exception` clause: a signal during a call that a handler of a log of the conversion
        // makes interrupts the execution of that call, if any, and never the converted one.
        let client = connect_blocking("sc://example.com");
        let registry = || std::sync::Arc::clone(&client.inflight_operations);
        let failed = InflightGuard::register(registry(), "failed".to_string());
        let _wait_failed = failed.waiting();
        assert_eq!(client.running_operation_id().as_deref(), Some("failed"));
        let failure = crate::error::grpcio_error(
            &tonic::Status::internal("failed"),
            GrpcCall::ServerStreaming,
        );
        let converting = Converting::start(
            &failure,
            client.operation_id_of_this_call(),
            client.identity(),
        );
        assert_eq!(
            converting_failure().and_then(|conversion| conversion.operation_id),
            Some("failed".to_string())
        );
        // As a config read of a handler of a log of the conversion.
        assert_eq!(client.running_operation_id(), None);
        // As an execution of such a handler.
        let nested = InflightGuard::register(registry(), "nested".to_string());
        let wait_nested = nested.waiting();
        assert_eq!(client.running_operation_id().as_deref(), Some("nested"));
        drop(wait_nested);
        drop(nested);
        assert_eq!(client.running_operation_id(), None);
        // Another thread's conversion is not of the calling thread.
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    assert!(converting_failure().is_none());
                    let other = InflightGuard::register(registry(), "other".to_string());
                    let _wait = other.waiting();
                    assert_eq!(client.running_operation_id().as_deref(), Some("other"));
                })
                .join()
                .unwrap();
        });
        drop(converting);
        assert_eq!(client.running_operation_id().as_deref(), Some("failed"));
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

    /// A loopback service of ExecutePlan and ReattachExecute only. Its first `execute_failures`
    /// ExecutePlan calls answer UNAVAILABLE; a later one opens a stream whose first response is
    /// UNAVAILABLE, and every ReattachExecute answers UNAVAILABLE. The calls of each are counted.
    struct FirstResponseCalls {
        execute_plan: std::sync::atomic::AtomicUsize,
        reattach: std::sync::atomic::AtomicUsize,
        execute_failures: usize,
    }

    #[derive(Clone)]
    struct FirstResponseService(std::sync::Arc<FirstResponseCalls>);

    impl tonic::server::NamedService for FirstResponseService {
        const NAME: &'static str = "spark.connect.SparkConnectService";
    }

    impl tonic::codegen::Service<tonic::codegen::http::Request<tonic::body::Body>>
        for FirstResponseService
    {
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
                // Each method has a codec of its own, for the message it decodes.
                match request.uri().path() {
                    "/spark.connect.SparkConnectService/ExecutePlan" => {
                        let mut grpc = tonic::server::Grpc::new(tonic_prost::ProstCodec::default());
                        Ok(grpc.server_streaming(ExecutePlanRpc(state), request).await)
                    }
                    "/spark.connect.SparkConnectService/ReattachExecute" => {
                        let mut grpc = tonic::server::Grpc::new(tonic_prost::ProstCodec::default());
                        Ok(grpc
                            .server_streaming(ReattachExecuteRpc(state), request)
                            .await)
                    }
                    _ => Ok(tonic::codegen::http::Response::builder()
                        .status(200)
                        .header("grpc-status", "12")
                        .header("content-type", "application/grpc")
                        .body(tonic::body::Body::empty())
                        .unwrap()),
                }
            })
        }
    }

    struct ExecutePlanRpc(std::sync::Arc<FirstResponseCalls>);

    impl tonic::server::ServerStreamingService<ExecutePlanRequest> for ExecutePlanRpc {
        type Response = spark_connect_proto::ExecutePlanResponse;
        type ResponseStream = futures::stream::Iter<
            std::vec::IntoIter<std::result::Result<Self::Response, tonic::Status>>,
        >;
        type Future =
            tonic::codegen::BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;

        fn call(&mut self, _request: tonic::Request<ExecutePlanRequest>) -> Self::Future {
            let state = self.0.clone();
            Box::pin(async move {
                let call = state
                    .execute_plan
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if call < state.execute_failures {
                    return Err(tonic::Status::unavailable("the server is restarting"));
                }
                let lost = tonic::Status::unavailable("the first response is lost");
                Ok(tonic::Response::new(futures::stream::iter(vec![Err(lost)])))
            })
        }
    }

    struct ReattachExecuteRpc(std::sync::Arc<FirstResponseCalls>);

    impl tonic::server::ServerStreamingService<spark_connect_proto::ReattachExecuteRequest>
        for ReattachExecuteRpc
    {
        type Response = spark_connect_proto::ExecutePlanResponse;
        type ResponseStream = futures::stream::Iter<
            std::vec::IntoIter<std::result::Result<Self::Response, tonic::Status>>,
        >;
        type Future =
            tonic::codegen::BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;

        fn call(
            &mut self,
            _request: tonic::Request<spark_connect_proto::ReattachExecuteRequest>,
        ) -> Self::Future {
            let state = self.0.clone();
            Box::pin(async move {
                state
                    .reattach
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Err(tonic::Status::unavailable("the server is restarting"))
            })
        }
    }

    /// The first response of an execution shares the attempts of its first RPC, as the reference's
    /// first `_has_next` does: an ExecutePlan that is unavailable twice, then opens a stream whose
    /// first response is unavailable, and a ReattachExecute that is unavailable, make four calls
    /// under the three retries of the policy. Each of the two waits having its own budget would
    /// make six.
    #[tokio::test]
    async fn the_first_response_shares_the_attempts_of_the_first_rpc_as_pyspark() {
        let calls = std::sync::Arc::new(FirstResponseCalls {
            execute_plan: std::sync::atomic::AtomicUsize::new(0),
            reattach: std::sync::atomic::AtomicUsize::new(0),
            execute_failures: 2,
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let service = FirstResponseService(calls.clone());
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        // `connect` reads the token of the environment too, so none is set until it returns:
        // another test may set one, which would make the channel TLS.
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let builder = ChannelBuilder::parse(&format!("sc://127.0.0.1:{}", address.port())).unwrap();
        let mut client = SparkConnectClient::connect(&builder).await.unwrap();
        drop(env);
        client.set_retry_policy(RetryPolicy {
            max_retries: Some(3),
            initial_backoff_ms: 1,
            max_backoff_ms: Some(2),
            ..RetryPolicy::default()
        });

        let mut stream = client
            .execute_plan_reattachable(ExecutePlanRequest {
                session_id: client.session_id().to_string(),
                plan: Some(Plan {
                    op_type: Some(PlanOpType::Root(Default::default())),
                }),
                ..Default::default()
            })
            .await
            .unwrap();
        let failed = stream.message().await;
        server.abort();
        failed.expect_err("the first response is unavailable and the retries are spent");
        let execute_plan = calls.execute_plan.load(std::sync::atomic::Ordering::SeqCst);
        let reattach = calls.reattach.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            (execute_plan, reattach),
            (3, 1),
            "the first response retried outside the budget of the first RPC"
        );
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
        // `connect` reads the token of the environment too, so none is set until it returns:
        // another test may set one, which would make the channel TLS.
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let builder = ChannelBuilder::parse(&format!("sc://127.0.0.1:{}", address.port())).unwrap();
        let mut client = SparkConnectClient::connect(&builder).await.unwrap();
        drop(env);
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

    /// The listener of the spent retries of the tests, which records what the calls of each thread
    /// tell it and the errors that they catch, and fails the calls of the threads that ask it to,
    /// as the listener is of the process, which other tests share.
    #[derive(Default)]
    struct SpentRetries {
        told: std::sync::Mutex<Vec<(std::thread::ThreadId, RetriesExceeded)>>,
        caught: std::sync::Mutex<Vec<(std::thread::ThreadId, String)>>,
        failing: std::sync::Mutex<std::collections::HashSet<std::thread::ThreadId>>,
        /// What the logs of the retries and of the errors of grpc were told, by thread.
        logged: std::sync::Mutex<Vec<(std::thread::ThreadId, String)>>,
    }

    impl SpentRetries {
        fn log(&self, entry: String) {
            let thread = std::thread::current().id();
            self.logged.lock().unwrap().push((thread, entry));
        }

        /// What the logs of the calls of this thread were told, which is forgotten.
        fn take_logged(&self) -> Vec<String> {
            let thread = std::thread::current().id();
            let mut logged = self.logged.lock().unwrap();
            let ours = logged
                .iter()
                .filter(|(t, _)| *t == thread)
                .map(|(_, entry)| entry.clone())
                .collect();
            logged.retain(|(t, _)| *t != thread);
            ours
        }
    }

    impl crate::retries::RetriesListener for SpentRetries {
        fn retries_exceeded(&self, exceeded: RetriesExceeded) -> Result<()> {
            let thread = std::thread::current().id();
            self.told.lock().unwrap().push((thread, exceeded));
            if self.failing.lock().unwrap().contains(&thread) {
                return Err(SparkError::value_msg("the warning"));
            }
            Ok(())
        }

        fn caught(&self, error: SparkError) -> Option<SparkError> {
            let thread = std::thread::current().id();
            self.caught.lock().unwrap().push((thread, error.message()));
            None
        }

        fn retrying(&self, error: &str, wait_ms: i128, policy: &str) -> Result<()> {
            let error = error.lines().next().unwrap_or_default();
            self.log(format!("retrying {error} {wait_ms} {policy}"));
            Ok(())
        }

        fn given_up(&self, error: &str) -> Result<()> {
            let error = error.lines().next().unwrap_or_default();
            self.log(format!("given up {error}"));
            Ok(())
        }

        fn rpc_error(&self, error: &crate::error::GrpcioError) -> crate::retries::Delivery {
            let class = error.text.lines().next().unwrap_or_default();
            self.log(format!("rpc error {class} {:?}", error.code));
            None
        }
    }

    impl SpentRetries {
        /// The listener of the process, which the first call installs.
        fn installed() -> &'static SpentRetries {
            static LISTENER: std::sync::OnceLock<std::sync::Arc<SpentRetries>> =
                std::sync::OnceLock::new();
            LISTENER.get_or_init(|| {
                let listener = std::sync::Arc::new(SpentRetries::default());
                crate::retries::set_listener(Some(listener.clone()));
                listener
            })
        }

        /// What the calls of this thread told, and the messages of the errors that they caught,
        /// which are forgotten.
        fn take(&self) -> (Vec<RetriesExceeded>, Vec<String>) {
            let thread = std::thread::current().id();
            let mut told = self.told.lock().unwrap();
            let mut caught = self.caught.lock().unwrap();
            let ours = told.iter().filter(|(t, _)| *t == thread).map(|(_, e)| *e);
            let ours_told = ours.collect();
            let ours = caught
                .iter()
                .filter(|(t, _)| *t == thread)
                .map(|(_, m)| m.clone());
            let ours_caught = ours.collect();
            told.retain(|(t, _)| *t != thread);
            caught.retain(|(t, _)| *t != thread);
            (ours_told, ours_caught)
        }

        /// Make the listener fail the calls of this thread, or not.
        fn fail(&self, failing: bool) {
            let thread = std::thread::current().id();
            let mut threads = self.failing.lock().unwrap();
            if failing {
                threads.insert(thread);
            } else {
                threads.remove(&thread);
            }
        }
    }

    /// A client of an address that refuses its connections, of a retry policy of one quick retry,
    /// and the socket that reserves the port while it lives: bound to it, with `SO_REUSEADDR`,
    /// and never listening, so that a connection to it is refused at once, and no listener of
    /// another test takes the port meanwhile, as a port that is freed goes to the next `bind` of
    /// port 0.
    async fn refused_client() -> (SparkConnectClient, tokio::net::TcpSocket) {
        let reserved = tokio::net::TcpSocket::new_v4().unwrap();
        reserved.set_reuseaddr(true).unwrap();
        reserved.bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let port = reserved.local_addr().unwrap().port();
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let builder = ChannelBuilder::parse(&format!("sc://127.0.0.1:{port}")).unwrap();
        let mut client = SparkConnectClient::connect(&builder).await.unwrap();
        drop(env);
        client.set_retry_policy(RetryPolicy {
            max_retries: Some(1),
            initial_backoff_ms: 1,
            max_backoff_ms: Some(1),
            ..RetryPolicy::default()
        });
        (client, reserved)
    }

    #[tokio::test]
    async fn the_listener_is_told_of_the_spent_retries_of_a_call_where_pyspark_warns() {
        // As PySpark's `Retrying._wait`, which warns `[RETRIES_EXCEEDED]` once the retries of a
        // failure that a policy retries are spent, before it raises the failure, and not of a
        // failure that no policy retries.
        let listener = SpentRetries::installed();
        let (client, _reserved) = refused_client().await;
        let error = client.interrupt_all().await.unwrap_err();
        assert_eq!(error.grpc_code, Some(tonic::Code::Unavailable as i32));
        assert!(
            error
                .message()
                .starts_with("[The service is currently unavailable] "),
            "{error:?}"
        );
        assert!(
            error
                .pyspark_message
                .as_deref()
                .is_some_and(|message| message
                    .starts_with("<_InactiveRpcError of RPC that terminated with:\n")),
            "{error:?}"
        );
        assert_eq!(listener.take(), (vec![RetriesExceeded::Policy], vec![]));
        client.replace_retry_policy(RetryPolicy {
            retries_failures: false,
            ..client.retry_policy()
        });
        client.interrupt_all().await.unwrap_err();
        assert_eq!(listener.take(), (vec![], vec![]));

        // The error of the listener is that of the call, unconverted, as PySpark raises the
        // exception of the warning of a filter of "error".
        listener.fail(true);
        let (client, _reserved) = refused_client().await;
        assert_eq!(
            client.interrupt_all().await.unwrap_err().message(),
            "the warning"
        );
        assert_eq!(listener.take(), (vec![RetriesExceeded::Policy], vec![]));
        // The lookup of the plan compression of the first plan catches its error, as PySpark's
        // `except Exception` does, and the listener is given it back. The execution is the call
        // of the error then.
        let relation = ExecutePlanRequest {
            session_id: client.session_id().to_string(),
            plan: Some(Plan {
                op_type: Some(PlanOpType::Root(Default::default())),
            }),
            ..Default::default()
        };
        let executed = client.execute_plan_reattachable(relation).await;
        assert_eq!(executed.err().unwrap().message(), "the warning");
        let spent = vec![RetriesExceeded::Policy, RetriesExceeded::Policy];
        assert_eq!(listener.take(), (spent, vec!["the warning".to_string()]));
        // So does the read of the conf of the stack trace of an error of the server, which
        // PySpark's `_display_server_stack_trace` retries, and whose failure shows the trace.
        assert_eq!(client.conf_while_raising("k").await.unwrap(), None);
        assert!(client.display_server_stacktrace().await.unwrap());
        let spent = vec![RetriesExceeded::Policy; 2];
        let caught = vec!["the warning".to_string(); 2];
        assert_eq!(listener.take(), (spent, caught));
        listener.fail(false);
    }

    #[tokio::test]
    async fn the_listener_is_told_of_the_retries_and_the_errors_of_grpc_as_pyspark_logs_them() {
        // As PySpark's `Retrying._wait`, which logs each retry and the failure that ends the
        // retries, and its `_handle_rpc_error`, which logs each error of grpc that it converts.
        let listener = SpentRetries::installed();
        let (client, _reserved) = refused_client().await;
        client.replace_retry_policy(RetryPolicy {
            max_retries: Some(1),
            initial_backoff_ms: 1,
            jitter_ms: 0,
            ..client.retry_policy()
        });
        client.interrupt_all().await.unwrap_err();
        let error = "<_InactiveRpcError of RPC that terminated with:";
        assert_eq!(
            listener.take_logged(),
            [
                format!("retrying {error} 1 DefaultPolicy"),
                format!("given up {error}"),
                format!("rpc error {error} Unavailable"),
            ]
        );
        listener.take();
    }

    /// A decider of the retries of a test, which retries each failure `retries` times after 1 ms,
    /// then fails the call with `failure` if it has one, else gives the failure up; and counts
    /// the calls that it started and the failures it decided.
    struct Deciding {
        retries: usize,
        failure: Option<&'static str>,
        started: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        decided: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    struct DecidingCall {
        retries: usize,
        failure: Option<&'static str>,
        decided: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl crate::retries::RetryDecider for Deciding {
        fn start(&self, _call: GrpcCall) -> Box<dyn crate::retries::CallRetries> {
            self.started
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::new(DecidingCall {
                retries: self.retries,
                failure: self.failure,
                decided: self.decided.clone(),
            })
        }
    }

    impl crate::retries::CallRetries for DecidingCall {
        fn decide<'a>(
            &'a mut self,
            _status: &'a tonic::Status,
        ) -> futures::future::BoxFuture<'a, crate::retries::Decision> {
            use crate::retries::Decision;
            self.decided
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let decision = if self.retries > 0 {
                self.retries -= 1;
                Decision::Retry(Duration::from_millis(1))
            } else {
                match self.failure {
                    Some(failure) => Decision::Failed(Box::new(SparkError::value_msg(failure))),
                    None => Decision::GiveUp,
                }
            };
            Box::pin(async move { decision })
        }
    }

    #[tokio::test]
    async fn a_decider_decides_the_retries_of_the_calls_in_place_of_the_policy() {
        // As PySpark's `Retrying` of policies of methods of their own, which the Python binding
        // runs: of each call, the retries that start with it, of each failure, the decision. The
        // decider warns of spent retries itself, so the listener is told of none.
        let listener = SpentRetries::installed();
        let (client, _reserved) = refused_client().await;
        let started = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let decided = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let decider = |retries, failure| {
            std::sync::Arc::new(Deciding {
                retries,
                failure,
                started: started.clone(),
                decided: decided.clone(),
            }) as std::sync::Arc<dyn crate::retries::RetryDecider>
        };
        client.set_retry_decider(Some(decider(2, None)));
        let error = client.interrupt_all().await.unwrap_err();
        assert_eq!(error.grpc_code, Some(tonic::Code::Unavailable as i32));
        let count = |n: &std::sync::Arc<std::sync::atomic::AtomicUsize>| {
            n.swap(0, std::sync::atomic::Ordering::SeqCst)
        };
        assert_eq!((count(&started), count(&decided)), (1, 3));
        assert_eq!(listener.take(), (vec![], vec![]));
        // The error of a decision is the error of the call; a clone has the decider, a client of
        // a new session has none.
        client.set_retry_decider(Some(decider(0, Some("of the decider"))));
        let error = client.clone().interrupt_all().await.unwrap_err();
        assert_eq!(error.message(), "of the decider");
        assert_eq!((count(&started), count(&decided)), (1, 1));
        let mut fresh = client.with_new_session_id();
        fresh.set_retry_policy(RetryPolicy::no_retries());
        fresh.interrupt_all().await.unwrap_err();
        assert_eq!((count(&started), count(&decided)), (0, 0));
        listener.take();
        listener.take_logged();
    }

    #[tokio::test]
    async fn the_listener_is_told_of_the_retries_at_once_that_ran_for_their_elapsed_time() {
        // As PySpark's `Retrying._wait`, which warns `[RETRIES_EXCEEDED]` of the retries of a
        // RetryException, such as of an execution past its deadline, once they ran for their
        // elapsed time, and raises its cause.
        let listener = SpentRetries::installed();
        // A server that never answers.
        let silent = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = silent.local_addr().unwrap().port();
        let held = tokio::spawn(async move {
            let mut sockets = Vec::new();
            while let Ok((socket, _)) = silent.accept().await {
                sockets.push(socket);
            }
        });
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let builder = ChannelBuilder::parse(&format!("sc://127.0.0.1:{port}")).unwrap();
        let mut client = SparkConnectClient::connect(&builder).await.unwrap();
        drop(env);
        client.set_rpc_deadlines(RpcDeadlines {
            reattachable_execute_plan: Some(Duration::from_millis(50)),
            ..RpcDeadlines::default()
        });
        client.set_retry_policy(RetryPolicy {
            max_retry_exception_elapsed_time_ms: Some(0),
            ..RetryPolicy::default()
        });
        // No lookup of the plan compression, which the server would not answer either.
        *client.plan_compression.lock().unwrap() = PlanCompression::Disabled;
        let command = ExecutePlanRequest {
            session_id: client.session_id().to_string(),
            plan: Some(Plan {
                op_type: Some(PlanOpType::Command(Default::default())),
            }),
            ..Default::default()
        };
        let mut stream = client.execute_plan_reattachable(command).await.unwrap();
        let error = stream.message().await.unwrap_err();
        held.abort();
        assert_eq!(error.grpc_code, Some(tonic::Code::DeadlineExceeded as i32));
        assert_eq!(
            listener.take(),
            (vec![RetriesExceeded::RetryException], vec![])
        );
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
    async fn a_refresher_is_kept_for_the_client_and_its_clones_and_not_a_new_session() {
        // What a binding keeps to bring its extensions up to date, which the client gives back,
        // as its clones do, while the client of a new session has none.
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        assert!(client.user_context_refresher().is_none());
        client.set_user_context_refresher(Some(std::sync::Arc::new(7u8)));
        let kept = client.clone().user_context_refresher().unwrap();
        assert_eq!(kept.downcast_ref::<u8>(), Some(&7));
        assert!(client
            .with_new_session_id()
            .user_context_refresher()
            .is_none());
        client.set_user_context_refresher(None);
        assert!(client.user_context_refresher().is_none());
    }

    #[tokio::test]
    async fn a_list_given_to_threads_is_theirs_and_its_changes_are_theirs() {
        // As the list of PySpark's thread_local.user_context_extensions, which a binding keeps and
        // changes: a new list that a thread is given is its extensions, whose changes are the
        // extensions of the threads that have it, and a cleared thread has none.
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = std::sync::Arc::new(SparkConnectClient::connect(&builder).await.unwrap());
        client.add_threadlocal_user_context_extension(any_of("own"));
        let list =
            SharedUserContextExtensions::new(vec![("threadlocal_a".to_string(), any_of("a"))]);
        client.share_threadlocal_user_context_extensions(&list);
        assert_eq!(extension_ids(&client), vec!["threadlocal_a"]);
        list.replace(vec![("threadlocal_b".to_string(), any_of("b"))]);
        assert_eq!(extension_ids(&client), vec!["threadlocal_b"]);
        assert_eq!(list.snapshot()[0].0, "threadlocal_b");

        let seen = {
            let client = client.clone();
            let list = list.clone();
            std::thread::spawn(move || {
                let before = extension_ids(&client);
                client.share_threadlocal_user_context_extensions(&list);
                list.replace(vec![("threadlocal_c".to_string(), any_of("c"))]);
                (before, extension_ids(&client))
            })
            .join()
            .unwrap()
        };
        assert_eq!(
            seen,
            (Vec::<String>::new(), vec!["threadlocal_c".to_string()])
        );
        assert_eq!(extension_ids(&client), vec!["threadlocal_c"]);

        client.clear_threadlocal_user_context_extensions();
        assert!(extension_ids(&client).is_empty());
        assert_eq!(list.snapshot()[0].0, "threadlocal_c");
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
    fn a_client_session_id_is_sent_and_checked_while_the_session_id_stays() {
        // As PySpark's `client._session_id`, which its requests read: a set changes the ID that
        // the clones of the client send and check the responses against, while `session_id`
        // stays the session's, as the `session_id` of PySpark's session; a client of a new
        // session has its own, and its builder no session ID, as PySpark's `newSession` pops it.
        let client =
            connect_blocking("sc://localhost/;session_id=7c4e2a0e-0f6b-4b9a-9d1e-0123456789ab");
        let original = client.session_id().to_string();
        assert_eq!(client.client_session_id(), original);
        let clone = client.clone();
        clone.set_client_session_id("changed".to_string());
        assert_eq!(client.client_session_id(), "changed");
        assert_eq!(client.session_id(), original);
        assert!(client.verify_response("changed", "").is_ok());
        let err = client.verify_response(&original, "").unwrap_err();
        assert!(err.message().contains(&format!("{original} != changed")));
        assert_eq!(
            client
                .channel_builder()
                .get(crate::channel::PARAM_SESSION_ID),
            Some(original.as_str())
        );
        let new = client.with_new_session_id();
        assert_eq!(new.client_session_id(), new.session_id());
        assert_ne!(new.session_id(), original);
        assert_eq!(
            new.channel_builder().get(crate::channel::PARAM_SESSION_ID),
            None
        );
        assert_eq!(
            client.channel_builder().port(),
            new.channel_builder().port()
        );
    }

    #[test]
    fn a_closed_client_reopens_and_the_status_rpc_has_a_deadline() {
        // As PySpark's `client._closed`, which a set to True closes and a set to False opens, so
        // the requests are sent again, and its `RpcDeadlines.get_status`, of 10 minutes, and none
        // when the deadlines are disabled. The artifacts are sent while it is closed, as PySpark's
        // ArtifactManager sends them of a stub of its own.
        let client = connect_blocking("sc://localhost");
        client.set_closed(true);
        assert!(client.is_closed() && client.ensure_open().is_err());
        assert!(client.ensure_channel_open().is_ok());
        client.clone().reopen();
        assert!(!client.is_closed() && client.ensure_open().is_ok());
        assert!(client.ensure_open_for_stream().is_ok());
        assert_eq!(
            RpcDeadlines::default().get_status,
            Some(Duration::from_secs(600))
        );
        assert_eq!(RpcDeadlines::disabled().get_status, None);
    }

    #[test]
    fn a_client_that_close_closed_fails_with_the_error_of_a_closed_channel_when_open() {
        // As PySpark's `close`, which closes its gRPC channel: its RPCs raise NO_ACTIVE_SESSION
        // while the client is closed, and grpc's ValueError of a closed channel once it is open
        // again, of the message of a server-streaming call and of that of the others, which the
        // artifacts raise whether it is closed or not. A client of a new session is open.
        let client = connect_blocking("sc://localhost");
        client.close();
        let closed = client.ensure_open().unwrap_err();
        assert_eq!(closed.error_class, "NO_ACTIVE_SESSION");
        assert_eq!(
            client.ensure_open_for_stream().unwrap_err().error_class,
            "NO_ACTIVE_SESSION"
        );
        let unary = "Cannot invoke RPC on closed channel!";
        let artifacts = client.ensure_channel_open().unwrap_err();
        assert_eq!(
            artifacts.kind,
            crate::error::SparkErrorKind::BuiltinValueError
        );
        assert_eq!(artifacts.message(), unary);
        client.reopen();
        assert_eq!(client.ensure_open().unwrap_err().message(), unary);
        let streaming = client.ensure_open_for_stream().unwrap_err();
        assert_eq!(
            streaming.kind,
            crate::error::SparkErrorKind::BuiltinValueError
        );
        assert_eq!(streaming.message(), "Cannot invoke RPC: Channel closed!");
        let new = client.with_new_session_id();
        assert!(new.ensure_open().is_ok() && new.ensure_channel_open().is_ok());
    }

    #[test]
    fn a_session_change_marks_the_client_closed_of_an_open_channel() {
        // As PySpark's `_closed = True` of a response of another server-side session, and of
        // INVALID_HANDLE.SESSION_CHANGED: the channel stays open, so the artifacts are sent, and
        // the client sends its requests again once it is open.
        let client = connect_blocking("sc://localhost");
        assert!(client.verify_response("", "server-a").is_ok());
        assert!(client.verify_response("", "server-b").is_err());
        assert!(client.is_closed() && client.ensure_open().is_err());
        assert!(client.ensure_channel_open().is_ok());
        client.reopen();
        assert!(client.ensure_open().is_ok() && client.ensure_open_for_stream().is_ok());
        #[derive(Clone, PartialEq, prost::Message)]
        struct ErrorInfo {
            #[prost(string, tag = "1")]
            reason: String,
            #[prost(map = "string, string", tag = "3")]
            metadata: std::collections::HashMap<String, String>,
        }
        #[derive(Clone, PartialEq, prost::Message)]
        struct RpcStatus {
            #[prost(message, repeated, tag = "3")]
            details: Vec<prost_types::Any>,
        }
        let info = ErrorInfo {
            reason: "org.apache.spark.SparkSQLException".into(),
            metadata: [(
                "errorClass".to_string(),
                "INVALID_HANDLE.SESSION_CHANGED".to_string(),
            )]
            .into(),
        };
        let details = RpcStatus {
            details: vec![prost_types::Any {
                type_url: "type.googleapis.com/google.rpc.ErrorInfo".into(),
                value: info.encode_to_vec(),
            }],
        };
        let status = tonic::Status::with_details(
            tonic::Code::Internal,
            "[INVALID_HANDLE.SESSION_CHANGED] changed",
            details.encode_to_vec().into(),
        );
        let _ = client.grpc_error(status, GrpcCall::Unary);
        assert!(client.is_closed() && client.ensure_channel_open().is_ok());
        client.reopen();
        assert!(client.ensure_open().is_ok());
    }

    #[test]
    fn the_call_settings_of_a_thread_are_its_own_and_its_executions_keep_them() {
        // As the `_rpc_deadlines`, `_allow_arrow_batch_chunking` and `_preferred_arrow_chunk_size`
        // of the client that a thread of the binding calls: its calls read those that it was
        // given, a request context given afterwards keeps them, a clone made for a call keeps them
        // for the requests that it sends from other threads, a thread of no request context of
        // the binding, as one of the transport that sends the requests of a call, takes those that
        // a thread was given last, and None gives the thread the client's again.
        let client = connect_blocking("sc://localhost/;user_id=alice");
        let own = client.rpc_deadlines().clone();
        let deadlines = RpcDeadlines {
            analyze_plan: Some(Duration::from_secs(7)),
            ..RpcDeadlines::disabled()
        };
        let chunking = ResultChunking {
            allow_arrow_batch_chunking: false,
            preferred_arrow_chunk_size: Some(4096),
        };
        // Of a thread of no request context: one of the client's session ID and user.
        client.set_thread_call_settings(Some(deadlines.clone()), Some(chunking));
        assert_eq!(client.call_rpc_deadlines(), deadlines);
        assert_eq!(client.call_result_chunking(), chunking);
        assert_eq!(client.client_session_id(), client.session_id());
        assert_eq!(client.request_user_id().as_deref(), Some("alice"));
        client.set_thread_request_context("thread-session".to_string(), None, vec![]);
        assert_eq!(client.call_rpc_deadlines(), deadlines);
        assert_eq!(client.client_session_id(), "thread-session");
        let of_call = client.of_call_settings();
        let other = client.clone();
        let seen = std::thread::spawn(move || {
            (
                other.call_rpc_deadlines(),
                other.call_result_chunking(),
                of_call.call_rpc_deadlines(),
                of_call.call_result_chunking(),
            )
        })
        .join()
        .unwrap();
        assert_eq!(
            seen,
            (deadlines.clone(), chunking, deadlines.clone(), chunking)
        );
        // A call that ran during the call of the thread gives it its settings back.
        let kept = client.thread_request_state();
        client.set_thread_call_settings(None, None);
        assert_eq!(client.call_rpc_deadlines(), own);
        assert_eq!(client.call_result_chunking(), ResultChunking::default());
        let other = client.clone();
        let seen =
            std::thread::spawn(move || (other.call_rpc_deadlines(), other.call_result_chunking()))
                .join()
                .unwrap();
        assert_eq!(seen, (own.clone(), ResultChunking::default()));
        client.restore_thread_request_state(kept);
        assert_eq!(client.call_result_chunking(), chunking);
        assert_eq!(client.call_rpc_deadlines(), deadlines);
    }

    #[test]
    fn the_plan_compression_settings_are_those_of_pyspark_s_client() {
        // As PySpark's `_plan_compression_threshold` and `_plan_compression_algorithm`: none
        // before the lookup; those of a failed lookup and of `CANNOT_PARSE`, -1 and `NONE`; and
        // those of a binding, of which the plans of more bytes than a threshold that is not
        // negative are compressed of the algorithm `ZSTD`, none of another, and of no algorithm
        // the settings are looked up again.
        let client = connect_blocking("sc://localhost/");
        assert_eq!(client.plan_compression_settings(), None);
        assert_eq!(client.known_plan_compression(), None);
        client.keep_plan_compression(PlanCompression::Disabled);
        assert_eq!(
            client.plan_compression_settings(),
            Some(PlanCompressionSettings {
                threshold: -1,
                algorithm: Some("NONE".to_string()),
            })
        );
        let settings = |threshold, algorithm: Option<&str>| PlanCompressionSettings {
            threshold,
            algorithm: algorithm.map(str::to_string),
        };
        for (threshold, algorithm, known) in [
            (100, Some("ZSTD"), Some(Some(100))),
            (0, Some("ZSTD"), Some(Some(0))),
            (-1, Some("ZSTD"), Some(None)),
            (100, Some("LZ4"), Some(None)),
            (100, Some("NONE"), Some(None)),
            (100, None, None),
        ] {
            client.set_plan_compression_settings(Some(settings(threshold, algorithm)));
            assert_eq!(
                client.known_plan_compression(),
                known,
                "{threshold} {algorithm:?}"
            );
            assert_eq!(
                client.clone().plan_compression_settings(),
                Some(settings(threshold, algorithm))
            );
        }
        client.set_plan_compression_settings(None);
        assert_eq!(client.known_plan_compression(), None);
        assert_eq!(client.plan_compression_settings(), None);
        client.keep_plan_compression(PlanCompression::Zstd { threshold: 9 });
        assert_eq!(
            client.plan_compression_settings(),
            Some(settings(9, Some("ZSTD")))
        );
    }

    #[test]
    fn the_request_context_of_a_thread_is_its_own() {
        // As the attributes of the client that a thread of the binding calls: the session ID, the
        // user and the global extensions that a thread was given are those of its requests, and
        // the other threads keep those of the client, until the thread ends.
        let client = connect_blocking("sc://localhost/;user_id=alice");
        let shared = client.client_session_id();
        let global = prost_types::Any {
            type_url: "type.googleapis.com/g".to_string(),
            value: vec![],
        };
        client.set_thread_request_context(
            "thread-session".to_string(),
            Some("bob".to_string()),
            vec![(String::new(), global.clone())],
        );
        assert_eq!(client.client_session_id(), "thread-session");
        assert_eq!(client.request_user_id().as_deref(), Some("bob"));
        assert_eq!(client.global_user_context_extensions().len(), 1);
        let mut context = None;
        client.extend_user_context(&mut context);
        assert_eq!(context.unwrap().extensions, vec![global]);
        let other = client.clone();
        let seen = std::thread::spawn(move || {
            (
                other.client_session_id(),
                other.request_user_id(),
                other.global_user_context_extensions().len(),
            )
        })
        .join()
        .unwrap();
        assert_eq!(seen, (shared, Some("alice".to_string()), 0));
        assert_eq!(
            client
                .with_new_session_id_of_user(Some("carol".to_string()))
                .user_id(),
            Some("carol")
        );
        let other = client.clone();
        std::thread::spawn(move || {
            other.set_thread_request_context("gone".to_string(), None, vec![]);
        })
        .join()
        .unwrap();
        let contexts = client
            .user_context_extensions
            .lock()
            .unwrap()
            .request_by_thread
            .len();
        assert_eq!(contexts, 1);
    }

    #[test]
    fn the_request_context_that_a_future_of_a_call_sets_is_the_callers() {
        // A future of a call that runs on another thread, inside a `current_thread` runtime, is of
        // the thread that calls: the request context and the user context extensions that it
        // gives the thread are those that the requests of that thread read, on it as in the
        // future, and they are forgotten when that thread ends.
        let client = connect_blocking("sc://localhost/;user_id=alice");
        let other = client.clone();
        std::thread::spawn(move || {
            let own = prost_types::Any {
                type_url: "type.googleapis.com/own".to_string(),
                value: vec![],
            };
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let in_the_future = runtime.block_on(async {
                crate::runtime::block_on(async {
                    other.set_thread_request_context(
                        "caller-session".to_string(),
                        Some("bob".to_string()),
                        vec![],
                    );
                    other.share_threadlocal_user_context_extensions(
                        &SharedUserContextExtensions::new(vec![(
                            "threadlocal_own".to_string(),
                            own.clone(),
                        )]),
                    );
                    (
                        std::thread::current().id(),
                        other.client_session_id(),
                        other.threadlocal_user_context_extensions().len(),
                    )
                })
            });
            assert_ne!(in_the_future.0, std::thread::current().id());
            assert_eq!(in_the_future.1, "caller-session");
            assert_eq!(in_the_future.2, 1);
            assert_eq!(other.client_session_id(), "caller-session");
            assert_eq!(other.request_user_id().as_deref(), Some("bob"));
            let mut context = None;
            other.extend_user_context(&mut context);
            assert_eq!(context.unwrap().extensions, vec![own]);
        })
        .join()
        .unwrap();
        let extensions = client.user_context_extensions.lock().unwrap();
        assert!(extensions.request_by_thread.is_empty());
        assert!(extensions.by_thread.is_empty());
    }

    #[test]
    fn a_kept_request_state_of_a_thread_is_given_back() {
        // As a binding keeps the context of the requests of a call around a call that runs during
        // it, such as one of a progress handler.
        let client = connect_blocking("sc://localhost/;user_id=alice");
        let any = |name: &str| prost_types::Any {
            type_url: format!("type.googleapis.com/{name}"),
            value: vec![],
        };
        client.set_thread_request_context(
            "outer".to_string(),
            None,
            vec![("global_outer".to_string(), any("outer"))],
        );
        let own = SharedUserContextExtensions::new(vec![("threadlocal_o".to_string(), any("o"))]);
        client.share_threadlocal_user_context_extensions(&own);
        let kept = client.thread_request_state();
        client.set_thread_request_context("inner".to_string(), Some("bob".to_string()), vec![]);
        client.clear_threadlocal_user_context_extensions();
        assert_eq!(client.client_session_id(), "inner");
        client.restore_thread_request_state(kept);
        assert_eq!(client.client_session_id(), "outer");
        assert_eq!(client.request_user_id(), None);
        let mut context = None;
        client.extend_user_context(&mut context);
        assert_eq!(context.unwrap().extensions, vec![any("outer"), any("o")]);
        // The list of the thread is the one that it shared, of the changes of the others.
        own.replace(vec![]);
        assert!(client.threadlocal_user_context_extensions().is_empty());
        // A thread that had none has none again.
        let kept = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let kept = client.thread_request_state();
                    client.set_thread_request_context("other".to_string(), None, vec![]);
                    client.restore_thread_request_state(kept);
                    client.client_session_id()
                })
                .join()
                .unwrap()
        });
        assert_eq!(kept, client.session_id());
    }

    #[test]
    fn the_user_of_the_requests_is_set_apart_from_that_of_the_artifacts() {
        // As PySpark's `client._user_id`, an attribute that its requests read, while its
        // ArtifactManager keeps the user that the client was made with; the clients of new
        // sessions take the user of the requests, unless the connection has one.
        let client = connect_blocking("sc://localhost");
        let original = client.user_id().map(String::from);
        assert_eq!(client.request_user_id(), original);
        client.clone().set_request_user_id(Some("bob".to_string()));
        assert_eq!(client.request_user_id().as_deref(), Some("bob"));
        assert_eq!(client.user_id().map(String::from), original);
        assert_eq!(
            client.new_user_context().user_id,
            original.clone().unwrap_or_default()
        );
        let mut context = None;
        client.fill_user_context(&mut context);
        assert_eq!(context.unwrap().user_id, "bob");
        let new = client.with_new_session_id();
        assert_eq!(new.user_id(), Some("bob"));
        assert_eq!(new.request_user_id().as_deref(), Some("bob"));
        let named = connect_blocking("sc://localhost/;user_id=alice");
        named.set_request_user_id(Some("bob".to_string()));
        assert_eq!(named.with_new_session_id().user_id(), Some("alice"));
        named.set_request_user_id(None);
        let mut context = None;
        named.fill_user_context(&mut context);
        assert_eq!(context.unwrap().user_id, "");
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
            looked_up_plan_compression(Some("10485760"), Some("ZSTD")).0,
            PlanCompression::Zstd {
                threshold: 10485760
            }
        );
        assert_eq!(
            looked_up_plan_compression(Some("0"), Some("ZSTD")).0,
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
                looked_up_plan_compression(threshold, algorithm).0,
                PlanCompression::Disabled,
                "threshold {threshold:?}, algorithm {algorithm:?}"
            );
        }
    }

    #[test]
    fn the_lookup_of_the_plan_compression_logs_what_pyspark_logs_of_it() {
        // As PySpark's `int(threshold)`, of its text: the threshold that it read, or the
        // ValueError of a threshold that is not an integer, which disables the compression.
        let texts = |threshold, algorithm| match looked_up_plan_compression(threshold, algorithm) {
            (_, crate::client_log::Message::Text(text)) => text,
            (compression, crate::client_log::Message::WithError { text, error }) => {
                assert_eq!(compression, PlanCompression::Disabled);
                assert_eq!(error.kind, SparkErrorKind::BuiltinValueError);
                format!("{text} % {}", error.message())
            }
            _ => unreachable!(),
        };
        assert_eq!(
            texts(Some(" +1_000 "), Some("ZSTD")),
            "Plan compression threshold: 1000, algorithm: ZSTD"
        );
        assert_eq!(
            texts(None, None),
            "Plan compression threshold: -1, algorithm: None"
        );
        assert_eq!(
            texts(Some("-5"), Some("LZ4")),
            "Plan compression threshold: -5, algorithm: LZ4"
        );
        assert_eq!(
            texts(Some("10m"), Some("ZSTD")),
            "Plan compression is disabled because the server does not support it. % invalid \
             literal for int() with base 10: '10m'"
        );
        assert_eq!(
            looked_up_plan_compression(Some("1_000"), Some("ZSTD")).0,
            PlanCompression::Zstd { threshold: 1000 }
        );
        assert_eq!(python_str_repr("a'b"), "\"a'b\"");
        assert_eq!(
            python_str_repr("a'\"b\\\n\u{1}\u{85}"),
            "'a\\'\"b\\\\\\n\\x01\\x85'"
        );
    }

    #[test]
    fn a_plan_is_compressed_as_python_zstandard_compresses_it_with_its_logs() {
        // The frame of python-zstandard 0.25's `ZstdCompressor().compress` (zstd 1.5.7), of the
        // size of its content, of the encoding of this relation, and what PySpark logs of it.
        let mut data: Vec<u8> = (0..=255u8).cycle().take(256 * 8).collect();
        data.extend("spark connect plan ".repeat(200).bytes());
        let compressed = zstd::bulk::compress(&data, zstd::DEFAULT_COMPRESSION_LEVEL).unwrap();
        let expected = "28b52ffd60d8150d09003411000102030405060708090a0b0c0d0e0f101112131415161718\
                        191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d\
                        3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162\
                        636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f8081828384858687\
                        88898a8b8c8d8e8f909192939495969798999a9b9c9d9e9fa0a1a2a3a4a5a6a7a8a9aaabac\
                        adaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecfd0d1\
                        d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeeff0f1f2f3f4f5f6\
                        f7f8f9fafbfcfdfeff737061726b20636f6e6e65637420706c616e200200856de38700fd0e\
                        fc740a";
        let hex: String = compressed.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, expected);

        let relation = relation_with(vec![7; PAYLOAD]);
        let encoded = relation.encode_to_vec();
        let mut plan = root(&relation);
        let logged = compress_over_threshold(&mut plan, encoded.len() - 1);
        let Some(PlanOpType::CompressedOperation(operation)) = &plan.op_type else {
            panic!("the plan over the threshold was not compressed");
        };
        let size = operation.data.len();
        let ratio = 1.0 - size as f64 / encoded.len() as f64;
        assert_eq!(logged.len(), 1);
        assert!(
            logged[0].starts_with(&format!(
                "Plan compression: original_size={}, compressed_size={size}, \
                 saving_ratio={ratio:.2}, duration_s=",
                encoded.len()
            )),
            "{logged:?}"
        );
        let random = relation_with(incompressible(PAYLOAD));
        let mut not_smaller = root(&random);
        let logged = compress_over_threshold(&mut not_smaller, 0);
        assert_eq!(logged.len(), 2, "{logged:?}");
        assert_eq!(
            logged[1],
            "Plan compression not effective. Using original plan."
        );
        // Of no log, a plan that is not over the threshold.
        assert!(compress_over_threshold(&mut root(&relation), encoded.len()).is_empty());
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

    /// The values of the `user-agent` header of each request that the server received, in order.
    type UserAgents = std::sync::Arc<std::sync::Mutex<Vec<Vec<Vec<u8>>>>>;

    #[derive(Clone)]
    struct Recorder(Received, UserAgents);

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
            let user_agents = request.headers().get_all(http::header::USER_AGENT);
            let user_agents = user_agents.iter().map(|v| v.as_bytes().to_vec()).collect();
            self.1.lock().unwrap().push(user_agents);
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

    /// A server whose executions send responses of 256 KiB of Arrow data without end, as a large
    /// result that the caller reads on slowly, and which answers the other RPCs at once.
    #[derive(Clone)]
    struct Flooding(Received);

    impl NamedService for Flooding {
        const NAME: &'static str = "spark.connect.SparkConnectService";
    }

    impl tonic::codegen::Service<http::Request<tonic::body::Body>> for Flooding {
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
                    "ExecutePlan" => grpc.server_streaming(Flood, request).await,
                    _ => grpc.unary(rpc, request).await,
                })
            })
        }
    }

    /// The execution of [`Flooding`], of responses without end.
    struct Flood;

    impl ServerStreamingService<Vec<u8>> for Flood {
        type Response = Vec<u8>;
        type ResponseStream =
            futures::stream::BoxStream<'static, std::result::Result<Vec<u8>, tonic::Status>>;
        type Future = BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;

        fn call(&mut self, _: tonic::Request<Vec<u8>>) -> Self::Future {
            use futures::StreamExt;
            let response = ExecutePlanResponse {
                server_side_session_id: SERVER_ID.to_string(),
                response_type: Some(ResponseType::ArrowBatch(
                    spark_connect_proto::execute_plan_response::ArrowBatch {
                        row_count: 1,
                        data: vec![0u8; 256 * 1024].into(),
                        ..Default::default()
                    },
                )),
                ..Default::default()
            }
            .encode_to_vec();
            let frames = futures::stream::repeat_with(move || Ok(response.clone())).boxed();
            Box::pin(async move { Ok(tonic::Response::new(frames)) })
        }
    }

    /// A server whose responses are of metadata of the size of its mode: the status of a unary
    /// call, of its trailers only, of details of `details` bytes; the headers of a response of
    /// `header` bytes; the trailers of a stream, after a response, of `trailer` bytes.
    #[derive(Clone, Copy)]
    struct LargeMetadata {
        details: usize,
        header: usize,
        trailer: usize,
    }

    impl NamedService for LargeMetadata {
        const NAME: &'static str = "spark.connect.SparkConnectService";
    }

    impl UnaryService<Vec<u8>> for LargeMetadata {
        type Response = Vec<u8>;
        type Future = BoxFuture<tonic::Response<Vec<u8>>, tonic::Status>;

        fn call(&mut self, _: tonic::Request<Vec<u8>>) -> Self::Future {
            let this = *self;
            Box::pin(async move {
                if this.header > 0 {
                    let mut response = tonic::Response::new(Vec::new());
                    response
                        .metadata_mut()
                        .insert("x-big", "a".repeat(this.header).parse().unwrap());
                    return Ok(response);
                }
                Err(tonic::Status::with_details(
                    tonic::Code::Internal,
                    "failed",
                    vec![b'z'; this.details].into(),
                ))
            })
        }
    }

    impl ServerStreamingService<Vec<u8>> for LargeMetadata {
        type Response = Vec<u8>;
        type ResponseStream =
            futures::stream::BoxStream<'static, std::result::Result<Vec<u8>, tonic::Status>>;
        type Future = BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;

        fn call(&mut self, _: tonic::Request<Vec<u8>>) -> Self::Future {
            use futures::StreamExt;
            let trailer = self.trailer;
            Box::pin(async move {
                let mut status = tonic::Status::internal("failed");
                status
                    .metadata_mut()
                    .insert("x-big", "a".repeat(trailer).parse().unwrap());
                let frames = futures::stream::iter(vec![
                    Ok(ExecutePlanResponse::default().encode_to_vec()),
                    Err(status),
                ])
                .boxed();
                Ok(tonic::Response::new(frames))
            })
        }
    }

    impl tonic::codegen::Service<http::Request<tonic::body::Body>> for LargeMetadata {
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
            let this = *self;
            let streaming = request.uri().path().ends_with("/ExecutePlan");
            Box::pin(async move {
                let mut grpc =
                    tonic::server::Grpc::new(BytesCodec).max_encoding_message_size(usize::MAX);
                Ok(if streaming {
                    grpc.server_streaming(this, request).await
                } else {
                    grpc.unary(this, request).await
                })
            })
        }
    }

    /// A client of a server of [`LargeMetadata`] of `mode`.
    async fn client_of_large_metadata(mode: LargeMetadata) -> SparkConnectClient {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .http2_max_header_list_size(64 << 20)
                .add_service(mode)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let builder =
            ChannelBuilder::parse(&format!("sc://127.0.0.1:{}/", address.port())).unwrap();
        let mut client = SparkConnectClient::connect(&builder).await.unwrap();
        drop(env);
        client.set_retry_policy(RetryPolicy::no_retries());
        client
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_response_of_metadata_past_the_hard_limit_of_grpc_fails_its_call_at_once() {
        // As gRPC's core: RESOURCE_EXHAUSTED of its text, at once also of metadata far larger,
        // of the status of a unary call, of the headers of a response, and of the trailers of a
        // stream, where hyper ended the stream or the connection.
        async fn at_once<F: std::future::Future>(what: F) -> F::Output {
            tokio::time::timeout(Duration::from_secs(10), what)
                .await
                .expect("the call ends at once")
        }
        for details in [40_000, 400_000] {
            let client = client_of_large_metadata(LargeMetadata {
                details,
                header: 0,
                trailer: 0,
            })
            .await;
            let status = at_once(
                client
                    .stub()
                    .config(spark_connect_proto::ConfigRequest::default()),
            )
            .await
            .unwrap_err();
            assert_eq!(status.code(), tonic::Code::ResourceExhausted, "{status:?}");
            assert!(crate::metadata::is_refused(&status));
            assert!(
                status
                    .message()
                    .starts_with("received metadata size exceeds hard limit (value length "),
                "{status:?}"
            );
        }
        let client = client_of_large_metadata(LargeMetadata {
            details: 0,
            header: 20_000,
            trailer: 0,
        })
        .await;
        let status = at_once(
            client
                .stub()
                .config(spark_connect_proto::ConfigRequest::default()),
        )
        .await
        .unwrap_err();
        assert_eq!(
            status.message(),
            "received metadata size exceeds hard limit (value length 20000 vs. 16384)"
        );
        let client = client_of_large_metadata(LargeMetadata {
            details: 0,
            header: 0,
            trailer: 20_000,
        })
        .await;
        let mut stream = at_once(client.stub().execute_plan(ExecutePlanRequest::default()))
            .await
            .unwrap()
            .into_inner();
        assert!(at_once(stream.message()).await.unwrap().is_some());
        let status = at_once(stream.message()).await.unwrap_err();
        assert_eq!(
            status.message(),
            "received metadata size exceeds hard limit (value length 20000 vs. 16384)"
        );
    }

    /// A server of HTTP/2 that resets the stream of each call with the error code `reason`,
    /// before the headers of its response, or after them if `after_headers`, as Spark's server of
    /// grpc-java resets the stream of a call whose metadata the client's settings do not take.
    async fn resetting_server(reason: u32, after_headers: bool) -> std::net::SocketAddr {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let frame = |kind: u8, flags: u8, stream: u32, payload: &[u8]| {
            let length = (payload.len() as u32).to_be_bytes();
            [
                &length[1..],
                &[kind, flags][..],
                &stream.to_be_bytes()[..],
                payload,
            ]
            .concat()
        };
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    socket.write_all(&frame(4, 0, 0, &[])).await?;
                    let mut preface = [0u8; 24];
                    socket.read_exact(&mut preface).await?;
                    loop {
                        let mut header = [0u8; 9];
                        socket.read_exact(&mut header).await?;
                        let length = u32::from_be_bytes([0, header[0], header[1], header[2]]);
                        let mut payload = vec![0u8; length as usize];
                        socket.read_exact(&mut payload).await?;
                        let (kind, flags) = (header[3], header[4]);
                        let stream =
                            u32::from_be_bytes([header[5] & 0x7f, header[6], header[7], header[8]]);
                        match kind {
                            4 if flags & 1 == 0 => socket.write_all(&frame(4, 1, 0, &[])).await?,
                            1 => {
                                if after_headers {
                                    // `:status: 200`, indexed, and a literal `content-type`.
                                    let block = [
                                        &[0x88, 0x00, 12][..],
                                        b"content-type",
                                        &[16][..],
                                        b"application/grpc",
                                    ]
                                    .concat();
                                    socket.write_all(&frame(1, 4, stream, &block)).await?;
                                }
                                let reset = frame(3, 0, stream, &reason.to_be_bytes());
                                socket.write_all(&reset).await?;
                            }
                            _ => {}
                        }
                    }
                    #[allow(unreachable_code)]
                    std::io::Result::Ok(())
                });
            }
        });
        address
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stream_that_the_server_resets_fails_as_grpcio_s_of_its_code_and_text() {
        // As gRPC's core: of the code of the reason, and of its text, `Stream removed (RST_STREAM
        // (Received RST_STREAM with error code <reason>))`, before the headers of the response or
        // after them, of a unary call and of a stream.
        for (reason, code) in [
            (1, tonic::Code::Internal),
            (2, tonic::Code::Internal),
            (8, tonic::Code::Cancelled),
            (11, tonic::Code::ResourceExhausted),
            (12, tonic::Code::PermissionDenied),
        ] {
            for after_headers in [false, true] {
                let address = resetting_server(reason, after_headers).await;
                let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
                let builder =
                    ChannelBuilder::parse(&format!("sc://127.0.0.1:{}/", address.port())).unwrap();
                let client = SparkConnectClient::connect(&builder).await.unwrap();
                drop(env);
                let text = format!(
                    "Stream removed (RST_STREAM (Received RST_STREAM with error code {reason}))"
                );
                let unary = client
                    .stub()
                    .config(spark_connect_proto::ConfigRequest::default())
                    .await
                    .unwrap_err();
                let streaming = match client
                    .stub()
                    .execute_plan(ExecutePlanRequest::default())
                    .await
                {
                    Ok(stream) => stream.into_inner().message().await.unwrap_err(),
                    Err(status) => status,
                };
                for status in [unary, streaming] {
                    let case = format!("{reason} {after_headers}: {status:?}");
                    assert_eq!(crate::retries::grpc_python_code(&status), code, "{case}");
                    assert_eq!(
                        crate::connector::stream_removed_in(&status, true).as_deref(),
                        Some(text.as_str()),
                        "{case}"
                    );
                    // Of tonic's text in the Rust client's dialect.
                    assert_eq!(crate::connector::stream_removed_in(&status, false), None);
                }
            }
        }
    }

    /// As gRPC's channels, whose connection windows grow with the windows of their streams: the
    /// results of the executions of a session that its caller does not read on, as of open
    /// `toLocalIterator()`s, do not starve the calls of another session of the connection, nor
    /// another execution of the session.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_unread_results_of_a_session_do_not_starve_the_other_calls_of_its_connection() {
        let received: Received = Default::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let service = Flooding(received);
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let builder =
            ChannelBuilder::parse(&format!("sc://127.0.0.1:{}/", address.port())).unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        drop(env);
        let other = client.with_new_session_id();
        let request = || ExecutePlanRequest {
            session_id: client.session_id().to_string(),
            plan: Some(Plan {
                op_type: Some(PlanOpType::Root(Default::default())),
            }),
            ..Default::default()
        };
        // More unread results than hyper's window of a connection, 5 MiB, holds, of 2 MiB each.
        let mut held = Vec::new();
        for _ in 0..6 {
            let stream = tokio::time::timeout(Duration::from_secs(10), async {
                let mut stream = client.execute_plan(request()).await.unwrap();
                stream.message().await.unwrap();
                stream
            })
            .await
            .expect("another execution of the session is not starved");
            held.push(stream);
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
        tokio::time::timeout(Duration::from_secs(10), other.get_configs(&["k"]))
            .await
            .expect("the call of the other session is not starved")
            .unwrap();
        drop(held);
    }

    #[test]
    fn a_message_over_the_size_limit_of_a_call_is_grpcio_s_resource_exhausted() {
        // tonic's status of a response that it does not read, and the client's of a request that
        // it does not send, are grpc-python's RESOURCE_EXHAUSTED, of gRPC core's texts in the
        // dialect of grpcio, of their length and limit.
        let received = tonic::Status::out_of_range(
            "Error, decoded message length too large: found 135266304 bytes, the limit is: \
             134217728 bytes",
        );
        assert_eq!(
            oversized(&received),
            Some((Oversized::Received, 135266304, 134217728))
        );
        assert_eq!(
            crate::retries::grpc_python_code(&received),
            tonic::Code::ResourceExhausted
        );
        assert_eq!(
            status_details_in(&received, true),
            "Stream removed (CLIENT: Received message larger than max (135266304 vs. 134217728))"
        );
        assert_eq!(status_details_in(&received, false), received.message());
        let sent = too_large_to_send(135266440);
        assert_eq!(
            oversized(&sent),
            Some((
                Oversized::Sent,
                135266440,
                GRPC_MAX_MESSAGE_LENGTH_DEFAULT as u64
            ))
        );
        assert_eq!(
            crate::retries::grpc_python_code(&sent),
            tonic::Code::ResourceExhausted
        );
        assert_eq!(
            status_details_in(&sent, true),
            "CLIENT: Sent message larger than max (135266440 vs. 134217728)"
        );
        // Not another status, of another code or text, nor one of details, which a server sends,
        // nor a server's of tonic's text of its own limit, as a server of tonic sends of a
        // request over its limit of 4 MiB, nor a server's of the text of a request that the
        // client did not send, which only the client's own is.
        for other in [
            tonic::Status::out_of_range("Error, decoded message length too large"),
            tonic::Status::internal(received.message()),
            tonic::Status::with_details(
                tonic::Code::OutOfRange,
                received.message(),
                bytes::Bytes::from_static(&[8, 11]),
            ),
            tonic::Status::out_of_range(
                "Error, decoded message length too large: found 5000000 bytes, the limit is: \
                 4194304 bytes",
            ),
            tonic::Status::out_of_range(sent.message()),
        ] {
            assert_eq!(oversized(&other), None, "{other:?}");
            assert_eq!(crate::retries::grpc_python_code(&other), other.code());
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tonic_s_status_of_a_response_over_the_limit_and_a_request_over_it_are_oversized() {
        // tonic's own text of a response that it does not read, of the limit of the stub, which
        // `oversized` reads; and a request over the limit is refused before it is sent, of the
        // same text of its encoding, where tonic's check there resets the stream of the call.
        let received: Received = Default::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let service = Flooding(received.clone());
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let builder =
            ChannelBuilder::parse(&format!("sc://127.0.0.1:{}/", address.port())).unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        drop(env);
        let mut small =
            SparkConnectServiceClient::new(client.channel().await).max_decoding_message_size(1024);
        let mut stream = small
            .execute_plan(ExecutePlanRequest::default())
            .await
            .unwrap()
            .into_inner();
        let status = stream.message().await.unwrap_err();
        match oversized_of(&status, 1024) {
            Some((Oversized::Received, length, 1024)) => assert!(length > 256 * 1024),
            other => panic!("{other:?} of {status:?}"),
        }
        // Not of the limit of the client's calls.
        assert_eq!(oversized(&status), None);
        let big = spark_connect_proto::ConfigRequest {
            client_type: Some("x".repeat(GRPC_MAX_MESSAGE_LENGTH_DEFAULT)),
            ..Default::default()
        };
        let length = big.encoded_len();
        let status = client.stub().config(big).await.unwrap_err();
        assert_eq!(
            oversized(&status),
            Some((
                Oversized::Sent,
                length as u64,
                GRPC_MAX_MESSAGE_LENGTH_DEFAULT as u64
            ))
        );
        assert!(received
            .lock()
            .unwrap()
            .iter()
            .all(|(method, _)| method != "Config"));
        // Of the client, the error of RESOURCE_EXHAUSTED.
        let error = client
            .get_configs(&["x".repeat(GRPC_MAX_MESSAGE_LENGTH_DEFAULT).as_str()])
            .await
            .unwrap_err();
        assert_eq!(error.grpc_code, Some(tonic::Code::ResourceExhausted as i32));
    }

    /// As gRPC's global subchannel pool, which the channels of PySpark's clients share: the
    /// clients of the new sessions of a client, and every other client of the same target, have
    /// one connection between them, which stays while one of them is open; a client of another
    /// keepalive has its own, and so has each client of a token.
    #[tokio::test]
    async fn the_clients_of_a_target_share_the_connection_of_the_process_to_it() {
        let received: Received = Default::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let accepted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = std::sync::Arc::clone(&accepted);
        let incoming = futures::stream::unfold(listener, move |listener| {
            let counted = std::sync::Arc::clone(&counted);
            async move {
                let socket = listener.accept().await.map(|(socket, _)| {
                    counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    socket
                });
                Some((socket, listener))
            }
        });
        let service = Recorder(received.clone(), Default::default());
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let url = format!("sc://127.0.0.1:{}/", address.port());
        let builder = ChannelBuilder::parse(&url).unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        let again = SparkConnectClient::connect(&builder).await.unwrap();
        let other_keepalive =
            ChannelBuilder::parse(&format!("{url};grpc_keepalive_time_ms=45000")).unwrap();
        let apart = SparkConnectClient::connect(&other_keepalive).await.unwrap();
        drop(env);
        let accepted = || accepted.load(std::sync::atomic::Ordering::SeqCst);
        client.get_configs(&["k"]).await.unwrap();
        let sessions: Vec<_> = (0..5).map(|_| client.with_new_session_id()).collect();
        for session in &sessions {
            session.get_configs(&["k"]).await.unwrap();
        }
        again.get_configs(&["k"]).await.unwrap();
        assert_eq!(accepted(), 1);
        apart.get_configs(&["k"]).await.unwrap();
        assert_eq!(accepted(), 2);
        // A closed client of the channel does not end it for the others.
        client.close();
        sessions[0].get_configs(&["k"]).await.unwrap();
        assert_eq!(accepted(), 2);
        // A session of a token has a connection of its own, as grpcio's channel of credentials of
        // its own, also a new session of it, and another of the same token.
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let of_token =
            ChannelBuilder::parse(&format!("sc://localhost:{}/;token=AAA", address.port()))
                .unwrap();
        let tokened = SparkConnectClient::connect(&of_token).await.unwrap();
        let tokened_again = SparkConnectClient::connect(&of_token).await.unwrap();
        drop(env);
        tokened.get_configs(&["k"]).await.unwrap();
        assert_eq!(accepted(), 3);
        tokened_again.get_configs(&["k"]).await.unwrap();
        assert_eq!(accepted(), 4);
        tokened
            .with_new_session_id()
            .get_configs(&["k"])
            .await
            .unwrap();
        assert_eq!(accepted(), 5);
        // The clones of a client are of its session, of its channel.
        tokened.clone().get_configs(&["k"]).await.unwrap();
        assert_eq!(accepted(), 5);
    }

    /// The method, the `authorization` header and the port of the client of each request that a
    /// server received, in order.
    type Peers = std::sync::Arc<std::sync::Mutex<Vec<(String, Option<String>, u16)>>>;

    /// A [`Recorder`] that records the connection of each request too, of the port of its client.
    #[derive(Clone)]
    struct PeerRecorder(Recorder, Peers);

    impl NamedService for PeerRecorder {
        const NAME: &'static str = "spark.connect.SparkConnectService";
    }

    impl tonic::codegen::Service<http::Request<tonic::body::Body>> for PeerRecorder {
        type Response = http::Response<tonic::body::Body>;
        type Error = std::convert::Infallible;
        type Future = BoxFuture<Self::Response, Self::Error>;

        fn poll_ready(
            &mut self,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<std::result::Result<(), Self::Error>> {
            self.0.poll_ready(cx)
        }

        fn call(&mut self, request: http::Request<tonic::body::Body>) -> Self::Future {
            let method = request
                .uri()
                .path()
                .trim_start_matches("/spark.connect.SparkConnectService/")
                .to_string();
            let authorization = request
                .headers()
                .get("authorization")
                .and_then(|value| value.to_str().ok())
                .map(String::from);
            let port = request
                .extensions()
                .get::<tonic::transport::server::TcpConnectInfo>()
                .and_then(|info| info.remote_addr())
                .map_or(0, |address| address.port());
            self.1.lock().unwrap().push((method, authorization, port));
            self.0.call(request)
        }
    }

    /// A server of a [`PeerRecorder`] on the loopback, its port, what it received of the
    /// connections of the requests, and the count of the connections that it accepted.
    async fn peer_server() -> (u16, Peers, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let accepted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = std::sync::Arc::clone(&accepted);
        let incoming = futures::stream::unfold(listener, move |listener| {
            let counted = std::sync::Arc::clone(&counted);
            async move {
                let socket = listener.accept().await.map(|(socket, _)| {
                    counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    socket
                });
                Some((socket, listener))
            }
        });
        let peers: Peers = Default::default();
        let service = PeerRecorder(
            Recorder(Default::default(), Default::default()),
            peers.clone(),
        );
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        (port, peers, accepted)
    }

    /// As grpcio's channels of the credentials of their tokens, each of which has subchannels of
    /// its own (SPARK-62257): the calls of a client, and the releases of its executions in the
    /// background, go over the connection of its own channel, of its own token, and those of
    /// another client of another token over another.
    #[tokio::test]
    async fn the_calls_and_the_releases_of_a_client_go_over_the_connection_of_its_channel() {
        let (port, peers, accepted) = peer_server().await;
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        // Of `localhost`, whose token is sent in plaintext (see `ChannelBuilder::secure`).
        let client_of = |token: &str| {
            ChannelBuilder::parse(&format!("sc://localhost:{port}/;token={token}")).unwrap()
        };
        let first = SparkConnectClient::connect(&client_of("AAA"))
            .await
            .unwrap();
        let second = SparkConnectClient::connect(&client_of("BBB"))
            .await
            .unwrap();
        drop(env);
        let mut stream = first
            .execute_plan_reattachable(ExecutePlanRequest {
                session_id: first.session_id().to_string(),
                ..Default::default()
            })
            .await
            .unwrap();
        while stream.message().await.unwrap().is_some() {}
        drop(stream);
        second.get_configs(&["k"]).await.unwrap();
        first.wait_for_releases().await;
        first.get_configs(&["k"]).await.unwrap();
        assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 2);
        let peers = peers.lock().unwrap().clone();
        let ports_of = |token: &str| -> std::collections::BTreeSet<u16> {
            let bearer = format!("Bearer {token}");
            peers
                .iter()
                .filter(|(_, authorization, _)| authorization.as_deref() == Some(bearer.as_str()))
                .map(|(_, _, port)| *port)
                .collect()
        };
        let (of_first, of_second) = (ports_of("AAA"), ports_of("BBB"));
        assert_eq!((of_first.len(), of_second.len()), (1, 1), "{peers:?}");
        assert!(of_first.is_disjoint(&of_second), "{peers:?}");
        assert!(
            peers
                .iter()
                .any(|(method, authorization, port)| method == "ReleaseExecute"
                    && authorization.as_deref() == Some("Bearer AAA")
                    && of_first.contains(port)),
            "{peers:?}"
        );
    }

    /// As grpcio's `close` of the channel of a client, which lets go of its subchannels (Q5 of
    /// the probes of SPARK-62257): the connection of two clients stays while one of them is open,
    /// and ends once the last one is closed, so that a new client of the target connects again.
    #[tokio::test]
    async fn the_close_of_the_last_client_of_a_connection_ends_it() {
        let (port, _, accepted) = peer_server().await;
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let builder = ChannelBuilder::parse(&format!("sc://127.0.0.1:{port}/")).unwrap();
        let first = SparkConnectClient::connect(&builder).await.unwrap();
        let second = first.with_new_session_id();
        drop(env);
        first.get_configs(&["k"]).await.unwrap();
        second.get_configs(&["k"]).await.unwrap();
        let accepted = || accepted.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(accepted(), 1);
        first.close();
        let third = second.with_new_session_id();
        third.get_configs(&["k"]).await.unwrap();
        assert_eq!(accepted(), 1);
        second.close();
        third.close();
        let fourth = first.with_new_session_id();
        fourth.get_configs(&["k"]).await.unwrap();
        assert_eq!(accepted(), 2);
    }

    /// Of a server that sends its settings on each connection and closes it at once (P10 of the
    /// review of SPARK-62257): a call of the Rust API fails of the end of the connection, of no
    /// retry, of the connection of its channel's pass and one more at most, which tonic's
    /// connector makes once the one held ended before the call took it, where the connector made
    /// connections until one was open as it looked, and the client makes no connection more of
    /// itself once the call failed.
    #[tokio::test]
    async fn a_call_of_a_server_that_closes_each_connection_at_once_makes_two_connections_at_most()
    {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let accepted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = std::sync::Arc::clone(&accepted);
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let _ = stream.write_all(&[0, 0, 0, 4, 0, 0, 0, 0, 0]).await;
                    let _ = stream.shutdown().await;
                    let mut buffer = [0u8; 4096];
                    while matches!(stream.read(&mut buffer).await, Ok(read) if read > 0) {}
                });
            }
        });
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let builder = ChannelBuilder::parse(&format!("sc://127.0.0.1:{port}/")).unwrap();
        let mut client = SparkConnectClient::connect(&builder).await.unwrap();
        drop(env);
        client.set_retry_policy(crate::retries::RetryPolicy::no_retries());
        client.set_rpc_deadlines(RpcDeadlines {
            config: Some(Duration::from_secs(10)),
            ..RpcDeadlines::default()
        });
        let accepted = || accepted.load(std::sync::atomic::Ordering::SeqCst);
        let call = tokio::time::timeout(Duration::from_secs(5), client.get_configs(&["k"])).await;
        assert!(matches!(call, Ok(Err(_))), "{call:?}");
        let made = accepted();
        assert!((1..=2).contains(&made), "{made}");
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(accepted(), made);
        client.close();
    }

    /// As PySpark's executions, which send the request that the hooks return as it is: a plan that
    /// a hook puts in is sent as it is, over the threshold too, and the plan of the execution,
    /// compressed once, before the hook, as `to_proto` compresses it, is kept so by a hook that
    /// keeps it.
    #[tokio::test]
    async fn the_request_that_a_hook_returns_is_sent_as_it_is_as_pyspark() {
        let received: Received = Default::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let service = Recorder(received.clone(), Default::default());
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        // `connect` reads the token of the environment too, so none is set until it returns:
        // another test may set one, which would make the channel TLS.
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let builder =
            ChannelBuilder::parse(&format!("sc://127.0.0.1:{}/", address.port())).unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        drop(env);
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
        let putting = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let puts = std::sync::Arc::clone(&putting);
        let put = large.clone();
        client.set_execute_plan_hook(Some(std::sync::Arc::new(move |mut request| {
            if puts.load(std::sync::atomic::Ordering::SeqCst) {
                request.plan = Some(Plan {
                    op_type: Some(PlanOpType::Root(put.clone())),
                });
            }
            Ok(request)
        })));
        let of = |relation: spark_connect_proto::Relation| ExecutePlanRequest {
            session_id: client.session_id().to_string(),
            plan: Some(Plan {
                op_type: Some(PlanOpType::Root(relation)),
            }),
            ..Default::default()
        };
        let mut stream = client.execute_plan(of(Default::default())).await.unwrap();
        while stream.message().await.unwrap().is_some() {}
        let mut stream = client
            .execute_plan_reattachable(of(Default::default()))
            .await
            .unwrap();
        while stream.message().await.unwrap().is_some() {}
        putting.store(false, std::sync::atomic::Ordering::SeqCst);
        let mut stream = client.execute_plan_reattachable(of(large)).await.unwrap();
        while stream.message().await.unwrap().is_some() {}
        let executions: Vec<_> = received
            .lock()
            .unwrap()
            .iter()
            .filter(|(method, _)| method == "ExecutePlan")
            .map(|(_, bytes)| ExecutePlanRequest::decode(bytes.as_slice()).unwrap())
            .collect();
        assert_eq!(executions.len(), 3);
        let op_types: Vec<_> = executions
            .into_iter()
            .map(|request| request.plan.and_then(|plan| plan.op_type))
            .collect();
        assert!(
            matches!(
                op_types.as_slice(),
                [
                    Some(PlanOpType::Root(_)),
                    Some(PlanOpType::Root(_)),
                    Some(PlanOpType::CompressedOperation(_))
                ]
            ),
            "{op_types:?}"
        );
    }

    #[tokio::test]
    async fn the_request_of_the_first_command_observes_no_server_side_session_as_pyspark() {
        // As PySpark's `execute_command`, which builds its request before it sets the command in
        // its plan, with the lookup of the plan compression of the first plan: the request of the
        // first command of a session observes no server-side session ID, which the response of
        // the lookup gives, and the next one does. The plan of a relation is built before its
        // request (`to_proto`), which then observes it (see the test below).
        let (client, received, server) = recorded_client().await;
        let command = || ExecutePlanRequest {
            session_id: client.session_id().to_string(),
            plan: Some(Plan {
                op_type: Some(PlanOpType::Command(Default::default())),
            }),
            ..Default::default()
        };
        let mut stream = client.execute_plan_reattachable(command()).await.unwrap();
        while stream.message().await.unwrap().is_some() {}
        let mut stream = client.execute_plan(command()).await.unwrap();
        while stream.message().await.unwrap().is_some() {}
        server.abort();
        let requests = received.lock().unwrap().clone();
        let observed: Vec<(&str, Option<String>)> = requests
            .iter()
            .filter(|(method, _)| method != "ReleaseExecute")
            .map(|(method, bytes)| (method.as_str(), fields_of(method, bytes).1))
            .collect();
        let server_side = Some(SERVER_ID.to_string());
        assert_eq!(
            observed,
            [
                ("Config", None),
                ("ExecutePlan", None),
                ("ExecutePlan", server_side)
            ]
        );
    }

    #[tokio::test]
    async fn a_plan_sent_uncompressed_is_sent_as_it_is_without_the_lookup_as_pyspark() {
        // As PySpark's registrations of `_execute`, which set their command in the plan of their
        // request as it is: neither compressed, over the threshold too, nor of the lookup of the
        // plan compression, which the next plan makes.
        let (client, received, server) = recorded_client().await;
        let large = || ExecutePlanRequest {
            session_id: client.session_id().to_string(),
            plan: Some(Plan {
                op_type: Some(PlanOpType::Command(spark_connect_proto::Command {
                    command_type: Some(spark_connect_proto::command::CommandType::Extension(
                        prost_types::Any {
                            type_url: "example.org/large".into(),
                            value: vec![0; 64 * 1024],
                        },
                    )),
                })),
            }),
            ..Default::default()
        };
        let executions = || -> Vec<(String, Option<PlanOpType>)> {
            let requests = received.lock().unwrap().clone();
            requests
                .into_iter()
                .filter(|(method, _)| method != "ReleaseExecute")
                .map(|(method, bytes)| {
                    let op_type = (method == "ExecutePlan")
                        .then(|| ExecutePlanRequest::decode(bytes.as_slice()).unwrap())
                        .and_then(|request| request.plan)
                        .and_then(|plan| plan.op_type);
                    (method, op_type)
                })
                .collect()
        };
        let mut stream = client
            .execute_plan_reattachable_uncompressed(large())
            .await
            .unwrap();
        while stream.message().await.unwrap().is_some() {}
        let sent = executions();
        assert_eq!(sent.len(), 1, "{sent:?}");
        assert_eq!(sent[0].0, "ExecutePlan");
        assert!(matches!(sent[0].1, Some(PlanOpType::Command(_))));
        assert_eq!(
            *client.plan_compression.lock().unwrap(),
            PlanCompression::Unknown
        );
        // Of a plan compression of the client too, which compresses the others.
        *client.plan_compression.lock().unwrap() = PlanCompression::Zstd { threshold: 1000 };
        for compressed in [false, true] {
            let mut stream = if compressed {
                client.execute_plan_reattachable(large()).await.unwrap()
            } else {
                client
                    .execute_plan_reattachable_uncompressed(large())
                    .await
                    .unwrap()
            };
            while stream.message().await.unwrap().is_some() {}
            let op_type = executions().pop().unwrap().1;
            if compressed {
                assert!(matches!(op_type, Some(PlanOpType::CompressedOperation(_))));
            } else {
                assert!(matches!(op_type, Some(PlanOpType::Command(_))));
            }
        }
        server.abort();
    }

    #[tokio::test]
    async fn a_failure_while_the_thread_converts_another_of_its_client_is_not_converted() {
        // As PySpark's `_handle_error`, which raises the error of a call as grpc gives it while
        // the thread handles another failure of the client (`inside_error_handling` of the
        // `thread_local` of the client), as of a call that a handler of a log of the conversion
        // makes: neither the details of the error nor the configs of its stack trace are asked
        // for, where they are of a failure of another client, or once the conversion ended.
        let received: Received = Default::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let service = Recorder(received.clone(), Default::default());
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let builder =
            ChannelBuilder::parse(&format!("sc://127.0.0.1:{}/;user_id=u1", address.port()))
                .unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        drop(env);
        let other = client.with_new_session_id();
        let asked = || {
            let mut received = received.lock().unwrap();
            let methods: Vec<String> = received.drain(..).map(|(method, _)| method).collect();
            methods
        };
        // The details, and the config of the stack trace of the server, which shows it as the
        // server sends none.
        let details_and_configs = vec!["FetchErrorDetails".to_string(), "Config".to_string()];
        client.error_from_status(error_kept_under("error-1")).await;
        assert_eq!(asked(), details_and_configs);
        // Within the polls of a conversion of the client, as a call of a handler of one of its
        // records.
        crate::runtime::polling_conversion(client.identity(), async {
            assert!(client.inside_error_handling());
            assert!(!other.inside_error_handling());
            let error = client.error_from_status(error_kept_under("error-2")).await;
            assert_eq!(asked(), Vec::<String>::new());
            assert_eq!(error.grpc_code, Some(tonic::Code::Internal as i32));
            assert_eq!(error.server_stacktrace, None);
            // Of a client of another session, whose `thread_local` is its own.
            other.error_from_status(error_kept_under("error-3")).await;
            assert_eq!(asked(), details_and_configs);
            // Of a clone of the client, which shares it.
            let error = client
                .clone()
                .error_from_status(error_kept_under("error-4"))
                .await;
            assert_eq!(asked(), Vec::<String>::new());
            assert_eq!(error.server_stacktrace, None);
            // Of another thread, which polls no conversion.
            let elsewhere = client.clone();
            std::thread::spawn(move || {
                assert!(!elsewhere.inside_error_handling());
            })
            .join()
            .unwrap();
        })
        .await;
        assert!(!client.inside_error_handling());
        client.error_from_status(error_kept_under("error-5")).await;
        assert_eq!(asked(), details_and_configs);
        // Of a conversion of a call that a watch waits for, of the Python binding, which blocks its
        // thread: between the polls of the conversion too, as of a signal handler that a poll of
        // the watch runs; but not of a conversion of no watch, of an async caller, whose other
        // tasks the thread polls between the polls of the conversion.
        let failure = crate::error::grpcio_error(
            &tonic::Status::internal("failed"),
            GrpcCall::ServerStreaming,
        );
        let watch: std::sync::Arc<crate::runtime::WaitWatch> =
            std::sync::Arc::new(|| crate::runtime::Watch::Continue);
        crate::runtime::with_wait_watch(watch, || {
            let _converting = Converting::start(&failure, None, client.identity());
            assert!(client.inside_error_handling());
            assert!(!other.inside_error_handling());
        });
        let converting = Converting::start(&failure, None, client.identity());
        assert!(!client.inside_error_handling());
        drop(converting);
        server.abort();
    }

    #[tokio::test]
    async fn requests_carry_the_client_type_extensions_and_observed_session_of_pyspark() {
        let received: Received = Default::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let service = Recorder(received.clone(), Default::default());
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        // `connect` reads the token of the environment too, so none is set until it returns:
        // another test may set one, which would make the channel TLS.
        let env = crate::channel::set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let builder =
            ChannelBuilder::parse(&format!("sc://127.0.0.1:{}/;user_id=u1", address.port()))
                .unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        drop(env);
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
        let user_agent = client.client_type().unwrap();
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

    #[tokio::test]
    async fn the_user_agent_is_the_client_type_of_the_requests_and_not_their_header() {
        // As PySpark: its user agent goes in the requests, as their client type, and the
        // `user-agent` header is grpcio's own. So a user agent of control characters, which no
        // header can carry, is sent, and a `user-agent` parameter is not, as grpcio replaces it.
        let (client, received, user_agents, server) =
            recording_client(";user_agent=my%0A%7Fagent;user-agent=other").await;
        let user_agent = client.client_type().unwrap();
        assert!(
            user_agent.starts_with("my\n\u{7f}agent spark/4.3.0 os/"),
            "{user_agent:?}"
        );
        let session_id = client.session_id().to_string();
        client
            .config(ConfigRequest {
                session_id: session_id.clone(),
                ..Default::default()
            })
            .await
            .unwrap();
        let mut stream = client
            .execute_plan_reattachable(ExecutePlanRequest {
                session_id,
                ..Default::default()
            })
            .await
            .unwrap();
        while stream.message().await.unwrap().is_some() {}
        // The release of the execution is sent in the background.
        for _ in 0..200 {
            if received.lock().unwrap().len() == 3 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        server.abort();
        let requests = received.lock().unwrap().clone();
        let methods: Vec<&str> = requests.iter().map(|(m, _)| m.as_str()).collect();
        assert_eq!(methods, ["Config", "ExecutePlan", "ReleaseExecute"]);
        for (method, bytes) in &requests {
            assert_eq!(
                fields_of(method, bytes).0,
                Some(user_agent.clone()),
                "{method}"
            );
        }
        // One header on each request, tonic's own, as `tonic/<version>`.
        let user_agents = user_agents.lock().unwrap().clone();
        assert_eq!(user_agents.len(), 3);
        for values in user_agents {
            let values: Vec<String> = values
                .into_iter()
                .map(|value| String::from_utf8(value).unwrap())
                .collect();
            assert_eq!(values.len(), 1, "{values:?}");
            assert!(
                values[0].starts_with("tonic/") && !values[0].contains(' '),
                "{values:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_too_long_user_agent_is_the_error_of_each_request_that_carries_it_as_pyspark() {
        // As PySpark's `userAgent`, which its requests read when they are built: the client is
        // made, and each request of the user agent raises its SparkConnectException and is not
        // sent. Those that PySpark builds without it, of the artifacts and CloneSession, are.
        let (client, received, _, server) =
            recording_client(&format!(";user_agent={}", "a".repeat(2049))).await;
        fn error_of<T>(result: Result<T>) -> SparkError {
            match result {
                Ok(_) => panic!("the request was made"),
                Err(error) => error,
            }
        }
        let too_long = |error: SparkError, call: &str| {
            assert_eq!(
                error.kind,
                SparkErrorKind::ConnectMessage,
                "{call}: {error:?}"
            );
            assert_eq!(
                error.message(),
                "'user_agent' parameter should not exceed 2048 characters after URL escaping, \
                 found 2049 characters.",
                "{call}"
            );
        };
        too_long(error_of(client.config(Default::default()).await), "Config");
        too_long(error_of(client.get_configs(&["k"]).await), "get_configs");
        let analyzed = client.analyze_plan(Default::default()).await;
        too_long(error_of(analyzed), "AnalyzePlan");
        too_long(
            error_of(client.execute_plan(Default::default()).await),
            "ExecutePlan",
        );
        let executed = client.execute_plan_reattachable(Default::default()).await;
        too_long(error_of(executed), "a reattachable ExecutePlan");
        too_long(error_of(client.interrupt_all().await), "Interrupt");
        let fetched = client.fetch_error_details(Default::default()).await;
        too_long(error_of(fetched), "FetchErrorDetails");
        too_long(error_of(client.release_session().await), "ReleaseSession");
        // PySpark builds the request of the details of an error of the server with its user
        // agent, whose error is then that of the call.
        let raised = client.error_from_status(error_kept_under("error-1")).await;
        too_long(raised, "an error of the server");
        assert!(received.lock().unwrap().is_empty());
        client.artifact_status(&["files/a.txt"]).await.unwrap();
        client.clone_session(None).await.unwrap();
        // So is a request of a closed client, as PySpark builds a request before it reads
        // `_stub`, which raises NO_ACTIVE_SESSION.
        client.close();
        too_long(
            error_of(client.config(Default::default()).await),
            "a closed client",
        );
        server.abort();
        let requests = received.lock().unwrap().clone();
        let methods: Vec<&str> = requests.iter().map(|(m, _)| m.as_str()).collect();
        assert_eq!(methods, ["ArtifactStatus", "CloneSession"]);
    }

    /// A client of a recording server, of the parameters `params` of its connection string, what
    /// that server received, the `user-agent` headers of the requests, and the server.
    async fn recording_client(
        params: &str,
    ) -> (
        SparkConnectClient,
        Received,
        UserAgents,
        tokio::task::JoinHandle<()>,
    ) {
        let received: Received = Default::default();
        let user_agents: UserAgents = Default::default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let service = Recorder(received.clone(), user_agents.clone());
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        // `connect` reads the token and the user agent of the environment too, so neither is
        // set until it returns: another test may set them, which would make the channel TLS, or
        // give it another user agent.
        let env = crate::channel::set_env(&[
            ("SPARK_CONNECT_AUTHENTICATE_TOKEN", None),
            ("SPARK_CONNECT_USER_AGENT", None),
        ]);
        let builder =
            ChannelBuilder::parse(&format!("sc://127.0.0.1:{}/{params}", address.port())).unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        drop(env);
        (client, received, user_agents, server)
    }

    /// A client of the user `u1` of a recording server, what that server received, and the
    /// server.
    async fn recorded_client() -> (SparkConnectClient, Received, tokio::task::JoinHandle<()>) {
        let (client, received, _, server) = recording_client(";user_id=u1").await;
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
        assert_eq!(request.client_type, Some(client.client_type().unwrap()));
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
    fn a_user_agent_of_control_characters_connects_as_pyspark() {
        // PySpark sends its user agent only as the client type of the requests, which takes
        // any character, so a control character is no error, of the connection or of a request.
        for (url, user_agent) in [
            ("sc://localhost/;user_agent=my%0Aagent", "my\nagent"),
            ("sc://localhost/;user_agent=my%7Fagent", "my\u{7f}agent"),
            ("sc://localhost/;user_agent=%00%0D", "\0\r"),
        ] {
            let client = connect(url).unwrap();
            let client_type = client.client_type().unwrap();
            assert!(
                client_type.starts_with(&format!("{user_agent} spark/")),
                "{url}: {client_type:?}"
            );
        }
    }

    #[test]
    fn a_user_agent_parameter_is_checked_as_metadata_and_not_sent() {
        // grpcio checks a `user-agent` of the metadata as any other, and sends its own header in
        // its place; here that is tonic's, as on every request.
        let client = connect("sc://localhost/;user-agent=other;x-hdr=v").unwrap();
        let mut request = Request::new(());
        client._attach_metadata(&mut request);
        for metadata in [request.metadata().clone(), client.execution_metadata()] {
            assert!(metadata.get("user-agent").is_none(), "{metadata:?}");
            assert_eq!(metadata.get("x-hdr").unwrap(), "v");
        }
        let e = connect_err("sc://localhost/;user-agent=a%0Ab");
        assert_eq!(e.kind, SparkErrorKind::BuiltinValueError);
        assert!(shown(&e).contains("metadata was invalid"), "{}", shown(&e));
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
            // The user agent is the client type of the requests, not a header.
            assert_eq!(metadata.len(), 3, "{metadata:?}");
        }
        assert!(client
            .client_type()
            .unwrap()
            .starts_with("caf\u{e9} spark/"));
    }

    #[test]
    fn the_token_follows_an_authorization_parameter() {
        // As grpcio sends the metadata of a call, of PySpark's `ChannelBuilder.metadata()`, and
        // then that of its call credentials, the token's: a server that reads the last value of a
        // key, as grpc-java's `Metadata.get`, takes the token.
        let client = connect("sc://localhost/;authorization=Basic%20abc;token=tok").unwrap();
        for metadata in [client.execution_metadata(), {
            let mut request = Request::new(());
            client._attach_metadata(&mut request);
            request.metadata().clone()
        }] {
            let all: Vec<_> = metadata.get_all("authorization").iter().collect();
            assert_eq!(all, ["Basic abc", "Bearer tok"]);
        }
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

/// The error of a response that the client fails to decode, of its codec, and of a status of the
/// server of the same text.
#[cfg(test)]
mod decode_failure_tests {
    use super::*;
    use crate::bytes_codec::BytesCodec;
    use spark_connect_proto::analyze_plan_request::SparkVersion;
    use tonic::codegen::{http, BoxFuture};
    use tonic::server::{NamedService, UnaryService};

    /// The status that a server sends of a request that it fails to decode, as a server of tonic
    /// does: the code and the text of the client's own failure to decode a response.
    const SERVERS_FAILURE: &str =
        "failed to decode Protobuf message: AnalyzePlanRequest.plan: buffer underflow";

    /// A server whose RPCs answer bytes that no response decodes of, or (`fails`) fail of the
    /// status of a server that fails to decode the request.
    #[derive(Clone)]
    struct Undecodable {
        fails: bool,
    }

    impl NamedService for Undecodable {
        const NAME: &'static str = "spark.connect.SparkConnectService";
    }

    impl tonic::codegen::Service<http::Request<tonic::body::Body>> for Undecodable {
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
            let rpc = self.clone();
            Box::pin(async move {
                let mut grpc = tonic::server::Grpc::new(BytesCodec);
                Ok(grpc.unary(rpc, request).await)
            })
        }
    }

    impl UnaryService<Vec<u8>> for Undecodable {
        type Response = Vec<u8>;
        type Future = BoxFuture<tonic::Response<Vec<u8>>, tonic::Status>;

        fn call(&mut self, _: tonic::Request<Vec<u8>>) -> Self::Future {
            let fails = self.fails;
            Box::pin(async move {
                if fails {
                    Err(tonic::Status::internal(SERVERS_FAILURE))
                } else {
                    // A field of the number 1 of a length past the end.
                    Ok(tonic::Response::new(vec![0x0a, 0x05]))
                }
            })
        }
    }

    /// The error of `AnalyzePlan` of a client of an [`Undecodable`] server.
    async fn analyze_error(fails: bool) -> SparkError {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(Undecodable { fails })
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        // As `recording_client`: no token nor user agent of the environment.
        let env = crate::channel::set_env(&[
            ("SPARK_CONNECT_AUTHENTICATE_TOKEN", None),
            ("SPARK_CONNECT_USER_AGENT", None),
        ]);
        let builder =
            ChannelBuilder::parse(&format!("sc://127.0.0.1:{}/", address.port())).unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        drop(env);
        let request = AnalyzePlanRequest {
            session_id: client.session_id().to_string(),
            analyze: Some(Analyze::SparkVersion(SparkVersion {})),
            ..Default::default()
        };
        let error = client.analyze_plan(request).await.unwrap_err();
        server.abort();
        error
    }

    #[tokio::test]
    async fn a_response_that_does_not_decode_is_pysparks_error_of_it_and_a_servers_status_its_own()
    {
        // The status of the client's codec of a response that does not decode is PySpark's error
        // of a response that grpcio fails to deserialize; the status of a server of the same code
        // and text, as of its failure to decode a request, is the text of grpcio's error of it.
        // The error of the Rust API is prost's of either.
        let undecodable = analyze_error(false).await;
        assert_eq!(
            undecodable.pyspark_message.as_deref(),
            Some(
                "<_InactiveRpcError of RPC that terminated with:\n\
                 \tstatus = StatusCode.INTERNAL\n\
                 \tdetails = \"Exception deserializing response!\"\n\
                 \tdebug_error_string = \"None\"\n>"
            )
        );
        assert!(
            undecodable
                .message()
                .contains("failed to decode Protobuf message: AnalyzePlanResponse"),
            "{}",
            undecodable.message()
        );
        let of_server = analyze_error(true).await;
        assert_eq!(
            of_server.pyspark_message,
            Some(format!(
                "<_InactiveRpcError of RPC that terminated with:\n\
                 \tstatus = StatusCode.INTERNAL\n\tdetails = \"{SERVERS_FAILURE}\"\n\
                 \tdebug_error_string = \"INTERNAL:{SERVERS_FAILURE}\"\n>"
            ))
        );
        assert!(of_server.message().contains(SERVERS_FAILURE));
    }
}
