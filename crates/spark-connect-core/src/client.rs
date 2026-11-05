//! Spark Connect client implementation.
//!
//! Mirrors `pyspark.sql.connect.client.core.SparkConnectClient`.

use std::collections::BTreeMap;
use std::str::FromStr;
use std::time::Duration;
use tonic::metadata::{MetadataMap, MetadataValue};
use tonic::transport::{Channel, ClientTlsConfig};
use tonic::{Request, Response, Streaming};
use uuid::Uuid;

use spark_connect_proto::spark_connect_service_client::SparkConnectServiceClient;
use spark_connect_proto::{
    AnalyzePlanRequest, AnalyzePlanResponse, ArtifactStatusesRequest, ConfigRequest,
    ConfigResponse, ExecutePlanRequest, ExecutePlanResponse, FetchErrorDetailsRequest,
    FetchErrorDetailsResponse, InterruptRequest, InterruptResponse, ReattachExecuteRequest,
    ReleaseExecuteRequest, ReleaseExecuteResponse, ReleaseSessionRequest, UserContext,
};

use crate::artifact::{
    build_artifact_request_stream, cache_artifact_requests, FileArtifact, NamedArtifact,
};
use crate::channel::{ChannelBuilder, GRPC_MAX_MESSAGE_LENGTH_DEFAULT};
use crate::error::{Result, SparkError};
use crate::reattach::ExecutePlanResponseReattachableIterator;
use crate::retries::{RetryPolicy, RetryPolicyState};

/// How long to wait for the lazy gRPC channel to become ready before surfacing a
/// retriable `UNAVAILABLE`. tonic's lazy channel waits indefinitely for an unreachable
/// or unresolvable host (unlike grpcio, which fails fast), so without this bound a call
/// to a down server would hang forever instead of raising `UNAVAILABLE`.
const READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

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

/// Hash up to 128 blobs totaling at most 256 KiB inline; offload larger batches.
/// Returns the original buffers for upload framing. If the caller is dropped after
/// blocking hashing starts, the task retains the blob buffers until it finishes.
async fn hash_artifact_blobs(blobs: Vec<Vec<u8>>) -> Result<(Vec<Vec<u8>>, Vec<String>)> {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;

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
            .map(|blob| {
                let mut hash = String::with_capacity(64);
                for byte in Sha256::digest(blob) {
                    write!(&mut hash, "{byte:02x}").expect("writing to a String cannot fail");
                }
                hash
            })
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
        std::panic::resume_unwind(error.into_panic());
    }
    SparkError::connect_msg(format!("Artifact hashing task failed: {error}"))
}

/// A Spark Connect client for communicating with a remote Spark server.
///
/// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient`.
#[derive(Clone)]
pub struct SparkConnectClient {
    /// The tonic gRPC channel.
    channel: Channel,
    /// The generated gRPC client stub.
    stub: SparkConnectServiceClient<Channel>,
    /// Session ID (UUID v4) identifying this session on the server.
    session_id: String,
    /// User context (user_id from the channel builder).
    user_id: Option<String>,
    /// Metadata headers to attach to every request.
    metadata: Vec<(String, String)>,
    /// User-agent header.
    user_agent: String,
    /// Retry policy applied to RPCs (exponential backoff on transient failures).
    retry_policy: RetryPolicy,
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

        // Parse user-agent.
        let user_agent = builder.user_agent()?;

        // Build gRPC channel. We connect *lazily* - the channel is created here but the
        // TCP/TLS connection is established on the first RPC (matching the reference
        // client, where getOrCreate() does not dial the server). This also means the
        // client can be constructed without a live server (e.g. in unit tests that only
        // build request protos).
        let endpoint = builder.endpoint();
        let channel = if builder.use_ssl() {
            // TLS connection with native root certificates
            // `with_native_roots()` is required: tonic >=0.11 does NOT load a trust store
            // from the `tls-native-roots` feature alone, so without this the TLS handshake
            // has no CA anchors, fails, and the retry policy masks it as a hang.
            let tls_config = ClientTlsConfig::new()
                .domain_name(builder.host())
                .with_native_roots();

            tonic::transport::Channel::from_shared(format!("https://{endpoint}"))
                .map_err(|e| SparkError::connect_msg(format!("Invalid endpoint: {}", e)))?
                .tls_config(tls_config)
                .map_err(|e| SparkError::connect_msg(format!("Failed to configure TLS: {}", e)))?
                .connect_timeout(READY_TIMEOUT)
                .connect_lazy()
        } else {
            // Plaintext connection (for localhost development)
            tonic::transport::Channel::from_shared(format!("http://{endpoint}"))
                .map_err(|e| SparkError::connect_msg(format!("Invalid endpoint: {}", e)))?
                .connect_timeout(READY_TIMEOUT)
                .connect_lazy()
        };

        // Raise the message-size cap from tonic's 4 MiB default (too small for real Arrow
        // `collect()` results) to 128 MiB, applied to both the stub and the raw Grpc calls
        // below. The reference client sets this unconditionally via GRPC_DEFAULT_OPTIONS
        // (grpc.max_{send,receive}_message_length); it is not a connection-string option.
        let stub = SparkConnectServiceClient::new(channel.clone())
            .max_decoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT)
            .max_encoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT);

        let mut metadata: Vec<(String, String)> = builder.metadata();
        // Spark Connect servers that require token auth expect the connection-string
        // `token` as an `Authorization: Bearer <token>` header. It is a reserved key
        // excluded from `metadata()`, so attach it explicitly here; this then propagates
        // to every request via `_attach_metadata` and the reattach stream.
        if let Some(tok) = builder.token() {
            // Never send the bearer token in cleartext. The scheme above is chosen from
            // `use_ssl()` alone, so without this guard `sc://host/;token=SECRET` (no
            // `use_ssl`) would ship `Authorization: Bearer SECRET` over plain HTTP. The
            // reference client cannot reach that state: `toChannel` sends a token only
            // over `ssl_channel_credentials()` or, for `localhost`, the loopback
            // `local_channel_credentials()`, and grpcio refuses call credentials on an
            // insecure channel outright. Mirror that by requiring TLS for a token unless
            // the endpoint is loopback; fail loudly rather than downgrade silently.
            if !builder.use_ssl() && !builder.is_loopback() {
                // A misconfigured connection string is a value error (surfaces as
                // `PySparkValueError`), not a generic runtime failure.
                return Err(SparkError::value_msg(format!(
                    "Refusing to send the authentication token to '{}' over an insecure \
                     connection. Set 'use_ssl=true' in the connection string to enable TLS \
                     (a token is only allowed without TLS when connecting to localhost).",
                    builder.host()
                )));
            }
            metadata.push((AUTHORIZATION_HEADER.to_string(), format!("Bearer {}", tok)));
        }

        Ok(Self {
            channel,
            stub,
            session_id,
            user_id: builder.user_id().map(String::from),
            metadata,
            user_agent,
            retry_policy: RetryPolicy::default(),
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
            stub: self.stub.clone(),
            session_id: Uuid::new_v4().to_string(),
            user_id: self.user_id.clone(),
            metadata: self.metadata.clone(),
            user_agent: self.user_agent.clone(),
            retry_policy: self.retry_policy.clone(),
        }
    }

    /// Get the user ID, if set.
    pub fn user_id(&self) -> Option<&str> {
        self.user_id.as_deref()
    }

    /// Override the retry policy applied to this client's RPCs.
    ///
    /// The transport-injection stub sets this to [`RetryPolicy::no_retries`] so it behaves
    /// like the single-shot grpcio stub the reference client expects (which does its own
    /// retrying on top); see that constructor for why double-retrying is wrong there.
    pub fn set_retry_policy(&mut self, policy: RetryPolicy) {
        self.retry_policy = policy;
    }

    /// Execute a plan and return a stream of responses.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.execute_plan`.
    pub async fn execute_plan(
        &self,
        request: ExecutePlanRequest,
    ) -> Result<Streaming<ExecutePlanResponse>> {
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
        request: ReattachExecuteRequest,
    ) -> Result<Streaming<ExecutePlanResponse>> {
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        match self.stub.clone().reattach_execute(req).await {
            Ok(response) => Ok(response.into_inner()),
            Err(status) => Err(self.error_from_status(status).await),
        }
    }

    /// Release a (portion of a) running execution's response stream.
    ///
    /// Mirrors the `ReleaseExecute` RPC used by the reattachable execute path.
    pub async fn release_execute(
        &self,
        request: ReleaseExecuteRequest,
    ) -> Result<ReleaseExecuteResponse> {
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        match self.stub.clone().release_execute(req).await {
            Ok(response) => Ok(response.into_inner()),
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
        let mut grpc = tonic::client::Grpc::new(self.channel.clone())
            .max_decoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT)
            .max_encoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT);
        wait_ready(&mut grpc).await?;
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        let path = tonic::codegen::http::uri::PathAndQuery::from_static(
            "/spark.connect.SparkConnectService/AnalyzePlan",
        );
        let resp = grpc
            .unary(req, path, crate::bytes_codec::BytesCodec)
            .await
            .map_err(SparkError::from_grpc_status)?;
        Ok(resp.into_inner())
    }

    /// Analyze a plan and return metadata about it.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.analyze_plan`.
    pub async fn analyze_plan(&self, request: AnalyzePlanRequest) -> Result<AnalyzePlanResponse> {
        self.with_retry(|| {
            let mut req = Request::new(request.clone());
            self._attach_metadata(&mut req);
            let mut stub = self.stub.clone();
            async move { stub.analyze_plan(req).await.map(Response::into_inner) }
        })
        .await
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
        let mut state = RetryPolicyState::new(self.retry_policy.clone());
        loop {
            match op().await {
                Ok(v) => return Ok(v),
                Err(status) => {
                    if self.retry_policy.can_retry(&status) {
                        if let Some(wait_ms) = state.next_attempt(None) {
                            tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
                            continue;
                        }
                    }
                    return Err(self.error_from_status(status).await);
                }
            }
        }
    }

    /// Execute a plan as a reattachable stream: retries the initial call and, if the
    /// response stream drops mid-flight (e.g. `UNAVAILABLE` / `INVALID_CURSOR.DISCONNECTED`)
    /// or the server ends it before `ResultComplete`, transparently resumes it with
    /// `ReattachExecute` from the last observed `response_id`. The stream releases the
    /// execution on the server with `ReleaseExecute` (see [`ReattachableResponseStream`]).
    /// Mirrors the reference reattachable execute path.
    pub async fn execute_plan_reattachable(
        &self,
        request: ExecutePlanRequest,
    ) -> Result<ReattachableResponseStream> {
        let iter = ExecutePlanResponseReattachableIterator::new(request);
        let start = iter.request().clone();
        let stream = self
            .with_retry(|| {
                let mut req = Request::new(start.clone());
                self._attach_metadata(&mut req);
                let mut stub = self.stub.clone();
                async move { stub.execute_plan(req).await.map(Response::into_inner) }
            })
            .await
            .map_err(|e| e.with_operation_id(iter.operation_id()))?;
        let retry_policy = self.retry_policy.clone();
        Ok(ReattachableResponseStream {
            client: self.clone(),
            retry_state: RetryPolicyState::new(retry_policy.clone()),
            retry_policy,
            iter,
            stream,
            done: false,
            released: false,
        })
    }

    /// Fetch enriched error details (full exception tree/stack trace) by error id.
    ///
    /// Mirrors `SparkConnectClient._fetch_enriched_error`; used to reconstruct the full
    /// server-side exception message (e.g. wrapped worker errors).
    pub async fn fetch_error_details(
        &self,
        request: FetchErrorDetailsRequest,
    ) -> Result<FetchErrorDetailsResponse> {
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        let resp = self.stub.clone().fetch_error_details(req).await;
        resp.map(Response::into_inner)
            .map_err(SparkError::from_grpc_status)
    }

    /// The error for a failed RPC, as `SparkConnectClient._handle_rpc_error` raises it. An
    /// error that the server raised gets the details that `FetchErrorDetails` returns when the
    /// server keeps them, and shows the server's stack trace when
    /// `spark.sql.connect.serverStacktrace.enabled` or `spark.sql.pyspark.jvmStacktrace.enabled`
    /// is true. As the error is raised either way, failing to read those is not.
    pub async fn error_from_status(&self, status: tonic::Status) -> SparkError {
        let error = SparkError::from_grpc_status(status);
        let Some(server) = &error.server else {
            return error;
        };
        let details = match &server.error_id {
            Some(error_id) => self.error_details(error_id).await,
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
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        let response = self.stub.clone().fetch_error_details(req).await.ok()?;
        Some(response.into_inner())
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
            ..Default::default()
        };
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        let response = self.stub.clone().config(req).await.ok()?.into_inner();
        response.pairs.into_iter().next()?.value
    }

    /// Update or fetch configurations.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.config`.
    pub async fn config(&self, request: ConfigRequest) -> Result<ConfigResponse> {
        self.with_retry(|| {
            let mut req = Request::new(request.clone());
            self._attach_metadata(&mut req);
            let mut stub = self.stub.clone();
            async move { stub.config(req).await.map(Response::into_inner) }
        })
        .await
    }

    /// Interrupt running operations on this session.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.interrupt`.
    pub async fn interrupt(&self, request: InterruptRequest) -> Result<InterruptResponse> {
        self.with_retry(|| {
            let mut req = Request::new(request.clone());
            self._attach_metadata(&mut req);
            let mut stub = self.stub.clone();
            async move { stub.interrupt(req).await.map(Response::into_inner) }
        })
        .await
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

    /// Set configuration values for the session.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.set_config`.
    pub async fn set_config(&self, key: &str, value: &str) -> Result<()> {
        let mut request = ConfigRequest::default();
        request.session_id = self.session_id.clone();
        request.user_context = Some(UserContext {
            user_id: self.user_id.clone().unwrap_or_default(),
            ..Default::default()
        });

        let mut set = spark_connect_proto::config_request::Set::default();
        set.pairs.push(spark_connect_proto::KeyValue {
            key: key.to_string(),
            value: Some(value.to_string()),
        });

        let mut operation = spark_connect_proto::config_request::Operation::default();
        operation.op_type = Some(spark_connect_proto::config_request::operation::OpType::Set(
            set,
        ));
        request.operation = Some(operation);

        let _ = self.config(request).await?;
        Ok(())
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

    /// Unset a configuration value.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.unset_config`.
    pub async fn unset_config(&self, key: &str) -> Result<()> {
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

        let _ = self.config(request).await?;
        Ok(())
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
    pub async fn release_session(&self) -> Result<()> {
        let mut request = ReleaseSessionRequest::default();
        request.session_id = self.session_id.clone();
        request.user_context = Some(UserContext {
            user_id: self.user_id.clone().unwrap_or_default(),
            ..Default::default()
        });

        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        match self.stub.clone().release_session(req).await {
            Ok(_) => Ok(()),
            Err(status) => Err(self.error_from_status(status).await),
        }
    }

    /// Add artifacts to this session.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.add_artifacts`.
    /// Handles chunking large artifacts and batching small ones according to CHUNK_SIZE.
    pub async fn add_artifacts(
        &self,
        paths: &[&str],
        _pyfile: bool,
        _archive: bool,
        _file: bool,
    ) -> Result<()> {
        // Build list of named artifacts from file paths
        let mut artifacts = Vec::new();
        for path in paths {
            // Extract just the filename as the artifact name
            let path_obj = std::path::Path::new(path);
            let file_name = path_obj
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| {
                    SparkError::connect_msg(format!("Invalid artifact path: {}", path))
                })?;

            // Determine artifact prefix based on file extension
            let artifact_name = if _pyfile
                && (file_name.ends_with(".py")
                    || file_name.ends_with(".zip")
                    || file_name.ends_with(".egg")
                    || file_name.ends_with(".jar"))
            {
                format!("pyfiles/{}", file_name)
            } else if _archive
                && (file_name.ends_with(".zip")
                    || file_name.ends_with(".jar")
                    || file_name.ends_with(".tar.gz")
                    || file_name.ends_with(".tgz")
                    || file_name.ends_with(".tar"))
            {
                format!("archives/{}", file_name)
            } else if _file {
                format!("files/{}", file_name)
            } else if file_name.ends_with(".jar") {
                format!("jars/{}", file_name)
            } else {
                return Err(SparkError::connect_msg(format!(
                    "Unsupported artifact type: {}",
                    path
                )));
            };

            // Create artifact from file
            artifacts.push(NamedArtifact::new(
                artifact_name,
                Box::new(FileArtifact::new(path)),
            ));
        }

        if artifacts.is_empty() {
            return Ok(());
        }

        // Build request stream with proper chunking/batching
        let requests = build_artifact_request_stream(self.session_id.clone(), artifacts)?;
        self.upload_artifact_requests(futures::stream::iter(requests), None)
            .await
    }

    /// Upload a single local file as an artifact under an explicit artifact name
    /// (rather than deriving `files/`/`jars/`… from the extension).
    ///
    /// Used by `SparkSession::copy_from_local_to_fs`, which names the artifact
    /// `forward_to_fs/<dest>` so the server writes it to the target filesystem path.
    pub async fn add_named_artifact(&self, name: &str, local_path: &str) -> Result<()> {
        let artifacts = vec![NamedArtifact::new(
            name.to_string(),
            Box::new(FileArtifact::new(local_path)),
        )];
        let requests = build_artifact_request_stream(self.session_id.clone(), artifacts)?;
        self.upload_artifact_requests(futures::stream::iter(requests), None)
            .await
    }

    /// Upload session-scoped cache blobs, returning SHA-256 hashes in input order.
    /// Existing artifacts and duplicate blobs are uploaded only once.
    pub async fn cache_artifacts(&self, blobs: Vec<Vec<u8>>) -> Result<Vec<String>> {
        use bytes::Bytes;
        use std::collections::HashSet;

        if blobs.is_empty() {
            return Ok(Vec::new());
        }
        let (blobs, hashes) = hash_artifact_blobs(blobs).await?;
        let names: Vec<String> = hashes.iter().map(|hash| format!("cache/{hash}")).collect();
        let unique: HashSet<&str> = names.iter().map(String::as_str).collect();
        let cached = self
            .artifact_status(&unique.into_iter().collect::<Vec<_>>())
            .await?;
        let mut seen = HashSet::new();
        let mut artifacts = Vec::new();
        for (blob, name) in blobs.into_iter().zip(names) {
            if !cached.get(&name).copied().unwrap_or(false) && seen.insert(name.clone()) {
                artifacts.push((name, Bytes::from(blob)));
            }
        }
        if artifacts.is_empty() {
            return Ok(hashes);
        }
        let expected: HashSet<String> = artifacts.iter().map(|(name, _)| name.clone()).collect();
        let requests = cache_artifact_requests(self.session_id.clone(), artifacts);
        self.upload_artifact_requests(futures::stream::iter(requests), Some(&expected))
            .await?;
        Ok(hashes)
    }

    /// PySpark and Scala ArtifactManager both send a UserContext even when no
    /// user ID is configured.
    fn artifact_user_context(&self) -> UserContext {
        UserContext {
            user_id: self.user_id.clone().unwrap_or_default(),
            ..Default::default()
        }
    }

    /// Send artifact frames and validate the response.
    /// Stamps user_context and client_type on every frame; all upload paths must
    /// go through here.
    /// Cache uploads also require an acknowledgement for every requested artifact.
    /// The stream is consumed once; retries must rebuild the frames and restart
    /// AddArtifacts rather than resume a partially consumed stream.
    async fn upload_artifact_requests(
        &self,
        requests: impl futures::Stream<Item = spark_connect_proto::AddArtifactsRequest> + Send + 'static,
        expected_cache_names: Option<&std::collections::HashSet<String>>,
    ) -> Result<()> {
        use futures::StreamExt;

        let user_context = self.artifact_user_context();
        let client_type = self.user_agent.clone();
        let requests = requests.map(move |mut request| {
            request.user_context = Some(user_context.clone());
            request.client_type = Some(client_type.clone());
            request
        });
        let mut req = Request::new(requests);
        self._attach_metadata(&mut req);
        let response = match self.stub.clone().add_artifacts(req).await {
            Ok(response) => response.into_inner(),
            Err(status) => return Err(self.error_from_status(status).await),
        };
        let mut received = std::collections::HashSet::new();
        for summary in response.artifacts {
            if !summary.is_crc_successful {
                return Err(SparkError::connect_msg(format!(
                    "CRC check failed for artifact: {}",
                    summary.name
                )));
            }
            if expected_cache_names.is_some() {
                received.insert(summary.name);
            }
        }
        if let Some(expected) = expected_cache_names {
            if !expected.is_subset(&received) {
                return Err(SparkError::connect_msg(
                    "Missing cache artifact upload acknowledgement",
                ));
            }
        }
        Ok(())
    }

    /// Check artifact status on the server.
    ///
    /// Mirrors functionality from `pyspark.sql.connect.client.artifact.ArtifactManager.is_cached_artifact`.
    pub async fn artifact_status(
        &self,
        names: &[&str],
    ) -> Result<std::collections::HashMap<String, bool>> {
        let mut request = ArtifactStatusesRequest::default();
        request.session_id = self.session_id.clone();
        request.user_context = Some(self.artifact_user_context());
        request.client_type = Some(self.user_agent.clone());
        request.names = names.iter().map(|n| n.to_string()).collect();

        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        let response = match self.stub.clone().artifact_status(req).await {
            Ok(response) => response.into_inner(),
            Err(status) => return Err(self.error_from_status(status).await),
        };

        // Build map of artifact name -> exists status
        let mut result = std::collections::HashMap::new();
        for (name, status) in response.statuses {
            result.insert(name, status.exists);
        }

        Ok(result)
    }

    /// Attach metadata headers to a request.
    ///
    /// Inserts session_id, user-agent, and token (if present) into the gRPC metadata.
    fn _attach_metadata<T>(&self, req: &mut Request<T>) {
        let metadata = req.metadata_mut();
        // NOTE: the session id is carried in the request *body* (proto), not in gRPC
        // metadata. The reference client sends no `session_id` header; sending one
        // (with our own client-generated id, which differs from the body's session id
        // when we forward a request built by another client) can make the server
        // associate work with the wrong session. So we do not attach it here.
        // Add user-agent
        if let Ok(header_value) = MetadataValue::from_str(&self.user_agent) {
            let _ = metadata.insert("user-agent", header_value);
        }
        // Add user_id if present
        if let Some(user_id) = &self.user_id {
            if let Ok(header_value) = MetadataValue::from_str(user_id) {
                let _ = metadata.insert("user_id", header_value);
            }
        }
        // Add custom metadata
        for (k, v) in &self.metadata {
            if let Ok(header_value) = MetadataValue::from_str(v) {
                if let Ok(key) = tonic::metadata::MetadataKey::from_bytes(k.as_bytes()) {
                    let _ = metadata.insert(key, header_value);
                }
            }
        }
    }
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
pub struct ReattachableResponseStream {
    /// The client that started the execution, whose stub, headers and session it uses.
    client: SparkConnectClient,
    retry_policy: RetryPolicy,
    retry_state: RetryPolicyState,
    iter: ExecutePlanResponseReattachableIterator,
    stream: Streaming<ExecutePlanResponse>,
    done: bool,
    /// Whether `ReleaseAll` was sent, after which nothing else is released.
    released: bool,
}

impl ReattachableResponseStream {
    /// The gRPC metadata of every request about this execution.
    fn metadata_map(&self) -> MetadataMap {
        let mut metadata = MetadataMap::new();
        if let Ok(v) = MetadataValue::from_str(&self.client.user_agent) {
            let _ = metadata.insert("user-agent", v);
        }
        if let Some(user_id) = &self.client.user_id {
            if let Ok(v) = MetadataValue::from_str(user_id) {
                let _ = metadata.insert("user_id", v);
            }
        }
        for (k, v) in &self.client.metadata {
            if let Ok(v) = MetadataValue::from_str(v) {
                if let Ok(key) = tonic::metadata::MetadataKey::from_bytes(k.as_bytes()) {
                    let _ = metadata.insert(key, v);
                }
            }
        }
        metadata
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
        loop {
            match self.stream.message().await {
                Ok(Some(resp)) => {
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
                Ok(None) => self.reattach().await?,
                Err(status) => {
                    // Only reattach a still-running execution on a retriable drop.
                    if self.iter.is_completed().await || !self.retry_policy.can_retry(&status) {
                        let error = self.client.error_from_status(status).await;
                        return Err(self.fail(error));
                    }
                    self.reattach().await?;
                }
            }
        }
    }

    /// Resumes the execution from the last response read, retrying transient failures.
    async fn reattach(&mut self) -> Result<()> {
        loop {
            let mut req = Request::new(self.iter.create_reattach_request().await);
            self.attach_metadata(&mut req);
            match self.client.stub.clone().reattach_execute(req).await {
                Ok(r) => {
                    self.stream = r.into_inner();
                    return Ok(());
                }
                Err(e) => {
                    if self.retry_policy.can_retry(&e) {
                        if let Some(w) = self.retry_state.next_attempt(None) {
                            tokio::time::sleep(Duration::from_millis(w)).await;
                            continue;
                        }
                    }
                    let error = self.client.error_from_status(e).await;
                    return Err(self.fail(error));
                }
            }
        }
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

    /// Sends `request` in the background, so that reading is not delayed, retrying under the
    /// client's retry policy. Like the reference client, a failure is only logged: the server
    /// removes an execution that is never released once it has been detached for
    /// `spark.connect.execute.manager.detachedTimeout`.
    fn spawn_release(&self, request: ReleaseExecuteRequest) {
        let mut stub = self.client.stub.clone();
        let metadata = self.metadata_map();
        let policy = self.retry_policy.clone();
        let release = async move {
            let mut state = RetryPolicyState::new(policy.clone());
            loop {
                let mut req = Request::new(request.clone());
                *req.metadata_mut() = metadata.clone();
                match stub.release_execute(req).await {
                    Ok(_) => return,
                    Err(status) => {
                        if policy.can_retry(&status) {
                            if let Some(w) = state.next_attempt(None) {
                                tokio::time::sleep(Duration::from_millis(w)).await;
                                continue;
                            }
                        }
                        tracing::warn!(
                            "ReleaseExecute of operation {} failed: {}",
                            request.operation_id,
                            status
                        );
                        return;
                    }
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
        let (pending, hashes) =
            hash_with_busy_blocking_pool(vec![Vec::new(), b"abc".to_vec(), b"abc".to_vec()]);
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

    #[tokio::test]
    async fn test_artifact_hashing_reports_task_cancellation() {
        let task = tokio::spawn(std::future::pending::<()>());
        task.abort();
        let error = artifact_hashing_task_error(task.await.unwrap_err());
        assert!(error.to_string().contains("Artifact hashing task failed:"));
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

    #[tokio::test]
    async fn test_token_without_ssl_to_remote_host_is_rejected() {
        // A token over a non-loopback host without `use_ssl=true` would ship the bearer
        // credential in cleartext; the client must refuse rather than downgrade silently.
        let builder = ChannelBuilder::parse("sc://example.com/;token=SECRET").unwrap();
        match SparkConnectClient::connect(&builder).await {
            Ok(_) => panic!("expected a token over cleartext to a remote host to be rejected"),
            Err(e) => assert!(
                e.to_string().contains("use_ssl=true"),
                "expected the error to point at use_ssl=true, got: {e}"
            ),
        }
    }

    #[tokio::test]
    async fn test_token_without_ssl_to_localhost_is_allowed() {
        // Local development mirrors upstream's `local_channel_credentials()` exception:
        // a token over plaintext loopback is permitted and the header is attached.
        for url in [
            "sc://localhost/;token=SECRET",
            "sc://127.0.0.1/;token=SECRET",
            "sc://[::1]/;token=SECRET",
        ] {
            let builder = ChannelBuilder::parse(url).unwrap();
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
    async fn test_get_configs_request_structure() {
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();

        // Test that get_configs returns a Vec of Option<String>
        // Note: This will fail against a real server, but tests the structure
        let _result = client.get_configs(&["spark.sql.shuffle.partitions"]);
        // We don't assert on the result since we may not have a server
    }

    #[tokio::test]
    async fn test_set_config_request_structure() {
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();

        // Test that set_config can be called without panicking
        let _result = client.set_config("spark.sql.shuffle.partitions", "200");
    }

    #[tokio::test]
    async fn test_unset_config_request_structure() {
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();

        // Test that unset_config can be called without panicking
        let _result = client.unset_config("spark.sql.shuffle.partitions");
    }

    #[tokio::test]
    async fn test_interrupt_all_request_structure() {
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();

        // Test that interrupt_all can be called without panicking
        let _result = client.interrupt_all();
    }

    #[tokio::test]
    async fn test_interrupt_tag_request_structure() {
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();

        // Test that interrupt_tag can be called without panicking
        let _result = client.interrupt_tag("my-tag");
    }

    #[tokio::test]
    async fn test_interrupt_operation_request_structure() {
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();

        // Test that interrupt_operation can be called without panicking
        let _result = client.interrupt_operation("operation-123");
    }

    #[tokio::test]
    async fn test_release_session_request_structure() {
        let builder = ChannelBuilder::parse("sc://localhost").unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();

        // Test that release_session can be called without panicking
        let _result = client.release_session();
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
}
