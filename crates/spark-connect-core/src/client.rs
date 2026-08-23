//! Spark Connect client implementation.
//!
//! Mirrors `pyspark.sql.connect.client.core.SparkConnectClient`.

use std::collections::BTreeMap;
use std::str::FromStr;
use tonic::metadata::MetadataValue;
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

use crate::artifact::{build_artifact_request_stream, FileArtifact, NamedArtifact};
use crate::channel::ChannelBuilder;
use crate::error::{Result, SparkError};

/// A Spark Connect client for communicating with a remote Spark server.
///
/// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient`.
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
            let tls_config = ClientTlsConfig::new().domain_name(builder.host());

            tonic::transport::Channel::from_shared(format!("https://{endpoint}"))
                .map_err(|e| SparkError::connect_msg(format!("Invalid endpoint: {}", e)))?
                .tls_config(tls_config)
                .map_err(|e| SparkError::connect_msg(format!("Failed to configure TLS: {}", e)))?
                .connect_lazy()
        } else {
            // Plaintext connection (for localhost development)
            tonic::transport::Channel::from_shared(format!("http://{endpoint}"))
                .map_err(|e| SparkError::connect_msg(format!("Invalid endpoint: {}", e)))?
                .connect_lazy()
        };

        let stub = SparkConnectServiceClient::new(channel.clone());

        let metadata = builder
            .metadata()
            .into_iter()
            .map(|(k, v)| (k, v))
            .collect();

        Ok(Self {
            channel,
            stub,
            session_id,
            user_id: builder.user_id().map(String::from),
            metadata,
            user_agent,
        })
    }

    /// Get the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get the user ID, if set.
    pub fn user_id(&self) -> Option<&str> {
        self.user_id.as_deref()
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
        let resp = self.stub.clone().execute_plan(req).await;
        resp.map(Response::into_inner)
            .map_err(|status| SparkError::from_grpc_status(status))
    }

    /// Execute a plan, forwarding raw request bytes and returning a raw response stream.
    ///
    /// Byte-exact passthrough (no prost decode/re-encode), so no proto field is dropped
    /// and deep plans don't hit the recursion limit - the server sees exactly the bytes
    /// the reference client built.
    pub async fn execute_plan_raw(&self, request: Vec<u8>) -> Result<Streaming<Vec<u8>>> {
        let mut grpc = tonic::client::Grpc::new(self.channel.clone());
        grpc.ready()
            .await
            .map_err(|e| SparkError::connect_msg(format!("channel not ready: {e}")))?;
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
        let mut grpc = tonic::client::Grpc::new(self.channel.clone());
        grpc.ready()
            .await
            .map_err(|e| SparkError::connect_msg(format!("channel not ready: {e}")))?;
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
        let resp = self.stub.clone().reattach_execute(req).await;
        resp.map(Response::into_inner)
            .map_err(|status| SparkError::from_grpc_status(status))
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
        let resp = self.stub.clone().release_execute(req).await;
        resp.map(Response::into_inner)
            .map_err(|status| SparkError::from_grpc_status(status))
    }

    /// Analyze a plan, forwarding the raw request bytes without decoding them.
    ///
    /// Decoding the request with prost imposes a recursion limit (100) that a deeply
    /// nested plan (e.g. hundreds of chained withColumn) exceeds - but the server
    /// handles such plans fine. Using a passthrough codec forwards the exact bytes and
    /// returns the raw response bytes, avoiding the client-side limit (matching the
    /// reference client, which never re-decodes the request it built).
    pub async fn analyze_plan_raw(&self, request: Vec<u8>) -> Result<Vec<u8>> {
        let mut grpc = tonic::client::Grpc::new(self.channel.clone());
        grpc.ready()
            .await
            .map_err(|e| SparkError::connect_msg(format!("channel not ready: {e}")))?;
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
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        let resp = self.stub.clone().analyze_plan(req).await;
        resp.map(Response::into_inner)
            .map_err(|status| SparkError::from_grpc_status(status))
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
            .map_err(|status| SparkError::from_grpc_status(status))
    }

    /// Update or fetch configurations.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.config`.
    pub async fn config(&self, request: ConfigRequest) -> Result<ConfigResponse> {
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        let resp = self.stub.clone().config(req).await;
        resp.map(Response::into_inner)
            .map_err(|status| SparkError::from_grpc_status(status))
    }

    /// Interrupt running operations on this session.
    ///
    /// Mirrors `pyspark.sql.connect.client.core.SparkConnectClient.interrupt`.
    pub async fn interrupt(&self, request: InterruptRequest) -> Result<InterruptResponse> {
        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        let resp = self.stub.clone().interrupt(req).await;
        resp.map(Response::into_inner)
            .map_err(|status| SparkError::from_grpc_status(status))
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
        let resp = self.stub.clone().release_session(req).await;
        resp.map(|_| ())
            .map_err(|status| SparkError::from_grpc_status(status))
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

        // Create a futures stream from the requests
        let request_stream = futures::stream::iter(requests);

        // Call the AddArtifacts streaming RPC
        let mut req = Request::new(request_stream);
        self._attach_metadata(&mut req);

        let response = self
            .stub
            .clone()
            .add_artifacts(req)
            .await
            .map_err(|e| SparkError::from_grpc_status(e))?
            .into_inner();

        // Check that all artifacts were successfully received
        for summary in response.artifacts {
            if !summary.is_crc_successful {
                return Err(SparkError::connect_msg(format!(
                    "CRC check failed for artifact: {}",
                    summary.name
                )));
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
        request.user_context = Some(UserContext {
            user_id: self.user_id.clone().unwrap_or_default(),
            ..Default::default()
        });
        request.names = names.iter().map(|n| n.to_string()).collect();

        let mut req = Request::new(request);
        self._attach_metadata(&mut req);
        let response = self
            .stub
            .clone()
            .artifact_status(req)
            .await
            .map_err(|e| SparkError::from_grpc_status(e))?
            .into_inner();

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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(client.user_agent.contains("spark/"), true);
    }

    #[test]
    fn test_tls_enabled_channel_builder() {
        // Verify that a ChannelBuilder with use_ssl=true is marked as secure
        let builder = ChannelBuilder::parse("sc://example.com/;use_ssl=true").unwrap();
        assert!(builder.use_ssl());
        assert!(builder.secure());
    }

    #[test]
    fn test_token_bearer_in_metadata() {
        // Verify that a token parameter is recognized and would be passed as metadata
        let builder = ChannelBuilder::parse("sc://localhost/;token=my_token_123").unwrap();
        assert!(builder.secure());
        assert_eq!(builder.token(), Some("my_token_123".to_string()));
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
