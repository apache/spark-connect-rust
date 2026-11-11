// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

//! Artifact uploads on the wire: the frames of every AddArtifacts attempt, its CRCs and its
//! retries, against a loopback gRPC service that receives and acknowledges them; no Spark
//! server required.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::{stream, StreamExt};
use prost::Message;
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_core::{
    compute_crc32, ChannelBuilder, RetryPolicy, SparkConnectClient, CHUNK_SIZE,
};
use spark_connect_proto::add_artifacts_request::Payload;
use spark_connect_proto::add_artifacts_response::ArtifactSummary;
use spark_connect_proto::{AddArtifactsRequest, AddArtifactsResponse};
use tokio::net::TcpListener;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::server::{ClientStreamingService, NamedService, UnaryService};
use tonic::transport::Server;
use tonic::{Code, Request, Response, Status, Streaming};

const WAIT: Duration = Duration::from_secs(5);
const SERVICE: &str = "/spark.connect.SparkConnectService/";

/// How the server handles one AddArtifacts attempt.
#[derive(Clone, Copy)]
enum Attempt {
    /// Every frame is received, and every artifact is acknowledged with a good CRC.
    Accept,
    /// The first `n` frames are received, then the call fails with UNAVAILABLE.
    FailAfter(usize),
    /// Every frame is received, but every artifact is acknowledged with a bad CRC.
    CorruptCrc,
    /// The call fails with the code before any frame is received.
    Refuse(Code),
}

struct State {
    attempts: Mutex<VecDeque<Attempt>>,
    /// The frames that each AddArtifacts attempt received, in order of the attempts.
    received: Mutex<Vec<Vec<AddArtifactsRequest>>>,
}

struct Fixture {
    state: Arc<State>,
    server: tokio::task::JoinHandle<()>,
    address: std::net::SocketAddr,
}

impl Fixture {
    async fn start(attempts: impl IntoIterator<Item = Attempt>) -> Self {
        let state = Arc::new(State {
            attempts: Mutex::new(attempts.into_iter().collect()),
            received: Mutex::new(Vec::new()),
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let service = TestService(state.clone());
        let server = tokio::spawn(async move {
            Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        Self {
            state,
            server,
            address,
        }
    }

    async fn client(&self) -> SparkConnectClient {
        let builder = ChannelBuilder::parse(&format!("sc://{}", self.address)).unwrap();
        let mut client = tokio::time::timeout(WAIT, SparkConnectClient::connect(&builder))
            .await
            .expect("connecting timed out")
            .unwrap();
        client.set_retry_policy(RetryPolicy {
            max_retries: Some(2),
            initial_backoff_ms: 1,
            max_backoff_ms: Some(2),
            ..RetryPolicy::default()
        });
        client
    }

    /// The frames that each AddArtifacts attempt received, in order.
    fn attempts(&self) -> Vec<Vec<AddArtifactsRequest>> {
        self.state.received.lock().unwrap().clone()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

/// Bytes that differ from one position to the next, so that a misplaced chunk shows.
fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

/// The data that the frames of one upload carry, which every chunk of a chunked artifact must
/// carry in order.
fn assembled(frames: &[AddArtifactsRequest]) -> Vec<u8> {
    let mut data = Vec::new();
    for frame in frames {
        match frame.payload.as_ref().unwrap() {
            Payload::Batch(batch) => {
                for artifact in &batch.artifacts {
                    data.extend_from_slice(&artifact.data.as_ref().unwrap().data);
                }
            }
            Payload::BeginChunk(begin) => {
                data.extend_from_slice(&begin.initial_chunk.as_ref().unwrap().data)
            }
            Payload::Chunk(chunk) => data.extend_from_slice(&chunk.data),
        }
    }
    data
}

/// The data length of each frame's chunk, in order.
fn chunk_lengths(frames: &[AddArtifactsRequest]) -> Vec<usize> {
    frames
        .iter()
        .map(|frame| match frame.payload.as_ref().unwrap() {
            Payload::Batch(batch) => batch.artifacts[0].data.as_ref().unwrap().data.len(),
            Payload::BeginChunk(begin) => begin.initial_chunk.as_ref().unwrap().data.len(),
            Payload::Chunk(chunk) => chunk.data.len(),
        })
        .collect()
}

/// A temporary file of `contents`, which is removed with the directory that holds it.
fn file_with(dir: &tempfile::TempDir, name: &str, contents: &[u8]) -> String {
    let path = dir.path().join(name);
    std::fs::write(&path, contents).unwrap();
    path.to_str().unwrap().to_string()
}

#[tokio::test]
async fn a_large_file_is_uploaded_as_one_chunked_artifact() {
    let fixture = Fixture::start([]).await;
    let client = fixture.client().await;
    let dir = tempfile::tempdir().unwrap();
    let data = pattern(3 * CHUNK_SIZE + 5);
    let path = file_with(&dir, "large.bin", &data);

    client
        .add_artifacts(&[path.as_str()], false, false, true)
        .await
        .unwrap();

    let attempts = fixture.attempts();
    assert_eq!(attempts.len(), 1);
    let frames = &attempts[0];
    // A BeginChunk with the first chunk, then a Chunk for each further chunk, the last short.
    assert_eq!(
        chunk_lengths(frames),
        vec![CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, 5]
    );
    let Some(Payload::BeginChunk(begin)) = &frames[0].payload else {
        panic!("the first frame must begin the chunked artifact");
    };
    assert_eq!(begin.name, "files/large.bin");
    assert_eq!(begin.total_bytes, data.len() as i64);
    assert_eq!(begin.num_chunks, 4);
    for frame in frames.iter().skip(1) {
        assert!(matches!(frame.payload, Some(Payload::Chunk(_))));
    }
    // Every frame carries the CRC of its chunk, and the session and user context of the client,
    // and, as the frames of PySpark's ArtifactManager, no client type.
    for frame in frames {
        assert_eq!(frame.session_id, client.session_id());
        assert!(frame.user_context.is_some());
        assert_eq!(frame.client_type, None);
    }
    let crcs: Vec<i64> = frames
        .iter()
        .map(|frame| match frame.payload.as_ref().unwrap() {
            Payload::BeginChunk(begin) => begin.initial_chunk.as_ref().unwrap().crc,
            Payload::Chunk(chunk) => chunk.crc,
            Payload::Batch(_) => unreachable!(),
        })
        .collect();
    let expected: Vec<i64> = data
        .chunks(CHUNK_SIZE)
        .map(|chunk| compute_crc32(chunk) as i64)
        .collect();
    assert_eq!(crcs, expected);
    assert_eq!(assembled(frames), data);
}

#[tokio::test]
async fn a_small_file_is_one_batch_named_by_its_extension() {
    let fixture = Fixture::start([]).await;
    let client = fixture.client().await;
    let dir = tempfile::tempdir().unwrap();
    let class = file_with(&dir, "A.class", b"cafebabe");

    // A class file is uploaded by the Rust API, as classes/.
    client
        .add_artifacts(&[class.as_str()], false, false, false)
        .await
        .unwrap();

    let attempts = fixture.attempts();
    assert_eq!(attempts.len(), 1);
    let [frame] = attempts[0].as_slice() else {
        panic!("expected one frame");
    };
    let Some(Payload::Batch(batch)) = &frame.payload else {
        panic!("expected a batch");
    };
    let [artifact] = batch.artifacts.as_slice() else {
        panic!("expected one artifact");
    };
    assert_eq!(artifact.name, "classes/A.class");
    let chunk = artifact.data.as_ref().unwrap();
    assert_eq!(chunk.data.as_ref(), b"cafebabe");
    assert_eq!(chunk.crc, compute_crc32(b"cafebabe") as i64);
}

#[tokio::test]
async fn a_transient_failure_retries_the_whole_upload_from_its_start() {
    // The first attempt receives two frames and then fails; the second receives the upload.
    let fixture = Fixture::start([Attempt::FailAfter(2), Attempt::Accept]).await;
    let client = fixture.client().await;
    let dir = tempfile::tempdir().unwrap();
    let data = pattern(3 * CHUNK_SIZE + 5);
    let path = file_with(&dir, "large.bin", &data);

    client
        .add_artifacts(&[path.as_str()], false, false, true)
        .await
        .unwrap();

    let attempts = fixture.attempts();
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].len(), 2);
    let whole = &attempts[1];
    assert!(matches!(whole[0].payload, Some(Payload::BeginChunk(_))));
    assert_eq!(
        chunk_lengths(whole),
        vec![CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE, 5]
    );
    assert_eq!(assembled(whole), data);
    // The frames of the retried attempt are those of the first, from the start.
    assert_eq!(&whole[..2], attempts[0].as_slice());
}

#[tokio::test]
async fn a_crc_mismatch_is_reported_and_not_retried() {
    let fixture = Fixture::start([Attempt::CorruptCrc]).await;
    let client = fixture.client().await;
    let dir = tempfile::tempdir().unwrap();
    let path = file_with(&dir, "small.txt", b"small");

    let error = client
        .add_artifacts(&[path.as_str()], false, false, true)
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("CRC check failed for artifact: files/small.txt"),
        "{error}"
    );
    assert_eq!(fixture.attempts().len(), 1);
}

#[tokio::test]
async fn a_failure_that_is_not_transient_is_not_retried() {
    let fixture = Fixture::start([Attempt::Refuse(Code::PermissionDenied)]).await;
    let client = fixture.client().await;
    let dir = tempfile::tempdir().unwrap();
    let path = file_with(&dir, "small.txt", b"small");

    assert!(client
        .add_artifacts(&[path.as_str()], false, false, true)
        .await
        .is_err());
    // The refused attempt received no frame, and there was no second attempt.
    assert_eq!(fixture.attempts(), vec![Vec::<AddArtifactsRequest>::new()]);
}

#[tokio::test]
async fn a_missing_file_fails_before_any_attempt() {
    let fixture = Fixture::start([]).await;
    let client = fixture.client().await;
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing.jar");

    assert!(client
        .add_artifacts(&[missing.to_str().unwrap()], false, false, false)
        .await
        .is_err());
    assert!(fixture.attempts().is_empty());
}

/// The AddArtifacts service, which receives each attempt's frames and acknowledges them.
#[derive(Clone)]
struct TestService(Arc<State>);

impl NamedService for TestService {
    const NAME: &'static str = "spark.connect.SparkConnectService";
}

impl Service<http::Request<Body>> for TestService {
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<Body>) -> Self::Future {
        let method = request.uri().path().trim_start_matches(SERVICE).to_string();
        let state = self.0.clone();
        Box::pin(async move {
            let mut grpc = tonic::server::Grpc::new(BytesCodec);
            Ok(match method.as_str() {
                "AddArtifacts" => grpc.client_streaming(AddArtifacts(state), request).await,
                _ => grpc.unary(Unary, request).await,
            })
        })
    }
}

struct AddArtifacts(Arc<State>);

impl ClientStreamingService<Vec<u8>> for AddArtifacts {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, request: Request<Streaming<Vec<u8>>>) -> Self::Future {
        let state = self.0.clone();
        Box::pin(async move {
            let attempt = state
                .attempts
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(Attempt::Accept);
            if let Attempt::Refuse(code) = attempt {
                state.received.lock().unwrap().push(Vec::new());
                return Err(Status::new(code, "refused"));
            }
            let limit = match attempt {
                Attempt::FailAfter(n) => n,
                _ => usize::MAX,
            };
            let mut frames = request.into_inner();
            let mut received = Vec::new();
            while received.len() < limit {
                let Some(bytes) = frames.next().await.transpose()? else {
                    break;
                };
                received.push(AddArtifactsRequest::decode(bytes.as_slice()).unwrap());
            }
            let complete = received.len() < limit;
            let response = acknowledge(&received, matches!(attempt, Attempt::CorruptCrc));
            state.received.lock().unwrap().push(received);
            if !complete {
                return Err(Status::unavailable("connection reset"));
            }
            Ok(Response::new(response.encode_to_vec()))
        })
    }
}

/// The response for frames: each artifact in order, with whether every one of its CRCs is good
/// (and not corrupted, when `corrupt`).
fn acknowledge(frames: &[AddArtifactsRequest], corrupt: bool) -> AddArtifactsResponse {
    /// Records one CRC of the artifact named `name`, and returns the artifact's index.
    fn record(artifacts: &mut Vec<(String, bool)>, name: &str, good: bool) -> usize {
        let index = match artifacts.iter().position(|(n, _)| n == name) {
            Some(index) => index,
            None => {
                artifacts.push((name.to_string(), true));
                artifacts.len() - 1
            }
        };
        artifacts[index].1 &= good;
        index
    }
    let mut artifacts: Vec<(String, bool)> = Vec::new();
    let mut current: Option<usize> = None;
    for frame in frames {
        match frame.payload.as_ref().unwrap() {
            Payload::Batch(batch) => {
                for artifact in &batch.artifacts {
                    let chunk = artifact.data.as_ref().unwrap();
                    let good = compute_crc32(&chunk.data) as i64 == chunk.crc;
                    record(&mut artifacts, &artifact.name, good);
                }
            }
            Payload::BeginChunk(begin) => {
                let chunk = begin.initial_chunk.as_ref().unwrap();
                let good = compute_crc32(&chunk.data) as i64 == chunk.crc;
                current = Some(record(&mut artifacts, &begin.name, good));
            }
            Payload::Chunk(chunk) => {
                let good = compute_crc32(&chunk.data) as i64 == chunk.crc;
                let index = current.expect("a chunk follows its BeginChunk");
                artifacts[index].1 &= good;
            }
        }
    }
    AddArtifactsResponse {
        // The server echoes the session ID of the request.
        session_id: frames
            .first()
            .map(|f| f.session_id.clone())
            .unwrap_or_default(),
        artifacts: artifacts
            .into_iter()
            .map(|(name, good)| ArtifactSummary {
                name,
                is_crc_successful: good && !corrupt,
            })
            .collect(),
        ..Default::default()
    }
}

/// Every other method is answered with its empty response message.
#[derive(Clone)]
struct Unary;

impl UnaryService<Vec<u8>> for Unary {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, _: Request<Vec<u8>>) -> Self::Future {
        Box::pin(async { Ok(Response::new(Vec::new())) })
    }
}
