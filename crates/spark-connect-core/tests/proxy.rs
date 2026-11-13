// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! The channel through the HTTP CONNECT proxy of the environment, as grpcio's channel of
//! PySpark: a loopback Spark Connect service behind a loopback CONNECT proxy, which records
//! what it is asked. The proxy variables are of the process, so the tests take turns.

mod common;

use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{FakeServer, FakeService};
use spark_connect_core::{ChannelBuilder, RetryPolicy, SparkConnectClient};
use spark_connect_proto::{ConfigRequest, ConfigResponse};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tonic::codegen::BoxFuture;
use tonic::server::UnaryService;
use tonic::{Code, Request, Response, Status};
use tonic_prost::ProstCodec;

const WAIT: Duration = Duration::from_secs(20);
const SESSION: &str = "00000000-0000-4000-8000-000000000001";
const TOKEN: &str = "proxy-fixture-token";

/// Every variable that grpcio or curl reads for a proxy, so that a test sees only its own.
const PROXY_VARIABLES: [&str; 10] = [
    "grpc_proxy",
    "https_proxy",
    "http_proxy",
    "no_grpc_proxy",
    "no_proxy",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "all_proxy",
];

static ENVIRONMENT: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn bounded<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(WAIT, future)
        .await
        .expect("test waited past the bound")
}

/// Runs `test` with exactly the proxy variables `vars` set.
async fn with_proxy_env<T>(vars: &[(&str, &str)], test: impl Future<Output = T>) -> T {
    let _turn = ENVIRONMENT.lock().await;
    for name in PROXY_VARIABLES {
        std::env::remove_var(name);
    }
    for (name, value) in vars {
        std::env::set_var(name, value);
    }
    let result = test.await;
    for name in PROXY_VARIABLES {
        std::env::remove_var(name);
    }
    result
}

/// Config, answered with the session of the request.
struct Config;

impl UnaryService<ConfigRequest> for Config {
    type Response = ConfigResponse;
    type Future = BoxFuture<Response<ConfigResponse>, Status>;

    fn call(&mut self, request: Request<ConfigRequest>) -> Self::Future {
        let session_id = request.into_inner().session_id;
        Box::pin(async move {
            Ok(Response::new(ConfigResponse {
                session_id,
                server_side_session_id: "server-session".into(),
                ..Default::default()
            }))
        })
    }
}

async fn spark_server() -> FakeServer {
    FakeService::new()
        .unary("Config", ProstCodec::default, |_| Config)
        .serve()
        .await
}

/// What the proxy does with a CONNECT.
#[derive(Clone)]
enum Behavior {
    /// Opens the tunnel to this address, whatever host the CONNECT names.
    Tunnel(SocketAddr),
    /// Answers with this status line, without a tunnel.
    Refuse(&'static str),
    /// Opens the tunnel, keeps the first bytes the client sends in it, and closes.
    Capture,
    /// Reads the CONNECT and never answers.
    Silent,
    /// Opens the tunnel to this address, and sends the first bytes of the server (a plaintext
    /// HTTP/2 server speaks first) in the same write as its own response.
    TunnelCoalesced(SocketAddr),
    /// Opens the tunnel to this address with this status line.
    TunnelWithStatus(SocketAddr, &'static str),
}

/// One CONNECT the proxy was asked.
#[derive(Clone, Debug, Default)]
struct Connect {
    request_line: String,
    headers: Vec<(String, String)>,
    captured: Vec<u8>,
}

impl Connect {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

/// A loopback HTTP CONNECT proxy that records every connection it accepts.
struct Proxy {
    address: SocketAddr,
    connects: Arc<Mutex<Vec<Connect>>>,
    accepted: Arc<Mutex<usize>>,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Proxy {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl Proxy {
    async fn start(behavior: Behavior) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let connects = Arc::new(Mutex::new(Vec::new()));
        let accepted = Arc::new(Mutex::new(0));
        let (seen, count) = (connects.clone(), accepted.clone());
        let task = tokio::spawn(async move {
            loop {
                let (socket, _) = listener.accept().await.unwrap();
                *count.lock().unwrap() += 1;
                tokio::spawn(serve_connect(socket, behavior.clone(), seen.clone()));
            }
        });
        Self {
            address,
            connects,
            accepted,
            task,
        }
    }

    fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    fn connects(&self) -> Vec<Connect> {
        self.connects.lock().unwrap().clone()
    }

    fn accepted(&self) -> usize {
        *self.accepted.lock().unwrap()
    }
}

async fn serve_connect(mut client: TcpStream, behavior: Behavior, seen: Arc<Mutex<Vec<Connect>>>) {
    let mut head = Vec::new();
    while !head.ends_with(b"\r\n\r\n") {
        let mut byte = [0u8; 1];
        if client.read(&mut byte).await.unwrap_or(0) == 0 {
            return;
        }
        head.push(byte[0]);
    }
    let text = String::from_utf8_lossy(&head).into_owned();
    let mut lines = text.split("\r\n");
    let mut connect = Connect {
        request_line: lines.next().unwrap_or_default().to_string(),
        ..Default::default()
    };
    connect.headers = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim().to_string(), value.trim().to_string()))
        .collect();
    match behavior {
        Behavior::Silent => {
            seen.lock().unwrap().push(connect);
            std::future::pending::<()>().await;
        }
        Behavior::Refuse(status) => {
            seen.lock().unwrap().push(connect);
            let _ = client
                .write_all(format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\n\r\n").as_bytes())
                .await;
        }
        Behavior::TunnelCoalesced(upstream) => {
            seen.lock().unwrap().push(connect);
            let mut upstream = TcpStream::connect(upstream).await.unwrap();
            let mut greeting = vec![0u8; 4096];
            let read = upstream.read(&mut greeting).await.unwrap();
            assert!(read > 0, "the server sends its SETTINGS first");
            let mut answer = b"HTTP/1.1 200 Connection Established\r\n\r\n".to_vec();
            answer.extend_from_slice(&greeting[..read]);
            client.write_all(&answer).await.unwrap();
            let _ = tokio::io::copy_bidirectional(&mut client, &mut upstream).await;
        }
        Behavior::Tunnel(upstream) | Behavior::TunnelWithStatus(upstream, _) => {
            let status = match behavior {
                Behavior::TunnelWithStatus(_, status) => status,
                _ => "200 Connection Established",
            };
            seen.lock().unwrap().push(connect);
            let mut upstream = TcpStream::connect(upstream).await.unwrap();
            client
                .write_all(format!("HTTP/1.1 {status}\r\n\r\n").as_bytes())
                .await
                .unwrap();
            let _ = tokio::io::copy_bidirectional(&mut client, &mut upstream).await;
        }
        Behavior::Capture => {
            client
                .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                .await
                .unwrap();
            let mut buffer = vec![0u8; 4096];
            let read = tokio::time::timeout(Duration::from_secs(5), client.read(&mut buffer))
                .await
                .ok()
                .and_then(Result::ok)
                .unwrap_or(0);
            connect.captured = buffer[..read].to_vec();
            seen.lock().unwrap().push(connect);
        }
    }
}

fn quick_retries() -> RetryPolicy {
    RetryPolicy {
        max_retries: Some(1),
        initial_backoff_ms: 1,
        max_backoff_ms: Some(2),
        ..RetryPolicy::default()
    }
}

async fn client(url: &str) -> SparkConnectClient {
    let builder = ChannelBuilder::parse(url).unwrap();
    let mut client = SparkConnectClient::connect(&builder).await.unwrap();
    client.set_retry_policy(quick_retries());
    client
}

fn config() -> ConfigRequest {
    ConfigRequest {
        session_id: SESSION.into(),
        ..Default::default()
    }
}

#[tokio::test]
async fn plaintext_channel_goes_through_the_connect_tunnel() {
    let server = spark_server().await;
    let proxy = Proxy::start(Behavior::Tunnel(server.address)).await;
    let port = server.address.port();
    with_proxy_env(&[("https_proxy", &proxy.url())], async {
        // A host that only the proxy can reach: the client must not resolve it itself.
        let client = client(&format!("sc://spark.invalid:{port}/;session_id={SESSION}")).await;
        bounded(client.config(config())).await.unwrap();
    })
    .await;
    let connects = proxy.connects();
    assert_eq!(connects.len(), 1, "{connects:?}");
    assert_eq!(
        connects[0].request_line,
        format!("CONNECT spark.invalid:{port} HTTP/1.1")
    );
    assert_eq!(connects[0].header("proxy-authorization"), None);
}

#[tokio::test]
async fn server_bytes_in_the_proxy_response_read_are_kept() {
    // The tunnel reads the response of the proxy and no more: the bytes after it are the
    // server's, whatever reads the proxy and the server's first bytes arrive in.
    let server = spark_server().await;
    let proxy = Proxy::start(Behavior::TunnelCoalesced(server.address)).await;
    let port = server.address.port();
    with_proxy_env(&[("https_proxy", &proxy.url())], async {
        let mut client = client(&format!("sc://spark.invalid:{port}/;session_id={SESSION}")).await;
        client.set_retry_policy(RetryPolicy {
            max_retries: Some(0),
            ..RetryPolicy::default()
        });
        bounded(client.config(config())).await.unwrap();
    })
    .await;
    assert_eq!(proxy.connects().len(), 1);
}

#[tokio::test]
async fn a_proxy_answer_of_any_2xx_opens_the_tunnel() {
    // As grpcio, which takes any 2xx, and RFC 9110.
    let server = spark_server().await;
    let port = server.address.port();
    for status in ["204 No Content", "299 Custom"] {
        let proxy = Proxy::start(Behavior::TunnelWithStatus(server.address, status)).await;
        with_proxy_env(&[("https_proxy", &proxy.url())], async {
            let client = client(&format!("sc://spark.invalid:{port}/;session_id={SESSION}")).await;
            bounded(client.config(config())).await.unwrap();
        })
        .await;
    }
}

#[tokio::test]
async fn tls_is_negotiated_with_the_server_inside_the_tunnel() {
    let proxy = Proxy::start(Behavior::Capture).await;
    let error = with_proxy_env(&[("https_proxy", &proxy.url())], async {
        let client = client(&format!(
            "sc://spark.invalid:443/;token={TOKEN};session_id={SESSION}"
        ))
        .await;
        bounded(client.config(config())).await.unwrap_err()
    })
    .await;
    // The proxy closes the tunnel after the ClientHello, which is UNAVAILABLE as any
    // connection that closes.
    assert_eq!(error.grpc_code, Some(Code::Unavailable as i32), "{error:?}");
    let connects = proxy.connects();
    assert!(!connects.is_empty());
    for connect in connects {
        assert_eq!(connect.request_line, "CONNECT spark.invalid:443 HTTP/1.1");
        let hello = &connect.captured;
        // A TLS handshake record, which names the Spark server, not the proxy.
        assert_eq!(hello.first(), Some(&0x16), "{hello:?}");
        assert!(contains(hello, b"spark.invalid"), "no SNI in {hello:?}");
        // The proxy never sees the token: it is sent only inside the TLS session.
        assert!(!contains(hello, TOKEN.as_bytes()));
    }
}

#[tokio::test]
async fn proxy_credentials_are_sent_as_basic_authorization() {
    let server = spark_server().await;
    let proxy = Proxy::start(Behavior::Tunnel(server.address)).await;
    let port = server.address.port();
    let url = format!("http://alice:p%40ss@{}", proxy.address);
    with_proxy_env(&[("grpc_proxy", &url)], async {
        let client = client(&format!("sc://spark.invalid:{port}/;session_id={SESSION}")).await;
        bounded(client.config(config())).await.unwrap();
    })
    .await;
    let connects = proxy.connects();
    assert_eq!(
        connects[0].header("proxy-authorization"),
        Some("Basic YWxpY2U6cEBzcw==") // alice:p@ss
    );
}

#[tokio::test]
async fn a_loopback_host_is_never_proxied() {
    // A token goes to localhost in plaintext, so a proxy must not carry it.
    let server = spark_server().await;
    let proxy = Proxy::start(Behavior::Tunnel(server.address)).await;
    let port = server.address.port();
    with_proxy_env(
        &[("grpc_proxy", &proxy.url()), ("https_proxy", &proxy.url())],
        async {
            for host in ["localhost", "127.0.0.1"] {
                let client = client(&format!("sc://{host}:{port}/;session_id={SESSION}")).await;
                bounded(client.config(config())).await.unwrap();
            }
            let client = client(&format!(
                "sc://localhost:{port}/;token={TOKEN};session_id={SESSION}"
            ))
            .await;
            bounded(client.config(config())).await.unwrap();
        },
    )
    .await;
    assert_eq!(proxy.accepted(), 0);
}

#[tokio::test]
async fn a_host_of_no_proxy_connects_directly() {
    let proxy = Proxy::start(Behavior::Refuse("403 Forbidden")).await;
    with_proxy_env(
        &[
            ("https_proxy", &proxy.url()),
            ("no_proxy", "other, .invalid"),
        ],
        async {
            let client = client(&format!("sc://spark.invalid:15002/;session_id={SESSION}")).await;
            // Directly, the host does not resolve.
            bounded(client.config(config())).await.unwrap_err();
        },
    )
    .await;
    assert_eq!(proxy.accepted(), 0);
}

#[tokio::test]
async fn a_proxy_that_refuses_the_tunnel_is_unavailable() {
    for status in [
        "407 Proxy Authentication Required",
        "403 Forbidden",
        "502 Bad Gateway",
    ] {
        let proxy = Proxy::start(Behavior::Refuse(status)).await;
        let error = with_proxy_env(&[("https_proxy", &proxy.url())], async {
            let client = client(&format!("sc://spark.invalid:15002/;session_id={SESSION}")).await;
            bounded(client.config(config())).await.unwrap_err()
        })
        .await;
        assert_eq!(
            error.grpc_code,
            Some(Code::Unavailable as i32),
            "{status}: {error:?}"
        );
        assert!(!proxy.connects().is_empty(), "{status}");
    }
}

#[tokio::test]
async fn a_proxy_that_is_down_is_unavailable() {
    // A port that nothing listens on.
    let address = TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap();
    let error = with_proxy_env(&[("https_proxy", &format!("http://{address}"))], async {
        let client = client(&format!("sc://spark.invalid:15002/;session_id={SESSION}")).await;
        bounded(client.config(config())).await.unwrap_err()
    })
    .await;
    assert_eq!(error.grpc_code, Some(Code::Unavailable as i32), "{error:?}");
}

#[tokio::test]
async fn a_proxy_that_never_answers_is_unavailable() {
    // As grpcio, which retries it: UNAVAILABLE once the tunnel is not open in time, rather than
    // the UNKNOWN that a connect timeout around the whole connection would raise.
    let proxy = Proxy::start(Behavior::Silent).await;
    let error = with_proxy_env(&[("https_proxy", &proxy.url())], async {
        let mut client = client(&format!(
            "sc://spark.invalid:443/;token={TOKEN};session_id={SESSION}"
        ))
        .await;
        client.set_retry_policy(RetryPolicy {
            max_retries: Some(0),
            ..RetryPolicy::default()
        });
        tokio::time::timeout(Duration::from_secs(30), client.config(config()))
            .await
            .expect("the tunnel is bounded")
            .unwrap_err()
    })
    .await;
    assert_eq!(error.grpc_code, Some(Code::Unavailable as i32), "{error:?}");
    assert!(
        error.to_string().contains("did not open the tunnel"),
        "{error}"
    );
    assert_eq!(proxy.connects().len(), 1);
}

#[tokio::test]
async fn an_unsupported_proxy_scheme_connects_directly() {
    let proxy = Proxy::start(Behavior::Refuse("403 Forbidden")).await;
    let server = spark_server().await;
    let port = server.address.port();
    for scheme in ["https", "socks5"] {
        with_proxy_env(
            &[("https_proxy", &format!("{scheme}://{}", proxy.address))],
            async {
                let client =
                    client(&format!("sc://spark.invalid:{port}/;session_id={SESSION}")).await;
                bounded(client.config(config())).await.unwrap_err();
            },
        )
        .await;
    }
    assert_eq!(proxy.accepted(), 0);
}

#[tokio::test]
async fn an_invalid_proxy_address_fails_the_connection() {
    let error = with_proxy_env(
        &[("https_proxy", "http://alice:hunter2@proxy:99999")],
        async {
            let builder = ChannelBuilder::parse("sc://spark.invalid:15002").unwrap();
            match SparkConnectClient::connect(&builder).await {
                Ok(_) => panic!("connected with an invalid proxy address"),
                Err(error) => error,
            }
        },
    )
    .await;
    let message = error.to_string();
    assert!(
        message.contains("Invalid HTTP proxy address in https_proxy"),
        "{message}"
    );
    assert!(!message.contains("hunter2"), "{message}");
}

#[tokio::test]
async fn uppercase_variables_are_not_read() {
    // As grpcio on POSIX, which reads only the lowercase names.
    let server = spark_server().await;
    let proxy = Proxy::start(Behavior::Tunnel(server.address)).await;
    with_proxy_env(
        &[("HTTPS_PROXY", &proxy.url()), ("HTTP_PROXY", &proxy.url())],
        async {
            let client = client(&format!("sc://spark.invalid:15002/;session_id={SESSION}")).await;
            bounded(client.config(config())).await.unwrap_err();
        },
    )
    .await;
    assert_eq!(proxy.accepted(), 0);
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}
