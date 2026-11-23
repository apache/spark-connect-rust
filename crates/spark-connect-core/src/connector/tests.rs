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

//! The connections of the connector, against loopback servers of the tests: the order and the
//! deadlines of the attempts to the addresses of a host, the connectivity state of a channel, the
//! preface and the settings that the connector exchanges for hyper, and the texts of the failures,
//! as grpcio gives them in its dialect and as the Rust client does otherwise.

use super::*;
use std::sync::atomic::AtomicUsize;
use tokio::net::TcpListener;

/// A certificate of the tests and the PKCS #8 DER of its key.
struct Issued {
    cert: rustls::pki_types::CertificateDer<'static>,
    key: Vec<u8>,
}

/// The certificates of the TLS tests, of P-256 keys, made once as the tests run, so that no key
/// is kept in the source: a CA, a certificate of `localhost` and `127.0.0.1` and one of
/// `other.test` that it signed, a self-signed one of `localhost`, and one of `localhost` that the
/// CA signed and that expired in 2020.
struct Certs {
    ca: rustls::pki_types::CertificateDer<'static>,
    localhost: Issued,
    other: Issued,
    /// Of `*.localhost`.
    wildcard: Issued,
    self_signed: Issued,
    expired: Issued,
}

/// The certificates of the TLS tests (see [`Certs`]).
fn certs() -> &'static Certs {
    use rcgen::{
        BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer,
        KeyPair, KeyUsagePurpose,
    };
    static CERTS: std::sync::OnceLock<Certs> = std::sync::OnceLock::new();
    CERTS.get_or_init(|| {
        // The parameters of a certificate of a server of the names `names`.
        let server = |names: &[&str]| {
            let names: Vec<String> = names.iter().map(|name| name.to_string()).collect();
            let mut params = CertificateParams::new(names.clone()).unwrap();
            params
                .distinguished_name
                .push(DnType::CommonName, names[0].as_str());
            params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
            params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
            params
        };
        let mut ca = CertificateParams::new(Vec::<String>::new()).unwrap();
        ca.distinguished_name
            .push(DnType::CommonName, "connector test CA");
        ca.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        ca.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        let ca_key = KeyPair::generate().unwrap();
        let ca_cert = ca.self_signed(&ca_key).unwrap();
        let issuer = Issuer::from_params(&ca, &ca_key);
        let signed = |params: CertificateParams| {
            let key = KeyPair::generate().unwrap();
            let cert = params.signed_by(&key, &issuer).unwrap();
            Issued {
                cert: cert.der().clone(),
                key: key.serialize_der(),
            }
        };
        let self_signed = {
            let key = KeyPair::generate().unwrap();
            let cert = server(&["localhost"]).self_signed(&key).unwrap();
            Issued {
                cert: cert.der().clone(),
                key: key.serialize_der(),
            }
        };
        let mut expired = server(&["localhost"]);
        expired.not_before = rcgen::date_time_ymd(2020, 1, 1);
        expired.not_after = rcgen::date_time_ymd(2020, 1, 2);
        Certs {
            ca: ca_cert.der().clone(),
            localhost: signed(server(&["localhost", "127.0.0.1"])),
            other: signed(server(&["other.test"])),
            wildcard: signed(server(&["*.localhost"])),
            self_signed,
            expired: signed(expired),
        }
    })
}

/// The PEM of the certificate `der`, as the roots of a channel are given.
fn pem(der: &[u8]) -> String {
    use base64::Engine;
    let text = base64::engine::general_purpose::STANDARD.encode(der);
    let mut pem = String::from("-----BEGIN CERTIFICATE-----\n");
    for line in text.as_bytes().chunks(64) {
        pem.push_str(std::str::from_utf8(line).unwrap());
        pem.push('\n');
    }
    pem.push_str("-----END CERTIFICATE-----\n");
    pem
}

/// A target of the addresses `addresses`, resolved as they are, in the dialect of grpcio if
/// `grpcio`.
fn target(addresses: Vec<SocketAddr>, grpcio: bool) -> Target {
    Target {
        host: "localhost".to_string(),
        port: addresses[0].port(),
        tunnel: None,
        tls: None,
        local: false,
        grpcio,
        form: None,
        names: None,
        resolved: Default::default(),
        addresses: Some(addresses),
        resolves_never: false,
        dns_config: None,
        lookups: Default::default(),
        name_lookups: Default::default(),
    }
}

/// A port of the loopback that nothing listens on, reserved while it lives: of a socket that is
/// bound to it, with `SO_REUSEADDR`, and never listens, so that a connection to it is refused at
/// once, and no listener of another test takes the port meanwhile, as a port that is freed goes to
/// the next `bind` of port 0.
struct ClosedPort {
    socket: tokio::net::TcpSocket,
    address: SocketAddr,
}

impl ClosedPort {
    /// A listener on the port from now on, of the socket that reserved it, so that no other test
    /// takes the port in between.
    fn listen(self) -> TcpListener {
        self.socket.listen(1024).unwrap()
    }
}

/// A port of the loopback that nothing listens on (see [`ClosedPort`]).
fn closed_port() -> ClosedPort {
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    socket.set_reuseaddr(true).unwrap();
    socket.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let address = socket.local_addr().unwrap();
    ClosedPort { socket, address }
}

/// A server on `listener` that does `serve` with each connection it accepts, and the count of
/// the connections it accepted.
fn serve<F, Fut>(listener: TcpListener, serve: F) -> Arc<AtomicUsize>
where
    F: Fn(TcpStream) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let accepted = Arc::new(AtomicUsize::new(0));
    let count = accepted.clone();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            count.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(serve(stream));
        }
    });
    accepted
}

/// A server that accepts connections and never answers.
async fn silent_server() -> (SocketAddr, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let accepted = serve(listener, |stream| async move {
        let _held = stream;
        std::future::pending::<()>().await;
    });
    (address, accepted)
}

/// A server of HTTP/2 that sends its settings at once on each connection, which it then holds,
/// on `listener`.
fn settings_server_on(listener: TcpListener) -> Arc<AtomicUsize> {
    serve(listener, |mut stream| async move {
        let _ = stream.write_all(&EMPTY_SETTINGS).await;
        let mut buffer = [0u8; 1024];
        while matches!(stream.read(&mut buffer).await, Ok(read) if read > 0) {}
    })
}

async fn settings_server() -> (SocketAddr, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    (address, settings_server_on(listener))
}

fn v4(port: u16) -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], port))
}

fn v6(port: u16) -> SocketAddr {
    SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, port))
}

/// An I/O failure of the OS error `errno`.
fn os_error(errno: i32) -> IoFailure {
    IoFailure::of(&std::io::Error::from_raw_os_error(errno))
}

#[test]
fn the_failures_have_grpc_core_s_texts_in_grpcio_s_dialect_and_the_rust_client_s_otherwise() {
    // The texts of grpcio 1.85.1 for each failure, of the probes against loopback servers.
    let refused = os_error(libc::ECONNREFUSED);
    let reset = os_error(libc::ECONNRESET);
    let at = |address: SocketAddr, failure| ChannelFailure::Addresses {
        address: address.into(),
        failure,
    };
    let all = "failed to connect to all addresses; last error:";
    let cases = [
        (
            at(v4(1), Failure::Connect(refused.clone())),
            format!("{all} UNKNOWN: ipv4:127.0.0.1:1: Failed to connect to remote host: Connection refused"),
            format!("{all} 127.0.0.1:1: tcp connect error: Connection refused (os error 111)"),
        ),
        (
            at(v6(1), Failure::Connect(refused)),
            format!("{all} UNKNOWN: ipv6:%5B::1%5D:1: Failed to connect to remote host: Connection refused"),
            format!("{all} [::1]:1: tcp connect error: Connection refused (os error 111)"),
        ),
        (
            at(v4(2), Failure::ConnectTimeout),
            format!("{all} UNKNOWN: ipv4:127.0.0.1:2: tcp handshaker shutdown"),
            format!("{all} 127.0.0.1:2: tcp connect error: timed out"),
        ),
        (
            at(v4(3), Failure::Timeout(Phase::Settings)),
            format!("{all} UNKNOWN: ipv4:127.0.0.1:3: connection attempt timed out before receiving SETTINGS frame"),
            format!("{all} 127.0.0.1:3: connection attempt timed out before receiving SETTINGS frame"),
        ),
        (
            at(v4(4), Failure::Closed(Phase::Settings)),
            format!("{all} UNAVAILABLE: ipv4:127.0.0.1:4: Socket closed"),
            format!("{all} 127.0.0.1:4: http2 settings: the connection was closed"),
        ),
        (
            at(v4(5), Failure::Read(Phase::Settings, reset.clone())),
            format!("{all} UNAVAILABLE: ipv4:127.0.0.1:5: recvmsg:Connection reset by peer (104)"),
            format!("{all} 127.0.0.1:5: http2 settings: Connection reset by peer (os error 104)"),
        ),
        (
            at(v4(6), Failure::Write(Phase::Settings, reset)),
            format!("{all} UNAVAILABLE: ipv4:127.0.0.1:6: Delayed close due to in-progress write (sendmsg: Connection reset by peer (104))"),
            format!("{all} 127.0.0.1:6: http2 settings: Connection reset by peer (os error 104)"),
        ),
        (
            at(v4(7), Failure::ProxyStatus(403)),
            format!("{all} UNKNOWN: ipv4:127.0.0.1:7: HTTP proxy returned response code 403"),
            format!("{all} 127.0.0.1:7: the proxy refused the tunnel (403)"),
        ),
        (
            at(v4(8), Failure::ProxyStatus(407)),
            format!("{all} UNKNOWN: ipv4:127.0.0.1:8: HTTP proxy returned response code 407"),
            format!("{all} 127.0.0.1:8: the proxy requires authentication (407)"),
        ),
        (
            at(v4(9), Failure::Timeout(Phase::Proxy)),
            format!("{all} UNAVAILABLE: ipv4:127.0.0.1:9: Endpoint closing"),
            format!("{all} 127.0.0.1:9: the proxy did not open the tunnel in time"),
        ),
        (
            at(v4(10), Failure::Closed(Phase::Proxy)),
            format!("{all} UNAVAILABLE: ipv4:127.0.0.1:10: Socket closed"),
            format!("{all} 127.0.0.1:10: proxy tunnel: the connection was closed"),
        ),
        (
            at(v4(11), Failure::ProxyNotHttp('H')),
            format!("{all} UNKNOWN: ipv4:127.0.0.1:11: Expected 'H'"),
            format!("{all} 127.0.0.1:11: the proxy did not answer in HTTP"),
        ),
        (
            at(v4(12), Failure::Timeout(Phase::Tls)),
            format!("{all} UNAVAILABLE: ipv4:127.0.0.1:12: Handshake read failed (Endpoint closing)"),
            format!("{all} 127.0.0.1:12: tls handshake: timed out"),
        ),
        (
            at(v4(13), Failure::Closed(Phase::Tls)),
            format!("{all} UNAVAILABLE: ipv4:127.0.0.1:13: Handshake read failed (Socket closed)"),
            format!("{all} 127.0.0.1:13: tls handshake: the connection was closed"),
        ),
        (
            at(v4(14), Failure::NoAlpn),
            format!("{all} UNKNOWN: ipv4:127.0.0.1:14: Cannot check peer: missing selected ALPN property."),
            format!("{all} 127.0.0.1:14: HTTP/2 was not negotiated."),
        ),
        (
            at(v4(15), Failure::NotSettings(80, Some(400))),
            format!("{all} INTERNAL: ipv4:127.0.0.1:15: Failed parsing HTTP/2 (Expected SETTINGS frame as the first frame, got frame type 80) (Trying to connect an http1.x server (HTTP status 400))"),
            format!("{all} 127.0.0.1:15: the server did not answer in HTTP/2: its first frame is of type 80, not SETTINGS"),
        ),
        (
            at(SocketAddr::from(([10, 92, 26, 67], 17839)), Failure::NotLoopback),
            format!("{all} UNKNOWN: ipv4:10.92.26.67:17839: Endpoint is neither UDS or TCP loopback address."),
            format!("{all} 10.92.26.67:17839: local credentials: the connection is not of the loopback"),
        ),
        (
            ChannelFailure::Resolve {
                name: "nonexistent.test:15002".to_string(),
                failure: ResolveFailure::NotFound("failed to lookup address information: Name or service not known".to_string()),
            },
            "errors resolving nonexistent.test:15002: [field:hostname lookup error:address lookup failed for nonexistent.test:15002: Domain name not found]".to_string(),
            "dns error: failed to look up nonexistent.test:15002: failed to lookup address information: Name or service not known".to_string(),
        ),
        (
            ChannelFailure::Resolve {
                name: "nonexistent.test:15002".to_string(),
                failure: ResolveFailure::NoServers("failed to lookup address information: Temporary failure in name resolution".to_string()),
            },
            "errors resolving nonexistent.test:15002: [field:hostname lookup error:address lookup failed for nonexistent.test:15002: Could not contact DNS servers]".to_string(),
            "dns error: failed to look up nonexistent.test:15002: failed to lookup address information: Temporary failure in name resolution".to_string(),
        ),
    ];
    for (failure, grpcio, rust) in cases {
        assert_eq!(failure.text(true), grpcio);
        assert_eq!(failure.text(false), rust);
    }
}

#[test]
fn a_connection_of_local_credentials_is_of_the_loopback_as_grpc_takes_it() {
    // As gRPC's local credentials of TCP, which take a connection whose local address is
    // `INADDR_LOOPBACK` or `::1`, also of IPv4 mapped to IPv6.
    for (address, loopback) in [
        ("127.0.0.1:1", true),
        ("[::1]:1", true),
        ("[::ffff:127.0.0.1]:1", true),
        ("127.0.0.2:1", false),
        ("10.92.26.67:1", false),
        ("[::ffff:10.92.26.67]:1", false),
        ("[2001:db8::1]:1", false),
        ("0.0.0.0:1", false),
        ("[::]:1", false),
    ] {
        assert_eq!(
            is_of_the_loopback(address.parse().unwrap()),
            loopback,
            "{address}"
        );
    }
}

#[tokio::test]
async fn a_target_of_local_credentials_connects_of_the_loopback_only() {
    // A connection whose local address is of the loopback is made as of no credentials, also of
    // `0.0.0.0`, and one to another address of the host, of a listener of every address, fails of
    // grpcio's text once it is connected, as gRPC's handshake of local credentials reads the
    // local address of the connection, not the address that it is made to.
    let local = |addresses| {
        let mut target = target(addresses, true);
        target.local = true;
        target
    };
    let (server, _) = settings_server().await;
    let ready = attempt(&local(vec![server]), Duration::from_secs(20)).await;
    assert!(ready.is_ok(), "{:?}", ready.err());
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    settings_server_on(listener);
    if cfg!(target_os = "linux") {
        let any = SocketAddr::from(([0, 0, 0, 0], port));
        let ready = attempt(&local(vec![any]), Duration::from_secs(20)).await;
        assert!(ready.is_ok(), "{:?}", ready.err());
    }
    // An address of the host that is not of the loopback, as its route to the outside has.
    let outside = std::net::UdpSocket::bind("0.0.0.0:0")
        .and_then(|socket| socket.connect("192.0.2.1:9").map(|()| socket))
        .and_then(|socket| socket.local_addr())
        .ok()
        .map(|address| address.ip())
        .filter(|ip| !ip.is_loopback() && !ip.is_unspecified());
    let Some(outside) = outside else {
        eprintln!("skipped: the host has no address outside the loopback");
        return;
    };
    let address = SocketAddr::new(outside, port);
    // Not of the TCP connection, which is made, but of its local address.
    match attempt(&local(vec![address]), Duration::from_secs(20)).await {
        Err(ChannelFailure::Addresses { failure, .. }) => {
            assert!(matches!(failure, Failure::NotLoopback), "{failure:?}");
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
    // Of no local credentials, the same address connects.
    assert!(
        attempt(&target(vec![address], true), Duration::from_secs(20))
            .await
            .is_ok()
    );
}

#[cfg(target_os = "linux")]
#[test]
fn the_failures_of_the_resolver_are_told_apart_by_the_text_of_their_code() {
    // As the standard library gives them, of the text of `gai_strerror`.
    let failure = |code| {
        // SAFETY: `gai_strerror` returns a static string for any code.
        let text = unsafe { std::ffi::CStr::from_ptr(libc::gai_strerror(code)) };
        ResolveFailure::of(&std::io::Error::other(format!(
            "failed to lookup address information: {}",
            text.to_string_lossy()
        )))
    };
    assert!(matches!(
        failure(libc::EAI_NONAME),
        ResolveFailure::NotFound(_)
    ));
    assert!(matches!(
        failure(libc::EAI_AGAIN),
        ResolveFailure::NoServers(_)
    ));
    assert!(matches!(
        failure(libc::EAI_FAIL),
        ResolveFailure::ServerFailure(_)
    ));
    assert!(matches!(failure(libc::EAI_NODATA), ResolveFailure::NoData));
    assert!(matches!(
        ResolveFailure::of(&std::io::Error::other("something else")),
        ResolveFailure::Other(_)
    ));
}

#[tokio::test]
async fn an_attempt_starts_the_next_address_250_ms_after_the_previous_one() {
    // As gRPC's happy eyeballs: the first address accepts and never answers, the next one starts
    // 250 ms later and connects, while the first is still waiting for its settings.
    let (silent, silent_accepted) = silent_server().await;
    let (server, server_accepted) = settings_server().await;
    let start = Instant::now();
    let ready = attempt(&target(vec![silent, server], true), Duration::from_secs(20)).await;
    let elapsed = start.elapsed();
    assert!(ready.is_ok(), "{:?}", ready.err());
    assert!(elapsed >= Duration::from_millis(240), "{elapsed:?}");
    assert!(elapsed < Duration::from_secs(5), "{elapsed:?}");
    assert_eq!(silent_accepted.load(Ordering::SeqCst), 1);
    assert_eq!(server_accepted.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn an_attempt_that_fails_starts_the_next_address_at_once() {
    // At once, of a delay of the next attempt far longer than the attempts take, whatever else
    // the machine runs.
    let refused = closed_port();
    let (server, _) = settings_server().await;
    let start = Instant::now();
    let ready = attempt_after(
        &target(vec![refused.address, server], true),
        Duration::from_secs(20),
        Duration::from_secs(10),
    )
    .await;
    assert!(ready.is_ok(), "{:?}", ready.err());
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{:?}",
        start.elapsed()
    );
}

#[tokio::test]
async fn the_failure_of_a_channel_is_that_of_the_last_address_to_fail() {
    // Each attempt has the deadline of its own: the refused address fails at once, the silent one
    // at its deadline, which is the last failure, of the text of grpcio.
    let refused = closed_port();
    let (silent, _) = silent_server().await;
    let deadline = Duration::from_millis(400);
    let start = Instant::now();
    let failure = attempt(&target(vec![refused.address, silent], true), deadline)
        .await
        .err()
        .expect("no address connects");
    assert!(start.elapsed() >= deadline, "{:?}", start.elapsed());
    assert_eq!(
        failure.text(true),
        format!(
            "failed to connect to all addresses; last error: UNKNOWN: ipv4:{silent}: connection \
             attempt timed out before receiving SETTINGS frame"
        )
    );
}

#[tokio::test]
async fn a_server_that_answers_in_http1_is_told_as_grpcio_tells_it() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    serve(listener, |mut stream| async move {
        let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\n\r\n").await;
        std::future::pending::<()>().await;
    });
    let failure = attempt(&target(vec![address], true), Duration::from_secs(5))
        .await
        .err()
        .expect("not HTTP/2");
    assert!(
        matches!(
            failure,
            ChannelFailure::Addresses {
                failure: Failure::NotSettings(80, Some(400)),
                ..
            }
        ),
        "{failure:?}"
    );
}

/// A proxy that records the head of each request and answers each with `answer`, and the heads.
async fn proxy(answer: &'static [u8]) -> (SocketAddr, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let heads = Arc::new(Mutex::new(Vec::new()));
    let seen = heads.clone();
    serve(listener, move |mut stream| {
        let seen = seen.clone();
        async move {
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                if stream.read(&mut byte).await.unwrap_or(0) == 0 {
                    return;
                }
                head.push(byte[0]);
            }
            seen.lock()
                .unwrap()
                .push(String::from_utf8_lossy(&head).into_owned());
            let _ = stream.write_all(answer).await;
            std::future::pending::<()>().await;
        }
    });
    (address, heads)
}

fn through_proxy(proxy: SocketAddr, grpcio: bool) -> Target {
    Target {
        tunnel: Some(Tunnel {
            authority: "spark.invalid:15002".to_string(),
            authorization: Some("Basic YWxpY2U6cEBzcw==".to_string()),
        }),
        ..target(vec![proxy], grpcio)
    }
}

#[tokio::test]
async fn the_connect_request_of_a_proxy_has_grpcio_s_user_agent_in_its_dialect() {
    let (address, heads) = proxy(b"HTTP/1.1 403 Forbidden\r\n\r\n").await;
    for grpcio in [true, false] {
        let failure = attempt(&through_proxy(address, grpcio), Duration::from_secs(5))
            .await
            .err()
            .expect("the proxy refuses the tunnel");
        let expected = if grpcio {
            format!(
                "failed to connect to all addresses; last error: UNKNOWN: ipv4:{address}: HTTP \
                 proxy returned response code 403"
            )
        } else {
            format!(
                "failed to connect to all addresses; last error: {address}: the proxy refused \
                 the tunnel (403)"
            )
        };
        assert_eq!(failure.text(grpcio), expected);
    }
    // As grpcio's request, whose headers are the host, its user agent and the credentials.
    assert_eq!(
        *heads.lock().unwrap(),
        [
            "CONNECT spark.invalid:15002 HTTP/1.1\r\nHost: spark.invalid:15002\r\nUser-Agent: \
             grpc-httpcli/0.0\r\nProxy-Authorization: Basic YWxpY2U6cEBzcw==\r\n\r\n",
            "CONNECT spark.invalid:15002 HTTP/1.1\r\nHost: spark.invalid:15002\r\n\
             Proxy-Authorization: Basic YWxpY2U6cEBzcw==\r\n\r\n",
        ]
    );
}

#[tokio::test]
async fn a_proxy_that_does_not_answer_in_http_fails_as_grpcio_s_parser() {
    let (address, _) = proxy(b"SSH-2.0-OpenSSH_8.2\r\n\r\n").await;
    let failure = attempt(&through_proxy(address, true), Duration::from_secs(5))
        .await
        .err()
        .expect("not HTTP");
    assert!(
        failure.text(true).ends_with(": Expected 'H'"),
        "{failure:?}"
    );
}

#[tokio::test]
async fn the_server_bytes_after_the_response_of_a_proxy_are_its_settings() {
    // The settings of a plaintext server can come in the same read as the response of the
    // proxy.
    let mut answer = b"HTTP/1.1 200 Connection established\r\n\r\n".to_vec();
    answer.extend_from_slice(&EMPTY_SETTINGS);
    let (address, _) = proxy(Box::leak(answer.into_boxed_slice())).await;
    let ready = attempt(&through_proxy(address, true), Duration::from_secs(5))
        .await
        .map_err(|failure| failure.text(true))
        .unwrap();
    assert_eq!(ready.replay, EMPTY_SETTINGS);
}

/// The bytes that `reader` gives until it would wait.
async fn available(reader: &mut (impl AsyncRead + Unpin)) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 64];
    while let Ok(Ok(read)) =
        tokio::time::timeout(Duration::from_millis(50), reader.read(&mut buffer)).await
    {
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..read]);
    }
    bytes
}

#[tokio::test]
async fn hyper_is_given_the_connection_without_the_preface_and_the_ack_of_the_connector() {
    // The connector sent the preface and its own empty settings, and read the server's settings:
    // hyper's own preface is not sent again, and the server's acknowledgment of the connector's
    // settings is not given to hyper, which sent none yet.
    let (client, mut server) = tokio::io::duplex(1024);
    let settings = [0, 0, 6, SETTINGS, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 100];
    let ack = [0, 0, 0, SETTINGS, ACK, 0, 0, 0, 0];
    let ping = [0, 0, 8, 6, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8];
    // The settings were read already; the acknowledgment and a ping come, in two reads.
    let mut handover = Handover::new(Ready {
        io: Box::new(client),
        replay: settings.to_vec(),
    });
    server.write_all(&ack[..4]).await.unwrap();
    let mut read = vec![0u8; settings.len()];
    handover.read_exact(&mut read).await.unwrap();
    assert_eq!(read, settings);
    server
        .write_all(&[&ack[4..], &ping[..]].concat())
        .await
        .unwrap();
    assert_eq!(available(&mut handover).await, ping);
    // Another acknowledgment, of hyper's settings, is hyper's.
    server.write_all(&ack).await.unwrap();
    assert_eq!(available(&mut handover).await, ack);
    // hyper's preface is dropped, whatever writes it comes in; the rest is sent.
    handover.write_all(&PREFACE[..10]).await.unwrap();
    handover
        .write_all(&[&PREFACE[10..], &settings[..]].concat())
        .await
        .unwrap();
    assert_eq!(available(&mut server).await, settings);
    // The end of the connection is grpcio's `Socket closed`.
    drop(server);
    let end = handover.read(&mut [0u8; 16]).await.unwrap_err();
    assert_eq!(end.kind(), std::io::ErrorKind::UnexpectedEof);
    assert_eq!(end.to_string(), SOCKET_CLOSED);
}

#[tokio::test]
async fn hyper_s_settings_advertise_the_header_list_size_of_grpcio() {
    // hyper's settings, the frame after its preface, advertise gRPC's hard limit of the metadata
    // of a call as the size of a header list, as grpcio's channel does, in the place of hyper's
    // own, whatever writes they come in; the frames after them are sent as they are, a later
    // SETTINGS too.
    let (client, mut server) = tokio::io::duplex(4096);
    let mut handover = Handover::new(Ready {
        io: Box::new(client),
        replay: vec![0, 0, 0, SETTINGS, 0, 0, 0, 0, 0],
    });
    let entry = |id: u16, value: u32| [&id.to_be_bytes()[..], &value.to_be_bytes()[..]].concat();
    let frame = |kind: u8, payload: &[u8]| {
        [
            &[0, 0, payload.len() as u8, kind, 0, 0, 0, 0, 0][..],
            payload,
        ]
        .concat()
    };
    let settings = |header_list: u32| {
        frame(
            SETTINGS,
            &[
                entry(2, 0),
                entry(4, 2 << 20),
                entry(6, header_list),
                entry(5, 16384),
            ]
            .concat(),
        )
    };
    let window_update = [0, 0, 4, 8, 0, 0, 0, 0, 0, 0, 0, 1, 0];
    let later = frame(SETTINGS, &entry(6, 16 << 20));
    let written = [&PREFACE[..], &settings(16 << 20), &window_update, &later].concat();
    // In writes of 7 bytes, which split the header, the identifiers and the values.
    for chunk in written.chunks(7) {
        handover.write_all(chunk).await.unwrap();
    }
    assert_eq!(
        available(&mut server).await,
        [&settings(16384)[..], &window_update, &later].concat()
    );
}

#[tokio::test]
async fn hyper_is_given_the_frames_of_the_connection_before_the_ack_without_the_ack() {
    // Any frame of stream 0 can come before the acknowledgment of the connector's settings, as an
    // ALTSVC, a PRIORITY_UPDATE or one of an unknown type, which hyper ignores: the
    // acknowledgment after them is still dropped, so that hyper sees one, of its own settings.
    let (client, mut server) = tokio::io::duplex(1024);
    let settings = [0, 0, 0, SETTINGS, 0, 0, 0, 0, 0];
    let altsvc = [0, 0, 2, 0x0a, 0, 0, 0, 0, 0, 0, 0];
    let priority_update = [0, 0, 4, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 1];
    let unknown = [0, 0, 1, 0xfa, 0, 0, 0, 0, 0, 7];
    let ack = [0, 0, 0, SETTINGS, ACK, 0, 0, 0, 0];
    let mut handover = Handover::new(Ready {
        io: Box::new(client),
        replay: settings.to_vec(),
    });
    let before = [&altsvc[..], &priority_update[..], &unknown[..]].concat();
    server
        .write_all(&[&before[..], &ack[..]].concat())
        .await
        .unwrap();
    let mut read = vec![0u8; settings.len() + before.len()];
    handover.read_exact(&mut read).await.unwrap();
    assert_eq!(read, [&settings[..], &before[..]].concat());
    // The next acknowledgment is hyper's; a frame of a stream before one ends the scan.
    server.write_all(&ack).await.unwrap();
    assert_eq!(available(&mut handover).await, ack);
}

#[tokio::test]
async fn a_preface_that_is_not_http2_is_refused() {
    let (client, _server) = tokio::io::duplex(64);
    let mut handover = Handover::new(Ready {
        io: Box::new(client),
        replay: Vec::new(),
    });
    assert!(handover.write_all(b"GET / HTTP/1.1\r\n").await.is_err());
}

/// An error whose source is `source`, as tonic's errors of a transport.
#[derive(Debug)]
struct Transport(Box<dyn std::error::Error + Send + Sync>);

impl std::fmt::Display for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("error reading a body from connection")
    }
}

impl std::error::Error for Transport {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

#[test]
fn a_call_that_the_end_of_its_connection_fails_is_a_removed_stream_in_grpcio_s_dialect() {
    // tonic's status of a body that hyper could not read, of the error of h2, of the I/O error
    // of the connection, which `Handover` gives.
    let ended =
        |error: std::io::Error| tonic::Status::from_error(Box::new(Transport(Box::new(error))));
    let closed = ended(std::io::Error::new(
        std::io::ErrorKind::UnexpectedEof,
        SOCKET_CLOSED,
    ));
    let reset = ended(read_error(std::io::Error::from_raw_os_error(
        libc::ECONNRESET,
    )));
    assert_eq!(
        stream_removed_in(&closed, true).as_deref(),
        Some("Stream removed (Socket closed)")
    );
    assert_eq!(
        stream_removed_in(&reset, true).as_deref(),
        Some("Stream removed (recvmsg:Connection reset by peer (104))")
    );
    // Not in the Rust client's, and not of another error.
    assert_eq!(stream_removed_in(&closed, false), None);
    let other = ended(std::io::Error::other("something else"));
    assert_eq!(stream_removed_in(&other, true), None);
    assert_eq!(
        stream_removed_in(&tonic::Status::unavailable("Socket closed"), true),
        None
    );
}

/// What a call of `connector` gets: the connection, or the text of its failure.
async fn connection(connector: &mut Connector) -> Result<Handover, String> {
    use tonic::codegen::Service;
    let uri = tonic::codegen::http::Uri::from_static("http://localhost");
    connector
        .call(uri)
        .await
        .map(TokioIo::into_inner)
        .map_err(|error| error.to_string())
}

#[tokio::test]
async fn a_channel_that_failed_fails_its_calls_at_once_and_connects_again_in_the_background() {
    // As gRPC's channel in TRANSIENT_FAILURE: the calls after a failure fail at once with its
    // error, while the channel connects again after its backoff of 1 s; the next call takes the
    // connection that it made.
    let closed = closed_port();
    let address = closed.address;
    let mut connector = Connector::new(target(vec![address], true));
    connector.acquire();
    let failure = connection(&mut connector).await.err().expect("refused");
    assert!(failure.ends_with("Connection refused"), "{failure}");
    let start = Instant::now();
    assert_eq!(
        connection(&mut connector).await.err(),
        Some(failure.clone())
    );
    assert!(
        start.elapsed() < Duration::from_millis(100),
        "{:?}",
        start.elapsed()
    );
    let listener = closed.listen();
    let accepted = settings_server_on(listener);
    // Still failing until the background connection is made, between 0.8 and 1.2 s after the
    // first attempt.
    assert_eq!(connection(&mut connector).await.err(), Some(failure));
    let held = async {
        while !matches!(*connector.0.state(), State::Held(_)) {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(5), held)
        .await
        .expect("connected in the background");
    assert!(connection(&mut connector).await.is_ok());
    assert_eq!(accepted.load(Ordering::SeqCst), 1);
    assert!(matches!(*connector.0.state(), State::Ready));
    // A connection that ended leaves the channel idle: the next call connects again.
    assert!(connection(&mut connector).await.is_ok());
    assert_eq!(accepted.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn the_backoff_of_the_connections_is_grpc_s() {
    let mut backoff = Backoff::new();
    let delays: Vec<f64> = (0..12).map(|_| backoff.next().as_secs_f64()).collect();
    let mut expected = 1.0_f64;
    for (attempt, delay) in delays.iter().enumerate() {
        if attempt > 0 {
            expected = (expected * 1.6).min(120.0);
        }
        assert!(
            (expected * 0.8..=expected * 1.2).contains(delay),
            "{attempt}: {delay} of {expected}"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn a_channel_stops_connecting_after_30_minutes_without_a_call() {
    let closed = closed_port();
    let address = closed.address;
    let mut connector = Connector::new(target(vec![address], true));
    connector.acquire();
    assert!(connection(&mut connector).await.is_err());
    tokio::time::sleep(Duration::from_secs(29 * 60)).await;
    assert!(matches!(*connector.0.state(), State::TransientFailure(_)));
    // The next attempt after the 30 minutes stops, within a backoff of at most 2 minutes.
    let idle = async {
        while !matches!(*connector.0.state(), State::Idle) {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(4 * 60), idle)
        .await
        .expect("idle");
    // The next call connects again, and fails again.
    assert!(connection(&mut connector).await.is_err());
    assert!(matches!(*connector.0.state(), State::TransientFailure(_)));
}

#[tokio::test(start_paused = true)]
async fn a_channel_that_no_open_client_uses_stops_connecting_in_the_background() {
    // As grpcio's channel that is closed: once its last open client is closed, the channel stops
    // connecting in the background, drops what it made there, and the next call of a client that
    // opens again connects at once.
    // Each pass of attempts, of the refused address, looks the name up once.
    let closed = closed_port();
    let target = target(vec![closed.address], true);
    let attempts = Arc::clone(&target.lookups);
    let mut connector = Connector::new(target);
    connector.acquire();
    connector.acquire();
    assert!(connection(&mut connector).await.is_err());
    // The attempts in the background after 1, 2.6 and 5.2 s, of the backoff with its jitter.
    tokio::time::sleep(Duration::from_secs(8)).await;
    let made = attempts.load(Ordering::SeqCst);
    assert!(made >= 3, "{made}");
    // One of two clients is closed: the channel goes on.
    connector.release();
    tokio::time::sleep(Duration::from_secs(30)).await;
    assert!(attempts.load(Ordering::SeqCst) > made);
    // The last one is closed: no attempt is made from then on.
    connector.release();
    assert!(matches!(*connector.0.state(), State::Idle));
    let made = attempts.load(Ordering::SeqCst);
    tokio::time::sleep(Duration::from_secs(10 * 60)).await;
    assert_eq!(attempts.load(Ordering::SeqCst), made);
    // A call of a client that opens again connects at once.
    connector.acquire();
    assert!(connection(&mut connector).await.is_err());
    assert_eq!(attempts.load(Ordering::SeqCst), made + 1);
}

#[tokio::test(start_paused = true)]
async fn a_notification_of_a_release_does_not_stop_a_channel_that_a_client_uses() {
    // The release of the last client of a channel notifies its connections in the background
    // after it stopped them; another client may open, fail and start them again in between, whose
    // loop the notification then wakes: the loop goes on, as a client uses the channel.
    let closed = closed_port();
    let target = target(vec![closed.address], true);
    let attempts = Arc::clone(&target.lookups);
    let mut connector = Connector::new(target);
    connector.acquire();
    assert!(connection(&mut connector).await.is_err());
    connector.0.shutdown.notify_waiters();
    tokio::time::sleep(Duration::from_secs(8)).await;
    let made = attempts.load(Ordering::SeqCst);
    assert!(made >= 3, "{made}");
    assert!(matches!(*connector.0.state(), State::TransientFailure(_)));
}

#[test]
fn the_failure_of_a_tunnel_is_logged_once_a_minute_at_most_as_grpcio_logs_it() {
    // As gRPC's core, whose line of a failed tunnel is logged once a minute at most, whatever the
    // attempts in between, where it logs the failure of each TLS handshake.
    let last = Mutex::new(None);
    let start = std::time::Instant::now();
    let at = |seconds: u64| start + Duration::from_secs(seconds);
    let logged: Vec<u64> = [0, 1, 3, 5, 9, 16, 27, 44, 59, 60, 71, 115, 119, 120]
        .into_iter()
        .filter(|&seconds| proxy_failure_logged_at(&last, at(seconds)))
        .collect();
    assert_eq!(logged, [0, 60, 120]);
}

#[tokio::test]
async fn a_rust_client_s_channel_connects_again_at_each_call() {
    // Out of the dialect of grpcio, a call that needs a connection connects, as tonic's channel
    // does, rather than failing at once while the channel connects in the background.
    let closed = closed_port();
    let address = closed.address;
    let mut connector = Connector::new(target(vec![address], false));
    connector.acquire();
    let failure = connection(&mut connector).await.err().expect("refused");
    assert!(failure.contains("tcp connect error"), "{failure}");
    assert!(matches!(*connector.0.state(), State::Idle));
    let listener = closed.listen();
    let accepted = settings_server_on(listener);
    assert!(connection(&mut connector).await.is_ok());
    assert_eq!(accepted.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn the_lookup_of_a_name_is_bounded_by_the_connect_deadline() {
    // As tonic's connect timeout bounded it: a resolver that does not answer fails the attempt at
    // the deadline, of c-ares's text of a timeout in the dialect of grpcio.
    let closed = closed_port();
    for grpcio in [true, false] {
        let target = Target {
            host: "slow.test".to_string(),
            resolves_never: true,
            ..target(vec![closed.address], grpcio)
        };
        let start = Instant::now();
        let failure = attempt(&target, CONNECT_TIMEOUT)
            .await
            .err()
            .expect("timed out");
        assert_eq!(start.elapsed(), CONNECT_TIMEOUT);
        let name = target.name();
        assert_eq!(
            failure.text(grpcio),
            if grpcio {
                format!(
                    "errors resolving {name}: [field:hostname lookup error:address lookup failed \
                     for {name}: Timeout while contacting DNS servers]"
                )
            } else {
                format!("dns error: failed to look up {name}: timed out")
            }
        );
    }
}

#[test]
fn localhost_is_the_loopback_of_both_families_as_c_ares_gives_it() {
    // c-ares, grpcio's resolver, gives `localhost` and the names under it the loopback of a
    // family of which the hosts file has no address (RFC 6761), and gRPC sorts `::1` first.
    assert!(is_localhost("localhost") && is_localhost("LocalHost."));
    assert!(is_localhost("spark.localhost") && !is_localhost("notlocalhost"));
    let v4 = |ip: [u8; 4]| SocketAddr::from((ip, 15002));
    let v6 = SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, 15002));
    let sorted = |addresses: Vec<SocketAddr>| {
        if ipv6_loopback_usable() {
            addresses
        } else {
            let (mut first, rest): (Vec<_>, Vec<_>) =
                addresses.into_iter().partition(SocketAddr::is_ipv4);
            first.extend(rest);
            first
        }
    };
    assert_eq!(
        localhost_addresses(vec![v4([127, 0, 0, 1])], 15002),
        sorted(vec![v6, v4([127, 0, 0, 1])])
    );
    assert_eq!(
        localhost_addresses(vec![v4([127, 0, 0, 2])], 15002),
        sorted(vec![v6, v4([127, 0, 0, 2])])
    );
    assert_eq!(
        localhost_addresses(vec![v6], 15002),
        sorted(vec![v6, v4([127, 0, 0, 1])])
    );
    assert_eq!(
        localhost_addresses(Vec::new(), 15002),
        sorted(vec![v6, v4([127, 0, 0, 1])])
    );
}

/// A TLS server of the certificate `issued`, which selects HTTP/2 by ALPN if `alpn`, and then
/// sends its settings if `settings`.
async fn tls_server(issued: &Issued, alpn: bool, settings: bool) -> SocketAddr {
    use rustls::pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer};
    let certs = vec![issued.cert.clone()];
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(issued.key.clone()));
    let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .unwrap()
    .with_no_client_auth()
    .with_single_cert(certs, key)
    .unwrap();
    if alpn {
        config.alpn_protocols = vec![b"h2".to_vec()];
    }
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    serve(listener, move |stream| {
        let acceptor = acceptor.clone();
        async move {
            if let Ok(mut stream) = acceptor.accept(stream).await {
                if settings {
                    let _ = stream.write_all(&EMPTY_SETTINGS).await;
                    let _ = stream.flush().await;
                }
                let mut buffer = [0u8; 1024];
                while matches!(stream.read(&mut buffer).await, Ok(read) if read > 0) {}
            }
        }
    });
    address
}

/// The target of a TLS server at `address` of the name `domain`, trusting the certificate `root`.
fn tls_target(address: SocketAddr, domain: &str, root: &[u8]) -> Target {
    Target {
        tls: Some(Tls::new(domain, Roots::Only(pem(root).as_bytes())).unwrap()),
        ..target(vec![address], true)
    }
}

/// The last error of the failure of a TLS server of the certificate `issued` at `domain`,
/// trusting the certificate `root`, in grpcio's dialect.
async fn tls_failure(issued: &Issued, alpn: bool, domain: &str, root: &[u8]) -> String {
    let address = tls_server(issued, alpn, true).await;
    let failure = attempt(&tls_target(address, domain, root), Duration::from_secs(5))
        .await
        .err()
        .expect("the TLS session fails");
    let text = failure.text(true);
    let prefix =
        format!("failed to connect to all addresses; last error: UNKNOWN: ipv4:{address}: ");
    text.strip_prefix(&prefix).unwrap_or(&text).to_string()
}

#[tokio::test]
async fn a_tls_session_is_made_by_the_connector_and_has_the_settings_of_the_server() {
    for domain in ["localhost", "127.0.0.1"] {
        let address = tls_server(&certs().localhost, true, true).await;
        let ready = attempt(
            &tls_target(address, domain, &certs().ca),
            Duration::from_secs(5),
        )
        .await;
        assert!(ready.is_ok(), "{domain}: {:?}", ready.err());
    }
}

#[tokio::test]
async fn the_settings_of_a_tls_server_are_bounded_by_the_connect_deadline() {
    // As grpcio's: the handshake completes, and the server never sends its settings.
    let address = tls_server(&certs().localhost, true, false).await;
    let failure = attempt(
        &tls_target(address, "localhost", &certs().ca),
        Duration::from_millis(500),
    )
    .await
    .err()
    .expect("no settings");
    assert!(
        failure
            .text(true)
            .ends_with(": connection attempt timed out before receiving SETTINGS frame"),
        "{failure:?}"
    );
}

#[tokio::test]
async fn the_failures_of_tls_have_boringssl_s_texts_in_grpcio_s_dialect() {
    let verify_failed = "Tls handshake failed (TSI_PROTOCOL_FAILURE): SSL_ERROR_SSL: \
                         error:1000007d:SSL routines:OPENSSL_internal:CERTIFICATE_VERIFY_FAILED:";
    assert_eq!(
        tls_failure(
            &certs().localhost,
            true,
            "localhost",
            &certs().self_signed.cert
        )
        .await,
        format!("{verify_failed} unable to get local issuer certificate: OK")
    );
    assert_eq!(
        tls_failure(&certs().self_signed, true, "localhost", &certs().ca).await,
        format!("{verify_failed} self signed certificate: OK")
    );
    assert_eq!(
        tls_failure(&certs().expired, true, "localhost", &certs().ca).await,
        format!("{verify_failed} certificate has expired: OK")
    );
    assert_eq!(
        tls_failure(&certs().other, true, "localhost", &certs().ca).await,
        "Custom verification check failed with error: UNAUTHENTICATED: Hostname Verification \
         Check failed."
    );
    assert_eq!(
        tls_failure(&certs().localhost, false, "localhost", &certs().ca).await,
        "Cannot check peer: missing selected ALPN property."
    );
    // A server that does not speak TLS.
    let (plaintext, _) = settings_server().await;
    let failure = attempt(
        &tls_target(plaintext, "localhost", &certs().ca),
        Duration::from_secs(5),
    )
    .await
    .err()
    .expect("not TLS");
    assert_eq!(
        failure.text(true),
        format!(
            "failed to connect to all addresses; last error: UNKNOWN: ipv4:{plaintext}: Tls \
             handshake failed (TSI_PROTOCOL_FAILURE): SSL_ERROR_SSL: error:100000f7:SSL \
             routines:OPENSSL_internal:WRONG_VERSION_NUMBER: Invalid certificate verification \
             context: OK"
        )
    );
}

#[tokio::test]
async fn a_tls_of_no_root_fails_each_attempt_as_grpcio_fails_its_handshaker() {
    // As grpcio of a roots file without a certificate: the channel is made, and each attempt
    // fails once the TCP connection is made, as the handshaker of its session cannot be made.
    let address = tls_server(&certs().localhost, true, true).await;
    let tls = Tls::new("localhost", Roots::Only(b"not a certificate")).unwrap();
    for grpcio in [true, false] {
        let target = Target {
            tls: Some(tls.clone()),
            ..target(vec![address], grpcio)
        };
        let failure = attempt(&target, Duration::from_secs(5))
            .await
            .err()
            .expect("no root");
        assert_eq!(
            failure.text(grpcio),
            if grpcio {
                format!(
                    "failed to connect to all addresses; last error: UNKNOWN: ipv4:{address}: \
                     Failed to create security handshaker."
                )
            } else {
                format!(
                    "failed to connect to all addresses; last error: {address}: tls handshake: \
                     no root certificate to trust"
                )
            }
        );
    }
}

#[tokio::test]
async fn the_bundled_roots_are_trusted_besides_those_of_the_system() {
    // The roots of the system's store and those of the bundle that the Python binding gives,
    // grpcio's: the CA of the tests, in the bundle, is trusted.
    let address = tls_server(&certs().localhost, true, true).await;
    let ca = pem(&certs().ca);
    let tls = Tls::new("localhost", Roots::System(Some(ca.as_bytes()))).unwrap();
    let target = Target {
        tls: Some(tls),
        ..target(vec![address], true)
    };
    let ready = attempt(&target, Duration::from_secs(5)).await;
    assert!(ready.is_ok(), "{:?}", ready.err());
}

#[test]
fn a_self_issued_certificate_is_one_whose_issuer_is_its_subject() {
    assert!(self_issued(&certs().self_signed.cert));
    assert!(self_issued(&certs().ca));
    assert!(!self_issued(&certs().localhost.cert));
    assert!(!self_issued(b"not a certificate"));
}

/// A target of the target of gRPC `form`, of no address that the tests resolve, in the dialect of
/// grpcio if `grpcio`.
fn form_target(form: ChannelTarget, grpcio: bool) -> Target {
    Target {
        host: "localhost".to_string(),
        port: 443,
        tunnel: None,
        tls: None,
        local: false,
        grpcio,
        form: None,
        names: None,
        resolved: Default::default(),
        addresses: None,
        resolves_never: false,
        dns_config: None,
        lookups: Default::default(),
        name_lookups: Default::default(),
    }
    .of_channel_target(Some(form), None)
}

/// The text of the failure of the attempts to `target`, in the dialect of grpcio.
async fn failure_of(target: &Target) -> String {
    attempt(target, Duration::from_secs(20))
        .await
        .err()
        .expect("the target does not connect")
        .text(true)
}

#[tokio::test]
async fn the_addresses_of_a_target_of_grpc_are_tried_in_turn() {
    // As gRPC's `pick_first` of the addresses of `ipv4:`, of no lookup: the refused address fails,
    // and the next one connects; of none, the calls fail of `empty address list`.
    let refused = closed_port();
    let (server, _) = settings_server().await;
    let addresses = |addresses: Vec<SocketAddr>| ChannelTarget::Addresses {
        addresses,
        authority: String::new(),
    };
    let target = form_target(addresses(vec![refused.address, server]), true);
    assert!(attempt(&target, Duration::from_secs(20)).await.is_ok());
    assert_eq!(target.lookups.load(Ordering::SeqCst), 1);
    let target = form_target(addresses(vec![refused.address]), true);
    assert_eq!(
        failure_of(&target).await,
        format!(
            "failed to connect to all addresses; last error: UNKNOWN: ipv4:{}: Failed to connect \
             to remote host: Connection refused",
            refused.address
        )
    );
    assert_eq!(
        failure_of(&form_target(addresses(vec![]), true)).await,
        "empty address list"
    );
    let none = attempt(
        &form_target(addresses(vec![]), false),
        Duration::from_secs(1),
    )
    .await;
    assert_eq!(none.err().unwrap().text(false), "the target has no address");
}

#[tokio::test]
async fn the_names_of_a_target_of_grpc_are_resolved_as_c_ares_resolves_them() {
    let (server, _) = settings_server().await;
    let name = |host: &str| ChannelTarget::Name {
        host: host.to_string(),
        port: server.port(),
        name: format!("{host}:{}", server.port()),
        server: None,
    };
    // Four decimal numbers are an address, of leading zeros too, and the names under `localhost`
    // the loopback, of any character.
    for host in [
        "127.000.0.1",
        "127.0.0.01",
        "x..localhost",
        "a!b.localhost",
        "LocalHost.",
    ] {
        let target = form_target(name(host), true);
        assert!(
            attempt(&target, Duration::from_secs(20)).await.is_ok(),
            "{host}"
        );
    }
    // A name of a character that c-ares does not take, of an empty label, or of a label of more
    // than 63 characters, is not resolved.
    let long_label = format!("{}.test", "a".repeat(64));
    for host in [
        "local!host",
        "local..host",
        ".foo.invalid",
        long_label.as_str(),
    ] {
        let target = form_target(name(host), true);
        let port = server.port();
        assert_eq!(
            failure_of(&target).await,
            format!(
                "errors resolving {host}:{port}: [field:hostname lookup error:address lookup \
                 failed for {host}:{port}: Misformatted domain name]"
            )
        );
    }
    // A name that gRPC's resolver does not take fails of the error that it gives.
    let unresolvable = ChannelTarget::Unresolvable {
        name: "localhost:abc".to_string(),
        detail: "Failed to parse port in name: localhost:abc".to_string(),
    };
    assert_eq!(
        failure_of(&form_target(unresolvable, true)).await,
        "errors resolving localhost:abc: [field:hostname lookup error:Failed to parse port in \
         name: localhost:abc]"
    );
}

#[test]
fn a_name_is_one_that_c_ares_asks_for() {
    for name in [
        "spark.example.com",
        "a-b_c/d*e",
        "a.",
        ".",
        "x".repeat(63).as_str(),
    ] {
        assert!(is_ares_name(name), "{name}");
    }
    let long = vec!["a".repeat(60); 5].join(".");
    for name in ["a b", "a!b", "a..b", "..", ".a", "a:b", long.as_str()] {
        assert!(!is_ares_name(name), "{name}");
    }
    if cfg!(unix) {
        for name in ["127.1", "0x7f.0.0.1", "2130706433", "127.0.0.1"] {
            assert!(read_as_an_address(name), "{name}");
        }
        assert!(!read_as_an_address("127.1."));
        assert!(!read_as_an_address("localhost"));
    }
}

#[test]
fn a_connect_that_fails_as_it_starts_has_grpc_core_s_text() {
    let unix = Address::Unix {
        path: b"/tmp/none.sock".to_vec(),
        abstract_name: false,
    };
    let failure = ChannelFailure::Addresses {
        address: unix.clone(),
        failure: Failure::ConnectNow(os_error(libc::ENOENT)),
    };
    assert_eq!(
        failure.text(true),
        "failed to connect to all addresses; last error: FAILED_PRECONDITION: \
         unix:/tmp/none.sock: connect failed: addr: unix:/tmp/none.sock error: No such file or \
         directory"
    );
    assert_eq!(
        failure.text(false),
        "failed to connect to all addresses; last error: unix:/tmp/none.sock: unix connect \
         error: No such file or directory (os error 2)"
    );
    let unreachable = ChannelFailure::Addresses {
        address: SocketAddr::V6(std::net::SocketAddrV6::new(
            "fe80::1".parse().unwrap(),
            15002,
            0,
            1,
        ))
        .into(),
        failure: Failure::ConnectNow(os_error(libc::ENETUNREACH)),
    };
    assert_eq!(
        unreachable.text(true),
        "failed to connect to all addresses; last error: FAILED_PRECONDITION: \
         ipv6:%5Bfe80::1%251%5D:15002: connect failed: addr: ipv6:%5Bfe80::1%251%5D:15002 error: \
         Network is unreachable"
    );
    let abstract_name = Address::Unix {
        path: b"name".to_vec(),
        abstract_name: true,
    };
    assert_eq!(grpc_address(&abstract_name), "unix-abstract:name");
    // Of the bytes of a path that are not of a URI's path, as gRPC's `URI::ToString` gives them.
    let path = Address::Unix {
        path: b"/tmp/a b%\xff\xef\xbf\xbd:@!~.sock".to_vec(),
        abstract_name: false,
    };
    assert_eq!(
        grpc_address(&path),
        "unix:/tmp/a%20b%25%FF%EF%BF%BD:@!~.sock"
    );
    let abstract_name = Address::Unix {
        path: b"a\0b/c".to_vec(),
        abstract_name: true,
    };
    assert_eq!(grpc_address(&abstract_name), "unix-abstract:a%00b/c");
}

#[cfg(unix)]
#[tokio::test]
async fn a_target_of_unix_sockets_connects_to_them_in_turn() {
    use tokio::net::UnixListener;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("spark.sock");
    let missing = directory.path().join("none.sock");
    let listener = UnixListener::bind(&path).unwrap();
    tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            tokio::spawn(async move {
                let _ = stream.write_all(&EMPTY_SETTINGS).await;
                let mut buffer = [0u8; 1024];
                while matches!(stream.read(&mut buffer).await, Ok(read) if read > 0) {}
            });
        }
    });
    let bytes = |path: &std::path::Path| path.to_str().unwrap().as_bytes().to_vec();
    let unix = |paths: Vec<Vec<u8>>| ChannelTarget::Unix {
        paths,
        abstract_names: false,
    };
    let target = form_target(unix(vec![bytes(&missing), bytes(&path)]), true);
    assert!(attempt(&target, Duration::from_secs(20)).await.is_ok());
    let missing_text = missing.to_str().unwrap();
    assert_eq!(
        failure_of(&form_target(unix(vec![bytes(&missing)]), true)).await,
        format!(
            "failed to connect to all addresses; last error: FAILED_PRECONDITION: \
             unix:{missing_text}: connect failed: addr: unix:{missing_text} error: No such file \
             or directory"
        )
    );
    // gRPC's local credentials of TCP refuse a Unix socket.
    let mut local = form_target(unix(vec![bytes(&path)]), true);
    local.local = true;
    assert_eq!(
        failure_of(&local).await,
        format!(
            "failed to connect to all addresses; last error: UNKNOWN: unix:{}: Endpoint is neither \
             UDS or TCP loopback address.",
            path.to_str().unwrap()
        )
    );
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn a_target_of_abstract_unix_sockets_connects_to_them() {
    use std::os::linux::net::SocketAddrExt;
    let name = format!("spark-connect-core-test-{}", std::process::id());
    let address = std::os::unix::net::SocketAddr::from_abstract_name(name.as_bytes()).unwrap();
    let listener = std::os::unix::net::UnixListener::bind_addr(&address).unwrap();
    listener.set_nonblocking(true).unwrap();
    let listener = tokio::net::UnixListener::from_std(listener).unwrap();
    tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            tokio::spawn(async move {
                let _ = stream.write_all(&EMPTY_SETTINGS).await;
                let mut buffer = [0u8; 1024];
                while matches!(stream.read(&mut buffer).await, Ok(read) if read > 0) {}
            });
        }
    });
    let abstract_target = |name: &str| {
        form_target(
            ChannelTarget::Unix {
                paths: vec![name.as_bytes().to_vec()],
                abstract_names: true,
            },
            true,
        )
    };
    assert!(attempt(&abstract_target(&name), Duration::from_secs(20))
        .await
        .is_ok());
    let none = format!("{name}-none");
    assert_eq!(
        failure_of(&abstract_target(&none)).await,
        format!(
            "failed to connect to all addresses; last error: FAILED_PRECONDITION: \
             unix-abstract:{none}: connect failed: addr: unix-abstract:{none} error: Connection \
             refused"
        )
    );
}

#[tokio::test]
async fn the_calls_of_a_target_that_grpc_refuses_fail_of_its_status() {
    // Of the code and the message of the status, as gRPC's channel of a target that is not valid
    // fails each call, and not of `UNAVAILABLE`, which a failure of a connection is.
    let message = "Failed to create channel to 'a?':the target uri is not valid: dns:///a?";
    let connector = Connector::new(form_target(
        ChannelTarget::Refused {
            code: 2,
            message: message.to_string(),
        },
        true,
    ));
    let channel = tonic::transport::Endpoint::from_static("http://localhost")
        .connect_with_connector_lazy(connector);
    let mut grpc = tonic::client::Grpc::new(channel);
    for _ in 0..2 {
        grpc.ready().await.unwrap();
        let status = grpc
            .unary::<spark_connect_proto::ConfigRequest, spark_connect_proto::ConfigResponse, _>(
                tonic::Request::new(spark_connect_proto::ConfigRequest::default()),
                tonic::codegen::http::uri::PathAndQuery::from_static(
                    "/spark.connect.SparkConnectService/Config",
                ),
                tonic_prost::ProstCodec::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unknown);
        assert_eq!(status.message(), message);
    }
    assert_eq!(grpc_message("a%b\u{e9}"), "a%25b%C3%A9");
}

/// The target of a TLS server at `address` of gRPC's TLS name `name` of a target, trusting the
/// certificate `root`.
fn tls_target_of_name(address: SocketAddr, name: Option<&str>, root: &[u8]) -> Target {
    Target {
        tls: Some(Tls::of_target_name(name, Roots::Only(pem(root).as_bytes())).unwrap()),
        ..target(vec![address], true)
    }
}

#[tokio::test]
async fn the_tls_of_a_target_of_grpc_verifies_grpcs_name_of_it() {
    // Of the probes of grpcio 1.85.1 (a certificate of `localhost` and `127.0.0.1`): the name of a
    // target is the host of its authority up to a `%`, an IP address checked against the addresses
    // of the certificate; a name that is neither an address nor a name of DNS (`x*y.localhost`, a
    // list of addresses, none) fails the check of the name once the chain is verified, as gRPC's
    // (`Hostname Verification Check failed`), and so does a name of another certificate.
    let address = tls_server(&certs().localhost, true, true).await;
    for name in ["localhost", "LOCALHOST", "localhost.", "127.0.0.1"] {
        let target = tls_target_of_name(address, Some(name), &certs().ca);
        assert!(
            attempt(&target, Duration::from_secs(5)).await.is_ok(),
            "{name}"
        );
    }
    let hostname = format!(
        "failed to connect to all addresses; last error: UNKNOWN: ipv4:{address}: Custom \
         verification check failed with error: UNAUTHENTICATED: Hostname Verification Check \
         failed."
    );
    for name in [
        Some("x*y.localhost"),
        Some("a%2Fb.localhost"),
        Some("127.0.0.1:1,127.0.0.1:2"),
        Some("other.test"),
        Some("::1"),
        Some("127.000.000.001"),
        Some("[127.0.0.1]"),
        None,
    ] {
        let target = tls_target_of_name(address, name, &certs().ca);
        assert_eq!(failure_of(&target).await, hostname, "{name:?}");
    }
    // The chain first: of a root that does not sign it, its error, whatever the name.
    let target = tls_target_of_name(address, Some("x*y.localhost"), &certs().other.cert);
    assert!(
        failure_of(&target)
            .await
            .contains("CERTIFICATE_VERIFY_FAILED"),
        "{}",
        failure_of(&target).await
    );
}

#[tokio::test]
async fn a_call_host_that_the_certificate_is_not_of_fails_each_call_of_unauthenticated() {
    // Of the probes of grpcio 1.85.1 (`p_tls_edges.py` of the review of SPARK-62220): a
    // certificate of `*.localhost` is of `xyz.localhost` once the handshake is done, as gRPC's
    // verifier takes a wildcard of one label, but not of the host of the calls, as its check of
    // the host of each call does not, of which each call fails of `UNAUTHENTICATED`, which a
    // retry policy does not retry as it retries a failure of a connection.
    let address = tls_server(&certs().wildcard, true, true).await;
    let channel = tonic::transport::Endpoint::from_static("http://localhost")
        .connect_with_connector_lazy(Connector::new(tls_target_of_name(
            address,
            Some("xyz.localhost"),
            &certs().ca,
        )));
    let mut grpc = tonic::client::Grpc::new(channel);
    for _ in 0..2 {
        grpc.ready().await.unwrap();
        let status = grpc
            .unary::<spark_connect_proto::ConfigRequest, spark_connect_proto::ConfigResponse, _>(
                tonic::Request::new(spark_connect_proto::ConfigRequest::default()),
                tonic::codegen::http::uri::PathAndQuery::from_static(
                    "/spark.connect.SparkConnectService/Config",
                ),
                tonic_prost::ProstCodec::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
        assert_eq!(status.message(), "call host does not match SSL server name");
    }
    // A name of two labels below the wildcard is not of the certificate at all: its handshake
    // fails, as gRPC's verifier fails it.
    assert_eq!(
        failure_of(&tls_target_of_name(
            address,
            Some("a.b.localhost"),
            &certs().ca
        ))
        .await,
        format!(
            "failed to connect to all addresses; last error: UNKNOWN: ipv4:{address}: Custom \
             verification check failed with error: UNAUTHENTICATED: Hostname Verification Check \
             failed."
        )
    );
}

#[test]
fn the_tunnel_of_a_target_of_grpc_names_the_server_of_grpcs_proxy_mapper() {
    let proxy = || {
        Some(crate::proxy::HttpProxy {
            authority: "proxy.test:3128".to_string(),
            host: "proxy.test".to_string(),
            port: 3128,
            authorization: None,
        })
    };
    let names = |authority: &str, proxied: Option<(&str, &str)>| TargetNames {
        authority: authority.to_string(),
        proxied: proxied.map(|(host, server)| (host.to_string(), server.to_string())),
    };
    let tunnel_of = |form: ChannelTarget, names: Option<TargetNames>| {
        Target::new("localhost", 443, proxy(), None)
            .of_channel_target(Some(form), names)
            .tunnel
            .map(|tunnel| tunnel.authority)
    };
    let addresses = ChannelTarget::Addresses {
        addresses: vec![
            "127.0.0.1:1".parse().unwrap(),
            "127.0.0.1:2".parse().unwrap(),
        ],
        authority: "127.0.0.1:1,127.0.0.1:2".to_string(),
    };
    assert_eq!(
        tunnel_of(
            addresses.clone(),
            Some(names(
                "127.0.0.1:1,127.0.0.1:2",
                Some(("127.0.0.1:1,127.0.0.1:2", "[127.0.0.1:1,127.0.0.1:2]:443"))
            ))
        ),
        Some("[127.0.0.1:1,127.0.0.1:2]:443".to_string())
    );
    assert_eq!(
        tunnel_of(addresses, None),
        Some("127.0.0.1:1,127.0.0.1:2".to_string())
    );
    // An abstract Unix socket is proxied as gRPC proxies it, through the proxy's name; one of a
    // path is not.
    let abstract_socket = ChannelTarget::Unix {
        paths: vec![b"rv62220ab".to_vec()],
        abstract_names: true,
    };
    let target = Target::new("localhost", 443, proxy(), None).of_channel_target(
        Some(abstract_socket),
        Some(names("rv62220ab", Some(("rv62220ab", "rv62220ab:443")))),
    );
    assert_eq!(
        target
            .tunnel
            .as_ref()
            .map(|tunnel| tunnel.authority.as_str()),
        Some("rv62220ab:443")
    );
    assert_eq!(target.name(), "proxy.test:3128");
    let path_socket = ChannelTarget::Unix {
        paths: vec![b"/tmp/a.sock".to_vec()],
        abstract_names: false,
    };
    assert_eq!(
        tunnel_of(path_socket, Some(names("tmp%2Fa.sock", None))),
        None
    );
}

#[tokio::test]
async fn a_name_of_a_dns_server_of_its_target_is_resolved_through_it() {
    use crate::dns::tests::{Answer, Responder};
    // As grpcio's channel of `dns://<server>/<name>`: the name is asked of the server, of each
    // family, and its addresses are tried; a failure of the lookup has c-ares's text.
    let (server, _) = settings_server().await;
    let dns = Responder::start(
        move |name: &str, _| match name {
            "spark.test" => Answer::Addresses(vec![server.ip()]),
            _ => Answer::Rcode(3),
        },
        true,
    )
    .await;
    let port = server.port();
    let of_server = |host: &str, grpcio: bool| {
        let mut target = form_target(
            ChannelTarget::Name {
                host: host.to_string(),
                port,
                name: format!("{host}:{port}"),
                server: Some(dns.address),
            },
            grpcio,
        );
        target.dns_config = Some(crate::dns::Config::default());
        target
    };
    let target = of_server("spark.test", true);
    assert!(attempt(&target, Duration::from_secs(20)).await.is_ok());
    assert!(dns
        .asked()
        .contains(&(false, "spark.test".to_string(), "A")));
    // The next attempt within 30 s of the resolution takes its addresses, as gRPC's channel does
    // not resolve the name again sooner (SPARK-62233).
    let asked = dns.asked().len();
    assert!(attempt(&target, Duration::from_secs(20)).await.is_ok());
    assert_eq!(dns.asked().len(), asked);
    assert_eq!(
        failure_of(&of_server("nx.test", true)).await,
        format!(
            "errors resolving nx.test:{port}: [field:hostname lookup error:address lookup failed \
             for nx.test:{port}: Domain name not found]"
        )
    );
    let rust = attempt(&of_server("nx.test", false), Duration::from_secs(20)).await;
    assert_eq!(
        rust.err().unwrap().text(false),
        format!("dns error: failed to look up nx.test:{port}: Domain name not found")
    );
    // An address, of leading zeros too, and the loopback, are not asked.
    let asked = dns.asked().len();
    for host in ["127.000.0.1", "localhost"] {
        assert!(
            attempt(&of_server(host, true), Duration::from_secs(20))
                .await
                .is_ok(),
            "{host}"
        );
    }
    assert_eq!(dns.asked().len(), asked);
}

#[tokio::test(start_paused = true)]
async fn a_name_is_resolved_again_no_sooner_than_30_s_after_its_last_resolution() {
    // As gRPC's resolver (`grpc.dns_min_time_between_resolutions_ms`, SPARK-62233): in the dialect
    // of grpcio, the addresses of the last resolution of a name that succeeded are taken for 30 s
    // from its start, by every clone of the target; a Rust client resolves the name for each pass
    // of its attempts, as tonic's channel does.
    for grpcio in [true, false] {
        let mut target = Target::new("localhost", 15002, None, None);
        target.grpcio = grpcio;
        let lookups = Arc::clone(&target.name_lookups);
        let first = target.resolve().await.unwrap();
        assert!(!first.is_empty());
        assert_eq!(target.clone().resolve().await.unwrap(), first);
        assert_eq!(lookups.load(Ordering::SeqCst), if grpcio { 1 } else { 2 });
        tokio::time::advance(Duration::from_secs(29)).await;
        target.resolve().await.unwrap();
        assert_eq!(lookups.load(Ordering::SeqCst), if grpcio { 1 } else { 3 });
        tokio::time::advance(Duration::from_secs(1)).await;
        target.resolve().await.unwrap();
        assert_eq!(lookups.load(Ordering::SeqCst), if grpcio { 2 } else { 4 });
    }
    // An address is no name, which is not looked up.
    let mut target = Target::new("127.0.0.1", 15002, None, None);
    target.grpcio = true;
    target.resolve().await.unwrap();
    assert_eq!(target.name_lookups.load(Ordering::SeqCst), 0);
}

#[test]
fn a_failed_resolution_is_tried_again_a_backoff_after_its_end() {
    // As gRPC's resolver, whose next resolution after one that failed is a backoff after the end of
    // that one, where a subchannel connects again a backoff after the start of its last attempt.
    let start = Instant::now();
    let end = start + Duration::from_secs(6);
    let delay = Duration::from_secs(1);
    let resolve = ChannelFailure::Resolve {
        name: "spark.test:15002".to_string(),
        failure: ResolveFailure::Timeout,
    };
    assert_eq!(next_attempt(start, end, delay, &resolve), end + delay);
    assert!(failed_to_resolve(&resolve));
    assert_eq!(
        next_attempt(start, end, delay, &ChannelFailure::NoAddress),
        start + delay
    );
}
