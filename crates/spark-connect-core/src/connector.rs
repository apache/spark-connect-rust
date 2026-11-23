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

//! The connections of a channel, made as grpcio makes those of PySpark's channel.
//!
//! A channel has one connection at a time, which tonic asks this connector for when a call needs
//! one: lazily, at the first call, and again once the connection it had is gone. The connector
//! keeps the connectivity state of the channel, as gRPC's channel does:
//!
//! - A connection is made to the addresses of the host (or of the proxy) in the order of their
//!   resolution, as gRPC's `pick_first` with its happy eyeballs: the attempt to an address starts
//!   250 ms after the previous one while that one is not done, or at once when it fails, each
//!   with a deadline of its own, 20 s from its start. The first that completes is the connection,
//!   and the others are dropped.
//! - An attempt completes when the server's HTTP/2 settings have arrived, within its deadline:
//!   after the TCP connection, the tunnel of a proxy, and the TLS handshake, which the connector
//!   makes itself, so that the settings of a TLS connection are bounded too.
//! - The addresses are those of the system's resolver, within the deadline of an attempt; in the
//!   dialect of grpcio, `localhost` and the names under it are the loopback of both families, as
//!   c-ares, grpcio's resolver, gives them.
//! - While tonic waits for a connection, its calls wait, as gRPC's wait in `CONNECTING`. In the
//!   dialect of grpcio, once a connection has failed, the channel is in `TRANSIENT_FAILURE`: its
//!   calls fail at once with the error of the failure, while it connects again in the
//!   background, with gRPC's backoff (1 s, times 1.6 up to 120 s, with a jitter of 20 %), until a
//!   connection succeeds; the next call takes it. After 30 minutes without a call, or once no open
//!   client uses the channel ([`Connector::release`]), the channel stops connecting, as gRPC's
//!   idles or shuts down, and the next call connects again. A Rust client's channel connects
//!   again at each call that needs a connection, as tonic's does.
//! - A connection that ends (the server closed it, or its keepalive) leaves the channel idle: the
//!   next call waits for a new connection.
//!
//! The connector does the HTTP/2 preface of the client itself, so that it can wait for the
//! settings of the server before tonic has the connection, and hands the connection to hyper
//! through [`Handover`], which drops the preface that hyper sends again and the acknowledgment of
//! the connector's own settings, and advertises in hyper's settings the size of a header list that
//! grpcio's channel advertises, gRPC's hard limit of the metadata of a call, while hyper takes far
//! larger ones (see [`crate::metadata`]).
//!
//! The texts of the failures are those of gRPC's C core in the dialect of grpcio, which the
//! Python binding sets ([`set_grpcio_dialect`]), and a Rust client's own otherwise: see
//! [`Failure`].

use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::stream::{FuturesUnordered, StreamExt};
use hyper_util::rt::TokioIo;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::TcpStream;
use tokio::time::Instant;
use tokio_rustls::rustls;

use crate::channel::{ChannelTarget, TargetNames};
use crate::proxy::HttpProxy;

/// Whether the transport speaks as grpcio, where a Rust client would not: the texts of the
/// failures of its connections, and the `User-Agent` of the `CONNECT` request of a proxy.
static GRPCIO_DIALECT: AtomicBool = AtomicBool::new(false);

/// Make the transport speak as grpcio, the gRPC library of PySpark's client, where a Rust client
/// would not, for the Python binding: the failures of the connections have gRPC core's texts
/// (`failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:1: Failed to
/// connect to remote host: Connection refused`), and the `CONNECT` request of a proxy has
/// grpcio's `User-Agent: grpc-httpcli/0.0`. Without it, the failures have the texts of the Rust
/// client (`failed to connect to all addresses; last error: 127.0.0.1:1: tcp connect error:
/// Connection refused (os error 111)`), and the `CONNECT` request has no `User-Agent`. Of the
/// connections made from then on. Not a stable API.
#[doc(hidden)]
pub fn set_grpcio_dialect(grpcio: bool) {
    GRPCIO_DIALECT.store(grpcio, Ordering::Relaxed);
}

/// Whether the transport speaks as grpcio (see [`set_grpcio_dialect`]).
pub(crate) fn grpcio_dialect() -> bool {
    GRPCIO_DIALECT.load(Ordering::Relaxed)
}

/// The deadline of an attempt to an address, from its start, as grpcio's connect deadline
/// (`grpc.min_reconnect_backoff_ms`... its minimum connect timeout of 20 s): of the TCP
/// connection, the tunnel of a proxy, the TLS handshake and the settings of the server.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

/// How long the attempt to an address runs alone before the next one starts, as gRPC's
/// `grpc.happy_eyeballs_connection_attempt_delay_ms`.
const ATTEMPT_DELAY: Duration = Duration::from_millis(250);

/// gRPC's backoff of the connections of a channel that failed: its first delay, multiplier,
/// jitter and longest delay (`grpc.initial_reconnect_backoff_ms`, ...).
const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const BACKOFF_MULTIPLIER: f64 = 1.6;
const BACKOFF_JITTER: f64 = 0.2;
const MAX_BACKOFF: Duration = Duration::from_secs(120);

/// How long a channel that failed connects again without a call before it stops, as gRPC's
/// channel idles after `grpc.client_idle_timeout_ms`.
const IDLE_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// The connection preface of an HTTP/2 client (RFC 9113, section 3.4).
const PREFACE: &[u8; 24] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

/// An empty SETTINGS frame, which the connector's preface ends with.
const EMPTY_SETTINGS: [u8; 9] = [0, 0, 0, 4, 0, 0, 0, 0, 0];

/// The length of the header of an HTTP/2 frame (RFC 9113, section 4.1).
const FRAME_HEADER_LEN: usize = 9;

/// The type of an HTTP/2 SETTINGS frame, and its ACK flag.
const SETTINGS: u8 = 4;
const ACK: u8 = 1;

/// The identifier of `SETTINGS_MAX_HEADER_LIST_SIZE` (RFC 9113, section 6.5.2), and the length
/// of an entry of a SETTINGS frame, of its identifier and its value.
const MAX_HEADER_LIST_SIZE: u16 = 6;
const SETTING_LEN: usize = 6;

/// The longest response of a proxy to a CONNECT that the connector reads, as hyper-util's tunnel.
const MAX_PROXY_RESPONSE: usize = 8192;

/// The `User-Agent` of grpcio's CONNECT requests, of its HTTP client (`GRPC_HTTPCLI_USER_AGENT`).
const GRPC_HTTPCLI_USER_AGENT: &str = "grpc-httpcli/0.0";

/// The roots that the TLS sessions of a channel trust.
#[derive(Clone, Copy)]
pub(crate) enum Roots<'a> {
    /// The certificates of this PEM only, as grpcio trusts those of the file of
    /// `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH` alone.
    Only(&'a [u8]),
    /// The roots of the system's store, and the certificates of this PEM too, if there is one,
    /// which the Python binding gives: grpcio's own bundle.
    System(Option<&'a [u8]>),
}

/// The TLS of a channel: the configuration of its sessions and the verifier of the certificates
/// of its servers, and the name of the server, which they are checked against.
#[derive(Clone)]
pub(crate) struct Tls {
    /// None of no root to trust: each attempt fails, as grpcio fails to create the handshaker of
    /// a session of a roots file without a certificate (see [`Failure::NoRoots`]).
    trust: Option<(
        Arc<rustls::ClientConfig>,
        Arc<rustls::client::WebPkiServerVerifier>,
    )>,
    server_name: rustls::pki_types::ServerName<'static>,
    /// The name that the certificates are checked against, as given, by which the connections of
    /// the process are told apart (see [`Target::key`]).
    name: String,
    /// How the name of the certificate of a server is checked.
    check: NameCheck,
}

/// How the name of the certificate of a server is checked: by webpki, of the name of the server
/// of a connection string, or as gRPC checks it, of the name of a target of gRPC (see
/// [`crate::tls_names`]).
#[derive(Clone, Debug)]
enum NameCheck {
    Webpki,
    /// Of the name that gRPC's TLS verifies, None of none, which no certificate is of.
    Grpc(Option<String>),
}

impl Tls {
    /// The TLS of a channel to `domain` (a host name or an IP address, without brackets), whose
    /// sessions trust `roots`. As tonic's TLS connector: TLS 1.2 and 1.3 of the default crypto
    /// provider, or of ring, and the ALPN of HTTP/2. The certificates of a PEM that cannot be
    /// parsed are skipped; of none to trust, the TLS is made all the same, and each attempt
    /// fails. The error is the message of what cannot be configured.
    pub(crate) fn new(domain: &str, roots: Roots<'_>) -> Result<Self, String> {
        use rustls::pki_types::ServerName;
        let (store, provider) = Self::store_of(roots);
        let server_name = ServerName::try_from(domain.to_string())
            .map_err(|error| format!("invalid server name {domain:?}: {error}"))?;
        Self::of_store(store, provider, server_name, domain, NameCheck::Webpki)
    }

    /// The TLS of a channel to a target of gRPC, of the name `name` that gRPC's TLS verifies of its
    /// authority (see `crate::channel::TargetNames::tls_name`), whose sessions trust `roots`. The
    /// chain of the certificate of a server is verified as of any TLS, and its name as gRPC checks
    /// it (see [`crate::tls_names`]): once the handshake is done, of the name, which fails the
    /// attempt of gRPC's error of it (`Hostname Verification Check failed`), so that nothing is sent
    /// to the server, and for each call, of the host of the calls (see
    /// [`Tls::of_the_call_host`]). Of no name, each attempt fails. SNI is sent of a name of DNS
    /// only, as rustls sends no other.
    pub(crate) fn of_target_name(name: Option<&str>, roots: Roots<'_>) -> Result<Self, String> {
        use rustls::pki_types::ServerName;
        let unbracketed = name.map(|name| name.trim_start_matches('[').trim_end_matches(']'));
        let server_name = unbracketed
            .and_then(|name| match name.parse::<std::net::IpAddr>() {
                Ok(ip) => Some(ServerName::IpAddress(ip.into())),
                Err(_) => ServerName::try_from(name.to_string()).ok(),
            })
            .unwrap_or(ServerName::IpAddress(
                std::net::IpAddr::from(std::net::Ipv4Addr::UNSPECIFIED).into(),
            ));
        let (store, provider) = Self::store_of(roots);
        Self::of_store(
            store,
            provider,
            server_name,
            name.unwrap_or_default(),
            NameCheck::Grpc(name.map(str::to_string)),
        )
    }

    /// Whether the certificate of the server of `stream`, a session of this TLS, is of the host of
    /// the calls, as gRPC's `SslCheckCallHost` checks it for each call, of the name of a target of
    /// gRPC (see [`crate::tls_names::PeerNames::of_call_host`]): else each call fails of
    /// `UNAUTHENTICATED`, which is not retried (see [`call_host_refused`]).
    fn of_the_call_host(&self, stream: &tokio_rustls::client::TlsStream<Socket>) -> bool {
        let NameCheck::Grpc(name) = &self.check else {
            return true;
        };
        let certificate = stream
            .get_ref()
            .1
            .peer_certificates()
            .and_then(|certificates| certificates.first());
        match (name, certificate) {
            (Some(name), Some(certificate)) => {
                crate::tls_names::PeerNames::of_certificate(certificate)
                    .is_some_and(|names| names.of_call_host(name))
            }
            _ => false,
        }
    }

    /// The roots of `roots`, and the crypto provider (see [`Tls::new`]).
    fn store_of(roots: Roots<'_>) -> (rustls::RootCertStore, Arc<rustls::crypto::CryptoProvider>) {
        use rustls::pki_types::{pem::PemObject, CertificateDer};
        let provider = rustls::crypto::CryptoProvider::get_default()
            .cloned()
            .unwrap_or_else(|| Arc::new(rustls::crypto::ring::default_provider()));
        let certificates = |pem: &[u8]| -> Vec<CertificateDer<'static>> {
            CertificateDer::pem_slice_iter(pem)
                .filter_map(Result::ok)
                .collect()
        };
        let mut store = rustls::RootCertStore::empty();
        match roots {
            Roots::Only(pem) => {
                store.add_parsable_certificates(certificates(pem));
            }
            Roots::System(bundle) => {
                let native = rustls_native_certs::load_native_certs();
                if !native.errors.is_empty() {
                    tracing::debug!(
                        "errors loading the native certificates: {:?}",
                        native.errors
                    );
                }
                store.add_parsable_certificates(native.certs);
                if let Some(bundle) = bundle {
                    store.add_parsable_certificates(certificates(bundle));
                }
            }
        }
        (store, provider)
    }

    /// The TLS of the roots `store` of `provider`, to the server of `server_name`, of the name
    /// `name`, checked of `check`.
    fn of_store(
        store: rustls::RootCertStore,
        provider: Arc<rustls::crypto::CryptoProvider>,
        server_name: rustls::pki_types::ServerName<'static>,
        name: &str,
        check: NameCheck,
    ) -> Result<Self, String> {
        let name = name.to_string();
        if store.is_empty() {
            return Ok(Self {
                trust: None,
                server_name,
                name,
                check,
            });
        }
        let webpki = rustls::client::WebPkiServerVerifier::builder_with_provider(
            Arc::new(store),
            provider.clone(),
        )
        .build()
        .map_err(|error| error.to_string())?;
        let mut config = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|error| error.to_string())?
            .with_webpki_verifier(webpki.clone())
            .with_no_client_auth();
        config.alpn_protocols = vec![b"h2".to_vec()];
        // As grpcio, which does not resume TLS sessions by default: no session of rustls's cache,
        // whose lock a thread of a parent of a `fork()` could hold in the child (see
        // `crate::fork`), is kept for another connection.
        config.resumption = rustls::client::Resumption::disabled();
        Ok(Self {
            trust: Some((Arc::new(config), webpki)),
            server_name,
            name,
            check,
        })
    }

    /// The configuration of one TLS session, whose verifier records how a chain of certificates
    /// that it does not trust is self-signed: None of no root to trust.
    fn session(&self) -> Option<(Arc<rustls::ClientConfig>, Arc<Verifier>)> {
        let (config, webpki) = self.trust.as_ref()?;
        let verifier = Arc::new(Verifier {
            inner: webpki.clone(),
            self_signed: Default::default(),
            check: self.check.clone(),
        });
        let mut config = (**config).clone();
        config
            .dangerous()
            .set_certificate_verifier(verifier.clone());
        Some((Arc::new(config), verifier))
    }
}

/// The verifier of the certificates of the servers, webpki's, which records whether a chain that
/// it did not trust is self-signed, as grpcio's error tells (`self signed certificate`).
#[derive(Debug)]
struct Verifier {
    inner: Arc<rustls::client::WebPkiServerVerifier>,
    /// How the last chain that the verifier did not trust is self-signed, if it is.
    self_signed: Mutex<Option<SelfSigned>>,
    /// How the name of the certificate is checked.
    check: NameCheck,
}

/// How a chain of certificates that is not trusted is self-signed, which BoringSSL's error tells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelfSigned {
    /// The certificate of the server is (`self signed certificate`).
    Leaf,
    /// The last certificate of the chain is (`self signed certificate in certificate chain`).
    Chain,
}

impl rustls::client::danger::ServerCertVerifier for Verifier {
    fn verify_server_cert(
        &self,
        end_entity: &rustls::pki_types::CertificateDer<'_>,
        intermediates: &[rustls::pki_types::CertificateDer<'_>],
        server_name: &rustls::pki_types::ServerName<'_>,
        ocsp_response: &[u8],
        now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        let verified = self.inner.verify_server_cert(
            end_entity,
            intermediates,
            server_name,
            ocsp_response,
            now,
        );
        if let Err(rustls::Error::InvalidCertificate(rustls::CertificateError::UnknownIssuer)) =
            &verified
        {
            let self_signed = if self_issued(end_entity) {
                Some(SelfSigned::Leaf)
            } else if intermediates.last().is_some_and(|last| self_issued(last)) {
                Some(SelfSigned::Chain)
            } else {
                None
            };
            *self.self_signed.lock().unwrap_or_else(|e| e.into_inner()) = self_signed;
        }
        let NameCheck::Grpc(name) = &self.check else {
            return verified;
        };
        // The chain first, as BoringSSL verifies it in the handshake, which webpki verifies before
        // the name, then the name, as gRPC checks it once the handshake is done.
        match verified {
            Ok(_)
            | Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::NotValidForName
                | rustls::CertificateError::NotValidForNameContext { .. },
            )) => {
                let of_the_name = name.as_deref().is_some_and(|name| {
                    crate::tls_names::PeerNames::of_certificate(end_entity)
                        .is_some_and(|names| names.of_target_name(name))
                });
                if of_the_name {
                    Ok(rustls::client::danger::ServerCertVerified::assertion())
                } else {
                    Err(rustls::Error::InvalidCertificate(
                        rustls::CertificateError::NotValidForName,
                    ))
                }
            }
            Err(error) => Err(error),
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls12_signature(message, cert, dss)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls13_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.inner.supported_verify_schemes()
    }
}

/// Whether the certificate `der` is self-issued: its issuer is its subject, as the DER of the two
/// names of its `TBSCertificate` (RFC 5280, section 4.1).
fn self_issued(der: &[u8]) -> bool {
    /// The tag, the value and the rest of the DER value of `input`.
    fn tlv(input: &[u8]) -> Option<(u8, &[u8], &[u8])> {
        let (&tag, rest) = input.split_first()?;
        let (&first, rest) = rest.split_first()?;
        let (len, rest) = if first < 0x80 {
            (usize::from(first), rest)
        } else {
            let count = usize::from(first & 0x7f);
            if count == 0 || count > 4 || rest.len() < count {
                return None;
            }
            let len = rest[..count]
                .iter()
                .fold(0usize, |len, &byte| (len << 8) | usize::from(byte));
            (len, &rest[count..])
        };
        (rest.len() >= len).then(|| (tag, &rest[..len], &rest[len..]))
    }
    let names = || -> Option<(&[u8], &[u8])> {
        let (_, certificate, _) = tlv(der)?;
        let (_, tbs, _) = tlv(certificate)?;
        let mut rest = tbs;
        // The explicit version, [0], is optional.
        let (tag, _, after) = tlv(rest)?;
        if tag == 0xa0 {
            rest = after;
        }
        let (_, _, rest) = tlv(rest)?; // serialNumber
        let (_, _, rest) = tlv(rest)?; // signature
        let (_, issuer, rest) = tlv(rest)?;
        let (_, _, rest) = tlv(rest)?; // validity
        let (_, subject, _) = tlv(rest)?;
        Some((issuer, subject))
    };
    names().is_some_and(|(issuer, subject)| issuer == subject)
}

/// What a channel connects to: the server, directly or through a proxy, and the TLS of its
/// sessions.
#[derive(Clone)]
pub(crate) struct Target {
    /// The host and the port that the connections are made to: the server's, or the proxy's. A
    /// host of an IPv6 address has no brackets.
    host: String,
    port: u16,
    /// The tunnel of a proxy to the server: the authority that its `CONNECT` names, and the
    /// credentials of the proxy.
    tunnel: Option<Tunnel>,
    tls: Option<Tls>,
    /// Whether the connections are of gRPC's local credentials (see
    /// [`crate::channel::Credentials::Local`]): of the loopback only.
    local: bool,
    /// Whether the connections speak as grpcio (see [`set_grpcio_dialect`]), as the transport
    /// did when the channel was made.
    grpcio: bool,
    /// The target of gRPC that the channel connects to in place of its host and its port (see
    /// [`Target::of_channel_target`]), if any.
    form: Option<ChannelTarget>,
    /// gRPC's names of the target of gRPC, if any, of which the tunnel of a proxy names the server.
    names: Option<TargetNames>,
    /// The addresses of the host, resolved by the tests rather than by the system's resolver.
    #[cfg(test)]
    addresses: Option<Vec<SocketAddr>>,
    /// Whether the lookup of the host never ends, of the tests of its deadline.
    #[cfg(test)]
    resolves_never: bool,
    /// The configuration of the resolver of a DNS server of the target, of the tests, in place of
    /// the system's.
    #[cfg(test)]
    dns_config: Option<crate::dns::Config>,
    /// The count of the lookups of the host, one for each pass of attempts, of the tests.
    #[cfg(test)]
    lookups: Arc<std::sync::atomic::AtomicUsize>,
}

/// The tunnel of a proxy to the server.
#[derive(Clone)]
struct Tunnel {
    /// The `host:port` of the server, as the endpoint names it, an IPv6 address in brackets.
    authority: String,
    /// The `Proxy-Authorization` of the credentials of the proxy, if it has any.
    authorization: Option<String>,
}

impl Target {
    /// What a channel to `host` (as the connection string has it, an IPv6 address in brackets)
    /// and `port` connects to: the server, or the server through `proxy`, with `tls` if it is
    /// TLS.
    pub(crate) fn new(host: &str, port: u16, proxy: Option<HttpProxy>, tls: Option<Tls>) -> Self {
        let (host, port, tunnel) = match proxy {
            Some(proxy) => (
                proxy.host,
                proxy.port,
                Some(Tunnel {
                    authority: format!("{host}:{port}"),
                    authorization: proxy.authorization,
                }),
            ),
            None => (unbracketed(host).to_string(), port, None),
        };
        Self {
            host,
            port,
            tunnel,
            tls,
            local: false,
            grpcio: grpcio_dialect(),
            form: None,
            names: None,
            #[cfg(test)]
            addresses: None,
            #[cfg(test)]
            resolves_never: false,
            #[cfg(test)]
            dns_config: None,
            #[cfg(test)]
            lookups: Default::default(),
        }
    }

    /// This target, of gRPC's local credentials when `local` (see
    /// [`crate::channel::Credentials::Local`]): its connections are of the loopback only, as
    /// gRPC's handshake of local credentials of TCP refuses another.
    pub(crate) fn of_local_credentials(mut self, local: bool) -> Self {
        self.local = local;
        self
    }

    /// This target, of the target of gRPC `form`, if any (see [`ChannelTarget`]), of gRPC's names
    /// `names` of it: the connections are made to the host and the port of a name, to the
    /// addresses or to the Unix sockets of the target in turn, or the calls of a target that gRPC
    /// refuses fail of its status (see [`refused_connection`]). The tunnel of a proxy names the
    /// server of gRPC's proxy mapper (`[127.0.0.1:1,127.0.0.1:2]:443`), or without names, the name
    /// or the addresses of the target; a socket of `unix:`, which gRPC does not proxy, and a target
    /// that gRPC refuses have none, and an abstract one is proxied as gRPC proxies it.
    pub(crate) fn of_channel_target(
        mut self,
        form: Option<ChannelTarget>,
        names: Option<TargetNames>,
    ) -> Self {
        match &form {
            Some(ChannelTarget::Name {
                host, port, name, ..
            }) => match &mut self.tunnel {
                Some(tunnel) => tunnel.authority = name.clone(),
                None => {
                    self.host = unbracketed(host).to_string();
                    self.port = *port;
                }
            },
            Some(ChannelTarget::Unresolvable { name, .. }) => {
                if let Some(tunnel) = &mut self.tunnel {
                    tunnel.authority = name.clone();
                }
            }
            Some(ChannelTarget::Addresses { authority, .. }) => {
                if let Some(tunnel) = &mut self.tunnel {
                    tunnel.authority = authority.clone();
                }
            }
            Some(ChannelTarget::Unix {
                abstract_names: false,
                ..
            })
            | Some(ChannelTarget::Refused { .. }) => {
                self.tunnel = None;
            }
            Some(ChannelTarget::Unix { .. }) | None => {}
        }
        if let (
            Some(tunnel),
            Some(TargetNames {
                proxied: Some((_, server)),
                ..
            }),
        ) = (&mut self.tunnel, &names)
        {
            tunnel.authority = server.clone();
        }
        self.form = form;
        self.names = names;
        self
    }

    /// What the connections of this target are made of, by which the channels of the process
    /// share them (see `crate::client`): the host and the port, the tunnel and its credentials,
    /// the name of the server of the TLS, whose roots are those of the process, local
    /// credentials, and the dialect.
    pub(crate) fn key(&self) -> TargetKey {
        TargetKey {
            host: self.host.clone(),
            port: self.port,
            tunnel: self
                .tunnel
                .as_ref()
                .map(|tunnel| (tunnel.authority.clone(), tunnel.authorization.clone())),
            tls: self.tls.as_ref().map(|tls| tls.name.clone()),
            local: self.local,
            grpcio: self.grpcio,
            form: self.form.clone(),
        }
    }

    /// What a new channel of the same server and TLS as this target connects to, through `proxy`,
    /// in the dialect of the transport now.
    pub(crate) fn renewed(&self, host: &str, port: u16, proxy: Option<HttpProxy>) -> Self {
        Self::new(host, port, proxy, self.tls.clone())
            .of_local_credentials(self.local)
            .of_channel_target(self.form.clone(), self.names.clone())
    }

    /// The name that gRPC resolves for this target, `host:port`, as its errors give it: of a name
    /// of a target of gRPC, the name as the target has it, but through a proxy, whose name is
    /// resolved.
    fn name(&self) -> String {
        match &self.form {
            Some(ChannelTarget::Name { name, .. })
            | Some(ChannelTarget::Unresolvable { name, .. })
                if self.tunnel.is_none() =>
            {
                return name.clone();
            }
            _ => {}
        }
        if self.host.contains(':') {
            format!("[{}]:{}", self.host, self.port)
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }

    /// The addresses of the target: of a target of gRPC, its addresses or its Unix sockets, or the
    /// failure of a name that gRPC's resolver does not resolve; else the addresses of the host (see
    /// [`Target::resolve_host`]).
    async fn resolve(&self) -> Result<Vec<Address>, ResolveFailure> {
        #[cfg(test)]
        self.lookups.fetch_add(1, Ordering::SeqCst);
        #[cfg(test)]
        if self.resolves_never {
            std::future::pending::<()>().await;
        }
        #[cfg(test)]
        if let Some(addresses) = &self.addresses {
            return Ok(addresses.iter().copied().map(Address::Ip).collect());
        }
        match &self.form {
            Some(ChannelTarget::Addresses { addresses, .. }) if self.tunnel.is_none() => {
                return Ok(addresses.iter().copied().map(Address::Ip).collect());
            }
            Some(ChannelTarget::Unresolvable { detail, .. }) if self.tunnel.is_none() => {
                return Err(ResolveFailure::Unparsable(detail.clone()));
            }
            Some(ChannelTarget::Unix {
                paths,
                abstract_names,
            }) if self.tunnel.is_none() => {
                return Ok(paths
                    .iter()
                    .map(|path| Address::Unix {
                        path: path.clone(),
                        abstract_name: *abstract_names,
                    })
                    .collect());
            }
            _ => {}
        }
        let addresses = self.resolve_host().await?;
        Ok(addresses.into_iter().map(Address::Ip).collect())
    }

    /// The addresses of the host, in the order of the system's resolver (RFC 6724), as gRPC's
    /// resolver sorts them; in the dialect of grpcio, as c-ares, gRPC's resolver, takes the host:
    /// an address of four decimal numbers, of leading zeros too, and an IPv6 address of a zone,
    /// are those addresses; `localhost` and the names under it are the loopback (see
    /// [`localhost_addresses`]); a name of a character other than `[A-Za-z0-9-._/*]`, of an empty
    /// label or of one of more than 63 characters, or longer than DNS takes, is not resolved
    /// (`Misformatted domain name`); and a name that the system's resolver would read as an
    /// address, as of `inet_aton` (`127.1`), is a name that it looks up.
    async fn resolve_host(&self) -> Result<Vec<SocketAddr>, ResolveFailure> {
        if let Ok(ip) = self.host.parse::<std::net::IpAddr>() {
            return Ok(vec![SocketAddr::new(ip, self.port)]);
        }
        if let Some(server) = self.dns_server() {
            return self.resolve_through(server).await;
        }
        let mut lookup = std::borrow::Cow::Borrowed(self.host.as_str());
        if self.grpcio {
            // As c-ares takes the name, a C string, up to its first NUL.
            let host = self.host.split('\0').next().unwrap_or_default();
            lookup = std::borrow::Cow::Borrowed(host);
            if let Some(ip) = crate::grpc_target::ares_numeric_v4(host) {
                return Ok(vec![SocketAddr::new(ip.into(), self.port)]);
            }
            if let Some(address) = crate::grpc_target::ipv6_literal(host, self.port) {
                return Ok(vec![address]);
            }
            if !is_localhost(host) {
                if !is_ares_name(host) {
                    return Err(ResolveFailure::BadName);
                }
                if read_as_an_address(host) {
                    lookup = std::borrow::Cow::Owned(format!("{host}."));
                }
            }
        }
        let resolved = match tokio::net::lookup_host((lookup.as_ref(), self.port)).await {
            Ok(addresses) => {
                let addresses: Vec<_> = addresses.collect();
                if addresses.is_empty() {
                    Err(ResolveFailure::NoData)
                } else {
                    Ok(addresses)
                }
            }
            Err(error) => Err(ResolveFailure::of(&error)),
        };
        if self.grpcio && is_localhost(self.host.split('\0').next().unwrap_or_default()) {
            return Ok(localhost_addresses(resolved.unwrap_or_default(), self.port));
        }
        resolved
    }

    /// The DNS server of the authority of the target of gRPC (`dns://<server>/<name>`), which
    /// its name is resolved through, if it has one: of none through the tunnel of a proxy, whose
    /// own name the system's resolver resolves, as gRPC's proxy mapper replaces the target of the
    /// channel with the proxy's.
    fn dns_server(&self) -> Option<SocketAddr> {
        match &self.form {
            Some(ChannelTarget::Name {
                server: Some(server),
                ..
            }) if self.tunnel.is_none() => Some(*server),
            _ => None,
        }
    }

    /// The addresses of the host through the DNS server `server`, as grpcio's channel resolves
    /// them (see [`crate::dns`]): of a C string, up to its first NUL; an address of four decimal
    /// numbers, of leading zeros too, and an IPv6 address of a zone, are those addresses, as c-ares
    /// takes them; else the name is looked up through the server, of the configuration of the
    /// system, as c-ares reads it for each lookup.
    async fn resolve_through(&self, server: SocketAddr) -> Result<Vec<SocketAddr>, ResolveFailure> {
        let host = self.host.split('\0').next().unwrap_or_default();
        if let Some(ip) = crate::grpc_target::ares_numeric_v4(host) {
            return Ok(vec![SocketAddr::new(ip.into(), self.port)]);
        }
        if let Some(address) = crate::grpc_target::ipv6_literal(host, self.port) {
            return Ok(vec![address]);
        }
        #[cfg(test)]
        let config = self
            .dns_config
            .clone()
            .unwrap_or_else(crate::dns::Config::system);
        #[cfg(not(test))]
        let config = crate::dns::Config::system();
        crate::dns::lookup(server, host, self.port, &config, ipv6_loopback_usable())
            .await
            .map_err(ResolveFailure::Server)
    }
}

/// What the connections of a [`Target`] are made of (see [`Target::key`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TargetKey {
    host: String,
    port: u16,
    tunnel: Option<(String, Option<String>)>,
    tls: Option<String>,
    local: bool,
    grpcio: bool,
    form: Option<ChannelTarget>,
}

/// Whether `host` is a name of the loopback of RFC 6761, `localhost` or a name under it, which
/// c-ares resolves to the loopback, whatever the system's resolver says.
fn is_localhost(host: &str) -> bool {
    let host = host.strip_suffix('.').unwrap_or(host).to_ascii_lowercase();
    host == "localhost" || host.ends_with(".localhost")
}

/// Whether c-ares takes `name` as a name of DNS that it asks for: of the characters
/// `[A-Za-z0-9-._/*]`, of labels of 1 to 63 characters but the root after a last `.`, and of no
/// more than the 255 bytes of a name of DNS.
fn is_ares_name(name: &str) -> bool {
    let of_name = |c: char| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_' | '/' | '*');
    if !name.chars().all(of_name) {
        return false;
    }
    let labels = name.strip_suffix('.').unwrap_or(name);
    if labels.is_empty() {
        return true;
    }
    let mut length = 1;
    for label in labels.split('.') {
        if label.is_empty() || label.len() > 63 {
            return false;
        }
        length += label.len() + 1;
    }
    length <= 255
}

/// Whether the system's resolver reads `name` as an IPv4 address, as `inet_aton` does, where
/// c-ares asks for it as a name.
#[cfg(unix)]
fn read_as_an_address(name: &str) -> bool {
    extern "C" {
        fn inet_aton(cp: *const libc::c_char, inp: *mut libc::in_addr) -> libc::c_int;
    }
    let Ok(name) = std::ffi::CString::new(name) else {
        return false;
    };
    let mut address = libc::in_addr { s_addr: 0 };
    // SAFETY: `name` is a C string, and `address` a place for the address, for the call.
    unsafe { inet_aton(name.as_ptr(), &mut address) != 0 }
}

/// Whether the system's resolver reads `name` as an IPv4 address: not known here.
#[cfg(not(unix))]
fn read_as_an_address(_name: &str) -> bool {
    false
}

/// The addresses of a name of the loopback (see [`is_localhost`]) as c-ares, grpcio's resolver,
/// gives them, of `resolved`, the system's: those of each family, or the loopback of a family
/// of none, as RFC 6761 asks, so that `localhost` of a hosts file of `127.0.0.1` only is `::1`
/// and `127.0.0.1`. gRPC sorts them by RFC 6724, which puts `::1` first, but where the system has
/// no IPv6 loopback, which it cannot connect from.
fn localhost_addresses(resolved: Vec<SocketAddr>, port: u16) -> Vec<SocketAddr> {
    let family = |v6: bool| -> Vec<SocketAddr> {
        let found: Vec<_> = resolved
            .iter()
            .filter(|address| address.is_ipv6() == v6)
            .copied()
            .collect();
        if !found.is_empty() {
            return found;
        }
        let loopback: std::net::IpAddr = if v6 {
            std::net::Ipv6Addr::LOCALHOST.into()
        } else {
            std::net::Ipv4Addr::LOCALHOST.into()
        };
        vec![SocketAddr::new(loopback, port)]
    };
    let (v6, v4) = (family(true), family(false));
    if ipv6_loopback_usable() {
        v6.into_iter().chain(v4).collect()
    } else {
        v4.into_iter().chain(v6).collect()
    }
}

/// Whether the system has an IPv6 loopback that a socket can be bound to, once per process: of no
/// lock, which a thread of a parent of a `fork()` would hold forever in the child.
fn ipv6_loopback_usable() -> bool {
    /// 0 until known, then 1 for an IPv6 loopback that is usable, else 2.
    static USABLE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
    match USABLE.load(Ordering::Relaxed) {
        0 => {
            let usable = std::net::UdpSocket::bind((std::net::Ipv6Addr::LOCALHOST, 0)).is_ok();
            USABLE.store(if usable { 1 } else { 2 }, Ordering::Relaxed);
            usable
        }
        known => known == 1,
    }
}

/// `host` without the brackets of an IPv6 address.
fn unbracketed(host: &str) -> &str {
    host.strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host)
}

/// Why the name of a target did not resolve, of the system's resolver.
#[derive(Debug, Clone)]
enum ResolveFailure {
    /// The name does not exist (`EAI_NONAME`).
    NotFound(String),
    /// The DNS servers could not be reached, or did not answer in time (`EAI_AGAIN`).
    NoServers(String),
    /// The DNS servers failed (`EAI_FAIL`).
    ServerFailure(String),
    /// The name has no address.
    NoData,
    /// The lookup did not end by the connect deadline, which bounds it as it bounds the rest of
    /// the attempt.
    Timeout,
    /// The name is not one that c-ares asks for (see [`is_ares_name`]).
    BadName,
    /// The name of a target of gRPC is not one that its resolver resolves, of the error that it
    /// gives (see [`ChannelTarget::Unresolvable`]).
    Unparsable(String),
    /// The lookup of the name through the DNS server of the target failed (see [`crate::dns`]).
    Server(crate::dns::Status),
    /// Any other error of the resolver, as it gives it.
    Other(String),
}

impl ResolveFailure {
    /// The failure of the lookup of a name that failed with `error`, the error of
    /// `getaddrinfo` that the standard library gives: its message, which ends with the text of
    /// `gai_strerror` of the code, or the code itself on Windows.
    fn of(error: &std::io::Error) -> Self {
        let message = error.to_string();
        #[cfg(unix)]
        {
            let ends_with = |code: libc::c_int| {
                // SAFETY: `gai_strerror` returns a static string for any code.
                let text = unsafe { std::ffi::CStr::from_ptr(libc::gai_strerror(code)) };
                message.ends_with(&*text.to_string_lossy())
            };
            if ends_with(libc::EAI_NONAME) {
                return Self::NotFound(message);
            }
            if ends_with(libc::EAI_AGAIN) {
                return Self::NoServers(message);
            }
            if ends_with(libc::EAI_FAIL) {
                return Self::ServerFailure(message);
            }
            #[cfg(any(target_os = "linux", target_os = "android"))]
            if ends_with(libc::EAI_NODATA) {
                return Self::NoData;
            }
        }
        #[cfg(windows)]
        match error.raw_os_error() {
            // WSAHOST_NOT_FOUND, WSATRY_AGAIN, WSANO_RECOVERY, WSANO_DATA.
            Some(11001) => return Self::NotFound(message),
            Some(11002) => return Self::NoServers(message),
            Some(11003) => return Self::ServerFailure(message),
            Some(11004) => return Self::NoData,
            _ => {}
        }
        Self::Other(message)
    }

    /// The text of c-ares, the resolver of grpcio, for this failure.
    fn ares_text(&self) -> &str {
        match self {
            Self::NotFound(_) => "Domain name not found",
            Self::NoServers(_) => "Could not contact DNS servers",
            Self::ServerFailure(_) => "DNS server returned general failure",
            Self::NoData => "DNS server returned answer with no data",
            Self::Timeout => "Timeout while contacting DNS servers",
            Self::BadName => "Misformatted domain name",
            Self::Server(status) => status.text(),
            Self::Unparsable(detail) | Self::Other(detail) => detail,
        }
    }

    /// The text of the Rust client for this failure: the resolver's.
    fn rust_text(&self) -> &str {
        match self {
            Self::NotFound(message)
            | Self::NoServers(message)
            | Self::ServerFailure(message)
            | Self::Unparsable(message)
            | Self::Other(message) => message,
            Self::NoData => "the name has no address",
            Self::Timeout => "timed out",
            Self::BadName => "misformatted domain name",
            Self::Server(status) => status.text(),
        }
    }
}

/// The part of the connection to an address where it failed, which grpcio's errors tell apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// The `CONNECT` request of the proxy.
    Proxy,
    /// The TLS handshake.
    Tls,
    /// The HTTP/2 settings of the server.
    Settings,
}

impl Phase {
    /// The phase as the Rust client's texts name it.
    fn rust_name(self) -> &'static str {
        match self {
            Self::Proxy => "proxy tunnel",
            Self::Tls => "tls handshake",
            Self::Settings => "http2 settings",
        }
    }
}

/// An I/O error, as grpcio's texts and the Rust client's give it: its kind, its OS error number,
/// if it has one, and its text.
#[derive(Debug, Clone)]
struct IoFailure {
    errno: Option<i32>,
    text: String,
}

impl IoFailure {
    fn of(error: &std::io::Error) -> Self {
        Self {
            errno: error.raw_os_error(),
            text: error.to_string(),
        }
    }

    /// The text of the OS error, as `strerror` gives it, without the `(os error N)` that Rust
    /// adds.
    fn strerror(&self) -> String {
        match self.errno {
            Some(errno) => {
                let text = std::io::Error::from_raw_os_error(errno).to_string();
                match text.rfind(" (os error ") {
                    Some(end) => text[..end].to_string(),
                    None => text,
                }
            }
            None => self.text.clone(),
        }
    }

    /// The text of a failed read in gRPC's core: `recvmsg:Connection reset by peer (104)`.
    fn recvmsg(&self) -> String {
        match self.errno {
            Some(errno) => format!("recvmsg:{} ({errno})", self.strerror()),
            None => format!("recvmsg:{}", self.text),
        }
    }

    /// The text of a failed write in gRPC's core: `sendmsg: Broken pipe (32)`.
    fn sendmsg(&self) -> String {
        match self.errno {
            Some(errno) => format!("sendmsg: {} ({errno})", self.strerror()),
            None => format!("sendmsg: {}", self.text),
        }
    }
}

/// Why the connection to one address failed.
#[derive(Debug, Clone)]
enum Failure {
    /// The connection failed once it was started, as when it is refused.
    Connect(IoFailure),
    /// The connection failed as it was started, of the error of `connect`, as of a Unix socket
    /// of no file or a network that is not reachable.
    ConnectNow(IoFailure),
    /// The TCP connection was not made by the deadline.
    ConnectTimeout,
    /// The peer closed the connection in `phase`.
    Closed(Phase),
    /// A read failed in `phase`.
    Read(Phase, IoFailure),
    /// A write failed in `phase`.
    Write(Phase, IoFailure),
    /// `phase` did not end by the deadline.
    Timeout(Phase),
    /// The proxy answered the CONNECT with this status, which is not a 2xx.
    ProxyStatus(u16),
    /// The proxy did not answer in HTTP: the character that grpcio's parser expected first.
    ProxyNotHttp(char),
    /// The response of the proxy is longer than [`MAX_PROXY_RESPONSE`].
    ProxyResponseTooLong,
    /// The TLS handshake failed with this error of rustls, and the verifier's account of a
    /// chain that it did not trust.
    Tls(rustls::Error, Option<SelfSigned>),
    /// The server did not select HTTP/2 by ALPN.
    NoAlpn,
    /// The first frame of the server is not SETTINGS: its type, and the status of a response of
    /// HTTP/1.x, as grpcio tells it.
    NotSettings(u8, Option<u16>),
    /// The TLS of the channel has no root to trust, as of a roots file without a certificate,
    /// with which grpcio fails to create the handshaker of each session.
    NoRoots,
    /// The local address of a connection of gRPC's local credentials is not of the loopback,
    /// which their handshake refuses.
    NotLoopback,
}

impl Failure {
    /// The code and the text of this failure of a connection to `address` in gRPC's core, as
    /// grpcio's `last error` gives them (`UNKNOWN`, `Failed to connect to remote host: Connection
    /// refused`).
    fn grpc(&self, address: &Address) -> (&'static str, String) {
        match self {
            Self::Connect(error) => (
                "UNKNOWN",
                format!("Failed to connect to remote host: {}", error.strerror()),
            ),
            Self::ConnectNow(error) => (
                "FAILED_PRECONDITION",
                format!(
                    "connect failed: addr: {} error: {}",
                    grpc_address(address),
                    error.strerror()
                ),
            ),
            Self::ConnectTimeout => ("UNKNOWN", "tcp handshaker shutdown".to_string()),
            Self::Closed(Phase::Tls) => (
                "UNAVAILABLE",
                "Handshake read failed (Socket closed)".to_string(),
            ),
            Self::Closed(_) => ("UNAVAILABLE", "Socket closed".to_string()),
            Self::Read(Phase::Tls, error) => (
                "UNAVAILABLE",
                format!("Handshake read failed ({})", error.recvmsg()),
            ),
            Self::Read(_, error) => ("UNAVAILABLE", error.recvmsg()),
            Self::Write(Phase::Tls, error) => (
                "UNAVAILABLE",
                format!("Handshake write failed ({})", error.sendmsg()),
            ),
            Self::Write(_, error) => (
                "UNAVAILABLE",
                format!(
                    "Delayed close due to in-progress write ({})",
                    error.sendmsg()
                ),
            ),
            Self::Timeout(Phase::Proxy) => ("UNAVAILABLE", "Endpoint closing".to_string()),
            Self::Timeout(Phase::Tls) => (
                "UNAVAILABLE",
                "Handshake read failed (Endpoint closing)".to_string(),
            ),
            Self::Timeout(Phase::Settings) => (
                "UNKNOWN",
                "connection attempt timed out before receiving SETTINGS frame".to_string(),
            ),
            Self::ProxyStatus(status) => (
                "UNKNOWN",
                format!("HTTP proxy returned response code {status}"),
            ),
            Self::ProxyNotHttp(expected) => ("UNKNOWN", format!("Expected '{expected}'")),
            Self::ProxyResponseTooLong => ("UNKNOWN", "Response too long".to_string()),
            Self::Tls(error, self_signed) => ("UNKNOWN", tls_text(error, *self_signed)),
            Self::NoAlpn => (
                "UNKNOWN",
                "Cannot check peer: missing selected ALPN property.".to_string(),
            ),
            Self::NotSettings(frame_type, http1) => {
                let mut text = format!(
                    "Failed parsing HTTP/2 (Expected SETTINGS frame as the first frame, got \
                     frame type {frame_type})"
                );
                if let Some(status) = http1 {
                    text.push_str(&format!(
                        " (Trying to connect an http1.x server (HTTP status {status}))"
                    ));
                }
                ("INTERNAL", text)
            }
            Self::NoRoots => (
                "UNKNOWN",
                "Failed to create security handshaker.".to_string(),
            ),
            Self::NotLoopback => (
                "UNKNOWN",
                "Endpoint is neither UDS or TCP loopback address.".to_string(),
            ),
        }
    }

    /// The text of this failure of a connection to `address` of the Rust client.
    fn rust(&self, address: &Address) -> String {
        let kind = match address {
            Address::Ip(_) => "tcp",
            Address::Unix { .. } => "unix",
        };
        match self {
            Self::Connect(error) | Self::ConnectNow(error) => {
                format!("{kind} connect error: {}", error.text)
            }
            Self::ConnectTimeout => format!("{kind} connect error: timed out"),
            Self::Closed(phase) => format!("{}: the connection was closed", phase.rust_name()),
            Self::Read(phase, error) | Self::Write(phase, error) => {
                format!("{}: {}", phase.rust_name(), error.text)
            }
            Self::Timeout(Phase::Proxy) => "the proxy did not open the tunnel in time".to_string(),
            Self::Timeout(Phase::Tls) => "tls handshake: timed out".to_string(),
            Self::Timeout(Phase::Settings) => {
                "connection attempt timed out before receiving SETTINGS frame".to_string()
            }
            Self::ProxyStatus(407) => "the proxy requires authentication (407)".to_string(),
            Self::ProxyStatus(status) => format!("the proxy refused the tunnel ({status})"),
            Self::ProxyNotHttp(_) => "the proxy did not answer in HTTP".to_string(),
            Self::ProxyResponseTooLong => "the response of the proxy is too long".to_string(),
            Self::Tls(error, _) => error.to_string(),
            Self::NoAlpn => "HTTP/2 was not negotiated.".to_string(),
            Self::NotSettings(frame_type, _) => format!(
                "the server did not answer in HTTP/2: its first frame is of type {frame_type}, \
                 not SETTINGS"
            ),
            Self::NoRoots => "tls handshake: no root certificate to trust".to_string(),
            Self::NotLoopback => {
                "local credentials: the connection is not of the loopback".to_string()
            }
        }
    }

    /// BoringSSL's error of a TLS handshake that this failure is, which grpcio logs: none of a
    /// certificate of another name, which grpc checks itself once the handshake is done.
    fn tls_handshake_error(&self) -> Option<String> {
        match self {
            Self::Tls(error, _) if is_not_valid_for_name(error) => None,
            Self::Tls(error, self_signed) => Some(boringssl_error(error, *self_signed)),
            _ => None,
        }
    }
}

/// Whether `error` is of a certificate that is not of the name of the server.
fn is_not_valid_for_name(error: &rustls::Error) -> bool {
    matches!(
        error,
        rustls::Error::InvalidCertificate(
            rustls::CertificateError::NotValidForName
                | rustls::CertificateError::NotValidForNameContext { .. }
        )
    )
}

/// The text of a TLS handshake that failed with `error` in gRPC's core, of BoringSSL's error:
/// `Tls handshake failed (TSI_PROTOCOL_FAILURE): SSL_ERROR_SSL: error:1000007d:SSL
/// routines:OPENSSL_internal:CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate:
/// OK`, but of a certificate that is not of the name of the server, which grpc checks itself once
/// the handshake is done.
fn tls_text(error: &rustls::Error, self_signed: Option<SelfSigned>) -> String {
    if is_not_valid_for_name(error) {
        return "Custom verification check failed with error: UNAUTHENTICATED: Hostname \
                Verification Check failed."
            .to_string();
    }
    format!(
        "Tls handshake failed (TSI_PROTOCOL_FAILURE): {}: OK",
        boringssl_error(error, self_signed)
    )
}

/// BoringSSL's error of a TLS handshake that failed with `error` of rustls, as grpcio logs it:
/// `SSL_ERROR_SSL: error:1000007d:SSL routines:OPENSSL_internal:CERTIFICATE_VERIFY_FAILED: unable
/// to get local issuer certificate`, the packed error, its reason, and the text of the result of
/// the verification of the certificate of the server.
fn boringssl_error(error: &rustls::Error, self_signed: Option<SelfSigned>) -> String {
    use rustls::CertificateError as C;
    /// The text of `X509_verify_cert_error_string` of a handshake that verified no certificate.
    const NO_VERIFICATION: &str = "Invalid certificate verification context";
    let (reason_code, reason, verify): (u32, String, String) = match error {
        rustls::Error::InvalidCertificate(certificate) => {
            let verify = match certificate {
                C::UnknownIssuer => match self_signed {
                    Some(SelfSigned::Leaf) => "self signed certificate",
                    Some(SelfSigned::Chain) => "self signed certificate in certificate chain",
                    None => "unable to get local issuer certificate",
                },
                C::Expired | C::ExpiredContext { .. } => "certificate has expired",
                C::NotValidYet | C::NotValidYetContext { .. } => "certificate is not yet valid",
                C::Revoked => "certificate revoked",
                C::InvalidPurpose | C::InvalidPurposeContext { .. } => {
                    "unsupported certificate purpose"
                }
                C::BadSignature | C::UnsupportedSignatureAlgorithmContext { .. } => {
                    "certificate signature failure"
                }
                _ => "certificate verify failed",
            };
            (
                125,
                "CERTIFICATE_VERIFY_FAILED".to_string(),
                verify.to_string(),
            )
        }
        rustls::Error::AlertReceived(alert) => {
            let code = u8::from(*alert);
            let name = format!("{alert:?}")
                .chars()
                .flat_map(|c| {
                    let upper = c.is_ascii_uppercase();
                    (upper.then_some('_').into_iter())
                        .chain(std::iter::once(c.to_ascii_uppercase()))
                })
                .collect::<String>();
            let name = name.trim_start_matches('_');
            // BoringSSL's names of the alerts of SSLv3, and of TLS 1.0 and later.
            let prefix = if matches!(code, 0 | 10 | 20 | 30 | 40..=47) {
                "SSLV3_ALERT_"
            } else {
                "TLSV1_ALERT_"
            };
            (
                1000 + u32::from(code),
                format!("{prefix}{name}"),
                NO_VERIFICATION.to_string(),
            )
        }
        rustls::Error::NoApplicationProtocol => (
            1120,
            "TLSV1_ALERT_NO_APPLICATION_PROTOCOL".to_string(),
            NO_VERIFICATION.to_string(),
        ),
        rustls::Error::InvalidMessage(_) => (
            247,
            "WRONG_VERSION_NUMBER".to_string(),
            NO_VERIFICATION.to_string(),
        ),
        other => (
            0,
            other.to_string().replace(' ', "_").to_ascii_uppercase(),
            NO_VERIFICATION.to_string(),
        ),
    };
    format!(
        "SSL_ERROR_SSL: error:{:08x}:SSL routines:OPENSSL_internal:{reason}: {verify}",
        0x1000_0000 | reason_code
    )
}

/// An address that a connection is made to: of TCP, or of a Unix socket, of its path or of its
/// abstract name.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Address {
    Ip(SocketAddr),
    Unix { path: Vec<u8>, abstract_name: bool },
}

impl From<SocketAddr> for Address {
    fn from(address: SocketAddr) -> Self {
        Address::Ip(address)
    }
}

/// The address as the Rust client's texts give it: `127.0.0.1:1`, `[::1]:1`, `unix:/tmp/a.sock`,
/// `unix-abstract:name`.
impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Address::Ip(address) => write!(f, "{address}"),
            Address::Unix { .. } => f.write_str(&grpc_address(self)),
        }
    }
}

/// The URI of an address in gRPC's errors: `ipv4:127.0.0.1:1`, `ipv6:%5B::1%5D:1`,
/// `unix:/tmp/a.sock`, `unix-abstract:name`.
fn grpc_address(address: &Address) -> String {
    let address = match address {
        Address::Ip(address) => address,
        Address::Unix {
            path,
            abstract_name,
        } => {
            let scheme = if *abstract_name {
                "unix-abstract"
            } else {
                "unix"
            };
            return format!("{scheme}:{}", crate::grpc_target::percent_encode_path(path));
        }
    };
    match address {
        SocketAddr::V4(address) => format!("ipv4:{address}"),
        SocketAddr::V6(address) => {
            let mut ip = address.ip().to_string();
            if address.scope_id() != 0 {
                ip.push_str(&format!("%25{}", address.scope_id()));
            }
            format!("ipv6:%5B{ip}%5D:{}", address.port())
        }
    }
}

/// Why a channel could not connect, which fails its calls: the name did not resolve, the target
/// has no address, or the attempts to all its addresses failed, of which the last one to fail is
/// given.
#[derive(Debug, Clone)]
enum ChannelFailure {
    Resolve {
        name: String,
        failure: ResolveFailure,
    },
    /// A target of gRPC of no address, as `ipv4:` (see [`ChannelTarget::Addresses`]).
    NoAddress,
    Addresses {
        address: Address,
        failure: Failure,
    },
    /// The connection to the server of a target that gRPC refuses failed (see
    /// [`refused_connection`]), which it does not.
    Refused(String),
}

impl ChannelFailure {
    /// The details of the failed calls of the channel, in the dialect of grpcio if `grpcio`.
    fn text(&self, grpcio: bool) -> String {
        match self {
            Self::Resolve {
                name,
                failure: ResolveFailure::Unparsable(detail),
            } if grpcio => {
                format!("errors resolving {name}: [field:hostname lookup error:{detail}]")
            }
            Self::Resolve { name, failure } if grpcio => format!(
                "errors resolving {name}: [field:hostname lookup error:address lookup failed for \
                 {name}: {}]",
                failure.ares_text()
            ),
            Self::Resolve { name, failure } => {
                format!(
                    "dns error: failed to look up {name}: {}",
                    failure.rust_text()
                )
            }
            Self::NoAddress if grpcio => "empty address list".to_string(),
            Self::NoAddress => "the target has no address".to_string(),
            Self::Addresses { address, failure } if grpcio => {
                let (code, text) = failure.grpc(address);
                format!(
                    "failed to connect to all addresses; last error: {code}: {}: {text}",
                    grpc_address(address)
                )
            }
            Self::Addresses { address, failure } => format!(
                "failed to connect to all addresses; last error: {address}: {}",
                failure.rust(address)
            ),
            Self::Refused(text) => text.clone(),
        }
    }
}

/// The error that the connector gives tonic for a channel that could not connect, whose text is
/// the details of the `UNAVAILABLE` status of each call that it fails (tonic gives a connector's
/// error the code `UNAVAILABLE` and its text as the message).
#[derive(Debug, Clone)]
pub(crate) struct ConnectError {
    text: Arc<str>,
}

impl std::fmt::Display for ConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

impl std::error::Error for ConnectError {}

impl ConnectError {
    /// The error of `failure`, in the dialect of grpcio if `grpcio`.
    fn of(failure: &ChannelFailure, grpcio: bool) -> Self {
        Self {
            text: failure.text(grpcio).into(),
        }
    }
}

/// The stream of a connection, of TCP or of TLS.
trait Io: AsyncRead + AsyncWrite + Send + Unpin {}
impl<T: AsyncRead + AsyncWrite + Send + Unpin> Io for T {}

/// A connection whose server has sent its settings: its stream, and the bytes of the server that
/// the connector read, which hyper is given first.
struct Ready {
    io: Box<dyn Io>,
    replay: Vec<u8>,
}

impl Ready {
    /// Whether the connection is still open: nothing read from it now is its end or an error.
    /// What it reads is kept for hyper.
    fn alive(&mut self) -> bool {
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut buffer = [0u8; 4096];
        loop {
            let mut read = ReadBuf::new(&mut buffer);
            match Pin::new(&mut self.io).poll_read(&mut cx, &mut read) {
                Poll::Pending => return true,
                Poll::Ready(Ok(())) if read.filled().is_empty() => return false,
                Poll::Ready(Ok(())) => self.replay.extend_from_slice(read.filled()),
                Poll::Ready(Err(_)) => return false,
            }
        }
    }
}

/// The stream of a connection: of TCP, or of a Unix socket.
enum Stream {
    Tcp(TcpStream),
    #[cfg(unix)]
    Unix(tokio::net::UnixStream),
}

impl Stream {
    #[cfg(unix)]
    fn as_raw_fd(&self) -> std::os::fd::RawFd {
        match self {
            Stream::Tcp(tcp) => std::os::fd::AsRawFd::as_raw_fd(tcp),
            Stream::Unix(unix) => std::os::fd::AsRawFd::as_raw_fd(unix),
        }
    }
}

impl AsyncRead for Stream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Stream::Tcp(stream) => Pin::new(stream).poll_read(cx, buf),
            #[cfg(unix)]
            Stream::Unix(stream) => Pin::new(stream).poll_read(cx, buf),
        }
    }
}

impl AsyncWrite for Stream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.get_mut() {
            Stream::Tcp(stream) => Pin::new(stream).poll_write(cx, buf),
            #[cfg(unix)]
            Stream::Unix(stream) => Pin::new(stream).poll_write(cx, buf),
        }
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[std::io::IoSlice<'_>],
    ) -> Poll<std::io::Result<usize>> {
        match self.get_mut() {
            Stream::Tcp(stream) => Pin::new(stream).poll_write_vectored(cx, bufs),
            #[cfg(unix)]
            Stream::Unix(stream) => Pin::new(stream).poll_write_vectored(cx, bufs),
        }
    }

    fn is_write_vectored(&self) -> bool {
        match self {
            Stream::Tcp(stream) => stream.is_write_vectored(),
            #[cfg(unix)]
            Stream::Unix(stream) => stream.is_write_vectored(),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Stream::Tcp(stream) => Pin::new(stream).poll_flush(cx),
            #[cfg(unix)]
            Stream::Unix(stream) => Pin::new(stream).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Stream::Tcp(stream) => Pin::new(stream).poll_shutdown(cx),
            #[cfg(unix)]
            Stream::Unix(stream) => Pin::new(stream).poll_shutdown(cx),
        }
    }
}

/// The connection of an attempt, of the generation of the process that made it (see
/// [`crate::fork`]): its socket is one that a child of a `fork()` closes, and a drop of it in a
/// child leaves it as it is, rather than removing it from the epoll instance of the runtime of the
/// parent, which the child shares, as that would stop the parent from reading it.
struct Socket(crate::fork::ForkLocal<Stream>);

impl Socket {
    fn new(stream: Stream) -> Self {
        #[cfg(unix)]
        crate::fork::opened(stream.as_raw_fd());
        Self(crate::fork::ForkLocal::new(stream))
    }
}

impl Drop for Socket {
    fn drop(&mut self) {
        #[cfg(unix)]
        if self.0.is_current() {
            crate::fork::closing(self.0.as_raw_fd());
        }
    }
}

impl AsyncRead for Socket {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        Pin::new(&mut *self.get_mut().0).poll_read(cx, buf)
    }
}

impl AsyncWrite for Socket {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut *self.get_mut().0).poll_write(cx, buf)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[std::io::IoSlice<'_>],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut *self.get_mut().0).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.0.is_write_vectored()
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut *self.get_mut().0).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut *self.get_mut().0).poll_shutdown(cx)
    }
}

/// The connection to `address` of `target`, with the deadline `deadline`.
async fn connect_address(
    target: &Target,
    address: &Address,
    deadline: Instant,
) -> Result<Ready, Failure> {
    let (mut tcp, refused) = match address {
        Address::Ip(address) => {
            let tcp = match tokio::time::timeout_at(deadline, connect_tcp(*address)).await {
                Ok(Ok(tcp)) => tcp,
                Ok(Err(failure)) => return Err(failure),
                Err(_) => return Err(Failure::ConnectTimeout),
            };
            // As the handshake of gRPC's local credentials of TCP, which reads the local address
            // of the connection, that to the proxy of a tunnel too, and not the address it was
            // made to, so that one to `0.0.0.0` is of the loopback.
            let refused = target.local && !tcp.local_addr().is_ok_and(is_of_the_loopback);
            // As tonic's connector of an endpoint, of its defaults.
            let _ = tcp.set_nodelay(true);
            (Socket::new(Stream::Tcp(tcp)), refused)
        }
        Address::Unix {
            path,
            abstract_name,
        } => {
            let unix =
                match tokio::time::timeout_at(deadline, connect_unix(path, *abstract_name)).await {
                    Ok(Ok(unix)) => unix,
                    Ok(Err(failure)) => return Err(failure),
                    Err(_) => return Err(Failure::ConnectTimeout),
                };
            // gRPC's local credentials of TCP refuse a Unix socket.
            (unix, target.local)
        }
    };
    let mut replay = Vec::new();
    if let Some(tunnel) = &target.tunnel {
        let mut dropped = TunnelDropped(target.grpcio);
        let tunnel = open_tunnel(&mut tcp, tunnel, target.grpcio, deadline).await;
        dropped.0 = false;
        replay = tunnel?;
    }
    if refused {
        return Err(Failure::NotLoopback);
    }
    let mut io: Box<dyn Io> = match &target.tls {
        Some(tls) => {
            let stream = handshake(tls, tcp, deadline).await?;
            if !tls.of_the_call_host(&stream) {
                return call_host_refused().await;
            }
            Box::new(stream)
        }
        None => Box::new(tcp),
    };
    // Bytes of the server that the proxy's response came with are of the plaintext connection;
    // a TLS server does not speak first.
    let settings = read_settings(&mut io, replay, deadline).await?;
    Ok(Ready {
        io,
        replay: settings,
    })
}

/// The TCP connection to `address`: of [`Failure::ConnectNow`] when `connect` fails as it starts,
/// as gRPC's core tells it from one that fails later ([`Failure::Connect`]).
#[cfg(unix)]
async fn connect_tcp(address: SocketAddr) -> Result<TcpStream, Failure> {
    let now = |error: std::io::Error| Failure::ConnectNow(IoFailure::of(&error));
    let socket = socket2::Socket::new(
        socket2::Domain::for_address(address),
        socket2::Type::STREAM,
        Some(socket2::Protocol::TCP),
    )
    .map_err(now)?;
    socket.set_nonblocking(true).map_err(now)?;
    if let Err(error) = socket.connect(&address.into()) {
        if error.raw_os_error() != Some(libc::EINPROGRESS) {
            return Err(now(error));
        }
    }
    let later = |error: std::io::Error| Failure::Connect(IoFailure::of(&error));
    let tcp = TcpStream::from_std(std::net::TcpStream::from(socket)).map_err(later)?;
    // As tokio's connection, which is made once the socket is writable, of its error, if any.
    tcp.writable().await.map_err(later)?;
    match tcp.take_error() {
        Ok(None) => Ok(tcp),
        Ok(Some(error)) | Err(error) => Err(later(error)),
    }
}

/// The TCP connection to `address`.
#[cfg(not(unix))]
async fn connect_tcp(address: SocketAddr) -> Result<TcpStream, Failure> {
    TcpStream::connect(address)
        .await
        .map_err(|error| Failure::Connect(IoFailure::of(&error)))
}

/// The connection to the Unix socket of the path `path`, or of the abstract name `path` if
/// `abstract_name`, as gRPC's core makes it: of [`Failure::ConnectNow`] when `connect` fails, as
/// it does at once of a socket of no file or of no listener.
#[cfg(unix)]
async fn connect_unix(path: &[u8], abstract_name: bool) -> Result<Socket, Failure> {
    use std::os::unix::ffi::OsStrExt;
    let now = |error: std::io::Error| Failure::ConnectNow(IoFailure::of(&error));
    let mut name = Vec::with_capacity(path.len() + 1);
    if abstract_name {
        name.push(0);
    }
    name.extend_from_slice(path);
    let address = socket2::SockAddr::unix(std::ffi::OsStr::from_bytes(&name)).map_err(now)?;
    let socket =
        socket2::Socket::new(socket2::Domain::UNIX, socket2::Type::STREAM, None).map_err(now)?;
    socket.set_nonblocking(true).map_err(now)?;
    if let Err(error) = socket.connect(&address) {
        // A listener whose backlog is full takes the connection later, as gRPC waits for it.
        if !matches!(
            error.raw_os_error(),
            Some(libc::EINPROGRESS) | Some(libc::EAGAIN)
        ) {
            return Err(now(error));
        }
    }
    let later = |error: std::io::Error| Failure::Connect(IoFailure::of(&error));
    let unix = tokio::net::UnixStream::from_std(std::os::unix::net::UnixStream::from(socket))
        .map_err(later)?;
    unix.writable().await.map_err(later)?;
    match unix.take_error() {
        Ok(None) => Ok(Socket::new(Stream::Unix(unix))),
        Ok(Some(error)) | Err(error) => Err(later(error)),
    }
}

/// The connection to a Unix socket, which this system does not make.
#[cfg(not(unix))]
async fn connect_unix(_path: &[u8], _abstract_name: bool) -> Result<Socket, Failure> {
    Err(Failure::ConnectNow(IoFailure {
        errno: None,
        text: "Unix sockets are not supported".to_string(),
    }))
}

/// Whether `address`, the local address of a connection, is of the loopback as gRPC's local
/// credentials of TCP take it: `127.0.0.1` (`INADDR_LOOPBACK`), also mapped to IPv6, or `::1`.
fn is_of_the_loopback(address: SocketAddr) -> bool {
    match address.ip().to_canonical() {
        std::net::IpAddr::V4(ip) => ip == std::net::Ipv4Addr::LOCALHOST,
        std::net::IpAddr::V6(ip) => ip == std::net::Ipv6Addr::LOCALHOST,
    }
}

/// Logs, in the dialect of grpcio, an attempt that is dropped while its tunnel opens, as when
/// another address connects first: grpcio logs the end of the endpoint of its tunnel as a failure
/// of the tunnel (`HTTP proxy handshake with [unknown] failed: UNAVAILABLE: Endpoint closing`).
struct TunnelDropped(bool);

impl Drop for TunnelDropped {
    fn drop(&mut self) {
        if self.0 {
            log_proxy_failure(
                "HTTP proxy handshake with [unknown] failed: UNAVAILABLE: Endpoint closing"
                    .to_string(),
            );
        }
    }
}

/// Open the tunnel of `tunnel` on `tcp`, the connection to the proxy (RFC 9110, section 9.3.6),
/// by `deadline`: the bytes after the response of the proxy, which are the server's.
async fn open_tunnel(
    tcp: &mut Socket,
    tunnel: &Tunnel,
    grpcio: bool,
    deadline: Instant,
) -> Result<Vec<u8>, Failure> {
    let authority = &tunnel.authority;
    let mut request = format!("CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n");
    if grpcio {
        request.push_str(&format!("User-Agent: {GRPC_HTTPCLI_USER_AGENT}\r\n"));
    }
    if let Some(authorization) = &tunnel.authorization {
        request.push_str(&format!("Proxy-Authorization: {authorization}\r\n"));
    }
    request.push_str("\r\n");
    let exchange = async {
        tcp.write_all(request.as_bytes())
            .await
            .map_err(|error| Failure::Write(Phase::Proxy, IoFailure::of(&error)))?;
        let mut response = Vec::new();
        let mut buffer = [0u8; 1024];
        let end = loop {
            if let Some(end) = find(&response, b"\r\n\r\n") {
                break end + 4;
            }
            // As grpcio's parser, which fails at the first byte that is not of `HTTP/1.`.
            if let Some(expected) = not_http(&response) {
                return Err(Failure::ProxyNotHttp(expected));
            }
            if response.len() >= MAX_PROXY_RESPONSE {
                return Err(Failure::ProxyResponseTooLong);
            }
            let read = tcp
                .read(&mut buffer)
                .await
                .map_err(|error| Failure::Read(Phase::Proxy, IoFailure::of(&error)))?;
            if read == 0 {
                return Err(Failure::Closed(Phase::Proxy));
            }
            response.extend_from_slice(&buffer[..read]);
        };
        let status_line = response.split(|&b| b == b'\r').next().unwrap_or_default();
        let status = status_line
            .splitn(3, |&b| b == b' ')
            .nth(1)
            .and_then(|status| std::str::from_utf8(status).ok()?.parse::<u16>().ok());
        // As grpcio, any 2xx opens the tunnel.
        match status {
            Some(200..=299) => Ok(response.split_off(end)),
            Some(status) => Err(Failure::ProxyStatus(status)),
            None => Err(Failure::ProxyNotHttp('H')),
        }
    };
    match tokio::time::timeout_at(deadline, exchange).await {
        Ok(result) => result,
        Err(_) => Err(Failure::Timeout(Phase::Proxy)),
    }
}

/// The first byte of `response` that does not continue `HTTP/1.`, as the character expected
/// there, if there is one.
fn not_http(response: &[u8]) -> Option<char> {
    const START: &[u8] = b"HTTP/1.";
    response
        .iter()
        .zip(START)
        .find(|(got, expected)| got != expected)
        .map(|(_, &expected)| char::from(expected))
}

/// The offset of `needle` in `haystack`, if it is there.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// The TLS session of `tls` on `tcp`, made by `deadline`, which must select HTTP/2.
async fn handshake(
    tls: &Tls,
    tcp: Socket,
    deadline: Instant,
) -> Result<tokio_rustls::client::TlsStream<Socket>, Failure> {
    let Some((config, verifier)) = tls.session() else {
        return Err(Failure::NoRoots);
    };
    let connector = tokio_rustls::TlsConnector::from(config);
    let connecting = connector.connect(tls.server_name.clone(), tcp);
    let stream = match tokio::time::timeout_at(deadline, connecting).await {
        Err(_) => return Err(Failure::Timeout(Phase::Tls)),
        Ok(Ok(stream)) => stream,
        Ok(Err(error)) => {
            let self_signed = *verifier
                .self_signed
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            return Err(
                match error
                    .get_ref()
                    .and_then(|inner| inner.downcast_ref::<rustls::Error>())
                {
                    Some(rustls_error) => Failure::Tls(rustls_error.clone(), self_signed),
                    None if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                        Failure::Closed(Phase::Tls)
                    }
                    None => Failure::Read(Phase::Tls, IoFailure::of(&error)),
                },
            );
        }
    };
    if stream.get_ref().1.alpn_protocol() != Some(b"h2") {
        return Err(Failure::NoAlpn);
    }
    Ok(stream)
}

/// Send the client's HTTP/2 preface on `io`, and read the server's first frame, its SETTINGS, by
/// `deadline`, after `replay`, the bytes of the server already read: all that was read of the
/// server, which hyper is given first.
async fn read_settings(
    io: &mut Box<dyn Io>,
    mut replay: Vec<u8>,
    deadline: Instant,
) -> Result<Vec<u8>, Failure> {
    let exchange = async {
        // What the server sent or did before the preface is written: gRPC's transport reads the
        // end that it finds there, of a peer that closed at once (`Socket closed`), and its write
        // fails of a connection that the peer reset already.
        let mut preface = PREFACE.to_vec();
        preface.extend_from_slice(&EMPTY_SETTINGS);
        match pending_read(io) {
            Some(Ok(bytes)) if bytes.is_empty() => return Err(Failure::Closed(Phase::Settings)),
            Some(Ok(bytes)) => replay.extend_from_slice(&bytes),
            // A socket that is not connected, as of a Unix socket whose listener's backlog was
            // full: its read fails of `EINVAL`, where gRPC's write, which comes first, fails of
            // `ENOTCONN`, of which its error is.
            #[cfg(unix)]
            Some(Err(error)) if error.raw_os_error() == Some(libc::EINVAL) => {
                let written = io.write_all(&preface).await.err().unwrap_or(error);
                return Err(Failure::Write(Phase::Settings, IoFailure::of(&written)));
            }
            Some(Err(error)) => {
                return Err(Failure::Write(Phase::Settings, IoFailure::of(&error)));
            }
            None => {}
        }
        io.write_all(&preface)
            .await
            .map_err(|error| Failure::Write(Phase::Settings, IoFailure::of(&error)))?;
        io.flush()
            .await
            .map_err(|error| Failure::Write(Phase::Settings, IoFailure::of(&error)))?;
        let mut buffer = [0u8; 4096];
        loop {
            if replay.len() >= FRAME_HEADER_LEN {
                let frame_type = replay[3];
                if frame_type != SETTINGS {
                    let http1 = replay
                        .starts_with(b"HTTP/1.")
                        .then(|| http1_status(&replay))
                        .flatten();
                    return Err(Failure::NotSettings(frame_type, http1));
                }
                let length = u32::from_be_bytes([0, replay[0], replay[1], replay[2]]) as usize;
                if replay.len() >= FRAME_HEADER_LEN + length {
                    return Ok(std::mem::take(&mut replay));
                }
            }
            let read = io
                .read(&mut buffer)
                .await
                .map_err(|error| Failure::Read(Phase::Settings, IoFailure::of(&error)))?;
            if read == 0 {
                return Err(Failure::Closed(Phase::Settings));
            }
            replay.extend_from_slice(&buffer[..read]);
        }
    };
    match tokio::time::timeout_at(deadline, exchange).await {
        Ok(result) => result,
        Err(_) => Err(Failure::Timeout(Phase::Settings)),
    }
}

/// What `io` gives now without waiting: its bytes (none at its end) or its error, or None when
/// it would wait.
fn pending_read(io: &mut Box<dyn Io>) -> Option<std::io::Result<Vec<u8>>> {
    let waker = futures::task::noop_waker();
    let mut cx = Context::from_waker(&waker);
    let mut buffer = [0u8; 4096];
    let mut read = ReadBuf::new(&mut buffer);
    match Pin::new(io).poll_read(&mut cx, &mut read) {
        Poll::Pending => None,
        Poll::Ready(Ok(())) => Some(Ok(read.filled().to_vec())),
        Poll::Ready(Err(error)) => Some(Err(error)),
    }
}

/// The status of the HTTP/1.x response that `bytes` begin with, if they hold its status line.
fn http1_status(bytes: &[u8]) -> Option<u16> {
    let line = bytes.split(|&b| b == b'\r' || b == b'\n').next()?;
    std::str::from_utf8(line.splitn(3, |&b| b == b' ').nth(1)?)
        .ok()?
        .parse()
        .ok()
}

/// One pass of attempts to the addresses of `target`, as gRPC's happy eyeballs makes them: in
/// the order of the addresses, each [`ATTEMPT_DELAY`] after the previous one while that one is
/// not done, or at once when it fails, each with the deadline `per_address` from its start. The
/// first connection that completes, or the failure of the last attempt to fail.
async fn attempt(target: &Target, per_address: Duration) -> Result<Ready, ChannelFailure> {
    attempt_after(target, per_address, ATTEMPT_DELAY).await
}

/// As [`attempt`], of the attempt to an address starting `delay` after the previous one while
/// that one is not done.
async fn attempt_after(
    target: &Target,
    per_address: Duration,
    delay: Duration,
) -> Result<Ready, ChannelFailure> {
    // The lookup has the deadline of an attempt too, as tonic's connect timeout bounded it, but
    // through a DNS server, whose lookup ends of the timeout of gRPC's resolution itself.
    let resolved = if target.dns_server().is_some() {
        target.resolve().await
    } else {
        match tokio::time::timeout(per_address, target.resolve()).await {
            Ok(resolved) => resolved,
            Err(_) => Err(ResolveFailure::Timeout),
        }
    };
    let addresses = resolved.map_err(|failure| ChannelFailure::Resolve {
        name: target.name(),
        failure,
    })?;
    if addresses.is_empty() {
        return Err(ChannelFailure::NoAddress);
    }
    let mut started = 0;
    let mut pending = FuturesUnordered::new();
    let start = |address: Address| {
        let deadline = Instant::now() + per_address;
        async move {
            let result = connect_address(target, &address, deadline).await;
            (address, result)
        }
    };
    pending.push(start(addresses[0].clone()));
    started += 1;
    let mut next_start = Instant::now() + delay;
    loop {
        let more = started < addresses.len();
        tokio::select! {
            Some((address, result)) = pending.next() => match result {
                Ok(ready) => return Ok(ready),
                Err(failure) => {
                    log_failure(target.grpcio, &address, &failure);
                    if more {
                        pending.push(start(addresses[started].clone()));
                        started += 1;
                        next_start = Instant::now() + delay;
                    } else if pending.is_empty() {
                        // The last attempt to fail.
                        return Err(ChannelFailure::Addresses { address, failure });
                    }
                }
            },
            _ = tokio::time::sleep_until(next_start), if more => {
                pending.push(start(addresses[started].clone()));
                started += 1;
                next_start = Instant::now() + delay;
            }
        }
    }
}

/// How often the failure of the tunnel of a proxy is logged at most, as gRPC's core logs it, once
/// a minute, whatever the attempts in between, where it logs the failure of each TLS handshake.
const PROXY_FAILURE_LOG_INTERVAL: Duration = Duration::from_secs(60);

/// When the failure of the tunnel of a proxy was last logged, of any channel of the process, as
/// gRPC's core limits the line of its source: of each generation of the process (see
/// [`crate::fork`]), whose threads take the lock.
static PROXY_FAILURE_LOGGED: crate::fork::PerGeneration<Mutex<Option<std::time::Instant>>> =
    crate::fork::PerGeneration::new();

/// Whether the failure of a tunnel at `now` is logged, of the last time `last` that one was:
/// once in [`PROXY_FAILURE_LOG_INTERVAL`] at most.
fn proxy_failure_logged_at(
    last: &Mutex<Option<std::time::Instant>>,
    now: std::time::Instant,
) -> bool {
    let mut last = last.lock().unwrap_or_else(|e| e.into_inner());
    match *last {
        Some(at) if now.saturating_duration_since(at) < PROXY_FAILURE_LOG_INTERVAL => false,
        _ => {
            *last = Some(now);
            true
        }
    }
}

/// Log `message`, the failure of the tunnel of a proxy, as grpcio logs it: once a minute at most.
fn log_proxy_failure(message: String) {
    let logged = PROXY_FAILURE_LOGGED.get_or_init(|_| Mutex::new(None));
    if proxy_failure_logged_at(logged, std::time::Instant::now()) {
        crate::grpc_log::error(message);
    }
}

/// Log the failure of the attempt to `address`, as gRPC's core logs the failures of the TLS
/// handshake and of the tunnel of a proxy to stderr (see [`crate::grpc_log`]).
fn log_failure(grpcio: bool, address: &Address, failure: &Failure) {
    if !grpcio {
        return;
    }
    if let Some(error) = failure.tls_handshake_error() {
        crate::grpc_log::info(format!("Handshake failed with error {error}"));
    }
    if let Failure::NoRoots = failure {
        crate::grpc_log::error(
            "Handshaker factory creation failed with TSI_INVALID_ARGUMENT".to_string(),
        );
        crate::grpc_log::error("Update handshaker factory failed.".to_string());
    }
    if let Failure::ProxyStatus(_)
    | Failure::ProxyNotHttp(_)
    | Failure::ProxyResponseTooLong
    | Failure::Closed(Phase::Proxy)
    | Failure::Read(Phase::Proxy, _)
    | Failure::Write(Phase::Proxy, _)
    | Failure::Timeout(Phase::Proxy) = failure
    {
        let (code, text) = failure.grpc(address);
        // grpcio names an endpoint that is closing `[unknown]`, as a tunnel that timed out.
        let address = if matches!(failure, Failure::Timeout(Phase::Proxy)) {
            "[unknown]".to_string()
        } else {
            grpc_address(address)
        };
        log_proxy_failure(format!(
            "HTTP proxy handshake with {address} failed: {code}: {text}"
        ));
    }
}

/// The connectivity state of a channel, as gRPC's.
enum State {
    /// No connection, nor any attempt: the next call connects.
    Idle,
    /// A call is connecting.
    Connecting,
    /// tonic has the connection.
    Ready,
    /// A connection made in the background, which the next call takes.
    Held(Ready),
    /// The last connection failed: the calls fail at once with this, while the channel connects
    /// again in the background.
    TransientFailure(ChannelFailure),
}

/// What the connector of a channel shares with its connections in the background.
struct Shared {
    target: Target,
    state: Mutex<State>,
    /// When a call last asked for a connection, for [`IDLE_TIMEOUT`].
    last_call: Mutex<Instant>,
    /// How many open clients use the channel (see [`Connector::acquire`]).
    users: std::sync::atomic::AtomicUsize,
    /// Told when the last open client of the channel stops using it, which ends its connections
    /// in the background.
    shutdown: Arc<tokio::sync::Notify>,
}

impl Shared {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether no open client uses the channel.
    fn unused(&self) -> bool {
        self.users.load(Ordering::SeqCst) == 0
    }

    /// Stop connecting in the background, as a channel that no open client uses: the connection
    /// made there and the failure are dropped, and the next call connects.
    fn stop(&self) {
        let mut state = self.state();
        if matches!(*state, State::Held(_) | State::TransientFailure(_)) {
            *state = State::Idle;
        }
    }
}

/// The connector of a channel, which tonic calls for each connection it needs (see the module).
#[derive(Clone)]
pub(crate) struct Connector(Arc<Shared>);

impl Connector {
    pub(crate) fn new(target: Target) -> Self {
        Self(Arc::new(Shared {
            target,
            state: Mutex::new(State::Idle),
            last_call: Mutex::new(Instant::now()),
            users: Default::default(),
            shutdown: Default::default(),
        }))
    }

    /// An open client uses the channel, until it releases it ([`Connector::release`]).
    pub(crate) fn acquire(&self) {
        self.0.users.fetch_add(1, Ordering::SeqCst);
    }

    /// An open client stops using the channel: it is closed, or gone. Once no open client uses
    /// it, the channel stops connecting in the background and drops the connection that it made
    /// there, as grpcio's channel that is closed shuts its subchannels down; the next call of a
    /// client that opens again connects.
    pub(crate) fn release(&self) {
        if self.0.users.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.0.stop();
            self.0.shutdown.notify_waiters();
        }
    }
}

/// Resets a channel that is still connecting to idle when the call that connects drops its
/// attempt, so that the next call connects.
struct ConnectingGuard(Option<Arc<Shared>>);

impl Drop for ConnectingGuard {
    fn drop(&mut self) {
        if let Some(shared) = self.0.take() {
            let mut state = shared.state();
            if matches!(*state, State::Connecting) {
                *state = State::Idle;
            }
        }
    }
}

impl tonic::codegen::Service<tonic::codegen::http::Uri> for Connector {
    type Response = TokioIo<Handover>;
    type Error = ConnectError;
    type Future = tonic::codegen::BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _: tonic::codegen::http::Uri) -> Self::Future {
        let shared = self.0.clone();
        Box::pin(async move {
            if let Some(ChannelTarget::Refused { code, message }) = &shared.target.form {
                return match refused_connection(*code, message).await {
                    Ok(ready) => Ok(TokioIo::new(Handover::new(ready))),
                    Err(failure) => Err(ConnectError::of(&failure, shared.target.grpcio)),
                };
            }
            *shared.last_call.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
            {
                let mut state = shared.state();
                match std::mem::replace(&mut *state, State::Connecting) {
                    State::Held(mut ready) => {
                        if ready.alive() {
                            *state = State::Ready;
                            return Ok(TokioIo::new(Handover::new(ready)));
                        }
                    }
                    State::TransientFailure(failure) => {
                        let error = ConnectError::of(&failure, shared.target.grpcio);
                        *state = State::TransientFailure(failure);
                        return Err(error);
                    }
                    // A held connection that is gone, or none: connect.
                    _ => {}
                }
            }
            let mut guard = ConnectingGuard(Some(shared.clone()));
            let started = Instant::now();
            let result = attempt(&shared.target, CONNECT_TIMEOUT).await;
            guard.0 = None;
            let mut state = shared.state();
            match result {
                Ok(ready) => {
                    *state = State::Ready;
                    Ok(TokioIo::new(Handover::new(ready)))
                }
                // In the dialect of grpcio, as gRPC's channel: the calls fail at once from now on,
                // while the channel connects again in the background. A Rust client connects
                // again at each call that needs a connection, as tonic's channel does.
                Err(failure) if shared.target.grpcio && !shared.unused() => {
                    let error = ConnectError::of(&failure, shared.target.grpcio);
                    *state = State::TransientFailure(failure);
                    drop(state);
                    reconnect_in_background(&shared, started);
                    Err(error)
                }
                Err(failure) => {
                    *state = State::Idle;
                    Err(ConnectError::of(&failure, shared.target.grpcio))
                }
            }
        })
    }
}

/// The connection of a target that gRPC refuses (see [`ChannelTarget::Refused`]): to a server of
/// HTTP/2 in memory, which answers each call of the status of `code` and `message`, in the
/// trailers of a response of no message, so that the calls fail of the code and the message, as
/// gRPC's channel of the target fails them, and are not retried as a failure of a connection
/// (`UNAVAILABLE`) would be.
async fn refused_connection(code: i32, message: &str) -> Result<Ready, ChannelFailure> {
    let (client, server) = tokio::io::duplex(64 * 1024);
    tokio::spawn(serve_refusals(server, code, grpc_message(message)));
    let mut io: Box<dyn Io> = Box::new(client);
    let replay = read_settings(&mut io, Vec::new(), Instant::now() + CONNECT_TIMEOUT)
        .await
        .map_err(|failure| ChannelFailure::Refused(format!("{failure:?}")))?;
    Ok(Ready { io, replay })
}

/// The connection of a TLS session whose certificate is of the name of its target but not of the
/// host of its calls (see [`Tls::of_the_call_host`]), of a server in memory that fails each call
/// of gRPC's `UNAUTHENTICATED` (`call host does not match SSL server name`), as gRPC's check of the
/// host of each call fails it, of no retry, on a connection that its handshake made.
async fn call_host_refused() -> Result<Ready, Failure> {
    refused_connection(
        tonic::Code::Unauthenticated as i32,
        "call host does not match SSL server name",
    )
    .await
    .map_err(|_| Failure::Closed(Phase::Settings))
}

/// The server of [`refused_connection`] on `io`: each call is answered of the status `code` and
/// `message`, a `grpc-message` already encoded.
async fn serve_refusals(io: tokio::io::DuplexStream, code: i32, message: String) {
    let Ok(mut connection) = h2::server::handshake(io).await else {
        return;
    };
    while let Some(Ok((_, mut respond))) = connection.accept().await {
        let response = tonic::codegen::http::Response::builder()
            .status(200)
            .header("content-type", "application/grpc")
            .header("grpc-status", code.to_string())
            .header("grpc-message", message.as_str())
            .body(());
        if let Ok(response) = response {
            let _ = respond.send_response(response, true);
        }
    }
}

/// `message` as the `grpc-message` of gRPC encodes it: its bytes that are not printable ASCII,
/// and `%`, percent-encoded.
fn grpc_message(message: &str) -> String {
    message
        .bytes()
        .map(|b| {
            if (0x20..=0x7e).contains(&b) && b != b'%' {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

/// gRPC's backoff of the connections of a channel that failed.
struct Backoff {
    current: Duration,
    initial: bool,
}

impl Backoff {
    fn new() -> Self {
        Self {
            current: INITIAL_BACKOFF,
            initial: true,
        }
    }

    /// The delay from the start of an attempt to the start of the next one, as gRPC's
    /// `BackOff::NextAttemptDelay`: 1 s, then 1.6 times the previous one up to 120 s, each with a
    /// jitter of 20 %.
    fn next(&mut self) -> Duration {
        if self.initial {
            self.initial = false;
        } else {
            self.current = self.current.mul_f64(BACKOFF_MULTIPLIER).min(MAX_BACKOFF);
        }
        self.current
            .mul_f64(1.0 + BACKOFF_JITTER * (2.0 * random_fraction() - 1.0))
    }
}

/// A pseudo-random number in `[0, 1)`, for the jitter of the backoff, which needs no strength.
fn random_fraction() -> f64 {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|time| time.as_nanos())
            .unwrap_or_default(),
    );
    (hasher.finish() >> 11) as f64 / (1u64 << 53) as f64
}

/// Connect the channel of `shared`, whose attempt that started at `started` failed, again in the
/// background, until a connection succeeds, which the next call takes, or the channel is gone,
/// or no call came for [`IDLE_TIMEOUT`]. As gRPC's, each attempt starts a backoff after the start
/// of the previous one, and has the backoff as its deadline, but not less than
/// [`CONNECT_TIMEOUT`].
fn reconnect_in_background(shared: &Arc<Shared>, started: Instant) {
    let weak: Weak<Shared> = Arc::downgrade(shared);
    let shutdown = Arc::clone(&shared.shutdown);
    let mut backoff = Backoff::new();
    let mut next = started + backoff.next();
    // On the runtime of the channel, whose worker calls the connector.
    let spawn = |task| match tokio::runtime::Handle::try_current() {
        Ok(runtime) => drop(runtime.spawn(task)),
        Err(_) => drop(crate::runtime::get_runtime().spawn(task)),
    };
    spawn(async move {
        loop {
            // The wait for the next attempt, which ends once no open client uses the channel. A
            // notification of a release is checked against the clients that use it now: another
            // client may have opened since, whose failed call started this loop.
            loop {
                // Registered before the check of the users, so that a release after it is seen.
                let released = shutdown.notified();
                tokio::pin!(released);
                released.as_mut().enable();
                match weak.upgrade() {
                    Some(shared) if shared.unused() => return shared.stop(),
                    Some(_) => {}
                    None => return,
                }
                tokio::select! {
                    _ = tokio::time::sleep_until(next) => break,
                    _ = &mut released => {}
                }
            }
            let Some(shared) = weak.upgrade() else {
                return;
            };
            if shared.unused() {
                return shared.stop();
            }
            let last_call = *shared.last_call.lock().unwrap_or_else(|e| e.into_inner());
            if last_call.elapsed() >= IDLE_TIMEOUT {
                let mut state = shared.state();
                if matches!(*state, State::TransientFailure(_)) {
                    *state = State::Idle;
                }
                return;
            }
            let start = Instant::now();
            next = start + backoff.next();
            let per_address = (next - start).max(CONNECT_TIMEOUT);
            // The attempt, dropped once no open client uses the channel, and not of a notification
            // that another client's open followed.
            let attempting = attempt(&shared.target, per_address);
            tokio::pin!(attempting);
            let result = loop {
                let released = shutdown.notified();
                tokio::pin!(released);
                released.as_mut().enable();
                if shared.unused() {
                    return shared.stop();
                }
                tokio::select! {
                    result = &mut attempting => break result,
                    _ = &mut released => {}
                }
            };
            if shared.unused() {
                return shared.stop();
            }
            let mut state = shared.state();
            if !matches!(*state, State::TransientFailure(_)) {
                return;
            }
            match result {
                Ok(ready) => {
                    *state = State::Held(ready);
                    return;
                }
                Err(failure) => *state = State::TransientFailure(failure),
            }
        }
    });
}

/// A connection, as the connector gives it to hyper: the connector sent the client's preface and
/// read the server's settings already, so the preface that hyper writes is dropped, the bytes
/// that the connector read are given first, and the server's acknowledgment of the connector's
/// own settings, its first SETTINGS frame with the ACK flag, is dropped, as hyper sent no
/// settings that it acknowledges. The end of the connection, or an error of its reads, is an
/// I/O error of grpcio's text of it, which the errors of the calls that it ends give (see
/// [`stream_removed_in`]).
///
/// hyper's settings, the frame that it writes after its preface, advertise the size of a header
/// list of grpcio's channel, gRPC's hard limit of the metadata of a call, in the place of hyper's
/// own limit, which is far larger: a server that honours it, as Spark's server of grpc-java does,
/// resets the stream of a call whose metadata would pass it, as it does of grpcio's, rather than
/// sending it, and the client tells the metadata of a server that does not past the limit itself
/// (see [`crate::metadata`]).
pub(crate) struct Handover {
    io: Box<dyn Io>,
    /// The bytes read of the server that hyper has not been given.
    pending: Vec<u8>,
    /// How many bytes of `pending` are of whole frames that hyper can be given, while the
    /// acknowledgment is looked for.
    deliverable: usize,
    /// Whether the acknowledgment of the connector's settings is still to drop.
    looking_for_ack: bool,
    /// How many bytes of the preface that hyper writes are still to drop.
    preface_left: usize,
    /// The bytes of hyper's settings, the frame after its preface, that were written, as hyper
    /// wrote them, until the frame is whole: None then.
    settings_written: Option<Vec<u8>>,
}

impl Handover {
    fn new(ready: Ready) -> Self {
        let mut handover = Self {
            io: ready.io,
            pending: ready.replay,
            deliverable: 0,
            looking_for_ack: true,
            preface_left: PREFACE.len(),
            settings_written: Some(Vec::new()),
        };
        handover.scan();
        handover
    }

    /// Write `buf`, of hyper's settings, the frame after its preface: the bytes of the frame, of
    /// the size of a header list that grpcio's channel advertises (see [`Handover`]), and none of
    /// the frames after it, which hyper writes again. A frame that is not SETTINGS is written as
    /// it is.
    fn poll_write_settings(
        &mut self,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        let Some(written) = self.settings_written.as_ref() else {
            return Pin::new(&mut self.io).poll_write(cx, buf);
        };
        let at = |position: usize| match position.checked_sub(written.len()) {
            Some(index) => buf[index],
            None => written[position],
        };
        let known = written.len() + buf.len();
        let frame = if known < FRAME_HEADER_LEN {
            // Bytes of the header, which nothing rewrites.
            buf.to_vec()
        } else {
            if at(3) != SETTINGS {
                self.settings_written = None;
                return Pin::new(&mut self.io).poll_write(cx, buf);
            }
            let end = FRAME_HEADER_LEN + u32::from_be_bytes([0, at(0), at(1), at(2)]) as usize;
            let count = end.saturating_sub(written.len()).min(buf.len());
            let advertised = crate::metadata::ADVERTISED_HEADER_LIST_SIZE.to_be_bytes();
            (written.len()..written.len() + count)
                .map(|position| {
                    let Some(offset) = position.checked_sub(FRAME_HEADER_LEN) else {
                        return at(position);
                    };
                    let (entry, within) = (offset / SETTING_LEN, offset % SETTING_LEN);
                    let identifier = FRAME_HEADER_LEN + entry * SETTING_LEN;
                    if within >= 2
                        && u16::from_be_bytes([at(identifier), at(identifier + 1)])
                            == MAX_HEADER_LIST_SIZE
                    {
                        advertised[within - 2]
                    } else {
                        at(position)
                    }
                })
                .collect()
        };
        let sent = match Pin::new(&mut self.io).poll_write(cx, &frame) {
            Poll::Ready(Ok(sent)) => sent,
            other => return other,
        };
        let mut written = self.settings_written.take().unwrap_or_default();
        written.extend_from_slice(&buf[..sent]);
        let whole = written.len() >= FRAME_HEADER_LEN
            && written.len()
                >= FRAME_HEADER_LEN
                    + u32::from_be_bytes([0, written[0], written[1], written[2]]) as usize;
        if !whole {
            self.settings_written = Some(written);
        }
        Poll::Ready(Ok(sent))
    }

    /// Scan the whole frames of `pending` after `deliverable` for the acknowledgment of the
    /// connector's settings, which is removed, and make them deliverable. Any frame of the
    /// connection itself (of stream 0), such as SETTINGS, PING, GOAWAY, WINDOW_UPDATE, ALTSVC,
    /// ORIGIN or one of a type that hyper ignores, can come before it; a frame of a stream ends
    /// the scan, as the acknowledgment comes before the frames of the streams.
    fn scan(&mut self) {
        while self.looking_for_ack {
            let rest = &self.pending[self.deliverable..];
            if rest.len() < FRAME_HEADER_LEN {
                return;
            }
            let length = u32::from_be_bytes([0, rest[0], rest[1], rest[2]]) as usize;
            let (frame_type, flags) = (rest[3], rest[4]);
            let stream = u32::from_be_bytes([rest[5] & 0x7f, rest[6], rest[7], rest[8]]);
            if frame_type == SETTINGS && flags & ACK != 0 && length == 0 {
                self.pending
                    .drain(self.deliverable..self.deliverable + FRAME_HEADER_LEN);
                self.looking_for_ack = false;
                return;
            }
            if stream != 0 {
                self.looking_for_ack = false;
                return;
            }
            if rest.len() < FRAME_HEADER_LEN + length {
                return;
            }
            self.deliverable += FRAME_HEADER_LEN + length;
        }
    }

    /// How many bytes of `pending` hyper can be given now.
    fn ready_bytes(&self) -> usize {
        if self.looking_for_ack {
            self.deliverable
        } else {
            self.pending.len()
        }
    }
}

/// grpcio's text of the end of a connection: of the peer that closed it.
const SOCKET_CLOSED: &str = "Socket closed";

impl AsyncRead for Handover {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let this = &mut *self;
        loop {
            let ready = this.ready_bytes();
            if ready > 0 {
                let count = ready.min(buf.remaining());
                buf.put_slice(&this.pending[..count]);
                this.pending.drain(..count);
                this.deliverable = this.deliverable.saturating_sub(count);
                return Poll::Ready(Ok(()));
            }
            if !this.looking_for_ack && this.pending.is_empty() {
                // Read straight into hyper's buffer.
                let filled = buf.filled().len();
                return match Pin::new(&mut this.io).poll_read(cx, buf) {
                    Poll::Ready(Ok(())) if buf.filled().len() == filled && buf.remaining() > 0 => {
                        Poll::Ready(Err(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            SOCKET_CLOSED,
                        )))
                    }
                    Poll::Ready(Err(error)) => Poll::Ready(Err(read_error(error))),
                    other => other,
                };
            }
            let mut buffer = [0u8; 4096];
            let mut read = ReadBuf::new(&mut buffer);
            match Pin::new(&mut this.io).poll_read(cx, &mut read) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(error)) => return Poll::Ready(Err(read_error(error))),
                Poll::Ready(Ok(())) if read.filled().is_empty() => {
                    // The end: what is left goes to hyper, then the end.
                    this.looking_for_ack = false;
                    if this.pending.is_empty() {
                        return Poll::Ready(Err(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            SOCKET_CLOSED,
                        )));
                    }
                }
                Poll::Ready(Ok(())) => {
                    this.pending.extend_from_slice(read.filled());
                    this.scan();
                }
            }
        }
    }
}

/// A failed read of a connection, of grpcio's text of it (`recvmsg:Connection reset by peer
/// (104)`), which keeps its kind.
fn read_error(error: std::io::Error) -> std::io::Error {
    std::io::Error::new(error.kind(), IoFailure::of(&error).recvmsg())
}

impl AsyncWrite for Handover {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        let this = &mut *self;
        if this.preface_left > 0 {
            let start = PREFACE.len() - this.preface_left;
            let count = this.preface_left.min(buf.len());
            if buf[..count] != PREFACE[start..start + count] {
                return Poll::Ready(Err(std::io::Error::other(
                    "the client did not start with the HTTP/2 preface",
                )));
            }
            this.preface_left -= count;
            return Poll::Ready(Ok(count));
        }
        if this.settings_written.is_some() {
            return this.poll_write_settings(cx, buf);
        }
        Pin::new(&mut this.io).poll_write(cx, buf)
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[std::io::IoSlice<'_>],
    ) -> Poll<std::io::Result<usize>> {
        if self.preface_left > 0 || self.settings_written.is_some() {
            let first = bufs.iter().find(|buf| !buf.is_empty());
            return match first {
                Some(buf) => self.poll_write(cx, buf),
                None => Poll::Ready(Ok(0)),
            };
        }
        Pin::new(&mut self.io).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.io.is_write_vectored()
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.io).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.io).poll_shutdown(cx)
    }
}

/// The details of a call that the end of its connection or of its stream failed, as grpcio gives
/// them in its dialect, `grpcio`: `Stream removed (Socket closed)`, `Stream removed
/// (recvmsg:Connection reset by peer (104))`, `Stream removed (ping timeout)` of a keepalive that
/// timed out, and `Stream removed (RST_STREAM (Received RST_STREAM with error code 1))` of a
/// stream that the server reset, of the code of its reason. None for any other status, or out of
/// the dialect of grpcio.
pub(crate) fn stream_removed_in(status: &tonic::Status, grpcio: bool) -> Option<String> {
    if !grpcio {
        return None;
    }
    let mut errors =
        std::iter::successors(std::error::Error::source(status), |error| error.source());
    errors.find_map(|error| {
        if let Some(error) = error.downcast_ref::<hyper::Error>() {
            if error.is_timeout() {
                return Some("Stream removed (ping timeout)".to_string());
            }
        }
        if let Some(error) = error.downcast_ref::<h2::Error>() {
            if let Some(reason) = error
                .reason()
                .filter(|_| error.is_reset() && error.is_remote())
            {
                return Some(format!(
                    "Stream removed (RST_STREAM (Received RST_STREAM with error code {}))",
                    u32::from(reason)
                ));
            }
        }
        let io = error
            .downcast_ref::<std::io::Error>()
            .or_else(|| error.downcast_ref::<h2::Error>()?.get_io())?;
        let text = io.to_string();
        (text == SOCKET_CLOSED || text.starts_with("recvmsg:"))
            .then(|| format!("Stream removed ({text})"))
    })
}

#[cfg(test)]
mod tests;
