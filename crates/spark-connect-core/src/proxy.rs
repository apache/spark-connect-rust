//! The HTTP CONNECT proxy of a channel, chosen from the environment as grpc-python does.
//!
//! PySpark's client has no proxy setting of its own: grpcio's core (`HttpProxyMapper`) reads
//! the environment for every channel, so a client behind an egress proxy reaches the server by
//! an HTTP/1.1 `CONNECT` tunnel. This module selects the proxy with the same rules, so that the
//! Rust client connects wherever PySpark does:
//!
//! - the proxy is the first of `grpc_proxy`, `https_proxy` and `http_proxy` that is set, even
//!   empty (an empty value means no proxy, as does one that is not a proxy URI, with the error
//!   that grpcio logs for it, see [`crate::grpc_log`], and one that is not UTF-8, of which grpcio
//!   reads the bytes);
//! - only the lowercase names are read, as gRPC reads them on POSIX;
//! - only an `http` proxy is supported; another scheme, or no host, connects directly, with that
//!   error, for any host, as grpcio logs it for every channel;
//! - a host or port of the proxy that is not valid is an error of a host that the proxy applies
//!   to (where grpcio does not connect, but takes a port past 65535 modulo 65536 and connects to
//!   that port), and of no other: grpcio resolves the address of the proxy only for a host that
//!   it proxies, so a host that the no-proxy list excludes connects directly;
//! - the userinfo of the proxy URI, percent-decoded, is sent as `Proxy-Authorization: Basic`;
//! - a proxy without a port is on port 443;
//! - `no_grpc_proxy`, or else `no_proxy`, lists the hosts that are not proxied: a host matches
//!   an entry that is the host itself or a parent domain of it (a leading `.` is optional), or
//!   an IP host, as gRPC reads an address (an IPv6 one of its zone too), is in an entry's CIDR
//!   range; an entry of a port matches no host;
//! - a loopback host is proxied as any other, as gRPC proxies it, of the local address of the
//!   connection to the proxy, which gRPC's local credentials check.
//!
//! One difference is deliberate, of a channel of no credentials of the Python binding only: a
//! loopback host of a channel that sends its token in plaintext, of the rule of
//! [`crate::channel::ChannelBuilder::secure`] (a token to `localhost`), is never proxied, as no
//! local credentials keep the token from a proxy that is not on the machine.
//!
//! The tunnel only carries bytes: TLS is negotiated through it with the Spark server, against
//! the server's own name, so the proxy sees neither the token nor the data of a TLS channel.

use std::ffi::OsString;
use std::net::IpAddr;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use tonic::codegen::http::uri::Authority;

/// The default port of a proxy without one, as gRPC resolves the proxy name.
const DEFAULT_PROXY_PORT: u16 = 443;

/// The variables that name the proxy, in the order of precedence.
const PROXY_VARS: [&str; 3] = ["grpc_proxy", "https_proxy", "http_proxy"];

/// The variables that list the hosts that are not proxied, in the order of precedence.
const NO_PROXY_VARS: [&str; 2] = ["no_grpc_proxy", "no_proxy"];

/// The proxy settings of the environment: which proxy variable is set and to what, and the
/// no-proxy list.
pub(crate) struct ProxyEnv {
    proxy: Option<(&'static str, OsString)>,
    no_proxy: Option<OsString>,
}

/// Shows which variable names the proxy, but not its value: it can hold a password.
impl std::fmt::Debug for ProxyEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProxyEnv")
            .field("proxy_var", &self.proxy.as_ref().map(|(name, _)| *name))
            .field("no_proxy", &self.no_proxy)
            .finish()
    }
}

impl ProxyEnv {
    /// The proxy settings of the environment of the process.
    pub(crate) fn from_process() -> Self {
        Self::from_lookup(|name| std::env::var_os(name))
    }

    /// The proxy settings that `lookup` gives for each variable name.
    pub(crate) fn from_lookup(lookup: impl Fn(&str) -> Option<OsString>) -> Self {
        Self {
            proxy: PROXY_VARS
                .iter()
                .find_map(|name| lookup(name).map(|value| (*name, value))),
            no_proxy: NO_PROXY_VARS.iter().find_map(|name| lookup(name)),
        }
    }

    /// The proxy to reach `host` (as the connection string has it, an IPv6 address in
    /// brackets) through, or `None` to connect directly, of a loopback host directly too if
    /// `direct_loopback` (see the module). The error is a proxy address that cannot be connected
    /// to, of a host that the proxy applies to; it names the variable but not its value, which can
    /// hold a password.
    pub(crate) fn proxy_for(
        &self,
        host: &str,
        direct_loopback: bool,
    ) -> Result<Option<HttpProxy>, String> {
        let Some((name, value)) = self.proxy.as_ref() else {
            return Ok(None);
        };
        let Some(value) = value.to_str() else {
            tracing::warn!("{name} is not valid UTF-8; not using a proxy");
            return Ok(None);
        };
        if value.is_empty() {
            return Ok(None);
        }
        // As grpcio, which reads the proxy URI for every channel and logs one that is not a
        // proxy, whatever the host, and then resolves the address of the proxy only for a host
        // that it proxies: an address that is not valid is the error of no other host.
        let proxy = match HttpProxy::parse(value) {
            Ok(proxy) => Ok(proxy),
            Err(ProxyUriError::NotAProxy(reason)) => {
                crate::grpc_log::error(format!("{reason} in {name}; not using a proxy"));
                return Ok(None);
            }
            Err(ProxyUriError::Invalid(reason)) => Err(reason),
        };
        let host = host.trim_start_matches('[').trim_end_matches(']');
        let excluded = (direct_loopback && is_loopback(host))
            || self
                .no_proxy
                .as_ref()
                .and_then(|list| list.to_str())
                .is_some_and(|list| no_proxy_matches(list, host));
        if excluded {
            return Ok(None);
        }
        proxy
            .map(Some)
            .map_err(|reason| format!("Invalid HTTP proxy address in {name}: {reason}"))
    }
}

/// Why a proxy URI is not a proxy to connect through.
enum ProxyUriError {
    /// Not an HTTP proxy, which grpcio ignores: the channel connects directly.
    NotAProxy(String),
    /// An HTTP proxy at an address that grpcio cannot connect to.
    Invalid(String),
}

/// An HTTP proxy: its `host:port`, its host (an IPv6 address without brackets) and its port, and
/// the `Proxy-Authorization` value of its credentials.
pub(crate) struct HttpProxy {
    pub(crate) authority: String,
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) authorization: Option<String>,
}

/// Shows the address of the proxy, but not its credentials.
impl std::fmt::Debug for HttpProxy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpProxy")
            .field("authority", &self.authority)
            .field(
                "authorization",
                &self.authorization.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

impl HttpProxy {
    /// The proxy of the URI `uri`, which gRPC's `URI::Parse` would read the same way. The
    /// error says why it is not a proxy, without the URI, which can hold a password.
    fn parse(uri: &str) -> Result<Self, ProxyUriError> {
        use ProxyUriError::{Invalid, NotAProxy};
        let (scheme, rest) = match uri.find(':') {
            Some(offset) if offset > 0 => (&uri[..offset], &uri[offset + 1..]),
            _ => return Err(NotAProxy("no scheme in the proxy URI".to_string())),
        };
        if !scheme.starts_with(|c: char| c.is_ascii_alphabetic())
            || !scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        {
            return Err(NotAProxy("an invalid scheme in the proxy URI".to_string()));
        }
        let scheme = scheme.to_ascii_lowercase();
        if scheme != "http" {
            return Err(NotAProxy(format!(
                "the '{scheme}' scheme of the proxy URI is not supported"
            )));
        }
        let authority = match rest.strip_prefix("//") {
            Some(rest) => &rest[..rest.find(['/', '?', '#']).unwrap_or(rest.len())],
            None => "",
        };
        let (userinfo, host_port) = match authority.rfind('@') {
            Some(at) => (
                percent_decode(&authority[..at]),
                percent_decode(&authority[at + 1..]),
            ),
            None => (String::new(), percent_decode(authority)),
        };
        if host_port.is_empty() {
            return Err(NotAProxy("no host in the proxy URI".to_string()));
        }
        let (host, port) = split_host_port(&host_port);
        if host.is_empty() {
            return Err(Invalid("no host".to_string()));
        }
        let port = match port {
            None => DEFAULT_PROXY_PORT,
            Some(port) => port
                .parse::<u16>()
                .map_err(|_| Invalid("the port is not a number from 0 to 65535".to_string()))?,
        };
        let authority = if host.contains(':') {
            format!("[{host}]:{port}")
        } else {
            format!("{host}:{port}")
        };
        if authority.parse::<Authority>().is_err() {
            return Err(Invalid("the host is not valid".to_string()));
        }
        let authorization =
            (!userinfo.is_empty()).then(|| format!("Basic {}", STANDARD.encode(userinfo)));
        Ok(Self {
            authority,
            host: host.to_string(),
            port,
            authorization,
        })
    }
}

/// Whether `host` (an IPv6 address without brackets) is a loopback host, which a channel that sends
/// its token in plaintext never proxies (see the module). Wider than
/// [`crate::channel::ChannelBuilder::is_loopback`], which names only the hosts that a token is
/// sent to in plaintext or not.
fn is_loopback(host: &str) -> bool {
    // A trailing dot names the same host, and every name under `.localhost` is loopback
    // (RFC 6761). An IPv4-mapped IPv6 address is the IPv4 address it maps.
    let name = host.strip_suffix('.').unwrap_or(host).to_ascii_lowercase();
    name == "localhost"
        || name
            .strip_suffix(".localhost")
            .is_some_and(|rest| !rest.is_empty())
        || host
            .parse::<IpAddr>()
            .is_ok_and(|ip| ip.to_canonical().is_loopback())
}

/// `host` and `port` of `host_port`, as gRPC's `SplitHostPort`: an IPv6 address is in brackets
/// when it has a port, and a host with more than one `:` is an IPv6 address without one.
fn split_host_port(host_port: &str) -> (&str, Option<&str>) {
    if let Some(rest) = host_port.strip_prefix('[') {
        return match rest.split_once(']') {
            Some((host, "")) => (host, None),
            Some((host, port)) => (host, Some(port.strip_prefix(':').unwrap_or(port))),
            None => ("", None),
        };
    }
    match host_port.split_once(':') {
        Some((host, port)) if !port.contains(':') => (host, Some(port)),
        _ => (host_port, None),
    }
}

/// Whether the no-proxy list `list` excludes `host`, as gRPC's `AddressIncluded`, of the address
/// of the host as gRPC's `StringToSockaddr` reads it, of an IPv6 address of a zone too.
fn no_proxy_matches(list: &str, host: &str) -> bool {
    let joined = if host.contains(':') {
        format!("[{host}]:0")
    } else {
        format!("{host}:0")
    };
    let ip = crate::grpc_target::parse_host_port(&joined, None).map(|address| address.ip());
    list.split(',')
        .map(str::trim_ascii)
        .filter(|entry| !entry.is_empty())
        .any(|entry| {
            host_or_parent_domain(host, entry) || ip.is_some_and(|ip| in_cidr_range(ip, entry))
        })
}

/// Whether `entry` is `host` or a parent domain of it, ignoring case and a leading `.`.
/// gRPC compares an IP host as text too, so `0.1` matches `127.0.0.1`.
fn host_or_parent_domain(host: &str, entry: &str) -> bool {
    let entry = entry.strip_prefix('.').unwrap_or(entry);
    if entry.is_empty() {
        return false;
    }
    // As bytes: a suffix that splits a character cannot equal `entry`, so needs no check.
    let (host, entry) = (host.as_bytes(), entry.as_bytes());
    host.eq_ignore_ascii_case(entry)
        || host
            .len()
            .checked_sub(entry.len() + 1)
            .is_some_and(|dot| host[dot] == b'.' && host[dot + 1..].eq_ignore_ascii_case(entry))
}

/// Whether `ip` is in the CIDR range `entry` (`address/bits`).
fn in_cidr_range(ip: IpAddr, entry: &str) -> bool {
    let Some((network, bits)) = entry.split_once('/') else {
        return false;
    };
    let (Ok(network), Ok(bits)) = (network.parse::<IpAddr>(), bits.parse::<u32>()) else {
        return false;
    };
    match (ip, network) {
        (IpAddr::V4(ip), IpAddr::V4(network)) => {
            let mask = u32::MAX.checked_shl(32 - bits.min(32)).unwrap_or(0);
            u32::from(ip) & mask == u32::from(network) & mask
        }
        (IpAddr::V6(ip), IpAddr::V6(network)) => {
            let mask = u128::MAX.checked_shl(128 - bits.min(128)).unwrap_or(0);
            u128::from(ip) & mask == u128::from(network) & mask
        }
        _ => false,
    }
}

/// `text` with its `%XX` triplets decoded, and any other `%` kept, as gRPC's `PercentDecode`.
fn percent_decode(text: &str) -> String {
    percent_encoding::percent_decode_str(text)
        .decode_utf8_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(vars: &[(&str, &str)]) -> ProxyEnv {
        let vars: HashMap<String, OsString> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), OsString::from(v)))
            .collect();
        ProxyEnv::from_lookup(|name| vars.get(name).cloned())
    }

    fn authority(vars: &[(&str, &str)], host: &str) -> Option<String> {
        env(vars)
            .proxy_for(host, false)
            .unwrap()
            .map(|proxy| proxy.authority)
    }

    const PROXY: (&str, &str) = ("https_proxy", "http://proxy.corp:3128");

    #[test]
    fn precedence_of_the_proxy_variables() {
        // As gRPC: grpc_proxy, then https_proxy, then http_proxy.
        let all = [
            ("grpc_proxy", "http://g:1"),
            ("https_proxy", "http://s:2"),
            ("http_proxy", "http://h:3"),
        ];
        assert_eq!(authority(&all, "spark.example"), Some("g:1".into()));
        assert_eq!(authority(&all[1..], "spark.example"), Some("s:2".into()));
        assert_eq!(authority(&all[2..], "spark.example"), Some("h:3".into()));
        assert_eq!(authority(&[], "spark.example"), None);
    }

    #[test]
    fn only_lowercase_variables_are_read() {
        // gRPC reads `https_proxy` and `no_proxy`, not `HTTPS_PROXY` nor `NO_PROXY`.
        assert_eq!(
            authority(&[("HTTPS_PROXY", "http://p:1")], "spark.example"),
            None
        );
        assert_eq!(
            authority(&[("HTTP_PROXY", "http://p:1")], "spark.example"),
            None
        );
        assert_eq!(
            authority(&[PROXY, ("NO_PROXY", "spark.example")], "spark.example"),
            Some("proxy.corp:3128".into())
        );
    }

    #[test]
    fn an_empty_variable_is_no_proxy_and_hides_the_next_one() {
        // gRPC takes the first variable that is set, so an empty https_proxy is no proxy,
        // whatever http_proxy is.
        let vars = [("https_proxy", ""), ("http_proxy", "http://h:3")];
        assert_eq!(authority(&vars, "spark.example"), None);
    }

    #[test]
    fn loopback_hosts_are_proxied_but_of_a_token_in_plaintext() {
        // As gRPC, which proxies a loopback host as any other (of the local credentials of the
        // connection to the proxy), but of a channel that sends its token in plaintext.
        for host in ["localhost", "127.0.0.1", "[::1]", "spark.localhost"] {
            assert_eq!(
                authority(&[PROXY], host),
                Some("proxy.corp:3128".into()),
                "{host}"
            );
        }
        let direct = |host: &str| {
            env(&[PROXY])
                .proxy_for(host, true)
                .unwrap()
                .map(|proxy| proxy.authority)
        };
        for host in [
            "localhost",
            "LOCALHOST",
            "127.0.0.1",
            "127.1.2.3",
            "[::1]",
            "::1",
            "localhost.",
            "spark.localhost",
            "SPARK.LOCALHOST.",
            "::ffff:127.0.0.1",
            "[::ffff:127.0.0.1]",
        ] {
            assert_eq!(direct(host), None, "{host}");
        }
        for host in [
            "[2001:db8::1]",
            "localhost.example",
            "notlocalhost",
            "localhost..",
            ".localhost",
            "[::ffff:10.0.0.1]",
        ] {
            assert_eq!(direct(host), Some("proxy.corp:3128".into()), "{host}");
        }
    }

    #[test]
    fn proxy_uris() {
        for (uri, expected) in [
            ("http://p:3128", Some("p:3128")),
            ("HTTP://p:3128", Some("p:3128")),
            ("http://p:3128/", Some("p:3128")),
            ("http://p:3128/some/path?q#f", Some("p:3128")),
            ("http://p", Some("p:443")),
            ("http://[::1]:3128", Some("[::1]:3128")),
            ("http://[2001:db8::1]", Some("[2001:db8::1]:443")),
            ("http://10.0.0.1:8080", Some("10.0.0.1:8080")),
            // Not an http proxy, or no proxy at all: as gRPC, a direct connection.
            ("https://p:3128", None),
            ("socks5://p:1080", None),
            ("p:3128", None),
            ("127.0.0.1:3128", None),
            ("http:p:3128", None),
            ("http://", None),
            ("http://alice@", None),
            ("://p:3128", None),
            ("1http://p", None),
        ] {
            assert_eq!(
                authority(&[("https_proxy", uri)], "spark.example").as_deref(),
                expected,
                "{uri}"
            );
        }
    }

    #[test]
    fn a_value_that_is_not_a_proxy_is_logged_for_every_host() {
        // As grpcio, which logs it as an error of its core for every channel, whatever its host,
        // a host of the no-proxy list or a loopback host too.
        let vars = [
            ("grpc_proxy", "socks5://p:1080"),
            ("no_proxy", "spark.example"),
        ];
        let (direct, logged) = crate::grpc_log::logged_by(|| {
            ["spark.example", "other.example", "localhost"]
                .map(|host| env(&vars).proxy_for(host, false).unwrap().is_none())
        });
        assert_eq!(direct, [true; 3]);
        let messages: Vec<_> = logged.iter().map(|entry| entry.message.as_str()).collect();
        let message = "the 'socks5' scheme of the proxy URI is not supported in grpc_proxy; \
                       not using a proxy";
        assert_eq!(messages, [message; 3]);
        assert!(logged[0].location.file().ends_with("proxy.rs"));
    }

    #[test]
    fn an_http_proxy_at_an_invalid_address_is_an_error() {
        // grpcio does not connect with these, but for a port past 65535, which it takes modulo
        // 65536 (99999 as 34463). The channel is an error rather than a connection that the user
        // did not ask for, direct or to another port.
        for uri in [
            "http://:3128",
            "http://p:notaport",
            "http://p:99999",
            "http://p:",
            "http://bad host:3128",
            "http://[::1",
        ] {
            let error = env(&[("https_proxy", uri)])
                .proxy_for("spark.example", false)
                .unwrap_err();
            assert!(
                error.starts_with("Invalid HTTP proxy address in https_proxy"),
                "{uri}: {error}"
            );
            assert!(!error.contains(uri), "{error}");
        }
    }

    #[test]
    fn an_invalid_proxy_address_is_not_the_error_of_a_host_that_is_not_proxied() {
        // As grpcio, which resolves the address of the proxy only for a host that it proxies:
        // a host of the no-proxy list connects directly, as a loopback host does here.
        for uri in [
            "http://:3128",
            "http://p:notaport",
            "http://p:80837",
            "http://p:",
            "http://bad host:3128",
            "http://[::1",
        ] {
            for no_proxy in [("no_proxy", "spark.example"), ("no_grpc_proxy", ".example")] {
                let proxy = env(&[("grpc_proxy", uri), no_proxy]).proxy_for("spark.example", false);
                assert!(proxy.unwrap().is_none(), "{uri} {no_proxy:?}");
            }
            let proxy = env(&[("grpc_proxy", uri)]).proxy_for("localhost", true);
            assert!(proxy.unwrap().is_none(), "{uri}");
            let error = env(&[("grpc_proxy", uri), ("no_proxy", "other.example")])
                .proxy_for("spark.example", false)
                .unwrap_err();
            assert!(error.starts_with("Invalid HTTP proxy address in grpc_proxy"));
        }
    }

    #[test]
    fn credentials_are_percent_decoded_basic_auth() {
        // As gRPC: the userinfo after decoding, split at the last '@' of the authority.
        let auth = |uri: &str| {
            env(&[("https_proxy", uri)])
                .proxy_for("spark.example", false)
                .unwrap()
                .unwrap()
                .authorization
        };
        assert_eq!(
            auth("http://alice:p%40ss%3Aw0rd@p:1"),
            Some("Basic YWxpY2U6cEBzczp3MHJk".into())
        );
        assert_eq!(auth("http://alice@p:1"), Some("Basic YWxpY2U=".into()));
        assert_eq!(
            auth("http://a@b:c@p:1"),
            Some(format!("Basic {}", STANDARD.encode(b"a@b:c")))
        );
        assert_eq!(auth("http://p:1"), None);
        assert_eq!(
            auth("http://bad%zz%4@p:1"),
            Some(format!("Basic {}", STANDARD.encode(b"bad%zz%4")))
        );
    }

    #[test]
    fn no_proxy_lists() {
        let proxied = |no_proxy: &str, host: &str| {
            env(&[PROXY, ("no_proxy", no_proxy)])
                .proxy_for(host, false)
                .unwrap()
                .is_some()
        };
        // A host or a parent domain, on a label boundary, ignoring case and spaces.
        assert!(!proxied("spark.example", "spark.example"));
        assert!(!proxied("example", "spark.example"));
        assert!(!proxied(".example", "spark.example"));
        assert!(!proxied("SPARK.EXAMPLE", "spark.example"));
        assert!(!proxied(" other , spark.example ", "spark.example"));
        assert!(proxied("ark.example", "spark.example"));
        assert!(proxied("other.example", "spark.example"));
        // No wildcard and no port, as gRPC.
        assert!(proxied("*", "spark.example"));
        assert!(proxied("spark.example:443", "spark.example"));
        // IP hosts: as text, and by CIDR range.
        assert!(!proxied("10.1.2.3", "10.1.2.3"));
        assert!(!proxied("10.0.0.0/8", "10.1.2.3"));
        assert!(proxied("10.0.0.0/16", "10.1.2.3"));
        assert!(!proxied("2.3", "10.1.2.3"));
        assert!(!proxied("2001:db8::/32", "[2001:db8::1]"));
        assert!(!proxied("2001:db8::1", "[2001:db8::1]"));
        assert!(proxied("2001:db9::/32", "[2001:db8::1]"));
        assert!(!proxied("0.0.0.0/0", "10.1.2.3"));
        assert!(proxied("10.0.0.0/x", "10.1.2.3"));
        // A CIDR range is not a domain.
        assert!(proxied("10.0.0.0/8", "spark.example"));
        // As gRPC's `StringToSockaddr`, an IPv6 address of a zone is in a range, and an entry of a
        // port matches no host, a loopback one either.
        assert!(!proxied("::1/128", "[::1%1]"));
        assert!(!proxied("::/0", "fe80::1%1"));
        assert!(proxied("::1/128", "[::1%nosuchinterface0]"));
        assert!(proxied("127.0.0.1:17843", "127.0.0.1"));
        assert!(!proxied("127.0.0.0/8", "127.0.0.1"));
    }

    #[test]
    fn no_grpc_proxy_takes_precedence_over_no_proxy() {
        let vars = [
            PROXY,
            ("no_grpc_proxy", "other"),
            ("no_proxy", "spark.example"),
        ];
        assert_eq!(
            authority(&vars, "spark.example"),
            Some("proxy.corp:3128".into())
        );
    }

    #[test]
    fn debug_never_shows_credentials() {
        let env = env(&[("https_proxy", "http://alice:hunter2@p:1")]);
        let proxy = env.proxy_for("spark.example", false).unwrap().unwrap();
        for shown in [format!("{env:?}"), format!("{proxy:?}")] {
            assert!(
                !shown.contains("hunter2") && !shown.contains("YWxp"),
                "{shown}"
            );
        }
    }

    #[test]
    fn percent_decoding_keeps_invalid_triplets() {
        assert_eq!(percent_decode("a%40b"), "a@b");
        assert_eq!(percent_decode("a%4"), "a%4");
        assert_eq!(percent_decode("%"), "%");
        assert_eq!(percent_decode("%zz"), "%zz");
        assert_eq!(percent_decode("%e2%82%ac"), "€");
    }
}
