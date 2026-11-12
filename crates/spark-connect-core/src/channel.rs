//! Spark Connect connection-string parsing and channel configuration.
//!
//! Mirrors `pyspark.sql.connect.client.core.ChannelBuilder` /
//! `DefaultChannelBuilder`. Parsing follows the Spark Connect URL spec and
//! reproduces CPython's `urllib.parse.urlparse` semantics for the `params`
//! component (the segment after the first `;` following the last `/`).

#![allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API

use std::collections::BTreeMap;

use crate::error::{Result, SparkError};

/// Client version reported in the user-agent (mirrors `pyspark.__version__`).
pub const SPARK_VERSION: &str = "4.3.0";

/// Default Spark Connect server port.
pub const DEFAULT_PORT: u16 = 15002;

/// Max gRPC message length default (128 MiB).
pub const GRPC_MAX_MESSAGE_LENGTH_DEFAULT: usize = 128 * 1024 * 1024;

pub const PARAM_USE_SSL: &str = "use_ssl";
pub const PARAM_TOKEN: &str = "token";
pub const PARAM_USER_ID: &str = "user_id";
pub const PARAM_USER_AGENT: &str = "user_agent";
pub const PARAM_SESSION_ID: &str = "session_id";
pub const PARAM_GRPC_KEEPALIVE_ENABLED: &str = "grpc_keepalive_enabled";
pub const PARAM_GRPC_KEEPALIVE_TIME_MS: &str = "grpc_keepalive_time_ms";
pub const PARAM_GRPC_KEEPALIVE_TIMEOUT_MS: &str = "grpc_keepalive_timeout_ms";
pub const PARAM_GRPC_KEEPALIVE_WITHOUT_CALLS: &str = "grpc_keepalive_without_calls";

const GRPC_DEFAULT_KEEPALIVE_ENABLED: bool = true;
const GRPC_DEFAULT_KEEPALIVE_TIME_MS: i64 = 60 * 1000;
const GRPC_DEFAULT_KEEPALIVE_TIMEOUT_MS: i64 = 20 * 1000;
const GRPC_DEFAULT_KEEPALIVE_WITHOUT_CALLS: bool = true;

/// Parsed Spark Connect connection string plus channel parameters.
#[derive(Clone)]
pub struct ChannelBuilder {
    params: BTreeMap<String, String>,
    host: String, // display host; IPv6 wrapped in [ ]
    port: u16,
}

/// The gRPC/HTTP2 keepalive of a channel: PINGs after `time` without activity, a connection that
/// does not acknowledge one within `timeout` is dead, and `without_calls` sends them also when
/// no call is in flight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Keepalive {
    pub time: std::time::Duration,
    pub timeout: std::time::Duration,
    pub without_calls: bool,
}

/// Shows the params with the values of the token and of an `authorization` header redacted:
/// they are secrets, which a log of the builder must not hold.
impl std::fmt::Debug for ChannelBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params: BTreeMap<&str, &str> = self
            .params
            .iter()
            .map(|(key, value)| {
                (
                    key.as_str(),
                    if is_secret_param(key) {
                        "<redacted>"
                    } else {
                        value.as_str()
                    },
                )
            })
            .collect();
        f.debug_struct("ChannelBuilder")
            .field("params", &params)
            .field("host", &self.host)
            .field("port", &self.port)
            .finish()
    }
}

impl ChannelBuilder {
    /// Parse a `sc://host[:port][/;k=v;...]` connection string.
    ///
    /// Mirrors `DefaultChannelBuilder.__init__` + `_extract_attributes`.
    pub fn parse(url: &str) -> Result<Self> {
        if !url.starts_with("sc://") {
            return Err(SparkError::value(
                "INVALID_CONNECT_URL",
                &[(
                    "detail",
                    "The URL must start with 'sc://'. Please update the URL to \
                     follow the correct format, e.g., 'sc://hostname:port'.",
                )],
            ));
        }
        let rest = &url["sc://".len()..];

        // Split authority from the remainder at the first of '/', '?', '#'.
        let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let authority = &rest[..auth_end];
        let tail = &rest[auth_end..];
        // urlsplit checks the netloc before the path and the params are read.
        let (host_text, port_text) = split_hostinfo(authority)?;

        // Path is the tail up to any query/fragment.
        let path_end = tail.find(['?', '#']).unwrap_or(tail.len());
        let path = &tail[..path_end];

        // CPython `_splitparams`: params begin at the first ';' after the last '/'.
        let (path_only, params_str) = split_params(path);

        if !path_only.is_empty() && path_only != "/" {
            return Err(SparkError::value(
                "INVALID_CONNECT_URL",
                &[(
                    "detail",
                    &format!(
                        "The path component '{path_only}' must be empty. Please update \
                         the URL to follow the correct format, e.g., 'sc://hostname:port'."
                    ),
                )],
            ));
        }

        let mut params: BTreeMap<String, String> = BTreeMap::new();
        if !params_str.is_empty() {
            for p in params_str.split(';') {
                let kv: Vec<&str> = p.split('=').collect();
                if kv.len() != 2 {
                    // The key only: the value of a parameter may be the token.
                    let key = p.split_once('=').map_or(p, |(key, _)| key);
                    return Err(SparkError::value(
                        "INVALID_CONNECT_URL",
                        &[(
                            "detail",
                            &format!(
                                "Parameter '{key}' should be provided as a key-value pair \
                                 separated by an equal sign (=). Please update the parameter \
                                 to follow the correct format, e.g., 'key=value'."
                            ),
                        )],
                    ));
                }
                params.insert(kv[0].to_string(), unquote(kv[1]));
            }
        }

        let (host_raw, port) = parse_authority(host_text, port_text, &redact_url(url))?;
        // urllib lowercases the hostname.
        let hostname = host_raw.to_ascii_lowercase();
        let host = if hostname.contains(':') {
            format!("[{hostname}]")
        } else {
            hostname
        };

        Ok(Self { params, host, port })
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.params.get(key).map(String::as_str)
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.params.insert(key.to_string(), value.to_string());
    }

    /// `host:port` endpoint used to dial the gRPC channel.
    pub fn endpoint(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn use_ssl(&self) -> bool {
        self.params
            .get(PARAM_USE_SSL)
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    /// Token from the connection string, falling back to the env var.
    pub fn token(&self) -> Option<String> {
        self.params
            .get(PARAM_TOKEN)
            .cloned()
            .or_else(|| std::env::var("SPARK_CONNECT_AUTHENTICATE_TOKEN").ok())
    }

    /// Whether the channel uses TLS, which is the scheme `SparkConnectClient::connect` dials.
    ///
    /// The reference client (`core.py` `ChannelBuilder.secure`) is TLS when `use_ssl` is set
    /// or a token is present, except for a token on `localhost`, which stays plaintext for
    /// local development (`core.py` `toChannel` uses local channel credentials there). A
    /// token is therefore never sent in cleartext to any other host, `127.0.0.1` and `[::1]`
    /// included.
    pub fn secure(&self) -> bool {
        self.use_ssl() || (self.token().is_some() && !self.is_localhost())
    }

    /// Whether the host is `localhost`. The reference client compares `localhost` after
    /// `urllib` has lowercased the host (`core.py` `toChannel`), so the match is
    /// case-insensitive. The IP loopback literals are not `localhost`, so they use TLS with
    /// a token.
    pub fn is_localhost(&self) -> bool {
        self.host.eq_ignore_ascii_case("localhost")
    }

    /// Whether the endpoint is a loopback address: `localhost` or the IPv4/IPv6 loopback
    /// literals. Only `localhost` keeps a token in plaintext; see [`Self::secure`].
    pub fn is_loopback(&self) -> bool {
        self.is_localhost() || matches!(self.host.as_str(), "127.0.0.1" | "[::1]")
    }

    pub fn user_id(&self) -> Option<&str> {
        self.get(PARAM_USER_ID)
    }

    /// Validates the `session_id` param is a UUID of any version, and returns it as given.
    /// As `uuid.UUID(session_id, version=4)` in PySpark, which does not check the version.
    pub fn session_id(&self) -> Result<Option<String>> {
        match self.params.get(PARAM_SESSION_ID) {
            None => Ok(None),
            Some(s) => match uuid::Uuid::parse_str(s) {
                Ok(_) => Ok(Some(s.clone())),
                Err(_) => Err(SparkError::value(
                    "INVALID_SESSION_UUID_ID",
                    &[("arg_name", "session_id"), ("origin", "invalid UUID")],
                )),
            },
        }
    }

    pub fn keepalive_enabled(&self) -> bool {
        self.bool_param(PARAM_GRPC_KEEPALIVE_ENABLED, GRPC_DEFAULT_KEEPALIVE_ENABLED)
    }

    pub fn keepalive_time_ms(&self) -> i64 {
        self.int_param(PARAM_GRPC_KEEPALIVE_TIME_MS, GRPC_DEFAULT_KEEPALIVE_TIME_MS)
    }

    pub fn keepalive_timeout_ms(&self) -> i64 {
        self.int_param(
            PARAM_GRPC_KEEPALIVE_TIMEOUT_MS,
            GRPC_DEFAULT_KEEPALIVE_TIMEOUT_MS,
        )
    }

    pub fn keepalive_without_calls(&self) -> bool {
        self.bool_param(
            PARAM_GRPC_KEEPALIVE_WITHOUT_CALLS,
            GRPC_DEFAULT_KEEPALIVE_WITHOUT_CALLS,
        )
    }

    /// The keepalive that the channel applies, as PySpark's `_effective_channel_options` sets
    /// `grpc.keepalive_time_ms`, `grpc.keepalive_timeout_ms` and
    /// `grpc.keepalive_permit_without_calls`: none when `grpc_keepalive_enabled` is false (or the
    /// time is not positive, which no PING interval can be).
    pub fn keepalive(&self) -> Option<Keepalive> {
        let millis = |ms: i64| std::time::Duration::from_millis(u64::try_from(ms).unwrap_or(0));
        let time = self.keepalive_time_ms();
        if !self.keepalive_enabled() || time <= 0 {
            return None;
        }
        Some(Keepalive {
            time: millis(time),
            timeout: millis(self.keepalive_timeout_ms()),
            without_calls: self.keepalive_without_calls(),
        })
    }

    /// The error of a keepalive parameter that is not an integer, as PySpark's client raises it
    /// when it opens its channel: with the keepalive enabled, its channel options read the time
    /// and then the timeout with `int()`, which raises `ValueError`.
    pub(crate) fn check_keepalive(&self) -> Result<()> {
        if !self.keepalive_enabled() {
            return Ok(());
        }
        for key in [
            PARAM_GRPC_KEEPALIVE_TIME_MS,
            PARAM_GRPC_KEEPALIVE_TIMEOUT_MS,
        ] {
            match self.params.get(key) {
                Some(value) if python_int(value).is_none() => {
                    return Err(SparkError::builtin_value_msg(format!(
                        "invalid literal for int() with base 10: {}",
                        python_repr(value)
                    )));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// The user-agent string, matching Python's format and 2048-char cap.
    pub fn user_agent(&self) -> Result<String> {
        let ua = self
            .params
            .get(PARAM_USER_AGENT)
            .cloned()
            .or_else(|| std::env::var("SPARK_CONNECT_USER_AGENT").ok())
            .unwrap_or_else(|| "_SPARK_CONNECT_PYTHON".to_string());
        let ua_len = quote_len(&ua);
        if ua_len > 2048 {
            return Err(SparkError::connect_exception_msg(format!(
                "'user_agent' parameter should not exceed 2048 characters after URL \
                 escaping, found {ua_len} characters."
            )));
        }
        Ok(format!(
            "{ua} spark/{SPARK_VERSION} os/{} python/rust",
            os_name(std::env::consts::OS)
        ))
    }

    /// gRPC metadata: every param except the reserved channel-config keys.
    pub fn metadata(&self) -> Vec<(String, String)> {
        const RESERVED: &[&str] = &[
            PARAM_TOKEN,
            PARAM_USE_SSL,
            PARAM_USER_ID,
            PARAM_USER_AGENT,
            PARAM_SESSION_ID,
            PARAM_GRPC_KEEPALIVE_ENABLED,
            PARAM_GRPC_KEEPALIVE_TIME_MS,
            PARAM_GRPC_KEEPALIVE_TIMEOUT_MS,
            PARAM_GRPC_KEEPALIVE_WITHOUT_CALLS,
        ];
        self.params
            .iter()
            .filter(|(k, _)| !RESERVED.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn bool_param(&self, key: &str, default: bool) -> bool {
        match self.params.get(key) {
            Some(v) => v.eq_ignore_ascii_case("true"),
            None => default,
        }
    }

    fn int_param(&self, key: &str, default: i64) -> i64 {
        self.params
            .get(key)
            .and_then(|v| python_int(v))
            .unwrap_or(default)
    }
}

/// The name of the operating system `os` (of `std::env::consts::OS`) as Python's
/// `platform.uname().system.lower()` gives it, which PySpark's user agent carries.
fn os_name(os: &str) -> &str {
    match os {
        "macos" => "darwin",
        "solaris" | "illumos" => "sunos",
        "hurd" => "gnu",
        os => os,
    }
}

/// `text` as Python's `int(text)` reads it: an optional sign and digits, which single
/// underscores may separate, between whitespace. Python also takes the decimal digits of other
/// scripts, which this does not. A value past the range of `i64` saturates.
fn python_int(text: &str) -> Option<i64> {
    let text = text.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c));
    let (negative, digits) = match text.strip_prefix('-') {
        Some(digits) => (true, digits),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    if digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
    {
        return None;
    }
    let mut value: i64 = 0;
    for c in digits.chars().filter(|&c| c != '_') {
        let digit = i64::from(c.to_digit(10)?);
        value = value.saturating_mul(10).saturating_add(digit);
    }
    Some(if negative { -value } else { value })
}

/// `text` as Python's `repr` of a string shows it, as in the message of `int()`: quoted with
/// `'`, or with `"` when it has a `'` and no `"`, and with its backslashes and control
/// characters escaped.
fn python_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut shown = String::from(quote);
    for c in text.chars() {
        match c {
            '\\' => shown.push_str("\\\\"),
            '\t' => shown.push_str("\\t"),
            '\n' => shown.push_str("\\n"),
            '\r' => shown.push_str("\\r"),
            c if c == quote => {
                shown.push('\\');
                shown.push(c);
            }
            c if c.is_ascii_control() => shown.push_str(&format!("\\x{:02x}", c as u32)),
            c => shown.push(c),
        }
    }
    shown.push(quote);
    shown
}

/// CPython `urllib.parse._splitparams`: params begin at the first `;` at or
/// after the last `/`; if there is no `/`, at the first `;` anywhere.
fn split_params(path: &str) -> (&str, &str) {
    let search_from = path.rfind('/');
    let semi = match search_from {
        Some(last_slash) => path[last_slash..].find(';').map(|i| last_slash + i),
        None => path.find(';'),
    };
    match semi {
        Some(i) => (&path[..i], &path[i + 1..]),
        None => (path, ""),
    }
}

/// Whether a parameter is a secret: its value is shown redacted, and never in an error.
pub(crate) fn is_secret_param(key: &str) -> bool {
    key.eq_ignore_ascii_case(PARAM_TOKEN) || key.eq_ignore_ascii_case("authorization")
}

/// The URL with the value of each secret parameter replaced by `<redacted>`, for the error
/// messages that quote the URL.
fn redact_url(url: &str) -> String {
    url.split(';')
        .map(|part| match part.split_once('=') {
            Some((key, _)) if is_secret_param(key.rsplit('/').next().unwrap_or(key)) => {
                format!("{key}=<redacted>")
            }
            _ => part.to_string(),
        })
        .collect::<Vec<_>>()
        .join(";")
}

/// The host and the port texts of a netloc, as `urllib.parse` splits them (`urlsplit` and
/// `_hostinfo`): the host is what follows the last `@`, a bracketed host ends at its `]`, and
/// otherwise the host ends at the first `:`. The checks are those of `urlsplit`, which raises
/// their ValueErrors before the path is read. The messages quote the text with its secrets
/// redacted.
fn split_hostinfo(authority: &str) -> Result<(&str, Option<&str>)> {
    // A `[` and a `]` come in pairs in the netloc, the userinfo included.
    if authority.contains('[') != authority.contains(']') {
        return Err(SparkError::builtin_value_msg("Invalid IPv6 URL"));
    }
    let hostinfo = authority
        .rsplit_once('@')
        .map_or(authority, |(_, hostinfo)| hostinfo);
    let (host, port) = match hostinfo.split_once('[') {
        Some((before, bracketed)) => {
            // `_check_bracketed_netloc`: no data before the bracket, nor after it but a port.
            let (addr, after) = bracketed.split_once(']').unwrap_or((bracketed, ""));
            if !before.is_empty() || !(after.is_empty() || after.starts_with(':')) {
                return Err(SparkError::builtin_value_msg("Invalid IPv6 URL"));
            }
            (addr, after.strip_prefix(':'))
        }
        None => match hostinfo.split_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (hostinfo, None),
        },
    };
    // A netloc with brackets checks its host as an address, bracketed or not.
    if authority.contains('[') {
        check_bracketed_host(host)?;
    }
    // An empty port is no port.
    Ok((host, port.filter(|port| !port.is_empty())))
}

/// `urllib`'s check of the host of a netloc with brackets, as `ipaddress.ip_address` makes it: an
/// IPv6 address (with a zone), or an IPvFuture (`v` hex digits `.` text), but no IPv4 address.
fn check_bracketed_host(host: &str) -> Result<()> {
    if let Some(future) = host.strip_prefix('v') {
        let hex = future.bytes().take_while(u8::is_ascii_hexdigit).count();
        let text = future[hex..].strip_prefix('.');
        return match text {
            Some(text) if hex > 0 && !text.is_empty() && !text.contains('\n') => Ok(()),
            _ => Err(SparkError::builtin_value_msg(
                "IPvFuture address is invalid",
            )),
        };
    }
    if host.parse::<std::net::Ipv4Addr>().is_ok() {
        return Err(SparkError::builtin_value_msg(
            "An IPv4 address cannot be in brackets",
        ));
    }
    let ipv6 = match host.split_once('%') {
        // A zone is the text after the `%`, which may not be empty or hold another `%`.
        Some((addr, zone)) => {
            !zone.is_empty() && !zone.contains('%') && addr.parse::<std::net::Ipv6Addr>().is_ok()
        }
        None => host.parse::<std::net::Ipv6Addr>().is_ok(),
    };
    if ipv6 {
        Ok(())
    } else {
        Err(SparkError::builtin_value_msg(format!(
            "{} does not appear to be an IPv4 or IPv6 address",
            python_repr(&redact_url(host))
        )))
    }
}

/// The host and the port of a connection string, once its params are read: the host must not be
/// empty (urllib's `hostname`), and the port is the one of the netloc, or the default.
fn parse_authority(host: &str, port: Option<&str>, shown_url: &str) -> Result<(String, u16)> {
    if host.is_empty() {
        return Err(SparkError::value(
            "INVALID_CONNECT_URL",
            &[(
                "detail",
                &format!(
                    "Hostname is missing in the URL: '{shown_url}'. Please update the URL \
                     to follow the correct format, e.g., 'sc://hostname:port'."
                ),
            )],
        ));
    }

    let port = match port {
        None => DEFAULT_PORT,
        Some(p) => parse_port(p)?,
    };

    Ok((host.to_string(), port))
}

/// The port of a connection string as `urllib.parse`'s `port` reads it: ASCII digits, within
/// 0-65535. Its messages are those of its ValueError, which the reference client raises
/// unwrapped and without the URL; the text is quoted with its secrets redacted.
fn parse_port(text: &str) -> Result<u16> {
    if !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(SparkError::builtin_value_msg(format!(
            "Port could not be cast to integer value as {}",
            python_repr(&redact_url(text))
        )));
    }
    // Saturating, so that a long run of digits is out of range rather than wrapping around.
    let port = text.bytes().fold(0u64, |port, digit| {
        port.saturating_mul(10)
            .saturating_add(u64::from(digit - b'0'))
    });
    u16::try_from(port).map_err(|_| SparkError::builtin_value_msg("Port out of range 0-65535"))
}

/// Decode `%XX` percent-escapes (CPython `urllib.parse.unquote`).
fn unquote(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Length of `urllib.parse.quote(s)` with the default `safe='/'`.
fn quote_len(s: &str) -> usize {
    fn unreserved(b: u8) -> bool {
        b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-' | b'~' | b'/')
    }
    s.bytes().map(|b| if unreserved(b) { 1 } else { 3 }).sum()
}

/// Serializes the tests that set or read the environment variables that the channel reads,
/// since the test threads share one environment.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Sets environment variables for one test, under `ENV_LOCK`, and restores them on drop.
/// A `None` value removes the variable.
#[cfg(test)]
pub(crate) struct EnvGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

#[cfg(test)]
pub(crate) fn set_env(vars: &[(&'static str, Option<&str>)]) -> EnvGuard {
    let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut saved = Vec::new();
    for (key, value) in vars {
        saved.push((*key, std::env::var_os(key)));
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }
    EnvGuard { _lock: lock, saved }
}

#[cfg(test)]
impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in self.saved.iter().rev() {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_host_default_port() {
        let _env = set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        let c = ChannelBuilder::parse("sc://localhost").unwrap();
        assert_eq!(c.endpoint(), "localhost:15002");
        assert!(!c.secure());
        assert!(!c.use_ssl());
    }

    #[test]
    fn host_with_port() {
        let c = ChannelBuilder::parse("sc://example.com:1234").unwrap();
        assert_eq!(c.endpoint(), "example.com:1234");
    }

    #[test]
    fn params_ssl_and_token() {
        let c = ChannelBuilder::parse("sc://localhost/;use_ssl=true;token=aaa").unwrap();
        assert!(c.use_ssl());
        assert!(c.secure());
        assert_eq!(c.token().as_deref(), Some("aaa"));
        assert_eq!(c.endpoint(), "localhost:15002");
    }

    #[test]
    fn keepalive_override() {
        let c = ChannelBuilder::parse("sc://localhost/;grpc_keepalive_time_ms=30000").unwrap();
        assert_eq!(c.keepalive_time_ms(), 30000);
        assert_eq!(c.keepalive_timeout_ms(), 20000); // default
        assert!(c.keepalive_enabled());
    }

    #[test]
    fn keepalive_is_that_of_the_channel_options_of_pyspark() {
        use std::time::Duration;
        // PySpark's defaults (SPARK-58094): a PING after 60 s, 20 s for its ack, also idle.
        assert_eq!(
            ChannelBuilder::parse("sc://localhost").unwrap().keepalive(),
            Some(Keepalive {
                time: Duration::from_secs(60),
                timeout: Duration::from_secs(20),
                without_calls: true,
            })
        );
        let custom = ChannelBuilder::parse(
            "sc://localhost/;grpc_keepalive_time_ms=30000;grpc_keepalive_timeout_ms=5000;\
             grpc_keepalive_without_calls=false",
        )
        .unwrap();
        assert_eq!(
            custom.keepalive(),
            Some(Keepalive {
                time: Duration::from_secs(30),
                timeout: Duration::from_secs(5),
                without_calls: false,
            })
        );
        // Turned off, as PySpark then adds none of the keepalive options.
        let off = ChannelBuilder::parse("sc://localhost/;grpc_keepalive_enabled=FALSE").unwrap();
        assert_eq!(off.keepalive(), None);
        let zero = ChannelBuilder::parse("sc://localhost/;grpc_keepalive_time_ms=0").unwrap();
        assert_eq!(zero.keepalive(), None);
        // The keepalive params are channel options, not request metadata.
        assert!(custom.metadata().is_empty());
    }

    #[test]
    fn keepalive_params_are_read_with_python_int_as_pyspark() {
        // As `int()` reads them: between whitespace, with a sign, and single underscores.
        for (value, ms) in [
            ("%2030000%20", 30000),
            ("3_0000", 30000),
            ("+30000", 30000),
            ("-5", -5),
            ("007", 7),
        ] {
            let url = format!("sc://localhost/;grpc_keepalive_time_ms={value}");
            let c = ChannelBuilder::parse(&url).unwrap();
            assert!(c.check_keepalive().is_ok(), "{value}");
            assert_eq!(c.keepalive_time_ms(), ms, "{value}");
        }
        // The values that `int()` raises for, which PySpark's channel options read when the
        // keepalive is enabled: the time, then the timeout.
        for (params, shown) in [
            ("grpc_keepalive_time_ms=abc", "'abc'"),
            ("grpc_keepalive_timeout_ms=abc", "'abc'"),
            (
                "grpc_keepalive_time_ms=abc;grpc_keepalive_timeout_ms=xyz",
                "'abc'",
            ),
            ("grpc_keepalive_time_ms=", "''"),
            ("grpc_keepalive_time_ms=1.5", "'1.5'"),
            ("grpc_keepalive_time_ms=1__0", "'1__0'"),
            ("grpc_keepalive_time_ms=_10", "'_10'"),
            ("grpc_keepalive_time_ms=%2B%201", "'+ 1'"),
            ("grpc_keepalive_time_ms=a'b", "\"a'b\""),
            ("grpc_keepalive_time_ms=a%5Cb", "'a\\\\b'"),
            ("grpc_keepalive_time_ms=%09", "'\\t'"),
        ] {
            let c = ChannelBuilder::parse(&format!("sc://localhost/;{params}")).unwrap();
            let error = c.check_keepalive().unwrap_err();
            assert_eq!(
                error.kind,
                crate::error::SparkErrorKind::BuiltinValueError,
                "{params}"
            );
            assert_eq!(
                error.message(),
                format!("invalid literal for int() with base 10: {shown}"),
                "{params}"
            );
        }
        // Neither is read when the keepalive is disabled.
        for params in [
            "grpc_keepalive_enabled=false;grpc_keepalive_time_ms=abc",
            "grpc_keepalive_enabled=nope;grpc_keepalive_timeout_ms=abc",
            "grpc_keepalive_without_calls=abc",
        ] {
            let c = ChannelBuilder::parse(&format!("sc://localhost/;{params}")).unwrap();
            assert!(c.check_keepalive().is_ok(), "{params}");
        }
    }

    #[test]
    fn port_is_read_as_urllib_reads_it() {
        // `urllib.parse`'s `port`: ASCII digits only, within 0-65535, else a ValueError. Its
        // messages are a builtin ValueError's, which the reference client raises unwrapped.
        for (port, expected) in [
            ("15003", 15003),
            ("0080", 80),
            ("0", 0),
            ("65535", 65535),
            ("", DEFAULT_PORT), // no port, or an empty one
        ] {
            let c = ChannelBuilder::parse(&format!("sc://localhost:{port}/;token=SECRET")).unwrap();
            assert_eq!(c.port(), expected, "{port}");
        }
        for (url, message) in [
            ("sc://localhost:99999/", "Port out of range 0-65535"),
            ("sc://localhost:65536", "Port out of range 0-65535"),
            (
                "sc://localhost:01234567890123456789",
                "Port out of range 0-65535",
            ),
            ("sc://[::1]:99999/", "Port out of range 0-65535"),
            (
                "sc://localhost:+80/",
                "Port could not be cast to integer value as '+80'",
            ),
            (
                "sc://localhost:-1/",
                "Port could not be cast to integer value as '-1'",
            ),
            (
                "sc://localhost:8 0/",
                "Port could not be cast to integer value as '8 0'",
            ),
            (
                "sc://localhost:80x/",
                "Port could not be cast to integer value as '80x'",
            ),
            (
                "sc://localhost:1_0/",
                "Port could not be cast to integer value as '1_0'",
            ),
            (
                "sc://localhost:%38%30/",
                "Port could not be cast to integer value as '%38%30'",
            ),
            (
                "sc://[::1]:x/",
                "Port could not be cast to integer value as 'x'",
            ),
            // Not ASCII digits, even when they are the Arabic-Indic one and two.
            (
                "sc://localhost:\u{0661}\u{0662}/",
                "Port could not be cast to integer value as '\u{0661}\u{0662}'",
            ),
        ] {
            let e = ChannelBuilder::parse(url).unwrap_err();
            assert_eq!(
                e.kind,
                crate::error::SparkErrorKind::BuiltinValueError,
                "{url}"
            );
            assert_eq!(e.message(), message, "{url}");
        }
    }

    #[test]
    fn authority_is_read_as_urllib_reads_it() {
        // The host (lowercased, a bracketed IPv6 address wrapped in `[ ]`) and the port of the
        // netloc: the host follows the last `@`, a bracketed host ends at its `]`, and otherwise
        // the host ends at the first `:`.
        for (url, host, port) in [
            (
                "sc://user@localhost:15012/;token=SECRET",
                "localhost",
                15012,
            ),
            ("sc://user:pw@localhost:15012/", "localhost", 15012),
            ("sc://u]@[::1]:15002", "[::1]", 15002),
            ("sc://[::1]:15012/", "[::1]", 15012),
            ("sc://LOCALHOST:15012/", "localhost", 15012),
            ("sc://[v1.x]:15002", "v1.x", 15002),
            ("sc://[fe80::1%25eth0]:15002", "[fe80::1%25eth0]", 15002),
            ("sc://[::1]:/", "[::1]", DEFAULT_PORT),
        ] {
            let c = ChannelBuilder::parse(url).unwrap();
            assert_eq!(c.host(), host, "{url}");
            assert_eq!(c.port(), port, "{url}");
        }
        // The messages are the ValueErrors of urllib, with the secrets of the URL redacted.
        for (url, message) in [
            (
                "sc://localhost:15012:1/",
                "Port could not be cast to integer value as '15012:1'",
            ),
            (
                "sc://a:1:2",
                "Port could not be cast to integer value as '1:2'",
            ),
            (
                "sc://localhost:x;token=SECRET",
                "Port could not be cast to integer value as 'x;token=<redacted>'",
            ),
            (
                "sc://localhost:15002;token=SECRET",
                "Port could not be cast to integer value as '15002;token=<redacted>'",
            ),
            ("sc://[::1]x", "Invalid IPv6 URL"),
            ("sc://[::1]x:15002", "Invalid IPv6 URL"),
            (
                "sc://]@[x",
                "'x' does not appear to be an IPv4 or IPv6 address",
            ),
            (
                "sc://[::1]@host:15002",
                "'host' does not appear to be an IPv4 or IPv6 address",
            ),
            (
                "sc://[::1@x]",
                "'x]' does not appear to be an IPv4 or IPv6 address",
            ),
            (
                "sc://[1.2.3.4]:15002",
                "An IPv4 address cannot be in brackets",
            ),
            (
                "sc://[localhost]:15002",
                "'localhost' does not appear to be an IPv4 or IPv6 address",
            ),
            (
                "sc://[]:15002",
                "'' does not appear to be an IPv4 or IPv6 address",
            ),
            ("sc://[vz]:15002", "IPvFuture address is invalid"),
            (
                "sc://[a;token=SECRET]:1",
                "'a;token=<redacted>' does not appear to be an IPv4 or IPv6 address",
            ),
        ] {
            let e = ChannelBuilder::parse(url).unwrap_err();
            assert_eq!(
                e.kind,
                crate::error::SparkErrorKind::BuiltinValueError,
                "{url}"
            );
            assert_eq!(e.message(), message, "{url}");
            assert!(!e.message().contains("SECRET"), "{url}");
        }
        // A missing host is the INVALID_CONNECT_URL of the reference client, read after params.
        let e = ChannelBuilder::parse("sc://user:pw@:15002").unwrap_err();
        assert_eq!(e.kind, crate::error::SparkErrorKind::ValueError);
        assert!(
            e.message().contains("Hostname is missing"),
            "{}",
            e.message()
        );
    }

    #[test]
    fn a_too_long_user_agent_is_a_connect_exception() {
        // PySpark raises a SparkConnectException of this message, not a RuntimeError.
        let _env = set_env(&[("SPARK_CONNECT_USER_AGENT", None)]);
        let url = |len: usize| format!("sc://localhost/;user_agent={}", "a".repeat(len));
        assert!(ChannelBuilder::parse(&url(2048))
            .unwrap()
            .user_agent()
            .is_ok());
        let e = ChannelBuilder::parse(&url(2049))
            .unwrap()
            .user_agent()
            .unwrap_err();
        assert_eq!(e.kind, crate::error::SparkErrorKind::ConnectMessage);
        assert_eq!(
            e.message(),
            "'user_agent' parameter should not exceed 2048 characters after URL escaping, \
             found 2049 characters."
        );
    }

    #[test]
    fn user_agent_names_the_os_as_pyspark() {
        // As Python's `platform.uname().system.lower()`.
        for (os, name) in [
            ("macos", "darwin"),
            ("linux", "linux"),
            ("windows", "windows"),
            ("freebsd", "freebsd"),
            ("illumos", "sunos"),
        ] {
            assert_eq!(os_name(os), name, "{os}");
        }
        let _env = set_env(&[("SPARK_CONNECT_USER_AGENT", None)]);
        let c = ChannelBuilder::parse("sc://localhost/;user_agent=abc").unwrap();
        let shown = os_name(std::env::consts::OS);
        assert_eq!(
            c.user_agent().unwrap(),
            format!("abc spark/4.3.0 os/{shown} python/rust")
        );
    }

    #[test]
    fn debug_of_a_builder_redacts_its_token() {
        let c = ChannelBuilder::parse(
            "sc://spark.example.com:443/;token=SECRET123;authorization=Bearer%20XYZ;x-custom=1",
        )
        .unwrap();
        let shown = format!("{c:?}");
        assert!(!shown.contains("SECRET123"), "{shown}");
        assert!(!shown.contains("XYZ"), "{shown}");
        assert!(shown.contains("<redacted>"), "{shown}");
        assert!(shown.contains("x-custom"), "{shown}");
        assert!(shown.contains("spark.example.com"), "{shown}");
        // The token itself is unchanged.
        assert_eq!(c.token().as_deref(), Some("SECRET123"));
    }

    #[test]
    fn parse_errors_do_not_echo_the_token() {
        // A bad port names the port only, as PySpark's message does; the URL is not quoted.
        let e = ChannelBuilder::parse("sc://localhost:notaport/;token=SECRET").unwrap_err();
        assert_eq!(e.kind, crate::error::SparkErrorKind::BuiltinValueError);
        let shown = e.to_string();
        assert!(
            shown.contains("Port could not be cast to integer value as 'notaport'"),
            "{shown}"
        );
        assert!(
            !shown.contains("SECRET") && !shown.contains("localhost"),
            "{shown}"
        );
        // A malformed parameter names its key only: its value may be the token.
        let e = ChannelBuilder::parse("sc://localhost/;token=SE=CRET").unwrap_err();
        let shown = e.to_string();
        assert!(shown.contains("Parameter 'token' should be"), "{shown}");
        assert!(
            !shown.contains("SE=CRET") && !shown.contains("CRET"),
            "{shown}"
        );
        // A missing host quotes the URL with the token value redacted, as the Debug does.
        let e = ChannelBuilder::parse("sc://:15002/;token=SECRET;user_id=bob").unwrap_err();
        let shown = e.to_string();
        assert!(
            shown.contains("sc://:15002/;token=<redacted>;user_id=bob"),
            "{shown}"
        );
        assert!(!shown.contains("SECRET"), "{shown}");
    }

    #[test]
    fn scheme_is_tls_for_a_token_on_any_host_but_localhost() {
        let _env = set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", None)]);
        // (url, TLS). A token goes over TLS to any host but `localhost`, as PySpark does, so
        // the loopback literals use TLS too. `use_ssl` is TLS for every host.
        for (url, tls) in [
            ("sc://example.com", false),
            ("sc://example.com/;use_ssl=true", true),
            ("sc://example.com/;token=T", true),
            ("sc://example.com/;use_ssl=false;token=T", true),
            ("sc://10.0.0.1:443/;token=T", true),
            ("sc://localhost/;token=T", false),
            ("sc://localhost/;use_ssl=true", true),
            ("sc://localhost/;use_ssl=true;token=T", true),
            ("sc://LOCALHOST/;token=T", false),
            ("sc://127.0.0.1/;token=T", true),
            ("sc://127.0.0.1/;use_ssl=true", true),
            ("sc://127.0.0.1/", false),
            ("sc://[::1]:15002/;token=T", true),
            ("sc://[::1]:15002/;use_ssl=true", true),
            ("sc://[::1]:15002/", false),
        ] {
            let c = ChannelBuilder::parse(url).unwrap();
            assert_eq!(c.secure(), tls, "{url}");
        }
    }

    #[test]
    fn scheme_uses_the_token_of_the_environment() {
        let _env = set_env(&[("SPARK_CONNECT_AUTHENTICATE_TOKEN", Some("ENV"))]);
        let remote = ChannelBuilder::parse("sc://example.com").unwrap();
        assert_eq!(remote.token().as_deref(), Some("ENV"));
        assert!(remote.secure());
        let local = ChannelBuilder::parse("sc://localhost").unwrap();
        assert_eq!(local.token().as_deref(), Some("ENV"));
        assert!(!local.secure());
        // The token of the connection string wins over the environment.
        let explicit = ChannelBuilder::parse("sc://example.com/;token=T").unwrap();
        assert_eq!(explicit.token().as_deref(), Some("T"));
    }

    #[test]
    fn ipv6_host_is_bracketed() {
        let c = ChannelBuilder::parse("sc://[::1]:15003").unwrap();
        assert_eq!(c.host(), "[::1]");
        assert_eq!(c.endpoint(), "[::1]:15003");
    }

    #[test]
    fn percent_decoded_param_value() {
        let c = ChannelBuilder::parse("sc://localhost/;user_agent=my%20agent").unwrap();
        assert_eq!(c.get("user_agent"), Some("my agent"));
    }

    #[test]
    fn metadata_excludes_reserved() {
        let c = ChannelBuilder::parse("sc://localhost/;token=t;user_id=u;x-custom=v").unwrap();
        let md = c.metadata();
        assert_eq!(md, vec![("x-custom".to_string(), "v".to_string())]);
    }

    #[test]
    fn rejects_missing_scheme() {
        let e = ChannelBuilder::parse("localhost:15002").unwrap_err();
        assert_eq!(e.error_class, "INVALID_CONNECT_URL");
    }

    #[test]
    fn rejects_nonempty_path() {
        let e = ChannelBuilder::parse("sc://localhost/foo").unwrap_err();
        assert_eq!(e.error_class, "INVALID_CONNECT_URL");
    }

    #[test]
    fn rejects_bad_param_pair() {
        let e = ChannelBuilder::parse("sc://localhost/;use_ssl").unwrap_err();
        assert_eq!(e.error_class, "INVALID_CONNECT_URL");
    }

    #[test]
    fn rejects_missing_host() {
        let e = ChannelBuilder::parse("sc://:15002").unwrap_err();
        assert_eq!(e.error_class, "INVALID_CONNECT_URL");
    }

    #[test]
    fn session_id_accepts_a_uuid_of_any_version_as_given() {
        // Versions 1, 4, 7 and the nil UUID, as `uuid.UUID(session_id, version=4)` accepts.
        for id in [
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "550e8400-e29b-41d4-a716-446655440000",
            "018f0b8e-0000-7000-8000-000000000000",
            "00000000-0000-0000-0000-000000000000",
        ] {
            let c = ChannelBuilder::parse(&format!("sc://localhost/;session_id={id}")).unwrap();
            assert_eq!(c.session_id().unwrap().as_deref(), Some(id));
        }
        let none = ChannelBuilder::parse("sc://localhost").unwrap();
        assert_eq!(none.session_id().unwrap(), None);
    }

    #[test]
    fn session_id_rejects_a_malformed_uuid() {
        for id in ["not-a-uuid", "550e8400-e29b-41d4-a716-44665544000", "x"] {
            let c = ChannelBuilder::parse(&format!("sc://localhost/;session_id={id}")).unwrap();
            let e = c.session_id().unwrap_err();
            assert_eq!(e.error_class, "INVALID_SESSION_UUID_ID", "{id}");
        }
    }
}
