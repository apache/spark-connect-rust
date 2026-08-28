//! Spark Connect connection-string parsing and channel configuration.
//!
//! Mirrors `pyspark.sql.connect.client.core.ChannelBuilder` /
//! `DefaultChannelBuilder`. Parsing follows the Spark Connect URL spec and
//! reproduces CPython's `urllib.parse.urlparse` semantics for the `params`
//! component (the segment after the first `;` following the last `/`).

use std::collections::BTreeMap;

use crate::error::{Result, SparkError};

/// Client version reported in the user-agent (mirrors `pyspark.__version__`).
pub const SPARK_VERSION: &str = "4.1.0.dev0";

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
#[derive(Debug, Clone)]
pub struct ChannelBuilder {
    params: BTreeMap<String, String>,
    host: String, // display host; IPv6 wrapped in [ ]
    port: u16,
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
                    return Err(SparkError::value(
                        "INVALID_CONNECT_URL",
                        &[(
                            "detail",
                            &format!(
                                "Parameter '{p}' should be provided as a key-value pair \
                                 separated by an equal sign (=). Please update the parameter \
                                 to follow the correct format, e.g., 'key=value'."
                            ),
                        )],
                    ));
                }
                params.insert(kv[0].to_string(), unquote(kv[1]));
            }
        }

        let (host_raw, port) = parse_authority(authority, url)?;
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

    /// A channel is secure when SSL is requested or a token is present.
    pub fn secure(&self) -> bool {
        self.use_ssl() || self.token().is_some()
    }

    pub fn user_id(&self) -> Option<&str> {
        self.get(PARAM_USER_ID)
    }

    /// Validates the `session_id` param is a v4 UUID (mirrors the Python check).
    pub fn session_id(&self) -> Result<Option<String>> {
        match self.params.get(PARAM_SESSION_ID) {
            None => Ok(None),
            Some(s) => match uuid::Uuid::parse_str(s) {
                Ok(u) if u.get_version_num() == 4 => Ok(Some(s.clone())),
                Ok(_) | Err(_) => Err(SparkError::value(
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
            return Err(SparkError::connect_msg(format!(
                "'user_agent' parameter should not exceed 2048 characters after URL \
                 escaping, found {ua_len} characters."
            )));
        }
        let os = std::env::consts::OS.to_lowercase();
        Ok(format!("{ua} spark/{SPARK_VERSION} os/{os} python/rust"))
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
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
}

/// CPython `urllib.parse._splitparams`: params begin at the first `;` at or
/// after the last `/`; if there is no `/`, at the first `;` anywhere.
fn split_params(path: &str) -> (&str, &str) {
    let search_from = path.rfind('/').map(|i| i + 0);
    let semi = match search_from {
        Some(last_slash) => path[last_slash..].find(';').map(|i| last_slash + i),
        None => path.find(';'),
    };
    match semi {
        Some(i) => (&path[..i], &path[i + 1..]),
        None => (path, ""),
    }
}

/// Parse `host`, `host:port`, `[ipv6]`, or `[ipv6]:port` into (host, port).
fn parse_authority(authority: &str, full_url: &str) -> Result<(String, u16)> {
    let missing_host = || {
        SparkError::value(
            "INVALID_CONNECT_URL",
            &[(
                "detail",
                &format!(
                    "Hostname is missing in the URL: '{full_url}'. Please update the URL \
                     to follow the correct format, e.g., 'sc://hostname:port'."
                ),
            )],
        )
    };

    let (host, port_str) = if let Some(bracket_end) = authority.strip_prefix('[') {
        // IPv6: [addr] or [addr]:port
        let close = bracket_end.find(']').ok_or_else(missing_host)?;
        let addr = &bracket_end[..close];
        let after = &bracket_end[close + 1..];
        let port = after.strip_prefix(':');
        (addr.to_string(), port)
    } else if let Some((h, p)) = authority.rsplit_once(':') {
        (h.to_string(), Some(p))
    } else {
        (authority.to_string(), None)
    };

    if host.is_empty() {
        return Err(missing_host());
    }

    let port = match port_str {
        None | Some("") => DEFAULT_PORT,
        Some(p) => p.parse::<u16>().map_err(|_| {
            SparkError::value(
                "INVALID_CONNECT_URL",
                &[(
                    "detail",
                    &format!("Port '{p}' in URL '{full_url}' is not a valid integer."),
                )],
            )
        })?,
    };

    Ok((host, port))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_host_default_port() {
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
    fn token_implies_secure() {
        let c = ChannelBuilder::parse("sc://localhost/;token=xyz").unwrap();
        assert!(c.secure());
        assert!(!c.use_ssl());
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
    fn session_id_must_be_uuid4() {
        let c = ChannelBuilder::parse(
            "sc://localhost/;session_id=550e8400-e29b-41d4-a716-446655440000",
        )
        .unwrap();
        assert!(c.session_id().is_ok());
        let bad = ChannelBuilder::parse("sc://localhost/;session_id=not-a-uuid").unwrap();
        assert!(bad.session_id().is_err());
    }
}
