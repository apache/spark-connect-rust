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

//! The targets of gRPC, as grpcio's channel of PySpark reads the target that `toChannel` gives it
//! (grpcio 1.85): what a channel connects to of a target, as a [`ChannelTarget`]. Not a stable
//! API.
//!
//! - The target is a URI of the scheme of a resolver of gRPC, `dns`, `ipv4`, `ipv6`, `unix` or
//!   `unix-abstract`, of any case, or else `dns:///` and the target, as gRPC's registry of
//!   resolvers prefixes it. A target that is not a URI then is not valid. A URI is read as gRPC's
//!   parser reads it: its authority after `//`, up to the first `/`, `?` or `#`, its path up to
//!   the first `?` or `#`, both percent-decoded, a query, which is dropped, of no empty one and of
//!   its characters only, and a fragment, which is dropped, of its characters only.
//! - Of `dns:`, the name is the path without its first `/`, of a host and a port, as gRPC's
//!   resolver splits it: of no port, that of `https`, 443; of the ports `http` and `https`, 80
//!   and 443; else the port that `absl::SimpleAtoi` reads, of a sign and of spaces around it,
//!   taken modulo 65536 as gRPC's resolver gives it to the addresses (`-1` is 65535). One of no
//!   path is not valid, and one of an authority that is not an IP address and a port fails each
//!   call as gRPC's resolver fails it (`Cannot parse authority`).
//! - Of `ipv4:` and `ipv6:`, the path is the addresses, separated by `,`, of which an empty one
//!   is skipped: each an IP address of the family of the scheme and a port that `sscanf("%d")`
//!   reads, of 0 to 65535, an IPv6 one of a zone of a number or of the name of an interface.
//! - Of `unix:` and `unix-abstract:`, the path is the paths or the abstract names of the sockets,
//!   separated by `,`, of 107 bytes at most.
//!
//! A target that is not valid fails each call of `UNKNOWN`, as grpcio's channel of it does
//! (`Failed to create channel to '...':the target uri is not valid: ...`).
//!
//! The target is of bytes, up to its first NUL, as grpcio gives gRPC's core a C string, and so are
//! its decoded authority and path: a path of a Unix socket is the bytes that it decodes to. Of each
//! target, the channel takes gRPC's names of it besides what it connects to (see
//! [`TargetNames`]): its default authority, of its calls and of the name that its TLS verifies,
//! and the server of the tunnel of a proxy, as gRPC's proxy mapper takes it.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV6};

use crate::channel::{ChannelTarget, TargetNames};

/// The code of gRPC's `UNKNOWN`, of a target that is not valid.
const UNKNOWN: i32 = 2;

/// The code of gRPC's `INTERNAL`, of a resolver that fails of a status that a resolver does not
/// give (`Illegal status code from resolver`).
const INTERNAL: i32 = 13;

/// The schemes of gRPC's resolvers, of grpcio 1.85.
const SCHEMES: [&str; 7] = [
    "dns",
    "ipv4",
    "ipv6",
    "unix",
    "unix-abstract",
    "xds",
    "google-c2p",
];

/// The code of gRPC's `UNAVAILABLE`, of a resolver that fails.
const UNAVAILABLE: i32 = 14;

/// The prefix that gRPC's registry of resolvers gives a target of no scheme of a resolver.
const DEFAULT_PREFIX: &str = "dns:///";

/// The longest path of a Unix socket that gRPC takes, the length of `sun_path` but its last byte.
const MAX_UNIX_PATH: usize = 107;

/// The longest IPv6 address before its zone that gRPC takes (`GRPC_INET6_ADDRSTRLEN`).
const MAX_IPV6_TEXT: usize = 46;

/// What a channel connects to of a target of gRPC, gRPC's names of the target, and the host and
/// the port that its connection string has in place of it, which stand for it only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parsed {
    /// What the channel connects to.
    pub target: ChannelTarget,
    /// gRPC's names of the target: its default authority, of the calls and of the name that TLS
    /// verifies, and the server of the tunnel of a proxy (see [`TargetNames`]).
    pub names: TargetNames,
    /// The host of the connection string: of a name of DNS, its host, an IPv6 one in brackets,
    /// of an address, the first one, else `localhost`. A host that a connection string cannot
    /// hold is `localhost`. It stands for the target in the URL of the connection only, and
    /// decides nothing of its TLS, its proxy or its calls, which are of [`Parsed::names`].
    pub host: String,
    /// The port of the connection string.
    pub port: u16,
}

/// What a channel connects to of the target `target`, of its bytes up to its first NUL, as
/// grpcio's channel reads it (see the module).
pub fn parse(target: impl AsRef<[u8]>) -> Parsed {
    let target = target.as_ref();
    // As grpcio gives the target to gRPC's core, a C string.
    let target = &target[..target.iter().position(|&b| b == 0).unwrap_or(target.len())];
    let text = String::from_utf8_lossy(target);
    let not_valid = |canonical: &str| {
        refused(
            UNKNOWN,
            format!(
                "Failed to create channel to '{text}':the target uri is not valid: {canonical}"
            ),
        )
    };
    let (uri, canonical) = match Uri::parse(target) {
        Ok(uri) if SCHEMES.contains(&uri.scheme.as_str()) => (uri, text.to_string()),
        _ => {
            let canonical = [DEFAULT_PREFIX.as_bytes(), target].concat();
            let canonical_text = String::from_utf8_lossy(&canonical).into_owned();
            match Uri::parse(&canonical) {
                Ok(uri) => (uri, canonical_text),
                Err(()) => return not_valid(&canonical_text),
            }
        }
    };
    let parsed = match uri.scheme.as_str() {
        // Of a machine that is not of Google's cloud, gRPC's resolver of `google-c2p` is its DNS.
        "dns" | "google-c2p" => dns(&uri),
        "ipv4" => addresses(&uri, false),
        "ipv6" => addresses(&uri, true),
        "unix" => sockets(&uri, false),
        "xds" => xds(&uri),
        _ => sockets(&uri, true),
    };
    match parsed {
        Some(mut parsed) => {
            if !matches!(parsed.target, ChannelTarget::Refused { .. }) {
                parsed.names = names_of(&uri);
            }
            parsed
        }
        None => not_valid(&canonical),
    }
}

/// gRPC's names of the target of `uri` (see [`TargetNames`]): its default authority, the path of
/// no first `/`, percent-encoded as an authority (`ResolverFactory::GetDefaultAuthority`); and,
/// but of `unix:`, which gRPC's proxy mapper does not proxy, the host of the path
/// (`SplitHostPort`), of the exclusions of the proxy, and the `host:port` of it, of the port 443
/// without one, as the server of the tunnel.
fn names_of(uri: &Uri) -> TargetNames {
    let path = uri.path.strip_prefix(b"/").unwrap_or(&uri.path);
    let proxied = (uri.scheme != "unix").then(|| {
        let path = String::from_utf8_lossy(path).into_owned();
        match split_host_port(&path) {
            Some((host, port)) => {
                let port = port.filter(|port| !port.is_empty()).unwrap_or("443");
                let tunnel = if host.contains(':') {
                    format!("[{host}]:{port}")
                } else {
                    format!("{host}:{port}")
                };
                // Of a C string, the argument of the server of gRPC's `CONNECT`, which ends at a
                // NUL of the path (`localhost` of `localhost%00:15002`).
                let tunnel = tunnel.split('\0').next().unwrap_or_default().to_string();
                (host.to_string(), tunnel)
            }
            None => {
                let tunnel = path.split('\0').next().unwrap_or_default().to_string();
                (path.clone(), tunnel)
            }
        }
    });
    TargetNames {
        authority: percent_encode_authority(path),
        proxied,
    }
}

/// `bytes` percent-encoded as gRPC's `URI::PercentEncodeAuthority` encodes an authority: of its
/// unreserved characters, its sub-delims, `:`, `[`, `]` and `@` as they are, and of any other
/// byte as `%` and two upper-case hexadecimal digits.
pub(crate) fn percent_encode_authority(bytes: &[u8]) -> String {
    percent_encode(bytes, b":[]@")
}

/// `bytes` percent-encoded as gRPC's `URI::ToString` encodes a path, as of the URI of the address
/// of a Unix socket in its errors: of its unreserved characters, its sub-delims, `:`, `@` and `/`
/// as they are, and of any other byte as `%` and two upper-case hexadecimal digits.
pub(crate) fn percent_encode_path(bytes: &[u8]) -> String {
    percent_encode(bytes, b":@/")
}

/// `bytes` of their unreserved characters, their sub-delims and the bytes of `kept` as they are,
/// and of any other byte as `%` and two upper-case hexadecimal digits, as gRPC's `PercentEncode`.
fn percent_encode(bytes: &[u8], kept: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        if b.is_ascii_alphanumeric() || b"-._~!$&'()*+,;=".contains(&b) || kept.contains(&b) {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// A target whose calls fail of `code` and `message`, of the host and port of no server.
fn refused(code: i32, message: String) -> Parsed {
    of_localhost(ChannelTarget::Refused { code, message })
}

/// `target`, of the host `localhost` and the port 443, which name no server it connects to.
fn of_localhost(target: ChannelTarget) -> Parsed {
    Parsed {
        target,
        names: TargetNames::default(),
        host: "localhost".to_string(),
        port: 443,
    }
}

/// A target of `dns:`, of its name (see the module), or None of one of no name, which is not
/// valid.
fn dns(uri: &Uri) -> Option<Parsed> {
    let path = String::from_utf8_lossy(&uri.path);
    let name = path.strip_prefix('/').unwrap_or(&path);
    if name.is_empty() {
        return None;
    }
    // The DNS server of the authority, as gRPC's resolver gives it to c-ares: an IP address and a
    // port, else no resolver.
    let authority = String::from_utf8_lossy(&uri.authority);
    let server = match authority.as_ref() {
        "" => None,
        authority => match parse_host_port(authority, None) {
            Some(server) => Some(server),
            None => {
                return Some(refused(
                    INTERNAL,
                    format!(
                        "Illegal status code from resolver; original status: INVALID_ARGUMENT: \
                         Cannot parse authority: {authority}"
                    ),
                ));
            }
        },
    };
    let unresolvable = |detail: String| {
        Some(of_localhost(ChannelTarget::Unresolvable {
            name: name.to_string(),
            detail,
        }))
    };
    let Some((host, port)) = split_host_port(name) else {
        return unresolvable(format!("Unparsable name: {name}"));
    };
    if host.is_empty() {
        return unresolvable(format!("host must not be empty in name: {name}"));
    }
    let port = match port {
        None | Some("") | Some("https") => 443,
        Some("http") => 80,
        Some(port) => match simple_atoi(port) {
            // As gRPC's resolver gives the port to the addresses, of 16 bits.
            Some(port) => port as u16,
            None => return unresolvable(format!("Failed to parse port in name: {name}")),
        },
    };
    Some(Parsed {
        host: url_host(host),
        port,
        names: TargetNames::default(),
        target: ChannelTarget::Name {
            host: host.to_string(),
            port,
            name: name.to_string(),
            server,
        },
    })
}

/// A target of `xds:`, whose calls fail of `UNAVAILABLE`, as grpcio's resolver fails to create its
/// client of xDS: of no bootstrap of the environment, of its text; of one, of the transport's, which
/// has no client of xDS. None of one of no path, which is not valid.
fn xds(uri: &Uri) -> Option<Parsed> {
    if uri.path.strip_prefix(b"/").unwrap_or(&uri.path).is_empty() {
        return None;
    }
    let bootstrap = ["GRPC_XDS_BOOTSTRAP", "GRPC_XDS_BOOTSTRAP_CONFIG"]
        .iter()
        .any(|name| std::env::var_os(name).is_some());
    let message = if bootstrap {
        "Failed to create XdsClient: the transport has no client of xDS"
    } else {
        "Failed to create XdsClient: Environment variables GRPC_XDS_BOOTSTRAP or \
         GRPC_XDS_BOOTSTRAP_CONFIG not defined"
    };
    Some(refused(UNAVAILABLE, message.to_string()))
}

/// A target of `ipv4:` or of `ipv6:` (of `v6`), of its addresses, or None of one that is not
/// valid: of an authority, or of an address that is not one of the family and a port.
fn addresses(uri: &Uri, v6: bool) -> Option<Parsed> {
    if !uri.authority.is_empty() {
        return None;
    }
    let path = std::str::from_utf8(&uri.path).ok()?;
    let mut addresses = Vec::new();
    for entry in path.split(',').filter(|entry| !entry.is_empty()) {
        let entry = entry.strip_prefix('/').unwrap_or(entry);
        addresses.push(parse_host_port(entry, Some(v6))?);
    }
    let (host, port) = match addresses.first() {
        Some(SocketAddr::V4(address)) => (address.ip().to_string(), address.port()),
        Some(SocketAddr::V6(address)) => (format!("[{}]", address.ip()), address.port()),
        None => ("localhost".to_string(), 443),
    };
    Some(Parsed {
        target: ChannelTarget::Addresses {
            addresses,
            authority: path.strip_prefix('/').unwrap_or(path).to_string(),
        },
        names: TargetNames::default(),
        host,
        port,
    })
}

/// A target of `unix:` or of `unix-abstract:` (of `abstract_names`), of its sockets, or None of
/// one that is not valid: of an authority, or of a path longer than gRPC takes.
fn sockets(uri: &Uri, abstract_names: bool) -> Option<Parsed> {
    if !uri.authority.is_empty() {
        return None;
    }
    let mut paths = Vec::new();
    for entry in uri
        .path
        .split(|&b| b == b',')
        .filter(|entry| !entry.is_empty())
    {
        if entry.len() > MAX_UNIX_PATH {
            return None;
        }
        paths.push(entry.to_vec());
    }
    Some(of_localhost(ChannelTarget::Unix {
        paths,
        abstract_names,
    }))
}

/// The host of a connection string of the host `host` of a name: an IPv6 address in brackets, a
/// name of the characters of a host of a URL as it is, else `localhost`, which stands for it.
fn url_host(host: &str) -> String {
    if let Ok(ip) = host.parse::<Ipv6Addr>() {
        return format!("[{ip}]");
    }
    let hostlike = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_');
    if host.chars().all(hostlike) {
        host.to_ascii_lowercase()
    } else {
        "localhost".to_string()
    }
}

/// A URI as gRPC's parser reads it (`URI::Parse`): its scheme, in lower case, its authority and
/// its path, percent-decoded, of their bytes. The query and the fragment are dropped, once checked.
#[derive(Debug)]
struct Uri {
    scheme: String,
    authority: Vec<u8>,
    path: Vec<u8>,
}

impl Uri {
    fn parse(text: &[u8]) -> Result<Self, ()> {
        let colon = text.iter().position(|&b| b == b':').ok_or(())?;
        let scheme = &text[..colon];
        let scheme_char = |b: &u8| b.is_ascii_alphanumeric() || b"+-.".contains(b);
        if scheme.is_empty() || !scheme.iter().all(scheme_char) || !scheme[0].is_ascii_alphabetic()
        {
            return Err(());
        }
        let find = |text: &[u8], of: &[u8]| text.iter().position(|b| of.contains(b));
        let mut rest = &text[colon + 1..];
        let mut authority = Vec::new();
        if let Some(after) = rest.strip_prefix(b"//") {
            let end = find(after, b"/?#").unwrap_or(after.len());
            authority = percent_decode(&after[..end]);
            rest = &after[end..];
        }
        let end = find(rest, b"?#").unwrap_or(rest.len());
        let path = percent_decode(&rest[..end]);
        rest = &rest[end..];
        if let Some(after) = rest.strip_prefix(b"?") {
            let end = find(after, b"#").unwrap_or(after.len());
            let query = &after[..end];
            if query.is_empty() || !is_query_or_fragment(query) {
                return Err(());
            }
            rest = &after[end..];
        }
        if let Some(fragment) = rest.strip_prefix(b"#") {
            if !is_query_or_fragment(fragment) {
                return Err(());
            }
        }
        Ok(Self {
            scheme: String::from_utf8_lossy(scheme).to_ascii_lowercase(),
            authority,
            path,
        })
    }
}

/// Whether `text` is of the characters of a query or a fragment of RFC 3986 and `%`, as gRPC's
/// parser takes them: unreserved, sub-delims, `:`, `@`, `/` and `?`.
fn is_query_or_fragment(text: &[u8]) -> bool {
    text.iter()
        .all(|&b| b.is_ascii_alphanumeric() || b"-._~!$&'()*+,;=:@/?%".contains(&b))
}

/// `text` percent-decoded as gRPC's parser decodes it: a `%` of two hexadecimal digits is their
/// byte, and any other `%` stays.
fn percent_decode(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let hex = |at: usize| text.get(at).and_then(|&b| (b as char).to_digit(16));
        match (text[i], hex(i + 1), hex(i + 2)) {
            (b'%', Some(high), Some(low)) => {
                out.push((high * 16 + low) as u8);
                i += 3;
            }
            (byte, _, _) => {
                out.push(byte);
                i += 1;
            }
        }
    }
    out
}

/// `name` split into its host and its port, as gRPC's `SplitHostPort` splits it: a host in
/// brackets, which holds a `:`, and its port after a `:`; else a host and a port of the one `:`
/// of the name, or the name of no or of more `:` as a host of no port. None of a name of a `[`
/// that no `]` closes, of a `]` that no `:` follows but at its end, or of a host in brackets of
/// no `:`.
pub(crate) fn split_host_port(name: &str) -> Option<(&str, Option<&str>)> {
    if let Some(after) = name.strip_prefix('[') {
        let close = after.find(']')?;
        let host = &after[..close];
        let port = match &after[close + 1..] {
            "" => None,
            rest => Some(rest.strip_prefix(':')?),
        };
        if !host.contains(':') {
            return None;
        }
        return Some((host, port));
    }
    match name.split_once(':') {
        Some((host, port)) if !port.contains(':') => Some((host, Some(port))),
        _ => Some((name, None)),
    }
}

/// The port of the text `port` as `absl::SimpleAtoi` reads an `int`: of ASCII spaces around it,
/// a sign and decimal digits, of no more than 32 bits.
fn simple_atoi(port: &str) -> Option<i32> {
    let port = port.trim_matches(|c: char| c.is_ascii_whitespace() || c == '\x0b');
    let (negative, digits) = match port.as_bytes().first() {
        Some(b'-') => (true, &port[1..]),
        Some(b'+') => (false, &port[1..]),
        _ => (false, port),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut value: i64 = 0;
    for b in digits.bytes() {
        value = value * 10 + i64::from(b - b'0');
        if value > i64::from(i32::MAX) + 1 {
            return None;
        }
    }
    let value = if negative { -value } else { value };
    i32::try_from(value).ok()
}

/// The port of the text `port` as `sscanf("%d")` reads it, which gRPC's parsers of addresses use:
/// after spaces, a sign and decimal digits, of any text after them, the `long` of `strtol`, of
/// its limits past them, as an `int`. None of no digit.
fn sscanf_port(port: &str) -> Option<i32> {
    let port = port.trim_start_matches(|c: char| c.is_ascii_whitespace() || c == '\x0b');
    let (negative, rest) = match port.as_bytes().first() {
        Some(b'-') => (true, &port[1..]),
        Some(b'+') => (false, &port[1..]),
        _ => (false, port),
    };
    let digits: &str = &rest[..rest.bytes().take_while(u8::is_ascii_digit).count()];
    if digits.is_empty() {
        return None;
    }
    let mut value: i128 = 0;
    for b in digits.bytes() {
        value = (value * 10 + i128::from(b - b'0')).min(i128::from(i64::MAX) + 1);
    }
    let value = if negative { -value } else { value };
    let long = value.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
    Some(long as i32)
}

/// The address of `hostport`, an IP address and a port, as gRPC's `grpc_parse_ipv4_hostport` and
/// `grpc_parse_ipv6_hostport` read it, of the family of `v6`, or of either when None: an address
/// that `inet_pton` reads, of a zone of an IPv6 one (`%` and a number or the name of an
/// interface), and a port of 0 to 65535 (see [`sscanf_port`]). None of another.
pub(crate) fn parse_host_port(hostport: &str, v6: Option<bool>) -> Option<SocketAddr> {
    let (host, port) = split_host_port(hostport)?;
    let ip = match v6 {
        Some(false) => IpAddr::V4(inet_pton_v4(host)?),
        Some(true) => return ipv6_of(host, port),
        None => match inet_pton_v4(host) {
            Some(ip) => IpAddr::V4(ip),
            None => return ipv6_of(host, port),
        },
    };
    let port = u16::try_from(sscanf_port(port?)?).ok()?;
    Some(SocketAddr::new(ip, port))
}

/// The IPv6 address of `host`, of a zone, and of the port `port` (see [`parse_host_port`]).
fn ipv6_of(host: &str, port: Option<&str>) -> Option<SocketAddr> {
    let (address, scope) = match host.rfind('%') {
        Some(at) => {
            if at > MAX_IPV6_TEXT {
                return None;
            }
            (&host[..at], Some(&host[at + 1..]))
        }
        None => (host, None),
    };
    let ip: Ipv6Addr = address.parse().ok()?;
    let scope_id = match scope {
        None => 0,
        Some(zone) => match parse_bytes_to_u32(zone) {
            Some(id) => id,
            None => interface_index(zone)?,
        },
    };
    let port = u16::try_from(sscanf_port(port?)?).ok()?;
    Some(SocketAddr::V6(SocketAddrV6::new(ip, port, 0, scope_id)))
}

/// The number of a zone of decimal digits, of 32 bits, as gRPC's `gpr_parse_bytes_to_uint32`
/// reads it: None of an empty zone, of another character, or of a number past 32 bits.
fn parse_bytes_to_u32(zone: &str) -> Option<u32> {
    if zone.is_empty() || !zone.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    zone.parse().ok()
}

/// The index of the interface `name` (`if_nametoindex`), or None of no such interface.
#[cfg(unix)]
fn interface_index(name: &str) -> Option<u32> {
    let name = std::ffi::CString::new(name).ok()?;
    // SAFETY: `name` is a C string that lives for the call.
    let index = unsafe { libc::if_nametoindex(name.as_ptr()) };
    (index != 0).then_some(index)
}

/// The index of the interface `name`: none, as zones of names are not read here.
#[cfg(not(unix))]
fn interface_index(_name: &str) -> Option<u32> {
    None
}

/// The IPv4 address of `host` as `inet_pton` reads it: four decimal numbers of 0 to 255, of no
/// leading zero, separated by `.`.
fn inet_pton_v4(host: &str) -> Option<Ipv4Addr> {
    let parts: Vec<&str> = host.split('.').collect();
    if parts.len() != 4 {
        return None;
    }
    let mut octets = [0u8; 4];
    for (octet, part) in octets.iter_mut().zip(&parts) {
        if part.is_empty()
            || part.len() > 3
            || !part.bytes().all(|b| b.is_ascii_digit())
            || (part.len() > 1 && part.starts_with('0'))
        {
            return None;
        }
        *octet = part.parse().ok()?;
    }
    Some(Ipv4Addr::from(octets))
}

/// The IPv4 address of `host` as c-ares, gRPC's resolver, reads a name of digits and of three
/// dots as an address (`fake_addrinfo`), of leading zeros too: four decimal numbers of 0 to 255.
pub(crate) fn ares_numeric_v4(host: &str) -> Option<Ipv4Addr> {
    let parts: Vec<&str> = host.split('.').collect();
    if parts.len() != 4 {
        return None;
    }
    let mut octets = [0u8; 4];
    for (octet, part) in octets.iter_mut().zip(&parts) {
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let value: u32 = part.trim_start_matches('0').parse().unwrap_or(0);
        *octet = u8::try_from(value).ok()?;
    }
    Some(Ipv4Addr::from(octets))
}

/// The IPv6 address of `host`, of a zone, as gRPC's resolver reads an IP address before it asks
/// c-ares (`grpc_parse_ipv6_hostport`), of the port `port`.
pub(crate) fn ipv6_literal(host: &str, port: u16) -> Option<SocketAddr> {
    let mut address = ipv6_of(host, Some("0"))?;
    address.set_port(port);
    Some(address)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(host: &str, port: u16, name: &str) -> ChannelTarget {
        ChannelTarget::Name {
            host: host.to_string(),
            port,
            name: name.to_string(),
            server: None,
        }
    }

    fn name_of_server(host: &str, port: u16, name: &str, server: &str) -> ChannelTarget {
        ChannelTarget::Name {
            host: host.to_string(),
            port,
            name: name.to_string(),
            server: Some(server.parse().unwrap()),
        }
    }

    fn unresolvable(name: &str, detail: &str) -> ChannelTarget {
        ChannelTarget::Unresolvable {
            name: name.to_string(),
            detail: detail.to_string(),
        }
    }

    fn not_valid(target: &str, canonical: &str) -> ChannelTarget {
        ChannelTarget::Refused {
            code: UNKNOWN,
            message: format!(
                "Failed to create channel to '{target}':the target uri is not valid: {canonical}"
            ),
        }
    }

    fn addresses(addresses: &[&str], authority: &str) -> ChannelTarget {
        ChannelTarget::Addresses {
            addresses: addresses.iter().map(|a| a.parse().unwrap()).collect(),
            authority: authority.to_string(),
        }
    }

    #[test]
    fn the_targets_are_read_as_grpcio_reads_them() {
        // Of the outcomes of grpcio 1.85.1, of the probes of PySpark's client.
        let cases = [
            (
                "localhost:15002",
                name("localhost", 15002, "localhost:15002"),
            ),
            (
                "dns:///localhost:15002",
                name("localhost", 15002, "localhost:15002"),
            ),
            (
                "DNS:///localhost:15002",
                name("localhost", 15002, "localhost:15002"),
            ),
            (
                "dns:localhost:15002",
                name("localhost", 15002, "localhost:15002"),
            ),
            (
                "dns:/localhost:15002",
                name("localhost", 15002, "localhost:15002"),
            ),
            ("dns:///localhost", name("localhost", 443, "localhost")),
            ("localhost:", name("localhost", 443, "localhost:")),
            ("[::1]:", name("::1", 443, "[::1]:")),
            ("localhost:http", name("localhost", 80, "localhost:http")),
            ("[::1]:http", name("::1", 80, "[::1]:http")),
            (
                "localhost:+15002",
                name("localhost", 15002, "localhost:+15002"),
            ),
            (
                "localhost: 15002 ",
                name("localhost", 15002, "localhost: 15002 "),
            ),
            (
                "localhost:015002",
                name("localhost", 15002, "localhost:015002"),
            ),
            ("localhost:-1", name("localhost", 65535, "localhost:-1")),
            ("localhost:65536", name("localhost", 0, "localhost:65536")),
            (
                "localhost:99999",
                name("localhost", 34463, "localhost:99999"),
            ),
            (
                "localhost:2147483647",
                name("localhost", 65535, "localhost:2147483647"),
            ),
            (
                "localhost:-113233",
                name("localhost", 15002 + 2837, "localhost:-113233"),
            ),
            // The query and the fragment are dropped, of their characters.
            (
                "127.0.0.1:15002?x=1",
                name("127.0.0.1", 15002, "127.0.0.1:15002"),
            ),
            (
                "127.0.0.1:15002?x=1#y",
                name("127.0.0.1", 15002, "127.0.0.1:15002"),
            ),
            (
                "127.0.0.1:15002#",
                name("127.0.0.1", 15002, "127.0.0.1:15002"),
            ),
            (
                "localhost:15002?x/y?z",
                name("localhost", 15002, "localhost:15002"),
            ),
            (
                "localhost:15002?x=%zz",
                name("localhost", 15002, "localhost:15002"),
            ),
            (
                "localhost:15002?x#",
                name("localhost", 15002, "localhost:15002"),
            ),
            (
                "dns://127.0.0.1:53/localhost:15002?x=1",
                name_of_server("localhost", 15002, "localhost:15002", "127.0.0.1:53"),
            ),
            // Percent-decoded, of two hexadecimal digits only.
            (
                "loc%61lhost:15002",
                name("localhost", 15002, "localhost:15002"),
            ),
            (
                "localhost%3a15002",
                name("localhost", 15002, "localhost:15002"),
            ),
            (
                "127.0.0.1%3A15002",
                name("127.0.0.1", 15002, "127.0.0.1:15002"),
            ),
            (
                "%6c%6f%63%61%6c%68%6f%73%74:15002",
                name("localhost", 15002, "localhost:15002"),
            ),
            (
                "local%zzhost:15002",
                name("local%zzhost", 15002, "local%zzhost:15002"),
            ),
            (
                "dns:%2F%2F%2Flocalhost:15002",
                name("//localhost", 15002, "//localhost:15002"),
            ),
            (
                "[fe80::1%25lo]:15002",
                name("fe80::1%lo", 15002, "[fe80::1%lo]:15002"),
            ),
            (
                "localhost:abc",
                unresolvable(
                    "localhost:abc",
                    "Failed to parse port in name: localhost:abc",
                ),
            ),
            (
                "localhost:15002x",
                unresolvable(
                    "localhost:15002x",
                    "Failed to parse port in name: localhost:15002x",
                ),
            ),
            (
                "localhost:HTTP",
                unresolvable(
                    "localhost:HTTP",
                    "Failed to parse port in name: localhost:HTTP",
                ),
            ),
            (
                "localhost:4294967296",
                unresolvable(
                    "localhost:4294967296",
                    "Failed to parse port in name: localhost:4294967296",
                ),
            ),
            (
                "localhost:15002/x",
                unresolvable(
                    "localhost:15002/x",
                    "Failed to parse port in name: localhost:15002/x",
                ),
            ),
            (
                "127.0.0.1:15002%3F",
                unresolvable(
                    "127.0.0.1:15002?",
                    "Failed to parse port in name: 127.0.0.1:15002?",
                ),
            ),
            (
                ":15002",
                unresolvable(":15002", "host must not be empty in name: :15002"),
            ),
            (
                "[::1:15002",
                unresolvable("[::1:15002", "Unparsable name: [::1:15002"),
            ),
            (
                "[::1]x:15002",
                unresolvable("[::1]x:15002", "Unparsable name: [::1]x:15002"),
            ),
            (
                "[127.0.0.1]:15002",
                unresolvable("[127.0.0.1]:15002", "Unparsable name: [127.0.0.1]:15002"),
            ),
            // Not a URI of a resolver, then not a URI.
            (
                "127.0.0.1:15002?",
                not_valid("127.0.0.1:15002?", "dns:///127.0.0.1:15002?"),
            ),
            (
                "localhost:15002?[",
                not_valid("localhost:15002?[", "dns:///localhost:15002?["),
            ),
            (
                "localhost:15002#a#b",
                not_valid("localhost:15002#a#b", "dns:///localhost:15002#a#b"),
            ),
            (
                "127.0.0.1:15002#y z",
                not_valid("127.0.0.1:15002#y z", "dns:///127.0.0.1:15002#y z"),
            ),
            ("dns:///", not_valid("dns:///", "dns:///")),
            (
                "dns://localhost:15002",
                not_valid("dns://localhost:15002", "dns://localhost:15002"),
            ),
            ("", not_valid("", "dns:///")),
            // The addresses of `ipv4:` and `ipv6:`.
            (
                "ipv4:127.0.0.1:1,127.0.0.1:15002",
                addresses(
                    &["127.0.0.1:1", "127.0.0.1:15002"],
                    "127.0.0.1:1,127.0.0.1:15002",
                ),
            ),
            (
                "ipv4:127.0.0.1:1,,127.0.0.1:15002",
                addresses(
                    &["127.0.0.1:1", "127.0.0.1:15002"],
                    "127.0.0.1:1,,127.0.0.1:15002",
                ),
            ),
            (
                "ipv4:127.0.0.1:1%2C127.0.0.1:15002",
                addresses(
                    &["127.0.0.1:1", "127.0.0.1:15002"],
                    "127.0.0.1:1,127.0.0.1:15002",
                ),
            ),
            (
                "ipv4:///127.0.0.1:15002",
                addresses(&["127.0.0.1:15002"], "127.0.0.1:15002"),
            ),
            (
                "Ipv4:127.0.0.1:15002x",
                addresses(&["127.0.0.1:15002"], "127.0.0.1:15002x"),
            ),
            (
                "ipv4:127.0.0.1: 15002",
                addresses(&["127.0.0.1:15002"], "127.0.0.1: 15002"),
            ),
            (
                "ipv4:127.0.0.1:0x10",
                addresses(&["127.0.0.1:0"], "127.0.0.1:0x10"),
            ),
            (
                "ipv4:127.0.0.1:15002#y",
                addresses(&["127.0.0.1:15002"], "127.0.0.1:15002"),
            ),
            (
                "ipv6:[::1]:+15002",
                addresses(&["[::1]:15002"], "[::1]:+15002"),
            ),
            (
                "ipv6:[::ffff:127.0.0.1]:15002",
                addresses(&["[::ffff:127.0.0.1]:15002"], "[::ffff:127.0.0.1]:15002"),
            ),
            ("ipv4:", addresses(&[], "")),
            ("ipv6:,", addresses(&[], ",")),
            (
                "ipv4:127.0.0.1",
                not_valid("ipv4:127.0.0.1", "ipv4:127.0.0.1"),
            ),
            (
                "ipv4:127.0.0.1:99999",
                not_valid("ipv4:127.0.0.1:99999", "ipv4:127.0.0.1:99999"),
            ),
            (
                "ipv4:127.0.0.1:-1",
                not_valid("ipv4:127.0.0.1:-1", "ipv4:127.0.0.1:-1"),
            ),
            (
                "ipv4:127.0.0.1:http",
                not_valid("ipv4:127.0.0.1:http", "ipv4:127.0.0.1:http"),
            ),
            (
                "ipv4:127.000.0.1:15002",
                not_valid("ipv4:127.000.0.1:15002", "ipv4:127.000.0.1:15002"),
            ),
            (
                "ipv4:127.1:15002",
                not_valid("ipv4:127.1:15002", "ipv4:127.1:15002"),
            ),
            (
                "ipv4:localhost:15002",
                not_valid("ipv4:localhost:15002", "ipv4:localhost:15002"),
            ),
            (
                "ipv4:[127.0.0.1]:15002",
                not_valid("ipv4:[127.0.0.1]:15002", "ipv4:[127.0.0.1]:15002"),
            ),
            (
                "ipv4://127.0.0.1:15002",
                not_valid("ipv4://127.0.0.1:15002", "ipv4://127.0.0.1:15002"),
            ),
            (
                "ipv6:127.0.0.1:15002",
                not_valid("ipv6:127.0.0.1:15002", "ipv6:127.0.0.1:15002"),
            ),
            (
                "ipv6:::1:15002",
                not_valid("ipv6:::1:15002", "ipv6:::1:15002"),
            ),
            ("ipv6:[::1]", not_valid("ipv6:[::1]", "ipv6:[::1]")),
            (
                "ipv6:[::1%999]:15002",
                not_valid("ipv6:[::1%999]:15002", "ipv6:[::1%999]:15002"),
            ),
            (
                "ipv6:[::1%nosuch]:15002",
                not_valid("ipv6:[::1%nosuch]:15002", "ipv6:[::1%nosuch]:15002"),
            ),
            (
                "ipv4:127.0.0.1:15002,[::1]:15002",
                not_valid(
                    "ipv4:127.0.0.1:15002,[::1]:15002",
                    "ipv4:127.0.0.1:15002,[::1]:15002",
                ),
            ),
            // The sockets of `unix:` and `unix-abstract:`.
            ("unix:", of_localhost_unix(&[], false)),
            (
                "unix:/tmp/a.sock,/tmp/b.sock",
                of_localhost_unix(&["/tmp/a.sock", "/tmp/b.sock"], false),
            ),
            (
                "unix:///tmp/a.sock",
                of_localhost_unix(&["/tmp/a.sock"], false),
            ),
            (
                "UNIX:/tmp/a.sock?x=1#f",
                of_localhost_unix(&["/tmp/a.sock"], false),
            ),
            (
                "unix:relative%2Fx.sock",
                of_localhost_unix(&["relative/x.sock"], false),
            ),
            ("unix-abstract:a,b", of_localhost_unix(&["a", "b"], true)),
            (
                "unix://relative.sock",
                not_valid("unix://relative.sock", "unix://relative.sock"),
            ),
            (
                "unix-abstract://x/y",
                not_valid("unix-abstract://x/y", "unix-abstract://x/y"),
            ),
        ];
        for (target, expected) in cases {
            assert_eq!(parse(target).target, expected, "{target}");
        }
        let long = format!("unix:/tmp/{}.sock", "x".repeat(98));
        assert_eq!(long.len() - "unix:".len(), 108);
        assert_eq!(parse(&long).target, not_valid(&long, &long));
        let longest = format!("unix-abstract:{}", "y".repeat(107));
        assert!(matches!(parse(&longest).target, ChannelTarget::Unix { .. }));
        let too_long = format!("unix-abstract:{}", "y".repeat(108));
        assert_eq!(parse(&too_long).target, not_valid(&too_long, &too_long));
    }

    fn of_localhost_unix(paths: &[&str], abstract_names: bool) -> ChannelTarget {
        ChannelTarget::Unix {
            paths: paths.iter().map(|p| p.as_bytes().to_vec()).collect(),
            abstract_names,
        }
    }

    #[test]
    fn an_authority_of_dns_is_an_ip_address_and_a_port() {
        let internal = |authority: &str| ChannelTarget::Refused {
            code: INTERNAL,
            message: format!(
                "Illegal status code from resolver; original status: INVALID_ARGUMENT: Cannot \
                 parse authority: {authority}"
            ),
        };
        for (target, authority) in [
            ("dns://8.8.8.8/localhost:15002", "8.8.8.8"),
            ("dns://x/localhost:15002", "x"),
            ("dns://x:53/localhost:15002", "x:53"),
            ("dns://:53/localhost:15002", ":53"),
            ("dns://127.0.0.1:/localhost:15002", "127.0.0.1:"),
            ("dns://127.0.0.1:99999/localhost:15002", "127.0.0.1:99999"),
            ("dns://[::1]/localhost:15002", "[::1]"),
            ("dns://127.0.0.1:-1/localhost:15002", "127.0.0.1:-1"),
        ] {
            assert_eq!(parse(target).target, internal(authority), "{target}");
        }
        // The server of an authority of an IP address and a port, which the name is resolved
        // through: of `sscanf("%d")`'s port, and of an IPv6 address of its zone.
        for (target, server) in [
            ("dns://127.0.0.1:53/localhost:15002", "127.0.0.1:53"),
            ("dns://127.0.0.1%3A53/localhost:15002", "127.0.0.1:53"),
            ("dns://%31%32%37.0.0.1:53/localhost:15002", "127.0.0.1:53"),
            ("dns://127.0.0.1:53x/localhost:15002", "127.0.0.1:53"),
            ("dns://127.0.0.1:+5353/localhost:15002", "127.0.0.1:5353"),
            ("dns://[::1]:53/localhost:15002", "[::1]:53"),
            ("dns://[::1]:0/localhost:15002", "[::1]:0"),
        ] {
            assert_eq!(
                parse(target).target,
                name_of_server("localhost", 15002, "localhost:15002", server),
                "{target}"
            );
        }
        // A name that gRPC's resolver does not take fails as it does of no authority.
        assert_eq!(
            parse("dns://127.0.0.1:53/localhost:abc").target,
            unresolvable(
                "localhost:abc",
                "Failed to parse port in name: localhost:abc"
            )
        );
    }

    #[test]
    fn the_host_of_the_connection_string_stands_for_the_target() {
        let host = |target: &str| {
            let parsed = parse(target);
            (parsed.host, parsed.port)
        };
        assert_eq!(
            host("Spark.Example.com:15002"),
            ("spark.example.com".into(), 15002)
        );
        assert_eq!(host("[::1]:15002"), ("[::1]".into(), 15002));
        assert_eq!(host("ipv4:10.0.0.1:1,10.0.0.2:2"), ("10.0.0.1".into(), 1));
        assert_eq!(host("ipv6:[::1]:7"), ("[::1]".into(), 7));
        for target in [
            "unix:/tmp/a.sock",
            "x/y.localhost:15002",
            "ipv4:",
            "localhost:abc",
            "a?",
        ] {
            assert_eq!(host(target).0, "localhost", "{target}");
        }
    }

    #[test]
    fn the_names_of_a_target_are_grpcs() {
        // Of the probes of grpcio 1.85.1: the `:authority` of each form, and the `CONNECT` of a
        // proxy (`no_proxy` aside).
        let proxied = |host: &str, server: &str| Some((host.to_string(), server.to_string()));
        for (target, authority, proxy) in [
            (
                "localhost:15002",
                "localhost:15002",
                proxied("localhost", "localhost:15002"),
            ),
            (
                "dns:///localhost",
                "localhost",
                proxied("localhost", "localhost:443"),
            ),
            (
                "LOCALHOST:18611",
                "LOCALHOST:18611",
                proxied("LOCALHOST", "LOCALHOST:18611"),
            ),
            (
                "localhost:+18611",
                "localhost:+18611",
                proxied("localhost", "localhost:+18611"),
            ),
            (
                "loc%61lhost:18611",
                "localhost:18611",
                proxied("localhost", "localhost:18611"),
            ),
            (
                "127.0.0.1:18611?x=1",
                "127.0.0.1:18611",
                proxied("127.0.0.1", "127.0.0.1:18611"),
            ),
            (
                "[::1%25lo]:17843",
                "[::1%25lo]:17843",
                proxied("::1%lo", "[::1%lo]:17843"),
            ),
            (
                "a/b.localhost:18620",
                "a%2Fb.localhost:18620",
                proxied("a/b.localhost", "a/b.localhost:18620"),
            ),
            (
                "127.0.0.1%00:18620",
                "127.0.0.1%00:18620",
                proxied("127.0.0.1\0", "127.0.0.1"),
            ),
            (
                "ipv4:127.0.0.1:1,127.0.0.1:18611",
                "127.0.0.1:1,127.0.0.1:18611",
                proxied(
                    "127.0.0.1:1,127.0.0.1:18611",
                    "[127.0.0.1:1,127.0.0.1:18611]:443",
                ),
            ),
            (
                "unix:/tmp/rv62220/rec.sock",
                "tmp%2Frv62220%2Frec.sock",
                None,
            ),
            (
                "unix-abstract:rv62220ab",
                "rv62220ab",
                proxied("rv62220ab", "rv62220ab:443"),
            ),
            (
                "dns://127.0.0.1:53/localhost:15002",
                "localhost:15002",
                proxied("localhost", "localhost:15002"),
            ),
        ] {
            let names = parse(target).names;
            assert_eq!(names.authority, authority, "{target}");
            assert_eq!(names.proxied, proxy, "{target}");
        }
        // The name that TLS verifies: the host of the authority, up to a `%`.
        for (target, name) in [
            ("localhost%00:18620", Some("localhost")),
            ("127.0.0.1%00:18620", Some("127.0.0.1")),
            ("[::1%25lo]:18621", Some("::1")),
            ("x*y.localhost:18620", Some("x*y.localhost")),
            ("a/b.localhost:18620", Some("a")),
            ("unix:/tmp/rv62220/tls.sock", Some("tmp")),
            (
                "ipv4:127.0.0.1:18622,127.0.0.1:1",
                Some("127.0.0.1:18622,127.0.0.1:1"),
            ),
            ("ipv4:127.0.0.1:18622", Some("127.0.0.1")),
            ("ipv6:[::1]:18621", Some("::1")),
            ("%25:1", None),
        ] {
            assert_eq!(parse(target).names.tls_name(), name, "{target}");
        }
        // A target that gRPC refuses has none.
        assert_eq!(parse("a?").names, TargetNames::default());
    }

    #[test]
    fn the_targets_of_xds_and_google_c2p_are_read_as_grpcio_reads_them() {
        // Of the probes of grpcio 1.85.1, of no bootstrap of xDS in the environment.
        if std::env::var_os("GRPC_XDS_BOOTSTRAP").is_none()
            && std::env::var_os("GRPC_XDS_BOOTSTRAP_CONFIG").is_none()
        {
            for target in [
                "xds:///localhost:15002",
                "XDS:///a",
                "xds://x/localhost:15002",
            ] {
                assert_eq!(
                    parse(target).target,
                    ChannelTarget::Refused {
                        code: UNAVAILABLE,
                        message: "Failed to create XdsClient: Environment variables \
                                  GRPC_XDS_BOOTSTRAP or GRPC_XDS_BOOTSTRAP_CONFIG not defined"
                            .to_string(),
                    },
                    "{target}"
                );
            }
        }
        assert_eq!(parse("xds:").target, not_valid("xds:", "xds:"));
        assert_eq!(
            parse("google-c2p:///localhost:15002").target,
            name("localhost", 15002, "localhost:15002")
        );
    }

    #[test]
    fn a_target_is_of_bytes_up_to_its_first_nul() {
        // Of the probes: grpcio's target is a C string, and the path of a Unix socket is the
        // bytes that it decodes to.
        for (target, expected) in [
            (
                &b"localhost\0garbage:17843"[..],
                name("localhost", 443, "localhost"),
            ),
            (
                b"localhost:17843\0garbage",
                name("localhost", 17843, "localhost:17843"),
            ),
            (b"127.0.0.1\0:17843", name("127.0.0.1", 443, "127.0.0.1")),
            (
                b"ipv4:127.0.0.1:17843\0junk",
                addresses(&["127.0.0.1:17843"], "127.0.0.1:17843"),
            ),
        ] {
            assert_eq!(parse(target).target, expected, "{target:?}");
        }
        let unix = |path: &[u8], abstract_names: bool| ChannelTarget::Unix {
            paths: vec![path.to_vec()],
            abstract_names,
        };
        assert_eq!(
            parse(b"unix:/tmp/rv62220/\xff.sock").target,
            unix(b"/tmp/rv62220/\xff.sock", false)
        );
        assert_eq!(
            parse("unix:/tmp/rv62220/%FF.sock").target,
            unix(b"/tmp/rv62220/\xff.sock", false)
        );
        assert_eq!(
            parse("unix:/tmp/rv62220/%EF%BF%BD.sock").target,
            unix("/tmp/rv62220/\u{fffd}.sock".as_bytes(), false)
        );
        assert_eq!(
            parse("unix-abstract:rv62220%FF").target,
            unix(b"rv62220\xff", true)
        );
        assert_eq!(
            parse(b"unix:/tmp/a.sock\0junk").target,
            unix(b"/tmp/a.sock", false)
        );
        assert_eq!(parse(b"a?\0b").target, not_valid("a?", "dns:///a?"));
    }

    #[test]
    fn the_numbers_are_read_as_grpc_reads_them() {
        assert_eq!(simple_atoi(" +17 "), Some(17));
        assert_eq!(simple_atoi("-0"), Some(0));
        assert_eq!(simple_atoi("2147483647"), Some(i32::MAX));
        assert_eq!(simple_atoi("-2147483648"), Some(i32::MIN));
        assert_eq!(simple_atoi("2147483648"), None);
        assert_eq!(simple_atoi("1 2"), None);
        assert_eq!(simple_atoi("0x10"), None);
        assert_eq!(sscanf_port(" 17x"), Some(17));
        assert_eq!(sscanf_port("0x10"), Some(0));
        assert_eq!(sscanf_port("x"), None);
        assert_eq!(sscanf_port("-1"), Some(-1));
        // A `long` past 32 bits is its `int`: 2^32 + 17 is 17.
        assert_eq!(sscanf_port("4294967313"), Some(17));
        assert_eq!(sscanf_port("99999999999999999999999"), Some(-1));
        assert_eq!(inet_pton_v4("127.0.0.1"), Some(Ipv4Addr::LOCALHOST));
        assert_eq!(inet_pton_v4("127.000.0.1"), None);
        assert_eq!(inet_pton_v4("127.1"), None);
        assert_eq!(ares_numeric_v4("127.000.0.1"), Some(Ipv4Addr::LOCALHOST));
        assert_eq!(
            ares_numeric_v4("127.0.0.010"),
            Some(Ipv4Addr::new(127, 0, 0, 10))
        );
        assert_eq!(ares_numeric_v4("127.1"), None);
        assert_eq!(ares_numeric_v4("256.0.0.1"), None);
        assert_eq!(percent_decode(b"%41%4%%zz%4h%3F"), b"A%4%%zz%4h?");
        assert_eq!(percent_decode(b"%FF%00"), [0xff, 0]);
        assert_eq!(
            percent_encode_authority(b"a/b %\x00\xff:[]@*,"),
            "a%2Fb%20%25%00%FF:[]@*,"
        );
    }
}
