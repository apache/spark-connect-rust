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
//! Of each target, the channel takes gRPC's names of it besides what it connects to (see
//! [`TargetNames`]): its default authority, of its calls and of the name that its TLS verifies,
//! and the server of the tunnel of a proxy, as gRPC's proxy mapper takes it.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV6};

use crate::channel::{ChannelTarget, TargetNames};

/// The code of gRPC's `UNKNOWN`, of a target that is not valid.
const UNKNOWN: i32 = 2;

/// The code of gRPC's `INTERNAL`, of a resolver that fails of a status that a resolver does not
/// give (`Illegal status code from resolver`).
const INTERNAL: i32 = 13;

/// The schemes of gRPC's resolvers.
const SCHEMES: [&str; 5] = ["dns", "ipv4", "ipv6", "unix", "unix-abstract"];

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

/// What a channel connects to of the target `target`, as grpcio's channel reads it (see the
/// module).
pub fn parse(target: &str) -> Parsed {
    let not_valid = |canonical: &str| {
        refused(
            UNKNOWN,
            format!(
                "Failed to create channel to '{target}':the target uri is not valid: {canonical}"
            ),
        )
    };
    let (uri, canonical) = match Uri::parse(target) {
        Ok(uri) if SCHEMES.contains(&uri.scheme.as_str()) => (uri, target.to_string()),
        _ => {
            let canonical = format!("{DEFAULT_PREFIX}{target}");
            match Uri::parse(&canonical) {
                Ok(uri) => (uri, canonical),
                Err(()) => return not_valid(&canonical),
            }
        }
    };
    let parsed = match uri.scheme.as_str() {
        "dns" => dns(&uri),
        "ipv4" => addresses(&uri, false),
        "ipv6" => addresses(&uri, true),
        "unix" => sockets(&uri, false),
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
    let path = uri.path.strip_prefix('/').unwrap_or(&uri.path);
    let proxied = (uri.scheme != "unix").then(|| match split_host_port(path) {
        Some((host, port)) => {
            let port = port.filter(|port| !port.is_empty()).unwrap_or("443");
            let tunnel = if host.contains(':') {
                format!("[{host}]:{port}")
            } else {
                format!("{host}:{port}")
            };
            (host.to_string(), tunnel)
        }
        None => (path.to_string(), path.to_string()),
    });
    TargetNames {
        authority: percent_encode_authority(path.as_bytes()),
        proxied,
    }
}

/// `bytes` percent-encoded as gRPC's `URI::PercentEncodeAuthority` encodes an authority: of its
/// unreserved characters, its sub-delims, `:`, `[`, `]` and `@` as they are, and of any other
/// byte as `%` and two upper-case hexadecimal digits.
pub(crate) fn percent_encode_authority(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        if b.is_ascii_alphanumeric() || b"-._~!$&'()*+,;=:[]@".contains(&b) {
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
    let name = uri.path.strip_prefix('/').unwrap_or(&uri.path);
    if name.is_empty() {
        return None;
    }
    if !uri.authority.is_empty() && parse_host_port(&uri.authority, None).is_none() {
        return Some(refused(
            INTERNAL,
            format!(
                "Illegal status code from resolver; original status: INVALID_ARGUMENT: Cannot \
                 parse authority: {}",
                uri.authority
            ),
        ));
    }
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
        },
    })
}

/// A target of `ipv4:` or of `ipv6:` (of `v6`), of its addresses, or None of one that is not
/// valid: of an authority, or of an address that is not one of the family and a port.
fn addresses(uri: &Uri, v6: bool) -> Option<Parsed> {
    if !uri.authority.is_empty() {
        return None;
    }
    let mut addresses = Vec::new();
    for entry in uri.path.split(',').filter(|entry| !entry.is_empty()) {
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
            authority: uri.path.strip_prefix('/').unwrap_or(&uri.path).to_string(),
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
    for entry in uri.path.split(',').filter(|entry| !entry.is_empty()) {
        if entry.len() > MAX_UNIX_PATH {
            return None;
        }
        paths.push(entry.as_bytes().to_vec());
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
/// its path, percent-decoded. The query and the fragment are dropped, once checked.
#[derive(Debug)]
struct Uri {
    scheme: String,
    authority: String,
    path: String,
}

impl Uri {
    fn parse(text: &str) -> Result<Self, ()> {
        let colon = text.find(':').ok_or(())?;
        let scheme = &text[..colon];
        let scheme_char = |c: char| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.');
        if scheme.is_empty()
            || !scheme.chars().all(scheme_char)
            || !scheme.starts_with(|c: char| c.is_ascii_alphabetic())
        {
            return Err(());
        }
        let mut rest = &text[colon + 1..];
        let mut authority = String::new();
        if let Some(after) = rest.strip_prefix("//") {
            let end = after.find(['/', '?', '#']).unwrap_or(after.len());
            authority = percent_decode(&after[..end]);
            rest = &after[end..];
        }
        let end = rest.find(['?', '#']).unwrap_or(rest.len());
        let path = percent_decode(&rest[..end]);
        rest = &rest[end..];
        if let Some(after) = rest.strip_prefix('?') {
            let end = after.find('#').unwrap_or(after.len());
            let query = &after[..end];
            if query.is_empty() || !is_query_or_fragment(query) {
                return Err(());
            }
            rest = &after[end..];
        }
        if let Some(fragment) = rest.strip_prefix('#') {
            if !is_query_or_fragment(fragment) {
                return Err(());
            }
        }
        Ok(Self {
            scheme: scheme.to_ascii_lowercase(),
            authority,
            path,
        })
    }
}

/// Whether `text` is of the characters of a query or a fragment of RFC 3986 and `%`, as gRPC's
/// parser takes them: unreserved, sub-delims, `:`, `@`, `/` and `?`.
fn is_query_or_fragment(text: &str) -> bool {
    text.chars().all(|c| {
        c.is_ascii_alphanumeric()
            || matches!(
                c,
                '-' | '.'
                    | '_'
                    | '~'
                    | '!'
                    | '$'
                    | '&'
                    | '\''
                    | '('
                    | ')'
                    | '*'
                    | '+'
                    | ','
                    | ';'
                    | '='
                    | ':'
                    | '@'
                    | '/'
                    | '?'
                    | '%'
            )
    })
}

/// `text` percent-decoded as gRPC's parser decodes it: a `%` of two hexadecimal digits is their
/// byte, and any other `%` stays. A decoded byte that is not of UTF-8 is its replacement.
fn percent_decode(text: &str) -> String {
    if !text.contains('%') {
        return text.to_string();
    }
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |at: usize| bytes.get(at).and_then(|&b| (b as char).to_digit(16));
        match (bytes[i], hex(i + 1), hex(i + 2)) {
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
    String::from_utf8_lossy(&out).into_owned()
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
                name("localhost", 15002, "localhost:15002"),
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
        for target in [
            "dns://127.0.0.1:53/localhost:15002",
            "dns://127.0.0.1%3A53/localhost:15002",
            "dns://%31%32%37.0.0.1:53/localhost:15002",
            "dns://127.0.0.1:53x/localhost:15002",
            "dns://[::1]:53/localhost:15002",
        ] {
            assert_eq!(
                parse(target).target,
                name("localhost", 15002, "localhost:15002"),
                "{target}"
            );
        }
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
                proxied("127.0.0.1\0", "127.0.0.1\0:18620"),
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
        assert_eq!(percent_decode("%41%4%%zz%4h%3F"), "A%4%%zz%4h?");
        assert_eq!(
            percent_encode_authority(b"a/b %\x00\xff:[]@*,"),
            "a%2Fb%20%25%00%FF:[]@*,"
        );
    }
}
