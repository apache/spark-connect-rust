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

//! The configuration of the resolver, as c-ares reads that of the system for each channel of
//! gRPC's resolver: the search domains, `ndots`, the timeout and the attempts of
//! `/etc/resolv.conf` and of the environment (`LOCALDOMAIN`, `RES_OPTIONS`), the order of the hosts
//! file and of DNS (`/etc/nsswitch.conf`), the hosts file (`/etc/hosts`), and the aliases of
//! `HOSTALIASES`.

use std::collections::HashMap;
use std::net::IpAddr;
use std::time::Duration;

/// c-ares's timeout of the first attempt of a query when the system sets none.
const DEFAULT_TIMEOUT: Duration = Duration::from_millis(2000);

/// The longest timeout of the first attempt of a query that c-ares takes, of any configuration.
const MAX_TIMEOUT: Duration = Duration::from_millis(5000);

/// c-ares's count of the attempts of a query when the system sets none.
const DEFAULT_TRIES: u32 = 3;

/// Where the resolver looks a name up, in its order: the hosts file, or DNS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Source {
    Files,
    Dns,
}

/// The configuration of a lookup (see the module).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Config {
    /// The search domains, of their order.
    pub(crate) domains: Vec<String>,
    /// How many dots a name has at least for it to be asked as it is before the search domains.
    pub(crate) ndots: usize,
    /// The timeout of the first attempt of a query, doubled for each attempt after it.
    pub(crate) timeout: Duration,
    /// How many times a query is sent at most.
    pub(crate) tries: u32,
    /// Where a name is looked up, of its order.
    pub(crate) lookups: Vec<Source>,
    /// The hosts file.
    pub(crate) hosts: Hosts,
    /// The aliases of single names, of `HOSTALIASES`.
    pub(crate) aliases: HashMap<String, String>,
    /// How often the resolution of gRPC processes the timeouts of c-ares, besides when a socket
    /// of the lookup is ready: once a second.
    pub(crate) poll: Duration,
    /// How long gRPC's resolution runs before it cancels its lookups
    /// (`grpc.dns_ares_query_timeout`, 120 s).
    pub(crate) query_timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            domains: Vec::new(),
            ndots: 1,
            timeout: DEFAULT_TIMEOUT,
            tries: DEFAULT_TRIES,
            lookups: vec![Source::Files, Source::Dns],
            hosts: Hosts::default(),
            aliases: HashMap::new(),
            poll: Duration::from_secs(1),
            query_timeout: Duration::from_secs(120),
        }
    }
}

impl Config {
    /// The configuration of the system now, as c-ares reads it for a lookup.
    pub(crate) fn system() -> Self {
        let read = |path: &str| std::fs::read_to_string(path).ok();
        let env = |name: &str| std::env::var(name).ok();
        let hostname = local_hostname();
        let hosts_path = hosts_path();
        Self::of(
            &|path| match path {
                "hosts" => hosts_path.as_deref().and_then(read),
                "resolv.conf" => read("/etc/resolv.conf"),
                "nsswitch.conf" => read("/etc/nsswitch.conf"),
                _ => None,
            },
            &env,
            hostname.as_deref(),
        )
    }

    /// The configuration of the files of `file` (`resolv.conf`, `nsswitch.conf` and `hosts`, of
    /// their contents), of the environment `env` and of the host name `hostname`,
    /// as c-ares reads them.
    pub(crate) fn of(
        file: &dyn Fn(&str) -> Option<String>,
        env: &dyn Fn(&str) -> Option<String>,
        hostname: Option<&str>,
    ) -> Self {
        let mut config = Self::default();
        let mut domains: Option<Vec<String>> = None;
        let mut timeout = None;
        let mut tries = None;
        let mut lookups = None;
        if let Some(text) = file("resolv.conf") {
            for line in text.lines() {
                let line = line.trim_start();
                if line.starts_with(['#', ';']) {
                    continue;
                }
                let mut words = line.split_whitespace();
                match words.next() {
                    // `domain` is the search domain of a configuration of no `search`.
                    Some("domain") if domains.is_none() => {
                        if let Some(domain) = words.next() {
                            domains = Some(vec![domain.to_string()]);
                        }
                    }
                    Some("search") => {
                        domains = Some(words.map(str::to_string).collect());
                    }
                    Some("options") => {
                        apply_options(words, &mut config.ndots, &mut timeout, &mut tries);
                    }
                    Some("lookup") => lookups = sources(words).or(lookups),
                    _ => {}
                }
            }
        }
        if let Some(text) = file("nsswitch.conf") {
            for line in text.lines() {
                if let Some(rest) = line.trim_start().strip_prefix("hosts:") {
                    lookups = sources(rest.split_whitespace()).or(lookups);
                }
            }
        }
        // The environment overrides the files: of `LOCALDOMAIN`, its first domain.
        if let Some(local) = env("LOCALDOMAIN") {
            domains = Some(
                local
                    .split_whitespace()
                    .take(1)
                    .map(str::to_string)
                    .collect(),
            );
        }
        if let Some(options) = env("RES_OPTIONS") {
            apply_options(
                options.split_whitespace(),
                &mut config.ndots,
                &mut timeout,
                &mut tries,
            );
        }
        // Of no search domain, the domain of the host name, as c-ares takes it.
        config.domains = domains.unwrap_or_default();
        if config.domains.is_empty() {
            if let Some((_, domain)) = hostname.and_then(|name| name.split_once('.')) {
                config.domains.push(domain.to_string());
            }
        }
        if let Some(timeout) = timeout.filter(|timeout| !timeout.is_zero()) {
            config.timeout = timeout;
        }
        config.timeout = config.timeout.min(MAX_TIMEOUT);
        if let Some(tries) = tries.filter(|&tries| tries > 0) {
            config.tries = tries;
        }
        if let Some(lookups) = lookups {
            config.lookups = lookups;
        }
        if let Some(text) = file("hosts") {
            config.hosts = Hosts::parse(&text);
        }
        if let Some(path) = env("HOSTALIASES") {
            if let Ok(text) = std::fs::read_to_string(path) {
                config.aliases = parse_aliases(&text);
            }
        }
        config
    }

    /// The alias of `name`, a name of no dot, of `HOSTALIASES`, if it has one.
    pub(crate) fn alias(&self, name: &str) -> Option<&str> {
        if name.contains('.') {
            return None;
        }
        self.aliases
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }
}

/// Apply the options `words` of `resolv.conf` or of `RES_OPTIONS`, as c-ares takes them: `ndots`,
/// `timeout` (in seconds) and `attempts`.
fn apply_options<'a>(
    words: impl Iterator<Item = &'a str>,
    ndots: &mut usize,
    timeout: &mut Option<Duration>,
    tries: &mut Option<u32>,
) {
    for word in words {
        let Some((key, value)) = word.split_once(':') else {
            continue;
        };
        let Ok(value) = value.parse::<u32>() else {
            continue;
        };
        match key {
            "ndots" => *ndots = value as usize,
            "timeout" => *timeout = Some(Duration::from_secs(u64::from(value))),
            "attempts" => *tries = Some(value),
            _ => {}
        }
    }
}

/// The order of the sources of `words`, of a line of `nsswitch.conf` or a `lookup` of
/// `resolv.conf`, each once, as c-ares takes them: `dns`, `bind`, `resolv` and `resolve` of DNS,
/// `files`, `file` and `local` of the hosts file. None of none of them.
fn sources<'a>(words: impl Iterator<Item = &'a str>) -> Option<Vec<Source>> {
    let mut order = Vec::new();
    for word in words {
        let of = |names: &[&str]| names.iter().any(|name| name.eq_ignore_ascii_case(word));
        let source = if of(&["dns", "bind", "resolv", "resolve"]) {
            Source::Dns
        } else if of(&["files", "file", "local"]) {
            Source::Files
        } else {
            continue;
        };
        if !order.contains(&source) {
            order.push(source);
        }
    }
    (!order.is_empty()).then_some(order)
}

/// The aliases of the file of `HOSTALIASES`: lines of a name and its alias, the first of a name.
fn parse_aliases(text: &str) -> HashMap<String, String> {
    let mut aliases = HashMap::new();
    for line in text.lines() {
        let mut words = line.split_whitespace();
        if let (Some(name), Some(alias)) = (words.next(), words.next()) {
            aliases
                .entry(name.to_ascii_lowercase())
                .or_insert_with(|| alias.to_string());
        }
    }
    aliases
}

/// The hosts file, as c-ares reads it: its entries of addresses and names, of a line merged into
/// the entry of one of its addresses, else into that of one of its names, and of each name the
/// first entry of it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Hosts {
    /// The addresses of each entry, of their order.
    entries: Vec<Vec<IpAddr>>,
    /// The entry of each address and of each name, in lower case.
    by_address: HashMap<IpAddr, usize>,
    by_name: HashMap<String, usize>,
}

impl Hosts {
    /// The hosts of the text of a hosts file: lines of an address and names, of comments after
    /// `#`.
    pub(crate) fn parse(text: &str) -> Self {
        let mut hosts = Self::default();
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or_default();
            let mut words = line.split_whitespace();
            let Some(address) = words.next().and_then(parse_address) else {
                continue;
            };
            let names: Vec<String> = words.map(|name| name.to_ascii_lowercase()).collect();
            if names.is_empty() {
                continue;
            }
            let entry = match hosts.by_address.get(&address) {
                Some(&entry) => entry,
                None => match names.iter().find_map(|name| hosts.by_name.get(name)) {
                    Some(&entry) => {
                        hosts.entries[entry].push(address);
                        hosts.by_address.entry(address).or_insert(entry);
                        entry
                    }
                    None => {
                        hosts.entries.push(vec![address]);
                        let entry = hosts.entries.len() - 1;
                        hosts.by_address.insert(address, entry);
                        entry
                    }
                },
            };
            for name in names {
                hosts.by_name.entry(name).or_insert(entry);
            }
        }
        hosts
    }

    /// The addresses of `name` (of any case), of the family of `v6`, of their order, if the file
    /// has the name.
    pub(crate) fn lookup(&self, name: &str, v6: bool) -> Option<Vec<IpAddr>> {
        let entry = self.by_name.get(&name.to_ascii_lowercase())?;
        Some(
            self.entries[*entry]
                .iter()
                .filter(|address| address.is_ipv6() == v6)
                .copied()
                .collect(),
        )
    }
}

/// The address of the text of an entry of the hosts file, as c-ares reads it: IPv4 of four
/// decimal numbers, of leading zeros too, or IPv6 of no zone.
fn parse_address(text: &str) -> Option<IpAddr> {
    if let Some(address) = crate::grpc_target::ares_numeric_v4(text) {
        return Some(IpAddr::V4(address));
    }
    text.parse::<std::net::Ipv6Addr>().ok().map(IpAddr::V6)
}

/// The path of the hosts file of the system.
fn hosts_path() -> Option<String> {
    #[cfg(windows)]
    {
        let root = std::env::var("SystemRoot").ok()?;
        Some(format!("{root}\\System32\\drivers\\etc\\hosts"))
    }
    #[cfg(not(windows))]
    {
        Some("/etc/hosts".to_string())
    }
}

/// The host name of the system, as `gethostname` gives it.
fn local_hostname() -> Option<String> {
    #[cfg(unix)]
    {
        let mut buffer = [0u8; 256];
        // SAFETY: `buffer` is writable for its length, which the call is given.
        let result = unsafe { libc::gethostname(buffer.as_mut_ptr().cast(), buffer.len()) };
        if result != 0 {
            return None;
        }
        let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
        Some(String::from_utf8_lossy(&buffer[..end]).into_owned())
    }
    #[cfg(not(unix))]
    {
        None
    }
}
