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

//! The resolution of the name of a target of gRPC whose authority names a DNS server
//! (`dns://<server>/<name>`), as grpcio's channel resolves it: gRPC's resolver asks c-ares (1.34)
//! to look the name up through that server alone, in a channel of c-ares of its own, for each
//! family, and sorts the addresses of both; its `pick_first` tries them in turn.
//!
//! - The families are looked up at once, IPv4 first, and IPv6 too when the system has an IPv6
//!   loopback (`IsIpv6LoopbackAvailable`). Each is looked up as c-ares's `ares_gethostbyname`
//!   does: a name of `.onion` is not found; a name of no dot that `HOSTALIASES` aliases is asked
//!   as its alias alone; else the name is looked up in the hosts file and in DNS, in the order of
//!   `/etc/nsswitch.conf`, where `localhost` and the names under it are the loopback of the hosts
//!   file or of the system, and are not asked.
//! - DNS is asked for the records of the family alone (`A` or `AAAA`), of the name as it is (of a
//!   last `.` the name alone) and of each search domain, the name first if it has `ndots` dots at
//!   least, until an answer has an address: an answer of `NXDOMAIN` or of no address of the family
//!   asks the next name, any other failure ends the family, but `SERVFAIL` and `REFUSED` of a name
//!   of one label. Once all are asked, the family fails of `ENODATA` if an answer had no data,
//!   else of `ENOTFOUND`.
//! - A query is sent over UDP, of recursion desired, no EDNS and an ID of its own, which its
//!   retries keep, and again over TCP when its answer is truncated. An answer of `SERVFAIL`,
//!   `NOTIMP` or `REFUSED`, a timeout, or the failure of the connection that it was sent on, sends
//!   it again, `tries` times at most. The timeout of its first attempt is that of the system (2 s,
//!   5 s at most), and each attempt after it has twice the timeout of the previous one but a
//!   random part of a half at most. gRPC's resolver has c-ares process its timeouts once a second
//!   and whenever a socket of the lookup is ready, so a query is sent again then. A TCP connection
//!   that fails to connect cancels the lookup, as gRPC's resolver cancels its channel of c-ares of
//!   a socket that is shut down while it waits to read it.
//! - The addresses are those of both families, in the order in which the families ended; of none,
//!   the lookup fails of the failure of the family that ended last. gRPC sorts them (RFC 6724),
//!   and `pick_first` interleaves their families (see [`sort`]). A lookup that runs for 120 s,
//!   gRPC's timeout of a resolution, is cancelled.
//!
//! The connector resolves through this module only the name of a target of a DNS server (see
//! `crate::connector`), of [`lookup`] alone, of the configuration of the system that [`Config`]
//! reads; the messages of DNS are those of `message`, so another implementation of the queries
//! replaces that module, and another resolver [`lookup`].

mod config;
#[cfg(target_os = "macos")]
mod macos;
mod message;
mod sort;
mod system;
#[cfg(test)]
pub(crate) mod tests;
#[cfg(windows)]
mod windows;

pub(crate) use config::Config;
use config::Source;

use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, SocketAddrV6};
use std::pin::Pin;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, UdpSocket};
use tokio::time::Instant;

/// The failure of a lookup, as c-ares's status of it, of the text of `ares_strerror`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Status {
    /// `ARES_ENODATA`: the name has no address of the family.
    NoData,
    /// `ARES_EFORMERR`: an answer of `FORMERR`.
    FormErr,
    /// `ARES_ESERVFAIL`: answers of `SERVFAIL`.
    ServFail,
    /// `ARES_ENOTFOUND`: the name does not exist.
    NotFound,
    /// `ARES_ENOTIMP`: answers of `NOTIMP`.
    NotImp,
    /// `ARES_EREFUSED`: answers of `REFUSED`.
    Refused,
    /// `ARES_EBADNAME`: a name that c-ares does not ask for.
    BadName,
    /// `ARES_ECONNREFUSED`: the connections to the server failed.
    ConnRefused,
    /// `ARES_ETIMEOUT`: the attempts timed out.
    Timeout,
    /// `ARES_ECANCELLED`: the lookup was cancelled.
    Cancelled,
}

impl Status {
    /// The text of c-ares of this failure (`ares_strerror`).
    pub(crate) fn text(self) -> &'static str {
        match self {
            Self::NoData => "DNS server returned answer with no data",
            Self::FormErr => "DNS server claims query was misformatted",
            Self::ServFail => "DNS server returned general failure",
            Self::NotFound => "Domain name not found",
            Self::NotImp => "DNS server does not implement requested operation",
            Self::Refused => "DNS server refused query",
            Self::BadName => "Misformatted domain name",
            Self::ConnRefused => "Could not contact DNS servers",
            Self::Timeout => "Timeout while contacting DNS servers",
            Self::Cancelled => "DNS query cancelled",
        }
    }
}

/// The addresses of `name` through the DNS server `server`, with the port `port`, of the
/// configuration `config`, of both families if `ipv6`, else of IPv4, in the order in which grpcio's
/// channel tries them, or the failure of the lookup (see the module).
pub(crate) async fn lookup(
    server: SocketAddr,
    name: &str,
    port: u16,
    config: &Config,
    ipv6: bool,
) -> Result<Vec<SocketAddr>, Status> {
    // gRPC's resolver gives c-ares the address and the port of the server, of no zone.
    let server = match server {
        SocketAddr::V6(server) => {
            SocketAddr::V6(SocketAddrV6::new(*server.ip(), server.port(), 0, 0))
        }
        server => server,
    };
    let mut lookup = Lookup::new(server, config);
    lookup.start(name, false);
    if ipv6 {
        lookup.start(name, true);
    }
    lookup.run().await;
    let mut addresses = Vec::new();
    let mut failure = Status::ConnRefused;
    for &index in &lookup.ended {
        match &lookup.families[index].outcome {
            Some(Ok(found)) => addresses.extend(found.iter().map(|&ip| SocketAddr::new(ip, port))),
            Some(Err(status)) => failure = *status,
            None => {}
        }
    }
    if addresses.is_empty() {
        return Err(failure);
    }
    Ok(sort::order(addresses, &sort::system_source))
}

/// The lookup of one family, as c-ares's `ares_gethostbyname` of it (see the module).
struct Family {
    v6: bool,
    /// The name as it is given, which the hosts file and the loopback are looked up of.
    name: String,
    /// The names that DNS is asked for, of their order, and the next one.
    names: Vec<String>,
    next_name: usize,
    /// The sources that are looked up, of their order, and the next one.
    lookups: Vec<Source>,
    next_lookup: usize,
    /// Whether an answer had no data of the family.
    nodata: bool,
    outcome: Option<Result<Vec<IpAddr>, Status>>,
}

/// A query in flight.
struct Query {
    id: u16,
    family: usize,
    name: String,
    qtype: u16,
    packet: Vec<u8>,
    /// How many attempts failed.
    tries: u32,
    deadline: Instant,
    tcp: bool,
    /// The failure of its last attempt.
    error: Status,
}

/// The TCP connection to the server, which the queries of truncated answers are sent on.
enum Tcp {
    None,
    /// Being made, of the bytes to write once it is.
    Connecting(
        Pin<Box<dyn Future<Output = std::io::Result<TcpStream>> + Send>>,
        Vec<u8>,
    ),
    /// Made, of the bytes read that are not a whole message yet, and of the bytes to write.
    Connected {
        stream: TcpStream,
        read: Vec<u8>,
        write: Vec<u8>,
    },
}

/// The UDP socket of the lookup: to receive of the runtime, and to send at once, of a clone.
struct Udp {
    receive: UdpSocket,
    send: std::net::UdpSocket,
}

/// What wakes a lookup.
enum Event {
    Udp(std::io::Result<usize>),
    TcpConnected(std::io::Result<TcpStream>),
    TcpRead(std::io::Result<usize>),
    TcpWrote(std::io::Result<usize>),
    Tick,
    Cancel,
}

/// The lookups of the families of a name, as one channel of c-ares of gRPC's resolver.
struct Lookup<'a> {
    server: SocketAddr,
    config: &'a Config,
    families: Vec<Family>,
    /// The families of an outcome, in the order in which they ended.
    ended: Vec<usize>,
    queries: Vec<Query>,
    udp: Option<Udp>,
    tcp: Tcp,
}

impl<'a> Lookup<'a> {
    fn new(server: SocketAddr, config: &'a Config) -> Self {
        Self {
            server,
            config,
            families: Vec::new(),
            ended: Vec::new(),
            queries: Vec::new(),
            udp: None,
            tcp: Tcp::None,
        }
    }

    /// Start the lookup of the family of `v6` of `name`.
    fn start(&mut self, name: &str, v6: bool) {
        let index = self.families.len();
        self.families.push(Family {
            v6,
            name: name.to_string(),
            names: Vec::new(),
            next_name: 0,
            lookups: self.config.lookups.clone(),
            next_lookup: 0,
            nodata: false,
            outcome: None,
        });
        if is_onion(name) {
            return self.end(index, Err(Status::NotFound));
        }
        self.families[index].names = self.names_of(name);
        self.next_lookup(index, Status::ConnRefused);
    }

    /// The names that DNS is asked for of `name`, as c-ares's `ares_search_name_list` gives them.
    fn names_of(&self, name: &str) -> Vec<String> {
        if let Some(alias) = self.config.alias(name) {
            return vec![alias.to_string()];
        }
        let domains = &self.config.domains;
        if domains.is_empty() || name.ends_with('.') {
            return vec![name.to_string()];
        }
        let as_is_first = name.matches('.').count() >= self.config.ndots;
        let mut names = Vec::with_capacity(domains.len() + 1);
        if as_is_first {
            names.push(name.to_string());
        }
        names.extend(domains.iter().map(|domain| format!("{name}.{domain}")));
        if !as_is_first {
            names.push(name.to_string());
        }
        names
    }

    /// End the family `index` of `outcome`.
    fn end(&mut self, index: usize, outcome: Result<Vec<IpAddr>, Status>) {
        let family = &mut self.families[index];
        if family.outcome.is_none() {
            family.outcome = Some(outcome);
            self.ended.push(index);
        }
    }

    /// Look the family `index` up in its next source, of the failure `status` so far, as c-ares's
    /// `next_lookup`.
    fn next_lookup(&mut self, index: usize, status: Status) {
        loop {
            let family = &mut self.families[index];
            let Some(&source) = family.lookups.get(family.next_lookup) else {
                return self.end(index, Err(status));
            };
            match source {
                Source::Dns => {
                    if !is_localhost(&family.name) {
                        if let Some(name) = family.names.get(family.next_name).cloned() {
                            family.next_name += 1;
                            return self.ask(index, name);
                        }
                    }
                }
                Source::Files => {
                    if let Some(found) = self.file_lookup(index) {
                        return self.end(index, Ok(found));
                    }
                }
            }
            self.families[index].next_lookup += 1;
        }
    }

    /// The addresses of the family `index` of the hosts file, else the loopback of a name of the
    /// loopback (`ares_addrinfo_localhost`), if any.
    fn file_lookup(&self, index: usize) -> Option<Vec<IpAddr>> {
        let family = &self.families[index];
        if let Some(found) = self.config.hosts.lookup(&family.name, family.v6) {
            if !found.is_empty() {
                return Some(found);
            }
        }
        is_localhost(&family.name).then(|| loopback_addresses(family.v6))
    }

    /// Ask the server for the records of the family `index` of `name`.
    fn ask(&mut self, index: usize, name: String) {
        let qtype = if self.families[index].v6 {
            message::TYPE_AAAA
        } else {
            message::TYPE_A
        };
        let id = self.new_id();
        let Some(packet) = message::query(id, &name, qtype) else {
            return self.on_answer(index, &name, Err(Status::BadName));
        };
        self.queries.push(Query {
            id,
            family: index,
            name,
            qtype,
            packet,
            tries: 0,
            deadline: Instant::now(),
            tcp: false,
            error: Status::Timeout,
        });
        let at = self.queries.len() - 1;
        self.send(at, Instant::now());
    }

    /// An ID of no query in flight.
    fn new_id(&self) -> u16 {
        loop {
            let bytes = *uuid::Uuid::new_v4().as_bytes();
            let id = u16::from_be_bytes([bytes[0], bytes[1]]);
            if !self.queries.iter().any(|query| query.id == id) {
                return id;
            }
        }
    }

    /// The timeout of the attempt after `tries` attempts of a query, as c-ares's
    /// `ares_calc_query_timeout` of one server: the timeout of the system, doubled for each attempt
    /// after the first one, but a random part of a half at most.
    fn timeout(&self, tries: u32) -> std::time::Duration {
        let base = self.config.timeout;
        if tries == 0 {
            return base;
        }
        let doubled = base * 2u32.saturating_pow(tries.min(30));
        let bytes = *uuid::Uuid::new_v4().as_bytes();
        let random = f64::from(u16::from_be_bytes([bytes[0], bytes[1]])) / f64::from(u16::MAX);
        doubled.mul_f64(1.0 - random * 0.5).max(base)
    }

    /// Send the query `at`, of its transport, with the deadline of its attempt.
    fn send(&mut self, at: usize, now: Instant) {
        let timeout = self.timeout(self.queries[at].tries);
        self.queries[at].deadline = now + timeout;
        if self.queries[at].tcp {
            let mut framed = (self.queries[at].packet.len() as u16)
                .to_be_bytes()
                .to_vec();
            framed.extend_from_slice(&self.queries[at].packet);
            match &mut self.tcp {
                Tcp::Connected { write, .. } => write.extend_from_slice(&framed),
                Tcp::Connecting(_, pending) => pending.extend_from_slice(&framed),
                Tcp::None => {
                    let server = self.server;
                    self.tcp = Tcp::Connecting(Box::pin(TcpStream::connect(server)), framed);
                }
            }
            return;
        }
        if self.udp.is_none() {
            match open_udp(self.server) {
                Ok(socket) => self.udp = Some(socket),
                Err(_) => return self.attempt_failed(at, Status::ConnRefused, now),
            }
        }
        let sent = self
            .udp
            .as_ref()
            .map(|udp| udp.send.send(&self.queries[at].packet));
        if !matches!(sent, Some(Ok(_))) {
            self.udp = None;
            self.connection_failed(false, now);
        }
    }

    /// The attempt of the query `at` failed of `status`: it is sent again, if it has attempts
    /// left, else its family is told of its failure (`ares_requeue_query`).
    fn attempt_failed(&mut self, at: usize, status: Status, now: Instant) {
        let query = &mut self.queries[at];
        query.error = status;
        query.tries += 1;
        if query.tries < self.config.tries {
            return self.send(at, now);
        }
        let query = self.queries.remove(at);
        self.on_answer(query.family, &query.name, Err(query.error));
    }

    /// The connection of TCP if `tcp`, else of UDP, failed: its queries are sent again.
    fn connection_failed(&mut self, tcp: bool, now: Instant) {
        let ids: Vec<u16> = self
            .queries
            .iter()
            .filter(|query| query.tcp == tcp)
            .map(|query| query.id)
            .collect();
        for id in ids {
            if let Some(at) = self.queries.iter().position(|query| query.id == id) {
                self.attempt_failed(at, Status::ConnRefused, now);
            }
        }
    }

    /// Cancel the lookup, as gRPC's resolver cancels its channel of c-ares (`ares_cancel`).
    fn cancel(&mut self) {
        for query in std::mem::take(&mut self.queries) {
            self.end(query.family, Err(Status::Cancelled));
        }
    }

    /// The answer `packet` on the connection of TCP if `tcp`, else of UDP.
    fn on_packet(&mut self, packet: &[u8], tcp: bool, now: Instant) {
        let Some(response) = message::parse(packet) else {
            return;
        };
        let Some(at) = self
            .queries
            .iter()
            .position(|query| query.id == response.id)
        else {
            return;
        };
        let query = &self.queries[at];
        let same_question = response.questions.len() == 1 && {
            let (name, qtype, class) = &response.questions[0];
            name.eq_ignore_ascii_case(query.name.strip_suffix('.').unwrap_or(&query.name))
                && *qtype == query.qtype
                && *class == 1
        };
        if !same_question {
            return;
        }
        if response.truncated && !tcp {
            self.queries[at].tcp = true;
            return self.send(at, now);
        }
        let status = match response.rcode {
            message::RCODE_SERVFAIL => Some(Status::ServFail),
            message::RCODE_NOTIMP => Some(Status::NotImp),
            message::RCODE_REFUSED => Some(Status::Refused),
            _ => None,
        };
        if let Some(status) = status {
            return self.attempt_failed(at, status, now);
        }
        let query = self.queries.remove(at);
        let outcome = match response.rcode {
            message::RCODE_NOERROR if response.answers.is_empty() => Err(Status::NoData),
            message::RCODE_FORMERR => Err(Status::FormErr),
            message::RCODE_NXDOMAIN => Err(Status::NotFound),
            _ => Ok(addresses_of(&response.answers)),
        };
        self.on_answer(query.family, &query.name, outcome);
    }

    /// The outcome of the query of `name` of the family `index`: the addresses of its answer, or
    /// its failure, as c-ares's `host_callback` takes it.
    fn on_answer(&mut self, index: usize, name: &str, outcome: Result<Vec<IpAddr>, Status>) {
        match outcome {
            Ok(found) if !found.is_empty() => self.end(index, Ok(found)),
            Ok(_) | Err(Status::NoData) => {
                self.families[index].nodata = true;
                self.next_lookup(index, Status::NoData);
            }
            Err(Status::Cancelled) => self.end(index, Err(Status::Cancelled)),
            Err(Status::NotFound) => {
                let status = if self.families[index].nodata {
                    Status::NoData
                } else {
                    Status::NotFound
                };
                self.next_lookup(index, status);
            }
            // c-ares asks on of a name of one label whose server fails or refuses it.
            Err(status @ (Status::ServFail | Status::Refused)) if labels(name) == 1 => {
                let status = if self.families[index].nodata {
                    Status::NoData
                } else {
                    status
                };
                self.next_lookup(index, status);
            }
            Err(status) => self.end(index, Err(status)),
        }
    }

    /// Whether every family has an outcome.
    fn done(&self) -> bool {
        self.families.iter().all(|family| family.outcome.is_some())
    }

    /// Process the attempts whose deadline has passed at `now` (`process_timeouts`).
    fn process_timeouts(&mut self, now: Instant) {
        loop {
            let Some(at) = self
                .queries
                .iter()
                .enumerate()
                .filter(|(_, query)| query.deadline <= now)
                .min_by_key(|(_, query)| query.deadline)
                .map(|(at, _)| at)
            else {
                return;
            };
            self.attempt_failed(at, Status::Timeout, now);
        }
    }

    /// Run the queries until every family has an outcome, or the timeout of the resolution of
    /// gRPC (`grpc.dns_ares_query_timeout`) cancels them.
    async fn run(&mut self) {
        let start = Instant::now();
        let mut next_tick = start + self.config.poll;
        let cancel_at = start + self.config.query_timeout;
        let mut buffer = vec![0u8; 65536];
        let mut tcp_buffer = vec![0u8; 65536];
        while !self.done() {
            let event = {
                let udp = self.udp.as_ref().map(|udp| &udp.receive);
                let tcp = &mut self.tcp;
                let tcp_active = !matches!(tcp, Tcp::None);
                tokio::select! {
                    received = async { udp.expect("a socket").recv(&mut buffer).await }, if udp.is_some() => {
                        Event::Udp(received)
                    }
                    event = tcp_event(tcp, &mut tcp_buffer), if tcp_active => event,
                    _ = tokio::time::sleep_until(next_tick) => Event::Tick,
                    _ = tokio::time::sleep_until(cancel_at) => Event::Cancel,
                }
            };
            let now = Instant::now();
            match event {
                Event::Cancel => {
                    self.cancel();
                    continue;
                }
                Event::Tick => next_tick = now + self.config.poll,
                Event::Udp(Ok(length)) => {
                    let mut packets = vec![buffer[..length].to_vec()];
                    let mut failed = false;
                    if let Some(udp) = &self.udp {
                        loop {
                            match udp.receive.try_recv(&mut buffer) {
                                Ok(length) => packets.push(buffer[..length].to_vec()),
                                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                                    break
                                }
                                Err(_) => {
                                    failed = true;
                                    break;
                                }
                            }
                        }
                    }
                    for packet in packets {
                        self.on_packet(&packet, false, now);
                    }
                    if failed {
                        self.udp = None;
                        self.connection_failed(false, now);
                    }
                }
                Event::Udp(Err(_)) => {
                    self.udp = None;
                    self.connection_failed(false, now);
                }
                Event::TcpConnected(Ok(stream)) => {
                    let write = match std::mem::replace(&mut self.tcp, Tcp::None) {
                        Tcp::Connecting(_, pending) => pending,
                        _ => Vec::new(),
                    };
                    let _ = stream.set_nodelay(true);
                    self.tcp = Tcp::Connected {
                        stream,
                        read: Vec::new(),
                        write,
                    };
                }
                Event::TcpWrote(Ok(written)) if written > 0 => {
                    if let Tcp::Connected { write, .. } = &mut self.tcp {
                        write.drain(..written.min(write.len()));
                    }
                }
                // As gRPC's resolver, which shuts the socket down while it waits to read it.
                Event::TcpConnected(Err(_)) => {
                    self.tcp = Tcp::None;
                    self.cancel();
                }
                Event::TcpRead(Ok(length)) if length > 0 => {
                    let mut messages = Vec::new();
                    if let Tcp::Connected { read, .. } = &mut self.tcp {
                        read.extend_from_slice(&tcp_buffer[..length]);
                        while read.len() >= 2 {
                            let size = usize::from(u16::from_be_bytes([read[0], read[1]]));
                            if read.len() < 2 + size {
                                break;
                            }
                            messages.push(read[2..2 + size].to_vec());
                            read.drain(..2 + size);
                        }
                    }
                    for message in messages {
                        self.on_packet(&message, true, now);
                    }
                }
                Event::TcpRead(_) | Event::TcpWrote(_) => {
                    self.tcp = Tcp::None;
                    self.connection_failed(true, now);
                }
            }
            self.process_timeouts(now);
        }
    }
}

/// The next event of the TCP connection `tcp`: its connection, the write of what it has to write,
/// or else a read of it into `buffer`.
async fn tcp_event(tcp: &mut Tcp, buffer: &mut [u8]) -> Event {
    match tcp {
        Tcp::Connecting(connecting, _) => Event::TcpConnected(connecting.await),
        Tcp::Connected { stream, write, .. } if !write.is_empty() => {
            Event::TcpWrote(stream.write(write).await)
        }
        Tcp::Connected { stream, .. } => Event::TcpRead(stream.read(buffer).await),
        Tcp::None => std::future::pending().await,
    }
}

/// A socket of UDP connected to `server`, as c-ares's, of a port of the system.
fn open_udp(server: SocketAddr) -> std::io::Result<Udp> {
    let local: SocketAddr = match server {
        SocketAddr::V4(_) => (Ipv4Addr::UNSPECIFIED, 0).into(),
        SocketAddr::V6(_) => (Ipv6Addr::UNSPECIFIED, 0).into(),
    };
    let socket = std::net::UdpSocket::bind(local)?;
    socket.connect(server)?;
    socket.set_nonblocking(true)?;
    Ok(Udp {
        send: socket.try_clone()?,
        receive: UdpSocket::from_std(socket)?,
    })
}

/// The addresses of the records `answers` of a query of one family, as c-ares gives them to
/// gRPC's resolver: those of the family of the first address of the answer.
fn addresses_of(answers: &[message::Record]) -> Vec<IpAddr> {
    let found: Vec<IpAddr> = answers
        .iter()
        .filter_map(|record| match record {
            message::Record::Address(address) => Some(*address),
            _ => None,
        })
        .collect();
    let Some(first) = found.first() else {
        return found;
    };
    let v6 = first.is_ipv6();
    found
        .into_iter()
        .filter(|address| address.is_ipv6() == v6)
        .collect()
}

/// How many labels `name` has, of no root (`ares_name_label_cnt`).
fn labels(name: &str) -> usize {
    let name = name.strip_suffix('.').unwrap_or(name);
    if name.is_empty() {
        0
    } else {
        name.matches('.').count() + 1
    }
}

/// Whether `name` is `localhost` or a name under it, of any case and of no root, as c-ares's
/// `ares_is_localhost`.
fn is_localhost(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower == "localhost" || lower.ends_with(".localhost")
}

/// Whether `name` is of `.onion` (RFC 7686), which c-ares does not look up.
fn is_onion(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".onion") || lower.ends_with(".onion.")
}

/// The loopback addresses of the family of `v6` of the system, as c-ares gives those of a name of
/// the loopback that the hosts file does not have (`ares_addrinfo_localhost`): those of its
/// interfaces of the loopback, else `127.0.0.1` or `::1`.
fn loopback_addresses(v6: bool) -> Vec<IpAddr> {
    let found = interface_loopbacks(v6);
    if !found.is_empty() {
        return found;
    }
    vec![if v6 {
        IpAddr::V6(Ipv6Addr::LOCALHOST)
    } else {
        IpAddr::V4(Ipv4Addr::LOCALHOST)
    }]
}

/// The addresses of the family of `v6` of the interfaces of the loopback of the system.
#[cfg(unix)]
fn interface_loopbacks(v6: bool) -> Vec<IpAddr> {
    let mut found = Vec::new();
    let mut list: *mut libc::ifaddrs = std::ptr::null_mut();
    // SAFETY: `list` is a place for the list, which `freeifaddrs` frees once it is read.
    if unsafe { libc::getifaddrs(&mut list) } != 0 {
        return found;
    }
    let mut entry = list;
    while !entry.is_null() {
        // SAFETY: `entry` is a node of the list that `getifaddrs` gave, valid until it is freed.
        let node = unsafe { &*entry };
        entry = node.ifa_next;
        if node.ifa_addr.is_null() || node.ifa_flags & libc::IFF_LOOPBACK as libc::c_uint == 0 {
            continue;
        }
        // SAFETY: `ifa_addr` is a socket address of the family that it names.
        let family = i32::from(unsafe { (*node.ifa_addr).sa_family });
        if family == libc::AF_INET && !v6 {
            // SAFETY: an address of `AF_INET` is a `sockaddr_in`.
            let address = unsafe { &*(node.ifa_addr as *const libc::sockaddr_in) };
            found.push(IpAddr::V4(Ipv4Addr::from(u32::from_be(
                address.sin_addr.s_addr,
            ))));
        } else if family == libc::AF_INET6 && v6 {
            // SAFETY: an address of `AF_INET6` is a `sockaddr_in6`.
            let address = unsafe { &*(node.ifa_addr as *const libc::sockaddr_in6) };
            found.push(IpAddr::V6(Ipv6Addr::from(address.sin6_addr.s6_addr)));
        }
    }
    // SAFETY: `list` is the list that `getifaddrs` gave, freed once.
    unsafe { libc::freeifaddrs(list) };
    found
}

/// The addresses of the interfaces of the loopback of the system: none known here.
#[cfg(not(unix))]
fn interface_loopbacks(_v6: bool) -> Vec<IpAddr> {
    Vec::new()
}
