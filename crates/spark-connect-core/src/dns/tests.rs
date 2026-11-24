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

//! The lookups of the resolver against a DNS server of the tests, in the process, of answers built
//! by hand: the queries that grpcio's resolver sends, as the probes of grpcio 1.85.1 (c-ares
//! 1.34.5) against a server of the probes received them, and the outcomes that it gives.

use super::*;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// What the server of the tests answers to a question.
#[derive(Clone, Debug)]
pub(crate) enum Answer {
    /// `NOERROR`, of the addresses of the type of the question.
    Addresses(Vec<IpAddr>),
    /// A `CNAME` of the name, and the addresses of the type of the question of the name.
    Cname(String, Vec<IpAddr>),
    /// An answer of the response code and no record.
    Rcode(u8),
    /// No answer.
    Drop,
    /// Over UDP, a truncated answer of no record; over TCP, the answer, or a close of None.
    Truncated(Option<Box<Answer>>),
    /// The answer, after the delay.
    Delayed(Duration, Box<Answer>),
}

/// A question that the server of the tests was asked: over TCP or UDP, its name, of its case, its
/// type, and its ID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Asked {
    pub(crate) tcp: bool,
    pub(crate) name: String,
    pub(crate) qtype: &'static str,
    pub(crate) id: u16,
    pub(crate) at: Instant,
}

/// The zone of a server of the tests: the answer to a name, in lower case, and a type.
type Zone = Arc<dyn Fn(&str, u16) -> Answer + Send + Sync>;

/// A DNS server of the tests on the loopback, of UDP and, unless told not to, of TCP on the same
/// port, which answers of its zone and records what it was asked. Its tasks end when it drops.
pub(crate) struct Responder {
    pub(crate) address: SocketAddr,
    asked: Arc<Mutex<Vec<Asked>>>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
    /// Of a server of no TCP, a TCP socket of its port, bound and not listening, so that a connect
    /// to the port is refused, whatever else listens on the machine.
    #[cfg(unix)]
    _refusing: Option<socket2::Socket>,
}

impl Drop for Responder {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

impl Responder {
    /// A server of the zone `zone` on the loopback of IPv4, of TCP if `tcp`.
    pub(crate) async fn start(
        zone: impl Fn(&str, u16) -> Answer + Send + Sync + 'static,
        tcp: bool,
    ) -> Self {
        Self::start_on(IpAddr::V4(Ipv4Addr::LOCALHOST), zone, tcp).await
    }

    /// A server of the zone `zone` on `ip`, of TCP if `tcp`.
    pub(crate) async fn start_on(
        ip: IpAddr,
        zone: impl Fn(&str, u16) -> Answer + Send + Sync + 'static,
        tcp: bool,
    ) -> Self {
        let zone: Zone = Arc::new(zone);
        let asked = Arc::new(Mutex::new(Vec::new()));
        // The UDP and the TCP socket of one port: of a port that the system gave the UDP one, and
        // that TCP may not take, then another. Of no TCP, the TCP socket of the port is bound and
        // does not listen, so that a connect to it is refused, and not taken by a listener of
        // another process of the machine, which a lookup would then wait for.
        #[cfg(unix)]
        let mut refusing = None;
        let (udp, listener) = loop {
            let udp = UdpSocket::bind((ip, 0)).await.unwrap();
            let address = udp.local_addr().unwrap();
            if !tcp {
                #[cfg(unix)]
                {
                    let socket = socket2::Socket::new(
                        socket2::Domain::for_address(address),
                        socket2::Type::STREAM,
                        None,
                    )
                    .unwrap();
                    if socket.bind(&address.into()).is_err() {
                        continue;
                    }
                    refusing = Some(socket);
                }
                break (udp, None);
            }
            match tokio::net::TcpListener::bind(address).await {
                Ok(listener) => break (udp, Some(listener)),
                Err(_) => continue,
            }
        };
        let address = udp.local_addr().unwrap();
        let udp = Arc::new(udp);
        let mut tasks = vec![tokio::spawn(serve_udp(udp, zone.clone(), asked.clone()))];
        if let Some(listener) = listener {
            tasks.push(tokio::spawn(serve_tcp(listener, zone, asked.clone())));
        }
        Self {
            address,
            asked,
            tasks,
            #[cfg(unix)]
            _refusing: refusing,
        }
    }

    /// What the server was asked, in order: over TCP or UDP, the name and the type.
    pub(crate) fn asked(&self) -> Vec<(bool, String, &'static str)> {
        self.asked_at()
            .into_iter()
            .map(|asked| (asked.tcp, asked.name, asked.qtype))
            .collect()
    }

    /// What the server was asked, in order.
    pub(crate) fn asked_at(&self) -> Vec<Asked> {
        self.asked.lock().unwrap().clone()
    }
}

/// `asked` as the server of the tests gives it, of names in UDP.
pub(crate) fn udp(asked: &[(&str, &'static str)]) -> Vec<(bool, String, &'static str)> {
    asked
        .iter()
        .map(|(name, qtype)| (false, name.to_string(), *qtype))
        .collect()
}

/// The name, the end and the type of the question of the query `packet`.
fn question(packet: &[u8]) -> Option<(String, usize, u16)> {
    let mut at = 12;
    let mut labels = Vec::new();
    loop {
        let length = usize::from(*packet.get(at)?);
        if length == 0 {
            at += 1;
            break;
        }
        labels.push(String::from_utf8_lossy(packet.get(at + 1..at + 1 + length)?).into_owned());
        at += 1 + length;
    }
    let qtype = u16::from_be_bytes([*packet.get(at)?, *packet.get(at + 1)?]);
    Some((labels.join("."), at + 4, qtype))
}

/// The labels of `name`.
fn wire_name(name: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for label in name.split('.').filter(|label| !label.is_empty()) {
        out.push(label.len() as u8);
        out.extend_from_slice(label.as_bytes());
    }
    out.push(0);
    out
}

/// A record of an answer of the server of the tests: its owner, as it is written, its type, and
/// its data.
type Record = (Vec<u8>, u16, Vec<u8>);

/// The answer of `answer` to the query `packet` over TCP if `tcp`: None of no answer, and the delay
/// before it.
fn respond(packet: &[u8], answer: &Answer, tcp: bool) -> Option<(Vec<u8>, Duration)> {
    let (_, end, qtype) = question(packet)?;
    let mut delay = Duration::ZERO;
    let mut answer = answer.clone();
    if let Answer::Delayed(by, inner) = answer {
        delay = by;
        answer = *inner;
    }
    let mut truncated = false;
    if let Answer::Truncated(inner) = answer {
        match (tcp, inner) {
            (false, _) => {
                truncated = true;
                answer = Answer::Addresses(Vec::new());
            }
            (true, Some(inner)) => answer = *inner,
            (true, None) => return None,
        }
    }
    let of_type = |addresses: &[IpAddr]| -> Vec<IpAddr> {
        addresses
            .iter()
            .filter(|address| address.is_ipv6() == (qtype == message::TYPE_AAAA))
            .copied()
            .collect()
    };
    let (rcode, records): (u8, Vec<Record>) = match &answer {
        Answer::Drop => return None,
        Answer::Rcode(rcode) => (*rcode, Vec::new()),
        Answer::Addresses(addresses) => (
            0,
            of_type(addresses)
                .into_iter()
                .map(|address| (vec![0xc0, 0x0c], qtype, address_bytes(address)))
                .collect(),
        ),
        Answer::Cname(target, addresses) => {
            let mut records = vec![(vec![0xc0, 0x0c], message::TYPE_CNAME, wire_name(target))];
            records.extend(
                of_type(addresses)
                    .into_iter()
                    .map(|address| (wire_name(target), qtype, address_bytes(address))),
            );
            (0, records)
        }
        Answer::Truncated(_) | Answer::Delayed(..) => (2, Vec::new()),
    };
    let mut out = packet[..2].to_vec();
    out.push(0x80 | (packet[2] & 0x79) | if truncated { 0x02 } else { 0 });
    out.push(0x80 | rcode);
    out.extend_from_slice(&[0, 1]);
    out.extend_from_slice(&(records.len() as u16).to_be_bytes());
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&packet[12..end]);
    for (owner, rtype, data) in records {
        out.extend_from_slice(&owner);
        out.extend_from_slice(&rtype.to_be_bytes());
        out.extend_from_slice(&[0, 1, 0, 0, 0, 60]);
        out.extend_from_slice(&(data.len() as u16).to_be_bytes());
        out.extend_from_slice(&data);
    }
    Some((out, delay))
}

fn address_bytes(address: IpAddr) -> Vec<u8> {
    match address {
        IpAddr::V4(ip) => ip.octets().to_vec(),
        IpAddr::V6(ip) => ip.octets().to_vec(),
    }
}

/// Record the question of `packet` over TCP if `tcp`, and the answer of `zone` to it.
fn ask(zone: &Zone, asked: &Mutex<Vec<Asked>>, packet: &[u8], tcp: bool) -> Option<Answer> {
    let (name, _, qtype) = question(packet)?;
    asked.lock().unwrap().push(Asked {
        tcp,
        name: name.clone(),
        qtype: match qtype {
            message::TYPE_A => "A",
            message::TYPE_AAAA => "AAAA",
            _ => "other",
        },
        id: u16::from_be_bytes([packet[0], packet[1]]),
        at: Instant::now(),
    });
    Some(zone(&name.to_ascii_lowercase(), qtype))
}

async fn serve_udp(socket: Arc<UdpSocket>, zone: Zone, asked: Arc<Mutex<Vec<Asked>>>) {
    let mut buffer = vec![0u8; 65536];
    while let Ok((length, peer)) = socket.recv_from(&mut buffer).await {
        let packet = buffer[..length].to_vec();
        let Some(answer) = ask(&zone, &asked, &packet, false) else {
            continue;
        };
        let Some((response, delay)) = respond(&packet, &answer, false) else {
            continue;
        };
        let socket = socket.clone();
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            let _ = socket.send_to(&response, peer).await;
        });
    }
}

async fn serve_tcp(listener: tokio::net::TcpListener, zone: Zone, asked: Arc<Mutex<Vec<Asked>>>) {
    while let Ok((mut stream, _)) = listener.accept().await {
        let (zone, asked) = (zone.clone(), asked.clone());
        tokio::spawn(async move {
            loop {
                let mut length = [0u8; 2];
                if stream.read_exact(&mut length).await.is_err() {
                    return;
                }
                let mut packet = vec![0u8; usize::from(u16::from_be_bytes(length))];
                if stream.read_exact(&mut packet).await.is_err() {
                    return;
                }
                let Some(answer) = ask(&zone, &asked, &packet, true) else {
                    return;
                };
                let Some((response, delay)) = respond(&packet, &answer, true) else {
                    return;
                };
                tokio::time::sleep(delay).await;
                let mut framed = (response.len() as u16).to_be_bytes().to_vec();
                framed.extend_from_slice(&response);
                if stream.write_all(&framed).await.is_err() {
                    return;
                }
            }
        });
    }
}

fn ip(text: &str) -> IpAddr {
    text.parse().unwrap()
}

fn addresses(texts: &[&str]) -> Answer {
    Answer::Addresses(texts.iter().map(|text| ip(text)).collect())
}

/// The configuration of the tests: of the search domains `domains` and no hosts, of timeouts of
/// 200 ms and a poll of 50 ms.
fn config(domains: &[&str]) -> Config {
    Config {
        domains: domains.iter().map(|domain| domain.to_string()).collect(),
        timeout: Duration::from_millis(200),
        poll: Duration::from_millis(50),
        ..Config::default()
    }
}

fn sockets(port: u16, texts: &[&str]) -> Vec<SocketAddr> {
    texts
        .iter()
        .map(|text| SocketAddr::new(ip(text), port))
        .collect()
}

#[tokio::test]
async fn each_family_is_asked_and_the_addresses_are_tried_as_grpcio_tries_them() {
    // Of the probes: A of [.3, .2] and AAAA of [::ffff:.4, ::1] are tried ::1, .3, ::ffff:.4, .2
    // (RFC 6724, then the families in turn).
    let server = Responder::start(
        |_, _| addresses(&["127.0.0.3", "127.0.0.2", "::ffff:127.0.0.4", "::1"]),
        true,
    )
    .await;
    let found = lookup(server.address, "order.test", 7, &config(&[]), true).await;
    assert_eq!(
        found.unwrap(),
        sockets(7, &["::1", "127.0.0.3", "::ffff:127.0.0.4", "127.0.0.2"])
    );
    assert_eq!(
        server.asked(),
        udp(&[("order.test", "A"), ("order.test", "AAAA")])
    );
    // Of IPv4 alone, of no IPv6 loopback, the order of the answer, which RFC 6724 keeps.
    let found = lookup(server.address, "order.test", 7, &config(&[]), false).await;
    assert_eq!(found.unwrap(), sockets(7, &["127.0.0.3", "127.0.0.2"]));
}

#[tokio::test]
async fn the_names_keep_their_case_and_the_queries_their_id() {
    let server = Responder::start(|_, _| addresses(&["127.0.0.1"]), true).await;
    let found = lookup(server.address, "Mixed.Case.TEST", 7, &config(&[]), true).await;
    assert_eq!(found.unwrap(), sockets(7, &["127.0.0.1"]));
    assert_eq!(
        server.asked(),
        udp(&[("Mixed.Case.TEST", "A"), ("Mixed.Case.TEST", "AAAA")])
    );
}

#[tokio::test]
async fn each_family_asks_the_search_domains_on_its_own() {
    // Of the probes: the A of `a4` is that of `a4.probe.test`, and its AAAA, of no data there, is
    // asked of the other domain and of the name as it is.
    let zone = |name: &str, _: u16| match name {
        "a4.probe.test" => addresses(&["127.0.0.1"]),
        "two.other.test" => addresses(&["127.0.0.1"]),
        _ => Answer::Rcode(message::RCODE_NXDOMAIN),
    };
    let server = Responder::start(zone, true).await;
    let config = config(&["probe.test", "other.test"]);
    let found = lookup(server.address, "a4", 7, &config, true).await;
    assert_eq!(found.unwrap(), sockets(7, &["127.0.0.1"]));
    assert_eq!(
        server.asked(),
        udp(&[
            ("a4.probe.test", "A"),
            ("a4.probe.test", "AAAA"),
            ("a4.other.test", "AAAA"),
            ("a4", "AAAA"),
        ])
    );
    // Of a dot, as many as `ndots`, the name is asked first.
    let server = Responder::start(zone, true).await;
    let found = lookup(server.address, "two.sub", 7, &config, true).await;
    assert_eq!(found.unwrap_err(), Status::NotFound);
    assert_eq!(
        server.asked(),
        udp(&[
            ("two.sub", "A"),
            ("two.sub", "AAAA"),
            ("two.sub.probe.test", "A"),
            ("two.sub.probe.test", "AAAA"),
            ("two.sub.other.test", "A"),
            ("two.sub.other.test", "AAAA"),
        ])
    );
    // Of fewer dots than `ndots`, last; of a last `.`, alone.
    let server = Responder::start(zone, true).await;
    let ndots = Config {
        ndots: 2,
        ..config.clone()
    };
    let _ = lookup(server.address, "two.sub", 7, &ndots, false).await;
    let _ = lookup(server.address, "a4.probe.test.", 7, &ndots, false).await;
    assert_eq!(
        server.asked(),
        udp(&[
            ("two.sub.probe.test", "A"),
            ("two.sub.other.test", "A"),
            ("two.sub", "A"),
            ("a4.probe.test", "A"),
        ])
    );
}

#[tokio::test]
async fn an_answer_of_no_data_is_told_from_a_name_that_is_not_found() {
    // Of the probes: NOERROR of no record, of the name and of NXDOMAIN of the search domain, is
    // `DNS server returned answer with no data`; NXDOMAIN of both is `Domain name not found`; a
    // CNAME of no address is no data too.
    let zone = |name: &str, _: u16| match name {
        "nodata.test" => Answer::Addresses(Vec::new()),
        "cname.test" => Answer::Cname("nowhere.test".to_string(), Vec::new()),
        "chain.test" => Answer::Cname("target.test".to_string(), vec![ip("127.0.0.9")]),
        _ => Answer::Rcode(message::RCODE_NXDOMAIN),
    };
    let server = Responder::start(zone, true).await;
    let config = config(&["search.test"]);
    for (name, outcome) in [
        ("nodata.test", Err(Status::NoData)),
        ("nx.test", Err(Status::NotFound)),
        ("cname.test", Err(Status::NoData)),
        ("chain.test", Ok(sockets(7, &["127.0.0.9"]))),
    ] {
        assert_eq!(
            lookup(server.address, name, 7, &config, true).await,
            outcome,
            "{name}"
        );
    }
}

#[tokio::test]
async fn a_failure_of_the_server_is_retried_and_ends_the_search() {
    // Of the probes: SERVFAIL, REFUSED and NOTIMP are sent 3 times, and the search ends; FORMERR
    // ends at once.
    let zone = |name: &str, _: u16| match name.split('.').next().unwrap() {
        "servfail" => Answer::Rcode(message::RCODE_SERVFAIL),
        "refused" => Answer::Rcode(message::RCODE_REFUSED),
        "notimp" => Answer::Rcode(message::RCODE_NOTIMP),
        "formerr" => Answer::Rcode(message::RCODE_FORMERR),
        _ => Answer::Rcode(message::RCODE_NXDOMAIN),
    };
    let config = config(&["search.test"]);
    for (name, status, tries) in [
        ("servfail.test", Status::ServFail, 3),
        ("refused.test", Status::Refused, 3),
        ("notimp.test", Status::NotImp, 3),
        ("formerr.test", Status::FormErr, 1),
    ] {
        let server = Responder::start(zone, true).await;
        let found = lookup(server.address, name, 7, &config, false).await;
        assert_eq!(found, Err(status), "{name}");
        assert_eq!(server.asked(), udp(&vec![(name, "A"); tries]), "{name}");
        // Retries keep the ID of the query.
        let ids: std::collections::HashSet<u16> =
            server.asked_at().iter().map(|asked| asked.id).collect();
        assert_eq!(ids.len(), 1);
    }
    // A name of one label asks on of SERVFAIL and REFUSED (c-ares's issue 852), as with `ndots:0`,
    // of the probes, `sf1` and then `sf1.probe.test`.
    let zone = |name: &str, _: u16| match name {
        "sf1" => Answer::Rcode(message::RCODE_SERVFAIL),
        "sf1.probe.test" => addresses(&["127.0.0.1"]),
        _ => Answer::Rcode(message::RCODE_NXDOMAIN),
    };
    let server = Responder::start(zone, true).await;
    let ndots0 = Config {
        ndots: 0,
        ..crate::dns::tests::config(&["probe.test"])
    };
    let found = lookup(server.address, "sf1", 7, &ndots0, false).await;
    assert_eq!(found.unwrap(), sockets(7, &["127.0.0.1"]));
    assert_eq!(
        server.asked(),
        udp(&[
            ("sf1", "A"),
            ("sf1", "A"),
            ("sf1", "A"),
            ("sf1.probe.test", "A")
        ])
    );
    // Of `attempts:1`, once.
    let server = Responder::start(zone, true).await;
    let once = Config {
        tries: 1,
        domains: Vec::new(),
        ..ndots0
    };
    assert_eq!(
        lookup(server.address, "sf1", 7, &once, false).await,
        Err(Status::ServFail)
    );
    assert_eq!(server.asked(), udp(&[("sf1", "A")]));
}

#[tokio::test]
async fn the_failure_is_that_of_the_family_that_ended_last() {
    // Of the probes: A of SERVFAIL and AAAA of NXDOMAIN 0.5 s later is `Domain name not found`;
    // the other way, `DNS server returned general failure`.
    let delayed =
        |rcode: u8| Answer::Delayed(Duration::from_millis(100), Box::new(Answer::Rcode(rcode)));
    let config = config(&[]);
    for (a, aaaa, status) in [
        (
            Answer::Rcode(message::RCODE_SERVFAIL),
            delayed(message::RCODE_NXDOMAIN),
            Status::NotFound,
        ),
        (
            delayed(message::RCODE_SERVFAIL),
            Answer::Rcode(message::RCODE_NXDOMAIN),
            Status::ServFail,
        ),
        (
            delayed(message::RCODE_NXDOMAIN),
            Answer::Addresses(Vec::new()),
            Status::NotFound,
        ),
    ] {
        let (a, aaaa) = (a.clone(), aaaa.clone());
        let server = Responder::start(
            move |_, qtype| {
                if qtype == message::TYPE_A {
                    a.clone()
                } else {
                    aaaa.clone()
                }
            },
            true,
        )
        .await;
        assert_eq!(
            lookup(server.address, "x.test", 7, &config, true).await,
            Err(status)
        );
    }
}

#[tokio::test]
async fn an_address_of_one_family_is_kept_when_the_other_fails() {
    let server = Responder::start(
        |_, qtype| {
            if qtype == message::TYPE_A {
                addresses(&["127.0.0.1"])
            } else {
                Answer::Rcode(message::RCODE_SERVFAIL)
            }
        },
        true,
    )
    .await;
    let found = lookup(server.address, "x.test", 7, &config(&[]), true).await;
    assert_eq!(found.unwrap(), sockets(7, &["127.0.0.1"]));
    assert_eq!(
        server.asked(),
        udp(&[
            ("x.test", "A"),
            ("x.test", "AAAA"),
            ("x.test", "AAAA"),
            ("x.test", "AAAA")
        ])
    );
}

#[tokio::test]
async fn a_truncated_answer_is_asked_again_over_tcp() {
    let server = Responder::start(
        |_, _| Answer::Truncated(Some(Box::new(addresses(&["127.0.0.1", "::1"])))),
        true,
    )
    .await;
    let found = lookup(server.address, "tc.test", 7, &config(&[]), true).await;
    assert_eq!(found.unwrap(), sockets(7, &["::1", "127.0.0.1"]));
    let asked = server.asked();
    assert_eq!(asked.len(), 4);
    assert_eq!(asked[..2], udp(&[("tc.test", "A"), ("tc.test", "AAAA")]));
    let mut over_tcp: Vec<_> = asked[2..].to_vec();
    over_tcp.sort();
    assert_eq!(
        over_tcp,
        vec![
            (true, "tc.test".to_string(), "A"),
            (true, "tc.test".to_string(), "AAAA")
        ]
    );
    // Over the same ID.
    let at = server.asked_at();
    assert!(at[2..]
        .iter()
        .all(|tcp| at[..2].iter().any(|udp| udp.id == tcp.id)));
}

#[tokio::test]
async fn a_tcp_connection_that_the_server_closes_is_retried_and_one_that_fails_cancels() {
    // Of the probes: a close of each TCP query sends it 3 times, then `Could not contact DNS
    // servers`; of no listener of TCP, the lookup is cancelled at once (`DNS query cancelled`).
    let server = Responder::start(|_, _| Answer::Truncated(None), true).await;
    let found = lookup(server.address, "tc.test", 7, &config(&[]), false).await;
    assert_eq!(found, Err(Status::ConnRefused));
    assert_eq!(
        server.asked(),
        vec![
            (false, "tc.test".to_string(), "A"),
            (true, "tc.test".to_string(), "A"),
            (true, "tc.test".to_string(), "A"),
            (true, "tc.test".to_string(), "A"),
        ]
    );
    let server = Responder::start(|_, _| Answer::Truncated(None), false).await;
    let found = lookup(server.address, "tc.test", 7, &config(&[]), true).await;
    assert_eq!(found, Err(Status::Cancelled));
}

#[tokio::test]
async fn a_server_that_does_not_answer_times_out_after_its_attempts() {
    // c-ares's timeouts of 3 attempts: the timeout of the system, then twice it but a random part
    // of a half at most, each, as gRPC processes them (here of a poll of 50 ms).
    let server = Responder::start(|_, _| Answer::Drop, true).await;
    let config = config(&[]);
    let started = Instant::now();
    let found = lookup(server.address, "drop.test", 7, &config, false).await;
    let elapsed = started.elapsed();
    assert_eq!(found, Err(Status::Timeout));
    let at: Vec<Duration> = server
        .asked_at()
        .iter()
        .map(|asked| asked.at - started)
        .collect();
    assert_eq!(at.len(), 3);
    let base = config.timeout;
    let slack = config.poll + Duration::from_millis(100);
    assert!(at[1] >= base && at[1] <= base + slack, "{at:?}");
    let second = at[2] - at[1];
    assert!(second >= base && second <= base * 2 + slack, "{at:?}");
    let last = elapsed - at[2];
    assert!(
        last >= base * 2 && last <= base * 4 + slack,
        "{at:?} {elapsed:?}"
    );
}

#[tokio::test]
async fn a_late_answer_to_an_attempt_that_timed_out_is_taken() {
    // Of the probes: an answer 2.5 s after the first attempt, after its retry at 2 s, ends it.
    let server = Responder::start(
        |_, _| {
            Answer::Delayed(
                Duration::from_millis(300),
                Box::new(addresses(&["127.0.0.1"])),
            )
        },
        true,
    )
    .await;
    let started = Instant::now();
    let found = lookup(server.address, "slow.test", 7, &config(&[]), false).await;
    assert_eq!(found.unwrap(), sockets(7, &["127.0.0.1"]));
    assert!(started.elapsed() < Duration::from_millis(400));
    assert_eq!(server.asked().len(), 2);
}

#[tokio::test]
async fn a_server_of_no_listener_cannot_be_contacted() {
    // The port of a socket that is bound and never read: ICMP refuses its queries at once.
    let closed = std::net::UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let address = closed.local_addr().unwrap();
    drop(closed);
    let found = lookup(address, "x.test", 7, &config(&[]), true).await;
    assert_eq!(found, Err(Status::ConnRefused));
}

#[tokio::test]
async fn the_resolution_of_grpc_cancels_the_lookup_after_its_timeout() {
    // Of the probes: of `attempts:7`, gRPC's timeout of 120 s cancels the lookup.
    let server = Responder::start(|_, _| Answer::Drop, true).await;
    let config = Config {
        tries: 7,
        query_timeout: Duration::from_millis(500),
        ..config(&[])
    };
    let found = lookup(server.address, "drop.test", 7, &config, true).await;
    assert_eq!(found, Err(Status::Cancelled));
}

#[tokio::test]
async fn an_ipv6_server_is_asked() {
    if std::net::UdpSocket::bind((Ipv6Addr::LOCALHOST, 0)).is_err() {
        return;
    }
    let server = Responder::start_on(
        IpAddr::V6(Ipv6Addr::LOCALHOST),
        |_, _| addresses(&["127.0.0.1", "::1"]),
        true,
    )
    .await;
    let found = lookup(server.address, "x.test", 7, &config(&[]), true).await;
    assert_eq!(found.unwrap(), sockets(7, &["::1", "127.0.0.1"]));
    assert_eq!(server.asked().len(), 2);
}

#[tokio::test]
async fn the_hosts_file_and_the_loopback_are_looked_up_before_dns() {
    // Of the probes: `ip6-loopback` of `::1` in the hosts file asks DNS for its A alone;
    // `localhost` and the names under it are not asked, but `localhost.`, which is not
    // `localhost`; a name of `.onion` is not found.
    let server = Responder::start(|_, _| addresses(&["127.0.0.5"]), true).await;
    let config = Config {
        hosts: config::Hosts::parse(
            "127.0.0.1 localhost\n::1 ip6-localhost ip6-loopback\n127.0.0.9 hostsonly.test\n",
        ),
        ..config(&[])
    };
    let found = lookup(server.address, "ip6-loopback", 7, &config, true).await;
    assert_eq!(found.unwrap(), sockets(7, &["::1", "127.0.0.5"]));
    assert_eq!(server.asked(), udp(&[("ip6-loopback", "A")]));
    for name in ["localhost", "LOCALHOST", "x.localhost"] {
        let found = lookup(server.address, name, 7, &config, true).await;
        assert_eq!(found.unwrap(), sockets(7, &["::1", "127.0.0.1"]), "{name}");
    }
    let found = lookup(server.address, "HostsOnly.test", 7, &config, false).await;
    assert_eq!(found.unwrap(), sockets(7, &["127.0.0.9"]));
    assert_eq!(server.asked().len(), 1);
    let _ = lookup(server.address, "localhost.", 7, &config, false).await;
    assert_eq!(server.asked()[1..], udp(&[("localhost", "A")])[..]);
    assert_eq!(
        lookup(server.address, "x.onion", 7, &config, true).await,
        Err(Status::NotFound)
    );
    // Of DNS before the hosts file, DNS first.
    let dns_first = Config {
        lookups: vec![Source::Dns, Source::Files],
        ..config.clone()
    };
    let found = lookup(server.address, "hostsonly.test", 7, &dns_first, false).await;
    assert_eq!(found.unwrap(), sockets(7, &["127.0.0.5"]));
}

#[tokio::test]
async fn a_name_that_c_ares_does_not_write_is_misformatted() {
    let server = Responder::start(|_, _| addresses(&["127.0.0.1"]), true).await;
    let long = format!("{}.test", "a".repeat(64));
    for name in ["local!host", "a..b", long.as_str()] {
        assert_eq!(
            lookup(server.address, name, 7, &config(&[]), true).await,
            Err(Status::BadName),
            "{name}"
        );
    }
    assert!(server.asked().is_empty());
}

#[tokio::test]
async fn an_alias_of_hostaliases_is_asked_alone() {
    let server = Responder::start(
        |name: &str, _| match name {
            "a4.probe.test" => addresses(&["127.0.0.1"]),
            _ => Answer::Rcode(message::RCODE_NXDOMAIN),
        },
        true,
    )
    .await;
    let mut config = config(&["search.test"]);
    config
        .aliases
        .insert("myalias".to_string(), "a4.probe.test".to_string());
    let found = lookup(server.address, "MyAlias", 7, &config, true).await;
    assert_eq!(found.unwrap(), sockets(7, &["127.0.0.1"]));
    assert_eq!(
        server.asked(),
        udp(&[("a4.probe.test", "A"), ("a4.probe.test", "AAAA")])
    );
}

#[test]
fn the_configuration_is_read_as_c_ares_reads_it() {
    let files = |resolv: &'static str, nsswitch: &'static str| {
        move |name: &str| match name {
            "resolv.conf" => Some(resolv.to_string()),
            "nsswitch.conf" => Some(nsswitch.to_string()),
            "hosts" => Some("127.0.0.1 localhost\n".to_string()),
            _ => None,
        }
    };
    let no_env = |_: &str| None;
    let hosts_first = "hosts: files mdns4_minimal [NOTFOUND=return] dns\n";
    // Of the probes: `search` over `domain`, of either order; `options` of seconds, 5 s at most.
    let config = Config::of(
        &files(
            "nameserver 10.0.0.1\nsearch other.test\ndomain probe.test\noptions ndots:2 timeout:1 attempts:2\n",
            hosts_first,
        ),
        &no_env,
        Some("host"),
    );
    assert_eq!(config.domains, vec!["other.test".to_string()]);
    assert_eq!(config.ndots, 2);
    assert_eq!(config.timeout, Duration::from_secs(1));
    assert_eq!(config.tries, 2);
    assert_eq!(config.lookups, vec![Source::Files, Source::Dns]);
    let config = Config::of(
        &files(
            "domain probe.test\nsearch a.test b.test\noptions timeout:30\n",
            "hosts: dns files\n",
        ),
        &no_env,
        None,
    );
    assert_eq!(
        config.domains,
        vec!["a.test".to_string(), "b.test".to_string()]
    );
    assert_eq!(config.timeout, Duration::from_secs(5));
    assert_eq!(config.tries, 3);
    assert_eq!(config.lookups, vec![Source::Dns, Source::Files]);
    // The environment over the files: the first domain of `LOCALDOMAIN`, and `RES_OPTIONS`.
    let env = |name: &str| match name {
        "LOCALDOMAIN" => Some("probe.test other.test".to_string()),
        "RES_OPTIONS" => Some("ndots:3 attempts:1".to_string()),
        _ => None,
    };
    let config = Config::of(&files("search x.test\n", hosts_first), &env, None);
    assert_eq!(config.domains, vec!["probe.test".to_string()]);
    assert_eq!((config.ndots, config.tries), (3, 1));
    // Of no search domain, that of the host name.
    let config = Config::of(&files("", hosts_first), &no_env, Some("node.example.test"));
    assert_eq!(config.domains, vec!["example.test".to_string()]);
    let empty = |name: &str| (name == "LOCALDOMAIN").then(String::new);
    let config = Config::of(&files("search x.test\n", hosts_first), &empty, Some("node"));
    assert!(config.domains.is_empty());
    // Of no file, c-ares's defaults.
    let config = Config::of(&|_| None, &no_env, None);
    assert_eq!(config, Config::default());
}

#[test]
fn the_hosts_file_is_merged_as_c_ares_merges_it() {
    // Of the probes: a line of the address of an entry joins it, else that of a name of one, and a
    // name is of its first entry, so `multi` of `127.0.0.1` and of `::1` is of the entry of
    // `127.0.0.1` alone, whose `::1` is `ip6-localhost`'s.
    let hosts = config::Hosts::parse(
        "127.0.0.1 localhost\n::1 ip6-localhost ip6-loopback\n127.0.0.1 multi.test\n\
         ::1 multi.test\n127.0.0.4 dup.test\n127.0.0.3 dup.test # a comment\n\
         # 127.0.0.2 commented.test\n127.0.0.1 Mixed.Test alias.test\n",
    );
    let v4 = |name: &str| hosts.lookup(name, false);
    let v6 = |name: &str| hosts.lookup(name, true);
    assert_eq!(v4("multi.test"), Some(vec![ip("127.0.0.1")]));
    assert_eq!(v6("multi.test"), Some(Vec::new()));
    assert_eq!(v4("DUP.test"), Some(vec![ip("127.0.0.4"), ip("127.0.0.3")]));
    assert_eq!(v4("mixed.test"), Some(vec![ip("127.0.0.1")]));
    assert_eq!(v4("alias.test"), Some(vec![ip("127.0.0.1")]));
    assert_eq!(v6("ip6-loopback"), Some(vec![ip("::1")]));
    assert_eq!(v4("commented.test"), None);
    assert_eq!(v4("multi.test."), None);
}

#[test]
fn the_messages_are_written_and_read_as_c_ares_does() {
    let query = message::query(0x1234, "Ab.test.", message::TYPE_AAAA).unwrap();
    assert_eq!(
        query,
        [
            &[0x12, 0x34, 0x01, 0x00, 0, 1, 0, 0, 0, 0, 0, 0][..],
            &[2, b'A', b'b', 4, b't', b'e', b's', b't', 0, 0, 28, 0, 1]
        ]
        .concat()
    );
    assert_eq!(message::query(1, "a b", message::TYPE_A), None);
    assert!(message::query(1, &"a.".repeat(127), message::TYPE_A).is_some());
    assert_eq!(message::query(1, &"a.".repeat(128), message::TYPE_A), None);
    // An answer of a compressed CNAME, its A, and a record of another type.
    let mut answer = vec![0x12, 0x34, 0x81, 0x80, 0, 1, 0, 3, 0, 0, 0, 0];
    answer.extend_from_slice(&[1, b'x', 4, b't', b'e', b's', b't', 0, 0, 1, 0, 1]);
    answer.extend_from_slice(&[0xc0, 12, 0, 5, 0, 1, 0, 0, 0, 60, 0, 4, 1, b'y', 0xc0, 14]);
    answer.extend_from_slice(&[0xc0, 28, 0, 1, 0, 1, 0, 0, 0, 60, 0, 4, 127, 0, 0, 7]);
    answer.extend_from_slice(&[0xc0, 12, 0, 16, 0, 1, 0, 0, 0, 60, 0, 1, 0]);
    let response = message::parse(&answer).unwrap();
    assert_eq!(response.id, 0x1234);
    assert_eq!(response.questions, vec![("x.test".to_string(), 1, 1)]);
    assert_eq!(
        response.answers,
        vec![
            message::Record::Cname,
            message::Record::Address(ip("127.0.0.7")),
            message::Record::Other
        ]
    );
    assert_eq!(addresses_of(&response.answers), vec![ip("127.0.0.7")]);
    // Not a response, a pointer out of it, a record past its end.
    let mut query_flags = answer.clone();
    query_flags[2] = 0x01;
    assert_eq!(message::parse(&query_flags), None);
    let mut pointer = answer.clone();
    pointer[39] = 0xff;
    assert_eq!(message::parse(&pointer), None);
    assert_eq!(message::parse(&answer[..answer.len() - 1]), None);
}

#[test]
fn the_addresses_are_sorted_as_grpc_sorts_them() {
    let source = |address: SocketAddr| -> Option<SocketAddr> {
        match address.ip() {
            IpAddr::V4(ip) if ip.is_loopback() => Some("127.0.0.1:1".parse().unwrap()),
            IpAddr::V6(ip) if ip.is_loopback() => Some("[::1]:1".parse().unwrap()),
            IpAddr::V6(ip) if ip.to_ipv4_mapped().is_some() => {
                Some("[::ffff:127.0.0.1]:1".parse().unwrap())
            }
            _ => None,
        }
    };
    let order = |texts: &[&str]| -> Vec<SocketAddr> { sort::order(sockets(1, texts), &source) };
    // Of the probes: a destination of no source last; ::1 first, then the IPv4 ones of their
    // order, the families in turn from the first.
    assert_eq!(
        order(&["fc00::1", "127.0.0.3"]),
        sockets(1, &["127.0.0.3", "fc00::1"])
    );
    assert_eq!(
        order(&["127.0.0.3", "127.0.0.2", "::1"]),
        sockets(1, &["::1", "127.0.0.3", "127.0.0.2"])
    );
    assert_eq!(
        order(&["127.0.0.4", "127.0.0.2", "127.0.0.3"]),
        sockets(1, &["127.0.0.4", "127.0.0.2", "127.0.0.3"])
    );
    assert_eq!(
        order(&[
            "127.0.0.3",
            "127.0.0.2",
            "::ffff:127.0.0.4",
            "::ffff:127.0.0.5"
        ]),
        sockets(
            1,
            &[
                "127.0.0.3",
                "::ffff:127.0.0.4",
                "127.0.0.2",
                "::ffff:127.0.0.5"
            ]
        )
    );
    assert_eq!(
        order(&["2001:db8::1", "::1"]),
        sockets(1, &["::1", "2001:db8::1"])
    );
}
