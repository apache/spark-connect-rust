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

//! The messages of DNS (RFC 1035, section 4) that the resolver sends and reads: a query of one
//! question, as c-ares writes it for grpcio (recursion desired, no EDNS), and the parts of an
//! answer that c-ares reads of it.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// The types of the records that the resolver asks for and reads.
pub(super) const TYPE_A: u16 = 1;
pub(super) const TYPE_CNAME: u16 = 5;
pub(super) const TYPE_AAAA: u16 = 28;

/// The class of the Internet.
const CLASS_IN: u16 = 1;

/// The length of the header of a message.
const HEADER_LEN: usize = 12;

/// The longest name of DNS, of its labels and their lengths and the root (RFC 1035, 2.3.4).
const MAX_NAME_LEN: usize = 255;

/// The response codes that the resolver tells apart (RFC 1035, 4.1.1).
pub(super) const RCODE_NOERROR: u8 = 0;
pub(super) const RCODE_FORMERR: u8 = 1;
pub(super) const RCODE_SERVFAIL: u8 = 2;
pub(super) const RCODE_NXDOMAIN: u8 = 3;
pub(super) const RCODE_NOTIMP: u8 = 4;
pub(super) const RCODE_REFUSED: u8 = 5;

/// The query of `name` for the records of `qtype`, of the ID `id`, as c-ares writes it: of
/// recursion desired, no EDNS, and the name as it is given, but its root (a last `.`). None of a
/// name that c-ares does not write (`ARES_EBADNAME`): of a character other than `[A-Za-z0-9-._/*]`,
/// of an empty label or of one of more than 63 bytes, or longer than DNS takes.
pub(super) fn query(id: u16, name: &str, qtype: u16) -> Option<Vec<u8>> {
    let mut packet = Vec::with_capacity(HEADER_LEN + name.len() + 6);
    packet.extend_from_slice(&id.to_be_bytes());
    // QR 0, opcode QUERY, RD.
    packet.extend_from_slice(&0x0100u16.to_be_bytes());
    // One question, no other record.
    packet.extend_from_slice(&[0, 1, 0, 0, 0, 0, 0, 0]);
    packet.extend_from_slice(&encode_name(name)?);
    packet.extend_from_slice(&qtype.to_be_bytes());
    packet.extend_from_slice(&CLASS_IN.to_be_bytes());
    Some(packet)
}

/// Whether `ch` is of the characters that c-ares takes in a name (`ares_is_hostnamech`).
pub(super) fn is_name_char(ch: u8) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, b'-' | b'.' | b'_' | b'/' | b'*')
}

/// The name `name` as a sequence of labels, or None of one that c-ares does not write (see
/// [`query`]).
fn encode_name(name: &str) -> Option<Vec<u8>> {
    if !name.bytes().all(is_name_char) {
        return None;
    }
    let labels = name.strip_suffix('.').unwrap_or(name);
    let mut out = Vec::with_capacity(labels.len() + 2);
    if !labels.is_empty() {
        for label in labels.split('.') {
            if label.is_empty() || label.len() > 63 {
                return None;
            }
            out.push(label.len() as u8);
            out.extend_from_slice(label.as_bytes());
        }
    }
    out.push(0);
    (out.len() <= MAX_NAME_LEN).then_some(out)
}

/// What the resolver reads of a response: its ID, whether it is truncated, its response code, its
/// question, and the records of its answer section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Response {
    pub(super) id: u16,
    pub(super) truncated: bool,
    pub(super) rcode: u8,
    /// The questions: the name, of the case of the response, and the type and the class.
    pub(super) questions: Vec<(String, u16, u16)>,
    /// The records of the answer section, of their order.
    pub(super) answers: Vec<Record>,
}

/// A record of an answer section, as c-ares reads it: of the class of the Internet, an address of
/// a record of `A` or of `AAAA`, a `CNAME`, or another one, which it skips.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Record {
    Address(IpAddr),
    Cname,
    Other,
}

/// The response of `packet`, or None of one that is not a response that DNS reads (of its
/// header, its questions and the records of its sections), which c-ares drops.
pub(super) fn parse(packet: &[u8]) -> Option<Response> {
    if packet.len() < HEADER_LEN {
        return None;
    }
    let word = |at: usize| u16::from_be_bytes([packet[at], packet[at + 1]]);
    let id = word(0);
    let flags = word(2);
    if flags & 0x8000 == 0 {
        return None;
    }
    let counts = [word(4), word(6), word(8), word(10)];
    let mut at = HEADER_LEN;
    let mut questions = Vec::with_capacity(usize::from(counts[0]));
    for _ in 0..counts[0] {
        let (name, next) = read_name(packet, at)?;
        let fields = packet.get(next..next + 4)?;
        questions.push((
            name,
            u16::from_be_bytes([fields[0], fields[1]]),
            u16::from_be_bytes([fields[2], fields[3]]),
        ));
        at = next + 4;
    }
    let mut answers = Vec::with_capacity(usize::from(counts[1]));
    for (section, &count) in counts.iter().enumerate().skip(1) {
        for _ in 0..count {
            let (_, next) = read_name(packet, at)?;
            let fields = packet.get(next..next + 10)?;
            let rtype = u16::from_be_bytes([fields[0], fields[1]]);
            let class = u16::from_be_bytes([fields[2], fields[3]]);
            let length = usize::from(u16::from_be_bytes([fields[8], fields[9]]));
            let data = packet.get(next + 10..next + 10 + length)?;
            at = next + 10 + length;
            if section != 1 {
                continue;
            }
            let record = match (rtype, class, data.len()) {
                (TYPE_A, CLASS_IN, 4) => Record::Address(IpAddr::V4(Ipv4Addr::new(
                    data[0], data[1], data[2], data[3],
                ))),
                (TYPE_AAAA, CLASS_IN, 16) => {
                    let mut octets = [0u8; 16];
                    octets.copy_from_slice(data);
                    Record::Address(IpAddr::V6(Ipv6Addr::from(octets)))
                }
                (TYPE_A | TYPE_AAAA, CLASS_IN, _) => return None,
                (TYPE_CNAME, CLASS_IN, _) => {
                    read_name(packet, next + 10)?;
                    Record::Cname
                }
                _ => Record::Other,
            };
            answers.push(record);
        }
    }
    Some(Response {
        id,
        truncated: flags & 0x0200 != 0,
        rcode: (flags & 0x000f) as u8,
        questions,
        answers,
    })
}

/// The name at `at` of `packet`, of its compression (RFC 1035, 4.1.4), with the offset after it,
/// or None of one that is not valid: of a pointer out of the packet or of a loop, or longer than
/// DNS takes. Its labels are joined by `.`, of their bytes as characters.
fn read_name(packet: &[u8], mut at: usize) -> Option<(String, usize)> {
    let mut name = String::new();
    let mut end = None;
    let mut length = 1;
    for _ in 0..128 {
        let byte = *packet.get(at)?;
        match byte & 0xc0 {
            0xc0 => {
                let pointer = usize::from(u16::from_be_bytes([byte & 0x3f, *packet.get(at + 1)?]));
                end.get_or_insert(at + 2);
                at = pointer;
            }
            0x00 if byte == 0 => return Some((name, end.unwrap_or(at + 1))),
            0x00 => {
                let label = packet.get(at + 1..at + 1 + usize::from(byte))?;
                length += label.len() + 1;
                if length > MAX_NAME_LEN {
                    return None;
                }
                if !name.is_empty() {
                    name.push('.');
                }
                name.extend(label.iter().map(|&b| char::from(b)));
                at += 1 + usize::from(byte);
            }
            _ => return None,
        }
    }
    None
}
