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

//! The order in which a channel of grpcio tries the addresses of a name: sorted as gRPC's
//! `address_sorting` sorts them (RFC 6724, section 6, of the source address that the system would
//! take for each), then interleaved by family as its `pick_first` does (RFC 8305, section 4).

use std::net::{IpAddr, Ipv6Addr, SocketAddr};

/// The scopes of RFC 6724 that gRPC's sorting tells apart.
const SCOPE_LINK_LOCAL: u8 = 0x2;
const SCOPE_SITE_LOCAL: u8 = 0x5;
const SCOPE_GLOBAL: u8 = 0xe;

/// `addresses` in the order in which grpcio's channel tries them (see the module), of the source
/// address of each that `source` gives, or None of one that the system cannot reach.
pub(super) fn order(
    addresses: Vec<SocketAddr>,
    source: &dyn Fn(SocketAddr) -> Option<SocketAddr>,
) -> Vec<SocketAddr> {
    let mut sortable: Vec<(SocketAddr, Option<SocketAddr>, usize)> = addresses
        .into_iter()
        .enumerate()
        .map(|(index, address)| (address, source(address), index))
        .collect();
    sortable.sort_by(compare);
    interleave(
        sortable
            .into_iter()
            .map(|(address, _, _)| address)
            .collect(),
    )
}

/// The addresses of `addresses` of each family in turn, from the family of the first one, as
/// gRPC's `pick_first` flattens them (gRFC A61).
fn interleave(addresses: Vec<SocketAddr>) -> Vec<SocketAddr> {
    let Some(first) = addresses.first() else {
        return addresses;
    };
    let first_v6 = first.is_ipv6();
    let (mut firsts, mut seconds): (Vec<_>, Vec<_>) = addresses
        .iter()
        .partition(|address| address.is_ipv6() == first_v6);
    if seconds.is_empty() {
        return addresses;
    }
    let mut out = Vec::with_capacity(addresses.len());
    let (mut firsts, mut seconds) = (firsts.drain(..), seconds.drain(..));
    loop {
        match (firsts.next(), seconds.next()) {
            (None, None) => return out,
            (a, b) => out.extend(a.into_iter().chain(b).copied()),
        }
    }
}

/// gRPC's comparison of two destinations of RFC 6724 (`rfc_6724_compare`): of a source address
/// first, of a scope and of a label of the destination that are its source's, of the higher
/// precedence, of the smaller scope, of the longer prefix that an IPv6 destination shares with its
/// source, else of their order.
fn compare(
    (a, a_source, a_index): &(SocketAddr, Option<SocketAddr>, usize),
    (b, b_source, b_index): &(SocketAddr, Option<SocketAddr>, usize),
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let exists = |source: &Option<SocketAddr>| source.is_some();
    if exists(a_source) != exists(b_source) {
        return if exists(a_source) {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let matches = |dest: &SocketAddr, source: &Option<SocketAddr>, of: fn(Option<IpAddr>) -> u8| {
        of(Some(dest.ip())) == of(source.map(|source| source.ip()))
    };
    for of in [scope as fn(Option<IpAddr>) -> u8, label] {
        let (a_matches, b_matches) = (matches(a, a_source, of), matches(b, b_source, of));
        if a_matches != b_matches {
            return if a_matches {
                Ordering::Less
            } else {
                Ordering::Greater
            };
        }
    }
    let by_precedence = precedence(Some(b.ip())).cmp(&precedence(Some(a.ip())));
    if by_precedence != Ordering::Equal {
        return by_precedence;
    }
    let by_scope = scope(Some(a.ip())).cmp(&scope(Some(b.ip())));
    if by_scope != Ordering::Equal {
        return by_scope;
    }
    if let (Some(IpAddr::V6(a_source)), Some(IpAddr::V6(b_source))) = (
        a_source.map(|source| source.ip()),
        b_source.map(|source| source.ip()),
    ) {
        if let (IpAddr::V6(a_ip), IpAddr::V6(b_ip)) = (a.ip(), b.ip()) {
            let by_prefix = common_prefix(&b_source, &b_ip).cmp(&common_prefix(&a_source, &a_ip));
            if by_prefix != Ordering::Equal {
                return by_prefix;
            }
        }
    }
    a_index.cmp(b_index)
}

/// How many leading bits `a` and `b` share.
fn common_prefix(a: &Ipv6Addr, b: &Ipv6Addr) -> u32 {
    (u128::from(*a) ^ u128::from(*b)).leading_zeros()
}

/// gRPC's scope of an address (`sockaddr_get_scope`): global of IPv4, link-local of the IPv6
/// loopback and of the link-local IPv6 addresses, site-local of the site-local ones, else global.
/// Of no address, none.
fn scope(address: Option<IpAddr>) -> u8 {
    match address {
        None => 0,
        Some(IpAddr::V4(_)) => SCOPE_GLOBAL,
        Some(IpAddr::V6(ip)) => {
            let segments = ip.segments();
            if ip.is_loopback() || segments[0] & 0xffc0 == 0xfe80 {
                SCOPE_LINK_LOCAL
            } else if segments[0] & 0xffc0 == 0xfec0 {
                SCOPE_SITE_LOCAL
            } else {
                SCOPE_GLOBAL
            }
        }
    }
}

/// The kind of an IPv6 address of the policy table of RFC 6724, as gRPC tells it.
fn kind(ip: &Ipv6Addr) -> Kind {
    let segments = ip.segments();
    if ip.is_loopback() {
        Kind::Loopback
    } else if ip.to_ipv4_mapped().is_some() {
        Kind::V4Mapped
    } else if segments[0] == 0x2002 {
        Kind::SixToFour
    } else if segments[0] == 0x2001 && segments[1] == 0 {
        Kind::Teredo
    } else if segments[0] & 0xfe00 == 0xfc00 {
        Kind::Ula
    } else if segments[..6] == [0; 6] {
        // `::/96` but the loopback, of `::` too, as gRPC's `in6_is_addr_v4compat`.
        Kind::V4Compatible
    } else if segments[0] & 0xffc0 == 0xfec0 {
        Kind::SiteLocal
    } else if segments[0] == 0x3ffe {
        Kind::SixBone
    } else {
        Kind::Other
    }
}

/// The kinds of addresses of the policy table of RFC 6724 (section 2.1), as gRPC's table has them.
enum Kind {
    Loopback,
    V4Mapped,
    SixToFour,
    Teredo,
    Ula,
    V4Compatible,
    SiteLocal,
    SixBone,
    Other,
}

/// gRPC's label of an address (`get_label_value`): of no address, 1.
fn label(address: Option<IpAddr>) -> u8 {
    match address {
        None => 1,
        Some(IpAddr::V4(_)) => 4,
        Some(IpAddr::V6(ip)) => match kind(&ip) {
            Kind::Loopback => 0,
            Kind::V4Mapped => 4,
            Kind::SixToFour => 2,
            Kind::Teredo => 5,
            Kind::Ula => 13,
            Kind::V4Compatible => 3,
            Kind::SiteLocal => 11,
            Kind::SixBone => 12,
            Kind::Other => 1,
        },
    }
}

/// gRPC's precedence of an address (`get_precedence_value`): of no address, 1.
fn precedence(address: Option<IpAddr>) -> u8 {
    match address {
        None => 1,
        Some(IpAddr::V4(_)) => 35,
        Some(IpAddr::V6(ip)) => match kind(&ip) {
            Kind::Loopback => 50,
            Kind::V4Mapped => 35,
            Kind::SixToFour => 30,
            Kind::Teredo => 5,
            Kind::Ula => 3,
            Kind::V4Compatible | Kind::SiteLocal | Kind::SixBone => 1,
            Kind::Other => 40,
        },
    }
}

/// The source address that the system takes for a connection to `address`, as gRPC's sorting
/// finds it (`posix_source_addr_factory_get_source_addr`): the local address of a socket of UDP
/// of its family connected to it, or None of one that cannot connect to it.
pub(super) fn system_source(address: SocketAddr) -> Option<SocketAddr> {
    let local: SocketAddr = match address {
        SocketAddr::V4(_) => (std::net::Ipv4Addr::UNSPECIFIED, 0).into(),
        SocketAddr::V6(_) => (Ipv6Addr::UNSPECIFIED, 0).into(),
    };
    let socket = std::net::UdpSocket::bind(local).ok()?;
    socket.connect(address).ok()?;
    socket.local_addr().ok()
}
