//! The names of the certificate of a TLS server, as gRPC's checks of a target of gRPC read them.
//!
//! gRPC checks the name of a TLS server twice: once its handshake is done, of the name of the
//! target (`HostNameCertificateVerifier`), which fails the connection (`Custom verification check
//! failed with error: UNAUTHENTICATED: Hostname Verification Check failed.`), and for each call, of
//! the host of the call (`SslCheckCallHost`), which fails the call (`UNAUTHENTICATED`, `call host
//! does not match SSL server name`). The two take the names of the certificate as BoringSSL gives
//! them to gRPC's peer: its subject alternative names, of DNS, e-mail and URI as text and of IP
//! addresses as `inet_ntop` writes them, and the first common name of its subject; and they match a
//! name of them by rules of their own, which this module follows, where webpki's differ: a
//! wildcard of one label (`*.localhost`) and a name of a `*` pass the first; an IP address is its
//! text (`0:0:0:0:0:0:0:1` is not `::1`); a certificate of no name of DNS or IP is of its common
//! name.

/// The names of the certificate of a server, as gRPC's peer has them.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct PeerNames {
    /// The subject alternative names, in their order.
    names: Vec<Name>,
    /// The first common name of the subject.
    common_name: Option<String>,
}

/// A subject alternative name.
#[derive(Debug, PartialEq, Eq)]
enum Name {
    /// Of DNS.
    Dns(String),
    /// An IP address, as `inet_ntop` writes it.
    Ip(String),
    /// An e-mail address or a URI.
    Other(String),
}

impl PeerNames {
    /// The names of the DER certificate `der`, None of one that is not of X.509.
    pub(crate) fn of_certificate(der: &[u8]) -> Option<Self> {
        let (SEQUENCE, certificate, _) = tlv(der)? else {
            return None;
        };
        let (SEQUENCE, tbs, _) = tlv(certificate)? else {
            return None;
        };
        // TBSCertificate (RFC 5280, section 4.1): [0] version, then the serial number, the
        // signature, the issuer and the validity, then the subject and the key.
        let mut rest = tbs;
        let (tag, _, after) = tlv(rest)?;
        if tag == 0xa0 {
            rest = after;
        }
        for _ in 0..4 {
            rest = tlv(rest)?.2;
        }
        let (SEQUENCE, subject, after) = tlv(rest)? else {
            return None;
        };
        rest = tlv(after)?.2;
        let mut names = Vec::new();
        // [1] and [2], the unique IDs, and [3], the extensions.
        while !rest.is_empty() {
            let (tag, value, after) = tlv(rest)?;
            rest = after;
            if tag == 0xa3 {
                names = subject_alt_names(value)?;
            }
        }
        Some(Self {
            names,
            common_name: common_name(subject),
        })
    }

    /// Whether the certificate is of the name `name` of a target, as gRPC's
    /// `HostNameCertificateVerifier` checks it once the handshake is done: of a name of DNS of
    /// it, of the rules of `VerifySubjectAlternativeName`, an IP address of it as its text, or of
    /// a certificate of neither, its common name.
    pub(crate) fn of_target_name(&self, name: &str) -> bool {
        if name.is_empty() {
            return false;
        }
        let mut dns = self.names.iter().filter_map(|name| match name {
            Name::Dns(dns) => Some(dns.as_str()),
            _ => None,
        });
        let mut ips = self.names.iter().filter_map(|name| match name {
            Name::Ip(ip) => Some(ip.as_str()),
            _ => None,
        });
        if dns
            .clone()
            .any(|dns| verify_subject_alternative_name(dns, name))
            || ips.clone().any(|ip| ip == name)
        {
            return true;
        }
        dns.next().is_none()
            && ips.next().is_none()
            && self
                .common_name
                .as_deref()
                .is_some_and(|common_name| verify_subject_alternative_name(common_name, name))
    }

    /// Whether the certificate is of `host`, the host of a call, as gRPC's `SslCheckCallHost`
    /// checks it for each call (`tsi_ssl_peer_matches_name`): of a name that looks like an IP
    /// address, a subject alternative name that is its text; of another, one that matches it, of
    /// its rules of a wildcard (`*.example.com`, of two labels at least); of a certificate of no
    /// subject alternative name, its common name, but of a name of an IP address.
    pub(crate) fn of_call_host(&self, host: &str) -> bool {
        if host.is_empty() {
            return false;
        }
        let like_ip = looks_like_ip_address(host);
        let text = |name: &Name| match name {
            Name::Dns(text) | Name::Ip(text) | Name::Other(text) => text.clone(),
        };
        if self.names.iter().any(|name| {
            let entry = text(name);
            if like_ip {
                entry == host
            } else {
                does_entry_match_name(&entry, host)
            }
        }) {
            return true;
        }
        self.names.is_empty()
            && !like_ip
            && self
                .common_name
                .as_deref()
                .is_some_and(|common_name| does_entry_match_name(common_name, host))
    }
}

/// The DER tag of a SEQUENCE.
const SEQUENCE: u8 = 0x30;

/// The tag, the contents and the rest of the DER value of `input`, of a tag of one byte and a
/// definite length.
fn tlv(input: &[u8]) -> Option<(u8, &[u8], &[u8])> {
    let (&tag, rest) = input.split_first()?;
    let (&first, rest) = rest.split_first()?;
    let (length, rest) = if first < 0x80 {
        (usize::from(first), rest)
    } else {
        let bytes = usize::from(first & 0x7f);
        if bytes == 0 || bytes > 4 || rest.len() < bytes {
            return None;
        }
        let length = rest[..bytes]
            .iter()
            .fold(0usize, |length, &b| (length << 8) | usize::from(b));
        (length, &rest[bytes..])
    };
    if rest.len() < length {
        return None;
    }
    Some((tag, &rest[..length], &rest[length..]))
}

/// The subject alternative names of `extensions`, the contents of the `[3]` of a TBSCertificate,
/// or none of no extension of them.
fn subject_alt_names(extensions: &[u8]) -> Option<Vec<Name>> {
    let (SEQUENCE, mut extensions, _) = tlv(extensions)? else {
        return None;
    };
    let mut names = Vec::new();
    while !extensions.is_empty() {
        let (SEQUENCE, extension, after) = tlv(extensions)? else {
            return None;
        };
        extensions = after;
        // id-ce-subjectAltName, 2.5.29.17.
        let (0x06, oid, fields) = tlv(extension)? else {
            return None;
        };
        if oid != [0x55, 0x1d, 0x11] {
            continue;
        }
        // Of a `critical` BOOLEAN, then the OCTET STRING of the GeneralNames.
        let (mut tag, mut value, after) = tlv(fields)?;
        if tag == 0x01 {
            (tag, value, _) = tlv(after)?;
        }
        if tag != 0x04 {
            return None;
        }
        let (SEQUENCE, mut general_names, _) = tlv(value)? else {
            return None;
        };
        while !general_names.is_empty() {
            let (tag, value, after) = tlv(general_names)?;
            general_names = after;
            match tag {
                // dNSName [2], rfc822Name [1] and uniformResourceIdentifier [6], IA5Strings.
                0x82 => names.push(Name::Dns(latin1(value))),
                0x81 | 0x86 => names.push(Name::Other(latin1(value))),
                // iPAddress [7].
                0x87 => names.push(Name::Ip(ip_text(value)?)),
                _ => {}
            }
        }
    }
    Some(names)
}

/// The first common name of `subject`, the contents of a Name, as text of UTF-8, as BoringSSL's
/// `ASN1_STRING_to_UTF8` gives it, None of none.
fn common_name(subject: &[u8]) -> Option<String> {
    let mut rdns = subject;
    while !rdns.is_empty() {
        let (0x31, mut attributes, after) = tlv(rdns)? else {
            return None;
        };
        rdns = after;
        while !attributes.is_empty() {
            let (SEQUENCE, attribute, after) = tlv(attributes)? else {
                return None;
            };
            attributes = after;
            let (0x06, oid, value) = tlv(attribute)? else {
                return None;
            };
            // id-at-commonName, 2.5.4.3.
            if oid == [0x55, 0x04, 0x03] {
                let (tag, value, _) = tlv(value)?;
                return string_of(tag, value);
            }
        }
    }
    None
}

/// The text of a string of ASN.1 of `tag`: of UTF-8 as it is, of UTF-16 (BMPString) and of UTF-32
/// (UniversalString) decoded, and of the strings of one byte (PrintableString, T61String,
/// IA5String, VisibleString) of Latin-1, as `ASN1_STRING_to_UTF8` takes them.
fn string_of(tag: u8, value: &[u8]) -> Option<String> {
    match tag {
        0x0c => String::from_utf8(value.to_vec()).ok(),
        0x1e => {
            let units: Vec<u16> = value
                .chunks(2)
                .map(|pair| u16::from_be_bytes([pair[0], *pair.get(1)?]).into())
                .collect::<Option<_>>()?;
            String::from_utf16(&units).ok()
        }
        0x1c => value
            .chunks(4)
            .map(|quad| {
                let quad: [u8; 4] = quad.try_into().ok()?;
                char::from_u32(u32::from_be_bytes(quad))
            })
            .collect(),
        0x13 | 0x14 | 0x16 | 0x1a => Some(latin1(value)),
        _ => None,
    }
}

/// `bytes` of Latin-1, as text.
fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| char::from(b)).collect()
}

/// The text of the IP address of `bytes`, of 4 or 16 bytes, as glibc's `inet_ntop` writes it:
/// an IPv6 address in lower case, of its longest run of zeros of two groups at least, the first
/// of the longest, as `::`, and of an IPv4 address of its last 32 bits of the first 80 of zeros
/// and then of zeros or `ffff` (`::ffff:127.0.0.1`, `::127.0.0.1`) in dotted decimal. None of
/// another length.
fn ip_text(bytes: &[u8]) -> Option<String> {
    if let Ok(v4) = <[u8; 4]>::try_from(bytes) {
        return Some(std::net::Ipv4Addr::from(v4).to_string());
    }
    let bytes: [u8; 16] = bytes.try_into().ok()?;
    let words: Vec<u16> = bytes
        .chunks(2)
        .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
        .collect();
    // The longest run of zeros, the first of the longest.
    // Of a run of zeros, its first group and its length.
    let mut best: Option<(usize, usize)> = None;
    let mut current: Option<(usize, usize)> = None;
    for (i, &word) in words.iter().enumerate() {
        if word == 0 {
            current = Some(match current {
                Some((base, len)) => (base, len + 1),
                None => (i, 1),
            });
        } else if let Some(run) = current.take() {
            if best.is_none_or(|best| run.1 > best.1) {
                best = Some(run);
            }
        }
    }
    if let Some(run) = current {
        if best.is_none_or(|best| run.1 > best.1) {
            best = Some(run);
        }
    }
    let best = best.filter(|&(_, len)| len >= 2);
    let mut text = String::new();
    let mut i = 0;
    while i < 8 {
        if let Some((base, len)) = best {
            if i >= base && i < base + len {
                if i == base {
                    text.push(':');
                }
                i += 1;
                continue;
            }
        }
        if i != 0 {
            text.push(':');
        }
        if i == 6
            && best.is_some_and(|(base, len)| {
                base == 0 && (len == 6 || (len == 5 && words[5] == 0xffff))
            })
        {
            text.push_str(
                &std::net::Ipv4Addr::new(bytes[12], bytes[13], bytes[14], bytes[15]).to_string(),
            );
            return Some(text);
        }
        text.push_str(&format!("{:x}", words[i]));
        i += 1;
    }
    if best.is_some_and(|(base, len)| base + len == 8) {
        text.push(':');
    }
    Some(text)
}

/// Whether `name` looks like an IP address, as gRPC's `looks_like_ip_address` tells it: of a `:`,
/// or of four groups of one to three digits.
fn looks_like_ip_address(name: &str) -> bool {
    let (mut dots, mut digits) = (0, 0);
    for b in name.bytes() {
        match b {
            b':' => return true,
            b'0'..=b'9' => {
                if digits > 3 {
                    return false;
                }
                digits += 1;
            }
            b'.' => {
                if dots > 3 || digits == 0 {
                    return false;
                }
                dots += 1;
                digits = 0;
            }
            _ => return false,
        }
    }
    dots >= 3 && digits > 0
}

/// Whether the name `entry` of a certificate matches `name`, as gRPC's `does_entry_match_name`:
/// of either case, of a last `.` of either, and of a wildcard `*.` of the first label alone, whose
/// rest has two labels at least.
fn does_entry_match_name(entry: &str, name: &str) -> bool {
    if entry.is_empty() {
        return false;
    }
    let name = name.strip_suffix('.').unwrap_or(name);
    let entry = match entry.strip_suffix('.') {
        Some("") => return false,
        Some(entry) => entry,
        None => entry,
    };
    if name.eq_ignore_ascii_case(entry) {
        return true;
    }
    if !entry.starts_with('*') {
        return false;
    }
    if entry.len() < 3 || entry.as_bytes()[1] != b'.' {
        return false;
    }
    let Some(at) = name.find('.') else {
        return false;
    };
    if name.len() >= 2 && at >= name.len() - 2 {
        return false;
    }
    let subdomain = &name[at + 1..];
    match subdomain.find('.') {
        None => return false,
        Some(dot) if dot == subdomain.len() - 1 => return false,
        _ => {}
    }
    let subdomain = subdomain.strip_suffix('.').unwrap_or(subdomain);
    let entry = &entry[2..];
    !entry.is_empty() && subdomain.eq_ignore_ascii_case(entry)
}

/// Whether the name `san` of a certificate matches `matcher`, as gRPC's
/// `VerifySubjectAlternativeName`: of either case and of a last `.` of either, and of a wildcard
/// `*.` of the first label alone, which matches one label, of any rest but one of none.
fn verify_subject_alternative_name(san: &str, matcher: &str) -> bool {
    if san.is_empty() || san.starts_with('.') || matcher.is_empty() || matcher.starts_with('.') {
        return false;
    }
    let absolute = |name: &str| {
        let mut name = if name.ends_with('.') {
            name.to_string()
        } else {
            format!("{name}.")
        };
        name.make_ascii_lowercase();
        name
    };
    let (san, matcher) = (absolute(san), absolute(matcher));
    if !san.contains('*') {
        return san == matcher;
    }
    if !san.starts_with("*.") || san == "*." {
        return false;
    }
    let suffix = &san[1..];
    if suffix.contains('*') || !matcher.ends_with(suffix) {
        return false;
    }
    let start = matcher.len() - suffix.len();
    start == 0 || !matcher[..start].contains('.')
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The names of a certificate of the subject alternative names `names` (of IP addresses as
    /// rcgen reads them) and of the common name `common_name`.
    fn names_of(names: &[&str], common_name: Option<&str>) -> PeerNames {
        let mut params =
            rcgen::CertificateParams::new(names.iter().map(|n| n.to_string()).collect::<Vec<_>>())
                .unwrap();
        // Of no default name of rcgen.
        params.distinguished_name = rcgen::DistinguishedName::new();
        if let Some(common_name) = common_name {
            params
                .distinguished_name
                .push(rcgen::DnType::CommonName, common_name);
        }
        let key = rcgen::KeyPair::generate().unwrap();
        let cert = params.self_signed(&key).unwrap();
        PeerNames::of_certificate(cert.der()).unwrap()
    }

    #[test]
    fn the_names_of_a_certificate_are_read_as_boringssl_gives_them() {
        let names = names_of(
            &[
                "localhost",
                "*.localhost",
                "127.0.0.1",
                "::1",
                "::ffff:127.0.0.1",
            ],
            Some("cn.example"),
        );
        assert_eq!(
            names,
            PeerNames {
                names: vec![
                    Name::Dns("localhost".into()),
                    Name::Dns("*.localhost".into()),
                    Name::Ip("127.0.0.1".into()),
                    Name::Ip("::1".into()),
                    Name::Ip("::ffff:127.0.0.1".into()),
                ],
                common_name: Some("cn.example".into()),
            }
        );
        assert_eq!(names_of(&[], None), PeerNames::default());
        assert!(PeerNames::of_certificate(b"\x30\x03\x02\x01").is_none());
    }

    #[test]
    fn ip_addresses_are_of_the_text_of_inet_ntop() {
        let text = |ip: &str| ip_text(&ip.parse::<std::net::Ipv6Addr>().unwrap().octets()).unwrap();
        for (ip, expected) in [
            ("::1", "::1"),
            ("::", "::"),
            ("0:0:0:0:0:0:0:1", "::1"),
            ("2001:db8:0:0:1:0:0:1", "2001:db8::1:0:0:1"),
            ("2001:0:0:1:0:0:0:1", "2001:0:0:1::1"),
            ("fe80::1:0:0:0", "fe80::1:0:0:0"),
            ("fe80:0:0:1::", "fe80:0:0:1::"),
            ("1:0:2:3:4:5:6:7", "1:0:2:3:4:5:6:7"),
            ("::ffff:127.0.0.1", "::ffff:127.0.0.1"),
            ("::7f00:1", "::127.0.0.1"),
            ("::ffff:0:1", "::ffff:0.0.0.1"),
            ("ABCD::EF", "abcd::ef"),
        ] {
            assert_eq!(text(ip), expected, "{ip}");
        }
        assert_eq!(ip_text(&[127, 0, 0, 1]).unwrap(), "127.0.0.1");
        assert!(ip_text(&[1, 2, 3]).is_none());
    }

    #[test]
    fn the_name_of_a_target_is_checked_as_grpcs_verifier_checks_it() {
        // Of the probes of grpcio 1.85.1 (`p_tls_edges.py` of the review of SPARK-62220).
        let wildcard = names_of(&["*.localhost"], None);
        for name in [
            "xyz.localhost",
            "x*y.localhost",
            "XYZ.LOCALHOST",
            "xyz.localhost.",
        ] {
            assert!(wildcard.of_target_name(name), "{name}");
        }
        for name in ["a.b.localhost", "localhost", ".localhost", ""] {
            assert!(!wildcard.of_target_name(name), "{name}");
        }
        let ip6 = names_of(&["::1"], None);
        assert!(ip6.of_target_name("::1"));
        assert!(!ip6.of_target_name("0:0:0:0:0:0:0:1"));
        let ip4 = names_of(&["127.0.0.1"], None);
        assert!(ip4.of_target_name("127.0.0.1"));
        assert!(!ip4.of_target_name("127.000.000.001"));
        let dns = names_of(&["localhost"], Some("other.example"));
        assert!(dns.of_target_name("LOCALHOST"));
        assert!(dns.of_target_name("localhost."));
        assert!(!dns.of_target_name("other.example"));
        // Of no name of DNS or IP, the common name.
        let common = names_of(&[], Some("cn.example"));
        assert!(common.of_target_name("cn.example"));
        assert!(!common.of_target_name("other.example"));
        assert!(!names_of(&["*."], None).of_target_name("a"));
        assert!(!names_of(&["a*.localhost"], None).of_target_name("ab.localhost"));
    }

    #[test]
    fn the_host_of_a_call_is_checked_as_grpcs_call_host_check_checks_it() {
        // `*.localhost` passes the check of the handshake but not that of each call, of which
        // grpcio's calls fail of `UNAUTHENTICATED`.
        let wildcard = names_of(&["*.localhost"], None);
        assert!(!wildcard.of_call_host("xyz.localhost"));
        assert!(!wildcard.of_call_host("x*y.localhost"));
        let wildcard = names_of(&["*.spark.example"], None);
        assert!(wildcard.of_call_host("a.spark.example"));
        assert!(wildcard.of_call_host("A.SPARK.EXAMPLE."));
        assert!(!wildcard.of_call_host("a.b.spark.example"));
        assert!(!wildcard.of_call_host("spark.example"));
        let dns = names_of(&["localhost", "a"], None);
        assert!(dns.of_call_host("localhost"));
        assert!(dns.of_call_host("a"));
        assert!(!dns.of_call_host("b"));
        // An address is its text, of any name.
        let ip = names_of(&["::1", "127.0.0.1"], None);
        assert!(ip.of_call_host("::1"));
        assert!(ip.of_call_host("127.0.0.1"));
        assert!(!ip.of_call_host("0:0:0:0:0:0:0:1"));
        // The common name of a certificate of no subject alternative name, but of an address.
        let common = names_of(&[], Some("127.0.0.1"));
        assert!(!common.of_call_host("127.0.0.1"));
        let common = names_of(&[], Some("cn.example"));
        assert!(common.of_call_host("cn.example"));
        // As gRPC's `looks_like_ip_address`, which reads up to a `:` or a character of no number,
        // and takes a group of four digits.
        for name in [
            "1.2.3.4",
            "::1",
            "0:0:0:0:0:0:0:1",
            "1.2.3.4444",
            "127.000.000.001",
        ] {
            assert!(looks_like_ip_address(name), "{name}");
        }
        for name in [
            "1.2.3",
            "1.2.3.44444",
            "1.2.3.4.",
            "x.2.3.4",
            "fe80::1",
            "a:b",
        ] {
            assert!(!looks_like_ip_address(name), "{name}");
        }
    }
}
