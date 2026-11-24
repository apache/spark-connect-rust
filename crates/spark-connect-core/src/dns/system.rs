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

//! The configuration of DNS of the sources of a system of its own, as c-ares reads them in place
//! of `/etc/resolv.conf`: the resolvers of macOS's configuration (dnsinfo, see `super::macos`) and
//! the search list of Windows's registry (see `super::windows`), as the text of a `resolv.conf`,
//! which [`super::config::Config`] reads as it reads that file.

/// A resolver of macOS's configuration of DNS (`dns_resolver_t` of dnsinfo), of what c-ares reads
/// of it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct Resolver {
    /// The domain of the names that it resolves, of a resolver of some domains alone.
    pub(crate) domain: Option<String>,
    /// Whether its servers are reachable, or can be of traffic to them.
    pub(crate) reachable: bool,
    /// How many name servers it has.
    pub(crate) nameservers: usize,
    /// Its search domains, of their order.
    pub(crate) search: Vec<String>,
    /// Its options, as those of a `resolv.conf` (`ndots:2 timeout:3`).
    pub(crate) options: Option<String>,
}

/// The `resolv.conf` of the resolvers `resolvers` of macOS's configuration of DNS, as c-ares's
/// `read_resolvers` takes them (`ares_sysconfig_mac.c`): of each resolver of no domain, reachable
/// and of a name server, in their order, its search domains, but a domain of a resolver before it,
/// of either case, and its options.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn resolv_conf_of_resolvers(resolvers: &[Resolver]) -> String {
    let mut search: Vec<&str> = Vec::new();
    let mut options = Vec::new();
    for resolver in resolvers {
        if resolver.domain.is_some() || !resolver.reachable || resolver.nameservers == 0 {
            continue;
        }
        for domain in &resolver.search {
            if !search
                .iter()
                .any(|known| known.eq_ignore_ascii_case(domain))
            {
                search.push(domain);
            }
        }
        options.extend(resolver.options.as_deref());
    }
    let mut text = String::new();
    if !search.is_empty() {
        text.push_str(&format!("search {}\n", search.join(" ")));
    }
    for options in options {
        text.push_str(&format!("options {options}\n"));
    }
    text
}

/// The `resolv.conf` of the search list `line` of Windows's registry, as c-ares's
/// `get_SuffixList_Windows` joins its values of `,` and `ares_strsplit` splits it, of `,` and
/// spaces: of no empty domain, and of no domain twice, of either case.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn resolv_conf_of_search_list(line: &str) -> String {
    let mut search: Vec<&str> = Vec::new();
    for domain in line.split([',', ' ']).filter(|domain| !domain.is_empty()) {
        if !search
            .iter()
            .any(|known| known.eq_ignore_ascii_case(domain))
        {
            search.push(domain);
        }
    }
    if search.is_empty() {
        String::new()
    } else {
        format!("search {}\n", search.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolver(
        domain: Option<&str>,
        reachable: bool,
        nameservers: usize,
        search: &[&str],
    ) -> Resolver {
        Resolver {
            domain: domain.map(str::to_string),
            reachable,
            nameservers,
            search: search.iter().map(|domain| domain.to_string()).collect(),
            options: None,
        }
    }

    #[test]
    fn the_resolvers_of_macos_are_read_as_c_ares_reads_them() {
        // Of each resolver of no domain, reachable and of a name server, its search domains,
        // joined of no domain twice, and its options, of their order.
        let mut first = resolver(None, true, 1, &["corp.example", "example"]);
        first.options = Some("ndots:2".to_string());
        let mut second = resolver(None, true, 2, &["EXAMPLE", "vpn.example"]);
        second.options = Some("timeout:3 attempts:1".to_string());
        let resolvers = [
            first,
            // A resolver of a domain (`local`, a split one of a VPN), not reachable, or of no
            // name server, is not read.
            resolver(Some("local"), true, 1, &["local.example"]),
            resolver(None, false, 1, &["unreachable.example"]),
            resolver(None, true, 0, &["serverless.example"]),
            second,
        ];
        assert_eq!(
            resolv_conf_of_resolvers(&resolvers),
            "search corp.example example vpn.example\noptions ndots:2\noptions timeout:3 attempts:1\n"
        );
        assert_eq!(resolv_conf_of_resolvers(&[]), "");
        // Read as a `resolv.conf`, of its options of each resolver.
        let config = super::super::config::Config::of(
            &|file| (file == "resolv.conf").then(|| resolv_conf_of_resolvers(&resolvers)),
            &|_| None,
            None,
        );
        assert_eq!(config.domains, ["corp.example", "example", "vpn.example"]);
        assert_eq!(config.ndots, 2);
        assert_eq!(config.tries, 1);
        assert_eq!(config.timeout, std::time::Duration::from_secs(3));
    }

    #[test]
    fn the_search_list_of_windows_is_read_as_c_ares_reads_it() {
        assert_eq!(
            resolv_conf_of_search_list(
                "corp.example,example, other.example  corp.example,,CORP.EXAMPLE"
            ),
            "search corp.example example other.example\n"
        );
        assert_eq!(resolv_conf_of_search_list(", ,"), "");
    }
}
