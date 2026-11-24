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

//! macOS's configuration of DNS, as c-ares reads it in place of `/etc/resolv.conf`
//! (`ares_sysconfig_mac.c`): the resolvers of `dns_configuration_copy` of dnsinfo, of libSystem or
//! of SystemConfiguration, which `dlopen` loads (see [`resolvers`]).

use std::ffi::{c_char, CStr};

use super::system::Resolver;

/// `dns_resolver_t` of dnsinfo (`thirdparty/apple/dnsinfo.h` of c-ares), packed of 4 bytes, of
/// pointers of 64 bits (`DNS_PTR`, a union of a pointer and a `uint64_t`).
#[repr(C, packed(4))]
struct DnsResolver {
    domain: u64,
    n_nameserver: i32,
    nameserver: u64,
    port: u16,
    n_search: i32,
    search: u64,
    n_sortaddr: i32,
    sortaddr: u64,
    options: u64,
    timeout: u32,
    search_order: u32,
    if_index: u32,
    flags: u32,
    reserved: [u32; 6],
    reach_flags: u32,
    reserved_after: [u32; 5],
}

/// `dns_config_t` of dnsinfo.
#[repr(C, packed(4))]
struct DnsConfig {
    n_resolver: i32,
    resolver: u64,
    n_scoped_resolver: i32,
    scoped_resolver: u64,
    reserved: [u32; 5],
}

/// `kSCNetworkFlagsReachable` and `kSCNetworkReachabilityFlagsConnectionOnTraffic`, of which a
/// resolver is read.
const REACHABLE: u32 = (1 << 1) | (1 << 3);

/// The resolvers of macOS's configuration of DNS (the `resolver` ones, as c-ares takes them, and
/// not the scoped ones), of `dns_configuration_copy` of the first library of it that `dlopen`
/// loads, as c-ares's `dnsinfo_init`; None of none, which leaves `/etc/resolv.conf` to read.
pub(crate) fn resolvers() -> Option<Vec<Resolver>> {
    type Copy = unsafe extern "C" fn() -> *mut DnsConfig;
    type Free = unsafe extern "C" fn(*mut DnsConfig);
    for library in [
        c"/usr/lib/libSystem.dylib",
        c"/System/Library/Frameworks/SystemConfiguration.framework/SystemConfiguration",
    ] {
        // SAFETY: a path of a NUL-terminated string, and a handle that is closed below.
        let handle = unsafe { libc::dlopen(library.as_ptr(), libc::RTLD_LAZY) };
        if handle.is_null() {
            continue;
        }
        // SAFETY: names of NUL-terminated strings, of a handle that `dlopen` gave.
        let (copy, free) = unsafe {
            (
                libc::dlsym(handle, c"dns_configuration_copy".as_ptr()),
                libc::dlsym(handle, c"dns_configuration_free".as_ptr()),
            )
        };
        if copy.is_null() || free.is_null() {
            // SAFETY: the handle that `dlopen` gave, closed once.
            unsafe { libc::dlclose(handle) };
            continue;
        }
        // SAFETY: the functions of dnsinfo.h, of those signatures.
        let (copy, free) = unsafe {
            (
                std::mem::transmute::<*mut libc::c_void, Copy>(copy),
                std::mem::transmute::<*mut libc::c_void, Free>(free),
            )
        };
        // SAFETY: the configuration that `dns_configuration_copy` gives is read, then freed of
        // `dns_configuration_free`, and the handle closed, once.
        let resolvers = unsafe {
            let config = copy();
            let resolvers = (!config.is_null()).then(|| read(config));
            if !config.is_null() {
                free(config);
            }
            libc::dlclose(handle);
            resolvers
        };
        return resolvers;
    }
    None
}

/// The resolvers of `config`, read unaligned, as c-ares reads them of `memcpy`.
///
/// # Safety
///
/// `config` is a configuration of `dns_configuration_copy`, not freed.
unsafe fn read(config: *const DnsConfig) -> Vec<Resolver> {
    let count = std::ptr::addr_of!((*config).n_resolver).read_unaligned();
    let array = std::ptr::addr_of!((*config).resolver).read_unaligned() as usize
        as *const *const DnsResolver;
    if array.is_null() {
        return Vec::new();
    }
    (0..usize::try_from(count).unwrap_or(0))
        .filter_map(|i| {
            let resolver = array.add(i).read_unaligned();
            (!resolver.is_null()).then(|| resolver_of(resolver))
        })
        .collect()
}

/// What c-ares reads of `resolver`.
///
/// # Safety
///
/// `resolver` is a resolver of a configuration of `dns_configuration_copy`, not freed.
unsafe fn resolver_of(resolver: *const DnsResolver) -> Resolver {
    let domain = std::ptr::addr_of!((*resolver).domain).read_unaligned();
    let nameservers = std::ptr::addr_of!((*resolver).n_nameserver).read_unaligned();
    let n_search = std::ptr::addr_of!((*resolver).n_search).read_unaligned();
    let search =
        std::ptr::addr_of!((*resolver).search).read_unaligned() as usize as *const *const c_char;
    let options = std::ptr::addr_of!((*resolver).options).read_unaligned();
    let reach_flags = std::ptr::addr_of!((*resolver).reach_flags).read_unaligned();
    let search = if search.is_null() {
        Vec::new()
    } else {
        (0..usize::try_from(n_search).unwrap_or(0))
            .filter_map(|i| string(search.add(i).read_unaligned()))
            .collect()
    };
    Resolver {
        domain: string(domain as usize as *const c_char),
        reachable: reach_flags & REACHABLE != 0,
        nameservers: usize::try_from(nameservers).unwrap_or(0),
        search,
        options: string(options as usize as *const c_char),
    }
}

/// The text of the C string `pointer`, None of a null one.
///
/// # Safety
///
/// `pointer` is null, or of a NUL-terminated string that lives while it is read.
unsafe fn string(pointer: *const c_char) -> Option<String> {
    (!pointer.is_null()).then(|| CStr::from_ptr(pointer).to_string_lossy().into_owned())
}
