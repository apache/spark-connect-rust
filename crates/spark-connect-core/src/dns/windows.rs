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

//! Windows's search list of DNS, of its registry, as c-ares reads it (`get_SuffixList_Windows` of
//! `ares_sysconfig_win.c`): the `SearchList` and the `Domain` of TCP/IP, the `SearchList` of the
//! policies of Windows NT's DNS client, the `PrimaryDNSSuffix` of those of the system's, and the
//! `SearchList`, the `Domain` and the `DhcpDomain` of each interface of TCP/IP, joined of `,`.

use windows_sys::Win32::Foundation::{ERROR_MORE_DATA, ERROR_SUCCESS};
use windows_sys::Win32::System::Registry::{
    RegCloseKey, RegEnumKeyExA, RegOpenKeyExA, RegQueryValueExA, HKEY, HKEY_LOCAL_MACHINE,
    KEY_QUERY_VALUE, KEY_READ,
};

/// The key of TCP/IP's parameters (`WIN_NS_NT_KEY`).
const TCPIP: &[u8] = b"System\\CurrentControlSet\\Services\\Tcpip\\Parameters\0";
/// The key of the policies of Windows NT's DNS client (`WIN_NT_DNSCLIENT`).
const NT_DNS_CLIENT: &[u8] = b"Software\\Policies\\Microsoft\\Windows NT\\DNSClient\0";
/// The key of the policies of the system's DNS client (`WIN_DNSCLIENT`).
const DNS_CLIENT: &[u8] = b"Software\\Policies\\Microsoft\\System\\DNSClient\0";
/// The key of the interfaces of TCP/IP.
const INTERFACES: &[u8] = b"System\\CurrentControlSet\\Services\\Tcpip\\Parameters\\Interfaces\0";

/// The search list of the registry, of its values of c-ares's order joined of `,`, None of none.
pub(crate) fn search_list() -> Option<String> {
    let mut values = Vec::new();
    for (key, names) in [
        (TCPIP, &[&b"SearchList\0"[..], b"Domain\0"][..]),
        (NT_DNS_CLIENT, &[&b"SearchList\0"[..]][..]),
        (DNS_CLIENT, &[&b"PrimaryDNSSuffix\0"[..]][..]),
    ] {
        if let Some(key) = Key::open(HKEY_LOCAL_MACHINE, key, KEY_READ) {
            values.extend(names.iter().filter_map(|name| key.value(name)));
        }
    }
    if let Some(interfaces) = Key::open(HKEY_LOCAL_MACHINE, INTERFACES, KEY_READ) {
        for index in 0.. {
            let mut name = [0u8; 256];
            let mut size = name.len() as u32;
            // SAFETY: a buffer of `size` bytes, of no class and no time asked.
            let result = unsafe {
                RegEnumKeyExA(
                    interfaces.0,
                    index,
                    name.as_mut_ptr(),
                    &mut size,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            };
            if result != ERROR_SUCCESS {
                break;
            }
            // The name, NUL-terminated, as the call writes it of fewer than 256 characters.
            let end = usize::try_from(size)
                .unwrap_or(name.len())
                .min(name.len() - 1);
            name[end] = 0;
            let Some(interface) = Key::open(interfaces.0, &name[..=end], KEY_QUERY_VALUE) else {
                continue;
            };
            for value in [&b"SearchList\0"[..], b"Domain\0", b"DhcpDomain\0"] {
                values.extend(interface.value(value));
            }
        }
    }
    (!values.is_empty()).then(|| values.join(","))
}

/// An open key of the registry, closed as it drops.
struct Key(HKEY);

impl Key {
    /// The key `name` (NUL-terminated) of `parent`, of the access `access`, None of none.
    fn open(parent: HKEY, name: &[u8], access: u32) -> Option<Self> {
        let mut key: HKEY = std::ptr::null_mut();
        // SAFETY: a NUL-terminated name, and a key that is closed as it drops.
        let result = unsafe { RegOpenKeyExA(parent, name.as_ptr(), 0, access, &mut key) };
        (result == ERROR_SUCCESS).then_some(Self(key))
    }

    /// The string of the value `name` (NUL-terminated) of the key, as c-ares's `get_REG_SZ` reads
    /// it: None of none, or of an empty one.
    fn value(&self, name: &[u8]) -> Option<String> {
        let mut size = 0u32;
        // SAFETY: a NUL-terminated name, of no data asked but its size.
        let result = unsafe {
            RegQueryValueExA(
                self.0,
                name.as_ptr(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size,
            )
        };
        if (result != ERROR_SUCCESS && result != ERROR_MORE_DATA) || size == 0 {
            return None;
        }
        let mut buffer = vec![0u8; usize::try_from(size).ok()? + 1];
        // SAFETY: a buffer of `size` bytes and one more, as `size` says.
        let result = unsafe {
            RegQueryValueExA(
                self.0,
                name.as_ptr(),
                std::ptr::null(),
                std::ptr::null_mut(),
                buffer.as_mut_ptr(),
                &mut size,
            )
        };
        if result != ERROR_SUCCESS || size == 1 {
            return None;
        }
        buffer.truncate(usize::try_from(size).ok()?);
        let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
        Some(String::from_utf8_lossy(&buffer[..end]).into_owned())
    }
}

impl Drop for Key {
    fn drop(&mut self) {
        // SAFETY: a key that `RegOpenKeyExA` opened, closed once.
        unsafe { RegCloseKey(self.0) };
    }
}
