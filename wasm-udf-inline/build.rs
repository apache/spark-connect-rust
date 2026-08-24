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

//! Compile `src/main.rs` to a `wasm32` cdylib and embed it.
//!
//! We invoke `rustc` directly (not cargo) so the *same* file that contains the
//! host client also becomes the wasm module: `main` and the host-only code are
//! `#[cfg(not(target_arch = "wasm32"))]`, so the wasm build keeps only the
//! `#[spark_wasm_udf]` export. Direct `rustc` avoids build-script recursion and
//! needs just the `spark_connect_macros` proc-macro dylib (built ahead of us
//! because it is also a build-dependency).

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let main_rs = manifest_dir.join("src").join("main.rs");
    let wasm_out = out_dir.join("wasm_udf_inline.wasm");

    // deps dir: OUT_DIR = target/<profile>/build/<crate>-<hash>/out
    let deps_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("unexpected OUT_DIR layout")
        .join("deps");
    let macro_dylib = find_proc_macro(&deps_dir, "spark_connect_macros")
        .unwrap_or_else(|| panic!("could not find spark_connect_macros dylib in {deps_dir:?}"));

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let status = Command::new(&rustc)
        .args(["--edition", "2021", "--crate-type", "cdylib"])
        .args(["--target", "wasm32-unknown-unknown"])
        // Keep the embedded module small and self-contained.
        .args([
            "-C",
            "opt-level=s",
            "-C",
            "panic=abort",
            "-C",
            "strip=symbols",
        ])
        .arg("-L")
        .arg(format!("dependency={}", deps_dir.display()))
        .arg("--extern")
        .arg(format!("spark_connect_macros={}", macro_dylib.display()))
        .arg(&main_rs)
        .arg("-o")
        .arg(&wasm_out)
        .status()
        .expect("failed to invoke rustc for the wasm build");
    assert!(status.success(), "wasm32 build of src/main.rs failed");

    println!("cargo:rustc-env=WASM_UDFS_MODULE={}", wasm_out.display());
    println!("cargo:rerun-if-changed={}", main_rs.display());
}

/// Find `lib<name>-<hash>.{dylib,so,dll}` in `deps_dir`, newest first.
fn find_proc_macro(deps_dir: &Path, name: &str) -> Option<PathBuf> {
    let prefix = format!("lib{name}-");
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(deps_dir).ok()?.flatten() {
        let path = entry.path();
        let fname = path.file_name()?.to_string_lossy().into_owned();
        let is_dylib =
            fname.ends_with(".dylib") || fname.ends_with(".so") || fname.ends_with(".dll");
        if fname.starts_with(&prefix) && is_dylib {
            let mtime = entry.metadata().ok()?.modified().ok()?;
            if best.as_ref().map(|(t, _)| mtime > *t).unwrap_or(true) {
                best = Some((mtime, path));
            }
        }
    }
    best.map(|(_, p)| p)
}
