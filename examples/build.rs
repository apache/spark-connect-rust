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

//! Build script for the `examples` crate.
//!
//! When the `wasm-udf-macro` feature is enabled, compile the sibling
//! `wasm-udfs` crate to `wasm32-unknown-unknown` and expose the resulting
//! `.wasm` path via the `WASM_UDFS_MODULE` env var, so `wasm_udf_macro.rs` can
//! `include_bytes!` it. This is what lets `cargo run` "just work" without a
//! manual WASM build step.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only do the (cross-target) WASM build when the feature is on, so the
    // default `cargo build -p examples` never requires the wasm32 target.
    if std::env::var_os("CARGO_FEATURE_WASM_UDF_MACRO").is_none() {
        return;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let wasm_udfs_dir = manifest_dir.join("..").join("wasm-udfs");
    let wasm_manifest = wasm_udfs_dir.join("Cargo.toml");

    // Build into a separate target dir so this nested cargo invocation does not
    // deadlock on the outer build's target-dir lock.
    let nested_target = out_dir.join("wasm-build");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());

    let status = Command::new(&cargo)
        .args([
            "build",
            "-p",
            "wasm-udfs",
            "--target",
            "wasm32-unknown-unknown",
            "--release",
        ])
        .arg("--manifest-path")
        .arg(&wasm_manifest)
        .arg("--target-dir")
        .arg(&nested_target)
        .status()
        .expect("failed to invoke cargo to build wasm-udfs");
    assert!(status.success(), "building wasm-udfs for wasm32 failed");

    let wasm = nested_target
        .join("wasm32-unknown-unknown")
        .join("release")
        .join("wasm_udfs.wasm");
    let dest = out_dir.join("wasm_udfs.wasm");
    std::fs::copy(&wasm, &dest)
        .unwrap_or_else(|e| panic!("copying {} -> {}: {e}", wasm.display(), dest.display()));

    println!("cargo:rustc-env=WASM_UDFS_MODULE={}", dest.display());
    println!(
        "cargo:rerun-if-changed={}",
        wasm_udfs_dir.join("src").join("lib.rs").display()
    );
    println!("cargo:rerun-if-changed={}", wasm_manifest.display());
}
