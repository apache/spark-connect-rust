//! `build.rs` helper for Rust UDFs compiled to WebAssembly.
//!
//! Call [`embed_wasm_udf`] from a crate's `build.rs` to compile a Rust source
//! file containing `#[spark_wasm_udf]` functions to a `wasm32` module and expose
//! it to the crate as the `WASM_UDFS_MODULE` env var (for `include_bytes!`).
//!
//! # Requirements (build machine)
//!
//! * the `wasm32-unknown-unknown` target
//!   (`rustup target add wasm32-unknown-unknown`), and
//! * `apache-spark-connect-macros` listed as a **build-dependency** of the
//!   calling crate, so its proc-macro is compiled before `build.rs` runs and
//!   this helper can find its dylib.
//!
//! # Example `build.rs`
//!
//! ```no_run
//! fn main() {
//!     spark_connect_build::embed_wasm_udf("src/main.rs");
//! }
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

/// Compile `src_file` to a `wasm32-unknown-unknown` cdylib and expose the
/// resulting module to the crate.
///
/// Emits `cargo:rustc-env=WASM_UDFS_MODULE=<path>` (so the crate can
/// `include_bytes!(env!("WASM_UDFS_MODULE"))`) and a `rerun-if-changed` for
/// `src_file`, and returns the module path. Panics with a descriptive message
/// on failure (the intended behavior for a build script).
///
/// The source is compiled as a separate crate: any host-only code (the client,
/// the macro-generated constructors) must be gated behind
/// `#[cfg(not(target_arch = "wasm32"))]` so the wasm build keeps only the
/// exported UDF functions.
pub fn embed_wasm_udf(src_file: impl AsRef<Path>) -> PathBuf {
    let src_file = src_file.as_ref();
    let out_dir = PathBuf::from(env("OUT_DIR"));
    let manifest_dir = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    let src_abs = if src_file.is_absolute() {
        src_file.to_path_buf()
    } else {
        manifest_dir.join(src_file)
    };

    let stem = src_abs
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "wasm_udf".to_string());
    let wasm_out = out_dir.join(format!("{stem}.wasm"));

    // deps dir: OUT_DIR = target/<profile>/build/<crate>-<hash>/out
    let deps_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("unexpected OUT_DIR layout")
        .join("deps");
    let macro_dylib = find_proc_macro(&deps_dir, "spark_connect_macros").unwrap_or_else(|| {
        panic!(
            "could not find the spark_connect_macros proc-macro dylib in {}. \
             Add `apache-spark-connect-macros` to [build-dependencies].",
            deps_dir.display()
        )
    });

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
        .arg(&src_abs)
        .arg("-o")
        .arg(&wasm_out)
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke `{rustc}` for the wasm build: {e}"));
    assert!(
        status.success(),
        "wasm32 build of {} failed",
        src_abs.display()
    );

    println!("cargo:rustc-env=WASM_UDFS_MODULE={}", wasm_out.display());
    println!("cargo:rerun-if-changed={}", src_abs.display());
    wasm_out
}

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("`{key}` is not set (call from a build script)"))
}

/// Find `lib<name>-<hash>.{dylib,so,dll}` in `deps_dir`, newest first.
fn find_proc_macro(deps_dir: &Path, name: &str) -> Option<PathBuf> {
    let prefix = format!("lib{name}-");
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(deps_dir).ok()?.flatten() {
        let path = entry.path();
        let fname = match path.file_name() {
            Some(f) => f.to_string_lossy().into_owned(),
            None => continue,
        };
        let is_dylib =
            fname.ends_with(".dylib") || fname.ends_with(".so") || fname.ends_with(".dll");
        if fname.starts_with(&prefix) && is_dylib {
            if let Ok(mtime) = entry.metadata().and_then(|m| m.modified()) {
                if best.as_ref().map(|(t, _)| mtime > *t).unwrap_or(true) {
                    best = Some((mtime, path));
                }
            }
        }
    }
    best.map(|(_, p)| p)
}
