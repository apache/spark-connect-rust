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

    // The macro's wasm export wrappers reference `crate::spark_wasm_rt` for the
    // binary-ABI codec + allocator. Prepend that runtime and compile the
    // combined source, so the module exports `spark_udf_alloc`/`_dealloc` once.
    let user_src = std::fs::read_to_string(&src_abs)
        .unwrap_or_else(|e| panic!("reading {}: {e}", src_abs.display()));
    // User source first so its crate-level `//!` docs stay at the top; the
    // runtime is appended (item order does not affect name resolution).
    let wrapper = out_dir.join(format!("{stem}_wasm_udf_wrapper.rs"));
    std::fs::write(&wrapper, format!("{user_src}\n{WASM_RUNTIME}"))
        .unwrap_or_else(|e| panic!("writing {}: {e}", wrapper.display()));

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
        .arg(&wrapper)
        .arg("-o")
        .arg(&wasm_out)
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke `{rustc}` for the wasm build: {e}"));

    if !status.success() {
        eprintln!();
        eprintln!("cargo:warning=wasm32 build of {} failed", src_abs.display());
        eprintln!("cargo:warning=");
        eprintln!("cargo:warning=The `wasm32-unknown-unknown` Rust target is required to build WASM UDF crates.");
        eprintln!("cargo:warning=Install it with: rustup target add wasm32-unknown-unknown");
        eprintln!();
        panic!("wasm32 build of {} failed", src_abs.display());
    }

    println!("cargo:rustc-env=WASM_UDFS_MODULE={}", wasm_out.display());
    println!("cargo:rerun-if-changed={}", src_abs.display());
    wasm_out
}

/// The runtime prepended to the wasm build: the exported allocator and the
/// length-prefixed binary codec (`Abi`) the macro's export wrappers call. Must
/// stay byte-compatible with `spark_connect::wasm_udf::AbiType` and the Python
/// runner's codec in `pyspark_wasm_udf`.
const WASM_RUNTIME: &str = r#"
#[allow(dead_code)]
pub mod spark_wasm_rt {
    use std::alloc::{alloc as __alloc, dealloc as __dealloc, Layout};

    #[no_mangle]
    pub extern "C" fn spark_udf_alloc(len: u32) -> *mut u8 {
        if len == 0 { return 1 as *mut u8; }
        unsafe { __alloc(Layout::from_size_align(len as usize, 1).unwrap()) }
    }
    #[no_mangle]
    pub extern "C" fn spark_udf_dealloc(ptr: *mut u8, len: u32) {
        if len == 0 || ptr.is_null() { return; }
        unsafe { __dealloc(ptr, Layout::from_size_align(len as usize, 1).unwrap()) }
    }

    pub struct Reader<'a> { b: &'a [u8], off: usize }
    impl<'a> Reader<'a> {
        pub fn new(b: &'a [u8]) -> Self { Reader { b, off: 0 } }
        fn take(&mut self, n: usize) -> &'a [u8] {
            let s = &self.b[self.off..self.off + n]; self.off += n; s
        }
        fn u32(&mut self) -> usize {
            u32::from_le_bytes(self.take(4).try_into().unwrap()) as usize
        }
    }

    pub trait Abi: Sized {
        fn decode(r: &mut Reader) -> Self;
        fn encode(&self, out: &mut Vec<u8>);
    }

    macro_rules! prim { ($t:ty, $n:expr) => {
        impl Abi for $t {
            fn decode(r: &mut Reader) -> Self { <$t>::from_le_bytes(r.take($n).try_into().unwrap()) }
            fn encode(&self, out: &mut Vec<u8>) { out.extend_from_slice(&self.to_le_bytes()); }
        }
    }}
    prim!(i32, 4); prim!(i64, 8); prim!(f32, 4); prim!(f64, 8);

    impl Abi for bool {
        fn decode(r: &mut Reader) -> Self { r.take(1)[0] != 0 }
        fn encode(&self, out: &mut Vec<u8>) { out.push(*self as u8); }
    }
    impl Abi for u8 {
        fn decode(r: &mut Reader) -> Self { r.take(1)[0] }
        fn encode(&self, out: &mut Vec<u8>) { out.push(*self); }
    }
    impl Abi for String {
        fn decode(r: &mut Reader) -> Self {
            let n = r.u32();
            String::from_utf8(r.take(n).to_vec()).unwrap()
        }
        fn encode(&self, out: &mut Vec<u8>) {
            out.extend_from_slice(&(self.len() as u32).to_le_bytes());
            out.extend_from_slice(self.as_bytes());
        }
    }
    impl<T: Abi> Abi for Vec<T> {
        fn decode(r: &mut Reader) -> Self {
            let n = r.u32();
            (0..n).map(|_| T::decode(r)).collect()
        }
        fn encode(&self, out: &mut Vec<u8>) {
            out.extend_from_slice(&(self.len() as u32).to_le_bytes());
            for x in self { x.encode(out); }
        }
    }
    impl<T: Abi> Abi for Option<T> {
        fn decode(r: &mut Reader) -> Self {
            if r.take(1)[0] == 0 { None } else { Some(T::decode(r)) }
        }
        fn encode(&self, out: &mut Vec<u8>) {
            match self { None => out.push(0), Some(v) => { out.push(1); v.encode(out); } }
        }
    }

    pub fn finish(out: Vec<u8>) -> u64 {
        let p = spark_udf_alloc(out.len() as u32);
        unsafe { std::slice::from_raw_parts_mut(p, out.len()) }.copy_from_slice(&out);
        ((p as u64) << 32) | (out.len() as u64)
    }
}
"#;

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
