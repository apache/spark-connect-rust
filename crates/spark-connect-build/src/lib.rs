//! `build.rs` helper for Rust UDFs compiled to WebAssembly.
//!
//! Call [`embed_wasm_udf`] from a crate's `build.rs` to compile a Rust source
//! file containing `#[spark_wasm_udf]` functions to a `wasm32` module and expose
//! it to the crate as the `WASM_UDFS_MODULE` env var (for `include_bytes!`).
//!
//! # Requirements (build machine)
//!
//! * the `wasm32-unknown-unknown` target
//!   (`rustup target add wasm32-unknown-unknown`), or `wasm32-wasip1` for
//!   [`embed_wasm_udf_for`] with [`WasmTarget::Wasip1`], and
//! * `apache-spark-connect-macros` listed as a **build-dependency** of the
//!   calling crate, so its proc-macro is compiled before `build.rs` runs and
//!   this helper can find its dylib.
//!
//! # Example `build.rs`
//!
//! ```ignore
//! spark_connect_build::embed_wasm_udf("src/main.rs");
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
    embed_wasm_udf_for(src_file, WasmTarget::Unknown)
}

/// The WebAssembly target a UDF module is compiled for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum WasmTarget {
    /// `wasm32-unknown-unknown`: the module imports nothing from its host, so
    /// `std::env::var` always fails and `std::time` and `std::fs` are unavailable.
    #[default]
    Unknown,
    /// `wasm32-wasip1`: the runner gives the module, through WASI, a snapshot of the
    /// environment of the Python worker that runs it, taken when the module is
    /// instantiated: the `os.environ` a Python UDF sees, including the session's
    /// `spark.pythonWorkerEnv.*` variables, without the names Spark reserves for its
    /// workers (`SPARK_*`, `PYSPARK_*`, `PYTHON_WORKER_FACTORY_*` and a few others), so
    /// `std::env::var` reads it.
    ///
    /// WASI also gives the module clocks, randomness and sleep, so a UDF that reads the
    /// time or random numbers should be marked nondeterministic. It gets no files or
    /// arguments, and what it writes to standard output or error is discarded.
    Wasip1,
}

impl WasmTarget {
    fn triple(self) -> &'static str {
        match self {
            WasmTarget::Unknown => "wasm32-unknown-unknown",
            WasmTarget::Wasip1 => "wasm32-wasip1",
        }
    }
}

/// Like [`embed_wasm_udf`], but compile `src_file` for `target`: pick
/// [`WasmTarget::Wasip1`] for UDFs that read environment variables. Emits the same
/// `WASM_UDFS_MODULE` env var and returns the module path.
pub fn embed_wasm_udf_for(src_file: impl AsRef<Path>, target: WasmTarget) -> PathBuf {
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

    let macro_dirs = proc_macro_dirs(&out_dir, "apache-spark-connect-macros");
    let macro_dylib = find_proc_macro(&macro_dirs, "spark_connect_macros").unwrap_or_else(|| {
        panic!(
            "could not find the spark_connect_macros proc-macro dylib in {}. \
             Add `apache-spark-connect-macros` to [build-dependencies].",
            macro_dirs
                .iter()
                .map(|dir| dir.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    });
    // The directory of the dylib: rustc loads a proc-macro crate as it is, without the
    // dependencies that it was built with, so it needs no other `-L dependency=` directory.
    let deps_dir = macro_dylib
        .parent()
        .expect("a file has a parent directory")
        .to_path_buf();

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
        .args(["--target", target.triple()])
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
        let triple = target.triple();
        eprintln!(
            "cargo:warning=The `{triple}` Rust target is required to build this WASM UDF crate."
        );
        eprintln!("cargo:warning=Install it with: rustup target add {triple}");
        eprintln!();
        panic!("wasm32 build of {} failed", src_abs.display());
    }

    println!("cargo:rustc-env=WASM_UDFS_MODULE={}", wasm_out.display());
    println!("cargo:rerun-if-changed={}", src_abs.display());
    wasm_out
}

/// The runtime prepended to the wasm build: the exported allocator, the
/// length-prefixed binary codec (`Abi`) the macro's per-row export wrappers call,
/// and the Arrow-buffer codec (`batch`) of its batch exports. Must stay
/// byte-compatible with `spark_connect::wasm_udf::AbiType` and the Python
/// runners in the embedded packer (`spark-connect/src/wasm_packer.py`).
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

    thread_local! {
        static SCRATCH: std::cell::RefCell<Vec<u8>> = std::cell::RefCell::new(Vec::new());
        static KEEP: std::cell::RefCell<Vec<Vec<u8>>> = std::cell::RefCell::new(Vec::new());
    }

    // Per-row exports are `spark_udf_<fn>`, so batch names use another prefix to never collide.
    #[no_mangle]
    pub extern "C" fn spark_batch_scratch(len: u32) -> u32 {
        SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            if s.len() < len as usize {
                s.resize(len as usize, 0);
            }
            s.as_mut_ptr() as u32
        })
    }

    // Batch ABI of the `spark_batch_udf_<fn>` exports, little-endian u32 words. Input: [rows,
    // columns, then per column: validity ptr (0: no nulls), values ptr, values len, offsets ptr
    // (0: fixed width), offsets len]. Output, valid until the next call: [status (0 ok, 1 error),
    // null count, validity ptr, validity len, values ptr, values len, offsets ptr, offsets len],
    // with the error message at (validity ptr, validity len).
    pub mod batch {
        use super::KEEP;

        pub struct Col {
            validity: Option<&'static [u8]>,
            values: &'static [u8],
            offsets: Option<&'static [u8]>,
        }

        #[inline]
        fn bit(bytes: &[u8], i: usize) -> bool {
            (bytes[i >> 3] >> (i & 7)) & 1 == 1
        }

        impl Col {
            #[inline]
            fn is_valid(&self, i: usize) -> bool {
                self.validity.map_or(true, |v| bit(v, i))
            }
            #[inline]
            fn offset(&self, i: usize) -> usize {
                let o = self.offsets.expect("variable-width column without offsets");
                i32::from_le_bytes(o[i * 4..i * 4 + 4].try_into().unwrap()) as usize
            }
            #[inline]
            fn bytes(&self, i: usize) -> &'static [u8] {
                &self.values[self.offset(i)..self.offset(i + 1)]
            }
        }

        pub trait FromCol: Sized {
            fn get_valid(col: &Col, i: usize) -> Self;
            // None: a null in an argument that isn't an Option.
            #[inline]
            fn get(col: &Col, i: usize) -> Option<Self> {
                if col.is_valid(i) { Some(Self::get_valid(col, i)) } else { None }
            }
        }

        macro_rules! prim_in { ($t:ty, $n:expr) => {
            impl FromCol for $t {
                #[inline]
                fn get_valid(col: &Col, i: usize) -> Self {
                    <$t>::from_le_bytes(col.values[i * $n..i * $n + $n].try_into().unwrap())
                }
            }
        }}
        prim_in!(i32, 4); prim_in!(i64, 8); prim_in!(f32, 4); prim_in!(f64, 8);

        impl FromCol for bool {
            #[inline]
            fn get_valid(col: &Col, i: usize) -> Self { bit(col.values, i) }
        }
        impl FromCol for String {
            #[inline]
            fn get_valid(col: &Col, i: usize) -> Self {
                String::from_utf8(col.bytes(i).to_vec()).unwrap()
            }
        }
        impl FromCol for Vec<u8> {
            #[inline]
            fn get_valid(col: &Col, i: usize) -> Self { col.bytes(i).to_vec() }
        }
        impl<T: FromCol> FromCol for Option<T> {
            #[inline]
            fn get_valid(col: &Col, i: usize) -> Self { Some(T::get_valid(col, i)) }
            #[inline]
            fn get(col: &Col, i: usize) -> Option<Self> {
                Some(if col.is_valid(i) { Some(T::get_valid(col, i)) } else { None })
            }
        }

        pub struct Out {
            n: usize,
            row: usize,
            validity: Vec<u8>,
            null_count: u32,
            values: Vec<u8>,
            offsets: Vec<u8>,
        }

        impl Out {
            fn new(n: usize) -> Self {
                Out {
                    n,
                    row: 0,
                    validity: vec![0xFF; (n + 7) / 8],
                    null_count: 0,
                    values: Vec::new(),
                    offsets: Vec::new(),
                }
            }
            #[inline]
            fn set_null(&mut self) {
                let r = self.row;
                self.validity[r >> 3] &= !(1u8 << (r & 7));
                self.null_count += 1;
            }
            #[inline]
            fn start_var(&mut self) {
                if self.offsets.is_empty() {
                    self.offsets.reserve(4 * (self.n + 1));
                    self.offsets.extend_from_slice(&0i32.to_le_bytes());
                }
            }
            #[inline]
            fn end_var(&mut self) {
                // Lossless: a wasm32 Vec holds at most i32::MAX bytes.
                let end = self.values.len() as i32;
                self.offsets.extend_from_slice(&end.to_le_bytes());
            }
            #[inline]
            fn bits(&mut self) {
                if self.values.is_empty() {
                    self.values = vec![0u8; (self.n + 7) / 8];
                }
            }
            fn finish(self) -> u32 {
                let nulls = self.null_count;
                let (vp, vl) = if nulls == 0 { (0, 0) } else { keep(self.validity) };
                let (xp, xl) = keep(self.values);
                let (op, ol) = keep(self.offsets);
                descriptor(&[0, nulls, vp, vl, xp, xl, op, ol])
            }
        }

        pub trait ToCol: Default {
            fn push(&self, out: &mut Out);
            #[inline]
            fn push_null(out: &mut Out) {
                out.set_null();
                Self::default().push(out);
            }
        }

        macro_rules! prim_out { ($t:ty) => {
            impl ToCol for $t {
                #[inline]
                fn push(&self, out: &mut Out) {
                    out.values.extend_from_slice(&self.to_le_bytes());
                    out.row += 1;
                }
            }
        }}
        prim_out!(i32); prim_out!(i64); prim_out!(f32); prim_out!(f64);

        impl ToCol for bool {
            #[inline]
            fn push(&self, out: &mut Out) {
                out.bits();
                if *self {
                    let r = out.row;
                    out.values[r >> 3] |= 1u8 << (r & 7);
                }
                out.row += 1;
            }
        }
        impl ToCol for String {
            #[inline]
            fn push(&self, out: &mut Out) { push_bytes(out, self.as_bytes()) }
        }
        impl ToCol for Vec<u8> {
            #[inline]
            fn push(&self, out: &mut Out) { push_bytes(out, self) }
        }
        impl<T: ToCol> ToCol for Option<T> {
            #[inline]
            fn push(&self, out: &mut Out) {
                match self {
                    Some(v) => v.push(out),
                    None => T::push_null(out),
                }
            }
            #[inline]
            fn push_null(out: &mut Out) { T::push_null(out) }
        }

        #[inline]
        fn push_bytes(out: &mut Out, b: &[u8]) {
            out.start_var();
            out.values.extend_from_slice(b);
            out.end_var();
            out.row += 1;
        }

        fn keep(b: Vec<u8>) -> (u32, u32) {
            if b.is_empty() {
                return (0, 0);
            }
            let (p, l) = (b.as_ptr() as u32, b.len() as u32);
            KEEP.with(|k| k.borrow_mut().push(b));
            (p, l)
        }

        fn descriptor(words: &[u32]) -> u32 {
            let mut d = Vec::with_capacity(words.len() * 4);
            for w in words {
                d.extend_from_slice(&w.to_le_bytes());
            }
            keep(d).0
        }

        fn error(msg: String) -> u32 {
            let (mp, ml) = keep(msg.into_bytes());
            descriptor(&[1, 0, mp, ml, 0, 0, 0, 0])
        }

        fn raw(p: usize, l: usize) -> &'static [u8] {
            if l == 0 { &[] } else { unsafe { core::slice::from_raw_parts(p as *const u8, l) } }
        }

        fn words(p: usize, n: usize) -> impl Fn(usize) -> usize {
            let b = raw(p, 4 * n);
            move |k| u32::from_le_bytes(b[4 * k..4 * k + 4].try_into().unwrap()) as usize
        }

        // `f` evaluates row `i`; Err(arg) means a null in argument `arg`, which isn't an Option.
        pub fn run<F>(desc: u32, args: &[&str], mut f: F) -> u32
        where
            F: FnMut(&[Col], usize, &mut Out) -> Result<(), usize>,
        {
            KEEP.with(|k| k.borrow_mut().clear());
            let head = words(desc as usize, 2);
            let (n, ncols) = (head(0), head(1));
            if ncols != args.len() {
                return error(format!(
                    "the batch has {ncols} columns, but the function takes {}",
                    args.len()
                ));
            }
            let w = words(desc as usize + 8, 5 * ncols);
            let cols: Vec<Col> = (0..ncols)
                .map(|a| {
                    let b = 5 * a;
                    Col {
                        validity: if w(b) == 0 { None } else { Some(raw(w(b), (n + 7) / 8)) },
                        values: raw(w(b + 1), w(b + 2)),
                        offsets: if w(b + 3) == 0 { None } else { Some(raw(w(b + 3), w(b + 4))) },
                    }
                })
                .collect();
            let mut out = Out::new(n);
            for i in 0..n {
                if let Err(a) = f(&cols, i, &mut out) {
                    return error(format!(
                        "argument `{}` is null in row {i} of the batch, but its type is not an \
                         Option; declare it as Option<T> to accept nulls",
                        args[a]
                    ));
                }
            }
            out.finish()
        }
    }
}
"#;

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("`{key}` is not set (call from a build script)"))
}

/// The directories where cargo puts the dylib of the proc-macro package `package`, a
/// build-dependency, for the build script whose `OUT_DIR` is `out_dir`, in either layout of
/// cargo's build directory:
///
/// * before Rust 1.100, `OUT_DIR` is `<profile>/build/<crate>-<hash>/out`, and the artifacts
///   of all the dependencies are in `<profile>/deps`;
/// * from Rust 1.100, `OUT_DIR` is `<profile>/build/<package>/<hash>/out`, and the artifacts of
///   each unit are in the `out` directory of that unit, `<profile>/build/<package>/<hash>/out`.
fn proc_macro_dirs(out_dir: &Path, package: &str) -> Vec<PathBuf> {
    // `<profile>` in the layout before Rust 1.100, and `<profile>/build` in the later one.
    let Some(base) = out_dir.ancestors().nth(3) else {
        return Vec::new();
    };
    let mut dirs = vec![base.join("deps")];
    if let Ok(units) = std::fs::read_dir(base.join(package)) {
        let mut units: Vec<PathBuf> = units
            .flatten()
            .map(|unit| unit.path().join("out"))
            .collect();
        units.sort();
        dirs.extend(units);
    }
    dirs
}

/// Find `lib<name>-<hash>.{dylib,so}`, or `<name>-<hash>.dll` on Windows, in `dirs`: the newest
/// one, as a rebuild leaves the stale ones.
fn find_proc_macro(dirs: &[PathBuf], name: &str) -> Option<PathBuf> {
    let prefixes = [format!("lib{name}-"), format!("{name}-")];
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let fname = match path.file_name() {
                Some(f) => f.to_string_lossy().into_owned(),
                None => continue,
            };
            let is_dylib =
                fname.ends_with(".dylib") || fname.ends_with(".so") || fname.ends_with(".dll");
            if is_dylib && prefixes.iter().any(|prefix| fname.starts_with(prefix)) {
                if let Ok(mtime) = entry.metadata().and_then(|m| m.modified()) {
                    if best.as_ref().map(|(t, _)| mtime > *t).unwrap_or(true) {
                        best = Some((mtime, path));
                    }
                }
            }
        }
    }
    best.map(|(_, p)| p)
}
