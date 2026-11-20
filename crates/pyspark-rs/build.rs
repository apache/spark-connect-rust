//! The configuration of the interpreter that the extension is built for, as PyO3's own: a
//! free-threaded build of it, of the free-threaded ABI, is `cfg(Py_GIL_DISABLED)`, of which the
//! garbage collector sees no part of plans and expressions (see `src/gc.rs`).

fn main() {
    pyo3_build_config::use_pyo3_cfgs();
    println!("cargo:rerun-if-changed=build.rs");
}
