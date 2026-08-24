//! Tokio runtime management for synchronous PyO3 FFI.
//!
//! Provides a lazily-initialized shared multi-threaded `tokio::runtime::Runtime`
//! and a `block_on` helper so that synchronous Python code can call async
//! Rust functions at the PyO3 boundary.

use std::sync::{Once, OnceLock};
use tokio::runtime::Runtime;

static RUNTIME_INIT: Once = Once::new();
static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Get or initialize the shared tokio runtime.
///
/// Returns a reference to a multi-threaded runtime that is created once
/// and reused for all async operations.
pub fn get_runtime() -> &'static Runtime {
    RUNTIME_INIT.call_once(|| {
        let rt = Runtime::new().expect("Failed to create tokio runtime");
        let _ = RUNTIME.set(rt);
    });
    RUNTIME.get().expect("Runtime should be initialized")
}

/// Block on a future using the shared runtime.
///
/// Allows synchronous code (e.g., from PyO3) to execute an async function
/// and wait for its result. This should be called with the GIL released
/// in a Python context.
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    get_runtime().block_on(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_created_once() {
        let rt1 = get_runtime();
        let rt2 = get_runtime();
        // Both should point to the same runtime
        assert_eq!(rt1 as *const _, rt2 as *const _);
    }

    #[test]
    fn test_block_on_works() {
        let result = block_on(async { 42 });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_block_on_async_task() {
        let result = block_on(async {
            let x = 10;
            let y = 20;
            x + y
        });
        assert_eq!(result, 30);
    }
}
