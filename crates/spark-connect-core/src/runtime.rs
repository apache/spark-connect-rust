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
///
/// Also called from a callback that runs inside the runtime (a progress handler, or a
/// session hook of the executions), which may run a query itself: as PySpark does, the
/// nested query runs. The thread is then in the runtime's context, which `block_in_place`
/// leaves for the nested `block_on`. A `current_thread` runtime cannot block in place, so the
/// call keeps failing there.
pub fn block_on<F: std::future::Future>(f: F) -> F::Output {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| get_runtime().block_on(f))
        }
        _ => get_runtime().block_on(f),
    }
}

/// How many progress handlers and hooks may run nested on a thread, each one running a query
/// that runs the next one. Past it, an execution fails with a `RecursionError` (as PySpark's
/// recursion limit raises it) instead of recursing until the native stack of the thread
/// overflows. A level takes about 100 KiB of native stack (Python frames of the handler, and
/// the calls of PyO3 between them): on a thread of 512 KiB, four levels crash, so two levels
/// leave room for the frames of the thread itself.
pub const MAX_CALLBACK_DEPTH: usize = 2;

thread_local! {
    static CALLBACK_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Set once an execution failed past the bound, until the outermost callback returns: the
    /// callbacks of the levels above then fail their queries at once, rather than each level
    /// running a query per progress update again, which would grow exponentially.
    static CALLBACK_RECURSION_FAILED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A progress handler or a hook that runs on this thread, for as long as it lives.
pub struct CallbackGuard(());

impl CallbackGuard {
    /// Enter a callback that may run a query of its own.
    pub fn enter() -> Self {
        CALLBACK_DEPTH.with(|depth| depth.set(depth.get() + 1));
        CallbackGuard(())
    }
}

impl Drop for CallbackGuard {
    fn drop(&mut self) {
        let depth = CALLBACK_DEPTH.with(|depth| {
            depth.set(depth.get() - 1);
            depth.get()
        });
        if depth == 0 {
            CALLBACK_RECURSION_FAILED.with(|failed| failed.set(false));
        }
    }
}

/// Fail with a `RecursionError` when the executions started from this thread are nested past
/// [`MAX_CALLBACK_DEPTH`] callbacks, instead of the stack of the thread running out.
#[allow(clippy::result_large_err)] // SparkError is large, as in the other entry points
pub fn check_callback_depth() -> crate::error::Result<()> {
    let too_deep = CALLBACK_DEPTH.with(|depth| depth.get()) >= MAX_CALLBACK_DEPTH;
    if too_deep || CALLBACK_RECURSION_FAILED.with(|failed| failed.get()) {
        CALLBACK_RECURSION_FAILED.with(|failed| failed.set(true));
        return Err(crate::error::SparkError::recursion_msg(
            "maximum recursion depth exceeded in a progress handler or a hook of an execution",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_callback_depth_is_bounded_and_released() {
        assert!(check_callback_depth().is_ok());
        let guards: Vec<CallbackGuard> = (0..MAX_CALLBACK_DEPTH)
            .map(|_| CallbackGuard::enter())
            .collect();
        let err = check_callback_depth().unwrap_err();
        assert_eq!(err.kind, crate::error::SparkErrorKind::RecursionError);
        drop(guards);
        assert!(check_callback_depth().is_ok());
    }

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
    fn test_block_on_nested_in_block_on() {
        // A callback of a running execution blocks on a nested query, from within the runtime.
        let result = block_on(async { block_on(async { 7 }) + 1 });
        assert_eq!(result, 8);
    }

    #[test]
    fn test_block_on_nested_on_a_worker_thread() {
        let result =
            block_on(async { tokio::spawn(async { block_on(async { 5 }) }).await.unwrap() });
        assert_eq!(result, 5);
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
