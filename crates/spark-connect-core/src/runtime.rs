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
/// The synchronous API may be called from any thread, including inside a Tokio runtime of
/// either flavor: the call blocks the calling thread until the future completes. Inside a
/// multi-thread runtime the thread is blocked in place; inside a `current_thread` runtime it
/// cannot block in place, so the future runs on a thread of its own, which the calling thread
/// waits for (see [`block_on_other_thread`]).
///
/// Also called from a callback that runs inside the runtime (a progress handler, or a
/// session hook of the executions), which may run a query itself: as PySpark does, the
/// nested query runs. The thread is then in the runtime's context, which `block_in_place`
/// leaves for the nested `block_on`.
pub fn block_on<F>(f: F) -> F::Output
where
    F: std::future::Future + Send,
    F::Output: Send,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| get_runtime().block_on(f))
        }
        Ok(_) => block_on_other_thread(f),
        _ => get_runtime().block_on(f),
    }
}

/// Run a future to completion on the shared runtime, on a thread of its own, while the calling
/// thread waits for it: the calling thread is inside a `current_thread` runtime, which cannot
/// block in place and cannot start another runtime.
///
/// The future is polled on the other thread, so the state that the client keeps per thread is
/// that of the caller: [`calling_thread_id`] is the caller's ID there, and the callback depth
/// and the recursion failure of the caller carry over, then back, so a call made from inside a
/// callback bounds its recursion as on the caller's own thread.
fn block_on_other_thread<F>(f: F) -> F::Output
where
    F: std::future::Future + Send,
    F::Output: Send,
{
    let caller = std::thread::current().id();
    let depth = CALLBACK_DEPTH.with(std::cell::Cell::get);
    let recursion_failed = CALLBACK_RECURSION_FAILED.with(std::cell::Cell::get);
    let (output, recursion_failed) = std::thread::scope(|scope| {
        let worker = scope.spawn(move || {
            CALLER_THREAD.with(|caller_thread| caller_thread.set(Some(caller)));
            CALLBACK_DEPTH.with(|d| d.set(depth));
            CALLBACK_RECURSION_FAILED.with(|failed| failed.set(recursion_failed));
            let output = get_runtime().block_on(f);
            (output, CALLBACK_RECURSION_FAILED.with(std::cell::Cell::get))
        });
        match worker.join() {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    });
    CALLBACK_RECURSION_FAILED.with(|failed| failed.set(recursion_failed));
    output
}

thread_local! {
    /// The thread that the calls of this thread act for, while [`block_on_other_thread`] runs a
    /// future on this thread for it.
    static CALLER_THREAD: std::cell::Cell<Option<std::thread::ThreadId>> = const { std::cell::Cell::new(None) };
}

/// The thread whose per-thread state the calls on this thread use: the thread that calls the
/// synchronous API, also while a future of such a call is polled on another thread (see
/// [`block_on`]). Its operation tags and user context extensions are keyed by it.
pub fn calling_thread_id() -> std::thread::ThreadId {
    CALLER_THREAD
        .with(std::cell::Cell::get)
        .unwrap_or_else(|| std::thread::current().id())
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

    fn current_thread_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn test_block_on_inside_current_thread_runtime() {
        // The default flavor of `#[tokio::test]`: the sync API blocks the thread, which the
        // runtime cannot block in place.
        let result = current_thread_runtime().block_on(async { block_on(async { 42 }) });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_block_on_nested_inside_current_thread_runtime() {
        // A callback of the execution runs a query, which runs inside the current_thread runtime.
        let result = current_thread_runtime()
            .block_on(async { block_on(async { block_on(async { 7 }) + 1 }) });
        assert_eq!(result, 8);
    }

    #[test]
    fn test_block_on_inside_current_thread_runtime_keeps_caller_thread() {
        // The per-thread state (tags and extensions) of the caller is read as on its own thread.
        let caller = std::thread::current().id();
        let seen =
            current_thread_runtime().block_on(async { block_on(async { calling_thread_id() }) });
        assert_eq!(seen, caller);
        assert_eq!(calling_thread_id(), caller);
    }

    #[test]
    fn test_block_on_inside_current_thread_runtime_keeps_callback_depth() {
        // A call from a callback at the bound fails with a RecursionError on the other thread too.
        let guards: Vec<CallbackGuard> = (0..MAX_CALLBACK_DEPTH)
            .map(|_| CallbackGuard::enter())
            .collect();
        let err = current_thread_runtime()
            .block_on(async { block_on(async { check_callback_depth() }) })
            .unwrap_err();
        assert_eq!(err.kind, crate::error::SparkErrorKind::RecursionError);
        drop(guards);
        // The failure is released with the outermost callback, as on this thread.
        assert!(current_thread_runtime()
            .block_on(async { block_on(async { check_callback_depth() }) })
            .is_ok());
    }

    #[test]
    fn test_block_on_inside_current_thread_runtime_propagates_panic() {
        let result = std::panic::catch_unwind(|| {
            current_thread_runtime().block_on(async {
                block_on(async { panic!("from the future") });
            })
        });
        assert!(result.is_err());
    }
}
