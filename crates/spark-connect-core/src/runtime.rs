//! Tokio runtime management for synchronous PyO3 FFI.
//!
//! Provides a lazily-initialized shared multi-threaded `tokio::runtime::Runtime`
//! and a `block_on` helper so that synchronous Python code can call async
//! Rust functions at the PyO3 boundary.

use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Once, OnceLock};
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
///
/// The future is driven from the heap, so that the stack of the calling thread holds no copy of
/// it, however large it is.
pub fn block_on<F>(f: F) -> F::Output
where
    F: std::future::Future + Send,
    F::Output: Send,
{
    // The future of a request holds the state of its gRPC call and of its retries, tens of KiB,
    // and each frame that moves it on to the runtime (`drive`, the runtime's `block_on`, the
    // closure of `block_in_place`) keeps a copy of it on the calling thread, unless the compiler
    // inlines that frame, which depends on the crate that calls this one. A Python thread may
    // have a stack of 512 KiB. Moved from the heap, those frames hold only its pointer.
    let f = Box::pin(f);
    let watch = WAIT_WATCH.with(|watch| watch.borrow().clone());
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| drive(f, watch))
        }
        Ok(_) => block_on_other_thread(f),
        _ => drive(f, watch),
    }
}

/// What a wait for the server does at a poll of the signals of the process (see
/// [`with_wait_watch`]).
pub enum Watch {
    /// Keep waiting for the server.
    Continue,
    /// Interrupt the execution that the wait is for, by this future, which the wait drives to its
    /// end. The wait then ends with the server's cancellation of the execution.
    Interrupt(Pin<Box<dyn Future<Output = ()> + Send>>),
    /// Stop waiting, when no execution is in flight to interrupt. A wait of
    /// [`block_on_abortable`] ends at once, without the output of its future; any other wait
    /// keeps waiting, as for [`Watch::Continue`].
    Abort,
}

/// The poll of the signals of the process of a wait, which the wait calls every
/// [`POLL_INTERVAL`] on the thread that waits.
pub type WaitWatch = dyn Fn() -> Watch + Send + Sync;

/// The level of [`until_abort`] that a wait reaches once the first signal of the process
/// arrives while an execution of the wait is in flight: the retries, the backoffs and the RPCs
/// that have not been answered end at once, as PySpark's retry sleeps raise KeyboardInterrupt,
/// while the responses of a streaming execution are still read, so that the server's
/// cancellation of it ends them.
pub const ABORT_RETRIES: u8 = 1;

/// The level of [`until_abort`] that every wait reaches, when a signal arrives while no execution
/// of the wait is in flight, or a second signal arrives after the interrupt of the execution.
pub const ABORT_ALL: u8 = 2;

thread_local! {
    static WAIT_WATCH: RefCell<Option<Arc<WaitWatch>>> = const { RefCell::new(None) };
    /// The abort level of the wait that the calling thread runs, see [`until_abort`].
    static WAIT_ABORT: RefCell<Option<tokio::sync::watch::Receiver<u8>>> =
        const { RefCell::new(None) };
}

/// Runs the abort level of the wait of [`drive`] on this thread, until it drops.
struct AbortScope(Option<tokio::sync::watch::Receiver<u8>>);

impl AbortScope {
    fn enter(receiver: tokio::sync::watch::Receiver<u8>) -> Self {
        AbortScope(WAIT_ABORT.with(|abort| abort.replace(Some(receiver))))
    }
}

impl Drop for AbortScope {
    fn drop(&mut self) {
        let previous = self.0.take();
        WAIT_ABORT.with(|abort| *abort.borrow_mut() = previous);
    }
}

/// `f`, which a wait of the calling thread ends at once, dropping it unfinished, when the wait
/// reaches `level` (see [`ABORT_RETRIES`] and [`ABORT_ALL`]): None then, else the output of `f`.
/// Outside a wait that a watch polls, `f` is awaited as it is.
pub async fn until_abort<F: std::future::Future>(level: u8, f: F) -> Option<F::Output> {
    let receiver = WAIT_ABORT.with(|abort| abort.borrow().clone());
    let Some(mut receiver) = receiver else {
        return Some(f.await);
    };
    tokio::pin!(f);
    tokio::select! {
        biased;
        _ = async {
            // A wait whose sender is gone is not aborted by it.
            if receiver.wait_for(|reached| *reached >= level).await.is_err() {
                std::future::pending::<()>().await;
            }
        } => None,
        output = &mut f => Some(output),
    }
}

/// Run `f` on this thread, with `watch` called by the waits for the server that `f` makes on this
/// thread, as [`block_on`] makes them: the waits poll `watch` every [`POLL_INTERVAL`] while the
/// future of each waits. A wait on another thread (see [`block_on`]) does not poll it. The watch of
/// `f` ends with it, and the watch of an enclosing call is restored.
pub fn with_wait_watch<R>(watch: Arc<WaitWatch>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Arc<WaitWatch>>);

    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            WAIT_WATCH.with(|current| *current.borrow_mut() = previous);
        }
    }

    let previous = WAIT_WATCH.with(|current| current.replace(Some(watch)));
    let _restore = Restore(previous);
    f()
}

/// How often a wait for the server polls its [`WaitWatch`].
pub const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(200);

/// As [`block_on`], but the wait ends at once, with `None`, when the [`WaitWatch`] of the thread
/// returns [`Watch::Abort`]: the future is dropped, unfinished. For a wait that nothing on the
/// server stands behind, such as the retries of a session release, which a signal of the
/// process stops as it stops PySpark's sleep between the retries.
pub fn block_on_abortable<F>(f: F) -> Option<F::Output>
where
    F: std::future::Future + Send,
    F::Output: Send,
{
    // Driven from the heap, as `block_on` drives its future.
    let f = Box::pin(f);
    let watch = WAIT_WATCH.with(|watch| watch.borrow().clone());
    let drive = |f: Pin<Box<F>>| match watch {
        Some(watch) => get_runtime().block_on(watched_abortable(f, watch)),
        None => Some(get_runtime().block_on(f)),
    };
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| drive(f))
        }
        // The wait runs on another thread, which does not poll the watch (see `block_on`).
        Ok(_) => Some(block_on_other_thread(f)),
        _ => drive(f),
    }
}

/// `f`, polled as it is, and `watch` called every [`POLL_INTERVAL`]: `None` at once when it
/// returns [`Watch::Abort`], else the output of `f`. The first interrupt that `watch` returns is
/// spawned and awaited with `f`, as in [`watched`]. The polls go on after it, so that a second
/// signal ends the wait even while the interrupt hangs, and the running interrupt is cancelled
/// with the wait.
async fn watched_abortable<F: Future>(f: F, watch: Arc<WaitWatch>) -> Option<F::Output> {
    tokio::pin!(f);
    let mut ticks = tokio::time::interval(POLL_INTERVAL);
    let mut interrupt: Option<tokio::task::JoinHandle<()>> = None;
    loop {
        tokio::select! {
            output = &mut f => {
                // `f` has ended, so the wait only awaits the interrupt; a second signal, polled
                // meanwhile, ends that too.
                if let Some(mut interrupt) = interrupt {
                    loop {
                        tokio::select! {
                            _ = &mut interrupt => break,
                            _ = ticks.tick() => {
                                if let Watch::Abort = watch() {
                                    interrupt.abort();
                                    break;
                                }
                            }
                        }
                    }
                }
                return Some(output);
            }
            _ = ticks.tick() => match watch() {
                Watch::Abort => {
                    if let Some(interrupt) = interrupt {
                        interrupt.abort();
                    }
                    return None;
                }
                Watch::Interrupt(sent) => {
                    if interrupt.is_none() {
                        interrupt = Some(tokio::spawn(sent));
                    }
                }
                Watch::Continue => {}
            },
        }
    }
}

/// Drive `f` on the shared runtime, polling `watch` (if any) as the wait goes on.
fn drive<F: Future>(f: F, watch: Option<Arc<WaitWatch>>) -> F::Output {
    match watch {
        Some(watch) => {
            let (abort, receiver) = tokio::sync::watch::channel(0_u8);
            let _scope = AbortScope::enter(receiver);
            get_runtime().block_on(watched(f, watch, abort))
        }
        None => get_runtime().block_on(f),
    }
}

/// `f`, polled as it is, and `watch` called every [`POLL_INTERVAL`]. The first interrupt that it
/// returns is spawned, and the wait reaches [`ABORT_RETRIES`]: `f` is driven on, and the interrupt
/// is awaited with it, so that the cleanup of `f` runs as the server ends the execution. A wait
/// that `watch` aborts reaches [`ABORT_ALL`], which ends the waits of `f` that [`until_abort`]
/// covers, and the interrupt that is still running. The polls go on after an interrupt, so that
/// a second signal aborts the wait.
async fn watched<F: Future>(
    f: F,
    watch: Arc<WaitWatch>,
    abort: tokio::sync::watch::Sender<u8>,
) -> F::Output {
    tokio::pin!(f);
    let mut ticks = tokio::time::interval(POLL_INTERVAL);
    let mut interrupt: Option<tokio::task::JoinHandle<()>> = None;
    // Whether `watch` aborted the wait, which a second signal after an interrupt does.
    let mut aborted = false;
    loop {
        tokio::select! {
            output = &mut f => {
                // The interrupt is awaited, and a second signal, polled meanwhile, ends the wait
                // of it: the interrupt is then cancelled. When that signal already aborted the
                // wait, which ended `f`, the interrupt is cancelled at once, as PySpark stops at
                // the second signal while an interrupt hangs or is retried.
                if let Some(mut interrupt) = interrupt {
                    if aborted {
                        interrupt.abort();
                        return output;
                    }
                    loop {
                        tokio::select! {
                            _ = &mut interrupt => break,
                            _ = ticks.tick() => {
                                if let Watch::Abort = watch() {
                                    abort.send_replace(ABORT_ALL);
                                    interrupt.abort();
                                    break;
                                }
                            }
                        }
                    }
                }
                return output;
            }
            _ = ticks.tick() => match watch() {
                Watch::Continue => {}
                Watch::Interrupt(sent) => {
                    abort.send_replace(ABORT_RETRIES);
                    if interrupt.is_none() {
                        interrupt = Some(tokio::spawn(sent));
                    }
                }
                Watch::Abort => {
                    abort.send_replace(ABORT_ALL);
                    aborted = true;
                }
            },
        }
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

    /// A future of a large state, as the future of a request is of the state of its gRPC call,
    /// is driven from the heap: a thread of a small stack, which holds it once, holds no other
    /// copy of it in the frames that drive it, whatever the compiler inlines. (The frames of a
    /// debug build copy more, so its thread has the stack of a default thread of Linux.)
    #[test]
    fn test_block_on_holds_no_copy_of_a_large_future_on_a_small_stack() {
        const STATE: usize = 192 << 10;
        let stack = if cfg!(debug_assertions) {
            8 << 20
        } else {
            512 << 10
        };
        let output = std::thread::Builder::new()
            .stack_size(stack)
            .spawn(|| {
                block_on(async {
                    let state = [7u8; STATE];
                    tokio::task::yield_now().await;
                    state[STATE - 1]
                })
            })
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(output, 7);
    }

    #[test]
    fn test_watch_is_called_while_the_future_waits() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let watch: Arc<WaitWatch> = Arc::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Watch::Continue
        });
        let output = with_wait_watch(watch, || {
            block_on(async {
                tokio::time::sleep(POLL_INTERVAL * 5).await;
                7
            })
        });
        assert_eq!(output, 7);
        assert!(calls.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn test_interrupt_of_a_watch_is_driven_to_the_end_of_the_wait() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let fired = AtomicBool::new(false);
        let interrupted = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&interrupted);
        let watch: Arc<WaitWatch> = Arc::new(move || {
            if fired.swap(true, Ordering::SeqCst) {
                Watch::Continue
            } else {
                let flag = Arc::clone(&flag);
                Watch::Interrupt(Box::pin(async move { flag.store(true, Ordering::SeqCst) }))
            }
        });
        let output = with_wait_watch(watch, || {
            block_on(async {
                tokio::time::sleep(POLL_INTERVAL * 5).await;
                9
            })
        });
        assert_eq!(output, 9);
        assert!(interrupted.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_a_second_signal_ends_the_wait_without_awaiting_a_hung_interrupt() {
        // The first signal interrupts, and the interrupt never ends, as against a server that
        // does not answer it. The second aborts the wait, which ends its future, and the wait
        // returns at once, cancelling the interrupt, without the third signal that comes later.
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        struct Dropped(Arc<AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let polls = Arc::new(AtomicUsize::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let (counter, flag) = (Arc::clone(&polls), Arc::clone(&cancelled));
        let watch: Arc<WaitWatch> =
            Arc::new(move || match counter.fetch_add(1, Ordering::SeqCst) {
                0 => {
                    let dropped = Dropped(Arc::clone(&flag));
                    Watch::Interrupt(Box::pin(async move {
                        let _dropped = dropped;
                        std::future::pending::<()>().await
                    }))
                }
                1 => Watch::Abort,
                n if n >= 10 => Watch::Abort,
                _ => Watch::Continue,
            });
        let output = with_wait_watch(watch, || {
            block_on(until_abort(ABORT_ALL, std::future::pending::<u8>()))
        });
        assert_eq!(output, None);
        assert!(
            polls.load(Ordering::SeqCst) < 10,
            "the wait needed a third signal"
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !cancelled.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            cancelled.load(Ordering::SeqCst),
            "the interrupt was not cancelled"
        );
    }

    /// The output of `block_on_abortable(f)` under `watch`, run on a thread of its own, or None
    /// when the wait has not ended within 5 s: a wait that is not ended then fails the test,
    /// rather than hanging it.
    fn abortable_wait_under<F>(watch: Arc<WaitWatch>, f: F) -> Option<Option<F::Output>>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let output = with_wait_watch(watch, || block_on_abortable(f));
            let _ = sender.send(output);
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .ok()
    }

    #[test]
    fn test_a_second_signal_ends_an_abortable_wait_while_its_interrupt_hangs() {
        // As `watched`: the first signal interrupts, and the interrupt never ends. The second
        // signal aborts the wait, which ends at once without the output of its future, and the
        // interrupt is cancelled.
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        struct Dropped(Arc<AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let polls = Arc::new(AtomicUsize::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let (counter, flag) = (Arc::clone(&polls), Arc::clone(&cancelled));
        let watch: Arc<WaitWatch> =
            Arc::new(move || match counter.fetch_add(1, Ordering::SeqCst) {
                0 => {
                    let dropped = Dropped(Arc::clone(&flag));
                    Watch::Interrupt(Box::pin(async move {
                        let _dropped = dropped;
                        std::future::pending::<()>().await
                    }))
                }
                1 => Watch::Abort,
                _ => Watch::Continue,
            });
        let output = abortable_wait_under(watch, std::future::pending::<u8>());
        assert_eq!(
            output,
            Some(None),
            "the wait did not end at the second signal"
        );
        assert_eq!(polls.load(Ordering::SeqCst), 2);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !cancelled.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            cancelled.load(Ordering::SeqCst),
            "the interrupt was not cancelled"
        );
    }

    #[test]
    fn test_a_second_signal_ends_an_abortable_wait_while_its_future_waits_after_the_interrupt() {
        // The interrupt ends at once, and the future then waits, as a release that the server
        // does not answer: the first signal alone does not end the wait, the second one does.
        let polls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&polls);
        let watch: Arc<WaitWatch> =
            Arc::new(
                move || match counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) {
                    0 => Watch::Interrupt(Box::pin(async {})),
                    1 => Watch::Continue,
                    _ => Watch::Abort,
                },
            );
        let output = abortable_wait_under(watch, std::future::pending::<u8>());
        assert_eq!(
            output,
            Some(None),
            "the wait did not end at the second signal"
        );
        assert_eq!(polls.load(std::sync::atomic::Ordering::SeqCst), 3);

        // Without a second signal, the output of the future is still given after the interrupt.
        let watch: Arc<WaitWatch> = Arc::new({
            let first = std::sync::atomic::AtomicBool::new(true);
            move || {
                if first.swap(false, std::sync::atomic::Ordering::SeqCst) {
                    Watch::Interrupt(Box::pin(async {}))
                } else {
                    Watch::Continue
                }
            }
        });
        let ended = abortable_wait_under(watch, async {
            tokio::time::sleep(POLL_INTERVAL * 2).await;
            7_u8
        });
        assert_eq!(ended, Some(Some(7)));
    }

    #[test]
    fn test_an_abortable_wait_ends_when_its_watch_aborts_and_another_keeps_waiting() {
        // A watch that aborts ends an abortable wait at once, without the output of its future.
        let aborted = with_wait_watch(Arc::new(|| Watch::Abort), || {
            block_on_abortable(std::future::pending::<u8>())
        });
        assert_eq!(aborted, None);
        // Without an abort, an abortable wait gives the output of its future.
        let ended = with_wait_watch(Arc::new(|| Watch::Continue), || {
            block_on_abortable(async {
                tokio::time::sleep(POLL_INTERVAL * 2).await;
                7
            })
        });
        assert_eq!(ended, Some(7));
        assert_eq!(block_on_abortable(async { 8 }), Some(8));
        // Any other wait keeps waiting for its future, whatever the watch returns.
        let waited = with_wait_watch(Arc::new(|| Watch::Abort), || {
            block_on(async {
                tokio::time::sleep(POLL_INTERVAL * 2).await;
                9
            })
        });
        assert_eq!(waited, 9);
    }

    #[test]
    fn test_until_abort_ends_the_waits_that_the_watch_aborts_at_their_level() {
        // A wait of the watch's abort at ABORT_ALL ends at once, dropping its future unfinished.
        let aborted = with_wait_watch(Arc::new(|| Watch::Abort), || {
            block_on(until_abort(ABORT_ALL, std::future::pending::<u8>()))
        });
        assert_eq!(aborted, None);
        // An interrupt reaches ABORT_RETRIES only: the retries end, and the interrupt is awaited.
        let interrupted = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&interrupted);
        let retried = with_wait_watch(
            Arc::new(move || {
                let flag = Arc::clone(&flag);
                Watch::Interrupt(Box::pin(async move {
                    flag.store(true, std::sync::atomic::Ordering::SeqCst);
                }))
            }),
            || block_on(until_abort(ABORT_RETRIES, std::future::pending::<u8>())),
        );
        assert_eq!(retried, None);
        assert!(interrupted.load(std::sync::atomic::Ordering::SeqCst));
        // Outside a watch, a future is awaited as it is.
        assert_eq!(block_on(until_abort(ABORT_ALL, async { 8 })), Some(8));
    }

    #[test]
    fn test_with_wait_watch_restores_the_enclosing_watch() {
        let outer: Arc<WaitWatch> = Arc::new(|| Watch::Continue);
        let is_outer =
            |w: &Option<Arc<WaitWatch>>| w.as_ref().is_some_and(|w| Arc::ptr_eq(w, &outer));
        with_wait_watch(Arc::clone(&outer), || {
            with_wait_watch(Arc::new(|| Watch::Continue), || {});
            assert!(WAIT_WATCH.with(|w| is_outer(&w.borrow())));
        });
        assert!(WAIT_WATCH.with(|w| w.borrow().is_none()));
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
