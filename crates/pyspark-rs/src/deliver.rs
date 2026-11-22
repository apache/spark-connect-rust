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

//! The Python work of the threads of the transport.
//!
//! The threads of the Rust runtime, which make the connections of the channels and send the
//! requests that no caller waits for (the releases of the results of the executions, the removal
//! of the relation of a dropped checkpoint, the deletions of the ML cache, the interrupt of a
//! signal), are not threads of Python, and do not wait for the GIL: a thread of Python may hold
//! it while it waits for them, as a finalizer does. What PySpark does in Python from the threads
//! that send these requests, such as the warning of their spent retries, the logs of the
//! transport, and the methods of a retry policy, runs on the delivery thread instead, a thread of
//! its own that waits for the GIL, in the order it was given. From the exit of the interpreter on
//! (the `atexit` hook of the extension), nothing more runs there.
//!
//! From then on too, the exception of a failed call is not built on a thread other than the one
//! that exits, a daemon thread, which waits until the process ends instead (see [`python_work`]):
//! before 3.14, CPython's finalization ends a thread that waits for the GIL with `pthread_exit`,
//! whose unwind cannot pass the frames of the binding below the Python code that the binding runs,
//! and so aborts the process.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::OnceLock;

use pyo3::prelude::*;

/// A piece of Python work of a thread of the transport.
type Job = Box<dyn FnOnce(Python<'_>) + Send>;

/// The queue of the delivery thread of each generation of the process, which starts with its first
/// job: a child of a `fork()`, which has no thread of its parent, starts a delivery thread of its
/// own, and leaves the queue of its parent as it is (see `spark_connect_core::fork`). A job is
/// queued without a lock, which a thread of the parent could have held at the fork.
static QUEUE: spark_connect_core::fork::PerGeneration<OnceLock<Sender<Job>>> =
    spark_connect_core::fork::PerGeneration::new();

/// Whether the interpreter exits, after which nothing more is delivered.
static EXITING: AtomicBool = AtomicBool::new(false);

/// Whether the calling thread is a thread of Python, as that of a call of the client is, which
/// may attach to the interpreter: a thread of the runtime has no thread state of Python.
pub(crate) fn on_python_thread() -> bool {
    // SAFETY: reads the thread state that the interpreter keeps for this thread, without the
    // GIL, which it does not need.
    !unsafe { pyo3::ffi::PyGILState_GetThisThreadState() }.is_null()
}

/// Run `job` with the GIL: at once on a thread of Python, of the exception that is set on the
/// thread kept (see [`preserving_pending_exception`]), else on the delivery thread, later.
pub(crate) fn run(job: impl FnOnce(Python<'_>) + Send + 'static) {
    if on_python_thread() {
        // Not of an interpreter that finalizes (`try_attach`), as the other jobs of a thread of
        // Python: one there is not run.
        let _ = Python::try_attach(|py| preserving_pending_exception(py, job));
    } else {
        deliver(Box::new(job));
    }
}

/// Run `f` with the GIL of `py`, with the exception that is set on the thread, if any, set aside
/// before it and set again after it, as CPython keeps it around a finalizer (`slot_tp_finalize`
/// around a `__del__`, and the finalization of a generator): Python code that a native drop or
/// finalizer runs, or a call of the C API that clears or sets the indicator of the exception,
/// neither fails of the exception nor consumes or replaces it. A drop runs while an exception
/// propagates when CPython drops the values of a frame that it unwinds, which it does with the
/// exception set, as of a temporary DataFrame that holds the last reference to a fitted model
/// (`est.fit(df).transform(bad).columns`): of an exception that the drop consumed, CPython raises
/// `SystemError: error return without exception set` in its place, or crashes. Of CPython's own
/// fetch and restore, which take the exception as it is, where PyO3's `PyErr::take` would resume
/// the panic of a `PanicException`. An exception that `f` leaves set, which nothing raises, is
/// reported as an unraisable one, as CPython reports that of a `__del__`; the errors that `f`
/// returns are its own.
pub(crate) fn preserving_pending_exception<T>(
    py: Python<'_>,
    f: impl FnOnce(Python<'_>) -> T,
) -> T {
    let _pending = PendingException::take(py);
    f(py)
}

/// The exception that was set on the thread when [`preserving_pending_exception`] began, if any,
/// which is set again when this drops, also when its work unwinds of a panic: of the references of
/// its type, its value and its traceback, which `PyErr_Fetch` gave, or null pointers.
struct PendingException {
    kind: *mut pyo3::ffi::PyObject,
    value: *mut pyo3::ffi::PyObject,
    traceback: *mut pyo3::ffi::PyObject,
}

impl PendingException {
    /// Take the exception that is set on the thread of `_py`, which clears it.
    fn take(_py: Python<'_>) -> Self {
        let mut pending = PendingException {
            kind: std::ptr::null_mut(),
            value: std::ptr::null_mut(),
            traceback: std::ptr::null_mut(),
        };
        // SAFETY: with the GIL (`_py`), the fetch moves the references of the exception that is
        // set, if any, to the three pointers, and clears it.
        unsafe {
            pyo3::ffi::PyErr_Fetch(
                &mut pending.kind,
                &mut pending.value,
                &mut pending.traceback,
            )
        };
        pending
    }
}

impl Drop for PendingException {
    fn drop(&mut self) {
        // SAFETY: with the GIL, which the work of `preserving_pending_exception` holds as it
        // returns or unwinds: the report takes the exception that the work left set, if any, and
        // the restore takes back the references that the fetch gave, once.
        unsafe {
            if !pyo3::ffi::PyErr_Occurred().is_null() {
                pyo3::ffi::PyErr_WriteUnraisable(std::ptr::null_mut());
            }
            if !self.kind.is_null() {
                pyo3::ffi::PyErr_Restore(self.kind, self.value, self.traceback);
            }
        }
    }
}

/// Run `job`, Python of the client whose error is dropped, such as a log, as [`run`] runs it. A
/// KeyboardInterrupt of its error is kept (see [`keep_interrupt`]).
pub(crate) fn run_dropping_errors(job: impl FnOnce(Python<'_>) -> PyResult<()> + Send + 'static) {
    run(move |py| {
        if let Err(error) = job(py) {
            keep_interrupt(py, &error);
        }
    });
}

/// Keep the signal of `error`, an error that the client drops of Python code that it ran on the
/// main thread, if it is a KeyboardInterrupt: the handler of SIGINT, which the interpreter runs in
/// whatever Python code runs on the main thread when the signal comes, raised it there, in place
/// of the check of the signals of the wait of a call, which would not see the signal then. The
/// exception is kept for the thread, which the next check of the wait raises in place of the
/// signals' (see [`take_interrupt`]), once, of the first that comes, so that the handler runs once
/// for the signal, as PySpark's. Of another thread, which runs no handler of a signal, the error
/// is not of a signal, and is dropped. Whether the error is of a signal.
pub(crate) fn keep_interrupt(py: Python<'_>, error: &PyErr) -> bool {
    let of_a_signal =
        error.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) && on_main_thread(py);
    if of_a_signal {
        INTERRUPT.with(|kept| {
            let mut kept = kept.borrow_mut();
            if kept.is_none() {
                *kept = Some(error.clone_ref(py));
            }
        });
    }
    of_a_signal
}

/// The KeyboardInterrupt that [`keep_interrupt`] kept for this thread, if any, which a check of
/// the signals raises in place of theirs.
pub(crate) fn take_interrupt() -> Option<PyErr> {
    INTERRUPT.with(|kept| kept.borrow_mut().take())
}

/// Check the signals of the process, as `PyErr_CheckSignals`, which runs their handlers: the
/// error is the exception of a handler, or the KeyboardInterrupt that a handler raised in Python
/// code of the client on this thread, which the client kept (see [`keep_interrupt`]).
pub(crate) fn check_signals(py: Python<'_>) -> PyResult<()> {
    match take_interrupt() {
        Some(interrupt) => Err(interrupt),
        None => py.check_signals(),
    }
}

thread_local! {
    /// The KeyboardInterrupt of a signal that Python code of the client raised on this thread,
    /// which the client dropped (see [`keep_interrupt`]).
    static INTERRUPT: std::cell::RefCell<Option<PyErr>> = const { std::cell::RefCell::new(None) };
}

/// Whether the calling thread is the main thread of Python, which runs the handlers of the
/// signals: `threading.main_thread()`, false if it cannot be told.
fn on_main_thread(py: Python<'_>) -> bool {
    (|| -> PyResult<bool> {
        let threading = py.import("threading")?;
        let main = threading.call_method0("main_thread")?;
        Ok(main.is(&threading.call_method0("current_thread")?))
    })()
    .unwrap_or(false)
}

/// Run `job` with the GIL as [`run`] runs it, but of a thread of the runtime on a thread of its
/// own rather than on the delivery thread: Python code that may wait forever, which would hold
/// every later job of the delivery thread, where the code of grpcio's client that it stands for
/// holds only its own call (see `may_wait_forever` of the client). Of no order with the jobs of
/// the delivery thread.
pub(crate) fn run_apart(job: impl FnOnce(Python<'_>) + Send + 'static) {
    if on_python_thread() {
        // Not of an interpreter that finalizes, as `run`.
        let _ = Python::try_attach(|py| preserving_pending_exception(py, job));
        return;
    }
    if EXITING.load(Ordering::SeqCst) {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("pyspark-apart".to_string())
        .spawn(move || {
            if !EXITING.load(Ordering::SeqCst) {
                // A job that cannot attach, of an interpreter that is gone, is dropped.
                let _ = Python::try_attach(job);
            }
        });
    // Without the thread, the job is dropped.
    drop(spawned);
}

/// As [`call`], on a thread of its own as [`run_apart`] runs it, once the delivery thread ran the
/// jobs that it was given before, which a call gives it before this one, such as the logs of its
/// retries, which then precede it as they would on the delivery thread.
pub(crate) async fn call_apart<T: Send + 'static>(
    job: impl FnOnce(Python<'_>) -> T + Send + 'static,
) -> Option<T> {
    if on_python_thread() {
        return Python::try_attach(|py| preserving_pending_exception(py, job));
    }
    call(|_| ()).await;
    let (sender, receiver) = tokio::sync::oneshot::channel();
    run_apart(move |py| {
        let _ = sender.send(job(py));
    });
    receiver.await.ok()
}

/// `job` with the GIL, run as [`run`] runs it, awaited: None once the interpreter exits, or if
/// the job is dropped unrun.
pub(crate) async fn call<T: Send + 'static>(
    job: impl FnOnce(Python<'_>) -> T + Send + 'static,
) -> Option<T> {
    if on_python_thread() {
        return Python::try_attach(|py| preserving_pending_exception(py, job));
    }
    let (sender, receiver) = tokio::sync::oneshot::channel();
    deliver(Box::new(move |py| {
        let _ = sender.send(job(py));
    }));
    receiver.await.ok()
}

/// Queue `job` for the delivery thread, which starts with the first one.
fn deliver(job: Job) {
    if EXITING.load(Ordering::SeqCst) {
        return;
    }
    let queue = QUEUE.get_or_init(|_| OnceLock::new()).get_or_init(|| {
        let (sender, receiver) = channel::<Job>();
        let spawned = std::thread::Builder::new()
            .name("pyspark-delivery".to_string())
            .spawn(move || {
                for job in receiver {
                    // The `atexit` hook runs before the interpreter finalizes, after which a thread
                    // that waits for the GIL may be ended by it.
                    if EXITING.load(Ordering::SeqCst) {
                        continue;
                    }
                    // A job that cannot attach, of an interpreter that is gone, is dropped.
                    let _ = Python::try_attach(job);
                }
            });
        // Without the thread, the queue has no receiver, and drops the jobs.
        drop(spawned);
        sender
    });
    // A queue of no thread drops the job.
    let _ = queue.send(job);
}

/// Stop the deliveries, as the interpreter exits: the `atexit` hook of the extension, which the
/// thread that exits runs, once the threads that are not daemons have ended. From then on, the
/// Python work of the binding does not start on the other threads ([`python_work`]), and the work
/// that runs there ends first, with the GIL, which this releases meanwhile, so that none is under
/// way when the interpreter finalizes: within a bound, for work that waits for something that the
/// exit holds.
#[pyfunction]
pub(crate) fn _stop_deliveries(py: Python<'_>) {
    let _ = EXIT_THREAD.try_with(|exits| exits.set(true));
    EXITING.store(true, Ordering::SeqCst);
    if PYTHON_WORK.load(Ordering::SeqCst) == 0 {
        return;
    }
    crate::gc::detach_without_parts(py, || {
        let deadline = std::time::Instant::now() + EXIT_WAIT;
        while PYTHON_WORK.load(Ordering::SeqCst) != 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });
}

/// The Python work of the binding that runs now (see [`python_work`]).
static PYTHON_WORK: AtomicUsize = AtomicUsize::new(0);

/// The bound of the wait of the exit for the Python work that runs (see [`_stop_deliveries`]).
const EXIT_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

thread_local! {
    /// Whether this thread exits the interpreter: it ran the `atexit` hook of the extension.
    static EXIT_THREAD: Cell<bool> = const { Cell::new(false) };
    /// The Python work of the binding that runs on this thread, one within another (see
    /// [`python_work`]).
    static WORK_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Python work that the binding runs on this thread with the GIL, such as the build of the
/// exception of a failed call, which imports modules and runs Python code: counted while it runs,
/// so that the exit waits for it ([`_stop_deliveries`]). Once the exit has begun, a thread other
/// than the one that exits, a daemon thread, does not start it: it waits until the process ends,
/// without the GIL, as CPython 3.14 holds such a thread once it finalizes. Before 3.14, the
/// finalization ends a thread that waits for the GIL with `pthread_exit`, whose unwind cannot pass
/// the frames of the binding below the Python code that it ran, and so aborts the process; pyo3
/// holds the thread instead only where it takes the GIL itself. Work within work that runs goes
/// on, as the exit waits for the outer one. One count and one read otherwise.
pub(crate) fn python_work() -> PythonWork {
    let outer = WORK_DEPTH
        .try_with(|depth| depth.replace(depth.get() + 1) == 0)
        .unwrap_or(true);
    PYTHON_WORK.fetch_add(1, Ordering::SeqCst);
    let work = PythonWork(());
    if outer && EXITING.load(Ordering::SeqCst) && !EXIT_THREAD.try_with(Cell::get).unwrap_or(false)
    {
        drop(work);
        hold_until_the_process_ends();
    }
    work
}

/// Python work of the binding that runs, counted while it lives ([`python_work`]).
pub(crate) struct PythonWork(());

impl Drop for PythonWork {
    fn drop(&mut self) {
        PYTHON_WORK.fetch_sub(1, Ordering::SeqCst);
        let _ = WORK_DEPTH.try_with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Hold this thread until the process ends, without the GIL: it releases the GIL if it holds it,
/// and holds none of an interpreter that the GIL cannot be taken of any more.
fn hold_until_the_process_ends() -> ! {
    let _ = Python::try_attach(|py| {
        crate::gc::detach_without_parts(py, || loop {
            std::thread::park();
        })
    });
    loop {
        std::thread::park();
    }
}

/// Register [`_stop_deliveries`] with `atexit`, as the extension loads.
pub(crate) fn stop_at_exit(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let stop = module.getattr("_stop_deliveries")?;
    module
        .py()
        .import("atexit")?
        .call_method1("register", (stop,))?;
    Ok(())
}
