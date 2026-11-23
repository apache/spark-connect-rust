//! What Python's garbage collector may see of the objects of the binding.
//!
//! A Python object of PySpark holds its parts as Python objects, which the collector sees through
//! each holder: a DataFrame holds its plan object, which holds the plan objects of its inputs and
//! the functions of its UDFs, and a DataFrame derived from it holds a plan object that holds the
//! same plan object as its input. An object of the binding holds a value of the core instead,
//! whose plan and expressions are `Arc`s of nodes that other values share (see
//! `spark_connect::gc`), of which some hold Python objects: the functions of the UDFs, in their
//! picklers, and the plans of the Python client, the exceptions that they raised and the
//! Observations, as the owners of relations, of errors and of Observations.
//!
//! The collector sees those parts as PySpark's are seen: a part of the core that holds Python
//! objects, directly or below it, has one Python object of this module ([`SharedPart`]), which
//! holds the part, and a reference to the [`SharedPart`] of each such part that the part holds,
//! one for each of its references: so the [`SharedPart`]s mirror the parts, and each Python
//! object that a part holds is reported by the [`SharedPart`] of that part alone, as its only
//! reference to it. An object of the binding holds a reference to the [`SharedPart`] of each part
//! of its value that it holds ([`Parts`]), as PySpark's objects hold their plan objects. So the
//! collector counts each reference once, sees a function of a UDF through every object that
//! reaches it, and frees a reference cycle through any of them, as PySpark's.
//!
//! A part may also be held by what the collector does not see: a value that a Rust structure of
//! the binding keeps, such as the dataset of the summary of a model, a clone that a request
//! holds while it runs, or a node that an object of the binding holds without a [`Parts`] of it.
//! A reference to a [`SharedPart`] is reported only when every reference to its part is one that
//! the [`SharedPart`]s and the objects of the binding hold ([`SharedPart::covered`]); else it is
//! not, which keeps the part and what it holds alive, as a reference from outside: the collector
//! never frees a Python object that such a holder reaches, and frees the cycle once that holder
//! is gone.
//!
//! The collector must get the same references of an object in each of its passes over the
//! objects that it collects: it subtracts the references that it gets from the counts of their
//! objects first (`subtract_refs`), then follows them from the objects that it finds reachable
//! (`move_unreachable`), so a reference that it subtracted and does not get again is of an object
//! that it takes for unreachable while a live holder reaches it: it clears the weak references to
//! that object and runs the finalizers of what only it reaches, and frees them unless the object
//! is found reachable again then. A collection holds the GIL and runs no Python code between its
//! passes, so the Python objects and their references stay as they are; but the calls of the
//! binding that release the GIL run Rust code meanwhile, of the binding and of the core, which
//! may clone and drop the parts of plans and expressions, as a request does as it builds its
//! plan: the strong count of a part, and so whether it is covered, changes between the passes.
//! So each call without the GIL is counted from before it releases the GIL until after it takes
//! it back, as what it may change, which no collection that it overlaps reports anything of
//! ([`SharedPart::reported`]):
//! - a call that clones and drops only the parts that the holders that it names hold, as a request
//!   of a DataFrame builds its plan ([`in_use`], [`of_parts`]), or those that values of the core
//!   that it holds hold, such as the clone of the DataFrame that an ML fit is of, or the copy of
//!   the state of a writer that a write runs on ([`in_use_of_values`], [`of_values`]): the
//!   [`SharedPart`]s that those reach are busy while it runs ([`Busy`]), and report nothing, and so
//!   is one that is made meanwhile for a part of the values that had none; the others report as
//!   they would. A call marks the [`SharedPart`]s of the parts that it names, whatever the parts
//!   below them, and the first read of a collection stamps those that they reach ([`stamp`]);
//! - a call that clones no part, nor drops one that a holder reports, as the wait of a streaming
//!   query, a call of the configuration, the catalog or the session, or a request of a message
//!   that Python code built, once its placeholders are replaced with the GIL held, of identifiers,
//!   protobufs and bytes ([`detach_without_parts`], [`without_parts`]): it is not counted;
//! - a call of code that the binding does not know, which may clone or drop any part, as the
//!   events that the listeners of the Rust API of a session get ([`detach`]): no [`SharedPart`]
//!   reports anything while it runs ([`DETACHED`]).
//!
//! A count and a mark change only with the GIL held, so a collection sees the same counts and
//! marks in all its passes; and when a part is neither counted nor marked, no code clones it or
//! drops one of a holder that reports it, as the threads of the runtime and of the core that run
//! without the GIL outside those calls (the prefetch of the rows of an iterator, the delivery of
//! events and jobs) only drop references that no holder covers, which makes a part covered and
//! never one uncovered. Between the passes of a collection and those that it makes again over the
//! objects that it found unreachable, once their finalizers ran, Python code runs: the counts
//! that it changes, such as of a part that a finalizer gives to a value of the core that a Rust
//! structure keeps, are those that the second passes read. A collection that runs while a call
//! runs, on another thread or as a part of its own, frees no cycle through the parts that the call
//! marks, which its holders and values keep alive anyway, nor, while a call of any part runs,
//! through any part; a later one does, once the call has ended. In the child of a fork, the counts
//! and the marks are those of the forking thread, the only one there ([`after_fork_in_child`]).
//!
//! A call that clones a part that it does not name, or one taken for a call of no part that clones
//! one, would let a collection see a reference in one pass and not in the next. The tests of the
//! binding check each call that they make: the core tells the binding of the new references to the
//! parts that it makes, of which one without the GIL, outside a counted call, of a part that no
//! call marks busy, fails them ([`check_clone`], `spark_connect::gc::check_clones_with`); and, as
//! the core tells of some of its clones only, a test watches the counts of the parts that are
//! reported, with the GIL held, while calls run ([`_watch_reported_parts`]).
//!
//! On a free-threaded build of CPython, which runs Python code of several threads at once, the
//! binding makes no [`SharedPart`] ([`init`]): the cache of them by address ([`Known`]) takes
//! references to them that only the GIL keeps from racing with their deallocation, so no
//! reference of the parts is reported there, and a cycle through them is not collected. A
//! free-threaded build of the extension is told by its build ([`free_threaded`]).

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard};

use pyo3::prelude::*;
use spark_connect::gc::{Part, PartRef};

/// Whether the value of an object of a `#[pyclass(dict)]` class is written, which the
/// `__traverse__` of the class reads before anything else of the object.
///
/// The garbage collector of CPython 3.11 can traverse such an object while it is created. PyO3
/// makes the object, of the class or of a Python subclass of it, with `object.__new__`, which
/// tracks the object that it allocates and then allocates its `__dict__`; that allocation starts
/// a collection when the free list of dicts is empty, before PyO3 writes the value of the object.
/// The value is zeroed memory then, which is not a valid value of most fields: an `Arc` is a null
/// pointer, a `OnceLock` reads as set (its completed state is zero) to a null `Py`, and an
/// `Option` of a struct of a `Py` may read as `Some` of one. A `GcReady` of zeroed memory is not
/// ready, so a `__traverse__` that checks it first reads nothing else of an object that is being
/// created. Only a field of `Option<Py<_>>`, whose zero is `None`, is read safely without it.
///
/// `__clear__` needs no such check: the collector clears only unreachable objects, and an object
/// that is being created is reachable from the call that makes it.
pub(crate) struct GcReady(AtomicBool);

impl GcReady {
    /// The flag of a written value, which each constructor of the class sets.
    pub(crate) fn new() -> Self {
        GcReady(AtomicBool::new(true))
    }

    /// Whether the value is written, so that a `__traverse__` may read it.
    pub(crate) fn get(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// The flag of a clone of a written value, which is written too: for a class whose value is
/// cloned.
impl Clone for GcReady {
    fn clone(&self) -> Self {
        GcReady::new()
    }
}

/// A part of the core that holds Python objects, directly or below it, as Python's garbage
/// collector sees it (see the module): it holds the part, and a reference to the [`SharedPart`]
/// of each part that the part holds that holds Python objects, one for each of its references.
/// Its `__traverse__` reports those of them that are reported, and the Python object of the part,
/// of a pickler or an owner, which it alone reports, while it is reported itself (see
/// [`SharedPart::reported`]). Made by [`Parts::of`], and found again by the address of its part
/// while it lives.
#[pyclass(frozen, name = "SharedPart", module = "pyspark._pyspark")]
pub(crate) struct SharedPart {
    /// Set once the value is written (see [`GcReady`]).
    ready: GcReady,
    part: Part,
    /// The references to `part` of which an object of the binding or a [`SharedPart`] holds a
    /// reference to this one, which it reports: those that cover the part.
    holders: AtomicUsize,
    /// The generation of the marks of the calls without the GIL ([`MARKS`]) whose stamp found
    /// this one among the [`SharedPart`]s that they reach: it is busy while that generation is
    /// the one of the marks (see [`SharedPart::busy`]).
    stamp: AtomicU64,
    /// The [`SharedPart`]s of the parts of Python objects that `part` holds, one for each of its
    /// references to them.
    held: Box<[Py<SharedPart>]>,
}

impl SharedPart {
    /// Whether every reference to the part is one that covers it: of the objects of the binding
    /// and of the [`SharedPart`]s that hold a reference to this one, with that of this one. A
    /// reference that is not, of a value that the collector does not see, keeps the part alive
    /// as one from outside, and so the reference to this one is not reported.
    fn covered(&self) -> bool {
        // Of no arithmetic that can panic, as a panic in a `__traverse__` aborts the process.
        let holders = self.holders.load(Ordering::Acquire);
        holders.checked_add(1) == Some(self.part.as_ref().strong_count())
    }

    /// Whether the references to this one, and those of it, are reported: while it is covered, no
    /// call of the binding without the GIL that may clone or drop any part runs, and none that may
    /// clone or drop this one (see the module), so that a collection gets the same references in
    /// each of its passes.
    fn reported(&self) -> bool {
        DETACHED.load(Ordering::SeqCst) == 0 && !self.busy() && self.covered()
    }

    /// Whether a call of the binding without the GIL that runs may clone or drop the part: the
    /// mark of one reaches this one (see [`Busy`]). Read with the GIL held, as the marks change
    /// only then.
    fn busy(&self) -> bool {
        if MARKED.load(Ordering::SeqCst) == 0 {
            return false;
        }
        let generation = stamped();
        self.stamp.load(Ordering::SeqCst) == generation
    }
}

/// Whether the parts of plans and expressions are seen by the collector at all: not on a
/// free-threaded build of CPython, of which no [`SharedPart`] is made (see [`init`]).
static SEEN: AtomicBool = AtomicBool::new(true);

/// The calls of the binding without the GIL that may clone or drop any part (see [`detach`]).
static DETACHED: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// The calls of [`DETACHED`] that this thread runs, which are all of them in the child of a
    /// fork of this thread (see [`after_fork_in_child`]).
    static DEPTH: Cell<usize> = const { Cell::new(0) };
    /// How the next call without the GIL of this thread through [`run_detached`] is counted,
    /// which [`take_counting`] takes as the call starts (see [`of_parts`]).
    static NEXT_CALL: Cell<Counting> = const { Cell::new(Counting::Any) };
    /// The identifier of this thread among those that mark parts busy (see [`MARKS`]).
    static THREAD: u64 = NEXT_THREAD.fetch_add(1, Ordering::Relaxed);
}

static NEXT_THREAD: AtomicU64 = AtomicU64::new(1);

/// A call of the binding without the GIL that may clone or drop any part, counted in [`DETACHED`]
/// while it lives: made and dropped with the GIL held.
struct Detached;

impl Detached {
    fn enter() -> Self {
        let _ = DEPTH.try_with(|depth| depth.set(depth.get() + 1));
        DETACHED.fetch_add(1, Ordering::SeqCst);
        Detached
    }
}

impl Drop for Detached {
    fn drop(&mut self) {
        DETACHED.fetch_sub(1, Ordering::SeqCst);
        let _ = DEPTH.try_with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Run `f` without the GIL, as [`Python::detach`] does, as a call of the binding that may clone
/// and drop any part of plans and expressions meanwhile, of which no reference to a [`SharedPart`]
/// is reported (see the module). The call is counted from before the GIL is released until after
/// it is taken back, as a collection holds the GIL: so it is counted throughout each collection
/// that runs while `f` does, as it unwinds too. Every release of the GIL in the binding goes
/// through it, [`detach_without_parts`], [`detach_of_values`] or [`run_detached`]
/// (`Python::detach`, and the functions of the C API that release the GIL, are disallowed methods
/// of the crate's clippy configuration).
#[allow(clippy::disallowed_methods)]
pub(crate) fn detach<T, F>(py: Python<'_>, f: F) -> T
where
    F: pyo3::marker::Ungil + FnOnce() -> T,
    T: pyo3::marker::Ungil,
{
    check_no_lock_held("a release of the GIL by the binding");
    let _detached = Detached::enter();
    py.detach(f)
}

/// Run `f` without the GIL, uncounted: a call that clones no part of plans and expressions, nor
/// drops one that a holder reports. It touches sessions, identifiers, protobufs and bytes only,
/// or drops values of the core that no holder reports, which makes a part covered and never one
/// uncovered.
#[allow(clippy::disallowed_methods)]
pub(crate) fn detach_without_parts<T, F>(py: Python<'_>, f: F) -> T
where
    F: pyo3::marker::Ungil + FnOnce() -> T,
    T: pyo3::marker::Ungil,
{
    check_no_lock_held("a release of the GIL by the binding");
    py.detach(f)
}

/// A lock of a state whose Python objects a `__traverse__` reports, such as the options that the
/// shallow copies of a reader or a writer share, whose keys may be any objects: taken only with the
/// GIL, of a token of it ([`GilMutex::lock`]), and released with it, as its guard cannot be moved
/// into a call without the GIL. So a collection, which holds the GIL, finds the lock in the same
/// state in each of its passes, and so gets the same objects of the state in each (see the
/// module); its `__traverse__` takes the lock as it is, without waiting
/// ([`GilMutex::try_lock_in_traverse`]).
///
/// The lock is held only while no Python code runs and the GIL is not released, as a thread that
/// wants it waits with the GIL held: Python code under it, which may release the GIL (any Python
/// function may, as the comparison of the keys of options with their `==` does), would deadlock
/// such a thread. So a thread that holds the GIL never finds it held by another one; the check of
/// the clones of the tests logs a lock that is found held, and a release of the GIL by the binding
/// or Python code of the objects of a state while a lock is held ([`check_no_lock_held`]).
pub(crate) struct GilMutex<T>(Mutex<T>);

thread_local! {
    /// The locks of [`GilMutex`]es that this thread holds.
    static LOCKS_HELD: Cell<usize> = const { Cell::new(0) };
}

/// A lock of a [`GilMutex`] that this thread holds, counted in [`LOCKS_HELD`] while it lives.
struct LockHeld;

impl LockHeld {
    fn new() -> Self {
        let _ = LOCKS_HELD.try_with(|held| held.set(held.get() + 1));
        LockHeld
    }
}

impl Drop for LockHeld {
    fn drop(&mut self) {
        let _ = LOCKS_HELD.try_with(|held| held.set(held.get().saturating_sub(1)));
    }
}

/// The state of a [`GilMutex`] while this thread holds its lock, which it took with the GIL held.
/// It cannot be sent to another thread, nor into a call without the GIL.
pub(crate) struct GilGuard<'a, T> {
    guard: MutexGuard<'a, T>,
    _held: LockHeld,
}

impl<T> std::ops::Deref for GilGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> std::ops::DerefMut for GilGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

impl<T> GilMutex<T> {
    pub(crate) const fn new(state: T) -> Self {
        GilMutex(Mutex::new(state))
    }

    /// The state, of its lock, which this thread takes with the GIL held, the token `py` of it.
    /// No other thread holds it then (see [`GilMutex`]); the check of the tests logs one that does,
    /// which released the GIL while it held it, or this one, which ran Python code that took it
    /// again. Then this thread waits with the GIL released between its tries, so that the other
    /// one can take the GIL back to release the lock, and still takes the lock with the GIL held.
    pub(crate) fn lock(&self, py: Python<'_>) -> GilGuard<'_, T> {
        let mut logged = false;
        let guard = loop {
            match self.0.try_lock() {
                Ok(guard) => break guard,
                Err(std::sync::TryLockError::Poisoned(poisoned)) => break poisoned.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => {
                    if !logged {
                        logged = true;
                        log_violation(
                            "a lock of the state of options found held by another thread, which \
                             released the GIL while it held it, or by this one, which ran Python \
                             code that took it again",
                        );
                    }
                    detach_without_parts(py, std::thread::yield_now);
                }
            }
        };
        GilGuard {
            guard,
            _held: LockHeld::new(),
        }
    }

    /// The state, unless the lock is held, as the `__traverse__` of the object that reports its
    /// Python objects takes it: with the GIL of the collection, `_visit` of which is the visitor,
    /// and without waiting. A state whose lock is held, or which a panic poisoned, is skipped,
    /// in every pass of the collection alike (see [`GilMutex`]).
    pub(crate) fn try_lock_in_traverse(
        &self,
        _visit: &pyo3::PyVisit<'_>,
    ) -> Option<GilGuard<'_, T>> {
        self.try_lock_now()
    }

    /// The state, unless the lock is held or poisoned, as a `__clear__` takes it, with the GIL
    /// (`_py`).
    pub(crate) fn try_lock(&self, _py: Python<'_>) -> Option<GilGuard<'_, T>> {
        self.try_lock_now()
    }

    fn try_lock_now(&self) -> Option<GilGuard<'_, T>> {
        let guard = self.0.try_lock().ok()?;
        Some(GilGuard {
            guard,
            _held: LockHeld::new(),
        })
    }

    /// Replace the state with `state`, and return the one that it replaces, which the caller
    /// drops once the lock is released, as the drop of its Python objects may run Python code.
    #[must_use = "the replaced state must be dropped once the lock is released"]
    pub(crate) fn replace(&self, py: Python<'_>, state: T) -> T {
        std::mem::replace(&mut *self.lock(py), state)
    }
}

/// Log a violation of the check of the clones of the tests, of `what`, with its backtrace (see
/// [`_take_part_clone_violations`]).
fn log_violation(what: &str) {
    if !CHECKING.load(Ordering::SeqCst) {
        return;
    }
    // Of the lock of the log, which a fork must not find held (see `CheckRunning`).
    let Some(_running) = CheckRunning::start() else {
        return;
    };
    VIOLATIONS.fetch_add(1, Ordering::Relaxed);
    let mut log = VIOLATION_LOG.lock().unwrap_or_else(|e| e.into_inner());
    if log.len() < LOGGED_VIOLATIONS {
        let thread = std::thread::current();
        log.push(format!(
            "{what}, on the thread {:?}:\n{}",
            thread.name().unwrap_or("<unnamed>"),
            std::backtrace::Backtrace::force_capture(),
        ));
    }
}

/// Log a violation of the check of the clones of the tests when this thread holds the lock of a
/// [`GilMutex`] as `what` happens: a release of the GIL by the binding, or Python code of the
/// objects of a state, such as the `==` of the keys of options, which a thread that waits for the
/// lock with the GIL held would deadlock with (see [`GilMutex`]).
pub(crate) fn check_no_lock_held(what: &str) {
    if !CHECKING.load(Ordering::Relaxed) || LOCKS_HELD.try_with(Cell::get).unwrap_or(0) == 0 {
        return;
    }
    log_violation(&format!(
        "{what} while this thread holds the lock of a state of options (a GilMutex)"
    ));
}

/// Run `f` without the GIL, as a call that clones and drops only the parts that `values` hold,
/// directly or below, values of the core that the caller holds for the time of the call, such as
/// the clone of the DataFrame that an ML fit is of: they are busy while it runs (see
/// [`in_use_of_values`]), and the call is not counted as one that may clone any part.
pub(crate) fn detach_of_values<'a, T, F>(
    py: Python<'_>,
    values: impl IntoIterator<Item = PartRef<'a>>,
    f: F,
) -> T
where
    F: pyo3::marker::Ungil + FnOnce() -> T,
    T: pyo3::marker::Ungil,
{
    let _busy = Busy::of_values(py, values);
    detach_without_parts(py, f)
}

/// How a call without the GIL is counted (see [`of_parts`]).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Counting {
    /// As one that may clone and drop any part ([`detach`]).
    Any,
    /// Not, as one that clones and drops only the parts of the holders and the values that
    /// [`of_parts`] or [`of_values`] named, which are busy while it runs, or that clones and
    /// drops no part ([`without_parts`]).
    Uncounted,
}

/// How the call without the GIL that starts now on this thread is counted, which [`of_parts`],
/// [`of_values`] or [`without_parts`] set for it: taken, so that a call of Python code that it
/// runs, or that runs before it, is counted as any part's.
pub(crate) fn take_counting() -> Counting {
    NEXT_CALL
        .try_with(|next| next.replace(Counting::Any))
        .unwrap_or(Counting::Any)
}

/// Run `f` without the GIL as `counting` says: the one call without the GIL of a call of the
/// binding that took its counting first ([`take_counting`]).
pub(crate) fn run_detached<T, F>(py: Python<'_>, counting: Counting, f: F) -> T
where
    F: pyo3::marker::Ungil + FnOnce() -> T,
    T: pyo3::marker::Ungil,
{
    match counting {
        Counting::Any => detach(py, f),
        Counting::Uncounted => detach_without_parts(py, f),
    }
}

/// The counting of the next call of this thread, set for the time of a call of the binding.
struct NextCall(Counting);

impl NextCall {
    fn set(counting: Counting) -> Self {
        NextCall(
            NEXT_CALL
                .try_with(|next| next.replace(counting))
                .unwrap_or(Counting::Any),
        )
    }
}

impl Drop for NextCall {
    fn drop(&mut self) {
        let _ = NEXT_CALL.try_with(|next| next.set(self.0));
    }
}

/// Run `call`, a call of the binding whose one call without the GIL takes its counting
/// ([`take_counting`], as `interrupt::run_interruptible` does), and which clones and drops only
/// the parts that `holders` hold, directly or below, as a request of a DataFrame builds its plan
/// (see [`in_use`]).
pub(crate) fn of_parts<R>(py: Python<'_>, holders: &[&Parts], call: impl FnOnce() -> R) -> R {
    let _in_use = in_use(py, holders);
    call()
}

/// Run `call`, a call of the binding whose one call without the GIL takes its counting, and which
/// clones and drops only the parts that `values` hold, directly or below (see
/// [`in_use_of_values`]).
pub(crate) fn of_values<'a, R>(
    py: Python<'_>,
    values: impl IntoIterator<Item = PartRef<'a>>,
    call: impl FnOnce() -> R,
) -> R {
    let _in_use = in_use_of_values(py, values);
    call()
}

/// The parts that `holders` hold, directly or below, for the next call without the GIL of this
/// thread that takes its counting ([`take_counting`]), which clones and drops only those: their
/// [`SharedPart`]s are busy until this is dropped ([`Busy`]), and report nothing meanwhile, and
/// the call is not counted as one that may clone any part. A call that reaches a part otherwise,
/// through a Rust structure or a weak reference only, names it as a value
/// ([`in_use_of_values`]) or is not of this. Made and dropped with the GIL held, around the call.
pub(crate) fn in_use(py: Python<'_>, holders: &[&Parts]) -> InUse {
    InUse {
        _busy: Busy::of(py, holders),
        _next: NextCall::set(Counting::Uncounted),
    }
}

/// As [`in_use`], of the parts that `values` hold, directly or below: values of the core that the
/// caller holds for the time of the call, such as a clone of the value of a holder, the dataset of
/// the summary of a model, or a plan that a Rust structure keeps. The [`SharedPart`]s that their
/// parts reach are busy, and so is one that is made for a part of them while the call runs, of a
/// part that had none (see [`Mark::unknown`]).
pub(crate) fn in_use_of_values<'a>(
    py: Python<'_>,
    values: impl IntoIterator<Item = PartRef<'a>>,
) -> InUse {
    InUse {
        _busy: Busy::of_values(py, values),
        _next: NextCall::set(Counting::Uncounted),
    }
}

/// The parts of the holders of a call of the binding that are in use while it runs (see
/// [`in_use`]).
pub(crate) struct InUse {
    _busy: Option<Busy>,
    _next: NextCall,
}

/// Run `call`, a call of the binding whose one call without the GIL takes its counting, as a call
/// that clones and drops no part (see [`detach_without_parts`]).
pub(crate) fn without_parts<R>(call: impl FnOnce() -> R) -> R {
    let _next = NextCall::set(Counting::Uncounted);
    call()
}

/// The parts that a call of the thread `thread` may clone and drop while it runs: those that the
/// [`SharedPart`]s of `roots` reach, which the mark keeps alive, and the parts at the addresses of
/// `unknown`.
struct Mark {
    id: u64,
    thread: u64,
    /// The [`SharedPart`]s of the parts that the call names, or of the first ones below them
    /// that have one: those and the ones that they reach are busy.
    roots: Vec<Py<SharedPart>>,
    /// The addresses of the parts of the values that the call names that had no [`SharedPart`]
    /// when it was marked, which the values keep for the time of the call: one that is made for
    /// one of them while the call runs takes its place among `roots` ([`mark_made`]).
    unknown: Addresses,
}

/// A set of the addresses of parts.
type Addresses = std::collections::HashSet<usize, std::hash::BuildHasherDefault<AddressHasher>>;

/// The marks of the calls of the binding without the GIL that run, of the parts that they may
/// clone and drop, which are busy while they run ([`Busy`]). In the child of a fork, those of the
/// threads of the parent, which are not in the child, are dropped ([`after_fork_in_child`]). Its
/// lock is held while no Python code runs, and only with the GIL held but by the check of the
/// clones of the tests ([`check_clone`]), which a fork waits for.
static MARKS: LazyLock<Mutex<Vec<Mark>>> = LazyLock::new(Default::default);

/// The generation of [`MARKS`], of which each change makes the next, with its lock held.
static MARKS_GENERATION: AtomicU64 = AtomicU64::new(1);

/// The generation of [`MARKS`] of which the [`SharedPart`]s that its marks reach are stamped (see
/// [`stamped`]).
static STAMPED: AtomicU64 = AtomicU64::new(0);

/// The number of the marks of [`MARKS`]: no part is busy while there is none.
static MARKED: AtomicUsize = AtomicUsize::new(0);

/// The number of the marks of [`MARKS`] of parts of no [`SharedPart`] (see [`Mark::unknown`]).
static MARKED_UNKNOWN: AtomicUsize = AtomicUsize::new(0);

static NEXT_MARK: AtomicU64 = AtomicU64::new(1);

fn marks() -> MutexGuard<'static, Vec<Mark>> {
    MARKS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Count `marks`, the marks after a change, of their lock, as a generation of its own.
fn changed(marks: &[Mark]) {
    MARKED.store(marks.len(), Ordering::SeqCst);
    MARKED_UNKNOWN.store(
        marks.iter().filter(|mark| !mark.unknown.is_empty()).count(),
        Ordering::SeqCst,
    );
    MARKS_GENERATION.fetch_add(1, Ordering::SeqCst);
}

/// The generation of the marks, of which the [`SharedPart`]s that they reach are stamped: they are
/// first, of the lock of the marks, unless they are already.
fn stamped() -> u64 {
    let generation = MARKS_GENERATION.load(Ordering::SeqCst);
    if STAMPED.load(Ordering::SeqCst) == generation {
        return generation;
    }
    stamp(&marks())
}

/// Stamp the [`SharedPart`]s that `marks`, the marks of the calls that run, reach, each once, of
/// the generation of the marks, unless they are of it already, and return it: of the lock of the
/// marks, with which every change of them is made. A mark is a few roots, whatever the parts that
/// they reach, so a call marks them at once; the [`SharedPart`]s below them are stamped once of a
/// generation, at the first read of a collection or of the check of the clones, which needs them.
fn stamp(marks: &[Mark]) -> u64 {
    let generation = MARKS_GENERATION.load(Ordering::SeqCst);
    if STAMPED.load(Ordering::SeqCst) != generation {
        let mut pending: Vec<&SharedPart> = marks
            .iter()
            .flat_map(|mark| mark.roots.iter().map(Py::get))
            .collect();
        while let Some(shared) = pending.pop() {
            if shared.stamp.swap(generation, Ordering::SeqCst) != generation {
                pending.extend(shared.held.iter().map(Py::get));
            }
        }
        STAMPED.store(generation, Ordering::SeqCst);
    }
    generation
}

/// The mark of a call of the binding without the GIL, of the parts that it may clone or drop,
/// which are busy while it lives: made and dropped with the GIL held (see [`MARKS`]).
struct Busy {
    mark: u64,
}

impl Busy {
    /// Mark busy the [`SharedPart`]s that `holders` reach; None of none.
    fn of(py: Python<'_>, holders: &[&Parts]) -> Option<Busy> {
        let roots = holders
            .iter()
            .flat_map(|parts| parts.shared.iter())
            .map(|shared| shared.clone_ref(py))
            .collect();
        Busy::mark(roots, Addresses::default())
    }

    /// Mark busy the [`SharedPart`]s that the parts of `values` reach, values of the core that the
    /// caller holds (see [`in_use_of_values`]), and those that are made, while the mark lives, of
    /// the parts of them that have none now; None of none.
    fn of_values<'a>(
        py: Python<'_>,
        values: impl IntoIterator<Item = PartRef<'a>>,
    ) -> Option<Busy> {
        if !SEEN.load(Ordering::Relaxed) {
            return None;
        }
        let (roots, unknown) = shared_parts_of(py, &known(), values);
        Busy::mark(roots, unknown)
    }

    fn mark(roots: Vec<Py<SharedPart>>, unknown: Addresses) -> Option<Busy> {
        if roots.is_empty() && unknown.is_empty() {
            return None;
        }
        let id = NEXT_MARK.fetch_add(1, Ordering::Relaxed);
        let thread = THREAD.try_with(|thread| *thread).unwrap_or(0);
        let mut marks = marks();
        marks.push(Mark {
            id,
            thread,
            roots,
            unknown,
        });
        changed(&marks);
        Some(Busy { mark: id })
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        let removed = {
            let mut marks = marks();
            let removed = marks
                .iter()
                .position(|mark| mark.id == self.mark)
                .map(|index| marks.swap_remove(index));
            changed(&marks);
            removed
        };
        // Its references to the SharedParts, which may be the last ones, once the lock is
        // released.
        drop(removed);
    }
}

/// The [`SharedPart`]s of the parts of `values`, each once, or, of a part that has none, of the
/// first parts below it that have one, and the addresses of the parts that have none, which a
/// walk of `values` meets without recursing; none of values of no Python object, of whose parts
/// none ever has one. A part known as of no Python object has no [`SharedPart`] below it. Of
/// `known`, while no Python code runs: the walk takes handles of the parts that it walks, which
/// `values` keep, and references to the [`SharedPart`]s that it finds.
fn shared_parts_of<'a>(
    py: Python<'_>,
    known: &KnownParts,
    values: impl IntoIterator<Item = PartRef<'a>>,
) -> (Vec<Py<SharedPart>>, Addresses) {
    let mut roots = Vec::new();
    let mut found = Addresses::default();
    let mut unknown = Addresses::default();
    let mut objects = false;
    let mut pending: Vec<Part> = Vec::new();
    let mut meet = |reference: PartRef<'_>, pending: &mut Vec<Part>| {
        let address = reference.address();
        match known_as(py, known, address) {
            Some(Resolved::Shared(shared)) => {
                if found.insert(address) {
                    roots.push(shared);
                }
            }
            Some(_) => {}
            None => {
                if unknown.insert(address) {
                    pending.push(reference.to_part());
                }
            }
        }
    };
    for value in values {
        meet(value, &mut pending);
    }
    while let Some(part) = pending.pop() {
        objects |= python_object(&part).is_some();
        for reference in part.held() {
            meet(reference, &mut pending);
        }
    }
    if roots.is_empty() && !objects {
        unknown.clear();
    }
    (roots, unknown)
}

/// Mark busy `shared`, the [`SharedPart`] just made of the part at `address`, of each mark of a
/// call that runs of a value that holds that part, which had none when the call was marked (see
/// [`Mark::unknown`]).
fn mark_made(py: Python<'_>, address: usize, shared: &Py<SharedPart>) {
    let mut marks = marks();
    let mut added = false;
    for mark in marks.iter_mut() {
        if mark.unknown.contains(&address) {
            mark.roots.push(shared.clone_ref(py));
            added = true;
        }
    }
    if added {
        changed(&marks);
    }
}

/// In the child of a fork, of the forking thread alone, with the GIL: the calls without the GIL
/// are its own, of which it may run some, as Python code of one forks, and the parts that the
/// calls of the threads of the parent marked busy are not any more. The marks of those calls,
/// which no thread ends in the child, are dropped, and so are their references to the
/// [`SharedPart`]s; the values that their calls held, on the stacks of their threads, are not.
#[pyfunction]
fn _after_fork_in_child() {
    after_fork_in_child();
}

fn after_fork_in_child() {
    // The checks of the clones and the forks of the threads of the parent, which are not in the
    // child (see `_hold_checks_for_fork`).
    CHECKS_RUNNING.store(0, Ordering::SeqCst);
    FORKS.store(0, Ordering::SeqCst);
    DETACHED.store(DEPTH.try_with(Cell::get).unwrap_or(0), Ordering::SeqCst);
    let thread = THREAD.try_with(|thread| *thread).ok();
    let others = {
        let mut marks = marks();
        let (own, others): (Vec<Mark>, Vec<Mark>) = std::mem::take(&mut *marks)
            .into_iter()
            .partition(|mark| Some(mark.thread) == thread);
        *marks = own;
        changed(&marks);
        others
    };
    drop(others);
}

/// Set the collection of the parts up as the extension loads: on a free-threaded build of CPython
/// the collector sees none of them, as the scheme of this module needs the GIL; and a fork resets
/// the counts of the calls without the GIL in the child ([`after_fork_in_child`]).
pub(crate) fn init(py: Python<'_>) -> PyResult<()> {
    if free_threaded(py) {
        SEEN.store(false, Ordering::SeqCst);
    }
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item(
        "after_in_child",
        wrap_pyfunction!(_after_fork_in_child, py)?,
    )?;
    register_at_fork(py, &kwargs)
}

/// Register the handlers of `kwargs` with `os.register_at_fork`, where there is one.
fn register_at_fork(py: Python<'_>, kwargs: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
    let os = py.import("os")?;
    if os.hasattr("register_at_fork")? {
        os.call_method("register_at_fork", (), Some(kwargs))?;
    }
    Ok(())
}

/// Whether the interpreter may run Python code of several threads at once: a free-threaded build
/// (`Py_GIL_DISABLED`), which only CPython 3.13 and later have. Of the build of the extension
/// first, as an interpreter of the free-threaded ABI loads only an extension built for it; then of
/// the interpreter, whose GIL is disabled, or whose build or ABI flags say so; and also when it
/// cannot be told, of which nothing is reported then.
fn free_threaded(py: Python<'_>) -> bool {
    if cfg!(Py_GIL_DISABLED) {
        return true;
    }
    (|| -> PyResult<bool> {
        let sys = py.import("sys")?;
        let version = sys.getattr("version_info")?;
        let version: (u32, u32) = (
            version.getattr("major")?.extract()?,
            version.getattr("minor")?.extract()?,
        );
        if version < (3, 13) {
            return Ok(false);
        }
        // The GIL of a free-threaded interpreter that is enabled may be disabled again (of a
        // module whose load fails), so the build is read too.
        if !sys.call_method0("_is_gil_enabled")?.is_truthy()? {
            return Ok(true);
        }
        if let Ok(flags) = sys.getattr("abiflags") {
            if flags.extract::<String>()?.contains('t') {
                return Ok(true);
            }
        }
        let disabled = py
            .import("sysconfig")?
            .call_method1("get_config_var", ("Py_GIL_DISABLED",))?;
        disabled.is_truthy()
    })()
    .unwrap_or(true)
}

/// The clones without the GIL, outside a counted call of their thread, that the check of the
/// clones of the tests saw ([`check_clone`]).
static CLONES_CHECKED: AtomicU64 = AtomicU64::new(0);

/// Of [`CLONES_CHECKED`], the clones of parts of a [`SharedPart`].
static CLONES_OF_SHARED: AtomicU64 = AtomicU64::new(0);

/// Of [`CLONES_OF_SHARED`], the clones of parts that no call marks busy: the violations.
static VIOLATIONS: AtomicU64 = AtomicU64::new(0);

/// The violations since the last take of them ([`_take_part_clone_violations`]), the first ones
/// of each take, with their backtraces.
static VIOLATION_LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// The violations of a take that are logged.
const LOGGED_VIOLATIONS: usize = 20;

/// Whether the check of the clones is on ([`_check_part_clones`]).
static CHECKING: AtomicBool = AtomicBool::new(false);

/// The forks of the process that are under way, each from its handler before the fork until its
/// handler after it, while the check of the clones is on: a check returns at once while one is
/// (see [`_hold_checks_for_fork`]). A count, as threads may fork at once: CPython runs the
/// handlers before a fork without a lock, so one fork may be over while another is not.
static FORKS: AtomicUsize = AtomicUsize::new(0);

/// The checks of the clones that run, without the GIL, which a fork waits for.
static CHECKS_RUNNING: AtomicUsize = AtomicUsize::new(0);

/// A check of the clones that runs, counted in [`CHECKS_RUNNING`] while it lives, which it
/// outlives the locks of.
struct CheckRunning;

impl CheckRunning {
    /// A check that runs now, unless a fork is under way.
    fn start() -> Option<CheckRunning> {
        CHECKS_RUNNING.fetch_add(1, Ordering::SeqCst);
        let running = CheckRunning;
        (FORKS.load(Ordering::SeqCst) == 0).then_some(running)
    }
}

impl Drop for CheckRunning {
    fn drop(&mut self) {
        CHECKS_RUNNING.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(not(windows))]
extern "C" {
    /// Whether this thread holds the GIL: of the C API of CPython, not of its limited API, of
    /// which the extension is built; the interpreter that loads the extension has it. Not on
    /// Windows, where an extension of the limited API links against the DLL of that API only.
    fn PyGILState_Check() -> std::os::raw::c_int;
}

/// Whether a thread of the process can tell whether it holds the GIL: `PyGILState_Check` of a
/// thread of no thread state of the interpreter says no, unless CPython's check of the state of
/// the GIL is off, as it is once a subinterpreter is made, after which it says yes of any thread.
fn gil_state_is_checked() -> bool {
    #[cfg(not(windows))]
    {
        // SAFETY: as in `holds_the_gil`.
        std::thread::spawn(|| unsafe { PyGILState_Check() == 0 })
            .join()
            .unwrap_or(false)
    }
    #[cfg(windows)]
    {
        false
    }
}

/// Whether this thread holds the GIL, where it can be told.
fn holds_the_gil() -> bool {
    #[cfg(not(windows))]
    {
        // SAFETY: of no arguments, and safe to call on any thread, which has a thread state of
        // the interpreter or not.
        unsafe { PyGILState_Check() != 0 }
    }
    #[cfg(windows)]
    {
        true
    }
}

/// Check the new reference to the part at `address` that the core makes (see
/// `spark_connect::gc::check_clones_with`), for the tests: one that is made without the GIL,
/// outside a counted call of its thread, of a part that has a [`SharedPart`] that no call marks
/// busy, is a reference that the collector may get in one pass and not in the next, unless the
/// part is not covered then. Such a clone is a call of the binding that clones a part that it does
/// not name ([`in_use`], [`in_use_of_values`]), or that it takes for one that clones none
/// ([`without_parts`]): it is logged, with the backtrace of the clone.
fn check_clone(address: usize) {
    // With the GIL, no collection runs its passes meanwhile; the code of the binding that holds
    // the lock of the known parts holds the GIL, so a clone that it makes returns here.
    if holds_the_gil() {
        return;
    }
    if DEPTH.try_with(Cell::get).unwrap_or(0) > 0 {
        return;
    }
    // Of the locks of the known parts, of the marks and of the log, which a fork must not find
    // held: it waits for the checks that run, and none starts while a fork is under way.
    let Some(_running) = CheckRunning::start() else {
        return;
    };
    CLONES_CHECKED.fetch_add(1, Ordering::Relaxed);
    let known = known();
    let Some(Known::Shared { value, .. }) = known.get(&address) else {
        return;
    };
    // SAFETY: the SharedPart lives while the lock is held, which its drop takes first.
    let shared = unsafe { &*(*value as *const SharedPart) };
    CLONES_OF_SHARED.fetch_add(1, Ordering::Relaxed);
    let busy = {
        let marks = marks();
        MARKED.load(Ordering::SeqCst) > 0 && {
            let generation = stamp(&marks);
            shared.stamp.load(Ordering::SeqCst) == generation
        }
    };
    if busy {
        return;
    }
    let covered = shared.covered();
    let kind = match &shared.part {
        Part::Plan(_) => "a plan",
        Part::Expression(_) => "an expression",
        Part::Pickler(_) => "a pickler",
        Part::Owner(_) => "an owner",
    };
    drop(known);
    VIOLATIONS.fetch_add(1, Ordering::Relaxed);
    let mut log = VIOLATION_LOG.lock().unwrap_or_else(|e| e.into_inner());
    if log.len() < LOGGED_VIOLATIONS {
        let thread = std::thread::current();
        log.push(format!(
            "a clone without the GIL, outside a counted call, of {kind} that has a SharedPart \
             that no call marks busy (covered: {covered}; counted calls: {}), on the thread {:?}:\n{}",
            DETACHED.load(Ordering::SeqCst),
            thread.name().unwrap_or("<unnamed>"),
            std::backtrace::Backtrace::force_capture(),
        ));
    }
}

/// Turn on the check of the clones of the parts of plans and expressions that the core makes, for
/// the tests: a clone that runs without the GIL, outside a counted call of its thread, of a part
/// of a [`SharedPart`] that no call marks busy, is logged, with its backtrace (see
/// [`check_clone`], [`_take_part_clone_violations`]). True if it is on, now or already; false
/// where it cannot be: on Windows, and where a thread cannot tell whether it holds the GIL (see
/// [`gil_state_is_checked`]).
#[pyfunction]
pub(crate) fn _check_part_clones(py: Python<'_>) -> PyResult<bool> {
    if cfg!(windows) || !gil_state_is_checked() {
        return Ok(false);
    }
    if spark_connect::gc::check_clones_with(check_clone) {
        // A check takes the locks of the known parts, of the marks and of the log without the
        // GIL: a fork waits for the checks that run, and none starts until it is over, so that
        // the locks are free at the fork. A lock is not held across the fork, as Python code runs
        // in the parent and in the child before the handlers after it, which may free a holder
        // or read the marks, and so take them.
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("before", wrap_pyfunction!(_hold_checks_for_fork, py)?)?;
        kwargs.set_item(
            "after_in_parent",
            wrap_pyfunction!(_release_checks_after_fork, py)?,
        )?;
        kwargs.set_item(
            "after_in_child",
            wrap_pyfunction!(_release_checks_in_child, py)?,
        )?;
        register_at_fork(py, &kwargs)?;
        CHECKING.store(true, Ordering::SeqCst);
    }
    Ok(CHECKING.load(Ordering::SeqCst))
}

/// Whether the check of the clones is on and can tell whether a thread holds the GIL now, which a
/// subinterpreter made since turns off (see [`gil_state_is_checked`]).
#[pyfunction]
pub(crate) fn _part_clones_checked() -> bool {
    CHECKING.load(Ordering::SeqCst) && gil_state_is_checked()
}

/// Before a fork, with the GIL, of the check of the clones: no check starts until the fork is
/// over, nor until any other fork that is under way is, and the ones that run, without the GIL,
/// end, so that no thread holds the locks of the known parts, of the marks or of the log at the
/// fork. A check takes nothing that the forking thread holds, so it ends.
#[pyfunction]
fn _hold_checks_for_fork() {
    FORKS.fetch_add(1, Ordering::SeqCst);
    while CHECKS_RUNNING.load(Ordering::SeqCst) != 0 {
        std::thread::yield_now();
    }
}

/// After a fork, in the parent: the fork is over, and the checks of the clones run again once no
/// other fork is under way. Never below none, of a fork that was under way when the check turned
/// on, whose handler before it did not run.
#[pyfunction]
fn _release_checks_after_fork() {
    let mut forks = FORKS.load(Ordering::SeqCst);
    while forks > 0 {
        match FORKS.compare_exchange(forks, forks - 1, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => break,
            Err(now) => forks = now,
        }
    }
}

/// After a fork, in the child: the checks of the clones run again, of none of the threads of the
/// parent, which are not in the child, nor of their forks (as [`after_fork_in_child`] does too).
#[pyfunction]
fn _release_checks_in_child() {
    CHECKS_RUNNING.store(0, Ordering::SeqCst);
    FORKS.store(0, Ordering::SeqCst);
}

/// The clones that the check of the clones saw ([`_check_part_clones`]): those without the GIL
/// outside a counted call of their thread, of them those of parts of a [`SharedPart`], and of
/// those the violations, since the check is on; and the logs of the violations since the last
/// call, which this takes, the first ones of them.
#[pyfunction]
pub(crate) fn _take_part_clone_violations() -> (u64, u64, u64, Vec<String>) {
    let log = std::mem::take(&mut *VIOLATION_LOG.lock().unwrap_or_else(|e| e.into_inner()));
    (
        CLONES_CHECKED.load(Ordering::Relaxed),
        CLONES_OF_SHARED.load(Ordering::Relaxed),
        VIOLATIONS.load(Ordering::Relaxed),
        log,
    )
}

/// Watch the parts that the collector sees, for the tests, of nothing that the core tells: with
/// the GIL held throughout, so that no collection runs and no count of the calls, no mark and no
/// holder changes meanwhile, and the lock of the known parts, read the strong counts and the
/// holders of the parts of the [`SharedPart`]s that are reported now, again and again, for
/// `spin_us` microseconds. A reported part is covered, so no code but its holders holds it: a
/// change is a new reference to it, made without the GIL, of a call that clones a part that it
/// does not name, whatever code makes it (see [`check_clone`], which the core tells of the clones
/// that it makes). Returns the [`SharedPart`]s watched, the reads of them, the parts that changed,
/// and the logs of the first ones.
#[pyfunction]
pub(crate) fn _watch_reported_parts(_py: Python<'_>, spin_us: u64) -> (u64, u64, u64, Vec<String>) {
    let known = known();
    let (detached, marked) = (
        DETACHED.load(Ordering::SeqCst),
        MARKED.load(Ordering::SeqCst),
    );
    let mut watched: Vec<(&SharedPart, usize, usize)> = Vec::new();
    for entry in known.values() {
        if let Known::Shared { value, .. } = entry {
            // SAFETY: the SharedPart lives while the lock is held, which its drop takes first.
            let shared = unsafe { &*(*value as *const SharedPart) };
            if shared.ready.get() && shared.reported() {
                watched.push((
                    shared,
                    shared.part.as_ref().strong_count(),
                    shared.holders.load(Ordering::SeqCst),
                ));
            }
        }
    }
    let start = std::time::Instant::now();
    let (mut sweeps, mut changes) = (0u64, 0u64);
    let mut logs = Vec::new();
    let mut seen = vec![false; watched.len()];
    loop {
        sweeps += 1;
        for (index, (shared, count, holders)) in watched.iter().enumerate() {
            let (count_now, holders_now) = (
                shared.part.as_ref().strong_count(),
                shared.holders.load(Ordering::SeqCst),
            );
            if seen[index] || (count_now == *count && holders_now == *holders) {
                continue;
            }
            seen[index] = true;
            changes += 1;
            if logs.len() < LOGGED_VIOLATIONS {
                let kind = match &shared.part {
                    Part::Plan(_) => "a plan",
                    Part::Expression(_) => "an expression",
                    Part::Pickler(_) => "a pickler",
                    Part::Owner(_) => "an owner",
                };
                logs.push(format!(
                    "{kind} reported of {count} references and {holders} holders had \
                     {count_now} references and {holders_now} holders, with the GIL held, after \
                     {} us (counted calls {detached}, marks {marked})",
                    start.elapsed().as_micros()
                ));
            }
        }
        if start.elapsed().as_micros() >= u128::from(spin_us) {
            break;
        }
    }
    if DETACHED.load(Ordering::SeqCst) != detached || MARKED.load(Ordering::SeqCst) != marked {
        logs.push(format!(
            "the counted calls or the marks changed with the GIL held: {detached} to {}, {marked} \
             to {}",
            DETACHED.load(Ordering::SeqCst),
            MARKED.load(Ordering::SeqCst)
        ));
    }
    drop(known);
    (watched.len() as u64, sweeps, changes, logs)
}

#[pymethods]
impl SharedPart {
    /// As the plan objects of PySpark, which Python's garbage collector sees: the [`SharedPart`]s
    /// of the parts that this part holds, while they are reported, and the Python object of a
    /// pickler or an owner, while this one is (see [`SharedPart::reported`]). A reference to this
    /// one that is reported is that of a holder of a reference to the part, so a Python object
    /// that the part holds is reported once, by this one, of its one reference.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.ready.get() {
            return Ok(());
        }
        for held in self.held.iter() {
            if held.get().reported() {
                visit.call(held)?;
            }
        }
        if let Some(object) = python_object(&self.part) {
            if self.reported() {
                visit.call(object)?;
            }
        }
        Ok(())
    }

    /// The kind of the part, its holders and its references, as
    /// `<SharedPart of a plan, 2 holders of 3 references>`: a part that has more references than
    /// holders, and so is not covered, keeps what it holds alive.
    fn __repr__(&self) -> String {
        let kind = match &self.part {
            Part::Plan(_) => "a plan",
            Part::Expression(_) => "an expression",
            Part::Pickler(_) => "a pickler",
            Part::Owner(_) => "an owner",
        };
        let holders = self.holders.load(Ordering::Acquire);
        // The reference of this one is not of a holder.
        let references = self.part.as_ref().strong_count().saturating_sub(1);
        let plural = |n: usize| if n == 1 { "" } else { "s" };
        format!(
            "<SharedPart of {kind}, {holders} holder{} of {references} reference{}>",
            plural(holders),
            plural(references)
        )
    }
}

impl Drop for SharedPart {
    /// Forget this part, then drop the references to the [`SharedPart`]s that it holds, of which
    /// it is no holder any more, one at a time: the parts of a plan of many levels would
    /// otherwise drop each the one below it.
    fn drop(&mut self) {
        forget_shared(
            self.part.as_ref().address(),
            self as *const SharedPart as usize,
        );
        for held in self.held.iter() {
            held.get().holders.fetch_sub(1, Ordering::AcqRel);
        }
        drop_in_turn(std::mem::take(&mut self.held));
    }
}

/// The Python object that a part holds itself: the object of the pickler of a command of the
/// binding, or a Python object as the object of a pickler or as an owner.
fn python_object(part: &Part) -> Option<&Py<PyAny>> {
    let object = part.object()?;
    crate::udf_command::python_object_of(object).or_else(|| object.downcast_ref::<Py<PyAny>>())
}

thread_local! {
    /// The references to [`SharedPart`]s that the drops of this thread drop in turn, while one
    /// of them drops them.
    static DROPPING: RefCell<Option<Vec<Py<SharedPart>>>> = const { RefCell::new(None) };
}

/// Drop `held`, and the references that the [`SharedPart`]s that they free hold, one at a time.
fn drop_in_turn(held: Box<[Py<SharedPart>]>) {
    if held.is_empty() {
        return;
    }
    let mut held = Some(held.into_vec());
    let queued = DROPPING
        .try_with(|cell| {
            let mut slot = cell.borrow_mut();
            match slot.as_mut() {
                // Of a drop within the drop of a reference by this thread, which drops these in
                // its turn.
                Some(pending) => {
                    pending.extend(held.take().into_iter().flatten());
                    true
                }
                None => {
                    *slot = held.take();
                    false
                }
            }
        })
        .unwrap_or(false);
    if queued {
        return;
    }
    if held.is_some() {
        // The thread's slot is gone, at its exit: drop them as they are.
        drop(held);
        return;
    }
    while let Some(next) = DROPPING
        .try_with(|cell| cell.borrow_mut().as_mut().and_then(Vec::pop))
        .ok()
        .flatten()
    {
        drop(next);
    }
    let _ = DROPPING.try_with(|cell| cell.borrow_mut().take());
}

/// What the binding knows of a part, by its address: the [`SharedPart`] of a part of Python
/// objects while it lives, and a part of none that objects of the binding hold, of the number of
/// their references to it, so that the parts of values made of it do not walk it again.
enum Known {
    /// The [`SharedPart`], of the addresses of its Python object and of its value.
    Shared { object: usize, value: usize },
    /// A part of no Python object, of the number of the [`Parts`] that hold it.
    Plain(usize),
}

/// A hasher of the addresses of parts, which are aligned and distinct.
#[derive(Default)]
struct AddressHasher(u64);

impl std::hash::Hasher for AddressHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(*byte)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
    }

    fn write_usize(&mut self, address: usize) {
        self.0 = (address as u64 >> 3).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

type KnownParts = HashMap<usize, Known, std::hash::BuildHasherDefault<AddressHasher>>;

/// The parts that the binding knows of. Its lock is held only while no Python code runs.
static KNOWN: LazyLock<Mutex<KnownParts>> = LazyLock::new(Default::default);

fn known() -> std::sync::MutexGuard<'static, KnownParts> {
    KNOWN.lock().unwrap_or_else(|e| e.into_inner())
}

/// Forget the [`SharedPart`] of the part at `address` whose value is at `value`, unless another
/// one is known for it.
fn forget_shared(address: usize, value: usize) {
    let mut known = known();
    if matches!(known.get(&address), Some(Known::Shared { value: v, .. }) if *v == value) {
        known.remove(&address);
    }
}

/// What [`resolve`] found of a part: no Python object below it, of whether it holds parts (so
/// that it is worth remembering as a root), the [`SharedPart`] of a part that has one, or the one
/// of the creations of the walk that it gets.
enum Resolved {
    Plain { holds: bool },
    Shared(Py<SharedPart>),
    New(usize),
}

impl Resolved {
    /// The resolution of another reference to the same part: of a reference of its own to a
    /// [`SharedPart`].
    fn again(&self, py: Python<'_>) -> Resolved {
        match self {
            Resolved::Plain { holds } => Resolved::Plain { holds: *holds },
            Resolved::Shared(shared) => Resolved::Shared(shared.clone_ref(py)),
            Resolved::New(index) => Resolved::New(*index),
        }
    }
}

/// A [`SharedPart`] to make, of its part and of what each of its references to a part of Python
/// objects resolved to, a [`SharedPart`] or an earlier creation.
struct Creation {
    part: Part,
    held: Vec<Resolved>,
}

/// A walk of [`resolve`]: what it resolved `root` to, the [`SharedPart`]s to make, each after
/// those of the parts that it holds, and what it resolved the parts below to, which hold
/// references to [`SharedPart`]s, dropped once the lock of the known parts is released.
struct Walk {
    root: Resolved,
    creations: Vec<Creation>,
    walked: Walked,
}

/// The parts that a walk resolved, by address: in a list while they are few, as most walks are
/// of a node or two over parts that are known, else in a map.
#[derive(Default)]
struct Walked {
    few: Vec<(usize, Resolved)>,
    many: HashMap<usize, Resolved, std::hash::BuildHasherDefault<AddressHasher>>,
}

impl Walked {
    const FEW: usize = 16;

    fn get(&self, address: usize) -> Option<&Resolved> {
        if self.many.is_empty() {
            self.few
                .iter()
                .find(|(walked, _)| *walked == address)
                .map(|(_, resolved)| resolved)
        } else {
            self.many.get(&address)
        }
    }

    fn insert(&mut self, address: usize, resolved: Resolved) {
        if self.many.is_empty() && self.few.len() < Walked::FEW {
            self.few.push((address, resolved));
        } else {
            self.many.extend(self.few.drain(..));
            self.many.insert(address, resolved);
        }
    }
}

/// Whether the [`SharedPart`] object at `object`, which the known parts know, is being
/// deallocated: its last reference is gone, and its drop, which forgets it, has not run yet.
/// Python code runs in between, on the thread that deallocates it, as PyO3 first drops the
/// references that threads dropped without the GIL, whose finalizers may make holders of its part
/// through a holder that the collector does not see, as the core of a native model keeps its
/// dataset. Of the lock of the known parts and with the GIL.
fn being_deallocated(object: usize) -> bool {
    // SAFETY: a known SharedPart is not freed while the lock of the known parts is held, which its
    // drop takes first, and its count of references changes only with the GIL.
    unsafe { pyo3::ffi::Py_REFCNT(object as *mut pyo3::ffi::PyObject) == 0 }
}

/// The [`SharedPart`] that a part known at `address` has, of a reference of its own, or what it
/// is known for. A part whose [`SharedPart`] is being deallocated is unknown, so that it gets one
/// of its own (see [`being_deallocated`]).
fn known_as(py: Python<'_>, known: &KnownParts, address: usize) -> Option<Resolved> {
    match known.get(&address)? {
        Known::Shared { object, .. } => {
            // The [`SharedPart`] lives, of the references of its holders, and it is not freed
            // while the lock is held, which its drop takes first.
            if being_deallocated(*object) {
                return None;
            }
            let object = *object as *mut pyo3::ffi::PyObject;
            let shared = unsafe { Bound::from_borrowed_ptr(py, object) };
            Some(Resolved::Shared(
                shared.cast_into::<SharedPart>().ok()?.unbind(),
            ))
        }
        Known::Plain(_) => Some(Resolved::Plain { holds: true }),
    }
}

/// Resolve `root`: a part that the binding knows is as it is known, and another one by the parts
/// that it holds, each once, without recursing, so that a part that holds a part of Python
/// objects, or a Python object itself, gets a [`SharedPart`]. Of `known`, while no Python code
/// runs: it makes no Python object, and takes the references to the [`SharedPart`]s that it finds.
fn resolve(py: Python<'_>, known: &KnownParts, root: Part) -> Walk {
    enum Step {
        Enter(Part),
        /// A part that the binding knows, or that this walk resolved, of its resolution: taken
        /// as it is, of no handle of the part.
        Known(Resolved),
        Leave(usize, Part, usize),
    }
    let mut walk = Walk {
        root: Resolved::Plain { holds: false },
        creations: Vec::new(),
        walked: Walked::default(),
    };
    // The resolutions of the references of the parts that are being walked, in order: a `Leave`
    // takes those of its part's references from the end.
    let mut results: Vec<Resolved> = Vec::new();
    let mut steps = vec![Step::Enter(root)];
    while let Some(step) = steps.pop() {
        match step {
            Step::Known(resolved) => results.push(resolved),
            Step::Enter(part) => {
                let address = part.as_ref().address();
                if let Some(resolved) = walk.walked.get(address) {
                    results.push(resolved.again(py));
                } else if let Some(resolved) = known_as(py, known, address) {
                    walk.walked.insert(address, resolved.again(py));
                    results.push(resolved);
                } else {
                    // The parts that it holds, in order: those that are known as they are, and
                    // the others walked of handles of their own, which keep them for the time of
                    // the walk. A part known as of no Python object, as most are, is resolved
                    // again by its address wherever the walk meets it.
                    let held: Vec<Step> = part
                        .held()
                        .into_iter()
                        .map(|reference| {
                            let address = reference.address();
                            match walk.walked.get(address) {
                                Some(resolved) => Step::Known(resolved.again(py)),
                                None => match known_as(py, known, address) {
                                    Some(resolved) => Step::Known(resolved),
                                    None => Step::Enter(reference.to_part()),
                                },
                            }
                        })
                        .collect();
                    steps.push(Step::Leave(address, part, held.len()));
                    steps.extend(held.into_iter().rev());
                }
            }
            Step::Leave(address, part, count) => {
                let start = results.len() - count;
                let held: Vec<Resolved> = results
                    .drain(start..)
                    .filter(|resolved| !matches!(resolved, Resolved::Plain { .. }))
                    .collect();
                let resolved = if held.is_empty() && python_object(&part).is_none() {
                    Resolved::Plain { holds: count > 0 }
                } else {
                    walk.creations.push(Creation { part, held });
                    Resolved::New(walk.creations.len() - 1)
                };
                walk.walked.insert(address, resolved.again(py));
                results.push(resolved);
            }
        }
    }
    if let Some(root) = results.pop() {
        walk.root = root;
    }
    walk
}

/// Make the [`SharedPart`]s of `creations`, in order, each holding a reference to those of its
/// parts, which it holds as a holder of their parts, and known by the address of its part, unless
/// another one is known for it already (as one that a collection that a creation started made,
/// of a finalizer): then both are, and neither is covered while both live. One that is being
/// deallocated is replaced (see [`being_deallocated`]), whose drop then forgets nothing. None if
/// one fails.
fn make(py: Python<'_>, creations: Vec<Creation>) -> Option<Vec<Py<SharedPart>>> {
    let mut made: Vec<Py<SharedPart>> = Vec::with_capacity(creations.len());
    for creation in creations {
        let held: Vec<Py<SharedPart>> = creation
            .held
            .into_iter()
            .map(|resolved| match resolved {
                Resolved::Shared(shared) => shared,
                Resolved::New(index) => made[index].clone_ref(py),
                Resolved::Plain { .. } => unreachable!("plain parts are not held"),
            })
            .collect();
        for shared in &held {
            shared.get().holders.fetch_add(1, Ordering::AcqRel);
        }
        let address = creation.part.as_ref().address();
        let shared = Py::new(
            py,
            SharedPart {
                ready: GcReady::new(),
                part: creation.part,
                holders: AtomicUsize::new(0),
                stamp: AtomicU64::new(0),
                held: held.into_boxed_slice(),
            },
        )
        .ok()?;
        {
            let mut known = known();
            let known_already = match known.get(&address) {
                Some(Known::Shared { object, .. }) => !being_deallocated(*object),
                _ => false,
            };
            if !known_already {
                known.insert(
                    address,
                    Known::Shared {
                        object: shared.as_ptr() as usize,
                        value: shared.get() as *const SharedPart as usize,
                    },
                );
            }
        }
        // Of a part of a value that a call that runs named, which had none then.
        if MARKED_UNKNOWN.load(Ordering::SeqCst) > 0 {
            mark_made(py, address, &shared);
        }
        made.push(shared);
    }
    Some(made)
}

/// The references of an object of the binding to the parts of its value, which it holds as
/// PySpark's objects hold their plan objects (see the module): a reference to the [`SharedPart`]
/// of each part of Python objects, of which it is a holder, which its `__traverse__` reports
/// ([`Parts::visit`]), and the address of each other part that holds parts, which the binding
/// remembers while it is held, so that the parts of values made of it do not walk it again. The
/// object holds the references to the parts that it gives to [`Parts::of`] for as long as it holds
/// these, and drops these when it drops those, or before ([`Parts::release`]): its references keep
/// the parts at those addresses. One that it drops first, as a field of an object that its drop
/// drops before these, is forgotten when these are dropped: a part that takes its address
/// meanwhile is taken for one of no Python object, whose Python objects are not reported then,
/// which keeps them alive.
#[derive(Default)]
pub(crate) struct Parts {
    shared: Box<[Py<SharedPart>]>,
    plain: Box<[usize]>,
}

impl Parts {
    /// The parts of no references.
    pub(crate) fn none() -> Self {
        Parts::default()
    }

    /// The parts of `references`, the references that an object of the binding holds to the
    /// parts of its value, each of one field of the value, as `held_parts` of a value of the core
    /// gives them.
    pub(crate) fn of<'a>(
        py: Python<'_>,
        references: impl IntoIterator<Item = PartRef<'a>>,
    ) -> Self {
        if !SEEN.load(Ordering::Relaxed) {
            return Parts::none();
        }
        let mut shared_parts: Vec<Py<SharedPart>> = Vec::new();
        let mut plain_parts: Vec<usize> = Vec::new();
        for reference in references {
            let address = reference.address();
            let walk = {
                let mut known = known();
                let walk = resolve(py, &known, reference.to_part());
                if let Resolved::Plain { holds: true } = walk.root {
                    match known.entry(address).or_insert(Known::Plain(0)) {
                        Known::Plain(count) => *count += 1,
                        Known::Shared { .. } => unreachable!("a part of no Python object"),
                    }
                    plain_parts.push(address);
                }
                walk
            };
            let Walk {
                root,
                creations,
                walked,
            } = walk;
            let shared = match root {
                Resolved::Plain { .. } => None,
                Resolved::Shared(shared) => Some(shared),
                Resolved::New(index) => make(py, creations)
                    .and_then(|mut made| (index < made.len()).then(|| made.swap_remove(index))),
            };
            drop(walked);
            if let Some(shared) = shared {
                shared.get().holders.fetch_add(1, Ordering::AcqRel);
                shared_parts.push(shared);
            }
        }
        Parts {
            shared: shared_parts.into_boxed_slice(),
            plain: plain_parts.into_boxed_slice(),
        }
    }

    /// Report the [`SharedPart`]s of the parts of Python objects, those that are reported (see
    /// [`SharedPart::reported`]), for the `__traverse__` of the object that holds these parts.
    pub(crate) fn visit(&self, visit: &pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        for shared in &self.shared {
            if shared.get().reported() {
                visit.call(shared)?;
            }
        }
        Ok(())
    }

    /// Drop these references, as the object drops its references to the parts, for its
    /// `__clear__` and when it replaces its value; also without the GIL, in a call of the binding
    /// that releases it (see [`detach`]), of which PyO3 drops the references later.
    pub(crate) fn release(&mut self) {
        for shared in std::mem::take(&mut self.shared).into_vec() {
            shared.get().holders.fetch_sub(1, Ordering::AcqRel);
            drop(shared);
        }
        let plain = std::mem::take(&mut self.plain);
        if !plain.is_empty() {
            let mut known = known();
            for address in plain.iter() {
                if let Some(Known::Plain(count)) = known.get_mut(address) {
                    *count -= 1;
                    if *count == 0 {
                        known.remove(address);
                    }
                }
            }
        }
    }
}

impl Drop for Parts {
    fn drop(&mut self) {
        self.release();
    }
}

/// Release the GIL until no call of the binding without the GIL is counted, as one of another
/// test may be: none starts or ends while the calling thread holds the GIL. For tests of what the
/// collector sees.
#[cfg(test)]
#[allow(clippy::disallowed_methods)]
pub(crate) fn wait_for_no_counted_call(py: Python<'_>) {
    while DETACHED.load(Ordering::SeqCst) != 0 {
        py.detach(|| std::thread::sleep(std::time::Duration::from_millis(1)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyDict, PyList, PyTuple};
    use spark_connect::expression::{Expression, ExpressionNode};
    use spark_connect::udf::{CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

    /// The expression of a UDF of `object`, as the binding makes it.
    fn udf_of(object: Py<PyAny>) -> Expression {
        let pickler = crate::udf_command::udf_pickler(object);
        ExpressionNode::CommonInlineUserDefinedFunction(Box::new(
            CommonInlineUserDefinedFunctionExpression::new(
                "f".to_string(),
                true,
                vec![Expression::column("x")],
                PythonUDFPayload::pickled_at_build(
                    spark_connect::types::DataType::Long,
                    100,
                    pickler,
                    "3.11".to_string(),
                ),
            ),
        ))
        .into()
    }

    /// Whether the collector sees `target` among the referents of `of`, once no call of the
    /// binding runs without the GIL, as one of another test may: none starts or ends while this
    /// thread holds the GIL.
    fn reports(py: Python<'_>, of: &Py<SharedPart>, target: &Bound<'_, PyAny>) -> bool {
        wait_for_no_call_without_the_gil(py);
        reports_now(py, of, target)
    }

    /// Whether the collector sees `target` among the referents of `of` now.
    fn reports_now(py: Python<'_>, of: &Py<SharedPart>, target: &Bound<'_, PyAny>) -> bool {
        let referents = py
            .import("gc")
            .unwrap()
            .call_method1("get_referents", (of,))
            .unwrap();
        let referents = referents.cast::<PyList>().unwrap();
        referents.iter().any(|referent| referent.is(target))
    }

    /// Run `f` without the GIL, uncounted, as a wait of a test that clones and drops no part.
    #[allow(clippy::disallowed_methods)]
    fn uncounted<T: Send>(py: Python<'_>, f: impl FnOnce() -> T + Send) -> T {
        py.detach(f)
    }

    /// Release the GIL until no call of the binding runs without it.
    fn wait_for_no_call_without_the_gil(py: Python<'_>) {
        while DETACHED.load(Ordering::SeqCst) != 0 {
            uncounted(py, || {
                std::thread::sleep(std::time::Duration::from_millis(1))
            });
        }
    }

    /// Release the GIL until `thread` ends.
    fn join<T: Send>(py: Python<'_>, thread: std::thread::JoinHandle<T>) -> T {
        uncounted(py, || thread.join().unwrap())
    }

    #[test]
    fn a_part_that_a_holder_that_the_collector_does_not_see_holds_is_not_reported() {
        Python::initialize();
        Python::attach(|py| {
            let object = PyTuple::new(py, [py.None(), py.None()]).unwrap().into_any();
            let udf = udf_of(object.clone().unbind());
            let parts = Parts::of(py, [PartRef::Expression(&udf)]);
            let root = parts.shared[0].clone_ref(py);
            // The expression of the UDF and its pickler, which reports the object, once.
            assert!(root.get().covered());
            assert_eq!(root.get().held.len(), 1);
            let pickler = root.get().held[0].clone_ref(py);
            assert!(pickler.get().covered());
            assert!(reports(py, &root, pickler.bind(py).as_any()));
            assert!(reports(py, &pickler, &object));
            // A clone that no object of the binding holds, as a value that a Rust structure
            // keeps: the part is not covered, so the collector sees a reference from outside.
            let clone = udf.clone();
            assert!(!root.get().covered());
            drop(clone);
            assert!(root.get().covered());
            // A second holder of the expression, of parts of its own, shares the SharedPart and
            // covers its reference; a plan of a DataFrame over the expression does too.
            let other = udf.clone();
            let other_parts = Parts::of(py, [PartRef::Expression(&other)]);
            assert!(other_parts.shared[0].is(&root));
            assert_eq!(root.get().holders.load(Ordering::Acquire), 2);
            assert!(root.get().covered());
            // An Arc of the pickler that no part holds keeps its object from being reported.
            let held = root
                .get()
                .part
                .held()
                .into_iter()
                .find_map(|part| match part {
                    PartRef::Pickler(pickler) => Some(pickler.clone()),
                    _ => None,
                })
                .expect("the pickler of the UDF");
            assert!(!pickler.get().covered());
            drop(held);
            assert!(pickler.get().covered());
            // Once the holders release the parts, the SharedParts go, and the walk of a new
            // holder makes new ones.
            drop((other_parts, parts));
            drop(root);
            // The pickler's part has no holder left, and the node of the UDF holds it: so its
            // object is not reported by it any more.
            assert_eq!(pickler.get().holders.load(Ordering::Acquire), 0);
            assert!(!pickler.get().covered());
            assert!(!reports(py, &pickler, &object));
            drop(pickler);
            let again = Parts::of(py, [PartRef::Expression(&udf)]);
            assert_eq!(again.shared.len(), 1);
            // `other` holds the expression still, of no parts: a holder that the collector does
            // not see, until it goes.
            assert!(!again.shared[0].get().covered());
            drop(other);
            assert!(again.shared[0].get().covered());
        });
    }

    #[test]
    fn a_plan_of_no_python_object_has_no_shared_part_and_is_known_while_it_is_held() {
        Python::initialize();
        Python::attach(|py| {
            let plain = Expression::function("upper", vec![Expression::column("y")]);
            let parts = Parts::of(py, [PartRef::Expression(&plain)]);
            assert!(parts.shared.is_empty());
            assert_eq!(parts.plain.len(), 1);
            let address = PartRef::Expression(&plain).address();
            assert!(matches!(known().get(&address), Some(Known::Plain(1))));
            let again = Parts::of(py, [PartRef::Expression(&plain)]);
            assert!(matches!(known().get(&address), Some(Known::Plain(2))));
            drop((parts, again));
            assert!(known().get(&address).is_none());
            // A leaf is not remembered: its walk is of itself.
            let leaf = Expression::column("z");
            let parts = Parts::of(py, [PartRef::Expression(&leaf)]);
            assert!(parts.shared.is_empty() && parts.plain.is_empty());
        });
    }

    #[test]
    fn shared_parts_of_a_deep_plan_are_made_and_dropped_without_recursing() {
        Python::initialize();
        Python::attach(|py| {
            let object = PyTuple::new(py, [py.None(), py.None()]).unwrap().into_any();
            let mut expression = udf_of(object.unbind());
            for _ in 0..20_000 {
                expression = Expression::function("abs", vec![expression]);
            }
            // On a thread of a small stack, as a part per level is made and dropped in turn.
            let parts = std::thread::Builder::new()
                .stack_size(256 * 1024)
                .spawn(move || {
                    Python::attach(|py| {
                        let parts = Parts::of(py, [PartRef::Expression(&expression)]);
                        assert_eq!(parts.shared.len(), 1);
                        drop(parts);
                        drop(expression);
                    })
                })
                .unwrap();
            join(py, parts);
        });
    }

    /// A holder of an expression that the collector sees, as an object of the binding: it reports
    /// the parts of its value, and its clear drops them.
    #[pyclass]
    struct Holder {
        value: Option<Expression>,
        parts: Parts,
    }

    #[pymethods]
    impl Holder {
        fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
            self.parts.visit(&visit)
        }

        fn __clear__(&mut self) {
            self.parts.release();
            self.value = None;
        }
    }

    /// The clones of the values of holders that a Rust structure keeps, which no object reports.
    static KEPT: std::sync::Mutex<Vec<Expression>> = std::sync::Mutex::new(Vec::new());

    /// Keep a clone of the value of `holder` (see [`KEPT`]).
    #[pyfunction]
    fn keep_a_clone(holder: PyRef<'_, Holder>) {
        if let Some(value) = &holder.value {
            KEPT.lock().unwrap().push(value.clone());
        }
    }

    #[test]
    fn a_clone_that_a_finalizer_keeps_while_the_collector_runs_keeps_what_it_reaches() {
        Python::initialize();
        Python::attach(|py| {
            let namespace = PyDict::new(py);
            namespace
                .set_item("keep_a_clone", wrap_pyfunction!(keep_a_clone, py).unwrap())
                .unwrap();
            py.run(
                c"class Finalizer:\n    def __del__(self):\n        keep_a_clone(self.holder)\n\n\
                  class Box:\n    pass\n",
                Some(&namespace),
                None,
            )
            .unwrap();
            // A reference cycle through a holder of the expression of a UDF, the object of its
            // pickler (a box) and an object of a finalizer: the finalizer keeps a clone of the
            // value of the holder in a structure that no object reports, as the collector runs.
            let boxed = namespace.get_item("Box").unwrap().unwrap().call0().unwrap();
            let udf = udf_of(boxed.clone().unbind());
            let parts = Parts::of(py, [PartRef::Expression(&udf)]);
            let holder = Py::new(
                py,
                Holder {
                    value: Some(udf),
                    parts,
                },
            )
            .unwrap();
            let finalizer = namespace
                .get_item("Finalizer")
                .unwrap()
                .unwrap()
                .call0()
                .unwrap();
            finalizer.setattr("holder", &holder).unwrap();
            let items = PyList::new(py, [holder.bind(py).as_any(), &finalizer]).unwrap();
            boxed.setattr("items", items).unwrap();
            drop((boxed, holder, finalizer));
            // The collection finds the cycle unreachable, runs the finalizer, and finds the
            // objects again with the counts of then, of the clone: so it frees none of them, and
            // the box that the clone's pickler holds keeps its items, which a clear would drop.
            wait_for_no_call_without_the_gil(py);
            py.import("gc").unwrap().call_method0("collect").unwrap();
            let kept = KEPT
                .lock()
                .unwrap()
                .pop()
                .expect("the clone of the finalizer");
            let pickler = Part::Expression(kept.clone())
                .held()
                .into_iter()
                .find_map(|part| match part {
                    PartRef::Pickler(pickler) => Some(pickler.clone()),
                    _ => None,
                })
                .expect("the pickler of the UDF");
            let boxed = crate::udf_command::python_object_of(pickler.object())
                .unwrap()
                .bind(py)
                .clone();
            let items = boxed.getattr("items").unwrap();
            assert_eq!(items.len().unwrap(), 2);
            let weak_box = py
                .import("weakref")
                .unwrap()
                .call_method1("ref", (&boxed,))
                .unwrap();
            // Once the clone goes, the cycle is collected.
            drop((pickler, boxed, items));
            drop(kept);
            wait_for_no_call_without_the_gil(py);
            py.import("gc").unwrap().call_method0("collect").unwrap();
            assert!(weak_box.call0().unwrap().is_none());
        });
    }

    #[test]
    fn a_holder_that_the_deallocation_of_a_shared_part_makes_before_its_drop_gets_one_of_its_own() {
        // The holder that a finalizer made, and whether the SharedPart was being deallocated then.
        static MADE: Mutex<Option<(Parts, bool)>> = Mutex::new(None);
        Python::initialize();
        Python::attach(|py| {
            let namespace = PyDict::new(py);
            py.run(
                c"class Finalizer:\n    def __del__(self):\n        self.make()\n",
                Some(&namespace),
                None,
            )
            .unwrap();
            let finalizer_class = namespace.get_item("Finalizer").unwrap().unwrap();
            // Tries until the finalizer runs in the deallocation, as it does unless the thread of
            // another test drops the references that PyO3 keeps first.
            for _ in 0..20 {
                let object = PyTuple::new(py, [py.None()]).unwrap().into_any();
                let udf = udf_of(object.unbind());
                // A holder of the expression that the collector does not see, as the core of a
                // native model keeps the dataset that it was fit on.
                let hidden = udf.clone();
                let address = PartRef::Expression(&udf).address();
                let parts = Parts::of(py, [PartRef::Expression(&udf)]);
                drop(udf);
                let dying = parts.shared[0].as_ptr() as usize;
                let make = pyo3::types::PyCFunction::new_closure(py, None, None, move |args, _| {
                    // SAFETY: the SharedPart lives, or is being deallocated, while a finalizer
                    // that runs in its deallocation runs.
                    let deallocated =
                        unsafe { pyo3::ffi::Py_REFCNT(dying as *mut pyo3::ffi::PyObject) } == 0;
                    let made = Parts::of(args.py(), [PartRef::Expression(&hidden)]);
                    *MADE.lock().unwrap() = Some((made, deallocated));
                })
                .unwrap();
                let finalizer = finalizer_class.call0().unwrap();
                finalizer.setattr("make", make).unwrap();
                // Dropped on a thread that is not attached: PyO3 keeps the reference, and the
                // deallocation of the next object of the binding drops it first, before it drops
                // the value of the object, whose drop forgets the SharedPart.
                let pending = finalizer.unbind();
                std::thread::spawn(move || drop(pending)).join().unwrap();
                // The last reference to the SharedPart of the expression.
                drop(parts);
                let (made, deallocated) = MADE.lock().unwrap().take().expect("the finalizer ran");
                if !deallocated {
                    continue;
                }
                if made
                    .shared
                    .iter()
                    .any(|shared| shared.as_ptr() as usize == dying)
                {
                    // Freed under the holder, which must not drop it again.
                    std::mem::forget(made);
                    panic!("the holder took the SharedPart that was being deallocated");
                }
                // A SharedPart of its own, which is known for the part, and forgotten with it.
                assert_eq!(made.shared.len(), 1);
                let own = made.shared[0].as_ptr() as usize;
                assert!(matches!(
                    known().get(&address),
                    Some(Known::Shared { object, .. }) if *object == own
                ));
                drop(made);
                assert!(known().get(&address).is_none());
                return;
            }
            panic!("the finalizer never ran in the deallocation of the SharedPart");
        });
    }

    #[test]
    fn the_counts_and_the_marks_of_calls_that_panic_are_restored_as_they_unwind() {
        Python::initialize();
        Python::attach(|py| {
            // A counted call without the GIL that panics: its count is restored as it unwinds,
            // once the GIL is taken back, as a collection may run then. Of this thread's calls,
            // as other tests count theirs.
            let depth = DEPTH.with(Cell::get);
            let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                detach(py, || {
                    assert_eq!(DEPTH.with(Cell::get), depth + 1);
                    panic!("a counted call that panics without the GIL")
                })
            }));
            assert!(unwound.is_err());
            assert_eq!(DEPTH.with(Cell::get), depth);
            // A call of the parts of a holder that panics: they are not busy any more, and are
            // reported again.
            let object = PyTuple::new(py, [py.None(), py.None()]).unwrap().into_any();
            let udf = udf_of(object.clone().unbind());
            let parts = Parts::of(py, [PartRef::Expression(&udf)]);
            let pickler = parts.shared[0].get().held[0].clone_ref(py);
            assert!(reports(py, &pickler, &object));
            let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                of_parts(py, &[&parts], || {
                    assert!(!reports_now(py, &pickler, &object));
                    panic!("a call of the parts of a holder that panics")
                })
            }));
            assert!(unwound.is_err());
            assert!(reports(py, &pickler, &object));
            // A call of the parts of values that panics without the GIL, as an ML fit runs.
            let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                detach_of_values(py, [PartRef::Expression(&udf)], || {
                    panic!("a call of the parts of values that panics without the GIL")
                })
            }));
            assert!(unwound.is_err());
            assert!(reports(py, &pickler, &object));
            drop((pickler, parts, udf));
        });
    }

    #[test]
    fn a_call_of_the_parts_of_named_holders_makes_those_parts_alone_busy() {
        Python::initialize();
        Python::attach(|py| {
            let object = PyTuple::new(py, [py.None(), py.None()]).unwrap().into_any();
            let other_object = PyTuple::new(py, [py.None()]).unwrap().into_any();
            let udf = udf_of(object.clone().unbind());
            let other = udf_of(other_object.clone().unbind());
            let parts = Parts::of(py, [PartRef::Expression(&udf)]);
            let other_parts = Parts::of(py, [PartRef::Expression(&other)]);
            let root = parts.shared[0].clone_ref(py);
            let pickler = root.get().held[0].clone_ref(py);
            let other_root = other_parts.shared[0].clone_ref(py);
            assert!(reports(py, &pickler, &object));
            // A call of the parts of `parts`, as a request of a DataFrame: they and those below
            // them are busy, and report nothing, until it ends; those of another holder report.
            {
                let _in_use = in_use(py, &[&parts]);
                assert!(root.get().busy() && pickler.get().busy());
                assert!(!other_root.get().busy());
                assert!(!reports_now(py, &root, pickler.bind(py).as_any()));
                assert!(!reports_now(py, &pickler, &object));
                assert!(reports(
                    py,
                    &other_root,
                    other_root.get().held[0].bind(py).as_any()
                ));
                // Its one call without the GIL takes its counting, of no count of any part; a
                // call that Python code of it makes is counted as any part's.
                assert!(take_counting() == Counting::Uncounted);
                assert!(take_counting() == Counting::Any);
                // Two calls of the same parts: they are busy until both end.
                let again = in_use(py, &[&parts, &parts]);
                drop(again);
                assert!(pickler.get().busy());
            }
            assert!(!root.get().busy() && !pickler.get().busy());
            assert!(take_counting() == Counting::Any);
            assert!(reports(py, &root, pickler.bind(py).as_any()));
            assert!(reports(py, &pickler, &object));
            // A call of holders of no parts of Python objects marks nothing.
            let plain = Parts::of(py, [PartRef::Expression(&Expression::column("x"))]);
            assert!(Busy::of(py, &[&plain]).is_none());
        });
    }

    #[test]
    fn a_call_of_values_makes_busy_what_they_reach_and_what_is_made_of_them_while_it_runs() {
        Python::initialize();
        Python::attach(|py| {
            let object = PyTuple::new(py, [py.None(), py.None()]).unwrap().into_any();
            let udf = udf_of(object.clone().unbind());
            let parts = Parts::of(py, [PartRef::Expression(&udf)]);
            let root = parts.shared[0].clone_ref(py);
            let pickler = root.get().held[0].clone_ref(py);
            // A value that a Rust structure holds, over the expression of a holder, as the dataset
            // of the summary of a model: its own node has no SharedPart.
            let value = Expression::function("abs", vec![udf.clone()]);
            let address = PartRef::Expression(&value).address();
            {
                let _in_use = in_use_of_values(py, [PartRef::Expression(&value)]);
                assert!(take_counting() == Counting::Uncounted);
                assert!(root.get().busy() && pickler.get().busy());
                assert!(!reports_now(py, &pickler, &object));
                // A holder of the value made while the call runs, which makes the SharedPart of
                // the node of the value: busy too, until the call ends.
                let holder = Parts::of(py, [PartRef::Expression(&value)]);
                let made = holder.shared[0].clone_ref(py);
                assert_eq!(PartRef::Expression(&value).address(), address);
                assert!(made.get().busy());
                drop(holder);
                assert!(made.get().busy());
            }
            assert!(!root.get().busy() && !pickler.get().busy());
            assert!(reports(py, &pickler, &object));
            // Values of no Python object, and of a part known as of none, mark nothing.
            let plain = Expression::function("upper", vec![Expression::column("y")]);
            let plain_parts = Parts::of(py, [PartRef::Expression(&plain)]);
            assert!(Busy::of_values(py, [PartRef::Expression(&plain)]).is_none());
            let over_plain = Expression::function("lower", vec![plain.clone()]);
            assert!(Busy::of_values(py, [PartRef::Expression(&over_plain)]).is_none());
            drop(plain_parts);
        });
    }

    #[test]
    fn the_child_of_a_fork_counts_the_calls_of_the_forking_thread_alone() {
        let _alone = of_the_check_alone();
        Python::initialize();
        Python::attach(|py| {
            let os = py.import("os").unwrap();
            if !os.hasattr("fork").unwrap() {
                return;
            }
            init(py).unwrap();
            let object = PyTuple::new(py, [py.None()]).unwrap().into_any();
            let udf = udf_of(object.clone().unbind());
            let parts = Parts::of(py, [PartRef::Expression(&udf)]);
            let root = parts.shared[0].clone_ref(py);
            let pickler = root.get().held[0].clone_ref(py);
            // Another thread in a call of any part and in a call of the parts of `parts`.
            let (entered, on_entered) = std::sync::mpsc::channel::<()>();
            let (release, on_release) = std::sync::mpsc::channel::<()>();
            let marked = parts.shared[0].clone_ref(py);
            let call = std::thread::spawn(move || {
                Python::attach(|py| {
                    let holder = Parts {
                        shared: vec![marked].into_boxed_slice(),
                        plain: Box::default(),
                    };
                    let busy = Busy::of(py, &[&holder]);
                    detach(py, move || {
                        entered.send(()).unwrap();
                        on_release.recv().unwrap();
                    });
                    drop(busy);
                    // The parts of no holder: not a holder's release.
                    std::mem::forget(holder);
                })
            });
            uncounted(py, move || on_entered.recv().unwrap());
            assert!(DETACHED.load(Ordering::SeqCst) >= 1);
            assert!(pickler.get().busy());
            let refcount = |of: &Py<SharedPart>| -> isize {
                py.import("sys")
                    .unwrap()
                    .call_method1("getrefcount", (of,))
                    .unwrap()
                    .extract()
                    .unwrap()
            };
            let references = refcount(&root);
            let child = os.call_method0("fork").unwrap().extract::<i64>().unwrap();
            if child == 0 {
                // In the child, of this thread alone: no call is counted, nothing is busy, and
                // the mark of the other thread's call, which ends not here, is dropped with its
                // reference to the SharedPart.
                let clean = DETACHED.load(Ordering::SeqCst) == 0
                    && !pickler.get().busy()
                    && marks().iter().all(|mark| mark.roots.is_empty())
                    && refcount(&root) < references;
                let _ = os.call_method1("_exit", (if clean { 0 } else { 1 },));
                unreachable!("_exit returns not");
            }
            let status = os
                .call_method1("waitpid", (child, 0))
                .unwrap()
                .get_item(1)
                .unwrap()
                .extract::<i64>()
                .unwrap();
            assert_eq!(status, 0, "the child counted calls of the parent");
            release.send(()).unwrap();
            join(py, call);
        });
    }

    /// The lock of the tests of the check of the clones and of a fork, which stops the checks of
    /// the other tests while it runs (see `_hold_checks_for_fork`).
    static OF_THE_CHECK: Mutex<()> = Mutex::new(());

    fn of_the_check_alone() -> MutexGuard<'static, ()> {
        OF_THE_CHECK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The violations of the check of the clones on the thread of the name `name`, which are
    /// taken.
    fn violations_of(name: &str) -> Vec<String> {
        _take_part_clone_violations()
            .3
            .into_iter()
            .filter(|log| log.contains(&format!("{name:?}")))
            .collect()
    }

    #[test]
    fn the_check_of_the_clones_logs_one_without_the_gil_of_a_part_that_no_call_marks_busy() {
        let _alone = of_the_check_alone();
        Python::initialize();
        Python::attach(|py| {
            assert!(_check_part_clones(py).unwrap());
            assert!(_part_clones_checked());
            let object = PyTuple::new(py, [py.None(), py.None()]).unwrap().into_any();
            let udf = udf_of(object.clone().unbind());
            let parts = Parts::of(py, [PartRef::Expression(&udf)]);
            // A clone of the expression of a holder without the GIL, in a call that names no part
            // of it: a planted violation, logged with its backtrace.
            let planted = "planted-clone-of-an-unnamed-part";
            let clone_on = |name: &'static str, udf: &Expression| {
                let udf = udf.clone();
                std::thread::Builder::new()
                    .name(name.to_string())
                    .spawn(move || drop(udf.clone()))
                    .unwrap()
            };
            let thread = clone_on(planted, &udf);
            join(py, thread);
            let logs = violations_of(planted);
            assert_eq!(logs.len(), 1, "{logs:?}");
            assert!(logs[0].contains("of an expression that has a SharedPart"));
            assert!(logs[0].contains("check_clone"), "{}", logs[0]);
            // The same clone while a call marks the parts of the holder busy, or while it is
            // counted, on its thread: none.
            {
                let _busy = Busy::of(py, &[&parts]);
                let named = "clone-of-a-named-part";
                join(py, clone_on(named, &udf));
                assert!(violations_of(named).is_empty());
            }
            let counted = "clone-in-a-counted-call";
            let udf_of_call = udf.clone();
            let thread = std::thread::Builder::new()
                .name(counted.to_string())
                .spawn(move || Python::attach(|py| detach(py, || drop(udf_of_call.clone()))))
                .unwrap();
            join(py, thread);
            assert!(violations_of(counted).is_empty());
            // A clone with the GIL: none.
            drop(udf.clone());
            drop(parts);
        });
    }

    #[test]
    fn a_fork_waits_for_the_checks_that_run_and_starts_none_until_every_fork_is_over() {
        let _alone = of_the_check_alone();
        Python::initialize();
        Python::attach(|py| {
            assert!(_check_part_clones(py).unwrap());
            let object = PyTuple::new(py, [py.None()]).unwrap().into_any();
            let udf = udf_of(object.unbind());
            let parts = Parts::of(py, [PartRef::Expression(&udf)]);
            // A thread that clones the expression without the GIL, again and again: each clone
            // is a check, which takes the locks of the known parts, of the marks and of the log.
            let stop = std::sync::Arc::new(AtomicBool::new(false));
            let (cloned, of_thread) = (udf.clone(), stop.clone());
            let cloning = std::thread::Builder::new()
                .name("checks-across-a-fork".to_string())
                .spawn(move || {
                    while !of_thread.load(Ordering::SeqCst) {
                        drop(cloned.clone());
                    }
                })
                .unwrap();
            uncounted(py, || {
                std::thread::sleep(std::time::Duration::from_millis(20))
            });
            // Before a fork: the checks that ran have ended, none takes the locks until it is
            // over, and so none is held, as a fork may then take them, with the GIL.
            _hold_checks_for_fork();
            let checked = CLONES_CHECKED.load(Ordering::SeqCst);
            for _ in 0..1000 {
                assert!(KNOWN.try_lock().is_ok() && MARKS.try_lock().is_ok());
                assert!(VIOLATION_LOG.try_lock().is_ok());
            }
            uncounted(py, || {
                std::thread::sleep(std::time::Duration::from_millis(20))
            });
            assert_eq!(CLONES_CHECKED.load(Ordering::SeqCst), checked);
            // A second fork, of another thread, starts before the first is over and ends first:
            // the first is still under way, so no check starts.
            _hold_checks_for_fork();
            _release_checks_after_fork();
            uncounted(py, || {
                std::thread::sleep(std::time::Duration::from_millis(20))
            });
            assert_eq!(CLONES_CHECKED.load(Ordering::SeqCst), checked);
            assert!(KNOWN.try_lock().is_ok() && MARKS.try_lock().is_ok());
            // After the first, the checks run again; a handler after a fork whose handler before
            // it did not run, as of a check turned on during the fork, does not count below none.
            _release_checks_after_fork();
            _release_checks_after_fork();
            assert_eq!(FORKS.load(Ordering::SeqCst), 0);
            uncounted(py, || {
                std::thread::sleep(std::time::Duration::from_millis(20))
            });
            assert!(CLONES_CHECKED.load(Ordering::SeqCst) > checked);
            stop.store(true, Ordering::SeqCst);
            join(py, cloning);
            let _ = violations_of("checks-across-a-fork");
            drop(parts);
        });
    }

    #[test]
    #[cfg(not(Py_GIL_DISABLED))]
    fn this_build_on_an_interpreter_of_the_gil_is_not_free_threaded() {
        Python::initialize();
        Python::attach(|py| assert!(!free_threaded(py)));
    }

    #[test]
    fn no_reference_is_reported_while_a_call_of_the_binding_runs_without_the_gil() {
        Python::initialize();
        Python::attach(|py| {
            let object = PyTuple::new(py, [py.None(), py.None()]).unwrap().into_any();
            let udf = udf_of(object.clone().unbind());
            let parts = Parts::of(py, [PartRef::Expression(&udf)]);
            let root = parts.shared[0].clone_ref(py);
            let pickler = root.get().held[0].clone_ref(py);
            assert!(reports(py, &root, pickler.bind(py).as_any()));
            assert!(reports(py, &pickler, &object));
            // A call that runs without the GIL, as a request that builds the plan of the UDF,
            // which clones its pickler and drops the clone: from before it releases the GIL
            // until after it takes it back, nothing is reported, whatever the count of the
            // pickler, so that the passes of a collection meanwhile get the same referents.
            let (entered, on_entered) = std::sync::mpsc::channel::<()>();
            let (release, on_release) = std::sync::mpsc::channel::<()>();
            let of_call = pickler.clone_ref(py);
            let call = std::thread::spawn(move || {
                Python::attach(|py| {
                    detach(py, move || {
                        entered.send(()).unwrap();
                        on_release.recv().unwrap();
                        let clone = of_call.get().part.clone();
                        drop(clone);
                        drop(of_call);
                    })
                })
            });
            uncounted(py, move || on_entered.recv().unwrap());
            assert!(root.get().covered() && pickler.get().covered());
            assert!(!reports_now(py, &root, pickler.bind(py).as_any()));
            assert!(!reports_now(py, &pickler, &object));
            let clone = pickler.get().part.clone();
            assert!(!pickler.get().covered() && !reports_now(py, &pickler, &object));
            drop(clone);
            assert!(pickler.get().covered() && !reports_now(py, &pickler, &object));
            release.send(()).unwrap();
            join(py, call);
            assert!(reports(py, &root, pickler.bind(py).as_any()));
            assert!(reports(py, &pickler, &object));
        });
    }

    #[test]
    fn the_check_logs_a_release_of_the_gil_and_python_code_while_a_gil_mutex_is_held() {
        let _alone = of_the_check_alone();
        Python::initialize();
        Python::attach(|py| assert!(_check_part_clones(py).unwrap()));
        let name = "gil-mutex-held";
        let thread = std::thread::Builder::new()
            .name(name.to_string())
            .spawn(|| {
                Python::attach(|py| {
                    let state = GilMutex::new(vec![1]);
                    let held = || LOCKS_HELD.with(Cell::get);
                    // A release of the GIL by the binding with no lock held: no violation.
                    detach_without_parts(py, || ());
                    check_no_lock_held("the Python code of the test");
                    {
                        let guard = state.lock(py);
                        assert_eq!((held(), guard.as_slice()), (1, &[1][..]));
                        // A lock that is held is not taken again, as a traverse finds it.
                        assert!(state.try_lock(py).is_none());
                        // While it is held: each release of the GIL by the binding, and the Python
                        // code of the objects of a state, are violations.
                        detach_without_parts(py, || ());
                        detach(py, || ());
                        check_no_lock_held("the Python code of the test");
                    }
                    assert_eq!(held(), 0);
                    let replaced = state.replace(py, vec![2]);
                    assert_eq!((replaced, held()), (vec![1], 0));
                    assert_eq!(state.try_lock(py).as_deref(), Some(&vec![2]));
                })
            })
            .unwrap();
        Python::attach(|py| join(py, thread));
        let logs = violations_of(name);
        assert_eq!(logs.len(), 3, "{logs:?}");
        assert!(logs[0].starts_with(
            "a release of the GIL by the binding while this thread holds the lock of a state of \
             options"
        ));
        assert!(logs[2].starts_with("the Python code of the test while this thread holds"));
        assert!(logs.iter().all(|log| log.contains("check_no_lock_held")));
    }

    #[test]
    fn a_gil_mutex_found_held_is_logged_and_waited_for_with_the_gil_released() {
        let _alone = of_the_check_alone();
        Python::initialize();
        Python::attach(|py| {
            assert!(_check_part_clones(py).unwrap());
            let state = std::sync::Arc::new(GilMutex::new(0u32));
            let (held, on_held) = std::sync::mpsc::channel::<()>();
            let (release, on_release) = std::sync::mpsc::channel::<()>();
            // A thread that holds the lock while it releases the GIL, against the rule of the
            // lock: a thread that then wants it must not wait for it with the GIL held, which the
            // holder needs back to release it.
            let of_holder = state.clone();
            let holder = std::thread::Builder::new()
                .name("gil-mutex-holder".to_string())
                .spawn(move || {
                    Python::attach(|py| {
                        let mut guard = of_holder.lock(py);
                        *guard += 1;
                        uncounted(py, move || {
                            held.send(()).unwrap();
                            on_release.recv().unwrap();
                        });
                        *guard += 1;
                    })
                })
                .unwrap();
            uncounted(py, move || on_held.recv().unwrap());
            let of_waiter = state.clone();
            let name = "gil-mutex-waiter";
            let waiter = std::thread::Builder::new()
                .name(name.to_string())
                .spawn(move || Python::attach(|py| *of_waiter.lock(py)))
                .unwrap();
            uncounted(py, || {
                std::thread::sleep(std::time::Duration::from_millis(50))
            });
            release.send(()).unwrap();
            join(py, holder);
            // The waiter took the lock once the holder released it, of the state that it left.
            assert_eq!(join(py, waiter), 2);
            let logs = violations_of(name);
            assert_eq!(logs.len(), 1, "{logs:?}");
            assert!(logs[0].starts_with("a lock of the state of options found held"));
        });
    }
}
