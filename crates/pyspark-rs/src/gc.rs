//! What Python's garbage collector may see of the objects of the binding.

use std::sync::atomic::{AtomicBool, Ordering};

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
