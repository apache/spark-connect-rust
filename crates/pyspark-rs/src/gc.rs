//! What Python's garbage collector may see of the objects of the binding.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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

/// The holders of the value of an object, of which its `__traverse__` reports what the value
/// alone holds to Python's garbage collector: the object itself, and the objects that keep the
/// object alive and hold a clone of its value, each with a [`Share`] of these holders.
///
/// A Python object of PySpark holds its parts as Python objects, which the collector sees through
/// each holder: PySpark's grouped data holds the DataFrame that it groups, whose plan holds the
/// functions of its UDFs. Here the grouped data holds the DataFrame and a clone of its plan, an
/// `Arc` that the two share. The collector counts each reference once, so the objects that the
/// plan holds are reported by one of its holders, the DataFrame, when each other holder of the
/// plan is such a clone: as each of those keeps the DataFrame alive and reports it, an object of
/// the plan is reachable through the DataFrame whenever it is reachable at all, and the collector
/// frees it only with all of them, as PySpark's. The grouped data, which may be unreachable while
/// the DataFrame is not, must not report it. A plan that some other holder shares, such as a
/// DataFrame derived from it, is reported by none, which keeps a cycle through it alive rather
/// than let the collector free a reachable object. A share is made with its clone and dropped
/// with it, so that the count never takes a holder for a clone that is gone.
pub(crate) struct Holders(Arc<()>);

impl Holders {
    /// The holders of a value that no clone shares yet: the object itself.
    pub(crate) fn new() -> Self {
        Holders(Arc::new(()))
    }

    /// The number of the holders, the object and each share.
    pub(crate) fn count(&self) -> usize {
        Arc::strong_count(&self.0)
    }

    /// A share of these holders, for a clone of the value that an object keeping this object
    /// holds.
    pub(crate) fn share(&self) -> Share {
        Share(Some(Arc::clone(&self.0)))
    }
}

/// One of the [`Holders`] of the value of an object, which an object that keeps that object holds
/// with a clone of the value: it is made with the clone, and dropped with it, before it. A share
/// of no holders is that of a clone that is not counted, which keeps the value from being
/// reported while the clone lives.
pub(crate) struct Share(Option<Arc<()>>);

impl Share {
    /// The share of a clone that is not counted.
    pub(crate) fn none() -> Self {
        Share(None)
    }

    /// Another share of the same holders, for another clone of the same value, which an object
    /// that keeps the holder of this share holds.
    pub(crate) fn again(&self) -> Share {
        Share(self.0.clone())
    }
}
