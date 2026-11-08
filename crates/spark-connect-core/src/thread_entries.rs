//! Entries of a map per thread, which end with their thread.
//!
//! PySpark keeps the per-thread state of its client in a `threading.local()`, which is freed
//! when its thread ends. A map from thread to entry must do the same, or every thread that ends
//! leaves its entry behind. [`forget_when_thread_ends`] registers the current thread with the
//! map when it inserts its entry, and the entry is removed from the map when the thread ends.
//! The registration holds the map only weakly, so a map that is dropped is not kept alive.

use std::cell::RefCell;
use std::sync::{Arc, Mutex, Weak};
use std::thread::ThreadId;

/// A map that a thread has an entry of, which is removed when the thread ends.
trait Registered {
    fn is_alive(&self) -> bool;
    fn address(&self) -> usize;
    fn forget(&self, thread: ThreadId);
}

struct Owned<T: Send + 'static> {
    owner: Weak<Mutex<T>>,
    forget: fn(&mut T, ThreadId),
}

impl<T: Send + 'static> Registered for Owned<T> {
    fn is_alive(&self) -> bool {
        self.owner.strong_count() > 0
    }

    fn address(&self) -> usize {
        self.owner.as_ptr() as usize
    }

    fn forget(&self, thread: ThreadId) {
        if let Some(owner) = self.owner.upgrade() {
            let mut value = owner.lock().unwrap_or_else(|e| e.into_inner());
            (self.forget)(&mut value, thread);
        }
    }
}

/// The maps that the current thread has entries of. Its drop, when the thread ends, removes
/// those entries.
#[derive(Default)]
struct ThreadOwners {
    thread: Option<ThreadId>,
    owners: Vec<Box<dyn Registered>>,
}

impl Drop for ThreadOwners {
    fn drop(&mut self) {
        if let Some(thread) = self.thread {
            for owner in &self.owners {
                owner.forget(thread);
            }
        }
    }
}

thread_local! {
    static OWNERS: RefCell<ThreadOwners> = RefCell::new(ThreadOwners::default());
}

/// Register the current thread with `owner`, a map that the thread has just inserted its entry
/// of: when the thread ends, `forget` removes that entry from the map, which is then no longer
/// kept alive by the registration. Registering again, as after the entry was removed and
/// inserted once more, is a no-op.
pub fn forget_when_thread_ends<T: Send + 'static>(
    owner: &Arc<Mutex<T>>,
    forget: fn(&mut T, ThreadId),
) {
    let thread = std::thread::current().id();
    let address = Arc::as_ptr(owner) as usize;
    // Thread-local storage that is being destroyed has no registrations left to make.
    let _ = OWNERS.try_with(|owners| {
        let mut owners = owners.borrow_mut();
        owners.owners.retain(|owner| owner.is_alive());
        owners.thread.get_or_insert(thread);
        if !owners.owners.iter().any(|owner| owner.address() == address) {
            owners.owners.push(Box::new(Owned {
                owner: Arc::downgrade(owner),
                forget,
            }));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::mpsc;

    type Map = Arc<Mutex<HashMap<ThreadId, u32>>>;

    fn forget(map: &mut HashMap<ThreadId, u32>, thread: ThreadId) {
        map.remove(&thread);
    }

    fn insert_and_register(map: &Map, value: u32) {
        let thread = std::thread::current().id();
        map.lock().unwrap().insert(thread, value);
        forget_when_thread_ends(map, forget);
    }

    #[test]
    fn entries_of_finished_threads_are_removed() {
        let map: Map = Arc::default();
        for _ in 0..1000 {
            let map = map.clone();
            std::thread::spawn(move || insert_and_register(&map, 1))
                .join()
                .unwrap();
        }
        assert!(map.lock().unwrap().is_empty());
        let owners = OWNERS.with(|owners| owners.borrow().owners.len());
        assert_eq!(owners, 0);
    }

    #[test]
    fn entry_of_a_living_thread_stays_until_it_ends() {
        let map: Map = Arc::default();
        let (inserted_tx, inserted_rx) = mpsc::channel();
        let (end_tx, end_rx) = mpsc::channel::<()>();
        let child = {
            let map = map.clone();
            std::thread::spawn(move || {
                insert_and_register(&map, 7);
                // Registering again is a no-op: the thread is registered once with the map.
                forget_when_thread_ends(&map, forget);
                let owners = OWNERS.with(|owners| owners.borrow().owners.len());
                inserted_tx.send(owners).unwrap();
                end_rx.recv().unwrap();
            })
        };
        assert_eq!(inserted_rx.recv().unwrap(), 1);
        assert_eq!(
            map.lock().unwrap().values().copied().collect::<Vec<_>>(),
            [7]
        );
        end_tx.send(()).unwrap();
        child.join().unwrap();
        assert!(map.lock().unwrap().is_empty());
    }

    #[test]
    fn registration_does_not_keep_a_map_alive() {
        let map: Map = Arc::default();
        let weak = Arc::downgrade(&map);
        std::thread::spawn(move || insert_and_register(&map, 1))
            .join()
            .unwrap();
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn the_registration_of_a_dropped_map_is_pruned() {
        std::thread::spawn(|| {
            let first: Map = Arc::default();
            insert_and_register(&first, 1);
            drop(first);
            let second: Map = Arc::default();
            insert_and_register(&second, 2);
            let owners = OWNERS.with(|owners| owners.borrow().owners.len());
            assert_eq!(owners, 1);
        })
        .join()
        .unwrap();
    }
}
