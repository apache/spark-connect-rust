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

//! The generations of the process: of the process itself, and of each child of its `fork()`s, as
//! grpcio's fork support tells them apart.
//!
//! A child of a `fork()` has none of the threads of its parent: of its runtime, of what runs on
//! it, or of the delivery of the Python binding. It gets its own of each, of its generation (see
//! [`generation`]), and leaves those of its parent as they are, never dropped there: a drop of an
//! object of the parent, such as the socket of a connection, would remove the socket from the
//! epoll instance that the child shares with its parent, and so stop the parent's runtime from
//! reading it ([`ForkLocal`]). The descriptors that the parent's objects hold are closed in the
//! child instead, as the fork returns, without a call on the epoll instance: the sockets of the
//! connections, and the epoll instances of the runtimes, which a plain `close` takes from the child
//! only. The eventfd of a runtime of the parent, which an object of the parent writes to when it
//! wakes a task of that runtime, is made an eventfd of the child's own, rather than closed: a
//! descriptor that the child opens later could take its number, and get those writes.
//!
//! The state that the threads of the process keep under a lock is of its generation too
//! ([`PerGeneration`]): a lock that a thread of the parent held at the fork is never released in
//! the child, whose own state is a fresh one.

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::Mutex;

/// The generation of the process: 0, and one more in each child of a `fork()`.
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// The generation of the process (see the module), of which the state of the process that its
/// threads keep is. Not a stable API.
#[doc(hidden)]
pub fn generation() -> u64 {
    watch_forks();
    current()
}

/// The generation of the process, as the handler of the child reads it.
fn current() -> u64 {
    GENERATION.load(Ordering::SeqCst)
}

/// Register the handler of `fork()` in the child, once per process, which makes the child of
/// another generation and closes what the objects of its parent hold. Not a stable API.
#[doc(hidden)]
pub fn watch_forks() {
    #[cfg(unix)]
    {
        // A flag rather than a `Once`: a `Once` that a thread runs while another forks would be
        // running forever in the child.
        static WATCHING: AtomicBool = AtomicBool::new(false);
        if !WATCHING.load(Ordering::Acquire) && !WATCHING.swap(true, Ordering::AcqRel) {
            // SAFETY: registers a handler of the child, which makes only calls that a child may.
            unsafe {
                libc::pthread_atfork(None, None, Some(forked_child));
            }
        }
    }
}

/// The handler of `fork()` in the child: of another generation, it closes the descriptors that the
/// objects of its parent hold (see the module). It waits for no lock, as a thread of the parent
/// may have held one at the fork, nor allocates.
#[cfg(unix)]
extern "C" fn forked_child() {
    GENERATION.fetch_add(1, Ordering::SeqCst);
    close_inherited();
}

/// A value of the generation of the process that made it, which a drop in another generation
/// leaves as it is, never dropped: in a child of a `fork()`, an object of the parent, such as a
/// lease of a channel, a stream of a call or the socket of a connection, is of the runtime of the
/// parent, whose epoll instance, locks and tasks the child must not touch (see the module). Not a
/// stable API.
#[doc(hidden)]
pub struct ForkLocal<T> {
    generation: u64,
    value: ManuallyDrop<T>,
}

impl<T> ForkLocal<T> {
    /// `value`, of the generation of the process.
    pub fn new(value: T) -> Self {
        Self::of_generation(generation(), value)
    }

    fn of_generation(generation: u64, value: T) -> Self {
        Self {
            generation,
            value: ManuallyDrop::new(value),
        }
    }

    /// Whether the value is of the generation of the process, which drops it.
    pub fn is_current(&self) -> bool {
        self.generation == current()
    }
}

impl<T> std::ops::Deref for ForkLocal<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> std::ops::DerefMut for ForkLocal<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T> Drop for ForkLocal<T> {
    fn drop(&mut self) {
        if self.is_current() {
            // SAFETY: dropped once, here, and never used after.
            unsafe { ManuallyDrop::drop(&mut self.value) };
        }
    }
}

/// A value of each generation of the process: a child of a fork makes its own at its first use,
/// whose locks no thread of its parent holds, and leaves that of its parent as it is, never
/// dropped. The value of a generation lives as long as the container. Not a stable API.
#[doc(hidden)]
pub struct PerGeneration<T> {
    current: AtomicPtr<Generation<T>>,
}

/// The value of a generation, and that of the generation before it, of the parent, if any.
struct Generation<T> {
    generation: u64,
    value: T,
    parent: *mut Generation<T>,
}

// SAFETY: the values are shared by the threads of the process as those of a `Mutex<T>`'s are,
// and a value is dropped only by the drop of the container.
unsafe impl<T: Send + Sync> Sync for PerGeneration<T> {}
unsafe impl<T: Send> Send for PerGeneration<T> {}

impl<T> PerGeneration<T> {
    /// A container of no value yet.
    pub const fn new() -> Self {
        Self {
            current: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// The value of the generation of the process: at its first use in the generation, a value of
    /// `init`, which is given the value of the generation before it, of the parent, if the
    /// container has one. Threads of the generation that make one at the same time make one
    /// each, and all but one are dropped.
    pub fn get_or_init(&self, init: impl FnMut(Option<&T>) -> T) -> &T {
        self.of(generation(), init)
    }

    /// The value of `generation`, as [`PerGeneration::get_or_init`] gives it.
    fn of(&self, generation: u64, mut init: impl FnMut(Option<&T>) -> T) -> &T {
        loop {
            let current = self.current.load(Ordering::Acquire);
            // SAFETY: a value that was set lives as long as the container.
            let parent = unsafe { current.as_ref() };
            if let Some(parent) = parent {
                if parent.generation == generation {
                    return &parent.value;
                }
            }
            let made = Box::into_raw(Box::new(Generation {
                generation,
                value: init(parent.map(|parent| &parent.value)),
                parent: current,
            }));
            match self
                .current
                .compare_exchange(current, made, Ordering::AcqRel, Ordering::Acquire)
            {
                // SAFETY: set, so it lives as long as the container; that of the parent too.
                Ok(_) => return unsafe { &(*made).value },
                // Another thread of the generation set one first, which this one gives up.
                // SAFETY: of the box made above, which no other thread has seen.
                Err(_) => drop(unsafe { Box::from_raw(made) }),
            }
        }
    }

    /// The values of the generations other than `generation`, newest first: those of the
    /// parents of a child. Takes no lock.
    #[cfg_attr(not(unix), allow(dead_code))]
    fn others(&self, generation: u64) -> impl Iterator<Item = &T> {
        let mut next = self.current.load(Ordering::Acquire);
        std::iter::from_fn(move || loop {
            // SAFETY: a value that was set lives as long as the container.
            let value = unsafe { next.as_ref() }?;
            next = value.parent;
            if value.generation != generation {
                return Some(&value.value);
            }
        })
    }
}

impl<T> Default for PerGeneration<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for PerGeneration<T> {
    /// Drops the value of the generation of the process; those of its parents are left as they
    /// are.
    fn drop(&mut self) {
        let current = *self.current.get_mut();
        if current.is_null() {
            return;
        }
        // SAFETY: the value of the container, which no one uses any more.
        let value = unsafe { Box::from_raw(current) };
        if value.generation == crate::fork::current() {
            drop(value);
        } else {
            std::mem::forget(value);
        }
    }
}

/// A value that the threads of the process share, which they read without a lock and replace
/// under a lock of their generation: a thread of the parent of a `fork()` that reads or replaces
/// it at the fork leaves nothing locked in the child, whose threads read and replace the value as
/// the parent had it then, as the state of a session that the background work of the session
/// reads, which a child keeps. A replaced value is freed by a later replacement once no thread
/// reads the value, or with the container. Not a stable API.
#[doc(hidden)]
pub struct Shared<T> {
    current: AtomicPtr<T>,
    /// The replaced values, which no thread may read once none reads the value.
    retired: AtomicPtr<Retired<T>>,
    /// How many threads read the value now: of a thread of a parent that read it at the fork too,
    /// so that a child never frees a replaced value then, but with the container.
    readers: std::sync::atomic::AtomicUsize,
    /// The lock of the replacements of each generation.
    replacing: PerGeneration<Mutex<()>>,
}

/// A replaced value of a [`Shared`], and the one replaced before it.
struct Retired<T> {
    value: *mut T,
    next: *mut Retired<T>,
}

// SAFETY: the values are shared by the threads of the process as those of an `RwLock<T>` are,
// and a value is dropped only by the drop of the container.
unsafe impl<T: Send + Sync> Sync for Shared<T> {}
unsafe impl<T: Send> Send for Shared<T> {}

impl<T> Shared<T> {
    /// A container of `value`.
    pub fn new(value: T) -> Self {
        Self {
            current: AtomicPtr::new(Box::into_raw(Box::new(value))),
            retired: AtomicPtr::new(std::ptr::null_mut()),
            readers: std::sync::atomic::AtomicUsize::new(0),
            replacing: PerGeneration::new(),
        }
    }

    /// What `read` makes of the value, without a lock.
    pub fn read<R>(&self, read: impl FnOnce(&T) -> R) -> R {
        /// Counts the read until it ends, also when it unwinds.
        struct Reading<'a>(&'a std::sync::atomic::AtomicUsize);

        impl Drop for Reading<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }

        // Counted before the value is loaded: see `update`.
        self.readers.fetch_add(1, Ordering::SeqCst);
        let _reading = Reading(&self.readers);
        // SAFETY: a value that a reader loads is freed only once no thread reads (see `update`).
        read(unsafe { &*self.current.load(Ordering::SeqCst) })
    }

    /// A clone of the value, without a lock.
    pub fn get(&self) -> T
    where
        T: Clone,
    {
        self.read(T::clone)
    }

    /// Replace the value with `value`.
    pub fn set(&self, value: T) {
        let mut value = Some(value);
        self.update(|_| value.take());
    }

    /// Replace the value with what `update` makes of it, if it makes one, as one step with the
    /// other replacements: whether it replaced it.
    pub fn update(&self, update: impl FnOnce(&T) -> Option<T>) -> bool {
        let unread = {
            let _replacing = self
                .replacing
                .get_or_init(|_| Mutex::new(()))
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let current = self.current.load(Ordering::SeqCst);
            // SAFETY: the value, which only a replacement, which waits for this one, frees.
            let Some(value) = update(unsafe { &*current }) else {
                return false;
            };
            self.current
                .store(Box::into_raw(Box::new(value)), Ordering::SeqCst);
            let retired = Box::into_raw(Box::new(Retired {
                value: current,
                next: std::ptr::null_mut(),
            }));
            loop {
                let next = self.retired.load(Ordering::SeqCst);
                // SAFETY: of the box made above, which no other thread has seen.
                unsafe { (*retired).next = next };
                if self
                    .retired
                    .compare_exchange(next, retired, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    break;
                }
            }
            // A thread that reads a replaced value counted itself before it loaded it, before the
            // value was replaced: once no thread reads, none reads a replaced value, nor will.
            if self.readers.load(Ordering::SeqCst) == 0 {
                self.retired.swap(std::ptr::null_mut(), Ordering::SeqCst)
            } else {
                std::ptr::null_mut()
            }
        };
        // SAFETY: the replaced values, which no thread reads any more, freed once the lock is
        // released, as the drop of a value may run code that replaces the value again.
        unsafe { free(unread) };
        true
    }
}

/// Free the replaced values of the list `next`.
///
/// # Safety
///
/// No thread reads them, nor will.
unsafe fn free<T>(mut next: *mut Retired<T>) {
    while !next.is_null() {
        let retired = Box::from_raw(next);
        drop(Box::from_raw(retired.value));
        next = retired.next;
    }
}

impl<T: Default> Default for Shared<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Shared<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.read(|value| f.debug_tuple("Shared").field(value).finish())
    }
}

impl<T> Drop for Shared<T> {
    fn drop(&mut self) {
        // SAFETY: the values of the container, which no one reads any more.
        unsafe {
            drop(Box::from_raw(*self.current.get_mut()));
            free(*self.retired.get_mut());
        }
    }
}

/// The open sockets of the connections of each generation, which a child closes.
#[cfg(unix)]
static SOCKETS: PerGeneration<Mutex<Vec<std::os::fd::RawFd>>> = PerGeneration::new();

/// The sockets of the connections of this generation.
#[cfg(unix)]
fn sockets() -> std::sync::MutexGuard<'static, Vec<std::os::fd::RawFd>> {
    SOCKETS
        .get_or_init(|_| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// A socket of a connection of this generation is open, `fd`, which a child of a fork closes.
#[cfg(unix)]
pub(crate) fn opened(fd: std::os::fd::RawFd) {
    sockets().push(fd);
}

/// The socket `fd` of a connection of this generation is about to be closed.
#[cfg(unix)]
pub(crate) fn closing(fd: std::os::fd::RawFd) {
    let mut sockets = sockets();
    if let Some(position) = sockets.iter().position(|&open| open == fd) {
        sockets.swap_remove(position);
    }
}

/// A descriptor of a runtime: of its epoll instance, or of the eventfd that it watches.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Anonymous {
    Epoll,
    EventFd,
}

/// The descriptors of the runtimes of each generation, which a child closes.
#[cfg(target_os = "linux")]
static RUNTIME_FDS: PerGeneration<Mutex<Vec<(std::os::fd::RawFd, Anonymous)>>> =
    PerGeneration::new();

/// Record the descriptors of `runtime`, the shared runtime of this generation, which a child of a
/// fork closes (see [`runtime_fds`]).
pub(crate) fn record_runtime(runtime: &tokio::runtime::Runtime) {
    #[cfg(target_os = "linux")]
    {
        let fds = runtime_fds(runtime);
        RUNTIME_FDS
            .get_or_init(|_| Mutex::new(Vec::new()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend(fds);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = runtime;
}

/// The descriptors of `runtime`: those of its epoll instance, which watches a probe that is
/// registered with its reactor for the moment, its own and that of its registry, and the eventfd
/// that they watch, its waker. Of no other runtime, whatever another thread makes or drops
/// meanwhile, as the probe is watched by this one alone.
#[cfg(target_os = "linux")]
fn runtime_fds(runtime: &tokio::runtime::Runtime) -> Vec<(std::os::fd::RawFd, Anonymous)> {
    use std::os::fd::AsRawFd;
    let probe = {
        let _entered = runtime.enter();
        match tokio::net::UnixStream::pair() {
            Ok(probe) => probe,
            Err(_) => return Vec::new(),
        }
    };
    let probe_fd = probe.0.as_raw_fd();
    let Some(probe_inode) = inode_of(probe_fd) else {
        return Vec::new();
    };
    let mut epolls = Vec::new();
    let mut eventfds = Vec::new();
    for (fd, kind) in anonymous_fds() {
        if kind != Anonymous::Epoll {
            continue;
        }
        let watched = watched(fd);
        if !watched.contains(&(probe_fd, probe_inode)) {
            continue;
        }
        epolls.push((fd, Anonymous::Epoll));
        for (watched_fd, _) in watched {
            if is_anonymous(watched_fd, Anonymous::EventFd) && !eventfds.contains(&watched_fd) {
                eventfds.push(watched_fd);
            }
        }
    }
    drop(probe);
    epolls.extend(eventfds.into_iter().map(|fd| (fd, Anonymous::EventFd)));
    epolls
}

/// The inode of the file of `fd`, which tells one socket from another.
#[cfg(target_os = "linux")]
#[allow(clippy::useless_conversion)] // `ino_t` is of 32 bits on some targets
fn inode_of(fd: std::os::fd::RawFd) -> Option<u64> {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `fstat` writes the status of `fd` to `stat`, which is read only once it did.
    unsafe {
        (libc::fstat(fd, stat.as_mut_ptr()) == 0).then(|| u64::from(stat.assume_init().st_ino))
    }
}

/// The descriptors of the process of an epoll instance or of an eventfd.
#[cfg(target_os = "linux")]
fn anonymous_fds() -> Vec<(std::os::fd::RawFd, Anonymous)> {
    let Ok(entries) = std::fs::read_dir("/proc/self/fd") else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let fd = entry.file_name().to_str()?.parse().ok()?;
            let kind = match std::fs::read_link(entry.path()).ok()?.to_str()? {
                "anon_inode:[eventpoll]" => Anonymous::Epoll,
                "anon_inode:[eventfd]" => Anonymous::EventFd,
                _ => return None,
            };
            Some((fd, kind))
        })
        .collect()
}

/// The descriptors that the epoll instance `fd` watches, and the inodes of their files, as its
/// `fdinfo` lists them (`tfd: <fd> ... ino:<inode, in hex>`).
#[cfg(target_os = "linux")]
fn watched(fd: std::os::fd::RawFd) -> Vec<(std::os::fd::RawFd, u64)> {
    std::fs::read_to_string(format!("/proc/self/fdinfo/{fd}"))
        .map(|info| {
            info.lines()
                .filter_map(|line| {
                    let mut fields = line.trim_start().strip_prefix("tfd:")?.split_whitespace();
                    let watched = fields.next()?.parse().ok()?;
                    let inode = fields.find_map(|field| field.strip_prefix("ino:"))?;
                    Some((watched, u64::from_str_radix(inode, 16).ok()?))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Close, in a child of a fork, the descriptors that the objects of its parents hold: the sockets
/// of their connections and the epoll instances of their runtimes, with a plain `close`, which
/// takes them from the child only, and make each eventfd of their runtimes an eventfd of the
/// child's own (see the module). A list that a thread of the parent held at the fork is left as it
/// is, with its descriptors, and so is a descriptor that is not of its kind any more.
#[cfg(unix)]
fn close_inherited() {
    let generation = current();
    for sockets in SOCKETS.others(generation) {
        let Some(mut sockets) = try_lock(sockets) else {
            continue;
        };
        for fd in sockets.drain(..) {
            if is_socket(fd) {
                // SAFETY: a socket of the parent, whose object is never dropped in the child.
                unsafe { libc::close(fd) };
            }
        }
    }
    #[cfg(target_os = "linux")]
    for fds in RUNTIME_FDS.others(generation) {
        let Some(mut fds) = try_lock(fds) else {
            continue;
        };
        for (fd, kind) in fds.drain(..) {
            if !is_anonymous(fd, kind) {
                continue;
            }
            match kind {
                // SAFETY: of a runtime of the parent, which is never used in the child.
                Anonymous::Epoll => unsafe {
                    libc::close(fd);
                },
                // SAFETY: the descriptor stays an eventfd, of the child's own.
                Anonymous::EventFd => unsafe {
                    let own = libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK);
                    if own >= 0 {
                        libc::dup3(own, fd, libc::O_CLOEXEC);
                        libc::close(own);
                    }
                },
            }
        }
    }
}

/// Whether `fd` is a socket, without allocating.
#[cfg(unix)]
fn is_socket(fd: std::os::fd::RawFd) -> bool {
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `fstat` writes the status of `fd` to `stat`, which is read only once it did.
    unsafe {
        libc::fstat(fd, stat.as_mut_ptr()) == 0
            && stat.assume_init().st_mode & libc::S_IFMT == libc::S_IFSOCK
    }
}

/// Whether `fd` is a descriptor of the kind `kind`, without allocating: the name of its link in
/// `/proc/self/fd`.
#[cfg(target_os = "linux")]
fn is_anonymous(fd: std::os::fd::RawFd, kind: Anonymous) -> bool {
    const PREFIX: &[u8] = b"/proc/self/fd/";
    let Ok(mut number) = u32::try_from(fd) else {
        return false;
    };
    let mut path = [0u8; 32];
    path[..PREFIX.len()].copy_from_slice(PREFIX);
    let mut digits = [0u8; 10];
    let mut count = 0;
    loop {
        digits[count] = b'0' + (number % 10) as u8;
        number /= 10;
        count += 1;
        if number == 0 {
            break;
        }
    }
    for (place, digit) in path[PREFIX.len()..]
        .iter_mut()
        .zip(digits[..count].iter().rev())
    {
        *place = *digit;
    }
    let expected: &[u8] = match kind {
        Anonymous::Epoll => b"anon_inode:[eventpoll]",
        Anonymous::EventFd => b"anon_inode:[eventfd]",
    };
    let mut link = [0u8; 32];
    // SAFETY: `path` ends with a NUL, and `readlink` writes at most `link.len()` bytes to `link`.
    let read =
        unsafe { libc::readlink(path.as_ptr().cast(), link.as_mut_ptr().cast(), link.len()) };
    usize::try_from(read).is_ok_and(|read| &link[..read] == expected)
}

/// The guard of `mutex` if no thread holds it, else None, without waiting.
#[cfg(unix)]
fn try_lock<T>(mutex: &Mutex<T>) -> Option<std::sync::MutexGuard<'_, T>> {
    match mutex.try_lock() {
        Ok(guard) => Some(guard),
        Err(std::sync::TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Of generations far from the process's own, which the other tests do not reach.
    const FIRST: u64 = 1_000_001;

    #[test]
    fn a_generation_has_a_value_of_its_own_and_leaves_those_of_its_parents() {
        let values: PerGeneration<Mutex<Vec<u64>>> = PerGeneration::new();
        let fresh = |_: Option<&Mutex<Vec<u64>>>| Mutex::new(Vec::new());
        values.of(FIRST, fresh).lock().unwrap().push(1);
        assert_eq!(*values.of(FIRST, fresh).lock().unwrap(), [1]);
        // The value of the next generation is a fresh one, whose lock no one holds, also while a
        // thread of the parent holds the parent's.
        let held = values.of(FIRST, fresh).lock().unwrap();
        assert!(values.of(FIRST + 1, fresh).try_lock().unwrap().is_empty());
        drop(held);
        values.of(FIRST + 1, fresh).lock().unwrap().push(2);
        let others = |generation| -> Vec<Vec<u64>> {
            values
                .others(generation)
                .map(|value| value.lock().unwrap().clone())
                .collect()
        };
        assert_eq!(others(FIRST + 1), [vec![1]]);
        values.of(FIRST + 2, fresh).lock().unwrap().push(3);
        assert_eq!(others(FIRST + 2), [vec![2], vec![1]]);
        // A value is made of that of its parent.
        let counts: PerGeneration<u64> = PerGeneration::new();
        assert_eq!(*counts.of(FIRST, |parent| parent.map_or(0, |n| n + 1)), 0);
        assert_eq!(
            *counts.of(FIRST + 1, |parent| parent.map_or(0, |n| n + 1)),
            1
        );
    }

    /// Counts its drops.
    struct Counted(Arc<std::sync::atomic::AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn a_value_of_another_generation_is_never_dropped() {
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        drop(ForkLocal::new(Counted(drops.clone())));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        // A value of a parent, as a child of a fork has it, is left as it is.
        let parent = ForkLocal::of_generation(current() + FIRST, Counted(drops.clone()));
        assert!(!parent.is_current());
        drop(parent);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        // So is the value of a parent in a container, and that of the generation is dropped.
        let values: PerGeneration<Counted> = PerGeneration::new();
        values.of(current() + FIRST, |_| Counted(drops.clone()));
        drop(values);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let values: PerGeneration<Counted> = PerGeneration::new();
        values.of(current() + FIRST, |_| Counted(drops.clone()));
        values.get_or_init(|_| Counted(drops.clone()));
        drop(values);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_shared_value_is_read_without_a_lock_and_its_replaced_values_are_freed_once_unread() {
        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let shared = Shared::new((1, Counted(drops.clone())));
        assert_eq!(shared.read(|(n, _)| *n), 1);
        // A reader of the replaced value still reads it.
        shared.read(|first| {
            shared.set((2, Counted(drops.clone())));
            assert_eq!(first.0, 1);
            assert_eq!(shared.read(|(n, _)| *n), 2);
        });
        // Read while it was replaced, so kept; and a replacement that makes no value frees none.
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(!shared.update(|_| None));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        // A replacement while no thread reads frees the values that were replaced.
        assert!(shared.update(|(n, _)| Some((n + 1, Counted(drops.clone())))));
        assert_eq!(shared.read(|(n, _)| *n), 3);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        drop(shared);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
        // Replaced by the threads of the process, each of its own value, one at a time.
        let shared = Arc::new(Shared::new(0_u64));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let shared = Arc::clone(&shared);
                std::thread::spawn(move || {
                    for _ in 0..1000 {
                        shared.update(|n| Some(n + 1));
                        assert!(shared.get() >= 1);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(shared.get(), 8000);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_descriptors_of_a_runtime_are_its_epoll_instances_and_the_eventfd_that_they_watch() {
        // Of this runtime alone, whatever runtimes the other tests make and drop meanwhile: its
        // epoll instance, its own descriptor and that of its registry, which watch what is
        // registered with its reactor, and the eventfd that they watch.
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let fds = runtime_fds(&runtime);
        let epolls: Vec<_> = fds
            .iter()
            .filter(|(_, kind)| *kind == Anonymous::Epoll)
            .map(|(fd, _)| *fd)
            .collect();
        let eventfds: Vec<_> = fds
            .iter()
            .filter(|(_, kind)| *kind == Anonymous::EventFd)
            .map(|(fd, _)| *fd)
            .collect();
        assert!(!epolls.is_empty(), "{fds:?}");
        assert_eq!(eventfds.len(), 1, "{fds:?}");
        for (fd, kind) in &fds {
            assert!(is_anonymous(*fd, *kind), "{fd} is not of {kind:?}");
            assert!(!is_socket(*fd));
        }
        let probe = {
            let _entered = runtime.enter();
            tokio::net::UnixStream::pair().unwrap()
        };
        use std::os::fd::AsRawFd;
        let probe = (
            probe.0.as_raw_fd(),
            inode_of(probe.0.as_raw_fd()).unwrap(),
            probe,
        );
        for epoll in &epolls {
            let watched = watched(*epoll);
            assert!(watched.contains(&(probe.0, probe.1)), "{watched:?}");
            assert!(
                watched.iter().any(|(fd, _)| *fd == eventfds[0]),
                "{watched:?}"
            );
        }
        // And of no other runtime, whose epoll instance does not watch the probe.
        let other = tokio::runtime::Runtime::new().unwrap();
        let others = runtime_fds(&other);
        assert!(
            others.iter().all(|fd| !fds.contains(fd)),
            "{others:?} {fds:?}"
        );
        // As the list of a child has them: a descriptor of another kind is not one of them.
        assert!(!is_anonymous(epolls[0], Anonymous::EventFd));
        drop(probe);
    }

    #[cfg(unix)]
    #[test]
    fn a_socket_is_told_apart_from_the_other_descriptors() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        assert!(is_socket(std::os::fd::AsRawFd::as_raw_fd(&listener)));
        let file = std::fs::File::open("/dev/null").unwrap();
        assert!(!is_socket(std::os::fd::AsRawFd::as_raw_fd(&file)));
        assert!(!is_socket(-1));
    }
}
