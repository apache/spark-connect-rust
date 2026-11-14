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

//! Typed JNI implementation of the shared engine contract.
//! No Spark Connect RPC handler or request/response envelope crosses this boundary.
mod zero_copy;
#[cfg(test)]
mod zero_copy_tests;

use super::{take_buffer, BridgeBuffer, MAX_BYTES};
use arrow::{array::ffi::FFI_ArrowArray, datatypes::SchemaRef, ipc::reader::StreamReader};
use futures::stream;
use prost::Message;
use spark_connect_proto as proto;
use spark_connect_server::engine::{
    BatchStream, ConfigResult, Engine, EngineSession, PreparedOperation, PreparedQuery,
    SessionContext,
};
use std::{
    collections::HashMap,
    ffi::c_void,
    io::Cursor,
    marker::PhantomData,
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering},
        mpsc, Arc, Condvar, Mutex, OnceLock, Weak,
    },
    time::Duration,
};
use tokio::sync::{
    OnceCell, OwnedSemaphorePermit, RwLock, RwLockReadGuard, RwLockWriteGuard, Semaphore,
};
use tokio::time::Instant;
use tonic::Status;

const SESSION_ADMISSION_TIMEOUT: Duration = Duration::from_secs(2);
const CLEANUP_ATTACHMENT_IDLE: Duration = Duration::from_millis(10);

extern "C" {
    fn spark_connect_engine_thread_attach(
        engine: *mut c_void,
        error: *mut BridgeBuffer,
    ) -> *mut c_void;
    fn spark_connect_engine_thread_detach(attachment: *mut c_void, error: *mut BridgeBuffer)
        -> i32;
    fn spark_connect_engine_task_new(engine: *mut c_void, error: *mut BridgeBuffer) -> *mut c_void;
    fn spark_connect_engine_open_session(
        engine: *mut c_void,
        user: *const u8,
        user_len: usize,
        id: *const u8,
        id_len: usize,
        task: *mut c_void,
        error: *mut BridgeBuffer,
    ) -> *mut c_void;
    fn spark_connect_engine_prepare(
        session: *mut c_void,
        relation: *const u8,
        len: usize,
        task: *mut c_void,
        error: *mut BridgeBuffer,
    ) -> *mut c_void;
    fn spark_connect_engine_schema(
        query: *mut c_void,
        out: *mut BridgeBuffer,
        error: *mut BridgeBuffer,
    ) -> i32;
    fn spark_connect_engine_next_arrow(
        query: *mut c_void,
        out: *mut FFI_ArrowArray,
        lease: *mut c_void,
        defer_release: unsafe extern "C" fn(*mut c_void, *mut c_void),
        error: *mut BridgeBuffer,
    ) -> i32;
    fn spark_connect_engine_arrow_owner_free_checked(
        owner: *mut c_void,
        error: *mut BridgeBuffer,
    ) -> i32;
    fn spark_connect_engine_analyze(
        session: *mut c_void,
        opcode: i32,
        relation: *const u8,
        len: usize,
        task: *mut c_void,
        out: *mut BridgeBuffer,
        error: *mut BridgeBuffer,
    ) -> i32;
    fn spark_connect_engine_config(
        session: *mut c_void,
        opcode: i32,
        fields: *const u8,
        len: usize,
        task: *mut c_void,
        out: *mut BridgeBuffer,
        error: *mut BridgeBuffer,
    ) -> i32;
    fn spark_connect_engine_cancel(task: *mut c_void);
    fn spark_connect_engine_free(handle: *mut c_void);
    fn spark_connect_engine_free_checked(handle: *mut c_void, error: *mut BridgeBuffer) -> i32;
}

fn engine_error(error: BridgeBuffer) -> Status {
    let bytes = take_buffer(error);
    decode_error(&String::from_utf8_lossy(&bytes))
}
fn decode_error(message: &str) -> Status {
    if let Some((code, text)) = message
        .strip_prefix("JVM_ENGINE_ERROR:")
        .and_then(|text| text.split_once(':'))
    {
        if let Ok(code @ 1..=16) = code.parse::<i32>() {
            return Status::new(tonic::Code::from_i32(code), text.to_owned());
        }
    }
    Status::internal(format!("JVM engine: {message}"))
}

#[derive(Default)]
struct Counts {
    workers: usize,
    exports: usize,
    handles: usize,
    tasks: HashMap<u64, Weak<Task>>,
    failed_cleanup: Vec<FailedCleanup>,
}
pub(super) struct JvmEngine {
    factory: AtomicPtr<c_void>,
    closed: AtomicBool,
    counts: Mutex<Counts>,
    drained: Condvar,
    permits: Arc<Semaphore>,
    session_permits: Arc<Semaphore>,
    worker_attachments: AtomicUsize,
}
// C owns JavaVM/global references. JNI worker attachments remain thread-local.
unsafe impl Send for JvmEngine {}
unsafe impl Sync for JvmEngine {}
impl Drop for JvmEngine {
    fn drop(&mut self) {
        // Fallback for startup failures that never reached normal host shutdown.
        unsafe {
            spark_connect_engine_free(self.factory.swap(std::ptr::null_mut(), Ordering::SeqCst))
        };
    }
}
impl JvmEngine {
    pub(super) fn new(factory: *mut c_void) -> Arc<Self> {
        Arc::new(Self {
            factory: AtomicPtr::new(factory),
            closed: AtomicBool::new(false),
            counts: Mutex::new(Counts::default()),
            drained: Condvar::new(),
            permits: Arc::new(Semaphore::new(64)),
            session_permits: Arc::new(Semaphore::new(64)),
            worker_attachments: AtomicUsize::new(0),
        })
    }
    pub(super) fn attach_worker(self: &Arc<Self>) -> Result<WorkerAttachment, Status> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(Status::unavailable("Spark engine is stopping"));
        }
        let mut error = BridgeBuffer::default();
        let pointer = unsafe {
            spark_connect_engine_thread_attach(self.factory.load(Ordering::SeqCst), &mut error)
        };
        if pointer.is_null() {
            return Err(engine_error(error));
        }
        let _ = take_buffer(error);
        self.worker_attachments.fetch_add(1, Ordering::SeqCst);
        Ok(WorkerAttachment {
            pointer,
            core: self.clone(),
            _thread: PhantomData,
        })
    }
    /// Check after the runtime stops; wait_drained cannot include this count
    /// because the runtime owns the workers that must detach during its stop.
    pub(super) fn verify_workers_detached(&self) -> Result<(), String> {
        let remaining = self.worker_attachments.load(Ordering::SeqCst);
        if remaining == 0 {
            Ok(())
        } else {
            Err(format!(
                "{remaining} JNI worker attachments remain after runtime stop"
            ))
        }
    }
    #[cfg(test)]
    pub(super) fn set_worker_attachment_count_for_test(&self, count: usize) {
        self.worker_attachments.store(count, Ordering::SeqCst);
    }
    #[cfg(test)]
    pub(super) fn fail_export_cleanup_for_test(self: &Arc<Self>) {
        cleanup_with(
            Cleanup::Arrow {
                owner: 0,
                lease: ExportLease::new(self.clone()),
            },
            |_, _| Err(Status::internal("injected Arrow cleanup failure")),
        );
    }
    pub(super) fn shutdown(&self) {
        let tasks = {
            let counts = self
                .counts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            self.closed.store(true, Ordering::SeqCst);
            counts
                .tasks
                .values()
                .filter_map(Weak::upgrade)
                .collect::<Vec<_>>()
        };
        self.permits.close();
        self.session_permits.close();
        for task in tasks {
            task.cancel();
        }
    }
    pub(super) fn wait_drained(&self) {
        let mut counts = self
            .counts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while counts.workers != 0 || counts.handles != 0 || counts.exports != 0 {
            counts = self
                .drained
                .wait(counts)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
    pub(super) fn cleanup_failure(&self) -> Option<String> {
        let counts = self
            .counts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        counts.failed_cleanup.first().map(|failure| {
            format!(
                "JNI cleanup is incomplete; retained cleanup failed: {} ({} pending); retry close after the failure is resolved",
                failure.error,
                counts.failed_cleanup.len()
            )
        })
    }
    fn take_failed_cleanup(&self) -> Vec<FailedCleanup> {
        std::mem::take(
            &mut self
                .counts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .failed_cleanup,
        )
    }
    /// Only an explicit owner close retries failed cleanup. Preserve each item's
    /// original lane, Java ownership, admission and Arrow lease until success.
    pub(super) fn retry_failed_cleanup(&self) {
        for failure in self.take_failed_cleanup() {
            queue(failure.item);
        }
    }
    fn take_factory_after_drain(&self) -> Result<*mut c_void, String> {
        if !self.closed.load(Ordering::SeqCst) {
            return Err("Spark engine factory release requires shutdown".into());
        }
        self.verify_workers_detached()?;
        let counts = self
            .counts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if counts.workers != 0 || counts.handles != 0 || counts.exports != 0 {
            return Err("Spark engine factory release requires drained work".into());
        }
        // Shutdown prevents new producers, and every outstanding JNI operation
        // or Arrow lease is represented above. Never hold the mutex across JNI.
        drop(counts);
        Ok(self.factory.swap(std::ptr::null_mut(), Ordering::SeqCst))
    }
    /// The host holds its stopping mutex, serializing this synchronous release
    /// with retries, and retains the engine until the native free returns.
    pub(super) fn release_factory(&self) -> Result<(), String> {
        self.release_factory_with(release_native)
    }
    fn release_factory_with(
        &self,
        release: impl FnOnce(CleanupResource, usize) -> Result<(), Status>,
    ) -> Result<(), String> {
        let factory = self.take_factory_after_drain()?;
        // stop_server retains its own Arc throughout this call, so Drop cannot
        // race to free the factory. Cleanup workers may still hold harmless core
        // Arcs after publishing zero counters; their later drops see null.
        match release(CleanupResource::Handle, factory as usize) {
            Ok(()) => Ok(()),
            Err(error) => {
                // The checked ABI did not consume the handle. The host stopping
                // mutex excludes another close; retain the same factory for retry.
                self.factory.store(factory, Ordering::SeqCst);
                Err(format!(
                    "JNI cleanup is incomplete; retained cleanup failed: {error}"
                ))
            }
        }
    }
    async fn admit_session(
        self: &Arc<Self>,
        deadline: Instant,
    ) -> Result<OwnedSemaphorePermit, Status> {
        tokio::time::timeout_at(deadline, self.session_permits.clone().acquire_owned())
            .await
            .map_err(|_| Status::resource_exhausted("Spark engine session capacity exhausted"))?
            .map_err(|_| Status::unavailable("Spark engine is stopping"))
    }
    async fn task(self: &Arc<Self>, deadline: Instant) -> Result<TaskGuard, Status> {
        let permit = tokio::time::timeout_at(deadline, self.permits.clone().acquire_owned())
            .await
            .map_err(|_| Status::resource_exhausted("Spark engine task capacity exhausted"))?
            .map_err(|_| Status::unavailable("Spark engine is stopping"))?;
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let task = Arc::new(Task {
            core: self.clone(),
            id,
            handle: Mutex::new(None),
            cancelled: AtomicBool::new(false),
            permit: Mutex::new(Some(permit)),
        });
        {
            let mut counts = self
                .counts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if self.closed.load(Ordering::SeqCst) {
                return Err(Status::unavailable("Spark engine is stopping"));
            }
            counts.tasks.insert(id, Arc::downgrade(&task));
        }
        Ok(TaskGuard {
            task,
            complete: false,
        })
    }
    async fn blocking<T: Send + 'static>(
        self: &Arc<Self>,
        action: impl FnOnce() -> Result<T, Status> + Send + 'static,
    ) -> Result<T, Status> {
        // Reserve activity before spawning. Dropping a future cancels its TaskGuard;
        // the worker and all global references remain retained until JNI returns.
        let activity = {
            let mut counts = self
                .counts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if self.closed.load(Ordering::SeqCst) {
                return Err(Status::unavailable("Spark engine is stopping"));
            }
            counts.workers += 1;
            Activity(self.clone())
        };
        tokio::task::spawn_blocking(move || {
            let worker_activity = activity;
            super::attach_engine_worker(&worker_activity.0);
            action()
        })
        .await
        .map_err(|e| Status::internal(format!("Spark engine worker failed: {e}")))?
    }
    fn handle(
        self: &Arc<Self>,
        pointer: *mut c_void,
        error: BridgeBuffer,
        kind: HandleKind,
        retained_task: Option<Arc<Handle>>,
    ) -> Result<Arc<Handle>, Status> {
        if pointer.is_null() {
            return Err(engine_error(error));
        }
        // A successful C call must not return an owned error buffer.
        let _ = take_buffer(error);
        self.counts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .handles += 1;
        Ok(Arc::new(Handle {
            pointer,
            core: self.clone(),
            permit: Mutex::new(None),
            kind,
            retained_task,
        }))
    }
}

/// Retains the JVM until the same runtime worker detaches on thread exit.
/// Rc's marker prevents moving or sharing the guard across threads. Drop logs
/// a detach failure without retrying; C retains the attachment/context for
/// safety, and the attachment count makes shutdown report that failure.
pub(super) struct WorkerAttachment {
    pointer: *mut c_void,
    core: Arc<JvmEngine>,
    _thread: PhantomData<Rc<()>>,
}
impl WorkerAttachment {
    pub(super) fn belongs_to(&self, core: &Arc<JvmEngine>) -> bool {
        Arc::ptr_eq(&self.core, core)
    }
}
impl Drop for WorkerAttachment {
    fn drop(&mut self) {
        let mut error = BridgeBuffer::default();
        if unsafe { spark_connect_engine_thread_detach(self.pointer, &mut error) } != 0 {
            eprintln!("JVM worker detach failed: {}", engine_error(error));
        } else {
            let _ = take_buffer(error);
            self.core.worker_attachments.fetch_sub(1, Ordering::SeqCst);
        }
    }
}
struct Activity(Arc<JvmEngine>);
impl Drop for Activity {
    fn drop(&mut self) {
        let mut c = self
            .0
            .counts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        c.workers -= 1;
        self.0.drained.notify_all();
    }
}
struct Handle {
    pointer: *mut c_void,
    core: Arc<JvmEngine>,
    // Keep task/session admission until its final Java reference is released.
    permit: Mutex<Option<OwnedSemaphorePermit>>,
    kind: HandleKind,
    // Query teardown must precede reclamation of its admission-holding task.
    retained_task: Option<Arc<Handle>>,
}
#[derive(Clone, Copy)]
enum HandleKind {
    Task,
    Session,
    Query,
}
#[derive(Clone, Copy)]
enum CleanupLane {
    Task,
    Query,
    Session,
}
impl HandleKind {
    fn cleanup_lane(self) -> CleanupLane {
        match self {
            Self::Task => CleanupLane::Task,
            Self::Query => CleanupLane::Query,
            Self::Session => CleanupLane::Session,
        }
    }
}
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}
impl Handle {
    fn pointer(&self) -> *mut c_void {
        self.pointer
    }
}
impl Drop for Handle {
    fn drop(&mut self) {
        queue(Cleanup::Free {
            pointer: self.pointer as usize,
            core: self.core.clone(),
            lane: self.kind.cleanup_lane(),
            retained_task: self.retained_task.take(),
            permit: self
                .permit
                .get_mut()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take(),
        });
    }
}
enum Cleanup {
    Cancel(Arc<Handle>),
    Arrow {
        owner: usize,
        lease: Box<ExportLease>,
    },
    Free {
        pointer: usize,
        core: Arc<JvmEngine>,
        permit: Option<OwnedSemaphorePermit>,
        lane: CleanupLane,
        retained_task: Option<Arc<Handle>>,
    },
    #[cfg(test)]
    TestAction {
        lane: CleanupLane,
        action: Box<dyn FnOnce() + Send>,
    },
}
struct FailedCleanup {
    item: Cleanup,
    error: Status,
}
#[derive(Clone, Copy)]
enum CleanupResource {
    Handle,
    Arrow,
}
impl Cleanup {
    fn core(&self) -> Option<&Arc<JvmEngine>> {
        match self {
            Self::Cancel(handle) => Some(&handle.core),
            Self::Arrow { lease, .. } => Some(&lease.0),
            Self::Free { core, .. } => Some(core),
            #[cfg(test)]
            Self::TestAction { .. } => None,
        }
    }
    fn lane(&self) -> CleanupLane {
        match self {
            Self::Cancel(_) | Self::Arrow { .. } => CleanupLane::Task,
            Self::Free { lane, .. } => *lane,
            #[cfg(test)]
            Self::TestAction { lane, .. } => *lane,
        }
    }
}

/// Cleanup lanes outlive individual servers. Retain an attachment only during
/// a burst, and count that whole lifetime as activity so shutdown waits for the
/// same-thread detach before it releases the engine factory.
struct CleanupAttachment<W> {
    worker: Option<W>,
    activity: Activity,
}
impl<W> CleanupAttachment<W> {
    fn attach_with(
        core: &Arc<JvmEngine>,
        attach: impl FnOnce(&Arc<JvmEngine>) -> Result<W, Status>,
    ) -> Result<Self, Status> {
        let activity = {
            let mut counts = core
                .counts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if core.closed.load(Ordering::SeqCst) {
                return Err(Status::unavailable("Spark engine is stopping"));
            }
            counts.workers += 1;
            Activity(core.clone())
        };
        // Each queued item still owns a handle or export, retaining the factory
        // across attachment acquisition and any concurrent shutdown.
        let worker = attach(core)?;
        Ok(Self {
            worker: Some(worker),
            activity,
        })
    }
    fn attach_or_fallback(
        core: &Arc<JvmEngine>,
        attach: impl FnOnce(&Arc<JvmEngine>) -> Result<W, Status>,
        reported: &mut Option<Weak<JvmEngine>>,
        report: impl FnOnce(&Status),
    ) -> Option<Self> {
        match Self::attach_with(core, attach) {
            Ok(attachment) => {
                *reported = None;
                Some(attachment)
            }
            Err(error) if error.code() == tonic::Code::Unavailable => None,
            Err(error) => {
                // Suppress consecutive reports for the same engine until an
                // attachment succeeds. Alternating failing engines report once
                // per switch. The weak identity does not retain the engine.
                let identity = Arc::downgrade(core);
                if reported
                    .as_ref()
                    .is_none_or(|previous| !Weak::ptr_eq(previous, &identity))
                {
                    *reported = Some(identity);
                    report(&error);
                }
                None
            }
        }
    }
    fn belongs_to(&self, core: &Arc<JvmEngine>) -> bool {
        Arc::ptr_eq(&self.activity.0, core)
    }
}
impl<W> Drop for CleanupAttachment<W> {
    fn drop(&mut self) {
        // Activity must stay reserved until detach completes. A detach failure
        // remains visible through verify_workers_detached's attachment count.
        drop(self.worker.take());
    }
}

fn cleanup_loop(rx: mpsc::Receiver<Cleanup>) {
    cleanup_loop_with(rx, CLEANUP_ATTACHMENT_IDLE, JvmEngine::attach_worker);
}

// The generic worker constructor lets tests exercise successful attachment
// lifetimes without a JVM; the production loop still uses the native guard.
fn cleanup_loop_with<W>(
    rx: mpsc::Receiver<Cleanup>,
    idle: Duration,
    mut attach: impl FnMut(&Arc<JvmEngine>) -> Result<W, Status>,
) {
    let mut attachment: Option<CleanupAttachment<W>> = None;
    let mut reported = None;
    loop {
        let item = if attachment.is_some() {
            match rx.recv_timeout(idle) {
                Ok(item) => item,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    drop(attachment.take());
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        } else {
            match rx.recv() {
                Ok(item) => item,
                Err(_) => break,
            }
        };
        match item.core() {
            Some(core) => {
                if attachment
                    .as_ref()
                    .is_none_or(|current| !current.belongs_to(core))
                {
                    drop(attachment.take());
                    // Per-call JNI scopes remain the fallback on attachment
                    // failure or when shutdown has already closed admission.
                    attachment = CleanupAttachment::attach_or_fallback(
                        core,
                        &mut attach,
                        &mut reported,
                        |error| {
                            eprintln!(
                                "JVM cleanup lane attachment failed; using per-call attachment: {error}"
                            );
                        },
                    );
                }
            }
            None => drop(attachment.take()),
        }
        cleanup(item);
    }
}

fn cleanup_sender(lane: CleanupLane) -> &'static Result<mpsc::Sender<Cleanup>, String> {
    // Session close may block on Spark teardown. Isolate it from Query frees
    // that hold task admission, and Task/cancel/Arrow reclamation itself.
    static TASK: OnceLock<Result<mpsc::Sender<Cleanup>, String>> = OnceLock::new();
    static QUERY: OnceLock<Result<mpsc::Sender<Cleanup>, String>> = OnceLock::new();
    static SESSION: OnceLock<Result<mpsc::Sender<Cleanup>, String>> = OnceLock::new();
    let (sender, name) = match lane {
        CleanupLane::Task => (&TASK, "spark-connect-engine-task-cleanup"),
        CleanupLane::Query => (&QUERY, "spark-connect-engine-query-cleanup"),
        CleanupLane::Session => (&SESSION, "spark-connect-engine-session-cleanup"),
    };
    sender.get_or_init(|| {
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name(name.into())
            .spawn(move || cleanup_loop(rx))
            .map_err(|e| e.to_string())?;
        Ok(tx)
    })
}
fn release_native(resource: CleanupResource, pointer: usize) -> Result<(), Status> {
    let mut error = BridgeBuffer::default();
    let result = unsafe {
        match resource {
            CleanupResource::Handle => {
                spark_connect_engine_free_checked(pointer as *mut c_void, &mut error)
            }
            CleanupResource::Arrow => {
                spark_connect_engine_arrow_owner_free_checked(pointer as *mut c_void, &mut error)
            }
        }
    };
    if result != 0 {
        Err(engine_error(error))
    } else {
        let _ = take_buffer(error);
        Ok(())
    }
}
fn cleanup(item: Cleanup) {
    cleanup_with(item, release_native);
}
fn cleanup_with(item: Cleanup, release: impl FnOnce(CleanupResource, usize) -> Result<(), Status>) {
    let result = match &item {
        Cleanup::Arrow { owner, .. } => release(CleanupResource::Arrow, *owner),
        Cleanup::Free { pointer, .. } => release(CleanupResource::Handle, *pointer),
        _ => Ok(()),
    };
    if let Err(error) = result {
        eprintln!("JNI cleanup is incomplete; retained cleanup failed: {error}");
        // This deliberate core/item cycle retains the raw Java ownership and
        // budgets even if the request disappears. Explicit close breaks it by
        // requeuing the item; a permanent Java cleanup error remains retained.
        let core = item.core().expect("native cleanup has an engine").clone();
        core.counts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .failed_cleanup
            .push(FailedCleanup { item, error });
        core.drained.notify_all();
        return;
    }
    match item {
        Cleanup::Arrow { lease, .. } => {
            drop(lease);
        }
        Cleanup::Cancel(handle) => unsafe {
            spark_connect_engine_cancel(handle.pointer());
        },
        Cleanup::Free {
            pointer: _,
            core,
            permit,
            lane: _,
            retained_task,
        } => {
            // Publish resource/admission release before reporting this handle
            // drained. Query dependencies may schedule their own Task free.
            drop(retained_task);
            drop(permit);
            let mut counts = core
                .counts
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            counts.handles -= 1;
            core.drained.notify_all();
        }
        #[cfg(test)]
        Cleanup::TestAction { action, .. } => action(),
    }
}
fn queue(item: Cleanup) {
    match cleanup_sender(item.lane()) {
        Ok(tx) => {
            if let Err(e) = tx.send(item) {
                cleanup(e.0);
            }
        }
        Err(_) => cleanup(item),
    }
}
// Counts exported batches until the *actual* deferred Java allocator close,
// not just until the final Rust Buffer schedules its release callback.
struct ExportLease(Arc<JvmEngine>);
impl ExportLease {
    fn new(core: Arc<JvmEngine>) -> Box<Self> {
        core.counts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .exports += 1;
        Box::new(Self(core))
    }
}
impl Drop for ExportLease {
    fn drop(&mut self) {
        let mut counts = self
            .0
            .counts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        counts.exports -= 1;
        self.0.drained.notify_all();
    }
}
// C consumes the lease on every result, and invokes this after EOF/error or the
// final imported Buffer release. Java cleanup normally runs on its own lane;
// if that worker is unavailable, queue releases inline.
unsafe extern "C" fn defer_arrow_release(lease: *mut c_void, owner: *mut c_void) {
    let lease = unsafe { Box::from_raw(lease.cast::<ExportLease>()) };
    queue(Cleanup::Arrow {
        owner: owner as usize,
        lease,
    });
}

struct Task {
    core: Arc<JvmEngine>,
    id: u64,
    handle: Mutex<Option<Arc<Handle>>>,
    cancelled: AtomicBool,
    permit: Mutex<Option<OwnedSemaphorePermit>>,
}
impl Task {
    fn initialize(&self) -> Result<Arc<Handle>, Status> {
        let existing = self
            .handle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let handle = match existing {
            Some(handle) => handle,
            None => {
                if self.cancelled.load(Ordering::SeqCst) {
                    return Err(Status::cancelled("Spark work cancelled"));
                }
                // Never hold the publication mutex across JNI: future drop and
                // native stop must be able to mark cancellation immediately.
                let mut error = BridgeBuffer::default();
                let pointer = unsafe {
                    spark_connect_engine_task_new(
                        self.core.factory.load(Ordering::SeqCst),
                        &mut error,
                    )
                };
                let created = self.core.handle(pointer, error, HandleKind::Task, None)?;
                self.handle
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get_or_insert(created)
                    .clone()
            }
        };
        if let Some(permit) = self
            .permit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            *handle
                .permit
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(permit);
        }
        if self.cancelled.load(Ordering::SeqCst) {
            unsafe {
                spark_connect_engine_cancel(handle.pointer());
            }
            return Err(Status::cancelled("Spark work cancelled"));
        }
        Ok(handle)
    }
    fn cancel(&self) {
        if !self.cancelled.swap(true, Ordering::SeqCst) {
            if let Some(handle) = self
                .handle
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
            {
                queue(Cleanup::Cancel(handle));
            }
        }
    }
}
impl Drop for Task {
    fn drop(&mut self) {
        self.core
            .counts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .tasks
            .remove(&self.id);
    }
}
struct TaskGuard {
    task: Arc<Task>,
    complete: bool,
}
impl TaskGuard {
    fn finish(&mut self) {
        self.complete = true;
    }
}
impl Drop for TaskGuard {
    fn drop(&mut self) {
        if !self.complete {
            self.task.cancel();
        }
    }
}

pub(super) struct EngineFactory(pub Arc<JvmEngine>);
impl Engine for EngineFactory {
    fn create_session(&self, context: SessionContext) -> Result<Arc<dyn EngineSession>, Status> {
        Ok(Arc::new(JvmSession {
            core: self.0.clone(),
            context,
            handle: OnceCell::new(),
            catalog: RwLock::new(()),
        }))
    }
}
struct JvmSession {
    core: Arc<JvmEngine>,
    context: SessionContext,
    handle: OnceCell<Arc<Handle>>,
    catalog: RwLock<()>,
}
#[derive(Clone, Copy)]
enum CatalogAccess {
    Read,
    Write,
}
enum CatalogGuard<'a> {
    Read { _guard: RwLockReadGuard<'a, ()> },
    Write { _guard: RwLockWriteGuard<'a, ()> },
}
impl CatalogAccess {
    // JNI Config ABI: 1..=5 are read-only, 6/7 mutate. Unknown opcodes stay
    // exclusive conservatively. The Java adapter mirrors this contract.
    fn config(opcode: i32) -> Self {
        match opcode {
            1..=5 => Self::Read,
            _ => Self::Write,
        }
    }
    async fn lock(self, catalog: &RwLock<()>) -> CatalogGuard<'_> {
        match self {
            Self::Read => CatalogGuard::Read {
                _guard: catalog.read().await,
            },
            Self::Write => CatalogGuard::Write {
                _guard: catalog.write().await,
            },
        }
    }
}
#[derive(Clone, Copy)]
#[repr(i32)]
enum AnalysisMode {
    Schema = 0,
    IsLocal = 1,
    IsStreaming = 2,
}
struct Metadata {
    schema: SchemaRef,
    flag: bool,
}
impl JvmSession {
    async fn handle(&self, deadline: Instant) -> Result<Arc<Handle>, Status> {
        self.handle
            .get_or_try_init(|| async {
                // Bound heavyweight Java sessions through actual deferred close,
                // without consuming the independent in-flight task budget.
                // Keep the fixed operation-entry session-admission deadline
                // across catalog ordering, OnceCell waits and initializer retries.
                let session_permit = self.core.admit_session(deadline).await?;
                let mut guard = self.core.task(deadline).await?;
                let task = guard.task.clone();
                let core = self.core.clone();
                let user = self
                    .context
                    .user_context
                    .as_ref()
                    .map(|u| u.user_id.clone())
                    .unwrap_or_default();
                let id = self.context.session_id.clone();
                let handle = self
                    .core
                    .blocking(move || {
                        let task = task.initialize()?;
                        let mut error = BridgeBuffer::default();
                        let pointer = unsafe {
                            spark_connect_engine_open_session(
                                core.factory.load(Ordering::SeqCst),
                                user.as_ptr(),
                                user.len(),
                                id.as_ptr(),
                                id.len(),
                                task.pointer(),
                                &mut error,
                            )
                        };
                        let handle = core.handle(pointer, error, HandleKind::Session, None)?;
                        *handle
                            .permit
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) =
                            Some(session_permit);
                        Ok(handle)
                    })
                    .await?;
                guard.finish();
                Ok(handle)
            })
            .await
            .cloned()
    }
    async fn metadata(
        &self,
        relation: proto::Relation,
        mode: AnalysisMode,
        deadline: Instant,
    ) -> Result<Metadata, Status> {
        let session = self.handle(deadline).await?;
        let mut guard = self.core.task(deadline).await?;
        let task = guard.task.clone();
        let bytes = relation.encode_to_vec();
        let metadata = self
            .core
            .blocking(move || {
                let task = task.initialize()?;
                let bytes = output(|out, error| unsafe {
                    spark_connect_engine_analyze(
                        session.pointer(),
                        mode as i32,
                        bytes.as_ptr(),
                        bytes.len(),
                        task.pointer(),
                        out,
                        error,
                    )
                })?;
                decode_metadata(bytes, mode)
            })
            .await?;
        guard.finish();
        Ok(metadata)
    }
    async fn config_pairs(
        &self,
        opcode: i32,
        pairs: Vec<proto::KeyValue>,
        deadline: Instant,
    ) -> Result<Vec<proto::KeyValue>, Status> {
        let bytes = encode_pairs(&pairs)?;
        let session = self.handle(deadline).await?;
        let mut guard = self.core.task(deadline).await?;
        let task = guard.task.clone();
        let bytes = self
            .core
            .blocking(move || {
                let task = task.initialize()?;
                output(|out, error| unsafe {
                    spark_connect_engine_config(
                        session.pointer(),
                        opcode,
                        bytes.as_ptr(),
                        bytes.len(),
                        task.pointer(),
                        out,
                        error,
                    )
                })
            })
            .await?;
        guard.finish();
        decode_pairs(&bytes)
    }
    async fn relation(
        &self,
        relation: proto::Relation,
        deadline: Instant,
    ) -> Result<JvmQuery, Status> {
        let session = self.handle(deadline).await?;
        let guard = self.core.task(deadline).await?;
        let task = guard.task.clone();
        let core = self.core.clone();
        let bytes = relation.encode_to_vec();
        let (handle, schema) = self
            .core
            .blocking(move || {
                let task = task.initialize()?;
                let mut error = BridgeBuffer::default();
                let pointer = unsafe {
                    spark_connect_engine_prepare(
                        session.pointer(),
                        bytes.as_ptr(),
                        bytes.len(),
                        task.pointer(),
                        &mut error,
                    )
                };
                // Preserve the task-admission bound through actual query close,
                // even though task frees use the independent cleanup lane.
                let query = core.handle(pointer, error, HandleKind::Query, Some(task))?;
                let bytes = output(|out, error| unsafe {
                    spark_connect_engine_schema(query.pointer(), out, error)
                })?;
                let reader = ipc(bytes)?;
                let schema = reader.schema();
                Ok((query, schema))
            })
            .await?;
        Ok(JvmQuery {
            core: self.core.clone(),
            handle,
            guard,
            schema,
        })
    }
}
fn decode_metadata(bytes: Vec<u8>, mode: AnalysisMode) -> Result<Metadata, Status> {
    decode_metadata_with_limit(bytes, mode, MAX_BYTES)
}
fn decode_metadata_with_limit(
    bytes: Vec<u8>,
    mode: AnalysisMode,
    ipc_limit: usize,
) -> Result<Metadata, Status> {
    let flag = match bytes.first() {
        Some(0) => false,
        Some(1) if !matches!(mode, AnalysisMode::Schema) => true,
        _ => return Err(Status::internal("invalid Spark metadata flag")),
    };
    // The transport prefix is not part of the Arrow IPC size limit. Keep the
    // allocation intact and start the reader after it, avoiding a payload copy.
    if bytes.len() - 1 > ipc_limit {
        return Err(Status::resource_exhausted(
            "Spark Arrow batch exceeds JNI limit",
        ));
    }
    let mut cursor = Cursor::new(bytes);
    cursor.set_position(1);
    let reader = StreamReader::try_new(cursor, None)
        .map_err(|e| Status::internal(format!("invalid Spark Arrow stream: {e}")))?;
    Ok(Metadata {
        schema: reader.schema(),
        flag,
    })
}
fn ipc(bytes: Vec<u8>) -> Result<StreamReader<Cursor<Vec<u8>>>, Status> {
    if bytes.len() > MAX_BYTES {
        return Err(Status::resource_exhausted(
            "Spark Arrow batch exceeds JNI limit",
        ));
    }
    StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|e| Status::internal(format!("invalid Spark Arrow stream: {e}")))
}
fn output(
    action: impl FnOnce(*mut BridgeBuffer, *mut BridgeBuffer) -> i32,
) -> Result<Vec<u8>, Status> {
    let mut out = BridgeBuffer::default();
    let mut error = BridgeBuffer::default();
    if action(&mut out, &mut error) < 0 {
        let _ = take_buffer(out);
        Err(engine_error(error))
    } else {
        let _ = take_buffer(error);
        Ok(take_buffer(out))
    }
}
struct JvmQuery {
    core: Arc<JvmEngine>,
    handle: Arc<Handle>,
    guard: TaskGuard,
    schema: SchemaRef,
}
#[tonic::async_trait]
impl PreparedQuery for JvmQuery {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    async fn execute(self: Box<Self>) -> Result<BatchStream, Status> {
        Ok(Box::pin(stream::try_unfold(
            *self,
            |mut query| async move {
                let handle = query.handle.clone();
                let task = query.guard.task.clone();
                let core = query.core.clone();
                let schema = query.schema.clone();
                let result = query
                    .core
                    .blocking(move || {
                        task.initialize()?;
                        let lease = ExportLease::new(core);
                        let lease_pointer = Box::into_raw(lease);
                        let mut out = FFI_ArrowArray::empty();
                        let mut error = BridgeBuffer::default();
                        let result = unsafe {
                            spark_connect_engine_next_arrow(
                                handle.pointer(),
                                &mut out,
                                lease_pointer.cast(),
                                defer_arrow_release,
                                &mut error,
                            )
                        };
                        if result != 1 {
                            // C consumes the lease on every outcome. Partial
                            // exports retain it through deferred Java cleanup too.
                            if result < 0 {
                                return Err(engine_error(error));
                            }
                            let _ = take_buffer(error);
                            return Ok(None);
                        }
                        let _ = take_buffer(error);
                        // The C descriptor now owns the lease through its callback;
                        // errors, panics, early drops, and cloned/sliced buffers all
                        // eventually schedule the same independent Java batch close.
                        let batch = unsafe { zero_copy::import_batch(out, schema) }?;
                        Ok(Some(batch))
                    })
                    .await?;
                match result {
                    Some(batch) => Ok(Some((batch, query))),
                    None => {
                        query.guard.finish();
                        Ok(None)
                    }
                }
            },
        )))
    }
}
fn root(plan: Option<proto::Plan>) -> Result<proto::Relation, Status> {
    match plan.and_then(|p| p.op_type) {
        Some(proto::plan::OpType::Root(r)) => Ok(r),
        _ => Err(Status::invalid_argument(
            "analysis requires a root relation",
        )),
    }
}
#[tonic::async_trait]
impl EngineSession for JvmSession {
    async fn prepare(&self, operation: proto::plan::OpType) -> Result<PreparedOperation, Status> {
        let deadline = Instant::now() + SESSION_ADMISSION_TIMEOUT;
        // Preparation accepts read-only plans and excludes configuration writes
        // during logical planning. Physical planning stays lazy until the first
        // fetch, so configuration changes after prepare can affect execution.
        let _catalog = CatalogAccess::Read.lock(&self.catalog).await;
        match operation {
            proto::plan::OpType::Root(relation) => Ok(PreparedOperation::Query(Box::new(
                self.relation(relation, deadline).await?,
            ))),
            proto::plan::OpType::Command(command) => match command.command_type {
                Some(proto::command::CommandType::SqlCommand(sql)) => {
                    // Inspect legacy argument fields to reject unsupported encodings.
                    #[allow(deprecated)]
                    if !sql.args.is_empty()
                        || !sql.pos_args.is_empty()
                        || !sql.named_arguments.is_empty()
                        || !sql.pos_arguments.is_empty()
                    {
                        return Err(Status::unimplemented("Spark engine SQL command arguments"));
                    }
                    // Accept the legacy SQL string when the relation input is absent.
                    #[allow(deprecated)]
                    let relation = match sql.input {
                        Some(r) => r,
                        None if !sql.sql.is_empty() => proto::Relation {
                            rel_type: Some(proto::relation::RelType::Sql(proto::Sql {
                                query: sql.sql,
                                ..Default::default()
                            })),
                            ..Default::default()
                        },
                        _ => return Err(Status::invalid_argument("missing SQL input")),
                    };
                    // Validate as a read-only query before returning the relation.
                    let _metadata = self
                        .metadata(relation.clone(), AnalysisMode::Schema, deadline)
                        .await?;
                    Ok(PreparedOperation::Command(vec![
                        proto::execute_plan_response::ResponseType::SqlCommandResult(
                            proto::execute_plan_response::SqlCommandResult {
                                relation: Some(relation),
                            },
                        ),
                    ]))
                }
                _ => Err(Status::unimplemented(
                    "Spark engine command is not implemented",
                )),
            },
            _ => Err(Status::unimplemented("compressed Spark engine plans")),
        }
    }
    async fn analyze(
        &self,
        input: proto::analyze_plan_request::Analyze,
    ) -> Result<proto::analyze_plan_response::Result, Status> {
        use proto::{analyze_plan_request::Analyze, analyze_plan_response as out};
        let deadline = Instant::now() + SESSION_ADMISSION_TIMEOUT;
        let _catalog = CatalogAccess::Read.lock(&self.catalog).await;
        Ok(match input {
            Analyze::Schema(input) => {
                let metadata = self
                    .metadata(root(input.plan)?, AnalysisMode::Schema, deadline)
                    .await?;
                out::Result::Schema(out::Schema {
                    schema: Some(spark_connect_server::engine::spark_schema(
                        metadata.schema.as_ref(),
                    )?),
                })
            }
            Analyze::IsLocal(input) => {
                let metadata = self
                    .metadata(root(input.plan)?, AnalysisMode::IsLocal, deadline)
                    .await?;
                out::Result::IsLocal(out::IsLocal {
                    is_local: metadata.flag,
                })
            }
            Analyze::IsStreaming(input) => {
                let metadata = self
                    .metadata(root(input.plan)?, AnalysisMode::IsStreaming, deadline)
                    .await?;
                out::Result::IsStreaming(out::IsStreaming {
                    is_streaming: metadata.flag,
                })
            }
            _ => {
                return Err(Status::unimplemented(
                    "Spark engine analysis is not implemented",
                ))
            }
        })
    }
    async fn config(
        &self,
        input: proto::config_request::operation::OpType,
    ) -> Result<ConfigResult, Status> {
        use proto::config_request::operation::OpType;
        let deadline = Instant::now() + SESSION_ADMISSION_TIMEOUT;
        let silent = matches!(&input, OpType::Set(value) if value.silent == Some(true));
        let keys = |items: Vec<String>| {
            items
                .into_iter()
                .map(|key| proto::KeyValue { key, value: None })
                .collect::<Vec<_>>()
        };
        let (opcode, pairs) = match input {
            OpType::Get(v) => (1, keys(v.keys)),
            OpType::GetOption(v) => (2, keys(v.keys)),
            OpType::GetWithDefault(v) => (3, v.pairs),
            OpType::GetAll(v) => (4, v.prefix.map(|p| keys(vec![p])).unwrap_or_default()),
            OpType::IsModifiable(v) => (5, keys(v.keys)),
            OpType::Set(v) => (6, v.pairs),
            OpType::Unset(v) => (7, keys(v.keys)),
        };
        let _catalog = CatalogAccess::config(opcode).lock(&self.catalog).await;
        if silent {
            // Spark's silent Set applies valid keys even when another key fails.
            // Each native call has its own task: completing one cannot mask
            // cancellation of the next. The catalog writer spans the full set.
            return silent_config_set(pairs, |pair| {
                self.config_pairs(opcode, vec![pair], deadline)
            })
            .await;
        }
        Ok(ConfigResult {
            pairs: self.config_pairs(opcode, pairs, deadline).await?,
            warnings: Vec::new(),
        })
    }
}

async fn silent_config_set<F, Fut>(
    pairs: Vec<proto::KeyValue>,
    mut set: F,
) -> Result<ConfigResult, Status>
where
    F: FnMut(proto::KeyValue) -> Fut,
    Fut: std::future::Future<Output = Result<Vec<proto::KeyValue>, Status>>,
{
    let mut warnings = Vec::new();
    for pair in pairs {
        let key = pair.key.clone();
        match set(pair).await {
            Ok(_) => {}
            Err(error)
                if matches!(
                    error.code(),
                    tonic::Code::InvalidArgument
                        | tonic::Code::NotFound
                        | tonic::Code::AlreadyExists
                        | tonic::Code::OutOfRange
                ) =>
            {
                // Config errors can quote rejected values; warnings must not.
                warnings.push(format!("Failed to set {key} ({:?})", error.code()));
            }
            // Capacity, cancellation, shutdown and adapter failures belong to
            // the operation rather than an individual configuration key.
            Err(error) => return Err(error),
        }
    }
    Ok(ConfigResult {
        pairs: Vec::new(),
        warnings,
    })
}

fn encode_pairs(pairs: &[proto::KeyValue]) -> Result<Vec<u8>, Status> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(
        &u32::try_from(pairs.len())
            .map_err(|_| Status::resource_exhausted("too many config fields"))?
            .to_be_bytes(),
    );
    for pair in pairs {
        let key_len = i32::try_from(pair.key.len())
            .map_err(|_| Status::resource_exhausted("config key too large"))?;
        bytes.extend_from_slice(&key_len.to_be_bytes());
        bytes.extend_from_slice(pair.key.as_bytes());
        if let Some(value) = &pair.value {
            let len = i32::try_from(value.len())
                .map_err(|_| Status::resource_exhausted("config value too large"))?;
            bytes.extend_from_slice(&len.to_be_bytes());
            bytes.extend_from_slice(value.as_bytes());
        } else {
            bytes.extend_from_slice(&(-1_i32).to_be_bytes());
        }
    }
    Ok(bytes)
}

// SparkJvmEngine uses DataInputStream/DataOutputStream: each field starts with
// a big-endian 32-bit length, and only nullable values may use the -1 sentinel.
struct ConfigCursor<'a>(&'a [u8]);
impl<'a> ConfigCursor<'a> {
    fn integer(&mut self) -> Result<u32, Status> {
        let bytes = self
            .0
            .get(..4)
            .ok_or_else(|| Status::internal("truncated Spark config integer"))?;
        let value = u32::from_be_bytes(bytes.try_into().unwrap());
        self.0 = &self.0[4..];
        Ok(value)
    }

    fn bytes(&mut self) -> Result<&'a [u8], Status> {
        let length = self.integer()? as usize;
        let bytes = self
            .0
            .get(..length)
            .ok_or_else(|| Status::internal("invalid Spark config key length"))?;
        self.0 = &self.0[length..];
        Ok(bytes)
    }
}

fn decode_pairs(bytes: &[u8]) -> Result<Vec<proto::KeyValue>, Status> {
    let mut cursor = ConfigCursor(bytes);
    let count = cursor.integer()? as usize;
    if count > cursor.0.len() / 8 {
        return Err(Status::internal("invalid Spark config pair count"));
    }
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let key = String::from_utf8(cursor.bytes()?.to_vec())
            .map_err(|_| Status::internal("invalid Spark config UTF8"))?;
        let len = cursor.integer()? as i32;
        let value = if len == -1 {
            None
        } else if len < 0 {
            return Err(Status::internal("invalid Spark config value length"));
        } else {
            let len = len as usize;
            let field = cursor
                .0
                .get(..len)
                .ok_or_else(|| Status::internal("truncated Spark config value"))?;
            cursor.0 = &cursor.0[len..];
            Some(
                String::from_utf8(field.to_vec())
                    .map_err(|_| Status::internal("invalid Spark config UTF8"))?,
            )
        };
        result.push(proto::KeyValue { key, value });
    }
    if !cursor.0.is_empty() {
        return Err(Status::internal("trailing Spark config data"));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    static CLEANUP_TEST_LOCK: Mutex<()> = Mutex::new(());
    fn metadata_payload(flag: u8) -> (Vec<u8>, arrow::datatypes::Schema) {
        use arrow::{
            datatypes::{DataType, Field, Schema},
            ipc::writer::StreamWriter,
        };
        let schema = Schema::new(vec![Field::new("时区 😀", DataType::Utf8, true)]);
        let mut bytes = vec![flag];
        // Deliberately only a schema header: metadata preserves the existing
        // StreamReader::try_new/schema validation boundary, without reading data.
        drop(StreamWriter::try_new(&mut bytes, &schema).unwrap());
        (bytes, schema)
    }
    #[test]
    fn metadata_decoder_preserves_schema_and_boolean_results() {
        for mode in [
            AnalysisMode::Schema,
            AnalysisMode::IsLocal,
            AnalysisMode::IsStreaming,
        ] {
            let (bytes, schema) = metadata_payload(0);
            let metadata = decode_metadata(bytes, mode).unwrap();
            assert_eq!(metadata.schema.as_ref(), &schema);
            assert!(!metadata.flag);
        }
        for mode in [AnalysisMode::IsLocal, AnalysisMode::IsStreaming] {
            let (bytes, schema) = metadata_payload(1);
            let metadata = decode_metadata(bytes, mode).unwrap();
            assert_eq!(metadata.schema.as_ref(), &schema);
            assert!(metadata.flag);
        }
    }
    #[test]
    fn metadata_decoder_rejects_missing_invalid_and_schema_true_flags() {
        for bytes in [vec![], vec![2], vec![255]] {
            for mode in [
                AnalysisMode::Schema,
                AnalysisMode::IsLocal,
                AnalysisMode::IsStreaming,
            ] {
                let status = decode_metadata(bytes.clone(), mode).err().unwrap();
                assert_eq!(status.code(), tonic::Code::Internal);
                assert_eq!(status.message(), "invalid Spark metadata flag");
            }
        }
        let (bytes, _) = metadata_payload(1);
        assert_eq!(
            decode_metadata(bytes, AnalysisMode::Schema)
                .err()
                .unwrap()
                .message(),
            "invalid Spark metadata flag"
        );
    }
    #[test]
    fn metadata_decoder_validates_ipc_for_every_mode() {
        for mode in [
            AnalysisMode::Schema,
            AnalysisMode::IsLocal,
            AnalysisMode::IsStreaming,
        ] {
            for bytes in [vec![0], vec![0, 1, 2, 3], vec![0, 255, 255, 255, 255]] {
                let status = decode_metadata(bytes, mode).err().unwrap();
                assert_eq!(status.code(), tonic::Code::Internal);
                assert!(status.message().starts_with("invalid Spark Arrow stream:"));
            }
        }
    }
    #[test]
    fn metadata_decoder_size_limit_excludes_transport_prefix() {
        let (bytes, schema) = metadata_payload(0);
        let ipc_limit = bytes.len() - 1;
        assert_eq!(
            decode_metadata_with_limit(bytes.clone(), AnalysisMode::Schema, ipc_limit)
                .unwrap()
                .schema
                .as_ref(),
            &schema
        );
        assert_eq!(
            decode_metadata_with_limit(bytes, AnalysisMode::Schema, ipc_limit - 1)
                .err()
                .unwrap()
                .code(),
            tonic::Code::ResourceExhausted
        );
    }

    #[derive(Debug)]
    struct CleanupAttachmentEvent {
        action: &'static str,
        core: usize,
        thread: std::thread::ThreadId,
        workers: usize,
        handles: usize,
    }
    struct TestCleanupWorker {
        core: Arc<JvmEngine>,
        events: mpsc::Sender<CleanupAttachmentEvent>,
        release: Option<mpsc::Receiver<()>>,
        _thread: PhantomData<Rc<()>>,
    }
    impl TestCleanupWorker {
        fn new(
            core: &Arc<JvmEngine>,
            events: &mpsc::Sender<CleanupAttachmentEvent>,
            release: Option<mpsc::Receiver<()>>,
        ) -> Self {
            let worker = Self {
                core: core.clone(),
                events: events.clone(),
                release,
                _thread: PhantomData,
            };
            worker.record("attach");
            worker
        }
        fn record(&self, action: &'static str) {
            let counts = self.core.counts.lock().unwrap();
            let _ = self.events.send(CleanupAttachmentEvent {
                action,
                core: Arc::as_ptr(&self.core) as usize,
                thread: std::thread::current().id(),
                workers: counts.workers,
                handles: counts.handles,
            });
        }
    }
    impl Drop for TestCleanupWorker {
        fn drop(&mut self) {
            self.record("detach");
            if let Some(release) = &self.release {
                // The test owns a CleanupBlock that also releases on unwind.
                let _ = release.recv();
            }
        }
    }
    fn empty_cleanup(core: &Arc<JvmEngine>) -> Cleanup {
        core.counts.lock().unwrap().handles += 1;
        Cleanup::Free {
            pointer: 0,
            core: core.clone(),
            permit: None,
            lane: CleanupLane::Task,
            retained_task: None,
        }
    }
    fn attachment_event(rx: &mpsc::Receiver<CleanupAttachmentEvent>) -> CleanupAttachmentEvent {
        rx.recv_timeout(Duration::from_secs(5)).unwrap()
    }
    struct CleanupTestLane {
        thread: std::thread::JoinHandle<()>,
        done: mpsc::Receiver<()>,
    }
    impl CleanupTestLane {
        fn spawn(action: impl FnOnce() + Send + 'static) -> Self {
            let (done_tx, done) = mpsc::channel();
            let thread = std::thread::spawn(move || {
                action();
                let _ = done_tx.send(());
            });
            Self { thread, done }
        }
        fn join(self) {
            self.done.recv_timeout(Duration::from_secs(5)).unwrap();
            self.thread.join().unwrap();
        }
    }
    fn cleanup_drained(core: Arc<JvmEngine>) {
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            core.wait_drained();
            let _ = done_tx.send(());
        });
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        waiter.join().unwrap();
    }
    #[test]
    fn cleanup_attachment_detaches_on_idle_without_closing_the_lane() {
        let core = JvmEngine::new(std::ptr::null_mut());
        let (tx, rx) = mpsc::channel();
        let (events_tx, events_rx) = mpsc::channel();
        assert!(tx.send(empty_cleanup(&core)).is_ok());
        let lane = CleanupTestLane::spawn(move || {
            // Zero makes the empty-queue timeout deterministic, independent of
            // scheduler speed; the sender stays open until detach is observed.
            cleanup_loop_with(rx, Duration::ZERO, |core| {
                Ok(TestCleanupWorker::new(core, &events_tx, None))
            });
        });
        let attached = attachment_event(&events_rx);
        let detached = attachment_event(&events_rx);
        assert_eq!((attached.action, detached.action), ("attach", "detach"));
        assert_eq!(attached.thread, detached.thread);
        assert_eq!((detached.workers, detached.handles), (1, 0));
        cleanup_drained(core.clone());
        assert_eq!(core.counts.lock().unwrap().workers, 0);
        drop(tx);
        lane.join();
    }
    #[test]
    fn cleanup_attachment_reuses_a_burst_and_detaches_before_switching_engines() {
        let first = JvmEngine::new(std::ptr::null_mut());
        let second = JvmEngine::new(std::ptr::null_mut());
        let (tx, rx) = mpsc::channel();
        let (events_tx, events_rx) = mpsc::channel();
        for core in [&first, &first, &second] {
            assert!(tx.send(empty_cleanup(core)).is_ok());
        }
        drop(tx);
        let first_core = first.clone();
        let lane = CleanupTestLane::spawn(move || {
            // All work and disconnect are queued before the receiver starts.
            cleanup_loop_with(rx, Duration::from_secs(60), |core| {
                if !Arc::ptr_eq(core, &first_core) {
                    assert_eq!(first_core.counts.lock().unwrap().workers, 0);
                }
                Ok(TestCleanupWorker::new(core, &events_tx, None))
            });
        });
        lane.join();
        let events = events_rx.try_iter().collect::<Vec<_>>();
        assert_eq!(
            events.iter().map(|event| event.action).collect::<Vec<_>>(),
            ["attach", "detach", "attach", "detach"]
        );
        assert_eq!(events[0].core, Arc::as_ptr(&first) as usize);
        assert_eq!(events[1].core, events[0].core);
        assert_eq!(events[2].core, Arc::as_ptr(&second) as usize);
        assert_eq!(events[3].core, events[2].core);
        assert!(events.iter().all(|event| event.thread == events[0].thread));
        assert!(events.iter().all(|event| event.workers == 1));
        assert_eq!((events[1].handles, events[3].handles), (0, 0));
        for core in [first, second] {
            assert_eq!(core.counts.lock().unwrap().workers, 0);
            cleanup_drained(core);
        }
    }
    #[test]
    fn cleanup_attachment_activity_outlives_a_blocked_detach_during_shutdown() {
        let core = JvmEngine::new(std::ptr::null_mut());
        let (tx, rx) = mpsc::channel();
        let (events_tx, events_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release = CleanupBlock(release_tx);
        assert!(tx.send(empty_cleanup(&core)).is_ok());
        drop(tx);
        let lane = CleanupTestLane::spawn(move || {
            let mut release_rx = Some(release_rx);
            cleanup_loop_with(rx, Duration::from_secs(60), |core| {
                Ok(TestCleanupWorker::new(core, &events_tx, release_rx.take()))
            });
        });
        let attached = attachment_event(&events_rx);
        let detaching = attachment_event(&events_rx);
        assert_eq!((attached.action, detaching.action), ("attach", "detach"));
        assert_eq!(attached.thread, detaching.thread);
        assert_eq!((detaching.workers, detaching.handles), (1, 0));
        core.shutdown();
        assert!(core
            .take_factory_after_drain()
            .unwrap_err()
            .contains("drained work"));
        let waiting_core = core.clone();
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            waiting_core.wait_drained();
            done_tx.send(()).unwrap();
        });
        assert_eq!(core.counts.lock().unwrap().workers, 1);
        drop(release);
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        waiter.join().unwrap();
        lane.join();
        assert_eq!(core.counts.lock().unwrap().workers, 0);
        core.release_factory().unwrap();
    }
    #[test]
    fn cleanup_attachment_failure_releases_activity_and_closed_engines_still_clean_up() {
        let core = JvmEngine::new(std::ptr::null_mut());
        let status = CleanupAttachment::attach_with(&core, |core| {
            assert_eq!(core.counts.lock().unwrap().workers, 1);
            Err::<TestCleanupWorker, _>(Status::internal("injected attach failure"))
        })
        .err()
        .unwrap();
        assert_eq!(status.code(), tonic::Code::Internal);
        assert_eq!(core.counts.lock().unwrap().workers, 0);
        core.shutdown();
        let (tx, rx) = mpsc::channel();
        assert!(tx.send(empty_cleanup(&core)).is_ok());
        drop(tx);
        cleanup_loop_with(
            rx,
            Duration::ZERO,
            |_| -> Result<TestCleanupWorker, Status> {
                panic!("closed cleanup must use the per-call fallback")
            },
        );
        assert_eq!(core.counts.lock().unwrap().handles, 0);
        assert_eq!(core.counts.lock().unwrap().workers, 0);
        cleanup_drained(core.clone());
        core.release_factory().unwrap();
    }

    #[test]
    fn cleanup_attachment_success_resets_failure_reporting() {
        let core = JvmEngine::new(std::ptr::null_mut());
        let (events_tx, _events_rx) = mpsc::channel();
        let mut attempts = 0;
        let mut attach = |core: &Arc<JvmEngine>| {
            attempts += 1;
            if attempts == 2 {
                Ok(TestCleanupWorker::new(core, &events_tx, None))
            } else {
                Err(Status::internal("injected attach failure"))
            }
        };
        let mut reported = None;
        let mut reports = 0;
        for (failed, expected_reports) in [(true, 1), (false, 1), (true, 2)] {
            let attachment =
                CleanupAttachment::attach_or_fallback(&core, &mut attach, &mut reported, |_| {
                    reports += 1
                });
            assert_eq!(attachment.is_none(), failed);
            assert_eq!(reports, expected_reports);
            assert_eq!(
                reported
                    .as_ref()
                    .is_some_and(|reported| Weak::ptr_eq(reported, &Arc::downgrade(&core))),
                failed
            );
            drop(attachment);
            assert_eq!(core.counts.lock().unwrap().workers, 0);
        }
        assert_eq!(attempts, 3);
        cleanup_drained(core);
    }

    #[test]
    fn cleanup_attachment_failure_reporting_tracks_engines_without_retaining_them() {
        let first = JvmEngine::new(std::ptr::null_mut());
        let second = JvmEngine::new(std::ptr::null_mut());
        let mut reported = None;
        let mut reports = 0;
        for (core, expected_reports) in [
            (&first, 1),
            (&first, 1),
            (&second, 2),
            (&second, 2),
            (&first, 3),
        ] {
            let attachment = CleanupAttachment::attach_or_fallback(
                core,
                |_| -> Result<TestCleanupWorker, Status> {
                    Err(Status::internal("injected attach failure"))
                },
                &mut reported,
                |_| reports += 1,
            );
            assert!(attachment.is_none());
            assert_eq!(reports, expected_reports);
            assert!(Weak::ptr_eq(
                reported.as_ref().unwrap(),
                &Arc::downgrade(core)
            ));
            assert_eq!(Arc::strong_count(core), 1);
            assert_eq!(core.counts.lock().unwrap().workers, 0);
        }
        cleanup_drained(first.clone());
        cleanup_drained(second.clone());
        drop(first);
        drop(second);
        assert!(reported.unwrap().upgrade().is_none());
    }

    #[test]
    fn cleanup_attachment_loop_retries_failures_and_drains_every_item() {
        let first = JvmEngine::new(std::ptr::null_mut());
        let second = JvmEngine::new(std::ptr::null_mut());
        let (tx, rx) = mpsc::channel();
        for core in [&first, &first, &second, &second] {
            assert!(tx.send(empty_cleanup(core)).is_ok());
        }
        drop(tx);
        let attempts = Arc::new(AtomicUsize::new(0));
        let worker_attempts = attempts.clone();
        let lane = CleanupTestLane::spawn(move || {
            cleanup_loop_with(
                rx,
                Duration::ZERO,
                |_| -> Result<TestCleanupWorker, Status> {
                    worker_attempts.fetch_add(1, Ordering::SeqCst);
                    Err(Status::internal("injected attach failure"))
                },
            );
        });
        lane.join();
        assert_eq!(attempts.load(Ordering::SeqCst), 4);
        for core in [first, second] {
            {
                let counts = core.counts.lock().unwrap();
                assert_eq!((counts.handles, counts.workers), (0, 0));
            }
            cleanup_drained(core);
        }
    }

    #[test]
    fn factory_release_requires_drain_and_transfers_ownership_once_with_core_arcs_retained() {
        // This is only an opaque ownership token, never a JNI handle. Prevent
        // the fallback destructor from receiving it if a test assertion fails.
        let token = std::ptr::NonNull::<u8>::dangling().as_ptr().cast();
        let core = std::mem::ManuallyDrop::new(JvmEngine::new(token));
        let error = core.take_factory_after_drain().unwrap_err();
        assert!(error.contains("requires shutdown"), "{error}");
        core.shutdown();
        let lease = ExportLease::new(Arc::clone(&core));
        let error = core.take_factory_after_drain().unwrap_err();
        assert!(error.contains("requires drained work"), "{error}");
        drop(lease);
        core.set_worker_attachment_count_for_test(1);
        let error = core.take_factory_after_drain().unwrap_err();
        assert!(error.contains("1 JNI worker attachments remain"), "{error}");
        core.set_worker_attachment_count_for_test(0);
        // A cleanup worker can retain an Arc after decrementing its counter.
        // Factory release must already complete while that Arc is still alive.
        let delayed_cleanup = Arc::clone(&core);
        assert_eq!(core.take_factory_after_drain().unwrap(), token);
        assert!(core.take_factory_after_drain().unwrap().is_null());
        assert!(delayed_cleanup.factory.load(Ordering::SeqCst).is_null());
        core.release_factory().unwrap();
        drop(delayed_cleanup);
        // Ownership was taken above; the remaining destructor is a C no-op.
        drop(std::mem::ManuallyDrop::into_inner(core));
    }
    #[test]
    fn shutdown_waits_for_exported_buffers_to_be_released() {
        let _serial = CLEANUP_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let core = JvmEngine::new(std::ptr::null_mut());
        let lease = ExportLease::new(core.clone());
        core.shutdown();
        let (started_tx, started_rx) = mpsc::channel();
        let (drained_tx, drained_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            core.wait_drained();
            drained_tx.send(()).unwrap();
        });
        started_rx.recv().unwrap();
        assert!(drained_rx.recv_timeout(Duration::from_millis(20)).is_err());
        // A real final-buffer callback queues this pair. Null owner exercises
        // that path without a JVM; C treats it as an empty cleanup operation.
        queue(Cleanup::Arrow { owner: 0, lease });
        drained_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        waiter.join().unwrap();
    }
    #[test]
    fn failed_cleanup_retains_session_admission_until_explicit_retry() {
        let core = JvmEngine::new(std::ptr::null_mut());
        let permit = core
            .session_permits
            .clone()
            .try_acquire_many_owned(64)
            .unwrap();
        core.counts.lock().unwrap().handles += 1;
        let item = Cleanup::Free {
            pointer: 0,
            core: core.clone(),
            permit: Some(permit),
            lane: CleanupLane::Session,
            retained_task: None,
        };
        let attempts = AtomicUsize::new(0);
        cleanup_with(item, |resource, pointer| {
            assert!(matches!(resource, CleanupResource::Handle));
            assert_eq!(pointer, 0);
            attempts.fetch_add(1, Ordering::SeqCst);
            Err(Status::internal("injected Java close failure"))
        });
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert_eq!(core.session_permits.available_permits(), 0);
        assert_eq!(core.counts.lock().unwrap().handles, 1);
        assert!(core
            .cleanup_failure()
            .unwrap()
            .contains("injected Java close failure"));
        let mut failed = core.take_failed_cleanup();
        assert_eq!(failed.len(), 1);
        assert!(matches!(failed[0].item.lane(), CleanupLane::Session));
        cleanup_with(failed.pop().unwrap().item, |_, _| {
            attempts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(core.session_permits.available_permits(), 64);
        assert_eq!(core.counts.lock().unwrap().handles, 0);
        assert!(core.cleanup_failure().is_none());
    }
    #[test]
    fn failed_query_cleanup_retains_its_task_and_retries_on_the_query_lane() {
        let _serial = CLEANUP_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let core = JvmEngine::new(std::ptr::null_mut());
        let permit = core.permits.clone().try_acquire_many_owned(64).unwrap();
        let task = empty_handle(&core, HandleKind::Task, Some(permit), None);
        core.counts.lock().unwrap().handles += 1;
        cleanup_with(
            Cleanup::Free {
                pointer: 0,
                core: core.clone(),
                permit: None,
                lane: CleanupLane::Query,
                retained_task: Some(task),
            },
            |_, _| Err(Status::internal("injected query close failure")),
        );
        cleanup_fence(CleanupLane::Task);
        assert_eq!(core.permits.available_permits(), 0);
        assert_eq!(core.counts.lock().unwrap().handles, 2);
        let blocked = block_cleanup(CleanupLane::Query);
        core.retry_failed_cleanup();
        cleanup_fence(CleanupLane::Task);
        assert_eq!(core.permits.available_permits(), 0);
        assert_eq!(core.counts.lock().unwrap().handles, 2);
        drop(blocked);
        cleanup_fence(CleanupLane::Query);
        cleanup_fence(CleanupLane::Task);
        assert_eq!(core.permits.available_permits(), 64);
        assert_eq!(core.counts.lock().unwrap().handles, 0);
        assert!(core.cleanup_failure().is_none());
    }
    #[test]
    fn failed_arrow_cleanup_keeps_shutdown_pending_until_successful_retry() {
        let core = JvmEngine::new(std::ptr::null_mut());
        let lease = ExportLease::new(core.clone());
        cleanup_with(Cleanup::Arrow { owner: 0, lease }, |resource, _| {
            assert!(matches!(resource, CleanupResource::Arrow));
            Err(Status::internal("injected Arrow allocator close failure"))
        });
        assert_eq!(core.counts.lock().unwrap().exports, 1);
        core.shutdown();
        let waiting = core.clone();
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            waiting.wait_drained();
            done_tx.send(()).unwrap();
        });
        started_rx.recv().unwrap();
        assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());
        core.retry_failed_cleanup();
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        waiter.join().unwrap();
        assert_eq!(core.counts.lock().unwrap().exports, 0);
        assert!(core.cleanup_failure().is_none());
    }
    #[test]
    fn cleanup_retry_snapshot_preserves_failures_arriving_during_retry() {
        let core = JvmEngine::new(std::ptr::null_mut());
        cleanup_with(empty_cleanup(&core), |_, _| Err(Status::internal("first")));
        let mut retry = core.take_failed_cleanup();
        cleanup_with(empty_cleanup(&core), |_, _| {
            Err(Status::internal("concurrent"))
        });
        cleanup_with(retry.pop().unwrap().item, |_, _| Ok(()));
        assert_eq!(core.counts.lock().unwrap().handles, 1);
        assert!(core.cleanup_failure().unwrap().contains("concurrent"));
        let mut remaining = core.take_failed_cleanup();
        assert_eq!(remaining.len(), 1);
        cleanup_with(remaining.pop().unwrap().item, |_, _| Ok(()));
        assert_eq!(core.counts.lock().unwrap().handles, 0);
    }
    #[test]
    fn failed_factory_cleanup_restores_the_same_handle_for_explicit_retry() {
        let token = std::ptr::NonNull::<u8>::dangling().as_ptr().cast();
        // Do not let an assertion unwind send the opaque token to real JNI.
        let core = std::mem::ManuallyDrop::new(JvmEngine::new(token));
        core.shutdown();
        let error = core
            .release_factory_with(|resource, pointer| {
                assert!(matches!(resource, CleanupResource::Handle));
                assert_eq!(pointer, token as usize);
                Err(Status::internal("injected factory close failure"))
            })
            .unwrap_err();
        assert!(error.starts_with("JNI cleanup is incomplete; retained cleanup failed:"));
        assert_eq!(core.factory.load(Ordering::SeqCst), token);
        core.release_factory_with(|_, pointer| {
            assert_eq!(pointer, token as usize);
            Ok(())
        })
        .unwrap();
        assert!(core.factory.load(Ordering::SeqCst).is_null());
        drop(std::mem::ManuallyDrop::into_inner(core));
    }
    struct CleanupBlock(mpsc::Sender<()>);
    impl Drop for CleanupBlock {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }
    fn block_cleanup(lane: CleanupLane) -> CleanupBlock {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        queue(Cleanup::TestAction {
            lane,
            action: Box::new(move || {
                if entered_tx.send(()).is_ok() {
                    let _ = release_rx.recv();
                }
            }),
        });
        let block = CleanupBlock(release_tx);
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        block
    }
    fn cleanup_fence(lane: CleanupLane) {
        let (done_tx, done_rx) = mpsc::channel();
        queue(Cleanup::TestAction {
            lane,
            action: Box::new(move || {
                let _ = done_tx.send(());
            }),
        });
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    }
    fn empty_handle(
        core: &Arc<JvmEngine>,
        kind: HandleKind,
        permit: Option<OwnedSemaphorePermit>,
        retained_task: Option<Arc<Handle>>,
    ) -> Arc<Handle> {
        core.counts.lock().unwrap().handles += 1;
        Arc::new(Handle {
            // C spark_connect_engine_free(NULL) is a no-op; counters/permits still take the exact
            // production deferred cleanup path without a running JVM.
            pointer: std::ptr::null_mut(),
            core: core.clone(),
            permit: Mutex::new(permit),
            kind,
            retained_task,
        })
    }
    #[test]
    fn slow_session_close_does_not_block_task_admission_reclamation() {
        let _serial = CLEANUP_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let blocked = block_cleanup(CleanupLane::Session);
        let core = JvmEngine::new(std::ptr::null_mut());
        let session_permit = core
            .session_permits
            .clone()
            .try_acquire_many_owned(64)
            .unwrap();
        let task_permit = core.permits.clone().try_acquire_many_owned(64).unwrap();
        drop(empty_handle(
            &core,
            HandleKind::Session,
            Some(session_permit),
            None,
        ));
        drop(empty_handle(
            &core,
            HandleKind::Task,
            Some(task_permit),
            None,
        ));
        cleanup_fence(CleanupLane::Task);
        assert_eq!(core.permits.available_permits(), 64);
        assert_eq!(core.session_permits.available_permits(), 0);
        assert_eq!(core.counts.lock().unwrap().handles, 1);
        drop(blocked);
        core.wait_drained();
        assert_eq!(core.session_permits.available_permits(), 64);
    }
    #[test]
    fn query_close_retains_task_admission_until_query_cleanup_finishes() {
        let _serial = CLEANUP_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let blocked = block_cleanup(CleanupLane::Query);
        let core = JvmEngine::new(std::ptr::null_mut());
        let permit = core.permits.clone().try_acquire_many_owned(64).unwrap();
        let task = empty_handle(&core, HandleKind::Task, Some(permit), None);
        drop(empty_handle(&core, HandleKind::Query, None, Some(task)));
        cleanup_fence(CleanupLane::Task);
        assert_eq!(core.permits.available_permits(), 0);
        assert_eq!(core.counts.lock().unwrap().handles, 2);
        drop(blocked);
        core.wait_drained();
        assert_eq!(core.permits.available_permits(), 64);
    }
    #[test]
    fn slow_session_close_does_not_block_query_task_reclamation() {
        let _serial = CLEANUP_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let blocked = block_cleanup(CleanupLane::Session);
        let core = JvmEngine::new(std::ptr::null_mut());
        let permit = core.permits.clone().try_acquire_many_owned(64).unwrap();
        let task = empty_handle(&core, HandleKind::Task, Some(permit), None);
        drop(empty_handle(&core, HandleKind::Query, None, Some(task)));
        cleanup_fence(CleanupLane::Query);
        cleanup_fence(CleanupLane::Task);
        assert_eq!(core.permits.available_permits(), 64);
        assert_eq!(core.counts.lock().unwrap().handles, 0);
        drop(blocked);
        core.wait_drained();
    }
    #[tokio::test]
    async fn session_admission_times_out_and_recovers_after_actual_session_free() {
        let core = JvmEngine::new(std::ptr::null_mut());
        let held = core
            .session_permits
            .clone()
            .try_acquire_many_owned(64)
            .unwrap();
        let status = core
            .admit_session(Instant::now() + Duration::from_millis(2))
            .await
            .err()
            .unwrap();
        assert_eq!(status.code(), tonic::Code::ResourceExhausted, "{status}");
        assert_eq!(core.permits.available_permits(), 64);
        {
            let _serial = CLEANUP_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            drop(empty_handle(&core, HandleKind::Session, Some(held), None));
            cleanup_fence(CleanupLane::Session);
        }
        let restored = core
            .admit_session(Instant::now() + Duration::from_millis(2))
            .await
            .unwrap();
        assert_eq!(core.session_permits.available_permits(), 63);
        drop(restored);
        core.wait_drained();
    }
    #[tokio::test(start_paused = true)]
    async fn task_admission_bounds_shared_query_calls_and_recovers_after_capacity_returns() {
        let core = JvmEngine::new(std::ptr::null_mut());
        let held = core.permits.clone().try_acquire_many_owned(64).unwrap();
        let session = JvmSession {
            core: core.clone(),
            context: SessionContext {
                session_id: "task-capacity-session".into(),
                user_context: None,
            },
            handle: OnceCell::new(),
            catalog: RwLock::new(()),
        };
        // Cache a synthetic session so every call reaches task admission without
        // crossing JNI. Its null handle is safe to reclaim through the C shim.
        assert!(session
            .handle
            .set(empty_handle(&core, HandleKind::Session, None, None))
            .is_ok());
        let relation = || proto::Relation {
            rel_type: Some(proto::relation::RelType::Sql(proto::Sql {
                query: "SELECT 1".into(),
                ..Default::default()
            })),
            ..Default::default()
        };
        {
            // Read-only preparation must overlap an existing catalog reader.
            // An accidental exclusive prepare lock would never reach its timer.
            let _reader = CatalogAccess::Read.lock(&session.catalog).await;
            let calls = async {
                futures::join!(
                    async {
                        session
                            .prepare(proto::plan::OpType::Root(relation()))
                            .await
                            .map(|_| ())
                    },
                    async {
                        session
                            .analyze(proto::analyze_plan_request::Analyze::Schema(
                                proto::analyze_plan_request::Schema {
                                    plan: Some(proto::Plan {
                                        op_type: Some(proto::plan::OpType::Root(relation())),
                                    }),
                                },
                            ))
                            .await
                            .map(|_| ())
                    },
                    async { session.config(config_get()).await.map(|_| ()) },
                )
            };
            tokio::pin!(calls);
            let started = Instant::now();
            assert!(futures::poll!(&mut calls).is_pending());
            assert!(core.counts.lock().unwrap().tasks.is_empty());
            tokio::time::advance(SESSION_ADMISSION_TIMEOUT).await;
            let results = match futures::poll!(&mut calls) {
                std::task::Poll::Ready(results) => results,
                std::task::Poll::Pending => panic!("task admission exceeded its deadline"),
            };
            for result in [results.0, results.1, results.2] {
                let status = result.unwrap_err();
                assert_eq!(status.code(), tonic::Code::ResourceExhausted, "{status}");
                assert_eq!(status.message(), "Spark engine task capacity exhausted");
            }
            assert!(started.elapsed() <= SESSION_ADMISSION_TIMEOUT);
        }
        assert!(session.catalog.try_write().is_ok());
        assert!(core.counts.lock().unwrap().tasks.is_empty());
        assert_eq!(core.counts.lock().unwrap().workers, 0);
        assert_eq!(core.session_permits.available_permits(), 64);
        drop(held);
        let restored = core
            .task(Instant::now() + SESSION_ADMISSION_TIMEOUT)
            .await
            .unwrap();
        assert_eq!(core.permits.available_permits(), 63);
        drop(restored);
        assert_eq!(core.permits.available_permits(), 64);
        drop(session);
        core.wait_drained();
    }
    #[tokio::test(start_paused = true)]
    async fn task_admission_waits_are_cancelled_by_future_drop_and_engine_shutdown() {
        let core = JvmEngine::new(std::ptr::null_mut());
        let held = core.permits.clone().try_acquire_many_owned(64).unwrap();
        let mut cancelled = Box::pin(core.task(Instant::now() + SESSION_ADMISSION_TIMEOUT));
        assert!(futures::poll!(&mut cancelled).is_pending());
        drop(cancelled);
        drop(held);
        assert!(core.counts.lock().unwrap().tasks.is_empty());
        // Cancelling the wait must neither leak a permit nor create a JNI task.
        let _held = core.permits.clone().try_acquire_many_owned(64).unwrap();
        let pending = core.task(Instant::now() + SESSION_ADMISSION_TIMEOUT);
        tokio::pin!(pending);
        assert!(futures::poll!(&mut pending).is_pending());
        core.shutdown();
        let status = pending.await.err().unwrap();
        assert_eq!(status.code(), tonic::Code::Unavailable, "{status}");
        assert!(core.counts.lock().unwrap().tasks.is_empty());
        assert_eq!(core.counts.lock().unwrap().workers, 0);
        core.wait_drained();
    }
    fn blocked_session(id: &str) -> (Arc<JvmEngine>, OwnedSemaphorePermit, Arc<dyn EngineSession>) {
        let core = JvmEngine::new(std::ptr::null_mut());
        let held = core
            .session_permits
            .clone()
            .try_acquire_many_owned(64)
            .unwrap();
        let factory = EngineFactory(core.clone());
        let session = factory
            .create_session(SessionContext {
                session_id: id.into(),
                user_context: None,
            })
            .unwrap();
        (core, held, session)
    }
    async fn assert_config_admission_deadlines(
        operations: Vec<proto::config_request::operation::OpType>,
    ) {
        let (core, _held, session) = blocked_session("concurrent-blocked-session");
        let calls = futures::future::join_all(operations.into_iter().map(|input| {
            let session = &session;
            async move {
                let started = Instant::now();
                let result = session.config(input).await;
                (result, started.elapsed())
            }
        }));
        tokio::pin!(calls);
        // Poll every real Config call before advancing time. The scenarios
        // exercise shared readers, queued writers and mixed access under
        // heavyweight session capacity pressure.
        assert!(futures::poll!(&mut calls).is_pending());
        assert_eq!(core.permits.available_permits(), 64);
        tokio::time::advance(SESSION_ADMISSION_TIMEOUT).await;
        for (result, elapsed) in calls.await {
            assert!(
                matches!(&result, Err(status) if status.code() == tonic::Code::ResourceExhausted),
                "{:?}",
                result.as_ref().err().map(|s| (s.code(), s.message()))
            );
            assert!(elapsed <= SESSION_ADMISSION_TIMEOUT, "{elapsed:?}");
        }
        assert_eq!(core.permits.available_permits(), 64);
        core.wait_drained();
    }
    fn config_get() -> proto::config_request::operation::OpType {
        proto::config_request::operation::OpType::Get(proto::config_request::Get {
            keys: vec!["key".into()],
        })
    }
    fn config_set() -> proto::config_request::operation::OpType {
        proto::config_request::operation::OpType::Set(proto::config_request::Set {
            pairs: vec![proto::KeyValue {
                key: "key".into(),
                value: Some("value".into()),
            }],
            silent: None,
        })
    }
    #[tokio::test(start_paused = true)]
    async fn concurrent_first_config_calls_do_not_restart_admission_timeouts() {
        assert_config_admission_deadlines(vec![config_get(), config_get(), config_get()]).await;
    }
    #[tokio::test(start_paused = true)]
    async fn silent_config_requests_preserve_native_admission_errors() {
        let input = proto::config_request::operation::OpType::Set(proto::config_request::Set {
            pairs: vec![proto::KeyValue {
                key: "spark.application.option".into(),
                value: Some("value".into()),
            }],
            silent: Some(true),
        });
        assert_config_admission_deadlines(vec![input]).await;
    }
    #[tokio::test(start_paused = true)]
    async fn concurrent_first_config_writers_do_not_restart_admission_timeouts() {
        assert_config_admission_deadlines(vec![config_set(), config_set(), config_set()]).await;
    }
    #[tokio::test(start_paused = true)]
    async fn config_readers_behind_writers_keep_their_original_admission_deadlines() {
        assert_config_admission_deadlines(vec![config_set(), config_get(), config_set()]).await;
    }
    #[tokio::test(start_paused = true)]
    async fn prepare_and_analyze_behind_config_keep_their_original_admission_deadlines() {
        let (core, _held, session) = blocked_session("mixed-blocked-session");
        let relation = || proto::Relation {
            rel_type: Some(proto::relation::RelType::Sql(proto::Sql {
                query: "SELECT 1".into(),
                ..Default::default()
            })),
            ..Default::default()
        };
        let calls = async {
            futures::join!(
                async { session.config(config_set()).await.map(|_| ()) },
                async {
                    session
                        .prepare(proto::plan::OpType::Root(relation()))
                        .await
                        .map(|_| ())
                },
                async {
                    session
                        .analyze(proto::analyze_plan_request::Analyze::Schema(
                            proto::analyze_plan_request::Schema {
                                plan: Some(proto::Plan {
                                    op_type: Some(proto::plan::OpType::Root(relation())),
                                }),
                            },
                        ))
                        .await
                        .map(|_| ())
                },
            )
        };
        tokio::pin!(calls);
        let started = Instant::now();
        // Config Set enters first; prepare and analyze queue behind its writer.
        assert!(futures::poll!(&mut calls).is_pending());
        assert_eq!(core.permits.available_permits(), 64);
        tokio::time::advance(SESSION_ADMISSION_TIMEOUT).await;
        let (config, prepare, analyze) = calls.await;
        for result in [config, prepare, analyze] {
            assert!(
                matches!(&result, Err(status) if status.code() == tonic::Code::ResourceExhausted),
                "{:?}",
                result.as_ref().err().map(|s| (s.code(), s.message()))
            );
        }
        let elapsed = started.elapsed();
        assert!(elapsed <= SESSION_ADMISSION_TIMEOUT, "{elapsed:?}");
        assert_eq!(core.permits.available_permits(), 64);
        core.wait_drained();
    }
    #[tokio::test(start_paused = true)]
    async fn invalid_inputs_preserve_errors_without_waiting_for_session_admission() {
        let (core, _held, session) = blocked_session("invalid-input-session");
        let started = Instant::now();
        let status = session
            .analyze(proto::analyze_plan_request::Analyze::Explain(
                Default::default(),
            ))
            .await
            .err()
            .unwrap();
        assert_eq!(status.code(), tonic::Code::Unimplemented, "{status}");
        assert_eq!(status.message(), "Spark engine analysis is not implemented");
        let status = session
            .analyze(proto::analyze_plan_request::Analyze::Schema(
                proto::analyze_plan_request::Schema { plan: None },
            ))
            .await
            .err()
            .unwrap();
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status}");
        assert_eq!(status.message(), "analysis requires a root relation");
        let status = session
            .prepare(proto::plan::OpType::Command(proto::Command {
                command_type: Some(proto::command::CommandType::SqlCommand(Default::default())),
            }))
            .await
            .err()
            .unwrap();
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status}");
        assert_eq!(status.message(), "missing SQL input");
        // Exercise legacy argument rejection before missing-input validation.
        #[allow(deprecated)]
        let sql_with_arguments = proto::SqlCommand {
            pos_arguments: vec![Default::default()],
            ..Default::default()
        };
        let status = session
            .prepare(proto::plan::OpType::Command(proto::Command {
                command_type: Some(proto::command::CommandType::SqlCommand(sql_with_arguments)),
            }))
            .await
            .err()
            .unwrap();
        assert_eq!(status.code(), tonic::Code::Unimplemented, "{status}");
        assert_eq!(status.message(), "Spark engine SQL command arguments");
        let result = session
            .config(proto::config_request::operation::OpType::Set(
                proto::config_request::Set {
                    silent: Some(true),
                    ..Default::default()
                },
            ))
            .await
            .unwrap();
        assert!(result.pairs.is_empty() && result.warnings.is_empty());
        assert_eq!(started.elapsed(), Duration::ZERO);
        assert_eq!(core.session_permits.available_permits(), 0);
        assert_eq!(core.permits.available_permits(), 64);
        assert_eq!(core.counts.lock().unwrap().handles, 0);
        core.wait_drained();
    }
    #[tokio::test]
    async fn queued_config_writer_precedes_later_catalog_readers() {
        let catalog = RwLock::new(());
        let first_reader = CatalogAccess::Read.lock(&catalog).await;
        let writer = CatalogAccess::config(6).lock(&catalog);
        tokio::pin!(writer);
        assert!(futures::poll!(&mut writer).is_pending());
        let late_reader = CatalogAccess::config(1).lock(&catalog);
        tokio::pin!(late_reader);
        assert!(futures::poll!(&mut late_reader).is_pending());
        drop(first_reader);
        let writer_guard = writer.await;
        assert!(matches!(&writer_guard, CatalogGuard::Write { .. }));
        assert!(futures::poll!(&mut late_reader).is_pending());
        drop(writer_guard);
        let late_reader_guard = late_reader.await;
        assert!(matches!(&late_reader_guard, CatalogGuard::Read { .. }));
        // A Config Get guard remains shared while another Analyze reader enters.
        let overlapping_reader = CatalogAccess::Read.lock(&catalog);
        tokio::pin!(overlapping_reader);
        assert!(futures::poll!(&mut overlapping_reader).is_ready());
    }
    #[tokio::test]
    async fn shutdown_wakes_pending_session_admission_without_consuming_task_slots() {
        let (core, _held, session) = blocked_session("blocked-session");
        let pending = session.config(config_get());
        tokio::pin!(pending);
        assert!(futures::poll!(&mut pending).is_pending());
        assert_eq!(core.permits.available_permits(), 64);
        core.shutdown();
        let result = pending.await;
        assert!(
            matches!(&result, Err(status) if status.code() == tonic::Code::Unavailable),
            "{:?}",
            result.as_ref().err().map(|s| (s.code(), s.message()))
        );
        core.wait_drained();
    }
    #[test]
    fn typed_errors_preserve_codes_and_reject_ok_or_invalid_codes() {
        assert_eq!(
            decode_error("JVM_ENGINE_ERROR:12:unsupported: feature").code(),
            tonic::Code::Unimplemented
        );
        assert_eq!(
            decode_error("JVM_ENGINE_ERROR:1:cancelled 😀").message(),
            "cancelled 😀"
        );
        for text in [
            "JVM_ENGINE_ERROR:0:bad",
            "JVM_ENGINE_ERROR:99:bad",
            "plain failure",
        ] {
            assert_eq!(decode_error(text).code(), tonic::Code::Internal);
        }
    }
    #[tokio::test]
    async fn shutdown_cancels_pending_tokens_before_their_first_jni_call() {
        // No JNI call occurs: cancellation is observed before task publication.
        let core = JvmEngine::new(std::ptr::null_mut());
        let guard = core
            .task(Instant::now() + SESSION_ADMISSION_TIMEOUT)
            .await
            .unwrap();
        core.shutdown();
        assert!(
            matches!(guard.task.initialize(), Err(status) if status.code() == tonic::Code::Cancelled)
        );
        assert!(core
            .task(Instant::now() + SESSION_ADMISSION_TIMEOUT)
            .await
            .is_err());
        drop(guard);
        core.wait_drained();
    }
    #[test]
    fn config_codec_preserves_null_unicode_and_empty() {
        let pairs = vec![
            proto::KeyValue {
                key: "时区".into(),
                value: Some("😀".into()),
            },
            proto::KeyValue {
                key: "".into(),
                value: None,
            },
            proto::KeyValue {
                key: "empty".into(),
                value: Some("".into()),
            },
        ];
        assert_eq!(pairs, decode_pairs(&encode_pairs(&pairs).unwrap()).unwrap());
    }
    #[tokio::test]
    async fn silent_config_sets_continue_after_per_key_errors_without_disclosing_values() {
        let pairs = ["first", "blocked", "last"]
            .into_iter()
            .map(|key| proto::KeyValue {
                key: key.into(),
                value: Some("sensitive-value".into()),
            })
            .collect::<Vec<_>>();
        let mut attempted = Vec::new();
        let result = silent_config_set(pairs.clone(), |pair| {
            attempted.push(pair.clone());
            std::future::ready(if pair.key == "blocked" {
                Err(Status::invalid_argument("rejected sensitive-value"))
            } else {
                Ok(Vec::new())
            })
        })
        .await
        .unwrap();
        assert_eq!(attempted, pairs);
        assert!(result.pairs.is_empty());
        assert_eq!(result.warnings, ["Failed to set blocked (InvalidArgument)"]);
    }
    #[tokio::test]
    async fn silent_config_sets_stop_on_cancellation_shutdown_and_adapter_failures() {
        for code in [
            tonic::Code::Cancelled,
            tonic::Code::DeadlineExceeded,
            tonic::Code::ResourceExhausted,
            tonic::Code::FailedPrecondition,
            tonic::Code::Unavailable,
            tonic::Code::Internal,
        ] {
            let mut attempted = 0;
            let pairs = ["first", "never"]
                .into_iter()
                .map(|key| proto::KeyValue {
                    key: key.into(),
                    value: Some("value".into()),
                })
                .collect();
            let error = silent_config_set(pairs, |_| {
                attempted += 1;
                std::future::ready(Err(Status::new(code, "operation failed")))
            })
            .await
            .err()
            .unwrap();
            assert_eq!(error.code(), code);
            assert_eq!(error.message(), "operation failed");
            assert_eq!(attempted, 1);
        }
    }
    #[test]
    fn config_codec_matches_java_data_stream_format() {
        let pairs = vec![proto::KeyValue {
            key: "k".into(),
            value: Some("v".into()),
        }];
        let bytes = [0, 0, 0, 1, 0, 0, 0, 1, b'k', 0, 0, 0, 1, b'v'];
        assert_eq!(encode_pairs(&pairs).unwrap(), bytes);
        assert_eq!(decode_pairs(&bytes).unwrap(), pairs);
    }
    #[test]
    fn config_codec_rejects_trailing_and_invalid_lengths() {
        assert!(decode_pairs(&[0, 0, 0, 0, 1]).is_err());
        assert!(decode_pairs(&[0, 0, 0, 1, 0, 0, 0, 0, 255, 255, 255, 254]).is_err());
        assert!(decode_pairs(&[]).is_err());
        assert!(decode_pairs(&[255, 255, 255, 255]).is_err());
        assert!(decode_pairs(&[0, 0, 0, 1, 255, 255, 255, 255, 0, 0, 0, 0]).is_err());
        assert!(decode_pairs(&[0, 0, 0, 1, 0, 0, 0, 2, b'k', 0, 0, 0, 0]).is_err());
    }
}
