//! SparkSession implementation mirroring `pyspark.sql.SparkSession`.
//!
//! Provides the entry point for DataFrame operations and SQL queries.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use spark_connect_core::channel::ChannelBuilder;
use spark_connect_core::client::{ExecutePlanHook, SparkConnectClient};
use spark_connect_core::client_log::ExecutionLog;
use spark_connect_core::error::{Result, SparkError};
use spark_connect_core::runtime::{block_on, calling_thread_id, check_callback_depth};
use spark_connect_core::thread_entries::forget_when_thread_ends;
use spark_connect_proto as proto;

use crate::dataframe::DataFrame;
use crate::plan::{LogicalPlan, PlanIds, SharedPlan};
use crate::profiler::PythonAccumulators;
use crate::progress::{ExecutionStream, ProgressTracker, ProgressUpdate};
use crate::row::{Row, Value};
use crate::types::DataType;

/// A progress handler of the executions of a session, as PySpark's `ProgressHandler`: it is
/// called with each progress of the server that has tasks, and once more, as done, when the
/// execution ends. Mirrors `SparkSession.registerProgressHandler`.
pub type ProgressHandler = Arc<dyn Fn(&ProgressUpdate) + Send + Sync>;

/// A progress handler that may fail, as [`SparkSession::register_fallible_progress_handler`]
/// registers: its error fails the execution that calls it.
pub(crate) type FallibleProgressHandler = Arc<dyn Fn(&ProgressUpdate) -> Result<()> + Send + Sync>;

/// The shared registry of progress handlers and their ids, without retaining a session.
pub(crate) type ProgressHandlers = Arc<Mutex<Vec<(u64, FallibleProgressHandler)>>>;

/// A check of the metrics that an execution observed for an Observation of its plan, made before
/// they are set, as [`SparkSession::set_observation_metrics_check`] sets it.
pub type ObservationMetricsCheck =
    Arc<dyn Fn(&proto::execute_plan_response::ObservedMetrics) -> Result<()> + Send + Sync>;

/// A handler of the warning of an ML fit, as [`SparkSession::set_ml_fit_warning_handler`] sets
/// it.
type MlFitWarningHandler = Arc<dyn Fn(&str) -> Result<()> + Send + Sync>;

/// Metrics captured from the last executed query. Mirrors the data behind
/// `pyspark.sql.DataFrame.executionInfo` / `client.core.ExecutionInfo`.
#[derive(Debug, Clone, Default)]
pub struct ExecutionInfo {
    /// Plan metrics (typically present on the final response of an execution).
    pub metrics: Option<proto::execute_plan_response::Metrics>,
    /// Observed metrics collected during execution (also where `observe(...)`
    /// and UDF-profiler results surface).
    pub observed_metrics: Vec<proto::execute_plan_response::ObservedMetrics>,
    /// The ExecutePlan operation id: assigned by the client before the request is sent
    /// (the request's id, or a new UUID) and echoed by the server in
    /// `ExecutePlanResponse.operation_id`. Mirrors `ExecutionInfo.operation_id` (new in
    /// pyspark 4.3.0).
    pub operation_id: Option<String>,
}

/// The operation tags of a session, per thread, as PySpark Connect keeps them in the
/// `thread_local` of its client: those that a thread added, which the operations that it
/// starts carry. A new thread has none. The tags of a thread are a set that it may share with
/// other threads ([`SharedTags`]), as the threads of PySpark's `inheritable_thread_target` share
/// the set of tags that it copied. A thread is the calling thread of the core
/// ([`calling_thread_id`]), whose user context extensions are keyed by it too.
#[derive(Default)]
pub(crate) struct ThreadTags {
    by_thread: Arc<Mutex<std::collections::HashMap<std::thread::ThreadId, SharedTags>>>,
}

/// Forget the tags of a thread that ends, as PySpark's `thread_local` is freed with its thread.
fn forget_thread_tags(
    by_thread: &mut std::collections::HashMap<std::thread::ThreadId, SharedTags>,
    thread: std::thread::ThreadId,
) {
    by_thread.remove(&thread);
}

/// A set of operation tags of threads: a copy of the tags of a thread, of
/// [`SparkSession::copy_thread_tags`], which the threads given it by
/// [`SparkSession::share_thread_tags`] share, as the threads of PySpark's
/// `inheritable_thread_target` share the copy of the `thread_local` of the client of the thread
/// that decorated their function. The tags that one of them adds or removes are those of all.
#[derive(Clone, Default)]
pub struct SharedTags(Arc<Mutex<Vec<Tag>>>);

impl SharedTags {
    /// A set of the tags `tags`, which no thread has yet (see [`SparkSession::share_thread_tags`]),
    /// as the set that a binding keeps of the tags of PySpark's `thread_local`. Not a stable API.
    #[doc(hidden)]
    pub fn new(tags: Vec<Tag>) -> Self {
        SharedTags(Arc::new(Mutex::new(tags)))
    }

    /// Replace the tags of this set with `tags`, which the requests of the threads that have it
    /// carry from now on, as a change of the set of PySpark's `thread_local.tags`. Not a stable
    /// API.
    #[doc(hidden)]
    pub fn replace(&self, tags: Vec<Tag>) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = tags;
    }

    /// The tags of this set. Not a stable API.
    #[doc(hidden)]
    pub fn tags(&self) -> Vec<Tag> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// An operation tag: its text, or the UTF-32 little-endian bytes of a str of Python that UTF-8
/// cannot encode, such as one of a lone surrogate, which PySpark's client keeps as a tag, and
/// with which each request that carries the tags fails (see
/// [`SparkSession::add_unencodable_tag`]). Not a stable API.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tag {
    /// A tag of its text.
    Text(String),
    /// The UTF-32 little-endian bytes of a tag that UTF-8 cannot encode.
    Unencodable(Vec<u8>),
    /// A tag of PySpark's client that is neither a str nor bytes, which the string field of the
    /// tags of its request refuses with the `TypeError` of this message when it builds an
    /// execution, as a tags value that is not iterable fails `list()` of it there.
    NotText(String),
    /// A tag of bytes that are not UTF-8, which the string field of the tags of PySpark's request
    /// refuses with the `UnicodeDecodeError` of its decoder when it builds an execution.
    NotUtf8(Vec<u8>),
}

impl ThreadTags {
    /// The tags of the current thread.
    fn current(&self) -> Vec<Tag> {
        let shared = self
            .by_thread
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&calling_thread_id())
            .cloned();
        shared
            .map(|tags| tags.0.lock().unwrap_or_else(|e| e.into_inner()).clone())
            .unwrap_or_default()
    }

    /// The tags of the current thread, which has an empty set of them from now on if it had
    /// none, as PySpark's `get_tags` gives a `thread_local` of no tags a set.
    fn current_or_new(&self) -> Vec<Tag> {
        let shared = {
            let mut by_thread = self.by_thread.lock().unwrap_or_else(|e| e.into_inner());
            let thread = calling_thread_id();
            if !by_thread.contains_key(&thread) {
                forget_when_thread_ends(&self.by_thread, forget_thread_tags);
            }
            by_thread.entry(thread).or_default().clone()
        };
        let tags = shared.0.lock().unwrap_or_else(|e| e.into_inner());
        tags.clone()
    }

    /// Whether the current thread has a set of tags.
    fn has_current(&self) -> bool {
        self.by_thread
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&calling_thread_id())
    }

    /// Change the tags of the current thread, and forget a thread without tags that it shares
    /// with no other.
    fn update(&self, change: impl FnOnce(&mut Vec<Tag>)) {
        let mut by_thread = self.by_thread.lock().unwrap_or_else(|e| e.into_inner());
        let thread = calling_thread_id();
        if !by_thread.contains_key(&thread) {
            forget_when_thread_ends(&self.by_thread, forget_thread_tags);
        }
        let shared = by_thread.entry(thread).or_default().clone();
        let mut tags = shared.0.lock().unwrap_or_else(|e| e.into_inner());
        change(&mut tags);
        // The set is held by the map and here only: no other thread or copy shares it.
        if tags.is_empty() && Arc::strong_count(&shared.0) == 2 {
            drop(tags);
            by_thread.remove(&thread);
        }
    }

    /// A new empty set of tags of the current thread, which shares no tags with another, as
    /// PySpark Connect's `clearTags` sets the tags of the `thread_local` to a new set.
    fn clear(&self) {
        self.by_thread
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&calling_thread_id());
    }

    /// A copy of the tags of the current thread.
    fn copy(&self) -> SharedTags {
        SharedTags(Arc::new(Mutex::new(self.current())))
    }

    /// The set of tags of the calling thread, if it has one, of its ID.
    fn of_calling_thread(&self) -> (std::thread::ThreadId, Option<SharedTags>) {
        let thread = calling_thread_id();
        let tags = self
            .by_thread
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&thread)
            .cloned();
        (thread, tags)
    }

    /// Give `thread` the set `tags`, or none, which it had (see [`ThreadTags::of_calling_thread`]):
    /// a thread that had a set is registered still, until it ends.
    fn restore(&self, thread: std::thread::ThreadId, tags: Option<SharedTags>) {
        let mut by_thread = self.by_thread.lock().unwrap_or_else(|e| e.into_inner());
        match tags {
            Some(tags) => {
                by_thread.insert(thread, tags);
            }
            None => {
                by_thread.remove(&thread);
            }
        }
    }

    /// Make `tags` those of the current thread.
    fn share(&self, tags: &SharedTags) {
        let mut by_thread = self.by_thread.lock().unwrap_or_else(|e| e.into_inner());
        let thread = calling_thread_id();
        if !by_thread.contains_key(&thread) {
            forget_when_thread_ends(&self.by_thread, forget_thread_tags);
        }
        by_thread.insert(thread, tags.clone());
    }
}

/// The context of the requests of a thread, which it has again when this drops (see
/// [`SparkSession::keep_thread_request_context`]). Not a stable API.
#[doc(hidden)]
#[must_use = "the context is given back when the guard drops"]
pub struct KeptThreadRequestContext<'a> {
    session: &'a SparkSession,
    thread: std::thread::ThreadId,
    client: Option<spark_connect_core::client::ThreadRequestState>,
    tags: Option<SharedTags>,
}

impl Drop for KeptThreadRequestContext<'_> {
    fn drop(&mut self) {
        if let Some(client) = self.client.take() {
            self.session.client().restore_thread_request_state(client);
        }
        self.session
            .inner
            .tags
            .restore(self.thread, self.tags.take());
    }
}

/// The hook of [`set_finalizer_request_hook`].
static FINALIZER_REQUEST_HOOK: std::sync::OnceLock<FinalizerRequestHook> =
    std::sync::OnceLock::new();

/// The functions that a session calls on the thread of a finalizer around its request, the
/// release of the relation of a checkpoint or of a cached ML object when its last handle is
/// dropped (see [`set_finalizer_request_hook`]). Not a stable API.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct FinalizerRequestHook {
    /// Called before the request is built: a binding brings the tags, the user context
    /// extensions, the session ID and the user of the requests of the thread up to date then, of
    /// the state of its client, as PySpark's finalizers build their requests when they run, and
    /// returns whether to send the request, which is not sent when PySpark's would fail to be
    /// built.
    pub before: fn(&SparkSession) -> bool,
    /// Called once the request is built, or is not sent, when the thread has the context of its
    /// requests of before `before` again (see [`SparkSession::keep_thread_request_context`]): a
    /// binding reads that of a call that the finalizer ran during again, of its client.
    pub after: fn(&SparkSession),
}

/// Set, once for the process, the functions that a session calls around the request of a
/// finalizer (see [`FinalizerRequestHook`]). A later call does nothing. Not a stable API.
#[doc(hidden)]
pub fn set_finalizer_request_hook(hook: FinalizerRequestHook) {
    let _ = FINALIZER_REQUEST_HOOK.set(hook);
}

/// The request of a finalizer of a session on the calling thread, while it is built: the thread
/// has the context of its requests of before it again when this drops, after which the `after`
/// function of the hook of [`set_finalizer_request_hook`] runs (see
/// [`SparkSession::finalizer_request`]).
struct FinalizerRequest<'a> {
    session: &'a SparkSession,
    kept: Option<KeptThreadRequestContext<'a>>,
}

impl Drop for FinalizerRequest<'_> {
    fn drop(&mut self) {
        drop(self.kept.take());
        if let Some(hook) = FINALIZER_REQUEST_HOOK.get() {
            (hook.after)(self.session);
        }
    }
}

/// The lock that `stop()` holds, one for all sessions of the process, as PySpark's `stop()` holds
/// `SparkSession._lock`, a class-wide `RLock`: a `stop()` of another thread waits for the one in
/// progress, whichever session it stops, and a `stop()` of the thread that holds it, as from a
/// signal handler that runs while the release waits, enters it again.
#[derive(Default)]
struct StopLock {
    /// The thread that holds the lock, and how many times it entered it.
    holder: Mutex<Option<(std::thread::ThreadId, usize)>>,
    /// Wakes the waits for the lock when it is left.
    left: tokio::sync::Notify,
}

/// The lock that `stop()` holds.
fn stop_lock() -> &'static StopLock {
    static LOCK: std::sync::OnceLock<StopLock> = std::sync::OnceLock::new();
    LOCK.get_or_init(StopLock::default)
}

impl StopLock {
    /// Enter the lock for `thread`, unless another thread holds it.
    fn try_enter(&self, thread: std::thread::ThreadId) -> bool {
        let mut holder = self.holder.lock().unwrap_or_else(|e| e.into_inner());
        match holder.as_mut() {
            None => *holder = Some((thread, 1)),
            Some((owner, depth)) if *owner == thread => *depth += 1,
            Some(_) => return false,
        }
        true
    }

    /// Enter the lock for the calling thread, waiting while another thread holds it. As the
    /// acquire of PySpark's lock, which a signal of the process interrupts, the wait ends with
    /// None when a signal ends it in the Python client (see
    /// [`block_on_abortable`](spark_connect_core::runtime::block_on_abortable)).
    fn enter(&self) -> Option<StopLockHold> {
        let thread = spark_connect_core::runtime::calling_thread_id();
        if !self.try_enter(thread) {
            spark_connect_core::runtime::block_on_abortable(async {
                loop {
                    // Made before the check, so that the lock left after it wakes the wait.
                    let left = self.left.notified();
                    if self.try_enter(thread) {
                        return;
                    }
                    left.await;
                }
            })?;
        }
        Some(StopLockHold { thread })
    }

    /// Leave the lock by one entry of `thread`, waking the waits for it. A hold exists only while
    /// its thread holds the lock, so the entry is that of the hold, on whichever thread it drops.
    fn leave(&self, thread: std::thread::ThreadId) {
        let mut holder = self.holder.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((owner, depth)) = holder.as_mut() {
            if *owner == thread {
                *depth -= 1;
                if *depth == 0 {
                    *holder = None;
                }
            }
        }
        drop(holder);
        self.left.notify_waiters();
    }
}

/// A hold of the lock that `stop()` takes, which is left when it drops. See
/// [`enter_stop_lock`].
pub struct StopLockHold {
    /// The thread that entered the lock, whose entry the hold leaves.
    thread: std::thread::ThreadId,
}

impl Drop for StopLockHold {
    fn drop(&mut self) {
        stop_lock().leave(self.thread);
    }
}

/// Enter the lock that `stop()` holds, for as long as the returned hold lives, as PySpark's
/// `getOrCreate` takes `SparkSession._lock` while it looks up or creates a session. A `stop()` in
/// progress on another thread, of any session, is waited for, so that the lookup sees a session
/// that it stops as stopped; one on the calling thread is entered within, as the lock is reentrant.
/// The wait ends with None when a signal of the process ends it in the Python client, as the
/// acquire of PySpark's lock is interrupted; without that, it waits for the stop.
pub fn enter_stop_lock() -> Option<StopLockHold> {
    stop_lock().enter()
}

/// A Spark Connect session for interacting with a remote Spark cluster.
///
/// Mirrors `pyspark.sql.SparkSession`. A clone is another handle of the same session, which
/// shares all of its state.
#[derive(Clone)]
pub struct SparkSession {
    /// The state of the session, which the session and its clones share: a clone is one more
    /// reference to it.
    inner: Arc<SessionInner>,
}

/// The state of a session, which the session and its clones share (see [`SparkSession`]), and
/// which the last of them drops, field by field, in this order. A part of it that an object holds
/// apart from the session is an `Arc` of its own: the client, the plan IDs, the registry of the
/// progress handlers and the listener bus.
struct SessionInner {
    /// Shared gRPC client
    client: Arc<SparkConnectClient>,
    /// Stable plan IDs for the plan nodes this session serializes.
    plan_ids: Arc<PlanIds>,
    /// Operation tags attached to every ExecutePlan request that a thread starts from this
    /// session, those of the thread. Mirrors `pyspark.sql.connect.session.SparkSession`'s
    /// addTag/removeTag/getTags, of the `thread_local` of its client.
    tags: ThreadTags,
    /// Metrics/observed-metrics captured from the most recent execution on this
    /// session. Backs `DataFrame::execution_info` and the profiler accessor.
    last_execution: Mutex<Option<ExecutionInfo>>,
    /// Registered progress handlers, keyed by id (for removal). Invoked with each progress of
    /// an execution, and when it ends. Mirrors `registerProgressHandler`.
    progress_handlers: ProgressHandlers,
    /// Monotonic id source for `register_progress_handler`.
    progress_handler_id: AtomicU64,
    /// The handler of the updates of the Python accumulators of its executions.
    python_accumulators: PythonAccumulators,
    /// The check of the metrics of the Observations of its executions (shared across clones).
    observation_metrics_check: Mutex<Option<ObservationMetricsCheck>>,
    /// The handler of the warnings of its ML fits (shared across clones).
    ml_fit_warning_handler: Mutex<Option<MlFitWarningHandler>>,
    /// Whether this session was stopped, by `stop()` once its release ended, or by `close()`
    /// (shared across clones). Backs `is_stopped`.
    stopped: AtomicBool,
    /// Successful server version, shared by handles for this session only.
    server_version: Mutex<Option<String>>,
    /// The client-side listener bus of the streaming queries of this session (shared across
    /// clones), which each of its `StreamingQueryManager`s adds to, removes from and posts to,
    /// as the one `StreamingQueryListenerBus` of PySpark Connect's `spark.streams`.
    listener_bus: Arc<crate::streaming::ListenerBus>,
}

impl SparkSession {
    /// Create a new SparkSession with a client, of none of the state of another session.
    pub(crate) fn new(client: SparkConnectClient) -> Self {
        SparkSession {
            inner: Arc::new(SessionInner {
                client: Arc::new(client),
                plan_ids: Arc::new(PlanIds::new()),
                tags: ThreadTags::default(),
                last_execution: Mutex::new(None),
                progress_handlers: Arc::new(Mutex::new(Vec::new())),
                progress_handler_id: AtomicU64::new(0),
                python_accumulators: PythonAccumulators::default(),
                observation_metrics_check: Mutex::new(None),
                ml_fit_warning_handler: Mutex::new(None),
                stopped: AtomicBool::new(false),
                server_version: Mutex::new(None),
                listener_bus: Arc::default(),
            }),
        }
    }

    /// Add a tag to be attached to all subsequent operations that this thread starts from
    /// this session.
    ///
    /// Mirrors `SparkSession.addTag`, whose tags are those of the current thread. Tags cannot
    /// be empty or contain a comma (see [`SparkSession::check_tag`]).
    pub fn add_tag(&self, tag: &str) -> Result<()> {
        Self::check_tag(tag)?;
        self.add(Tag::Text(tag.to_string()));
        Ok(())
    }

    /// Add a tag of a str of Python that UTF-8 cannot encode, given as the UTF-32 little-endian
    /// bytes of its code points, as PySpark's client adds it to the tags of the thread: each
    /// execution that the thread starts then fails with the `UnicodeEncodeError` of the tag,
    /// when its request is built, after its plan, until the tag is removed. The caller checks the
    /// tag as [`SparkSession::check_tag`] checks one of text. Not a stable API.
    #[doc(hidden)]
    pub fn add_unencodable_tag(&self, utf32le: Vec<u8>) {
        self.add(Tag::Unencodable(utf32le));
    }

    fn add(&self, tag: Tag) {
        self.inner.tags.update(|tags| {
            if !tags.contains(&tag) {
                tags.push(tag);
            }
        });
    }

    /// The errors of a tag, as PySpark Connect's `_throw_if_invalid_tag`: `VALUE_ALLOWED` for a
    /// tag of a comma, which separates the tags of Spark jobs, and `VALUE_NOT_NON_EMPTY_STR` for
    /// an empty tag. `add_tag` checks them, as does the `removeTag` of the Python client.
    pub fn check_tag(tag: &str) -> Result<()> {
        if tag.contains(',') {
            return Err(SparkError::value(
                "VALUE_ALLOWED",
                &[("arg_name", "Spark Connect tag"), ("disallowed_value", ",")],
            ));
        }
        if tag.is_empty() {
            return Err(SparkError::value(
                "VALUE_NOT_NON_EMPTY_STR",
                &[("arg_name", "Spark Connect tag"), ("arg_value", "")],
            ));
        }
        Ok(())
    }

    /// Remove a tag that this thread added. Mirrors `SparkSession.removeTag`.
    pub fn remove_tag(&self, tag: &str) {
        self.inner
            .tags
            .update(|tags| tags.retain(|t| !matches!(t, Tag::Text(text) if text == tag)));
    }

    /// Remove a tag of [`SparkSession::add_unencodable_tag`] of these bytes. Not a stable API.
    #[doc(hidden)]
    pub fn remove_unencodable_tag(&self, utf32le: &[u8]) {
        self.inner.tags.update(|tags| {
            tags.retain(|t| !matches!(t, Tag::Unencodable(bytes) if bytes == utf32le))
        });
    }

    /// Get the tags that this thread set on this session. Mirrors `SparkSession.getTags`.
    pub fn get_tags(&self) -> Vec<String> {
        self.inner
            .tags
            .current()
            .into_iter()
            .filter_map(|tag| match tag {
                Tag::Text(text) => Some(text),
                Tag::Unencodable(_) | Tag::NotText(_) | Tag::NotUtf8(_) => None,
            })
            .collect()
    }

    /// The tags of [`SparkSession::add_unencodable_tag`] of this thread, as their bytes, which
    /// [`SparkSession::get_tags`] does not give. Not a stable API.
    #[doc(hidden)]
    pub fn unencodable_tags(&self) -> Vec<Vec<u8>> {
        self.inner
            .tags
            .current()
            .into_iter()
            .filter_map(|tag| match tag {
                Tag::Unencodable(bytes) => Some(bytes),
                Tag::Text(_) | Tag::NotText(_) | Tag::NotUtf8(_) => None,
            })
            .collect()
    }

    /// Clear the tags that this thread set on this session. Mirrors `SparkSession.clearTags`:
    /// the thread gets a new empty set of tags, which it shares with no other thread.
    pub fn clear_tags(&self) {
        self.inner.tags.clear();
    }

    /// A copy of the tags of the current thread, as of now, which the threads that are given it
    /// with [`SparkSession::share_thread_tags`] share. Mirrors how PySpark's
    /// `inheritable_thread_target(session)` copies the tags of the thread that decorates a
    /// function, which every thread that runs the function then has.
    pub fn copy_thread_tags(&self) -> SharedTags {
        self.inner.tags.copy()
    }

    /// Make `tags` the tags of the current thread, which it shares with the other threads given
    /// them: a tag that one of them adds or removes is added or removed for all, until a thread
    /// clears its tags.
    pub fn share_thread_tags(&self, tags: &SharedTags) {
        self.inner.tags.share(tags);
    }

    /// Register a handler of the progress of the executions of this session: it is called with
    /// each progress of the server that has tasks, and once more, as done, when the execution
    /// ends, also of an error, as PySpark's `Progress` calls its handlers. Returns an id usable
    /// with [`SparkSession::remove_progress_handler`]. Mirrors `registerProgressHandler`.
    ///
    /// As PySpark's, the handlers are called on the thread that runs the execution (that calls
    /// `collect`, `count`, or `next` of a local iterator), also inside a `current_thread` Tokio
    /// runtime. The prefetching iterator of `to_local_iterator(true)` reads ahead on a thread of
    /// the runtime, which calls them.
    pub fn register_progress_handler(
        &self,
        handler: impl Fn(&ProgressUpdate) + Send + Sync + 'static,
    ) -> u64 {
        self.register_fallible_progress_handler(move |update| {
            handler(update);
            Ok(())
        })
    }

    /// Register a progress handler that may fail, otherwise as
    /// [`SparkSession::register_progress_handler`]. An error it returns fails the execution
    /// that calls it: the execution ends, the error goes to the caller of that execution, and
    /// the handlers after it are not called for that update, as PySpark's client does when a
    /// handler raises. The handler is still called as done when the execution ends, and an
    /// error of that done call is the error of the execution, also replacing the execution's own
    /// error, as the exception of PySpark's `Progress.__exit__` is. For an execution dropped
    /// before its end, as an iterator that its caller stopped reading, the error of the done call
    /// has no caller left: it goes to the handler of
    /// [`set_unreturned_error_handler`](crate::progress::set_unreturned_error_handler). Used by
    /// the Python drop-in to raise a handler's exception from the call that ran it.
    pub fn register_fallible_progress_handler(
        &self,
        handler: impl Fn(&ProgressUpdate) -> Result<()> + Send + Sync + 'static,
    ) -> u64 {
        let id = self
            .inner
            .progress_handler_id
            .fetch_add(1, Ordering::SeqCst);
        self.inner
            .progress_handlers
            .lock()
            .unwrap()
            .push((id, Arc::new(handler)));
        id
    }

    /// Remove a progress handler by the id returned from
    /// [`SparkSession::register_progress_handler`]. Mirrors `removeProgressHandler`.
    pub fn remove_progress_handler(&self, id: u64) {
        self.inner
            .progress_handlers
            .lock()
            .unwrap()
            .retain(|(hid, _)| *hid != id);
    }

    /// Remove all progress handlers. Mirrors `clearProgressHandlers`.
    pub fn clear_progress_handlers(&self) {
        self.inner.progress_handlers.lock().unwrap().clear();
    }

    /// Share the progress-handler registry without retaining the session or its listener bus.
    pub(crate) fn progress_handlers(&self) -> ProgressHandlers {
        Arc::clone(&self.inner.progress_handlers)
    }

    /// Execute `request` on the server, with the progress of the execution for the progress
    /// handlers. Its operation ID is that of the request, or a new one, which the client
    /// assigns before it sends the request, as PySpark's `Progress` of the request has it.
    /// The logs of the execution are those of the method of PySpark's client that it stands for,
    /// as `log` says (see [`ExecutionLog`]).
    pub(crate) fn execute_plan(
        &self,
        request: proto::ExecutePlanRequest,
        log: ExecutionLog,
    ) -> Result<ExecutionStream> {
        self.execute_plan_with_handlers(request, self.progress_handlers(), true, log)
    }

    /// As [`execute_plan`](Self::execute_plan), for a request whose plan is sent as it is,
    /// neither compressed nor of the lookup of the plan compression of the session: a plan whose
    /// compression was tried as PySpark's `to_proto` tries it (see
    /// `SparkConnectClient::prepare_plan`), or a plan that PySpark's client sends as it is given it.
    pub(crate) fn execute_plan_as_is(
        &self,
        request: proto::ExecutePlanRequest,
        log: ExecutionLog,
    ) -> Result<ExecutionStream> {
        self.execute_plan_with_handlers(request, self.progress_handlers(), false, log)
    }

    /// As [`execute_plan`](Self::execute_plan), but the progress of the execution, and its done
    /// notification, does not reach the progress handlers: PySpark's listener bus has no
    /// `Progress`, so its long-running stream reports none.
    pub(crate) fn execute_plan_unreported(
        &self,
        request: proto::ExecutePlanRequest,
        log: ExecutionLog,
    ) -> Result<ExecutionStream> {
        self.execute_plan_with_handlers(request, ProgressHandlers::default(), true, log)
    }

    /// As [`execute_plan_unreported`](Self::execute_plan_unreported), of a plan sent as it is (see
    /// [`execute_plan_as_is`](Self::execute_plan_as_is)).
    pub(crate) fn execute_plan_unreported_as_is(
        &self,
        request: proto::ExecutePlanRequest,
        log: ExecutionLog,
    ) -> Result<ExecutionStream> {
        self.execute_plan_with_handlers(request, ProgressHandlers::default(), false, log)
    }

    /// As [`execute_plan_unreported`](Self::execute_plan_unreported), for a registration of
    /// PySpark's `_execute`, of a function, a table function, a data source or a Java function,
    /// whose command PySpark sets in the plan of its request as it is: neither compressed nor of
    /// the lookup of the plan compression of the session, and of the logs of `_execute`.
    pub(crate) fn execute_registration(
        &self,
        request: proto::ExecutePlanRequest,
    ) -> Result<ExecutionStream> {
        self.execute_plan_with_handlers(
            request,
            ProgressHandlers::default(),
            false,
            ExecutionLog::Execute,
        )
    }

    /// The execution of `request`, whose progress reaches `handlers`, whose plan is compressed
    /// when `compressed`, and of the logs of `log`.
    fn execute_plan_with_handlers(
        &self,
        mut request: proto::ExecutePlanRequest,
        handlers: ProgressHandlers,
        compressed: bool,
        log: ExecutionLog,
    ) -> Result<ExecutionStream> {
        let operation_id = request
            .operation_id
            .get_or_insert_with(|| uuid::Uuid::new_v4().to_string())
            .clone();
        // Before the request: an execution that fails to start ends, as done, too.
        check_callback_depth()?;
        // As PySpark's client, which builds the request, and logs it, before the `Progress` of
        // the execution starts: an error then, as of a record, ends no progress. One wait, so that
        // a signal that ends the lookup of the plan compression of the first plan ends the rest.
        let client = self.client();
        let started = block_on(async {
            match client.build_execution(request, compressed, log).await {
                Ok(built) => Ok(client.execute_built(built).await),
                Err(error) => Err(error),
            }
        })?;
        let mut progress = ProgressTracker::with_handlers(handlers, Some(operation_id));
        // An execution that fails to start ends too: the error of the done call is its error, as
        // the `Progress` of PySpark wraps its request.
        let stream = match started {
            Ok(stream) => stream,
            Err(error) => return Err(progress.finish_after(error)),
        };
        Ok(ExecutionStream::new(stream, progress))
    }

    /// Store the metrics captured from the most recent execution. Called by the
    /// execute loop; not part of the public API.
    pub(crate) fn record_execution(&self, info: ExecutionInfo) {
        *self.inner.last_execution.lock().unwrap() = Some(info);
    }

    /// The metrics captured from the most recent execution on this session, if any.
    /// Backs `DataFrame::execution_info`.
    pub fn last_execution_info(&self) -> Option<ExecutionInfo> {
        self.inner.last_execution.lock().unwrap().clone()
    }

    /// Raw observed metrics from the most recent execution only.
    ///
    /// This is a low-level snapshot of the last execution's observed metrics. For
    /// the UDF profiles of `SparkSession.profile`, whose results the server sends as the updates
    /// of Python accumulators, see [`SparkSession::set_python_accumulator_handler`].
    pub fn profile(&self) -> Vec<proto::execute_plan_response::ObservedMetrics> {
        self.last_execution_info()
            .map(|i| i.observed_metrics)
            .unwrap_or_default()
    }

    /// Internal accessor used by the execute-request builders to attach the tags of the
    /// current thread, which builds the request: a tag that UTF-8 cannot encode fails the
    /// request with its `UnicodeEncodeError`, as protobuf's string field does in PySpark, and so
    /// does one that the field does not take (see [`Tag`]). A thread of no tags has an empty set
    /// of them from then on (see [`SparkSession::has_thread_tags`]), as PySpark's request reads
    /// them with `get_tags`, which gives its `thread_local` a set.
    pub(crate) fn tags(&self) -> Result<Vec<String>> {
        self.inner
            .tags
            .current_or_new()
            .into_iter()
            .map(|tag| match tag {
                Tag::Text(text) => Ok(text),
                Tag::Unencodable(bytes) => Err(SparkError::unencodable_str(&bytes)),
                Tag::NotText(message) => Err(SparkError {
                    kind: spark_connect_core::error::SparkErrorKind::BuiltinTypeError,
                    ..SparkError::value_msg(message)
                }),
                Tag::NotUtf8(bytes) => Err(SparkError::invalid_utf8(0, &bytes)),
            })
            .collect()
    }

    /// Whether the current thread has a set of tags: tags that it added or was given, or the
    /// empty set that an execution of a thread of none gives it, as PySpark's `get_tags` sets
    /// one in the `thread_local` of its client. Not a stable API.
    #[doc(hidden)]
    pub fn has_thread_tags(&self) -> bool {
        self.inner.tags.has_current()
    }

    /// Start a brand-new session of the same endpoint, over the connection of the process to it,
    /// as PySpark's client of a new session makes a channel of the subchannels of the process.
    /// Mirrors `SparkSession.newSession` — a fresh server-side session (new session id), with its
    /// own tags and plan IDs.
    pub fn new_session(&self) -> SparkSession {
        SparkSession::new(self.client().with_new_session_id())
    }

    /// As [`SparkSession::new_session`], of the user `user_id` as it is given, whom its requests
    /// and those of its artifacts carry, also when the connection has a `user_id`, which
    /// `new_session` takes in its place, as PySpark's `newSession()` does (see
    /// `SparkConnectClient::with_new_session_id_of_user`). Not a stable API.
    #[doc(hidden)]
    pub fn new_session_of_user(&self, user_id: Option<String>) -> SparkSession {
        SparkSession::new(self.client().with_new_session_id_of_user(user_id))
    }

    /// Clone this session on the server, mirroring `SparkSession.cloneSession()`: the new
    /// session has a copy of the server-side state of this one, its SQL configurations,
    /// temporary views, registered functions and catalog state, and changes to either do not
    /// show in the other.
    pub fn clone_session(&self) -> Result<SparkSession> {
        self.clone_session_with_id(None)
    }

    /// Clone this session on the server as [`SparkSession::clone_session`], into a session of
    /// the id `new_session_id`, a UUID, if one is given, mirroring
    /// `SparkSession.cloneSession(new_session_id)`.
    pub fn clone_session_with_id(&self, new_session_id: Option<&str>) -> Result<SparkSession> {
        // Of the one client that the clone makes, as PySpark's: a channel of the proxy of the
        // environment, which logs what grpcio logs of it once.
        let client = block_on(self.client().clone_session(new_session_id))?;
        Ok(SparkSession::new(client))
    }

    /// As [`SparkSession::clone_session_with_id`], of the user `user_id` as it is given, whom the
    /// requests of the clone and those of its artifacts carry, also when the connection has a
    /// `user_id`, which `clone_session_with_id` takes in its place, as PySpark's `cloneSession()`
    /// does (see `SparkConnectClient::clone_session_of_user`). Not a stable API.
    #[doc(hidden)]
    pub fn clone_session_with_id_of_user(
        &self,
        new_session_id: Option<&str>,
        user_id: Option<String>,
    ) -> Result<SparkSession> {
        let client = block_on(self.client().clone_session_of_user(new_session_id, user_id))?;
        Ok(SparkSession::new(client))
    }

    /// Set the hook of the executions of this session, and of its clones, or clear it: the
    /// hook is called with each `ExecutePlanRequest` before it is sent, and the request sent is
    /// the one it returns, of the operation ID of the original. Mirrors the hooks of
    /// `SparkSession.Hook.on_execute_plan`, of `SparkSession.Builder._registerHook`.
    pub fn set_execute_plan_hook(&self, hook: Option<ExecutePlanHook>) {
        self.client().set_execute_plan_hook(hook);
    }

    /// The stable plan IDs of the plan nodes this session serializes.
    pub(crate) fn plan_ids(&self) -> &PlanIds {
        &self.inner.plan_ids
    }

    pub(crate) fn plan_ids_shared(&self) -> &Arc<PlanIds> {
        &self.inner.plan_ids
    }

    /// Get the underlying client.
    pub(crate) fn client(&self) -> &Arc<SparkConnectClient> {
        &self.inner.client
    }

    /// Whether `other` is this session or a clone of it, which shares its client. Two sessions
    /// made apart are not the same session, even of the same session ID.
    pub fn is_same_session(&self, other: &SparkSession) -> bool {
        Arc::ptr_eq(self.client(), other.client())
    }

    /// A handle of this session that does not keep it alive, as `Arc::downgrade` makes a `Weak`
    /// of an `Arc`, for the exit hook of the Python binding. See [`WeakSparkSession`]. Not a
    /// stable API.
    #[doc(hidden)]
    pub fn downgrade(&self) -> WeakSparkSession {
        WeakSparkSession {
            inner: Arc::downgrade(&self.inner),
        }
    }

    /// Create a builder for a new SparkSession.
    pub fn builder() -> SparkSessionBuilder {
        SparkSessionBuilder::new()
    }

    /// Create a DataFrame representing a range of integers.
    ///
    /// Mirrors `pyspark.sql.SparkSession.range(start, end=None, step=1, numPartitions=None)`.
    pub fn range(&self, end: i64) -> Result<DataFrame> {
        self.range_full(0, end, 1, None)
    }

    /// Create a DataFrame representing a range with full parameters.
    pub fn range_full(
        &self,
        start: i64,
        end: i64,
        step: i64,
        num_partitions: Option<i32>,
    ) -> Result<DataFrame> {
        let plan = LogicalPlan::Range {
            start,
            end,
            step,
            num_partitions,
        };
        Ok(DataFrame::new(self.clone(), plan))
    }

    /// Parse a DDL string on the server, mirroring PySpark Connect's `_parse_ddl`: the
    /// `DDLParse` of an `AnalyzePlanRequest`, of a schema or a type, as Spark parses it.
    pub fn parse_ddl(&self, ddl: &str) -> Result<DataType> {
        let request = proto::AnalyzePlanRequest {
            session_id: self.client().session_id().to_string(),
            user_context: Some(proto::UserContext::default()),
            analyze: Some(proto::analyze_plan_request::Analyze::DdlParse(
                proto::analyze_plan_request::DdlParse {
                    ddl_string: ddl.to_string(),
                },
            )),
            ..Default::default()
        };
        let response = block_on(self.client().analyze_plan(request))?;
        match response.result {
            Some(proto::analyze_plan_response::Result::DdlParse(r)) => {
                DataType::from_proto(&r.parsed.unwrap_or_default())
            }
            _ => Err(SparkError::connect_msg(
                "AnalyzePlan response did not contain a DDL parse result",
            )),
        }
    }

    /// Convert the JSON of a schema to DDL on the server, mirroring PySpark Connect's
    /// `_to_ddl`: the `JsonToDDL` of an `AnalyzePlanRequest`, as Spark writes it.
    pub fn json_to_ddl(&self, json: &str) -> Result<String> {
        let request = proto::AnalyzePlanRequest {
            session_id: self.client().session_id().to_string(),
            user_context: Some(proto::UserContext::default()),
            analyze: Some(proto::analyze_plan_request::Analyze::JsonToDdl(
                proto::analyze_plan_request::JsonToDdl {
                    json_string: json.to_string(),
                },
            )),
            ..Default::default()
        };
        let response = block_on(self.client().analyze_plan(request))?;
        match response.result {
            Some(proto::analyze_plan_response::Result::JsonToDdl(r)) => Ok(r.ddl_string),
            _ => Err(SparkError::connect_msg(
                "AnalyzePlan response did not contain a DDL string",
            )),
        }
    }

    /// Execute a SQL statement and return a DataFrame.
    ///
    /// Mirrors `pyspark.sql.SparkSession.sql(sqlQuery)`: the statement is sent at once, so a
    /// command such as `CREATE VIEW` or `SET` takes effect here and a statement that does not
    /// parse or resolve fails here, while a query still runs only when an action is taken on
    /// the DataFrame.
    pub fn sql(&self, query: &str) -> Result<DataFrame> {
        self.sql_with_args(query, Vec::new(), std::collections::HashMap::new())
    }

    /// SQL with parameters: positional (`pos_args`) and/or named (`named_args`) expression
    /// bindings, mirroring `SparkSession.sql(query, args=...)`.
    pub fn sql_with_args(
        &self,
        query: &str,
        pos_args: Vec<crate::expression::Expression>,
        named_args: std::collections::HashMap<String, crate::expression::Expression>,
    ) -> Result<DataFrame> {
        self.sql_with_references(query, pos_args, named_args, Vec::new())
    }

    /// SQL with parameters and with DataFrames that the query names, mirroring
    /// `SparkSession.sql(query, args, **kwargs)` with DataFrames in `kwargs`: each
    /// `(DataFrame, name)` is sent along with the query, as a reference that the server
    /// resolves `name` in the query to.
    ///
    /// The server creates each name as a temporary view while it runs the statement, and
    /// drops it after, also at each later action on a query's DataFrame. A name must not be
    /// that of a temporary view to keep, which the statement would replace and drop;
    /// PySpark's `sql()` names its DataFrames `_pyspark_connect_temp_view_<uuid>` for this.
    pub fn sql_with_references(
        &self,
        query: &str,
        pos_args: Vec<crate::expression::Expression>,
        named_args: std::collections::HashMap<String, crate::expression::Expression>,
        references: Vec<(DataFrame, String)>,
    ) -> Result<DataFrame> {
        let plan = Self::sql_plan(query, pos_args, named_args, references);
        // As pyspark, run the statement as a command now, without the Observations of its
        // DataFrames, which the DataFrame of its result does not have either. The server executes
        // a command and returns a relation of its result; for a query it returns the query's
        // relation, which then runs at an action.
        let command = proto::SqlCommand {
            input: Some(crate::dataframe::build_input_relation(&plan, self)?),
            ..Default::default()
        };
        let responses = crate::dataframe::execute_command_collect(
            self,
            proto::command::CommandType::SqlCommand(command),
        )?;
        let returned = responses
            .into_iter()
            .find_map(|response| match response.response_type {
                Some(proto::execute_plan_response::ResponseType::SqlCommandResult(result)) => {
                    result.relation
                }
                _ => None,
            });
        Ok(DataFrame::new(
            self.clone(),
            match returned {
                Some(relation) => LogicalPlan::CachedRelation {
                    relation: Box::new(relation),
                },
                // A server that returns no result relation runs the query lazily, as pyspark
                // then does.
                None => Arc::unwrap_or_clone(plan),
            },
        ))
    }

    /// The relation of a SQL statement with its parameters and references.
    fn sql_plan(
        query: &str,
        pos_args: Vec<crate::expression::Expression>,
        named_args: std::collections::HashMap<String, crate::expression::Expression>,
        references: Vec<(DataFrame, String)>,
    ) -> Arc<LogicalPlan> {
        let sql = Arc::new(LogicalPlan::Sql {
            query: query.to_string(),
            pos_args,
            named_args,
        });
        if references.is_empty() {
            return sql;
        }
        Arc::new(LogicalPlan::WithRelations {
            root: sql.into(),
            references: references
                .into_iter()
                .map(|(df, alias)| {
                    SharedPlan::new(LogicalPlan::SubqueryAlias {
                        input: df.shared_plan(),
                        alias,
                    })
                })
                .collect(),
        })
    }

    /// Internal: the DataFrame of a SQL statement that is not run until an action, for a
    /// statement within a pipeline flow function, whose relation the server analyzes in the
    /// pipeline instead (PySpark's server returns the relation without running it there).
    /// Not a stable API.
    #[doc(hidden)]
    pub fn sql_without_running(
        &self,
        query: &str,
        pos_args: Vec<crate::expression::Expression>,
        named_args: std::collections::HashMap<String, crate::expression::Expression>,
        references: Vec<(DataFrame, String)>,
    ) -> DataFrame {
        let plan = Self::sql_plan(query, pos_args, named_args, references);
        DataFrame::new(self.clone(), Arc::unwrap_or_clone(plan))
    }

    /// Create a DataFrame from a collection of Rows and a schema.
    ///
    /// Mirrors `pyspark.sql.SparkSession.createDataFrame(rows, schema)`. As PySpark's, rows that
    /// are not empty read the settings of `createDataFrame` of the server in one request, and a
    /// relation of `spark.sql.session.localRelationCacheThreshold` bytes or more, of any size, is
    /// uploaded as cached artifacts; a smaller one is inline. The rows are converted before that
    /// request, so that rows that do not fit the schema fail without one. No rows send the
    /// schema alone, with no request.
    pub fn create_dataframe(&self, rows: Vec<Row>, schema: DataType) -> Result<DataFrame> {
        self.create_dataframe_of(rows, schema, None)
    }

    /// Internal binding bridge: [`Self::create_dataframe`] with the settings that the caller
    /// read already, as PySpark's `createDataFrame` reads them once before it converts any
    /// data, so that no other request reads them; not a stable API.
    #[doc(hidden)]
    pub fn create_dataframe_with_options(
        &self,
        rows: Vec<Row>,
        schema: DataType,
        options: &crate::ArrowInputOptions,
    ) -> Result<DataFrame> {
        self.create_dataframe_of(rows, schema, Some(options))
    }

    fn create_dataframe_of(
        &self,
        rows: Vec<Row>,
        schema: DataType,
        options: Option<&crate::ArrowInputOptions>,
    ) -> Result<DataFrame> {
        // Empty data -> send the schema only (no Arrow `data`); the server builds an empty
        // relation from the schema. A zero-row Arrow payload would be empty bytes, which the
        // server cannot parse ("Unexpected end of input. Missing schema.").
        if rows.is_empty() {
            let plan = LogicalPlan::LocalRelation {
                schema: Some(options.map_or(schema.clone(), |o| o.sent_schema(schema))),
                data: None,
            };
            return Ok(DataFrame::new(self.clone(), plan));
        }
        let batch = rows_to_arrow_batch(&rows, &schema)?;
        Ok(DataFrame::new(
            self.clone(),
            self.local_relation(batch, schema, options)?,
        ))
    }

    /// The local relation of `batch`, as PySpark's `createDataFrame` sends one: of the server's
    /// local-relation settings, which are read unless the caller read them already, a relation
    /// of the cache threshold or more is uploaded as cached artifacts at any size, which fails
    /// past the size limit; a smaller one is inline. Inline below [`INLINE_IPC_BYTES`], it is
    /// the IPC stream of the batch; from it, as the Arrow importer writes one.
    fn local_relation(
        &self,
        batch: arrow::record_batch::RecordBatch,
        schema: DataType,
        options: Option<&crate::ArrowInputOptions>,
    ) -> Result<LogicalPlan> {
        let fetched;
        let options = match options {
            Some(options) => options,
            None => {
                fetched = crate::ArrowInputOptions::from_client(self.client())?;
                &fetched
            }
        };
        // As PySpark's table, of its large types under `useLargeVarTypes`, whose `nbytes` it
        // compares with the cache threshold.
        let large = options.use_large_var_types() == Some("true");
        let batch = with_large_var_types(batch, &schema, large)?;
        let schema = options.sent_schema(schema);
        let bytes = crate::arrow_input::batch_data_size(&batch);
        if bytes >= INLINE_IPC_BYTES || bytes >= options.cache_threshold() {
            return crate::arrow_input::create_relation(
                self.client(),
                vec![batch],
                Some(schema),
                Some(options),
            );
        }
        Ok(LogicalPlan::LocalRelation {
            schema: Some(schema),
            data: Some(batch_to_arrow_ipc(&batch)?),
        })
    }

    /// Create a DataFrame from columns, with the same encoding as local rows, and sent as
    /// [`Self::create_dataframe`] sends rows: of the server's local-relation settings, read once,
    /// inline, or uploaded at the cache threshold. Columns of no rows send the schema alone, with
    /// no request, as no rows do. Callers must supply one column per struct field.
    pub fn create_dataframe_from_columns(
        &self,
        columns: Vec<arrow::array::ArrayRef>,
        schema: DataType,
    ) -> Result<DataFrame> {
        self.create_dataframe_from_columns_of(columns, schema, None)
    }

    /// Internal binding bridge: [`Self::create_dataframe_from_columns`] with the settings that
    /// the caller read already (see [`Self::create_dataframe_with_options`]); not a stable API.
    #[doc(hidden)]
    pub fn create_dataframe_from_columns_with_options(
        &self,
        columns: Vec<arrow::array::ArrayRef>,
        schema: DataType,
        options: &crate::ArrowInputOptions,
    ) -> Result<DataFrame> {
        self.create_dataframe_from_columns_of(columns, schema, Some(options))
    }

    fn create_dataframe_from_columns_of(
        &self,
        columns: Vec<arrow::array::ArrayRef>,
        schema: DataType,
        options: Option<&crate::ArrowInputOptions>,
    ) -> Result<DataFrame> {
        let arrow_schema = arrow::datatypes::Schema::new(schema_to_arrow_fields(&schema)?);
        let row_count = columns.first().map_or(0, |column| column.len());
        let columns = with_udt_field_metadata(columns, &arrow_schema)?;
        let batch = columns_to_arrow_batch(columns, arrow_schema, row_count)?;
        // Columns of no rows are the schema alone, as `create_dataframe` sends no rows, and
        // PySpark its empty data (`LocalRelation(table=None, schema)`), before it reads any
        // setting: so no request reads them.
        let plan = if batch.num_rows() == 0 {
            LogicalPlan::LocalRelation {
                schema: Some(options.map_or(schema.clone(), |o| o.sent_schema(schema))),
                data: None,
            }
        } else {
            self.local_relation(batch, schema, options)?
        };
        Ok(DataFrame::new(self.clone(), plan))
    }

    /// Internal binding bridge for Arrow IPC already validated and encoded from
    /// local rows or columns. Retains the row path's transport policy: of the server's
    /// local-relation settings, read once, inline, or uploaded at the cache threshold.
    /// IPC of [`INLINE_IPC_BYTES`] or more, or uploaded, is decoded and its map field names
    /// normalized before re-encoding; smaller inline IPC is passed through unchanged.
    #[doc(hidden)]
    pub fn create_dataframe_from_inline_arrow_ipc(
        &self,
        data: Vec<u8>,
        schema: DataType,
    ) -> Result<DataFrame> {
        self.create_dataframe_from_inline_arrow_ipc_of(data, schema, None)
    }

    /// Internal binding bridge: [`Self::create_dataframe_from_inline_arrow_ipc`] with the
    /// settings that the caller read already (see [`Self::create_dataframe_with_options`]); not
    /// a stable API.
    #[doc(hidden)]
    pub fn create_dataframe_from_inline_arrow_ipc_with_options(
        &self,
        data: Vec<u8>,
        schema: DataType,
        options: &crate::ArrowInputOptions,
    ) -> Result<DataFrame> {
        self.create_dataframe_from_inline_arrow_ipc_of(data, schema, Some(options))
    }

    fn create_dataframe_from_inline_arrow_ipc_of(
        &self,
        data: Vec<u8>,
        schema: DataType,
        options: Option<&crate::ArrowInputOptions>,
    ) -> Result<DataFrame> {
        let fetched;
        let options = match options {
            Some(options) => options,
            None => {
                fetched = crate::ArrowInputOptions::from_client(self.client())?;
                &fetched
            }
        };
        // Under `useLargeVarTypes`, the strings and binaries of the IPC are cast to the large types
        // of PySpark's table, which then decides between inline and cached by its size.
        let large = options.use_large_var_types() == Some("true") && has_var_types(&schema);
        let inline =
            !large && data.len() < INLINE_IPC_BYTES && data.len() < options.cache_threshold();
        let plan = if !inline {
            let reader =
                arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(data), None)
                    .map_err(|e| {
                        SparkError::connect_msg(format!("Failed to read Arrow IPC: {e}"))
                    })?;
            let batches = reader
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| SparkError::connect_msg(format!("Failed to read Arrow IPC: {e}")))?
                .into_iter()
                .map(|batch| with_large_var_types(batch, &schema, large))
                .collect::<Result<Vec<_>>>()?;
            let schema = options.sent_schema(schema);
            crate::arrow_input::create_relation(
                self.client(),
                batches,
                Some(schema),
                Some(options),
            )?
        } else {
            LogicalPlan::LocalRelation {
                schema: Some(options.sent_schema(schema)),
                data: Some(data),
            }
        };
        Ok(DataFrame::new(self.clone(), plan))
    }

    /// Create a local DataFrame from Arrow batches without materializing rows.
    ///
    /// `schema` optionally supplies the Spark schema applied by the server. With
    /// `None`, Spark infers it from Arrow. All batches must share the same Arrow
    /// schema; IPC serialization is still required for transport.
    /// Map child fields are normalized to Spark's `entries`, `key`, and `value`
    /// names, including nested maps, without copying their data buffers.
    /// Columns rebuilt for map names are validated against the schema before any
    /// RPC, rejecting incompatible child types or union ordering.
    /// Non-empty inputs read the server's local-relation configuration. Large inputs
    /// are uploaded as cache artifacts and referenced by hash in the resulting plan.
    /// Batches of no rows are sent with no request: of a schema, as the schema alone, as
    /// PySpark's `createDataFrame` sends empty data, and without one, as their Arrow schema.
    pub fn create_dataframe_from_arrow(
        &self,
        batches: Vec<arrow::record_batch::RecordBatch>,
        schema: impl Into<Option<DataType>>,
    ) -> Result<DataFrame> {
        let plan = match schema.into() {
            // As PySpark's createDataFrame of a table of no rows, `len(data) == 0`.
            Some(schema) if batches.iter().all(|batch| batch.num_rows() == 0) => {
                crate::arrow_input::empty_relation(&batches, schema)?
            }
            schema => crate::arrow_input::create_relation(self.client(), batches, schema, None)?,
        };
        Ok(DataFrame::new(self.clone(), plan))
    }

    /// Internal settings handoff for the Python Arrow importer; not a stable API.
    #[doc(hidden)]
    pub fn arrow_input_options(&self) -> Result<crate::ArrowInputOptions> {
        crate::ArrowInputOptions::from_client(self.client())
    }

    /// Internal Arrow import entry point using the normalization settings snapshot. Its caller
    /// checks for empty data first, as PySpark's `len(data) == 0` comes before the conversion of
    /// the table, so batches of no rows here are sent as they are: those of a table of no columns,
    /// whose rows the conversion drops, which PySpark sends as its IPC stream.
    #[doc(hidden)]
    pub fn create_dataframe_from_arrow_with_options(
        &self,
        batches: Vec<arrow::record_batch::RecordBatch>,
        schema: impl Into<Option<DataType>>,
        options: &crate::ArrowInputOptions,
    ) -> Result<DataFrame> {
        let schema = schema.into().map(|schema| options.sent_schema(schema));
        let plan =
            crate::arrow_input::create_relation(self.client(), batches, schema, Some(options))?;
        Ok(DataFrame::new(self.clone(), plan))
    }

    /// Create a DataFrameReader for reading data from various sources.
    ///
    /// Mirrors `pyspark.sql.SparkSession.read`.
    pub fn read(&self) -> crate::readwriter::DataFrameReader {
        crate::readwriter::DataFrameReader::new(self.clone())
    }

    /// Create a DataStreamReader for reading streaming data from various sources.
    ///
    /// Mirrors `pyspark.sql.SparkSession.readStream`.
    pub fn read_stream(&self) -> crate::streaming::DataStreamReader {
        crate::streaming::DataStreamReader::new(self.clone())
    }

    /// Get the streaming query manager.
    ///
    /// Mirrors `pyspark.sql.SparkSession.streams`.
    pub fn streams(&self) -> crate::streaming::StreamingQueryManager {
        crate::streaming::StreamingQueryManager::new(self.clone())
    }

    /// The client-side listener bus of the streaming queries of this session.
    pub(crate) fn listener_bus(&self) -> Arc<crate::streaming::ListenerBus> {
        Arc::clone(&self.inner.listener_bus)
    }

    /// Get the catalog for this session.
    ///
    /// Mirrors `pyspark.sql.SparkSession.catalog`.
    pub fn catalog(&self) -> crate::catalog::Catalog {
        crate::catalog::Catalog::new(self.clone())
    }

    /// Register a Java UDF/UDAF by class name, mirroring the reference
    /// `client.register_java(name, javaClassName, return_type, aggregate)` used by
    /// `UDFRegistration.registerJavaFunction` / `registerJavaUDAF`. `return_type_ddl`, a DDL
    /// string, is sent unparsed, for the server to parse, as PySpark sends a string: see
    /// [`SparkSession::register_java_function_with`].
    pub fn register_java_function(
        &self,
        name: &str,
        java_class_name: &str,
        return_type_ddl: Option<&str>,
        aggregate: bool,
    ) -> Result<()> {
        let return_type = return_type_ddl.map(|ddl| DataType::Unparsed {
            data_type_string: ddl.to_string(),
        });
        self.register_java_function_with(name, java_class_name, return_type, aggregate)
    }

    /// Register a Java UDF/UDAF by class name, of the return type `return_type`, as PySpark's
    /// `UDFRegistration.registerJavaFunction` / `registerJavaUDAF`: the
    /// `CommonInlineUserDefinedFunction` of a `JavaUDF`, which is not deterministic, sent as the
    /// `RegisterFunction` command. `aggregate` is only used without a return type (matching
    /// the reference, which omits it for a function of an output type).
    pub fn register_java_function_with(
        &self,
        name: &str,
        java_class_name: &str,
        return_type: Option<DataType>,
        aggregate: bool,
    ) -> Result<()> {
        let mut java_udf = proto::JavaUdf::default();
        java_udf.class_name = java_class_name.to_string();
        if let Some(return_type) = return_type {
            java_udf.output_type = Some(return_type.to_proto());
        } else {
            java_udf.aggregate = aggregate;
        }
        let mut fun = proto::CommonInlineUserDefinedFunction::default();
        fun.function_name = name.to_string();
        // As PySpark's CommonInlineUserDefinedFunction, which is not deterministic by default.
        fun.deterministic = false;
        fun.function =
            Some(proto::common_inline_user_defined_function::Function::JavaUdf(java_udf));
        crate::dataframe::execute_registration(
            self,
            proto::command::CommandType::RegisterFunction(fun),
        )?;
        Ok(())
    }

    /// Execute `command`, a command built outside the core, such as the ML commands of
    /// `pyspark.ml`, and return the responses of the server, as the Python client's
    /// `execute_command` reads them, without Observations, as `pyspark.ml` runs them. The
    /// placeholders of its relations are replaced, as `DataFrame::from_relation` does, of the
    /// search that its encoding allows, if the core is given it (see
    /// [`crate::plan::BuiltOutside`]). Not a stable API.
    #[doc(hidden)]
    pub fn execute_command_proto(
        &self,
        command: impl Into<crate::plan::BuiltOutside<spark_connect_proto::Command>>,
    ) -> Result<Vec<spark_connect_proto::ExecutePlanResponse>> {
        let (mut command, urls) = command.into().into_parts();
        crate::plan::resolve_command(&mut command, urls)?;
        let Some(command_type) = command.command_type else {
            return Err(SparkError::value_msg("a command of no command type"));
        };
        let (responses, _) = crate::dataframe::execute_command_with_info(self, command_type, &[])?;
        Ok(responses)
    }

    /// Execute `plan`, a plan built outside the core, such as one that PySpark's
    /// `SparkConnectClient.to_table` or `execute_command_as_iterator` is given, and return its
    /// responses as the server sends them, once they are read (see
    /// [`PlanExecution`](crate::dataframe::PlanExecution)). The placeholders of its relation or its
    /// command are replaced, as [`SparkSession::execute_command_proto`] replaces them. The metrics
    /// that it observes set `observations`, by their names, as PySpark's client sets the
    /// Observations that it is given, and its progress reaches the progress handlers when
    /// `reported`, as PySpark reports the progress of all its executions but those of
    /// `execute_command_as_iterator`. A plan of a relation is sent as it is, as `to_table` sends
    /// the plan that it is given, and one of a command compressed, as `execute_command` sets its
    /// command in its plan. Not a stable API.
    #[doc(hidden)]
    pub fn execute_plan_proto(
        &self,
        plan: impl Into<crate::plan::BuiltOutside<proto::Plan>>,
        observations: Vec<(String, crate::observation::Observation)>,
        reported: bool,
    ) -> Result<crate::dataframe::PlanExecution> {
        let plan = plan.into();
        let log = match plan.message().op_type.as_ref() {
            Some(proto::plan::OpType::Root(_)) => ExecutionLog::ToTable,
            _ => ExecutionLog::ExecuteCommand,
        };
        crate::dataframe::PlanExecution::start(self, plan, observations, reported, log)
    }

    /// As [`SparkSession::execute_plan_proto`], of the logs of the method of PySpark's client that
    /// is given the plan, as `log` says, and of its compression: a plan of `to_table`,
    /// `to_table_as_iterator` and `to_pandas` is sent as it is, and the command of
    /// `execute_command` and `execute_command_as_iterator` compressed. Not a stable API.
    #[doc(hidden)]
    pub fn execute_plan_proto_logged(
        &self,
        plan: impl Into<crate::plan::BuiltOutside<proto::Plan>>,
        observations: Vec<(String, crate::observation::Observation)>,
        reported: bool,
        log: ExecutionLog,
    ) -> Result<crate::dataframe::PlanExecution> {
        crate::dataframe::PlanExecution::start(self, plan.into(), observations, reported, log)
    }

    /// The response of the server to `request`, an analysis built outside the core, such as the
    /// analyses of PySpark's `SparkConnectClient.schema`, `explain_string`, `same_semantics` and
    /// `semantic_hash`: of this session, with the placeholders of its plans replaced, of the search
    /// that the encoding of the request allows, if the core is given it (see
    /// [`crate::plan::BuiltOutside`]), retried and checked as the analyses of the core are. Its
    /// plans are sent as they are, neither compressed nor of the lookup of the plan compression of
    /// the session, as PySpark's `_analyze` sends the plans that it is given, which `to_proto`
    /// compressed. Not a stable API.
    #[doc(hidden)]
    pub fn analyze_plan_proto(
        &self,
        request: impl Into<crate::plan::BuiltOutside<proto::AnalyzePlanRequest>>,
    ) -> Result<proto::AnalyzePlanResponse> {
        let (mut request, urls) = request.into().into_parts();
        request.session_id = self.client().session_id().to_string();
        if let Some(analyze) = request.analyze.as_mut() {
            crate::plan::resolve_analysis(analyze, urls)?;
        }
        block_on(
            self.client()
                .analyze_plan_as_is(request, spark_connect_core::client_log::AnalysisLog::Silent),
        )
    }

    /// `plan`, a plan built outside the core (see [`crate::plan::BuiltOutside`]), with its
    /// placeholders replaced, as the session sends it. Not a stable API.
    #[doc(hidden)]
    pub fn resolved_plan_proto(
        &self,
        plan: impl Into<crate::plan::BuiltOutside<proto::Plan>>,
    ) -> Result<proto::Plan> {
        let (mut plan, urls) = plan.into().into_parts();
        crate::plan::resolve_plan(&mut plan, urls)?;
        Ok(plan)
    }

    /// `plan`, a plan built outside the core of a relation or a command (see
    /// [`crate::plan::BuiltOutside`]), compressed as the session compresses its plans, after the
    /// lookup of the plan compression of its first plan, as PySpark's `_try_compress_and_set_plan`
    /// compresses the plan of `to_proto`, with what it logs then (see
    /// `SparkConnectClient::prepare_plan`): of its placeholders replaced, and None when it is not
    /// compressed, as a plan of a relation or a command of no more bytes than the threshold, or of
    /// a compression that is not effective, which PySpark's sends as it is. Not a stable API.
    #[doc(hidden)]
    pub fn compressed_plan_proto(
        &self,
        plan: impl Into<crate::plan::BuiltOutside<proto::Plan>>,
    ) -> Result<Option<proto::Plan>> {
        let mut plan = self.resolved_plan_proto(plan)?;
        block_on(self.client().prepare_plan(&mut plan))?;
        Ok(match plan.op_type {
            Some(proto::plan::OpType::CompressedOperation(_)) => Some(plan),
            _ => None,
        })
    }

    /// The response of the server to `request`, a configuration request built outside the core,
    /// such as that of PySpark's `SparkConnectClient.config`: of this session, retried and checked
    /// as the configuration requests of the core are. Not a stable API.
    #[doc(hidden)]
    pub fn config_proto(&self, mut request: proto::ConfigRequest) -> Result<proto::ConfigResponse> {
        request.session_id = self.client().session_id().to_string();
        block_on(self.client().config(request))
    }

    /// Cache `blobs` as artifacts of the session, of their SHA-256 hashes, which are returned in
    /// the order of the blobs, as PySpark's `SparkConnectClient.cache_artifacts`: one
    /// `ArtifactStatus` request asks which are cached on the server already, and one
    /// `AddArtifacts` uploads each other one once.
    pub fn cache_artifacts(&self, blobs: Vec<Vec<u8>>) -> Result<Vec<String>> {
        block_on(self.client().cache_artifacts(blobs))
    }

    /// The retry policy of the RPCs of this session, as the policies of PySpark's
    /// `SparkConnectClient.get_retry_policies` retry them.
    pub fn retry_policy(&self) -> spark_connect_core::retries::RetryPolicy {
        self.client().retry_policy()
    }

    /// Replace the retry policy of the RPCs of this session and of its clones, as PySpark's
    /// `SparkConnectClient.set_retry_policies`: the calls that start from now on retry under
    /// `policy`. A new session ([`SparkSession::new_session`], [`SparkSession::clone_session`])
    /// has the default policy, as PySpark's new client does.
    pub fn set_retry_policy(&self, policy: spark_connect_core::retries::RetryPolicy) {
        self.client().replace_retry_policy(policy);
    }

    /// Read the user agent of the requests of this session again (see
    /// [`spark_connect_core::client::SparkConnectClient::refresh_user_agent`]). Not a stable API.
    #[doc(hidden)]
    pub fn refresh_user_agent(&self) {
        self.client().refresh_user_agent();
    }

    /// The result chunking options of the executions of the client of this session, as PySpark's
    /// client sends them with each `ExecutePlanRequest` (see
    /// `SparkConnectClient::result_chunking`). Not a stable API.
    #[doc(hidden)]
    pub fn result_chunking(&self) -> spark_connect_core::client::ResultChunking {
        self.client().result_chunking()
    }

    /// The plan compression of the client of this session once its settings are read, and None
    /// before (see `SparkConnectClient::known_plan_compression`): of a plan of no more bytes than
    /// the threshold, or of none, [`SparkSession::compressed_plan_proto`] gives None, the plan as
    /// it is, but of a placeholder of a DataFrame that holds no relation, which may stand for more.
    /// Not a stable API.
    #[doc(hidden)]
    pub fn known_plan_compression(&self) -> Option<Option<usize>> {
        self.client().known_plan_compression()
    }

    /// The per-RPC deadlines of the calls of this session (see
    /// [`spark_connect_core::client::RpcDeadlines`]).
    pub fn rpc_deadlines(&self) -> spark_connect_core::client::RpcDeadlines {
        self.client().rpc_deadlines().clone()
    }

    /// Make `decider` decide the retries of the RPCs of this session and of its clones that start
    /// from now on, in place of its retry policy, or the policy again with none (see
    /// [`spark_connect_core::retries::RetryDecider`]). A new session has none. Not a stable API.
    #[doc(hidden)]
    pub fn set_retry_decider(
        &self,
        decider: Option<Arc<dyn spark_connect_core::retries::RetryDecider>>,
    ) {
        self.client().set_retry_decider(decider);
    }

    /// Give the calls of this session the metadata of `source` from now on, in place of the
    /// metadata of its connection, or that of the connection again with none (see
    /// `SparkConnectClient::set_metadata_source`). A new session has none. Not a stable API.
    #[doc(hidden)]
    pub fn set_metadata_source(
        &self,
        calls: Option<Arc<dyn spark_connect_core::channel::MetadataSource>>,
        artifacts: Option<Arc<dyn spark_connect_core::channel::MetadataSource>>,
    ) {
        self.client().set_metadata_source(calls, artifacts);
    }

    /// Make the executions of this session reattachable or not, from the next one on, as PySpark's
    /// `SparkConnectClient.enable_reattachable_execute` and `disable_reattachable_execute` (see
    /// [`SparkConnectClient::set_reattachable_execute`]). A new session keeps the setting, as
    /// PySpark's new client does.
    pub fn set_reattachable_execute(&self, reattachable: bool) {
        self.client().set_reattachable_execute(reattachable);
    }

    /// Whether the executions of this session are reattachable, which they are by default.
    pub fn reattachable_execute(&self) -> bool {
        self.client().reattachable_execute()
    }

    /// The `token` parameter of the connection string of this session, if it has one, which
    /// PySpark's `SparkConnectClient.token` reads before the environment's
    /// `SPARK_CONNECT_AUTHENTICATE_TOKEN`.
    pub fn token_parameter(&self) -> Option<&str> {
        self.client().token_parameter()
    }

    /// Make the requests of the executions, the analyses and the configurations of the calling
    /// thread carry `call_stack`, the call stack of the user's code of its call, or none (see
    /// `SparkConnectClient::set_thread_call_stack`). Not a stable API.
    #[doc(hidden)]
    pub fn set_thread_call_stack(&self, call_stack: Option<prost_types::Any>) {
        self.client().set_thread_call_stack(call_stack);
    }

    /// Delete the ML cached objects `refs` on the server, as PySpark's `RemoteModelRef` does
    /// when the last reference to a model goes: through its `_delete_ml_cache`, one
    /// `execute_command` of the tags and the user context extensions of the thread that drops
    /// the model, the thread of its finalizer. It does not wait for the server: the request is
    /// built on this thread and runs on the shared runtime (see
    /// `SparkConnectClient::spawn_execute_plan_reattachable`), and its errors (a closed channel)
    /// are ignored. As `release_cached_remote_relation`, a stopped session is not released:
    /// stopping it released its server state, ML cache included. Safe to call from `Drop`.
    pub(crate) fn release_ml_objects(&self, refs: Vec<proto::ObjectRef>) {
        if self.is_stopped() {
            return;
        }
        let (_request, send) = self.finalizer_request();
        if !send {
            return;
        }
        // A tag that UTF-8 cannot encode fails PySpark's request before it is sent, and
        // `_delete_ml_cache` ignores the error.
        let Ok(tags) = self.tags() else {
            return;
        };
        let command = proto::Command {
            command_type: Some(proto::command::CommandType::MlCommand(proto::MlCommand {
                command: Some(proto::ml_command::Command::Delete(
                    proto::ml_command::Delete {
                        obj_refs: refs,
                        // As PySpark's `_delete_ml_cache`, whose `evict_only` is False.
                        evict_only: Some(false),
                    },
                )),
            })),
        };
        let request = proto::ExecutePlanRequest {
            session_id: self.client().session_id().to_string(),
            user_context: Some(proto::UserContext::default()),
            operation_id: Some(uuid::Uuid::new_v4().to_string()),
            tags,
            plan: Some(proto::Plan {
                op_type: Some(proto::plan::OpType::Command(command)),
            }),
            ..Default::default()
        };
        self.client().spawn_execute_plan_reattachable(request);
    }

    /// Remove the cached remote relation `relation_id` on the server, as PySpark's
    /// `CachedRemoteRelation.__del__` does when its DataFrame goes: one ExecutePlan, which is not
    /// reattachable and goes to no hook, of the tags of the thread that drops the DataFrame, as
    /// PySpark's request is of the thread of its finalizer (see
    /// `SparkConnectClient::spawn_unary_execute_plan`). It does not wait for the server, so that
    /// a drop never blocks, and its errors are ignored. A stopped session is not released:
    /// stopping it released its server state already, as PySpark skips a closed client. Safe to
    /// call from `Drop`.
    pub(crate) fn release_cached_remote_relation(&self, relation_id: String) {
        if self.is_stopped() {
            return;
        }
        let (_request, send) = self.finalizer_request();
        if !send {
            return;
        }
        // A tag that UTF-8 cannot encode fails PySpark's request before it is sent.
        let Ok(tags) = self.tags() else {
            return;
        };
        let command = proto::Command {
            command_type: Some(
                proto::command::CommandType::RemoveCachedRemoteRelationCommand(
                    proto::RemoveCachedRemoteRelationCommand {
                        relation: Some(proto::CachedRemoteRelation { relation_id }),
                    },
                ),
            ),
        };
        let request = proto::ExecutePlanRequest {
            session_id: self.client().session_id().to_string(),
            tags,
            plan: Some(proto::Plan {
                op_type: Some(proto::plan::OpType::Command(command)),
            }),
            ..Default::default()
        };
        self.client().spawn_unary_execute_plan(request);
    }

    /// The request of a finalizer of this session on the calling thread, and whether to send it,
    /// after the `before` function of the hook of [`set_finalizer_request_hook`], if any, brought
    /// the context of the requests of the thread up to date: as PySpark's finalizers build their
    /// requests when they run, of the attributes of the client then, and send none when they fail
    /// to build them. The thread has its own context again once the returned request drops, after
    /// the request is built, so that a call during which the finalizer runs, as on a collection
    /// of the garbage collector, sends the rest of its requests of its own, which the `after`
    /// function of the hook reads again then.
    fn finalizer_request(&self) -> (FinalizerRequest<'_>, bool) {
        let request = FinalizerRequest {
            session: self,
            kept: Some(self.keep_thread_request_context()),
        };
        let send = match FINALIZER_REQUEST_HOOK.get() {
            Some(hook) => (hook.before)(self),
            None => true,
        };
        (request, send)
    }

    /// Get the runtime configuration for this session.
    ///
    /// Mirrors `pyspark.sql.SparkSession.conf`.
    pub fn conf(&self) -> crate::conf::RuntimeConf {
        crate::conf::RuntimeConf::new(Arc::clone(self.client()))
    }

    /// Get table-valued functions for this session.
    ///
    /// Mirrors `pyspark.sql.SparkSession.tvf`.
    pub fn tvf(&self) -> crate::tvf::TableValuedFunction {
        crate::tvf::TableValuedFunction::new(self.clone())
    }

    /// The session ID of this session.
    ///
    /// Mirrors `pyspark.sql.SparkSession.session_id` / the connect client session id.
    pub fn session_id(&self) -> &str {
        self.client().session_id()
    }

    /// The display host of the endpoint of this session, as PySpark's `SparkConnectClient.host`.
    pub fn host(&self) -> &str {
        self.client().host()
    }

    /// The server-side session ID that the responses of this session carried, or None before the
    /// first response, as PySpark's `SparkConnectClient._server_session_id`.
    pub fn server_session_id(&self) -> Option<String> {
        self.client().observed_server_session_id()
    }

    /// Set the server-side session ID that the requests of this session observe.
    pub fn set_server_session_id(&self, id: Option<String>) {
        self.client().set_observed_server_session_id(id);
    }

    /// Add `extension` to the user context of the requests that the current thread sends from
    /// this session, and return its ID. Mirrors PySpark's
    /// `SparkConnectClient.add_threadlocal_user_context_extension`.
    pub fn add_threadlocal_user_context_extension(&self, extension: prost_types::Any) -> String {
        self.client()
            .add_threadlocal_user_context_extension(extension)
    }

    /// Add `extension` to the user context of the requests of all the threads of this session,
    /// and return its ID. Mirrors PySpark's `add_global_user_context_extension`.
    pub fn add_global_user_context_extension(&self, extension: prost_types::Any) -> String {
        self.client().add_global_user_context_extension(extension)
    }

    /// Replace the global user context extensions with `extensions`, of their IDs, in their order.
    /// Mirrors a change of PySpark's `global_user_context_extensions` of a client, which its
    /// requests take.
    pub fn set_global_user_context_extensions(&self, extensions: Vec<(String, prost_types::Any)>) {
        self.client().set_global_user_context_extensions(extensions);
    }

    /// Remove the user context extension of `extension_id`. Mirrors PySpark's
    /// `remove_user_context_extension`.
    pub fn remove_user_context_extension(&self, extension_id: &str) {
        self.client().remove_user_context_extension(extension_id);
    }

    /// The user context extensions of the current thread, of their IDs. Mirrors PySpark's
    /// `thread_local.user_context_extensions` of a client.
    pub fn threadlocal_user_context_extensions(&self) -> Vec<(String, prost_types::Any)> {
        self.client().threadlocal_user_context_extensions()
    }

    /// The global user context extensions, of their IDs. Mirrors PySpark's
    /// `global_user_context_extensions` of a client.
    pub fn global_user_context_extensions(&self) -> Vec<(String, prost_types::Any)> {
        self.client().global_user_context_extensions()
    }

    /// Remove the user context extensions of the current thread and the global ones. Mirrors
    /// PySpark's `clear_user_context_extensions`.
    pub fn clear_user_context_extensions(&self) {
        self.client().clear_user_context_extensions();
    }

    /// A copy of the user context extensions of the current thread, which the threads that are
    /// given it with [`SparkSession::share_threadlocal_user_context_extensions`] share. Mirrors
    /// how PySpark's `inheritable_thread_target(session)` copies the `thread_local` of the client
    /// of the thread that decorates a function.
    pub fn copy_threadlocal_user_context_extensions(
        &self,
    ) -> spark_connect_core::client::SharedUserContextExtensions {
        self.client().copy_threadlocal_user_context_extensions()
    }

    /// Make `extensions` the user context extensions of the current thread, which it shares with
    /// the other threads given them, as `inheritable_thread_target` gives every thread that runs
    /// its function the copy of the `thread_local` of the client.
    pub fn share_threadlocal_user_context_extensions(
        &self,
        extensions: &spark_connect_core::client::SharedUserContextExtensions,
    ) {
        self.client()
            .share_threadlocal_user_context_extensions(extensions);
    }

    /// The session ID that the requests of this session carry, as PySpark's `client._session_id`
    /// (see [`SparkConnectClient::client_session_id`]). Not a stable API.
    #[doc(hidden)]
    pub fn client_session_id(&self) -> String {
        self.client().client_session_id()
    }

    /// Make the requests of this session carry `session_id`, as a set of PySpark's
    /// `client._session_id` (see [`SparkConnectClient::set_client_session_id`]):
    /// [`SparkSession::session_id`] stays the ID of the session. Not a stable API.
    #[doc(hidden)]
    pub fn set_client_session_id(&self, session_id: String) {
        self.client().set_client_session_id(session_id);
    }

    /// The channel builder of the connection of this session, as PySpark's `client._builder`
    /// (see [`SparkConnectClient::channel_builder`]). Not a stable API.
    #[doc(hidden)]
    pub fn channel_builder(&self) -> &ChannelBuilder {
        self.client().channel_builder()
    }

    /// The user of the connection of this session: of the connection string, else of the
    /// environment, if any, which the requests of its artifacts carry. Not a stable API.
    #[doc(hidden)]
    pub fn user_id(&self) -> Option<&str> {
        self.client().user_id()
    }

    /// The user that the other requests of this session carry, as PySpark's `client._user_id`
    /// (see [`SparkConnectClient::request_user_id`]). Not a stable API.
    #[doc(hidden)]
    pub fn request_user_id(&self) -> Option<String> {
        self.client().request_user_id()
    }

    /// Make the requests of this session but those of its artifacts carry the user `user_id`, or
    /// none, as a set of PySpark's `client._user_id` (see
    /// [`SparkConnectClient::set_request_user_id`]). Not a stable API.
    #[doc(hidden)]
    pub fn set_request_user_id(&self, user_id: Option<String>) {
        self.client().set_request_user_id(user_id);
    }

    /// Make the requests of the current thread carry the session ID, the user and the global user
    /// context extensions of the client that it calls (see
    /// [`SparkConnectClient::set_thread_request_context`]). Not a stable API.
    #[doc(hidden)]
    pub fn set_thread_request_context(
        &self,
        session_id: String,
        user_id: Option<String>,
        global: Vec<(String, prost_types::Any)>,
    ) {
        self.client()
            .set_thread_request_context(session_id, user_id, global);
    }

    /// Give the calls of the current thread the deadlines of their RPCs and the result chunking
    /// options of their executions of the client that it calls, or those of the session again of
    /// None (see [`SparkConnectClient::set_thread_call_settings`]). Binding hook; not a stable
    /// API.
    #[doc(hidden)]
    pub fn set_thread_call_settings(
        &self,
        rpc_deadlines: Option<spark_connect_core::client::RpcDeadlines>,
        result_chunking: Option<spark_connect_core::client::ResultChunking>,
    ) {
        self.client()
            .set_thread_call_settings(rpc_deadlines, result_chunking);
    }

    /// The settings of the plan compression of the client of this session, as PySpark's client
    /// keeps them, once they are read (see `SparkConnectClient::plan_compression_settings`).
    /// Binding hook; not a stable API.
    #[doc(hidden)]
    pub fn plan_compression_settings(
        &self,
    ) -> Option<spark_connect_core::client::PlanCompressionSettings> {
        self.client().plan_compression_settings()
    }

    /// Make `settings` the plan compression of the client of this session and of its clones, or
    /// look it up again with the next plan of None (see
    /// `SparkConnectClient::set_plan_compression_settings`). Binding hook; not a stable API.
    #[doc(hidden)]
    pub fn set_plan_compression_settings(
        &self,
        settings: Option<spark_connect_core::client::PlanCompressionSettings>,
    ) {
        self.client().set_plan_compression_settings(settings);
    }

    /// Keep the context of the requests of the calling thread, as it is now, until the returned
    /// guard drops, when the thread has it again: the session ID, the user and the global user
    /// context extensions of [`SparkSession::set_thread_request_context`], the user context
    /// extensions of the thread and its tags. As a binding keeps them around a call that runs
    /// during another call of the thread, such as one of a progress handler or of a finalizer,
    /// whose own context would otherwise be that of the requests that the other call sends once
    /// it returned, as the details of its failure, where PySpark's requests read the attributes of
    /// their own client. Not a stable API.
    #[doc(hidden)]
    pub fn keep_thread_request_context(&self) -> KeptThreadRequestContext<'_> {
        let (thread, tags) = self.inner.tags.of_calling_thread();
        KeptThreadRequestContext {
            session: self,
            thread,
            client: Some(self.client().thread_request_state()),
            tags,
        }
    }

    /// Open this session again, which `close()` closed, as PySpark's `client._closed = False`
    /// lets the requests of its client be sent again: of the `ValueError` of a closed channel,
    /// once `close()` closed it (see [`SparkConnectClient::close`]). Not a stable API.
    #[doc(hidden)]
    pub fn reopen(&self) {
        self.inner.stopped.store(false, Ordering::SeqCst);
        self.client().reopen();
    }

    /// Mark this session closed, as PySpark's `client._closed = True`: it is stopped, and its
    /// requests raise `NO_ACTIVE_SESSION`, but its channel stays open, so that those of its
    /// artifacts are sent, as PySpark's ArtifactManager sends them, and `reopen()` lets the
    /// others be sent again. Not a stable API.
    #[doc(hidden)]
    pub fn mark_closed(&self) {
        self.client().set_closed(true);
    }

    /// The statuses of the operations of this session of `request`, a GetStatus request built
    /// outside the core, as PySpark's `_get_operation_statuses` (see
    /// [`SparkConnectClient::get_status_proto`]). Not a stable API.
    #[doc(hidden)]
    pub fn get_status_proto(
        &self,
        request: proto::GetStatusRequest,
    ) -> Result<proto::GetStatusResponse> {
        block_on(self.client().get_status_proto(request))
    }

    /// Keep `refresher`, or none, for this session: what a binding keeps to bring the user
    /// context extensions up to date before a call of a thread sends requests (see
    /// [`SparkConnectClient::set_user_context_refresher`]). Not a stable API.
    #[doc(hidden)]
    pub fn set_user_context_refresher(
        &self,
        refresher: Option<Arc<dyn std::any::Any + Send + Sync>>,
    ) {
        self.client().set_user_context_refresher(refresher);
    }

    /// The refresher of [`SparkSession::set_user_context_refresher`], if any. Not a stable API.
    #[doc(hidden)]
    pub fn user_context_refresher(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        self.client().user_context_refresher()
    }

    /// Forget the user context extensions of the current thread, as PySpark's
    /// `thread_local` once its `user_context_extensions` is deleted. Not a stable API.
    #[doc(hidden)]
    pub fn clear_threadlocal_user_context_extensions(&self) {
        self.client().clear_threadlocal_user_context_extensions();
    }

    /// Return the Spark version of the connected server.
    ///
    /// Mirrors `pyspark.sql.SparkSession.version`.
    /// Only successful lookups are cached, including after `stop()`, matching
    /// Python Connect's cached property. New sessions resolve independently.
    pub fn version(&self) -> Result<String> {
        // Serialize the first lookup across clones; errors leave the cache empty.
        // A panicking lookup cannot leave a partially initialized String behind.
        let mut cached = self
            .inner
            .server_version
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(version) = cached.as_ref() {
            return Ok(version.clone());
        }
        let mut request = proto::AnalyzePlanRequest::default();
        request.session_id = self.client().session_id().to_string();
        request.user_context = Some(proto::UserContext::default());
        request.analyze = Some(proto::analyze_plan_request::Analyze::SparkVersion(
            proto::analyze_plan_request::SparkVersion::default(),
        ));
        let resp = block_on(self.client().analyze_plan(request))?;
        match resp.result {
            Some(proto::analyze_plan_response::Result::SparkVersion(v)) => {
                *cached = Some(v.version.clone());
                Ok(v.version)
            }
            _ => Err(SparkError::connect_msg(
                "AnalyzePlan response did not contain a spark version",
            )),
        }
    }

    /// Return the DataFrame for the given table/view.
    ///
    /// Mirrors `pyspark.sql.SparkSession.table(tableName)`.
    pub fn table(&self, table_name: &str) -> Result<DataFrame> {
        Ok(self.read().table(table_name))
    }

    /// Return an empty DataFrame with no rows and an empty schema.
    ///
    /// Mirrors `pyspark.sql.SparkSession.createDataFrame([], StructType([]))` /
    /// `SparkSession.emptyDataFrame`.
    pub fn empty_data_frame(&self) -> Result<DataFrame> {
        let plan = LogicalPlan::LocalRelation {
            schema: Some(DataType::Struct { fields: vec![] }),
            data: None,
        };
        Ok(DataFrame::new(self.clone(), plan))
    }

    /// Interrupt all operations of this session.
    ///
    /// Mirrors `pyspark.sql.SparkSession.interruptAll()`. Returns interrupted operation ids.
    pub fn interrupt_all(&self) -> Result<Vec<String>> {
        block_on(self.client().interrupt_all())
    }

    /// Interrupt all operations of this session with the given tag.
    ///
    /// Mirrors `pyspark.sql.SparkSession.interruptTag(tag)`.
    pub fn interrupt_tag(&self, tag: &str) -> Result<Vec<String>> {
        block_on(self.client().interrupt_tag(tag))
    }

    /// Interrupt the operation with the given operation id.
    ///
    /// Mirrors `pyspark.sql.SparkSession.interruptOperation(opId)`.
    pub fn interrupt_operation(&self, operation_id: &str) -> Result<Vec<String>> {
        block_on(self.client().interrupt_operation(operation_id))
    }

    /// Interrupt the operation with the given operation id, as a future, for a wait that drives
    /// it (see [`spark_connect_core::runtime::with_wait_watch`]).
    pub async fn interrupt_operation_async(&self, operation_id: &str) -> Result<Vec<String>> {
        self.client().interrupt_operation(operation_id).await
    }

    /// The operation ID of the innermost execution in flight that the calling thread started, if
    /// any. A wait for an execution interrupts it with [`Self::interrupt_operation_async`] when a
    /// signal of the process raises, as PySpark's `KeyboardInterrupt` does.
    pub fn running_operation_id(&self) -> Option<String> {
        self.client().running_operation_id()
    }

    /// Whether the calling thread converts the failure of a call of the session now, during which
    /// the failure of another call of the session is not converted (see
    /// [`spark_connect_core::client::SparkConnectClient::inside_error_handling`]). Not a stable
    /// API.
    #[doc(hidden)]
    pub fn inside_error_handling(&self) -> bool {
        self.client().inside_error_handling()
    }

    /// Add local files, or `file:` URIs, as artifacts of the session, mirroring
    /// `pyspark.sql.SparkSession.addArtifacts(*path)`: a `.jar` is added to the classpath and
    /// a `.class` to the classes (a Rust extension: PySpark does not upload class files); any
    /// other file is not supported without a flag of
    /// [`add_artifacts_with`](Self::add_artifacts_with).
    pub fn add_artifacts(&self, paths: &[&str]) -> Result<()> {
        self.add_artifacts_with(paths, false, false, false)
    }

    /// Add local files as artifacts of the session, mirroring
    /// `pyspark.sql.SparkSession.addArtifacts(*path, pyfile, archive, file)`:
    /// - `pyfile`: a `.py`, `.egg`, `.zip` or `.jar` is importable by Python UDFs;
    /// - `archive`: a `.zip`, `.jar`, `.tar.gz`, `.tgz` or `.tar` is extracted, into the
    ///   directory that a `#<name>` fragment of its path names;
    /// - `file`: any file is added to the files of the session.
    ///
    /// At most one of the flags may be true.
    pub fn add_artifacts_with(
        &self,
        paths: &[&str],
        pyfile: bool,
        archive: bool,
        file: bool,
    ) -> Result<()> {
        if [pyfile, archive, file].iter().filter(|flag| **flag).count() > 1 {
            return Err(SparkError::connect_msg(
                "[INVALID_MULTIPLE_ARGUMENT_CONDITIONS] ['pyfile', 'archive' and/or 'file'] \
                 cannot be True together.",
            ));
        }
        self.add_artifacts_of_flags(paths, pyfile, archive, file)
    }

    /// Add local files as artifacts of the session, as [`SparkSession::add_artifacts_with`],
    /// with any of the flags true: each path is named by the first flag that its extension
    /// matches, `pyfile`, then `archive`, then `file`, as PySpark's
    /// `SparkConnectClient.add_artifacts` names it, which does not check the flags as
    /// `SparkSession.addArtifacts` does.
    pub fn add_artifacts_of_flags(
        &self,
        paths: &[&str],
        pyfile: bool,
        archive: bool,
        file: bool,
    ) -> Result<()> {
        block_on(self.client().add_artifacts(paths, pyfile, archive, file))
    }

    /// Add a single local file as an artifact to the session.
    ///
    /// Mirrors `pyspark.sql.SparkSession.addArtifact(path)`.
    pub fn add_artifact(&self, path: &str) -> Result<()> {
        self.add_artifacts(&[path])
    }

    /// Copy a local file to the driver's filesystem at `dest_path`.
    ///
    /// Mirrors `pyspark.sql.SparkSession.copyFromLocalToFs`: uploads the file as a
    /// `forward_to_fs<dest_path>` artifact, which the server writes to `dest_path`. As the
    /// reference, `dest_path` is an absolute path, so the name is the prefix and the path.
    pub fn copy_from_local_to_fs(&self, local_path: &str, dest_path: &str) -> Result<()> {
        block_on(
            self.client()
                .add_named_artifact(&forward_to_fs_artifact_name(dest_path), local_path),
        )
    }

    /// Register a user-defined function on the session so it can be referenced by
    /// name in SQL / expressions.
    ///
    /// Mirrors the server-side effect of `pyspark.sql.SparkSession.udf.register` /
    /// `udtf.register`: the UDF is cloudpickled on the client (see
    /// [`crate::udf`]) and sent as a `RegisterFunction` command.
    pub fn register_function(
        &self,
        udf: crate::udf::CommonInlineUserDefinedFunctionExpression,
    ) -> Result<()> {
        // An argument of the function is serialized as a request serializes its expressions.
        let function = crate::expression::with_expression_depth_bound(|| udf.to_proto())?;
        crate::dataframe::execute_registration(
            self,
            proto::command::CommandType::RegisterFunction(function),
        )
    }

    /// Register a Python user-defined table function (UDTF) as `name` on the server, so that
    /// SQL calls it: a `register_table_function` command of the
    /// `CommonInlineUserDefinedTableFunction` of the UDTF, without arguments. Mirrors
    /// `SparkConnectClient.register_udtf`, of `spark.udtf.register`. The Python client
    /// cloudpickles the handler class as `command`, and sets `eval_type` (300 SQL_TABLE_UDF,
    /// 301 SQL_ARROW_TABLE_UDF, 302 SQL_ARROW_UDTF).
    pub fn register_table_function(
        &self,
        name: &str,
        return_type: Option<&DataType>,
        eval_type: i32,
        command: &[u8],
        python_ver: &str,
        deterministic: bool,
    ) -> Result<()> {
        let function = crate::plan::common_inline_udtf(
            name,
            deterministic,
            &[],
            return_type,
            eval_type,
            (command, None),
            python_ver,
        );
        crate::dataframe::execute_registration(
            self,
            proto::command::CommandType::RegisterTableFunction(function),
        )
    }

    /// Build and register a ResourceProfile with the server.
    ///
    /// Sends a `CreateResourceProfileCommand` to the server with the specified executor
    /// and task resource requests, and returns the server-assigned profile id.
    /// The profile can then be used with `DataFrame.withResources(profile_id)`.
    ///
    /// Mirrors `pyspark.sql.SparkSession._build_resource_profile` (internal).
    pub fn build_resource_profile(
        &self,
        profile: &crate::resource::ResourceProfile,
    ) -> Result<i32> {
        let mut cmd = proto::CreateResourceProfileCommand::default();
        cmd.profile = Some(profile.proto().clone());

        let responses = crate::dataframe::execute_command_collect(
            self,
            proto::command::CommandType::CreateResourceProfileCommand(cmd),
        )?;

        for resp in responses {
            if let Some(
                proto::execute_plan_response::ResponseType::CreateResourceProfileCommandResult(res),
            ) = resp.response_type
            {
                return Ok(res.profile_id);
            }
        }

        Err(SparkError::connect_msg(
            "build_resource_profile: server returned no CreateResourceProfileCommandResult",
        ))
    }

    /// Register a user-defined data source on the session so it can be referenced
    /// in SQL queries.
    ///
    /// Mirrors the server-side effect of `pyspark.sql.SparkSession.dataSource.register`.
    /// The data source is cloudpickled on the Python client and sent as a
    /// `RegisterDataSource` command. Since Rust cannot cloudpickle Python classes, the
    /// command bytes must be prepared on the client (typically by a Python wrapper).
    pub fn register_data_source(
        &self,
        data_source: crate::datasource::CommonInlineUserDefinedDataSourceExpression,
    ) -> Result<()> {
        crate::dataframe::execute_registration(
            self,
            proto::command::CommandType::RegisterDataSource(data_source.to_proto()),
        )
    }

    /// Set the handler of the updates of the Python accumulators of the executions of this
    /// session: each pickled `(id, update)` of their `__python_accumulator__` observed metrics,
    /// of which the UDF profilers send their results, as PySpark Connect's client unpickles each
    /// into its `ConnectProfilerCollector`.
    pub fn set_python_accumulator_handler(&self, handler: impl Fn(&[u8]) + Send + Sync + 'static) {
        self.inner
            .python_accumulators
            .set_handler(Arc::new(handler));
    }

    /// The handler of the updates of the Python accumulators, of the executions.
    pub(crate) fn python_accumulators(&self) -> &PythonAccumulators {
        &self.inner.python_accumulators
    }

    /// Set the check of the metrics that the executions of this session, and of its clones,
    /// observe for the Observations of their plans, made before the metrics of each are set: an
    /// error of the check is the error of the execution, and that Observation keeps the metrics
    /// that it had, as do those whose metrics come after it, as PySpark's client raises from an
    /// execution at the first metric that it does not convert. Used by the Python client to raise
    /// PySpark's `UNSUPPORTED_LITERAL` for a metric of a kind that PySpark does not convert.
    pub fn set_observation_metrics_check(
        &self,
        check: impl Fn(&proto::execute_plan_response::ObservedMetrics) -> Result<()>
            + Send
            + Sync
            + 'static,
    ) {
        *self
            .inner
            .observation_metrics_check
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(check));
    }

    /// The check of the metrics of the Observations of the executions, if one is set.
    pub(crate) fn observation_metrics_check(&self) -> Option<ObservationMetricsCheck> {
        self.inner
            .observation_metrics_check
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Set the handler of the warnings of the ML fits of this session, and of its clones. The
    /// server sends a warning with the model of a fit (`warning_message`) when the training of a
    /// tree model stopped early because the model grew beyond the threshold of the memory control
    /// of its ML cache (of the session's
    /// `spark.connect.session.connectML.mlCache.memoryControl.maxModelSize`). The handler is
    /// called with it on the thread of the fit, once the result of the fit has arrived and before
    /// the model is made of it or any other request is sent, as PySpark's `try_remote_fit` logs
    /// it then: of each fit, also of those of the stages of a pipeline, of the classes of a
    /// OneVsRest and of a validator. An error that it returns is the error of the fit. Used by
    /// the Python client to log the warnings as PySpark does; the fitted model keeps its warning
    /// too, as [`Model::fit_warning`](crate::ml::Model::fit_warning) gives it.
    pub fn set_ml_fit_warning_handler(
        &self,
        handler: impl Fn(&str) -> Result<()> + Send + Sync + 'static,
    ) {
        *self
            .inner
            .ml_fit_warning_handler
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(handler));
    }

    /// Hand the warning of an ML fit to the handler of the session (see
    /// [`SparkSession::set_ml_fit_warning_handler`]), if one is set: its error is the fit's.
    pub(crate) fn report_ml_fit_warning(&self, warning: &str) -> Result<()> {
        // Called out of the lock, so that the handler may set another one.
        let handler = self
            .inner
            .ml_fit_warning_handler
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        handler.map_or(Ok(()), |handler| handler(warning))
    }

    /// Stop this Spark session.
    ///
    /// Idempotent, mirroring pyspark: the session is marked stopped even if releasing it
    /// on the server fails, and stopping an already stopped session is a no-op.
    ///
    /// As PySpark's `stop()`, which runs under `SparkSession._lock`, the `stop()`s of all the
    /// sessions of the process run one at a time: a `stop()` of another thread waits for the one
    /// in progress, whichever session it stops, and releases the session itself if that one was
    /// interrupted before it closed the session; a `stop()` of the thread that runs one (a signal
    /// handler of the Python client) runs within it. The session is stopped once its release has
    /// ended, as PySpark's `is_stopped` is the client's `is_closed`, which its `stop()` sets after
    /// the release.
    ///
    /// Release the server session first, then cancel local listener registration/dispatch and
    /// close the client even if release fails. ReleaseSession makes the server send the
    /// terminated events of the queries still running, which the listeners must still receive.
    /// As PySpark's stop, a client that is already closed, by a server-side session change, is
    /// not released. This does not make a separate listener-removal RPC: ReleaseSession
    /// releases the server's listeners too.
    /// Teardown does not wait for listener callbacks; the current event's listener snapshot
    /// may finish after stop returns, while no later event is dispatched.
    pub fn stop(&self) -> Result<()> {
        // A signal of the process ends the wait for another thread's `stop()` in the Python
        // client, as it interrupts the acquire of PySpark's lock.
        let Some(_stopping) = enter_stop_lock() else {
            return Err(SparkError::connect_msg(
                "the wait for the stop of the session on another thread was interrupted",
            ));
        };
        if self.inner.stopped.load(Ordering::SeqCst) {
            return Ok(());
        }
        let released = if self.client().is_closed() {
            Ok(())
        } else {
            let interrupted = || {
                SparkError::connect_msg(
                    "the release of the session was interrupted; the session is still open",
                )
            };
            match spark_connect_core::runtime::block_on_abortable(self.client().release_session()) {
                Some(Err(error)) if spark_connect_core::client::is_wait_aborted(&error) => {
                    return Err(interrupted());
                }
                Some(released) => released,
                // The wait was stopped, by a signal of the process in the Python client: as
                // PySpark's `stop()`, which the exception of the signal leaves before it closes
                // the client, the session stays open, and a later `stop()` releases it. The
                // release of an interrupted backoff gives the same error, which leaves it so.
                None => return Err(interrupted()),
            }
        };
        // As the `close` of PySpark's `stop()`, which waits for the releases of the executions
        // of the client before it closes the channel (see `SparkSession::close`): a signal of the
        // process that ends the wait leaves the session open, as the exception of the signal
        // leaves PySpark's `close`.
        if spark_connect_core::runtime::block_on_abortable(self.client().wait_for_releases())
            .is_none()
        {
            return Err(SparkError::connect_msg(
                "the wait for the releases of the executions of the session was interrupted; \
                 the session is still open",
            ));
        }
        // Before the listener bus is cancelled: an add of a listener that the cancellation ends
        // finds the session stopped.
        self.inner.stopped.store(true, Ordering::SeqCst);
        self.streams().cancel_listener_bus();
        // As PySpark's `stop()`, close the client even if server cleanup fails, so its
        // future requests raise `NO_ACTIVE_SESSION`.
        self.client().close();
        released
    }

    /// Release the session on the server, as `SparkConnectClient.release_session`: its state
    /// there, without stopping this session. After [`SparkSession::stop`] or
    /// [`SparkSession::close`], it sends nothing and is the `NO_ACTIVE_SESSION` error, as the
    /// `_stub` of PySpark's closed client raises it.
    pub fn release_session(&self) -> Result<()> {
        if self.client().is_closed() {
            return Err(SparkError::classed(
                spark_connect_core::error::SparkErrorKind::Connect,
                "NO_ACTIVE_SESSION",
                &[],
            ));
        }
        block_on(self.client().release_session())
    }

    /// Close this session without releasing it on the server, as `SparkConnectClient.close`
    /// closes its channel: it is stopped, and `stop()` does nothing more. Cancel local listener
    /// registration/dispatch before closing the client. Callbacks are not joined; the current
    /// event's listener snapshot may finish after close returns.
    ///
    /// As PySpark's `close`, it first waits for the releases in the background of the executions
    /// whose results were read, at most 10 s (see
    /// [`spark_connect_core::client::SparkConnectClient::wait_for_releases`]), which the close of
    /// the channel would cancel.
    pub fn close(&self) {
        let _ = spark_connect_core::runtime::block_on_abortable(self.client().wait_for_releases());
        self.close_at_once();
    }

    /// As [`SparkSession::close`], of a wait for the releases that a signal of the process ends
    /// in the Python client: the session is then left open, and None, as the exception of the
    /// signal leaves PySpark's `close` before it closes the channel. Not a stable API.
    #[doc(hidden)]
    pub fn close_after_releases(&self) -> Option<()> {
        spark_connect_core::runtime::block_on_abortable(self.client().wait_for_releases())?;
        self.close_at_once();
        Some(())
    }

    /// Whether the close of the session currently has releases of its executions to wait for (see
    /// [`SparkSession::close`]). Not a stable API.
    #[doc(hidden)]
    pub fn has_releases_to_wait_for(&self) -> bool {
        self.client().has_releases_to_wait_for()
    }

    /// As [`SparkSession::close`], at once, of no wait for the releases of its executions, which
    /// the close of the channel cancels if they are in flight. Not a stable API.
    #[doc(hidden)]
    pub fn close_at_once(&self) {
        self.inner.stopped.store(true, Ordering::SeqCst);
        self.streams().cancel_listener_bus();
        self.client().close();
    }

    /// Whether this session was stopped by `stop()`, once its release ended, or by `close()`, or
    /// its client is closed, as after a server-side session change. Mirrors
    /// `pyspark.sql.connect.session.SparkSession.is_stopped`, which is the client's `is_closed`:
    /// a session whose release is in flight is not stopped yet.
    pub fn is_stopped(&self) -> bool {
        self.inner.stopped.load(Ordering::SeqCst) || self.client().is_closed()
    }
}

/// The name of the artifact that copies a file to `dest_path`, as the reference's
/// `FORWARD_TO_FS_PREFIX + dest_path`: the prefix is followed by the absolute path itself.
fn forward_to_fs_artifact_name(dest_path: &str) -> String {
    format!("forward_to_fs{}", dest_path)
}

/// A handle of a session that does not keep it alive, of [`SparkSession::downgrade`], as a
/// `std::sync::Weak` is of an `Arc`. [`WeakSparkSession::upgrade`] gives the session back while a
/// handle of it is alive, the session itself, a clone, or one that an object of the session
/// holds, such as a DataFrame, a reader, a writer, a catalog or a stream, and None once they are
/// all dropped. The handle keeps none of the state of the session: neither its client, which is
/// dropped with the last handle, nor the handlers, the listeners and the hook that the session
/// holds, which may hold objects that hold the session. Not a stable API.
#[doc(hidden)]
#[derive(Clone)]
pub struct WeakSparkSession {
    inner: Weak<SessionInner>,
}

impl WeakSparkSession {
    /// The session, while a handle of it is alive, or None once they are all dropped. Each
    /// handle holds all of the state of the session, so the state is alive as long as one is.
    pub fn upgrade(&self) -> Option<SparkSession> {
        Some(SparkSession {
            inner: self.inner.upgrade()?,
        })
    }
}

/// The configuration of the client of a session that a [`SparkSessionBuilder`] builds.
type ClientConfiguration = Box<dyn FnOnce(&mut SparkConnectClient) + Send>;

/// Builder for creating a SparkSession, as PySpark's `SparkSession.builder` of Spark Connect.
///
/// ```no_run
/// use spark_connect::SparkSession;
///
/// let spark = SparkSession::builder()
///     .remote("sc://localhost:15002")
///     .app_name("my-app")
///     .config("spark.sql.shuffle.partitions", 20)
///     .configs([("spark.sql.ansi.enabled", "true")])
///     .get_or_create()?;
/// # Ok::<(), spark_connect::SparkError>(())
/// ```
pub struct SparkSessionBuilder {
    /// The options of the builder, in the order that they were first given, of their last values,
    /// as PySpark's `_options` dict: `spark.remote` (see [`SparkSessionBuilder::remote`]) and the
    /// configurations that the session gets once it connects (see
    /// [`SparkSessionBuilder::config`]).
    options: Vec<(String, String)>,
    /// The configuration of the client of the session (see
    /// [`SparkSessionBuilder::configure_client`]).
    configure: Option<ClientConfiguration>,
    /// The credentials of the channel (see [`SparkSessionBuilder::channel_credentials`]).
    credentials: Option<spark_connect_core::channel::Credentials>,
    /// The target of gRPC of the channel (see [`SparkSessionBuilder::channel_target`]).
    target: Option<spark_connect_core::channel::ChannelTarget>,
    /// gRPC's names of the target, if any (see [`SparkSessionBuilder::channel_target_names`]).
    target_names: Option<spark_connect_core::channel::TargetNames>,
    /// The metadata of the requests (see [`SparkSessionBuilder::channel_metadata`]).
    metadata: Option<Vec<(String, String)>>,
}

impl SparkSessionBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        SparkSessionBuilder {
            options: Vec::new(),
            configure: None,
            credentials: None,
            target: None,
            target_names: None,
            metadata: None,
        }
    }

    /// Give the channel of the session the credentials `credentials`, which decide its TLS and
    /// its token in place of the rule of the parameters of the connection string, as the Python
    /// binding gives them of a channel builder of PySpark's `toChannel` (see
    /// `ChannelBuilder::with_credentials`). Not a stable API.
    #[doc(hidden)]
    pub fn channel_credentials(
        mut self,
        credentials: spark_connect_core::channel::Credentials,
    ) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Connect the channel of the session to the target of gRPC `target` in place of the host and
    /// the port of the connection string, as the Python binding gives it of the `endpoint` of a
    /// channel builder (see `ChannelBuilder::with_target`). Not a stable API.
    #[doc(hidden)]
    pub fn channel_target(mut self, target: spark_connect_core::channel::ChannelTarget) -> Self {
        self.target = Some(target);
        self
    }

    /// Take `names`, gRPC's names of the target of the channel, for the `:authority` of its calls,
    /// the name that its TLS verifies and its proxy (see `ChannelBuilder::with_target_names`), as
    /// the Python binding gives them of the `endpoint` of a channel builder. Not a stable API.
    #[doc(hidden)]
    pub fn channel_target_names(mut self, names: spark_connect_core::channel::TargetNames) -> Self {
        self.target_names = Some(names);
        self
    }

    /// Send `metadata` with the requests of the session in place of the parameters of the
    /// connection string that are not the channel's own, as the Python binding gives it of the
    /// `metadata()` of a channel builder (see `ChannelBuilder::with_metadata`). Not a stable API.
    #[doc(hidden)]
    pub fn channel_metadata(mut self, metadata: Vec<(String, String)>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Configure the client of the session once it is connected, before the session is built
    /// of it, with the settings that PySpark's `SparkConnectClient` takes when it is made, such
    /// as its RPC deadlines and the result chunking of its executions. Not a stable API.
    #[doc(hidden)]
    pub fn configure_client(
        mut self,
        configure: impl FnOnce(&mut SparkConnectClient) + Send + 'static,
    ) -> Self {
        self.configure = Some(Box::new(configure));
        self
    }

    /// Set the remote Spark Connect server URL, the option `spark.remote`, as PySpark's
    /// `remote`: `config("spark.remote", url)` is the same.
    pub fn remote(self, url: &str) -> Self {
        self.config("spark.remote", url)
    }

    /// Set the option `key` to `value`, as PySpark's `config(key, value)`: the session gets it
    /// as a configuration once it connects (see [`SparkSessionBuilder::get_or_create`]), but
    /// `spark.remote`, the URL of the server, and `spark.master`, which Spark Connect does not
    /// take. A value given again of a key replaces the earlier one. A `bool` is `true` or
    /// `false`, as PySpark's `to_str` writes it.
    pub fn config(mut self, key: &str, value: impl ToString) -> Self {
        let value = value.to_string();
        match self.options.iter_mut().find(|(k, _)| k == key) {
            Some((_, v)) => *v = value,
            None => self.options.push((key.to_string(), value)),
        }
        self
    }

    /// Set the options of `pairs`, in their order, as PySpark's `config(map=...)` (see
    /// [`SparkSessionBuilder::config`]).
    pub fn configs<K: AsRef<str>, V: ToString>(
        mut self,
        pairs: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        for (key, value) in pairs {
            self = self.config(key.as_ref(), value);
        }
        self
    }

    /// Set the name of the application, the configuration `spark.app.name`, as PySpark's
    /// `appName`.
    pub fn app_name(self, name: &str) -> Self {
        self.config("spark.app.name", name)
    }

    /// Take the master URL of a Spark cluster, of no effect, as PySpark's `master` of Spark
    /// Connect: the server is the one of [`SparkSessionBuilder::remote`].
    pub fn master(self, _master: &str) -> Self {
        self
    }

    /// The configurations that a session of this builder gets once it connects, as PySpark's
    /// `_apply_options`: the options but `spark.remote` and `spark.master`.
    fn configurations(&self) -> Vec<(String, String)> {
        self.options
            .iter()
            .filter(|(key, _)| key != "spark.remote" && key != "spark.master")
            .cloned()
            .collect()
    }

    /// Build and return a SparkSession, of a new connection to the server of
    /// [`SparkSessionBuilder::remote`], which gets the configurations of the builder (see
    /// [`SparkSessionBuilder::config`]) in one silent `ConfigRequest`, as PySpark's
    /// `_apply_options`. The Rust API keeps no active or default session, so each call creates
    /// a session, as [`SparkSessionBuilder::create`] does, where PySpark's `getOrCreate` returns
    /// its active or default session if there is one.
    pub fn get_or_create(self) -> Result<SparkSession> {
        self.create()
    }

    /// Build and return a new SparkSession, as PySpark's `create` (see
    /// [`SparkSessionBuilder::get_or_create`]).
    pub fn create(self) -> Result<SparkSession> {
        let configurations = self.configurations();
        let url = self
            .options
            .iter()
            .find(|(key, _)| key == "spark.remote")
            .map(|(_, url)| url.clone())
            .ok_or_else(|| {
                SparkError::value(
                    "NO_REMOTE_URL",
                    &[("detail", "Must call .remote(url) before .get_or_create()")],
                )
            })?;

        let mut builder = ChannelBuilder::parse(&url)?;
        if let Some(credentials) = self.credentials {
            builder = builder.with_credentials(credentials);
        }
        if let Some(target) = self.target {
            builder = builder.with_target(target);
        }
        if let Some(names) = self.target_names {
            builder = builder.with_target_names(names);
        }
        if let Some(metadata) = self.metadata {
            builder = builder.with_metadata(metadata);
        }
        let mut client = block_on(SparkConnectClient::connect(&builder))?;
        if let Some(configure) = self.configure {
            configure(&mut client);
        }
        let session = SparkSession::new(client);
        if !configurations.is_empty() {
            session.conf().set_all(&configurations, true)?;
        }
        Ok(session)
    }
}

impl Default for SparkSessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The error for a row of `create_dataframe` whose width differs from the schema's. The
/// Python client checks widths itself, raising pyspark's `AXIS_LENGTH_MISMATCH`.
fn row_width_mismatch(row_index: usize, actual: usize, expected: usize) -> SparkError {
    SparkError::value_msg(format!(
        "Row {row_index} has {actual} values, but the schema has {expected} fields"
    ))
}

/// The size of a local relation of `createDataFrame`, in Arrow data, below which a relation under
/// `spark.sql.session.localRelationCacheThreshold` is sent inline in the IPC that encoded it: the
/// stream of the batch of its rows or columns, or the IPC that the binding gave. From it, the
/// relation is sent as the Arrow importer sends one, which writes the IPC of its batches again.
const INLINE_IPC_BYTES: usize = 64 << 10;

/// Internal binding bridge for encoding already converted local rows. The Python
/// binding also uses it for native columns beside imported pandas Arrow columns.
/// Not a stable public API.
#[doc(hidden)]
pub fn rows_to_arrow_batch(
    rows: &[Row],
    schema: &DataType,
) -> Result<arrow::record_batch::RecordBatch> {
    use arrow::datatypes::Schema as ArrowSchema;

    // Convert our schema to Arrow schema
    let arrow_schema_vec = schema_to_arrow_fields(schema)?;
    let arrow_schema = ArrowSchema::new(arrow_schema_vec);

    // Build column arrays from rows
    let mut columns = vec![];
    let fields = match schema {
        DataType::Struct { fields } => fields,
        _ => return Err(SparkError::connect_msg("Schema is not a struct type")),
    };

    // Conversion is positional, so every row must match the schema width.
    for (index, row) in rows.iter().enumerate() {
        if row.len() != fields.len() {
            return Err(row_width_mismatch(index, row.len(), fields.len()));
        }
    }

    for (field_idx, field) in fields.iter().enumerate() {
        let column: Vec<&Value> = rows.iter().map(|row| &row.values()[field_idx]).collect();
        columns.push(values_to_arrow(&column, &field.data_type)?);
    }

    columns_to_arrow_batch(columns, arrow_schema, rows.len())
}

/// The Arrow IPC stream of local rows, as the server reads a local relation.
#[cfg(test)]
fn rows_to_arrow_ipc(rows: &[Row], schema: &DataType) -> Result<Vec<u8>> {
    if rows.is_empty() {
        return Ok(vec![]);
    }
    batch_to_arrow_ipc(&rows_to_arrow_batch(rows, schema)?)
}

/// `columns`, of which a column of the Arrow type of its field of `arrow_schema` but the metadata
/// of the fields of the SQL type of a user-defined type, which [`arrow_type_of`] gives the type
/// and a caller's column need not, takes the type of its field. Another column is kept, so that a
/// column of another type fails as it did.
fn with_udt_field_metadata(
    columns: Vec<arrow::array::ArrayRef>,
    arrow_schema: &arrow::datatypes::Schema,
) -> Result<Vec<arrow::array::ArrayRef>> {
    columns
        .into_iter()
        .enumerate()
        .map(|(i, column)| match arrow_schema.fields().get(i) {
            Some(field)
                if column.data_type() != field.data_type()
                    && column.data_type().equals_datatype(field.data_type())
                    && field.data_type().contains(column.data_type()) =>
            {
                crate::arrow_input::with_schema_type(column.to_data(), field.data_type())
                    .map(arrow::array::make_array)
            }
            _ => Ok(column),
        })
        .collect()
}

fn columns_to_arrow_batch(
    columns: Vec<arrow::array::ArrayRef>,
    arrow_schema: arrow::datatypes::Schema,
    row_count: usize,
) -> Result<arrow::record_batch::RecordBatch> {
    use arrow::record_batch::RecordBatch;

    // Preserve the row count even for a batch without columns.
    let options = arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(row_count));
    RecordBatch::try_new_with_options(Arc::new(arrow_schema), columns, &options)
        .map_err(|e| SparkError::connect_msg(format!("Failed to create Arrow batch: {}", e)))
}

fn batch_to_arrow_ipc(batch: &arrow::record_batch::RecordBatch) -> Result<Vec<u8>> {
    use arrow::ipc::writer::StreamWriter;

    // Write to IPC stream
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &batch.schema())
            .map_err(|e| SparkError::connect_msg(format!("Failed to create IPC writer: {}", e)))?;
        writer
            .write(batch)
            .map_err(|e| SparkError::connect_msg(format!("Failed to write batch to IPC: {}", e)))?;
        writer
            .finish()
            .map_err(|e| SparkError::connect_msg(format!("Failed to finish IPC writer: {}", e)))?;
    }

    Ok(buffer)
}

/// Convert DataType fields to Arrow fields.
fn schema_to_arrow_fields(schema: &DataType) -> Result<Vec<arrow::datatypes::Field>> {
    use arrow::datatypes::Field;

    match schema {
        DataType::Struct { fields } => {
            let mut arrow_fields = vec![];
            for field in fields {
                arrow_fields.push(Field::new(
                    &field.name,
                    datatype_to_arrow(&field.data_type)?,
                    field.nullable,
                ));
            }
            Ok(arrow_fields)
        }
        _ => Err(SparkError::connect_msg("Schema is not a struct")),
    }
}

/// The names of nested struct fields as PySpark's `_dedup_names` gives them: a name that occurs
/// more than once is numbered per occurrence, as `name_0`, `name_1`, and others are kept.
fn dedup_field_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let names: Vec<&str> = names.into_iter().collect();
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for name in &names {
        *counts.entry(*name).or_insert(0) += 1;
    }
    let mut next: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    names
        .iter()
        .map(|&name| {
            if counts[name] > 1 {
                let i = next.entry(name).or_insert(0);
                let deduped = format!("{name}_{i}");
                *i += 1;
                deduped
            } else {
                name.to_string()
            }
        })
        .collect()
}

/// The precision and scale of an Arrow `Decimal128` for a Spark `decimal(precision, scale)`.
///
/// `DataType::Decimal` holds `i32`s and, as PySpark's `DecimalType`, is not validated when it is
/// built or parsed, so they are narrowed here with checks rather than `as` casts, which would
/// wrap `decimal(300,0)` to a precision of 44. Arrow's own limits apply, as they do in PySpark's
/// `to_arrow_type` (`pa.decimal128` raises for a precision outside 1 to 38).
fn decimal128_precision_and_scale(precision: i32, scale: i32) -> Result<(u8, i8)> {
    use arrow::datatypes::{
        validate_decimal_precision_and_scale, Decimal128Type, DECIMAL128_MAX_PRECISION,
    };
    let unsupported = |reason: String| {
        SparkError::connect_msg(format!(
            "unsupported decimal type decimal({precision},{scale}): {reason}"
        ))
    };
    let arrow_precision = u8::try_from(precision)
        .ok()
        .filter(|p| (1..=DECIMAL128_MAX_PRECISION).contains(p))
        .ok_or_else(|| {
            unsupported(format!(
                "precision must be between 1 and {DECIMAL128_MAX_PRECISION}"
            ))
        })?;
    let arrow_scale = i8::try_from(scale)
        .map_err(|_| unsupported(format!("scale must be between {} and {}", i8::MIN, i8::MAX)))?;
    validate_decimal_precision_and_scale::<Decimal128Type>(arrow_precision, arrow_scale)
        .map_err(|e| unsupported(e.to_string()))?;
    Ok((arrow_precision, arrow_scale))
}

/// Map a Spark `DataType` to its Arrow `DataType`, recursively (array/map/struct included).
fn datatype_to_arrow(dt: &DataType) -> Result<arrow::datatypes::DataType> {
    datatype_to_arrow_of(dt, false)
}

/// [`datatype_to_arrow`], with PySpark's large types if `large`, as its
/// `to_arrow_type(..., prefers_large_types=True)` makes them under
/// `spark.sql.execution.arrow.useLargeVarTypes`: a string or a binary, also in an array, a map, a
/// struct or the SQL type of a user-defined type, is `LargeUtf8` or `LargeBinary`, while the
/// binaries of a variant, a geometry and a geography stay `Binary`, as PySpark keeps them.
fn datatype_to_arrow_of(dt: &DataType, large: bool) -> Result<arrow::datatypes::DataType> {
    arrow_type_of(dt, large, false)
}

/// The Arrow field metadata of the metadata of a field in the SQL type of a user-defined type,
/// as PySpark's `to_arrow_type` writes it: `json.dumps` of it under `SPARK::metadata::json`, and
/// none of no metadata. PySpark's Arrow schema of local data keeps no metadata of the other
/// fields, as `_deduplicate_field_names` drops it, which leaves a user-defined type as it is.
fn udt_field_metadata(
    metadata: &crate::types::Metadata,
) -> std::collections::HashMap<String, String> {
    if metadata.is_empty() {
        return std::collections::HashMap::new();
    }
    std::collections::HashMap::from([("SPARK::metadata::json".to_string(), metadata.json())])
}

/// [`datatype_to_arrow_of`], whose struct fields keep their metadata ([`udt_field_metadata`]) if
/// `in_udt`, as in the SQL type of a user-defined type.
fn arrow_type_of(dt: &DataType, large: bool, in_udt: bool) -> Result<arrow::datatypes::DataType> {
    use arrow::datatypes::{DataType as A, Field, Fields, TimeUnit};
    use std::collections::HashMap;
    Ok(match dt {
        DataType::Null => A::Null,
        DataType::Boolean => A::Boolean,
        DataType::Byte => A::Int8,
        DataType::Short => A::Int16,
        DataType::Integer => A::Int32,
        DataType::Long => A::Int64,
        DataType::Float => A::Float32,
        DataType::Double => A::Float64,
        // CHAR/VARCHAR are string-backed on the wire.
        DataType::String { .. } | DataType::Char { .. } | DataType::Varchar { .. } if large => {
            A::LargeUtf8
        }
        DataType::String { .. } | DataType::Char { .. } | DataType::Varchar { .. } => A::Utf8,
        DataType::Binary if large => A::LargeBinary,
        DataType::Binary => A::Binary,
        DataType::Date => A::Date32,
        // TIMESTAMP (LTZ) and TIMESTAMP_NTZ are both micros; NTZ carries no zone.
        DataType::Timestamp => A::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        DataType::TimestampNtz => A::Timestamp(TimeUnit::Microsecond, None),
        DataType::TimestampLtzNanos { .. } => {
            A::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()))
        }
        DataType::TimestampNtzNanos { .. } => A::Timestamp(TimeUnit::Nanosecond, None),
        // As PySpark's `to_arrow_type` of TimeType.
        DataType::Time { .. } => A::Time64(TimeUnit::Nanosecond),
        // As Spark's Arrow conversion of DayTimeIntervalType.
        DataType::DayTimeInterval { .. } => A::Duration(TimeUnit::Microsecond),
        // As Spark's Arrow conversion of YearMonthIntervalType, the months of each interval.
        DataType::YearMonthInterval { .. } => {
            A::Interval(arrow::datatypes::IntervalUnit::YearMonth)
        }
        DataType::Decimal { precision, scale } => {
            let (precision, scale) = decimal128_precision_and_scale(*precision, *scale)?;
            A::Decimal128(precision, scale)
        }
        DataType::Array {
            element_type,
            contains_null,
        } => A::List(Arc::new(Field::new(
            "element",
            arrow_type_of(element_type, large, in_udt)?,
            *contains_null,
        ))),
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => {
            let entry_fields: Fields = vec![
                Field::new("key", arrow_type_of(key_type, large, in_udt)?, false),
                Field::new(
                    "value",
                    arrow_type_of(value_type, large, in_udt)?,
                    *value_contains_null,
                ),
            ]
            .into();
            A::Map(
                Arc::new(Field::new("entries", A::Struct(entry_fields), false)),
                false,
            )
        }
        // The names of nested fields are deduplicated, as PySpark's `_deduplicate_field_names`
        // does for the Arrow schema it sends. The server cannot read duplicate names, while the
        // schema of the relation keeps the names of the fields. In the SQL type of a user-defined
        // type, which `_deduplicate_field_names` leaves as it is, the fields keep their names,
        // also of duplicates, which the server then fails to read as PySpark's, and their
        // metadata, as PySpark's `to_arrow_type` writes it.
        DataType::Struct { fields } => {
            let names = if in_udt {
                fields.iter().map(|field| field.name.clone()).collect()
            } else {
                dedup_field_names(fields.iter().map(|field| field.name.as_str()))
            };
            let f: Fields = fields
                .iter()
                .zip(names)
                .map(|(field, name)| {
                    let arrow_field = Field::new(
                        name,
                        arrow_type_of(&field.data_type, large, in_udt)?,
                        field.nullable,
                    );
                    Ok(if in_udt {
                        arrow_field.with_metadata(udt_field_metadata(&field.metadata))
                    } else {
                        arrow_field
                    })
                })
                .collect::<Result<Vec<_>>>()?
                .into();
            A::Struct(f)
        }
        // As PySpark's `to_arrow_type`: the struct of the binary value and metadata of a
        // VARIANT, and of the srid and WKB of a GEOMETRY or a GEOGRAPHY, whose metadata tags
        // the struct as one of these types.
        DataType::Variant => A::Struct(
            vec![
                Field::new("value", A::Binary, false),
                Field::new("metadata", A::Binary, false)
                    .with_metadata(HashMap::from([("variant".to_string(), "true".to_string())])),
            ]
            .into(),
        ),
        DataType::Geometry { srid } | DataType::Geography { srid } => {
            let kind = if matches!(dt, DataType::Geometry { .. }) {
                "geometry"
            } else {
                "geography"
            };
            A::Struct(
                vec![
                    Field::new("srid", A::Int32, false),
                    Field::new("wkb", A::Binary, false).with_metadata(HashMap::from([
                        (kind.to_string(), "true".to_string()),
                        ("srid".to_string(), srid.to_string()),
                    ])),
                ]
                .into(),
            )
        }
        // As PySpark's, a user-defined type is sent as its SQL type, of the metadata of its fields.
        DataType::Udt {
            sql_type: Some(sql_type),
            ..
        } => arrow_type_of(sql_type, large, true)?,
        other => {
            return Err(SparkError::connect_msg(format!(
                "Unsupported type for Arrow createDataFrame conversion: {other:?}"
            )))
        }
    })
}

/// `batch`, the Arrow data of local rows or columns of the struct `schema`, as PySpark's
/// `createDataFrame` makes its table under `spark.sql.execution.arrow.useLargeVarTypes`, if
/// `large`: each column of a string or a binary, also nested, is cast to the large types of
/// [`datatype_to_arrow_of`]. The cast keeps the values buffers; it writes 64-bit offsets.
fn with_large_var_types(
    batch: arrow::record_batch::RecordBatch,
    schema: &DataType,
    large: bool,
) -> Result<arrow::record_batch::RecordBatch> {
    use arrow::record_batch::{RecordBatch, RecordBatchOptions};
    let DataType::Struct { fields } = schema else {
        return Ok(batch);
    };
    if !large || fields.len() != batch.num_columns() {
        return Ok(batch);
    }
    let arrow_schema = batch.schema();
    let mut arrow_fields = Vec::with_capacity(fields.len());
    let mut columns = Vec::with_capacity(fields.len());
    for ((field, arrow_field), column) in fields
        .iter()
        .zip(arrow_schema.fields())
        .zip(batch.columns())
    {
        let target = datatype_to_arrow_of(&field.data_type, true)?;
        if &target == arrow_field.data_type() {
            arrow_fields.push(Arc::clone(arrow_field));
            columns.push(Arc::clone(column));
        } else {
            columns.push(arrow::compute::cast(column, &target).map_err(|e| {
                SparkError::connect_msg(format!("Failed to cast to large Arrow types: {e}"))
            })?);
            arrow_fields.push(Arc::new(
                arrow_field.as_ref().clone().with_data_type(target),
            ));
        }
    }
    let schema =
        arrow::datatypes::Schema::new_with_metadata(arrow_fields, arrow_schema.metadata().clone());
    RecordBatch::try_new_with_options(
        Arc::new(schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )
    .map_err(|e| SparkError::connect_msg(format!("Failed to build Arrow batch: {e}")))
}

/// Whether `dt` has a string or a binary that [`datatype_to_arrow_of`] makes large.
fn has_var_types(dt: &DataType) -> bool {
    match dt {
        DataType::String { .. }
        | DataType::Char { .. }
        | DataType::Varchar { .. }
        | DataType::Binary => true,
        DataType::Array { element_type, .. } => has_var_types(element_type),
        DataType::Map {
            key_type,
            value_type,
            ..
        } => has_var_types(key_type) || has_var_types(value_type),
        DataType::Struct { fields } => fields.iter().any(|field| has_var_types(&field.data_type)),
        DataType::Udt {
            sql_type: Some(sql_type),
            ..
        } => has_var_types(sql_type),
        _ => false,
    }
}

/// Recursively build an Arrow array from a column of `Value`s against a declared
/// `DataType`, covering scalars and the nested array/map/struct types. Temporary
/// columns borrow their values; building the Arrow output still copies payloads.
fn values_to_arrow(values: &[&Value], dt: &DataType) -> Result<Arc<dyn arrow::array::Array>> {
    values_to_arrow_of(values, dt, false)
}

/// [`values_to_arrow`], of the Arrow types of [`arrow_type_of`] if `in_udt`, in the SQL type of a
/// user-defined type.
fn values_to_arrow_of(
    values: &[&Value],
    dt: &DataType,
    in_udt: bool,
) -> Result<Arc<dyn arrow::array::Array>> {
    use arrow::array::*;
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::{Field, Fields};

    macro_rules! prim {
        ($arr:ty, $pat:path) => {{
            let vals: Result<Vec<_>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    $pat(x) => Ok(Some(*x)),
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(<$arr>::from(vals?)) as Arc<dyn Array>)
        }};
    }

    // Numeric encoders coerce across integer/float widths, so a value the schema-unaware
    // Python->Value conversion produced as a wider/narrower numeric (e.g. a nested-struct
    // int arriving as Long) still matches its declared field type. An integer that does
    // not fit the declared type is an error, as PySpark's Arrow conversion raises for it.
    macro_rules! int_prim {
        ($arr:ty, $t:ty, $sql:expr) => {{
            let narrow = |x: i64| {
                <$t>::try_from(x).map(Some).map_err(|_| {
                    SparkError::value_msg(format!(
                        "Value {x} does not fit the {} type of its column, whose range is {}..={}",
                        $sql,
                        <$t>::MIN,
                        <$t>::MAX
                    ))
                })
            };
            let vals: Result<Vec<_>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    Value::Byte(x) => narrow(*x as i64),
                    Value::Short(x) => narrow(*x as i64),
                    Value::Integer(x) => narrow(*x as i64),
                    Value::Long(x) => narrow(*x),
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(<$arr>::from(vals?)) as Arc<dyn Array>)
        }};
    }
    macro_rules! float_prim {
        ($arr:ty, $t:ty) => {{
            let vals: Result<Vec<_>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    Value::Byte(x) => Ok(Some(*x as $t)),
                    Value::Short(x) => Ok(Some(*x as $t)),
                    Value::Float(x) => Ok(Some(*x as $t)),
                    Value::Double(x) => Ok(Some(*x as $t)),
                    Value::Integer(x) => Ok(Some(*x as $t)),
                    Value::Long(x) => Ok(Some(*x as $t)),
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(<$arr>::from(vals?)) as Arc<dyn Array>)
        }};
    }

    match dt {
        DataType::Null => Ok(Arc::new(NullArray::new(values.len()))),
        DataType::Boolean => prim!(BooleanArray, Value::Bool),
        DataType::Byte => int_prim!(Int8Array, i8, "TINYINT"),
        DataType::Short => int_prim!(Int16Array, i16, "SMALLINT"),
        DataType::Integer => int_prim!(Int32Array, i32, "INT"),
        DataType::Long => int_prim!(Int64Array, i64, "BIGINT"),
        DataType::Float => float_prim!(Float32Array, f32),
        DataType::Double => float_prim!(Float64Array, f64),
        DataType::String { .. } | DataType::Char { .. } | DataType::Varchar { .. } => {
            let vals: Result<Vec<Option<&str>>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    Value::String(s) => Ok(Some(s.as_str())),
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(StringArray::from(vals?)))
        }
        DataType::Binary => {
            let vals: Result<Vec<Option<&[u8]>>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    Value::Binary(b) => Ok(Some(b.as_slice())),
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(BinaryArray::from(vals?)))
        }
        DataType::Date => prim!(Date32Array, Value::Date),
        // As PySpark's `to_arrow_type`: nanoseconds since midnight, as `TimeType.toInternal`.
        DataType::Time { .. } => prim!(Time64NanosecondArray, Value::Time),
        DataType::DayTimeInterval { .. } => prim!(DurationMicrosecondArray, Value::DayTimeInterval),
        DataType::YearMonthInterval { .. } => {
            prim!(IntervalYearMonthArray, Value::YearMonthInterval)
        }
        DataType::Timestamp | DataType::TimestampNtz => {
            let vals: Result<Vec<Option<i64>>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    Value::Timestamp(t) | Value::TimestampNtz(t) => Ok(Some(*t)),
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            let arr = TimestampMicrosecondArray::from(vals?);
            let arr = if matches!(dt, DataType::TimestampNtz) {
                arr
            } else {
                arr.with_timezone("UTC")
            };
            Ok(Arc::new(arr))
        }
        // Row values are microseconds (collect truncates nanosecond columns to micros), so
        // scale them up to the nanosecond unit `to_arrow_type` declares for these types.
        DataType::TimestampNtzNanos { .. } | DataType::TimestampLtzNanos { .. } => {
            let vals: Result<Vec<Option<i64>>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    Value::Timestamp(t) | Value::TimestampNtz(t) => {
                        t.checked_mul(1_000).map(Some).ok_or_else(|| {
                            SparkError::connect_msg(
                                "Timestamp out of range for nanosecond precision",
                            )
                        })
                    }
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            let arr = TimestampNanosecondArray::from(vals?);
            let arr = if matches!(dt, DataType::TimestampNtzNanos { .. }) {
                arr
            } else {
                arr.with_timezone("UTC")
            };
            Ok(Arc::new(arr))
        }
        DataType::Decimal { precision, scale } => {
            let col_scale = *scale;
            // Checked before the values, so a column of nulls (or no rows) is validated too.
            let (arrow_precision, arrow_scale) =
                decimal128_precision_and_scale(*precision, col_scale)?;
            let vals: Result<Vec<Option<i128>>> = values
                .iter()
                .map(|v| match *v {
                    Value::Null => Ok(None),
                    Value::Decimal { value, .. } => {
                        decimal_str_to_unscaled(value, *precision, col_scale).map(Some)
                    }
                    _ => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            // Use the DECLARED precision/scale so the array matches the Arrow schema field
            // (Decimal128(precision, scale)); hardcoding 38 mismatched a DecimalType(10, 2).
            let arr = Decimal128Array::from(vals?)
                .with_precision_and_scale(arrow_precision, arrow_scale)
                .map_err(|e| SparkError::connect_msg(format!("decimal build: {e}")))?;
            Ok(Arc::new(arr))
        }
        DataType::Array {
            element_type,
            contains_null,
        } => {
            let mut child: Vec<&Value> = Vec::new();
            let mut offsets: Vec<i32> = vec![0];
            let mut valid = arrow::array::builder::BooleanBufferBuilder::new(values.len());
            for v in values {
                match *v {
                    Value::Null => {
                        valid.append(false);
                        offsets.push(*offsets.last().unwrap());
                    }
                    Value::List(items) => {
                        valid.append(true);
                        child.extend(items.iter());
                        offsets.push(child.len() as i32);
                    }
                    _ => return Err(SparkError::connect_msg("Type mismatch: expected array")),
                }
            }
            let child_arr = values_to_arrow_of(&child, element_type, in_udt)?;
            let field = Arc::new(Field::new(
                "element",
                arrow_type_of(element_type, false, in_udt)?,
                *contains_null,
            ));
            // A non-nullable element that has nulls is an error, not a panic, as for a struct.
            ListArray::try_new(
                field,
                OffsetBuffer::new(offsets.into()),
                child_arr,
                Some(NullBuffer::new(valid.finish())),
            )
            .map(|array| Arc::new(array) as Arc<dyn Array>)
            .map_err(|e| SparkError::connect_msg(format!("list build: {e}")))
        }
        // The struct of a VARIANT's value and metadata, and of a GEOMETRY's or a GEOGRAPHY's
        // srid and WKB, as `datatype_to_arrow` declares; their fields are not nullable, so a
        // null value has empty ones.
        DataType::Variant | DataType::Geometry { .. } | DataType::Geography { .. } => {
            let mut first = Vec::with_capacity(values.len());
            let mut second: Vec<&[u8]> = Vec::with_capacity(values.len());
            let mut valid = arrow::array::builder::BooleanBufferBuilder::new(values.len());
            for v in values {
                match (*v, dt) {
                    (Value::Null, _) => {
                        valid.append(false);
                        first.push(None);
                        second.push(&[]);
                    }
                    (Value::Variant { value, metadata }, DataType::Variant) => {
                        valid.append(true);
                        first.push(Some(value.as_slice()));
                        second.push(metadata.as_slice());
                    }
                    (Value::Struct(sf), DataType::Geometry { .. } | DataType::Geography { .. }) => {
                        let (Some(Value::Integer(_)), Some(Value::Binary(wkb))) =
                            (sf.get_by_name("srid"), sf.get_by_name("wkb"))
                        else {
                            return Err(SparkError::connect_msg("Type mismatch in row data"));
                        };
                        valid.append(true);
                        first.push(None);
                        second.push(wkb.as_slice());
                    }
                    _ => return Err(SparkError::connect_msg("Type mismatch in row data")),
                }
            }
            let arrow::datatypes::DataType::Struct(afields) = datatype_to_arrow(dt)? else {
                unreachable!("these types are structs")
            };
            let nulls = Some(NullBuffer::new(valid.finish()));
            let arrays: Vec<Arc<dyn Array>> = match dt {
                DataType::Variant => vec![
                    Arc::new(BinaryArray::from(
                        first
                            .iter()
                            .map(|v| v.unwrap_or(&[]))
                            .collect::<Vec<&[u8]>>(),
                    )),
                    Arc::new(BinaryArray::from(second)),
                ],
                _ => {
                    let srids: Vec<i32> = values
                        .iter()
                        .map(|v| match *v {
                            Value::Struct(sf) => match sf.get_by_name("srid") {
                                Some(Value::Integer(srid)) => *srid,
                                _ => 0,
                            },
                            _ => 0,
                        })
                        .collect();
                    vec![
                        Arc::new(Int32Array::from(srids)),
                        Arc::new(BinaryArray::from(second)),
                    ]
                }
            };
            Ok(Arc::new(
                StructArray::try_new(afields, arrays, nulls).map_err(|e| {
                    SparkError::connect_msg(format!("Arrow struct build error: {e}"))
                })?,
            ))
        }
        DataType::Udt {
            sql_type: Some(sql_type),
            ..
        } => values_to_arrow_of(values, sql_type, true),
        DataType::Struct { fields } => {
            let mut cols: Vec<Vec<&Value>> = vec![Vec::with_capacity(values.len()); fields.len()];
            let mut valid = arrow::array::builder::BooleanBufferBuilder::new(values.len());
            for v in values {
                match *v {
                    Value::Null => {
                        valid.append(false);
                        for c in cols.iter_mut() {
                            c.push(&Value::Null);
                        }
                    }
                    Value::Struct(sf) => {
                        valid.append(true);
                        // A struct of the declared fields in order, as the Python client
                        // converts one, is taken by position, so that fields of the same
                        // name keep their own values. Otherwise match by field name, else
                        // by position.
                        let in_order = sf.len() == fields.len()
                            && sf.fields().iter().zip(fields).all(|(n, f)| *n == f.name);
                        for (i, f) in fields.iter().enumerate() {
                            let val = if in_order {
                                sf.get(i)
                            } else {
                                sf.get_by_name(&f.name).or_else(|| sf.get(i))
                            };
                            cols[i].push(val.unwrap_or(&Value::Null));
                        }
                    }
                    // A Python dict supplied for a struct field (e.g. array<struct> data)
                    // arrives as a Map; match its entries to the struct fields by name.
                    Value::Map(m) => {
                        valid.append(true);
                        for (i, f) in fields.iter().enumerate() {
                            let value = m.iter().find_map(|(key, value)| match key {
                                Value::String(key) if *key == f.name => Some(value),
                                _ => None,
                            });
                            cols[i].push(value.unwrap_or(&Value::Null));
                        }
                    }
                    _ => return Err(SparkError::connect_msg("Type mismatch: expected struct")),
                }
            }
            let arrays: Vec<Arc<dyn Array>> = fields
                .iter()
                .enumerate()
                .map(|(i, f)| values_to_arrow_of(&cols[i], &f.data_type, in_udt))
                .collect::<Result<_>>()?;
            // The Arrow fields of the struct, with the deduplicated names of `datatype_to_arrow`,
            // and the metadata in a user-defined type, which the column's type has.
            let arrow::datatypes::DataType::Struct(afields) = arrow_type_of(dt, false, in_udt)?
            else {
                unreachable!("a struct is a struct")
            };
            // A struct without fields takes its length from the values.
            StructArray::try_new_with_length(
                afields,
                arrays,
                Some(NullBuffer::new(valid.finish())),
                values.len(),
            )
            .map(|array| Arc::new(array) as Arc<dyn Array>)
            .map_err(|e| SparkError::connect_msg(format!("struct build: {e}")))
        }
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => {
            let mut keys: Vec<&Value> = Vec::new();
            let mut vals: Vec<&Value> = Vec::new();
            let mut offsets: Vec<i32> = vec![0];
            let mut valid = arrow::array::builder::BooleanBufferBuilder::new(values.len());
            for v in values {
                match *v {
                    Value::Null => {
                        valid.append(false);
                        offsets.push(*offsets.last().unwrap());
                    }
                    Value::Map(m) => {
                        valid.append(true);
                        for (k, val) in m {
                            if matches!(k, Value::Null) {
                                return Err(SparkError::connect_msg("A map key cannot be null"));
                            }
                            keys.push(k);
                            vals.push(val);
                        }
                        offsets.push(keys.len() as i32);
                    }
                    _ => return Err(SparkError::connect_msg("Type mismatch: expected map")),
                }
            }
            let key_arr = values_to_arrow_of(&keys, key_type, in_udt)?;
            let val_arr = values_to_arrow_of(&vals, value_type, in_udt)?;
            let entry_fields: Fields = vec![
                Field::new("key", arrow_type_of(key_type, false, in_udt)?, false),
                Field::new(
                    "value",
                    arrow_type_of(value_type, false, in_udt)?,
                    *value_contains_null,
                ),
            ]
            .into();
            let entries = StructArray::try_new(entry_fields.clone(), vec![key_arr, val_arr], None)
                .map_err(|e| SparkError::connect_msg(format!("map build: {e}")))?;
            let entries_field = Arc::new(Field::new(
                "entries",
                arrow::datatypes::DataType::Struct(entry_fields),
                false,
            ));
            MapArray::try_new(
                entries_field,
                OffsetBuffer::new(offsets.into()),
                entries,
                Some(NullBuffer::new(valid.finish())),
                false,
            )
            .map(|array| Arc::new(array) as Arc<dyn Array>)
            .map_err(|e| SparkError::connect_msg(format!("map build: {e}")))
        }
        other => Err(SparkError::connect_msg(format!(
            "Unsupported type for Arrow createDataFrame conversion: {other:?}"
        ))),
    }
}

/// Converts a decimal string, in fixed-point (`-1.50`) or scientific (`1.5E-7`) notation, to
/// its unscaled `i128` at `scale` (`-1.50` at scale 2 is -150). The value must fit
/// `decimal(precision, scale)` exactly, as Arrow requires of the Decimals of PySpark's pandas
/// conversion: a nonzero digit beyond the scale, or a value beyond the precision, is an error
/// rather than truncated. The work is bounded by the length of the string, never by its
/// exponent, so `1E+1000000000` fails at once.
pub fn decimal_str_to_unscaled(s: &str, precision: i32, scale: i32) -> Result<i128> {
    if !(1..=38).contains(&precision) || !(0..=precision).contains(&scale) {
        return Err(SparkError::connect_msg(format!(
            "unsupported decimal type decimal({precision},{scale})"
        )));
    }
    let invalid = || SparkError::connect_msg(format!("invalid decimal '{s}'"));
    let inexact = || {
        SparkError::connect_msg(format!(
            "decimal '{s}' does not fit decimal({precision},{scale}) exactly"
        ))
    };
    let text = s.trim();
    let (negative, unsigned) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let (coefficient, exponent) = match unsigned.split_once(['e', 'E']) {
        Some((coefficient, exponent)) => {
            (coefficient, exponent.parse::<i64>().map_err(|_| invalid())?)
        }
        None => (unsigned, 0),
    };
    let mut point = false;
    let mut fractional = 0_i128;
    let mut digits = 0_usize;
    for byte in coefficient.bytes() {
        match byte {
            b'.' if !point => point = true,
            b'0'..=b'9' => {
                digits += 1;
                if point {
                    fractional += 1;
                }
            }
            _ => return Err(invalid()),
        }
    }
    if digits == 0 {
        return Err(invalid());
    }
    // The digits from the first nonzero one: the value is them times 10^(exponent -
    // fractional), and the unscaled value them times 10^shift.
    let significant: Vec<u8> = coefficient
        .bytes()
        .filter(|b| *b != b'.')
        .skip_while(|b| *b == b'0')
        .collect();
    // i128 arithmetic, so that i64::MIN and i64::MAX exponents cannot overflow.
    let shift = i128::from(scale) + i128::from(exponent) - fractional;
    let dropped = if shift < 0 {
        (-shift).min(significant.len() as i128) as usize
    } else {
        0
    };
    let (kept, lost) = significant.split_at(significant.len() - dropped);
    if lost.iter().any(|b| *b != b'0') {
        return Err(inexact());
    }
    if kept.is_empty() {
        return Ok(0);
    }
    if kept.len() as i128 + shift.max(0) > i128::from(precision) {
        return Err(inexact());
    }
    // At most 38 digits: no overflow.
    let mut value = kept
        .iter()
        .fold(0_i128, |value, digit| value * 10 + i128::from(digit - b'0'));
    for _ in 0..shift.max(0) {
        value *= 10;
    }
    Ok(if negative { -value } else { value })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        BinaryArray, BooleanArray, Float32Array, Int16Array, Int32Array, StringArray, StructArray,
    };
    use arrow::ipc::reader::StreamReader;
    use arrow::record_batch::RecordBatch;
    use std::io::Cursor;

    #[test]
    fn decimal_strings_convert_in_fixed_point_and_scientific_notation() {
        for (text, precision, scale, expected) in [
            ("1E+3", 18, 7, 10_000_000_000),
            ("1E-7", 18, 7, 1),
            ("-1E-7", 18, 7, -1),
            ("+1.25e+2", 38, 2, 12_500),
            ("-1.25E-2", 38, 4, -125),
            ("0001.2500e+2", 38, 2, 12_500),
            ("0E+999999999", 18, 7, 0),
            ("-0E-999999999", 18, 7, 0),
            ("123.45", 38, 2, 12_345),
            ("-123.450", 38, 2, -12_345),
            (".00100E+3", 38, 2, 100),
            ("  +1.e2  ", 38, 0, 100),
            // Trailing zeros beyond the scale lose nothing.
            ("1.2300E+0", 3, 2, 123),
            ("1.2300", 3, 2, 123),
            ("9.9E+1", 2, 0, 99),
            ("99999999999.9999999", 18, 7, 999_999_999_999_999_999),
            ("-99999999999.9999999", 18, 7, -999_999_999_999_999_999),
            ("1E-38", 38, 38, 1),
            // A long coefficient may still fit once it is scaled.
            ("100000000000000000000000000000000000000000E-41", 38, 0, 1),
        ] {
            assert_eq!(
                decimal_str_to_unscaled(text, precision, scale).unwrap(),
                expected,
                "{text}"
            );
        }
        let max = "9".repeat(38);
        assert_eq!(
            decimal_str_to_unscaled(&max, 38, 0).unwrap(),
            max.parse::<i128>().unwrap()
        );
        assert_eq!(
            decimal_str_to_unscaled(&format!("-{max}"), 38, 0).unwrap(),
            -max.parse::<i128>().unwrap()
        );
    }

    #[test]
    fn decimal_strings_that_do_not_fit_exactly_are_errors() {
        for (text, precision, scale) in [
            // A nonzero digit beyond the scale, in either notation: PySpark's Arrow
            // conversion of a pandas column fails rather than dropping it.
            ("1.5E-7", 18, 7),
            ("-1.5E-7", 18, 7),
            ("1E-8", 18, 7),
            ("0.00000015", 18, 7),
            ("1.235", 10, 2),
            ("1E-9223372036854775808", 18, 7),
            // Beyond the precision.
            ("123456E-3", 9, 7),
            ("1E+2", 2, 0),
            ("-1E+2", 2, 0),
            ("100", 2, 0),
            ("100000000000", 18, 7),
            ("1E+38", 38, 0),
            ("1E+999999999", 38, 0),
            ("1E+9223372036854775807", 38, 0),
            ("170141183460469231731687303715884105727", 38, 0),
        ] {
            let error = decimal_str_to_unscaled(text, precision, scale).unwrap_err();
            assert!(
                error.to_string().contains("does not fit"),
                "{text}: {error}"
            );
        }
        for text in [
            "",
            ".",
            "-",
            "+",
            "1E",
            "E3",
            "1E1E1",
            "1.2.3",
            "1E1.5",
            "NaN",
            "Infinity",
            "1E 3",
            "1_000",
            "1E+9223372036854775808",
        ] {
            let error = decimal_str_to_unscaled(text, 38, 0).unwrap_err();
            assert!(
                error.to_string().contains("invalid decimal"),
                "{text}: {error}"
            );
        }
        // Spark has no such decimal types.
        for (precision, scale) in [(0, 0), (39, 0), (10, -2), (10, 11), (38, i32::MAX)] {
            assert!(decimal_str_to_unscaled("1", precision, scale).is_err());
        }
    }

    #[test]
    fn scientific_decimal_ipc_cells_match_fixed_point() {
        use arrow::array::Decimal128Array;
        let values = [
            "1E+3",
            "1E-7",
            "-1E-7",
            "-0E+3",
            "99999999999.9999999",
            "-99999999999.9999999",
        ];
        let rows: Vec<_> = values
            .iter()
            .map(|value| {
                row(vec![Value::Decimal {
                    value: value.to_string(),
                    precision: None,
                    scale: None,
                }])
            })
            .collect();
        let batch = read_batch(
            &rows,
            &schema(vec![DataType::Decimal {
                precision: 18,
                scale: 7,
            }]),
        );
        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap();
        assert_eq!(
            array.values().as_ref(),
            &[
                10_000_000_000,
                1,
                -1,
                0,
                999_999_999_999_999_999,
                -999_999_999_999_999_999
            ]
        );
    }

    fn decimal(precision: i32, scale: i32) -> DataType {
        DataType::Decimal { precision, scale }
    }

    #[test]
    fn decimal_arrow_type_rejects_an_out_of_range_precision_or_scale() {
        // `as u8` / `as i8` used to wrap these: decimal(300,0) to a precision of 44,
        // decimal(257,0) to the valid Decimal128(1, 0) and decimal(38,256) to a scale of 0.
        for (precision, scale, reason) in [
            (300, 0, "precision must be between 1 and 38"),
            (257, 0, "precision must be between 1 and 38"),
            (39, 0, "precision must be between 1 and 38"),
            (0, 0, "precision must be between 1 and 38"),
            (-1, 0, "precision must be between 1 and 38"),
            (38, 256, "scale must be between -128 and 127"),
            (38, 128, "scale must be between -128 and 127"),
            (38, -129, "scale must be between -128 and 127"),
            (5, 6, "scale 6 is greater than precision 5"),
            (38, 39, "scale 39 is greater than max 38"),
        ] {
            let expected = format!("unsupported decimal type decimal({precision},{scale})");
            let dt = decimal(precision, scale);

            let error = datatype_to_arrow(&dt).unwrap_err().to_string();
            assert!(error.contains(&expected), "{error}");
            assert!(error.contains(reason), "{error}");

            // The schema of a nested type is checked as well.
            let nested = DataType::Array {
                element_type: Box::new(dt.clone()),
                contains_null: true,
            };
            let error = datatype_to_arrow(&nested).unwrap_err().to_string();
            assert!(error.contains(&expected), "{error}");

            // A column of nulls has no value to fail on, so the type itself must.
            let error = values_to_arrow(&[&Value::Null], &dt)
                .unwrap_err()
                .to_string();
            assert!(error.contains(&expected), "{error}");
            assert!(error.contains(reason), "{error}");

            let error = rows_to_arrow_ipc(&[row(vec![Value::Null])], &schema(vec![dt]))
                .unwrap_err()
                .to_string();
            assert!(error.contains(&expected), "{error}");
        }
    }

    #[test]
    fn decimal_arrow_type_keeps_the_in_range_boundaries() {
        use arrow::datatypes::DataType as A;
        for (precision, scale) in [(1, 0), (1, 1), (38, 0), (38, 38), (10, 2)] {
            let dt = decimal(precision, scale);
            let expected = A::Decimal128(precision as u8, scale as i8);
            assert_eq!(datatype_to_arrow(&dt).unwrap(), expected);

            let zero = Value::Decimal {
                value: "0".to_string(),
                precision: None,
                scale: None,
            };
            let array = values_to_arrow(&[&Value::Null, &zero], &dt).unwrap();
            assert_eq!(array.data_type(), &expected);
            assert_eq!(array.len(), 2);
            assert!(array.is_null(0));
        }
        // Arrow represents a negative scale, so the type is not refused here; whether Spark
        // takes it is the server's call (`spark.sql.legacy.allowNegativeScaleOfDecimal`).
        assert_eq!(
            datatype_to_arrow(&decimal(10, -2)).unwrap(),
            A::Decimal128(10, -2)
        );
    }

    fn struct_field(name: &str, data_type: DataType) -> crate::types::StructField {
        crate::types::StructField {
            name: name.to_string(),
            data_type,
            nullable: true,
            metadata: Default::default(),
        }
    }

    #[test]
    fn interval_input_ipc_is_exact_and_recursive() {
        let interval = DataType::DayTimeInterval {
            start_field: 0,
            end_field: 3,
        };
        let nested = DataType::Struct {
            fields: vec![
                struct_field("leaf", interval.clone()),
                struct_field(
                    "items",
                    DataType::Array {
                        element_type: Box::new(interval.clone()),
                        contains_null: true,
                    },
                ),
                struct_field(
                    "mapping",
                    DataType::Map {
                        key_type: Box::new(text_type()),
                        value_type: Box::new(interval.clone()),
                        value_contains_null: true,
                    },
                ),
            ],
        };
        let schema = schema(vec![interval, nested]);
        let mut rows = vec![row(vec![Value::Null, Value::Null])];
        for micros in [
            i64::MIN,
            i64::MIN + 1,
            -172_800_000_001,
            -1,
            0,
            1,
            i64::MAX - 1,
            i64::MAX,
        ] {
            let value = Value::DayTimeInterval(micros);
            rows.push(row(vec![
                value.clone(),
                Value::Struct(
                    Row::new(
                        vec!["leaf".into(), "items".into(), "mapping".into()],
                        vec![
                            value.clone(),
                            Value::List(vec![Value::Null, value.clone()]),
                            Value::Map(vec![
                                (Value::String("k".into()), value),
                                (Value::String("null".into()), Value::Null),
                            ]),
                        ],
                    )
                    .into(),
                ),
            ]));
        }
        let batch = read_batch(&rows, &schema);
        assert_eq!(
            batch.column(0).data_type(),
            &arrow::datatypes::DataType::Duration(arrow::datatypes::TimeUnit::Microsecond)
        );
        let decoded =
            crate::dataframe::decode_arrow_batch(&proto::execute_plan_response::ArrowBatch {
                row_count: rows.len() as i64,
                data: rows_to_arrow_ipc(&rows, &schema).unwrap().into(),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(decoded.len(), rows.len());
        for (actual, expected) in decoded.iter().zip(&rows) {
            assert_eq!(actual.values(), expected.values());
        }
        // A duration schema must reject untyped integers rather than silently treating them as micros.
        assert!(rows_to_arrow_ipc(&[row(vec![Value::Long(1), Value::Null])], &schema).is_err());
    }

    #[test]
    fn small_integer_rows_encode_as_floating_ipc() {
        use arrow::array::{Array, Float32Array, Float64Array, ListArray, MapArray, StructArray};
        use arrow::ipc::reader::StreamReader;
        use std::io::Cursor;

        let inputs = vec![
            Value::Byte(i8::MIN),
            Value::Byte(-1),
            Value::Byte(0),
            Value::Byte(i8::MAX),
            Value::Short(i16::MIN),
            Value::Short(-1),
            Value::Short(0),
            Value::Short(i16::MAX),
            Value::Null,
        ];
        let expected = vec![
            Some(-128.0),
            Some(-1.0),
            Some(0.0),
            Some(127.0),
            Some(-32768.0),
            Some(-1.0),
            Some(0.0),
            Some(32767.0),
            None,
        ];
        for target in [DataType::Float, DataType::Double] {
            let schema = DataType::Struct {
                fields: vec![
                    struct_field("scalar", target.clone()),
                    struct_field(
                        "array",
                        DataType::Array {
                            element_type: Box::new(target.clone()),
                            contains_null: true,
                        },
                    ),
                    struct_field(
                        "map",
                        DataType::Map {
                            key_type: Box::new(DataType::String {
                                collation: "UTF8_BINARY".into(),
                            }),
                            value_type: Box::new(target.clone()),
                            value_contains_null: true,
                        },
                    ),
                    struct_field(
                        "struct",
                        DataType::Struct {
                            fields: vec![struct_field("n", target.clone())],
                        },
                    ),
                ],
            };
            let rows: Vec<_> = inputs
                .iter()
                .map(|value| {
                    Row::new(
                        vec![
                            "scalar".into(),
                            "array".into(),
                            "map".into(),
                            "struct".into(),
                        ],
                        vec![
                            value.clone(),
                            Value::List(vec![value.clone()]),
                            Value::Map(vec![(Value::String("k".to_string()), value.clone())]),
                            Value::Struct(Row::new(vec!["n".into()], vec![value.clone()]).into()),
                        ],
                    )
                })
                .collect();
            let bytes = rows_to_arrow_ipc(&rows, &schema).unwrap();
            let mut reader = StreamReader::try_new(Cursor::new(bytes), None).unwrap();
            let batch = reader.next().unwrap().unwrap();
            assert_eq!(batch.num_rows(), inputs.len());
            let array = batch
                .column(1)
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap();
            let map = batch.column(2).as_any().downcast_ref::<MapArray>().unwrap();
            let nested = batch
                .column(3)
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap();
            for column in [
                batch.column(0),
                array.values(),
                map.values(),
                nested.column(0),
            ] {
                match target {
                    DataType::Float => assert_eq!(
                        column.as_any().downcast_ref::<Float32Array>().unwrap(),
                        &Float32Array::from(
                            expected
                                .iter()
                                .map(|v| v.map(|n| n as f32))
                                .collect::<Vec<_>>()
                        ),
                    ),
                    DataType::Double => assert_eq!(
                        column.as_any().downcast_ref::<Float64Array>().unwrap(),
                        &Float64Array::from(expected.clone()),
                    ),
                    _ => unreachable!(),
                }
                assert_eq!(column.null_count(), 1);
            }
            assert!(reader.next().is_none());
            for invalid in [
                Value::Bool(true),
                Value::String("7".into()),
                Value::Binary(vec![7]),
            ] {
                for (value, data_type) in [
                    (invalid.clone(), target.clone()),
                    (
                        Value::List(vec![invalid.clone()]),
                        DataType::Array {
                            element_type: Box::new(target.clone()),
                            contains_null: true,
                        },
                    ),
                    (
                        Value::Map(vec![(Value::String("k".into()), invalid.clone())]),
                        DataType::Map {
                            key_type: Box::new(text_type()),
                            value_type: Box::new(target.clone()),
                            value_contains_null: true,
                        },
                    ),
                    (
                        Value::Struct(Row::new(vec!["n".into()], vec![invalid.clone()]).into()),
                        DataType::Struct {
                            fields: vec![struct_field("n", target.clone())],
                        },
                    ),
                ] {
                    let rows = vec![Row::new(vec!["n".into()], vec![value])];
                    let schema = DataType::Struct {
                        fields: vec![struct_field("n", data_type)],
                    };
                    assert!(rows_to_arrow_ipc(&rows, &schema)
                        .unwrap_err()
                        .to_string()
                        .contains("Type mismatch in row data"));
                }
            }
        }
    }

    #[test]
    fn nested_struct_ipc_preserves_first_name_position_and_numeric_types() {
        use arrow::array::{
            Array, Int16Array, Int64Array, Int8Array, ListArray, MapArray, StringArray, StructArray,
        };

        let source = Value::Struct(
            Row::with_shared_fields(
                vec!["dup", "dup", "other", "nested", "items", "map"]
                    .into_iter()
                    .map(str::to_string)
                    .collect::<Arc<[String]>>(),
                vec![
                    Value::Long(25),
                    Value::Long(999),
                    Value::Long(-2),
                    Value::Struct(Row::new(vec!["n".to_string()], vec![Value::Long(30)]).into()),
                    Value::List(vec![Value::Long(655), Value::Null]),
                    Value::Map(vec![(Value::String("k".to_string()), Value::Long(656))]),
                ],
            )
            .into(),
        );
        let target = DataType::Struct {
            fields: vec![
                struct_field("dup", DataType::Byte),
                struct_field("positional", DataType::Short),
                struct_field("dup", DataType::Long),
                struct_field(
                    "nested",
                    DataType::Struct {
                        fields: vec![struct_field("n", DataType::Byte)],
                    },
                ),
                struct_field(
                    "items",
                    DataType::Array {
                        element_type: Box::new(DataType::Short),
                        contains_null: true,
                    },
                ),
                struct_field(
                    "map",
                    DataType::Map {
                        key_type: Box::new(DataType::String {
                            collation: "UTF8_BINARY".to_string(),
                        }),
                        value_type: Box::new(DataType::Short),
                        value_contains_null: true,
                    },
                ),
                struct_field("missing", DataType::Integer),
            ],
        };
        let rows = [
            Row::new(vec!["payload".into()], vec![source]),
            Row::new(vec!["payload".into()], vec![Value::Null]),
            Row::new(
                vec!["payload".into()],
                vec![Value::Struct(Row::new(vec![], vec![]).into())],
            ),
        ];
        let schema = DataType::Struct {
            fields: vec![struct_field("payload", target)],
        };
        // The values fit their declared types: an out-of-range value is not part of what this
        // checks (the encoder narrows it with `as`, where PySpark raises ArrowInvalid).
        // Read the actual IPC with Arrow rather than the client's decoder, so
        // symmetric encoding/decoding errors cannot hide a coercion regression.
        let bytes = rows_to_arrow_ipc(&rows, &schema).unwrap();
        let mut reader = StreamReader::try_new(Cursor::new(bytes), None).unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert!(reader.next().is_none());
        let actual = batch
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(actual.len(), 3);
        assert!(!actual.is_null(0));
        assert!(actual.is_null(1));
        assert!(!actual.is_null(2));
        assert_eq!(
            actual
                .column(0)
                .as_any()
                .downcast_ref::<Int8Array>()
                .unwrap(),
            &Int8Array::from(vec![Some(25), None, None]),
        );
        assert_eq!(
            actual
                .column(1)
                .as_any()
                .downcast_ref::<Int16Array>()
                .unwrap(),
            &Int16Array::from(vec![Some(999), None, None]),
        );
        assert_eq!(
            actual
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap(),
            &Int64Array::from(vec![Some(25), None, None]),
        );
        let nested = actual
            .column(3)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(nested.null_count(), 2);
        assert_eq!(
            nested
                .column(0)
                .as_any()
                .downcast_ref::<Int8Array>()
                .unwrap(),
            &Int8Array::from(vec![Some(30), None, None]),
        );
        let items = actual
            .column(4)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(items.null_count(), 2);
        assert_eq!(items.value_offsets(), &[0, 2, 2, 2]);
        assert_eq!(
            items
                .values()
                .as_any()
                .downcast_ref::<Int16Array>()
                .unwrap(),
            &Int16Array::from(vec![Some(655), None]),
        );
        let map = actual
            .column(5)
            .as_any()
            .downcast_ref::<MapArray>()
            .unwrap();
        assert_eq!(map.null_count(), 2);
        assert_eq!(map.value_offsets(), &[0, 1, 1, 1]);
        assert_eq!(
            map.keys().as_any().downcast_ref::<StringArray>().unwrap(),
            &StringArray::from(vec!["k"]),
        );
        assert_eq!(
            map.values().as_any().downcast_ref::<Int16Array>().unwrap(),
            &Int16Array::from(vec![Some(656)]),
        );
        assert_eq!(actual.column(6).null_count(), 3);
    }

    #[test]
    fn nested_duplicate_field_names_are_numbered_per_occurrence_as_pyspark() {
        assert_eq!(dedup_field_names(["a", "b"]), vec!["a", "b"]);
        assert_eq!(
            dedup_field_names(["a", "a", "b", "a"]),
            vec!["a_0", "a_1", "b", "a_2"]
        );
    }

    #[test]
    fn a_non_nullable_element_with_nulls_is_an_error_not_a_panic() {
        // An array of a non-nullable NullType that has a null element: Arrow rejects the
        // element field, which must be an error, as for a struct field with nulls.
        let array = DataType::Array {
            element_type: Box::new(DataType::Null),
            contains_null: false,
        };
        let value = Value::List(vec![Value::Null]);
        let error = values_to_arrow(&[&value], &array)
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(error.contains("list build"), "{error}");
        assert!(error.contains("cannot contain nulls"), "{error}");
        // An empty list has no null element, which Arrow accepts.
        let empty = Value::List(vec![]);
        assert!(values_to_arrow(&[&empty], &array).is_ok());
    }

    #[test]
    fn struct_arrow_encoding_preserves_lookup_map_and_null_semantics() {
        use arrow::array::{Array, Int16Array, Int32Array, Int64Array, StructArray};

        let target = DataType::Struct {
            fields: vec![
                struct_field("dup", DataType::Integer),
                struct_field("positional", DataType::Short),
                struct_field("dup", DataType::Long),
                struct_field("missing", DataType::Integer),
            ],
        };
        let values = [
            Value::Struct(
                Row::new(
                    vec!["dup".to_string(), "dup".to_string(), "other".to_string()],
                    vec![Value::Long(257), Value::Long(999), Value::Long(-2)],
                )
                .into(),
            ),
            Value::Map(vec![
                (Value::String("dup".to_string()), Value::Long(5)),
                (Value::String("other".to_string()), Value::Long(9)),
            ]),
            Value::Null,
            Value::Struct(Row::new(vec!["dup".to_string()], vec![Value::Null]).into()),
        ];
        let array = values_to_arrow(&values.iter().collect::<Vec<_>>(), &target).unwrap();
        let array = array.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(array.len(), 4);
        assert!(!array.is_null(0));
        assert!(!array.is_null(1));
        assert!(array.is_null(2));
        assert!(!array.is_null(3));
        let first = array
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(
            first,
            &Int32Array::from(vec![Some(257), Some(5), None, None])
        );
        let positional = array
            .column(1)
            .as_any()
            .downcast_ref::<Int16Array>()
            .unwrap();
        assert_eq!(
            positional,
            &Int16Array::from(vec![Some(999), None, None, None])
        );
        let duplicate = array
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(
            duplicate,
            &Int64Array::from(vec![Some(257), Some(5), None, None])
        );
        assert_eq!(array.column(3).null_count(), 4);
    }

    #[test]
    fn struct_arrow_encoding_recurses_through_arrays_and_maps() {
        use arrow::array::{Int16Array, ListArray, MapArray, StructArray};

        let leaf_type = DataType::Struct {
            fields: vec![struct_field("n", DataType::Short)],
        };
        let leaf_value =
            || Value::Struct(Row::new(vec!["n".to_string()], vec![Value::Long(300)]).into());
        let target = DataType::Struct {
            fields: vec![
                struct_field(
                    "items",
                    DataType::Array {
                        element_type: Box::new(leaf_type.clone()),
                        contains_null: true,
                    },
                ),
                struct_field(
                    "lookup",
                    DataType::Map {
                        key_type: Box::new(DataType::String {
                            collation: "UTF8_BINARY".to_string(),
                        }),
                        value_type: Box::new(leaf_type),
                        value_contains_null: true,
                    },
                ),
            ],
        };
        let source = Value::Struct(
            Row::new(
                vec!["items".to_string(), "lookup".to_string()],
                vec![
                    Value::List(vec![leaf_value(), Value::Null]),
                    Value::Map(vec![(Value::String("k".to_string()), leaf_value())]),
                ],
            )
            .into(),
        );
        let batch = read_batch(&[row(vec![source])], &schema(vec![target]));
        let array = batch.column(0);
        let array = array.as_any().downcast_ref::<StructArray>().unwrap();
        let items = array
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let items = items.value(0);
        let items = items.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(
            items
                .column(0)
                .as_any()
                .downcast_ref::<Int16Array>()
                .unwrap(),
            &Int16Array::from(vec![Some(300), None])
        );
        let lookup = array.column(1).as_any().downcast_ref::<MapArray>().unwrap();
        let entries = lookup.value(0);
        let value = entries
            .column(1)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(
            value
                .column(0)
                .as_any()
                .downcast_ref::<Int16Array>()
                .unwrap(),
            &Int16Array::from(vec![Some(300)])
        );
    }

    fn session() -> SparkSession {
        SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("failed to build session")
    }

    #[test]
    fn tags_are_those_of_each_thread() {
        // As PySpark Connect's, whose tags are of the current thread: a new thread has none,
        // and the tags it adds are on its own requests only.
        let spark = session();
        spark.add_tag("main").unwrap();
        let other = spark.clone();
        let child = std::thread::spawn(move || {
            let before = other.get_tags();
            other.add_tag("child").unwrap();
            let added = (other.get_tags(), other.tags().unwrap());
            other.remove_tag("child");
            (before, added, other.get_tags())
        })
        .join()
        .unwrap();
        let child_tags = vec!["child".to_string()];
        assert_eq!(child, (vec![], (child_tags.clone(), child_tags), vec![]));
        assert_eq!(spark.get_tags(), vec!["main".to_string()]);
        assert_eq!(spark.tags().unwrap(), vec!["main".to_string()]);
        spark.clear_tags();
        assert!(spark.get_tags().is_empty());
        assert!(spark.inner.tags.by_thread.lock().unwrap().is_empty());
    }

    /// A tag of a str that UTF-8 cannot encode, as PySpark's client keeps it: the requests that
    /// carry the tags fail with its error until it is removed, and it is not a tag of text.
    #[test]
    fn an_unencodable_tag_fails_the_requests_until_it_is_removed() {
        let spark = session();
        spark.add_tag("a").unwrap();
        let surrogate = [0x61, 0, 0, 0, 0x00, 0xd8, 0, 0];
        spark.add_unencodable_tag(surrogate.to_vec());
        spark.add_unencodable_tag(surrogate.to_vec());
        let error = spark.tags().unwrap_err();
        assert_eq!(
            error.kind,
            spark_connect_core::error::SparkErrorKind::UnicodeEncode
        );
        assert_eq!(error.unencodable_value(), Some(surrogate.to_vec()));
        assert_eq!(spark.get_tags(), vec!["a".to_string()]);
        assert_eq!(spark.unencodable_tags(), vec![surrogate.to_vec()]);
        spark.remove_tag("a");
        spark.remove_unencodable_tag(&surrogate);
        assert!(spark.tags().unwrap().is_empty());
        assert!(spark.unencodable_tags().is_empty());
    }

    /// Tags that protobuf's string field does not take, as the set of PySpark's client keeps them:
    /// the executions fail with the `TypeError` of an object that is neither a str nor bytes and
    /// with the `UnicodeDecodeError` of bytes that are not UTF-8, until they are gone; they are
    /// not tags of text. An execution of a thread of no tags gives it an empty set of them, as
    /// PySpark's `get_tags` does.
    #[test]
    fn tags_that_the_field_does_not_take_fail_the_executions() {
        let spark = session();
        let refused = SharedTags::new(vec![
            Tag::Text("a".to_string()),
            Tag::NotText("bad argument type for built-in operation".to_string()),
        ]);
        spark.share_thread_tags(&refused);
        let error = spark.tags().unwrap_err();
        assert_eq!(
            error.kind,
            spark_connect_core::error::SparkErrorKind::BuiltinTypeError
        );
        assert_eq!(error.message(), "bad argument type for built-in operation");
        assert_eq!(spark.get_tags(), vec!["a".to_string()]);
        spark.share_thread_tags(&SharedTags::new(vec![Tag::NotUtf8(vec![0xff])]));
        let error = spark.tags().unwrap_err();
        assert_eq!(error.invalid_utf8_value(), Some(vec![0xff]));
        assert!(spark.get_tags().is_empty() && spark.unencodable_tags().is_empty());
        spark.clear_tags();
        assert!(!spark.has_thread_tags());
        assert!(spark.tags().unwrap().is_empty());
        assert!(spark.has_thread_tags());
        spark.clear_tags();
    }

    #[test]
    fn threads_given_a_copy_of_tags_share_it() {
        // As the threads of PySpark's inheritable_thread_target: they share the copy of the tags
        // of the thread that decorated their function, of then, until one clears its tags.
        let spark = session();
        spark.add_tag("a").unwrap();
        let copy = spark.copy_thread_tags();
        spark.add_tag("later").unwrap();
        let run = |copy: SharedTags, tag: &'static str| {
            let spark = spark.clone();
            std::thread::spawn(move || {
                spark.share_thread_tags(&copy);
                spark.add_tag(tag).unwrap();
                spark.get_tags()
            })
            .join()
            .unwrap()
        };
        assert_eq!(run(copy.clone(), "b"), vec!["a", "b"]);
        assert_eq!(run(copy.clone(), "c"), vec!["a", "b", "c"]);
        assert_eq!(spark.get_tags(), vec!["a", "later"]);
        let cleared = {
            let spark = spark.clone();
            let copy = copy.clone();
            std::thread::spawn(move || {
                spark.share_thread_tags(&copy);
                spark.remove_tag("c");
                spark.clear_tags();
                spark.add_tag("d").unwrap();
                spark.get_tags()
            })
            .join()
            .unwrap()
        };
        assert_eq!(cleared, vec!["d"]);
        assert_eq!(run(copy, "e"), vec!["a", "b", "e"]);
    }

    #[test]
    fn a_set_of_tags_given_to_threads_is_theirs_and_its_changes_are_theirs() {
        // As the set of PySpark's thread_local.tags, which a binding keeps and changes: a new set
        // that a thread is given is its tags, whose changes are the tags of the threads that have
        // it, of their text or of the bytes of a str that UTF-8 cannot encode, and a thread that is
        // not given it keeps its own.
        let spark = session();
        spark.add_tag("own").unwrap();
        let set = SharedTags::new(vec![Tag::Text("a".to_string())]);
        spark.share_thread_tags(&set);
        assert_eq!(spark.get_tags(), vec!["a"]);
        let surrogate = vec![0x00, 0xd8, 0x00, 0x00];
        set.replace(vec![
            Tag::Text("b".to_string()),
            Tag::Unencodable(surrogate.clone()),
        ]);
        assert_eq!(spark.get_tags(), vec!["b"]);
        assert_eq!(spark.unencodable_tags(), vec![surrogate.clone()]);
        assert!(spark.tags().is_err());
        assert_eq!(set.tags().len(), 2);

        let seen = {
            let spark = spark.clone();
            let set = set.clone();
            std::thread::spawn(move || {
                let before = spark.get_tags();
                spark.share_thread_tags(&set);
                set.replace(vec![Tag::Text("c".to_string())]);
                (before, spark.get_tags())
            })
            .join()
            .unwrap()
        };
        assert_eq!(seen, (Vec::<String>::new(), vec!["c".to_string()]));
        assert_eq!(spark.get_tags(), vec!["c"]);
        spark.clear_tags();
        assert!(spark.get_tags().is_empty());
        assert_eq!(set.tags(), vec![Tag::Text("c".to_string())]);
    }

    #[test]
    fn finished_threads_leave_no_tags() {
        // The threads that share a copy of the tags are forgotten when they end, as PySpark's
        // thread_local is freed with its thread: 1000 short threads leave no entry.
        let spark = session();
        let copy = spark.copy_thread_tags();
        for _ in 0..1000 {
            let spark = spark.clone();
            let copy = copy.clone();
            std::thread::spawn(move || {
                spark.share_thread_tags(&copy);
                spark.add_tag("task").unwrap();
            })
            .join()
            .unwrap();
        }
        assert!(spark.inner.tags.by_thread.lock().unwrap().is_empty());
        assert!(spark.get_tags().is_empty());
    }

    #[test]
    fn the_tags_that_a_future_of_a_call_sets_are_the_callers() {
        // As the user context extensions of the core: a future of a call that runs on another
        // thread, inside a `current_thread` runtime, is of the thread that calls, whose tags it
        // reads and sets, and which are forgotten when that thread ends.
        let spark = session();
        let other = spark.clone();
        std::thread::spawn(move || {
            other.add_tag("caller").unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let in_the_future = runtime.block_on(async {
                block_on(async {
                    other.add_tag("future").unwrap();
                    (std::thread::current().id(), other.get_tags())
                })
            });
            assert_ne!(in_the_future.0, std::thread::current().id());
            assert_eq!(in_the_future.1, vec!["caller", "future"]);
            assert_eq!(other.get_tags(), vec!["caller", "future"]);
        })
        .join()
        .unwrap();
        assert!(spark.inner.tags.by_thread.lock().unwrap().is_empty());
    }

    #[test]
    fn a_kept_request_context_of_a_thread_is_given_back() {
        // As a binding keeps the context of the requests of a call around a call that runs
        // during it, such as one of a progress handler or of a finalizer: the session ID, the user
        // and the user context extensions of the requests of the thread, and its tags.
        let spark = session();
        let any = |name: &str| prost_types::Any {
            type_url: format!("type.googleapis.com/{name}"),
            value: vec![],
        };
        std::thread::spawn(move || {
            spark.add_tag("outer").unwrap();
            spark.set_thread_request_context(
                "outer-session".to_string(),
                Some("alice".to_string()),
                vec![("global_outer".to_string(), any("outer"))],
            );
            {
                let _kept = spark.keep_thread_request_context();
                spark.clear_tags();
                spark.add_tag("inner").unwrap();
                spark.set_thread_request_context("inner-session".to_string(), None, vec![]);
                spark.share_threadlocal_user_context_extensions(
                    &spark_connect_core::client::SharedUserContextExtensions::new(vec![(
                        "threadlocal_inner".to_string(),
                        any("inner"),
                    )]),
                );
                assert_eq!(spark.get_tags(), vec!["inner"]);
                assert_eq!(spark.client_session_id(), "inner-session");
            }
            assert_eq!(spark.get_tags(), vec!["outer"]);
            assert_eq!(spark.client_session_id(), "outer-session");
            assert_eq!(spark.request_user_id().as_deref(), Some("alice"));
            assert!(spark
                .client()
                .threadlocal_user_context_extensions()
                .is_empty());
            assert_eq!(spark.client().global_user_context_extensions().len(), 1);
            // Of a thread of no tags, the tags of the call during it are gone with it.
            spark.clear_tags();
            {
                let _kept = spark.keep_thread_request_context();
                spark.add_tag("inner").unwrap();
            }
            assert!(!spark.has_thread_tags());
        })
        .join()
        .unwrap();
    }

    #[test]
    fn a_living_thread_keeps_its_tags_until_it_ends() {
        let spark = session();
        let copy = spark.copy_thread_tags();
        let (tags_tx, tags_rx) = std::sync::mpsc::channel();
        let (end_tx, end_rx) = std::sync::mpsc::channel::<()>();
        let child = {
            let spark = spark.clone();
            std::thread::spawn(move || {
                spark.share_thread_tags(&copy);
                spark.add_tag("a").unwrap();
                tags_tx.send(spark.get_tags()).unwrap();
                end_rx.recv().unwrap();
                spark.get_tags()
            })
        };
        assert_eq!(tags_rx.recv().unwrap(), vec!["a"]);
        assert_eq!(spark.inner.tags.by_thread.lock().unwrap().len(), 1);
        end_tx.send(()).unwrap();
        assert_eq!(child.join().unwrap(), vec!["a"]);
        assert!(spark.inner.tags.by_thread.lock().unwrap().is_empty());
    }

    #[test]
    fn arrow_local_relation_preserves_batches_and_rejects_schema_mismatch() {
        use arrow::array::Array;
        use arrow::datatypes::{DataType as ArrowType, Field, Schema};
        let arrow_schema = Arc::new(Schema::new(vec![Field::new("x", ArrowType::Int32, true)]));
        let batch = RecordBatch::try_new(
            arrow_schema,
            vec![Arc::new(Int32Array::from(vec![Some(1), None]))],
        )
        .unwrap();
        let schema = DataType::Struct {
            fields: vec![struct_field("x", DataType::Integer)],
        };
        let spark = session();
        let plan = crate::arrow_input::inline_relation(
            &[batch.clone(), batch.clone()],
            Some(schema.clone()),
        )
        .unwrap();
        let LogicalPlan::LocalRelation {
            data: Some(data), ..
        } = &plan
        else {
            panic!("expected Arrow local relation")
        };
        let batches = StreamReader::try_new(Cursor::new(data), None)
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], batch);
        assert!(batches[1].column(0).is_null(1));
        let other = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("y", ArrowType::Int32, true)])),
            batch.columns().to_vec(),
        )
        .unwrap();
        assert!(spark
            .create_dataframe_from_arrow(vec![batch, other], schema.clone())
            .is_err());
        assert!(matches!(
            spark
                .create_dataframe_from_arrow(vec![], schema)
                .unwrap()
                .plan(),
            LogicalPlan::LocalRelation { data: None, .. }
        ));
    }

    #[test]
    fn stop_is_idempotent_and_marks_stopped() {
        // Port 1: nothing listens, and a release there waits for the retry policy, as PySpark's
        // does, so the session is closed without it: stop() then only marks it stopped, which
        // is what a second stop() must find.
        let spark = SparkSession::builder()
            .remote("sc://localhost:1")
            .get_or_create()
            .expect("failed to build session");
        let other = spark.clone();
        spark.close();
        assert!(spark.is_stopped());
        assert!(other.is_stopped(), "stopped flag is shared across clones");
        assert!(spark.stop().is_ok());
    }

    #[test]
    fn requests_of_a_stopped_session_raise_no_active_session() {
        // Port 1: nothing listens, so a request that reached the wire would fail differently.
        // As PySpark's closed client, the stopped session raises before it sends anything.
        let spark = SparkSession::builder()
            .remote("sc://localhost:1")
            .get_or_create()
            .expect("failed to build session");
        spark.close();
        let err = spark
            .conf()
            .get("spark.sql.shuffle.partitions")
            .expect_err("a request of a stopped session must fail");
        assert_eq!(err.get_condition().as_deref(), Some("NO_ACTIVE_SESSION"));
    }

    /// As the `_stub` of PySpark's closed client: `release_session` after `stop()` or `close()`
    /// raises `NO_ACTIVE_SESSION` and sends nothing.
    #[test]
    fn release_session_of_a_stopped_session_raises_no_active_session_as_pyspark() {
        // No server and no retries: a release that reached the wire would fail at once, with
        // another error.
        let unreachable = || {
            let builder = ChannelBuilder::parse("sc://127.0.0.1:1").expect("valid URL");
            let mut client =
                block_on(SparkConnectClient::connect(&builder)).expect("failed to build client");
            client.set_retry_policy(spark_connect_core::retries::RetryPolicy::no_retries());
            SparkSession::new(client)
        };
        let closed = unreachable();
        closed.close();
        let stopped = unreachable();
        assert!(
            stopped.stop().is_err(),
            "the release of stop() reaches no server"
        );
        for spark in [closed, stopped] {
            let err = spark.release_session().expect_err("a stopped session");
            assert_eq!(err.get_condition().as_deref(), Some("NO_ACTIVE_SESSION"));
        }
    }

    /// A port that takes connections and never answers them, so an RPC to it waits until the
    /// connections close, when this drops.
    struct UnansweringServer {
        port: u16,
        connections: Arc<Mutex<Vec<std::net::TcpStream>>>,
        done: Arc<AtomicBool>,
    }

    impl UnansweringServer {
        fn start() -> Self {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a local port");
            listener
                .set_nonblocking(true)
                .expect("a nonblocking listener");
            let port = listener.local_addr().expect("the local address").port();
            let connections: Arc<Mutex<Vec<std::net::TcpStream>>> = Arc::default();
            let done = Arc::new(AtomicBool::new(false));
            let (taken, ended) = (Arc::clone(&connections), Arc::clone(&done));
            std::thread::spawn(move || {
                while !ended.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((connection, _)) => taken.lock().unwrap().push(connection),
                        Err(_) => std::thread::sleep(std::time::Duration::from_millis(5)),
                    }
                }
            });
            UnansweringServer {
                port,
                connections,
                done,
            }
        }
    }

    impl Drop for UnansweringServer {
        fn drop(&mut self) {
            self.done.store(true, Ordering::SeqCst);
            self.connections.lock().unwrap().clear();
        }
    }

    /// As PySpark's `stop()`, which runs under `SparkSession._lock`: the `stop()` of another
    /// thread waits for the one in progress, and when a signal ends that one (in the Python
    /// client; here a watch that aborts its wait), it releases the session itself. The session
    /// is stopped once that release has ended, as PySpark's `is_stopped` is the client's
    /// `is_closed`, which its `stop()` sets after the release.
    #[test]
    fn a_stop_that_waits_for_an_interrupted_stop_releases_the_session_as_pyspark() {
        use spark_connect_core::runtime::{with_wait_watch, WaitWatch, Watch};
        use std::sync::mpsc::{channel, RecvTimeoutError};
        use std::time::{Duration, Instant};

        let server = UnansweringServer::start();
        let builder =
            ChannelBuilder::parse(&format!("sc://127.0.0.1:{}", server.port)).expect("valid URL");
        let mut client =
            block_on(SparkConnectClient::connect(&builder)).expect("failed to build client");
        // The release fails at once when its connection closes.
        client.set_retry_policy(spark_connect_core::retries::RetryPolicy::no_retries());
        let spark = SparkSession::new(client);
        let aborted = Arc::new(AtomicBool::new(false));
        let watch: Arc<WaitWatch> = {
            let aborted = Arc::clone(&aborted);
            Arc::new(move || {
                if aborted.load(Ordering::SeqCst) {
                    Watch::Abort
                } else {
                    Watch::Continue
                }
            })
        };
        let (first_ended, first) = channel();
        let interrupted = spark.clone();
        std::thread::spawn(move || {
            let _ = first_ended.send(with_wait_watch(watch, || interrupted.stop()));
        });
        // The release of the first stop has connected, so that stop holds the lock.
        let deadline = Instant::now() + Duration::from_secs(5);
        while server.connections.lock().unwrap().is_empty() {
            assert!(
                Instant::now() < deadline,
                "the release of the first stop connects"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let (second_ended, second) = channel();
        let waiting = spark.clone();
        std::thread::spawn(move || {
            let _ = second_ended.send(waiting.stop());
        });
        assert!(
            matches!(
                second.recv_timeout(Duration::from_millis(300)),
                Err(RecvTimeoutError::Timeout)
            ),
            "a stop returned while another one released the session"
        );
        aborted.store(true, Ordering::SeqCst);
        first
            .recv_timeout(Duration::from_secs(5))
            .expect("the watch ends the first stop")
            .expect_err("the first stop is interrupted");
        // The second stop releases the session now; until that release ends, it is not stopped.
        assert!(matches!(
            second.recv_timeout(Duration::from_millis(300)),
            Err(RecvTimeoutError::Timeout)
        ));
        assert!(!spark.is_stopped());
        drop(server);
        second
            .recv_timeout(Duration::from_secs(5))
            .expect("the second stop ends with its release")
            .expect_err("the release of the second stop fails as its connection closes");
        assert!(spark.is_stopped());
    }

    /// A signal that interrupts the stop in the backoff of its release makes the stop raise, and
    /// leaves the session open for a later stop to release, as PySpark's `stop()` does. The
    /// server closes each connection at once, so the release fails as UNAVAILABLE and backs off
    /// for a minute, unless the signal ends the backoff first.
    #[test]
    fn a_stop_that_a_signal_interrupts_in_its_backoff_leaves_the_session_open() {
        use spark_connect_core::runtime::{with_wait_watch, WaitWatch, Watch};
        use std::sync::mpsc::channel;
        use std::time::{Duration, Instant};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a local port");
        listener
            .set_nonblocking(true)
            .expect("a nonblocking listener");
        let port = listener.local_addr().expect("the local address").port();
        let accepted = Arc::new(AtomicU64::new(0));
        let ended = Arc::new(AtomicBool::new(false));
        let (taken, done) = (Arc::clone(&accepted), Arc::clone(&ended));
        std::thread::spawn(move || {
            while !done.load(Ordering::SeqCst) {
                match listener.accept() {
                    // The connection is dropped at once, which closes it.
                    Ok(_) => {
                        taken.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(_) => std::thread::sleep(Duration::from_millis(5)),
                }
            }
        });
        let builder = ChannelBuilder::parse(&format!("sc://127.0.0.1:{port}")).expect("valid URL");
        let mut client =
            block_on(SparkConnectClient::connect(&builder)).expect("failed to build client");
        client.set_retry_policy(spark_connect_core::retries::RetryPolicy {
            initial_backoff_ms: 60_000,
            max_backoff_ms: Some(60_000),
            ..spark_connect_core::retries::RetryPolicy::default()
        });
        let spark = SparkSession::new(client);
        // One interrupt, once the release has connected, and half a second later, so that the
        // failed release is backing off by then.
        let watch: Arc<WaitWatch> = {
            let taken = Arc::clone(&accepted);
            let interrupted = Arc::new(AtomicBool::new(false));
            let since: Arc<Mutex<Option<Instant>>> = Arc::default();
            Arc::new(move || {
                if taken.load(Ordering::SeqCst) == 0 {
                    return Watch::Continue;
                }
                let first = since
                    .lock()
                    .unwrap()
                    .get_or_insert_with(Instant::now)
                    .to_owned();
                if first.elapsed() >= Duration::from_millis(500)
                    && !interrupted.swap(true, Ordering::SeqCst)
                {
                    Watch::Interrupt(Box::pin(async {}))
                } else {
                    Watch::Continue
                }
            })
        };
        let (stopped, outcome) = channel();
        let stopping = spark.clone();
        std::thread::spawn(move || {
            let _ = stopped.send(with_wait_watch(watch, || stopping.stop()));
        });
        let stop = outcome
            .recv_timeout(Duration::from_secs(5))
            .expect("the signal ends the backoff of the release");
        let error = stop.expect_err("the interrupted stop raises");
        assert!(
            error
                .to_string()
                .contains("the release of the session was interrupted; the session is still open"),
            "the interrupted stop raises the error of an interrupted release: {error}"
        );
        assert!(
            !spark.is_stopped(),
            "an interrupted stop marked the session stopped"
        );
        assert!(
            !spark.client().is_closed(),
            "an interrupted stop closed the client"
        );
        ended.store(true, Ordering::SeqCst);
    }

    /// As PySpark's `stop()`, whose `SparkSession._lock` is one lock of the class: the `stop()` of
    /// another session waits for the release of the one in progress, sends its own release only
    /// after it, and is not stopped until that release has ended.
    #[test]
    fn stops_of_different_sessions_run_one_at_a_time_as_pyspark() {
        use spark_connect_core::runtime::{with_wait_watch, WaitWatch, Watch};
        use std::sync::mpsc::{channel, RecvTimeoutError};
        use std::time::{Duration, Instant};

        let server = UnansweringServer::start();
        // Of two keepalives, so that each session has a connection of its own, by which the
        // release of each is seen: the sessions of one target share a connection.
        let new_session = |keepalive_ms: u64| {
            let builder = ChannelBuilder::parse(&format!(
                "sc://127.0.0.1:{}/;grpc_keepalive_time_ms={keepalive_ms}",
                server.port
            ))
            .expect("valid URL");
            let mut client =
                block_on(SparkConnectClient::connect(&builder)).expect("failed to build client");
            // The release fails at once when its connection closes.
            client.set_retry_policy(spark_connect_core::retries::RetryPolicy::no_retries());
            SparkSession::new(client)
        };
        let first_session = new_session(10_000);
        let second_session = new_session(20_000);
        let aborted = Arc::new(AtomicBool::new(false));
        let watch: Arc<WaitWatch> = {
            let aborted = Arc::clone(&aborted);
            Arc::new(move || {
                if aborted.load(Ordering::SeqCst) {
                    Watch::Abort
                } else {
                    Watch::Continue
                }
            })
        };
        let (first_ended, first) = channel();
        let interrupted = first_session.clone();
        std::thread::spawn(move || {
            let _ = first_ended.send(with_wait_watch(watch, || interrupted.stop()));
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        while server.connections.lock().unwrap().is_empty() {
            assert!(
                Instant::now() < deadline,
                "the release of the first stop connects"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let (second_ended, second) = channel();
        let waiting = second_session.clone();
        std::thread::spawn(move || {
            let _ = second_ended.send(waiting.stop());
        });
        assert!(
            matches!(
                second.recv_timeout(Duration::from_millis(300)),
                Err(RecvTimeoutError::Timeout)
            ),
            "a stop of another session returned while the release of the first was in progress"
        );
        let connected = server.connections.lock().unwrap().len();
        assert_eq!(
            connected, 1,
            "the release of the other session was sent while the first stop was in progress"
        );
        aborted.store(true, Ordering::SeqCst);
        first
            .recv_timeout(Duration::from_secs(5))
            .expect("the watch ends the first stop")
            .expect_err("the first stop is interrupted");
        // The second stop releases its session once the first one has left the lock.
        let deadline = Instant::now() + Duration::from_secs(5);
        while server.connections.lock().unwrap().len() < 2 {
            assert!(
                Instant::now() < deadline,
                "the second stop sends its release once the first has left the lock"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(!second_session.is_stopped());
        drop(server);
        second
            .recv_timeout(Duration::from_secs(5))
            .expect("the second stop ends with its release")
            .expect_err("the release of the second stop fails as its connection closes");
        assert!(second_session.is_stopped());
        assert!(!first_session.is_stopped());
    }

    #[test]
    fn forward_to_fs_artifact_name_is_the_prefix_and_the_absolute_path() {
        // As the reference's `FORWARD_TO_FS_PREFIX + dest_path`, with no separator added.
        assert_eq!(forward_to_fs_artifact_name("/tmp/x"), "forward_to_fs/tmp/x");
    }

    #[test]
    fn version_cache_follows_session_identity() {
        let spark = session();
        let handle = spark.clone();
        *spark.inner.server_version.lock().unwrap() = Some("4.3.0".into());
        assert_eq!(handle.version().unwrap(), "4.3.0");
        assert!(std::ptr::eq(
            &spark.inner.server_version,
            &handle.inner.server_version
        ));
        for other in [spark.new_session(), session()] {
            assert!(!std::ptr::eq(
                &spark.inner.server_version,
                &other.inner.server_version
            ));
            assert!(other.inner.server_version.lock().unwrap().is_none());
        }
        spark.inner.stopped.store(true, Ordering::SeqCst);
        assert_eq!(handle.version().unwrap(), "4.3.0");
    }

    #[test]
    fn version_cache_recovers_after_lookup_panics() {
        let spark = session();
        let handle = spark.clone();
        // A panic under the cache lock, as a panicking lookup leaves it, must not permanently
        // break this session's cache.
        let cache = &spark.inner.server_version;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = cache.lock().unwrap();
            panic!("a lookup panics while the cache lock is held");
        }));
        assert!(result.is_err());
        assert!(cache.is_poisoned());
        let mut cached = cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(cached.is_none());
        *cached = Some("4.3.0".into());
        drop(cached);
        assert_eq!(handle.version().unwrap(), "4.3.0");
    }

    #[test]
    fn session_is_not_stopped_initially() {
        let spark = session();
        assert!(!spark.is_stopped());
    }

    #[test]
    fn session_id_is_set() {
        let spark = session();
        let session_id = spark.session_id();
        assert!(!session_id.is_empty());
    }

    #[test]
    fn session_builder_without_remote_fails() {
        let result = SparkSessionBuilder::new().get_or_create();
        assert!(result.is_err());
    }

    #[test]
    fn the_options_of_a_builder_are_pysparks_of_their_first_order_and_last_values() {
        // As PySpark's `_options` dict and `_apply_options`: an option given again keeps its
        // place and takes its last value; `remote` is `spark.remote`, `app_name` is
        // `spark.app.name`, `master` sets nothing, and a bool is `true` or `false`; the session
        // gets all but `spark.remote` and `spark.master`.
        let builder = SparkSessionBuilder::new()
            .config("spark.sql.shuffle.partitions", 3)
            .remote("sc://a:1")
            .app_name("first")
            .master("local[2]")
            .configs([("spark.sql.ansi.enabled", true), ("spark.master", false)])
            .config("spark.sql.shuffle.partitions", "7")
            .app_name("second")
            .config("spark.remote", "sc://b:2");
        let options: Vec<(&str, &str)> = builder
            .options
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        assert_eq!(
            options,
            [
                ("spark.sql.shuffle.partitions", "7"),
                ("spark.remote", "sc://b:2"),
                ("spark.app.name", "second"),
                ("spark.sql.ansi.enabled", "true"),
                ("spark.master", "false"),
            ]
        );
        assert_eq!(
            builder.configurations(),
            [
                ("spark.sql.shuffle.partitions".to_string(), "7".to_string()),
                ("spark.app.name".to_string(), "second".to_string()),
                ("spark.sql.ansi.enabled".to_string(), "true".to_string()),
            ]
        );
        match SparkSessionBuilder::new()
            .config("spark.sql.shuffle.partitions", 3)
            .create()
        {
            Err(error) => assert_eq!(error.get_condition().as_deref(), Some("NO_REMOTE_URL")),
            Ok(_) => panic!("a builder of no remote created a session"),
        }
    }

    #[test]
    fn threadlocal_user_context_extensions_of_a_copy_are_shared_as_pyspark() {
        // As the threads of PySpark's inheritable_thread_target: the threads given a copy of the
        // extensions of the thread that decorated their function share it, an extension that one
        // of them adds is in the copy of all, and one that it removes is removed for it alone.
        fn any_of(type_url: &str) -> prost_types::Any {
            prost_types::Any {
                type_url: type_url.to_string(),
                value: Vec::new(),
            }
        }
        fn ids(spark: &SparkSession) -> Vec<String> {
            spark
                .threadlocal_user_context_extensions()
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        }
        let spark = session();
        let main = spark.add_threadlocal_user_context_extension(any_of("main"));
        let copy = spark.copy_threadlocal_user_context_extensions();
        let late = spark.add_threadlocal_user_context_extension(any_of("late"));
        let run = |work: fn(&SparkSession) -> Vec<Vec<String>>| {
            let spark = spark.clone();
            let copy = copy.clone();
            std::thread::spawn(move || {
                spark.share_threadlocal_user_context_extensions(&copy);
                work(&spark)
            })
            .join()
            .unwrap()
        };
        let first = run(|spark| {
            let mut seen = vec![ids(spark)];
            spark.add_threadlocal_user_context_extension(any_of("added"));
            seen.push(ids(spark));
            let main_id = seen[0][0].clone();
            spark.remove_user_context_extension(&main_id);
            seen.push(ids(spark));
            seen
        });
        assert_eq!(first[0], vec![main.clone()]);
        assert_eq!(first[1].len(), 2);
        assert_eq!(first[1][0], main);
        assert_eq!(first[2].len(), 1);
        assert_eq!(run(|spark| vec![ids(spark)]), vec![first[1].clone()]);
        assert_eq!(run(|spark| vec![ids(spark)]), vec![first[1].clone()]);
        assert_eq!(ids(&spark), vec![main, late]);
    }

    #[test]
    fn integers_that_do_not_fit_the_declared_type_are_errors_not_wrapped() {
        // PySpark's Arrow conversion raises for an integer outside the declared type's
        // range, so the encoder must not wrap it with an `as` cast.
        for (value, dt, sql_type) in [
            (Value::Long(4_294_967_297), DataType::Integer, "INT"),
            (Value::Long(-2_147_483_649), DataType::Integer, "INT"),
            (Value::Long(300), DataType::Byte, "TINYINT"),
            (Value::Integer(200), DataType::Byte, "TINYINT"),
            (Value::Short(-129), DataType::Byte, "TINYINT"),
            (Value::Integer(70_000), DataType::Short, "SMALLINT"),
        ] {
            let error = values_to_arrow(&[&value], &dt)
                .map(|_| ())
                .unwrap_err()
                .to_string();
            let number = match &value {
                Value::Long(x) => x.to_string(),
                Value::Integer(x) => x.to_string(),
                Value::Short(x) => x.to_string(),
                _ => unreachable!(),
            };
            assert!(error.contains(&number), "{error}");
            assert!(error.contains(sql_type), "{error}");
            // The batch-level path reports the same error.
            let rows = [row(vec![value.clone()])];
            assert!(rows_to_arrow_batch(&rows, &schema(vec![dt])).is_err());
        }
    }

    #[test]
    fn integers_that_fit_the_declared_type_still_narrow_and_widen() {
        use arrow::array::{Int16Array, Int32Array, Int64Array, Int8Array};

        let bytes = values_to_arrow(
            &[&Value::Long(127), &Value::Integer(-128), &Value::Null],
            &DataType::Byte,
        )
        .unwrap();
        assert_eq!(
            bytes.as_any().downcast_ref::<Int8Array>().unwrap(),
            &Int8Array::from(vec![Some(127), Some(-128), None])
        );
        let shorts =
            values_to_arrow(&[&Value::Byte(-5), &Value::Long(32_767)], &DataType::Short).unwrap();
        assert_eq!(
            shorts.as_any().downcast_ref::<Int16Array>().unwrap(),
            &Int16Array::from(vec![-5, 32_767])
        );
        let ints = values_to_arrow(
            &[&Value::Long(i32::MIN as i64), &Value::Long(i32::MAX as i64)],
            &DataType::Integer,
        )
        .unwrap();
        assert_eq!(
            ints.as_any().downcast_ref::<Int32Array>().unwrap(),
            &Int32Array::from(vec![i32::MIN, i32::MAX])
        );
        let longs = values_to_arrow(
            &[&Value::Integer(-7), &Value::Long(i64::MAX)],
            &DataType::Long,
        )
        .unwrap();
        assert_eq!(
            longs.as_any().downcast_ref::<Int64Array>().unwrap(),
            &Int64Array::from(vec![-7, i64::MAX])
        );
    }

    #[test]
    fn a_double_beyond_the_float_range_is_infinite_as_in_pyspark() {
        // PySpark's Arrow conversion does not reject a finite double beyond the float range
        // (it becomes infinity), so the float encoder keeps that behavior.
        let array = values_to_arrow(&[&Value::Double(1e300)], &DataType::Float).unwrap();
        let floats = array.as_any().downcast_ref::<Float32Array>().unwrap();
        assert!(floats.value(0).is_infinite());
    }

    #[test]
    fn session_tags_add_and_remove() {
        let spark = session();
        spark.add_tag("test_tag").unwrap();
        let tags = spark.get_tags();
        assert!(tags.contains(&"test_tag".to_string()));

        spark.remove_tag("test_tag");
        let tags = spark.get_tags();
        assert!(!tags.contains(&"test_tag".to_string()));
    }

    #[test]
    fn tags_have_the_errors_of_pyspark() {
        // As PySpark Connect's `_throw_if_invalid_tag`: a comma, then an empty tag.
        for (tag, error_class) in [
            (",", "VALUE_ALLOWED"),
            ("a,b", "VALUE_ALLOWED"),
            ("", "VALUE_NOT_NON_EMPTY_STR"),
        ] {
            let error = SparkSession::check_tag(tag).unwrap_err();
            assert_eq!(error.get_error_class().as_deref(), Some(error_class));
            assert_eq!(
                error.get_message_parameters().unwrap()["arg_name"],
                "Spark Connect tag"
            );
        }
        assert!(SparkSession::check_tag("ok").is_ok());
    }

    #[test]
    fn session_tags_cannot_be_empty() {
        let spark = session();
        let result = spark.add_tag("");
        assert!(result.is_err());
    }

    #[test]
    fn session_tags_cannot_contain_comma() {
        let spark = session();
        let result = spark.add_tag("tag,with,comma");
        assert!(result.is_err());
    }

    #[test]
    fn session_tags_clear() {
        let spark = session();
        spark.add_tag("tag1").unwrap();
        spark.add_tag("tag2").unwrap();
        spark.clear_tags();
        assert!(spark.get_tags().is_empty());
    }

    #[test]
    fn session_tags_no_duplicates() {
        let spark = session();
        spark.add_tag("tag").unwrap();
        spark.add_tag("tag").unwrap();
        let tags = spark.get_tags();
        assert_eq!(tags.len(), 1);
    }

    #[test]
    fn session_clone_shares_state() {
        let spark1 = session();
        spark1.add_tag("test").unwrap();
        let spark2 = spark1.clone();
        // Both should have the same tag
        let tags = spark2.get_tags();
        assert!(tags.contains(&"test".to_string()));
    }

    #[test]
    fn a_weak_session_upgrades_while_a_handle_of_it_is_alive() {
        // As a Weak of an Arc: the handle gives back the session itself, of its state, while a
        // handle of it is alive, the session, a clone, or one that a DataFrame of it holds, and
        // None once they are all dropped.
        let spark = session();
        spark.add_tag("a").unwrap();
        let weak = spark.downgrade();
        let upgraded = weak.upgrade().expect("the session is alive");
        assert!(upgraded.is_same_session(&spark));
        assert_eq!(upgraded.get_tags(), vec!["a".to_string()]);
        drop(upgraded);
        let df = spark.range(3).unwrap();
        drop(spark);
        let upgraded = weak.upgrade().expect("its DataFrame holds the session");
        drop(df);
        assert!(weak.clone().upgrade().is_some(), "a handle of it is left");
        drop(upgraded);
        assert!(weak.upgrade().is_none());
        // A new session is a session of its own, which its handle does not keep.
        let spark = session();
        let new_session = spark.new_session().downgrade();
        assert!(new_session.upgrade().is_none());
        assert!(spark.downgrade().upgrade().is_some());
    }

    #[test]
    fn a_clone_of_a_session_is_one_reference_to_its_state() {
        // A clone, an upgraded weak handle and the session of a DataFrame are references to the
        // one state of the session, of all of its fields: a clone counts that state once, and
        // none of the parts that the state holds. A session is that one reference.
        assert_eq!(
            std::mem::size_of::<SparkSession>(),
            std::mem::size_of::<usize>()
        );
        let spark = session();
        assert_eq!(Arc::strong_count(&spark.inner), 1);
        let df = spark.range(3).unwrap();
        let client = Arc::strong_count(spark.client());
        let plan_ids = Arc::strong_count(spark.plan_ids_shared());
        let progress_handlers = Arc::strong_count(&spark.progress_handlers());
        let listener_bus = Arc::strong_count(&spark.listener_bus());
        let clone = spark.clone();
        let upgraded = spark.downgrade().upgrade().expect("the session is alive");
        for other in [&clone, &upgraded, &df.session] {
            assert!(Arc::ptr_eq(&spark.inner, &other.inner));
        }
        assert_eq!(Arc::strong_count(&spark.inner), 4);
        assert_eq!(Arc::strong_count(spark.client()), client);
        assert_eq!(Arc::strong_count(spark.plan_ids_shared()), plan_ids);
        assert_eq!(
            Arc::strong_count(&spark.progress_handlers()),
            progress_handlers
        );
        assert_eq!(Arc::strong_count(&spark.listener_bus()), listener_bus);
        drop((clone, upgraded, df));
        assert_eq!(Arc::strong_count(&spark.inner), 1);
    }

    #[test]
    fn sessions_and_their_weak_handles_are_send_and_sync() {
        // A session and its weak handles are sent to and shared by other threads, as the classes
        // of the Python binding that hold them, and the static list of the weak handles of its
        // exit hook, need.
        fn send_and_sync<T: Send + Sync>() {}
        send_and_sync::<SparkSession>();
        send_and_sync::<WeakSparkSession>();
    }

    #[test]
    fn a_weak_session_keeps_none_of_the_state_of_the_session() {
        // The handle holds none of the state of the session: its client, its listener bus and
        // the handlers that it holds, which may hold objects that hold the session, are dropped
        // with the last handle of the session.
        let spark = session();
        let marker = Arc::new(());
        let held = Arc::clone(&marker);
        spark.register_progress_handler(move |_| {
            let _ = &held;
        });
        let client = Arc::downgrade(spark.client());
        let listener_bus = Arc::downgrade(&spark.listener_bus());
        let weak = spark.downgrade();
        drop(spark);
        assert!(weak.upgrade().is_none());
        assert!(client.upgrade().is_none());
        assert!(listener_bus.upgrade().is_none());
        assert_eq!(Arc::strong_count(&marker), 1);
    }

    // createDataFrame input IPC fixtures.
    fn text_type() -> DataType {
        DataType::String {
            collation: "UTF8_BINARY".to_string(),
        }
    }

    fn schema(types: Vec<DataType>) -> DataType {
        DataType::Struct {
            fields: types
                .into_iter()
                .enumerate()
                .map(|(i, dt)| struct_field(&format!("schema_{i}"), dt))
                .collect(),
        }
    }

    fn row(values: Vec<Value>) -> Row {
        // Top-level conversion is positional, even when input and schema names differ.
        Row::new(
            (0..values.len()).map(|i| format!("input_{i}")).collect(),
            values,
        )
    }

    fn read_batch(rows: &[Row], schema: &DataType) -> RecordBatch {
        let data = rows_to_arrow_ipc(rows, schema).unwrap();
        let mut reader = StreamReader::try_new(Cursor::new(data), None).unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert!(reader.next().is_none());
        assert_eq!(batch.num_rows(), rows.len());
        batch
    }

    #[test]
    fn input_ipc_writes_day_time_intervals_as_microsecond_durations() {
        // As Spark's Arrow conversion of DayTimeIntervalType.
        let interval = DataType::DayTimeInterval {
            start_field: 0,
            end_field: 3,
        };
        let rows = vec![
            row(vec![Value::DayTimeInterval(86_402_000_003)]),
            row(vec![Value::Null]),
            row(vec![Value::DayTimeInterval(-999_995)]),
        ];
        let batch = read_batch(&rows, &schema(vec![interval]));
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::DurationMicrosecondArray>()
            .unwrap();
        assert_eq!(
            column.iter().collect::<Vec<_>>(),
            vec![Some(86_402_000_003), None, Some(-999_995)]
        );
    }

    #[test]
    fn input_ipc_preserves_scalar_coercion_and_nulls() {
        let schema = schema(vec![
            DataType::Integer,
            DataType::Float,
            DataType::Boolean,
            text_type(),
            DataType::Binary,
        ]);
        let rows = vec![
            row(vec![
                Value::Long(3),
                Value::Double(1.25),
                Value::Bool(true),
                Value::String("é\0名前".to_string()),
                Value::Binary(vec![0, 255, 1]),
            ]),
            row(vec![Value::Null; 5]),
            row(vec![
                Value::Long(-2_147_483_648),
                Value::Long(16_777_217),
                Value::Bool(false),
                Value::String(String::new()),
                Value::Binary(vec![]),
            ]),
        ];
        let batch = read_batch(&rows, &schema);
        assert_eq!(batch.num_columns(), 5);
        assert_eq!(batch.schema().field(0).name(), "schema_0");
        macro_rules! column_eq {
            ($index:expr, $array:ty, $values:expr) => {
                assert_eq!(
                    batch
                        .column($index)
                        .as_any()
                        .downcast_ref::<$array>()
                        .unwrap(),
                    &<$array>::from($values)
                );
            };
        }
        column_eq!(0, Int32Array, vec![Some(3), None, Some(i32::MIN)]);
        column_eq!(1, Float32Array, vec![Some(1.25), None, Some(16_777_216.0)]);
        column_eq!(2, BooleanArray, vec![Some(true), None, Some(false)]);
        column_eq!(3, StringArray, vec![Some("é\0名前"), None, Some("")]);
        column_eq!(
            4,
            BinaryArray,
            vec![Some(&[0_u8, 255, 1][..]), None, Some(&[][..])]
        );
    }

    #[test]
    fn input_ipc_sends_times_as_nanoseconds_as_pyspark() {
        // As PySpark's `to_arrow_type` of TimeType: time64[ns] of the nanoseconds since
        // midnight, whatever the precision.
        use arrow::array::Time64NanosecondArray;
        use arrow::datatypes::{DataType as A, TimeUnit};
        let schema = schema(vec![DataType::Time { precision: 3 }]);
        let rows = vec![
            row(vec![Value::Time(3_723_456_789_000)]),
            row(vec![Value::Null]),
        ];
        let batch = read_batch(&rows, &schema);
        assert_eq!(
            batch.schema().field(0).data_type(),
            &A::Time64(TimeUnit::Nanosecond)
        );
        assert_eq!(
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Time64NanosecondArray>()
                .unwrap(),
            &Time64NanosecondArray::from(vec![Some(3_723_456_789_000), None])
        );
    }

    #[test]
    fn input_ipc_sends_variants_spatial_and_udt_values_as_pyspark() {
        // As PySpark's `to_arrow_type`: the structs of a VARIANT's value and metadata, and of a
        // GEOMETRY's srid and WKB, which their metadata tags, and a UDT as its SQL type.
        use arrow::array::{Array, BinaryArray, Int32Array, ListArray, StructArray};
        let udt = DataType::Udt {
            type_str: "udt".to_string(),
            jvm_class: None,
            python_class: Some("pyspark.testing.objects.ExamplePointUDT".to_string()),
            serialized_python_class: None,
            sql_type: Some(Box::new(DataType::Array {
                element_type: Box::new(DataType::Double),
                contains_null: false,
            })),
        };
        let schema = schema(vec![
            DataType::Variant,
            DataType::Geometry { srid: 4326 },
            udt,
        ]);
        let geometry = Value::Struct(
            Row::new(
                vec!["srid".to_string(), "wkb".to_string()],
                vec![Value::Integer(4326), Value::Binary(vec![1, 2])],
            )
            .into(),
        );
        let rows = vec![
            row(vec![
                Value::Variant {
                    value: vec![12, 1],
                    metadata: vec![1, 0, 0],
                },
                geometry,
                Value::List(vec![Value::Double(1.0), Value::Double(2.0)]),
            ]),
            row(vec![Value::Null, Value::Null, Value::Null]),
        ];
        let batch = read_batch(&rows, &schema);
        let variant = batch
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let fields = variant.fields();
        assert_eq!(
            fields[1].metadata().get("variant").map(String::as_str),
            Some("true")
        );
        let metadata = variant
            .column(1)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        assert_eq!(metadata.value(0), [1, 0, 0]);
        assert!(variant.is_null(1));
        let geometry = batch
            .column(1)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let wkb_field = &geometry.fields()[1];
        assert_eq!(
            wkb_field.metadata().get("geometry").map(String::as_str),
            Some("true")
        );
        assert_eq!(
            wkb_field.metadata().get("srid").map(String::as_str),
            Some("4326")
        );
        let srid = geometry
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(srid.value(0), 4326);
        assert!(geometry.is_null(1));
        assert!(batch
            .column(2)
            .as_any()
            .downcast_ref::<ListArray>()
            .is_some());
    }

    #[test]
    fn input_ipc_sends_the_metadata_of_the_fields_of_the_sql_type_of_a_udt_as_pyspark() {
        // As PySpark's Arrow schema of local rows (`LocalDataToArrowConversion.convert`): no
        // metadata of a field of the schema or of a struct in it, which `_deduplicate_field_names`
        // drops, while it leaves a user-defined type as it is, whose SQL type `to_arrow_type`
        // converts with `json.dumps` of the metadata of each of its fields, at any depth, under
        // `SPARK::metadata::json`. Of large types too.
        use arrow::array::{Array, ListArray, StructArray};
        use arrow::datatypes::DataType as A;
        let field = |name: &str, data_type, json: &str| crate::types::StructField {
            name: name.to_string(),
            data_type,
            nullable: true,
            metadata: crate::types::Metadata::from_json(json).unwrap(),
        };
        let string = DataType::String {
            collation: "UTF8_BINARY".into(),
        };
        let inner = DataType::Struct {
            fields: vec![field("y", string, r#"{"10": "a", "9": 1e+300}"#)],
        };
        let sql_type = DataType::Struct {
            fields: vec![
                field(
                    "x",
                    DataType::Double,
                    r#"{"t": 74334700308107.62, "b": [1, 18446744073709551621]}"#,
                ),
                field(
                    "a",
                    DataType::Array {
                        element_type: Box::new(inner.clone()),
                        contains_null: true,
                    },
                    "{}",
                ),
            ],
        };
        let udt = DataType::Udt {
            type_str: "udt".to_string(),
            jvm_class: None,
            python_class: Some("pyspark.testing.objects.ExamplePointUDT".to_string()),
            serialized_python_class: None,
            sql_type: Some(Box::new(sql_type)),
        };
        let schema = DataType::Struct {
            fields: vec![
                field("u", udt, r#"{"top": 1}"#),
                field("s", inner, r#"{"o": 2}"#),
            ],
        };
        let y = |text: &str| {
            Value::Struct(Row::new(vec!["y".to_string()], vec![Value::String(text.into())]).into())
        };
        let x = Row::new(
            vec!["x".to_string(), "a".to_string()],
            vec![Value::Double(1.0), Value::List(vec![y("v")])],
        );
        let rows = vec![
            row(vec![Value::Struct(x.into()), y("w")]),
            row(vec![Value::Null, Value::Null]),
        ];
        let key = "SPARK::metadata::json";
        for large in [false, true] {
            let batch = rows_to_arrow_batch(&rows, &schema).unwrap();
            let batch = with_large_var_types(batch, &schema, large).unwrap();
            let data = batch_to_arrow_ipc(&batch).unwrap();
            let batch = StreamReader::try_new(Cursor::new(data), None)
                .unwrap()
                .next()
                .unwrap()
                .unwrap();
            let arrow_schema = batch.schema();
            assert!(arrow_schema
                .fields()
                .iter()
                .all(|f| f.metadata().is_empty()));
            let A::Struct(udt_fields) = arrow_schema.field(0).data_type() else {
                panic!("a UDT of a struct is a struct");
            };
            assert_eq!(
                udt_fields[0].metadata().get(key).map(String::as_str),
                Some(r#"{"t": 74334700308107.62, "b": [1, 18446744073709551621]}"#)
            );
            assert!(udt_fields[1].metadata().is_empty());
            let A::List(element) = udt_fields[1].data_type() else {
                panic!("an array is a list");
            };
            let A::Struct(element_fields) = element.data_type() else {
                panic!("a struct is a struct");
            };
            assert_eq!(
                element_fields[0].metadata().get(key).map(String::as_str),
                Some(r#"{"10": "a", "9": 1e+300}"#)
            );
            let string = if large { A::LargeUtf8 } else { A::Utf8 };
            assert_eq!(element_fields[0].data_type(), &string);
            let A::Struct(struct_fields) = arrow_schema.field(1).data_type() else {
                panic!("a struct is a struct");
            };
            assert!(struct_fields[0].metadata().is_empty());
            let udt_column = batch
                .column(0)
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap();
            assert!(udt_column.is_null(1));
            let list = udt_column
                .column(1)
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap();
            assert_eq!(list.value(0).len(), 1);
        }
    }

    #[test]
    fn columns_of_no_metadata_of_the_sql_type_of_a_udt_take_the_types_of_the_schema() {
        // `create_dataframe_from_columns` of a UDT column made without the metadata of the
        // fields of its SQL type, which the Arrow types of the schema hold, sends the types of the
        // schema; a column of another type fails as it did.
        use arrow::array::Int64Array;
        let sql_type = DataType::Struct {
            fields: vec![crate::types::StructField {
                name: "x".to_string(),
                data_type: DataType::Double,
                nullable: true,
                metadata: crate::types::Metadata::from_json(r#"{"m": 1}"#).unwrap(),
            }],
        };
        let udt = DataType::Udt {
            type_str: "udt".to_string(),
            jvm_class: None,
            python_class: None,
            serialized_python_class: None,
            sql_type: Some(Box::new(sql_type.clone())),
        };
        let schema = schema(vec![udt]);
        let arrow_schema = arrow::datatypes::Schema::new(schema_to_arrow_fields(&schema).unwrap());
        let point = Value::Struct(Row::new(vec!["x".to_string()], vec![Value::Double(1.0)]).into());
        let plain = values_to_arrow(&[&point, &Value::Null], &sql_type).unwrap();
        assert_ne!(plain.data_type(), arrow_schema.field(0).data_type());
        let columns = with_udt_field_metadata(vec![plain], &arrow_schema).unwrap();
        let batch = columns_to_arrow_batch(columns, arrow_schema.clone(), 2).unwrap();
        assert_eq!(
            batch.schema().field(0).data_type(),
            arrow_schema.field(0).data_type()
        );
        let other: arrow::array::ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        let columns = with_udt_field_metadata(vec![other], &arrow_schema).unwrap();
        assert!(columns_to_arrow_batch(columns, arrow_schema, 2).is_err());
    }

    #[test]
    fn input_ipc_keeps_the_duplicate_names_of_the_sql_type_of_a_udt_as_pyspark() {
        // As PySpark's Arrow schema of local rows, whose `_deduplicate_field_names` stops at a
        // user-defined type: the fields of its SQL type keep their names, also of duplicates,
        // which the server then fails to read, as PySpark's, where they were deduplicated; a
        // struct of the schema has deduplicated names.
        use arrow::array::{Int64Array, StructArray};
        use arrow::datatypes::DataType as A;
        let pair = DataType::Struct {
            fields: vec![
                struct_field("d", DataType::Long),
                struct_field("d", DataType::Long),
            ],
        };
        let udt = DataType::Udt {
            type_str: "udt".to_string(),
            jvm_class: None,
            python_class: None,
            serialized_python_class: None,
            sql_type: Some(Box::new(pair.clone())),
        };
        let value = || {
            Value::Struct(
                Row::new(
                    vec!["d".to_string(), "d".to_string()],
                    vec![Value::Long(1), Value::Long(2)],
                )
                .into(),
            )
        };
        let batch = read_batch(&[row(vec![value(), value()])], &schema(vec![udt, pair]));
        let names = |i: usize| match batch.schema().field(i).data_type() {
            A::Struct(fields) => fields.iter().map(|f| f.name().clone()).collect::<Vec<_>>(),
            other => panic!("not a struct: {other:?}"),
        };
        assert_eq!(names(0), ["d", "d"]);
        assert_eq!(names(1), ["d_0", "d_1"]);
        for i in 0..2 {
            let column = batch
                .column(i)
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap();
            let second = column.column(1).as_any().downcast_ref::<Int64Array>();
            assert_eq!(second.unwrap().value(0), 2);
        }
    }

    #[test]
    fn input_ipc_scales_nanos_timestamps_and_keeps_zone() {
        use arrow::array::{ListArray, TimestampNanosecondArray};
        use arrow::datatypes::{DataType as A, TimeUnit};
        let ntz = DataType::TimestampNtzNanos { precision: 9 };
        let ltz = DataType::TimestampLtzNanos { precision: 7 };
        let array = DataType::Array {
            element_type: Box::new(ltz.clone()),
            contains_null: true,
        };
        let schema = schema(vec![ntz, ltz, array]);
        let rows = vec![
            row(vec![
                Value::Timestamp(1_700_000_000_123_456),
                Value::Timestamp(-1),
                Value::List(vec![Value::Timestamp(2), Value::Null]),
            ]),
            row(vec![Value::Null, Value::Null, Value::List(vec![])]),
        ];
        let batch = read_batch(&rows, &schema);
        let column = |i: usize| {
            batch
                .column(i)
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap()
                .clone()
        };
        assert_eq!(
            batch.schema().field(0).data_type(),
            &A::Timestamp(TimeUnit::Nanosecond, None)
        );
        assert_eq!(
            batch.schema().field(1).data_type(),
            &A::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()))
        );
        assert_eq!(
            column(0),
            TimestampNanosecondArray::from(vec![Some(1_700_000_000_123_456_000), None])
        );
        assert_eq!(
            column(1),
            TimestampNanosecondArray::from(vec![Some(-1_000), None]).with_timezone("UTC")
        );
        let list = batch
            .column(2)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let first = list.value(0);
        assert_eq!(
            first
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap(),
            &TimestampNanosecondArray::from(vec![Some(2_000), None]).with_timezone("UTC")
        );
        assert_eq!(list.value_length(1), 0);

        // An empty input with a nanos schema builds too, as for the micros types.
        let empty = rows_to_arrow_ipc(
            &[],
            &schema_one(DataType::TimestampNtzNanos { precision: 9 }),
        );
        assert!(empty.is_ok(), "{empty:?}");

        let overflow = rows_to_arrow_ipc(
            &[row(vec![Value::Timestamp(i64::MAX)])],
            &schema_one(DataType::TimestampNtzNanos { precision: 9 }),
        )
        .unwrap_err()
        .to_string();
        assert!(overflow.contains("out of range"), "{overflow}");
    }

    fn schema_one(dt: DataType) -> DataType {
        schema(vec![dt])
    }

    #[test]
    fn input_ipc_resolves_duplicate_names_before_numeric_conversion() {
        let nested_type = DataType::Struct {
            fields: vec![
                struct_field("dup", DataType::Float),
                struct_field("positional", DataType::Short),
                struct_field("dup", DataType::Double),
            ],
        };
        let nested = Value::Struct(
            Row::new(
                vec!["dup".to_string(), "dup".to_string(), "other".to_string()],
                vec![
                    Value::Double(16_777_217.0),
                    Value::Long(999),
                    Value::Long(-2),
                ],
            )
            .into(),
        );
        let batch = read_batch(&[row(vec![nested])], &schema(vec![nested_type]));
        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(
            array
                .column(0)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .value(0),
            16_777_216.0
        );
        assert_eq!(
            array
                .column(1)
                .as_any()
                .downcast_ref::<Int16Array>()
                .unwrap()
                .value(0),
            999
        );
        // Both fields select the first source name, but each conversion must
        // use the original value: Float rounding must not affect Double.
        assert_eq!(
            array
                .column(2)
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap()
                .value(0),
            16_777_217.0
        );
    }

    #[test]
    fn input_ipc_reports_schema_and_column_errors() {
        let array = DataType::Array {
            element_type: Box::new(DataType::Long),
            contains_null: true,
        };
        let map = DataType::Map {
            key_type: Box::new(text_type()),
            value_type: Box::new(DataType::Long),
            value_contains_null: true,
        };
        for (rows, schema, message) in [
            (
                vec![row(vec![Value::Long(1)])],
                DataType::Long,
                "Schema is not a struct",
            ),
            (
                vec![row(vec![Value::String("bad".to_string())])],
                schema(vec![DataType::Integer]),
                "Type mismatch in row data",
            ),
            (
                vec![row(vec![Value::Long(1), Value::Long(2)])],
                schema(vec![array, map]),
                "Type mismatch: expected array",
            ),
            // Schema conversion still precedes the width check.
            (
                vec![row(vec![Value::Long(1)])],
                schema(vec![DataType::CalendarInterval, DataType::Long]),
                "Unsupported type for Arrow createDataFrame conversion",
            ),
        ] {
            let error = rows_to_arrow_ipc(&rows, &schema).unwrap_err().to_string();
            assert!(error.contains(message), "{error}");
        }
    }

    #[test]
    fn input_ipc_round_trips_nested_text_and_binary() {
        let list = |element_type| DataType::Array {
            element_type: Box::new(element_type),
            contains_null: true,
        };
        let map = |value_type| DataType::Map {
            key_type: Box::new(text_type()),
            value_type: Box::new(value_type),
            value_contains_null: true,
        };
        let nested_type = DataType::Struct {
            fields: vec![
                struct_field("binary", list(DataType::Binary)),
                struct_field("text", map(text_type())),
            ],
        };
        let schema = schema(vec![list(text_type()), map(DataType::Binary), nested_type]);
        let make_rows = || {
            vec![
                row(vec![
                    Value::List(vec![
                        Value::String("é\0名前".into()),
                        Value::String(String::new()),
                        Value::Null,
                    ]),
                    Value::Map(
                        [
                            (
                                Value::String("é\0名前".into()),
                                Value::Binary(vec![0, 255, 1]),
                            ),
                            (Value::String(String::new()), Value::Binary(vec![])),
                            (Value::String("null".into()), Value::Null),
                        ]
                        .into(),
                    ),
                    Value::Struct(
                        Row::new(
                            vec!["binary".into(), "text".into()],
                            vec![
                                Value::List(vec![
                                    Value::Binary(vec![0, 255]),
                                    Value::Binary(vec![]),
                                    Value::Null,
                                ]),
                                Value::Map(
                                    [
                                        (
                                            Value::String("key\0".into()),
                                            Value::String("名前\0é".into()),
                                        ),
                                        (
                                            Value::String(String::new()),
                                            Value::String(String::new()),
                                        ),
                                        (Value::String("null".into()), Value::Null),
                                    ]
                                    .into(),
                                ),
                            ],
                        )
                        .into(),
                    ),
                ]),
                row(vec![
                    Value::List(vec![]),
                    Value::Map(Default::default()),
                    Value::Null,
                ]),
                row(vec![Value::Null; 3]),
            ]
        };
        let rows = make_rows();
        let data = rows_to_arrow_ipc(&rows, &schema).unwrap();
        drop(rows);
        // The IPC is an owned buffer: decode it after the rows it was encoded from are gone.
        let response = proto::execute_plan_response::ArrowBatch {
            data: data.into(),
            ..Default::default()
        };
        let decoded = crate::dataframe::decode_arrow_batch(&response).unwrap();
        // Build the expected rows anew rather than keeping the originals.
        let expected = make_rows();
        assert_eq!(decoded.len(), expected.len());
        for (actual, expected) in decoded.iter().zip(expected) {
            assert_eq!(actual.values(), expected.values());
        }
    }

    #[test]
    fn input_ipc_preserves_nested_error_order() {
        let list = |element_type| DataType::Array {
            element_type: Box::new(element_type),
            contains_null: true,
        };
        // Validate all parent values before recursing into children.
        let rows = [
            row(vec![Value::List(vec![Value::String("bad integer".into())])]),
            row(vec![Value::Long(1)]),
        ];
        let error = rows_to_arrow_ipc(&rows, &schema(vec![list(DataType::Long)]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("Type mismatch: expected array"), "{error}");

        // Map key validation precedes map value validation, including nested schemas.
        let map_type = DataType::Map {
            key_type: Box::new(list(text_type())),
            value_type: Box::new(DataType::Long),
            value_contains_null: true,
        };
        let rows = [row(vec![Value::Map(
            [(
                Value::String("key".into()),
                Value::String("bad integer".into()),
            )]
            .into(),
        )])];
        let error = rows_to_arrow_ipc(&rows, &schema(vec![map_type]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("Type mismatch: expected array"), "{error}");
    }

    #[test]
    fn a_null_map_key_is_an_error() {
        let map = DataType::Map {
            key_type: Box::new(DataType::Long),
            value_type: Box::new(DataType::Long),
            value_contains_null: true,
        };
        let value = Value::Map(vec![(Value::Null, Value::Long(1))]);
        let error = values_to_arrow(&[&value], &map).unwrap_err().to_string();
        assert!(error.contains("A map key cannot be null"), "{error}");
    }

    #[test]
    fn struct_values_of_the_declared_fields_are_taken_by_position() {
        // Fields of the same name keep their own values.
        let fields = vec![
            struct_field("a", DataType::Long),
            struct_field("a", DataType::Long),
        ];
        let names = vec!["a".to_string(), "a".to_string()];
        let value = Value::Struct(Row::new(names, vec![Value::Long(1), Value::Long(2)]).into());
        let array = values_to_arrow(&[&value], &DataType::Struct { fields }).unwrap();
        let array = array.as_any().downcast_ref::<StructArray>().unwrap();
        let column = |i: usize| {
            let column = array
                .column(i)
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>();
            column.unwrap().value(0)
        };
        assert_eq!((column(0), column(1)), (1, 2));
    }

    #[test]
    fn struct_values_without_fields_keep_their_length() {
        let empty = Value::Struct(Row::empty().into());
        let array = values_to_arrow(
            &[&empty, &Value::Null],
            &DataType::Struct { fields: vec![] },
        );
        let array = array.unwrap();
        assert_eq!(array.len(), 2);
        assert!(array.is_valid(0));
        assert!(array.is_null(1));
    }

    #[test]
    fn input_ipc_preserves_empty_input_and_rows_without_columns() {
        assert!(rows_to_arrow_ipc(&[], &DataType::Long).unwrap().is_empty());
        // Rows without columns keep their number, as PySpark's table without columns does.
        let data = rows_to_arrow_ipc(&[Row::empty(), Row::empty()], &schema(vec![])).unwrap();
        let batches = StreamReader::try_new(Cursor::new(data), None)
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_columns(), 0);
        assert_eq!(batches[0].num_rows(), 2);
    }

    #[test]
    fn input_ipc_checks_all_widths_before_encoding() {
        let rows = [
            row(vec![
                Value::String("bad integer".to_string()),
                Value::Bool(true),
            ]),
            row(vec![Value::Long(1)]),
        ];
        let error = rows_to_arrow_ipc(&rows, &schema(vec![DataType::Integer, DataType::Boolean]))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("Row 1 has 1 values, but the schema has 2 fields"),
            "{error}"
        );
    }

    #[test]
    fn create_dataframe_rejects_short_and_long_rows() {
        let spark = session();
        for values in [vec![Value::Long(1)], vec![Value::Long(1); 3]] {
            let width = values.len();
            let result = spark.create_dataframe(
                vec![row(values)],
                schema(vec![DataType::Long, DataType::Long]),
            );
            let error = result.err().expect("expected a width error").to_string();
            assert!(
                error.contains(&format!(
                    "Row 0 has {width} values, but the schema has 2 fields"
                )),
                "{error}"
            );
        }
    }

    #[test]
    fn input_ipc_roundtrips_through_batch_decoder() {
        let nested_type = DataType::Struct {
            fields: vec![
                struct_field("count", DataType::Long),
                struct_field("text", text_type()),
            ],
        };
        let nested_value = Value::Struct(
            Row::new(
                vec!["count".to_string(), "text".to_string()],
                vec![Value::Long(3), Value::String("名前".to_string())],
            )
            .into(),
        );
        for (types, rows) in [
            (
                vec![DataType::Long, DataType::Long],
                vec![
                    row(vec![Value::Long(i64::MIN), Value::Null]),
                    row(vec![Value::Long(i64::MAX), Value::Long(0)]),
                ],
            ),
            (
                vec![DataType::Long, nested_type],
                vec![
                    row(vec![Value::Long(-1), nested_value]),
                    row(vec![Value::Null, Value::Null]),
                ],
            ),
        ] {
            let schema = schema(types);
            let data = rows_to_arrow_ipc(&rows, &schema).unwrap();
            let response = proto::execute_plan_response::ArrowBatch {
                data: data.into(),
                ..Default::default()
            };
            let decoded = crate::dataframe::decode_arrow_batch(&response).unwrap();
            assert_eq!(decoded.len(), rows.len());
            for (actual, expected) in decoded.iter().zip(&rows) {
                assert_eq!(
                    actual.fields(),
                    &["schema_0".to_string(), "schema_1".to_string()]
                );
                assert_eq!(actual.values(), expected.values());
            }
        }
    }

    #[test]
    fn a_session_whose_client_closed_is_stopped_and_is_not_released() {
        // Port 1: nothing listens, so a ReleaseSession that reached the wire would fail. As
        // PySpark's stop(), a client that a server-side session change closed is not released.
        let spark = SparkSession::builder()
            .remote("sc://localhost:1")
            .get_or_create()
            .expect("failed to build session");
        let other = spark.clone();
        spark.client().close();
        assert!(
            spark.is_stopped(),
            "a closed client makes the session stopped"
        );
        assert!(
            other.is_stopped(),
            "the closed client is shared across clones"
        );
        assert!(
            spark.stop().is_ok(),
            "stop() does not release a closed client"
        );
        assert!(spark.inner.stopped.load(Ordering::SeqCst));
    }

    #[test]
    fn host_and_server_session_id_are_the_clients_of_the_session() {
        let spark = session();
        assert_eq!(spark.host(), "localhost");
        assert_eq!(spark.server_session_id(), None);
        spark.set_server_session_id(Some("server-a".to_string()));
        assert_eq!(spark.server_session_id().as_deref(), Some("server-a"));
        // The session is a clone of the same client, which shares the observed ID.
        assert_eq!(
            spark.clone().server_session_id().as_deref(),
            Some("server-a")
        );
        spark.set_server_session_id(None);
        assert_eq!(spark.server_session_id(), None);
    }
}

/// The hook of the executions without a server: a session of a dead port whose requests are not
/// retried, since the plan compression of the first plan reads the confs of the server before
/// the hook runs, as PySpark does, and the default retry policy retries that read for minutes.
#[cfg(test)]
mod execute_plan_hook_tests {
    use super::*;

    fn unreachable_session_without_retries() -> SparkSession {
        let builder = ChannelBuilder::parse("sc://127.0.0.1:1").expect("valid URL");
        let mut client =
            block_on(SparkConnectClient::connect(&builder)).expect("failed to build client");
        client.set_retry_policy(spark_connect_core::retries::RetryPolicy::no_retries());
        SparkSession::new(client)
    }

    #[test]
    fn the_error_of_a_hook_is_the_error_of_the_execution() {
        let session = unreachable_session_without_retries();
        let seen: Arc<Mutex<Vec<proto::ExecutePlanRequest>>> = Arc::default();
        let requests = Arc::clone(&seen);
        let hook: ExecutePlanHook = Arc::new(move |request| {
            requests.lock().unwrap().push(request);
            Err(SparkError::value_msg("the hook failed"))
        });
        session.set_execute_plan_hook(Some(hook));
        let error = session.range(3).unwrap().collect().unwrap_err();
        assert!(error.to_string().contains("the hook failed"), "{error}");
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].session_id, session.session_id());
        assert!(seen[0].operation_id.is_some());
        assert!(seen[0].plan.is_some());
    }

    #[test]
    fn a_cleared_or_replaced_hook_is_not_called() {
        let session = unreachable_session_without_retries();
        let calls = Arc::new(Mutex::new(0));
        let counted = Arc::clone(&calls);
        let first: ExecutePlanHook = Arc::new(move |_| {
            *counted.lock().unwrap() += 1;
            Err(SparkError::value_msg("the first hook"))
        });
        session.set_execute_plan_hook(Some(first));
        session.set_execute_plan_hook(None);
        let second: ExecutePlanHook = Arc::new(|_| Err(SparkError::value_msg("the second hook")));
        session.set_execute_plan_hook(Some(second));
        let error = session.range(1).unwrap().collect().unwrap_err();
        assert!(error.to_string().contains("the second hook"), "{error}");
        assert_eq!(*calls.lock().unwrap(), 0);
    }
}
