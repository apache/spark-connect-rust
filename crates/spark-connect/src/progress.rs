//! The progress of the executions of a session, as PySpark Connect's
//! `pyspark.sql.connect.shell.progress`: the progress handlers of the session are called with
//! each progress of the server that has tasks, and once more, as done, when the execution ends.

use std::sync::Arc;

use spark_connect_core::client::{InflightHandle, ReattachableResponseStream};
use spark_connect_core::error::{Result, SparkError};
use spark_connect_core::runtime::CallbackGuard;
use spark_connect_proto as proto;

use crate::session::{FallibleProgressHandler, ProgressHandlers};

/// The progress of a stage of an execution, as PySpark's `StageInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageInfo {
    pub stage_id: i64,
    pub num_tasks: i64,
    pub num_completed_tasks: i64,
    pub num_bytes_read: i64,
    pub done: bool,
}

impl StageInfo {
    /// The stage of a progress of the server, as PySpark's `from_proto`.
    fn from_proto(stage: &proto::execute_plan_response::execution_progress::StageInfo) -> Self {
        StageInfo {
            stage_id: stage.stage_id,
            num_tasks: stage.num_tasks,
            num_completed_tasks: stage.num_completed_tasks,
            num_bytes_read: stage.input_bytes_read,
            done: stage.done,
        }
    }
}

/// A progress of an execution, of which PySpark calls each `ProgressHandler` with the keyword
/// arguments: the stages and the running tasks of the last progress of the server, the
/// operation, and whether the execution ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressUpdate {
    pub stages: Vec<StageInfo>,
    pub inflight_tasks: i64,
    pub operation_id: Option<String>,
    pub done: bool,
}

/// The error of a progress handler that no caller is left to return to.
#[derive(Debug)]
pub enum UnreturnedError<'a> {
    /// The error of the done call of an execution dropped before its end, as an iterator that
    /// its caller stopped reading: PySpark's generator reports the exception of its
    /// `Progress.__exit__` as unraisable then.
    DroppedDone(SparkError),
    /// The error of an execution that the error of its done call, `done`, replaced: as the
    /// exception of PySpark's `Progress.__exit__` replaces that of the body of its `with`, which
    /// it keeps as the `__context__` of the one it raises.
    Replaced {
        error: SparkError,
        done: &'a SparkError,
    },
}

type UnreturnedErrorHandler = Arc<dyn for<'a> Fn(UnreturnedError<'a>) + Send + Sync>;

static UNRETURNED_ERROR_HANDLER: std::sync::Mutex<Option<UnreturnedErrorHandler>> =
    std::sync::Mutex::new(None);

/// Set what becomes of the errors of progress handlers that no caller is left to return to (see
/// [`UnreturnedError`]), for every session; by default they are dropped. The Python client
/// reports a dropped execution's to `sys.unraisablehook`, as PySpark does.
pub fn set_unreturned_error_handler(
    handler: impl for<'a> Fn(UnreturnedError<'a>) + Send + Sync + 'static,
) {
    *UNRETURNED_ERROR_HANDLER
        .lock()
        .unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(handler));
}

fn hand_over(error: UnreturnedError<'_>) {
    let handler = UNRETURNED_ERROR_HANDLER
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    if let Some(handler) = handler {
        handler(error);
    }
}

/// The progress of one execution, as PySpark's `Progress` of the execution of a request.
pub(crate) struct ProgressTracker {
    handlers: ProgressHandlers,
    operation_id: Option<String>,
    stages: Vec<StageInfo>,
    inflight_tasks: i64,
    /// Whether the done call was made.
    finished: bool,
}

impl ProgressTracker {
    #[cfg(test)]
    pub(crate) fn new(
        session: &crate::session::SparkSession,
        operation_id: Option<String>,
    ) -> Self {
        Self::with_handlers(session.progress_handlers(), operation_id)
    }

    /// A tracker that calls `handlers`, which is empty for an execution that reports no progress.
    pub(crate) fn with_handlers(handlers: ProgressHandlers, operation_id: Option<String>) -> Self {
        ProgressTracker {
            handlers,
            operation_id,
            stages: Vec::new(),
            inflight_tasks: 0,
            finished: false,
        }
    }

    /// The done call of the handlers, as PySpark's `Progress.__exit__` makes it when its
    /// execution ends, also of an error: made once. Its error is the error of the execution, as
    /// the exception of `__exit__` is raised from the execution.
    pub(crate) fn finish(&mut self) -> Result<()> {
        if std::mem::replace(&mut self.finished, true) {
            return Ok(());
        }
        self.notify(true)
    }

    /// The error of an execution that ends with `error`, after the done call: the error of the
    /// done call replaces it, as the exception of PySpark's `Progress.__exit__` does.
    pub(crate) fn finish_after(&mut self, error: SparkError) -> SparkError {
        match self.finish() {
            Ok(()) => error,
            Err(done) => {
                // The error of a wait that a signal ended is not one that a caller returns: the
                // signal's exception replaces it, so nothing is handed over.
                if !spark_connect_core::client::is_wait_aborted(&error) {
                    hand_over(UnreturnedError::Replaced { error, done: &done });
                }
                done
            }
        }
    }

    /// A progress of the server, as PySpark's `Progress.update_ticks`: the handlers are called
    /// only for one that has tasks, whose stages and running tasks the last call, of `done`,
    /// has too. The error of a handler fails the execution, as the exception of a handler
    /// propagates out of PySpark's result loop.
    pub(crate) fn update(
        &mut self,
        progress: &proto::execute_plan_response::ExecutionProgress,
        operation_id: &str,
    ) -> Result<()> {
        if self.operation_id.as_deref().is_none_or(str::is_empty) {
            self.operation_id = Some(operation_id.to_string());
        }
        let stages: Vec<StageInfo> = progress.stages.iter().map(StageInfo::from_proto).collect();
        if stages.iter().map(|stage| stage.num_tasks).sum::<i64>() > 0 {
            self.stages = stages;
            self.inflight_tasks = progress.num_inflight_tasks;
            return self.notify(false);
        }
        Ok(())
    }

    fn notify(&self, done: bool) -> Result<()> {
        // Clone handlers out of the registry lock so callbacks can add or remove handlers.
        let handlers: Vec<FallibleProgressHandler> = self
            .handlers
            .lock()
            .unwrap()
            .iter()
            .map(|(_, handler)| Arc::clone(handler))
            .collect();
        if handlers.is_empty() {
            return Ok(());
        }
        let update = ProgressUpdate {
            stages: self.stages.clone(),
            inflight_tasks: self.inflight_tasks,
            operation_id: self.operation_id.clone(),
            done,
        };
        // A handler may run a query whose progress calls the handlers again. Preserve the
        // execution callback-depth guard without retaining the session or its listener bus.
        let _callback = CallbackGuard::enter();
        for handler in handlers {
            handler(&update)?;
        }
        Ok(())
    }
}

impl Drop for ProgressTracker {
    /// The done call of an execution dropped before its end, as PySpark's `Progress.finish` of a
    /// generator that its caller closed: an error of it has no caller left to return to, so it
    /// is handed over as [`UnreturnedError::DroppedDone`]. While a panic unwinds, a handler is
    /// not called: one that panics again would abort the process.
    fn drop(&mut self) {
        if !self.finished && !std::thread::panicking() {
            self.finished = true;
            if let Err(error) = self.notify(true) {
                hand_over(UnreturnedError::DroppedDone(error));
            }
        }
    }
}

/// The responses of an execution, with its progress for the handlers of the session: each
/// `ExecutionProgress` updates it, and it ends when the stream is dropped.
pub(crate) struct ExecutionStream {
    stream: ReattachableResponseStream,
    progress: ProgressTracker,
    chunks: crate::arrow_chunks::ArrowChunkAssembler,
}

impl ExecutionStream {
    pub(crate) fn new(stream: ReattachableResponseStream, progress: ProgressTracker) -> Self {
        ExecutionStream {
            stream,
            progress,
            chunks: Default::default(),
        }
    }

    /// The handle of the execution, to mark the waits of its caller for it (see
    /// [`ReattachableResponseStream::inflight_handle`]).
    pub(crate) fn inflight_handle(&self) -> InflightHandle {
        self.stream.inflight_handle()
    }

    /// The next response of the execution, as `ReattachableResponseStream::message`, with the
    /// chunks of an Arrow batch joined into the batch: the response of a chunk that is not the
    /// last one has no batch.
    /// When the execution ends, of its result or of an error, the handlers get their done call,
    /// whose error is the error of the execution, as PySpark's `Progress.__exit__` raises it.
    /// The handlers run on the thread that polls this; a synchronous caller uses
    /// [`ExecutionStream::next_message`], which calls them on its own thread.
    pub(crate) async fn message(&mut self) -> Result<Option<proto::ExecutePlanResponse>> {
        let response = self.stream.message().await;
        self.handle(response)
    }

    /// As [`ExecutionStream::message`], waiting on the calling thread, which the handlers are
    /// called on, as PySpark calls them on the thread that iterates the responses: also inside
    /// a `current_thread` runtime, where the response is awaited on a thread of its own.
    pub(crate) fn next_message(&mut self) -> Result<Option<proto::ExecutePlanResponse>> {
        let response = spark_connect_core::runtime::block_on(self.stream.message());
        // On a worker of a multi-thread runtime, the handlers block in place, as the wait did.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| self.handle(response))
            }
            _ => self.handle(response),
        }
    }

    /// `error`, of the caller of an execution that ends before its stream does, such as of the
    /// decoding of a batch: the handlers get their done call now, whose error replaces
    /// `error`, as PySpark's `Progress.__exit__` replaces an error raised in the `with` of the
    /// execution.
    pub(crate) fn fail(&mut self, error: SparkError) -> SparkError {
        self.progress.finish_after(error)
    }

    /// The response of the stream, `response`, after its progress has updated the handlers and
    /// its chunk is joined into its batch.
    fn handle(
        &mut self,
        response: Result<Option<proto::ExecutePlanResponse>>,
    ) -> Result<Option<proto::ExecutePlanResponse>> {
        let response = match response {
            Ok(response) => response,
            Err(error) => return Err(self.progress.finish_after(error)),
        };
        if let Some(response) = &response {
            if let Some(proto::execute_plan_response::ResponseType::ExecutionProgress(progress)) =
                &response.response_type
            {
                if let Err(error) = self.progress.update(progress, &response.operation_id) {
                    return Err(self.progress.finish_after(error));
                }
            }
        }
        match self.chunks.join(response) {
            Ok(None) => {
                self.progress.finish()?;
                Ok(None)
            }
            Ok(Some(response)) => Ok(Some(response)),
            Err(error) => Err(self.progress.finish_after(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::session::SparkSession;

    fn progress(
        stages: &[(i64, i64)],
        inflight: i64,
    ) -> proto::execute_plan_response::ExecutionProgress {
        proto::execute_plan_response::ExecutionProgress {
            stages: stages
                .iter()
                .map(|&(stage_id, num_tasks)| {
                    proto::execute_plan_response::execution_progress::StageInfo {
                        stage_id,
                        num_tasks,
                        num_completed_tasks: 1,
                        input_bytes_read: 8,
                        done: false,
                    }
                })
                .collect(),
            num_inflight_tasks: inflight,
        }
    }

    /// As PySpark's `Progress`: the handlers are called with each progress that has tasks, of the
    /// operation of the first response, and once more, as done, with the last one, when the
    /// execution ends.
    #[test]
    fn handlers_are_called_as_the_progress_of_pyspark_calls_them() {
        let session = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let updates = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&updates);
        let id = session.register_progress_handler(move |update| {
            seen.lock().unwrap().push(update.clone());
        });
        {
            let mut tracker = ProgressTracker::new(&session, None);
            tracker.update(&progress(&[(1, 0)], 0), "op").unwrap();
            tracker
                .update(&progress(&[(1, 4), (2, 2)], 3), "other")
                .unwrap();
        }
        let updates = updates.lock().unwrap().clone();
        assert_eq!(updates.len(), 2);
        assert!(!updates[0].done && updates[1].done);
        assert_eq!(updates[0].operation_id.as_deref(), Some("op"));
        assert_eq!(updates[1].stages, updates[0].stages);
        assert_eq!(updates[1].inflight_tasks, 3);
        assert_eq!(
            updates[0].stages[1],
            StageInfo {
                stage_id: 2,
                num_tasks: 2,
                num_completed_tasks: 1,
                num_bytes_read: 8,
                done: false,
            }
        );
        session.remove_progress_handler(id);
    }

    /// A handler that panics on every call, as a user's `unwrap` may: the panic of the update
    /// unwinds out of the tracker, and its drop must not call the handler again while
    /// unwinding, which would abort the process.
    #[test]
    fn a_handler_that_panics_does_not_abort_when_the_tracker_unwinds() {
        let session = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let id = session.register_progress_handler(move |_| {
            seen.fetch_add(1, Ordering::SeqCst);
            panic!("handler panic");
        });
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut tracker = ProgressTracker::new(&session, None);
            tracker.update(&progress(&[(1, 1)], 0), "op").unwrap();
        }));
        assert!(result.is_err());
        // Only the update called the handler; the done call during unwinding is skipped.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        session.remove_progress_handler(id);
    }

    /// A handler that fails, as a Python handler that raises: the update returns its error, the
    /// handlers after it are not called for that update, and the done call still reaches every
    /// handler when the tracker ends.
    #[test]
    fn a_failing_handler_fails_the_update_and_is_still_called_as_done() {
        let session = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let later = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&later);
        let failing = session.register_fallible_progress_handler(|update| {
            if update.done {
                Ok(())
            } else {
                Err(spark_connect_core::error::SparkError::connect_msg(
                    "handler failed",
                ))
            }
        });
        let after = session.register_progress_handler(move |update| {
            seen.lock().unwrap().push(update.done);
        });
        {
            let mut tracker = ProgressTracker::new(&session, None);
            let error = tracker
                .update(&progress(&[(1, 1)], 0), "op")
                .expect_err("the failing handler fails the update");
            assert_eq!(error.message(), "handler failed");
            assert!(later.lock().unwrap().is_empty());
        }
        assert_eq!(*later.lock().unwrap(), vec![true]);
        session.remove_progress_handler(failing);
        session.remove_progress_handler(after);
    }

    /// As PySpark's `Progress.__exit__`: the done call is made once and its error is the error
    /// of the execution, replacing an error of the execution, which is handed over; a tracker
    /// dropped before its end hands the error of its done call over.
    #[test]
    fn the_error_of_the_done_call_is_the_error_of_the_execution() {
        let session = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let done_calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&done_calls);
        let id = session.register_fallible_progress_handler(move |update| {
            if update.done {
                seen.fetch_add(1, Ordering::SeqCst);
                return Err(SparkError::connect_msg("s16 done failed"));
            }
            Ok(())
        });
        let handed = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&handed);
        set_unreturned_error_handler(move |error| {
            let (kind, error, done) = match error {
                UnreturnedError::DroppedDone(error) => ("dropped", error, String::new()),
                UnreturnedError::Replaced { error, done } => {
                    ("replaced", error, format!(" after {}", done.message()))
                }
            };
            if error.message().starts_with("s16") {
                sink.lock()
                    .unwrap()
                    .push(format!("{kind}: {}{done}", error.message()));
            }
        });

        let mut tracker = ProgressTracker::new(&session, None);
        assert_eq!(tracker.finish().unwrap_err().message(), "s16 done failed");
        assert!(tracker.finish().is_ok());
        drop(tracker);
        assert_eq!(done_calls.load(Ordering::SeqCst), 1);

        let mut tracker = ProgressTracker::new(&session, None);
        let error = tracker.finish_after(SparkError::connect_msg("s16 execution failed"));
        assert_eq!(error.message(), "s16 done failed");
        drop(tracker);
        drop(ProgressTracker::new(&session, None));
        assert_eq!(done_calls.load(Ordering::SeqCst), 3);
        assert_eq!(
            *handed.lock().unwrap(),
            vec![
                "replaced: s16 execution failed after s16 done failed",
                "dropped: s16 done failed"
            ]
        );
        session.remove_progress_handler(id);
    }

    #[test]
    fn detached_progress_registry_preserves_the_execution_recursion_bound() {
        use spark_connect_core::error::SparkErrorKind;
        use spark_connect_core::runtime::{check_callback_depth, MAX_CALLBACK_DEPTH};

        let session = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let handlers = Arc::downgrade(&session.progress_handlers());
        let calls = Arc::new(AtomicUsize::new(0));
        let failures = Arc::new(AtomicUsize::new(0));
        let seen_calls = Arc::clone(&calls);
        let seen_failures = Arc::clone(&failures);
        session.register_progress_handler(move |update| {
            if update.done {
                return;
            }
            let call = seen_calls.fetch_add(1, Ordering::SeqCst);
            match check_callback_depth() {
                Err(error) => {
                    assert_eq!(error.kind, SparkErrorKind::RecursionError);
                    seen_failures.fetch_add(1, Ordering::SeqCst);
                }
                Ok(()) => {
                    // Bound this test even if the guard regresses; no network execution or
                    // unbounded native recursion is needed to observe the missing guard.
                    assert!(call < MAX_CALLBACK_DEPTH);
                    let nested = ProgressTracker {
                        handlers: handlers.upgrade().expect("registry is alive"),
                        operation_id: Some("nested".to_string()),
                        stages: Vec::new(),
                        inflight_tasks: 0,
                        finished: false,
                    };
                    nested.notify(false).unwrap();
                }
            }
        });
        let tracker = ProgressTracker::new(&session, None);
        tracker.notify(false).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), MAX_CALLBACK_DEPTH);
        assert_eq!(failures.load(Ordering::SeqCst), 1);
        assert!(check_callback_depth().is_ok());
        session.clear_progress_handlers();
    }
}
