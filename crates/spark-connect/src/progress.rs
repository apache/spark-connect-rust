//! The progress of the executions of a session, as PySpark Connect's
//! `pyspark.sql.connect.shell.progress`: the progress handlers of the session are called with
//! each progress of the server that has tasks, and once more, as done, when the execution ends.

use std::sync::Arc;

use spark_connect_core::client::ReattachableResponseStream;
use spark_connect_core::error::Result;
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

/// The progress of one execution, as PySpark's `Progress` of the execution of a request.
pub(crate) struct ProgressTracker {
    handlers: ProgressHandlers,
    operation_id: Option<String>,
    stages: Vec<StageInfo>,
    inflight_tasks: i64,
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
    /// As PySpark's `Progress.finish` when its execution ends, also of an error. While a panic
    /// unwinds, a handler is not called: one that panics again would abort the process. An
    /// error of the done call has no caller left to return to, so it is dropped here; the
    /// handlers of the drop-in report theirs themselves.
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let _ = self.notify(true);
        }
    }
}

/// The responses of an execution, with its progress for the handlers of the session: each
/// `ExecutionProgress` updates it, and it ends when the stream is dropped.
pub(crate) struct ExecutionStream {
    stream: ReattachableResponseStream,
    progress: ProgressTracker,
}

impl ExecutionStream {
    pub(crate) fn new(stream: ReattachableResponseStream, progress: ProgressTracker) -> Self {
        ExecutionStream { stream, progress }
    }

    /// The next response of the execution, as `ReattachableResponseStream::message`.
    pub(crate) async fn message(&mut self) -> Result<Option<proto::ExecutePlanResponse>> {
        let response = self.stream.message().await?;
        if let Some(response) = &response {
            if let Some(proto::execute_plan_response::ResponseType::ExecutionProgress(progress)) =
                &response.response_type
            {
                self.progress.update(progress, &response.operation_id)?;
            }
        }
        Ok(response)
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
