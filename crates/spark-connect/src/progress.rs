//! The progress of the executions of a session, as PySpark Connect's
//! `pyspark.sql.connect.shell.progress`: the progress handlers of the session are called with
//! each progress of the server that has tasks, and once more, as done, when the execution ends.

use spark_connect_core::client::ReattachableResponseStream;
use spark_connect_core::error::Result;
use spark_connect_proto as proto;

use crate::session::SparkSession;

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
    session: SparkSession,
    operation_id: Option<String>,
    stages: Vec<StageInfo>,
    inflight_tasks: i64,
}

impl ProgressTracker {
    pub(crate) fn new(session: &SparkSession, operation_id: Option<String>) -> Self {
        ProgressTracker {
            session: session.clone(),
            operation_id,
            stages: Vec::new(),
            inflight_tasks: 0,
        }
    }

    /// A progress of the server, as PySpark's `Progress.update_ticks`: the handlers are called
    /// only for one that has tasks, whose stages and running tasks the last call, of `done`,
    /// has too.
    pub(crate) fn update(
        &mut self,
        progress: &proto::execute_plan_response::ExecutionProgress,
        operation_id: &str,
    ) {
        if self.operation_id.as_deref().is_none_or(str::is_empty) {
            self.operation_id = Some(operation_id.to_string());
        }
        let stages: Vec<StageInfo> = progress.stages.iter().map(StageInfo::from_proto).collect();
        if stages.iter().map(|stage| stage.num_tasks).sum::<i64>() > 0 {
            self.stages = stages;
            self.inflight_tasks = progress.num_inflight_tasks;
            self.notify(false);
        }
    }

    fn notify(&self, done: bool) {
        self.session.notify_progress(|| ProgressUpdate {
            stages: self.stages.clone(),
            inflight_tasks: self.inflight_tasks,
            operation_id: self.operation_id.clone(),
            done,
        });
    }
}

impl Drop for ProgressTracker {
    /// As PySpark's `Progress.finish` when its execution ends, also of an error.
    fn drop(&mut self) {
        self.notify(true);
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
                self.progress.update(progress, &response.operation_id);
            }
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

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
            tracker.update(&progress(&[(1, 0)], 0), "op");
            tracker.update(&progress(&[(1, 4), (2, 2)], 3), "other");
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
}
