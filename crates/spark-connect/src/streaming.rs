//! Structured Streaming support mirroring `pyspark.sql.connect.streaming`.
//!
//! Provides DataStreamReader, DataStreamWriter, StreamingQuery, and StreamingQueryManager
//! for building and executing streaming workloads.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use uuid;

use spark_connect_core::error::{Result, SparkError, SparkErrorKind};
use spark_connect_core::runtime::block_on;
use spark_connect_proto as proto;

use crate::dataframe::DataFrame;
use crate::plan::LogicalPlan;
use crate::readwriter::ReadType;
use crate::session::SparkSession;
use crate::udf::PythonUDFPayload;

/// DataStreamReader for reading streaming data from various sources.
///
/// Mirrors `pyspark.sql.connect.streaming.DataStreamReader`.
#[derive(Clone)]
pub struct DataStreamReader {
    session: SparkSession,
    format: Option<String>,
    schema: String,
    options: HashMap<String, String>,
    source_name: Option<String>,
}

impl DataStreamReader {
    /// Create a new DataStreamReader.
    pub(crate) fn new(session: SparkSession) -> Self {
        DataStreamReader {
            session,
            format: None,
            schema: String::new(),
            options: HashMap::new(),
            source_name: None,
        }
    }

    /// Set the format/source type (e.g., "rate", "socket", "kafka", "json", "parquet", "csv").
    pub fn format(mut self, source: &str) -> Self {
        self.format = Some(source.to_string());
        self
    }

    /// Unset the format, so that the server reads its default source
    /// (`spark.sql.sources.default`), as PySpark's `format(None)`.
    pub fn clear_format(mut self) -> Self {
        self.format = None;
        self
    }

    /// Set the schema from a DDL string or JSON string.
    pub fn schema(mut self, schema: impl Into<String>) -> Self {
        self.schema = schema.into();
        self
    }

    /// Set a single option key-value pair.
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_string(), value.to_string());
        self
    }

    /// Set multiple options.
    pub fn options(mut self, options: HashMap<String, String>) -> Self {
        self.options.extend(options);
        self
    }

    /// Set the source name for checkpoint stability.
    pub fn name(mut self, source_name: &str) -> Self {
        self.source_name = Some(source_name.to_string());
        self
    }

    /// The configured source format, if any.
    pub fn get_format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    /// The configured schema (a DDL/JSON string; empty when unset).
    pub fn get_schema(&self) -> &str {
        &self.schema
    }

    /// The configured read options.
    pub fn get_options(&self) -> &HashMap<String, String> {
        &self.options
    }

    /// Load streaming data from the specified path(s).
    pub fn load(self, path: Option<&str>) -> DataFrame {
        let paths = path.map(|p| vec![p.to_string()]);
        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format.clone(),
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema.clone())
                },
                options: self.options.clone(),
                paths: paths.unwrap_or_default(),
                predicates: vec![],
                source_name: self.source_name.clone(),
            },
            is_streaming: true,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read a named streaming table.
    pub fn table(self, table_name: &str) -> DataFrame {
        let plan = LogicalPlan::Read {
            read_type: ReadType::NamedTable {
                table_name: table_name.to_string(),
                options: self.options.clone(),
            },
            is_streaming: true,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read the streaming CDC changes of a named table. Mirrors
    /// `DataStreamReader.changes` (`is_streaming = true`).
    pub fn changes(self, table_name: &str) -> DataFrame {
        let plan = LogicalPlan::RelationChanges {
            table_name: table_name.to_string(),
            options: self.options.clone(),
            is_streaming: Some(true),
        };
        DataFrame::new(self.session, plan)
    }

    /// Read streaming JSON data from a path.
    pub fn json(mut self, path: &str) -> DataFrame {
        self.format = Some("json".to_string());
        let paths = vec![path.to_string()];
        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format.clone(),
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema.clone())
                },
                options: self.options.clone(),
                paths,
                predicates: vec![],
                source_name: self.source_name.clone(),
            },
            is_streaming: true,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read streaming Parquet data from a path.
    pub fn parquet(mut self, path: &str) -> DataFrame {
        self.format = Some("parquet".to_string());
        let paths = vec![path.to_string()];
        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format.clone(),
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema.clone())
                },
                options: self.options.clone(),
                paths,
                predicates: vec![],
                source_name: self.source_name.clone(),
            },
            is_streaming: true,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read streaming CSV data from a path.
    pub fn csv(mut self, path: &str) -> DataFrame {
        self.format = Some("csv".to_string());
        let paths = vec![path.to_string()];
        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format.clone(),
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema.clone())
                },
                options: self.options.clone(),
                paths,
                predicates: vec![],
                source_name: self.source_name.clone(),
            },
            is_streaming: true,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read streaming ORC data from a path.
    pub fn orc(mut self, path: &str) -> DataFrame {
        self.format = Some("orc".to_string());
        let paths = vec![path.to_string()];
        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format.clone(),
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema.clone())
                },
                options: self.options.clone(),
                paths,
                predicates: vec![],
                source_name: self.source_name.clone(),
            },
            is_streaming: true,
        };
        DataFrame::new(self.session, plan)
    }

    /// Read streaming text data from a path.
    pub fn text(mut self, path: &str) -> DataFrame {
        self.format = Some("text".to_string());
        let paths = vec![path.to_string()];
        let plan = LogicalPlan::Read {
            read_type: ReadType::DataSource {
                format: self.format.clone(),
                schema: if self.schema.is_empty() {
                    None
                } else {
                    Some(self.schema.clone())
                },
                options: self.options.clone(),
                paths,
                predicates: vec![],
                source_name: self.source_name.clone(),
            },
            is_streaming: true,
        };
        DataFrame::new(self.session, plan)
    }
}

/// Trigger for streaming queries.
#[derive(Debug, Clone)]
pub enum Trigger {
    /// Process every `interval` milliseconds or duration string (e.g., "10 seconds").
    ProcessingTime(String),
    /// Process only once.
    Once,
    /// Process as soon as available data arrives.
    AvailableNow,
    /// Continuous processing with checkpoint interval.
    Continuous(String),
    /// Real-time processing, of batches of the duration (e.g., "5 seconds"), as PySpark's
    /// `trigger(realTime=...)`.
    RealTime(String),
}

impl Trigger {
    /// The trigger of a `WriteStreamOperationStart`.
    pub(crate) fn to_proto(&self) -> proto::write_stream_operation_start::Trigger {
        use proto::write_stream_operation_start::Trigger as Proto;
        match self {
            Trigger::ProcessingTime(interval) => Proto::ProcessingTimeInterval(interval.clone()),
            Trigger::Once => Proto::Once(true),
            Trigger::AvailableNow => Proto::AvailableNow(true),
            Trigger::Continuous(interval) => Proto::ContinuousCheckpointInterval(interval.clone()),
            Trigger::RealTime(duration) => Proto::RealTimeBatchDuration(duration.clone()),
        }
    }
}

/// DataStreamWriter for writing streaming data to various sinks.
///
/// Mirrors `pyspark.sql.connect.streaming.DataStreamWriter`.
#[derive(Clone)]
pub struct DataStreamWriter {
    session: SparkSession,
    plan: Arc<LogicalPlan>,
    format: Option<String>,
    output_mode: Option<String>,
    options: HashMap<String, String>,
    partitioning_columns: Vec<String>,
    clustering_columns: Vec<String>,
    query_name: Option<String>,
    trigger: Option<Trigger>,
    path: Option<String>,
    table_name: Option<String>,
    foreach_batch_payload: Option<PythonUDFPayload>,
    foreach_payload: Option<PythonUDFPayload>,
}

impl DataStreamWriter {
    #[cfg(test)]
    pub(crate) fn plan(&self) -> &LogicalPlan {
        self.plan.as_ref()
    }

    /// Create a new DataStreamWriter.
    pub(crate) fn new(session: SparkSession, plan: Arc<LogicalPlan>) -> Self {
        DataStreamWriter {
            session,
            plan,
            format: None,
            output_mode: None,
            options: HashMap::new(),
            partitioning_columns: vec![],
            clustering_columns: vec![],
            query_name: None,
            trigger: None,
            path: None,
            table_name: None,
            foreach_batch_payload: None,
            foreach_payload: None,
        }
    }

    /// Set the output mode ("append", "update", "complete").
    pub fn output_mode(mut self, mode: &str) -> Self {
        self.output_mode = Some(mode.to_string());
        self
    }

    /// Set the format/sink type (e.g., "parquet", "json", "csv", "console", "noop", "kafka").
    pub fn format(mut self, source: &str) -> Self {
        self.format = Some(source.to_string());
        self
    }

    /// Set a single option key-value pair.
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_string(), value.to_string());
        self
    }

    /// Set multiple options.
    pub fn options(mut self, options: HashMap<String, String>) -> Self {
        self.options.extend(options);
        self
    }

    /// Set partitioning columns.
    pub fn partition_by(mut self, columns: Vec<&str>) -> Self {
        self.partitioning_columns = columns.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set clustering columns.
    pub fn cluster_by(mut self, columns: Vec<&str>) -> Self {
        self.clustering_columns = columns.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set the query name.
    pub fn query_name(mut self, name: &str) -> Self {
        self.query_name = Some(name.to_string());
        self
    }

    /// Set the trigger type.
    pub fn trigger(mut self, trigger: Trigger) -> Self {
        self.trigger = Some(trigger);
        self
    }

    /// Set a foreach batch function (PythonUDF payload).
    pub fn foreach_batch(mut self, payload: PythonUDFPayload) -> Self {
        self.foreach_batch_payload = Some(payload);
        self
    }

    /// Set a foreach function (PythonUDF payload).
    pub fn foreach(mut self, payload: PythonUDFPayload) -> Self {
        self.foreach_payload = Some(payload);
        self
    }

    /// Start the streaming query writing to a path, returning a StreamingQuery handle.
    pub fn start(self, path: &str) -> Result<StreamingQuery> {
        self.start_with_event_handler(path, |manager, event| {
            manager.post_to_all(event);
            Ok(())
        })
    }

    /// Start a query with a language-specific handler for its synchronous started event.
    /// The handler runs even when no listeners are registered and must deserialize the event
    /// before dispatching it. A handler error is returned instead of the query handle.
    pub fn start_with_event_handler(
        mut self,
        path: &str,
        on_started: impl FnOnce(&StreamingQueryManager, &StreamingQueryListenerEvent) -> Result<()>,
    ) -> Result<StreamingQuery> {
        self.path = Some(path.to_string());
        self._start_internal(on_started)
    }

    /// Start the streaming query writing to a table, returning a StreamingQuery handle.
    pub fn to_table(self, table_name: &str) -> Result<StreamingQuery> {
        self.to_table_with_event_handler(table_name, |manager, event| {
            manager.post_to_all(event);
            Ok(())
        })
    }

    /// The table-writing counterpart of
    /// [`start_with_event_handler`](Self::start_with_event_handler).
    pub fn to_table_with_event_handler(
        mut self,
        table_name: &str,
        on_started: impl FnOnce(&StreamingQueryManager, &StreamingQueryListenerEvent) -> Result<()>,
    ) -> Result<StreamingQuery> {
        self.table_name = Some(table_name.to_string());
        self._start_internal(on_started)
    }

    /// Internal method to build and execute the write stream command.
    fn _start_internal(
        self,
        on_started: impl FnOnce(&StreamingQueryManager, &StreamingQueryListenerEvent) -> Result<()>,
    ) -> Result<StreamingQuery> {
        let mut write_op = proto::WriteStreamOperationStart::default();
        write_op.input = Some(crate::plan::relation_with_ids(
            &self.plan,
            self.session.plan_ids(),
        )?);

        if let Some(fmt) = &self.format {
            write_op.format = fmt.clone();
        }

        if let Some(mode) = &self.output_mode {
            write_op.output_mode = mode.clone();
        }

        write_op.options = self.options;
        write_op.partitioning_column_names = self.partitioning_columns;
        write_op.clustering_column_names = self.clustering_columns;

        if let Some(name) = &self.query_name {
            write_op.query_name = name.clone();
        }

        write_op.trigger = self.trigger.as_ref().map(Trigger::to_proto);

        if let Some(path) = &self.path {
            // A memory/console/foreach sink has no path; only set the destination for a
            // real (non-empty) path so those sinks send an unset sink_destination.
            if !path.is_empty() {
                write_op.sink_destination = Some(
                    proto::write_stream_operation_start::SinkDestination::Path(path.clone()),
                );
            }
        }

        if let Some(table) = &self.table_name {
            write_op.sink_destination = Some(
                proto::write_stream_operation_start::SinkDestination::TableName(table.clone()),
            );
        }

        if let Some(foreach_batch) = &self.foreach_batch_payload {
            let mut foreach_func = proto::StreamingForeachFunction::default();
            foreach_func.function =
                Some(proto::streaming_foreach_function::Function::PythonFunction(
                    foreach_batch.to_proto(),
                ));
            write_op.foreach_batch = Some(foreach_func);
        }

        if let Some(foreach) = &self.foreach_payload {
            let mut foreach_func = proto::StreamingForeachFunction::default();
            foreach_func.function = Some(
                proto::streaming_foreach_function::Function::PythonFunction(foreach.to_proto()),
            );
            write_op.foreach_writer = Some(foreach_func);
        }

        // Run the WriteStreamOperationStart command, as other commands are, on the
        // reattachable path with the session's tags, so that an error carries the operation
        // id of the request, as in PySpark. Its WriteStreamOperationStartResult carries the
        // server-assigned query id / run id / name.
        let responses = crate::dataframe::execute_command_collect(
            &self.session,
            proto::command::CommandType::WriteStreamOperationStart(write_op),
        )?;
        let mut query_id = String::new();
        let mut run_id = String::new();
        let mut name = self.query_name.clone();
        let mut started_event = None;
        for resp in responses {
            if let Some(
                proto::execute_plan_response::ResponseType::WriteStreamOperationStartResult(res),
            ) = resp.response_type
            {
                if let Some(qid) = res.query_id {
                    query_id = qid.id;
                    run_id = qid.run_id;
                }
                if !res.name.is_empty() {
                    name = Some(res.name);
                }
                started_event = res.query_started_event_json;
            }
        }
        if query_id.is_empty() {
            return Err(SparkError::connect_msg(
                "writeStream.start: server returned no WriteStreamOperationStartResult",
            ));
        }
        // As PySpark Connect's DataStreamWriter: the server sends the started event of the query
        // with the result of its start, as the session has client-side listeners, and the client
        // posts it to them before it gives back the query.
        if let Some(event_json) = started_event {
            let manager = self.session.streams();
            // Mark the delivery before its listener snapshot is taken, so that a removal
            // from another thread waits for it, as PySpark's bus holds its lock for it.
            let _dispatch = manager.started_dispatch();
            on_started(
                &manager,
                &StreamingQueryListenerEvent {
                    event_type: QUERY_STARTED_EVENT,
                    event_json,
                },
            )?;
        }

        Ok(StreamingQuery {
            session: self.session,
            query_id,
            run_id,
            name,
        })
    }
}

/// A handle to an active streaming query.
///
/// Mirrors `pyspark.sql.connect.streaming.StreamingQuery`.
#[derive(Clone)]
pub struct StreamingQuery {
    session: SparkSession,
    query_id: String,
    run_id: String,
    name: Option<String>,
}

impl StreamingQuery {
    /// Get the query ID.
    pub fn id(&self) -> &str {
        &self.query_id
    }

    /// Get the run ID.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Get the query name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Check if the query is actively running.
    pub fn is_active(&self) -> Result<bool> {
        self._fetch_status().map(|status| status.is_active)
    }

    /// Get the current status of the query.
    pub fn status(&self) -> Result<StreamingQueryStatus> {
        self._fetch_status()
    }

    /// Stop the streaming query.
    pub fn stop(&self) -> Result<()> {
        let mut cmd = proto::StreamingQueryCommand::default();
        cmd.command = Some(proto::streaming_query_command::Command::Stop(true));
        self._execute_command(cmd)?;
        Ok(())
    }

    /// Wait for the query to terminate with optional timeout in seconds.
    pub fn await_termination(&self, timeout_sec: Option<f64>) -> Result<Option<bool>> {
        let mut cmd = proto::StreamingQueryCommand::default();
        let mut await_term = proto::streaming_query_command::AwaitTerminationCommand::default();

        if let Some(timeout) = timeout_sec {
            if timeout <= 0.0 {
                return Err(SparkError::value(
                    "INVALID_TIMEOUT",
                    &[("value", &timeout.to_string())],
                ));
            }
            await_term.timeout_ms = Some((timeout * 1000.0) as i64);
        }

        cmd.command = Some(proto::streaming_query_command::Command::AwaitTermination(
            await_term,
        ));

        let result = self._execute_command(cmd)?;

        if let Some(proto::streaming_query_command_result::ResultType::AwaitTermination(
            await_result,
        )) = result.result_type
        {
            if timeout_sec.is_some() {
                Ok(Some(await_result.terminated))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Get the last streaming progress, if available.
    pub fn last_progress(&self) -> Result<Option<String>> {
        let mut cmd = proto::StreamingQueryCommand::default();
        cmd.command = Some(proto::streaming_query_command::Command::LastProgress(true));

        let result = self._execute_command(cmd)?;

        if let Some(proto::streaming_query_command_result::ResultType::RecentProgress(progress)) =
            result.result_type
        {
            if let Some(progress_result) = progress.recent_progress_json.last() {
                return Ok(Some(progress_result.clone()));
            }
        }

        Ok(None)
    }

    /// Get recent streaming progress results.
    pub fn recent_progress(&self) -> Result<Vec<String>> {
        let mut cmd = proto::StreamingQueryCommand::default();
        cmd.command = Some(proto::streaming_query_command::Command::RecentProgress(
            true,
        ));

        let result = self._execute_command(cmd)?;

        if let Some(proto::streaming_query_command_result::ResultType::RecentProgress(progress)) =
            result.result_type
        {
            Ok(progress.recent_progress_json)
        } else {
            Ok(vec![])
        }
    }

    /// Process all available data in the streaming query.
    pub fn process_all_available(&self) -> Result<()> {
        let mut cmd = proto::StreamingQueryCommand::default();
        cmd.command = Some(proto::streaming_query_command::Command::ProcessAllAvailable(true));
        self._execute_command(cmd)?;
        Ok(())
    }

    /// Print the execution plan of the streaming query.
    pub fn explain(&self, extended: bool) -> Result<String> {
        let mut cmd = proto::StreamingQueryCommand::default();
        let mut explain = proto::streaming_query_command::ExplainCommand::default();
        explain.extended = extended;
        cmd.command = Some(proto::streaming_query_command::Command::Explain(explain));

        let result = self._execute_command(cmd)?;

        if let Some(proto::streaming_query_command_result::ResultType::Explain(explain_result)) =
            result.result_type
        {
            Ok(explain_result.result)
        } else {
            Ok(String::new())
        }
    }

    /// Get any exception that occurred in the streaming query.
    pub fn exception(&self) -> Result<Option<StreamingQueryException>> {
        let mut cmd = proto::StreamingQueryCommand::default();
        cmd.command = Some(proto::streaming_query_command::Command::Exception(true));

        let result = self._execute_command(cmd)?;

        if let Some(proto::streaming_query_command_result::ResultType::Exception(exc)) =
            result.result_type
        {
            if let Some(msg) = exc.exception_message {
                if !msg.is_empty() {
                    return Ok(Some(StreamingQueryException {
                        message: msg,
                        error_class: exc.error_class.unwrap_or_default(),
                        stack_trace: exc.stack_trace,
                    }));
                }
            }
        }

        Ok(None)
    }

    /// Fetch the current status of the query.
    fn _fetch_status(&self) -> Result<StreamingQueryStatus> {
        let mut cmd = proto::StreamingQueryCommand::default();
        cmd.command = Some(proto::streaming_query_command::Command::Status(true));

        let result = self._execute_command(cmd)?;

        if let Some(proto::streaming_query_command_result::ResultType::Status(status)) =
            result.result_type
        {
            Ok(StreamingQueryStatus {
                is_active: status.is_active,
                status_message: status.status_message,
                is_data_available: status.is_data_available,
                is_trigger_active: status.is_trigger_active,
            })
        } else {
            Err(SparkError::connect_msg(
                "Missing status in StreamingQueryCommandResult",
            ))
        }
    }

    /// Execute a streaming query command and return the parsed result.
    ///
    /// The server replies on the execute-plan stream with a
    /// `StreamingQueryCommandResult` in the `response_type` oneof; we drain the
    /// stream (via the shared collector, so metrics/progress are captured too) and
    /// return the first such result. Earlier this discarded the stream and returned
    /// a default, so every status/isActive/explain/exception/progress call saw an
    /// empty result — status/isActive then failed with "Missing status".
    fn _execute_command(
        &self,
        mut cmd: proto::StreamingQueryCommand,
    ) -> Result<proto::StreamingQueryCommandResult> {
        let mut query_id = proto::StreamingQueryInstanceId::default();
        query_id.id = self.query_id.clone();
        query_id.run_id = self.run_id.clone();
        cmd.query_id = Some(query_id);

        let responses = crate::dataframe::execute_command_collect(
            &self.session,
            proto::command::CommandType::StreamingQueryCommand(cmd),
        )?;

        for resp in responses {
            if let Some(proto::execute_plan_response::ResponseType::StreamingQueryCommandResult(
                result,
            )) = resp.response_type
            {
                return Ok(result);
            }
        }

        Ok(proto::StreamingQueryCommandResult::default())
    }
}

/// Status information for a streaming query.
#[derive(Debug, Clone)]
pub struct StreamingQueryStatus {
    pub is_active: bool,
    pub status_message: String,
    pub is_data_available: bool,
    pub is_trigger_active: bool,
}

/// Exception information for a streaming query.
#[derive(Debug, Clone)]
pub struct StreamingQueryException {
    pub message: String,
    pub error_class: String,
    /// The JVM stack trace of the exception, if the server sends it.
    pub stack_trace: Option<String>,
}

/// An iterator over streaming query listener events from the server.
/// Yields events incrementally as they arrive, without buffering the entire stream.
pub struct ListenerEventStream {
    stream: crate::progress::ExecutionStream,
    buffered_events: std::vec::IntoIter<(i32, String)>,
    done: bool,
    cancel: Option<tokio::sync::oneshot::Receiver<()>>,
}

impl ListenerEventStream {
    /// Read the next response unless canceled, checking cancellation before a ready response.
    /// Cancellation consumes its oneshot receiver and ends the stream like EOF; once done,
    /// the transport is never polled again.
    fn message(&mut self) -> Result<Option<proto::ExecutePlanResponse>> {
        if self.done {
            return Ok(None);
        }
        let response = block_on(async {
            if let Some(cancel) = &mut self.cancel {
                tokio::select! {
                    biased;
                    _ = cancel => None,
                    response = self.stream.message() => Some(response),
                }
            } else {
                Some(self.stream.message().await)
            }
        });
        match response {
            None | Some(Ok(None)) => {
                // A completed oneshot must never be polled again.
                self.cancel.take();
                self.done = true;
                Ok(None)
            }
            Some(Err(error)) => {
                self.done = true;
                Err(error)
            }
            Some(response) => response,
        }
    }

    /// Read the responses of the server until the one that tells that it added the listener of
    /// the session (`listener_bus_listener_added`), keeping the events of them, which the
    /// iterator gives first, as PySpark Connect's `_register_server_side_listener` waits for it.
    fn wait_until_listener_added(&mut self) -> Result<()> {
        loop {
            match self.message() {
                Ok(Some(resp)) => {
                    if let Some(
                        proto::execute_plan_response::ResponseType::StreamingQueryListenerEventsResult(res),
                    ) = resp.response_type
                    {
                        if !res.events.is_empty() {
                            let mut events: Vec<(i32, String)> =
                                std::mem::take(&mut self.buffered_events).collect();
                            events.extend(
                                res.events
                                    .into_iter()
                                    .map(|event| (event.event_type, event.event_json)),
                            );
                            self.buffered_events = events.into_iter();
                        }
                        if res.listener_bus_listener_added.is_some() {
                            return Ok(());
                        }
                    }
                }
                Ok(None) => {
                    // Terminal cancellation ends this wait like EOF. Teardown marks the
                    // session stopped first; add_listener's post-wait check rejects this
                    // outcome instead of publishing a worker on the canceled stream.
                    self.done = true;
                    return Ok(());
                }
                Err(e) => {
                    self.done = true;
                    return Err(e);
                }
            }
        }
    }
}

impl Iterator for ListenerEventStream {
    type Item = Result<(i32, String)>;

    fn next(&mut self) -> Option<Self::Item> {
        // First, yield any buffered events from the last response
        if let Some(event) = self.buffered_events.next() {
            return Some(Ok(event));
        }

        if self.done {
            return None;
        }

        loop {
            match self.message() {
                Ok(Some(resp)) => {
                    if let Some(
                        proto::execute_plan_response::ResponseType::StreamingQueryListenerEventsResult(res),
                    ) = resp.response_type
                    {
                        if !res.events.is_empty() {
                            let mut events = vec![];
                            for event in res.events {
                                events.push((event.event_type, event.event_json));
                            }
                            self.buffered_events = events.into_iter();
                            // Yield the first buffered event
                            if let Some(event) = self.buffered_events.next() {
                                return Some(Ok(event));
                            }
                        }
                    }
                    // Keep pulling for events if this response had none
                }
                Ok(None) => {
                    self.done = true;
                    return None;
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        }
    }
}

/// Event-type constants for listener events (mirror pyspark's values).
pub const QUERY_PROGRESS_EVENT: i32 = 1;
pub const QUERY_TERMINATED_EVENT: i32 = 2;
pub const QUERY_IDLE_EVENT: i32 = 3;

/// The started event of a query. The server sends it with the result of the start of the query
/// (`query_started_event_json`), when the session has a client-side listener, rather than on the
/// listener bus, and the client posts it to its listeners as the query starts, as PySpark
/// Connect's `DataStreamWriter.start` does: so it is not of the server's
/// `StreamingQueryEventType`.
pub const QUERY_STARTED_EVENT: i32 = -1;

/// A streaming-query listener event delivered by the client-side listener bus.
///
/// `event_json` is the server-provided JSON for the event; `event_type` is one of
/// [`QUERY_STARTED_EVENT`], [`QUERY_PROGRESS_EVENT`], [`QUERY_TERMINATED_EVENT`],
/// [`QUERY_IDLE_EVENT`].
#[derive(Debug, Clone)]
pub struct StreamingQueryListenerEvent {
    pub event_type: i32,
    pub event_json: String,
}

/// A client-side listener for streaming-query events. Implement this trait (Rust
/// clients) and register it with [`StreamingQueryManager::add_listener`]; the manager
/// runs a background bus that streams events from the server and dispatches them here.
pub trait StreamingQueryListener: Send + Sync {
    fn on_event(&self, event: &StreamingQueryListenerEvent);

    /// Optionally handle an asynchronous bus event for the whole listener snapshot, for
    /// adapters that deserialize a shared event object once before invoking any callback.
    /// Synchronous started events instead use [`DataStreamWriter::start_with_event_handler`] or
    /// [`DataStreamWriter::to_table_with_event_handler`] for shared deserialization.
    /// The first adapter returning `Some` owns dispatch for every entry, including native
    /// listeners and duplicates. Returning `None` must not invoke callbacks; the bus then
    /// tries the next adapter or falls back to each listener's
    /// [`try_on_event`](Self::try_on_event).
    fn try_post_to_all(
        &self,
        _listeners: &[Arc<dyn StreamingQueryListener>],
        _event: &StreamingQueryListenerEvent,
    ) -> Option<Result<()>> {
        None
    }

    /// Deserialize and dispatch an asynchronous event. An error is any failure the adapter
    /// does not isolate, including protocol/deserialization errors and adapter-defined
    /// unisolated callback failures. It terminates this subscription and clears its registrations.
    fn try_on_event(&self, event: &StreamingQueryListenerEvent) -> Result<()> {
        self.on_event(event);
        Ok(())
    }

    /// Report a terminal subscription failure from transport, deserialization or dispatch.
    /// Return true when reported, so the bus emits only one notification even if a listener
    /// is registered more than once. The bus invokes this without its state lock, before
    /// clearing the failed generation. Like an event callback, a concurrent removal
    /// of a non-last listener from another thread waits for this hook to return.
    fn on_stream_error(&self, _error: &SparkError) -> bool {
        false
    }

    /// This listener as `Any`, for a client that wraps its own listeners to find them among
    /// [`StreamingQueryManager::listeners`]. `None` by default.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}

/// The client-side listener bus of one session. Handles of the same session share it;
/// a new server session has a separate bus. It never owns a session or a manager.
#[derive(Default)]
pub(crate) struct ListenerBus {
    state: Mutex<ListenerBusState>,
    changed: Condvar,
}

#[derive(Default)]
struct ListenerBusState {
    listeners: Vec<(String, Arc<dyn StreamingQueryListener>)>,
    thread: Option<std::thread::JoinHandle<()>>,
    // Dropping the bus wakes a blocked reader, so its worker does not retain a session.
    cancel: Option<tokio::sync::oneshot::Sender<()>>,
    phase: ListenerBusPhase,
    worker_id: Option<std::thread::ThreadId>,
    generation: u64,
    /// The generation whose worker is delivering an event, or reporting a terminal
    /// failure, to its snapshot of the listeners. PySpark's bus holds its lock during
    /// delivery, so a removal from another thread waits for it.
    dispatching: Option<u64>,
    /// The threads delivering a synchronous started event in `start`/`toTable`, one entry
    /// per delivery. As with `dispatching`, a removal from another thread waits for them.
    started_dispatches: Vec<std::thread::ThreadId>,
    /// Set when the worker of the current generation failed in a callback, so the server still
    /// holds its listener: the server removes it only on REMOVE_LISTENER_BUS_LISTENER, and
    /// refuses a new ADD_LISTENER_BUS_LISTENER until then. A new subscription removes it first.
    server_registration_held: bool,
}

/// A synchronous started event in delivery by the calling thread of `start`/`toTable`.
struct StartedDispatch {
    bus: Arc<ListenerBus>,
    thread: std::thread::ThreadId,
}

impl Drop for StartedDispatch {
    fn drop(&mut self) {
        let mut state = self.bus.state.lock().unwrap();
        if let Some(index) = state
            .started_dispatches
            .iter()
            .rposition(|thread| *thread == self.thread)
        {
            state.started_dispatches.remove(index);
        }
        self.bus.changed.notify_all();
    }
}

/// An event delivery or terminal failure notification by the worker of `generation`.
struct Dispatch<'a> {
    bus: &'a Weak<ListenerBus>,
    generation: u64,
}

impl Drop for Dispatch<'_> {
    fn drop(&mut self) {
        if let Some(bus) = self.bus.upgrade() {
            let mut state = bus.state.lock().unwrap();
            if state.dispatching == Some(self.generation) {
                state.dispatching = None;
                bus.changed.notify_all();
            }
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
enum ListenerBusPhase {
    #[default]
    Idle,
    Starting,
    Stopping,
}

impl ListenerBus {
    /// Clear only the registrations of the worker that has finished this generation.
    fn finish_generation(&self, generation: u64) {
        let mut state = self.state.lock().unwrap();
        if state.generation == generation {
            state.listeners.clear();
            state.cancel.take();
        }
    }

    /// Record that the server still holds the listener of `generation`, whose worker failed in
    /// a callback or a decoder (not in the transport, which may have ended the server's side).
    fn mark_server_registration_held(&self, generation: u64) {
        let mut state = self.state.lock().unwrap();
        if state.generation == generation {
            state.server_registration_held = true;
        }
    }

    /// Post a synchronous start event without holding the state lock across callbacks.
    fn post_to_all(&self, event: &StreamingQueryListenerEvent) {
        let listeners: Vec<Arc<dyn StreamingQueryListener>> = self
            .state
            .lock()
            .unwrap()
            .listeners
            .iter()
            .map(|(_, listener)| Arc::clone(listener))
            .collect();
        for listener in listeners {
            listener.on_event(event);
        }
    }
}

/// Ends the generation of a worker when its loop returns or unwinds, so the registrations
/// of a worker that is gone never stay on the bus.
struct FinishGeneration {
    bus: Weak<ListenerBus>,
    generation: u64,
}

impl Drop for FinishGeneration {
    fn drop(&mut self) {
        if let Some(bus) = self.bus.upgrade() {
            bus.finish_generation(self.generation);
        }
    }
}

/// The error of a panic in a native listener's callback: a failure of that callback, as a
/// returned error from it.
fn listener_panic_error(payload: &(dyn std::any::Any + Send)) -> SparkError {
    let message = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("a non-string payload");
    SparkError::connect_msg(format!("a streaming query listener panicked: {message}"))
}

/// The worker retains neither the session nor its bus. The stream owns only transport
/// and progress-handler state; dropping the last session/manager cancels the reader.
fn run_listener_event_loop(
    stream: impl IntoIterator<Item = Result<(i32, String)>>,
    bus: Weak<ListenerBus>,
    generation: u64,
) {
    // Declared first so it runs last, after the dispatch guard, also when a callback unwinds.
    let _finish = FinishGeneration {
        bus: bus.clone(),
        generation,
    };
    let mut stream = stream.into_iter();
    let mut failure = None;
    for item in &mut stream {
        let Some(owner) = bus.upgrade() else {
            break;
        };
        let listeners: Vec<Arc<dyn StreamingQueryListener>> = {
            let mut state = owner.state.lock().unwrap();
            if state.generation != generation || state.listeners.is_empty() {
                break;
            }
            state.dispatching = Some(generation);
            state
                .listeners
                .iter()
                .map(|(_, listener)| Arc::clone(listener))
                .collect()
        };
        drop(owner);
        let dispatch = Dispatch {
            bus: &bus,
            generation,
        };
        let (event_type, event_json) = match item {
            Ok(event) => event,
            Err(error) => {
                failure = Some((listeners, error, dispatch));
                break;
            }
        };
        let event = StreamingQueryListenerEvent {
            event_type,
            event_json,
        };
        // A panic of a callback is a failure of the subscription, as a returned error is.
        let dispatched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            listeners
                .iter()
                .find_map(|listener| listener.try_post_to_all(&listeners, &event))
                .unwrap_or_else(|| {
                    listeners
                        .iter()
                        .try_for_each(|listener| listener.try_on_event(&event))
                })
        }))
        .unwrap_or_else(|payload| Err(listener_panic_error(payload.as_ref())));
        if let Err(error) = dispatched {
            if let Some(owner) = bus.upgrade() {
                owner.mark_server_registration_held(generation);
            }
            failure = Some((listeners, error, dispatch));
            break;
        }
        drop(dispatch);
    }
    if let Some((listeners, error, dispatch)) = failure {
        // A removal from another thread waits for this terminal notification, just
        // as it waits for an ordinary event. No state lock or strong bus owner is
        // held across either callback path; the worker can still remove itself.
        for listener in listeners {
            let reported = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                listener.on_stream_error(&error)
            }));
            if reported.unwrap_or(false) {
                break;
            }
        }
        drop(dispatch);
    }
}

/// Manager for active streaming queries.
///
/// Mirrors `pyspark.sql.connect.streaming.StreamingQueryManager`, including a native
/// client-side listener bus (so Rust clients get the listener feature too), which is that of
/// its session.
pub struct StreamingQueryManager {
    session: SparkSession,
    bus: Arc<ListenerBus>,
}

impl StreamingQueryManager {
    /// Create a handle to this session's shared StreamingQueryManager state.
    pub(crate) fn new(session: SparkSession) -> Self {
        let bus = session.listener_bus();
        StreamingQueryManager { session, bus }
    }

    /// Wait for another caller's start/stop operation without holding a lock needed by
    /// callbacks. A callback during shutdown may remove/close the bus as a no-op, but
    /// cannot wait for its own worker to finish in order to start a new subscription.
    fn idle_state(&self, removing: bool) -> Result<Option<MutexGuard<'_, ListenerBusState>>> {
        let mut state = self.bus.state.lock().unwrap();
        while state.phase != ListenerBusPhase::Idle {
            if state.worker_id == Some(std::thread::current().id()) {
                if removing && state.phase == ListenerBusPhase::Stopping {
                    return Ok(None);
                }
                return Err(SparkError::connect_msg(
                    "cannot change the listener bus from its own callback while another \
                     lifecycle operation is in progress",
                ));
            }
            state = self.bus.changed.wait(state).unwrap();
        }
        Ok(Some(state))
    }

    /// The error of registering a listener on a stopped session, as PySpark Connect's client.
    fn no_active_session() -> SparkError {
        SparkError::classed(SparkErrorKind::Connect, "NO_ACTIVE_SESSION", &[])
    }

    /// Register a listener, waiting for the server's acknowledgement when opening the bus.
    /// The returned id can be removed through any manager handle of this session.
    /// A terminal event-stream error or end clears all registrations; add the listeners
    /// again to open a new subscription. Terminal teardown cancels acknowledgement waits
    /// once the RPC stream is established. Establishing the stream itself follows the
    /// transport's deadline and retry policy; a stopped session then reports NO_ACTIVE_SESSION.
    pub fn add_listener(&self, listener: Arc<dyn StreamingQueryListener>) -> Result<String> {
        let mut state = self.idle_state(false)?.expect("adding obtains the state");
        if self.session.is_stopped() {
            return Err(Self::no_active_session());
        }
        let id = uuid::Uuid::new_v4().to_string();
        let worker_alive = state
            .thread
            .as_ref()
            .is_some_and(|worker| !worker.is_finished());
        if !state.listeners.is_empty() && worker_alive {
            state.listeners.push((id.clone(), listener));
            return Ok(id);
        }
        // The registrations of a worker that has ended get no more events: they are
        // cleared, and this listener opens a new subscription.
        let stale = std::mem::take(&mut state.listeners);
        let release_server = std::mem::take(&mut state.server_registration_held);
        state.phase = ListenerBusPhase::Starting;
        // An earlier worker must not clear this registration's cancellation sender.
        state.generation = state.generation.wrapping_add(1);
        let generation = state.generation;
        let (cancel, receiver) = tokio::sync::oneshot::channel();
        state.cancel = Some(cancel);
        let previous = state.thread.take();
        drop(state);
        drop(stale);
        Self::join_listener_thread(previous);
        if release_server {
            // Best effort: the server's listener is gone if the removal fails, and then the
            // add below reports it.
            let _ = self.remove_server_listener();
        }
        let stream = self.listener_event_stream().and_then(|mut stream| {
            // Cancel a pending acknowledgement once the RPC stream exists. Establishing
            // the stream above still follows the transport's deadline and retry policy.
            stream.cancel = Some(receiver);
            stream.wait_until_listener_added()?;
            Ok(stream)
        });
        let mut state = self.bus.state.lock().unwrap();
        state.phase = ListenerBusPhase::Idle;
        self.bus.changed.notify_all();
        // Terminal teardown sets stopped before canceling the stream. Its acknowledgement
        // wait treats cancellation like EOF, so this check rejects a canceled registration.
        if self.session.is_stopped() {
            state.cancel.take();
            return Err(Self::no_active_session());
        }
        match stream {
            Ok(stream) => {
                state.listeners.push((id.clone(), listener));
                let bus = Arc::downgrade(&self.bus);
                let worker = std::thread::spawn(move || {
                    run_listener_event_loop(stream, bus, generation);
                });
                state.worker_id = Some(worker.thread().id());
                state.thread = Some(worker);
                Ok(id)
            }
            Err(error) => {
                state.cancel.take();
                Err(error)
            }
        }
    }

    /// Remove a listener by id. Unknown ids do nothing. Removing the last listener drains
    /// the server event stream before returning, as PySpark's removeListener does.
    /// A callback during another caller's shutdown returns without waiting for its own
    /// worker; that caller owns the final result and may retry a failed removal.
    pub fn remove_listener(&self, id: &str) -> Result<()> {
        self.remove_listener_found(id).map(|_| ())
    }

    /// As [`remove_listener`](Self::remove_listener), returning whether a registration of `id`
    /// was removed from this bus. An absent id or a callback coalesced with an in-progress
    /// shutdown returns false, so a wrapper retains its token if that shutdown later fails.
    /// An id from another session's bus is absent here and leaves that registration untouched.
    pub fn remove_listener_found(&self, id: &str) -> Result<bool> {
        self.remove_listeners(Some(id))
    }

    /// Remove all listeners and stop this session's shared dispatch thread.
    /// A server removal failure retains the registrations and worker so closing can be retried.
    pub fn close(&self) -> Result<()> {
        self.remove_listeners(None).map(|_| ())
    }

    fn remove_listeners(&self, id: Option<&str>) -> Result<bool> {
        let Some(mut state) = self.idle_state(true)? else {
            // Another caller waits for this callback. It owns the shutdown result; keep
            // the registration token until that caller has actually removed the listener.
            return Ok(false);
        };
        if let Some(id) = id {
            let Some(index) = state
                .listeners
                .iter()
                .position(|(listener_id, _)| listener_id == id)
            else {
                return Ok(false);
            };
            if state.listeners.len() > 1 {
                state.listeners.remove(index);
                // As in PySpark's bus, a removal waits for delivery to the snapshot,
                // including the removed listener, to finish. This also covers a
                // terminal error notification. A callback of the bus does not wait,
                // as it would wait for its own delivery.
                let current = std::thread::current().id();
                if state.worker_id != Some(current) {
                    let generation = state.generation;
                    while state.dispatching == Some(generation)
                        || state
                            .started_dispatches
                            .iter()
                            .any(|thread| *thread != current)
                    {
                        state = self.bus.changed.wait(state).unwrap();
                    }
                }
                return Ok(true);
            }
        }
        let had_listeners = !state.listeners.is_empty();
        let generation = state.generation;
        state.phase = ListenerBusPhase::Stopping;
        let worker = state.thread.take();
        drop(state);
        let removal = if had_listeners {
            self.remove_server_listener()
        } else {
            Ok(())
        };
        self.finish_listener_removal(generation, worker, removal)
            .map(|_| had_listeners)
    }

    /// Reconcile an unlocked removal RPC, joining canceled generations outside the state lock.
    fn finish_listener_removal(
        &self,
        generation: u64,
        worker: Option<std::thread::JoinHandle<()>>,
        removal: Result<()>,
    ) -> Result<()> {
        let mut state = self.bus.state.lock().unwrap();
        if removal.is_err() && state.generation == generation {
            // A failed ordinary removal keeps its registrations and worker for a retry.
            state.thread = worker;
        } else {
            // Keep worker_id and Stopping until after the join: callbacks must still be
            // able to recognize their worker and coalesce a concurrent remove/close.
            drop(state);
            Self::join_listener_thread(worker);
            state = self.bus.state.lock().unwrap();
            state.listeners.clear();
            state.cancel.take();
            state.worker_id = None;
        }
        state.phase = ListenerBusPhase::Idle;
        self.bus.changed.notify_all();
        removal
    }

    /// The client-side listeners of the session, in the order that they were added, as the
    /// `_listener_bus` of PySpark Connect's `StreamingQueryListenerBus`.
    pub fn listeners(&self) -> Vec<Arc<dyn StreamingQueryListener>> {
        self.bus
            .state
            .lock()
            .unwrap()
            .listeners
            .iter()
            .map(|(_, listener)| Arc::clone(listener))
            .collect()
    }

    /// Mark the calling thread as delivering a synchronous started event until the returned
    /// guard drops. Take it before the listener snapshot of the delivery.
    fn started_dispatch(&self) -> StartedDispatch {
        let thread = std::thread::current().id();
        self.bus
            .state
            .lock()
            .unwrap()
            .started_dispatches
            .push(thread);
        StartedDispatch {
            bus: Arc::clone(&self.bus),
            thread,
        }
    }

    /// Post `event` to the client-side listeners of the session.
    pub(crate) fn post_to_all(&self, event: &StreamingQueryListenerEvent) {
        self.bus.post_to_all(event);
    }

    /// Locally cancel registration and dispatch of a stopped/closed session before or
    /// without server cleanup. Do not wait for callbacks: the current event's listener
    /// snapshot may finish, but the generation change prevents another event dispatch.
    /// Never wait for a lifecycle phase: this can be called by its worker callback while
    /// another caller is already draining that worker. That caller keeps its join handle.
    pub(crate) fn cancel_listener_bus(&self) {
        let worker = {
            let mut state = self.bus.state.lock().unwrap();
            state.listeners.clear();
            state.cancel.take();
            state.generation = state.generation.wrapping_add(1);
            state.thread.take()
        };
        // A callback may wait for an RPC ended by ReleaseSession, or for a lock held
        // by the terminal caller. Detach instead of making teardown wait for it.
        drop(worker);
    }

    /// Join the event worker unless called from that worker itself.
    fn join_listener_thread(handle: Option<std::thread::JoinHandle<()>>) {
        if let Some(handle) = handle {
            // A listener may remove itself from its callback. It cannot join itself;
            // after the callback returns the stopped stream lets its worker exit.
            if handle.thread().id() != std::thread::current().id() {
                let _ = handle.join();
            }
        }
    }

    /// Send RemoveListenerBusListener so the server ends the event stream.
    fn remove_server_listener(&self) -> Result<()> {
        let mut command = proto::StreamingQueryListenerBusCommand::default();
        command.command = Some(
            proto::streaming_query_listener_bus_command::Command::RemoveListenerBusListener(true),
        );
        crate::dataframe::execute_command_collect(
            &self.session,
            proto::command::CommandType::StreamingQueryListenerBusCommand(command),
        )?;
        Ok(())
    }

    /// Get all active streaming queries.
    pub fn active(&self) -> Result<Vec<StreamingQuery>> {
        let mut cmd = proto::StreamingQueryManagerCommand::default();
        cmd.command = Some(proto::streaming_query_manager_command::Command::Active(
            true,
        ));

        let result = self._execute_manager_command(cmd)?;

        if let Some(proto::streaming_query_manager_command_result::ResultType::Active(active)) =
            result.result_type
        {
            let queries = active
                .active_queries
                .into_iter()
                .map(|q| {
                    let query_id = q.id.as_ref().map(|id| id.id.clone()).unwrap_or_default();
                    let run_id =
                        q.id.as_ref()
                            .map(|id| id.run_id.clone())
                            .unwrap_or_default();
                    let name = q.name;

                    StreamingQuery {
                        session: self.session.clone(),
                        query_id,
                        run_id,
                        name,
                    }
                })
                .collect();
            return Ok(queries);
        }

        Ok(vec![])
    }

    /// Get a specific streaming query by ID.
    pub fn get(&self, id: &str) -> Result<Option<StreamingQuery>> {
        let mut cmd = proto::StreamingQueryManagerCommand::default();
        cmd.command = Some(proto::streaming_query_manager_command::Command::GetQuery(
            id.to_string(),
        ));

        let result = self._execute_manager_command(cmd)?;

        if let Some(proto::streaming_query_manager_command_result::ResultType::Query(query)) =
            result.result_type
        {
            let query_id = query
                .id
                .as_ref()
                .map(|id| id.id.clone())
                .unwrap_or_default();
            let run_id = query
                .id
                .as_ref()
                .map(|id| id.run_id.clone())
                .unwrap_or_default();
            let name = query.name;

            return Ok(Some(StreamingQuery {
                session: self.session.clone(),
                query_id,
                run_id,
                name,
            }));
        }

        Ok(None)
    }

    /// Wait for any streaming query to terminate with optional timeout in seconds.
    pub fn await_any_termination(&self, timeout_sec: Option<f64>) -> Result<Option<bool>> {
        let mut cmd = proto::StreamingQueryManagerCommand::default();
        let mut await_term =
            proto::streaming_query_manager_command::AwaitAnyTerminationCommand::default();

        if let Some(timeout) = timeout_sec {
            if timeout <= 0.0 {
                return Err(SparkError::value(
                    "INVALID_TIMEOUT",
                    &[("value", &timeout.to_string())],
                ));
            }
            await_term.timeout_ms = Some((timeout * 1000.0) as i64);
        }

        cmd.command =
            Some(proto::streaming_query_manager_command::Command::AwaitAnyTermination(await_term));

        let result = self._execute_manager_command(cmd)?;

        if let Some(
            proto::streaming_query_manager_command_result::ResultType::AwaitAnyTermination(
                await_result,
            ),
        ) = result.result_type
        {
            if timeout_sec.is_some() {
                return Ok(Some(await_result.terminated));
            } else {
                return Ok(None);
            }
        }

        Ok(None)
    }

    /// Reset terminated streaming queries.
    pub fn reset_terminated(&self) -> Result<()> {
        let mut cmd = proto::StreamingQueryManagerCommand::default();
        cmd.command = Some(proto::streaming_query_manager_command::Command::ResetTerminated(true));

        self._execute_manager_command(cmd)?;
        Ok(())
    }

    /// Register a server-side listener from a cloudpickled PythonUDF payload (the
    /// server runs it in a Python worker). This is distinct from the client-side
    /// listener bus ([`add_listener`](Self::add_listener)); it sends the
    /// `AddListener` manager command and returns the server listener id.
    pub fn register_python_listener(&self, payload: PythonUDFPayload) -> Result<String> {
        let listener_id = uuid::Uuid::new_v4().to_string();
        let mut cmd = proto::StreamingQueryManagerCommand::default();
        let mut listener_cmd =
            proto::streaming_query_manager_command::StreamingQueryListenerCommand::default();
        listener_cmd.python_listener_payload = Some(payload.to_proto());
        listener_cmd.id = listener_id.clone();
        cmd.command =
            Some(proto::streaming_query_manager_command::Command::AddListener(listener_cmd));

        let result = self._execute_manager_command(cmd)?;

        if let Some(proto::streaming_query_manager_command_result::ResultType::AddListener(true)) =
            result.result_type
        {
            Ok(listener_id)
        } else {
            Err(SparkError::connect_msg("Failed to add listener"))
        }
    }

    /// Remove a server-side listener registered via
    /// [`register_python_listener`](Self::register_python_listener), by id.
    pub fn unregister_python_listener(&self, listener_id: &str) -> Result<()> {
        let mut cmd = proto::StreamingQueryManagerCommand::default();
        let mut listener_cmd =
            proto::streaming_query_manager_command::StreamingQueryListenerCommand::default();
        listener_cmd.id = listener_id.to_string();
        cmd.command =
            Some(proto::streaming_query_manager_command::Command::RemoveListener(listener_cmd));

        self._execute_manager_command(cmd)?;
        Ok(())
    }

    /// Stream listener events from the server incrementally (live).
    /// Returns a ListenerEventStream that yields events as they arrive.
    pub fn listener_event_stream(&self) -> Result<ListenerEventStream> {
        let mut cmd = proto::Command::default();
        let mut listener_bus_cmd = proto::StreamingQueryListenerBusCommand::default();
        // Subscribe to receive events via the oneof command field
        listener_bus_cmd.command = Some(
            proto::streaming_query_listener_bus_command::Command::AddListenerBusListener(true),
        );
        cmd.command_type =
            Some(proto::command::CommandType::StreamingQueryListenerBusCommand(listener_bus_cmd));

        let mut plan = proto::Plan::default();
        plan.op_type = Some(proto::plan::OpType::Command(cmd));

        let request = proto::ExecutePlanRequest {
            session_id: self.session.client().session_id().to_string(),
            user_context: Some(proto::UserContext::default()),
            plan: Some(plan),
            ..Default::default()
        };

        let response_stream = self.session.execute_plan(request)?;

        Ok(ListenerEventStream {
            stream: response_stream,
            buffered_events: vec![].into_iter(),
            done: false,
            cancel: None,
        })
    }

    /// Execute a streaming query manager command and return the parsed result.
    ///
    /// Like `StreamingQuery::_execute_command`, the server's reply carries a
    /// `StreamingQueryManagerCommandResult` in the response stream; drain it and
    /// return the first such result (was discarded before, so active/get/etc.
    /// always saw an empty result).
    fn _execute_manager_command(
        &self,
        cmd: proto::StreamingQueryManagerCommand,
    ) -> Result<proto::StreamingQueryManagerCommandResult> {
        let responses = crate::dataframe::execute_command_collect(
            &self.session,
            proto::command::CommandType::StreamingQueryManagerCommand(cmd),
        )?;

        for resp in responses {
            if let Some(
                proto::execute_plan_response::ResponseType::StreamingQueryManagerCommandResult(
                    result,
                ),
            ) = resp.response_type
            {
                return Ok(result);
            }
        }

        Ok(proto::StreamingQueryManagerCommandResult::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SparkSession;

    fn session() -> SparkSession {
        SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("failed to build session")
    }

    #[test]
    fn progress_tracker_does_not_keep_listener_bus_alive() {
        let session = session();
        let bus = Arc::downgrade(&session.listener_bus());
        let tracker = crate::progress::ProgressTracker::new(&session, None);
        drop(session);
        assert!(bus.upgrade().is_none());
        drop(tracker);
    }

    #[test]
    fn a_callback_coalesced_with_shutdown_does_not_consume_its_registration() {
        struct Listener;
        impl StreamingQueryListener for Listener {
            fn on_event(&self, _: &StreamingQueryListenerEvent) {}
        }
        let spark = session();
        let manager = spark.streams();
        {
            let mut state = manager.bus.state.lock().unwrap();
            state
                .listeners
                .push(("registered".to_string(), Arc::new(Listener)));
            state.phase = ListenerBusPhase::Stopping;
            state.worker_id = Some(std::thread::current().id());
        }
        assert!(!manager.remove_listener_found("registered").unwrap());
        assert_eq!(manager.listeners().len(), 1);
        let mut state = manager.bus.state.lock().unwrap();
        state.phase = ListenerBusPhase::Idle;
        state.worker_id = None;
    }

    #[test]
    fn removing_a_duplicate_registration_preserves_other_and_foreign_ids() {
        struct Listener;
        impl StreamingQueryListener for Listener {
            fn on_event(&self, _: &StreamingQueryListenerEvent) {}
        }
        let spark = session();
        let manager = spark.streams();
        let listener: Arc<dyn StreamingQueryListener> = Arc::new(Listener);
        {
            let mut state = manager.bus.state.lock().unwrap();
            state
                .listeners
                .push(("first".to_string(), Arc::clone(&listener)));
            state
                .listeners
                .push(("second".to_string(), Arc::clone(&listener)));
        }
        assert!(!manager.remove_listener_found("foreign").unwrap());
        assert_eq!(manager.listeners().len(), 2);
        assert!(manager.remove_listener_found("first").unwrap());
        let state = manager.bus.state.lock().unwrap();
        assert_eq!(state.listeners.len(), 1);
        assert_eq!(state.listeners[0].0, "second");
        assert!(Arc::ptr_eq(&state.listeners[0].1, &listener));
    }

    /// Install a synthetic quiet worker to check cancellation sender/receiver wiring.
    /// This bypasses run_listener_event_loop and does not establish its Weak bus ownership.
    fn waiting_listener_worker(session: &SparkSession) -> std::sync::mpsc::Receiver<()> {
        struct Listener;
        impl StreamingQueryListener for Listener {
            fn on_event(&self, _: &StreamingQueryListenerEvent) {}
        }

        let (cancel, receiver) = tokio::sync::oneshot::channel();
        let (finished, observed) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _ = block_on(receiver);
            let _ = finished.send(());
        });
        let bus = session.listener_bus();
        let mut state = bus.state.lock().unwrap();
        state
            .listeners
            .push(("listener".to_string(), Arc::new(Listener)));
        state.cancel = Some(cancel);
        state.worker_id = Some(worker.thread().id());
        state.thread = Some(worker);
        observed
    }

    #[test]
    fn dropping_the_last_native_session_cancels_its_listener_worker() {
        let session = session();
        let finished = waiting_listener_worker(&session);
        drop(session);
        finished
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("dropping the bus cancels its quiet worker");
    }

    #[test]
    fn closing_a_session_eventually_cancels_its_listener_worker() {
        let session = session();
        let finished = waiting_listener_worker(&session);
        session.close();
        assert!(session.is_stopped());
        assert!(session.streams().listeners().is_empty());
        finished
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("close eventually cancels the quiet worker without joining it");
    }

    #[test]
    fn a_failed_removal_joins_its_terminally_canceled_worker() {
        let session = session();
        let finished = waiting_listener_worker(&session);
        let manager = session.streams();
        let (generation, worker) = {
            let mut state = manager.bus.state.lock().unwrap();
            state.phase = ListenerBusPhase::Stopping;
            (state.generation, state.thread.take())
        };
        // The in-flight removal owns the join handle, so terminal close only cancels it.
        session.close();
        let error = manager
            .finish_listener_removal(
                generation,
                worker,
                Err(SparkError::connect_msg("listener removal failed")),
            )
            .expect_err("the caller still receives its removal failure");
        assert_eq!(error.to_string(), "listener removal failed");
        finished
            .try_recv()
            .expect("failed removal joins a generation canceled during its RPC");
        let state = manager.bus.state.lock().unwrap();
        assert_eq!(state.phase, ListenerBusPhase::Idle);
        assert!(state.worker_id.is_none());
        assert!(state.thread.is_none());
        assert!(state.listeners.is_empty());
    }

    #[test]
    fn an_old_generation_cannot_clear_new_listeners_or_cancel_their_worker() {
        let session = session();
        let finished = waiting_listener_worker(&session);
        let bus = session.listener_bus();
        let old_generation = {
            let mut state = bus.state.lock().unwrap();
            let old_generation = state.generation;
            state.generation = state.generation.wrapping_add(1);
            old_generation
        };
        bus.finish_generation(old_generation);
        {
            let state = bus.state.lock().unwrap();
            assert_eq!(state.listeners.len(), 1);
            assert!(state.cancel.is_some());
            assert!(!state.thread.as_ref().unwrap().is_finished());
        }
        session.close();
        finished
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("only terminal close eventually cancels the current worker");
    }

    #[test]
    fn listener_bus_reports_transport_and_decode_failures_once_before_clearing() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Listener {
            bus: Weak<ListenerBus>,
            events: AtomicUsize,
            errors: Mutex<Vec<String>>,
        }
        impl StreamingQueryListener for Listener {
            fn on_event(&self, _: &StreamingQueryListenerEvent) {
                self.events.fetch_add(1, Ordering::SeqCst);
            }

            fn try_on_event(&self, event: &StreamingQueryListenerEvent) -> Result<()> {
                if event.event_json == "invalid" {
                    return Err(SparkError::connect_msg("event decode failed"));
                }
                self.on_event(event);
                Ok(())
            }

            fn on_stream_error(&self, error: &SparkError) -> bool {
                // Reporting runs before cleanup, without holding the bus lock.
                let bus = self.bus.upgrade().unwrap();
                let state = bus.state.lock().unwrap();
                assert_eq!(state.listeners.len(), 2);
                assert_eq!(state.dispatching, Some(0));
                drop(state);
                self.errors.lock().unwrap().push(error.message());
                true
            }
        }

        for decode_failure in [false, true] {
            let bus = Arc::new(ListenerBus::default());
            let listener = Arc::new(Listener {
                bus: Arc::downgrade(&bus),
                events: AtomicUsize::new(0),
                errors: Mutex::new(vec![]),
            });
            {
                let mut state = bus.state.lock().unwrap();
                state
                    .listeners
                    .push(("first".to_string(), listener.clone()));
                state
                    .listeners
                    .push(("second".to_string(), listener.clone()));
            }
            let failure = if decode_failure {
                Ok((QUERY_IDLE_EVENT, "invalid".to_string()))
            } else {
                Err(SparkError::connect_msg("transport failed"))
            };
            run_listener_event_loop(
                vec![failure, Ok((QUERY_IDLE_EVENT, "valid".to_string()))],
                Arc::downgrade(&bus),
                0,
            );
            assert_eq!(listener.events.load(Ordering::SeqCst), 0);
            assert_eq!(
                *listener.errors.lock().unwrap(),
                vec![if decode_failure {
                    "event decode failed"
                } else {
                    "transport failed"
                }],
            );
            let state = bus.state.lock().unwrap();
            assert!(state.listeners.is_empty());
            assert!(state.dispatching.is_none());
        }
    }

    #[test]
    fn nonlast_removal_waits_for_transport_decode_and_batch_error_notifications() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        struct ReleaseOnDrop(mpsc::Sender<()>);
        impl Drop for ReleaseOnDrop {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }

        struct Listener {
            batch_failure: bool,
            entered: mpsc::Sender<()>,
            release: Mutex<mpsc::Receiver<()>>,
            order: Arc<Mutex<Vec<&'static str>>>,
            events: AtomicUsize,
            errors: AtomicUsize,
        }
        impl StreamingQueryListener for Listener {
            fn on_event(&self, _: &StreamingQueryListenerEvent) {
                self.events.fetch_add(1, Ordering::SeqCst);
            }

            fn try_post_to_all(
                &self,
                _: &[Arc<dyn StreamingQueryListener>],
                _: &StreamingQueryListenerEvent,
            ) -> Option<Result<()>> {
                self.batch_failure
                    .then(|| Err(SparkError::connect_msg("batch decode failed")))
            }

            fn try_on_event(&self, _: &StreamingQueryListenerEvent) -> Result<()> {
                Err(SparkError::connect_msg("event decode failed"))
            }

            fn on_stream_error(&self, _: &SparkError) -> bool {
                self.errors.fetch_add(1, Ordering::SeqCst);
                self.order.lock().unwrap().push("reporting");
                self.entered.send(()).unwrap();
                self.release
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(15))
                    .expect("the test releases the terminal notification");
                self.order.lock().unwrap().push("reported");
                true
            }
        }

        for failure_kind in ["transport", "decode", "batch"] {
            let spark = session();
            let manager = spark.streams();
            let (entered, reporting) = mpsc::channel();
            let (release, released) = mpsc::channel();
            let order = Arc::new(Mutex::new(vec![]));
            let listener = Arc::new(Listener {
                batch_failure: failure_kind == "batch",
                entered,
                release: Mutex::new(released),
                order: Arc::clone(&order),
                events: AtomicUsize::new(0),
                errors: AtomicUsize::new(0),
            });
            {
                let mut state = manager.bus.state.lock().unwrap();
                state
                    .listeners
                    .push(("first".to_string(), listener.clone()));
                state
                    .listeners
                    .push(("second".to_string(), listener.clone()));
            }
            let failure = if failure_kind == "transport" {
                Err(SparkError::connect_msg("transport failed"))
            } else {
                Ok((QUERY_IDLE_EVENT, "event".to_string()))
            };
            let bus = Arc::downgrade(&manager.bus);
            let (start, started) = mpsc::channel();
            let (removed, removal) = mpsc::channel();
            std::thread::scope(|scope| {
                // Release the hook before scoped threads are joined even if a
                // main-thread assertion fails while it is blocked.
                let _release_on_drop = ReleaseOnDrop(release.clone());
                let worker = scope.spawn(move || {
                    started.recv_timeout(Duration::from_secs(5)).unwrap();
                    run_listener_event_loop(vec![failure], bus, 0);
                });
                manager.bus.state.lock().unwrap().worker_id = Some(worker.thread().id());
                start.send(()).unwrap();
                reporting.recv_timeout(Duration::from_secs(5)).unwrap();
                let removal_order = Arc::clone(&order);
                let manager = &manager;
                let remover = scope.spawn(move || {
                    assert!(manager.remove_listener_found("second").unwrap());
                    removal_order.lock().unwrap().push("removed");
                    removed.send(()).unwrap();
                });
                let deadline = Instant::now() + Duration::from_secs(5);
                loop {
                    if manager.bus.state.lock().unwrap().listeners.len() == 1 {
                        break;
                    }
                    assert!(Instant::now() < deadline, "remover did not reach the bus");
                    std::thread::yield_now();
                }
                assert!(matches!(
                    removal.recv_timeout(Duration::from_millis(100)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ));
                release.send(()).unwrap();
                let removal_result = removal.recv_timeout(Duration::from_secs(5));
                if removal_result.is_err() {
                    // Keep a broken guard-release regression from stranding the
                    // scoped remover while the assertion reports its failure.
                    let mut state = manager.bus.state.lock().unwrap();
                    if state.dispatching == Some(0) {
                        state.dispatching = None;
                        manager.bus.changed.notify_all();
                    }
                }
                worker.join().unwrap();
                remover.join().unwrap();
                assert!(
                    removal_result.is_ok(),
                    "removal did not finish after reporting"
                );
            });
            assert_eq!(*order.lock().unwrap(), ["reporting", "reported", "removed"]);
            assert_eq!(listener.events.load(Ordering::SeqCst), 0);
            assert_eq!(listener.errors.load(Ordering::SeqCst), 1);
            let state = manager.bus.state.lock().unwrap();
            assert!(state.listeners.is_empty());
            assert!(state.dispatching.is_none());
        }
    }

    #[test]
    fn a_terminal_error_notification_does_not_keep_the_listener_bus_alive() {
        use std::sync::mpsc;
        use std::time::Duration;

        struct Listener {
            entered: mpsc::Sender<()>,
            release: Mutex<mpsc::Receiver<()>>,
        }
        impl StreamingQueryListener for Listener {
            fn on_event(&self, _: &StreamingQueryListenerEvent) {}

            fn on_stream_error(&self, _: &SparkError) -> bool {
                self.entered.send(()).unwrap();
                self.release
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(5))
                    .expect("the test releases the terminal notification");
                true
            }
        }

        let bus = Arc::new(ListenerBus::default());
        let weak = Arc::downgrade(&bus);
        let (entered, reporting) = mpsc::channel();
        let (release, released) = mpsc::channel();
        bus.state.lock().unwrap().listeners.push((
            "listener".to_string(),
            Arc::new(Listener {
                entered,
                release: Mutex::new(released),
            }),
        ));
        let worker_bus = weak.clone();
        let worker = std::thread::spawn(move || {
            run_listener_event_loop(
                vec![Err(SparkError::connect_msg("transport failed"))],
                worker_bus,
                0,
            );
        });
        reporting.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(bus);
        assert!(weak.upgrade().is_none());
        release.send(()).unwrap();
        worker.join().unwrap();
    }

    /// A panic of a native listener's callback fails the subscription as a returned error does:
    /// the listeners are told, the registrations are cleared, and no later event is read.
    #[test]
    fn a_panicking_native_listener_fails_the_subscription_like_an_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Listener {
            events: AtomicUsize,
            errors: Mutex<Vec<String>>,
        }
        impl StreamingQueryListener for Listener {
            fn on_event(&self, _: &StreamingQueryListenerEvent) {
                self.events.fetch_add(1, Ordering::SeqCst);
                panic!("listener panic on event");
            }

            fn on_stream_error(&self, error: &SparkError) -> bool {
                self.errors.lock().unwrap().push(error.message());
                true
            }
        }
        let bus = Arc::new(ListenerBus::default());
        let listener = Arc::new(Listener {
            events: AtomicUsize::new(0),
            errors: Mutex::new(vec![]),
        });
        {
            let mut state = bus.state.lock().unwrap();
            state.generation = 3;
            state
                .listeners
                .push(("first".to_string(), listener.clone()));
        }
        run_listener_event_loop(
            vec![
                Ok((QUERY_IDLE_EVENT, "panics".to_string())),
                Ok((QUERY_IDLE_EVENT, "never read".to_string())),
            ],
            Arc::downgrade(&bus),
            3,
        );
        assert_eq!(listener.events.load(Ordering::SeqCst), 1);
        assert_eq!(
            *listener.errors.lock().unwrap(),
            vec!["a streaming query listener panicked: listener panic on event".to_string()],
        );
        let state = bus.state.lock().unwrap();
        assert!(state.listeners.is_empty());
        assert!(state.dispatching.is_none());
        assert!(state.cancel.is_none());
        assert!(state.server_registration_held);
    }

    #[test]
    fn a_stale_worker_does_not_report_errors_to_a_new_generation() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Listener(AtomicUsize);
        impl StreamingQueryListener for Listener {
            fn on_event(&self, _: &StreamingQueryListenerEvent) {
                panic!("a stale worker must not dispatch events");
            }

            fn on_stream_error(&self, _: &SparkError) -> bool {
                self.0.fetch_add(1, Ordering::SeqCst);
                true
            }
        }
        let bus = Arc::new(ListenerBus::default());
        let listener = Arc::new(Listener(AtomicUsize::new(0)));
        {
            let mut state = bus.state.lock().unwrap();
            state.generation = 2;
            state
                .listeners
                .push(("current".to_string(), listener.clone()));
        }
        run_listener_event_loop(
            vec![Err(SparkError::connect_msg("old transport failed"))],
            Arc::downgrade(&bus),
            1,
        );
        assert_eq!(listener.0.load(Ordering::SeqCst), 0);
        assert_eq!(bus.state.lock().unwrap().listeners.len(), 1);
    }

    #[test]
    fn listener_bus_batch_hook_runs_once_per_event_before_any_callback() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct NativeListener(AtomicUsize);
        impl StreamingQueryListener for NativeListener {
            fn on_event(&self, _: &StreamingQueryListenerEvent) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        struct BatchListener {
            fail: bool,
            prepared: AtomicUsize,
            callbacks: AtomicUsize,
            errors: AtomicUsize,
        }
        impl StreamingQueryListener for BatchListener {
            fn on_event(&self, _: &StreamingQueryListenerEvent) {
                self.callbacks.fetch_add(1, Ordering::SeqCst);
            }

            fn try_post_to_all(
                &self,
                listeners: &[Arc<dyn StreamingQueryListener>],
                event: &StreamingQueryListenerEvent,
            ) -> Option<Result<()>> {
                self.prepared.fetch_add(1, Ordering::SeqCst);
                if self.fail {
                    return Some(Err(SparkError::connect_msg("batch decode failed")));
                }
                for listener in listeners {
                    listener.on_event(event);
                }
                Some(Ok(()))
            }

            fn on_stream_error(&self, _: &SparkError) -> bool {
                self.errors.fetch_add(1, Ordering::SeqCst);
                true
            }
        }

        for fail in [false, true] {
            let bus = Arc::new(ListenerBus::default());
            let native = Arc::new(NativeListener(AtomicUsize::new(0)));
            let batch = Arc::new(BatchListener {
                fail,
                prepared: AtomicUsize::new(0),
                callbacks: AtomicUsize::new(0),
                errors: AtomicUsize::new(0),
            });
            {
                let mut state = bus.state.lock().unwrap();
                // A native listener precedes duplicate adapter registrations. Batch
                // preparation must run before even that first native callback.
                state.listeners.push(("native".to_string(), native.clone()));
                state.listeners.push(("batch-1".to_string(), batch.clone()));
                state.listeners.push(("batch-2".to_string(), batch.clone()));
            }
            run_listener_event_loop(
                vec![
                    Ok((QUERY_IDLE_EVENT, "first".to_string())),
                    Ok((QUERY_IDLE_EVENT, "second".to_string())),
                ],
                Arc::downgrade(&bus),
                0,
            );
            assert_eq!(
                batch.prepared.load(Ordering::SeqCst),
                if fail { 1 } else { 2 }
            );
            assert_eq!(native.0.load(Ordering::SeqCst), if fail { 0 } else { 2 });
            assert_eq!(
                batch.callbacks.load(Ordering::SeqCst),
                if fail { 0 } else { 4 }
            );
            assert_eq!(batch.errors.load(Ordering::SeqCst), usize::from(fail));
            assert!(bus.state.lock().unwrap().listeners.is_empty());
        }
    }

    #[test]
    fn stream_reader_format_option() {
        let spark = session();
        let reader = spark.read_stream();
        let reader = reader.format("kafka").option("brokers", "localhost:9092");
        assert_eq!(reader.format, Some("kafka".to_string()));
        assert_eq!(
            reader.options.get("brokers"),
            Some(&"localhost:9092".to_string())
        );
    }

    #[test]
    fn stream_reader_schema() {
        let spark = session();
        let reader = spark.read_stream();
        let reader = reader.schema("id INT, name STRING".to_string());
        assert_eq!(reader.schema, "id INT, name STRING");
    }

    #[test]
    fn stream_reader_source_name() {
        let spark = session();
        let reader = spark.read_stream();
        let reader = reader.name("my_source");
        assert_eq!(reader.source_name, Some("my_source".to_string()));
    }

    #[test]
    fn stream_reader_load_creates_streaming_dataframe() {
        let spark = session();
        let reader = spark.read_stream().format("kafka");
        let df = reader.load(Some("/path/to/data"));
        assert!(matches!(
            df.plan.as_ref(),
            crate::plan::LogicalPlan::Read {
                is_streaming: true,
                ..
            }
        ));
    }

    #[test]
    fn stream_reader_json() {
        let spark = session();
        let reader = spark.read_stream();
        let df = reader.json("/path/to/json");
        // Verify that the plan has the streaming flag set
        match df.plan.as_ref() {
            crate::plan::LogicalPlan::Read { is_streaming, .. } => {
                assert!(*is_streaming);
            }
            _ => panic!("expected Read plan with streaming"),
        }
    }

    #[test]
    fn stream_writer_format_output_mode() {
        let spark = session();
        let df = spark.read_stream().format("kafka").load(None);
        let writer = df.write_stream();
        let writer = writer.format("parquet").output_mode("append");
        assert_eq!(writer.format, Some("parquet".to_string()));
        assert_eq!(writer.output_mode, Some("append".to_string()));
    }

    #[test]
    fn stream_writer_partition_by() {
        let spark = session();
        let df = spark.read_stream().format("kafka").load(None);
        let writer = df.write_stream();
        let writer = writer.partition_by(vec!["date", "region"]);
        assert_eq!(writer.partitioning_columns, vec!["date", "region"]);
    }

    #[test]
    fn stream_writer_cluster_by() {
        let spark = session();
        let df = spark.read_stream().format("kafka").load(None);
        let writer = df.write_stream();
        let writer = writer.cluster_by(vec!["user_id", "session_id"]);
        assert_eq!(writer.clustering_columns, vec!["user_id", "session_id"]);
    }

    #[test]
    fn stream_writer_query_name() {
        let spark = session();
        let df = spark.read_stream().format("kafka").load(None);
        let writer = df.write_stream();
        let writer = writer.query_name("my_query");
        assert_eq!(writer.query_name, Some("my_query".to_string()));
    }

    #[test]
    fn stream_writer_trigger_processing_time() {
        let spark = session();
        let df = spark.read_stream().format("kafka").load(None);
        let writer = df.write_stream();
        let trigger = Trigger::ProcessingTime("10 seconds".to_string());
        let writer = writer.trigger(trigger);
        assert!(writer.trigger.is_some());
    }

    #[test]
    fn stream_writer_trigger_once() {
        let spark = session();
        let df = spark.read_stream().format("kafka").load(None);
        let writer = df.write_stream();
        let trigger = Trigger::Once;
        let writer = writer.trigger(trigger);
        assert!(writer.trigger.is_some());
    }

    #[test]
    fn stream_writer_trigger_available_now() {
        let spark = session();
        let df = spark.read_stream().format("kafka").load(None);
        let writer = df.write_stream();
        let trigger = Trigger::AvailableNow;
        let writer = writer.trigger(trigger);
        assert!(writer.trigger.is_some());
    }

    #[test]
    fn stream_writer_trigger_continuous() {
        let spark = session();
        let df = spark.read_stream().format("kafka").load(None);
        let writer = df.write_stream();
        let trigger = Trigger::Continuous("1 minute".to_string());
        let writer = writer.trigger(trigger);
        assert!(writer.trigger.is_some());
    }

    #[test]
    fn triggers_are_sent_as_pyspark_sends_them() {
        use proto::write_stream_operation_start::Trigger as Proto;
        let cases = [
            (
                Trigger::ProcessingTime("1 second".to_string()),
                Proto::ProcessingTimeInterval("1 second".to_string()),
            ),
            (Trigger::Once, Proto::Once(true)),
            (Trigger::AvailableNow, Proto::AvailableNow(true)),
            (
                Trigger::Continuous("1 minute".to_string()),
                Proto::ContinuousCheckpointInterval("1 minute".to_string()),
            ),
            // The real_time_batch_duration of PySpark's `trigger(realTime=...)`.
            (
                Trigger::RealTime("5 seconds".to_string()),
                Proto::RealTimeBatchDuration("5 seconds".to_string()),
            ),
        ];
        for (trigger, sent) in cases {
            assert_eq!(trigger.to_proto(), sent);
        }
    }

    #[test]
    fn stream_writer_option_options() {
        let spark = session();
        let df = spark.read_stream().format("kafka").load(None);
        let writer = df.write_stream();
        let writer = writer.option("key1", "val1");
        assert_eq!(writer.options.get("key1"), Some(&"val1".to_string()));

        let mut opts = std::collections::HashMap::new();
        opts.insert("key2".to_string(), "val2".to_string());
        let writer = writer.options(opts);
        assert_eq!(writer.options.get("key2"), Some(&"val2".to_string()));
    }

    #[test]
    fn remove_listener_of_an_id_the_bus_does_not_have_is_not_found() {
        let spark = session();
        let streams = spark.streams();
        assert!(!streams.remove_listener_found("no-such-id").unwrap());
        streams.remove_listener("no-such-id").unwrap();
    }

    #[test]
    fn nonlast_removal_waits_for_a_started_event_in_delivery_from_another_thread() {
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        struct Listener;
        impl StreamingQueryListener for Listener {
            fn on_event(&self, _: &StreamingQueryListenerEvent) {}
        }

        let spark = session();
        let manager = spark.streams();
        {
            let mut state = manager.bus.state.lock().unwrap();
            state
                .listeners
                .push(("first".to_string(), Arc::new(Listener)));
            state
                .listeners
                .push(("second".to_string(), Arc::new(Listener)));
        }
        let dispatch = manager.started_dispatch();
        let (removed, removal) = mpsc::channel();
        std::thread::scope(|scope| {
            // Drop the delivery guard even if an assertion fails, so the scoped remover
            // is released before the scope joins it.
            let dispatch = dispatch;
            let manager = &manager;
            scope.spawn(move || {
                assert!(manager.remove_listener_found("second").unwrap());
                removed.send(()).unwrap();
            });
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                if manager.bus.state.lock().unwrap().listeners.len() == 1 {
                    break;
                }
                assert!(Instant::now() < deadline, "remover did not reach the bus");
                std::thread::yield_now();
            }
            assert!(matches!(
                removal.recv_timeout(Duration::from_millis(100)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ));
            drop(dispatch);
            removal
                .recv_timeout(Duration::from_secs(5))
                .expect("removal finishes once the started event is delivered");
        });
        let state = manager.bus.state.lock().unwrap();
        assert!(state.started_dispatches.is_empty());
        assert_eq!(state.listeners.len(), 1);
    }

    #[test]
    fn nonlast_removal_from_the_started_event_thread_does_not_wait_for_itself() {
        use std::sync::mpsc;
        use std::time::Duration;

        struct Listener;
        impl StreamingQueryListener for Listener {
            fn on_event(&self, _: &StreamingQueryListenerEvent) {}
        }

        let spark = session();
        {
            let bus = spark.listener_bus();
            let mut state = bus.state.lock().unwrap();
            state
                .listeners
                .push(("first".to_string(), Arc::new(Listener)));
            state
                .listeners
                .push(("second".to_string(), Arc::new(Listener)));
        }
        let (result, returned) = mpsc::channel();
        // The delivery and its callback removal run on one thread; a self-wait would hang.
        std::thread::spawn(move || {
            let manager = spark.streams();
            let _dispatch = manager.started_dispatch();
            result
                .send(manager.remove_listener_found("second"))
                .unwrap();
        });
        let removed = returned
            .recv_timeout(Duration::from_secs(5))
            .expect("a removal from its own started event does not wait for itself")
            .unwrap();
        assert!(removed);
    }
}
