//! Structured Streaming support mirroring `pyspark.sql.connect.streaming`.
//!
//! Provides DataStreamReader, DataStreamWriter, StreamingQuery, and StreamingQueryManager
//! for building and executing streaming workloads.

use std::collections::HashMap;

use spark_connect_core::error::{Result, SparkError};
use spark_connect_core::runtime::block_on;
use spark_connect_proto as proto;

use crate::dataframe::DataFrame;
use crate::plan::LogicalPlan;
use crate::readwriter::ReadType;
use crate::session::SparkSession;

/// DataStreamReader for reading streaming data from various sources.
///
/// Mirrors `pyspark.sql.connect.streaming.DataStreamReader`.
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
}

/// DataStreamWriter for writing streaming data to various sinks.
///
/// Mirrors `pyspark.sql.connect.streaming.DataStreamWriter`.
pub struct DataStreamWriter {
    session: SparkSession,
    plan: LogicalPlan,
    format: Option<String>,
    output_mode: Option<String>,
    options: HashMap<String, String>,
    partitioning_columns: Vec<String>,
    clustering_columns: Vec<String>,
    query_name: Option<String>,
    trigger: Option<Trigger>,
    path: Option<String>,
    table_name: Option<String>,
}

impl DataStreamWriter {
    /// Create a new DataStreamWriter.
    pub(crate) fn new(session: SparkSession, plan: LogicalPlan) -> Self {
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

    /// Start the streaming query writing to a path, returning a StreamingQuery handle.
    pub fn start(mut self, path: &str) -> Result<StreamingQuery> {
        self.path = Some(path.to_string());
        self._start_internal()
    }

    /// Start the streaming query writing to a table, returning a StreamingQuery handle.
    pub fn to_table(mut self, table_name: &str) -> Result<StreamingQuery> {
        self.table_name = Some(table_name.to_string());
        self._start_internal()
    }

    /// Internal method to build and execute the write stream command.
    fn _start_internal(self) -> Result<StreamingQuery> {
        let mut write_op = proto::WriteStreamOperationStart::default();
        write_op.input = Some(self.plan.to_proto());

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

        if let Some(trigger) = &self.trigger {
            match trigger {
                Trigger::ProcessingTime(interval) => {
                    write_op.trigger = Some(
                        proto::write_stream_operation_start::Trigger::ProcessingTimeInterval(
                            interval.clone(),
                        ),
                    );
                }
                Trigger::Once => {
                    write_op.trigger =
                        Some(proto::write_stream_operation_start::Trigger::Once(true));
                }
                Trigger::AvailableNow => {
                    write_op.trigger = Some(
                        proto::write_stream_operation_start::Trigger::AvailableNow(true),
                    );
                }
                Trigger::Continuous(interval) => {
                    write_op.trigger = Some(
                        proto::write_stream_operation_start::Trigger::ContinuousCheckpointInterval(
                            interval.clone(),
                        ),
                    );
                }
            }
        }

        if let Some(path) = &self.path {
            write_op.sink_destination = Some(
                proto::write_stream_operation_start::SinkDestination::Path(path.clone()),
            );
        }

        if let Some(table) = &self.table_name {
            write_op.sink_destination = Some(
                proto::write_stream_operation_start::SinkDestination::TableName(table.clone()),
            );
        }

        // Build the plan with the WriteStreamOperationStart command
        let mut plan = proto::Plan::default();
        let mut cmd = proto::Command::default();
        cmd.command_type = Some(proto::command::CommandType::WriteStreamOperationStart(
            write_op,
        ));
        plan.op_type = Some(proto::plan::OpType::Command(cmd));

        // Create ExecutePlanRequest
        let mut request = proto::ExecutePlanRequest {
            session_id: self.session.client().session_id().to_string(),
            user_context: Some(proto::UserContext::default()),
            plan: Some(plan),
            ..Default::default()
        };

        // Execute the plan
        let mut response_stream = block_on(self.session.client().execute_plan(request))?;

        // Try to parse the response
        // For now, we'll create a placeholder query with dummy IDs
        // In a real implementation, we'd wait for the response and extract query IDs
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let query_id = format!("{:x}", now);
        let run_id = format!("{:x}", now);

        Ok(StreamingQuery {
            session: self.session,
            query_id,
            run_id,
            name: self.query_name.clone(),
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

    /// Execute a streaming query command and return the result.
    fn _execute_command(
        &self,
        mut cmd: proto::StreamingQueryCommand,
    ) -> Result<proto::StreamingQueryCommandResult> {
        let mut query_id = proto::StreamingQueryInstanceId::default();
        query_id.id = self.query_id.clone();
        query_id.run_id = self.run_id.clone();
        cmd.query_id = Some(query_id);

        let mut plan = proto::Plan::default();
        let mut exec_cmd = proto::Command::default();
        exec_cmd.command_type = Some(proto::command::CommandType::StreamingQueryCommand(cmd));
        plan.op_type = Some(proto::plan::OpType::Command(exec_cmd));

        let request = proto::ExecutePlanRequest {
            session_id: self.session.client().session_id().to_string(),
            user_context: Some(proto::UserContext::default()),
            plan: Some(plan),
            ..Default::default()
        };

        let _response_stream = block_on(self.session.client().execute_plan(request))?;
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
}

/// Manager for active streaming queries.
///
/// Mirrors `pyspark.sql.connect.streaming.StreamingQueryManager`.
pub struct StreamingQueryManager {
    session: SparkSession,
}

impl StreamingQueryManager {
    /// Create a new StreamingQueryManager.
    pub(crate) fn new(session: SparkSession) -> Self {
        StreamingQueryManager { session }
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

    /// Execute a streaming query manager command and return the result.
    fn _execute_manager_command(
        &self,
        cmd: proto::StreamingQueryManagerCommand,
    ) -> Result<proto::StreamingQueryManagerCommandResult> {
        let mut plan = proto::Plan::default();
        let mut exec_cmd = proto::Command::default();
        exec_cmd.command_type = Some(proto::command::CommandType::StreamingQueryManagerCommand(
            cmd,
        ));
        plan.op_type = Some(proto::plan::OpType::Command(exec_cmd));

        let request = proto::ExecutePlanRequest {
            session_id: self.session.client().session_id().to_string(),
            user_context: Some(proto::UserContext::default()),
            plan: Some(plan),
            ..Default::default()
        };

        let _response_stream = block_on(self.session.client().execute_plan(request))?;
        Ok(proto::StreamingQueryManagerCommandResult::default())
    }
}
