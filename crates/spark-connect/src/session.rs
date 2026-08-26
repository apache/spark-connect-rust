//! SparkSession implementation mirroring `pyspark.sql.SparkSession`.
//!
//! Provides the entry point for DataFrame operations and SQL queries.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use spark_connect_core::channel::ChannelBuilder;
use spark_connect_core::client::SparkConnectClient;
use spark_connect_core::error::{Result, SparkError};
use spark_connect_core::runtime::block_on;
use spark_connect_proto as proto;

use crate::catalog::Catalog;
use crate::dataframe::DataFrame;
use crate::plan::LogicalPlan;
use crate::profiler::ProfilerCollector;
use crate::row::{Row, Value};
use crate::types::DataType;

/// A progress handler invoked with each `ExecutionProgress` message received on
/// the response stream. Mirrors `SparkSession.registerProgressHandler`.
pub type ProgressHandler =
    Arc<dyn Fn(&proto::execute_plan_response::ExecutionProgress) + Send + Sync>;

/// Metrics captured from the last executed query. Mirrors the data behind
/// `pyspark.sql.DataFrame.executionInfo` / `client.core.ExecutionInfo`.
#[derive(Debug, Clone, Default)]
pub struct ExecutionInfo {
    /// Plan metrics (typically present on the final response of an execution).
    pub metrics: Option<proto::execute_plan_response::Metrics>,
    /// Observed metrics collected during execution (also where `observe(...)`
    /// and UDF-profiler results surface).
    pub observed_metrics: Vec<proto::execute_plan_response::ObservedMetrics>,
}

/// A Spark Connect session for interacting with a remote Spark cluster.
///
/// Mirrors `pyspark.sql.SparkSession`.
pub struct SparkSession {
    /// Shared gRPC client
    client: Arc<SparkConnectClient>,
    /// Plan ID counter for unique IDs in nested plans
    plan_id_counter: Arc<AtomicI64>,
    /// Operation tags attached to every ExecutePlan request from this session.
    /// Mirrors `pyspark.sql.connect.session.SparkSession`'s addTag/removeTag/getTags.
    tags: Arc<Mutex<Vec<String>>>,
    /// Metrics/observed-metrics captured from the most recent execution on this
    /// session. Backs `DataFrame::execution_info` and the profiler accessor.
    last_execution: Arc<Mutex<Option<ExecutionInfo>>>,
    /// Registered progress handlers, keyed by id (for removal). Invoked on each
    /// `ExecutionProgress` message. Mirrors `registerProgressHandler`.
    progress_handlers: Arc<Mutex<Vec<(u64, ProgressHandler)>>>,
    /// Monotonic id source for `register_progress_handler`.
    progress_handler_id: Arc<AtomicU64>,
    /// Profiler collector for accumulating profile results across executions.
    profiler: Arc<ProfilerCollector>,
}

impl SparkSession {
    /// Create a new SparkSession with a client.
    fn new(client: SparkConnectClient) -> Self {
        SparkSession {
            client: Arc::new(client),
            plan_id_counter: Arc::new(AtomicI64::new(1)),
            tags: Arc::new(Mutex::new(Vec::new())),
            last_execution: Arc::new(Mutex::new(None)),
            progress_handlers: Arc::new(Mutex::new(Vec::new())),
            progress_handler_id: Arc::new(AtomicU64::new(0)),
            profiler: Arc::new(ProfilerCollector::new()),
        }
    }

    /// Add a tag to be attached to all subsequent operations from this session.
    ///
    /// Mirrors `SparkSession.addTag`. Tags cannot be empty or contain a comma.
    pub fn add_tag(&self, tag: &str) -> Result<()> {
        if tag.is_empty() {
            return Err(SparkError::value(
                "INVALID_TAG",
                &[("detail", "Tag cannot be empty")],
            ));
        }
        if tag.contains(',') {
            return Err(SparkError::value(
                "INVALID_TAG",
                &[("detail", "Tag cannot contain ','")],
            ));
        }
        let mut tags = self.tags.lock().unwrap();
        if !tags.iter().any(|t| t == tag) {
            tags.push(tag.to_string());
        }
        Ok(())
    }

    /// Remove a previously added tag. Mirrors `SparkSession.removeTag`.
    pub fn remove_tag(&self, tag: &str) {
        self.tags.lock().unwrap().retain(|t| t != tag);
    }

    /// Get the tags currently set on this session. Mirrors `SparkSession.getTags`.
    pub fn get_tags(&self) -> Vec<String> {
        self.tags.lock().unwrap().clone()
    }

    /// Clear all tags set on this session. Mirrors `SparkSession.clearTags`.
    pub fn clear_tags(&self) {
        self.tags.lock().unwrap().clear();
    }

    /// Register a handler invoked for every `ExecutionProgress` message received
    /// during query execution. Returns an id usable with
    /// [`SparkSession::remove_progress_handler`]. Mirrors `registerProgressHandler`.
    pub fn register_progress_handler(
        &self,
        handler: impl Fn(&proto::execute_plan_response::ExecutionProgress) + Send + Sync + 'static,
    ) -> u64 {
        let id = self.progress_handler_id.fetch_add(1, Ordering::SeqCst);
        self.progress_handlers
            .lock()
            .unwrap()
            .push((id, Arc::new(handler)));
        id
    }

    /// Remove a progress handler by the id returned from
    /// [`SparkSession::register_progress_handler`]. Mirrors `removeProgressHandler`.
    pub fn remove_progress_handler(&self, id: u64) {
        self.progress_handlers
            .lock()
            .unwrap()
            .retain(|(hid, _)| *hid != id);
    }

    /// Remove all progress handlers. Mirrors `clearProgressHandlers`.
    pub fn clear_progress_handlers(&self) {
        self.progress_handlers.lock().unwrap().clear();
    }

    /// Invoke all registered progress handlers with a progress message. Called by
    /// the execute loop; not part of the public API.
    pub(crate) fn notify_progress(
        &self,
        progress: &proto::execute_plan_response::ExecutionProgress,
    ) {
        // Clone the handler `Arc`s out of the lock so a handler can't deadlock by
        // touching the session's handler list.
        let handlers: Vec<ProgressHandler> = {
            self.progress_handlers
                .lock()
                .unwrap()
                .iter()
                .map(|(_, h)| Arc::clone(h))
                .collect()
        };
        for h in handlers {
            h(progress);
        }
    }

    /// Store the metrics captured from the most recent execution. Called by the
    /// execute loop; not part of the public API.
    pub(crate) fn record_execution(&self, info: ExecutionInfo) {
        *self.last_execution.lock().unwrap() = Some(info);
    }

    /// The metrics captured from the most recent execution on this session, if any.
    /// Backs `DataFrame::execution_info`.
    pub fn last_execution_info(&self) -> Option<ExecutionInfo> {
        self.last_execution.lock().unwrap().clone()
    }

    /// Raw observed metrics from the most recent execution only.
    ///
    /// This is a low-level snapshot of the last execution's observed metrics. For
    /// the profiler surface that mirrors `SparkSession.profile` (results accumulated
    /// across executions, with show/dump/clear), use [`SparkSession::profiler`].
    pub fn profile(&self) -> Vec<proto::execute_plan_response::ObservedMetrics> {
        self.last_execution_info()
            .map(|i| i.observed_metrics)
            .unwrap_or_default()
    }

    /// Internal accessor used by the execute-request builders to attach tags.
    pub(crate) fn tags(&self) -> Vec<String> {
        self.tags.lock().unwrap().clone()
    }

    /// Start a brand-new session over the same connection. Mirrors
    /// `SparkSession.newSession` — a fresh server-side session (new session id),
    /// with its own tags and plan-id counter.
    pub fn new_session(&self) -> SparkSession {
        SparkSession {
            client: Arc::new(self.client.with_new_session_id()),
            plan_id_counter: Arc::new(AtomicI64::new(1)),
            tags: Arc::new(Mutex::new(Vec::new())),
            last_execution: Arc::new(Mutex::new(None)),
            progress_handlers: Arc::new(Mutex::new(Vec::new())),
            progress_handler_id: Arc::new(AtomicU64::new(0)),
            profiler: Arc::new(ProfilerCollector::new()),
        }
    }

    /// Alias of [`SparkSession::new_session`]; mirrors the reference
    /// `SparkSession.cloneSession`, which creates a new session on the same client.
    pub fn clone_session(&self) -> SparkSession {
        self.new_session()
    }

    /// Get the next plan ID (atomic post-increment).
    pub(crate) fn next_plan_id(&self) -> i64 {
        self.plan_id_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Get the underlying client.
    pub(crate) fn client(&self) -> &Arc<SparkConnectClient> {
        &self.client
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

    /// Execute a SQL query and return a DataFrame.
    ///
    /// Mirrors `pyspark.sql.SparkSession.sql(sqlQuery)`.
    pub fn sql(&self, query: &str) -> Result<DataFrame> {
        let plan = LogicalPlan::Sql {
            query: query.to_string(),
        };
        Ok(DataFrame::new(self.clone(), plan))
    }

    /// Create a DataFrame from a collection of Rows and a schema.
    ///
    /// Mirrors `pyspark.sql.SparkSession.createDataFrame(rows, schema)`.
    pub fn create_dataframe(&self, rows: Vec<Row>, schema: DataType) -> Result<DataFrame> {
        // Build Arrow RecordBatch from rows and schema
        let arrow_data = rows_to_arrow_ipc(&rows, &schema)?;

        let plan = LogicalPlan::LocalRelation {
            schema,
            data: Some(arrow_data),
        };
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

    /// Get the catalog for this session.
    ///
    /// Mirrors `pyspark.sql.SparkSession.catalog`.
    pub fn catalog(&self) -> crate::catalog::Catalog {
        crate::catalog::Catalog::new(self.clone())
    }

    /// Get the runtime configuration for this session.
    ///
    /// Mirrors `pyspark.sql.SparkSession.conf`.
    pub fn conf(&self) -> crate::conf::RuntimeConf {
        crate::conf::RuntimeConf::new(Arc::clone(&self.client))
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
        self.client.session_id()
    }

    /// Return the Spark version of the connected server.
    ///
    /// Mirrors `pyspark.sql.SparkSession.version`.
    pub fn version(&self) -> Result<String> {
        let mut request = proto::AnalyzePlanRequest::default();
        request.session_id = self.client.session_id().to_string();
        request.user_context = Some(proto::UserContext::default());
        request.analyze = Some(proto::analyze_plan_request::Analyze::SparkVersion(
            proto::analyze_plan_request::SparkVersion::default(),
        ));
        let resp = block_on(self.client.analyze_plan(request))?;
        match resp.result {
            Some(proto::analyze_plan_response::Result::SparkVersion(v)) => Ok(v.version),
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
            schema: DataType::Struct { fields: vec![] },
            data: None,
        };
        Ok(DataFrame::new(self.clone(), plan))
    }

    /// Interrupt all operations of this session.
    ///
    /// Mirrors `pyspark.sql.SparkSession.interruptAll()`. Returns interrupted operation ids.
    pub fn interrupt_all(&self) -> Result<Vec<String>> {
        block_on(self.client.interrupt_all())
    }

    /// Interrupt all operations of this session with the given tag.
    ///
    /// Mirrors `pyspark.sql.SparkSession.interruptTag(tag)`.
    pub fn interrupt_tag(&self, tag: &str) -> Result<Vec<String>> {
        block_on(self.client.interrupt_tag(tag))
    }

    /// Interrupt the operation with the given operation id.
    ///
    /// Mirrors `pyspark.sql.SparkSession.interruptOperation(opId)`.
    pub fn interrupt_operation(&self, operation_id: &str) -> Result<Vec<String>> {
        block_on(self.client.interrupt_operation(operation_id))
    }

    /// Add local files as artifacts to the session (e.g. `.py`, `.jar`, `.zip`).
    ///
    /// Mirrors `pyspark.sql.SparkSession.addArtifacts(*path)`.
    pub fn add_artifacts(&self, paths: &[&str]) -> Result<()> {
        block_on(self.client.add_artifacts(paths, false, false, true))
    }

    /// Add a single local file as an artifact to the session.
    ///
    /// Mirrors `pyspark.sql.SparkSession.addArtifact(path)`.
    pub fn add_artifact(&self, path: &str) -> Result<()> {
        block_on(self.client.add_artifacts(&[path], false, false, true))
    }

    /// Copy a local file to the driver's filesystem at `dest_path`.
    ///
    /// Mirrors `pyspark.sql.SparkSession.copyFromLocalToFs`: uploads the file as a
    /// `forward_to_fs/<dest_path>` artifact, which the server writes to `dest_path`.
    pub fn copy_from_local_to_fs(&self, local_path: &str, dest_path: &str) -> Result<()> {
        let name = format!("forward_to_fs/{}", dest_path);
        block_on(self.client.add_named_artifact(&name, local_path))
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
        crate::dataframe::execute_command(
            self,
            proto::command::CommandType::RegisterFunction(udf.to_proto()),
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
        crate::dataframe::execute_command(
            self,
            proto::command::CommandType::RegisterDataSource(data_source.to_proto()),
        )
    }

    /// Get the profiler collector for this session.
    ///
    /// Mirrors the client-visible surface of `SparkSession.profile`. Profile data is
    /// accumulated across query executions and can be shown, dumped, or cleared via the
    /// returned collector. Profile data is populated by the server only when UDF profiling
    /// is enabled via `spark.python.profile*` or `spark.sql.pyspark.udf.profiler` configuration.
    pub fn profiler(&self) -> Arc<ProfilerCollector> {
        Arc::clone(&self.profiler)
    }

    /// Stop this Spark session.
    pub fn stop(&self) -> Result<()> {
        block_on(self.client.release_session())?;
        Ok(())
    }
}

impl Clone for SparkSession {
    fn clone(&self) -> Self {
        SparkSession {
            client: Arc::clone(&self.client),
            plan_id_counter: Arc::clone(&self.plan_id_counter),
            tags: Arc::clone(&self.tags),
            last_execution: Arc::clone(&self.last_execution),
            progress_handlers: Arc::clone(&self.progress_handlers),
            progress_handler_id: Arc::clone(&self.progress_handler_id),
            profiler: Arc::clone(&self.profiler),
        }
    }
}

/// Builder for creating a SparkSession.
pub struct SparkSessionBuilder {
    remote_url: Option<String>,
}

impl SparkSessionBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        SparkSessionBuilder { remote_url: None }
    }

    /// Set the remote Spark Connect server URL.
    pub fn remote(mut self, url: &str) -> Self {
        self.remote_url = Some(url.to_string());
        self
    }

    /// Build and return a SparkSession.
    pub fn get_or_create(self) -> Result<SparkSession> {
        let url = self.remote_url.ok_or_else(|| {
            SparkError::value(
                "NO_REMOTE_URL",
                &[("detail", "Must call .remote(url) before .get_or_create()")],
            )
        })?;

        let builder = ChannelBuilder::parse(&url)?;
        let client = block_on(SparkConnectClient::connect(&builder))?;
        Ok(SparkSession::new(client))
    }
}

impl Default for SparkSessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert Arrow RecordBatch data to IPC bytes.
/// Used by createDataFrame to package local data.
fn rows_to_arrow_ipc(rows: &[Row], schema: &DataType) -> Result<Vec<u8>> {
    use arrow::array::*;
    use arrow::datatypes::Schema as ArrowSchema;
    use arrow::ipc::writer::StreamWriter;
    use arrow::record_batch::RecordBatch;
    use std::io::Cursor;

    if rows.is_empty() {
        return Ok(vec![]);
    }

    // Convert our schema to Arrow schema
    let arrow_schema_vec = schema_to_arrow_fields(schema)?;
    let arrow_schema = ArrowSchema::new(arrow_schema_vec);

    // Build column arrays from rows
    let mut columns = vec![];
    let fields = match schema {
        DataType::Struct { fields } => fields,
        _ => return Err(SparkError::connect_msg("Schema is not a struct type")),
    };

    // Coerce each value to its declared field type so the built arrays match the
    // Arrow schema (e.g. a Python int decodes as Long but "a int" wants Int32).
    let coerced: Vec<Row> = rows
        .iter()
        .map(|r| {
            let vals: Vec<Value> = fields
                .iter()
                .enumerate()
                .map(|(i, f)| coerce_value(r.get(i), &f.data_type))
                .collect();
            Row::new(r.fields().to_vec(), vals)
        })
        .collect();

    for field_idx in 0..fields.len() {
        let array = build_arrow_array(&coerced, field_idx, Some(&fields[field_idx].data_type))?;
        columns.push(array);
    }

    // Create RecordBatch
    let batch = RecordBatch::try_new(Arc::new(arrow_schema), columns)
        .map_err(|e| SparkError::connect_msg(format!("Failed to create Arrow batch: {}", e)))?;

    // Write to IPC stream
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &batch.schema())
            .map_err(|e| SparkError::connect_msg(format!("Failed to create IPC writer: {}", e)))?;
        writer
            .write(&batch)
            .map_err(|e| SparkError::connect_msg(format!("Failed to write batch to IPC: {}", e)))?;
        writer
            .finish()
            .map_err(|e| SparkError::connect_msg(format!("Failed to finish IPC writer: {}", e)))?;
    }

    Ok(buffer)
}

/// Convert DataType fields to Arrow fields.
fn schema_to_arrow_fields(schema: &DataType) -> Result<Vec<arrow::datatypes::Field>> {
    use arrow::datatypes::{DataType as ArrowDataType, Field, TimeUnit};

    match schema {
        DataType::Struct { fields } => {
            let mut arrow_fields = vec![];
            for field in fields {
                let arrow_type = match &field.data_type {
                    DataType::Null => ArrowDataType::Null,
                    DataType::Boolean => ArrowDataType::Boolean,
                    DataType::Byte => ArrowDataType::Int8,
                    DataType::Short => ArrowDataType::Int16,
                    DataType::Integer => ArrowDataType::Int32,
                    DataType::Long => ArrowDataType::Int64,
                    DataType::Float => ArrowDataType::Float32,
                    DataType::Double => ArrowDataType::Float64,
                    // CHAR/VARCHAR are string-backed on the wire.
                    DataType::String { .. } | DataType::Char { .. } | DataType::Varchar { .. } => {
                        ArrowDataType::Utf8
                    }
                    DataType::Binary => ArrowDataType::Binary,
                    DataType::Date => ArrowDataType::Date32,
                    // TIMESTAMP (LTZ) and TIMESTAMP_NTZ are both micros; NTZ carries no zone.
                    DataType::Timestamp => {
                        ArrowDataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
                    }
                    DataType::TimestampNtz => ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
                    DataType::Time { .. } => ArrowDataType::Time64(TimeUnit::Microsecond),
                    DataType::Decimal { precision, scale } => {
                        ArrowDataType::Decimal128(*precision as u8, *scale as i8)
                    }
                    other => {
                        return Err(SparkError::connect_msg(format!(
                            "Unsupported type for Arrow createDataFrame conversion: {other:?}"
                        )))
                    }
                };
                arrow_fields.push(Field::new(&field.name, arrow_type, field.nullable));
            }
            Ok(arrow_fields)
        }
        _ => Err(SparkError::connect_msg("Schema is not a struct")),
    }
}

/// Coerce a value to a declared field type (numeric widening/narrowing from the
/// Python-inferred type). Non-numeric or already-matching values pass through, so an
/// explicit `createDataFrame` schema produces arrays that match the Arrow schema.
fn coerce_value(v: Option<&Value>, target: &DataType) -> Value {
    let v = match v {
        Some(v) => v,
        None => return Value::Null,
    };
    match (target, v) {
        (_, Value::Null) => Value::Null,
        (DataType::Byte, Value::Long(n)) => Value::Byte(*n as i8),
        (DataType::Byte, Value::Integer(n)) => Value::Byte(*n as i8),
        (DataType::Short, Value::Long(n)) => Value::Short(*n as i16),
        (DataType::Short, Value::Integer(n)) => Value::Short(*n as i16),
        (DataType::Integer, Value::Long(n)) => Value::Integer(*n as i32),
        (DataType::Long, Value::Integer(n)) => Value::Long(*n as i64),
        // Float target from any narrower numeric (a Python int decodes as Long/Integer
        // but "a float" wants Float32); without these, an Int64Array is built against a
        // Float32 field and RecordBatch::try_new rejects the mismatch.
        (DataType::Float, Value::Double(f)) => Value::Float(*f as f32),
        (DataType::Float, Value::Long(n)) => Value::Float(*n as f32),
        (DataType::Float, Value::Integer(n)) => Value::Float(*n as f32),
        (DataType::Double, Value::Long(n)) => Value::Double(*n as f64),
        (DataType::Double, Value::Integer(n)) => Value::Double(*n as f64),
        (DataType::Double, Value::Float(f)) => Value::Double(*f as f64),
        _ => v.clone(),
    }
}

/// Build an Arrow array for a specific field across all rows.
fn build_arrow_array(
    rows: &[Row],
    field_idx: usize,
    target: Option<&DataType>,
) -> Result<Arc<dyn arrow::array::Array>> {
    use arrow::array::*;

    if rows.is_empty() {
        return Ok(Arc::new(NullArray::new(0)));
    }

    // Peek at the first row to determine the type
    let first_val = rows[0]
        .get(field_idx)
        .ok_or_else(|| SparkError::connect_msg("Invalid row index"))?;

    match first_val {
        Value::Null => Ok(Arc::new(NullArray::new(rows.len()))),
        Value::Bool(_) => {
            let values: Result<Vec<Option<bool>>> = rows
                .iter()
                .map(|r| match r.get(field_idx) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::Bool(b)) => Ok(Some(*b)),
                    Some(_) => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(BooleanArray::from(values?)))
        }
        Value::Byte(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| match r.get(field_idx) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::Byte(b)) => Ok(Some(*b)),
                    Some(_) => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(Int8Array::from(values?)))
        }
        Value::Short(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| match r.get(field_idx) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::Short(s)) => Ok(Some(*s)),
                    Some(_) => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(Int16Array::from(values?)))
        }
        Value::Integer(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| match r.get(field_idx) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::Integer(i)) => Ok(Some(*i)),
                    Some(_) => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(Int32Array::from(values?)))
        }
        Value::Long(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| match r.get(field_idx) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::Long(l)) => Ok(Some(*l)),
                    Some(_) => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(Int64Array::from(values?)))
        }
        Value::Float(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| match r.get(field_idx) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::Float(f)) => Ok(Some(*f)),
                    Some(_) => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(Float32Array::from(values?)))
        }
        Value::Double(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| match r.get(field_idx) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::Double(d)) => Ok(Some(*d)),
                    Some(_) => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(Float64Array::from(values?)))
        }
        Value::String(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| match r.get(field_idx) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::String(s)) => Ok(Some(s.as_str())),
                    Some(_) => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(StringArray::from(values?)))
        }
        Value::Binary(_) => {
            let values: Result<Vec<_>> = rows
                .iter()
                .map(|r| match r.get(field_idx) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::Binary(b)) => Ok(Some(b.as_slice())),
                    Some(_) => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(BinaryArray::from(values?)))
        }
        Value::Date(_) => {
            let values: Result<Vec<Option<i32>>> = rows
                .iter()
                .map(|r| match r.get(field_idx) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::Date(d)) => Ok(Some(*d)),
                    Some(_) => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            Ok(Arc::new(Date32Array::from(values?)))
        }
        Value::Timestamp(_) => {
            let values: Result<Vec<Option<i64>>> = rows
                .iter()
                .map(|r| match r.get(field_idx) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::Timestamp(t)) => Ok(Some(*t)),
                    Some(_) => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            // `Value::Timestamp` covers both TIMESTAMP (LTZ) and TIMESTAMP_NTZ; the
            // Arrow array must carry the timezone the schema declares, or
            // `RecordBatch::try_new` rejects it (LTZ schema field = Timestamp(_, UTC),
            // NTZ = Timestamp(_, None)). Default to UTC unless the target is NTZ.
            let arr = TimestampMicrosecondArray::from(values?);
            let arr = if matches!(target, Some(DataType::TimestampNtz)) {
                arr
            } else {
                arr.with_timezone("UTC")
            };
            Ok(Arc::new(arr))
        }
        Value::Decimal { scale, .. } => {
            // Column-wide precision/scale come from the first value (Spark decimals in a
            // column share one precision/scale). Each value's string is parsed to an
            // unscaled i128 at that scale.
            let col_scale = scale.unwrap_or(0);
            let values: Result<Vec<Option<i128>>> = rows
                .iter()
                .map(|r| match r.get(field_idx) {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::Decimal { value, .. }) => {
                        decimal_str_to_unscaled(value, col_scale).map(Some)
                    }
                    Some(_) => Err(SparkError::connect_msg("Type mismatch in row data")),
                })
                .collect();
            let arr = Decimal128Array::from(values?)
                .with_precision_and_scale(38, col_scale as i8)
                .map_err(|e| SparkError::connect_msg(format!("decimal build: {e}")))?;
            Ok(Arc::new(arr))
        }
        _ => Err(SparkError::connect_msg("Unsupported value type")),
    }
}

/// Parse a decimal string (e.g. "-1.50") into an unscaled `i128` at the given scale
/// (e.g. scale 2 → -150). Pads/truncates the fractional part to `scale` digits.
fn decimal_str_to_unscaled(s: &str, scale: i32) -> Result<i128> {
    let s = s.trim();
    let (neg, s) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let (int_part, frac_part) = match s.split_once('.') {
        Some((i, f)) => (i, f),
        None => (s, ""),
    };
    let scale = scale.max(0) as usize;
    let mut digits = String::with_capacity(int_part.len() + scale);
    digits.push_str(int_part);
    // Pad or truncate the fractional digits to exactly `scale`.
    let frac: String = frac_part
        .chars()
        .chain(std::iter::repeat('0'))
        .take(scale)
        .collect();
    digits.push_str(&frac);
    let digits = digits.trim_start_matches('0');
    let mag: i128 = if digits.is_empty() {
        0
    } else {
        digits
            .parse::<i128>()
            .map_err(|e| SparkError::connect_msg(format!("invalid decimal '{s}': {e}")))?
    };
    Ok(if neg { -mag } else { mag })
}
