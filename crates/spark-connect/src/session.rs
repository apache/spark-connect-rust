//! SparkSession implementation mirroring `pyspark.sql.SparkSession`.
//!
//! Provides the entry point for DataFrame operations and SQL queries.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use spark_connect_core::channel::ChannelBuilder;
use spark_connect_core::client::{ExecutePlanHook, SparkConnectClient};
use spark_connect_core::error::{Result, SparkError};
use spark_connect_core::runtime::{block_on, check_callback_depth};
use spark_connect_core::thread_entries::forget_when_thread_ends;
use spark_connect_proto as proto;

use crate::dataframe::DataFrame;
use crate::plan::{LogicalPlan, PlanIds};
use crate::profiler::PythonAccumulators;
use crate::progress::{ExecutionStream, ProgressTracker, ProgressUpdate};
use crate::row::{Row, Value};
use crate::types::DataType;

/// A progress handler of the executions of a session, as PySpark's `ProgressHandler`: it is
/// called with each progress of the server that has tasks, and once more, as done, when the
/// execution ends. Mirrors `SparkSession.registerProgressHandler`.
pub type ProgressHandler = Arc<dyn Fn(&ProgressUpdate) + Send + Sync>;

/// The shared registry of progress handlers and their ids, without retaining a session.
pub(crate) type ProgressHandlers = Arc<Mutex<Vec<(u64, ProgressHandler)>>>;

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
/// the set of tags that it copied.
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
pub struct SharedTags(Arc<Mutex<Vec<String>>>);

impl ThreadTags {
    /// The tags of the current thread.
    fn current(&self) -> Vec<String> {
        let shared = self
            .by_thread
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&std::thread::current().id())
            .cloned();
        shared
            .map(|tags| tags.0.lock().unwrap_or_else(|e| e.into_inner()).clone())
            .unwrap_or_default()
    }

    /// Change the tags of the current thread, and forget a thread without tags that it shares
    /// with no other.
    fn update(&self, change: impl FnOnce(&mut Vec<String>)) {
        let mut by_thread = self.by_thread.lock().unwrap_or_else(|e| e.into_inner());
        let thread = std::thread::current().id();
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
            .remove(&std::thread::current().id());
    }

    /// A copy of the tags of the current thread.
    fn copy(&self) -> SharedTags {
        SharedTags(Arc::new(Mutex::new(self.current())))
    }

    /// Make `tags` those of the current thread.
    fn share(&self, tags: &SharedTags) {
        let mut by_thread = self.by_thread.lock().unwrap_or_else(|e| e.into_inner());
        let thread = std::thread::current().id();
        if !by_thread.contains_key(&thread) {
            forget_when_thread_ends(&self.by_thread, forget_thread_tags);
        }
        by_thread.insert(thread, tags.clone());
    }
}

/// A Spark Connect session for interacting with a remote Spark cluster.
///
/// Mirrors `pyspark.sql.SparkSession`.
pub struct SparkSession {
    /// Shared gRPC client
    client: Arc<SparkConnectClient>,
    /// Stable plan IDs for the plan nodes this session serializes.
    plan_ids: Arc<PlanIds>,
    /// Operation tags attached to every ExecutePlan request that a thread starts from this
    /// session, those of the thread. Mirrors `pyspark.sql.connect.session.SparkSession`'s
    /// addTag/removeTag/getTags, of the `thread_local` of its client.
    tags: Arc<ThreadTags>,
    /// Metrics/observed-metrics captured from the most recent execution on this
    /// session. Backs `DataFrame::execution_info` and the profiler accessor.
    last_execution: Arc<Mutex<Option<ExecutionInfo>>>,
    /// Registered progress handlers, keyed by id (for removal). Invoked with each progress of
    /// an execution, and when it ends. Mirrors `registerProgressHandler`.
    progress_handlers: ProgressHandlers,
    /// Monotonic id source for `register_progress_handler`.
    progress_handler_id: Arc<AtomicU64>,
    /// The handler of the updates of the Python accumulators of its executions.
    python_accumulators: PythonAccumulators,
    /// Whether `stop()` has been called on this session (shared across clones).
    /// Backs `is_stopped`.
    stopped: Arc<AtomicBool>,
    /// Successful server version, shared by handles for this session only.
    server_version: Arc<Mutex<Option<String>>>,
    /// The client-side listener bus of the streaming queries of this session (shared across
    /// clones), which each of its `StreamingQueryManager`s adds to, removes from and posts to,
    /// as the one `StreamingQueryListenerBus` of PySpark Connect's `spark.streams`.
    listener_bus: Arc<crate::streaming::ListenerBus>,
}

impl SparkSession {
    /// Create a new SparkSession with a client.
    pub(crate) fn new(client: SparkConnectClient) -> Self {
        SparkSession {
            client: Arc::new(client),
            plan_ids: Arc::new(PlanIds::new()),
            tags: Arc::new(ThreadTags::default()),
            last_execution: Arc::new(Mutex::new(None)),
            progress_handlers: Arc::new(Mutex::new(Vec::new())),
            progress_handler_id: Arc::new(AtomicU64::new(0)),
            python_accumulators: PythonAccumulators::default(),
            stopped: Arc::new(AtomicBool::new(false)),
            server_version: Arc::new(Mutex::new(None)),
            listener_bus: Arc::default(),
        }
    }

    /// Add a tag to be attached to all subsequent operations that this thread starts from
    /// this session.
    ///
    /// Mirrors `SparkSession.addTag`, whose tags are those of the current thread. Tags cannot
    /// be empty or contain a comma (see [`SparkSession::check_tag`]).
    pub fn add_tag(&self, tag: &str) -> Result<()> {
        Self::check_tag(tag)?;
        self.tags.update(|tags| {
            if !tags.iter().any(|t| t == tag) {
                tags.push(tag.to_string());
            }
        });
        Ok(())
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
        self.tags.update(|tags| tags.retain(|t| t != tag));
    }

    /// Get the tags that this thread set on this session. Mirrors `SparkSession.getTags`.
    pub fn get_tags(&self) -> Vec<String> {
        self.tags.current()
    }

    /// Clear the tags that this thread set on this session. Mirrors `SparkSession.clearTags`:
    /// the thread gets a new empty set of tags, which it shares with no other thread.
    pub fn clear_tags(&self) {
        self.tags.clear();
    }

    /// A copy of the tags of the current thread, as of now, which the threads that are given it
    /// with [`SparkSession::share_thread_tags`] share. Mirrors how PySpark's
    /// `inheritable_thread_target(session)` copies the tags of the thread that decorates a
    /// function, which every thread that runs the function then has.
    pub fn copy_thread_tags(&self) -> SharedTags {
        self.tags.copy()
    }

    /// Make `tags` the tags of the current thread, which it shares with the other threads given
    /// them: a tag that one of them adds or removes is added or removed for all, until a thread
    /// clears its tags.
    pub fn share_thread_tags(&self, tags: &SharedTags) {
        self.tags.share(tags);
    }

    /// Register a handler of the progress of the executions of this session: it is called with
    /// each progress of the server that has tasks, and once more, as done, when the execution
    /// ends, also of an error, as PySpark's `Progress` calls its handlers. Returns an id usable
    /// with [`SparkSession::remove_progress_handler`]. Mirrors `registerProgressHandler`.
    pub fn register_progress_handler(
        &self,
        handler: impl Fn(&ProgressUpdate) + Send + Sync + 'static,
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

    /// Share the progress-handler registry without retaining the session or its listener bus.
    pub(crate) fn progress_handlers(&self) -> ProgressHandlers {
        Arc::clone(&self.progress_handlers)
    }

    /// Execute `request` on the server, with the progress of the execution for the progress
    /// handlers. Its operation ID is that of the request, or a new one, which the client
    /// assigns before it sends the request, as PySpark's `Progress` of the request has it.
    pub(crate) fn execute_plan(
        &self,
        request: proto::ExecutePlanRequest,
    ) -> Result<ExecutionStream> {
        self.execute_plan_with_handlers(request, self.progress_handlers())
    }

    /// As [`execute_plan`](Self::execute_plan), but the progress of the execution, and its done
    /// notification, does not reach the progress handlers: PySpark's listener bus has no
    /// `Progress`, so its long-running stream reports none.
    pub(crate) fn execute_plan_unreported(
        &self,
        request: proto::ExecutePlanRequest,
    ) -> Result<ExecutionStream> {
        self.execute_plan_with_handlers(request, ProgressHandlers::default())
    }

    fn execute_plan_with_handlers(
        &self,
        mut request: proto::ExecutePlanRequest,
        handlers: ProgressHandlers,
    ) -> Result<ExecutionStream> {
        let operation_id = request
            .operation_id
            .get_or_insert_with(|| uuid::Uuid::new_v4().to_string())
            .clone();
        // Before the request: an execution that fails to start ends, as done, too.
        check_callback_depth()?;
        let progress = ProgressTracker::with_handlers(handlers, Some(operation_id));
        let stream = block_on(self.client().execute_plan_reattachable(request))?;
        Ok(ExecutionStream::new(stream, progress))
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
    /// the UDF profiles of `SparkSession.profile`, whose results the server sends as the updates
    /// of Python accumulators, see [`SparkSession::set_python_accumulator_handler`].
    pub fn profile(&self) -> Vec<proto::execute_plan_response::ObservedMetrics> {
        self.last_execution_info()
            .map(|i| i.observed_metrics)
            .unwrap_or_default()
    }

    /// Internal accessor used by the execute-request builders to attach the tags of the
    /// current thread, which builds the request.
    pub(crate) fn tags(&self) -> Vec<String> {
        self.tags.current()
    }

    /// Start a brand-new session over the same connection. Mirrors
    /// `SparkSession.newSession` — a fresh server-side session (new session id),
    /// with its own tags and plan IDs.
    pub fn new_session(&self) -> SparkSession {
        SparkSession {
            client: Arc::new(self.client.with_new_session_id()),
            plan_ids: Arc::new(PlanIds::new()),
            tags: Arc::new(ThreadTags::default()),
            last_execution: Arc::new(Mutex::new(None)),
            progress_handlers: Arc::new(Mutex::new(Vec::new())),
            progress_handler_id: Arc::new(AtomicU64::new(0)),
            python_accumulators: PythonAccumulators::default(),
            stopped: Arc::new(AtomicBool::new(false)),
            server_version: Arc::new(Mutex::new(None)),
            listener_bus: Arc::default(),
        }
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
        let client = block_on(self.client.clone_session(new_session_id))?;
        let mut session = self.new_session();
        session.client = Arc::new(client);
        Ok(session)
    }

    /// Set the hook of the executions of this session, and of its clones, or clear it: the
    /// hook is called with each `ExecutePlanRequest` before it is sent, and the request sent is
    /// the one it returns, of the operation ID of the original. Mirrors the hooks of
    /// `SparkSession.Hook.on_execute_plan`, of `SparkSession.Builder._registerHook`.
    pub fn set_execute_plan_hook(&self, hook: Option<ExecutePlanHook>) {
        self.client.set_execute_plan_hook(hook);
    }

    /// The stable plan IDs of the plan nodes this session serializes.
    pub(crate) fn plan_ids(&self) -> &PlanIds {
        &self.plan_ids
    }

    pub(crate) fn plan_ids_shared(&self) -> &Arc<PlanIds> {
        &self.plan_ids
    }

    /// Get the underlying client.
    pub(crate) fn client(&self) -> &Arc<SparkConnectClient> {
        &self.client
    }

    /// Whether `other` is this session or a clone of it, which shares its client. Two sessions
    /// made apart are not the same session, even of the same session ID.
    pub fn is_same_session(&self, other: &SparkSession) -> bool {
        Arc::ptr_eq(&self.client, &other.client)
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
        // As pyspark, run the statement as a command now. The server executes a command and
        // returns a relation of its result; for a query it returns the query's relation, which
        // then runs at an action.
        let command = proto::SqlCommand {
            input: Some(crate::dataframe::build_input_relation(&plan, self)),
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
            root: sql,
            references: references
                .into_iter()
                .map(|(df, alias)| {
                    Arc::new(LogicalPlan::SubqueryAlias {
                        input: df.plan,
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
    /// Mirrors `pyspark.sql.SparkSession.createDataFrame(rows, schema)`.
    pub fn create_dataframe(&self, rows: Vec<Row>, schema: DataType) -> Result<DataFrame> {
        // Empty data -> send the schema only (no Arrow `data`); the server builds an empty
        // relation from the schema. A zero-row Arrow payload would be empty bytes, which the
        // server cannot parse ("Unexpected end of input. Missing schema.").
        if rows.is_empty() {
            let plan = LogicalPlan::LocalRelation {
                schema: Some(schema),
                data: None,
            };
            return Ok(DataFrame::new(self.clone(), plan));
        }
        let batch = rows_to_arrow_batch(&rows, &schema)?;
        Ok(DataFrame::new(
            self.clone(),
            self.local_relation(batch, schema)?,
        ))
    }

    /// The local relation of `batch`: inline, without reading the server's configuration, when
    /// it is smaller than [`LOCAL_RELATION_SETTINGS_BYTES`]; otherwise as the Arrow importer
    /// sends one, which reads the server's local-relation settings and applies the cache
    /// threshold and the size limit, as PySpark does. A single gRPC message cannot carry a
    /// relation beyond the server's message size limit.
    fn local_relation(
        &self,
        batch: arrow::record_batch::RecordBatch,
        schema: DataType,
    ) -> Result<LogicalPlan> {
        if crate::arrow_input::batch_data_size(&batch) >= LOCAL_RELATION_SETTINGS_BYTES {
            return crate::arrow_input::create_relation(
                &self.client,
                vec![batch],
                Some(schema),
                None,
            );
        }
        Ok(LogicalPlan::LocalRelation {
            schema: Some(schema),
            data: Some(batch_to_arrow_ipc(&batch)?),
        })
    }

    /// Create an inline local relation from columns with the same encoding as local rows.
    ///
    /// Unlike imported Arrow input, this path does not read server configuration or
    /// upload artifacts. Callers must supply one column per struct field.
    pub fn create_dataframe_from_columns(
        &self,
        columns: Vec<arrow::array::ArrayRef>,
        schema: DataType,
    ) -> Result<DataFrame> {
        let arrow_schema = arrow::datatypes::Schema::new(schema_to_arrow_fields(&schema)?);
        let row_count = columns.first().map_or(0, |column| column.len());
        let batch = columns_to_arrow_batch(columns, arrow_schema, row_count)?;
        Ok(DataFrame::new(
            self.clone(),
            self.local_relation(batch, schema)?,
        ))
    }

    /// Internal binding bridge for Arrow IPC already validated and encoded from
    /// local rows or columns. Retains the row path's transport policy: inline, unless the
    /// IPC is large enough to consult the server's local-relation settings.
    /// Large IPC is decoded and its map field names normalized before re-encoding;
    /// smaller IPC is passed through unchanged.
    #[doc(hidden)]
    pub fn create_dataframe_from_inline_arrow_ipc(
        &self,
        data: Vec<u8>,
        schema: DataType,
    ) -> Result<DataFrame> {
        let plan = if data.len() >= LOCAL_RELATION_SETTINGS_BYTES {
            let reader =
                arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(data), None)
                    .map_err(|e| {
                        SparkError::connect_msg(format!("Failed to read Arrow IPC: {e}"))
                    })?;
            let batches = reader
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| SparkError::connect_msg(format!("Failed to read Arrow IPC: {e}")))?;
            crate::arrow_input::create_relation(&self.client, batches, Some(schema), None)?
        } else {
            LogicalPlan::LocalRelation {
                schema: Some(schema),
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
    pub fn create_dataframe_from_arrow(
        &self,
        batches: Vec<arrow::record_batch::RecordBatch>,
        schema: impl Into<Option<DataType>>,
    ) -> Result<DataFrame> {
        let plan = crate::arrow_input::create_relation(&self.client, batches, schema.into(), None)?;
        Ok(DataFrame::new(self.clone(), plan))
    }

    /// Internal settings handoff for the Python Arrow importer; not a stable API.
    #[doc(hidden)]
    pub fn arrow_input_options(&self) -> Result<crate::ArrowInputOptions> {
        crate::ArrowInputOptions::from_client(&self.client)
    }

    /// Internal Arrow import entry point using the normalization settings snapshot.
    #[doc(hidden)]
    pub fn create_dataframe_from_arrow_with_options(
        &self,
        batches: Vec<arrow::record_batch::RecordBatch>,
        schema: impl Into<Option<DataType>>,
        options: &crate::ArrowInputOptions,
    ) -> Result<DataFrame> {
        let plan = crate::arrow_input::create_relation(
            &self.client,
            batches,
            schema.into(),
            Some(options),
        )?;
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
        Arc::clone(&self.listener_bus)
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
        crate::dataframe::execute_command_collect(
            self,
            proto::command::CommandType::RegisterFunction(fun),
        )?;
        Ok(())
    }

    /// Execute `command`, a command built outside the core, such as the ML commands of
    /// `pyspark.ml`, and return the responses of the server, as the Python client's
    /// `execute_command` reads them. The placeholders of its relations are replaced, as
    /// `DataFrame::from_relation` does. Not a stable API.
    #[doc(hidden)]
    pub fn execute_command_proto(
        &self,
        mut command: spark_connect_proto::Command,
    ) -> Result<Vec<spark_connect_proto::ExecutePlanResponse>> {
        crate::plan::resolve_command(&mut command)?;
        let Some(command_type) = command.command_type else {
            return Err(SparkError::value_msg("a command of no command type"));
        };
        let (responses, _) = crate::dataframe::execute_command_with_info(self, command_type)?;
        Ok(responses)
    }

    /// Delete the ML cached objects `refs` on the server, as PySpark's `RemoteModelRef` does
    /// when the last reference to a model goes. It does not wait for the server: the request
    /// runs on the shared runtime, and its errors (a stopped session, a closed channel) are
    /// ignored. Safe to call from `Drop`.
    pub(crate) fn release_ml_objects(&self, refs: Vec<proto::ObjectRef>) {
        let command = proto::Command {
            command_type: Some(proto::command::CommandType::MlCommand(proto::MlCommand {
                command: Some(proto::ml_command::Command::Delete(
                    proto::ml_command::Delete {
                        obj_refs: refs,
                        evict_only: None,
                    },
                )),
            })),
        };
        self.execute_command_in_background(command);
    }

    /// Remove the cached remote relation `relation_id` on the server, as PySpark's
    /// `CachedRemoteRelation.__del__` does when its DataFrame goes. Like `release_ml_objects`,
    /// it does not wait for the server, and its errors are ignored. A stopped session is not
    /// released: stopping it released its server state already, as PySpark skips a closed
    /// client. Safe to call from `Drop`.
    pub(crate) fn release_cached_remote_relation(&self, relation_id: String) {
        if self.is_stopped() {
            return;
        }
        let command = proto::Command {
            command_type: Some(
                proto::command::CommandType::RemoveCachedRemoteRelationCommand(
                    proto::RemoveCachedRemoteRelationCommand {
                        relation: Some(proto::CachedRemoteRelation { relation_id }),
                    },
                ),
            ),
        };
        self.execute_command_in_background(command);
    }

    /// Execute `command` on the server without waiting for its responses, on the shared
    /// runtime. Its errors are ignored; safe to call from `Drop`.
    fn execute_command_in_background(&self, command: proto::Command) {
        let mut plan = proto::Plan::default();
        plan.op_type = Some(proto::plan::OpType::Command(command));
        let request = proto::ExecutePlanRequest {
            session_id: self.client.session_id().to_string(),
            user_context: Some(proto::UserContext::default()),
            operation_id: Some(uuid::Uuid::new_v4().to_string()),
            plan: Some(plan),
            ..Default::default()
        };
        let client = Arc::clone(&self.client);
        spark_connect_core::runtime::get_runtime().spawn(async move {
            if let Ok(mut stream) = client.execute_plan_reattachable(request).await {
                while let Ok(Some(_)) = stream.message().await {}
            }
        });
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

    /// The display host of the endpoint of this session, as PySpark's `SparkConnectClient.host`.
    pub fn host(&self) -> &str {
        self.client.host()
    }

    /// The server-side session ID that the responses of this session carried, or None before the
    /// first response, as PySpark's `SparkConnectClient._server_session_id`.
    pub fn server_session_id(&self) -> Option<String> {
        self.client.observed_server_session_id()
    }

    /// Set the server-side session ID that the requests of this session observe.
    pub fn set_server_session_id(&self, id: Option<String>) {
        self.client.set_observed_server_session_id(id);
    }

    /// Add `extension` to the user context of the requests that the current thread sends from
    /// this session, and return its ID. Mirrors PySpark's
    /// `SparkConnectClient.add_threadlocal_user_context_extension`.
    pub fn add_threadlocal_user_context_extension(&self, extension: prost_types::Any) -> String {
        self.client
            .add_threadlocal_user_context_extension(extension)
    }

    /// Add `extension` to the user context of the requests of all the threads of this session,
    /// and return its ID. Mirrors PySpark's `add_global_user_context_extension`.
    pub fn add_global_user_context_extension(&self, extension: prost_types::Any) -> String {
        self.client.add_global_user_context_extension(extension)
    }

    /// Remove the user context extension of `extension_id`. Mirrors PySpark's
    /// `remove_user_context_extension`.
    pub fn remove_user_context_extension(&self, extension_id: &str) {
        self.client.remove_user_context_extension(extension_id);
    }

    /// The user context extensions of the current thread, of their IDs. Mirrors PySpark's
    /// `thread_local.user_context_extensions` of a client.
    pub fn threadlocal_user_context_extensions(&self) -> Vec<(String, prost_types::Any)> {
        self.client.threadlocal_user_context_extensions()
    }

    /// The global user context extensions, of their IDs. Mirrors PySpark's
    /// `global_user_context_extensions` of a client.
    pub fn global_user_context_extensions(&self) -> Vec<(String, prost_types::Any)> {
        self.client.global_user_context_extensions()
    }

    /// Remove the user context extensions of the current thread and the global ones. Mirrors
    /// PySpark's `clear_user_context_extensions`.
    pub fn clear_user_context_extensions(&self) {
        self.client.clear_user_context_extensions();
    }

    /// A copy of the user context extensions of the current thread, which the threads that are
    /// given it with [`SparkSession::share_threadlocal_user_context_extensions`] share. Mirrors
    /// how PySpark's `inheritable_thread_target(session)` copies the `thread_local` of the client
    /// of the thread that decorates a function.
    pub fn copy_threadlocal_user_context_extensions(
        &self,
    ) -> spark_connect_core::client::SharedUserContextExtensions {
        self.client.copy_threadlocal_user_context_extensions()
    }

    /// Make `extensions` the user context extensions of the current thread, which it shares with
    /// the other threads given them, as `inheritable_thread_target` gives every thread that runs
    /// its function the copy of the `thread_local` of the client.
    pub fn share_threadlocal_user_context_extensions(
        &self,
        extensions: &spark_connect_core::client::SharedUserContextExtensions,
    ) {
        self.client
            .share_threadlocal_user_context_extensions(extensions);
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
            .server_version
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(version) = cached.as_ref() {
            return Ok(version.clone());
        }
        let mut request = proto::AnalyzePlanRequest::default();
        request.session_id = self.client.session_id().to_string();
        request.user_context = Some(proto::UserContext::default());
        request.analyze = Some(proto::analyze_plan_request::Analyze::SparkVersion(
            proto::analyze_plan_request::SparkVersion::default(),
        ));
        let resp = block_on(self.client.analyze_plan(request))?;
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
        block_on(self.client.add_artifacts(paths, pyfile, archive, file))
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
            self.client
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
        crate::dataframe::execute_command(
            self,
            proto::command::CommandType::RegisterFunction(udf.to_proto()),
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
            command,
            python_ver,
        );
        crate::dataframe::execute_command(
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
        crate::dataframe::execute_command(
            self,
            proto::command::CommandType::RegisterDataSource(data_source.to_proto()),
        )
    }

    /// Set the handler of the updates of the Python accumulators of the executions of this
    /// session: each pickled `(id, update)` of their `__python_accumulator__` observed metrics,
    /// of which the UDF profilers send their results, as PySpark Connect's client unpickles each
    /// into its `ConnectProfilerCollector`.
    pub fn set_python_accumulator_handler(&self, handler: impl Fn(&[u8]) + Send + Sync + 'static) {
        self.python_accumulators.set_handler(Arc::new(handler));
    }

    /// The handler of the updates of the Python accumulators, of the executions.
    pub(crate) fn python_accumulators(&self) -> &PythonAccumulators {
        &self.python_accumulators
    }

    /// Stop this Spark session.
    ///
    /// Idempotent, mirroring pyspark: the session is marked stopped even if releasing it
    /// on the server fails, and stopping an already stopped session is a no-op.
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
        if self.stopped.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        let released = if self.client.is_closed() {
            Ok(())
        } else {
            block_on(self.client.release_session())
        };
        self.streams().cancel_listener_bus();
        // As PySpark's `stop()`, close the client even if server cleanup fails, so its
        // future requests raise `NO_ACTIVE_SESSION`.
        self.client.close();
        released
    }

    /// Release the session on the server, as `SparkConnectClient.release_session`: its state
    /// there, without stopping this session.
    pub fn release_session(&self) -> Result<()> {
        block_on(self.client.release_session())
    }

    /// Close this session without releasing it on the server, as `SparkConnectClient.close`
    /// closes its channel: it is stopped, and `stop()` does nothing more. Cancel local listener
    /// registration/dispatch before closing the client. Callbacks are not joined; the current
    /// event's listener snapshot may finish after close returns.
    pub fn close(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        self.streams().cancel_listener_bus();
        self.client.close();
    }

    /// Whether `stop()` has been called on this session, or its client is closed, as after a
    /// server-side session change. Mirrors `pyspark.sql.connect.session.SparkSession.is_stopped`,
    /// which is the client's `is_closed`.
    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst) || self.client.is_closed()
    }
}

/// The name of the artifact that copies a file to `dest_path`, as the reference's
/// `FORWARD_TO_FS_PREFIX + dest_path`: the prefix is followed by the absolute path itself.
fn forward_to_fs_artifact_name(dest_path: &str) -> String {
    format!("forward_to_fs{}", dest_path)
}

impl Clone for SparkSession {
    fn clone(&self) -> Self {
        SparkSession {
            client: Arc::clone(&self.client),
            plan_ids: Arc::clone(&self.plan_ids),
            tags: Arc::clone(&self.tags),
            last_execution: Arc::clone(&self.last_execution),
            progress_handlers: Arc::clone(&self.progress_handlers),
            progress_handler_id: Arc::clone(&self.progress_handler_id),
            python_accumulators: self.python_accumulators.clone(),
            stopped: Arc::clone(&self.stopped),
            server_version: Arc::clone(&self.server_version),
            listener_bus: Arc::clone(&self.listener_bus),
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

/// The error for a row of `create_dataframe` whose width differs from the schema's. The
/// Python client checks widths itself, raising pyspark's `AXIS_LENGTH_MISMATCH`.
fn row_width_mismatch(row_index: usize, actual: usize, expected: usize) -> SparkError {
    SparkError::value_msg(format!(
        "Row {row_index} has {actual} values, but the schema has {expected} fields"
    ))
}

/// Convert Arrow RecordBatch data to IPC bytes.
/// Used by createDataFrame to package local data.
/// The size of a local relation, in Arrow data, from which `createDataFrame` reads the
/// server's local-relation settings and sends the relation as the Arrow importer does: inline
/// below `spark.sql.session.localRelationCacheThreshold`, uploaded at it, and failing past
/// `spark.sql.session.localRelationSizeLimit`, as PySpark does. Smaller relations are sent
/// inline without that RPC: 64 KiB is far below the server's gRPC message size limit (128 MiB
/// by default), so a relation this small cannot be too large to send, and the RPC would cost
/// more than the transfer.
const LOCAL_RELATION_SETTINGS_BYTES: usize = 64 << 10;

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

/// Map a Spark `DataType` to its Arrow `DataType`, recursively (array/map/struct included).
fn datatype_to_arrow(dt: &DataType) -> Result<arrow::datatypes::DataType> {
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
        DataType::String { .. } | DataType::Char { .. } | DataType::Varchar { .. } => A::Utf8,
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
        DataType::Decimal { precision, scale } => A::Decimal128(*precision as u8, *scale as i8),
        DataType::Array {
            element_type,
            contains_null,
        } => A::List(Arc::new(Field::new(
            "element",
            datatype_to_arrow(element_type)?,
            *contains_null,
        ))),
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => {
            let entry_fields: Fields = vec![
                Field::new("key", datatype_to_arrow(key_type)?, false),
                Field::new(
                    "value",
                    datatype_to_arrow(value_type)?,
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
        // schema of the relation keeps the names of the fields.
        DataType::Struct { fields } => {
            let names = dedup_field_names(fields.iter().map(|field| field.name.as_str()));
            let f: Fields = fields
                .iter()
                .zip(names)
                .map(|(field, name)| {
                    Ok(Field::new(
                        name,
                        datatype_to_arrow(&field.data_type)?,
                        field.nullable,
                    ))
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
        // As PySpark's, a user-defined type is sent as its SQL type.
        DataType::Udt {
            sql_type: Some(sql_type),
            ..
        } => datatype_to_arrow(sql_type)?,
        other => {
            return Err(SparkError::connect_msg(format!(
                "Unsupported type for Arrow createDataFrame conversion: {other:?}"
            )))
        }
    })
}

/// Recursively build an Arrow array from a column of `Value`s against a declared
/// `DataType`, covering scalars and the nested array/map/struct types. Temporary
/// columns borrow their values; building the Arrow output still copies payloads.
fn values_to_arrow(values: &[&Value], dt: &DataType) -> Result<Arc<dyn arrow::array::Array>> {
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
                .with_precision_and_scale(*precision as u8, col_scale as i8)
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
            let child_arr = values_to_arrow(&child, element_type)?;
            let field = Arc::new(Field::new(
                "element",
                datatype_to_arrow(element_type)?,
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
        } => values_to_arrow(values, sql_type),
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
                .map(|(i, f)| values_to_arrow(&cols[i], &f.data_type))
                .collect::<Result<_>>()?;
            // The Arrow fields of the struct, with the deduplicated names of `datatype_to_arrow`,
            // which the column's type has.
            let arrow::datatypes::DataType::Struct(afields) = datatype_to_arrow(dt)? else {
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
            let key_arr = values_to_arrow(&keys, key_type)?;
            let val_arr = values_to_arrow(&vals, value_type)?;
            let entry_fields: Fields = vec![
                Field::new("key", datatype_to_arrow(key_type)?, false),
                Field::new(
                    "value",
                    datatype_to_arrow(value_type)?,
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
        let values = vec![
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
            let added = (other.get_tags(), other.tags());
            other.remove_tag("child");
            (before, added, other.get_tags())
        })
        .join()
        .unwrap();
        let child_tags = vec!["child".to_string()];
        assert_eq!(child, (vec![], (child_tags.clone(), child_tags), vec![]));
        assert_eq!(spark.get_tags(), vec!["main".to_string()]);
        assert_eq!(spark.tags(), vec!["main".to_string()]);
        spark.clear_tags();
        assert!(spark.get_tags().is_empty());
        assert!(spark.tags.by_thread.lock().unwrap().is_empty());
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
        assert!(spark.tags.by_thread.lock().unwrap().is_empty());
        assert!(spark.get_tags().is_empty());
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
        assert_eq!(spark.tags.by_thread.lock().unwrap().len(), 1);
        end_tx.send(()).unwrap();
        assert_eq!(child.join().unwrap(), vec!["a"]);
        assert!(spark.tags.by_thread.lock().unwrap().is_empty());
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
        // Port 1: nothing listens, so the ReleaseSession RPC fails fast. The session is
        // still marked stopped (pyspark's stop() is best-effort) and a second stop() is a
        // no-op that does not retry the release.
        let spark = SparkSession::builder()
            .remote("sc://localhost:1")
            .get_or_create()
            .expect("failed to build session");
        let other = spark.clone();
        let _ = spark.stop();
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
        let _ = spark.stop();
        let err = spark
            .conf()
            .get("spark.sql.shuffle.partitions")
            .expect_err("a request of a stopped session must fail");
        assert_eq!(err.get_condition().as_deref(), Some("NO_ACTIVE_SESSION"));
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
        *spark.server_version.lock().unwrap() = Some("4.3.0".into());
        assert_eq!(handle.version().unwrap(), "4.3.0");
        assert!(Arc::ptr_eq(&spark.server_version, &handle.server_version));
        for other in [spark.new_session(), session()] {
            assert!(!Arc::ptr_eq(&spark.server_version, &other.server_version));
            assert!(other.server_version.lock().unwrap().is_none());
        }
        spark.stopped.store(true, Ordering::SeqCst);
        assert_eq!(handle.version().unwrap(), "4.3.0");
    }

    #[test]
    fn version_cache_recovers_after_lookup_panics() {
        let spark = session();
        let handle = spark.clone();
        // A panic under the cache lock, as a panicking lookup leaves it, must not permanently
        // break this session's cache.
        let cache = std::sync::Arc::clone(&spark.server_version);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = cache.lock().unwrap();
            panic!("a lookup panics while the cache lock is held");
        }));
        assert!(result.is_err());
        assert!(spark.server_version.is_poisoned());
        let mut cached = spark
            .server_version
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
        spark.client.close();
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
        assert!(spark.stopped.load(Ordering::SeqCst));
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
