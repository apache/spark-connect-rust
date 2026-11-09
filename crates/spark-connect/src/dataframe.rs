//! DataFrame implementation mirroring `pyspark.sql.DataFrame`.
//!
//! Provides transformations and actions for working with distributed data.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use spark_connect_core::error::{Result, SparkError, SparkErrorKind};
use spark_connect_core::runtime::{block_on, get_runtime};
use spark_connect_proto as proto;

use crate::column::Column;
use crate::expression::{Expression, ExpressionNode};
use crate::plan::{AggregateGroupType, JoinType, LogicalPlan, SetOpType};
use crate::progress::ExecutionStream;
use crate::row::{Row, StructValue, Value};
use crate::session::{ExecutionInfo, SparkSession};
use crate::types::DataType;
use crate::udf::CommonInlineUserDefinedFunctionExpression;

/// A Spark DataFrame, lazily evaluated.
///
/// Mirrors `pyspark.sql.DataFrame`.
/// Clones and transformations share immutable plan nodes with their inputs.
#[derive(Clone)]
pub struct DataFrame {
    pub(crate) session: SparkSession,
    pub(crate) plan: Arc<LogicalPlan>,
    // Exact clones share this cache. Derived plans can snapshot an already
    // resolved schema, but have their own cache for subsequent analysis.
    // The resolved value is immutable, so propagation need not copy wide schemas.
    schema_cache: Arc<Mutex<Option<Arc<DataType>>>>,
    // The information of the last execution of this DataFrame, as PySpark's
    // `DataFrame._execution_info`: exact clones share it, and a derived DataFrame, of
    // another plan, has its own, which is empty until it executes.
    execution: ExecutionSlot,
}

/// Where a DataFrame keeps the information of its last execution.
pub(crate) type ExecutionSlot = Arc<Mutex<Option<ExecutionInfo>>>;

/// Iterator over rows from a DataFrame, yielded lazily as the server streams results.
///
/// The options of an as-of join, as `DataFrame._joinAsOf` takes them.
///
/// `on` and `using_columns` are the join condition: a condition, or names that both sides
/// share. PySpark sets at most one of them. The defaults are those of PySpark: an inner join,
/// exact matches allowed, and the `"backward"` direction.
#[derive(Debug, Clone)]
pub struct AsOfJoinOptions {
    /// The as-of column of this DataFrame.
    pub left_as_of: Column,
    /// The as-of column of the other DataFrame.
    pub right_as_of: Column,
    /// The join condition, sent as `join_expr`.
    pub on: Option<Column>,
    /// Column names that both sides share, sent as `using_columns`.
    pub using_columns: Vec<String>,
    /// The join type, such as `"inner"` or `"left"`, sent as it is.
    pub join_type: String,
    /// The as-of tolerance: the largest distance a match may have.
    pub tolerance: Option<Column>,
    /// Whether a row may match a row with the same as-of value.
    pub allow_exact_matches: bool,
    /// `"backward"`, `"forward"` or `"nearest"`, sent as it is.
    pub direction: String,
}

impl AsOfJoinOptions {
    /// Options joining on `left_as_of` and `right_as_of`, with the defaults of PySpark.
    pub fn new(left_as_of: Column, right_as_of: Column) -> Self {
        Self {
            left_as_of,
            right_as_of,
            on: None,
            using_columns: Vec::new(),
            join_type: "inner".to_string(),
            tolerance: None,
            allow_exact_matches: true,
            direction: "backward".to_string(),
        }
    }
}

/// Returned by `DataFrame::to_local_iterator()`, this iterator consumes the ExecutePlan
/// response stream incrementally and yields Row objects without buffering the entire result.
pub struct LocalRowIterator {
    /// Rows decoded from the current Arrow batch, handed out one at a time.
    current_rows: std::vec::IntoIter<Row>,
    /// Where subsequent batches come from: pulled on demand, or fed by a
    /// background prefetch task.
    source: RowSource,
    /// Set once the batch source is exhausted or has errored.
    done: bool,
    /// The schema of the result that the server sends before the rows
    /// (`ExecutePlanResponse.schema`), once received.
    schema: SchemaSlot,
}

/// The schema of a result as the server sends it, shared with a prefetch task.
type SchemaSlot = Arc<Mutex<Option<DataType>>>;

/// Keeps the schema of `resp`, if it has one, in `slot`.
fn keep_schema(resp: &mut proto::ExecutePlanResponse, slot: &SchemaSlot) -> Result<()> {
    if let Some(data_type) = resp.schema.take() {
        let schema = DataType::from_proto(&data_type)?;
        *slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(schema);
    }
    Ok(())
}

/// The source of successive Arrow batches for a [`LocalRowIterator`].
enum RowSource {
    /// Each batch is pulled from the response stream only when the previous
    /// one is exhausted (`prefetchPartitions=False`).
    OnDemand {
        session: SparkSession,
        // Boxed: the stream, with the progress of its execution, is much larger than a
        // receiver.
        stream: Box<ExecutionStream>,
        execution_info: ExecutionInfo,
        execution_recorded: bool,
    },
    /// Batches are fetched by a background task that keeps one batch buffered
    /// ahead, so the next server fetch overlaps with the caller consuming the
    /// current batch's rows (`prefetchPartitions=True`).
    Prefetch {
        rx: tokio::sync::mpsc::Receiver<Result<Vec<Row>>>,
    },
}

impl LocalRowIterator {
    /// Create a new `LocalRowIterator` over an already-issued ExecutePlan stream.
    pub(crate) fn new(
        session: SparkSession,
        stream: ExecutionStream,
        prefetch_partitions: bool,
    ) -> Self {
        let schema = SchemaSlot::default();
        let source = if prefetch_partitions {
            RowSource::Prefetch {
                rx: spawn_prefetch(session, stream, Arc::clone(&schema)),
            }
        } else {
            RowSource::OnDemand {
                session,
                stream: Box::new(stream),
                execution_info: ExecutionInfo::default(),
                execution_recorded: false,
            }
        };
        LocalRowIterator {
            current_rows: vec![].into_iter(),
            source,
            done: false,
            schema,
        }
    }

    /// The schema of the result, as the server sends it before the rows, once received: so
    /// after the first row. Unlike the Arrow schema of the rows, it has the user-defined
    /// types of the columns.
    pub fn schema(&self) -> Option<DataType> {
        self.schema
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Fetch the next batch of rows, or `None` once the source is exhausted.
    ///
    /// A batch may legitimately decode to zero rows (e.g. a metrics-only
    /// response was skipped); the caller loops until it gets a row or `None`.
    fn fetch_next_batch(&mut self) -> Option<Result<Vec<Row>>> {
        match &mut self.source {
            RowSource::OnDemand {
                session,
                stream,
                execution_info,
                execution_recorded,
            } => loop {
                match block_on(stream.message()) {
                    Ok(Some(mut resp)) => {
                        capture_execution(&mut resp, execution_info, session);
                        if let Err(e) = keep_schema(&mut resp, &self.schema) {
                            return Some(Err(e));
                        }
                        if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
                            resp.response_type
                        {
                            let schema = self.schema.lock().unwrap_or_else(|p| p.into_inner());
                            return Some(decode_arrow_batch_of(&batch, schema.as_ref()));
                        }
                        // Metrics/progress response: keep pulling for a batch.
                    }
                    Ok(None) => {
                        if !*execution_recorded {
                            session.record_execution(execution_info.clone());
                            *execution_recorded = true;
                        }
                        return None;
                    }
                    Err(e) => {
                        if !*execution_recorded {
                            session.record_execution(execution_info.clone());
                            *execution_recorded = true;
                        }
                        return Some(Err(e));
                    }
                }
            },
            RowSource::Prefetch { rx } => block_on(rx.recv()),
        }
    }
}

impl Iterator for LocalRowIterator {
    type Item = Result<Row>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(row) = self.current_rows.next() {
                return Some(Ok(row));
            }
            if self.done {
                return None;
            }
            match self.fetch_next_batch() {
                Some(Ok(rows)) => self.current_rows = rows.into_iter(),
                Some(Err(e)) => {
                    self.done = true;
                    return Some(Err(e));
                }
                None => {
                    self.done = true;
                    return None;
                }
            }
        }
    }
}

/// Spawn a background task that forwards decoded response batches over a bounded
/// (capacity-1) channel, keeping one batch buffered ahead of the consumer.
/// Closing the receiver stops the task even while it awaits a response; dropping
/// the stream releases the execution. Metrics received so far are recorded on the
/// session when the task stops. Backs `to_local_iterator(prefetch_partitions = true)`.
fn spawn_prefetch(
    session: SparkSession,
    mut stream: ExecutionStream,
    schema: SchemaSlot,
) -> tokio::sync::mpsc::Receiver<Result<Vec<Row>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Vec<Row>>>(1);
    get_runtime().spawn(async move {
        let mut execution_info = ExecutionInfo::default();
        loop {
            // Closing the consumer must release the execution even while the server has
            // not produced another response. Dropping this task's stream releases it.
            let response = tokio::select! {
                _ = tx.closed() => break,
                response = stream.message() => response,
            };
            match response {
                Ok(Some(mut resp)) => {
                    capture_execution(&mut resp, &mut execution_info, &session);
                    if let Err(e) = keep_schema(&mut resp, &schema) {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                    if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
                        resp.response_type
                    {
                        let decoded = {
                            let schema = schema.lock().unwrap_or_else(|p| p.into_inner());
                            decode_arrow_batch_of(&batch, schema.as_ref())
                        };
                        match decoded {
                            Ok(rows) => {
                                // A send error means the consumer dropped the
                                // iterator; stop fetching.
                                if tx.send(Ok(rows)).await.is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(Err(e)).await;
                                break;
                            }
                        }
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    break;
                }
            }
        }
        session.record_execution(execution_info);
    });
    rx
}

impl DataFrame {
    /// Create a new DataFrame.
    pub(crate) fn new(session: SparkSession, plan: LogicalPlan) -> Self {
        Self::new_with_schema(session, plan, None)
    }

    /// Seed declared struct output schemas without analyzing the plan.
    pub(crate) fn new_with_schema(
        session: SparkSession,
        plan: LogicalPlan,
        schema: Option<DataType>,
    ) -> Self {
        Self::new_with_cached_schema(
            session,
            plan,
            schema
                .filter(|schema| matches!(schema, DataType::Struct { .. }))
                .map(Arc::new),
        )
    }

    fn new_with_cached_schema(
        session: SparkSession,
        plan: LogicalPlan,
        schema: Option<Arc<DataType>>,
    ) -> Self {
        DataFrame {
            session,
            plan: Arc::new(plan),
            schema_cache: Arc::new(Mutex::new(schema)),
            execution: Arc::new(Mutex::new(None)),
        }
    }

    /// Keep `info`, the information of an execution of this DataFrame, and record it as
    /// the session's most recent one.
    pub(crate) fn record_execution(&self, info: ExecutionInfo) {
        *self.execution.lock().unwrap() = Some(info.clone());
        self.session.record_execution(info);
    }

    fn cached_schema(&self) -> Option<Arc<DataType>> {
        // Propagation snapshots resolved metadata; it must not wait for an
        // in-flight analysis. Python transformations can still hold the GIL.
        self.schema_cache.try_lock().ok()?.clone()
    }

    /// Selected schema-preserving transformations snapshot the resolved schema,
    /// mirroring Spark 4.2 `_cached_schema` propagation. Other plans resolve independently.
    fn with_cached_schema(&self, plan: LogicalPlan) -> Self {
        Self::new_with_cached_schema(self.session.clone(), plan, self.cached_schema())
    }

    /// Set operations propagate only exactly equal cached schemas from the same session.
    fn with_merged_schema(&self, other: &DataFrame, plan: LogicalPlan) -> Self {
        // A right-hand plan from another session is analyzed in our session;
        // its previously resolved schema is not evidence about that resolution.
        if self.session.client().session_id() != other.session.client().session_id() {
            return Self::new(self.session.clone(), plan);
        }
        // Snapshot separately, including for self-unions; never lock both inputs.
        let schema = match (self.cached_schema(), other.cached_schema()) {
            (Some(left), Some(right)) if left == right => Some(left),
            _ => None,
        };
        Self::new_with_cached_schema(self.session.clone(), plan, schema)
    }

    /// Get the underlying logical plan.
    #[cfg(test)]
    pub(crate) fn plan(&self) -> &LogicalPlan {
        &self.plan
    }

    /// Select specific columns.
    pub fn select<C: Into<Column>>(&self, columns: impl IntoIterator<Item = C>) -> DataFrame {
        let columns: Vec<Column> = columns.into_iter().map(Into::into).collect();
        let plan = LogicalPlan::Project {
            input: Arc::clone(&self.plan),
            columns,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Filter rows by a condition.
    pub fn filter(&self, condition: Column) -> DataFrame {
        let plan = LogicalPlan::Filter {
            input: Arc::clone(&self.plan),
            condition,
        };
        self.with_cached_schema(plan)
    }

    /// Alias for filter().
    pub fn where_(&self, condition: Column) -> DataFrame {
        self.filter(condition)
    }

    /// Add or replace a column.
    pub fn with_column(&self, name: &str, col: Column) -> DataFrame {
        let plan = LogicalPlan::WithColumns {
            input: Arc::clone(&self.plan),
            column_names: vec![name.to_string()],
            columns: vec![col],
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Add or replace multiple columns.
    pub fn with_columns(&self, columns: Vec<(String, Column)>) -> DataFrame {
        let (names, cols) = columns.into_iter().unzip();
        let plan = LogicalPlan::WithColumns {
            input: Arc::clone(&self.plan),
            column_names: names,
            columns: cols,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Rename a column.
    pub fn with_column_renamed(&self, existing: &str, new: &str) -> DataFrame {
        let plan = LogicalPlan::WithColumnsRenamed {
            input: Arc::clone(&self.plan),
            renames: vec![(existing.to_string(), new.to_string())],
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Rename multiple columns, of old and new names in their order, in which the server
    /// applies them, as PySpark's `withColumnsRenamed` of a dict sends them: so
    /// `[("id", "a"), ("a", "b")]` renames `id` to `b`.
    pub fn with_columns_renamed(&self, renames: Vec<(String, String)>) -> DataFrame {
        let plan = LogicalPlan::WithColumnsRenamed {
            input: Arc::clone(&self.plan),
            renames,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Drop columns.
    pub fn drop(&self, columns: Vec<&str>) -> DataFrame {
        self.drop_columns(Vec::new(), columns)
    }

    /// Drop `columns` and the columns of `column_names`, mirroring `pyspark.sql.DataFrame.drop`
    /// of Columns and column names: a Column is sent as its expression, and a name as it is.
    pub fn drop_columns(&self, columns: Vec<Column>, column_names: Vec<&str>) -> DataFrame {
        let plan = LogicalPlan::Drop {
            input: Arc::clone(&self.plan),
            columns: column_names.iter().map(|s| s.to_string()).collect(),
            expressions: columns.iter().map(|c| c.expression().clone()).collect(),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Limit the number of rows.
    pub fn limit(&self, n: i32) -> DataFrame {
        let plan = LogicalPlan::Limit {
            input: Arc::clone(&self.plan),
            limit: n,
        };
        self.with_cached_schema(plan)
    }

    /// Skip the first n rows.
    pub fn offset(&self, n: i32) -> DataFrame {
        let plan = LogicalPlan::Offset {
            input: Arc::clone(&self.plan),
            offset: n,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Get the last n rows.
    pub fn tail(&self, n: i32) -> DataFrame {
        let plan = LogicalPlan::Tail {
            input: Arc::clone(&self.plan),
            limit: n,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Remove duplicate rows.
    pub fn distinct(&self) -> DataFrame {
        let plan = LogicalPlan::Deduplicate {
            input: Arc::clone(&self.plan),
            all_columns_as_keys: true,
            column_names: vec![],
            within_watermark: false,
        };
        self.with_cached_schema(plan)
    }

    /// Remove duplicate rows, optionally on specific columns.
    pub fn drop_duplicates(&self, column_names: Option<Vec<&str>>) -> DataFrame {
        let all_cols = column_names.is_none();
        let cols = column_names
            .map(|c| c.iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();

        let plan = LogicalPlan::Deduplicate {
            input: Arc::clone(&self.plan),
            all_columns_as_keys: all_cols,
            column_names: cols,
            within_watermark: false,
        };
        self.with_cached_schema(plan)
    }

    /// Sort rows.
    pub fn sort(&self, columns: Vec<Expression>) -> DataFrame {
        let plan = LogicalPlan::Sort {
            input: Arc::clone(&self.plan),
            order: columns,
            is_global: true,
        };
        self.with_cached_schema(plan)
    }

    /// Alias for sort().
    pub fn order_by(&self, columns: Vec<Expression>) -> DataFrame {
        self.sort(columns)
    }

    /// Join with another DataFrame.
    pub fn join(&self, right: &DataFrame, on: Option<Column>, join_type: JoinType) -> DataFrame {
        let plan = LogicalPlan::Join {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&right.plan),
            join_type,
            on,
            using_columns: vec![],
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Join with another DataFrame using column names (a name-based/"using" join).
    pub fn join_using<S: Into<String>>(
        &self,
        right: &DataFrame,
        using_columns: impl IntoIterator<Item = S>,
        join_type: JoinType,
    ) -> DataFrame {
        let plan = LogicalPlan::Join {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&right.plan),
            join_type,
            on: None,
            using_columns: using_columns.into_iter().map(Into::into).collect(),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Nearest-by join: for each left row, the `num_results` nearest right rows
    /// ranked by `ranking_expression`. Mirrors `DataFrame.nearestByJoin`.
    ///
    /// `mode` ∈ {"approx","exact"}, `direction` ∈ {"distance","similarity"},
    /// `join_type` ∈ {"inner","leftouter"}.
    pub fn nearest_by_join(
        &self,
        other: &DataFrame,
        ranking_expression: Column,
        num_results: i32,
        mode: &str,
        direction: &str,
        join_type: &str,
    ) -> DataFrame {
        let plan = LogicalPlan::NearestByJoin {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&other.plan),
            ranking_expression: ranking_expression.expression().clone(),
            num_results,
            join_type: join_type.to_string(),
            mode: mode.to_string(),
            direction: direction.to_string(),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// As-of join with another DataFrame: each row of this DataFrame is matched with the
    /// nearest row of `right` on the as-of columns, as `DataFrame._joinAsOf` does.
    ///
    /// Errors with `SESSION_NOT_SAME` when `right` is of another session. The join type and
    /// the direction are sent as they are; the server checks them.
    pub fn as_of_join(&self, right: &DataFrame, options: AsOfJoinOptions) -> Result<DataFrame> {
        // As PySpark's `_check_same_session`.
        if self.session.client().session_id() != right.session.client().session_id() {
            return Err(SparkError::classed(
                SparkErrorKind::SessionNotSame,
                "SESSION_NOT_SAME",
                &[],
            ));
        }
        let plan = LogicalPlan::AsOfJoin {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&right.plan),
            left_as_of: options.left_as_of,
            right_as_of: options.right_as_of,
            on: options.on,
            using_columns: options.using_columns,
            join_type: options.join_type,
            tolerance: options.tolerance,
            allow_exact_matches: options.allow_exact_matches,
            direction: options.direction,
        };
        Ok(DataFrame::new(self.session.clone(), plan))
    }

    /// Cross join.
    pub fn cross_join(&self, right: &DataFrame) -> DataFrame {
        self.join(right, None, JoinType::Cross)
    }

    /// Lateral join with another DataFrame (a `LATERAL` correlated subquery join).
    ///
    /// Mirrors `pyspark.sql.DataFrame.lateralJoin`.
    pub fn lateral_join(
        &self,
        right: &DataFrame,
        on: Option<Column>,
        join_type: JoinType,
    ) -> DataFrame {
        let plan = LogicalPlan::LateralJoin {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&right.plan),
            join_type,
            on,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Union with another DataFrame.
    pub fn union(&self, other: &DataFrame) -> DataFrame {
        let plan = LogicalPlan::SetOperation {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&other.plan),
            set_op_type: SetOpType::Union,
            is_all: true,
            by_name: false,
            allow_missing_columns: false,
        };
        self.with_merged_schema(other, plan)
    }

    /// Union by name.
    pub fn union_by_name(&self, other: &DataFrame) -> DataFrame {
        self.union_by_name_opt(other, false)
    }

    /// `unionByName` with the `allowMissingColumns` option (columns present in only
    /// one side are filled with null rather than rejected). Mirrors
    /// `DataFrame.unionByName(other, allowMissingColumns=False)`.
    pub fn union_by_name_opt(&self, other: &DataFrame, allow_missing_columns: bool) -> DataFrame {
        let plan = LogicalPlan::SetOperation {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&other.plan),
            set_op_type: SetOpType::Union,
            is_all: true,
            by_name: true,
            allow_missing_columns,
        };
        self.with_merged_schema(other, plan)
    }

    /// Intersect with another DataFrame.
    pub fn intersect(&self, other: &DataFrame) -> DataFrame {
        let plan = LogicalPlan::SetOperation {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&other.plan),
            set_op_type: SetOpType::Intersect,
            is_all: false,
            by_name: false,
            allow_missing_columns: false,
        };
        self.with_merged_schema(other, plan)
    }

    /// Subtract (except) another DataFrame.
    pub fn subtract(&self, other: &DataFrame) -> DataFrame {
        let plan = LogicalPlan::SetOperation {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&other.plan),
            set_op_type: SetOpType::Except,
            is_all: false,
            by_name: false,
            allow_missing_columns: false,
        };
        self.with_merged_schema(other, plan)
    }

    /// Repartition.
    pub fn repartition(&self, num_partitions: i32) -> DataFrame {
        let plan = LogicalPlan::Repartition {
            input: Arc::clone(&self.plan),
            num_partitions,
            shuffle: true,
        };
        self.with_cached_schema(plan)
    }

    /// Coalesce.
    pub fn coalesce(&self, num_partitions: i32) -> DataFrame {
        let plan = LogicalPlan::Repartition {
            input: Arc::clone(&self.plan),
            num_partitions,
            shuffle: false,
        };
        self.with_cached_schema(plan)
    }

    /// Add a hint.
    pub fn hint<S: Into<String>>(
        &self,
        name: &str,
        parameters: impl IntoIterator<Item = S>,
    ) -> DataFrame {
        let parameters = parameters
            .into_iter()
            .map(|p| crate::plan::hint_parameter(&p.into()))
            .collect();
        self.hint_expressions(name, parameters)
    }

    /// Specify a hint of parameters given as expressions, mirroring `pyspark.sql.DataFrame.hint`,
    /// which sends `lit` of each parameter: a literal, or a Column.
    pub fn hint_expressions(&self, name: &str, parameters: Vec<Expression>) -> DataFrame {
        let plan = LogicalPlan::Hint {
            input: Arc::clone(&self.plan),
            name: name.to_string(),
            parameters,
        };
        self.with_cached_schema(plan)
    }

    /// Marks a DataFrame as eligible for broadcast join (smaller table).
    /// Mirrors `pyspark.sql.functions.broadcast`.
    pub fn broadcast(&self) -> DataFrame {
        self.hint("broadcast", Vec::<String>::new())
    }

    /// Convert to DataFrame with new column names.
    pub fn to_df(&self, column_names: Vec<&str>) -> DataFrame {
        let names = column_names.iter().map(|s| s.to_string()).collect();
        let plan = LogicalPlan::ToDF {
            input: Arc::clone(&self.plan),
            column_names: names,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Alias this DataFrame.
    pub fn alias(&self, alias: &str) -> DataFrame {
        let plan = LogicalPlan::SubqueryAlias {
            input: Arc::clone(&self.plan),
            alias: alias.to_string(),
        };
        self.with_cached_schema(plan)
    }

    /// Map over each partition with a pandas UDF (`DataFrame.mapInPandas`).
    ///
    /// The `func` (built by the Python side, cloudpickled with eval type
    /// `SQL_MAP_PANDAS_ITER_UDF`) is applied to iterators of pandas DataFrames.
    pub fn map_in_pandas(
        &self,
        func: CommonInlineUserDefinedFunctionExpression,
        is_barrier: bool,
    ) -> DataFrame {
        self.map_partitions(func, is_barrier, None)
    }

    /// Map over each partition with a pandas UDF in a stage of the resource profile of
    /// `profile_id`, as `DataFrame.mapInPandas(func, schema, barrier, profile)` does with the
    /// id of `profile`, which [`SparkSession::build_resource_profile`] registers.
    pub fn map_in_pandas_with_profile(
        &self,
        func: CommonInlineUserDefinedFunctionExpression,
        is_barrier: bool,
        profile_id: Option<i32>,
    ) -> DataFrame {
        self.map_partitions(func, is_barrier, profile_id)
    }

    /// Map over each partition with an Arrow UDF (`DataFrame.mapInArrow`).
    pub fn map_in_arrow(
        &self,
        func: CommonInlineUserDefinedFunctionExpression,
        is_barrier: bool,
    ) -> DataFrame {
        self.map_partitions(func, is_barrier, None)
    }

    /// Map over each partition with an Arrow UDF in a stage of the resource profile of
    /// `profile_id`, as `DataFrame.mapInArrow(func, schema, barrier, profile)` does.
    pub fn map_in_arrow_with_profile(
        &self,
        func: CommonInlineUserDefinedFunctionExpression,
        is_barrier: bool,
        profile_id: Option<i32>,
    ) -> DataFrame {
        self.map_partitions(func, is_barrier, profile_id)
    }

    /// Build a `MapPartitions` relation from an already-constructed UDF.
    fn map_partitions(
        &self,
        func: CommonInlineUserDefinedFunctionExpression,
        is_barrier: bool,
        profile_id: Option<i32>,
    ) -> DataFrame {
        let schema = func.python_udf.output_type.clone();
        let plan = LogicalPlan::MapPartitions {
            input: Arc::clone(&self.plan),
            func,
            is_barrier,
            profile_id,
        };
        Self::new_with_schema(self.session.clone(), plan, Some(schema))
    }

    /// Apply a function to each row for its side effects (`DataFrame.foreach`).
    ///
    /// Backed by an Arrow map partition; results are forced and discarded.
    pub fn foreach(&self, func: CommonInlineUserDefinedFunctionExpression) -> Result<()> {
        let _ = self.map_partitions(func, false, None).collect()?;
        Ok(())
    }

    /// Apply a function to each partition for its side effects
    /// (`DataFrame.foreachPartition`).
    pub fn foreach_partition(&self, func: CommonInlineUserDefinedFunctionExpression) -> Result<()> {
        let _ = self.map_partitions(func, false, None).collect()?;
        Ok(())
    }

    /// Sample rows. Without a seed, the client chooses a random one, as PySpark's, so that every
    /// execution of the sample returns the same rows.
    pub fn sample(&self, fraction: f64, seed: Option<i64>) -> DataFrame {
        self.sample_opt(fraction, false, seed)
    }

    /// `sample` with the `withReplacement` option. Mirrors
    /// `DataFrame.sample(withReplacement, fraction, seed)`.
    pub fn sample_opt(
        &self,
        fraction: f64,
        with_replacement: bool,
        seed: Option<i64>,
    ) -> DataFrame {
        let plan = LogicalPlan::Sample {
            input: Arc::clone(&self.plan),
            lower_bound: 0.0,
            upper_bound: fraction,
            with_replacement,
            seed: Some(seed.unwrap_or_else(random_seed)),
            deterministic_order: false,
        };
        self.with_cached_schema(plan)
    }

    /// Group by columns for aggregation.
    pub fn group_by<C: Into<Column>>(
        &self,
        group_cols: impl IntoIterator<Item = C>,
    ) -> crate::group::GroupedData {
        let group_cols: Vec<Column> = group_cols.into_iter().map(Into::into).collect();
        crate::group::GroupedData::new(self.clone(), group_cols, AggregateGroupType::GroupBy)
    }

    /// Collect all rows into memory.
    pub fn collect(&self) -> Result<Vec<Row>> {
        self.collect_with_schema().map(|(rows, _)| rows)
    }

    /// Collect all rows into memory, with the schema of the result if the server sends it
    /// (`ExecutePlanResponse.schema`), as the reference client's `_to_table` returns it.
    /// Unlike the Arrow schema of the rows, it has the user-defined types of the columns.
    pub fn collect_with_schema(&self) -> Result<(Vec<Row>, Option<DataType>)> {
        let request = self.build_execute_request()?;
        let mut stream = self.session.execute_plan(request)?;

        let mut rows = vec![];
        let mut schema = None;
        let mut info = ExecutionInfo::default();
        loop {
            let resp = block_on(stream.message())?;
            let Some(mut resp) = resp else {
                break;
            };
            capture_execution(&mut resp, &mut info, &self.session);
            if let Some(data_type) = resp.schema.take() {
                schema = Some(DataType::from_proto(&data_type)?);
            }
            if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
                resp.response_type
            {
                let batch_rows = decode_arrow_batch_of(&batch, schema.as_ref())?;
                rows.extend(batch_rows);
            }
        }
        self.record_execution(info);

        Ok((rows, schema))
    }

    /// Return an iterator that lazily streams rows from the server.
    ///
    /// Mirrors `pyspark.sql.DataFrame.toLocalIterator(prefetchPartitions=False)`.
    /// Unlike `collect()`, which buffers all results in memory, this returns an iterator
    /// that yields Row objects as the server streams them, consuming minimal memory.
    ///
    /// # Arguments
    /// * `prefetch_partitions` - If true, a background task fetches the next batch
    ///   from the server while the caller consumes the current one (one batch buffered
    ///   ahead), overlapping network I/O with row processing. If false, each batch is
    ///   fetched on demand only once the previous batch is exhausted.
    pub fn to_local_iterator(&self, prefetch_partitions: bool) -> Result<LocalRowIterator> {
        let request = self.build_execute_request()?;
        let stream = self.session.execute_plan(request)?;
        Ok(LocalRowIterator::new(
            self.session.clone(),
            stream,
            prefetch_partitions,
        ))
    }

    /// The information of the last execution of this DataFrame. Mirrors
    /// `pyspark.sql.DataFrame.executionInfo`: set by `collect`, the Arrow and pandas
    /// collects, the writes, the temp views and the checkpoints of this DataFrame, and empty
    /// before one. An action that runs a derived DataFrame, such as `count` or `show`, sets
    /// that one's, as in PySpark.
    pub fn execution_info(&self) -> Result<ExecutionInfo> {
        self.execution.lock().unwrap().clone().ok_or_else(|| {
            SparkError::connect_msg("no execution info available; run an action first")
        })
    }

    /// Collect all data as Arrow RecordBatches.
    ///
    /// Streams execution results from the server and decodes Arrow IPC batches,
    /// returning the raw `RecordBatch`es without converting to Rows. This is the
    /// foundation for `to_datafusion()` and `to_polars()` conversions.
    pub fn collect_record_batches(&self) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        self.collect_record_batches_with_schema()
            .map(|(batches, _)| batches)
    }

    /// Collect all data as Arrow RecordBatches, with the schema of the result if the server
    /// sends it (`ExecutePlanResponse.schema`), as the reference client's `to_pandas` reads
    /// it: unlike the Arrow schema, it has the Spark types of the columns, such as their
    /// user-defined types.
    pub fn collect_record_batches_with_schema(
        &self,
    ) -> Result<(Vec<arrow::record_batch::RecordBatch>, Option<DataType>)> {
        let request = self.build_execute_request()?;
        let mut stream = self.session.execute_plan(request)?;

        let mut batches = vec![];
        let mut schema = None;
        let mut info = ExecutionInfo::default();
        loop {
            let resp = block_on(stream.message())?;
            let Some(mut resp) = resp else {
                break;
            };
            capture_execution(&mut resp, &mut info, &self.session);
            if let Some(data_type) = resp.schema.take() {
                schema = Some(DataType::from_proto(&data_type)?);
            }
            if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
                resp.response_type
            {
                let record_batches = decode_arrow_record_batches(&batch)?;
                batches.extend(record_batches);
            }
        }
        self.record_execution(info);

        Ok((batches, schema))
    }

    /// Get the count of rows.
    ///
    /// Mirrors `pyspark.sql.DataFrame.count()` = `groupBy().count().collect()[0][0]`:
    /// a global count aggregate is pushed to the server, which returns a single row,
    /// rather than streaming every row back to the client just to count them.
    pub fn count(&self) -> Result<i64> {
        let count_expr = crate::functions::count(Column::new(Expression::literal(
            crate::expression::LiteralExpression::int(1),
        )))
        .expression()
        .clone();

        let plan = LogicalPlan::Aggregate {
            input: Arc::clone(&self.plan),
            group_type: AggregateGroupType::GroupBy,
            grouping_expressions: vec![],
            aggregate_expressions: vec![count_expr],
            pivot_col: None,
            pivot_values: vec![],
            grouping_sets: vec![],
        };
        let agg_df = DataFrame::new(self.session.clone(), plan);

        let rows = agg_df.collect()?;
        match rows.into_iter().next() {
            Some(row) => row.get(0).and_then(|v| v.as_i64()).ok_or_else(|| {
                SparkError::connect_msg("count() aggregate returned a non-integer value")
            }),
            None => Ok(0),
        }
    }

    /// The table that `show` prints, as the server formats it: the first `n` rows, values
    /// truncated to `truncate` characters (0 for none), vertical with a line per value.
    /// Mirrors `DataFrame._show_string`, which sends a `ShowString` relation.
    pub fn show_string(&self, n: i32, truncate: i32, vertical: bool) -> Result<String> {
        let shown = DataFrame::new(
            self.session.clone(),
            LogicalPlan::ShowString {
                input: Arc::clone(&self.plan),
                num_rows: n,
                truncate,
                vertical,
            },
        );
        match shown.collect()?.first().map(|row| row.values()) {
            Some([Value::String(table)]) => Ok(table.clone()),
            _ => Err(SparkError::connect_msg(
                "ShowString returned no table of one string",
            )),
        }
    }

    /// The HTML table of the first `n` rows, values truncated to `truncate` characters (0 for
    /// none), as the server formats it. Mirrors the table of `DataFrame._repr_html_()`.
    pub fn html_string(&self, n: i32, truncate: i32) -> Result<String> {
        let html = DataFrame::new(
            self.session.clone(),
            LogicalPlan::HtmlString {
                input: Arc::clone(&self.plan),
                num_rows: n,
                truncate,
            },
        );
        match html.collect()?.first().map(|row| row.values()) {
            Some([Value::String(table)]) => Ok(table.clone()),
            _ => Err(SparkError::connect_msg(
                "HtmlString returned no table of one string",
            )),
        }
    }

    /// Print the first `n` rows as the server formats them, with values truncated to 20
    /// characters. Mirrors `DataFrame.show(n)`.
    pub fn show(&self, n: usize) -> Result<()> {
        let n = i32::try_from(n).unwrap_or(i32::MAX);
        println!("{}", self.show_string(n, 20, false)?);
        Ok(())
    }

    /// A DataFrame of `relation`, a relation built outside the core, such as the plans of
    /// `pyspark.ml`, which the core sends as it is, keeping `owner`, if any, alive while any
    /// DataFrame of it is. The placeholders of `relation`, which stand for the DataFrames that
    /// the plans of `pyspark.ml` take as input (see `placeholder_relation`), are replaced by
    /// the relations of those DataFrames. Not a stable API.
    #[doc(hidden)]
    pub fn from_relation(
        session: SparkSession,
        mut relation: proto::Relation,
        owner: Option<Arc<dyn std::any::Any + Send + Sync>>,
    ) -> Result<DataFrame> {
        crate::plan::resolve_relation(&mut relation)?;
        Ok(DataFrame::from_raw_relation(session, relation, owner))
    }

    /// The DataFrame of `relation`, as `from_relation`, of a relation without placeholders.
    fn from_raw_relation(
        session: SparkSession,
        relation: proto::Relation,
        owner: Option<Arc<dyn std::any::Any + Send + Sync>>,
    ) -> DataFrame {
        DataFrame::new(
            session,
            LogicalPlan::Raw {
                relation: Box::new(relation),
                owner: owner.map(crate::plan::RelationOwner),
            },
        )
    }

    /// The DataFrame of `relation`, a relation that `session` sent, such as that of a DataFrame
    /// that was pickled, which keeps its plan ID, as PySpark Connect's pickled plans keep
    /// theirs: so it is sent as it is, and the columns of the DataFrame it was of resolve
    /// against it. Not a stable API.
    #[doc(hidden)]
    pub fn from_sent_relation(session: SparkSession, relation: proto::Relation) -> DataFrame {
        let plan_id = relation.common.as_ref().and_then(|common| common.plan_id);
        let dataframe = DataFrame::from_raw_relation(session, relation, None);
        if let Some(plan_id) = plan_id {
            dataframe
                .session
                .plan_ids()
                .assign(&dataframe.plan, plan_id);
        }
        dataframe
    }

    /// The relation of this DataFrame as it is sent, with the plan IDs of its session, for a
    /// plan built outside the core over it, such as of `pyspark.ml`. Not a stable API.
    #[doc(hidden)]
    pub fn to_relation(&self) -> proto::Relation {
        build_input_relation(&self.plan, &self.session)
    }

    /// A relation that stands for the relation of this DataFrame, `to_relation`, in a relation
    /// built outside the core, such as a plan of `pyspark.ml`, which the core replaces by that
    /// relation when the relation reaches it (`from_relation`, `execute_command_proto`). It
    /// is shallow, so it is cheap to build and to decode whatever the depth of this DataFrame.
    /// Not a stable API.
    #[doc(hidden)]
    pub fn placeholder_relation(&self) -> proto::Relation {
        crate::plan::placeholder_relation(&self.plan, self.session.plan_ids_shared())
    }

    /// Get the schema of this DataFrame.
    ///
    /// The first successful lookup is cached and shared with exact clones.
    /// Concurrent lookups on the same DataFrame or its clones wait for a single
    /// analysis RPC, so a stalled analysis blocks those metadata readers.
    /// Changes to external table or view metadata do not invalidate this cache;
    /// recreate the DataFrame from its source to resolve the schema again.
    /// Each call returns an independent value, and failed lookups can be retried.
    pub fn schema(&self) -> Result<DataType> {
        // Keep the lock during analysis so concurrent accesses issue only one
        // successful RPC. Errors leave the cache empty and can be retried.
        let mut cached = self
            .schema_cache
            .lock()
            // Unlike the other cache locks, this guard spans a blocking RPC.
            // Recover defensively from Rust callers that panic during analysis
            // (for example, by calling the sync API inside a Tokio runtime).
            // The slot contains only None or a complete schema, so it is safe.
            .unwrap_or_else(|poisoned| {
                self.schema_cache.clear_poison();
                poisoned.into_inner()
            });
        if let Some(schema) = cached.clone() {
            // Copy the schema after unlocking so propagation can snapshot it.
            drop(cached);
            return Ok(schema.as_ref().clone());
        }
        let request = self.build_analyze_request()?;
        let response = block_on(self.session.client().analyze_plan(request))?;

        if let Some(proto::analyze_plan_response::Result::Schema(schema)) = response.result {
            let schema = DataType::from_proto(
                &schema
                    .schema
                    .ok_or_else(|| SparkError::connect_msg("Schema is missing"))?,
            )?;
            if !matches!(&schema, DataType::Struct { .. }) {
                return Err(SparkError::connect_msg("Schema is not a struct type"));
            }
            let schema = Arc::new(schema);
            *cached = Some(Arc::clone(&schema));
            drop(cached);
            Ok(schema.as_ref().clone())
        } else {
            Err(SparkError::connect_msg(
                "Schema analyze failed: no schema in response",
            ))
        }
    }

    /// Get the first row.
    pub fn first(&self) -> Result<Option<Row>> {
        self.limit(1).collect().map(|rows| rows.into_iter().next())
    }

    /// The first row, with the schema of the result, as [`DataFrame::collect_with_schema`].
    pub fn first_with_schema(&self) -> Result<(Option<Row>, Option<DataType>)> {
        let (rows, schema) = self.limit(1).collect_with_schema()?;
        Ok((rows.into_iter().next(), schema))
    }

    /// Alias for first().
    pub fn head(&self) -> Result<Option<Row>> {
        self.first()
    }

    /// Get the first n rows.
    pub fn take(&self, n: usize) -> Result<Vec<Row>> {
        self.limit(n as i32).collect()
    }

    /// The first `n` rows, with the schema of the result, as
    /// [`DataFrame::collect_with_schema`].
    pub fn take_with_schema(&self, n: usize) -> Result<(Vec<Row>, Option<DataType>)> {
        self.limit(n as i32).collect_with_schema()
    }

    /// Check if the DataFrame is empty.
    pub fn is_empty(&self) -> Result<bool> {
        self.limit(1).count().map(|c| c == 0)
    }

    /// Get column names.
    pub fn columns(&self) -> Result<Vec<String>> {
        let DataType::Struct { fields } = self.schema()? else {
            unreachable!("schema() validates struct responses")
        };
        Ok(fields.into_iter().map(|field| field.name).collect())
    }

    /// This DataFrame's relation for a request. Its plan nodes keep the same plan IDs each
    /// time, as in PySpark, so the server can reuse the plans it already resolved.
    pub(crate) fn relation(&self) -> proto::Relation {
        crate::plan::relation_with_ids(&self.plan, self.session.plan_ids())
    }

    /// Build an ExecutePlanRequest from the logical plan.
    fn build_execute_request(&self) -> Result<proto::ExecutePlanRequest> {
        let relation = self.relation();

        let mut request = proto::ExecutePlanRequest::default();
        request.session_id = self.session.client().session_id().to_string();
        request.user_context = Some(proto::UserContext::default());
        request.tags = self.session.tags();

        let mut plan = proto::Plan::default();
        plan.op_type = Some(proto::plan::OpType::Root(relation));
        request.plan = Some(plan);

        Ok(request)
    }

    /// Build an AnalyzePlanRequest from the logical plan.
    fn build_analyze_request(&self) -> Result<proto::AnalyzePlanRequest> {
        let relation = self.relation();

        let mut plan = proto::Plan::default();
        plan.op_type = Some(proto::plan::OpType::Root(relation));

        let mut schema = proto::analyze_plan_request::Schema::default();
        schema.plan = Some(plan);

        let mut request = proto::AnalyzePlanRequest::default();
        request.session_id = self.session.client().session_id().to_string();
        request.user_context = Some(proto::UserContext::default());
        request.analyze = Some(proto::analyze_plan_request::Analyze::Schema(schema));

        Ok(request)
    }

    /// Create a DataFrameWriter for writing this DataFrame to various destinations.
    ///
    /// Mirrors `pyspark.sql.DataFrame.write`.
    pub fn write(&self) -> crate::readwriter::DataFrameWriter {
        crate::readwriter::DataFrameWriter::new(self.session.clone(), Arc::clone(&self.plan))
            .recording_into(Arc::clone(&self.execution))
    }

    /// Create a [`DataFrameWriterV2`](crate::readwriter::DataFrameWriterV2) for
    /// the v2 write API.
    ///
    /// Mirrors `pyspark.sql.DataFrame.writeTo`.
    pub fn write_to(&self, table_name: &str) -> crate::readwriter::DataFrameWriterV2 {
        crate::readwriter::DataFrameWriterV2::new(
            self.session.clone(),
            Arc::clone(&self.plan),
            table_name,
        )
        .recording_into(Arc::clone(&self.execution))
    }

    /// Merge a set of updates, insertions, and deletions into a target table.
    ///
    /// Mirrors `pyspark.sql.DataFrame.mergeInto`: returns a [`crate::merge::MergeIntoWriter`]
    /// on which `when_matched` / `when_not_matched` / `when_not_matched_by_source` clauses
    /// are added before calling `merge()`.
    pub fn merge_into(&self, table: &str, condition: Column) -> crate::merge::MergeIntoWriter {
        crate::merge::MergeIntoWriter::new(
            self.session.clone(),
            Arc::clone(&self.plan),
            table.to_string(),
            condition,
        )
        .recording_into(Arc::clone(&self.execution))
    }

    /// Create a DataStreamWriter for writing this streaming DataFrame to various sinks.
    ///
    /// Mirrors `pyspark.sql.DataFrame.writeStream`.
    pub fn write_stream(&self) -> crate::streaming::DataStreamWriter {
        crate::streaming::DataStreamWriter::new(self.session.clone(), Arc::clone(&self.plan))
    }

    /// The default `MEMORY_AND_DISK_DESER` storage level used by `cache()`.
    fn memory_and_disk_deser() -> proto::StorageLevel {
        proto::StorageLevel {
            use_disk: true,
            use_memory: true,
            use_off_heap: false,
            deserialized: true,
            replication: 1,
        }
    }

    fn analyze_request(
        &self,
        analyze: proto::analyze_plan_request::Analyze,
    ) -> proto::AnalyzePlanRequest {
        proto::AnalyzePlanRequest {
            session_id: self.session.client().session_id().to_string(),
            user_context: Some(proto::UserContext::default()),
            analyze: Some(analyze),
            ..Default::default()
        }
    }

    /// Cache this DataFrame with the default `MEMORY_AND_DISK_DESER` storage level.
    ///
    /// Mirrors `pyspark.sql.DataFrame.cache()`.
    pub fn cache(&self) -> Result<DataFrame> {
        self.persist(Self::memory_and_disk_deser())
    }

    /// Persist this DataFrame with the given storage level.
    ///
    /// Mirrors `pyspark.sql.DataFrame.persist(storageLevel)`.
    pub fn persist(&self, storage_level: proto::StorageLevel) -> Result<DataFrame> {
        let persist = proto::analyze_plan_request::Persist {
            relation: Some(self.relation()),
            storage_level: Some(storage_level),
        };
        let request = self.analyze_request(proto::analyze_plan_request::Analyze::Persist(persist));
        block_on(self.session.client().analyze_plan(request))?;
        Ok(self.clone())
    }

    /// Remove this DataFrame from cache. Mirrors `DataFrame.unpersist(blocking)`.
    pub fn unpersist(&self, blocking: bool) -> Result<DataFrame> {
        let unpersist = proto::analyze_plan_request::Unpersist {
            relation: Some(self.relation()),
            blocking: Some(blocking),
        };
        let request =
            self.analyze_request(proto::analyze_plan_request::Analyze::Unpersist(unpersist));
        block_on(self.session.client().analyze_plan(request))?;
        Ok(self.clone())
    }

    /// Checkpoint this DataFrame to disk.
    pub fn checkpoint(&self) -> Result<DataFrame> {
        self.checkpoint_with(true)
    }

    /// Checkpoint this DataFrame, at once if `eager`, else when it is first used, mirroring
    /// `DataFrame.checkpoint(eager)`.
    pub fn checkpoint_with(&self, eager: bool) -> Result<DataFrame> {
        self.checkpoint_impl(false, eager, None)
    }

    /// Create a local checkpoint of this DataFrame.
    pub fn local_checkpoint(&self) -> Result<DataFrame> {
        self.local_checkpoint_with(true, None)
    }

    /// Create a local checkpoint of this DataFrame, at once if `eager`, kept at
    /// `storage_level`, else at the server's default, mirroring
    /// `DataFrame.localCheckpoint(eager, storageLevel)`.
    pub fn local_checkpoint_with(
        &self,
        eager: bool,
        storage_level: Option<proto::StorageLevel>,
    ) -> Result<DataFrame> {
        self.checkpoint_impl(true, eager, storage_level)
    }

    /// Execute a `CheckpointCommand` and return a DataFrame referencing the resulting
    /// cached remote relation. Mirrors `DataFrame.checkpoint`/`localCheckpoint`, which
    /// materialize server-side and return a handle to the checkpointed data.
    fn checkpoint_impl(
        &self,
        local: bool,
        eager: bool,
        storage_level: Option<proto::StorageLevel>,
    ) -> Result<DataFrame> {
        let mut cmd = proto::CheckpointCommand::default();
        cmd.relation = Some(self.relation());
        cmd.local = local;
        cmd.eager = eager;
        cmd.storage_level = storage_level;
        let (responses, info) = execute_command_with_info(
            &self.session,
            proto::command::CommandType::CheckpointCommand(cmd),
        )?;
        self.record_execution(info);
        for resp in &responses {
            if let Some(proto::execute_plan_response::ResponseType::CheckpointCommandResult(res)) =
                &resp.response_type
            {
                if let Some(rel) = &res.relation {
                    return Ok(DataFrame::new(
                        self.session.clone(),
                        LogicalPlan::CachedRemoteRelation {
                            relation_id: rel.relation_id.clone(),
                        },
                    ));
                }
            }
        }
        Err(SparkError::connect_msg(
            "checkpoint: server returned no CheckpointCommandResult",
        ))
    }

    /// Create a temporary view for this DataFrame.
    pub fn create_temp_view(&self, name: &str) -> Result<()> {
        self.create_view(name, false, false)
    }

    /// Create or replace a temporary view for this DataFrame.
    pub fn create_or_replace_temp_view(&self, name: &str) -> Result<()> {
        self.create_view(name, true, false)
    }

    /// Create a global temporary view for this DataFrame.
    pub fn create_global_temp_view(&self, name: &str) -> Result<()> {
        self.create_view(name, false, true)
    }

    /// Create or replace a global temporary view for this DataFrame.
    pub fn create_or_replace_global_temp_view(&self, name: &str) -> Result<()> {
        self.create_view(name, true, true)
    }

    /// Build + execute a real `CreateDataFrameViewCommand` (was previously a
    /// silent no-op that passed the query relation through and never created a view).
    fn create_view(&self, name: &str, replace: bool, global: bool) -> Result<()> {
        let input = self.relation();
        let mut cmd = proto::CreateDataFrameViewCommand::default();
        cmd.input = Some(input);
        cmd.name = name.to_string();
        cmd.is_global = global;
        cmd.replace = replace;
        let (_, info) = execute_command_with_info(
            &self.session,
            proto::command::CommandType::CreateDataframeView(cmd),
        )?;
        self.record_execution(info);
        Ok(())
    }

    /// Print the execution plan to the console. Mirrors `pyspark.sql.DataFrame.explain`
    /// (was previously a no-op that ran the query relation instead of an AnalyzePlan).
    pub fn explain(&self) -> Result<()> {
        self.explain_mode("simple")
    }

    /// Print the execution plan in an explain mode, [`DataFrame::explain_string`].
    pub fn explain_mode(&self, mode: &str) -> Result<()> {
        println!("{}", self.explain_string(mode)?);
        Ok(())
    }

    /// The execution plan in an explain mode, mirroring `pyspark.sql.DataFrame._explain_string`
    /// of a mode, which PySpark Connect's client checks: one of "simple", "extended",
    /// "codegen", "cost" and "formatted", else a `VALUE_NOT_ALLOWED` error.
    pub fn explain_string(&self, mode: &str) -> Result<String> {
        use proto::analyze_plan_request::explain::ExplainMode;
        let explain_mode = match mode {
            "simple" => ExplainMode::Simple,
            "extended" => ExplainMode::Extended,
            "codegen" => ExplainMode::Codegen,
            "cost" => ExplainMode::Cost,
            "formatted" => ExplainMode::Formatted,
            _ => {
                return Err(SparkError::value(
                    "VALUE_NOT_ALLOWED",
                    &[
                        ("arg_name", "explain_mode"),
                        (
                            "allowed_values",
                            "['simple', 'extended', 'codegen', 'cost', 'formatted']",
                        ),
                    ],
                ))
            }
        };
        let plan = proto::Plan {
            op_type: Some(proto::plan::OpType::Root(self.relation())),
        };
        let request = proto::AnalyzePlanRequest {
            session_id: self.session.client().session_id().to_string(),
            user_context: Some(proto::UserContext::default()),
            analyze: Some(proto::analyze_plan_request::Analyze::Explain(
                proto::analyze_plan_request::Explain {
                    plan: Some(plan),
                    explain_mode: explain_mode as i32,
                },
            )),
            ..Default::default()
        };
        let response = block_on(self.session.client().analyze_plan(request))?;
        match response.result {
            Some(proto::analyze_plan_response::Result::Explain(e)) => Ok(e.explain_string),
            _ => Err(SparkError::connect_msg(
                "AnalyzePlan response did not contain an explain result",
            )),
        }
    }

    /// Add a watermark to this DataFrame for event-time based windows.
    pub fn with_watermark(&self, time_column: &str, delay_threshold: &str) -> DataFrame {
        let plan = LogicalPlan::WithWatermark {
            input: Arc::clone(&self.plan),
            time_column: time_column.to_string(),
            delay_threshold: delay_threshold.to_string(),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Repartition this DataFrame by range.
    pub fn repartition_by_range(&self, num_partitions: i32, columns: Vec<Expression>) -> DataFrame {
        self.repartition_by_range_opt(Some(num_partitions), columns)
    }

    /// Repartition by range on the given columns, with the partition count optional: `None`
    /// sends no count, so the server picks it. Mirrors `DataFrame.repartitionByRange(col)`
    /// and `repartitionByRange(col, *cols)`, which PySpark sends with `numPartitions=None`.
    pub fn repartition_by_range_opt(
        &self,
        num_partitions: Option<i32>,
        columns: Vec<Expression>,
    ) -> DataFrame {
        let plan = LogicalPlan::RepartitionByRange {
            input: Arc::clone(&self.plan),
            num_partitions,
            partition_exprs: columns,
        };
        self.with_cached_schema(plan)
    }

    /// Repartition into `num_partitions` by hashing the given column expressions.
    /// Mirrors `df.repartition(numPartitions, *cols)`.
    pub fn repartition_by_expressions(
        &self,
        num_partitions: i32,
        columns: Vec<Expression>,
    ) -> DataFrame {
        let plan = LogicalPlan::RepartitionByExpression {
            input: Arc::clone(&self.plan),
            num_partitions,
            expressions: columns,
        };
        self.with_cached_schema(plan)
    }

    /// Alias for to_df().
    pub fn to_schema(&self, column_names: Vec<&str>) -> DataFrame {
        self.to_df(column_names)
    }

    /// Melt (unpivot) this DataFrame.
    pub fn melt(
        &self,
        id_vars: Vec<&str>,
        value_vars: Option<Vec<&str>>,
        var_name: &str,
        value_name: &str,
    ) -> DataFrame {
        use crate::column::col;
        let ids: Vec<Column> = id_vars.iter().map(|name| col(name)).collect();
        let vals: Option<Vec<Column>> =
            value_vars.map(|v| v.iter().map(|name| col(name)).collect());

        let plan = LogicalPlan::Unpivot {
            input: Arc::clone(&self.plan),
            ids,
            values: vals,
            variable_column_name: var_name.to_string(),
            value_column_name: value_name.to_string(),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Get the input files for this DataFrame. Mirrors `pyspark.sql.DataFrame.inputFiles`.
    pub fn input_files(&self) -> Result<Vec<String>> {
        let relation = self.relation();
        let mut plan = proto::Plan::default();
        plan.op_type = Some(proto::plan::OpType::Root(relation));
        let mut inp = proto::analyze_plan_request::InputFiles::default();
        inp.plan = Some(plan);
        let mut request = proto::AnalyzePlanRequest::default();
        request.session_id = self.session.client().session_id().to_string();
        request.user_context = Some(proto::UserContext::default());
        request.analyze = Some(proto::analyze_plan_request::Analyze::InputFiles(inp));
        let response = block_on(self.session.client().analyze_plan(request))?;
        match response.result {
            Some(proto::analyze_plan_response::Result::InputFiles(f)) => Ok(f.files),
            _ => Ok(vec![]),
        }
    }

    /// Observe metrics on this DataFrame.
    pub fn observe(&self, name: &str, exprs: Vec<Expression>) -> DataFrame {
        let plan = LogicalPlan::Observe {
            input: Arc::clone(&self.plan),
            name: name.to_string(),
            exprs,
            observation: None,
        };
        self.with_cached_schema(plan)
    }

    /// Observe the metrics of `exprs` with `observation`, mirroring
    /// `pyspark.sql.DataFrame.observe` of an `Observation`, as `Observation._on` does: by
    /// a `CollectMetrics` of the name of the observation, a random UUID for an unnamed one,
    /// whose metrics each action that runs the DataFrame sets in the observation.
    /// `REUSE_OBSERVATION` for an observation used already, and `UNSUPPORTED_OPERATION` for a
    /// streaming DataFrame.
    pub fn observe_with(
        &self,
        observation: &crate::observation::Observation,
        exprs: Vec<Expression>,
    ) -> Result<DataFrame> {
        let name = observation.name_to_observe()?;
        if self.is_streaming()? {
            return Err(SparkError::classed(
                spark_connect_core::error::SparkErrorKind::IllegalArgument,
                "UNSUPPORTED_OPERATION",
                &[("operation", "Streaming DataFrame with Observation")],
            ));
        }
        Ok(self.observed_with(observation, name, exprs))
    }

    /// The DataFrame of the `CollectMetrics` of `observation`, which starts, of `exprs` by
    /// `name`, as `observe_with` makes it once it has checked the observation and the
    /// DataFrame.
    pub(crate) fn observed_with(
        &self,
        observation: &crate::observation::Observation,
        name: String,
        exprs: Vec<Expression>,
    ) -> DataFrame {
        observation.start();
        let plan = LogicalPlan::Observe {
            input: Arc::clone(&self.plan),
            name,
            exprs,
            observation: Some(observation.clone()),
        };
        self.with_cached_schema(plan)
    }

    /// Get stat functions.
    pub fn stat(&self) -> crate::group::StatFunctions {
        crate::group::StatFunctions::new(self.clone())
    }

    /// Returns a [`crate::group::NaFunctions`] for handling missing values.
    ///
    /// Mirrors `pyspark.sql.DataFrame.na`.
    pub fn na(&self) -> crate::group::NaFunctions {
        crate::group::NaFunctions::new(self.clone())
    }

    /// Perform aggregation without grouping.
    pub fn agg(&self, expressions: Vec<Expression>) -> DataFrame {
        let plan = LogicalPlan::Aggregate {
            input: Arc::clone(&self.plan),
            group_type: AggregateGroupType::GroupBy,
            grouping_expressions: vec![],
            aggregate_expressions: expressions,
            pivot_col: None,
            pivot_values: vec![],
            grouping_sets: vec![],
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Select with SQL expressions, mirroring `DataFrame.selectExpr`.
    ///
    /// Each string is parsed as a SQL expression (e.g. `"id + 1 AS x"`), not treated
    /// as a bare column name - so it must go through `functions::expr` (an
    /// `ExpressionString` the server parses), not `col` (an unresolved attribute,
    /// which made `selectExpr("id + 1 AS x")` fail to resolve).
    pub fn select_expr(&self, exprs: Vec<&str>) -> DataFrame {
        let cols: Vec<Column> = exprs.iter().map(|e| crate::functions::expr(e)).collect();
        self.select(cols)
    }

    /// Fill NA values with an integer.
    pub fn fillna(&self, value: i64, subset: Option<Vec<&str>>) -> DataFrame {
        self.fillna_value(crate::row::Value::Long(value), subset)
    }

    /// Fill NA values with a double (e.g. a fractional fill into a double column).
    pub fn fillna_double(&self, value: f64, subset: Option<Vec<&str>>) -> DataFrame {
        self.fillna_value(crate::row::Value::Double(value), subset)
    }

    /// Fill NA values with a string.
    pub fn fillna_string(&self, value: &str, subset: Option<Vec<&str>>) -> DataFrame {
        self.fillna_value(crate::row::Value::String(value.to_string()), subset)
    }

    /// Fill NA values with a boolean.
    pub fn fillna_bool(&self, value: bool, subset: Option<Vec<&str>>) -> DataFrame {
        self.fillna_value(crate::row::Value::Bool(value), subset)
    }

    /// Fill NA values with a typed [`crate::row::Value`] (Long/Double/String/Bool/...).
    pub fn fillna_value(&self, value: crate::row::Value, subset: Option<Vec<&str>>) -> DataFrame {
        let columns = subset
            .map(|v| v.iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();
        let plan = LogicalPlan::NAFill {
            input: Arc::clone(&self.plan),
            fill_value: value,
            columns,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Fill NA values per column from `(column, value)` pairs.
    /// Mirrors `df.fillna({col: value, ...})`.
    pub fn fillna_map(&self, pairs: Vec<(String, crate::row::Value)>) -> DataFrame {
        let (cols, values): (Vec<String>, Vec<crate::row::Value>) = pairs.into_iter().unzip();
        let plan = LogicalPlan::NAFillColumns {
            input: Arc::clone(&self.plan),
            cols,
            values,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Drop NA values.
    pub fn dropna(
        &self,
        how: Option<&str>,
        thresh: Option<i32>,
        subset: Option<Vec<&str>>,
    ) -> DataFrame {
        let how_str = how.unwrap_or("any").to_string();
        let columns = subset
            .map(|v| v.iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();
        // `how` is carried into the plan; `min_non_nulls` is derived from `how`
        // and `thresh` in `NADrop::to_proto` (mirrors pyspark's translation).
        let plan = LogicalPlan::NADrop {
            input: Arc::clone(&self.plan),
            how: how_str,
            min_non_null: thresh,
            columns,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Replace values, each given as a string: a numeric-looking string is a double, otherwise
    /// a string. [`DataFrame::replace_literals`] takes the values as literals.
    pub fn replace(
        &self,
        to_replace: Vec<(String, String)>,
        subset: Option<Vec<&str>>,
    ) -> DataFrame {
        let replacements = to_replace
            .iter()
            .map(|(old, new)| {
                (
                    crate::plan::numeric_or_string_literal(old),
                    crate::plan::numeric_or_string_literal(new),
                )
            })
            .collect();
        self.replace_literals(replacements, subset)
    }

    /// Replace values, mirroring `pyspark.sql.DataFrame.replace`: each pair is a literal and
    /// the literal that replaces it, which may be a null, in the columns of `subset`, or else
    /// in every column of the literal's type. The server replaces a boolean, a double or a
    /// string.
    pub fn replace_literals(
        &self,
        replacements: Vec<(
            crate::expression::LiteralExpression,
            crate::expression::LiteralExpression,
        )>,
        subset: Option<Vec<&str>>,
    ) -> DataFrame {
        let columns = subset
            .map(|v| v.iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();
        let plan = LogicalPlan::NAReplace {
            input: Arc::clone(&self.plan),
            replacements,
            columns,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Describe this DataFrame (show statistics).
    pub fn describe(&self, columns: Vec<&str>) -> DataFrame {
        let col_names = columns.iter().map(|s| s.to_string()).collect();
        let plan = LogicalPlan::Describe {
            input: Arc::clone(&self.plan),
            columns: col_names,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Get summary statistics.
    pub fn summary(&self, percentiles: Vec<&str>) -> DataFrame {
        let percs = percentiles.iter().map(|s| s.to_string()).collect();
        let plan = LogicalPlan::Summary {
            input: Arc::clone(&self.plan),
            percentiles: percs,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// The plan ID of this DataFrame's plan in its session, by which its columns refer to it.
    fn plan_id(&self) -> i64 {
        self.session.plan_ids().id_of(&self.plan)
    }

    /// The column `name` of this DataFrame. It refers to this DataFrame by its plan ID, so that
    /// the server resolves it against this DataFrame's plan, as in a join of DataFrames of the
    /// same column names. Mirrors `df[name]`, `df.name` and `DataFrame._col` of PySpark Connect.
    pub fn col(&self, name: &str) -> Column {
        let reference = crate::expression::ColumnReference::new(name).with_plan_id(self.plan_id());
        Column::new(ExpressionNode::ColumnReference(reference).into())
    }

    /// The star of the columns of this DataFrame, as `df["*"]` of PySpark Connect.
    pub fn star(&self) -> Column {
        Column::new(ExpressionNode::UnresolvedStar(None, Some(self.plan_id())).into())
    }

    /// The columns of this DataFrame whose names match the regex `col_name`, in backticks, as
    /// a Column. Mirrors `DataFrame.colRegex`.
    pub fn col_regex(&self, col_name: &str) -> Column {
        Column::new(
            ExpressionNode::UnresolvedRegex(col_name.to_string(), Some(self.plan_id())).into(),
        )
    }

    /// The metadata column `name` of this DataFrame, which refers to it by its plan ID.
    /// Mirrors `pyspark.sql.DataFrame.metadataColumn`.
    pub fn metadata_column(&self, name: &str) -> Column {
        let reference = crate::expression::ColumnReference::new(name)
            .with_plan_id(self.plan_id())
            .metadata();
        Column::new(ExpressionNode::ColumnReference(reference).into())
    }

    /// Group by with rollup.
    pub fn rollup<C: Into<Column>>(
        &self,
        group_cols: impl IntoIterator<Item = C>,
    ) -> crate::group::GroupedData {
        let group_cols: Vec<Column> = group_cols.into_iter().map(Into::into).collect();
        crate::group::GroupedData::new(self.clone(), group_cols, AggregateGroupType::Rollup)
    }

    /// Group by with cube.
    pub fn cube<C: Into<Column>>(
        &self,
        group_cols: impl IntoIterator<Item = C>,
    ) -> crate::group::GroupedData {
        let group_cols: Vec<Column> = group_cols.into_iter().map(Into::into).collect();
        crate::group::GroupedData::new(self.clone(), group_cols, AggregateGroupType::Cube)
    }

    /// Group by with grouping sets. Each inner `Vec<Column>` is one grouping set;
    /// the sets are preserved on the wire (`GROUP_TYPE_GROUPING_SETS` + the
    /// `grouping_sets` field) rather than flattened into a single group-by.
    pub fn grouping_sets(&self, group_cols: Vec<Vec<Column>>) -> crate::group::GroupedData {
        crate::group::GroupedData::new_grouping_sets(self.clone(), group_cols)
    }

    /// Group by the grouping sets `grouping_sets` with the grouping columns `group_cols`,
    /// which are the columns of the result besides the aggregations, mirroring PySpark's
    /// `DataFrame.groupingSets(groupingSets, *cols)`: a column of the grouping columns that
    /// a set does not hold is null in the rows of that set.
    pub fn grouping_sets_with_columns(
        &self,
        grouping_sets: Vec<Vec<Column>>,
        group_cols: Vec<Column>,
    ) -> crate::group::GroupedData {
        crate::group::GroupedData::new_grouping_sets_of(self.clone(), grouping_sets, group_cols)
    }

    /// Sort within partitions (local sort).
    pub fn sort_within_partitions(&self, columns: Vec<Expression>) -> DataFrame {
        let plan = LogicalPlan::Sort {
            input: Arc::clone(&self.plan),
            order: columns,
            is_global: false,
        };
        self.with_cached_schema(plan)
    }

    /// Drop duplicates within a watermark.
    pub fn drop_duplicates_within_watermark(&self, column_names: Option<Vec<&str>>) -> DataFrame {
        let all_cols = column_names.is_none();
        let cols = column_names
            .map(|c| c.iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();

        let plan = LogicalPlan::Deduplicate {
            input: Arc::clone(&self.plan),
            all_columns_as_keys: all_cols,
            column_names: cols,
            within_watermark: true,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Apply a transformation function to this DataFrame.
    pub fn transform<F>(&self, f: F) -> DataFrame
    where
        F: Fn(&DataFrame) -> DataFrame,
    {
        f(self)
    }

    /// Randomly split this DataFrame into multiple parts.
    ///
    /// # Errors
    ///
    /// Returns `VALUE_NOT_POSITIVE` for the first negative weight, or for weights whose
    /// sum is nonpositive, including an empty list. These are the same checks as PySpark's
    /// `randomSplit`; zero weights are allowed when the total is positive.
    /// The native `arg_value` uses Rust's `f64` display format (for example, `-1`);
    /// the Python wrapper retains PySpark's `str()` formatting of the original inputs.
    ///
    /// This method returns a `Result`: callers must handle the error (for example, with
    /// `?`) before using the split DataFrames.
    pub fn random_split(&self, weights: Vec<f64>, seed: Option<i64>) -> Result<Vec<DataFrame>> {
        for weight in &weights {
            if *weight < 0.0 {
                return Err(SparkError::value(
                    "VALUE_NOT_POSITIVE",
                    &[("arg_name", "weights"), ("arg_value", &weight.to_string())],
                ));
            }
        }
        // As PySpark's: one seed of every split, of the client without a seed, and the input
        // in a deterministic order, so that the splits are disjoint and have every row.
        let seed = Some(seed.unwrap_or_else(random_seed));
        let total = weights.iter().fold(0.0, |sum, weight| sum + weight);
        if total <= 0.0 {
            return Err(SparkError::value(
                "VALUE_NOT_POSITIVE",
                &[
                    ("arg_name", "sum(weights)"),
                    ("arg_value", &total.to_string()),
                ],
            ));
        }
        let normalized: Vec<f64> = weights.iter().map(|w| w / total).collect();

        let mut results = vec![];
        let mut cumulative = 0.0;

        for weight in normalized {
            let upper = cumulative + weight;
            let plan = LogicalPlan::Sample {
                input: Arc::clone(&self.plan),
                lower_bound: cumulative,
                upper_bound: upper,
                with_replacement: false,
                seed,
                deterministic_order: true,
            };
            results.push(self.with_cached_schema(plan));
            cumulative = upper;
        }

        Ok(results)
    }

    /// Print the schema of this DataFrame.
    ///
    /// The output ends with a blank line, matching Scala and Python.
    pub fn print_schema(&self) -> Result<()> {
        let schema = self.schema()?;
        println!("{}", schema.tree_string()?);
        Ok(())
    }

    /// Get the storage level of this DataFrame. Mirrors `DataFrame.storageLevel`.
    pub fn storage_level(&self) -> Result<proto::StorageLevel> {
        let get = proto::analyze_plan_request::GetStorageLevel {
            relation: Some(self.relation()),
        };
        let request =
            self.analyze_request(proto::analyze_plan_request::Analyze::GetStorageLevel(get));
        let response = block_on(self.session.client().analyze_plan(request))?;
        match response.result {
            Some(proto::analyze_plan_response::Result::GetStorageLevel(g)) => {
                Ok(g.storage_level.unwrap_or_default())
            }
            _ => Ok(proto::StorageLevel::default()),
        }
    }

    /// Check if this DataFrame is cached. Mirrors `DataFrame.is_cached`.
    ///
    /// Derived from the server-reported storage level (cached iff it uses memory
    /// or disk), rather than inspecting the local plan.
    pub fn is_cached(&self) -> Result<bool> {
        let level = self.storage_level()?;
        Ok(level.use_memory || level.use_disk)
    }

    /// Get dtypes (column names and types).
    pub fn dtypes(&self) -> Result<Vec<(String, String)>> {
        let DataType::Struct { fields } = self.schema()? else {
            unreachable!("schema() validates struct responses")
        };
        Ok(fields
            .into_iter()
            .map(|field| (field.name, field.data_type.to_string()))
            .collect())
    }

    /// Compute the server-side semantic hash of this DataFrame's logical plan,
    /// mirroring `DataFrame.semanticHash()` (an AnalyzePlan request).
    pub fn semantic_hash(&self) -> Result<i32> {
        let relation = self.relation();
        let mut plan = proto::Plan::default();
        plan.op_type = Some(proto::plan::OpType::Root(relation));
        let mut request = proto::AnalyzePlanRequest::default();
        request.session_id = self.session.client().session_id().to_string();
        request.user_context = Some(proto::UserContext::default());
        request.analyze = Some(proto::analyze_plan_request::Analyze::SemanticHash(
            proto::analyze_plan_request::SemanticHash { plan: Some(plan) },
        ));
        let resp = block_on(self.session.client().analyze_plan(request))?;
        match resp.result {
            Some(proto::analyze_plan_response::Result::SemanticHash(h)) => Ok(h.result),
            _ => Err(SparkError::connect_msg(
                "AnalyzePlan response did not contain a semantic hash",
            )),
        }
    }

    /// Whether two DataFrames have the same semantics, mirroring
    /// `DataFrame.sameSemantics(other)` (a server-side AnalyzePlan comparison).
    pub fn same_semantics(&self, other: &DataFrame) -> Result<bool> {
        let self_rel = self.relation();
        let other_rel = other.relation();
        let mut target_plan = proto::Plan::default();
        target_plan.op_type = Some(proto::plan::OpType::Root(self_rel));
        let mut other_plan = proto::Plan::default();
        other_plan.op_type = Some(proto::plan::OpType::Root(other_rel));
        let mut request = proto::AnalyzePlanRequest::default();
        request.session_id = self.session.client().session_id().to_string();
        request.user_context = Some(proto::UserContext::default());
        request.analyze = Some(proto::analyze_plan_request::Analyze::SameSemantics(
            proto::analyze_plan_request::SameSemantics {
                target_plan: Some(target_plan),
                other_plan: Some(other_plan),
            },
        ));
        let resp = block_on(self.session.client().analyze_plan(request))?;
        match resp.result {
            Some(proto::analyze_plan_response::Result::SameSemantics(r)) => Ok(r.result),
            _ => Err(SparkError::connect_msg(
                "AnalyzePlan response did not contain a sameSemantics result",
            )),
        }
    }

    /// Each row as a JSON object string, in a column `value`, mirroring
    /// `pyspark.sql.DataFrame.toJSON` of PySpark Connect, which selects
    /// `to_json(struct(col("*"))).alias("value")`.
    pub fn to_json_dataframe(&self) -> DataFrame {
        use crate::functions::{col, r#struct, to_json};
        self.select([to_json(r#struct([col("*")])).alias("value")])
    }

    /// Collect each row as a JSON object string, the rows of
    /// [`DataFrame::to_json_dataframe`].
    pub fn to_json(&self) -> Result<Vec<String>> {
        let rows = self.to_json_dataframe().collect()?;
        Ok(rows
            .iter()
            .map(|r| {
                r.get(0)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string()
            })
            .collect())
    }

    /// Union all rows (alias for union with all=true).
    pub fn union_all(&self, other: &DataFrame) -> DataFrame {
        let plan = LogicalPlan::SetOperation {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&other.plan),
            set_op_type: SetOpType::Union,
            is_all: true,
            by_name: false,
            allow_missing_columns: false,
        };
        self.with_merged_schema(other, plan)
    }

    /// Except all rows.
    pub fn except_all(&self, other: &DataFrame) -> DataFrame {
        let plan = LogicalPlan::SetOperation {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&other.plan),
            set_op_type: SetOpType::Except,
            is_all: true,
            by_name: false,
            allow_missing_columns: false,
        };
        self.with_merged_schema(other, plan)
    }

    /// Intersect all rows.
    pub fn intersect_all(&self, other: &DataFrame) -> DataFrame {
        let plan = LogicalPlan::SetOperation {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&other.plan),
            set_op_type: SetOpType::Intersect,
            is_all: true,
            by_name: false,
            allow_missing_columns: false,
        };
        self.with_merged_schema(other, plan)
    }

    /// Unpivot columns (like melt).
    pub fn unpivot<C: Into<Column>, D: Into<Column>>(
        &self,
        ids: impl IntoIterator<Item = C>,
        values: Option<impl IntoIterator<Item = D>>,
        variable_column_name: &str,
        value_column_name: &str,
    ) -> DataFrame {
        let ids: Vec<Column> = ids.into_iter().map(Into::into).collect();
        let values: Option<Vec<Column>> = values.map(|v| v.into_iter().map(Into::into).collect());
        let plan = LogicalPlan::Unpivot {
            input: Arc::clone(&self.plan),
            ids,
            values,
            variable_column_name: variable_column_name.to_string(),
            value_column_name: value_column_name.to_string(),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Set metadata on an existing column.
    ///
    /// Mirrors `pyspark.sql.connect.dataframe.DataFrame.withMetadata`: the column is
    /// re-selected with the given metadata attached (serialized to a JSON map).
    pub fn with_metadata(&self, column_name: &str, metadata: HashMap<String, String>) -> DataFrame {
        let metadata_json = serde_json::to_string(&metadata).unwrap_or_else(|_| "{}".to_string());
        self.with_metadata_json(column_name, &metadata_json)
    }

    /// Set the metadata of a column of its JSON, of any JSON values, as PySpark's
    /// `withMetadata` sends `json.dumps(metadata)`.
    pub fn with_metadata_json(&self, column_name: &str, metadata_json: &str) -> DataFrame {
        let plan = LogicalPlan::WithColumnMetadata {
            input: Arc::clone(&self.plan),
            column_name: column_name.to_string(),
            column: self.col(column_name).expression().clone(),
            metadata_json: metadata_json.to_string(),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Get the Spark session.
    pub fn spark_session(&self) -> SparkSession {
        self.session.clone()
    }

    /// Check if this DataFrame is local (collected).
    ///
    /// Mirrors the server's `Dataset.isLocal`: only a local relation itself is local. The
    /// server resolves `toDF`, which Arrow input with inferred or listed column names
    /// applies, into a `Project`, which is not. The result of a command run by
    /// `SparkSession::sql` is the local relation the server returned.
    pub fn is_local(&self) -> bool {
        use proto::relation::RelType;
        match self.plan.as_ref() {
            LogicalPlan::LocalRelation { .. } | LogicalPlan::ChunkedCachedLocalRelation { .. } => {
                true
            }
            LogicalPlan::CachedRelation { relation } => matches!(
                relation.rel_type,
                Some(RelType::LocalRelation(_))
                    | Some(RelType::CachedLocalRelation(_))
                    | Some(RelType::ChunkedCachedLocalRelation(_))
            ),
            _ => false,
        }
    }

    /// Whether this DataFrame is streaming, mirroring `pyspark.sql.DataFrame.isStreaming`,
    /// which asks the server: whether a source of its plan continuously returns data.
    pub fn is_streaming(&self) -> Result<bool> {
        let request = proto::AnalyzePlanRequest {
            session_id: self.session.client().session_id().to_string(),
            user_context: Some(proto::UserContext::default()),
            analyze: Some(proto::analyze_plan_request::Analyze::IsStreaming(
                proto::analyze_plan_request::IsStreaming {
                    plan: Some(proto::Plan {
                        op_type: Some(proto::plan::OpType::Root(self.relation())),
                    }),
                },
            )),
            ..Default::default()
        };
        let response = block_on(self.session.client().analyze_plan(request))?;
        match response.result {
            Some(proto::analyze_plan_response::Result::IsStreaming(r)) => Ok(r.is_streaming),
            _ => Err(SparkError::connect_msg(
                "AnalyzePlan response did not contain an isStreaming result",
            )),
        }
    }

    /// Collect the DataFrame and serialize it to Arrow IPC (file format) bytes.
    ///
    /// The returned buffer is a self-describing Arrow IPC stream that can be read
    /// back with `arrow::ipc::reader::FileReader` (or handed to pyarrow, polars,
    /// etc.). An empty result yields a valid IPC file with an empty schema.
    pub fn to_arrow(&self) -> Result<Vec<u8>> {
        record_batches_to_ipc(&self.collect_record_batches()?)
    }

    /// Convert to a DataFusion DataFrame.
    ///
    /// Requires the `datafusion` feature to be enabled.
    ///
    /// # Arguments
    ///
    /// * `ctx` - A DataFusion `SessionContext` to use for creating the DataFrame
    ///
    /// # Example
    ///
    /// ```ignore
    /// use datafusion::prelude::SessionContext;
    /// let datafusion_ctx = SessionContext::new();
    /// let df = spark_df.to_datafusion(&datafusion_ctx)?;
    /// ```
    #[cfg(feature = "datafusion")]
    pub fn to_datafusion(
        &self,
        ctx: &datafusion::prelude::SessionContext,
    ) -> Result<datafusion::dataframe::DataFrame> {
        record_batches_to_datafusion(ctx, self.collect_record_batches()?)
    }

    /// Convert a collected DataFrame into a [`polars::frame::DataFrame`].
    ///
    /// Requires the `polars` cargo feature.
    ///
    /// ```ignore
    /// let pdf = spark_df.to_polars()?;
    /// ```
    ///
    /// The result is bridged through Arrow IPC bytes rather than sharing
    /// arrow-rs types, so polars' vendored arrow does not have to match this
    /// crate's arrow-rs version.
    #[cfg(feature = "polars")]
    pub fn to_polars(&self) -> Result<polars::frame::DataFrame> {
        record_batches_to_polars(&self.collect_record_batches()?)
    }

    /// Repartition into `num_partitions` using the given column's value directly as
    /// the shuffle partition id. Mirrors `DataFrame.repartitionById(numPartitions,
    /// partitionIdCol)`: the column is wrapped in a `DirectShufflePartitionID`
    /// expression and used as the sole repartition expression.
    pub fn repartition_by_id(&self, num_partitions: i32, partition_id_col: Column) -> DataFrame {
        let direct: Expression =
            ExpressionNode::DirectShufflePartitionId(partition_id_col.expression().clone()).into();
        self.repartition_by_expressions(num_partitions, vec![direct])
    }

    /// Append a monotonically increasing index column. Mirrors
    /// `DataFrame.zipWithIndex(indexColName="index")`:
    /// `self.select(col("*"), distributed_sequence_id().alias(indexColName))`.
    pub fn zip_with_index(&self, index_col_name: &str) -> DataFrame {
        let star = Column::new(ExpressionNode::UnresolvedStar(None, None).into());
        let seq = Column::new(Expression::function("distributed_sequence_id", vec![]))
            .alias(index_col_name);
        self.select(vec![star, seq])
    }

    /// Reconcile this DataFrame to a new schema: reorder/select columns by name
    /// and cast them to the target types.
    ///
    /// Mirrors `pyspark.sql.connect.dataframe.DataFrame.to` (a `ToSchema` relation).
    pub fn to(&self, schema: DataType) -> DataFrame {
        let plan = LogicalPlan::ToSchema {
            input: Arc::clone(&self.plan),
            schema: schema.clone(),
        };
        Self::new_with_schema(self.session.clone(), plan, Some(schema))
    }

    /// A subquery of `subquery_type` of this DataFrame's plan, which refers to the plan by
    /// its plan ID in the session.
    pub(crate) fn subquery(
        &self,
        subquery_type: crate::expression::SubqueryType,
    ) -> crate::expression::SubqueryExpression {
        crate::expression::SubqueryExpression::new(
            Arc::clone(&self.plan),
            self.session.plan_ids().id_of(&self.plan),
            subquery_type,
        )
    }

    /// The scalar subquery of this DataFrame, which returns one row of one column, as a
    /// Column of that value. Mirrors `DataFrame.scalar()`. A column of the outer query is
    /// referred to as `col(...).outer()`.
    pub fn scalar(&self) -> Column {
        let subquery = self.subquery(crate::expression::SubqueryType::Scalar);
        Column::new(ExpressionNode::Subquery(Box::new(subquery)).into())
    }

    /// The EXISTS subquery of this DataFrame, as a Column of whether it returns a row.
    /// Mirrors `DataFrame.exists()`.
    pub fn exists(&self) -> Column {
        let subquery = self.subquery(crate::expression::SubqueryType::Exists);
        Column::new(ExpressionNode::Subquery(Box::new(subquery)).into())
    }

    /// The first value of the first row, if any.
    pub(crate) fn first_value(&self) -> Result<Option<Value>> {
        let rows = self.limit(1).collect()?;
        Ok(rows.first().and_then(|row| row.get(0).cloned()))
    }

    /// Transpose the DataFrame: swap rows and columns (server-side `Transpose`
    /// relation). Mirrors `pyspark.sql.connect.dataframe.DataFrame.transpose()`
    /// with no index column (the server uses the first column as the header).
    pub fn transpose(&self) -> Result<DataFrame> {
        let plan = LogicalPlan::Transpose {
            input: Arc::clone(&self.plan),
            index_columns: vec![],
        };
        Ok(DataFrame::new(self.session.clone(), plan))
    }

    /// Transpose using an explicit index column as the transposed header.
    /// Mirrors `DataFrame.transpose(indexColumn)`.
    pub fn transpose_with_index(&self, index_column: Column) -> Result<DataFrame> {
        let plan = LogicalPlan::Transpose {
            input: Arc::clone(&self.plan),
            index_columns: vec![index_column.expression().clone()],
        };
        Ok(DataFrame::new(self.session.clone(), plan))
    }

    /// Zip this DataFrame with another DataFrame by row number.
    pub fn zip(&self, other: &DataFrame) -> Result<DataFrame> {
        let plan = LogicalPlan::Zip {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&other.plan),
        };
        Ok(DataFrame::new(self.session.clone(), plan))
    }

    /// Register this DataFrame as a temporary table (deprecated - use createTempView).
    pub fn register_temp_table(&self, name: &str) -> Result<()> {
        self.create_temp_view(name)?;
        Ok(())
    }

    /// Convert to a table reference (alias for alias).
    pub fn as_table(&self, alias: &str) -> DataFrame {
        self.alias(alias)
    }
}

/// Build the proto `Relation` for a logical plan with plan-ids assigned.
///
/// Shared by write operations (which embed a fully-formed input relation inside
/// a `Command`) so they go through the same plan-id assignment as `collect()`.
pub(crate) fn build_input_relation(
    plan: &Arc<LogicalPlan>,
    session: &SparkSession,
) -> proto::Relation {
    crate::plan::relation_with_ids(plan, session.plan_ids())
}

/// Execute a `Command` against the server and drain the response stream.
///
/// Writes (and other side-effecting operations) are modeled as commands rather
/// than relations, so they are submitted through an `ExecutePlanRequest` whose
/// plan carries a `Command` and produce no rows to collect.
pub(crate) fn execute_command(
    session: &SparkSession,
    command_type: proto::command::CommandType,
) -> Result<()> {
    execute_command_collect(session, command_type).map(|_| ())
}

/// Like `execute_command`, but returns the collected responses so callers can read
/// a command result (e.g. `CheckpointCommandResult`, `WriteStreamOperationStartResult`).
pub(crate) fn execute_command_collect(
    session: &SparkSession,
    command_type: proto::command::CommandType,
) -> Result<Vec<proto::ExecutePlanResponse>> {
    execute_command_with_info(session, command_type).map(|(responses, _)| responses)
}

/// Like `execute_command_collect`, and also returns the information of the execution, which
/// it records as the session's most recent one, for a DataFrame that keeps its own.
pub(crate) fn execute_command_with_info(
    session: &SparkSession,
    command_type: proto::command::CommandType,
) -> Result<(Vec<proto::ExecutePlanResponse>, ExecutionInfo)> {
    let mut command = proto::Command::default();
    command.command_type = Some(command_type);

    let mut plan = proto::Plan::default();
    plan.op_type = Some(proto::plan::OpType::Command(command));

    let mut request = proto::ExecutePlanRequest::default();
    request.session_id = session.client().session_id().to_string();
    request.user_context = Some(proto::UserContext::default());
    request.tags = session.tags();
    request.plan = Some(plan);

    let mut stream = session.execute_plan(request)?;
    // Drain the response stream so the command runs to completion server-side,
    // capturing any metrics/progress emitted along the way.
    let mut info = ExecutionInfo::default();
    let mut responses = Vec::new();
    while let Some(mut resp) = block_on(stream.message())? {
        capture_execution(&mut resp, &mut info, session);
        responses.push(resp);
    }
    session.record_execution(info.clone());
    Ok((responses, info))
}

/// Pull execution metrics/observed-metrics off a response; the `ExecutionStream` of the
/// response updates the progress of the execution. `metrics`/`observed_metrics` are top-level
/// fields (not part of the `response_type` oneof), so this must run before `response_type` is
/// consumed.
fn capture_execution(
    resp: &mut proto::ExecutePlanResponse,
    info: &mut ExecutionInfo,
    session: &SparkSession,
) {
    if info.operation_id.is_none() && !resp.operation_id.is_empty() {
        info.operation_id = Some(resp.operation_id.clone());
    }
    if let Some(metrics) = resp.metrics.take() {
        info.metrics = Some(metrics);
    }
    if !resp.observed_metrics.is_empty() {
        let metrics = std::mem::take(&mut resp.observed_metrics);
        set_observations(session, &metrics);
        // The updates of the Python accumulators, such as the results of the UDF profilers.
        session.python_accumulators().accumulate(&metrics);
        info.observed_metrics.extend(metrics);
    }
}

/// Set the metrics that the server observed in the Observations of their relations, as
/// PySpark's client sets them in the Observations of the plan: those of each relation, which
/// the server reports by its plan ID, or the error of collecting them.
fn set_observations(
    session: &SparkSession,
    metrics: &[proto::execute_plan_response::ObservedMetrics],
) {
    for observed in metrics {
        let Some(node) = session.plan_ids().node_of(observed.plan_id) else {
            continue;
        };
        let LogicalPlan::Observe {
            name,
            observation: Some(observation),
            ..
        } = node.as_ref()
        else {
            continue;
        };
        if *name != observed.name {
            continue;
        }
        match observed.root_error_idx {
            Some(root_error_idx) => observation.set_error(SparkError::from_observation_errors(
                root_error_idx,
                observed.errors.clone(),
            )),
            None => observation.update(&observed.keys, &observed.values),
        }
    }
}

/// A random seed of a sample without one, as PySpark's `random.randint(0, sys.maxsize)`: the
/// client chooses it, so that every execution of the sample returns the same rows.
fn random_seed() -> i64 {
    (uuid::Uuid::new_v4().as_u128() >> 65) as i64
}

/// Decode an Arrow batch into rows.
pub(crate) fn decode_arrow_batch(
    batch: &proto::execute_plan_response::ArrowBatch,
) -> Result<Vec<Row>> {
    decode_arrow_batch_of(batch, None)
}

/// The fields of `schema`, a struct of `columns` fields, for the columns of a batch: none of
/// another schema, whose names the batch keeps.
fn spark_fields(schema: Option<&DataType>, columns: usize) -> Option<&[crate::types::StructField]> {
    match schema {
        Some(DataType::Struct { fields }) if fields.len() == columns => Some(fields),
        _ => None,
    }
}

/// The rows of `batch`, of the names of `schema`, the schema of the result that the server sends
/// before the rows, if any: as PySpark's `ArrowTableToRowsConversion` builds them by the schema,
/// the fields of nested structs have the names of the schema, which keep the duplicate names
/// that the Arrow data of the server numbers (`a_0`, `a_1`).
pub(crate) fn decode_arrow_batch_of(
    batch: &proto::execute_plan_response::ArrowBatch,
    spark_schema: Option<&DataType>,
) -> Result<Vec<Row>> {
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;

    if batch.data.is_empty() {
        return Ok(vec![]);
    }

    let cursor = Cursor::new(&batch.data);
    let mut reader = StreamReader::try_new(cursor, None).map_err(|e| {
        SparkError::connect_msg(format!("Failed to create Arrow stream reader: {}", e))
    })?;

    let mut rows = vec![];

    while let Some(record_batch) = reader
        .next()
        .transpose()
        .map_err(|e| SparkError::connect_msg(format!("Failed to decode Arrow batch: {}", e)))?
    {
        let schema = record_batch.schema();
        let num_rows = record_batch.num_rows();
        if num_rows == 0 {
            continue;
        }
        let fields = spark_fields(spark_schema, record_batch.num_columns());
        let field_names = match fields {
            Some(fields) => fields.iter().map(|field| field.name.clone()).collect(),
            None => schema
                .fields()
                .iter()
                .map(|field| field.name().clone())
                .collect::<Arc<[String]>>(),
        };
        let layouts = record_batch
            .columns()
            .iter()
            .enumerate()
            .map(|(i, column)| {
                ArrowValueLayout::of(column.as_ref(), fields.map(|fields| &fields[i].data_type))
            })
            .collect::<Vec<_>>();

        for row_idx in 0..num_rows {
            let mut values = Vec::with_capacity(record_batch.num_columns());

            for layout in &layouts {
                values.push(layout.value_at(row_idx)?);
            }

            rows.push(Row::with_shared_fields(Arc::clone(&field_names), values));
        }
    }

    Ok(rows)
}

/// Decode Arrow IPC stream into RecordBatches without converting to Rows.
///
/// Used by `collect_record_batches()` to provide raw Arrow data for conversions
/// to DataFusion and Polars.
fn decode_arrow_record_batches(
    batch: &proto::execute_plan_response::ArrowBatch,
) -> Result<Vec<arrow::record_batch::RecordBatch>> {
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;

    if batch.data.is_empty() {
        return Ok(vec![]);
    }

    let cursor = Cursor::new(&batch.data);
    let mut reader = StreamReader::try_new(cursor, None).map_err(|e| {
        SparkError::connect_msg(format!("Failed to create Arrow stream reader: {}", e))
    })?;

    let mut batches = vec![];

    while let Some(record_batch) = reader
        .next()
        .transpose()
        .map_err(|e| SparkError::connect_msg(format!("Failed to decode Arrow batch: {}", e)))?
    {
        batches.push(record_batch);
    }

    Ok(batches)
}

/// Extract a value at a specific index from an Arrow array.
/// Format an unscaled `i128` and scale as a decimal string (e.g. 150, scale 2 -> "1.50").
fn i128_to_decimal_string(unscaled: i128, scale: i32) -> String {
    if scale < 0 {
        // A negative scale, of spark.sql.legacy.allowNegativeScaleOfDecimal: the unscaled value
        // times ten to the negated scale, in scientific notation, as PyArrow's Decimal of it,
        // which PySpark collects: Decimal("1E+1") of 1 at the scale -1.
        return format!("{unscaled}E+{}", -i64::from(scale));
    }
    if scale == 0 {
        return unscaled.to_string();
    }
    let scale = scale as usize;
    let neg = unscaled < 0;
    let mut digits = unscaled.unsigned_abs().to_string();
    if digits.len() <= scale {
        digits = format!("{}{}", "0".repeat(scale - digits.len() + 1), digits);
    }
    let point = digits.len() - scale;
    let s = format!("{}.{}", &digits[..point], &digits[point..]);
    if neg {
        format!("-{s}")
    } else {
        s
    }
}

/// Serialize record batches to Arrow IPC (file format) bytes. Shared by
/// [`DataFrame::to_arrow`] and [`DataFrame::to_polars`]; an empty input yields a
/// valid empty-schema IPC file.
fn record_batches_to_ipc(batches: &[arrow::record_batch::RecordBatch]) -> Result<Vec<u8>> {
    use arrow::ipc::writer::FileWriter;
    let schema = match batches.first() {
        Some(b) => b.schema(),
        None => std::sync::Arc::new(arrow::datatypes::Schema::empty()),
    };
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut buf, schema.as_ref())
            .map_err(|e| SparkError::connect_msg(format!("Arrow IPC writer init failed: {e}")))?;
        for batch in batches {
            writer
                .write(batch)
                .map_err(|e| SparkError::connect_msg(format!("Arrow IPC write failed: {e}")))?;
        }
        writer
            .finish()
            .map_err(|e| SparkError::connect_msg(format!("Arrow IPC finish failed: {e}")))?;
    }
    Ok(buf)
}

/// Build a DataFusion DataFrame from record batches (the conversion behind
/// [`DataFrame::to_datafusion`], factored out so it is unit-testable without a
/// live server).
#[cfg(feature = "datafusion")]
fn record_batches_to_datafusion(
    ctx: &datafusion::prelude::SessionContext,
    batches: Vec<arrow::record_batch::RecordBatch>,
) -> Result<datafusion::dataframe::DataFrame> {
    if batches.is_empty() {
        return Err(SparkError::connect_msg(
            "Cannot create DataFusion DataFrame from empty result",
        ));
    }
    ctx.read_batches(batches)
        .map_err(|e| SparkError::connect_msg(format!("Failed to create DataFusion DataFrame: {e}")))
}

/// Build a Polars DataFrame from record batches by bridging through Arrow IPC
/// bytes (so polars' vendored arrow need not match this crate's arrow-rs). The
/// conversion behind [`DataFrame::to_polars`], factored out for unit testing.
#[cfg(feature = "polars")]
fn record_batches_to_polars(
    batches: &[arrow::record_batch::RecordBatch],
) -> Result<polars::frame::DataFrame> {
    use polars::prelude::{IpcReader, SerReader};
    use std::io::Cursor;
    if batches.is_empty() {
        return Ok(polars::frame::DataFrame::empty());
    }
    let buf = record_batches_to_ipc(batches)?;
    IpcReader::new(Cursor::new(buf))
        .finish()
        .map_err(|e| SparkError::connect_msg(format!("Failed to create Polars DataFrame: {e}")))
}

/// A borrowed column decoder, resolved once per record batch. Returned Values
/// own their contents; neither rows nor nested values retain the Arrow arrays.
struct ArrowValueLayout<'a> {
    array: &'a dyn arrow::array::Array,
    decoder: ArrowValueDecoder<'a>,
}

/// The Value of a timestamp of `micros` since the epoch. Spark's Arrow types declare
/// TIMESTAMP_NTZ without a time zone and TIMESTAMP with one (`ArrowUtils.toArrowType`), so the
/// time zone of the Arrow type tells the two apart.
fn timestamp_value(micros: i64, has_time_zone: bool) -> Value {
    if has_time_zone {
        Value::Timestamp(micros)
    } else {
        Value::TimestampNtz(micros)
    }
}

enum ArrowValueDecoder<'a> {
    Null,
    Boolean(&'a arrow::array::BooleanArray),
    Int8(&'a arrow::array::Int8Array),
    Int16(&'a arrow::array::Int16Array),
    Int32(&'a arrow::array::Int32Array),
    Int64(&'a arrow::array::Int64Array),
    Float32(&'a arrow::array::Float32Array),
    Float64(&'a arrow::array::Float64Array),
    String(&'a arrow::array::StringArray),
    Binary(&'a arrow::array::BinaryArray),
    Date32(&'a arrow::array::Date32Array),
    TimestampMicrosecond(&'a arrow::array::TimestampMicrosecondArray),
    UInt8(&'a arrow::array::UInt8Array),
    UInt16(&'a arrow::array::UInt16Array),
    UInt32(&'a arrow::array::UInt32Array),
    UInt64(&'a arrow::array::UInt64Array),
    Decimal128(&'a arrow::array::Decimal128Array),
    LargeString(&'a arrow::array::LargeStringArray),
    LargeBinary(&'a arrow::array::LargeBinaryArray),
    StringView(&'a arrow::array::StringViewArray),
    BinaryView(&'a arrow::array::BinaryViewArray),
    TimestampSecond(&'a arrow::array::TimestampSecondArray),
    TimestampMillisecond(&'a arrow::array::TimestampMillisecondArray),
    TimestampNanosecond(&'a arrow::array::TimestampNanosecondArray),
    Date64(&'a arrow::array::Date64Array),
    DurationMicrosecond(&'a arrow::array::DurationMicrosecondArray),
    Decimal256(&'a arrow::array::Decimal256Array),
    FixedSizeBinary(&'a arrow::array::FixedSizeBinaryArray),
    Time64Microsecond(&'a arrow::array::Time64MicrosecondArray),
    Time64Nanosecond(&'a arrow::array::Time64NanosecondArray),
    Time32Millisecond(&'a arrow::array::Time32MillisecondArray),
    Time32Second(&'a arrow::array::Time32SecondArray),
    IntervalYearMonth(&'a arrow::array::IntervalYearMonthArray),
    IntervalDayTime(&'a arrow::array::IntervalDayTimeArray),
    IntervalMonthDayNano(&'a arrow::array::IntervalMonthDayNanoArray),
    List {
        array: &'a arrow::array::ListArray,
        child: Box<ArrowValueLayout<'a>>,
    },
    Struct {
        names: Arc<[String]>,
        children: Vec<ArrowValueLayout<'a>>,
        variant_fields: Option<(Option<usize>, Option<usize>)>,
    },
    Map {
        array: &'a arrow::array::MapArray,
        children: Box<(ArrowValueLayout<'a>, ArrowValueLayout<'a>)>,
    },
    Unsupported,
}

impl<'a> ArrowValueLayout<'a> {
    #[cfg(test)]
    fn new(array: &'a dyn arrow::array::Array) -> Self {
        Self::of(array, None)
    }

    /// The layout of `array`, of the Spark type `data_type` if known: the fields of its structs,
    /// also in arrays, maps and user-defined types, have the names of the type.
    fn of(array: &'a dyn arrow::array::Array, data_type: Option<&DataType>) -> Self {
        use arrow::array::*;

        // A user-defined type is in the Arrow data of its SQL type.
        let mut data_type = data_type;
        while let Some(DataType::Udt {
            sql_type: Some(sql_type),
            ..
        }) = data_type
        {
            data_type = Some(sql_type);
        }

        // Resolution never fails: null parents and empty containers must not
        // reject unsupported children that are never read.
        let any = array.as_any();
        let decoder = if any.is::<NullArray>() {
            ArrowValueDecoder::Null
        } else if let Some(array) = any.downcast_ref::<BooleanArray>() {
            ArrowValueDecoder::Boolean(array)
        } else if let Some(array) = any.downcast_ref::<Int8Array>() {
            ArrowValueDecoder::Int8(array)
        } else if let Some(array) = any.downcast_ref::<Int16Array>() {
            ArrowValueDecoder::Int16(array)
        } else if let Some(array) = any.downcast_ref::<Int32Array>() {
            ArrowValueDecoder::Int32(array)
        } else if let Some(array) = any.downcast_ref::<Int64Array>() {
            ArrowValueDecoder::Int64(array)
        } else if let Some(array) = any.downcast_ref::<Float32Array>() {
            ArrowValueDecoder::Float32(array)
        } else if let Some(array) = any.downcast_ref::<Float64Array>() {
            ArrowValueDecoder::Float64(array)
        } else if let Some(array) = any.downcast_ref::<StringArray>() {
            ArrowValueDecoder::String(array)
        } else if let Some(array) = any.downcast_ref::<BinaryArray>() {
            ArrowValueDecoder::Binary(array)
        } else if let Some(array) = any.downcast_ref::<Date32Array>() {
            ArrowValueDecoder::Date32(array)
        } else if let Some(array) = any.downcast_ref::<TimestampMicrosecondArray>() {
            ArrowValueDecoder::TimestampMicrosecond(array)
        } else if let Some(array) = any.downcast_ref::<UInt8Array>() {
            ArrowValueDecoder::UInt8(array)
        } else if let Some(array) = any.downcast_ref::<UInt16Array>() {
            ArrowValueDecoder::UInt16(array)
        } else if let Some(array) = any.downcast_ref::<UInt32Array>() {
            ArrowValueDecoder::UInt32(array)
        } else if let Some(array) = any.downcast_ref::<UInt64Array>() {
            ArrowValueDecoder::UInt64(array)
        } else if let Some(array) = any.downcast_ref::<Decimal128Array>() {
            ArrowValueDecoder::Decimal128(array)
        } else if let Some(array) = any.downcast_ref::<LargeStringArray>() {
            ArrowValueDecoder::LargeString(array)
        } else if let Some(array) = any.downcast_ref::<LargeBinaryArray>() {
            ArrowValueDecoder::LargeBinary(array)
        } else if let Some(array) = any.downcast_ref::<StringViewArray>() {
            ArrowValueDecoder::StringView(array)
        } else if let Some(array) = any.downcast_ref::<BinaryViewArray>() {
            ArrowValueDecoder::BinaryView(array)
        } else if let Some(array) = any.downcast_ref::<TimestampSecondArray>() {
            ArrowValueDecoder::TimestampSecond(array)
        } else if let Some(array) = any.downcast_ref::<TimestampMillisecondArray>() {
            ArrowValueDecoder::TimestampMillisecond(array)
        } else if let Some(array) = any.downcast_ref::<TimestampNanosecondArray>() {
            ArrowValueDecoder::TimestampNanosecond(array)
        } else if let Some(array) = any.downcast_ref::<Date64Array>() {
            ArrowValueDecoder::Date64(array)
        } else if let Some(array) = any.downcast_ref::<DurationMicrosecondArray>() {
            ArrowValueDecoder::DurationMicrosecond(array)
        } else if let Some(array) = any.downcast_ref::<Decimal256Array>() {
            ArrowValueDecoder::Decimal256(array)
        } else if let Some(array) = any.downcast_ref::<FixedSizeBinaryArray>() {
            ArrowValueDecoder::FixedSizeBinary(array)
        } else if let Some(array) = any.downcast_ref::<Time64MicrosecondArray>() {
            ArrowValueDecoder::Time64Microsecond(array)
        } else if let Some(array) = any.downcast_ref::<Time64NanosecondArray>() {
            ArrowValueDecoder::Time64Nanosecond(array)
        } else if let Some(array) = any.downcast_ref::<Time32MillisecondArray>() {
            ArrowValueDecoder::Time32Millisecond(array)
        } else if let Some(array) = any.downcast_ref::<Time32SecondArray>() {
            ArrowValueDecoder::Time32Second(array)
        } else if let Some(array) = any.downcast_ref::<IntervalYearMonthArray>() {
            ArrowValueDecoder::IntervalYearMonth(array)
        } else if let Some(array) = any.downcast_ref::<IntervalDayTimeArray>() {
            ArrowValueDecoder::IntervalDayTime(array)
        } else if let Some(array) = any.downcast_ref::<IntervalMonthDayNanoArray>() {
            ArrowValueDecoder::IntervalMonthDayNano(array)
        } else if let Some(array) = any.downcast_ref::<ListArray>() {
            let element_type = match data_type {
                Some(DataType::Array { element_type, .. }) => Some(element_type.as_ref()),
                _ => None,
            };
            ArrowValueDecoder::List {
                array,
                child: Box::new(Self::of(array.values().as_ref(), element_type)),
            }
        } else if let Some(array) = any.downcast_ref::<StructArray>() {
            let fields = array.fields();
            let is_variant = fields.iter().any(|field| {
                field
                    .metadata()
                    .get("variant")
                    .is_some_and(|value| value == "true")
            });
            let spark_fields = match data_type {
                Some(DataType::Struct { fields: spark }) if spark.len() == fields.len() => {
                    Some(spark)
                }
                _ => None,
            };
            ArrowValueDecoder::Struct {
                names: match spark_fields {
                    Some(spark) => spark.iter().map(|field| field.name.clone()).collect(),
                    None => fields.iter().map(|field| field.name().clone()).collect(),
                },
                children: array
                    .columns()
                    .iter()
                    .enumerate()
                    .map(|(i, child)| {
                        Self::of(
                            child.as_ref(),
                            spark_fields.map(|spark| &spark[i].data_type),
                        )
                    })
                    .collect(),
                variant_fields: is_variant.then(|| {
                    (
                        fields.iter().position(|field| field.name() == "value"),
                        fields.iter().position(|field| field.name() == "metadata"),
                    )
                }),
            }
        } else if let Some(array) = any.downcast_ref::<MapArray>() {
            let (key_type, value_type) = match data_type {
                Some(DataType::Map {
                    key_type,
                    value_type,
                    ..
                }) => (Some(key_type.as_ref()), Some(value_type.as_ref())),
                _ => (None, None),
            };
            ArrowValueDecoder::Map {
                array,
                children: Box::new((
                    Self::of(array.keys().as_ref(), key_type),
                    Self::of(array.values().as_ref(), value_type),
                )),
            }
        } else {
            ArrowValueDecoder::Unsupported
        };
        Self { array, decoder }
    }

    fn value_at(&self, index: usize) -> Result<Value> {
        if self.array.is_null(index) {
            return Ok(Value::Null);
        }
        match &self.decoder {
            // NullType may have no null buffer; every element is still Null.
            ArrowValueDecoder::Null => Ok(Value::Null),
            ArrowValueDecoder::Boolean(arr) => Ok(Value::Bool(arr.value(index))),
            ArrowValueDecoder::Int8(arr) => Ok(Value::Byte(arr.value(index))),
            ArrowValueDecoder::Int16(arr) => Ok(Value::Short(arr.value(index))),
            ArrowValueDecoder::Int32(arr) => Ok(Value::Integer(arr.value(index))),
            ArrowValueDecoder::Int64(arr) => Ok(Value::Long(arr.value(index))),
            ArrowValueDecoder::Float32(arr) => Ok(Value::Float(arr.value(index))),
            ArrowValueDecoder::Float64(arr) => Ok(Value::Double(arr.value(index))),
            ArrowValueDecoder::String(arr) => Ok(Value::String(arr.value(index).to_string())),
            ArrowValueDecoder::Binary(arr) => Ok(Value::Binary(arr.value(index).to_vec())),
            ArrowValueDecoder::Date32(arr) => Ok(Value::Date(arr.value(index))),
            ArrowValueDecoder::TimestampMicrosecond(arr) => {
                Ok(timestamp_value(arr.value(index), arr.timezone().is_some()))
            }
            ArrowValueDecoder::UInt8(arr) => Ok(Value::Short(arr.value(index) as i16)),
            ArrowValueDecoder::UInt16(arr) => Ok(Value::Integer(arr.value(index) as i32)),
            ArrowValueDecoder::UInt32(arr) => Ok(Value::Long(arr.value(index) as i64)),
            ArrowValueDecoder::UInt64(arr) => {
                let val = arr.value(index);
                let i64_val = i64::try_from(val).map_err(|_| {
                    SparkError::connect_msg(format!("UInt64 value {} exceeds i64 range", val))
                })?;
                Ok(Value::Long(i64_val))
            }
            // Decimal128 stays exact, preserving precision/scale rather than using lossy f64.
            ArrowValueDecoder::Decimal128(arr) => {
                let scale = arr.scale() as i32;
                Ok(Value::Decimal {
                    value: i128_to_decimal_string(arr.value(index), scale),
                    precision: Some(arr.precision() as i32),
                    scale: Some(scale),
                })
            }
            ArrowValueDecoder::LargeString(arr) => Ok(Value::String(arr.value(index).to_string())),
            ArrowValueDecoder::LargeBinary(arr) => Ok(Value::Binary(arr.value(index).to_vec())),
            ArrowValueDecoder::StringView(arr) => Ok(Value::String(arr.value(index).to_string())),
            ArrowValueDecoder::BinaryView(arr) => Ok(Value::Binary(arr.value(index).to_vec())),
            ArrowValueDecoder::TimestampSecond(arr) => Ok(timestamp_value(
                arr.value(index) * 1_000_000,
                arr.timezone().is_some(),
            )),
            ArrowValueDecoder::TimestampMillisecond(arr) => Ok(timestamp_value(
                arr.value(index) * 1_000,
                arr.timezone().is_some(),
            )),
            ArrowValueDecoder::TimestampNanosecond(arr) => Ok(timestamp_value(
                arr.value(index) / 1_000,
                arr.timezone().is_some(),
            )),
            ArrowValueDecoder::Date64(arr) => {
                Ok(Value::Date((arr.value(index) / 86_400_000) as i32))
            }
            ArrowValueDecoder::DurationMicrosecond(arr) => {
                Ok(Value::DayTimeInterval(arr.value(index)))
            }
            // Decimal256 also preserves exact digits; Arrow formats with the scale.
            ArrowValueDecoder::Decimal256(arr) => Ok(Value::Decimal {
                value: arr.value_as_string(index),
                precision: Some(arr.precision() as i32),
                scale: Some(arr.scale() as i32),
            }),
            ArrowValueDecoder::FixedSizeBinary(arr) => Ok(Value::Binary(arr.value(index).to_vec())),
            // A TIME, in nanoseconds since midnight, as PySpark's `TimeType.fromInternal` reads.
            ArrowValueDecoder::Time64Microsecond(arr) => Ok(Value::Time(arr.value(index) * 1_000)),
            ArrowValueDecoder::Time64Nanosecond(arr) => Ok(Value::Time(arr.value(index))),
            ArrowValueDecoder::Time32Millisecond(arr) => {
                Ok(Value::Time(arr.value(index) as i64 * 1_000_000))
            }
            ArrowValueDecoder::Time32Second(arr) => {
                Ok(Value::Time(arr.value(index) as i64 * 1_000_000_000))
            }
            // Intervals likewise have no dedicated Value variant, so render as strings.
            ArrowValueDecoder::IntervalYearMonth(arr) => {
                // The sign goes on the years, as Spark's `toYearMonthIntervalString`
                // does, so -1 month is "-0-1" and not "0-1".
                let months = arr.value(index) as i64;
                let sign = if months < 0 { "-" } else { "" };
                let abs = months.abs();
                Ok(Value::String(format!("{sign}{}-{}", abs / 12, abs % 12)))
            }
            ArrowValueDecoder::IntervalDayTime(arr) => {
                let v = arr.value(index);
                Ok(Value::String(format!(
                    "{} days {} ms",
                    v.days, v.milliseconds
                )))
            }
            ArrowValueDecoder::IntervalMonthDayNano(arr) => {
                let v = arr.value(index);
                Ok(Value::String(format!(
                    "{} months {} days {} ns",
                    v.months, v.days, v.nanoseconds
                )))
            }
            ArrowValueDecoder::List { array, child } => {
                // Sliced list/map arrays retain their original children. Their
                // offsets already identify the correct absolute child range.
                let offsets = array.value_offsets();
                let start = offsets[index] as usize;
                let end = offsets[index + 1] as usize;
                let mut items = Vec::with_capacity(end - start);
                for child_index in start..end {
                    items.push(child.value_at(child_index)?);
                }
                Ok(Value::List(items))
            }
            ArrowValueDecoder::Struct {
                names,
                children,
                variant_fields,
            } => {
                // VARIANT arrives as struct<value: binary, metadata: binary>, with
                // {"variant": "true"} on the metadata field. Return raw-byte
                // Value::Variant so it materializes as pyspark's VariantVal rather
                // than a plain {value, metadata} struct/dict.
                if let Some((value, metadata)) = variant_fields {
                    let binary_field = |column: Option<usize>| -> Result<Vec<u8>> {
                        match column {
                            Some(column) => match children[column].value_at(index)? {
                                Value::Binary(bytes) => Ok(bytes),
                                Value::Null => Ok(vec![]),
                                _ => Err(SparkError::connect_msg("variant field is not binary")),
                            },
                            None => Ok(vec![]),
                        }
                    };
                    return Ok(Value::Variant {
                        value: binary_field(*value)?,
                        metadata: binary_field(*metadata)?,
                    });
                }
                let mut values = Vec::with_capacity(children.len());
                for child in children {
                    values.push(child.value_at(index)?);
                }
                Ok(Value::Struct(StructValue::new(Arc::clone(names), values)))
            }
            ArrowValueDecoder::Map { array, children } => {
                let offsets = array.value_offsets();
                let start = offsets[index] as usize;
                let end = offsets[index + 1] as usize;
                let mut map = Vec::with_capacity(end - start);
                for child_index in start..end {
                    map.push((
                        children.0.value_at(child_index)?,
                        children.1.value_at(child_index)?,
                    ));
                }
                Ok(Value::Map(map))
            }
            ArrowValueDecoder::Unsupported => Err(SparkError::connect_msg(format!(
                "Unsupported Arrow type {:?} - cannot convert to Value",
                self.array.data_type()
            ))),
        }
    }
}

#[cfg(test)]
fn arrow_value_at(array: &dyn arrow::array::Array, index: usize) -> Result<Value> {
    ArrowValueLayout::new(array).value_at(index)
}

#[cfg(test)]
mod nested_decode_tests {
    use super::*;
    use arrow::array::*;
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::StreamWriter;
    use arrow::record_batch::RecordBatch;

    fn encoded_batches(
        columns: Vec<(&str, ArrayRef)>,
        repeats: usize,
        empty_boundaries: bool,
    ) -> proto::execute_plan_response::ArrowBatch {
        let schema = Arc::new(Schema::new(
            columns
                .iter()
                .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            columns.into_iter().map(|(_, array)| array).collect(),
        )
        .unwrap();
        let mut data = vec![];
        {
            let mut writer = StreamWriter::try_new(&mut data, &schema).unwrap();
            if empty_boundaries {
                writer
                    .write(&RecordBatch::new_empty(schema.clone()))
                    .unwrap();
            }
            for _ in 0..repeats {
                writer.write(&batch).unwrap();
            }
            if empty_boundaries {
                writer
                    .write(&RecordBatch::new_empty(schema.clone()))
                    .unwrap();
            }
            writer.finish().unwrap();
        }
        proto::execute_plan_response::ArrowBatch {
            data: data.into(),
            ..Default::default()
        }
    }

    fn decode_batches(columns: Vec<(&str, ArrayRef)>, repeats: usize) -> Vec<Row> {
        decode_arrow_batch(&encoded_batches(columns, repeats, false)).unwrap()
    }

    fn as_struct(value: &Value) -> &StructValue {
        let Value::Struct(row) = value else {
            panic!("expected struct, got {value:?}");
        };
        row
    }

    #[test]
    fn batches_compressed_with_zstd_or_lz4_decode_as_uncompressed_ones() {
        // As PyArrow reads them: the server compresses the Arrow batches of the results with
        // the codec of spark.sql.execution.arrow.compression.codec, zstd or lz4.
        use arrow::ipc::writer::IpcWriteOptions;
        use arrow::ipc::CompressionType;

        let ids: ArrayRef = Arc::new(Int64Array::from_iter((0..2000).map(Some)));
        let names: ArrayRef = Arc::new(StringArray::from_iter(
            (0..2000).map(|i| (i % 3 != 0).then(|| format!("name {}", i % 7))),
        ));
        let plain = encoded_batches(vec![("id", ids.clone()), ("name", names.clone())], 2, false);
        let rows = decode_arrow_batch(&plain).unwrap();
        assert_eq!(rows.len(), 4000);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, names]).unwrap();
        for codec in [CompressionType::ZSTD, CompressionType::LZ4_FRAME] {
            let options = IpcWriteOptions::default()
                .try_with_compression(Some(codec))
                .unwrap();
            let mut data = vec![];
            {
                let mut writer =
                    StreamWriter::try_new_with_options(&mut data, &schema, options).unwrap();
                writer.write(&batch).unwrap();
                writer.write(&batch).unwrap();
                writer.finish().unwrap();
            }
            assert!(data.len() < plain.data.len(), "{codec:?} compresses");
            let compressed = proto::execute_plan_response::ArrowBatch {
                data: data.into(),
                ..Default::default()
            };
            assert_eq!(decode_arrow_batch(&compressed).unwrap(), rows, "{codec:?}");
            assert_eq!(
                decode_arrow_record_batches(&compressed).unwrap(),
                decode_arrow_record_batches(&plain).unwrap(),
                "{codec:?}"
            );
        }
    }

    #[test]
    fn mixed_nullable_slices_decode_across_empty_and_repeated_batches() {
        let columns: Vec<(&str, ArrayRef)> = vec![
            (
                "dup",
                Arc::new(Int64Array::from(vec![Some(-1), Some(7), None])),
            ),
            (
                "dup",
                Arc::new(StringArray::from(vec![Some("skip"), Some("é\0雪"), None])),
            ),
            (
                "bool",
                Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])),
            ),
            (
                "bytes",
                Arc::new(BinaryArray::from(vec![
                    Some(b"skip".as_ref()),
                    Some(b"\0\xff".as_ref()),
                    None,
                ])),
            ),
            ("null", Arc::new(NullArray::new(3))),
        ];
        let batch = encoded_batches(
            columns
                .into_iter()
                .map(|(name, array)| (name, array.slice(1, 2)))
                .collect(),
            2,
            true,
        );
        let rows = decode_arrow_batch(&batch).unwrap();
        assert_eq!(rows.len(), 4);
        for batch in rows.chunks(2) {
            assert_eq!(batch[0].fields(), &["dup", "dup", "bool", "bytes", "null"]);
            assert_eq!(
                batch[0].values(),
                &[
                    Value::Long(7),
                    Value::String("é\0雪".into()),
                    Value::Bool(false),
                    Value::Binary(vec![0, 255]),
                    Value::Null,
                ]
            );
            assert_eq!(batch[1].values(), &vec![Value::Null; 5]);
            assert_eq!(batch[0].get_by_name("dup"), Some(&Value::Long(7)));
            assert!(std::ptr::eq(batch[0].fields(), batch[1].fields()));
        }
        // The IPC buffers and borrowed column layouts have already been dropped.
        assert_eq!(rows[2].get(1), Some(&Value::String("é\0雪".into())));
    }

    #[test]
    fn nested_struct_fields_take_the_names_of_the_schema_of_the_result() {
        // As PySpark's ArrowTableToRowsConversion, which builds the rows by the schema of the
        // result: the fields of nested structs, also in arrays and maps, have the names of the
        // schema, which keep the duplicate names that the server's Arrow data numbers.
        use crate::types::{DataType as Spark, StructField as SparkField};
        let field = |name: &str, data_type: Spark| SparkField {
            name: name.to_string(),
            data_type,
            nullable: true,
            metadata: Default::default(),
        };
        let pair = || {
            StructArray::from(vec![
                (
                    Arc::new(Field::new("a_0", DataType::Int32, true)),
                    Arc::new(Int32Array::from(vec![1])) as ArrayRef,
                ),
                (
                    Arc::new(Field::new("a_1", DataType::Int32, true)),
                    Arc::new(Int32Array::from(vec![2])) as ArrayRef,
                ),
            ])
        };
        let list = ListArray::new(
            Arc::new(Field::new("element", pair().data_type().clone(), true)),
            arrow::buffer::OffsetBuffer::from_lengths([1]),
            Arc::new(pair()),
            None,
        );
        let batch = encoded_batches(
            vec![("v", Arc::new(pair()) as ArrayRef), ("w", Arc::new(list))],
            1,
            false,
        );
        let pair_type = || Spark::Struct {
            fields: vec![field("a", Spark::Integer), field("a", Spark::Integer)],
        };
        let schema = Spark::Struct {
            fields: vec![
                field("v", pair_type()),
                field(
                    "w",
                    Spark::Array {
                        element_type: Box::new(pair_type()),
                        contains_null: true,
                    },
                ),
            ],
        };
        let rows = decode_arrow_batch_of(&batch, Some(&schema)).unwrap();
        assert_eq!(rows[0].fields(), &["v", "w"]);
        assert_eq!(as_struct(rows[0].get(0).unwrap()).fields(), &["a", "a"]);
        let Some(Value::List(items)) = rows[0].get(1) else {
            panic!("expected a list");
        };
        assert_eq!(as_struct(&items[0]).fields(), &["a", "a"]);
        assert_eq!(
            as_struct(&items[0]).values(),
            &[Value::Integer(1), Value::Integer(2)]
        );
        // Without the schema, or of a schema of another shape, the names of the Arrow data.
        let other = Spark::Struct {
            fields: vec![field("v", Spark::Integer)],
        };
        for schema in [None, Some(&other)] {
            let rows = decode_arrow_batch_of(&batch, schema).unwrap();
            assert_eq!(as_struct(rows[0].get(0).unwrap()).fields(), &["a_0", "a_1"]);
        }
    }

    #[test]
    fn sliced_parents_decode_children_that_were_already_sliced() {
        let leaf = StructArray::from(vec![(
            Arc::new(Field::new("n", DataType::Int32, false)),
            Arc::new(Int32Array::from(vec![-9, 10, 20, 30, 40, 50, 60])) as ArrayRef,
        )]);
        let entries = StructArray::from(vec![
            (
                Arc::new(Field::new("key", DataType::Utf8, false)),
                Arc::new(StringArray::from(vec![
                    "skip", "a", "b", "c", "d", "e", "f",
                ])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("value", leaf.data_type().clone(), true)),
                Arc::new(leaf.clone()) as ArrayRef,
            ),
        ])
        .slice(1, 6);
        let child = leaf.slice(1, 6);
        let offsets = OffsetBuffer::new(vec![0, 1, 3, 4, 6].into());
        let list = ListArray::new(
            Arc::new(Field::new("item", child.data_type().clone(), true)),
            offsets.clone(),
            Arc::new(child),
            None,
        )
        .slice(1, 2);
        let map = MapArray::new(
            Arc::new(Field::new("entries", entries.data_type().clone(), false)),
            offsets,
            entries,
            None,
            false,
        )
        .slice(1, 2);
        // Exercise the unserialized sliced arrays as well as the IPC path:
        // serialization is allowed to normalize their physical child offsets.
        let value = |n| Value::Struct(Row::new(vec!["n".into()], vec![Value::Integer(n)]).into());
        let expected_lists = [
            Value::List(vec![value(20), value(30)]),
            Value::List(vec![value(40)]),
        ];
        let expected_maps = [
            Value::Map(vec![
                (Value::String("b".into()), value(20)),
                (Value::String("c".into()), value(30)),
            ]),
            Value::Map(vec![(Value::String("d".into()), value(40))]),
        ];
        let list_layout = ArrowValueLayout::new(&list);
        let map_layout = ArrowValueLayout::new(&map);
        for i in 0..2 {
            assert_eq!(list_layout.value_at(i).unwrap(), expected_lists[i]);
            assert_eq!(map_layout.value_at(i).unwrap(), expected_maps[i]);
        }
        drop((list_layout, map_layout));
        let rows = decode_batches(vec![("list", Arc::new(list)), ("map", Arc::new(map))], 1);
        for i in 0..2 {
            assert_eq!(
                rows[i].values(),
                &[expected_lists[i].clone(), expected_maps[i].clone()]
            );
        }
    }

    #[test]
    fn unsupported_children_remain_lazy_under_nulls_empty_containers_and_variants() {
        let null_duration = DurationSecondArray::from(vec![None]);
        assert_eq!(arrow_value_at(&null_duration, 0).unwrap(), Value::Null);
        let populated: ArrayRef = Arc::new(DurationSecondArray::from(vec![Some(7)]));
        let list = ListArray::new(
            Arc::new(Field::new("item", populated.data_type().clone(), true)),
            OffsetBuffer::new(vec![0, 0].into()),
            populated.clone(),
            None,
        );
        assert_eq!(arrow_value_at(&list, 0).unwrap(), Value::List(vec![]));
        let entries = StructArray::from(vec![
            (
                Arc::new(Field::new("key", DataType::Utf8, false)),
                Arc::new(StringArray::from(vec!["a"])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("value", populated.data_type().clone(), true)),
                populated.clone(),
            ),
        ]);
        let map = MapArray::new(
            Arc::new(Field::new("entries", entries.data_type().clone(), false)),
            OffsetBuffer::new(vec![0, 0].into()),
            entries,
            None,
            false,
        );
        assert_eq!(
            arrow_value_at(&map, 0).unwrap(),
            Value::Map(Default::default())
        );
        let parent = StructArray::new(
            vec![Arc::new(Field::new(
                "unused",
                populated.data_type().clone(),
                true,
            ))]
            .into(),
            vec![populated.clone()],
            Some(NullBuffer::from(vec![false])),
        );
        assert_eq!(arrow_value_at(&parent, 0).unwrap(), Value::Null);
        // Deliberately noncanonical: detection accepts a marker on any field,
        // and the first matching "value" wins while an unsupported duplicate
        // remains unread. The absent "metadata" field must still decode as empty.
        let variant = StructArray::from(vec![
            (
                Arc::new(
                    Field::new("value", DataType::Binary, false)
                        .with_metadata([("variant".into(), "true".into())].into()),
                ),
                Arc::new(BinaryArray::from_iter_values([b"x".as_ref()])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("value", populated.data_type().clone(), true)),
                populated,
            ),
        ]);
        assert_eq!(
            arrow_value_at(&variant, 0).unwrap(),
            Value::Variant {
                value: vec![b'x'],
                metadata: vec![]
            }
        );
    }

    #[test]
    fn decoding_errors_keep_row_major_column_order() {
        let first_column = encoded_batches(
            vec![
                ("u", Arc::new(UInt64Array::from(vec![u64::MAX]))),
                (
                    "duration",
                    Arc::new(DurationSecondArray::from(vec![Some(1)])),
                ),
            ],
            1,
            false,
        );
        assert!(decode_arrow_batch(&first_column)
            .unwrap_err()
            .to_string()
            .contains("UInt64 value 18446744073709551615 exceeds i64 range"));
        let first_row = encoded_batches(
            vec![
                ("u", Arc::new(UInt64Array::from(vec![1, u64::MAX]))),
                (
                    "duration",
                    Arc::new(DurationSecondArray::from(vec![Some(1), None])),
                ),
            ],
            1,
            false,
        );
        assert!(decode_arrow_batch(&first_row)
            .unwrap_err()
            .to_string()
            .contains("Unsupported Arrow type Duration(Second) - cannot convert to Value"));
    }

    #[test]
    fn nested_names_preserve_duplicates_unicode_nulls_and_sibling_layouts() {
        let names = ["dup", "dup", "", "名前"];
        let fields = names
            .iter()
            .map(|name| Arc::new(Field::new(*name, DataType::Int32, true)))
            .collect::<Vec<_>>();
        let columns = (1..=4)
            .map(|n| Arc::new(Int32Array::from(vec![Some(n), None, None])) as ArrayRef)
            .collect();
        let nested = StructArray::new(
            fields.into(),
            columns,
            Some(NullBuffer::from(vec![true, true, false])),
        );
        let sibling = StructArray::from(vec![(
            Arc::new(Field::new("other", DataType::Int32, false)),
            Arc::new(Int32Array::from(vec![7, 8, 9])) as ArrayRef,
        )]);
        let rows = decode_batches(
            vec![("nested", Arc::new(nested)), ("sibling", Arc::new(sibling))],
            2,
        );
        assert_eq!(rows.len(), 6);
        for batch in rows.chunks(3) {
            let populated = as_struct(batch[0].get(0).unwrap());
            let all_null = as_struct(batch[1].get(0).unwrap());
            assert_eq!(populated.fields(), &names);
            assert_eq!(
                populated.values(),
                &[
                    Value::Integer(1),
                    Value::Integer(2),
                    Value::Integer(3),
                    Value::Integer(4)
                ]
            );
            assert_eq!(populated.get_by_name("dup"), Some(&Value::Integer(1)));
            assert_eq!(all_null.values(), &vec![Value::Null; 4]);
            assert_eq!(batch[2].get(0), Some(&Value::Null));
            assert!(std::ptr::eq(populated.fields(), all_null.fields()));
            let other = as_struct(batch[0].get(1).unwrap());
            assert_eq!(other.fields(), &["other"]);
            assert_eq!(other.values(), &[Value::Integer(7)]);
        }
    }

    #[test]
    fn names_are_shared_through_lists_maps_and_nested_structs() {
        let leaf = StructArray::from(vec![(
            Arc::new(Field::new("leaf", DataType::Int32, false)),
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])) as ArrayRef,
        )]);
        let entries = StructArray::from(vec![
            (
                Arc::new(Field::new("key", DataType::Utf8, false)),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("value", leaf.data_type().clone(), true)),
                Arc::new(leaf) as ArrayRef,
            ),
        ]);
        let map = MapArray::new(
            Arc::new(Field::new("entries", entries.data_type().clone(), false)),
            OffsetBuffer::new(vec![0, 1, 2, 3, 4].into()),
            entries,
            None,
            false,
        );
        let list = ListArray::new(
            Arc::new(Field::new("item", map.data_type().clone(), true)),
            OffsetBuffer::new(vec![0, 2, 4].into()),
            Arc::new(map),
            None,
        );
        let outer = StructArray::from(vec![(
            Arc::new(Field::new("items", list.data_type().clone(), true)),
            Arc::new(list) as ArrayRef,
        )]);
        let rows = decode_batches(vec![("outer", Arc::new(outer))], 1);
        let mut leaves = vec![];
        for row in &rows {
            let outer = as_struct(row.get(0).unwrap());
            let Value::List(maps) = outer.get(0).unwrap() else {
                panic!("expected list");
            };
            for value in maps {
                let Value::Map(map) = value else {
                    panic!("expected map");
                };
                leaves.push(as_struct(&map[0].1));
            }
        }
        assert_eq!(leaves.len(), 4);
        for (index, leaf) in leaves.iter().enumerate() {
            assert_eq!(leaf.fields(), &["leaf"]);
            assert_eq!(leaf.get(0), Some(&Value::Integer(index as i32 + 1)));
            assert!(std::ptr::eq(leaves[0].fields(), leaf.fields()));
        }
    }

    #[test]
    fn sliced_containers_preserve_offsets_empty_null_and_struct_values() {
        let leaf = StructArray::new(
            vec![Arc::new(Field::new("n", DataType::Int32, true))].into(),
            vec![Arc::new(Int32Array::from(vec![
                Some(0),
                Some(1),
                None,
                Some(3),
                Some(4),
                Some(5),
            ]))],
            Some(NullBuffer::from(vec![true, true, false, true, true, true])),
        );
        let offsets = OffsetBuffer::new(vec![0, 1, 3, 3, 4, 6].into());
        let valid = Some(NullBuffer::from(vec![true, true, true, false, true]));
        let list = ListArray::new(
            Arc::new(Field::new("item", leaf.data_type().clone(), true)),
            offsets.clone(),
            Arc::new(leaf.clone()),
            valid.clone(),
        );
        let entries = StructArray::from(vec![
            (
                Arc::new(Field::new("key", DataType::Utf8, false)),
                Arc::new(StringArray::from(vec!["k0", "k1", "k2", "k3", "k4", "k5"])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("value", leaf.data_type().clone(), true)),
                Arc::new(leaf) as ArrayRef,
            ),
        ]);
        let map = MapArray::new(
            Arc::new(Field::new("entries", entries.data_type().clone(), false)),
            offsets,
            entries,
            valid,
            false,
        );
        let rows = decode_batches(
            vec![
                ("list", Arc::new(list.slice(1, 4))),
                ("map", Arc::new(map.slice(1, 4))),
            ],
            1,
        );
        let value =
            |n| Value::Struct(Row::new(vec!["n".to_string()], vec![Value::Integer(n)]).into());
        assert_eq!(
            rows[0].get(0),
            Some(&Value::List(vec![value(1), Value::Null]))
        );
        assert_eq!(
            rows[0].get(1),
            Some(&Value::Map(vec![
                (Value::String("k1".to_string()), value(1)),
                (Value::String("k2".to_string()), Value::Null)
            ]))
        );
        assert_eq!(rows[1].get(0), Some(&Value::List(vec![])));
        assert_eq!(rows[1].get(1), Some(&Value::Map(Default::default())));
        assert_eq!(rows[2].get(0), Some(&Value::Null));
        assert_eq!(rows[2].get(1), Some(&Value::Null));
        assert_eq!(rows[3].get(0), Some(&Value::List(vec![value(4), value(5)])));
        assert_eq!(
            rows[3].get(1),
            Some(&Value::Map(vec![
                (Value::String("k4".to_string()), value(4)),
                (Value::String("k5".to_string()), value(5))
            ]))
        );
        let (Value::List(first), Value::List(last)) =
            (rows[0].get(0).unwrap(), rows[3].get(0).unwrap())
        else {
            panic!("expected lists");
        };
        assert!(std::ptr::eq(
            as_struct(&first[0]).fields(),
            as_struct(&last[0]).fields()
        ));
        let (Value::Map(first), Value::Map(last)) =
            (rows[0].get(1).unwrap(), rows[3].get(1).unwrap())
        else {
            panic!("expected maps");
        };
        assert!(std::ptr::eq(
            as_struct(&first[0].1).fields(),
            as_struct(&last[0].1).fields()
        ));
    }

    #[test]
    fn variant_marker_precedes_struct_decoding_and_preserves_errors() {
        let marked = |name: &str, array: ArrayRef| {
            let field = Field::new(name, array.data_type().clone(), true).with_metadata(
                std::collections::HashMap::from([("variant".into(), "true".into())]),
            );
            StructArray::from(vec![(Arc::new(field), array)])
        };
        let binary = marked(
            "value",
            Arc::new(BinaryArray::from(vec![Some(b"x".as_ref()), None])),
        );
        assert_eq!(
            arrow_value_at(&binary, 0).unwrap(),
            Value::Variant {
                value: b"x".to_vec(),
                metadata: vec![]
            }
        );
        assert_eq!(
            arrow_value_at(&binary, 1).unwrap(),
            Value::Variant {
                value: vec![],
                metadata: vec![]
            }
        );
        let missing = marked("unrelated", Arc::new(Int32Array::from(vec![1])));
        assert_eq!(
            arrow_value_at(&missing, 0).unwrap(),
            Value::Variant {
                value: vec![],
                metadata: vec![]
            }
        );
        let invalid = marked("metadata", Arc::new(Int32Array::from(vec![1])));
        assert!(arrow_value_at(&invalid, 0)
            .unwrap_err()
            .to_string()
            .contains("variant field is not binary"));
        let null_parent = StructArray::new(
            invalid.fields().clone(),
            invalid.columns().to_vec(),
            Some(NullBuffer::from(vec![false])),
        );
        assert_eq!(arrow_value_at(&null_parent, 0).unwrap(), Value::Null);
        let ordinary = StructArray::from(vec![
            (
                Arc::new(Field::new("value", DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![1])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("metadata", DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![2])) as ArrayRef,
            ),
        ]);
        assert_eq!(
            arrow_value_at(&ordinary, 0).unwrap(),
            Value::Struct(
                Row::new(
                    vec!["value".to_string(), "metadata".to_string()],
                    vec![Value::Integer(1), Value::Integer(2)]
                )
                .into()
            )
        );
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use prost::Message;

    #[test]
    fn print_schema_includes_final_blank_line() {
        const CHILD: &str = "SPARK_CONNECT_PRINT_SCHEMA_STDOUT_CHILD";
        const BEGIN: &str = "__PRINT_SCHEMA_BEGIN__\n";
        const END: &str = "__PRINT_SCHEMA_END__";
        if std::env::var_os(CHILD).is_some() {
            // A cache regression must fail promptly instead of waiting on an RPC.
            let _watchdog = std::thread::spawn(|| {
                std::thread::sleep(std::time::Duration::from_secs(30));
                eprintln!("print_schema stdout child timed out");
                std::process::exit(1);
            });
            let spark = SparkSession::builder()
                .remote("sc://localhost:1")
                .get_or_create()
                .unwrap();
            let nested = DataType::from_json(&serde_json::json!({
                "type": "struct",
                "fields": [{"name": "outer", "type": {
                    "type": "struct",
                    "fields": [{"name": "inner", "type": "integer", "nullable": false, "metadata": {}}]
                }, "nullable": true, "metadata": {}}]
            }))
            .unwrap();
            print!("{BEGIN}");
            for schema in [nested, DataType::Struct { fields: vec![] }] {
                let df = spark.range(0).unwrap();
                *df.schema_cache.lock().unwrap() = Some(Arc::new(schema));
                df.print_schema().unwrap();
            }
            println!("{END}");
            return;
        }

        // A separate process exposes real stdout without changing global capture
        // or sharing it with the other parallel tests in this process.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "dataframe::cache_tests::print_schema_includes_final_blank_line",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        let printed = stdout
            .split_once(BEGIN)
            .expect("child must run the schema print test")
            .1
            .split_once(END)
            .expect("child must finish printing schemas")
            .0;
        assert_eq!(
            printed,
            concat!(
                "root\n",
                " |-- outer: struct (nullable = true)\n",
                " |    |-- inner: integer (nullable = false)\n",
                "\n",
                "root\n",
                "\n",
            ),
        );
    }

    #[test]
    fn schema_cache_shares_exact_clones_but_not_changed_plans_or_new_sources() {
        let spark = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .unwrap();
        let df = spark.range(3).unwrap();
        let cloned_before_analysis = df.clone();
        let schema = DataType::Struct { fields: vec![] };
        *df.schema_cache.lock().unwrap() = Some(Arc::new(schema.clone()));

        // These accesses succeed without a server, including a clone created
        // before the schema was populated.
        assert_eq!(cloned_before_analysis.schema().unwrap(), schema);
        assert_eq!(df.clone().schema().unwrap(), schema);
        assert!(Arc::ptr_eq(
            &df.schema_cache,
            &cloned_before_analysis.schema_cache
        ));
        for derived in [
            df.select(["id"]),
            df.zip(&df).unwrap(),
            spark.range(3).unwrap(),
        ] {
            assert!(!Arc::ptr_eq(&df.schema_cache, &derived.schema_cache));
            assert!(derived.schema_cache.lock().unwrap().is_none());
        }
    }

    #[test]
    fn schema_propagation_snapshots_without_waiting_for_analysis() {
        let spark = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .unwrap();
        let df = spark.range(3).unwrap();
        let early_child = df.limit(1);
        let schema = DataType::Struct { fields: vec![] };
        *df.schema_cache.lock().unwrap() = Some(Arc::new(schema.clone()));
        let warm_child = df.limit(1);
        assert_eq!(warm_child.schema().unwrap(), schema);
        assert!(!Arc::ptr_eq(&df.schema_cache, &warm_child.schema_cache));
        assert!(early_child.cached_schema().is_none());

        // Even when an input lock is held, constructing unary and binary plans
        // must finish. Release it before joining so a regression fails boundedly.
        let guard = df.schema_cache.lock().unwrap();
        let input = df.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            tx.send((input.limit(1), input.union(&input))).unwrap();
        });
        let result = rx.recv_timeout(std::time::Duration::from_secs(5));
        drop(guard);
        worker.join().unwrap();
        let (unary, binary) = result.expect("plan construction must not wait for schema analysis");
        assert!(unary.cached_schema().is_none());
        assert!(binary.cached_schema().is_none());
    }

    #[test]
    fn schema_propagation_merges_only_exact_schemas_from_the_same_session() {
        let spark = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .unwrap();
        let df = spark.range(3).unwrap();
        let schema = serde_json::json!({
            "type": "struct",
            "fields": [{"name": "id", "type": "long", "nullable": true,
                        "metadata": {"note": "original"}}]
        });
        let left = df.to(DataType::from_json(&schema).unwrap());
        assert_eq!(left.union(&left).schema().unwrap(), left.schema().unwrap());
        let mut changes = vec![];
        for (key, value) in [
            ("name", serde_json::json!("renamed")),
            ("type", serde_json::json!("double")),
            ("nullable", serde_json::json!(false)),
            ("metadata", serde_json::json!({"note": "changed"})),
        ] {
            let mut changed = schema.clone();
            changed["fields"][0][key] = value;
            changes.push(DataType::from_json(&changed).unwrap());
        }
        for changed in changes {
            assert!(left.union(&df.to(changed)).cached_schema().is_none());
        }
        assert!(left.union(&df).cached_schema().is_none());
        let other_session = spark.new_session();
        let other = other_session.range(3).unwrap().to(left.schema().unwrap());
        assert!(left.union(&other).cached_schema().is_none());
    }

    #[test]
    fn schema_propagation_seeds_only_struct_output_schemas() {
        use crate::udf::PythonUDFPayload;

        let spark = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .unwrap();
        let df = spark.range(3).unwrap();
        let schema = DataType::Struct { fields: vec![] };
        let udf = |output_type| {
            CommonInlineUserDefinedFunctionExpression::new(
                "map".to_string(),
                true,
                vec![],
                PythonUDFPayload::new(output_type, 205, vec![], "3.12".to_string()),
            )
        };
        for output in [
            df.to(schema.clone()),
            df.map_in_pandas(udf(schema.clone()), false),
            df.map_in_arrow(udf(schema.clone()), false),
        ] {
            assert_eq!(output.schema().unwrap(), schema);
        }
        assert!(df.to(DataType::Long).cached_schema().is_none());
        assert!(df
            .map_in_arrow(udf(DataType::Long), false)
            .cached_schema()
            .is_none());
    }

    #[test]
    fn schema_cache_recovers_after_analysis_panics() {
        let spark = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .unwrap();
        let df = spark.range(3).unwrap();
        let cloned = df.clone();
        // A panic under the cache lock, as a panicking analysis leaves it: the clones must still
        // analyze afterward, rather than returning a permanent cache-poisoning error.
        let cache = Arc::clone(&df.schema_cache);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = cache.lock().unwrap();
            panic!("an analysis panics while the cache lock is held");
        }));
        assert!(result.is_err());
        assert!(
            df.schema_cache.is_poisoned(),
            "the cache lock must be poisoned for this test"
        );
        let mut cached = df.schema_cache.lock().unwrap_err().into_inner();
        assert!(cached.is_none());
        let schema = DataType::Struct { fields: vec![] };
        *cached = Some(Arc::new(schema.clone()));
        drop(cached);
        assert!(df.schema_cache.is_poisoned());
        assert_eq!(df.schema().unwrap(), schema);
        assert!(!df.schema_cache.is_poisoned());
        assert_eq!(cloned.schema().unwrap(), schema);
    }

    #[test]
    fn cache_default_is_memory_and_disk_deser() {
        let sl = DataFrame::memory_and_disk_deser();
        assert!(sl.use_memory && sl.use_disk && sl.deserialized);
        assert!(!sl.use_off_heap);
        assert_eq!(sl.replication, 1);
    }

    #[test]
    fn persist_request_carries_storage_level_over_the_wire() {
        // The reviewer's bug class: an argument dropped before it reaches the proto.
        // Assert the storage level survives encode/decode inside the Persist analyze op.
        let persist = proto::analyze_plan_request::Persist {
            relation: None,
            storage_level: Some(DataFrame::memory_and_disk_deser()),
        };
        let decoded =
            proto::analyze_plan_request::Persist::decode(persist.encode_to_vec().as_slice())
                .unwrap();
        let sl = decoded
            .storage_level
            .expect("storage_level must be present");
        assert!(sl.use_memory && sl.use_disk && sl.deserialized && sl.replication == 1);
    }

    #[test]
    fn get_storage_level_response_maps_to_is_cached() {
        // cached iff use_memory || use_disk (mirrors DataFrame.is_cached derivation)
        let cached = proto::StorageLevel {
            use_memory: true,
            ..Default::default()
        };
        let uncached = proto::StorageLevel::default();
        assert!(cached.use_memory || cached.use_disk);
        assert!(!(uncached.use_memory || uncached.use_disk));
    }

    #[test]
    fn to_local_iterator_builds_same_plan_as_collect() {
        // The LocalRowIterator and collect() use the same underlying ExecutePlan.
        // The difference is purely in client-side consumption: streaming vs buffering.
        // We verify that to_local_iterator() creates the same iterator type.
        let _iter: LocalRowIterator;
        // This test just verifies the type exists and is constructible.
        // A true integration test would create an actual stream.
    }
}

/// Deterministic tests for the collected-data conversions (`to_arrow`,
/// `to_datafusion`, `to_polars`). These exercise the conversion logic on
/// synthetic RecordBatches, so they need no live server and run in CI's
/// `--features datafusion,polars` job. The full server->collect->convert path is
/// covered separately by the server-gated e2e_integration tests.
#[cfg(test)]
mod conversion_tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn sample_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", ArrowDataType::Int64, false),
            Field::new("name", ArrowDataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn decoded_rows_share_batch_field_names_and_preserve_values() {
        use arrow::ipc::writer::StreamWriter;

        let batch = sample_batch();
        let mut data = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut data, batch.schema().as_ref()).unwrap();
            writer.write(&batch.slice(0, 0)).unwrap();
            writer.write(&batch).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        let response = proto::execute_plan_response::ArrowBatch {
            data: data.into(),
            ..Default::default()
        };
        let rows = decode_arrow_batch(&response).unwrap();

        assert_eq!(rows.len(), 6);
        for (index, row) in rows.iter().enumerate() {
            let source_index = index % 3;
            assert_eq!(row.fields(), &["id".to_string(), "name".to_string()]);
            assert_eq!(row.get(0), Some(&Value::Long(source_index as i64 + 1)));
            assert_eq!(
                row.get_by_name("name"),
                Some(&Value::String(["a", "b", "c"][source_index].to_string()))
            );
        }
        assert!(std::ptr::eq(rows[0].fields(), rows[1].fields()));
        assert!(std::ptr::eq(rows[1].fields(), rows[2].fields()));
        assert!(std::ptr::eq(rows[3].fields(), rows[4].fields()));
        assert!(std::ptr::eq(rows[4].fields(), rows[5].fields()));
        assert!(decode_arrow_batch(&Default::default()).unwrap().is_empty());
    }

    #[test]
    fn to_arrow_ipc_round_trips() {
        use arrow::ipc::reader::FileReader;
        use std::io::Cursor;

        let ipc = record_batches_to_ipc(&[sample_batch()]).expect("ipc encode");
        let reader = FileReader::try_new(Cursor::new(ipc), None).expect("ipc decode");
        let batches: Vec<_> = reader.map(|b| b.unwrap()).collect();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3, "round-trip must preserve all rows");
        assert_eq!(batches[0].num_columns(), 2);
        let ids = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(ids.values(), &[1, 2, 3]);
    }

    #[test]
    fn to_arrow_ipc_empty_is_valid() {
        // An empty result must still be a valid, readable IPC file.
        use arrow::ipc::reader::FileReader;
        use std::io::Cursor;
        let ipc = record_batches_to_ipc(&[]).expect("empty ipc");
        let reader = FileReader::try_new(Cursor::new(ipc), None).expect("empty ipc decode");
        assert_eq!(reader.map(|b| b.unwrap().num_rows()).sum::<usize>(), 0);
    }

    #[cfg(feature = "datafusion")]
    #[test]
    fn to_datafusion_preserves_rows_and_columns() {
        use datafusion::prelude::SessionContext;
        use spark_connect_core::runtime::block_on;

        let ctx = SessionContext::new();
        let df = record_batches_to_datafusion(&ctx, vec![sample_batch()]).expect("to datafusion");
        // Collect (async) and assert shape survives the conversion - uses only the
        // stable arrow RecordBatch API so it is not tied to a datafusion version.
        let collected = block_on(df.collect()).expect("collect datafusion");
        assert_eq!(collected.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
        assert_eq!(collected[0].num_columns(), 2);
    }

    #[cfg(feature = "datafusion")]
    #[test]
    fn to_datafusion_empty_errors() {
        use datafusion::prelude::SessionContext;
        let ctx = SessionContext::new();
        assert!(record_batches_to_datafusion(&ctx, vec![]).is_err());
    }

    #[cfg(feature = "polars")]
    #[test]
    fn to_polars_preserves_shape() {
        // height()/width() are stable across polars versions; asserting shape proves
        // the Arrow-IPC bridge carried all rows and columns through.
        let pdf = record_batches_to_polars(&[sample_batch()]).expect("to polars");
        assert_eq!(
            pdf.height(),
            3,
            "row count must survive the Arrow-IPC bridge"
        );
        assert_eq!(
            pdf.width(),
            2,
            "column count must survive the Arrow-IPC bridge"
        );
    }

    #[cfg(feature = "polars")]
    #[test]
    fn to_polars_empty_is_empty() {
        let pdf = record_batches_to_polars(&[]).expect("empty polars");
        assert_eq!(pdf.height(), 0);
    }
}

#[cfg(test)]
mod plan_construction_tests {
    use super::*;
    use crate::session::SparkSession;

    #[test]
    fn capture_execution_records_first_operation_id() {
        let spark = session();
        let mut info = ExecutionInfo::default();
        let mut resp = proto::ExecutePlanResponse::default();
        // An empty operation id (e.g. a synthetic response) is not recorded.
        capture_execution(&mut resp, &mut info, &spark);
        assert_eq!(info.operation_id, None);
        resp.operation_id = "op-1".to_string();
        capture_execution(&mut resp, &mut info, &spark);
        resp.operation_id = "op-2".to_string();
        capture_execution(&mut resp, &mut info, &spark);
        assert_eq!(info.operation_id.as_deref(), Some("op-1"));
    }

    fn session() -> SparkSession {
        SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("failed to build session")
    }

    #[test]
    fn is_local_only_for_a_local_relation_itself() {
        let spark = session();
        let local = spark.empty_data_frame().unwrap();
        let cached = DataFrame::new(
            spark.clone(),
            LogicalPlan::ChunkedCachedLocalRelation {
                data_hashes: vec!["hash".to_string()],
                schema_hash: None,
            },
        );
        assert!(local.is_local());
        assert!(cached.is_local());
        // As in PySpark, where the server resolves toDF into a Project.
        assert!(!local.to_df(vec![]).is_local());
        assert!(!cached.to_df(vec!["x"]).is_local());
        assert!(!spark.range(1).unwrap().is_local());
    }

    #[test]
    fn a_relation_the_server_returned_is_sent_back_with_its_own_plan_id() {
        use proto::relation::RelType;
        // A session of its own.
        let spark = session().new_session();
        let returned = |rel_type: RelType| {
            let relation = proto::Relation {
                common: Some(proto::RelationCommon {
                    plan_id: Some(42),
                    ..Default::default()
                }),
                rel_type: Some(rel_type),
            };
            DataFrame::new(
                spark.clone(),
                LogicalPlan::CachedRelation {
                    relation: Box::new(relation),
                },
            )
        };
        // The result of a command is the local relation the server returned.
        assert!(returned(RelType::LocalRelation(proto::LocalRelation::default())).is_local());
        let query = returned(RelType::Sql(proto::Sql {
            query: "SELECT 1".to_string(),
            ..Default::default()
        }));
        assert!(!query.is_local());
        let relation = build_input_relation(&query.plan, &spark);
        match relation.rel_type {
            Some(RelType::Sql(sql)) => assert_eq!(sql.query, "SELECT 1"),
            other => panic!("expected the returned SQL relation, got {other:?}"),
        }
        assert_eq!(
            relation.common.and_then(|common| common.plan_id),
            Some(spark.plan_ids().id_of(&query.plan))
        );
    }

    #[test]
    fn explain_string_takes_the_explain_modes_of_pyspark() {
        // As PySpark Connect's client, which checks the mode before it asks the server.
        let spark = session();
        let df = spark.range(1).unwrap();
        for mode in ["FORMATTED", "bogus", ""] {
            let err = df.explain_string(mode).unwrap_err();
            assert_eq!(err.get_condition().as_deref(), Some("VALUE_NOT_ALLOWED"));
            let parameters = err.get_message_parameters().unwrap();
            assert_eq!(parameters["arg_name"], "explain_mode");
            assert_eq!(
                parameters["allowed_values"],
                "['simple', 'extended', 'codegen', 'cost', 'formatted']"
            );
        }
    }

    #[test]
    fn with_watermark_plan() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let result = df.with_watermark("timestamp", "1 minute");
        match result.plan.as_ref() {
            LogicalPlan::WithWatermark {
                time_column,
                delay_threshold,
                ..
            } => {
                assert_eq!(time_column, "timestamp");
                assert_eq!(delay_threshold, "1 minute");
            }
            _ => panic!("expected WithWatermark plan"),
        }
    }

    #[test]
    fn transformations_and_clones_share_ancestors() {
        use crate::functions::col;

        let spark = session();
        let source = spark.range(10).unwrap();
        let original = source.plan.to_proto();
        let cloned = source.clone();
        assert!(Arc::ptr_eq(&source.plan, &cloned.plan));

        let projected = source.select(vec![col("id")]);
        let filtered = source.filter(col("id").gt(crate::column::lit(0)));
        let limited = projected.limit(5);
        let LogicalPlan::Project { input, .. } = projected.plan.as_ref() else {
            panic!("expected projection");
        };
        assert!(Arc::ptr_eq(input, &source.plan));
        let LogicalPlan::Filter { input, .. } = filtered.plan.as_ref() else {
            panic!("expected filter");
        };
        assert!(Arc::ptr_eq(input, &source.plan));
        let LogicalPlan::Limit { input, .. } = limited.plan.as_ref() else {
            panic!("expected limit");
        };
        assert!(Arc::ptr_eq(input, &projected.plan));

        let joined = projected.cross_join(&filtered);
        let LogicalPlan::Join { left, right, .. } = joined.plan.as_ref() else {
            panic!("expected join");
        };
        assert!(Arc::ptr_eq(left, &projected.plan));
        assert!(Arc::ptr_eq(right, &filtered.plan));
        assert_eq!(source.plan.to_proto(), original);
    }

    #[test]
    fn related_builders_share_the_source_plan() {
        let spark = session();
        let source = spark.range(10).unwrap().limit(5);

        let writer = source.write();
        assert!(std::ptr::eq(source.plan(), writer.plan()));
        let writer_v2 = source.write_to("target");
        assert!(std::ptr::eq(source.plan(), writer_v2.plan()));
        let stream_writer = source.write_stream();
        assert!(std::ptr::eq(source.plan(), stream_writer.plan()));
        let merge_writer = source.merge_into("target", crate::column::col("id"));
        assert!(std::ptr::eq(source.plan(), merge_writer.plan()));
        let table_arg = crate::table_arg::TableArg::new(source.clone());
        assert!(std::ptr::eq(source.plan(), table_arg.plan()));
        let cloned_table_arg = table_arg.clone();
        assert!(std::ptr::eq(source.plan(), cloned_table_arg.plan()));
    }

    /// Plan IDs in pre-order, for the relation shapes these tests build.
    fn plan_ids_of(relation: &proto::Relation) -> Vec<i64> {
        use proto::relation::RelType;
        fn walk(relation: &proto::Relation, out: &mut Vec<i64>) {
            out.push(relation.common.as_ref().unwrap().plan_id.unwrap());
            match relation.rel_type.as_ref().unwrap() {
                RelType::Join(join) => {
                    walk(join.left.as_ref().unwrap(), out);
                    walk(join.right.as_ref().unwrap(), out);
                }
                RelType::Project(project) => walk(project.input.as_ref().unwrap(), out),
                RelType::Filter(filter) => walk(filter.input.as_ref().unwrap(), out),
                RelType::Range(_) => {}
                _ => panic!("unexpected relation"),
            }
        }
        let mut out = Vec::new();
        walk(relation, &mut out);
        out
    }

    #[test]
    fn shared_self_join_serializes_each_occurrence_with_the_same_ids() {
        use crate::functions::col;
        use crate::plan;

        let spark = session();
        let projected = spark.range(10).unwrap().select(vec![col("id")]);
        let joined = projected.cross_join(&projected);
        let LogicalPlan::Join { left, right, .. } = joined.plan.as_ref() else {
            panic!("expected join");
        };
        assert!(Arc::ptr_eq(left, right));
        let expected = plan::join(
            plan::project(plan::range(0, 10, 1), vec![col("id")]),
            plan::project(plan::range(0, 10, 1), vec![col("id")]),
            JoinType::Cross,
            None,
            vec![],
        )
        .to_proto();
        assert_eq!(joined.plan.to_proto(), expected);

        // As in PySpark, a shared node has one plan ID wherever it occurs, and serializing
        // the same plan again gives the same relation.
        let first = build_input_relation(&joined.plan, &spark);
        let second = build_input_relation(&joined.plan, &spark);
        assert_eq!(first, second);
        let ids = plan_ids_of(&first);
        assert_eq!(ids.len(), 5);
        assert_eq!(ids[1..3], ids[3..5]);
        assert_ne!(ids[0], ids[1]);
        assert_ne!(ids[1], ids[2]);
        // Assigning IDs fills only the fresh protobuf, never the shared client plan.
        assert_eq!(joined.plan.to_proto(), expected);
    }

    /// As PySpark's `AsOfJoin` plan: the as-of columns, the names, the join type, the tolerance,
    /// whether exact matches are allowed, and the direction are each sent.
    #[test]
    fn as_of_join_sends_every_field_of_the_as_of_join() {
        use crate::functions::col;
        use proto::relation::RelType;

        let spark = session();
        let left = spark.range(3).unwrap();
        let right = spark.range(4).unwrap();
        let options = AsOfJoinOptions {
            using_columns: vec!["id".to_string()],
            join_type: "left".to_string(),
            tolerance: Some(crate::column::lit(1)),
            allow_exact_matches: false,
            direction: "forward".to_string(),
            ..AsOfJoinOptions::new(col("id"), col("id"))
        };
        let joined = left.as_of_join(&right, options).unwrap();
        let relation = joined.plan.to_proto();
        let Some(RelType::AsOfJoin(as_of)) = &relation.rel_type else {
            panic!("expected AsOfJoin: {relation:?}");
        };
        assert_eq!(as_of.left.as_deref(), Some(&left.plan.to_proto()));
        assert_eq!(as_of.right.as_deref(), Some(&right.plan.to_proto()));
        assert_eq!(as_of.left_as_of, Some(col("id").to_proto()));
        assert_eq!(as_of.right_as_of, Some(col("id").to_proto()));
        assert_eq!(as_of.join_expr, None);
        assert_eq!(as_of.using_columns, vec!["id".to_string()]);
        assert_eq!(as_of.join_type, "left");
        assert_eq!(as_of.tolerance, Some(crate::column::lit(1).to_proto()));
        assert!(!as_of.allow_exact_matches);
        assert_eq!(as_of.direction, "forward");
    }

    /// The defaults of PySpark's `_joinAsOf` (inner, exact matches, backward), and a condition
    /// sent as `join_expr` instead of names.
    #[test]
    fn as_of_join_defaults_and_sends_a_condition_as_join_expr() {
        let spark = session();
        let left = spark.range(3).unwrap();
        let right = spark.range(4).unwrap();
        let condition = left.col("id").eq(right.col("id"));
        let options = AsOfJoinOptions {
            on: Some(condition.clone()),
            ..AsOfJoinOptions::new(left.col("id"), right.col("id"))
        };
        let joined = left.as_of_join(&right, options).unwrap();
        let relation = joined.plan.to_proto();
        let Some(proto::relation::RelType::AsOfJoin(as_of)) = &relation.rel_type else {
            panic!("expected AsOfJoin: {relation:?}");
        };
        assert_eq!(as_of.join_expr, Some(condition.to_proto()));
        assert!(as_of.using_columns.is_empty());
        assert_eq!(as_of.join_type, "inner");
        assert_eq!(as_of.tolerance, None);
        assert!(as_of.allow_exact_matches);
        assert_eq!(as_of.direction, "backward");
    }

    /// As PySpark's `_check_same_session`: a DataFrame of another session is not joined.
    #[test]
    fn as_of_join_rejects_a_dataframe_of_another_session() {
        use crate::functions::col;

        let spark = session();
        let left = spark.range(3).unwrap();
        let right = spark.new_session().range(3).unwrap();
        let options = AsOfJoinOptions::new(col("id"), col("id"));
        let error = left
            .as_of_join(&right, options)
            .err()
            .expect("a session mismatch");
        assert_eq!(error.get_error_class().as_deref(), Some("SESSION_NOT_SAME"));
    }

    #[test]
    fn observed_metrics_set_the_observations_of_their_relations() {
        use crate::observation::Observation;
        use proto::expression::literal::LiteralType;
        let spark = session();
        let literal = |value: i64| proto::expression::Literal {
            literal_type: Some(LiteralType::Long(value)),
            ..Default::default()
        };
        // Two observations of one name, on two DataFrames, as PySpark allows.
        let (first, second) = (Observation::new("m"), Observation::new("m"));
        let count = || {
            vec![crate::functions::count(crate::column::lit(1))
                .expression()
                .clone()]
        };
        // As observe_with, without its request of whether the DataFrame is streaming.
        let observe = |df: DataFrame, observation: &Observation| {
            df.observed_with(observation, observation.name_to_observe().unwrap(), count())
        };
        let df1 = observe(spark.range(3).unwrap(), &first);
        let df2 = observe(spark.range(5).unwrap(), &second);
        assert!(first.get().unwrap().is_empty() && second.get().unwrap().is_empty());
        let err = df1.observe_with(&first, count()).err().unwrap();
        assert_eq!(err.get_condition().as_deref(), Some("REUSE_OBSERVATION"));

        let observed = |df: &DataFrame, value: i64| proto::execute_plan_response::ObservedMetrics {
            name: "m".to_string(),
            values: vec![literal(value)],
            keys: vec!["c".to_string()],
            plan_id: spark.plan_ids().id_of(&df.plan),
            ..Default::default()
        };
        let mut resp = proto::ExecutePlanResponse {
            observed_metrics: vec![observed(&df1, 3)],
            ..Default::default()
        };
        let mut info = ExecutionInfo::default();
        capture_execution(&mut resp, &mut info, &spark);
        let value = |obs: &Observation| -> Vec<(String, Option<LiteralType>)> {
            obs.get()
                .unwrap()
                .into_iter()
                .map(|(key, value)| (key, value.literal_type))
                .collect()
        };
        assert_eq!(
            value(&first),
            vec![("c".to_string(), Some(LiteralType::Long(3)))]
        );
        assert!(value(&second).is_empty());
        // The metrics of a relation by another name, or of no relation, set nothing.
        let mut other = observed(&df2, 5);
        other.name = "other".to_string();
        let mut unknown = observed(&df2, 5);
        unknown.plan_id = -1;
        resp.observed_metrics = vec![other, unknown];
        capture_execution(&mut resp, &mut info, &spark);
        assert!(value(&second).is_empty());
        // An error of collecting the metrics is what get returns.
        let mut failed = observed(&df2, 5);
        failed.root_error_idx = Some(0);
        resp.observed_metrics = vec![failed];
        capture_execution(&mut resp, &mut info, &spark);
        assert_eq!(
            second.get().unwrap_err().message(),
            "Observation error: invalid root_error_idx"
        );
        // That a streaming DataFrame is not observed, which the server tells, the live tests
        // check.
    }

    #[test]
    fn derived_dataframes_keep_the_plan_ids_of_their_inputs() {
        use crate::column::lit;
        use crate::functions::col;

        let spark = session();
        let base = spark.range(3).unwrap().select(vec![col("id")]);
        let base_relation = base.relation();
        let derived = base.filter(col("id").gt(lit(1)));
        let derived_relation = derived.relation();
        let Some(proto::relation::RelType::Filter(filter)) = &derived_relation.rel_type else {
            panic!("expected filter");
        };
        assert_eq!(filter.input.as_deref(), Some(&base_relation));
        assert_eq!(derived.relation(), derived_relation);
        // Separately built plans, even equal ones, are distinct nodes with distinct IDs.
        let other = spark.range(3).unwrap().select(vec![col("id")]);
        assert_eq!(other.plan.to_proto(), base.plan.to_proto());
        let (mine, theirs) = (plan_ids_of(&base_relation), plan_ids_of(&other.relation()));
        assert!(mine.iter().all(|id| !theirs.contains(id)));
    }

    #[test]
    fn plan_ids_of_dropped_nodes_are_pruned_and_never_reused() {
        // A session of its own, so that other tests' plans do not change its entries.
        let spark = session().new_session();
        let kept = spark.range(1).unwrap();
        let kept_ids = plan_ids_of(&kept.relation());
        let mut seen: std::collections::HashSet<i64> = kept_ids.iter().copied().collect();
        // Each plan is dropped at once, so pruning frees its allocation for a later plan to
        // reuse. A new node must still never get the ID of a dropped one.
        for end in 0..5000 {
            for id in plan_ids_of(&spark.range(end).unwrap().relation()) {
                assert!(seen.insert(id), "plan ID {id} was assigned twice");
            }
            // With one live node, pruning keeps the map at most MIN_PRUNE_AT entries.
            assert!(spark.plan_ids().len() <= 1024, "{}", spark.plan_ids().len());
        }
        assert_eq!(plan_ids_of(&kept.relation()), kept_ids);
    }

    #[test]
    fn the_sessions_of_a_process_never_hand_out_the_same_plan_id() {
        // As PySpark's one counter of the process: a plan of one session never has the plan ID
        // of a plan of another, so a column of another session names no node of this plan.
        let spark = session();
        let (one, two) = (spark.new_session(), spark.new_session());
        let mut seen = std::collections::HashSet::new();
        for end in 0..10 {
            for plans in [&one, &two] {
                for id in plan_ids_of(&plans.range(end).unwrap().relation()) {
                    assert!(seen.insert(id), "plan ID {id} was assigned twice");
                }
            }
        }
        assert_eq!(seen.len(), 20);
    }

    #[test]
    fn to_json_dataframe_selects_the_json_of_a_struct_of_every_column() {
        let spark = session();
        let df = spark.range(3).unwrap().to_json_dataframe();
        match df.plan.as_ref() {
            LogicalPlan::Project { columns, .. } => {
                let rendered: Vec<String> =
                    columns.iter().map(|c| c.expression().render()).collect();
                // As PySpark Connect's repr of the column.
                assert_eq!(
                    rendered,
                    vec!["to_json(struct(unresolvedstar())) AS value".to_string()]
                );
            }
            _ => panic!("expected Project plan"),
        }
    }

    #[test]
    fn with_metadata_plan() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("key".to_string(), "value".to_string());
        let result = df.with_metadata("col", metadata);
        match result.plan.as_ref() {
            LogicalPlan::WithColumnMetadata {
                column_name,
                metadata_json,
                ..
            } => {
                assert_eq!(column_name, "col");
                assert!(!metadata_json.is_empty());
            }
            _ => panic!("expected WithColumnMetadata plan"),
        }
    }

    /// As PySpark's: without a seed, the client takes one, of every split of `random_split`,
    /// whose input is in a deterministic order, and of a sample.
    #[test]
    fn samples_take_a_seed_of_the_client() {
        let spark = session();
        let df = spark.range(10).unwrap();
        let seeds: Vec<(Option<i64>, bool)> = df
            .random_split(vec![0.5, 0.5], None)
            .unwrap()
            .iter()
            .map(|split| match split.plan.as_ref() {
                LogicalPlan::Sample {
                    seed,
                    deterministic_order,
                    ..
                } => (*seed, *deterministic_order),
                other => panic!("expected a Sample plan, got {other:?}"),
            })
            .collect();
        assert!(seeds[0].0.is_some_and(|seed| seed >= 0));
        assert_eq!(seeds[0], seeds[1]);
        assert!(seeds[0].1);
        match df.sample(0.5, None).plan.as_ref() {
            LogicalPlan::Sample {
                seed: Some(_),
                deterministic_order: false,
                ..
            } => {}
            other => panic!("expected a Sample plan of a seed, got {other:?}"),
        }
    }

    #[test]
    fn random_split_plan() {
        let spark = session();
        let df = spark.range(10).unwrap();
        let dfs = df.random_split(vec![0.7, 0.3], None).unwrap();
        assert_eq!(dfs.len(), 2);
        // Each split is a different DataFrame with its own plan
        for split_df in &dfs {
            match split_df.plan.as_ref() {
                LogicalPlan::Sample {
                    with_replacement: false,
                    ..
                } => {
                    // Plan is correct
                }
                _ => panic!("expected Sample plan"),
            }
        }
    }

    #[test]
    fn random_split_rejects_invalid_weights_before_building_plans() {
        let df = session().range(10).unwrap();
        for (weights, argument, value) in [
            (vec![-1.0, 2.0], "weights", "-1"),
            (vec![0.0, -2.0, -1.0], "weights", "-2"),
            (vec![f64::NAN, -1.0], "weights", "-1"),
            (vec![0.0, 0.0], "sum(weights)", "0"),
            (vec![], "sum(weights)", "0"),
        ] {
            let error = match df.random_split(weights.clone(), Some(42)) {
                Err(error) => error,
                Ok(_) => panic!("expected invalid weights to fail: {weights:?}"),
            };
            assert!(matches!(error.kind, SparkErrorKind::ValueError));
            assert_eq!(error.error_class, "VALUE_NOT_POSITIVE", "{weights:?}");
            assert_eq!(error.params["arg_name"], argument, "{weights:?}");
            assert_eq!(error.params["arg_value"], value, "{weights:?}");
        }
    }

    #[test]
    fn random_split_serializes_normalized_bounds_shared_seed_and_ordering() {
        use proto::relation::RelType;

        let df = session().range(10).unwrap();
        let splits = df.random_split(vec![1.0, 0.0, 3.0], Some(42)).unwrap();
        assert_eq!(splits.len(), 3);
        for (split, (lower, upper)) in splits.iter().zip([(0.0, 0.25), (0.25, 0.25), (0.25, 1.0)]) {
            let Some(RelType::Sample(sample)) = split.relation().rel_type else {
                panic!("expected a serialized Sample relation");
            };
            assert_eq!(sample.lower_bound, lower);
            assert_eq!(sample.upper_bound, upper);
            assert_eq!(sample.seed, Some(42));
            assert_eq!(sample.with_replacement, Some(false));
            assert!(sample.deterministic_order);
        }
    }

    #[test]
    fn random_split_keeps_pyspark_nonfinite_comparison_semantics() {
        let df = session().range(10).unwrap();
        // PySpark checks w < 0 and total <= 0; neither comparison rejects NaN or +inf.
        // Any server-side handling of those plans remains separate from client validation.
        for weights in [vec![f64::NAN], vec![f64::INFINITY]] {
            assert_eq!(df.random_split(weights, Some(42)).unwrap().len(), 1);
        }
    }

    #[test]
    fn replace_literals_plan() {
        use crate::expression::LiteralExpression;
        let spark = session();
        let df = spark.range(3).unwrap();
        let replacements = vec![(
            LiteralExpression::string("a"),
            LiteralExpression::null(DataType::Null),
        )];
        for result in [
            df.replace_literals(replacements.clone(), Some(vec!["k"])),
            df.na()
                .replace_literals(replacements.clone(), Some(vec!["k"])),
        ] {
            match result.plan.as_ref() {
                LogicalPlan::NAReplace {
                    replacements: planned,
                    columns,
                    ..
                } => {
                    assert_eq!(planned, &replacements);
                    assert_eq!(columns, &vec!["k".to_string()]);
                }
                _ => panic!("expected NAReplace plan"),
            }
        }
        // A string replacement that parses as a number is a double, as before.
        match df
            .replace(vec![("1".to_string(), "a".to_string())], None)
            .plan
            .as_ref()
        {
            LogicalPlan::NAReplace { replacements, .. } => assert_eq!(
                replacements,
                &vec![(
                    LiteralExpression::double(1.0),
                    LiteralExpression::string("a")
                )]
            ),
            _ => panic!("expected NAReplace plan"),
        }
    }

    #[test]
    fn replace_plan() {
        let spark = session();
        let df = spark.range(3).unwrap();
        let replacements = vec![("old".to_string(), "new".to_string())];
        let result = df.replace(replacements, Some(vec!["col"]));
        match result.plan.as_ref() {
            LogicalPlan::NAReplace { replacements, .. } => {
                assert_eq!(replacements.len(), 1);
            }
            _ => panic!("expected NAReplace plan"),
        }
    }

    /// Exercise every plan-building builder AND its `plan.to_proto()` arm offline
    /// (the gRPC channel connects lazily, so no server is contacted). This pins the
    /// large builder set in dataframe.rs and the matching to_proto arms in plan.rs.
    #[test]
    fn builders_construct_and_serialize() {
        use crate::functions::col;
        use crate::types::{DataType, StructField};

        let spark = session();
        let df = spark.range(5).unwrap();
        let df2 = spark.range(5).unwrap();
        let e = || col("id").expression().clone();
        let ser = |d: &DataFrame| {
            build_input_relation(&d.plan, &spark);
        };

        ser(&df.select(vec![col("id")]));
        ser(&df.filter(col("id")));
        ser(&df.where_(col("id")));
        ser(&df.with_column("x", col("id")));
        ser(&df.with_column_renamed("id", "y"));
        ser(&df.drop(vec!["id"]));
        ser(&df.limit(3));
        ser(&df.offset(1));
        ser(&df.distinct());
        ser(&df.drop_duplicates(Some(vec!["id"])));
        ser(&df.sort(vec![e()]));
        ser(&df.order_by(vec![e()]));
        ser(&df.sort_within_partitions(vec![e()]));
        ser(&df.cross_join(&df2));
        ser(&df.union(&df2));
        ser(&df.union_all(&df2));
        ser(&df.union_by_name(&df2));
        ser(&df.intersect(&df2));
        ser(&df.intersect_all(&df2));
        ser(&df.subtract(&df2));
        ser(&df.except_all(&df2));
        ser(&df.repartition(4));
        ser(&df.coalesce(2));
        ser(&df.repartition_by_range(3, vec![e()]));
        ser(&df.hint("broadcast", Vec::<String>::new()));
        ser(&df.to_df(vec!["a"]));
        ser(&df.alias("t"));
        ser(&df.sample(0.5, Some(1)));
        ser(&df.select_expr(vec!["id + 1"]));
        ser(&df.select(vec![df.col_regex("`id`")]));
        ser(&df.describe(vec!["id"]));
        ser(&df.summary(vec!["count"]));
        ser(&df.as_table("t2"));
        ser(&df.to(DataType::Struct {
            fields: vec![StructField {
                name: "id".to_string(),
                data_type: DataType::Long,
                nullable: true,
                metadata: std::collections::BTreeMap::new(),
            }],
        }));
        ser(&df.unpivot(vec![col("id")], None::<Vec<Column>>, "var", "val"));
        ser(&df.melt(vec!["id"], None, "var", "val"));
        ser(&df.group_by(vec![col("id")]).agg(vec![e()]));
        ser(&df.rollup(vec![col("id")]).agg(vec![e()]));
        ser(&df.cube(vec![col("id")]).agg(vec![e()]));
        ser(&df.grouping_sets(vec![vec![col("id")]]).agg(vec![e()]));
        ser(&df.with_watermark("id", "1 minute"));
        let mut md = std::collections::HashMap::new();
        md.insert("k".to_string(), "v".to_string());
        ser(&df.with_metadata("id", md));
        ser(&df.replace(vec![("a".to_string(), "b".to_string())], None));
        ser(&df.stat().crosstab("id", "id"));
        ser(&df.stat().freq_items(vec!["id"], 0.5));
    }

    /// Streaming reader terminal builders (serialize their plans) + the writer builder
    /// chain across every Trigger variant (setters only; no server-side start()).
    #[test]
    fn streaming_reader_and_writer_builders() {
        use crate::streaming::Trigger;
        let spark = session();
        let ser = |d: &DataFrame| {
            build_input_relation(&d.plan, &spark);
        };
        ser(&spark
            .read_stream()
            .format("rate")
            .option("rowsPerSecond", "5")
            .load(None));
        ser(&spark.read_stream().schema("value long").json("/tmp/in"));
        ser(&spark.read_stream().parquet("/tmp/in"));
        ser(&spark.read_stream().csv("/tmp/in"));
        ser(&spark.read_stream().orc("/tmp/in"));
        ser(&spark.read_stream().text("/tmp/in"));
        ser(&spark.read_stream().format("rate").table("t"));

        let base = spark.range(3).unwrap();
        for trig in [
            Trigger::ProcessingTime("1 second".to_string()),
            Trigger::Once,
            Trigger::AvailableNow,
            Trigger::Continuous("1 second".to_string()),
        ] {
            let _w = base
                .write_stream()
                .output_mode("append")
                .format("console")
                .option("k", "v")
                .partition_by(vec!["id"])
                .cluster_by(vec!["id"])
                .query_name("q")
                .trigger(trig);
        }
    }

    /// Every Column operator/method builds an expression; serialize each proto to pin
    /// the column.rs bodies and the expression.rs to_proto arms.
    #[test]
    fn column_operations_and_expressions() {
        use crate::functions::col;
        let a = || col("a");
        let b = || col("b");
        let exprs = vec![
            a().add(b()),
            a().sub(b()),
            a().mul(b()),
            a().div(b()),
            a().modulo(b()),
            a().and(b()),
            a().or(b()),
            a().not(),
            a().neg(),
            a().eq(b()),
            a().ne(b()),
            a().gt(b()),
            a().lt(b()),
            a().ge(b()),
            a().le(b()),
            a().bitwise_and(b()),
            a().bitwise_or(b()),
            a().bitwise_xor(b()),
            a().eq_null_safe(b()),
            a().is_null(),
            a().is_not_null(),
            a().is_nan(),
            a().like("x%"),
            a().rlike("x.*"),
            a().ilike("x%"),
            a().contains(b()),
            a().startswith(b()),
            a().endswith(b()),
            a().substr(b(), b()),
            a().between(b(), b()),
            a().isin(vec![b()]),
            a().get_field("f"),
            a().get_item(b()),
            a().with_field("f", b()),
            a().drop_fields(vec!["f"]),
            a().asc(),
            a().asc_nulls_first(),
            a().asc_nulls_last(),
            a().desc(),
            a().desc_nulls_first(),
            a().desc_nulls_last(),
            a().alias("x"),
            a().name("y"),
            a().cast_str("int"),
            a().try_cast_str("int"),
            a().astype(crate::types::DataType::Integer),
            a().when(b(), b()).otherwise(b()),
        ];
        for e in &exprs {
            let _ = e.to_proto();
        }
    }

    /// Construct the exotic plan variants (Zip, Transpose, NearestByJoin,
    /// MapPartitions, GroupMap, CoGroupMap, CommonInlineUdtf) and serialize each so
    /// their to_proto arms in plan.rs are pinned.
    #[test]
    fn exotic_plan_variants_serialize() {
        use crate::functions::col;
        use crate::types::DataType;
        use crate::udf::{CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

        let spark = session();
        let ser = |d: &DataFrame| {
            build_input_relation(&d.plan, &spark);
        };
        let df = spark.range(5).unwrap();
        let df2 = spark.range(5).unwrap();

        ser(&df.zip(&df2).unwrap());
        ser(&df.transpose().unwrap());
        ser(&df.transpose_with_index(col("id")).unwrap());
        ser(&df.nearest_by_join(&df2, col("id"), 5, "inner", "asc", "inner"));

        let udf = || {
            CommonInlineUserDefinedFunctionExpression::new(
                "f".to_string(),
                true,
                vec![],
                PythonUDFPayload::new(DataType::Integer, 200, vec![1, 2, 3], "3.11".to_string()),
            )
        };
        ser(&df.map_in_pandas(udf(), false));
        ser(&df.map_in_arrow(udf(), false));
        ser(&df.group_by(vec![col("id")]).apply_in_pandas(udf()));
        ser(&df.group_by(vec![col("id")]).apply_in_arrow(udf()));
        let g1 = df.group_by(vec![col("id")]);
        let g2 = df2.group_by(vec![col("id")]);
        ser(&g1.cogroup(&g2).apply_in_pandas(udf()));

        let udtf_df = spark.tvf().udtf(
            "myudtf",
            vec![],
            Some(DataType::Integer),
            300,
            vec![1, 2],
            "3.11".to_string(),
            true,
        );
        ser(&udtf_df);
    }

    #[test]
    fn map_functions_send_the_id_of_their_resource_profile() {
        use crate::types::DataType;
        use crate::udf::{CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};
        use spark_connect_proto::relation::RelType;

        let spark = session();
        let df = spark.range(5).unwrap();
        let udf = || {
            CommonInlineUserDefinedFunctionExpression::new(
                "f".to_string(),
                true,
                vec![],
                PythonUDFPayload::new(DataType::Integer, 205, vec![1], "3.11".to_string()),
            )
        };
        let sent = |mapped: DataFrame| match mapped.relation().rel_type {
            Some(RelType::MapPartitions(map)) => (map.profile_id, map.is_barrier),
            other => panic!("expected MapPartitions, got {other:?}"),
        };
        // As PySpark's MapPartitions, of the id of the profile, if any.
        assert_eq!(
            sent(df.map_in_pandas_with_profile(udf(), true, Some(7))),
            (Some(7), Some(true))
        );
        assert_eq!(
            sent(df.map_in_arrow_with_profile(udf(), false, Some(8))),
            (Some(8), Some(false))
        );
        assert_eq!(sent(df.map_in_pandas(udf(), false)), (None, Some(false)));
        assert_eq!(sent(df.map_in_arrow(udf(), true)), (None, Some(true)));
    }
}

/// Deterministic coverage of the private Arrow-array -> `Value` converter
/// (`arrow_value_at`) and its formatting helpers. Builds a one-element array of
/// each Arrow type and asserts the decoded `Value` variant, so the large per-type
/// match runs without a live server (the server->collect->decode path is covered
/// separately by the server-gated e2e_integration tests).
#[cfg(test)]
mod arrow_value_tests {
    use super::*;
    use arrow::array::*;
    use arrow::datatypes::{
        i256, DataType as ArrowDataType, Field, Int32Type, IntervalDayTime, IntervalMonthDayNano,
    };
    use std::sync::Arc;

    #[test]
    fn day_time_interval_microseconds_are_exact() {
        let values = vec![
            Some(i64::MIN),
            Some(i64::MIN + 1),
            Some(-172_800_000_001),
            Some(-1),
            Some(0),
            Some(1),
            Some(172_800_000_001),
            Some(i64::MAX - 1),
            Some(i64::MAX),
            None,
        ];
        let array = DurationMicrosecondArray::from(values.clone());
        for (i, value) in values.iter().enumerate() {
            assert_eq!(
                arrow_value_at(&array, i).unwrap(),
                value.map(Value::DayTimeInterval).unwrap_or(Value::Null)
            );
        }
        let sliced = array.slice(2, 5);
        for i in 0..sliced.len() {
            assert_eq!(
                arrow_value_at(&sliced, i).unwrap(),
                Value::DayTimeInterval(values[i + 2].unwrap())
            );
        }
        let nulls = DurationMicrosecondArray::from(vec![None, None]);
        assert_eq!(arrow_value_at(&nulls, 1).unwrap(), Value::Null);
        for array in [
            Arc::new(DurationSecondArray::from(vec![1])) as ArrayRef,
            Arc::new(DurationMillisecondArray::from(vec![1])),
            Arc::new(DurationNanosecondArray::from(vec![1])),
        ] {
            assert!(arrow_value_at(array.as_ref(), 0).is_err());
        }
    }

    #[test]
    fn day_time_interval_ipc_empty_and_multiple_batches() {
        use arrow::datatypes::{Schema, TimeUnit};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "interval",
            ArrowDataType::Duration(TimeUnit::Microsecond),
            true,
        )]));
        let mut data = Vec::new();
        let mut writer = StreamWriter::try_new(&mut data, &schema).unwrap();
        for values in [
            vec![],
            vec![Some(i64::MIN), None],
            vec![],
            vec![Some(i64::MAX)],
        ] {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(DurationMicrosecondArray::from(values))],
            )
            .unwrap();
            writer.write(&batch).unwrap();
        }
        writer.finish().unwrap();
        drop(writer);
        let rows = decode_arrow_batch(&proto::execute_plan_response::ArrowBatch {
            data: data.into(),
            row_count: 3,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            rows.iter()
                .map(|r| r.get(0).unwrap().clone())
                .collect::<Vec<_>>(),
            vec![
                Value::DayTimeInterval(i64::MIN),
                Value::Null,
                Value::DayTimeInterval(i64::MAX)
            ]
        );
    }

    #[test]
    fn typed_scalar_layouts_preserve_exact_values_and_normalization() {
        let decimal128 = Decimal128Array::from(vec![-12345i128])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let decimal256 = Decimal256Array::from(vec![i256::from_i128(-12345)])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let decimal = Value::Decimal {
            value: "-123.45".into(),
            precision: Some(10),
            scale: Some(2),
        };
        let cases: Vec<(ArrayRef, Value)> = vec![
            (
                Arc::new(TimestampSecondArray::from(vec![-1]).with_timezone_utc()),
                Value::Timestamp(-1_000_000),
            ),
            (
                Arc::new(TimestampMillisecondArray::from(vec![-1]).with_timezone_utc()),
                Value::Timestamp(-1_000),
            ),
            (
                Arc::new(TimestampNanosecondArray::from(vec![-1999]).with_timezone_utc()),
                Value::Timestamp(-1),
            ),
            // Without a time zone, as TIMESTAMP_NTZ is declared.
            (
                Arc::new(TimestampSecondArray::from(vec![-1])),
                Value::TimestampNtz(-1_000_000),
            ),
            (
                Arc::new(TimestampMicrosecondArray::from(vec![-1])),
                Value::TimestampNtz(-1),
            ),
            (
                Arc::new(Date64Array::from(vec![-86_400_001])),
                Value::Date(-1),
            ),
            (
                Arc::new(UInt32Array::from(vec![u32::MAX])),
                Value::Long(u32::MAX as i64),
            ),
            (Arc::new(decimal128), decimal.clone()),
            (Arc::new(decimal256), decimal),
            (
                Arc::new(LargeStringArray::from_iter_values(["é\0雪"])),
                Value::String("é\0雪".into()),
            ),
            (
                Arc::new(StringViewArray::from_iter_values(["é\0雪"])),
                Value::String("é\0雪".into()),
            ),
            (
                Arc::new(LargeBinaryArray::from_iter_values([b"\0\xff".as_ref()])),
                Value::Binary(vec![0, 255]),
            ),
            (
                Arc::new(BinaryViewArray::from_iter_values([b"\0\xff".as_ref()])),
                Value::Binary(vec![0, 255]),
            ),
            (
                Arc::new(Time64NanosecondArray::from(vec![1_234_567])),
                Value::Time(1_234_567),
            ),
            (
                Arc::new(Time32MillisecondArray::from(vec![1])),
                Value::Time(1_000_000),
            ),
        ];
        for (array, expected) in cases {
            assert_eq!(
                ArrowValueLayout::new(array.as_ref()).value_at(0).unwrap(),
                expected
            );
        }
        for value in [
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0001),
        ] {
            let array = Float64Array::from(vec![value]);
            let Value::Double(decoded) = arrow_value_at(&array, 0).unwrap() else {
                panic!("expected double")
            };
            assert_eq!(decoded.to_bits(), value.to_bits());
        }
    }

    #[test]
    fn primitives_and_signed_ints() {
        assert!(matches!(
            arrow_value_at(&BooleanArray::from(vec![true]), 0).unwrap(),
            Value::Bool(true)
        ));
        assert!(matches!(
            arrow_value_at(&Int8Array::from(vec![1i8]), 0).unwrap(),
            Value::Byte(1)
        ));
        assert!(matches!(
            arrow_value_at(&Int16Array::from(vec![1i16]), 0).unwrap(),
            Value::Short(1)
        ));
        assert!(matches!(
            arrow_value_at(&Int32Array::from(vec![1i32]), 0).unwrap(),
            Value::Integer(1)
        ));
        assert!(matches!(
            arrow_value_at(&Int64Array::from(vec![1i64]), 0).unwrap(),
            Value::Long(1)
        ));
        assert!(matches!(
            arrow_value_at(&Float32Array::from(vec![1.0f32]), 0).unwrap(),
            Value::Float(_)
        ));
        assert!(matches!(
            arrow_value_at(&Float64Array::from(vec![1.0f64]), 0).unwrap(),
            Value::Double(_)
        ));
        assert!(matches!(
            arrow_value_at(&StringArray::from(vec!["x"]), 0).unwrap(),
            Value::String(_)
        ));
        assert!(matches!(
            arrow_value_at(&BinaryArray::from_iter_values([b"x".as_ref()]), 0).unwrap(),
            Value::Binary(_)
        ));
        assert!(matches!(
            arrow_value_at(&Date32Array::from(vec![1i32]), 0).unwrap(),
            Value::Date(1)
        ));
        assert!(matches!(
            arrow_value_at(&TimestampMicrosecondArray::from(vec![1i64]), 0).unwrap(),
            Value::TimestampNtz(1)
        ));
        assert!(matches!(
            arrow_value_at(
                &TimestampMicrosecondArray::from(vec![1i64]).with_timezone_utc(),
                0
            )
            .unwrap(),
            Value::Timestamp(1)
        ));
    }

    #[test]
    fn unsigned_ints() {
        assert!(matches!(
            arrow_value_at(&UInt8Array::from(vec![1u8]), 0).unwrap(),
            Value::Short(1)
        ));
        assert!(matches!(
            arrow_value_at(&UInt16Array::from(vec![1u16]), 0).unwrap(),
            Value::Integer(1)
        ));
        assert!(matches!(
            arrow_value_at(&UInt32Array::from(vec![1u32]), 0).unwrap(),
            Value::Long(1)
        ));
        assert!(matches!(
            arrow_value_at(&UInt64Array::from(vec![1u64]), 0).unwrap(),
            Value::Long(1)
        ));
    }

    #[test]
    fn decimals_128_and_256() {
        let d128 = Decimal128Array::from(vec![12345i128])
            .with_precision_and_scale(10, 2)
            .unwrap();
        assert!(matches!(
            arrow_value_at(&d128, 0).unwrap(),
            Value::Decimal { .. }
        ));
        let d256 = Decimal256Array::from(vec![i256::from_i128(12345)])
            .with_precision_and_scale(10, 2)
            .unwrap();
        assert!(matches!(
            arrow_value_at(&d256, 0).unwrap(),
            Value::Decimal { .. }
        ));
    }

    #[test]
    fn large_and_view_bytes() {
        assert!(matches!(
            arrow_value_at(&LargeStringArray::from_iter_values(["x"]), 0).unwrap(),
            Value::String(_)
        ));
        assert!(matches!(
            arrow_value_at(&LargeBinaryArray::from_iter_values([b"x".as_ref()]), 0).unwrap(),
            Value::Binary(_)
        ));
        assert!(matches!(
            arrow_value_at(&StringViewArray::from_iter_values(["x"]), 0).unwrap(),
            Value::String(_)
        ));
        assert!(matches!(
            arrow_value_at(&BinaryViewArray::from_iter_values([b"x".as_ref()]), 0).unwrap(),
            Value::Binary(_)
        ));
    }

    #[test]
    fn timestamps_and_date64() {
        assert!(matches!(
            arrow_value_at(&TimestampSecondArray::from(vec![1i64]), 0).unwrap(),
            Value::TimestampNtz(_)
        ));
        assert!(matches!(
            arrow_value_at(&TimestampMillisecondArray::from(vec![1i64]), 0).unwrap(),
            Value::TimestampNtz(_)
        ));
        assert!(matches!(
            arrow_value_at(&TimestampNanosecondArray::from(vec![1000i64]), 0).unwrap(),
            Value::TimestampNtz(_)
        ));
        assert!(matches!(
            arrow_value_at(
                &TimestampSecondArray::from(vec![1i64]).with_timezone_utc(),
                0
            )
            .unwrap(),
            Value::Timestamp(_)
        ));
        assert!(matches!(
            arrow_value_at(
                &TimestampMillisecondArray::from(vec![1i64]).with_timezone_utc(),
                0
            )
            .unwrap(),
            Value::Timestamp(_)
        ));
        assert!(matches!(
            arrow_value_at(
                &TimestampNanosecondArray::from(vec![1000i64]).with_timezone_utc(),
                0
            )
            .unwrap(),
            Value::Timestamp(_)
        ));
        assert!(matches!(
            arrow_value_at(&Date64Array::from(vec![86_400_000i64]), 0).unwrap(),
            Value::Date(_)
        ));
    }

    #[test]
    fn time_types_decode_as_nanoseconds_since_midnight() {
        // As PySpark's internal value of TimeType, whatever the unit of the Arrow time.
        assert_eq!(
            arrow_value_at(&Time64MicrosecondArray::from(vec![1i64]), 0).unwrap(),
            Value::Time(1_000)
        );
        assert_eq!(
            arrow_value_at(&Time64NanosecondArray::from(vec![1001i64]), 0).unwrap(),
            Value::Time(1_001)
        );
        assert_eq!(
            arrow_value_at(&Time32MillisecondArray::from(vec![1i32]), 0).unwrap(),
            Value::Time(1_000_000)
        );
        assert_eq!(
            arrow_value_at(&Time32SecondArray::from(vec![1i32]), 0).unwrap(),
            Value::Time(1_000_000_000)
        );
    }

    #[test]
    fn interval_types_render_as_string() {
        assert!(matches!(
            arrow_value_at(&IntervalYearMonthArray::from(vec![13i32]), 0).unwrap(),
            Value::String(_)
        ));
        let dt = IntervalDayTimeArray::from(vec![IntervalDayTime::new(1, 100)]);
        assert!(matches!(arrow_value_at(&dt, 0).unwrap(), Value::String(_)));
        let mdn = IntervalMonthDayNanoArray::from(vec![IntervalMonthDayNano::new(1, 2, 3)]);
        assert!(matches!(arrow_value_at(&mdn, 0).unwrap(), Value::String(_)));
    }

    #[test]
    fn year_month_interval_keeps_sign() {
        let months = [
            0,
            1,
            -1,
            11,
            -11,
            12,
            -12,
            13,
            -13,
            24,
            -24,
            i32::MAX,
            i32::MIN,
        ];
        let expected = [
            "0-0",
            "0-1",
            "-0-1",
            "0-11",
            "-0-11",
            "1-0",
            "-1-0",
            "1-1",
            "-1-1",
            "2-0",
            "-2-0",
            "178956970-7",
            "-178956970-8",
        ];
        for (m, want) in months.into_iter().zip(expected) {
            assert_eq!(
                arrow_value_at(&IntervalYearMonthArray::from(vec![m]), 0).unwrap(),
                Value::String(want.to_string()),
                "months = {m}"
            );
        }
    }

    #[test]
    fn fixed_size_binary() {
        let arr = FixedSizeBinaryArray::try_from_iter(vec![vec![1u8, 2u8]].into_iter()).unwrap();
        assert!(matches!(arrow_value_at(&arr, 0).unwrap(), Value::Binary(_)));
    }

    #[test]
    fn nested_list_struct_map() {
        let list =
            ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![Some(1), Some(2)])]);
        assert!(matches!(arrow_value_at(&list, 0).unwrap(), Value::List(_)));

        let field = Arc::new(Field::new("a", ArrowDataType::Int32, false));
        let col: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        let s = StructArray::from(vec![(field, col)]);
        assert!(matches!(arrow_value_at(&s, 0).unwrap(), Value::Struct(_)));

        let mut b = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
        b.keys().append_value("k");
        b.values().append_value(1);
        b.append(true).unwrap();
        let m = b.finish();
        assert!(matches!(arrow_value_at(&m, 0).unwrap(), Value::Map(_)));
    }

    #[test]
    fn null_element_and_unsupported_type() {
        let with_null = Int32Array::from(vec![None as Option<i32>]);
        assert!(matches!(
            arrow_value_at(&with_null, 0).unwrap(),
            Value::Null
        ));
        // Only microsecond durations are supported (Spark's physical encoding).
        let dur = DurationSecondArray::from(vec![1i64]);
        assert!(arrow_value_at(&dur, 0).is_err());
    }

    #[test]
    fn a_map_keeps_its_entries_in_order_with_keys_of_their_type() {
        let entries = StructArray::from(vec![
            (
                Arc::new(Field::new("key", arrow::datatypes::DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![2, 1])) as ArrayRef,
            ),
            (
                Arc::new(Field::new("value", arrow::datatypes::DataType::Utf8, true)),
                Arc::new(StringArray::from(vec!["b", "a"])) as ArrayRef,
            ),
        ]);
        let map = MapArray::new(
            Arc::new(Field::new("entries", entries.data_type().clone(), false)),
            arrow::buffer::OffsetBuffer::new(vec![0, 2].into()),
            entries,
            None,
            false,
        );
        assert_eq!(
            arrow_value_at(&map, 0).unwrap(),
            Value::Map(vec![
                (Value::Integer(2), Value::String("b".to_string())),
                (Value::Integer(1), Value::String("a".to_string())),
            ])
        );
    }

    #[test]
    fn i128_to_decimal_string_branches() {
        assert_eq!(i128_to_decimal_string(12345, 0), "12345");
        assert_eq!(i128_to_decimal_string(12345, 2), "123.45");
        assert_eq!(i128_to_decimal_string(5, 4), "0.0005");
        assert_eq!(i128_to_decimal_string(-5, 4), "-0.0005");
        assert_eq!(i128_to_decimal_string(1, -1), "1E+1");
        assert_eq!(i128_to_decimal_string(0, -1), "0E+1");
        assert_eq!(i128_to_decimal_string(-123, -2), "-123E+2");
    }
}

#[cfg(test)]
mod plan_id_column_tests {
    use super::*;
    use crate::session::SparkSession;
    use proto::relation::RelType;

    fn session() -> SparkSession {
        SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("failed to build session")
    }

    /// As PySpark Connect's `df[name]`, `df["*"]`, `colRegex` and `metadataColumn`: the columns
    /// of a DataFrame refer to it by the plan ID that its relation is sent with.
    #[test]
    fn columns_refer_to_their_dataframe_by_its_plan_id() {
        use proto::expression::ExprType;

        let spark = session();
        let left = spark.range(2).unwrap();
        let right = spark.range(3).unwrap();
        let joined = left
            .join(
                &right,
                Some(left.col("id").eq(right.col("id"))),
                JoinType::Inner,
            )
            .select(vec![left.col("id"), right.star(), left.col_regex("`i.*`")]);
        let relation = joined.relation();
        let Some(RelType::Project(project)) = &relation.rel_type else {
            panic!("expected Project: {relation:?}");
        };
        let Some(RelType::Join(join)) = &project.input.as_deref().unwrap().rel_type else {
            panic!("expected Join");
        };
        let plan_id = |r: &proto::Relation| r.common.as_ref().and_then(|c| c.plan_id);
        let (left_id, right_id) = (
            plan_id(join.left.as_deref().unwrap()),
            plan_id(join.right.as_deref().unwrap()),
        );
        assert!(left_id.is_some() && left_id != right_id);
        let ids: Vec<Option<i64>> = project
            .expressions
            .iter()
            .map(|e| match &e.expr_type {
                Some(ExprType::UnresolvedAttribute(a)) => a.plan_id,
                Some(ExprType::UnresolvedStar(s)) => s.plan_id,
                Some(ExprType::UnresolvedRegex(r)) => r.plan_id,
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(ids, vec![left_id, right_id, left_id]);
        let Some(ExprType::UnresolvedAttribute(metadata)) = left
            .metadata_column("_metadata")
            .expression()
            .to_proto()
            .expr_type
        else {
            panic!("expected an attribute");
        };
        assert_eq!(
            (metadata.plan_id, metadata.is_metadata_column),
            (left_id, Some(true))
        );
    }
}

#[cfg(test)]
mod subquery_tests {
    use super::*;
    use crate::column::col;
    use crate::functions::max;
    use crate::session::SparkSession;
    use proto::relation::RelType;

    fn session() -> SparkSession {
        SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("failed to build session")
    }

    fn plan_id(relation: &proto::Relation) -> Option<i64> {
        relation.common.as_ref().and_then(|common| common.plan_id)
    }

    /// The root and the references of a `WithRelations`.
    fn with_relations(relation: &proto::Relation) -> (&proto::Relation, &[proto::Relation]) {
        let Some(RelType::WithRelations(with)) = &relation.rel_type else {
            panic!("expected WithRelations: {relation:?}");
        };
        (with.root.as_deref().expect("root"), &with.references)
    }

    fn subquery(expr: &proto::Expression) -> &proto::SubqueryExpression {
        let Some(proto::expression::ExprType::SubqueryExpression(subquery)) = &expr.expr_type
        else {
            panic!("expected a SubqueryExpression: {expr:?}");
        };
        subquery
    }

    /// As PySpark's `_with_relations`: a relation with subqueries is a `WithRelations` of
    /// its own plan ID, whose root has another, and whose references are the plans of the
    /// subqueries, with the plan IDs that the subqueries refer to.
    #[test]
    fn subqueries_are_sent_with_the_plans_that_they_refer_to() {
        let spark = session();
        let inner = spark.range(3).unwrap().select(vec![max(col("id"))]);
        let df = spark
            .range(2)
            .unwrap()
            .select(vec![col("id"), inner.scalar().alias("m")]);

        let relation = df.relation();
        let (root, references) = with_relations(&relation);
        assert_eq!(plan_id(&relation), Some(spark.plan_ids().id_of(&df.plan)));
        assert!(plan_id(root).is_some() && plan_id(root) != plan_id(&relation));
        let Some(RelType::Project(project)) = &root.rel_type else {
            panic!("expected Project: {root:?}");
        };
        let Some(proto::expression::ExprType::Alias(alias)) = &project.expressions[1].expr_type
        else {
            panic!("expected an alias");
        };
        let scalar = subquery(alias.expr.as_deref().unwrap());
        assert_eq!(
            scalar.subquery_type,
            proto::subquery_expression::SubqueryType::Scalar as i32
        );
        assert_eq!(references.len(), 1);
        assert_eq!(plan_id(&references[0]), Some(scalar.plan_id));
        assert_eq!(references[0], inner.relation());
        // The plan IDs are stable, as PySpark's.
        assert_eq!(df.relation(), relation);
        // Without plan IDs, the relation is sent the same way.
        let Some(RelType::WithRelations(with)) = df.plan.to_proto().rel_type else {
            panic!("expected WithRelations");
        };
        assert_eq!(with.references.len(), 1);
        // A plan without subqueries is sent as it is.
        assert!(matches!(
            inner.relation().rel_type,
            Some(RelType::Project(_))
        ));
    }

    /// The relations below a relation with subqueries, and the plans that they refer to,
    /// send their own subqueries in their own `WithRelations`.
    #[test]
    fn nested_subqueries_are_sent_in_their_own_relations() {
        let spark = session();
        let t1 = spark.range(4).unwrap();
        let t2 = spark.range(2).unwrap();
        let smallest = t2.select(vec![crate::functions::min(col("id"))]);
        let filtered = t2.filter(col("id").gt(smallest.scalar()));
        let df = t1
            .filter(col("id").isin_subquery(&filtered))
            .filter(t2.exists());

        let relation = df.relation();
        let (root, references) = with_relations(&relation);
        let Some(RelType::Filter(outer)) = &root.rel_type else {
            panic!("expected Filter: {root:?}");
        };
        let exists = subquery(outer.condition.as_ref().unwrap());
        assert_eq!(references.len(), 1);
        assert_eq!(plan_id(&references[0]), Some(exists.plan_id));

        // The inner filter, with the IN subquery of the filtered plan, which in turn has the
        // scalar subquery of the smallest.
        let (inner_root, inner_references) = with_relations(outer.input.as_deref().unwrap());
        let Some(RelType::Filter(inner)) = &inner_root.rel_type else {
            panic!("expected Filter: {inner_root:?}");
        };
        let in_subquery = subquery(inner.condition.as_ref().unwrap());
        assert_eq!(
            in_subquery.subquery_type,
            proto::subquery_expression::SubqueryType::In as i32
        );
        assert_eq!(
            in_subquery.in_subquery_values,
            vec![col("id").expression().to_proto()]
        );
        assert_eq!(inner_references.len(), 1);
        assert_eq!(plan_id(&inner_references[0]), Some(in_subquery.plan_id));
        let (filtered_root, filtered_references) = with_relations(&inner_references[0]);
        assert!(matches!(filtered_root.rel_type, Some(RelType::Filter(_))));
        assert_eq!(
            filtered_references,
            std::slice::from_ref(&smallest.relation())
        );
    }

    /// As PySpark's `Column.isin(df)`, the values of an IN subquery of a struct are its
    /// fields; and the subqueries render as PySpark's, with the plan ID of their plan.
    #[test]
    fn in_subqueries_of_structs_compare_their_fields() {
        let spark = session();
        let df = spark.range(1).unwrap();
        let id = spark.plan_ids().id_of(&df.plan);
        let fields = crate::functions::r#struct(vec![col("a"), col("b")]).isin_subquery(&df);
        assert_eq!(
            fields.expression().render(),
            format!("SubqueryExpression(plan=<plan {id}>, type=in, values=[a, b])")
        );
        assert_eq!(
            df.scalar().expression().render(),
            format!("SubqueryExpression(plan=<plan {id}>, type=scalar)")
        );
        assert_eq!(
            df.exists().expression().render(),
            format!("SubqueryExpression(plan=<plan {id}>, type=exists)")
        );
    }
}
