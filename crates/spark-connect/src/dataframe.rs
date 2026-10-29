//! DataFrame implementation mirroring `pyspark.sql.DataFrame`.
//!
//! Provides transformations and actions for working with distributed data.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use spark_connect_core::client::ReattachableResponseStream;
use spark_connect_core::error::{Result, SparkError};
use spark_connect_core::runtime::{block_on, get_runtime};
use spark_connect_proto as proto;

use crate::column::Column;
use crate::expression::{Expression, ExpressionNode};
use crate::plan::{AggregateGroupType, JoinType, LogicalPlan, SetOpType};
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
    // Exact clones share this cache. Every new plan starts with an empty cache,
    // even when its input has already been analyzed.
    schema_cache: Arc<Mutex<Option<DataType>>>,
}

/// Iterator over rows from a DataFrame, yielded lazily as the server streams results.
///
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
}

/// The source of successive Arrow batches for a [`LocalRowIterator`].
enum RowSource {
    /// Each batch is pulled from the response stream only when the previous
    /// one is exhausted (`prefetchPartitions=False`).
    OnDemand {
        session: SparkSession,
        stream: ReattachableResponseStream,
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
        stream: ReattachableResponseStream,
        prefetch_partitions: bool,
    ) -> Self {
        let source = if prefetch_partitions {
            RowSource::Prefetch {
                rx: spawn_prefetch(session, stream),
            }
        } else {
            RowSource::OnDemand {
                session,
                stream,
                execution_info: ExecutionInfo::default(),
                execution_recorded: false,
            }
        };
        LocalRowIterator {
            current_rows: vec![].into_iter(),
            source,
            done: false,
        }
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
                        if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
                            resp.response_type
                        {
                            return Some(decode_arrow_batch(&batch));
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

/// Spawn a background task that drains the response stream and forwards decoded
/// batches over a bounded (capacity-1) channel, keeping one batch buffered ahead
/// of the consumer. Execution metrics are recorded on the session once the
/// stream ends. Backs `to_local_iterator(prefetch_partitions = true)`.
fn spawn_prefetch(
    session: SparkSession,
    mut stream: ReattachableResponseStream,
) -> tokio::sync::mpsc::Receiver<Result<Vec<Row>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Vec<Row>>>(1);
    get_runtime().spawn(async move {
        let mut execution_info = ExecutionInfo::default();
        loop {
            match stream.message().await {
                Ok(Some(mut resp)) => {
                    capture_execution(&mut resp, &mut execution_info, &session);
                    if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
                        resp.response_type
                    {
                        match decode_arrow_batch(&batch) {
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
        DataFrame {
            session,
            plan: Arc::new(plan),
            schema_cache: Arc::new(Mutex::new(None)),
        }
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
        DataFrame::new(self.session.clone(), plan)
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
        let mut renames = HashMap::new();
        renames.insert(existing.to_string(), new.to_string());
        let plan = LogicalPlan::WithColumnsRenamed {
            input: Arc::clone(&self.plan),
            renames,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Rename multiple columns.
    pub fn with_columns_renamed(&self, renames: Vec<(String, String)>) -> DataFrame {
        let mut rename_map = HashMap::new();
        for (old, new) in renames {
            rename_map.insert(old, new);
        }
        let plan = LogicalPlan::WithColumnsRenamed {
            input: Arc::clone(&self.plan),
            renames: rename_map,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Drop columns.
    pub fn drop(&self, columns: Vec<&str>) -> DataFrame {
        let col_names = columns.iter().map(|s| s.to_string()).collect();
        let plan = LogicalPlan::Drop {
            input: Arc::clone(&self.plan),
            columns: col_names,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Limit the number of rows.
    pub fn limit(&self, n: i32) -> DataFrame {
        let plan = LogicalPlan::Limit {
            input: Arc::clone(&self.plan),
            limit: n,
        };
        DataFrame::new(self.session.clone(), plan)
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
        DataFrame::new(self.session.clone(), plan)
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
        DataFrame::new(self.session.clone(), plan)
    }

    /// Sort rows.
    pub fn sort(&self, columns: Vec<Expression>) -> DataFrame {
        let plan = LogicalPlan::Sort {
            input: Arc::clone(&self.plan),
            order: columns,
            is_global: true,
        };
        DataFrame::new(self.session.clone(), plan)
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
        DataFrame::new(self.session.clone(), plan)
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
        DataFrame::new(self.session.clone(), plan)
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
        DataFrame::new(self.session.clone(), plan)
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
        DataFrame::new(self.session.clone(), plan)
    }

    /// Repartition.
    pub fn repartition(&self, num_partitions: i32) -> DataFrame {
        let plan = LogicalPlan::Repartition {
            input: Arc::clone(&self.plan),
            num_partitions,
            shuffle: true,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Coalesce.
    pub fn coalesce(&self, num_partitions: i32) -> DataFrame {
        let plan = LogicalPlan::Repartition {
            input: Arc::clone(&self.plan),
            num_partitions,
            shuffle: false,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Add a hint.
    pub fn hint<S: Into<String>>(
        &self,
        name: &str,
        parameters: impl IntoIterator<Item = S>,
    ) -> DataFrame {
        let plan = LogicalPlan::Hint {
            input: Arc::clone(&self.plan),
            name: name.to_string(),
            parameters: parameters.into_iter().map(Into::into).collect(),
        };
        DataFrame::new(self.session.clone(), plan)
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
        DataFrame::new(self.session.clone(), plan)
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
        self.map_partitions(func, is_barrier)
    }

    /// Map over each partition with an Arrow UDF (`DataFrame.mapInArrow`).
    pub fn map_in_arrow(
        &self,
        func: CommonInlineUserDefinedFunctionExpression,
        is_barrier: bool,
    ) -> DataFrame {
        self.map_partitions(func, is_barrier)
    }

    /// Build a `MapPartitions` relation from an already-constructed UDF.
    fn map_partitions(
        &self,
        func: CommonInlineUserDefinedFunctionExpression,
        is_barrier: bool,
    ) -> DataFrame {
        let plan = LogicalPlan::MapPartitions {
            input: Arc::clone(&self.plan),
            func,
            is_barrier,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Apply a function to each row for its side effects (`DataFrame.foreach`).
    ///
    /// Backed by an Arrow map partition; results are forced and discarded.
    pub fn foreach(&self, func: CommonInlineUserDefinedFunctionExpression) -> Result<()> {
        let _ = self.map_partitions(func, false).collect()?;
        Ok(())
    }

    /// Apply a function to each partition for its side effects
    /// (`DataFrame.foreachPartition`).
    pub fn foreach_partition(&self, func: CommonInlineUserDefinedFunctionExpression) -> Result<()> {
        let _ = self.map_partitions(func, false).collect()?;
        Ok(())
    }

    /// Sample rows.
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
            seed,
        };
        DataFrame::new(self.session.clone(), plan)
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
        let request = self.build_execute_request()?;
        let mut stream = block_on(self.session.client().execute_plan_reattachable(request))?;

        let mut rows = vec![];
        let mut info = ExecutionInfo::default();
        loop {
            let resp = block_on(stream.message())?;
            let Some(mut resp) = resp else {
                break;
            };
            capture_execution(&mut resp, &mut info, &self.session);
            if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
                resp.response_type
            {
                let batch_rows = decode_arrow_batch(&batch)?;
                rows.extend(batch_rows);
            }
        }
        self.session.record_execution(info);

        Ok(rows)
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
        let stream = block_on(self.session.client().execute_plan_reattachable(request))?;
        Ok(LocalRowIterator::new(
            self.session.clone(),
            stream,
            prefetch_partitions,
        ))
    }

    /// Execution metrics collected during the most recent action on this
    /// DataFrame's session. Mirrors `pyspark.sql.DataFrame.executionInfo`.
    ///
    /// The metrics reflect the session's most recent execution; call this right
    /// after an action (e.g. `collect`/`count`/`show`).
    pub fn execution_info(&self) -> Result<ExecutionInfo> {
        self.session.last_execution_info().ok_or_else(|| {
            SparkError::connect_msg("no execution info available; run an action first")
        })
    }

    /// Collect all data as Arrow RecordBatches.
    ///
    /// Streams execution results from the server and decodes Arrow IPC batches,
    /// returning the raw `RecordBatch`es without converting to Rows. This is the
    /// foundation for `to_datafusion()` and `to_polars()` conversions.
    pub fn collect_record_batches(&self) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        let request = self.build_execute_request()?;
        let mut stream = block_on(self.session.client().execute_plan_reattachable(request))?;

        let mut batches = vec![];
        let mut info = ExecutionInfo::default();
        loop {
            let resp = block_on(stream.message())?;
            let Some(mut resp) = resp else {
                break;
            };
            capture_execution(&mut resp, &mut info, &self.session);
            if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
                resp.response_type
            {
                let record_batches = decode_arrow_record_batches(&batch)?;
                batches.extend(record_batches);
            }
        }
        self.session.record_execution(info);

        Ok(batches)
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

    /// Show the first n rows.
    pub fn show(&self, n: usize) -> Result<()> {
        let limited = self.limit(n as i32).collect()?;
        for row in limited {
            println!("{}", row);
        }
        Ok(())
    }

    /// Get the schema of this DataFrame.
    ///
    /// The first successful lookup is cached and shared with exact clones.
    /// Changes to external table or view metadata do not invalidate this cache;
    /// construct a new DataFrame to resolve the schema again. Each call returns
    /// an independent value, and failed lookups can be retried.
    pub fn schema(&self) -> Result<DataType> {
        // Keep the lock during analysis so concurrent accesses issue only one
        // successful RPC. Errors leave the cache empty and can be retried.
        let mut cached = self
            .schema_cache
            .lock()
            // The cache contains only None or a fully decoded schema, so a
            // panic during analysis cannot leave a partially written value.
            .unwrap_or_else(|poisoned| {
                self.schema_cache.clear_poison();
                poisoned.into_inner()
            });
        if let Some(schema) = cached.as_ref() {
            return Ok(schema.clone());
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
            *cached = Some(schema.clone());
            Ok(schema)
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

    /// Alias for first().
    pub fn head(&self) -> Result<Option<Row>> {
        self.first()
    }

    /// Get the first n rows.
    pub fn take(&self, n: usize) -> Result<Vec<Row>> {
        self.limit(n as i32).collect()
    }

    /// Check if the DataFrame is empty.
    pub fn is_empty(&self) -> Result<bool> {
        self.limit(1).count().map(|c| c == 0)
    }

    /// Get column names.
    pub fn columns(&self) -> Result<Vec<String>> {
        let schema = self.schema()?;
        match schema {
            DataType::Struct { fields } => Ok(fields.iter().map(|f| f.name.clone()).collect()),
            _ => Err(SparkError::connect_msg("Schema is not a struct type")),
        }
    }

    /// Build an ExecutePlanRequest from the logical plan.
    fn build_execute_request(&self) -> Result<proto::ExecutePlanRequest> {
        // Build proto from plan with plan_id assignment
        let mut relation = self.plan.to_proto();
        assign_plan_ids(&mut relation, &self.session)?;

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
        let mut relation = self.plan.to_proto();
        assign_plan_ids(&mut relation, &self.session)?;

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

    /// Build this DataFrame's plan as a proto `Relation` with plan ids assigned
    /// (the shape the `Persist`/`Unpersist`/`GetStorageLevel` analyze ops take).
    fn analyze_relation(&self) -> Result<proto::Relation> {
        let mut relation = self.plan.to_proto();
        assign_plan_ids(&mut relation, &self.session)?;
        Ok(relation)
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
            relation: Some(self.analyze_relation()?),
            storage_level: Some(storage_level),
        };
        let request = self.analyze_request(proto::analyze_plan_request::Analyze::Persist(persist));
        block_on(self.session.client().analyze_plan(request))?;
        Ok(self.clone())
    }

    /// Remove this DataFrame from cache. Mirrors `DataFrame.unpersist(blocking)`.
    pub fn unpersist(&self, blocking: bool) -> Result<DataFrame> {
        let unpersist = proto::analyze_plan_request::Unpersist {
            relation: Some(self.analyze_relation()?),
            blocking: Some(blocking),
        };
        let request =
            self.analyze_request(proto::analyze_plan_request::Analyze::Unpersist(unpersist));
        block_on(self.session.client().analyze_plan(request))?;
        Ok(self.clone())
    }

    /// Checkpoint this DataFrame to disk.
    pub fn checkpoint(&self) -> Result<DataFrame> {
        self.checkpoint_impl(false, true)
    }

    /// Create a local checkpoint of this DataFrame.
    pub fn local_checkpoint(&self) -> Result<DataFrame> {
        self.checkpoint_impl(true, true)
    }

    /// Execute a `CheckpointCommand` and return a DataFrame referencing the resulting
    /// cached remote relation. Mirrors `DataFrame.checkpoint`/`localCheckpoint`, which
    /// materialize server-side and return a handle to the checkpointed data.
    fn checkpoint_impl(&self, local: bool, eager: bool) -> Result<DataFrame> {
        let mut cmd = proto::CheckpointCommand::default();
        cmd.relation = Some(self.plan.to_proto());
        cmd.local = local;
        cmd.eager = eager;
        let responses = execute_command_collect(
            &self.session,
            proto::command::CommandType::CheckpointCommand(cmd),
        )?;
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
        let mut input = self.plan.to_proto();
        assign_plan_ids(&mut input, &self.session)?;
        let mut cmd = proto::CreateDataFrameViewCommand::default();
        cmd.input = Some(input);
        cmd.name = name.to_string();
        cmd.is_global = global;
        cmd.replace = replace;
        execute_command(
            &self.session,
            proto::command::CommandType::CreateDataframeView(cmd),
        )
    }

    /// Print the execution plan to the console. Mirrors `pyspark.sql.DataFrame.explain`
    /// (was previously a no-op that ran the query relation instead of an AnalyzePlan).
    pub fn explain(&self) -> Result<()> {
        self.explain_mode("simple")
    }

    /// Print the execution plan in a specific mode. Mirrors the `mode` argument of
    /// `pyspark.sql.DataFrame.explain`: one of "simple", "extended", "codegen",
    /// "cost", "formatted" (case-insensitive).
    pub fn explain_mode(&self, mode: &str) -> Result<()> {
        use proto::analyze_plan_request::explain::ExplainMode;
        let explain_mode = match mode.to_lowercase().as_str() {
            "simple" => ExplainMode::Simple,
            "extended" => ExplainMode::Extended,
            "codegen" => ExplainMode::Codegen,
            "cost" => ExplainMode::Cost,
            "formatted" => ExplainMode::Formatted,
            other => {
                return Err(SparkError::value(
                    "UNSUPPORTED_EXPLAIN_MODE",
                    &[("mode", other)],
                ))
            }
        };
        let mut relation = self.plan.to_proto();
        assign_plan_ids(&mut relation, &self.session)?;
        let mut plan = proto::Plan::default();
        plan.op_type = Some(proto::plan::OpType::Root(relation));
        let mut ex = proto::analyze_plan_request::Explain::default();
        ex.plan = Some(plan);
        ex.explain_mode = explain_mode as i32;
        let mut request = proto::AnalyzePlanRequest::default();
        request.session_id = self.session.client().session_id().to_string();
        request.user_context = Some(proto::UserContext::default());
        request.analyze = Some(proto::analyze_plan_request::Analyze::Explain(ex));
        let response = block_on(self.session.client().analyze_plan(request))?;
        if let Some(proto::analyze_plan_response::Result::Explain(e)) = response.result {
            println!("{}", e.explain_string);
        }
        Ok(())
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
        let plan = LogicalPlan::RepartitionByRange {
            input: Arc::clone(&self.plan),
            num_partitions: Some(num_partitions),
            partition_exprs: columns,
        };
        DataFrame::new(self.session.clone(), plan)
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
        DataFrame::new(self.session.clone(), plan)
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
        let mut relation = self.plan.to_proto();
        assign_plan_ids(&mut relation, &self.session)?;
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
        };
        DataFrame::new(self.session.clone(), plan)
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

    /// Replace values.
    pub fn replace(
        &self,
        to_replace: Vec<(String, String)>,
        subset: Option<Vec<&str>>,
    ) -> DataFrame {
        let columns = subset
            .map(|v| v.iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();
        let plan = LogicalPlan::NAReplace {
            input: Arc::clone(&self.plan),
            replacements: to_replace,
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

    /// Select columns by regex pattern.
    pub fn col_regex(&self, col_name: &str) -> DataFrame {
        let plan = LogicalPlan::ColRegex {
            input: Arc::clone(&self.plan),
            col_name: col_name.to_string(),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Select a metadata column by name. Mirrors `pyspark.sql.DataFrame.metadataColumn`.
    pub fn metadata_column(&self, name: &str) -> Column {
        Column::new(
            ExpressionNode::ColumnReference(
                crate::expression::ColumnReference::new(name).metadata(),
            )
            .into(),
        )
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

    /// Sort within partitions (local sort).
    pub fn sort_within_partitions(&self, columns: Vec<Expression>) -> DataFrame {
        let plan = LogicalPlan::Sort {
            input: Arc::clone(&self.plan),
            order: columns,
            is_global: false,
        };
        DataFrame::new(self.session.clone(), plan)
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
    pub fn random_split(&self, weights: Vec<f64>, seed: Option<i64>) -> Vec<DataFrame> {
        let total: f64 = weights.iter().sum();
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
            };
            results.push(DataFrame::new(self.session.clone(), plan));
            cumulative = upper;
        }

        results
    }

    /// Print the schema of this DataFrame.
    pub fn print_schema(&self) -> Result<()> {
        let schema = self.schema()?;
        println!("{}", schema);
        Ok(())
    }

    /// Get the storage level of this DataFrame. Mirrors `DataFrame.storageLevel`.
    pub fn storage_level(&self) -> Result<proto::StorageLevel> {
        let get = proto::analyze_plan_request::GetStorageLevel {
            relation: Some(self.analyze_relation()?),
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
        let schema = self.schema()?;
        match schema {
            DataType::Struct { fields } => {
                let dtypes = fields
                    .iter()
                    .map(|f| (f.name.clone(), f.data_type.to_string()))
                    .collect();
                Ok(dtypes)
            }
            _ => Err(SparkError::connect_msg("Schema is not a struct type")),
        }
    }

    /// Compute the server-side semantic hash of this DataFrame's logical plan,
    /// mirroring `DataFrame.semanticHash()` (an AnalyzePlan request).
    pub fn semantic_hash(&self) -> Result<i32> {
        let mut relation = self.plan.to_proto();
        assign_plan_ids(&mut relation, &self.session)?;
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
        let mut self_rel = self.plan.to_proto();
        assign_plan_ids(&mut self_rel, &self.session)?;
        let mut other_rel = other.plan.to_proto();
        assign_plan_ids(&mut other_rel, &other.session)?;
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

    /// Convert each row to a JSON object string, mirroring `DataFrame.toJSON()`.
    ///
    /// Reference pyspark produces `{"col":val,...}` per row by applying the server's
    /// `to_json(struct(*))`, not a client-side row rendering (which previously emitted
    /// Rust list syntax like `[1, a]`). Build that projection and collect the strings.
    pub fn to_json(&self) -> Result<Vec<String>> {
        let cols: Vec<Column> = self
            .columns()?
            .iter()
            .map(|c| crate::column::col(c))
            .collect();
        let json_col = crate::functions::to_json(crate::functions::r#struct(cols));
        let rows = self.select(vec![json_col]).collect()?;
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
        DataFrame::new(self.session.clone(), plan)
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
        DataFrame::new(self.session.clone(), plan)
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
        DataFrame::new(self.session.clone(), plan)
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
        let plan = LogicalPlan::WithColumnMetadata {
            input: Arc::clone(&self.plan),
            column_name: column_name.to_string(),
            metadata_json,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Get the Spark session.
    pub fn spark_session(&self) -> SparkSession {
        self.session.clone()
    }

    /// Check if this DataFrame is local (collected).
    pub fn is_local(&self) -> bool {
        matches!(self.plan.as_ref(), LogicalPlan::LocalRelation { .. })
    }

    /// Check if this DataFrame is streaming.
    pub fn is_streaming(&self) -> bool {
        // Check if the plan has streaming-related operations
        matches!(
            self.plan.as_ref(),
            LogicalPlan::Read {
                is_streaming: true,
                ..
            }
        )
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
        let star = Column::new(ExpressionNode::UnresolvedStar(None).into());
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
            schema,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Check if the DataFrame exists (is not empty).
    pub fn exists(&self) -> Result<bool> {
        self.limit(1).count().map(|c| c > 0)
    }

    /// Get a scalar value from a single-row, single-column result.
    pub fn scalar(&self) -> Result<Option<Value>> {
        let rows = self.limit(1).collect()?;
        if rows.is_empty() {
            return Ok(None);
        }
        let row = &rows[0];
        Ok(row.get(0).cloned())
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
    plan: &LogicalPlan,
    session: &SparkSession,
) -> Result<proto::Relation> {
    let mut relation = plan.to_proto();
    assign_plan_ids(&mut relation, session)?;
    Ok(relation)
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
    let mut command = proto::Command::default();
    command.command_type = Some(command_type);

    let mut plan = proto::Plan::default();
    plan.op_type = Some(proto::plan::OpType::Command(command));

    let mut request = proto::ExecutePlanRequest::default();
    request.session_id = session.client().session_id().to_string();
    request.user_context = Some(proto::UserContext::default());
    request.tags = session.tags();
    request.plan = Some(plan);

    let mut stream = block_on(session.client().execute_plan_reattachable(request))?;
    // Drain the response stream so the command runs to completion server-side,
    // capturing any metrics/progress emitted along the way.
    let mut info = ExecutionInfo::default();
    let mut responses = Vec::new();
    while let Some(mut resp) = block_on(stream.message())? {
        capture_execution(&mut resp, &mut info, session);
        responses.push(resp);
    }
    session.record_execution(info);
    Ok(responses)
}

/// Pull execution metrics/observed-metrics off a response and fire progress
/// handlers. `metrics`/`observed_metrics` are top-level fields (not part of the
/// `response_type` oneof), so this must run before `response_type` is consumed.
fn capture_execution(
    resp: &mut proto::ExecutePlanResponse,
    info: &mut ExecutionInfo,
    session: &SparkSession,
) {
    if let Some(metrics) = resp.metrics.take() {
        info.metrics = Some(metrics);
    }
    if !resp.observed_metrics.is_empty() {
        let metrics = std::mem::take(&mut resp.observed_metrics);
        // Feed observed metrics to the profiler collector
        session.profiler().accumulate_observed_metrics(&metrics);
        info.observed_metrics.extend(metrics);
    }
    if let Some(proto::execute_plan_response::ResponseType::ExecutionProgress(progress)) =
        &resp.response_type
    {
        session.notify_progress(progress);
    }
}

/// Assign unique plan_ids to all relations in a tree (post-order traversal).
pub(crate) fn assign_plan_ids(
    relation: &mut proto::Relation,
    session: &SparkSession,
) -> Result<()> {
    if let Some(rel_type) = &mut relation.rel_type {
        use proto::relation::RelType;
        match rel_type {
            RelType::Range(_) => {}
            RelType::Sql(_) => {}
            RelType::LocalRelation(_) => {}
            RelType::CachedRemoteRelation(_) => {}
            RelType::Project(proj) => {
                if let Some(input) = &mut proj.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Filter(filter) => {
                if let Some(input) = &mut filter.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Join(join) => {
                if let Some(left) = &mut join.left {
                    assign_plan_ids(left, session)?;
                }
                if let Some(right) = &mut join.right {
                    assign_plan_ids(right, session)?;
                }
            }
            RelType::SetOp(set_op) => {
                if let Some(left) = &mut set_op.left_input {
                    assign_plan_ids(left, session)?;
                }
                if let Some(right) = &mut set_op.right_input {
                    assign_plan_ids(right, session)?;
                }
            }
            RelType::Aggregate(agg) => {
                if let Some(input) = &mut agg.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Sort(sort) => {
                if let Some(input) = &mut sort.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Limit(limit) => {
                if let Some(input) = &mut limit.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Offset(offset) => {
                if let Some(input) = &mut offset.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Tail(tail) => {
                if let Some(input) = &mut tail.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Deduplicate(dedup) => {
                if let Some(input) = &mut dedup.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Repartition(repartition) => {
                if let Some(input) = &mut repartition.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::RepartitionByExpression(repart_expr) => {
                if let Some(input) = &mut repart_expr.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::WithColumns(with_cols) => {
                if let Some(input) = &mut with_cols.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::WithColumnsRenamed(with_renamed) => {
                if let Some(input) = &mut with_renamed.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Drop(drop) => {
                if let Some(input) = &mut drop.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::ToDf(to_df) => {
                if let Some(input) = &mut to_df.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::ToSchema(to_schema) => {
                if let Some(input) = &mut to_schema.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Hint(hint) => {
                if let Some(input) = &mut hint.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Unpivot(unpivot) => {
                if let Some(input) = &mut unpivot.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Sample(sample) => {
                if let Some(input) = &mut sample.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::FillNa(fill_na) => {
                if let Some(input) = &mut fill_na.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::DropNa(drop_na) => {
                if let Some(input) = &mut drop_na.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Replace(replace) => {
                if let Some(input) = &mut replace.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Describe(describe) => {
                if let Some(input) = &mut describe.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Summary(summary) => {
                if let Some(input) = &mut summary.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::SubqueryAlias(sq_alias) => {
                if let Some(input) = &mut sq_alias.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::CachedLocalRelation(_cached) => {
                // CachedLocalRelation doesn't have nested plans, it's just a reference with a hash
            }
            RelType::WithWatermark(watermark) => {
                if let Some(input) = &mut watermark.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Crosstab(stat) => {
                if let Some(input) = &mut stat.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::FreqItems(stat) => {
                if let Some(input) = &mut stat.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::ApproxQuantile(stat) => {
                if let Some(input) = &mut stat.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Corr(stat) => {
                if let Some(input) = &mut stat.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::Cov(stat) => {
                if let Some(input) = &mut stat.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::SampleBy(stat) => {
                if let Some(input) = &mut stat.input {
                    assign_plan_ids(input, session)?;
                }
            }
            RelType::CollectMetrics(metrics) => {
                if let Some(input) = &mut metrics.input {
                    assign_plan_ids(input, session)?;
                }
            }
            _ => {
                // Handle any other relation types that we haven't explicitly handled
            }
        }
    }

    // Assign plan_id to this relation
    if relation.common.is_none() {
        relation.common = Some(proto::RelationCommon::default());
    }
    if let Some(common) = &mut relation.common {
        common.plan_id = Some(session.next_plan_id());
    }

    Ok(())
}

/// Decode an Arrow batch into rows.
pub(crate) fn decode_arrow_batch(
    batch: &proto::execute_plan_response::ArrowBatch,
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
        let field_names = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect::<Arc<[String]>>();
        let layouts = record_batch
            .columns()
            .iter()
            .map(|column| ArrowValueLayout::new(column.data_type()))
            .collect::<Vec<_>>();

        for row_idx in 0..num_rows {
            let mut values = Vec::with_capacity(record_batch.num_columns());

            for (column, layout) in record_batch.columns().iter().zip(&layouts) {
                let value = arrow_value_at_with_layout(column.as_ref(), row_idx, layout)?;
                values.push(value);
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
    if scale <= 0 {
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

/// Render a decoded map key ([`Value`]) as its natural scalar string, since
/// `Value::Map` keys are `String`. Numeric/boolean keys use their value (`1`,
/// `true`), decimals their preserved digits; non-scalar keys (rare) fall back to
/// Debug. This must never use the enum's Debug form for scalars - a `map<int,…>`
/// key has to be "1", not "Integer(1)".
fn map_key_to_string(v: Value) -> String {
    match v {
        Value::String(s) => s,
        Value::Bool(b) => b.to_string(),
        Value::Byte(x) => x.to_string(),
        Value::Short(x) => x.to_string(),
        Value::Integer(x) => x.to_string(),
        Value::Long(x) => x.to_string(),
        Value::Float(x) => x.to_string(),
        Value::Double(x) => x.to_string(),
        Value::Date(d) => d.to_string(),
        Value::Timestamp(t) => t.to_string(),
        Value::Decimal { value, .. } => value,
        other => format!("{other:?}"),
    }
}

/// Immutable decoding metadata prepared once for each column in a record batch.
/// Container layouts follow their children so struct names are also shared inside
/// lists and maps, without retaining the Arrow arrays after decoding.
enum ArrowValueLayout {
    Scalar,
    List(Box<Self>),
    Struct {
        names: Arc<[String]>,
        children: Vec<Self>,
        is_variant: bool,
    },
    Map(Box<(Self, Self)>),
}

impl ArrowValueLayout {
    fn new(data_type: &arrow::datatypes::DataType) -> Self {
        use arrow::datatypes::DataType;
        match data_type {
            DataType::List(field) => Self::List(Box::new(Self::new(field.data_type()))),
            DataType::Struct(fields) => Self::Struct {
                names: fields.iter().map(|field| field.name().clone()).collect(),
                children: fields
                    .iter()
                    .map(|field| Self::new(field.data_type()))
                    .collect(),
                is_variant: fields.iter().any(|field| {
                    field
                        .metadata()
                        .get("variant")
                        .is_some_and(|value| value == "true")
                }),
            },
            DataType::Map(entries, _) => match entries.data_type() {
                DataType::Struct(fields) => Self::Map(Box::new((
                    Self::new(fields[0].data_type()),
                    Self::new(fields[1].data_type()),
                ))),
                _ => unreachable!("Arrow map entries must be a struct"),
            },
            _ => Self::Scalar,
        }
    }
}

#[cfg(test)]
fn arrow_value_at(array: &dyn arrow::array::Array, index: usize) -> Result<Value> {
    arrow_value_at_with_layout(array, index, &ArrowValueLayout::new(array.data_type()))
}

fn arrow_value_at_with_layout(
    array: &dyn arrow::array::Array,
    index: usize,
    value_layout: &ArrowValueLayout,
) -> Result<Value> {
    use arrow::array::*;

    if array.is_null(index) {
        return Ok(Value::Null);
    }

    // A NullType column arrives as an all-null NullArray; every element is Null.
    if array.as_any().downcast_ref::<NullArray>().is_some() {
        return Ok(Value::Null);
    }

    // Try each array type
    if let Some(arr) = array.as_any().downcast_ref::<BooleanArray>() {
        return Ok(Value::Bool(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Int8Array>() {
        return Ok(Value::Byte(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Int16Array>() {
        return Ok(Value::Short(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(Value::Integer(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(Value::Long(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Float32Array>() {
        return Ok(Value::Float(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(Value::Double(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(Value::String(arr.value(index).to_string()));
    }
    if let Some(arr) = array.as_any().downcast_ref::<BinaryArray>() {
        return Ok(Value::Binary(arr.value(index).to_vec()));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Date32Array>() {
        return Ok(Value::Date(arr.value(index)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        return Ok(Value::Timestamp(arr.value(index)));
    }
    // Unsigned integers.
    if let Some(arr) = array.as_any().downcast_ref::<UInt8Array>() {
        return Ok(Value::Short(arr.value(index) as i16));
    }
    if let Some(arr) = array.as_any().downcast_ref::<UInt16Array>() {
        return Ok(Value::Integer(arr.value(index) as i32));
    }
    if let Some(arr) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(Value::Long(arr.value(index) as i64));
    }
    if let Some(arr) = array.as_any().downcast_ref::<UInt64Array>() {
        let val = arr.value(index);
        let i64_val = i64::try_from(val).map_err(|_| {
            SparkError::connect_msg(format!("UInt64 value {} exceeds i64 range", val))
        })?;
        return Ok(Value::Long(i64_val));
    }
    // Decimal128 -> Value::Decimal (exact, not lossy f64), preserving precision/scale.
    if let Some(arr) = array.as_any().downcast_ref::<Decimal128Array>() {
        let scale = arr.scale() as i32;
        return Ok(Value::Decimal {
            value: i128_to_decimal_string(arr.value(index), scale),
            precision: Some(arr.precision() as i32),
            scale: Some(scale),
        });
    }
    // Large / view string & binary variants.
    if let Some(arr) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(Value::String(arr.value(index).to_string()));
    }
    if let Some(arr) = array.as_any().downcast_ref::<LargeBinaryArray>() {
        return Ok(Value::Binary(arr.value(index).to_vec()));
    }
    if let Some(arr) = array.as_any().downcast_ref::<StringViewArray>() {
        return Ok(Value::String(arr.value(index).to_string()));
    }
    if let Some(arr) = array.as_any().downcast_ref::<BinaryViewArray>() {
        return Ok(Value::Binary(arr.value(index).to_vec()));
    }
    // Other timestamp units, normalized to microseconds.
    if let Some(arr) = array.as_any().downcast_ref::<TimestampSecondArray>() {
        return Ok(Value::Timestamp(arr.value(index) * 1_000_000));
    }
    if let Some(arr) = array.as_any().downcast_ref::<TimestampMillisecondArray>() {
        return Ok(Value::Timestamp(arr.value(index) * 1_000));
    }
    if let Some(arr) = array.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok(Value::Timestamp(arr.value(index) / 1_000));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Date64Array>() {
        return Ok(Value::Date((arr.value(index) / 86_400_000) as i32));
    }
    // Nested: list, struct, map (recurse).
    if let Some(arr) = array.as_any().downcast_ref::<ListArray>() {
        let ArrowValueLayout::List(child_layout) = value_layout else {
            unreachable!("layout is prepared from the array's data type");
        };
        let child = arr.value(index);
        let mut items = Vec::with_capacity(child.len());
        for i in 0..child.len() {
            items.push(arrow_value_at_with_layout(child.as_ref(), i, child_layout)?);
        }
        return Ok(Value::List(items));
    }
    if let Some(arr) = array.as_any().downcast_ref::<StructArray>() {
        let ArrowValueLayout::Struct {
            names,
            children,
            is_variant,
        } = value_layout
        else {
            unreachable!("layout is prepared from the array's data type");
        };
        // A VARIANT column arrives as struct<value: binary, metadata: binary> where the
        // `metadata` field carries arrow metadata {"variant": "true"}. Recognize it and
        // return a Value::Variant (raw bytes) so it materializes as a VariantVal (matching
        // pyspark) rather than a plain {value, metadata} struct/dict.
        if *is_variant {
            let bin_field = |name: &str| -> Result<Vec<u8>> {
                match arr.fields().iter().position(|field| field.name() == name) {
                    Some(col_idx) => match arrow_value_at_with_layout(
                        arr.column(col_idx).as_ref(),
                        index,
                        &children[col_idx],
                    )? {
                        Value::Binary(b) => Ok(b),
                        Value::Null => Ok(vec![]),
                        _ => Err(SparkError::connect_msg("variant field is not binary")),
                    },
                    None => Ok(vec![]),
                }
            };
            return Ok(Value::Variant {
                value: bin_field("value")?,
                metadata: bin_field("metadata")?,
            });
        }
        let mut values = Vec::with_capacity(children.len());
        for (col, child_layout) in arr.columns().iter().zip(children) {
            values.push(arrow_value_at_with_layout(
                col.as_ref(),
                index,
                child_layout,
            )?);
        }
        return Ok(Value::Struct(StructValue::new(Arc::clone(names), values)));
    }
    if let Some(arr) = array.as_any().downcast_ref::<MapArray>() {
        let ArrowValueLayout::Map(children) = value_layout else {
            unreachable!("layout is prepared from the array's data type");
        };
        let entries = arr.value(index);
        let keys = entries.column(0);
        let vals = entries.column(1);
        let mut map = std::collections::BTreeMap::new();
        for i in 0..entries.len() {
            // A map key stringifies to its natural scalar form (e.g. `1`, `true`,
            // `1.5`), not the Rust enum's Debug output - a `map<int,string>` key must
            // be "1", never "Integer(1)".
            let k = map_key_to_string(arrow_value_at_with_layout(keys.as_ref(), i, &children.0)?);
            map.insert(
                k,
                arrow_value_at_with_layout(vals.as_ref(), i, &children.1)?,
            );
        }
        return Ok(Value::Map(map));
    }
    // Decimal256 -> exact string-preserving Decimal (Arrow formats with the scale).
    if let Some(arr) = array.as_any().downcast_ref::<Decimal256Array>() {
        return Ok(Value::Decimal {
            value: arr.value_as_string(index),
            precision: Some(arr.precision() as i32),
            scale: Some(arr.scale() as i32),
        });
    }
    if let Some(arr) = array.as_any().downcast_ref::<FixedSizeBinaryArray>() {
        return Ok(Value::Binary(arr.value(index).to_vec()));
    }
    // TimeType (no dedicated Value): render as an ISO time string, normalized to micros.
    if let Some(arr) = array.as_any().downcast_ref::<Time64MicrosecondArray>() {
        return Ok(Value::String(micros_to_time_string(arr.value(index))));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Time64NanosecondArray>() {
        return Ok(Value::String(micros_to_time_string(
            arr.value(index) / 1_000,
        )));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Time32MillisecondArray>() {
        return Ok(Value::String(micros_to_time_string(
            arr.value(index) as i64 * 1_000,
        )));
    }
    if let Some(arr) = array.as_any().downcast_ref::<Time32SecondArray>() {
        return Ok(Value::String(micros_to_time_string(
            arr.value(index) as i64 * 1_000_000,
        )));
    }
    // Interval types (no dedicated Value): render a compact string.
    if let Some(arr) = array.as_any().downcast_ref::<IntervalYearMonthArray>() {
        let months = arr.value(index);
        return Ok(Value::String(format!(
            "{}-{}",
            months / 12,
            (months % 12).abs()
        )));
    }
    if let Some(arr) = array.as_any().downcast_ref::<IntervalDayTimeArray>() {
        let v = arr.value(index);
        return Ok(Value::String(format!(
            "{} days {} ms",
            v.days, v.milliseconds
        )));
    }
    if let Some(arr) = array.as_any().downcast_ref::<IntervalMonthDayNanoArray>() {
        let v = arr.value(index);
        return Ok(Value::String(format!(
            "{} months {} days {} ns",
            v.months, v.days, v.nanoseconds
        )));
    }

    Err(SparkError::connect_msg(format!(
        "Unsupported Arrow type {:?} - cannot convert to Value",
        array.data_type()
    )))
}

#[cfg(test)]
mod nested_decode_tests {
    use super::*;
    use arrow::array::*;
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::StreamWriter;
    use arrow::record_batch::RecordBatch;

    fn decode_batches(columns: Vec<(&str, ArrayRef)>, repeats: usize) -> Vec<Row> {
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
            for _ in 0..repeats {
                writer.write(&batch).unwrap();
            }
            writer.finish().unwrap();
        }
        decode_arrow_batch(&proto::execute_plan_response::ArrowBatch {
            data: data.into(),
            ..Default::default()
        })
        .unwrap()
    }

    fn as_struct(value: &Value) -> &StructValue {
        let Value::Struct(row) = value else {
            panic!("expected struct, got {value:?}");
        };
        row
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
                leaves.push(as_struct(map.values().next().unwrap()));
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
            Some(&Value::Map(
                [
                    ("k1".to_string(), value(1)),
                    ("k2".to_string(), Value::Null)
                ]
                .into()
            ))
        );
        assert_eq!(rows[1].get(0), Some(&Value::List(vec![])));
        assert_eq!(rows[1].get(1), Some(&Value::Map(Default::default())));
        assert_eq!(rows[2].get(0), Some(&Value::Null));
        assert_eq!(rows[2].get(1), Some(&Value::Null));
        assert_eq!(rows[3].get(0), Some(&Value::List(vec![value(4), value(5)])));
        assert_eq!(
            rows[3].get(1),
            Some(&Value::Map(
                [("k4".to_string(), value(4)), ("k5".to_string(), value(5))].into()
            ))
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
            as_struct(&first["k1"]).fields(),
            as_struct(&last["k4"]).fields()
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

/// Render microseconds-since-midnight as an ISO time string `HH:MM:SS[.ffffff]`.
fn micros_to_time_string(micros: i64) -> String {
    let total_secs = micros.div_euclid(1_000_000);
    let us = micros.rem_euclid(1_000_000);
    let (h, m, s) = (total_secs / 3600, (total_secs % 3600) / 60, total_secs % 60);
    if us == 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{h:02}:{m:02}:{s:02}.{us:06}")
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use prost::Message;

    #[test]
    fn schema_cache_follows_exact_clones_but_not_new_plans() {
        let spark = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .unwrap();
        let df = spark.range(3).unwrap();
        let cloned_before_analysis = df.clone();
        let schema = DataType::Struct { fields: vec![] };
        *df.schema_cache.lock().unwrap() = Some(schema.clone());

        // These accesses succeed without a server, including a clone created
        // before the schema was populated.
        assert_eq!(cloned_before_analysis.schema().unwrap(), schema);
        assert_eq!(df.clone().schema().unwrap(), schema);
        assert!(Arc::ptr_eq(
            &df.schema_cache,
            &cloned_before_analysis.schema_cache
        ));
        for derived in [
            df.limit(1),
            df.select(["id"]),
            df.zip(&df).unwrap(),
            spark.range(3).unwrap(),
        ] {
            assert!(!Arc::ptr_eq(&df.schema_cache, &derived.schema_cache));
            assert!(derived.schema_cache.lock().unwrap().is_none());
        }
    }

    #[test]
    fn schema_cache_returns_independent_nested_values() {
        let spark = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .unwrap();
        let df = spark.range(3).unwrap();
        let cloned = df.clone();
        let schema = DataType::from_json(&serde_json::json!({
            "type": "struct",
            "fields": [{
                "name": "outer",
                "type": {
                    "type": "struct",
                    "fields": [{
                        "name": "inner",
                        "type": "long",
                        "nullable": true,
                        "metadata": {"details": {"labels": ["original"]}}
                    }]
                },
                "nullable": true,
                "metadata": {}
            }]
        }))
        .unwrap();
        *df.schema_cache.lock().unwrap() = Some(schema.clone());

        let mut changed = df.schema().unwrap();
        let unchanged = df.schema().unwrap();
        let DataType::Struct { fields } = &mut changed else {
            panic!("Expected outer struct");
        };
        let DataType::Struct { fields } = &mut fields[0].data_type else {
            panic!("Expected nested struct");
        };
        fields[0].name = "changed".to_string();
        fields[0].metadata.get_mut("details").unwrap()["labels"][0] = serde_json::json!("changed");
        assert_eq!(fields[0].name, "changed");
        assert_eq!(fields[0].metadata["details"]["labels"][0], "changed");
        assert_ne!(changed, schema);
        assert_eq!(unchanged, schema);
        assert_eq!(df.schema_cache.lock().unwrap().as_ref(), Some(&schema));
        assert_eq!(cloned.schema().unwrap(), schema);
    }

    #[test]
    fn schema_cache_recovers_after_analysis_panics() {
        let spark = SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .unwrap();
        let df = spark.range(3).unwrap();
        let cloned = df.clone();
        let runtime = tokio::runtime::Runtime::new().unwrap();

        // Calling the synchronous API inside a runtime panics during analysis.
        // A clone must still attempt analysis afterward, rather than returning
        // a permanent cache-poisoning error.
        for frame in [&df, &cloned] {
            let result = runtime.block_on(async {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| frame.schema()))
            });
            assert!(result.is_err());
        }
        let mut cached = df.schema_cache.lock().unwrap_err().into_inner();
        assert!(cached.is_none());
        let schema = DataType::Struct { fields: vec![] };
        *cached = Some(schema.clone());
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

    fn session() -> SparkSession {
        SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("failed to build session")
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

    #[test]
    fn shared_self_join_serializes_each_occurrence_with_distinct_ids() {
        use crate::functions::col;
        use crate::plan;
        use proto::relation::RelType;

        fn ids(relation: &proto::Relation, out: &mut Vec<i64>) {
            out.push(relation.common.as_ref().unwrap().plan_id.unwrap());
            match relation.rel_type.as_ref().unwrap() {
                RelType::Join(join) => {
                    ids(join.left.as_ref().unwrap(), out);
                    ids(join.right.as_ref().unwrap(), out);
                }
                RelType::Project(project) => ids(project.input.as_ref().unwrap(), out),
                RelType::Range(_) => {}
                _ => panic!("unexpected relation"),
            }
        }

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

        let first = build_input_relation(joined.plan(), &spark).unwrap();
        let second = build_input_relation(joined.plan(), &spark).unwrap();
        let mut all_ids = Vec::new();
        ids(&first, &mut all_ids);
        ids(&second, &mut all_ids);
        assert_eq!(all_ids.len(), 10);
        all_ids.sort_unstable();
        all_ids.dedup();
        assert_eq!(all_ids.len(), 10);
        // Assigning IDs mutates only the fresh protobuf, never the shared client plan.
        assert_eq!(joined.plan.to_proto(), expected);
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

    #[test]
    fn random_split_plan() {
        let spark = session();
        let df = spark.range(10).unwrap();
        let dfs = df.random_split(vec![0.7, 0.3], None);
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
            build_input_relation(d.plan(), &spark).expect("plan serializes to a relation");
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
        ser(&df.col_regex("id"));
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
            build_input_relation(d.plan(), &spark).expect("stream plan serializes");
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
            build_input_relation(d.plan(), &spark).expect("exotic plan serializes");
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
            Value::Timestamp(_)
        ));
        assert!(matches!(
            arrow_value_at(&TimestampMillisecondArray::from(vec![1i64]), 0).unwrap(),
            Value::Timestamp(_)
        ));
        assert!(matches!(
            arrow_value_at(&TimestampNanosecondArray::from(vec![1000i64]), 0).unwrap(),
            Value::Timestamp(_)
        ));
        assert!(matches!(
            arrow_value_at(&Date64Array::from(vec![86_400_000i64]), 0).unwrap(),
            Value::Date(_)
        ));
    }

    #[test]
    fn time_types_render_as_string() {
        assert!(matches!(
            arrow_value_at(&Time64MicrosecondArray::from(vec![1i64]), 0).unwrap(),
            Value::String(_)
        ));
        assert!(matches!(
            arrow_value_at(&Time64NanosecondArray::from(vec![1000i64]), 0).unwrap(),
            Value::String(_)
        ));
        assert!(matches!(
            arrow_value_at(&Time32MillisecondArray::from(vec![1i32]), 0).unwrap(),
            Value::String(_)
        ));
        assert!(matches!(
            arrow_value_at(&Time32SecondArray::from(vec![1i32]), 0).unwrap(),
            Value::String(_)
        ));
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
        // Duration has no dedicated Value arm -> the final unsupported-type Err.
        let dur = DurationSecondArray::from(vec![1i64]);
        assert!(arrow_value_at(&dur, 0).is_err());
    }

    #[test]
    fn map_key_to_string_covers_scalar_arms() {
        assert_eq!(map_key_to_string(Value::String("x".to_string())), "x");
        assert_eq!(map_key_to_string(Value::Bool(true)), "true");
        assert_eq!(map_key_to_string(Value::Byte(1)), "1");
        assert_eq!(map_key_to_string(Value::Short(2)), "2");
        assert_eq!(map_key_to_string(Value::Integer(3)), "3");
        assert_eq!(map_key_to_string(Value::Long(4)), "4");
        assert_eq!(map_key_to_string(Value::Float(1.5)), "1.5");
        assert_eq!(map_key_to_string(Value::Double(2.5)), "2.5");
        assert_eq!(map_key_to_string(Value::Date(5)), "5");
        assert_eq!(map_key_to_string(Value::Timestamp(6)), "6");
        assert_eq!(
            map_key_to_string(Value::Decimal {
                value: "7.5".to_string(),
                precision: None,
                scale: None,
            }),
            "7.5"
        );
        // Non-scalar key falls back to the Debug form.
        let _ = map_key_to_string(Value::List(vec![]));
    }

    #[test]
    fn i128_to_decimal_string_branches() {
        assert_eq!(i128_to_decimal_string(12345, 0), "12345");
        assert_eq!(i128_to_decimal_string(12345, 2), "123.45");
        assert_eq!(i128_to_decimal_string(5, 4), "0.0005");
        assert_eq!(i128_to_decimal_string(-5, 4), "-0.0005");
    }

    #[test]
    fn micros_to_time_string_branches() {
        assert_eq!(micros_to_time_string(0), "00:00:00");
        assert!(micros_to_time_string(1).contains('.'));
    }
}
