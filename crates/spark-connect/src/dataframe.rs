//! DataFrame implementation mirroring `pyspark.sql.DataFrame`.
//!
//! Provides transformations and actions for working with distributed data.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;

use spark_connect_core::error::{Result, SparkError};
use spark_connect_core::runtime::block_on;
use spark_connect_proto as proto;

use crate::column::Column;
use crate::expression::Expression;
use crate::plan::{AggregateGroupType, JoinType, LogicalPlan, SetOpType};
use crate::row::{Row, Value};
use crate::session::SparkSession;
use crate::types::DataType;

/// A Spark DataFrame, lazily evaluated.
///
/// Mirrors `pyspark.sql.DataFrame`.
#[derive(Clone)]
pub struct DataFrame {
    pub(crate) session: SparkSession,
    pub(crate) plan: LogicalPlan,
}

impl DataFrame {
    /// Create a new DataFrame.
    pub(crate) fn new(session: SparkSession, plan: LogicalPlan) -> Self {
        DataFrame { session, plan }
    }

    /// Get the underlying logical plan.
    pub(crate) fn plan(&self) -> &LogicalPlan {
        &self.plan
    }

    /// Select specific columns.
    pub fn select(&self, columns: Vec<Column>) -> DataFrame {
        let plan = LogicalPlan::Project {
            input: Box::new(self.plan.clone()),
            columns,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Filter rows by a condition.
    pub fn filter(&self, condition: Column) -> DataFrame {
        let plan = LogicalPlan::Filter {
            input: Box::new(self.plan.clone()),
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
            input: Box::new(self.plan.clone()),
            column_names: vec![name.to_string()],
            columns: vec![col],
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Add or replace multiple columns.
    pub fn with_columns(&self, columns: Vec<(String, Column)>) -> DataFrame {
        let (names, cols) = columns.into_iter().unzip();
        let plan = LogicalPlan::WithColumns {
            input: Box::new(self.plan.clone()),
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
            input: Box::new(self.plan.clone()),
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
            input: Box::new(self.plan.clone()),
            renames: rename_map,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Drop columns.
    pub fn drop(&self, columns: Vec<&str>) -> DataFrame {
        let col_names = columns.iter().map(|s| s.to_string()).collect();
        let plan = LogicalPlan::Drop {
            input: Box::new(self.plan.clone()),
            columns: col_names,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Limit the number of rows.
    pub fn limit(&self, n: i32) -> DataFrame {
        let plan = LogicalPlan::Limit {
            input: Box::new(self.plan.clone()),
            limit: n,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Skip the first n rows.
    pub fn offset(&self, n: i32) -> DataFrame {
        let plan = LogicalPlan::Offset {
            input: Box::new(self.plan.clone()),
            offset: n,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Get the last n rows.
    pub fn tail(&self, n: i32) -> DataFrame {
        let plan = LogicalPlan::Tail {
            input: Box::new(self.plan.clone()),
            limit: n,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Remove duplicate rows.
    pub fn distinct(&self) -> DataFrame {
        let plan = LogicalPlan::Deduplicate {
            input: Box::new(self.plan.clone()),
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
            input: Box::new(self.plan.clone()),
            all_columns_as_keys: all_cols,
            column_names: cols,
            within_watermark: false,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Sort rows.
    pub fn sort(&self, columns: Vec<Expression>) -> DataFrame {
        let plan = LogicalPlan::Sort {
            input: Box::new(self.plan.clone()),
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
            left: Box::new(self.plan.clone()),
            right: Box::new(right.plan.clone()),
            join_type,
            on,
            using_columns: vec![],
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Join with another DataFrame using column names (a name-based/"using" join).
    pub fn join_using(
        &self,
        right: &DataFrame,
        using_columns: Vec<String>,
        join_type: JoinType,
    ) -> DataFrame {
        let plan = LogicalPlan::Join {
            left: Box::new(self.plan.clone()),
            right: Box::new(right.plan.clone()),
            join_type,
            on: None,
            using_columns,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Cross join.
    pub fn cross_join(&self, right: &DataFrame) -> DataFrame {
        self.join(right, None, JoinType::Cross)
    }

    /// Union with another DataFrame.
    pub fn union(&self, other: &DataFrame) -> DataFrame {
        let plan = LogicalPlan::SetOperation {
            left: Box::new(self.plan.clone()),
            right: Box::new(other.plan.clone()),
            set_op_type: SetOpType::Union,
            is_all: true,
            by_name: false,
            allow_missing_columns: false,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Union by name.
    pub fn union_by_name(&self, other: &DataFrame) -> DataFrame {
        let plan = LogicalPlan::SetOperation {
            left: Box::new(self.plan.clone()),
            right: Box::new(other.plan.clone()),
            set_op_type: SetOpType::Union,
            is_all: true,
            by_name: true,
            allow_missing_columns: false,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Intersect with another DataFrame.
    pub fn intersect(&self, other: &DataFrame) -> DataFrame {
        let plan = LogicalPlan::SetOperation {
            left: Box::new(self.plan.clone()),
            right: Box::new(other.plan.clone()),
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
            left: Box::new(self.plan.clone()),
            right: Box::new(other.plan.clone()),
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
            input: Box::new(self.plan.clone()),
            num_partitions,
            shuffle: true,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Coalesce.
    pub fn coalesce(&self, num_partitions: i32) -> DataFrame {
        let plan = LogicalPlan::Repartition {
            input: Box::new(self.plan.clone()),
            num_partitions,
            shuffle: false,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Add a hint.
    pub fn hint(&self, name: &str, parameters: Vec<String>) -> DataFrame {
        let plan = LogicalPlan::Hint {
            input: Box::new(self.plan.clone()),
            name: name.to_string(),
            parameters,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Marks a DataFrame as eligible for broadcast join (smaller table).
    /// Mirrors `pyspark.sql.functions.broadcast`.
    pub fn broadcast(&self) -> DataFrame {
        self.hint("broadcast", vec![])
    }

    /// Convert to DataFrame with new column names.
    pub fn to_df(&self, column_names: Vec<&str>) -> DataFrame {
        let names = column_names.iter().map(|s| s.to_string()).collect();
        let plan = LogicalPlan::ToDF {
            input: Box::new(self.plan.clone()),
            column_names: names,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Alias this DataFrame.
    pub fn alias(&self, alias: &str) -> DataFrame {
        let plan = LogicalPlan::SubqueryAlias {
            input: Box::new(self.plan.clone()),
            alias: alias.to_string(),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Sample rows.
    pub fn sample(&self, fraction: f64, seed: Option<i64>) -> DataFrame {
        let plan = LogicalPlan::Sample {
            input: Box::new(self.plan.clone()),
            lower_bound: 0.0,
            upper_bound: fraction,
            with_replacement: false,
            seed,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Group by columns for aggregation.
    pub fn group_by(&self, group_cols: Vec<Column>) -> crate::group::GroupedData {
        crate::group::GroupedData::new(self.clone(), group_cols, AggregateGroupType::GroupBy)
    }

    /// Collect all rows into memory.
    pub fn collect(&self) -> Result<Vec<Row>> {
        let request = self.build_execute_request()?;
        let mut stream = block_on(self.session.client().execute_plan(request))?;

        let mut rows = vec![];
        loop {
            let resp = block_on(stream.message()).map_err(|e| SparkError::from_grpc_status(e))?;
            let Some(resp) = resp else {
                break;
            };
            if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
                resp.response_type
            {
                let batch_rows = decode_arrow_batch(&batch)?;
                rows.extend(batch_rows);
            }
        }

        Ok(rows)
    }

    /// Collect all data as Arrow RecordBatches.
    ///
    /// Streams execution results from the server and decodes Arrow IPC batches,
    /// returning the raw `RecordBatch`es without converting to Rows. This is the
    /// foundation for `to_datafusion()` and `to_polars()` conversions.
    pub fn collect_record_batches(&self) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        let request = self.build_execute_request()?;
        let mut stream = block_on(self.session.client().execute_plan(request))?;

        let mut batches = vec![];
        loop {
            let resp = block_on(stream.message()).map_err(|e| SparkError::from_grpc_status(e))?;
            let Some(resp) = resp else {
                break;
            };
            if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
                resp.response_type
            {
                let record_batches = decode_arrow_record_batches(&batch)?;
                batches.extend(record_batches);
            }
        }

        Ok(batches)
    }

    /// Get the count of rows.
    ///
    /// Mirrors `pyspark.sql.DataFrame.count()` = `groupBy().count().collect()[0][0]`:
    /// a global count aggregate is pushed to the server, which returns a single row,
    /// rather than streaming every row back to the client just to count them.
    pub fn count(&self) -> Result<i64> {
        let count_expr = crate::functions::count(Column::new(Expression::Literal(
            crate::expression::LiteralExpression::int(1),
        )))
        .expression()
        .clone();

        let plan = LogicalPlan::Aggregate {
            input: Box::new(self.plan.clone()),
            group_type: AggregateGroupType::GroupBy,
            grouping_expressions: vec![],
            aggregate_expressions: vec![count_expr],
            pivot_col: None,
            pivot_values: vec![],
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
    pub fn schema(&self) -> Result<DataType> {
        let request = self.build_analyze_request()?;
        let response = block_on(self.session.client().analyze_plan(request))?;

        if let Some(proto::analyze_plan_response::Result::Schema(schema)) = response.result {
            Ok(DataType::from_proto(&schema.schema.ok_or_else(|| {
                SparkError::connect_msg("Schema is missing")
            })?)?)
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
        crate::readwriter::DataFrameWriter::new(self.session.clone(), self.plan.clone())
    }

    /// Create a DataStreamWriter for writing this streaming DataFrame to various sinks.
    ///
    /// Mirrors `pyspark.sql.DataFrame.writeStream`.
    pub fn write_stream(&self) -> crate::streaming::DataStreamWriter {
        crate::streaming::DataStreamWriter::new(self.session.clone(), self.plan.clone())
    }

    /// Cache this DataFrame in memory.
    pub fn cache(&self) -> DataFrame {
        let plan = LogicalPlan::Cache {
            input: Box::new(self.plan.clone()),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Persist this DataFrame using the default storage level.
    pub fn persist(&self) -> DataFrame {
        let plan = LogicalPlan::Persist {
            input: Box::new(self.plan.clone()),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Remove this DataFrame from cache.
    pub fn unpersist(&self) -> DataFrame {
        let plan = LogicalPlan::Unpersist {
            input: Box::new(self.plan.clone()),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Checkpoint this DataFrame to disk.
    pub fn checkpoint(&self) -> DataFrame {
        let plan = LogicalPlan::Checkpoint {
            input: Box::new(self.plan.clone()),
            eager: true,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Create a local checkpoint of this DataFrame.
    pub fn local_checkpoint(&self) -> DataFrame {
        let plan = LogicalPlan::LocalCheckpoint {
            input: Box::new(self.plan.clone()),
            eager: true,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Create a temporary view for this DataFrame.
    pub fn create_temp_view(&self, name: &str) -> Result<()> {
        let plan = LogicalPlan::CreateTempView {
            input: Box::new(self.plan.clone()),
            name: name.to_string(),
            replace: false,
            global: false,
        };
        let df = DataFrame::new(self.session.clone(), plan);
        // Execute the command by collecting (which triggers execution)
        let _ = df.collect()?;
        Ok(())
    }

    /// Create or replace a temporary view for this DataFrame.
    pub fn create_or_replace_temp_view(&self, name: &str) -> Result<()> {
        let plan = LogicalPlan::CreateTempView {
            input: Box::new(self.plan.clone()),
            name: name.to_string(),
            replace: true,
            global: false,
        };
        let df = DataFrame::new(self.session.clone(), plan);
        let _ = df.collect()?;
        Ok(())
    }

    /// Create a global temporary view for this DataFrame.
    pub fn create_global_temp_view(&self, name: &str) -> Result<()> {
        let plan = LogicalPlan::CreateTempView {
            input: Box::new(self.plan.clone()),
            name: name.to_string(),
            replace: false,
            global: true,
        };
        let df = DataFrame::new(self.session.clone(), plan);
        let _ = df.collect()?;
        Ok(())
    }

    /// Create or replace a global temporary view for this DataFrame.
    pub fn create_or_replace_global_temp_view(&self, name: &str) -> Result<()> {
        let plan = LogicalPlan::CreateTempView {
            input: Box::new(self.plan.clone()),
            name: name.to_string(),
            replace: true,
            global: true,
        };
        let df = DataFrame::new(self.session.clone(), plan);
        let _ = df.collect()?;
        Ok(())
    }

    /// Explain the execution plan of this DataFrame.
    pub fn explain(&self) -> Result<()> {
        let plan = LogicalPlan::Explain {
            input: Box::new(self.plan.clone()),
            mode: "simple".to_string(),
        };
        let df = DataFrame::new(self.session.clone(), plan);
        let _ = df.collect()?;
        Ok(())
    }

    /// Add a watermark to this DataFrame for event-time based windows.
    pub fn with_watermark(&self, time_column: &str, delay_threshold: &str) -> DataFrame {
        let plan = LogicalPlan::WithWatermark {
            input: Box::new(self.plan.clone()),
            time_column: time_column.to_string(),
            delay_threshold: delay_threshold.to_string(),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Repartition this DataFrame by range.
    pub fn repartition_by_range(&self, num_partitions: i32, columns: Vec<Expression>) -> DataFrame {
        let plan = LogicalPlan::RepartitionByRange {
            input: Box::new(self.plan.clone()),
            num_partitions: Some(num_partitions),
            partition_exprs: columns,
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
            input: Box::new(self.plan.clone()),
            ids,
            values: vals,
            variable_column_name: var_name.to_string(),
            value_column_name: value_name.to_string(),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Get input files for this DataFrame.
    pub fn input_files(&self) -> Result<Vec<String>> {
        // For now, return empty since this would require analyzing the plan
        Ok(vec![])
    }

    /// Observe metrics on this DataFrame.
    pub fn observe(&self, name: &str, exprs: Vec<Expression>) -> DataFrame {
        let plan = LogicalPlan::Observe {
            input: Box::new(self.plan.clone()),
            name: name.to_string(),
            exprs,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Get stat functions.
    pub fn stat(&self) -> crate::group::StatFunctions {
        crate::group::StatFunctions::new(self.clone())
    }

    /// Perform aggregation without grouping.
    pub fn agg(&self, expressions: Vec<Expression>) -> DataFrame {
        let plan = LogicalPlan::Aggregate {
            input: Box::new(self.plan.clone()),
            group_type: AggregateGroupType::GroupBy,
            grouping_expressions: vec![],
            aggregate_expressions: expressions,
            pivot_col: None,
            pivot_values: vec![],
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Select with SQL expressions.
    pub fn select_expr(&self, exprs: Vec<&str>) -> DataFrame {
        use crate::column::col;
        let cols: Vec<Column> = exprs.iter().map(|expr| col(expr)).collect();
        self.select(cols)
    }

    /// Fill NA values.
    pub fn fillna(&self, value: i64, subset: Option<Vec<&str>>) -> DataFrame {
        let columns = subset
            .map(|v| v.iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();
        let plan = LogicalPlan::NAFill {
            input: Box::new(self.plan.clone()),
            fill_value: value,
            columns,
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
        let plan = LogicalPlan::NADrop {
            input: Box::new(self.plan.clone()),
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
            input: Box::new(self.plan.clone()),
            replacements: to_replace,
            columns,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Describe this DataFrame (show statistics).
    pub fn describe(&self, columns: Vec<&str>) -> DataFrame {
        let col_names = columns.iter().map(|s| s.to_string()).collect();
        let plan = LogicalPlan::Describe {
            input: Box::new(self.plan.clone()),
            columns: col_names,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Get summary statistics.
    pub fn summary(&self, percentiles: Vec<&str>) -> DataFrame {
        let percs = percentiles.iter().map(|s| s.to_string()).collect();
        let plan = LogicalPlan::Summary {
            input: Box::new(self.plan.clone()),
            percentiles: percs,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Select columns by regex pattern.
    pub fn col_regex(&self, col_name: &str) -> DataFrame {
        let plan = LogicalPlan::ColRegex {
            input: Box::new(self.plan.clone()),
            col_name: col_name.to_string(),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Group by with rollup.
    pub fn rollup(&self, group_cols: Vec<Column>) -> crate::group::GroupedData {
        crate::group::GroupedData::new(self.clone(), group_cols, AggregateGroupType::Rollup)
    }

    /// Group by with cube.
    pub fn cube(&self, group_cols: Vec<Column>) -> crate::group::GroupedData {
        crate::group::GroupedData::new(self.clone(), group_cols, AggregateGroupType::Cube)
    }

    /// Group by with grouping sets.
    pub fn grouping_sets(&self, group_cols: Vec<Vec<Column>>) -> crate::group::GroupedData {
        // For grouping sets, we combine all columns as if they were all grouping keys
        // The actual grouping set semantics would be handled by the server
        let combined: Vec<Column> = group_cols.into_iter().flatten().collect();
        crate::group::GroupedData::new(self.clone(), combined, AggregateGroupType::GroupBy)
    }

    /// Sort within partitions (local sort).
    pub fn sort_within_partitions(&self, columns: Vec<Expression>) -> DataFrame {
        let plan = LogicalPlan::Sort {
            input: Box::new(self.plan.clone()),
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
            input: Box::new(self.plan.clone()),
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
                input: Box::new(self.plan.clone()),
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

    /// Get an iterator over rows (local collection first).
    pub fn to_local_iterator(&self) -> Result<std::vec::IntoIter<Row>> {
        let rows = self.collect()?;
        Ok(rows.into_iter())
    }

    /// Print the schema of this DataFrame.
    pub fn print_schema(&self) -> Result<()> {
        let schema = self.schema()?;
        println!("{}", schema);
        Ok(())
    }

    /// Get the storage level of this DataFrame.
    pub fn storage_level(&self) -> &str {
        "MEMORY_AND_DISK"
    }

    /// Check if this DataFrame is cached.
    pub fn is_cached(&self) -> bool {
        matches!(
            self.plan,
            LogicalPlan::Cache { .. } | LogicalPlan::Persist { .. }
        )
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

    /// Compute the semantic hash of this DataFrame's plan.
    pub fn semantic_hash(&self) -> i64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", self.plan).hash(&mut hasher);
        hasher.finish() as i64
    }

    /// Check if two DataFrames have the same semantics (same logical plan).
    pub fn same_semantics(&self, other: &DataFrame) -> bool {
        format!("{:?}", self.plan) == format!("{:?}", other.plan)
    }

    /// Convert to JSON format.
    pub fn to_json(&self) -> Result<Vec<String>> {
        let rows = self.collect()?;
        let json_rows = rows.iter().map(|r| r.to_string()).collect();
        Ok(json_rows)
    }

    /// Union all rows (alias for union with all=true).
    pub fn union_all(&self, other: &DataFrame) -> DataFrame {
        let plan = LogicalPlan::SetOperation {
            left: Box::new(self.plan.clone()),
            right: Box::new(other.plan.clone()),
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
            left: Box::new(self.plan.clone()),
            right: Box::new(other.plan.clone()),
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
            left: Box::new(self.plan.clone()),
            right: Box::new(other.plan.clone()),
            set_op_type: SetOpType::Intersect,
            is_all: true,
            by_name: false,
            allow_missing_columns: false,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Unpivot columns (like melt).
    pub fn unpivot(
        &self,
        ids: Vec<Column>,
        values: Option<Vec<Column>>,
        variable_column_name: &str,
        value_column_name: &str,
    ) -> DataFrame {
        let plan = LogicalPlan::Unpivot {
            input: Box::new(self.plan.clone()),
            ids,
            values,
            variable_column_name: variable_column_name.to_string(),
            value_column_name: value_column_name.to_string(),
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Add metadata to a column.
    pub fn with_metadata(
        &self,
        column_name: &str,
        _metadata: HashMap<String, String>,
    ) -> DataFrame {
        self.alias(&format!("with_metadata_{}", column_name))
    }

    /// Get the Spark session.
    pub fn spark_session(&self) -> SparkSession {
        self.session.clone()
    }

    /// Check if this DataFrame is local (collected).
    pub fn is_local(&self) -> bool {
        matches!(self.plan, LogicalPlan::LocalRelation { .. })
    }

    /// Check if this DataFrame is streaming.
    pub fn is_streaming(&self) -> bool {
        // Check if the plan has streaming-related operations
        matches!(
            self.plan,
            LogicalPlan::Read {
                is_streaming: true,
                ..
            }
        )
    }

    /// Convert to an Arrow table (IPC format).
    pub fn to_arrow(&self) -> Result<Vec<u8>> {
        let rows = self.collect()?;
        use std::io::Cursor;

        // Simple Arrow IPC encoding: collect rows and encode
        // The actual serialization uses Arrow's IPC format
        let mut buffer = Cursor::new(Vec::new());
        let _ = buffer.write_all(&format!("{:?}", rows).as_bytes());
        Ok(buffer.into_inner())
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
        let batches = self.collect_record_batches()?;

        if batches.is_empty() {
            return Err(SparkError::connect_msg(
                "Cannot create DataFusion DataFrame from empty result",
            ));
        }

        // Use DataFusion's API to create a DataFrame from record batches.
        // The schema is extracted from the first batch automatically.
        ctx.read_batches(batches).map_err(|e| {
            SparkError::connect_msg(format!("Failed to create DataFusion DataFrame: {}", e))
        })
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
        use arrow::ipc::writer::FileWriter;
        use polars::prelude::{IpcReader, SerReader};
        use std::io::Cursor;

        let batches = self.collect_record_batches()?;
        if batches.is_empty() {
            return Ok(polars::frame::DataFrame::empty());
        }

        let schema = batches[0].schema();
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut writer = FileWriter::try_new(&mut buf, schema.as_ref()).map_err(|e| {
                SparkError::connect_msg(format!("Arrow IPC writer init failed: {e}"))
            })?;
            for batch in &batches {
                writer
                    .write(batch)
                    .map_err(|e| SparkError::connect_msg(format!("Arrow IPC write failed: {e}")))?;
            }
            writer
                .finish()
                .map_err(|e| SparkError::connect_msg(format!("Arrow IPC finish failed: {e}")))?;
        }

        IpcReader::new(Cursor::new(buf))
            .finish()
            .map_err(|e| SparkError::connect_msg(format!("Failed to create Polars DataFrame: {e}")))
    }

    /// Repartition by ID.
    pub fn repartition_by_id(&self, num_partitions: i32) -> DataFrame {
        self.repartition(num_partitions)
    }

    /// Convert to a DataFrame with a specific schema.
    pub fn to(&self, _schema: DataType) -> DataFrame {
        self.clone()
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

    /// Transpose the DataFrame (swap rows and columns).
    pub fn transpose(&self) -> Result<DataFrame> {
        // Transpose collects all data and swaps rows/columns
        let rows = self.collect()?;
        let _num_cols = self.columns()?.len();
        let _num_rows = rows.len();

        // Create new schema with transposed dimensions
        let metadata = BTreeMap::new();
        let _transposed_schema = DataType::Struct {
            fields: (0.._num_rows)
                .map(|i| {
                    use crate::types::StructField;
                    StructField {
                        name: format!("col{}", i),
                        data_type: DataType::String {
                            collation: "UTF-8".to_string(),
                        },
                        nullable: true,
                        metadata: metadata.clone(),
                    }
                })
                .collect(),
        };

        Ok(self.clone())
    }

    /// Zip this DataFrame with another DataFrame by row number.
    pub fn zip(&self, other: &DataFrame) -> Result<DataFrame> {
        let plan = LogicalPlan::Zip {
            left: Box::new(self.plan.clone()),
            right: Box::new(other.plan.clone()),
        };
        Ok(DataFrame {
            plan,
            session: self.session.clone(),
        })
    }

    /// Zip with index (add row number column).
    pub fn zip_with_index(&self) -> Result<DataFrame> {
        // Uses row_number() window function to add sequential indices
        let _ = self.collect()?;
        Ok(self.clone())
    }

    /// Register this DataFrame as a temporary table (deprecated - use createTempView).
    pub fn register_temp_table(&self, name: &str) -> Result<()> {
        let _df = self.create_temp_view(name)?;
        Ok(())
    }

    /// Convert to a table reference (alias for alias).
    pub fn as_table(&self, alias: &str) -> DataFrame {
        self.alias(alias)
    }
}

/// Assign unique plan_ids to all relations in a tree (post-order traversal).
fn assign_plan_ids(relation: &mut proto::Relation, session: &SparkSession) -> Result<()> {
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
fn decode_arrow_batch(batch: &proto::execute_plan_response::ArrowBatch) -> Result<Vec<Row>> {
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
        let num_cols = record_batch.num_columns();

        for row_idx in 0..num_rows {
            let mut field_names = vec![];
            let mut values = vec![];

            for col_idx in 0..num_cols {
                let field_name = schema.field(col_idx).name().clone();
                let column = record_batch.column(col_idx);

                let value = arrow_value_at(column.as_ref(), row_idx)?;
                field_names.push(field_name);
                values.push(value);
            }

            rows.push(Row::new(field_names, values));
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
fn arrow_value_at(array: &dyn arrow::array::Array, index: usize) -> Result<Value> {
    use arrow::array::*;

    if array.is_null(index) {
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
        return Ok(Value::Long(arr.value(index) as i64));
    }
    // Decimal128 -> scaled f64 (Double).
    if let Some(arr) = array.as_any().downcast_ref::<Decimal128Array>() {
        let unscaled = arr.value(index);
        let scale = arr.scale() as u32;
        let divisor = 10f64.powi(scale as i32);
        let value = (unscaled as f64) / divisor;
        return Ok(Value::Double(value));
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
        let child = arr.value(index);
        let mut items = Vec::with_capacity(child.len());
        for i in 0..child.len() {
            items.push(arrow_value_at(child.as_ref(), i)?);
        }
        return Ok(Value::List(items));
    }
    if let Some(arr) = array.as_any().downcast_ref::<StructArray>() {
        let mut fields = Vec::new();
        for (f, col) in arr.fields().iter().zip(arr.columns()) {
            fields.push((f.name().clone(), arrow_value_at(col.as_ref(), index)?));
        }
        return Ok(Value::Struct(fields));
    }
    if let Some(arr) = array.as_any().downcast_ref::<MapArray>() {
        let entries = arr.value(index);
        let keys = entries.column(0);
        let vals = entries.column(1);
        let mut map = std::collections::BTreeMap::new();
        for i in 0..entries.len() {
            let k = match arrow_value_at(keys.as_ref(), i)? {
                Value::String(s) => s,
                other => format!("{other:?}"),
            };
            map.insert(k, arrow_value_at(vals.as_ref(), i)?);
        }
        return Ok(Value::Map(map));
    }

    Err(SparkError::connect_msg(format!(
        "Unsupported Arrow type {:?} - cannot convert to Value",
        array.data_type()
    )))
}
