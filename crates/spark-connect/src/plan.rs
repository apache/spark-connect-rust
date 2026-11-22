//! Logical plan nodes mirroring `pyspark.sql.connect.plan`.
//!
//! Each node builds a `spark.connect.Relation` (or `Command`) protobuf matching
//! the reference PySpark client (see `tests/golden/plans.jsonl`).

use spark_connect_proto as proto;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, Weak};

use spark_connect_core::error::{Result, SparkError};

use crate::column::Column;
use crate::expression::{Expression, ExpressionNode, LiteralExpression};
use crate::types::DataType;
use crate::udf::CommonInlineUserDefinedFunctionExpression;

/// Parameters for the `transformWithState` / `transformWithStateInPandas` operators
/// (mirrors `spark.connect.TransformWithStateInfo`).
#[derive(Debug, Clone)]
pub struct TransformWithStateInfo {
    pub time_mode: String,
    pub event_time_column_name: Option<String>,
    pub output_schema: Option<DataType>,
}

/// A logical plan node, mirroring `pyspark.sql.connect.plan.LogicalPlan`.
///
/// Each variant represents a different relation type that can be executed.
/// Child plans are immutable shared nodes, so adding a transformation does not clone
/// its ancestors. Call `to_proto()` to materialize a fresh protobuf tree without plan
/// IDs; requests give every shared node its stable plan ID (see `PlanIds`).
/// The owner of a relation built outside the core, which a plan of it keeps alive while any
/// DataFrame uses it: such as the Python plan of `pyspark.ml`, which keeps the models whose
/// cached objects on the server the relation refers to, as PySpark Connect's DataFrame keeps its
/// plan.
pub struct RelationOwner(pub Arc<dyn std::any::Any + Send + Sync>);

/// A handle of the same owner, of which the check of the clones of a binding is told (see
/// [`crate::gc::check_clones_with`]).
impl Clone for RelationOwner {
    fn clone(&self) -> Self {
        crate::gc::cloned(Arc::as_ptr(&self.0) as *const () as usize);
        RelationOwner(Arc::clone(&self.0))
    }
}

impl std::fmt::Debug for RelationOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RelationOwner")
    }
}

#[derive(Debug, Clone)]
// `GroupMap` is the largest variant; boxing its fields would change the public fields of the
// variant, and each plan node is shared in an `Arc`, so its size is paid once per node.
#[allow(clippy::large_enum_variant)]
pub enum LogicalPlan {
    /// A range of integers: `range(start, end, step)`.
    Range {
        start: i64,
        end: i64,
        step: i64,
        num_partitions: Option<i32>,
    },
    /// A SQL query: `sql("SELECT ...")`, with optional positional/named parameter bindings.
    Sql {
        query: String,
        pos_args: Vec<Expression>,
        named_args: HashMap<String, Expression>,
    },
    /// A projection (select): `df.select(cols...)`.
    Project {
        input: Arc<LogicalPlan>,
        columns: Vec<Column>,
    },
    /// A filter: `df.filter(condition)`.
    Filter {
        input: Arc<LogicalPlan>,
        condition: Column,
    },
    /// An aggregation: `df.groupBy(...).agg(...)`.
    Aggregate {
        input: Arc<LogicalPlan>,
        group_type: AggregateGroupType,
        grouping_expressions: Vec<Expression>,
        aggregate_expressions: Vec<Expression>,
        pivot_col: Option<Expression>,
        pivot_values: Vec<Expression>,
        /// Explicit grouping sets (each inner Vec is one set of grouping expressions),
        /// used when `group_type == GroupingSets`. Empty otherwise.
        grouping_sets: Vec<Vec<Expression>>,
    },
    /// A join: `df.join(other, on)`.
    Join {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        join_type: JoinType,
        on: Option<Column>,
        using_columns: Vec<String>,
    },
    /// A lateral join (`LATERAL` correlated subquery join).
    LateralJoin {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        join_type: JoinType,
        on: Option<Column>,
    },
    /// A set operation (union, intersect, except).
    SetOperation {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        set_op_type: SetOpType,
        is_all: bool,
        by_name: bool,
        allow_missing_columns: bool,
    },
    /// Limit: `df.limit(n)`.
    Limit { input: Arc<LogicalPlan>, limit: i32 },
    /// Offset: `df.offset(n)`.
    Offset {
        input: Arc<LogicalPlan>,
        offset: i32,
    },
    /// Tail: `df.tail(n)` - returns last n rows.
    Tail { input: Arc<LogicalPlan>, limit: i32 },
    /// ShowString: the table of `df.show(...)`, which the server formats as one string:
    /// `num_rows` rows, values truncated to `truncate` characters (0 for none), vertical
    /// with a line per value.
    ShowString {
        input: Arc<LogicalPlan>,
        num_rows: i32,
        truncate: i32,
        vertical: bool,
    },
    /// HtmlString: the HTML table of `df._repr_html_()`, which the server formats as one
    /// string: `num_rows` rows, values truncated to `truncate` characters (0 for none).
    HtmlString {
        input: Arc<LogicalPlan>,
        num_rows: i32,
        truncate: i32,
    },
    /// A relation built outside the core, as the plans of `pyspark.ml` build theirs
    /// (`pyspark.ml.connect.proto.TransformerRelation`), sent as it is, with the plan ID of the
    /// core, and the owner of the relation, if any, which the plan keeps alive. `depth` is the
    /// number of levels of the relation as a plan, with those of the plans of DataFrames that it
    /// holds (1 for a relation without them), which count for [`MAX_PLAN_DEPTH`]. `observations`
    /// are those of the DataFrames that the relation holds as its child, as PySpark's plans of
    /// `pyspark.ml` collect them (see `LogicalPlan::observations`).
    Raw {
        relation: Box<proto::Relation>,
        owner: Option<RelationOwner>,
        depth: usize,
        observations: Vec<(String, crate::observation::Observation)>,
        /// The holes of the relation, of a pickled DataFrame, whose picklers a request calls when
        /// it builds the plan, and whose messages it fills in, as PySpark's unpickled plan
        /// pickles the functions of its UDFs at each request (see [`crate::udf::SentHole`]).
        /// Not a stable API.
        #[doc(hidden)]
        holes: crate::udf::SentHoles,
    },
    /// Deduplicate: `df.distinct()` or `df.dropDuplicates()`.
    Deduplicate {
        input: Arc<LogicalPlan>,
        all_columns_as_keys: bool,
        column_names: Vec<String>,
        within_watermark: bool,
    },
    /// Sort: `df.sort(cols...)`.
    Sort {
        input: Arc<LogicalPlan>,
        order: Vec<Expression>,
        is_global: bool,
    },
    /// Sample: `df.sample(fraction)` or `df.sample(num_rows)`.
    Sample {
        input: Arc<LogicalPlan>,
        lower_bound: f64,
        upper_bound: f64,
        with_replacement: bool,
        seed: Option<i64>,
        /// Whether the input is sorted, or cached, to be in a deterministic order, as the
        /// splits of `randomSplit` are, so that they are disjoint.
        deterministic_order: bool,
    },
    /// Repartition: `df.repartition(num_partitions)`.
    Repartition {
        input: Arc<LogicalPlan>,
        num_partitions: i32,
        shuffle: bool,
    },
    /// RepartitionByExpression: `df.repartitionByRange()` or similar.
    RepartitionByExpression {
        input: Arc<LogicalPlan>,
        num_partitions: i32,
        expressions: Vec<Expression>,
    },
    /// WithColumns: `df.withColumn(name, col)` or `df.withColumns(...)`.
    WithColumns {
        input: Arc<LogicalPlan>,
        column_names: Vec<String>,
        columns: Vec<Column>,
    },
    /// WithColumnMetadata: set metadata (a JSON map) on a single existing column.
    /// Mirrors `DataFrame.withMetadata` - a `WithColumns` whose one alias re-aliases
    /// the column to itself, carrying the metadata JSON.
    WithColumnMetadata {
        input: Arc<LogicalPlan>,
        column_name: String,
        /// The column as a reference to the input's plan ID, as PySpark's `self[columnName]`.
        column: Expression,
        metadata_json: String,
    },
    /// WithColumnsRenamed: `df.withColumnRenamed(old, new)` or `df.withColumnsRenamed(...)`,
    /// of the renames in their order, in which the server applies them, as PySpark sends them.
    WithColumnsRenamed {
        input: Arc<LogicalPlan>,
        renames: Vec<(String, String)>,
    },
    /// Drop: `df.drop(column_names...)`.
    Drop {
        input: Arc<LogicalPlan>,
        columns: Vec<String>,
        /// The columns to drop as expressions, as PySpark's `Drop` of Columns sends them.
        expressions: Vec<Expression>,
    },
    /// ToDF: `df.toDF(*column_names)`.
    ToDF {
        input: Arc<LogicalPlan>,
        column_names: Vec<String>,
    },
    /// ToSchema: set schema to match the provided type.
    ToSchema {
        input: Arc<LogicalPlan>,
        schema: DataType,
        /// The reader of the schema of a client, which a request calls when it builds the plan,
        /// in place of `schema`, as PySpark's `ToSchema` converts the type object of its schema
        /// then: it gives the bytes of the `spark.connect.DataType` of the schema. Not a stable
        /// API.
        #[doc(hidden)]
        schema_reader: Option<crate::udf::CommandPickler>,
    },
    /// A plan that PySpark's client fails to build when a request sends it, with `error`: the
    /// error is raised by each request of the plan, and the plan is not sent, as PySpark's
    /// `DataFrame.to` of a schema that is not a DataType, and `unionByName` of a flag that a
    /// protobuf bool does not take. `input` is serialized first, and the error is recorded after
    /// it, in the slot of the expressions that a request fails to build (see
    /// [`crate::expression::Expression::unbuildable`]): a request raises the first error of
    /// either kind, in the order in which PySpark builds the plan, its inputs before itself.
    Unbuildable {
        input: Arc<LogicalPlan>,
        error: SparkError,
    },
    /// Hint: `df.hint(name, parameters...)`.
    Hint {
        input: Arc<LogicalPlan>,
        name: String,
        /// The parameters, literal expressions or columns, as PySpark's `lit` of them.
        parameters: Vec<Expression>,
    },
    /// Parse: a DataFrame of one string column parsed into a structured DataFrame, as
    /// PySpark's `Parse` of `spark.read.json`, `csv` and `xml` of a DataFrame.
    Parse {
        input: Arc<LogicalPlan>,
        format: ParseFormat,
        schema: Option<DataType>,
        options: HashMap<String, String>,
    },
    /// Unpivot: `df.unpivot(...)`.
    Unpivot {
        input: Arc<LogicalPlan>,
        ids: Vec<Column>,
        values: Option<Vec<Column>>,
        variable_column_name: String,
        value_column_name: String,
    },
    /// NAFill: `df.fillna(value, subset)`.
    NAFill {
        input: Arc<LogicalPlan>,
        fill_value: crate::row::Value,
        columns: Vec<String>,
    },
    /// NAFill with a per-column value: `df.fillna({col: value, ...})`.
    NAFillColumns {
        input: Arc<LogicalPlan>,
        cols: Vec<String>,
        values: Vec<crate::row::Value>,
    },
    /// NADrop: `df.dropna(how, thresh, subset)`.
    NADrop {
        input: Arc<LogicalPlan>,
        how: String,
        min_non_null: Option<i32>,
        columns: Vec<String>,
    },
    /// NAReplace: `df.replace(to_replace, value, subset)`, each replacement a literal and
    /// the literal that replaces it.
    NAReplace {
        input: Arc<LogicalPlan>,
        replacements: Vec<(LiteralExpression, LiteralExpression)>,
        columns: Vec<String>,
    },
    /// Describe: `df.describe(cols...)`.
    Describe {
        input: Arc<LogicalPlan>,
        columns: Vec<String>,
    },
    /// Summary: `df.summary(percentiles...)`.
    Summary {
        input: Arc<LogicalPlan>,
        percentiles: Vec<String>,
    },
    /// ColRegex: column selection by regex.
    ColRegex {
        input: Arc<LogicalPlan>,
        col_name: String,
    },
    /// SubqueryAlias: `df.as(alias)`.
    SubqueryAlias {
        input: Arc<LogicalPlan>,
        alias: String,
    },
    /// WithRelations: a plan with the plans that it refers to, such as a SQL query with the
    /// DataFrames that `SparkSession.sql` takes as keyword arguments, each a `SubqueryAlias`
    /// whose alias the query names.
    WithRelations {
        root: Arc<LogicalPlan>,
        references: Vec<Arc<LogicalPlan>>,
    },
    /// LocalRelation: data supplied locally.
    LocalRelation {
        schema: Option<DataType>,
        data: Option<Vec<u8>>,
    },
    /// Local Arrow data uploaded as session-scoped cache artifacts.
    ChunkedCachedLocalRelation {
        data_hashes: Vec<String>,
        schema_hash: Option<String>,
    },
    /// CachedRemoteRelation: a reference to a cached remote relation, such as the result of
    /// `checkpoint()`. The owner, if any, keeps the relation alive on the server while any
    /// plan of it is alive, and removes it when the last one drops, as PySpark's
    /// `CachedRemoteRelation.__del__` does.
    CachedRemoteRelation {
        relation_id: String,
        owner: Option<RelationOwner>,
    },
    /// A relation the server returned, such as the result of `SparkSession::sql`, sent back
    /// as is (pyspark's `CachedRelation`).
    CachedRelation { relation: Box<proto::Relation> },
    /// Read: `spark.read.format(...).load(...)` or `spark.read.table(...)`.
    Read {
        read_type: crate::readwriter::ReadType,
        is_streaming: bool,
    },
    /// RelationChanges: `spark.read.changes(table)` / `spark.readStream.changes(table)`
    /// (CDC read for a named table).
    RelationChanges {
        table_name: String,
        options: std::collections::HashMap<String, String>,
        is_streaming: Option<bool>,
    },
    /// WithWatermark: `df.withWatermark(timeColumn, delayThreshold)`.
    WithWatermark {
        input: Arc<LogicalPlan>,
        time_column: String,
        delay_threshold: String,
    },
    /// RepartitionByRange: `df.repartitionByRange(numPartitions, *cols)`.
    RepartitionByRange {
        input: Arc<LogicalPlan>,
        num_partitions: Option<i32>,
        partition_exprs: Vec<Expression>,
    },
    /// StatCrosstab: `df.stat.crosstab(col1, col2)`.
    StatCrosstab {
        input: Arc<LogicalPlan>,
        col1: String,
        col2: String,
    },
    /// StatFreqItems: `df.stat.freqItems(columns, support)`.
    StatFreqItems {
        input: Arc<LogicalPlan>,
        columns: Vec<String>,
        support: f64,
    },
    /// StatApproxQuantile: `df.stat.approxQuantile(columns, probabilities, relativeError)`.
    StatApproxQuantile {
        input: Arc<LogicalPlan>,
        columns: Vec<String>,
        probabilities: Vec<f64>,
        relative_error: f64,
    },
    /// StatCorr: `df.stat.corr(col1, col2, method)`, of the method that PySpark sends, "pearson".
    StatCorr {
        input: Arc<LogicalPlan>,
        col1: String,
        col2: String,
        method: String,
    },
    /// StatCov: `df.stat.cov(col1, col2)`.
    StatCov {
        input: Arc<LogicalPlan>,
        col1: String,
        col2: String,
    },
    /// StatSampleBy: `df.stat.sampleBy(col, fractions, seed)`.
    StatSampleBy {
        input: Arc<LogicalPlan>,
        col: Expression,
        fractions: Vec<(Expression, f64)>,
        seed: Option<i64>,
    },
    /// Observe: `df.observe(name, exprs)` - collect metrics.
    Observe {
        input: Arc<LogicalPlan>,
        name: String,
        exprs: Vec<Expression>,
        /// The Observation that the metrics that the server observes for this relation set,
        /// as PySpark's `CollectMetrics` of an `Observation`.
        observation: Option<crate::observation::Observation>,
    },
    /// UnresolvedTableValuedFunction: a TVF like explode, inline, range.
    UnresolvedTableValuedFunction {
        name: String,
        arguments: Vec<Expression>,
    },
    /// Zip: `df.zip(other)` - combine two DataFrames column-wise.
    Zip {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
    },
    /// ML Transformation: wraps an MlRelation proto directly. The proto carries the input
    /// inline, so `input` keeps the input plan, and the owners in it, alive; `owner` keeps
    /// the cached model that the relation refers to alive, if any.
    MlTransform {
        ml_relation: proto::Relation,
        input: Arc<LogicalPlan>,
        owner: Option<RelationOwner>,
    },
    /// A catalog operation exposed as a relation (e.g. `listTables`, `listColumns`),
    /// evaluated lazily so `.collect()` returns the server's rows - including zero
    /// rows for an empty database, which must not be an error.
    Catalog { catalog: proto::Catalog },
    /// Transpose: swap rows and columns. `index_columns` optionally names the
    /// column(s) used as the transposed header (empty = server default).
    Transpose {
        input: Arc<LogicalPlan>,
        index_columns: Vec<Expression>,
    },
    /// MapPartitions: `DataFrame.mapInPandas` / `mapInArrow` (also backs foreach), of the id
    /// of the resource profile of its stage, if any.
    MapPartitions {
        input: Arc<LogicalPlan>,
        func: CommonInlineUserDefinedFunctionExpression,
        is_barrier: bool,
        profile_id: Option<i32>,
        /// The id of the resource profile as a request reads it when it builds the plan, after
        /// the function and the flag, in the place of `profile_id`: as PySpark's plan reads
        /// `profile.id`, which creates the profile on the server the first time. It is read as a
        /// command is pickled (see [`crate::udf::CommandPickler`]), of the four bytes of the id
        /// as a little-endian `i32`. Not a stable API.
        #[doc(hidden)]
        profile_id_reader: Option<crate::udf::CommandPickler>,
    },
    /// GroupMap: `GroupedData.applyInPandas` / `applyInArrow`, and the stateful
    /// `transformWithState[InPandas]`.
    GroupMap {
        input: Arc<LogicalPlan>,
        grouping_expressions: Vec<Expression>,
        func: CommonInlineUserDefinedFunctionExpression,
        sorting_expressions: Vec<Expression>,
        // Stateful fields (all default to None/empty for plain applyInPandas/applyInArrow).
        initial_input: Option<Arc<LogicalPlan>>,
        initial_grouping_expressions: Vec<Expression>,
        is_map_groups_with_state: Option<bool>,
        output_mode: Option<String>,
        timeout_conf: Option<String>,
        state_schema: Option<DataType>,
        transform_with_state_info: Option<TransformWithStateInfo>,
    },
    /// ApplyInPandasWithState: `GroupedData.applyInPandasWithState`. The schemas are the
    /// strings that it sends, a DDL string or the JSON of a type, which the server parses.
    ApplyInPandasWithState {
        input: Arc<LogicalPlan>,
        grouping_expressions: Vec<Expression>,
        func: CommonInlineUserDefinedFunctionExpression,
        output_schema: String,
        state_schema: String,
        output_mode: String,
        timeout_conf: String,
    },
    /// CoGroupMap: `GroupedData.cogroup(...).applyInPandas` / `applyInArrow`.
    CoGroupMap {
        input: Arc<LogicalPlan>,
        input_grouping_expressions: Vec<Expression>,
        other: Arc<LogicalPlan>,
        other_grouping_expressions: Vec<Expression>,
        func: CommonInlineUserDefinedFunctionExpression,
    },
    /// AsOfJoin: `DataFrame._joinAsOf`, which pandas-on-Spark's `merge_asof` calls.
    AsOfJoin {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        left_as_of: Column,
        right_as_of: Column,
        on: Option<Column>,
        using_columns: Vec<String>,
        join_type: AsOfJoinType,
        tolerance: Option<Column>,
        allow_exact_matches: bool,
        direction: AsOfJoinDirection,
    },
    /// NearestByJoin: `DataFrame.nearestByJoin`.
    NearestByJoin {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        ranking_expression: Expression,
        num_results: i32,
        join_type: String,
        mode: String,
        direction: String,
    },
    /// CommonInlineUserDefinedTableFunction: a Python UDTF invocation (`functions.udtf`).
    CommonInlineUdtf {
        function_name: String,
        deterministic: bool,
        arguments: Vec<Expression>,
        return_type: Option<crate::types::DataType>,
        eval_type: i32,
        command: Vec<u8>,
        python_ver: String,
        /// The pickling of the command, which a request calls when it builds the plan, in place
        /// of `command`, as PySpark's `PythonUDTF.to_plan` pickles the handler class for each
        /// plan that it builds. One of the return type too
        /// ([`crate::udf::CommandPickler::with_output_type`]) gives the return type of the build,
        /// in place of `return_type`, as `to_plan` converts it then. Not a stable API.
        #[doc(hidden)]
        pickler: Option<crate::udf::CommandPickler>,
    },
}

/// A part of a plan node, of [`LogicalPlan::parts_mut`]: an input, an expression, a UDF, the
/// pickler of a UDTF, the Observation of an `Observe` of its name, or the Observations of a `Raw`.
pub(crate) enum PlanPart<'a> {
    Input(&'a mut Arc<LogicalPlan>),
    Expression(&'a mut Expression),
    Udf(&'a mut CommonInlineUserDefinedFunctionExpression),
    Pickler(&'a mut Option<crate::udf::CommandPickler>),
    Observation(&'a str, &'a mut Option<crate::observation::Observation>),
    Observations(&'a mut Vec<(String, crate::observation::Observation)>),
}

/// Aggregation group type (GROUP BY, ROLLUP, CUBE, PIVOT).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateGroupType {
    GroupBy,
    Rollup,
    Cube,
    Pivot,
    GroupingSets,
}

/// Join type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    LeftOuter,
    RightOuter,
    FullOuter,
    LeftSemi,
    LeftAnti,
    Cross,
}

/// The join type of an as-of join (`DataFrame._joinAsOf`): the server joins as of only an
/// inner or a left outer join.
///
/// PySpark does not check `how`: it sends the string as it is, and the server reads it as any
/// join type, without regard to case or underscores (`"LEFT"`, `"left_outer"`), and fails the
/// analysis for a join type that is not one, or that an as-of join does not have. So a string
/// that is not exactly the name of a variant is kept in `Other`, and is sent as it is.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum AsOfJoinType {
    /// `"inner"`, the default: the rows of the left side that have a match.
    #[default]
    Inner,
    /// `"left"`: every row of the left side.
    LeftOuter,
    /// Any other string, sent as it is, for the server to read or to refuse.
    Other(String),
}

impl AsOfJoinType {
    /// The join type of `how`: `"inner"`, `"left"`, or the string as it is.
    pub fn parse(how: &str) -> Self {
        match how {
            "inner" => AsOfJoinType::Inner,
            "left" => AsOfJoinType::LeftOuter,
            other => AsOfJoinType::Other(other.to_string()),
        }
    }

    /// The string that is sent as the `join_type` of the as-of join.
    pub fn as_str(&self) -> &str {
        match self {
            AsOfJoinType::Inner => "inner",
            AsOfJoinType::LeftOuter => "left",
            AsOfJoinType::Other(how) => how,
        }
    }
}

impl From<&str> for AsOfJoinType {
    fn from(how: &str) -> Self {
        AsOfJoinType::parse(how)
    }
}

/// The direction in which an as-of join (`DataFrame._joinAsOf`) searches for the match of a
/// row.
///
/// PySpark does not check `direction`: it sends the string as it is, and the server reads it
/// without regard to case, and fails the analysis with `AS_OF_JOIN.UNSUPPORTED_DIRECTION` for
/// any other. So a string that is not exactly the name of a variant is kept in `Other`, and is
/// sent as it is.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum AsOfJoinDirection {
    /// `"backward"`, the default: the last row whose as-of value is not greater.
    #[default]
    Backward,
    /// `"forward"`: the first row whose as-of value is not smaller.
    Forward,
    /// `"nearest"`: the row whose as-of value is the closest.
    Nearest,
    /// Any other string, sent as it is, for the server to read or to refuse.
    Other(String),
}

impl AsOfJoinDirection {
    /// The direction of `direction`: `"backward"`, `"forward"`, `"nearest"`, or the string as
    /// it is.
    pub fn parse(direction: &str) -> Self {
        match direction {
            "backward" => AsOfJoinDirection::Backward,
            "forward" => AsOfJoinDirection::Forward,
            "nearest" => AsOfJoinDirection::Nearest,
            other => AsOfJoinDirection::Other(other.to_string()),
        }
    }

    /// The string that is sent as the `direction` of the as-of join.
    pub fn as_str(&self) -> &str {
        match self {
            AsOfJoinDirection::Backward => "backward",
            AsOfJoinDirection::Forward => "forward",
            AsOfJoinDirection::Nearest => "nearest",
            AsOfJoinDirection::Other(direction) => direction,
        }
    }
}

impl From<&str> for AsOfJoinDirection {
    fn from(direction: &str) -> Self {
        AsOfJoinDirection::parse(direction)
    }
}

/// Set operation type (UNION, INTERSECT, EXCEPT).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOpType {
    Union,
    Intersect,
    Except,
}

/// The plan ID of the next new node, of all the sessions of the process: as PySpark's
/// `LogicalPlan._nextPlanId`, one counter, so that a column of a DataFrame of another session
/// names no node of this session's plan. The plans that a client builds outside the core take
/// theirs from it too ([`fresh_plan_id`]).
static NEXT_PLAN_ID: AtomicI64 = AtomicI64::new(1);

/// A new plan ID of the counter of the process, from which the core numbers the nodes of its
/// plans, for a plan that a client builds outside the core, as PySpark's
/// `LogicalPlan._fresh_plan_id()` numbers each plan: the Python client's `LogicalPlan` takes its
/// IDs here, so that the ID of such a plan, which a DataFrame of it keeps as its plan ID
/// ([`crate::dataframe::DataFrame::assign_plan_id`]), is of no other node of any session. Not a
/// stable API.
#[doc(hidden)]
pub fn fresh_plan_id() -> i64 {
    NEXT_PLAN_ID.fetch_add(1, Ordering::Relaxed)
}

/// Plan IDs that stay the same each time a plan node is serialized, as PySpark assigns
/// them once per plan. The server caches resolved plans keyed by the whole relation, so
/// re-executing a DataFrame, or building on one it already resolved, reuses that work.
///
/// A node is identified by its `Arc` allocation. Each entry holds a `Weak` to the node,
/// which keeps the allocation, and so the key, from being reused while the entry exists.
/// Entries of dropped nodes are pruned as the map grows.
#[derive(Debug)]
pub(crate) struct PlanIds {
    nodes: Mutex<PlanIdNodes>,
}

/// The state of `PlanIds`, all guarded by its one lock.
#[derive(Debug)]
struct PlanIdNodes {
    /// Each key's node, its plan ID, and the order in which the key got that ID.
    ids: HashMap<usize, (Weak<LogicalPlan>, i64, u64)>,
    prune_at: usize,
    /// The order of the next key that gets an ID.
    next_order: u64,
}

impl PlanIdNodes {
    /// Give the key of `node` the plan ID `id`, once the entries of the dropped nodes are pruned,
    /// when the map has grown to twice the entries that it had at the last prune: so the map stays
    /// of about the nodes that are alive, however the keys get their IDs, numbered or given.
    fn insert(&mut self, key: usize, node: &Arc<LogicalPlan>, id: i64) {
        if self.ids.len() >= self.prune_at {
            self.ids.retain(|_, (node, _, _)| node.strong_count() > 0);
            self.prune_at = (self.ids.len() * 2).max(PlanIds::MIN_PRUNE_AT);
        }
        let order = self.next_order;
        self.next_order += 1;
        self.ids.insert(key, (Arc::downgrade(node), id, order));
    }
}

impl PlanIds {
    const MIN_PRUNE_AT: usize = 1024;

    pub(crate) fn new() -> Self {
        PlanIds {
            nodes: Mutex::new(PlanIdNodes {
                ids: HashMap::new(),
                prune_at: Self::MIN_PRUNE_AT,
                next_order: 0,
            }),
        }
    }

    /// The plan ID of `node`, assigned the first time it is serialized.
    pub(crate) fn id_of(&self, node: &Arc<LogicalPlan>) -> i64 {
        self.id_of_key(node, Arc::as_ptr(node) as usize)
    }

    /// The plan ID of the relation of `node` within the `WithRelations` that sends it with the
    /// plans of its subqueries, as PySpark's `_root_plan_id`: `id_of(node)` is that of the
    /// `WithRelations`, as PySpark's `_plan_id`. A node is aligned, so its address with the
    /// low bit set is the key of no other node.
    pub(crate) fn root_id_of(&self, node: &Arc<LogicalPlan>) -> i64 {
        self.id_of_key(node, Arc::as_ptr(node) as usize | 1)
    }

    fn id_of_key(&self, node: &Arc<LogicalPlan>, key: usize) -> i64 {
        let mut nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        // The entry's `Weak` keeps its allocation from being freed, so a live node at this
        // address is the node the entry was made for.
        if let Some((_, id, _)) = nodes.ids.get(&key) {
            return *id;
        }
        let id = NEXT_PLAN_ID.fetch_add(1, Ordering::Relaxed);
        nodes.insert(key, node, id);
        id
    }

    /// Give `node` the plan ID `id`, of a relation that this session sent: so a DataFrame of the
    /// relation, such as one that was pickled, keeps the plan ID of the DataFrame it was of. The
    /// counter of the process goes past `id`, so that no node is numbered `id` later: an ID of
    /// the counter, such as that of a plan of the Python client ([`fresh_plan_id`]), is below it
    /// already.
    pub(crate) fn assign(&self, node: &Arc<LogicalPlan>, id: i64) {
        self.assign_key(node, Arc::as_ptr(node) as usize, id);
    }

    /// As [`PlanIds::assign`], the plan ID of the relation of `node` within the `WithRelations`
    /// that sends it with the plans of its subqueries ([`PlanIds::root_id_of`]).
    pub(crate) fn assign_root(&self, node: &Arc<LogicalPlan>, id: i64) {
        self.assign_key(node, Arc::as_ptr(node) as usize | 1, id);
    }

    fn assign_key(&self, node: &Arc<LogicalPlan>, key: usize, id: i64) {
        NEXT_PLAN_ID.fetch_max(id.saturating_add(1), Ordering::Relaxed);
        let mut nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        nodes.insert(key, node, id);
    }

    /// Give each node of `plan` in these plan IDs the plan ID that it has in `from`, those of
    /// another session, as PySpark's plan objects keep their plan IDs in any session that sends
    /// them: so a DataFrame of `plan` in this session, such as a deep copy of a DataFrame into it
    /// or a `DataFrame(df._plan, session)`, sends the plan IDs of the DataFrame, which the
    /// columns of that DataFrame refer to, and a node with subqueries the root plan ID within its
    /// `WithRelations` too. The plans of the subqueries keep the plan IDs of their own sessions,
    /// which they are sent with. Computed without recursing, once for each node that several
    /// nodes share.
    pub(crate) fn assign_ids_of(&self, plan: &Arc<LogicalPlan>, from: &PlanIds) {
        let mut seen = std::collections::HashSet::new();
        let mut pending = vec![plan];
        while let Some(node) = pending.pop() {
            if !seen.insert(Arc::as_ptr(node) as usize) {
                continue;
            }
            self.assign(node, from.id_of(node));
            if has_subquery_references(node) {
                self.assign_root(node, from.root_id_of(node));
            }
            pending.extend(node.inputs());
        }
    }

    /// The live node of the plan ID `id` that was first given it, such as the DataFrame that a
    /// relation that this session sent is of, when it is alive.
    ///
    /// A plan ID is of one node that this session numbered (`id_of`, or `root_id_of` for the
    /// relation within the `WithRelations` that sends it), and of each node that was given it
    /// (`assign`): each DataFrame of a relation that the session sent, such as each unpickled
    /// copy of a DataFrame, or a deep copy of one, is a node of its own with the plan ID of the
    /// DataFrame it was of, which may be alive too. Each copy has Observations of its own
    /// (`DataFrame::deep_copy_with`, `DataFrame::from_sent_relation`), so the nodes of an ID may
    /// differ in them: the first given the ID is taken, which is the DataFrame that the copies
    /// were made of, as long as it is alive, so that an unpickled DataFrame copies the
    /// Observations of the DataFrame that was pickled, as PySpark's unpickled plan has copies of
    /// those of the pickled one.
    ///
    /// This scans every node of the session, once for each DataFrame made of a sent relation,
    /// which costs less than keeping a second map by plan ID in step with the first on every
    /// insert and prune.
    pub(crate) fn node_of(&self, id: i64) -> Option<Arc<LogicalPlan>> {
        let nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        nodes
            .ids
            .values()
            .filter(|(_, node_id, _)| *node_id == id)
            .filter_map(|(node, _, order)| node.upgrade().map(|node| (*order, node)))
            .min_by_key(|(order, _)| *order)
            .map(|(_, node)| {
                // A new reference to the node, which the check of the clones of a binding is
                // told of (see `crate::gc::check_clones_with`).
                crate::gc::cloned(Arc::as_ptr(&node) as usize);
                node
            })
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.nodes.lock().unwrap().ids.len()
    }
}

/// The `Any` type of a placeholder: the relation that stands for the relation of a DataFrame, in a
/// relation that is built outside the core, such as a plan of `pyspark.ml`, which the Python
/// client builds over the placeholder of each DataFrame it takes as input. The core builds the
/// relation of a DataFrame, and the Python client has no messages of it, so a placeholder is of
/// one level, an extension, whose value is the marker of its DataFrame, and, of the placeholder
/// of `DataFrame._plan.plan(session)`, the relation encoded too (see
/// [`standalone_placeholder_relation`]): the Python client never decodes the nesting of a deep
/// DataFrame, which a protobuf decoder of Python limits. The core replaces the placeholder by
/// the relation when the relation or the command that holds it reaches the core: by the relation
/// that it holds, so a standalone placeholder stands for the relation as it was built after its
/// DataFrame is gone too, as the relation of PySpark's `plan(session)` does, or else by the
/// relation of its DataFrame, which must be alive (see [`placeholder_relation`]).
const PLACEHOLDER_TYPE_URL: &str = "type.googleapis.com/spark.connect.rust.DataFramePlaceholder";

/// The placeholders of all the sessions, by the addresses of their plan node and of their
/// session's plan IDs, each with its marker, by which a placeholder of a marker alone finds its
/// DataFrame, and [`child_observations`] the Observations of the DataFrame of any placeholder.
/// Both are held weakly, as `PlanIds` holds its nodes, so an entry goes with its DataFrame, and
/// its address is not reused while the entry exists.
#[derive(Default)]
struct Placeholders {
    entries: HashMap<(usize, usize), Placeholder>,
    prune_at: usize,
}

struct Placeholder {
    marker: i64,
    node: Weak<LogicalPlan>,
    ids: Weak<PlanIds>,
}

static PLACEHOLDERS: LazyLock<Mutex<Placeholders>> =
    LazyLock::new(|| Mutex::new(Placeholders::default()));

/// The marker of the placeholder of the relation of `node`, a DataFrame of the session of `ids`:
/// the same marker each time the DataFrame gives it.
fn placeholder_marker(node: &Arc<LogicalPlan>, ids: &Arc<PlanIds>) -> i64 {
    const MIN_PRUNE_AT: usize = 1024;
    let mut placeholders = PLACEHOLDERS.lock().unwrap_or_else(|e| e.into_inner());
    // The entry's `Weak`s keep both allocations from being freed, so the node and the session
    // of an entry at these addresses are those of this call.
    let key = (Arc::as_ptr(node) as usize, Arc::as_ptr(ids) as usize);
    if let Some(entry) = placeholders.entries.get(&key) {
        return entry.marker;
    }
    if placeholders.entries.len() >= placeholders.prune_at.max(MIN_PRUNE_AT) {
        placeholders
            .entries
            .retain(|_, entry| entry.node.strong_count() > 0 && entry.ids.strong_count() > 0);
        placeholders.prune_at = placeholders.entries.len() * 2;
    }
    let marker = NEXT_PLAN_ID.fetch_add(1, Ordering::Relaxed);
    placeholders.entries.insert(
        key,
        Placeholder {
            marker,
            node: Arc::downgrade(node),
            ids: Arc::downgrade(ids),
        },
    );
    marker
}

/// The node and the plan IDs of the placeholder `marker`, if its DataFrame is still alive.
fn placeholder_target(marker: i64) -> Option<(Arc<LogicalPlan>, Arc<PlanIds>)> {
    let placeholders = PLACEHOLDERS.lock().unwrap_or_else(|e| e.into_inner());
    placeholders
        .entries
        .values()
        .find(|entry| entry.marker == marker)
        .and_then(|entry| Some((entry.node.upgrade()?, entry.ids.upgrade()?)))
        .inspect(|(node, _)| crate::gc::cloned(Arc::as_ptr(node) as usize))
}

/// The placeholder of the relation of `node`, a DataFrame of the session of `ids`, of its marker
/// alone: the relation that a plan that the core builds over the DataFrame, and resolves while
/// the DataFrame is alive, holds in its place, as the binding's ML transforms of a DataFrame of
/// another session do.
pub(crate) fn placeholder_relation(node: &Arc<LogicalPlan>, ids: &Arc<PlanIds>) -> proto::Relation {
    placeholder_of_value(placeholder_marker(node, ids).to_le_bytes().to_vec())
}

/// The placeholder of the relation of `node`, a DataFrame of the session of `ids`, that holds the
/// relation, as `DataFrame._plan.plan(session)` gives it to the Python client: the extension's
/// value is the marker of the DataFrame, the levels of its relation and the encoding of the
/// relation, which is built now, as PySpark's `plan(session)` builds it, with its errors, and
/// pickles the functions of its UDFs then. So it stands for the relation after the DataFrame is
/// gone, its encoding too, while it holds no object of the DataFrame.
pub(crate) fn standalone_placeholder_relation(
    node: &Arc<LogicalPlan>,
    ids: &Arc<PlanIds>,
) -> Result<proto::Relation> {
    use prost::Message;
    let (relation, levels) = relation_with_ids_and_levels(node, ids)?;
    let marker = placeholder_marker(node, ids);
    // The relation is encoded, and dropped, on a stack of its levels, as both recurse.
    let stack = raw_relation_stack(levels);
    let value = stacker::maybe_grow(stack, stack, move || {
        let mut value = Vec::with_capacity(16 + relation.encoded_len());
        value.extend_from_slice(&marker.to_le_bytes());
        value.extend_from_slice(&(levels as u64).to_le_bytes());
        relation.encode(&mut value).map(|()| value)
    })
    .map_err(|error| SparkError::value_msg(error.to_string()))?;
    Ok(placeholder_of_value(value))
}

/// The placeholder relation of the extension value `value`.
fn placeholder_of_value(value: Vec<u8>) -> proto::Relation {
    proto::Relation {
        common: None,
        rel_type: Some(proto::relation::RelType::Extension(prost_types::Any {
            type_url: PLACEHOLDER_TYPE_URL.to_string(),
            value,
        })),
    }
}

/// The relation that the placeholder of the extension value `value` stands for, and its levels:
/// that which it holds, decoded, or else that of its DataFrame, which must be alive. The decoder
/// of the drop-in has no limit of depth (`pyspark-rs` lifts prost's), and the core's own limits
/// a relation of more than 100 levels, which only the Python client makes standalone
/// placeholders of.
fn placeholder_contents(value: &[u8]) -> Result<(proto::Relation, usize)> {
    use prost::Message;
    let marker = <[u8; 8]>::try_from(value.get(..8).unwrap_or_default())
        .map(i64::from_le_bytes)
        .map_err(|_| SparkError::value_msg("a placeholder of no marker"))?;
    if value.len() == 8 {
        let (node, ids) = placeholder_target(marker).ok_or_else(|| {
            SparkError::value_msg("a placeholder of a DataFrame that no longer exists")
        })?;
        return relation_with_ids_and_levels(&node, &ids);
    }
    let malformed = || SparkError::value_msg("a placeholder of no relation");
    let levels = value
        .get(8..16)
        .and_then(|levels| <[u8; 8]>::try_from(levels).ok())
        .and_then(|levels| usize::try_from(u64::from_le_bytes(levels)).ok())
        .ok_or_else(malformed)?;
    let encoded = &value[16..];
    let stack = raw_relation_stack(levels);
    let relation = stacker::maybe_grow(stack, stack, || proto::Relation::decode(encoded))
        .map_err(|_| malformed())?;
    Ok((relation, levels))
}

/// The Observations of the DataFrames whose placeholders `relation`, a relation built in Python, holds
/// under it, as PySpark's `LogicalPlan.observations` collects them: those of the child of a
/// relation (its input; the input of a transform or the dataset of a model summary of
/// `pyspark.ml`; the root of a `WithRelations`), and so on down, and those of both sides of a
/// `Join`, an `AsOfJoin`, a `LateralJoin`, a `NearestByJoin`, a `Zip` and a `SetOperation`, whose
/// right side's override the left side's of the same name, as PySpark's `{**left, **right}` of
/// them. Taken before the placeholders are replaced, which then no longer name their DataFrames.
pub(crate) fn child_observations(
    relation: &proto::Relation,
) -> Vec<(String, crate::observation::Observation)> {
    use proto::relation::RelType;
    let of = |relation: Option<&proto::Relation>| {
        relation.map_or_else(Vec::new, |relation| {
            crate::expression::with_stack_for_a_level(|| child_observations(relation))
        })
    };
    let (left, right) = match &relation.rel_type {
        Some(RelType::Extension(any)) if any.type_url == PLACEHOLDER_TYPE_URL => {
            let target = any
                .value
                .get(..8)
                .and_then(|marker| <[u8; 8]>::try_from(marker).ok())
                .and_then(|marker| placeholder_target(i64::from_le_bytes(marker)));
            return target
                .map(|(node, _)| node.observations())
                .unwrap_or_default();
        }
        Some(RelType::MlRelation(ml)) => match &ml.ml_type {
            Some(proto::ml_relation::MlType::Transform(transform)) => {
                return of(transform.input.as_deref())
            }
            _ => return of(ml.model_summary_dataset.as_deref()),
        },
        Some(RelType::WithRelations(with)) => return of(with.root.as_deref()),
        Some(RelType::Join(join)) => (join.left.as_deref(), join.right.as_deref()),
        Some(RelType::AsOfJoin(join)) => (join.left.as_deref(), join.right.as_deref()),
        Some(RelType::LateralJoin(join)) => (join.left.as_deref(), join.right.as_deref()),
        Some(RelType::NearestByJoin(join)) => (join.left.as_deref(), join.right.as_deref()),
        Some(RelType::Zip(zip)) => (zip.left.as_deref(), zip.right.as_deref()),
        Some(RelType::SetOp(set_op)) => {
            (set_op.left_input.as_deref(), set_op.right_input.as_deref())
        }
        // The child of the other relations is their first input, as PySpark's `GroupMap` and
        // `CoGroupMap` collect the observations of their input only.
        _ => return of(relation_inputs(relation).into_iter().next()),
    };
    let mut observations = of(left);
    for (name, observation) in of(right) {
        match observations.iter_mut().find(|(seen, _)| *seen == name) {
            Some(slot) => slot.1 = observation,
            None => observations.push((name, observation)),
        }
    }
    observations
}

/// The inputs of a relation of each type of Spark Connect that holds relations, in the order of
/// its fields, for [`relation_inputs`] and [`relation_inputs_mut`]: the input of a relation of one
/// input, both sides of a join, a set operation or a zip, the input and the initial input of a
/// `GroupMap`, both inputs of a `CoGroupMap`, and the root and the references of a
/// `WithRelations`. An `MlRelation`, whose fetched arguments are a level of their own, and an
/// extension, whose message the core does not know, are none of them, nor is a relation of no
/// input.
macro_rules! relation_inputs_of {
    ($rel_type:expr, $deref:ident, $iter:ident) => {
        match $rel_type {
            Some(RelType::Project(r)) => vec![r.input.$deref()],
            Some(RelType::Filter(r)) => vec![r.input.$deref()],
            Some(RelType::Join(r)) => vec![r.left.$deref(), r.right.$deref()],
            Some(RelType::SetOp(r)) => vec![r.left_input.$deref(), r.right_input.$deref()],
            Some(RelType::Sort(r)) => vec![r.input.$deref()],
            Some(RelType::Limit(r)) => vec![r.input.$deref()],
            Some(RelType::Aggregate(r)) => vec![r.input.$deref()],
            Some(RelType::Sample(r)) => vec![r.input.$deref()],
            Some(RelType::Offset(r)) => vec![r.input.$deref()],
            Some(RelType::Deduplicate(r)) => vec![r.input.$deref()],
            Some(RelType::SubqueryAlias(r)) => vec![r.input.$deref()],
            Some(RelType::Repartition(r)) => vec![r.input.$deref()],
            Some(RelType::ToDf(r)) => vec![r.input.$deref()],
            Some(RelType::WithColumnsRenamed(r)) => vec![r.input.$deref()],
            Some(RelType::ShowString(r)) => vec![r.input.$deref()],
            Some(RelType::Drop(r)) => vec![r.input.$deref()],
            Some(RelType::Tail(r)) => vec![r.input.$deref()],
            Some(RelType::WithColumns(r)) => vec![r.input.$deref()],
            Some(RelType::Hint(r)) => vec![r.input.$deref()],
            Some(RelType::Unpivot(r)) => vec![r.input.$deref()],
            Some(RelType::ToSchema(r)) => vec![r.input.$deref()],
            Some(RelType::RepartitionByExpression(r)) => vec![r.input.$deref()],
            Some(RelType::MapPartitions(r)) => vec![r.input.$deref()],
            Some(RelType::CollectMetrics(r)) => vec![r.input.$deref()],
            Some(RelType::Parse(r)) => vec![r.input.$deref()],
            Some(RelType::GroupMap(r)) => vec![r.input.$deref(), r.initial_input.$deref()],
            Some(RelType::CoGroupMap(r)) => vec![r.input.$deref(), r.other.$deref()],
            Some(RelType::WithWatermark(r)) => vec![r.input.$deref()],
            Some(RelType::ApplyInPandasWithState(r)) => vec![r.input.$deref()],
            Some(RelType::HtmlString(r)) => vec![r.input.$deref()],
            Some(RelType::AsOfJoin(r)) => vec![r.left.$deref(), r.right.$deref()],
            Some(RelType::WithRelations(r)) => std::iter::once(r.root.$deref())
                .chain(r.references.$iter().map(Some))
                .collect(),
            Some(RelType::Transpose(r)) => vec![r.input.$deref()],
            Some(RelType::LateralJoin(r)) => vec![r.left.$deref(), r.right.$deref()],
            Some(RelType::NearestByJoin(r)) => vec![r.left.$deref(), r.right.$deref()],
            Some(RelType::Zip(r)) => vec![r.left.$deref(), r.right.$deref()],
            Some(RelType::FillNa(r)) => vec![r.input.$deref()],
            Some(RelType::DropNa(r)) => vec![r.input.$deref()],
            Some(RelType::Replace(r)) => vec![r.input.$deref()],
            Some(RelType::Summary(r)) => vec![r.input.$deref()],
            Some(RelType::Crosstab(r)) => vec![r.input.$deref()],
            Some(RelType::Describe(r)) => vec![r.input.$deref()],
            Some(RelType::Cov(r)) => vec![r.input.$deref()],
            Some(RelType::Corr(r)) => vec![r.input.$deref()],
            Some(RelType::ApproxQuantile(r)) => vec![r.input.$deref()],
            Some(RelType::FreqItems(r)) => vec![r.input.$deref()],
            Some(RelType::SampleBy(r)) => vec![r.input.$deref()],
            Some(
                RelType::Read(_)
                | RelType::Sql(_)
                | RelType::LocalRelation(_)
                | RelType::Range(_)
                | RelType::CachedLocalRelation(_)
                | RelType::CachedRemoteRelation(_)
                | RelType::CommonInlineUserDefinedTableFunction(_)
                | RelType::CommonInlineUserDefinedDataSource(_)
                | RelType::UnresolvedTableValuedFunction(_)
                | RelType::ChunkedCachedLocalRelation(_)
                | RelType::RelationChanges(_)
                | RelType::Catalog(_)
                | RelType::MlRelation(_)
                | RelType::Extension(_)
                | RelType::Unknown(_),
            )
            | None => Vec::new(),
        }
    };
}

/// The inputs of `relation` (see [`relation_inputs_of`]).
fn relation_inputs(relation: &proto::Relation) -> Vec<&proto::Relation> {
    use proto::relation::RelType;
    let inputs: Vec<Option<&proto::Relation>> =
        relation_inputs_of!(&relation.rel_type, as_deref, iter);
    inputs.into_iter().flatten().collect()
}

/// The inputs of `relation` (see [`relation_inputs_of`]), to replace the placeholders under them.
fn relation_inputs_mut(relation: &mut proto::Relation) -> Vec<&mut proto::Relation> {
    use proto::relation::RelType;
    let inputs: Vec<Option<&mut proto::Relation>> =
        relation_inputs_of!(&mut relation.rel_type, as_deref_mut, iter_mut);
    inputs.into_iter().flatten().collect()
}

/// A message that a client built outside the core, such as a relation, a plan, a command, an
/// analysis or a request of the Python client, with the type URLs of placeholders in the encoding
/// that it was decoded from, when the core is given that encoding ([`BuiltOutside::decoded`]). The
/// core replaces the placeholders of such a message wherever PySpark's message would hold the
/// relations of their DataFrames (see [`resolve_relation_levels`]). It meets those at the relations
/// of the message as it walks them; it encodes the fields of its relations and its commands, and
/// reads the messages of its extensions, for the others only when the encoding holds a type URL
/// that is not of a placeholder at a relation, as a placeholder in the extension of an expression
/// or in the message of a plugin's extension adds one. A message given as it is (`From`) is read
/// everywhere. Not a stable API.
#[doc(hidden)]
#[derive(Debug)]
pub struct BuiltOutside<M> {
    message: M,
    urls: Option<usize>,
}

impl<M> From<M> for BuiltOutside<M> {
    fn from(message: M) -> Self {
        BuiltOutside {
            message,
            urls: None,
        }
    }
}

impl<M> BuiltOutside<M> {
    /// The message that `decode` gives of `encoded`, the encoding of a message that a client built,
    /// with the type URLs of placeholders that `encoded` holds. The encoding of a message that
    /// holds a placeholder anywhere holds the type URL of the placeholder's `Any` as it is: in its
    /// own bytes, which those of the message that holds it hold in turn, also when that message is
    /// that of an extension, whose bytes are its encoding.
    pub fn decoded<'a, E>(
        encoded: &'a [u8],
        decode: impl FnOnce(&'a [u8]) -> std::result::Result<M, E>,
    ) -> std::result::Result<Self, E> {
        let message = decode(encoded)?;
        Ok(BuiltOutside {
            message,
            urls: Some(placeholder_urls(encoded)),
        })
    }

    /// The message, and the type URLs of placeholders in its encoding, if the core was given it.
    pub(crate) fn into_parts(self) -> (M, Option<usize>) {
        (self.message, self.urls)
    }

    /// The message, as it was given.
    pub(crate) fn message(&self) -> &M {
        &self.message
    }
}

/// A message built outside the core whose placeholders a request of it replaces (see
/// [`BuiltOutside::resolved`]). Not a stable API.
#[doc(hidden)]
pub trait Resolvable {
    /// Replace the placeholders of this message, whose encoding holds `urls` type URLs of
    /// placeholders, or None when the core was not given its encoding.
    fn resolve(&mut self, urls: Option<usize>) -> Result<()>;
}

impl Resolvable for proto::Plan {
    fn resolve(&mut self, urls: Option<usize>) -> Result<()> {
        resolve_plan(self, urls)
    }
}

impl Resolvable for proto::Command {
    fn resolve(&mut self, urls: Option<usize>) -> Result<()> {
        resolve_command(self, urls)
    }
}

impl Resolvable for proto::AnalyzePlanRequest {
    fn resolve(&mut self, urls: Option<usize>) -> Result<()> {
        match self.analyze.as_mut() {
            Some(analyze) => resolve_analysis(analyze, urls),
            None => Ok(()),
        }
    }
}

impl<M: Resolvable> BuiltOutside<M> {
    /// The message with its placeholders replaced by the relations of their DataFrames, as a
    /// request of it replaces them, which then holds none: so its request reads no DataFrame, as
    /// a binding that sends it without its lock, which it replaced them with, needs. Not a stable
    /// API.
    pub fn resolved(self) -> Result<Self> {
        let (mut message, urls) = self.into_parts();
        message.resolve(urls)?;
        Ok(BuiltOutside {
            message,
            urls: Some(0),
        })
    }
}

/// How far a resolution of a message built outside the core looks for its placeholders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Search {
    /// At the relations of the message, which the walk of them meets, counting them and replacing
    /// none: of a message whose encoding holds type URLs of placeholders, to tell whether those
    /// of its relations are all of them.
    Count,
    /// At the relations of the message, replacing them: of a message whose encoding holds no type
    /// URL of a placeholder but those of the placeholders of its relations, so that no field and
    /// no message of an extension of it holds one.
    Relations,
    /// Also in the fields of the relations and the commands built in Python (see
    /// [`resolve_own_fields`]) and in the messages of extensions (see [`resolve_extension`]): of a
    /// message given without its encoding, or whose encoding holds another type URL.
    Everywhere,
}

/// The search of a resolution, and the placeholders that it met at the relations of the message.
struct Resolver {
    search: Search,
    found: usize,
}

impl Resolver {
    fn new(search: Search) -> Resolver {
        Resolver { search, found: 0 }
    }
}

#[cfg(test)]
thread_local! {
    /// The search of the last resolution of a message built outside the core of this thread.
    static LAST_SEARCH: std::cell::Cell<Option<Search>> = const { std::cell::Cell::new(None) };
}

/// Run `resolve`, the resolution of a message built outside the core whose encoding holds `urls`
/// type URLs of placeholders, or None when the core was not given the encoding, of the search that
/// the encoding allows. A message whose encoding holds none, or only those of the placeholders at
/// its relations, which a first walk of them counts, is resolved at its relations: its fields and
/// the messages of its extensions hold no placeholder, so they are neither encoded nor read, as
/// the URL of each placeholder is in the encoding once. Any other is resolved everywhere: its
/// encoding holds a type URL elsewhere, such as in the extension of an expression, in the message
/// of an extension or in a string, which the resolution then tells apart.
fn resolve_with<T>(
    urls: Option<usize>,
    mut resolve: impl FnMut(&mut Resolver) -> Result<T>,
) -> Result<T> {
    let search = match urls {
        None => Search::Everywhere,
        Some(0) => Search::Relations,
        Some(urls) => {
            let mut count = Resolver::new(Search::Count);
            resolve(&mut count)?;
            if count.found == urls {
                Search::Relations
            } else {
                Search::Everywhere
            }
        }
    };
    #[cfg(test)]
    LAST_SEARCH.with(|last| last.set(Some(search)));
    resolve(&mut Resolver::new(search))
}

/// The number of type URLs of placeholders that `encoded` holds. They do not overlap, as no start
/// of the type URL is an end of it.
fn placeholder_urls(encoded: &[u8]) -> usize {
    memchr::memmem::find_iter(encoded, PLACEHOLDER_TYPE_URL.as_bytes()).count()
}

/// Replace the placeholders of `relation`, a relation built in Python, by the relations of their
/// DataFrames, wherever PySpark's plan would hold the relation of such a DataFrame: `relation`
/// itself, the inputs of every relation of Spark Connect under it (see [`relation_inputs_of`]),
/// those of the relations of `pyspark.ml` (an `ml_relation`'s inputs and fetched arguments), and
/// the messages of the extensions of a plugin there (see [`resolve_extension`]), and those of the
/// extensions of the expressions of the relations built in Python, as a plugin's expression may
/// hold the relation of a DataFrame in its message (see [`resolve_own_fields`]), of the search
/// that its encoding, which holds `urls` type URLs of placeholders, allows (see
/// [`resolve_with`]). Each replacement is the relation of a DataFrame of the core, which has no
/// placeholder. The levels that it gives are those of the relation as a plan of them, as
/// PySpark's serializer counts the levels of a plan of `pyspark.ml`, which serializes the plans of
/// its inputs within its own: one for the relation, one more for each relation that holds an
/// input, and those of the plan of the DataFrame of each placeholder under them.
pub(crate) fn resolve_relation_levels(
    relation: &mut proto::Relation,
    urls: Option<usize>,
) -> Result<usize> {
    resolve_with(urls, |resolver| resolve_relation_at(relation, 1, resolver))
}

/// As [`resolve_relation_levels`], of a relation given without its encoding, without its levels.
#[cfg(test)]
pub(crate) fn resolve_relation(relation: &mut proto::Relation) -> Result<()> {
    resolve_relation_levels(relation, None).map(|_| ())
}

/// Resolve `relation`, which is at the level `level` of the relation being resolved, of the search
/// of `resolver`, and the deepest level of its resolved relation.
fn resolve_relation_at(
    relation: &mut proto::Relation,
    level: usize,
    resolver: &mut Resolver,
) -> Result<usize> {
    if let Some(proto::relation::RelType::Extension(any)) = &relation.rel_type {
        if any.type_url == PLACEHOLDER_TYPE_URL {
            resolver.found += 1;
            if resolver.search == Search::Count {
                return Ok(level);
            }
            let (mut resolved, levels) = placeholder_contents(&any.value)?;
            // A `common` that the plan set on the placeholder, as on PySpark's relation of
            // `plan(session)` (`relation.common.plan_id = ...`), merged into the relation's.
            if let Some(common) = &relation.common {
                use prost::Message;
                merge_common(&mut resolved, &common.encode_to_vec())?;
            }
            *relation = resolved;
            // The relation of the DataFrame takes the place of the placeholder, at its level.
            return Ok(level - 1 + levels);
        }
    }
    let everywhere = resolver.search == Search::Everywhere;
    let mut deepest = level;
    match relation.rel_type.as_mut() {
        Some(proto::relation::RelType::MlRelation(ml)) => {
            if let Some(summary) = ml.model_summary_dataset.as_mut() {
                resolve_input_at(summary, level + 1, &mut deepest, resolver)?;
            }
            match ml.ml_type.as_mut() {
                Some(proto::ml_relation::MlType::Transform(transform)) => {
                    if let Some(input) = transform.input.as_mut() {
                        resolve_input_at(input, level + 1, &mut deepest, resolver)?;
                    }
                }
                Some(proto::ml_relation::MlType::Fetch(fetch)) => {
                    deepest = deepest.max(resolve_fetch_at(fetch, level + 1, resolver)?);
                }
                None => {}
            }
        }
        Some(proto::relation::RelType::Extension(extension)) => {
            if everywhere {
                resolve_extension(extension)?;
            }
        }
        // Every input of a relation of Spark Connect, as PySpark's plans hold the relations of
        // their children's DataFrames: those of `pyspark.sql.connect.plan` (a `SubqueryAlias`
        // and the views of a `SQL`, references of a `WithRelations`), of a plugin (Delta Lake's
        // `DeltaTable.alias`) and of the messages that `SparkSession.client` is given; and the
        // extensions of the expressions of the relation (see [`resolve_own_fields`]).
        _ => {
            if everywhere && holds_expressions(relation) {
                resolve_own_fields(relation, relation_inputs_mut)?;
            }
            for input in relation_inputs_mut(relation) {
                resolve_input_at(input, level + 1, &mut deepest, resolver)?;
            }
        }
    }
    Ok(deepest)
}

/// Resolve `input`, an input of a relation of the level `level - 1`, at the level `level`, of the
/// search of `resolver`, keeping in `deepest` the deepest level that it reaches.
fn resolve_input_at(
    input: &mut proto::Relation,
    level: usize,
    deepest: &mut usize,
    resolver: &mut Resolver,
) -> Result<()> {
    // Relations over relations are as deep as the Python plan that builds them.
    let levels =
        crate::expression::with_stack_for_a_level(|| resolve_relation_at(input, level, resolver))?;
    *deepest = (*deepest).max(levels);
    Ok(())
}

/// Whether `relation` holds expressions of its own (or a function, whose arguments are), whose
/// extensions may hold a placeholder, of each type of relation of Spark Connect: so the fields of
/// one that holds none, such as a `Limit` or the data of a `LocalRelation`, are not encoded to look
/// for one, as a literal (of `FillNa` or `Replace`) holds no extension.
fn holds_expressions(relation: &proto::Relation) -> bool {
    use proto::relation::RelType;
    match &relation.rel_type {
        Some(
            RelType::Project(_)
            | RelType::Filter(_)
            | RelType::Join(_)
            | RelType::Sort(_)
            | RelType::Aggregate(_)
            | RelType::Sql(_)
            | RelType::RepartitionByExpression(_)
            | RelType::WithColumns(_)
            | RelType::Hint(_)
            | RelType::Unpivot(_)
            | RelType::Drop(_)
            | RelType::MapPartitions(_)
            | RelType::CollectMetrics(_)
            | RelType::GroupMap(_)
            | RelType::CoGroupMap(_)
            | RelType::ApplyInPandasWithState(_)
            | RelType::CommonInlineUserDefinedTableFunction(_)
            | RelType::UnresolvedTableValuedFunction(_)
            | RelType::AsOfJoin(_)
            | RelType::LateralJoin(_)
            | RelType::NearestByJoin(_)
            | RelType::Transpose(_)
            | RelType::SampleBy(_),
        ) => true,
        Some(
            RelType::Read(_)
            | RelType::SetOp(_)
            | RelType::Limit(_)
            | RelType::Sample(_)
            | RelType::Offset(_)
            | RelType::Deduplicate(_)
            | RelType::SubqueryAlias(_)
            | RelType::Repartition(_)
            | RelType::ToDf(_)
            | RelType::WithColumnsRenamed(_)
            | RelType::ShowString(_)
            | RelType::Tail(_)
            | RelType::ToSchema(_)
            | RelType::Parse(_)
            | RelType::WithWatermark(_)
            | RelType::HtmlString(_)
            | RelType::WithRelations(_)
            | RelType::Zip(_)
            | RelType::FillNa(_)
            | RelType::DropNa(_)
            | RelType::Replace(_)
            | RelType::Summary(_)
            | RelType::Crosstab(_)
            | RelType::Describe(_)
            | RelType::Cov(_)
            | RelType::Corr(_)
            | RelType::ApproxQuantile(_)
            | RelType::FreqItems(_)
            | RelType::LocalRelation(_)
            | RelType::Range(_)
            | RelType::CachedLocalRelation(_)
            | RelType::CachedRemoteRelation(_)
            | RelType::CommonInlineUserDefinedDataSource(_)
            | RelType::ChunkedCachedLocalRelation(_)
            | RelType::RelationChanges(_)
            | RelType::Catalog(_)
            | RelType::MlRelation(_)
            | RelType::Extension(_)
            | RelType::Unknown(_),
        )
        | None => false,
    }
}

/// Merge `common`, the encoding of the `common` that a plan built in Python set on a placeholder,
/// into the `common` of `resolved`, the relation of the placeholder, as protobuf's `MergeFrom`
/// would set its fields on PySpark's relation of `plan(session)`.
fn merge_common(resolved: &mut proto::Relation, common: &[u8]) -> Result<()> {
    use prost::Message;
    resolved
        .common
        .get_or_insert_with(Default::default)
        .merge(common)
        .map_err(|error| SparkError::value_msg(error.to_string()))
}

/// The stack that the encode and the decode of the fields of a relation or a command built in
/// Python keep free, and the stack segment that they add when less is left: prost recurses once per
/// level of their expressions, with frames of a few hundred bytes in a release build and more in a
/// debug one, and the expressions have at most `MAX_EXPRESSION_DEPTH` levels.
const OWN_FIELDS_RED_ZONE: usize = if cfg!(debug_assertions) {
    4 << 20
} else {
    1 << 20
};
const OWN_FIELDS_STACK: usize = if cfg!(debug_assertions) {
    16 << 20
} else {
    4 << 20
};

/// Replace the placeholders in the fields of `message`, a relation or a command built in Python,
/// other than the relations that `inputs` gives of it, which the caller resolves: the extensions of
/// its expressions, of a plugin's messages, whose relations PySpark's plan sends as it built them.
/// The core does not walk the expressions of the message: the fields are encoded with the inputs
/// taken out, and only when that encoding holds the type URL of a placeholder, as an extension
/// that holds one does, are they resolved as any message (see [`resolve_encoded`]) and decoded
/// back. A relation that the core builds, such as that of a placeholder, is not encoded, and nor
/// is a message whose encoding the client gave, which holds no type URL of a placeholder but those
/// of the placeholders at its relations (see [`resolve_with`]).
fn resolve_own_fields<M: prost::Message + Default>(
    message: &mut M,
    inputs: impl Fn(&mut M) -> Vec<&mut proto::Relation>,
) -> Result<()> {
    let taken: Vec<proto::Relation> = inputs(message).into_iter().map(std::mem::take).collect();
    // The fields may nest expressions of many levels, which the encode and the decode recurse
    // once per level of: a stack that holds those of `MAX_EXPRESSION_DEPTH` levels.
    let encoded = stacker::maybe_grow(OWN_FIELDS_RED_ZONE, OWN_FIELDS_STACK, || {
        message.encode_to_vec()
    });
    let resolved = resolve_encoded(&encoded);
    let restore = |message: &mut M, taken: Vec<proto::Relation>| {
        for (input, taken) in inputs(message).into_iter().zip(taken) {
            *input = taken;
        }
    };
    match resolved {
        Ok(Some(bytes)) => match stacker::maybe_grow(OWN_FIELDS_RED_ZONE, OWN_FIELDS_STACK, || {
            M::decode(bytes.as_slice())
        }) {
            Ok(decoded) => {
                *message = decoded;
                restore(message, taken);
                Ok(())
            }
            Err(error) => {
                restore(message, taken);
                Err(SparkError::value_msg(error.to_string()))
            }
        },
        other => {
            restore(message, taken);
            other.map(|_| ())
        }
    }
}

/// Replace the placeholders that `extension`, the `Any` of the extension of a relation or of a
/// command, holds. Its message is a plugin's relation or command, which the plugin's Python client
/// builds over the plans of the DataFrames that it takes, as Delta Lake's `merge` builds its
/// `MergeIntoTable` over the plan of its source DataFrame. The core does not know the type of the
/// message, so it reads its encoding as that of any message (see [`resolve_encoded`]). The message
/// adds no level to the plan that holds it, as PySpark's serializer counts them: the client
/// serializes it on its own, into the bytes of the `Any` (`Any.Pack`).
fn resolve_extension(extension: &mut prost_types::Any) -> Result<()> {
    if let Some(value) = resolve_encoded(&extension.value)? {
        extension.value = value;
    }
    Ok(())
}

/// `encoded`, the encoding of a message of a type that the core need not know, with the
/// placeholders that it holds replaced by the relations of their DataFrames; None when it holds
/// none, as an encoding that does not hold the type URL of a placeholder does not. An encoding of
/// a placeholder, as `plan(session)` gives it or with a `common` that the plan set on it, is
/// replaced by that of the relation of its DataFrame, with that `common` merged; an encoding of
/// fields is read as a message: the value of each field of it that may be a message (of the wire
/// type of length-delimited values) and holds the type URL is resolved in turn, and the other
/// fields stay as they are. An encoding that does not read as a message, such as that of a string
/// that holds the type URL, holds no placeholder.
///
/// It takes a time of the length of the encoding, however deep its messages nest: the offsets of
/// the type URL are found once, a field is read only when its value holds one of them, and the
/// resolved encoding is written once, of the lengths of its changed fields, computed first.
fn resolve_encoded(encoded: &[u8]) -> Result<Option<Vec<u8>>> {
    let offsets = url_offsets(encoded);
    if offsets.is_empty() {
        return Ok(None);
    }
    let Some(change) = resolve_change(encoded, 0, &offsets)? else {
        return Ok(None);
    };
    let mut resolved = Vec::with_capacity(change.len());
    change.write(encoded, &mut resolved);
    Ok(Some(resolved))
}

/// The offsets in `encoded` at which the type URL of a placeholder starts, in order (see
/// [`placeholder_urls`]).
fn url_offsets(encoded: &[u8]) -> Vec<usize> {
    memchr::memmem::find_iter(encoded, PLACEHOLDER_TYPE_URL.as_bytes()).collect()
}

/// Whether the bytes `start..end` of the encoding of `offsets` hold the type URL of a placeholder
/// whole.
fn holds_url(offsets: &[usize], start: usize, end: usize) -> bool {
    let first = offsets.partition_point(|&offset| offset < start);
    offsets
        .get(first)
        .is_some_and(|&offset| offset + PLACEHOLDER_TYPE_URL.len() <= end)
}

/// The change of an encoding that [`resolve_encoded`] makes.
enum Change {
    /// The encoding of a placeholder, replaced by that of the relation of its DataFrame.
    Replaced(Vec<u8>),
    /// The fields of a message whose values change, in order, and the length of the encoding with
    /// the changes.
    Fields(Vec<ChangedField>, usize),
}

/// A field of a message whose value changes: the offsets in the message of the end of its key, the
/// start of its value and its end, and the change of its value.
struct ChangedField {
    key_end: usize,
    value_start: usize,
    end: usize,
    change: Change,
}

impl Change {
    /// The length of the changed encoding.
    fn len(&self) -> usize {
        match self {
            Change::Replaced(bytes) => bytes.len(),
            Change::Fields(_, len) => *len,
        }
    }

    /// Write the changed encoding of `encoded`, the encoding that changes, into `out`.
    fn write(&self, encoded: &[u8], out: &mut Vec<u8>) {
        match self {
            Change::Replaced(bytes) => out.extend_from_slice(bytes),
            Change::Fields(fields, _) => {
                let mut at = 0;
                for field in fields {
                    out.extend_from_slice(&encoded[at..field.key_end]);
                    prost::encoding::encode_varint(field.change.len() as u64, out);
                    let value = &encoded[field.value_start..field.end];
                    crate::expression::with_stack_for_a_level(|| field.change.write(value, out));
                    at = field.end;
                }
                out.extend_from_slice(&encoded[at..]);
            }
        }
    }
}

/// The change of `encoded`, the bytes at `base` of the encoding of `offsets`, as
/// [`resolve_encoded`] resolves it, or None when it holds no placeholder.
fn resolve_change(encoded: &[u8], base: usize, offsets: &[usize]) -> Result<Option<Change>> {
    use prost::encoding::{decode_varint, encoded_len_varint};

    if !holds_url(offsets, base, base + encoded.len()) {
        return Ok(None);
    }
    if let Some((common, value)) = placeholder_parts(encoded) {
        let (mut relation, levels) = placeholder_contents(value)?;
        if let Some(common) = common {
            merge_common(&mut relation, common)?;
        }
        let stack = raw_relation_stack(levels);
        let bytes = stacker::maybe_grow(stack, stack, move || {
            use prost::Message;
            relation.encode_to_vec()
        });
        return Ok(Some(Change::Replaced(bytes)));
    }
    let mut fields = Vec::new();
    let mut len = encoded.len();
    let mut rest = encoded;
    while !rest.is_empty() {
        let Ok(key) = decode_varint(&mut rest) else {
            return Ok(None);
        };
        let key_end = encoded.len() - rest.len();
        let wire_type = key & 7;
        let value_len = match wire_type {
            0 => match decode_varint(&mut rest) {
                Ok(_) => 0,
                Err(_) => return Ok(None),
            },
            1 => 8,
            2 => match decode_varint(&mut rest).map(usize::try_from) {
                Ok(Ok(len)) => len,
                _ => return Ok(None),
            },
            5 => 4,
            // A group, which a proto3 file has none of, is not read.
            _ => return Ok(None),
        };
        if key >> 3 == 0 || value_len > rest.len() {
            return Ok(None);
        }
        let value_start = encoded.len() - rest.len();
        rest = &rest[value_len..];
        let end = value_start + value_len;
        if wire_type == 2 && holds_url(offsets, base + value_start, base + end) {
            let value = &encoded[value_start..end];
            let below = crate::expression::with_stack_for_a_level(|| {
                resolve_change(value, base + value_start, offsets)
            })?;
            if let Some(change) = below {
                len =
                    len - (end - key_end) + encoded_len_varint(change.len() as u64) + change.len();
                fields.push(ChangedField {
                    key_end,
                    value_start,
                    end,
                    change,
                });
            }
        }
    }
    Ok((!fields.is_empty()).then_some(Change::Fields(fields, len)))
}

/// The `common` and the value of the extension of `encoded`, when it is the encoding of a
/// placeholder: a relation of no other field than an extension of the type URL of a placeholder,
/// as `plan(session)` gives it, and a `common`, which a plan may set on it, each once. Read of
/// the keys of the fields alone, so that an encoding that is not one costs no decoding.
fn placeholder_parts(encoded: &[u8]) -> Option<(Option<&[u8]>, &[u8])> {
    let fields = top_level_fields(encoded)?;
    let (mut common, mut extension) = (None, None);
    for (number, value) in fields {
        let slot = match number {
            COMMON_FIELD => &mut common,
            EXTENSION_FIELD => &mut extension,
            _ => return None,
        };
        if slot.replace(value).is_some() {
            return None;
        }
    }
    let (mut type_url, mut value) = (None, None);
    for (number, field) in top_level_fields(extension?)? {
        let slot = match number {
            1 => &mut type_url,
            2 => &mut value,
            _ => return None,
        };
        if slot.replace(field).is_some() {
            return None;
        }
    }
    (type_url? == PLACEHOLDER_TYPE_URL.as_bytes()).then_some((common, value.unwrap_or_default()))
}

/// The number of the field of the `common` of a relation, and of its extension.
const COMMON_FIELD: u64 = 1;
const EXTENSION_FIELD: u64 = 998;

/// The fields of `encoded`, a message whose fields are all length-delimited, by their numbers and
/// values; None when it is not such a message.
fn top_level_fields(encoded: &[u8]) -> Option<Vec<(u64, &[u8])>> {
    use prost::encoding::decode_varint;
    let mut fields = Vec::new();
    let mut rest = encoded;
    while !rest.is_empty() {
        let key = decode_varint(&mut rest).ok()?;
        if key & 7 != 2 || key >> 3 == 0 {
            return None;
        }
        let len = usize::try_from(decode_varint(&mut rest).ok()?).ok()?;
        if len > rest.len() {
            return None;
        }
        let (value, after) = rest.split_at(len);
        fields.push((key >> 3, value));
        rest = after;
    }
    Some(fields)
}

/// Replace the placeholders of the relations of `command`, a command built in Python, as
/// `resolve_relation_levels`: the relation of a command that holds one, as PySpark's commands hold
/// the relations of their DataFrames (the input of a write, of a view, of a stream or of a SQL
/// statement, the relation of a checkpoint, the source of a `MERGE INTO`, the relation of a
/// pipeline's flow), the datasets and the fetched arguments of the ML commands, the message of the
/// extension of a plugin's command, and the messages of the extensions of the expressions of the
/// command (see [`resolve_own_fields`]), of a command whose encoding holds `urls` type URLs of
/// placeholders (see [`resolve_with`]).
pub(crate) fn resolve_command(command: &mut proto::Command, urls: Option<usize>) -> Result<()> {
    resolve_with(urls, |resolver| resolve_command_with(command, resolver))
}

/// Resolve `command` of the search of `resolver` (see [`resolve_command`]).
fn resolve_command_with(command: &mut proto::Command, resolver: &mut Resolver) -> Result<()> {
    use proto::command::CommandType;
    use proto::pipeline_command::CommandType as PipelineCommandType;
    let everywhere = resolver.search == Search::Everywhere;
    match command.command_type.as_mut() {
        Some(CommandType::Extension(extension)) => {
            return if everywhere {
                resolve_extension(extension)
            } else {
                Ok(())
            };
        }
        Some(CommandType::PipelineCommand(proto::PipelineCommand {
            command_type: Some(PipelineCommandType::Extension(extension)),
        })) => {
            return if everywhere {
                resolve_extension(extension)
            } else {
                Ok(())
            };
        }
        Some(CommandType::MlCommand(ml)) => return resolve_ml_command(ml, resolver),
        None => return Ok(()),
        Some(_) => {}
    }
    if let Some(relation) = command_relation_mut(command) {
        resolve_relation_at(relation, 1, resolver)?;
    }
    if !everywhere {
        return Ok(());
    }
    // The extensions of the expressions of the command, such as of the conditions and the
    // assignments of a `MergeIntoTableCommand` and the partitioning of a `WriteOperationV2`.
    resolve_own_fields(command, |command| {
        command_relation_mut(command).into_iter().collect()
    })
}

/// Replace the placeholders of the datasets and of the fetched arguments of `ml`, an ML command
/// built in Python, of the search of `resolver`.
fn resolve_ml_command(ml: &mut proto::MlCommand, resolver: &mut Resolver) -> Result<()> {
    let dataset = match ml.command.as_mut() {
        Some(proto::ml_command::Command::Fit(fit)) => fit.dataset.as_mut(),
        Some(proto::ml_command::Command::Evaluate(evaluate)) => evaluate.dataset.as_mut(),
        Some(proto::ml_command::Command::CreateSummary(summary)) => summary.dataset.as_mut(),
        Some(proto::ml_command::Command::Fetch(fetch)) => {
            return resolve_fetch_at(fetch, 1, resolver).map(|_| ())
        }
        _ => None,
    };
    match dataset {
        Some(dataset) => resolve_relation_at(dataset, 1, resolver).map(|_| ()),
        None => Ok(()),
    }
}

/// The relation of `command`, of the commands that hold one, as `resolve_command` resolves it.
fn command_relation_mut(command: &mut proto::Command) -> Option<&mut proto::Relation> {
    use proto::command::CommandType;
    use proto::pipeline_command::define_flow::Details;
    use proto::pipeline_command::CommandType as PipelineCommandType;
    match command.command_type.as_mut()? {
        CommandType::WriteOperation(write) => write.input.as_mut(),
        CommandType::CreateDataframeView(view) => view.input.as_mut(),
        CommandType::WriteOperationV2(write) => write.input.as_mut(),
        CommandType::SqlCommand(sql) => sql.input.as_mut(),
        CommandType::WriteStreamOperationStart(write) => write.input.as_mut(),
        CommandType::CheckpointCommand(checkpoint) => checkpoint.relation.as_mut(),
        CommandType::MergeIntoTableCommand(merge) => merge.source_table_plan.as_mut(),
        CommandType::PipelineCommand(pipeline) => match pipeline.command_type.as_mut()? {
            PipelineCommandType::DefineFlow(flow) => match flow.details.as_mut()? {
                Details::RelationFlowDetails(details) => details.relation.as_mut(),
                _ => None,
            },
            PipelineCommandType::DefineFlowQueryFunctionResult(result) => result.relation.as_mut(),
            _ => None,
        },
        _ => None,
    }
}

/// Replace the placeholders of the relations that a fetch of `fetch` takes as arguments, each at
/// the level `level`, of the search of `resolver`, and the deepest level of them.
fn resolve_fetch_at(
    fetch: &mut proto::Fetch,
    level: usize,
    resolver: &mut Resolver,
) -> Result<usize> {
    let mut deepest = 0;
    for method in fetch.methods.iter_mut() {
        for args in method.args.iter_mut() {
            if let Some(proto::fetch::method::args::ArgsType::Input(input)) =
                args.args_type.as_mut()
            {
                deepest = deepest.max(resolve_relation_at(input, level, resolver)?);
            }
        }
    }
    Ok(deepest)
}

/// `request`, a request that a session hook built in Python, with the placeholders of its relation
/// or of its command replaced, as `DataFrame::from_relation` replaces them, of the search that
/// the encoding of the request allows, if the core is given it (see [`BuiltOutside`]). Not a
/// stable API.
#[doc(hidden)]
pub fn resolve_request(
    request: impl Into<BuiltOutside<proto::ExecutePlanRequest>>,
) -> Result<proto::ExecutePlanRequest> {
    let (mut request, urls) = request.into().into_parts();
    if let Some(plan) = request.plan.as_mut() {
        resolve_plan(plan, urls)?;
    }
    Ok(request)
}

/// Replace the placeholders of `plan`, a plan built outside the core whose encoding holds `urls`
/// type URLs of placeholders (see [`resolve_with`]), of its relation or of its command, as
/// [`resolve_request`] does for the plan of a request.
pub(crate) fn resolve_plan(plan: &mut proto::Plan, urls: Option<usize>) -> Result<()> {
    resolve_with(urls, |resolver| resolve_plan_with(plan, resolver))
}

/// Resolve `plan` of the search of `resolver` (see [`resolve_plan`]).
fn resolve_plan_with(plan: &mut proto::Plan, resolver: &mut Resolver) -> Result<()> {
    match plan.op_type.as_mut() {
        Some(proto::plan::OpType::Root(relation)) => {
            resolve_relation_at(relation, 1, resolver).map(|_| ())
        }
        Some(proto::plan::OpType::Command(command)) => resolve_command_with(command, resolver),
        _ => Ok(()),
    }
}

/// Replace the placeholders of the plans and the relations of `analyze`, an analysis built
/// outside the core, as [`resolve_plan`] does for each, of an analysis whose request's encoding
/// holds `urls` type URLs of placeholders (see [`resolve_with`]).
pub(crate) fn resolve_analysis(
    analyze: &mut proto::analyze_plan_request::Analyze,
    urls: Option<usize>,
) -> Result<()> {
    resolve_with(urls, |resolver| resolve_analysis_with(analyze, resolver))
}

/// Resolve `analyze` of the search of `resolver` (see [`resolve_analysis`]).
fn resolve_analysis_with(
    analyze: &mut proto::analyze_plan_request::Analyze,
    resolver: &mut Resolver,
) -> Result<()> {
    use proto::analyze_plan_request::Analyze;
    let (plans, relation) = match analyze {
        Analyze::Schema(request) => (vec![&mut request.plan], None),
        Analyze::Explain(request) => (vec![&mut request.plan], None),
        Analyze::TreeString(request) => (vec![&mut request.plan], None),
        Analyze::IsLocal(request) => (vec![&mut request.plan], None),
        Analyze::IsStreaming(request) => (vec![&mut request.plan], None),
        Analyze::InputFiles(request) => (vec![&mut request.plan], None),
        Analyze::SemanticHash(request) => (vec![&mut request.plan], None),
        Analyze::SameSemantics(request) => (
            vec![&mut request.target_plan, &mut request.other_plan],
            None,
        ),
        Analyze::Persist(request) => (Vec::new(), request.relation.as_mut()),
        Analyze::Unpersist(request) => (Vec::new(), request.relation.as_mut()),
        Analyze::GetStorageLevel(request) => (Vec::new(), request.relation.as_mut()),
        Analyze::SparkVersion(_) | Analyze::DdlParse(_) | Analyze::JsonToDdl(_) => {
            (Vec::new(), None)
        }
    };
    for plan in plans.into_iter().flatten() {
        resolve_plan_with(plan, resolver)?;
    }
    match relation {
        Some(relation) => resolve_relation_at(relation, 1, resolver).map(|_| ()),
        None => Ok(()),
    }
}

/// How many levels deep a plan may be to be sent, with the plans of its subqueries. It protects
/// this client: the serialization of a plan, the clone of its request and the encoding of the
/// request recurse once per level, so a deeper plan is refused here with a `RecursionError`,
/// before any request is built, and the levels that are sent fit the stack that the client grows
/// for them. PySpark's own point is not this number, and depends on its protobuf backend: its
/// serializer recurses in Python, so it raises `RecursionError` from about 245 levels with the
/// pure-Python backend, and from about 992 with upb (measured with pyspark-client 4.3.0). The
/// server itself rejects a plan of about 510 levels or more (its marshaller nests at most 1024
/// messages), with an error that gives no reason, so the plans between the two fail there.
pub const MAX_PLAN_DEPTH: usize = 746;

/// The stack that the clone of a [`LogicalPlan::Raw`] relation takes for each of its levels:
/// prost's `Clone` recurses for each, with frames of about 1 KiB in a release build and 4 KiB in a
/// debug one, and this leaves as much again.
const RAW_CLONE_STACK_PER_LEVEL: usize = if cfg!(debug_assertions) {
    8 << 10
} else {
    2 << 10
};

/// The stack that the clone of a [`LogicalPlan::Raw`] relation of `levels` levels takes: see
/// [`RAW_CLONE_STACK_PER_LEVEL`].
fn raw_relation_stack(levels: usize) -> usize {
    levels
        .saturating_mul(RAW_CLONE_STACK_PER_LEVEL)
        .saturating_add(64 << 10)
}

thread_local! {
    /// The level of the plan that this thread serializes now, the plans of subqueries
    /// included, and the deepest one since the start of the innermost `with_plan_depth_bound`.
    static PLAN_LEVEL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static DEEPEST_PLAN_LEVEL: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Whether this thread serializes in a `with_plan_depth_bound`, which stops at the bound.
    static PLAN_DEPTH_BOUNDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// One level of a plan that is being serialized, left when it is dropped (also on a panic).
struct PlanLevel(usize);

impl PlanLevel {
    fn enter() -> Self {
        let level = PLAN_LEVEL.with(|current| {
            let level = current.get() + 1;
            current.set(level);
            level
        });
        DEEPEST_PLAN_LEVEL.with(|deepest| deepest.set(deepest.get().max(level)));
        PlanLevel(level)
    }
}

impl Drop for PlanLevel {
    fn drop(&mut self) {
        PLAN_LEVEL.with(|current| current.set(current.get() - 1));
    }
}

/// Run `serialize`, which serializes plans for a request, and fail as PySpark's serializer does
/// if one of them, with the plans of its subqueries, is deeper than [`MAX_PLAN_DEPTH`]. Within
/// it, a plan is serialized down to the bound only: its proto is dropped and encoded by
/// recursing too, so it must not be deeper. Returns the levels that the plans reach.
fn with_plan_depth_bound<T>(serialize: impl FnOnce() -> T) -> Result<(T, usize)> {
    struct Scope {
        outer_bounded: bool,
        outer_deepest: usize,
    }
    impl Drop for Scope {
        fn drop(&mut self) {
            PLAN_DEPTH_BOUNDED.with(|bounded| bounded.set(self.outer_bounded));
            DEEPEST_PLAN_LEVEL.with(|deepest| deepest.set(deepest.get().max(self.outer_deepest)));
        }
    }
    let _scope = Scope {
        outer_bounded: PLAN_DEPTH_BOUNDED.with(|bounded| bounded.replace(true)),
        outer_deepest: DEEPEST_PLAN_LEVEL.with(|deepest| deepest.replace(0)),
    };
    let base = PLAN_LEVEL.with(std::cell::Cell::get);
    let value = serialize();
    let deepest = DEEPEST_PLAN_LEVEL.with(std::cell::Cell::get);
    if deepest > MAX_PLAN_DEPTH {
        return Err(SparkError::recursion_msg(format!(
            "maximum recursion depth exceeded while serializing a plan and the plans of its \
             subqueries; a plan may be at most {MAX_PLAN_DEPTH} levels deep"
        )));
    }
    Ok((value, deepest.saturating_sub(base)))
}

/// The `RecursionError` of a plan of `depth` levels, deeper than [`MAX_PLAN_DEPTH`].
pub(crate) fn too_deep_a_plan(depth: usize) -> SparkError {
    SparkError::recursion_msg(format!(
        "maximum recursion depth exceeded while serializing a plan of {depth} levels; \
         a plan may be at most {MAX_PLAN_DEPTH} levels deep"
    ))
}

/// The name that the `CollectMetrics` of an `Observe` of `name` and `observation` is built of, as
/// PySpark's sets the `str` of the `_name` of its Observation each time it builds the plan: that
/// which the binding of the Observation gives now (see
/// [`crate::observation::MetricsMirror::name`]), or its error, else `name`.
fn observed_name(
    name: &str,
    observation: Option<&crate::observation::Observation>,
) -> Result<String> {
    match observation.and_then(crate::observation::Observation::name_of_plan) {
        Some(current) => current,
        None => Ok(name.to_string()),
    }
}

/// Serialize a shared plan node for a request: the node and every node below it carry
/// their stable plan IDs. A plan deeper than [`MAX_PLAN_DEPTH`], with the plans of its
/// subqueries, or with an expression deeper than [`crate::expression::MAX_EXPRESSION_DEPTH`], is
/// a `RecursionError`, as in PySpark. The plan is serialized on the calling thread, whose stack
/// grows on demand for its levels.
pub(crate) fn relation_with_ids(node: &Arc<LogicalPlan>, ids: &PlanIds) -> Result<proto::Relation> {
    relation_with_ids_and_levels(node, ids).map(|(relation, _)| relation)
}

/// As [`relation_with_ids`], and the levels of the relation, those of the plans of its
/// subqueries included.
fn relation_with_ids_and_levels(
    node: &Arc<LogicalPlan>,
    ids: &PlanIds,
) -> Result<(proto::Relation, usize)> {
    // The levels of the plan itself are known before it is serialized; those of the plans of
    // its subqueries are counted as they are serialized. A plan or an expression that a request
    // fails to build raises its error after the serialization, the first of them in PySpark's
    // order (`LogicalPlan::Unbuildable`).
    let depth = node.depth();
    if depth > MAX_PLAN_DEPTH {
        return Err(too_deep_a_plan(depth));
    }
    let (relation, levels) = with_plan_depth_bound(|| {
        crate::expression::with_expression_depth_bound(|| relation_of(node, Some(ids)))
    })?;
    Ok((relation?, levels))
}

/// Serialize a child node, with its stable plan ID when `ids` is given.
fn relation_of(node: &Arc<LogicalPlan>, ids: Option<&PlanIds>) -> proto::Relation {
    relation_with_references(node, Some(node), ids)
}

/// A plan that a subquery expression refers to, with the plan IDs of the session of its
/// DataFrame, if it has them.
type Reference = (Arc<LogicalPlan>, Option<Arc<PlanIds>>);

thread_local! {
    /// For each relation being serialized, innermost last, the plans that the subquery
    /// expressions of its own expressions refer to.
    static SUBQUERY_REFERENCES: std::cell::RefCell<Vec<Vec<Reference>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Record that a subquery expression being serialized refers to `plan`, of the plan IDs `ids`
/// of the session of its DataFrame, which the relation that contains the expression sends in
/// its `WithRelations`.
pub(crate) fn refer_to(plan: &Arc<LogicalPlan>, ids: Option<&Arc<PlanIds>>) {
    SUBQUERY_REFERENCES.with(|frames| {
        if let Some(frame) = frames.borrow_mut().last_mut() {
            // A new reference to the plan, which the check of the clones of a binding is told of
            // (see `crate::gc::check_clones_with`).
            crate::gc::cloned(Arc::as_ptr(plan) as usize);
            frame.push((Arc::clone(plan), ids.cloned()));
        }
    });
}

/// The plans that the subqueries of the own expressions of `node` refer to (its children's are of
/// them), each once, in the order of its expressions and of their nodes, as the references of
/// PySpark's plan of the node hold them, which its pickle takes after its child: of a subquery of
/// an IN, of a table argument and of the arguments of a subquery's own expressions too.
pub(crate) fn subquery_plans(node: &LogicalPlan) -> Vec<&Arc<LogicalPlan>> {
    use crate::udf::PicklerHolder;
    let mut plans: Vec<&Arc<LogicalPlan>> = Vec::new();
    let mut pending = Vec::new();
    node.push_own_pickler_holders(&mut pending);
    // The first is visited first.
    pending.reverse();
    while let Some(holder) = pending.pop() {
        let mut below = Vec::new();
        match holder {
            PicklerHolder::Expression(expression) => {
                expression.node().push_pickler_holders(&mut below)
            }
            PicklerHolder::Subquery(subquery) => {
                if !plans.iter().any(|plan| Arc::ptr_eq(plan, &subquery.plan)) {
                    plans.push(&subquery.plan);
                }
                below.extend(
                    subquery
                        .partition_spec
                        .iter()
                        .chain(&subquery.order_spec)
                        .chain(&subquery.in_subquery_values)
                        .map(PicklerHolder::Expression),
                );
            }
            PicklerHolder::Udf(udf) => {
                below.extend(udf.arguments.iter().map(PicklerHolder::Expression));
            }
            PicklerHolder::Plan(_) | PicklerHolder::Pickler(_) | PicklerHolder::SentPlans(_) => {}
        }
        pending.extend(below.into_iter().rev());
    }
    plans
}

/// Whether `plan` is the node of a relation built outside the core whose build failed with
/// `error` (see [`crate::dataframe::DataFrame::from_unbuilt_relation`]), rather than a plan over
/// one.
pub(crate) fn is_unbuilt_with(plan: &LogicalPlan, error: &SparkError) -> bool {
    match plan {
        LogicalPlan::Unbuildable { input, error: own } => {
            matches!(input.as_ref(), LogicalPlan::Raw { .. })
                && own.error_class == error.error_class
                && own.params == error.params
        }
        _ => false,
    }
}

/// Whether `node` is sent in a `WithRelations` of the plans that the subqueries of its own
/// expressions refer to: of a subquery ([`subquery_plans`]), or of an expression as it was sent,
/// of a pickled Column of subqueries, which sends the plans of its subqueries too.
pub(crate) fn has_subquery_references(node: &LogicalPlan) -> bool {
    use crate::udf::PicklerHolder;
    if !subquery_plans(node).is_empty() {
        return true;
    }
    let mut pending = Vec::new();
    node.push_own_pickler_holders(&mut pending);
    while let Some(holder) = pending.pop() {
        match holder {
            PicklerHolder::Expression(expression) => {
                expression.node().push_pickler_holders(&mut pending)
            }
            PicklerHolder::Udf(udf) => {
                pending.extend(udf.arguments.iter().map(PicklerHolder::Expression));
            }
            PicklerHolder::SentPlans(plans) => {
                if plans.plans().next().is_some() {
                    return true;
                }
            }
            PicklerHolder::Subquery(_) | PicklerHolder::Plan(_) | PicklerHolder::Pickler(_) => {}
        }
    }
    false
}

/// The relation of `node`, and the plans that the subqueries of its own expressions refer
/// to; its children collect those of theirs as they are serialized.
fn relation_and_references(
    node: &LogicalPlan,
    ids: Option<&PlanIds>,
) -> (proto::Relation, Vec<Reference>) {
    struct Frame;
    impl Drop for Frame {
        fn drop(&mut self) {
            SUBQUERY_REFERENCES.with(|frames| {
                frames.borrow_mut().pop();
            });
        }
    }
    SUBQUERY_REFERENCES.with(|frames| frames.borrow_mut().push(Vec::new()));
    let _frame = Frame;
    let relation = node.to_proto_with(ids);
    let references = SUBQUERY_REFERENCES.with(|frames| {
        frames
            .borrow_mut()
            .last_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    });
    (relation, references)
}

/// The relation of `node`, with the stable plan ID of `shared` when `ids` is given. As
/// PySpark's `LogicalPlan._with_relations`, a node whose expressions have subqueries is sent
/// in a `WithRelations` with the plans that they refer to:
///
/// ```text
/// with_relations [plan ID of the node]
///     root: the relation of the node [its root plan ID]
///     references: the relation of each plan [its plan ID]
/// ```
fn relation_with_references(
    node: &LogicalPlan,
    shared: Option<&Arc<LogicalPlan>>,
    ids: Option<&PlanIds>,
) -> proto::Relation {
    // A level serializes the levels below it and the plans of its subqueries: grow the stack for
    // a plan of many levels, and count them for `with_plan_depth_bound`.
    crate::expression::with_stack_for_a_level(|| {
        let level = PlanLevel::enter();
        // A relation built outside the core counts its own levels below this one.
        let deepest = level.0 + node.own_levels() - 1;
        DEEPEST_PLAN_LEVEL.with(|levels| levels.set(levels.get().max(deepest)));
        if deepest > MAX_PLAN_DEPTH && PLAN_DEPTH_BOUNDED.with(std::cell::Cell::get) {
            // The request fails with a `RecursionError`: stop here, so that the proto of the
            // plan is no deeper than the bound.
            return proto::Relation::default();
        }
        relation_with_references_of_a_level(node, shared, ids)
    })
}

fn relation_with_references_of_a_level(
    node: &LogicalPlan,
    shared: Option<&Arc<LogicalPlan>>,
    ids: Option<&PlanIds>,
) -> proto::Relation {
    let (mut relation, references) = relation_and_references(node, ids);
    let plan_ids = ids.zip(shared);
    if references.is_empty() {
        if let Some((ids, shared)) = plan_ids {
            relation.common.get_or_insert_with(Default::default).plan_id = Some(ids.id_of(shared));
        }
        return relation;
    }
    if let Some((ids, shared)) = plan_ids {
        relation.common.get_or_insert_with(Default::default).plan_id = Some(ids.root_id_of(shared));
    }
    // As PySpark's `ref.plan(session)`, each plan with its own plan IDs, which the subquery
    // and the columns of its plan refer to: those of the session of its DataFrame, which may be
    // another session than this relation's.
    let with_relations = proto::WithRelations {
        root: Some(Box::new(relation)),
        references: references
            .iter()
            .map(|(reference, own)| {
                relation_of(reference, ids.map(|ids| own.as_deref().unwrap_or(ids)))
            })
            .collect(),
    };
    proto::Relation {
        common: Some(proto::RelationCommon {
            plan_id: plan_ids.map(|(ids, shared)| ids.id_of(shared)),
            ..Default::default()
        }),
        rel_type: Some(proto::relation::RelType::WithRelations(Box::new(
            with_relations,
        ))),
    }
}

/// The plans that `$plan` reads, as references of the same mutability as `$plan`.
macro_rules! inputs_of {
    ($plan:expr) => {{
        let mut inputs = Vec::new();
        match $plan {
            LogicalPlan::Project { input, .. }
            | LogicalPlan::Filter { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Offset { input, .. }
            | LogicalPlan::Tail { input, .. }
            | LogicalPlan::ShowString { input, .. }
            | LogicalPlan::HtmlString { input, .. }
            | LogicalPlan::Deduplicate { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Sample { input, .. }
            | LogicalPlan::Repartition { input, .. }
            | LogicalPlan::RepartitionByExpression { input, .. }
            | LogicalPlan::WithColumns { input, .. }
            | LogicalPlan::WithColumnMetadata { input, .. }
            | LogicalPlan::WithColumnsRenamed { input, .. }
            | LogicalPlan::Drop { input, .. }
            | LogicalPlan::ToDF { input, .. }
            | LogicalPlan::ToSchema { input, .. }
            | LogicalPlan::Unbuildable { input, .. }
            | LogicalPlan::Hint { input, .. }
            | LogicalPlan::Parse { input, .. }
            | LogicalPlan::Unpivot { input, .. }
            | LogicalPlan::NAFill { input, .. }
            | LogicalPlan::NAFillColumns { input, .. }
            | LogicalPlan::NADrop { input, .. }
            | LogicalPlan::NAReplace { input, .. }
            | LogicalPlan::Describe { input, .. }
            | LogicalPlan::Summary { input, .. }
            | LogicalPlan::ColRegex { input, .. }
            | LogicalPlan::SubqueryAlias { input, .. }
            | LogicalPlan::WithWatermark { input, .. }
            | LogicalPlan::RepartitionByRange { input, .. }
            | LogicalPlan::StatCrosstab { input, .. }
            | LogicalPlan::StatFreqItems { input, .. }
            | LogicalPlan::StatApproxQuantile { input, .. }
            | LogicalPlan::StatCorr { input, .. }
            | LogicalPlan::StatCov { input, .. }
            | LogicalPlan::StatSampleBy { input, .. }
            | LogicalPlan::Observe { input, .. }
            | LogicalPlan::MlTransform { input, .. }
            | LogicalPlan::Transpose { input, .. }
            | LogicalPlan::MapPartitions { input, .. }
            | LogicalPlan::ApplyInPandasWithState { input, .. } => inputs.push(input),
            LogicalPlan::Join { left, right, .. }
            | LogicalPlan::LateralJoin { left, right, .. }
            | LogicalPlan::SetOperation { left, right, .. }
            | LogicalPlan::Zip { left, right, .. }
            | LogicalPlan::AsOfJoin { left, right, .. }
            | LogicalPlan::NearestByJoin { left, right, .. } => {
                inputs.push(left);
                inputs.push(right);
            }
            LogicalPlan::CoGroupMap { input, other, .. } => {
                inputs.push(input);
                inputs.push(other);
            }
            LogicalPlan::GroupMap {
                input,
                initial_input,
                ..
            } => {
                inputs.push(input);
                inputs.extend(initial_input);
            }
            LogicalPlan::WithRelations { root, references } => {
                inputs.push(root);
                inputs.extend(references);
            }
            LogicalPlan::Range { .. }
            | LogicalPlan::Sql { .. }
            | LogicalPlan::Raw { .. }
            | LogicalPlan::LocalRelation { .. }
            | LogicalPlan::ChunkedCachedLocalRelation { .. }
            | LogicalPlan::CachedRemoteRelation { .. }
            | LogicalPlan::CachedRelation { .. }
            | LogicalPlan::Read { .. }
            | LogicalPlan::RelationChanges { .. }
            | LogicalPlan::UnresolvedTableValuedFunction { .. }
            | LogicalPlan::Catalog { .. }
            | LogicalPlan::CommonInlineUdtf { .. } => {}
        }
        inputs
    }};
}

/// A leaf that a dropped node leaves in place of each of its inputs, so that the inputs are
/// dropped one at a time rather than each inside the drop of the node above it.
static DROPPED_INPUT: LazyLock<Arc<LogicalPlan>> = LazyLock::new(|| {
    Arc::new(LogicalPlan::Range {
        start: 0,
        end: 0,
        step: 1,
        num_partitions: None,
    })
});

impl Drop for LogicalPlan {
    /// Drop the plans below this one that nothing else holds, without recursing: a plan of
    /// many levels would otherwise overflow the stack of the thread that drops it.
    fn drop(&mut self) {
        let mut pending: Vec<Arc<LogicalPlan>> = Vec::new();
        let take_inputs = |plan: &mut LogicalPlan, pending: &mut Vec<Arc<LogicalPlan>>| {
            for input in inputs_of!(plan) {
                if !Arc::ptr_eq(input, &DROPPED_INPUT) {
                    pending.push(std::mem::replace(input, Arc::clone(&DROPPED_INPUT)));
                }
            }
        };
        take_inputs(self, &mut pending);
        while let Some(input) = pending.pop() {
            if let Some(mut plan) = Arc::into_inner(input) {
                take_inputs(&mut plan, &mut pending);
            }
        }
    }
}

impl LogicalPlan {
    /// The plans that this one reads directly.
    pub(crate) fn inputs(&self) -> Vec<&Arc<LogicalPlan>> {
        inputs_of!(self)
    }

    /// The inputs of this node, then its expressions, UDFs, the pickler of its UDTF and its
    /// Observations, which a deep copy of the functions of its UDFs replaces (see
    /// [`crate::udf::PicklerRebuild`]).
    pub(crate) fn parts_mut(&mut self) -> Vec<PlanPart<'_>> {
        use PlanPart::{Expression as Of, Input, Observation, Observations, Pickler, Udf};
        fn columns(columns: &mut [Column]) -> impl Iterator<Item = PlanPart<'_>> {
            columns.iter_mut().map(|column| Of(column.expression_mut()))
        }
        let mut parts = Vec::new();
        match self {
            LogicalPlan::Sql {
                pos_args,
                named_args,
                ..
            } => {
                parts.extend(pos_args.iter_mut().map(Of));
                parts.extend(named_args.values_mut().map(Of));
            }
            LogicalPlan::Project { input, columns: c }
            | LogicalPlan::WithColumns {
                input, columns: c, ..
            } => {
                parts.push(Input(input));
                parts.extend(columns(c));
            }
            LogicalPlan::Filter { input, condition } => {
                parts.push(Input(input));
                parts.push(Of(condition.expression_mut()));
            }
            LogicalPlan::Aggregate {
                input,
                grouping_expressions,
                aggregate_expressions,
                pivot_col,
                pivot_values,
                grouping_sets,
                ..
            } => {
                parts.push(Input(input));
                parts.extend(grouping_expressions.iter_mut().map(Of));
                parts.extend(aggregate_expressions.iter_mut().map(Of));
                parts.extend(pivot_col.iter_mut().map(Of));
                parts.extend(pivot_values.iter_mut().map(Of));
                parts.extend(grouping_sets.iter_mut().flatten().map(Of));
            }
            LogicalPlan::Join {
                left, right, on, ..
            }
            | LogicalPlan::LateralJoin {
                left, right, on, ..
            } => {
                parts.extend([Input(left), Input(right)]);
                parts.extend(on.iter_mut().map(|on| Of(on.expression_mut())));
            }
            LogicalPlan::SetOperation { left, right, .. } | LogicalPlan::Zip { left, right } => {
                parts.extend([Input(left), Input(right)])
            }
            LogicalPlan::Sort {
                input,
                order: expressions,
                ..
            }
            | LogicalPlan::RepartitionByExpression {
                input, expressions, ..
            }
            | LogicalPlan::Drop {
                input, expressions, ..
            }
            | LogicalPlan::Hint {
                input,
                parameters: expressions,
                ..
            }
            | LogicalPlan::RepartitionByRange {
                input,
                partition_exprs: expressions,
                ..
            }
            | LogicalPlan::Transpose {
                input,
                index_columns: expressions,
            } => {
                parts.push(Input(input));
                parts.extend(expressions.iter_mut().map(Of));
            }
            LogicalPlan::Observe {
                input,
                name,
                exprs,
                observation,
            } => {
                parts.push(Input(input));
                parts.extend(exprs.iter_mut().map(Of));
                parts.push(Observation(name.as_str(), observation));
            }
            LogicalPlan::Raw {
                observations,
                holes,
                ..
            } => {
                parts.push(Observations(observations));
                parts.extend(holes.0.iter_mut().map(|hole| Pickler(&mut hole.pickler)));
            }
            LogicalPlan::UnresolvedTableValuedFunction { arguments, .. } => {
                parts.extend(arguments.iter_mut().map(Of));
            }
            LogicalPlan::WithColumnMetadata { input, column, .. } => {
                parts.push(Input(input));
                parts.push(Of(column));
            }
            LogicalPlan::Unpivot {
                input, ids, values, ..
            } => {
                parts.push(Input(input));
                parts.extend(columns(ids));
                if let Some(values) = values {
                    parts.extend(columns(values));
                }
            }
            LogicalPlan::StatSampleBy {
                input,
                col,
                fractions,
                ..
            } => {
                parts.push(Input(input));
                parts.push(Of(col));
                parts.extend(fractions.iter_mut().map(|(stratum, _)| Of(stratum)));
            }
            LogicalPlan::MapPartitions { input, func, .. } => {
                parts.push(Input(input));
                parts.push(Udf(func));
            }
            LogicalPlan::GroupMap {
                input,
                grouping_expressions,
                func,
                sorting_expressions,
                initial_input,
                initial_grouping_expressions,
                ..
            } => {
                parts.push(Input(input));
                parts.extend(grouping_expressions.iter_mut().map(Of));
                parts.extend(initial_input.iter_mut().map(Input));
                parts.extend(initial_grouping_expressions.iter_mut().map(Of));
                parts.push(Udf(func));
                parts.extend(sorting_expressions.iter_mut().map(Of));
            }
            LogicalPlan::ApplyInPandasWithState {
                input,
                grouping_expressions,
                func,
                ..
            } => {
                parts.push(Input(input));
                parts.extend(grouping_expressions.iter_mut().map(Of));
                parts.push(Udf(func));
            }
            LogicalPlan::CoGroupMap {
                input,
                input_grouping_expressions,
                other,
                other_grouping_expressions,
                func,
            } => {
                parts.push(Input(input));
                parts.extend(input_grouping_expressions.iter_mut().map(Of));
                parts.push(Input(other));
                parts.extend(other_grouping_expressions.iter_mut().map(Of));
                parts.push(Udf(func));
            }
            LogicalPlan::AsOfJoin {
                left,
                right,
                left_as_of,
                right_as_of,
                on,
                tolerance,
                ..
            } => {
                parts.extend([Input(left), Input(right)]);
                parts.push(Of(left_as_of.expression_mut()));
                parts.push(Of(right_as_of.expression_mut()));
                parts.extend(on.iter_mut().map(|on| Of(on.expression_mut())));
                parts.extend(tolerance.iter_mut().map(|t| Of(t.expression_mut())));
            }
            LogicalPlan::NearestByJoin {
                left,
                right,
                ranking_expression,
                ..
            } => {
                parts.extend([Input(left), Input(right)]);
                parts.push(Of(ranking_expression));
            }
            LogicalPlan::CommonInlineUdtf {
                arguments, pickler, ..
            } => {
                parts.extend(arguments.iter_mut().map(Of));
                parts.push(Pickler(pickler));
            }
            LogicalPlan::WithRelations { root, references } => {
                parts.push(Input(root));
                parts.extend(references.iter_mut().map(Input));
            }
            LogicalPlan::Limit { input, .. }
            | LogicalPlan::Offset { input, .. }
            | LogicalPlan::Tail { input, .. }
            | LogicalPlan::ShowString { input, .. }
            | LogicalPlan::HtmlString { input, .. }
            | LogicalPlan::Deduplicate { input, .. }
            | LogicalPlan::Sample { input, .. }
            | LogicalPlan::Repartition { input, .. }
            | LogicalPlan::WithColumnsRenamed { input, .. }
            | LogicalPlan::ToDF { input, .. }
            | LogicalPlan::Unbuildable { input, .. }
            | LogicalPlan::Parse { input, .. }
            | LogicalPlan::NAFill { input, .. }
            | LogicalPlan::NAFillColumns { input, .. }
            | LogicalPlan::NADrop { input, .. }
            | LogicalPlan::NAReplace { input, .. }
            | LogicalPlan::Describe { input, .. }
            | LogicalPlan::Summary { input, .. }
            | LogicalPlan::ColRegex { input, .. }
            | LogicalPlan::SubqueryAlias { input, .. }
            | LogicalPlan::WithWatermark { input, .. }
            | LogicalPlan::StatCrosstab { input, .. }
            | LogicalPlan::StatFreqItems { input, .. }
            | LogicalPlan::StatApproxQuantile { input, .. }
            | LogicalPlan::StatCorr { input, .. }
            | LogicalPlan::StatCov { input, .. }
            | LogicalPlan::MlTransform { input, .. } => parts.push(Input(input)),
            // A deep copy copies the schema object of the reader, as PySpark's copies the schema.
            LogicalPlan::ToSchema {
                input,
                schema_reader,
                ..
            } => {
                parts.push(Input(input));
                parts.push(Pickler(schema_reader));
            }
            LogicalPlan::Range { .. }
            | LogicalPlan::LocalRelation { .. }
            | LogicalPlan::ChunkedCachedLocalRelation { .. }
            | LogicalPlan::CachedRemoteRelation { .. }
            | LogicalPlan::CachedRelation { .. }
            | LogicalPlan::Read { .. }
            | LogicalPlan::RelationChanges { .. }
            | LogicalPlan::Catalog { .. } => {}
        }
        parts
    }

    /// The expressions, the UDFs and the pickler of the UDTF of this node, without those of its
    /// inputs, whose picklers [`crate::udf::all_picklers`] visits.
    pub(crate) fn push_own_pickler_holders<'a>(
        &'a self,
        pending: &mut Vec<crate::udf::PicklerHolder<'a>>,
    ) {
        use crate::udf::PicklerHolder::{Expression as Of, Pickler, Udf};
        let of_column = |column: &'a Column| Of(column.expression());
        match self {
            LogicalPlan::Sql {
                pos_args,
                named_args,
                ..
            } => {
                pending.extend(pos_args.iter().map(Of));
                pending.extend(named_args.values().map(Of));
            }
            LogicalPlan::Project { columns, .. } | LogicalPlan::WithColumns { columns, .. } => {
                pending.extend(columns.iter().map(of_column));
            }
            LogicalPlan::Filter { condition, .. } => pending.push(of_column(condition)),
            LogicalPlan::Aggregate {
                grouping_expressions,
                aggregate_expressions,
                pivot_col,
                pivot_values,
                grouping_sets,
                ..
            } => {
                pending.extend(grouping_expressions.iter().map(Of));
                pending.extend(aggregate_expressions.iter().map(Of));
                pending.extend(pivot_col.iter().map(Of));
                pending.extend(pivot_values.iter().map(Of));
                pending.extend(grouping_sets.iter().flatten().map(Of));
            }
            LogicalPlan::Join { on, .. } | LogicalPlan::LateralJoin { on, .. } => {
                pending.extend(on.iter().map(of_column));
            }
            LogicalPlan::Sort {
                order: expressions, ..
            }
            | LogicalPlan::RepartitionByExpression { expressions, .. }
            | LogicalPlan::Drop { expressions, .. }
            | LogicalPlan::Hint {
                parameters: expressions,
                ..
            }
            | LogicalPlan::RepartitionByRange {
                partition_exprs: expressions,
                ..
            }
            | LogicalPlan::Observe {
                exprs: expressions, ..
            }
            | LogicalPlan::UnresolvedTableValuedFunction {
                arguments: expressions,
                ..
            }
            | LogicalPlan::Transpose {
                index_columns: expressions,
                ..
            } => pending.extend(expressions.iter().map(Of)),
            LogicalPlan::WithColumnMetadata { column, .. } => pending.push(Of(column)),
            LogicalPlan::Unpivot { ids, values, .. } => {
                pending.extend(ids.iter().chain(values.iter().flatten()).map(of_column));
            }
            LogicalPlan::StatSampleBy { col, fractions, .. } => {
                pending.push(Of(col));
                pending.extend(fractions.iter().map(|(stratum, _)| Of(stratum)));
            }
            LogicalPlan::MapPartitions {
                func,
                profile_id_reader,
                ..
            } => {
                pending.push(Udf(func));
                pending.extend(profile_id_reader.iter().map(Pickler));
            }
            LogicalPlan::GroupMap {
                grouping_expressions,
                func,
                sorting_expressions,
                initial_grouping_expressions,
                ..
            } => {
                pending.push(Udf(func));
                pending.extend(
                    grouping_expressions
                        .iter()
                        .chain(sorting_expressions)
                        .chain(initial_grouping_expressions)
                        .map(Of),
                );
            }
            LogicalPlan::ApplyInPandasWithState {
                grouping_expressions,
                func,
                ..
            } => {
                pending.push(Udf(func));
                pending.extend(grouping_expressions.iter().map(Of));
            }
            LogicalPlan::CoGroupMap {
                input_grouping_expressions,
                other_grouping_expressions,
                func,
                ..
            } => {
                pending.push(Udf(func));
                pending.extend(
                    input_grouping_expressions
                        .iter()
                        .chain(other_grouping_expressions)
                        .map(Of),
                );
            }
            LogicalPlan::AsOfJoin {
                left_as_of,
                right_as_of,
                on,
                tolerance,
                ..
            } => {
                pending.extend([of_column(left_as_of), of_column(right_as_of)]);
                pending.extend(on.iter().chain(tolerance).map(of_column));
            }
            LogicalPlan::NearestByJoin {
                ranking_expression, ..
            } => pending.push(Of(ranking_expression)),
            LogicalPlan::CommonInlineUdtf {
                arguments, pickler, ..
            } => {
                pending.extend(arguments.iter().map(Of));
                pending.extend(pickler.iter().map(Pickler));
            }
            LogicalPlan::ToSchema { schema_reader, .. } => {
                pending.extend(schema_reader.iter().map(Pickler));
            }
            // The picklers of the holes of a relation that was sent, of a pickled DataFrame.
            LogicalPlan::Raw { holes, .. } => {
                pending.extend(
                    holes
                        .0
                        .iter()
                        .filter_map(|hole| hole.pickler.as_ref())
                        .map(Pickler),
                );
            }
            LogicalPlan::Range { .. }
            | LogicalPlan::SetOperation { .. }
            | LogicalPlan::Limit { .. }
            | LogicalPlan::Offset { .. }
            | LogicalPlan::Tail { .. }
            | LogicalPlan::ShowString { .. }
            | LogicalPlan::HtmlString { .. }
            | LogicalPlan::Deduplicate { .. }
            | LogicalPlan::Sample { .. }
            | LogicalPlan::Repartition { .. }
            | LogicalPlan::WithColumnsRenamed { .. }
            | LogicalPlan::ToDF { .. }
            | LogicalPlan::Unbuildable { .. }
            | LogicalPlan::Parse { .. }
            | LogicalPlan::NAFill { .. }
            | LogicalPlan::NAFillColumns { .. }
            | LogicalPlan::NADrop { .. }
            | LogicalPlan::NAReplace { .. }
            | LogicalPlan::Describe { .. }
            | LogicalPlan::Summary { .. }
            | LogicalPlan::ColRegex { .. }
            | LogicalPlan::SubqueryAlias { .. }
            | LogicalPlan::WithRelations { .. }
            | LogicalPlan::LocalRelation { .. }
            | LogicalPlan::ChunkedCachedLocalRelation { .. }
            | LogicalPlan::CachedRemoteRelation { .. }
            | LogicalPlan::CachedRelation { .. }
            | LogicalPlan::Read { .. }
            | LogicalPlan::RelationChanges { .. }
            | LogicalPlan::WithWatermark { .. }
            | LogicalPlan::StatCrosstab { .. }
            | LogicalPlan::StatFreqItems { .. }
            | LogicalPlan::StatApproxQuantile { .. }
            | LogicalPlan::StatCorr { .. }
            | LogicalPlan::StatCov { .. }
            | LogicalPlan::Zip { .. }
            | LogicalPlan::MlTransform { .. }
            | LogicalPlan::Catalog { .. } => {}
        }
    }

    /// The Observations of this plan by their names, as PySpark's `LogicalPlan.observations`
    /// collects those whose metrics an action of the plan sets: of each `Observe` node of an
    /// [`Observation`](crate::observation::Observation) on the inputs that PySpark's plans hold
    /// as their child, both sides of a join, a set operation and a zip, and the DataFrames that
    /// a relation of `pyspark.ml` holds as its child. Not of the plans that an expression refers
    /// to (a subquery), the DataFrames of `SparkSession.sql`, the other input of a `CoGroupMap`
    /// or the initial state of a `GroupMap`, which PySpark's plans hold apart from their child.
    /// Of two Observations of one name, the one that PySpark's dict keeps: the outer one, and
    /// that of the right side.
    pub(crate) fn observations(&self) -> Vec<(String, crate::observation::Observation)> {
        // PySpark merges `{**child, **own}` and `{**left, **right}`, so the last Observation of
        // a name in the post-order of the plan is kept. This walks the reverse of that order (a
        // node, then its right side, then its left side) and keeps the first one of a name,
        // without recursing, as a plan may be deep. A node that several inputs share is walked
        // once: the Observations under it come after those of its first walk.
        let mut observations: Vec<(String, crate::observation::Observation)> = Vec::new();
        let mut keep = |name: &String, observation: &crate::observation::Observation| {
            if !observations.iter().any(|(kept, _)| kept == name) {
                observations.push((name.clone(), observation.clone()));
            }
        };
        let mut walked: std::collections::HashSet<*const LogicalPlan> = Default::default();
        let mut pending: Vec<&LogicalPlan> = vec![self];
        while let Some(plan) = pending.pop() {
            let key: *const LogicalPlan = plan;
            if !walked.insert(key) {
                continue;
            }
            match plan {
                LogicalPlan::Observe {
                    name,
                    observation: Some(observation),
                    ..
                } => {
                    // By the name of the plan as it is built now, as PySpark's `observations`
                    // names the Observation by the `str` of its `_name`.
                    let name =
                        observed_name(name, Some(observation)).unwrap_or_else(|_| name.clone());
                    keep(&name, observation)
                }
                LogicalPlan::Raw {
                    observations: held, ..
                } => {
                    for (name, observation) in held {
                        keep(name, observation);
                    }
                }
                _ => {}
            }
            match plan {
                // The right side is walked first, as it is popped first.
                LogicalPlan::Join { left, right, .. }
                | LogicalPlan::LateralJoin { left, right, .. }
                | LogicalPlan::SetOperation { left, right, .. }
                | LogicalPlan::Zip { left, right, .. }
                | LogicalPlan::AsOfJoin { left, right, .. }
                | LogicalPlan::NearestByJoin { left, right, .. } => {
                    pending.push(left);
                    pending.push(right);
                }
                LogicalPlan::CoGroupMap { input, .. } | LogicalPlan::GroupMap { input, .. } => {
                    pending.push(input)
                }
                LogicalPlan::WithRelations { root, .. } => pending.push(root),
                // The other plans hold their input as PySpark's hold their child.
                _ => pending.extend(plan.inputs().into_iter().map(|input| input.as_ref())),
            }
        }
        observations
    }

    /// The levels of this node itself, under which its inputs are: one, or those of the relation
    /// of a [`LogicalPlan::Raw`], which holds the plans of its inputs.
    fn own_levels(&self) -> usize {
        match self {
            LogicalPlan::Raw { depth, .. } => (*depth).max(1),
            _ => 1,
        }
    }

    /// The number of levels of this plan: one for a leaf (the levels of its relation for a
    /// [`LogicalPlan::Raw`]), and one more than its deepest input otherwise. Computed without
    /// recursing, and once for each node that several plans share.
    pub fn depth(&self) -> usize {
        let mut depths: HashMap<*const LogicalPlan, usize> = HashMap::new();
        let mut pending: Vec<(&LogicalPlan, bool)> = vec![(self, false)];
        while let Some((plan, inputs_done)) = pending.pop() {
            let key: *const LogicalPlan = plan;
            if depths.contains_key(&key) {
                continue;
            }
            let inputs = plan.inputs();
            if inputs_done {
                let deepest = inputs
                    .iter()
                    .map(|input| depths[&Arc::as_ptr(input)])
                    .max()
                    .unwrap_or(0);
                depths.insert(key, deepest + plan.own_levels());
            } else {
                pending.push((plan, true));
                pending.extend(
                    inputs
                        .into_iter()
                        .filter(|input| !depths.contains_key(&Arc::as_ptr(input)))
                        .map(|input| (input.as_ref(), false)),
                );
            }
        }
        depths[&(self as *const LogicalPlan)]
    }

    /// Convert this logical plan to a `spark.connect.Relation` protobuf, without plan IDs.
    ///
    /// Any depth is serialized, on a stack that grows for its levels; but prost encodes and drops
    /// a protobuf by recursing, so the proto of a plan of many thousands of levels can overflow a
    /// small stack there (a request is bounded, see [`LogicalPlan::try_to_proto`]).
    pub fn to_proto(&self) -> proto::Relation {
        relation_with_references(self, None, None)
    }

    /// Convert this logical plan to a `spark.connect.Relation` protobuf, without plan IDs, as
    /// a request serializes it: a plan deeper than [`MAX_PLAN_DEPTH`], with the plans of its
    /// subqueries, or with an expression deeper than [`crate::expression::MAX_EXPRESSION_DEPTH`],
    /// is a `RecursionError`, as PySpark's serializer raises it, and its proto is not built.
    pub fn try_to_proto(&self) -> Result<proto::Relation> {
        let depth = self.depth();
        if depth > MAX_PLAN_DEPTH {
            return Err(too_deep_a_plan(depth));
        }
        let (relation, _) = with_plan_depth_bound(|| {
            crate::expression::with_expression_depth_bound(|| self.to_proto())
        })?;
        relation
    }

    fn to_proto_with(&self, plan_ids: Option<&PlanIds>) -> proto::Relation {
        let mut relation = proto::Relation::default();
        relation.common = Some(proto::RelationCommon::default());

        match self {
            LogicalPlan::Range {
                start,
                end,
                step,
                num_partitions,
            } => {
                let mut range = proto::Range::default();
                range.start = Some(*start);
                range.end = *end;
                range.step = *step;
                if let Some(n) = num_partitions {
                    range.num_partitions = Some(*n);
                }
                relation.rel_type = Some(proto::relation::RelType::Range(range));
            }

            LogicalPlan::Sql {
                query,
                pos_args,
                named_args,
            } => {
                let mut sql = proto::Sql::default();
                sql.query = query.clone();
                for e in pos_args {
                    sql.pos_arguments.push(e.to_proto());
                }
                for (k, v) in named_args {
                    sql.named_arguments.insert(k.clone(), v.to_proto());
                }
                relation.rel_type = Some(proto::relation::RelType::Sql(sql));
            }

            LogicalPlan::Project { input, columns } => {
                let mut project = proto::Project::default();
                project.input = Some(Box::new(relation_of(input, plan_ids)));
                for col in columns {
                    project.expressions.push(col.to_proto());
                }
                relation.rel_type = Some(proto::relation::RelType::Project(Box::new(project)));
            }

            LogicalPlan::Filter { input, condition } => {
                let mut filter = proto::Filter::default();
                filter.input = Some(Box::new(relation_of(input, plan_ids)));
                filter.condition = Some(condition.to_proto());
                relation.rel_type = Some(proto::relation::RelType::Filter(Box::new(filter)));
            }

            LogicalPlan::Aggregate {
                input,
                group_type,
                grouping_expressions,
                aggregate_expressions,
                pivot_col,
                pivot_values,
                grouping_sets,
            } => {
                let mut agg = proto::Aggregate::default();
                agg.input = Some(Box::new(relation_of(input, plan_ids)));
                agg.group_type = match group_type {
                    AggregateGroupType::GroupBy => proto::aggregate::GroupType::Groupby as i32,
                    AggregateGroupType::Rollup => proto::aggregate::GroupType::Rollup as i32,
                    AggregateGroupType::Cube => proto::aggregate::GroupType::Cube as i32,
                    AggregateGroupType::Pivot => proto::aggregate::GroupType::Pivot as i32,
                    AggregateGroupType::GroupingSets => {
                        proto::aggregate::GroupType::GroupingSets as i32
                    }
                };
                for expr in grouping_expressions {
                    agg.grouping_expressions.push(expr.to_proto());
                }
                // Explicit grouping sets (group_type == GroupingSets). Each set is
                // serialized as its own `GroupingSets` message; without this the sets
                // would be lost and the query would degrade to a plain group-by.
                for set in grouping_sets {
                    let mut gs = proto::aggregate::GroupingSets::default();
                    for e in set {
                        gs.grouping_set.push(e.to_proto());
                    }
                    agg.grouping_sets.push(gs);
                }
                for expr in aggregate_expressions {
                    agg.aggregate_expressions.push(expr.to_proto());
                }
                if let Some(pcol) = pivot_col {
                    let mut pivot = proto::aggregate::Pivot::default();
                    pivot.col = Some(pcol.to_proto());
                    // Serialize explicit pivot values (literals). When empty, the server
                    // computes the distinct values itself.
                    for v in pivot_values {
                        if let Some(proto::expression::ExprType::Literal(lit)) =
                            v.to_proto().expr_type
                        {
                            pivot.values.push(lit);
                        }
                    }
                    agg.pivot = Some(pivot);
                }
                relation.rel_type = Some(proto::relation::RelType::Aggregate(Box::new(agg)));
            }

            LogicalPlan::Join {
                left,
                right,
                join_type,
                on,
                using_columns,
            } => {
                let mut join = proto::Join::default();
                join.left = Some(Box::new(relation_of(left, plan_ids)));
                join.right = Some(Box::new(relation_of(right, plan_ids)));
                join.join_type = match join_type {
                    JoinType::Inner => proto::join::JoinType::Inner as i32,
                    JoinType::LeftOuter => proto::join::JoinType::LeftOuter as i32,
                    JoinType::RightOuter => proto::join::JoinType::RightOuter as i32,
                    JoinType::FullOuter => proto::join::JoinType::FullOuter as i32,
                    JoinType::LeftSemi => proto::join::JoinType::LeftSemi as i32,
                    JoinType::LeftAnti => proto::join::JoinType::LeftAnti as i32,
                    JoinType::Cross => proto::join::JoinType::Cross as i32,
                };
                if let Some(condition) = on {
                    join.join_condition = Some(condition.to_proto());
                }
                join.using_columns.extend(using_columns.clone());
                relation.rel_type = Some(proto::relation::RelType::Join(Box::new(join)));
            }

            LogicalPlan::AsOfJoin {
                left,
                right,
                left_as_of,
                right_as_of,
                on,
                using_columns,
                join_type,
                tolerance,
                allow_exact_matches,
                direction,
            } => {
                let aj = proto::AsOfJoin {
                    left: Some(Box::new(relation_of(left, plan_ids))),
                    right: Some(Box::new(relation_of(right, plan_ids))),
                    left_as_of: Some(left_as_of.to_proto()),
                    right_as_of: Some(right_as_of.to_proto()),
                    join_expr: on.as_ref().map(|condition| condition.to_proto()),
                    using_columns: using_columns.clone(),
                    join_type: join_type.as_str().to_string(),
                    tolerance: tolerance.as_ref().map(|tolerance| tolerance.to_proto()),
                    allow_exact_matches: *allow_exact_matches,
                    direction: direction.as_str().to_string(),
                };
                relation.rel_type = Some(proto::relation::RelType::AsOfJoin(Box::new(aj)));
            }

            LogicalPlan::LateralJoin {
                left,
                right,
                join_type,
                on,
            } => {
                let mut lj = proto::LateralJoin::default();
                lj.left = Some(Box::new(relation_of(left, plan_ids)));
                lj.right = Some(Box::new(relation_of(right, plan_ids)));
                lj.join_type = match join_type {
                    JoinType::Inner => proto::join::JoinType::Inner as i32,
                    JoinType::LeftOuter => proto::join::JoinType::LeftOuter as i32,
                    JoinType::RightOuter => proto::join::JoinType::RightOuter as i32,
                    JoinType::FullOuter => proto::join::JoinType::FullOuter as i32,
                    JoinType::LeftSemi => proto::join::JoinType::LeftSemi as i32,
                    JoinType::LeftAnti => proto::join::JoinType::LeftAnti as i32,
                    JoinType::Cross => proto::join::JoinType::Cross as i32,
                };
                if let Some(condition) = on {
                    lj.join_condition = Some(condition.to_proto());
                }
                relation.rel_type = Some(proto::relation::RelType::LateralJoin(Box::new(lj)));
            }

            LogicalPlan::SetOperation {
                left,
                right,
                set_op_type,
                is_all,
                by_name,
                allow_missing_columns,
            } => {
                let mut set_op = proto::SetOperation::default();
                set_op.left_input = Some(Box::new(relation_of(left, plan_ids)));
                set_op.right_input = Some(Box::new(relation_of(right, plan_ids)));
                set_op.set_op_type = match set_op_type {
                    SetOpType::Union => proto::set_operation::SetOpType::Union as i32,
                    SetOpType::Intersect => proto::set_operation::SetOpType::Intersect as i32,
                    SetOpType::Except => proto::set_operation::SetOpType::Except as i32,
                };
                set_op.is_all = Some(*is_all);
                set_op.by_name = Some(*by_name);
                set_op.allow_missing_columns = Some(*allow_missing_columns);
                relation.rel_type = Some(proto::relation::RelType::SetOp(Box::new(set_op)));
            }

            LogicalPlan::Limit { input, limit } => {
                let mut lim = proto::Limit::default();
                lim.input = Some(Box::new(relation_of(input, plan_ids)));
                lim.limit = *limit;
                relation.rel_type = Some(proto::relation::RelType::Limit(Box::new(lim)));
            }

            LogicalPlan::Offset { input, offset } => {
                let mut off = proto::Offset::default();
                off.input = Some(Box::new(relation_of(input, plan_ids)));
                off.offset = *offset;
                relation.rel_type = Some(proto::relation::RelType::Offset(Box::new(off)));
            }

            LogicalPlan::Tail { input, limit } => {
                let mut tail = proto::Tail::default();
                tail.input = Some(Box::new(relation_of(input, plan_ids)));
                tail.limit = *limit;
                relation.rel_type = Some(proto::relation::RelType::Tail(Box::new(tail)));
            }

            LogicalPlan::ShowString {
                input,
                num_rows,
                truncate,
                vertical,
            } => {
                let show = proto::ShowString {
                    input: Some(Box::new(relation_of(input, plan_ids))),
                    num_rows: *num_rows,
                    truncate: *truncate,
                    vertical: *vertical,
                };
                relation.rel_type = Some(proto::relation::RelType::ShowString(Box::new(show)));
            }

            LogicalPlan::HtmlString {
                input,
                num_rows,
                truncate,
            } => {
                let html = proto::HtmlString {
                    input: Some(Box::new(relation_of(input, plan_ids))),
                    num_rows: *num_rows,
                    truncate: *truncate,
                };
                relation.rel_type = Some(proto::relation::RelType::HtmlString(Box::new(html)));
            }

            LogicalPlan::Raw {
                relation: raw,
                depth,
                holes,
                ..
            } => {
                // The clone recurses for each level of the relation: grow the stack for them all.
                let stack = raw_relation_stack(*depth);
                relation = if holes.is_empty() {
                    stacker::maybe_grow(stack, stack, || raw.as_ref().clone())
                } else {
                    // Of the messages that the picklers of the holes give now, in the order of
                    // their build, as PySpark's unpickled plan pickles the functions of its UDFs.
                    let fill = crate::udf::SENT_FILL_STACK;
                    match stacker::maybe_grow(fill, fill, || {
                        use prost::Message;
                        holes
                            .fill(|| raw.encode_to_vec())
                            .and_then(|filled| match filled {
                                Some(filled) => proto::Relation::decode(filled.as_slice())
                                    .map_err(|error| SparkError::connect_msg(error.to_string())),
                                // The first build of a request, which meets the picklers.
                                None => Ok(raw.as_ref().clone()),
                            })
                    }) {
                        Ok(filled) => filled,
                        Err(error) => {
                            crate::expression::record_unbuildable(&error);
                            stacker::maybe_grow(stack, stack, || raw.as_ref().clone())
                        }
                    }
                };
                relation.common.get_or_insert_with(Default::default);
            }

            LogicalPlan::Deduplicate {
                input,
                all_columns_as_keys,
                column_names,
                within_watermark,
            } => {
                let mut dedup = proto::Deduplicate::default();
                dedup.input = Some(Box::new(relation_of(input, plan_ids)));
                dedup.all_columns_as_keys = Some(*all_columns_as_keys);
                dedup.column_names.extend(column_names.clone());
                dedup.within_watermark = Some(*within_watermark);
                relation.rel_type = Some(proto::relation::RelType::Deduplicate(Box::new(dedup)));
            }

            LogicalPlan::Sort {
                input,
                order,
                is_global,
            } => {
                // Use the generated proto enum constants rather than bare 1/2 magic
                // numbers, so a future proto reorder can't silently change meaning.
                use proto::expression::sort_order::{NullOrdering as PbNulls, SortDirection};
                let mut sort = proto::Sort::default();
                sort.input = Some(Box::new(relation_of(input, plan_ids)));
                for expr in order {
                    // Extract SortOrder from Expression if it's SortOrder type
                    if let ExpressionNode::SortOrder(so) = expr.node() {
                        let mut sort_order = proto::expression::SortOrder::default();
                        sort_order.child = Some(Box::new(so.child.to_proto()));
                        sort_order.direction = if so.ascending {
                            SortDirection::Ascending as i32
                        } else {
                            SortDirection::Descending as i32
                        };
                        sort_order.null_ordering = match so.null_ordering {
                            crate::expression::NullOrdering::First => {
                                PbNulls::SortNullsFirst as i32
                            }
                            crate::expression::NullOrdering::Last => PbNulls::SortNullsLast as i32,
                        };
                        sort.order.push(sort_order);
                    } else {
                        let expr_proto = expr.to_proto();
                        if let Some(proto::expression::ExprType::SortOrder(so)) =
                            expr_proto.expr_type
                        {
                            sort.order.push(*so);
                        } else {
                            // A bare column (not a SortOrder): default to ascending,
                            // nulls first - matching reference pyspark's sort/orderBy.
                            // Without this the column was dropped, leaving an empty
                            // order and an invalid Sort plan.
                            let mut sort_order = proto::expression::SortOrder::default();
                            sort_order.child = Some(Box::new(expr_proto));
                            sort_order.direction = SortDirection::Ascending as i32;
                            sort_order.null_ordering = PbNulls::SortNullsFirst as i32;
                            sort.order.push(sort_order);
                        }
                    }
                }
                sort.is_global = Some(*is_global);
                relation.rel_type = Some(proto::relation::RelType::Sort(Box::new(sort)));
            }

            LogicalPlan::Sample {
                input,
                lower_bound,
                upper_bound,
                with_replacement,
                seed,
                deterministic_order,
            } => {
                let mut sample = proto::Sample::default();
                sample.input = Some(Box::new(relation_of(input, plan_ids)));
                sample.lower_bound = *lower_bound;
                sample.upper_bound = *upper_bound;
                sample.with_replacement = Some(*with_replacement);
                if let Some(s) = seed {
                    sample.seed = Some(*s);
                }
                sample.deterministic_order = *deterministic_order;
                relation.rel_type = Some(proto::relation::RelType::Sample(Box::new(sample)));
            }

            LogicalPlan::Repartition {
                input,
                num_partitions,
                shuffle,
            } => {
                let mut repart = proto::Repartition::default();
                repart.input = Some(Box::new(relation_of(input, plan_ids)));
                repart.num_partitions = *num_partitions;
                repart.shuffle = Some(*shuffle);
                relation.rel_type = Some(proto::relation::RelType::Repartition(Box::new(repart)));
            }

            LogicalPlan::RepartitionByExpression {
                input,
                num_partitions,
                expressions,
            } => {
                let mut repart = proto::RepartitionByExpression::default();
                // The partition expressions before the input, as PySpark's plan builds them, so
                // that an expression that a request fails to build raises its error first.
                for expr in expressions {
                    repart.partition_exprs.push(expr.to_proto());
                }
                repart.input = Some(Box::new(relation_of(input, plan_ids)));
                // num_partitions is optional in the proto: <= 0 means "unset" so the
                // server uses the default (the `df.repartition(*cols)` column-only form,
                // where no partition count is given). A real count is always positive.
                repart.num_partitions = if *num_partitions > 0 {
                    Some(*num_partitions)
                } else {
                    None
                };
                relation.rel_type = Some(proto::relation::RelType::RepartitionByExpression(
                    Box::new(repart),
                ));
            }

            LogicalPlan::WithColumns {
                input,
                column_names,
                columns,
            } => {
                let mut wc = proto::WithColumns::default();
                wc.input = Some(Box::new(relation_of(input, plan_ids)));
                for (name, col) in column_names.iter().zip(columns.iter()) {
                    // Create an Alias expression for each column
                    let mut alias = proto::expression::Alias::default();
                    alias.expr = Some(Box::new(col.to_proto()));
                    alias.name = vec![name.clone()];
                    wc.aliases.push(alias);
                }
                relation.rel_type = Some(proto::relation::RelType::WithColumns(Box::new(wc)));
            }

            LogicalPlan::WithColumnMetadata {
                input,
                column_name,
                column,
                metadata_json,
            } => {
                let mut wc = proto::WithColumns::default();
                wc.input = Some(Box::new(relation_of(input, plan_ids)));
                wc.aliases.push(proto::expression::Alias {
                    expr: Some(Box::new(column.to_proto())),
                    name: vec![column_name.clone()],
                    metadata: Some(metadata_json.clone()),
                });
                relation.rel_type = Some(proto::relation::RelType::WithColumns(Box::new(wc)));
            }

            LogicalPlan::WithColumnsRenamed { input, renames } => {
                let mut wcr = proto::WithColumnsRenamed::default();
                wcr.input = Some(Box::new(relation_of(input, plan_ids)));
                for (old_name, new_name) in renames.iter() {
                    let mut rename = proto::with_columns_renamed::Rename::default();
                    rename.col_name = old_name.clone();
                    rename.new_col_name = new_name.clone();
                    wcr.renames.push(rename);
                }
                relation.rel_type =
                    Some(proto::relation::RelType::WithColumnsRenamed(Box::new(wcr)));
            }

            LogicalPlan::Drop {
                input,
                columns,
                expressions,
            } => {
                let mut drop = proto::Drop::default();
                drop.input = Some(Box::new(relation_of(input, plan_ids)));
                drop.columns
                    .extend(expressions.iter().map(|e| e.to_proto()));
                drop.column_names.extend(columns.clone());
                relation.rel_type = Some(proto::relation::RelType::Drop(Box::new(drop)));
            }

            LogicalPlan::ToDF {
                input,
                column_names,
            } => {
                let mut to_df = proto::ToDf::default();
                to_df.input = Some(Box::new(relation_of(input, plan_ids)));
                to_df.column_names.extend(column_names.clone());
                relation.rel_type = Some(proto::relation::RelType::ToDf(Box::new(to_df)));
            }

            LogicalPlan::Unbuildable { input, error } => {
                // The input, as PySpark builds it first, and then the error, which the request
                // raises; the relation is not sent.
                relation.rel_type = relation_of(input, plan_ids).rel_type;
                crate::expression::record_unbuildable(error);
            }

            LogicalPlan::ToSchema {
                input,
                schema,
                schema_reader,
            } => {
                let mut to_schema = proto::ToSchema::default();
                to_schema.input = Some(Box::new(relation_of(input, plan_ids)));
                // As PySpark's, which converts the schema once it built the input: of the reader,
                // the schema that it gives, else `schema`, as before the reader gives one.
                let read = schema_reader
                    .as_ref()
                    .map(crate::expression::pickled_command)
                    .filter(|read| !read.is_empty())
                    .and_then(|read| {
                        use prost::Message;
                        proto::DataType::decode(read.as_slice()).ok()
                    });
                to_schema.schema = read.or_else(|| Some(schema.to_proto()));
                relation.rel_type = Some(proto::relation::RelType::ToSchema(Box::new(to_schema)));
            }

            LogicalPlan::Hint {
                input,
                name,
                parameters,
            } => {
                let mut hint = proto::Hint::default();
                hint.input = Some(Box::new(relation_of(input, plan_ids)));
                hint.name = name.clone();
                hint.parameters
                    .extend(parameters.iter().map(|p| p.to_proto()));
                relation.rel_type = Some(proto::relation::RelType::Hint(Box::new(hint)));
            }

            LogicalPlan::Parse {
                input,
                format,
                schema,
                options,
            } => {
                relation.rel_type = Some(proto::relation::RelType::Parse(Box::new(proto::Parse {
                    input: Some(Box::new(relation_of(input, plan_ids))),
                    format: format.to_proto() as i32,
                    schema: schema.as_ref().map(|s| s.to_proto()),
                    options: options.clone(),
                })));
            }

            LogicalPlan::Unpivot {
                input,
                ids,
                values,
                variable_column_name,
                value_column_name,
            } => {
                let mut unpivot = proto::Unpivot::default();
                unpivot.input = Some(Box::new(relation_of(input, plan_ids)));
                for col in ids {
                    unpivot.ids.push(col.to_proto());
                }
                if let Some(v) = values {
                    let mut vals = proto::unpivot::Values::default();
                    for col in v {
                        vals.values.push(col.to_proto());
                    }
                    unpivot.values = Some(vals);
                }
                unpivot.variable_column_name = variable_column_name.clone();
                unpivot.value_column_name = value_column_name.clone();
                relation.rel_type = Some(proto::relation::RelType::Unpivot(Box::new(unpivot)));
            }

            LogicalPlan::NAFill {
                input,
                fill_value,
                columns,
            } => {
                let mut na_fill = proto::NaFill::default();
                na_fill.input = Some(Box::new(relation_of(input, plan_ids)));
                na_fill.cols.extend(columns.clone());
                na_fill.values.push(value_to_proto_literal(fill_value));
                relation.rel_type = Some(proto::relation::RelType::FillNa(Box::new(na_fill)));
            }

            LogicalPlan::NAFillColumns {
                input,
                cols,
                values,
            } => {
                // Per-column fill: cols[i] is filled with values[i] (Spark aligns them
                // positionally when both are non-empty).
                let mut na_fill = proto::NaFill::default();
                na_fill.input = Some(Box::new(relation_of(input, plan_ids)));
                na_fill.cols.extend(cols.clone());
                na_fill
                    .values
                    .extend(values.iter().map(value_to_proto_literal));
                relation.rel_type = Some(proto::relation::RelType::FillNa(Box::new(na_fill)));
            }

            LogicalPlan::NADrop {
                input,
                how,
                min_non_null,
                columns,
            } => {
                let mut na_drop = proto::NaDrop::default();
                na_drop.input = Some(Box::new(relation_of(input, plan_ids)));
                na_drop.cols.extend(columns.clone());
                // Mirror pyspark: an explicit `thresh` (min_non_null) wins; otherwise
                // `how="all"` -> min_non_nulls=1 and `how="any"` -> unset (server default).
                // Previously `how` was dropped (`how: _`), so "all" and "any" built
                // byte-identical plans and both behaved as "any".
                na_drop.min_non_nulls = match min_non_null {
                    Some(m) => Some(*m),
                    None if how == "all" => Some(1),
                    None => None,
                };
                relation.rel_type = Some(proto::relation::RelType::DropNa(Box::new(na_drop)));
            }

            LogicalPlan::NAReplace {
                input,
                replacements,
                columns,
            } => {
                let mut na_replace = proto::NaReplace::default();
                na_replace.input = Some(Box::new(relation_of(input, plan_ids)));
                for (old_val, new_val) in replacements.iter() {
                    na_replace
                        .replacements
                        .push(proto::na_replace::Replacement {
                            old_value: Some(old_val.to_literal_proto()),
                            new_value: Some(new_val.to_literal_proto()),
                        });
                }
                na_replace.cols.extend(columns.clone());
                relation.rel_type = Some(proto::relation::RelType::Replace(Box::new(na_replace)));
            }

            LogicalPlan::Describe { input, columns } => {
                let mut describe = proto::StatDescribe::default();
                describe.input = Some(Box::new(relation_of(input, plan_ids)));
                describe.cols.extend(columns.clone());
                relation.rel_type = Some(proto::relation::RelType::Describe(Box::new(describe)));
            }

            LogicalPlan::Summary { input, percentiles } => {
                let mut summary = proto::StatSummary::default();
                summary.input = Some(Box::new(relation_of(input, plan_ids)));
                summary.statistics.extend(percentiles.clone());
                relation.rel_type = Some(proto::relation::RelType::Summary(Box::new(summary)));
            }

            LogicalPlan::ColRegex { input, col_name } => {
                // ColRegex is implemented as a Project with UnresolvedRegex expressions
                let mut project = proto::Project::default();
                project.input = Some(Box::new(relation_of(input, plan_ids)));
                let mut expr = proto::Expression::default();
                expr.expr_type = Some(proto::expression::ExprType::UnresolvedRegex(
                    proto::expression::UnresolvedRegex {
                        col_name: col_name.clone(),
                        plan_id: None,
                    },
                ));
                project.expressions.push(expr);
                relation.rel_type = Some(proto::relation::RelType::Project(Box::new(project)));
            }

            LogicalPlan::SubqueryAlias { input, alias } => {
                let mut sq_alias = proto::SubqueryAlias::default();
                sq_alias.input = Some(Box::new(relation_of(input, plan_ids)));
                sq_alias.alias = alias.clone();
                relation.rel_type =
                    Some(proto::relation::RelType::SubqueryAlias(Box::new(sq_alias)));
            }

            LogicalPlan::WithRelations { root, references } => {
                let with_relations = proto::WithRelations {
                    root: Some(Box::new(relation_of(root, plan_ids))),
                    references: references
                        .iter()
                        .map(|reference| relation_of(reference, plan_ids))
                        .collect(),
                };
                relation.rel_type = Some(proto::relation::RelType::WithRelations(Box::new(
                    with_relations,
                )));
            }

            LogicalPlan::ChunkedCachedLocalRelation {
                data_hashes,
                schema_hash,
            } => {
                relation.rel_type = Some(proto::relation::RelType::ChunkedCachedLocalRelation(
                    proto::ChunkedCachedLocalRelation {
                        data_hashes: data_hashes.clone(),
                        schema_hash: schema_hash.clone(),
                    },
                ));
            }
            LogicalPlan::LocalRelation { schema, data } => {
                let mut local = proto::LocalRelation::default();
                if let Some(d) = data {
                    local.data = Some(d.clone().into());
                }
                // Explicit schemas apply server-side casts. Inferred Arrow input
                // omits this to avoid a redundant projection on every query. The JSON is
                // PySpark's, of the types that the Python client lowered too.
                local.schema = schema.as_ref().map(DataType::json_as_pyspark);
                relation.rel_type = Some(proto::relation::RelType::LocalRelation(local));
            }

            LogicalPlan::CachedRemoteRelation { relation_id, .. } => {
                let mut cached = proto::CachedRemoteRelation::default();
                cached.relation_id = relation_id.clone();
                relation.rel_type = Some(proto::relation::RelType::CachedRemoteRelation(cached));
            }

            LogicalPlan::CachedRelation { relation: returned } => {
                // Keep the relation as the server returned it; only its plan ID is this node's.
                let common = relation.common.take();
                relation = (**returned).clone();
                relation.common = common;
            }

            LogicalPlan::Read {
                read_type,
                is_streaming,
            } => {
                let mut read = proto::Read::default();
                read.is_streaming = *is_streaming;

                match read_type {
                    crate::readwriter::ReadType::DataSource {
                        format,
                        schema,
                        options,
                        paths,
                        predicates,
                        source_name,
                    } => {
                        let mut data_source = proto::read::DataSource::default();
                        if let Some(fmt) = format {
                            data_source.format = Some(fmt.clone());
                        }
                        if let Some(sch) = schema {
                            data_source.schema = Some(sch.clone());
                        }
                        data_source.options.extend(options.clone());
                        data_source.paths.extend(paths.clone());
                        data_source.predicates.extend(predicates.clone());
                        if let Some(sn) = source_name {
                            data_source.source_name = Some(sn.clone());
                        }
                        read.read_type = Some(proto::read::ReadType::DataSource(data_source));
                    }
                    crate::readwriter::ReadType::NamedTable {
                        table_name,
                        options,
                    } => {
                        let mut named_table = proto::read::NamedTable::default();
                        named_table.unparsed_identifier = table_name.clone();
                        named_table.options.extend(options.clone());
                        read.read_type = Some(proto::read::ReadType::NamedTable(named_table));
                    }
                }

                relation.rel_type = Some(proto::relation::RelType::Read(read));
            }

            LogicalPlan::RelationChanges {
                table_name,
                options,
                is_streaming,
            } => {
                let mut changes = proto::RelationChanges::default();
                changes.unparsed_identifier = table_name.clone();
                changes.options.extend(options.clone());
                if let Some(s) = is_streaming {
                    changes.is_streaming = *s;
                }
                relation.rel_type = Some(proto::relation::RelType::RelationChanges(changes));
            }

            LogicalPlan::WithWatermark {
                input,
                time_column,
                delay_threshold,
            } => {
                let mut watermark = proto::WithWatermark::default();
                watermark.input = Some(Box::new(relation_of(input, plan_ids)));
                watermark.event_time = time_column.clone();
                watermark.delay_threshold = delay_threshold.clone();
                relation.rel_type =
                    Some(proto::relation::RelType::WithWatermark(Box::new(watermark)));
            }

            LogicalPlan::RepartitionByRange {
                input,
                num_partitions,
                partition_exprs,
            } => {
                let mut repart = proto::RepartitionByExpression::default();
                // The partition expressions before the input, as PySpark's plan builds them.
                for expr in partition_exprs {
                    repart.partition_exprs.push(expr.to_proto());
                }
                repart.input = Some(Box::new(relation_of(input, plan_ids)));
                if let Some(n) = num_partitions {
                    repart.num_partitions = Some(*n);
                }
                relation.rel_type = Some(proto::relation::RelType::RepartitionByExpression(
                    Box::new(repart),
                ));
            }

            LogicalPlan::StatCrosstab { input, col1, col2 } => {
                let mut stat = proto::StatCrosstab::default();
                stat.input = Some(Box::new(relation_of(input, plan_ids)));
                stat.col1 = col1.clone();
                stat.col2 = col2.clone();
                relation.rel_type = Some(proto::relation::RelType::Crosstab(Box::new(stat)));
            }

            LogicalPlan::StatFreqItems {
                input,
                columns,
                support,
            } => {
                let mut stat = proto::StatFreqItems::default();
                stat.input = Some(Box::new(relation_of(input, plan_ids)));
                stat.cols.extend(columns.clone());
                stat.support = Some(*support);
                relation.rel_type = Some(proto::relation::RelType::FreqItems(Box::new(stat)));
            }

            LogicalPlan::StatApproxQuantile {
                input,
                columns,
                probabilities,
                relative_error,
            } => {
                let mut stat = proto::StatApproxQuantile::default();
                stat.input = Some(Box::new(relation_of(input, plan_ids)));
                stat.cols.extend(columns.clone());
                stat.probabilities.extend(probabilities.clone());
                stat.relative_error = *relative_error;
                relation.rel_type = Some(proto::relation::RelType::ApproxQuantile(Box::new(stat)));
            }

            LogicalPlan::StatCorr {
                input,
                col1,
                col2,
                method,
            } => {
                let mut stat = proto::StatCorr::default();
                stat.input = Some(Box::new(relation_of(input, plan_ids)));
                stat.col1 = col1.clone();
                stat.col2 = col2.clone();
                stat.method = Some(method.clone());
                relation.rel_type = Some(proto::relation::RelType::Corr(Box::new(stat)));
            }

            LogicalPlan::StatCov { input, col1, col2 } => {
                let mut stat = proto::StatCov::default();
                stat.input = Some(Box::new(relation_of(input, plan_ids)));
                stat.col1 = col1.clone();
                stat.col2 = col2.clone();
                relation.rel_type = Some(proto::relation::RelType::Cov(Box::new(stat)));
            }

            LogicalPlan::StatSampleBy {
                input,
                col,
                fractions,
                seed,
            } => {
                let mut stat = proto::StatSampleBy::default();
                stat.input = Some(Box::new(relation_of(input, plan_ids)));
                stat.col = Some(col.to_proto());
                // As PySpark's: each stratum, a literal, with its fraction.
                for (expr, fraction) in fractions {
                    let stratum = match expr.to_proto().expr_type {
                        Some(proto::expression::ExprType::Literal(lit)) => Some(lit),
                        _ => None,
                    };
                    stat.fractions.push(proto::stat_sample_by::Fraction {
                        stratum,
                        fraction: *fraction,
                    });
                }
                if let Some(s) = seed {
                    stat.seed = Some(*s);
                }
                relation.rel_type = Some(proto::relation::RelType::SampleBy(Box::new(stat)));
            }

            LogicalPlan::Observe {
                input,
                name,
                exprs,
                observation,
            } => {
                let mut collect_metrics = proto::CollectMetrics::default();
                collect_metrics.input = Some(Box::new(relation_of(input, plan_ids)));
                // As PySpark's `CollectMetrics`, which sets the `str` of the `_name` of its
                // Observation when it builds the plan, after its input: its error fails the
                // request.
                collect_metrics.name = match observed_name(name, observation.as_ref()) {
                    Ok(name) => name,
                    Err(error) => {
                        crate::expression::record_unbuildable(&error);
                        name.clone()
                    }
                };
                for expr in exprs {
                    collect_metrics.metrics.push(expr.to_proto());
                }
                relation.rel_type = Some(proto::relation::RelType::CollectMetrics(Box::new(
                    collect_metrics,
                )));
            }

            LogicalPlan::UnresolvedTableValuedFunction { name, arguments } => {
                let mut tvf = proto::UnresolvedTableValuedFunction::default();
                tvf.function_name = name.clone();
                for arg in arguments {
                    tvf.arguments.push(arg.to_proto());
                }
                relation.rel_type =
                    Some(proto::relation::RelType::UnresolvedTableValuedFunction(tvf));
            }

            LogicalPlan::Zip { left, right } => {
                let mut zip = proto::Zip::default();
                zip.left = Some(Box::new(relation_of(left, plan_ids)));
                zip.right = Some(Box::new(relation_of(right, plan_ids)));
                relation.rel_type = Some(proto::relation::RelType::Zip(Box::new(zip)));
            }

            LogicalPlan::MlTransform { ml_relation, .. } => {
                return ml_relation.clone();
            }
            LogicalPlan::Catalog { catalog } => {
                relation.rel_type = Some(proto::relation::RelType::Catalog(catalog.clone()));
                return relation;
            }
            LogicalPlan::Transpose {
                input,
                index_columns,
            } => {
                let mut transpose = proto::Transpose::default();
                transpose.input = Some(Box::new(relation_of(input, plan_ids)));
                transpose.index_columns = index_columns.iter().map(|e| e.to_proto()).collect();
                relation.rel_type = Some(proto::relation::RelType::Transpose(Box::new(transpose)));
                return relation;
            }
            LogicalPlan::MapPartitions {
                input,
                func,
                is_barrier,
                profile_id,
                profile_id_reader,
            } => {
                let mut mp = proto::MapPartitions::default();
                mp.input = Some(Box::new(relation_of(input, plan_ids)));
                mp.func = Some(func.to_proto());
                mp.is_barrier = Some(*is_barrier);
                // As PySpark's, which reads `profile.id` once it set the function and the flag.
                mp.profile_id = match profile_id_reader {
                    Some(reader) => <[u8; 4]>::try_from(crate::expression::pickled_command(reader))
                        .ok()
                        .map(i32::from_le_bytes),
                    None => *profile_id,
                };
                relation.rel_type = Some(proto::relation::RelType::MapPartitions(Box::new(mp)));
            }
            LogicalPlan::GroupMap {
                input,
                grouping_expressions,
                func,
                sorting_expressions,
                initial_input,
                initial_grouping_expressions,
                is_map_groups_with_state,
                output_mode,
                timeout_conf,
                state_schema,
                transform_with_state_info,
            } => {
                let mut gm = proto::GroupMap::default();
                gm.input = Some(Box::new(relation_of(input, plan_ids)));
                for e in grouping_expressions {
                    gm.grouping_expressions.push(e.to_proto());
                }
                // As PySpark's transformWithState plan, which builds the initial state before the
                // function, whose command it pickles then.
                if let Some(ii) = initial_input {
                    gm.initial_input = Some(Box::new(relation_of(ii, plan_ids)));
                }
                for e in initial_grouping_expressions {
                    gm.initial_grouping_expressions.push(e.to_proto());
                }
                gm.func = Some(func.to_proto());
                for e in sorting_expressions {
                    gm.sorting_expressions.push(e.to_proto());
                }
                gm.is_map_groups_with_state = *is_map_groups_with_state;
                gm.output_mode = output_mode.clone();
                gm.timeout_conf = timeout_conf.clone();
                gm.state_schema = state_schema.as_ref().map(|d| d.to_proto());
                gm.transform_with_state_info =
                    transform_with_state_info
                        .as_ref()
                        .map(|t| proto::TransformWithStateInfo {
                            time_mode: t.time_mode.clone(),
                            event_time_column_name: t.event_time_column_name.clone(),
                            output_schema: transform_with_state_output_schema(
                                t,
                                func,
                                gm.func.as_ref(),
                            ),
                        });
                relation.rel_type = Some(proto::relation::RelType::GroupMap(Box::new(gm)));
            }
            LogicalPlan::ApplyInPandasWithState {
                input,
                grouping_expressions,
                func,
                output_schema,
                state_schema,
                output_mode,
                timeout_conf,
            } => {
                let apply = proto::ApplyInPandasWithState {
                    input: Some(Box::new(relation_of(input, plan_ids))),
                    grouping_expressions: grouping_expressions
                        .iter()
                        .map(|e| e.to_proto())
                        .collect(),
                    func: Some(func.to_proto()),
                    output_schema: output_schema.clone(),
                    state_schema: state_schema.clone(),
                    output_mode: output_mode.clone(),
                    timeout_conf: timeout_conf.clone(),
                };
                relation.rel_type = Some(proto::relation::RelType::ApplyInPandasWithState(
                    Box::new(apply),
                ));
            }
            LogicalPlan::CoGroupMap {
                input,
                input_grouping_expressions,
                other,
                other_grouping_expressions,
                func,
            } => {
                let mut cg = proto::CoGroupMap::default();
                cg.input = Some(Box::new(relation_of(input, plan_ids)));
                for e in input_grouping_expressions {
                    cg.input_grouping_expressions.push(e.to_proto());
                }
                cg.other = Some(Box::new(relation_of(other, plan_ids)));
                for e in other_grouping_expressions {
                    cg.other_grouping_expressions.push(e.to_proto());
                }
                cg.func = Some(func.to_proto());
                relation.rel_type = Some(proto::relation::RelType::CoGroupMap(Box::new(cg)));
            }
            LogicalPlan::NearestByJoin {
                left,
                right,
                ranking_expression,
                num_results,
                join_type,
                mode,
                direction,
            } => {
                let mut nbj = proto::NearestByJoin::default();
                nbj.left = Some(Box::new(relation_of(left, plan_ids)));
                nbj.right = Some(Box::new(relation_of(right, plan_ids)));
                nbj.ranking_expression = Some(ranking_expression.to_proto());
                nbj.num_results = *num_results;
                nbj.join_type = join_type.clone();
                nbj.mode = mode.clone();
                nbj.direction = direction.clone();
                relation.rel_type = Some(proto::relation::RelType::NearestByJoin(Box::new(nbj)));
            }
            LogicalPlan::CommonInlineUdtf {
                function_name,
                deterministic,
                arguments,
                return_type,
                eval_type,
                command,
                python_ver,
                pickler,
            } => {
                let f = common_inline_udtf(
                    function_name,
                    *deterministic,
                    arguments,
                    return_type.as_ref(),
                    *eval_type,
                    (command, pickler.as_ref()),
                    python_ver,
                );
                relation.rel_type =
                    Some(proto::relation::RelType::CommonInlineUserDefinedTableFunction(f));
            }
        }

        relation
    }
}

/// The output schema of the `TransformWithStateInfo` `info` of a plan of the function `func`,
/// whose build is `built`. As PySpark's `BaseTransformWithStateInPySpark.plan` converts the output
/// schema when it builds the plan, of the type object that the output type of the function is of
/// too: a schema that is the output type of a function whose pickler converts that type as the
/// request builds the plan ([`crate::udf::CommandPickler::with_output_type`]) is the type of that
/// build, so a type changed in place since the call is sent as it is then, in both. A DDL string,
/// sent as it is, is not the output type of the function, which is the type that it parses to.
fn transform_with_state_output_schema(
    info: &TransformWithStateInfo,
    func: &CommonInlineUserDefinedFunctionExpression,
    built: Option<&proto::CommonInlineUserDefinedFunction>,
) -> Option<proto::DataType> {
    let of_build = func
        .python_udf
        .pickler
        .as_ref()
        .is_some_and(crate::udf::CommandPickler::gives_output_type);
    if of_build && info.output_schema.as_ref() == Some(&func.python_udf.output_type) {
        if let Some(proto::common_inline_user_defined_function::Function::PythonUdf(udf)) =
            built.and_then(|built| built.function.as_ref())
        {
            return udf.output_type.clone();
        }
    }
    info.output_schema.as_ref().map(DataType::to_proto)
}

/// The `CommonInlineUserDefinedTableFunction` of a Python UDTF: of a call with `arguments`,
/// or of its registration without any. Mirrors PySpark's `CommonInlineUserDefinedTableFunction`
/// of a `PythonUDTF`, which builds the arguments before the UDTF, whose command is the bytes of
/// `command`, or that of its pickler, pickled as the plan is built, with the return type that the
/// pickler gives then, if it gives one (`crate::udf::udtf_return_type_and_command`).
pub(crate) fn common_inline_udtf(
    function_name: &str,
    deterministic: bool,
    arguments: &[Expression],
    return_type: Option<&DataType>,
    eval_type: i32,
    command: (&[u8], Option<&crate::udf::CommandPickler>),
    python_ver: &str,
) -> proto::CommonInlineUserDefinedTableFunction {
    let arguments = arguments.iter().map(Expression::to_proto).collect();
    let (return_type, command) =
        crate::udf::udtf_return_type_and_command(return_type, command.0, command.1);
    let udtf = proto::PythonUdtf {
        return_type,
        eval_type,
        command,
        python_ver: python_ver.to_string(),
    };
    proto::CommonInlineUserDefinedTableFunction {
        function_name: function_name.to_string(),
        deterministic,
        arguments,
        function: Some(
            proto::common_inline_user_defined_table_function::Function::PythonUdtf(udtf),
        ),
    }
}

/// Create a Range plan.
pub fn range(start: i64, end: i64, step: i64) -> LogicalPlan {
    LogicalPlan::Range {
        start,
        end,
        step,
        num_partitions: None,
    }
}

/// Create a Range plan with num_partitions.
pub fn range_with_partitions(start: i64, end: i64, step: i64, num_partitions: i32) -> LogicalPlan {
    LogicalPlan::Range {
        start,
        end,
        step,
        num_partitions: Some(num_partitions),
    }
}

/// Create a SQL plan.
pub fn sql(query: impl Into<String>) -> LogicalPlan {
    LogicalPlan::Sql {
        query: query.into(),
        pos_args: Vec::new(),
        named_args: HashMap::new(),
    }
}

/// Create a Project plan.
pub fn project<C: Into<Column>>(
    input: impl Into<Arc<LogicalPlan>>,
    columns: impl IntoIterator<Item = C>,
) -> LogicalPlan {
    LogicalPlan::Project {
        input: input.into(),
        columns: columns.into_iter().map(Into::into).collect(),
    }
}

/// Create a Filter plan.
pub fn filter(input: impl Into<Arc<LogicalPlan>>, condition: Column) -> LogicalPlan {
    LogicalPlan::Filter {
        input: input.into(),
        condition,
    }
}

/// Create an Aggregate plan.
pub fn aggregate(
    input: impl Into<Arc<LogicalPlan>>,
    group_type: AggregateGroupType,
    grouping_expressions: Vec<Expression>,
    aggregate_expressions: Vec<Expression>,
) -> LogicalPlan {
    LogicalPlan::Aggregate {
        input: input.into(),
        group_type,
        grouping_expressions,
        aggregate_expressions,
        pivot_col: None,
        pivot_values: vec![],
        grouping_sets: vec![],
    }
}

/// Create an Aggregate plan with pivot.
pub fn aggregate_with_pivot(
    input: impl Into<Arc<LogicalPlan>>,
    group_type: AggregateGroupType,
    grouping_expressions: Vec<Expression>,
    aggregate_expressions: Vec<Expression>,
    pivot_col: Expression,
    pivot_values: Vec<Expression>,
) -> LogicalPlan {
    LogicalPlan::Aggregate {
        input: input.into(),
        group_type,
        grouping_expressions,
        aggregate_expressions,
        pivot_col: Some(pivot_col),
        pivot_values,
        grouping_sets: vec![],
    }
}

/// Create an Aggregate plan with explicit grouping sets.
pub fn aggregate_with_grouping_sets(
    input: impl Into<Arc<LogicalPlan>>,
    grouping_expressions: Vec<Expression>,
    aggregate_expressions: Vec<Expression>,
    grouping_sets: Vec<Vec<Expression>>,
) -> LogicalPlan {
    LogicalPlan::Aggregate {
        input: input.into(),
        group_type: AggregateGroupType::GroupingSets,
        grouping_expressions,
        aggregate_expressions,
        pivot_col: None,
        pivot_values: vec![],
        grouping_sets,
    }
}

/// Create a Join plan.
pub fn join(
    left: impl Into<Arc<LogicalPlan>>,
    right: impl Into<Arc<LogicalPlan>>,
    join_type: JoinType,
    on: Option<Column>,
    using_columns: Vec<String>,
) -> LogicalPlan {
    LogicalPlan::Join {
        left: left.into(),
        right: right.into(),
        join_type,
        on,
        using_columns,
    }
}

/// Create a SetOperation plan.
pub fn set_operation(
    left: impl Into<Arc<LogicalPlan>>,
    right: impl Into<Arc<LogicalPlan>>,
    set_op_type: SetOpType,
    is_all: bool,
    by_name: bool,
    allow_missing_columns: bool,
) -> LogicalPlan {
    LogicalPlan::SetOperation {
        left: left.into(),
        right: right.into(),
        set_op_type,
        is_all,
        by_name,
        allow_missing_columns,
    }
}

/// Create a Limit plan.
pub fn limit(input: impl Into<Arc<LogicalPlan>>, limit: i32) -> LogicalPlan {
    LogicalPlan::Limit {
        input: input.into(),
        limit,
    }
}

/// Create an Offset plan.
pub fn offset(input: impl Into<Arc<LogicalPlan>>, offset: i32) -> LogicalPlan {
    LogicalPlan::Offset {
        input: input.into(),
        offset,
    }
}

/// Create a Tail plan.
pub fn tail(input: impl Into<Arc<LogicalPlan>>, limit: i32) -> LogicalPlan {
    LogicalPlan::Tail {
        input: input.into(),
        limit,
    }
}

/// Create a Deduplicate plan.
pub fn deduplicate(
    input: impl Into<Arc<LogicalPlan>>,
    all_columns_as_keys: bool,
    column_names: Vec<String>,
    within_watermark: bool,
) -> LogicalPlan {
    LogicalPlan::Deduplicate {
        input: input.into(),
        all_columns_as_keys,
        column_names,
        within_watermark,
    }
}

/// Create a Sort plan.
pub fn sort(
    input: impl Into<Arc<LogicalPlan>>,
    order: Vec<Expression>,
    is_global: bool,
) -> LogicalPlan {
    LogicalPlan::Sort {
        input: input.into(),
        order,
        is_global,
    }
}

/// Create a Sample plan.
pub fn sample(
    input: impl Into<Arc<LogicalPlan>>,
    lower_bound: f64,
    upper_bound: f64,
    with_replacement: bool,
    seed: Option<i64>,
) -> LogicalPlan {
    LogicalPlan::Sample {
        input: input.into(),
        lower_bound,
        upper_bound,
        with_replacement,
        seed,
        deterministic_order: false,
    }
}

/// Create a Repartition plan.
pub fn repartition(
    input: impl Into<Arc<LogicalPlan>>,
    num_partitions: i32,
    shuffle: bool,
) -> LogicalPlan {
    LogicalPlan::Repartition {
        input: input.into(),
        num_partitions,
        shuffle,
    }
}

/// Create a RepartitionByExpression plan.
pub fn repartition_by_expression(
    input: impl Into<Arc<LogicalPlan>>,
    num_partitions: i32,
    expressions: Vec<Expression>,
) -> LogicalPlan {
    LogicalPlan::RepartitionByExpression {
        input: input.into(),
        num_partitions,
        expressions,
    }
}

/// Create a WithColumns plan.
pub fn with_columns(
    input: impl Into<Arc<LogicalPlan>>,
    column_names: Vec<String>,
    columns: Vec<Column>,
) -> LogicalPlan {
    LogicalPlan::WithColumns {
        input: input.into(),
        column_names,
        columns,
    }
}

/// Create a WithColumnsRenamed plan of `renames`, of old and new names, in their order, in
/// which the server applies them.
pub fn with_columns_renamed(
    input: impl Into<Arc<LogicalPlan>>,
    renames: impl IntoIterator<Item = (String, String)>,
) -> LogicalPlan {
    LogicalPlan::WithColumnsRenamed {
        input: input.into(),
        renames: renames.into_iter().collect(),
    }
}

/// Create a Drop plan.
pub fn drop(input: impl Into<Arc<LogicalPlan>>, columns: Vec<String>) -> LogicalPlan {
    LogicalPlan::Drop {
        input: input.into(),
        columns,
        expressions: Vec::new(),
    }
}

/// Create a ToDF plan.
pub fn to_df(input: impl Into<Arc<LogicalPlan>>, column_names: Vec<String>) -> LogicalPlan {
    LogicalPlan::ToDF {
        input: input.into(),
        column_names,
    }
}

/// Create a ToSchema plan.
pub fn to_schema(input: impl Into<Arc<LogicalPlan>>, schema: DataType) -> LogicalPlan {
    LogicalPlan::ToSchema {
        input: input.into(),
        schema,
        schema_reader: None,
    }
}

/// The format of a `Parse`: PySpark's `ParseFormat` of `spark.read.csv`, `json` and `xml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseFormat {
    Csv,
    Json,
    Xml,
}

impl ParseFormat {
    fn to_proto(self) -> proto::parse::ParseFormat {
        match self {
            ParseFormat::Csv => proto::parse::ParseFormat::Csv,
            ParseFormat::Json => proto::parse::ParseFormat::Json,
            ParseFormat::Xml => proto::parse::ParseFormat::Xml,
        }
    }
}

/// Create a Hint plan, of parameters given as strings: an integer-looking parameter is an
/// Integer (int32) literal, matching PySpark's `lit(int)`, otherwise a String literal. (A
/// Long literal is rejected as a partitionNum by e.g. the REBALANCE hint, which expects an
/// integral int.)
pub fn hint(
    input: impl Into<Arc<LogicalPlan>>,
    name: impl Into<String>,
    parameters: Vec<String>,
) -> LogicalPlan {
    LogicalPlan::Hint {
        input: input.into(),
        name: name.into(),
        parameters: parameters.iter().map(|p| hint_parameter(p)).collect(),
    }
}

/// The literal of a hint parameter given as a string: an Integer for an integer-looking one,
/// otherwise the String.
pub(crate) fn hint_parameter(parameter: &str) -> Expression {
    Expression::literal(match parameter.parse::<i32>() {
        Ok(n) => LiteralExpression::int(n),
        Err(_) => LiteralExpression::string(parameter),
    })
}

/// Create an Unpivot plan.
pub fn unpivot(
    input: impl Into<Arc<LogicalPlan>>,
    ids: Vec<Column>,
    values: Option<Vec<Column>>,
    variable_column_name: impl Into<String>,
    value_column_name: impl Into<String>,
) -> LogicalPlan {
    LogicalPlan::Unpivot {
        input: input.into(),
        ids,
        values,
        variable_column_name: variable_column_name.into(),
        value_column_name: value_column_name.into(),
    }
}

/// Create a NAFill plan.
pub fn na_fill(
    input: impl Into<Arc<LogicalPlan>>,
    fill_value: crate::row::Value,
    columns: Vec<String>,
) -> LogicalPlan {
    LogicalPlan::NAFill {
        input: input.into(),
        fill_value,
        columns,
    }
}

/// The proto `Literal` of a fill of NAFill, as PySpark's `NAFill._convert_value` sends it: a
/// bool as a boolean, an integer as a long, a float as a double and a string as a string, the
/// only types that the server's NAFill takes. `DataFrame::fillna_value` takes no other value. For
/// one that a plan built directly holds, the literal is of its own type, or empty for a nested
/// value, so that the server rejects it instead of filling a string column with a text of it.
fn value_to_proto_literal(v: &crate::row::Value) -> proto::expression::Literal {
    use crate::row::Value;
    use proto::expression::literal::LiteralType;
    let literal_type = match v {
        Value::Bool(b) => Some(LiteralType::Boolean(*b)),
        Value::Byte(x) => Some(LiteralType::Long(i64::from(*x))),
        Value::Short(x) => Some(LiteralType::Long(i64::from(*x))),
        Value::Integer(x) => Some(LiteralType::Long(i64::from(*x))),
        Value::Long(x) => Some(LiteralType::Long(*x)),
        Value::Float(x) => Some(LiteralType::Double(f64::from(*x))),
        Value::Double(x) => Some(LiteralType::Double(*x)),
        Value::String(s) => Some(LiteralType::String(s.clone())),
        Value::Binary(b) => Some(LiteralType::Binary(b.clone().into())),
        Value::Date(d) => Some(LiteralType::Date(*d)),
        Value::Timestamp(t) => Some(LiteralType::Timestamp(*t)),
        Value::TimestampNtz(t) => Some(LiteralType::TimestampNtz(*t)),
        Value::DayTimeInterval(t) => Some(LiteralType::DayTimeInterval(*t)),
        Value::YearMonthInterval(months) => Some(LiteralType::YearMonthInterval(*months)),
        Value::Decimal {
            value,
            precision,
            scale,
        } => Some(LiteralType::Decimal(proto::expression::literal::Decimal {
            value: value.clone(),
            precision: *precision,
            scale: *scale,
        })),
        Value::Null
        | Value::Time(_)
        | Value::List(_)
        | Value::Map(_)
        | Value::Struct(_)
        | Value::Variant { .. } => None,
    };
    proto::expression::Literal {
        literal_type,
        ..Default::default()
    }
}

/// Create a NADrop plan.
pub fn na_drop(
    input: impl Into<Arc<LogicalPlan>>,
    how: impl Into<String>,
    min_non_null: Option<i32>,
    columns: Vec<String>,
) -> LogicalPlan {
    LogicalPlan::NADrop {
        input: input.into(),
        how: how.into(),
        min_non_null,
        columns,
    }
}

/// Create a NAReplace plan of string replacements: each pair is a string and the string that
/// replaces it, both sent as string literals, as PySpark's `replace` of strings sends them, a
/// string that reads as a number too. A `LogicalPlan::NAReplace` of other literals replaces
/// booleans, numbers or nulls.
pub fn na_replace(
    input: impl Into<Arc<LogicalPlan>>,
    replacements: Vec<(String, String)>,
    columns: Vec<String>,
) -> LogicalPlan {
    LogicalPlan::NAReplace {
        input: input.into(),
        replacements: replacements
            .into_iter()
            .map(|(old, new)| {
                (
                    LiteralExpression::string(old),
                    LiteralExpression::string(new),
                )
            })
            .collect(),
        columns,
    }
}

/// Create a Describe plan.
pub fn describe(input: impl Into<Arc<LogicalPlan>>, columns: Vec<String>) -> LogicalPlan {
    LogicalPlan::Describe {
        input: input.into(),
        columns,
    }
}

/// Create a Summary plan.
pub fn summary(input: impl Into<Arc<LogicalPlan>>, percentiles: Vec<String>) -> LogicalPlan {
    LogicalPlan::Summary {
        input: input.into(),
        percentiles,
    }
}

/// Create a ColRegex plan.
pub fn col_regex(input: impl Into<Arc<LogicalPlan>>, col_name: impl Into<String>) -> LogicalPlan {
    LogicalPlan::ColRegex {
        input: input.into(),
        col_name: col_name.into(),
    }
}

/// Create a SubqueryAlias plan.
pub fn subquery_alias(input: impl Into<Arc<LogicalPlan>>, alias: impl Into<String>) -> LogicalPlan {
    LogicalPlan::SubqueryAlias {
        input: input.into(),
        alias: alias.into(),
    }
}

/// Create a LocalRelation plan.
pub fn local_relation(schema: DataType, data: Option<Vec<u8>>) -> LogicalPlan {
    LogicalPlan::LocalRelation {
        schema: Some(schema),
        data,
    }
}

/// Create a CachedRemoteRelation plan.
pub fn cached_remote_relation(relation_id: impl Into<String>) -> LogicalPlan {
    LogicalPlan::CachedRemoteRelation {
        relation_id: relation_id.into(),
        owner: None,
    }
}

#[cfg(test)]
mod argfix_tests {
    //! Regression tests for the silent arg-drop bugs: every argument accepted by
    //! the DataFrame API must reach the proto. Each test would fail against the
    //! pre-fix code (where the argument was dropped).
    use super::*;
    use crate::column::col;
    use crate::expression::LiteralExpression;
    use proto::expression::literal::LiteralType;
    use spark_connect_proto as proto;

    fn base() -> LogicalPlan {
        range(0, 10, 1)
    }

    #[test]
    fn local_relation_carries_schema() {
        // Regression: LocalRelation dropped the explicit schema (`schema: _`), so
        // createDataFrame(rows, schema) / emptyDataFrame lost the user's schema.
        match rel_type(local_relation(DataType::Struct { fields: vec![] }, None)) {
            proto::relation::RelType::LocalRelation(lr) => {
                assert!(lr.schema.is_some(), "LocalRelation must carry the schema");
            }
            _ => panic!("expected LocalRelation"),
        }
    }
    fn rel_type(p: LogicalPlan) -> proto::relation::RelType {
        p.to_proto().rel_type.expect("rel_type")
    }

    #[test]
    fn dropna_how_all_any_thresh_are_distinct() {
        let mk = |how: &str, thresh: Option<i32>| LogicalPlan::NADrop {
            input: Arc::new(base()),
            how: how.to_string(),
            min_non_null: thresh,
            columns: vec![],
        };
        let get = |p: LogicalPlan| match rel_type(p) {
            proto::relation::RelType::DropNa(d) => d.min_non_nulls,
            _ => panic!("expected DropNa"),
        };
        // how="all" -> 1 ; how="any" -> unset ; explicit thresh overrides.
        assert_eq!(get(mk("all", None)), Some(1));
        assert_eq!(get(mk("any", None)), None);
        assert_eq!(get(mk("any", Some(3))), Some(3));
        // The two `how` values must NOT build identical plans.
        assert_ne!(get(mk("all", None)), get(mk("any", None)));
    }

    #[test]
    fn hint_carries_parameters() {
        let p = hint(
            base(),
            "REPARTITION",
            vec!["10".to_string(), "name".to_string()],
        );
        match rel_type(p) {
            proto::relation::RelType::Hint(h) => {
                assert_eq!(h.parameters.len(), 2, "parameters must be forwarded");
                // First param "10" -> Integer(10) literal (int32, matching reference
                // pyspark's lit(int); a Long is rejected by the REBALANCE hint).
                let lit = match h.parameters[0].expr_type.as_ref().unwrap() {
                    proto::expression::ExprType::Literal(l) => l.literal_type.clone().unwrap(),
                    _ => panic!("expected literal param"),
                };
                assert!(matches!(lit, LiteralType::Integer(10)));
            }
            _ => panic!("expected Hint"),
        }
    }

    #[test]
    fn replace_sets_the_literals_of_their_types() {
        // As PySpark's: a bool, a double, a string, or a null replacement, as a literal of its
        // type.
        let p = LogicalPlan::NAReplace {
            input: Arc::new(base()),
            replacements: vec![
                (
                    LiteralExpression::boolean(true),
                    LiteralExpression::boolean(false),
                ),
                (
                    LiteralExpression::double(1.0),
                    LiteralExpression::null(DataType::Null),
                ),
                (
                    LiteralExpression::string("1"),
                    LiteralExpression::string("a"),
                ),
            ],
            columns: vec!["b".to_string()],
        };
        match rel_type(p) {
            proto::relation::RelType::Replace(r) => {
                let literals: Vec<(LiteralType, LiteralType)> = r
                    .replacements
                    .iter()
                    .map(|repl| {
                        let literal_type = |lit: &Option<proto::expression::Literal>| {
                            lit.clone().unwrap().literal_type.unwrap()
                        };
                        (literal_type(&repl.old_value), literal_type(&repl.new_value))
                    })
                    .collect();
                assert!(matches!(
                    &literals[0],
                    (LiteralType::Boolean(true), LiteralType::Boolean(false))
                ));
                assert!(matches!(
                    &literals[1],
                    (LiteralType::Double(v), LiteralType::Null(_)) if *v == 1.0
                ));
                assert!(matches!(
                    &literals[2],
                    (LiteralType::String(old), LiteralType::String(new)) if old == "1" && new == "a"
                ));
                assert_eq!(r.cols, vec!["b".to_string()]);
            }
            _ => panic!("expected Replace"),
        }
    }

    #[test]
    fn replace_sets_string_literals() {
        let p = na_replace(base(), vec![("foo".to_string(), "bar".to_string())], vec![]);
        match rel_type(p) {
            proto::relation::RelType::Replace(r) => {
                let repl = &r.replacements[0];
                // Both sides set (previously None for non-numeric -> silent no-op).
                let old = repl
                    .old_value
                    .as_ref()
                    .unwrap()
                    .literal_type
                    .clone()
                    .unwrap();
                let new = repl
                    .new_value
                    .as_ref()
                    .unwrap()
                    .literal_type
                    .clone()
                    .unwrap();
                assert!(matches!(old, LiteralType::String(ref s) if s == "foo"));
                assert!(matches!(new, LiteralType::String(ref s) if s == "bar"));
            }
            _ => panic!("expected Replace"),
        }
    }

    #[test]
    fn pivot_values_are_serialized() {
        let p = aggregate_with_pivot(
            base(),
            AggregateGroupType::Pivot,
            vec![],
            vec![],
            col("k").expression().clone(),
            vec![
                Expression::literal(LiteralExpression::string("a")),
                Expression::literal(LiteralExpression::string("b")),
            ],
        );
        match rel_type(p) {
            proto::relation::RelType::Aggregate(a) => {
                let pivot = a.pivot.expect("pivot set");
                assert_eq!(
                    pivot.values.len(),
                    2,
                    "explicit pivot values must be serialized"
                );
            }
            _ => panic!("expected Aggregate"),
        }
    }

    #[test]
    fn fillna_double_literal() {
        let p = LogicalPlan::NAFill {
            input: Arc::new(base()),
            fill_value: crate::row::Value::Double(1.5),
            columns: vec![],
        };
        match rel_type(p) {
            proto::relation::RelType::FillNa(f) => {
                let lit = f.values[0].literal_type.clone().unwrap();
                assert!(matches!(lit, LiteralType::Double(v) if (v - 1.5).abs() < 1e-9));
            }
            _ => panic!("expected FillNa"),
        }
    }
}

#[cfg(test)]
mod with_relations_tests {
    use super::*;
    use spark_connect_proto as proto;

    /// As PySpark sends `sql()` with DataFrames: a `WithRelations` whose root is the query
    /// and whose references are `SubqueryAlias`es of the DataFrames, each node with a plan
    /// ID of its own.
    #[test]
    fn with_relations_sends_the_root_and_its_references() {
        let ids = PlanIds::new();
        let df = Arc::new(range(0, 3, 1));
        let plan = Arc::new(LogicalPlan::WithRelations {
            root: Arc::new(sql("SELECT * FROM v")),
            references: vec![Arc::new(subquery_alias(df.clone(), "v"))],
        });
        let relation = relation_with_ids(&plan, &ids).unwrap();
        let Some(proto::relation::RelType::WithRelations(with)) = &relation.rel_type else {
            panic!("expected WithRelations: {relation:?}");
        };
        let root = with.root.as_deref().expect("root");
        assert!(
            matches!(&root.rel_type, Some(proto::relation::RelType::Sql(s)) if s.query == "SELECT * FROM v")
        );
        assert_eq!(with.references.len(), 1);
        let Some(proto::relation::RelType::SubqueryAlias(alias)) = &with.references[0].rel_type
        else {
            panic!("expected SubqueryAlias");
        };
        assert_eq!(alias.alias, "v");
        let input = alias.input.as_deref().expect("input");
        assert!(matches!(
            &input.rel_type,
            Some(proto::relation::RelType::Range(_))
        ));
        let id = |r: &proto::Relation| r.common.as_ref().and_then(|c| c.plan_id);
        let all = [id(&relation), id(root), id(&with.references[0]), id(input)];
        assert!(all.iter().all(Option::is_some), "{all:?}");
        let distinct: std::collections::HashSet<_> = all.iter().collect();
        assert_eq!(distinct.len(), 4, "{all:?}");
        // The DataFrame keeps its own plan ID as a reference.
        assert_eq!(id(input), Some(ids.id_of(&df)));
    }
}

#[cfg(test)]
mod plan_id_tests {
    use super::*;

    /// Whether `node` is one of the allocations of `expected`.
    fn one_of(node: Option<Arc<LogicalPlan>>, expected: &[&Arc<LogicalPlan>]) -> bool {
        node.is_some_and(|node| expected.iter().any(|expected| Arc::ptr_eq(&node, expected)))
    }

    #[test]
    fn a_numbered_node_is_the_one_node_of_its_plan_id() {
        let ids = PlanIds::new();
        let (first, second) = (Arc::new(range(0, 3, 1)), Arc::new(range(0, 5, 1)));
        let (first_id, second_id) = (ids.id_of(&first), ids.id_of(&second));
        assert_ne!(first_id, second_id);
        assert!(one_of(ids.node_of(first_id), &[&first]));
        assert!(one_of(ids.node_of(second_id), &[&second]));
        // The plan ID of the root of a node within its `WithRelations` is another ID of it.
        let root_id = ids.root_id_of(&first);
        assert_ne!(root_id, first_id);
        assert!(one_of(ids.node_of(root_id), &[&first]));
        assert!(ids.node_of(-1).is_none());
    }

    /// A relation that the session sent keeps its plan ID in each DataFrame of it, so a plan
    /// ID is of all of them and of the node it was first of: the node of the ID is one of
    /// those that are alive.
    #[test]
    fn the_node_of_a_plan_id_that_was_given_is_one_of_its_live_nodes() {
        let ids = PlanIds::new();
        let numbered = Arc::new(range(0, 3, 1));
        let id = ids.id_of(&numbered);
        let (copy, other_copy) = (Arc::new(range(0, 3, 1)), Arc::new(range(0, 3, 1)));
        ids.assign(&copy, id);
        ids.assign(&other_copy, id);
        assert_eq!(ids.id_of(&copy), id);
        assert!(one_of(ids.node_of(id), &[&numbered, &copy, &other_copy]));
        // A node that is dropped is no longer a node of the ID; the others still are.
        std::mem::drop(numbered);
        assert!(one_of(ids.node_of(id), &[&copy, &other_copy]));
        std::mem::drop(copy);
        assert!(one_of(ids.node_of(id), &[&other_copy]));
        std::mem::drop(other_copy);
        assert!(ids.node_of(id).is_none());
        // A plan ID that was given is not one that the session numbers a later node with.
        let later = Arc::new(range(0, 1, 1));
        assert_ne!(ids.id_of(&later), id);
    }

    /// The node of a plan ID is the first of its nodes that was given the ID and is alive, so a
    /// copy that was made later, whose Observations may differ, is taken only once that is gone.
    #[test]
    fn the_node_of_a_plan_id_is_the_first_given_one_that_is_alive() {
        let ids = PlanIds::new();
        let numbered = Arc::new(range(0, 3, 1));
        let id = ids.id_of(&numbered);
        let (copy, other_copy) = (Arc::new(range(0, 3, 1)), Arc::new(range(0, 3, 1)));
        ids.assign(&copy, id);
        ids.assign(&other_copy, id);
        assert!(one_of(ids.node_of(id), &[&numbered]));
        std::mem::drop(numbered);
        assert!(one_of(ids.node_of(id), &[&copy]));
        std::mem::drop(copy);
        assert!(one_of(ids.node_of(id), &[&other_copy]));
    }
}

#[cfg(test)]
mod inline_udtf_tests {
    use super::*;

    /// As PySpark's `CommonInlineUserDefinedTableFunction` of a `PythonUDTF`: that of a
    /// registration has no arguments, and keeps a DDL return type unparsed.
    #[test]
    fn registration_udtf_carries_the_handler_without_arguments() {
        let ddl = DataType::Unparsed {
            data_type_string: "x int".to_string(),
        };
        let f = common_inline_udtf(
            "count_udtf",
            false,
            &[],
            Some(&ddl),
            302,
            (b"pickled", None),
            "3.11",
        );
        assert_eq!(f.function_name, "count_udtf");
        assert!(!f.deterministic && f.arguments.is_empty());
        let Some(proto::common_inline_user_defined_table_function::Function::PythonUdtf(udtf)) =
            f.function
        else {
            panic!("expected a PythonUDTF");
        };
        assert_eq!(udtf.return_type, Some(ddl.to_proto()));
        assert_eq!(udtf.eval_type, 302);
        assert_eq!(udtf.command.as_ref(), b"pickled");
        assert_eq!(udtf.python_ver, "3.11");
        let dynamic = common_inline_udtf("dynamic", true, &[], None, 300, (b"", None), "3.11");
        let Some(proto::common_inline_user_defined_table_function::Function::PythonUdtf(udtf)) =
            dynamic.function
        else {
            panic!("expected a PythonUDTF");
        };
        assert_eq!(udtf.return_type, None);
    }

    /// A pickler that records `tag` in `calls` at each pickling and gives it, or, when it
    /// `fails`, the error of the tag.
    fn recording_pickler(
        calls: &Arc<Mutex<Vec<&'static str>>>,
        tag: &'static str,
        fails: bool,
    ) -> crate::udf::CommandPickler {
        crate::udf::CommandPickler::new(Arc::clone(calls), move |calls| {
            calls.lock().unwrap().push(tag);
            if fails {
                return Err(SparkError::value_msg(tag));
            }
            Ok(tag.as_bytes().to_vec())
        })
    }

    /// A Python UDF of no arguments, of the command of `pickler`.
    fn pickled_udf(
        pickler: crate::udf::CommandPickler,
    ) -> CommonInlineUserDefinedFunctionExpression {
        CommonInlineUserDefinedFunctionExpression::new(
            "f".to_string(),
            true,
            vec![],
            crate::udf::PythonUDFPayload::pickled_at_build(
                DataType::Long,
                crate::udf::eval_type::SQL_BATCHED_UDF,
                pickler,
                "3.11".to_string(),
            ),
        )
    }

    /// As PySpark's `CommonInlineUserDefinedTableFunction.plan`, which builds the arguments of a
    /// UDTF before its `PythonUDTF`, whose `to_plan` pickles the handler class at each build:
    /// an argument that fails to build is the error of a request, and the handler is not
    /// pickled; otherwise each request pickles it.
    #[test]
    fn a_udtf_pickles_its_handler_after_its_arguments_at_each_request() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let udtf = |arguments: Vec<Expression>, fails: bool| LogicalPlan::CommonInlineUdtf {
            function_name: "t".to_string(),
            deterministic: false,
            arguments,
            return_type: None,
            eval_type: 300,
            command: Vec::new(),
            python_ver: "3.11".to_string(),
            pickler: Some(recording_pickler(&calls, "handler", fails)),
        };
        let command = |relation: proto::Relation| match relation.rel_type {
            Some(proto::relation::RelType::CommonInlineUserDefinedTableFunction(call)) => {
                match call.function {
                    Some(
                        proto::common_inline_user_defined_table_function::Function::PythonUdtf(
                            udtf,
                        ),
                    ) => udtf.command.to_vec(),
                    _ => panic!("expected a PythonUDTF"),
                }
            }
            _ => panic!("expected an inline UDTF"),
        };
        let plain = udtf(vec![Expression::column("x")], false);
        for _ in 0..2 {
            assert_eq!(command(plain.try_to_proto().unwrap()), b"handler");
        }
        assert_eq!(*calls.lock().unwrap(), ["handler", "handler"]);
        calls.lock().unwrap().clear();
        let error = SparkError::value_msg("argument");
        let argument = Expression::unbuildable(Expression::column("x"), Some(error), None, None);
        let error = udtf(vec![argument], true).try_to_proto().unwrap_err();
        assert_eq!(error.message, "argument");
        assert!(calls.lock().unwrap().is_empty());
        let error = udtf(vec![], true).try_to_proto().unwrap_err();
        assert_eq!(error.message, "handler");
        // Of the bytes of a command, as of the Rust API: sent as they are.
        let bytes = LogicalPlan::CommonInlineUdtf {
            function_name: "t".to_string(),
            deterministic: false,
            arguments: vec![],
            return_type: None,
            eval_type: 300,
            command: b"pickled".to_vec(),
            python_ver: "3.11".to_string(),
            pickler: None,
        };
        calls.lock().unwrap().clear();
        assert_eq!(command(bytes.try_to_proto().unwrap()), b"pickled");
        assert!(calls.lock().unwrap().is_empty());
    }

    /// As PySpark's transformWithState plan, which builds its initial state before the function,
    /// whose command it pickles then: the functions of the initial state are pickled first, and
    /// one that does not pickle is the error, before the function is pickled.
    #[test]
    fn a_group_map_pickles_its_function_after_its_initial_state() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let group_map = |initial_fails: bool| {
            let initial = LogicalPlan::Project {
                input: Arc::new(range(0, 1, 1)),
                columns: vec![Column::new(
                    ExpressionNode::CommonInlineUserDefinedFunction(Box::new(pickled_udf(
                        recording_pickler(&calls, "initial", initial_fails),
                    )))
                    .into(),
                )],
            };
            LogicalPlan::GroupMap {
                input: Arc::new(range(0, 1, 1)),
                grouping_expressions: vec![Expression::column("id")],
                func: pickled_udf(recording_pickler(&calls, "function", false)),
                sorting_expressions: vec![],
                initial_input: Some(Arc::new(initial)),
                initial_grouping_expressions: vec![Expression::column("id")],
                is_map_groups_with_state: None,
                output_mode: Some("append".to_string()),
                timeout_conf: None,
                state_schema: None,
                transform_with_state_info: None,
            }
        };
        assert!(group_map(false).try_to_proto().is_ok());
        assert_eq!(*calls.lock().unwrap(), ["initial", "function"]);
        calls.lock().unwrap().clear();
        let error = group_map(true).try_to_proto().unwrap_err();
        assert_eq!(error.message, "initial");
        assert_eq!(*calls.lock().unwrap(), ["initial"]);
    }
}

#[cfg(test)]
mod placeholder_tests {
    use super::*;
    use prost::Message;
    use spark_connect_proto as proto;

    /// The placeholder of `node`, of the session of `ids`, as the relation that the plans of
    /// `pyspark.ml` send in its place.
    fn placeholder(node: &Arc<LogicalPlan>, ids: &Arc<PlanIds>) -> proto::Relation {
        standalone_placeholder_relation(node, ids).unwrap()
    }

    /// The `ml_relation` of a transform of `input`, as `pyspark.ml.connect.proto` builds it.
    fn transform_of(input: proto::Relation) -> proto::Relation {
        proto::Relation {
            common: None,
            rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                proto::MlRelation {
                    model_summary_dataset: None,
                    ml_type: Some(proto::ml_relation::MlType::Transform(Box::new(
                        proto::ml_relation::Transform {
                            input: Some(Box::new(input)),
                            params: None,
                            operator: None,
                        },
                    ))),
                },
            ))),
        }
    }

    #[test]
    fn a_placeholder_is_of_one_level_and_stands_for_the_relation_of_its_node() {
        // The plan is converted recursively, so a plan of 200 levels takes more than the 2 MiB
        // stack of a test thread in a debug build, whose frames are much larger than those of a
        // release build.
        let run = || {
            let ids = Arc::new(PlanIds::new());
            let mut node = Arc::new(range(0, 3, 1));
            for _ in 0..200 {
                node = Arc::new(LogicalPlan::Limit {
                    input: node,
                    limit: 3,
                });
            }
            let relation = relation_with_ids(&node, &ids).unwrap();
            let placeholder = placeholder(&node, &ids);
            // An extension of the encoded relation: a decoder of Python reads one level of it.
            let Some(proto::relation::RelType::Extension(any)) = &placeholder.rel_type else {
                panic!("not an extension: {placeholder:?}");
            };
            assert!(any.value.ends_with(&relation.encode_to_vec()));
            let mut resolved = transform_of(placeholder.clone());
            resolve_relation(&mut resolved).unwrap();
            assert_eq!(resolved, transform_of(relation.clone()));
            // As PySpark's relation of `plan(session)`, it stands for the relation after its
            // DataFrame is gone, and so do its bytes, decoded again; it keeps nothing alive.
            let (weak_node, weak_ids) = (Arc::downgrade(&node), Arc::downgrade(&ids));
            std::mem::drop((node, ids));
            assert!(weak_node.upgrade().is_none() && weak_ids.upgrade().is_none());
            let decoded = proto::Relation::decode(placeholder.encode_to_vec().as_slice()).unwrap();
            for mut kept in [transform_of(placeholder), transform_of(decoded)] {
                resolve_relation(&mut kept).unwrap();
                assert_eq!(kept, transform_of(relation.clone()));
            }
        };
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(run)
            .unwrap()
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    }

    #[test]
    fn a_placeholder_of_a_marker_alone_stands_for_the_relation_of_its_node_while_it_lives() {
        // As the core's own plans over a DataFrame hold it, which they resolve at once.
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let relation = relation_with_ids(&node, &ids).unwrap();
        let placeholder = placeholder_relation(&node, &ids);
        assert!(
            placeholder.encoded_len() < 128,
            "{}",
            placeholder.encoded_len()
        );
        let mut resolved = transform_of(placeholder.clone());
        resolve_relation(&mut resolved).unwrap();
        assert_eq!(resolved, transform_of(relation));
        std::mem::drop(node);
        let mut gone = transform_of(placeholder);
        let error = resolve_relation(&mut gone).unwrap_err();
        assert!(error.message().contains("no longer exists"), "{error}");
    }

    #[test]
    fn a_placeholder_is_the_same_for_the_same_node_and_session() {
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        assert_eq!(placeholder(&node, &ids), placeholder(&node, &ids));
        let other = Arc::new(PlanIds::new());
        assert_ne!(placeholder(&node, &ids), placeholder(&node, &other));
        let different = Arc::new(range(0, 4, 1));
        assert_ne!(placeholder(&node, &ids), placeholder(&different, &ids));
    }

    #[test]
    fn placeholders_resolve_in_every_position_that_the_ml_plans_build_over_their_inputs() {
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let relation = relation_with_ids(&node, &ids).unwrap();
        let shallow = placeholder(&node, &ids);
        let fetch = || proto::Fetch {
            obj_ref: None,
            methods: vec![proto::fetch::Method {
                method: "transform".to_string(),
                args: vec![proto::fetch::method::Args {
                    args_type: Some(proto::fetch::method::args::ArgsType::Input(shallow.clone())),
                }],
            }],
        };
        let mut relations = vec![
            // An ML summary of a model, over its dataset.
            proto::Relation {
                common: None,
                rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                    proto::MlRelation {
                        model_summary_dataset: Some(Box::new(shallow.clone())),
                        ml_type: None,
                    },
                ))),
            },
            // A fetch of a summary, whose argument is a dataset.
            proto::Relation {
                common: None,
                rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                    proto::MlRelation {
                        model_summary_dataset: None,
                        ml_type: Some(proto::ml_relation::MlType::Fetch(fetch())),
                    },
                ))),
            },
            // Two transforms, one over the other, over the input of the second.
            transform_of(transform_of(shallow.clone())),
            // The references of a relation with its subqueries.
            proto::Relation {
                common: None,
                rel_type: Some(proto::relation::RelType::WithRelations(Box::new(
                    proto::WithRelations {
                        root: Some(Box::new(shallow.clone())),
                        references: vec![shallow.clone()],
                    },
                ))),
            },
        ];
        for relation_with_placeholder in relations.iter_mut() {
            resolve_relation(relation_with_placeholder).unwrap();
        }
        // The placeholders are replaced by the relation of the node, everywhere they were.
        let expected = [
            proto::Relation {
                common: None,
                rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                    proto::MlRelation {
                        model_summary_dataset: Some(Box::new(relation.clone())),
                        ml_type: None,
                    },
                ))),
            },
            proto::Relation {
                common: None,
                rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                    proto::MlRelation {
                        model_summary_dataset: None,
                        ml_type: Some(proto::ml_relation::MlType::Fetch(proto::Fetch {
                            obj_ref: None,
                            methods: vec![proto::fetch::Method {
                                method: "transform".to_string(),
                                args: vec![proto::fetch::method::Args {
                                    args_type: Some(proto::fetch::method::args::ArgsType::Input(
                                        relation.clone(),
                                    )),
                                }],
                            }],
                        })),
                    },
                ))),
            },
            transform_of(transform_of(relation.clone())),
            proto::Relation {
                common: None,
                rel_type: Some(proto::relation::RelType::WithRelations(Box::new(
                    proto::WithRelations {
                        root: Some(Box::new(relation.clone())),
                        references: vec![relation.clone()],
                    },
                ))),
            },
        ];
        assert_eq!(relations, expected);
    }

    #[test]
    fn the_ml_commands_resolve_their_datasets_and_fetched_arguments() {
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let relation = relation_with_ids(&node, &ids).unwrap();
        let shallow = placeholder(&node, &ids);
        let command = |ml_command: proto::ml_command::Command| proto::Command {
            command_type: Some(proto::command::CommandType::MlCommand(proto::MlCommand {
                command: Some(ml_command),
            })),
        };
        let fit = |dataset: proto::Relation| {
            command(proto::ml_command::Command::Fit(proto::ml_command::Fit {
                estimator: None,
                params: None,
                dataset: Some(dataset),
            }))
        };
        let mut fitted = fit(shallow.clone());
        resolve_command(&mut fitted, None).unwrap();
        assert_eq!(fitted, fit(relation.clone()));

        let mut evaluated = command(proto::ml_command::Command::Evaluate(
            proto::ml_command::Evaluate {
                evaluator: None,
                params: None,
                dataset: Some(shallow.clone()),
            },
        ));
        resolve_command(&mut evaluated, None).unwrap();
        assert_eq!(
            evaluated,
            command(proto::ml_command::Command::Evaluate(
                proto::ml_command::Evaluate {
                    evaluator: None,
                    params: None,
                    dataset: Some(relation.clone()),
                },
            ))
        );

        let mut summary = command(proto::ml_command::Command::CreateSummary(
            proto::ml_command::CreateSummary {
                model_ref: None,
                dataset: Some(shallow.clone()),
            },
        ));
        resolve_command(&mut summary, None).unwrap();
        assert_eq!(
            summary,
            command(proto::ml_command::Command::CreateSummary(
                proto::ml_command::CreateSummary {
                    model_ref: None,
                    dataset: Some(relation.clone()),
                },
            ))
        );

        // A request of a session hook: the root of its plan, or its command.
        let root = proto::ExecutePlanRequest {
            plan: Some(proto::Plan {
                op_type: Some(proto::plan::OpType::Root(transform_of(shallow.clone()))),
            }),
            ..Default::default()
        };
        let root = resolve_request(root).unwrap();
        assert_eq!(
            root.plan.unwrap().op_type,
            Some(proto::plan::OpType::Root(transform_of(relation)))
        );
    }

    /// A relation of the extension `message`, of a plugin's type.
    fn extension_of(message: &impl Message) -> proto::Relation {
        proto::Relation {
            common: None,
            rel_type: Some(proto::relation::RelType::Extension(prost_types::Any {
                type_url: "type.googleapis.com/tests.PluginRelation".to_string(),
                value: message.encode_to_vec(),
            })),
        }
    }

    /// A relation of a project over a range, of the expressions `expressions`.
    fn project_of(expressions: Vec<proto::Expression>) -> proto::Relation {
        proto::Relation {
            common: None,
            rel_type: Some(proto::relation::RelType::Project(Box::new(
                proto::Project {
                    input: Some(Box::new(proto::Relation {
                        common: None,
                        rel_type: Some(proto::relation::RelType::Range(proto::Range {
                            end: 1,
                            ..Default::default()
                        })),
                    })),
                    expressions,
                },
            ))),
        }
    }

    /// An expression of a plugin, whose message is `message`.
    fn expression_extension_of(message: &impl Message) -> proto::Expression {
        proto::Expression {
            common: None,
            expr_type: Some(proto::expression::ExprType::Extension(prost_types::Any {
                type_url: "type.googleapis.com/tests.PluginExpression".to_string(),
                value: message.encode_to_vec(),
            })),
        }
    }

    #[test]
    fn placeholders_resolve_in_the_extensions_of_expressions() {
        // As PySpark's plan sends the relation of a DataFrame that a plugin's expression holds in
        // its message: in a relation built in Python, also nested in an expression, and in a
        // command. Spark's SubqueryAlias stands for the plugin's message here.
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let relation = relation_with_ids(&node, &ids).unwrap();
        let standalone = placeholder(&node, &ids);
        let message = |input: &proto::Relation| proto::SubqueryAlias {
            input: Some(Box::new(input.clone())),
            alias: "s".to_string(),
            qualifier: Vec::new(),
        };
        let nested = |input: &proto::Relation| proto::Expression {
            common: None,
            expr_type: Some(proto::expression::ExprType::Alias(Box::new(
                proto::expression::Alias {
                    expr: Some(Box::new(expression_extension_of(&message(input)))),
                    name: vec!["a".to_string()],
                    metadata: None,
                },
            ))),
        };
        let mut plan = project_of(vec![
            expression_extension_of(&message(&standalone)),
            nested(&standalone),
        ]);
        resolve_relation(&mut plan).unwrap();
        assert_eq!(
            plan,
            project_of(vec![
                expression_extension_of(&message(&relation)),
                nested(&relation)
            ])
        );
        // The merge condition of a command.
        let merge = |condition: proto::Expression| proto::Command {
            command_type: Some(proto::command::CommandType::MergeIntoTableCommand(
                proto::MergeIntoTableCommand {
                    target_table_name: "t".to_string(),
                    source_table_plan: Some(relation.clone()),
                    merge_condition: Some(condition),
                    ..Default::default()
                },
            )),
        };
        let mut command = merge(expression_extension_of(&message(&standalone)));
        resolve_command(&mut command, None).unwrap();
        assert_eq!(command, merge(expression_extension_of(&message(&relation))));
        // A plan of no placeholder is left as it is.
        let mut plain = project_of(vec![expression_extension_of(&message(&relation))]);
        let before = plain.clone();
        resolve_relation(&mut plain).unwrap();
        assert_eq!(plain, before);
    }

    #[test]
    fn a_placeholder_that_a_plan_changes_is_sent_with_the_change() {
        // As PySpark's relation of `plan(session)`, whose `common` a plan or a plugin sets
        // (`relation.common.plan_id = ...`): the relation is sent with the `common` merged into
        // its own, as a relation of a plan and in the message of an extension.
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let relation = relation_with_ids(&node, &ids).unwrap();
        let mut changed = placeholder(&node, &ids);
        changed.common = Some(proto::RelationCommon {
            plan_id: Some(424242),
            ..Default::default()
        });
        let mut expected = relation.clone();
        expected.common.get_or_insert_with(Default::default).plan_id = Some(424242);
        assert_ne!(expected, relation);

        let mut plan = transform_of(changed.clone());
        resolve_relation(&mut plan).unwrap();
        assert_eq!(plan, transform_of(expected.clone()));

        let alias = |input: &proto::Relation| proto::SubqueryAlias {
            input: Some(Box::new(input.clone())),
            alias: "s".to_string(),
            qualifier: Vec::new(),
        };
        let mut plan = extension_of(&alias(&changed));
        resolve_relation(&mut plan).unwrap();
        assert_eq!(plan, extension_of(&alias(&expected)));
    }

    #[test]
    fn deep_messages_of_extensions_resolve_in_a_time_of_their_length() {
        // A crafted message of an extension nests 30,000 messages, of a placeholder at the bottom
        // and of none: each is resolved once, in a time of its length, not of its length times
        // its depth, which took about a minute in a debug build.
        let run = || {
            let ids = Arc::new(PlanIds::new());
            let node = Arc::new(range(0, 3, 1));
            let relation = relation_with_ids(&node, &ids).unwrap();
            let standalone = placeholder(&node, &ids).encode_to_vec();
            let nested = |depth: usize, inner: &[u8]| {
                let mut data = inner.to_vec();
                for _ in 0..depth {
                    let mut level = vec![0x0a];
                    prost::encoding::encode_varint(data.len() as u64, &mut level);
                    level.extend_from_slice(&data);
                    data = level;
                }
                data
            };
            let extension = |value: Vec<u8>| proto::Relation {
                common: None,
                rel_type: Some(proto::relation::RelType::Extension(prost_types::Any {
                    type_url: "type.googleapis.com/tests.Crafted".to_string(),
                    value,
                })),
            };
            let started = std::time::Instant::now();
            let mut plan = extension(nested(30_000, &standalone));
            resolve_relation(&mut plan).unwrap();
            assert_eq!(plan, extension(nested(30_000, &relation.encode_to_vec())));
            let mut plain = extension(nested(30_000, b"\x08\x01"));
            resolve_relation(&mut plain).unwrap();
            assert_eq!(plain, extension(nested(30_000, b"\x08\x01")));
            assert!(
                started.elapsed() < std::time::Duration::from_secs(10),
                "{:?}",
                started.elapsed()
            );
        };
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(run)
            .unwrap()
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    }

    /// `relation` resolved as `DataFrame::from_relation` resolves the relation of an encoding of the
    /// Python client: decoded of its encoding, with its levels and the search of the resolution.
    fn resolved_of_its_encoding(relation: &proto::Relation) -> (proto::Relation, usize, Search) {
        let encoded = relation.encode_to_vec();
        let (mut decoded, urls) =
            BuiltOutside::decoded(encoded.as_slice(), proto::Relation::decode)
                .unwrap()
                .into_parts();
        LAST_SEARCH.with(|last| last.set(None));
        let levels = resolve_relation_levels(&mut decoded, urls).unwrap();
        let search = LAST_SEARCH.with(|last| last.get()).unwrap();
        (decoded, levels, search)
    }

    /// `relation` resolved as it was before the encodings were read: everywhere, as a relation of
    /// the core given without its encoding is.
    fn resolved_everywhere(relation: &proto::Relation) -> (proto::Relation, usize) {
        let mut resolved = relation.clone();
        LAST_SEARCH.with(|last| last.set(None));
        let levels = resolve_relation_levels(&mut resolved, None).unwrap();
        assert_eq!(
            LAST_SEARCH.with(|last| last.get()),
            Some(Search::Everywhere)
        );
        (resolved, levels)
    }

    /// The column `name` of no plan ID.
    fn column_of(name: &str) -> proto::Expression {
        proto::Expression {
            common: None,
            expr_type: Some(proto::expression::ExprType::UnresolvedAttribute(
                proto::expression::UnresolvedAttribute {
                    unparsed_identifier: name.to_string(),
                    plan_id: None,
                    is_metadata_column: None,
                },
            )),
        }
    }

    /// A project over `input` of `expressions`.
    fn project_over(
        input: &proto::Relation,
        expressions: Vec<proto::Expression>,
    ) -> proto::Relation {
        proto::Relation {
            common: None,
            rel_type: Some(proto::relation::RelType::Project(Box::new(
                proto::Project {
                    input: Some(Box::new(input.clone())),
                    expressions,
                },
            ))),
        }
    }

    /// A plugin's message that holds `input`, as Spark's SubqueryAlias stands for one here.
    fn message_of(input: &proto::Relation) -> proto::SubqueryAlias {
        proto::SubqueryAlias {
            input: Some(Box::new(input.clone())),
            alias: "s".to_string(),
            qualifier: Vec::new(),
        }
    }

    #[test]
    fn an_encoding_of_the_type_urls_of_the_placeholders_at_its_relations_alone_is_resolved_at_them()
    {
        // The fields and the messages of the extensions of a relation that the Python client
        // encoded are not read when its encoding holds no type URL of a placeholder but those of
        // the placeholders at its relations: of none, of inputs, of the inputs of a join, of an
        // ML transform, of a fetched argument, of a `WithRelations`, and of a changed placeholder,
        // whose `common` is merged. The relation and its levels are those of the resolution of
        // every field and message.
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let other = Arc::new(range(0, 5, 1));
        let standalone = placeholder(&node, &ids);
        let other_standalone = placeholder(&other, &ids);
        let mut changed = standalone.clone();
        changed.common = Some(proto::RelationCommon {
            plan_id: Some(424242),
            ..Default::default()
        });
        let columns = vec![column_of("id"), column_of("k")];
        let join = proto::Relation {
            common: None,
            rel_type: Some(proto::relation::RelType::Join(Box::new(proto::Join {
                left: Some(Box::new(project_over(&standalone, columns.clone()))),
                right: Some(Box::new(other_standalone.clone())),
                join_condition: Some(column_of("c")),
                ..Default::default()
            }))),
        };
        let fetch = proto::Relation {
            common: None,
            rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                proto::MlRelation {
                    model_summary_dataset: Some(Box::new(other_standalone.clone())),
                    ml_type: Some(proto::ml_relation::MlType::Fetch(proto::Fetch {
                        obj_ref: None,
                        methods: vec![proto::fetch::Method {
                            method: "transform".to_string(),
                            args: vec![proto::fetch::method::Args {
                                args_type: Some(proto::fetch::method::args::ArgsType::Input(
                                    standalone.clone(),
                                )),
                            }],
                        }],
                    })),
                },
            ))),
        };
        let with_relations = proto::Relation {
            common: None,
            rel_type: Some(proto::relation::RelType::WithRelations(Box::new(
                proto::WithRelations {
                    root: Some(Box::new(project_over(&standalone, columns.clone()))),
                    references: vec![other_standalone.clone()],
                },
            ))),
        };
        let cases = [
            ("no placeholder", project_of(columns.clone())),
            ("an input", project_over(&standalone, columns.clone())),
            ("the inputs of a join", join),
            (
                "an ML transform",
                transform_of(transform_of(standalone.clone())),
            ),
            ("a fetched argument and a summary's dataset", fetch),
            ("a WithRelations", with_relations),
            (
                "a changed placeholder",
                project_over(&changed, columns.clone()),
            ),
        ];
        for (name, relation) in cases {
            let (resolved, levels, search) = resolved_of_its_encoding(&relation);
            assert_eq!(search, Search::Relations, "{name}");
            assert_eq!((resolved, levels), resolved_everywhere(&relation), "{name}");
        }
        // The changed placeholder is sent with its change, at its level.
        let mut expected = relation_with_ids(&node, &ids).unwrap();
        expected.common.get_or_insert_with(Default::default).plan_id = Some(424242);
        let (resolved, _, _) = resolved_of_its_encoding(&project_over(&changed, columns.clone()));
        assert_eq!(resolved, project_over(&expected, columns));
    }

    #[test]
    fn an_encoding_of_a_type_url_elsewhere_is_resolved_everywhere() {
        // A type URL of a placeholder that is not of one at the relations of the plan, which only
        // a read of the fields or of the messages of the extensions tells apart: of a placeholder
        // in the extension of an expression, also with one at an input, in the message of a
        // relation's extension, also deep in it, in a string, which holds no placeholder, and in
        // the bytes of a placeholder, whose relation holds the string. Each is resolved as every
        // field and message are read, and so its placeholders are all replaced.
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let relation = relation_with_ids(&node, &ids).unwrap();
        let standalone = placeholder(&node, &ids);
        let url = PLACEHOLDER_TYPE_URL;
        // A placeholder whose relation holds the type URL as a string, in its bytes.
        let named = project_over(&relation, vec![column_of(url)]);
        let mut value = 7i64.to_le_bytes().to_vec();
        value.extend_from_slice(&2u64.to_le_bytes());
        value.extend_from_slice(&named.encode_to_vec());
        let holding = placeholder_of_value(value);
        let deep = {
            let mut message = message_of(&standalone);
            for _ in 0..50 {
                message = message_of(&extension_of(&message));
            }
            extension_of(&message)
        };
        let cases = [
            (
                "an expression",
                project_of(vec![expression_extension_of(&message_of(&standalone))]),
                project_of(vec![expression_extension_of(&message_of(&relation))]),
            ),
            (
                "an expression and an input",
                project_over(
                    &standalone,
                    vec![expression_extension_of(&message_of(&standalone))],
                ),
                project_over(
                    &relation,
                    vec![expression_extension_of(&message_of(&relation))],
                ),
            ),
            (
                "the message of an extension",
                extension_of(&message_of(&standalone)),
                extension_of(&message_of(&relation)),
            ),
            (
                "a string",
                project_over(&standalone, vec![column_of(url)]),
                project_over(&relation, vec![column_of(url)]),
            ),
            (
                "the bytes of a placeholder",
                project_over(&holding, Vec::new()),
                { project_over(&named, Vec::new()) },
            ),
        ];
        for (name, plan, expected) in cases {
            let (resolved, levels, search) = resolved_of_its_encoding(&plan);
            assert_eq!(search, Search::Everywhere, "{name}");
            assert_eq!(resolved, expected, "{name}");
            assert_eq!((resolved, levels), resolved_everywhere(&plan), "{name}");
        }
        let (resolved, _, search) = resolved_of_its_encoding(&deep);
        assert_eq!(search, Search::Everywhere);
        assert_eq!(resolved, resolved_everywhere(&deep).0);
        assert!(!resolved
            .encode_to_vec()
            .windows(url.len())
            .any(|w| w == url.as_bytes()));
    }

    #[test]
    fn the_commands_plans_analyses_and_requests_of_an_encoding_are_resolved_of_its_search() {
        // As the relations: a command, a plan, an analysis and a request of a session hook whose
        // encoding holds no type URL of a placeholder but those of the placeholders at their
        // relations are resolved at them alone; another, of a placeholder in the condition of a
        // merge, in the message of a command's extension or in an extension of the user context
        // of a request, everywhere. Each is resolved as it is when it is given without its
        // encoding.
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let standalone = placeholder(&node, &ids);
        let write = proto::Command {
            command_type: Some(proto::command::CommandType::WriteOperation(
                proto::WriteOperation {
                    input: Some(project_over(&standalone, vec![column_of("id")])),
                    source: Some("noop".to_string()),
                    ..Default::default()
                },
            )),
        };
        let fit = proto::Command {
            command_type: Some(proto::command::CommandType::MlCommand(proto::MlCommand {
                command: Some(proto::ml_command::Command::Fit(proto::ml_command::Fit {
                    estimator: None,
                    params: None,
                    dataset: Some(standalone.clone()),
                })),
            })),
        };
        let merge = proto::Command {
            command_type: Some(proto::command::CommandType::MergeIntoTableCommand(
                proto::MergeIntoTableCommand {
                    target_table_name: "t".to_string(),
                    source_table_plan: Some(standalone.clone()),
                    merge_condition: Some(expression_extension_of(&message_of(&standalone))),
                    ..Default::default()
                },
            )),
        };
        let extension = proto::Command {
            command_type: Some(proto::command::CommandType::Extension(prost_types::Any {
                type_url: "type.googleapis.com/tests.PluginCommand".to_string(),
                value: message_of(&standalone).encode_to_vec(),
            })),
        };
        let resolved = |command: &proto::Command, encoded: bool| {
            let bytes = command.encode_to_vec();
            let (mut command, urls) = if encoded {
                BuiltOutside::decoded(bytes.as_slice(), proto::Command::decode)
                    .unwrap()
                    .into_parts()
            } else {
                BuiltOutside::from(command.clone()).into_parts()
            };
            resolve_command(&mut command, urls).unwrap();
            (command, LAST_SEARCH.with(|last| last.get()).unwrap())
        };
        for (command, search) in [
            (&write, Search::Relations),
            (&fit, Search::Relations),
            (&merge, Search::Everywhere),
            (&extension, Search::Everywhere),
        ] {
            let (of_encoding, of) = resolved(command, true);
            assert_eq!(of, search, "{command:?}");
            assert_eq!(resolved(command, false), (of_encoding, Search::Everywhere));
        }

        // A plan, and a request of a session hook, of its plan's search.
        let plan = |command: &proto::Command| proto::Plan {
            op_type: Some(proto::plan::OpType::Command(command.clone())),
        };
        for (command, search) in [(&write, Search::Relations), (&merge, Search::Everywhere)] {
            let bytes = plan(command).encode_to_vec();
            let (mut decoded, urls) = BuiltOutside::decoded(bytes.as_slice(), proto::Plan::decode)
                .unwrap()
                .into_parts();
            resolve_plan(&mut decoded, urls).unwrap();
            assert_eq!(LAST_SEARCH.with(|last| last.get()), Some(search));
            let mut everywhere = plan(command);
            resolve_plan(&mut everywhere, None).unwrap();
            assert_eq!(decoded, everywhere);
        }
        let request = |context: Vec<prost_types::Any>| proto::ExecutePlanRequest {
            plan: Some(plan(&write)),
            user_context: Some(proto::UserContext {
                extensions: context,
                ..Default::default()
            }),
            ..Default::default()
        };
        let marked = prost_types::Any {
            type_url: PLACEHOLDER_TYPE_URL.to_string(),
            value: Vec::new(),
        };
        for (context, search) in [
            (Vec::new(), Search::Relations),
            (vec![marked], Search::Everywhere),
        ] {
            let bytes = request(context.clone()).encode_to_vec();
            let of_encoding = resolve_request(
                BuiltOutside::decoded(bytes.as_slice(), proto::ExecutePlanRequest::decode).unwrap(),
            )
            .unwrap();
            assert_eq!(LAST_SEARCH.with(|last| last.get()), Some(search));
            assert_eq!(of_encoding, resolve_request(request(context)).unwrap());
        }

        // An analysis of two plans, each over a placeholder.
        let same = |command: &proto::Command| {
            proto::analyze_plan_request::Analyze::SameSemantics(
                proto::analyze_plan_request::SameSemantics {
                    target_plan: Some(plan(command)),
                    other_plan: Some(proto::Plan {
                        op_type: Some(proto::plan::OpType::Root(standalone.clone())),
                    }),
                },
            )
        };
        for (command, search) in [
            (&write, Search::Relations),
            (&extension, Search::Everywhere),
        ] {
            let bytes = proto::AnalyzePlanRequest {
                analyze: Some(same(command)),
                ..Default::default()
            }
            .encode_to_vec();
            let (decoded, urls) =
                BuiltOutside::decoded(bytes.as_slice(), proto::AnalyzePlanRequest::decode)
                    .unwrap()
                    .into_parts();
            let mut analyze = decoded.analyze.unwrap();
            resolve_analysis(&mut analyze, urls).unwrap();
            assert_eq!(LAST_SEARCH.with(|last| last.get()), Some(search));
            let mut everywhere = same(command);
            resolve_analysis(&mut everywhere, None).unwrap();
            assert_eq!(analyze, everywhere);
        }
    }

    #[test]
    fn a_message_that_does_not_decode_is_its_decoder_s_error() {
        // `BuiltOutside::decoded` gives the error of the decoder of the message, as the client's
        // decode of it does.
        let error = BuiltOutside::decoded(&[0xff, 0xff][..], proto::Relation::decode).unwrap_err();
        assert_eq!(
            error.to_string(),
            proto::Relation::decode(&[0xff, 0xff][..])
                .unwrap_err()
                .to_string()
        );
        assert_eq!(
            placeholder_urls(PLACEHOLDER_TYPE_URL.repeat(3).as_bytes()),
            3
        );
    }

    #[test]
    fn placeholders_resolve_in_the_messages_of_the_extensions_of_plugins() {
        // As the relation of Delta Lake's merge, the extension of a relation whose message, of a
        // type that the core does not know, holds the plan of a DataFrame in a field, and below
        // in a message of a field: Spark's Join stands for a plugin's message here. Its string
        // fields that hold the type URL of a placeholder, which do not read as messages, stay.
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let relation = relation_with_ids(&node, &ids).unwrap();
        let shallow = placeholder(&node, &ids);
        let join = |input: &proto::Relation| proto::Join {
            left: Some(Box::new(input.clone())),
            right: Some(Box::new(proto::Relation {
                common: None,
                rel_type: Some(proto::relation::RelType::SubqueryAlias(Box::new(
                    proto::SubqueryAlias {
                        input: Some(Box::new(input.clone())),
                        alias: PLACEHOLDER_TYPE_URL.to_string(),
                        qualifier: Vec::new(),
                    },
                ))),
            })),
            join_condition: None,
            join_type: 2,
            using_columns: vec![PLACEHOLDER_TYPE_URL.to_string()],
            join_data_type: None,
        };
        let mut plan = extension_of(&join(&shallow));
        // The message of the extension, which the client serializes on its own, adds no level to
        // the plan, as PySpark's serializer counts them.
        assert_eq!(resolve_relation_levels(&mut plan, None).unwrap(), 1);
        assert_eq!(plan, extension_of(&join(&relation)));

        // Of the extension of a command, as a plugin's Python client sends it.
        let alias = proto::SubqueryAlias {
            input: Some(Box::new(shallow.clone())),
            alias: "s".to_string(),
            qualifier: Vec::new(),
        };
        let command = |message: &proto::SubqueryAlias| proto::Command {
            command_type: Some(proto::command::CommandType::Extension(prost_types::Any {
                type_url: "type.googleapis.com/tests.PluginCommand".to_string(),
                value: message.encode_to_vec(),
            })),
        };
        let mut sent = command(&alias);
        resolve_command(&mut sent, None).unwrap();
        let resolved = proto::SubqueryAlias {
            input: Some(Box::new(relation)),
            ..alias.clone()
        };
        assert_eq!(sent, command(&resolved));

        // An extension of no placeholder is left as it is, of its levels.
        let mut plain = extension_of(&resolved);
        assert_eq!(resolve_relation_levels(&mut plain, None).unwrap(), 1);
        assert_eq!(plain, extension_of(&resolved));
        // So is one whose message holds the type URL in a string, which does not read as a
        // message, and in an Any of no marker.
        let mut text = extension_of(&prost_types::Any {
            type_url: PLACEHOLDER_TYPE_URL.to_string(),
            value: b"\x04\x05".to_vec(),
        });
        let before = text.clone();
        resolve_relation(&mut text).unwrap();
        assert_eq!(text, before);

        // A placeholder of a dropped DataFrame in an extension stands for its relation as one
        // elsewhere, and one of a marker alone is the error of one elsewhere.
        let gone = Arc::new(range(0, 4, 1));
        let gone_relation = relation_with_ids(&gone, &ids).unwrap();
        let mut standalone = extension_of(&join(&placeholder(&gone, &ids)));
        let mut marked = extension_of(&join(&placeholder_relation(&gone, &ids)));
        std::mem::drop(gone);
        resolve_relation(&mut standalone).unwrap();
        assert_eq!(standalone, extension_of(&join(&gone_relation)));
        let error = resolve_relation(&mut marked).unwrap_err();
        assert!(error.to_string().contains("no longer exists"), "{error}");
    }

    #[test]
    fn placeholders_resolve_under_the_subquery_aliases_of_python_plans() {
        // As the SubqueryAlias of pyspark.sql.connect.plan over the plan of a DataFrame
        // (`SubqueryAlias(df._plan, "s")`), the views of its SQL, which are such aliases among
        // the references of a WithRelations, and an alias over a plugin's extension that holds
        // the plan of a DataFrame: the placeholders are replaced, a level below the alias.
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let (relation, levels) = relation_with_ids_and_levels(&node, &ids).unwrap();
        let shallow = placeholder(&node, &ids);
        let alias = |input: proto::Relation| proto::Relation {
            common: None,
            rel_type: Some(proto::relation::RelType::SubqueryAlias(Box::new(
                proto::SubqueryAlias {
                    input: Some(Box::new(input)),
                    alias: "s".to_string(),
                    qualifier: Vec::new(),
                },
            ))),
        };
        let mut plan = alias(alias(shallow.clone()));
        assert_eq!(
            resolve_relation_levels(&mut plan, None).unwrap(),
            2 + levels
        );
        assert_eq!(plan, alias(alias(relation.clone())));

        let sql = |view: proto::Relation| proto::Relation {
            common: None,
            rel_type: Some(proto::relation::RelType::WithRelations(Box::new(
                proto::WithRelations {
                    root: Some(Box::new(proto::Relation {
                        common: None,
                        rel_type: Some(proto::relation::RelType::Sql(proto::Sql {
                            query: "SELECT * FROM s".to_string(),
                            ..Default::default()
                        })),
                    })),
                    references: vec![view],
                },
            ))),
        };
        let mut plan = sql(alias(shallow.clone()));
        resolve_relation(&mut plan).unwrap();
        assert_eq!(plan, sql(alias(relation.clone())));

        let plugin = |input: &proto::Relation| proto::SubqueryAlias {
            input: Some(Box::new(input.clone())),
            alias: "p".to_string(),
            qualifier: Vec::new(),
        };
        let mut plan = alias(extension_of(&plugin(&shallow)));
        resolve_relation(&mut plan).unwrap();
        assert_eq!(plan, alias(extension_of(&plugin(&relation))));

        // And under a relation of another type, as a plan that Python builds of its own relations
        // holds one.
        let limit = |input: proto::Relation| proto::Relation {
            common: None,
            rel_type: Some(proto::relation::RelType::Limit(Box::new(proto::Limit {
                input: Some(Box::new(input)),
                limit: 1,
            }))),
        };
        let mut plan = alias(limit(shallow.clone()));
        assert_eq!(
            resolve_relation_levels(&mut plan, None).unwrap(),
            2 + levels
        );
        assert_eq!(plan, alias(limit(relation)));
    }

    /// `rel_type` as a relation of no common.
    fn relation_of(rel_type: proto::relation::RelType) -> proto::Relation {
        proto::Relation {
            common: None,
            rel_type: Some(rel_type),
        }
    }

    /// Relations of each shape of input of Spark Connect over `input` and `other`: one input, two
    /// sides, and the extra input of a group map and of a cogroup map.
    fn relations_over(input: &proto::Relation, other: &proto::Relation) -> Vec<proto::Relation> {
        use proto::relation::RelType;
        let boxed = |relation: &proto::Relation| Some(Box::new(relation.clone()));
        vec![
            relation_of(RelType::Project(Box::new(proto::Project {
                input: boxed(input),
                expressions: Vec::new(),
            }))),
            relation_of(RelType::Filter(Box::new(proto::Filter {
                input: boxed(input),
                condition: None,
            }))),
            relation_of(RelType::Join(Box::new(proto::Join {
                left: boxed(input),
                right: boxed(other),
                ..Default::default()
            }))),
            relation_of(RelType::SetOp(Box::new(proto::SetOperation {
                left_input: boxed(input),
                right_input: boxed(other),
                ..Default::default()
            }))),
            relation_of(RelType::Aggregate(Box::new(proto::Aggregate {
                input: boxed(input),
                ..Default::default()
            }))),
            relation_of(RelType::Sort(Box::new(proto::Sort {
                input: boxed(input),
                ..Default::default()
            }))),
            relation_of(RelType::GroupMap(Box::new(proto::GroupMap {
                input: boxed(input),
                initial_input: boxed(other),
                ..Default::default()
            }))),
            relation_of(RelType::CoGroupMap(Box::new(proto::CoGroupMap {
                input: boxed(input),
                other: boxed(other),
                ..Default::default()
            }))),
            relation_of(RelType::AsOfJoin(Box::new(proto::AsOfJoin {
                left: boxed(input),
                right: boxed(other),
                ..Default::default()
            }))),
            relation_of(RelType::LateralJoin(Box::new(proto::LateralJoin {
                left: boxed(input),
                right: boxed(other),
                ..Default::default()
            }))),
            relation_of(RelType::Zip(Box::new(proto::Zip {
                left: boxed(input),
                right: boxed(other),
            }))),
            relation_of(RelType::CollectMetrics(Box::new(proto::CollectMetrics {
                input: boxed(input),
                ..Default::default()
            }))),
            relation_of(RelType::Describe(Box::new(proto::StatDescribe {
                input: boxed(input),
                cols: Vec::new(),
            }))),
            relation_of(RelType::WithRelations(Box::new(proto::WithRelations {
                root: boxed(input),
                references: vec![other.clone()],
            }))),
        ]
    }

    #[test]
    fn placeholders_resolve_under_every_input_of_the_relations_of_spark_connect() {
        // As PySpark's plans hold the relations of their DataFrames under any relation, such as
        // the plans that `SparkSession.client` is given: each input of each shape, a level below
        // its relation.
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let other_node = Arc::new(range(0, 5, 1));
        let (relation, levels) = relation_with_ids_and_levels(&node, &ids).unwrap();
        let other = relation_with_ids(&other_node, &ids).unwrap();
        let (shallow, shallow_other) = (placeholder(&node, &ids), placeholder(&other_node, &ids));
        let plans = relations_over(&shallow, &shallow_other);
        let expected = relations_over(&relation, &other);
        assert_eq!(plans.len(), expected.len());
        for (mut plan, expected) in plans.into_iter().zip(expected) {
            assert_eq!(
                resolve_relation_levels(&mut plan, None).unwrap(),
                1 + levels
            );
            assert_eq!(plan, expected);
        }
        // And down a chain of them, of its levels.
        let limit = |input: proto::Relation| {
            relation_of(proto::relation::RelType::Limit(Box::new(proto::Limit {
                input: Some(Box::new(input)),
                limit: 1,
            })))
        };
        let mut plan = limit(limit(limit(shallow)));
        assert_eq!(
            resolve_relation_levels(&mut plan, None).unwrap(),
            3 + levels
        );
        assert_eq!(plan, limit(limit(limit(relation))));
    }

    #[test]
    fn placeholders_resolve_in_the_relations_of_commands() {
        // As PySpark's commands hold the relations of their DataFrames: of a write, a view, a
        // stream, a SQL statement, a checkpoint, a MERGE INTO and a pipeline's flow.
        use proto::command::CommandType;
        use proto::pipeline_command::{
            define_flow, CommandType as Pipeline, DefineFlow, DefineFlowQueryFunctionResult,
        };
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let relation = relation_with_ids(&node, &ids).unwrap();
        let shallow = placeholder(&node, &ids);
        let commands = |input: &proto::Relation| {
            let input = Some(input.clone());
            let pipeline = |command: Pipeline| {
                CommandType::PipelineCommand(proto::PipelineCommand {
                    command_type: Some(command),
                })
            };
            vec![
                CommandType::WriteOperation(proto::WriteOperation {
                    input: input.clone(),
                    ..Default::default()
                }),
                CommandType::CreateDataframeView(proto::CreateDataFrameViewCommand {
                    input: input.clone(),
                    ..Default::default()
                }),
                CommandType::WriteOperationV2(proto::WriteOperationV2 {
                    input: input.clone(),
                    ..Default::default()
                }),
                CommandType::SqlCommand(proto::SqlCommand {
                    input: input.clone(),
                    ..Default::default()
                }),
                CommandType::WriteStreamOperationStart(proto::WriteStreamOperationStart {
                    input: input.clone(),
                    ..Default::default()
                }),
                CommandType::CheckpointCommand(proto::CheckpointCommand {
                    relation: input.clone(),
                    ..Default::default()
                }),
                CommandType::MergeIntoTableCommand(proto::MergeIntoTableCommand {
                    source_table_plan: input.clone(),
                    ..Default::default()
                }),
                pipeline(Pipeline::DefineFlow(DefineFlow {
                    details: Some(define_flow::Details::RelationFlowDetails(
                        define_flow::WriteRelationFlowDetails {
                            relation: input.clone(),
                        },
                    )),
                    ..Default::default()
                })),
                pipeline(Pipeline::DefineFlowQueryFunctionResult(
                    DefineFlowQueryFunctionResult {
                        relation: input,
                        ..Default::default()
                    },
                )),
            ]
        };
        let (sent, expected) = (commands(&shallow), commands(&relation));
        assert_eq!(sent.len(), expected.len());
        for (command_type, expected) in sent.into_iter().zip(expected) {
            let mut command = proto::Command {
                command_type: Some(command_type),
            };
            resolve_command(&mut command, None).unwrap();
            assert_eq!(command.command_type, Some(expected));
        }
    }

    #[test]
    fn relations_without_placeholders_are_left_as_they_are() {
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let mut relation = relation_with_ids(&node, &ids).unwrap();
        let before = relation.clone();
        resolve_relation(&mut relation).unwrap();
        assert_eq!(relation, before);
        let mut command = proto::Command::default();
        resolve_command(&mut command, None).unwrap();
        assert_eq!(command, proto::Command::default());
    }

    #[test]
    fn a_placeholder_of_a_dropped_node_stands_for_its_relation_and_keeps_nothing_alive() {
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let relation = relation_with_ids(&node, &ids).unwrap();
        let alive = Arc::downgrade(&node);
        let standalone = placeholder(&node, &ids);
        std::mem::drop(node);
        assert!(alive.upgrade().is_none());
        let mut resolved = transform_of(standalone);
        resolve_relation(&mut resolved).unwrap();
        assert_eq!(resolved, transform_of(relation));
    }

    #[test]
    fn a_malformed_placeholder_is_an_error() {
        let mut relation = proto::Relation {
            common: None,
            rel_type: Some(proto::relation::RelType::Extension(prost_types::Any {
                type_url: PLACEHOLDER_TYPE_URL.to_string(),
                value: vec![1, 2, 3],
            })),
        };
        let error = resolve_relation(&mut relation).unwrap_err();
        assert!(error.to_string().contains("no marker"), "{error}");
    }
}

#[cfg(test)]
mod deep_plan_tests {
    use super::*;
    use spark_connect_core::error::SparkErrorKind;

    /// A plan of `levels` levels: a range under limits.
    fn chain(levels: usize) -> Arc<LogicalPlan> {
        let mut plan = Arc::new(range(0, 10, 1));
        for _ in 1..levels {
            plan = Arc::new(limit(plan, 1));
        }
        plan
    }

    /// Run `f` on a thread with a stack of 512 KiB, the smallest that the client supports.
    fn on_a_small_stack<T: Send>(f: impl FnOnce() -> T + Send) -> T {
        std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(512 << 10)
                .spawn_scoped(scope, f)
                .unwrap()
                .join()
                .unwrap()
        })
    }

    /// How many relations are nested in `relation`, following the inputs of limits.
    fn nested_limits(relation: &proto::Relation) -> usize {
        let mut levels = 1;
        let mut relation = relation;
        while let Some(proto::relation::RelType::Limit(limit)) = &relation.rel_type {
            relation = limit.input.as_deref().expect("input");
            levels += 1;
        }
        levels
    }

    #[test]
    fn depth_counts_the_levels_of_the_deepest_input() {
        assert_eq!(chain(1).depth(), 1);
        assert_eq!(chain(5).depth(), 5);
        let with = LogicalPlan::WithRelations {
            root: chain(3),
            references: vec![chain(2), chain(7)],
        };
        assert_eq!(with.depth(), 8);
    }

    #[test]
    fn depth_visits_an_input_that_plans_share_once() {
        // Each level refers to the one below twice: 2^64 paths, but 65 nodes.
        let mut plan = Arc::new(range(0, 10, 1));
        for _ in 0..64 {
            plan = Arc::new(LogicalPlan::WithRelations {
                root: Arc::clone(&plan),
                references: vec![plan],
            });
        }
        assert_eq!(plan.depth(), 65);
    }

    #[test]
    fn a_plan_at_the_bound_is_serialized_on_a_thread_with_a_small_stack() {
        let plan = chain(MAX_PLAN_DEPTH);
        let ids = PlanIds::new();
        let relation = on_a_small_stack(|| relation_with_ids(&plan, &ids).unwrap());
        assert_eq!(nested_limits(&relation), MAX_PLAN_DEPTH);
    }

    #[test]
    fn a_plan_past_the_bound_is_a_recursion_error_as_in_pyspark() {
        for levels in [MAX_PLAN_DEPTH + 1, 2_000, 100_000] {
            let plan = chain(levels);
            let error = on_a_small_stack(|| relation_with_ids(&plan, &PlanIds::new()).unwrap_err());
            assert_eq!(
                error.kind,
                SparkErrorKind::RecursionError,
                "{levels}: {error}"
            );
            assert!(
                error
                    .message()
                    .contains(&format!("a plan of {levels} levels")),
                "{error}"
            );
        }
    }

    /// A projection of `id + 1` chained `levels` times, over a plan of `plan_levels` levels.
    fn projection_of_additions(plan_levels: usize, levels: usize) -> Arc<LogicalPlan> {
        let mut expr = Expression::column("id");
        for _ in 0..levels {
            expr = Expression::function(
                "+",
                vec![
                    expr,
                    Expression::literal(crate::expression::LiteralExpression::Integer(1)),
                ],
            );
        }
        Arc::new(LogicalPlan::Project {
            input: chain(plan_levels),
            columns: vec![crate::column::Column::new(expr)],
        })
    }

    /// An expression deeper than `MAX_EXPRESSION_DEPTH` is a `RecursionError`, as PySpark's
    /// serializer raises it, whether the plan is serialized on the calling thread or on one of
    /// its own, and an expression at the bound is serialized; neither overflows a small stack.
    #[test]
    fn an_expression_past_the_bound_is_a_recursion_error_as_in_pyspark() {
        use crate::expression::MAX_EXPRESSION_DEPTH;
        for plan_levels in [1, 100] {
            let at_bound = projection_of_additions(plan_levels, MAX_EXPRESSION_DEPTH - 1);
            on_a_small_stack(|| relation_with_ids(&at_bound, &PlanIds::new()).unwrap());
            for levels in [MAX_EXPRESSION_DEPTH, 100_000] {
                let plan = projection_of_additions(plan_levels, levels);
                let error =
                    on_a_small_stack(|| relation_with_ids(&plan, &PlanIds::new()).unwrap_err());
                assert_eq!(
                    error.kind,
                    SparkErrorKind::RecursionError,
                    "{levels}: {error}"
                );
                assert!(
                    error.message().contains("an expression may be at most"),
                    "{error}"
                );
                on_a_small_stack(move || std::mem::drop(plan));
            }
        }
    }

    /// `limits` limits over a filter whose condition is the EXISTS subquery of a plan of
    /// `levels` levels.
    fn limits_over_a_filter_of_a_subquery(limits: usize, levels: usize) -> Arc<LogicalPlan> {
        let subquery = crate::expression::SubqueryExpression::new(
            chain(levels),
            1,
            crate::expression::SubqueryType::Exists,
        );
        let mut plan = Arc::new(LogicalPlan::Filter {
            input: Arc::new(range(0, 10, 1)),
            condition: crate::column::Column::new(Expression::new(ExpressionNode::Subquery(
                Box::new(subquery),
            ))),
        });
        for _ in 0..limits {
            plan = Arc::new(limit(plan, 1));
        }
        plan
    }

    /// The plan of a subquery is sent under the plan that has it, so its levels count for the
    /// bound under the levels above it, and neither overflows a small stack.
    #[test]
    fn a_subquery_counts_for_the_bound_under_the_levels_above_it() {
        for limits in [0, 33] {
            // The filter is a level under the limits, and the subquery one under the filter.
            let within = limits_over_a_filter_of_a_subquery(limits, MAX_PLAN_DEPTH - limits - 1);
            on_a_small_stack(|| relation_with_ids(&within, &PlanIds::new()).unwrap());
            let past = limits_over_a_filter_of_a_subquery(limits, MAX_PLAN_DEPTH - limits);
            let error = on_a_small_stack(|| relation_with_ids(&past, &PlanIds::new()).unwrap_err());
            assert_eq!(error.kind, SparkErrorKind::RecursionError, "{error}");
            assert!(
                error.message().contains("the plans of its subqueries"),
                "{error}"
            );
        }
        let deep = limits_over_a_filter_of_a_subquery(33, 100_000);
        let error = on_a_small_stack(|| relation_with_ids(&deep, &PlanIds::new()).unwrap_err());
        assert_eq!(error.kind, SparkErrorKind::RecursionError, "{error}");
        on_a_small_stack(move || std::mem::drop(deep));
    }

    /// The proto of a plan at the bound is encoded and dropped by recursing, as the transport
    /// encodes a request on a thread of the runtime, whose stack is 2 MiB.
    #[test]
    fn a_plan_at_the_bound_is_encoded_on_a_thread_of_the_runtime() {
        use prost::Message;

        let relation = relation_with_ids(&chain(MAX_PLAN_DEPTH), &PlanIds::new()).unwrap();
        let encoded = std::thread::Builder::new()
            .stack_size(2 << 20)
            .spawn(move || relation.encode_to_vec().len())
            .unwrap()
            .join()
            .unwrap();
        assert!(encoded > MAX_PLAN_DEPTH);
    }

    #[test]
    fn dropping_a_deep_plan_does_not_overflow_a_small_stack() {
        on_a_small_stack(|| std::mem::drop(chain(1_000_000)));
    }

    #[test]
    fn dropping_a_plan_keeps_the_inputs_that_another_plan_holds() {
        let shared = chain(10);
        let other = Arc::new(limit(Arc::clone(&shared), 1));
        std::mem::drop(chain(3));
        std::mem::drop(Arc::new(limit(Arc::clone(&shared), 1)));
        assert_eq!(shared.depth(), 10);
        assert_eq!(other.depth(), 11);
        let relation = relation_with_ids(&other, &PlanIds::new()).unwrap();
        assert_eq!(nested_limits(&relation), 11);
    }

    /// Run `f` on a thread whose stack takes the comparisons and drops of deep protos, which
    /// recurse, also in a debug build.
    fn on_a_large_stack<T: Send>(f: impl FnOnce() -> T + Send) -> T {
        std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(64 << 20)
                .spawn_scoped(scope, f)
                .unwrap()
                .join()
                .unwrap()
        })
    }

    /// A relation built outside the core over a DataFrame of `levels` levels, as `pyspark.ml`
    /// builds a transform over the placeholder of its input, resolved as `from_relation` does.
    fn transform_of_a_plan(levels: usize, ids: &Arc<PlanIds>) -> Arc<LogicalPlan> {
        let input = chain(levels);
        let mut relation = proto::Relation {
            common: None,
            rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                proto::MlRelation {
                    model_summary_dataset: None,
                    ml_type: Some(proto::ml_relation::MlType::Transform(Box::new(
                        proto::ml_relation::Transform {
                            input: Some(Box::new(placeholder_relation(&input, ids))),
                            params: None,
                            operator: None,
                        },
                    ))),
                },
            ))),
        };
        let depth = resolve_relation_levels(&mut relation, None).unwrap();
        Arc::new(LogicalPlan::Raw {
            relation: Box::new(relation),
            owner: None,
            depth,
            observations: Vec::new(),
            holes: Default::default(),
        })
    }

    /// As PySpark serializes the plan of its input within that of a plan of `pyspark.ml`, the
    /// levels of the DataFrame under a relation built outside the core count for the bound,
    /// under the levels above that relation.
    #[test]
    fn the_levels_of_the_input_of_a_relation_built_outside_the_core_count_as_in_pyspark() {
        on_a_large_stack(|| {
            let ids = Arc::new(PlanIds::new());
            let transform = transform_of_a_plan(400, &ids);
            // The transform is a level over the 400 of its input.
            assert_eq!(transform.depth(), 401);
            let mut plan = Arc::clone(&transform);
            for _ in 0..MAX_PLAN_DEPTH - 401 {
                plan = Arc::new(limit(plan, 1));
            }
            assert_eq!(plan.depth(), MAX_PLAN_DEPTH);
            relation_with_ids(&plan, &ids).unwrap();
            let past = Arc::new(limit(plan, 1));
            let error = relation_with_ids(&past, &ids).unwrap_err();
            assert_eq!(error.kind, SparkErrorKind::RecursionError, "{error}");
            assert!(
                error
                    .message()
                    .contains(&format!("a plan of {} levels", MAX_PLAN_DEPTH + 1)),
                "{error}"
            );
            assert!(past.try_to_proto().is_err());
            // In a subquery, which the levels of the plan do not count, they count as serialized.
            let subquery = crate::expression::SubqueryExpression::new(
                transform_of_a_plan(MAX_PLAN_DEPTH - 2, &ids),
                1,
                crate::expression::SubqueryType::Exists,
            );
            let filter = Arc::new(LogicalPlan::Filter {
                input: Arc::new(range(0, 10, 1)),
                condition: crate::column::Column::new(Expression::new(ExpressionNode::Subquery(
                    Box::new(subquery),
                ))),
            });
            relation_with_ids(&filter, &ids).unwrap();
            let error = relation_with_ids(&Arc::new(limit(filter, 1)), &ids).unwrap_err();
            assert!(
                error.message().contains("the plans of its subqueries"),
                "{error}"
            );
        });
    }

    /// The relation of a plan of `pyspark.ml` at the bound, over a DataFrame of as many levels,
    /// is copied into the request on a thread with a small stack, which grows for its levels.
    #[test]
    fn a_relation_built_outside_the_core_at_the_bound_is_serialized_on_a_small_stack() {
        on_a_large_stack(|| {
            let ids = Arc::new(PlanIds::new());
            let transform = transform_of_a_plan(MAX_PLAN_DEPTH - 1, &ids);
            assert_eq!(transform.depth(), MAX_PLAN_DEPTH);
            let relation = on_a_small_stack(|| relation_with_ids(&transform, &ids).unwrap());
            let Some(proto::relation::RelType::MlRelation(ml)) = &relation.rel_type else {
                panic!("an ML relation");
            };
            let Some(proto::ml_relation::MlType::Transform(transform)) = &ml.ml_type else {
                panic!("a transform");
            };
            assert_eq!(
                nested_limits(transform.input.as_deref().unwrap()),
                MAX_PLAN_DEPTH - 1
            );
        });
    }

    /// `try_to_proto` bounds a plan as a request does, and `to_proto` serializes one of any
    /// depth, on a small stack.
    #[test]
    fn try_to_proto_bounds_the_plan_as_a_request_and_to_proto_serializes_any_depth() {
        on_a_large_stack(|| {
            let at_bound = chain(MAX_PLAN_DEPTH);
            let relation = on_a_small_stack(|| at_bound.try_to_proto().unwrap());
            assert_eq!(nested_limits(&relation), MAX_PLAN_DEPTH);
            assert!(relation.common.unwrap().plan_id.is_none());
            for levels in [MAX_PLAN_DEPTH + 1, 100_000] {
                let plan = chain(levels);
                let error = on_a_small_stack(|| plan.try_to_proto().unwrap_err());
                assert_eq!(error.kind, SparkErrorKind::RecursionError, "{error}");
                on_a_small_stack(move || std::mem::drop(plan));
            }
            let deep = chain(10_000);
            let relation = on_a_small_stack(|| deep.to_proto());
            assert_eq!(nested_limits(&relation), 10_000);
        });
    }

    /// A plan that fails to build with an error of `value` as its message.
    fn unbuildable(input: Arc<LogicalPlan>, value: &str) -> Arc<LogicalPlan> {
        let error = SparkError::classed(
            SparkErrorKind::ValueError,
            "UNSUPPORTED_OPERATION",
            &[("operation", &format!("data type {value}"))],
        );
        Arc::new(LogicalPlan::Unbuildable { input, error })
    }

    #[test]
    fn an_unbuildable_plan_fails_each_request_with_its_error() {
        let plan = Arc::new(limit(unbuildable(Arc::new(range(0, 10, 1)), "a bigint"), 1));
        for error in [
            relation_with_ids(&plan, &PlanIds::new()).unwrap_err(),
            plan.try_to_proto().unwrap_err(),
        ] {
            assert_eq!(error.kind, SparkErrorKind::ValueError, "{error}");
            assert_eq!(error.error_class, "UNSUPPORTED_OPERATION");
            assert!(error.message().contains("data type a bigint"), "{error}");
        }
    }

    #[test]
    fn the_error_of_an_unbuildable_plan_is_that_of_its_deepest_leftmost_input() {
        let left = unbuildable(Arc::new(range(0, 10, 1)), "left");
        let right = unbuildable(Arc::new(range(0, 10, 1)), "right");
        let both = LogicalPlan::SetOperation {
            left: unbuildable(Arc::clone(&left), "outer left"),
            right,
            set_op_type: SetOpType::Union,
            is_all: true,
            by_name: true,
            allow_missing_columns: false,
        };
        let error = both.try_to_proto().unwrap_err();
        assert!(error.message().contains("data type left"), "{error}");
        assert!(range(0, 10, 1).try_to_proto().is_ok());
    }

    /// An expression of a plan that a request fails to build, below an unbuildable plan,
    /// raises its error first, as PySpark builds the input of a plan before the plan; and one
    /// above it raises its error after that of the plan.
    #[test]
    fn unbuildable_plans_and_expressions_fail_in_the_order_of_pysparks_build() {
        let unbuildable_column = |name: &str| {
            let error = SparkError::classed(
                SparkErrorKind::ValueError,
                "UNSUPPORTED_OPERATION",
                &[("operation", &format!("data type {name}"))],
            );
            crate::column::Column::new(Expression::unbuildable(
                Expression::column(name),
                Some(error),
                None,
                None,
            ))
        };
        let filter = |input: Arc<LogicalPlan>, name: &str| {
            Arc::new(LogicalPlan::Filter {
                input,
                condition: unbuildable_column(name),
            })
        };
        let below = unbuildable(filter(Arc::new(range(0, 10, 1)), "inner"), "plan");
        let error = relation_with_ids(&below, &PlanIds::new()).unwrap_err();
        assert!(error.message().contains("data type inner"), "{error}");
        let above = filter(unbuildable(Arc::new(range(0, 10, 1)), "plan"), "outer");
        let error = relation_with_ids(&above, &PlanIds::new()).unwrap_err();
        assert!(error.message().contains("data type plan"), "{error}");
    }

    /// As PySpark's pickle of a Column of a subquery, which keeps the plan objects of its
    /// subqueries, of their plan IDs: the expression that the pickle sends, with the relations of
    /// the plans of its subqueries (`subquery_relations`), is sent in a relation as the expression
    /// was, with the same relations in its `WithRelations`, also from another session's plan IDs,
    /// under another expression and pickled again.
    #[test]
    fn a_sent_expression_of_subqueries_sends_their_plans_as_the_expression_did() {
        use crate::expression::{SubqueryExpression, SubqueryType};

        let ids = Arc::new(PlanIds::new());
        let inner = Arc::new(LogicalPlan::Project {
            input: Arc::new(range(0, 5, 1)),
            columns: vec![crate::column::Column::new(Expression::function(
                "max",
                vec![Expression::column("id")],
            ))],
        });
        let subquery = SubqueryExpression {
            plan_ids: Some(Arc::clone(&ids)),
            ..SubqueryExpression::new(Arc::clone(&inner), ids.id_of(&inner), SubqueryType::Scalar)
        };
        let expression = Expression::function(
            "+",
            vec![
                Expression::column("id"),
                Expression::new(ExpressionNode::Subquery(Box::new(subquery))),
            ],
        );
        let project = |expression: &Expression| {
            Arc::new(LogicalPlan::Project {
                input: Arc::new(range(0, 3, 1)),
                columns: vec![crate::column::Column::new(expression.clone())],
            })
        };
        // The expressions of the root of the `WithRelations` of `relation`, and its references.
        let sent = |relation: proto::Relation| {
            let Some(proto::relation::RelType::WithRelations(with)) = relation.rel_type else {
                panic!("expected WithRelations: {relation:?}");
            };
            let Some(proto::relation::RelType::Project(root)) = with.root.and_then(|r| r.rel_type)
            else {
                panic!("expected a projection");
            };
            (root.expressions, with.references)
        };
        let original = sent(relation_with_ids(&project(&expression), &ids).unwrap());
        let relations = expression.subquery_relations().unwrap();
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].0, original.1[0]);
        let (built, levels) = expression.try_to_proto_with_levels().unwrap();
        let unpickled =
            Expression::sent_with_subqueries(built, expression.render(), levels, relations.clone())
                .unwrap();
        assert_eq!(unpickled.render(), expression.render());
        assert_eq!(
            sent(relation_with_ids(&project(&unpickled), &PlanIds::new()).unwrap()),
            original
        );
        assert_eq!(unpickled.subquery_relations().unwrap(), relations);
        let over = Expression::function("abs", vec![unpickled]);
        assert_eq!(over.subquery_relations().unwrap(), relations);
        let (_, references) = sent(relation_with_ids(&project(&over), &PlanIds::new()).unwrap());
        assert_eq!(references, original.1);
        // A relation deeper than a plan may be is the `RecursionError` of a pickled DataFrame's.
        let error = Expression::sent_with_subqueries(
            proto::Expression::default(),
            String::new(),
            1,
            vec![(relations[0].0.clone(), MAX_PLAN_DEPTH + 1)],
        )
        .unwrap_err();
        assert_eq!(error.kind, SparkErrorKind::RecursionError, "{error}");
    }

    /// The plan of a subquery that a request fails to build fails the request of the plan
    /// that refers to it, as PySpark builds the plans of its references with the plan.
    #[test]
    fn an_unbuildable_plan_of_a_subquery_fails_the_request() {
        let subquery = crate::expression::SubqueryExpression::new(
            unbuildable(Arc::new(range(0, 10, 1)), "subquery"),
            7,
            crate::expression::SubqueryType::Scalar,
        );
        let plan = Arc::new(LogicalPlan::Project {
            input: Arc::new(range(0, 1, 1)),
            columns: vec![crate::column::Column::new(Expression::new(
                ExpressionNode::Subquery(Box::new(subquery)),
            ))],
        });
        let error = relation_with_ids(&plan, &PlanIds::new()).unwrap_err();
        assert!(error.message().contains("data type subquery"), "{error}");
    }
}
