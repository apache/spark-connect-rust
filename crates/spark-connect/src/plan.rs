//! Logical plan nodes mirroring `pyspark.sql.connect.plan`.
//!
//! Each node builds a `spark.connect.Relation` (or `Command`) protobuf matching
//! the reference PySpark client (see `tests/golden/plans.jsonl`).

use spark_connect_proto as proto;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

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
#[derive(Clone)]
pub struct RelationOwner(pub Arc<dyn std::any::Any + Send + Sync>);

impl std::fmt::Debug for RelationOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RelationOwner")
    }
}

#[derive(Debug, Clone)]
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
    /// core, and the owner of the relation, if any, which the plan keeps alive.
    Raw {
        relation: Box<proto::Relation>,
        owner: Option<RelationOwner>,
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
        metadata_json: String,
    },
    /// WithColumnsRenamed: `df.withColumnRenamed(old, new)` or `df.withColumnsRenamed(...)`.
    WithColumnsRenamed {
        input: Arc<LogicalPlan>,
        renames: HashMap<String, String>,
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
    /// CachedRemoteRelation: a reference to a cached remote relation.
    CachedRemoteRelation { relation_id: String },
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
    /// StatCorr: `df.stat.corr(col1, col2)`.
    StatCorr {
        input: Arc<LogicalPlan>,
        col1: String,
        col2: String,
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
    /// ML Transformation: wraps an MlRelation proto directly.
    MlTransform { ml_relation: proto::Relation },
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
    /// ApplyInPandasWithState: `GroupedData.applyInPandasWithState`. The schemas are sent
    /// as JSON.
    ApplyInPandasWithState {
        input: Arc<LogicalPlan>,
        grouping_expressions: Vec<Expression>,
        func: CommonInlineUserDefinedFunctionExpression,
        output_schema: DataType,
        state_schema: DataType,
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
    },
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

/// Set operation type (UNION, INTERSECT, EXCEPT).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOpType {
    Union,
    Intersect,
    Except,
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
    next: i64,
    ids: HashMap<usize, (Weak<LogicalPlan>, i64)>,
    prune_at: usize,
}

impl PlanIds {
    const MIN_PRUNE_AT: usize = 1024;

    pub(crate) fn new() -> Self {
        PlanIds {
            nodes: Mutex::new(PlanIdNodes {
                next: 1,
                ids: HashMap::new(),
                prune_at: Self::MIN_PRUNE_AT,
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
        if let Some((_, id)) = nodes.ids.get(&key) {
            return *id;
        }
        if nodes.ids.len() >= nodes.prune_at {
            nodes.ids.retain(|_, (node, _)| node.strong_count() > 0);
            nodes.prune_at = (nodes.ids.len() * 2).max(Self::MIN_PRUNE_AT);
        }
        let id = nodes.next;
        nodes.next += 1;
        nodes.ids.insert(key, (Arc::downgrade(node), id));
        id
    }

    /// The live node of the plan ID `id`, as the server reports a relation by its plan ID.
    pub(crate) fn node_of(&self, id: i64) -> Option<Arc<LogicalPlan>> {
        let nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        nodes
            .ids
            .values()
            .find(|(_, node_id)| *node_id == id)
            .and_then(|(node, _)| node.upgrade())
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.nodes.lock().unwrap().ids.len()
    }
}

/// Serialize a shared plan node for a request: the node and every node below it carry
/// their stable plan IDs.
pub(crate) fn relation_with_ids(node: &Arc<LogicalPlan>, ids: &PlanIds) -> proto::Relation {
    relation_of(node, Some(ids))
}

/// Serialize a child node, with its stable plan ID when `ids` is given.
fn relation_of(node: &Arc<LogicalPlan>, ids: Option<&PlanIds>) -> proto::Relation {
    relation_with_references(node, Some(node), ids)
}

thread_local! {
    /// For each relation being serialized, innermost last, the plans that the subquery
    /// expressions of its own expressions refer to.
    static SUBQUERY_REFERENCES: std::cell::RefCell<Vec<Vec<Arc<LogicalPlan>>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Record that a subquery expression being serialized refers to `plan`, which the relation
/// that contains the expression sends in its `WithRelations`.
pub(crate) fn refer_to(plan: &Arc<LogicalPlan>) {
    SUBQUERY_REFERENCES.with(|frames| {
        if let Some(frame) = frames.borrow_mut().last_mut() {
            frame.push(Arc::clone(plan));
        }
    });
}

/// The relation of `node`, and the plans that the subqueries of its own expressions refer
/// to; its children collect those of theirs as they are serialized.
fn relation_and_references(
    node: &LogicalPlan,
    ids: Option<&PlanIds>,
) -> (proto::Relation, Vec<Arc<LogicalPlan>>) {
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
    let with_relations = proto::WithRelations {
        root: Some(Box::new(relation)),
        references: references
            .iter()
            .map(|reference| relation_of(reference, ids))
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

impl LogicalPlan {
    /// Convert this logical plan to a `spark.connect.Relation` protobuf, without plan IDs.
    pub fn to_proto(&self) -> proto::Relation {
        relation_with_references(self, None, None)
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

            LogicalPlan::Raw { relation: raw, .. } => {
                relation = raw.as_ref().clone();
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
                repart.input = Some(Box::new(relation_of(input, plan_ids)));
                // num_partitions is optional in the proto: <= 0 means "unset" so the
                // server uses the default (the `df.repartition(*cols)` column-only form,
                // where no partition count is given). A real count is always positive.
                repart.num_partitions = if *num_partitions > 0 {
                    Some(*num_partitions)
                } else {
                    None
                };
                for expr in expressions {
                    repart.partition_exprs.push(expr.to_proto());
                }
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
                metadata_json,
            } => {
                let mut wc = proto::WithColumns::default();
                wc.input = Some(Box::new(relation_of(input, plan_ids)));
                let col_expr = crate::expression::Expression::column(column_name.clone());
                let mut alias = proto::expression::Alias::default();
                alias.expr = Some(Box::new(col_expr.to_proto()));
                alias.name = vec![column_name.clone()];
                alias.metadata = Some(metadata_json.clone());
                wc.aliases.push(alias);
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

            LogicalPlan::ToSchema { input, schema } => {
                let mut to_schema = proto::ToSchema::default();
                to_schema.input = Some(Box::new(relation_of(input, plan_ids)));
                to_schema.schema = Some(schema.to_proto());
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
                // omits this to avoid a redundant projection on every query.
                local.schema = schema.as_ref().map(DataType::json);
                relation.rel_type = Some(proto::relation::RelType::LocalRelation(local));
            }

            LogicalPlan::CachedRemoteRelation { relation_id } => {
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
                repart.input = Some(Box::new(relation_of(input, plan_ids)));
                if let Some(n) = num_partitions {
                    repart.num_partitions = Some(*n);
                }
                for expr in partition_exprs {
                    repart.partition_exprs.push(expr.to_proto());
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

            LogicalPlan::StatCorr { input, col1, col2 } => {
                let mut stat = proto::StatCorr::default();
                stat.input = Some(Box::new(relation_of(input, plan_ids)));
                stat.col1 = col1.clone();
                stat.col2 = col2.clone();
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
                input, name, exprs, ..
            } => {
                let mut collect_metrics = proto::CollectMetrics::default();
                collect_metrics.input = Some(Box::new(relation_of(input, plan_ids)));
                collect_metrics.name = name.clone();
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

            LogicalPlan::MlTransform { ml_relation } => {
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
            } => {
                let mut mp = proto::MapPartitions::default();
                mp.input = Some(Box::new(relation_of(input, plan_ids)));
                mp.func = Some(func.to_proto());
                mp.is_barrier = Some(*is_barrier);
                mp.profile_id = *profile_id;
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
                gm.func = Some(func.to_proto());
                for e in sorting_expressions {
                    gm.sorting_expressions.push(e.to_proto());
                }
                if let Some(ii) = initial_input {
                    gm.initial_input = Some(Box::new(relation_of(ii, plan_ids)));
                }
                for e in initial_grouping_expressions {
                    gm.initial_grouping_expressions.push(e.to_proto());
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
                            output_schema: t.output_schema.as_ref().map(|d| d.to_proto()),
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
                    output_schema: output_schema.json(),
                    state_schema: state_schema.json(),
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
            } => {
                let f = common_inline_udtf(
                    function_name,
                    *deterministic,
                    arguments,
                    return_type.as_ref(),
                    *eval_type,
                    command,
                    python_ver,
                );
                relation.rel_type =
                    Some(proto::relation::RelType::CommonInlineUserDefinedTableFunction(f));
            }
        }

        relation
    }
}

/// The `CommonInlineUserDefinedTableFunction` of a Python UDTF: of a call with `arguments`,
/// or of its registration without any. Mirrors PySpark's `CommonInlineUserDefinedTableFunction`
/// of a `PythonUDTF`.
pub(crate) fn common_inline_udtf(
    function_name: &str,
    deterministic: bool,
    arguments: &[Expression],
    return_type: Option<&DataType>,
    eval_type: i32,
    command: &[u8],
    python_ver: &str,
) -> proto::CommonInlineUserDefinedTableFunction {
    let udtf = proto::PythonUdtf {
        return_type: return_type.map(DataType::to_proto),
        eval_type,
        command: bytes::Bytes::copy_from_slice(command),
        python_ver: python_ver.to_string(),
    };
    proto::CommonInlineUserDefinedTableFunction {
        function_name: function_name.to_string(),
        deterministic,
        arguments: arguments.iter().map(Expression::to_proto).collect(),
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

/// Create a WithColumnsRenamed plan.
pub fn with_columns_renamed(
    input: impl Into<Arc<LogicalPlan>>,
    renames: HashMap<String, String>,
) -> LogicalPlan {
    LogicalPlan::WithColumnsRenamed {
        input: input.into(),
        renames,
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

/// Build a proto `Literal` from a [`crate::row::Value`] (used by NAFill).
fn value_to_proto_literal(v: &crate::row::Value) -> proto::expression::Literal {
    use crate::row::Value;
    use proto::expression::literal::LiteralType;
    let mut lit = proto::expression::Literal::default();
    lit.literal_type = Some(match v {
        Value::Bool(b) => LiteralType::Boolean(*b),
        Value::Byte(x) => LiteralType::Byte(*x as i32),
        Value::Short(x) => LiteralType::Short(*x as i32),
        Value::Integer(x) => LiteralType::Integer(*x),
        Value::Long(x) => LiteralType::Long(*x),
        Value::Float(x) => LiteralType::Float(*x),
        Value::Double(x) => LiteralType::Double(*x),
        Value::String(s) => LiteralType::String(s.clone()),
        Value::Date(d) => LiteralType::Date(*d),
        Value::Timestamp(t) => LiteralType::Timestamp(*t),
        Value::TimestampNtz(t) => LiteralType::TimestampNtz(*t),
        // Spark rejects interval NA fills. Preserve the type so it errors instead
        // of silently filling string columns with the Value's Debug text.
        Value::DayTimeInterval(t) => LiteralType::DayTimeInterval(*t),
        Value::Decimal {
            value,
            precision,
            scale,
        } => {
            let mut decimal = proto::expression::literal::Decimal::default();
            decimal.value = value.clone();
            if let Some(p) = precision {
                decimal.precision = Some(*p);
            }
            if let Some(s) = scale {
                decimal.scale = Some(*s);
            }
            LiteralType::Decimal(decimal)
        }
        other => LiteralType::String(format!("{:?}", other)),
    });
    lit
}

/// The literal of a replacement given as a string: a double for a numeric-looking string,
/// otherwise the string (used by the NAReplace of strings).
pub(crate) fn numeric_or_string_literal(s: &str) -> LiteralExpression {
    match s.parse::<f64>() {
        Ok(v) => LiteralExpression::double(v),
        Err(_) => LiteralExpression::string(s),
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

/// Create a NAReplace plan, of replacements given as strings: a numeric-looking string is a
/// double, otherwise a string.
pub fn na_replace(
    input: impl Into<Arc<LogicalPlan>>,
    replacements: Vec<(String, String)>,
    columns: Vec<String>,
) -> LogicalPlan {
    LogicalPlan::NAReplace {
        input: input.into(),
        replacements: replacements
            .iter()
            .map(|(old, new)| {
                (
                    numeric_or_string_literal(old),
                    numeric_or_string_literal(new),
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
        let relation = relation_with_ids(&plan, &ids);
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
            b"pickled",
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
        let dynamic = common_inline_udtf("dynamic", true, &[], None, 300, b"", "3.11");
        let Some(proto::common_inline_user_defined_table_function::Function::PythonUdtf(udtf)) =
            dynamic.function
        else {
            panic!("expected a PythonUDTF");
        };
        assert_eq!(udtf.return_type, None);
    }
}
