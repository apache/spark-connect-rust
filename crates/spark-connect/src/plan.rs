//! Logical plan nodes mirroring `pyspark.sql.connect.plan`.
//!
//! Each node builds a `spark.connect.Relation` (or `Command`) protobuf matching
//! the reference PySpark client (see `tests/golden/plans.jsonl`).

use smallvec::SmallVec;
use spark_connect_proto as proto;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, Weak};

use spark_connect_core::error::{Result, SparkError};

use crate::column::Column;
#[cfg(test)]
use crate::expression::ExpressionSpec;
use crate::expression::{Expression, ExpressionKind, LiteralExpression};
use crate::types::DataType;
use crate::udf::CommonInlineUserDefinedFunctionExpression;

#[cfg(test)]
mod structure_tests;

/// Parameters for the `transformWithState` / `transformWithStateInPandas` operators
/// (mirrors `spark.connect.TransformWithStateInfo`).
#[derive(Debug, Clone)]
pub struct TransformWithStateInfo {
    pub time_mode: String,
    pub event_time_column_name: Option<String>,
    pub output_schema: Option<DataType>,
}

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

/// An opaque protobuf relation with a measured depth for guarded cloning.
///
/// The protobuf is immutable through this wrapper, so its cached depth cannot
/// become stale. The guard covers relation nesting; protobuf expressions and
/// data types, encoding, and destruction retain their recursive behavior.
pub struct OpaqueRelation {
    relation: Box<proto::Relation>,
    depth: usize,
}

impl std::fmt::Debug for OpaqueRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpaqueRelation")
            .field("relation", &self.relation)
            .finish_non_exhaustive()
    }
}

impl OpaqueRelation {
    /// Measure and retain an opaque relation without cloning it.
    pub fn new(relation: impl Into<Box<proto::Relation>>) -> Self {
        let relation = relation.into();
        let depth = opaque_relation_depth(&relation);
        Self { relation, depth }
    }

    /// Borrow the protobuf without allowing changes to its measured depth.
    pub fn as_proto(&self) -> &proto::Relation {
        &self.relation
    }

    /// Take the protobuf out without cloning it.
    ///
    /// The returned value no longer has this wrapper's guarded clone operation.
    pub fn into_proto(self) -> proto::Relation {
        *self.relation
    }

    fn clone_proto(&self) -> proto::Relation {
        clone_opaque_relation(&self.relation, self.depth)
    }
}

impl Clone for OpaqueRelation {
    fn clone(&self) -> Self {
        Self {
            relation: Box::new(self.clone_proto()),
            depth: self.depth,
        }
    }
}

/// A client-side unresolved query node.
///
/// Operator payload and ordered shared inputs are independent. DataFrames retain
/// shared roots; cloning this value clones only its own payload while sharing its
/// inputs. Requests preserve stable node identity.
#[derive(Clone)]
pub struct LogicalPlan {
    kind: LogicalPlanKind,
    children: SmallVec<[Arc<LogicalPlan>; 2]>,
}

impl std::fmt::Debug for LogicalPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogicalPlan")
            .field("kind", &self.kind)
            .field("children", &self.children)
            .finish_non_exhaustive()
    }
}

/// An operator supplied with an invalid number of structural inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct InvalidPlanArity {
    /// Name of the operator whose input count was rejected.
    pub operator: &'static str,
    /// Smallest accepted number of inputs.
    pub expected_min: usize,
    /// Largest accepted number of inputs, or `None` when unbounded.
    pub expected_max: Option<usize>,
    /// Number of inputs supplied by the caller.
    pub actual: usize,
}

impl std::fmt::Display for InvalidPlanArity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = if self.expected_min == 1 {
            "input"
        } else {
            "inputs"
        };
        match self.expected_max {
            None => write!(
                f,
                "{} expects at least {} {}, got {}",
                self.operator, self.expected_min, label, self.actual
            ),
            Some(max) if self.expected_min == max => write!(
                f,
                "{} expects {} {}, got {}",
                self.operator, self.expected_min, label, self.actual
            ),
            Some(max) => write!(
                f,
                "{} expects {}..={} inputs, got {}",
                self.operator, self.expected_min, max, self.actual
            ),
        }
    }
}

impl std::error::Error for InvalidPlanArity {}

/// Why a plan's structural inputs could not be rebuilt safely.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanRebuildError {
    /// The supplied input count is outside the operator's accepted range.
    InvalidArity(InvalidPlanArity),
    /// Replacing an input would leave an opaque serialized relation out of sync.
    #[non_exhaustive]
    OpaqueInputReplacement {
        /// Name of the operator whose embedded input cannot be rewritten.
        operator: &'static str,
    },
}

impl From<InvalidPlanArity> for PlanRebuildError {
    fn from(error: InvalidPlanArity) -> Self {
        Self::InvalidArity(error)
    }
}

impl std::fmt::Display for PlanRebuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArity(error) => std::fmt::Display::fmt(error, f),
            Self::OpaqueInputReplacement { operator } => write!(
                f,
                "{} has an opaque wire input; rebuilding requires the original shared input",
                operator
            ),
        }
    }
}

impl std::error::Error for PlanRebuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidArity(error) => Some(error),
            Self::OpaqueInputReplacement { .. } => None,
        }
    }
}

impl LogicalPlan {
    /// Construct a node after checking its operator's input count.
    ///
    /// Binary inputs are left then right; CoGroupMap uses input then other.
    /// GroupMap uses its primary input then optional initial state. WithRelations
    /// uses its root followed by any number of references.
    /// MlTransform's keepalive input must correspond to its embedded wire input;
    /// construction checks the input count, not that correspondence.
    /// Raw's declared depth still governs request bounds independently of the
    /// actual opaque depth measured for clone stack sizing.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidPlanArity`] if the input count is outside the operator's
    /// arity bounds.
    pub fn try_new(
        kind: LogicalPlanKind,
        children: impl IntoIterator<Item = Arc<Self>>,
    ) -> std::result::Result<Self, InvalidPlanArity> {
        let children: SmallVec<[Arc<Self>; 2]> = children.into_iter().collect();
        Self::validate_arity(&kind, children.len())?;
        Ok(Self { kind, children })
    }

    fn validate_arity(
        kind: &LogicalPlanKind,
        actual: usize,
    ) -> std::result::Result<(), InvalidPlanArity> {
        let arity = kind.arity();
        if !arity.contains(&actual) {
            return Err(InvalidPlanArity {
                operator: kind.name(),
                expected_min: *arity.start(),
                expected_max: (*arity.end() != usize::MAX).then_some(*arity.end()),
                actual,
            });
        }
        Ok(())
    }

    /// Construct a node from a client builder's known input layout.
    ///
    /// # Panics
    ///
    /// Panics if the input count is outside the operator's arity bounds.
    #[track_caller]
    pub(crate) fn from_builder(
        kind: LogicalPlanKind,
        children: impl IntoIterator<Item = Arc<Self>>,
    ) -> Self {
        Self::try_new(kind, children).expect("client builder supplies the operator's input count")
    }

    /// The operator's payload, without its structural inputs.
    ///
    /// Opaque protobuf payloads retain their measured depth and guard cloning
    /// through [`OpaqueRelation`].
    pub fn kind(&self) -> &LogicalPlanKind {
        &self.kind
    }

    /// Immediate shared inputs in protocol order.
    ///
    /// MlTransform's input also retains its existing lifetime and depth dependency;
    /// its embedded protobuf relation is serialized opaquely. Raw and CachedRelation
    /// have no structural inputs: their protobuf descendants remain opaque.
    pub fn children(&self) -> &[Arc<Self>] {
        &self.children
    }

    /// Take this owned node's payload and shared inputs without cloning them.
    ///
    /// The inputs retain their Arc allocations and opaque payloads retain their
    /// measured clone depth. Reconstructing the node with [`Self::try_new`] checks
    /// only its input count. The caller must preserve the correspondence between
    /// an MlTransform input and its opaque protobuf payload. A new Arc around
    /// that node has a new request identity.
    pub fn into_parts(mut self) -> (LogicalPlanKind, Vec<Arc<Self>>) {
        // A payload-free sentinel leaves nothing for Drop to retain.
        let kind = std::mem::replace(&mut self.kind, LogicalPlanKind::WithRelations);
        let children = std::mem::take(&mut self.children).into_vec();
        (kind, children)
    }

    /// Replace inputs while retaining this node's payload.
    ///
    /// MlTransform embeds its wire input in an opaque protobuf payload, so its
    /// structural input must remain the same Arc allocation. Replacing it would
    /// change ownership and depth without changing the serialized query. For
    /// MlTransform this operation is therefore a validated clone retaining its input.
    ///
    /// The result is a new owned node even if every input is unchanged. Wrapping
    /// it in a new Arc gives it a new stable request ID. Retain the original Arc
    /// when an unchanged node should keep its stable request ID so the server can
    /// reuse its cached resolution of the plan.
    ///
    /// # Errors
    ///
    /// Returns [`PlanRebuildError::InvalidArity`] if the input count is invalid,
    /// or [`PlanRebuildError::OpaqueInputReplacement`] if an MlTransform input
    /// is replaced by a different Arc allocation.
    pub fn with_children(
        &self,
        children: impl IntoIterator<Item = Arc<Self>>,
    ) -> std::result::Result<Self, PlanRebuildError> {
        let children: SmallVec<[Arc<Self>; 2]> = children.into_iter().collect();
        Self::validate_arity(&self.kind, children.len())?;
        if matches!(&self.kind, LogicalPlanKind::MlTransform { .. })
            && !self
                .children
                .iter()
                .zip(&children)
                .all(|(old, new)| Arc::ptr_eq(old, new))
        {
            return Err(PlanRebuildError::OpaqueInputReplacement {
                operator: self.kind.name(),
            });
        }
        Ok(Self {
            kind: self.kind.clone(),
            children,
        })
    }

    /// Visit each distinct node once in first-encounter preorder, without recursing.
    ///
    /// Inputs are considered in their structural order. Shared nodes are identified
    /// by their allocation, so equivalent but distinct nodes are visited separately.
    /// Work is linear in the number of nodes and input edges, with storage for the
    /// visited nodes and pending inputs. Expressions and opaque protobuf descendants
    /// are not visited. Use [`Self::visit_all_paths`] to visit every occurrence.
    pub fn visit(&self, mut visitor: impl FnMut(&Self)) {
        let mut visited = std::collections::HashSet::<*const Self>::new();
        let mut pending = vec![self];
        while let Some(plan) = pending.pop() {
            if visited.insert(plan as *const Self) {
                visitor(plan);
                pending.extend(plan.children.iter().rev().map(|input| input.as_ref()));
            }
        }
    }

    /// Visit every occurrence in the unfolded input tree in preorder, without recursing.
    ///
    /// A shared subtree is revisited for every path to it, so traversal can be
    /// exponential in the number of distinct nodes of a shared DAG. Expressions
    /// and opaque protobuf descendants are not visited. Use [`Self::visit`] for
    /// work linear in the number of distinct nodes and input edges.
    pub fn visit_all_paths(&self, mut visitor: impl FnMut(&Self)) {
        let mut pending = vec![self];
        while let Some(plan) = pending.pop() {
            visitor(plan);
            pending.extend(plan.children.iter().rev().map(|input| input.as_ref()));
        }
    }

    /// Rewrite each distinct node once, after its inputs, without recursing.
    ///
    /// Inputs are processed in structural order. Results are memoized by original
    /// Arc allocation, preserving sharing even when a node has several parents.
    /// If all input allocations are unchanged, `rewrite` receives the original
    /// Arc; otherwise it receives a new node built through [`Self::with_children`].
    /// Returning that Arc unchanged retains its stable request identity.
    ///
    /// The callback sees rewritten inputs and may return any replacement subtree.
    /// Replacement subtrees are not traversed again. Expressions and opaque
    /// protobuf descendants are not rewritten. Traversal work and storage are
    /// linear in the original nodes and edges, excluding payload clones and the
    /// callback's own work.
    ///
    /// # Examples
    ///
    /// Rewrite a shared leaf once while retaining both references to its replacement:
    ///
    /// ```
    /// use std::sync::Arc;
    /// use spark_connect::plan::{self, LogicalPlanKind, PlanRebuildError, SetOpType};
    ///
    /// let input = Arc::new(plan::range(0, 10, 1));
    /// let root = Arc::new(plan::set_operation(
    ///     Arc::clone(&input), Arc::clone(&input), SetOpType::Union, true, false, false,
    /// ));
    /// let mut replaced = 0;
    /// let rewritten = root.try_transform_up::<PlanRebuildError>(|node| {
    ///     if matches!(node.kind(), LogicalPlanKind::Range { .. }) {
    ///         replaced += 1;
    ///         Ok(Arc::new(plan::range(0, 20, 1)))
    ///     } else {
    ///         Ok(Arc::clone(node))
    ///     }
    /// }).expect("the union accepts its rewritten inputs");
    /// assert_eq!(replaced, 1);
    /// assert!(Arc::ptr_eq(&rewritten.children()[0], &rewritten.children()[1]));
    /// assert!(!Arc::ptr_eq(&root, &rewritten));
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates callback errors or a [`PlanRebuildError`] converted into `E`.
    /// If an MlTransform's rewritten direct input is a different Arc allocation,
    /// rebuilding fails before its callback runs, even if that callback would
    /// replace the whole MlTransform. Replacing the whole node in its callback is
    /// possible when its direct input remains unchanged. Its opaque wire input
    /// cannot be updated through this structural rewrite.
    pub fn try_transform_up<E>(
        self: &Arc<Self>,
        mut rewrite: impl FnMut(&Arc<Self>) -> std::result::Result<Arc<Self>, E>,
    ) -> std::result::Result<Arc<Self>, E>
    where
        E: From<PlanRebuildError>,
    {
        let mut completed = HashMap::<*const Self, Arc<Self>>::new();
        let mut pending = vec![(self, false)];
        while let Some((plan, children_done)) = pending.pop() {
            let key = Arc::as_ptr(plan);
            if completed.contains_key(&key) {
                continue;
            }
            if !children_done {
                pending.push((plan, true));
                pending.extend(plan.children.iter().rev().map(|input| (input, false)));
                continue;
            }
            let children: SmallVec<[Arc<Self>; 2]> = plan
                .children
                .iter()
                .map(|input| Arc::clone(&completed[&Arc::as_ptr(input)]))
                .collect();
            let candidate = if plan
                .children
                .iter()
                .zip(&children)
                .all(|(old, new)| Arc::ptr_eq(old, new))
            {
                Arc::clone(plan)
            } else {
                Arc::new(plan.with_children(children)?)
            };
            completed.insert(key, rewrite(&candidate)?);
        }
        Ok(completed
            .remove(&Arc::as_ptr(self))
            .expect("root was processed"))
    }
}

/// Operator-specific payload without structural input plans.
///
/// Binary inputs are left then right, CoGroupMap uses input then other, and
/// GroupMap uses its primary input followed by optional initial state.
/// WithRelations uses its root followed by its references. Opaque protobuf
/// payloads in Raw, CachedRelation and MlTransform keep their existing format.
///
/// Cloning grows the stack for the actual relation depth of `Raw`,
/// `CachedRelation` and `MlTransform`, using their [`OpaqueRelation`] caches.
/// Protobuf expressions, data types and other owned nested payloads retain
/// recursive cloning; [`Expression`] and [`Column`] handles remain shared.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LogicalPlanKind {
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
    Project { columns: Vec<Column> },
    /// A filter: `df.filter(condition)`.
    Filter { condition: Column },
    /// An aggregation: `df.groupBy(...).agg(...)`.
    Aggregate {
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
        join_type: JoinType,
        on: Option<Column>,
        using_columns: Vec<String>,
    },
    /// A lateral join (`LATERAL` correlated subquery join).
    LateralJoin {
        join_type: JoinType,
        on: Option<Column>,
    },
    /// A set operation (union, intersect, except).
    SetOperation {
        set_op_type: SetOpType,
        is_all: bool,
        by_name: bool,
        allow_missing_columns: bool,
    },
    /// Limit: `df.limit(n)`.
    Limit { limit: i32 },
    /// Offset: `df.offset(n)`.
    Offset { offset: i32 },
    /// Tail: `df.tail(n)` - returns last n rows.
    Tail { limit: i32 },
    /// ShowString: the table of `df.show(...)`, which the server formats as one string:
    /// `num_rows` rows, values truncated to `truncate` characters (0 for none), vertical
    /// with a line per value.
    ShowString {
        num_rows: i32,
        truncate: i32,
        vertical: bool,
    },
    /// HtmlString: the HTML table of `df._repr_html_()`, which the server formats as one
    /// string: `num_rows` rows, values truncated to `truncate` characters (0 for none).
    HtmlString { num_rows: i32, truncate: i32 },
    /// A relation built outside the core, as the plans of `pyspark.ml` build theirs
    /// (`pyspark.ml.connect.proto.TransformerRelation`), sent as it is, with the plan ID of the
    /// core, and the owner of the relation, if any, which the plan keeps alive. `depth` is the
    /// number of levels of the relation as a plan, with those of the plans of DataFrames that it
    /// holds (1 for a relation without them), which count for [`MAX_PLAN_DEPTH`]. `observations`
    /// are those of the DataFrames that the relation holds as its child, as PySpark's plans of
    /// `pyspark.ml` collect them (see [`LogicalPlan::observations`]).
    Raw {
        relation: OpaqueRelation,
        owner: Option<RelationOwner>,
        depth: usize,
        observations: Vec<(String, crate::observation::Observation)>,
    },
    /// Deduplicate: `df.distinct()` or `df.dropDuplicates()`.
    Deduplicate {
        all_columns_as_keys: bool,
        column_names: Vec<String>,
        within_watermark: bool,
    },
    /// Sort: `df.sort(cols...)`.
    Sort {
        order: Vec<Expression>,
        is_global: bool,
    },
    /// Sample: `df.sample(fraction)` or `df.sample(num_rows)`.
    Sample {
        lower_bound: f64,
        upper_bound: f64,
        with_replacement: bool,
        seed: Option<i64>,
        /// Whether the input is sorted, or cached, to be in a deterministic order, as the
        /// splits of `randomSplit` are, so that they are disjoint.
        deterministic_order: bool,
    },
    /// Repartition: `df.repartition(num_partitions)`.
    Repartition { num_partitions: i32, shuffle: bool },
    /// RepartitionByExpression: `df.repartitionByRange()` or similar.
    RepartitionByExpression {
        num_partitions: i32,
        expressions: Vec<Expression>,
    },
    /// WithColumns: `df.withColumn(name, col)` or `df.withColumns(...)`.
    WithColumns {
        column_names: Vec<String>,
        columns: Vec<Column>,
    },
    /// WithColumnMetadata: set metadata (a JSON map) on a single existing column.
    /// Mirrors `DataFrame.withMetadata` - a `WithColumns` whose one alias re-aliases
    /// the column to itself, carrying the metadata JSON.
    WithColumnMetadata {
        column_name: String,
        /// The column as a reference to the input's plan ID, as PySpark's `self[columnName]`.
        column: Expression,
        metadata_json: String,
    },
    /// WithColumnsRenamed: `df.withColumnRenamed(old, new)` or `df.withColumnsRenamed(...)`,
    /// of the renames in their order, in which the server applies them, as PySpark sends them.
    WithColumnsRenamed { renames: Vec<(String, String)> },
    /// Drop: `df.drop(column_names...)`.
    Drop {
        columns: Vec<String>,
        /// The columns to drop as expressions, as PySpark's `Drop` of Columns sends them.
        expressions: Vec<Expression>,
    },
    /// ToDF: `df.toDF(*column_names)`.
    ToDF { column_names: Vec<String> },
    /// ToSchema: set schema to match the provided type.
    ToSchema { schema: DataType },
    /// Hint: `df.hint(name, parameters...)`.
    Hint {
        name: String,
        /// The parameters, literal expressions or columns, as PySpark's `lit` of them.
        parameters: Vec<Expression>,
    },
    /// Parse: a DataFrame of one string column parsed into a structured DataFrame, as
    /// PySpark's `Parse` of `spark.read.json`, `csv` and `xml` of a DataFrame.
    Parse {
        format: ParseFormat,
        schema: Option<DataType>,
        options: HashMap<String, String>,
    },
    /// Unpivot: `df.unpivot(...)`.
    Unpivot {
        ids: Vec<Column>,
        values: Option<Vec<Column>>,
        variable_column_name: String,
        value_column_name: String,
    },
    /// NAFill: `df.fillna(value, subset)`.
    NAFill {
        fill_value: crate::row::Value,
        columns: Vec<String>,
    },
    /// NAFill with a per-column value: `df.fillna({col: value, ...})`.
    NAFillColumns {
        cols: Vec<String>,
        values: Vec<crate::row::Value>,
    },
    /// NADrop: `df.dropna(how, thresh, subset)`.
    NADrop {
        how: String,
        min_non_null: Option<i32>,
        columns: Vec<String>,
    },
    /// NAReplace: `df.replace(to_replace, value, subset)`, each replacement a literal and
    /// the literal that replaces it.
    NAReplace {
        replacements: Vec<(LiteralExpression, LiteralExpression)>,
        columns: Vec<String>,
    },
    /// Describe: `df.describe(cols...)`.
    Describe { columns: Vec<String> },
    /// Summary: `df.summary(percentiles...)`.
    Summary { percentiles: Vec<String> },
    /// ColRegex: column selection by regex.
    ColRegex { col_name: String },
    /// SubqueryAlias: `df.as(alias)`.
    SubqueryAlias { alias: String },
    /// WithRelations: a plan with the plans that it refers to, such as a SQL query with the
    /// DataFrames that `SparkSession.sql` takes as keyword arguments, each a `SubqueryAlias`
    /// whose alias the query names.
    WithRelations,
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
    CachedRelation { relation: OpaqueRelation },
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
        time_column: String,
        delay_threshold: String,
    },
    /// RepartitionByRange: `df.repartitionByRange(numPartitions, *cols)`.
    RepartitionByRange {
        num_partitions: Option<i32>,
        partition_exprs: Vec<Expression>,
    },
    /// StatCrosstab: `df.stat.crosstab(col1, col2)`.
    StatCrosstab { col1: String, col2: String },
    /// StatFreqItems: `df.stat.freqItems(columns, support)`.
    StatFreqItems { columns: Vec<String>, support: f64 },
    /// StatApproxQuantile: `df.stat.approxQuantile(columns, probabilities, relativeError)`.
    StatApproxQuantile {
        columns: Vec<String>,
        probabilities: Vec<f64>,
        relative_error: f64,
    },
    /// StatCorr: `df.stat.corr(col1, col2)`.
    StatCorr { col1: String, col2: String },
    /// StatCov: `df.stat.cov(col1, col2)`.
    StatCov { col1: String, col2: String },
    /// StatSampleBy: `df.stat.sampleBy(col, fractions, seed)`.
    StatSampleBy {
        col: Expression,
        fractions: Vec<(Expression, f64)>,
        seed: Option<i64>,
    },
    /// Observe: `df.observe(name, exprs)` - collect metrics.
    Observe {
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
    Zip,
    /// ML Transformation: wraps an MlRelation proto directly. The proto carries the input
    /// inline, so the structural input keeps that plan and its owners alive; `owner` keeps
    /// the cached model that the relation refers to alive, if any.
    MlTransform {
        ml_relation: OpaqueRelation,
        owner: Option<RelationOwner>,
    },
    /// A catalog operation exposed as a relation (e.g. `listTables`, `listColumns`),
    /// evaluated lazily so `.collect()` returns the server's rows - including zero
    /// rows for an empty database, which must not be an error.
    Catalog { catalog: proto::Catalog },
    /// Transpose: swap rows and columns. `index_columns` optionally names the
    /// column(s) used as the transposed header (empty = server default).
    Transpose { index_columns: Vec<Expression> },
    /// MapPartitions: `DataFrame.mapInPandas` / `mapInArrow` (also backs foreach), of the id
    /// of the resource profile of its stage, if any.
    MapPartitions {
        func: CommonInlineUserDefinedFunctionExpression,
        is_barrier: bool,
        profile_id: Option<i32>,
    },
    /// GroupMap: `GroupedData.applyInPandas` / `applyInArrow`, and the stateful
    /// `transformWithState[InPandas]`.
    /// Callers changing its optional initial-state input must keep the stateful
    /// payload consistent; input arity alone does not validate that relationship.
    GroupMap {
        grouping_expressions: Vec<Expression>,
        func: CommonInlineUserDefinedFunctionExpression,
        sorting_expressions: Vec<Expression>,
        // Stateful fields (None or empty for plain applyInPandas/applyInArrow).
        // The optional initial-state plan is the second structural input.
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
        grouping_expressions: Vec<Expression>,
        func: CommonInlineUserDefinedFunctionExpression,
        output_schema: DataType,
        state_schema: DataType,
        output_mode: String,
        timeout_conf: String,
    },
    /// CoGroupMap: `GroupedData.cogroup(...).applyInPandas` / `applyInArrow`.
    CoGroupMap {
        input_grouping_expressions: Vec<Expression>,
        other_grouping_expressions: Vec<Expression>,
        func: CommonInlineUserDefinedFunctionExpression,
    },
    /// AsOfJoin: `DataFrame._joinAsOf`, which pandas-on-Spark's `merge_asof` calls.
    AsOfJoin {
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

impl LogicalPlanKind {
    /// The operator's stable descriptive name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Range { .. } => "Range",
            Self::Sql { .. } => "Sql",
            Self::Project { .. } => "Project",
            Self::Filter { .. } => "Filter",
            Self::Aggregate { .. } => "Aggregate",
            Self::Join { .. } => "Join",
            Self::LateralJoin { .. } => "LateralJoin",
            Self::SetOperation { .. } => "SetOperation",
            Self::Limit { .. } => "Limit",
            Self::Offset { .. } => "Offset",
            Self::Tail { .. } => "Tail",
            Self::ShowString { .. } => "ShowString",
            Self::HtmlString { .. } => "HtmlString",
            Self::Raw { .. } => "Raw",
            Self::Deduplicate { .. } => "Deduplicate",
            Self::Sort { .. } => "Sort",
            Self::Sample { .. } => "Sample",
            Self::Repartition { .. } => "Repartition",
            Self::RepartitionByExpression { .. } => "RepartitionByExpression",
            Self::WithColumns { .. } => "WithColumns",
            Self::WithColumnMetadata { .. } => "WithColumnMetadata",
            Self::WithColumnsRenamed { .. } => "WithColumnsRenamed",
            Self::Drop { .. } => "Drop",
            Self::ToDF { .. } => "ToDF",
            Self::ToSchema { .. } => "ToSchema",
            Self::Hint { .. } => "Hint",
            Self::Parse { .. } => "Parse",
            Self::Unpivot { .. } => "Unpivot",
            Self::NAFill { .. } => "NAFill",
            Self::NAFillColumns { .. } => "NAFillColumns",
            Self::NADrop { .. } => "NADrop",
            Self::NAReplace { .. } => "NAReplace",
            Self::Describe { .. } => "Describe",
            Self::Summary { .. } => "Summary",
            Self::ColRegex { .. } => "ColRegex",
            Self::SubqueryAlias { .. } => "SubqueryAlias",
            Self::WithRelations => "WithRelations",
            Self::LocalRelation { .. } => "LocalRelation",
            Self::ChunkedCachedLocalRelation { .. } => "ChunkedCachedLocalRelation",
            Self::CachedRemoteRelation { .. } => "CachedRemoteRelation",
            Self::CachedRelation { .. } => "CachedRelation",
            Self::Read { .. } => "Read",
            Self::RelationChanges { .. } => "RelationChanges",
            Self::WithWatermark { .. } => "WithWatermark",
            Self::RepartitionByRange { .. } => "RepartitionByRange",
            Self::StatCrosstab { .. } => "StatCrosstab",
            Self::StatFreqItems { .. } => "StatFreqItems",
            Self::StatApproxQuantile { .. } => "StatApproxQuantile",
            Self::StatCorr { .. } => "StatCorr",
            Self::StatCov { .. } => "StatCov",
            Self::StatSampleBy { .. } => "StatSampleBy",
            Self::Observe { .. } => "Observe",
            Self::UnresolvedTableValuedFunction { .. } => "UnresolvedTableValuedFunction",
            Self::Zip => "Zip",
            Self::MlTransform { .. } => "MlTransform",
            Self::Catalog { .. } => "Catalog",
            Self::Transpose { .. } => "Transpose",
            Self::MapPartitions { .. } => "MapPartitions",
            Self::GroupMap { .. } => "GroupMap",
            Self::ApplyInPandasWithState { .. } => "ApplyInPandasWithState",
            Self::CoGroupMap { .. } => "CoGroupMap",
            Self::AsOfJoin { .. } => "AsOfJoin",
            Self::NearestByJoin { .. } => "NearestByJoin",
            Self::CommonInlineUdtf { .. } => "CommonInlineUdtf",
        }
    }

    /// The inclusive accepted range of structural input counts.
    ///
    /// WithRelations accepts a root and any number of references. MlTransform's
    /// one input retains the plan that its opaque relation embeds.
    pub fn arity(&self) -> std::ops::RangeInclusive<usize> {
        match self {
            Self::Range { .. } => 0..=0,
            Self::Sql { .. } => 0..=0,
            Self::Project { .. } => 1..=1,
            Self::Filter { .. } => 1..=1,
            Self::Aggregate { .. } => 1..=1,
            Self::Join { .. } => 2..=2,
            Self::LateralJoin { .. } => 2..=2,
            Self::SetOperation { .. } => 2..=2,
            Self::Limit { .. } => 1..=1,
            Self::Offset { .. } => 1..=1,
            Self::Tail { .. } => 1..=1,
            Self::ShowString { .. } => 1..=1,
            Self::HtmlString { .. } => 1..=1,
            Self::Raw { .. } => 0..=0,
            Self::Deduplicate { .. } => 1..=1,
            Self::Sort { .. } => 1..=1,
            Self::Sample { .. } => 1..=1,
            Self::Repartition { .. } => 1..=1,
            Self::RepartitionByExpression { .. } => 1..=1,
            Self::WithColumns { .. } => 1..=1,
            Self::WithColumnMetadata { .. } => 1..=1,
            Self::WithColumnsRenamed { .. } => 1..=1,
            Self::Drop { .. } => 1..=1,
            Self::ToDF { .. } => 1..=1,
            Self::ToSchema { .. } => 1..=1,
            Self::Hint { .. } => 1..=1,
            Self::Parse { .. } => 1..=1,
            Self::Unpivot { .. } => 1..=1,
            Self::NAFill { .. } => 1..=1,
            Self::NAFillColumns { .. } => 1..=1,
            Self::NADrop { .. } => 1..=1,
            Self::NAReplace { .. } => 1..=1,
            Self::Describe { .. } => 1..=1,
            Self::Summary { .. } => 1..=1,
            Self::ColRegex { .. } => 1..=1,
            Self::SubqueryAlias { .. } => 1..=1,
            Self::WithRelations => 1..=usize::MAX,
            Self::LocalRelation { .. } => 0..=0,
            Self::ChunkedCachedLocalRelation { .. } => 0..=0,
            Self::CachedRemoteRelation { .. } => 0..=0,
            Self::CachedRelation { .. } => 0..=0,
            Self::Read { .. } => 0..=0,
            Self::RelationChanges { .. } => 0..=0,
            Self::WithWatermark { .. } => 1..=1,
            Self::RepartitionByRange { .. } => 1..=1,
            Self::StatCrosstab { .. } => 1..=1,
            Self::StatFreqItems { .. } => 1..=1,
            Self::StatApproxQuantile { .. } => 1..=1,
            Self::StatCorr { .. } => 1..=1,
            Self::StatCov { .. } => 1..=1,
            Self::StatSampleBy { .. } => 1..=1,
            Self::Observe { .. } => 1..=1,
            Self::UnresolvedTableValuedFunction { .. } => 0..=0,
            Self::Zip => 2..=2,
            Self::MlTransform { .. } => 1..=1,
            Self::Catalog { .. } => 0..=0,
            Self::Transpose { .. } => 1..=1,
            Self::MapPartitions { .. } => 1..=1,
            Self::GroupMap { .. } => 1..=2,
            Self::ApplyInPandasWithState { .. } => 1..=1,
            Self::CoGroupMap { .. } => 2..=2,
            Self::AsOfJoin { .. } => 2..=2,
            Self::NearestByJoin { .. } => 2..=2,
            Self::CommonInlineUdtf { .. } => 0..=0,
        }
    }
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
/// names no node of this session's plan.
static NEXT_PLAN_ID: AtomicI64 = AtomicI64::new(1);

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
    fn insert(&mut self, key: usize, node: &Arc<LogicalPlan>, id: i64) {
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
        if nodes.ids.len() >= nodes.prune_at {
            nodes.ids.retain(|_, (node, _, _)| node.strong_count() > 0);
            nodes.prune_at = (nodes.ids.len() * 2).max(Self::MIN_PRUNE_AT);
        }
        let id = NEXT_PLAN_ID.fetch_add(1, Ordering::Relaxed);
        nodes.insert(key, node, id);
        id
    }

    /// Give `node` the plan ID `id`, of a relation that this session sent: so a DataFrame of the
    /// relation, such as one that was pickled, keeps the plan ID of the DataFrame it was of.
    pub(crate) fn assign(&self, node: &Arc<LogicalPlan>, id: i64) {
        NEXT_PLAN_ID.fetch_max(id.saturating_add(1), Ordering::Relaxed);
        let mut nodes = self.nodes.lock().unwrap_or_else(|e| e.into_inner());
        nodes.insert(Arc::as_ptr(node) as usize, node, id);
    }

    /// The live node of the plan ID `id` that was first given it, such as the DataFrame that a
    /// relation that this session sent is of, when it is alive.
    ///
    /// A plan ID is of one node that this session numbered (`id_of`, or `root_id_of` for the
    /// relation within the `WithRelations` that sends it), and of each node that was given it
    /// (`assign`): each DataFrame of a relation that the session sent, such as each unpickled
    /// copy of a DataFrame, or a deep copy of one, is a node of its own with the plan ID of the
    /// DataFrame it was of, which may be alive too. A deep copy has Observations of its own
    /// (`DataFrame::detached_copy`), so the nodes of an ID may differ in them: the first given
    /// the ID is taken, which is the DataFrame that the copies were made of, as long as it is
    /// alive, so that a copy of a relation shares its Observations as PySpark's `copy.copy` does.
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
            .map(|(_, node)| node)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.nodes.lock().unwrap().ids.len()
    }
}

/// The `Any` type of a placeholder: the relation that stands for the relation of a DataFrame, in a
/// relation that is built outside the core, such as a plan of `pyspark.ml`, which the Python
/// client builds over the placeholder of each DataFrame it takes as input. The value of a
/// placeholder is its marker. The core replaces it by the relation of its DataFrame when the
/// relation or the command that holds it reaches the core, so the nesting of a deep DataFrame is
/// never decoded: a protobuf decoder limits the depth of what it decodes.
const PLACEHOLDER_TYPE_URL: &str = "type.googleapis.com/spark.connect.rust.DataFramePlaceholder";

/// The placeholders of all the sessions, by the addresses of their plan node and of their
/// session's plan IDs, each with its marker. Both are held weakly, as `PlanIds` holds its nodes,
/// so an entry goes with its DataFrame, and its address is not reused while the entry exists.
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

/// The placeholder of the relation of `node`, a DataFrame of the session of `ids`: the same
/// marker each time the DataFrame gives it.
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
}

/// The placeholder of the relation of `node`, a DataFrame of the session of `ids`, as the
/// relation that the Python client sends in its place.
pub(crate) fn placeholder_relation(node: &Arc<LogicalPlan>, ids: &Arc<PlanIds>) -> proto::Relation {
    let marker = placeholder_marker(node, ids);
    proto::Relation {
        common: None,
        rel_type: Some(proto::relation::RelType::Extension(prost_types::Any {
            type_url: PLACEHOLDER_TYPE_URL.to_string(),
            value: marker.to_le_bytes().to_vec(),
        })),
    }
}

/// The Observations of the DataFrame whose placeholder `relation`, a relation of `pyspark.ml`,
/// holds as its child: the relation itself, or the input of a transform, the dataset of a model
/// summary or the root of a `WithRelations`, and so on down, as PySpark's `TransformerRelation`,
/// `AttributeRelation` and `WithRelations` collect the `observations` of their child (see
/// [`LogicalPlan::observations`]). Taken before the placeholders are replaced, which then no
/// longer name their DataFrames.
pub(crate) fn child_observations(
    relation: &proto::Relation,
) -> Vec<(String, crate::observation::Observation)> {
    let mut relation = relation;
    loop {
        let child = match &relation.rel_type {
            Some(proto::relation::RelType::Extension(any))
                if any.type_url == PLACEHOLDER_TYPE_URL =>
            {
                let target = <[u8; 8]>::try_from(any.value.as_slice())
                    .ok()
                    .and_then(|marker| placeholder_target(i64::from_le_bytes(marker)));
                return target
                    .map(|(node, _)| node.observations())
                    .unwrap_or_default();
            }
            Some(proto::relation::RelType::MlRelation(ml)) => match &ml.ml_type {
                Some(proto::ml_relation::MlType::Transform(transform)) => {
                    transform.input.as_deref()
                }
                _ => ml.model_summary_dataset.as_deref(),
            },
            Some(proto::relation::RelType::WithRelations(with)) => with.root.as_deref(),
            _ => None,
        };
        match child {
            Some(child) => relation = child,
            None => return Vec::new(),
        }
    }
}

/// Replace the placeholders of `relation` by the relations of their DataFrames, and of the
/// relations that the plans of `pyspark.ml` take as input (their `ml_relation`, `with_relations`
/// and fetched arguments). Each replacement is the relation of a DataFrame of the core, which has
/// no placeholder.
pub(crate) fn resolve_relation(relation: &mut proto::Relation) -> Result<()> {
    resolve_relation_at(relation, 1).map(|_| ())
}

/// As [`resolve_relation`], and the levels of the relation as a plan of them, as PySpark's
/// serializer counts the levels of a plan of `pyspark.ml`, which serializes the plans of its
/// inputs within its own: one for the relation, one more for each relation that holds an input,
/// and those of the plan of the DataFrame of each placeholder under them.
pub(crate) fn resolve_relation_levels(relation: &mut proto::Relation) -> Result<usize> {
    resolve_relation_at(relation, 1)
}

/// Resolve `relation`, which is at the level `level` of the relation being resolved, and the
/// deepest level of its resolved relation.
fn resolve_relation_at(relation: &mut proto::Relation, level: usize) -> Result<usize> {
    if let Some(proto::relation::RelType::Extension(any)) = &relation.rel_type {
        if any.type_url == PLACEHOLDER_TYPE_URL {
            let marker = i64::from_le_bytes(
                any.value
                    .as_slice()
                    .try_into()
                    .map_err(|_| SparkError::value_msg("a placeholder of no marker"))?,
            );
            let (node, ids) = placeholder_target(marker).ok_or_else(|| {
                SparkError::value_msg("a placeholder of a DataFrame that no longer exists")
            })?;
            let (resolved, levels) = relation_with_ids_and_levels(&node, &ids)?;
            *relation = resolved;
            // The relation of the DataFrame takes the place of the placeholder, at its level.
            return Ok(level - 1 + levels);
        }
    }
    let mut deepest = level;
    let mut below = |relation: &mut proto::Relation| -> Result<()> {
        deepest = deepest.max(resolve_relation_at(relation, level + 1)?);
        Ok(())
    };
    match relation.rel_type.as_mut() {
        Some(proto::relation::RelType::MlRelation(ml)) => {
            if let Some(summary) = ml.model_summary_dataset.as_mut() {
                below(summary)?;
            }
            match ml.ml_type.as_mut() {
                Some(proto::ml_relation::MlType::Transform(transform)) => {
                    if let Some(input) = transform.input.as_mut() {
                        below(input)?;
                    }
                }
                Some(proto::ml_relation::MlType::Fetch(fetch)) => {
                    deepest = deepest.max(resolve_fetch_at(fetch, level + 1)?);
                }
                None => {}
            }
        }
        Some(proto::relation::RelType::WithRelations(with)) => {
            if let Some(root) = with.root.as_mut() {
                below(root)?;
            }
            for reference in with.references.iter_mut() {
                below(reference)?;
            }
        }
        _ => {}
    }
    Ok(deepest)
}

/// Replace the placeholders of the relations of the ML command `command`, as `resolve_relation`.
pub(crate) fn resolve_command(command: &mut proto::Command) -> Result<()> {
    let Some(proto::command::CommandType::MlCommand(ml)) = command.command_type.as_mut() else {
        return Ok(());
    };
    let dataset = match ml.command.as_mut() {
        Some(proto::ml_command::Command::Fit(fit)) => fit.dataset.as_mut(),
        Some(proto::ml_command::Command::Evaluate(evaluate)) => evaluate.dataset.as_mut(),
        Some(proto::ml_command::Command::CreateSummary(summary)) => summary.dataset.as_mut(),
        Some(proto::ml_command::Command::Fetch(fetch)) => {
            return resolve_fetch_at(fetch, 1).map(|_| ())
        }
        _ => None,
    };
    match dataset {
        Some(dataset) => resolve_relation(dataset),
        None => Ok(()),
    }
}

/// Replace the placeholders of the relations that a fetch of `fetch` takes as arguments, each at
/// the level `level`, and the deepest level of them.
fn resolve_fetch_at(fetch: &mut proto::Fetch, level: usize) -> Result<usize> {
    let mut deepest = 0;
    for method in fetch.methods.iter_mut() {
        for args in method.args.iter_mut() {
            if let Some(proto::fetch::method::args::ArgsType::Input(input)) =
                args.args_type.as_mut()
            {
                deepest = deepest.max(resolve_relation_at(input, level)?);
            }
        }
    }
    Ok(deepest)
}

/// Replace the placeholders of the relation or of the command of `request`, a request that a
/// session hook built in Python, as `DataFrame::from_relation` does. Not a stable API.
#[doc(hidden)]
pub fn resolve_request(request: &mut proto::ExecutePlanRequest) -> Result<()> {
    match request.plan.as_mut().and_then(|plan| plan.op_type.as_mut()) {
        Some(proto::plan::OpType::Root(relation)) => resolve_relation(relation),
        Some(proto::plan::OpType::Command(command)) => resolve_command(command),
        _ => Ok(()),
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

/// Stack per opaque relation level, allowing for the additional clone frames in
/// ML fetch-input chains. Expression and data-type nesting are separate limits.
/// ML wrapper frames get twice the former Raw-only guard's per-level budgets:
/// 2 KiB in release and 8 KiB in debug. Tests do not prove a portable bound.
const OPAQUE_CLONE_STACK_PER_LEVEL: usize = if cfg!(debug_assertions) {
    16 << 10
} else {
    4 << 10
};

/// The depth of actual opaque wire relations, including ML datasets and fetched arguments.
/// This iterative sizing pass does not trust Raw's declared depth or an ML lifetime input.
fn opaque_relation_depth(relation: &proto::Relation) -> usize {
    let mut pending = vec![(relation, 1_usize)];
    let mut deepest = 1;
    while let Some((relation, level)) = pending.pop() {
        deepest = deepest.max(level);
        let below = level.saturating_add(1);
        let Some(kind) = &relation.rel_type else {
            continue;
        };
        use proto::relation::RelType;
        match kind {
            RelType::Read(_) => {}
            RelType::Project(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Filter(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Join(value) => {
                if let Some(input) = &value.left {
                    pending.push((input, below));
                }
                if let Some(input) = &value.right {
                    pending.push((input, below));
                }
            }
            RelType::SetOp(value) => {
                if let Some(input) = &value.left_input {
                    pending.push((input, below));
                }
                if let Some(input) = &value.right_input {
                    pending.push((input, below));
                }
            }
            RelType::Sort(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Limit(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Aggregate(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Sql(_) => {}
            RelType::LocalRelation(_) => {}
            RelType::Sample(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Offset(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Deduplicate(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Range(_) => {}
            RelType::SubqueryAlias(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Repartition(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::ToDf(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::WithColumnsRenamed(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::ShowString(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Drop(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Tail(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::WithColumns(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Hint(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Unpivot(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::ToSchema(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::RepartitionByExpression(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::MapPartitions(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::CollectMetrics(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Parse(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::GroupMap(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
                if let Some(input) = &value.initial_input {
                    pending.push((input, below));
                }
            }
            RelType::CoGroupMap(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
                if let Some(input) = &value.other {
                    pending.push((input, below));
                }
            }
            RelType::WithWatermark(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::ApplyInPandasWithState(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::HtmlString(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::CachedLocalRelation(_) => {}
            RelType::CachedRemoteRelation(_) => {}
            RelType::CommonInlineUserDefinedTableFunction(_) => {}
            RelType::AsOfJoin(value) => {
                if let Some(input) = &value.left {
                    pending.push((input, below));
                }
                if let Some(input) = &value.right {
                    pending.push((input, below));
                }
            }
            RelType::CommonInlineUserDefinedDataSource(_) => {}
            RelType::WithRelations(value) => {
                if let Some(input) = &value.root {
                    pending.push((input, below));
                }
                pending.extend(value.references.iter().map(|input| (input, below)));
            }
            RelType::Transpose(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::UnresolvedTableValuedFunction(_) => {}
            RelType::LateralJoin(value) => {
                if let Some(input) = &value.left {
                    pending.push((input, below));
                }
                if let Some(input) = &value.right {
                    pending.push((input, below));
                }
            }
            RelType::ChunkedCachedLocalRelation(_) => {}
            RelType::RelationChanges(_) => {}
            RelType::NearestByJoin(value) => {
                if let Some(input) = &value.left {
                    pending.push((input, below));
                }
                if let Some(input) = &value.right {
                    pending.push((input, below));
                }
            }
            RelType::Zip(value) => {
                if let Some(input) = &value.left {
                    pending.push((input, below));
                }
                if let Some(input) = &value.right {
                    pending.push((input, below));
                }
            }
            RelType::FillNa(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::DropNa(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Replace(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Summary(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Crosstab(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Describe(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Cov(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Corr(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::ApproxQuantile(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::FreqItems(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::SampleBy(value) => {
                if let Some(input) = &value.input {
                    pending.push((input, below));
                }
            }
            RelType::Catalog(_) => {}
            RelType::Unknown(_) => {}
            RelType::Extension(_) => {}
            RelType::MlRelation(ml) => {
                if let Some(input) = &ml.model_summary_dataset {
                    pending.push((input, below));
                }
                match &ml.ml_type {
                    Some(proto::ml_relation::MlType::Transform(transform)) => {
                        if let Some(input) = &transform.input {
                            pending.push((input, below));
                        }
                    }
                    Some(proto::ml_relation::MlType::Fetch(fetch)) => {
                        for method in &fetch.methods {
                            for argument in &method.args {
                                if let Some(proto::fetch::method::args::ArgsType::Input(input)) =
                                    &argument.args_type
                                {
                                    pending.push((input, below));
                                }
                            }
                        }
                    }
                    None => {}
                }
            }
        }
    }
    deepest
}

/// Clone an opaque protobuf using the depth measured by its OpaqueRelation at
/// construction. Wrapper, kind and node clones all reuse that cached depth. The
/// caller's structural inputs need not describe the opaque tree. Expression and
/// data-type nesting retain their existing clone limits.
fn clone_opaque_relation(relation: &proto::Relation, levels: usize) -> proto::Relation {
    debug_assert!(levels > 0, "opaque relation depth must be positive");
    let stack = levels
        .saturating_mul(OPAQUE_CLONE_STACK_PER_LEVEL)
        .saturating_add(64 << 10);
    stacker::maybe_grow(stack, stack, || relation.clone())
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
    // its subqueries are counted as they are serialized.
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

impl Drop for LogicalPlan {
    /// Release uniquely owned input subtrees without recursive node destruction.
    fn drop(&mut self) {
        let mut pending = std::mem::take(&mut self.children);
        while let Some(input) = pending.pop() {
            if let Some(mut plan) = Arc::into_inner(input) {
                pending.extend(plan.children.drain(..));
            }
        }
    }
}

impl LogicalPlan {
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
            match plan.kind() {
                LogicalPlanKind::Observe {
                    name,
                    observation: Some(observation),
                    ..
                } => keep(name, observation),
                LogicalPlanKind::Raw {
                    observations: held, ..
                } => {
                    for (name, observation) in held {
                        keep(name, observation);
                    }
                }
                _ => {}
            }
            match plan.kind() {
                // These plans collect only their primary child's observations.
                LogicalPlanKind::CoGroupMap { .. }
                | LogicalPlanKind::GroupMap { .. }
                | LogicalPlanKind::WithRelations => pending.push(plan.children()[0].as_ref()),
                // All other kinds walk every input in order. Binary inputs are
                // pushed left then right, so the right side is popped first
                // and wins for a duplicate name.
                _ => pending.extend(plan.children().iter().map(|input| input.as_ref())),
            }
        }
        observations
    }

    /// The levels of this node itself, under which its inputs are: one, or those of the relation
    /// of a [`LogicalPlanKind::Raw`], which holds the plans of its inputs.
    /// Raw follows PySpark's logical serializer levels. Automatically generated
    /// WithRelations wrappers add physical Relation nesting without adding a logical
    /// plan level, so taking the maximum with the opaque relation's measured depth
    /// would count those wrappers as logical levels. That depth sizes clones only.
    /// Custom understated declarations retain the existing request-bound limitation.
    fn own_levels(&self) -> usize {
        match &self.kind {
            LogicalPlanKind::Raw { depth, .. } => (*depth).max(1),
            _ => 1,
        }
    }

    /// The number of levels of this plan: one for a leaf (the levels of its relation for a
    /// [`LogicalPlanKind::Raw`]), and one more than its deepest input otherwise. Computed without
    /// recursing, and once for each node that several plans share.
    pub fn depth(&self) -> usize {
        let mut depths: HashMap<*const LogicalPlan, usize> = HashMap::new();
        let mut pending: Vec<(&LogicalPlan, bool)> = vec![(self, false)];
        while let Some((plan, inputs_done)) = pending.pop() {
            let key: *const LogicalPlan = plan;
            if depths.contains_key(&key) {
                continue;
            }
            let inputs = plan.children();
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
                        .iter()
                        .filter(|input| !depths.contains_key(&Arc::as_ptr(input)))
                        .map(|input| (input.as_ref(), false)),
                );
            }
        }
        depths[&(self as *const LogicalPlan)]
    }

    /// Convert this logical plan to a `spark.connect.Relation` protobuf without assigning
    /// session plan IDs. Existing opaque relation metadata is preserved according
    /// to the operator's serialization behavior.
    ///
    /// Any depth is serialized, on a stack that grows for its levels; but prost encodes and drops
    /// a protobuf by recursing, so the proto of a plan of many thousands of levels can overflow a
    /// small stack there (a request is bounded, see [`LogicalPlan::try_to_proto`]).
    pub fn to_proto(&self) -> proto::Relation {
        relation_with_references(self, None, None)
    }

    /// Convert this logical plan without assigning session plan IDs, bounded as
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

        match &self.kind {
            LogicalPlanKind::Range {
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

            LogicalPlanKind::Sql {
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

            LogicalPlanKind::Project { columns } => {
                let input = &self.children[0];
                let mut project = proto::Project::default();
                project.input = Some(Box::new(relation_of(input, plan_ids)));
                for col in columns {
                    project.expressions.push(col.to_proto());
                }
                relation.rel_type = Some(proto::relation::RelType::Project(Box::new(project)));
            }

            LogicalPlanKind::Filter { condition } => {
                let input = &self.children[0];
                let mut filter = proto::Filter::default();
                filter.input = Some(Box::new(relation_of(input, plan_ids)));
                filter.condition = Some(condition.to_proto());
                relation.rel_type = Some(proto::relation::RelType::Filter(Box::new(filter)));
            }

            LogicalPlanKind::Aggregate {
                group_type,
                grouping_expressions,
                aggregate_expressions,
                pivot_col,
                pivot_values,
                grouping_sets,
            } => {
                let input = &self.children[0];
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

            LogicalPlanKind::Join {
                join_type,
                on,
                using_columns,
            } => {
                let left = &self.children[0];
                let right = &self.children[1];
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

            LogicalPlanKind::AsOfJoin {
                left_as_of,
                right_as_of,
                on,
                using_columns,
                join_type,
                tolerance,
                allow_exact_matches,
                direction,
            } => {
                let left = &self.children[0];
                let right = &self.children[1];
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

            LogicalPlanKind::LateralJoin { join_type, on } => {
                let left = &self.children[0];
                let right = &self.children[1];
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

            LogicalPlanKind::SetOperation {
                set_op_type,
                is_all,
                by_name,
                allow_missing_columns,
            } => {
                let left = &self.children[0];
                let right = &self.children[1];
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

            LogicalPlanKind::Limit { limit } => {
                let input = &self.children[0];
                let mut lim = proto::Limit::default();
                lim.input = Some(Box::new(relation_of(input, plan_ids)));
                lim.limit = *limit;
                relation.rel_type = Some(proto::relation::RelType::Limit(Box::new(lim)));
            }

            LogicalPlanKind::Offset { offset } => {
                let input = &self.children[0];
                let mut off = proto::Offset::default();
                off.input = Some(Box::new(relation_of(input, plan_ids)));
                off.offset = *offset;
                relation.rel_type = Some(proto::relation::RelType::Offset(Box::new(off)));
            }

            LogicalPlanKind::Tail { limit } => {
                let input = &self.children[0];
                let mut tail = proto::Tail::default();
                tail.input = Some(Box::new(relation_of(input, plan_ids)));
                tail.limit = *limit;
                relation.rel_type = Some(proto::relation::RelType::Tail(Box::new(tail)));
            }

            LogicalPlanKind::ShowString {
                num_rows,
                truncate,
                vertical,
            } => {
                let input = &self.children[0];
                let show = proto::ShowString {
                    input: Some(Box::new(relation_of(input, plan_ids))),
                    num_rows: *num_rows,
                    truncate: *truncate,
                    vertical: *vertical,
                };
                relation.rel_type = Some(proto::relation::RelType::ShowString(Box::new(show)));
            }

            LogicalPlanKind::HtmlString { num_rows, truncate } => {
                let input = &self.children[0];
                let html = proto::HtmlString {
                    input: Some(Box::new(relation_of(input, plan_ids))),
                    num_rows: *num_rows,
                    truncate: *truncate,
                };
                relation.rel_type = Some(proto::relation::RelType::HtmlString(Box::new(html)));
            }

            LogicalPlanKind::Raw { relation: raw, .. } => {
                relation = raw.clone_proto();
                relation.common.get_or_insert_with(Default::default);
            }

            LogicalPlanKind::Deduplicate {
                all_columns_as_keys,
                column_names,
                within_watermark,
            } => {
                let input = &self.children[0];
                let mut dedup = proto::Deduplicate::default();
                dedup.input = Some(Box::new(relation_of(input, plan_ids)));
                dedup.all_columns_as_keys = Some(*all_columns_as_keys);
                dedup.column_names.extend(column_names.clone());
                dedup.within_watermark = Some(*within_watermark);
                relation.rel_type = Some(proto::relation::RelType::Deduplicate(Box::new(dedup)));
            }

            LogicalPlanKind::Sort { order, is_global } => {
                let input = &self.children[0];
                // Use the generated proto enum constants rather than bare 1/2 magic
                // numbers, so a future proto reorder can't silently change meaning.
                use proto::expression::sort_order::{NullOrdering as PbNulls, SortDirection};
                let mut sort = proto::Sort::default();
                sort.input = Some(Box::new(relation_of(input, plan_ids)));
                for expr in order {
                    // Extract SortOrder from Expression if it's SortOrder type
                    if let ExpressionKind::SortOrder {
                        ascending,
                        null_ordering,
                    } = expr.node().kind()
                    {
                        let mut sort_order = proto::expression::SortOrder::default();
                        sort_order.child = Some(Box::new(expr.node().children()[0].to_proto()));
                        sort_order.direction = if *ascending {
                            SortDirection::Ascending as i32
                        } else {
                            SortDirection::Descending as i32
                        };
                        sort_order.null_ordering = match null_ordering {
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

            LogicalPlanKind::Sample {
                lower_bound,
                upper_bound,
                with_replacement,
                seed,
                deterministic_order,
            } => {
                let input = &self.children[0];
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

            LogicalPlanKind::Repartition {
                num_partitions,
                shuffle,
            } => {
                let input = &self.children[0];
                let mut repart = proto::Repartition::default();
                repart.input = Some(Box::new(relation_of(input, plan_ids)));
                repart.num_partitions = *num_partitions;
                repart.shuffle = Some(*shuffle);
                relation.rel_type = Some(proto::relation::RelType::Repartition(Box::new(repart)));
            }

            LogicalPlanKind::RepartitionByExpression {
                num_partitions,
                expressions,
            } => {
                let input = &self.children[0];
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

            LogicalPlanKind::WithColumns {
                column_names,
                columns,
            } => {
                let input = &self.children[0];
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

            LogicalPlanKind::WithColumnMetadata {
                column_name,
                column,
                metadata_json,
            } => {
                let input = &self.children[0];
                let mut wc = proto::WithColumns::default();
                wc.input = Some(Box::new(relation_of(input, plan_ids)));
                wc.aliases.push(proto::expression::Alias {
                    expr: Some(Box::new(column.to_proto())),
                    name: vec![column_name.clone()],
                    metadata: Some(metadata_json.clone()),
                });
                relation.rel_type = Some(proto::relation::RelType::WithColumns(Box::new(wc)));
            }

            LogicalPlanKind::WithColumnsRenamed { renames } => {
                let input = &self.children[0];
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

            LogicalPlanKind::Drop {
                columns,
                expressions,
            } => {
                let input = &self.children[0];
                let mut drop = proto::Drop::default();
                drop.input = Some(Box::new(relation_of(input, plan_ids)));
                drop.columns
                    .extend(expressions.iter().map(|e| e.to_proto()));
                drop.column_names.extend(columns.clone());
                relation.rel_type = Some(proto::relation::RelType::Drop(Box::new(drop)));
            }

            LogicalPlanKind::ToDF { column_names } => {
                let input = &self.children[0];
                let mut to_df = proto::ToDf::default();
                to_df.input = Some(Box::new(relation_of(input, plan_ids)));
                to_df.column_names.extend(column_names.clone());
                relation.rel_type = Some(proto::relation::RelType::ToDf(Box::new(to_df)));
            }

            LogicalPlanKind::ToSchema { schema } => {
                let input = &self.children[0];
                let mut to_schema = proto::ToSchema::default();
                to_schema.input = Some(Box::new(relation_of(input, plan_ids)));
                to_schema.schema = Some(schema.to_proto());
                relation.rel_type = Some(proto::relation::RelType::ToSchema(Box::new(to_schema)));
            }

            LogicalPlanKind::Hint { name, parameters } => {
                let input = &self.children[0];
                let mut hint = proto::Hint::default();
                hint.input = Some(Box::new(relation_of(input, plan_ids)));
                hint.name = name.clone();
                hint.parameters
                    .extend(parameters.iter().map(|p| p.to_proto()));
                relation.rel_type = Some(proto::relation::RelType::Hint(Box::new(hint)));
            }

            LogicalPlanKind::Parse {
                format,
                schema,
                options,
            } => {
                let input = &self.children[0];
                relation.rel_type = Some(proto::relation::RelType::Parse(Box::new(proto::Parse {
                    input: Some(Box::new(relation_of(input, plan_ids))),
                    format: format.to_proto() as i32,
                    schema: schema.as_ref().map(|s| s.to_proto()),
                    options: options.clone(),
                })));
            }

            LogicalPlanKind::Unpivot {
                ids,
                values,
                variable_column_name,
                value_column_name,
            } => {
                let input = &self.children[0];
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

            LogicalPlanKind::NAFill {
                fill_value,
                columns,
            } => {
                let input = &self.children[0];
                let mut na_fill = proto::NaFill::default();
                na_fill.input = Some(Box::new(relation_of(input, plan_ids)));
                na_fill.cols.extend(columns.clone());
                na_fill.values.push(value_to_proto_literal(fill_value));
                relation.rel_type = Some(proto::relation::RelType::FillNa(Box::new(na_fill)));
            }

            LogicalPlanKind::NAFillColumns { cols, values } => {
                let input = &self.children[0];
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

            LogicalPlanKind::NADrop {
                how,
                min_non_null,
                columns,
            } => {
                let input = &self.children[0];
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

            LogicalPlanKind::NAReplace {
                replacements,
                columns,
            } => {
                let input = &self.children[0];
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

            LogicalPlanKind::Describe { columns } => {
                let input = &self.children[0];
                let mut describe = proto::StatDescribe::default();
                describe.input = Some(Box::new(relation_of(input, plan_ids)));
                describe.cols.extend(columns.clone());
                relation.rel_type = Some(proto::relation::RelType::Describe(Box::new(describe)));
            }

            LogicalPlanKind::Summary { percentiles } => {
                let input = &self.children[0];
                let mut summary = proto::StatSummary::default();
                summary.input = Some(Box::new(relation_of(input, plan_ids)));
                summary.statistics.extend(percentiles.clone());
                relation.rel_type = Some(proto::relation::RelType::Summary(Box::new(summary)));
            }

            LogicalPlanKind::ColRegex { col_name } => {
                let input = &self.children[0];
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

            LogicalPlanKind::SubqueryAlias { alias } => {
                let input = &self.children[0];
                let mut sq_alias = proto::SubqueryAlias::default();
                sq_alias.input = Some(Box::new(relation_of(input, plan_ids)));
                sq_alias.alias = alias.clone();
                relation.rel_type =
                    Some(proto::relation::RelType::SubqueryAlias(Box::new(sq_alias)));
            }

            LogicalPlanKind::WithRelations => {
                let root = &self.children[0];
                let references = &self.children[1..];
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

            LogicalPlanKind::ChunkedCachedLocalRelation {
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
            LogicalPlanKind::LocalRelation { schema, data } => {
                let mut local = proto::LocalRelation::default();
                if let Some(d) = data {
                    local.data = Some(d.clone().into());
                }
                // Explicit schemas apply server-side casts. Inferred Arrow input
                // omits this to avoid a redundant projection on every query.
                local.schema = schema.as_ref().map(DataType::json);
                relation.rel_type = Some(proto::relation::RelType::LocalRelation(local));
            }

            LogicalPlanKind::CachedRemoteRelation { relation_id, .. } => {
                let mut cached = proto::CachedRemoteRelation::default();
                cached.relation_id = relation_id.clone();
                relation.rel_type = Some(proto::relation::RelType::CachedRemoteRelation(cached));
            }

            LogicalPlanKind::CachedRelation { relation: returned } => {
                // Keep the relation as the server returned it; only its plan ID is this node's.
                let common = relation.common.take();
                relation = returned.clone_proto();
                relation.common = common;
            }

            LogicalPlanKind::Read {
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

            LogicalPlanKind::RelationChanges {
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

            LogicalPlanKind::WithWatermark {
                time_column,
                delay_threshold,
            } => {
                let input = &self.children[0];
                let mut watermark = proto::WithWatermark::default();
                watermark.input = Some(Box::new(relation_of(input, plan_ids)));
                watermark.event_time = time_column.clone();
                watermark.delay_threshold = delay_threshold.clone();
                relation.rel_type =
                    Some(proto::relation::RelType::WithWatermark(Box::new(watermark)));
            }

            LogicalPlanKind::RepartitionByRange {
                num_partitions,
                partition_exprs,
            } => {
                let input = &self.children[0];
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

            LogicalPlanKind::StatCrosstab { col1, col2 } => {
                let input = &self.children[0];
                let mut stat = proto::StatCrosstab::default();
                stat.input = Some(Box::new(relation_of(input, plan_ids)));
                stat.col1 = col1.clone();
                stat.col2 = col2.clone();
                relation.rel_type = Some(proto::relation::RelType::Crosstab(Box::new(stat)));
            }

            LogicalPlanKind::StatFreqItems { columns, support } => {
                let input = &self.children[0];
                let mut stat = proto::StatFreqItems::default();
                stat.input = Some(Box::new(relation_of(input, plan_ids)));
                stat.cols.extend(columns.clone());
                stat.support = Some(*support);
                relation.rel_type = Some(proto::relation::RelType::FreqItems(Box::new(stat)));
            }

            LogicalPlanKind::StatApproxQuantile {
                columns,
                probabilities,
                relative_error,
            } => {
                let input = &self.children[0];
                let mut stat = proto::StatApproxQuantile::default();
                stat.input = Some(Box::new(relation_of(input, plan_ids)));
                stat.cols.extend(columns.clone());
                stat.probabilities.extend(probabilities.clone());
                stat.relative_error = *relative_error;
                relation.rel_type = Some(proto::relation::RelType::ApproxQuantile(Box::new(stat)));
            }

            LogicalPlanKind::StatCorr { col1, col2 } => {
                let input = &self.children[0];
                let mut stat = proto::StatCorr::default();
                stat.input = Some(Box::new(relation_of(input, plan_ids)));
                stat.col1 = col1.clone();
                stat.col2 = col2.clone();
                relation.rel_type = Some(proto::relation::RelType::Corr(Box::new(stat)));
            }

            LogicalPlanKind::StatCov { col1, col2 } => {
                let input = &self.children[0];
                let mut stat = proto::StatCov::default();
                stat.input = Some(Box::new(relation_of(input, plan_ids)));
                stat.col1 = col1.clone();
                stat.col2 = col2.clone();
                relation.rel_type = Some(proto::relation::RelType::Cov(Box::new(stat)));
            }

            LogicalPlanKind::StatSampleBy {
                col,
                fractions,
                seed,
            } => {
                let input = &self.children[0];
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

            LogicalPlanKind::Observe { name, exprs, .. } => {
                let input = &self.children[0];
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

            LogicalPlanKind::UnresolvedTableValuedFunction { name, arguments } => {
                let mut tvf = proto::UnresolvedTableValuedFunction::default();
                tvf.function_name = name.clone();
                for arg in arguments {
                    tvf.arguments.push(arg.to_proto());
                }
                relation.rel_type =
                    Some(proto::relation::RelType::UnresolvedTableValuedFunction(tvf));
            }

            LogicalPlanKind::Zip => {
                let left = &self.children[0];
                let right = &self.children[1];
                let mut zip = proto::Zip::default();
                zip.left = Some(Box::new(relation_of(left, plan_ids)));
                zip.right = Some(Box::new(relation_of(right, plan_ids)));
                relation.rel_type = Some(proto::relation::RelType::Zip(Box::new(zip)));
            }

            LogicalPlanKind::MlTransform { ml_relation, .. } => {
                return ml_relation.clone_proto();
            }
            LogicalPlanKind::Catalog { catalog } => {
                relation.rel_type = Some(proto::relation::RelType::Catalog(catalog.clone()));
                return relation;
            }
            LogicalPlanKind::Transpose { index_columns } => {
                let input = &self.children[0];
                let mut transpose = proto::Transpose::default();
                transpose.input = Some(Box::new(relation_of(input, plan_ids)));
                transpose.index_columns = index_columns.iter().map(|e| e.to_proto()).collect();
                relation.rel_type = Some(proto::relation::RelType::Transpose(Box::new(transpose)));
                return relation;
            }
            LogicalPlanKind::MapPartitions {
                func,
                is_barrier,
                profile_id,
            } => {
                let input = &self.children[0];
                let mut mp = proto::MapPartitions::default();
                mp.input = Some(Box::new(relation_of(input, plan_ids)));
                mp.func = Some(func.to_proto());
                mp.is_barrier = Some(*is_barrier);
                mp.profile_id = *profile_id;
                relation.rel_type = Some(proto::relation::RelType::MapPartitions(Box::new(mp)));
            }
            LogicalPlanKind::GroupMap {
                grouping_expressions,
                func,
                sorting_expressions,
                initial_grouping_expressions,
                is_map_groups_with_state,
                output_mode,
                timeout_conf,
                state_schema,
                transform_with_state_info,
            } => {
                let input = &self.children[0];
                let initial_input = self.children.get(1);
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
            LogicalPlanKind::ApplyInPandasWithState {
                grouping_expressions,
                func,
                output_schema,
                state_schema,
                output_mode,
                timeout_conf,
            } => {
                let input = &self.children[0];
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
            LogicalPlanKind::CoGroupMap {
                input_grouping_expressions,
                other_grouping_expressions,
                func,
            } => {
                let input = &self.children[0];
                let other = &self.children[1];
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
            LogicalPlanKind::NearestByJoin {
                ranking_expression,
                num_results,
                join_type,
                mode,
                direction,
            } => {
                let left = &self.children[0];
                let right = &self.children[1];
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
            LogicalPlanKind::CommonInlineUdtf {
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
    LogicalPlan::from_builder(
        LogicalPlanKind::Range {
            start,
            end,
            step,
            num_partitions: None,
        },
        [],
    )
}

/// Create a Range plan with num_partitions.
pub fn range_with_partitions(start: i64, end: i64, step: i64, num_partitions: i32) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::Range {
            start,
            end,
            step,
            num_partitions: Some(num_partitions),
        },
        [],
    )
}

/// Create a SQL plan.
pub fn sql(query: impl Into<String>) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::Sql {
            query: query.into(),
            pos_args: Vec::new(),
            named_args: HashMap::new(),
        },
        [],
    )
}

/// Create a Project plan.
pub fn project<C: Into<Column>>(
    input: impl Into<Arc<LogicalPlan>>,
    columns: impl IntoIterator<Item = C>,
) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::Project {
            columns: columns.into_iter().map(Into::into).collect(),
        },
        [input.into()],
    )
}

/// Create a Filter plan.
pub fn filter(input: impl Into<Arc<LogicalPlan>>, condition: Column) -> LogicalPlan {
    LogicalPlan::from_builder(LogicalPlanKind::Filter { condition }, [input.into()])
}

/// Create an Aggregate plan.
pub fn aggregate(
    input: impl Into<Arc<LogicalPlan>>,
    group_type: AggregateGroupType,
    grouping_expressions: Vec<Expression>,
    aggregate_expressions: Vec<Expression>,
) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::Aggregate {
            group_type,
            grouping_expressions,
            aggregate_expressions,
            pivot_col: None,
            pivot_values: vec![],
            grouping_sets: vec![],
        },
        [input.into()],
    )
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
    LogicalPlan::from_builder(
        LogicalPlanKind::Aggregate {
            group_type,
            grouping_expressions,
            aggregate_expressions,
            pivot_col: Some(pivot_col),
            pivot_values,
            grouping_sets: vec![],
        },
        [input.into()],
    )
}

/// Create an Aggregate plan with explicit grouping sets.
pub fn aggregate_with_grouping_sets(
    input: impl Into<Arc<LogicalPlan>>,
    grouping_expressions: Vec<Expression>,
    aggregate_expressions: Vec<Expression>,
    grouping_sets: Vec<Vec<Expression>>,
) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::Aggregate {
            group_type: AggregateGroupType::GroupingSets,
            grouping_expressions,
            aggregate_expressions,
            pivot_col: None,
            pivot_values: vec![],
            grouping_sets,
        },
        [input.into()],
    )
}

/// Create a Join plan.
pub fn join(
    left: impl Into<Arc<LogicalPlan>>,
    right: impl Into<Arc<LogicalPlan>>,
    join_type: JoinType,
    on: Option<Column>,
    using_columns: Vec<String>,
) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::Join {
            join_type,
            on,
            using_columns,
        },
        [left.into(), right.into()],
    )
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
    LogicalPlan::from_builder(
        LogicalPlanKind::SetOperation {
            set_op_type,
            is_all,
            by_name,
            allow_missing_columns,
        },
        [left.into(), right.into()],
    )
}

/// Create a Limit plan.
pub fn limit(input: impl Into<Arc<LogicalPlan>>, limit: i32) -> LogicalPlan {
    LogicalPlan::from_builder(LogicalPlanKind::Limit { limit }, [input.into()])
}

/// Create an Offset plan.
pub fn offset(input: impl Into<Arc<LogicalPlan>>, offset: i32) -> LogicalPlan {
    LogicalPlan::from_builder(LogicalPlanKind::Offset { offset }, [input.into()])
}

/// Create a Tail plan.
pub fn tail(input: impl Into<Arc<LogicalPlan>>, limit: i32) -> LogicalPlan {
    LogicalPlan::from_builder(LogicalPlanKind::Tail { limit }, [input.into()])
}

/// Create a Deduplicate plan.
pub fn deduplicate(
    input: impl Into<Arc<LogicalPlan>>,
    all_columns_as_keys: bool,
    column_names: Vec<String>,
    within_watermark: bool,
) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::Deduplicate {
            all_columns_as_keys,
            column_names,
            within_watermark,
        },
        [input.into()],
    )
}

/// Create a Sort plan.
pub fn sort(
    input: impl Into<Arc<LogicalPlan>>,
    order: Vec<Expression>,
    is_global: bool,
) -> LogicalPlan {
    LogicalPlan::from_builder(LogicalPlanKind::Sort { order, is_global }, [input.into()])
}

/// Create a Sample plan.
pub fn sample(
    input: impl Into<Arc<LogicalPlan>>,
    lower_bound: f64,
    upper_bound: f64,
    with_replacement: bool,
    seed: Option<i64>,
) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::Sample {
            lower_bound,
            upper_bound,
            with_replacement,
            seed,
            deterministic_order: false,
        },
        [input.into()],
    )
}

/// Create a Repartition plan.
pub fn repartition(
    input: impl Into<Arc<LogicalPlan>>,
    num_partitions: i32,
    shuffle: bool,
) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::Repartition {
            num_partitions,
            shuffle,
        },
        [input.into()],
    )
}

/// Create a RepartitionByExpression plan.
pub fn repartition_by_expression(
    input: impl Into<Arc<LogicalPlan>>,
    num_partitions: i32,
    expressions: Vec<Expression>,
) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::RepartitionByExpression {
            num_partitions,
            expressions,
        },
        [input.into()],
    )
}

/// Create a WithColumns plan.
pub fn with_columns(
    input: impl Into<Arc<LogicalPlan>>,
    column_names: Vec<String>,
    columns: Vec<Column>,
) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::WithColumns {
            column_names,
            columns,
        },
        [input.into()],
    )
}

/// Create a WithColumnsRenamed plan of `renames`, of old and new names, in their order, in
/// which the server applies them.
pub fn with_columns_renamed(
    input: impl Into<Arc<LogicalPlan>>,
    renames: impl IntoIterator<Item = (String, String)>,
) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::WithColumnsRenamed {
            renames: renames.into_iter().collect(),
        },
        [input.into()],
    )
}

/// Create a Drop plan.
pub fn drop(input: impl Into<Arc<LogicalPlan>>, columns: Vec<String>) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::Drop {
            columns,
            expressions: Vec::new(),
        },
        [input.into()],
    )
}

/// Create a ToDF plan.
pub fn to_df(input: impl Into<Arc<LogicalPlan>>, column_names: Vec<String>) -> LogicalPlan {
    LogicalPlan::from_builder(LogicalPlanKind::ToDF { column_names }, [input.into()])
}

/// Create a ToSchema plan.
pub fn to_schema(input: impl Into<Arc<LogicalPlan>>, schema: DataType) -> LogicalPlan {
    LogicalPlan::from_builder(LogicalPlanKind::ToSchema { schema }, [input.into()])
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
    LogicalPlan::from_builder(
        LogicalPlanKind::Hint {
            name: name.into(),
            parameters: parameters.iter().map(|p| hint_parameter(p)).collect(),
        },
        [input.into()],
    )
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
    LogicalPlan::from_builder(
        LogicalPlanKind::Unpivot {
            ids,
            values,
            variable_column_name: variable_column_name.into(),
            value_column_name: value_column_name.into(),
        },
        [input.into()],
    )
}

/// Create a NAFill plan.
pub fn na_fill(
    input: impl Into<Arc<LogicalPlan>>,
    fill_value: crate::row::Value,
    columns: Vec<String>,
) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::NAFill {
            fill_value,
            columns,
        },
        [input.into()],
    )
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
    LogicalPlan::from_builder(
        LogicalPlanKind::NADrop {
            how: how.into(),
            min_non_null,
            columns,
        },
        [input.into()],
    )
}

/// Create a NAReplace plan, of replacements given as strings: a numeric-looking string is a
/// double, otherwise a string.
pub fn na_replace(
    input: impl Into<Arc<LogicalPlan>>,
    replacements: Vec<(String, String)>,
    columns: Vec<String>,
) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::NAReplace {
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
        },
        [input.into()],
    )
}

/// Create a Describe plan.
pub fn describe(input: impl Into<Arc<LogicalPlan>>, columns: Vec<String>) -> LogicalPlan {
    LogicalPlan::from_builder(LogicalPlanKind::Describe { columns }, [input.into()])
}

/// Create a Summary plan.
pub fn summary(input: impl Into<Arc<LogicalPlan>>, percentiles: Vec<String>) -> LogicalPlan {
    LogicalPlan::from_builder(LogicalPlanKind::Summary { percentiles }, [input.into()])
}

/// Create a ColRegex plan.
pub fn col_regex(input: impl Into<Arc<LogicalPlan>>, col_name: impl Into<String>) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::ColRegex {
            col_name: col_name.into(),
        },
        [input.into()],
    )
}

/// Create a SubqueryAlias plan.
pub fn subquery_alias(input: impl Into<Arc<LogicalPlan>>, alias: impl Into<String>) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::SubqueryAlias {
            alias: alias.into(),
        },
        [input.into()],
    )
}

/// Create a LocalRelation plan.
pub fn local_relation(schema: DataType, data: Option<Vec<u8>>) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::LocalRelation {
            schema: Some(schema),
            data,
        },
        [],
    )
}

/// Create a CachedRemoteRelation plan.
pub fn cached_remote_relation(relation_id: impl Into<String>) -> LogicalPlan {
    LogicalPlan::from_builder(
        LogicalPlanKind::CachedRemoteRelation {
            relation_id: relation_id.into(),
            owner: None,
        },
        [],
    )
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
        let mk = |how: &str, thresh: Option<i32>| {
            LogicalPlan::from_builder(
                LogicalPlanKind::NADrop {
                    how: how.to_string(),
                    min_non_null: thresh,
                    columns: vec![],
                },
                [Arc::new(base())],
            )
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
        let p = LogicalPlan::from_builder(
            LogicalPlanKind::NAReplace {
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
            },
            [Arc::new(base())],
        );
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
        let p = LogicalPlan::from_builder(
            LogicalPlanKind::NAFill {
                fill_value: crate::row::Value::Double(1.5),
                columns: vec![],
            },
            [Arc::new(base())],
        );
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
        let plan = Arc::new(LogicalPlan::from_builder(
            LogicalPlanKind::WithRelations,
            [
                Arc::new(sql("SELECT * FROM v")),
                Arc::new(subquery_alias(df.clone(), "v")),
            ],
        ));
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

#[cfg(test)]
mod placeholder_tests {
    use super::*;
    use prost::Message;
    use spark_connect_proto as proto;

    /// The placeholder of `node`, of the session of `ids`, as the relation that the plans of
    /// `pyspark.ml` send in its place.
    fn placeholder(node: &Arc<LogicalPlan>, ids: &Arc<PlanIds>) -> proto::Relation {
        placeholder_relation(node, ids)
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
    fn a_placeholder_is_shallow_and_stands_for_the_relation_of_its_node() {
        // The plan is converted recursively, so a plan of 200 levels takes more than the 2 MiB
        // stack of a test thread in a debug build, whose frames are much larger than those of a
        // release build.
        let run = || {
            let ids = Arc::new(PlanIds::new());
            let mut node = Arc::new(range(0, 3, 1));
            for _ in 0..200 {
                node = Arc::new(LogicalPlan::from_builder(
                    LogicalPlanKind::Limit { limit: 3 },
                    [node],
                ));
            }
            let relation = relation_with_ids(&node, &ids).unwrap();
            let shallow = placeholder(&node, &ids);
            assert!(shallow.encoded_len() < 128, "{}", shallow.encoded_len());
            let mut resolved = transform_of(shallow.clone());
            resolve_relation(&mut resolved).unwrap();
            assert_eq!(resolved, transform_of(relation));
        };
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(run)
            .unwrap()
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
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
        resolve_command(&mut fitted).unwrap();
        assert_eq!(fitted, fit(relation.clone()));

        let mut evaluated = command(proto::ml_command::Command::Evaluate(
            proto::ml_command::Evaluate {
                evaluator: None,
                params: None,
                dataset: Some(shallow.clone()),
            },
        ));
        resolve_command(&mut evaluated).unwrap();
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
        resolve_command(&mut summary).unwrap();
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
        let mut root = proto::ExecutePlanRequest {
            plan: Some(proto::Plan {
                op_type: Some(proto::plan::OpType::Root(transform_of(shallow.clone()))),
            }),
            ..Default::default()
        };
        resolve_request(&mut root).unwrap();
        assert_eq!(
            root.plan.unwrap().op_type,
            Some(proto::plan::OpType::Root(transform_of(relation)))
        );
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
        resolve_command(&mut command).unwrap();
        assert_eq!(command, proto::Command::default());
    }

    #[test]
    fn a_placeholder_of_a_dropped_node_is_an_error_and_keeps_nothing_alive() {
        let ids = Arc::new(PlanIds::new());
        let node = Arc::new(range(0, 3, 1));
        let alive = Arc::downgrade(&node);
        let shallow = placeholder(&node, &ids);
        std::mem::drop(node);
        assert!(alive.upgrade().is_none());
        let mut relation = transform_of(shallow);
        let error = resolve_relation(&mut relation).unwrap_err();
        assert!(error.to_string().contains("no longer exists"), "{error}");
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
        let with = LogicalPlan::from_builder(
            LogicalPlanKind::WithRelations,
            [chain(3), chain(2), chain(7)],
        );
        assert_eq!(with.depth(), 8);
    }

    #[test]
    fn depth_visits_an_input_that_plans_share_once() {
        // Each level refers to the one below twice: 2^64 paths, but 65 nodes.
        let mut plan = Arc::new(range(0, 10, 1));
        for _ in 0..64 {
            plan = Arc::new(LogicalPlan::from_builder(
                LogicalPlanKind::WithRelations,
                [Arc::clone(&plan), plan],
            ));
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
        Arc::new(LogicalPlan::from_builder(
            LogicalPlanKind::Project {
                columns: vec![crate::column::Column::new(expr)],
            },
            [chain(plan_levels)],
        ))
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
        let mut plan = Arc::new(LogicalPlan::from_builder(
            LogicalPlanKind::Filter {
                condition: crate::column::Column::new(Expression::new(ExpressionSpec::Subquery(
                    Box::new(subquery),
                ))),
            },
            [Arc::new(range(0, 10, 1))],
        ));
        for _ in 0..limits {
            plan = Arc::new(limit(plan, 1));
        }
        plan
    }

    #[test]
    fn raw_depth_keeps_logical_levels_when_subquery_wrappers_add_wire_nesting() {
        let ids = Arc::new(PlanIds::new());
        let input = limits_over_a_filter_of_a_subquery(0, 1);
        let (wire, levels) = relation_with_ids_and_levels(&input, &ids).unwrap();
        assert_eq!(levels, 2);
        assert_eq!(opaque_relation_depth(&wire), 3);

        let mut relation = proto::Relation {
            common: None,
            rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                proto::MlRelation {
                    model_summary_dataset: None,
                    ml_type: Some(proto::ml_relation::MlType::Transform(Box::new(
                        proto::ml_relation::Transform {
                            input: Some(Box::new(placeholder_relation(&input, &ids))),
                            params: None,
                            operator: None,
                        },
                    ))),
                },
            ))),
        };
        let declared = resolve_relation_levels(&mut relation).unwrap();
        assert_eq!(declared, 3);
        assert_eq!(opaque_relation_depth(&relation), 4);
        let raw = Arc::new(LogicalPlan::from_builder(
            LogicalPlanKind::Raw {
                relation: OpaqueRelation::new(relation),
                owner: None,
                depth: declared,
                observations: Vec::new(),
            },
            [],
        ));
        assert_eq!(raw.depth(), 3);
        let LogicalPlanKind::Raw { relation, .. } = raw.kind() else {
            unreachable!();
        };
        assert_eq!(relation.depth, 4);
        assert_eq!(relation_with_ids_and_levels(&raw, &ids).unwrap().1, 3);
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
        let depth = resolve_relation_levels(&mut relation).unwrap();
        Arc::new(LogicalPlan::from_builder(
            LogicalPlanKind::Raw {
                relation: OpaqueRelation::new(relation),
                owner: None,
                depth,
                observations: Vec::new(),
            },
            [],
        ))
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
            let filter = Arc::new(LogicalPlan::from_builder(
                LogicalPlanKind::Filter {
                    condition: crate::column::Column::new(Expression::new(
                        ExpressionSpec::Subquery(Box::new(subquery)),
                    )),
                },
                [Arc::new(range(0, 10, 1))],
            ));
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
}

#[cfg(test)]
mod opaque_metadata_tests {
    use super::*;

    #[allow(deprecated)]
    fn metadata() -> proto::RelationCommon {
        proto::RelationCommon {
            source_info: "node provenance".to_string(),
            plan_id: Some(-1),
            ..Default::default()
        }
    }

    fn raw_with_metadata(common: proto::RelationCommon) -> LogicalPlan {
        let mut relation = range(0, 3, 1).to_proto();
        relation.common = Some(common);
        LogicalPlan::from_builder(
            LogicalPlanKind::Raw {
                relation: OpaqueRelation::new(relation),
                owner: None,
                depth: 1,
                observations: Vec::new(),
            },
            [],
        )
    }

    #[test]
    fn request_ids_override_only_the_opaque_metadata_id() {
        let common = metadata();
        let node = Arc::new(raw_with_metadata(common.clone()));
        assert_eq!(node.to_proto().common, Some(common.clone()));

        let ids = PlanIds::new();
        let mut expected = common;
        expected.plan_id = Some(ids.id_of(&node));
        assert_eq!(
            relation_with_ids(&node, &ids).unwrap().common,
            Some(expected.clone())
        );
        assert_eq!(
            relation_with_ids(&Arc::clone(&node), &ids).unwrap().common,
            Some(expected)
        );
    }

    #[test]
    fn shared_opaque_metadata_keeps_one_stable_id_in_both_positions() {
        let common = metadata();
        let shared = Arc::new(raw_with_metadata(common.clone()));
        let root = Arc::new(LogicalPlan::from_builder(
            LogicalPlanKind::Zip,
            [Arc::clone(&shared), Arc::clone(&shared)],
        ));
        let ids = PlanIds::new();
        let mut expected = common;
        expected.plan_id = Some(ids.id_of(&shared));
        for _ in 0..2 {
            let relation = relation_with_ids(&root, &ids).unwrap();
            assert_eq!(
                relation.common.as_ref().unwrap().plan_id,
                Some(ids.id_of(&root))
            );
            let Some(proto::relation::RelType::Zip(zip)) = relation.rel_type else {
                panic!("a zip of the shared node");
            };
            assert_eq!(zip.left.as_ref().unwrap().common, Some(expected.clone()));
            assert_eq!(zip.right.as_ref().unwrap().common, Some(expected.clone()));
        }
        assert_ne!(ids.id_of(&root), ids.id_of(&shared));
    }
}

#[cfg(test)]
mod observation_plan_tests {
    use super::*;
    use crate::observation::Observation;

    fn observe(input: Arc<LogicalPlan>, observation: &Observation) -> Arc<LogicalPlan> {
        Arc::new(LogicalPlan::from_builder(
            LogicalPlanKind::Observe {
                name: observation.name().unwrap(),
                exprs: vec![],
                observation: Some(observation.clone()),
            },
            [input],
        ))
    }

    #[test]
    fn observations_keep_outer_and_right_precedence_across_a_shared_dag() {
        let shared = Observation::new("shared");
        let input = observe(Arc::new(range(0, 3, 1)), &shared);
        let left = Observation::new("duplicate");
        let right = Observation::new("duplicate");
        let mut root = Arc::new(LogicalPlan::from_builder(
            LogicalPlanKind::Zip,
            [observe(Arc::clone(&input), &left), observe(input, &right)],
        ));
        let collected = root.observations();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].0, "duplicate");
        assert!(collected[0].1.same(&right));
        assert_eq!(collected[1].0, "shared");
        assert!(collected[1].1.same(&shared));

        let outer = Observation::new("duplicate");
        root = observe(root, &outer);
        for _ in 0..24 {
            // Precedence stays the same when input plans are shared.
            root = Arc::new(LogicalPlan::from_builder(
                LogicalPlanKind::Zip,
                [Arc::clone(&root), root],
            ));
        }
        let collected = root.observations();
        assert_eq!(collected.len(), 2);
        assert!(collected[0].1.same(&outer));
        assert!(collected[1].1.same(&shared));
    }

    #[test]
    fn cloning_and_rebuilding_raw_nodes_keep_their_observation_handles() {
        let observation = Observation::new("held");
        let node = LogicalPlan::try_new(
            LogicalPlanKind::Raw {
                relation: OpaqueRelation::new(range(0, 3, 1).to_proto()),
                owner: None,
                depth: 1,
                observations: vec![("held".into(), observation.clone())],
            },
            vec![],
        )
        .unwrap();

        for copy in [node.clone(), node.with_children(vec![]).unwrap()] {
            let observations = copy.observations();
            assert_eq!(observations.len(), 1);
            assert_eq!(observations[0].0, "held");
            assert!(observations[0].1.same(&observation));
        }
    }
}

#[cfg(test)]
mod opaque_clone_tests {
    use super::*;

    fn wire_chain(levels: usize) -> proto::Relation {
        let mut wire = range(0, 3, 1).to_proto();
        for _ in 1..levels {
            wire = proto::Relation {
                common: None,
                rel_type: Some(proto::relation::RelType::Limit(Box::new(proto::Limit {
                    input: Some(Box::new(wire)),
                    limit: 1,
                }))),
            };
        }
        wire
    }

    fn fetch_chain(levels: usize) -> proto::Relation {
        let mut wire = range(0, 3, 1).to_proto();
        for level in 1..levels {
            wire = proto::Relation {
                common: None,
                rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                    proto::MlRelation {
                        model_summary_dataset: None,
                        ml_type: Some(proto::ml_relation::MlType::Fetch(proto::Fetch {
                            obj_ref: None,
                            methods: vec![proto::fetch::Method {
                                method: format!("fetch_{level}"),
                                args: vec![proto::fetch::method::Args {
                                    args_type: Some(proto::fetch::method::args::ArgsType::Input(
                                        wire,
                                    )),
                                }],
                            }],
                        })),
                    },
                ))),
            };
        }
        wire
    }

    fn opaque_payload(plan: &LogicalPlan) -> &OpaqueRelation {
        match plan.kind() {
            LogicalPlanKind::Raw { relation, .. }
            | LogicalPlanKind::CachedRelation { relation } => relation,
            LogicalPlanKind::MlTransform { ml_relation, .. } => ml_relation,
            _ => panic!("an opaque relation"),
        }
    }

    fn opaque_wire(plan: &LogicalPlan) -> &proto::Relation {
        opaque_payload(plan).as_proto()
    }

    fn on_stack<T: Send>(bytes: usize, f: impl FnOnce() -> T + Send) -> T {
        std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(bytes)
                .spawn_scoped(scope, f)
                .unwrap()
                .join()
                .unwrap()
        })
    }

    #[test]
    fn opaque_sizing_includes_secondary_inputs_and_ml_datasets() {
        use proto::relation::RelType;

        let deep = wire_chain(3);
        let cases = [
            RelType::WithRelations(Box::new(proto::WithRelations {
                root: Some(Box::new(wire_chain(1))),
                references: vec![deep.clone()],
            })),
            RelType::GroupMap(Box::new(proto::GroupMap {
                input: Some(Box::new(wire_chain(1))),
                initial_input: Some(Box::new(deep.clone())),
                ..Default::default()
            })),
            RelType::MlRelation(Box::new(proto::MlRelation {
                model_summary_dataset: Some(Box::new(deep.clone())),
                ml_type: None,
            })),
            RelType::MlRelation(Box::new(proto::MlRelation {
                model_summary_dataset: None,
                ml_type: Some(proto::ml_relation::MlType::Transform(Box::new(
                    proto::ml_relation::Transform {
                        input: Some(Box::new(deep.clone())),
                        ..Default::default()
                    },
                ))),
            })),
            RelType::MlRelation(Box::new(proto::MlRelation {
                model_summary_dataset: None,
                ml_type: Some(proto::ml_relation::MlType::Fetch(proto::Fetch {
                    obj_ref: None,
                    methods: vec![proto::fetch::Method {
                        method: "transform".into(),
                        args: vec![proto::fetch::method::Args {
                            args_type: Some(proto::fetch::method::args::ArgsType::Input(deep)),
                        }],
                    }],
                })),
            })),
        ];
        for kind in cases {
            let wire = proto::Relation {
                common: None,
                rel_type: Some(kind),
            };
            assert_eq!(opaque_relation_depth(&wire), 4);
            assert_eq!(clone_opaque_relation(&wire, 4), wire);
        }
        let extension = proto::Relation {
            common: None,
            rel_type: Some(RelType::Extension(prost_types::Any::default())),
        };
        assert_eq!(opaque_relation_depth(&extension), 1);
    }

    #[test]
    fn node_clones_and_rebuilds_use_the_opaque_wire_depth() {
        // Prost also drops recursively, so construct and release these deep wire
        // payloads on a large stack. Payload/node cloning, rebuilding and serialization
        // run on 512 KiB; recursive wire equality and destruction run on 64 MiB.
        on_stack(64 << 20, || {
            let levels = 2_000;
            for wire in [wire_chain(levels), fetch_chain(levels)] {
                let nodes = [
                    LogicalPlan::try_new(
                        LogicalPlanKind::Raw {
                            relation: OpaqueRelation::new(wire.clone()),
                            owner: None,
                            // A custom Raw may declare fewer levels than its proto.
                            // Clone sizing caches the actual relation tree's depth.
                            depth: 1,
                            observations: Vec::new(),
                        },
                        vec![],
                    )
                    .unwrap(),
                    LogicalPlan::from_builder(
                        LogicalPlanKind::CachedRelation {
                            relation: OpaqueRelation::new(wire.clone()),
                        },
                        [],
                    ),
                    LogicalPlan::try_new(
                        LogicalPlanKind::MlTransform {
                            ml_relation: OpaqueRelation::new(wire),
                            owner: None,
                        },
                        // Deliberately mismatched custom input: clone sizing must
                        // still use the actual opaque wire depth.
                        vec![Arc::new(range(0, 3, 1))],
                    )
                    .unwrap(),
                ];
                for node in nodes {
                    assert_eq!(opaque_payload(&node).depth, levels);
                    let kind_clone =
                        on_stack(512 << 10, || <LogicalPlanKind as Clone>::clone(node.kind()));
                    let payload_clone =
                        LogicalPlan::try_new(kind_clone, node.children().to_vec()).unwrap();
                    let cloned = on_stack(512 << 10, || node.clone());
                    let rebuilt = on_stack(512 << 10, || {
                        node.with_children(node.children().to_vec()).unwrap()
                    });
                    for copy in [payload_clone, cloned, rebuilt] {
                        assert_eq!(opaque_payload(&copy).depth, opaque_payload(&node).depth);
                        assert_eq!(opaque_wire(&copy), opaque_wire(&node));
                    }
                    let serialized = on_stack(512 << 10, || node.to_proto());
                    let mut expected = opaque_wire(&node).clone();
                    if !matches!(node.kind(), LogicalPlanKind::MlTransform { .. }) {
                        expected.common = Some(proto::RelationCommon::default());
                    }
                    assert_eq!(serialized, expected);
                }
            }
        });
    }
}
