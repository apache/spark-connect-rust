//! Logical plan nodes mirroring `pyspark.sql.connect.plan`.
//!
//! Each node builds a `spark.connect.Relation` (or `Command`) protobuf matching
//! the reference PySpark client (see `tests/golden/plans.jsonl`).

use spark_connect_proto as proto;
use std::collections::HashMap;

use crate::column::Column;
use crate::expression::Expression;
use crate::types::{DataType, StructField};
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
/// Call `to_proto()` to convert to a `spark.connect.Relation` protobuf.
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// A range of integers: `range(start, end, step)`.
    Range {
        start: i64,
        end: i64,
        step: i64,
        num_partitions: Option<i32>,
    },
    /// A SQL query: `sql("SELECT ...")`.
    Sql { query: String },
    /// A projection (select): `df.select(cols...)`.
    Project {
        input: Box<LogicalPlan>,
        columns: Vec<Column>,
    },
    /// A filter: `df.filter(condition)`.
    Filter {
        input: Box<LogicalPlan>,
        condition: Column,
    },
    /// An aggregation: `df.groupBy(...).agg(...)`.
    Aggregate {
        input: Box<LogicalPlan>,
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
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        join_type: JoinType,
        on: Option<Column>,
        using_columns: Vec<String>,
    },
    /// A lateral join (`LATERAL` correlated subquery join).
    LateralJoin {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        join_type: JoinType,
        on: Option<Column>,
    },
    /// A set operation (union, intersect, except).
    SetOperation {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        set_op_type: SetOpType,
        is_all: bool,
        by_name: bool,
        allow_missing_columns: bool,
    },
    /// Limit: `df.limit(n)`.
    Limit { input: Box<LogicalPlan>, limit: i32 },
    /// Offset: `df.offset(n)`.
    Offset {
        input: Box<LogicalPlan>,
        offset: i32,
    },
    /// Tail: `df.tail(n)` - returns last n rows.
    Tail { input: Box<LogicalPlan>, limit: i32 },
    /// Deduplicate: `df.distinct()` or `df.dropDuplicates()`.
    Deduplicate {
        input: Box<LogicalPlan>,
        all_columns_as_keys: bool,
        column_names: Vec<String>,
        within_watermark: bool,
    },
    /// Sort: `df.sort(cols...)`.
    Sort {
        input: Box<LogicalPlan>,
        order: Vec<Expression>,
        is_global: bool,
    },
    /// Sample: `df.sample(fraction)` or `df.sample(num_rows)`.
    Sample {
        input: Box<LogicalPlan>,
        lower_bound: f64,
        upper_bound: f64,
        with_replacement: bool,
        seed: Option<i64>,
    },
    /// Repartition: `df.repartition(num_partitions)`.
    Repartition {
        input: Box<LogicalPlan>,
        num_partitions: i32,
        shuffle: bool,
    },
    /// RepartitionByExpression: `df.repartitionByRange()` or similar.
    RepartitionByExpression {
        input: Box<LogicalPlan>,
        num_partitions: i32,
        expressions: Vec<Expression>,
    },
    /// WithColumns: `df.withColumn(name, col)` or `df.withColumns(...)`.
    WithColumns {
        input: Box<LogicalPlan>,
        column_names: Vec<String>,
        columns: Vec<Column>,
    },
    /// WithColumnsRenamed: `df.withColumnRenamed(old, new)` or `df.withColumnsRenamed(...)`.
    WithColumnsRenamed {
        input: Box<LogicalPlan>,
        renames: HashMap<String, String>,
    },
    /// Drop: `df.drop(column_names...)`.
    Drop {
        input: Box<LogicalPlan>,
        columns: Vec<String>,
    },
    /// ToDF: `df.toDF(*column_names)`.
    ToDF {
        input: Box<LogicalPlan>,
        column_names: Vec<String>,
    },
    /// ToSchema: set schema to match the provided type.
    ToSchema {
        input: Box<LogicalPlan>,
        schema: DataType,
    },
    /// Hint: `df.hint(name, parameters...)`.
    Hint {
        input: Box<LogicalPlan>,
        name: String,
        parameters: Vec<String>,
    },
    /// Unpivot: `df.unpivot(...)`.
    Unpivot {
        input: Box<LogicalPlan>,
        ids: Vec<Column>,
        values: Option<Vec<Column>>,
        variable_column_name: String,
        value_column_name: String,
    },
    /// NAFill: `df.fillna(value, subset)`.
    NAFill {
        input: Box<LogicalPlan>,
        fill_value: crate::row::Value,
        columns: Vec<String>,
    },
    /// NADrop: `df.dropna(how, thresh, subset)`.
    NADrop {
        input: Box<LogicalPlan>,
        how: String,
        min_non_null: Option<i32>,
        columns: Vec<String>,
    },
    /// NAReplace: `df.replace(to_replace, value, subset)`.
    NAReplace {
        input: Box<LogicalPlan>,
        replacements: Vec<(String, String)>,
        columns: Vec<String>,
    },
    /// Describe: `df.describe(cols...)`.
    Describe {
        input: Box<LogicalPlan>,
        columns: Vec<String>,
    },
    /// Summary: `df.summary(percentiles...)`.
    Summary {
        input: Box<LogicalPlan>,
        percentiles: Vec<String>,
    },
    /// ColRegex: column selection by regex.
    ColRegex {
        input: Box<LogicalPlan>,
        col_name: String,
    },
    /// SubqueryAlias: `df.as(alias)`.
    SubqueryAlias {
        input: Box<LogicalPlan>,
        alias: String,
    },
    /// LocalRelation: data supplied locally.
    LocalRelation {
        schema: DataType,
        data: Option<Vec<u8>>,
    },
    /// CachedRemoteRelation: a reference to a cached remote relation.
    CachedRemoteRelation { relation_id: String },
    /// Read: `spark.read.format(...).load(...)` or `spark.read.table(...)`.
    Read {
        read_type: crate::readwriter::ReadType,
        is_streaming: bool,
    },
    /// WithWatermark: `df.withWatermark(timeColumn, delayThreshold)`.
    WithWatermark {
        input: Box<LogicalPlan>,
        time_column: String,
        delay_threshold: String,
    },
    /// RepartitionByRange: `df.repartitionByRange(numPartitions, *cols)`.
    RepartitionByRange {
        input: Box<LogicalPlan>,
        num_partitions: Option<i32>,
        partition_exprs: Vec<Expression>,
    },
    /// StatCrosstab: `df.stat.crosstab(col1, col2)`.
    StatCrosstab {
        input: Box<LogicalPlan>,
        col1: String,
        col2: String,
    },
    /// StatFreqItems: `df.stat.freqItems(columns, support)`.
    StatFreqItems {
        input: Box<LogicalPlan>,
        columns: Vec<String>,
        support: f64,
    },
    /// StatApproxQuantile: `df.stat.approxQuantile(columns, probabilities, relativeError)`.
    StatApproxQuantile {
        input: Box<LogicalPlan>,
        columns: Vec<String>,
        probabilities: Vec<f64>,
        relative_error: f64,
    },
    /// StatCorr: `df.stat.corr(col1, col2)`.
    StatCorr {
        input: Box<LogicalPlan>,
        col1: String,
        col2: String,
    },
    /// StatCov: `df.stat.cov(col1, col2)`.
    StatCov {
        input: Box<LogicalPlan>,
        col1: String,
        col2: String,
    },
    /// StatSampleBy: `df.stat.sampleBy(col, fractions, seed)`.
    StatSampleBy {
        input: Box<LogicalPlan>,
        col: String,
        fractions: Vec<(Expression, f64)>,
        seed: Option<i64>,
    },
    /// Observe: `df.observe(name, exprs)` - collect metrics.
    Observe {
        input: Box<LogicalPlan>,
        name: String,
        exprs: Vec<Expression>,
    },
    /// UnresolvedTableValuedFunction: a TVF like explode, inline, range.
    UnresolvedTableValuedFunction {
        name: String,
        arguments: Vec<Expression>,
    },
    /// Zip: `df.zip(other)` - combine two DataFrames column-wise.
    Zip {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
    },
    /// ML Transformation: wraps an MlRelation proto directly.
    MlTransform { ml_relation: proto::Relation },
    /// MapPartitions: `DataFrame.mapInPandas` / `mapInArrow` (also backs foreach).
    MapPartitions {
        input: Box<LogicalPlan>,
        func: CommonInlineUserDefinedFunctionExpression,
        is_barrier: bool,
    },
    /// GroupMap: `GroupedData.applyInPandas` / `applyInArrow`, and the stateful
    /// variants `applyInPandasWithState` / `transformWithState[InPandas]`.
    GroupMap {
        input: Box<LogicalPlan>,
        grouping_expressions: Vec<Expression>,
        func: CommonInlineUserDefinedFunctionExpression,
        sorting_expressions: Vec<Expression>,
        // Stateful fields (all default to None/empty for plain applyInPandas/applyInArrow).
        initial_input: Option<Box<LogicalPlan>>,
        initial_grouping_expressions: Vec<Expression>,
        is_map_groups_with_state: Option<bool>,
        output_mode: Option<String>,
        timeout_conf: Option<String>,
        state_schema: Option<DataType>,
        transform_with_state_info: Option<TransformWithStateInfo>,
    },
    /// CoGroupMap: `GroupedData.cogroup(...).applyInPandas` / `applyInArrow`.
    CoGroupMap {
        input: Box<LogicalPlan>,
        input_grouping_expressions: Vec<Expression>,
        other: Box<LogicalPlan>,
        other_grouping_expressions: Vec<Expression>,
        func: CommonInlineUserDefinedFunctionExpression,
    },
    /// NearestByJoin: `DataFrame.nearestByJoin`.
    NearestByJoin {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
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

impl LogicalPlan {
    /// Convert this logical plan to a `spark.connect.Relation` protobuf.
    pub fn to_proto(&self) -> proto::Relation {
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

            LogicalPlan::Sql { query } => {
                let mut sql = proto::Sql::default();
                sql.query = query.clone();
                relation.rel_type = Some(proto::relation::RelType::Sql(sql));
            }

            LogicalPlan::Project { input, columns } => {
                let mut project = proto::Project::default();
                project.input = Some(Box::new(input.to_proto()));
                for col in columns {
                    project.expressions.push(col.to_proto());
                }
                relation.rel_type = Some(proto::relation::RelType::Project(Box::new(project)));
            }

            LogicalPlan::Filter { input, condition } => {
                let mut filter = proto::Filter::default();
                filter.input = Some(Box::new(input.to_proto()));
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
                agg.input = Some(Box::new(input.to_proto()));
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
                join.left = Some(Box::new(left.to_proto()));
                join.right = Some(Box::new(right.to_proto()));
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
                lj.left = Some(Box::new(left.to_proto()));
                lj.right = Some(Box::new(right.to_proto()));
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
                set_op.left_input = Some(Box::new(left.to_proto()));
                set_op.right_input = Some(Box::new(right.to_proto()));
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
                lim.input = Some(Box::new(input.to_proto()));
                lim.limit = *limit;
                relation.rel_type = Some(proto::relation::RelType::Limit(Box::new(lim)));
            }

            LogicalPlan::Offset { input, offset } => {
                let mut off = proto::Offset::default();
                off.input = Some(Box::new(input.to_proto()));
                off.offset = *offset;
                relation.rel_type = Some(proto::relation::RelType::Offset(Box::new(off)));
            }

            LogicalPlan::Tail { input, limit } => {
                let mut tail = proto::Tail::default();
                tail.input = Some(Box::new(input.to_proto()));
                tail.limit = *limit;
                relation.rel_type = Some(proto::relation::RelType::Tail(Box::new(tail)));
            }

            LogicalPlan::Deduplicate {
                input,
                all_columns_as_keys,
                column_names,
                within_watermark,
            } => {
                let mut dedup = proto::Deduplicate::default();
                dedup.input = Some(Box::new(input.to_proto()));
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
                let mut sort = proto::Sort::default();
                sort.input = Some(Box::new(input.to_proto()));
                for expr in order {
                    // Extract SortOrder from Expression if it's SortOrder type
                    if let Expression::SortOrder(so) = expr {
                        let mut sort_order = proto::expression::SortOrder::default();
                        sort_order.child = Some(Box::new(so.child.to_proto()));
                        sort_order.direction = if so.ascending { 1i32 } else { 2i32 };
                        sort_order.null_ordering = match so.null_ordering {
                            crate::expression::NullOrdering::First => 1i32,
                            crate::expression::NullOrdering::Last => 2i32,
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
                            sort_order.direction = 1i32; // ascending
                            sort_order.null_ordering = 1i32; // nulls first
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
            } => {
                let mut sample = proto::Sample::default();
                sample.input = Some(Box::new(input.to_proto()));
                sample.lower_bound = *lower_bound;
                sample.upper_bound = *upper_bound;
                sample.with_replacement = Some(*with_replacement);
                if let Some(s) = seed {
                    sample.seed = Some(*s);
                }
                relation.rel_type = Some(proto::relation::RelType::Sample(Box::new(sample)));
            }

            LogicalPlan::Repartition {
                input,
                num_partitions,
                shuffle,
            } => {
                let mut repart = proto::Repartition::default();
                repart.input = Some(Box::new(input.to_proto()));
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
                repart.input = Some(Box::new(input.to_proto()));
                repart.num_partitions = Some(*num_partitions);
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
                wc.input = Some(Box::new(input.to_proto()));
                for (name, col) in column_names.iter().zip(columns.iter()) {
                    // Create an Alias expression for each column
                    let mut alias = proto::expression::Alias::default();
                    alias.expr = Some(Box::new(col.to_proto()));
                    alias.name = vec![name.clone()];
                    wc.aliases.push(alias);
                }
                relation.rel_type = Some(proto::relation::RelType::WithColumns(Box::new(wc)));
            }

            LogicalPlan::WithColumnsRenamed { input, renames } => {
                let mut wcr = proto::WithColumnsRenamed::default();
                wcr.input = Some(Box::new(input.to_proto()));
                for (old_name, new_name) in renames.iter() {
                    let mut rename = proto::with_columns_renamed::Rename::default();
                    rename.col_name = old_name.clone();
                    rename.new_col_name = new_name.clone();
                    wcr.renames.push(rename);
                }
                relation.rel_type =
                    Some(proto::relation::RelType::WithColumnsRenamed(Box::new(wcr)));
            }

            LogicalPlan::Drop { input, columns } => {
                let mut drop = proto::Drop::default();
                drop.input = Some(Box::new(input.to_proto()));
                drop.column_names.extend(columns.clone());
                relation.rel_type = Some(proto::relation::RelType::Drop(Box::new(drop)));
            }

            LogicalPlan::ToDF {
                input,
                column_names,
            } => {
                let mut to_df = proto::ToDf::default();
                to_df.input = Some(Box::new(input.to_proto()));
                to_df.column_names.extend(column_names.clone());
                relation.rel_type = Some(proto::relation::RelType::ToDf(Box::new(to_df)));
            }

            LogicalPlan::ToSchema { input, schema } => {
                let mut to_schema = proto::ToSchema::default();
                to_schema.input = Some(Box::new(input.to_proto()));
                to_schema.schema = Some(schema.to_proto());
                relation.rel_type = Some(proto::relation::RelType::ToSchema(Box::new(to_schema)));
            }

            LogicalPlan::Hint {
                input,
                name,
                parameters,
            } => {
                let mut hint = proto::Hint::default();
                hint.input = Some(Box::new(input.to_proto()));
                hint.name = name.clone();
                // Hint parameters are literal Expressions (e.g. `REPARTITION 10`).
                // An integer-looking parameter becomes a Long literal, otherwise a
                // String literal.
                for p in parameters {
                    let lit = if let Ok(n) = p.parse::<i64>() {
                        crate::expression::LiteralExpression::long(n)
                    } else {
                        crate::expression::LiteralExpression::string(p.clone())
                    };
                    hint.parameters.push(lit.to_proto());
                }
                relation.rel_type = Some(proto::relation::RelType::Hint(Box::new(hint)));
            }

            LogicalPlan::Unpivot {
                input,
                ids,
                values,
                variable_column_name,
                value_column_name,
            } => {
                let mut unpivot = proto::Unpivot::default();
                unpivot.input = Some(Box::new(input.to_proto()));
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
                na_fill.input = Some(Box::new(input.to_proto()));
                na_fill.cols.extend(columns.clone());
                na_fill.values.push(value_to_proto_literal(fill_value));
                relation.rel_type = Some(proto::relation::RelType::FillNa(Box::new(na_fill)));
            }

            LogicalPlan::NADrop {
                input,
                how,
                min_non_null,
                columns,
            } => {
                let mut na_drop = proto::NaDrop::default();
                na_drop.input = Some(Box::new(input.to_proto()));
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
                na_replace.input = Some(Box::new(input.to_proto()));
                for (old_val, new_val) in replacements.iter() {
                    let mut replace = proto::na_replace::Replacement::default();
                    // A numeric-looking value becomes a Double literal; otherwise a
                    // String literal. Both sides are always set (never left None, which
                    // previously made string replacements a silent no-op).
                    replace.old_value = Some(str_to_proto_literal(old_val));
                    replace.new_value = Some(str_to_proto_literal(new_val));
                    na_replace.replacements.push(replace);
                }
                na_replace.cols.extend(columns.clone());
                relation.rel_type = Some(proto::relation::RelType::Replace(Box::new(na_replace)));
            }

            LogicalPlan::Describe { input, columns } => {
                let mut describe = proto::StatDescribe::default();
                describe.input = Some(Box::new(input.to_proto()));
                describe.cols.extend(columns.clone());
                relation.rel_type = Some(proto::relation::RelType::Describe(Box::new(describe)));
            }

            LogicalPlan::Summary { input, percentiles } => {
                let mut summary = proto::StatSummary::default();
                summary.input = Some(Box::new(input.to_proto()));
                summary.statistics.extend(percentiles.clone());
                relation.rel_type = Some(proto::relation::RelType::Summary(Box::new(summary)));
            }

            LogicalPlan::ColRegex { input, col_name } => {
                // ColRegex is implemented as a Project with UnresolvedRegex expressions
                let mut project = proto::Project::default();
                project.input = Some(Box::new(input.to_proto()));
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
                sq_alias.input = Some(Box::new(input.to_proto()));
                sq_alias.alias = alias.clone();
                relation.rel_type =
                    Some(proto::relation::RelType::SubqueryAlias(Box::new(sq_alias)));
            }

            LogicalPlan::LocalRelation { schema, data } => {
                let mut local = proto::LocalRelation::default();
                if let Some(d) = data {
                    local.data = Some(d.clone().into());
                }
                // Carry the schema (JSON) so the server applies the user's column
                // names/types; required when no Arrow `data` is provided (emptyDataFrame).
                local.schema = Some(schema.json());
                relation.rel_type = Some(proto::relation::RelType::LocalRelation(local));
            }

            LogicalPlan::CachedRemoteRelation { relation_id } => {
                let mut cached = proto::CachedRemoteRelation::default();
                cached.relation_id = relation_id.clone();
                relation.rel_type = Some(proto::relation::RelType::CachedRemoteRelation(cached));
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

            LogicalPlan::WithWatermark {
                input,
                time_column,
                delay_threshold,
            } => {
                let mut watermark = proto::WithWatermark::default();
                watermark.input = Some(Box::new(input.to_proto()));
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
                repart.input = Some(Box::new(input.to_proto()));
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
                stat.input = Some(Box::new(input.to_proto()));
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
                stat.input = Some(Box::new(input.to_proto()));
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
                stat.input = Some(Box::new(input.to_proto()));
                stat.cols.extend(columns.clone());
                stat.probabilities.extend(probabilities.clone());
                stat.relative_error = *relative_error;
                relation.rel_type = Some(proto::relation::RelType::ApproxQuantile(Box::new(stat)));
            }

            LogicalPlan::StatCorr { input, col1, col2 } => {
                let mut stat = proto::StatCorr::default();
                stat.input = Some(Box::new(input.to_proto()));
                stat.col1 = col1.clone();
                stat.col2 = col2.clone();
                relation.rel_type = Some(proto::relation::RelType::Corr(Box::new(stat)));
            }

            LogicalPlan::StatCov { input, col1, col2 } => {
                let mut stat = proto::StatCov::default();
                stat.input = Some(Box::new(input.to_proto()));
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
                stat.input = Some(Box::new(input.to_proto()));
                // col field expects an Expression; create one from the column name
                let mut col_expr = proto::Expression::default();
                col_expr.expr_type = Some(proto::expression::ExprType::UnresolvedAttribute(
                    proto::expression::UnresolvedAttribute {
                        unparsed_identifier: col.clone(),
                        plan_id: None,
                        is_metadata_column: None,
                    },
                ));
                stat.col = Some(col_expr);
                for (expr, _frac) in fractions {
                    // fractions expect Expression.Literal type; create Fraction proto
                    let mut fraction = proto::stat_sample_by::Fraction::default();
                    let expr_proto = expr.to_proto();
                    // Extract the Literal from the Expression proto
                    if let Some(proto::expression::ExprType::Literal(lit)) = expr_proto.expr_type {
                        fraction.stratum = Some(lit);
                    }
                    stat.fractions.push(fraction);
                }
                if let Some(s) = seed {
                    stat.seed = Some(*s);
                }
                relation.rel_type = Some(proto::relation::RelType::SampleBy(Box::new(stat)));
            }

            LogicalPlan::Observe { input, name, exprs } => {
                let mut collect_metrics = proto::CollectMetrics::default();
                collect_metrics.input = Some(Box::new(input.to_proto()));
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
                zip.left = Some(Box::new(left.to_proto()));
                zip.right = Some(Box::new(right.to_proto()));
                relation.rel_type = Some(proto::relation::RelType::Zip(Box::new(zip)));
            }

            LogicalPlan::MlTransform { ml_relation } => {
                return ml_relation.clone();
            }
            LogicalPlan::MapPartitions {
                input,
                func,
                is_barrier,
            } => {
                let mut mp = proto::MapPartitions::default();
                mp.input = Some(Box::new(input.to_proto()));
                mp.func = Some(func.to_proto());
                mp.is_barrier = Some(*is_barrier);
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
                gm.input = Some(Box::new(input.to_proto()));
                for e in grouping_expressions {
                    gm.grouping_expressions.push(e.to_proto());
                }
                gm.func = Some(func.to_proto());
                for e in sorting_expressions {
                    gm.sorting_expressions.push(e.to_proto());
                }
                if let Some(ii) = initial_input {
                    gm.initial_input = Some(Box::new(ii.to_proto()));
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
            LogicalPlan::CoGroupMap {
                input,
                input_grouping_expressions,
                other,
                other_grouping_expressions,
                func,
            } => {
                let mut cg = proto::CoGroupMap::default();
                cg.input = Some(Box::new(input.to_proto()));
                for e in input_grouping_expressions {
                    cg.input_grouping_expressions.push(e.to_proto());
                }
                cg.other = Some(Box::new(other.to_proto()));
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
                nbj.left = Some(Box::new(left.to_proto()));
                nbj.right = Some(Box::new(right.to_proto()));
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
                let mut udtf = proto::PythonUdtf::default();
                if let Some(rt) = return_type {
                    udtf.return_type = Some(rt.to_proto());
                }
                udtf.eval_type = *eval_type;
                udtf.command = bytes::Bytes::copy_from_slice(command);
                udtf.python_ver = python_ver.clone();
                let mut f = proto::CommonInlineUserDefinedTableFunction::default();
                f.function_name = function_name.clone();
                f.deterministic = *deterministic;
                f.arguments = arguments.iter().map(|a| a.to_proto()).collect();
                f.function = Some(
                    proto::common_inline_user_defined_table_function::Function::PythonUdtf(udtf),
                );
                relation.rel_type =
                    Some(proto::relation::RelType::CommonInlineUserDefinedTableFunction(f));
            }
        }

        relation
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
    }
}

/// Create a Project plan.
pub fn project(input: LogicalPlan, columns: Vec<Column>) -> LogicalPlan {
    LogicalPlan::Project {
        input: Box::new(input),
        columns,
    }
}

/// Create a Filter plan.
pub fn filter(input: LogicalPlan, condition: Column) -> LogicalPlan {
    LogicalPlan::Filter {
        input: Box::new(input),
        condition,
    }
}

/// Create an Aggregate plan.
pub fn aggregate(
    input: LogicalPlan,
    group_type: AggregateGroupType,
    grouping_expressions: Vec<Expression>,
    aggregate_expressions: Vec<Expression>,
) -> LogicalPlan {
    LogicalPlan::Aggregate {
        input: Box::new(input),
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
    input: LogicalPlan,
    group_type: AggregateGroupType,
    grouping_expressions: Vec<Expression>,
    aggregate_expressions: Vec<Expression>,
    pivot_col: Expression,
    pivot_values: Vec<Expression>,
) -> LogicalPlan {
    LogicalPlan::Aggregate {
        input: Box::new(input),
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
    input: LogicalPlan,
    grouping_expressions: Vec<Expression>,
    aggregate_expressions: Vec<Expression>,
    grouping_sets: Vec<Vec<Expression>>,
) -> LogicalPlan {
    LogicalPlan::Aggregate {
        input: Box::new(input),
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
    left: LogicalPlan,
    right: LogicalPlan,
    join_type: JoinType,
    on: Option<Column>,
    using_columns: Vec<String>,
) -> LogicalPlan {
    LogicalPlan::Join {
        left: Box::new(left),
        right: Box::new(right),
        join_type,
        on,
        using_columns,
    }
}

/// Create a SetOperation plan.
pub fn set_operation(
    left: LogicalPlan,
    right: LogicalPlan,
    set_op_type: SetOpType,
    is_all: bool,
    by_name: bool,
    allow_missing_columns: bool,
) -> LogicalPlan {
    LogicalPlan::SetOperation {
        left: Box::new(left),
        right: Box::new(right),
        set_op_type,
        is_all,
        by_name,
        allow_missing_columns,
    }
}

/// Create a Limit plan.
pub fn limit(input: LogicalPlan, limit: i32) -> LogicalPlan {
    LogicalPlan::Limit {
        input: Box::new(input),
        limit,
    }
}

/// Create an Offset plan.
pub fn offset(input: LogicalPlan, offset: i32) -> LogicalPlan {
    LogicalPlan::Offset {
        input: Box::new(input),
        offset,
    }
}

/// Create a Tail plan.
pub fn tail(input: LogicalPlan, limit: i32) -> LogicalPlan {
    LogicalPlan::Tail {
        input: Box::new(input),
        limit,
    }
}

/// Create a Deduplicate plan.
pub fn deduplicate(
    input: LogicalPlan,
    all_columns_as_keys: bool,
    column_names: Vec<String>,
    within_watermark: bool,
) -> LogicalPlan {
    LogicalPlan::Deduplicate {
        input: Box::new(input),
        all_columns_as_keys,
        column_names,
        within_watermark,
    }
}

/// Create a Sort plan.
pub fn sort(input: LogicalPlan, order: Vec<Expression>, is_global: bool) -> LogicalPlan {
    LogicalPlan::Sort {
        input: Box::new(input),
        order,
        is_global,
    }
}

/// Create a Sample plan.
pub fn sample(
    input: LogicalPlan,
    lower_bound: f64,
    upper_bound: f64,
    with_replacement: bool,
    seed: Option<i64>,
) -> LogicalPlan {
    LogicalPlan::Sample {
        input: Box::new(input),
        lower_bound,
        upper_bound,
        with_replacement,
        seed,
    }
}

/// Create a Repartition plan.
pub fn repartition(input: LogicalPlan, num_partitions: i32, shuffle: bool) -> LogicalPlan {
    LogicalPlan::Repartition {
        input: Box::new(input),
        num_partitions,
        shuffle,
    }
}

/// Create a RepartitionByExpression plan.
pub fn repartition_by_expression(
    input: LogicalPlan,
    num_partitions: i32,
    expressions: Vec<Expression>,
) -> LogicalPlan {
    LogicalPlan::RepartitionByExpression {
        input: Box::new(input),
        num_partitions,
        expressions,
    }
}

/// Create a WithColumns plan.
pub fn with_columns(
    input: LogicalPlan,
    column_names: Vec<String>,
    columns: Vec<Column>,
) -> LogicalPlan {
    LogicalPlan::WithColumns {
        input: Box::new(input),
        column_names,
        columns,
    }
}

/// Create a WithColumnsRenamed plan.
pub fn with_columns_renamed(input: LogicalPlan, renames: HashMap<String, String>) -> LogicalPlan {
    LogicalPlan::WithColumnsRenamed {
        input: Box::new(input),
        renames,
    }
}

/// Create a Drop plan.
pub fn drop(input: LogicalPlan, columns: Vec<String>) -> LogicalPlan {
    LogicalPlan::Drop {
        input: Box::new(input),
        columns,
    }
}

/// Create a ToDF plan.
pub fn to_df(input: LogicalPlan, column_names: Vec<String>) -> LogicalPlan {
    LogicalPlan::ToDF {
        input: Box::new(input),
        column_names,
    }
}

/// Create a ToSchema plan.
pub fn to_schema(input: LogicalPlan, schema: DataType) -> LogicalPlan {
    LogicalPlan::ToSchema {
        input: Box::new(input),
        schema,
    }
}

/// Create a Hint plan.
pub fn hint(input: LogicalPlan, name: impl Into<String>, parameters: Vec<String>) -> LogicalPlan {
    LogicalPlan::Hint {
        input: Box::new(input),
        name: name.into(),
        parameters,
    }
}

/// Create an Unpivot plan.
pub fn unpivot(
    input: LogicalPlan,
    ids: Vec<Column>,
    values: Option<Vec<Column>>,
    variable_column_name: impl Into<String>,
    value_column_name: impl Into<String>,
) -> LogicalPlan {
    LogicalPlan::Unpivot {
        input: Box::new(input),
        ids,
        values,
        variable_column_name: variable_column_name.into(),
        value_column_name: value_column_name.into(),
    }
}

/// Create a NAFill plan.
pub fn na_fill(
    input: LogicalPlan,
    fill_value: crate::row::Value,
    columns: Vec<String>,
) -> LogicalPlan {
    LogicalPlan::NAFill {
        input: Box::new(input),
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

/// Build a proto `Literal` from a string: a numeric-looking string becomes a
/// Double literal, otherwise a String literal (used by NAReplace so string
/// replacements are not silently dropped).
fn str_to_proto_literal(s: &str) -> proto::expression::Literal {
    use proto::expression::literal::LiteralType;
    let mut lit = proto::expression::Literal::default();
    lit.literal_type = Some(match s.parse::<f64>() {
        Ok(v) => LiteralType::Double(v),
        Err(_) => LiteralType::String(s.to_string()),
    });
    lit
}

/// Create a NADrop plan.
pub fn na_drop(
    input: LogicalPlan,
    how: impl Into<String>,
    min_non_null: Option<i32>,
    columns: Vec<String>,
) -> LogicalPlan {
    LogicalPlan::NADrop {
        input: Box::new(input),
        how: how.into(),
        min_non_null,
        columns,
    }
}

/// Create a NAReplace plan.
pub fn na_replace(
    input: LogicalPlan,
    replacements: Vec<(String, String)>,
    columns: Vec<String>,
) -> LogicalPlan {
    LogicalPlan::NAReplace {
        input: Box::new(input),
        replacements,
        columns,
    }
}

/// Create a Describe plan.
pub fn describe(input: LogicalPlan, columns: Vec<String>) -> LogicalPlan {
    LogicalPlan::Describe {
        input: Box::new(input),
        columns,
    }
}

/// Create a Summary plan.
pub fn summary(input: LogicalPlan, percentiles: Vec<String>) -> LogicalPlan {
    LogicalPlan::Summary {
        input: Box::new(input),
        percentiles,
    }
}

/// Create a ColRegex plan.
pub fn col_regex(input: LogicalPlan, col_name: impl Into<String>) -> LogicalPlan {
    LogicalPlan::ColRegex {
        input: Box::new(input),
        col_name: col_name.into(),
    }
}

/// Create a SubqueryAlias plan.
pub fn subquery_alias(input: LogicalPlan, alias: impl Into<String>) -> LogicalPlan {
    LogicalPlan::SubqueryAlias {
        input: Box::new(input),
        alias: alias.into(),
    }
}

/// Create a LocalRelation plan.
pub fn local_relation(schema: DataType, data: Option<Vec<u8>>) -> LogicalPlan {
    LogicalPlan::LocalRelation { schema, data }
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
            input: Box::new(base()),
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
        let p = LogicalPlan::Hint {
            input: Box::new(base()),
            name: "REPARTITION".to_string(),
            parameters: vec!["10".to_string(), "name".to_string()],
        };
        match rel_type(p) {
            proto::relation::RelType::Hint(h) => {
                assert_eq!(h.parameters.len(), 2, "parameters must be forwarded");
                // First param "10" -> Long(10) literal.
                let lit = match h.parameters[0].expr_type.as_ref().unwrap() {
                    proto::expression::ExprType::Literal(l) => l.literal_type.clone().unwrap(),
                    _ => panic!("expected literal param"),
                };
                assert!(matches!(lit, LiteralType::Long(10)));
            }
            _ => panic!("expected Hint"),
        }
    }

    #[test]
    fn replace_sets_string_literals() {
        let p = LogicalPlan::NAReplace {
            input: Box::new(base()),
            replacements: vec![("foo".to_string(), "bar".to_string())],
            columns: vec![],
        };
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
                Expression::Literal(LiteralExpression::string("a")),
                Expression::Literal(LiteralExpression::string("b")),
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
            input: Box::new(base()),
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
