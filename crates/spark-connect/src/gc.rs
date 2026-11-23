//! The parts of plans and expressions that values share, as a binding of another language shows
//! them to its garbage collector. Not a stable API.
//!
//! A value of the core, such as a [`DataFrame`](crate::DataFrame) or a [`Column`](crate::Column),
//! holds its plan or its expression as `Arc`s of immutable nodes, which other values share: a
//! DataFrame derived from another holds the plan of that one, and a plan holds the expressions of
//! the Columns that built it. Some parts hold objects of the binding: the pickler of a UDF holds
//! its function, and the owner of a relation built outside the core holds the plan of the client.
//! A [`PartRef`] is one reference that a value or a part holds to a shared part, and
//! [`Part::held`] gives those that a part holds, each field once, so that the references that the
//! holders of a part give, with those that other code holds, add up to the strong count of the
//! part ([`PartRef::strong_count`]): a binding tells by it whether it knows of every reference to a
//! part.
//!
//! A binding may also have the core tell it of the clones of the parts that it makes
//! ([`check_clones_with`]), as the tests of one do to find a clone that runs where the binding does
//! not expect any.

use std::any::Any;
use std::sync::{Arc, OnceLock};

use crate::column::Column;
use crate::expression::{Expression, ExpressionNode, SubqueryExpression};
use crate::plan::{LogicalPlan, SharedPlan};
use crate::udf::{CommandPickler, CommonInlineUserDefinedFunctionExpression};

/// The check of the clones of the parts that a binding set, if any (see [`check_clones_with`]).
static CLONE_CHECK: OnceLock<fn(usize)> = OnceLock::new();

/// Have the core call `check` with the address of a part each time that it makes a new reference
/// to one, as the tests of a binding do to find a reference that the binding does not expect. The
/// core tells it of the clones of the handles of expressions, of picklers, of DataFrames, of owners,
/// of the inputs of plan nodes ([`SharedPlan`]) and of [`Part`]s; of the plans and the owners that
/// the plan of a new DataFrame holds, of which it was made, also below the nodes made for it alone;
/// of the plans of the subqueries of a relation that it serializes; and of the plans that a lookup
/// of a plan ID or of a placeholder finds. A clone of an `Arc` of a plan node that ends in none of
/// these, as of the plan of a DataFrame or of the input of an ML transform, is not told. Set once
/// for the process: false if a check is set already. Not a stable API.
#[doc(hidden)]
pub fn check_clones_with(check: fn(usize)) -> bool {
    CLONE_CHECK.set(check).is_ok()
}

/// Tell the check of the clones, if any, of a new reference to the part at `address`.
#[inline]
pub(crate) fn cloned(address: usize) {
    if let Some(check) = CLONE_CHECK.get() {
        check(address);
    }
}

/// Tell the check of the clones, if any, of the references to plans and to owners that `plan`, a
/// node that was just made, holds, which were cloned to make it: of the nodes and the expressions
/// below it that were made for it alone too, of no other reference, such as the subquery aliases of
/// the references of a statement.
pub(crate) fn cloned_into(plan: &LogicalPlan) {
    if CLONE_CHECK.get().is_none() {
        return;
    }
    let mut pending = Vec::new();
    plan_parts(plan, &mut pending);
    while let Some(reference) = pending.pop() {
        match reference {
            PartRef::Plan(node) | PartRef::Node(node, _) if Arc::strong_count(node) == 1 => {
                plan_parts(node, &mut pending)
            }
            PartRef::Expression(expression) if expression.node_strong_count() == 1 => {
                expression_parts(expression.node(), &mut pending)
            }
            PartRef::Plan(_) | PartRef::Node(..) | PartRef::Owner(_) => cloned(reference.address()),
            PartRef::Expression(_) | PartRef::Pickler(_) => {}
        }
    }
}

/// One reference to a shared part: a plan node, the node of an expression, the pickler of a UDF,
/// a UDTF or of another value that a request reads when it builds the plan, or an owner, of a
/// relation built outside the core, of the errors of an expression, or of an Observation of a
/// plan. Not a stable API.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum PartRef<'a> {
    /// A plan node, of whether an object of a binding is at or below it, which its fields tell.
    Plan(&'a Arc<LogicalPlan>),
    /// A plan node, of whether an object of a binding is at or below it as its holder keeps it:
    /// an input of a node ([`SharedPlan`]) or the plan of a DataFrame.
    Node(&'a Arc<LogicalPlan>, bool),
    Expression(&'a Expression),
    Pickler(&'a CommandPickler),
    Owner(&'a Arc<dyn Any + Send + Sync>),
}

/// A shared part, which keeps it alive (see [`PartRef`]). Not a stable API.
#[doc(hidden)]
pub enum Part {
    Plan(Arc<LogicalPlan>),
    Expression(Expression),
    Pickler(CommandPickler),
    Owner(Arc<dyn Any + Send + Sync>),
}

impl Clone for Part {
    fn clone(&self) -> Self {
        self.as_ref().to_part()
    }
}

impl<'a> PartRef<'a> {
    /// The address of the part, which all its references have.
    pub fn address(self) -> usize {
        match self {
            PartRef::Plan(plan) | PartRef::Node(plan, _) => Arc::as_ptr(plan) as usize,
            PartRef::Expression(expression) => expression.node_address(),
            PartRef::Pickler(pickler) => pickler.address(),
            PartRef::Owner(owner) => Arc::as_ptr(owner) as *const () as usize,
        }
    }

    /// The number of the references to the part, of its holders and of any other code.
    pub fn strong_count(self) -> usize {
        match self {
            PartRef::Plan(plan) | PartRef::Node(plan, _) => Arc::strong_count(plan),
            PartRef::Expression(expression) => expression.node_strong_count(),
            PartRef::Pickler(pickler) => pickler.strong_count(),
            PartRef::Owner(owner) => Arc::strong_count(owner),
        }
    }

    /// The part, which a reference more keeps alive.
    pub fn to_part(self) -> Part {
        match self {
            PartRef::Plan(plan) | PartRef::Node(plan, _) => {
                cloned(self.address());
                Part::Plan(Arc::clone(plan))
            }
            PartRef::Expression(expression) => Part::Expression(expression.clone()),
            PartRef::Pickler(pickler) => Part::Pickler(pickler.clone()),
            PartRef::Owner(owner) => {
                cloned(self.address());
                Part::Owner(Arc::clone(owner))
            }
        }
    }
}

impl Part {
    /// The reference of this handle to the part.
    pub fn as_ref(&self) -> PartRef<'_> {
        match self {
            Part::Plan(plan) => PartRef::Plan(plan),
            Part::Expression(expression) => PartRef::Expression(expression),
            Part::Pickler(pickler) => PartRef::Pickler(pickler),
            Part::Owner(owner) => PartRef::Owner(owner),
        }
    }

    /// The references that the part holds to other shared parts, each of its fields once: the
    /// inputs, the expressions, the picklers and the owners of a plan node; the expressions below
    /// the node of an expression, its pickler, the plans of its subqueries and the owner of its
    /// errors; the plans that the owner of the relation of a method of an ML object keeps, of the
    /// summary dataset of its target and of the DataFrames of its arguments (see
    /// `ml::owner_parts`). A pickler and the other owners hold none.
    pub fn held(&self) -> Vec<PartRef<'_>> {
        let mut held = Vec::new();
        match self {
            Part::Plan(plan) => plan_parts(plan, &mut held),
            Part::Expression(expression) => expression_parts(expression.node(), &mut held),
            Part::Owner(owner) => crate::ml::owner_parts(owner.as_ref(), &mut held),
            Part::Pickler(_) => {}
        }
        held
    }

    /// The object of a pickler or of an owner, which may be an object of the binding.
    pub fn object(&self) -> Option<&(dyn Any + Send + Sync)> {
        match self {
            Part::Pickler(pickler) => Some(pickler.object()),
            Part::Owner(owner) => Some(owner.as_ref()),
            Part::Plan(_) | Part::Expression(_) => None,
        }
    }
}

/// What a walk of the references that a part holds does with each ([`plan_parts`],
/// [`expression_parts`]): a list of them takes them, and [`Objects`] tells whether they reach an
/// object of a binding.
pub(crate) trait Held<'a> {
    /// A reference that the part holds.
    fn part(&mut self, part: PartRef<'a>);

    /// The input of a plan node, which is a reference to its node.
    fn input(&mut self, input: &'a SharedPlan) {
        self.part(PartRef::Node(input.arc(), input.holds_objects()));
    }

    /// The references `parts` that the part holds, in order.
    fn parts(&mut self, parts: impl IntoIterator<Item = PartRef<'a>>) {
        for part in parts {
            self.part(part);
        }
    }
}

impl<'a> Held<'a> for Vec<PartRef<'a>> {
    fn part(&mut self, part: PartRef<'a>) {
        self.push(part);
    }
}

/// Whether the references that a node holds reach an object of a binding: a pickler or an owner,
/// directly or below an input or an expression, each of which tells it of its own. A plan node
/// that a node holds as an `Arc`, of an ML transform and of a subquery, tells it by its fields,
/// which are walked in turn ([`Objects::of_plan`]).
#[derive(Default)]
struct Objects<'a> {
    found: bool,
    plans: Vec<&'a LogicalPlan>,
}

impl<'a> Held<'a> for Objects<'a> {
    fn part(&mut self, part: PartRef<'a>) {
        match part {
            PartRef::Plan(plan) => self.plans.push(plan),
            PartRef::Node(_, objects) => self.found |= objects,
            PartRef::Expression(expression) => self.found |= expression.holds_objects(),
            PartRef::Pickler(_) | PartRef::Owner(_) => self.found = true,
        }
    }
}

impl<'a> Objects<'a> {
    /// Whether the plan node `plan` holds an object of a binding, directly or below it.
    fn of_plan(plan: &'a LogicalPlan) -> bool {
        let mut objects = Objects::default();
        objects.plans.push(plan);
        objects.walk()
    }

    /// Whether the expression node `node` holds an object of a binding, directly or below it.
    fn of_expression(node: &'a ExpressionNode) -> bool {
        let mut objects = Objects::default();
        expression_parts(node, &mut objects);
        objects.walk()
    }

    /// Whether what was met, and the fields of the plan nodes met as `Arc`s, reach an object.
    fn walk(mut self) -> bool {
        while !self.found {
            let Some(plan) = self.plans.pop() else {
                break;
            };
            plan_parts(plan, &mut self);
        }
        self.found
    }
}

/// Whether the expression node `node` holds an object of a binding, directly or below it, of the
/// expressions and the inputs that it holds, each of which tells it: for the node of a new
/// expression.
pub(crate) fn expression_holds_objects(node: &ExpressionNode) -> bool {
    Objects::of_expression(node)
}

impl PartRef<'_> {
    /// Whether an object of a binding may be below the part, or the part be one: a pickler or an
    /// owner is (its object may be one of a binding), and a plan node or an expression is of any
    /// of them that it holds, directly or below it, which each node tells of the nodes that it
    /// holds as it is made. A part of none holds no part that a binding needs to see, so the
    /// binding needs not walk it. Not a stable API.
    pub fn holds_objects(self) -> bool {
        match self {
            PartRef::Plan(plan) => Objects::of_plan(plan),
            PartRef::Node(_, objects) => objects,
            PartRef::Expression(expression) => expression.holds_objects(),
            PartRef::Pickler(_) | PartRef::Owner(_) => true,
        }
    }
}

/// The references of `columns` to the nodes of their expressions.
fn columns<'a>(columns: &'a [Column], held: &mut impl Held<'a>) {
    held.parts(
        columns
            .iter()
            .map(|column| PartRef::Expression(column.expression())),
    );
}

/// The references of `expressions` to their nodes.
fn expressions<'a>(expressions: &'a [Expression], held: &mut impl Held<'a>) {
    held.parts(expressions.iter().map(PartRef::Expression));
}

/// The references of a UDF: its arguments and its pickler.
fn udf_parts<'a>(udf: &'a CommonInlineUserDefinedFunctionExpression, held: &mut impl Held<'a>) {
    expressions(&udf.arguments, held);
    held.parts(udf.python_udf.pickler.iter().map(PartRef::Pickler));
}

/// The references of a subquery: its plan, and its partitioning, ordering and values.
pub(crate) fn subquery_parts<'a>(subquery: &'a SubqueryExpression, held: &mut impl Held<'a>) {
    held.part(PartRef::Plan(&subquery.plan));
    expressions(&subquery.partition_spec, held);
    expressions(&subquery.order_spec, held);
    expressions(&subquery.in_subquery_values, held);
}

/// The references that the plan node `plan` holds, each of its fields once.
fn plan_parts<'a>(plan: &'a LogicalPlan, held: &mut impl Held<'a>) {
    use PartRef::{Expression as Of, Owner, Pickler, Plan};
    match plan {
        LogicalPlan::Range { .. }
        | LogicalPlan::LocalRelation { .. }
        | LogicalPlan::ChunkedCachedLocalRelation { .. }
        | LogicalPlan::CachedRelation { .. }
        | LogicalPlan::Read { .. }
        | LogicalPlan::RelationChanges { .. }
        | LogicalPlan::Catalog { .. } => {}
        LogicalPlan::Sql {
            pos_args,
            named_args,
            ..
        } => {
            expressions(pos_args, held);
            held.parts(named_args.values().map(Of));
        }
        LogicalPlan::Project { input, columns: of }
        | LogicalPlan::WithColumns {
            input, columns: of, ..
        } => {
            held.input(input);
            columns(of, held);
        }
        LogicalPlan::Filter { input, condition } => {
            held.input(input);
            held.part(Of(condition.expression()));
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
            held.input(input);
            expressions(grouping_expressions, held);
            expressions(aggregate_expressions, held);
            held.parts(pivot_col.iter().map(Of));
            expressions(pivot_values, held);
            for set in grouping_sets {
                expressions(set, held);
            }
        }
        LogicalPlan::Join {
            left, right, on, ..
        }
        | LogicalPlan::LateralJoin {
            left, right, on, ..
        } => {
            held.input(left);
            held.input(right);
            held.parts(on.iter().map(|on| Of(on.expression())));
        }
        LogicalPlan::SetOperation { left, right, .. } | LogicalPlan::Zip { left, right } => {
            held.input(left);
            held.input(right);
        }
        LogicalPlan::Raw {
            owner,
            observations,
            holes,
            ..
        } => {
            held.parts(owner.iter().map(|owner| Owner(&owner.0)));
            held.parts(
                observations
                    .iter()
                    .filter_map(|(_, observation)| observation.owner())
                    .map(Owner),
            );
            held.parts(
                holes
                    .0
                    .iter()
                    .filter_map(|hole| hole.pickler.as_ref())
                    .map(Pickler),
            );
        }
        LogicalPlan::Sort {
            input, order: of, ..
        }
        | LogicalPlan::RepartitionByExpression {
            input,
            expressions: of,
            ..
        }
        | LogicalPlan::Drop {
            input,
            expressions: of,
            ..
        }
        | LogicalPlan::Hint {
            input,
            parameters: of,
            ..
        }
        | LogicalPlan::RepartitionByRange {
            input,
            partition_exprs: of,
            ..
        }
        | LogicalPlan::Transpose {
            input,
            index_columns: of,
        } => {
            held.input(input);
            expressions(of, held);
        }
        LogicalPlan::WithColumnMetadata { input, column, .. } => {
            held.input(input);
            held.part(Of(column));
        }
        LogicalPlan::ToSchema {
            input,
            schema_reader,
            ..
        } => {
            held.input(input);
            held.parts(schema_reader.iter().map(Pickler));
        }
        LogicalPlan::Unpivot {
            input, ids, values, ..
        } => {
            held.input(input);
            columns(ids, held);
            if let Some(values) = values {
                columns(values, held);
            }
        }
        LogicalPlan::WithRelations { root, references } => {
            held.input(root);
            references
                .iter()
                .for_each(|reference| held.input(reference));
        }
        LogicalPlan::CachedRemoteRelation { owner, .. } => {
            held.parts(owner.iter().map(|owner| Owner(&owner.0)));
        }
        LogicalPlan::StatSampleBy {
            input,
            col,
            fractions,
            ..
        } => {
            held.input(input);
            held.part(Of(col));
            held.parts(fractions.iter().map(|(stratum, _)| Of(stratum)));
        }
        LogicalPlan::Observe {
            input,
            exprs,
            observation,
            ..
        } => {
            held.input(input);
            expressions(exprs, held);
            held.parts(observation.iter().filter_map(|o| o.owner()).map(Owner));
        }
        LogicalPlan::UnresolvedTableValuedFunction { arguments, .. } => {
            expressions(arguments, held);
        }
        LogicalPlan::MlTransform { input, owner, .. } => {
            held.part(Plan(input));
            held.parts(owner.iter().map(|owner| Owner(&owner.0)));
        }
        LogicalPlan::MapPartitions {
            input,
            func,
            profile_id_reader,
            ..
        } => {
            held.input(input);
            udf_parts(func, held);
            held.parts(profile_id_reader.iter().map(Pickler));
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
            held.input(input);
            expressions(grouping_expressions, held);
            udf_parts(func, held);
            expressions(sorting_expressions, held);
            initial_input.iter().for_each(|input| held.input(input));
            expressions(initial_grouping_expressions, held);
        }
        LogicalPlan::ApplyInPandasWithState {
            input,
            grouping_expressions,
            func,
            ..
        } => {
            held.input(input);
            expressions(grouping_expressions, held);
            udf_parts(func, held);
        }
        LogicalPlan::CoGroupMap {
            input,
            input_grouping_expressions,
            other,
            other_grouping_expressions,
            func,
        } => {
            held.input(input);
            held.input(other);
            expressions(input_grouping_expressions, held);
            expressions(other_grouping_expressions, held);
            udf_parts(func, held);
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
            held.input(left);
            held.input(right);
            held.parts([Of(left_as_of.expression()), Of(right_as_of.expression())]);
            held.parts(on.iter().chain(tolerance).map(|c| Of(c.expression())));
        }
        LogicalPlan::NearestByJoin {
            left,
            right,
            ranking_expression,
            ..
        } => {
            held.input(left);
            held.input(right);
            held.part(Of(ranking_expression));
        }
        LogicalPlan::CommonInlineUdtf {
            arguments, pickler, ..
        } => {
            expressions(arguments, held);
            held.parts(pickler.iter().map(Pickler));
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
        | LogicalPlan::StatCov { input, .. } => held.input(input),
    }
}

/// The references that the expression node `node` holds, each of its fields once.
pub(crate) fn expression_parts<'a>(node: &'a ExpressionNode, held: &mut impl Held<'a>) {
    use PartRef::{Expression as Of, Owner, Pickler, Plan};
    match node {
        ExpressionNode::Literal(_)
        | ExpressionNode::ColumnReference(_)
        | ExpressionNode::UnresolvedStar(..)
        | ExpressionNode::UnresolvedRegex(..)
        | ExpressionNode::SQLExpression(_)
        | ExpressionNode::UnresolvedNamedLambdaVariable(_) => {}
        ExpressionNode::UnresolvedFunction(function) => expressions(&function.args, held),
        ExpressionNode::Alias(alias) => held.part(Of(&alias.child)),
        ExpressionNode::Cast(cast) => {
            held.part(Of(&cast.child));
            if let crate::expression::CastTarget::Read(read) = &cast.target {
                held.parts(read.type_reader.iter().map(Pickler));
                held.parts(read.shown_reader.iter().map(Pickler));
            }
        }
        ExpressionNode::DirectShufflePartitionId(child) => held.part(Of(child)),
        ExpressionNode::SortOrder(sort) => held.part(Of(&sort.child)),
        ExpressionNode::CaseWhen(case_when) => {
            for (condition, value) in &case_when.branches {
                held.parts([Of(condition), Of(value)]);
            }
            held.parts(case_when.else_expr.iter().map(Of));
        }
        ExpressionNode::UnresolvedExtractValue(extract) => {
            held.parts([Of(&extract.child), Of(&extract.extraction)]);
        }
        ExpressionNode::UpdateFields(update) => {
            held.part(Of(&update.struct_expression));
            held.parts(update.value_expression.iter().map(Of));
        }
        ExpressionNode::CallFunction(call) => expressions(&call.arguments, held),
        ExpressionNode::WindowExpression(window) => {
            held.part(Of(&window.window_function));
            expressions(&window.partition_spec, held);
            held.parts(window.order_spec.iter().map(|sort| Of(&sort.child)));
        }
        ExpressionNode::LambdaFunction(lambda) => held.part(Of(&lambda.function)),
        ExpressionNode::CommonInlineUserDefinedFunction(udf) => udf_parts(udf, held),
        ExpressionNode::NamedArgument(argument) => held.part(Of(&argument.value)),
        ExpressionNode::Subquery(subquery) => subquery_parts(subquery, held),
        ExpressionNode::Sent(sent) => {
            held.parts(
                sent.holes
                    .0
                    .iter()
                    .filter_map(|hole| hole.pickler.as_ref())
                    .map(Pickler),
            );
            held.parts(sent.subqueries.plans().map(Plan));
        }
        ExpressionNode::Unbuildable(unbuildable) => {
            held.part(Of(&unbuildable.expression));
            held.parts(unbuildable.owner.iter().map(|owner| Owner(&owner.0)));
        }
        ExpressionNode::PlanBuildOrigin(made) => held.part(Of(&made.expression)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::dataframe::DataFrame;
    use crate::expression::ExpressionNode;
    use crate::functions::{col, max};
    use crate::plan::JoinType;
    use crate::types::DataType;
    use crate::udf::PythonUDFPayload;

    fn session() -> crate::session::SparkSession {
        crate::session::SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("a session")
    }

    /// A pickler whose object is `tag`.
    fn pickler(tag: &'static str) -> CommandPickler {
        CommandPickler::new(tag, |tag| Ok(tag.as_bytes().to_vec()))
    }

    /// A UDF of `id`, whose pickler's object is `tag`.
    fn udf(tag: &'static str) -> CommonInlineUserDefinedFunctionExpression {
        CommonInlineUserDefinedFunctionExpression::new(
            tag.to_string(),
            true,
            vec![Expression::column("id")],
            PythonUDFPayload::pickled_at_build(DataType::Long, 100, pickler(tag), "3.11".into()),
        )
    }

    /// A Column of `udf(tag)`.
    fn udf_column(tag: &'static str) -> Column {
        Column::new(ExpressionNode::CommonInlineUserDefinedFunction(Box::new(udf(tag))).into())
    }

    /// A relation built outside the core, of no owner, whose hole's pickler's object is `tag`, or
    /// of no hole.
    fn raw(tag: Option<&'static str>) -> Arc<LogicalPlan> {
        Arc::new(LogicalPlan::Raw {
            relation: Box::default(),
            owner: None,
            depth: 1,
            observations: Vec::new(),
            holes: crate::udf::SentHoles(
                tag.into_iter()
                    .map(|tag| crate::udf::SentHole {
                        marker: tag.as_bytes().to_vec(),
                        kind: crate::udf::SentHoleKind::PythonUdf,
                        pickler: Some(pickler(tag)),
                    })
                    .collect(),
            ),
        })
    }

    /// A Column of an expression as it was sent, as a pickled Column unpickles, whose subquery
    /// refers to `plan`.
    fn sent_over(plan: Arc<LogicalPlan>) -> Column {
        Column::new(
            ExpressionNode::Sent(Box::new(crate::expression::SentExpression {
                expression: spark_connect_proto::Expression::default(),
                rendered: String::new(),
                levels: 1,
                subqueries: crate::expression::SentSubqueries::of_plans(vec![plan]),
                holes: crate::udf::SentHoles(Vec::new()),
            }))
            .into(),
        )
    }

    /// A group map over `input` whose function has a command of its own, of no pickler, of the
    /// initial state `initial`.
    fn group_map(input: SharedPlan, initial: SharedPlan) -> LogicalPlan {
        LogicalPlan::GroupMap {
            input,
            grouping_expressions: vec![Expression::column("id")],
            func: Box::new(CommonInlineUserDefinedFunctionExpression::new(
                "g".to_string(),
                true,
                Vec::new(),
                PythonUDFPayload::new(DataType::Long, 200, vec![1], "3.11".into()),
            )),
            sorting_expressions: Vec::new(),
            initial_input: Some(initial),
            initial_grouping_expressions: vec![Expression::column("id")],
            is_map_groups_with_state: None,
            output_mode: None,
            timeout_conf: None,
            state_schema: None,
            transform_with_state_info: None,
        }
    }

    /// The tags of the objects of the picklers and of the owners of `&str` that `roots` reach.
    fn tags(roots: &[PartRef<'_>]) -> Vec<&'static str> {
        let mut tags: Vec<&'static str> = references(roots)
            .values()
            .filter_map(|(_, part)| part.object()?.downcast_ref::<&'static str>().copied())
            .collect();
        tags.sort();
        tags
    }

    /// The parts of plans of each kind of node, built by the API and as nodes, of UDFs, owners
    /// and Observations, all of which hold the same base plan and Columns, give every reference
    /// to them: no reference is given twice, and none is missed.
    #[test]
    fn the_parts_of_plans_of_every_kind_are_every_reference_to_them() {
        use crate::plan::LogicalPlan;
        let spark = session();
        let relation = spark_connect_proto::Relation {
            common: None,
            rel_type: Some(spark_connect_proto::relation::RelType::Sql(
                spark_connect_proto::Sql {
                    query: "SELECT 1 AS id".to_string(),
                    ..Default::default()
                },
            )),
        };
        let owned =
            DataFrame::from_relation(spark.clone(), relation, Some(Arc::new("owner"))).unwrap();
        let c = udf_column("c");
        let base = owned.select([c.clone().alias("v"), col("id")]);
        let other = spark
            .range(3)
            .unwrap()
            .select([udf_column("other"), col("id")]);
        let plan = |plan: LogicalPlan| DataFrame::new(spark.clone(), plan);
        let input = || Arc::clone(&base.plan);
        let observation = crate::observation::Observation::new("o").with_owner(Arc::new("obs"));
        let name = observation.name_to_observe().unwrap();
        let frames = vec![
            base.filter(c.clone().gt(crate::column::lit(0))),
            base.with_column("w", udf_column("with")),
            base.join(&other, Some(col("id").eq(col("id"))), JoinType::Inner),
            base.join(&base, None, JoinType::Cross),
            base.lateral_join(&other, Some(udf_column("lateral")), JoinType::Inner),
            base.union(&other),
            base.sort([udf_column("sort")]),
            base.group_by([c.clone()]).agg([max(udf_column("agg"))]),
            base.group_by([col("id")])
                .pivot(udf_column("pivot"), None)
                .unwrap()
                .agg([max(col("v"))]),
            base.limit(1).offset(1).distinct(),
            base.sample(0.5, Some(1)).coalesce(2).alias("a"),
            base.repartition_by_range(2, [udf_column("range")]).unwrap(),
            base.hint("h", ["p"]),
            base.unpivot([col("id")], Some([udf_column("unpivot")]), "k", "x"),
            base.observed_with(
                &observation,
                name,
                vec![udf_column("observed").expression().clone()],
            ),
            base.with_watermark("t", "1 second").describe(["v"]),
            base.map_in_pandas(udf("map"), false),
            base.group_by([col("id")]).apply_in_pandas(udf("group map")),
            base.group_by([col("id")])
                .cogroup(&other.group_by([col("id")]))
                .apply_in_pandas(udf("cogroup")),
            base.select([other.select([max(col("id"))]).scalar(), other.exists()]),
            base.select([
                col("id").isin_subquery(&other),
                c.clone().isin([udf_column("isin")]),
            ]),
            base.transpose().unwrap(),
            base.zip(&other).unwrap(),
            base.to(DataType::Long),
            plan(LogicalPlan::Sql {
                query: "SELECT ?".to_string(),
                pos_args: vec![udf_column("sql").expression().clone()],
                named_args: HashMap::from([(
                    "a".to_string(),
                    udf_column("named").expression().clone(),
                )]),
            }),
            plan(LogicalPlan::WithRelations {
                root: input().into(),
                references: vec![input().into(), other.shared_plan()],
            }),
            plan(LogicalPlan::ToSchema {
                input: input().into(),
                schema: DataType::Long,
                schema_reader: Some(pickler("schema")),
            }),
            plan(LogicalPlan::StatSampleBy {
                input: input().into(),
                col: udf_column("sample by").expression().clone(),
                fractions: vec![(udf_column("stratum").expression().clone(), 0.5)],
                seed: None,
            }),
            plan(LogicalPlan::CommonInlineUdtf {
                function_name: "t".to_string(),
                deterministic: true,
                arguments: vec![udf_column("udtf argument").expression().clone()],
                return_type: None,
                eval_type: 300,
                command: Vec::new(),
                python_ver: "3.11".to_string(),
                pickler: Some(pickler("udtf")),
            }),
            plan(LogicalPlan::MlTransform {
                ml_relation: Box::default(),
                input: input(),
                owner: Some(crate::plan::RelationOwner(Arc::new("ml"))),
            }),
            plan(LogicalPlan::CachedRemoteRelation {
                relation_id: "r".to_string(),
                owner: Some(crate::plan::RelationOwner(Arc::new("cached"))),
            }),
            plan(LogicalPlan::Raw {
                relation: Box::default(),
                owner: Some(crate::plan::RelationOwner(Arc::new("raw"))),
                depth: 1,
                observations: vec![(
                    "r".to_string(),
                    crate::observation::Observation::new("r").with_owner(Arc::new("raw obs")),
                )],
                holes: crate::udf::SentHoles(vec![crate::udf::SentHole {
                    marker: b"m".to_vec(),
                    kind: crate::udf::SentHoleKind::PythonUdf,
                    pickler: Some(pickler("hole")),
                }]),
            }),
            base.select([sent_over(raw(Some("sent subquery")))]),
            plan(group_map(
                base.shared_plan(),
                base.select([udf_column("initial")]).shared_plan(),
            )),
        ];
        drop(observation);
        let mut roots = vec![PartRef::Expression(c.expression())];
        roots.extend(base.held_parts());
        roots.extend(other.held_parts());
        roots.extend(owned.held_parts());
        for frame in &frames {
            roots.extend(frame.held_parts());
        }
        assert_exact(&roots);
        assert_objects(&roots);
        let found = tags(&roots);
        for tag in [
            "c",
            "other",
            "owner",
            "with",
            "lateral",
            "sort",
            "agg",
            "pivot",
            "range",
            "unpivot",
            "observed",
            "obs",
            "map",
            "group map",
            "cogroup",
            "isin",
            "sql",
            "named",
            "schema",
            "sample by",
            "stratum",
            "udtf argument",
            "udtf",
            "ml",
            "cached",
            "raw",
            "raw obs",
            "hole",
            "sent subquery",
            "initial",
        ] {
            assert!(found.contains(&tag), "{tag} is not reached: {found:?}");
        }
    }

    /// Plans and expressions of no pickler and no owner below them hold no object of a binding,
    /// as their nodes tell, and those of one do: of their inputs, their expressions, the plans of
    /// their subqueries and of their ML transforms, which hold their inputs as `Arc`s.
    #[test]
    fn a_part_holds_an_object_of_a_binding_when_a_pickler_or_an_owner_is_below_it() {
        let spark = session();
        let base = spark.range(3).unwrap();
        let plan = |plan: LogicalPlan| DataFrame::new(spark.clone(), plan);
        let ml = |input: &DataFrame, owner: Option<crate::plan::RelationOwner>| {
            plan(LogicalPlan::MlTransform {
                ml_relation: Box::default(),
                input: Arc::clone(&input.plan),
                owner,
            })
        };
        let plain = vec![
            base.filter(col("id")),
            base.join(&base, Some(col("id").eq(col("id"))), JoinType::Inner),
            base.select([base.select([max(col("id"))]).scalar(), base.exists()]),
            base.group_by([col("id")]).agg([max(col("id"))]),
            base.union(&base).limit(1).alias("a"),
            ml(&ml(&base.filter(col("id")), None), None),
            base.select([sent_over(raw(None))]),
            plan(group_map(
                base.shared_plan(),
                base.filter(col("id")).shared_plan(),
            )),
        ];
        let of_udf = base.select([udf_column("u"), col("id")]);
        let owned = crate::plan::RelationOwner(Arc::new("owner"));
        let with = vec![
            of_udf.filter(col("id")).limit(2),
            base.join(&of_udf, None, JoinType::Cross),
            base.select([of_udf.select([max(col("id"))]).scalar()]),
            base.filter(col("id").isin_subquery(&of_udf)),
            ml(&ml(&of_udf, None), None).limit(1),
            ml(&base, Some(owned)).filter(col("id")),
            base.map_in_pandas(udf("map"), false),
            base.select([sent_over(raw(Some("sent")))]),
            plan(group_map(base.shared_plan(), of_udf.shared_plan())),
        ];
        for frame in &plain {
            assert!(
                !PartRef::Plan(&frame.plan).holds_objects(),
                "{:?}",
                frame.plan
            );
        }
        for frame in &with {
            assert!(
                PartRef::Plan(&frame.plan).holds_objects(),
                "{:?}",
                frame.plan
            );
        }
        assert!(
            !PartRef::Expression(col("id").eq(crate::column::lit(1)).expression()).holds_objects()
        );
        assert!(PartRef::Expression(udf_column("e").alias("a").expression()).holds_objects());
        let roots: Vec<PartRef<'_>> = plain
            .iter()
            .chain(&with)
            .flat_map(|frame| frame.held_parts())
            .collect();
        assert_objects(&roots);
    }

    /// Assert that whether each part that `roots` reach holds an object of a binding, as its node
    /// tells, is whether a pickler or an owner is at or below it.
    #[track_caller]
    pub(crate) fn assert_objects(roots: &[PartRef<'_>]) {
        fn below(part: &Part, memo: &mut HashMap<usize, bool>) -> bool {
            let address = part.as_ref().address();
            if let Some(found) = memo.get(&address) {
                return *found;
            }
            let found = match part {
                Part::Pickler(_) | Part::Owner(_) => true,
                Part::Plan(_) | Part::Expression(_) => part
                    .held()
                    .into_iter()
                    .any(|held| below(&held.to_part(), memo)),
            };
            memo.insert(address, found);
            found
        }
        let mut memo = HashMap::new();
        for root in roots {
            // A plan node of the bit that its holder keeps: the bit of its fields.
            assert_eq!(
                root.holds_objects(),
                root.to_part().as_ref().holds_objects(),
                "the root at {:#x}",
                root.address()
            );
        }
        for (address, (_, part)) in references(roots) {
            assert_eq!(
                part.as_ref().holds_objects(),
                below(&part, &mut memo),
                "the part at {address:#x}"
            );
        }
    }

    /// The references to each part reachable from `roots`, of `roots` and of the parts that they
    /// reach, by the address of the part, with a handle of the part.
    pub(crate) fn references(roots: &[PartRef<'_>]) -> HashMap<usize, (usize, Part)> {
        let mut counts: HashMap<usize, (usize, Part)> = HashMap::new();
        let mut pending: Vec<Part> = Vec::new();
        for root in roots {
            let entry = counts
                .entry(root.address())
                .or_insert_with(|| (0, root.to_part()));
            entry.0 += 1;
            if entry.0 == 1 {
                pending.push(root.to_part());
            }
        }
        while let Some(part) = pending.pop() {
            for held in part.held() {
                let entry = counts
                    .entry(held.address())
                    .or_insert_with(|| (0, held.to_part()));
                entry.0 += 1;
                if entry.0 == 1 {
                    pending.push(held.to_part());
                }
            }
        }
        counts
    }

    /// The references to the parts that `roots` reach that neither `roots` nor those parts hold:
    /// of the holders that the walk does not know of.
    pub(crate) fn outside_references(roots: &[PartRef<'_>]) -> usize {
        references(roots)
            .values()
            .map(|(count, part)| part.as_ref().strong_count() - 1 - count)
            .sum()
    }

    thread_local! {
        /// The addresses that the check of the clones was told of on this thread, while a test of
        /// this thread records them (see [`told`]).
        static TOLD: std::cell::RefCell<Option<Vec<usize>>> = const { std::cell::RefCell::new(None) };
    }

    /// The check of the clones of the tests: it records the addresses of the thread that records.
    fn tell(address: usize) {
        let _ = TOLD.try_with(|told| {
            if let Some(told) = told.borrow_mut().as_mut() {
                told.push(address);
            }
        });
    }

    /// The addresses that the check of the clones was told of on this thread since the last call,
    /// which records them from the first.
    fn told() -> Vec<usize> {
        TOLD.with(|told| std::mem::take(told.borrow_mut().get_or_insert_with(Vec::new)))
    }

    /// Each new reference to a part that the core makes tells the check of the clones of its
    /// address first: of the clone of a handle of an expression, a pickler, a DataFrame, an owner,
    /// an Observation of one and a part, of the input of a plan node, also as a node is cloned, of
    /// the plans of the node of a new DataFrame, which were cloned to make it, and of a lookup of a
    /// plan ID.
    #[test]
    fn the_check_of_the_clones_is_told_of_each_new_reference_to_a_part() {
        // The one check of the tests of this crate.
        let _ = check_clones_with(tell);
        let _ = told();
        let spark = session();
        let c = udf_column("c");
        let expression = c.expression();
        drop(expression.clone());
        assert_eq!(told(), vec![PartRef::Expression(expression).address()]);
        let pickler = pickler("p");
        drop(pickler.clone());
        assert_eq!(told(), vec![PartRef::Pickler(&pickler).address()]);
        let base = spark.range(3).unwrap();
        let root = PartRef::Plan(&base.plan).address();
        let _ = told();
        drop(base.clone());
        assert_eq!(told(), vec![root]);
        let derived = base.filter(col("id"));
        assert_eq!(told(), vec![root]);
        // Of a clone of the input of a plan node, and of the inputs of a clone of a node.
        let LogicalPlan::Filter { input, .. } = derived.plan.as_ref() else {
            unreachable!("a filter")
        };
        drop(input.clone());
        assert_eq!(told(), vec![root]);
        let node: LogicalPlan = derived.plan.as_ref().clone();
        let condition = match &node {
            LogicalPlan::Filter { condition, .. } => PartRef::Expression(condition.expression()),
            _ => unreachable!("a filter"),
        };
        assert_eq!(told(), vec![root, condition.address()]);
        drop(node);
        // Of a plan of a node made for it alone over the shared plan: the shared plan, below it.
        let nested = DataFrame::new(
            spark.clone(),
            LogicalPlan::Limit {
                input: SharedPlan::new(LogicalPlan::Limit {
                    input: base.shared_plan(),
                    limit: 2,
                }),
                limit: 1,
            },
        );
        assert_eq!(told(), vec![root]);
        drop(nested);
        let owner = crate::plan::RelationOwner(Arc::new("owner"));
        drop(owner.clone());
        assert_eq!(told(), vec![PartRef::Owner(&owner.0).address()]);
        let observation = crate::observation::Observation::new("o").with_owner(Arc::new("o"));
        drop(observation.clone());
        let owned = PartRef::Owner(observation.owner().unwrap()).address();
        assert_eq!(told(), vec![owned]);
        let part = PartRef::Plan(&derived.plan).to_part();
        drop(part.clone());
        let of_derived = PartRef::Plan(&derived.plan).address();
        assert_eq!(told(), vec![of_derived, of_derived]);
        let id = 0x7e57_c10e;
        spark.plan_ids().assign(&derived.plan, id);
        let found = spark
            .plan_ids()
            .node_of(id)
            .expect("the node of the plan ID");
        assert_eq!(told(), vec![of_derived]);
        drop(found);
        TOLD.with(|told| told.borrow_mut().take());
    }

    /// Assert that the references that `roots` and the parts below them hold are every reference
    /// to each of those parts: the strong count of each, less the handles of this check, which
    /// hold one each in `references` and one each of `pending`, of a part that it walked.
    #[track_caller]
    pub(crate) fn assert_exact(roots: &[PartRef<'_>]) {
        let counts = references(roots);
        assert!(!counts.is_empty());
        for (address, (count, part)) in &counts {
            // The handle of the map holds one reference.
            let others = part.as_ref().strong_count() - 1;
            let kind = match part {
                Part::Plan(plan) => format!("plan {plan:?}"),
                Part::Expression(expression) => format!("expression {expression:?}"),
                Part::Pickler(_) => "pickler".to_string(),
                Part::Owner(_) => "owner".to_string(),
            };
            assert_eq!(
                others, *count,
                "the part at {address:#x} has {others} references, of which the walk gave \
                 {count}: {kind:.300}"
            );
        }
    }
}
