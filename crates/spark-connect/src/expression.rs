//! Expression tree mirroring PySpark's `pyspark.sql.connect.expressions`.
//!
//! Defines the Expression hierarchy and conversion functions to Spark Connect protobufs.

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use spark_connect_core::error::{Result, SparkError};
use spark_connect_proto as proto;

use crate::plan::LogicalPlan;
use crate::types::DataType;
use crate::udf::CommonInlineUserDefinedFunctionExpression;

/// Thread-local counter for generating unique lambda variable names.
static LAMBDA_VAR_COUNTER: AtomicU32 = AtomicU32::new(0);

/// Get the next unique lambda variable suffix.
pub fn next_lambda_var_index() -> u32 {
    LAMBDA_VAR_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// The most levels of an expression that a plan of a request may carry. It protects this client,
/// as [`crate::plan::MAX_PLAN_DEPTH`] does for plans: a deeper expression is a `RecursionError`
/// here, before any request is built (see [`with_expression_depth_bound`]). PySpark's own point is
/// not this number, and depends on its protobuf backend: it builds the proto of an expression by
/// recursing in Python, so with the pure-Python backend its serializer raises `RecursionError` from
/// about 244 additions, as in `select((col("id") + 1 + ... + 1).alias("x"))`, and with upb it
/// raises a `DecodeError` from 48 additions in some paths (measured with pyspark-client 4.3.0).
pub const MAX_EXPRESSION_DEPTH: usize = 745;

/// The stack that serializing or rendering an expression keeps free before it goes down a level,
/// and the size of a stack segment that it adds when less is left: a level recurses once, so an
/// expression of any number of levels grows the stack instead of overflowing it. A frame of a
/// debug build is many times larger than one of a release build.
const EXPRESSION_RED_ZONE: usize = if cfg!(debug_assertions) {
    1 << 20
} else {
    128 << 10
};
const EXPRESSION_STACK_SEGMENT: usize = if cfg!(debug_assertions) {
    16 << 20
} else {
    4 << 20
};

thread_local! {
    /// The level of the expression that this thread serializes now, and the deepest one since
    /// the start of the innermost [`with_expression_depth_bound`].
    static EXPRESSION_LEVEL: Cell<usize> = const { Cell::new(0) };
    static DEEPEST_EXPRESSION_LEVEL: Cell<usize> = const { Cell::new(0) };
    /// Whether this thread serializes in a [`with_expression_depth_bound`], which stops at the
    /// bound.
    static EXPRESSION_DEPTH_BOUNDED: Cell<bool> = const { Cell::new(false) };
}

/// One level of an expression that is being serialized, left when it is dropped (also on a
/// panic, so that the level of the thread stays right).
struct ExpressionLevel(usize);

impl ExpressionLevel {
    fn enter() -> Self {
        let level = EXPRESSION_LEVEL.with(|current| {
            let level = current.get() + 1;
            current.set(level);
            level
        });
        DEEPEST_EXPRESSION_LEVEL.with(|deepest| deepest.set(deepest.get().max(level)));
        ExpressionLevel(level)
    }
}

impl Drop for ExpressionLevel {
    fn drop(&mut self) {
        EXPRESSION_LEVEL.with(|current| current.set(current.get() - 1));
    }
}

/// The [`with_expression_depth_bound`] of this thread, until it is dropped: the bound and the
/// deepest level of the serialization that encloses it, if any, are restored.
struct DepthBoundScope {
    outer_bounded: bool,
    outer_deepest: usize,
}

impl DepthBoundScope {
    fn enter() -> Self {
        DepthBoundScope {
            outer_bounded: EXPRESSION_DEPTH_BOUNDED.with(|bounded| bounded.replace(true)),
            outer_deepest: DEEPEST_EXPRESSION_LEVEL.with(|deepest| deepest.replace(0)),
        }
    }

    /// The deepest level since the scope was entered.
    fn deepest(&self) -> usize {
        DEEPEST_EXPRESSION_LEVEL.with(Cell::get)
    }
}

impl Drop for DepthBoundScope {
    fn drop(&mut self) {
        EXPRESSION_DEPTH_BOUNDED.with(|bounded| bounded.set(self.outer_bounded));
        DEEPEST_EXPRESSION_LEVEL.with(|deepest| deepest.set(deepest.get().max(self.outer_deepest)));
    }
}

/// Run `f`, which goes down a level of an expression or of a plan, with at least
/// [`EXPRESSION_RED_ZONE`] of stack left.
pub(crate) fn with_stack_for_a_level<T>(f: impl FnOnce() -> T) -> T {
    stacker::maybe_grow(EXPRESSION_RED_ZONE, EXPRESSION_STACK_SEGMENT, f)
}

/// Run `serialize`, which serializes expressions for a request, and fail as PySpark's serializer
/// does if one of them is deeper than [`MAX_EXPRESSION_DEPTH`]: with a `RecursionError`, instead
/// of a request that overflows the stack or that the server cannot parse. Within it, an
/// expression is serialized down to the bound only: its proto is dropped and encoded by
/// recursing too, so it must not be deeper.
pub(crate) fn with_expression_depth_bound<T>(serialize: impl FnOnce() -> T) -> Result<T> {
    let scope = DepthBoundScope::enter();
    let value = serialize();
    if scope.deepest() > MAX_EXPRESSION_DEPTH {
        return Err(SparkError::recursion_msg(format!(
            "maximum recursion depth exceeded while serializing an expression; an expression \
             may be at most {MAX_EXPRESSION_DEPTH} levels deep"
        )));
    }
    Ok(value)
}

/// An immutable, shared expression, mirroring PySpark's base Expression type.
///
/// Cloning a handle shares its node rather than copying the expression subtree.
/// Build common expressions with [`Expression::literal`], [`Expression::column`],
/// and [`Expression::function`], or use [`Expression::new`] for other node types.
/// Inspect expressions with [`Expression::node`]. An expression keeps the origin in user code
/// of its construction, if any, as PySpark's does (see [`crate::origin`]).
#[derive(Clone)]
pub struct Expression {
    node: Arc<ExpressionNode>,
    origin: Option<Arc<crate::origin::Origin>>,
}

impl std::fmt::Debug for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        with_stack_for_a_level(|| std::fmt::Debug::fmt(self.node(), f))
    }
}

impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        // Compare values even for the same allocation: NaN must remain unequal to itself.
        with_stack_for_a_level(|| self.node().eq(other.node()))
    }
}

/// The node that a dropped expression holds once its own node has gone, which has no levels
/// below it.
static DROPPED_NODE: std::sync::LazyLock<Arc<ExpressionNode>> =
    std::sync::LazyLock::new(|| Arc::new(ExpressionNode::SQLExpression(String::new())));

impl Drop for Expression {
    /// The last owner of a node drops the levels below it, each in turn: give them the stack
    /// that an expression of many levels needs, as for serializing it.
    fn drop(&mut self) {
        if Arc::strong_count(&self.node) == 1 {
            let node = std::mem::replace(&mut self.node, Arc::clone(&DROPPED_NODE));
            with_stack_for_a_level(move || drop(node));
        }
    }
}

impl From<ExpressionNode> for Expression {
    fn from(node: ExpressionNode) -> Self {
        Self::new(node)
    }
}

/// The contents of one expression node.
///
/// Child expressions are shared [`Expression`] handles. Direct enum construction
/// and matching use this type; the public Column API continues to use Expression.
#[derive(Debug, Clone, PartialEq)]
pub enum ExpressionNode {
    /// `pyspark.sql.connect.expressions.LiteralExpression` - a constant value.
    Literal(LiteralExpression),
    /// `pyspark.sql.connect.expressions.ColumnReference` - a reference to a column.
    ColumnReference(ColumnReference),
    /// `pyspark.sql.connect.expressions.UnresolvedFunction` - a function call.
    UnresolvedFunction(UnresolvedFunction),
    /// `pyspark.sql.connect.expressions.UnresolvedStar` - a star expression (`*`).
    /// Carries an optional target ending in `.*` (e.g. `df.*` from `col("df.*")`), and the plan
    /// ID of the DataFrame of `df["*"]`, whose columns it is.
    UnresolvedStar(Option<String>, Option<i64>),
    /// `pyspark.sql.connect.expressions.ColumnAlias` / `Alias` - an aliased expression.
    Alias(Box<Alias>),
    /// `pyspark.sql.connect.expressions.CastExpression` - a cast expression.
    Cast(Box<Cast>),
    /// `spark.connect.Expression.DirectShufflePartitionID` - wraps a child expression
    /// that evaluates to a partition id (used by `DataFrame.repartitionById`).
    DirectShufflePartitionId(Expression),
    /// `pyspark.sql.connect.expressions.UnresolvedRegex` - a regex column reference, of the
    /// columns of the DataFrame of the plan ID of `df.colRegex(...)`.
    UnresolvedRegex(String, Option<i64>),
    /// `pyspark.sql.connect.expressions.SortOrder` - a sort order expression.
    SortOrder(Box<SortOrder>),
    /// `pyspark.sql.connect.expressions.CaseWhen` - a CASE WHEN expression.
    CaseWhen(Box<CaseWhen>),
    /// `pyspark.sql.connect.expressions.UnresolvedExtractValue` - `col[k]` / `getField`.
    UnresolvedExtractValue(Box<ExtractValue>),
    /// `pyspark.sql.connect.expressions.UpdateFields` - `withField` / `dropFields`.
    UpdateFields(Box<UpdateFieldsExpr>),
    /// `pyspark.sql.connect.expressions.SQLExpression` - a raw SQL expression.
    SQLExpression(String),
    /// `pyspark.sql.connect.expressions.CallFunction` - a direct function call.
    CallFunction(Box<CallFunctionWrapper>),
    /// `pyspark.sql.connect.expressions.WindowExpression` - a window function call.
    WindowExpression(Box<WindowExpressionWrapper>),
    /// `pyspark.sql.connect.expressions.LambdaFunction` - a lambda function.
    LambdaFunction(Box<LambdaFunction>),
    /// `pyspark.sql.connect.expressions.UnresolvedNamedLambdaVariable` - a lambda variable.
    UnresolvedNamedLambdaVariable(UnresolvedNamedLambdaVariable),
    /// `pyspark.sql.connect.expressions.CommonInlineUserDefinedFunction` - a UDF expression.
    CommonInlineUserDefinedFunction(Box<CommonInlineUserDefinedFunctionExpression>),
    /// `pyspark.sql.connect.expressions.NamedArgumentExpression` - a named argument of a
    /// function call, such as `a => 1` of a Python UDF called with `a=lit(1)`.
    NamedArgument(Box<NamedArgument>),
    /// `pyspark.sql.connect.expressions.SubqueryExpression` - a scalar, EXISTS or IN
    /// subquery of a DataFrame, or a table argument of a table-valued function.
    Subquery(Box<SubqueryExpression>),
    /// An expression as it was sent, of its `spark.connect.Expression` and its rendering, as a
    /// pickled Column unpickles: the client sends the expression as it is.
    Sent(Box<SentExpression>),
}

/// An expression of [`ExpressionNode::Sent`]: the `spark.connect.Expression` of a Column as it
/// was sent, and its rendering, as `Column<'...'>` shows it.
#[derive(Debug, Clone, PartialEq)]
pub struct SentExpression {
    pub expression: proto::Expression,
    pub rendered: String,
}

/// The type of a [`SubqueryExpression`]. Mirrors the `subquery_type` of
/// `pyspark.sql.connect.expressions.SubqueryExpression`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubqueryType {
    /// A scalar subquery, of `DataFrame.scalar()`.
    Scalar,
    /// An EXISTS subquery, of `DataFrame.exists()`.
    Exists,
    /// A table argument, of `DataFrame.asTable()`.
    TableArg,
    /// An IN subquery, of `Column.isin(df)`.
    In,
}

impl SubqueryType {
    fn name(self) -> &'static str {
        match self {
            SubqueryType::Scalar => "scalar",
            SubqueryType::Exists => "exists",
            SubqueryType::TableArg => "table_arg",
            SubqueryType::In => "in",
        }
    }
}

/// A subquery of the plan of a DataFrame. Mirrors
/// `pyspark.sql.connect.expressions.SubqueryExpression`: it refers to the plan by its plan
/// ID, and the relation that contains it sends the plan in a `WithRelations`, as PySpark's
/// `LogicalPlan._with_relations` does.
#[derive(Debug, Clone)]
pub struct SubqueryExpression {
    /// The plan of the subquery.
    pub(crate) plan: Arc<LogicalPlan>,
    /// The plan ID of `plan` in the session of its DataFrame.
    pub(crate) plan_id: i64,
    /// The type of the subquery.
    pub subquery_type: SubqueryType,
    /// The partitioning of a table argument.
    pub partition_spec: Vec<Expression>,
    /// The ordering of a table argument, within its partitions.
    pub order_spec: Vec<Expression>,
    /// Whether a table argument is a single partition, when set.
    pub with_single_partition: Option<bool>,
    /// The values of an IN subquery.
    pub in_subquery_values: Vec<Expression>,
}

impl PartialEq for SubqueryExpression {
    fn eq(&self, other: &Self) -> bool {
        // Plans have no equality: the subqueries of the same plan are equal.
        Arc::ptr_eq(&self.plan, &other.plan)
            && self.plan_id == other.plan_id
            && self.subquery_type == other.subquery_type
            && self.partition_spec == other.partition_spec
            && self.order_spec == other.order_spec
            && self.with_single_partition == other.with_single_partition
            && self.in_subquery_values == other.in_subquery_values
    }
}

impl SubqueryExpression {
    /// A subquery of `subquery_type` of `plan`, whose plan ID is `plan_id`.
    pub(crate) fn new(plan: Arc<LogicalPlan>, plan_id: i64, subquery_type: SubqueryType) -> Self {
        SubqueryExpression {
            plan,
            plan_id,
            subquery_type,
            partition_spec: Vec::new(),
            order_spec: Vec::new(),
            with_single_partition: None,
            in_subquery_values: Vec::new(),
        }
    }

    /// As `SubqueryExpression.__repr__`, but with the plan ID of the plan, which PySpark
    /// shows as the address of its plan object.
    fn render(&self) -> String {
        let mut parts = vec![
            format!("plan=<plan {}>", self.plan_id),
            format!("type={}", self.subquery_type.name()),
        ];
        let list = |exprs: &[Expression]| {
            let rendered: Vec<String> = exprs.iter().map(Expression::render).collect();
            format!("[{}]", rendered.join(", "))
        };
        match self.subquery_type {
            SubqueryType::TableArg => {
                if !self.partition_spec.is_empty() {
                    parts.push(format!("partition_spec={}", list(&self.partition_spec)));
                }
                if !self.order_spec.is_empty() {
                    parts.push(format!("order_spec={}", list(&self.order_spec)));
                }
                if let Some(single) = self.with_single_partition {
                    let single = if single { "True" } else { "False" };
                    parts.push(format!("with_single_partition={single}"));
                }
            }
            SubqueryType::In => parts.push(format!("values={}", list(&self.in_subquery_values))),
            SubqueryType::Scalar | SubqueryType::Exists => {}
        }
        format!("SubqueryExpression({})", parts.join(", "))
    }

    /// As `SubqueryExpression.to_plan`. The relation being serialized sends `plan` in its
    /// `WithRelations`.
    fn to_proto(&self) -> proto::Expression {
        use proto::subquery_expression::SubqueryType as ProtoType;

        crate::plan::refer_to(&self.plan);
        let subquery_type = match self.subquery_type {
            SubqueryType::Scalar => ProtoType::Scalar,
            SubqueryType::Exists => ProtoType::Exists,
            SubqueryType::TableArg => ProtoType::TableArg,
            SubqueryType::In => ProtoType::In,
        } as i32;
        // As PySpark, which sets the options of a table argument only when one is given.
        let table_arg_options = (self.subquery_type == SubqueryType::TableArg
            && (!self.partition_spec.is_empty()
                || !self.order_spec.is_empty()
                || self.with_single_partition.is_some()))
        .then(|| proto::subquery_expression::TableArgOptions {
            partition_spec: self
                .partition_spec
                .iter()
                .map(Expression::to_proto)
                .collect(),
            order_spec: self.order_spec.iter().map(sort_order_proto).collect(),
            with_single_partition: self.with_single_partition,
        });
        let in_subquery_values = match self.subquery_type {
            SubqueryType::In => self
                .in_subquery_values
                .iter()
                .map(Expression::to_proto)
                .collect(),
            _ => Vec::new(),
        };
        let subquery = proto::SubqueryExpression {
            plan_id: self.plan_id,
            subquery_type,
            table_arg_options,
            in_subquery_values,
        };
        proto::Expression {
            expr_type: Some(proto::expression::ExprType::SubqueryExpression(subquery)),
            ..Default::default()
        }
    }
}

/// The sort order of an ordering expression of a table argument: an explicit sort order
/// as it is, and another expression ascending, nulls first, as PySpark's `_sort_col`.
fn sort_order_proto(expr: &Expression) -> proto::expression::SortOrder {
    let wrapped = match expr.node() {
        ExpressionNode::SortOrder(_) => expr.to_proto(),
        _ => SortOrder::asc_nulls_first(expr.clone()).to_proto(),
    };
    match wrapped.expr_type {
        Some(proto::expression::ExprType::SortOrder(sort)) => *sort,
        // SortOrder::to_proto always yields a SortOrder expr_type.
        _ => unreachable!("expected a SortOrder expression"),
    }
}

/// A named argument of a function call. Mirrors
/// `pyspark.sql.connect.expressions.NamedArgumentExpression`.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedArgument {
    /// The name of the argument.
    pub key: String,
    /// The value of the argument.
    pub value: Expression,
}

impl NamedArgument {
    /// The named argument `key => value`.
    pub fn new(key: impl Into<String>, value: Expression) -> Self {
        NamedArgument {
            key: key.into(),
            value,
        }
    }
}

impl Expression {
    /// Allocate an immutable expression node, of the current origin, as PySpark's
    /// `Expression.__init__` takes it.
    pub fn new(node: ExpressionNode) -> Self {
        Self {
            node: Arc::new(node),
            origin: crate::origin::current(),
        }
    }

    /// The origin in user code of this expression, if any.
    pub fn origin(&self) -> Option<&crate::origin::Origin> {
        self.origin.as_deref()
    }

    /// Create a literal expression.
    pub fn literal(value: LiteralExpression) -> Self {
        Self::new(ExpressionNode::Literal(value))
    }

    /// Create an unbound column reference.
    pub fn column(name: impl Into<String>) -> Self {
        Self::new(ExpressionNode::ColumnReference(ColumnReference::new(name)))
    }

    /// Create a non-distinct unresolved function call.
    pub fn function(name: impl Into<String>, args: Vec<Expression>) -> Self {
        Self::new(ExpressionNode::UnresolvedFunction(UnresolvedFunction::new(
            name, args,
        )))
    }

    /// Borrow this expression's node without exposing mutable shared state.
    pub fn node(&self) -> &ExpressionNode {
        self.node.as_ref()
    }

    /// Render this expression to a human-readable string, mirroring the
    /// `__repr__` of `pyspark.sql.connect.expressions.*`.
    ///
    /// This is what `Column.__repr__` wraps as `Column<'...'>`. pandas-on-Spark's
    /// `spark_column_equals` compares these strings (after stripping backticks) to
    /// decide column equality, so the rendering must be deterministic and mirror
    /// PySpark's format for the common operators.
    pub fn render(&self) -> String {
        // A level renders the levels below it: grow the stack for an expression of many levels.
        with_stack_for_a_level(|| self.render_node())
    }

    fn render_node(&self) -> String {
        match self.node() {
            ExpressionNode::Literal(lit) => lit.render(),
            ExpressionNode::ColumnReference(col_ref) => col_ref.name.clone(),
            ExpressionNode::UnresolvedFunction(func) => func.render(),
            ExpressionNode::UnresolvedStar(target, _) => match target {
                Some(t) => format!("unresolvedstar({t})"),
                None => "unresolvedstar()".to_string(),
            },
            // As `ColumnAlias.__repr__`: the names joined by commas.
            ExpressionNode::Alias(alias) => {
                format!("{} AS {}", alias.child.render(), alias.names.join(","))
            }
            // As `CastExpression.__repr__`: the type in upper case.
            ExpressionNode::Cast(cast) => {
                let type_str = match &cast.target {
                    CastTarget::Type(dt) => dt.simple_string(),
                    CastTarget::TypeStr(s) => s.clone(),
                }
                .to_uppercase();
                let name = if cast.eval_mode == Some(CastEvalMode::Try) {
                    "TRY_CAST"
                } else {
                    "CAST"
                };
                format!("{name}({} AS {type_str})", cast.child.render())
            }
            ExpressionNode::DirectShufflePartitionId(child) => {
                format!("DirectShufflePartitionID(child={})", child.render())
            }
            ExpressionNode::UnresolvedRegex(col_name, _) => format!("UnresolvedRegex({col_name})"),
            ExpressionNode::SortOrder(sort) => sort.render(),
            ExpressionNode::CaseWhen(case_when) => case_when.render(),
            ExpressionNode::UnresolvedExtractValue(ev) => {
                format!("{}['{}']", ev.child.render(), ev.extraction.render())
            }
            ExpressionNode::UpdateFields(uf) => match &uf.value_expression {
                Some(v) => format!(
                    "update_field({}, {}, {})",
                    uf.struct_expression.render(),
                    uf.field_name,
                    v.render()
                ),
                None => format!(
                    "drop_field({}, {})",
                    uf.struct_expression.render(),
                    uf.field_name
                ),
            },
            ExpressionNode::SQLExpression(sql) => sql.clone(),
            ExpressionNode::CallFunction(cf) => cf.render(),
            ExpressionNode::WindowExpression(we) => we.render(),
            // As `LambdaFunction.__repr__` and `UnresolvedNamedLambdaVariable.__repr__`.
            ExpressionNode::LambdaFunction(lf) => {
                let args: Vec<&str> = lf.arguments.iter().map(|a| a.name_parts.as_str()).collect();
                format!(
                    "LambdaFunction({}, {})",
                    lf.function.render(),
                    args.join(", ")
                )
            }
            ExpressionNode::UnresolvedNamedLambdaVariable(var) => var.name_parts.clone(),
            // As `CommonInlineUserDefinedFunction.__repr__`.
            ExpressionNode::CommonInlineUserDefinedFunction(udf) => {
                let args: Vec<String> = udf.arguments.iter().map(Expression::render).collect();
                format!("{}({})", udf.function_name, args.join(", "))
            }
            // As `NamedArgumentExpression.__repr__`.
            ExpressionNode::NamedArgument(arg) => format!("{} => {}", arg.key, arg.value.render()),
            ExpressionNode::Subquery(subquery) => subquery.render(),
            ExpressionNode::Sent(sent) => sent.rendered.clone(),
        }
    }

    /// Converts the expression to a Spark Connect protobuf expression, with its origin, as
    /// PySpark's `Expression._create_proto_expression` sends it.
    pub fn to_proto(&self) -> proto::Expression {
        // A level serializes the levels below it: grow the stack for an expression of many
        // levels, and count them for `with_expression_depth_bound`.
        with_stack_for_a_level(|| {
            let level = ExpressionLevel::enter();
            if level.0 > MAX_EXPRESSION_DEPTH && EXPRESSION_DEPTH_BOUNDED.with(Cell::get) {
                // The request fails with a `RecursionError`: stop here, so that the proto of the
                // expression is no deeper than the bound.
                return proto::Expression::default();
            }
            self.to_proto_with_origin()
        })
    }

    fn to_proto_with_origin(&self) -> proto::Expression {
        let mut expr = self.node_to_proto();
        if let Some(origin) = &self.origin {
            expr.common = Some(proto::ExpressionCommon {
                origin: Some(proto::Origin {
                    function: Some(proto::origin::Function::PythonOrigin(proto::PythonOrigin {
                        fragment: origin.fragment.clone(),
                        call_site: origin.call_site.clone(),
                    })),
                }),
            });
        }
        expr
    }

    fn node_to_proto(&self) -> proto::Expression {
        match self.node() {
            ExpressionNode::Literal(lit) => lit.to_proto(),
            ExpressionNode::ColumnReference(col_ref) => col_ref.to_proto(),
            ExpressionNode::UnresolvedFunction(func) => func.to_proto(),
            ExpressionNode::UnresolvedStar(target, plan_id) => proto::Expression {
                expr_type: Some(proto::expression::ExprType::UnresolvedStar(
                    proto::expression::UnresolvedStar {
                        unparsed_target: target.clone(),
                        plan_id: *plan_id,
                    },
                )),
                ..Default::default()
            },
            ExpressionNode::Alias(alias) => alias.to_proto(),
            ExpressionNode::Cast(cast) => cast.to_proto(),
            ExpressionNode::DirectShufflePartitionId(child) => {
                let mut expr = proto::Expression::default();
                expr.expr_type = Some(proto::expression::ExprType::DirectShufflePartitionId(
                    Box::new(proto::expression::DirectShufflePartitionId {
                        child: Some(Box::new(child.to_proto())),
                    }),
                ));
                expr
            }
            ExpressionNode::UnresolvedRegex(col_name, plan_id) => proto::Expression {
                expr_type: Some(proto::expression::ExprType::UnresolvedRegex(
                    proto::expression::UnresolvedRegex {
                        col_name: col_name.clone(),
                        plan_id: *plan_id,
                    },
                )),
                ..Default::default()
            },
            ExpressionNode::SortOrder(sort) => sort.to_proto(),
            ExpressionNode::CaseWhen(case_when) => case_when.to_proto(),
            ExpressionNode::UnresolvedExtractValue(ev) => ev.to_proto(),
            ExpressionNode::UpdateFields(uf) => uf.to_proto(),
            ExpressionNode::SQLExpression(sql) => {
                let mut expr = proto::Expression::default();
                expr.expr_type = Some(proto::expression::ExprType::ExpressionString(
                    proto::expression::ExpressionString {
                        expression: sql.clone(),
                    },
                ));
                expr
            }
            ExpressionNode::CallFunction(cf) => cf.to_proto(),
            ExpressionNode::WindowExpression(we) => we.to_proto(),
            ExpressionNode::LambdaFunction(lf) => lf.to_proto(),
            ExpressionNode::UnresolvedNamedLambdaVariable(var) => var.to_proto(),
            ExpressionNode::CommonInlineUserDefinedFunction(udf) => {
                let mut expr = proto::Expression::default();
                expr.expr_type = Some(
                    proto::expression::ExprType::CommonInlineUserDefinedFunction(udf.to_proto()),
                );
                expr
            }
            ExpressionNode::NamedArgument(arg) => proto::Expression {
                expr_type: Some(proto::expression::ExprType::NamedArgumentExpression(
                    Box::new(proto::NamedArgumentExpression {
                        key: arg.key.clone(),
                        value: Some(Box::new(arg.value.to_proto())),
                    }),
                )),
                ..Default::default()
            },
            ExpressionNode::Subquery(subquery) => subquery.to_proto(),
            ExpressionNode::Sent(sent) => sent.expression.clone(),
        }
    }
}

/// `pyspark.sql.connect.expressions.LiteralExpression`
#[derive(Debug, Clone, PartialEq)]
pub enum LiteralExpression {
    Null(DataType),
    Boolean(bool),
    Byte(i32),
    Short(i32),
    Integer(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    Decimal {
        value: String,
        precision: i32,
        scale: i32,
    },
    String(String),
    Binary(Vec<u8>),
    Date(i32),
    Timestamp(i64),
    TimestampNtz(i64),
    /// Signed microseconds in a Spark day-time interval.
    DayTimeInterval(i64),
    /// Signed months in a Spark year-month interval, which the server reads as an interval of
    /// `YearMonthIntervalType()`, of years to months, as the Scala client sends a `Period`.
    YearMonthInterval(i32),
    Time {
        nano: i64,
        precision: i32,
    },
    Array {
        element_type: Box<DataType>,
        elements: Vec<LiteralExpression>,
    },
}

impl LiteralExpression {
    pub fn to_proto(&self) -> proto::Expression {
        let mut expr = proto::Expression::default();
        let literal_type = match self {
            LiteralExpression::Null(data_type) => {
                proto::expression::literal::LiteralType::Null(data_type.to_proto())
            }
            LiteralExpression::Boolean(b) => proto::expression::literal::LiteralType::Boolean(*b),
            LiteralExpression::Byte(v) => proto::expression::literal::LiteralType::Byte(*v),
            LiteralExpression::Short(v) => proto::expression::literal::LiteralType::Short(*v),
            LiteralExpression::Integer(v) => proto::expression::literal::LiteralType::Integer(*v),
            LiteralExpression::Long(v) => proto::expression::literal::LiteralType::Long(*v),
            LiteralExpression::Float(v) => proto::expression::literal::LiteralType::Float(*v),
            LiteralExpression::Double(v) => proto::expression::literal::LiteralType::Double(*v),
            LiteralExpression::Decimal {
                value,
                precision,
                scale,
            } => {
                let mut decimal = proto::expression::literal::Decimal::default();
                decimal.value = value.clone();
                decimal.precision = Some(*precision);
                decimal.scale = Some(*scale);
                proto::expression::literal::LiteralType::Decimal(decimal)
            }
            LiteralExpression::String(v) => {
                proto::expression::literal::LiteralType::String(v.clone())
            }
            LiteralExpression::Binary(v) => {
                proto::expression::literal::LiteralType::Binary(v.clone().into())
            }
            LiteralExpression::Date(v) => proto::expression::literal::LiteralType::Date(*v),
            LiteralExpression::Timestamp(v) => {
                proto::expression::literal::LiteralType::Timestamp(*v)
            }
            LiteralExpression::TimestampNtz(v) => {
                proto::expression::literal::LiteralType::TimestampNtz(*v)
            }
            LiteralExpression::DayTimeInterval(v) => {
                proto::expression::literal::LiteralType::DayTimeInterval(*v)
            }
            LiteralExpression::YearMonthInterval(months) => {
                proto::expression::literal::LiteralType::YearMonthInterval(*months)
            }
            LiteralExpression::Time { nano, precision } => {
                let mut time = proto::expression::literal::Time::default();
                time.nano = *nano;
                time.precision = Some(*precision);
                proto::expression::literal::LiteralType::Time(time)
            }
            LiteralExpression::Array {
                element_type,
                elements,
            } => {
                let mut array = proto::expression::literal::Array::default();
                // The type of the elements, which the server requires of an array literal
                // (`ARRAY_LITERAL_MISSING_DATA_TYPE`), as PySpark's `LiteralExpression.to_plan`
                // writes it, also where the proto deprecates the field.
                #[allow(deprecated)]
                {
                    array.element_type = Some(element_type.to_proto());
                }
                for elem in elements {
                    let elem_proto = elem.to_proto();
                    if let Some(proto::expression::ExprType::Literal(lit)) = elem_proto.expr_type {
                        array.elements.push(lit);
                    }
                }
                proto::expression::literal::LiteralType::Array(array)
            }
        };

        let mut literal = proto::expression::Literal::default();
        literal.literal_type = Some(literal_type);
        expr.expr_type = Some(proto::expression::ExprType::Literal(literal));
        expr
    }

    /// The literal as a proto `Literal`, as `to_proto` sets it in its `Expression`.
    pub fn to_literal_proto(&self) -> proto::expression::Literal {
        match self.to_proto().expr_type {
            Some(proto::expression::ExprType::Literal(literal)) => literal,
            _ => proto::expression::Literal::default(),
        }
    }

    /// Render the literal value, mirroring `LiteralExpression.__repr__`: the Python value,
    /// as `f"{value}"` shows it, and a date, time, timestamp or day-time interval in the
    /// formats that it gives them.
    pub fn render(&self) -> String {
        match self {
            LiteralExpression::Null(_) => "NULL".to_string(),
            LiteralExpression::Boolean(b) => if *b { "True" } else { "False" }.to_string(),
            LiteralExpression::Byte(v)
            | LiteralExpression::Short(v)
            | LiteralExpression::Integer(v) => v.to_string(),
            LiteralExpression::Long(v) => v.to_string(),
            LiteralExpression::Float(v) => python_float_repr(*v as f64),
            LiteralExpression::Double(v) => python_float_repr(*v),
            LiteralExpression::Decimal { value, .. } => value.clone(),
            LiteralExpression::String(v) => v.clone(),
            LiteralExpression::Binary(v) => python_bytes_repr(v),
            LiteralExpression::Date(v) => date_repr(*v),
            LiteralExpression::Timestamp(v) => timestamp_repr(*v, true),
            LiteralExpression::TimestampNtz(v) => timestamp_repr(*v, false),
            LiteralExpression::DayTimeInterval(v) => iso_duration(*v),
            // The value, as `f"{value}"` shows a value of a type that has no format of its own.
            LiteralExpression::YearMonthInterval(months) => months.to_string(),
            LiteralExpression::Time { nano, .. } => time_repr(*nano),
            // A Python list, as `str(list)` shows its elements.
            LiteralExpression::Array { elements, .. } => {
                let inner: Vec<String> = elements.iter().map(|e| e.render_in_container()).collect();
                format!("[{}]", inner.join(", "))
            }
        }
    }

    /// The literal as Python's `repr` shows it within a container, such as a list: a string
    /// quoted, and None for a null.
    fn render_in_container(&self) -> String {
        match self {
            LiteralExpression::Null(_) => "None".to_string(),
            LiteralExpression::String(v) => python_str_repr(v),
            other => other.render(),
        }
    }

    /// Construct a null literal of a specific type.
    pub fn null(data_type: DataType) -> Self {
        LiteralExpression::Null(data_type)
    }

    /// Construct an integer literal.
    pub fn int(value: i32) -> Self {
        LiteralExpression::Integer(value)
    }

    /// Construct a long literal.
    pub fn long(value: i64) -> Self {
        LiteralExpression::Long(value)
    }

    /// Construct a double literal.
    pub fn double(value: f64) -> Self {
        LiteralExpression::Double(value)
    }

    /// Construct a string literal.
    pub fn string(value: impl Into<String>) -> Self {
        LiteralExpression::String(value.into())
    }

    /// Construct a boolean literal.
    pub fn boolean(value: bool) -> Self {
        LiteralExpression::Boolean(value)
    }

    /// Construct a binary literal (byte array).
    pub fn binary(value: Vec<u8>) -> Self {
        LiteralExpression::Binary(value)
    }
}

/// `pyspark.sql.connect.expressions.ColumnReference`
/// Represents a reference to a column by name (unresolved attribute).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnReference {
    /// The column name / unparsed identifier.
    pub name: String,
    /// Plan id this attribute is bound to. `None` for a free `col(...)`; set only
    /// when the column is resolved against a specific DataFrame (mirrors
    /// `ColumnReference._plan_id`).
    pub plan_id: Option<i64>,
    /// Whether this references a metadata column (mirrors
    /// `DataFrame.metadataColumn`). `false` for a normal `col(...)`.
    pub is_metadata_column: bool,
}

impl ColumnReference {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            plan_id: None,
            is_metadata_column: false,
        }
    }

    /// Bind this attribute to a DataFrame plan id.
    pub fn with_plan_id(mut self, plan_id: i64) -> Self {
        self.plan_id = Some(plan_id);
        self
    }

    /// Mark this reference as a metadata column (mirrors `DataFrame.metadataColumn`).
    pub fn metadata(mut self) -> Self {
        self.is_metadata_column = true;
        self
    }

    pub fn to_proto(&self) -> proto::Expression {
        let mut expr = proto::Expression::default();
        expr.expr_type = Some(proto::expression::ExprType::UnresolvedAttribute(
            proto::expression::UnresolvedAttribute {
                unparsed_identifier: self.name.clone(),
                plan_id: self.plan_id,
                is_metadata_column: Some(self.is_metadata_column),
            },
        ));
        expr
    }
}

/// `pyspark.sql.connect.expressions.UnresolvedExtractValue` - `col[key]` / `getField`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractValue {
    pub child: Expression,
    pub extraction: Expression,
}

impl ExtractValue {
    pub fn new(child: Expression, extraction: Expression) -> Self {
        Self { child, extraction }
    }

    pub fn to_proto(&self) -> proto::Expression {
        let mut expr = proto::Expression::default();
        expr.expr_type = Some(proto::expression::ExprType::UnresolvedExtractValue(
            Box::new(proto::expression::UnresolvedExtractValue {
                child: Some(Box::new(self.child.to_proto())),
                extraction: Some(Box::new(self.extraction.to_proto())),
            }),
        ));
        expr
    }
}

/// `pyspark.sql.connect.expressions.Expression.UpdateFields` - add/replace
/// (`withField`) or drop (`dropFields`, `value` = None) a struct field.
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateFieldsExpr {
    pub struct_expression: Expression,
    pub field_name: String,
    pub value_expression: Option<Expression>,
}

impl UpdateFieldsExpr {
    pub fn new(
        struct_expression: Expression,
        field_name: impl Into<String>,
        value_expression: Option<Expression>,
    ) -> Self {
        Self {
            struct_expression,
            field_name: field_name.into(),
            value_expression,
        }
    }

    pub fn to_proto(&self) -> proto::Expression {
        let mut expr = proto::Expression::default();
        expr.expr_type = Some(proto::expression::ExprType::UpdateFields(Box::new(
            proto::expression::UpdateFields {
                struct_expression: Some(Box::new(self.struct_expression.to_proto())),
                field_name: self.field_name.clone(),
                value_expression: self
                    .value_expression
                    .as_ref()
                    .map(|e| Box::new(e.to_proto())),
            },
        )));
        expr
    }
}

/// `pyspark.sql.connect.expressions.UnresolvedFunction`
/// Represents a function call with a name and arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct UnresolvedFunction {
    pub name: String,
    pub args: Vec<Expression>,
    pub is_distinct: bool,
}

impl UnresolvedFunction {
    pub fn new(name: impl Into<String>, args: Vec<Expression>) -> Self {
        Self {
            name: name.into(),
            args,
            is_distinct: false,
        }
    }

    pub fn new_distinct(name: impl Into<String>, args: Vec<Expression>) -> Self {
        Self {
            name: name.into(),
            args,
            is_distinct: true,
        }
    }

    /// Render this function call, mirroring `UnresolvedFunction.__repr__`:
    /// binary operators render infix as `(a op b)`, the unary `not`/negate render
    /// prefixed, everything else as `name(arg, arg, ...)`.
    pub fn render(&self) -> String {
        // As `UnresolvedFunction.__repr__`, which renders an operator as a function too.
        let inner: Vec<String> = self.args.iter().map(|a| a.render()).collect();
        if self.is_distinct {
            format!("{}(distinct {})", self.name, inner.join(", "))
        } else {
            format!("{}({})", self.name, inner.join(", "))
        }
    }

    pub fn to_proto(&self) -> proto::Expression {
        let mut expr = proto::Expression::default();
        let mut func = proto::expression::UnresolvedFunction::default();
        func.function_name = self.name.clone();
        func.is_distinct = self.is_distinct;
        for arg in &self.args {
            func.arguments.push(arg.to_proto());
        }
        expr.expr_type = Some(proto::expression::ExprType::UnresolvedFunction(func));
        expr
    }
}

/// `pyspark.sql.connect.expressions.Alias` / `ColumnAlias`
#[derive(Debug, Clone, PartialEq)]
pub struct Alias {
    pub child: Expression,
    pub names: Vec<String>,
    pub metadata: Option<String>,
}

impl Alias {
    pub fn new(child: Expression, name: impl Into<String>) -> Self {
        Self {
            child,
            names: vec![name.into()],
            metadata: None,
        }
    }

    /// An alias of several names, or of none, as the reference client's `ColumnAlias`
    /// takes them: a generator's columns are named in order.
    pub fn with_names<S: Into<String>>(
        child: Expression,
        names: impl IntoIterator<Item = S>,
    ) -> Self {
        Self {
            child,
            names: names.into_iter().map(Into::into).collect(),
            metadata: None,
        }
    }

    pub fn with_metadata(mut self, metadata: String) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn to_proto(&self) -> proto::Expression {
        let mut expr = proto::Expression::default();
        let mut alias = proto::expression::Alias::default();
        alias.expr = Some(Box::new(self.child.to_proto()));
        alias.name = self.names.clone();
        if let Some(meta) = &self.metadata {
            alias.metadata = Some(meta.clone());
        }
        expr.expr_type = Some(proto::expression::ExprType::Alias(Box::new(alias)));
        expr
    }
}

/// `pyspark.sql.connect.expressions.CastExpression` / `Cast`
#[derive(Debug, Clone, PartialEq)]
pub struct Cast {
    pub child: Expression,
    pub target: CastTarget,
    pub eval_mode: Option<CastEvalMode>,
}

/// The cast target: a structured `DataType` or a DDL type string. Mirrors the
/// `cast_to_type` oneof (`type` vs `type_str`); `Column.cast("string")` uses the
/// string form, `Column.cast(IntegerType())` the structured form.
#[derive(Debug, Clone, PartialEq)]
pub enum CastTarget {
    Type(DataType),
    TypeStr(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastEvalMode {
    Legacy,
    Ansi,
    Try,
}

impl Cast {
    pub fn new(child: Expression, to_type: DataType) -> Self {
        Self {
            child,
            target: CastTarget::Type(to_type),
            eval_mode: None,
        }
    }

    /// Cast to a DDL type string (mirrors `Column.cast("string")`).
    pub fn new_str(child: Expression, type_str: impl Into<String>) -> Self {
        Self {
            child,
            target: CastTarget::TypeStr(type_str.into()),
            eval_mode: None,
        }
    }

    pub fn with_eval_mode(mut self, mode: CastEvalMode) -> Self {
        self.eval_mode = Some(mode);
        self
    }

    pub fn to_proto(&self) -> proto::Expression {
        let mut expr = proto::Expression::default();
        let mut cast = proto::expression::Cast::default();
        cast.expr = Some(Box::new(self.child.to_proto()));
        cast.cast_to_type = Some(match &self.target {
            CastTarget::Type(dt) => proto::expression::cast::CastToType::Type(dt.to_proto()),
            CastTarget::TypeStr(s) => proto::expression::cast::CastToType::TypeStr(s.clone()),
        });

        if let Some(mode) = self.eval_mode {
            cast.eval_mode = match mode {
                CastEvalMode::Legacy => 1i32,
                CastEvalMode::Ansi => 2i32,
                CastEvalMode::Try => 3i32,
            };
        }

        expr.expr_type = Some(proto::expression::ExprType::Cast(Box::new(cast)));
        expr
    }
}

/// `pyspark.sql.connect.expressions.SortOrder`
#[derive(Debug, Clone, PartialEq)]
pub struct SortOrder {
    pub child: Expression,
    pub ascending: bool,
    pub null_ordering: NullOrdering,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullOrdering {
    First,
    Last,
}

impl SortOrder {
    /// Render this sort order, mirroring `SortOrder.__repr__`.
    pub fn render(&self) -> String {
        let dir = if self.ascending { "ASC" } else { "DESC" };
        let nulls = match self.null_ordering {
            NullOrdering::First => "NULLS FIRST",
            NullOrdering::Last => "NULLS LAST",
        };
        format!("{} {} {}", self.child.render(), dir, nulls)
    }

    pub fn asc_nulls_first(child: Expression) -> Self {
        Self {
            child,
            ascending: true,
            null_ordering: NullOrdering::First,
        }
    }

    pub fn asc_nulls_last(child: Expression) -> Self {
        Self {
            child,
            ascending: true,
            null_ordering: NullOrdering::Last,
        }
    }

    pub fn desc_nulls_first(child: Expression) -> Self {
        Self {
            child,
            ascending: false,
            null_ordering: NullOrdering::First,
        }
    }

    pub fn desc_nulls_last(child: Expression) -> Self {
        Self {
            child,
            ascending: false,
            null_ordering: NullOrdering::Last,
        }
    }

    pub fn to_proto(&self) -> proto::Expression {
        let mut expr = proto::Expression::default();
        let mut sort = proto::expression::SortOrder::default();
        sort.child = Some(Box::new(self.child.to_proto()));
        sort.direction = if self.ascending { 1i32 } else { 2i32 };
        sort.null_ordering = match self.null_ordering {
            NullOrdering::First => 1i32,
            NullOrdering::Last => 2i32,
        };
        expr.expr_type = Some(proto::expression::ExprType::SortOrder(Box::new(sort)));
        expr
    }
}

/// `pyspark.sql.connect.expressions.CaseWhen`
#[derive(Debug, Clone, PartialEq)]
pub struct CaseWhen {
    pub branches: Vec<(Expression, Expression)>,
    pub else_expr: Option<Expression>,
}

impl CaseWhen {
    pub fn new(branches: Vec<(Expression, Expression)>) -> Self {
        Self {
            branches,
            else_expr: None,
        }
    }

    pub fn with_else(mut self, else_expr: Expression) -> Self {
        self.else_expr = Some(else_expr);
        self
    }

    /// Render this CASE WHEN, mirroring `CaseWhen.__repr__`.
    pub fn render(&self) -> String {
        let mut parts = vec!["CASE".to_string()];
        for (cond, value) in &self.branches {
            parts.push(format!("WHEN {} THEN {}", cond.render(), value.render()));
        }
        if let Some(else_expr) = &self.else_expr {
            parts.push(format!("ELSE {}", else_expr.render()));
        }
        parts.push("END".to_string());
        parts.join(" ")
    }

    pub fn to_proto(&self) -> proto::Expression {
        let mut args = Vec::new();
        for (condition, value) in &self.branches {
            args.push(condition.clone());
            args.push(value.clone());
        }
        if let Some(else_expr) = &self.else_expr {
            args.push(else_expr.clone());
        }
        let func = UnresolvedFunction::new("when", args);
        func.to_proto()
    }
}

/// Wrapper for `spark.connect.CallFunction` protobuf message.
///
/// Mirrors `pyspark.sql.connect.expressions.CallFunction`: a named function call
/// carrying its argument expressions.
#[derive(Debug, Clone, PartialEq)]
pub struct CallFunctionWrapper {
    pub function_name: String,
    pub arguments: Vec<Expression>,
}

impl CallFunctionWrapper {
    /// Create a new CallFunctionWrapper with the given argument expressions.
    pub fn new(function_name: impl Into<String>, arguments: Vec<Expression>) -> Self {
        CallFunctionWrapper {
            function_name: function_name.into(),
            arguments,
        }
    }

    /// Mirrors `CallFunction.__repr__`: `CallFunction('<name>', <arg>, ...)`.
    pub fn render(&self) -> String {
        if self.arguments.is_empty() {
            format!("CallFunction('{}')", self.function_name)
        } else {
            let args: Vec<String> = self.arguments.iter().map(Expression::render).collect();
            format!(
                "CallFunction('{}', {})",
                self.function_name,
                args.join(", ")
            )
        }
    }

    /// Convert to protobuf.
    pub fn to_proto(&self) -> proto::Expression {
        let mut expr = proto::Expression::default();
        expr.expr_type = Some(proto::expression::ExprType::CallFunction(
            proto::CallFunction {
                function_name: self.function_name.clone(),
                arguments: self.arguments.iter().map(|a| a.to_proto()).collect(),
            },
        ));
        expr
    }
}

/// Wrapper for `spark.connect.Window` (window expression with OVER clause).
#[derive(Debug, Clone, PartialEq)]
pub struct WindowExpressionWrapper {
    pub window_function: Expression,
    pub partition_spec: Vec<Expression>,
    pub order_spec: Vec<SortOrder>,
    pub frame_spec: Option<(u32, FrameBoundary, FrameBoundary)>,
}

/// Frame boundary for window specification.
#[derive(Debug, Clone, PartialEq)]
pub enum FrameBoundary {
    UnboundedPreceding,
    Preceding(i64),
    CurrentRow,
    Following(i64),
    UnboundedFollowing,
}

impl FrameBoundary {
    /// The boundary as the int that `rowsBetween` and `rangeBetween` take, as
    /// `WindowFrame.__repr__` shows it.
    pub fn value(&self) -> i64 {
        match self {
            FrameBoundary::UnboundedPreceding => i64::MIN,
            FrameBoundary::Preceding(n) => -n,
            FrameBoundary::CurrentRow => 0,
            FrameBoundary::Following(n) => *n,
            FrameBoundary::UnboundedFollowing => i64::MAX,
        }
    }
}

/// As `WindowSpec.__repr__` of PySpark Connect: its partitioning, ordering and frame. A
/// frame type of 1 is a row frame, and another a range frame, as in the proto.
pub fn window_spec_repr(
    partition_spec: &[Expression],
    order_spec: &[SortOrder],
    frame_spec: Option<&(u32, FrameBoundary, FrameBoundary)>,
) -> String {
    let mut parts = Vec::new();
    if !partition_spec.is_empty() {
        let partitions: Vec<String> = partition_spec.iter().map(Expression::render).collect();
        parts.push(format!("PartitionBy({})", partitions.join(", ")));
    }
    if !order_spec.is_empty() {
        let orders: Vec<String> = order_spec.iter().map(SortOrder::render).collect();
        parts.push(format!("OrderBy({})", orders.join(", ")));
    }
    if let Some((frame_type, lower, upper)) = frame_spec {
        let kind = if *frame_type == 1 {
            "ROW_FRAME"
        } else {
            "RANGE_FRAME"
        };
        parts.push(format!(
            "WindowFrame({kind}, {}, {})",
            lower.value(),
            upper.value()
        ));
    }
    format!("WindowSpec({})", parts.join(", "))
}

impl WindowExpressionWrapper {
    /// Create a new WindowExpressionWrapper.
    pub fn new(
        window_function: Expression,
        partition_spec: Vec<Expression>,
        order_spec: Vec<SortOrder>,
        frame_spec: Option<(u32, FrameBoundary, FrameBoundary)>,
    ) -> Self {
        Self {
            window_function,
            partition_spec,
            order_spec,
            frame_spec,
        }
    }

    /// Convert to protobuf.
    /// As `WindowExpression.__repr__`.
    pub fn render(&self) -> String {
        let spec = window_spec_repr(
            &self.partition_spec,
            &self.order_spec,
            self.frame_spec.as_ref(),
        );
        format!(
            "WindowExpression({}, ({spec}))",
            self.window_function.render()
        )
    }

    pub fn to_proto(&self) -> proto::Expression {
        let mut expr = proto::Expression::default();
        let mut window = proto::expression::Window::default();

        // Set the window function
        window.window_function = Some(Box::new(self.window_function.to_proto()));

        // Set partition spec
        window.partition_spec = self.partition_spec.iter().map(|e| e.to_proto()).collect();

        // Set order spec
        window.order_spec = self
            .order_spec
            .iter()
            .map(|s| s.to_proto_sort_order())
            .collect();

        // Set frame spec if present
        if let Some((frame_type, lower, upper)) = &self.frame_spec {
            let mut frame = proto::expression::window::WindowFrame::default();
            frame.frame_type = *frame_type as i32;
            // A frame type of 1 is a row frame, and another a range frame.
            let rows = *frame_type == 1;
            frame.lower = Some(Box::new(to_proto_frame_boundary(lower, rows)));
            frame.upper = Some(Box::new(to_proto_frame_boundary(upper, rows)));
            window.frame_spec = Some(Box::new(frame));
        }

        expr.expr_type = Some(proto::expression::ExprType::Window(Box::new(window)));
        expr
    }
}

/// Convert FrameBoundary to proto FrameBoundary, as PySpark Connect's `WindowExpression`
/// sends the boundary of a value: an integer literal in a row frame, and a long literal in a
/// range frame.
fn to_proto_frame_boundary(
    boundary: &FrameBoundary,
    rows: bool,
) -> proto::expression::window::window_frame::FrameBoundary {
    let offset = |value: i64| {
        // A ROWS-frame offset must be an IntegerType literal (Spark rejects bigint with
        // SPECIFIED_WINDOW_FRAME_UNACCEPTED_TYPE); one out of its range is sent as a long,
        // for the server to reject, rather than truncated.
        let literal = match i32::try_from(value) {
            Ok(value) if rows => LiteralExpression::int(value),
            _ => LiteralExpression::long(value),
        };
        Some(
            proto::expression::window::window_frame::frame_boundary::Boundary::Value(Box::new(
                Expression::literal(literal).to_proto(),
            )),
        )
    };
    let boundary = match boundary {
        FrameBoundary::UnboundedPreceding => {
            Some(proto::expression::window::window_frame::frame_boundary::Boundary::Unbounded(true))
        }
        FrameBoundary::UnboundedFollowing => {
            Some(proto::expression::window::window_frame::frame_boundary::Boundary::Unbounded(true))
        }
        FrameBoundary::CurrentRow => Some(
            proto::expression::window::window_frame::frame_boundary::Boundary::CurrentRow(true),
        ),
        // A preceding boundary is a negative offset.
        FrameBoundary::Preceding(n) => offset(n.saturating_neg()),
        FrameBoundary::Following(n) => offset(*n),
    };
    proto::expression::window::window_frame::FrameBoundary { boundary }
}

impl SortOrder {
    /// Convert to proto SortOrder (used by window).
    pub fn to_proto_sort_order(&self) -> proto::expression::SortOrder {
        let mut sort = proto::expression::SortOrder::default();
        sort.child = Some(Box::new(self.child.to_proto()));
        sort.direction = if self.ascending { 1i32 } else { 2i32 };
        sort.null_ordering = match self.null_ordering {
            NullOrdering::First => 1i32,
            NullOrdering::Last => 2i32,
        };
        sort
    }
}

/// `pyspark.sql.connect.expressions.LambdaFunction` - a lambda function with body and arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct LambdaFunction {
    pub function: Expression,
    pub arguments: Vec<UnresolvedNamedLambdaVariable>,
}

impl LambdaFunction {
    pub fn new(function: Expression, arguments: Vec<UnresolvedNamedLambdaVariable>) -> Self {
        Self {
            function,
            arguments,
        }
    }

    pub fn to_proto(&self) -> proto::Expression {
        let mut expr = proto::Expression::default();
        let mut lambda = proto::expression::LambdaFunction::default();
        lambda.function = Some(Box::new(self.function.to_proto()));
        for arg in &self.arguments {
            lambda
                .arguments
                .push(proto::expression::UnresolvedNamedLambdaVariable {
                    name_parts: vec![arg.name_parts.clone()],
                });
        }
        expr.expr_type = Some(proto::expression::ExprType::LambdaFunction(Box::new(
            lambda,
        )));
        expr
    }
}

/// `pyspark.sql.connect.expressions.UnresolvedNamedLambdaVariable` - a lambda variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedNamedLambdaVariable {
    pub name_parts: String,
}

impl UnresolvedNamedLambdaVariable {
    pub fn new(name_parts: impl Into<String>) -> Self {
        Self {
            name_parts: name_parts.into(),
        }
    }

    pub fn to_proto(&self) -> proto::Expression {
        let mut expr = proto::Expression::default();
        expr.expr_type = Some(proto::expression::ExprType::UnresolvedNamedLambdaVariable(
            proto::expression::UnresolvedNamedLambdaVariable {
                name_parts: vec![self.name_parts.clone()],
            },
        ));
        expr
    }
}

/// Python's `repr` of a float, as PySpark's reprs show one: the shortest digits that
/// round-trip, in fixed notation for a decimal exponent from -4 to 15, otherwise in
/// scientific notation with a signed exponent of at least two digits.
fn python_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let scientific = format!("{x:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(mantissa) => ("-", mantissa),
        None => ("", mantissa),
    };
    let digits = mantissa.replace('.', "");
    if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        let fixed = if point <= 0 {
            format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{fixed}")
    } else {
        let mantissa = if digits.len() == 1 {
            digits
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exponent_sign}{:02}", exponent.abs())
    }
}

/// The quote that Python's `repr` puts around a string or bytes: a double quote if it has
/// a single quote and no double quote, otherwise a single quote.
fn python_quote(has_single: bool, has_double: bool) -> char {
    if has_single && !has_double {
        '"'
    } else {
        '\''
    }
}

/// Python's `repr` of `bytes`, which `f"{value}"` of bytes also is.
fn python_bytes_repr(bytes: &[u8]) -> String {
    let quote = python_quote(bytes.contains(&b'\''), bytes.contains(&b'"'));
    let mut out = String::from("b");
    out.push(quote);
    for &byte in bytes {
        match byte {
            b'\\' => out.push_str("\\\\"),
            b'\t' => out.push_str("\\t"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            byte if byte as char == quote => {
                out.push('\\');
                out.push(quote);
            }
            0x20..=0x7e => out.push(byte as char),
            byte => out.push_str(&format!("\\x{byte:02x}")),
        }
    }
    out.push(quote);
    out
}

/// Python's `repr` of a string, as a list shows its strings.
pub(crate) fn python_str_repr(value: &str) -> String {
    let quote = python_quote(value.contains('\''), value.contains('"'));
    let mut out = String::new();
    out.push(quote);
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if c.is_control() => {
                let code = c as u32;
                if code < 0x100 {
                    out.push_str(&format!("\\x{code:02x}"));
                } else if code < 0x10000 {
                    out.push_str(&format!("\\u{code:04x}"));
                } else {
                    out.push_str(&format!("\\U{code:08x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// A date of `days` since the epoch, as `DateType.fromInternal` and `%Y-%m-%d` give it.
fn date_repr(days: i32) -> String {
    chrono::NaiveDate::from_num_days_from_ce_opt(days.saturating_add(719_163))
        .map(|date| date.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| days.to_string())
}

/// A timestamp of `micros` since the epoch, as `%Y-%m-%d %H:%M:%S.%f` gives it: in the local
/// time zone, as `TimestampType.fromInternal` does, or in UTC, as `TimestampNTZType`'s does.
fn timestamp_repr(micros: i64, local: bool) -> String {
    const FORMAT: &str = "%Y-%m-%d %H:%M:%S%.6f";
    match chrono::DateTime::from_timestamp_micros(micros) {
        Some(utc) if local => utc.with_timezone(&chrono::Local).format(FORMAT).to_string(),
        Some(utc) => utc.naive_utc().format(FORMAT).to_string(),
        None => micros.to_string(),
    }
}

/// A time of `nano` since midnight, as `TimeType.fromInternal` and `%H:%M:%S.%f` give it.
fn time_repr(nano: i64) -> String {
    let (hours, rest) = (nano / 3_600_000_000_000, nano % 3_600_000_000_000);
    let (minutes, rest) = (rest / 60_000_000_000, rest % 60_000_000_000);
    let (seconds, rest) = (rest / 1_000_000_000, rest % 1_000_000_000);
    format!("{hours:02}:{minutes:02}:{seconds:02}.{:06}", rest / 1_000)
}

/// A day-time interval of `micros`, as pandas' `Timedelta.isoformat` gives it: the days of
/// the interval, which are negative for a negative one, and the time of day after them, with
/// the seconds of the fraction without trailing zeros.
fn iso_duration(micros: i64) -> String {
    const DAY: i64 = 86_400_000_000;
    let days = micros.div_euclid(DAY);
    let rest = micros.rem_euclid(DAY);
    let (hours, rest) = (rest / 3_600_000_000, rest % 3_600_000_000);
    let (minutes, rest) = (rest / 60_000_000, rest % 60_000_000);
    let (seconds, micro) = (rest / 1_000_000, rest % 1_000_000);
    let fraction = format!("{seconds}.{micro:06}000");
    let seconds = fraction.trim_end_matches('0').trim_end_matches('.');
    format!("P{days}DT{hours}H{minutes}M{seconds}S")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(name: &str) -> Expression {
        Expression::column(name)
    }

    #[test]
    fn test_render_matches_pyspark_connect_format() {
        // Column reference and literal.
        assert_eq!(col("x").render(), "x");
        assert_eq!(
            Expression::literal(LiteralExpression::Integer(0)).render(),
            "0"
        );
        assert_eq!(
            Expression::literal(LiteralExpression::null(DataType::Integer)).render(),
            "NULL"
        );

        // An operator renders as a function, as UnresolvedFunction.__repr__.
        let add = Expression::function(
            "+",
            vec![col("x"), Expression::literal(LiteralExpression::Integer(1))],
        );
        assert_eq!(add.render(), "+(x, 1)");

        let eq = Expression::function("==", vec![col("a"), col("b")]);
        assert_eq!(eq.render(), "==(a, b)");

        // Unary not.
        let neq = Expression::function("not", vec![eq.clone()]);
        assert_eq!(neq.render(), "not(==(a, b))");

        // Non-operator function renders as name(args).
        let f = Expression::function("coalesce", vec![col("a"), col("b")]);
        assert_eq!(f.render(), "coalesce(a, b)");

        // Alias, cast, star.
        assert_eq!(
            Expression::from(ExpressionNode::Alias(Box::new(Alias::new(col("x"), "y")))).render(),
            "x AS y"
        );
        assert_eq!(
            Expression::from(ExpressionNode::Cast(Box::new(Cast {
                child: col("x"),
                target: CastTarget::TypeStr("int".to_string()),
                eval_mode: None,
            })))
            .render(),
            "CAST(x AS INT)"
        );
        assert_eq!(
            Expression::from(ExpressionNode::UnresolvedStar(None, None)).render(),
            "unresolvedstar()"
        );

        // Equal expressions render identically; different ones differ
        // (the property spark_column_equals relies on).
        assert_eq!(add.render(), add.render());
        assert_ne!(add.render(), eq.render());
    }

    /// `id + 1` chained `levels` times, as `col("id") + 1` in a loop of PySpark builds it.
    fn chained_additions(levels: usize) -> Expression {
        let mut expr = col("id");
        for _ in 0..levels {
            expr = Expression::function(
                "+",
                vec![expr, Expression::literal(LiteralExpression::Integer(1))],
            );
        }
        expr
    }

    /// The levels of a serialized expression of `chained_additions`.
    fn serialized_levels(expr: &proto::Expression) -> usize {
        let mut levels = 1;
        let mut expr = expr;
        while let Some(proto::expression::ExprType::UnresolvedFunction(function)) = &expr.expr_type
        {
            expr = &function.arguments[0];
            levels += 1;
        }
        levels
    }

    /// An expression of any number of levels is serialized, rendered, compared, formatted and
    /// dropped on a small stack, which grows on demand; one deeper than `MAX_EXPRESSION_DEPTH`
    /// is a `RecursionError`, as PySpark's serializer raises it.
    #[test]
    fn deep_expressions_do_not_overflow_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(512 << 10)
            .spawn(|| {
                let at_bound = chained_additions(MAX_EXPRESSION_DEPTH - 1);
                let proto =
                    with_expression_depth_bound(|| at_bound.to_proto()).expect("at the bound");
                assert_eq!(serialized_levels(&proto), MAX_EXPRESSION_DEPTH);

                let deep = chained_additions(100_000);
                let error = with_expression_depth_bound(|| deep.to_proto()).expect_err("deep");
                assert_eq!(
                    error.kind,
                    spark_connect_core::error::SparkErrorKind::RecursionError
                );
                assert!(
                    error.message().contains("maximum recursion depth"),
                    "{error}"
                );
                // Out of a request, every level is serialized.
                let full = deep.to_proto();
                assert_eq!(serialized_levels(&full), 100_001);
                // The caller drops it: a proto this deep needs the stack of a big thread.
                std::thread::Builder::new()
                    .stack_size(256 << 20)
                    .spawn(move || std::mem::drop(full))
                    .unwrap()
                    .join()
                    .unwrap();
                let rendered = chained_additions(10_000).render();
                assert!(rendered.starts_with("+(+(+(") && rendered.contains("+(id, 1), 1)"));
                assert!(rendered.ends_with(", 1), 1), 1)"));
                assert_eq!(rendered.len(), "id".len() + 10_000 * "+(, 1)".len());
                assert!(deep == deep.clone());
                assert!(format!("{deep:?}").starts_with("UnresolvedFunction("));
                std::mem::drop(deep);

                // The deepest level is counted again for each serialization.
                assert!(with_expression_depth_bound(|| at_bound.to_proto()).is_ok());
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn test_literal_integer() {
        let lit = LiteralExpression::int(42);
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
    }

    #[test]
    fn test_column_reference() {
        let col = ColumnReference::new("x");
        let proto = col.to_proto();
        assert!(proto.expr_type.is_some());
    }

    #[test]
    fn test_literal_decimal() {
        let lit = LiteralExpression::Decimal {
            value: "123.45".to_string(),
            precision: 5,
            scale: 2,
        };
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::Decimal(decimal)) =
                literal.literal_type
            {
                assert_eq!(decimal.value, "123.45");
                assert_eq!(decimal.precision, Some(5));
                assert_eq!(decimal.scale, Some(2));
            } else {
                panic!("Expected decimal literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn test_literal_date() {
        let lit = LiteralExpression::Date(18993); // some days since epoch
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::Date(days)) = literal.literal_type
            {
                assert_eq!(days, 18993);
            } else {
                panic!("Expected date literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn test_literal_timestamp() {
        let lit = LiteralExpression::Timestamp(1693526400000000); // micros since epoch
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::Timestamp(micros)) =
                literal.literal_type
            {
                assert_eq!(micros, 1693526400000000);
            } else {
                panic!("Expected timestamp literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn sent_expressions_are_sent_and_rendered_as_they_were() {
        // As a pickled Column unpickles: the expression as it was sent, of its rendering, which
        // the expressions over it send and render.
        let sum = Expression::function(
            "+",
            vec![col("a"), Expression::literal(LiteralExpression::Long(1))],
        );
        let original = Expression::new(ExpressionNode::Alias(Box::new(Alias::new(sum, "b"))));
        let sent = Expression::new(ExpressionNode::Sent(Box::new(SentExpression {
            expression: original.to_proto(),
            rendered: original.render(),
        })));
        assert_eq!(sent.to_proto(), original.to_proto());
        assert_eq!(sent.render(), "+(a, 1) AS b");
        let over = Expression::function("abs", vec![sent]);
        let over_original = Expression::function("abs", vec![original]);
        assert_eq!(over.to_proto(), over_original.to_proto());
        assert_eq!(over.render(), over_original.render());
    }

    // Additional tests for expression render() methods and uncovered branches
    #[test]
    fn test_literal_render_byte() {
        let lit = LiteralExpression::Byte(42);
        assert_eq!(lit.render(), "42");
    }

    #[test]
    fn test_literal_render_short() {
        let lit = LiteralExpression::Short(1000);
        assert_eq!(lit.render(), "1000");
    }

    #[test]
    fn test_literal_render_long() {
        let lit = LiteralExpression::Long(9999999999i64);
        assert_eq!(lit.render(), "9999999999");
    }

    #[test]
    fn test_literal_render_float() {
        let lit = LiteralExpression::Float(3.14);
        assert_eq!(lit.render(), "3.140000104904175");
    }

    #[test]
    fn test_literal_render_double() {
        let lit = LiteralExpression::Double(2.71828);
        assert_eq!(lit.render(), "2.71828");
    }

    #[test]
    fn test_literal_render_string() {
        let lit = LiteralExpression::String("hello".to_string());
        assert_eq!(lit.render(), "hello");
    }

    #[test]
    fn test_literal_render_binary() {
        let lit = LiteralExpression::Binary(vec![1, 2, 3, b'a', b'\'']);
        // As Python's repr of bytes.
        assert_eq!(lit.render(), "b\"\\x01\\x02\\x03a'\"");
    }

    #[test]
    fn test_literal_render_date() {
        let lit = LiteralExpression::Date(18993);
        assert_eq!(lit.render(), "2022-01-01");
    }

    #[test]
    fn test_literal_render_timestamp_ntz() {
        let lit = LiteralExpression::TimestampNtz(1693526400000000);
        assert_eq!(lit.render(), "2023-09-01 00:00:00.000000");
    }

    #[test]
    fn test_literal_render_time() {
        let lit = LiteralExpression::Time {
            nano: 3600000000000i64,
            precision: 9,
        };
        assert_eq!(lit.render(), "01:00:00.000000");
    }

    #[test]
    fn test_literal_render_array() {
        let lit = LiteralExpression::Array {
            element_type: Box::new(DataType::Integer),
            elements: vec![
                LiteralExpression::int(1),
                LiteralExpression::int(2),
                LiteralExpression::int(3),
            ],
        };
        assert_eq!(lit.render(), "[1, 2, 3]");
    }

    #[test]
    fn test_literal_render_array_empty() {
        let lit = LiteralExpression::Array {
            element_type: Box::new(DataType::Integer),
            elements: vec![],
        };
        assert_eq!(lit.render(), "[]");
    }

    #[test]
    fn test_literal_render_decimal() {
        let lit = LiteralExpression::Decimal {
            value: "123.45".to_string(),
            precision: 5,
            scale: 2,
        };
        assert_eq!(lit.render(), "123.45");
    }

    #[test]
    fn test_expression_render_unresolved_star_with_target() {
        let expr: Expression =
            ExpressionNode::UnresolvedStar(Some("table.*".to_string()), None).into();
        assert_eq!(expr.render(), "unresolvedstar(table.*)");
    }

    #[test]
    fn test_expression_render_unresolved_regex() {
        let expr: Expression = ExpressionNode::UnresolvedRegex("`col_.*`".to_string(), None).into();
        assert_eq!(expr.render(), "UnresolvedRegex(`col_.*`)");
    }

    #[test]
    fn test_expression_render_direct_shuffle_partition_id() {
        let child = col("x");
        let expr: Expression = ExpressionNode::DirectShufflePartitionId(child).into();
        assert_eq!(expr.render(), "DirectShufflePartitionID(child=x)");
    }

    #[test]
    fn test_expression_render_unresolved_extract_value() {
        let child = col("struct_col");
        let extraction = Expression::literal(LiteralExpression::string("field"));
        let ev = ExtractValue::new(child, extraction);
        let expr: Expression = ExpressionNode::UnresolvedExtractValue(Box::new(ev)).into();
        assert_eq!(expr.render(), "struct_col['field']");
    }

    #[test]
    fn test_expression_render_update_fields_with_value() {
        let struct_expr = col("s");
        let value_expr = Expression::literal(LiteralExpression::int(42));
        let uf = UpdateFieldsExpr::new(struct_expr, "f1", Some(value_expr));
        let expr: Expression = ExpressionNode::UpdateFields(Box::new(uf)).into();
        assert_eq!(expr.render(), "update_field(s, f1, 42)");
    }

    #[test]
    fn test_expression_render_update_fields_drop() {
        let struct_expr = col("s");
        let uf = UpdateFieldsExpr::new(struct_expr, "f1", None);
        let expr: Expression = ExpressionNode::UpdateFields(Box::new(uf)).into();
        assert_eq!(expr.render(), "drop_field(s, f1)");
    }

    #[test]
    fn test_sort_order_render_asc_nulls_first() {
        let sort = SortOrder::asc_nulls_first(col("x"));
        assert_eq!(sort.render(), "x ASC NULLS FIRST");
    }

    #[test]
    fn test_sort_order_render_asc_nulls_last() {
        let sort = SortOrder::asc_nulls_last(col("x"));
        assert_eq!(sort.render(), "x ASC NULLS LAST");
    }

    #[test]
    fn test_sort_order_render_desc_nulls_first() {
        let sort = SortOrder::desc_nulls_first(col("x"));
        assert_eq!(sort.render(), "x DESC NULLS FIRST");
    }

    #[test]
    fn test_sort_order_render_desc_nulls_last() {
        let sort = SortOrder::desc_nulls_last(col("x"));
        assert_eq!(sort.render(), "x DESC NULLS LAST");
    }

    #[test]
    fn test_expression_render_sort_order() {
        let sort = SortOrder::asc_nulls_first(col("a"));
        let expr: Expression = ExpressionNode::SortOrder(Box::new(sort)).into();
        assert_eq!(expr.render(), "a ASC NULLS FIRST");
    }

    #[test]
    fn test_case_when_render_single_branch() {
        let cw = CaseWhen::new(vec![(
            Expression::literal(LiteralExpression::boolean(true)),
            Expression::literal(LiteralExpression::int(1)),
        )]);
        assert_eq!(cw.render(), "CASE WHEN True THEN 1 END");
    }

    #[test]
    fn test_case_when_render_multiple_branches() {
        let cw = CaseWhen::new(vec![
            (
                Expression::literal(LiteralExpression::boolean(true)),
                Expression::literal(LiteralExpression::int(1)),
            ),
            (
                Expression::literal(LiteralExpression::boolean(false)),
                Expression::literal(LiteralExpression::int(2)),
            ),
        ]);
        assert_eq!(cw.render(), "CASE WHEN True THEN 1 WHEN False THEN 2 END");
    }

    #[test]
    fn test_case_when_render_with_else() {
        let cw = CaseWhen::new(vec![(
            Expression::literal(LiteralExpression::boolean(true)),
            Expression::literal(LiteralExpression::int(1)),
        )])
        .with_else(Expression::literal(LiteralExpression::int(99)));
        assert_eq!(cw.render(), "CASE WHEN True THEN 1 ELSE 99 END");
    }

    #[test]
    fn test_expression_render_case_when() {
        let cw = CaseWhen::new(vec![(
            col("cond"),
            Expression::literal(LiteralExpression::int(1)),
        )]);
        let expr: Expression = ExpressionNode::CaseWhen(Box::new(cw)).into();
        assert_eq!(expr.render(), "CASE WHEN cond THEN 1 END");
    }

    #[test]
    fn test_unresolved_function_render_negate() {
        let func = UnresolvedFunction::new("negate", vec![col("x")]);
        assert_eq!(func.render(), "negate(x)");
    }

    #[test]
    fn test_unresolved_function_render_negative() {
        let func = UnresolvedFunction::new("negative", vec![col("x")]);
        assert_eq!(func.render(), "negative(x)");
    }

    #[test]
    fn test_unresolved_function_render_multiple_args() {
        let func = UnresolvedFunction::new(
            "concat",
            vec![
                Expression::literal(LiteralExpression::string("a")),
                Expression::literal(LiteralExpression::string("b")),
                Expression::literal(LiteralExpression::string("c")),
            ],
        );
        assert_eq!(func.render(), "concat(a, b, c)");
    }

    #[test]
    fn test_alias_render_multiple_names() {
        let alias = Alias {
            child: col("x"),
            names: vec!["a".to_string(), "b".to_string()],
            metadata: None,
        };
        let expr: Expression = ExpressionNode::Alias(Box::new(alias)).into();
        // As `ColumnAlias.__repr__`: the names joined by commas.
        assert_eq!(expr.render(), "x AS a,b");
        let none: Expression =
            ExpressionNode::Alias(Box::new(Alias::with_names(col("x"), Vec::<String>::new())))
                .into();
        assert_eq!(none.render(), "x AS ");
    }

    #[test]
    fn test_cast_render_with_datatype() {
        let cast = Cast::new(
            col("x"),
            DataType::String {
                collation: "".to_string(),
            },
        );
        let expr: Expression = ExpressionNode::Cast(Box::new(cast)).into();
        assert_eq!(expr.render(), "CAST(x AS STRING)");
    }

    #[test]
    fn test_sql_expression_render() {
        let expr: Expression =
            ExpressionNode::SQLExpression("SELECT * FROM table".to_string()).into();
        assert_eq!(expr.render(), "SELECT * FROM table");
    }

    #[test]
    fn test_literal_boolean_true_render() {
        let lit = LiteralExpression::boolean(true);
        assert_eq!(lit.render(), "True");
    }

    #[test]
    fn test_literal_boolean_false_render() {
        let lit = LiteralExpression::boolean(false);
        assert_eq!(lit.render(), "False");
    }

    #[test]
    fn test_unresolved_function_render_not() {
        let func = UnresolvedFunction::new(
            "not",
            vec![Expression::literal(LiteralExpression::boolean(true))],
        );
        assert_eq!(func.render(), "not(True)");
    }

    #[test]
    fn test_infix_operators_render() {
        let ops = vec![
            ("+", "+(1, 2)"),
            ("-", "-(1, 2)"),
            ("*", "*(1, 2)"),
            ("/", "/(1, 2)"),
            ("%", "%(1, 2)"),
            ("==", "==(1, 2)"),
            ("!=", "!=(1, 2)"),
            ("<", "<(1, 2)"),
            ("<=", "<=(1, 2)"),
            (">", ">(1, 2)"),
            (">=", ">=(1, 2)"),
            ("&", "&(1, 2)"),
            ("|", "|(1, 2)"),
            ("^", "^(1, 2)"),
            ("<=>", "<=>(1, 2)"),
        ];

        for (op_name, expected_result) in ops.iter().take(15) {
            let func = UnresolvedFunction::new(
                *op_name,
                vec![
                    Expression::literal(LiteralExpression::int(1)),
                    Expression::literal(LiteralExpression::int(2)),
                ],
            );
            assert_eq!(
                func.render(),
                *expected_result,
                "Failed for operator: {}",
                op_name
            );
        }

        // Test 'and' and 'or' separately
        let and_func = UnresolvedFunction::new(
            "and",
            vec![
                Expression::literal(LiteralExpression::boolean(true)),
                Expression::literal(LiteralExpression::boolean(false)),
            ],
        );
        assert_eq!(and_func.render(), "and(True, False)");

        let or_func = UnresolvedFunction::new(
            "or",
            vec![
                Expression::literal(LiteralExpression::boolean(true)),
                Expression::literal(LiteralExpression::boolean(false)),
            ],
        );
        assert_eq!(or_func.render(), "or(True, False)");
    }

    #[test]
    fn named_argument_renders_and_serializes_as_pyspark() {
        // As NamedArgumentExpression: `key => value`, sent as a named_argument_expression.
        let value = Expression::column("x");
        let arg: Expression =
            ExpressionNode::NamedArgument(Box::new(NamedArgument::new("b", value.clone()))).into();
        assert_eq!(arg.render(), "b => x");
        let Some(proto::expression::ExprType::NamedArgumentExpression(named)) =
            arg.to_proto().expr_type
        else {
            panic!("expected a NamedArgumentExpression");
        };
        assert_eq!(named.key, "b");
        assert_eq!(named.value.map(|v| *v), Some(value.to_proto()));
    }

    #[test]
    fn test_column_reference_with_plan_id() {
        let mut col_ref = ColumnReference::new("x");
        col_ref = col_ref.with_plan_id(123);
        assert_eq!(col_ref.plan_id, Some(123));
    }

    #[test]
    fn test_column_reference_metadata() {
        let mut col_ref = ColumnReference::new("x");
        col_ref = col_ref.metadata();
        assert!(col_ref.is_metadata_column);
    }

    #[test]
    fn test_call_function_wrapper_render() {
        let cf = CallFunctionWrapper::new("my_func", vec![col("a"), col("b")]);
        let expr: Expression = ExpressionNode::CallFunction(Box::new(cf)).into();
        // As `CallFunction.__repr__` in PySpark Connect.
        assert_eq!(expr.render(), "CallFunction('my_func', a, b)");
        let no_args = CallFunctionWrapper::new("current_date", vec![]);
        let expr: Expression = ExpressionNode::CallFunction(Box::new(no_args)).into();
        assert_eq!(expr.render(), "CallFunction('current_date')");
    }

    #[test]
    fn test_window_expression_render() {
        let we = WindowExpressionWrapper::new(
            Expression::function("sum", vec![col("x")]),
            vec![col("group_col")],
            vec![],
            None,
        );
        let expr: Expression = ExpressionNode::WindowExpression(Box::new(we)).into();
        // As WindowExpression.__repr__.
        assert_eq!(
            expr.render(),
            "WindowExpression(sum(x), (WindowSpec(PartitionBy(group_col))))"
        );
    }

    #[test]
    fn test_lambda_function_render() {
        let lf = LambdaFunction::new(col("x"), vec![UnresolvedNamedLambdaVariable::new("x")]);
        let expr: Expression = ExpressionNode::LambdaFunction(Box::new(lf)).into();
        assert_eq!(expr.render(), "LambdaFunction(x, x)");
    }

    #[test]
    fn test_unresolved_named_lambda_variable_render() {
        let var = UnresolvedNamedLambdaVariable::new("x");
        let expr: Expression = ExpressionNode::UnresolvedNamedLambdaVariable(var).into();
        assert_eq!(expr.render(), "x");
    }

    #[test]
    fn test_to_proto_literal_null() {
        let lit = LiteralExpression::null(DataType::Integer);
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::Null(_)) = literal.literal_type {
                // Expected
            } else {
                panic!("Expected null literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn test_to_proto_literal_boolean() {
        let lit = LiteralExpression::boolean(true);
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::Boolean(b)) = literal.literal_type
            {
                assert!(b);
            } else {
                panic!("Expected boolean literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn test_to_proto_literal_binary() {
        let lit = LiteralExpression::binary(vec![1, 2, 3]);
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::Binary(b)) = literal.literal_type {
                assert_eq!(b.as_ref(), [1, 2, 3]);
            } else {
                panic!("Expected binary literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn test_to_proto_literal_string() {
        let lit = LiteralExpression::string("hello");
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::String(s)) = literal.literal_type {
                assert_eq!(s, "hello");
            } else {
                panic!("Expected string literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn test_to_proto_literal_array() {
        let lit = LiteralExpression::Array {
            element_type: Box::new(DataType::Integer),
            elements: vec![LiteralExpression::int(1), LiteralExpression::int(2)],
        };
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::Array(arr)) = literal.literal_type
            {
                assert_eq!(arr.elements.len(), 2);
                #[allow(deprecated)]
                let element_type = arr.element_type;
                assert_eq!(element_type, Some(DataType::Integer.to_proto()));
            } else {
                panic!("Expected array literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn test_to_proto_unresolved_star_none() {
        let expr: Expression = ExpressionNode::UnresolvedStar(None, None).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::UnresolvedStar(star)) = proto.expr_type {
            assert!(star.unparsed_target.is_none());
        } else {
            panic!("Expected unresolved star expression type");
        }
    }

    #[test]
    fn test_to_proto_unresolved_star_with_target() {
        let expr: Expression =
            ExpressionNode::UnresolvedStar(Some("table.*".to_string()), None).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::UnresolvedStar(star)) = proto.expr_type {
            assert_eq!(star.unparsed_target, Some("table.*".to_string()));
        } else {
            panic!("Expected unresolved star expression type");
        }
    }

    #[test]
    fn test_to_proto_unresolved_regex() {
        let expr: Expression = ExpressionNode::UnresolvedRegex("`col_.*`".to_string(), None).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::UnresolvedRegex(regex)) = proto.expr_type {
            assert_eq!(regex.col_name, "`col_.*`");
        } else {
            panic!("Expected unresolved regex expression type");
        }
    }

    #[test]
    fn test_to_proto_direct_shuffle_partition_id() {
        let child = col("x");
        let expr: Expression = ExpressionNode::DirectShufflePartitionId(child).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::DirectShufflePartitionId(dspi)) = proto.expr_type {
            assert!(dspi.child.is_some());
        } else {
            panic!("Expected direct shuffle partition id expression type");
        }
    }

    #[test]
    fn test_to_proto_unresolved_extract_value() {
        let child = col("struct_col");
        let extraction = Expression::literal(LiteralExpression::string("field"));
        let ev = ExtractValue::new(child, extraction);
        let expr: Expression = ExpressionNode::UnresolvedExtractValue(Box::new(ev)).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::UnresolvedExtractValue(uev)) = proto.expr_type {
            assert!(uev.child.is_some());
            assert!(uev.extraction.is_some());
        } else {
            panic!("Expected unresolved extract value expression type");
        }
    }

    #[test]
    fn test_to_proto_update_fields() {
        let struct_expr = col("s");
        let value_expr = Expression::literal(LiteralExpression::int(42));
        let uf = UpdateFieldsExpr::new(struct_expr, "f1", Some(value_expr));
        let expr: Expression = ExpressionNode::UpdateFields(Box::new(uf)).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::UpdateFields(uf_proto)) = proto.expr_type {
            assert_eq!(uf_proto.field_name, "f1");
            assert!(uf_proto.value_expression.is_some());
        } else {
            panic!("Expected update fields expression type");
        }
    }

    #[test]
    fn test_to_proto_alias_with_metadata() {
        let alias = Alias::new(col("x"), "y").with_metadata("metadata_str".to_string());
        let expr: Expression = ExpressionNode::Alias(Box::new(alias)).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Alias(alias_proto)) = proto.expr_type {
            assert_eq!(alias_proto.name, vec!["y".to_string()]);
            assert_eq!(alias_proto.metadata, Some("metadata_str".to_string()));
        } else {
            panic!("Expected alias expression type");
        }
    }

    #[test]
    fn test_to_proto_cast_with_datatype() {
        let cast = Cast::new(col("x"), DataType::Integer);
        let expr: Expression = ExpressionNode::Cast(Box::new(cast)).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Cast(cast_proto)) = proto.expr_type {
            assert!(cast_proto.expr.is_some());
            assert!(cast_proto.cast_to_type.is_some());
        } else {
            panic!("Expected cast expression type");
        }
    }

    #[test]
    fn test_to_proto_cast_with_eval_mode_legacy() {
        let cast = Cast::new(col("x"), DataType::Integer).with_eval_mode(CastEvalMode::Legacy);
        let expr: Expression = ExpressionNode::Cast(Box::new(cast)).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Cast(cast_proto)) = proto.expr_type {
            assert_eq!(cast_proto.eval_mode, 1i32);
        } else {
            panic!("Expected cast expression type");
        }
    }

    #[test]
    fn test_to_proto_cast_with_eval_mode_ansi() {
        let cast = Cast::new(col("x"), DataType::Integer).with_eval_mode(CastEvalMode::Ansi);
        let expr: Expression = ExpressionNode::Cast(Box::new(cast)).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Cast(cast_proto)) = proto.expr_type {
            assert_eq!(cast_proto.eval_mode, 2i32);
        } else {
            panic!("Expected cast expression type");
        }
    }

    #[test]
    fn test_to_proto_cast_with_eval_mode_try() {
        let cast = Cast::new(col("x"), DataType::Integer).with_eval_mode(CastEvalMode::Try);
        let expr: Expression = ExpressionNode::Cast(Box::new(cast)).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Cast(cast_proto)) = proto.expr_type {
            assert_eq!(cast_proto.eval_mode, 3i32);
        } else {
            panic!("Expected cast expression type");
        }
    }

    #[test]
    fn test_to_proto_cast_str() {
        let cast = Cast::new_str(col("x"), "integer");
        let expr: Expression = ExpressionNode::Cast(Box::new(cast)).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Cast(cast_proto)) = proto.expr_type {
            assert!(cast_proto.cast_to_type.is_some());
        } else {
            panic!("Expected cast expression type");
        }
    }

    #[test]
    fn test_to_proto_sort_order() {
        let sort = SortOrder::asc_nulls_first(col("x"));
        let expr: Expression = ExpressionNode::SortOrder(Box::new(sort)).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::SortOrder(sort_proto)) = proto.expr_type {
            assert_eq!(sort_proto.direction, 1i32); // ASC
            assert_eq!(sort_proto.null_ordering, 1i32); // FIRST
        } else {
            panic!("Expected sort order expression type");
        }
    }

    #[test]
    fn test_to_proto_case_when() {
        let cw = CaseWhen::new(vec![(
            Expression::literal(LiteralExpression::boolean(true)),
            Expression::literal(LiteralExpression::int(1)),
        )])
        .with_else(Expression::literal(LiteralExpression::int(99)));
        let expr: Expression = ExpressionNode::CaseWhen(Box::new(cw)).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::UnresolvedFunction(func_proto)) = proto.expr_type {
            assert_eq!(func_proto.function_name, "when");
            assert_eq!(func_proto.arguments.len(), 3); // 2 branches + 1 else
        } else {
            panic!("Expected unresolved function expression type for case when");
        }
    }

    #[test]
    fn test_to_proto_sql_expression() {
        let expr: Expression =
            ExpressionNode::SQLExpression("SELECT * FROM table".to_string()).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::ExpressionString(es)) = proto.expr_type {
            assert_eq!(es.expression, "SELECT * FROM table");
        } else {
            panic!("Expected expression string type");
        }
    }

    #[test]
    fn test_to_proto_call_function() {
        let cf = CallFunctionWrapper::new("my_func", vec![col("a"), col("b")]);
        let expr: Expression = ExpressionNode::CallFunction(Box::new(cf)).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::CallFunction(cf_proto)) = proto.expr_type {
            assert_eq!(cf_proto.function_name, "my_func");
            assert_eq!(cf_proto.arguments.len(), 2);
        } else {
            panic!("Expected call function expression type");
        }
    }

    #[test]
    fn test_to_proto_lambda_function() {
        let lf = LambdaFunction::new(col("x"), vec![UnresolvedNamedLambdaVariable::new("x")]);
        let expr: Expression = ExpressionNode::LambdaFunction(Box::new(lf)).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::LambdaFunction(lf_proto)) = proto.expr_type {
            assert!(lf_proto.function.is_some());
            assert_eq!(lf_proto.arguments.len(), 1);
        } else {
            panic!("Expected lambda function expression type");
        }
    }

    #[test]
    fn test_to_proto_unresolved_named_lambda_variable() {
        let var = UnresolvedNamedLambdaVariable::new("x");
        let expr: Expression = ExpressionNode::UnresolvedNamedLambdaVariable(var).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::UnresolvedNamedLambdaVariable(var_proto)) =
            proto.expr_type
        {
            assert_eq!(var_proto.name_parts.len(), 1);
        } else {
            panic!("Expected unresolved named lambda variable expression type");
        }
    }

    #[test]
    fn test_window_expression_to_proto() {
        let we = WindowExpressionWrapper::new(
            Expression::function("sum", vec![col("x")]),
            vec![col("group_col")],
            vec![SortOrder::asc_nulls_last(col("sort_col"))],
            Some((
                1u32,
                FrameBoundary::UnboundedPreceding,
                FrameBoundary::CurrentRow,
            )),
        );
        let expr: Expression = ExpressionNode::WindowExpression(Box::new(we)).into();
        let proto = expr.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Window(window_proto)) = proto.expr_type {
            assert!(window_proto.window_function.is_some());
            assert_eq!(window_proto.partition_spec.len(), 1);
            assert_eq!(window_proto.order_spec.len(), 1);
            assert!(window_proto.frame_spec.is_some());
        } else {
            panic!("Expected window expression type");
        }
    }

    #[test]
    fn test_literal_byte_to_proto() {
        let lit = LiteralExpression::Byte(42);
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::Byte(b)) = literal.literal_type {
                assert_eq!(b, 42);
            } else {
                panic!("Expected byte literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn test_literal_short_to_proto() {
        let lit = LiteralExpression::Short(1000);
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::Short(s)) = literal.literal_type {
                assert_eq!(s, 1000);
            } else {
                panic!("Expected short literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn test_literal_float_to_proto() {
        let lit = LiteralExpression::Float(3.14);
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::Float(f)) = literal.literal_type {
                assert!((f - 3.14).abs() < 0.01);
            } else {
                panic!("Expected float literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn test_literal_double_to_proto() {
        let lit = LiteralExpression::Double(2.71828);
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::Double(d)) = literal.literal_type {
                assert!((d - 2.71828).abs() < 0.00001);
            } else {
                panic!("Expected double literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn test_literal_time_to_proto() {
        let lit = LiteralExpression::Time {
            nano: 3600000000000i64,
            precision: 9,
        };
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::Time(t)) = literal.literal_type {
                assert_eq!(t.nano, 3600000000000i64);
                assert_eq!(t.precision, Some(9));
            } else {
                panic!("Expected time literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn test_literal_timestamp_ntz_to_proto() {
        let lit = LiteralExpression::TimestampNtz(1693526400000000);
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::TimestampNtz(ts)) =
                literal.literal_type
            {
                assert_eq!(ts, 1693526400000000);
            } else {
                panic!("Expected timestamp ntz literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }
}
