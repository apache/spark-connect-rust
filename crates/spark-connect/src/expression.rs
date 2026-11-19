//! Expression tree mirroring PySpark's `pyspark.sql.connect.expressions`.
//!
//! Defines the Expression hierarchy and conversion functions to Spark Connect protobufs.

use std::cell::{Cell, RefCell};
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
/// here, before any request is built (see `with_expression_depth_bound`). PySpark's own point is
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
    /// The error of the first [`ExpressionNode::Unbuildable`] that this thread serialized since
    /// the start of the innermost [`with_expression_depth_bound`], which raises it.
    static UNBUILDABLE_EXPRESSION_ERROR: RefCell<Option<SparkError>> = const { RefCell::new(None) };
    /// The error of the first [`ExpressionNode::Unbuildable`] that this thread rendered since the
    /// start of the innermost [`Expression::try_render`], which raises it; and whether it renders
    /// in one.
    static UNRENDERABLE_EXPRESSION_ERROR: RefCell<Option<SparkError>> = const { RefCell::new(None) };
    static RENDERING_TRIED: Cell<bool> = const { Cell::new(false) };
    /// The commands of the UDFs that this thread serializes since the start of the innermost
    /// [`with_expression_depth_bound`], which their client pickles (see [`pickled_command`]).
    static UDF_COMMANDS: RefCell<UdfCommands> = const { RefCell::new(UdfCommands::new()) };
}

/// The commands of the UDFs of a plan that a request builds, of their
/// [`CommandPickler`](crate::udf::CommandPickler)s.
///
/// PySpark's client pickles the function of each UDF as it builds the plan, in the order of the
/// build, and the first error of the build, of a pickling or of another expression that it fails
/// to build, is raised. Here the plan is built twice when it has UDFs of picklers: the first build
/// meets the picklers, in that order, up to the first expression that it does not build; they are
/// called in turn once it is done, on the stack of the caller of the request, outside the
/// recursion of the build, so the Python code of a pickling runs where PySpark's does; and when
/// none fails and no expression failed after them, the second build takes their commands, in the
/// same order. A build is deterministic, so the two meet the same UDFs.
///
/// PySpark's `WindowExpression.to_plan` warns of a window of no partitions as it builds it, in
/// the same order: the first build warns of one that comes before the first pickler as it meets
/// it, and of one after a pickler in its turn among the picklings, so a pickling that fails before
/// it raises its error without the warning, and the error of the warning (of an `"error"` filter
/// of the warnings) is raised before the picklings after it. The second build warns no more.
#[derive(Default)]
struct UdfCommands {
    /// The picklers that the first build met, in the order of the build.
    met: Vec<crate::udf::CommandPickler>,
    /// The windows of no partitions that the first build met after a pickler, each the number of
    /// picklers that it met before it, of which [`pickle_in_turn`] warns in their turn.
    windows: Vec<usize>,
    /// The commands that they gave, which the second build takes in turn.
    pickled: std::collections::VecDeque<Vec<u8>>,
    /// Whether this is the second build, which takes `pickled`.
    pickled_build: bool,
}

impl UdfCommands {
    const fn new() -> Self {
        UdfCommands {
            met: Vec::new(),
            windows: Vec::new(),
            pickled: std::collections::VecDeque::new(),
            pickled_build: false,
        }
    }
}

/// The command of the UDF of `pickler` in the plan that this thread builds. In a request (a
/// [`with_expression_depth_bound`]), the first build of the plan meets the pickler, unless an
/// expression before it failed to build, as PySpark's client raises its error before it pickles
/// the functions after it; the second build takes the command that it gave (see [`UdfCommands`]).
/// Outside a request, as the Rust API's `to_proto` builds a plan, it is pickled now, and is empty
/// if it fails.
pub(crate) fn pickled_command(pickler: &crate::udf::CommandPickler) -> Vec<u8> {
    if !EXPRESSION_DEPTH_BOUNDED.with(Cell::get) {
        return pickler.pickle().unwrap_or_default();
    }
    UDF_COMMANDS.with(|commands| {
        let mut commands = commands.borrow_mut();
        if commands.pickled_build {
            return commands.pickled.pop_front().unwrap_or_default();
        }
        if UNBUILDABLE_EXPRESSION_ERROR.with(|slot| slot.borrow().is_none()) {
            commands.met.push(pickler.clone());
        }
        Vec::new()
    })
}

/// Warn of a window of no partitions that this thread builds, as PySpark's
/// `WindowExpression.to_plan` warns once it has built its window function: now, unless the first
/// build of a request met a pickler before it, after which it is warned of in its turn among the
/// picklings, unless an expression before it failed to build; and not in the second build, which
/// warns no more (see [`UdfCommands`]).
fn warn_of_a_window_without_partitions() {
    let now = UDF_COMMANDS.with(|commands| {
        let mut commands = commands.borrow_mut();
        if commands.pickled_build {
            return false;
        }
        if commands.met.is_empty() {
            return true;
        }
        if UNBUILDABLE_EXPRESSION_ERROR.with(|slot| slot.borrow().is_none()) {
            let picklers_before = commands.met.len();
            commands.windows.push(picklers_before);
        }
        false
    });
    if now {
        tell_plan_build_listener(|listener| listener.window_without_partitions());
    }
}

/// Tell the listener of the plans of the process ([`crate::plan_build`]), if any, of a window of
/// no partitions that the first build of a request met after a pickler: its error is the
/// request's.
fn warn_of_a_met_window() -> Result<()> {
    match crate::plan_build::listener() {
        Some(listener) => listener.window_without_partitions(),
        None => Ok(()),
    }
}

/// The commands of `picklers`, each pickled in turn, each after the warnings of the `windows` of
/// no partitions that the build met before it (each window is the number of picklers met before
/// it), and then the warnings of the windows after them; or the error of the first pickling or
/// warning that fails, after which none is pickled or warned of, as PySpark's build stops at it.
fn pickle_in_turn(
    picklers: Vec<crate::udf::CommandPickler>,
    windows: &[usize],
) -> Result<std::collections::VecDeque<Vec<u8>>> {
    let mut windows = windows.iter().peekable();
    let mut pickled = std::collections::VecDeque::with_capacity(picklers.len());
    for (picklers_before, pickler) in picklers.iter().enumerate() {
        while windows
            .next_if(|&&window| window <= picklers_before)
            .is_some()
        {
            warn_of_a_met_window()?;
        }
        pickled.push_back(pickler.pickle()?);
    }
    for _ in windows {
        warn_of_a_met_window()?;
    }
    Ok(pickled)
}

impl ExpressionNode {
    /// The expressions, the UDFs and the plans right below this node, whose picklers
    /// [`crate::udf::exclusive_pickler_objects`] visits.
    pub(crate) fn push_pickler_holders<'a>(
        &'a self,
        pending: &mut Vec<crate::udf::PicklerHolder<'a>>,
    ) {
        use crate::udf::PicklerHolder::{Expression as Of, Subquery, Udf};
        match self {
            ExpressionNode::Literal(_)
            | ExpressionNode::ColumnReference(_)
            | ExpressionNode::UnresolvedStar(..)
            | ExpressionNode::UnresolvedRegex(..)
            | ExpressionNode::SQLExpression(_)
            | ExpressionNode::UnresolvedNamedLambdaVariable(_)
            | ExpressionNode::Sent(_) => {}
            ExpressionNode::UnresolvedFunction(function) => {
                pending.extend(function.args.iter().map(Of));
            }
            ExpressionNode::Alias(alias) => pending.push(Of(&alias.child)),
            ExpressionNode::Cast(cast) => pending.push(Of(&cast.child)),
            ExpressionNode::DirectShufflePartitionId(child) => pending.push(Of(child)),
            ExpressionNode::SortOrder(sort) => pending.push(Of(&sort.child)),
            ExpressionNode::CaseWhen(case_when) => {
                for (condition, value) in &case_when.branches {
                    pending.extend([Of(condition), Of(value)]);
                }
                pending.extend(case_when.else_expr.iter().map(Of));
            }
            ExpressionNode::UnresolvedExtractValue(extract) => {
                pending.extend([Of(&extract.child), Of(&extract.extraction)]);
            }
            ExpressionNode::UpdateFields(update) => {
                pending.push(Of(&update.struct_expression));
                pending.extend(update.value_expression.iter().map(Of));
            }
            ExpressionNode::CallFunction(call) => pending.extend(call.arguments.iter().map(Of)),
            ExpressionNode::WindowExpression(window) => {
                pending.push(Of(&window.window_function));
                pending.extend(window.partition_spec.iter().map(Of));
                pending.extend(window.order_spec.iter().map(|sort| Of(&sort.child)));
            }
            ExpressionNode::LambdaFunction(lambda) => pending.push(Of(&lambda.function)),
            ExpressionNode::CommonInlineUserDefinedFunction(udf) => pending.push(Udf(udf)),
            ExpressionNode::NamedArgument(argument) => pending.push(Of(&argument.value)),
            ExpressionNode::Subquery(subquery) => pending.push(Subquery(subquery)),
            ExpressionNode::Unbuildable(unbuildable) => pending.push(Of(&unbuildable.expression)),
            ExpressionNode::PlanBuildOrigin(made) => pending.push(Of(&made.expression)),
        }
    }

    /// The expressions, the UDFs and the subqueries right below this node, which a deep copy of
    /// the functions of its UDFs replaces (see [`crate::udf::PicklerRebuild`]).
    pub(crate) fn parts_mut(&mut self) -> Vec<ExpressionPart<'_>> {
        use ExpressionPart::{Expression as Of, Subquery, Udf};
        match self {
            ExpressionNode::Literal(_)
            | ExpressionNode::ColumnReference(_)
            | ExpressionNode::UnresolvedStar(..)
            | ExpressionNode::UnresolvedRegex(..)
            | ExpressionNode::SQLExpression(_)
            | ExpressionNode::UnresolvedNamedLambdaVariable(_)
            | ExpressionNode::Sent(_) => Vec::new(),
            ExpressionNode::UnresolvedFunction(function) => {
                function.args.iter_mut().map(Of).collect()
            }
            ExpressionNode::Alias(alias) => vec![Of(&mut alias.child)],
            ExpressionNode::Cast(cast) => vec![Of(&mut cast.child)],
            ExpressionNode::DirectShufflePartitionId(child) => vec![Of(child)],
            ExpressionNode::SortOrder(sort) => vec![Of(&mut sort.child)],
            ExpressionNode::CaseWhen(case_when) => {
                let case_when = &mut **case_when;
                let mut parts = Vec::new();
                for (condition, value) in case_when.branches.iter_mut() {
                    parts.extend([Of(condition), Of(value)]);
                }
                parts.extend(case_when.else_expr.iter_mut().map(Of));
                parts
            }
            ExpressionNode::UnresolvedExtractValue(extract) => {
                let extract = &mut **extract;
                vec![Of(&mut extract.child), Of(&mut extract.extraction)]
            }
            ExpressionNode::UpdateFields(update) => {
                let update = &mut **update;
                let mut parts = vec![Of(&mut update.struct_expression)];
                parts.extend(update.value_expression.iter_mut().map(Of));
                parts
            }
            ExpressionNode::CallFunction(call) => call.arguments.iter_mut().map(Of).collect(),
            ExpressionNode::WindowExpression(window) => {
                let window = &mut **window;
                let mut parts = vec![Of(&mut window.window_function)];
                parts.extend(window.partition_spec.iter_mut().map(Of));
                parts.extend(window.order_spec.iter_mut().map(|sort| Of(&mut sort.child)));
                parts
            }
            ExpressionNode::LambdaFunction(lambda) => vec![Of(&mut lambda.function)],
            ExpressionNode::CommonInlineUserDefinedFunction(udf) => vec![Udf(udf)],
            ExpressionNode::NamedArgument(argument) => vec![Of(&mut argument.value)],
            ExpressionNode::Subquery(subquery) => vec![Subquery(subquery)],
            ExpressionNode::Unbuildable(unbuildable) => vec![Of(&mut unbuildable.expression)],
            ExpressionNode::PlanBuildOrigin(made) => vec![Of(&mut made.expression)],
        }
    }
}

/// A part of an expression node, of [`ExpressionNode::parts_mut`].
pub(crate) enum ExpressionPart<'a> {
    Expression(&'a mut Expression),
    Udf(&'a mut CommonInlineUserDefinedFunctionExpression),
    Subquery(&'a mut SubqueryExpression),
}

/// Record `error`, of an expression or a plan (`LogicalPlan::Unbuildable`) that the serialization
/// of this thread does not build, as the error that its [`with_expression_depth_bound`] raises,
/// unless one is recorded already: the first one, in the order in which PySpark's client builds
/// the plan, is raised.
pub(crate) fn record_unbuildable(error: &SparkError) {
    if EXPRESSION_DEPTH_BOUNDED.with(Cell::get) {
        UNBUILDABLE_EXPRESSION_ERROR.with(|slot| {
            slot.borrow_mut().get_or_insert_with(|| error.clone());
        });
    }
}

/// Record `error`, of an expression that the rendering of this thread does not render, as the
/// error that its [`Expression::try_render`] raises, unless one is recorded already.
fn record_unrenderable(error: &SparkError) {
    if RENDERING_TRIED.with(Cell::get) {
        UNRENDERABLE_EXPRESSION_ERROR.with(|slot| {
            slot.borrow_mut().get_or_insert_with(|| error.clone());
        });
    }
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
    outer_unbuildable: Option<SparkError>,
    outer_commands: UdfCommands,
}

impl DepthBoundScope {
    fn enter() -> Self {
        DepthBoundScope {
            outer_bounded: EXPRESSION_DEPTH_BOUNDED.with(|bounded| bounded.replace(true)),
            outer_deepest: DEEPEST_EXPRESSION_LEVEL.with(|deepest| deepest.replace(0)),
            outer_unbuildable: UNBUILDABLE_EXPRESSION_ERROR.with(|slot| slot.borrow_mut().take()),
            outer_commands: UDF_COMMANDS
                .with(|commands| std::mem::take(&mut *commands.borrow_mut())),
        }
    }

    /// The error of the first expression that was not built since the scope was entered.
    fn unbuildable(&self) -> Option<SparkError> {
        UNBUILDABLE_EXPRESSION_ERROR.with(|slot| slot.borrow_mut().take())
    }

    /// The deepest level since the scope was entered.
    fn deepest(&self) -> usize {
        DEEPEST_EXPRESSION_LEVEL.with(Cell::get)
    }

    /// The picklers that the build met since the scope was entered, and the windows of no
    /// partitions that it met after them (see [`UdfCommands`]).
    fn met_picklers_and_windows(&self) -> (Vec<crate::udf::CommandPickler>, Vec<usize>) {
        UDF_COMMANDS.with(|commands| {
            let mut commands = commands.borrow_mut();
            (
                std::mem::take(&mut commands.met),
                std::mem::take(&mut commands.windows),
            )
        })
    }

    /// Make the next build take `pickled`, the commands of the picklers that the first met.
    fn build_with(&self, pickled: std::collections::VecDeque<Vec<u8>>) {
        UDF_COMMANDS.with(|commands| {
            *commands.borrow_mut() = UdfCommands {
                pickled,
                pickled_build: true,
                ..UdfCommands::new()
            }
        });
    }
}

impl Drop for DepthBoundScope {
    fn drop(&mut self) {
        EXPRESSION_DEPTH_BOUNDED.with(|bounded| bounded.set(self.outer_bounded));
        DEEPEST_EXPRESSION_LEVEL.with(|deepest| deepest.set(deepest.get().max(self.outer_deepest)));
        let outer = self.outer_unbuildable.take();
        UNBUILDABLE_EXPRESSION_ERROR.with(|slot| *slot.borrow_mut() = outer);
        let outer = std::mem::take(&mut self.outer_commands);
        UDF_COMMANDS.with(|commands| *commands.borrow_mut() = outer);
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
/// recursing too, so it must not be deeper. The commands of the UDFs of picklers are pickled as
/// [`UdfCommands`] tells, so `serialize` may run twice.
pub(crate) fn with_expression_depth_bound<T>(serialize: impl Fn() -> T) -> Result<T> {
    with_expression_levels(serialize).map(|(value, _)| value)
}

/// As [`with_expression_depth_bound`], and the levels that the expressions of `serialize` reach.
fn with_expression_levels<T>(serialize: impl Fn() -> T) -> Result<(T, usize)> {
    let (value, levels, unbuildable) = with_expression_levels_and_unbuildable(serialize)?;
    // An expression that PySpark's client fails to build, which it raises as it builds the plan.
    if let Some(error) = unbuildable {
        return Err(error);
    }
    Ok((value, levels))
}

/// As [`with_expression_levels`], of the error of the first expression that `serialize` does not
/// build ([`Expression::unbuildable`]), or of the first command that does not pickle, kept apart,
/// rather than raised.
fn with_expression_levels_and_unbuildable<T>(
    serialize: impl Fn() -> T,
) -> Result<(T, usize, Option<SparkError>)> {
    let scope = DepthBoundScope::enter();
    let mut value = serialize();
    let levels = scope.deepest();
    let mut unbuildable = scope.unbuildable();
    let (picklers, windows) = scope.met_picklers_and_windows();
    if levels > MAX_EXPRESSION_DEPTH {
        // The request fails with its `RecursionError`, which stands for the errors of the build,
        // and no function is pickled: the windows after a pickler warn, as those before it did in
        // the build, up to the first that fails.
        for _ in &windows {
            if warn_of_a_met_window().is_err() {
                break;
            }
        }
        return Err(SparkError::recursion_msg(format!(
            "maximum recursion depth exceeded while serializing an expression; an expression \
             may be at most {MAX_EXPRESSION_DEPTH} levels deep"
        )));
    }
    if !picklers.is_empty() {
        // The picklers, and the windows after the first of them, come before the first
        // expression that failed to build, if any: the first of them that fails is the error,
        // else that expression's.
        match pickle_in_turn(picklers, &windows) {
            Err(error) => unbuildable = Some(error),
            Ok(pickled) if unbuildable.is_none() => {
                drop(value);
                scope.build_with(pickled);
                value = serialize();
                unbuildable = scope.unbuildable();
            }
            Ok(_) => {}
        }
    }
    Ok((value, levels, unbuildable))
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
    /// An expression that PySpark's client builds when a request sends the plan, and fails to
    /// build: the request raises its error. Not a stable API (see
    /// [`Expression::unbuildable`]).
    #[doc(hidden)]
    Unbuildable(Box<UnbuildableExpression>),
    /// An expression that PySpark's client makes when it builds the plan, rather than when the
    /// Column is made, so that its origin is that of its API function at that time. Not a stable
    /// API (see [`Expression::with_plan_build_origin`]).
    #[doc(hidden)]
    PlanBuildOrigin(Box<PlanBuildOrigin>),
}

/// An expression of [`ExpressionNode::Unbuildable`]: `expression`, as this client builds it, and
/// the errors that PySpark's client raises when it builds the plan of it, `before` it builds
/// `expression` (the names of a `ColumnAlias`, which its `to_plan` sets first) and `after` (its
/// metadata, which it sets last), and the error of its `__repr__` (`unrenderable`, PySpark's
/// `','.join` of names that are not all `str`). Not a stable API.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct UnbuildableExpression {
    pub expression: Expression,
    pub before: Option<SparkError>,
    pub after: Option<SparkError>,
    pub unrenderable: Option<SparkError>,
}

impl PartialEq for UnbuildableExpression {
    fn eq(&self, other: &Self) -> bool {
        fn same(left: &Option<SparkError>, right: &Option<SparkError>) -> bool {
            match (left, right) {
                (Some(left), Some(right)) => {
                    left.kind == right.kind && left.message() == right.message()
                }
                (None, None) => true,
                _ => false,
            }
        }
        self.expression == other.expression
            && same(&self.before, &other.before)
            && same(&self.after, &other.after)
            && same(&self.unrenderable, &other.unrenderable)
    }
}

impl UnbuildableExpression {
    /// The proto of `expression`, as a request builds it: the errors are recorded for the request
    /// to raise, in the order in which PySpark's client raises them.
    fn to_proto(&self) -> proto::Expression {
        if let Some(error) = &self.before {
            record_unbuildable(error);
        }
        let expr = self.expression.to_proto();
        if let Some(error) = &self.after {
            record_unbuildable(error);
        }
        expr
    }

    /// The rendering of `expression`, after which PySpark's `__repr__` fails, if it does.
    fn render(&self) -> String {
        let text = self.expression.render();
        if let Some(error) = &self.unrenderable {
            record_unrenderable(error);
        }
        text
    }
}

/// An expression of [`ExpressionNode::PlanBuildOrigin`]: `expression`, which PySpark's client
/// makes by the API function `fragment` when it builds the plan, such as the `and` of the
/// conditions of a join of a list of Columns, of `__and__`. Not a stable API.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct PlanBuildOrigin {
    pub expression: Expression,
    pub fragment: String,
}

impl PlanBuildOrigin {
    /// The proto of `expression`, as a request builds it, of the origin that the listener of the
    /// plans gives for `fragment` now ([`crate::plan_build::PlanBuildListener::origin`]), and of
    /// none without one: PySpark's client takes it before it builds the expressions below.
    fn to_proto(&self) -> proto::Expression {
        let origin = tell_plan_build_listener(|listener| listener.origin(&self.fragment)).flatten();
        let mut expr = self.expression.to_proto();
        expr.common = origin.map(|origin| origin_proto(&origin));
        expr
    }
}

/// Tell the listener of the plans of the process ([`crate::plan_build`]) of what PySpark's
/// client does as it builds the plan, by `tell`, unless the request that builds it has failed
/// already, as PySpark's client stops at the first error. The error of `tell` is the request's,
/// as the error of an expression that PySpark fails to build. None without a listener.
fn tell_plan_build_listener<T>(
    tell: impl FnOnce(&dyn crate::plan_build::PlanBuildListener) -> Result<T>,
) -> Option<T> {
    let listener = crate::plan_build::listener()?;
    if UNBUILDABLE_EXPRESSION_ERROR.with(|slot| slot.borrow().is_some()) {
        return None;
    }
    match tell(listener.as_ref()) {
        Ok(value) => Some(value),
        Err(error) => {
            record_unbuildable(&error);
            None
        }
    }
}

/// The common fields of an expression of `origin`, as PySpark's `_create_proto_expression` sends
/// them.
fn origin_proto(origin: &crate::origin::Origin) -> proto::ExpressionCommon {
    proto::ExpressionCommon {
        origin: Some(proto::Origin {
            function: Some(proto::origin::Function::PythonOrigin(proto::PythonOrigin {
                fragment: origin.fragment.clone(),
                call_site: origin.call_site.clone(),
            })),
        }),
    }
}

/// An expression of [`ExpressionNode::Sent`]: the `spark.connect.Expression` of a Column as it
/// was sent, its rendering, as `Column<'...'>` shows it, and its levels, as
/// [`Expression::try_to_proto_with_levels`] counts them when the Column was pickled, and the
/// plans that its subqueries refer to.
#[derive(Debug, Clone, PartialEq)]
pub struct SentExpression {
    pub expression: proto::Expression,
    pub rendered: String,
    pub levels: usize,
    pub subqueries: SentSubqueries,
}

/// The plans that the subqueries of a [`SentExpression`] refer to by their plan IDs, which the
/// relation that holds the expression sends in its `WithRelations`, as it sends the plan of a
/// [`SubqueryExpression`]: each the relation of a plan as it was sent
/// ([`Expression::subquery_relations`]), of its levels, which keeps its plan ID in plan IDs of its
/// own, as PySpark's unpickled Column holds the plan objects of its subqueries, which keep their
/// `_plan_id`. Two are equal when they are of the same plans.
#[derive(Debug, Clone, Default)]
pub struct SentSubqueries(Vec<(Arc<LogicalPlan>, Arc<crate::plan::PlanIds>)>);

impl SentSubqueries {
    /// The plans of `relations`, the relations of the plans that the subqueries of an expression
    /// refer to, each of its levels. A relation of more levels than
    /// [`crate::plan::MAX_PLAN_DEPTH`] is a `RecursionError`, as the API raises it when it sends
    /// one.
    fn of(relations: Vec<(proto::Relation, usize)>) -> Result<Self> {
        relations
            .into_iter()
            .map(|(relation, levels)| {
                if levels > crate::plan::MAX_PLAN_DEPTH {
                    return Err(crate::plan::too_deep_a_plan(levels));
                }
                let plan_id = relation.common.as_ref().and_then(|common| common.plan_id);
                let plan = Arc::new(LogicalPlan::Raw {
                    relation: Box::new(relation),
                    owner: None,
                    depth: levels,
                    observations: Vec::new(),
                });
                let ids = Arc::new(crate::plan::PlanIds::new());
                if let Some(plan_id) = plan_id {
                    ids.assign(&plan, plan_id);
                }
                Ok((plan, ids))
            })
            .collect::<Result<_>>()
            .map(SentSubqueries)
    }
}

impl PartialEq for SentSubqueries {
    fn eq(&self, other: &Self) -> bool {
        // Plans have no equality: the subqueries of the same plans are equal.
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(&other.0)
                .all(|((plan, _), (other, _))| Arc::ptr_eq(plan, other))
    }
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
    /// The plan IDs of the session of the DataFrame of `plan`: the relation that contains the
    /// subquery sends `plan` in its `WithRelations` with them, also when it is of another session,
    /// and a deep copy of the subquery gives the plan that it rebuilds the plan ID of `plan` in
    /// them.
    pub(crate) plan_ids: Option<Arc<crate::plan::PlanIds>>,
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
            plan_ids: None,
        }
    }

    /// The holders of the picklers of this subquery: its plan, which a walk that visits only
    /// what it alone holds visits when nothing else shares it, and its expressions.
    pub(crate) fn push_pickler_holders<'a>(
        &'a self,
        pending: &mut Vec<crate::udf::PicklerHolder<'a>>,
        exclusive: bool,
    ) {
        use crate::udf::PicklerHolder::{Expression as Of, Plan};
        if !exclusive || Arc::strong_count(&self.plan) == 1 {
            pending.push(Plan(&self.plan));
        }
        pending.extend(self.partition_spec.iter().map(Of));
        pending.extend(self.order_spec.iter().map(Of));
        pending.extend(self.in_subquery_values.iter().map(Of));
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

        crate::plan::refer_to(&self.plan, self.plan_ids.as_ref());
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

    /// Whether no other expression shares the node of this one.
    pub(crate) fn is_exclusive(&self) -> bool {
        self.is_held_by(1)
    }

    /// Whether `holders` expressions, this one and its clones, hold the node of this one.
    pub(crate) fn is_held_by(&self, holders: usize) -> bool {
        Arc::strong_count(&self.node) == holders
    }

    /// The address of the node of this expression, which the expressions that share it have too.
    pub(crate) fn node_address(&self) -> usize {
        Arc::as_ptr(&self.node) as usize
    }

    /// The expression of `node`, of the origin of this one.
    pub(crate) fn of_node(&self, node: ExpressionNode) -> Expression {
        Expression {
            node: Arc::new(node),
            origin: self.origin.clone(),
        }
    }

    /// The picklers of the UDFs of this expression, below it and in the plans of its subqueries,
    /// each once. Not a stable API.
    #[doc(hidden)]
    pub fn picklers(&self) -> Vec<crate::udf::CommandPickler> {
        crate::udf::all_picklers(vec![crate::udf::PicklerHolder::Expression(self)])
    }

    /// This expression with the copies `copies` of its picklers, of the copies of their objects,
    /// as a deep copy of PySpark's expression copies the functions of its UDFs: the nodes above a
    /// pickler of a copy are new, and the others shared. Not a stable API.
    #[doc(hidden)]
    pub fn with_pickler_copies(&self, copies: &crate::udf::PicklerCopies) -> Expression {
        crate::udf::PicklerRebuild::new(copies, &[])
            .expression(self)
            .unwrap_or_else(|| self.clone())
    }

    /// The objects of the picklers of the UDFs of this expression that it alone holds, which a
    /// Python Column tells its garbage collector of (see
    /// [`crate::udf::exclusive_pickler_objects`]). Not a stable API.
    #[doc(hidden)]
    pub fn exclusive_pickler_objects(&self) -> Vec<&(dyn std::any::Any + Send + Sync)> {
        crate::udf::exclusive_pickler_objects(vec![crate::udf::PicklerHolder::Expression(self)])
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
                    CastTarget::TypeStr(s) | CastTarget::Shown(_, s) => s.clone(),
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
            ExpressionNode::Unbuildable(unbuildable) => unbuildable.render(),
            ExpressionNode::PlanBuildOrigin(made) => made.expression.render(),
        }
    }

    /// As [`Expression::render`], and the error of PySpark's `__repr__` of an expression that
    /// it fails to render (an [`Expression::unbuildable`] of an `unrenderable` error), which a
    /// Column's `__repr__` raises: of the first one, in the order in which the `__repr__` of the
    /// expressions render them. Not a stable API.
    #[doc(hidden)]
    pub fn try_render(&self) -> Result<String> {
        let outer_tried = RENDERING_TRIED.with(|tried| tried.replace(true));
        let outer_error = UNRENDERABLE_EXPRESSION_ERROR.with(|slot| slot.borrow_mut().take());
        let text = self.render();
        let error = UNRENDERABLE_EXPRESSION_ERROR.with(|slot| slot.borrow_mut().take());
        RENDERING_TRIED.with(|tried| tried.set(outer_tried));
        UNRENDERABLE_EXPRESSION_ERROR.with(|slot| *slot.borrow_mut() = outer_error);
        match error {
            Some(error) => Err(error),
            None => Ok(text),
        }
    }

    /// `expression`, which PySpark's client fails to build when a request sends its plan: the
    /// request raises `before` before it builds `expression`, or `after` after it, as PySpark's
    /// `ColumnAlias.to_plan` sets its names before it builds its child, and its metadata after;
    /// and [`Expression::try_render`] raises `unrenderable`. `expression` itself, of no error.
    /// Not a stable API.
    #[doc(hidden)]
    pub fn unbuildable(
        expression: Expression,
        before: Option<SparkError>,
        after: Option<SparkError>,
        unrenderable: Option<SparkError>,
    ) -> Expression {
        if before.is_none() && after.is_none() && unrenderable.is_none() {
            return expression;
        }
        Expression::new(ExpressionNode::Unbuildable(Box::new(
            UnbuildableExpression {
                expression,
                before,
                after,
                unrenderable,
            },
        )))
    }

    /// `expression`, which PySpark's client makes by the API function `fragment` when it builds
    /// the plan, rather than when the Column is made: a request sends it of the origin that the
    /// listener of the plans ([`crate::plan_build`]) gives for `fragment` as it builds the plan,
    /// and of none without one. So the `and` of the conditions of a join of a list of Columns
    /// is of `__and__` at its call site then, as PySpark's `functools.reduce(lambda c1, c2:
    /// c1 & c2, on)` makes it at each plan build. Not a stable API.
    #[doc(hidden)]
    pub fn with_plan_build_origin(expression: Expression, fragment: &str) -> Expression {
        Expression::new(ExpressionNode::PlanBuildOrigin(Box::new(PlanBuildOrigin {
            expression,
            fragment: fragment.to_string(),
        })))
    }

    /// The expression that a request builds of this one: that of an [`Expression::unbuildable`],
    /// whose error the request raises, else this one. Not a stable API.
    #[doc(hidden)]
    pub fn built(&self) -> &Expression {
        match self.node() {
            ExpressionNode::Unbuildable(unbuildable) => unbuildable.expression.built(),
            _ => self,
        }
    }

    /// Converts the expression to a Spark Connect protobuf expression, with its origin, as
    /// PySpark's `Expression._create_proto_expression` sends it.
    pub fn to_proto(&self) -> proto::Expression {
        // An unbuildable expression is the proto of its expression, of no level of its own, and
        // so is an expression of an origin of the plan build.
        match self.node() {
            ExpressionNode::Unbuildable(unbuildable) => return unbuildable.to_proto(),
            ExpressionNode::PlanBuildOrigin(made) => return made.to_proto(),
            _ => {}
        }
        // A level serializes the levels below it: grow the stack for an expression of many
        // levels, and count them for `with_expression_depth_bound`.
        with_stack_for_a_level(|| {
            let level = ExpressionLevel::enter();
            // A sent expression has the levels that it was pickled with, below this one.
            let deepest = match self.node() {
                ExpressionNode::Sent(sent) => {
                    let deepest = level.0 + sent.levels.saturating_sub(1);
                    DEEPEST_EXPRESSION_LEVEL
                        .with(|current| current.set(current.get().max(deepest)));
                    deepest
                }
                _ => level.0,
            };
            if deepest > MAX_EXPRESSION_DEPTH && EXPRESSION_DEPTH_BOUNDED.with(Cell::get) {
                // The request fails with a `RecursionError`: stop here, so that the proto of the
                // expression is no deeper than the bound.
                return proto::Expression::default();
            }
            self.to_proto_with_origin()
        })
    }

    /// Converts the expression to a Spark Connect protobuf expression as a request serializes
    /// it, outside the plan of a relation: the expressions of a command, such as the condition of
    /// a merge or of an overwrite, and of a pickled Column. One deeper than
    /// [`MAX_EXPRESSION_DEPTH`] is a `RecursionError`, as PySpark's serializer raises it before
    /// any request is sent, and its proto is not built past the bound: prost encodes and drops a
    /// proto by recursing, so a deeper one could overflow the stack there.
    pub fn try_to_proto(&self) -> Result<proto::Expression> {
        with_expression_depth_bound(|| self.to_proto())
    }

    /// As [`Expression::try_to_proto`], and the levels of the expression, as a pickled Column
    /// records them, so that its unpickled copy, [`Expression::sent`], is as deep as it was.
    pub fn try_to_proto_with_levels(&self) -> Result<(proto::Expression, usize)> {
        with_expression_levels(|| self.to_proto())
    }

    /// As [`Expression::try_to_proto_with_levels`], of an expression that a request fails to
    /// build too (of an [`Expression::unbuildable`]): its proto, of what this client builds of
    /// it, its levels, and the error that a request of it raises, which a pickled Column keeps,
    /// as PySpark's Column pickles and fails when the plan of its unpickled copy is built. Not a
    /// stable API.
    #[doc(hidden)]
    pub fn try_to_proto_with_levels_and_unbuildable(
        &self,
    ) -> Result<(proto::Expression, usize, Option<SparkError>)> {
        with_expression_levels_and_unbuildable(|| self.to_proto())
    }

    /// The expression of `expression`, the `spark.connect.Expression` of a Column as it was
    /// sent, with its `rendered` rendering and its `levels`, as a pickled Column unpickles: it
    /// sends the expression as it is. An expression of more levels than
    /// [`MAX_EXPRESSION_DEPTH`] is a `RecursionError`, as the API raises it when it sends one.
    /// Not a stable API.
    #[doc(hidden)]
    pub fn sent(
        expression: proto::Expression,
        rendered: String,
        levels: usize,
    ) -> Result<Expression> {
        Expression::sent_with_subqueries(expression, rendered, levels, Vec::new())
    }

    /// As [`Expression::sent`], of an expression whose subqueries refer to the plans of
    /// `subqueries`, the relations that [`Expression::subquery_relations`] gave when the Column
    /// was pickled, each of its levels: the relation that holds the expression sends them in its
    /// `WithRelations`, of their plan IDs, as PySpark's unpickled Column holds the plans of its
    /// subqueries, and its plan sends them. A relation of more levels than
    /// [`crate::plan::MAX_PLAN_DEPTH`] is a `RecursionError`. Not a stable API.
    #[doc(hidden)]
    pub fn sent_with_subqueries(
        expression: proto::Expression,
        rendered: String,
        levels: usize,
        subqueries: Vec<(proto::Relation, usize)>,
    ) -> Result<Expression> {
        if levels > MAX_EXPRESSION_DEPTH {
            return Err(SparkError::recursion_msg(format!(
                "maximum recursion depth exceeded while unpickling an expression of {levels} \
                 levels; an expression may be at most {MAX_EXPRESSION_DEPTH} levels deep"
            )));
        }
        Ok(Expression::new(ExpressionNode::Sent(Box::new(
            SentExpression {
                expression,
                rendered,
                levels,
                subqueries: SentSubqueries::of(subqueries)?,
            },
        ))))
    }

    /// The relations of the plans that the subqueries of this expression refer to, each once, in
    /// the order of the expression, of the plan IDs of the session of the DataFrame of each, as a
    /// relation that holds the expression sends them in its `WithRelations`, and the levels of
    /// each, which a pickled Column keeps for [`Expression::sent_with_subqueries`]: those of its
    /// subqueries, of an IN subquery and of a table argument too, of the expressions of a
    /// subquery, and of an expression that was sent. They are built as a request builds them, so
    /// the functions of their UDFs are pickled, and the first error of a build is the error. Not a
    /// stable API.
    #[doc(hidden)]
    pub fn subquery_relations(&self) -> Result<Vec<(proto::Relation, usize)>> {
        use crate::udf::PicklerHolder;
        let mut plans: Vec<(&Arc<LogicalPlan>, Option<&Arc<crate::plan::PlanIds>>)> = Vec::new();
        let mut add = |plan, ids| {
            if !plans.iter().any(|(known, _)| Arc::ptr_eq(known, plan)) {
                plans.push((plan, ids));
            }
        };
        let mut pending = vec![PicklerHolder::Expression(self)];
        while let Some(holder) = pending.pop() {
            let mut below = Vec::new();
            match holder {
                PicklerHolder::Expression(expression) => match expression.node() {
                    ExpressionNode::Sent(sent) => {
                        for (plan, ids) in &sent.subqueries.0 {
                            add(plan, Some(ids));
                        }
                    }
                    node => node.push_pickler_holders(&mut below),
                },
                PicklerHolder::Subquery(subquery) => {
                    add(&subquery.plan, subquery.plan_ids.as_ref());
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
                PicklerHolder::Plan(_) | PicklerHolder::Pickler(_) => {}
            }
            // The first is visited first.
            pending.extend(below.into_iter().rev());
        }
        plans
            .into_iter()
            .map(|(plan, ids)| {
                let ids = ids
                    .cloned()
                    .unwrap_or_else(|| Arc::new(crate::plan::PlanIds::new()));
                Ok((crate::plan::relation_with_ids(plan, &ids)?, plan.depth()))
            })
            .collect()
    }

    fn to_proto_with_origin(&self) -> proto::Expression {
        let mut expr = self.node_to_proto();
        // PySpark's `CaseWhen.to_plan` sends a new `UnresolvedFunction("when")` of its branches,
        // made when the plan is built, outside the `when` and `otherwise` that made the CaseWhen:
        // so of no origin. Its branches keep theirs.
        if matches!(self.node(), ExpressionNode::CaseWhen(_)) {
            return expr;
        }
        if let Some(origin) = &self.origin {
            expr.common = Some(origin_proto(origin));
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
            ExpressionNode::Sent(sent) => {
                // The relation that holds the expression sends the plans of its subqueries, as
                // those of a subquery expression.
                for (plan, ids) in &sent.subqueries.0 {
                    crate::plan::refer_to(plan, Some(ids));
                }
                // The clone recurses for each level of the expression: grow the stack for them
                // all, as the clone of a raw relation does.
                let stack = sent_expression_stack(sent.levels);
                stacker::maybe_grow(stack, stack, || sent.expression.clone())
            }
            ExpressionNode::Unbuildable(unbuildable) => unbuildable.to_proto(),
            ExpressionNode::PlanBuildOrigin(made) => made.to_proto(),
        }
    }
}

/// The stack that the clone of the expression of a [`SentExpression`] takes for each of its
/// levels: prost's `Clone` recurses through an `Expression` and its `ExprType` for each, with
/// about 1.6 KiB of frames per level in a release build (measured on macOS x86_64) and several
/// times that in a debug one, and this leaves more than as much again.
const SENT_CLONE_STACK_PER_LEVEL: usize = if cfg!(debug_assertions) {
    16 << 10
} else {
    4 << 10
};

/// The stack that the clone of the expression of a [`SentExpression`] of `levels` levels takes:
/// see [`SENT_CLONE_STACK_PER_LEVEL`].
fn sent_expression_stack(levels: usize) -> usize {
    levels
        .saturating_mul(SENT_CLONE_STACK_PER_LEVEL)
        .saturating_add(64 << 10)
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
    /// A binary literal of a Python `bytearray`, or of a subclass of it, which is sent as
    /// [`LiteralExpression::Binary`] of `value` and renders as `rendered`, the `str()` of the
    /// value, as PySpark's `LiteralExpression.__repr__` renders the value that it keeps:
    /// `bytearray(b'xy')`. Not a stable API (see [`LiteralExpression::bytearray`]).
    #[doc(hidden)]
    ByteArray {
        value: Vec<u8>,
        rendered: String,
    },
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
            LiteralExpression::Binary(v) | LiteralExpression::ByteArray { value: v, .. } => {
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
            LiteralExpression::ByteArray { rendered, .. } => rendered.clone(),
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

    /// The binary literal of a Python `bytearray` of the bytes `value`, which renders as
    /// `rendered`, the `str()` of the bytearray (see [`LiteralExpression::ByteArray`]). Not a
    /// stable API.
    #[doc(hidden)]
    pub fn bytearray(value: Vec<u8>, rendered: impl Into<String>) -> Self {
        LiteralExpression::ByteArray {
            value,
            rendered: rendered.into(),
        }
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
    /// A structured `DataType`, which the cast sends as [`CastTarget::Type`] does, and which its
    /// render shows as the simple string `shown`: PySpark's `CastExpression.__repr__` shows the
    /// `simpleString()` of the type object, which the core's type does not tell of a Python
    /// type, such as a user-defined type of its own `simpleString()`. Not a stable API.
    #[doc(hidden)]
    Shown(DataType, String),
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
            CastTarget::Type(dt) | CastTarget::Shown(dt, _) => {
                proto::expression::cast::CastToType::Type(dt.to_proto())
            }
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

    /// The boundary as the int of PySpark's frame, as [`FrameBoundary::value`], of the
    /// `Preceding(i64::MIN)` too, which is beyond a long: `Window.unboundedPreceding`
    /// (`JVM_LONG_MIN`), the negated count of a preceding boundary, 0, the count of a following
    /// one, or `Window.unboundedFollowing` (`JVM_LONG_MAX`).
    pub(crate) fn offset(&self) -> i128 {
        match self {
            FrameBoundary::UnboundedPreceding => i64::MIN.into(),
            FrameBoundary::Preceding(n) => -i128::from(*n),
            FrameBoundary::CurrentRow => 0,
            FrameBoundary::Following(n) => (*n).into(),
            FrameBoundary::UnboundedFollowing => i64::MAX.into(),
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
            lower.offset(),
            upper.offset()
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

        // Set partition spec. PySpark's `to_plan` warns of a window of no partitions here, in the
        // order of the build, among the picklings of the UDFs of the plan (see [`UdfCommands`]).
        if self.partition_spec.is_empty() {
            warn_of_a_window_without_partitions();
        }
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
            frame.lower = Some(Box::new(to_proto_frame_boundary(lower, rows, true)));
            frame.upper = Some(Box::new(to_proto_frame_boundary(upper, rows, false)));
            window.frame_spec = Some(Box::new(frame));
        }

        expr.expr_type = Some(proto::expression::ExprType::Window(Box::new(window)));
        expr
    }
}

/// Convert FrameBoundary to proto FrameBoundary, the lower boundary of a frame if `lower`, else
/// its upper one, as PySpark Connect's `WindowExpression.to_plan` sends the int of a boundary
/// ([`FrameBoundary::offset`]): 0 is the current row, `Window.unboundedPreceding` as the lower
/// boundary and `Window.unboundedFollowing` as the upper one are unbounded, and another int is
/// the int literal of a row frame, or the long literal of a range frame. So an unbounded
/// boundary on the other side is the literal of its int.
///
/// As PySpark's, an int beyond an int32 in a row frame fails the request with `VALUE_NOT_BETWEEN`
/// of `start` or `end`, and one beyond a long in a range frame with protobuf's `ValueError` of its
/// field; the boundary is sent as a long literal of the int, saturated, by a serialization that
/// does not fail.
fn to_proto_frame_boundary(
    boundary: &FrameBoundary,
    rows: bool,
    lower: bool,
) -> proto::expression::window::window_frame::FrameBoundary {
    use proto::expression::window::window_frame::frame_boundary::Boundary;

    let offset = boundary.offset();
    let unbounded = if lower { i64::MIN } else { i64::MAX };
    let boundary = if offset == 0 {
        Boundary::CurrentRow(true)
    } else if offset == i128::from(unbounded) {
        Boundary::Unbounded(true)
    } else {
        let literal = match (rows, i32::try_from(offset), i64::try_from(offset)) {
            (true, Ok(value), _) => LiteralExpression::int(value),
            (false, _, Ok(value)) => LiteralExpression::long(value),
            _ => {
                record_unbuildable(&if rows {
                    SparkError::value(
                        "VALUE_NOT_BETWEEN",
                        &[
                            ("arg_name", if lower { "start" } else { "end" }),
                            ("min", &i32::MIN.to_string()),
                            ("max", &i32::MAX.to_string()),
                        ],
                    )
                } else {
                    SparkError::builtin_value_msg(format!("Value out of range: {offset}"))
                });
                let saturated =
                    i64::try_from(offset).unwrap_or(if offset < 0 { i64::MIN } else { i64::MAX });
                LiteralExpression::long(saturated)
            }
        };
        Boundary::Value(Box::new(Expression::literal(literal).to_proto()))
    };
    proto::expression::window::window_frame::FrameBoundary {
        boundary: Some(boundary),
    }
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
/// round-trip, the nearer of two such, and of an even last digit of a tie, as Python's `repr`
/// (David Gay's `dtoa`) gives them (see [`shortest_float_digits`]), in fixed notation for a
/// decimal exponent from -4 to 15, otherwise in scientific notation with a signed exponent of at
/// least two digits.
pub(crate) fn python_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let sign = if x < 0.0 { "-" } else { "" };
    let (digits, point) = shortest_float_digits(x.abs());
    let exponent = point - 1;
    if (-4..16).contains(&exponent) {
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

/// The shortest decimal digits of the positive finite `x` that read back as `x`, without leading
/// or trailing zeros, and the place of their decimal point: `x` is `0.<digits>` times ten to the
/// `point`. Of two such digits, they are the nearer to `x`, and of a tie those of an even last
/// digit, as `ryu` rounds them and as Python's `repr` does; the formatting of the standard library
/// rounds a tie up, so `74334700308107.625` would be `74334700308107.63` of it, where Python's
/// `repr` is `74334700308107.62`.
pub(crate) fn shortest_float_digits(x: f64) -> (String, i32) {
    let mut buffer = ryu::Buffer::new();
    let text = buffer.format_finite(x);
    let (mantissa, exponent) = match text.split_once(['e', 'E']) {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().unwrap_or(0)),
        None => (text, 0),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let all = format!("{whole}{fraction}");
    let significant = all.trim_start_matches('0');
    let leading = (all.len() - significant.len()) as i32;
    let digits = significant.trim_end_matches('0').to_string();
    (digits, whole.len() as i32 + exponent - leading)
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
    fn python_float_repr_rounds_a_tie_of_the_shortest_digits_to_even_as_python() {
        // The reprs of official Python 3.11: of two shortest digits as near, the even last digit,
        // where the formatting of the standard library rounds up (`74334700308107.63`). The
        // ties are exact sums, as their literals would read as the shorter digits of the float.
        let tie = 74_334_700_308_107.0 + 0.625;
        for (value, repr) in [
            (tie, "74334700308107.62"),
            (810_452_564_081.0 + 0.40625, "810452564081.4062"),
            (-(2_226_063_173_806_980.0 + 0.25), "-2226063173806980.2"),
            (0.5, "0.5"),
            (2.5e-5, "2.5e-05"),
            (9007199254740993.0, "9007199254740992.0"),
            (5e-324, "5e-324"),
            (1e23, "1e+23"),
            (8.5e-5, "8.5e-05"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (1e-5, "1e-05"),
            (123.0, "123.0"),
            (f64::MAX, "1.7976931348623157e+308"),
            (1.0 / 3.0, "0.3333333333333333"),
            (1.2345678901234568e17, "1.2345678901234568e+17"),
            (-0.0, "-0.0"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(python_float_repr(value), repr, "{value:?}");
        }
        assert_eq!(
            Expression::literal(LiteralExpression::Double(tie)).render(),
            "74334700308107.62"
        );
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

    /// The expression of an unpickled Column, as it was sent, of as many levels as the client
    /// sends, is serialized on a thread of 512 KiB: its clone, which recurses for each level, runs
    /// on a stack grown for its levels, as the clone of a raw relation does.
    #[test]
    fn a_deep_sent_expression_is_serialized_on_a_small_stack() {
        let protos = std::thread::Builder::new()
            .stack_size(512 << 10)
            .spawn(|| {
                let proto = chained_additions(MAX_EXPRESSION_DEPTH - 1).to_proto();
                assert_eq!(serialized_levels(&proto), MAX_EXPRESSION_DEPTH);
                let sent = Expression::sent(proto, "deep".to_string(), MAX_EXPRESSION_DEPTH)
                    .expect("at the bound");
                let serialized = sent.to_proto();
                assert_eq!(serialized_levels(&serialized), MAX_EXPRESSION_DEPTH);
                (sent, serialized)
            })
            .unwrap()
            .join()
            .unwrap();
        // The caller drops them: protos this deep need the stack of a big thread.
        std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(move || std::mem::drop(protos))
            .unwrap()
            .join()
            .unwrap();
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

    /// The expressions of a command are serialized as those of a plan: as PySpark's serializer,
    /// which raises `RecursionError` for a merge condition or an overwrite condition of about
    /// 1,000 levels before any request, one deeper than `MAX_EXPRESSION_DEPTH` is a
    /// `RecursionError`, rather than a request whose proto overflows the stack where it is
    /// encoded or dropped. The session has no server and does not retry, so a request that the
    /// client should not send fails the test at once.
    #[test]
    fn deep_expressions_of_commands_are_recursion_errors_as_pyspark() {
        use crate::column::{col, Column};

        std::thread::Builder::new()
            .stack_size(512 << 10)
            .spawn(|| {
                let builder =
                    spark_connect_core::channel::ChannelBuilder::parse("sc://127.0.0.1:1")
                        .expect("valid URL");
                let mut client = spark_connect_core::runtime::block_on(
                    spark_connect_core::client::SparkConnectClient::connect(&builder),
                )
                .expect("failed to build client");
                client.set_retry_policy(spark_connect_core::retries::RetryPolicy::no_retries());
                let spark = crate::session::SparkSession::new(client);
                let df = spark.range(3).unwrap();
                let deep = Column::new(chained_additions(100_000));
                let assert_too_deep = |result: Result<()>| {
                    let error = result.expect_err("too deep to be sent");
                    assert_eq!(
                        error.kind,
                        spark_connect_core::error::SparkErrorKind::RecursionError,
                        "{error}"
                    );
                    assert!(
                        error
                            .message()
                            .contains("an expression may be at most 745 levels deep"),
                        "{error}"
                    );
                };

                // The condition of a merge, and the condition and an assignment of its clauses.
                assert_too_deep(
                    df.merge_into("t", deep.clone())
                        .when_matched(None)
                        .update_all()
                        .merge(),
                );
                assert_too_deep(
                    df.merge_into("t", col("id"))
                        .when_not_matched_by_source(Some(deep.clone()))
                        .delete()
                        .merge(),
                );
                let assignments =
                    std::collections::HashMap::from([("v".to_string(), deep.clone())]);
                assert_too_deep(
                    df.merge_into("t", col("id"))
                        .when_not_matched(None)
                        .insert(assignments)
                        .merge(),
                );
                // The condition of an overwrite, and a partitioning column of `writeTo`.
                assert_too_deep(df.write_to("t").overwrite(deep.clone()));
                assert_too_deep(df.write_to("t").partition_by([deep.clone()]).create());
                // An argument of a function to register.
                let function = crate::udf::CommonInlineUserDefinedFunctionExpression::new(
                    "f".to_string(),
                    false,
                    vec![deep.expression().clone()],
                    crate::udf::PythonUDFPayload::new(
                        DataType::Integer,
                        crate::udf::eval_type::SQL_BATCHED_UDF,
                        vec![0],
                        "3.11".to_string(),
                    ),
                );
                assert_too_deep(spark.register_function(function));
                // The grouping sets of a column too deep to be sent group it, and the plan of
                // the grouping is the `RecursionError`.
                let grouped = df.grouping_sets(vec![vec![deep.clone()], vec![deep.clone()]]);
                assert_too_deep(grouped.count().to_relation().map(drop));
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
    fn a_sent_expression_has_its_levels_and_counts_them_under_another_expression() {
        // As a pickled Column of 300 additions: 301 levels, which the Column keeps when it is
        // unpickled, and which count under an expression over it, as the levels of the API do.
        let mut expr = col("id");
        for _ in 0..300 {
            expr = Expression::function(
                "+",
                vec![expr, Expression::literal(LiteralExpression::Long(1))],
            );
        }
        let rendered = expr.render();
        let (sent, levels) = expr.try_to_proto_with_levels().unwrap();
        assert_eq!(levels, 301);
        let unpickled = Expression::sent(sent.clone(), rendered.clone(), levels).unwrap();
        assert_eq!(unpickled.to_proto(), sent);
        let over = Expression::function("abs", vec![unpickled.clone()]);
        assert_eq!(over.try_to_proto_with_levels().unwrap().1, 302);
        assert_eq!(
            over.try_to_proto().unwrap(),
            Expression::function("abs", vec![expr])
                .try_to_proto()
                .unwrap()
        );
        // At the bound it is sent, and one level more, under an expression, is a RecursionError.
        let at_bound = Expression::sent(sent.clone(), rendered.clone(), MAX_EXPRESSION_DEPTH);
        let at_bound = at_bound.unwrap();
        assert!(at_bound.try_to_proto().is_ok());
        let error = Expression::function("abs", vec![at_bound])
            .try_to_proto()
            .unwrap_err();
        assert_eq!(
            error.kind,
            spark_connect_core::error::SparkErrorKind::RecursionError
        );
        // A sent expression deeper than the bound is refused as it is unpickled.
        let error = Expression::sent(sent, rendered, MAX_EXPRESSION_DEPTH + 1)
            .err()
            .unwrap();
        assert_eq!(
            error.kind,
            spark_connect_core::error::SparkErrorKind::RecursionError
        );
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
            levels: 3,
            subqueries: SentSubqueries::default(),
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
        let lit = LiteralExpression::Float(1.1);
        assert_eq!(lit.render(), "1.100000023841858");
    }

    #[test]
    fn test_literal_render_double() {
        let lit = LiteralExpression::Double(1.23456);
        assert_eq!(lit.render(), "1.23456");
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
    fn test_literal_render_bytearray() {
        // A literal of a Python bytearray renders as its `str()`, as PySpark's
        // `LiteralExpression.__repr__`, and is sent as the binary literal of its bytes.
        let lit = LiteralExpression::bytearray(vec![b'a', b'\''], "bytearray(b\"a\\'\")");
        assert_eq!(lit.render(), "bytearray(b\"a\\'\")");
        assert_eq!(
            Expression::literal(LiteralExpression::Array {
                element_type: Box::new(DataType::Binary),
                elements: vec![lit.clone()],
            })
            .render(),
            "[bytearray(b\"a\\'\")]"
        );
        assert_eq!(
            lit.to_proto(),
            LiteralExpression::Binary(vec![b'a', b'\'']).to_proto()
        );
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

    use spark_connect_core::error::SparkErrorKind;

    /// An alias of `names` of `child`, whose request raises `before` and `after` and whose
    /// rendering raises `unrenderable`, as the drop-in's `Column.alias` of names that protobuf
    /// refuses.
    fn unbuildable_alias(
        child: Expression,
        name: &str,
        before: Option<SparkError>,
        after: Option<SparkError>,
        unrenderable: Option<SparkError>,
    ) -> Expression {
        let alias: Expression = ExpressionNode::Alias(Box::new(Alias::new(child, name))).into();
        Expression::unbuildable(alias, before, after, unrenderable)
    }

    fn type_error(message: &str) -> SparkError {
        SparkError {
            kind: SparkErrorKind::BuiltinTypeError,
            ..SparkError::value_msg(message)
        }
    }

    #[test]
    fn test_an_unbuildable_expression_fails_its_request_in_pysparks_build_order() {
        // Of no error, the expression itself.
        let plain = Expression::unbuildable(col("x"), None, None, None);
        assert!(matches!(plain.node(), ExpressionNode::ColumnReference(_)));
        // The request fails of the first error as PySpark builds the plan: the names of an alias
        // are set before its child is built, its metadata after.
        let inner = unbuildable_alias(col("x"), "", Some(type_error("names")), None, None);
        let outer = unbuildable_alias(inner.clone(), "y", None, Some(type_error("metadata")), None);
        let error = outer.try_to_proto().expect_err("an unbuildable alias");
        assert_eq!(error.kind, SparkErrorKind::BuiltinTypeError);
        assert_eq!(error.message, "names");
        let metadata_first =
            unbuildable_alias(col("x"), "y", None, Some(type_error("metadata")), None);
        let both = unbuildable_alias(metadata_first, "z", Some(type_error("outer")), None, None);
        assert_eq!(both.try_to_proto().expect_err("both").message, "outer");
        // Its proto is that of what this client builds, of no level of its own, which a pickled
        // Column keeps with the error.
        let (proto, levels, unbuildable) = inner
            .try_to_proto_with_levels_and_unbuildable()
            .expect("not too deep");
        assert_eq!(levels, 2);
        assert_eq!(unbuildable.expect("the error").message, "names");
        assert!(matches!(
            proto.expr_type,
            Some(proto::expression::ExprType::Alias(_))
        ));
        // Outside a request, the proto is built as it is.
        assert!(matches!(
            inner.to_proto().expr_type,
            Some(proto::expression::ExprType::Alias(_))
        ));
        // A request of a plain expression fails no more.
        assert!(col("x").try_to_proto().is_ok());
    }

    /// A pickler that records `tag` in `calls` at each pickling, and gives the tag and the number
    /// of the pickling, or, when it `fails`, the error of the tag.
    fn recording_pickler(
        calls: &std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        tag: &'static str,
        fails: bool,
    ) -> crate::udf::CommandPickler {
        let state = (std::sync::Arc::clone(calls), AtomicU32::new(0));
        crate::udf::CommandPickler::new(state, move |(calls, count)| {
            let n = count.fetch_add(1, Ordering::SeqCst);
            calls.lock().unwrap().push(tag.to_string());
            if fails {
                return Err(type_error(tag));
            }
            Ok(format!("{tag}{n}").into_bytes())
        })
    }

    /// A Python UDF of `x`, of the command of `pickler`.
    fn pickled_udf(pickler: crate::udf::CommandPickler) -> Expression {
        let payload = crate::udf::PythonUDFPayload::pickled_at_build(
            DataType::Long,
            crate::udf::eval_type::SQL_BATCHED_UDF,
            pickler,
            "3.11".to_string(),
        );
        ExpressionNode::CommonInlineUserDefinedFunction(Box::new(
            CommonInlineUserDefinedFunctionExpression::new(
                "f".to_string(),
                true,
                vec![col("x")],
                payload,
            ),
        ))
        .into()
    }

    /// The commands of the UDFs that are the arguments of the function `expression`.
    fn argument_commands(expression: &proto::Expression) -> Vec<String> {
        let Some(proto::expression::ExprType::UnresolvedFunction(function)) = &expression.expr_type
        else {
            panic!("a function");
        };
        function
            .arguments
            .iter()
            .map(|argument| {
                let Some(proto::expression::ExprType::CommonInlineUserDefinedFunction(udf)) =
                    &argument.expr_type
                else {
                    panic!("a UDF");
                };
                let Some(proto::common_inline_user_defined_function::Function::PythonUdf(python)) =
                    &udf.function
                else {
                    panic!("a Python UDF");
                };
                String::from_utf8(python.command.to_vec()).unwrap()
            })
            .collect()
    }

    #[test]
    fn test_the_commands_of_udfs_are_pickled_at_each_request_in_pysparks_build_order() {
        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let taken = || std::mem::take(&mut *calls.lock().unwrap());
        let a = pickled_udf(recording_pickler(&calls, "a", false));
        let b = pickled_udf(recording_pickler(&calls, "b", false));
        // As PySpark's `PythonUDF.to_plan`, each request pickles each UDF as it builds the plan,
        // in its order, a UDF that is twice in the plan twice, and again at the next request.
        let both = Expression::function("struct", vec![a.clone(), b.clone(), a.clone()]);
        let sent = both.try_to_proto().expect("picklable");
        assert_eq!(argument_commands(&sent), ["a0", "b0", "a1"]);
        assert_eq!(taken(), ["a", "b", "a"]);
        let sent = both.try_to_proto().expect("picklable");
        assert_eq!(argument_commands(&sent), ["a2", "b1", "a3"]);
        assert_eq!(taken(), ["a", "b", "a"]);
        // The first pickling that fails is the error of the request, after which none is
        // pickled, as PySpark's build stops at it.
        let bad = pickled_udf(recording_pickler(&calls, "bad", true));
        let error = Expression::function("struct", vec![a.clone(), bad.clone(), b.clone()])
            .try_to_proto()
            .expect_err("a function that does not pickle");
        assert_eq!(error.message, "bad");
        assert_eq!(taken(), ["a", "bad"]);
        // An expression that fails to build before the UDFs is the error, and none is pickled;
        // after them, they are pickled first.
        let names = unbuildable_alias(col("x"), "", Some(type_error("names")), None, None);
        let error = Expression::function("struct", vec![names.clone(), a.clone()])
            .try_to_proto()
            .expect_err("names");
        assert_eq!((error.message.as_str(), taken()), ("names", vec![]));
        let error = Expression::function("struct", vec![a.clone(), names.clone()])
            .try_to_proto()
            .expect_err("names");
        assert_eq!(
            (error.message.as_str(), taken()),
            ("names", vec!["a".into()])
        );
        let error = Expression::function("struct", vec![bad.clone(), names.clone()])
            .try_to_proto()
            .expect_err("bad");
        assert_eq!(
            (error.message.as_str(), taken()),
            ("bad", vec!["bad".into()])
        );
        // The error of a pickling is kept apart, as that of an expression, for a pickled Column.
        let (_, _, kept) = Expression::function("struct", vec![bad.clone()])
            .try_to_proto_with_levels_and_unbuildable()
            .expect("not too deep");
        assert_eq!(kept.expect("the error").message, "bad");
        assert_eq!(taken(), ["bad"]);
        // Outside a request, as the Rust API's `to_proto`, the command is pickled as it is built.
        let built = Expression::function("struct", vec![a.clone(), bad.clone()]).to_proto();
        assert_eq!(argument_commands(&built), ["a6", ""]);
        assert_eq!(taken(), ["a", "bad"]);
        // A request of an expression without UDFs pickles nothing.
        assert!(col("x").try_to_proto().is_ok());
        assert!(taken().is_empty());
    }

    #[test]
    fn test_an_unrenderable_expression_fails_try_render_as_pysparks_repr() {
        let unrenderable = unbuildable_alias(
            col("x"),
            "q",
            None,
            None,
            Some(type_error(
                "sequence item 0: expected str instance, bytes found",
            )),
        );
        // The plan builds: only the `__repr__` fails, as PySpark's of an alias of bytes.
        assert!(unrenderable.try_to_proto().is_ok());
        assert_eq!(unrenderable.render(), "x AS q");
        let error = unrenderable.try_render().expect_err("a bytes name");
        assert_eq!(
            error.message,
            "sequence item 0: expected str instance, bytes found"
        );
        // Within another expression, as PySpark's `__repr__` renders the arguments.
        let upper = Expression::function("upper", vec![unrenderable]);
        assert!(upper.try_render().is_err());
        assert_eq!(col("x").try_render().expect("plain"), "x");
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
        let lit = LiteralExpression::Float(1.1);
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::Float(f)) = literal.literal_type {
                assert!((f - 1.1).abs() < 0.01);
            } else {
                panic!("Expected float literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn test_literal_double_to_proto() {
        let lit = LiteralExpression::Double(1.23456);
        let proto = lit.to_proto();
        assert!(proto.expr_type.is_some());
        if let Some(proto::expression::ExprType::Literal(literal)) = proto.expr_type {
            if let Some(proto::expression::literal::LiteralType::Double(d)) = literal.literal_type {
                assert!((d - 1.23456).abs() < 0.00001);
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
