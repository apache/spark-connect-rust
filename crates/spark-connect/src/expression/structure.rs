use super::{
    with_stack_for_a_level, Alias, CallFunctionWrapper, CaseWhen, Cast, CastEvalMode, CastTarget,
    ColumnReference, Expression, ExpressionSpec, ExtractValue, FrameBoundary, LambdaFunction,
    LiteralExpression, NamedArgument, NullOrdering, SentExpressionPayload, SortOrder,
    SubqueryExpression, SubqueryType, UnresolvedFunction, UnresolvedNamedLambdaVariable,
    UpdateFieldsExpr, WindowExpressionWrapper,
};
use crate::{plan::LogicalPlan, types::DataType, udf::CommonInlineUserDefinedFunctionExpression};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// The type explicitly declared by a payload, or an expression awaiting server analysis.
/// This does not perform name resolution, coercion or function return-type inference.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ExpressionType {
    Unresolved,
    Known(Arc<DataType>),
}

/// A shared expression's payload, ordered inputs, declared type and construction origin.
/// Construct expressions through [`Expression::try_new`] or the typed builder specifications.
pub struct ExpressionNode {
    pub(super) kind: ExpressionKind,
    pub(super) children: SmallVec<[Expression; 2]>,
    pub(super) ty: ExpressionType,
    pub(super) origin: Option<Arc<crate::origin::Origin>>,
}

impl ExpressionNode {
    /// Borrow the operator payload, excluding structural expression inputs.
    pub fn kind(&self) -> &ExpressionKind {
        &self.kind
    }
    /// Borrow structural expression inputs in the operator's documented order.
    pub fn children(&self) -> &[Expression] {
        &self.children
    }
    /// Borrow the explicitly declared type state; no server inference is performed.
    pub fn ty(&self) -> &ExpressionType {
        &self.ty
    }
}

impl std::fmt::Debug for ExpressionNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpressionNode")
            .field("kind", &self.kind)
            .field("children", &self.children)
            .field("ty", &self.ty)
            .finish_non_exhaustive()
    }
}

impl PartialEq for ExpressionNode {
    fn eq(&self, other: &Self) -> bool {
        // Origins do not participate in value equality. Never shortcut shared NaNs.
        self.kind == other.kind && self.children == other.children
    }
}

impl Drop for ExpressionNode {
    fn drop(&mut self) {
        if !self.children.is_empty() {
            let mut pending = std::mem::take(&mut self.children);
            while let Some(expression) = pending.pop() {
                // Leaves cannot recurse through structural inputs. Drop them in place
                // rather than moving their complete node out of its Arc allocation.
                if expression.children().is_empty() {
                    drop(expression);
                } else if let Some(mut node) = Arc::into_inner(expression.node) {
                    pending.extend(std::mem::take(&mut node.children));
                }
            }
        }
        // These payloads have no recursive fields once their inputs are drained.
        // Keep recursive literal/type/protobuf/plan payloads on the guarded path.
        let shallow_payload = match &self.kind {
            ExpressionKind::Literal(literal) => matches!(
                literal,
                LiteralExpression::Boolean(_)
                    | LiteralExpression::Byte(_)
                    | LiteralExpression::Short(_)
                    | LiteralExpression::Integer(_)
                    | LiteralExpression::Long(_)
                    | LiteralExpression::Float(_)
                    | LiteralExpression::Double(_)
                    | LiteralExpression::Decimal { .. }
                    | LiteralExpression::String(_)
                    | LiteralExpression::Binary(_)
                    | LiteralExpression::Date(_)
                    | LiteralExpression::Timestamp(_)
                    | LiteralExpression::TimestampNtz(_)
                    | LiteralExpression::DayTimeInterval(_)
                    | LiteralExpression::YearMonthInterval(_)
                    | LiteralExpression::Time { .. }
            ),
            ExpressionKind::ColumnReference(_) | ExpressionKind::UnresolvedFunction { .. } => true,
            _ => false,
        };
        if matches!(self.ty, ExpressionType::Unresolved) && shallow_payload {
            return;
        }
        // Opaque protobuf and type payloads retain their recursive destruction behavior.
        let kind = std::mem::replace(&mut self.kind, ExpressionKind::SQLExpression(String::new()));
        let ty = std::mem::replace(&mut self.ty, ExpressionType::Unresolved);
        with_stack_for_a_level(move || drop((kind, ty)));
    }
}

/// Ordering metadata of a window input. Its base expression lives in children.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortOrderProperties {
    pub ascending: bool,
    pub null_ordering: NullOrdering,
}

/// A plan dependency of a subquery, separate from its expression inputs.
/// Its plan and ID are retained together when expression inputs are rebuilt.
#[derive(Debug, Clone)]
pub struct SubqueryPlan {
    pub(crate) plan: Arc<LogicalPlan>,
    pub(crate) plan_id: i64,
    pub subquery_type: SubqueryType,
    pub partition_count: usize,
    pub order_count: usize,
    pub in_value_count: usize,
    pub with_single_partition: Option<bool>,
}

impl SubqueryPlan {
    pub fn plan(&self) -> &Arc<LogicalPlan> {
        &self.plan
    }
    pub fn plan_id(&self) -> i64 {
        self.plan_id
    }
}

impl PartialEq for SubqueryPlan {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.plan, &other.plan)
            && self.plan_id == other.plan_id
            && self.subquery_type == other.subquery_type
            && self.partition_count == other.partition_count
            && self.order_count == other.order_count
            && self.in_value_count == other.in_value_count
            && self.with_single_partition == other.with_single_partition
    }
}

/// Operator payload without expression inputs. Input order is documented in the design guide.
/// Referenced plans, lambda declarations and opaque Sent protobufs are payload dependencies,
/// rather than structural expression inputs.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ExpressionKind {
    Literal(LiteralExpression),
    ColumnReference(ColumnReference),
    UnresolvedFunction {
        name: String,
        is_distinct: bool,
    },
    UnresolvedStar(Option<String>, Option<i64>),
    Alias {
        names: Vec<String>,
        metadata: Option<String>,
    },
    Cast {
        target: CastTarget,
        eval_mode: Option<CastEvalMode>,
    },
    DirectShufflePartitionId,
    UnresolvedRegex(String, Option<i64>),
    SortOrder {
        ascending: bool,
        null_ordering: NullOrdering,
    },
    CaseWhen {
        branch_count: usize,
        has_else: bool,
    },
    UnresolvedExtractValue,
    UpdateFields {
        field_name: String,
        has_value: bool,
    },
    SQLExpression(String),
    CallFunction {
        function_name: String,
    },
    WindowExpression {
        partition_count: usize,
        order_spec: Vec<SortOrderProperties>,
        frame_spec: Option<Arc<(u32, FrameBoundary, FrameBoundary)>>,
    },
    LambdaFunction {
        arguments: Vec<UnresolvedNamedLambdaVariable>,
    },
    UnresolvedNamedLambdaVariable(UnresolvedNamedLambdaVariable),
    CommonInlineUserDefinedFunction {
        function_name: String,
        deterministic: bool,
        // Keep the uncommon large payload from enlarging every expression node.
        python_udf: Arc<crate::udf::PythonUDFPayload>,
    },
    NamedArgument {
        key: String,
    },
    Subquery(SubqueryPlan),
    Sent(Arc<SentExpressionPayload>),
}

/// Invalid input count, or segment metadata whose count cannot fit in usize.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidExpressionInputs {
    #[non_exhaustive]
    InvalidArity {
        operator: &'static str,
        expected_min: usize,
        expected_max: Option<usize>,
        actual: usize,
    },
    #[non_exhaustive]
    InputCountOverflow { operator: &'static str },
}

impl std::fmt::Display for InvalidExpressionInputs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArity {
                operator,
                expected_min,
                expected_max,
                actual,
            } => match expected_max {
                Some(max) if max == expected_min => write!(
                    f,
                    "{operator} expects {expected_min} expression inputs, got {actual}"
                ),
                Some(max) => write!(
                    f,
                    "{operator} expects {expected_min}..={max} expression inputs, got {actual}"
                ),
                None => write!(
                    f,
                    "{operator} expects at least {expected_min} expression inputs, got {actual}"
                ),
            },
            Self::InputCountOverflow { operator } => {
                write!(f, "{operator} expression input count overflows usize")
            }
        }
    }
}
impl std::error::Error for InvalidExpressionInputs {}

impl ExpressionKind {
    /// Return the stable operator name used in input-layout diagnostics.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Literal(_) => "Literal",
            Self::ColumnReference(_) => "ColumnReference",
            Self::UnresolvedFunction { .. } => "UnresolvedFunction",
            Self::UnresolvedStar(..) => "UnresolvedStar",
            Self::Alias { .. } => "Alias",
            Self::Cast { .. } => "Cast",
            Self::DirectShufflePartitionId => "DirectShufflePartitionId",
            Self::UnresolvedRegex(..) => "UnresolvedRegex",
            Self::SortOrder { .. } => "SortOrder",
            Self::CaseWhen { .. } => "CaseWhen",
            Self::UnresolvedExtractValue => "UnresolvedExtractValue",
            Self::UpdateFields { .. } => "UpdateFields",
            Self::SQLExpression(_) => "SQLExpression",
            Self::CallFunction { .. } => "CallFunction",
            Self::WindowExpression { .. } => "WindowExpression",
            Self::LambdaFunction { .. } => "LambdaFunction",
            Self::UnresolvedNamedLambdaVariable(_) => "UnresolvedNamedLambdaVariable",
            Self::CommonInlineUserDefinedFunction { .. } => "CommonInlineUserDefinedFunction",
            Self::NamedArgument { .. } => "NamedArgument",
            Self::Subquery(_) => "Subquery",
            Self::Sent(_) => "Sent",
        }
    }

    /// Structural input bounds. Variadic operators return `0..=usize::MAX`.
    /// Segment counts are checked before arithmetic overflows.
    ///
    /// # Errors
    /// Returns [`InvalidExpressionInputs::InputCountOverflow`] if segment counts overflow.
    pub fn arity(
        &self,
    ) -> std::result::Result<std::ops::RangeInclusive<usize>, InvalidExpressionInputs> {
        self.input_bounds()
            .map(|(min, max)| min..=max.unwrap_or(usize::MAX))
    }

    // Keep the unbounded case explicit rather than reconstructing it from the public range.
    fn input_bounds(&self) -> std::result::Result<(usize, Option<usize>), InvalidExpressionInputs> {
        let overflow = || InvalidExpressionInputs::InputCountOverflow {
            operator: self.name(),
        };
        let count = match self {
            Self::Literal(_)
            | Self::ColumnReference(_)
            | Self::UnresolvedStar(..)
            | Self::UnresolvedRegex(..)
            | Self::SQLExpression(_)
            | Self::UnresolvedNamedLambdaVariable(_)
            | Self::Sent(_) => 0,
            Self::Alias { .. }
            | Self::Cast { .. }
            | Self::DirectShufflePartitionId
            | Self::SortOrder { .. }
            | Self::LambdaFunction { .. }
            | Self::NamedArgument { .. } => 1,
            Self::UnresolvedExtractValue => 2,
            Self::UnresolvedFunction { .. }
            | Self::CallFunction { .. }
            | Self::CommonInlineUserDefinedFunction { .. } => return Ok((0, None)),
            Self::UpdateFields { has_value, .. } => 1 + usize::from(*has_value),
            Self::CaseWhen {
                branch_count,
                has_else,
            } => branch_count
                .checked_mul(2)
                .and_then(|n| n.checked_add(usize::from(*has_else)))
                .ok_or_else(overflow)?,
            Self::WindowExpression {
                partition_count,
                order_spec,
                ..
            } => partition_count
                .checked_add(1)
                .and_then(|n| n.checked_add(order_spec.len()))
                .ok_or_else(overflow)?,
            Self::Subquery(subquery) => subquery
                .partition_count
                .checked_add(subquery.order_count)
                .and_then(|n| n.checked_add(subquery.in_value_count))
                .ok_or_else(overflow)?,
        };
        Ok((count, Some(count)))
    }

    fn validate_inputs(&self, actual: usize) -> std::result::Result<(), InvalidExpressionInputs> {
        let (min, max) = self.input_bounds()?;
        if actual < min || max.is_some_and(|max| actual > max) {
            return Err(InvalidExpressionInputs::InvalidArity {
                operator: self.name(),
                expected_min: min,
                expected_max: max,
                actual,
            });
        }
        Ok(())
    }

    #[inline]
    fn declared_type(&self) -> ExpressionType {
        match self {
            Self::Literal(LiteralExpression::Null(data_type))
            | Self::Cast {
                target: CastTarget::Type(data_type),
                ..
            } => ExpressionType::Known(Arc::clone(data_type)),
            Self::CommonInlineUserDefinedFunction { python_udf, .. } => {
                ExpressionType::Known(Arc::new(clone_declared_type(&python_udf.output_type)))
            }
            _ => ExpressionType::Unresolved,
        }
    }
}

// Caching a declared type adds a clone during construction. Guard each nested type
// level so this new operation does not depend on the caller's remaining stack.
// Arbitrary JSON field metadata retains DataType's existing clone behavior.
fn clone_declared_type(ty: &DataType) -> DataType {
    with_stack_for_a_level(|| match ty {
        DataType::Array {
            element_type,
            contains_null,
        } => DataType::Array {
            element_type: Box::new(clone_declared_type(element_type)),
            contains_null: *contains_null,
        },
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => DataType::Map {
            key_type: Box::new(clone_declared_type(key_type)),
            value_type: Box::new(clone_declared_type(value_type)),
            value_contains_null: *value_contains_null,
        },
        DataType::Struct { fields } => DataType::Struct {
            fields: fields
                .iter()
                .map(|field| crate::types::StructField {
                    name: field.name.clone(),
                    data_type: clone_declared_type(&field.data_type),
                    nullable: field.nullable,
                    metadata: field.metadata.clone(),
                })
                .collect(),
        },
        DataType::Udt {
            type_str,
            jvm_class,
            python_class,
            serialized_python_class,
            sql_type,
        } => DataType::Udt {
            type_str: type_str.clone(),
            jvm_class: jvm_class.clone(),
            python_class: python_class.clone(),
            serialized_python_class: serialized_python_class.clone(),
            sql_type: sql_type
                .as_ref()
                .map(|ty| Box::new(clone_declared_type(ty))),
        },
        _ => ty.clone(),
    })
}

// Short argument lists stay inline even when their builder Vec has spare capacity.
// Wider lists transfer their allocation without copying the expression handles.
fn normalize_arguments(arguments: Vec<Expression>) -> SmallVec<[Expression; 2]> {
    let mut children = SmallVec::new();
    if arguments.len() <= children.inline_size() {
        children.extend(arguments);
        children
    } else {
        SmallVec::from_vec(arguments)
    }
}

#[inline]
fn collect_inputs(inputs: impl IntoIterator<Item = Expression>) -> SmallVec<[Expression; 2]> {
    let inputs = inputs.into_iter();
    let lower_bound = inputs.size_hint().0;
    let mut children = SmallVec::new();
    if lower_bound > children.inline_size() {
        // Fill wide buffers through Vec's optimized iterator extension, then
        // transfer the allocation into SmallVec without copying the inputs.
        let mut buffer = Vec::with_capacity(lower_bound);
        buffer.extend(inputs);
        return SmallVec::from_vec(buffer);
    }
    children.extend(inputs);
    children
}

impl Expression {
    /// Construct an expression with checked structural input layout and current origin.
    /// The origin is read after collecting `children`.
    ///
    /// # Errors
    /// Returns [`InvalidExpressionInputs`] for invalid arity or overflowing segment metadata.
    pub fn try_new(
        kind: ExpressionKind,
        children: impl IntoIterator<Item = Expression>,
    ) -> std::result::Result<Self, InvalidExpressionInputs> {
        let children = collect_inputs(children);
        kind.validate_inputs(children.len())?;
        Ok(Self::from_parts(kind, children, crate::origin::current()))
    }

    // Convenience leaf construction avoids routing common scalar literals and references
    // through the larger typed-builder normalization and generic input collection paths.
    #[inline]
    pub(super) fn new_leaf(kind: ExpressionKind) -> Self {
        debug_assert!(kind.validate_inputs(0).is_ok());
        Self::from_parts(kind, SmallVec::new(), crate::origin::current())
    }

    /// Create a non-distinct unresolved function call from an owned argument vector.
    /// Wide argument lists retain their buffer; up to two arguments stay inline.
    #[inline]
    pub fn function(name: impl Into<String>, args: Vec<Expression>) -> Self {
        let name = name.into();
        Self::new_function(name, normalize_arguments(args))
    }

    /// Create a non-distinct unresolved function call from ordered inputs.
    /// Arrays avoid a temporary argument vector for unary and binary calls.
    /// Pass an owned vector to [`Expression::function`] to reuse a wide argument buffer.
    #[inline]
    pub fn function_with_inputs(
        name: impl Into<String>,
        args: impl IntoIterator<Item = Expression>,
    ) -> Self {
        let name = name.into();
        Self::new_function(name, collect_inputs(args))
    }

    #[inline]
    fn new_function(name: String, children: SmallVec<[Expression; 2]>) -> Self {
        // Unresolved functions accept every input count. The generic construction
        // and rebuilding APIs still validate operator-specific input layouts.
        Self::from_parts(
            ExpressionKind::UnresolvedFunction {
                name,
                is_distinct: false,
            },
            children,
            crate::origin::current(),
        )
    }

    #[inline]
    fn from_parts(
        kind: ExpressionKind,
        children: SmallVec<[Expression; 2]>,
        origin: Option<Arc<crate::origin::Origin>>,
    ) -> Self {
        let ty = kind.declared_type();
        Self::allocate(kind, children, ty, origin)
    }

    #[inline]
    fn allocate(
        kind: ExpressionKind,
        children: SmallVec<[Expression; 2]>,
        ty: ExpressionType,
        origin: Option<Arc<crate::origin::Origin>>,
    ) -> Self {
        Self {
            node: Arc::new(ExpressionNode {
                kind,
                children,
                ty,
                origin,
            }),
        }
    }

    /// Borrow this node's operator payload without cloning its inputs or metadata.
    pub fn kind(&self) -> &ExpressionKind {
        &self.node.kind
    }
    /// Borrow structural inputs in the layout described by the expression design guide.
    pub fn children(&self) -> &[Expression] {
        &self.node.children
    }
    /// Borrow the explicit type declaration, or [`ExpressionType::Unresolved`].
    pub fn ty(&self) -> &ExpressionType {
        &self.node.ty
    }
    /// Return whether two handles share the same complete node allocation.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.node, &other.node)
    }

    /// Rebuild with the same payload and origin, retaining each supplied input handle.
    /// The rebuilt root has its own allocation even if its inputs are unchanged.
    ///
    /// # Errors
    /// Returns [`InvalidExpressionInputs`] for an input count inconsistent with the payload.
    pub fn with_children(
        &self,
        children: impl IntoIterator<Item = Expression>,
    ) -> std::result::Result<Self, InvalidExpressionInputs> {
        self.rebuild_collected(collect_inputs(children))
    }

    fn rebuild_collected(
        &self,
        children: SmallVec<[Expression; 2]>,
    ) -> std::result::Result<Self, InvalidExpressionInputs> {
        // Reject malformed layouts before cloning recursive or expensive payloads.
        self.kind().validate_inputs(children.len())?;
        Ok(Self::allocate(
            self.kind().clone(),
            children,
            self.node.ty.clone(),
            self.node.origin.clone(),
        ))
    }

    /// Visit each shared allocation once in first-encounter preorder, iteratively.
    /// Expected work is O(nodes + input edges). Opaque payloads and referenced plans are excluded.
    pub fn visit(&self, mut visitor: impl FnMut(&Expression)) {
        let mut pending = vec![self];
        let mut visited = HashSet::new();
        while let Some(expression) = pending.pop() {
            if visited.insert(Arc::as_ptr(&expression.node)) {
                visitor(expression);
                pending.extend(expression.children().iter().rev());
            }
        }
    }
    /// Visit every input occurrence in preorder, iteratively. Shared DAGs can expand
    /// exponentially. Opaque payloads and referenced plans are excluded.
    pub fn visit_all_paths(&self, mut visitor: impl FnMut(&Expression)) {
        let mut pending = vec![self];
        while let Some(expression) = pending.pop() {
            visitor(expression);
            pending.extend(expression.children().iter().rev());
        }
    }

    /// Rewrite structural inputs bottom-up once per original allocation, retaining sharing.
    /// Unchanged inputs retain their original node and origin. Replacement subtrees are not
    /// traversed again. Opaque protobuf contents and referenced plans are excluded.
    ///
    /// # Errors
    /// Propagates callback errors and errors rebuilding changed structural inputs.
    ///
    /// # Examples
    /// ```
    /// use spark_connect::expression::{Expression, InvalidExpressionInputs};
    /// let child = Expression::column("value");
    /// let root = Expression::function("+", vec![child.clone(), child]);
    /// let rewritten = root.try_transform_up::<InvalidExpressionInputs>(|node| Ok(node.clone()))?;
    /// assert!(root.ptr_eq(&rewritten));
    /// assert!(rewritten.children()[0].ptr_eq(&rewritten.children()[1]));
    /// # Ok::<(), InvalidExpressionInputs>(())
    /// ```
    pub fn try_transform_up<E>(
        &self,
        mut rewrite: impl FnMut(&Expression) -> std::result::Result<Expression, E>,
    ) -> std::result::Result<Expression, E>
    where
        E: From<InvalidExpressionInputs>,
    {
        let mut results: HashMap<*const ExpressionNode, Expression> = HashMap::new();
        let mut pending = vec![(self, false)];
        while let Some((expression, exiting)) = pending.pop() {
            let key = Arc::as_ptr(&expression.node);
            if results.contains_key(&key) {
                continue;
            }
            if !exiting {
                pending.push((expression, true));
                pending.extend(
                    expression
                        .children()
                        .iter()
                        .rev()
                        .map(|child| (child, false)),
                );
            } else {
                let unchanged = expression
                    .children()
                    .iter()
                    .all(|child| child.ptr_eq(&results[&Arc::as_ptr(&child.node)]));
                let candidate = if unchanged {
                    expression.clone()
                } else {
                    expression.rebuild_collected(collect_inputs(
                        expression
                            .children()
                            .iter()
                            .map(|child| results[&Arc::as_ptr(&child.node)].clone()),
                    ))?
                };
                results.insert(key, rewrite(&candidate)?);
            }
        }
        Ok(results
            .remove(&Arc::as_ptr(&self.node))
            .expect("root was rewritten"))
    }

    /// Normalize a typed construction specification.
    ///
    /// # Panics
    /// Panics if the normalization produces an inconsistent input layout, an internal bug.
    pub(super) fn from_spec(spec: ExpressionSpec) -> Self {
        use ExpressionKind as K;
        use ExpressionSpec as S;
        let mut children = SmallVec::<[Expression; 2]>::new();
        let kind = match spec {
            S::Literal(value) => K::Literal(value),
            S::ColumnReference(value) => K::ColumnReference(value),
            S::UnresolvedStar(target, id) => K::UnresolvedStar(target, id),
            S::UnresolvedRegex(name, id) => K::UnresolvedRegex(name, id),
            S::SQLExpression(value) => K::SQLExpression(value),
            S::UnresolvedNamedLambdaVariable(value) => K::UnresolvedNamedLambdaVariable(value),
            S::Sent(value) => K::Sent(Arc::new(SentExpressionPayload::new(*value))),
            S::UnresolvedFunction(value) => {
                children = normalize_arguments(value.args);
                K::UnresolvedFunction {
                    name: value.name,
                    is_distinct: value.is_distinct,
                }
            }
            S::Alias(value) => {
                let Alias {
                    child,
                    names,
                    metadata,
                } = *value;
                children.push(child);
                K::Alias { names, metadata }
            }
            S::Cast(value) => {
                let Cast {
                    child,
                    target,
                    eval_mode,
                } = *value;
                children.push(child);
                K::Cast { target, eval_mode }
            }
            S::DirectShufflePartitionId(child) => {
                children.push(child);
                K::DirectShufflePartitionId
            }
            S::SortOrder(value) => {
                let SortOrder {
                    child,
                    ascending,
                    null_ordering,
                } = *value;
                children.push(child);
                K::SortOrder {
                    ascending,
                    null_ordering,
                }
            }
            S::CaseWhen(case) => {
                let branch_count = case.branches.len();
                let has_else = case.else_expr.is_some();
                children.reserve_exact(branch_count * 2 + usize::from(has_else));
                for (condition, branch_value) in case.branches {
                    children.push(condition);
                    children.push(branch_value);
                }
                children.extend(case.else_expr);
                K::CaseWhen {
                    branch_count,
                    has_else,
                }
            }
            S::UnresolvedExtractValue(value) => {
                children.push(value.child);
                children.push(value.extraction);
                K::UnresolvedExtractValue
            }
            S::UpdateFields(value) => {
                children.push(value.struct_expression);
                let has_value = value.value_expression.is_some();
                children.extend(value.value_expression);
                K::UpdateFields {
                    field_name: value.field_name,
                    has_value,
                }
            }
            S::CallFunction(value) => {
                children = normalize_arguments(value.arguments);
                K::CallFunction {
                    function_name: value.function_name,
                }
            }
            S::WindowExpression(value) => {
                let partition_count = value.partition_spec.len();
                let order_count = value.order_spec.len();
                children.reserve_exact(1 + partition_count + order_count);
                children.push(value.window_function);
                children.extend(value.partition_spec);
                let order_spec = value
                    .order_spec
                    .into_iter()
                    .map(|order| {
                        children.push(order.child);
                        SortOrderProperties {
                            ascending: order.ascending,
                            null_ordering: order.null_ordering,
                        }
                    })
                    .collect();
                K::WindowExpression {
                    partition_count,
                    order_spec,
                    frame_spec: value.frame_spec.map(Arc::new),
                }
            }
            S::LambdaFunction(value) => {
                children.push(value.function);
                K::LambdaFunction {
                    arguments: value.arguments,
                }
            }
            S::CommonInlineUserDefinedFunction(value) => {
                children = normalize_arguments(value.arguments);
                K::CommonInlineUserDefinedFunction {
                    function_name: value.function_name,
                    deterministic: value.deterministic,
                    python_udf: Arc::new(value.python_udf),
                }
            }
            S::NamedArgument(value) => {
                children.push(value.value);
                K::NamedArgument { key: value.key }
            }
            S::Subquery(value) => {
                let partition_count = value.partition_spec.len();
                let order_count = value.order_spec.len();
                let in_value_count = value.in_subquery_values.len();
                children.reserve_exact(partition_count + order_count + in_value_count);
                children.extend(value.partition_spec);
                children.extend(value.order_spec);
                children.extend(value.in_subquery_values);
                K::Subquery(SubqueryPlan {
                    plan: value.plan,
                    plan_id: value.plan_id,
                    subquery_type: value.subquery_type,
                    partition_count,
                    order_count,
                    in_value_count,
                    with_single_partition: value.with_single_partition,
                })
            }
        };
        kind.validate_inputs(children.len())
            .expect("typed expression builder has its declared input layout");
        Self::from_parts(kind, children, crate::origin::current())
    }

    /// Reconstruct an owned typed builder specification, sharing its expression inputs.
    /// Prefer kind and children for borrowed inspection. Opaque Sent payloads are cloned.
    pub fn to_spec(&self) -> ExpressionSpec {
        use ExpressionKind as K;
        use ExpressionSpec as S;
        let children = self.children();
        match self.kind() {
            K::Literal(value) => S::Literal(value.clone()),
            K::ColumnReference(value) => S::ColumnReference(value.clone()),
            K::UnresolvedStar(target, id) => S::UnresolvedStar(target.clone(), *id),
            K::UnresolvedRegex(name, id) => S::UnresolvedRegex(name.clone(), *id),
            K::SQLExpression(value) => S::SQLExpression(value.clone()),
            K::UnresolvedNamedLambdaVariable(value) => {
                S::UnresolvedNamedLambdaVariable(value.clone())
            }
            K::Sent(value) => S::Sent(Box::new(value.to_spec())),
            K::UnresolvedFunction { name, is_distinct } => {
                S::UnresolvedFunction(UnresolvedFunction {
                    name: name.clone(),
                    args: children.to_vec(),
                    is_distinct: *is_distinct,
                })
            }
            K::Alias { names, metadata } => S::Alias(Box::new(Alias {
                child: children[0].clone(),
                names: names.clone(),
                metadata: metadata.clone(),
            })),
            K::Cast { target, eval_mode } => S::Cast(Box::new(Cast {
                child: children[0].clone(),
                target: target.clone(),
                eval_mode: *eval_mode,
            })),
            K::DirectShufflePartitionId => S::DirectShufflePartitionId(children[0].clone()),
            K::SortOrder {
                ascending,
                null_ordering,
            } => S::SortOrder(Box::new(SortOrder {
                child: children[0].clone(),
                ascending: *ascending,
                null_ordering: *null_ordering,
            })),
            K::CaseWhen {
                branch_count,
                has_else,
            } => S::CaseWhen(Box::new(CaseWhen {
                branches: children[..branch_count * 2]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|[condition, value]| (condition.clone(), value.clone()))
                    .collect(),
                else_expr: has_else.then(|| children[branch_count * 2].clone()),
            })),
            K::UnresolvedExtractValue => S::UnresolvedExtractValue(Box::new(ExtractValue {
                child: children[0].clone(),
                extraction: children[1].clone(),
            })),
            K::UpdateFields {
                field_name,
                has_value,
            } => S::UpdateFields(Box::new(UpdateFieldsExpr {
                struct_expression: children[0].clone(),
                field_name: field_name.clone(),
                value_expression: has_value.then(|| children[1].clone()),
            })),
            K::CallFunction { function_name } => S::CallFunction(Box::new(CallFunctionWrapper {
                function_name: function_name.clone(),
                arguments: children.to_vec(),
            })),
            K::WindowExpression {
                partition_count,
                order_spec,
                frame_spec,
            } => S::WindowExpression(Box::new(WindowExpressionWrapper {
                window_function: children[0].clone(),
                partition_spec: children[1..1 + partition_count].to_vec(),
                order_spec: order_spec
                    .iter()
                    .zip(&children[1 + partition_count..])
                    .map(|(properties, child)| SortOrder {
                        child: child.clone(),
                        ascending: properties.ascending,
                        null_ordering: properties.null_ordering,
                    })
                    .collect(),
                frame_spec: frame_spec.as_deref().cloned(),
            })),
            K::LambdaFunction { arguments } => S::LambdaFunction(Box::new(LambdaFunction {
                function: children[0].clone(),
                arguments: arguments.clone(),
            })),
            K::CommonInlineUserDefinedFunction {
                function_name,
                deterministic,
                python_udf,
            } => S::CommonInlineUserDefinedFunction(Box::new(
                CommonInlineUserDefinedFunctionExpression {
                    function_name: function_name.clone(),
                    deterministic: *deterministic,
                    arguments: children.to_vec(),
                    python_udf: python_udf.as_ref().clone(),
                },
            )),
            K::NamedArgument { key } => S::NamedArgument(Box::new(NamedArgument {
                key: key.clone(),
                value: children[0].clone(),
            })),
            K::Subquery(value) => S::Subquery(Box::new(SubqueryExpression {
                plan: value.plan.clone(),
                plan_id: value.plan_id,
                subquery_type: value.subquery_type,
                partition_spec: children[..value.partition_count].to_vec(),
                order_spec: children
                    [value.partition_count..value.partition_count + value.order_count]
                    .to_vec(),
                with_single_partition: value.with_single_partition,
                in_subquery_values: children[value.partition_count + value.order_count..].to_vec(),
            })),
        }
    }
}
