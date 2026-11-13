//! Stack sizing for the recursive protobuf clone of an opaque sent expression.

use super::{proto, SentExpression};

// Count intermediate message layers in the sizing walk, then leave room for
// generated Clone, Option, Box and iterator frames at each layer. These are
// conservative budgets, not a platform-independent proof of stack usage.
const CLONE_STACK_PER_LEVEL: usize = if cfg!(debug_assertions) {
    16 << 10
} else {
    4 << 10
};

/// An immutable sent-expression payload with cached physical protobuf depth.
///
/// Construction moves the mutable specification into this wrapper and measures
/// its nesting once. Accessors expose shared references only, so the cached
/// depth remains valid for the lifetime of the payload. The supplied logical
/// expression levels remain separate from physical stack sizing.
#[derive(Debug)]
pub struct SentExpressionPayload {
    original: SentExpression,
    actual_depth: usize,
}

impl SentExpressionPayload {
    /// Retain a sent specification and measure its actual protobuf nesting.
    ///
    /// This preserves the supplied logical levels without validating them, as the
    /// typed construction specification does. Use [`Expression::sent`](super::Expression::sent)
    /// for checked logical-depth construction; physical stack sizing is independent.
    pub fn new(sent: SentExpression) -> Self {
        let actual_depth = clone_depth(&sent.expression);
        Self {
            original: sent,
            actual_depth,
        }
    }

    /// The rendering retained when the expression was serialized.
    pub fn rendered(&self) -> &str {
        &self.original.rendered
    }

    /// The caller-supplied logical depth used by request depth checks.
    pub fn levels(&self) -> usize {
        self.original.levels
    }

    /// Borrow the retained protobuf without invalidating its measured depth.
    /// Use [`Self::clone_proto`] for a guarded owned copy.
    pub fn expression(&self) -> &proto::Expression {
        &self.original.expression
    }

    /// Clone the protobuf using its cached physical depth for stack sizing.
    /// Encoding and destruction of the returned protobuf remain recursive.
    pub fn clone_proto(&self) -> proto::Expression {
        clone_with_depth(&self.original.expression, self.actual_depth)
    }

    /// Reconstruct an independently mutable specification with a guarded clone.
    pub fn to_spec(&self) -> SentExpression {
        SentExpression {
            expression: self.clone_proto(),
            rendered: self.original.rendered.clone(),
            levels: self.original.levels,
        }
    }
}

impl PartialEq for SentExpressionPayload {
    fn eq(&self, other: &Self) -> bool {
        // Compare the original values, including protobuf NaNs, without an
        // allocation-identity shortcut. The depth cache is derived metadata.
        self.original == other.original
    }
}

fn clone_with_depth(expression: &proto::Expression, actual_depth: usize) -> proto::Expression {
    let stack = actual_depth
        .saturating_mul(CLONE_STACK_PER_LEVEL)
        .saturating_add(64 << 10);
    stacker::maybe_grow(stack, stack, || expression.clone())
}

impl SentExpression {
    /// Clone the wire payload using its measured nesting, independently of the
    /// logical expression depth supplied when the Column was pickled.
    pub(super) fn clone_proto(&self) -> proto::Expression {
        // This specification has public mutable fields, so a prior depth
        // measurement could be stale. Canonical payloads use the immutable
        // SentExpressionPayload cache instead.
        clone_with_depth(&self.expression, clone_depth(&self.expression))
    }
}

impl Clone for SentExpression {
    fn clone(&self) -> Self {
        Self {
            expression: self.clone_proto(),
            rendered: self.rendered.clone(),
            levels: self.levels,
        }
    }
}

enum CloneInput<'a> {
    Expression(&'a proto::Expression),
    Literal(&'a proto::expression::Literal),
    DataType(&'a proto::DataType),
}

/// Measure recursive messages without recursing. Intermediate wrapper messages
/// count too: a window frame's expression, for example, is four messages below
/// its enclosing Expression. Opaque extension bytes are not decoded.
#[allow(deprecated)]
fn clone_depth(expression: &proto::Expression) -> usize {
    use proto::data_type::Kind;
    use proto::expression::literal::LiteralType;
    use proto::expression::ExprType;

    let mut pending = vec![(CloneInput::Expression(expression), 1_usize)];
    let mut deepest = 1;
    while let Some((input, level)) = pending.pop() {
        deepest = deepest.max(level);
        let below = level.saturating_add(2);
        match input {
            CloneInput::Expression(expression) => {
                let Some(kind) = &expression.expr_type else {
                    continue;
                };
                match kind {
                    ExprType::Literal(value) => {
                        pending.push((CloneInput::Literal(value), level.saturating_add(1)));
                    }
                    ExprType::UnresolvedFunction(value) => {
                        pending.extend(
                            value
                                .arguments
                                .iter()
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                    }
                    ExprType::Alias(value) => {
                        pending.extend(
                            value
                                .expr
                                .iter()
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                    }
                    ExprType::Cast(value) => {
                        pending.extend(
                            value
                                .expr
                                .iter()
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                        if let Some(proto::expression::cast::CastToType::Type(data_type)) =
                            &value.cast_to_type
                        {
                            pending.push((CloneInput::DataType(data_type), below));
                        }
                    }
                    ExprType::SortOrder(value) => {
                        pending.extend(
                            value
                                .child
                                .iter()
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                    }
                    ExprType::LambdaFunction(value) => {
                        pending.extend(
                            value
                                .function
                                .iter()
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                    }
                    ExprType::Window(value) => {
                        pending.extend(
                            value
                                .window_function
                                .iter()
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                        pending.extend(
                            value
                                .partition_spec
                                .iter()
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                        for order in &value.order_spec {
                            pending.extend(order.child.iter().map(|child| {
                                (CloneInput::Expression(child), level.saturating_add(3))
                            }));
                        }
                        if let Some(frame) = &value.frame_spec {
                            for boundary in frame.lower.iter().chain(&frame.upper) {
                                if let Some(proto::expression::window::window_frame::frame_boundary::Boundary::Value(child)) = &boundary.boundary {
                                    pending.push((CloneInput::Expression(child), level.saturating_add(4)));
                                }
                            }
                        }
                    }
                    ExprType::UnresolvedExtractValue(value) => {
                        pending.extend(
                            value
                                .child
                                .iter()
                                .chain(&value.extraction)
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                    }
                    ExprType::UpdateFields(value) => {
                        pending.extend(
                            value
                                .struct_expression
                                .iter()
                                .chain(&value.value_expression)
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                    }
                    ExprType::CommonInlineUserDefinedFunction(value) => {
                        pending.extend(
                            value
                                .arguments
                                .iter()
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                        let type_level = level.saturating_add(3);
                        match &value.function {
                            Some(proto::common_inline_user_defined_function::Function::PythonUdf(udf)) => {
                                pending.extend(udf.output_type.iter().map(|ty| (CloneInput::DataType(ty), type_level)));
                            }
                            Some(proto::common_inline_user_defined_function::Function::ScalarScalaUdf(udf)) => {
                                push_scala_types(&mut pending, udf, type_level);
                            }
                            Some(proto::common_inline_user_defined_function::Function::JavaUdf(udf)) => {
                                pending.extend(udf.output_type.iter().map(|ty| (CloneInput::DataType(ty), type_level)));
                            }
                            None => {}
                        }
                    }
                    ExprType::CallFunction(value) => {
                        pending.extend(
                            value
                                .arguments
                                .iter()
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                    }
                    ExprType::NamedArgumentExpression(value) => {
                        pending.extend(
                            value
                                .value
                                .iter()
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                    }
                    ExprType::MergeAction(value) => {
                        pending.extend(
                            value
                                .condition
                                .iter()
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                        for assignment in &value.assignments {
                            pending.extend(assignment.key.iter().chain(&assignment.value).map(
                                |child| (CloneInput::Expression(child), level.saturating_add(3)),
                            ));
                        }
                    }
                    ExprType::TypedAggregateExpression(value) => {
                        if let Some(udf) = &value.scalar_scala_udf {
                            push_scala_types(&mut pending, udf, level.saturating_add(3));
                        }
                    }
                    ExprType::SubqueryExpression(value) => {
                        pending.extend(
                            value
                                .in_subquery_values
                                .iter()
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                        if let Some(options) = &value.table_arg_options {
                            pending.extend(options.partition_spec.iter().map(|child| {
                                (CloneInput::Expression(child), level.saturating_add(3))
                            }));
                            for order in &options.order_spec {
                                pending.extend(order.child.iter().map(|child| {
                                    (CloneInput::Expression(child), level.saturating_add(4))
                                }));
                            }
                        }
                    }
                    ExprType::DirectShufflePartitionId(value) => {
                        pending.extend(
                            value
                                .child
                                .iter()
                                .map(|child| (CloneInput::Expression(child), below)),
                        );
                    }
                    ExprType::UnresolvedAttribute(_)
                    | ExprType::ExpressionString(_)
                    | ExprType::UnresolvedStar(_)
                    | ExprType::UnresolvedRegex(_)
                    | ExprType::UnresolvedNamedLambdaVariable(_)
                    | ExprType::Extension(_) => {}
                }
            }
            CloneInput::Literal(literal) => {
                pending.extend(
                    literal
                        .data_type
                        .iter()
                        .map(|ty| (CloneInput::DataType(ty), level.saturating_add(1))),
                );
                let Some(kind) = &literal.literal_type else {
                    continue;
                };
                match kind {
                    LiteralType::Null(data_type) => {
                        pending.push((CloneInput::DataType(data_type), level.saturating_add(1)));
                    }
                    LiteralType::Array(value) => {
                        pending.extend(
                            value
                                .element_type
                                .iter()
                                .map(|ty| (CloneInput::DataType(ty), below)),
                        );
                        pending.extend(
                            value
                                .elements
                                .iter()
                                .map(|child| (CloneInput::Literal(child), below)),
                        );
                    }
                    LiteralType::Map(value) => {
                        pending.extend(
                            value
                                .key_type
                                .iter()
                                .chain(&value.value_type)
                                .map(|ty| (CloneInput::DataType(ty), below)),
                        );
                        pending.extend(
                            value
                                .keys
                                .iter()
                                .chain(&value.values)
                                .map(|child| (CloneInput::Literal(child), below)),
                        );
                    }
                    LiteralType::Struct(value) => {
                        pending.extend(
                            value
                                .struct_type
                                .iter()
                                .map(|ty| (CloneInput::DataType(ty), below)),
                        );
                        pending.extend(
                            value
                                .elements
                                .iter()
                                .map(|child| (CloneInput::Literal(child), below)),
                        );
                    }
                    LiteralType::Binary(_)
                    | LiteralType::Boolean(_)
                    | LiteralType::Byte(_)
                    | LiteralType::Short(_)
                    | LiteralType::Integer(_)
                    | LiteralType::Long(_)
                    | LiteralType::Float(_)
                    | LiteralType::Double(_)
                    | LiteralType::Decimal(_)
                    | LiteralType::String(_)
                    | LiteralType::Date(_)
                    | LiteralType::Timestamp(_)
                    | LiteralType::TimestampNtz(_)
                    | LiteralType::CalendarInterval(_)
                    | LiteralType::YearMonthInterval(_)
                    | LiteralType::DayTimeInterval(_)
                    | LiteralType::SpecializedArray(_)
                    | LiteralType::Time(_)
                    | LiteralType::TimestampNtzNanos(_)
                    | LiteralType::TimestampLtzNanos(_) => {}
                }
            }
            CloneInput::DataType(data_type) => {
                let Some(kind) = &data_type.kind else {
                    continue;
                };
                match kind {
                    Kind::Array(value) => {
                        pending.extend(
                            value
                                .element_type
                                .iter()
                                .map(|ty| (CloneInput::DataType(ty), below)),
                        );
                    }
                    Kind::Struct(value) => {
                        for field in &value.fields {
                            pending.extend(
                                field
                                    .data_type
                                    .iter()
                                    .map(|ty| (CloneInput::DataType(ty), level.saturating_add(3))),
                            );
                        }
                    }
                    Kind::Map(value) => {
                        pending.extend(
                            value
                                .key_type
                                .iter()
                                .chain(&value.value_type)
                                .map(|ty| (CloneInput::DataType(ty), below)),
                        );
                    }
                    Kind::Udt(value) => {
                        pending.extend(
                            value
                                .sql_type
                                .iter()
                                .map(|ty| (CloneInput::DataType(ty), below)),
                        );
                    }
                    Kind::Null(_)
                    | Kind::Binary(_)
                    | Kind::Boolean(_)
                    | Kind::Byte(_)
                    | Kind::Short(_)
                    | Kind::Integer(_)
                    | Kind::Long(_)
                    | Kind::Float(_)
                    | Kind::Double(_)
                    | Kind::Decimal(_)
                    | Kind::String(_)
                    | Kind::Char(_)
                    | Kind::VarChar(_)
                    | Kind::Date(_)
                    | Kind::Timestamp(_)
                    | Kind::TimestampNtz(_)
                    | Kind::CalendarInterval(_)
                    | Kind::YearMonthInterval(_)
                    | Kind::DayTimeInterval(_)
                    | Kind::Variant(_)
                    | Kind::Geometry(_)
                    | Kind::Geography(_)
                    | Kind::Unparsed(_)
                    | Kind::Time(_)
                    | Kind::TimestampNtzNanos(_)
                    | Kind::TimestampLtzNanos(_) => {}
                }
            }
        }
    }
    deepest
}

fn push_scala_types<'a>(
    pending: &mut Vec<(CloneInput<'a>, usize)>,
    udf: &'a proto::ScalarScalaUdf,
    level: usize,
) {
    pending.extend(
        udf.input_types
            .iter()
            .chain(&udf.output_type)
            .map(|ty| (CloneInput::DataType(ty), level)),
    );
}
