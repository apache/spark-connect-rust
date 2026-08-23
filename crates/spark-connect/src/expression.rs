//! Expression tree mirroring PySpark's `pyspark.sql.connect.expressions`.
//!
//! Defines the Expression hierarchy and conversion functions to Spark Connect protobufs.

use std::sync::atomic::{AtomicU32, Ordering};

use spark_connect_proto as proto;

use crate::types::DataType;
use crate::udf::CommonInlineUserDefinedFunctionExpression;

/// Thread-local counter for generating unique lambda variable names.
static LAMBDA_VAR_COUNTER: AtomicU32 = AtomicU32::new(0);

/// Get the next unique lambda variable suffix.
pub fn next_lambda_var_index() -> u32 {
    LAMBDA_VAR_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// The base Expression type, mirroring `pyspark.sql.connect.expressions.Expression`.
///
/// All expressions are represented as variants of this enum. Each variant
/// carries the data needed to construct the corresponding proto expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// `pyspark.sql.connect.expressions.LiteralExpression` - a constant value.
    Literal(LiteralExpression),
    /// `pyspark.sql.connect.expressions.ColumnReference` - a reference to a column.
    ColumnReference(ColumnReference),
    /// `pyspark.sql.connect.expressions.UnresolvedFunction` - a function call.
    UnresolvedFunction(UnresolvedFunction),
    /// `pyspark.sql.connect.expressions.UnresolvedStar` - a star expression (`*`).
    /// Carries an optional target ending in `.*` (e.g. `df.*` from `col("df.*")`).
    UnresolvedStar(Option<String>),
    /// `pyspark.sql.connect.expressions.ColumnAlias` / `Alias` - an aliased expression.
    Alias(Box<Alias>),
    /// `pyspark.sql.connect.expressions.CastExpression` - a cast expression.
    Cast(Box<Cast>),
    /// `pyspark.sql.connect.expressions.UnresolvedRegex` - a regex column reference.
    UnresolvedRegex(String),
    /// `pyspark.sql.connect.expressions.SortOrder` - a sort order expression.
    SortOrder(Box<SortOrder>),
    /// `pyspark.sql.connect.expressions.CaseWhen` - a CASE WHEN expression.
    CaseWhen(Box<CaseWhen>),
    /// `pyspark.sql.connect.expressions.UnresolvedExtractValue` - `col[k]` / `getField`.
    UnresolvedExtractValue(Box<ExtractValue>),
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
}

impl Expression {
    /// Converts the expression to a Spark Connect protobuf expression.
    pub fn to_proto(&self) -> proto::Expression {
        match self {
            Expression::Literal(lit) => lit.to_proto(),
            Expression::ColumnReference(col_ref) => col_ref.to_proto(),
            Expression::UnresolvedFunction(func) => func.to_proto(),
            Expression::UnresolvedStar(target) => {
                let mut expr = proto::Expression::default();
                expr.expr_type = Some(proto::expression::ExprType::UnresolvedStar(
                    proto::expression::UnresolvedStar {
                        unparsed_target: target.clone(),
                        plan_id: None,
                    },
                ));
                expr
            }
            Expression::Alias(alias) => alias.to_proto(),
            Expression::Cast(cast) => cast.to_proto(),
            Expression::UnresolvedRegex(col_name) => {
                let mut expr = proto::Expression::default();
                expr.expr_type = Some(proto::expression::ExprType::UnresolvedRegex(
                    proto::expression::UnresolvedRegex {
                        col_name: col_name.clone(),
                        plan_id: Some(0),
                    },
                ));
                expr
            }
            Expression::SortOrder(sort) => sort.to_proto(),
            Expression::CaseWhen(case_when) => case_when.to_proto(),
            Expression::UnresolvedExtractValue(ev) => ev.to_proto(),
            Expression::SQLExpression(sql) => {
                let mut expr = proto::Expression::default();
                expr.expr_type = Some(proto::expression::ExprType::ExpressionString(
                    proto::expression::ExpressionString {
                        expression: sql.clone(),
                    },
                ));
                expr
            }
            Expression::CallFunction(cf) => cf.to_proto(),
            Expression::WindowExpression(we) => we.to_proto(),
            Expression::LambdaFunction(lf) => lf.to_proto(),
            Expression::UnresolvedNamedLambdaVariable(var) => var.to_proto(),
            Expression::CommonInlineUserDefinedFunction(udf) => {
                let mut expr = proto::Expression::default();
                expr.expr_type = Some(
                    proto::expression::ExprType::CommonInlineUserDefinedFunction(udf.to_proto()),
                );
                expr
            }
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
            LiteralExpression::Time { nano, precision } => {
                let mut time = proto::expression::literal::Time::default();
                time.nano = *nano;
                time.precision = Some(*precision);
                proto::expression::literal::LiteralType::Time(time)
            }
            LiteralExpression::Array {
                element_type: _,
                elements,
            } => {
                let mut array = proto::expression::literal::Array::default();
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
}

impl ColumnReference {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            plan_id: None,
        }
    }

    /// Bind this attribute to a DataFrame plan id.
    pub fn with_plan_id(mut self, plan_id: i64) -> Self {
        self.plan_id = Some(plan_id);
        self
    }

    pub fn to_proto(&self) -> proto::Expression {
        let mut expr = proto::Expression::default();
        expr.expr_type = Some(proto::expression::ExprType::UnresolvedAttribute(
            proto::expression::UnresolvedAttribute {
                unparsed_identifier: self.name.clone(),
                plan_id: self.plan_id,
                is_metadata_column: Some(false),
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
    pub else_expr: Option<Box<Expression>>,
}

impl CaseWhen {
    pub fn new(branches: Vec<(Expression, Expression)>) -> Self {
        Self {
            branches,
            else_expr: None,
        }
    }

    pub fn with_else(mut self, else_expr: Expression) -> Self {
        self.else_expr = Some(Box::new(else_expr));
        self
    }

    pub fn to_proto(&self) -> proto::Expression {
        let mut args = Vec::new();
        for (condition, value) in &self.branches {
            args.push(condition.clone());
            args.push(value.clone());
        }
        if let Some(else_expr) = &self.else_expr {
            args.push((**else_expr).clone());
        }
        let func = UnresolvedFunction::new("when", args);
        func.to_proto()
    }
}

/// Wrapper for `spark.connect.CallFunction` protobuf message.
#[derive(Debug, Clone, PartialEq)]
pub struct CallFunctionWrapper {
    pub function_name: String,
}

impl CallFunctionWrapper {
    /// Create a new CallFunctionWrapper.
    pub fn new(function_name: impl Into<String>) -> Self {
        CallFunctionWrapper {
            function_name: function_name.into(),
        }
    }

    /// Convert to protobuf.
    pub fn to_proto(&self) -> proto::Expression {
        let mut expr = proto::Expression::default();
        expr.expr_type = Some(proto::expression::ExprType::CallFunction(
            proto::CallFunction {
                function_name: self.function_name.clone(),
                arguments: vec![],
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
            frame.lower = Some(Box::new(to_proto_frame_boundary(lower)));
            frame.upper = Some(Box::new(to_proto_frame_boundary(upper)));
            window.frame_spec = Some(Box::new(frame));
        }

        expr.expr_type = Some(proto::expression::ExprType::Window(Box::new(window)));
        expr
    }
}

/// Convert FrameBoundary to proto FrameBoundary.
fn to_proto_frame_boundary(
    boundary: &FrameBoundary,
) -> proto::expression::window::window_frame::FrameBoundary {
    let mut proto_boundary = proto::expression::window::window_frame::FrameBoundary::default();
    proto_boundary.boundary = match boundary {
        FrameBoundary::UnboundedPreceding => {
            Some(proto::expression::window::window_frame::frame_boundary::Boundary::Unbounded(true))
        }
        FrameBoundary::UnboundedFollowing => {
            Some(proto::expression::window::window_frame::frame_boundary::Boundary::Unbounded(true))
        }
        FrameBoundary::CurrentRow => Some(
            proto::expression::window::window_frame::frame_boundary::Boundary::CurrentRow(true),
        ),
        FrameBoundary::Preceding(n) => Some(
            proto::expression::window::window_frame::frame_boundary::Boundary::Value(Box::new(
                Expression::Literal(LiteralExpression::long(*n)).to_proto(),
            )),
        ),
        FrameBoundary::Following(n) => Some(
            proto::expression::window::window_frame::frame_boundary::Boundary::Value(Box::new(
                Expression::Literal(LiteralExpression::long(*n)).to_proto(),
            )),
        ),
    };
    proto_boundary
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
