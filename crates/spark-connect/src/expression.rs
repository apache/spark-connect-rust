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
    /// `spark.connect.Expression.DirectShufflePartitionID` - wraps a child expression
    /// that evaluates to a partition id (used by `DataFrame.repartitionById`).
    DirectShufflePartitionId(Box<Expression>),
    /// `pyspark.sql.connect.expressions.UnresolvedRegex` - a regex column reference.
    UnresolvedRegex(String),
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
}

impl Expression {
    /// Render this expression to a human-readable string, mirroring the
    /// `__repr__` of `pyspark.sql.connect.expressions.*`.
    ///
    /// This is what `Column.__repr__` wraps as `Column<'...'>`. pandas-on-Spark's
    /// `spark_column_equals` compares these strings (after stripping backticks) to
    /// decide column equality, so the rendering must be deterministic and mirror
    /// PySpark's format for the common operators.
    pub fn render(&self) -> String {
        match self {
            Expression::Literal(lit) => lit.render(),
            Expression::ColumnReference(col_ref) => col_ref.name.clone(),
            Expression::UnresolvedFunction(func) => func.render(),
            Expression::UnresolvedStar(target) => match target {
                Some(t) => t.clone(),
                None => "*".to_string(),
            },
            Expression::Alias(alias) => {
                let name = if alias.names.len() == 1 {
                    alias.names[0].clone()
                } else {
                    format!("({})", alias.names.join(", "))
                };
                format!("{} AS {}", alias.child.render(), name)
            }
            Expression::Cast(cast) => {
                let type_str = match &cast.target {
                    CastTarget::Type(dt) => dt.simple_string(),
                    CastTarget::TypeStr(s) => s.clone(),
                };
                format!("CAST({} AS {})", cast.child.render(), type_str)
            }
            Expression::DirectShufflePartitionId(child) => {
                format!("DIRECT_SHUFFLE_PARTITION_ID({})", child.render())
            }
            Expression::UnresolvedRegex(col_name) => col_name.clone(),
            Expression::SortOrder(sort) => sort.render(),
            Expression::CaseWhen(case_when) => case_when.render(),
            Expression::UnresolvedExtractValue(ev) => {
                format!("{}[{}]", ev.child.render(), ev.extraction.render())
            }
            Expression::UpdateFields(uf) => match &uf.value_expression {
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
            Expression::SQLExpression(sql) => sql.clone(),
            Expression::CallFunction(_) => format!("{self:?}"),
            Expression::WindowExpression(_) => format!("{self:?}"),
            Expression::LambdaFunction(lf) => format!("{lf:?}"),
            Expression::UnresolvedNamedLambdaVariable(var) => format!("{var:?}"),
            Expression::CommonInlineUserDefinedFunction(_) => format!("{self:?}"),
        }
    }

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
            Expression::DirectShufflePartitionId(child) => {
                let mut expr = proto::Expression::default();
                expr.expr_type = Some(proto::expression::ExprType::DirectShufflePartitionId(
                    Box::new(proto::expression::DirectShufflePartitionId {
                        child: Some(Box::new(child.to_proto())),
                    }),
                ));
                expr
            }
            Expression::UnresolvedRegex(col_name) => {
                let mut expr = proto::Expression::default();
                expr.expr_type = Some(proto::expression::ExprType::UnresolvedRegex(
                    proto::expression::UnresolvedRegex {
                        col_name: col_name.clone(),
                        plan_id: None,
                    },
                ));
                expr
            }
            Expression::SortOrder(sort) => sort.to_proto(),
            Expression::CaseWhen(case_when) => case_when.to_proto(),
            Expression::UnresolvedExtractValue(ev) => ev.to_proto(),
            Expression::UpdateFields(uf) => uf.to_proto(),
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

    /// Render the literal value, mirroring `LiteralExpression.__repr__`.
    pub fn render(&self) -> String {
        match self {
            LiteralExpression::Null(_) => "NULL".to_string(),
            LiteralExpression::Boolean(b) => {
                if *b {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            LiteralExpression::Byte(v)
            | LiteralExpression::Short(v)
            | LiteralExpression::Integer(v) => v.to_string(),
            LiteralExpression::Long(v) => v.to_string(),
            LiteralExpression::Float(v) => v.to_string(),
            LiteralExpression::Double(v) => v.to_string(),
            LiteralExpression::Decimal { value, .. } => value.clone(),
            LiteralExpression::String(v) => v.clone(),
            LiteralExpression::Binary(v) => format!("{v:?}"),
            LiteralExpression::Date(v) => v.to_string(),
            LiteralExpression::Timestamp(v) | LiteralExpression::TimestampNtz(v) => v.to_string(),
            LiteralExpression::Time { nano, .. } => nano.to_string(),
            LiteralExpression::Array { elements, .. } => {
                let inner: Vec<String> = elements.iter().map(|e| e.render()).collect();
                format!("[{}]", inner.join(", "))
            }
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
        const INFIX_OPS: &[&str] = &[
            "+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "and", "or", "&", "|", "^",
            "<=>",
        ];
        if self.args.len() == 2 && INFIX_OPS.contains(&self.name.as_str()) {
            return format!(
                "({} {} {})",
                self.args[0].render(),
                self.name,
                self.args[1].render()
            );
        }
        if self.args.len() == 1 {
            match self.name.as_str() {
                "not" => return format!("(NOT {})", self.args[0].render()),
                "negative" | "negate" => return format!("(- {})", self.args[0].render()),
                _ => {}
            }
        }
        let inner: Vec<String> = self.args.iter().map(|a| a.render()).collect();
        format!("{}({})", self.name, inner.join(", "))
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
            args.push((**else_expr).clone());
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

    fn col(name: &str) -> Expression {
        Expression::ColumnReference(ColumnReference::new(name))
    }

    #[test]
    fn test_render_matches_pyspark_connect_format() {
        // Column reference and literal.
        assert_eq!(col("x").render(), "x");
        assert_eq!(
            Expression::Literal(LiteralExpression::Integer(0)).render(),
            "0"
        );
        assert_eq!(
            Expression::Literal(LiteralExpression::null(DataType::Integer)).render(),
            "NULL"
        );

        // Binary operators render infix, mirroring UnresolvedFunction.__repr__.
        let add = Expression::UnresolvedFunction(UnresolvedFunction::new(
            "+",
            vec![col("x"), Expression::Literal(LiteralExpression::Integer(1))],
        ));
        assert_eq!(add.render(), "(x + 1)");

        let eq =
            Expression::UnresolvedFunction(UnresolvedFunction::new("==", vec![col("a"), col("b")]));
        assert_eq!(eq.render(), "(a == b)");

        // Unary not.
        let neq = Expression::UnresolvedFunction(UnresolvedFunction::new("not", vec![eq.clone()]));
        assert_eq!(neq.render(), "(NOT (a == b))");

        // Non-operator function renders as name(args).
        let f = Expression::UnresolvedFunction(UnresolvedFunction::new(
            "coalesce",
            vec![col("a"), col("b")],
        ));
        assert_eq!(f.render(), "coalesce(a, b)");

        // Alias, cast, star.
        assert_eq!(
            Expression::Alias(Box::new(Alias::new(col("x"), "y"))).render(),
            "x AS y"
        );
        assert_eq!(
            Expression::Cast(Box::new(Cast {
                child: col("x"),
                target: CastTarget::TypeStr("int".to_string()),
                eval_mode: None,
            }))
            .render(),
            "CAST(x AS int)"
        );
        assert_eq!(Expression::UnresolvedStar(None).render(), "*");

        // Equal expressions render identically; different ones differ
        // (the property spark_column_equals relies on).
        assert_eq!(add.render(), add.render());
        assert_ne!(add.render(), eq.render());
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
        assert_eq!(lit.render(), "3.14");
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
        let lit = LiteralExpression::Binary(vec![1, 2, 3]);
        let rendered = lit.render();
        assert!(rendered.contains("[1, 2, 3]"));
    }

    #[test]
    fn test_literal_render_date() {
        let lit = LiteralExpression::Date(18993);
        assert_eq!(lit.render(), "18993");
    }

    #[test]
    fn test_literal_render_timestamp_ntz() {
        let lit = LiteralExpression::TimestampNtz(1693526400000000);
        assert_eq!(lit.render(), "1693526400000000");
    }

    #[test]
    fn test_literal_render_time() {
        let lit = LiteralExpression::Time {
            nano: 3600000000000i64,
            precision: 9,
        };
        assert_eq!(lit.render(), "3600000000000");
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
        let expr = Expression::UnresolvedStar(Some("table.*".to_string()));
        assert_eq!(expr.render(), "table.*");
    }

    #[test]
    fn test_expression_render_unresolved_regex() {
        let expr = Expression::UnresolvedRegex("`col_.*`".to_string());
        assert_eq!(expr.render(), "`col_.*`");
    }

    #[test]
    fn test_expression_render_direct_shuffle_partition_id() {
        let child = col("x");
        let expr = Expression::DirectShufflePartitionId(Box::new(child));
        assert_eq!(expr.render(), "DIRECT_SHUFFLE_PARTITION_ID(x)");
    }

    #[test]
    fn test_expression_render_unresolved_extract_value() {
        let child = col("struct_col");
        let extraction = Expression::Literal(LiteralExpression::string("field"));
        let ev = ExtractValue::new(child, extraction);
        let expr = Expression::UnresolvedExtractValue(Box::new(ev));
        assert_eq!(expr.render(), "struct_col[field]");
    }

    #[test]
    fn test_expression_render_update_fields_with_value() {
        let struct_expr = col("s");
        let value_expr = Expression::Literal(LiteralExpression::int(42));
        let uf = UpdateFieldsExpr::new(struct_expr, "f1", Some(value_expr));
        let expr = Expression::UpdateFields(Box::new(uf));
        assert_eq!(expr.render(), "update_field(s, f1, 42)");
    }

    #[test]
    fn test_expression_render_update_fields_drop() {
        let struct_expr = col("s");
        let uf = UpdateFieldsExpr::new(struct_expr, "f1", None);
        let expr = Expression::UpdateFields(Box::new(uf));
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
        let expr = Expression::SortOrder(Box::new(sort));
        assert_eq!(expr.render(), "a ASC NULLS FIRST");
    }

    #[test]
    fn test_case_when_render_single_branch() {
        let cw = CaseWhen::new(vec![(
            Expression::Literal(LiteralExpression::boolean(true)),
            Expression::Literal(LiteralExpression::int(1)),
        )]);
        assert_eq!(cw.render(), "CASE WHEN true THEN 1 END");
    }

    #[test]
    fn test_case_when_render_multiple_branches() {
        let cw = CaseWhen::new(vec![
            (
                Expression::Literal(LiteralExpression::boolean(true)),
                Expression::Literal(LiteralExpression::int(1)),
            ),
            (
                Expression::Literal(LiteralExpression::boolean(false)),
                Expression::Literal(LiteralExpression::int(2)),
            ),
        ]);
        assert_eq!(cw.render(), "CASE WHEN true THEN 1 WHEN false THEN 2 END");
    }

    #[test]
    fn test_case_when_render_with_else() {
        let cw = CaseWhen::new(vec![(
            Expression::Literal(LiteralExpression::boolean(true)),
            Expression::Literal(LiteralExpression::int(1)),
        )])
        .with_else(Expression::Literal(LiteralExpression::int(99)));
        assert_eq!(cw.render(), "CASE WHEN true THEN 1 ELSE 99 END");
    }

    #[test]
    fn test_expression_render_case_when() {
        let cw = CaseWhen::new(vec![(
            col("cond"),
            Expression::Literal(LiteralExpression::int(1)),
        )]);
        let expr = Expression::CaseWhen(Box::new(cw));
        assert_eq!(expr.render(), "CASE WHEN cond THEN 1 END");
    }

    #[test]
    fn test_unresolved_function_render_negate() {
        let func = UnresolvedFunction::new("negate", vec![col("x")]);
        assert_eq!(func.render(), "(- x)");
    }

    #[test]
    fn test_unresolved_function_render_negative() {
        let func = UnresolvedFunction::new("negative", vec![col("x")]);
        assert_eq!(func.render(), "(- x)");
    }

    #[test]
    fn test_unresolved_function_render_multiple_args() {
        let func = UnresolvedFunction::new(
            "concat",
            vec![
                Expression::Literal(LiteralExpression::string("a")),
                Expression::Literal(LiteralExpression::string("b")),
                Expression::Literal(LiteralExpression::string("c")),
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
        let expr = Expression::Alias(Box::new(alias));
        assert_eq!(expr.render(), "x AS (a, b)");
    }

    #[test]
    fn test_cast_render_with_datatype() {
        let cast = Cast::new(
            col("x"),
            DataType::String {
                collation: "".to_string(),
            },
        );
        let expr = Expression::Cast(Box::new(cast));
        assert_eq!(expr.render(), "CAST(x AS string)");
    }

    #[test]
    fn test_sql_expression_render() {
        let expr = Expression::SQLExpression("SELECT * FROM table".to_string());
        assert_eq!(expr.render(), "SELECT * FROM table");
    }

    #[test]
    fn test_literal_boolean_true_render() {
        let lit = LiteralExpression::boolean(true);
        assert_eq!(lit.render(), "true");
    }

    #[test]
    fn test_literal_boolean_false_render() {
        let lit = LiteralExpression::boolean(false);
        assert_eq!(lit.render(), "false");
    }

    #[test]
    fn test_unresolved_function_render_not() {
        let func = UnresolvedFunction::new(
            "not",
            vec![Expression::Literal(LiteralExpression::boolean(true))],
        );
        assert_eq!(func.render(), "(NOT true)");
    }

    #[test]
    fn test_infix_operators_render() {
        let ops = vec![
            ("+", "(1 + 2)"),
            ("-", "(1 - 2)"),
            ("*", "(1 * 2)"),
            ("/", "(1 / 2)"),
            ("%", "(1 % 2)"),
            ("==", "(1 == 2)"),
            ("!=", "(1 != 2)"),
            ("<", "(1 < 2)"),
            ("<=", "(1 <= 2)"),
            (">", "(1 > 2)"),
            (">=", "(1 >= 2)"),
            ("&", "(1 & 2)"),
            ("|", "(1 | 2)"),
            ("^", "(1 ^ 2)"),
            ("<=>", "(1 <=> 2)"),
        ];

        for (op_name, expected_result) in ops.iter().take(15) {
            let func = UnresolvedFunction::new(
                *op_name,
                vec![
                    Expression::Literal(LiteralExpression::int(1)),
                    Expression::Literal(LiteralExpression::int(2)),
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
                Expression::Literal(LiteralExpression::boolean(true)),
                Expression::Literal(LiteralExpression::boolean(false)),
            ],
        );
        assert_eq!(and_func.render(), "(true and false)");

        let or_func = UnresolvedFunction::new(
            "or",
            vec![
                Expression::Literal(LiteralExpression::boolean(true)),
                Expression::Literal(LiteralExpression::boolean(false)),
            ],
        );
        assert_eq!(or_func.render(), "(true or false)");
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
        let expr = Expression::CallFunction(Box::new(cf));
        // CallFunction renders as debug format
        let rendered = expr.render();
        assert!(rendered.contains("CallFunctionWrapper"));
    }

    #[test]
    fn test_window_expression_render() {
        let we = WindowExpressionWrapper::new(
            Expression::UnresolvedFunction(UnresolvedFunction::new("sum", vec![col("x")])),
            vec![col("group_col")],
            vec![],
            None,
        );
        let expr = Expression::WindowExpression(Box::new(we));
        let rendered = expr.render();
        assert!(rendered.contains("WindowExpressionWrapper"));
    }

    #[test]
    fn test_lambda_function_render() {
        let lf = LambdaFunction::new(col("x"), vec![UnresolvedNamedLambdaVariable::new("x")]);
        let expr = Expression::LambdaFunction(Box::new(lf));
        let rendered = expr.render();
        assert!(rendered.contains("LambdaFunction"));
    }

    #[test]
    fn test_unresolved_named_lambda_variable_render() {
        let var = UnresolvedNamedLambdaVariable::new("x");
        let expr = Expression::UnresolvedNamedLambdaVariable(var);
        let rendered = expr.render();
        assert!(rendered.contains("UnresolvedNamedLambdaVariable"));
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
            } else {
                panic!("Expected array literal type");
            }
        } else {
            panic!("Expected literal expression type");
        }
    }

    #[test]
    fn test_to_proto_unresolved_star_none() {
        let expr = Expression::UnresolvedStar(None);
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
        let expr = Expression::UnresolvedStar(Some("table.*".to_string()));
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
        let expr = Expression::UnresolvedRegex("`col_.*`".to_string());
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
        let expr = Expression::DirectShufflePartitionId(Box::new(child));
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
        let extraction = Expression::Literal(LiteralExpression::string("field"));
        let ev = ExtractValue::new(child, extraction);
        let expr = Expression::UnresolvedExtractValue(Box::new(ev));
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
        let value_expr = Expression::Literal(LiteralExpression::int(42));
        let uf = UpdateFieldsExpr::new(struct_expr, "f1", Some(value_expr));
        let expr = Expression::UpdateFields(Box::new(uf));
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
        let expr = Expression::Alias(Box::new(alias));
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
        let expr = Expression::Cast(Box::new(cast));
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
        let expr = Expression::Cast(Box::new(cast));
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
        let expr = Expression::Cast(Box::new(cast));
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
        let expr = Expression::Cast(Box::new(cast));
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
        let expr = Expression::Cast(Box::new(cast));
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
        let expr = Expression::SortOrder(Box::new(sort));
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
            Expression::Literal(LiteralExpression::boolean(true)),
            Expression::Literal(LiteralExpression::int(1)),
        )])
        .with_else(Expression::Literal(LiteralExpression::int(99)));
        let expr = Expression::CaseWhen(Box::new(cw));
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
        let expr = Expression::SQLExpression("SELECT * FROM table".to_string());
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
        let expr = Expression::CallFunction(Box::new(cf));
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
        let expr = Expression::LambdaFunction(Box::new(lf));
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
        let expr = Expression::UnresolvedNamedLambdaVariable(var);
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
            Expression::UnresolvedFunction(UnresolvedFunction::new("sum", vec![col("x")])),
            vec![col("group_col")],
            vec![SortOrder::asc_nulls_last(col("sort_col"))],
            Some((
                1u32,
                FrameBoundary::UnboundedPreceding,
                FrameBoundary::CurrentRow,
            )),
        );
        let expr = Expression::WindowExpression(Box::new(we));
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
