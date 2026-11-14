//! Column API mirroring PySpark's `pyspark.sql.connect.column.Column`.
//!
//! Provides the Column type and builder functions for constructing expressions.

use std::ops::{Add, BitAnd, BitOr, Div, Mul, Not, Rem, Sub};

use spark_connect_proto as proto;

use crate::expression::{
    Alias, CaseWhen, Cast, CastEvalMode, Expression, ExpressionKind, ExpressionSpec, ExtractValue,
    FrameBoundary, LiteralExpression, SortOrder, UpdateFieldsExpr, WindowExpressionWrapper,
};
use crate::types::DataType;
use crate::window::WindowSpec;

/// `pyspark.sql.connect.column.Column`
///
/// Represents a column in a DataFrame, built from an Expression. All operations
/// return new Columns by composing expressions.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    expr: Expression,
}

impl Column {
    /// Create a Column from an Expression.
    pub fn new(expr: Expression) -> Self {
        Column { expr }
    }

    /// Get the underlying Expression.
    pub fn expression(&self) -> &Expression {
        &self.expr
    }

    /// Mirrors `pyspark.sql.column.Column.alias`.
    pub fn alias(self, name: &str) -> Column {
        Column {
            expr: ExpressionSpec::Alias(Box::new(Alias::new(self.expr, name))).into(),
        }
    }

    /// `Column.alias(name, metadata=...)` - attaches column metadata, serialized to
    /// a JSON map (matching the reference client's `json.dumps(metadata)`).
    pub fn alias_with_metadata(
        self,
        name: &str,
        metadata: std::collections::BTreeMap<String, String>,
    ) -> Column {
        let json = serde_json::to_string(&metadata).unwrap_or_else(|_| "{}".to_string());
        Column {
            expr: ExpressionSpec::Alias(Box::new(Alias::new(self.expr, name).with_metadata(json)))
                .into(),
        }
    }

    /// `Column.alias(*alias)` of several names, such as the key and the value of a map's
    /// `explode`. Mirrors Scala's `Column.as(aliases: Seq[String])`.
    pub fn aliases<S: Into<String>>(self, names: impl IntoIterator<Item = S>) -> Column {
        Column {
            expr: ExpressionSpec::Alias(Box::new(Alias::with_names(self.expr, names))).into(),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.name` (alias for `alias`).
    pub fn name(self, name: &str) -> Column {
        self.alias(name)
    }

    /// Mirrors `pyspark.sql.column.Column.cast`.
    pub fn cast(self, to_type: DataType) -> Column {
        Column {
            expr: ExpressionSpec::Cast(Box::new(Cast::new(self.expr, to_type))).into(),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.astype` (alias for `cast`).
    pub fn astype(self, to_type: DataType) -> Column {
        self.cast(to_type)
    }

    /// Mirrors `pyspark.sql.column.Column.cast` with a DDL type string, e.g.
    /// `col("x").cast("string")` → `cast { type_str: "string" }`.
    pub fn cast_str(self, type_name: &str) -> Column {
        Column {
            expr: ExpressionSpec::Cast(Box::new(Cast::new_str(self.expr, type_name))).into(),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.try_cast`.
    pub fn try_cast(self, to_type: DataType) -> Column {
        Column {
            expr: ExpressionSpec::Cast(Box::new(
                Cast::new(self.expr, to_type).with_eval_mode(CastEvalMode::Try),
            ))
            .into(),
        }
    }

    /// `Column.try_cast` with a DDL type string.
    pub fn try_cast_str(self, type_name: &str) -> Column {
        Column {
            expr: ExpressionSpec::Cast(Box::new(
                Cast::new_str(self.expr, type_name).with_eval_mode(CastEvalMode::Try),
            ))
            .into(),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.isNull`.
    pub fn is_null(self) -> Column {
        Column {
            expr: Expression::function_with_inputs("isNull", [self.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.isNotNull`.
    pub fn is_not_null(self) -> Column {
        Column {
            expr: Expression::function_with_inputs("isNotNull", [self.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.substr`.
    pub fn substr(self, start: Column, length: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("substr", [self.expr, start.expr, length.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.like`.
    pub fn like(self, pattern: &str) -> Column {
        Column {
            expr: Expression::function_with_inputs(
                "like",
                [
                    self.expr,
                    Expression::literal(LiteralExpression::string(pattern)),
                ],
            ),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.rlike`.
    pub fn rlike(self, pattern: &str) -> Column {
        Column {
            expr: Expression::function_with_inputs(
                "rlike",
                [
                    self.expr,
                    Expression::literal(LiteralExpression::string(pattern)),
                ],
            ),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.contains`.
    pub fn contains(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("contains", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.ilike` (case-insensitive `like`).
    pub fn ilike(self, pattern: &str) -> Column {
        Column {
            expr: Expression::function_with_inputs(
                "ilike",
                [
                    self.expr,
                    Expression::literal(LiteralExpression::string(pattern)),
                ],
            ),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.isNaN`.
    pub fn is_nan(self) -> Column {
        Column {
            expr: Expression::function_with_inputs("isNaN", [self.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.eqNullSafe` (null-safe equality `<=>`).
    pub fn eq_null_safe(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("<=>", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.bitwiseAND`.
    pub fn bitwise_and(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("&", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.bitwiseOR`.
    pub fn bitwise_or(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("|", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.bitwiseXOR`.
    pub fn bitwise_xor(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("^", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.between`:
    /// `(self >= lower) & (self <= upper)`.
    pub fn between(self, lower: Column, upper: Column) -> Column {
        let lo = Expression::function_with_inputs(">=", [self.expr.clone(), lower.expr]);
        let hi = Expression::function_with_inputs("<=", [self.expr, upper.expr]);
        Column {
            expr: Expression::function_with_inputs("and", [lo, hi]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.isin`: membership test against the
    /// given values (pass literals via `lit(..)` or other columns).
    pub fn isin<C: Into<Column>>(self, values: impl IntoIterator<Item = C>) -> Column {
        Column {
            expr: Expression::function_with_inputs(
                "in",
                std::iter::once(self.expr).chain(
                    values
                        .into_iter()
                        .map(|value| Into::<Column>::into(value).expr),
                ),
            ),
        }
    }

    /// The IN subquery of whether this column is in the rows of `df`, of one column; of a
    /// `struct(...)` of columns, whether they are in the rows of `df` of as many columns.
    /// Mirrors `Column.isin(df)`.
    pub fn isin_subquery(self, df: &crate::dataframe::DataFrame) -> Column {
        let values = match self.expr.node().kind() {
            ExpressionKind::UnresolvedFunction { name, .. } if name == "struct" => {
                self.expr.node().children().to_vec()
            }
            _ => vec![self.expr],
        };
        let mut subquery = df.subquery(crate::expression::SubqueryType::In);
        subquery.in_subquery_values = values;
        Column {
            expr: ExpressionSpec::Subquery(Box::new(subquery)).into(),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.withField`: add or replace a field in
    /// a struct-typed column.
    pub fn with_field(self, field_name: &str, value: Column) -> Column {
        Column {
            expr: ExpressionSpec::UpdateFields(Box::new(UpdateFieldsExpr::new(
                self.expr,
                field_name,
                Some(value.expr),
            )))
            .into(),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.dropFields`: drop one or more fields
    /// from a struct-typed column (chained `UpdateFields`, one per field).
    pub fn drop_fields<S: Into<String>>(self, field_names: impl IntoIterator<Item = S>) -> Column {
        let mut expr = self.expr;
        for name in field_names {
            expr = ExpressionSpec::UpdateFields(Box::new(UpdateFieldsExpr::new(expr, name, None)))
                .into();
        }
        Column { expr }
    }

    /// Mirrors `pyspark.sql.column.Column.startswith`.
    pub fn startswith(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("startsWith", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.endswith`.
    pub fn endswith(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("endsWith", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.asc`.
    pub fn asc(self) -> Column {
        self.asc_nulls_first()
    }

    /// Mirrors `pyspark.sql.column.Column.asc_nulls_first`.
    pub fn asc_nulls_first(self) -> Column {
        Column {
            expr: ExpressionSpec::SortOrder(Box::new(SortOrder::asc_nulls_first(self.expr))).into(),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.asc_nulls_last`.
    pub fn asc_nulls_last(self) -> Column {
        Column {
            expr: ExpressionSpec::SortOrder(Box::new(SortOrder::asc_nulls_last(self.expr))).into(),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.desc`.
    pub fn desc(self) -> Column {
        self.desc_nulls_last()
    }

    /// Mirrors `pyspark.sql.column.Column.desc_nulls_first`.
    pub fn desc_nulls_first(self) -> Column {
        Column {
            expr: ExpressionSpec::SortOrder(Box::new(SortOrder::desc_nulls_first(self.expr)))
                .into(),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.desc_nulls_last`.
    pub fn desc_nulls_last(self) -> Column {
        Column {
            expr: ExpressionSpec::SortOrder(Box::new(SortOrder::desc_nulls_last(self.expr))).into(),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.when`.
    pub fn when(self, condition: Column, value: Column) -> Column {
        if matches!(self.expr.node().kind(), ExpressionKind::CaseWhen { .. }) {
            let ExpressionSpec::CaseWhen(mut case_when) = self.expr.to_spec() else {
                unreachable!("CaseWhen kind reconstructs a CaseWhen specification");
            };
            case_when.branches.push((condition.expr, value.expr));
            Column {
                expr: ExpressionSpec::CaseWhen(case_when).into(),
            }
        } else {
            Column {
                expr: ExpressionSpec::CaseWhen(Box::new(CaseWhen {
                    branches: vec![(condition.expr, value.expr)],
                    else_expr: None,
                }))
                .into(),
            }
        }
    }

    /// Mirrors `pyspark.sql.column.Column.otherwise`.
    pub fn otherwise(self, value: Column) -> Column {
        if matches!(self.expr.node().kind(), ExpressionKind::CaseWhen { .. }) {
            let ExpressionSpec::CaseWhen(mut case_when) = self.expr.to_spec() else {
                unreachable!("CaseWhen kind reconstructs a CaseWhen specification");
            };
            case_when.else_expr = Some(value.expr);
            Column {
                expr: ExpressionSpec::CaseWhen(case_when).into(),
            }
        } else {
            Column { expr: self.expr }
        }
    }

    /// Mirrors `pyspark.sql.column.Column.getField` - struct field access.
    /// Builds an `UnresolvedExtractValue` with the field name as a string literal.
    pub fn get_field(self, name: &str) -> Column {
        let extraction = Expression::literal(LiteralExpression::string(name));
        Column {
            expr: ExpressionSpec::UnresolvedExtractValue(Box::new(ExtractValue::new(
                self.expr, extraction,
            )))
            .into(),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__getitem__` / `getItem` - map/array/struct
    /// extraction. Builds an `UnresolvedExtractValue`.
    pub fn get_item(self, key: Column) -> Column {
        Column {
            expr: ExpressionSpec::UnresolvedExtractValue(Box::new(ExtractValue::new(
                self.expr, key.expr,
            )))
            .into(),
        }
    }

    /// Convert to proto expression.
    pub fn to_proto(&self) -> proto::Expression {
        self.expr.to_proto()
    }

    /// Convert to proto expression as a request serializes it: an expression deeper than
    /// [`crate::expression::MAX_EXPRESSION_DEPTH`] is a `RecursionError`, as PySpark's
    /// serializer raises it (see [`Expression::try_to_proto`]).
    pub fn try_to_proto(&self) -> spark_connect_core::error::Result<proto::Expression> {
        self.expr.try_to_proto()
    }

    /// As [`Column::try_to_proto`], and the levels of the expression, which a pickled Column keeps
    /// for [`Column::sent`]. Not a stable API.
    #[doc(hidden)]
    pub fn try_to_proto_with_levels(
        &self,
    ) -> spark_connect_core::error::Result<(proto::Expression, usize)> {
        self.expr.try_to_proto_with_levels()
    }

    /// The Column of the `spark.connect.Expression` `expression`, as it was sent, with its
    /// `rendered` rendering and its `levels` (see [`Column::try_to_proto_with_levels`]), as a
    /// pickled Column unpickles. Not a stable API.
    #[doc(hidden)]
    pub fn sent(
        expression: proto::Expression,
        rendered: String,
        levels: usize,
    ) -> spark_connect_core::error::Result<Column> {
        Ok(Column::new(Expression::sent(expression, rendered, levels)?))
    }

    /// Mirrors `pyspark.sql.column.Column.over` - window function application.
    pub fn over(self, window_spec: WindowSpec) -> Column {
        // Convert window frame spec if present
        let frame_spec = window_spec.frame_spec.map(|(frame_type, lower, upper)| {
            let frame_type_val = match frame_type {
                crate::window::FrameType::Row => 1u32,
                crate::window::FrameType::Range => 2u32,
            };
            let lower_boundary = match lower {
                crate::window::FrameBound::UnboundedPreceding => FrameBoundary::UnboundedPreceding,
                crate::window::FrameBound::Preceding(n) => FrameBoundary::Preceding(n),
                crate::window::FrameBound::CurrentRow => FrameBoundary::CurrentRow,
                crate::window::FrameBound::Following(n) => FrameBoundary::Following(n),
                crate::window::FrameBound::UnboundedFollowing => FrameBoundary::UnboundedFollowing,
            };
            let upper_boundary = match upper {
                crate::window::FrameBound::UnboundedPreceding => FrameBoundary::UnboundedPreceding,
                crate::window::FrameBound::Preceding(n) => FrameBoundary::Preceding(n),
                crate::window::FrameBound::CurrentRow => FrameBoundary::CurrentRow,
                crate::window::FrameBound::Following(n) => FrameBoundary::Following(n),
                crate::window::FrameBound::UnboundedFollowing => FrameBoundary::UnboundedFollowing,
            };
            (frame_type_val, lower_boundary, upper_boundary)
        });

        let window_expr = WindowExpressionWrapper::new(
            self.expr,
            window_spec.partition_spec,
            window_spec.order_spec,
            frame_spec,
        );

        Column {
            expr: ExpressionSpec::WindowExpression(Box::new(window_expr)).into(),
        }
    }

    // Comparison operators

    /// Mirrors `pyspark.sql.column.Column.__eq__`.
    pub fn eq(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("==", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__ne__`.
    pub fn ne(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs(
                "not",
                [Expression::function_with_inputs(
                    "==",
                    [self.expr, other.expr],
                )],
            ),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__gt__`.
    pub fn gt(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs(">", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__lt__`.
    pub fn lt(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("<", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__ge__`.
    pub fn ge(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs(">=", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__le__`.
    pub fn le(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("<=", [self.expr, other.expr]),
        }
    }

    // Arithmetic operators

    /// Mirrors `pyspark.sql.column.Column.__add__`.
    pub fn add(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("+", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__sub__`.
    pub fn sub(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("-", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__mul__`.
    pub fn mul(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("*", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__truediv__`.
    pub fn div(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("/", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__mod__`.
    pub fn modulo(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("%", [self.expr, other.expr]),
        }
    }

    // Logical operators

    /// Mirrors `pyspark.sql.column.Column.__and__`.
    pub fn and(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("and", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__or__`.
    pub fn or(self, other: Column) -> Column {
        Column {
            expr: Expression::function_with_inputs("or", [self.expr, other.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__invert__`.
    pub fn not(self) -> Column {
        Column {
            expr: Expression::function_with_inputs("not", [self.expr]),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__neg__`.
    pub fn neg(self) -> Column {
        Column {
            expr: Expression::function_with_inputs("negative", [self.expr]),
        }
    }
}

// Operator trait implementations
impl Add for Column {
    type Output = Column;
    fn add(self, other: Column) -> Column {
        self.add(other)
    }
}

impl Sub for Column {
    type Output = Column;
    fn sub(self, other: Column) -> Column {
        self.sub(other)
    }
}

impl Mul for Column {
    type Output = Column;
    fn mul(self, other: Column) -> Column {
        self.mul(other)
    }
}

impl Div for Column {
    type Output = Column;
    fn div(self, other: Column) -> Column {
        self.div(other)
    }
}

impl Rem for Column {
    type Output = Column;
    fn rem(self, other: Column) -> Column {
        self.modulo(other)
    }
}

impl BitAnd for Column {
    type Output = Column;
    fn bitand(self, other: Column) -> Column {
        self.and(other)
    }
}

impl BitOr for Column {
    type Output = Column;
    fn bitor(self, other: Column) -> Column {
        self.or(other)
    }
}

impl Not for Column {
    type Output = Column;
    fn not(self) -> Column {
        Column {
            expr: Expression::function_with_inputs("not", [self.expr]),
        }
    }
}

// Builder functions

/// Mirrors `pyspark.sql.functions.col`.
pub fn col(name: &str) -> Column {
    Column {
        expr: Expression::column(name),
    }
}

// A string is accepted anywhere a `Column` is (resolved as a column reference), so the
// ergonomic `impl IntoIterator<Item = impl Into<Column>>` APIs accept `["a", "b"]` too.
impl From<&str> for Column {
    fn from(name: &str) -> Column {
        col(name)
    }
}
impl From<String> for Column {
    fn from(name: String) -> Column {
        col(&name)
    }
}
impl From<&String> for Column {
    fn from(name: &String) -> Column {
        col(name)
    }
}

// An expression is accepted anywhere a `Column` is, as the Column of it, so the APIs of
// `impl IntoIterator<Item = impl Into<Column>>` take the `Vec<Expression>` that some of them
// took before, such as `df.sort(vec![col("x").expression().clone()])`.
impl From<Expression> for Column {
    fn from(expr: Expression) -> Column {
        Column::new(expr)
    }
}

// A sort order is accepted as the Column of it that `asc` and `desc` make, so
// `WindowSpec::order_by` takes the `Vec<SortOrder>` that it took before.
impl From<SortOrder> for Column {
    fn from(sort: SortOrder) -> Column {
        Column {
            expr: ExpressionSpec::SortOrder(Box::new(sort)).into(),
        }
    }
}

/// The expressions of `columns`, as the plans of the APIs that take Columns hold them.
pub(crate) fn expressions_of<C: Into<Column>>(
    columns: impl IntoIterator<Item = C>,
) -> Vec<Expression> {
    columns
        .into_iter()
        .map(|column| column.into().expr)
        .collect()
}

/// The Columns of `columns`, for the APIs that keep a list of them.
pub(crate) fn columns_of<C: Into<Column>>(columns: impl IntoIterator<Item = C>) -> Vec<Column> {
    columns.into_iter().map(Into::into).collect()
}

/// Mirrors `pyspark.sql.functions.lit` for an integer value.
///
/// PySpark infers `IntegerType` for a Python int that fits in i32 and `LongType`
/// otherwise (see `LiteralExpression._infer_type`), so `lit(1)` is `Integer(1)`,
/// not `Long(1)`.
pub fn lit(value: i64) -> Column {
    let lit = if i32::try_from(value).is_ok() {
        LiteralExpression::int(value as i32)
    } else {
        LiteralExpression::long(value)
    };
    Column {
        expr: Expression::literal(lit),
    }
}

/// Create a literal from a string.
pub fn lit_string(value: &str) -> Column {
    Column {
        expr: Expression::literal(LiteralExpression::string(value)),
    }
}

/// Create a literal from a double.
pub fn lit_double(value: f64) -> Column {
    Column {
        expr: Expression::literal(LiteralExpression::double(value)),
    }
}

/// Create a literal from a boolean.
pub fn lit_boolean(value: bool) -> Column {
    Column {
        expr: Expression::literal(LiteralExpression::boolean(value)),
    }
}

/// Mirrors `pyspark.sql.functions.when` - starts a CASE WHEN chain. Chain more
/// branches with `Column::when` and finish with `Column::otherwise`.
pub fn when(condition: Column, value: Column) -> Column {
    Column {
        expr: ExpressionSpec::CaseWhen(Box::new(CaseWhen::new(vec![(condition.expr, value.expr)])))
            .into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isin_preserves_order_shared_inputs_and_wire_for_columns_and_expressions() {
        let left = col("a");
        let values = [col("b"), col("c"), col("d")];
        let inputs: Vec<_> = std::iter::once(left.expression().clone())
            .chain(values.iter().map(|column| column.expression().clone()))
            .collect();
        let expected = Expression::new(ExpressionSpec::UnresolvedFunction(
            crate::expression::UnresolvedFunction::new("in", inputs.clone()),
        ));
        for actual in [
            left.clone().isin(values.clone()),
            left.clone().isin(values.map(|column| column.expr)),
        ] {
            let ExpressionKind::UnresolvedFunction { name, is_distinct } = actual.expr.kind()
            else {
                panic!("expected an unresolved IN function");
            };
            assert_eq!(name, "in");
            assert!(!is_distinct);
            assert_eq!(actual.expr.children().len(), inputs.len());
            for (child, input) in actual.expr.children().iter().zip(&inputs) {
                assert!(child.ptr_eq(input));
            }
            assert_eq!(actual.to_proto(), expected.to_proto());
        }

        let empty = left.clone().isin(Vec::<Column>::new());
        assert_eq!(empty.expr.children().len(), 1);
        assert!(empty.expr.children()[0].ptr_eq(left.expression()));
        let expected = Expression::new(ExpressionSpec::UnresolvedFunction(
            crate::expression::UnresolvedFunction::new("in", vec![left.expr]),
        ));
        assert_eq!(empty.to_proto(), expected.to_proto());
    }

    #[test]
    fn aliases_name_every_column_of_a_generator() {
        let column = crate::functions::explode(col("m")).aliases(["k", "v"]);
        assert_eq!(column.expression().render(), "explode(m) AS k,v");
        match column.to_proto().expr_type {
            Some(spark_connect_proto::expression::ExprType::Alias(alias)) => {
                assert_eq!(alias.name, vec!["k".to_string(), "v".to_string()]);
                assert_eq!(alias.metadata, None);
            }
            other => panic!("expected an alias, got {other:?}"),
        }
    }

    #[test]
    fn test_col_creation() {
        let c = col("x");
        assert!(matches!(
            c.expr.node().kind(),
            ExpressionKind::ColumnReference(_)
        ));
    }

    #[test]
    fn test_lit_creation() {
        let c = lit(42);
        assert!(matches!(c.expr.node().kind(), ExpressionKind::Literal(_)));
    }

    #[test]
    fn test_addition() {
        let c1 = col("a");
        let c2 = lit(1);
        let result = c1.add(c2);
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));
    }

    #[test]
    fn test_alias() {
        let c = col("x");
        let aliased = c.alias("y");
        assert!(matches!(
            aliased.expr.node().kind(),
            ExpressionKind::Alias { .. }
        ));
    }

    #[test]
    fn test_cast() {
        let c = col("x");
        let casted = c.cast(DataType::String {
            collation: "UTF8_BINARY".to_string(),
        });
        assert!(matches!(
            casted.expr.node().kind(),
            ExpressionKind::Cast { .. }
        ));
    }

    #[test]
    fn test_comparison_operators() {
        let c1 = col("a");
        let c2 = col("b");

        // Test eq
        let result = c1.clone().eq(c2.clone());
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test ne
        let result = col("a").ne(col("b"));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test gt
        let result = col("a").gt(col("b"));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test lt
        let result = col("a").lt(col("b"));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test ge
        let result = col("a").ge(col("b"));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test le
        let result = col("a").le(col("b"));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));
    }

    #[test]
    fn test_arithmetic_operators() {
        // Test sub
        let result = col("a").sub(col("b"));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test mul
        let result = col("a").mul(col("b"));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test div
        let result = col("a").div(col("b"));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test modulo
        let result = col("a").modulo(col("b"));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));
    }

    #[test]
    fn test_logical_operators() {
        // Test and
        let result = col("a").and(col("b"));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test or
        let result = col("a").or(col("b"));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test not
        let result = col("a").not();
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test neg
        let result = col("a").neg();
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));
    }

    #[test]
    fn test_operator_traits() {
        let c1 = col("a");
        let c2 = col("b");

        // Test Add trait
        let result = c1.clone() + c2.clone();
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test Sub trait
        let result = col("a") - col("b");
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test Mul trait
        let result = col("a") * col("b");
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test Div trait
        let result = col("a") / col("b");
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test Rem trait
        let result = col("a") % col("b");
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test BitAnd trait
        let result = col("a") & col("b");
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test BitOr trait
        let result = col("a") | col("b");
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test Not trait
        let result = !col("a");
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));
    }

    #[test]
    fn test_from_implementations() {
        // Test From<&str>
        let c1: Column = "x".into();
        assert!(matches!(
            c1.expr.node().kind(),
            ExpressionKind::ColumnReference(_)
        ));

        // Test From<String>
        let c2: Column = "y".to_string().into();
        assert!(matches!(
            c2.expr.node().kind(),
            ExpressionKind::ColumnReference(_)
        ));

        // Test From<&String>
        let s = "z".to_string();
        let c3: Column = (&s).into();
        assert!(matches!(
            c3.expr.node().kind(),
            ExpressionKind::ColumnReference(_)
        ));
    }

    #[test]
    fn expressions_and_sort_orders_are_the_columns_of_them() {
        // An expression is the Column of it, which shares its node.
        let expression = (col("x") + lit(1)).expression().clone();
        let column = Column::from(expression.clone());
        assert_eq!(column, Column::new(expression.clone()));
        assert!(std::ptr::eq(column.expression().node(), expression.node()));
        // A sort order is the Column of it that `asc` and `desc` make.
        let sort = SortOrder::desc_nulls_first(col("x").expression().clone());
        assert_eq!(Column::from(sort), col("x").desc_nulls_first());
        // The expressions that the APIs of columns take, of Columns, names and expressions
        // alike, in their order.
        assert_eq!(
            expressions_of([col("a"), col("b").desc()]),
            vec![
                col("a").expression().clone(),
                col("b").desc().expression().clone()
            ]
        );
        assert_eq!(
            expressions_of(["a", "b"]),
            expressions_of(vec![Expression::column("a"), Expression::column("b")])
        );
        assert!(expressions_of(Vec::<Column>::new()).is_empty());
    }

    #[test]
    fn drop_fields_takes_any_names_as_a_vec_does() {
        let of_vec = col("s").drop_fields(vec!["a", "b"]);
        assert_eq!(col("s").drop_fields(["a", "b"]), of_vec);
        assert_eq!(col("s").drop_fields(["a", "b"].iter().copied()), of_vec);
        assert_eq!(col("s").drop_fields(["a", "b"].map(String::from)), of_vec);
    }

    #[test]
    fn test_lit_functions() {
        // Test lit with small number (fits in i32)
        let c = lit(42);
        assert!(matches!(
            c.expr.node().kind(),
            ExpressionKind::Literal(LiteralExpression::Integer(_))
        ));

        // Test lit with large number (needs i64)
        let c = lit(5_000_000_000i64);
        assert!(matches!(
            c.expr.node().kind(),
            ExpressionKind::Literal(LiteralExpression::Long(_))
        ));

        // Test lit_string
        let c = lit_string("hello");
        assert!(matches!(
            c.expr.node().kind(),
            ExpressionKind::Literal(LiteralExpression::String(_))
        ));

        // Test lit_double
        let c = lit_double(3.14);
        assert!(matches!(
            c.expr.node().kind(),
            ExpressionKind::Literal(LiteralExpression::Double(_))
        ));

        // Test lit_boolean
        let c = lit_boolean(true);
        assert!(matches!(
            c.expr.node().kind(),
            ExpressionKind::Literal(LiteralExpression::Boolean(_))
        ));
    }

    #[test]
    fn test_when_otherwise() {
        // Test when
        let cond = col("x").gt(lit(5));
        let result = when(cond, lit(1));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::CaseWhen { .. }
        ));

        // Test when with otherwise
        let cond2 = col("y").lt(lit(10));
        let result2 = result.when(cond2, lit(2));
        assert!(matches!(
            result2.expr.node().kind(),
            ExpressionKind::CaseWhen { .. }
        ));

        // Test otherwise
        let final_result = result2.otherwise(lit(99));
        assert!(matches!(
            final_result.expr.node().kind(),
            ExpressionKind::CaseWhen { .. }
        ));
    }

    #[test]
    fn test_is_null_and_is_not_null() {
        // Test is_null
        let result = col("a").is_null();
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test is_not_null
        let result = col("a").is_not_null();
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));
    }

    #[test]
    fn test_getitem() {
        // Test get_item
        let result = col("array").get_item(lit(0));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedExtractValue
        ));
    }

    #[test]
    fn test_between() {
        let result = col("a").between(lit(1), lit(10));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));
    }

    #[test]
    fn test_substring() {
        let result = col("a").substr(lit(1), lit(3));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));
    }

    #[test]
    fn test_various_methods() {
        // Test cast_str
        let result = col("a").cast_str("int");
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::Cast { .. }
        ));

        // Test desc
        let result = col("a").desc();
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::SortOrder { .. }
        ));

        // Test desc_nulls_first
        let result = col("a").desc_nulls_first();
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::SortOrder { .. }
        ));

        // Test desc_nulls_last
        let result = col("a").desc_nulls_last();
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::SortOrder { .. }
        ));

        // Test asc
        let result = col("a").asc();
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::SortOrder { .. }
        ));

        // Test asc_nulls_first
        let result = col("a").asc_nulls_first();
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::SortOrder { .. }
        ));

        // Test asc_nulls_last
        let result = col("a").asc_nulls_last();
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::SortOrder { .. }
        ));
    }

    #[test]
    fn test_get_field() {
        // Test get_field
        let result = col("struct").get_field("field_name");
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedExtractValue
        ));
    }

    #[test]
    fn test_string_functions() {
        // Test contains
        let result = col("a").contains(col("b"));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test startswith
        let result = col("a").startswith(col("b"));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test endswith
        let result = col("a").endswith(col("b"));
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test like
        let result = col("a").like("%pattern%");
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));

        // Test rlike
        let result = col("a").rlike("[0-9]+");
        assert!(matches!(
            result.expr.node().kind(),
            ExpressionKind::UnresolvedFunction { .. }
        ));
    }
}
