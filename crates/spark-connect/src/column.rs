//! Column API mirroring PySpark's `pyspark.sql.connect.column.Column`.
//!
//! Provides the Column type and builder functions for constructing expressions.

use std::ops::{Add, BitAnd, BitOr, Div, Mul, Not, Rem, Sub};

use spark_connect_proto as proto;

use crate::expression::{
    Alias, CaseWhen, Cast, CastEvalMode, ColumnReference, Expression, ExtractValue, FrameBoundary,
    LiteralExpression, SortOrder, UnresolvedFunction, WindowExpressionWrapper,
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
            expr: Expression::Alias(Box::new(Alias::new(self.expr, name))),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.name` (alias for `alias`).
    pub fn name(self, name: &str) -> Column {
        self.alias(name)
    }

    /// Mirrors `pyspark.sql.column.Column.cast`.
    pub fn cast(self, to_type: DataType) -> Column {
        Column {
            expr: Expression::Cast(Box::new(Cast::new(self.expr, to_type))),
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
            expr: Expression::Cast(Box::new(Cast::new_str(self.expr, type_name))),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.try_cast`.
    pub fn try_cast(self, to_type: DataType) -> Column {
        Column {
            expr: Expression::Cast(Box::new(
                Cast::new(self.expr, to_type).with_eval_mode(CastEvalMode::Try),
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.isNull`.
    pub fn is_null(self) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "isNull",
                vec![self.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.isNotNull`.
    pub fn is_not_null(self) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "isNotNull",
                vec![self.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.substr`.
    pub fn substr(self, start: Column, length: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "substr",
                vec![self.expr, start.expr, length.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.like`.
    pub fn like(self, pattern: &str) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "like",
                vec![
                    self.expr,
                    Expression::Literal(LiteralExpression::string(pattern)),
                ],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.rlike`.
    pub fn rlike(self, pattern: &str) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "rlike",
                vec![
                    self.expr,
                    Expression::Literal(LiteralExpression::string(pattern)),
                ],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.contains`.
    pub fn contains(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "contains",
                vec![self.expr, other.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.startswith`.
    pub fn startswith(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "startsWith",
                vec![self.expr, other.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.endswith`.
    pub fn endswith(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "endsWith",
                vec![self.expr, other.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.asc`.
    pub fn asc(self) -> Column {
        self.asc_nulls_first()
    }

    /// Mirrors `pyspark.sql.column.Column.asc_nulls_first`.
    pub fn asc_nulls_first(self) -> Column {
        Column {
            expr: Expression::SortOrder(Box::new(SortOrder::asc_nulls_first(self.expr))),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.asc_nulls_last`.
    pub fn asc_nulls_last(self) -> Column {
        Column {
            expr: Expression::SortOrder(Box::new(SortOrder::asc_nulls_last(self.expr))),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.desc`.
    pub fn desc(self) -> Column {
        self.desc_nulls_last()
    }

    /// Mirrors `pyspark.sql.column.Column.desc_nulls_first`.
    pub fn desc_nulls_first(self) -> Column {
        Column {
            expr: Expression::SortOrder(Box::new(SortOrder::desc_nulls_first(self.expr))),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.desc_nulls_last`.
    pub fn desc_nulls_last(self) -> Column {
        Column {
            expr: Expression::SortOrder(Box::new(SortOrder::desc_nulls_last(self.expr))),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.when`.
    pub fn when(self, condition: Column, value: Column) -> Column {
        // If we already have a CaseWhen, add a branch
        if let Expression::CaseWhen(case_when) = self.expr {
            let mut branches = case_when.branches.clone();
            branches.push((condition.expr, value.expr));
            Column {
                expr: Expression::CaseWhen(Box::new(CaseWhen {
                    branches,
                    else_expr: case_when.else_expr.clone(),
                })),
            }
        } else {
            // Start a new CaseWhen
            Column {
                expr: Expression::CaseWhen(Box::new(CaseWhen {
                    branches: vec![(condition.expr, value.expr)],
                    else_expr: None,
                })),
            }
        }
    }

    /// Mirrors `pyspark.sql.column.Column.otherwise`.
    pub fn otherwise(self, value: Column) -> Column {
        if let Expression::CaseWhen(case_when) = self.expr {
            Column {
                expr: Expression::CaseWhen(Box::new(CaseWhen {
                    branches: case_when.branches.clone(),
                    else_expr: Some(Box::new(value.expr)),
                })),
            }
        } else {
            Column { expr: self.expr }
        }
    }

    /// Mirrors `pyspark.sql.column.Column.getField` - struct field access.
    /// Builds an `UnresolvedExtractValue` with the field name as a string literal.
    pub fn get_field(self, name: &str) -> Column {
        let extraction = Expression::Literal(LiteralExpression::string(name));
        Column {
            expr: Expression::UnresolvedExtractValue(Box::new(ExtractValue::new(
                self.expr, extraction,
            ))),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__getitem__` / `getItem` - map/array/struct
    /// extraction. Builds an `UnresolvedExtractValue`.
    pub fn get_item(self, key: Column) -> Column {
        Column {
            expr: Expression::UnresolvedExtractValue(Box::new(ExtractValue::new(
                self.expr, key.expr,
            ))),
        }
    }

    /// Convert to proto expression.
    pub fn to_proto(&self) -> proto::Expression {
        self.expr.to_proto()
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
            expr: Expression::WindowExpression(Box::new(window_expr)),
        }
    }

    // Comparison operators

    /// Mirrors `pyspark.sql.column.Column.__eq__`.
    pub fn eq(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "==",
                vec![self.expr, other.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__ne__`.
    pub fn ne(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "not",
                vec![Expression::UnresolvedFunction(UnresolvedFunction::new(
                    "==",
                    vec![self.expr, other.expr],
                ))],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__gt__`.
    pub fn gt(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                ">",
                vec![self.expr, other.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__lt__`.
    pub fn lt(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "<",
                vec![self.expr, other.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__ge__`.
    pub fn ge(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                ">=",
                vec![self.expr, other.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__le__`.
    pub fn le(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "<=",
                vec![self.expr, other.expr],
            )),
        }
    }

    // Arithmetic operators

    /// Mirrors `pyspark.sql.column.Column.__add__`.
    pub fn add(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "+",
                vec![self.expr, other.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__sub__`.
    pub fn sub(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "-",
                vec![self.expr, other.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__mul__`.
    pub fn mul(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "*",
                vec![self.expr, other.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__truediv__`.
    pub fn div(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "/",
                vec![self.expr, other.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__mod__`.
    pub fn modulo(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "%",
                vec![self.expr, other.expr],
            )),
        }
    }

    // Logical operators

    /// Mirrors `pyspark.sql.column.Column.__and__`.
    pub fn and(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "and",
                vec![self.expr, other.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__or__`.
    pub fn or(self, other: Column) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "or",
                vec![self.expr, other.expr],
            )),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__invert__`.
    pub fn not(self) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new("not", vec![self.expr])),
        }
    }

    /// Mirrors `pyspark.sql.column.Column.__neg__`.
    pub fn neg(self) -> Column {
        Column {
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new(
                "negative",
                vec![self.expr],
            )),
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
            expr: Expression::UnresolvedFunction(UnresolvedFunction::new("not", vec![self.expr])),
        }
    }
}

// Builder functions

/// Mirrors `pyspark.sql.functions.col`.
pub fn col(name: &str) -> Column {
    Column {
        expr: Expression::ColumnReference(ColumnReference::new(name)),
    }
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
        expr: Expression::Literal(lit),
    }
}

/// Create a literal from a string.
pub fn lit_string(value: &str) -> Column {
    Column {
        expr: Expression::Literal(LiteralExpression::string(value)),
    }
}

/// Create a literal from a double.
pub fn lit_double(value: f64) -> Column {
    Column {
        expr: Expression::Literal(LiteralExpression::double(value)),
    }
}

/// Create a literal from a boolean.
pub fn lit_boolean(value: bool) -> Column {
    Column {
        expr: Expression::Literal(LiteralExpression::boolean(value)),
    }
}

/// Mirrors `pyspark.sql.functions.when` - starts a CASE WHEN chain. Chain more
/// branches with `Column::when` and finish with `Column::otherwise`.
pub fn when(condition: Column, value: Column) -> Column {
    Column {
        expr: Expression::CaseWhen(Box::new(CaseWhen::new(vec![(condition.expr, value.expr)]))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_col_creation() {
        let c = col("x");
        assert!(matches!(c.expr, Expression::ColumnReference(_)));
    }

    #[test]
    fn test_lit_creation() {
        let c = lit(42);
        assert!(matches!(c.expr, Expression::Literal(_)));
    }

    #[test]
    fn test_addition() {
        let c1 = col("a");
        let c2 = lit(1);
        let result = c1.add(c2);
        assert!(matches!(result.expr, Expression::UnresolvedFunction(_)));
    }

    #[test]
    fn test_alias() {
        let c = col("x");
        let aliased = c.alias("y");
        assert!(matches!(aliased.expr, Expression::Alias(_)));
    }

    #[test]
    fn test_cast() {
        let c = col("x");
        let casted = c.cast(DataType::String {
            collation: "UTF8_BINARY".to_string(),
        });
        assert!(matches!(casted.expr, Expression::Cast(_)));
    }
}
