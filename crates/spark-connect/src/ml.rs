//! ML module mirroring `pyspark.ml.connect.base` and related ML classes.
//!
//! This module provides the foundation for machine learning in Spark Connect:
//! - `Params`: parameter management for ML operators
//! - `MlOperator`: represents ML operators (estimators, transformers, models, evaluators)
//! - Base traits: `Estimator`, `Transformer`, `Model`, `Evaluator`
//! - Concrete example: `StandardScaler` estimator + `StandardScalerModel`

use spark_connect_core::error::{SparkError, SparkErrorKind};
use spark_connect_proto as proto;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Declares the ML params of one class: the typed builders and getters of its parameters, and
/// `PARAM_NAMES`. Usage: `ml_params! { Class { <rows> } }`, once per class, with one row per
/// parameter in the order of PySpark's `params`. A row is one of two kinds:
///
/// * A setter row, `set_x(value: T) => "wireName", put_fn; x -> Ret = |this| body;`, emits the
///   builder `set_x(value)`, which stores `put_fn("wireName", value)`, and the getter `x(&self)
///   -> Ret`, whose body `body` reads the parameter, or returns its pyspark.ml default.
/// * A getter-only row, `x -> Ret = |this| body => "wireName";`, is for a parameter with no
///   pyspark.ml setter (a model's parameter, which the server sets): it emits only the getter.
///
/// The `PARAM_NAMES` constant lists the wire name of every row, in row order. A second
/// invocation for the same class fails to compile on the duplicate constant.
///
/// This macro is defined before the area modules, so that all of them see it.
macro_rules! ml_params {
    ($name:ident { $($rows:tt)* }) => {
        ml_params! { @munch $name [] [] $($rows)* }
    };
    // A setter row and the getter of its parameter.
    (@munch $name:ident [$($items:tt)*] [$($wires:literal,)*]
        $set:ident($arg:ident: $ty:ty) => $wire:literal, $put:ident;
        $get:ident -> $ret:ty = |$this:ident| $body:expr; $($rest:tt)*
    ) => {
        ml_params! { @munch $name
            [$($items)*
                #[doc = concat!("Set `", $wire, "`.")]
                pub fn $set(mut self, $arg: $ty) -> Self {
                    self.params = self.params.$put($wire, $arg);
                    self
                }
                #[doc = "Get the parameter, or its pyspark.ml default when it is not set."]
                pub fn $get(&self) -> $ret {
                    let $this = self;
                    $body
                }
            ]
            [$($wires,)* $wire,]
            $($rest)*
        }
    };
    // A getter-only row.
    (@munch $name:ident [$($items:tt)*] [$($wires:literal,)*]
        $get:ident -> $ret:ty = |$this:ident| $body:expr => $wire:literal; $($rest:tt)*
    ) => {
        ml_params! { @munch $name
            [$($items)*
                #[doc = "Get the parameter, or its pyspark.ml default when it is not set."]
                pub fn $get(&self) -> $ret {
                    let $this = self;
                    $body
                }
            ]
            [$($wires,)* $wire,]
            $($rest)*
        }
    };
    // All rows are read: the impl block with the names of the rows, in row order.
    (@munch $name:ident [$($items:tt)*] [$($wires:literal,)*]) => {
        impl $name {
            /// The wire names of the parameters of this class, in row order.
            pub const PARAM_NAMES: &'static [&'static str] = &[$($wires),*];
            $($items)*
        }
    };
}

/// The default `seed` of a class when the client sends none: Java's `String.hashCode` of the JVM
/// class name, which the server takes as `getClass.getName.hashCode`, sign-extended to `Long`.
fn java_string_hash(class_name: &str) -> i64 {
    let mut hash: i32 = 0;
    for unit in class_name.encode_utf16() {
        hash = hash.wrapping_mul(31).wrapping_add(i32::from(unit));
    }
    i64::from(hash)
}

mod recommendation;
pub use recommendation::*;

mod stat;
pub use stat::*;

mod evaluation;
pub use evaluation::*;

mod feature;
pub use feature::*;
mod fpm;
pub use fpm::*;

mod object;
pub use object::*;
mod persistence;
pub use persistence::*;
mod tuning;
pub use tuning::*;
mod meta;
pub use meta::*;
mod validators;
pub use validators::*;
mod stage_kinds;

// The typed accessors, summaries and persistence of the models of each family, in the files of the
// areas that complete them (see `object` for their server-side objects).
mod classification_api;
#[allow(unused_imports)] // Until the area adds its first public item.
pub use classification_api::*;
mod regression_api;
#[allow(unused_imports)] // Until the area adds its first public item.
pub use regression_api::*;
mod clustering_api;
#[allow(unused_imports)] // Until the area adds its first public item.
pub use clustering_api::*;
// The members of the tree models, which the classification and regression trees share.
mod tree_api;
#[allow(unused_imports)] // Until the area adds its first public item.
pub use tree_api::*;

use crate::dataframe::DataFrame;
use crate::expression::LiteralExpression;
use crate::plan::{LogicalPlan, RelationOwner};
use crate::session::SparkSession;
use crate::types::DataType;

/// A dense or sparse ML vector used as an estimator parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum MlVector {
    Dense(Vec<f64>),
    Sparse {
        size: i32,
        indices: Vec<i32>,
        values: Vec<f64>,
    },
}

impl From<Vec<f64>> for MlVector {
    fn from(values: Vec<f64>) -> Self {
        Self::Dense(values)
    }
}

impl<const N: usize> From<[f64; N]> for MlVector {
    fn from(values: [f64; N]) -> Self {
        Self::Dense(values.into())
    }
}

impl MlVector {
    #[allow(deprecated)] // PySpark's ML serializer uses struct_type for vector UDTs.
    fn to_literal(&self) -> proto::expression::Literal {
        use proto::expression::literal::{specialized_array::ValueType as A, LiteralType as L};
        let lit = |value| proto::expression::Literal {
            literal_type: Some(value),
            ..Default::default()
        };
        let null = || lit(L::Null(DataType::Null.to_proto()));
        let doubles = |values: Vec<f64>| {
            lit(L::SpecializedArray(
                proto::expression::literal::SpecializedArray {
                    value_type: Some(A::Doubles(proto::Doubles { values })),
                },
            ))
        };
        let elements = match self {
            Self::Dense(values) => vec![lit(L::Byte(1)), null(), null(), doubles(values.clone())],
            Self::Sparse {
                size,
                indices,
                values,
            } => vec![
                lit(L::Byte(0)),
                lit(L::Integer(*size)),
                lit(L::SpecializedArray(
                    proto::expression::literal::SpecializedArray {
                        value_type: Some(A::Ints(proto::Ints {
                            values: indices.clone(),
                        })),
                    },
                )),
                doubles(values.clone()),
            ],
        };
        lit(L::Struct(proto::expression::literal::Struct {
            struct_type: Some(proto::DataType {
                kind: Some(proto::data_type::Kind::Udt(Box::new(
                    proto::data_type::Udt {
                        r#type: "udt".into(),
                        jvm_class: Some("org.apache.spark.ml.linalg.VectorUDT".into()),
                        ..Default::default()
                    },
                ))),
            }),
            elements,
        }))
    }

    fn from_literal(value: &proto::expression::literal::LiteralType) -> Option<Self> {
        use proto::expression::literal::{specialized_array::ValueType as A, LiteralType as L};
        let L::Struct(value) = value else {
            return None;
        };
        let fields = &value.elements;
        let L::SpecializedArray(array) = fields.get(3)?.literal_type.as_ref()? else {
            return None;
        };
        let A::Doubles(values) = array.value_type.as_ref()? else {
            return None;
        };
        match fields.first()?.literal_type.as_ref()? {
            L::Byte(1) => Some(Self::Dense(values.values.clone())),
            L::Byte(0) => {
                let L::Integer(size) = fields.get(1)?.literal_type.as_ref()? else {
                    return None;
                };
                let L::SpecializedArray(array) = fields.get(2)?.literal_type.as_ref()? else {
                    return None;
                };
                let A::Ints(indices) = array.value_type.as_ref()? else {
                    return None;
                };
                Some(Self::Sparse {
                    size: *size,
                    indices: indices.values.clone(),
                    values: values.values.clone(),
                })
            }
            _ => None,
        }
    }
}

/// A dense or sparse ML matrix used as an estimator parameter, such as the
/// `lowerBoundsOnCoefficients` of `LogisticRegression`. As Spark's `Matrix`, its values are in
/// column-major order, or in row-major order when it `is_transposed`; a sparse matrix is of the
/// compressed sparse column format (of rows, when it `is_transposed`).
#[derive(Debug, Clone, PartialEq)]
pub enum MlMatrix {
    Dense {
        num_rows: i32,
        num_cols: i32,
        values: Vec<f64>,
        is_transposed: bool,
    },
    Sparse {
        num_rows: i32,
        num_cols: i32,
        col_ptrs: Vec<i32>,
        row_indices: Vec<i32>,
        values: Vec<f64>,
        is_transposed: bool,
    },
}

impl MlMatrix {
    /// The literal of the matrix, as PySpark's ML serializer (`serialize_param` of
    /// `pyspark.ml.connect.serialize`) sends a `DenseMatrix` or a `SparseMatrix`: a struct of the
    /// MatrixUDT, of its type (1 dense, 0 sparse), its shape, its column pointers and row indices
    /// (nulls of a dense matrix), its values and whether it is transposed.
    #[allow(deprecated)] // PySpark's ML serializer uses struct_type for matrix UDTs.
    fn to_literal(&self) -> proto::expression::Literal {
        use proto::expression::literal::{specialized_array::ValueType as A, LiteralType as L};
        let lit = |value| proto::expression::Literal {
            literal_type: Some(value),
            ..Default::default()
        };
        let null = || lit(L::Null(DataType::Null.to_proto()));
        let array = |value| {
            lit(L::SpecializedArray(
                proto::expression::literal::SpecializedArray {
                    value_type: Some(value),
                },
            ))
        };
        let ints = |values: &[i32]| {
            array(A::Ints(proto::Ints {
                values: values.to_vec(),
            }))
        };
        let doubles = |values: &[f64]| {
            array(A::Doubles(proto::Doubles {
                values: values.to_vec(),
            }))
        };
        let elements = match self {
            Self::Dense {
                num_rows,
                num_cols,
                values,
                is_transposed,
            } => vec![
                lit(L::Byte(1)),
                lit(L::Integer(*num_rows)),
                lit(L::Integer(*num_cols)),
                null(),
                null(),
                doubles(values),
                lit(L::Boolean(*is_transposed)),
            ],
            Self::Sparse {
                num_rows,
                num_cols,
                col_ptrs,
                row_indices,
                values,
                is_transposed,
            } => vec![
                lit(L::Byte(0)),
                lit(L::Integer(*num_rows)),
                lit(L::Integer(*num_cols)),
                ints(col_ptrs),
                ints(row_indices),
                doubles(values),
                lit(L::Boolean(*is_transposed)),
            ],
        };
        lit(L::Struct(proto::expression::literal::Struct {
            struct_type: Some(proto::DataType {
                kind: Some(proto::data_type::Kind::Udt(Box::new(
                    proto::data_type::Udt {
                        r#type: "udt".into(),
                        jvm_class: Some("org.apache.spark.ml.linalg.MatrixUDT".into()),
                        ..Default::default()
                    },
                ))),
            }),
            elements,
        }))
    }
}

/// ML parameter handling: stores name -> literal value mappings.
///
/// Mirrors the structure of parameter handling in `pyspark.ml.param.Params`.
#[derive(Debug, Clone, Default)]
pub struct Params {
    /// User-supplied parameters as name -> Literal mappings
    params: HashMap<String, proto::Expression>,
}

impl Params {
    /// Create a new empty Params collection.
    pub fn new() -> Self {
        Params {
            params: HashMap::new(),
        }
    }

    /// Set a parameter to an integer value.
    pub fn set_param_int(mut self, name: &str, value: i64) -> Self {
        let mut literal = proto::Expression::default();
        let mut lit = proto::expression::Literal::default();
        lit.literal_type = Some(proto::expression::literal::LiteralType::Long(value));
        literal.expr_type = Some(proto::expression::ExprType::Literal(lit));
        self.params.insert(name.to_string(), literal);
        self
    }

    /// Set a parameter to a 32-bit integer (`Integer`) literal, as PySpark sends a Python int
    /// that fits 32 bits; for a parameter whose JVM type is `Int`, such as `maxIter`.
    pub fn set_param_int32(mut self, name: &str, value: i32) -> Self {
        let literal = proto::expression::Literal {
            literal_type: Some(proto::expression::literal::LiteralType::Integer(value)),
            ..Default::default()
        };
        self.params.insert(
            name.to_string(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(literal)),
                ..Default::default()
            },
        );
        self
    }

    /// Set a parameter to the narrowest integer literal that holds `value`, as PySpark sends a
    /// Python int: an `Integer` literal when it fits 32 bits, else a `Long`. The server converts
    /// it to the JVM type of the parameter (`Int` or `Long`, such as `seed`), as it converts the
    /// ints of PySpark.
    pub fn set_param_int_narrow(self, name: &str, value: i64) -> Self {
        match i32::try_from(value) {
            Ok(value) => self.set_param_int32(name, value),
            Err(_) => self.set_param_int(name, value),
        }
    }

    /// Set a parameter whose JVM type is `Array[String]` (such as `inputCols`), as PySpark sends
    /// a list of strings: an array literal of `string` elements.
    pub fn set_param_string_array<I, S>(mut self, name: &str, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let elements = values
            .into_iter()
            .map(|v| LiteralExpression::String(v.into()))
            .collect();
        let literal = LiteralExpression::Array {
            element_type: Box::new(DataType::String {
                collation: "UTF8_BINARY".to_string(),
            }),
            elements,
        }
        .to_literal_proto();
        self.params.insert(
            name.to_string(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(literal)),
                ..Default::default()
            },
        );
        self
    }

    /// Set a parameter to an array of 32-bit integers, as PySpark sends a list of ints.
    pub fn set_param_int32_array(self, name: &str, values: impl IntoIterator<Item = i32>) -> Self {
        self.set_param_int_array(name, values.into_iter().map(i64::from))
    }

    /// Set a parameter whose JVM type is `Array[Double]` (such as `thresholds`), as PySpark
    /// sends a list of floats: an array literal of `double` elements.
    pub fn set_param_double_array(
        mut self,
        name: &str,
        values: impl IntoIterator<Item = f64>,
    ) -> Self {
        let literal = LiteralExpression::Array {
            element_type: Box::new(DataType::Double),
            elements: values.into_iter().map(LiteralExpression::Double).collect(),
        }
        .to_literal_proto();
        self.params.insert(
            name.to_string(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(literal)),
                ..Default::default()
            },
        );
        self
    }

    /// Set a parameter to an integer array, preserving PySpark's literal width.
    #[allow(deprecated)] // Matches the array representation of PySpark's ML parameters.
    pub fn set_param_int_array(
        mut self,
        name: &str,
        values: impl IntoIterator<Item = i64>,
    ) -> Self {
        use proto::expression::literal::LiteralType as L;
        let values: Vec<i64> = values.into_iter().collect();
        let narrow = values.iter().all(|v| i32::try_from(*v).is_ok());
        let elements = values
            .into_iter()
            .map(|v| proto::expression::Literal {
                literal_type: Some(if narrow {
                    L::Integer(v as i32)
                } else {
                    L::Long(v)
                }),
                ..Default::default()
            })
            .collect();
        let literal = proto::expression::Literal {
            literal_type: Some(L::Array(proto::expression::literal::Array {
                element_type: Some(
                    if narrow {
                        DataType::Integer
                    } else {
                        DataType::Long
                    }
                    .to_proto(),
                ),
                elements,
            })),
            ..Default::default()
        };
        self.params.insert(
            name.into(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(literal)),
                ..Default::default()
            },
        );
        self
    }

    /// Set a vector parameter using Spark's VectorUDT representation.
    pub fn set_param_vector(mut self, name: &str, value: impl Into<MlVector>) -> Self {
        self.params.insert(
            name.into(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(
                    value.into().to_literal(),
                )),
                ..Default::default()
            },
        );
        self
    }

    /// Set a matrix parameter using Spark's MatrixUDT representation, as PySpark sends a
    /// `DenseMatrix` or a `SparseMatrix` (see [`MlMatrix`]).
    pub fn set_param_matrix(mut self, name: &str, value: impl Into<MlMatrix>) -> Self {
        self.params.insert(
            name.into(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(
                    value.into().to_literal(),
                )),
                ..Default::default()
            },
        );
        self
    }

    /// Set a parameter to a double value.
    pub fn set_param_double(mut self, name: &str, value: f64) -> Self {
        let mut literal = proto::Expression::default();
        let mut lit = proto::expression::Literal::default();
        lit.literal_type = Some(proto::expression::literal::LiteralType::Double(value));
        literal.expr_type = Some(proto::expression::ExprType::Literal(lit));
        self.params.insert(name.to_string(), literal);
        self
    }

    /// Set a parameter to a string value.
    pub fn set_param_string(mut self, name: &str, value: &str) -> Self {
        let mut literal = proto::Expression::default();
        let mut lit = proto::expression::Literal::default();
        lit.literal_type = Some(proto::expression::literal::LiteralType::String(
            value.to_string(),
        ));
        literal.expr_type = Some(proto::expression::ExprType::Literal(lit));
        self.params.insert(name.to_string(), literal);
        self
    }

    /// Set a parameter to a boolean value.
    pub fn set_param_bool(mut self, name: &str, value: bool) -> Self {
        let mut literal = proto::Expression::default();
        let mut lit = proto::expression::Literal::default();
        lit.literal_type = Some(proto::expression::literal::LiteralType::Boolean(value));
        literal.expr_type = Some(proto::expression::ExprType::Literal(lit));
        self.params.insert(name.to_string(), literal);
        self
    }

    /// Remove a parameter, so that it is not sent and the server uses its default, as
    /// PySpark's `Params.clear` does.
    pub fn clear_param(mut self, name: &str) -> Self {
        self.params.remove(name);
        self
    }

    /// Set a parameter to a null, as PySpark sends a None value of a param it was given.
    pub fn set_param_null(mut self, name: &str) -> Self {
        let literal = proto::expression::Literal {
            literal_type: Some(proto::expression::literal::LiteralType::Null(
                DataType::Null.to_proto(),
            )),
            ..Default::default()
        };
        self.params.insert(
            name.to_string(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(literal)),
                ..Default::default()
            },
        );
        self
    }

    /// Whether a parameter is an array without elements. PySpark cannot infer the element type
    /// of such an array, so it cannot send the params that hold one.
    pub fn has_empty_array(&self) -> bool {
        fn cannot_infer(lit: &proto::expression::Literal) -> bool {
            match &lit.literal_type {
                Some(proto::expression::literal::LiteralType::Array(array)) => {
                    array.elements.first().is_none_or(cannot_infer)
                }
                _ => false,
            }
        }
        self.params.values().any(|expr| match &expr.expr_type {
            Some(proto::expression::ExprType::Literal(lit)) => cannot_infer(lit),
            _ => false,
        })
    }

    /// Set a parameter to a literal, preserving its wire type.
    pub fn set_param_literal(mut self, name: &str, literal: proto::expression::Literal) -> Self {
        self.params.insert(
            name.into(),
            proto::Expression {
                expr_type: Some(proto::expression::ExprType::Literal(literal)),
                ..Default::default()
            },
        );
        self
    }

    /// Set nested double arrays, such as Bucketizer's per-column splits.
    pub fn set_param_double_arrays(
        self,
        name: &str,
        values: impl IntoIterator<Item = impl IntoIterator<Item = f64>>,
    ) -> Self {
        let literal = LiteralExpression::Array {
            element_type: Box::new(DataType::Array {
                element_type: Box::new(DataType::Double),
                contains_null: true,
            }),
            elements: values
                .into_iter()
                .map(|values| LiteralExpression::Array {
                    element_type: Box::new(DataType::Double),
                    elements: values.into_iter().map(LiteralExpression::Double).collect(),
                })
                .collect(),
        }
        .to_literal_proto();
        self.set_param_literal(name, literal)
    }

    /// Get a parameter value.
    pub fn get_param(&self, name: &str) -> Option<&proto::Expression> {
        self.params.get(name)
    }

    /// The literal value of the parameter `name`, if it is set.
    fn literal(&self, name: &str) -> Option<&proto::expression::literal::LiteralType> {
        match &self.params.get(name)?.expr_type {
            Some(proto::expression::ExprType::Literal(literal)) => literal.literal_type.as_ref(),
            _ => None,
        }
    }

    /// The string parameter `name`, or `default` when it is not set.
    fn string_or<'a>(&'a self, name: &str, default: &'a str) -> &'a str {
        self.string_opt(name).unwrap_or(default)
    }

    /// The `outputCol` parameter, or Spark's default of `HasOutputCol`, `<uid>__output`, of the
    /// object of `uid`, the column that the server writes when it is not set.
    fn output_col_or_default(&self, uid: &str) -> String {
        self.string_opt("outputCol")
            .map_or_else(|| format!("{uid}__output"), str::to_string)
    }

    /// The string parameter `name`, if it is set.
    fn string_opt(&self, name: &str) -> Option<&str> {
        match self.literal(name)? {
            proto::expression::literal::LiteralType::String(value) => Some(value),
            _ => None,
        }
    }

    /// The integer parameter `name`, if it is set, as the server reads it into a JVM `Long`.
    fn long_opt(&self, name: &str) -> Option<i64> {
        use proto::expression::literal::LiteralType as L;
        match self.literal(name)? {
            L::Integer(value) => Some(i64::from(*value)),
            L::Long(value) => Some(*value),
            // As the JVM's `Double.longValue`, which saturates.
            L::Double(value) => Some(*value as i64),
            _ => None,
        }
    }

    /// The integer parameter `name`, or `default` when it is not set, as the server reads it
    /// into a JVM `Int`: a `Long` wraps to 32 bits, as the JVM's `Long.intValue` does.
    fn int_or(&self, name: &str, default: i32) -> i32 {
        use proto::expression::literal::LiteralType as L;
        match self.literal(name) {
            Some(L::Integer(value)) => *value,
            Some(L::Long(value)) => *value as i32,
            Some(L::Double(value)) => *value as i32,
            _ => default,
        }
    }

    /// The floating-point parameter `name`, or `default` when it is not set.
    fn double_or(&self, name: &str, default: f64) -> f64 {
        use proto::expression::literal::LiteralType as L;
        match self.literal(name) {
            Some(L::Double(value)) => *value,
            Some(L::Integer(value)) => f64::from(*value),
            Some(L::Long(value)) => *value as f64,
            _ => default,
        }
    }

    /// The boolean parameter `name`, or `default` when it is not set.
    fn bool_or(&self, name: &str, default: bool) -> bool {
        match self.literal(name) {
            Some(proto::expression::literal::LiteralType::Boolean(value)) => *value,
            _ => default,
        }
    }

    /// The elements of the array parameter `name`, if it is set.
    fn array_opt(&self, name: &str) -> Option<&[proto::expression::Literal]> {
        match self.literal(name)? {
            proto::expression::literal::LiteralType::Array(array) => Some(&array.elements),
            _ => None,
        }
    }

    /// The string array parameter `name`, empty when it is not set.
    fn string_array(&self, name: &str) -> Vec<String> {
        self.array_opt(name)
            .unwrap_or_default()
            .iter()
            .filter_map(|element| match &element.literal_type {
                Some(proto::expression::literal::LiteralType::String(value)) => Some(value.clone()),
                _ => None,
            })
            .collect()
    }

    /// The floating-point array parameter `name`, if it is set.
    fn double_array_opt(&self, name: &str) -> Option<Vec<f64>> {
        let elements = self.array_opt(name)?;
        Some(
            elements
                .iter()
                .filter_map(|element| match element.literal_type {
                    Some(proto::expression::literal::LiteralType::Double(value)) => Some(value),
                    _ => None,
                })
                .collect(),
        )
    }
}

// The optional getters below are read by the rows that scripts/gen_rust_params_table.py prints
// for each converted class. Some have no row of the classes converted so far, so the allow stays.
#[allow(dead_code)]
impl Params {
    /// The integer parameter `name`, if it is set, as the server reads it into a JVM `Int`.
    fn int_opt(&self, name: &str) -> Option<i32> {
        use proto::expression::literal::LiteralType as L;
        match self.literal(name)? {
            L::Integer(value) => Some(*value),
            L::Long(value) => Some(*value as i32),
            L::Double(value) => Some(*value as i32),
            _ => None,
        }
    }

    /// The floating-point parameter `name`, if it is set.
    fn double_opt(&self, name: &str) -> Option<f64> {
        use proto::expression::literal::LiteralType as L;
        match self.literal(name)? {
            L::Double(value) => Some(*value),
            L::Integer(value) => Some(f64::from(*value)),
            L::Long(value) => Some(*value as f64),
            _ => None,
        }
    }

    /// The boolean parameter `name`, if it is set.
    fn bool_opt(&self, name: &str) -> Option<bool> {
        match self.literal(name)? {
            proto::expression::literal::LiteralType::Boolean(value) => Some(*value),
            _ => None,
        }
    }

    /// The string array parameter `name`, if it is set.
    fn string_array_opt(&self, name: &str) -> Option<Vec<String>> {
        let elements = self.array_opt(name)?;
        Some(
            elements
                .iter()
                .filter_map(|element| match &element.literal_type {
                    Some(proto::expression::literal::LiteralType::String(value)) => {
                        Some(value.clone())
                    }
                    _ => None,
                })
                .collect(),
        )
    }

    /// The integer array parameter `name`, if it is set, as the server reads it into `Int`s.
    fn int_array_opt(&self, name: &str) -> Option<Vec<i32>> {
        let elements = self.array_opt(name)?;
        Some(
            elements
                .iter()
                .filter_map(|element| match element.literal_type {
                    Some(proto::expression::literal::LiteralType::Integer(value)) => Some(value),
                    Some(proto::expression::literal::LiteralType::Long(value)) => {
                        Some(value as i32)
                    }
                    _ => None,
                })
                .collect(),
        )
    }

    /// The nested floating-point array parameter `name`, if it is set.
    fn double_arrays_opt(&self, name: &str) -> Option<Vec<Vec<f64>>> {
        use proto::expression::literal::LiteralType as L;
        let outer = self.array_opt(name)?;
        Some(
            outer
                .iter()
                .map(|element| match &element.literal_type {
                    Some(L::Array(inner)) => inner
                        .elements
                        .iter()
                        .filter_map(|value| match value.literal_type {
                            Some(L::Double(value)) => Some(value),
                            _ => None,
                        })
                        .collect(),
                    _ => Vec::new(),
                })
                .collect(),
        )
    }
}

impl Params {
    /// Convert to proto MlParams for transmission. MlParams stores `Literal`
    /// values, so we unwrap the literal from each stored Expression.
    pub fn to_proto(&self) -> proto::MlParams {
        let mut ml_params = proto::MlParams::default();
        for (name, expr) in &self.params {
            if let Some(proto::expression::ExprType::Literal(lit)) = &expr.expr_type {
                ml_params.params.insert(name.clone(), lit.clone());
            }
        }
        ml_params
    }

    /// Convert from proto MlParams (wrapping each Literal back into an Expression).
    pub fn from_proto(proto_params: &proto::MlParams) -> Self {
        let mut params = HashMap::new();
        for (name, lit) in &proto_params.params {
            let mut e = proto::Expression::default();
            e.expr_type = Some(proto::expression::ExprType::Literal(lit.clone()));
            params.insert(name.clone(), e);
        }
        Params { params }
    }
}

/// ML Operator type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorType {
    /// An estimator that learns from data via fit().
    Estimator,
    /// A transformer that applies transformations (possibly stateless).
    Transformer,
    /// An evaluator that measures model quality.
    Evaluator,
    /// A fitted model (result of fit).
    Model,
}

impl OperatorType {
    /// Convert to proto OperatorType.
    pub fn to_proto(&self) -> proto::ml_operator::OperatorType {
        match self {
            OperatorType::Estimator => proto::ml_operator::OperatorType::Estimator,
            OperatorType::Transformer => proto::ml_operator::OperatorType::Transformer,
            OperatorType::Evaluator => proto::ml_operator::OperatorType::Evaluator,
            OperatorType::Model => proto::ml_operator::OperatorType::Model,
        }
    }

    /// Convert from proto OperatorType.
    pub fn from_proto(proto_type: i32) -> Self {
        match proto::ml_operator::OperatorType::try_from(proto_type) {
            Ok(proto::ml_operator::OperatorType::Estimator) => OperatorType::Estimator,
            Ok(proto::ml_operator::OperatorType::Transformer) => OperatorType::Transformer,
            Ok(proto::ml_operator::OperatorType::Evaluator) => OperatorType::Evaluator,
            Ok(proto::ml_operator::OperatorType::Model) => OperatorType::Model,
            _ => OperatorType::Transformer,
        }
    }
}

/// ML Operator represents an ML class (Estimator, Transformer, Model, or Evaluator).
///
/// Mirrors `spark.connect.MlOperator` protobuf.
#[derive(Debug, Clone)]
pub struct MlOperator {
    /// Qualified class name (e.g., "org.apache.spark.ml.feature.StandardScaler")
    pub name: String,
    /// Unique identifier for this operator instance
    pub uid: String,
    /// Type of operator
    pub op_type: OperatorType,
}

impl MlOperator {
    /// Create a new MlOperator with an auto-generated UID: the simple name of the class and
    /// twelve random hex digits (`KMeans_1b2c3d4e5f60`), as PySpark's `Identifiable._randomUID`
    /// and Spark's `Identifiable.randomUID` make it, so that server messages name the class.
    pub fn new(name: &str, op_type: OperatorType) -> Self {
        let class = name.rsplit('.').next().unwrap_or(name);
        let hex = Uuid::new_v4().simple().to_string();
        MlOperator {
            name: name.to_string(),
            uid: format!("{class}_{}", &hex[hex.len() - 12..]),
            op_type,
        }
    }

    /// Create with a specific UID.
    pub fn with_uid(name: &str, uid: &str, op_type: OperatorType) -> Self {
        MlOperator {
            name: name.to_string(),
            uid: uid.to_string(),
            op_type,
        }
    }

    /// Convert to proto MlOperator.
    pub fn to_proto(&self) -> proto::MlOperator {
        proto::MlOperator {
            name: self.name.clone(),
            uid: self.uid.clone(),
            r#type: self.op_type.to_proto() as i32,
        }
    }

    /// Convert from proto MlOperator.
    pub fn from_proto(proto_op: &proto::MlOperator) -> Self {
        MlOperator {
            name: proto_op.name.clone(),
            uid: proto_op.uid.clone(),
            op_type: OperatorType::from_proto(proto_op.r#type),
        }
    }
}

/// Base trait for ML Estimators that fit to data and produce Models.
///
/// Mirrors `pyspark.ml.connect.base.Estimator`.
pub trait Estimator: Send + Sync {
    /// Get the operator definition for this estimator.
    fn operator(&self) -> &MlOperator;

    /// Get mutable operator definition.
    fn operator_mut(&mut self) -> &mut MlOperator;

    /// Get or initialize the operator.
    fn ensure_operator(&mut self) {
        if self.operator().uid.is_empty() {
            let new_op = MlOperator::new(&self.operator().name.clone(), OperatorType::Estimator);
            *self.operator_mut() = new_op;
        }
    }

    /// Get the parameters of this estimator.
    fn params(&self) -> &Params;

    /// Get mutable parameters.
    fn params_mut(&mut self) -> &mut Params;

    /// Fit this estimator to a DataFrame and return a Model.
    /// This is the main fitting method that must be implemented by subclasses.
    fn fit_impl(&mut self, _df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>>;

    /// Public fit method with optional parameter overrides.
    fn fit(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        self.ensure_operator();
        self.fit_impl(df)
    }

    /// A writer that saves the estimator, which is not fitted, of a session that
    /// [`MlWriter::session`] sets (see [`MlWriter`]), as `estimator.write()` of PySpark.
    fn write(&self) -> MlWriter {
        MlWriter::operator(self.operator(), self.params())
    }

    /// Set the values of `map` of the params of this estimator (of its uid), as PySpark's
    /// `copy(extra)` of a param map sets them; a meta-algorithm sets them in its stages.
    fn apply_param_map(&mut self, map: &ParamMap) -> spark_connect_core::error::Result<()> {
        self.ensure_operator();
        let uid = self.operator().uid.clone();
        meta::apply_pairs(self.params_mut(), &uid, map)
    }
}

/// Base trait for ML Transformers that apply transformations to data.
///
/// Mirrors `pyspark.ml.connect.base.Transformer`.
pub trait Transformer: Send + Sync {
    /// Get the operator definition for this transformer.
    fn operator(&self) -> &MlOperator;

    /// Get mutable operator definition.
    fn operator_mut(&mut self) -> &mut MlOperator;

    /// Get or initialize the operator.
    fn ensure_operator(&mut self) {
        if self.operator().uid.is_empty() {
            let new_op = MlOperator::new(&self.operator().name.clone(), OperatorType::Transformer);
            *self.operator_mut() = new_op;
        }
    }

    /// Get the parameters of this transformer.
    fn params(&self) -> &Params;

    /// Get mutable parameters.
    fn params_mut(&mut self) -> &mut Params;

    /// Transform a DataFrame by applying this transformer.
    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame>;

    /// Public transform method.
    fn transform(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.ensure_operator();
        self.transform_impl(df)
    }

    /// Set the values of `map` of the params of this transformer (of its uid), as PySpark's
    /// `copy(extra)` of a param map sets them; a pipeline model sets them in its stages.
    fn apply_param_map(&mut self, map: &ParamMap) -> spark_connect_core::error::Result<()> {
        self.ensure_operator();
        let uid = self.operator().uid.clone();
        meta::apply_pairs(self.params_mut(), &uid, map)
    }

    /// Build the MlRelation for this transformation.
    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        let mut transform = proto::ml_relation::Transform::default();
        transform.operator = Some(proto::ml_relation::transform::Operator::Transformer(
            self.operator().to_proto(),
        ));
        transform.input = Some(Box::new(input.relation()?));
        transform.params = Some(self.params().to_proto());

        let mut ml_relation = proto::MlRelation::default();
        ml_relation.ml_type = Some(proto::ml_relation::MlType::Transform(Box::new(transform)));
        Ok(ml_relation)
    }
}

/// Base trait for ML Models (result of fitting an Estimator).
///
/// Mirrors `pyspark.ml.connect.base.Model`.
pub trait Model: Transformer + AsAny {
    /// Clone this model into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Model>;

    /// The server-side object of the model, whose methods [`fetch`] reads (see
    /// [`MlTarget::object`]), of a model that the server holds. None of a model that holds no
    /// single object, such as a pipeline model.
    fn ml_ref(&self) -> Option<&MlRef> {
        None
    }

    /// The target of the methods of the model's server-side object, or an error of a model that
    /// the server does not hold.
    fn model_target(&self) -> spark_connect_core::error::Result<MlTarget> {
        self.ml_ref().map(MlTarget::object).ok_or_else(|| {
            SparkError::value_msg(format!(
                "{} is not a server-side model",
                self.operator().name
            ))
        })
    }

    /// The server's description of the model, as PySpark's `JavaModel.__repr__` reads it (its
    /// `toString`).
    fn describe(&self) -> spark_connect_core::error::Result<String> {
        fetch(&self.model_target()?, "toString", vec![])?.into_string()
    }

    /// A writer that saves the model (see [`MlWriter`]), of the session that holds it, as
    /// `model.write()` of PySpark.
    fn write(&self) -> spark_connect_core::error::Result<MlWriter> {
        let object = self.ml_ref().ok_or_else(|| {
            SparkError::value_msg(format!(
                "{} is not a server-side model to save",
                self.operator().name
            ))
        })?;
        Ok(MlWriter::object(object, self.params()))
    }
}

/// The concrete type of a boxed model, as `Model` is a supertrait of it: `model.downcast_ref`
/// of a `Box<dyn Model>` that a fit returns.
pub trait AsAny: std::any::Any {
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: std::any::Any> AsAny for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl dyn Model {
    /// The model as its concrete type `M`, if it is one (a `LinearRegressionModel` of the
    /// `Box<dyn Model>` of `LinearRegression::fit` of the trait).
    pub fn downcast_ref<M: Model>(&self) -> Option<&M> {
        self.as_any().downcast_ref::<M>()
    }
}

/// The members of every prediction model of `pyspark.ml` (`JavaPredictionModel`): the number of
/// the features, and the prediction of a single vector, of the model's server-side object.
pub trait PredictionModel: Model {
    /// The number of features that the model was trained on (`numFeatures`), or -1 if unknown.
    fn num_features(&self) -> spark_connect_core::error::Result<i32> {
        let value = fetch(&self.model_target()?, "numFeatures", vec![])?.into_value()?;
        value
            .as_i64()
            .map(|v| v as i32)
            .ok_or_else(|| SparkError::value_msg(format!("numFeatures is not an int: {value:?}")))
    }

    /// The prediction of a single vector of features (`predict`).
    fn predict(&self, features: &MlVector) -> spark_connect_core::error::Result<f64> {
        let value = fetch(
            &self.model_target()?,
            "predict",
            vec![FetchArg::Value(MlValue::Vector(features.clone()))],
        )?
        .into_value()?;
        value
            .as_f64()
            .ok_or_else(|| SparkError::value_msg(format!("predict is not a double: {value:?}")))
    }
}

/// Base trait for ML Evaluators that measure model quality.
///
/// Mirrors `pyspark.ml.connect.base.Evaluator`.
pub trait Evaluator: Send + Sync {
    /// Get the operator definition for this evaluator.
    fn operator(&self) -> &MlOperator;

    /// Get the parameters of this evaluator.
    fn params(&self) -> &Params;

    /// Get mutable parameters, where the evaluator exposes them. The default is None, so that an
    /// implementation outside this crate needs no change.
    fn params_mut(&mut self) -> Option<&mut Params> {
        None
    }

    /// Whether the metric should be maximized. Implementations may override this per metric.
    fn is_larger_better(&self) -> bool {
        true
    }

    /// Evaluate a DataFrame and return a metric value.
    fn evaluate(&self, _df: &DataFrame) -> spark_connect_core::error::Result<f64>;

    /// Set the values of `map` of the params of this evaluator (of its uid), as PySpark's
    /// `copy(extra)` of a param map sets them, of an evaluator whose params are mutable.
    fn apply_param_map(&mut self, map: &ParamMap) -> spark_connect_core::error::Result<()> {
        let uid = self.operator().uid.clone();
        match self.params_mut() {
            Some(params) => meta::apply_pairs(params, &uid, map),
            None => Ok(()),
        }
    }

    /// A writer that saves the evaluator, of a session that [`MlWriter::session`] sets (see
    /// [`MlWriter`]), as `evaluator.write()` of PySpark.
    fn write(&self) -> MlWriter {
        MlWriter::operator(self.operator(), self.params())
    }
}

/// The value of the method `method` of Spark Connect's ML helper, called with `args`, as PySpark's
/// `invoke_helper_attr` reads it: a `Fetch` of the helper, run as an ML command, whose result is
/// the literal of the value.
pub fn ml_helper_attribute(
    session: &SparkSession,
    method: &str,
    args: Vec<proto::expression::Literal>,
) -> spark_connect_core::error::Result<proto::expression::Literal> {
    use proto::fetch::method::args::ArgsType;
    let fetch = proto::Fetch {
        obj_ref: Some(proto::ObjectRef {
            id: "______ML_CONNECT_HELPER______".into(),
        }),
        methods: vec![proto::fetch::Method {
            method: method.into(),
            args: args
                .into_iter()
                .map(|literal| proto::fetch::method::Args {
                    args_type: Some(ArgsType::Param(literal)),
                })
                .collect(),
        }],
    };
    let mut ml_command = proto::MlCommand::default();
    ml_command.command = Some(proto::ml_command::Command::Fetch(fetch));
    let responses = crate::dataframe::execute_command_collect(
        session,
        proto::command::CommandType::MlCommand(ml_command),
    )?;
    for resp in responses {
        if let Some(proto::execute_plan_response::ResponseType::MlCommandResult(result)) =
            resp.response_type
        {
            if let Some(proto::ml_command_result::ResultType::Param(literal)) = result.result_type {
                return Ok(literal);
            }
        }
    }
    Err(SparkError::connect_msg(format!(
        "{method}: server returned no value"
    )))
}

/// The strings of `literal`, an array literal of strings, as the ML helper returns a `String[]`.
pub fn ml_string_array(
    literal: proto::expression::Literal,
) -> spark_connect_core::error::Result<Vec<String>> {
    use proto::expression::literal::LiteralType as L;
    let Some(L::Array(array)) = literal.literal_type else {
        return Err(SparkError::connect_msg("expected an array of strings"));
    };
    array
        .elements
        .into_iter()
        .map(|element| match element.literal_type {
            Some(L::String(value)) => Ok(value),
            _ => Err(SparkError::connect_msg("expected an array of strings")),
        })
        .collect()
}

/// Run an evaluator against a dataset via the ML `Evaluate` command and return the
/// scalar metric.
///
/// Mirrors `pyspark.ml.connect` evaluator.evaluate: build an `MlCommand::Evaluate`
/// carrying the evaluator operator, its params, and the dataset relation, submit
/// it, and read the returned metric literal from the `MlCommandResult`.
fn evaluate_via_command(
    operator: &MlOperator,
    params: &Params,
    df: &DataFrame,
) -> spark_connect_core::error::Result<f64> {
    let dataset = df.relation()?;
    let evaluate = proto::ml_command::Evaluate {
        evaluator: Some(operator.to_proto()),
        params: Some(params.to_proto()),
        dataset: Some(dataset),
    };
    let mut ml_command = proto::MlCommand::default();
    ml_command.command = Some(proto::ml_command::Command::Evaluate(evaluate));

    let responses = crate::dataframe::execute_command_collect(
        &df.session,
        proto::command::CommandType::MlCommand(ml_command),
    )?;
    for resp in responses {
        if let Some(proto::execute_plan_response::ResponseType::MlCommandResult(result)) =
            resp.response_type
        {
            if let Some(proto::ml_command_result::ResultType::Param(lit)) = result.result_type {
                if let Some(proto::expression::literal::LiteralType::Double(v)) = lit.literal_type {
                    return Ok(v);
                }
            }
        }
    }
    Err(spark_connect_core::error::SparkError::connect_msg(
        "evaluate: server returned no metric",
    ))
}

/// Concrete implementation: StandardScaler Estimator.
///
/// Scales features to have mean 0 and standard deviation 1.
#[derive(Debug, Clone)]
pub struct StandardScaler {
    operator: MlOperator,
    params: Params,
}

impl StandardScaler {
    /// Create a new StandardScaler with default parameters.
    pub fn new() -> Self {
        StandardScaler {
            operator: MlOperator::new(
                "org.apache.spark.ml.feature.StandardScaler",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<StandardScalerModel> {
        let cached = CachedModel::from_info(&info, session, "StandardScaler")?;
        Ok(StandardScalerModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.feature.StandardScalerModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
        })
    }
}

ml_params! {
    StandardScaler {
        set_input_col(value: &str) => "inputCol", set_param_string;
        input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
        set_output_col(value: &str) => "outputCol", set_param_string;
        output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
        set_with_mean(value: bool) => "withMean", set_param_bool;
        with_mean -> bool = |this| this.params.bool_or("withMean", false);
        set_with_std(value: bool) => "withStd", set_param_bool;
        with_std -> bool = |this| this.params.bool_or("withStd", true);
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for StandardScaler {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        // The server fits the scaler and caches the model, as PySpark does.
        let info = fit_via_command(&self.operator, &self.params, df)?;
        Ok(Box::new(self.model_from_info(info, &df.session)?))
    }
}

/// StandardScalerModel: a StandardScaler fitted by Spark Connect and cached for its fitting
/// session. Its transform refers to the cached model, as a fitted PySpark model's does.
#[derive(Debug, Clone)]
pub struct StandardScalerModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

impl Transformer for StandardScalerModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for StandardScalerModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

ml_params! {
    StandardScalerModel {
        set_input_col(value: &str) => "inputCol", set_param_string;
        input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
        set_output_col(value: &str) => "outputCol", set_param_string;
        output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
        with_mean -> bool = |this| this.params.bool_or("withMean", false) => "withMean";
        with_std -> bool = |this| this.params.bool_or("withStd", true) => "withStd";
    }
}

/// VectorAssembler Transformer: combines multiple columns into a single vector column.
#[derive(Debug, Clone)]
pub struct VectorAssembler {
    operator: MlOperator,
    params: Params,
}

impl VectorAssembler {
    pub fn new() -> Self {
        VectorAssembler {
            operator: MlOperator::new(
                "org.apache.spark.ml.feature.VectorAssembler",
                OperatorType::Transformer,
            ),
            params: Params::new(),
        }
    }
}

ml_params! {
    VectorAssembler {
        set_handle_invalid(value: &str) => "handleInvalid", set_param_string;
        handle_invalid -> &str = |this| this.params.string_or("handleInvalid", "error");
        set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) =>
            "inputCols", set_param_string_array;
        input_cols -> Vec<String> = |this| this.params.string_array("inputCols");
        set_output_col(value: &str) => "outputCol", set_param_string;
        output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    }
}

impl Default for VectorAssembler {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for VectorAssembler {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        let ml_relation = self.build_ml_relation(df)?;
        let mut relation = proto::Relation::default();
        relation.common = Some(proto::RelationCommon::default());
        relation.rel_type = Some(proto::relation::RelType::MlRelation(Box::new(ml_relation)));
        let plan = LogicalPlan::MlTransform {
            ml_relation: relation,
            input: df.plan.clone(),
            owner: None,
        };
        Ok(DataFrame::new(df.session.clone(), plan))
    }
}

/// StringIndexer Estimator: converts string columns to numeric indices.
#[derive(Debug, Clone)]
pub struct StringIndexer {
    operator: MlOperator,
    params: Params,
}

impl StringIndexer {
    pub fn new() -> Self {
        StringIndexer {
            operator: MlOperator::new(
                "org.apache.spark.ml.feature.StringIndexer",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<StringIndexerModel> {
        let cached = CachedModel::from_info(&info, session, "StringIndexer")?;
        Ok(StringIndexerModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.feature.StringIndexerModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
        })
    }
}

ml_params! {
    StringIndexer {
        set_handle_invalid(value: &str) => "handleInvalid", set_param_string;
        handle_invalid -> &str = |this| this.params.string_or("handleInvalid", "error");
        set_input_col(value: &str) => "inputCol", set_param_string;
        input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
        set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) =>
            "inputCols", set_param_string_array;
        input_cols -> Vec<String> = |this| this.params.string_array("inputCols");
        set_output_col(value: &str) => "outputCol", set_param_string;
        output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
        set_output_cols(value: impl IntoIterator<Item = impl Into<String>>) =>
            "outputCols", set_param_string_array;
        output_cols -> Vec<String> = |this| this.params.string_array("outputCols");
        set_string_order_type(value: &str) => "stringOrderType", set_param_string;
        string_order_type -> &str =
            |this| this.params.string_or("stringOrderType", "frequencyDesc");
    }
}

impl Default for StringIndexer {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for StringIndexer {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        // The server fits the indexer and caches the model, as PySpark does.
        let info = fit_via_command(&self.operator, &self.params, df)?;
        Ok(Box::new(self.model_from_info(info, &df.session)?))
    }
}

/// StringIndexerModel: a StringIndexer fitted by Spark Connect and cached for its fitting
/// session. Its transform refers to the cached model, as a fitted PySpark model's does.
#[derive(Debug, Clone)]
pub struct StringIndexerModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

impl Transformer for StringIndexerModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for StringIndexerModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

ml_params! {
    StringIndexerModel {
        set_handle_invalid(value: &str) => "handleInvalid", set_param_string;
        handle_invalid -> &str = |this| this.params.string_or("handleInvalid", "error");
        set_input_col(value: &str) => "inputCol", set_param_string;
        input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
        set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) =>
            "inputCols", set_param_string_array;
        input_cols -> Option<Vec<String>> = |this| this.params.string_array_opt("inputCols");
        set_output_col(value: &str) => "outputCol", set_param_string;
        output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
        set_output_cols(value: impl IntoIterator<Item = impl Into<String>>) =>
            "outputCols", set_param_string_array;
        output_cols -> Option<Vec<String>> = |this| this.params.string_array_opt("outputCols");
        string_order_type -> &str =
            |this| this.params.string_or("stringOrderType", "frequencyDesc") => "stringOrderType";
    }
}

/// MaxAbsScaler Estimator: rescales each feature to range [-1, 1].
#[derive(Debug, Clone)]
pub struct MaxAbsScaler {
    operator: MlOperator,
    params: Params,
}

impl MaxAbsScaler {
    pub fn new() -> Self {
        MaxAbsScaler {
            operator: MlOperator::new(
                "org.apache.spark.ml.feature.MaxAbsScaler",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<MaxAbsScalerModel> {
        let cached = CachedModel::from_info(&info, session, "MaxAbsScaler")?;
        Ok(MaxAbsScalerModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.feature.MaxAbsScalerModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
        })
    }
}

ml_params! {
    MaxAbsScaler {
        set_input_col(value: &str) => "inputCol", set_param_string;
        input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
        set_output_col(value: &str) => "outputCol", set_param_string;
        output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    }
}

impl Default for MaxAbsScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MaxAbsScaler {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        // The server fits the scaler and caches the model, as PySpark does.
        let info = fit_via_command(&self.operator, &self.params, df)?;
        Ok(Box::new(self.model_from_info(info, &df.session)?))
    }
}

/// MaxAbsScalerModel: a MaxAbsScaler fitted by Spark Connect and cached for its fitting
/// session. Its transform refers to the cached model, as a fitted PySpark model's does.
#[derive(Debug, Clone)]
pub struct MaxAbsScalerModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

impl Transformer for MaxAbsScalerModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for MaxAbsScalerModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

ml_params! {
    MaxAbsScalerModel {
        set_input_col(value: &str) => "inputCol", set_param_string;
        input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
        set_output_col(value: &str) => "outputCol", set_param_string;
        output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    }
}

/// Submit training data and parameters and read the fitted operator metadata.
fn fit_via_command(
    operator: &MlOperator,
    params: &Params,
    df: &DataFrame,
) -> spark_connect_core::error::Result<proto::ml_command_result::MlOperatorInfo> {
    let command = proto::MlCommand {
        command: Some(proto::ml_command::Command::Fit(proto::ml_command::Fit {
            estimator: Some(operator.to_proto()),
            params: Some(params.to_proto()),
            dataset: Some(df.relation()?),
        })),
    };
    let responses = crate::dataframe::execute_command_collect(
        &df.session,
        proto::command::CommandType::MlCommand(command),
    )?;
    for response in responses {
        if let Some(proto::execute_plan_response::ResponseType::MlCommandResult(result)) =
            response.response_type
        {
            if let Some(proto::ml_command_result::ResultType::OperatorInfo(info)) =
                result.result_type
            {
                return Ok(info);
            }
        }
    }
    Err(spark_connect_core::error::SparkError::connect_msg(format!(
        "{}.fit: server returned no fitted model",
        operator.name
    )))
}

/// A server-side ML object: a fitted model, or another object that a fetch returns (a submodel, a
/// tree of an ensemble), shared by its clones and by every DataFrame whose plan reads it (through
/// the [`RelationOwner`] of their plans). The last owner to drop deletes the cached object on the
/// server, as PySpark's `RemoteModelRef` does.
#[derive(Debug, Clone)]
pub struct MlRef(Arc<CachedModelRef>);

/// The name of [`MlRef`] in the code of the models, which caches them.
type CachedModel = MlRef;

struct CachedModelRef {
    model_ref: proto::ObjectRef,
    session: SparkSession,
}

impl std::fmt::Debug for CachedModelRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedModelRef")
            .field("model_ref", &self.model_ref)
            .finish_non_exhaustive()
    }
}

impl Drop for CachedModelRef {
    fn drop(&mut self) {
        // Sent without waiting, and its errors (a stopped session, a closed channel) are
        // ignored, as PySpark ignores them when it releases a model.
        self.session
            .release_ml_objects(vec![self.model_ref.clone()]);
    }
}

impl CachedModel {
    fn from_info(
        info: &proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
        estimator: &str,
    ) -> spark_connect_core::error::Result<Self> {
        match &info.r#type {
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(reference))
                if !reference.id.is_empty() =>
            {
                Ok(Self(Arc::new(CachedModelRef {
                    model_ref: reference.clone(),
                    session: session.clone(),
                })))
            }
            _ => Err(spark_connect_core::error::SparkError::connect_msg(format!(
                "{estimator}.fit: server returned no model object reference"
            ))),
        }
    }

    fn build_ml_relation(
        &self,
        df: &DataFrame,
        params: &Params,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        Ok(proto::MlRelation {
            ml_type: Some(proto::ml_relation::MlType::Transform(Box::new(
                proto::ml_relation::Transform {
                    operator: Some(proto::ml_relation::transform::Operator::ObjRef(
                        self.0.model_ref.clone(),
                    )),
                    input: Some(Box::new(df.relation()?)),
                    params: Some(params.to_proto()),
                },
            ))),
            ..Default::default()
        })
    }

    fn transform(
        &self,
        df: &DataFrame,
        params: &Params,
    ) -> spark_connect_core::error::Result<DataFrame> {
        if df.session.session_id() != self.0.session.session_id() {
            return Err(spark_connect_core::error::SparkError::value_msg(
                "A fitted ML model must transform data from its fitting session",
            ));
        }
        let relation = proto::Relation {
            rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                self.build_ml_relation(df, params)?,
            ))),
            ..Default::default()
        };
        // The DataFrame keeps this model, and the plan of its input, alive.
        Ok(DataFrame::new(
            df.session.clone(),
            LogicalPlan::MlTransform {
                ml_relation: relation,
                input: df.plan.clone(),
                owner: Some(RelationOwner(self.0.clone())),
            },
        ))
    }
}

/// LogisticRegression Estimator: binary/multiclass classification.
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    operator: MlOperator,
    params: Params,
}

impl LogisticRegression {
    pub fn new() -> Self {
        LogisticRegression {
            operator: MlOperator::new(
                "org.apache.spark.ml.classification.LogisticRegression",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<LogisticRegressionModel> {
        let cached = CachedModel::from_info(&info, session, "LogisticRegression")?;
        Ok(LogisticRegressionModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.classification.LogisticRegressionModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
            summary: None,
        })
    }
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LogisticRegression {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(self.fit(df)?))
    }
}

/// A LogisticRegression model fitted and cached by the Spark Connect server.
///
/// Obtain this model through [`Estimator::fit`]. Its cached object reference is
/// distinct from its operator UID and belongs to the session used for fitting.
/// The server keeps the cached model while a model, a clone of it, or a DataFrame
/// it transformed is alive, then deletes it.
#[derive(Debug, Clone)]
pub struct LogisticRegressionModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
    /// The target of the training summary, which a fit sets (`_summary`); None of a model that the
    /// server returns without a fit, such as a loaded one.
    summary: Option<MlTarget>,
}

impl LogisticRegressionModel {}

impl Transformer for LogisticRegressionModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for LogisticRegressionModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Linear support vector classifier using Spark's JVM implementation.
///
/// Supports binary labels (0 and 1), ML vector features and L2 regularization.
/// Unset parameters use Spark's defaults: 100 iterations, no regularization,
/// tolerance 1e-6, an intercept, standardization and threshold 0.
#[derive(Debug, Clone)]
pub struct LinearSVC {
    operator: MlOperator,
    params: Params,
}

impl LinearSVC {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.classification.LinearSVC",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<LinearSVCModel> {
        let cached = CachedModel::from_info(&info, session, "LinearSVC")?;
        Ok(LinearSVCModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.classification.LinearSVCModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
            summary: None,
        })
    }
}

impl Default for LinearSVC {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LinearSVC {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(self.fit(df)?))
    }
}

/// A LinearSVC model fitted by Spark Connect and cached for its fitting session.
/// Obtain it through [`Estimator::fit`]; prediction uses the cached model object,
/// which the server deletes once no model, clone or transformed DataFrame refers to it.
#[derive(Debug, Clone)]
pub struct LinearSVCModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
    /// The target of the training summary, which a fit sets (`_summary`); None of a model that the
    /// server returns without a fit, such as a loaded one.
    summary: Option<MlTarget>,
}

impl Transformer for LinearSVCModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for LinearSVCModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// FMClassifier using Spark's JVM implementation and ML vectors.
/// Unset parameters use Spark's defaults.
#[derive(Debug, Clone)]
pub struct FMClassifier {
    operator: MlOperator,
    params: Params,
}

impl FMClassifier {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.classification.FMClassifier",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    /// Fit the typed FM classification model, whose training summary the model holds.
    pub fn fit(
        &mut self,
        df: &DataFrame,
    ) -> spark_connect_core::error::Result<FMClassificationModel> {
        let info = fit_via_command(&self.operator, &self.params, df)?;
        let mut model = self.model_from_info(info, &df.session)?;
        model.summary_target = Some(
            MlTarget::object(&model.cached)
                .child("summary")
                .with_summary_dataset(df),
        );
        Ok(model)
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<FMClassificationModel> {
        let cached = CachedModel::from_info(&info, session, "FMClassifier")?;
        Ok(FMClassificationModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.classification.FMClassificationModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
            summary_target: None,
        })
    }
}

impl Default for FMClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for FMClassifier {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(self.fit(df)?))
    }
}

/// A FMClassifier model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the server-cached object until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct FMClassificationModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
    /// The target of the training summary, which a fit sets (`hasSummary` of PySpark's
    /// `HasTrainingSummary`); None of a loaded model.
    summary_target: Option<MlTarget>,
}

impl Transformer for FMClassificationModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for FMClassificationModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// MultilayerPerceptronClassifier using Spark's JVM implementation and ML vectors.
/// Unset parameters use Spark's defaults.
#[derive(Debug, Clone)]
pub struct MultilayerPerceptronClassifier {
    operator: MlOperator,
    params: Params,
}

impl MultilayerPerceptronClassifier {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.classification.MultilayerPerceptronClassifier",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    /// Fit the typed MLP classification model, whose training summary the model holds.
    pub fn fit(
        &mut self,
        df: &DataFrame,
    ) -> spark_connect_core::error::Result<MultilayerPerceptronClassificationModel> {
        let info = fit_via_command(&self.operator, &self.params, df)?;
        let mut model = self.model_from_info(info, &df.session)?;
        model.summary_target = Some(
            MlTarget::object(&model.cached)
                .child("summary")
                .with_summary_dataset(df),
        );
        Ok(model)
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<MultilayerPerceptronClassificationModel> {
        let cached = CachedModel::from_info(&info, session, "MultilayerPerceptronClassifier")?;
        Ok(MultilayerPerceptronClassificationModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
            summary_target: None,
        })
    }
}

impl Default for MultilayerPerceptronClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultilayerPerceptronClassifier {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(self.fit(df)?))
    }
}

/// A MultilayerPerceptronClassifier model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the server-cached object until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct MultilayerPerceptronClassificationModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
    /// The target of the training summary, which a fit sets (`hasSummary` of PySpark's
    /// `HasTrainingSummary`); None of a loaded model.
    summary_target: Option<MlTarget>,
}

impl Transformer for MultilayerPerceptronClassificationModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for MultilayerPerceptronClassificationModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// A native binary classifier that can be fitted independently for each OneVsRest class.
#[derive(Debug, Clone)]
pub enum Classifier {
    LogisticRegression(LogisticRegression),
    LinearSVC(LinearSVC),
    DecisionTreeClassifier(DecisionTreeClassifier),
    RandomForestClassifier(RandomForestClassifier),
    GBTClassifier(GBTClassifier),
    NaiveBayes(NaiveBayes),
    FMClassifier(FMClassifier),
    MultilayerPerceptronClassifier(MultilayerPerceptronClassifier),
}

macro_rules! classifier_from {
    ($($name:ident),* $(,)?) => {$(
        impl From<$name> for Classifier {
            fn from(value: $name) -> Self { Self::$name(value) }
        }
    )*};
}
classifier_from!(
    LogisticRegression,
    LinearSVC,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
    NaiveBayes,
    FMClassifier,
    MultilayerPerceptronClassifier
);

impl Classifier {
    fn estimator_mut(&mut self) -> &mut dyn Estimator {
        match self {
            Self::LogisticRegression(x) => x,
            Self::LinearSVC(x) => x,
            Self::DecisionTreeClassifier(x) => x,
            Self::RandomForestClassifier(x) => x,
            Self::GBTClassifier(x) => x,
            Self::NaiveBayes(x) => x,
            Self::FMClassifier(x) => x,
            Self::MultilayerPerceptronClassifier(x) => x,
        }
    }

    /// Whether this classifier accepts instance weights.
    pub fn supports_weight(&self) -> bool {
        !matches!(
            self,
            Self::FMClassifier(_) | Self::MultilayerPerceptronClassifier(_)
        )
    }
}

/// Multiclass reduction that fits one native binary classifier per class.
///
/// As in PySpark, class indices run from zero through the maximum label. Fits use
/// independent snapshots of the base classifier and at most `parallelism` threads.
#[derive(Debug, Clone)]
pub struct OneVsRest {
    operator: MlOperator,
    params: Params,
    classifier: Option<Classifier>,
}

impl OneVsRest {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.classification.OneVsRest",
                OperatorType::Estimator,
            ),
            params: Params::new(),
            classifier: None,
        }
    }

    /// Set the native binary classifier. Fits do not mutate the supplied estimator.
    pub fn set_classifier(mut self, classifier: impl Into<Classifier>) -> Self {
        self.classifier = Some(classifier.into());
        self
    }

    /// The `parallelism` that a fit reads, of either int width, unlike [`Self::parallelism`],
    /// which narrows it: None when it is set to a null.
    pub fn parallelism_i64(&self) -> Option<i64> {
        use proto::expression::literal::LiteralType as L;
        match self.params.literal("parallelism") {
            Some(L::Null(_)) => None,
            Some(L::Integer(value)) => Some(i64::from(*value)),
            Some(L::Long(value)) => Some(*value),
            _ => Some(1),
        }
    }

    /// The error of a column param of a fit that is set to a null, which a fit raises first.
    pub fn check_required_params(&self) -> spark_connect_core::error::Result<()> {
        for name in [
            "featuresCol",
            "labelCol",
            "predictionCol",
            "rawPredictionCol",
        ] {
            if matches!(
                self.params.literal(name),
                Some(proto::expression::literal::LiteralType::Null(_))
            ) {
                return Err(SparkError::value_msg(format!(
                    "OneVsRest parameter {name} cannot be null"
                )));
            }
        }
        Ok(())
    }

    /// The largest label of `df`, of which a fit takes its number of classes. As PySpark's fit,
    /// it reads it first, with `dataset.agg({labelCol: "max"})`: None when `df` has no rows or only
    /// null labels.
    pub fn max_label(&self, df: &DataFrame) -> spark_connect_core::error::Result<Option<f64>> {
        let rows = df
            .agg([crate::functions::max(
                ml_exact_column(self.label_col()).cast(DataType::Double),
            )])
            .collect()?;
        Ok(rows.first().and_then(|r| r.get(0)).and_then(|v| v.as_f64()))
    }

    /// Fit one binary classifier of each of `num_classes` classes on `df`, `threads` at a time:
    /// the fit after its checks, with the class count of [`Self::max_label`].
    pub fn fit_classes(
        &self,
        df: &DataFrame,
        num_classes: usize,
        threads: usize,
    ) -> spark_connect_core::error::Result<OneVsRestModel> {
        use crate::column::{lit_double, when};
        use std::sync::atomic::{AtomicUsize, Ordering};
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| SparkError::value_msg("OneVsRest requires a native classifier"))?;
        let count = num_classes;
        let weight = self
            .weight_col()
            .filter(|name| !name.is_empty() && classifier.supports_weight());
        let mut columns = vec![
            ml_exact_column(self.label_col()),
            ml_exact_column(self.features_col()),
        ];
        if let Some(weight) = weight {
            columns.push(ml_exact_column(weight));
        }
        let training = df.select(columns);
        let level = df.storage_level()?;
        let _cache = OneVsRestTrainingCache(if !level.use_memory && !level.use_disk {
            Some(training.persist(proto::StorageLevel {
                use_memory: true,
                use_disk: true,
                replication: 1,
                ..Default::default()
            })?)
        } else {
            None
        });
        let next = AtomicUsize::new(0);
        let results = std::thread::scope(|scope| {
            let mut workers = Vec::new();
            for _ in 0..threads.min(count) {
                let next = &next;
                let training = &training;
                let this = &self;
                workers.push(scope.spawn(move || {
                    let mut results = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= count {
                            break;
                        }
                        let binary_label = format!("{}_label_{}", this.operator.uid, index);
                        let binary = training.with_column(
                            &binary_label,
                            when(
                                ml_exact_column(this.label_col()).eq(lit_double(index as f64)),
                                lit_double(1.0),
                            )
                            .otherwise(lit_double(0.0)),
                        );
                        let mut base = classifier.clone();
                        let estimator = base.estimator_mut();
                        let mut params = std::mem::take(estimator.params_mut())
                            .set_param_string("labelCol", &binary_label)
                            .set_param_string("featuresCol", this.features_col())
                            .set_param_string("predictionCol", this.prediction_col());
                        if let Some(weight) = weight {
                            params = params.set_param_string("weightCol", weight);
                        }
                        *estimator.params_mut() = params;
                        results.push((index, estimator.fit(&binary)));
                    }
                    results
                }));
            }
            let mut results = Vec::new();
            for worker in workers {
                results.extend(
                    worker
                        .join()
                        .map_err(|_| SparkError::connect_msg("OneVsRest worker panicked"))?,
                );
            }
            Ok::<_, SparkError>(results)
        })?;
        let mut results = results;
        results.sort_by_key(|(index, _)| *index);
        let models = results
            .into_iter()
            .map(|(_, model)| model)
            .collect::<spark_connect_core::error::Result<Vec<_>>>()?;
        Ok(OneVsRestModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.classification.OneVsRestModel",
                &self.operator.uid,
                OperatorType::Model,
            ),
            params: self.params.clone(),
            models,
            session_id: df.session.session_id().to_string(),
            classifier: self.classifier.clone(),
        })
    }
}

impl Default for OneVsRest {
    fn default() -> Self {
        Self::new()
    }
}

// Release only the cache created for this fit, including after a failed binary fit.
struct OneVsRestTrainingCache(Option<DataFrame>);
impl Drop for OneVsRestTrainingCache {
    fn drop(&mut self) {
        if let Some(df) = &self.0 {
            let _ = df.unpersist(false);
        }
    }
}

fn ml_exact_column(name: &str) -> crate::column::Column {
    crate::column::col(&format!("`{}`", name.replace('`', "``")))
}

impl Estimator for OneVsRest {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }
    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(self.fit(df)?))
    }
}

impl OneVsRest {
    /// Fit the typed OneVsRest model, of one fitted binary classifier per class.
    pub fn fit(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<OneVsRestModel> {
        // The params are checked before any request. The Python client, which raises PySpark's
        // errors in PySpark's order, reads the max label first, with `max_label` and `fit_classes`.
        self.check_required_params()?;
        let parallelism = self.parallelism_i64().ok_or_else(|| {
            SparkError::value_msg("OneVsRest parameter parallelism cannot be null")
        })?;
        if self.classifier.is_none() {
            return Err(SparkError::value_msg(
                "OneVsRest requires a native classifier",
            ));
        }
        if parallelism < 1 {
            return Err(SparkError::value_msg("parallelism must be at least 1"));
        }
        let max = self.max_label(df)?.ok_or_else(|| {
            SparkError::value_msg("OneVsRest cannot fit an empty or null-labeled dataset")
        })?;
        if !max.is_finite() || max < 0.0 || max >= i32::MAX as f64 {
            return Err(SparkError::value_msg(
                "OneVsRest requires a finite nonnegative maximum class label",
            ));
        }
        let count = max as usize + 1;
        // At most one thread of each class, as PySpark's ThreadPool(min(parallelism, classes)).
        self.fit_classes(df, count, parallelism.min(count as i64) as usize)
    }
}

/// A composite model whose submodels remain alive through lazy transformed DataFrames.
pub struct OneVsRestModel {
    operator: MlOperator,
    params: Params,
    models: Vec<Box<dyn Model>>,
    session_id: String,
    /// The classifier that fitted the models (`classifier` of PySpark's `_OneVsRestParams`).
    classifier: Option<Classifier>,
}

impl Clone for OneVsRestModel {
    fn clone(&self) -> Self {
        Self {
            operator: self.operator.clone(),
            params: self.params.clone(),
            models: self.models.iter().map(|m| m.clone_box()).collect(),
            session_id: self.session_id.clone(),
            classifier: self.classifier.clone(),
        }
    }
}

impl OneVsRestModel {
    pub fn num_classes(&self) -> usize {
        self.models.len()
    }

    fn transformed(&self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        use crate::column::{lit, Column};
        use crate::expression::Expression;
        use crate::functions::{array, array_append, get};
        if df.session.session_id() != self.session_id {
            return Err(SparkError::value_msg(
                "A fitted ML model must transform data from its fitting session",
            ));
        }
        let original = df.columns()?;
        let uid = Uuid::new_v4().to_string();
        let acc = format!("{uid}_scores");
        let raw = format!("{uid}_raw");
        let mut result = df.clone();
        for (index, base) in self.models.iter().enumerate() {
            let mut model = base.clone_box();
            let mut params = std::mem::take(model.params_mut())
                .set_param_string(
                    "featuresCol",
                    self.params.string_or("featuresCol", "features"),
                )
                .set_param_string("rawPredictionCol", &raw)
                .set_param_string("predictionCol", "");
            if !model.operator().name.ends_with("LinearSVCModel") {
                params = params.set_param_string("probabilityCol", "");
            }
            *model.params_mut() = params;
            let transformed = model.transform(&result)?;
            let values = Column::new(Expression::function(
                "vector_to_array",
                vec![
                    ml_exact_column(&raw).expression().clone(),
                    crate::column::lit_string("float64").expression().clone(),
                ],
            ));
            let score = get(values, lit(1));
            result = transformed
                .with_column(
                    &acc,
                    if index == 0 {
                        array([score])
                    } else {
                        array_append(ml_exact_column(&acc), score)
                    },
                )
                .select(
                    original
                        .iter()
                        .map(|name| ml_exact_column(name))
                        .chain(std::iter::once(ml_exact_column(&acc))),
                );
        }
        let raw_output = self.params.string_or("rawPredictionCol", "rawPrediction");
        if !raw_output.is_empty() {
            result = result.with_column(
                raw_output,
                Column::new(Expression::function(
                    "array_to_vector",
                    vec![ml_exact_column(&acc).expression().clone()],
                )),
            );
        }
        let prediction = self.params.string_or("predictionCol", "prediction");
        if !prediction.is_empty() {
            result = result.with_column(prediction, one_vs_rest_prediction(ml_exact_column(&acc)));
        }
        Ok(result.drop_columns(std::iter::empty::<Column>(), [acc]))
    }
}

// PySpark's array_argmax ignores NaN/null, keeps the first tie and returns -1
// when no score exceeds negative infinity.
fn one_vs_rest_prediction(scores: crate::column::Column) -> crate::column::Column {
    use crate::column::{lit, lit_double, when};
    use crate::functions::{aggregate_with_finish, r#struct};
    aggregate_with_finish(
        scores,
        r#struct([
            lit_double(f64::NEG_INFINITY).alias("v"),
            lit(-1).alias("i"),
            lit(0).alias("j"),
        ]),
        |args| {
            let value = args[1].clone();
            let best = args[0].clone().get_field("v");
            let index = args[0].clone().get_field("i");
            let current = args[0].clone().get_field("j");
            when(
                value
                    .clone()
                    .is_nan()
                    .not()
                    .and(value.clone().is_null().not())
                    .and(value.clone().gt(best.clone())),
                r#struct([
                    value.alias("v"),
                    current.clone().alias("i"),
                    (current.clone() + lit(1)).alias("j"),
                ]),
            )
            .otherwise(r#struct([
                best.alias("v"),
                index.alias("i"),
                (current + lit(1)).alias("j"),
            ]))
        },
        |args| args[0].clone().get_field("i"),
    )
    .cast(DataType::Double)
}

impl Transformer for OneVsRestModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }
    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.transformed(df)
    }
    fn build_ml_relation(
        &self,
        df: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        // The composite transform is a SQL plan, represented as the input of an identity
        // SQLTransformer when a caller specifically requests an MlRelation.
        Ok(proto::MlRelation {
            ml_type: Some(proto::ml_relation::MlType::Transform(Box::new(
                proto::ml_relation::Transform {
                    operator: Some(proto::ml_relation::transform::Operator::Transformer(
                        MlOperator::new(
                            "org.apache.spark.ml.feature.SQLTransformer",
                            OperatorType::Transformer,
                        )
                        .to_proto(),
                    )),
                    input: Some(Box::new(self.transformed(df)?.relation()?)),
                    params: Some(
                        Params::new()
                            .set_param_string("statement", "SELECT * FROM __THIS__")
                            .to_proto(),
                    ),
                },
            ))),
            ..Default::default()
        })
    }
}
impl Model for OneVsRestModel {
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Linear regression using Spark's JVM implementation and ML vector features.
///
/// Supports squared-error and Huber loss. Unset parameters use Spark's defaults.
#[derive(Debug, Clone)]
pub struct LinearRegression {
    operator: MlOperator,
    params: Params,
}

impl LinearRegression {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.regression.LinearRegression",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<LinearRegressionModel> {
        let cached = CachedModel::from_info(&info, session, "LinearRegression")?;
        Ok(LinearRegressionModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.regression.LinearRegressionModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
            training_summary: None,
        })
    }
}

ml_params! {
    LinearRegression {
        set_aggregation_depth(value: i32) => "aggregationDepth", set_param_int32;
        aggregation_depth -> i32 = |this| this.params.int_or("aggregationDepth", 2);
        set_elastic_net_param(value: f64) => "elasticNetParam", set_param_double;
        elastic_net_param -> f64 = |this| this.params.double_or("elasticNetParam", 0.0);
        set_epsilon(value: f64) => "epsilon", set_param_double;
        epsilon -> f64 = |this| this.params.double_or("epsilon", 1.35);
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_fit_intercept(value: bool) => "fitIntercept", set_param_bool;
        fit_intercept -> bool = |this| this.params.bool_or("fitIntercept", true);
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_loss(value: &str) => "loss", set_param_string;
        loss -> &str = |this| this.params.string_or("loss", "squaredError");
        set_max_block_size_in_mb(value: f64) => "maxBlockSizeInMB", set_param_double;
        max_block_size_in_mb -> f64 = |this| this.params.double_or("maxBlockSizeInMB", 0.0);
        set_max_iter(value: i32) => "maxIter", set_param_int32;
        max_iter -> i32 = |this| this.params.int_or("maxIter", 100);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_reg_param(value: f64) => "regParam", set_param_double;
        reg_param -> f64 = |this| this.params.double_or("regParam", 0.0);
        set_solver(value: &str) => "solver", set_param_string;
        solver -> &str = |this| this.params.string_or("solver", "auto");
        set_standardization(value: bool) => "standardization", set_param_bool;
        standardization -> bool = |this| this.params.bool_or("standardization", true);
        set_tol(value: f64) => "tol", set_param_double;
        tol -> f64 = |this| this.params.double_or("tol", 1e-06);
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LinearRegression {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(LinearRegression::fit(self, df)?))
    }
}

/// A linear regression model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the server-cached object until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct LinearRegressionModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
    /// The summary of the model of its training dataset, of a model that this client fitted.
    training_summary: Option<MlTarget>,
}

ml_params! {
    LinearRegressionModel {
        aggregation_depth -> i32 = |this| this.params.int_or("aggregationDepth", 2) =>
            "aggregationDepth";
        elastic_net_param -> f64 = |this| this.params.double_or("elasticNetParam", 0.0) =>
            "elasticNetParam";
        epsilon -> f64 = |this| this.params.double_or("epsilon", 1.35) => "epsilon";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        fit_intercept -> bool = |this| this.params.bool_or("fitIntercept", true) => "fitIntercept";
        label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
        loss -> &str = |this| this.params.string_or("loss", "squaredError") => "loss";
        max_block_size_in_mb -> f64 = |this| this.params.double_or("maxBlockSizeInMB", 0.0) =>
            "maxBlockSizeInMB";
        max_iter -> i32 = |this| this.params.int_or("maxIter", 100) => "maxIter";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        reg_param -> f64 = |this| this.params.double_or("regParam", 0.0) => "regParam";
        solver -> &str = |this| this.params.string_or("solver", "auto") => "solver";
        standardization -> bool = |this| this.params.bool_or("standardization", true) =>
            "standardization";
        tol -> f64 = |this| this.params.double_or("tol", 1e-06) => "tol";
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

impl Transformer for LinearRegressionModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for LinearRegressionModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Generalized linear regression using Spark's JVM implementation and ML vector features.
///
/// Supports Gaussian, binomial, Poisson, Gamma and Tweedie families.
/// Unset parameters, including the family-dependent link, use Spark's defaults.
#[derive(Debug, Clone)]
pub struct GeneralizedLinearRegression {
    operator: MlOperator,
    params: Params,
}

impl GeneralizedLinearRegression {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.regression.GeneralizedLinearRegression",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<GeneralizedLinearRegressionModel> {
        let cached = CachedModel::from_info(&info, session, "GeneralizedLinearRegression")?;
        Ok(GeneralizedLinearRegressionModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.regression.GeneralizedLinearRegressionModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
            training_summary: None,
        })
    }
}

ml_params! {
    GeneralizedLinearRegression {
        set_aggregation_depth(value: i32) => "aggregationDepth", set_param_int32;
        aggregation_depth -> i32 = |this| this.params.int_or("aggregationDepth", 2);
        set_family(value: &str) => "family", set_param_string;
        family -> &str = |this| this.params.string_or("family", "gaussian");
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_fit_intercept(value: bool) => "fitIntercept", set_param_bool;
        fit_intercept -> bool = |this| this.params.bool_or("fitIntercept", true);
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_link(value: &str) => "link", set_param_string;
        link -> Option<&str> = |this| this.params.string_opt("link");
        set_link_power(value: f64) => "linkPower", set_param_double;
        link_power -> Option<f64> = |this| this.params.double_opt("linkPower");
        set_link_prediction_col(value: &str) => "linkPredictionCol", set_param_string;
        link_prediction_col -> Option<&str> = |this| this.params.string_opt("linkPredictionCol");
        set_max_iter(value: i32) => "maxIter", set_param_int32;
        max_iter -> i32 = |this| this.params.int_or("maxIter", 25);
        set_offset_col(value: &str) => "offsetCol", set_param_string;
        offset_col -> Option<&str> = |this| this.params.string_opt("offsetCol");
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_reg_param(value: f64) => "regParam", set_param_double;
        reg_param -> f64 = |this| this.params.double_or("regParam", 0.0);
        set_solver(value: &str) => "solver", set_param_string;
        solver -> &str = |this| this.params.string_or("solver", "irls");
        set_tol(value: f64) => "tol", set_param_double;
        tol -> f64 = |this| this.params.double_or("tol", 1e-06);
        set_variance_power(value: f64) => "variancePower", set_param_double;
        variance_power -> f64 = |this| this.params.double_or("variancePower", 0.0);
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

impl Default for GeneralizedLinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GeneralizedLinearRegression {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(GeneralizedLinearRegression::fit(self, df)?))
    }
}

/// A generalized linear regression model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the server-cached object until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct GeneralizedLinearRegressionModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
    /// The summary of the model of its training dataset, of a model that this client fitted.
    training_summary: Option<MlTarget>,
}

ml_params! {
    GeneralizedLinearRegressionModel {
        aggregation_depth -> i32 = |this| this.params.int_or("aggregationDepth", 2) =>
            "aggregationDepth";
        family -> &str = |this| this.params.string_or("family", "gaussian") => "family";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        fit_intercept -> bool = |this| this.params.bool_or("fitIntercept", true) => "fitIntercept";
        label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
        link -> Option<&str> = |this| this.params.string_opt("link") => "link";
        link_power -> Option<f64> = |this| this.params.double_opt("linkPower") => "linkPower";
        set_link_prediction_col(value: &str) => "linkPredictionCol", set_param_string;
        link_prediction_col -> Option<&str> = |this| this.params.string_opt("linkPredictionCol");
        max_iter -> i32 = |this| this.params.int_or("maxIter", 25) => "maxIter";
        offset_col -> Option<&str> = |this| this.params.string_opt("offsetCol") => "offsetCol";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        reg_param -> f64 = |this| this.params.double_or("regParam", 0.0) => "regParam";
        solver -> &str = |this| this.params.string_or("solver", "irls") => "solver";
        tol -> f64 = |this| this.params.double_or("tol", 1e-06) => "tol";
        variance_power -> f64 = |this| this.params.double_or("variancePower", 0.0) =>
            "variancePower";
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

impl Transformer for GeneralizedLinearRegressionModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for GeneralizedLinearRegressionModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Weibull accelerated failure time regression using Spark's JVM implementation.
///
/// Unset parameters use Spark's defaults.
#[derive(Debug, Clone)]
pub struct AFTSurvivalRegression {
    operator: MlOperator,
    params: Params,
}

impl AFTSurvivalRegression {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.regression.AFTSurvivalRegression",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<AFTSurvivalRegressionModel> {
        let cached = CachedModel::from_info(&info, session, "AFTSurvivalRegression")?;
        Ok(AFTSurvivalRegressionModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.regression.AFTSurvivalRegressionModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
        })
    }
}

ml_params! {
    AFTSurvivalRegression {
        set_aggregation_depth(value: i32) => "aggregationDepth", set_param_int32;
        aggregation_depth -> i32 = |this| this.params.int_or("aggregationDepth", 2);
        set_censor_col(value: &str) => "censorCol", set_param_string;
        censor_col -> &str = |this| this.params.string_or("censorCol", "censor");
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_fit_intercept(value: bool) => "fitIntercept", set_param_bool;
        fit_intercept -> bool = |this| this.params.bool_or("fitIntercept", true);
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_max_block_size_in_mb(value: f64) => "maxBlockSizeInMB", set_param_double;
        max_block_size_in_mb -> f64 = |this| this.params.double_or("maxBlockSizeInMB", 0.0);
        set_max_iter(value: i32) => "maxIter", set_param_int32;
        max_iter -> i32 = |this| this.params.int_or("maxIter", 100);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_quantile_probabilities(value: impl IntoIterator<Item = f64>) => "quantileProbabilities",
            set_param_double_array;
        quantile_probabilities -> Vec<f64> = |this| this.params.double_array_opt(
            "quantileProbabilities").unwrap_or_else(|| vec![0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9,
            0.95, 0.99]);
        set_quantiles_col(value: &str) => "quantilesCol", set_param_string;
        quantiles_col -> Option<&str> = |this| this.params.string_opt("quantilesCol");
        set_tol(value: f64) => "tol", set_param_double;
        tol -> f64 = |this| this.params.double_or("tol", 1e-06);
    }
}

impl Default for AFTSurvivalRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for AFTSurvivalRegression {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(AFTSurvivalRegression::fit(self, df)?))
    }
}

/// A fitted regression model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the server-cached object until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct AFTSurvivalRegressionModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

ml_params! {
    AFTSurvivalRegressionModel {
        aggregation_depth -> i32 = |this| this.params.int_or("aggregationDepth", 2) =>
            "aggregationDepth";
        censor_col -> &str = |this| this.params.string_or("censorCol", "censor") => "censorCol";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        fit_intercept -> bool = |this| this.params.bool_or("fitIntercept", true) => "fitIntercept";
        label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
        max_block_size_in_mb -> f64 = |this| this.params.double_or("maxBlockSizeInMB", 0.0) =>
            "maxBlockSizeInMB";
        max_iter -> i32 = |this| this.params.int_or("maxIter", 100) => "maxIter";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_quantile_probabilities(value: impl IntoIterator<Item = f64>) => "quantileProbabilities",
            set_param_double_array;
        quantile_probabilities -> Vec<f64> = |this| this.params.double_array_opt(
            "quantileProbabilities").unwrap_or_else(|| vec![0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9,
            0.95, 0.99]);
        set_quantiles_col(value: &str) => "quantilesCol", set_param_string;
        quantiles_col -> Option<&str> = |this| this.params.string_opt("quantilesCol");
        tol -> f64 = |this| this.params.double_or("tol", 1e-06) => "tol";
    }
}

impl Transformer for AFTSurvivalRegressionModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for AFTSurvivalRegressionModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Univariate isotonic or antitonic regression using Spark's JVM implementation.
///
/// Unset parameters use Spark's defaults.
#[derive(Debug, Clone)]
pub struct IsotonicRegression {
    operator: MlOperator,
    params: Params,
}

impl IsotonicRegression {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.regression.IsotonicRegression",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<IsotonicRegressionModel> {
        let cached = CachedModel::from_info(&info, session, "IsotonicRegression")?;
        Ok(IsotonicRegressionModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.regression.IsotonicRegressionModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
        })
    }
}

ml_params! {
    IsotonicRegression {
        set_feature_index(value: i32) => "featureIndex", set_param_int32;
        feature_index -> i32 = |this| this.params.int_or("featureIndex", 0);
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_isotonic(value: bool) => "isotonic", set_param_bool;
        isotonic -> bool = |this| this.params.bool_or("isotonic", true);
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

impl Default for IsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for IsotonicRegression {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(IsotonicRegression::fit(self, df)?))
    }
}

/// A fitted regression model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the server-cached object until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct IsotonicRegressionModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

ml_params! {
    IsotonicRegressionModel {
        set_feature_index(value: i32) => "featureIndex", set_param_int32;
        feature_index -> i32 = |this| this.params.int_or("featureIndex", 0);
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        isotonic -> bool = |this| this.params.bool_or("isotonic", true) => "isotonic";
        label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

impl Transformer for IsotonicRegressionModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for IsotonicRegressionModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Factorization machines regression using Spark's JVM implementation.
///
/// Unset parameters use Spark's defaults.
#[derive(Debug, Clone)]
pub struct FMRegressor {
    operator: MlOperator,
    params: Params,
}

impl FMRegressor {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.regression.FMRegressor",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<FMRegressionModel> {
        let cached = CachedModel::from_info(&info, session, "FMRegressor")?;
        Ok(FMRegressionModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.regression.FMRegressionModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
        })
    }
}

ml_params! {
    FMRegressor {
        set_factor_size(value: i32) => "factorSize", set_param_int32;
        factor_size -> i32 = |this| this.params.int_or("factorSize", 8);
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_fit_intercept(value: bool) => "fitIntercept", set_param_bool;
        fit_intercept -> bool = |this| this.params.bool_or("fitIntercept", true);
        set_fit_linear(value: bool) => "fitLinear", set_param_bool;
        fit_linear -> bool = |this| this.params.bool_or("fitLinear", true);
        set_init_std(value: f64) => "initStd", set_param_double;
        init_std -> f64 = |this| this.params.double_or("initStd", 0.01);
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_max_iter(value: i32) => "maxIter", set_param_int32;
        max_iter -> i32 = |this| this.params.int_or("maxIter", 100);
        set_mini_batch_fraction(value: f64) => "miniBatchFraction", set_param_double;
        mini_batch_fraction -> f64 = |this| this.params.double_or("miniBatchFraction", 1.0);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_reg_param(value: f64) => "regParam", set_param_double;
        reg_param -> f64 = |this| this.params.double_or("regParam", 0.0);
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.regression.FMRegressor"));
        set_solver(value: &str) => "solver", set_param_string;
        solver -> &str = |this| this.params.string_or("solver", "adamW");
        set_step_size(value: f64) => "stepSize", set_param_double;
        step_size -> f64 = |this| this.params.double_or("stepSize", 1.0);
        set_tol(value: f64) => "tol", set_param_double;
        tol -> f64 = |this| this.params.double_or("tol", 1e-06);
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

impl Default for FMRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for FMRegressor {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(FMRegressor::fit(self, df)?))
    }
}

/// A fitted regression model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the server-cached object until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct FMRegressionModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

ml_params! {
    FMRegressionModel {
        factor_size -> i32 = |this| this.params.int_or("factorSize", 8) => "factorSize";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        fit_intercept -> bool = |this| this.params.bool_or("fitIntercept", true) => "fitIntercept";
        fit_linear -> bool = |this| this.params.bool_or("fitLinear", true) => "fitLinear";
        init_std -> f64 = |this| this.params.double_or("initStd", 0.01) => "initStd";
        label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
        max_iter -> i32 = |this| this.params.int_or("maxIter", 100) => "maxIter";
        mini_batch_fraction -> f64 = |this| this.params.double_or("miniBatchFraction", 1.0) =>
            "miniBatchFraction";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        reg_param -> f64 = |this| this.params.double_or("regParam", 0.0) => "regParam";
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.regression.FMRegressionModel")) => "seed";
        solver -> &str = |this| this.params.string_or("solver", "adamW") => "solver";
        step_size -> f64 = |this| this.params.double_or("stepSize", 1.0) => "stepSize";
        tol -> f64 = |this| this.params.double_or("tol", 1e-06) => "tol";
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

impl Transformer for FMRegressionModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for FMRegressionModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Decision tree classification using Spark's JVM implementation and ML vector features.
///
/// Supports binary and multiclass labels. Unset parameters use Spark's defaults. A single
/// tree samples its rows to find split candidates only when there are more than
/// max(maxBins^2, 10000) rows; the seed picks that sample, or else Spark's default seed (the
/// hash of the class name), as with PySpark, so fits are reproducible either way. Equal-gain
/// splits are chosen deterministically, so the seed does not break ties.
#[derive(Debug, Clone)]
pub struct DecisionTreeClassifier {
    operator: MlOperator,
    params: Params,
}

impl DecisionTreeClassifier {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.classification.DecisionTreeClassifier",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<DecisionTreeClassificationModel> {
        let cached = CachedModel::from_info(&info, session, "DecisionTreeClassifier")?;
        Ok(DecisionTreeClassificationModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.classification.DecisionTreeClassificationModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
        })
    }
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for DecisionTreeClassifier {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(self.fit(df)?))
    }
}

/// A decision tree classification model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the cached server model until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct DecisionTreeClassificationModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

impl Transformer for DecisionTreeClassificationModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for DecisionTreeClassificationModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Naive Bayes classification using Spark's JVM implementation and ML vectors.
///
/// Supports multinomial, Bernoulli, complement and Gaussian models. Multinomial and
/// complement features must be nonnegative, Bernoulli features must be 0 or 1, and
/// Gaussian features may be continuous and negative. Unset parameters use Spark's defaults.
#[derive(Debug, Clone)]
pub struct NaiveBayes {
    operator: MlOperator,
    params: Params,
}

impl NaiveBayes {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.classification.NaiveBayes",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    /// Fit the typed Naive Bayes model, which holds the server-side model object.
    pub fn fit(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<NaiveBayesModel> {
        let info = fit_via_command(&self.operator, &self.params, df)?;
        self.model_from_info(info, &df.session)
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<NaiveBayesModel> {
        let cached = CachedModel::from_info(&info, session, "NaiveBayes")?;
        Ok(NaiveBayesModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.classification.NaiveBayesModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
        })
    }
}

impl Default for NaiveBayes {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for NaiveBayes {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(self.fit(df)?))
    }
}

/// A Naive Bayes classification model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the cached server model until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct NaiveBayesModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

impl Transformer for NaiveBayesModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for NaiveBayesModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Gradient-boosted binary classification using Spark's JVM implementation and ML vectors.
///
/// Labels must be 0 or 1. Unset parameters use Spark's defaults. A seed controls row sampling
/// for split candidates and subsampling, and feature sampling when a subset is requested.
#[derive(Debug, Clone)]
pub struct GBTClassifier {
    operator: MlOperator,
    params: Params,
}

impl GBTClassifier {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.classification.GBTClassifier",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    /// Relative improvement tolerance for validation-set early stopping.
    pub fn set_validation_tol(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("validationTol", value);
        self
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<GBTClassificationModel> {
        let cached = CachedModel::from_info(&info, session, "GBTClassifier")?;
        Ok(GBTClassificationModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.classification.GBTClassificationModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
        })
    }
}

impl Default for GBTClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GBTClassifier {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(self.fit(df)?))
    }
}

/// A gradient-boosted classification model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the cached server model until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct GBTClassificationModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

impl Transformer for GBTClassificationModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for GBTClassificationModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Random forest classification using Spark's JVM implementation and ML vector features.
///
/// Supports binary and multiclass labels. Unset parameters use Spark's defaults. Without a
/// seed, the sampling uses Spark's default seed (the hash of the class name), as with
/// PySpark, so fits are reproducible either way; set a seed to draw other samples.
#[derive(Debug, Clone)]
pub struct RandomForestClassifier {
    operator: MlOperator,
    params: Params,
}

impl RandomForestClassifier {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.classification.RandomForestClassifier",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<RandomForestClassificationModel> {
        let cached = CachedModel::from_info(&info, session, "RandomForestClassifier")?;
        Ok(RandomForestClassificationModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.classification.RandomForestClassificationModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
            training_summary: None,
        })
    }
}

impl Default for RandomForestClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RandomForestClassifier {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(self.fit(df)?))
    }
}

/// A random forest classification model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the cached server model until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct RandomForestClassificationModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
    /// The target of the training summary (`<model>.summary`), which a fit sets with the transform
    /// of its training dataset, so that the server can recreate the summary; None otherwise.
    training_summary: Option<MlTarget>,
}

impl Transformer for RandomForestClassificationModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for RandomForestClassificationModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Gradient-boosted regression using Spark's JVM implementation and ML vectors.
///
/// Labels are continuous numeric values. Unset parameters use Spark's defaults. A seed controls row sampling
/// for split candidates and subsampling, and feature sampling when a subset is requested.
#[derive(Debug, Clone)]
pub struct GBTRegressor {
    operator: MlOperator,
    params: Params,
}

impl GBTRegressor {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.regression.GBTRegressor",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<GBTRegressionModel> {
        let cached = CachedModel::from_info(&info, session, "GBTRegressor")?;
        Ok(GBTRegressionModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.regression.GBTRegressionModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
        })
    }
}

ml_params! {
    GBTRegressor {
        set_cache_node_ids(value: bool) => "cacheNodeIds", set_param_bool;
        cache_node_ids -> bool = |this| this.params.bool_or("cacheNodeIds", false);
        set_checkpoint_interval(value: i32) => "checkpointInterval", set_param_int32;
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10);
        set_feature_subset_strategy(value: &str) => "featureSubsetStrategy", set_param_string;
        feature_subset_strategy -> &str = |this| this.params.string_or("featureSubsetStrategy",
            "all");
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_impurity(value: &str) => "impurity", set_param_string;
        impurity -> &str = |this| this.params.string_or("impurity", "variance");
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_leaf_col(value: &str) => "leafCol", set_param_string;
        leaf_col -> &str = |this| this.params.string_or("leafCol", "");
        set_loss_type(value: &str) => "lossType", set_param_string;
        loss_type -> &str = |this| this.params.string_or("lossType", "squared");
        set_max_bins(value: i32) => "maxBins", set_param_int32;
        max_bins -> i32 = |this| this.params.int_or("maxBins", 32);
        set_max_depth(value: i32) => "maxDepth", set_param_int32;
        max_depth -> i32 = |this| this.params.int_or("maxDepth", 5);
        set_max_iter(value: i32) => "maxIter", set_param_int32;
        max_iter -> i32 = |this| this.params.int_or("maxIter", 20);
        set_max_memory_in_mb(value: i32) => "maxMemoryInMB", set_param_int32;
        max_memory_in_mb -> i32 = |this| this.params.int_or("maxMemoryInMB", 256);
        set_min_info_gain(value: f64) => "minInfoGain", set_param_double;
        min_info_gain -> f64 = |this| this.params.double_or("minInfoGain", 0.0);
        set_min_instances_per_node(value: i32) => "minInstancesPerNode", set_param_int32;
        min_instances_per_node -> i32 = |this| this.params.int_or("minInstancesPerNode", 1);
        set_min_weight_fraction_per_node(value: f64) => "minWeightFractionPerNode",
            set_param_double;
        min_weight_fraction_per_node -> f64 = |this| this.params.double_or(
            "minWeightFractionPerNode", 0.0);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.regression.GBTRegressor"));
        set_step_size(value: f64) => "stepSize", set_param_double;
        step_size -> f64 = |this| this.params.double_or("stepSize", 0.1);
        set_subsampling_rate(value: f64) => "subsamplingRate", set_param_double;
        subsampling_rate -> f64 = |this| this.params.double_or("subsamplingRate", 1.0);
        set_validation_indicator_col(value: &str) => "validationIndicatorCol", set_param_string;
        validation_indicator_col -> Option<&str> = |this| this.params.string_opt(
            "validationIndicatorCol");
        set_validation_tol(value: f64) => "validationTol", set_param_double;
        validation_tol -> f64 = |this| this.params.double_or("validationTol", 0.01);
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

impl Default for GBTRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GBTRegressor {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(GBTRegressor::fit(self, df)?))
    }
}

/// A gradient-boosted regression model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the cached server model until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct GBTRegressionModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

ml_params! {
    GBTRegressionModel {
        cache_node_ids -> bool = |this| this.params.bool_or("cacheNodeIds", false) =>
            "cacheNodeIds";
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10) =>
            "checkpointInterval";
        feature_subset_strategy -> &str = |this| this.params.string_or("featureSubsetStrategy",
            "all") => "featureSubsetStrategy";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        impurity -> &str = |this| this.params.string_or("impurity", "variance") => "impurity";
        label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
        set_leaf_col(value: &str) => "leafCol", set_param_string;
        leaf_col -> &str = |this| this.params.string_or("leafCol", "");
        loss_type -> &str = |this| this.params.string_or("lossType", "squared") => "lossType";
        max_bins -> i32 = |this| this.params.int_or("maxBins", 32) => "maxBins";
        max_depth -> i32 = |this| this.params.int_or("maxDepth", 5) => "maxDepth";
        max_iter -> i32 = |this| this.params.int_or("maxIter", 20) => "maxIter";
        max_memory_in_mb -> i32 = |this| this.params.int_or("maxMemoryInMB", 256) =>
            "maxMemoryInMB";
        min_info_gain -> f64 = |this| this.params.double_or("minInfoGain", 0.0) => "minInfoGain";
        min_instances_per_node -> i32 = |this| this.params.int_or("minInstancesPerNode", 1) =>
            "minInstancesPerNode";
        min_weight_fraction_per_node -> f64 = |this| this.params.double_or(
            "minWeightFractionPerNode", 0.0) => "minWeightFractionPerNode";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.regression.GBTRegressionModel")) => "seed";
        step_size -> f64 = |this| this.params.double_or("stepSize", 0.1) => "stepSize";
        subsampling_rate -> f64 = |this| this.params.double_or("subsamplingRate", 1.0) =>
            "subsamplingRate";
        validation_indicator_col -> Option<&str> = |this| this.params.string_opt(
            "validationIndicatorCol") => "validationIndicatorCol";
        validation_tol -> f64 = |this| this.params.double_or("validationTol", 0.01) =>
            "validationTol";
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

impl Transformer for GBTRegressionModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for GBTRegressionModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Decision tree regression using Spark's JVM implementation and ML vector features.
///
/// Supports continuous numeric labels. Unset parameters use Spark's defaults. A single tree
/// samples its rows to find split candidates only when there are more than
/// max(maxBins^2, 10000) rows; the seed picks that sample, or else Spark's default seed (the
/// hash of the class name), as with PySpark, so fits are reproducible either way. Equal-gain
/// splits are chosen deterministically, so the seed does not break ties.
#[derive(Debug, Clone)]
pub struct DecisionTreeRegressor {
    operator: MlOperator,
    params: Params,
}

impl DecisionTreeRegressor {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.regression.DecisionTreeRegressor",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<DecisionTreeRegressionModel> {
        let cached = CachedModel::from_info(&info, session, "DecisionTreeRegressor")?;
        Ok(DecisionTreeRegressionModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.regression.DecisionTreeRegressionModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
        })
    }
}

ml_params! {
    DecisionTreeRegressor {
        set_cache_node_ids(value: bool) => "cacheNodeIds", set_param_bool;
        cache_node_ids -> bool = |this| this.params.bool_or("cacheNodeIds", false);
        set_checkpoint_interval(value: i32) => "checkpointInterval", set_param_int32;
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10);
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_impurity(value: &str) => "impurity", set_param_string;
        impurity -> &str = |this| this.params.string_or("impurity", "variance");
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_leaf_col(value: &str) => "leafCol", set_param_string;
        leaf_col -> &str = |this| this.params.string_or("leafCol", "");
        set_max_bins(value: i32) => "maxBins", set_param_int32;
        max_bins -> i32 = |this| this.params.int_or("maxBins", 32);
        set_max_depth(value: i32) => "maxDepth", set_param_int32;
        max_depth -> i32 = |this| this.params.int_or("maxDepth", 5);
        set_max_memory_in_mb(value: i32) => "maxMemoryInMB", set_param_int32;
        max_memory_in_mb -> i32 = |this| this.params.int_or("maxMemoryInMB", 256);
        set_min_info_gain(value: f64) => "minInfoGain", set_param_double;
        min_info_gain -> f64 = |this| this.params.double_or("minInfoGain", 0.0);
        set_min_instances_per_node(value: i32) => "minInstancesPerNode", set_param_int32;
        min_instances_per_node -> i32 = |this| this.params.int_or("minInstancesPerNode", 1);
        set_min_weight_fraction_per_node(value: f64) => "minWeightFractionPerNode",
            set_param_double;
        min_weight_fraction_per_node -> f64 = |this| this.params.double_or(
            "minWeightFractionPerNode", 0.0);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.regression.DecisionTreeRegressor"));
        set_variance_col(value: &str) => "varianceCol", set_param_string;
        variance_col -> Option<&str> = |this| this.params.string_opt("varianceCol");
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

impl Default for DecisionTreeRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for DecisionTreeRegressor {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(DecisionTreeRegressor::fit(self, df)?))
    }
}

/// A decision tree regression model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the cached server model until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct DecisionTreeRegressionModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

ml_params! {
    DecisionTreeRegressionModel {
        cache_node_ids -> bool = |this| this.params.bool_or("cacheNodeIds", false) =>
            "cacheNodeIds";
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10) =>
            "checkpointInterval";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        impurity -> &str = |this| this.params.string_or("impurity", "variance") => "impurity";
        label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
        set_leaf_col(value: &str) => "leafCol", set_param_string;
        leaf_col -> &str = |this| this.params.string_or("leafCol", "");
        max_bins -> i32 = |this| this.params.int_or("maxBins", 32) => "maxBins";
        max_depth -> i32 = |this| this.params.int_or("maxDepth", 5) => "maxDepth";
        max_memory_in_mb -> i32 = |this| this.params.int_or("maxMemoryInMB", 256) =>
            "maxMemoryInMB";
        min_info_gain -> f64 = |this| this.params.double_or("minInfoGain", 0.0) => "minInfoGain";
        min_instances_per_node -> i32 = |this| this.params.int_or("minInstancesPerNode", 1) =>
            "minInstancesPerNode";
        min_weight_fraction_per_node -> f64 = |this| this.params.double_or(
            "minWeightFractionPerNode", 0.0) => "minWeightFractionPerNode";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.regression.DecisionTreeRegressionModel")) => "seed";
        set_variance_col(value: &str) => "varianceCol", set_param_string;
        variance_col -> Option<&str> = |this| this.params.string_opt("varianceCol");
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

impl Transformer for DecisionTreeRegressionModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for DecisionTreeRegressionModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}
/// Random forest regression using Spark's JVM implementation and ML vector features.
///
/// Supports continuous numeric labels. Unset parameters use Spark's defaults. Without a
/// seed, the sampling uses Spark's default seed (the hash of the class name), as with
/// PySpark, so fits are reproducible either way; set a seed to draw other samples.
#[derive(Debug, Clone)]
pub struct RandomForestRegressor {
    operator: MlOperator,
    params: Params,
}

impl RandomForestRegressor {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.regression.RandomForestRegressor",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<RandomForestRegressionModel> {
        let cached = CachedModel::from_info(&info, session, "RandomForestRegressor")?;
        Ok(RandomForestRegressionModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.regression.RandomForestRegressionModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
        })
    }
}

ml_params! {
    RandomForestRegressor {
        set_bootstrap(value: bool) => "bootstrap", set_param_bool;
        bootstrap -> bool = |this| this.params.bool_or("bootstrap", true);
        set_cache_node_ids(value: bool) => "cacheNodeIds", set_param_bool;
        cache_node_ids -> bool = |this| this.params.bool_or("cacheNodeIds", false);
        set_checkpoint_interval(value: i32) => "checkpointInterval", set_param_int32;
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10);
        set_feature_subset_strategy(value: &str) => "featureSubsetStrategy", set_param_string;
        feature_subset_strategy -> &str = |this| this.params.string_or("featureSubsetStrategy",
            "auto");
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_impurity(value: &str) => "impurity", set_param_string;
        impurity -> &str = |this| this.params.string_or("impurity", "variance");
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_leaf_col(value: &str) => "leafCol", set_param_string;
        leaf_col -> &str = |this| this.params.string_or("leafCol", "");
        set_max_bins(value: i32) => "maxBins", set_param_int32;
        max_bins -> i32 = |this| this.params.int_or("maxBins", 32);
        set_max_depth(value: i32) => "maxDepth", set_param_int32;
        max_depth -> i32 = |this| this.params.int_or("maxDepth", 5);
        set_max_memory_in_mb(value: i32) => "maxMemoryInMB", set_param_int32;
        max_memory_in_mb -> i32 = |this| this.params.int_or("maxMemoryInMB", 256);
        set_min_info_gain(value: f64) => "minInfoGain", set_param_double;
        min_info_gain -> f64 = |this| this.params.double_or("minInfoGain", 0.0);
        set_min_instances_per_node(value: i32) => "minInstancesPerNode", set_param_int32;
        min_instances_per_node -> i32 = |this| this.params.int_or("minInstancesPerNode", 1);
        set_min_weight_fraction_per_node(value: f64) => "minWeightFractionPerNode",
            set_param_double;
        min_weight_fraction_per_node -> f64 = |this| this.params.double_or(
            "minWeightFractionPerNode", 0.0);
        set_num_trees(value: i32) => "numTrees", set_param_int32;
        num_trees -> i32 = |this| this.params.int_or("numTrees", 20);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.regression.RandomForestRegressor"));
        set_subsampling_rate(value: f64) => "subsamplingRate", set_param_double;
        subsampling_rate -> f64 = |this| this.params.double_or("subsamplingRate", 1.0);
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

impl Default for RandomForestRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RandomForestRegressor {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(RandomForestRegressor::fit(self, df)?))
    }
}

/// A random forest regression model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the cached server model until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct RandomForestRegressionModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

ml_params! {
    RandomForestRegressionModel {
        bootstrap -> bool = |this| this.params.bool_or("bootstrap", true) => "bootstrap";
        cache_node_ids -> bool = |this| this.params.bool_or("cacheNodeIds", false) =>
            "cacheNodeIds";
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10) =>
            "checkpointInterval";
        feature_subset_strategy -> &str = |this| this.params.string_or("featureSubsetStrategy",
            "auto") => "featureSubsetStrategy";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        impurity -> &str = |this| this.params.string_or("impurity", "variance") => "impurity";
        label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
        set_leaf_col(value: &str) => "leafCol", set_param_string;
        leaf_col -> &str = |this| this.params.string_or("leafCol", "");
        max_bins -> i32 = |this| this.params.int_or("maxBins", 32) => "maxBins";
        max_depth -> i32 = |this| this.params.int_or("maxDepth", 5) => "maxDepth";
        max_memory_in_mb -> i32 = |this| this.params.int_or("maxMemoryInMB", 256) =>
            "maxMemoryInMB";
        min_info_gain -> f64 = |this| this.params.double_or("minInfoGain", 0.0) => "minInfoGain";
        min_instances_per_node -> i32 = |this| this.params.int_or("minInstancesPerNode", 1) =>
            "minInstancesPerNode";
        min_weight_fraction_per_node -> f64 = |this| this.params.double_or(
            "minWeightFractionPerNode", 0.0) => "minWeightFractionPerNode";
        num_trees -> i32 = |this| this.params.int_or("numTrees", 20) => "numTrees";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.regression.RandomForestRegressionModel")) => "seed";
        subsampling_rate -> f64 = |this| this.params.double_or("subsamplingRate", 1.0) =>
            "subsamplingRate";
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

impl Transformer for RandomForestRegressionModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for RandomForestRegressionModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// K-means clustering using Spark's JVM implementation and ML vector features.
///
/// Supports Euclidean and cosine distance. Unset parameters use Spark's defaults. Without a
/// seed, the initialization uses Spark's default seed (the hash of the class name), as with
/// PySpark, so fits are reproducible either way; set a seed to start from other centers.
#[derive(Debug, Clone)]
pub struct KMeans {
    operator: MlOperator,
    params: Params,
}

impl KMeans {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.clustering.KMeans",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<KMeansModel> {
        let cached = CachedModel::from_info(&info, session, "KMeans")?;
        Ok(KMeansModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.clustering.KMeansModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
            training_summary: None,
        })
    }
}

ml_params! {
    KMeans {
        set_distance_measure(value: &str) => "distanceMeasure", set_param_string;
        distance_measure -> &str = |this| this.params.string_or("distanceMeasure", "euclidean");
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_init_mode(value: &str) => "initMode", set_param_string;
        init_mode -> &str = |this| this.params.string_or("initMode", "k-means||");
        set_init_steps(value: i32) => "initSteps", set_param_int32;
        init_steps -> i32 = |this| this.params.int_or("initSteps", 2);
        set_intermediate_storage_level(value: &str) => "intermediateStorageLevel", set_param_string;
        intermediate_storage_level -> &str = |this| this.params.string_or(
            "intermediateStorageLevel", "MEMORY_AND_DISK");
        set_k(value: i32) => "k", set_param_int32;
        k -> i32 = |this| this.params.int_or("k", 2);
        set_max_block_size_in_mb(value: f64) => "maxBlockSizeInMB", set_param_double;
        max_block_size_in_mb -> f64 = |this| this.params.double_or("maxBlockSizeInMB", 0.0);
        set_max_iter(value: i32) => "maxIter", set_param_int32;
        max_iter -> i32 = |this| this.params.int_or("maxIter", 20);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.clustering.KMeans"));
        set_solver(value: &str) => "solver", set_param_string;
        solver -> &str = |this| this.params.string_or("solver", "auto");
        set_tol(value: f64) => "tol", set_param_double;
        tol -> f64 = |this| this.params.double_or("tol", 0.0001);
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

impl Default for KMeans {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for KMeans {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(KMeans::fit(self, df)?))
    }
}

/// A K-means model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the cached server model until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct KMeansModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
    /// The summary of the model of its training dataset, of a model that this client fitted.
    training_summary: Option<MlTarget>,
}

ml_params! {
    KMeansModel {
        distance_measure -> &str = |this| this.params.string_or("distanceMeasure", "euclidean") =>
            "distanceMeasure";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        init_mode -> &str = |this| this.params.string_or("initMode", "k-means||") => "initMode";
        init_steps -> i32 = |this| this.params.int_or("initSteps", 2) => "initSteps";
        intermediate_storage_level -> &str = |this| this.params.string_or(
            "intermediateStorageLevel", "MEMORY_AND_DISK") => "intermediateStorageLevel";
        k -> i32 = |this| this.params.int_or("k", 2) => "k";
        max_block_size_in_mb -> f64 = |this| this.params.double_or("maxBlockSizeInMB", 0.0) =>
            "maxBlockSizeInMB";
        max_iter -> i32 = |this| this.params.int_or("maxIter", 20) => "maxIter";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.clustering.KMeansModel")) => "seed";
        solver -> &str = |this| this.params.string_or("solver", "auto") => "solver";
        tol -> f64 = |this| this.params.double_or("tol", 0.0001) => "tol";
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

impl Transformer for KMeansModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for KMeansModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Hierarchical bisecting K-means using Spark's JVM implementation and ML vectors.
///
/// Starts with one cluster and repeatedly splits divisible leaf clusters until there
/// are k leaves or no further splits are possible. Supports Euclidean and cosine distance,
/// instance weights and seeded initialization. Unset parameters use Spark's defaults.
#[derive(Debug, Clone)]
pub struct BisectingKMeans {
    operator: MlOperator,
    params: Params,
}

impl BisectingKMeans {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.clustering.BisectingKMeans",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<BisectingKMeansModel> {
        let cached = CachedModel::from_info(&info, session, "BisectingKMeans")?;
        Ok(BisectingKMeansModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.clustering.BisectingKMeansModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
            training_summary: None,
        })
    }
}

ml_params! {
    BisectingKMeans {
        set_distance_measure(value: &str) => "distanceMeasure", set_param_string;
        distance_measure -> &str = |this| this.params.string_or("distanceMeasure", "euclidean");
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_k(value: i32) => "k", set_param_int32;
        k -> i32 = |this| this.params.int_or("k", 4);
        set_max_iter(value: i32) => "maxIter", set_param_int32;
        max_iter -> i32 = |this| this.params.int_or("maxIter", 20);
        set_min_divisible_cluster_size(value: f64) => "minDivisibleClusterSize", set_param_double;
        min_divisible_cluster_size -> f64 = |this| this.params.double_or("minDivisibleClusterSize",
            1.0);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.clustering.BisectingKMeans"));
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

impl Default for BisectingKMeans {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for BisectingKMeans {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(BisectingKMeans::fit(self, df)?))
    }
}

/// A bisecting K-means model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the cached server model until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct BisectingKMeansModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
    /// The summary of the model of its training dataset, of a model that this client fitted.
    training_summary: Option<MlTarget>,
}

ml_params! {
    BisectingKMeansModel {
        distance_measure -> &str = |this| this.params.string_or("distanceMeasure", "euclidean") =>
            "distanceMeasure";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        k -> i32 = |this| this.params.int_or("k", 4) => "k";
        max_iter -> i32 = |this| this.params.int_or("maxIter", 20) => "maxIter";
        min_divisible_cluster_size -> f64 = |this| this.params.double_or("minDivisibleClusterSize",
            1.0) => "minDivisibleClusterSize";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.clustering.BisectingKMeansModel")) => "seed";
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

impl Transformer for BisectingKMeansModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for BisectingKMeansModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Gaussian mixture clustering fitted by Spark's JVM implementation using ML vectors.
///
/// Learns k multivariate Gaussian components with expectation maximization. Transforms
/// produce cluster assignments and component probabilities. Unset parameters use Spark's defaults.
#[derive(Debug, Clone)]
pub struct GaussianMixture {
    operator: MlOperator,
    params: Params,
}

impl GaussianMixture {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.clustering.GaussianMixture",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<GaussianMixtureModel> {
        let cached = CachedModel::from_info(&info, session, "GaussianMixture")?;
        Ok(GaussianMixtureModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.clustering.GaussianMixtureModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
            training_summary: None,
        })
    }
}

ml_params! {
    GaussianMixture {
        set_aggregation_depth(value: i32) => "aggregationDepth", set_param_int32;
        aggregation_depth -> i32 = |this| this.params.int_or("aggregationDepth", 2);
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_k(value: i32) => "k", set_param_int32;
        k -> i32 = |this| this.params.int_or("k", 2);
        set_max_iter(value: i32) => "maxIter", set_param_int32;
        max_iter -> i32 = |this| this.params.int_or("maxIter", 100);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.clustering.GaussianMixture"));
        set_tol(value: f64) => "tol", set_param_double;
        tol -> f64 = |this| this.params.double_or("tol", 0.01);
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

impl Default for GaussianMixture {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GaussianMixture {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(GaussianMixture::fit(self, df)?))
    }
}

/// A Gaussian mixture model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the cached server model until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct GaussianMixtureModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
    /// The summary of the model of its training dataset, of a model that this client fitted.
    training_summary: Option<MlTarget>,
}

ml_params! {
    GaussianMixtureModel {
        aggregation_depth -> i32 = |this| this.params.int_or("aggregationDepth", 2) =>
            "aggregationDepth";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        k -> i32 = |this| this.params.int_or("k", 2) => "k";
        max_iter -> i32 = |this| this.params.int_or("maxIter", 100) => "maxIter";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.clustering.GaussianMixtureModel")) => "seed";
        tol -> f64 = |this| this.params.double_or("tol", 0.01) => "tol";
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

impl Transformer for GaussianMixtureModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for GaussianMixtureModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Latent Dirichlet Allocation, fitted by Spark Connect using the online or EM optimizer.
#[derive(Debug, Clone)]
pub struct LDA {
    operator: MlOperator,
    params: Params,
}

impl LDA {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.clustering.LDA",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }

    fn model_from_info(
        &self,
        info: proto::ml_command_result::MlOperatorInfo,
        session: &SparkSession,
    ) -> spark_connect_core::error::Result<LDAModel> {
        let cached = CachedModel::from_info(&info, session, "LDA")?;
        let name = if self.optimizer() == "em" {
            "org.apache.spark.ml.clustering.DistributedLDAModel"
        } else {
            "org.apache.spark.ml.clustering.LocalLDAModel"
        };
        Ok(LDAModel {
            operator: MlOperator::with_uid(
                name,
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached,
        })
    }
}

ml_params! {
    LDA {
        set_checkpoint_interval(value: i32) => "checkpointInterval", set_param_int32;
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10);
        set_doc_concentration(value: impl IntoIterator<Item = f64>) => "docConcentration",
            set_param_double_array;
        doc_concentration -> Option<Vec<f64>> = |this| this.params.double_array_opt(
            "docConcentration");
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_k(value: i32) => "k", set_param_int32;
        k -> i32 = |this| this.params.int_or("k", 10);
        set_keep_last_checkpoint(value: bool) => "keepLastCheckpoint", set_param_bool;
        keep_last_checkpoint -> bool = |this| this.params.bool_or("keepLastCheckpoint", true);
        set_learning_decay(value: f64) => "learningDecay", set_param_double;
        learning_decay -> f64 = |this| this.params.double_or("learningDecay", 0.51);
        set_learning_offset(value: f64) => "learningOffset", set_param_double;
        learning_offset -> f64 = |this| this.params.double_or("learningOffset", 1024.0);
        set_max_iter(value: i32) => "maxIter", set_param_int32;
        max_iter -> i32 = |this| this.params.int_or("maxIter", 20);
        set_optimize_doc_concentration(value: bool) => "optimizeDocConcentration", set_param_bool;
        optimize_doc_concentration -> bool = |this| this.params.bool_or("optimizeDocConcentration",
            true);
        set_optimizer(value: &str) => "optimizer", set_param_string;
        optimizer -> &str = |this| this.params.string_or("optimizer", "online");
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.clustering.LDA"));
        set_subsampling_rate(value: f64) => "subsamplingRate", set_param_double;
        subsampling_rate -> f64 = |this| this.params.double_or("subsamplingRate", 0.05);
        set_topic_concentration(value: f64) => "topicConcentration", set_param_double;
        topic_concentration -> Option<f64> = |this| this.params.double_opt("topicConcentration");
        set_topic_distribution_col(value: &str) => "topicDistributionCol", set_param_string;
        topic_distribution_col -> &str = |this| this.params.string_or("topicDistributionCol",
            "topicDistribution");
    }
}

impl Default for LDA {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LDA {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(LDA::fit(self, df)?))
    }
}

/// A LDA model fitted by Spark Connect.
///
/// Obtain it through [`Estimator::fit`]. The model, its clones and transformed
/// DataFrames retain the cached server model until the last owner is dropped.
#[derive(Debug, Clone)]
pub struct LDAModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

ml_params! {
    LDAModel {
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10) =>
            "checkpointInterval";
        doc_concentration -> Option<Vec<f64>> = |this| this.params.double_array_opt(
            "docConcentration") => "docConcentration";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        k -> i32 = |this| this.params.int_or("k", 10) => "k";
        keep_last_checkpoint -> bool = |this| this.params.bool_or("keepLastCheckpoint", true) =>
            "keepLastCheckpoint";
        learning_decay -> f64 = |this| this.params.double_or("learningDecay", 0.51) =>
            "learningDecay";
        learning_offset -> f64 = |this| this.params.double_or("learningOffset", 1024.0) =>
            "learningOffset";
        max_iter -> i32 = |this| this.params.int_or("maxIter", 20) => "maxIter";
        optimize_doc_concentration -> bool = |this| this.params.bool_or("optimizeDocConcentration",
            true) => "optimizeDocConcentration";
        optimizer -> &str = |this| this.params.string_or("optimizer", "online") => "optimizer";
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            &this.operator.name));
        subsampling_rate -> f64 = |this| this.params.double_or("subsamplingRate", 0.05) =>
            "subsamplingRate";
        topic_concentration -> Option<f64> = |this| this.params.double_opt("topicConcentration") =>
            "topicConcentration";
        set_topic_distribution_col(value: &str) => "topicDistributionCol", set_param_string;
        topic_distribution_col -> &str = |this| this.params.string_or("topicDistributionCol",
            "topicDistribution");
    }
}

impl Transformer for LDAModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(input, &self.params)
    }

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}

impl Model for LDAModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// Power Iteration Clustering of a graph's nonnegative edge similarities.
///
/// Like PySpark's class, this is not an estimator or transformer. Use
/// [`PowerIterationClustering::assign_clusters`] to obtain `(id, cluster)` assignments.
#[derive(Debug, Clone)]
pub struct PowerIterationClustering {
    /// The operator that saves the algorithm: the server's `PowerIterationClusteringWrapper`, of
    /// the uid of the algorithm, as PySpark writes it.
    operator: MlOperator,
    params: Params,
}

impl Default for PowerIterationClustering {
    fn default() -> Self {
        let uid = MlOperator::new(
            "org.apache.spark.ml.clustering.PowerIterationClustering",
            OperatorType::Transformer,
        )
        .uid;
        Self {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.clustering.PowerIterationClusteringWrapper",
                &uid,
                OperatorType::Transformer,
            ),
            params: Params::new(),
        }
    }
}

impl PowerIterationClustering {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn params(&self) -> &Params {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    /// Build the lazy assignment relation using Spark Connect's ML helper. The input plan
    /// is retained so cached models or other resources used by the graph remain alive.
    pub fn assign_clusters(
        &self,
        dataset: &DataFrame,
    ) -> spark_connect_core::error::Result<DataFrame> {
        use proto::expression::literal::LiteralType as L;
        use proto::fetch::method::args::ArgsType;

        let mut params = self.params.to_proto().params;
        let defaults = [
            ("k", L::Integer(2)),
            ("maxIter", L::Integer(20)),
            ("initMode", L::String("random".into())),
            ("srcCol", L::String("src".into())),
            ("dstCol", L::String("dst".into())),
            ("weightCol", L::String(String::new())),
        ];
        let mut args = vec![proto::fetch::method::Args {
            args_type: Some(ArgsType::Input(dataset.relation()?)),
        }];
        for (name, default) in defaults {
            // Preserve explicit nulls and integer widths, just as PySpark's helper call does.
            let literal = params
                .remove(name)
                .unwrap_or_else(|| proto::expression::Literal {
                    literal_type: Some(default),
                    ..Default::default()
                });
            args.push(proto::fetch::method::Args {
                args_type: Some(ArgsType::Param(literal)),
            });
        }
        let relation = proto::Relation {
            rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                proto::MlRelation {
                    ml_type: Some(proto::ml_relation::MlType::Fetch(proto::Fetch {
                        obj_ref: Some(proto::ObjectRef {
                            id: "______ML_CONNECT_HELPER______".into(),
                        }),
                        methods: vec![proto::fetch::Method {
                            method: "powerIterationClusteringAssignClusters".into(),
                            args,
                        }],
                    })),
                    ..Default::default()
                },
            ))),
            ..Default::default()
        };
        Ok(DataFrame::new(
            dataset.session.clone(),
            LogicalPlan::MlTransform {
                ml_relation: relation,
                input: dataset.plan.clone(),
                owner: None,
            },
        ))
    }
}

ml_params! {
    PowerIterationClustering {
        set_dst_col(value: &str) => "dstCol", set_param_string;
        dst_col -> &str = |this| this.params.string_or("dstCol", "dst");
        set_init_mode(value: &str) => "initMode", set_param_string;
        init_mode -> &str = |this| this.params.string_or("initMode", "random");
        set_k(value: i32) => "k", set_param_int32;
        k -> i32 = |this| this.params.int_or("k", 2);
        set_max_iter(value: i32) => "maxIter", set_param_int32;
        max_iter -> i32 = |this| this.params.int_or("maxIter", 20);
        set_src_col(value: &str) => "srcCol", set_param_string;
        src_col -> &str = |this| this.params.string_or("srcCol", "src");
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_string_hash_matches_java_hash_code() {
        // Values of `String.hashCode` in the JVM, including a negative one.
        assert_eq!(
            java_string_hash("org.apache.spark.ml.classification.RandomForestClassifier"),
            207_336_481
        );
        assert_eq!(java_string_hash("hello"), 99_162_322);
        assert_eq!(java_string_hash("polygenelubricants"), i64::from(i32::MIN));
    }

    // Used only by the tests below (the non-test build does not reference it).
    use crate::session::SparkSession;

    /// An evaluator outside this crate, which implements only the methods that were required
    /// before `params_mut`: it must still compile, and it exposes no params to mutate.
    struct ExternalEvaluator {
        operator: MlOperator,
        params: Params,
    }

    impl Evaluator for ExternalEvaluator {
        fn operator(&self) -> &MlOperator {
            &self.operator
        }

        fn params(&self) -> &Params {
            &self.params
        }

        fn evaluate(&self, _df: &DataFrame) -> spark_connect_core::error::Result<f64> {
            Ok(0.0)
        }
    }

    #[test]
    fn evaluator_params_mut_is_optional_for_external_implementations() {
        let mut evaluator = ExternalEvaluator {
            operator: MlOperator::new("external.Evaluator", OperatorType::Evaluator),
            params: Params::new(),
        };
        assert!(evaluator.params_mut().is_none());
        assert!(evaluator.params().get_param("labelCol").is_none());
        let mut regression = RegressionEvaluator::new();
        assert!(Evaluator::params_mut(&mut regression).is_some());
    }

    #[test]
    fn test_params_creation() {
        let params = Params::new()
            .set_param_string("inputCol", "features")
            .set_param_string("outputCol", "scaled")
            .set_param_double("mean", 0.5)
            .set_param_bool("withMean", true);

        assert!(params.get_param("inputCol").is_some());
        assert!(params.get_param("outputCol").is_some());
    }

    #[test]
    fn test_params_null_and_empty_arrays() {
        let params = Params::new()
            .set_param_null("weightCol")
            .set_param_string_array("inputCols", &["a".to_string()]);
        let null = params.to_proto();
        assert!(matches!(
            null.params["weightCol"].literal_type,
            Some(proto::expression::literal::LiteralType::Null(_))
        ));
        assert!(!params.has_empty_array());

        let empty = Params::new().set_param_string_array("inputCols", Vec::<String>::new());
        assert!(empty.has_empty_array());
        let empty = Params::new().set_param_double_array("thresholds", []);
        assert!(empty.has_empty_array());
        assert!(!Params::new().set_param_null("x").has_empty_array());
    }

    #[test]
    #[allow(deprecated)]
    fn test_string_array_param_has_the_default_collation() {
        // As PySpark infers the type of a list of strings: a string of the UTF8_BINARY collation.
        let params = Params::new().set_param_string_array("inputCols", &["a".to_string()]);
        let literal = params.to_proto().params["inputCols"].clone();
        let Some(proto::expression::literal::LiteralType::Array(array)) = literal.literal_type
        else {
            panic!("expected an array literal");
        };
        let element_type = array.element_type.expect("element type");
        let Some(proto::data_type::Kind::String(string)) = element_type.kind else {
            panic!("expected a string element type");
        };
        assert_eq!(string.collation, "UTF8_BINARY");
    }

    #[test]
    fn test_matrix_params_are_the_literals_of_pyspark() {
        // The bytes of `serialize_param` of PySpark 4.3.0 (`pyspark.ml.connect.serialize`) of
        // `DenseMatrix(1, 2, [0.0, -1.0])` and of
        // `SparseMatrix(2, 2, [0, 1, 2], [0, 1], [1.0, 2.0], True)`.
        use base64::{engine::general_purpose::STANDARD, Engine};
        use prost::Message;

        let dense =
            "wgFlCi66ASsKA3VkdBIkb3JnLmFwYWNoZS5zcGFyay5tbC5saW5hbGcuTWF0cml4VURUEgIgARICMA\
            ESAjACEgQKAgoAEgQKAgoAEhfKARQqEgoQAAAAAAAAAAAAAAAAAADwvxICGAA=";
        let sparse =
            "wgFwCi66ASsKA3VkdBIkb3JnLmFwYWNoZS5zcGFyay5tbC5saW5hbGcuTWF0cml4VURUEgIgABICM\
            AISAjACEgrKAQcSBQoDAAECEgnKAQYSBAoCAAESF8oBFCoSChAAAAAAAADwPwAAAAAAAABAEgIYAQ==";
        for (matrix, expected) in [
            (
                MlMatrix::Dense {
                    num_rows: 1,
                    num_cols: 2,
                    values: vec![0.0, -1.0],
                    is_transposed: false,
                },
                dense,
            ),
            (
                MlMatrix::Sparse {
                    num_rows: 2,
                    num_cols: 2,
                    col_ptrs: vec![0, 1, 2],
                    row_indices: vec![0, 1],
                    values: vec![1.0, 2.0],
                    is_transposed: true,
                },
                sparse,
            ),
        ] {
            let params = Params::new().set_param_matrix("lowerBoundsOnCoefficients", matrix);
            let literal = params.to_proto().params["lowerBoundsOnCoefficients"].clone();
            assert_eq!(STANDARD.encode(literal.encode_to_vec()), expected);
        }
    }

    #[test]
    fn test_ml_operator_creation() {
        let op = MlOperator::new(
            "org.apache.spark.ml.feature.StandardScaler",
            OperatorType::Estimator,
        );
        assert_eq!(op.name, "org.apache.spark.ml.feature.StandardScaler");
        assert_eq!(op.op_type, OperatorType::Estimator);
        assert!(!op.uid.is_empty());
    }

    #[test]
    fn test_standard_scaler_creation() {
        let scaler = StandardScaler::new()
            .set_input_col("my_features")
            .set_output_col("my_scaled");

        assert_eq!(scaler.input_col(), Some("my_features"));
        assert_eq!(scaler.output_col(), "my_scaled");
    }

    #[test]
    fn test_standard_scaler_with_mean() {
        // As Spark's default, the features are not centered unless the setter says so.
        let scaler = StandardScaler::new();
        assert!(!scaler.with_mean());
        assert!(scaler.params().get_param("withMean").is_none());

        let scaler = StandardScaler::new().set_with_mean(true);
        assert!(scaler.with_mean());
        let bool_param = scaler.params().get_param("withMean").unwrap();
        let Some(proto::expression::ExprType::Literal(literal)) = &bool_param.expr_type else {
            panic!("withMean is not a literal");
        };
        assert_eq!(
            literal.literal_type,
            Some(proto::expression::literal::LiteralType::Boolean(true))
        );

        let scaler = scaler.set_with_mean(false);
        assert!(!scaler.with_mean());
    }

    #[test]
    fn test_operator_to_proto() {
        let op = MlOperator::new("test.Operator", OperatorType::Transformer);
        let proto = op.to_proto();
        assert_eq!(proto.name, "test.Operator");
        assert_eq!(proto.r#type, OperatorType::Transformer.to_proto() as i32);
    }

    #[test]
    fn test_vector_assembler_creation() {
        let assembler = VectorAssembler::new()
            .set_input_cols(vec!["col1", "col2", "col3"])
            .set_output_col("vector_col");

        assert_eq!(assembler.input_cols().len(), 3);
        assert_eq!(assembler.output_col(), "vector_col");
    }

    #[test]
    fn test_vector_assembler_operator_name() {
        let assembler = VectorAssembler::new();
        let op = assembler.operator();
        assert_eq!(op.name, "org.apache.spark.ml.feature.VectorAssembler");
        assert_eq!(op.op_type, OperatorType::Transformer);
    }

    #[test]
    fn test_string_indexer_creation() {
        let indexer = StringIndexer::new()
            .set_input_col("category")
            .set_output_col("category_index");

        assert_eq!(indexer.input_col(), Some("category"));
        assert_eq!(indexer.output_col(), "category_index");
    }

    #[test]
    fn test_string_indexer_operator_name() {
        let indexer = StringIndexer::new();
        let op = indexer.operator();
        assert_eq!(op.name, "org.apache.spark.ml.feature.StringIndexer");
        assert_eq!(op.op_type, OperatorType::Estimator);
    }

    #[test]
    fn test_max_abs_scaler_creation() {
        let scaler = MaxAbsScaler::new()
            .set_input_col("features")
            .set_output_col("scaled");

        assert_eq!(scaler.input_col(), Some("features"));
        assert_eq!(scaler.output_col(), "scaled");
    }

    #[test]
    fn test_max_abs_scaler_operator_name() {
        let scaler = MaxAbsScaler::new();
        let op = scaler.operator();
        assert_eq!(op.name, "org.apache.spark.ml.feature.MaxAbsScaler");
        assert_eq!(op.op_type, OperatorType::Estimator);
    }

    #[test]
    fn test_logistic_regression_creation() {
        let lr = LogisticRegression::new()
            .set_features_col("features")
            .set_label_col("label")
            .set_prediction_col("pred")
            .set_max_iter(50);

        assert_eq!(lr.features_col(), "features");
        assert_eq!(lr.label_col(), "label");
        assert_eq!(lr.prediction_col(), "pred");
        assert_eq!(lr.max_iter(), 50);
    }

    #[test]
    fn test_logistic_regression_operator_name() {
        let lr = LogisticRegression::new();
        let op = lr.operator();
        assert_eq!(
            op.name,
            "org.apache.spark.ml.classification.LogisticRegression"
        );
        assert_eq!(op.op_type, OperatorType::Estimator);
    }

    #[test]
    fn test_regression_evaluator_creation() {
        let eval = RegressionEvaluator::new()
            .set_label_col("true_label")
            .set_prediction_col("predicted")
            .set_metric_name("r2");

        assert_eq!(eval.label_col(), "true_label");
        assert_eq!(eval.prediction_col(), "predicted");
        assert_eq!(eval.metric_name(), "r2");
    }

    #[test]
    fn test_regression_evaluator_operator_name() {
        let eval = RegressionEvaluator::new();
        let op = eval.operator();
        assert_eq!(
            op.name,
            "org.apache.spark.ml.evaluation.RegressionEvaluator"
        );
        assert_eq!(op.op_type, OperatorType::Evaluator);
    }

    #[test]
    fn test_binary_classification_evaluator_creation() {
        let eval = BinaryClassificationEvaluator::new()
            .set_label_col("label")
            .set_raw_prediction_col("score")
            .set_metric_name("areaUnderPR");

        assert_eq!(eval.label_col(), "label");
        assert_eq!(eval.raw_prediction_col(), "score");
        assert_eq!(eval.metric_name(), "areaUnderPR");
    }

    #[test]
    fn test_binary_classification_evaluator_operator_name() {
        let eval = BinaryClassificationEvaluator::new();
        let op = eval.operator();
        assert_eq!(
            op.name,
            "org.apache.spark.ml.evaluation.BinaryClassificationEvaluator"
        );
        assert_eq!(op.op_type, OperatorType::Evaluator);
    }

    #[test]
    fn test_pipeline_creation() {
        let assembler = VectorAssembler::new();
        let svc = LinearSVC::new();
        let pipeline = Pipeline::new().set_stages(vec![
            PipelineStage::transformer(assembler.clone()),
            PipelineStage::estimator(svc.clone()),
        ]);
        assert_eq!(pipeline.stages().len(), 2);
        assert_eq!(pipeline.stages()[0].uid(), assembler.operator().uid);
        assert!(!pipeline.stages()[0].is_estimator());
        assert_eq!(pipeline.stages()[1].uid(), svc.operator().uid);
        assert!(pipeline.stages()[1].is_estimator());
        assert!(pipeline.uid().starts_with("Pipeline_"));
    }

    #[test]
    fn test_pipeline_operator_name() {
        let pipeline = Pipeline::new();
        let op = pipeline.operator();
        assert_eq!(op.name, "org.apache.spark.ml.Pipeline");
        assert_eq!(op.op_type, OperatorType::Estimator);
    }

    #[test]
    fn test_all_transformer_types() {
        let transformers: Vec<Box<dyn Transformer>> = vec![
            Box::new(VectorAssembler::new()),
            Box::new(StringIndexerModel {
                operator: MlOperator::new("test", OperatorType::Model),
                params: Params::new(),
                cached: cached_model(&offline_session(), "indexer-1"),
            }),
            Box::new(MaxAbsScalerModel {
                operator: MlOperator::new("test", OperatorType::Model),
                params: Params::new(),
                cached: cached_model(&offline_session(), "max-abs-1"),
            }),
        ];

        for transformer in transformers {
            assert!(transformer.params().params.is_empty());
        }
    }

    #[test]
    fn test_all_estimator_types() {
        let estimators: Vec<(&str, Box<dyn Estimator>)> = vec![
            ("StringIndexer", Box::new(StringIndexer::new())),
            ("MaxAbsScaler", Box::new(MaxAbsScaler::new())),
            ("LogisticRegression", Box::new(LogisticRegression::new())),
            ("Pipeline", Box::new(Pipeline::new())),
        ];

        for (name, estimator) in estimators {
            assert_eq!(
                estimator.operator().op_type,
                OperatorType::Estimator,
                "Failed for {}",
                name
            );
        }
    }

    #[test]
    fn test_all_evaluator_types() {
        let evaluators: Vec<(&str, Box<dyn Evaluator>)> = vec![
            ("RegressionEvaluator", Box::new(RegressionEvaluator::new())),
            (
                "BinaryClassificationEvaluator",
                Box::new(BinaryClassificationEvaluator::new()),
            ),
        ];

        for (name, evaluator) in evaluators {
            assert_eq!(
                evaluator.operator().op_type,
                OperatorType::Evaluator,
                "Failed for {}",
                name
            );
        }
    }

    // gRPC connects lazily, so a session builds offline; fit/transform on the ML
    // operators only build plans (no RPC until collect), so these run server-free.
    fn offline_session() -> SparkSession {
        SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session")
    }

    /// The model that a fit of the server returns as the object `id`, cached for `s`.
    fn cached_model(s: &SparkSession, id: &str) -> CachedModel {
        let info = proto::ml_command_result::MlOperatorInfo {
            r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef { id: id.to_string() },
            )),
            uid: Some("uid".to_string()),
            ..Default::default()
        };
        CachedModel::from_info(&info, s, "test").expect("cached model")
    }

    /// The object reference of the model that the transform of `df` refers to.
    fn transform_model_ref(df: &DataFrame) -> String {
        let LogicalPlan::MlTransform { ml_relation, .. } = df.plan.as_ref() else {
            panic!("expected an ML transform plan");
        };
        let Some(proto::relation::RelType::MlRelation(ml)) = &ml_relation.rel_type else {
            panic!("expected an ML relation");
        };
        let Some(proto::ml_relation::MlType::Transform(transform)) = &ml.ml_type else {
            panic!("expected a transform");
        };
        match &transform.operator {
            Some(proto::ml_relation::transform::Operator::ObjRef(reference)) => {
                reference.id.clone()
            }
            other => panic!("expected the model reference, found {other:?}"),
        }
    }

    #[test]
    fn list_and_int_params_use_the_jvm_literal_types() {
        use proto::expression::literal::LiteralType as L;
        // An Array[String] param (inputCols) is an array literal of string
        // elements, as PySpark sends a list; the server rejects a string ("not an array").
        let va = VectorAssembler::new().set_input_cols(vec!["a", "b"]);
        let (name, params, expected) = ("inputCols", va.params(), vec!["a", "b"]);
        let Some(L::Array(array)) = params.to_proto().params[name].literal_type.clone() else {
            panic!("{name} is not an array literal");
        };
        #[allow(deprecated)]
        let element_type = array.element_type.as_ref().and_then(|t| t.kind.clone());
        assert!(
            matches!(element_type, Some(proto::data_type::Kind::String(_))),
            "{name} element type"
        );
        let elements: Vec<_> = array
            .elements
            .iter()
            .map(|e| e.literal_type.clone())
            .collect();
        let expected: Vec<_> = expected
            .iter()
            .map(|s| Some(L::String(s.to_string())))
            .collect();
        assert_eq!(elements, expected, "{name}");

        // Int params are Integer literals when the value fits 32 bits, as PySpark sends a Python
        // int; the JVM param is an IntParam (maxIter).
        let lr = LogisticRegression::new().set_max_iter(5);
        let params = lr.params().to_proto().params;
        assert_eq!(params["maxIter"].literal_type, Some(L::Integer(5)));
    }

    #[test]
    fn params_int_and_proto_roundtrip() {
        let p = Params::new()
            .set_param_int("maxIter", 7)
            .set_param_string("inputCol", "x");
        assert!(p.get_param("maxIter").is_some());
        assert!(p.get_param("missing").is_none());

        let proto_p = p.to_proto();
        assert!(proto_p.params.contains_key("maxIter"));
        let back = Params::from_proto(&proto_p);
        assert!(back.get_param("inputCol").is_some());
        assert!(back.get_param("maxIter").is_some());
    }

    #[test]
    fn ml_operator_with_uid_and_proto_roundtrip() {
        let op = MlOperator::with_uid("test.Op", "uid-123", OperatorType::Model);
        assert_eq!(op.uid, "uid-123");
        let proto_op = op.to_proto();
        assert_eq!(proto_op.uid, "uid-123");
        let back = MlOperator::from_proto(&proto_op);
        assert_eq!(back.name, "test.Op");
        assert_eq!(back.op_type, OperatorType::Model);
    }

    #[test]
    fn operator_type_proto_roundtrip_all_variants() {
        for t in [
            OperatorType::Estimator,
            OperatorType::Transformer,
            OperatorType::Evaluator,
            OperatorType::Model,
        ] {
            assert_eq!(OperatorType::from_proto(t.to_proto() as i32), t);
        }
    }

    #[test]
    fn transform_and_fit_build_ml_transform_plans() {
        let s = offline_session();
        let df = s.range(3).unwrap();

        // A Transformer builds an MlTransform relation plan (no RPC).
        let mut va = VectorAssembler::new()
            .set_input_cols(vec!["id"])
            .set_output_col("v");
        let out = va.transform(&df).unwrap();
        assert!(matches!(out.plan.as_ref(), LogicalPlan::MlTransform { .. }));

        // A fitted StandardScaler transforms through the model that the server cached: the
        // transform refers to its object reference, as PySpark's transform of a fitted
        // model does. The fit itself needs a server, so the cached model is built here.
        let ss = StandardScaler::new()
            .set_input_col("features")
            .set_output_col("scaled");
        let mut model = StandardScalerModel {
            operator: MlOperator::new("test", OperatorType::Model),
            params: ss.params().clone(),
            cached: cached_model(&s, "scaler-1"),
        };
        let scaled = model.transform(&df).unwrap();
        assert!(matches!(
            scaled.plan.as_ref(),
            LogicalPlan::MlTransform { .. }
        ));
        assert_eq!(transform_model_ref(&scaled), "scaler-1");
        let _cloned = model.clone_box();
    }

    #[test]
    fn multiclass_evaluator_creation_and_getters() {
        let e = MulticlassClassificationEvaluator::new()
            .set_label_col("y")
            .set_prediction_col("p")
            .set_metric_name("accuracy");
        assert_eq!(e.label_col(), "y");
        assert_eq!(e.prediction_col(), "p");
        assert_eq!(e.metric_name(), "accuracy");
        assert_eq!(e.operator().op_type, OperatorType::Evaluator);
        assert_eq!(
            e.operator().name,
            "org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator"
        );
        assert_eq!(
            MulticlassClassificationEvaluator::default().metric_name(),
            "f1"
        );
    }

    #[test]
    fn cross_validator_creation_getters_and_fit_errors() {
        let cv = CrossValidator::new()
            .set_num_folds(5)
            .set_parallelism(2)
            .set_seed(42);
        assert_eq!((cv.num_folds(), cv.parallelism(), cv.seed()), (5, 2, 42));
        assert_eq!(cv.operator().op_type, OperatorType::Estimator);
        assert_eq!(
            cv.operator().name,
            "org.apache.spark.ml.tuning.CrossValidator"
        );
        assert_eq!(CrossValidator::default().num_folds(), 3);
        // A fit of no estimator, or of no param maps (PySpark's pool of no thread), raises before
        // any request, and returns no model.
        let s = offline_session();
        let df = s.range(3).unwrap();
        let error = CrossValidator::new().fit(&df).expect_err("no estimator");
        assert!(error.to_string().contains("no estimator"), "{error}");
        let mut cv = CrossValidator::new()
            .set_estimator(LinearSVC::new())
            .set_evaluator(BinaryClassificationEvaluator::new());
        let error = cv.fit(&df).expect_err("no param maps");
        assert!(
            error
                .to_string()
                .contains("Number of processes must be at least 1"),
            "{error}"
        );
    }

    #[test]
    fn pipelines_of_no_estimator_fit_without_a_request() {
        // As `Pipeline._fit`: the stages after the last estimator (of none, all of them) are
        // kept as they are, in a model of a new uid that transforms by each in order.
        let s = offline_session();
        let df = s.range(3).unwrap();
        let mut empty = Pipeline::new();
        let mut model = empty.fit(&df).unwrap();
        assert!(model.operator().uid.starts_with("PipelineModel_"));
        assert_ne!(model.operator().uid, empty.uid());
        assert_eq!(
            format!("{:?}", model.transform(&df).unwrap().plan()),
            format!("{:?}", df.plan())
        );
        let assembler = VectorAssembler::new().set_input_cols(vec!["id"]);
        let mut only_transformers =
            Pipeline::new().set_stages(vec![PipelineStage::transformer(assembler.clone())]);
        let model = only_transformers.fit_pipeline(&df).unwrap();
        assert_eq!(model.stages().len(), 1);
        assert_eq!(model.stages()[0].uid(), assembler.operator().uid);
    }

    #[test]
    fn offline_estimators_fit_and_models_transform() {
        let s = offline_session();
        let df = s.range(3).unwrap();
        // The models of the server-fitted estimators transform through their cached model;
        // this exercises the offline model wrappers. The fits are tested against a server.
        let mas = MaxAbsScaler::new().set_input_col("f").set_output_col("o");
        assert_eq!(mas.input_col(), Some("f"));
        let mut m = MaxAbsScalerModel {
            operator: MlOperator::new("test", OperatorType::Model),
            params: mas.params().clone(),
            cached: cached_model(&s, "max-abs-1"),
        };
        let out = m.transform(&df).unwrap();
        assert_eq!(transform_model_ref(&out), "max-abs-1");
        let _ = m.clone_box();

        let si = StringIndexer::new().set_input_col("s").set_output_col("si");
        assert_eq!(si.output_col(), "si");
        let mut m = StringIndexerModel {
            operator: MlOperator::new("test", OperatorType::Model),
            params: si.params().clone(),
            cached: cached_model(&s, "indexer-1"),
        };
        let out = m.transform(&df).unwrap();
        assert_eq!(transform_model_ref(&out), "indexer-1");
        let _ = m.clone_box();

        // VectorAssembler transformer.
        let mut va = VectorAssembler::new()
            .set_input_cols(vec!["a", "b"])
            .set_output_col("v");
        assert_eq!(va.input_cols().len(), 2);
        let _ = va.transform(&df).unwrap();
    }

    fn fitted_logistic_regression(session: &SparkSession) -> LogisticRegressionModel {
        let estimator = LogisticRegression::new().set_prediction_col("pred");
        estimator
            .model_from_info(
                proto::ml_command_result::MlOperatorInfo {
                    r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                        proto::ObjectRef {
                            id: "cached-model-id".to_string(),
                        },
                    )),
                    uid: Some("operator-uid".to_string()),
                    ..Default::default()
                },
                session,
            )
            .unwrap()
    }

    #[test]
    fn logistic_regression_transform_uses_the_fitted_object_reference() {
        let session = offline_session();
        let mut model = fitted_logistic_regression(&session);
        assert_eq!(model.operator.uid, "operator-uid");
        assert_eq!(model.operator.op_type, OperatorType::Model);
        let data = session.range(3).unwrap();
        let transformed = model.transform(&data).unwrap();
        let relation = transformed.relation().unwrap();
        let Some(proto::relation::RelType::MlRelation(ml)) = relation.rel_type else {
            panic!("expected an ML relation");
        };
        let Some(proto::ml_relation::MlType::Transform(transform)) = ml.ml_type else {
            panic!("expected a model transform");
        };
        assert_eq!(
            transform.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "cached-model-id".to_string()
                },
            ))
        );
        assert!(transform
            .params
            .unwrap()
            .params
            .contains_key("predictionCol"));
        assert_eq!(*transform.input.unwrap(), data.relation().unwrap());
        let _ = model.clone_box();
    }

    #[test]
    fn logistic_regression_fit_requires_a_nonempty_model_reference() {
        let estimator = LogisticRegression::new();
        let session = offline_session();
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "operator".to_string(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            let info = proto::ml_command_result::MlOperatorInfo {
                r#type: kind,
                ..Default::default()
            };
            assert!(estimator.model_from_info(info, &session).is_err());
        }
    }

    #[test]
    fn logistic_regression_rejects_a_different_session() {
        let session = offline_session();
        let mut model = fitted_logistic_regression(&session);
        let other = session.new_session();
        assert!(model.transform(&other.range(3).unwrap()).is_err());
    }

    #[test]
    fn linear_svc_parameters_use_the_spark_names_and_literal_types() {
        use proto::expression::literal::LiteralType as L;
        let svc = LinearSVC::new()
            .set_features_col("input")
            .set_label_col("target")
            .set_prediction_col("pred")
            .set_raw_prediction_col("margin")
            .set_max_iter(50)
            .set_reg_param(0.1)
            .set_tol(1e-5)
            .set_fit_intercept(false)
            .set_standardization(false)
            .set_threshold(0.5)
            .set_weight_col("weight")
            .set_aggregation_depth(3)
            .set_max_block_size_in_mb(1.0);
        let params = svc.params().to_proto().params;
        let expected = [
            ("featuresCol", L::String("input".into())),
            ("labelCol", L::String("target".into())),
            ("predictionCol", L::String("pred".into())),
            ("rawPredictionCol", L::String("margin".into())),
            ("maxIter", L::Integer(50)),
            ("regParam", L::Double(0.1)),
            ("tol", L::Double(1e-5)),
            ("fitIntercept", L::Boolean(false)),
            ("standardization", L::Boolean(false)),
            ("threshold", L::Double(0.5)),
            ("weightCol", L::String("weight".into())),
            ("aggregationDepth", L::Integer(3)),
            ("maxBlockSizeInMB", L::Double(1.0)),
        ];
        assert_eq!(params.len(), expected.len());
        for (name, literal) in expected {
            assert_eq!(params[name].literal_type.as_ref(), Some(&literal), "{name}");
        }
    }

    #[test]
    fn linear_svc_model_uses_its_cached_reference_and_fitting_session() {
        let session = offline_session();
        let estimator = LinearSVC::default();
        assert!(estimator.params().to_proto().params.is_empty());
        assert_eq!(estimator.operator().op_type, OperatorType::Estimator);
        let info = proto::ml_command_result::MlOperatorInfo {
            r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef {
                    id: "svc-model-object".into(),
                },
            )),
            ..Default::default()
        };
        let mut model = estimator.model_from_info(info, &session).unwrap();
        assert_eq!(model.operator().uid, estimator.operator().uid);
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.classification.LinearSVCModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(transform) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("expected transform");
        };
        assert_eq!(
            transform.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "svc-model-object".into()
                },
            ))
        );
        assert_eq!(*transform.input.unwrap(), df.relation().unwrap());
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
        assert!(estimator
            .model_from_info(Default::default(), &session)
            .is_err());
    }

    #[test]
    fn linear_regression_model_validates_the_reference_and_retains_its_metadata() {
        let session = offline_session();
        let estimator = LinearRegression::default().set_prediction_col("pred");
        assert_eq!(
            estimator.operator().name,
            "org.apache.spark.ml.regression.LinearRegression"
        );
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "model".into(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo {
                        r#type: kind,
                        ..Default::default()
                    },
                    &session
                )
                .is_err());
        }
        let mut model = estimator
            .model_from_info(
                proto::ml_command_result::MlOperatorInfo {
                    r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                        proto::ObjectRef {
                            id: "trained-model".into(),
                        },
                    )),
                    uid: Some("returned-uid".into()),
                    ..Default::default()
                },
                &session,
            )
            .unwrap();
        assert_eq!(model.operator().uid, "returned-uid");
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.regression.LinearRegressionModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        assert_eq!(
            t.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "trained-model".into()
                }
            ))
        );
        assert_eq!(*t.input.unwrap(), df.relation().unwrap());
        assert!(t.params.unwrap().params.contains_key("predictionCol"));
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
    }

    #[test]
    fn fmclassifier_model_validates_the_reference_and_retains_its_metadata() {
        let session = offline_session();
        let estimator = FMClassifier::default().set_prediction_col("pred");
        assert_eq!(
            estimator.operator().name,
            "org.apache.spark.ml.classification.FMClassifier"
        );
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "model".into(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo {
                        r#type: kind,
                        ..Default::default()
                    },
                    &session
                )
                .is_err());
        }
        let mut model = estimator
            .model_from_info(
                proto::ml_command_result::MlOperatorInfo {
                    r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                        proto::ObjectRef {
                            id: "trained-model".into(),
                        },
                    )),
                    uid: Some("returned-uid".into()),
                    ..Default::default()
                },
                &session,
            )
            .unwrap();
        assert_eq!(model.operator().uid, "returned-uid");
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.classification.FMClassificationModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        assert_eq!(
            t.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "trained-model".into()
                }
            ))
        );
        assert_eq!(*t.input.unwrap(), df.relation().unwrap());
        assert!(t.params.unwrap().params.contains_key("predictionCol"));
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
    }

    #[test]
    fn multilayer_perceptron_classifier_model_validates_the_reference_and_retains_its_metadata() {
        let session = offline_session();
        let estimator = MultilayerPerceptronClassifier::default().set_prediction_col("pred");
        assert_eq!(
            estimator.operator().name,
            "org.apache.spark.ml.classification.MultilayerPerceptronClassifier"
        );
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "model".into(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo {
                        r#type: kind,
                        ..Default::default()
                    },
                    &session
                )
                .is_err());
        }
        let mut model = estimator
            .model_from_info(
                proto::ml_command_result::MlOperatorInfo {
                    r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                        proto::ObjectRef {
                            id: "trained-model".into(),
                        },
                    )),
                    uid: Some("returned-uid".into()),
                    ..Default::default()
                },
                &session,
            )
            .unwrap();
        assert_eq!(model.operator().uid, "returned-uid");
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        assert_eq!(
            t.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "trained-model".into()
                }
            ))
        );
        assert_eq!(*t.input.unwrap(), df.relation().unwrap());
        assert!(t.params.unwrap().params.contains_key("predictionCol"));
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
    }

    #[test]
    fn fmclassifier_defaults_and_configured_getters() {
        let default = FMClassifier::new();
        assert_eq!(default.features_col(), "features");
        assert_eq!(default.label_col(), "label");
        assert_eq!(default.prediction_col(), "prediction");
        assert_eq!(default.probability_col(), "probability");
        assert_eq!(default.raw_prediction_col(), "rawPrediction");
        assert_eq!(default.factor_size(), 8);
        assert!(default.fit_intercept());
        assert!(default.fit_linear());
        assert_eq!(default.reg_param(), 0.0);
        assert_eq!(default.mini_batch_fraction(), 1.0);
        assert_eq!(default.init_std(), 0.01);
        assert_eq!(default.max_iter(), 100);
        assert_eq!(default.step_size(), 1.0);
        assert_eq!(default.tol(), 1e-06);
        assert_eq!(default.solver(), "adamW");
        assert_eq!(default.thresholds(), None);
        assert_eq!(
            default.seed(),
            java_string_hash("org.apache.spark.ml.classification.FMClassifier")
        );
        let configured = FMClassifier::new()
            .set_features_col("input")
            .set_label_col("target")
            .set_prediction_col("pred")
            .set_probability_col("prob")
            .set_raw_prediction_col("scores")
            .set_factor_size(2)
            .set_fit_intercept(false)
            .set_fit_linear(false)
            .set_reg_param(0.1)
            .set_mini_batch_fraction(0.5)
            .set_init_std(0.2)
            .set_max_iter(50)
            .set_step_size(0.3)
            .set_tol(1e-08)
            .set_solver("gd")
            .set_thresholds([1.0, 2.0])
            .set_seed(8589934592);
        assert_eq!(configured.features_col(), "input");
        assert_eq!(configured.label_col(), "target");
        assert_eq!(configured.prediction_col(), "pred");
        assert_eq!(configured.probability_col(), "prob");
        assert_eq!(configured.raw_prediction_col(), "scores");
        assert_eq!(configured.factor_size(), 2);
        assert!(!configured.fit_intercept());
        assert!(!configured.fit_linear());
        assert_eq!(configured.reg_param(), 0.1);
        assert_eq!(configured.mini_batch_fraction(), 0.5);
        assert_eq!(configured.init_std(), 0.2);
        assert_eq!(configured.max_iter(), 50);
        assert_eq!(configured.step_size(), 0.3);
        assert_eq!(configured.tol(), 1e-08);
        assert_eq!(configured.solver(), "gd");
        assert_eq!(configured.thresholds(), Some(vec![1.0, 2.0]));
        assert_eq!(configured.seed(), 8589934592);
    }

    #[test]
    fn multilayer_perceptron_classifier_defaults_and_configured_getters() {
        let default = MultilayerPerceptronClassifier::new();
        assert_eq!(default.features_col(), "features");
        assert_eq!(default.label_col(), "label");
        assert_eq!(default.prediction_col(), "prediction");
        assert_eq!(default.max_iter(), 100);
        assert_eq!(default.tol(), 1e-06);
        assert_eq!(
            default.seed(),
            java_string_hash("org.apache.spark.ml.classification.MultilayerPerceptronClassifier")
        );
        assert_eq!(default.layers(), None);
        assert_eq!(default.block_size(), 128);
        assert_eq!(default.step_size(), 0.03);
        assert_eq!(default.solver(), "l-bfgs");
        assert_eq!(default.initial_weights(), None);
        assert_eq!(default.probability_col(), "probability");
        assert_eq!(default.raw_prediction_col(), "rawPrediction");
        assert_eq!(default.thresholds(), None);
        let configured = MultilayerPerceptronClassifier::new()
            .set_features_col("input")
            .set_label_col("target")
            .set_prediction_col("pred")
            .set_max_iter(50)
            .set_tol(1e-08)
            .set_seed(8589934592)
            .set_layers([2, 3, 2])
            .set_block_size(4)
            .set_step_size(0.1)
            .set_solver("gd")
            .set_initial_weights(MlVector::Dense(vec![
                0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13,
                0.14, 0.15, 0.16,
            ]))
            .set_probability_col("prob")
            .set_raw_prediction_col("scores")
            .set_thresholds([1.0, 2.0]);
        assert_eq!(configured.features_col(), "input");
        assert_eq!(configured.label_col(), "target");
        assert_eq!(configured.prediction_col(), "pred");
        assert_eq!(configured.max_iter(), 50);
        assert_eq!(configured.tol(), 1e-08);
        assert_eq!(configured.seed(), 8589934592);
        assert_eq!(configured.layers(), Some(vec![2, 3, 2]));
        assert_eq!(configured.block_size(), 4);
        assert_eq!(configured.step_size(), 0.1);
        assert_eq!(configured.solver(), "gd");
        assert_eq!(
            configured.initial_weights(),
            Some(MlVector::Dense(vec![
                0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13,
                0.14, 0.15, 0.16
            ]))
        );
        assert_eq!(configured.probability_col(), "prob");
        assert_eq!(configured.raw_prediction_col(), "scores");
        assert_eq!(configured.thresholds(), Some(vec![1.0, 2.0]));
    }

    #[test]
    fn one_vs_rest_rejects_null_client_parameters_before_requests() {
        let session = offline_session();
        let df = session.range(1).unwrap();
        for name in [
            "featuresCol",
            "labelCol",
            "predictionCol",
            "rawPredictionCol",
            "parallelism",
        ] {
            let mut estimator = OneVsRest::new().set_classifier(LinearSVC::new());
            *estimator.params_mut() = Params::new().set_param_null(name);
            assert!(estimator.fit(&df).is_err());
        }
    }

    #[test]
    fn one_vs_rest_defaults_and_configured_getters() {
        let default = OneVsRest::new();
        assert_eq!(default.features_col(), "features");
        assert_eq!(default.label_col(), "label");
        assert_eq!(default.prediction_col(), "prediction");
        assert_eq!(default.raw_prediction_col(), "rawPrediction");
        assert_eq!(default.weight_col(), None);
        assert_eq!(default.parallelism(), 1);
        let configured = OneVsRest::new()
            .set_features_col("input")
            .set_label_col("target")
            .set_prediction_col("pred")
            .set_raw_prediction_col("scores")
            .set_weight_col("weight")
            .set_parallelism(3);
        assert_eq!(configured.features_col(), "input");
        assert_eq!(configured.label_col(), "target");
        assert_eq!(configured.prediction_col(), "pred");
        assert_eq!(configured.raw_prediction_col(), "scores");
        assert_eq!(configured.weight_col(), Some("weight"));
        assert_eq!(configured.parallelism(), 3);
    }

    #[test]
    fn generalized_linear_regression_model_validates_the_reference_and_retains_its_metadata() {
        let session = offline_session();
        let estimator = GeneralizedLinearRegression::default().set_prediction_col("pred");
        assert_eq!(
            estimator.operator().name,
            "org.apache.spark.ml.regression.GeneralizedLinearRegression"
        );
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "model".into(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo {
                        r#type: kind,
                        ..Default::default()
                    },
                    &session
                )
                .is_err());
        }
        let mut model = estimator
            .model_from_info(
                proto::ml_command_result::MlOperatorInfo {
                    r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                        proto::ObjectRef {
                            id: "trained-model".into(),
                        },
                    )),
                    uid: Some("returned-uid".into()),
                    ..Default::default()
                },
                &session,
            )
            .unwrap();
        assert_eq!(model.operator().uid, "returned-uid");
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.regression.GeneralizedLinearRegressionModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        assert_eq!(
            t.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "trained-model".into()
                }
            ))
        );
        assert_eq!(*t.input.unwrap(), df.relation().unwrap());
        assert!(t.params.unwrap().params.contains_key("predictionCol"));
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
    }

    #[test]
    fn generalized_linear_regression_getters_follow_defaults_and_params() {
        let defaults = GeneralizedLinearRegression::default();
        assert_eq!(defaults.label_col(), "label");
        assert_eq!(defaults.features_col(), "features");
        assert_eq!(defaults.prediction_col(), "prediction");
        assert_eq!(defaults.family(), "gaussian");
        assert_eq!(defaults.link(), None);
        assert!(defaults.fit_intercept());
        assert_eq!(defaults.max_iter(), 25);
        assert_eq!(defaults.tol(), 1e-06);
        assert_eq!(defaults.reg_param(), 0.0);
        assert_eq!(defaults.weight_col(), None);
        assert_eq!(defaults.solver(), "irls");
        assert_eq!(defaults.link_prediction_col(), None);
        assert_eq!(defaults.variance_power(), 0.0);
        assert_eq!(defaults.link_power(), None);
        assert_eq!(defaults.offset_col(), None);
        assert_eq!(defaults.aggregation_depth(), 2);
        let mut configured = GeneralizedLinearRegression::new()
            .set_label_col("target")
            .set_features_col("input")
            .set_prediction_col("pred")
            .set_family("tweedie")
            .set_link("log")
            .set_fit_intercept(false)
            .set_max_iter(50)
            .set_tol(1e-08)
            .set_reg_param(0.25)
            .set_weight_col("weight")
            .set_solver("irls")
            .set_link_prediction_col("eta")
            .set_variance_power(1.5)
            .set_link_power(0.0)
            .set_offset_col("offset")
            .set_aggregation_depth(3);
        assert_eq!(configured.label_col(), "target");
        assert_eq!(configured.features_col(), "input");
        assert_eq!(configured.prediction_col(), "pred");
        assert_eq!(configured.family(), "tweedie");
        assert_eq!(configured.link(), Some("log"));
        assert!(!configured.fit_intercept());
        assert_eq!(configured.max_iter(), 50);
        assert_eq!(configured.tol(), 1e-08);
        assert_eq!(configured.reg_param(), 0.25);
        assert_eq!(configured.weight_col(), Some("weight"));
        assert_eq!(configured.solver(), "irls");
        assert_eq!(configured.link_prediction_col(), Some("eta"));
        assert_eq!(configured.variance_power(), 1.5);
        assert_eq!(configured.link_power(), Some(0.0));
        assert_eq!(configured.offset_col(), Some("offset"));
        assert_eq!(configured.aggregation_depth(), 3);
        *configured.params_mut() = Params::new()
            .set_param_string("family", "poisson")
            .set_param_int("linkPower", -1)
            .set_param_null("link");
        assert_eq!(configured.family(), "poisson");
        assert_eq!(configured.link_power(), Some(-1.0));
        assert_eq!(configured.link(), None);
        *configured.params_mut() = Params::new().set_param_int32("linkPower", 2);
        assert_eq!(configured.link_power(), Some(2.0));
        *configured.params_mut() = Params::new().set_param_null("linkPower");
        assert_eq!(configured.link_power(), None);
    }

    #[test]
    fn bisecting_kmeans_getters_follow_defaults_builders_and_mutable_params() {
        let mut estimator = BisectingKMeans::default();
        assert_eq!(estimator.features_col(), "features");
        assert_eq!(estimator.prediction_col(), "prediction");
        assert_eq!(estimator.max_iter(), 20);
        assert_eq!(
            estimator.seed(),
            java_string_hash("org.apache.spark.ml.clustering.BisectingKMeans")
        );
        assert_eq!(estimator.k(), 4);
        assert_eq!(estimator.min_divisible_cluster_size(), 1.0);
        assert_eq!(estimator.distance_measure(), "euclidean");
        assert_eq!(estimator.weight_col(), None);
        let built = BisectingKMeans::new()
            .set_features_col("input")
            .set_prediction_col("cluster")
            .set_max_iter(30)
            .set_seed(1_i64 << 33)
            .set_k(3)
            .set_min_divisible_cluster_size(0.1)
            .set_distance_measure("cosine")
            .set_weight_col("weight");
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("featuresCol", "input");
        assert_eq!(estimator.features_col(), "input");
        assert_eq!(built.features_col(), estimator.features_col());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("predictionCol", "cluster");
        assert_eq!(estimator.prediction_col(), "cluster");
        assert_eq!(built.prediction_col(), estimator.prediction_col());
        *estimator.params_mut() = estimator.params().clone().set_param_int32("maxIter", 30);
        assert_eq!(estimator.max_iter(), 30);
        assert_eq!(built.max_iter(), estimator.max_iter());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_int_narrow("seed", 1_i64 << 33);
        assert_eq!(estimator.seed(), 1_i64 << 33);
        assert_eq!(built.seed(), estimator.seed());
        *estimator.params_mut() = estimator.params().clone().set_param_int32("k", 3);
        assert_eq!(estimator.k(), 3);
        assert_eq!(built.k(), estimator.k());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_double("minDivisibleClusterSize", 0.1);
        assert_eq!(estimator.min_divisible_cluster_size(), 0.1);
        assert_eq!(
            built.min_divisible_cluster_size(),
            estimator.min_divisible_cluster_size()
        );
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("distanceMeasure", "cosine");
        assert_eq!(estimator.distance_measure(), "cosine");
        assert_eq!(built.distance_measure(), estimator.distance_measure());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("weightCol", "weight");
        assert_eq!(estimator.weight_col(), Some("weight"));
        assert_eq!(built.weight_col(), estimator.weight_col());
        assert_eq!(built.params().to_proto(), estimator.params().to_proto());
        assert!(estimator.operator_mut().uid.starts_with("BisectingKMeans_"));
    }

    #[test]
    fn gaussian_mixture_getters_follow_defaults_builders_and_mutable_params() {
        let mut estimator = GaussianMixture::default();
        assert_eq!(estimator.features_col(), "features");
        assert_eq!(estimator.prediction_col(), "prediction");
        assert_eq!(estimator.k(), 2);
        assert_eq!(estimator.probability_col(), "probability");
        assert_eq!(estimator.tol(), 0.01);
        assert_eq!(estimator.max_iter(), 100);
        assert_eq!(
            estimator.seed(),
            java_string_hash("org.apache.spark.ml.clustering.GaussianMixture")
        );
        assert_eq!(estimator.aggregation_depth(), 2);
        assert_eq!(estimator.weight_col(), None);
        let built = GaussianMixture::new()
            .set_features_col("input")
            .set_prediction_col("cluster")
            .set_k(3)
            .set_probability_col("probs")
            .set_tol(1e-5)
            .set_max_iter(50)
            .set_seed(1_i64 << 33)
            .set_aggregation_depth(3)
            .set_weight_col("weight");
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("featuresCol", "input");
        assert_eq!(estimator.features_col(), "input");
        assert_eq!(built.features_col(), estimator.features_col());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("predictionCol", "cluster");
        assert_eq!(estimator.prediction_col(), "cluster");
        assert_eq!(built.prediction_col(), estimator.prediction_col());
        *estimator.params_mut() = estimator.params().clone().set_param_int32("k", 3);
        assert_eq!(estimator.k(), 3);
        assert_eq!(built.k(), estimator.k());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("probabilityCol", "probs");
        assert_eq!(estimator.probability_col(), "probs");
        assert_eq!(built.probability_col(), estimator.probability_col());
        *estimator.params_mut() = estimator.params().clone().set_param_double("tol", 1e-5);
        assert_eq!(estimator.tol(), 1e-5);
        assert_eq!(built.tol(), estimator.tol());
        *estimator.params_mut() = estimator.params().clone().set_param_int32("maxIter", 50);
        assert_eq!(estimator.max_iter(), 50);
        assert_eq!(built.max_iter(), estimator.max_iter());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_int_narrow("seed", 1_i64 << 33);
        assert_eq!(estimator.seed(), 1_i64 << 33);
        assert_eq!(built.seed(), estimator.seed());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_int32("aggregationDepth", 3);
        assert_eq!(estimator.aggregation_depth(), 3);
        assert_eq!(built.aggregation_depth(), estimator.aggregation_depth());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("weightCol", "weight");
        assert_eq!(estimator.weight_col(), Some("weight"));
        assert_eq!(built.weight_col(), estimator.weight_col());
        assert_eq!(estimator.params().to_proto(), built.params().to_proto());
        assert!(estimator.operator_mut().uid.starts_with("GaussianMixture_"));
    }

    #[test]
    fn gaussian_mixture_validates_fitted_references_and_model_metadata() {
        let session = offline_session();
        let estimator = GaussianMixture::default().set_prediction_col("cluster");
        assert_eq!(
            estimator.operator().name,
            "org.apache.spark.ml.clustering.GaussianMixture"
        );
        assert_eq!(estimator.operator().op_type, OperatorType::Estimator);
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "model".into(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo {
                        r#type: kind,
                        ..Default::default()
                    },
                    &session
                )
                .is_err());
        }
        let info = |uid| proto::ml_command_result::MlOperatorInfo {
            r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef {
                    id: "trained-gaussian-mixture".into(),
                },
            )),
            uid,
            ..Default::default()
        };
        let mut model = estimator
            .model_from_info(info(Some("returned-uid".into())), &session)
            .unwrap();
        assert_eq!(model.operator().uid, "returned-uid");
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.clustering.GaussianMixtureModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let fallback = estimator.model_from_info(info(None), &session).unwrap();
        assert_eq!(fallback.operator().uid, estimator.operator().uid);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        assert_eq!(
            t.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "trained-gaussian-mixture".into()
                }
            ))
        );
        assert_eq!(*t.input.unwrap(), df.relation().unwrap());
        assert_eq!(t.params.unwrap(), estimator.params().to_proto());
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
    }
    #[test]
    fn bisecting_kmeans_validates_fitted_references_and_model_metadata() {
        let session = offline_session();
        let estimator = BisectingKMeans::default().set_prediction_col("cluster");
        assert_eq!(
            estimator.operator().name,
            "org.apache.spark.ml.clustering.BisectingKMeans"
        );
        assert_eq!(estimator.operator().op_type, OperatorType::Estimator);
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "model".into(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo {
                        r#type: kind,
                        ..Default::default()
                    },
                    &session
                )
                .is_err());
        }
        let info = |uid| proto::ml_command_result::MlOperatorInfo {
            r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef {
                    id: "trained-bisecting-kmeans".into(),
                },
            )),
            uid,
            ..Default::default()
        };
        let mut model = estimator
            .model_from_info(info(Some("returned-uid".into())), &session)
            .unwrap();
        assert_eq!(model.operator().uid, "returned-uid");
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.clustering.BisectingKMeansModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let fallback = estimator.model_from_info(info(None), &session).unwrap();
        assert_eq!(fallback.operator().uid, estimator.operator().uid);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        assert_eq!(
            t.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "trained-bisecting-kmeans".into()
                }
            ))
        );
        assert_eq!(*t.input.unwrap(), df.relation().unwrap());
        assert_eq!(t.params.unwrap(), estimator.params().to_proto());
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
    }
    #[test]
    fn kmeans_validates_fitted_references_and_model_metadata() {
        let session = offline_session();
        let estimator = KMeans::default().set_prediction_col("cluster");
        assert_eq!(
            estimator.operator().name,
            "org.apache.spark.ml.clustering.KMeans"
        );
        assert_eq!(estimator.operator().op_type, OperatorType::Estimator);
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "model".into(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo {
                        r#type: kind,
                        ..Default::default()
                    },
                    &session
                )
                .is_err());
        }
        let info = |uid| proto::ml_command_result::MlOperatorInfo {
            r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef {
                    id: "trained-kmeans".into(),
                },
            )),
            uid,
            ..Default::default()
        };
        let mut model = estimator
            .model_from_info(info(Some("returned-uid".into())), &session)
            .unwrap();
        assert_eq!(model.operator().uid, "returned-uid");
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.clustering.KMeansModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let fallback = estimator.model_from_info(info(None), &session).unwrap();
        assert_eq!(fallback.operator().uid, estimator.operator().uid);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        assert_eq!(
            t.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "trained-kmeans".into()
                }
            ))
        );
        assert_eq!(*t.input.unwrap(), df.relation().unwrap());
        assert_eq!(t.params.unwrap(), estimator.params().to_proto());
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
    }
    #[test]
    fn decision_tree_classifier_getters_follow_defaults_builders_and_mutable_params() {
        let mut tree = DecisionTreeClassifier::default();
        assert_eq!(tree.features_col(), "features");
        assert_eq!(tree.label_col(), "label");
        assert_eq!(tree.prediction_col(), "prediction");
        assert_eq!(tree.probability_col(), "probability");
        assert_eq!(tree.raw_prediction_col(), "rawPrediction");
        assert_eq!(tree.max_depth(), 5);
        assert_eq!(tree.max_bins(), 32);
        assert_eq!(tree.min_instances_per_node(), 1);
        assert_eq!(tree.min_info_gain(), 0.0);
        assert!(tree.prune_tree());
        assert_eq!(tree.max_memory_in_mb(), 256);
        assert!(!tree.cache_node_ids());
        assert_eq!(tree.checkpoint_interval(), 10);
        assert_eq!(tree.impurity(), "gini");
        assert_eq!(
            tree.seed(),
            java_string_hash("org.apache.spark.ml.classification.DecisionTreeClassifier")
        );
        assert_eq!(tree.weight_col(), None);
        assert_eq!(tree.leaf_col(), "");
        assert_eq!(tree.min_weight_fraction_per_node(), 0.0);
        assert_eq!(tree.thresholds(), None);
        let mut built = DecisionTreeClassifier::new()
            .set_features_col("input")
            .set_label_col("target")
            .set_prediction_col("class")
            .set_probability_col("probs")
            .set_raw_prediction_col("counts")
            .set_max_depth(3)
            .set_max_bins(16)
            .set_min_instances_per_node(2)
            .set_min_info_gain(0.01)
            .set_prune_tree(false)
            .set_max_memory_in_mb(64)
            .set_cache_node_ids(true)
            .set_checkpoint_interval(-1)
            .set_impurity("entropy")
            .set_seed(1_i64 << 33)
            .set_weight_col("weight")
            .set_leaf_col("leaf")
            .set_min_weight_fraction_per_node(0.1)
            .set_thresholds([0.25, 0.75]);
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("featuresCol", "input");
        assert_eq!(tree.features_col(), "input");
        assert_eq!(built.features_col(), tree.features_col());
        *tree.params_mut() = tree.params().clone().set_param_string("labelCol", "target");
        assert_eq!(tree.label_col(), "target");
        assert_eq!(built.label_col(), tree.label_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("predictionCol", "class");
        assert_eq!(tree.prediction_col(), "class");
        assert_eq!(built.prediction_col(), tree.prediction_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("probabilityCol", "probs");
        assert_eq!(tree.probability_col(), "probs");
        assert_eq!(built.probability_col(), tree.probability_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("rawPredictionCol", "counts");
        assert_eq!(tree.raw_prediction_col(), "counts");
        assert_eq!(built.raw_prediction_col(), tree.raw_prediction_col());
        *tree.params_mut() = tree.params().clone().set_param_int32("maxDepth", 3);
        assert_eq!(tree.max_depth(), 3);
        assert_eq!(built.max_depth(), tree.max_depth());
        *tree.params_mut() = tree.params().clone().set_param_int32("maxBins", 16);
        assert_eq!(tree.max_bins(), 16);
        assert_eq!(built.max_bins(), tree.max_bins());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_int32("minInstancesPerNode", 2);
        assert_eq!(tree.min_instances_per_node(), 2);
        assert_eq!(
            built.min_instances_per_node(),
            tree.min_instances_per_node()
        );
        *tree.params_mut() = tree.params().clone().set_param_double("minInfoGain", 0.01);
        assert_eq!(tree.min_info_gain(), 0.01);
        assert_eq!(built.min_info_gain(), tree.min_info_gain());
        *tree.params_mut() = tree.params().clone().set_param_bool("pruneTree", false);
        assert!(!tree.prune_tree());
        assert_eq!(built.prune_tree(), tree.prune_tree());
        *tree.params_mut() = tree.params().clone().set_param_int32("maxMemoryInMB", 64);
        assert_eq!(tree.max_memory_in_mb(), 64);
        assert_eq!(built.max_memory_in_mb(), tree.max_memory_in_mb());
        *tree.params_mut() = tree.params().clone().set_param_bool("cacheNodeIds", true);
        assert!(tree.cache_node_ids());
        assert_eq!(built.cache_node_ids(), tree.cache_node_ids());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_int32("checkpointInterval", -1);
        assert_eq!(tree.checkpoint_interval(), -1);
        assert_eq!(built.checkpoint_interval(), tree.checkpoint_interval());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("impurity", "entropy");
        assert_eq!(tree.impurity(), "entropy");
        assert_eq!(built.impurity(), tree.impurity());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_int_narrow("seed", 1_i64 << 33);
        assert_eq!(tree.seed(), 1_i64 << 33);
        assert_eq!(built.seed(), tree.seed());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("weightCol", "weight");
        assert_eq!(tree.weight_col(), Some("weight"));
        assert_eq!(built.weight_col(), tree.weight_col());
        *tree.params_mut() = tree.params().clone().set_param_string("leafCol", "leaf");
        assert_eq!(tree.leaf_col(), "leaf");
        assert_eq!(built.leaf_col(), tree.leaf_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_double("minWeightFractionPerNode", 0.1);
        assert_eq!(tree.min_weight_fraction_per_node(), 0.1);
        assert_eq!(
            built.min_weight_fraction_per_node(),
            tree.min_weight_fraction_per_node()
        );
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_double_array("thresholds", [0.25, 0.75]);
        assert_eq!(tree.thresholds(), Some(vec![0.25, 0.75]));
        assert_eq!(built.thresholds(), tree.thresholds());
        assert_eq!(tree.params().to_proto(), built.params().to_proto());
        assert!(built
            .operator_mut()
            .uid
            .starts_with("DecisionTreeClassifier_"));
    }

    #[test]
    fn naive_bayes_getters_follow_defaults_builders_and_mutable_params() {
        let mut estimator = NaiveBayes::default();
        assert_eq!(estimator.features_col(), "features");
        assert_eq!(estimator.label_col(), "label");
        assert_eq!(estimator.prediction_col(), "prediction");
        assert_eq!(estimator.probability_col(), "probability");
        assert_eq!(estimator.raw_prediction_col(), "rawPrediction");
        assert_eq!(estimator.smoothing(), 1.0);
        assert_eq!(estimator.model_type(), "multinomial");
        assert_eq!(estimator.thresholds(), None);
        assert_eq!(estimator.weight_col(), None);
        let built = NaiveBayes::new()
            .set_features_col("input")
            .set_label_col("target")
            .set_prediction_col("class")
            .set_probability_col("probs")
            .set_raw_prediction_col("scores")
            .set_smoothing(0.5)
            .set_model_type("gaussian")
            .set_thresholds(vec![0.5, 1.5])
            .set_weight_col("weight");
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("featuresCol", "input");
        assert_eq!(estimator.features_col(), "input");
        assert_eq!(built.features_col(), estimator.features_col());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("labelCol", "target");
        assert_eq!(estimator.label_col(), "target");
        assert_eq!(built.label_col(), estimator.label_col());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("predictionCol", "class");
        assert_eq!(estimator.prediction_col(), "class");
        assert_eq!(built.prediction_col(), estimator.prediction_col());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("probabilityCol", "probs");
        assert_eq!(estimator.probability_col(), "probs");
        assert_eq!(built.probability_col(), estimator.probability_col());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("rawPredictionCol", "scores");
        assert_eq!(estimator.raw_prediction_col(), "scores");
        assert_eq!(built.raw_prediction_col(), estimator.raw_prediction_col());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_double("smoothing", 0.5);
        assert_eq!(estimator.smoothing(), 0.5);
        assert_eq!(built.smoothing(), estimator.smoothing());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("modelType", "gaussian");
        assert_eq!(estimator.model_type(), "gaussian");
        assert_eq!(built.model_type(), estimator.model_type());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_double_array("thresholds", vec![0.5, 1.5]);
        assert_eq!(estimator.thresholds(), Some(vec![0.5, 1.5]));
        assert_eq!(built.thresholds(), estimator.thresholds());
        *estimator.params_mut() = estimator
            .params()
            .clone()
            .set_param_string("weightCol", "weight");
        assert_eq!(estimator.weight_col(), Some("weight"));
        assert_eq!(built.weight_col(), estimator.weight_col());
        assert_eq!(built.params().to_proto(), estimator.params().to_proto());
        assert!(estimator.operator_mut().uid.starts_with("NaiveBayes_"));
    }

    #[test]
    fn naive_bayes_validates_fitted_references_and_model_metadata() {
        let session = offline_session();
        let estimator = NaiveBayes::default().set_prediction_col("class");
        assert_eq!(
            estimator.operator().name,
            "org.apache.spark.ml.classification.NaiveBayes"
        );
        assert_eq!(estimator.operator().op_type, OperatorType::Estimator);
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "model".into(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo {
                        r#type: kind,
                        ..Default::default()
                    },
                    &session
                )
                .is_err());
        }
        let info = |uid| proto::ml_command_result::MlOperatorInfo {
            r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef {
                    id: "trained-naive-bayes".into(),
                },
            )),
            uid,
            ..Default::default()
        };
        let mut model = estimator
            .model_from_info(info(Some("returned-uid".into())), &session)
            .unwrap();
        assert_eq!(model.operator().uid, "returned-uid");
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.classification.NaiveBayesModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let fallback = estimator.model_from_info(info(None), &session).unwrap();
        assert_eq!(fallback.operator().uid, estimator.operator().uid);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        assert_eq!(
            t.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "trained-naive-bayes".into()
                }
            ))
        );
        assert_eq!(*t.input.unwrap(), df.relation().unwrap());
        assert_eq!(t.params.unwrap(), estimator.params().to_proto());
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
    }
    #[test]
    fn decision_tree_classifier_validates_fitted_references_and_model_metadata() {
        let session = offline_session();
        let estimator = DecisionTreeClassifier::default().set_prediction_col("class");
        assert_eq!(
            estimator.operator().name,
            "org.apache.spark.ml.classification.DecisionTreeClassifier"
        );
        assert_eq!(estimator.operator().op_type, OperatorType::Estimator);
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "model".into(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo {
                        r#type: kind,
                        ..Default::default()
                    },
                    &session
                )
                .is_err());
        }
        let info = |uid| proto::ml_command_result::MlOperatorInfo {
            r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef {
                    id: "trained-decision-tree-classifier".into(),
                },
            )),
            uid,
            ..Default::default()
        };
        let mut model = estimator
            .model_from_info(info(Some("returned-uid".into())), &session)
            .unwrap();
        assert_eq!(model.operator().uid, "returned-uid");
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.classification.DecisionTreeClassificationModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let fallback = estimator.model_from_info(info(None), &session).unwrap();
        assert_eq!(fallback.operator().uid, estimator.operator().uid);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        assert_eq!(
            t.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "trained-decision-tree-classifier".into()
                }
            ))
        );
        assert_eq!(*t.input.unwrap(), df.relation().unwrap());
        assert_eq!(t.params.unwrap(), estimator.params().to_proto());
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
    }
    #[test]
    fn gbt_classifier_getters_follow_defaults_builders_and_mutable_params() {
        let mut tree = GBTClassifier::default();
        assert_eq!(tree.features_col(), "features");
        assert_eq!(tree.label_col(), "label");
        assert_eq!(tree.prediction_col(), "prediction");
        assert_eq!(tree.probability_col(), "probability");
        assert_eq!(tree.raw_prediction_col(), "rawPrediction");
        assert_eq!(tree.max_depth(), 5);
        assert_eq!(tree.max_bins(), 32);
        assert_eq!(tree.min_instances_per_node(), 1);
        assert_eq!(tree.min_info_gain(), 0.0);
        assert_eq!(tree.max_memory_in_mb(), 256);
        assert!(!tree.cache_node_ids());
        assert_eq!(tree.checkpoint_interval(), 10);
        assert_eq!(tree.impurity(), "variance");
        assert_eq!(tree.feature_subset_strategy(), "all");
        assert_eq!(
            tree.seed(),
            java_string_hash("org.apache.spark.ml.classification.GBTClassifier")
        );
        assert_eq!(tree.subsampling_rate(), 1.0);
        assert_eq!(tree.leaf_col(), "");
        assert_eq!(tree.min_weight_fraction_per_node(), 0.0);
        assert_eq!(tree.weight_col(), None);
        assert_eq!(tree.thresholds(), None);
        assert_eq!(tree.loss_type(), "logistic");
        assert_eq!(tree.max_iter(), 20);
        assert_eq!(tree.step_size(), 0.1);
        assert_eq!(tree.validation_tol(), 0.01);
        assert_eq!(tree.validation_indicator_col(), None);
        let mut built = GBTClassifier::new()
            .set_features_col("input")
            .set_label_col("target")
            .set_prediction_col("class")
            .set_probability_col("probs")
            .set_raw_prediction_col("margin")
            .set_max_depth(3)
            .set_max_bins(16)
            .set_min_instances_per_node(2)
            .set_min_info_gain(0.01)
            .set_max_memory_in_mb(64)
            .set_cache_node_ids(true)
            .set_checkpoint_interval(-1)
            .set_impurity("variance")
            .set_feature_subset_strategy("sqrt")
            .set_seed(1_i64 << 33)
            .set_subsampling_rate(0.75)
            .set_leaf_col("leaves")
            .set_min_weight_fraction_per_node(0.1)
            .set_weight_col("weight")
            .set_thresholds([0.25, 0.75])
            .set_loss_type("logistic")
            .set_max_iter(4)
            .set_step_size(0.2)
            .set_validation_tol(0.05)
            .set_validation_indicator_col("validation");
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("featuresCol", "input");
        assert_eq!(tree.features_col(), "input");
        assert_eq!(built.features_col(), tree.features_col());
        *tree.params_mut() = tree.params().clone().set_param_string("labelCol", "target");
        assert_eq!(tree.label_col(), "target");
        assert_eq!(built.label_col(), tree.label_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("predictionCol", "class");
        assert_eq!(tree.prediction_col(), "class");
        assert_eq!(built.prediction_col(), tree.prediction_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("probabilityCol", "probs");
        assert_eq!(tree.probability_col(), "probs");
        assert_eq!(built.probability_col(), tree.probability_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("rawPredictionCol", "margin");
        assert_eq!(tree.raw_prediction_col(), "margin");
        assert_eq!(built.raw_prediction_col(), tree.raw_prediction_col());
        *tree.params_mut() = tree.params().clone().set_param_int32("maxDepth", 3);
        assert_eq!(tree.max_depth(), 3);
        assert_eq!(built.max_depth(), tree.max_depth());
        *tree.params_mut() = tree.params().clone().set_param_int32("maxBins", 16);
        assert_eq!(tree.max_bins(), 16);
        assert_eq!(built.max_bins(), tree.max_bins());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_int32("minInstancesPerNode", 2);
        assert_eq!(tree.min_instances_per_node(), 2);
        assert_eq!(
            built.min_instances_per_node(),
            tree.min_instances_per_node()
        );
        *tree.params_mut() = tree.params().clone().set_param_double("minInfoGain", 0.01);
        assert_eq!(tree.min_info_gain(), 0.01);
        assert_eq!(built.min_info_gain(), tree.min_info_gain());
        *tree.params_mut() = tree.params().clone().set_param_int32("maxMemoryInMB", 64);
        assert_eq!(tree.max_memory_in_mb(), 64);
        assert_eq!(built.max_memory_in_mb(), tree.max_memory_in_mb());
        *tree.params_mut() = tree.params().clone().set_param_bool("cacheNodeIds", true);
        assert!(tree.cache_node_ids());
        assert_eq!(built.cache_node_ids(), tree.cache_node_ids());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_int32("checkpointInterval", -1);
        assert_eq!(tree.checkpoint_interval(), -1);
        assert_eq!(built.checkpoint_interval(), tree.checkpoint_interval());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("impurity", "variance");
        assert_eq!(tree.impurity(), "variance");
        assert_eq!(built.impurity(), tree.impurity());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("featureSubsetStrategy", "sqrt");
        assert_eq!(tree.feature_subset_strategy(), "sqrt");
        assert_eq!(
            built.feature_subset_strategy(),
            tree.feature_subset_strategy()
        );
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_int_narrow("seed", 1_i64 << 33);
        assert_eq!(tree.seed(), 1_i64 << 33);
        assert_eq!(built.seed(), tree.seed());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_double("subsamplingRate", 0.75);
        assert_eq!(tree.subsampling_rate(), 0.75);
        assert_eq!(built.subsampling_rate(), tree.subsampling_rate());
        *tree.params_mut() = tree.params().clone().set_param_string("leafCol", "leaves");
        assert_eq!(tree.leaf_col(), "leaves");
        assert_eq!(built.leaf_col(), tree.leaf_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_double("minWeightFractionPerNode", 0.1);
        assert_eq!(tree.min_weight_fraction_per_node(), 0.1);
        assert_eq!(
            built.min_weight_fraction_per_node(),
            tree.min_weight_fraction_per_node()
        );
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("weightCol", "weight");
        assert_eq!(tree.weight_col(), Some("weight"));
        assert_eq!(built.weight_col(), tree.weight_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_double_array("thresholds", [0.25, 0.75]);
        assert_eq!(tree.thresholds(), Some(vec![0.25, 0.75]));
        assert_eq!(built.thresholds(), tree.thresholds());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("lossType", "logistic");
        assert_eq!(tree.loss_type(), "logistic");
        assert_eq!(built.loss_type(), tree.loss_type());
        *tree.params_mut() = tree.params().clone().set_param_int32("maxIter", 4);
        assert_eq!(tree.max_iter(), 4);
        assert_eq!(built.max_iter(), tree.max_iter());
        *tree.params_mut() = tree.params().clone().set_param_double("stepSize", 0.2);
        assert_eq!(tree.step_size(), 0.2);
        assert_eq!(built.step_size(), tree.step_size());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_double("validationTol", 0.05);
        assert_eq!(tree.validation_tol(), 0.05);
        assert_eq!(built.validation_tol(), tree.validation_tol());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("validationIndicatorCol", "validation");
        assert_eq!(tree.validation_indicator_col(), Some("validation"));
        assert_eq!(
            built.validation_indicator_col(),
            tree.validation_indicator_col()
        );
        assert_eq!(tree.params().to_proto(), built.params().to_proto());
        assert!(built.operator_mut().uid.starts_with("GBTClassifier_"));
    }

    #[test]
    fn gbt_regressor_getters_follow_defaults_builders_and_mutable_params() {
        let mut tree = GBTRegressor::default();
        assert_eq!(tree.features_col(), "features");
        assert_eq!(tree.label_col(), "label");
        assert_eq!(tree.prediction_col(), "prediction");
        assert_eq!(tree.max_depth(), 5);
        assert_eq!(tree.max_bins(), 32);
        assert_eq!(tree.min_instances_per_node(), 1);
        assert_eq!(tree.min_info_gain(), 0.0);
        assert_eq!(tree.max_memory_in_mb(), 256);
        assert!(!tree.cache_node_ids());
        assert_eq!(tree.checkpoint_interval(), 10);
        assert_eq!(tree.impurity(), "variance");
        assert_eq!(tree.feature_subset_strategy(), "all");
        assert_eq!(
            tree.seed(),
            java_string_hash("org.apache.spark.ml.regression.GBTRegressor")
        );
        assert_eq!(tree.subsampling_rate(), 1.0);
        assert_eq!(tree.leaf_col(), "");
        assert_eq!(tree.min_weight_fraction_per_node(), 0.0);
        assert_eq!(tree.weight_col(), None);
        assert_eq!(tree.loss_type(), "squared");
        assert_eq!(tree.max_iter(), 20);
        assert_eq!(tree.step_size(), 0.1);
        assert_eq!(tree.validation_tol(), 0.01);
        assert_eq!(tree.validation_indicator_col(), None);
        let mut built = GBTRegressor::new()
            .set_features_col("input")
            .set_label_col("target")
            .set_prediction_col("value")
            .set_max_depth(3)
            .set_max_bins(16)
            .set_min_instances_per_node(2)
            .set_min_info_gain(0.01)
            .set_max_memory_in_mb(64)
            .set_cache_node_ids(true)
            .set_checkpoint_interval(-1)
            .set_impurity("variance")
            .set_feature_subset_strategy("sqrt")
            .set_seed(1_i64 << 33)
            .set_subsampling_rate(0.75)
            .set_leaf_col("leaves")
            .set_min_weight_fraction_per_node(0.1)
            .set_weight_col("weight")
            .set_loss_type("absolute")
            .set_max_iter(4)
            .set_step_size(0.2)
            .set_validation_tol(0.05)
            .set_validation_indicator_col("validation");
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("featuresCol", "input");
        assert_eq!(tree.features_col(), "input");
        assert_eq!(built.features_col(), tree.features_col());
        *tree.params_mut() = tree.params().clone().set_param_string("labelCol", "target");
        assert_eq!(tree.label_col(), "target");
        assert_eq!(built.label_col(), tree.label_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("predictionCol", "value");
        assert_eq!(tree.prediction_col(), "value");
        assert_eq!(built.prediction_col(), tree.prediction_col());
        *tree.params_mut() = tree.params().clone().set_param_int32("maxDepth", 3);
        assert_eq!(tree.max_depth(), 3);
        assert_eq!(built.max_depth(), tree.max_depth());
        *tree.params_mut() = tree.params().clone().set_param_int32("maxBins", 16);
        assert_eq!(tree.max_bins(), 16);
        assert_eq!(built.max_bins(), tree.max_bins());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_int32("minInstancesPerNode", 2);
        assert_eq!(tree.min_instances_per_node(), 2);
        assert_eq!(
            built.min_instances_per_node(),
            tree.min_instances_per_node()
        );
        *tree.params_mut() = tree.params().clone().set_param_double("minInfoGain", 0.01);
        assert_eq!(tree.min_info_gain(), 0.01);
        assert_eq!(built.min_info_gain(), tree.min_info_gain());
        *tree.params_mut() = tree.params().clone().set_param_int32("maxMemoryInMB", 64);
        assert_eq!(tree.max_memory_in_mb(), 64);
        assert_eq!(built.max_memory_in_mb(), tree.max_memory_in_mb());
        *tree.params_mut() = tree.params().clone().set_param_bool("cacheNodeIds", true);
        assert!(tree.cache_node_ids());
        assert_eq!(built.cache_node_ids(), tree.cache_node_ids());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_int32("checkpointInterval", -1);
        assert_eq!(tree.checkpoint_interval(), -1);
        assert_eq!(built.checkpoint_interval(), tree.checkpoint_interval());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("impurity", "variance");
        assert_eq!(tree.impurity(), "variance");
        assert_eq!(built.impurity(), tree.impurity());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("featureSubsetStrategy", "sqrt");
        assert_eq!(tree.feature_subset_strategy(), "sqrt");
        assert_eq!(
            built.feature_subset_strategy(),
            tree.feature_subset_strategy()
        );
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_int_narrow("seed", 1_i64 << 33);
        assert_eq!(tree.seed(), 1_i64 << 33);
        assert_eq!(built.seed(), tree.seed());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_double("subsamplingRate", 0.75);
        assert_eq!(tree.subsampling_rate(), 0.75);
        assert_eq!(built.subsampling_rate(), tree.subsampling_rate());
        *tree.params_mut() = tree.params().clone().set_param_string("leafCol", "leaves");
        assert_eq!(tree.leaf_col(), "leaves");
        assert_eq!(built.leaf_col(), tree.leaf_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_double("minWeightFractionPerNode", 0.1);
        assert_eq!(tree.min_weight_fraction_per_node(), 0.1);
        assert_eq!(
            built.min_weight_fraction_per_node(),
            tree.min_weight_fraction_per_node()
        );
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("weightCol", "weight");
        assert_eq!(tree.weight_col(), Some("weight"));
        assert_eq!(built.weight_col(), tree.weight_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("lossType", "absolute");
        assert_eq!(tree.loss_type(), "absolute");
        assert_eq!(built.loss_type(), tree.loss_type());
        *tree.params_mut() = tree.params().clone().set_param_int32("maxIter", 4);
        assert_eq!(tree.max_iter(), 4);
        assert_eq!(built.max_iter(), tree.max_iter());
        *tree.params_mut() = tree.params().clone().set_param_double("stepSize", 0.2);
        assert_eq!(tree.step_size(), 0.2);
        assert_eq!(built.step_size(), tree.step_size());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_double("validationTol", 0.05);
        assert_eq!(tree.validation_tol(), 0.05);
        assert_eq!(built.validation_tol(), tree.validation_tol());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("validationIndicatorCol", "validation");
        assert_eq!(tree.validation_indicator_col(), Some("validation"));
        assert_eq!(
            built.validation_indicator_col(),
            tree.validation_indicator_col()
        );
        assert_eq!(tree.params().to_proto(), built.params().to_proto());
        assert!(built.operator_mut().uid.starts_with("GBTRegressor_"));
    }

    #[test]
    fn gbt_regressor_validates_fitted_references_and_model_metadata() {
        let session = offline_session();
        let estimator = GBTRegressor::default().set_prediction_col("class");
        assert_eq!(
            estimator.operator().name,
            "org.apache.spark.ml.regression.GBTRegressor"
        );
        assert_eq!(estimator.operator().op_type, OperatorType::Estimator);
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "model".into(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo {
                        r#type: kind,
                        ..Default::default()
                    },
                    &session
                )
                .is_err());
        }
        let info = |uid| proto::ml_command_result::MlOperatorInfo {
            r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef {
                    id: "trained-gbt-regressor".into(),
                },
            )),
            uid,
            ..Default::default()
        };
        let mut model = estimator
            .model_from_info(info(Some("returned-uid".into())), &session)
            .unwrap();
        assert_eq!(model.operator().uid, "returned-uid");
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.regression.GBTRegressionModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let fallback = estimator.model_from_info(info(None), &session).unwrap();
        assert_eq!(fallback.operator().uid, estimator.operator().uid);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        assert_eq!(
            t.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "trained-gbt-regressor".into()
                }
            ))
        );
        assert_eq!(*t.input.unwrap(), df.relation().unwrap());
        assert_eq!(t.params.unwrap(), estimator.params().to_proto());
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
    }
    #[test]
    fn gbt_classifier_validates_fitted_references_and_model_metadata() {
        let session = offline_session();
        let estimator = GBTClassifier::default().set_prediction_col("class");
        assert_eq!(
            estimator.operator().name,
            "org.apache.spark.ml.classification.GBTClassifier"
        );
        assert_eq!(estimator.operator().op_type, OperatorType::Estimator);
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "model".into(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo {
                        r#type: kind,
                        ..Default::default()
                    },
                    &session
                )
                .is_err());
        }
        let info = |uid| proto::ml_command_result::MlOperatorInfo {
            r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef {
                    id: "trained-gbt-classifier".into(),
                },
            )),
            uid,
            ..Default::default()
        };
        let mut model = estimator
            .model_from_info(info(Some("returned-uid".into())), &session)
            .unwrap();
        assert_eq!(model.operator().uid, "returned-uid");
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.classification.GBTClassificationModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let fallback = estimator.model_from_info(info(None), &session).unwrap();
        assert_eq!(fallback.operator().uid, estimator.operator().uid);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        assert_eq!(
            t.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "trained-gbt-classifier".into()
                }
            ))
        );
        assert_eq!(*t.input.unwrap(), df.relation().unwrap());
        assert_eq!(t.params.unwrap(), estimator.params().to_proto());
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
    }
    #[test]
    fn random_forest_validates_fitted_references_and_model_metadata() {
        let session = offline_session();
        let estimator = RandomForestClassifier::default().set_prediction_col("class");
        assert_eq!(
            estimator.operator().name,
            "org.apache.spark.ml.classification.RandomForestClassifier"
        );
        assert_eq!(estimator.operator().op_type, OperatorType::Estimator);
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "model".into(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo {
                        r#type: kind,
                        ..Default::default()
                    },
                    &session
                )
                .is_err());
        }
        let info = |uid| proto::ml_command_result::MlOperatorInfo {
            r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef {
                    id: "trained-random-forest".into(),
                },
            )),
            uid,
            ..Default::default()
        };
        let mut model = estimator
            .model_from_info(info(Some("returned-uid".into())), &session)
            .unwrap();
        assert_eq!(model.operator().uid, "returned-uid");
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.classification.RandomForestClassificationModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let fallback = estimator.model_from_info(info(None), &session).unwrap();
        assert_eq!(fallback.operator().uid, estimator.operator().uid);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        assert_eq!(
            t.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "trained-random-forest".into()
                }
            ))
        );
        assert_eq!(*t.input.unwrap(), df.relation().unwrap());
        assert_eq!(t.params.unwrap(), estimator.params().to_proto());
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
    }
    #[test]
    fn decision_tree_regressor_getters_follow_defaults_builders_and_mutable_params() {
        let mut tree = DecisionTreeRegressor::default();
        assert_eq!(tree.features_col(), "features");
        assert_eq!(tree.label_col(), "label");
        assert_eq!(tree.prediction_col(), "prediction");
        assert_eq!(tree.max_depth(), 5);
        assert_eq!(tree.max_bins(), 32);
        assert_eq!(tree.min_instances_per_node(), 1);
        assert_eq!(tree.min_info_gain(), 0.0);
        assert_eq!(tree.max_memory_in_mb(), 256);
        assert!(!tree.cache_node_ids());
        assert_eq!(tree.checkpoint_interval(), 10);
        assert_eq!(tree.impurity(), "variance");
        assert_eq!(
            tree.seed(),
            java_string_hash("org.apache.spark.ml.regression.DecisionTreeRegressor")
        );
        assert_eq!(tree.variance_col(), None);
        assert_eq!(tree.weight_col(), None);
        assert_eq!(tree.leaf_col(), "");
        assert_eq!(tree.min_weight_fraction_per_node(), 0.0);
        let mut built = DecisionTreeRegressor::new()
            .set_features_col("input")
            .set_label_col("target")
            .set_prediction_col("value")
            .set_max_depth(3)
            .set_max_bins(16)
            .set_min_instances_per_node(2)
            .set_min_info_gain(0.01)
            .set_max_memory_in_mb(64)
            .set_cache_node_ids(true)
            .set_checkpoint_interval(-1)
            .set_impurity("variance")
            .set_seed(1_i64 << 33)
            .set_variance_col("uncertainty")
            .set_weight_col("weight")
            .set_leaf_col("leaf")
            .set_min_weight_fraction_per_node(0.1);
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("featuresCol", "input");
        assert_eq!(tree.features_col(), "input");
        assert_eq!(built.features_col(), tree.features_col());
        *tree.params_mut() = tree.params().clone().set_param_string("labelCol", "target");
        assert_eq!(tree.label_col(), "target");
        assert_eq!(built.label_col(), tree.label_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("predictionCol", "value");
        assert_eq!(tree.prediction_col(), "value");
        assert_eq!(built.prediction_col(), tree.prediction_col());
        *tree.params_mut() = tree.params().clone().set_param_int32("maxDepth", 3);
        assert_eq!(tree.max_depth(), 3);
        assert_eq!(built.max_depth(), tree.max_depth());
        *tree.params_mut() = tree.params().clone().set_param_int32("maxBins", 16);
        assert_eq!(tree.max_bins(), 16);
        assert_eq!(built.max_bins(), tree.max_bins());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_int32("minInstancesPerNode", 2);
        assert_eq!(tree.min_instances_per_node(), 2);
        assert_eq!(
            built.min_instances_per_node(),
            tree.min_instances_per_node()
        );
        *tree.params_mut() = tree.params().clone().set_param_double("minInfoGain", 0.01);
        assert_eq!(tree.min_info_gain(), 0.01);
        assert_eq!(built.min_info_gain(), tree.min_info_gain());
        *tree.params_mut() = tree.params().clone().set_param_int32("maxMemoryInMB", 64);
        assert_eq!(tree.max_memory_in_mb(), 64);
        assert_eq!(built.max_memory_in_mb(), tree.max_memory_in_mb());
        *tree.params_mut() = tree.params().clone().set_param_bool("cacheNodeIds", true);
        assert!(tree.cache_node_ids());
        assert_eq!(built.cache_node_ids(), tree.cache_node_ids());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_int32("checkpointInterval", -1);
        assert_eq!(tree.checkpoint_interval(), -1);
        assert_eq!(built.checkpoint_interval(), tree.checkpoint_interval());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("impurity", "variance");
        assert_eq!(tree.impurity(), "variance");
        assert_eq!(built.impurity(), tree.impurity());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_int_narrow("seed", 1_i64 << 33);
        assert_eq!(tree.seed(), 1_i64 << 33);
        assert_eq!(built.seed(), tree.seed());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("varianceCol", "uncertainty");
        assert_eq!(tree.variance_col(), Some("uncertainty"));
        assert_eq!(built.variance_col(), tree.variance_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_string("weightCol", "weight");
        assert_eq!(tree.weight_col(), Some("weight"));
        assert_eq!(built.weight_col(), tree.weight_col());
        *tree.params_mut() = tree.params().clone().set_param_string("leafCol", "leaf");
        assert_eq!(tree.leaf_col(), "leaf");
        assert_eq!(built.leaf_col(), tree.leaf_col());
        *tree.params_mut() = tree
            .params()
            .clone()
            .set_param_double("minWeightFractionPerNode", 0.1);
        assert_eq!(tree.min_weight_fraction_per_node(), 0.1);
        assert_eq!(
            built.min_weight_fraction_per_node(),
            tree.min_weight_fraction_per_node()
        );
        assert_eq!(tree.params().to_proto(), built.params().to_proto());
        assert!(built
            .operator_mut()
            .uid
            .starts_with("DecisionTreeRegressor_"));
    }

    #[test]
    fn decision_tree_regressor_validates_fitted_references_and_model_metadata() {
        let session = offline_session();
        let estimator = DecisionTreeRegressor::default().set_prediction_col("value");
        assert_eq!(
            estimator.operator().name,
            "org.apache.spark.ml.regression.DecisionTreeRegressor"
        );
        assert_eq!(estimator.operator().op_type, OperatorType::Estimator);
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "model".into(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo {
                        r#type: kind,
                        ..Default::default()
                    },
                    &session
                )
                .is_err());
        }
        let info = |uid| proto::ml_command_result::MlOperatorInfo {
            r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef {
                    id: "trained-decision-tree-regressor".into(),
                },
            )),
            uid,
            ..Default::default()
        };
        let mut model = estimator
            .model_from_info(info(Some("returned-uid".into())), &session)
            .unwrap();
        assert_eq!(model.operator().uid, "returned-uid");
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.regression.DecisionTreeRegressionModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let fallback = estimator.model_from_info(info(None), &session).unwrap();
        assert_eq!(fallback.operator().uid, estimator.operator().uid);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        assert_eq!(
            t.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "trained-decision-tree-regressor".into()
                }
            ))
        );
        assert_eq!(*t.input.unwrap(), df.relation().unwrap());
        assert_eq!(t.params.unwrap(), estimator.params().to_proto());
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
    }

    #[test]
    fn random_forest_regressor_validates_fitted_references_and_model_metadata() {
        let session = offline_session();
        let estimator = RandomForestRegressor::default().set_prediction_col("value");
        assert_eq!(
            estimator.operator().name,
            "org.apache.spark.ml.regression.RandomForestRegressor"
        );
        assert_eq!(estimator.operator().op_type, OperatorType::Estimator);
        for kind in [
            None,
            Some(proto::ml_command_result::ml_operator_info::Type::Name(
                "model".into(),
            )),
            Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef::default(),
            )),
        ] {
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo {
                        r#type: kind,
                        ..Default::default()
                    },
                    &session
                )
                .is_err());
        }
        let info = |uid| proto::ml_command_result::MlOperatorInfo {
            r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                proto::ObjectRef {
                    id: "trained-random-forest-regressor".into(),
                },
            )),
            uid,
            ..Default::default()
        };
        let mut model = estimator
            .model_from_info(info(Some("returned-uid".into())), &session)
            .unwrap();
        assert_eq!(model.operator().uid, "returned-uid");
        assert_eq!(
            model.operator().name,
            "org.apache.spark.ml.regression.RandomForestRegressionModel"
        );
        assert_eq!(model.operator().op_type, OperatorType::Model);
        let fallback = estimator.model_from_info(info(None), &session).unwrap();
        assert_eq!(fallback.operator().uid, estimator.operator().uid);
        let df = session.range(2).unwrap();
        let proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&df).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        assert_eq!(
            t.operator,
            Some(proto::ml_relation::transform::Operator::ObjRef(
                proto::ObjectRef {
                    id: "trained-random-forest-regressor".into()
                }
            ))
        );
        assert_eq!(*t.input.unwrap(), df.relation().unwrap());
        assert_eq!(t.params.unwrap(), estimator.params().to_proto());
        assert!(model
            .transform(&session.new_session().range(2).unwrap())
            .is_err());
        assert!(model.clone_box().transform(&df).is_ok());
    }

    #[test]
    fn generated_uids_name_the_class_as_pyspark() {
        // As PySpark's `Identifiable._randomUID` and Spark's `Identifiable.randomUID`: the
        // simple name of the class and 12 hex digits, which the server's messages show.
        for (name, prefix) in [
            ("org.apache.spark.ml.clustering.KMeans", "KMeans_"),
            ("org.apache.spark.ml.Pipeline", "Pipeline_"),
            ("Unqualified", "Unqualified_"),
        ] {
            let uid = MlOperator::new(name, OperatorType::Estimator).uid;
            let hex = uid.strip_prefix(prefix).unwrap_or_else(|| panic!("{uid}"));
            assert_eq!(hex.len(), 12, "{uid}");
            assert!(
                hex.chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "{uid}"
            );
        }
        assert_ne!(KMeans::new().operator().uid, KMeans::new().operator().uid);
        assert!(RandomForestClassifier::new()
            .operator()
            .uid
            .starts_with("RandomForestClassifier_"));
    }

    #[test]
    fn seeds_are_sent_as_pyspark_sends_a_python_int() {
        // A seed, a JVM Long, is the narrowest integer literal that holds it, as PySpark sends
        // a Python int, for every estimator that has one.
        use proto::expression::literal::LiteralType as L;
        let sent = |params: &Params| params.to_proto().params["seed"].literal_type.clone();
        for (seed, expected) in [
            (7, L::Integer(7)),
            (i64::from(i32::MIN), L::Integer(i32::MIN)),
            (1 << 31, L::Long(1 << 31)),
        ] {
            let kmeans = KMeans::new().set_seed(seed);
            assert_eq!(sent(kmeans.params()), Some(expected.clone()));
            let gm = GaussianMixture::new().set_seed(seed);
            assert_eq!(sent(gm.params()), Some(expected.clone()));
            let bkm = BisectingKMeans::new().set_seed(seed);
            assert_eq!(sent(bkm.params()), Some(expected.clone()));
            let forest = RandomForestClassifier::new().set_seed(seed);
            assert_eq!(sent(forest.params()), Some(expected.clone()));
            let tree = DecisionTreeClassifier::new().set_seed(seed);
            assert_eq!(sent(tree.params()), Some(expected.clone()));
            let tree_regressor = DecisionTreeRegressor::new().set_seed(seed);
            assert_eq!(sent(tree_regressor.params()), Some(expected.clone()));
            let gbt = GBTClassifier::new().set_seed(seed);
            assert_eq!(sent(gbt.params()), Some(expected.clone()));
            let gbtr = GBTRegressor::new().set_seed(seed);
            assert_eq!(sent(gbtr.params()), Some(expected));
        }
    }

    #[test]
    fn column_getters_give_the_defaults_of_spark() {
        // As Spark's HasInputCol, which has no default, and HasOutputCol, whose default is
        // `<uid>__output`, the column that the server writes when it is not set.
        let scaler = StandardScaler::new();
        assert_eq!(scaler.input_col(), None);
        assert_eq!(
            scaler.output_col(),
            format!("{}__output", scaler.operator.uid)
        );
        let scaler = scaler.set_input_col("x").set_output_col("y");
        assert_eq!(
            (scaler.input_col(), scaler.output_col()),
            (Some("x"), "y".to_string())
        );
        let max_abs = MaxAbsScaler::new();
        assert_eq!(max_abs.input_col(), None);
        assert_eq!(
            max_abs.output_col(),
            format!("{}__output", max_abs.operator.uid)
        );
        let indexer = StringIndexer::new();
        assert_eq!(indexer.input_col(), None);
        assert_eq!(
            indexer.output_col(),
            format!("{}__output", indexer.operator.uid)
        );
        let assembler = VectorAssembler::new();
        assert_eq!(
            assembler.output_col(),
            format!("{}__output", assembler.operator.uid)
        );
        // The list setters take any strings, as `set_param_string_array`.
        let assembler = assembler.set_input_cols(["a".to_string(), "b".to_string()]);
        assert_eq!(assembler.input_cols(), vec!["a", "b"]);
    }

    #[test]
    fn native_estimators_get_their_params_or_the_spark_defaults() {
        // Every native estimator has a getter of each param: the value that is set, else the
        // default of Spark (None for a param that has none).
        let km = KMeans::new();
        macro_rules! all {
            ($km:expr) => {{
                let km: &KMeans = &$km;
                (
                    (
                        km.features_col(),
                        km.prediction_col(),
                        km.k(),
                        km.init_mode(),
                    ),
                    (km.init_steps(), km.tol(), km.max_iter(), km.seed()),
                    (km.distance_measure(), km.weight_col(), km.solver()),
                    km.max_block_size_in_mb(),
                )
            }};
        }
        assert_eq!(
            all!(km),
            (
                ("features", "prediction", 2, "k-means||"),
                (
                    2,
                    1e-4,
                    20,
                    java_string_hash("org.apache.spark.ml.clustering.KMeans")
                ),
                ("euclidean", None, "auto"),
                0.0
            )
        );
        let km = km
            .set_features_col("f")
            .set_prediction_col("p")
            .set_k(3)
            .set_init_mode("random")
            .set_init_steps(4)
            .set_tol(0.5)
            .set_max_iter(7)
            .set_seed(1 << 40)
            .set_distance_measure("cosine")
            .set_weight_col("w")
            .set_solver("row")
            .set_max_block_size_in_mb(1.5);
        assert_eq!(
            all!(km),
            (
                ("f", "p", 3, "random"),
                (4, 0.5, 7, 1 << 40),
                ("cosine", Some("w"), "row"),
                1.5
            )
        );

        let svc = LinearSVC::new();
        macro_rules! all {
            ($s:expr) => {{
                let s: &LinearSVC = &$s;
                (
                    (s.features_col(), s.label_col(), s.prediction_col()),
                    (s.raw_prediction_col(), s.max_iter(), s.reg_param(), s.tol()),
                    (s.fit_intercept(), s.standardization(), s.threshold()),
                    (
                        s.weight_col(),
                        s.aggregation_depth(),
                        s.max_block_size_in_mb(),
                    ),
                )
            }};
        }
        assert_eq!(
            all!(svc),
            (
                ("features", "label", "prediction"),
                ("rawPrediction", 100, 0.0, 1e-6),
                (true, true, 0.0),
                (None, 2, 0.0)
            )
        );
        let svc = svc
            .set_features_col("f")
            .set_label_col("l")
            .set_prediction_col("p")
            .set_raw_prediction_col("r")
            .set_max_iter(5)
            .set_reg_param(0.1)
            .set_tol(0.2)
            .set_fit_intercept(false)
            .set_standardization(false)
            .set_threshold(0.3)
            .set_weight_col("w")
            .set_aggregation_depth(3)
            .set_max_block_size_in_mb(1.5);
        assert_eq!(
            all!(svc),
            (
                ("f", "l", "p"),
                ("r", 5, 0.1, 0.2),
                (false, false, 0.3),
                (Some("w"), 3, 1.5)
            )
        );

        let lr = LinearRegression::new();
        macro_rules! all {
            ($r:expr) => {{
                let r: &LinearRegression = &$r;
                (
                    (
                        r.features_col(),
                        r.label_col(),
                        r.prediction_col(),
                        r.max_iter(),
                    ),
                    (
                        r.reg_param(),
                        r.elastic_net_param(),
                        r.tol(),
                        r.fit_intercept(),
                    ),
                    (r.standardization(), r.solver(), r.weight_col()),
                    (r.aggregation_depth(), r.loss(), r.epsilon()),
                    r.max_block_size_in_mb(),
                )
            }};
        }
        assert_eq!(
            all!(lr),
            (
                ("features", "label", "prediction", 100),
                (0.0, 0.0, 1e-6, true),
                (true, "auto", None),
                (2, "squaredError", 1.35),
                0.0
            )
        );
        let lr = lr
            .set_features_col("f")
            .set_label_col("l")
            .set_prediction_col("p")
            .set_max_iter(5)
            .set_reg_param(0.1)
            .set_elastic_net_param(0.2)
            .set_tol(0.3)
            .set_fit_intercept(false)
            .set_standardization(false)
            .set_solver("normal")
            .set_weight_col("w")
            .set_aggregation_depth(3)
            .set_loss("huber")
            .set_epsilon(1.5)
            .set_max_block_size_in_mb(2.5);
        assert_eq!(
            all!(lr),
            (
                ("f", "l", "p", 5),
                (0.1, 0.2, 0.3, false),
                (false, "normal", Some("w")),
                (3, "huber", 1.5),
                2.5
            )
        );

        let rf = RandomForestClassifier::new();
        macro_rules! all {
            ($f:expr) => {{
                let f: &RandomForestClassifier = &$f;
                (
                    (f.features_col(), f.label_col(), f.prediction_col()),
                    (f.probability_col(), f.raw_prediction_col(), f.max_depth()),
                    (f.max_bins(), f.min_instances_per_node(), f.min_info_gain()),
                    (f.prune_tree(), f.max_memory_in_mb(), f.cache_node_ids()),
                    (f.checkpoint_interval(), f.impurity(), f.num_trees()),
                    (f.feature_subset_strategy(), f.seed(), f.subsampling_rate()),
                    (
                        f.leaf_col(),
                        f.min_weight_fraction_per_node(),
                        f.weight_col(),
                    ),
                    (f.bootstrap(), f.thresholds()),
                )
            }};
        }
        assert_eq!(
            all!(rf),
            (
                ("features", "label", "prediction"),
                ("probability", "rawPrediction", 5),
                (32, 1, 0.0),
                (true, 256, false),
                (10, "gini", 20),
                (
                    "auto",
                    java_string_hash("org.apache.spark.ml.classification.RandomForestClassifier"),
                    1.0
                ),
                ("", 0.0, None),
                (true, None)
            )
        );
        let rf = rf
            .set_features_col("f")
            .set_label_col("l")
            .set_prediction_col("p")
            .set_probability_col("pr")
            .set_raw_prediction_col("r")
            .set_max_depth(3)
            .set_max_bins(16)
            .set_min_instances_per_node(2)
            .set_min_info_gain(0.1)
            .set_prune_tree(false)
            .set_max_memory_in_mb(64)
            .set_cache_node_ids(true)
            .set_checkpoint_interval(-1)
            .set_impurity("entropy")
            .set_num_trees(7)
            .set_feature_subset_strategy("all")
            .set_seed(5)
            .set_subsampling_rate(0.5)
            .set_leaf_col("leaves")
            .set_min_weight_fraction_per_node(0.2)
            .set_weight_col("w")
            .set_bootstrap(false)
            .set_thresholds([0.25, 0.75]);
        assert_eq!(
            all!(rf),
            (
                ("f", "l", "p"),
                ("pr", "r", 3),
                (16, 2, 0.1),
                (false, 64, true),
                (-1, "entropy", 7),
                ("all", 5, 0.5),
                ("leaves", 0.2, Some("w")),
                (false, Some(vec![0.25, 0.75]))
            )
        );

        let logistic = LogisticRegression::new();
        assert_eq!(
            (
                logistic.features_col(),
                logistic.label_col(),
                logistic.prediction_col(),
                logistic.max_iter()
            ),
            ("features", "label", "prediction", 100)
        );
        let cv = CrossValidator::new();
        assert_eq!((cv.num_folds(), cv.parallelism()), (3, 1));
        // The default seed is random of each validator, as PySpark's of the hash of the name of
        // its class differs between processes.
        assert_eq!(cv.seed(), cv.seed());
        let cv = cv.set_num_folds(5).set_parallelism(2).set_seed(9);
        assert_eq!((cv.num_folds(), cv.parallelism(), cv.seed()), (5, 2, 9));
    }

    #[test]
    fn getters_read_the_params_that_are_sent() {
        // The getters read the params themselves, so that a change through `params_mut` is
        // the value of the getter too. A Long literal of an Int param wraps to 32 bits, as the
        // server's `intValue` reads it.
        let mut lr = LogisticRegression::new().set_label_col("y").set_max_iter(5);
        let params = std::mem::take(lr.params_mut());
        *lr.params_mut() = params
            .set_param_int("maxIter", (1 << 33) + 3)
            .clear_param("labelCol");
        assert_eq!((lr.max_iter(), lr.label_col()), (3, "label"));

        let mut evaluator = RegressionEvaluator::new().set_metric_name("r2");
        assert_eq!(evaluator.metric_name(), "r2");
        let params = std::mem::take(evaluator.params_mut());
        *evaluator.params_mut() = params.clear_param("metricName");
        assert_eq!(evaluator.metric_name(), "rmse");
        assert!(evaluator.params().to_proto().params.is_empty());

        let mut va = VectorAssembler::new().set_input_cols(vec!["a", "b"]);
        assert_eq!(va.input_cols(), ["a", "b"]);
        let params = std::mem::take(va.params_mut());
        *va.params_mut() = params.set_param_string_array("inputCols", ["c"]);
        assert_eq!(va.input_cols(), ["c"]);
    }

    #[test]
    fn logistic_regression_model_reads_its_columns_from_its_params() {
        let session = offline_session();
        let model = fitted_logistic_regression(&session);
        assert_eq!(
            (
                model.features_col(),
                model.label_col(),
                model.prediction_col()
            ),
            ("features", "label", "pred")
        );
        assert_eq!(model.operator().uid, "operator-uid");
    }

    #[test]
    fn lda_selects_the_fitted_model_kind_and_pic_retains_its_input_owner() {
        let session = offline_session();
        for (optimizer, model_name) in [("online", "LocalLDAModel"), ("em", "DistributedLDAModel")]
        {
            let mut estimator = LDA::new()
                .set_optimizer(optimizer)
                .set_topic_distribution_col("topics");
            assert!(estimator
                .model_from_info(
                    proto::ml_command_result::MlOperatorInfo::default(),
                    &session
                )
                .is_err());
            let info = proto::ml_command_result::MlOperatorInfo {
                r#type: Some(proto::ml_command_result::ml_operator_info::Type::ObjRef(
                    proto::ObjectRef {
                        id: format!("{optimizer}-model"),
                    },
                )),
                uid: Some("fitted-uid".into()),
                ..Default::default()
            };
            let mut model = estimator.model_from_info(info, &session).unwrap();
            assert_eq!(
                model.operator().name,
                format!("org.apache.spark.ml.clustering.{model_name}")
            );
            assert_eq!(model.operator().uid, "fitted-uid");
            assert_eq!(model.operator().op_type, OperatorType::Model);
            *estimator.params_mut() = estimator
                .params()
                .clone()
                .set_param_string("topicDistributionCol", "changed");
            assert_eq!(
                model.params().string_or("topicDistributionCol", ""),
                "topics"
            );
            let weak = Arc::downgrade(&model.cached.0);
            let transformed = model.transform(&session.range(2).unwrap()).unwrap();
            let assignments = PowerIterationClustering::new()
                .assign_clusters(&transformed)
                .unwrap();
            drop(model);
            drop(transformed);
            assert!(weak.upgrade().is_some());
            drop(assignments);
            assert!(weak.upgrade().is_none());
        }
    }

    #[test]
    fn one_vs_rest_reads_its_parallelism_of_either_int_width() {
        let ovr = OneVsRest::new();
        assert_eq!(ovr.parallelism_i64(), Some(1));
        let mut ovr = ovr.set_parallelism(3);
        assert_eq!(ovr.parallelism_i64(), Some(3));
        // The Python client narrows an int beyond 32 bits to a Long, which `parallelism` wraps.
        *ovr.params_mut() = ovr
            .params()
            .clone()
            .set_param_int_narrow("parallelism", 1 << 40);
        assert_eq!(ovr.parallelism_i64(), Some(1 << 40));
        *ovr.params_mut() = ovr.params().clone().set_param_null("parallelism");
        assert_eq!(ovr.parallelism_i64(), None);
    }

    #[test]
    fn ml_string_array_reads_an_array_of_strings() {
        use proto::expression::literal::{Array, LiteralType as L};
        let string = |value: &str| proto::expression::Literal {
            literal_type: Some(L::String(value.into())),
            ..Default::default()
        };
        let array = |elements| proto::expression::Literal {
            literal_type: Some(L::Array(Array {
                elements,
                ..Default::default()
            })),
            ..Default::default()
        };
        assert_eq!(
            ml_string_array(array(vec![string("i"), string("me")])).unwrap(),
            ["i", "me"]
        );
        assert!(ml_string_array(string("i")).is_err());
        let long = proto::expression::Literal {
            literal_type: Some(L::Long(1)),
            ..Default::default()
        };
        assert!(ml_string_array(array(vec![long])).is_err());
    }
}
