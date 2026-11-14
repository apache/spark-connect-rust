// SPDX-License-Identifier: Apache-2.0
//! The objects of Spark Connect ML that the server holds, and the commands that read, save and
//! load them, as PySpark 4.3.0 sends them over Spark Connect:
//!
//! - [`MlRef`]: a fitted model or another server-side object that a fetch returns (a submodel, a
//!   tree of an ensemble), released on the server when its last owner drops, as PySpark's
//!   `RemoteModelRef`;
//! - [`MlTarget`] and [`fetch`]: a method of an object, or of an object that a method chain reaches
//!   (`<model>.summary`), whose value the server returns (`try_remote_call` of `pyspark.ml.util`);
//! - [`attribute_relation`]: a method whose value is a DataFrame (`invoke_remote_attribute_relation`);
//! - [`MlValue`]: the value of a fetch, as `pyspark.ml.connect.serialize.deserialize_param` reads it;
//! - [`write_operator`], [`write_object`] and [`read`]: the `Write` and `Read` commands of
//!   `pyspark.ml.connect.readwrite`.

use super::*;

/// The id of Spark Connect ML's helper object, whose methods (`invoke_helper_attr`,
/// `invoke_helper_relation` of `pyspark.ml.util`) the server runs without a cached object.
pub const ML_CONNECT_HELPER_ID: &str = "______ML_CONNECT_HELPER______";

impl MlRef {
    /// The id of the object on the server.
    pub fn id(&self) -> &str {
        &self.0.model_ref.id
    }

    /// The session that holds the object.
    pub fn session(&self) -> &SparkSession {
        &self.0.session
    }

    /// An object that the server returned by its id (an `operator_info` of a fetch or a read), of
    /// which this client is the first owner.
    pub(crate) fn from_id(session: &SparkSession, id: &str) -> Self {
        Self(Arc::new(CachedModelRef {
            model_ref: proto::ObjectRef { id: id.to_string() },
            session: session.clone(),
        }))
    }

    pub(crate) fn object_ref(&self) -> proto::ObjectRef {
        self.0.model_ref.clone()
    }
}

impl MlMatrix {
    /// The matrix of the struct of its UDT, as `deserialize_param` reads a `MatrixUDT`: its type
    /// (0 sparse, 1 dense), its shape, its column pointers and row indices (of a sparse matrix),
    /// its values and whether it is transposed.
    pub(crate) fn from_literal(value: &proto::expression::literal::LiteralType) -> Option<Self> {
        use proto::expression::literal::{specialized_array::ValueType as A, LiteralType as L};
        let L::Struct(value) = value else {
            return None;
        };
        let fields = &value.elements;
        let int = |i: usize| match fields.get(i)?.literal_type.as_ref()? {
            L::Integer(v) => Some(*v),
            _ => None,
        };
        let ints = |i: usize| match fields.get(i)?.literal_type.as_ref()? {
            L::SpecializedArray(array) => match array.value_type.as_ref()? {
                A::Ints(values) => Some(values.values.clone()),
                _ => None,
            },
            _ => None,
        };
        let values = match fields.get(5)?.literal_type.as_ref()? {
            L::SpecializedArray(array) => match array.value_type.as_ref()? {
                A::Doubles(values) => values.values.clone(),
                _ => return None,
            },
            _ => return None,
        };
        let is_transposed = match fields.get(6)?.literal_type.as_ref()? {
            L::Boolean(v) => *v,
            _ => return None,
        };
        match fields.first()?.literal_type.as_ref()? {
            L::Byte(1) => Some(Self::Dense {
                num_rows: int(1)?,
                num_cols: int(2)?,
                values,
                is_transposed,
            }),
            L::Byte(0) => Some(Self::Sparse {
                num_rows: int(1)?,
                num_cols: int(2)?,
                col_ptrs: ints(3)?,
                row_indices: ints(4)?,
                values,
                is_transposed,
            }),
            _ => None,
        }
    }
}

/// A value that the server returns of a method, or that a method takes as an argument: the types
/// of `pyspark.ml.connect.serialize.deserialize_param`, which reads an ML vector or matrix of its
/// UDT struct and another literal as `LiteralExpression._to_value` does.
#[derive(Debug, Clone, PartialEq)]
pub enum MlValue {
    Null,
    Bool(bool),
    Byte(i32),
    Short(i32),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    /// The decimal of its string value.
    Decimal(String),
    String(String),
    Binary(Vec<u8>),
    /// Days since the epoch.
    Date(i32),
    /// Microseconds since the epoch, in UTC.
    Timestamp(i64),
    /// Microseconds since the epoch, without a time zone.
    TimestampNtz(i64),
    Vector(MlVector),
    Matrix(MlMatrix),
    Array(Vec<MlValue>),
    Map(Vec<(MlValue, MlValue)>),
    /// A struct of its field names and values, which PySpark reads as a `Row`.
    Struct(Vec<(String, MlValue)>),
}

impl MlValue {
    /// The value of `literal`, as `deserialize_param` reads it: the struct of a `VectorUDT` or a
    /// `MatrixUDT` is a vector or a matrix, a struct of another UDT is an error (PySpark's
    /// `ValueError` of an unknown UDT), and another literal is its value.
    pub fn from_literal(
        literal: &proto::expression::Literal,
    ) -> spark_connect_core::error::Result<Self> {
        use proto::expression::literal::{specialized_array::ValueType as A, LiteralType as L};
        let unsupported =
            || SparkError::value_msg(format!("Unsupported literal of an ML result: {literal:?}"));
        let Some(value) = literal.literal_type.as_ref() else {
            return Err(unsupported());
        };
        Ok(match value {
            L::Null(_) => Self::Null,
            L::Boolean(v) => Self::Bool(*v),
            L::Byte(v) => Self::Byte(*v),
            L::Short(v) => Self::Short(*v),
            L::Integer(v) => Self::Int(*v),
            L::Long(v) => Self::Long(*v),
            L::Float(v) => Self::Float(*v),
            L::Double(v) => Self::Double(*v),
            L::Decimal(v) => Self::Decimal(v.value.clone()),
            L::String(v) => Self::String(v.clone()),
            L::Binary(v) => Self::Binary(v.to_vec()),
            L::Date(v) => Self::Date(*v),
            L::Timestamp(v) => Self::Timestamp(*v),
            L::TimestampNtz(v) => Self::TimestampNtz(*v),
            L::Array(array) => Self::Array(
                array
                    .elements
                    .iter()
                    .map(Self::from_literal)
                    .collect::<spark_connect_core::error::Result<_>>()?,
            ),
            L::Map(map) => Self::Map(
                map.keys
                    .iter()
                    .zip(&map.values)
                    .map(|(k, v)| Ok((Self::from_literal(k)?, Self::from_literal(v)?)))
                    .collect::<spark_connect_core::error::Result<_>>()?,
            ),
            L::SpecializedArray(array) => match array.value_type.as_ref() {
                Some(A::Bools(v)) => Self::Array(v.values.iter().map(|x| Self::Bool(*x)).collect()),
                Some(A::Ints(v)) => Self::Array(v.values.iter().map(|x| Self::Int(*x)).collect()),
                Some(A::Longs(v)) => Self::Array(v.values.iter().map(|x| Self::Long(*x)).collect()),
                Some(A::Floats(v)) => {
                    Self::Array(v.values.iter().map(|x| Self::Float(*x)).collect())
                }
                Some(A::Doubles(v)) => {
                    Self::Array(v.values.iter().map(|x| Self::Double(*x)).collect())
                }
                Some(A::Strings(v)) => {
                    Self::Array(v.values.iter().map(|x| Self::String(x.clone())).collect())
                }
                None => return Err(unsupported()),
            },
            L::Struct(s) => {
                #[allow(deprecated)] // PySpark's ML serializer types a UDT struct by struct_type.
                let udt = match s.struct_type.as_ref().and_then(|t| t.kind.as_ref()) {
                    Some(proto::data_type::Kind::Udt(udt)) => udt.jvm_class.clone(),
                    _ => None,
                };
                match udt.as_deref() {
                    Some("org.apache.spark.ml.linalg.VectorUDT") => {
                        Self::Vector(MlVector::from_literal(value).ok_or_else(unsupported)?)
                    }
                    Some("org.apache.spark.ml.linalg.MatrixUDT") => {
                        Self::Matrix(MlMatrix::from_literal(value).ok_or_else(unsupported)?)
                    }
                    Some(other) => {
                        return Err(SparkError::value_msg(format!("Unknown UDT {other}")))
                    }
                    None => {
                        #[allow(deprecated)]
                        let names: Vec<String> =
                            match s.struct_type.as_ref().and_then(|t| t.kind.as_ref()) {
                                Some(proto::data_type::Kind::Struct(st)) => {
                                    st.fields.iter().map(|f| f.name.clone()).collect()
                                }
                                _ => Vec::new(),
                            };
                        Self::Struct(
                            s.elements
                                .iter()
                                .enumerate()
                                .map(|(i, v)| {
                                    let name = names.get(i).cloned().unwrap_or_default();
                                    Ok((name, Self::from_literal(v)?))
                                })
                                .collect::<spark_connect_core::error::Result<_>>()?,
                        )
                    }
                }
            }
            _ => return Err(unsupported()),
        })
    }

    /// The literal of the value, as PySpark's `serialize_param` sends an argument: a vector or a
    /// matrix as its UDT struct, a scalar of its type, and an array of the type of its first
    /// element, as `LiteralExpression._infer_type` types a list. An array that is empty, or whose
    /// first element is null, cannot be typed (`CANNOT_INFER_ARRAY_ELEMENT_TYPE`): send it as a
    /// [`FetchArg::Typed`] instead.
    pub fn to_literal(&self) -> spark_connect_core::error::Result<proto::expression::Literal> {
        match self {
            Self::Vector(v) => Ok(v.to_literal()),
            Self::Matrix(m) => Ok(m.to_literal()),
            other => Ok(other.to_literal_expression(None)?.to_literal_proto()),
        }
    }

    /// The literal of the value as the type `data_type`, as PySpark sends a `(value, DataType)`
    /// pair of `serialize`, such as an empty list of strings.
    pub fn to_typed_literal(
        &self,
        data_type: &DataType,
    ) -> spark_connect_core::error::Result<proto::expression::Literal> {
        Ok(self
            .to_literal_expression(Some(data_type))?
            .to_literal_proto())
    }

    fn to_literal_expression(
        &self,
        data_type: Option<&DataType>,
    ) -> spark_connect_core::error::Result<LiteralExpression> {
        let cannot_infer = || {
            SparkError::classed(
                SparkErrorKind::TypeError,
                "CANNOT_INFER_ARRAY_ELEMENT_TYPE",
                &[],
            )
        };
        Ok(match self {
            Self::Null => LiteralExpression::Null(data_type.cloned().unwrap_or(DataType::Null)),
            Self::Bool(v) => LiteralExpression::Boolean(*v),
            Self::Byte(v) => LiteralExpression::Byte(*v),
            Self::Short(v) => LiteralExpression::Short(*v),
            Self::Int(v) => LiteralExpression::Integer(*v),
            Self::Long(v) => LiteralExpression::Long(*v),
            Self::Float(v) => LiteralExpression::Float(*v),
            Self::Double(v) => LiteralExpression::Double(*v),
            Self::Decimal(v) => {
                let (precision, scale) = match data_type {
                    Some(DataType::Decimal { precision, scale }) => (*precision, *scale),
                    _ => decimal_precision_scale(v),
                };
                LiteralExpression::Decimal {
                    value: v.clone(),
                    precision,
                    scale,
                }
            }
            Self::String(v) => LiteralExpression::String(v.clone()),
            Self::Binary(v) => LiteralExpression::Binary(v.clone()),
            Self::Date(v) => LiteralExpression::Date(*v),
            Self::Timestamp(v) => LiteralExpression::Timestamp(*v),
            Self::TimestampNtz(v) => LiteralExpression::TimestampNtz(*v),
            Self::Array(elements) => {
                let element_type = match data_type {
                    Some(DataType::Array { element_type, .. }) => (**element_type).clone(),
                    Some(_) | None => match elements.first() {
                        None | Some(Self::Null) => return Err(cannot_infer()),
                        Some(first) => first.data_type()?,
                    },
                };
                LiteralExpression::Array {
                    elements: elements
                        .iter()
                        .map(|e| e.to_literal_expression(Some(&element_type)))
                        .collect::<spark_connect_core::error::Result<_>>()?,
                    element_type: Box::new(element_type),
                }
            }
            Self::Vector(_) | Self::Matrix(_) | Self::Map(_) | Self::Struct(_) => {
                return Err(SparkError::value_msg(
                    "An ML vector, matrix, map or struct cannot be an element of an array \
                     argument",
                ))
            }
        })
    }

    /// The Spark type of the value, as `LiteralExpression._infer_type` types a value.
    fn data_type(&self) -> spark_connect_core::error::Result<DataType> {
        Ok(match self {
            Self::Null => DataType::Null,
            Self::Bool(_) => DataType::Boolean,
            Self::Byte(_) => DataType::Byte,
            Self::Short(_) => DataType::Short,
            Self::Int(_) => DataType::Integer,
            Self::Long(_) => DataType::Long,
            Self::Float(_) => DataType::Float,
            Self::Double(_) => DataType::Double,
            Self::Decimal(v) => {
                let (precision, scale) = decimal_precision_scale(v);
                DataType::Decimal { precision, scale }
            }
            Self::String(_) => DataType::String {
                collation: "UTF8_BINARY".to_string(),
            },
            Self::Binary(_) => DataType::Binary,
            Self::Date(_) => DataType::Date,
            Self::Timestamp(_) => DataType::Timestamp,
            Self::TimestampNtz(_) => DataType::TimestampNtz,
            Self::Array(elements) => match elements.first() {
                None | Some(Self::Null) => {
                    return Err(SparkError::classed(
                        SparkErrorKind::TypeError,
                        "CANNOT_INFER_ARRAY_ELEMENT_TYPE",
                        &[],
                    ))
                }
                Some(first) => DataType::Array {
                    element_type: Box::new(first.data_type()?),
                    contains_null: true,
                },
            },
            Self::Vector(_) | Self::Matrix(_) | Self::Map(_) | Self::Struct(_) => {
                return Err(SparkError::value_msg(
                    "An ML vector, matrix, map or struct cannot be an element of an array \
                     argument",
                ))
            }
        })
    }

    /// The value as a double, of a numeric value.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Double(v) => Some(*v),
            Self::Float(v) => Some(f64::from(*v)),
            Self::Int(v) | Self::Byte(v) | Self::Short(v) => Some(f64::from(*v)),
            Self::Long(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// The value as a long, of an integral value.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Long(v) => Some(*v),
            Self::Int(v) | Self::Byte(v) | Self::Short(v) => Some(i64::from(*v)),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_vector(&self) -> Option<&MlVector> {
        match self {
            Self::Vector(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_matrix(&self) -> Option<&MlMatrix> {
        match self {
            Self::Matrix(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MlValue]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// The precision and scale of the decimal string `value`, as `LiteralExpression._infer_type` types
/// a `decimal.Decimal`: its digits and the digits after its point, at least one more than its
/// scale.
fn decimal_precision_scale(value: &str) -> (i32, i32) {
    let digits = value.trim_start_matches(['-', '+']);
    let (int, frac) = digits.split_once('.').unwrap_or((digits, ""));
    let int_digits = int.trim_start_matches('0').len() as i32;
    let scale = frac.len() as i32;
    ((int_digits + scale).max(scale + 1).max(1), scale)
}

/// The object of a fetch: a server-side object, or the ML helper, and the chain of methods that
/// reaches the object of which a method is fetched, as `_extract_id_methods` of
/// `pyspark.ml.connect.util` splits `"<id>.summary"`. A summary target keeps the dataset that
/// recreates the summary when the server has offloaded the model (`CreateSummary`).
#[derive(Clone)]
pub struct MlTarget {
    root: TargetRoot,
    chain: Vec<String>,
    summary_dataset: Option<DataFrame>,
}

#[derive(Clone)]
enum TargetRoot {
    Object(MlRef),
    Helper(SparkSession),
    /// An object of an id that this client does not release, as PySpark does not release the
    /// summary that `evaluate` returns.
    Unowned(SparkSession, String),
}

impl std::fmt::Debug for MlTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlTarget")
            .field("id", &self.id())
            .field("summary_dataset", &self.summary_dataset.is_some())
            .finish()
    }
}

impl MlTarget {
    /// The object `object`.
    pub fn object(object: &MlRef) -> Self {
        Self {
            root: TargetRoot::Object(object.clone()),
            chain: Vec::new(),
            summary_dataset: None,
        }
    }

    /// The object of the id `id` of `session`, which this client does not release, as PySpark's
    /// summary of an `evaluate` (a `summary` result of a fetch).
    pub fn unowned(session: &SparkSession, id: &str) -> Self {
        Self {
            root: TargetRoot::Unowned(session.clone(), id.to_string()),
            chain: Vec::new(),
            summary_dataset: None,
        }
    }

    /// The ML helper of `session`.
    pub fn helper(session: &SparkSession) -> Self {
        Self {
            root: TargetRoot::Helper(session.clone()),
            chain: Vec::new(),
            summary_dataset: None,
        }
    }

    /// The object that the method `method` of this one returns, without arguments, as PySpark
    /// names a summary `"<id>.summary"`.
    pub fn child(&self, method: &str) -> Self {
        let mut child = self.clone();
        child.chain.push(method.to_string());
        child
    }

    /// The target, whose summary the server recreates of `dataset`, the summary dataset of the fit
    /// (PySpark's `_summary_dataset`, see [`training_summary`]), when it has lost it, and which the
    /// DataFrames of the summary read.
    pub fn with_summary_dataset(mut self, dataset: &DataFrame) -> Self {
        self.summary_dataset = Some(dataset.clone());
        self
    }

    /// The dataset of the summary of this target, if any.
    pub fn summary_dataset(&self) -> Option<&DataFrame> {
        self.summary_dataset.as_ref()
    }

    /// The object at the root of the target, unless it is the helper.
    pub fn root(&self) -> Option<&MlRef> {
        match &self.root {
            TargetRoot::Object(object) => Some(object),
            TargetRoot::Helper(_) | TargetRoot::Unowned(..) => None,
        }
    }

    /// The id of the target as PySpark names it: the id of its root, and its methods, by dots.
    pub fn id(&self) -> String {
        let root = match &self.root {
            TargetRoot::Object(object) => object.id().to_string(),
            TargetRoot::Helper(_) => ML_CONNECT_HELPER_ID.to_string(),
            TargetRoot::Unowned(_, id) => id.clone(),
        };
        std::iter::once(root)
            .chain(self.chain.iter().cloned())
            .collect::<Vec<_>>()
            .join(".")
    }

    pub fn session(&self) -> &SparkSession {
        match &self.root {
            TargetRoot::Object(object) => object.session(),
            TargetRoot::Helper(session) | TargetRoot::Unowned(session, _) => session,
        }
    }

    fn object_ref(&self) -> proto::ObjectRef {
        match &self.root {
            TargetRoot::Object(object) => object.object_ref(),
            TargetRoot::Helper(_) => proto::ObjectRef {
                id: ML_CONNECT_HELPER_ID.to_string(),
            },
            TargetRoot::Unowned(_, id) => proto::ObjectRef { id: id.clone() },
        }
    }

    /// The `Fetch` of the method `method` of the target, of the arguments `args`.
    fn fetch_proto(
        &self,
        method: &str,
        args: Vec<FetchArg>,
    ) -> spark_connect_core::error::Result<proto::Fetch> {
        use proto::fetch::method::{args::ArgsType, Args};
        let mut methods: Vec<proto::fetch::Method> = self
            .chain
            .iter()
            .map(|m| proto::fetch::Method {
                method: m.clone(),
                args: Vec::new(),
            })
            .collect();
        let args = args
            .into_iter()
            .map(|arg| {
                Ok(Args {
                    args_type: Some(match arg {
                        FetchArg::Value(value) => ArgsType::Param(value.to_literal()?),
                        FetchArg::Typed(value, data_type) => {
                            ArgsType::Param(value.to_typed_literal(&data_type)?)
                        }
                        FetchArg::Literal(literal) => ArgsType::Param(literal),
                        FetchArg::Input(df) => ArgsType::Input(df.relation()?),
                    }),
                })
            })
            .collect::<spark_connect_core::error::Result<_>>()?;
        methods.push(proto::fetch::Method {
            method: method.to_string(),
            args,
        });
        Ok(proto::Fetch {
            obj_ref: Some(self.object_ref()),
            methods,
        })
    }
}

/// An argument of a method of a fetch, as `serialize` of `pyspark.ml.connect.serialize` sends it.
#[derive(Clone)]
pub enum FetchArg {
    /// A value, as `serialize_param` sends it.
    Value(MlValue),
    /// A value of the given type, as PySpark sends a `(value, DataType)` pair.
    Typed(MlValue, DataType),
    /// A literal, as the caller has built it.
    Literal(proto::expression::Literal),
    /// A DataFrame, sent as its relation.
    Input(DataFrame),
}

impl std::fmt::Debug for FetchArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Value(value) => f.debug_tuple("Value").field(value).finish(),
            Self::Typed(value, data_type) => f
                .debug_tuple("Typed")
                .field(value)
                .field(data_type)
                .finish(),
            Self::Literal(literal) => f.debug_tuple("Literal").field(literal).finish(),
            Self::Input(_) => f.write_str("Input(DataFrame)"),
        }
    }
}

/// The result of a fetch, of the result types of `MlCommandResult`.
#[derive(Debug, Clone)]
pub enum FetchOutput {
    /// A value (`param`).
    Value(MlValue),
    /// The ids of new server-side objects (`operator_info.obj_ref`), as PySpark reads a
    /// comma-joined id of the trees of an ensemble: each its own object.
    Objects(Vec<MlRef>),
    /// A string result (`summary`), as of `toString` or `toDebugString`.
    Summary(String),
}

impl FetchOutput {
    /// The value of the result, or an error of another result.
    pub fn into_value(self) -> spark_connect_core::error::Result<MlValue> {
        match self {
            Self::Value(value) => Ok(value),
            other => Err(SparkError::value_msg(format!(
                "expected a value of the fetch, got {other:?}"
            ))),
        }
    }

    /// The single object of the result, or an error of another result.
    pub fn into_object(self) -> spark_connect_core::error::Result<MlRef> {
        match self {
            Self::Objects(mut objects) if objects.len() == 1 => Ok(objects.remove(0)),
            other => Err(SparkError::value_msg(format!(
                "expected one object of the fetch, got {other:?}"
            ))),
        }
    }

    /// The objects of the result, or an error of another result.
    pub fn into_objects(self) -> spark_connect_core::error::Result<Vec<MlRef>> {
        match self {
            Self::Objects(objects) => Ok(objects),
            other => Err(SparkError::value_msg(format!(
                "expected objects of the fetch, got {other:?}"
            ))),
        }
    }

    /// The string of the result, or an error of another result.
    pub fn into_string(self) -> spark_connect_core::error::Result<String> {
        match self {
            Self::Summary(text) => Ok(text),
            Self::Value(MlValue::String(text)) => Ok(text),
            other => Err(SparkError::value_msg(format!(
                "expected a string of the fetch, got {other:?}"
            ))),
        }
    }
}

/// The error class of a summary that the server has lost with its offloaded model.
const MODEL_SUMMARY_LOST: &str = "CONNECT_ML.MODEL_SUMMARY_LOST";

/// Fetch the method `method` of `target`, of the arguments `args`, as `try_remote_call` of
/// `pyspark.ml.util` does: a `Fetch` command of the target and its method chain. When the server
/// has lost a summary (`CONNECT_ML.MODEL_SUMMARY_LOST`), it recreates it of the summary dataset of
/// the target (`CreateSummary`) and fetches again, once.
pub fn fetch(
    target: &MlTarget,
    method: &str,
    args: Vec<FetchArg>,
) -> spark_connect_core::error::Result<FetchOutput> {
    let fetch = target.fetch_proto(method, args)?;
    match run_fetch(target, fetch.clone()) {
        Err(error) if error.get_error_class().as_deref() == Some(MODEL_SUMMARY_LOST) => {
            let (Some(model), Some(dataset)) = (target.root(), target.summary_dataset()) else {
                return Err(error);
            };
            run_ml_command(
                target.session(),
                proto::ml_command::Command::CreateSummary(proto::ml_command::CreateSummary {
                    model_ref: Some(model.object_ref()),
                    dataset: Some(dataset.relation()?),
                }),
            )?;
            run_fetch(target, fetch)
        }
        result => result,
    }
}

fn run_fetch(
    target: &MlTarget,
    fetch: proto::Fetch,
) -> spark_connect_core::error::Result<FetchOutput> {
    use proto::ml_command_result::{ml_operator_info::Type, ResultType};
    let result = run_ml_command(target.session(), proto::ml_command::Command::Fetch(fetch))?;
    match result.and_then(|r| r.result_type) {
        Some(ResultType::Param(literal)) => {
            Ok(FetchOutput::Value(MlValue::from_literal(&literal)?))
        }
        Some(ResultType::Summary(text)) => Ok(FetchOutput::Summary(text)),
        Some(ResultType::OperatorInfo(info)) => match info.r#type {
            Some(Type::ObjRef(object)) => Ok(FetchOutput::Objects(
                object
                    .id
                    .split(',')
                    .filter(|id| !id.is_empty())
                    .map(|id| MlRef::from_id(target.session(), id))
                    .collect(),
            )),
            _ => Err(SparkError::connect_msg(
                "the server returned an operator of no object of the fetch",
            )),
        },
        None => Err(SparkError::connect_msg(
            "the server returned no result of the fetch",
        )),
    }
}

/// Run the ML command `command`, and return its `MlCommandResult`, if the server sends one.
fn run_ml_command(
    session: &SparkSession,
    command: proto::ml_command::Command,
) -> spark_connect_core::error::Result<Option<proto::MlCommandResult>> {
    let responses = crate::dataframe::execute_command_collect(
        session,
        proto::command::CommandType::MlCommand(proto::MlCommand {
            command: Some(command),
        }),
    )?;
    Ok(responses
        .into_iter()
        .find_map(|response| match response.response_type {
            Some(proto::execute_plan_response::ResponseType::MlCommandResult(result)) => {
                Some(result)
            }
            _ => None,
        }))
}

/// The DataFrame of the method `method` of `target`, of the arguments `args`, as
/// `invoke_remote_attribute_relation` of `pyspark.ml.util` reads it: an `MlRelation` of the fetch,
/// whose `model_summary_dataset` is the summary dataset of a summary target. The DataFrame keeps
/// the object of the target alive, and the plans of the summary dataset and of the DataFrame
/// arguments, which may own other cached objects.
pub fn attribute_relation(
    target: &MlTarget,
    method: &str,
    args: Vec<FetchArg>,
) -> spark_connect_core::error::Result<DataFrame> {
    let inputs: Vec<Arc<LogicalPlan>> = args
        .iter()
        .filter_map(|arg| match arg {
            FetchArg::Input(df) => Some(df.plan.clone()),
            _ => None,
        })
        .collect();
    let fetch = target.fetch_proto(method, args)?;
    let relation = proto::Relation {
        rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
            proto::MlRelation {
                ml_type: Some(proto::ml_relation::MlType::Fetch(fetch)),
                model_summary_dataset: target
                    .summary_dataset()
                    .map(|df| df.relation().map(Box::new))
                    .transpose()?,
            },
        ))),
        ..Default::default()
    };
    let owner: Arc<dyn std::any::Any + Send + Sync> =
        Arc::new((target.root().cloned(), inputs.clone()));
    match target.summary_dataset() {
        Some(dataset) => Ok(DataFrame::new(
            target.session().clone(),
            LogicalPlan::MlTransform {
                ml_relation: relation,
                input: dataset.plan.clone(),
                owner: Some(RelationOwner(owner)),
            },
        )),
        None => match inputs.first() {
            Some(input) => Ok(DataFrame::new(
                target.session().clone(),
                LogicalPlan::MlTransform {
                    ml_relation: relation,
                    input: input.clone(),
                    owner: Some(RelationOwner(owner)),
                },
            )),
            None => DataFrame::from_relation(target.session().clone(), relation, Some(owner)),
        },
    }
}

/// Save the operator `operator` (an estimator, a transformer or an evaluator, which is not fitted)
/// of the params `params` to `path`, as `RemoteMLWriter.saveInstance` of
/// `pyspark.ml.connect.readwrite` writes one: a `Write` of the operator.
pub fn write_operator(
    session: &SparkSession,
    operator: &MlOperator,
    params: &Params,
    path: &str,
    overwrite: bool,
    options: &HashMap<String, String>,
) -> spark_connect_core::error::Result<()> {
    run_ml_command(
        session,
        proto::ml_command::Command::Write(proto::ml_command::Write {
            r#type: Some(proto::ml_command::write::Type::Operator(
                operator.to_proto(),
            )),
            params: Some(params.to_proto()),
            path: path.to_string(),
            should_overwrite: Some(overwrite),
            options: options.clone(),
        }),
    )?;
    Ok(())
}

/// Save the server-side object `object` (a fitted model) of the params `params` to `path`, as
/// `RemoteMLWriter.saveInstance` writes a model: a `Write` of its object reference.
pub fn write_object(
    object: &MlRef,
    params: &Params,
    path: &str,
    overwrite: bool,
    options: &HashMap<String, String>,
) -> spark_connect_core::error::Result<()> {
    run_ml_command(
        object.session(),
        proto::ml_command::Command::Write(proto::ml_command::Write {
            r#type: Some(proto::ml_command::write::Type::ObjRef(object.object_ref())),
            params: Some(params.to_proto()),
            path: path.to_string(),
            should_overwrite: Some(overwrite),
            options: options.clone(),
        }),
    )?;
    Ok(())
}

/// A writer of an estimator, transformer, evaluator or fitted model, as PySpark's `MLWriter`:
/// `.session(&spark)` sets the session that saves an operator (a model saves of its own session),
/// `.overwrite()` replaces a saved one, `.option(key, value)` passes an option of the server's
/// writer, and `.save(path)` sends the `Write`.
#[derive(Clone)]
pub struct MlWriter {
    target: WriterTarget,
    session: Option<SparkSession>,
    overwrite: bool,
    options: HashMap<String, String>,
}

#[derive(Clone)]
enum WriterTarget {
    Operator(MlOperator, Params),
    Object(MlRef, Params),
}

impl std::fmt::Debug for MlWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let target = match &self.target {
            WriterTarget::Operator(operator, _) => operator.name.clone(),
            WriterTarget::Object(object, _) => object.id().to_string(),
        };
        f.debug_struct("MlWriter")
            .field("target", &target)
            .field("overwrite", &self.overwrite)
            .field("options", &self.options)
            .finish()
    }
}

impl MlWriter {
    /// The writer of the operator `operator` of the params `params`, which is not fitted.
    pub fn operator(operator: &MlOperator, params: &Params) -> Self {
        Self {
            target: WriterTarget::Operator(operator.clone(), params.clone()),
            session: None,
            overwrite: false,
            options: HashMap::new(),
        }
    }

    /// The writer of the fitted model `object` of the params `params`, of its session.
    pub fn object(object: &MlRef, params: &Params) -> Self {
        Self {
            target: WriterTarget::Object(object.clone(), params.clone()),
            session: Some(object.session().clone()),
            overwrite: false,
            options: HashMap::new(),
        }
    }

    /// The session that saves, as `MLWriter.session` of PySpark.
    pub fn session(mut self, session: &SparkSession) -> Self {
        self.session = Some(session.clone());
        self
    }

    /// Replace a saved operator or model of the path, as `MLWriter.overwrite`.
    pub fn overwrite(mut self) -> Self {
        self.overwrite = true;
        self
    }

    /// An option of the server's writer, as `MLWriter.option`.
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_string(), value.to_string());
        self
    }

    /// Save to `path`, as `MLWriter.save`.
    pub fn save(&self, path: &str) -> spark_connect_core::error::Result<()> {
        match &self.target {
            WriterTarget::Object(object, params) => {
                write_object(object, params, path, self.overwrite, &self.options)
            }
            WriterTarget::Operator(operator, params) => {
                let session = self.session.as_ref().ok_or_else(|| {
                    SparkError::value_msg(format!(
                        "{} needs a session to save with: call .session(&spark) on its writer",
                        operator.name
                    ))
                })?;
                write_operator(
                    session,
                    operator,
                    params,
                    path,
                    self.overwrite,
                    &self.options,
                )
            }
        }
    }
}

/// What the server reads of a saved operator or model: the object of a model, which this client
/// owns, the uid and the params.
#[derive(Debug, Clone)]
pub struct ReadResult {
    /// The object of a model; None of an estimator, a transformer or an evaluator.
    pub object: Option<MlRef>,
    pub uid: String,
    pub params: Params,
}

/// Load the operator or model of the JVM class `jvm_class` of the type `op_type` from `path`, as
/// `RemoteMLReader.loadInstance` of `pyspark.ml.connect.readwrite` reads one: a `Read` of the
/// operator, whose result names the object of a model, its uid and its params.
pub fn read(
    session: &SparkSession,
    jvm_class: &str,
    op_type: OperatorType,
    path: &str,
) -> spark_connect_core::error::Result<ReadResult> {
    use proto::ml_command_result::{ml_operator_info::Type, ResultType};
    let result = run_ml_command(
        session,
        proto::ml_command::Command::Read(proto::ml_command::Read {
            operator: Some(proto::MlOperator {
                name: jvm_class.to_string(),
                uid: String::new(),
                r#type: op_type.to_proto() as i32,
            }),
            path: path.to_string(),
        }),
    )?;
    let Some(ResultType::OperatorInfo(info)) = result.and_then(|r| r.result_type) else {
        return Err(SparkError::connect_msg(format!(
            "the server returned no operator of the read of {jvm_class}"
        )));
    };
    let object = match &info.r#type {
        Some(Type::ObjRef(object)) if op_type == OperatorType::Model => {
            Some(MlRef::from_id(session, &object.id))
        }
        _ => None,
    };
    Ok(ReadResult {
        object,
        uid: info.uid.clone().unwrap_or_default(),
        params: info
            .params
            .as_ref()
            .map(Params::from_proto)
            .unwrap_or_default(),
    })
}

/// A model of a server-side object, of its operator and params, which transforms as every cached
/// model does: the model of an object that `read_model` loads, or that a fetch returns (a tree of an
/// ensemble), whose class this client does not otherwise model.
#[derive(Debug, Clone)]
pub struct ObjectModel {
    operator: MlOperator,
    params: Params,
    object: MlRef,
}

impl ObjectModel {
    /// The model of `object`, of the operator `operator` (of the model's JVM class and uid) and
    /// the params `params`.
    pub fn new(object: MlRef, operator: MlOperator, params: Params) -> Self {
        Self {
            operator,
            params,
            object,
        }
    }
}

impl Transformer for ObjectModel {
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
        self.object.transform(df, &self.params)
    }

    fn build_ml_relation(
        &self,
        input: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.object.build_ml_relation(input, &self.params)
    }
}

impl Model for ObjectModel {
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }

    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.object)
    }
}

/// The target of the training summary of the fitted model `object` (`"<id>.summary"`), of the
/// summary dataset `dataset`, as PySpark's `try_remote_fit` makes it: `dataset` is the training
/// dataset itself for the models whose `_summary_dataset` returns it (the classification models,
/// LinearRegression and GeneralizedLinearRegression), and [`summary_transform`] of it for the others
/// (the clustering models, of the default `_summary_dataset`).
pub fn training_summary(object: &MlRef, dataset: &DataFrame) -> MlTarget {
    MlTarget::object(object)
        .child("summary")
        .with_summary_dataset(dataset)
}

/// The transform of `dataset` by the fitted model `object`, of no params, as PySpark's default
/// `_summary_dataset` (`self.transform(train_dataset)`) transforms the training dataset before the
/// fit copies the estimator's params to the model: the server's model object transforms of its own
/// params.
pub fn summary_transform(
    object: &MlRef,
    dataset: &DataFrame,
) -> spark_connect_core::error::Result<DataFrame> {
    object.transform(dataset, &Params::new())
}

/// The target of the summary of an `evaluate`, of the id of the `summary` result that the server
/// returns, which this client does not release, as PySpark does not.
pub fn evaluated_summary(
    session: &SparkSession,
    output: FetchOutput,
) -> spark_connect_core::error::Result<MlTarget> {
    Ok(MlTarget::unowned(session, &output.into_string()?))
}

/// The estimator, transformer or evaluator of the JVM class `jvm_class` that `read` loads from
/// `path`: its operator, of the uid that it was saved with, and its params.
pub fn read_operator(
    session: &SparkSession,
    jvm_class: &str,
    op_type: OperatorType,
    path: &str,
) -> spark_connect_core::error::Result<(MlOperator, Params)> {
    let read = read(session, jvm_class, op_type, path)?;
    Ok((
        MlOperator::with_uid(jvm_class, &read.uid, op_type),
        read.params,
    ))
}

/// The model of the JVM class `jvm_class` that `read` loads from `path`: its object, its operator,
/// of the uid that it was saved with, and its params.
pub fn read_model(
    session: &SparkSession,
    jvm_class: &str,
    path: &str,
) -> spark_connect_core::error::Result<(MlRef, MlOperator, Params)> {
    let read = read(session, jvm_class, OperatorType::Model, path)?;
    let object = read.object.ok_or_else(|| {
        SparkError::connect_msg(format!(
            "the server returned no object of the read of {jvm_class}"
        ))
    })?;
    Ok((
        object,
        MlOperator::with_uid(jvm_class, &read.uid, OperatorType::Model),
        read.params,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ml_values_round_trip_through_their_literals() {
        let values = vec![
            MlValue::Null,
            MlValue::Bool(true),
            MlValue::Int(7),
            MlValue::Long(1 << 40),
            MlValue::Double(0.5),
            MlValue::String("s".into()),
            MlValue::Array(vec![MlValue::Double(1.0), MlValue::Double(2.5)]),
            MlValue::Array(vec![MlValue::String("a".into())]),
            MlValue::Vector(MlVector::Dense(vec![1.0, 2.0])),
            MlValue::Vector(MlVector::Sparse {
                size: 3,
                indices: vec![1],
                values: vec![2.0],
            }),
            MlValue::Matrix(MlMatrix::Dense {
                num_rows: 1,
                num_cols: 2,
                values: vec![0.0, -1.0],
                is_transposed: false,
            }),
        ];
        for value in values {
            let literal = value.to_literal().unwrap();
            assert_eq!(
                MlValue::from_literal(&literal).unwrap(),
                value,
                "{literal:?}"
            );
        }
    }

    #[test]
    fn ml_values_type_arrays_by_their_first_element_or_as_given() {
        let error = MlValue::Array(Vec::new()).to_literal().unwrap_err();
        assert_eq!(
            error.get_error_class().as_deref(),
            Some("CANNOT_INFER_ARRAY_ELEMENT_TYPE")
        );
        let typed = MlValue::Array(Vec::new())
            .to_typed_literal(&DataType::Array {
                element_type: Box::new(DataType::String {
                    collation: "UTF8_BINARY".to_string(),
                }),
                contains_null: true,
            })
            .unwrap();
        assert_eq!(
            MlValue::from_literal(&typed).unwrap(),
            MlValue::Array(Vec::new())
        );
        assert_eq!(MlValue::Int(3).as_f64(), Some(3.0));
        assert_eq!(MlValue::Long(4).as_i64(), Some(4));
    }

    #[test]
    fn ml_targets_chain_their_methods_as_pyspark_names_them() {
        let session = crate::session::SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let model = MlRef::from_id(&session, "abc");
        let summary = MlTarget::object(&model).child("summary");
        assert_eq!(summary.id(), "abc.summary");
        let fetch = summary
            .fetch_proto("accuracy", vec![FetchArg::Value(MlValue::Int(1))])
            .unwrap();
        assert_eq!(fetch.obj_ref.unwrap().id, "abc");
        let names: Vec<_> = fetch.methods.iter().map(|m| m.method.as_str()).collect();
        assert_eq!(names, ["summary", "accuracy"]);
        assert_eq!(MlTarget::helper(&session).id(), ML_CONNECT_HELPER_ID);
        std::mem::forget(model);
    }

    #[test]
    fn training_summaries_object_models_and_evaluated_summaries_are_of_their_objects() {
        let session = crate::session::SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let object = MlRef::from_id(&session, "fitted");
        let train = session.range(3).expect("range");

        // The training summary is the summary of the object, of the summary dataset.
        let summary = training_summary(&object, &train);
        assert_eq!(summary.id(), "fitted.summary");
        assert!(summary.summary_dataset().is_some());

        // A model of the object transforms of its params, as a cached model does.
        let operator = MlOperator::with_uid(
            "org.apache.spark.ml.clustering.KMeansModel",
            "KMeans_0123456789ab",
            OperatorType::Model,
        );
        let model = ObjectModel::new(
            object.clone(),
            operator,
            Params::new().set_param_int("k", 2),
        );
        assert_eq!(model.ml_ref().map(MlRef::id), Some("fitted"));
        assert_eq!(model.clone_box().operator().uid, "KMeans_0123456789ab");
        let relation = model.build_ml_relation(&train).expect("relation");
        let Some(proto::ml_relation::MlType::Transform(transform)) = relation.ml_type else {
            panic!("a transform: {relation:?}");
        };
        let Some(proto::ml_relation::transform::Operator::ObjRef(obj_ref)) = transform.operator
        else {
            panic!("of the object");
        };
        assert_eq!(obj_ref.id, "fitted");
        assert!(transform.params.expect("params").params.contains_key("k"));

        // The summary of an evaluate is the server's id; any other result is an error.
        let evaluated =
            evaluated_summary(&session, FetchOutput::Summary("evaluated".into())).unwrap();
        assert_eq!(evaluated.id(), "evaluated");
        assert!(evaluated_summary(&session, FetchOutput::Value(MlValue::Int(1))).is_err());
        std::mem::forget((model, object));
    }
}
