// SPDX-License-Identifier: Apache-2.0
//! The objects of Spark Connect ML that the server holds, and the commands that read, save and
//! load them, as PySpark 4.3.0 sends them over Spark Connect:
//!
//! - [`MlRef`]: a fitted model or another server-side object that a fetch returns (a submodel, a
//!   tree of an ensemble), released on the server when its last owner drops, as PySpark's
//!   `RemoteModelRef`;
//! - [`MlTarget`] and [`fetch`]: a method of an object, or of an object that a method chain reaches
//!   (`<model>.summary`), whose value the server returns (`try_remote_call` of `pyspark.ml.util`);
//! - [`attribute_relation`]: a method whose value is a DataFrame
//!   (`invoke_remote_attribute_relation`);
//! - [`MlValue`]: the value of a fetch, as PySpark's `deserialize_param` reads it;
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

    /// The warning that the server sent with the object, the model of a fit, of a tree training
    /// that the memory control of its ML cache stopped early (see [`Model::fit_warning`]). None of
    /// a fit without a warning, and of an object that a fetch or a read returns.
    pub fn fit_warning(&self) -> Option<&str> {
        self.0.fit_warning.as_deref()
    }

    /// An object that the server returned by its id (an `operator_info` of a fetch or a read), of
    /// which this client is the first owner.
    pub(crate) fn from_id(session: &SparkSession, id: &str) -> Self {
        Self(Arc::new(CachedModelRef {
            model_ref: proto::ObjectRef { id: id.to_string() },
            session: session.clone(),
            fit_warning: None,
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
    /// A struct of its field names and values: a struct literal that is neither a vector nor a
    /// matrix, for which PySpark's `deserialize_param` raises `ValueError("Unknown UDT ...")`.
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
                    _ => DECIMAL_TYPE_PRECISION_SCALE,
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
            Self::Decimal(_) => {
                let (precision, scale) = DECIMAL_TYPE_PRECISION_SCALE;
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

/// The precision and scale of a decimal value of no type, as `LiteralExpression._infer_type` types
/// a `decimal.Decimal`: `DecimalType()`, of precision 10 and scale 0 whatever the digits of the
/// value, which the server widens to them. PySpark's `serialize_param` sends every decimal so,
/// including one that a server returned and a loaded param or a fetch argument sends again.
const DECIMAL_TYPE_PRECISION_SCALE: (i32, i32) = (10, 0);

/// The object of a fetch: a server-side object, or the ML helper, and the chain of methods that
/// reaches the object of which a method is fetched, as `_extract_id_methods` of
/// `pyspark.ml.connect.util` splits `"<id>.summary"`. A summary target keeps the dataset that
/// recreates the summary when the server has offloaded the model (`CreateSummary`).
#[derive(Clone)]
pub struct MlTarget {
    root: TargetRoot,
    chain: Vec<String>,
    summary_dataset: Option<DataFrame>,
    /// A target that this one keeps alive, as PySpark's summary of an `evaluate` keeps its model
    /// (`__source_transformer__`).
    keep_alive: Option<Box<MlTarget>>,
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
            keep_alive: None,
        }
    }

    /// The object of the id `id` of `session`, which this client does not release, as PySpark's
    /// summary of an `evaluate` (a `summary` result of a fetch).
    pub fn unowned(session: &SparkSession, id: &str) -> Self {
        Self {
            root: TargetRoot::Unowned(session.clone(), id.to_string()),
            chain: Vec::new(),
            summary_dataset: None,
            keep_alive: None,
        }
    }

    /// The ML helper of `session`.
    pub fn helper(session: &SparkSession) -> Self {
        Self {
            root: TargetRoot::Helper(session.clone()),
            chain: Vec::new(),
            summary_dataset: None,
            keep_alive: None,
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
    /// The target, which keeps `other` alive (its object, which the server keeps while a target
    /// holds it), as PySpark's summary of an `evaluate` keeps its model.
    pub fn keeping(mut self, other: &MlTarget) -> Self {
        self.keep_alive = Some(Box::new(other.clone()));
        self
    }

    pub fn with_summary_dataset(mut self, dataset: &DataFrame) -> Self {
        self.summary_dataset = Some(dataset.clone());
        self
    }

    /// The dataset of the summary of this target, if any.
    pub fn summary_dataset(&self) -> Option<&DataFrame> {
        self.summary_dataset.as_ref()
    }

    /// The references of this target to the parts of plans, of the plans of its summary dataset
    /// and of those of the targets that it keeps alive (see [`crate::gc`]). Not a stable API.
    #[doc(hidden)]
    pub fn held_parts(&self) -> Vec<crate::gc::PartRef<'_>> {
        let mut held = Vec::new();
        let mut target = Some(self);
        while let Some(of) = target {
            if let Some(dataset) = &of.summary_dataset {
                held.extend(dataset.held_parts());
            }
            target = of.keep_alive.as_deref();
        }
        held
    }

    /// What PySpark's pickle of the object of this target, the `__source_instance__` of the plan
    /// of a DataFrame of one of its methods, reaches first that does not pickle, after the child of
    /// the plan, its summary dataset (see [`pickle_block`]): of a model, the lock of its
    /// `RemoteModelRef`; of the training summary of a fit, the session of its `_summary_dataset`;
    /// of the summary of an evaluation, the lock of the model that it keeps
    /// (`__source_transformer__`). The ML helper, and a summary of PySpark's constructor, of no
    /// object of this client, reach none.
    fn instance_pickle_block(&self) -> Option<MlPickleBlock> {
        match (&self.root, self.chain.is_empty(), &self.summary_dataset) {
            (TargetRoot::Object(_), true, _) => Some(MlPickleBlock::ModelLock),
            (TargetRoot::Object(_), false, Some(dataset)) => {
                Some(MlPickleBlock::SummarySession(dataset.clone()))
            }
            (TargetRoot::Object(_), false, None) => Some(MlPickleBlock::ModelLock),
            _ => self.keep_alive.as_ref().map(|_| MlPickleBlock::ModelLock),
        }
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

/// What PySpark's pickle of a DataFrame reaches first that does not pickle, of the objects of
/// `pyspark.ml` that its plan keeps, which `pickle` and `copy.deepcopy` of PySpark's DataFrame read
/// before its session (see [`pickle_block`]). Not a stable API.
#[doc(hidden)]
#[derive(Clone)]
pub enum MlPickleBlock {
    /// The lock of the `RemoteModelRef` of a model that the plan keeps: of a transform of the
    /// model (`__source_transformer__`), of a DataFrame of a method of the model, or of the
    /// summary of an evaluation, which keeps its model (`__source_instance__`).
    ModelLock,
    /// The session of the `_summary_dataset` of the training summary of a fit, of a DataFrame of
    /// a method of the summary (`__source_instance__`), which PySpark's pickle refuses (a local
    /// function of its `__reduce__`) before the lock of the model of the summary
    /// (`_remote_model_obj`), which comes next: cloudpickle, which pickles the function, and
    /// `copy.deepcopy` reach that lock, a deep copy after a copy of the summary dataset, which it
    /// holds.
    SummarySession(DataFrame),
}

impl std::fmt::Debug for MlPickleBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelLock => f.write_str("ModelLock"),
            Self::SummarySession(_) => f.write_str("SummarySession"),
        }
    }
}

/// What PySpark's pickle of the DataFrame `df` reaches first that does not pickle, of the objects
/// of `pyspark.ml` that its plan keeps, in the order that PySpark's pickle of the plan takes them:
/// the child of a plan first, as its attribute `_child` comes first, then the plans that the
/// subqueries of its expressions refer to (`exists()`, `scalar()`, `isin(df)`), as its
/// `_references` come next, and then the objects that the plan keeps, of each plan in the order of
/// its inputs. A transform of a model keeps the model (`__source_transformer__`); a DataFrame of a
/// method of a model or of a summary keeps that object (`__source_instance__`), and the summary
/// dataset of a summary as its child, but not the DataFrames of its arguments, which PySpark's plan
/// holds as their relations. Not a stable API.
#[doc(hidden)]
pub fn pickle_block(df: &DataFrame) -> Option<MlPickleBlock> {
    pickle_block_with(df, &mut |_| None)
}

/// [`pickle_block`] of the DataFrame `df`, of the block that `owned` gives of the owner of a
/// relation built outside the core (`DataFrame::from_relation`), such as the Python plan of
/// `pyspark.ml` of the Python client, whose pickle reaches what PySpark's pickle of the plan
/// reaches, where the walk reaches the relation. Not a stable API.
#[doc(hidden)]
pub fn pickle_block_with(
    df: &DataFrame,
    owned: &mut dyn FnMut(&(dyn std::any::Any + Send + Sync)) -> Option<MlPickleBlock>,
) -> Option<MlPickleBlock> {
    // A walk of its own stack, as a plan may be deeper than the thread's stack, which leaves a plan
    // after its inputs and the plans of its subqueries, and visits a plan that several plans read
    // (a self-join) once.
    enum Step<'a> {
        Enter(&'a Arc<LogicalPlan>),
        Leave(&'a LogicalPlan),
    }
    let mut steps = vec![Step::Enter(&df.plan)];
    let mut visited = std::collections::HashSet::new();
    while let Some(step) = steps.pop() {
        match step {
            Step::Enter(plan) => {
                if !visited.insert(Arc::as_ptr(plan)) {
                    continue;
                }
                steps.push(Step::Leave(plan.as_ref()));
                // The inputs that PySpark's plan pickles, of the attribute relations only the
                // summary dataset, which are entered in their order, and then the plans of the
                // subqueries of its expressions.
                let inputs = match relation_target_of(plan) {
                    Some(target) if target.target.summary_dataset.is_none() => Vec::new(),
                    _ => plan.inputs(),
                };
                let subqueries = crate::plan::subquery_plans(plan);
                steps.extend(subqueries.into_iter().rev().map(Step::Enter));
                steps.extend(inputs.into_iter().rev().map(Step::Enter));
            }
            Step::Leave(plan) => {
                let owner = match plan {
                    LogicalPlan::MlTransform {
                        owner: Some(owner), ..
                    }
                    | LogicalPlan::Raw {
                        owner: Some(owner), ..
                    } => owner,
                    _ => continue,
                };
                if owner.0.downcast_ref::<CachedModelRef>().is_some() {
                    return Some(MlPickleBlock::ModelLock);
                }
                if let Some(relation) = owner.0.downcast_ref::<RelationTarget>() {
                    if let Some(block) = relation.target.instance_pickle_block() {
                        return Some(block);
                    }
                } else if let Some(block) = owned(owner.0.as_ref()) {
                    return Some(block);
                }
            }
        }
    }
    None
}

/// The target of the attribute relation of `plan` (see [`attribute_relation`]), if it is one.
fn relation_target_of(plan: &LogicalPlan) -> Option<&RelationTarget> {
    match plan {
        LogicalPlan::MlTransform {
            owner: Some(owner), ..
        }
        | LogicalPlan::Raw {
            owner: Some(owner), ..
        } => owner.0.downcast_ref::<RelationTarget>(),
        _ => None,
    }
}

/// What the DataFrame of an attribute relation keeps alive, as PySpark's plan keeps the object of
/// its method (`__source_instance__`): the target, of its object, its summary dataset and the model
/// that the summary of an evaluation keeps, and the plans of the DataFrames of its arguments.
struct RelationTarget {
    target: MlTarget,
    #[allow(dead_code)] // Kept alive, as the relation refers to their plans.
    inputs: Vec<Arc<LogicalPlan>>,
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
    let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(RelationTarget {
        target: target.clone(),
        inputs: inputs.clone(),
    });
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
    write_operator_opt(session, operator, params, path, Some(overwrite), options)
}

/// [`write_operator`] of an overwrite that may be unset: `None` leaves the `should_overwrite` of
/// the `Write` unset, as PySpark's `MlCommand.Write(should_overwrite=None)` does of a writer whose
/// `shouldOverwrite` is None (which the server reads as false). Not a stable API.
#[doc(hidden)]
pub fn write_operator_opt(
    session: &SparkSession,
    operator: &MlOperator,
    params: &Params,
    path: &str,
    overwrite: Option<bool>,
    options: &HashMap<String, String>,
) -> spark_connect_core::error::Result<()> {
    run_ml_command(
        session,
        proto::ml_command::Command::Write(proto::ml_command::Write {
            r#type: Some(proto::ml_command::write::Type::Operator(
                operator.to_proto(),
            )),
            params: Some(params.sendable()?),
            path: path.to_string(),
            should_overwrite: overwrite,
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
    write_object_opt(object, params, path, Some(overwrite), options)
}

/// [`write_object`] of an overwrite that may be unset (see [`write_operator_opt`]). Not a stable
/// API.
#[doc(hidden)]
pub fn write_object_opt(
    object: &MlRef,
    params: &Params,
    path: &str,
    overwrite: Option<bool>,
    options: &HashMap<String, String>,
) -> spark_connect_core::error::Result<()> {
    run_ml_command(
        object.session(),
        proto::ml_command::Command::Write(proto::ml_command::Write {
            r#type: Some(proto::ml_command::write::Type::ObjRef(object.object_ref())),
            params: Some(params.sendable()?),
            path: path.to_string(),
            should_overwrite: overwrite,
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
    /// A meta-algorithm of PySpark's Python format (a pipeline, a validator, their models), which
    /// its own save writes, of the session of the writer.
    Meta(std::sync::Arc<dyn MetaSave>),
}

/// The save of a meta-algorithm of PySpark's Python format to a path, of a session and the
/// writer's options, as its `MLWriter.saveImpl` (see [`MlWriter::meta`]).
pub trait MetaSave: Send + Sync {
    /// The uid of the meta-algorithm, of the writer's `Debug`.
    fn name(&self) -> String;

    /// Save the meta-algorithm to `path`, of `session`, of the writer's `options`.
    fn save_meta(
        &self,
        session: &SparkSession,
        path: &str,
        options: &HashMap<String, String>,
    ) -> spark_connect_core::error::Result<()>;
}

impl std::fmt::Debug for MlWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let target = match &self.target {
            WriterTarget::Operator(operator, _) => operator.name.clone(),
            WriterTarget::Object(object, _) => object.id().to_string(),
            WriterTarget::Meta(meta) => meta.name(),
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

    /// The writer of a meta-algorithm of PySpark's Python format, which `meta` saves (see
    /// [`MetaSave`]), of a session that [`Self::session`] sets.
    pub fn meta(meta: std::sync::Arc<dyn MetaSave>) -> Self {
        Self {
            target: WriterTarget::Meta(meta),
            session: None,
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

    /// An option of the server's writer, as `MLWriter.option`, whose key is lowercased as
    /// PySpark's `optionMap[key.lower()]` does, so two spellings of a key are one option.
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_lowercase(), value.to_string());
        self
    }

    /// Save to `path`, as `MLWriter.save`.
    pub fn save(&self, path: &str) -> spark_connect_core::error::Result<()> {
        match &self.target {
            WriterTarget::Object(object, params) => {
                write_object(object, params, path, self.overwrite, &self.options)
            }
            WriterTarget::Meta(meta) => {
                let session = self.session.as_ref().ok_or_else(|| {
                    SparkError::value_msg(format!(
                        "{} needs a session to save with: call .session(&spark) on its writer",
                        meta.name()
                    ))
                })?;
                // As `RemoteMLWriter.saveInstance` of a meta-algorithm: the ML helper deletes the
                // path of an overwrite first.
                if self.overwrite {
                    handle_overwrite(session, path)?;
                }
                meta.save_meta(session, path, &self.options)
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
        params: read_params(info.params.as_ref()),
    })
}

/// The params that a `Read` returns, as PySpark sends them again: its load turns each param into a
/// Python value (`deserialize_param`), and its next save, fit or transform serializes that value
/// with the inference of `serialize_param`, of Python's one int and one float. So an integral
/// literal is an `Integer` when it fits in 32 bits, and a `Long` when it does not, and a `Float` is
/// a `Double`: a `seed` that the server reads back as a `Long` is saved again as an `Integer`, as
/// PySpark saves it. The other literals are serialized again as the value they read as, as
/// `serialize_param` infers it (a string array of UTF8_BINARY elements, a vector without its
/// Python class), and one that reads as no value is kept as it is.
fn read_params(params: Option<&proto::MlParams>) -> Params {
    use proto::expression::literal::LiteralType as L;
    let Some(params) = params else {
        return Params::default();
    };
    let mut params = params.clone();
    for literal in params.params.values_mut() {
        let retyped = match literal.literal_type {
            Some(L::Byte(value)) | Some(L::Short(value)) => Some(L::Integer(value)),
            Some(L::Long(value)) => i32::try_from(value).ok().map(L::Integer),
            Some(L::Float(value)) => Some(L::Double(f64::from(value))),
            _ => None,
        };
        // A literal of the new type, as PySpark builds it of the Python value. A Long that does not
        // fit in 32 bits is a Python int of the Long range, and so are the other literals, inferred
        // by their values.
        let inferred = match retyped {
            Some(literal_type) => proto::expression::Literal {
                literal_type: Some(literal_type),
                ..Default::default()
            },
            None => match MlValue::from_literal(literal).and_then(|value| value.to_literal()) {
                Ok(inferred) => inferred,
                Err(_) => continue,
            },
        };
        *literal = inferred;
    }
    Params::from_proto(&params)
}

/// A model of a server-side object, of its operator and params, which transforms as every cached
/// model does: the model of an object that `read_model` loads, or that a fetch returns (a tree of
/// an ensemble), whose class this client does not otherwise model.
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
/// LinearRegression and GeneralizedLinearRegression), and [`summary_transform`] of it for the
/// others (the clustering models, of the default `_summary_dataset`).
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

/// The target of the summary of an `evaluate` of the model `model`, of the id of the `summary`
/// result that the server returns, which this client does not release, as PySpark does not, and
/// which keeps the model alive, as PySpark's summary keeps its `__source_transformer__`.
pub fn evaluated_summary(
    model: &MlTarget,
    output: FetchOutput,
) -> spark_connect_core::error::Result<MlTarget> {
    let summary = MlTarget::unowned(model.session(), &output.into_string()?);
    Ok(summary.keeping(model))
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
    #[allow(deprecated)] // The element and struct types of the literals of PySpark.
    fn read_params_are_typed_as_pyspark_sends_them_again() {
        use proto::expression::literal::LiteralType as L;
        let literal = |literal_type| proto::expression::Literal {
            literal_type: Some(literal_type),
            ..Default::default()
        };
        // Just past the 32 bits of an Integer, on either side.
        let (above, below) = (i64::from(i32::MAX) + 1, i64::from(i32::MIN) - 1);
        let cases = [
            ("seed", L::Long(7), L::Integer(7)),
            ("max", L::Long(i64::from(i32::MAX)), L::Integer(i32::MAX)),
            ("min", L::Long(i64::from(i32::MIN)), L::Integer(i32::MIN)),
            ("above", L::Long(above), L::Long(above)),
            ("below", L::Long(below), L::Long(below)),
            ("short", L::Short(3), L::Integer(3)),
            ("byte", L::Byte(-2), L::Integer(-2)),
            ("int", L::Integer(5), L::Integer(5)),
            ("float", L::Float(0.5), L::Double(0.5)),
            ("double", L::Double(0.25), L::Double(0.25)),
            ("string", L::String("s".into()), L::String("s".into())),
            ("bool", L::Boolean(true), L::Boolean(true)),
        ];
        let mut sent = proto::MlParams::default();
        for (name, read, _) in &cases {
            sent.params.insert(name.to_string(), literal(read.clone()));
        }
        // A typed Long: PySpark's literal of the Python int has no type of its own.
        sent.params.insert(
            "typed".into(),
            proto::expression::Literal {
                literal_type: Some(L::Long(9)),
                data_type: Some(DataType::Long.to_proto()),
            },
        );
        // An array is kept as it is, elements and all: an int array param is an Array[Int] on the
        // server, which sends Integer elements, as PySpark re-infers them (no pyspark.ml param is
        // an array of longs).
        let array = literal(L::Array(proto::expression::literal::Array {
            elements: vec![literal(L::Integer(1))],
            ..Default::default()
        }));
        sent.params.insert("array".into(), array.clone());
        let info = proto::ml_command_result::MlOperatorInfo {
            params: Some(sent),
            ..Default::default()
        };

        let params = read_params(info.params.as_ref()).to_proto();
        for (name, _, expected) in cases {
            assert_eq!(params.params[name], literal(expected), "{name}");
        }
        assert_eq!(params.params["typed"], literal(L::Integer(9)));
        // Serialized again as its elements infer it: the element type of the first one.
        let inferred = literal(L::Array(proto::expression::literal::Array {
            element_type: Some(DataType::Integer.to_proto()),
            elements: vec![literal(L::Integer(1))],
        }));
        assert_eq!(params.params["array"], inferred);
        assert_eq!(params.params.len(), 14);
        assert_eq!(read_params(None).to_proto(), proto::MlParams::default());
    }

    #[test]
    #[allow(deprecated)] // The element and struct types of the literals of PySpark.
    fn read_params_serialize_other_values_again_as_pyspark_infers_them() {
        use proto::expression::literal::{Array, LiteralType as L};
        let string = |word: &str| proto::expression::Literal {
            literal_type: Some(L::String(word.into())),
            ..Default::default()
        };
        // The stopWords of the StopWordsRemover golden (feature_models_persistence.json): the
        // server reads its string array without a collation, which PySpark's save sends as
        // UTF8_BINARY.
        let string_array = |collation: &str| proto::expression::Literal {
            literal_type: Some(L::Array(Array {
                element_type: Some(
                    DataType::String {
                        collation: collation.into(),
                    }
                    .to_proto(),
                ),
                elements: vec![string("the"), string("gamma")],
            })),
            ..Default::default()
        };
        // A dense vector that the server reads with its Python class, which PySpark's save does
        // not send.
        let mut vector = MlValue::Vector(MlVector::Dense(vec![1.0, 2.0]))
            .to_literal()
            .unwrap();
        let Some(L::Struct(vector_struct)) = vector.literal_type.as_mut() else {
            panic!("not a vector: {vector:?}");
        };
        let Some(proto::data_type::Kind::Udt(udt)) = vector_struct
            .struct_type
            .as_mut()
            .and_then(|data_type| data_type.kind.as_mut())
        else {
            panic!("not a UDT: {vector_struct:?}");
        };
        udt.python_class = Some("pyspark.ml.linalg.DenseVector".into());
        let mut sent = proto::MlParams::default();
        sent.params.insert("stopWords".into(), string_array(""));
        sent.params.insert("vector".into(), vector);
        let params = read_params(Some(&sent)).to_proto();
        assert_eq!(params.params["stopWords"], string_array("UTF8_BINARY"));
        let expected = MlValue::Vector(MlVector::Dense(vec![1.0, 2.0]))
            .to_literal()
            .unwrap();
        assert_eq!(params.params["vector"], expected);
    }

    #[test]
    fn writer_option_keys_are_lowercased_as_pyspark_does() {
        let operator = MlOperator::with_uid(
            "org.apache.spark.ml.feature.StopWordsRemover",
            "uid",
            OperatorType::Model,
        );
        let writer = MlWriter::operator(&operator, &Params::default())
            .option("Foo", "1")
            .option("foo", "2");
        assert_eq!(writer.options.len(), 1);
        assert_eq!(writer.options["foo"], "2");
    }

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
    fn decimal_values_are_sent_as_decimal_type_as_pyspark_sends_them() {
        use proto::expression::literal::LiteralType as L;
        // PySpark's `_infer_type` of a decimal.Decimal is DecimalType(): precision 10 and scale 0,
        // whatever the digits of the value, and the server widens it.
        for value in ["1.25", "-0.5", "12345678901.5", "100"] {
            let literal = MlValue::Decimal(value.into()).to_literal().unwrap();
            let Some(L::Decimal(decimal)) = literal.literal_type else {
                panic!("not a decimal: {literal:?}");
            };
            assert_eq!(decimal.value, value);
            assert_eq!((decimal.precision, decimal.scale), (Some(10), Some(0)));
        }
        // A `(value, DecimalType(p, s))` pair is the literal of its type.
        let typed = MlValue::Decimal("1.25".into())
            .to_typed_literal(&DataType::Decimal {
                precision: 12,
                scale: 3,
            })
            .unwrap();
        let Some(L::Decimal(decimal)) = typed.literal_type else {
            panic!("not a decimal: {typed:?}");
        };
        assert_eq!((decimal.precision, decimal.scale), (Some(12), Some(3)));
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
    fn the_pickle_blocks_of_plans_are_those_of_pysparks_plans() {
        // As PySpark's pickle of a DataFrame, which pickles its plan, the child of a plan first,
        // before its session: the transform of a model keeps the model, whose lock comes first, of
        // a DataFrame of it and of a self-join too; so does a DataFrame of a method of a model or
        // of the summary of an evaluation, which keeps its model. A DataFrame of a method of a
        // training summary reaches the session of its summary dataset first, unless that dataset
        // reads a model. The DataFrame of the ML helper, whose arguments PySpark's plan holds as
        // their relations, the training dataset and the transform of a transformer keep none.
        use crate::plan::JoinType;

        fn block(df: &DataFrame) -> Option<&'static str> {
            pickle_block(df).map(|block| match block {
                MlPickleBlock::ModelLock => "lock",
                MlPickleBlock::SummarySession(_) => "session",
            })
        }

        let session = crate::session::SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let object = MlRef::from_id(&session, "fitted");
        let train = session.range(3).expect("range");
        let transformed = summary_transform(&object, &train).expect("transform");
        assert_eq!(block(&transformed), Some("lock"));
        assert_eq!(block(&transformed.limit(1)), Some("lock"));
        assert_eq!(
            block(&train.join(&transformed, None, JoinType::Cross)),
            Some("lock")
        );
        let self_joined = transformed.join(&transformed, None, JoinType::Cross);
        assert_eq!(block(&self_joined), Some("lock"));
        assert_eq!(block(&train), None);
        assert_eq!(block(&train.join(&train, None, JoinType::Cross)), None);
        let binarized = crate::ml::Binarizer::new()
            .transform(&train)
            .expect("transform");
        assert_eq!(block(&binarized), None);

        let relation = |target: &MlTarget, args: Vec<FetchArg>| {
            attribute_relation(target, "predictions", args).expect("relation")
        };
        let model = MlTarget::object(&object);
        assert_eq!(block(&relation(&model, Vec::new())), Some("lock"));
        let summary = training_summary(&object, &train);
        let of_summary = relation(&summary, Vec::new());
        assert_eq!(block(&of_summary), Some("session"));
        let Some(MlPickleBlock::SummarySession(dataset)) = pickle_block(&of_summary) else {
            panic!("of the session of the summary dataset");
        };
        assert!(Arc::ptr_eq(&dataset.plan, &train.plan));
        let of_transform = training_summary(&object, &transformed);
        assert_eq!(block(&relation(&of_transform, Vec::new())), Some("lock"));
        let evaluated =
            evaluated_summary(&model, FetchOutput::Summary("evaluated".into())).expect("summary");
        assert_eq!(block(&relation(&evaluated, Vec::new())), Some("lock"));
        let helper = MlTarget::helper(&session);
        let of_helper = relation(&helper, vec![FetchArg::Input(transformed.clone())]);
        assert_eq!(block(&of_helper), None);
        // A DataFrame that reads a model only through a subquery of its expressions, as the
        // references of PySpark's plan keep the plans of the subqueries, which its pickle takes
        // after its child: an EXISTS, a scalar and an IN subquery, also of a summary.
        assert_eq!(block(&train.where_(transformed.exists())), Some("lock"));
        assert_eq!(block(&train.select([transformed.scalar()])), Some("lock"));
        let isin = crate::functions::col("id").isin_subquery(&transformed);
        assert_eq!(block(&train.where_(isin)), Some("lock"));
        assert_eq!(block(&train.where_(of_summary.exists())), Some("session"));
        assert_eq!(block(&train.where_(train.exists())), None);
        std::mem::forget((transformed, object, evaluated, model, summary, of_transform));
    }

    #[test]
    fn the_owners_of_relations_built_outside_the_core_give_their_pickle_blocks() {
        // The block that `pickle_block_with` gives of the owner of a relation of
        // `DataFrame::from_relation` (a Python plan of `pyspark.ml` of the Python client), where
        // the walk reaches it: of a DataFrame over it too, and after the inputs before it.
        fn block(df: &DataFrame, of_owner: Option<&'static str>) -> Option<&'static str> {
            let mut owned = |owner: &(dyn std::any::Any + Send + Sync)| {
                owner
                    .downcast_ref::<&'static str>()
                    .and(of_owner)
                    .map(|_| MlPickleBlock::ModelLock)
            };
            pickle_block_with(df, &mut owned).map(|block| match block {
                MlPickleBlock::ModelLock => "lock",
                MlPickleBlock::SummarySession(_) => "session",
            })
        }

        let session = crate::session::SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session");
        let relation = session
            .range(1)
            .expect("range")
            .to_relation()
            .expect("relation");
        let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new("python plan");
        let built = DataFrame::from_relation(session.clone(), relation, Some(owner)).unwrap();
        assert_eq!(block(&built, Some("lock")), Some("lock"));
        assert_eq!(block(&built.limit(1), Some("lock")), Some("lock"));
        assert_eq!(block(&built, None), None);
        assert!(pickle_block(&built).is_none());
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
        let held = std::sync::Arc::strong_count(&object.0);
        let model_target = MlTarget::object(&object);
        let evaluated =
            evaluated_summary(&model_target, FetchOutput::Summary("evaluated".into())).unwrap();
        assert_eq!(evaluated.id(), "evaluated");
        assert!(evaluated_summary(&model_target, FetchOutput::Value(MlValue::Int(1))).is_err());
        // The summary keeps the model's object alive.
        drop(model_target);
        assert_eq!(std::sync::Arc::strong_count(&object.0), held + 1);
        std::mem::forget(evaluated);
        std::mem::forget((model, object));
    }
}
