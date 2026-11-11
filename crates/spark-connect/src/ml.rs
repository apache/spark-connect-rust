//! ML module mirroring `pyspark.ml.connect.base` and related ML classes.
//!
//! This module provides the foundation for machine learning in Spark Connect:
//! - `Params`: parameter management for ML operators
//! - `MlOperator`: represents ML operators (estimators, transformers, models, evaluators)
//! - Base traits: `Estimator`, `Transformer`, `Model`, `Evaluator`
//! - Concrete example: `StandardScaler` estimator + `StandardScalerModel`

use spark_connect_proto as proto;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::dataframe::DataFrame;
use crate::expression::LiteralExpression;
use crate::plan::{LogicalPlan, RelationOwner};
use crate::session::SparkSession;
use crate::types::DataType;

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
                collation: String::new(),
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
pub trait Model: Transformer {
    /// Clone this model into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Model>;
}

/// Base trait for ML Evaluators that measure model quality.
///
/// Mirrors `pyspark.ml.connect.base.Evaluator`.
pub trait Evaluator: Send + Sync {
    /// Get the operator definition for this evaluator.
    fn operator(&self) -> &MlOperator;

    /// Get the parameters of this evaluator.
    fn params(&self) -> &Params;

    /// Evaluate a DataFrame and return a metric value.
    fn evaluate(&self, _df: &DataFrame) -> spark_connect_core::error::Result<f64>;
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

    /// Set the input column name.
    pub fn set_input_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("inputCol", col);
        self
    }

    /// Get the input column name (default: "features").
    pub fn input_col(&self) -> &str {
        self.params.string_or("inputCol", "features")
    }

    /// Set the output column name.
    pub fn set_output_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("outputCol", col);
        self
    }

    /// Get the output column name (default: "scaled_features").
    pub fn output_col(&self) -> &str {
        self.params.string_or("outputCol", "scaled_features")
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
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
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

    pub fn set_input_cols(mut self, cols: Vec<&str>) -> Self {
        self.params = self.params.set_param_string_array("inputCols", cols);
        self
    }

    /// The input columns (default: none).
    pub fn input_cols(&self) -> Vec<String> {
        self.params.string_array("inputCols")
    }

    pub fn set_output_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("outputCol", col);
        self
    }

    /// The output column (default: "assembled").
    pub fn output_col(&self) -> &str {
        self.params.string_or("outputCol", "assembled")
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

    pub fn set_input_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("inputCol", col);
        self
    }

    /// The input column (default: empty).
    pub fn input_col(&self) -> &str {
        self.params.string_or("inputCol", "")
    }

    pub fn set_output_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("outputCol", col);
        self
    }

    /// The output column (default: "indexed").
    pub fn output_col(&self) -> &str {
        self.params.string_or("outputCol", "indexed")
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
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
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

    pub fn set_input_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("inputCol", col);
        self
    }

    /// The input column (default: "features").
    pub fn input_col(&self) -> &str {
        self.params.string_or("inputCol", "features")
    }

    pub fn set_output_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("outputCol", col);
        self
    }

    /// The output column (default: "maxAbs_scaled").
    pub fn output_col(&self) -> &str {
        self.params.string_or("outputCol", "maxAbs_scaled")
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
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
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

/// The server's cached model, shared by a model, its clones and every DataFrame that it
/// transforms (through the [`RelationOwner`] of their plans). The last owner to drop deletes
/// the cached object on the server, as PySpark's `RemoteModelRef` does.
#[derive(Debug, Clone)]
struct CachedModel(Arc<CachedModelRef>);

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

    pub fn set_feature_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("featuresCol", col);
        self
    }

    pub fn set_label_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("labelCol", col);
        self
    }

    pub fn set_prediction_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", col);
        self
    }

    /// Maximum number of optimizer iterations.
    pub fn set_max_iter(mut self, max_iter: i32) -> Self {
        self.params = self.params.set_param_int32("maxIter", max_iter);
        self
    }

    /// The feature vector column (default: "features").
    pub fn feature_col(&self) -> &str {
        self.params.string_or("featuresCol", "features")
    }

    /// The label column (default: "label").
    pub fn label_col(&self) -> &str {
        self.params.string_or("labelCol", "label")
    }

    /// The prediction output column (default: "prediction").
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }

    /// The maximum number of optimizer iterations (default: 100).
    pub fn max_iter(&self) -> i32 {
        self.params.int_or("maxIter", 100)
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
        let info = fit_via_command(&self.operator, &self.params, df)?;
        Ok(Box::new(self.model_from_info(info, &df.session)?))
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
}

impl LogisticRegressionModel {
    /// The feature vector column of the estimator that fitted the model (default: "features").
    pub fn feature_col(&self) -> &str {
        self.params.string_or("featuresCol", "features")
    }

    /// The label column of the estimator that fitted the model (default: "label").
    pub fn label_col(&self) -> &str {
        self.params.string_or("labelCol", "label")
    }

    /// The prediction output column of the model (default: "prediction").
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }
}

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

    /// Feature vector column.
    pub fn set_feature_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("featuresCol", value);
        self
    }

    /// Binary label column (0 or 1).
    pub fn set_label_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("labelCol", value);
        self
    }

    /// Prediction output column.
    pub fn set_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", value);
        self
    }

    /// Raw margin output column.
    pub fn set_raw_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("rawPredictionCol", value);
        self
    }

    /// Maximum number of optimizer iterations.
    pub fn set_max_iter(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("maxIter", value);
        self
    }

    /// L2 regularization strength.
    pub fn set_reg_param(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("regParam", value);
        self
    }

    /// Convergence tolerance.
    pub fn set_tol(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("tol", value);
        self
    }

    /// Whether to fit an intercept.
    pub fn set_fit_intercept(mut self, value: bool) -> Self {
        self.params = self.params.set_param_bool("fitIntercept", value);
        self
    }

    /// Whether to standardize training features.
    pub fn set_standardization(mut self, value: bool) -> Self {
        self.params = self.params.set_param_bool("standardization", value);
        self
    }

    /// Decision threshold applied to the model margin.
    pub fn set_threshold(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("threshold", value);
        self
    }

    /// Optional instance weight column.
    pub fn set_weight_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("weightCol", value);
        self
    }

    /// Tree aggregation depth.
    pub fn set_aggregation_depth(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("aggregationDepth", value);
        self
    }

    /// Maximum size in MB of stacked input blocks.
    pub fn set_max_block_size_in_mb(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("maxBlockSizeInMB", value);
        self
    }

    /// The feature vector column (default: "features").
    pub fn feature_col(&self) -> &str {
        self.params.string_or("featuresCol", "features")
    }

    /// The label column (default: "label").
    pub fn label_col(&self) -> &str {
        self.params.string_or("labelCol", "label")
    }

    /// The prediction output column (default: "prediction").
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }

    /// The raw margin output column (default: "rawPrediction").
    pub fn raw_prediction_col(&self) -> &str {
        self.params.string_or("rawPredictionCol", "rawPrediction")
    }

    /// The maximum number of optimizer iterations (default: 100).
    pub fn max_iter(&self) -> i32 {
        self.params.int_or("maxIter", 100)
    }

    /// The L2 regularization strength (default: 0).
    pub fn reg_param(&self) -> f64 {
        self.params.double_or("regParam", 0.0)
    }

    /// The convergence tolerance (default: 1e-6).
    pub fn tol(&self) -> f64 {
        self.params.double_or("tol", 1e-6)
    }

    /// Whether to fit an intercept (default: true).
    pub fn fit_intercept(&self) -> bool {
        self.params.bool_or("fitIntercept", true)
    }

    /// Whether to standardize training features (default: true).
    pub fn standardization(&self) -> bool {
        self.params.bool_or("standardization", true)
    }

    /// The decision threshold applied to the model margin (default: 0).
    pub fn threshold(&self) -> f64 {
        self.params.double_or("threshold", 0.0)
    }

    /// The instance weight column, if one is set.
    pub fn weight_col(&self) -> Option<&str> {
        self.params.string_opt("weightCol")
    }

    /// The tree aggregation depth (default: 2).
    pub fn aggregation_depth(&self) -> i32 {
        self.params.int_or("aggregationDepth", 2)
    }

    /// The maximum size in MB of stacked input blocks (default: 0, chosen by Spark).
    pub fn max_block_size_in_mb(&self) -> f64 {
        self.params.double_or("maxBlockSizeInMB", 0.0)
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
        let info = fit_via_command(&self.operator, &self.params, df)?;
        Ok(Box::new(self.model_from_info(info, &df.session)?))
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

    /// Feature vector column.
    pub fn set_feature_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("featuresCol", value);
        self
    }

    /// Numeric label column.
    pub fn set_label_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("labelCol", value);
        self
    }

    /// Prediction output column.
    pub fn set_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", value);
        self
    }

    /// Maximum number of optimizer iterations.
    pub fn set_max_iter(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("maxIter", value);
        self
    }

    /// Regularization strength.
    pub fn set_reg_param(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("regParam", value);
        self
    }

    /// Elastic-net mixing: 0 for L2, 1 for L1.
    pub fn set_elastic_net_param(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("elasticNetParam", value);
        self
    }

    /// Convergence tolerance.
    pub fn set_tol(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("tol", value);
        self
    }

    /// Whether to fit an intercept.
    pub fn set_fit_intercept(mut self, value: bool) -> Self {
        self.params = self.params.set_param_bool("fitIntercept", value);
        self
    }

    /// Whether to standardize training features.
    pub fn set_standardization(mut self, value: bool) -> Self {
        self.params = self.params.set_param_bool("standardization", value);
        self
    }

    /// Solver: auto, normal or l-bfgs.
    pub fn set_solver(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("solver", value);
        self
    }

    /// Optional instance weight column.
    pub fn set_weight_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("weightCol", value);
        self
    }

    /// Tree aggregation depth.
    pub fn set_aggregation_depth(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("aggregationDepth", value);
        self
    }

    /// Loss function: squaredError or huber.
    pub fn set_loss(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("loss", value);
        self
    }

    /// Huber loss shape parameter; must be greater than 1.
    pub fn set_epsilon(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("epsilon", value);
        self
    }

    /// Maximum size in MB of stacked input blocks.
    pub fn set_max_block_size_in_mb(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("maxBlockSizeInMB", value);
        self
    }

    /// The feature vector column (default: "features").
    pub fn feature_col(&self) -> &str {
        self.params.string_or("featuresCol", "features")
    }

    /// The label column (default: "label").
    pub fn label_col(&self) -> &str {
        self.params.string_or("labelCol", "label")
    }

    /// The prediction output column (default: "prediction").
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }

    /// The maximum number of optimizer iterations (default: 100).
    pub fn max_iter(&self) -> i32 {
        self.params.int_or("maxIter", 100)
    }

    /// The regularization strength (default: 0).
    pub fn reg_param(&self) -> f64 {
        self.params.double_or("regParam", 0.0)
    }

    /// The elastic-net mixing parameter (default: 0, L2).
    pub fn elastic_net_param(&self) -> f64 {
        self.params.double_or("elasticNetParam", 0.0)
    }

    /// The convergence tolerance (default: 1e-6).
    pub fn tol(&self) -> f64 {
        self.params.double_or("tol", 1e-6)
    }

    /// Whether to fit an intercept (default: true).
    pub fn fit_intercept(&self) -> bool {
        self.params.bool_or("fitIntercept", true)
    }

    /// Whether to standardize training features (default: true).
    pub fn standardization(&self) -> bool {
        self.params.bool_or("standardization", true)
    }

    /// The solver (default: "auto").
    pub fn solver(&self) -> &str {
        self.params.string_or("solver", "auto")
    }

    /// The instance weight column, if one is set.
    pub fn weight_col(&self) -> Option<&str> {
        self.params.string_opt("weightCol")
    }

    /// The tree aggregation depth (default: 2).
    pub fn aggregation_depth(&self) -> i32 {
        self.params.int_or("aggregationDepth", 2)
    }

    /// The loss function (default: "squaredError").
    pub fn loss(&self) -> &str {
        self.params.string_or("loss", "squaredError")
    }

    /// The Huber loss shape parameter (default: 1.35).
    pub fn epsilon(&self) -> f64 {
        self.params.double_or("epsilon", 1.35)
    }

    /// The maximum size in MB of stacked input blocks (default: 0, chosen by Spark).
    pub fn max_block_size_in_mb(&self) -> f64 {
        self.params.double_or("maxBlockSizeInMB", 0.0)
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
        })
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
        let info = fit_via_command(&self.operator, &self.params, df)?;
        Ok(Box::new(self.model_from_info(info, &df.session)?))
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

    /// Feature vector column.
    pub fn set_feature_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("featuresCol", value);
        self
    }

    /// Numeric label column; labels must be contiguous class indices starting at zero.
    pub fn set_label_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("labelCol", value);
        self
    }

    /// Predicted class output column.
    pub fn set_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", value);
        self
    }

    /// Class probability output column.
    pub fn set_probability_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("probabilityCol", value);
        self
    }

    /// Raw class prediction output column.
    pub fn set_raw_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("rawPredictionCol", value);
        self
    }

    /// Maximum tree depth, from 0 to 30.
    pub fn set_max_depth(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("maxDepth", value);
        self
    }

    /// Maximum number of bins for discretizing continuous features.
    pub fn set_max_bins(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("maxBins", value);
        self
    }

    /// Minimum number of training instances in each child node.
    pub fn set_min_instances_per_node(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("minInstancesPerNode", value);
        self
    }

    /// Minimum information gain for a split.
    pub fn set_min_info_gain(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("minInfoGain", value);
        self
    }

    /// Whether to prune redundant nodes, as supported by Spark 4.3.0.
    pub fn set_prune_tree(mut self, value: bool) -> Self {
        self.params = self.params.set_param_bool("pruneTree", value);
        self
    }

    /// Maximum memory in MB for histogram aggregation.
    pub fn set_max_memory_in_mb(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("maxMemoryInMB", value);
        self
    }

    /// Whether to cache node IDs for training instances.
    pub fn set_cache_node_ids(mut self, value: bool) -> Self {
        self.params = self.params.set_param_bool("cacheNodeIds", value);
        self
    }

    /// Node ID cache checkpoint interval; -1 disables checkpointing.
    pub fn set_checkpoint_interval(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("checkpointInterval", value);
        self
    }

    /// Impurity measure: gini or entropy.
    pub fn set_impurity(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("impurity", value);
        self
    }

    /// Number of trees in the forest; must be positive.
    pub fn set_num_trees(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("numTrees", value);
        self
    }

    /// Feature subset strategy: auto, all, onethird, sqrt, log2, or a numeric count/fraction.
    pub fn set_feature_subset_strategy(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("featureSubsetStrategy", value);
        self
    }

    /// Random seed for sampling.
    pub fn set_seed(mut self, value: i64) -> Self {
        self.params = self.params.set_param_int_narrow("seed", value);
        self
    }

    /// Fraction of training data used to train each tree, in (0, 1].
    pub fn set_subsampling_rate(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("subsamplingRate", value);
        self
    }

    /// Optional output vector of leaf indices, one per tree; empty disables it.
    pub fn set_leaf_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("leafCol", value);
        self
    }

    /// Minimum fraction of total weight in each child node, in [0, 0.5).
    pub fn set_min_weight_fraction_per_node(mut self, value: f64) -> Self {
        self.params = self
            .params
            .set_param_double("minWeightFractionPerNode", value);
        self
    }

    /// Optional instance weight column.
    pub fn set_weight_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("weightCol", value);
        self
    }

    /// Whether each tree samples training instances with replacement.
    pub fn set_bootstrap(mut self, value: bool) -> Self {
        self.params = self.params.set_param_bool("bootstrap", value);
        self
    }

    /// Per-class thresholds: predictions maximize probability divided by threshold.
    /// Provide one nonnegative value per class, with at most one zero.
    pub fn set_thresholds(mut self, values: impl IntoIterator<Item = f64>) -> Self {
        self.params = self.params.set_param_double_array("thresholds", values);
        self
    }

    /// The feature vector column (default: "features").
    pub fn feature_col(&self) -> &str {
        self.params.string_or("featuresCol", "features")
    }

    /// The label column (default: "label").
    pub fn label_col(&self) -> &str {
        self.params.string_or("labelCol", "label")
    }

    /// The predicted class output column (default: "prediction").
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }

    /// The class probability output column (default: "probability").
    pub fn probability_col(&self) -> &str {
        self.params.string_or("probabilityCol", "probability")
    }

    /// The raw class prediction output column (default: "rawPrediction").
    pub fn raw_prediction_col(&self) -> &str {
        self.params.string_or("rawPredictionCol", "rawPrediction")
    }

    /// The maximum tree depth (default: 5).
    pub fn max_depth(&self) -> i32 {
        self.params.int_or("maxDepth", 5)
    }

    /// The maximum number of bins for continuous features (default: 32).
    pub fn max_bins(&self) -> i32 {
        self.params.int_or("maxBins", 32)
    }

    /// The minimum number of training instances in each child node (default: 1).
    pub fn min_instances_per_node(&self) -> i32 {
        self.params.int_or("minInstancesPerNode", 1)
    }

    /// The minimum information gain for a split (default: 0).
    pub fn min_info_gain(&self) -> f64 {
        self.params.double_or("minInfoGain", 0.0)
    }

    /// Whether to prune redundant nodes (default: true).
    pub fn prune_tree(&self) -> bool {
        self.params.bool_or("pruneTree", true)
    }

    /// The maximum memory in MB for histogram aggregation (default: 256).
    pub fn max_memory_in_mb(&self) -> i32 {
        self.params.int_or("maxMemoryInMB", 256)
    }

    /// Whether to cache node IDs for training instances (default: false).
    pub fn cache_node_ids(&self) -> bool {
        self.params.bool_or("cacheNodeIds", false)
    }

    /// The node ID cache checkpoint interval (default: 10).
    pub fn checkpoint_interval(&self) -> i32 {
        self.params.int_or("checkpointInterval", 10)
    }

    /// The impurity measure (default: "gini").
    pub fn impurity(&self) -> &str {
        self.params.string_or("impurity", "gini")
    }

    /// The number of trees (default: 20).
    pub fn num_trees(&self) -> i32 {
        self.params.int_or("numTrees", 20)
    }

    /// The feature subset strategy (default: "auto").
    pub fn feature_subset_strategy(&self) -> &str {
        self.params.string_or("featureSubsetStrategy", "auto")
    }

    /// The random seed, if one is set; else Spark's default seed applies.
    pub fn seed(&self) -> Option<i64> {
        self.params.long_opt("seed")
    }

    /// The fraction of training data used to train each tree (default: 1).
    pub fn subsampling_rate(&self) -> f64 {
        self.params.double_or("subsamplingRate", 1.0)
    }

    /// The leaf index output column (default: empty, no column).
    pub fn leaf_col(&self) -> &str {
        self.params.string_or("leafCol", "")
    }

    /// The minimum fraction of total weight in each child node (default: 0).
    pub fn min_weight_fraction_per_node(&self) -> f64 {
        self.params.double_or("minWeightFractionPerNode", 0.0)
    }

    /// The instance weight column, if one is set.
    pub fn weight_col(&self) -> Option<&str> {
        self.params.string_opt("weightCol")
    }

    /// Whether each tree samples training instances with replacement (default: true).
    pub fn bootstrap(&self) -> bool {
        self.params.bool_or("bootstrap", true)
    }

    /// The per-class thresholds, if they are set.
    pub fn thresholds(&self) -> Option<Vec<f64>> {
        self.params.double_array_opt("thresholds")
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
        let info = fit_via_command(&self.operator, &self.params, df)?;
        Ok(Box::new(self.model_from_info(info, &df.session)?))
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

    /// Feature vector column.
    pub fn set_feature_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("featuresCol", value);
        self
    }

    /// Cluster assignment output column.
    pub fn set_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", value);
        self
    }

    /// Number of clusters; must be greater than 1.
    pub fn set_k(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("k", value);
        self
    }

    /// Initialization mode: random or k-means||.
    pub fn set_init_mode(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("initMode", value);
        self
    }

    /// Number of steps in k-means|| initialization.
    pub fn set_init_steps(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("initSteps", value);
        self
    }

    /// Convergence tolerance.
    pub fn set_tol(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("tol", value);
        self
    }

    /// Maximum number of clustering iterations.
    pub fn set_max_iter(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("maxIter", value);
        self
    }

    /// Random seed for initialization.
    pub fn set_seed(mut self, value: i64) -> Self {
        self.params = self.params.set_param_int_narrow("seed", value);
        self
    }

    /// Distance measure: euclidean or cosine.
    pub fn set_distance_measure(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("distanceMeasure", value);
        self
    }

    /// Optional instance weight column.
    pub fn set_weight_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("weightCol", value);
        self
    }

    /// Solver: auto, row or block.
    pub fn set_solver(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("solver", value);
        self
    }

    /// Maximum size in MB of stacked input blocks.
    pub fn set_max_block_size_in_mb(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("maxBlockSizeInMB", value);
        self
    }

    /// The feature vector column (default: "features").
    pub fn feature_col(&self) -> &str {
        self.params.string_or("featuresCol", "features")
    }

    /// The cluster assignment output column (default: "prediction").
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }

    /// The number of clusters (default: 2).
    pub fn k(&self) -> i32 {
        self.params.int_or("k", 2)
    }

    /// The initialization mode (default: "k-means||").
    pub fn init_mode(&self) -> &str {
        self.params.string_or("initMode", "k-means||")
    }

    /// The number of steps in k-means|| initialization (default: 2).
    pub fn init_steps(&self) -> i32 {
        self.params.int_or("initSteps", 2)
    }

    /// The convergence tolerance (default: 1e-4).
    pub fn tol(&self) -> f64 {
        self.params.double_or("tol", 1e-4)
    }

    /// The maximum number of clustering iterations (default: 20).
    pub fn max_iter(&self) -> i32 {
        self.params.int_or("maxIter", 20)
    }

    /// The random seed, if one is set; else Spark's default seed applies.
    pub fn seed(&self) -> Option<i64> {
        self.params.long_opt("seed")
    }

    /// The distance measure (default: "euclidean").
    pub fn distance_measure(&self) -> &str {
        self.params.string_or("distanceMeasure", "euclidean")
    }

    /// The instance weight column, if one is set.
    pub fn weight_col(&self) -> Option<&str> {
        self.params.string_opt("weightCol")
    }

    /// The solver (default: "auto").
    pub fn solver(&self) -> &str {
        self.params.string_or("solver", "auto")
    }

    /// The maximum size in MB of stacked input blocks (default: 0, chosen by Spark).
    pub fn max_block_size_in_mb(&self) -> f64 {
        self.params.double_or("maxBlockSizeInMB", 0.0)
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
        })
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
        let info = fit_via_command(&self.operator, &self.params, df)?;
        Ok(Box::new(self.model_from_info(info, &df.session)?))
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
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// RegressionEvaluator: evaluates regression models.
#[derive(Debug, Clone)]
pub struct RegressionEvaluator {
    operator: MlOperator,
    params: Params,
}

impl RegressionEvaluator {
    pub fn new() -> Self {
        RegressionEvaluator {
            operator: MlOperator::new(
                "org.apache.spark.ml.evaluation.RegressionEvaluator",
                OperatorType::Evaluator,
            ),
            params: Params::new(),
        }
    }

    pub fn set_label_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("labelCol", col);
        self
    }

    pub fn set_prediction_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", col);
        self
    }

    pub fn set_metric_name(mut self, metric: &str) -> Self {
        self.params = self.params.set_param_string("metricName", metric);
        self
    }

    /// The label column (default: "label").
    pub fn label_col(&self) -> &str {
        self.params.string_or("labelCol", "label")
    }

    /// The prediction column (default: "prediction").
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }

    /// The metric (default: "rmse").
    pub fn metric_name(&self) -> &str {
        self.params.string_or("metricName", "rmse")
    }

    /// Get mutable parameters, as estimators and transformers have them.
    pub fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }
}

impl Default for RegressionEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator for RegressionEvaluator {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn evaluate(&self, df: &DataFrame) -> spark_connect_core::error::Result<f64> {
        evaluate_via_command(&self.operator, &self.params, df)
    }
}

/// BinaryClassificationEvaluator: evaluates binary classification models.
#[derive(Debug, Clone)]
pub struct BinaryClassificationEvaluator {
    operator: MlOperator,
    params: Params,
}

impl BinaryClassificationEvaluator {
    pub fn new() -> Self {
        BinaryClassificationEvaluator {
            operator: MlOperator::new(
                "org.apache.spark.ml.evaluation.BinaryClassificationEvaluator",
                OperatorType::Evaluator,
            ),
            params: Params::new(),
        }
    }

    pub fn set_label_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("labelCol", col);
        self
    }

    /// Set the column of the scores (a vector of raw predictions), the JVM `rawPredictionCol`.
    pub fn set_raw_prediction_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("rawPredictionCol", col);
        self
    }

    pub fn set_metric_name(mut self, metric: &str) -> Self {
        self.params = self.params.set_param_string("metricName", metric);
        self
    }

    /// The label column (default: "label").
    pub fn label_col(&self) -> &str {
        self.params.string_or("labelCol", "label")
    }

    /// The column of the scores (default: "rawPrediction").
    pub fn raw_prediction_col(&self) -> &str {
        self.params.string_or("rawPredictionCol", "rawPrediction")
    }

    /// The metric (default: "areaUnderROC").
    pub fn metric_name(&self) -> &str {
        self.params.string_or("metricName", "areaUnderROC")
    }

    /// Get mutable parameters, as estimators and transformers have them.
    pub fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }
}

impl Default for BinaryClassificationEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator for BinaryClassificationEvaluator {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn evaluate(&self, df: &DataFrame) -> spark_connect_core::error::Result<f64> {
        evaluate_via_command(&self.operator, &self.params, df)
    }
}

/// Pipeline Estimator: chains multiple stages together.
#[derive(Debug, Clone)]
pub struct Pipeline {
    operator: MlOperator,
    params: Params,
}

impl Pipeline {
    pub fn new() -> Self {
        Pipeline {
            operator: MlOperator::new("org.apache.spark.ml.Pipeline", OperatorType::Estimator),
            params: Params::new(),
        }
    }

    pub fn set_stages(mut self, stage_names: Vec<&str>) -> Self {
        self.params = self.params.set_param_string_array("stages", stage_names);
        self
    }

    /// The names of the stages (default: none).
    pub fn stages(&self) -> Vec<String> {
        self.params.string_array("stages")
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for Pipeline {
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

    fn fit_impl(&mut self, _df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        let model = PipelineModel {
            operator: MlOperator::with_uid(
                &self.operator.name,
                &self.operator.uid,
                OperatorType::Model,
            ),
            params: self.params.clone(),
            stages: self.stages(),
        };
        Ok(Box::new(model))
    }
}

/// PipelineModel: fitted model from Pipeline.
#[derive(Debug, Clone)]
pub struct PipelineModel {
    pub operator: MlOperator,
    pub params: Params,
    pub stages: Vec<String>,
}

impl Transformer for PipelineModel {
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

impl Model for PipelineModel {
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// MulticlassClassificationEvaluator: metrics for multiclass classification.
#[derive(Debug, Clone)]
pub struct MulticlassClassificationEvaluator {
    operator: MlOperator,
    params: Params,
}

impl MulticlassClassificationEvaluator {
    pub fn new() -> Self {
        MulticlassClassificationEvaluator {
            operator: MlOperator::new(
                "org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator",
                OperatorType::Evaluator,
            ),
            params: Params::new(),
        }
    }
    pub fn set_label_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("labelCol", col);
        self
    }
    pub fn set_prediction_col(mut self, col: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", col);
        self
    }
    pub fn set_metric_name(mut self, metric: &str) -> Self {
        self.params = self.params.set_param_string("metricName", metric);
        self
    }
    /// The label column (default: "label").
    pub fn label_col(&self) -> &str {
        self.params.string_or("labelCol", "label")
    }
    /// The prediction column (default: "prediction").
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }
    /// The metric (default: "f1").
    pub fn metric_name(&self) -> &str {
        self.params.string_or("metricName", "f1")
    }
    /// Get mutable parameters, as estimators and transformers have them.
    pub fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }
}

impl Default for MulticlassClassificationEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator for MulticlassClassificationEvaluator {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }
    fn params(&self) -> &Params {
        &self.params
    }
    fn evaluate(&self, df: &DataFrame) -> spark_connect_core::error::Result<f64> {
        evaluate_via_command(&self.operator, &self.params, df)
    }
}

/// CrossValidator: k-fold cross-validation for hyperparameter tuning.
///
/// The nested estimator/evaluator/estimatorParamMaps have no scalar-literal
/// encoding, so this carries the numeric tuning params it can faithfully encode
/// (numFolds, seed, parallelism); the sub-estimator/evaluator are held for the
/// server-side fit. fit() produces a CrossValidatorModel the same lazy way the
/// other estimators do.
#[derive(Debug, Clone)]
pub struct CrossValidator {
    operator: MlOperator,
    params: Params,
}

impl CrossValidator {
    pub fn new() -> Self {
        CrossValidator {
            operator: MlOperator::new(
                "org.apache.spark.ml.tuning.CrossValidator",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }
    pub fn set_num_folds(mut self, num_folds: i32) -> Self {
        self.params = self.params.set_param_int32("numFolds", num_folds);
        self
    }
    pub fn set_parallelism(mut self, parallelism: i32) -> Self {
        self.params = self.params.set_param_int32("parallelism", parallelism);
        self
    }
    /// The random seed of the folds, sent as the narrowest integer literal that holds it,
    /// as PySpark sends a Python int.
    pub fn set_seed(mut self, seed: i64) -> Self {
        self.params = self.params.set_param_int_narrow("seed", seed);
        self
    }
    /// The number of folds (default: 3).
    pub fn num_folds(&self) -> i32 {
        self.params.int_or("numFolds", 3)
    }
    /// The number of models fitted in parallel (default: 1).
    pub fn parallelism(&self) -> i32 {
        self.params.int_or("parallelism", 1)
    }
    /// The random seed of the folds, if one is set; else Spark's default seed applies.
    pub fn seed(&self) -> Option<i64> {
        self.params.long_opt("seed")
    }
}

impl Default for CrossValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for CrossValidator {
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
    fn fit_impl(&mut self, _df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        let model = CrossValidatorModel {
            operator: MlOperator::with_uid(
                &self.operator.name,
                &self.operator.uid,
                OperatorType::Model,
            ),
            params: self.params.clone(),
        };
        Ok(Box::new(model))
    }
}

/// CrossValidatorModel: the best model selected by CrossValidator.fit.
#[derive(Debug, Clone)]
pub struct CrossValidatorModel {
    operator: MlOperator,
    params: Params,
}

impl Transformer for CrossValidatorModel {
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

impl Model for CrossValidatorModel {
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Used only by the tests below (the non-test build does not reference it).
    use crate::session::SparkSession;

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

        assert_eq!(scaler.input_col(), "my_features");
        assert_eq!(scaler.output_col(), "my_scaled");
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

        assert_eq!(indexer.input_col(), "category");
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

        assert_eq!(scaler.input_col(), "features");
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
            .set_feature_col("features")
            .set_label_col("label")
            .set_prediction_col("pred")
            .set_max_iter(50);

        assert_eq!(lr.feature_col(), "features");
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
        let pipeline = Pipeline::new().set_stages(vec!["stage1", "stage2", "stage3"]);

        assert_eq!(pipeline.stages().len(), 3);
        assert_eq!(pipeline.stages()[0], "stage1");
        assert_eq!(pipeline.stages()[1], "stage2");
        assert_eq!(pipeline.stages()[2], "stage3");
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
        // An Array[String] param (inputCols, Pipeline stages) is an array literal of string
        // elements, as PySpark sends a list; the server rejects a string ("not an array").
        let va = VectorAssembler::new().set_input_cols(vec!["a", "b"]);
        let pipeline = Pipeline::new().set_stages(vec!["scaler", "lr"]);
        for (name, params, expected) in [
            ("inputCols", va.params(), vec!["a", "b"]),
            ("stages", pipeline.params(), vec!["scaler", "lr"]),
        ] {
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
        }

        // Int params are Integer literals when the value fits 32 bits, as PySpark sends a Python
        // int; the JVM param is an IntParam (maxIter, numFolds).
        let lr = LogisticRegression::new().set_max_iter(5);
        let params = lr.params().to_proto().params;
        assert_eq!(params["maxIter"].literal_type, Some(L::Integer(5)));
        let cv = CrossValidator::new().set_num_folds(3);
        let params = cv.params().to_proto().params;
        assert_eq!(params["numFolds"].literal_type, Some(L::Integer(3)));
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
    fn cross_validator_creation_getters_and_fit() {
        let cv = CrossValidator::new()
            .set_num_folds(5)
            .set_parallelism(2)
            .set_seed(42);
        assert_eq!(cv.num_folds(), 5);
        assert_eq!(cv.parallelism(), 2);
        assert_eq!(cv.operator().op_type, OperatorType::Estimator);
        assert_eq!(
            cv.operator().name,
            "org.apache.spark.ml.tuning.CrossValidator"
        );
        assert!(cv.params().get_param("numFolds").is_some());
        assert!(cv.params().get_param("seed").is_some());
        assert_eq!(CrossValidator::default().num_folds(), 3);

        // fit builds a CrossValidatorModel; its transform yields an MlTransform plan.
        let s = offline_session();
        let df = s.range(3).unwrap();
        let mut cv2 = CrossValidator::new().set_num_folds(2);
        let mut model = cv2.fit(&df).unwrap();
        let out = model.transform(&df).unwrap();
        assert!(matches!(out.plan.as_ref(), LogicalPlan::MlTransform { .. }));
        let _ = model.clone_box();
    }

    #[test]
    fn offline_estimators_fit_and_models_transform() {
        let s = offline_session();
        let df = s.range(3).unwrap();
        // The models of the server-fitted estimators transform through their cached model;
        // this exercises the offline model wrappers. The fits are tested against a server.
        let mas = MaxAbsScaler::new().set_input_col("f").set_output_col("o");
        assert_eq!(mas.input_col(), "f");
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

        let mut pipe = Pipeline::new().set_stages(vec!["a", "b"]);
        assert_eq!(pipe.stages().len(), 2);
        let mut m = pipe.fit(&df).unwrap();
        let _ = m.transform(&df).unwrap();
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
            .set_feature_col("input")
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
            let forest = RandomForestClassifier::new().set_seed(seed);
            assert_eq!(sent(forest.params()), Some(expected.clone()));
            let cv = CrossValidator::new().set_seed(seed);
            assert_eq!(sent(cv.params()), Some(expected));
        }
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
                        km.feature_col(),
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
                (2, 1e-4, 20, None),
                ("euclidean", None, "auto"),
                0.0
            )
        );
        let km = km
            .set_feature_col("f")
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
                (4, 0.5, 7, Some(1 << 40)),
                ("cosine", Some("w"), "row"),
                1.5
            )
        );

        let svc = LinearSVC::new();
        macro_rules! all {
            ($s:expr) => {{
                let s: &LinearSVC = &$s;
                (
                    (s.feature_col(), s.label_col(), s.prediction_col()),
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
            .set_feature_col("f")
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
                        r.feature_col(),
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
            .set_feature_col("f")
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
                    (f.feature_col(), f.label_col(), f.prediction_col()),
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
                ("auto", None, 1.0),
                ("", 0.0, None),
                (true, None)
            )
        );
        let rf = rf
            .set_feature_col("f")
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
                ("all", Some(5), 0.5),
                ("leaves", 0.2, Some("w")),
                (false, Some(vec![0.25, 0.75]))
            )
        );

        let logistic = LogisticRegression::new();
        assert_eq!(
            (
                logistic.feature_col(),
                logistic.label_col(),
                logistic.prediction_col(),
                logistic.max_iter()
            ),
            ("features", "label", "prediction", 100)
        );
        let cv = CrossValidator::new();
        assert_eq!((cv.num_folds(), cv.parallelism(), cv.seed()), (3, 1, None));
        let cv = cv.set_num_folds(5).set_parallelism(2).set_seed(9);
        assert_eq!(
            (cv.num_folds(), cv.parallelism(), cv.seed()),
            (5, 2, Some(9))
        );
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
        let pipeline = Pipeline::new().set_stages(vec!["s"]);
        assert_eq!(pipeline.stages(), ["s"]);
    }

    #[test]
    fn logistic_regression_model_reads_its_columns_from_its_params() {
        let session = offline_session();
        let model = fitted_logistic_regression(&session);
        assert_eq!(
            (
                model.feature_col(),
                model.label_col(),
                model.prediction_col()
            ),
            ("features", "label", "pred")
        );
        assert_eq!(model.operator().uid, "operator-uid");
    }
}
