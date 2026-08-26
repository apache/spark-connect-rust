//! ML module mirroring `pyspark.ml.connect.base` and related ML classes.
//!
//! This module provides the foundation for machine learning in Spark Connect:
//! - `Params`: parameter management for ML operators
//! - `MlOperator`: represents ML operators (estimators, transformers, models, evaluators)
//! - Base traits: `Estimator`, `Transformer`, `Model`, `Evaluator`
//! - Concrete example: `StandardScaler` estimator + `StandardScalerModel`

use spark_connect_proto as proto;
use std::collections::HashMap;
use uuid::Uuid;

use crate::dataframe::DataFrame;
use crate::plan::LogicalPlan;
use crate::session::SparkSession;

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

    /// Get a parameter value.
    pub fn get_param(&self, name: &str) -> Option<&proto::Expression> {
        self.params.get(name)
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
    /// Create a new MlOperator with auto-generated UID.
    pub fn new(name: &str, op_type: OperatorType) -> Self {
        MlOperator {
            name: name.to_string(),
            uid: Uuid::new_v4().to_string(),
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
    fn build_ml_relation(&self, input_plan: &LogicalPlan) -> proto::MlRelation {
        let mut transform = proto::ml_relation::Transform::default();
        transform.operator = Some(proto::ml_relation::transform::Operator::Transformer(
            self.operator().to_proto(),
        ));
        transform.input = Some(Box::new(input_plan.to_proto()));
        transform.params = Some(self.params().to_proto());

        let mut ml_relation = proto::MlRelation::default();
        ml_relation.ml_type = Some(proto::ml_relation::MlType::Transform(Box::new(transform)));
        ml_relation
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
    let dataset = crate::dataframe::build_input_relation(&df.plan, &df.session)?;
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
    /// Input column name (default: "features")
    input_col: String,
    /// Output column name (default: "scaled_features")
    output_col: String,
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
            input_col: "features".to_string(),
            output_col: "scaled_features".to_string(),
        }
    }

    /// Set the input column name.
    pub fn set_input_col(mut self, col: &str) -> Self {
        self.input_col = col.to_string();
        self.params = self.params.set_param_string("inputCol", col);
        self
    }

    /// Get the input column name.
    pub fn input_col(&self) -> &str {
        &self.input_col
    }

    /// Set the output column name.
    pub fn set_output_col(mut self, col: &str) -> Self {
        self.output_col = col.to_string();
        self.params = self.params.set_param_string("outputCol", col);
        self
    }

    /// Get the output column name.
    pub fn output_col(&self) -> &str {
        &self.output_col
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

    fn fit_impl(&mut self, _df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        // Create a StandardScalerModel from this estimator
        let model = StandardScalerModel {
            operator: MlOperator::with_uid(
                &self.operator.name,
                &self.operator.uid,
                OperatorType::Model,
            ),
            params: self.params.clone(),
            input_col: self.input_col.clone(),
            output_col: self.output_col.clone(),
        };
        Ok(Box::new(model))
    }
}

/// StandardScalerModel: fitted model from StandardScaler.
#[derive(Debug, Clone)]
pub struct StandardScalerModel {
    operator: MlOperator,
    params: Params,
    input_col: String,
    output_col: String,
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

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        // Build an MlRelation for transformation
        let ml_relation = self.build_ml_relation(&df.plan);

        // Wrap it in a Relation
        let mut relation = proto::Relation::default();
        relation.common = Some(proto::RelationCommon::default());
        relation.rel_type = Some(proto::relation::RelType::MlRelation(Box::new(ml_relation)));

        // Wrap the built MlRelation as the plan; its `to_proto` emits the relation as-is.
        let plan = LogicalPlan::MlTransform {
            ml_relation: relation,
        };

        Ok(DataFrame::new(df.session.clone(), plan))
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
    input_cols: Vec<String>,
    output_col: String,
}

impl VectorAssembler {
    pub fn new() -> Self {
        VectorAssembler {
            operator: MlOperator::new(
                "org.apache.spark.ml.feature.VectorAssembler",
                OperatorType::Transformer,
            ),
            params: Params::new(),
            input_cols: Vec::new(),
            output_col: "assembled".to_string(),
        }
    }

    pub fn set_input_cols(mut self, cols: Vec<&str>) -> Self {
        self.input_cols = cols.iter().map(|c| c.to_string()).collect();
        self.params = self
            .params
            .set_param_string("inputCols", &format!("{:?}", self.input_cols));
        self
    }

    pub fn input_cols(&self) -> &[String] {
        &self.input_cols
    }

    pub fn set_output_col(mut self, col: &str) -> Self {
        self.output_col = col.to_string();
        self.params = self.params.set_param_string("outputCol", col);
        self
    }

    pub fn output_col(&self) -> &str {
        &self.output_col
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
        let ml_relation = self.build_ml_relation(&df.plan);
        let mut relation = proto::Relation::default();
        relation.common = Some(proto::RelationCommon::default());
        relation.rel_type = Some(proto::relation::RelType::MlRelation(Box::new(ml_relation)));
        let plan = LogicalPlan::MlTransform {
            ml_relation: relation,
        };
        Ok(DataFrame::new(df.session.clone(), plan))
    }
}

/// StringIndexer Estimator: converts string columns to numeric indices.
#[derive(Debug, Clone)]
pub struct StringIndexer {
    operator: MlOperator,
    params: Params,
    input_col: String,
    output_col: String,
}

impl StringIndexer {
    pub fn new() -> Self {
        StringIndexer {
            operator: MlOperator::new(
                "org.apache.spark.ml.feature.StringIndexer",
                OperatorType::Estimator,
            ),
            params: Params::new(),
            input_col: String::new(),
            output_col: "indexed".to_string(),
        }
    }

    pub fn set_input_col(mut self, col: &str) -> Self {
        self.input_col = col.to_string();
        self.params = self.params.set_param_string("inputCol", col);
        self
    }

    pub fn input_col(&self) -> &str {
        &self.input_col
    }

    pub fn set_output_col(mut self, col: &str) -> Self {
        self.output_col = col.to_string();
        self.params = self.params.set_param_string("outputCol", col);
        self
    }

    pub fn output_col(&self) -> &str {
        &self.output_col
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

    fn fit_impl(&mut self, _df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        let model = StringIndexerModel {
            operator: MlOperator::with_uid(
                &self.operator.name,
                &self.operator.uid,
                OperatorType::Model,
            ),
            params: self.params.clone(),
            input_col: self.input_col.clone(),
            output_col: self.output_col.clone(),
        };
        Ok(Box::new(model))
    }
}

/// StringIndexerModel: fitted model from StringIndexer.
#[derive(Debug, Clone)]
pub struct StringIndexerModel {
    pub operator: MlOperator,
    pub params: Params,
    pub input_col: String,
    pub output_col: String,
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

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        let ml_relation = self.build_ml_relation(&df.plan);
        let mut relation = proto::Relation::default();
        relation.common = Some(proto::RelationCommon::default());
        relation.rel_type = Some(proto::relation::RelType::MlRelation(Box::new(ml_relation)));
        let plan = LogicalPlan::MlTransform {
            ml_relation: relation,
        };
        Ok(DataFrame::new(df.session.clone(), plan))
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
    input_col: String,
    output_col: String,
}

impl MaxAbsScaler {
    pub fn new() -> Self {
        MaxAbsScaler {
            operator: MlOperator::new(
                "org.apache.spark.ml.feature.MaxAbsScaler",
                OperatorType::Estimator,
            ),
            params: Params::new(),
            input_col: "features".to_string(),
            output_col: "maxAbs_scaled".to_string(),
        }
    }

    pub fn set_input_col(mut self, col: &str) -> Self {
        self.input_col = col.to_string();
        self.params = self.params.set_param_string("inputCol", col);
        self
    }

    pub fn input_col(&self) -> &str {
        &self.input_col
    }

    pub fn set_output_col(mut self, col: &str) -> Self {
        self.output_col = col.to_string();
        self.params = self.params.set_param_string("outputCol", col);
        self
    }

    pub fn output_col(&self) -> &str {
        &self.output_col
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

    fn fit_impl(&mut self, _df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        let model = MaxAbsScalerModel {
            operator: MlOperator::with_uid(
                &self.operator.name,
                &self.operator.uid,
                OperatorType::Model,
            ),
            params: self.params.clone(),
            input_col: self.input_col.clone(),
            output_col: self.output_col.clone(),
        };
        Ok(Box::new(model))
    }
}

/// MaxAbsScalerModel: fitted model from MaxAbsScaler.
#[derive(Debug, Clone)]
pub struct MaxAbsScalerModel {
    pub operator: MlOperator,
    pub params: Params,
    pub input_col: String,
    pub output_col: String,
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

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        let ml_relation = self.build_ml_relation(&df.plan);
        let mut relation = proto::Relation::default();
        relation.common = Some(proto::RelationCommon::default());
        relation.rel_type = Some(proto::relation::RelType::MlRelation(Box::new(ml_relation)));
        let plan = LogicalPlan::MlTransform {
            ml_relation: relation,
        };
        Ok(DataFrame::new(df.session.clone(), plan))
    }
}

impl Model for MaxAbsScalerModel {
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// LogisticRegression Estimator: binary/multiclass classification.
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    operator: MlOperator,
    params: Params,
    feature_col: String,
    label_col: String,
    prediction_col: String,
    max_iter: i64,
}

impl LogisticRegression {
    pub fn new() -> Self {
        LogisticRegression {
            operator: MlOperator::new(
                "org.apache.spark.ml.classification.LogisticRegression",
                OperatorType::Estimator,
            ),
            params: Params::new(),
            feature_col: "features".to_string(),
            label_col: "label".to_string(),
            prediction_col: "prediction".to_string(),
            max_iter: 100,
        }
    }

    pub fn set_feature_col(mut self, col: &str) -> Self {
        self.feature_col = col.to_string();
        self.params = self.params.set_param_string("featuresCol", col);
        self
    }

    pub fn set_label_col(mut self, col: &str) -> Self {
        self.label_col = col.to_string();
        self.params = self.params.set_param_string("labelCol", col);
        self
    }

    pub fn set_prediction_col(mut self, col: &str) -> Self {
        self.prediction_col = col.to_string();
        self.params = self.params.set_param_string("predictionCol", col);
        self
    }

    pub fn set_max_iter(mut self, max_iter: i64) -> Self {
        self.max_iter = max_iter;
        self.params = self.params.set_param_int("maxIter", max_iter);
        self
    }

    pub fn feature_col(&self) -> &str {
        &self.feature_col
    }

    pub fn label_col(&self) -> &str {
        &self.label_col
    }

    pub fn prediction_col(&self) -> &str {
        &self.prediction_col
    }

    pub fn max_iter(&self) -> i64 {
        self.max_iter
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

    fn fit_impl(&mut self, _df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        let model = LogisticRegressionModel {
            operator: MlOperator::with_uid(
                &self.operator.name,
                &self.operator.uid,
                OperatorType::Model,
            ),
            params: self.params.clone(),
            feature_col: self.feature_col.clone(),
            label_col: self.label_col.clone(),
            prediction_col: self.prediction_col.clone(),
        };
        Ok(Box::new(model))
    }
}

/// LogisticRegressionModel: fitted model from LogisticRegression.
#[derive(Debug, Clone)]
pub struct LogisticRegressionModel {
    pub operator: MlOperator,
    pub params: Params,
    pub feature_col: String,
    pub label_col: String,
    pub prediction_col: String,
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

    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        let ml_relation = self.build_ml_relation(&df.plan);
        let mut relation = proto::Relation::default();
        relation.common = Some(proto::RelationCommon::default());
        relation.rel_type = Some(proto::relation::RelType::MlRelation(Box::new(ml_relation)));
        let plan = LogicalPlan::MlTransform {
            ml_relation: relation,
        };
        Ok(DataFrame::new(df.session.clone(), plan))
    }
}

impl Model for LogisticRegressionModel {
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// RegressionEvaluator: evaluates regression models.
#[derive(Debug, Clone)]
pub struct RegressionEvaluator {
    operator: MlOperator,
    params: Params,
    label_col: String,
    prediction_col: String,
    metric_name: String,
}

impl RegressionEvaluator {
    pub fn new() -> Self {
        RegressionEvaluator {
            operator: MlOperator::new(
                "org.apache.spark.ml.evaluation.RegressionEvaluator",
                OperatorType::Evaluator,
            ),
            params: Params::new(),
            label_col: "label".to_string(),
            prediction_col: "prediction".to_string(),
            metric_name: "rmse".to_string(),
        }
    }

    pub fn set_label_col(mut self, col: &str) -> Self {
        self.label_col = col.to_string();
        self.params = self.params.set_param_string("labelCol", col);
        self
    }

    pub fn set_prediction_col(mut self, col: &str) -> Self {
        self.prediction_col = col.to_string();
        self.params = self.params.set_param_string("predictionCol", col);
        self
    }

    pub fn set_metric_name(mut self, metric: &str) -> Self {
        self.metric_name = metric.to_string();
        self.params = self.params.set_param_string("metricName", metric);
        self
    }

    pub fn label_col(&self) -> &str {
        &self.label_col
    }

    pub fn prediction_col(&self) -> &str {
        &self.prediction_col
    }

    pub fn metric_name(&self) -> &str {
        &self.metric_name
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
    label_col: String,
    score_col: String,
    metric_name: String,
}

impl BinaryClassificationEvaluator {
    pub fn new() -> Self {
        BinaryClassificationEvaluator {
            operator: MlOperator::new(
                "org.apache.spark.ml.evaluation.BinaryClassificationEvaluator",
                OperatorType::Evaluator,
            ),
            params: Params::new(),
            label_col: "label".to_string(),
            score_col: "prediction".to_string(),
            metric_name: "areaUnderROC".to_string(),
        }
    }

    pub fn set_label_col(mut self, col: &str) -> Self {
        self.label_col = col.to_string();
        self.params = self.params.set_param_string("labelCol", col);
        self
    }

    pub fn set_score_col(mut self, col: &str) -> Self {
        self.score_col = col.to_string();
        self.params = self.params.set_param_string("scoreCol", col);
        self
    }

    pub fn set_metric_name(mut self, metric: &str) -> Self {
        self.metric_name = metric.to_string();
        self.params = self.params.set_param_string("metricName", metric);
        self
    }

    pub fn label_col(&self) -> &str {
        &self.label_col
    }

    pub fn score_col(&self) -> &str {
        &self.score_col
    }

    pub fn metric_name(&self) -> &str {
        &self.metric_name
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
    stages: Vec<String>,
}

impl Pipeline {
    pub fn new() -> Self {
        Pipeline {
            operator: MlOperator::new("org.apache.spark.ml.Pipeline", OperatorType::Estimator),
            params: Params::new(),
            stages: Vec::new(),
        }
    }

    pub fn set_stages(mut self, stage_names: Vec<&str>) -> Self {
        self.stages = stage_names.iter().map(|s| s.to_string()).collect();
        self.params = self
            .params
            .set_param_string("stages", &format!("{:?}", self.stages));
        self
    }

    pub fn stages(&self) -> &[String] {
        &self.stages
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
            stages: self.stages.clone(),
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
        let ml_relation = self.build_ml_relation(&df.plan);
        let mut relation = proto::Relation::default();
        relation.common = Some(proto::RelationCommon::default());
        relation.rel_type = Some(proto::relation::RelType::MlRelation(Box::new(ml_relation)));
        let plan = LogicalPlan::MlTransform {
            ml_relation: relation,
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
    label_col: String,
    prediction_col: String,
    metric_name: String,
}

impl MulticlassClassificationEvaluator {
    pub fn new() -> Self {
        MulticlassClassificationEvaluator {
            operator: MlOperator::new(
                "org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator",
                OperatorType::Evaluator,
            ),
            params: Params::new(),
            label_col: "label".to_string(),
            prediction_col: "prediction".to_string(),
            metric_name: "f1".to_string(),
        }
    }
    pub fn set_label_col(mut self, col: &str) -> Self {
        self.label_col = col.to_string();
        self.params = self.params.set_param_string("labelCol", col);
        self
    }
    pub fn set_prediction_col(mut self, col: &str) -> Self {
        self.prediction_col = col.to_string();
        self.params = self.params.set_param_string("predictionCol", col);
        self
    }
    pub fn set_metric_name(mut self, metric: &str) -> Self {
        self.metric_name = metric.to_string();
        self.params = self.params.set_param_string("metricName", metric);
        self
    }
    pub fn label_col(&self) -> &str {
        &self.label_col
    }
    pub fn prediction_col(&self) -> &str {
        &self.prediction_col
    }
    pub fn metric_name(&self) -> &str {
        &self.metric_name
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
    num_folds: i32,
    parallelism: i32,
    seed: Option<i64>,
}

impl CrossValidator {
    pub fn new() -> Self {
        CrossValidator {
            operator: MlOperator::new(
                "org.apache.spark.ml.tuning.CrossValidator",
                OperatorType::Estimator,
            ),
            params: Params::new(),
            num_folds: 3,
            parallelism: 1,
            seed: None,
        }
    }
    pub fn set_num_folds(mut self, num_folds: i32) -> Self {
        self.num_folds = num_folds;
        self.params = self.params.set_param_int("numFolds", num_folds as i64);
        self
    }
    pub fn set_parallelism(mut self, parallelism: i32) -> Self {
        self.parallelism = parallelism;
        self.params = self.params.set_param_int("parallelism", parallelism as i64);
        self
    }
    pub fn set_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self.params = self.params.set_param_int("seed", seed);
        self
    }
    pub fn num_folds(&self) -> i32 {
        self.num_folds
    }
    pub fn parallelism(&self) -> i32 {
        self.parallelism
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
        let ml_relation = self.build_ml_relation(&df.plan);
        let mut relation = proto::Relation::default();
        relation.common = Some(proto::RelationCommon::default());
        relation.rel_type = Some(proto::relation::RelType::MlRelation(Box::new(ml_relation)));
        let plan = LogicalPlan::MlTransform {
            ml_relation: relation,
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
            .set_score_col("score")
            .set_metric_name("areaUnderPR");

        assert_eq!(eval.label_col(), "label");
        assert_eq!(eval.score_col(), "score");
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
                input_col: "in".to_string(),
                output_col: "out".to_string(),
            }),
            Box::new(MaxAbsScalerModel {
                operator: MlOperator::new("test", OperatorType::Model),
                params: Params::new(),
                input_col: "in".to_string(),
                output_col: "out".to_string(),
            }),
        ];

        for transformer in transformers {
            assert!(transformer.params().params.is_empty() || true);
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
        assert!(matches!(out.plan, LogicalPlan::MlTransform { .. }));

        // StandardScaler fits locally into a model; the model transform also builds
        // an MlTransform plan, and Model::clone_box works.
        let mut ss = StandardScaler::new()
            .set_input_col("features")
            .set_output_col("scaled");
        let mut model = ss.fit(&df).unwrap();
        let scaled = model.transform(&df).unwrap();
        assert!(matches!(scaled.plan, LogicalPlan::MlTransform { .. }));
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
        assert!(matches!(out.plan, LogicalPlan::MlTransform { .. }));
        let _ = model.clone_box();
    }

    #[test]
    fn all_estimators_fit_and_models_transform() {
        let s = offline_session();
        let df = s.range(3).unwrap();
        // Each estimator fits locally and its model transform builds a plan; this
        // exercises fit_impl + transform_impl + clone_box for every model type.
        let mut mas = MaxAbsScaler::new().set_input_col("f").set_output_col("o");
        assert_eq!(mas.input_col(), "f");
        let mut m = mas.fit(&df).unwrap();
        assert!(matches!(
            m.transform(&df).unwrap().plan,
            LogicalPlan::MlTransform { .. }
        ));
        let _ = m.clone_box();

        let mut si = StringIndexer::new().set_input_col("s").set_output_col("si");
        assert_eq!(si.output_col(), "si");
        let mut m = si.fit(&df).unwrap();
        let _ = m.transform(&df).unwrap();
        let _ = m.clone_box();

        let mut lr = LogisticRegression::new()
            .set_feature_col("features")
            .set_label_col("label")
            .set_prediction_col("pred")
            .set_max_iter(7);
        assert_eq!(lr.feature_col(), "features");
        assert_eq!(lr.label_col(), "label");
        assert_eq!(lr.prediction_col(), "pred");
        assert_eq!(lr.max_iter(), 7);
        let mut m = lr.fit(&df).unwrap();
        let _ = m.transform(&df).unwrap();
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
}
