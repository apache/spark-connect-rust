//! Golden parity test for additional ML operators and relations.
//!
//! The transforms of more Rust ML stages (feature transformers and estimators, a classifier and a
//! pipeline) must serialize to the exact same protobuf that the reference PySpark client produces
//! for the same transforms of `pyspark.ml`. The goldens live in `tests/golden/ml_extra.jsonl`
//! (captured by `scripts/capture_ml_extra_golden.py`), and are compared as `ml_golden.rs` compares
//! those of `ml.jsonl`: the fit of a model through the replay of its golden commands, and the
//! relation of its transform. A case that is missing or mismatches FAILS the test.

mod common;
#[path = "common/ml_golden.rs"]
mod ml_golden;

use std::collections::{BTreeSet, HashMap};

use common::MlGolden;
use ml_golden::{assert_commands, ReplayServer};
use serde_json::json;
use spark_connect::column::{col, lit};
use spark_connect::ml::{
    BinaryClassificationEvaluator, ClassificationModel, Estimator, Evaluator, LogisticRegression,
    MaxAbsScaler, OperatorType, Pipeline, PipelineModel, PipelineStage, RegressionEvaluator,
    StringIndexer, Transformer, VectorAssembler,
};
use spark_connect::{DataFrame, SparkSession};

/// Compares the case `name` of the golden: `case` runs the Rust stage of the case on a session of
/// the replay of its golden fit, and gives the transformed DataFrame. The ML commands that it
/// sends are the golden fit's (none of a transformer), and the relation of the DataFrame is that
/// of PySpark's transform.
fn check(
    goldens: &HashMap<String, MlGolden>,
    name: &str,
    case: impl FnOnce(&SparkSession) -> DataFrame,
) {
    let golden = goldens
        .get(name)
        .unwrap_or_else(|| panic!("{name}: MISSING from ml_extra.jsonl"));
    let fit = golden
        .fit
        .clone()
        .unwrap_or_else(|| json!({"commands": [], "results": []}));
    let (server, session) = ReplayServer::start(&[&fit]);
    let transformed = case(&session);
    assert_commands(&server, &fit, &format!("{name}: the fit"));
    let mut relation = transformed.to_relation().unwrap();
    common::normalize_relation(&mut relation);
    assert_eq!(relation, golden.relation, "{name}");
}

#[test]
fn ml_transforms_match_official_pyspark() {
    let goldens = common::load_ml_goldens("ml_extra.jsonl");
    let cases = [
        "vector_assembler_transform",
        "string_indexer_transform",
        "max_abs_scaler_transform",
        "logistic_regression_transform",
        "pipeline_transform",
    ];
    let names: BTreeSet<&str> = goldens.keys().map(String::as_str).collect();
    assert_eq!(names, BTreeSet::from(cases), "the cases of ml_extra.jsonl");

    // A transformer, which sends no ML command: its transform names the transformer's class,
    // where that of a fitted model names the model's object.
    check(&goldens, "vector_assembler_transform", |session| {
        let mut assembler = VectorAssembler::new()
            .set_input_cols(vec!["features"])
            .set_output_col("vector");
        assembler
            .transform(&common::ml_vectors(session, 3))
            .unwrap()
    });
    check(&goldens, "string_indexer_transform", |session| {
        let categories = session
            .range(3)
            .unwrap()
            .select(vec![col("id").cast_str("string").alias("category")]);
        let mut indexer = StringIndexer::new()
            .set_input_col("category")
            .set_output_col("category_index");
        indexer
            .fit(&categories)
            .unwrap()
            .transform(&categories)
            .unwrap()
    });
    check(&goldens, "max_abs_scaler_transform", |session| {
        let data = common::ml_vectors(session, 3);
        let mut scaler = MaxAbsScaler::new()
            .set_input_col("features")
            .set_output_col("scaled_features");
        scaler.fit(&data).unwrap().transform(&data).unwrap()
    });
    check(&goldens, "logistic_regression_transform", |session| {
        let label = (col("id") % lit(2)).cast_str("double").alias("label");
        let labelled = session
            .range(4)
            .unwrap()
            .select(vec![common::ml_features().alias("features"), label]);
        let mut lr = LogisticRegression::new()
            .set_features_col("features")
            .set_label_col("label")
            .set_prediction_col("prediction");
        let mut model = lr.fit(&labelled).unwrap();
        // PySpark's fit reads `numClasses` right after the Fit, to choose the class of its
        // training summary; the Rust model reads it when its summary is asked (see
        // docs/compatibility.md), so the test reads it here, as the golden's second command.
        model.num_classes().unwrap();
        model.transform(&labelled).unwrap()
    });
    // A pipeline fits its estimator on the output of the transformer before it, and its model
    // transforms by the stages in order.
    check(&goldens, "pipeline_transform", |session| {
        let data = common::ml_vectors(session, 3);
        let assembler = VectorAssembler::new()
            .set_input_cols(vec!["features"])
            .set_output_col("vector");
        let scaler = MaxAbsScaler::new()
            .set_input_col("vector")
            .set_output_col("scaled");
        let mut pipeline = Pipeline::new().set_stages([
            PipelineStage::transformer(assembler),
            PipelineStage::estimator(scaler),
        ]);
        pipeline
            .fit_pipeline(&data)
            .unwrap()
            .transform(&data)
            .unwrap()
    });
}

#[test]
fn test_vector_assembler_operator_name() {
    let assembler = VectorAssembler::new();
    let op = assembler.operator();
    assert_eq!(op.name, "org.apache.spark.ml.feature.VectorAssembler");
    assert_eq!(op.op_type, OperatorType::Transformer);
}

#[test]
fn test_vector_assembler_proto() {
    let assembler = VectorAssembler::new()
        .set_input_cols(vec!["col1", "col2"])
        .set_output_col("vector");

    let proto_op = assembler.operator().to_proto();
    assert_eq!(proto_op.name, "org.apache.spark.ml.feature.VectorAssembler");

    let params = assembler.params().to_proto();
    assert!(params.params.contains_key("inputCols"));
    assert!(params.params.contains_key("outputCol"));
}

#[test]
fn test_string_indexer_operator_name() {
    let indexer = StringIndexer::new();
    let op = indexer.operator();
    assert_eq!(op.name, "org.apache.spark.ml.feature.StringIndexer");
    assert_eq!(op.op_type, OperatorType::Estimator);
}

#[test]
fn test_max_abs_scaler_operator_name() {
    let scaler = MaxAbsScaler::new();
    let op = scaler.operator();
    assert_eq!(op.name, "org.apache.spark.ml.feature.MaxAbsScaler");
    assert_eq!(op.op_type, OperatorType::Estimator);
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
fn test_regression_evaluator_evaluate() {
    let eval = RegressionEvaluator::new().set_metric_name("r2");
    assert_eq!(eval.metric_name(), "r2");
    assert_eq!(eval.label_col(), "label");
    assert_eq!(eval.prediction_col(), "prediction");
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
fn test_binary_classification_evaluator_metrics() {
    let eval = BinaryClassificationEvaluator::new().set_metric_name("areaUnderPR");
    assert_eq!(eval.metric_name(), "areaUnderPR");
    assert_eq!(eval.label_col(), "label");
    assert_eq!(eval.raw_prediction_col(), "rawPrediction");
}

#[test]
fn test_pipeline_operator_name() {
    let pipeline = Pipeline::new();
    let op = pipeline.operator();
    assert_eq!(op.name, "org.apache.spark.ml.Pipeline");
    assert_eq!(op.op_type, OperatorType::Estimator);
}

#[test]
fn test_pipeline_model_operator_type() {
    let model = PipelineModel::of_stages(vec![]);
    let op = model.operator();
    assert_eq!(op.op_type, OperatorType::Model);
    assert_eq!(op.name, "org.apache.spark.ml.PipelineModel");
}

#[test]
fn test_operator_types_proto_roundtrip() {
    let types = vec![
        OperatorType::Estimator,
        OperatorType::Transformer,
        OperatorType::Evaluator,
        OperatorType::Model,
    ];

    for op_type in types {
        let proto_type = op_type.to_proto();
        let back = OperatorType::from_proto(proto_type as i32);
        assert_eq!(op_type, back);
    }
}
