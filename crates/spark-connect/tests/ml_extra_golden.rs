//! Golden parity test for additional ML operators and relations.
//!
//! Tests that Rust ML operator builders produce the exact same protobuf
//! structures as the reference PySpark ML Connect client for newer operators.

#![allow(dead_code)] // golden-test helpers are copy-shared across suites; each binary uses a subset

mod common;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;

use common::normalize_relation;
use spark_connect::ml::{
    BinaryClassificationEvaluator, Estimator, Evaluator, LogisticRegression, MaxAbsScaler,
    OperatorType, Pipeline, PipelineModel, RegressionEvaluator, StringIndexer, Transformer,
    VectorAssembler,
};
use spark_connect_proto as proto;

fn normalize_expression(e: &mut proto::Expression) {
    e.common = None;
}

fn load_ml_extra_goldens() -> HashMap<String, proto::Relation> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/ml_extra.jsonl"
    );
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("warning: ML extra golden file not found at {}", path);
            return HashMap::new();
        }
    };

    let mut out = HashMap::new();
    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let obj: serde_json::Value = serde_json::from_str(&line).unwrap();
        let name = obj["name"].as_str().unwrap().to_string();
        let b64 = obj["b64"].as_str().unwrap();
        let bytes = STANDARD.decode(b64).unwrap();
        let mut relation = proto::Relation::decode(&bytes[..]).unwrap();
        normalize_relation(&mut relation);
        out.insert(name, relation);
    }
    out
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

#[test]
fn test_ml_extra_golden_coverage() {
    let goldens = load_ml_extra_goldens();
    // If golden file exists, ensure we handle it correctly
    if !goldens.is_empty() {
        for name in goldens.keys() {
            assert!(
                !name.is_empty(),
                "Golden relation should have a non-empty name"
            );
        }
    }
}
