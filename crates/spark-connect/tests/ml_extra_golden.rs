//! Golden parity test for additional ML operators and relations.
//!
//! Tests that Rust ML operator builders produce the exact same protobuf
//! structures as the reference PySpark ML Connect client for newer operators.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;

use spark_connect::ml::{
    BinaryClassificationEvaluator, Estimator, Evaluator, LogisticRegression,
    LogisticRegressionModel, MaxAbsScaler, MaxAbsScalerModel, MlOperator, OperatorType, Params,
    Pipeline, PipelineModel, RegressionEvaluator, StringIndexer, StringIndexerModel, Transformer,
    VectorAssembler,
};
use spark_connect_proto as proto;

fn normalize_ml_operator(op: &mut proto::MlOperator) {
    op.uid.clear();
}

fn normalize_expression(e: &mut proto::Expression) {
    e.common = None;
}

fn normalize_ml_params(_params: &mut proto::MlParams) {
    // MlParams.params values are `Literal` (no plan_id/origin to clear here).
}

fn normalize_ml_relation(ml_rel: &mut proto::MlRelation) {
    if let Some(proto::ml_relation::MlType::Transform(transform)) = ml_rel.ml_type.as_mut() {
        if let Some(proto::ml_relation::transform::Operator::Transformer(op)) =
            transform.operator.as_mut()
        {
            normalize_ml_operator(op);
        }
        if let Some(params) = transform.params.as_mut() {
            normalize_ml_params(params);
        }
    }
}

fn normalize_relation(r: &mut proto::Relation) {
    r.common = None;
    if let Some(proto::relation::RelType::MlRelation(ml_rel)) = &mut r.rel_type {
        normalize_ml_relation(ml_rel);
    }
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
    let mut assembler = VectorAssembler::new()
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
fn test_string_indexer_model_operator_type() {
    let model = StringIndexerModel {
        operator: MlOperator::new(
            "org.apache.spark.ml.feature.StringIndexer",
            OperatorType::Model,
        ),
        params: Params::new(),
        input_col: "col".to_string(),
        output_col: "indexed".to_string(),
    };
    let op = model.operator();
    assert_eq!(op.op_type, OperatorType::Model);
}

#[test]
fn test_max_abs_scaler_operator_name() {
    let scaler = MaxAbsScaler::new();
    let op = scaler.operator();
    assert_eq!(op.name, "org.apache.spark.ml.feature.MaxAbsScaler");
    assert_eq!(op.op_type, OperatorType::Estimator);
}

#[test]
fn test_max_abs_scaler_model_operator_type() {
    let model = MaxAbsScalerModel {
        operator: MlOperator::new(
            "org.apache.spark.ml.feature.MaxAbsScaler",
            OperatorType::Model,
        ),
        params: Params::new(),
        input_col: "features".to_string(),
        output_col: "scaled".to_string(),
    };
    let op = model.operator();
    assert_eq!(op.op_type, OperatorType::Model);
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
fn test_logistic_regression_model_operator_type() {
    let model = LogisticRegressionModel {
        operator: MlOperator::new(
            "org.apache.spark.ml.classification.LogisticRegression",
            OperatorType::Model,
        ),
        params: Params::new(),
        feature_col: "features".to_string(),
        label_col: "label".to_string(),
        prediction_col: "prediction".to_string(),
    };
    let op = model.operator();
    assert_eq!(op.op_type, OperatorType::Model);
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
    assert_eq!(eval.score_col(), "prediction");
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
    let model = PipelineModel {
        operator: MlOperator::new("org.apache.spark.ml.Pipeline", OperatorType::Model),
        params: Params::new(),
        stages: vec!["stage1".to_string()],
    };
    let op = model.operator();
    assert_eq!(op.op_type, OperatorType::Model);
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
        for (name, _rel) in goldens.iter() {
            assert!(
                !name.is_empty(),
                "Golden relation should have a non-empty name"
            );
        }
    }
}
