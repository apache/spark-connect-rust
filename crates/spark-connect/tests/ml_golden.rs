//! Golden parity test for ML operators and relations.
//!
//! The transforms of the Rust ML API must serialize to the exact same protobuf that the reference
//! PySpark client produces for the same transforms of `pyspark.ml`. The goldens live in
//! `tests/golden/ml.jsonl` (captured by `scripts/capture_ml_golden.py`): the plan of each
//! transformed DataFrame, and the ML commands of the fit of its model with their results, which
//! the replay of `common/ml_golden.rs` serves here, so that no Spark server is needed. The plan IDs,
//! origins and transformer uids of the relations are cleared on both sides
//! (`common::normalize_relation`), and the fit's commands are compared as the replay normalizes
//! them. A case that is missing or mismatches FAILS the test.

mod common;
#[path = "common/ml_golden.rs"]
mod ml_golden;

use std::collections::BTreeSet;

use ml_golden::{assert_commands, ReplayServer};
use spark_connect::ml::{Estimator, MlOperator, OperatorType, Params, StandardScaler, Transformer};

/// The StandardScaler of the capture, fitted on its vectors through the replay of the golden fit,
/// sends PySpark's Fit, and transforms the vectors into the relation of PySpark's transform: of
/// the model's object, the vectors as its input, and the params of the fit.
#[test]
fn standard_scaler_transform_matches_official_pyspark() {
    let goldens = common::load_ml_goldens("ml.jsonl");
    let names: BTreeSet<&str> = goldens.keys().map(String::as_str).collect();
    assert_eq!(
        names,
        BTreeSet::from(["standard_scaler_transform"]),
        "the cases of ml.jsonl"
    );
    let golden = &goldens["standard_scaler_transform"];
    let fit = golden.fit.as_ref().expect("the golden fits a model");
    let (server, session) = ReplayServer::start(&[fit]);
    let data = common::ml_vectors(&session, 3);
    let mut scaler = StandardScaler::new()
        .set_input_col("features")
        .set_output_col("scaled_features");
    let mut model = scaler.fit(&data).unwrap();
    assert_commands(&server, fit, "standard_scaler_transform: the fit");
    let mut relation = model.transform(&data).unwrap().to_relation().unwrap();
    common::normalize_relation(&mut relation);
    assert_eq!(relation, golden.relation, "standard_scaler_transform");
}

#[test]
fn test_ml_operator_proto() {
    // Test: MlOperator proto serialization
    let op = MlOperator::new(
        "org.apache.spark.ml.feature.StandardScaler",
        OperatorType::Estimator,
    );
    let proto_op = op.to_proto();

    assert_eq!(proto_op.name, "org.apache.spark.ml.feature.StandardScaler");
    assert_eq!(proto_op.r#type, OperatorType::Estimator.to_proto() as i32);
    assert!(!proto_op.uid.is_empty());
}

#[test]
fn test_ml_params_proto() {
    // Test: Params proto serialization
    let params = Params::new()
        .set_param_string("inputCol", "features")
        .set_param_string("outputCol", "scaled_features");

    let proto_params = params.to_proto();
    assert!(proto_params.params.contains_key("inputCol"));
    assert!(proto_params.params.contains_key("outputCol"));
}

#[test]
fn test_ml_operator_types() {
    // Test: all operator types convert correctly to/from proto
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
fn test_standard_scaler_operator_name() {
    // Test: StandardScaler operator has correct qualified name
    let scaler = spark_connect::ml::StandardScaler::new();
    let op = scaler.operator();
    assert_eq!(op.name, "org.apache.spark.ml.feature.StandardScaler");
    assert_eq!(op.op_type, OperatorType::Estimator);
}
