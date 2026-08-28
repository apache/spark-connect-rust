//! Extra unit coverage for `ml.rs`.
//!
//! Exercises the pure-logic surface the golden tests don't reach directly:
//! `Params`/`OperatorType`/`MlOperator` proto conversions, every ML class's
//! builders + getters + `Default`, and the `Estimator`/`Transformer`/`Evaluator`
//! trait accessors (`operator`/`operator_mut`/`params`/`params_mut`). Nothing here
//! contacts a Spark server: no `fit`/`transform`/`evaluate`/`collect` is called, so
//! these run in the default (server-less) `cargo test` pass.

use spark_connect::ml::{
    BinaryClassificationEvaluator, CrossValidator, Estimator, Evaluator, LogisticRegression,
    LogisticRegressionModel, MaxAbsScaler, MaxAbsScalerModel, MlOperator, Model,
    MulticlassClassificationEvaluator, OperatorType, Params, Pipeline, PipelineModel,
    RegressionEvaluator, StandardScaler, StringIndexer, StringIndexerModel, Transformer,
    VectorAssembler,
};

#[test]
fn params_operator_type_and_ml_operator_conversions() {
    // Params setters + get_param (Some/None) + to_proto (literal branch) + from_proto round-trip.
    let p = Params::new()
        .set_param_int("i", 7)
        .set_param_double("d", 1.5)
        .set_param_string("s", "v")
        .set_param_bool("b", true);
    assert!(p.get_param("i").is_some());
    assert!(p.get_param("missing").is_none());
    let mp = p.to_proto();
    let p2 = Params::from_proto(&mp);
    assert!(p2.get_param("s").is_some());

    // OperatorType round-trips for every variant, plus the unknown-code fallback.
    for ot in [
        OperatorType::Estimator,
        OperatorType::Transformer,
        OperatorType::Evaluator,
        OperatorType::Model,
    ] {
        let code = ot.to_proto() as i32;
        assert_eq!(OperatorType::from_proto(code), ot);
    }
    assert_eq!(OperatorType::from_proto(999), OperatorType::Transformer);

    // MlOperator new/with_uid/to_proto/from_proto.
    let op = MlOperator::with_uid("n", "u", OperatorType::Model);
    let round = MlOperator::from_proto(&op.to_proto());
    assert_eq!(round.name, "n");
    assert_eq!(round.uid, "u");
    let _ = MlOperator::new("x", OperatorType::Estimator);
}

#[test]
fn estimator_builders_and_accessors() {
    // StandardScaler
    let mut e = StandardScaler::new().set_input_col("f").set_output_col("o");
    assert_eq!(e.input_col(), "f");
    assert_eq!(e.output_col(), "o");
    let _ = e.operator();
    let _ = e.operator_mut();
    let _ = e.params();
    let _ = e.params_mut();
    let _ = StandardScaler::default();

    // StringIndexer
    let mut e = StringIndexer::new().set_input_col("f").set_output_col("o");
    assert_eq!(e.input_col(), "f");
    assert_eq!(e.output_col(), "o");
    let _ = e.operator();
    let _ = e.operator_mut();
    let _ = e.params();
    let _ = e.params_mut();
    let _ = StringIndexer::default();

    // MaxAbsScaler
    let mut e = MaxAbsScaler::new().set_input_col("f").set_output_col("o");
    assert_eq!(e.input_col(), "f");
    assert_eq!(e.output_col(), "o");
    let _ = e.operator();
    let _ = e.operator_mut();
    let _ = e.params();
    let _ = e.params_mut();
    let _ = MaxAbsScaler::default();

    // LogisticRegression
    let mut e = LogisticRegression::new()
        .set_feature_col("f")
        .set_label_col("l")
        .set_prediction_col("p")
        .set_max_iter(5);
    assert_eq!(e.feature_col(), "f");
    assert_eq!(e.label_col(), "l");
    assert_eq!(e.prediction_col(), "p");
    assert_eq!(e.max_iter(), 5);
    let _ = e.operator();
    let _ = e.operator_mut();
    let _ = e.params();
    let _ = e.params_mut();
    let _ = LogisticRegression::default();

    // Pipeline
    let mut e = Pipeline::new().set_stages(vec!["a", "b"]);
    assert_eq!(e.stages().len(), 2);
    let _ = e.operator();
    let _ = e.operator_mut();
    let _ = e.params();
    let _ = e.params_mut();
    let _ = Pipeline::default();

    // CrossValidator
    let mut e = CrossValidator::new()
        .set_num_folds(4)
        .set_parallelism(2)
        .set_seed(42);
    assert_eq!(e.num_folds(), 4);
    assert_eq!(e.parallelism(), 2);
    let _ = e.operator();
    let _ = e.operator_mut();
    let _ = e.params();
    let _ = e.params_mut();
    let _ = CrossValidator::default();
}

#[test]
fn transformer_and_model_accessors() {
    // VectorAssembler (a Transformer).
    let mut t = VectorAssembler::new()
        .set_input_cols(vec!["a", "b"])
        .set_output_col("o");
    assert_eq!(t.input_cols().len(), 2);
    assert_eq!(t.output_col(), "o");
    let _ = t.operator();
    let _ = t.operator_mut();
    let _ = t.params();
    let _ = t.params_mut();
    let _ = VectorAssembler::default();

    // Models with public fields: construct directly and exercise Transformer + Model methods.
    let mut m = StringIndexerModel {
        operator: MlOperator::new("si", OperatorType::Model),
        params: Params::new(),
        input_col: "a".to_string(),
        output_col: "b".to_string(),
    };
    let _ = m.operator();
    let _ = m.operator_mut();
    let _ = m.params();
    let _ = m.params_mut();
    let _ = m.clone_box();

    let mut m = MaxAbsScalerModel {
        operator: MlOperator::new("ma", OperatorType::Model),
        params: Params::new(),
        input_col: "a".to_string(),
        output_col: "b".to_string(),
    };
    let _ = m.operator();
    let _ = m.operator_mut();
    let _ = m.params();
    let _ = m.params_mut();
    let _ = m.clone_box();

    let mut m = LogisticRegressionModel {
        operator: MlOperator::new("lr", OperatorType::Model),
        params: Params::new(),
        feature_col: "f".to_string(),
        label_col: "l".to_string(),
        prediction_col: "p".to_string(),
    };
    let _ = m.operator();
    let _ = m.operator_mut();
    let _ = m.params();
    let _ = m.params_mut();
    let _ = m.clone_box();

    let mut m = PipelineModel {
        operator: MlOperator::new("pl", OperatorType::Model),
        params: Params::new(),
        stages: vec!["a".to_string()],
    };
    let _ = m.operator();
    let _ = m.operator_mut();
    let _ = m.params();
    let _ = m.params_mut();
    let _ = m.clone_box();
}

#[test]
fn evaluator_builders_and_accessors() {
    let e = RegressionEvaluator::new()
        .set_label_col("l")
        .set_prediction_col("p")
        .set_metric_name("mae");
    assert_eq!(e.label_col(), "l");
    assert_eq!(e.prediction_col(), "p");
    assert_eq!(e.metric_name(), "mae");
    let _ = e.operator();
    let _ = e.params();
    let _ = RegressionEvaluator::default();

    let e = BinaryClassificationEvaluator::new()
        .set_label_col("l")
        .set_score_col("s")
        .set_metric_name("areaUnderPR");
    assert_eq!(e.label_col(), "l");
    assert_eq!(e.score_col(), "s");
    assert_eq!(e.metric_name(), "areaUnderPR");
    let _ = e.operator();
    let _ = e.params();
    let _ = BinaryClassificationEvaluator::default();

    let e = MulticlassClassificationEvaluator::new()
        .set_label_col("l")
        .set_prediction_col("p")
        .set_metric_name("accuracy");
    assert_eq!(e.label_col(), "l");
    assert_eq!(e.prediction_col(), "p");
    assert_eq!(e.metric_name(), "accuracy");
    let _ = e.operator();
    let _ = e.params();
    let _ = MulticlassClassificationEvaluator::default();
}
