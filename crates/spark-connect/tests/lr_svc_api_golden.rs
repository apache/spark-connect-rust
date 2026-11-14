// SPDX-License-Identifier: Apache-2.0
//! The typed accessors of LogisticRegression, LinearSVC and their models and summaries, against
//! the entries that official PySpark 4.3.0 captured (`tests/golden/classification_*.json`). No
//! Spark server is needed: the replay server of `common/ml_golden.rs` answers each ML command with
//! the golden result of that command, and the test checks that the accessor sent the golden
//! commands and returned the golden value. The commands are compared after the golden's
//! normalization, per entry (see `normalize_commands`).

#[path = "common/ml_golden.rs"]
mod ml_golden;

use base64::{engine::general_purpose::STANDARD, Engine};
use ml_golden::{assert_commands, ReplayServer};
use prost::Message;
use serde_json::{json, Value};
use spark_connect::ml::{
    BinaryClassificationSummary, BinaryLogisticRegressionSummary,
    BinaryLogisticRegressionTrainingSummary, ClassificationModel, ClassificationSummary, Estimator,
    LinearSVC, LinearSVCModel, LinearSVCSummary, LinearSVCTrainingSummary, LogisticRegression,
    LogisticRegressionEvaluation, LogisticRegressionModel, LogisticRegressionModelSummary,
    LogisticRegressionSummary, LogisticRegressionSummaryApi, LogisticRegressionTrainingSummary,
    MlMatrix, MlVector, Model, Params, PredictionModel, ProbabilisticClassificationModel,
    TrainingSummary, Transformer,
};
use spark_connect::SparkSession;
use spark_connect_core::error::Result as CoreResult;
use spark_connect_proto as proto;

const MODELS: &str = include_str!("../../../tests/golden/classification_models.json");
const SUMMARIES: &str = include_str!("../../../tests/golden/classification_summaries.json");
const PERSISTENCE: &str = include_str!("../../../tests/golden/classification_persistence.json");

fn golden(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

/// Assert that `actual` is the golden `expected`: numbers to 1e-9 relative, the rest exactly.
fn assert_close(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!(
                (a - e).abs() <= 1e-9 * e.abs().max(1.0),
                "{path}: {a} != {e}"
            );
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "{path}: length");
            for (index, (a, e)) in a.iter().zip(e).enumerate() {
                assert_close(a, e, &format!("{path}[{index}]"));
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            let mut keys: Vec<_> = a.keys().collect();
            let mut expected_keys: Vec<_> = e.keys().collect();
            keys.sort();
            expected_keys.sort();
            assert_eq!(keys, expected_keys, "{path}: keys");
            for (key, e) in e {
                assert_close(&a[key], e, &format!("{path}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

fn vector_json(vector: &MlVector) -> Value {
    match vector {
        MlVector::Dense(values) => json!({
            "size": values.len(), "type": "dense", "values": values,
        }),
        MlVector::Sparse {
            size,
            indices,
            values,
        } => json!({
            "size": size, "type": "sparse", "indices": indices, "values": values,
        }),
    }
}

fn matrix_json(matrix: &MlMatrix) -> Value {
    match matrix {
        MlMatrix::Dense {
            num_rows,
            num_cols,
            values,
            is_transposed,
        } => json!({
            "numRows": num_rows, "numCols": num_cols, "values": values,
            "isTransposed": is_transposed, "type": "dense",
        }),
        other => panic!("no golden sparse matrix in this family: {other:?}"),
    }
}

/// The member `key` of a golden model entry, as a feature vector of its name's suffix.
fn feature_vector(key: &str) -> MlVector {
    match key.rsplit('-').next().unwrap() {
        "dense" => MlVector::Dense(vec![0.5, 1.5]),
        "dense2" => MlVector::Dense(vec![3.0, -0.25]),
        "sparse" => MlVector::Sparse {
            size: 2,
            indices: vec![1],
            values: vec![2.0],
        },
        other => panic!("unknown feature vector {other}"),
    }
}

type Member = CoreResult<Value>;

fn num<T: Into<Value>>(value: CoreResult<T>) -> Member {
    value.map(Into::into)
}

fn list(value: CoreResult<Vec<f64>>) -> Member {
    value.map(|values| json!(values))
}

/// The members of a classification summary (`_ClassificationSummary`).
fn classification<S: ClassificationSummary>(summary: &S, key: &str) -> Option<Member> {
    if let Some(beta) = key.strip_prefix("fMeasureByLabel-") {
        return Some(list(summary.f_measure_by_label(beta.parse().unwrap())));
    }
    if let Some(beta) = key.strip_prefix("weightedFMeasure-") {
        return Some(num(summary.weighted_f_measure(beta.parse().unwrap())));
    }
    Some(match key {
        "accuracy" => num(summary.accuracy()),
        "falsePositiveRateByLabel" => list(summary.false_positive_rate_by_label()),
        "labelCol" => num(summary.label_col()),
        "labels" => list(summary.labels()),
        "precisionByLabel" => list(summary.precision_by_label()),
        "predictionCol" => num(summary.prediction_col()),
        "recallByLabel" => list(summary.recall_by_label()),
        "truePositiveRateByLabel" => list(summary.true_positive_rate_by_label()),
        "weightCol" => num(summary.weight_col()),
        "weightedFalsePositiveRate" => num(summary.weighted_false_positive_rate()),
        "weightedPrecision" => num(summary.weighted_precision()),
        "weightedRecall" => num(summary.weighted_recall()),
        "weightedTruePositiveRate" => num(summary.weighted_true_positive_rate()),
        _ => return None,
    })
}

/// The members of a binary classification summary (`_BinaryClassificationSummary`).
fn binary<S: BinaryClassificationSummary>(summary: &S, key: &str) -> Option<Member> {
    Some(match key {
        "areaUnderROC" => num(summary.area_under_roc()),
        "scoreCol" => num(summary.score_col()),
        _ => return None,
    })
}

/// The members of a training summary (`_TrainingSummary`).
fn training<S: TrainingSummary>(summary: &S, key: &str) -> Option<Member> {
    Some(match key {
        "objectiveHistory" => list(summary.objective_history()),
        "totalIterations" => num(summary.total_iterations()),
        _ => return None,
    })
}

/// The members of the logistic regression summaries (`_LogisticRegressionSummary`).
fn logistic<S: LogisticRegressionSummaryApi>(summary: &S, key: &str) -> Option<Member> {
    Some(match key {
        "featuresCol" => num(summary.features_col()),
        "probabilityCol" => num(summary.probability_col()),
        _ => return None,
    })
}

/// The DataFrame-valued members, which the e2e test checks against the server's rows, and the
/// Python-only error of a multiclass summary's `roc`, which its type does not have.
const SKIPPED_MEMBERS: &[&str] = &[
    "fMeasureByThreshold",
    "pr",
    "precisionByThreshold",
    "predictions",
    "recallByThreshold",
    "roc",
    "roc-on-multiclass",
];

/// Check every golden member of a summary: its value or its error, and its commands. The
/// dispatch returns None for a member the summary does not have, which is a mistake here.
fn check_summary_members(
    server: &ReplayServer,
    members: &Value,
    dispatch: &dyn Fn(&str) -> Option<Member>,
    label: &str,
) {
    for (key, entry) in members.as_object().unwrap() {
        if key.starts_with("repr")
            || key.starts_with("str")
            || SKIPPED_MEMBERS.contains(&key.as_str())
        {
            continue;
        }
        let Some(member) = dispatch(key) else {
            panic!("{label}: no accessor for {key}");
        };
        check_member(server, entry, member, &format!("{label}.{key}"));
    }
}

/// Check one member of a golden entry against its actual value or error, and its commands.
fn check_member(server: &ReplayServer, entry: &Value, actual: Member, label: &str) {
    match entry.get("error") {
        Some(error) => {
            let message = error["message"].as_str().unwrap();
            let error = actual.expect_err(&format!("{label}: expected an error"));
            assert!(
                error.to_string().contains(message),
                "{label}: {error} does not carry {message}"
            );
        }
        None => assert_close(
            &actual.unwrap_or_else(|error| panic!("{label}: {error}")),
            &entry["value"],
            label,
        ),
    }
    assert_commands(server, entry, label);
}

/// The summary class name of a golden entry, as PySpark names it.
fn class_of(value: &Value) -> &str {
    value["type"].as_str().unwrap()
}

/// The model's members of a golden entry, for a model of the given family.
fn check_model_members(
    server: &ReplayServer,
    members: &Value,
    model: &dyn Fn(&str) -> Option<Member>,
    label: &str,
) {
    for (key, entry) in members.as_object().unwrap() {
        if key.starts_with("repr") || key.starts_with("str") || key == "transform" {
            continue;
        }
        if key == "summary" || key == "evaluate" || key.starts_with("summary") {
            continue;
        }
        let Some(member) = model(key) else {
            panic!("{label}: no accessor for {key}");
        };
        check_member(server, entry, member, &format!("{label}.{key}"));
    }
}

/// The LogisticRegression model's members, of the golden's names.
fn logistic_model_member(model: &LogisticRegressionModel, key: &str) -> Option<Member> {
    if key.starts_with("predictRaw-") {
        return Some(match model.predict_raw(&feature_vector(key)) {
            Ok(v) => Ok(vector_json(&v)),
            Err(e) => Err(e),
        });
    }
    if key.starts_with("predictProbability-") {
        return Some(match model.predict_probability(&feature_vector(key)) {
            Ok(v) => Ok(vector_json(&v)),
            Err(e) => Err(e),
        });
    }
    if key.starts_with("predict-") {
        return Some(num(model.predict(&feature_vector(key))));
    }
    Some(match key {
        "coefficients" => model.coefficients().map(|v| vector_json(&v)),
        "coefficientMatrix" => model.coefficient_matrix().map(|m| matrix_json(&m)),
        "intercept" => num(model.intercept()),
        "interceptVector" => model.intercept_vector().map(|v| vector_json(&v)),
        "numClasses" => num(model.num_classes()),
        "numFeatures" => num(model.num_features()),
        "hasSummary" => Ok(json!(model.has_summary())),
        _ => return None,
    })
}

/// The LinearSVC model's members, of the golden's names.
fn linear_svc_model_member(model: &LinearSVCModel, key: &str) -> Option<Member> {
    if key.starts_with("predictRaw-") {
        return Some(
            model
                .predict_raw(&feature_vector(key))
                .map(|v| vector_json(&v)),
        );
    }
    if key.starts_with("predict-") {
        return Some(num(model.predict(&feature_vector(key))));
    }
    Some(match key {
        "coefficients" => model.coefficients().map(|v| vector_json(&v)),
        "intercept" => num(model.intercept()),
        "numClasses" => num(model.num_classes()),
        "numFeatures" => num(model.num_features()),
        "hasSummary" => Ok(json!(model.has_summary())),
        _ => return None,
    })
}

/// Fit a LogisticRegression on a dataset of the golden's fit, with the fit's own params, which
/// the golden fit sent. Returns the model.
fn fit_logistic(
    server: &ReplayServer,
    session: &SparkSession,
    fit: &Value,
) -> LogisticRegressionModel {
    let fit_entry = &fit["fit"];
    let command = proto::MlCommand::decode(
        STANDARD
            .decode(fit_entry["commands"][0].as_str().unwrap())
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    let Some(proto::ml_command::Command::Fit(sent)) = command.command else {
        panic!("the golden fit is not a Fit");
    };
    let mut estimator = LogisticRegression::new();
    *estimator.params_mut() = Params::from_proto(&sent.params.unwrap());
    let train = session.range(1).unwrap();
    let model = estimator.fit(&train).unwrap();
    // PySpark's fit fetches numClasses for the class of its training summary, which the Rust
    // model fetches when `summary()` is called: the fit and its first summary send the commands
    // of the golden fit.
    model.summary().unwrap();
    assert_commands(server, fit_entry, "fit");
    model
}

fn fit_linear_svc(server: &ReplayServer, session: &SparkSession, fit: &Value) -> LinearSVCModel {
    let fit_entry = &fit["fit"];
    let command = proto::MlCommand::decode(
        STANDARD
            .decode(fit_entry["commands"][0].as_str().unwrap())
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    let Some(proto::ml_command::Command::Fit(sent)) = command.command else {
        panic!("the golden fit is not a Fit");
    };
    let mut estimator = LinearSVC::new();
    *estimator.params_mut() = Params::from_proto(&sent.params.unwrap());
    let train = session.range(1).unwrap();
    let model = estimator.fit(&train).unwrap();
    assert_commands(server, fit_entry, "fit");
    model
}

/// The members of a LogisticRegression model's golden entry, checked on a fitted model. The
/// model's evaluate and summary are checked with their own summaries.
fn check_logistic_model(
    server: &ReplayServer,
    session: &SparkSession,
    models: &Value,
    family: &str,
) {
    let entry = &models["models"][family];
    let model = fit_logistic(server, session, &models["fits"][family]);
    check_model_members(
        server,
        &entry["members"],
        &|key| logistic_model_member(&model, key),
        family,
    );
    // The summary of a fit: a Binary or a multiclass summary, as its numClasses decides.
    let summary = &entry["members"]["summary"];
    let fitted = model.summary().unwrap();
    // That summary fetched numClasses again, which PySpark's `summary` (made at the fit) does not.
    server.take_requests();
    match (&fitted, class_of(&summary["value"])) {
        (LogisticRegressionModelSummary::Binary(_), "BinaryLogisticRegressionTrainingSummary") => {}
        (LogisticRegressionModelSummary::Multiclass(_), "LogisticRegressionTrainingSummary") => {}
        (_, other) => panic!("{family}: summary is not {other}"),
    }
    let training = golden_summary_members(&models_summaries(), family, "training");
    match fitted {
        LogisticRegressionModelSummary::Binary(s) => {
            check_summary_members(
                server,
                &training,
                &|key| binary_training_member(&s, key),
                family,
            );
        }
        LogisticRegressionModelSummary::Multiclass(s) => {
            check_summary_members(
                server,
                &training,
                &|key| multiclass_training_member(&s, key),
                family,
            );
        }
    }
    // The evaluate of the model on a dataset: its own summary, binary or multiclass.
    let evaluate = &entry["members"]["evaluate"];
    let golden = golden_summary_members(&models_summaries(), family, "evaluate");
    let eval_df = session.range(1).unwrap();
    let evaluated = model.evaluate(&eval_df).unwrap();
    assert_commands(server, evaluate, "evaluate");
    match evaluated {
        LogisticRegressionEvaluation::Binary(s) => {
            assert_eq!(
                class_of(&evaluate["value"]),
                "BinaryLogisticRegressionSummary"
            );
            check_summary_members(
                server,
                &golden,
                &|key| binary_evaluate_member(&s, key),
                family,
            );
        }
        LogisticRegressionEvaluation::Multiclass(s) => {
            assert_eq!(class_of(&evaluate["value"]), "LogisticRegressionSummary");
            check_summary_members(
                server,
                &golden,
                &|key| multiclass_evaluate_member(&s, key),
                family,
            );
        }
    }
}

fn models_summaries() -> Value {
    golden(SUMMARIES)["summaries"].clone()
}

/// The golden members of a summary of a family: `training` or `evaluate`.
fn golden_summary_members(summaries: &Value, family: &str, kind: &str) -> Value {
    summaries[format!("{family}.{kind}").as_str()]["members"].clone()
}

fn binary_training_member(
    s: &BinaryLogisticRegressionTrainingSummary,
    key: &str,
) -> Option<Member> {
    classification(s, key)
        .or_else(|| binary(s, key))
        .or_else(|| logistic(s, key))
        .or_else(|| training(s, key))
}

fn multiclass_training_member(s: &LogisticRegressionTrainingSummary, key: &str) -> Option<Member> {
    classification(s, key)
        .or_else(|| logistic(s, key))
        .or_else(|| training(s, key))
}

fn binary_evaluate_member(s: &BinaryLogisticRegressionSummary, key: &str) -> Option<Member> {
    classification(s, key)
        .or_else(|| binary(s, key))
        .or_else(|| logistic(s, key))
}

fn multiclass_evaluate_member(s: &LogisticRegressionSummary, key: &str) -> Option<Member> {
    classification(s, key).or_else(|| logistic(s, key))
}

fn linear_svc_training_member(s: &LinearSVCTrainingSummary, key: &str) -> Option<Member> {
    classification(s, key)
        .or_else(|| binary(s, key))
        .or_else(|| training(s, key))
}

fn linear_svc_evaluate_member(s: &LinearSVCSummary, key: &str) -> Option<Member> {
    classification(s, key).or_else(|| binary(s, key))
}

/// The golden entries of a family's model, which its test replays: the fit, the model entry with
/// its members, and the training and evaluate summaries.
fn model_entries<'a>(models: &'a Value, summaries: &'a Value, family: &str) -> [&'a Value; 4] {
    [
        &models["fits"][family]["fit"],
        &models["models"][family],
        &summaries["summaries"][format!("{family}.training").as_str()],
        &summaries["summaries"][format!("{family}.evaluate").as_str()],
    ]
}

/// The LogisticRegression model of `family`, against its golden entries.
fn check_logistic_family(family: &str) {
    let models = golden(MODELS);
    let summaries = golden(SUMMARIES);
    let (server, session) = ReplayServer::start(&model_entries(&models, &summaries, family));
    check_logistic_model(&server, &session, &models, family);
}

#[test]
fn logistic_regression_binomial_model_matches_official_pyspark() {
    check_logistic_family("LogisticRegression.binomial");
}

#[test]
fn logistic_regression_multinomial_model_matches_official_pyspark() {
    check_logistic_family("LogisticRegression.multinomial");
}

#[test]
fn linear_svc_model_matches_official_pyspark() {
    let models = golden(MODELS);
    let summaries = golden(SUMMARIES);
    let family = "LinearSVC";
    let (server, session) = ReplayServer::start(&model_entries(&models, &summaries, family));
    let entry = &models["models"][family];
    let model = fit_linear_svc(&server, &session, &models["fits"][family]);
    check_model_members(
        &server,
        &entry["members"],
        &|key| linear_svc_model_member(&model, key),
        family,
    );
    assert_eq!(
        class_of(&entry["members"]["summary()"]["value"]),
        "LinearSVCTrainingSummary"
    );
    let training = model.summary().unwrap();
    let golden_training = golden_summary_members(&summaries["summaries"], family, "training");
    check_summary_members(
        &server,
        &golden_training,
        &|key| linear_svc_training_member(&training, key),
        family,
    );
    let evaluate = &entry["members"]["evaluate"];
    let golden_evaluate = golden_summary_members(&summaries["summaries"], family, "evaluate");
    let eval_df = session.range(1).unwrap();
    let evaluated = model.evaluate(&eval_df).unwrap();
    assert_commands(&server, evaluate, "evaluate");
    assert_eq!(class_of(&evaluate["value"]), "LinearSVCSummary");
    check_summary_members(
        &server,
        &golden_evaluate,
        &|key| linear_svc_evaluate_member(&evaluated, key),
        family,
    );
}

/// The persistence of an estimator and of its model: `load`, then `save` of the loaded object,
/// as PySpark's persistence entries record them.
#[test]
fn logistic_regression_persistence_matches_official_pyspark() {
    let persistence = golden(PERSISTENCE)["persistence"].clone();
    for family in [
        "LogisticRegression.binomial",
        "LogisticRegression.multinomial",
    ] {
        let estimator_entry = &persistence[format!("{family}.estimator").as_str()];
        let model_entry = &persistence[format!("{family}.model").as_str()];
        let (server, session) = ReplayServer::start(&[estimator_entry, model_entry]);
        let estimator = LogisticRegression::load(&session, "/unused/path").unwrap();
        assert_commands(&server, &estimator_entry["load"], family);
        assert_eq!(
            estimator.operator().uid,
            result_uid(&estimator_entry["load"])
        );
        assert_eq!(
            Estimator::params(&estimator).to_proto(),
            params_proto(&estimator_entry["load"])
        );
        estimator
            .write()
            .session(&session)
            .overwrite()
            .save("/unused/path")
            .unwrap();
        assert_commands(&server, &estimator_entry["save"], family);

        let model = LogisticRegressionModel::load(&session, "/unused/path").unwrap();
        assert_commands(&server, &model_entry["load"], family);
        assert_eq!(model.operator().uid, result_uid(&model_entry["load"]));
        assert!(!model.has_summary());
        check_model_members(
            &server,
            &model_entry["members"]["members"],
            &|key| logistic_model_member(&model, key),
            family,
        );
        model
            .write()
            .unwrap()
            .overwrite()
            .save("/unused/path")
            .unwrap();
        assert_commands(&server, &model_entry["save"], family);
    }
}

#[test]
fn linear_svc_persistence_matches_official_pyspark() {
    let persistence = golden(PERSISTENCE)["persistence"].clone();
    let family = "LinearSVC";
    let estimator_entry = &persistence[format!("{family}.estimator").as_str()];
    let model_entry = &persistence[format!("{family}.model").as_str()];
    let (server, session) = ReplayServer::start(&[estimator_entry, model_entry]);
    let estimator = LinearSVC::load(&session, "/unused/path").unwrap();
    assert_commands(&server, &estimator_entry["load"], family);
    assert_eq!(
        estimator.operator().uid,
        result_uid(&estimator_entry["load"])
    );
    estimator
        .write()
        .session(&session)
        .overwrite()
        .save("/unused/path")
        .unwrap();
    assert_commands(&server, &estimator_entry["save"], family);

    let model = LinearSVCModel::load(&session, "/unused/path").unwrap();
    assert_commands(&server, &model_entry["load"], family);
    assert_eq!(model.operator().uid, result_uid(&model_entry["load"]));
    check_model_members(
        &server,
        &model_entry["members"]["members"],
        &|key| linear_svc_model_member(&model, key),
        family,
    );
    model
        .write()
        .unwrap()
        .overwrite()
        .save("/unused/path")
        .unwrap();
    assert_commands(&server, &model_entry["save"], family);
}

/// The uid that a golden load read: the golden normalizes it to a placeholder, so the loaded
/// uid is the placeholder too.
fn result_uid(load: &Value) -> String {
    let result = proto::MlCommandResult::decode(
        STANDARD
            .decode(load["results"][0].as_str().unwrap())
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    match result.result_type {
        Some(proto::ml_command_result::ResultType::OperatorInfo(info)) => info.uid.unwrap(),
        other => panic!("not an operator info: {other:?}"),
    }
}

/// The params that a golden load of an operator read: the `MlParams` of its operator info.
fn params_proto(load: &Value) -> proto::MlParams {
    let result = proto::MlCommandResult::decode(
        STANDARD
            .decode(load["results"][0].as_str().unwrap())
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    match result.result_type {
        Some(proto::ml_command_result::ResultType::OperatorInfo(info)) => info.params.unwrap(),
        other => panic!("not an operator info: {other:?}"),
    }
}

/// The setters of the models set the params of their transform, as PySpark's `_set` does.
#[test]
fn model_setters_set_the_params_of_the_transform() {
    use spark_connect_proto::expression::literal::LiteralType;
    let persistence = golden(PERSISTENCE)["persistence"].clone();
    let lr_entry = &persistence["LogisticRegression.binomial.model"];
    let svc_entry = &persistence["LinearSVC.model"];
    let (server, session) = ReplayServer::start(&[lr_entry, svc_entry]);
    let model = LogisticRegressionModel::load(&session, "/unused/path").unwrap();
    assert_commands(&server, &lr_entry["load"], "load");
    let model = model
        .set_features_col("x")
        .set_prediction_col("p")
        .set_probability_col("q")
        .set_raw_prediction_col("r")
        .set_threshold(0.7);
    let params = Transformer::params(&model).to_proto();
    let literal = |name: &str| params.params[name].literal_type.clone();
    assert_eq!(
        literal("featuresCol"),
        Some(LiteralType::String("x".into()))
    );
    assert_eq!(
        literal("predictionCol"),
        Some(LiteralType::String("p".into()))
    );
    assert_eq!(
        literal("probabilityCol"),
        Some(LiteralType::String("q".into()))
    );
    assert_eq!(
        literal("rawPredictionCol"),
        Some(LiteralType::String("r".into()))
    );
    assert_eq!(literal("threshold"), Some(LiteralType::Double(0.7)));
    // The threshold logic of PySpark over Spark Connect, on the model (reference probe): the
    // setters leave the other param set, so the thresholds that disagree with the threshold
    // raise the consistency error, and `thresholds` of two classes give the threshold.
    let model = model.set_thresholds([0.4, 0.6]);
    assert!(params_of(&model).contains_key("threshold"));
    let message = "Logistic Regression getThreshold found inconsistent values for threshold (0.7) \
                   and thresholds (equivalent to 0.6)";
    assert_eq!(model.threshold().unwrap_err().to_string(), message);
    assert_eq!(model.thresholds().unwrap_err().to_string(), message);
    let model = model.set_threshold(0.6);
    assert_eq!(model.thresholds().unwrap(), vec![1.0 - 0.6, 0.6]);
    assert!((model.threshold().unwrap() - 0.6).abs() < 1e-12);

    let svc = LinearSVCModel::load(&session, "/unused/path").unwrap();
    assert_commands(&server, &svc_entry["load"], "load");
    let svc = svc
        .set_features_col("x")
        .set_prediction_col("p")
        .set_raw_prediction_col("r")
        .set_threshold(0.7);
    let params = Transformer::params(&svc).to_proto();
    let literal = |name: &str| params.params[name].literal_type.clone();
    assert_eq!(
        literal("featuresCol"),
        Some(LiteralType::String("x".into()))
    );
    assert_eq!(
        literal("predictionCol"),
        Some(LiteralType::String("p".into()))
    );
    assert_eq!(
        literal("rawPredictionCol"),
        Some(LiteralType::String("r".into()))
    );
    assert_eq!(literal("threshold"), Some(LiteralType::Double(0.7)));
}

/// The params of a model's transform, as the wire sends them.
fn params_of(
    model: &LogisticRegressionModel,
) -> std::collections::HashMap<String, spark_connect_proto::expression::literal::LiteralType> {
    Transformer::params(model)
        .to_proto()
        .params
        .into_iter()
        .filter_map(|(name, literal)| literal.literal_type.map(|t| (name, t)))
        .collect()
}
