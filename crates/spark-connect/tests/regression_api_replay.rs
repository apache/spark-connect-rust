// SPDX-License-Identifier: Apache-2.0
//! The typed accessors of the regression models, their summaries and their persistence, against
//! the entries that official PySpark 4.3.0 captured (`tests/golden/regression_*.json`). No Spark
//! server is needed: the replay server of `common/ml_golden.rs` answers each ML command with the
//! golden result of that command, and the test checks that the accessor sent the golden commands
//! and returned the golden value. The commands are compared after the golden's normalization, per
//! entry (see `normalize_commands`).
//!
//! The replay answers ML commands only. A DataFrame-valued member (the `transform`, the
//! `predictions` and `residuals` of a summary, the `describeTopics` of LDA) has a relation as its
//! command, so it is skipped. PySpark fetches the trees of an ensemble right after its fit and its
//! load, and the golden records that fetch in the entry of the fit or the load: the Rust fit and
//! load leave it to the first call of `trees`, so the test makes that call right after them. The
//! golden `trees` member is not compared: PySpark reads it of that fetch with no command of its
//! own, and the golden's three ids are one comma-joined result, normalized to one placeholder.

#[path = "common/ml_golden.rs"]
mod ml_golden;

use base64::{engine::general_purpose::STANDARD, Engine};
use ml_golden::{
    assert_commands, fixture, golden_commands, matrix, num, nums, vector, ReplayServer,
};
use prost::Message;
use serde_json::{json, Value};
use spark_connect::ml::{
    fetch, AFTSurvivalRegression, AFTSurvivalRegressionModel, DecisionTreeModel,
    DecisionTreeRegressionModel, DecisionTreeRegressor, Estimator, FMRegressionModel, FMRegressor,
    GBTRegressionModel, GBTRegressor, GeneralizedLinearRegression,
    GeneralizedLinearRegressionModel, GeneralizedLinearRegressionSummaryApi,
    GeneralizedLinearRegressionTrainingSummary, IsotonicRegression, IsotonicRegressionModel,
    LinearRegression, LinearRegressionModel, LinearRegressionSummaryApi,
    LinearRegressionTrainingSummary, MlMatrix, MlTarget, MlVector, Model, Params, PredictionModel,
    RandomForestRegressionModel, RandomForestRegressor, TrainingSummary, TreeEnsembleModel,
};
use spark_connect::{DataFrame, SparkSession};
use spark_connect_core::error::Result as CoreResult;
use spark_connect_proto as proto;

const MODELS: &str = include_str!("../../../tests/golden/regression_models.json");
const SUMMARIES: &str = include_str!("../../../tests/golden/regression_summaries.json");
const PERSISTENCE: &str = include_str!("../../../tests/golden/regression_persistence.json");

/// The path that the persistence entries save to and load from: the golden normalizes it.
const PATH: &str = "/unused/path";

fn golden(text: &str) -> Value {
    fixture(text)
}

/// Assert that `actual` is the golden `expected`: numbers to 1e-9 relative, the rest exactly. The
/// golden renders NaN and the infinities as strings, as `ml_golden::num` does.
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

/// A member's value or its error, as the typed accessor returns it, rendered as the golden.
type Member = CoreResult<Value>;

fn plain<T: Into<Value>>(value: CoreResult<T>) -> Member {
    value.map(Into::into)
}

fn double(value: CoreResult<f64>) -> Member {
    value.map(num)
}

fn doubles(value: CoreResult<Vec<f64>>) -> Member {
    value.map(|values| nums(&values))
}

fn vector_of(value: CoreResult<MlVector>) -> Member {
    value.map(|v| vector(&v))
}

fn matrix_of(value: CoreResult<MlMatrix>) -> Member {
    value.map(|m| matrix(&m))
}

/// The feature vector of a golden member's name of a model of three features: `dense` is
/// `Vectors.dense([2.5, 1.0, 1.5])`, and `sparse` is `Vectors.sparse(3, [0, 2], [2.5, 1.5])`.
fn features3(key: &str) -> MlVector {
    match key.rsplit('-').next().unwrap() {
        "dense" => MlVector::Dense(vec![2.5, 1.0, 1.5]),
        "sparse" => MlVector::Sparse {
            size: 3,
            indices: vec![0, 2],
            values: vec![2.5, 1.5],
        },
        other => panic!("unknown feature vector {other}"),
    }
}

/// The feature vector of a model of two features (AFT survival): `Vectors.dense([1.0, 0.5])`.
fn features2(key: &str) -> MlVector {
    match key.rsplit('-').next().unwrap() {
        "dense" => MlVector::Dense(vec![1.0, 0.5]),
        other => panic!("unknown feature vector {other}"),
    }
}

/// The value that a golden `predict-<value>` member of an isotonic model predicts.
fn feature_value(key: &str) -> f64 {
    key.rsplit('-').next().unwrap().parse().unwrap()
}

/// The class attributes of PySpark's models (`supportedImpurities` and the like): constants of the
/// Python class, with no command and no accessor in the Rust client.
const CLASS_ATTRIBUTES: &[&str] = &[
    "supportedFeatureSubsetStrategies",
    "supportedImpurities",
    "supportedLossTypes",
];

/// Whether a golden member is a DataFrame, which the replay does not serve (see the module doc).
fn is_frame(entry: &Value) -> bool {
    entry["value"]["schema"].is_string()
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

/// Check the golden members of an object (a model, a summary or a loaded copy) that `dispatch`
/// reads. The members are skipped that the replay does not serve: the Python `repr` and `str`,
/// the class attributes, the DataFrames, and the members of `skip` that the caller checks itself.
/// Returns the number of members checked.
fn check_members(
    server: &ReplayServer,
    members: &Value,
    dispatch: &dyn Fn(&str) -> Option<Member>,
    skip: &[&str],
    label: &str,
) -> usize {
    let mut checked = 0;
    for (key, entry) in members.as_object().unwrap() {
        if key.starts_with("repr")
            || key.starts_with("str")
            || skip.contains(&key.as_str())
            || CLASS_ATTRIBUTES.contains(&key.as_str())
            || is_frame(entry)
        {
            continue;
        }
        let Some(member) = dispatch(key) else {
            panic!("{label}: no accessor for {key}");
        };
        check_member(server, entry, member, &format!("{label}.{key}"));
        checked += 1;
    }
    checked
}

/// The params that a golden fit sent: the fit's params, which the estimator of the replay takes.
fn fitted_params(fit: &Value) -> Params {
    match golden_commands(&fit["fit"])[0].command.clone() {
        Some(proto::ml_command::Command::Fit(sent)) => Params::from_proto(&sent.params.unwrap()),
        other => panic!("the golden fit is not a Fit: {other:?}"),
    }
}

/// The uid that a golden load read: the golden normalizes it to a placeholder, so the loaded uid
/// is the placeholder too.
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

/// The params that a golden save of an operator sent: the `MlParams` of its `Write`, which are the
/// params that PySpark read with the load, as it sends them again (see `read` of the ML object
/// module), so that the loaded params of the Rust client are compared with them.
fn saved_params(save: &Value) -> proto::MlParams {
    match golden_commands(save)[0].command.clone() {
        Some(proto::ml_command::Command::Write(write)) => write.params.unwrap_or_default(),
        other => panic!("the golden save is not a Write: {other:?}"),
    }
}

/// The estimator of a golden persistence entry, after its load: the load's commands, uid and
/// params, then the save's commands.
fn check_loaded_estimator<E: Estimator>(
    server: &ReplayServer,
    session: &SparkSession,
    entry: &Value,
    estimator: &E,
    label: &str,
) {
    assert_commands(server, &entry["load"], label);
    assert_eq!(
        estimator.operator().uid,
        result_uid(&entry["load"]),
        "{label}: uid"
    );
    assert_eq!(
        Estimator::params(estimator).to_proto(),
        saved_params(&entry["save"]),
        "{label}: params"
    );
    estimator
        .write()
        .session(session)
        .overwrite()
        .save(PATH)
        .unwrap();
    assert_commands(server, &entry["save"], label);
}

/// The model of a golden persistence entry, after its load: the load's commands and uid, its
/// members (the predictions of the copy), then the save's commands.
fn check_loaded_model<M: Model>(
    server: &ReplayServer,
    entry: &Value,
    model: &M,
    dispatch: &dyn Fn(&str) -> Option<Member>,
    label: &str,
) {
    assert_commands(server, &entry["load"], label);
    assert_eq!(
        model.operator().uid,
        result_uid(&entry["load"]),
        "{label}: uid"
    );
    check_members(server, &entry["members"]["members"], dispatch, &[], label);
    model.write().unwrap().overwrite().save(PATH).unwrap();
    assert_commands(server, &entry["save"], label);
}

// The members of the models. Each returns None for a member it has no accessor for.

fn linear_model_member(model: &LinearRegressionModel, key: &str) -> Option<Member> {
    if key.starts_with("predict-") {
        return Some(double(model.predict(&features3(key))));
    }
    Some(match key {
        "coefficients" => vector_of(model.coefficients()),
        "intercept" => double(model.intercept()),
        "numFeatures" => plain(model.num_features()),
        "scale" => double(model.scale()),
        "hasSummary" => Ok(json!(model.has_summary())),
        _ => return None,
    })
}

fn glr_model_member(model: &GeneralizedLinearRegressionModel, key: &str) -> Option<Member> {
    if key.starts_with("predict-") {
        return Some(double(model.predict(&features3(key))));
    }
    Some(match key {
        "coefficients" => vector_of(model.coefficients()),
        "intercept" => double(model.intercept()),
        "numFeatures" => plain(model.num_features()),
        "hasSummary" => Ok(json!(model.has_summary())),
        _ => return None,
    })
}

fn aft_model_member(model: &AFTSurvivalRegressionModel, key: &str) -> Option<Member> {
    if key.starts_with("predictQuantiles-") {
        return Some(vector_of(model.predict_quantiles(&features2(key))));
    }
    if key.starts_with("predict-") {
        return Some(double(model.predict(&features2(key))));
    }
    Some(match key {
        "coefficients" => vector_of(model.coefficients()),
        "intercept" => double(model.intercept()),
        "numFeatures" => plain(model.num_features()),
        "scale" => double(model.scale()),
        _ => return None,
    })
}

fn isotonic_model_member(model: &IsotonicRegressionModel, key: &str) -> Option<Member> {
    if key.starts_with("predict-") {
        return Some(double(model.predict(feature_value(key))));
    }
    Some(match key {
        "boundaries" => vector_of(model.boundaries()),
        "numFeatures" => plain(model.num_features()),
        "predictions" => vector_of(model.predictions()),
        _ => return None,
    })
}

fn fm_model_member(model: &FMRegressionModel, key: &str) -> Option<Member> {
    if key.starts_with("predict-") {
        return Some(double(model.predict(&features3(key))));
    }
    Some(match key {
        "factors" => matrix_of(model.factors()),
        "intercept" => double(model.intercept()),
        "linear" => vector_of(model.linear()),
        "numFeatures" => plain(model.num_features()),
        _ => return None,
    })
}

/// A decision tree's members: its own, and those of its prediction.
fn tree_member(model: &DecisionTreeRegressionModel, key: &str) -> Option<Member> {
    if key.starts_with("predictLeaf-") {
        return Some(double(model.predict_leaf(&features3(key))));
    }
    if key.starts_with("predict-") {
        return Some(double(model.predict(&features3(key))));
    }
    Some(match key {
        "depth" => plain(model.depth()),
        "featureImportances" => vector_of(model.feature_importances()),
        "numFeatures" => plain(model.num_features()),
        "numNodes" => plain(model.num_nodes()),
        "toDebugString" => plain(model.to_debug_string()),
        _ => return None,
    })
}

/// An ensemble's members: its own, and those of its prediction. Its `trees` are checked by the
/// caller, see the module doc.
fn ensemble_member<M: TreeEnsembleModel + PredictionModel>(model: &M, key: &str) -> Option<Member> {
    if key.starts_with("predictLeaf-") {
        return Some(vector_of(model.predict_leaf(&features3(key))));
    }
    if key.starts_with("predict-") {
        return Some(double(model.predict(&features3(key))));
    }
    Some(match key {
        "featureImportances" => vector_of(model.feature_importances()),
        "getNumTrees" => plain(model.get_num_trees()),
        "numFeatures" => plain(model.num_features()),
        "toDebugString" => plain(model.to_debug_string()),
        "totalNumNodes" => plain(model.total_num_nodes()),
        "treeWeights" => doubles(model.tree_weights()),
        _ => return None,
    })
}

fn gbt_model_member(model: &GBTRegressionModel, key: &str, dataset: &DataFrame) -> Option<Member> {
    if let Some(loss) = key.strip_prefix("evaluateEachIteration-") {
        return Some(doubles(model.evaluate_each_iteration(dataset, loss)));
    }
    ensemble_member(model, key)
}

// The members of the summaries.

/// The members of a linear regression summary, of an evaluation and of the training.
fn linear_summary_member<S: LinearRegressionSummaryApi>(summary: &S, key: &str) -> Option<Member> {
    Some(match key {
        "coefficientStandardErrors" => doubles(summary.coefficient_standard_errors()),
        "degreesOfFreedom" => plain(summary.degrees_of_freedom()),
        "devianceResiduals" => doubles(summary.deviance_residuals()),
        "explainedVariance" => double(summary.explained_variance()),
        "featuresCol" => plain(summary.features_col()),
        "labelCol" => plain(summary.label_col()),
        "meanAbsoluteError" => double(summary.mean_absolute_error()),
        "meanSquaredError" => double(summary.mean_squared_error()),
        "numInstances" => plain(summary.num_instances()),
        "pValues" => doubles(summary.p_values()),
        "predictionCol" => plain(summary.prediction_col()),
        "r2" => double(summary.r2()),
        "r2adj" => double(summary.r2adj()),
        "rootMeanSquaredError" => double(summary.root_mean_squared_error()),
        "tValues" => doubles(summary.t_values()),
        _ => return None,
    })
}

/// The members of a linear regression training summary: those of the summary, and the fit's.
fn linear_training_member(summary: &LinearRegressionTrainingSummary, key: &str) -> Option<Member> {
    linear_summary_member(summary, key).or_else(|| {
        Some(match key {
            "objectiveHistory" => doubles(summary.objective_history()),
            "totalIterations" => plain(summary.total_iterations()),
            _ => return None,
        })
    })
}

/// The members of a generalized linear regression summary, of an evaluation and of the training.
fn glr_summary_member<S: GeneralizedLinearRegressionSummaryApi>(
    summary: &S,
    key: &str,
) -> Option<Member> {
    Some(match key {
        "aic" => double(summary.aic()),
        "degreesOfFreedom" => plain(summary.degrees_of_freedom()),
        "deviance" => double(summary.deviance()),
        "dispersion" => double(summary.dispersion()),
        "nullDeviance" => double(summary.null_deviance()),
        "numInstances" => plain(summary.num_instances()),
        "predictionCol" => plain(summary.prediction_col()),
        "rank" => plain(summary.rank()),
        "residualDegreeOfFreedom" => plain(summary.residual_degree_of_freedom()),
        "residualDegreeOfFreedomNull" => plain(summary.residual_degree_of_freedom_null()),
        _ => return None,
    })
}

/// The members of a generalized linear regression training summary: those of the summary, and
/// the fit's.
fn glr_training_member(
    summary: &GeneralizedLinearRegressionTrainingSummary,
    key: &str,
) -> Option<Member> {
    glr_summary_member(summary, key).or_else(|| {
        Some(match key {
            "coefficientStandardErrors" => doubles(summary.coefficient_standard_errors()),
            "numIterations" => plain(summary.num_iterations()),
            "pValues" => doubles(summary.p_values()),
            "solver" => plain(summary.solver()),
            "tValues" => doubles(summary.t_values()),
            _ => return None,
        })
    })
}

/// The `LinearRegression` family's model and its summaries, checked on a fitted model: the
/// summary of the fit, and the summary of `evaluate` on the held-out rows.
fn check_linear_model(
    server: &ReplayServer,
    session: &SparkSession,
    model: &LinearRegressionModel,
    entry: &Value,
    summaries: &Value,
    family: &str,
) {
    let members = &entry["members"];
    check_members(
        server,
        members,
        &|key| linear_model_member(model, key),
        &["summary", "evaluate"],
        family,
    );
    let summary = model.summary().unwrap();
    assert_commands(server, &members["summary"], family);
    assert_eq!(
        class_of(&members["summary"]["value"]),
        "LinearRegressionTrainingSummary"
    );
    let training = &summaries[format!("{family}.training").as_str()]["members"];
    check_members(
        server,
        training,
        &|key| linear_training_member(&summary, key),
        &[],
        &format!("{family}.training"),
    );
    let evaluated = model.evaluate(&session.range(1).unwrap()).unwrap();
    assert_commands(server, &members["evaluate"], family);
    assert_eq!(
        class_of(&members["evaluate"]["value"]),
        "LinearRegressionSummary"
    );
    let evaluate = &summaries[format!("{family}.evaluate").as_str()]["members"];
    check_members(
        server,
        evaluate,
        &|key| linear_summary_member(&evaluated, key),
        &[],
        &format!("{family}.evaluate"),
    );
}

/// The `GeneralizedLinearRegression` family's model and its summaries.
fn check_glr_model(
    server: &ReplayServer,
    session: &SparkSession,
    model: &GeneralizedLinearRegressionModel,
    entry: &Value,
    summaries: &Value,
    family: &str,
) {
    let members = &entry["members"];
    check_members(
        server,
        members,
        &|key| glr_model_member(model, key),
        &["summary", "evaluate"],
        family,
    );
    let summary = model.summary().unwrap();
    assert_commands(server, &members["summary"], family);
    assert_eq!(
        class_of(&members["summary"]["value"]),
        "GeneralizedLinearRegressionTrainingSummary"
    );
    let training = &summaries[format!("{family}.training").as_str()]["members"];
    check_members(
        server,
        training,
        &|key| glr_training_member(&summary, key),
        &[],
        &format!("{family}.training"),
    );
    let evaluated = model.evaluate(&session.range(1).unwrap()).unwrap();
    assert_commands(server, &members["evaluate"], family);
    assert_eq!(
        class_of(&members["evaluate"]["value"]),
        "GeneralizedLinearRegressionSummary"
    );
    let evaluate = &summaries[format!("{family}.evaluate").as_str()]["members"];
    check_members(
        server,
        evaluate,
        &|key| glr_summary_member(&evaluated, key),
        &[],
        &format!("{family}.evaluate"),
    );
}

/// The golden entries of a family: its fit, its model, and its summaries when it has them.
fn family_entries<'a>(models: &'a Value, summaries: &'a Value, family: &str) -> Vec<&'a Value> {
    let mut entries = vec![&models["fits"][family]["fit"], &models["models"][family]];
    for kind in ["training", "evaluate"] {
        if let Some(summary) = summaries["summaries"].get(format!("{family}.{kind}").as_str()) {
            entries.push(summary);
        }
    }
    entries
}

/// Fit a LinearRegression of the golden's family, with the params that its golden fit sent, and
/// check the fit, the model, its summaries and its evaluation.
fn check_linear_family(family: &str) {
    let models = golden(MODELS);
    let summaries = golden(SUMMARIES);
    let fit = &models["fits"][family];
    let (server, session) = ReplayServer::start(&family_entries(&models, &summaries, family));
    let mut estimator = LinearRegression::new();
    *estimator.params_mut() = fitted_params(fit);
    let model = estimator.fit(&session.range(1).unwrap()).unwrap();
    assert_commands(&server, &fit["fit"], family);
    check_linear_model(
        &server,
        &session,
        &model,
        &models["models"][family],
        &summaries["summaries"],
        family,
    );
}

/// Fit a GeneralizedLinearRegression of the golden's family, and check it as `check_linear_family`.
fn check_glr_family(family: &str) {
    let models = golden(MODELS);
    let summaries = golden(SUMMARIES);
    let fit = &models["fits"][family];
    let (server, session) = ReplayServer::start(&family_entries(&models, &summaries, family));
    let mut estimator = GeneralizedLinearRegression::new();
    *estimator.params_mut() = fitted_params(fit);
    let model = estimator.fit(&session.range(1).unwrap()).unwrap();
    assert_commands(&server, &fit["fit"], family);
    check_glr_model(
        &server,
        &session,
        &model,
        &models["models"][family],
        &summaries["summaries"],
        family,
    );
}

#[test]
fn linear_regression_normal_matches_official_pyspark() {
    check_linear_family("LinearRegression.normal");
}

#[test]
fn linear_regression_l_bfgs_matches_official_pyspark() {
    check_linear_family("LinearRegression.l-bfgs");
}

#[test]
fn linear_regression_huber_matches_official_pyspark() {
    check_linear_family("LinearRegression.huber");
}

#[test]
fn generalized_linear_regression_gaussian_matches_official_pyspark() {
    check_glr_family("GeneralizedLinearRegression.gaussian");
}

#[test]
fn generalized_linear_regression_poisson_matches_official_pyspark() {
    check_glr_family("GeneralizedLinearRegression.poisson");
}

#[test]
fn generalized_linear_regression_gamma_matches_official_pyspark() {
    check_glr_family("GeneralizedLinearRegression.gamma");
}

#[test]
fn generalized_linear_regression_tweedie_matches_official_pyspark() {
    check_glr_family("GeneralizedLinearRegression.tweedie");
}

#[test]
fn generalized_linear_regression_no_intercept_matches_official_pyspark() {
    check_glr_family("GeneralizedLinearRegression.no-intercept");
}

#[test]
fn aft_survival_regression_matches_official_pyspark() {
    let models = golden(MODELS);
    let family = "AFTSurvivalRegression.default";
    let fit = &models["fits"][family];
    let entry = &models["models"][family];
    let (server, session) = ReplayServer::start(&[&fit["fit"], entry]);
    let mut estimator = AFTSurvivalRegression::new();
    *estimator.params_mut() = fitted_params(fit);
    let model = estimator.fit(&session.range(1).unwrap()).unwrap();
    assert_commands(&server, &fit["fit"], family);
    check_members(
        &server,
        &entry["members"],
        &|key| aft_model_member(&model, key),
        &[],
        family,
    );
}

#[test]
fn isotonic_regression_matches_official_pyspark() {
    let models = golden(MODELS);
    for family in [
        "IsotonicRegression.weighted",
        "IsotonicRegression.antitonic",
    ] {
        let fit = &models["fits"][family];
        let entry = &models["models"][family];
        let (server, session) = ReplayServer::start(&[&fit["fit"], entry]);
        let mut estimator = IsotonicRegression::new();
        *estimator.params_mut() = fitted_params(fit);
        let model = estimator.fit(&session.range(1).unwrap()).unwrap();
        assert_commands(&server, &fit["fit"], family);
        check_members(
            &server,
            &entry["members"],
            &|key| isotonic_model_member(&model, key),
            &[],
            family,
        );
    }
}

#[test]
fn fm_regressor_matches_official_pyspark() {
    let models = golden(MODELS);
    let family = "FMRegressor.default";
    let fit = &models["fits"][family];
    let entry = &models["models"][family];
    let (server, session) = ReplayServer::start(&[&fit["fit"], entry]);
    let mut estimator = FMRegressor::new();
    *estimator.params_mut() = fitted_params(fit);
    let model = estimator.fit(&session.range(1).unwrap()).unwrap();
    assert_commands(&server, &fit["fit"], family);
    check_members(
        &server,
        &entry["members"],
        &|key| fm_model_member(&model, key),
        &[],
        family,
    );
}

#[test]
fn decision_tree_regressor_matches_official_pyspark() {
    let models = golden(MODELS);
    let family = "DecisionTreeRegressor.default";
    let fit = &models["fits"][family];
    let entry = &models["models"][family];
    let (server, session) = ReplayServer::start(&[&fit["fit"], entry]);
    let mut estimator = DecisionTreeRegressor::new();
    *estimator.params_mut() = fitted_params(fit);
    let model = estimator.fit(&session.range(1).unwrap()).unwrap();
    assert_commands(&server, &fit["fit"], family);
    check_members(
        &server,
        &entry["members"],
        &|key| tree_member(&model, key),
        &[],
        family,
    );
}

#[test]
fn random_forest_regressor_matches_official_pyspark() {
    let models = golden(MODELS);
    let family = "RandomForestRegressor.default";
    let tree_key = "RandomForestRegressor.default.trees[0]";
    let fit = &models["fits"][family];
    let entry = &models["models"][family];
    let (server, session) = ReplayServer::start(&[&fit["fit"], entry, &models["models"][tree_key]]);
    let mut estimator = RandomForestRegressor::new();
    *estimator.params_mut() = fitted_params(fit);
    let model = estimator.fit(&session.range(1).unwrap()).unwrap();
    // PySpark's fit fetches the trees right after the Fit: the golden fit has that fetch too.
    let trees = model.trees().unwrap();
    assert_commands(&server, &fit["fit"], family);
    check_members(
        &server,
        &entry["members"],
        &|key| ensemble_member(&model, key),
        &["trees"],
        family,
    );
    check_members(
        &server,
        &models["models"][tree_key]["members"],
        &|key| tree_member(&trees[0], key),
        &[],
        tree_key,
    );
}

#[test]
fn gradient_boosted_tree_regressor_matches_official_pyspark() {
    let models = golden(MODELS);
    let family = "GBTRegressor.default";
    let tree_key = "GBTRegressor.default.trees[0]";
    let fit = &models["fits"][family];
    let entry = &models["models"][family];
    let (server, session) = ReplayServer::start(&[&fit["fit"], entry, &models["models"][tree_key]]);
    let mut estimator = GBTRegressor::new();
    *estimator.params_mut() = fitted_params(fit);
    let model = estimator.fit(&session.range(1).unwrap()).unwrap();
    // PySpark's fit fetches the trees right after the Fit: the golden fit has that fetch too.
    let trees = model.trees().unwrap();
    assert_commands(&server, &fit["fit"], family);
    let dataset = session.range(1).unwrap();
    check_members(
        &server,
        &entry["members"],
        &|key| gbt_model_member(&model, key, &dataset),
        &["trees"],
        family,
    );
    check_members(
        &server,
        &models["models"][tree_key]["members"],
        &|key| tree_member(&trees[0], key),
        &[],
        tree_key,
    );
}

/// The message of a panic, as the test reports it.
fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    match panic.downcast::<String>() {
        Ok(text) => *text,
        Err(panic) => panic
            .downcast::<&str>()
            .map_or_else(|_| "a panic".to_string(), |text| text.to_string()),
    }
}

/// The persistence of each family: its estimator and its model are saved and loaded back, and
/// the loaded model's predictions and commands are those of PySpark's persistence entries. Every
/// family runs even when an earlier one fails, and the failures are reported together.
#[test]
fn regression_persistence_matches_official_pyspark() {
    let persistence = golden(PERSISTENCE)["persistence"].clone();
    let estimator = |family: &str| persistence[format!("{family}.estimator").as_str()].clone();
    let model = |family: &str| persistence[format!("{family}.model").as_str()].clone();
    let mut failures = Vec::new();
    let mut check = |family: &str, run: &dyn Fn()| {
        if let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
            failures.push(format!("{family}: {}", panic_message(panic)));
        }
    };

    for family in [
        "LinearRegression.normal",
        "LinearRegression.l-bfgs",
        "LinearRegression.huber",
    ] {
        check(family, &|| {
            let (estimator_entry, model_entry) = (estimator(family), model(family));
            let (server, session) = ReplayServer::start(&[&estimator_entry, &model_entry]);
            let loaded = LinearRegression::load(&session, PATH).unwrap();
            check_loaded_estimator(&server, &session, &estimator_entry, &loaded, family);
            let loaded = LinearRegressionModel::load(&session, PATH).unwrap();
            check_loaded_model(
                &server,
                &model_entry,
                &loaded,
                &|key| linear_model_member(&loaded, key),
                family,
            );
        });
    }

    for family in [
        "GeneralizedLinearRegression.gaussian",
        "GeneralizedLinearRegression.poisson",
        "GeneralizedLinearRegression.gamma",
        "GeneralizedLinearRegression.tweedie",
        "GeneralizedLinearRegression.no-intercept",
    ] {
        check(family, &|| {
            let (estimator_entry, model_entry) = (estimator(family), model(family));
            let (server, session) = ReplayServer::start(&[&estimator_entry, &model_entry]);
            let loaded = GeneralizedLinearRegression::load(&session, PATH).unwrap();
            check_loaded_estimator(&server, &session, &estimator_entry, &loaded, family);
            let loaded = GeneralizedLinearRegressionModel::load(&session, PATH).unwrap();
            check_loaded_model(
                &server,
                &model_entry,
                &loaded,
                &|key| glr_model_member(&loaded, key),
                family,
            );
        });
    }

    let family = "AFTSurvivalRegression.default";
    check(family, &|| {
        let (estimator_entry, model_entry) = (estimator(family), model(family));
        let (server, session) = ReplayServer::start(&[&estimator_entry, &model_entry]);
        let loaded = AFTSurvivalRegression::load(&session, PATH).unwrap();
        check_loaded_estimator(&server, &session, &estimator_entry, &loaded, family);
        let loaded = AFTSurvivalRegressionModel::load(&session, PATH).unwrap();
        check_loaded_model(
            &server,
            &model_entry,
            &loaded,
            &|key| aft_model_member(&loaded, key),
            family,
        );
    });

    for family in [
        "IsotonicRegression.weighted",
        "IsotonicRegression.antitonic",
    ] {
        check(family, &|| {
            let (estimator_entry, model_entry) = (estimator(family), model(family));
            let (server, session) = ReplayServer::start(&[&estimator_entry, &model_entry]);
            let loaded = IsotonicRegression::load(&session, PATH).unwrap();
            check_loaded_estimator(&server, &session, &estimator_entry, &loaded, family);
            let loaded = IsotonicRegressionModel::load(&session, PATH).unwrap();
            check_loaded_model(
                &server,
                &model_entry,
                &loaded,
                &|key| isotonic_model_member(&loaded, key),
                family,
            );
        });
    }

    let family = "FMRegressor.default";
    check(family, &|| {
        let (estimator_entry, model_entry) = (estimator(family), model(family));
        let (server, session) = ReplayServer::start(&[&estimator_entry, &model_entry]);
        let loaded = FMRegressor::load(&session, PATH).unwrap();
        check_loaded_estimator(&server, &session, &estimator_entry, &loaded, family);
        let loaded = FMRegressionModel::load(&session, PATH).unwrap();
        check_loaded_model(
            &server,
            &model_entry,
            &loaded,
            &|key| fm_model_member(&loaded, key),
            family,
        );
    });

    let family = "DecisionTreeRegressor.default";
    check(family, &|| {
        let (estimator_entry, model_entry) = (estimator(family), model(family));
        let (server, session) = ReplayServer::start(&[&estimator_entry, &model_entry]);
        let loaded = DecisionTreeRegressor::load(&session, PATH).unwrap();
        check_loaded_estimator(&server, &session, &estimator_entry, &loaded, family);
        let loaded = DecisionTreeRegressionModel::load(&session, PATH).unwrap();
        check_loaded_model(
            &server,
            &model_entry,
            &loaded,
            &|key| tree_member(&loaded, key),
            family,
        );
    });

    // The loaded ensembles fetch their trees, as PySpark's load does (see the module doc).
    let family = "RandomForestRegressor.default";
    check(family, &|| {
        let (estimator_entry, model_entry) = (estimator(family), model(family));
        let (server, session) = ReplayServer::start(&[&estimator_entry, &model_entry]);
        let loaded = RandomForestRegressor::load(&session, PATH).unwrap();
        check_loaded_estimator(&server, &session, &estimator_entry, &loaded, family);
        let loaded = RandomForestRegressionModel::load(&session, PATH).unwrap();
        loaded.trees().unwrap();
        check_loaded_model(
            &server,
            &model_entry,
            &loaded,
            &|key| ensemble_member(&loaded, key),
            family,
        );
    });

    let family = "GBTRegressor.default";
    check(family, &|| {
        let (estimator_entry, model_entry) = (estimator(family), model(family));
        let (server, session) = ReplayServer::start(&[&estimator_entry, &model_entry]);
        let loaded = GBTRegressor::load(&session, PATH).unwrap();
        check_loaded_estimator(&server, &session, &estimator_entry, &loaded, family);
        let loaded = GBTRegressionModel::load(&session, PATH).unwrap();
        loaded.trees().unwrap();
        let dataset = session.range(1).unwrap();
        check_loaded_model(
            &server,
            &model_entry,
            &loaded,
            &|key| gbt_model_member(&loaded, key, &dataset),
            family,
        );
    });

    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn a_request_that_differs_from_its_golden_fails_the_check() {
    let models = golden(MODELS);
    let coefficients = &models["models"]["LinearRegression.normal"]["members"]["coefficients"];
    let (server, session) = ReplayServer::start(&[coefficients]);

    // The golden has the fetch of `coefficients`: a fetch of `intercept` is refused by the replay,
    // and the commands that were sent do not equal the golden's.
    let model = MlTarget::unowned(&session, "<id0>");
    assert!(fetch(&model, "intercept", vec![]).is_err());
    let check = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_commands(&server, coefficients, "coefficients")
    }));
    assert!(check.is_err(), "the differing request passed the check");
}
