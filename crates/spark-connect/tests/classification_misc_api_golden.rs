// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, summaries and persistence of NaiveBayesModel,
//! MultilayerPerceptronClassificationModel and FMClassificationModel, replayed against the golden
//! of official PySpark 4.3.0 with no server.
//!
//! The goldens are replayed by the replay server of `common/ml_golden.rs`, which answers each ML
//! command with the golden result of the same command. Each call's commands are checked against the
//! golden's commands, normalized as the golden is, and each value that a call returns is compared
//! with the golden's value; so every request that a call sends must be one of the golden's.

#[path = "common/ml_golden.rs"]
mod ml_golden;

use std::collections::BTreeSet;

use ml_golden::{assert_commands, ReplayServer};
use serde_json::{json, Value};
use spark_connect::ml::{
    BinaryClassificationSummary, ClassificationSummary, FMClassificationModel, FMClassifier,
    MlMatrix, MlSummary, MlTarget, MlVector, Model, MultilayerPerceptronClassificationModel,
    MultilayerPerceptronClassifier, NaiveBayes, NaiveBayesModel, PredictionModel,
    ProbabilisticClassificationModel, TrainingSummary,
};
use spark_connect::SparkSession;
use spark_connect_core::error::Result;

const MODELS: &str = include_str!("../../../tests/golden/classification_models.json");
const SUMMARIES: &str = include_str!("../../../tests/golden/classification_summaries.json");
const PERSISTENCE: &str = include_str!("../../../tests/golden/classification_persistence.json");

/// A JSON form of a value, as the golden writes it: floats are numbers, NaN and infinities are
/// strings, and vectors and matrices are objects of their type, shape and values.
trait Golden {
    fn golden(&self) -> Value;
}

fn number(x: f64) -> Value {
    if x.is_nan() {
        json!("NaN")
    } else if x.is_infinite() {
        json!(if x > 0.0 { "Infinity" } else { "-Infinity" })
    } else {
        json!(x)
    }
}

impl Golden for f64 {
    fn golden(&self) -> Value {
        number(*self)
    }
}

impl Golden for i32 {
    fn golden(&self) -> Value {
        json!(self)
    }
}

impl Golden for bool {
    fn golden(&self) -> Value {
        json!(self)
    }
}

impl Golden for String {
    fn golden(&self) -> Value {
        json!(self)
    }
}

impl Golden for Vec<f64> {
    fn golden(&self) -> Value {
        Value::Array(self.iter().map(|x| number(*x)).collect())
    }
}

impl Golden for MlVector {
    fn golden(&self) -> Value {
        match self {
            MlVector::Dense(values) => json!({
                "type": "dense",
                "size": values.len(),
                "values": values.iter().map(|x| number(*x)).collect::<Vec<_>>(),
            }),
            MlVector::Sparse {
                size,
                indices,
                values,
            } => json!({
                "type": "sparse",
                "size": size,
                "indices": indices,
                "values": values.iter().map(|x| number(*x)).collect::<Vec<_>>(),
            }),
        }
    }
}

impl Golden for MlMatrix {
    fn golden(&self) -> Value {
        match self {
            MlMatrix::Dense {
                num_rows,
                num_cols,
                values,
                is_transposed,
            } => json!({
                "type": "dense",
                "numRows": num_rows,
                "numCols": num_cols,
                "isTransposed": is_transposed,
                "values": values.iter().map(|x| number(*x)).collect::<Vec<_>>(),
            }),
            MlMatrix::Sparse {
                num_rows,
                num_cols,
                col_ptrs,
                row_indices,
                values,
                is_transposed,
            } => json!({
                "type": "sparse",
                "numRows": num_rows,
                "numCols": num_cols,
                "isTransposed": is_transposed,
                "colPtrs": col_ptrs,
                "rowIndices": row_indices,
                "values": values.iter().map(|x| number(*x)).collect::<Vec<_>>(),
            }),
        }
    }
}

impl Golden for Value {
    fn golden(&self) -> Value {
        self.clone()
    }
}

/// A summary or model object, as the golden writes an object: its class and its reference.
fn reference(class: &str, target: &MlTarget) -> Value {
    json!({"type": class, "ref": target.id()})
}

/// Asserts that `actual` is the golden `expected`. The golden is JSON, whose decimals are read
/// within an ulp by serde_json, so numbers compare within 1e-12 relative; everything else exactly.
fn assert_golden(actual: &Value, expected: &Value, label: &str) {
    fn same(actual: &Value, expected: &Value) -> bool {
        match (actual, expected) {
            (Value::Number(a), Value::Number(b)) => match (a.as_f64(), b.as_f64()) {
                (Some(a), Some(b)) => a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs()),
                _ => a == b,
            },
            (Value::Array(a), Value::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same(a, b))
            }
            (Value::Object(a), Value::Object(b)) => {
                a.len() == b.len() && a.iter().all(|(k, a)| b.get(k).is_some_and(|b| same(a, b)))
            }
            _ => actual == expected,
        }
    }
    assert!(
        same(actual, expected),
        "{label}: value {actual} is not the golden {expected}"
    );
}

/// Checks the members of one golden entry: each call's commands and value against the golden's.
/// Each check takes the commands that the server has received since the previous check, so it
/// sees the commands of its own call only.
struct Checker<'a> {
    server: &'a ReplayServer,
    members: Value,
    checked: BTreeSet<String>,
}

impl<'a> Checker<'a> {
    /// A check of `members`. The commands received so far, of the setup of the test's objects, are
    /// discarded.
    fn new(server: &'a ReplayServer, members: &Value) -> Self {
        server.take_requests();
        Self {
            server,
            members: members.clone(),
            checked: BTreeSet::new(),
        }
    }

    fn entry(&self, label: &str) -> Value {
        self.members
            .get(label)
            .unwrap_or_else(|| panic!("the golden has no member {label}"))
            .clone()
    }

    /// The value of `label`: the call sends the golden's commands, and returns the golden's value.
    fn value<T: Golden>(&mut self, label: &str, call: impl FnOnce() -> Result<T>) {
        let entry = self.entry(label);
        let actual = call().unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_commands(self.server, &entry, label);
        assert_golden(&actual.golden(), &entry["value"], label);
        self.checked.insert(label.to_string());
    }

    /// The value of `label` of a call that the client makes without a request.
    fn local<T: Golden>(&mut self, label: &str, call: impl FnOnce() -> T) {
        let entry = self.entry(label);
        let actual = call();
        assert!(self.server.take_requests().is_empty(), "{label}: requests");
        assert_golden(&actual.golden(), &entry["value"], label);
        self.checked.insert(label.to_string());
    }

    /// The error of `label`, whose message is the golden's.
    fn error<T: Golden>(&mut self, label: &str, call: impl FnOnce() -> Result<T>) {
        let entry = self.entry(label);
        let expected = entry["error"]["message"].as_str().unwrap().to_string();
        let error = match call() {
            Ok(_) => panic!("{label}: the call succeeded"),
            Err(error) => error.to_string(),
        };
        self.server.take_requests();
        assert!(
            error.contains(&expected),
            "{label}: {error:?} does not carry {expected:?}"
        );
        self.checked.insert(label.to_string());
    }

    /// Asserts that every golden member of the entry is checked, or is one of `skipped`.
    fn finish(self, skipped: &[&str]) {
        let golden: BTreeSet<String> = self.members.as_object().unwrap().keys().cloned().collect();
        let skipped: BTreeSet<String> = skipped.iter().map(|s| s.to_string()).collect();
        let missing: Vec<&String> = golden
            .iter()
            .filter(|label| !self.checked.contains(*label) && !skipped.contains(*label))
            .collect();
        assert!(
            missing.is_empty(),
            "golden members not checked: {missing:?}"
        );
    }
}

fn dense(values: &[f64]) -> MlVector {
    MlVector::Dense(values.to_vec())
}

/// The vectors that the golden predicts, of the Naive Bayes models and of the others.
fn vectors(naive_bayes: bool) -> [(&'static str, MlVector); 3] {
    let sparse = |value: f64| MlVector::Sparse {
        size: 2,
        indices: vec![1],
        values: vec![value],
    };
    if naive_bayes {
        [
            ("dense", dense(&[1.0, 0.0])),
            ("dense2", dense(&[0.0, 1.0])),
            ("sparse", sparse(1.0)),
        ]
    } else {
        [
            ("dense", dense(&[0.5, 1.5])),
            ("dense2", dense(&[3.0, -0.25])),
            ("sparse", sparse(2.0)),
        ]
    }
}

/// The members that every classification model has (`numClasses`, `numFeatures`, and the
/// prediction of each vector), as the golden captures them.
fn classification_members<M: ProbabilisticClassificationModel>(
    checker: &mut Checker,
    model: &M,
    naive_bayes: bool,
) {
    for (name, vector) in vectors(naive_bayes) {
        checker.value(&format!("predict-{name}"), || model.predict(&vector));
        checker.value(&format!("predictRaw-{name}"), || model.predict_raw(&vector));
        checker.value(&format!("predictProbability-{name}"), || {
            model.predict_probability(&vector)
        });
    }
    checker.value("numClasses", || model.num_classes());
    checker.value("numFeatures", || model.num_features());
}

/// The members of a classification summary (`_ClassificationSummary`), except the DataFrames.
fn classification_summary_members<S: ClassificationSummary>(checker: &mut Checker, s: &S) {
    checker.value("accuracy", || s.accuracy());
    checker.value("fMeasureByLabel-0.5", || s.f_measure_by_label(0.5));
    checker.value("fMeasureByLabel-1.0", || s.f_measure_by_label(1.0));
    checker.value("falsePositiveRateByLabel", || {
        s.false_positive_rate_by_label()
    });
    checker.value("labelCol", || s.label_col());
    checker.value("labels", || s.labels());
    checker.value("precisionByLabel", || s.precision_by_label());
    checker.value("predictionCol", || s.prediction_col());
    checker.value("recallByLabel", || s.recall_by_label());
    checker.value("truePositiveRateByLabel", || {
        s.true_positive_rate_by_label()
    });
    checker.value("weightCol", || s.weight_col());
    checker.value("weightedFMeasure-0.5", || s.weighted_f_measure(0.5));
    checker.value("weightedFMeasure-1.0", || s.weighted_f_measure(1.0));
    checker.value("weightedFalsePositiveRate", || {
        s.weighted_false_positive_rate()
    });
    checker.value("weightedPrecision", || s.weighted_precision());
    checker.value("weightedRecall", || s.weighted_recall());
    checker.value("weightedTruePositiveRate", || {
        s.weighted_true_positive_rate()
    });
}

/// The scalar members of a binary classification summary (`_BinaryClassificationSummary`).
fn binary_summary_members<S: BinaryClassificationSummary>(checker: &mut Checker, s: &S) {
    checker.value("areaUnderROC", || s.area_under_roc());
    checker.value("scoreCol", || s.score_col());
}

/// The members of a training summary (`_TrainingSummary`).
fn training_summary_members<S: TrainingSummary>(checker: &mut Checker, s: &S) {
    checker.value("objectiveHistory", || s.objective_history());
    checker.value("totalIterations", || s.total_iterations());
}

/// The DataFrame members of a summary, which the golden captures in the server's own relations,
/// and the Python-only representations.
const SUMMARY_SKIPPED: [&str; 8] = [
    "predictions",
    "roc",
    "pr",
    "fMeasureByThreshold",
    "precisionByThreshold",
    "recallByThreshold",
    "repr()",
    "str()",
];

fn fit_naive_bayes(session: &SparkSession, estimator: &mut NaiveBayes) -> Result<NaiveBayesModel> {
    estimator.fit(&session.range(3)?)
}

#[test]
fn naive_bayes_members_match_official_pyspark() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    for (key, model_type) in [
        ("NaiveBayes.multinomial", "multinomial"),
        ("NaiveBayes.complement", "complement"),
        ("NaiveBayes.bernoulli", "bernoulli"),
        ("NaiveBayes.gaussian", "gaussian"),
    ] {
        let entry = &models["models"][key];
        let errors = models["models"].get(format!("{key}.errors"));
        let fit = &models["fits"][key]["fit"];
        let mut entries = vec![fit, entry];
        entries.extend(errors);
        let (server, session) = ReplayServer::start(&entries);
        let mut estimator = NaiveBayes::new()
            .set_model_type(model_type)
            .set_weight_col("weight");
        let model = fit_naive_bayes(&session, &mut estimator).unwrap();
        assert_commands(&server, fit, "fit");

        let mut checker = Checker::new(&server, &entry["members"]);
        checker.value("pi", || model.pi());
        checker.value("theta", || model.theta());
        checker.value("sigma", || model.sigma());
        classification_members(&mut checker, &model, true);
        if let Some(errors) = errors {
            // The Bernoulli model takes 0 or 1 features only: the server's error, as the golden.
            let mut failures = Checker::new(&server, &errors["members"]);
            failures.error("predict-outside-0-1", || model.predict(&dense(&[0.5, 1.5])));
            failures.finish(&[]);
        }
        checker.finish(&["repr()", "str()", "transform"]);
    }
}

#[test]
fn naive_bayes_load_and_save_match_official_pyspark() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let persistence: Value = serde_json::from_str(PERSISTENCE).unwrap();
    for (key, model_type) in [
        ("NaiveBayes.multinomial", "multinomial"),
        ("NaiveBayes.complement", "complement"),
        ("NaiveBayes.bernoulli", "bernoulli"),
        ("NaiveBayes.gaussian", "gaussian"),
    ] {
        let saved = &persistence["persistence"][format!("{key}.model")];

        // Load: a read of the saved model, whose members are the loaded model's.
        let (server, session) = ReplayServer::start(&[saved]);
        let loaded = NaiveBayesModel::load(&session, "<path>").unwrap();
        let mut checker = Checker::new(&server, &saved["members"]["members"]);
        checker.value("predict-dense", || loaded.predict(&dense(&[1.0, 0.0])));
        checker.finish(&["repr()", "str()", "transform"]);

        // Save: a write of the fitted model, with the params that its estimator sets.
        let fit = &models["fits"][key]["fit"];
        let (server, session) = ReplayServer::start(&[fit, &models["models"][key], saved]);
        let mut estimator = NaiveBayes::new()
            .set_model_type(model_type)
            .set_weight_col("weight");
        let model = fit_naive_bayes(&session, &mut estimator).unwrap();
        assert_commands(&server, fit, "fit");
        model.write().unwrap().overwrite().save("<path>").unwrap();
        assert_commands(&server, &saved["save"], &format!("{key}: save"));
    }
}

#[test]
fn multilayer_perceptron_members_and_summaries_match_official_pyspark() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let summaries: Value = serde_json::from_str(SUMMARIES).unwrap();
    let key = "MultilayerPerceptronClassifier";
    let entry = &models["models"][key];
    let training = &summaries["summaries"][format!("{key}.training")];
    let evaluate = &summaries["summaries"][format!("{key}.evaluate")];
    let fit = &models["fits"][key]["fit"];
    let (server, session) = ReplayServer::start(&[fit, entry, training, evaluate]);
    let mut estimator = MultilayerPerceptronClassifier::new()
        .set_layers([2, 4, 3])
        .set_max_iter(30)
        .set_seed(42)
        .set_block_size(4)
        .set_step_size(0.1);
    let model = estimator.fit(&session.range(3).unwrap()).unwrap();
    assert_commands(&server, fit, "fit");

    let mut checker = Checker::new(&server, &entry["members"]);
    checker.value("weights", || model.weights());
    checker.local("hasSummary", || model.has_summary());
    classification_members(&mut checker, &model, false);
    checker.value("summary()", || {
        model.summary().map(|s| {
            reference(
                "MultilayerPerceptronClassificationTrainingSummary",
                s.summary_target(),
            )
        })
    });
    checker.value("evaluate", || {
        let df = session.range(3)?;
        model.evaluate(&df).map(|s| {
            reference(
                "MultilayerPerceptronClassificationSummary",
                s.summary_target(),
            )
        })
    });
    checker.finish(&["repr()", "str()", "transform"]);

    let summary = model.summary().unwrap();
    let mut checker = Checker::new(&server, &training["members"]);
    classification_summary_members(&mut checker, &summary);
    training_summary_members(&mut checker, &summary);
    checker.finish(&SUMMARY_SKIPPED);

    let summary = model.evaluate(&session.range(3).unwrap()).unwrap();
    let mut checker = Checker::new(&server, &evaluate["members"]);
    classification_summary_members(&mut checker, &summary);
    checker.finish(&SUMMARY_SKIPPED);
}

#[test]
fn multilayer_perceptron_load_and_save_match_official_pyspark() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let persistence: Value = serde_json::from_str(PERSISTENCE).unwrap();
    let key = "MultilayerPerceptronClassifier";
    let saved = &persistence["persistence"][format!("{key}.model")];

    let (server, session) = ReplayServer::start(&[saved]);
    let loaded = MultilayerPerceptronClassificationModel::load(&session, "<path>").unwrap();
    let mut checker = Checker::new(&server, &saved["members"]["members"]);
    checker.value("predict-dense", || loaded.predict(&dense(&[0.5, 1.5])));
    checker.finish(&["repr()", "str()", "transform"]);
    // A loaded model has no training summary, as PySpark's.
    assert!(!loaded.has_summary());
    assert!(loaded.summary().is_err());

    let fit = &models["fits"][key]["fit"];
    let (server, session) = ReplayServer::start(&[fit, &models["models"][key], saved]);
    let model = MultilayerPerceptronClassifier::new()
        .set_layers([2, 4, 3])
        .set_max_iter(30)
        .set_seed(42)
        .set_block_size(4)
        .set_step_size(0.1)
        .fit(&session.range(3).unwrap())
        .unwrap();
    assert_commands(&server, fit, "fit");
    model.write().unwrap().overwrite().save("<path>").unwrap();
    assert_commands(&server, &saved["save"], &format!("{key}: save"));
}

#[test]
fn fm_members_and_summaries_match_official_pyspark() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let summaries: Value = serde_json::from_str(SUMMARIES).unwrap();
    let key = "FMClassifier";
    let entry = &models["models"][key];
    let training = &summaries["summaries"][format!("{key}.training")];
    let evaluate = &summaries["summaries"][format!("{key}.evaluate")];
    let fit = &models["fits"][key]["fit"];
    let (server, session) = ReplayServer::start(&[fit, entry, training, evaluate]);
    let model: FMClassificationModel = FMClassifier::new()
        .set_factor_size(2)
        .set_max_iter(20)
        .set_seed(42)
        .set_reg_param(0.01)
        .set_step_size(0.1)
        .fit(&session.range(3).unwrap())
        .unwrap();
    assert_commands(&server, fit, "fit");

    let mut checker = Checker::new(&server, &entry["members"]);
    checker.value("factors", || model.factors());
    checker.value("linear", || model.linear());
    checker.value("intercept", || model.intercept());
    checker.local("hasSummary", || model.has_summary());
    classification_members(&mut checker, &model, false);
    checker.value("summary()", || {
        model
            .summary()
            .map(|s| reference("FMClassificationTrainingSummary", s.summary_target()))
    });
    checker.value("evaluate", || {
        let df = session.range(3)?;
        model
            .evaluate(&df)
            .map(|s| reference("FMClassificationSummary", s.summary_target()))
    });
    checker.finish(&["repr()", "str()", "transform"]);

    let summary = model.summary().unwrap();
    let mut checker = Checker::new(&server, &training["members"]);
    classification_summary_members(&mut checker, &summary);
    binary_summary_members(&mut checker, &summary);
    training_summary_members(&mut checker, &summary);
    checker.finish(&SUMMARY_SKIPPED);

    let summary = model.evaluate(&session.range(3).unwrap()).unwrap();
    let mut checker = Checker::new(&server, &evaluate["members"]);
    classification_summary_members(&mut checker, &summary);
    binary_summary_members(&mut checker, &summary);
    checker.finish(&SUMMARY_SKIPPED);
}

#[test]
fn fm_load_and_save_match_official_pyspark() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let persistence: Value = serde_json::from_str(PERSISTENCE).unwrap();
    let key = "FMClassifier";
    let saved = &persistence["persistence"][format!("{key}.model")];

    let (server, session) = ReplayServer::start(&[saved]);
    let loaded = FMClassificationModel::load(&session, "<path>").unwrap();
    let mut checker = Checker::new(&server, &saved["members"]["members"]);
    checker.value("predict-dense", || loaded.predict(&dense(&[0.5, 1.5])));
    checker.finish(&["repr()", "str()", "transform"]);
    assert!(!loaded.has_summary());
    assert!(loaded.summary().is_err());

    let fit = &models["fits"][key]["fit"];
    let (server, session) = ReplayServer::start(&[fit, &models["models"][key], saved]);
    let model = FMClassifier::new()
        .set_factor_size(2)
        .set_max_iter(20)
        .set_seed(42)
        .set_reg_param(0.01)
        .set_step_size(0.1)
        .fit(&session.range(3).unwrap())
        .unwrap();
    assert_commands(&server, fit, "fit");
    model.write().unwrap().overwrite().save("<path>").unwrap();
    assert_commands(&server, &saved["save"], &format!("{key}: save"));
}
