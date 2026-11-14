// SPDX-License-Identifier: Apache-2.0
//! The typed accessors of LogisticRegression, LinearSVC and their models and summaries against a
//! live Spark Connect server, on the datasets that official PySpark 4.3.0 captured: the values
//! are the golden's (`tests/golden/classification_*.json`), to 1e-9 relative. Set SPARK_REMOTE to
//! run against a live Spark Connect server.
mod common;

use common::live_session;
use serde_json::Value;
use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::ml::{
    BinaryClassificationSummary, ClassificationModel, ClassificationSummary, Estimator, LinearSVC,
    LinearSVCModel, LogisticRegression, LogisticRegressionEvaluation, LogisticRegressionModel,
    LogisticRegressionModelSummary, LogisticRegressionSummaryApi, MlMatrix, MlVector, Model,
    Params, PredictionModel, ProbabilisticClassificationModel, TrainingSummary,
};
use spark_connect::{Column, DataFrame, SparkSession};

const MODELS: &str = include_str!("../../../tests/golden/classification_models.json");
const SUMMARIES: &str = include_str!("../../../tests/golden/classification_summaries.json");

fn golden() -> Value {
    serde_json::from_str(MODELS).unwrap()
}

fn summaries() -> Value {
    serde_json::from_str::<Value>(SUMMARIES).unwrap()["summaries"].clone()
}

/// The golden value of the member `member` of the summary `summary`, as `<family>.<kind>`.
fn summary_value(summaries: &Value, summary: &str, member: &str) -> Value {
    summaries[summary]["members"][member]["value"].clone()
}

/// A DataFrame of the golden dataset `name` (label, weight, features), as its rows.
fn dataset(s: &SparkSession, golden: &Value, name: &str) -> DataFrame {
    let rows = golden["datasets"][name]["rows"].as_array().unwrap();
    let values: Vec<String> = rows
        .iter()
        .map(|row| {
            let features: Vec<String> = row[2]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| format!("CAST({:?} AS DOUBLE)", x.as_f64().unwrap()))
                .collect();
            format!(
                "(CAST({:?} AS DOUBLE), CAST({:?} AS DOUBLE), array({}))",
                row[0].as_f64().unwrap(),
                row[1].as_f64().unwrap(),
                features.join(", ")
            )
        })
        .collect();
    s.sql(&format!(
        "SELECT label, weight, xs FROM VALUES {} AS t(label, weight, xs)",
        values.join(", ")
    ))
    .unwrap()
    .with_column(
        "features",
        Column::new(Expression::function(
            "array_to_vector",
            vec![col("xs").expression().clone()],
        )),
    )
}

/// The params that the golden fit of `family` sent, which the test fits with.
fn fit_params(golden: &Value, family: &str) -> Params {
    let bytes = base64_decode(
        golden["fits"][family]["fit"]["commands"][0]
            .as_str()
            .unwrap(),
    );
    let command =
        <spark_connect_proto::MlCommand as prost::Message>::decode(bytes.as_slice()).unwrap();
    match command.command {
        Some(spark_connect_proto::ml_command::Command::Fit(fit)) => {
            Params::from_proto(&fit.params.unwrap())
        }
        other => panic!("not a fit: {other:?}"),
    }
}

fn base64_decode(text: &str) -> Vec<u8> {
    use base64::{engine::general_purpose::STANDARD, Engine};
    STANDARD.decode(text).unwrap()
}

/// Assert that `actual` is the golden `expected`: numbers to 1e-9 relative, the rest exactly.
fn assert_golden(actual: &Value, expected: &Value, path: &str) {
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
                assert_golden(a, e, &format!("{path}[{index}]"));
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            assert_eq!(a.len(), e.len(), "{path}: keys");
            for (key, e) in e {
                assert_golden(&a[key], e, &format!("{path}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

fn vector_json(vector: &MlVector) -> Value {
    match vector {
        MlVector::Dense(values) => serde_json::json!({
            "size": values.len(), "type": "dense", "values": values,
        }),
        other => panic!("golden vectors are dense here: {other:?}"),
    }
}

fn matrix_json(matrix: &MlMatrix) -> Value {
    match matrix {
        MlMatrix::Dense {
            num_rows,
            num_cols,
            values,
            is_transposed,
        } => serde_json::json!({
            "numRows": num_rows, "numCols": num_cols, "values": values,
            "isTransposed": is_transposed, "type": "dense",
        }),
        other => panic!("golden matrices are dense here: {other:?}"),
    }
}

fn feature_vector(values: &[f64]) -> MlVector {
    MlVector::Dense(values.to_vec())
}

/// The rows of a DataFrame of doubles, sorted, as the golden's rows are compared.
fn sorted_rows(df: DataFrame) -> Vec<Vec<f64>> {
    let mut rows: Vec<Vec<f64>> = df
        .collect()
        .unwrap()
        .into_iter()
        .map(|row| {
            (0..row.len())
                .map(|i| row.get(i).unwrap().as_f64().unwrap())
                .collect()
        })
        .collect();
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rows
}

/// Assert that two row sets are the golden's: the same rows in order, to 1e-9 relative.
fn assert_rows_close(actual: Vec<Vec<f64>>, expected: Vec<Vec<f64>>, path: &str) {
    assert_eq!(actual.len(), expected.len(), "{path}: row count");
    for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(a.len(), e.len(), "{path}[{index}]: width");
        for (column, (a, e)) in a.iter().zip(e).enumerate() {
            assert!(
                (a - e).abs() <= 1e-9 * e.abs().max(1.0),
                "{path}[{index}][{column}]: {a} != {e}"
            );
        }
    }
}

fn golden_rows(value: &Value) -> Vec<Vec<f64>> {
    let mut rows: Vec<Vec<f64>> = value["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect();
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rows
}

/// The scalar members of a classification summary that the golden records, against the summary.
fn check_classification(
    summary: &impl ClassificationSummary,
    golden: &Value,
    key: &str,
    path: &str,
) {
    let expected = golden[key]["value"].clone();
    let actual = match key {
        "accuracy" => serde_json::json!(summary.accuracy().unwrap()),
        "falsePositiveRateByLabel" => {
            serde_json::json!(summary.false_positive_rate_by_label().unwrap())
        }
        "fMeasureByLabel-1.0" => serde_json::json!(summary.f_measure_by_label(1.0).unwrap()),
        "labelCol" => serde_json::json!(summary.label_col().unwrap()),
        "labels" => serde_json::json!(summary.labels().unwrap()),
        "precisionByLabel" => serde_json::json!(summary.precision_by_label().unwrap()),
        "predictionCol" => serde_json::json!(summary.prediction_col().unwrap()),
        "recallByLabel" => serde_json::json!(summary.recall_by_label().unwrap()),
        "truePositiveRateByLabel" => {
            serde_json::json!(summary.true_positive_rate_by_label().unwrap())
        }
        "weightCol" => serde_json::json!(summary.weight_col().unwrap()),
        "weightedFMeasure-1.0" => serde_json::json!(summary.weighted_f_measure(1.0).unwrap()),
        "weightedFalsePositiveRate" => {
            serde_json::json!(summary.weighted_false_positive_rate().unwrap())
        }
        "weightedPrecision" => serde_json::json!(summary.weighted_precision().unwrap()),
        "weightedRecall" => serde_json::json!(summary.weighted_recall().unwrap()),
        "weightedTruePositiveRate" => {
            serde_json::json!(summary.weighted_true_positive_rate().unwrap())
        }
        other => panic!("no check for {other}"),
    };
    assert_golden(&actual, &expected, &format!("{path}.{key}"));
}

#[test]
fn logistic_regression_binomial_fit_predict_and_summaries_match_official_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let golden = golden();
    let summaries = summaries();
    let family = "LogisticRegression.binomial";
    let train = dataset(&s, &golden, "binary_train");
    let mut estimator = LogisticRegression::new();
    *estimator.params_mut() = fit_params(&golden, family);
    let model = estimator.fit(&train).unwrap();

    assert!(model.has_summary());
    let members = &golden["models"][family]["members"];
    assert_golden(
        &serde_json::json!(model.num_classes().unwrap()),
        &members["numClasses"]["value"],
        "numClasses",
    );
    assert_golden(
        &serde_json::json!(model.num_features().unwrap()),
        &members["numFeatures"]["value"],
        "numFeatures",
    );
    assert_golden(
        &serde_json::json!(model.intercept().unwrap()),
        &members["intercept"]["value"],
        "intercept",
    );
    assert_golden(
        &vector_json(&model.coefficients().unwrap()),
        &members["coefficients"]["value"],
        "coefficients",
    );
    assert_golden(
        &vector_json(&model.intercept_vector().unwrap()),
        &members["interceptVector"]["value"],
        "interceptVector",
    );
    assert_golden(
        &matrix_json(&model.coefficient_matrix().unwrap()),
        &members["coefficientMatrix"]["value"],
        "coefficientMatrix",
    );
    for (suffix, features) in [("dense", vec![0.5, 1.5]), ("dense2", vec![3.0, -0.25])] {
        let x = feature_vector(&features);
        assert_golden(
            &serde_json::json!(model.predict(&x).unwrap()),
            &members[format!("predict-{suffix}")]["value"],
            suffix,
        );
        assert_golden(
            &vector_json(&model.predict_raw(&x).unwrap()),
            &members[format!("predictRaw-{suffix}")]["value"],
            suffix,
        );
        assert_golden(
            &vector_json(&model.predict_probability(&x).unwrap()),
            &members[format!("predictProbability-{suffix}")]["value"],
            suffix,
        );
    }

    // The training summary: binary, with its scalars and its curves.
    let LogisticRegressionModelSummary::Binary(training) = model.summary().unwrap() else {
        panic!("a binary model has a binary summary");
    };
    let golden_training = &summaries[format!("{family}.training").as_str()]["members"];
    for key in [
        "accuracy",
        "falsePositiveRateByLabel",
        "fMeasureByLabel-1.0",
        "labelCol",
        "labels",
        "precisionByLabel",
        "predictionCol",
        "recallByLabel",
        "truePositiveRateByLabel",
        "weightCol",
        "weightedFMeasure-1.0",
        "weightedFalsePositiveRate",
        "weightedPrecision",
        "weightedRecall",
        "weightedTruePositiveRate",
    ] {
        check_classification(&training, golden_training, key, "training");
    }
    assert_golden(
        &serde_json::json!(training.area_under_roc().unwrap()),
        &golden_training["areaUnderROC"]["value"],
        "training.areaUnderROC",
    );
    assert_golden(
        &serde_json::json!(training.score_col().unwrap()),
        &golden_training["scoreCol"]["value"],
        "training.scoreCol",
    );
    assert_golden(
        &serde_json::json!(training.features_col().unwrap()),
        &golden_training["featuresCol"]["value"],
        "training.featuresCol",
    );
    assert_golden(
        &serde_json::json!(training.probability_col().unwrap()),
        &golden_training["probabilityCol"]["value"],
        "training.probabilityCol",
    );
    assert_golden(
        &serde_json::json!(training.objective_history().unwrap()),
        &golden_training["objectiveHistory"]["value"],
        "training.objectiveHistory",
    );
    assert_golden(
        &serde_json::json!(training.total_iterations().unwrap()),
        &golden_training["totalIterations"]["value"],
        "training.totalIterations",
    );
    assert_rows_close(
        sorted_rows(training.roc().unwrap()),
        golden_rows(&golden_training["roc"]["value"]),
        "training roc",
    );
    assert_rows_close(
        sorted_rows(training.pr().unwrap()),
        golden_rows(&golden_training["pr"]["value"]),
        "training pr",
    );
    assert_rows_close(
        sorted_rows(training.precision_by_threshold().unwrap()),
        golden_rows(&golden_training["precisionByThreshold"]["value"]),
        "training precisionByThreshold",
    );
    assert_rows_close(
        sorted_rows(training.recall_by_threshold().unwrap()),
        golden_rows(&golden_training["recallByThreshold"]["value"]),
        "training recallByThreshold",
    );
    assert_rows_close(
        sorted_rows(training.f_measure_by_threshold().unwrap()),
        golden_rows(&golden_training["fMeasureByThreshold"]["value"]),
        "training fMeasureByThreshold",
    );

    // The evaluate of the model on the evaluation dataset: a binary summary of its own.
    let eval = dataset(&s, &golden, "binary_eval");
    let LogisticRegressionEvaluation::Binary(evaluated) = model.evaluate(&eval).unwrap() else {
        panic!("a binary model evaluates to a binary summary");
    };
    let golden_evaluate = &summaries[format!("{family}.evaluate").as_str()]["members"];
    for key in [
        "accuracy",
        "falsePositiveRateByLabel",
        "fMeasureByLabel-1.0",
        "labelCol",
        "labels",
        "precisionByLabel",
        "predictionCol",
        "recallByLabel",
        "truePositiveRateByLabel",
        "weightCol",
        "weightedFMeasure-1.0",
        "weightedFalsePositiveRate",
        "weightedPrecision",
        "weightedRecall",
        "weightedTruePositiveRate",
    ] {
        check_classification(&evaluated, golden_evaluate, key, "evaluate");
    }
    assert_golden(
        &serde_json::json!(evaluated.area_under_roc().unwrap()),
        &golden_evaluate["areaUnderROC"]["value"],
        "evaluate.areaUnderROC",
    );
    assert_rows_close(
        sorted_rows(evaluated.roc().unwrap()),
        golden_rows(&golden_evaluate["roc"]["value"]),
        "evaluate roc",
    );

    // A model saved and loaded has no training summary, and predicts as the model it saved.
    let path = scratch_path("lr-binomial");
    model.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = LogisticRegressionModel::load(&s, &path).unwrap();
    assert!(!loaded.has_summary());
    assert!(loaded.summary().is_err());
    assert_golden(
        &serde_json::json!(loaded.predict(&feature_vector(&[0.5, 1.5])).unwrap()),
        &golden["models"][family]["members"]["predict-dense"]["value"],
        "loaded predict",
    );
    let estimator_path = scratch_path("lr-binomial-estimator");
    estimator
        .write()
        .session(&s)
        .overwrite()
        .save(&estimator_path)
        .unwrap();
    let reloaded = LogisticRegression::load(&s, &estimator_path).unwrap();
    assert_eq!(
        Estimator::params(&reloaded).to_proto(),
        Estimator::params(&estimator).to_proto()
    );
}

#[test]
fn logistic_regression_multinomial_coefficients_and_summary_match_official_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let golden = golden();
    let summaries = summaries();
    let family = "LogisticRegression.multinomial";
    let train = dataset(&s, &golden, "multi_train");
    let mut estimator = LogisticRegression::new();
    *estimator.params_mut() = fit_params(&golden, family);
    let model = estimator.fit(&train).unwrap();

    // The server's error for the vector of a multinomial model, and its matrix.
    let error = model.coefficients().unwrap_err();
    assert!(
        error.to_string().contains(
            "Multinomial models contain a matrix of coefficients, use coefficientMatrix instead."
        ),
        "{error}"
    );
    assert!(model.intercept().is_err());
    let members = &golden["models"][family]["members"];
    assert_golden(
        &matrix_json(&model.coefficient_matrix().unwrap()),
        &members["coefficientMatrix"]["value"],
        "coefficientMatrix",
    );
    assert_golden(
        &vector_json(&model.intercept_vector().unwrap()),
        &members["interceptVector"]["value"],
        "interceptVector",
    );

    let LogisticRegressionModelSummary::Multiclass(training) = model.summary().unwrap() else {
        panic!("a multiclass model has a multiclass summary");
    };
    let golden_training = &summaries[format!("{family}.training").as_str()]["members"];
    for key in [
        "accuracy",
        "falsePositiveRateByLabel",
        "labelCol",
        "labels",
        "precisionByLabel",
        "predictionCol",
        "recallByLabel",
        "truePositiveRateByLabel",
        "weightCol",
        "weightedFalsePositiveRate",
        "weightedPrecision",
        "weightedRecall",
        "weightedTruePositiveRate",
        "fMeasureByLabel-1.0",
        "weightedFMeasure-1.0",
    ] {
        check_classification(&training, golden_training, key, "training");
    }
    assert_golden(
        &serde_json::json!(training.total_iterations().unwrap()),
        &golden_training["totalIterations"]["value"],
        "training.totalIterations",
    );
    assert_golden(
        &serde_json::json!(training.objective_history().unwrap()),
        &golden_training["objectiveHistory"]["value"],
        "training.objectiveHistory",
    );

    let eval = dataset(&s, &golden, "multi_eval");
    let LogisticRegressionEvaluation::Multiclass(evaluated) = model.evaluate(&eval).unwrap() else {
        panic!("a multiclass model evaluates to a multiclass summary");
    };
    let golden_evaluate = &summaries[format!("{family}.evaluate").as_str()]["members"];
    assert_golden(
        &serde_json::json!(evaluated.accuracy().unwrap()),
        &golden_evaluate["accuracy"]["value"],
        "evaluate.accuracy",
    );
    assert_golden(
        &serde_json::json!(evaluated.weighted_f_measure(0.5).unwrap()),
        &golden_evaluate["weightedFMeasure-0.5"]["value"],
        "evaluate.weightedFMeasure-0.5",
    );
    assert_eq!(
        summary_value(&summaries, &format!("{family}.evaluate"), "labels"),
        serde_json::json!(evaluated.labels().unwrap())
    );
}

#[test]
fn linear_svc_model_and_summaries_match_official_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let golden = golden();
    let summaries = summaries();
    let family = "LinearSVC";
    let train = dataset(&s, &golden, "binary_train");
    let mut estimator = LinearSVC::new();
    *estimator.params_mut() = fit_params(&golden, family);
    let model: LinearSVCModel = estimator.fit(&train).unwrap();
    assert!(model.has_summary());

    let members = &golden["models"][family]["members"];
    assert_golden(
        &vector_json(&model.coefficients().unwrap()),
        &members["coefficients"]["value"],
        "coefficients",
    );
    assert_golden(
        &serde_json::json!(model.intercept().unwrap()),
        &members["intercept"]["value"],
        "intercept",
    );
    assert_golden(
        &vector_json(&model.predict_raw(&feature_vector(&[0.5, 1.5])).unwrap()),
        &members["predictRaw-dense"]["value"],
        "predictRaw-dense",
    );
    assert_golden(
        &serde_json::json!(model.predict(&feature_vector(&[0.5, 1.5])).unwrap()),
        &members["predict-dense"]["value"],
        "predict-dense",
    );

    let training = model.summary().unwrap();
    let golden_training = &summaries[format!("{family}.training").as_str()]["members"];
    for key in [
        "accuracy",
        "labelCol",
        "labels",
        "predictionCol",
        "weightCol",
    ] {
        check_classification(&training, golden_training, key, "training");
    }
    assert_golden(
        &serde_json::json!(training.area_under_roc().unwrap()),
        &golden_training["areaUnderROC"]["value"],
        "training.areaUnderROC",
    );
    assert_golden(
        &serde_json::json!(training.objective_history().unwrap()),
        &golden_training["objectiveHistory"]["value"],
        "training.objectiveHistory",
    );
    assert_golden(
        &serde_json::json!(training.total_iterations().unwrap()),
        &golden_training["totalIterations"]["value"],
        "training.totalIterations",
    );
    assert_rows_close(
        sorted_rows(training.roc().unwrap()),
        golden_rows(&golden_training["roc"]["value"]),
        "training roc",
    );
    assert_rows_close(
        sorted_rows(training.pr().unwrap()),
        golden_rows(&golden_training["pr"]["value"]),
        "training pr",
    );

    let eval = dataset(&s, &golden, "binary_eval");
    let evaluated = model.evaluate(&eval).unwrap();
    let golden_evaluate = &summaries[format!("{family}.evaluate").as_str()]["members"];
    check_classification(&evaluated, golden_evaluate, "accuracy", "evaluate");
    assert_golden(
        &serde_json::json!(evaluated.area_under_roc().unwrap()),
        &golden_evaluate["areaUnderROC"]["value"],
        "evaluate.areaUnderROC",
    );

    let path = scratch_path("linear-svc");
    model.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = LinearSVCModel::load(&s, &path).unwrap();
    assert!(!loaded.has_summary());
    assert!(loaded.summary().is_err());
    assert_golden(
        &vector_json(&loaded.coefficients().unwrap()),
        &members["coefficients"]["value"],
        "loaded coefficients",
    );
}

/// The typed accessors of a model fitted with a param map: one fit per map, in order, each over the
/// estimator's own params.
#[test]
fn logistic_regression_fit_multiple_fits_one_model_per_param_map() {
    let Some(s) = live_session() else {
        return;
    };
    let golden = golden();
    let family = "LogisticRegression.binomial";
    let train = dataset(&s, &golden, "binary_train");
    let mut estimator = LogisticRegression::new();
    *estimator.params_mut() = fit_params(&golden, family);
    let maps = [
        Params::new().set_param_int32("maxIter", 20),
        Params::new().set_param_int32("maxIter", 1),
    ];
    let models = estimator.fit_multiple(&train, &maps).unwrap();
    assert_eq!(models.len(), 2);
    assert_golden(
        &vector_json(&models[0].coefficients().unwrap()),
        &golden["models"][family]["members"]["coefficients"]["value"],
        "maxIter 20",
    );
    assert_ne!(
        models[0].coefficients().unwrap(),
        models[1].coefficients().unwrap(),
        "one iteration does not reach the golden coefficients"
    );
    assert!(models.iter().all(|model| model.has_summary()));
}

fn scratch_path(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "spark-connect-rust-lr-svc-{}-{name}",
        std::process::id()
    ));
    dir.to_string_lossy().into_owned()
}
