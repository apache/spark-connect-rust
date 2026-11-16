// SPDX-License-Identifier: Apache-2.0
//! The typed fits, accessors, summaries and persistence of NaiveBayesModel,
//! MultilayerPerceptronClassificationModel, FMClassificationModel and OneVsRestModel against a
//! live server, on the golden's datasets: each value is compared with official PySpark 4.3.0's
//! within 1e-9 relative. Set SPARK_REMOTE to run them.
mod common;
#[path = "common/ml_golden.rs"]
mod ml_golden;

use common::live_session;
use serde_json::Value;
use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::ml::{
    fetch, load_metadata, BinaryClassificationEvaluator, BinaryClassificationSummary,
    ClassificationModel, ClassificationSummary, Classifier, CrossValidator, CrossValidatorModel,
    Estimator, FMClassificationModel, FMClassifier, FetchArg, LogisticRegression, MlMatrix,
    MlTarget, MlValue, MlVector, Model, MultilayerPerceptronClassificationModel,
    MultilayerPerceptronClassifier, NaiveBayes, NaiveBayesModel, OneVsRest, OneVsRestModel,
    ParamGridBuilder, PredictionModel, ProbabilisticClassificationModel, SubModel, TrainingSummary,
    Transformer,
};
use spark_connect::{Column, DataFrame, SparkSession};

const MODELS: &str = include_str!("../../../tests/golden/classification_models.json");
const SUMMARIES: &str = include_str!("../../../tests/golden/classification_summaries.json");

fn golden() -> Value {
    serde_json::from_str(MODELS).unwrap()
}

fn summaries() -> Value {
    serde_json::from_str(SUMMARIES).unwrap()
}

/// The golden member `member` of the model entry `entry` of `section`.
fn member<'a>(root: &'a Value, section: &str, entry: &str, member: &str) -> &'a Value {
    &root[section][entry]["members"][member]["value"]
}

/// Asserts that `actual` is the golden `expected` within 1e-9 relative.
fn close(actual: f64, expected: &Value) {
    let expected = expected
        .as_f64()
        .unwrap_or_else(|| panic!("not a number: {expected}"));
    assert!(
        actual == expected || (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
        "{actual} is not the golden {expected}"
    );
}

/// Asserts that `actual` are the golden's values: an array, or a vector or matrix of `values`.
fn close_all(actual: &[f64], expected: &Value) {
    let expected = expected
        .get("values")
        .unwrap_or(expected)
        .as_array()
        .unwrap();
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        close(*a, e);
    }
}

fn vector_values(v: &MlVector) -> Vec<f64> {
    match v {
        MlVector::Dense(values) => values.clone(),
        MlVector::Sparse {
            size,
            indices,
            values,
        } => {
            let mut dense = vec![0.0; *size as usize];
            for (i, value) in indices.iter().zip(values) {
                dense[*i as usize] = *value;
            }
            dense
        }
    }
}

fn matrix_values(m: &MlMatrix) -> (i32, i32, Vec<f64>) {
    match m {
        MlMatrix::Dense {
            num_rows,
            num_cols,
            values,
            ..
        } => (*num_rows, *num_cols, values.clone()),
        MlMatrix::Sparse { .. } => panic!("the golden has no sparse matrix here"),
    }
}

/// The DataFrame of a golden dataset: its `label`, `weight` and `features` columns.
fn dataset(s: &SparkSession, root: &Value, name: &str) -> DataFrame {
    let rows: Vec<String> = root["datasets"][name]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let features: Vec<String> = row[2]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| format!("{:?}D", x.as_f64().unwrap()))
                .collect();
            format!(
                "({:?}D, {:?}D, array({}))",
                row[0].as_f64().unwrap(),
                row[1].as_f64().unwrap(),
                features.join(", ")
            )
        })
        .collect();
    let sql = format!(
        "SELECT label, weight, xs FROM VALUES {} AS t(label, weight, xs)",
        rows.join(", ")
    );
    s.sql(&sql).unwrap().coalesce(1).with_column(
        "features",
        Column::new(Expression::function(
            "array_to_vector",
            vec![col("xs").expression().clone()],
        )),
    )
}

fn scratch_path(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "spark-connect-rust-classification-misc-{}-{name}",
        std::process::id()
    ));
    dir.to_string_lossy().into_owned()
}

/// The predictions of a model of `vectors`, each the golden's `predict-*`, `predictRaw-*` and
/// `predictProbability-*` of `entry`.
fn check_predictions<M: ProbabilisticClassificationModel>(
    model: &M,
    root: &Value,
    section: &str,
    entry: &str,
    names: &[(&str, MlVector)],
) {
    for (name, vector) in names {
        close(
            model.predict(vector).unwrap(),
            member(root, section, entry, &format!("predict-{name}")),
        );
        close_all(
            &vector_values(&model.predict_raw(vector).unwrap()),
            member(root, section, entry, &format!("predictRaw-{name}")),
        );
        close_all(
            &vector_values(&model.predict_probability(vector).unwrap()),
            member(root, section, entry, &format!("predictProbability-{name}")),
        );
    }
}

fn nb_vectors() -> Vec<(&'static str, MlVector)> {
    vec![
        ("dense", MlVector::Dense(vec![1.0, 0.0])),
        ("dense2", MlVector::Dense(vec![0.0, 1.0])),
        (
            "sparse",
            MlVector::Sparse {
                size: 2,
                indices: vec![1],
                values: vec![1.0],
            },
        ),
    ]
}

fn mlp_vectors() -> Vec<(&'static str, MlVector)> {
    vec![
        ("dense", MlVector::Dense(vec![0.5, 1.5])),
        ("dense2", MlVector::Dense(vec![3.0, -0.25])),
        (
            "sparse",
            MlVector::Sparse {
                size: 2,
                indices: vec![1],
                values: vec![2.0],
            },
        ),
    ]
}

#[test]
fn naive_bayes_fits_match_official_pyspark_for_each_model_type() {
    let Some(s) = live_session() else {
        return;
    };
    let root = golden();
    for (key, model_type, data) in [
        ("NaiveBayes.multinomial", "multinomial", "nb_counts"),
        ("NaiveBayes.complement", "complement", "nb_counts"),
        ("NaiveBayes.bernoulli", "bernoulli", "nb_binary"),
        ("NaiveBayes.gaussian", "gaussian", "binary_train"),
    ] {
        let frame = dataset(&s, &root, data);
        let mut model: NaiveBayesModel = NaiveBayes::new()
            .set_model_type(model_type)
            .set_weight_col("weight")
            .fit(&frame)
            .unwrap();
        close(
            model.num_classes().unwrap() as f64,
            member(&root, "models", key, "numClasses"),
        );
        close(
            model.num_features().unwrap() as f64,
            member(&root, "models", key, "numFeatures"),
        );
        close_all(
            &vector_values(&model.pi().unwrap()),
            member(&root, "models", key, "pi"),
        );
        let (rows, cols, values) = matrix_values(&model.theta().unwrap());
        let theta = member(&root, "models", key, "theta");
        assert_eq!(
            (rows as i64, cols as i64),
            (
                theta["numRows"].as_i64().unwrap(),
                theta["numCols"].as_i64().unwrap()
            )
        );
        close_all(&values, &theta["values"]);
        let (_, _, sigma) = matrix_values(&model.sigma().unwrap());
        close_all(&sigma, &member(&root, "models", key, "sigma")["values"]);
        check_predictions(&model, &root, "models", key, &nb_vectors());
        // The transform of the golden's evaluation dataset: its schema and rows, as PySpark's.
        let transform = &root["models"][key]["members"]["transform"];
        // The helper's dataset keeps its raw `xs` column, which the golden's does not have.
        let eval = dataset(&s, &root, ml_golden::call_dataset(&transform["call"])).drop(["xs"]);
        let output = model.transform(&eval).unwrap();
        let actual = ml_golden::frame(&output).unwrap();
        ml_golden::assert_matches(
            &ml_golden::normalized(actual),
            &ml_golden::normalized(transform["value"].clone()),
            &format!("{key}: transform"),
        );
    }
}

#[test]
fn naive_bayes_model_saves_and_loads_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let root = golden();
    let frame = dataset(&s, &root, "nb_counts");
    let model = NaiveBayes::new()
        .set_model_type("multinomial")
        .set_weight_col("weight")
        .fit(&frame)
        .unwrap();
    let path = scratch_path("nb");
    model.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = NaiveBayesModel::load(&s, &path).unwrap();
    assert_eq!(
        Transformer::operator(&loaded).uid,
        Transformer::operator(&model).uid
    );
    check_predictions(
        &loaded,
        &root,
        "models",
        "NaiveBayes.multinomial",
        &nb_vectors()[..1],
    );
    assert!(loaded.predict(&MlVector::Dense(vec![1.0, 0.0])).is_ok());
}

#[test]
fn multilayer_perceptron_fit_summaries_and_evaluation_match_official_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let root = golden();
    let summaries = summaries();
    let key = "MultilayerPerceptronClassifier";
    let frame = dataset(&s, &root, "multi_train");
    let model: MultilayerPerceptronClassificationModel = MultilayerPerceptronClassifier::new()
        .set_layers([2, 4, 3])
        .set_max_iter(30)
        .set_seed(42)
        .set_block_size(4)
        .set_step_size(0.1)
        .fit(&frame)
        .unwrap();
    close_all(
        &vector_values(&model.weights().unwrap()),
        member(&root, "models", key, "weights"),
    );
    check_predictions(&model, &root, "models", key, &mlp_vectors());

    assert!(model.has_summary());
    let training = model.summary().unwrap();
    let golden_training = &summaries["summaries"][format!("{key}.training")]["members"];
    close(
        training.accuracy().unwrap(),
        &golden_training["accuracy"]["value"],
    );
    close(
        training.weighted_f_measure(1.0).unwrap(),
        &golden_training["weightedFMeasure-1.0"]["value"],
    );
    close_all(
        &training.objective_history().unwrap(),
        &golden_training["objectiveHistory"]["value"],
    );
    assert_eq!(
        training.total_iterations().unwrap() as i64,
        golden_training["totalIterations"]["value"]
            .as_i64()
            .unwrap()
    );

    let frame = dataset(&s, &root, "multi_eval");
    let evaluated = model.evaluate(&frame).unwrap();
    let golden_eval = &summaries["summaries"][format!("{key}.evaluate")]["members"];
    close(
        evaluated.accuracy().unwrap(),
        &golden_eval["accuracy"]["value"],
    );
    close_all(
        &evaluated.labels().unwrap(),
        &golden_eval["labels"]["value"],
    );
}

#[test]
fn fm_fit_summaries_and_evaluation_match_official_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let root = golden();
    let summaries = summaries();
    let key = "FMClassifier";
    let frame = dataset(&s, &root, "binary_train");
    let model: FMClassificationModel = FMClassifier::new()
        .set_factor_size(2)
        .set_max_iter(20)
        .set_seed(42)
        .set_reg_param(0.01)
        .set_step_size(0.1)
        .fit(&frame)
        .unwrap();
    let (rows, cols, factors) = matrix_values(&model.factors().unwrap());
    let golden_factors = member(&root, "models", key, "factors");
    assert_eq!(
        (rows as i64, cols as i64),
        (
            golden_factors["numRows"].as_i64().unwrap(),
            golden_factors["numCols"].as_i64().unwrap()
        )
    );
    close_all(&factors, &golden_factors["values"]);
    close_all(
        &vector_values(&model.linear().unwrap()),
        member(&root, "models", key, "linear"),
    );
    close(
        model.intercept().unwrap(),
        member(&root, "models", key, "intercept"),
    );
    check_predictions(&model, &root, "models", key, &mlp_vectors());

    let training = model.summary().unwrap();
    let golden_training = &summaries["summaries"][format!("{key}.training")]["members"];
    close(
        training.accuracy().unwrap(),
        &golden_training["accuracy"]["value"],
    );
    close(
        training.area_under_roc().unwrap(),
        &golden_training["areaUnderROC"]["value"],
    );
    close_all(
        &training.objective_history().unwrap(),
        &golden_training["objectiveHistory"]["value"],
    );

    // The ROC curve is a DataFrame of FPR and TPR, which the golden collects (rows sorted).
    let mut points: Vec<Vec<f64>> = training
        .roc()
        .unwrap()
        .collect()
        .unwrap()
        .into_iter()
        .map(|row| {
            vec![
                row.get(0).unwrap().as_f64().unwrap(),
                row.get(1).unwrap().as_f64().unwrap(),
            ]
        })
        .collect();
    points.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let golden_roc = golden_training["roc"]["value"]["rows"].as_array().unwrap();
    assert_eq!(points.len(), golden_roc.len());
    for (point, expected) in points.iter().zip(golden_roc) {
        close_all(point, expected);
    }

    let frame = dataset(&s, &root, "binary_eval");
    let evaluated = model.evaluate(&frame).unwrap();
    let golden_eval = &summaries["summaries"][format!("{key}.evaluate")]["members"];
    close(
        evaluated.accuracy().unwrap(),
        &golden_eval["accuracy"]["value"],
    );
    close(
        evaluated.area_under_roc().unwrap(),
        &golden_eval["areaUnderROC"]["value"],
    );
}

#[test]
fn fm_model_saves_and_loads_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let root = golden();
    let frame = dataset(&s, &root, "binary_train");
    let model = FMClassifier::new()
        .set_factor_size(2)
        .set_max_iter(20)
        .set_seed(42)
        .set_reg_param(0.01)
        .set_step_size(0.1)
        .fit(&frame)
        .unwrap();
    assert!(model.has_summary());
    let path = scratch_path("fm");
    model.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = FMClassificationModel::load(&s, &path).unwrap();
    close(
        loaded.intercept().unwrap(),
        member(&root, "models", "FMClassifier", "intercept"),
    );
    // A loaded model has no training summary, as PySpark's.
    assert!(!loaded.has_summary());
    assert!(loaded.summary().is_err());
}

#[test]
fn multilayer_perceptron_model_saves_and_loads_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let root = golden();
    let frame = dataset(&s, &root, "multi_train");
    let model = MultilayerPerceptronClassifier::new()
        .set_layers([2, 4, 3])
        .set_max_iter(30)
        .set_seed(42)
        .set_block_size(4)
        .set_step_size(0.1)
        .fit(&frame)
        .unwrap();
    let path = scratch_path("mlp");
    model.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = MultilayerPerceptronClassificationModel::load(&s, &path).unwrap();
    close_all(
        &vector_values(&loaded.weights().unwrap()),
        member(&root, "models", "MultilayerPerceptronClassifier", "weights"),
    );
    assert!(!loaded.has_summary());
}

#[test]
fn one_vs_rest_models_are_the_binary_submodels_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let root = golden();
    let frame = dataset(&s, &root, "multi_train");
    let classifier = LogisticRegression::new().set_max_iter(20);
    let model = OneVsRest::new()
        .set_classifier(classifier)
        .set_weight_col("weight")
        .fit(&frame)
        .unwrap();
    assert_eq!(model.num_classes(), 3);
    let models = model.models();
    assert_eq!(models.len(), 3);
    for (index, sub) in models.iter().enumerate() {
        let SubModel::LogisticRegression(binary) = sub else {
            panic!("a logistic regression submodel");
        };
        // The submodel's own prediction, of the golden's vector.
        let value = fetch(
            &MlTarget::object(binary.ml_ref().unwrap()),
            "predict",
            vec![FetchArg::Value(MlValue::Vector(MlVector::Dense(vec![
                0.5, 1.5,
            ])))],
        )
        .unwrap()
        .into_value()
        .unwrap();
        close(
            value.as_f64().unwrap(),
            member(
                &root,
                "models",
                &format!("OneVsRest.models[{index}]"),
                "predict-dense",
            ),
        );
    }
}

/// The golden's `OneVsRest.transform` of `multi_eval`, as (label, weight, rawPrediction,
/// prediction) rows: PySpark's own values.
fn one_vs_rest_golden_transform(root: &Value) -> Vec<(f64, f64, Vec<f64>, f64)> {
    member(root, "models", "OneVsRest", "transform")["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let raw = row[3]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            (
                row[0].as_f64().unwrap(),
                row[1].as_f64().unwrap(),
                raw,
                row[4].as_f64().unwrap(),
            )
        })
        .collect()
}

/// The rows of the transform `df` as (label, weight, rawPrediction, prediction), of its three
/// raw scores and its prediction.
fn one_vs_rest_rows(df: &DataFrame) -> Vec<(f64, f64, Vec<f64>, f64)> {
    use spark_connect::column::lit;
    use spark_connect::functions::get;
    let scores = Column::new(Expression::function(
        "vector_to_array",
        vec![
            col("rawPrediction").expression().clone(),
            spark_connect::column::lit_string("float64")
                .expression()
                .clone(),
        ],
    ));
    let raw = |index: i64, name: &str| get(scores.clone(), lit(index)).alias(name);
    let rows = df
        .select([
            col("label"),
            col("weight"),
            raw(0, "raw0"),
            raw(1, "raw1"),
            raw(2, "raw2"),
            col("prediction"),
        ])
        .collect()
        .unwrap();
    rows.iter()
        .map(|row| {
            let value = |index: usize| row.get(index).and_then(|v| v.as_f64()).unwrap();
            (
                value(0),
                value(1),
                vec![value(2), value(3), value(4)],
                value(5),
            )
        })
        .collect()
}

/// Asserts that the rows of `actual` are PySpark's golden rows, within 1e-9 relative, matched by
/// their label and weight.
fn assert_one_vs_rest_golden(
    actual: &[(f64, f64, Vec<f64>, f64)],
    golden: &[(f64, f64, Vec<f64>, f64)],
) {
    assert_eq!(actual.len(), golden.len());
    for (label, weight, raw, prediction) in golden {
        let row = actual
            .iter()
            .find(|row| row.0 == *label && row.1 == *weight)
            .unwrap_or_else(|| panic!("no row of label {label} and weight {weight}"));
        assert_eq!(row.2.len(), raw.len());
        for (actual, expected) in row.2.iter().zip(raw) {
            assert!(
                (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "{actual} is not the golden {expected}"
            );
        }
        assert!(
            (row.3 - prediction).abs() <= 1e-9 * prediction.abs().max(1.0),
            "{} is not the golden {prediction}",
            row.3
        );
    }
}

/// The OneVsRest model of the golden's fit, as `OneVsRest.fit(multi_train)` of PySpark.
fn one_vs_rest_golden_model(s: &SparkSession, root: &Value) -> OneVsRestModel {
    let frame = dataset(s, root, "multi_train");
    OneVsRest::new()
        .set_classifier(
            LogisticRegression::new()
                .set_max_iter(20)
                .set_reg_param(0.01),
        )
        .set_weight_col("weight")
        .fit(&frame)
        .unwrap()
}

#[test]
fn one_vs_rest_transform_matches_pyspark_on_the_golden_eval_rows() {
    let Some(s) = live_session() else {
        return;
    };
    let root = golden();
    let mut model = one_vs_rest_golden_model(&s, &root);
    let eval = dataset(&s, &root, "multi_eval");
    let actual = one_vs_rest_rows(&model.transform(&eval).unwrap());
    assert_one_vs_rest_golden(&actual, &one_vs_rest_golden_transform(&root));
}

#[test]
fn one_vs_rest_transform_reads_the_features_of_its_submodels_as_pyspark() {
    // PySpark's OneVsRestModel.setFeaturesCol("missing") transforms with the features col of each
    // submodel, which exists, as the model's own featuresCol does not.
    let Some(s) = live_session() else {
        return;
    };
    let root = golden();
    let mut model = one_vs_rest_golden_model(&s, &root);
    let params = std::mem::take(model.params_mut()).set_param_string("featuresCol", "missing");
    *model.params_mut() = params;
    let eval = dataset(&s, &root, "multi_eval");
    let actual = one_vs_rest_rows(&model.transform(&eval).unwrap());
    assert_one_vs_rest_golden(&actual, &one_vs_rest_golden_transform(&root));
}

#[test]
fn one_vs_rest_transform_of_another_raw_prediction_col_fails_as_pyspark() {
    // The submodels write rawPrediction, which the model's rawPredictionCol "scores" does not
    // name: PySpark's select of it fails to resolve the column, when the transform is evaluated.
    let Some(s) = live_session() else {
        return;
    };
    let root = golden();
    let mut model = one_vs_rest_golden_model(&s, &root);
    let params = std::mem::take(model.params_mut()).set_param_string("rawPredictionCol", "scores");
    *model.params_mut() = params;
    let eval = dataset(&s, &root, "multi_eval");
    let error = match model.transform(&eval).and_then(|df| df.collect()) {
        Err(error) => error,
        Ok(_) => panic!("the transform of an unresolved rawPredictionCol succeeded"),
    };
    assert_eq!(error.error_class, "UNRESOLVED_COLUMN.WITH_SUGGESTION");
}

#[test]
fn naive_bayes_bernoulli_rejects_features_outside_zero_one_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let root = golden();
    let frame = dataset(&s, &root, "nb_binary");
    let model = NaiveBayes::new()
        .set_model_type("bernoulli")
        .set_weight_col("weight")
        .fit(&frame)
        .unwrap();
    let error = model
        .predict(&MlVector::Dense(vec![0.5, 1.5]))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("Bernoulli naive Bayes requires 0 or 1"),
        "{error}"
    );
}

#[test]
fn fitted_summaries_keep_their_dataframes_and_columns() {
    let Some(s) = live_session() else {
        return;
    };
    let root = golden();
    let frame = dataset(&s, &root, "binary_train");
    let model = FMClassifier::new()
        .set_factor_size(2)
        .set_max_iter(20)
        .set_seed(42)
        .set_reg_param(0.01)
        .set_step_size(0.1)
        .fit(&frame)
        .unwrap();
    let summary = model.summary().unwrap();
    assert_eq!(summary.label_col().unwrap(), "label");
    assert_eq!(summary.prediction_col().unwrap(), "prediction");
    assert_eq!(summary.score_col().unwrap(), "probability");
    let predictions = summary.predictions().unwrap();
    assert_eq!(predictions.count().unwrap(), 8);
}

/// The predictions of `df` in its row order, as its `prediction` column.
fn predictions(df: &DataFrame) -> Vec<f64> {
    let index = df
        .columns()
        .unwrap()
        .iter()
        .position(|name| name == "prediction")
        .expect("a prediction column");
    df.collect()
        .unwrap()
        .iter()
        .map(|row| row.get(index).and_then(|value| value.as_f64()).unwrap())
        .collect()
}

#[test]
fn one_vs_rest_saves_and_loads_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let root = golden();
    let frame = dataset(&s, &root, "multi_train");
    let mut estimator = OneVsRest::new()
        .set_classifier(LogisticRegression::new().set_max_iter(20))
        .set_weight_col("weight");
    let uid = estimator.operator().uid.clone();
    let path = scratch_path("ovr");
    estimator
        .write()
        .session(&s)
        .overwrite()
        .save(&path)
        .unwrap();
    // The metadata of PySpark's OneVsRestWriter: the params without the classifier, which its own
    // writer saves at `classifier`.
    let metadata = load_metadata(&s, &path, Some("pyspark.ml.classification.OneVsRest")).unwrap();
    assert_eq!(metadata["uid"], uid.as_str());
    assert_eq!(metadata["paramMap"]["weightCol"], "weight");
    assert_eq!(metadata["paramMap"]["parallelism"], 1);
    assert!(metadata["paramMap"].get("classifier").is_none());
    assert_eq!(metadata["defaultParamMap"]["featuresCol"], "features");
    assert!(std::path::Path::new(&format!("{path}/classifier/metadata")).exists());
    let loaded = OneVsRest::load(&s, &path).unwrap();
    assert_eq!(loaded.operator().uid, uid);
    assert_eq!(loaded.weight_col(), Some("weight"));
    assert_eq!(loaded.parallelism(), 1);
    assert!(matches!(
        loaded.classifier(),
        Some(Classifier::LogisticRegression(_))
    ));

    // The model: its metadata with `numClasses`, its classifier, and one model per class.
    let mut model = estimator.fit(&frame).unwrap();
    let model_path = scratch_path("ovr-model");
    model
        .write()
        .unwrap()
        .session(&s)
        .overwrite()
        .save(&model_path)
        .unwrap();
    let metadata = load_metadata(
        &s,
        &model_path,
        Some("pyspark.ml.classification.OneVsRestModel"),
    )
    .unwrap();
    assert_eq!(metadata["numClasses"], 3);
    assert!(metadata["paramMap"].get("parallelism").is_none());
    for index in 0..3 {
        assert!(std::path::Path::new(&format!("{model_path}/model_{index}/metadata")).exists());
    }
    let mut loaded_model = OneVsRestModel::load(&s, &model_path).unwrap();
    assert_eq!(loaded_model.num_classes(), 3);
    assert_eq!(
        Transformer::operator(&loaded_model).uid,
        Transformer::operator(&model).uid
    );
    assert_eq!(
        predictions(&loaded_model.transform(&frame).unwrap()),
        predictions(&model.transform(&frame).unwrap())
    );
    // A model that holds its classifier and its models saves them again, as PySpark's does.
    let resaved = scratch_path("ovr-model-again");
    loaded_model
        .write()
        .unwrap()
        .session(&s)
        .overwrite()
        .save(&resaved)
        .unwrap();
    assert_eq!(load_metadata(&s, &resaved, None).unwrap()["numClasses"], 3);
}

#[test]
fn cross_validator_over_one_vs_rest_saves_and_loads_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let root = golden();
    let frame = dataset(&s, &root, "multi_train");
    let classifier = LogisticRegression::new().set_max_iter(5);
    let classifier_uid = classifier.operator().uid.clone();
    let grid = ParamGridBuilder::new()
        .add_grid(
            &classifier_uid,
            "regParam",
            [MlValue::Double(0.0), MlValue::Double(0.5)],
        )
        .build();
    let mut cv = CrossValidator::new()
        .set_estimator(OneVsRest::new().set_classifier(classifier))
        .set_estimator_param_maps(grid.clone())
        .set_evaluator(BinaryClassificationEvaluator::new())
        .set_num_folds(2)
        .set_seed(7);
    // The save of the validator checks that each param of its maps is of a nested stage of its
    // estimator, the classifier of the OneVsRest, which PySpark's `getAllNestedStages` finds.
    let path = scratch_path("cv-ovr");
    cv.write().session(&s).overwrite().save(&path).unwrap();
    let metadata = load_metadata(&s, &path, Some("pyspark.ml.tuning.CrossValidator")).unwrap();
    assert_eq!(
        metadata["paramMap"]["estimatorParamMaps"][1][0]["parent"],
        classifier_uid.as_str()
    );
    let loaded = CrossValidator::load(&s, &path).unwrap();
    assert_eq!(loaded.estimator_param_maps(), grid.as_slice());
    let mut model: CrossValidatorModel = cv.fit_validator(&frame).unwrap();
    let model_path = scratch_path("cv-ovr-model");
    model
        .write()
        .unwrap()
        .session(&s)
        .overwrite()
        .save(&model_path)
        .unwrap();
    let mut loaded_model = CrossValidatorModel::load(&s, &model_path).unwrap();
    assert_eq!(loaded_model.avg_metrics(), model.avg_metrics());
    assert_eq!(
        predictions(&loaded_model.transform(&frame).unwrap()),
        predictions(&model.transform(&frame).unwrap())
    );
}
