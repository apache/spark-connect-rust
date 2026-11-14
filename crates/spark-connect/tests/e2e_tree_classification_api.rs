// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, summaries and persistence of the decision tree and random forest
//! classification models. The decision tree is deterministic, so its values are PySpark 4.3.0's
//! on the golden datasets. A random forest's trees depend on the partitioning of the server, so
//! its values are checked against invariants (the predictions, the summaries' counts) and the
//! golden values that do not depend on it. Set SPARK_REMOTE to run against a live Spark Connect
//! server.
mod common;
#[path = "common/tree_classification.rs"]
mod tree_classification;

use common::live_session;
use serde_json::{json, Value};
use spark_connect::column::{col, lit_string};
use spark_connect::expression::Expression;
use spark_connect::ml::{
    BinaryClassificationSummary, ClassificationModel, ClassificationSummary,
    DecisionTreeClassificationModel, DecisionTreeClassifier, DecisionTreeModel, Estimator,
    GBTClassificationModel, GBTClassifier, MlSummary, MlVector, Model, PredictionModel,
    ProbabilisticClassificationModel, RandomForestClassificationModel, RandomForestClassifier,
    RandomForestSummary, RandomForestTrainingSummary, TrainingSummary, Transformer,
    TreeEnsembleModel,
};
use spark_connect::row::{Row, Value as RowValue};
use spark_connect::Column;
use spark_connect::DataFrame;
use tree_classification::{assert_close, dataset, expected, vector_json};

/// The golden of the models and their datasets, with the summaries under `summaries`.
fn golden() -> Value {
    let mut models = tree_classification::golden("classification_models.json");
    let summaries = tree_classification::golden("classification_summaries.json");
    models["summaries"] = summaries["summaries"].clone();
    models
}

/// The rows of a golden dataset's features, in order: the dense vectors of `binary_eval`.
fn features_of(golden: &Value, name: &str) -> Vec<MlVector> {
    golden["datasets"][name]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            MlVector::Dense(
                row[2]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
            )
        })
        .collect()
}

/// The features of the golden's predict calls: dense, dense again, and sparse.
fn probes() -> Vec<(&'static str, MlVector)> {
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

fn scratch_path(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "spark-connect-rust-tree-api-{}-{name}",
        std::process::id()
    ));
    dir.to_string_lossy().into_owned()
}

fn check(golden: &Value, section: &str, key: &str, member: &str, actual: Value) {
    assert_close(
        &actual,
        expected(golden, section, key, member),
        &format!("{key}.{member}"),
    );
}

/// A model's description with its uid taken out (`uid=X,` or `uid = X,`): the uid is the
/// instance's, not the server's.
fn without_uid(text: &str) -> String {
    let start = text.find("uid").expect("a uid");
    let end = start + text[start..].find(',').expect("the uid's end");
    format!("{}uid*{}", &text[..start], &text[end..])
}

/// The values of `name` of `frame`, in the order of its `id`.
fn column(frame: DataFrame, name: &str) -> Vec<f64> {
    frame
        .order_by(vec![col("id").asc().expression().clone()])
        .select([col(name)])
        .collect()
        .unwrap()
        .into_iter()
        .map(|row| row.get(0).unwrap().as_f64().unwrap())
        .collect()
}

/// A row as numbers, in order: each number of the row, and the numbers of each list.
fn numbers(row: &Row) -> Vec<f64> {
    fn push(value: &RowValue, out: &mut Vec<f64>) {
        match value {
            RowValue::Double(v) => out.push(*v),
            RowValue::Float(v) => out.push(f64::from(*v)),
            RowValue::Integer(v) => out.push(f64::from(*v)),
            RowValue::Long(v) => out.push(*v as f64),
            RowValue::List(items) => items.iter().for_each(|item| push(item, out)),
            other => panic!("not a number: {other:?}"),
        }
    }
    let mut out = Vec::new();
    row.values().iter().for_each(|value| push(value, &mut out));
    out
}

/// The numbers of a golden JSON value: its numbers, and the values of its vectors.
fn flatten(value: &Value, out: &mut Vec<f64>) {
    match value {
        Value::Number(n) => out.push(n.as_f64().unwrap()),
        Value::Array(items) => items.iter().for_each(|item| flatten(item, out)),
        Value::Object(_) => {
            if let Some(values) = value.get("values") {
                flatten(values, out);
            }
        }
        _ => {}
    }
}

/// Sort rows by their values, rounded to 1e-9, so that rows that agree compare in the same order.
fn sort_rows(rows: &mut [Vec<f64>]) {
    rows.sort_by_key(|row| {
        row.iter()
            .map(|x| (x * 1e9).round() as i64)
            .collect::<Vec<_>>()
    });
}

/// The rows of `frame` as sorted numbers.
fn sorted_rows(frame: DataFrame) -> Vec<Vec<f64>> {
    let mut rows: Vec<Vec<f64>> = frame.collect().unwrap().iter().map(numbers).collect();
    sort_rows(&mut rows);
    rows
}

/// The rows of a golden DataFrame (`{rows, schema}`) as sorted numbers, of its cells at `keep`
/// (all of them when None).
fn golden_rows(frame: &Value, keep: Option<&[usize]>) -> Vec<Vec<f64>> {
    let mut rows: Vec<Vec<f64>> = frame["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            let mut out = Vec::new();
            for (i, cell) in row.as_array().unwrap().iter().enumerate() {
                if keep.is_none_or(|keep| keep.contains(&i)) {
                    flatten(cell, &mut out);
                }
            }
            out
        })
        .collect();
    sort_rows(&mut rows);
    rows
}

/// The rows of a training or evaluation's predictions, as the golden's (label, weight, raw
/// prediction, probability, prediction; the features are not compared).
fn predictions_rows(frame: DataFrame) -> Vec<Vec<f64>> {
    sorted_rows(frame.select([
        col("label"),
        col("weight"),
        Column::new(Expression::function(
            "vector_to_array",
            vec![
                col("rawPrediction").expression().clone(),
                lit_string("float64").expression().clone(),
            ],
        )),
        Column::new(Expression::function(
            "vector_to_array",
            vec![
                col("probability").expression().clone(),
                lit_string("float64").expression().clone(),
            ],
        )),
        col("prediction"),
    ]))
}

/// Assert that the rows of `actual` are the rows of `expected`, within 1e-9 relative.
fn assert_rows(actual: Vec<Vec<f64>>, expected: Vec<Vec<f64>>, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: rows");
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(a.len(), e.len(), "{what}[{i}]: width");
        for (x, y) in a.iter().zip(e) {
            assert!(
                (x - y).abs() <= 1e-9 * y.abs().max(1.0),
                "{what}[{i}]: {x} is not {y}"
            );
        }
    }
}

/// The members of a classification summary (`_ClassificationSummary`), against the golden
/// summary `key`, the predictions as rows.
fn check_classification<S: ClassificationSummary>(s: &S, golden: &Value, key: &str) {
    let section = "summaries";
    let c = |member: &str, actual: Value| check(golden, section, key, member, actual);
    c("labelCol", json!(s.label_col().unwrap()));
    c("predictionCol", json!(s.prediction_col().unwrap()));
    c("weightCol", json!(s.weight_col().unwrap()));
    c("labels", json!(s.labels().unwrap()));
    c("accuracy", json!(s.accuracy().unwrap()));
    c("precisionByLabel", json!(s.precision_by_label().unwrap()));
    c("recallByLabel", json!(s.recall_by_label().unwrap()));
    c(
        "truePositiveRateByLabel",
        json!(s.true_positive_rate_by_label().unwrap()),
    );
    c(
        "falsePositiveRateByLabel",
        json!(s.false_positive_rate_by_label().unwrap()),
    );
    c(
        "fMeasureByLabel-0.5",
        json!(s.f_measure_by_label(0.5).unwrap()),
    );
    c(
        "fMeasureByLabel-1.0",
        json!(s.f_measure_by_label(1.0).unwrap()),
    );
    c(
        "weightedFMeasure-0.5",
        json!(s.weighted_f_measure(0.5).unwrap()),
    );
    c(
        "weightedFMeasure-1.0",
        json!(s.weighted_f_measure(1.0).unwrap()),
    );
    c("weightedPrecision", json!(s.weighted_precision().unwrap()));
    c("weightedRecall", json!(s.weighted_recall().unwrap()));
    c(
        "weightedTruePositiveRate",
        json!(s.weighted_true_positive_rate().unwrap()),
    );
    c(
        "weightedFalsePositiveRate",
        json!(s.weighted_false_positive_rate().unwrap()),
    );
    assert_rows(
        predictions_rows(s.predictions().unwrap()),
        golden_rows(
            expected(golden, section, key, "predictions"),
            Some(&[0, 1, 3, 4, 5]),
        ),
        &format!("{key}.predictions"),
    );
}

/// The members of a binary classification summary, against the golden summary `key`.
fn check_binary<S: BinaryClassificationSummary>(s: &S, golden: &Value, key: &str) {
    let section = "summaries";
    check(
        golden,
        section,
        key,
        "scoreCol",
        json!(s.score_col().unwrap()),
    );
    check(
        golden,
        section,
        key,
        "areaUnderROC",
        json!(s.area_under_roc().unwrap()),
    );
    let curves = [
        ("roc", s.roc().unwrap()),
        ("pr", s.pr().unwrap()),
        ("fMeasureByThreshold", s.f_measure_by_threshold().unwrap()),
        ("precisionByThreshold", s.precision_by_threshold().unwrap()),
        ("recallByThreshold", s.recall_by_threshold().unwrap()),
    ];
    for (member, frame) in curves {
        assert_rows(
            sorted_rows(frame),
            golden_rows(expected(golden, section, key, member), None),
            &format!("{key}.{member}"),
        );
    }
}

/// The members of a training summary (`_TrainingSummary`), against the golden summary `key`.
fn check_training<S: TrainingSummary>(s: &S, golden: &Value, key: &str) {
    let section = "summaries";
    check(
        golden,
        section,
        key,
        "objectiveHistory",
        json!(s.objective_history().unwrap()),
    );
    check(
        golden,
        section,
        key,
        "totalIterations",
        json!(s.total_iterations().unwrap()),
    );
}

#[test]
fn decision_tree_model_matches_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let g = golden();
    let mut estimator = DecisionTreeClassifier::new()
        .set_max_depth(3)
        .set_seed(42)
        .set_weight_col("weight");
    let model = estimator.fit(&dataset(&s, &g, "multi_train")).unwrap();
    let key = "DecisionTreeClassifier";
    let c = |member: &str, actual: Value| check(&g, "models", key, member, actual);
    c("depth", json!(model.depth().unwrap()));
    c("numNodes", json!(model.num_nodes().unwrap()));
    c("numClasses", json!(model.num_classes().unwrap()));
    c("numFeatures", json!(model.num_features().unwrap()));
    c(
        "featureImportances",
        vector_json(&model.feature_importances().unwrap()),
    );
    assert_close(
        &json!(without_uid(&model.to_debug_string().unwrap())),
        &json!(without_uid(
            expected(&g, "models", key, "toDebugString")
                .as_str()
                .unwrap()
        )),
        "DecisionTreeClassifier.toDebugString",
    );
    c("supportedImpurities", json!(model.supported_impurities()));
    for (name, features) in probes() {
        c(
            &format!("predict-{name}"),
            json!(model.predict(&features).unwrap()),
        );
        c(
            &format!("predictLeaf-{name}"),
            json!(model.predict_leaf(&features).unwrap()),
        );
        c(
            &format!("predictRaw-{name}"),
            vector_json(&model.predict_raw(&features).unwrap()),
        );
        c(
            &format!("predictProbability-{name}"),
            vector_json(&model.predict_probability(&features).unwrap()),
        );
    }
}

#[test]
fn decision_tree_model_and_estimator_round_trip_through_save_and_load() {
    let Some(s) = live_session() else {
        return;
    };
    let g = golden();
    let estimator = DecisionTreeClassifier::new()
        .set_max_depth(3)
        .set_seed(42)
        .set_weight_col("weight");
    let estimator_path = scratch_path("dt-estimator");
    estimator
        .write()
        .session(&s)
        .overwrite()
        .save(&estimator_path)
        .unwrap();
    let loaded = DecisionTreeClassifier::load(&s, &estimator_path).unwrap();
    let mut before: Vec<String> = estimator
        .params()
        .to_proto()
        .params
        .keys()
        .cloned()
        .collect();
    let mut after: Vec<String> = loaded.params().to_proto().params.keys().cloned().collect();
    before.sort();
    after.sort();
    assert_eq!(before, after, "the estimator's params survive the save");

    let model_path = scratch_path("dt-model");
    let mut fitted = estimator.clone();
    let model = fitted.fit(&dataset(&s, &g, "multi_train")).unwrap();
    model
        .write()
        .unwrap()
        .overwrite()
        .save(&model_path)
        .unwrap();
    let loaded = DecisionTreeClassificationModel::load(&s, &model_path).unwrap();
    let key = "DecisionTreeClassifier";
    check(&g, "models", key, "depth", json!(loaded.depth().unwrap()));
    check(
        &g,
        "models",
        key,
        "predict-dense",
        json!(loaded.predict(&MlVector::Dense(vec![0.5, 1.5])).unwrap()),
    );
}

/// A random forest of three trees with the golden's seed and weight column, and the golden's
/// default feature subset, "auto". Its training dataset is the golden's one partition.
fn forest() -> RandomForestClassifier {
    RandomForestClassifier::new()
        .set_num_trees(3)
        .set_max_depth(3)
        .set_seed(42)
        .set_weight_col("weight")
}

#[test]
fn random_forest_binary_model_matches_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let g = golden();
    let eval = dataset(&s, &g, "binary_eval");
    let mut model = forest().fit(&dataset(&s, &g, "binary_train")).unwrap();
    let key = "RandomForestClassifier.binary";
    let c = |member: &str, actual: Value| check(&g, "models", key, member, actual);
    assert!(model.has_summary());
    c("numClasses", json!(model.num_classes().unwrap()));
    c("numFeatures", json!(model.num_features().unwrap()));
    c("getNumTrees", json!(model.get_num_trees().unwrap()));
    c("totalNumNodes", json!(model.total_num_nodes().unwrap()));
    c("treeWeights", json!(model.tree_weights().unwrap()));
    c(
        "featureImportances",
        vector_json(&model.feature_importances().unwrap()),
    );
    assert_close(
        &json!(without_uid(&model.to_debug_string().unwrap())),
        &json!(without_uid(
            expected(&g, "models", key, "toDebugString")
                .as_str()
                .unwrap()
        )),
        "RandomForestClassifier.binary.toDebugString",
    );
    c(
        "supportedFeatureSubsetStrategies",
        json!(model.supported_feature_subset_strategies()),
    );
    c("supportedImpurities", json!(model.supported_impurities()));
    for (name, features) in probes() {
        c(
            &format!("predict-{name}"),
            json!(model.predict(&features).unwrap()),
        );
        c(
            &format!("predictLeaf-{name}"),
            vector_json(&model.predict_leaf(&features).unwrap()),
        );
        c(
            &format!("predictRaw-{name}"),
            vector_json(&model.predict_raw(&features).unwrap()),
        );
        c(
            &format!("predictProbability-{name}"),
            vector_json(&model.predict_probability(&features).unwrap()),
        );
    }
    // The transform of the eval set predicts what predict does on each of its rows.
    let predicted = column(model.transform(&eval).unwrap(), "prediction");
    for (i, features) in features_of(&g, "binary_eval").iter().enumerate() {
        assert_eq!(model.predict(features).unwrap(), predicted[i], "row {i}");
    }

    // The first of the three trees is a decision tree model of its own.
    let trees = model.trees().unwrap();
    assert_eq!(
        trees.len(),
        expected(&g, "models", key, "trees")
            .as_array()
            .unwrap()
            .len()
    );
    let tree_key = "RandomForestClassifier.binary.trees[0]";
    let t = |member: &str, actual: Value| check(&g, "models", tree_key, member, actual);
    t("depth", json!(trees[0].depth().unwrap()));
    t("numNodes", json!(trees[0].num_nodes().unwrap()));
    t("numClasses", json!(trees[0].num_classes().unwrap()));
    t("numFeatures", json!(trees[0].num_features().unwrap()));
    t(
        "featureImportances",
        vector_json(&trees[0].feature_importances().unwrap()),
    );
    for (name, features) in probes() {
        t(
            &format!("predict-{name}"),
            json!(trees[0].predict(&features).unwrap()),
        );
    }

    let summary = match model.summary().unwrap() {
        RandomForestTrainingSummary::Binary(summary) => summary,
        other => panic!("two classes give a binary summary, got {other:?}"),
    };
    let training = "RandomForestClassifier.binary.training";
    check_classification(&summary, &g, training);
    check_binary(&summary, &g, training);
    check_training(&summary, &g, training);
    assert_eq!(
        summary.summary_target().id(),
        format!("{}.summary", summary.summary_target().root().unwrap().id())
    );

    let evaluated = match model.evaluate(&eval).unwrap() {
        RandomForestSummary::Binary(summary) => summary,
        other => panic!("two classes give a binary summary, got {other:?}"),
    };
    let evaluation = "RandomForestClassifier.binary.evaluate";
    check_classification(&evaluated, &g, evaluation);
    check_binary(&evaluated, &g, evaluation);
}

#[test]
fn random_forest_multiclass_model_matches_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let g = golden();
    let model = forest().fit(&dataset(&s, &g, "multi_train")).unwrap();
    let key = "RandomForestClassifier.multiclass";
    let c = |member: &str, actual: Value| check(&g, "models", key, member, actual);
    c("numClasses", json!(model.num_classes().unwrap()));
    c("numFeatures", json!(model.num_features().unwrap()));
    c("getNumTrees", json!(model.get_num_trees().unwrap()));
    c("totalNumNodes", json!(model.total_num_nodes().unwrap()));
    c("treeWeights", json!(model.tree_weights().unwrap()));
    c(
        "featureImportances",
        vector_json(&model.feature_importances().unwrap()),
    );
    assert_close(
        &json!(without_uid(&model.to_debug_string().unwrap())),
        &json!(without_uid(
            expected(&g, "models", key, "toDebugString")
                .as_str()
                .unwrap()
        )),
        "RandomForestClassifier.multiclass.toDebugString",
    );
    for (name, features) in probes() {
        c(
            &format!("predict-{name}"),
            json!(model.predict(&features).unwrap()),
        );
        c(
            &format!("predictLeaf-{name}"),
            vector_json(&model.predict_leaf(&features).unwrap()),
        );
        c(
            &format!("predictRaw-{name}"),
            vector_json(&model.predict_raw(&features).unwrap()),
        );
        c(
            &format!("predictProbability-{name}"),
            vector_json(&model.predict_probability(&features).unwrap()),
        );
    }

    let summary = match model.summary().unwrap() {
        RandomForestTrainingSummary::Multiclass(summary) => summary,
        other => panic!("three classes give a multiclass summary, got {other:?}"),
    };
    let training = "RandomForestClassifier.multiclass.training";
    check_classification(&summary, &g, training);
    check_training(&summary, &g, training);

    let evaluated = match model.evaluate(&dataset(&s, &g, "multi_eval")).unwrap() {
        RandomForestSummary::Multiclass(summary) => summary,
        other => panic!("three classes give a multiclass summary, got {other:?}"),
    };
    check_classification(&evaluated, &g, "RandomForestClassifier.multiclass.evaluate");
}

#[test]
fn random_forest_model_loads_without_a_training_summary() {
    let Some(s) = live_session() else {
        return;
    };
    let g = golden();
    let model = forest().fit(&dataset(&s, &g, "binary_train")).unwrap();
    let path = scratch_path("rf-model");
    model.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = RandomForestClassificationModel::load(&s, &path).unwrap();
    assert!(!loaded.has_summary());
    assert!(loaded.summary().is_err());
    assert_close(
        &json!(loaded.predict(&MlVector::Dense(vec![0.5, 1.5])).unwrap()),
        expected(
            &g,
            "models",
            "RandomForestClassifier.binary",
            "predict-dense",
        ),
        "loaded predict",
    );
}

/// A gradient-boosted classifier of three iterations, with the golden's depth, seed, step size,
/// loss and weight column. Boosting has no sampling, so the model is the golden's on any server.
fn boosted() -> GBTClassifier {
    GBTClassifier::new()
        .set_max_depth(2)
        .set_max_iter(3)
        .set_seed(42)
        .set_step_size(0.1)
        .set_loss_type("logistic")
        .set_feature_subset_strategy("all")
        .set_weight_col("weight")
}

#[test]
fn gradient_boosted_model_matches_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let g = golden();
    let eval = dataset(&s, &g, "binary_eval");
    let model = boosted().fit(&dataset(&s, &g, "binary_train")).unwrap();
    let key = "GBTClassifier";
    let c = |member: &str, actual: Value| check(&g, "models", key, member, actual);
    c("numClasses", json!(model.num_classes().unwrap()));
    c("numFeatures", json!(model.num_features().unwrap()));
    c("getNumTrees", json!(model.get_num_trees().unwrap()));
    c("totalNumNodes", json!(model.total_num_nodes().unwrap()));
    c("treeWeights", json!(model.tree_weights().unwrap()));
    c(
        "featureImportances",
        vector_json(&model.feature_importances().unwrap()),
    );
    assert_close(
        &json!(without_uid(&model.to_debug_string().unwrap())),
        &json!(without_uid(
            expected(&g, "models", key, "toDebugString")
                .as_str()
                .unwrap()
        )),
        "GBTClassifier.toDebugString",
    );
    c(
        "supportedFeatureSubsetStrategies",
        json!(model.supported_feature_subset_strategies()),
    );
    c("supportedImpurities", json!(model.supported_impurities()));
    c("supportedLossTypes", json!(model.supported_loss_types()));
    c(
        "evaluateEachIteration",
        json!(model.evaluate_each_iteration(&eval).unwrap()),
    );
    for (name, features) in probes() {
        c(
            &format!("predict-{name}"),
            json!(model.predict(&features).unwrap()),
        );
        c(
            &format!("predictLeaf-{name}"),
            vector_json(&model.predict_leaf(&features).unwrap()),
        );
        c(
            &format!("predictRaw-{name}"),
            vector_json(&model.predict_raw(&features).unwrap()),
        );
        c(
            &format!("predictProbability-{name}"),
            vector_json(&model.predict_probability(&features).unwrap()),
        );
    }

    // The first tree is a regression tree of its own.
    let trees = model.trees().unwrap();
    assert_eq!(trees.len(), 3);
    let tree_key = "GBTClassifier.trees[0]";
    let t = |member: &str, actual: Value| check(&g, "models", tree_key, member, actual);
    t("depth", json!(trees[0].depth().unwrap()));
    t("numNodes", json!(trees[0].num_nodes().unwrap()));
    t("numFeatures", json!(trees[0].num_features().unwrap()));
    t(
        "featureImportances",
        vector_json(&trees[0].feature_importances().unwrap()),
    );
    for (name, features) in probes() {
        t(
            &format!("predict-{name}"),
            json!(trees[0].predict(&features).unwrap()),
        );
    }
}

#[test]
fn gradient_boosted_model_round_trips_through_save_and_load() {
    let Some(s) = live_session() else {
        return;
    };
    let g = golden();
    let model = boosted().fit(&dataset(&s, &g, "binary_train")).unwrap();
    let path = scratch_path("gbt-model");
    model.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = GBTClassificationModel::load(&s, &path).unwrap();
    check(
        &g,
        "models",
        "GBTClassifier",
        "predict-dense",
        json!(loaded.predict(&MlVector::Dense(vec![0.5, 1.5])).unwrap()),
    );
    assert_eq!(loaded.trees().unwrap().len(), 3);
}
