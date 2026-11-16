// SPDX-License-Identifier: Apache-2.0
//! The meta-algorithms of Spark ML over Spark Connect against a live server, as PySpark 4.3.0 runs
//! them: the validators nested in a grid, which set their own params as PySpark's `copy(extra)`
//! does, and the pipelines that PySpark saved, which this client loads, fits and transforms (the
//! models' classes and params). The expected values are PySpark 4.3.0's on the same server. Set
//! SPARK_REMOTE to run against a live Spark Connect server, which reads the fixtures of
//! `tests/fixtures/ml_meta` from this host.
mod common;

use common::live_session;
use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::ml::{
    read_operator, BinaryClassificationEvaluator, CrossValidator, Estimator, LinearSVC, MlValue,
    OperatorEstimator, OperatorType, ParamGridBuilder, Pipeline, PipelineModel, PipelineStage,
    RegressionEvaluator, TrainValidationSplit,
};
use spark_connect::row::Row;
use spark_connect::{Column, DataFrame, SparkSession};

/// The fixture pipeline `name` of `tests/fixtures/ml_meta`, saved by PySpark 4.3.0 before a fit.
fn fixture(name: &str) -> String {
    format!(
        "{}/tests/fixtures/ml_meta/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// The training data of PySpark's probes: a features vector of two numbers and a label.
fn training(s: &SparkSession) -> DataFrame {
    s.sql(
        "SELECT array(CAST(id % 7 AS DOUBLE), CAST(id % 3 AS DOUBLE)) AS xs, \
         CAST(id % 2 AS DOUBLE) AS label FROM range(40)",
    )
    .unwrap()
    .with_column(
        "features",
        Column::new(Expression::function(
            "array_to_vector",
            vec![col("xs").expression().clone()],
        )),
    )
    .coalesce(1)
}

/// The ratings of PySpark's ALS probe.
fn ratings(s: &SparkSession) -> DataFrame {
    s.sql(
        "SELECT CAST(id % 5 AS INT) AS user, CAST(id % 7 AS INT) AS item, \
         CAST((id * 3) % 5 AS FLOAT) AS rating FROM range(60)",
    )
    .unwrap()
    .coalesce(1)
}

fn assert_all_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= tolerance, "{actual:?} vs {expected:?}");
    }
}

/// The values of the column `name` of the rows, as doubles.
fn column(rows: &[Row], name: &str) -> Vec<f64> {
    rows.iter()
        .map(|row| {
            row.get_by_name(name)
                .and_then(|v| v.as_f64())
                .unwrap_or_else(|| panic!("no double of {name} in {row:?}"))
        })
        .collect()
}

/// Validators nested in a grid of an outer cross validator: an inner validator's own param (its
/// numFolds, or its trainRatio) that the grid sets, as PySpark's `copy(extra)` sets it.
#[test]
fn validators_nested_in_a_grid_set_their_own_params_as_pyspark_does() {
    let Some(s) = live_session() else {
        return;
    };
    let df = training(&s);
    let svc = LinearSVC::new().set_max_iter(3);
    let svc_uid = svc.operator().uid.clone();
    let svc_grid = ParamGridBuilder::new()
        .add_grid(
            &svc_uid,
            "regParam",
            [MlValue::Double(0.1), MlValue::Double(1.0)],
        )
        .build();
    let evaluator = BinaryClassificationEvaluator::new();

    // The numFolds of a cross validator, set by the grid of an outer one, in a pipeline.
    let inner = CrossValidator::new()
        .set_estimator(svc.clone())
        .set_estimator_param_maps(svc_grid.clone())
        .set_evaluator(evaluator.clone())
        .set_num_folds(2)
        .set_seed(7)
        .set_collect_sub_models(true);
    let inner_uid = inner.uid().to_string();
    let mut outer = CrossValidator::new()
        .set_estimator(Pipeline::new().set_stages(vec![PipelineStage::estimator(inner)]))
        .set_estimator_param_maps(
            ParamGridBuilder::new()
                .add_grid(&inner_uid, "numFolds", [MlValue::Int(2), MlValue::Int(4)])
                .build(),
        )
        .set_evaluator(evaluator.clone())
        .set_num_folds(2)
        .set_seed(3);
    let model = outer.fit_validator(&df).unwrap();
    assert_all_close(
        model.avg_metrics(),
        &[0.4245580808080808, 0.42077020202020204],
        1e-12,
    );

    // The trainRatio of a train validation split, set by the grid of an outer cross validator.
    let inner = TrainValidationSplit::new()
        .set_estimator(svc)
        .set_estimator_param_maps(svc_grid)
        .set_evaluator(evaluator.clone())
        .set_seed(7)
        .set_collect_sub_models(true);
    let inner_uid = inner.uid().to_string();
    let mut outer = CrossValidator::new()
        .set_estimator(Pipeline::new().set_stages(vec![PipelineStage::estimator(inner)]))
        .set_estimator_param_maps(
            ParamGridBuilder::new()
                .add_grid(
                    &inner_uid,
                    "trainRatio",
                    [MlValue::Double(0.3), MlValue::Double(0.9)],
                )
                .build(),
        )
        .set_evaluator(evaluator)
        .set_num_folds(2)
        .set_seed(3)
        .set_collect_sub_models(true);
    let model = outer.fit_validator(&df).unwrap();
    assert_all_close(
        model.avg_metrics(),
        &[0.4384469696969697, 0.43465909090909094],
        1e-12,
    );
}

/// A pipeline of ALS that PySpark saved: the fitted model keeps the ALS params that its model has,
/// and its transform sends them (PySpark's `Param rank does not exist.` otherwise).
#[test]
fn a_loaded_als_pipeline_fits_and_transforms_as_pyspark_does() {
    let Some(s) = live_session() else {
        return;
    };
    let mut pipeline = Pipeline::load(&s, &fixture("als")).unwrap();
    let mut model = pipeline.fit(&ratings(&s)).unwrap();
    let rows = model.transform(&ratings(&s)).unwrap().take(4).unwrap();
    let predictions: Vec<f64> = column(&rows, "prediction")
        .iter()
        .map(|p| (p * 1e4).round() / 1e4)
        .collect();
    assert_all_close(&predictions, &[2.8547, 3.7921, 0.9552, 0.0], 1e-9);
}

/// A pipeline of QuantileDiscretizer that PySpark saved: its fitted model is a Bucketizer, which
/// has none of the discretizer's params (numBuckets), and its transform gives PySpark's buckets.
/// The Bucketizer is a stage of a new uid, as PySpark's fit gives it, not the discretizer's.
#[test]
fn a_loaded_quantile_discretizer_pipeline_fits_as_pyspark_does() {
    let Some(s) = live_session() else {
        return;
    };
    let data = s
        .sql("SELECT CAST(id AS DOUBLE) AS x FROM range(40)")
        .unwrap()
        .coalesce(1);
    let mut pipeline = Pipeline::load(&s, &fixture("qd")).unwrap();
    let discretizer_uid = pipeline.stages()[0].uid().to_string();
    let mut model = pipeline.fit(&data).unwrap();
    let pipeline_model = (*model)
        .as_any()
        .downcast_ref::<PipelineModel>()
        .expect("a pipeline model");
    let PipelineStage::Transformer(bucketizer) = &pipeline_model.stages()[0] else {
        panic!("a Bucketizer is a transformer stage");
    };
    assert_eq!(
        bucketizer.operator().name,
        "org.apache.spark.ml.feature.Bucketizer"
    );
    assert_ne!(bucketizer.operator().uid, discretizer_uid);
    let rows = model.transform(&data).unwrap().collect().unwrap();
    assert_all_close(&column(&rows, "qx"), &EXPECTED_BUCKETS, 0.0);
}

/// A cross validator that the JVM writer saved (its class names are the JVM's, as classic PySpark
/// or Scala write them) over a pipeline of LinearSVC: it loads, as PySpark loads it, and fits to
/// PySpark's metrics.
#[test]
fn a_validator_saved_by_the_jvm_writer_with_a_pipeline_loads_as_pyspark_does() {
    let Some(s) = live_session() else {
        return;
    };
    let cv = CrossValidator::load(&s, &fixture("jvm_cv")).unwrap();
    let pipeline = cv
        .estimator()
        .and_then(|e| e.as_any_stage())
        .and_then(|any| any.downcast_ref::<Pipeline>())
        .expect("a pipeline");
    assert_eq!(pipeline.operator().name, "org.apache.spark.ml.Pipeline");
    let svc_uid = pipeline.stages()[0].uid().to_string();
    let regs: Vec<Option<&MlValue>> = cv
        .estimator_param_maps()
        .iter()
        .map(|map| map.get(&svc_uid, "regParam"))
        .collect();
    assert_eq!(
        regs,
        vec![Some(&MlValue::Double(1e-5)), Some(&MlValue::Double(0.1))]
    );
    let mut cv = cv;
    let model = cv.fit_validator(&training(&s)).unwrap();
    assert_all_close(
        model.avg_metrics(),
        &[0.43465909090909094, 0.43465909090909094],
        1e-12,
    );
}

/// A cross validator whose pipeline holds one stage uid twice, as a load of PySpark's save with a
/// duplicate uid: PySpark refuses it, with its `getUidMap` message of the pipeline's class.
#[test]
fn a_validator_whose_pipeline_repeats_a_uid_fails_to_load_as_pyspark_does() {
    let Some(s) = live_session() else {
        return;
    };
    let error = CrossValidator::load(&s, &fixture("dup_cv"))
        .expect_err("a duplicate uid")
        .to_string();
    assert!(
        error.contains(
            "pyspark.ml.pipeline.Pipeline.load found a compound estimator with stages with \
             duplicate UIDs. List of UIDs: ['Pipeline_d2597241192d', 'LinearSVC_ce605c8a54d3']."
        ),
        "{error}"
    );
}

/// A pipeline of MinHashLSH that PySpark saved: its seed and number of hash tables are params of
/// its model too (PySpark's `Param seed does not exist.` otherwise), and its hashes are PySpark's.
#[test]
fn a_loaded_minhash_lsh_pipeline_fits_and_transforms_as_pyspark_does() {
    let Some(s) = live_session() else {
        return;
    };
    let data = s
        .sql(
            "SELECT array(CAST(id % 5 + 1 AS DOUBLE), CAST(id % 3 + 1 AS DOUBLE)) AS xs \
             FROM range(20)",
        )
        .unwrap()
        .coalesce(1)
        .with_column(
            "vec",
            Column::new(Expression::function(
                "array_to_vector",
                vec![col("xs").expression().clone()],
            )),
        );
    let mut pipeline = Pipeline::load(&s, &fixture("lsh")).unwrap();
    let mut model = pipeline.fit(&data).unwrap();
    let rows = model.transform(&data).unwrap().take(3).unwrap();
    for row in &rows {
        // One vector of a value per hash table: PySpark's values of the first three rows.
        let hashes = format!("{:?}", row.get_by_name("h"));
        for hash in [1316787817.0_f64, 598179266.0] {
            assert!(hashes.contains(&format!("Double({hash:?})")), "{hashes}");
        }
    }
}

/// A pipeline of DecisionTreeClassifier that PySpark saved: its fitted model is of the JVM class
/// that PySpark's `_create_model` names, DecisionTreeClassificationModel, and gives PySpark's
/// predictions.
#[test]
fn a_loaded_decision_tree_pipeline_fits_as_its_pyspark_model_class() {
    let Some(s) = live_session() else {
        return;
    };
    let data = training(&s);
    let mut pipeline = Pipeline::load(&s, &fixture("pipe_dt")).unwrap();
    let mut model = pipeline.fit(&data).unwrap();
    // The model itself, not the box of it: `as_any` of the box would be the box's type.
    let pipeline_model = (*model)
        .as_any()
        .downcast_ref::<PipelineModel>()
        .expect("a pipeline model");
    let PipelineStage::Model(tree) = &pipeline_model.stages()[0] else {
        panic!("a model stage");
    };
    assert_eq!(
        tree.operator().name,
        "org.apache.spark.ml.classification.DecisionTreeClassificationModel"
    );
    let rows = model.transform(&data).unwrap().collect().unwrap();
    assert_all_close(&column(&rows, "prediction"), &EXPECTED_TREE, 0.0);
}

/// A cross validator over a loaded ALS estimator: its grid sets the regParam of the loaded stage,
/// and its evaluation transforms by the ALS models, which have the ALS params that they have.
#[test]
fn a_cross_validator_over_a_loaded_als_estimator_evaluates_as_pyspark_does() {
    let Some(s) = live_session() else {
        return;
    };
    let (operator, params) = read_operator(
        &s,
        "org.apache.spark.ml.recommendation.ALS",
        OperatorType::Estimator,
        &fixture("als/stages/0_ALS_dac8df2d4b65"),
    )
    .unwrap();
    let uid = operator.uid.clone();
    let als = OperatorEstimator::new(operator, params);
    let evaluator = RegressionEvaluator::new().set_label_col("rating");
    let mut cv = CrossValidator::new()
        .set_estimator(als)
        .set_estimator_param_maps(
            ParamGridBuilder::new()
                .add_grid(
                    &uid,
                    "regParam",
                    [MlValue::Double(0.1), MlValue::Double(0.5)],
                )
                .build(),
        )
        .set_evaluator(evaluator)
        .set_num_folds(2)
        .set_seed(3);
    let model = cv.fit_validator(&ratings(&s)).unwrap();
    // ALS computes in floats, so the metrics of two servers (a local one and CI's Linux one)
    // differ in their eighth digit: 0.6690446412218276 locally, 0.6690446591958767 on CI.
    assert_all_close(
        model.avg_metrics(),
        &[0.6690446412218276, 0.8557945260815127],
        1e-6,
    );
}

/// The buckets of PySpark's QuantileDiscretizer(numBuckets=3) of the ids 0 to 39.
const EXPECTED_BUCKETS: [f64; 40] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0,
];

/// The predictions of PySpark's DecisionTreeClassifier(maxDepth=2) of the training data.
const EXPECTED_TREE: [f64; 40] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0,
];
