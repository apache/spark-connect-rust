// SPDX-License-Identifier: Apache-2.0
//! The typed models of the regression estimators: their members, their summaries and their
//! persistence, as PySpark 4.3.0 reads them over Spark Connect. Set SPARK_REMOTE to run against a
//! live Spark Connect server.
mod common;

use common::live_session;
use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::ml::{
    AFTSurvivalRegression, AFTSurvivalRegressionModel, DecisionTreeModel,
    DecisionTreeRegressionModel, DecisionTreeRegressor, Estimator, FMRegressionModel, FMRegressor,
    GBTRegressionModel, GBTRegressor, GeneralizedLinearRegression,
    GeneralizedLinearRegressionModel, GeneralizedLinearRegressionSummaryApi, IsotonicRegression,
    IsotonicRegressionModel, LinearRegression, LinearRegressionModel, LinearRegressionSummaryApi,
    MlMatrix, MlVector, Model, PredictionModel, RandomForestRegressionModel, RandomForestRegressor,
    TrainingSummary, Transformer, TreeEnsembleModel,
};
use spark_connect::{Column, DataFrame, SparkSession};

/// Eight rows of two features, whose label is exactly `2 * x0 - 3 * x1 + 1`.
fn training(s: &SparkSession) -> DataFrame {
    s.sql(
        "SELECT array(CAST(x0 AS DOUBLE), CAST(x1 AS DOUBLE)) AS xs, \
         CAST(2 * x0 - 3 * x1 + 1 AS DOUBLE) AS label, \
         CAST(x0 + 1 AS DOUBLE) AS weight, CAST(x0 % 2 AS DOUBLE) AS censor \
         FROM VALUES (0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (3, 2), (2, 3) AS t(x0, x1)",
    )
    .unwrap()
    .with_column(
        "features",
        Column::new(Expression::function(
            "array_to_vector",
            vec![col("xs").expression().clone()],
        )),
    )
}

/// The same rows, of positive labels, for the survival times of AFT.
fn survival(s: &SparkSession) -> DataFrame {
    training(s).with_column("label", col("weight") + col("label") * col("label"))
}

fn scratch_path(name: &str) -> String {
    std::env::temp_dir()
        .join(format!(
            "spark-connect-rust-ml-regression-{}-{name}",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned()
}

fn dense(vector: &MlVector) -> Vec<f64> {
    match vector {
        MlVector::Dense(values) => values.clone(),
        MlVector::Sparse {
            size,
            indices,
            values,
        } => {
            let mut dense = vec![0.0; *size as usize];
            for (i, v) in indices.iter().zip(values) {
                dense[*i as usize] = *v;
            }
            dense
        }
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "{actual} is not close to {expected}"
    );
}

#[test]
fn linear_regression_models_have_the_members_and_summaries_of_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let train = training(&s);
    let mut estimator = LinearRegression::new().set_solver("normal");
    let model = estimator.fit(&train).unwrap();
    let coefficients = dense(&model.coefficients().unwrap());
    assert_close(coefficients[0], 2.0);
    assert_close(coefficients[1], -3.0);
    assert_close(model.intercept().unwrap(), 1.0);
    assert_close(model.scale().unwrap(), 1.0);
    assert_eq!(model.num_features().unwrap(), 2);
    assert_close(
        model.predict(&MlVector::Dense(vec![1.0, 1.0])).unwrap(),
        0.0,
    );
    assert!(model
        .describe()
        .unwrap()
        .starts_with("LinearRegressionModel"));

    // The training summary of the fit, as PySpark's model.summary of a remote fit.
    assert!(model.has_summary());
    let summary = model.summary().unwrap();
    assert_close(summary.r2().unwrap(), 1.0);
    assert_close(summary.mean_squared_error().unwrap(), 0.0);
    assert_eq!(summary.num_instances().unwrap(), 8);
    assert_eq!(summary.degrees_of_freedom().unwrap(), 5);
    assert_eq!(summary.coefficient_standard_errors().unwrap().len(), 3);
    assert_eq!(summary.p_values().unwrap().len(), 3);
    assert_eq!(summary.t_values().unwrap().len(), 3);
    assert_eq!(summary.label_col().unwrap(), "label");
    assert_eq!(summary.features_col().unwrap(), "features");
    assert_eq!(summary.prediction_col().unwrap(), "prediction");
    assert_eq!(summary.predictions().unwrap().count().unwrap(), 8);
    assert_eq!(summary.residuals().unwrap().count().unwrap(), 8);
    assert!(summary.total_iterations().unwrap() >= 0);
    assert!(!summary.objective_history().unwrap().is_empty());

    // The summary of an evaluation, of its own server-side object.
    let evaluated = model.evaluate(&train).unwrap();
    assert_close(evaluated.r2().unwrap(), 1.0);
    assert_eq!(evaluated.num_instances().unwrap(), 8);
    assert_eq!(evaluated.predictions().unwrap().count().unwrap(), 8);

    // A loaded model has the same members, and no training summary.
    let path = scratch_path("lr-model");
    model.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = LinearRegressionModel::load(&s, &path).unwrap();
    assert_eq!(loaded.operator().uid, model.operator().uid);
    assert_eq!(
        loaded.coefficients().unwrap(),
        model.coefficients().unwrap()
    );
    assert!(!loaded.has_summary());
    assert!(loaded.summary().is_err());

    let path = scratch_path("lr");
    Estimator::write(&estimator)
        .session(&s)
        .overwrite()
        .save(&path)
        .unwrap();
    let loaded = LinearRegression::load(&s, &path).unwrap();
    assert_eq!(
        Estimator::operator(&loaded).uid,
        Estimator::operator(&estimator).uid
    );
    assert_eq!(loaded.solver(), "normal");
    s.stop().unwrap();
}

#[test]
fn generalized_linear_regression_models_have_the_members_and_summaries_of_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let train = training(&s);
    let model = GeneralizedLinearRegression::new()
        .set_family("gaussian")
        .fit(&train)
        .unwrap();
    let coefficients = dense(&model.coefficients().unwrap());
    assert_close(coefficients[0], 2.0);
    assert_close(coefficients[1], -3.0);
    assert_close(model.intercept().unwrap(), 1.0);
    assert_eq!(model.num_features().unwrap(), 2);

    let summary = model.summary().unwrap();
    assert_eq!(summary.num_instances().unwrap(), 8);
    assert_eq!(summary.rank().unwrap(), 3);
    assert_eq!(summary.degrees_of_freedom().unwrap(), 5);
    assert_eq!(summary.residual_degree_of_freedom().unwrap(), 5);
    assert_eq!(summary.residual_degree_of_freedom_null().unwrap(), 7);
    assert_close(summary.deviance().unwrap(), 0.0);
    assert!(summary.null_deviance().unwrap() > 0.0);
    assert!(summary.dispersion().unwrap() >= 0.0);
    assert!(summary.aic().unwrap().is_finite() || summary.aic().unwrap().is_infinite());
    assert_eq!(summary.solver().unwrap(), "irls");
    assert!(summary.num_iterations().unwrap() >= 1);
    assert_eq!(summary.coefficient_standard_errors().unwrap().len(), 3);
    assert_eq!(summary.t_values().unwrap().len(), 3);
    assert_eq!(summary.p_values().unwrap().len(), 3);
    assert_eq!(summary.prediction_col().unwrap(), "prediction");
    for residuals_type in ["deviance", "pearson", "working", "response"] {
        let residuals = summary.residuals(residuals_type).unwrap();
        assert_eq!(residuals.count().unwrap(), 8, "{residuals_type}");
    }
    assert!(summary.describe().unwrap().contains("Coefficients"));

    let evaluated = model.evaluate(&train).unwrap();
    assert_eq!(evaluated.rank().unwrap(), 3);
    assert_eq!(evaluated.predictions().unwrap().count().unwrap(), 8);

    let path = scratch_path("glr-model");
    model.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = GeneralizedLinearRegressionModel::load(&s, &path).unwrap();
    assert_eq!(loaded.intercept().unwrap(), model.intercept().unwrap());
    assert!(!loaded.has_summary());
    s.stop().unwrap();
}

#[test]
fn aft_isotonic_and_fm_models_have_the_members_of_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let features = MlVector::Dense(vec![1.0, 1.0]);
    let aft = AFTSurvivalRegression::new().fit(&survival(&s)).unwrap();
    assert_eq!(dense(&aft.coefficients().unwrap()).len(), 2);
    assert!(aft.intercept().unwrap().is_finite());
    assert!(aft.scale().unwrap() > 0.0);
    assert!(aft.predict(&features).unwrap() > 0.0);
    // The quantiles at the default probabilities of AFTSurvivalRegression.
    let quantiles = dense(&aft.predict_quantiles(&features).unwrap());
    assert_eq!(quantiles.len(), 9);
    assert!(quantiles.windows(2).all(|pair| pair[0] <= pair[1]));
    let path = scratch_path("aft-model");
    aft.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = AFTSurvivalRegressionModel::load(&s, &path).unwrap();
    assert_eq!(loaded.scale().unwrap(), aft.scale().unwrap());

    let isotonic = IsotonicRegression::new()
        .set_feature_index(0)
        .fit(&training(&s))
        .unwrap();
    let boundaries = dense(&isotonic.boundaries().unwrap());
    let predictions = dense(&isotonic.predictions().unwrap());
    assert_eq!(boundaries.len(), predictions.len());
    assert!(predictions.windows(2).all(|pair| pair[0] <= pair[1]));
    // The model reads one feature, of featureIndex, as Spark counts it.
    assert_eq!(isotonic.num_features().unwrap(), 1);
    let predicted = isotonic.predict(1.5).unwrap();
    assert!(predicted >= predictions[0] && predicted <= predictions[predictions.len() - 1]);
    let path = scratch_path("isotonic-model");
    isotonic.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = IsotonicRegressionModel::load(&s, &path).unwrap();
    assert_eq!(loaded.boundaries().unwrap(), isotonic.boundaries().unwrap());

    let fm = FMRegressor::new()
        .set_factor_size(3)
        .set_seed(7)
        .set_max_iter(5)
        .fit(&training(&s))
        .unwrap();
    assert!(fm.intercept().unwrap().is_finite());
    assert_eq!(dense(&fm.linear().unwrap()).len(), 2);
    let MlMatrix::Dense {
        num_rows, num_cols, ..
    } = fm.factors().unwrap()
    else {
        panic!("dense factors")
    };
    assert_eq!((num_rows, num_cols), (2, 3));
    assert!(fm.predict(&features).unwrap().is_finite());
    let path = scratch_path("fm-model");
    fm.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = FMRegressionModel::load(&s, &path).unwrap();
    assert_eq!(loaded.linear().unwrap(), fm.linear().unwrap());
    s.stop().unwrap();
}

#[test]
fn tree_regression_models_have_the_members_of_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let train = training(&s);
    let features = MlVector::Dense(vec![1.0, 2.0]);

    let tree = DecisionTreeRegressor::new()
        .set_max_depth(3)
        .set_seed(1)
        .fit(&train)
        .unwrap();
    assert!(tree.depth().unwrap() >= 1);
    assert!(tree.num_nodes().unwrap() >= 3);
    assert_eq!(dense(&tree.feature_importances().unwrap()).len(), 2);
    assert!(tree.predict_leaf(&features).unwrap() >= 0.0);
    assert!(tree
        .to_debug_string()
        .unwrap()
        .starts_with("DecisionTreeRegressionModel"));
    assert_eq!(tree.num_features().unwrap(), 2);
    let path = scratch_path("dt-model");
    tree.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = DecisionTreeRegressionModel::load(&s, &path).unwrap();
    assert_eq!(loaded.num_nodes().unwrap(), tree.num_nodes().unwrap());

    let forest = RandomForestRegressor::new()
        .set_num_trees(3)
        .set_max_depth(2)
        .set_seed(1)
        .fit(&train)
        .unwrap();
    let trees = forest.trees().unwrap();
    assert_eq!(trees.len(), 3);
    assert_eq!(forest.get_num_trees().unwrap(), 3);
    assert_eq!(forest.tree_weights().unwrap(), vec![1.0; 3]);
    let nodes: i32 = trees.iter().map(|t| t.num_nodes().unwrap()).sum();
    assert_eq!(forest.total_num_nodes().unwrap(), nodes);
    assert_eq!(dense(&forest.predict_leaf(&features).unwrap()).len(), 3);
    // Each tree is a model of its own, which predicts by itself.
    assert!(trees[0].predict(&features).unwrap().is_finite());
    let path = scratch_path("rf-model");
    forest.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = RandomForestRegressionModel::load(&s, &path).unwrap();
    assert_eq!(loaded.total_num_nodes().unwrap(), nodes);

    let boosted = GBTRegressor::new()
        .set_max_iter(3)
        .set_max_depth(2)
        .set_seed(1)
        .fit(&train)
        .unwrap();
    assert_eq!(boosted.trees().unwrap().len(), 3);
    assert_eq!(boosted.tree_weights().unwrap().len(), 3);
    for loss in ["squared", "absolute"] {
        let errors = boosted.evaluate_each_iteration(&train, loss).unwrap();
        assert_eq!(errors.len(), 3, "{loss}");
    }
    let path = scratch_path("gbt-model");
    boosted.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = GBTRegressionModel::load(&s, &path).unwrap();
    assert_eq!(loaded.get_num_trees().unwrap(), 3);
    s.stop().unwrap();
}
