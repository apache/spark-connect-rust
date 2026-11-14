//! Live native regression tests. Set SPARK_REMOTE to run against Spark Connect.
mod common;

use common::live_session;
use spark_connect::column::{col, lit, lit_double, lit_string};
use spark_connect::expression::Expression;
use spark_connect::ml::{
    AFTSurvivalRegression, Estimator, FMRegressor, IsotonicRegression, Transformer,
};
use spark_connect::{Column, DataFrame};

fn vectorize(df: DataFrame) -> DataFrame {
    df.with_column(
        "input",
        Column::new(Expression::function(
            "array_to_vector",
            vec![col("xs").expression().clone()],
        )),
    )
}

fn predictions(df: DataFrame) -> Vec<f64> {
    df.order_by(vec![col("id").asc().expression().clone()])
        .select([col("pred")])
        .collect()
        .unwrap()
        .into_iter()
        .map(|r| r.get(0).unwrap().as_f64().unwrap())
        .collect()
}

fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= tolerance * e.abs().max(1.0),
            "expected {e}, got {a}"
        );
    }
}

#[test]
fn aft_matches_official_censoring_quantiles_and_intercept() {
    let Some(s) = live_session() else {
        return;
    };
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/golden/remaining_regressors.json"
    ))
    .unwrap();
    for case in fixture["aft"].as_array().unwrap() {
        let rows = fixture["aft_rows"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(id, row)| {
                let censor = if case["censored"].as_bool().unwrap() {
                    row[1].as_f64().unwrap()
                } else {
                    1.0
                };
                format!(
                    "({id}, {}D, {censor}D, {}D, {}D)",
                    row[0].as_f64().unwrap(),
                    row[2][0].as_f64().unwrap(),
                    row[2][1].as_f64().unwrap()
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let train = vectorize(s.sql(&format!("SELECT id, duration, observed, array(x,y) AS xs FROM VALUES {rows} AS t(id,duration,observed,x,y)")).unwrap());
        let mut estimator = AFTSurvivalRegression::new()
            .set_features_col("input")
            .set_label_col("duration")
            .set_prediction_col("pred")
            .set_censor_col("observed")
            .set_fit_intercept(case["options"]["fitIntercept"].as_bool().unwrap())
            .set_max_iter(100)
            .set_tol(1e-8)
            .set_quantile_probabilities([0.2, 0.5, 0.8])
            .set_quantiles_col("quantiles")
            .set_aggregation_depth(2)
            .set_max_block_size_in_mb(1.0);
        let mut model = estimator.fit(&train).unwrap();
        let quantiles = Expression::function(
            "vector_to_array",
            vec![
                col("quantiles").expression().clone(),
                lit_string("float64").expression().clone(),
            ],
        );
        let mut cols = vec![col("pred")];
        for i in 0..3 {
            cols.push(Column::new(Expression::function(
                "get",
                vec![quantiles.clone(), lit(i).expression().clone()],
            )));
        }
        let lazy = model
            .transform(&train)
            .unwrap()
            .order_by(vec![col("id").asc().expression().clone()])
            .select(cols);
        drop(model);
        let actual = lazy.collect().unwrap();
        assert_eq!(actual.len(), case["predictions"].as_array().unwrap().len());
        for (i, row) in actual.iter().enumerate() {
            let expected = [
                case["predictions"][i].as_f64().unwrap(),
                case["quantiles"][i][0].as_f64().unwrap(),
                case["quantiles"][i][1].as_f64().unwrap(),
                case["quantiles"][i][2].as_f64().unwrap(),
            ];
            let got = (0..4)
                .map(|j| row.get(j).unwrap().as_f64().unwrap())
                .collect::<Vec<_>>();
            close(&got, &expected, 1e-6);
        }
    }
}

#[test]
fn isotonic_interpolates_weighted_scalar_and_vector_features_in_both_directions() {
    let Some(s) = live_session() else {
        return;
    };
    for isotonic in [true, false] {
        for vector in [true, false] {
            let sign = if isotonic { 1.0 } else { -1.0 };
            let train = s.sql("SELECT x AS input, array(99.0D,x) AS xs, y AS target, w AS weight FROM VALUES (0.0D,0.0D,1.0D),(1.0D,3.0D,1.0D),(2.0D,1.0D,3.0D),(3.0D,4.0D,1.0D) AS t(x,y,w)").unwrap().with_column("target", col("target") * lit_double(sign));
            let held = s.sql("SELECT id, x AS input, array(99.0D,x) AS xs FROM VALUES (0,-1.0D),(1,0.5D),(2,1.5D),(3,2.5D),(4,4.0D) AS t(id,x)").unwrap();
            let (train, held) = if vector {
                (vectorize(train), vectorize(held))
            } else {
                (train, held)
            };
            let mut estimator = IsotonicRegression::new()
                .set_features_col("input")
                .set_label_col("target")
                .set_prediction_col("pred")
                .set_weight_col("weight")
                .set_isotonic(isotonic)
                .set_feature_index(if vector { 1 } else { 0 });
            let mut model = estimator.fit(&train).unwrap();
            let lazy = model.transform(&held).unwrap();
            drop(model);
            close(
                &predictions(lazy),
                &[0.0, 0.75 * sign, 1.5 * sign, 2.75 * sign, 4.0 * sign],
                1e-9,
            );
        }
    }
}

#[test]
fn fm_regressor_learns_interactions_with_both_solvers_and_owns_each_fitted_model() {
    let Some(s) = live_session() else {
        return;
    };
    let train = vectorize(s.sql("SELECT array(x,y) AS xs, x*y AS target FROM VALUES (-1.0D),(0.0D),(1.0D) AS a(x) CROSS JOIN VALUES (-1.0D),(0.0D),(1.0D) AS b(y)").unwrap());
    let held = vectorize(s.sql("SELECT id,array(x,y) AS xs FROM VALUES (0,0.5D,-2.0D),(1,-0.5D,-1.0D),(2,0.0D,1.5D) AS t(id,x,y)").unwrap());
    for solver in ["adamW", "gd"] {
        let mut estimator = FMRegressor::new()
            .set_features_col("input")
            .set_label_col("target")
            .set_prediction_col("pred")
            .set_factor_size(2)
            .set_fit_intercept(false)
            .set_fit_linear(false)
            .set_reg_param(0.0)
            .set_mini_batch_fraction(1.0)
            .set_init_std(0.1)
            .set_max_iter(300)
            .set_step_size(if solver == "gd" { 1.0 } else { 0.1 })
            .set_tol(1e-8)
            .set_solver(solver)
            .set_seed(42);
        let mut first = estimator.fit(&train).unwrap();
        let mut second = estimator
            .fit(&train.with_column("target", lit(0) - col("target")))
            .unwrap();
        let lazy = first.transform(&held).unwrap();
        let other = second.transform(&held).unwrap();
        drop(first);
        drop(second);
        close(&predictions(lazy), &[-1.0, 0.5, 0.0], 0.025);
        close(&predictions(other), &[1.0, -0.5, 0.0], 0.025);
    }
}

#[test]
fn new_regressors_reject_invalid_parameters() {
    let Some(s) = live_session() else {
        return;
    };
    let train = vectorize(
        s.sql("SELECT array(1.0D) AS xs, 1.0D AS label, 1.0D AS censor")
            .unwrap(),
    );
    let estimators: Vec<Box<dyn Estimator>> = vec![
        Box::new(AFTSurvivalRegression::new().set_quantile_probabilities([0.0, 1.0])),
        Box::new(AFTSurvivalRegression::new().set_aggregation_depth(1)),
        Box::new(AFTSurvivalRegression::new().set_max_block_size_in_mb(-1.0)),
        Box::new(IsotonicRegression::new().set_feature_index(-1)),
        Box::new(FMRegressor::new().set_factor_size(0)),
        Box::new(FMRegressor::new().set_solver("invalid")),
        Box::new(FMRegressor::new().set_mini_batch_fraction(0.0)),
        Box::new(FMRegressor::new().set_reg_param(-1.0)),
    ];
    let train = train.with_column_renamed("input", "features");
    for mut estimator in estimators {
        assert!(
            estimator.fit(&train).is_err(),
            "{}",
            estimator.operator().name
        );
    }
}
