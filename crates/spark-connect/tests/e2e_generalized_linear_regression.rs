//! Numerical tests of native GeneralizedLinearRegression. Set SPARK_REMOTE to use a live server.
mod common;

use common::live_session;
use spark_connect::column::{col, lit};
use spark_connect::expression::Expression;
use spark_connect::ml::{
    Evaluator, GeneralizedLinearRegression, Model, RegressionEvaluator, Transformer,
};
use spark_connect::{Column, DataFrame};

fn vectorize(df: DataFrame, name: &str) -> DataFrame {
    df.with_column(
        name,
        Column::new(Expression::function(
            "array_to_vector",
            vec![col("xs").expression().clone()],
        )),
    )
}

fn predictions(df: DataFrame, name: &str) -> Vec<f64> {
    df.order_by(vec![col("id").asc().expression().clone()])
        .select([col(name)])
        .collect()
        .unwrap()
        .into_iter()
        .map(|r| r.get(0).unwrap().as_f64().unwrap())
        .collect()
}

fn assert_close(actual: Vec<f64>, expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.into_iter().zip(expected) {
        assert!((a - e).abs() < tolerance, "expected {e}, got {a}");
    }
}

#[test]
fn generalized_linear_regression_fits_different_lines_and_keeps_lazy_predictions_alive() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectorize(
        s.sql(
            "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(2*x+3 AS DOUBLE) AS target, 1.0D AS weight
         FROM VALUES (-4), (-3), (-2), (-1), (1), (2), (3), (4) AS t(x)",
        )
        .unwrap(),
        "input",
    );
    let held_out = vectorize(
        s.sql(
            "SELECT id, array(CAST(x AS DOUBLE)) AS xs, CAST(2*x+3 AS DOUBLE) AS target
         FROM VALUES (0, -1.5), (1, 0.5), (2, 4.5) AS t(id, x)",
        )
        .unwrap(),
        "input",
    );
    let mut estimator = GeneralizedLinearRegression::new()
        .set_features_col("input")
        .set_label_col("target")
        .set_prediction_col("pred")
        .set_max_iter(100)
        .set_reg_param(0.0)
        .set_tol(1e-8)
        .set_fit_intercept(true)
        .set_solver("irls")
        .set_weight_col("weight")
        .set_aggregation_depth(2);
    let mut first = estimator.fit(&training).unwrap();
    let mut second = estimator
        .fit(&training.with_column("target", lit(10) - col("target")))
        .unwrap();
    let object_id = |model: &dyn spark_connect::ml::Model| {
        let spark_connect_proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&held_out).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        let Some(spark_connect_proto::ml_relation::transform::Operator::ObjRef(r)) = t.operator
        else {
            panic!("cached model")
        };
        r.id
    };
    assert_ne!(object_id(&first), object_id(&second));
    let mut cloned = first.clone_box();
    let delayed =
        first
            .transform(&held_out)
            .unwrap()
            .select([col("id"), col("target"), col("pred")]);
    drop(first);
    assert_close(
        predictions(cloned.transform(&held_out).unwrap(), "pred"),
        &[0.0, 4.0, 12.0],
        1e-6,
    );
    drop(cloned);
    assert_close(
        predictions(delayed.clone(), "pred"),
        &[0.0, 4.0, 12.0],
        1e-6,
    );
    let rmse = RegressionEvaluator::new()
        .set_label_col("target")
        .set_prediction_col("pred")
        .set_metric_name("rmse")
        .evaluate(&delayed)
        .unwrap();
    assert!(rmse < 1e-6, "RMSE {rmse}");
    assert_close(
        predictions(second.transform(&held_out).unwrap(), "pred"),
        &[10.0, 6.0, -2.0],
        1e-6,
    );
    assert!(second
        .transform(&s.new_session().range(1).unwrap())
        .is_err());
    s.stop().unwrap();
}

#[test]
fn generalized_linear_regression_default_columns_and_zero_weight_outlier() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectorize(s.sql(
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, CAST(w AS DOUBLE) AS weight
         FROM VALUES (-3, -3, 1), (-2, -1, 1), (2, 7, 1), (3, 9, 1), (100, -1000, 0) AS t(x,y,w)"
    ).unwrap(), "features");
    let held_out = vectorize(
        s.sql(
            "SELECT id, array(CAST(x AS DOUBLE)) AS xs FROM VALUES (0, -1.5), (1, 4.5) AS t(id,x)",
        )
        .unwrap(),
        "features",
    );
    let mut default_model = GeneralizedLinearRegression::default()
        .fit(&training.filter(col("weight").gt(lit(0))))
        .unwrap();
    let mut weighted_model = GeneralizedLinearRegression::new()
        .set_weight_col("weight")
        .fit(&training)
        .unwrap();
    assert_close(
        predictions(default_model.transform(&held_out).unwrap(), "prediction"),
        &[0.0, 12.0],
        1e-6,
    );
    assert_close(
        predictions(weighted_model.transform(&held_out).unwrap(), "prediction"),
        &[0.0, 12.0],
        1e-6,
    );
    s.stop().unwrap();
}

#[test]
fn generalized_linear_regression_families_offsets_and_link_predictions() {
    let Some(s) = live_session() else {
        return;
    };
    for (family, response) in [
        ("gaussian", "eta"),
        ("binomial", "1.0D / (1.0D + exp(-eta))"),
        ("poisson", "exp(eta)"),
        ("gamma", "1.0D / eta"),
        ("tweedie", "pow(eta, -2.0D)"),
    ] {
        let training = vectorize(s.sql(&format!(
            "SELECT array(x) xs, {response} label, off FROM (SELECT id/5.0D x, (id % 3)/10.0D off, 1.0D + 0.3D * (id/5.0D) + (id % 3)/10.0D eta FROM range(12))"
        )).unwrap(), "features");
        let held_out = vectorize(s.sql(
            "SELECT id, array(x) xs, off FROM VALUES (0,0.7D,0.15D),(1,1.3D,0.25D) AS t(id,x,off)"
        ).unwrap(), "features");
        let mut estimator = GeneralizedLinearRegression::new()
            .set_family(family)
            .set_max_iter(100)
            .set_tol(1e-10)
            .set_offset_col("off")
            .set_link_prediction_col("eta");
        if family == "tweedie" {
            estimator = estimator.set_variance_power(1.5);
        }
        let mut model = estimator.fit(&training).unwrap();
        let output = model.transform(&held_out).unwrap();
        assert_close(predictions(output.clone(), "eta"), &[1.36, 1.64], 1e-5);
        let expected: Vec<f64> = [1.36_f64, 1.64]
            .into_iter()
            .map(|eta| match family {
                "gaussian" => eta,
                "binomial" => 1.0 / (1.0 + (-eta).exp()),
                "poisson" => eta.exp(),
                "gamma" => 1.0 / eta,
                "tweedie" => eta.powi(-2),
                _ => unreachable!(),
            })
            .collect();
        assert_close(predictions(output, "prediction"), &expected, 1e-5);
    }
    s.stop().unwrap();
}

#[test]
fn generalized_linear_regression_tweedie_log_link_and_regularization() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectorize(
        s.sql("SELECT array(id/5.0D) xs, exp(0.3D * (id/5.0D)) label FROM range(12)")
            .unwrap(),
        "features",
    );
    let point = vectorize(s.sql("SELECT 0 id, array(1.3D) xs").unwrap(), "features");
    let mut model = GeneralizedLinearRegression::new()
        .set_family("tweedie")
        .set_variance_power(1.5)
        .set_link_power(0.0)
        .set_fit_intercept(false)
        .set_max_iter(100)
        .set_tol(1e-10)
        .fit(&training)
        .unwrap();
    assert_close(
        predictions(model.transform(&point).unwrap(), "prediction"),
        &[0.39_f64.exp()],
        1e-5,
    );
    let mut regularized = GeneralizedLinearRegression::new()
        .set_family("tweedie")
        .set_variance_power(1.5)
        .set_link_power(0.0)
        .set_fit_intercept(false)
        .set_reg_param(2.0)
        .fit(&training)
        .unwrap();
    let value = predictions(regularized.transform(&point).unwrap(), "prediction")[0];
    assert!(value > 1.0 && value < 0.39_f64.exp());
    s.stop().unwrap();
}

#[test]
fn generalized_linear_regression_rejects_invalid_parameters_and_inputs() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectorize(
        s.sql("SELECT array(id*1.0D) xs, id*1.0D label FROM range(1,5)")
            .unwrap(),
        "features",
    );
    for mut estimator in [
        GeneralizedLinearRegression::new().set_family("unknown"),
        GeneralizedLinearRegression::new()
            .set_family("poisson")
            .set_link("logit"),
        GeneralizedLinearRegression::new().set_solver("unknown"),
        GeneralizedLinearRegression::new().set_max_iter(-1),
        GeneralizedLinearRegression::new().set_tol(-1.0),
        GeneralizedLinearRegression::new().set_reg_param(-1.0),
        GeneralizedLinearRegression::new().set_aggregation_depth(1),
        GeneralizedLinearRegression::new()
            .set_family("tweedie")
            .set_variance_power(0.5),
    ] {
        assert!(estimator.fit(&training).is_err());
    }
    assert!(GeneralizedLinearRegression::new()
        .set_weight_col("weight")
        .fit(&training.with_column("weight", lit(-1)))
        .is_err());
    assert!(GeneralizedLinearRegression::new()
        .set_family("poisson")
        .fit(&training.with_column("label", lit(-1)))
        .is_err());
    assert!(GeneralizedLinearRegression::new()
        .fit(&training.with_column("features", lit(1)))
        .is_err());
    assert!(GeneralizedLinearRegression::new()
        .fit(&training.limit(0))
        .is_err());
    s.stop().unwrap();
}
