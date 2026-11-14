//! Numerical tests of native LinearRegression. Set SPARK_REMOTE to use a live server.
mod common;

use common::live_session;
use spark_connect::column::{col, lit};
use spark_connect::expression::Expression;
use spark_connect::ml::{Evaluator, LinearRegression, Model, RegressionEvaluator, Transformer};
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
fn linear_regression_fits_different_lines_and_keeps_lazy_predictions_alive() {
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
    let mut estimator = LinearRegression::new()
        .set_features_col("input")
        .set_label_col("target")
        .set_prediction_col("pred")
        .set_max_iter(100)
        .set_reg_param(0.0)
        .set_elastic_net_param(0.0)
        .set_tol(1e-8)
        .set_fit_intercept(true)
        .set_standardization(false)
        .set_solver("normal")
        .set_weight_col("weight")
        .set_aggregation_depth(2)
        .set_loss("squaredError")
        .set_epsilon(1.35)
        .set_max_block_size_in_mb(1.0);
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
fn linear_regression_default_columns_and_zero_weight_outlier() {
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
    let mut default_model = LinearRegression::default()
        .fit(&training.filter(col("weight").gt(lit(0))))
        .unwrap();
    let mut weighted_model = LinearRegression::new()
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
fn linear_regression_iterative_elastic_net_and_huber_loss() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectorize(
        s.sql(
            "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(2*x AS DOUBLE) AS label
         FROM VALUES (-3), (-2), (-1), (1), (2), (3) AS t(x)",
        )
        .unwrap(),
        "features",
    );
    let held_out = vectorize(
        s.sql("SELECT 0 AS id, array(2.0D) AS xs").unwrap(),
        "features",
    );
    let mut regularized = LinearRegression::new()
        .set_solver("l-bfgs")
        .set_fit_intercept(false)
        .set_reg_param(1.0)
        .set_elastic_net_param(0.5)
        .fit(&training)
        .unwrap();
    let value = predictions(regularized.transform(&held_out).unwrap(), "prediction")[0];
    assert!(
        value > 0.0 && value < 3.9,
        "regularization should shrink slope: {value}"
    );
    let robust_training = vectorize(s.sql(
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label
         FROM VALUES (-3, -3.01), (-2, -0.98), (-1, 1.02), (1, 4.98), (2, 7.01), (3, 8.99), (0, 1000) AS t(x,y)"
    ).unwrap(), "features");
    let mut huber = LinearRegression::new()
        .set_loss("huber")
        .set_solver("l-bfgs")
        .set_max_iter(200)
        .set_tol(1e-8)
        .fit(&robust_training)
        .unwrap();
    assert_close(
        predictions(huber.transform(&held_out).unwrap(), "prediction"),
        &[7.0],
        0.2,
    );
    s.stop().unwrap();
}

#[test]
fn linear_regression_fit_propagates_invalid_data_and_parameter_errors() {
    let Some(s) = live_session() else {
        return;
    };
    assert!(LinearRegression::new().fit(&s.range(3).unwrap()).is_err());
    let training = vectorize(
        s.sql("SELECT array(CAST(id AS DOUBLE)) AS xs, CAST(id AS DOUBLE) AS label FROM range(5)")
            .unwrap(),
        "features",
    );
    for mut estimator in [
        LinearRegression::new().set_reg_param(-1.0),
        LinearRegression::new().set_elastic_net_param(1.5),
        LinearRegression::new().set_loss("huber").set_epsilon(1.0),
        LinearRegression::new()
            .set_solver("normal")
            .set_loss("huber"),
    ] {
        assert!(estimator.fit(&training).is_err());
    }
    assert!(LinearRegression::new()
        .fit(&training.with_column("label", spark_connect::column::lit_string("invalid")))
        .is_err());
    s.stop().unwrap();
}
