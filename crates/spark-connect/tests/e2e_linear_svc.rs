//! Native LinearSVC tests. Set SPARK_REMOTE to run against a live Spark Connect server.
use spark_connect::column::{col, lit};
use spark_connect::expression::Expression;
use spark_connect::ml::{Estimator, LinearSVC};
use spark_connect::{Column, DataFrame, SparkSession};

fn session() -> Option<SparkSession> {
    let remote = std::env::var("SPARK_REMOTE").ok()?;
    Some(
        SparkSession::builder()
            .remote(&remote)
            .get_or_create()
            .unwrap(),
    )
}

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
        .map(|row| row.get(0).unwrap().as_f64().unwrap())
        .collect()
}

#[test]
fn linear_svc_fits_separate_models_and_predicts_on_unseen_data() {
    let Some(s) = session() else {
        return;
    };
    let training = vectorize(
        s.sql(
            "SELECT array(CAST(x AS DOUBLE)) AS xs, \
         CAST(CASE WHEN x > 0 THEN 1 ELSE 0 END AS DOUBLE) AS target, 1.0D AS weight \
         FROM VALUES (-4), (-3), (-2), (-1), (1), (2), (3), (4) AS t(x)",
        )
        .unwrap(),
        "input",
    );
    let held_out = vectorize(
        s.sql(
            "SELECT id, array(CAST(x AS DOUBLE)) AS xs \
         FROM VALUES (0, -3.5), (1, -1.5), (2, 1.5), (3, 3.5) AS t(id, x)",
        )
        .unwrap(),
        "input",
    );
    let mut estimator = LinearSVC::new()
        .set_feature_col("input")
        .set_label_col("target")
        .set_prediction_col("pred")
        .set_raw_prediction_col("margin")
        .set_max_iter(50)
        .set_reg_param(0.1)
        .set_tol(1e-6)
        .set_fit_intercept(false)
        .set_standardization(false)
        .set_threshold(0.0)
        .set_weight_col("weight")
        .set_aggregation_depth(2)
        .set_max_block_size_in_mb(1.0);
    let mut first = estimator.fit(&training).expect("fit binary classifier");
    let mut second = estimator
        .fit(&training.with_column("target", lit(1) - col("target")))
        .unwrap();
    let object_id = |model: &dyn spark_connect::ml::Model| {
        let spark_connect_proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&held_out).ml_type.unwrap()
        else {
            panic!("transform")
        };
        let Some(spark_connect_proto::ml_relation::transform::Operator::ObjRef(r)) = t.operator
        else {
            panic!("cached model")
        };
        r.id
    };
    assert_ne!(object_id(first.as_ref()), object_id(second.as_ref()));
    let mut cloned = first.clone_box();
    let delayed = first.transform(&held_out).unwrap();
    drop(first);
    let output = cloned.transform(&held_out).unwrap();
    assert!(output.columns().unwrap().contains(&"margin".to_string()));
    assert_eq!(predictions(delayed, "pred"), vec![0.0, 0.0, 1.0, 1.0]);
    assert_eq!(predictions(output, "pred"), vec![0.0, 0.0, 1.0, 1.0]);
    assert_eq!(
        predictions(second.transform(&held_out).unwrap(), "pred"),
        vec![1.0, 1.0, 0.0, 0.0]
    );
    let mut params = cloned.params().clone().set_param_double("threshold", 1e6);
    std::mem::swap(cloned.params_mut(), &mut params);
    assert_eq!(
        predictions(cloned.transform(&held_out).unwrap(), "pred"),
        vec![0.0; 4]
    );
    assert!(second
        .transform(&s.new_session().range(1).unwrap())
        .is_err());
    s.stop().unwrap();
}

#[test]
fn linear_svc_defaults_and_zero_weight_training_examples() {
    let Some(s) = session() else {
        return;
    };
    let training = vectorize(s.sql(
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, CAST(w AS DOUBLE) AS weight \
         FROM VALUES (-3, 0, 1), (-2, 0, 1), (2, 1, 1), (3, 1, 1), (100, 0, 0) AS t(x, y, w)"
    ).unwrap(), "features");
    let held_out = vectorize(
        s.sql(
            "SELECT id, array(CAST(x AS DOUBLE)) AS xs FROM VALUES (0, -1.5), (1, 1.5) AS t(id, x)",
        )
        .unwrap(),
        "features",
    );
    let mut default_model = LinearSVC::default()
        .fit(&training.filter(col("weight").gt(lit(0))))
        .unwrap();
    let mut weighted_model = LinearSVC::new()
        .set_weight_col("weight")
        .fit(&training)
        .unwrap();
    assert_eq!(
        predictions(default_model.transform(&held_out).unwrap(), "prediction"),
        vec![0.0, 1.0]
    );
    assert_eq!(
        predictions(weighted_model.transform(&held_out).unwrap(), "prediction"),
        vec![0.0, 1.0]
    );
    s.stop().unwrap();
}

#[test]
fn linear_svc_fit_rejects_invalid_training_data_and_parameters() {
    let Some(s) = session() else {
        return;
    };
    assert!(LinearSVC::new().fit(&s.range(3).unwrap()).is_err());
    let multiclass = vectorize(s.sql(
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(x AS DOUBLE) AS label FROM VALUES (0), (1), (2) AS t(x)"
    ).unwrap(), "features");
    assert!(LinearSVC::new().fit(&multiclass).is_err());
    assert!(LinearSVC::new()
        .set_reg_param(-1.0)
        .fit(&multiclass)
        .is_err());
    s.stop().unwrap();
}
