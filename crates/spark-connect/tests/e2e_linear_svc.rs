//! Native LinearSVC tests. Set SPARK_REMOTE to run against a live Spark Connect server.
mod common;

use common::live_session;
use spark_connect::column::{col, lit};
use spark_connect::expression::Expression;
use spark_connect::ml::{LinearSVC, Model, Transformer};
use spark_connect::{Column, DataFrame, SparkSession};

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
    let Some(s) = live_session() else {
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
        .set_features_col("input")
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
    let Some(s) = live_session() else {
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
    let Some(s) = live_session() else {
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

/// The number of models that the server caches for the session, from `MlCommand.GetCacheInfo`.
fn cached_models(s: &SparkSession) -> usize {
    let command = spark_connect_proto::Command {
        command_type: Some(spark_connect_proto::command::CommandType::MlCommand(
            spark_connect_proto::MlCommand {
                command: Some(spark_connect_proto::ml_command::Command::GetCacheInfo(
                    spark_connect_proto::ml_command::GetCacheInfo {},
                )),
            },
        )),
    };
    let responses = s.execute_command_proto(command).unwrap();
    responses
        .into_iter()
        .find_map(|response| match response.response_type {
            Some(spark_connect_proto::execute_plan_response::ResponseType::MlCommandResult(
                result,
            )) => match result.result_type {
                Some(spark_connect_proto::ml_command_result::ResultType::Param(literal)) => {
                    match literal.literal_type {
                        Some(spark_connect_proto::expression::literal::LiteralType::Array(
                            array,
                        )) => Some(array.elements.len()),
                        _ => None,
                    }
                }
                _ => None,
            },
            _ => None,
        })
        .expect("GetCacheInfo returns the cached objects as an array")
}

/// Wait until the server holds `expected` cached models. The release of a model is sent
/// without waiting for the server, so its deletion can land after the drop.
fn wait_for_cached_models(s: &SparkSession, expected: usize) -> usize {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let count = cached_models(s);
        if count == expected || std::time::Instant::now() >= deadline {
            return count;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

#[test]
fn linear_svc_model_is_deleted_from_the_server_when_its_last_owner_is_dropped() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectorize(
        s.sql(
            "SELECT array(CAST(x AS DOUBLE)) AS xs, \
         CAST(CASE WHEN x > 0 THEN 1 ELSE 0 END AS DOUBLE) AS target \
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
        .set_features_col("input")
        .set_label_col("target")
        .set_max_iter(10);
    let before = cached_models(&s);
    // Models that are dropped right after fitting leave nothing on the server.
    for _ in 0..3 {
        let model = estimator.fit(&training).unwrap();
        drop(model);
    }
    assert_eq!(wait_for_cached_models(&s, before), before);

    // A clone and a lazy DataFrame that the model transformed keep it cached.
    let mut model = estimator.fit(&training).unwrap();
    let mut other = estimator
        .clone()
        .set_prediction_col("second")
        .set_raw_prediction_col("second_margin")
        .fit(&training)
        .unwrap();
    assert_eq!(cached_models(&s), before + 2);
    let expected = predictions(model.transform(&held_out).unwrap(), "prediction");
    let clone = model.clone_box();
    let lazy = model.transform(&held_out).unwrap();
    // A DataFrame of another model over `lazy` keeps the first model alive through the input
    // plan, since the relation of the input is sent inline.
    let chained = other.transform(&lazy).unwrap();
    drop(model);
    drop(clone);
    drop(other);
    drop(lazy);
    assert_eq!(cached_models(&s), before + 2);
    assert_eq!(predictions(chained.clone(), "second"), expected);
    drop(chained);
    assert_eq!(wait_for_cached_models(&s, before), before);
    s.stop().unwrap();
}
