//! Native random forest regression tests. Set SPARK_REMOTE to run against a live Connect server.
mod common;

use common::live_session;
use spark_connect::column::{col, lit_double, lit_string};
use spark_connect::expression::Expression;
use spark_connect::ml::{Estimator, Model, RandomForestRegressor, Transformer};
use spark_connect::{Column, DataFrame, SparkSession};

fn vectors(s: &SparkSession, sql: &str, name: &str) -> DataFrame {
    s.sql(sql).unwrap().with_column(
        name,
        Column::new(Expression::function(
            "array_to_vector",
            vec![col("xs").expression().clone()],
        )),
    )
}

fn vector_array(name: &str) -> Column {
    Column::new(Expression::function(
        "vector_to_array",
        vec![
            col(name).expression().clone(),
            lit_string("float64").expression().clone(),
        ],
    ))
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

fn deterministic() -> RandomForestRegressor {
    RandomForestRegressor::new()
        .set_seed(42)
        .set_num_trees(3)
        .set_feature_subset_strategy("all")
        .set_bootstrap(false)
}

#[test]
fn random_forest_regressor_learns_nonlinear_labels_and_keeps_cached_models_alive() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectors(
        &s,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS target, 1.0D AS weight
         FROM VALUES (-3,-4.0),(-2,-4.0),(-1,5.5),(0,5.5),(1,5.5),(2,-4.0),(3,-4.0) AS t(x,y)",
        "input",
    );
    let held_out = vectors(
        &s,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs
         FROM VALUES (0,-2.5),(1,-0.5),(2,0.5),(3,2.5) AS t(id,x)",
        "input",
    );
    let mut estimator = deterministic()
        .set_features_col("input")
        .set_label_col("target")
        .set_prediction_col("value")
        .set_leaf_col("leaves")
        .set_max_depth(3)
        .set_max_bins(16)
        .set_min_instances_per_node(1)
        .set_min_info_gain(0.0)
        .set_max_memory_in_mb(64)
        .set_cache_node_ids(true)
        .set_checkpoint_interval(-1)
        .set_impurity("variance")
        .set_subsampling_rate(1.0)
        .set_min_weight_fraction_per_node(0.0)
        .set_weight_col("weight");
    let mut first = estimator.fit(&training).unwrap();
    let mut second = estimator
        .fit(&training.with_column("target", col("target") + lit_double(10.0)))
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
    let delayed = first
        .transform(&held_out)
        .unwrap()
        .select([col("id"), col("value")]);
    drop(first);
    let output = cloned.transform(&held_out).unwrap();
    assert_eq!(
        predictions(output.clone(), "value"),
        vec![-4.0, 5.5, 5.5, -4.0]
    );
    let leaves = output.select([vector_array("leaves")]).collect().unwrap();
    for row in leaves {
        let spark_connect::row::Value::List(values) = row.get(0).unwrap() else {
            panic!("leaf vector")
        };
        assert_eq!(values.len(), 3);
        assert!(values.iter().all(|v| v.as_f64().unwrap() >= 0.0));
    }
    let mut repeated = estimator.fit(&training).unwrap();
    assert_eq!(
        predictions(repeated.transform(&held_out).unwrap(), "value"),
        vec![-4.0, 5.5, 5.5, -4.0]
    );
    drop(cloned);
    assert_eq!(predictions(delayed, "value"), vec![-4.0, 5.5, 5.5, -4.0]);
    assert_eq!(
        predictions(second.transform(&held_out).unwrap(), "value"),
        vec![6.0, 15.5, 15.5, 6.0]
    );
    assert!(second
        .transform(&s.new_session().range(1).unwrap())
        .is_err());
    s.stop().unwrap();
}

#[test]
fn random_forest_regressor_weights_change_the_predicted_mean() {
    let Some(s) = live_session() else {
        return;
    };
    let data = vectors(
        &s,
        "SELECT 0 AS id, array(0.0D) AS xs, CAST(y AS DOUBLE) AS label, CAST(w AS DOUBLE) AS weight
         FROM VALUES (-2,1),(-2,1),(-2,1),(10,9) AS t(y,w)",
        "features",
    );
    let mut plain = deterministic().set_max_depth(0).fit(&data).unwrap();
    let mut weighted = deterministic()
        .set_max_depth(0)
        .set_weight_col("weight")
        .fit(&data)
        .unwrap();
    assert_eq!(
        predictions(plain.transform(&data).unwrap(), "prediction"),
        vec![1.0; 4]
    );
    assert_eq!(
        predictions(weighted.transform(&data).unwrap(), "prediction"),
        vec![7.0; 4]
    );
    s.stop().unwrap();
}

#[test]
fn random_forest_regressor_bootstrap_sampling_is_reproducible() {
    let Some(s) = live_session() else {
        return;
    };
    let data = vectors(
        &s,
        "SELECT id, array(CAST(id AS DOUBLE), CAST(id % 3 AS DOUBLE)) AS xs,
         CAST(CASE WHEN id < 20 THEN -3 ELSE 8 END AS DOUBLE) AS label FROM range(40)",
        "features",
    );
    let mut estimator = RandomForestRegressor::new()
        .set_seed(1_i64 << 33)
        .set_num_trees(7)
        .set_max_depth(3)
        .set_bootstrap(true)
        .set_subsampling_rate(0.8)
        .set_feature_subset_strategy("sqrt");
    let mut first = estimator.fit(&data).unwrap();
    let mut second = estimator.fit(&data).unwrap();
    let actual = predictions(first.transform(&data).unwrap(), "prediction");
    assert_eq!(
        actual,
        predictions(second.transform(&data).unwrap(), "prediction")
    );
    assert!(actual
        .iter()
        .all(|x| x.is_finite() && *x >= -3.0 && *x <= 8.0));
    assert!(actual[0] < actual[39]);
    s.stop().unwrap();
}

#[test]
fn random_forest_regressor_rejects_invalid_parameters_and_data() {
    let Some(s) = live_session() else {
        return;
    };
    let data = vectors(
        &s,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, 1.0D AS weight
         FROM VALUES (-1,-2.5),(1,3.5) AS t(x,y)",
        "features",
    );
    for mut estimator in [
        deterministic().set_num_trees(0),
        deterministic().set_max_depth(31),
        deterministic().set_max_bins(1),
        deterministic().set_min_instances_per_node(0),
        deterministic().set_min_info_gain(-1.0),
        deterministic().set_max_memory_in_mb(-1),
        deterministic().set_checkpoint_interval(0),
        deterministic().set_impurity("gini"),
        deterministic().set_feature_subset_strategy("invalid"),
        deterministic().set_subsampling_rate(0.0),
        deterministic().set_min_weight_fraction_per_node(0.5),
    ] {
        assert!(
            estimator.fit(&data).is_err(),
            "invalid parameters: {:?}",
            estimator.params()
        );
    }
    assert!(deterministic().fit(&data.drop(vec!["label"])).is_err());
    assert!(deterministic()
        .fit(&data.with_column("label", lit_double(f64::NAN)))
        .is_err());
    assert!(deterministic()
        .set_weight_col("weight")
        .fit(&data.with_column("weight", lit_double(-1.0)))
        .is_err());
    assert!(deterministic()
        .fit(&data.with_column("features", lit_double(1.0)))
        .is_err());
    s.stop().unwrap();
}
