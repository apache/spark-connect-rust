//! Native gradient-boosted classifier tests. Set SPARK_REMOTE to run against a live Connect server.
mod common;

use common::live_session;
use spark_connect::column::{col, lit_double, lit_string};
use spark_connect::expression::Expression;
use spark_connect::ml::{Estimator, GBTClassifier, Model, Transformer};
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

fn deterministic() -> GBTClassifier {
    GBTClassifier::new()
        .set_seed(42)
        .set_max_iter(3)
        .set_feature_subset_strategy("all")
}

#[test]
fn gbt_classifier_learns_nonlinear_labels_and_keeps_cached_models_alive() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectors(
        &s,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS target, 1.0D AS weight
         FROM VALUES (-3,0),(-2,0),(-1,1),(0,1),(1,1),(2,0),(3,0) AS t(x,y)",
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
        .set_prediction_col("class")
        .set_probability_col("probs")
        .set_raw_prediction_col("votes")
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
        .fit(&training.with_column("target", lit_double(1.0) - col("target")))
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
        .select([col("id"), col("class")]);
    drop(first);
    let output = cloned.transform(&held_out).unwrap();
    assert_eq!(
        predictions(output.clone(), "class"),
        vec![0.0, 1.0, 1.0, 0.0]
    );
    let vectors = output
        .select([
            vector_array("probs"),
            vector_array("votes"),
            vector_array("leaves"),
        ])
        .collect()
        .unwrap();
    for row in vectors {
        for (i, size) in [(0, 2), (1, 2), (2, 3)] {
            let spark_connect::row::Value::List(values) = row.get(i).unwrap() else {
                panic!("vector array")
            };
            assert_eq!(values.len(), size);
        }
    }
    let mut repeated = estimator.fit(&training).unwrap();
    assert_eq!(
        predictions(repeated.transform(&held_out).unwrap(), "class"),
        vec![0.0, 1.0, 1.0, 0.0]
    );
    drop(cloned);
    assert_eq!(predictions(delayed, "class"), vec![0.0, 1.0, 1.0, 0.0]);
    assert_eq!(
        predictions(second.transform(&held_out).unwrap(), "class"),
        vec![1.0, 0.0, 0.0, 1.0]
    );
    assert!(second
        .transform(&s.new_session().range(1).unwrap())
        .is_err());
    s.stop().unwrap();
}

#[test]
fn gbt_classifier_weights_and_class_thresholds_change_predictions() {
    let Some(s) = live_session() else {
        return;
    };
    // No split: the class probability is the (weighted) label frequency.
    let training = vectors(
        &s,
        "SELECT array(0.0D) AS xs, CAST(y AS DOUBLE) AS label, CAST(w AS DOUBLE) AS weight
         FROM VALUES (0,1),(0,1),(0,1),(1,9) AS t(y,w)",
        "features",
    );
    let held_out = vectors(&s, "SELECT 0 AS id, array(0.0D) AS xs", "features");
    let mut plain = deterministic().set_max_depth(0).fit(&training).unwrap();
    let mut weighted = deterministic()
        .set_max_depth(0)
        .set_weight_col("weight")
        .fit(&training)
        .unwrap();
    let mut thresholded = deterministic()
        .set_max_depth(0)
        .set_thresholds([1.0, 0.1])
        .fit(&training)
        .unwrap();
    assert_eq!(
        predictions(plain.transform(&held_out).unwrap(), "prediction"),
        vec![0.0]
    );
    assert_eq!(
        predictions(weighted.transform(&held_out).unwrap(), "prediction"),
        vec![1.0]
    );
    assert_eq!(
        predictions(thresholded.transform(&held_out).unwrap(), "prediction"),
        vec![1.0]
    );
    s.stop().unwrap();
}

#[test]
fn gbt_classifier_subsampling_and_feature_sampling_train_reproducibly() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectors(
        &s,
        "SELECT id, array(CAST(id AS DOUBLE), CAST(id % 3 AS DOUBLE)) AS xs,
         CAST(CASE WHEN id < 20 THEN 0 ELSE 1 END AS DOUBLE) AS label FROM range(40)",
        "features",
    );
    let mut estimator = GBTClassifier::default()
        .set_seed(1_i64 << 33)
        .set_max_iter(5)
        .set_max_depth(3)
        .set_subsampling_rate(0.8)
        .set_feature_subset_strategy("sqrt");
    let mut first = estimator.fit(&training).unwrap();
    let mut repeated = estimator.fit(&training).unwrap();
    let first = predictions(first.transform(&training).unwrap(), "prediction");
    assert_eq!(
        first,
        predictions(repeated.transform(&training).unwrap(), "prediction")
    );
    assert!(first[..15].iter().all(|p| *p == 0.0));
    assert!(first[25..].iter().all(|p| *p == 1.0));
    s.stop().unwrap();
}

#[test]
fn gbt_classifier_rejects_invalid_parameters_and_data() {
    let Some(s) = live_session() else {
        return;
    };
    let data = vectors(
        &s,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, 1.0D AS weight
         FROM VALUES (-1,0),(1,1) AS t(x,y)",
        "features",
    );
    for mut estimator in [
        deterministic().set_max_iter(0),
        deterministic().set_max_depth(31),
        deterministic().set_max_bins(1),
        deterministic().set_min_instances_per_node(0),
        deterministic().set_min_info_gain(-1.0),
        deterministic().set_max_memory_in_mb(-1),
        deterministic().set_checkpoint_interval(0),
        deterministic().set_loss_type("squared"),
        deterministic().set_step_size(0.0),
        deterministic().set_validation_tol(-0.1),
        deterministic().set_feature_subset_strategy("invalid"),
        deterministic().set_subsampling_rate(0.0),
        deterministic().set_min_weight_fraction_per_node(0.5),
        deterministic().set_thresholds([1.0, 1.0, 1.0]),
    ] {
        assert!(
            estimator.fit(&data).is_err(),
            "invalid params: {:?}",
            estimator.params()
        );
    }
    assert!(deterministic().fit(&data.drop(vec!["label"])).is_err());
    assert!(deterministic()
        .fit(&data.with_column("label", lit_double(2.0)))
        .is_err());
    assert!(deterministic()
        .set_weight_col("weight")
        .fit(&data.with_column("weight", lit_double(-1.0)))
        .is_err());
    s.stop().unwrap();
}

#[test]
fn gbt_classifier_validation_rows_stop_boosting_early() {
    let Some(s) = live_session() else {
        return;
    };
    let data = vectors(
        &s,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, validation
         FROM VALUES (0,-3,0,false),(1,-2,0,false),(2,2,1,false),(3,3,1,false),
         (4,-3,1,true),(5,-2,1,true),(6,2,0,true),(7,3,0,true) AS t(id,x,y,validation)",
        "features",
    );
    let mut model = GBTClassifier::new()
        .set_seed(42)
        .set_max_depth(1)
        .set_max_iter(6)
        .set_step_size(0.2)
        .set_leaf_col("leaves")
        .set_validation_tol(0.01)
        .set_validation_indicator_col("validation")
        .fit(&data)
        .unwrap();
    let output = model.transform(&data).unwrap();
    assert_eq!(
        predictions(output.clone(), "prediction"),
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    );
    for row in output.select([vector_array("leaves")]).collect().unwrap() {
        let spark_connect::row::Value::List(leaves) = row.get(0).unwrap() else {
            panic!("leaves")
        };
        assert_eq!(leaves.len(), 1);
    }
    s.stop().unwrap();
}
