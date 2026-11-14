//! Native decision tree tests. Set SPARK_REMOTE to run against a live Connect server.
mod common;

use common::live_session;
use spark_connect::column::{col, lit_double, lit_string};
use spark_connect::expression::Expression;
use spark_connect::ml::{DecisionTreeClassifier, Estimator, Model, Transformer};
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

fn deterministic() -> DecisionTreeClassifier {
    DecisionTreeClassifier::new().set_seed(42)
}

#[test]
fn decision_tree_classifier_learns_nonlinear_labels_and_keeps_cached_models_alive() {
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
        .set_prune_tree(false)
        .set_max_memory_in_mb(64)
        .set_cache_node_ids(true)
        .set_checkpoint_interval(-1)
        .set_impurity("entropy")
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
        .select([vector_array("probs"), vector_array("votes"), col("leaves")])
        .collect()
        .unwrap();
    for row in vectors {
        assert!(row.get(2).unwrap().as_f64().unwrap() >= 0.0);
        for (i, size) in [(0, 2), (1, 2)] {
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
fn decision_tree_classifier_supports_multiclass_probabilities_and_leaf_indices() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectors(
        &s,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label
         FROM VALUES (-3,0),(-2,0),(-1,1),(0,1),(1,2),(2,2) AS t(x,y)",
        "features",
    );
    let held_out = vectors(&s,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs FROM VALUES (0,-2.5),(1,-0.5),(2,1.5) AS t(id,x)", "features");
    let mut model = deterministic()
        .set_max_depth(3)
        .set_leaf_col("leaves")
        .fit(&training)
        .unwrap();
    let output = model.transform(&held_out).unwrap();
    assert_eq!(
        predictions(output.clone(), "prediction"),
        vec![0.0, 1.0, 2.0]
    );
    let rows = output
        .order_by(vec![col("id").asc().expression().clone()])
        .select([
            vector_array("probability"),
            vector_array("rawPrediction"),
            col("leaves"),
        ])
        .collect()
        .unwrap();
    for (class, row) in rows.iter().enumerate() {
        let spark_connect::row::Value::List(probs) = row.get(0).unwrap() else {
            panic!("probability vector")
        };
        let spark_connect::row::Value::List(raw) = row.get(1).unwrap() else {
            panic!("raw vector")
        };
        assert_eq!(row.get(2).unwrap().as_f64(), Some(class as f64));
        assert_eq!(probs.len(), 3);
        assert_eq!(raw.len(), 3);
        for i in 0..3 {
            assert_eq!(probs[i].as_f64(), Some(if i == class { 1.0 } else { 0.0 }));
            assert_eq!(raw[i].as_f64(), Some(if i == class { 2.0 } else { 0.0 }));
        }
    }
    s.stop().unwrap();
}

#[test]
fn decision_tree_classifier_weights_and_class_thresholds_change_predictions() {
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
fn decision_tree_classifier_rejects_invalid_parameters_and_data() {
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
        deterministic().set_max_depth(31),
        deterministic().set_max_bins(1),
        deterministic().set_min_instances_per_node(0),
        deterministic().set_min_info_gain(-1.0),
        deterministic().set_max_memory_in_mb(-1),
        deterministic().set_checkpoint_interval(0),
        deterministic().set_impurity("invalid"),
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
        .fit(&data.with_column("label", lit_double(-1.0)))
        .is_err());
    assert!(deterministic()
        .set_weight_col("weight")
        .fit(&data.with_column("weight", lit_double(-1.0)))
        .is_err());
    s.stop().unwrap();
}
