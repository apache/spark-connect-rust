//! Native BisectingKMeans tests. Set SPARK_REMOTE to run against a live Connect server.
mod common;

use common::live_session;
use spark_connect::column::{col, lit_double};
use spark_connect::expression::Expression;
use spark_connect::ml::{BisectingKMeans, Model, Transformer};
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

fn predictions(df: DataFrame, name: &str) -> Vec<i64> {
    df.order_by(vec![col("id").asc().expression().clone()])
        .select([col(name)])
        .collect()
        .unwrap()
        .into_iter()
        .map(|r| r.get(0).unwrap().as_i64().unwrap())
        .collect()
}

// Cluster numbers may be permuted without changing the clustering.
fn assert_partition(actual: &[i64], expected: &[i64]) {
    assert_eq!(actual.len(), expected.len());
    for i in 0..actual.len() {
        for j in 0..actual.len() {
            assert_eq!(
                actual[i] == actual[j],
                expected[i] == expected[j],
                "{actual:?}"
            );
        }
    }
}

#[test]
fn bisecting_kmeans_learns_distinct_models_and_keeps_lazy_predictions_alive() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectorize(
        s.sql(
            "SELECT array(CAST(x AS DOUBLE)) AS xs, 1.0D AS weight
         FROM VALUES (9), (10), (11), (29), (30), (31) AS t(x)",
        )
        .unwrap(),
        "input",
    );
    let shifted = vectorize(
        s.sql(
            "SELECT array(CAST(x AS DOUBLE)) AS xs, 1.0D AS weight
         FROM VALUES (29), (30), (31), (49), (50), (51) AS t(x)",
        )
        .unwrap(),
        "input",
    );
    let held_out = vectorize(
        s.sql(
            "SELECT id, array(CAST(x AS DOUBLE)) AS xs
         FROM VALUES (0,10), (1,19), (2,30), (3,41), (4,50) AS t(id,x)",
        )
        .unwrap(),
        "input",
    );
    let mut estimator = BisectingKMeans::new()
        .set_features_col("input")
        .set_prediction_col("cluster")
        .set_k(2)
        .set_min_divisible_cluster_size(1.0)
        .set_max_iter(50)
        .set_seed(42)
        .set_distance_measure("euclidean")
        .set_weight_col("weight");
    let mut first = estimator.fit(&training).unwrap();
    let mut second = estimator.fit(&shifted).unwrap();
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
        .select([col("id"), col("cluster")]);
    drop(first);
    let original = predictions(cloned.transform(&held_out).unwrap(), "cluster");
    assert_partition(&original, &[0, 0, 1, 1, 1]);
    let mut repeated = estimator.fit(&training).unwrap();
    assert_eq!(
        original,
        predictions(repeated.transform(&held_out).unwrap(), "cluster")
    );
    drop(cloned);
    assert_partition(&predictions(delayed, "cluster"), &[0, 0, 1, 1, 1]);
    assert_partition(
        &predictions(second.transform(&held_out).unwrap(), "cluster"),
        &[0, 0, 0, 1, 1],
    );
    assert!(second
        .transform(&s.new_session().range(1).unwrap())
        .is_err());
    s.stop().unwrap();
}

#[test]
fn bisecting_kmeans_weights_move_a_cluster_boundary() {
    let Some(s) = live_session() else {
        return;
    };
    let training=vectorize(s.sql("SELECT array(x) xs, weight FROM VALUES (-10.0D,1.0D),(-9.0D,1.0D),(-8.0D,1.0D),(2.0D,100.0D),(3.0D,1.0D),(4.0D,1.0D),(10.0D,1.0D),(11.0D,1.0D) AS t(x,weight)").unwrap(),"features");
    let held_out = vectorize(
        s.sql(
            "SELECT id, array(x) xs FROM VALUES (0,-9.0D),(1,-2.0D),(2,2.0D),(3,11.0D) AS t(id,x)",
        )
        .unwrap(),
        "features",
    );
    let mut plain = BisectingKMeans::new()
        .set_k(2)
        .set_seed(42)
        .fit(&training)
        .unwrap();
    let mut weighted = BisectingKMeans::new()
        .set_k(2)
        .set_seed(42)
        .set_weight_col("weight")
        .fit(&training)
        .unwrap();
    assert_partition(
        &predictions(plain.transform(&held_out).unwrap(), "prediction"),
        &[0, 0, 1, 1],
    );
    assert_partition(
        &predictions(weighted.transform(&held_out).unwrap(), "prediction"),
        &[0, 1, 1, 1],
    );
}

#[test]
fn bisecting_kmeans_divisibility_limits_the_number_of_leaf_clusters() {
    let Some(s) = live_session() else {
        return;
    };
    // The nonzero centroid avoids coincident initial split centers.
    let data=vectorize(s.sql("SELECT id, array(x,y) xs FROM VALUES (0,9.0D,20.0D),(1,11.0D,20.0D),(2,9.0D,40.0D),(3,11.0D,40.0D),(4,29.0D,20.0D),(5,31.0D,20.0D),(6,29.0D,40.0D),(7,31.0D,40.0D) AS t(id,x,y)").unwrap(),"features");
    for (size, expected) in [(1.0, 4), (0.75, 2), (9.0, 1)] {
        let mut estimator = BisectingKMeans::new().set_seed(42);
        if size != 1.0 {
            estimator = estimator.set_k(4).set_min_divisible_cluster_size(size);
        }
        let mut model = estimator.fit(&data).unwrap();
        let labels = predictions(model.transform(&data).unwrap(), "prediction");
        assert_eq!(
            labels
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            expected
        );
    }
}

#[test]
fn bisecting_kmeans_cosine_distance_groups_directions_across_magnitudes() {
    let Some(s) = live_session() else {
        return;
    };
    let training=vectorize(s.sql("SELECT array(x,y) xs FROM VALUES (1.0D,0.0D),(5.0D,0.1D),(0.0D,1.0D),(0.1D,5.0D),(-1.0D,-1.0D),(-5.0D,-4.0D) AS t(x,y)").unwrap(),"features");
    let held_out=vectorize(s.sql("SELECT id, array(x,y) xs FROM VALUES (0,100.0D,0.0D),(1,0.01D,0.0D),(2,0.0D,100.0D),(3,0.0D,0.01D),(4,-100.0D,-100.0D),(5,-0.01D,-0.01D) AS t(id,x,y)").unwrap(),"features");
    let mut model = BisectingKMeans::new()
        .set_k(3)
        .set_seed(42)
        .set_distance_measure("cosine")
        .fit(&training)
        .unwrap();
    assert_partition(
        &predictions(model.transform(&held_out).unwrap(), "prediction"),
        &[0, 0, 1, 1, 2, 2],
    );
}

#[test]
fn bisecting_kmeans_rejects_invalid_parameters_and_data() {
    let Some(s) = live_session() else {
        return;
    };
    let training=vectorize(s.sql("SELECT array(x,y) xs FROM VALUES (1.0D,1.0D),(2.0D,2.0D),(9.0D,9.0D),(10.0D,10.0D) AS t(x,y)").unwrap(),"features");
    for mut estimator in [
        BisectingKMeans::new().set_k(1),
        BisectingKMeans::new().set_max_iter(-1),
        BisectingKMeans::new().set_min_divisible_cluster_size(-1.0),
        BisectingKMeans::new().set_distance_measure("unknown"),
    ] {
        assert!(estimator.fit(&training).is_err());
    }
    assert!(BisectingKMeans::new()
        .set_weight_col("weight")
        .fit(&training.with_column("weight", lit_double(-1.0)))
        .is_err());
    assert!(BisectingKMeans::new()
        .fit(&training.with_column("features", lit_double(1.0)))
        .is_err());
    let zero = vectorize(
        s.sql("SELECT array(0.0D,0.0D) xs FROM range(4)").unwrap(),
        "features",
    );
    assert!(BisectingKMeans::new()
        .set_distance_measure("cosine")
        .fit(&zero)
        .is_err());
    let empty = vectorize(
        s.sql("SELECT array(1.0D,1.0D) xs WHERE false").unwrap(),
        "features",
    );
    assert!(BisectingKMeans::new().fit(&empty).is_err());
}
