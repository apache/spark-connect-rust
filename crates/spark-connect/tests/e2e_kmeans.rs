//! Native KMeans tests. Set SPARK_REMOTE to run against a live Connect server.
mod common;

use common::live_session;
use spark_connect::column::{col, lit};
use spark_connect::expression::Expression;
use spark_connect::ml::{KMeans, Model, Transformer};
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
fn kmeans_learns_distinct_models_and_keeps_lazy_predictions_alive() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectorize(
        s.sql(
            "SELECT array(CAST(x AS DOUBLE)) AS xs, 1.0D AS weight
         FROM VALUES (-11), (-10), (-9), (9), (10), (11) AS t(x)",
        )
        .unwrap(),
        "input",
    );
    let shifted = vectorize(
        s.sql(
            "SELECT array(CAST(x AS DOUBLE)) AS xs, 1.0D AS weight
         FROM VALUES (9), (10), (11), (29), (30), (31) AS t(x)",
        )
        .unwrap(),
        "input",
    );
    let held_out = vectorize(
        s.sql(
            "SELECT id, array(CAST(x AS DOUBLE)) AS xs
         FROM VALUES (0,-10), (1,-1), (2,10), (3,21), (4,30) AS t(id,x)",
        )
        .unwrap(),
        "input",
    );
    let mut estimator = KMeans::new()
        .set_features_col("input")
        .set_prediction_col("cluster")
        .set_k(2)
        .set_init_mode("k-means||")
        .set_init_steps(3)
        .set_tol(1e-6)
        .set_max_iter(50)
        .set_seed(42)
        .set_distance_measure("euclidean")
        .set_weight_col("weight")
        .set_solver("row")
        .set_max_block_size_in_mb(1.0);
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
fn kmeans_weights_move_the_boundary_with_default_columns() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectorize(
        s.sql(
            "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(w AS DOUBLE) AS weight
         FROM VALUES (0,1), (2,9), (10,1), (12,9) AS t(x,w)",
        )
        .unwrap(),
        "features",
    );
    let held_out = vectorize(s.sql(
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs FROM VALUES (0,0), (1,6.5), (2,12) AS t(id,x)"
    ).unwrap(), "features");
    let mut unweighted = KMeans::default().set_seed(42).fit(&training).unwrap();
    let mut weighted = KMeans::new()
        .set_seed(42)
        .set_weight_col("weight")
        .fit(&training)
        .unwrap();
    assert_partition(
        &predictions(unweighted.transform(&held_out).unwrap(), "prediction"),
        &[0, 1, 1],
    );
    assert_partition(
        &predictions(weighted.transform(&held_out).unwrap(), "prediction"),
        &[0, 0, 1],
    );
    s.stop().unwrap();
}

#[test]
fn kmeans_block_solver_random_initialization_and_three_clusters() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectorize(s.sql(
        "SELECT array(CAST(x AS DOUBLE)) AS xs FROM VALUES (-11),(-10),(-9),(-1),(0),(1),(9),(10),(11) AS t(x)"
    ).unwrap(), "features");
    let held_out = vectorize(s.sql(
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs FROM VALUES (0,-10.5),(1,-9.5),(2,0.5),(3,9.5),(4,10.5) AS t(id,x)"
    ).unwrap(), "features");
    let mut model = KMeans::new()
        .set_k(3)
        .set_solver("block")
        .set_init_mode("random")
        .set_max_iter(50)
        .set_seed(42)
        .set_max_block_size_in_mb(1.0)
        .fit(&training)
        .unwrap();
    assert_partition(
        &predictions(model.transform(&held_out).unwrap(), "prediction"),
        &[0, 0, 1, 2, 2],
    );
    s.stop().unwrap();
}

#[test]
fn kmeans_cosine_distance_groups_vectors_by_direction() {
    let Some(s) = live_session() else {
        return;
    };
    let training = vectorize(s.sql(
        "SELECT array(CAST(x AS DOUBLE),CAST(y AS DOUBLE)) AS xs FROM VALUES (1,0),(2,0),(0,1),(0,2) AS t(x,y)"
    ).unwrap(), "features");
    let held_out = vectorize(
        s.sql(
            "SELECT id, array(CAST(x AS DOUBLE),CAST(y AS DOUBLE)) AS xs
         FROM VALUES (0,100,1),(1,1,0.01),(2,1,100),(3,0.01,1) AS t(id,x,y)",
        )
        .unwrap(),
        "features",
    );
    let mut model = KMeans::new()
        .set_seed(42)
        .set_distance_measure("cosine")
        .set_solver("row")
        .fit(&training)
        .unwrap();
    assert_partition(
        &predictions(model.transform(&held_out).unwrap(), "prediction"),
        &[0, 0, 1, 1],
    );
    s.stop().unwrap();
}

#[test]
fn kmeans_fit_rejects_invalid_data_and_parameters() {
    let Some(s) = live_session() else {
        return;
    };
    assert!(KMeans::new().fit(&s.range(3).unwrap()).is_err());
    let training = vectorize(
        s.sql("SELECT array(CAST(id AS DOUBLE)) AS xs, 1.0D AS weight FROM range(5)")
            .unwrap(),
        "features",
    );
    for mut estimator in [
        KMeans::new().set_k(1),
        KMeans::new().set_max_iter(-1),
        KMeans::new().set_init_steps(0),
        KMeans::new().set_distance_measure("invalid"),
        KMeans::new().set_solver("invalid"),
    ] {
        assert!(estimator.fit(&training).is_err());
    }
    assert!(KMeans::new()
        .set_weight_col("weight")
        .fit(&training.with_column("weight", lit(-1)))
        .is_err());
    s.stop().unwrap();
}
