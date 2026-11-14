//! Native GaussianMixture tests. Set SPARK_REMOTE to use a live Connect server.
mod common;

use common::live_session;
use spark_connect::column::{col, lit_double, lit_string};
use spark_connect::expression::Expression;
use spark_connect::ml::{GaussianMixture, Model, Transformer};
use spark_connect::row::Value;
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

fn training(s: &SparkSession, shift: i32, name: &str) -> DataFrame {
    vectors(s,&format!("SELECT array(CASE WHEN id<40 THEN -5.0D ELSE 5.0D END + {shift} + sin(id)*0.5D) xs, 1.0D weight FROM range(80)"),name)
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

fn outputs(df: DataFrame, prediction: &str, probability: &str) -> Vec<(i64, Vec<f64>)> {
    df.order_by(vec![col("id").asc().expression().clone()])
        .select([col(prediction), vector_array(probability)])
        .collect()
        .unwrap()
        .into_iter()
        .map(|r| {
            let label = r.get(0).unwrap().as_i64().unwrap();
            let Value::List(p) = r.get(1).unwrap() else {
                panic!("probability array")
            };
            (label, p.iter().map(|v| v.as_f64().unwrap()).collect())
        })
        .collect()
}

fn assert_partition(rows: &[(i64, Vec<f64>)], expected: &[i64]) {
    assert_eq!(rows.len(), expected.len());
    for i in 0..rows.len() {
        for j in 0..rows.len() {
            assert_eq!(
                rows[i].0 == rows[j].0,
                expected[i] == expected[j],
                "{rows:?}"
            );
        }
    }
}

#[test]
fn gaussian_mixture_fits_independent_models_and_retains_lazy_predictions() {
    let Some(s) = live_session() else {
        return;
    };
    let data = training(&s, 0, "input");
    let held = vectors(
        &s,
        "SELECT id,array(x) xs FROM VALUES (0,-5.0D),(1,4.5D),(2,5.5D),(3,15.0D) AS t(id,x)",
        "input",
    );
    let mut estimator = GaussianMixture::new()
        .set_features_col("input")
        .set_prediction_col("cluster")
        .set_probability_col("probs")
        .set_k(2)
        .set_tol(1e-5)
        .set_max_iter(100)
        .set_seed(42)
        .set_aggregation_depth(3)
        .set_weight_col("weight");
    let mut first = estimator.fit(&data).unwrap();
    let mut second = estimator.fit(&training(&s, 10, "input")).unwrap();
    let object_id = |model: &dyn spark_connect::ml::Model| {
        let spark_connect_proto::ml_relation::MlType::Transform(t) =
            model.build_ml_relation(&held).unwrap().ml_type.unwrap()
        else {
            panic!("transform")
        };
        let Some(spark_connect_proto::ml_relation::transform::Operator::ObjRef(r)) = t.operator
        else {
            panic!("cached reference")
        };
        r.id
    };
    assert_ne!(object_id(&first), object_id(&second));
    assert!(first.transform(&s.new_session().range(1).unwrap()).is_err());
    let mut cloned = first.clone_box();
    let lazy = cloned.transform(&held).unwrap();
    drop(first);
    drop(cloned);
    assert_partition(&outputs(lazy, "cluster", "probs"), &[0, 1, 1, 1]);
    assert_partition(
        &outputs(second.transform(&held).unwrap(), "cluster", "probs"),
        &[0, 0, 0, 1],
    );
}

#[test]
fn gaussian_mixture_multivariate_predictions_have_normalized_probabilities() {
    let Some(s) = live_session() else {
        return;
    };
    let data=vectors(&s,"SELECT array(CASE WHEN id<40 THEN -5.0D ELSE 5.0D END + sin(id)*0.5D, CASE WHEN id<40 THEN -3.0D ELSE 3.0D END + cos(id)*0.7D) xs FROM range(80)","features");
    let held=vectors(&s,"SELECT id,array(x,y) xs FROM VALUES (0,-5.0D,-3.0D),(1,-4.5D,-3.5D),(2,5.0D,3.0D),(3,4.5D,3.5D) AS t(id,x,y)","features");
    let mut model = GaussianMixture::new()
        .set_seed(42)
        .set_tol(1e-5)
        .fit(&data)
        .unwrap();
    let rows = outputs(model.transform(&held).unwrap(), "prediction", "probability");
    assert_partition(&rows, &[0, 0, 1, 1]);
    for (label, p) in rows {
        assert_eq!(p.len(), 2);
        assert!(p.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)));
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        let best = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(label, best as i64);
    }
}

#[test]
fn gaussian_mixture_three_components_are_reproducible_with_a_wide_seed() {
    let Some(s) = live_session() else {
        return;
    };
    let data=vectors(&s,"SELECT array(CASE WHEN id<40 THEN -8.0D WHEN id<80 THEN 0.0D ELSE 8.0D END + sin(id)*0.5D) xs FROM range(120)","features");
    let held=vectors(&s,"SELECT id,array(x) xs FROM VALUES (0,-8.0D),(1,-7.5D),(2,-0.2D),(3,0.2D),(4,7.5D),(5,8.0D) AS t(id,x)","features");
    let mut estimator = GaussianMixture::new()
        .set_k(3)
        .set_seed(1_i64 << 33)
        .set_tol(1e-6)
        .set_max_iter(100);
    let a = outputs(
        estimator.fit(&data).unwrap().transform(&held).unwrap(),
        "prediction",
        "probability",
    );
    let b = outputs(
        estimator.fit(&data).unwrap().transform(&held).unwrap(),
        "prediction",
        "probability",
    );
    assert_partition(&a, &[0, 0, 1, 1, 2, 2]);
    for ((label, p), (other, q)) in a.into_iter().zip(b) {
        assert_eq!(label, other);
        assert_eq!(p.len(), 3);
        assert_eq!(q.len(), 3);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        for (x, y) in p.into_iter().zip(q) {
            assert!((x - y).abs() < 1e-10);
        }
    }
}

#[test]
fn gaussian_mixture_weights_change_component_probabilities() {
    let Some(s) = live_session() else {
        return;
    };
    let data=vectors(&s,"SELECT array(x) xs,CASE WHEN x<0 THEN 9.0D ELSE 1.0D END weight FROM VALUES (-5.0D),(-4.5D),(-4.0D),(-3.5D),(-3.0D),(3.0D),(3.5D),(4.0D),(4.5D),(5.0D) AS t(x)","features");
    let held = vectors(
        &s,
        "SELECT id,array(x) xs FROM VALUES (0,-4.0D),(1,0.0D),(2,4.0D) AS t(id,x)",
        "features",
    );
    for weighted in [false, true] {
        let mut estimator = GaussianMixture::new().set_seed(42).set_tol(1e-8);
        if weighted {
            estimator = estimator.set_weight_col("weight");
        }
        let rows = outputs(
            estimator.fit(&data).unwrap().transform(&held).unwrap(),
            "prediction",
            "probability",
        );
        assert_ne!(rows[0].0, rows[2].0);
        let left = rows[0].0 as usize;
        let expected = if weighted { 0.9 } else { 0.5 };
        assert!((rows[1].1[left] - expected).abs() < 0.002);
    }
}

#[test]
fn gaussian_mixture_supports_disabled_probabilities_and_rejects_invalid_inputs() {
    let Some(s) = live_session() else {
        return;
    };
    let data = training(&s, 0, "features");
    let mut model = GaussianMixture::new()
        .set_seed(42)
        .set_probability_col("")
        .set_prediction_col("cluster")
        .fit(&data)
        .unwrap();
    let transformed = model.transform(&data).unwrap();
    assert!(!transformed
        .columns()
        .unwrap()
        .iter()
        .any(|name| name == "probability"));
    assert_eq!(
        transformed
            .select([col("cluster")])
            .collect()
            .unwrap()
            .len(),
        80
    );
    for mut estimator in [
        GaussianMixture::new().set_k(1),
        GaussianMixture::new().set_max_iter(-1),
        GaussianMixture::new().set_tol(-1.0),
        GaussianMixture::new().set_aggregation_depth(1),
    ] {
        assert!(estimator.fit(&data).is_err());
    }
    assert!(GaussianMixture::new()
        .set_weight_col("weight")
        .fit(&data.with_column("weight", lit_double(-1.0)))
        .is_err());
    assert!(GaussianMixture::new()
        .fit(&data.with_column("features", lit_double(1.0)))
        .is_err());
    let empty = vectors(&s, "SELECT array(1.0D) xs WHERE false", "features");
    assert!(GaussianMixture::new().fit(&empty).is_err());
}
