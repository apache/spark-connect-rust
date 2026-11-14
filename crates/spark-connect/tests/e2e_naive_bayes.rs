//! Native NaiveBayes behaviors. Set SPARK_REMOTE to use a live Connect server.
mod common;

use common::live_session;
use spark_connect::column::{col, lit_double, lit_string};
use spark_connect::expression::Expression;
use spark_connect::ml::{Model, NaiveBayes, Transformer};
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
        .map(|r| r.get(0).unwrap().as_f64().unwrap())
        .collect()
}

fn list(value: &spark_connect::row::Value) -> &[spark_connect::row::Value] {
    let spark_connect::row::Value::List(values) = value else {
        panic!("array result")
    };
    values
}

fn weighted_training(s: &SparkSession) -> DataFrame {
    vectors(s, "SELECT array(1.0D,1.0D) xs, label, weight FROM VALUES (0.0D,1.0D),(1.0D,9.0D) AS t(label,weight)", "features")
}

#[test]
fn naive_bayes_model_types_learn_multiclass_labels_and_keep_independent_cached_models() {
    let Some(s) = live_session() else {
        return;
    };
    for model_type in ["multinomial", "bernoulli", "complement", "gaussian"] {
        let (training_sql, test_sql) = match model_type {
            "gaussian" => (
                "SELECT target, array(x,y) xs, 1.0D weight FROM VALUES (0.0D,-3.0D,-2.0D),(0.0D,-2.0D,-3.0D),(1.0D,2.0D,3.0D),(1.0D,3.0D,2.0D),(2.0D,7.0D,8.0D),(2.0D,8.0D,7.0D) AS t(target,x,y)",
                "SELECT id, array(x,y) xs FROM VALUES (0,-2.5D,-2.5D),(1,2.5D,2.5D),(2,7.5D,7.5D) AS t(id,x,y)"),
            "bernoulli" => (
                "SELECT target, array(x,y,z) xs, 1.0D weight FROM VALUES (0.0D,1.0D,0.0D,0.0D),(0.0D,1.0D,1.0D,0.0D),(1.0D,0.0D,1.0D,0.0D),(1.0D,0.0D,1.0D,1.0D),(2.0D,0.0D,0.0D,1.0D),(2.0D,1.0D,0.0D,1.0D) AS t(target,x,y,z)",
                "SELECT id, array(x,y,z) xs FROM VALUES (0,1.0D,0.0D,0.0D),(1,0.0D,1.0D,0.0D),(2,0.0D,0.0D,1.0D) AS t(id,x,y,z)"),
            _ => (
                "SELECT target, array(x,y,z) xs, 1.0D weight FROM VALUES (0.0D,4.0D,1.0D,0.0D),(0.0D,5.0D,0.0D,1.0D),(1.0D,0.0D,4.0D,1.0D),(1.0D,1.0D,5.0D,0.0D),(2.0D,1.0D,0.0D,4.0D),(2.0D,0.0D,1.0D,5.0D) AS t(target,x,y,z)",
                "SELECT id, array(x,y,z) xs FROM VALUES (0,4.0D,0.0D,0.0D),(1,0.0D,4.0D,0.0D),(2,0.0D,0.0D,4.0D) AS t(id,x,y,z)"),
        };
        let training = vectors(&s, training_sql, "input");
        let held_out = vectors(&s, test_sql, "input");
        let mut estimator = NaiveBayes::new()
            .set_features_col("input")
            .set_label_col("target")
            .set_prediction_col("class")
            .set_probability_col("probs")
            .set_raw_prediction_col("scores")
            .set_smoothing(0.5)
            .set_model_type(model_type)
            .set_thresholds([1.0, 1.0, 1.0])
            .set_weight_col("weight");
        let mut first = estimator.fit(&training).unwrap();
        let mut second = estimator
            .fit(&training.with_column("target", lit_double(2.0) - col("target")))
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
        assert!(first.transform(&s.new_session().range(1).unwrap()).is_err());
        let mut cloned = first.clone_box();
        let lazy = cloned.transform(&held_out).unwrap();
        drop(first);
        drop(cloned);
        assert_eq!(
            predictions(lazy.clone(), "class"),
            vec![0.0, 1.0, 2.0],
            "{model_type}"
        );
        assert_eq!(
            predictions(second.transform(&held_out).unwrap(), "class"),
            vec![2.0, 1.0, 0.0],
            "{model_type}"
        );
        let rows = lazy
            .select([vector_array("probs"), vector_array("scores")])
            .collect()
            .unwrap();
        for row in rows {
            let probabilities = list(row.get(0).unwrap());
            let raw = list(row.get(1).unwrap());
            assert_eq!(probabilities.len(), 3);
            assert_eq!(raw.len(), 3);
            let values: Vec<f64> = probabilities.iter().map(|v| v.as_f64().unwrap()).collect();
            assert!(values
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
            assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-10);
            assert!(raw.iter().all(|v| v.as_f64().unwrap().is_finite()));
        }
    }
}

#[test]
fn naive_bayes_weights_and_smoothing_produce_the_expected_priors() {
    let Some(s) = live_session() else {
        return;
    };
    let data = weighted_training(&s);
    for smoothing in [0.0, 1.0, 10.0] {
        for weighted in [false, true] {
            let mut estimator = NaiveBayes::new().set_smoothing(smoothing);
            if weighted {
                estimator = estimator.set_weight_col("weight");
            }
            let mut model = estimator.fit(&data).unwrap();
            let rows = model
                .transform(&data)
                .unwrap()
                .select([
                    col("prediction"),
                    vector_array("probability"),
                    vector_array("rawPrediction"),
                ])
                .collect()
                .unwrap();
            let prior: f64 = if weighted {
                (1.0 + smoothing) / (10.0 + 2.0 * smoothing)
            } else {
                0.5
            };
            assert_eq!(
                rows[0].get(0).unwrap().as_f64().unwrap(),
                if weighted { 1.0 } else { 0.0 }
            );
            let p = list(rows[0].get(1).unwrap());
            let r = list(rows[0].get(2).unwrap());
            for (i, expected) in [prior, 1.0 - prior].into_iter().enumerate() {
                assert!((p[i].as_f64().unwrap() - expected).abs() < 1e-10);
                assert!(
                    (r[i].as_f64().unwrap() - (expected.ln() - 2.0 * 2.0_f64.ln())).abs() < 1e-10
                );
            }
        }
    }
}

#[test]
fn naive_bayes_thresholds_and_optional_vector_outputs() {
    let Some(s) = live_session() else {
        return;
    };
    let data = weighted_training(&s);
    let mut normal = NaiveBayes::new()
        .set_weight_col("weight")
        .fit(&data)
        .unwrap();
    let mut adjusted = NaiveBayes::new()
        .set_weight_col("weight")
        .set_thresholds([0.1, 1.0])
        .fit(&data)
        .unwrap();
    let outputs = |model: &mut dyn spark_connect::ml::Model| {
        model
            .transform(&data)
            .unwrap()
            .select([
                col("prediction"),
                vector_array("probability"),
                vector_array("rawPrediction"),
            ])
            .collect()
            .unwrap()
    };
    let a = outputs(&mut normal);
    let b = outputs(&mut adjusted);
    assert_eq!(a[0].get(0).unwrap().as_f64(), Some(1.0));
    assert_eq!(b[0].get(0).unwrap().as_f64(), Some(0.0));
    assert_eq!(a[0].get(1), b[0].get(1));
    assert_eq!(a[0].get(2), b[0].get(2));
    let mut model = NaiveBayes::new()
        .set_weight_col("weight")
        .set_prediction_col("class")
        .set_probability_col("")
        .set_raw_prediction_col("")
        .fit(&data)
        .unwrap();
    let transformed = model.transform(&data).unwrap();
    let names = transformed.columns().unwrap();
    assert!(!names
        .iter()
        .any(|n| n == "probability" || n == "rawPrediction"));
    assert_eq!(
        transformed.select([col("class")]).collect().unwrap()[0]
            .get(0)
            .unwrap()
            .as_f64(),
        Some(1.0)
    );
}

#[test]
fn naive_bayes_rejects_invalid_parameters_and_data() {
    let Some(s) = live_session() else {
        return;
    };
    let data = weighted_training(&s);
    for mut estimator in [
        NaiveBayes::new().set_smoothing(-1.0),
        NaiveBayes::new().set_model_type("unknown"),
        NaiveBayes::new().set_thresholds([1.0]),
    ] {
        match estimator.fit(&data) {
            Err(_) => (),
            Ok(mut model) => assert!(model.transform(&data).and_then(|df| df.collect()).is_err()),
        }
    }
    for (model_type, values) in [
        ("multinomial", "array(-1.0D,1.0D)"),
        ("complement", "array(-1.0D,1.0D)"),
        ("bernoulli", "array(2.0D,1.0D)"),
    ] {
        let invalid = vectors(
            &s,
            &format!("SELECT {values} xs, label FROM VALUES (0.0D),(1.0D) AS t(label)"),
            "features",
        );
        assert!(NaiveBayes::new()
            .set_model_type(model_type)
            .fit(&invalid)
            .is_err());
    }
    assert!(NaiveBayes::new()
        .set_weight_col("weight")
        .fit(&data.with_column("weight", lit_double(-1.0)))
        .is_err());
}
