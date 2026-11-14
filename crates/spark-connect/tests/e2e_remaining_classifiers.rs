//! Live numerical and ownership checks for the remaining native classifiers.
mod common;
use common::live_session;
use spark_connect::column::{col, lit};
use spark_connect::expression::Expression;
use spark_connect::ml::{
    Estimator, FMClassifier, LinearSVC, MlVector, Model, MultilayerPerceptronClassifier, OneVsRest,
    Transformer,
};
use spark_connect::{Column, DataFrame};
use std::sync::Arc;

fn vectors(df: DataFrame) -> DataFrame {
    df.with_column(
        "features",
        Column::new(Expression::function(
            "array_to_vector",
            vec![col("xs").expression().clone()],
        )),
    )
}
fn predictions(df: DataFrame) -> Vec<f64> {
    df.order_by([col("id")])
        .select([col("prediction")])
        .collect()
        .unwrap()
        .iter()
        .map(|r| r.get(0).unwrap().as_f64().unwrap())
        .collect()
}
#[test]
fn fm_classifier_learns_interactions_and_owns_lazy_models() {
    let Some(s) = live_session() else {
        return;
    };
    let frame=vectors(s.sql("SELECT id,array(x,y) xs,label FROM VALUES (0,-1.0D,-1.0D,1.0D),(1,-1.0D,1.0D,0.0D),(2,1.0D,-1.0D,0.0D),(3,1.0D,1.0D,1.0D) AS t(id,x,y,label)").unwrap());
    let mut model = FMClassifier::new()
        .set_factor_size(2)
        .set_fit_linear(false)
        .set_init_std(0.1)
        .set_step_size(0.1)
        .set_max_iter(200)
        .set_seed(42)
        .fit(&frame)
        .unwrap();
    let mut cloned = model.clone_box();
    let lazy = model.transform(&frame).unwrap();
    drop(model);
    assert_eq!(
        predictions(cloned.transform(&frame).unwrap()),
        vec![1.0, 0.0, 0.0, 1.0]
    );
    assert!(cloned
        .transform(&s.new_session().range(1).unwrap())
        .is_err());
    drop(cloned);
    assert_eq!(predictions(lazy), vec![1.0, 0.0, 0.0, 1.0]);
    s.stop().unwrap();
}
#[test]
fn multilayer_perceptron_uses_dense_and_sparse_initial_weights() {
    let Some(s) = live_session() else {
        return;
    };
    let frame=vectors(s.sql("SELECT id,array(x,y) xs,label FROM VALUES (0,-2.0D,-2.0D,0.0D),(1,2.0D,-2.0D,1.0D),(2,0.0D,2.0D,2.0D) AS t(id,x,y,label)").unwrap());
    for weights in [
        MlVector::Dense((0..9).map(|i| i as f64 * 0.01).collect()),
        MlVector::Sparse {
            size: 9,
            indices: vec![0, 8],
            values: vec![0.1, -0.1],
        },
    ] {
        let mut model = MultilayerPerceptronClassifier::new()
            .set_layers([2, 3])
            .set_initial_weights(weights)
            .set_max_iter(100)
            .set_seed(1_i64 << 33)
            .fit(&frame)
            .unwrap();
        let lazy = model.transform(&frame).unwrap();
        drop(model);
        assert_eq!(predictions(lazy), vec![0.0, 1.0, 2.0]);
    }
    s.stop().unwrap();
}
#[test]
fn one_vs_rest_parallel_models_and_ml_relations_keep_submodels_alive() {
    let Some(s) = live_session() else {
        return;
    };
    let frame=vectors(s.sql("SELECT id,array(x,y) xs,label FROM VALUES (0,-2.0D,-2.0D,0.0D),(1,2.0D,-2.0D,1.0D),(2,0.0D,2.0D,2.0D) AS t(id,x,y,label)").unwrap());
    for parallelism in [1, 3] {
        let mut estimator = OneVsRest::new()
            .set_classifier(LinearSVC::new().set_max_iter(100))
            .set_parallelism(parallelism);
        let mut model = estimator.fit(&frame).unwrap();
        let mut other = estimator
            .fit(&frame.with_column("label", (col("label") + lit(1)) % lit(3)))
            .unwrap();
        let lazy = model.transform(&frame).unwrap();
        let relation = model.build_ml_relation(&frame).unwrap();
        let identity = DataFrame::from_relation(
            s.clone(),
            spark_connect_proto::Relation {
                rel_type: Some(spark_connect_proto::relation::RelType::MlRelation(
                    Box::new(relation),
                )),
                ..Default::default()
            },
            Some(Arc::new(model.clone_box())),
        )
        .unwrap();
        assert!(model.transform(&s.new_session().range(1).unwrap()).is_err());
        drop(model);
        assert_eq!(predictions(lazy), vec![0.0, 1.0, 2.0]);
        assert_eq!(predictions(identity), vec![0.0, 1.0, 2.0]);
        assert_eq!(
            predictions(other.transform(&frame).unwrap()),
            vec![1.0, 2.0, 0.0]
        );
    }
    s.stop().unwrap();
}
#[test]
fn one_vs_rest_reads_the_max_label_and_takes_a_parallelism_of_either_int_width() {
    let Some(s) = live_session() else {
        return;
    };
    let frame = vectors(
        s.sql("SELECT array(1.0D,2.0D) xs,0.0D label UNION ALL SELECT array(-1.0D,0.5D),1.0D")
            .unwrap(),
    );
    let mut estimator = OneVsRest::new().set_classifier(LinearSVC::new().set_max_iter(5));
    // The max label, of which the Python client takes the classes before PySpark's checks: None
    // of no rows, and the server's error of a label column that is not there.
    assert_eq!(estimator.max_label(&frame).unwrap(), Some(1.0));
    assert_eq!(estimator.max_label(&frame.limit(0)).unwrap(), None);
    let missing = estimator
        .max_label(&frame.select([col("features")]))
        .err()
        .unwrap();
    assert!(
        missing.error_class.starts_with("UNRESOLVED_COLUMN"),
        "{missing}"
    );
    // A parallelism beyond the classes, as the Python client sends an int beyond 32 bits, fits
    // one thread of each class rather than wrapping.
    *estimator.params_mut() = estimator
        .params()
        .clone()
        .set_param_int_narrow("parallelism", 1 << 40);
    estimator.fit(&frame).unwrap();
    s.stop().unwrap();
}
#[test]
fn new_classifiers_reject_invalid_parameters_and_missing_one_vs_rest_classifier() {
    let Some(s) = live_session() else {
        return;
    };
    let frame = vectors(s.sql("SELECT array(1.0D,2.0D) xs,0.0D label").unwrap());
    assert!(FMClassifier::new().set_factor_size(0).fit(&frame).is_err());
    assert!(MultilayerPerceptronClassifier::new()
        .set_layers([2])
        .fit(&frame)
        .is_err());
    assert!(OneVsRest::new().fit(&frame).is_err());
    assert!(OneVsRest::new()
        .set_classifier(LinearSVC::new())
        .set_parallelism(0)
        .fit(&frame)
        .is_err());
    assert!(OneVsRest::new()
        .set_classifier(LinearSVC::new().set_max_iter(-1))
        .set_parallelism(2)
        .fit(&frame)
        .is_err());
    s.stop().unwrap();
}
