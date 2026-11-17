//! Live numerical and ownership checks for the remaining native classifiers.
mod common;
use common::live_session;
use spark_connect::column::{col, lit};
use spark_connect::expression::Expression;
use spark_connect::ml::{
    Estimator, FMClassifier, LinearSVC, MlVector, Model, MultilayerPerceptronClassifier, OneVsRest,
    Transformer,
};
use spark_connect::row::Value;
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
    // The max label, of which a fit takes its classes, as PySpark's `_fit` reads it: the field
    // `max(label)` of `dataset.agg({"label": "max"}).head()`, of the label's own type; a null of
    // no rows, and the server's error of a label column that is not there.
    assert_eq!(estimator.max_label(&frame).unwrap(), Value::Double(1.0));
    assert_eq!(estimator.max_label(&frame.limit(0)).unwrap(), Value::Null);
    let missing = estimator
        .max_label(&frame.select([col("features")]))
        .err()
        .unwrap();
    assert!(
        missing.error_class.starts_with("UNRESOLVED_COLUMN"),
        "{missing}"
    );
    // A string label is its greatest string, as the server's `max` of the column as it is.
    let strings = vectors(
        s.sql("SELECT array(1.0D) xs, label FROM VALUES ('10'), ('9'), ('0') AS t(label)")
            .unwrap(),
    );
    assert_eq!(
        estimator.max_label(&strings).unwrap(),
        Value::String("9".into())
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
/// The first line of the message of the error of a fit of a OneVsRest.
fn fit_error(estimator: &mut OneVsRest, df: &DataFrame) -> String {
    match estimator.fit(df) {
        Ok(_) => panic!("the fit succeeded"),
        Err(error) => error
            .to_string()
            .lines()
            .next()
            .unwrap_or_default()
            .to_string(),
    }
}

#[test]
fn one_vs_rest_fits_the_classes_of_pythons_int_of_the_max_label_as_pyspark() {
    // PySpark's `_fit` fits `int(max) + 1` classes, of Python's `int` of the max label that it
    // collects, and then makes its pool of `min(parallelism, numClasses)` threads, after the
    // dataset is cached: the messages are official PySpark 4.3.0's of the same fits.
    let Some(s) = live_session() else {
        return;
    };
    let frame = |label: &str, rows: i64| {
        vectors(
            s.sql(&format!(
                "SELECT array(CAST(id % 3 AS DOUBLE), CAST(id % 2 AS DOUBLE)) AS xs, {label} AS \
                 label FROM range({rows})"
            ))
            .unwrap(),
        )
    };
    let estimator = || {
        OneVsRest::new()
            .set_classifier(spark_connect::ml::LogisticRegression::new().set_max_iter(2))
    };
    let none = "int() argument must be a string, a bytes-like object or a real number, not \
                'NoneType'";
    for (label, rows, message) in [
        ("CAST(NULL AS DOUBLE)", 4, none),
        ("CAST(id AS DOUBLE)", 0, none),
        (
            "CASE WHEN id = 3 THEN double('NaN') ELSE CAST(id % 2 AS DOUBLE) END",
            4,
            "cannot convert float NaN to integer",
        ),
        (
            "CASE WHEN id = 3 THEN double('Infinity') ELSE CAST(id % 2 AS DOUBLE) END",
            4,
            "cannot convert float infinity to integer",
        ),
        (
            "CAST(-1 - id AS DOUBLE)",
            4,
            "Number of processes must be at least 1",
        ),
        (
            "element_at(array('b', 'a'), CAST(id % 2 + 1 AS INT))",
            4,
            "invalid literal for int() with base 10: 'b'",
        ),
    ] {
        assert_eq!(
            fit_error(&mut estimator(), &frame(label, rows)),
            message,
            "{label}"
        );
    }
    // A fraction is truncated toward zero: a max label of -0.5 fits one class.
    let model = estimator()
        .fit(&frame("CAST(-0.5 - id AS DOUBLE)", 4))
        .unwrap();
    assert_eq!(model.num_classes(), 1);
    // The max of a string column is its greatest string, "9" of these: ten classes, so that the
    // label "10" is of no class, and its row is predicted of class 0, as PySpark predicts.
    let strings = vectors(
        s.sql(
            "SELECT array(1.0D, 0.0D) AS xs, '10' AS label UNION ALL \
             SELECT array(0.0D, 1.0D), '9' UNION ALL SELECT array(1.0D, 1.0D), '0'",
        )
        .unwrap(),
    );
    let mut model = estimator().fit(&strings).unwrap();
    assert_eq!(model.num_classes(), 10);
    let predictions: Vec<f64> = model
        .transform(&strings)
        .unwrap()
        .select([col("prediction")])
        .collect()
        .unwrap()
        .iter()
        .map(|row| row.get(0).unwrap().as_f64().unwrap())
        .collect();
    assert_eq!(predictions, vec![0.0, 9.0, 0.0]);
    // The label column is the column of its name as given, which the server parses: a dot names
    // a field of a struct, so `my.label` does not resolve, as PySpark's max query does not.
    let dotted = frame("CAST(id % 2 AS DOUBLE)", 4).with_column_renamed("label", "my.label");
    let error = estimator()
        .set_label_col("my.label")
        .fit(&dotted)
        .err()
        .unwrap();
    assert!(
        error.error_class.starts_with("UNRESOLVED_COLUMN"),
        "{error}"
    );
    // A parallelism of no thread, or a null one, fails after the max label is read and the dataset
    // is cached, as PySpark's pool does: the error of the labels comes first.
    let labels = frame("CAST(id % 2 AS DOUBLE)", 4);
    let mut zero = estimator().set_parallelism(0);
    assert_eq!(
        fit_error(&mut zero, &labels),
        "Number of processes must be at least 1"
    );
    assert_eq!(fit_error(&mut zero, &labels.limit(0)), none);
    let mut null = estimator();
    *null.params_mut() = null.params().clone().set_param_null("parallelism");
    assert_eq!(
        fit_error(&mut null, &labels),
        "'<' not supported between instances of 'int' and 'NoneType'"
    );
    // A OneVsRest of no classifier raises PySpark's KeyError of the param, before any request.
    let mut unset = OneVsRest::new();
    let uid = Estimator::operator(&unset).uid.clone();
    assert_eq!(
        fit_error(&mut unset, &labels),
        format!("Param(parent='{uid}', name='classifier', doc='base binary classifier')")
    );
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
