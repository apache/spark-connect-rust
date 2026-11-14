// SPDX-License-Identifier: Apache-2.0
//! The server-side objects of Spark Connect ML: a fetch of a model's methods, of its summary and of
//! the ML helper, the DataFrames of a summary, the trees of an ensemble, and the save and load of
//! models and estimators, as PySpark 4.3.0 reads and writes them. Set SPARK_REMOTE to run against
//! a live Spark Connect server.
mod common;

use common::live_session;
use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::ml::{
    attribute_relation, evaluated_summary, fetch, read, read_model, read_operator,
    summary_transform, training_summary, write_object, write_operator, Estimator, FetchArg, KMeans,
    LinearSVC, LinearSVCModel, MlTarget, MlValue, MlVector, ObjectModel, OperatorType,
    RandomForestClassifier, Transformer,
};
use spark_connect::{Column, DataFrame, SparkSession};

fn training(s: &SparkSession) -> DataFrame {
    s.sql(
        "SELECT array(CAST(x AS DOUBLE), 1.0D) AS xs, \
         CAST(CASE WHEN x > 0 THEN 1 ELSE 0 END AS DOUBLE) AS label \
         FROM VALUES (-4), (-3), (-2), (-1), (1), (2), (3), (4) AS t(x)",
    )
    .unwrap()
    .with_column(
        "features",
        Column::new(Expression::function(
            "array_to_vector",
            vec![col("xs").expression().clone()],
        )),
    )
}

fn scratch_path(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "spark-connect-rust-ml-object-{}-{name}",
        std::process::id()
    ));
    dir.to_string_lossy().into_owned()
}

#[test]
fn ml_objects_fetch_their_methods_summaries_and_helpers_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let train = training(&s);
    let model = LinearSVC::new().set_max_iter(5).fit(&train).unwrap();
    let object = model
        .ml_ref()
        .expect("a LinearSVC model is a server-side object");
    let target = MlTarget::object(object);

    // Values of the model, as LinearSVCModel's coefficients, intercept, numFeatures and numClasses.
    let coefficients = fetch(&target, "coefficients", vec![])
        .unwrap()
        .into_value()
        .unwrap();
    assert!(matches!(&coefficients, MlValue::Vector(MlVector::Dense(v)) if v.len() == 2));
    assert!(fetch(&target, "intercept", vec![])
        .unwrap()
        .into_value()
        .unwrap()
        .as_f64()
        .is_some());
    let num_features = fetch(&target, "numFeatures", vec![]).unwrap().into_value();
    assert_eq!(num_features.unwrap(), MlValue::Int(2));
    let num_classes = fetch(&target, "numClasses", vec![]).unwrap().into_value();
    assert_eq!(num_classes.unwrap(), MlValue::Int(2));
    // A method of an argument, as predict(value) sends the vector.
    let prediction = fetch(
        &target,
        "predict",
        vec![FetchArg::Value(MlValue::Vector(MlVector::Dense(vec![
            3.0, 1.0,
        ])))],
    )
    .unwrap()
    .into_value()
    .unwrap();
    assert_eq!(prediction, MlValue::Double(1.0));

    // The training summary, reached by its chain, and its DataFrame, of the summary dataset.
    let mut predicting = model.clone_box();
    let summary_dataset = predicting.transform(&train).unwrap();
    let summary = target
        .child("summary")
        .with_summary_dataset(&summary_dataset);
    assert_eq!(summary.id(), format!("{}.summary", object.id()));
    let accuracy = fetch(&summary, "accuracy", vec![]).unwrap().into_value();
    assert_eq!(accuracy.unwrap(), MlValue::Double(1.0));
    let predictions = attribute_relation(&summary, "predictions", vec![]).unwrap();
    assert_eq!(predictions.count().unwrap(), 8);

    // The summary of an evaluate, whose id the server returns, as LinearSVCModel.evaluate; the
    // client does not release it, as PySpark does not.
    let evaluated = fetch(&target, "evaluate", vec![FetchArg::Input(train.clone())])
        .unwrap()
        .into_string()
        .unwrap();
    let evaluation = MlTarget::unowned(&s, &evaluated);
    let accuracy = fetch(&evaluation, "accuracy", vec![]).unwrap().into_value();
    assert_eq!(accuracy.unwrap(), MlValue::Double(1.0));

    // A method of the ML helper, as StopWordsRemover.loadDefaultStopWords fetches it.
    let words = fetch(
        &MlTarget::helper(&s),
        "stopWordsRemoverLoadDefaultStopWords",
        vec![FetchArg::Value(MlValue::String("english".into()))],
    )
    .unwrap()
    .into_value()
    .unwrap();
    assert_eq!(words.as_array().unwrap().len(), 181);
    assert_eq!(words.as_array().unwrap()[0], MlValue::String("i".into()));
    s.stop().unwrap();
}

#[test]
fn ml_objects_of_a_fetch_are_each_their_own_object_as_pyspark_splits_the_trees() {
    let Some(s) = live_session() else {
        return;
    };
    let model = RandomForestClassifier::new()
        .set_num_trees(3)
        .fit(&training(&s))
        .unwrap();
    let target = MlTarget::object(model.ml_ref().unwrap());
    let trees = fetch(&target, "trees", vec![])
        .unwrap()
        .into_objects()
        .unwrap();
    assert_eq!(trees.len(), 3);
    for tree in &trees {
        let nodes = fetch(&MlTarget::object(tree), "numNodes", vec![])
            .unwrap()
            .into_value()
            .unwrap();
        assert!(nodes.as_i64().unwrap() >= 1);
    }
    s.stop().unwrap();
}

#[test]
fn ml_models_and_estimators_save_and_load_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let train = training(&s);
    let estimator = LinearSVC::new().set_max_iter(5);
    let model = estimator.clone().fit(&train).unwrap();
    let object = model.ml_ref().unwrap();
    let expected = fetch(&MlTarget::object(object), "coefficients", vec![])
        .unwrap()
        .into_value()
        .unwrap();

    let model_path = scratch_path("model");
    let options = Default::default();
    write_object(object, model.params(), &model_path, true, &options).unwrap();
    let loaded = read(
        &s,
        "org.apache.spark.ml.classification.LinearSVCModel",
        OperatorType::Model,
        &model_path,
    )
    .unwrap();
    let loaded_object = loaded
        .object
        .expect("a model loads as a server-side object");
    assert_eq!(loaded.uid, model.operator().uid);
    let coefficients = fetch(&MlTarget::object(&loaded_object), "coefficients", vec![])
        .unwrap()
        .into_value()
        .unwrap();
    assert_eq!(coefficients, expected);
    // A second save to the same path without overwrite fails on the server, as with PySpark.
    assert!(write_object(object, model.params(), &model_path, false, &options).is_err());
    // The writer of the model, of its own session, as `model.write().overwrite().save(path)`.
    model
        .write()
        .unwrap()
        .overwrite()
        .save(&model_path)
        .unwrap();
    assert!(model.write().unwrap().save(&model_path).is_err());
    // The typed model of the fit, and the server's description of it (PySpark's `__repr__`).
    assert!(model.downcast_ref::<LinearSVCModel>().is_some());
    assert!(model.describe().unwrap().starts_with("LinearSVCModel"));

    let (reloaded, operator, params) = read_model(
        &s,
        "org.apache.spark.ml.classification.LinearSVCModel",
        &model_path,
    )
    .unwrap();
    assert_eq!(operator.uid, model.operator().uid);
    assert_eq!(operator.op_type, OperatorType::Model);
    assert!(params.get_param("maxIter").is_some());
    assert_eq!(
        fetch(&MlTarget::object(&reloaded), "coefficients", vec![])
            .unwrap()
            .into_value()
            .unwrap(),
        expected
    );

    let estimator_path = scratch_path("estimator");
    // An estimator saves of the session that its writer sets, as `MLWriter.session`.
    let unset = Estimator::write(&estimator)
        .save(&estimator_path)
        .unwrap_err();
    assert!(unset.to_string().contains("needs a session"), "{unset}");
    Estimator::write(&estimator)
        .session(&s)
        .overwrite()
        .save(&estimator_path)
        .unwrap();
    write_operator(
        &s,
        Estimator::operator(&estimator),
        Estimator::params(&estimator),
        &estimator_path,
        true,
        &options,
    )
    .unwrap();
    let loaded = read(
        &s,
        "org.apache.spark.ml.classification.LinearSVC",
        OperatorType::Estimator,
        &estimator_path,
    )
    .unwrap();
    assert!(loaded.object.is_none());
    assert_eq!(loaded.uid, Estimator::operator(&estimator).uid);
    let (operator, _) = read_operator(
        &s,
        "org.apache.spark.ml.classification.LinearSVC",
        OperatorType::Estimator,
        &estimator_path,
    )
    .unwrap();
    assert_eq!(operator.uid, Estimator::operator(&estimator).uid);
    assert_eq!(
        loaded.params.get_param("maxIter").map(|e| format!("{e:?}")),
        Estimator::params(&estimator)
            .get_param("maxIter")
            .map(|e| format!("{e:?}"))
    );
    s.stop().unwrap();
}

#[test]
fn ml_summaries_of_a_transform_and_of_an_evaluate_and_models_of_loaded_objects_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let train = training(&s);
    // The training summary of a clustering model is of the transform of the training dataset by
    // the model, of no params, as PySpark's default `_summary_dataset`.
    let model = KMeans::new().set_k(2).fit(&train).unwrap();
    let object = model
        .ml_ref()
        .expect("a KMeans model is a server-side object");
    let dataset = summary_transform(object, &train).unwrap();
    assert!(dataset
        .columns()
        .unwrap()
        .contains(&"prediction".to_string()));
    let summary = training_summary(object, &dataset);
    let k = fetch(&summary, "k", vec![]).unwrap().into_value().unwrap();
    assert_eq!(k, MlValue::Int(2));
    let predictions = attribute_relation(&summary, "predictions", vec![]).unwrap();
    assert_eq!(predictions.count().unwrap(), 8);

    // The summary of an evaluate, of the id that the server returns.
    let svc = LinearSVC::new().set_max_iter(5).fit(&train).unwrap();
    let target = MlTarget::object(svc.ml_ref().unwrap());
    let output = fetch(&target, "evaluate", vec![FetchArg::Input(train.clone())]).unwrap();
    let evaluation = evaluated_summary(&s, output).unwrap();
    let accuracy = fetch(&evaluation, "accuracy", vec![]).unwrap().into_value();
    assert_eq!(accuracy.unwrap(), MlValue::Double(1.0));

    // A model of a loaded object transforms as the model that was saved.
    let path = scratch_path("object-model");
    svc.write().unwrap().overwrite().save(&path).unwrap();
    let (object, operator, params) = read_model(
        &s,
        "org.apache.spark.ml.classification.LinearSVCModel",
        &path,
    )
    .unwrap();
    let mut loaded = ObjectModel::new(object, operator, params);
    let transformed = loaded.transform(&train).unwrap();
    assert!(transformed
        .columns()
        .unwrap()
        .contains(&"prediction".to_string()));
    assert_eq!(transformed.count().unwrap(), 8);
    s.stop().unwrap();
}
