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
    attribute_relation, evaluated_summary, fetch, handle_overwrite, json_object, k_fold,
    load_metadata, load_stage, read, read_model, read_operator, save_metadata, stage_path,
    summary_transform, train_validation_split, training_summary, write_object, write_operator,
    BinaryClassificationEvaluator, CrossValidator, CrossValidatorModel, Estimator, FetchArg,
    KMeans, LinearSVC, LinearSVCModel, Metadata, MlTarget, MlValue, MlVector, Model, ObjectModel,
    OperatorType, ParamGridBuilder, Pipeline, PipelineModel, PipelineStage, RandomForestClassifier,
    TrainValidationSplit, TrainValidationSplitModel, Transformer,
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
    let _: &LinearSVCModel = &model;
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
    let evaluation = evaluated_summary(&target, output).unwrap();
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

#[test]
fn ml_metadata_of_the_python_format_saves_loads_and_overwrites_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    // The metadata of a Pipeline, as `DefaultParamsWriter.saveMetadata` writes it: a text file of
    // one line at `<path>/metadata`, which `loadMetadata` reads.
    let path = scratch_path("metadata");
    handle_overwrite(&s, &path).unwrap();
    let param_map = json_object(&[
        ("stageUids", "[\"KMeans_0123456789ab\"]".to_string()),
        ("language", "\"Python\"".to_string()),
    ]);
    let version = s.version().unwrap();
    let metadata = Metadata::new(
        "pyspark.ml.pipeline.Pipeline",
        "Pipeline_0123456789ab",
        &version,
        param_map,
        "{}".to_string(),
    );
    save_metadata(&s, &path, &metadata.to_json()).unwrap();
    let loaded = load_metadata(&s, &path, Some("pyspark.ml.pipeline.Pipeline")).unwrap();
    assert_eq!(loaded["uid"], "Pipeline_0123456789ab");
    assert_eq!(loaded["sparkVersion"], version.as_str());
    assert_eq!(loaded["paramMap"]["stageUids"][0], "KMeans_0123456789ab");
    // Another class is an error, as `_parseMetaData` raises it.
    let error = load_metadata(&s, &path, Some("pyspark.ml.tuning.CrossValidator")).unwrap_err();
    assert!(error.to_string().contains("Expected class name"), "{error}");
    // A second save to the path fails, as the metadata's default save mode does, until the ML
    // helper deletes the path of an overwrite.
    assert!(save_metadata(&s, &path, &metadata.to_json()).is_err());
    handle_overwrite(&s, &path).unwrap();
    save_metadata(&s, &path, &metadata.to_json()).unwrap();
    assert_eq!(
        stage_path("KMeans_0123456789ab", 3, 12, &format!("{path}/stages")),
        format!("{path}/stages/03_KMeans_0123456789ab")
    );
    s.stop().unwrap();
}

#[test]
fn ml_validators_split_their_datasets_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let df = s
        .sql("SELECT id, CAST(id % 2 AS INT) AS fold FROM range(20)")
        .unwrap()
        .coalesce(1);
    let count = |df: &DataFrame| df.count().unwrap();
    // A random split: each row is in the validation set of one fold, of `rand(seed)`.
    let folds = k_fold(&df, 3, "", Some(7), "CrossValidator_0123456789ab").unwrap();
    assert_eq!(folds.len(), 3);
    let validated: i64 = folds.iter().map(|(_, validation)| count(validation)).sum();
    assert_eq!(validated, 20);
    for (train, validation) in &folds {
        assert_eq!(count(train) + count(validation), 20);
        assert!(train
            .columns()
            .unwrap()
            .contains(&"CrossValidator_0123456789ab_rand".to_string()));
    }
    // The same seed splits the same rows.
    let again = k_fold(&df, 3, "", Some(7), "CrossValidator_0123456789ab").unwrap();
    assert_eq!(count(&folds[1].1), count(&again[1].1));
    // A fold column: the rows of each fold number.
    let folds = k_fold(&df, 2, "fold", None, "CrossValidator_0123456789ab").unwrap();
    assert_eq!(count(&folds[0].1), 10);
    assert_eq!(count(&folds[1].0), 10);
    // A fold number out of the range fails the query of its fold, and an empty fold is an error.
    let Err(error) = k_fold(&df, 1, "fold", None, "CrossValidator_0123456789ab") else {
        panic!("a fold of a number out of the range is an error");
    };
    assert!(
        error
            .to_string()
            .contains("Fold number must be in range [0, 1), but got 1"),
        "{error}"
    );
    let Err(error) = k_fold(&df, 3, "fold", None, "CrossValidator_0123456789ab") else {
        panic!("a fold of no rows is an error");
    };
    assert_eq!(error.message, "The validation data at fold 2 is empty.");
    // A train validation split: the rows of `rand(seed)` of at least the ratio validate.
    let (train, validation) = train_validation_split(&df, 0.75, Some(7), "TVS_0123456789ab");
    assert_eq!(count(&train) + count(&validation), 20);
    let (_, again) = train_validation_split(&df, 0.75, Some(7), "TVS_0123456789ab");
    assert_eq!(count(&validation), count(&again));
    s.stop().unwrap();
}

/// 40 rows of two features and a binary label, of one partition.
fn tuning_frame(s: &SparkSession) -> DataFrame {
    s.sql(
        "SELECT array(CAST(id % 7 AS DOUBLE), CAST(id % 3 AS DOUBLE)) AS xs, \
         CAST(id % 2 AS DOUBLE) AS label FROM range(40)",
    )
    .unwrap()
    .with_column(
        "features",
        Column::new(Expression::function(
            "array_to_vector",
            vec![col("xs").expression().clone()],
        )),
    )
    .coalesce(1)
}

#[test]
fn ml_meta_algorithms_fit_transform_save_and_load_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let df = tuning_frame(&s);
    let rows = df.count().unwrap();
    let svc = LinearSVC::new().set_max_iter(3);
    let uid = svc.operator().uid.clone();
    let grid = ParamGridBuilder::new()
        .add_grid(
            &uid,
            "regParam",
            [MlValue::Double(0.1), MlValue::Double(1.0)],
        )
        .build();

    // A pipeline fits its estimator, into a model of a new uid that transforms.
    let mut pipeline = Pipeline::new().set_stages(vec![PipelineStage::estimator(svc.clone())]);
    let mut model = pipeline.fit_pipeline(&df).unwrap();
    assert_ne!(model.uid(), pipeline.uid());
    assert_eq!(model.stages()[0].uid(), uid);
    let predicted = model.transform(&df).unwrap();
    assert!(predicted
        .columns()
        .unwrap()
        .contains(&"prediction".to_string()));
    assert_eq!(predicted.count().unwrap(), rows);
    // Of PySpark's Python format: its metadata, its stages at stages/<index>_<uid>.
    let path = scratch_path("pipeline");
    handle_overwrite(&s, &path).unwrap();
    pipeline.write().session(&s).save(&path).unwrap();
    let metadata = load_metadata(&s, &path, Some("pyspark.ml.pipeline.Pipeline")).unwrap();
    assert_eq!(metadata["paramMap"]["stageUids"][0], uid.as_str());
    assert_eq!(metadata["paramMap"]["language"], "Python");
    assert!(std::path::Path::new(&format!("{path}/stages/0_{uid}/metadata")).exists());
    let loaded = Pipeline::load(&s, &path).unwrap();
    assert_eq!(loaded.uid(), pipeline.uid());
    assert_eq!(loaded.stages()[0].uid(), uid);
    assert!(loaded.stages()[0].is_estimator());
    let model_path = scratch_path("pipeline-model");
    model
        .write()
        .unwrap()
        .session(&s)
        .overwrite()
        .save(&model_path)
        .unwrap();
    let mut loaded_model = PipelineModel::load(&s, &model_path).unwrap();
    assert_eq!(loaded_model.uid(), model.uid());
    assert_eq!(loaded_model.transform(&df).unwrap().count().unwrap(), rows);
    assert!(matches!(
        load_stage(&s, &model_path).unwrap(),
        PipelineStage::Model(_)
    ));

    // A cross validation of two folds, of the sub-models of each fold and param map.
    let mut cv = CrossValidator::new()
        .set_estimator(svc.clone())
        .set_estimator_param_maps(grid.clone())
        .set_evaluator(BinaryClassificationEvaluator::new())
        .set_num_folds(2)
        .set_seed(7)
        .set_parallelism(2)
        .set_collect_sub_models(true);
    let cv_model = cv.fit_validator(&df).unwrap();
    assert_eq!(cv_model.avg_metrics().len(), 2);
    assert_eq!(cv_model.std_metrics().len(), 2);
    let folds = cv_model.sub_models().unwrap();
    assert_eq!((folds.len(), folds[0].len()), (2, 2));
    assert!(cv_model.uid().starts_with("CrossValidatorModel_"));
    // The same seed splits and fits the same.
    let again = cv.fit_validator(&df).unwrap();
    assert_eq!(again.avg_metrics(), cv_model.avg_metrics());
    // Of the trait: a model that downcasts to the typed model.
    let boxed = cv.fit(&df).unwrap();
    assert!(boxed.downcast_ref::<CrossValidatorModel>().is_some());
    let cv_path = scratch_path("cv");
    cv.write().session(&s).overwrite().save(&cv_path).unwrap();
    let metadata = load_metadata(&s, &cv_path, Some("pyspark.ml.tuning.CrossValidator")).unwrap();
    let keys: Vec<&String> = metadata["paramMap"].as_object().unwrap().keys().collect();
    assert_eq!(
        keys,
        [
            "collectSubModels",
            "estimatorParamMaps",
            "foldCol",
            "numFolds",
            "parallelism",
            "seed"
        ]
    );
    assert_eq!(metadata["paramMap"]["numFolds"], 2);
    assert_eq!(
        metadata["paramMap"]["estimatorParamMaps"][1][0]["value"],
        "1.0"
    );
    let loaded_cv = CrossValidator::load(&s, &cv_path).unwrap();
    assert_eq!(loaded_cv.uid(), cv.uid());
    assert_eq!(
        (
            loaded_cv.num_folds(),
            loaded_cv.seed(),
            loaded_cv.parallelism()
        ),
        (2, 7, 2)
    );
    assert_eq!(loaded_cv.estimator_param_maps(), grid.as_slice());
    let cv_model_path = scratch_path("cv-model");
    cv_model
        .write()
        .unwrap()
        .session(&s)
        .overwrite()
        .save(&cv_model_path)
        .unwrap();
    assert!(std::path::Path::new(&format!("{cv_model_path}/subModels/fold1/1/metadata")).exists());
    let mut loaded_cv_model = CrossValidatorModel::load(&s, &cv_model_path).unwrap();
    assert_eq!(loaded_cv_model.avg_metrics(), cv_model.avg_metrics());
    assert_eq!(loaded_cv_model.sub_models().unwrap().len(), 2);
    assert_eq!(
        loaded_cv_model.transform(&df).unwrap().count().unwrap(),
        rows
    );
    // persistSubModels=false saves none.
    cv_model
        .write()
        .unwrap()
        .session(&s)
        .overwrite()
        .option("persistSubModels", "false")
        .save(&cv_model_path)
        .unwrap();
    assert!(CrossValidatorModel::load(&s, &cv_model_path)
        .unwrap()
        .sub_models()
        .is_none());

    // A train validation split, of a pipeline estimator.
    let mut tvs = TrainValidationSplit::new()
        .set_estimator(pipeline.clone())
        .set_estimator_param_maps(grid.clone())
        .set_evaluator(BinaryClassificationEvaluator::new())
        .set_seed(7);
    let tvs_model = tvs.fit_validator(&df).unwrap();
    assert_eq!(tvs_model.validation_metrics().len(), 2);
    assert!(tvs_model.sub_models().is_none());
    let tvs_path = scratch_path("tvs-model");
    tvs_model
        .write()
        .unwrap()
        .session(&s)
        .overwrite()
        .save(&tvs_path)
        .unwrap();
    let loaded_tvs = TrainValidationSplitModel::load(&s, &tvs_path).unwrap();
    assert_eq!(
        loaded_tvs.validation_metrics(),
        tvs_model.validation_metrics()
    );
    assert_eq!(loaded_tvs.train_ratio(), 0.75);
    assert!(matches!(
        load_stage(&s, &tvs_path).unwrap(),
        PipelineStage::Model(_)
    ));
    s.stop().unwrap();
}
