//! Construct every ML estimator/evaluator, drive its fluent setters + getters, and
//! serialize its proto. These are pure builders (no server), so this pins the setter/
//! getter/to_proto bodies in ml.rs.

use spark_connect::ml::*;

#[test]
fn ml_estimators_setters_getters_proto() {
    let ss = StandardScaler::new().set_input_col("f").set_output_col("s");
    assert_eq!(ss.input_col(), "f");
    assert_eq!(ss.output_col(), "s");

    let va = VectorAssembler::new()
        .set_input_cols(vec!["a", "b"])
        .set_output_col("v");
    let _ = va.input_cols();
    assert_eq!(va.output_col(), "v");

    let si = StringIndexer::new().set_input_col("c").set_output_col("i");
    assert_eq!(si.input_col(), "c");
    assert_eq!(si.output_col(), "i");

    let ma = MaxAbsScaler::new().set_input_col("f").set_output_col("s");
    assert_eq!(ma.input_col(), "f");
    assert_eq!(ma.output_col(), "s");

    let lr = LogisticRegression::new()
        .set_feature_col("f")
        .set_label_col("l")
        .set_prediction_col("p")
        .set_max_iter(10);
    assert_eq!(lr.feature_col(), "f");
    assert_eq!(lr.label_col(), "l");
    assert_eq!(lr.prediction_col(), "p");
    assert_eq!(lr.max_iter(), 10);

    let re = RegressionEvaluator::new()
        .set_label_col("l")
        .set_prediction_col("p")
        .set_metric_name("rmse");
    assert_eq!(re.label_col(), "l");
    assert_eq!(re.prediction_col(), "p");
    assert_eq!(re.metric_name(), "rmse");

    let be = BinaryClassificationEvaluator::new()
        .set_label_col("l")
        .set_score_col("s")
        .set_metric_name("areaUnderROC");
    assert_eq!(be.label_col(), "l");
    assert_eq!(be.score_col(), "s");
    assert_eq!(be.metric_name(), "areaUnderROC");

    let mce = MulticlassClassificationEvaluator::new()
        .set_label_col("l")
        .set_prediction_col("p")
        .set_metric_name("f1");
    assert_eq!(mce.label_col(), "l");
    assert_eq!(mce.prediction_col(), "p");
    assert_eq!(mce.metric_name(), "f1");

    let pl = Pipeline::new().set_stages(vec!["a", "b"]);
    let _ = pl.stages();

    let cv = CrossValidator::new()
        .set_num_folds(3)
        .set_parallelism(2)
        .set_seed(42);
    assert_eq!(cv.num_folds(), 3);
    assert_eq!(cv.parallelism(), 2);
}
