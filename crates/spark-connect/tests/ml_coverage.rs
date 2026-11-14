//! Construct every ML estimator/evaluator, drive its fluent setters + getters, and
//! serialize its proto. These are pure builders (no server), so this pins the setter/
//! getter/to_proto bodies in ml.rs.

use spark_connect::ml::*;

#[test]
fn ml_estimators_setters_getters_proto() {
    let ss = StandardScaler::new().set_input_col("f").set_output_col("s");
    assert_eq!(ss.input_col(), Some("f"));
    assert_eq!(ss.output_col(), "s");

    let va = VectorAssembler::new()
        .set_input_cols(vec!["a", "b"])
        .set_output_col("v");
    let _ = va.input_cols();
    assert_eq!(va.output_col(), "v");

    let si = StringIndexer::new().set_input_col("c").set_output_col("i");
    assert_eq!(si.input_col(), Some("c"));
    assert_eq!(si.output_col(), "i");

    let ma = MaxAbsScaler::new().set_input_col("f").set_output_col("s");
    assert_eq!(ma.input_col(), Some("f"));
    assert_eq!(ma.output_col(), "s");

    let lr = LogisticRegression::new()
        .set_features_col("f")
        .set_label_col("l")
        .set_prediction_col("p")
        .set_max_iter(10);
    assert_eq!(lr.features_col(), "f");
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
        .set_raw_prediction_col("s")
        .set_metric_name("areaUnderROC");
    assert_eq!(be.label_col(), "l");
    assert_eq!(be.raw_prediction_col(), "s");
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

#[test]
fn params_builders_send_the_literals_of_pyspark_from_outside_the_crate() {
    // The public `Params` builder sends what PySpark sends of a Python value: an int that fits
    // 32 bits as an `Integer`, a larger one as a `Long`, and a list of strings or floats as an
    // array literal of that element type; `clear_param` leaves a param unset.
    use spark_connect_proto::expression::literal::LiteralType as L;
    let params = Params::new()
        .set_param_int32("maxIter", 5)
        .set_param_int_narrow("k", 3)
        .set_param_int_narrow("seed", 1 << 40)
        .set_param_string_array("inputCols", ["a", "b"])
        .set_param_double_array("thresholds", [0.25, 0.75])
        .set_param_string("weightCol", "w")
        .clear_param("weightCol")
        .to_proto()
        .params;
    assert_eq!(params["maxIter"].literal_type, Some(L::Integer(5)));
    assert_eq!(params["k"].literal_type, Some(L::Integer(3)));
    assert_eq!(params["seed"].literal_type, Some(L::Long(1 << 40)));
    assert!(!params.contains_key("weightCol"));
    let elements = |name: &str| match &params[name].literal_type {
        Some(L::Array(array)) => array
            .elements
            .iter()
            .map(|element| element.literal_type.clone())
            .collect::<Vec<_>>(),
        other => panic!("{name}: {other:?}"),
    };
    assert_eq!(
        elements("inputCols"),
        [Some(L::String("a".into())), Some(L::String("b".into()))]
    );
    assert_eq!(
        elements("thresholds"),
        [Some(L::Double(0.25)), Some(L::Double(0.75))]
    );
}
