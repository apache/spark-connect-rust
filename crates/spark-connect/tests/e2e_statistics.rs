// SPDX-License-Identifier: Apache-2.0
//! Live statistics comparisons to official PySpark, including input model ownership.
mod common;
#[path = "common/statistics.rs"]
mod statistics;
use spark_connect::column::col;
use spark_connect::ml::{
    ChiSquareTest, Correlation, Estimator, KolmogorovSmirnovTest, StandardScaler, Summarizer,
};

#[test]
fn statistical_tests_and_all_summaries_match_official_pyspark() {
    let Some(spark) = common::live_session() else {
        return;
    };
    let fixture = statistics::fixture();
    for case in fixture["helpers"].as_array().unwrap() {
        let df = statistics::dataset(&spark, case["dataset"].as_str().unwrap());
        statistics::assert_result(&statistics::helper(case, &df), &case["result"]);
    }
    let numeric = statistics::dataset(&spark, "numeric");
    for case in fixture["summaries"].as_array().unwrap() {
        statistics::assert_result(
            &numeric.select([statistics::summary(case)]),
            &case["result"],
        );
    }
    let grouped = numeric
        .group_by([col("group_id")])
        .agg([Summarizer::metrics(["mean", "count"])
            .summary(col("features"), Some(col("weight")))
            .alias("summary")]);
    statistics::assert_result(&grouped, &fixture["grouped"]);
    assert!(Correlation::corr(&numeric, "features", "invalid")
        .unwrap()
        .collect()
        .is_err());
    assert!(
        KolmogorovSmirnovTest::test(&numeric, "sample", "invalid", [])
            .unwrap()
            .collect()
            .is_err()
    );
    assert!(numeric
        .select([Summarizer::metrics(["invalid"]).summary(col("features"), None)])
        .collect()
        .is_err());
    spark.stop().unwrap();
}
#[test]
fn statistics_keep_upstream_models_alive() {
    let Some(spark) = common::live_session() else {
        return;
    };
    let df = statistics::dataset(&spark, "categorical");
    let mut model = StandardScaler::new()
        .set_input_col("features")
        .set_output_col("scaled")
        .set_with_std(false)
        .set_with_mean(false)
        .fit(&df)
        .unwrap();
    let transformed = model.transform(&df).unwrap();
    let results = [
        ChiSquareTest::test(&transformed, "scaled", "label", false).unwrap(),
        Correlation::corr(&transformed, "scaled", "pearson").unwrap(),
        KolmogorovSmirnovTest::test(&transformed, "label", "norm", []).unwrap(),
        transformed.select([Summarizer::mean(col("scaled"), None)]),
    ];
    drop(transformed);
    drop(model);
    for result in results {
        assert_eq!(result.count().unwrap(), 1);
    }
    spark.stop().unwrap();
}
