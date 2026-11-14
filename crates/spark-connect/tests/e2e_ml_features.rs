// SPDX-License-Identifier: Apache-2.0
//! All standard feature stages against Spark. Set SPARK_REMOTE to run.
mod common;
use common::live_session;
use serde_json::{json, Value as Json};
use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::ml::*;
use spark_connect::row::Value;
use spark_connect::{Column, DataFrame};

fn output(id: &str, input: &DataFrame) -> DataFrame {
    match id {
        "Binarizer" => Binarizer::new()
            .set_input_col("x")
            .set_output_col("out")
            .set_threshold(2.5)
            .transform(input)
            .unwrap(),
        "BucketedRandomProjectionLSH" => BucketedRandomProjectionLSH::new()
            .set_input_col("features")
            .set_output_col("out")
            .set_num_hash_tables(2)
            .set_bucket_length(1.5)
            .set_seed(42)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "Bucketizer" => Bucketizer::new()
            .set_input_col("x")
            .set_output_col("out")
            .set_splits([f64::NEG_INFINITY, 2.0, 5.0, f64::INFINITY])
            .transform(input)
            .unwrap(),
        "ChiSqSelector" => ChiSqSelector::new()
            .set_features_col("features")
            .set_output_col("out")
            .set_label_col("label")
            .set_num_top_features(2)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "CountVectorizer" => CountVectorizer::new()
            .set_input_col("words")
            .set_output_col("out")
            .set_vocab_size(4)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "DCT" => DCT::new()
            .set_input_col("features")
            .set_output_col("out")
            .transform(input)
            .unwrap(),
        "ElementwiseProduct" => ElementwiseProduct::new()
            .set_input_col("features")
            .set_output_col("out")
            .set_scaling_vec([2.0, 1.0, 0.0, -1.0])
            .transform(input)
            .unwrap(),
        "FeatureHasher" => FeatureHasher::new()
            .set_input_cols(["x", "y", "category"])
            .set_output_col("out")
            .set_num_features(8)
            .transform(input)
            .unwrap(),
        "HashingTF" => HashingTF::new()
            .set_input_col("words")
            .set_output_col("out")
            .set_num_features(8)
            .transform(input)
            .unwrap(),
        "IDF" => IDF::new()
            .set_input_col("features")
            .set_output_col("out")
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "Imputer" => Imputer::new()
            .set_input_col("missing")
            .set_output_col("out")
            .set_missing_value(f64::NAN)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "IndexToString" => IndexToString::new()
            .set_input_col("index")
            .set_output_col("out")
            .set_labels(["a", "b", "c"])
            .transform(input)
            .unwrap(),
        "Interaction" => Interaction::new()
            .set_input_cols(["x", "y"])
            .set_output_col("out")
            .transform(input)
            .unwrap(),
        "MaxAbsScaler" => MaxAbsScaler::new()
            .set_input_col("features")
            .set_output_col("out")
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "MinHashLSH" => MinHashLSH::new()
            .set_input_col("features")
            .set_output_col("out")
            .set_num_hash_tables(2)
            .set_seed(42)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "MinMaxScaler" => MinMaxScaler::new()
            .set_input_col("features")
            .set_output_col("out")
            .set_min(-1.0)
            .set_max(1.0)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "NGram" => NGram::new()
            .set_input_col("words")
            .set_output_col("out")
            .set_n(2)
            .transform(input)
            .unwrap(),
        "Normalizer" => Normalizer::new()
            .set_input_col("features")
            .set_output_col("out")
            .set_p(1.0)
            .transform(input)
            .unwrap(),
        "OneHotEncoder" => OneHotEncoder::new()
            .set_input_col("index")
            .set_output_col("out")
            .set_drop_last(false)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "PCA" => PCA::new()
            .set_input_col("features")
            .set_output_col("out")
            .set_k(2)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "PolynomialExpansion" => PolynomialExpansion::new()
            .set_input_col("features")
            .set_output_col("out")
            .set_degree(2)
            .transform(input)
            .unwrap(),
        "QuantileDiscretizer" => QuantileDiscretizer::new()
            .set_input_col("x")
            .set_output_col("out")
            .set_num_buckets(3)
            .set_relative_error(0.0)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "RobustScaler" => RobustScaler::new()
            .set_input_col("features")
            .set_output_col("out")
            .set_with_centering(true)
            .set_relative_error(0.0)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "RegexTokenizer" => RegexTokenizer::new()
            .set_input_col("text")
            .set_output_col("out")
            .set_pattern("\\s+")
            .set_gaps(true)
            .transform(input)
            .unwrap(),
        "RFormula" => RFormula::new()
            .set_formula("label ~ x + category")
            .set_features_col("out")
            .set_label_col("target")
            .set_string_indexer_order_type("alphabetAsc")
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "SQLTransformer" => SQLTransformer::new()
            .set_statement("SELECT *, x + y AS out FROM __THIS__")
            .transform(input)
            .unwrap(),
        "StandardScaler" => StandardScaler::new()
            .set_input_col("features")
            .set_output_col("out")
            .set_with_mean(true)
            .set_with_std(true)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "StopWordsRemover" => StopWordsRemover::new()
            .set_input_col("words")
            .set_output_col("out")
            .set_stop_words(["the", "gamma"])
            .transform(input)
            .unwrap(),
        "StringIndexer" => StringIndexer::new()
            .set_input_col("category")
            .set_output_col("out")
            .set_string_order_type("alphabetAsc")
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "TargetEncoder" => TargetEncoder::new()
            .set_input_col("index")
            .set_output_col("out")
            .set_label_col("label")
            .set_smoothing(1.0)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "Tokenizer" => Tokenizer::new()
            .set_input_col("text")
            .set_output_col("out")
            .transform(input)
            .unwrap(),
        "UnivariateFeatureSelector" => UnivariateFeatureSelector::new()
            .set_features_col("features")
            .set_output_col("out")
            .set_label_col("label")
            .set_feature_type("categorical")
            .set_label_type("categorical")
            .set_selection_threshold(2.0)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "VarianceThresholdSelector" => VarianceThresholdSelector::new()
            .set_features_col("features")
            .set_output_col("out")
            .set_variance_threshold(1.0)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "VectorAssembler" => VectorAssembler::new()
            .set_input_cols(["x", "y"])
            .set_output_col("out")
            .set_handle_invalid("error")
            .transform(input)
            .unwrap(),
        "VectorIndexer" => VectorIndexer::new()
            .set_input_col("features")
            .set_output_col("out")
            .set_max_categories(2)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "VectorSizeHint" => VectorSizeHint::new()
            .set_input_col("rawFeatures")
            .set_size(4)
            .transform(input)
            .unwrap(),
        "VectorSlicer" => VectorSlicer::new()
            .set_input_col("features")
            .set_output_col("out")
            .set_indices([0, 2])
            .transform(input)
            .unwrap(),
        "Word2Vec" => Word2Vec::new()
            .set_input_col("words")
            .set_output_col("out")
            .set_vector_size(3)
            .set_min_count(1)
            .set_seed(42)
            .set_num_partitions(1)
            .set_max_iter(2)
            .set_window_size(2)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "Binarizer-multi" => Binarizer::new()
            .set_input_cols(["x", "y"])
            .set_output_cols(["out", "other"])
            .set_thresholds([2.0, 4.0])
            .transform(input)
            .unwrap(),
        "Bucketizer-multi" => Bucketizer::new()
            .set_input_cols(["x", "y"])
            .set_output_cols(["out", "other"])
            .set_splits_array([
                [f64::NEG_INFINITY, 3.0, f64::INFINITY],
                [f64::NEG_INFINITY, 5.0, f64::INFINITY],
            ])
            .transform(input)
            .unwrap(),
        "Imputer-multi" => Imputer::new()
            .set_input_cols(["missing", "x"])
            .set_output_cols(["out", "other"])
            .set_strategy("median")
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "OneHotEncoder-multi" => OneHotEncoder::new()
            .set_input_cols(["index", "label"])
            .set_output_cols(["out", "other"])
            .set_drop_last(false)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "StopWordsRemover-multi" => StopWordsRemover::new()
            .set_input_cols(["words", "words2"])
            .set_output_cols(["out", "other"])
            .set_stop_words(["the", "gamma"])
            .transform(input)
            .unwrap(),
        "StringIndexer-multi" => StringIndexer::new()
            .set_input_cols(["category", "text"])
            .set_output_cols(["out", "other"])
            .set_string_order_type("alphabetAsc")
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "TargetEncoder-multi" => TargetEncoder::new()
            .set_input_cols(["index", "label"])
            .set_output_cols(["out", "other"])
            .set_label_col("x")
            .set_target_type("continuous")
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "QuantileDiscretizer-multi" => QuantileDiscretizer::new()
            .set_input_cols(["x", "y"])
            .set_output_cols(["out", "other"])
            .set_num_buckets_array([2, 3])
            .set_relative_error(0.0)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "StandardScaler-variant" => StandardScaler::new()
            .set_input_col("features")
            .set_output_col("out")
            .set_with_mean(false)
            .set_with_std(false)
            .fit(input)
            .unwrap()
            .transform(input)
            .unwrap(),
        "RegexTokenizer-variant" => RegexTokenizer::new()
            .set_input_col("text")
            .set_output_col("out")
            .set_pattern("[a-z]+")
            .set_gaps(false)
            .set_to_lowercase(false)
            .transform(input)
            .unwrap(),
        _ => panic!("uncovered feature case {id}"),
    }
}

fn normalized(v: &Value) -> Json {
    match v {
        Value::Null => Json::Null,
        Value::String(v) => json!(v),
        Value::Bool(v) => json!(v),
        Value::List(v) => v.iter().map(normalized).collect(),
        Value::Struct(row) if row.get_by_name("type").is_some() => {
            // Spark VectorUDT: dense values, or sparse indices and values.
            let values = normalized(row.get_by_name("values").unwrap());
            let kind = row.get_by_name("type").unwrap();
            if matches!(kind, Value::Byte(1)) {
                return values;
            }
            let size = row.get_by_name("size").unwrap().as_i64().unwrap() as usize;
            let mut dense = vec![json!(0.0); size];
            let Value::List(indices) = row.get_by_name("indices").unwrap() else {
                panic!("indices")
            };
            for (index, value) in indices.iter().zip(values.as_array().unwrap()) {
                dense[index.as_i64().unwrap() as usize] = value.clone();
            }
            json!(dense)
        }
        _ => json!(v.as_f64().unwrap()),
    }
}

fn assert_values(actual: &Json, expected: &Json, pca: bool) {
    match expected {
        Json::Array(values) => {
            let actual = actual.as_array().unwrap();
            assert_eq!(actual.len(), values.len());
            for (a, e) in actual.iter().zip(values) {
                assert_values(a, e, pca);
            }
        }
        Json::Number(n) => {
            let (mut a, mut e) = (actual.as_f64().unwrap(), n.as_f64().unwrap());
            if pca {
                a = a.abs();
                e = e.abs();
            }
            assert!((a - e).abs() <= 1e-7 + 1e-5 * e.abs(), "{a} != {e}");
        }
        _ => assert_eq!(actual, expected),
    }
}

fn wire_schema(mut value: Json) -> Json {
    // JVM UDTs arrive by class name; Connect omits their backing SQL schema.
    match &mut value {
        Json::Object(map) => {
            if map.get("type").is_some_and(|v| v == "udt") {
                map.remove("sqlType");
            }
            for v in map.values_mut() {
                *v = wire_schema(v.take());
            }
        }
        Json::Array(values) => {
            for v in values {
                *v = wire_schema(v.take());
            }
        }
        _ => {}
    }
    value
}

#[test]
fn all_feature_stages_match_official_results_and_retain_models() {
    let Some(session) = live_session() else {
        return;
    };
    let golden: Json =
        serde_json::from_str(include_str!("../../../tests/golden/ml_features.json")).unwrap();
    let vector = Column::new(Expression::function(
        "array_to_vector",
        vec![col("xs").expression().clone()],
    ));
    let input = session
        .sql(golden["sql"].as_str().unwrap())
        .unwrap()
        .with_column("features", vector.clone())
        .with_column("rawFeatures", vector)
        .coalesce(1);
    for case in golden["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        eprintln!("feature: {id}");
        // Each builder/model drops inside output(); the lazy plan keeps its owners alive.
        let result = output(id, &input)
            .order_by(vec![col("id").asc().expression().clone()])
            .select(
                case["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| col(v.as_str().unwrap())),
            );
        let actual: Json = result
            .collect()
            .unwrap()
            .iter()
            .map(|row| row.values().iter().map(normalized).collect::<Json>())
            .collect();
        assert_values(&actual, &case["rows"], case["name"] == "PCA");
        assert_eq!(
            wire_schema(result.schema().unwrap().json_value()),
            wire_schema(case["schema"].clone()),
            "{id} schema"
        );
    }
}

#[test]
fn stop_words_remover_loads_the_default_stop_words_of_the_server() {
    let Some(session) = live_session() else {
        return;
    };
    // As PySpark's StopWordsRemover.loadDefaultStopWords, a call of the ML helper.
    let english =
        spark_connect::ml::StopWordsRemover::load_default_stop_words(&session, "english").unwrap();
    assert_eq!(english.len(), 181);
    assert_eq!(&english[..4], ["i", "me", "my", "myself"]);
    let error = spark_connect::ml::StopWordsRemover::load_default_stop_words(&session, "klingon")
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("klingon is not in the supported language list"),
        "{error}"
    );
    session.stop().unwrap();
}
