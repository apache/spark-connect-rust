// SPDX-License-Identifier: Apache-2.0
//! Typed Rust feature builders must emit the same parameter literals as official PySpark.
use base64::Engine;
use prost::Message;
use serde_json::Value as Json;
use spark_connect::ml::*;
use spark_connect_proto as proto;

#[test]
fn all_feature_builders_match_official_parameter_literals() {
    let golden: Json =
        serde_json::from_str(include_str!("../../../tests/golden/ml_features.json")).unwrap();
    let stages: Vec<(&str, Params)> = vec![
        (
            "Binarizer",
            Binarizer::new()
                .set_input_col("sample")
                .set_input_cols(["x", "y"])
                .set_output_col("sample")
                .set_output_cols(["x", "y"])
                .set_threshold(0.0)
                .set_thresholds([0.0, 1.0])
                .params()
                .clone(),
        ),
        (
            "BucketedRandomProjectionLSH",
            BucketedRandomProjectionLSH::new()
                .set_bucket_length(0.5)
                .set_input_col("sample")
                .set_num_hash_tables(1)
                .set_output_col("sample")
                .set_seed(42)
                .params()
                .clone(),
        ),
        (
            "Bucketizer",
            Bucketizer::new()
                .set_handle_invalid("error")
                .set_input_col("sample")
                .set_input_cols(["x", "y"])
                .set_output_col("sample")
                .set_output_cols(["x", "y"])
                .set_splits([0.0, 1.0])
                .set_splits_array([[0.0, 1.0], [2.0, 3.0]])
                .params()
                .clone(),
        ),
        (
            "ChiSqSelector",
            ChiSqSelector::new()
                .set_fdr(0.05)
                .set_features_col("features")
                .set_fpr(0.05)
                .set_fwe(0.05)
                .set_label_col("label")
                .set_num_top_features(50)
                .set_output_col("sample")
                .set_percentile(0.1)
                .set_selector_type("numTopFeatures")
                .params()
                .clone(),
        ),
        (
            "CountVectorizer",
            CountVectorizer::new()
                .set_binary(false)
                .set_input_col("sample")
                .set_max_df(9.223372036854776e+18)
                .set_min_df(1.0)
                .set_min_tf(1.0)
                .set_output_col("sample")
                .set_vocab_size(262144)
                .params()
                .clone(),
        ),
        (
            "DCT",
            DCT::new()
                .set_input_col("sample")
                .set_inverse(false)
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "ElementwiseProduct",
            ElementwiseProduct::new()
                .set_input_col("sample")
                .set_output_col("sample")
                .set_scaling_vec([2.0, -1.0])
                .params()
                .clone(),
        ),
        (
            "FeatureHasher",
            FeatureHasher::new()
                .set_categorical_cols(["x", "y"])
                .set_input_cols(["x", "y"])
                .set_num_features(262144)
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "HashingTF",
            HashingTF::new()
                .set_binary(false)
                .set_input_col("sample")
                .set_num_features(262144)
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "IDF",
            IDF::new()
                .set_input_col("sample")
                .set_min_doc_freq(0)
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "Imputer",
            Imputer::new()
                .set_input_col("sample")
                .set_input_cols(["x", "y"])
                .set_missing_value(f64::NAN)
                .set_output_col("sample")
                .set_output_cols(["x", "y"])
                .set_relative_error(0.001)
                .set_strategy("mean")
                .params()
                .clone(),
        ),
        (
            "IndexToString",
            IndexToString::new()
                .set_input_col("sample")
                .set_labels(["x", "y"])
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "Interaction",
            Interaction::new()
                .set_input_cols(["x", "y"])
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "MaxAbsScaler",
            MaxAbsScaler::new()
                .set_input_col("sample")
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "MinHashLSH",
            MinHashLSH::new()
                .set_input_col("sample")
                .set_num_hash_tables(1)
                .set_output_col("sample")
                .set_seed(42)
                .params()
                .clone(),
        ),
        (
            "MinMaxScaler",
            MinMaxScaler::new()
                .set_input_col("sample")
                .set_max(1.0)
                .set_min(0.0)
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "NGram",
            NGram::new()
                .set_input_col("sample")
                .set_n(2)
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "Normalizer",
            Normalizer::new()
                .set_input_col("sample")
                .set_output_col("sample")
                .set_p(2.0)
                .params()
                .clone(),
        ),
        (
            "OneHotEncoder",
            OneHotEncoder::new()
                .set_drop_last(true)
                .set_handle_invalid("error")
                .set_input_col("sample")
                .set_input_cols(["x", "y"])
                .set_output_col("sample")
                .set_output_cols(["x", "y"])
                .params()
                .clone(),
        ),
        (
            "PCA",
            PCA::new()
                .set_input_col("sample")
                .set_k(42)
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "PolynomialExpansion",
            PolynomialExpansion::new()
                .set_degree(2)
                .set_input_col("sample")
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "QuantileDiscretizer",
            QuantileDiscretizer::new()
                .set_handle_invalid("error")
                .set_input_col("sample")
                .set_input_cols(["x", "y"])
                .set_num_buckets(2)
                .set_num_buckets_array([0, 2])
                .set_output_col("sample")
                .set_output_cols(["x", "y"])
                .set_relative_error(0.001)
                .params()
                .clone(),
        ),
        (
            "RobustScaler",
            RobustScaler::new()
                .set_input_col("sample")
                .set_lower(0.25)
                .set_output_col("sample")
                .set_relative_error(0.001)
                .set_upper(0.75)
                .set_with_centering(false)
                .set_with_scaling(true)
                .params()
                .clone(),
        ),
        (
            "RegexTokenizer",
            RegexTokenizer::new()
                .set_gaps(true)
                .set_input_col("sample")
                .set_min_token_length(1)
                .set_output_col("sample")
                .set_pattern("\\s+")
                .set_to_lowercase(true)
                .params()
                .clone(),
        ),
        (
            "RFormula",
            RFormula::new()
                .set_features_col("features")
                .set_force_index_label(false)
                .set_formula("sample")
                .set_handle_invalid("error")
                .set_label_col("label")
                .set_string_indexer_order_type("frequencyDesc")
                .params()
                .clone(),
        ),
        (
            "SQLTransformer",
            SQLTransformer::new()
                .set_statement("sample")
                .params()
                .clone(),
        ),
        (
            "StandardScaler",
            StandardScaler::new()
                .set_input_col("sample")
                .set_output_col("sample")
                .set_with_mean(false)
                .set_with_std(true)
                .params()
                .clone(),
        ),
        (
            "StopWordsRemover",
            StopWordsRemover::new()
                .set_case_sensitive(false)
                .set_input_col("sample")
                .set_input_cols(["x", "y"])
                .set_locale("sample")
                .set_output_col("sample")
                .set_output_cols(["x", "y"])
                .set_stop_words(["x", "y"])
                .params()
                .clone(),
        ),
        (
            "StringIndexer",
            StringIndexer::new()
                .set_handle_invalid("error")
                .set_input_col("sample")
                .set_input_cols(["x", "y"])
                .set_output_col("sample")
                .set_output_cols(["x", "y"])
                .set_string_order_type("frequencyDesc")
                .params()
                .clone(),
        ),
        (
            "TargetEncoder",
            TargetEncoder::new()
                .set_handle_invalid("error")
                .set_input_col("sample")
                .set_input_cols(["x", "y"])
                .set_label_col("label")
                .set_output_col("sample")
                .set_output_cols(["x", "y"])
                .set_smoothing(0.0)
                .set_target_type("binary")
                .params()
                .clone(),
        ),
        (
            "Tokenizer",
            Tokenizer::new()
                .set_input_col("sample")
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "UnivariateFeatureSelector",
            UnivariateFeatureSelector::new()
                .set_feature_type("sample")
                .set_features_col("features")
                .set_label_col("label")
                .set_label_type("sample")
                .set_output_col("sample")
                .set_selection_mode("numTopFeatures")
                .set_selection_threshold(0.5)
                .params()
                .clone(),
        ),
        (
            "VarianceThresholdSelector",
            VarianceThresholdSelector::new()
                .set_features_col("features")
                .set_output_col("sample")
                .set_variance_threshold(0.0)
                .params()
                .clone(),
        ),
        (
            "VectorAssembler",
            VectorAssembler::new()
                .set_handle_invalid("error")
                .set_input_cols(["x", "y"])
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "VectorIndexer",
            VectorIndexer::new()
                .set_handle_invalid("error")
                .set_input_col("sample")
                .set_max_categories(20)
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "VectorSizeHint",
            VectorSizeHint::new()
                .set_handle_invalid("error")
                .set_input_col("sample")
                .set_size(42)
                .params()
                .clone(),
        ),
        (
            "VectorSlicer",
            VectorSlicer::new()
                .set_indices([0, 2])
                .set_input_col("sample")
                .set_names(["x", "y"])
                .set_output_col("sample")
                .params()
                .clone(),
        ),
        (
            "Word2Vec",
            Word2Vec::new()
                .set_input_col("sample")
                .set_max_iter(1)
                .set_max_sentence_length(1000)
                .set_min_count(5)
                .set_num_partitions(1)
                .set_output_col("sample")
                .set_seed(42)
                .set_step_size(0.025)
                .set_vector_size(100)
                .set_window_size(5)
                .params()
                .clone(),
        ),
    ];
    assert_eq!(stages.len(), 38);
    for (name, params) in stages {
        let case = golden["params"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| {
                c["name"] == name
                    && !c["values"].as_object().unwrap().is_empty()
                    && c["values"]
                        .as_object()
                        .unwrap()
                        .values()
                        .any(|v| !v.is_null())
            })
            .unwrap();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(case["params_b64"].as_str().unwrap())
            .unwrap();
        let expected = proto::MlParams::decode(bytes.as_slice()).unwrap();
        let actual = params.to_proto();
        assert_eq!(actual.params.len(), expected.params.len(), "{name}");
        for (key, value) in expected.params {
            assert_eq!(
                actual.params[&key].encode_to_vec(),
                value.encode_to_vec(),
                "{name}.{key}"
            );
        }
    }
}

#[test]
fn feature_defaults_and_nested_array_inference() {
    assert!(StandardScaler::new().with_std());
    assert!(!StandardScaler::new().with_mean());
    assert_eq!(VectorAssembler::new().handle_invalid(), "error");
    assert_eq!(StringIndexer::new().string_order_type(), "frequencyDesc");
    assert_eq!(PCA::new().k(), None);
    assert!(Imputer::new().missing_value().is_nan());
    assert_eq!(QuantileDiscretizer::new().num_buckets(), 2);
    assert_eq!(RegexTokenizer::new().pattern(), "\\s+");
    assert!(RegexTokenizer::new().gaps());
    assert_eq!(StopWordsRemover::new().locale(), None);
    assert_eq!(
        Bucketizer::new()
            .set_splits_array([[0., 1.], [2., 3.]])
            .splits_array(),
        Some(vec![vec![0., 1.], vec![2., 3.]])
    );
    assert!(Params::new()
        .set_param_double_arrays("splitsArray", [vec![], vec![0., 1.]])
        .has_empty_array());
    assert!(!Params::new()
        .set_param_double_arrays("splitsArray", [vec![0., 1.], vec![]])
        .has_empty_array());
}

#[test]
fn elementwise_product_preserves_sparse_scaling_vector() {
    let vector = MlVector::Sparse {
        size: 4,
        indices: vec![1, 3],
        values: vec![2.0, -1.0],
    };
    assert_eq!(
        ElementwiseProduct::new()
            .set_scaling_vec(vector.clone())
            .scaling_vec(),
        Some(vector)
    );
}
