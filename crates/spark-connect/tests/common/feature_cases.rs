// SPDX-License-Identifier: Apache-2.0
//! The stages of the cases of tests/golden/ml_features.json, configured as the feature parity tests
//! configure them, for every test that needs them.
//!
//! `feature_cases!(callback)` invokes `callback! { transformers: [id => stage, ...], estimators:
//! [id => stage, ...] }`, each `stage` an expression that builds the configured stage of the case
//! `id`. A test file includes this module with `#[macro_use]` and defines its callback.

macro_rules! feature_cases {
    ($callback:ident) => {
        $callback! {
            transformers: [
                "Binarizer" => Binarizer::new()
                    .set_input_col("x")
                    .set_output_col("out")
                    .set_threshold(2.5),
                "Bucketizer" => Bucketizer::new()
                    .set_input_col("x")
                    .set_output_col("out")
                    .set_splits([f64::NEG_INFINITY, 2.0, 5.0, f64::INFINITY]),
                "DCT" => DCT::new()
                    .set_input_col("features")
                    .set_output_col("out"),
                "ElementwiseProduct" => ElementwiseProduct::new()
                    .set_input_col("features")
                    .set_output_col("out")
                    .set_scaling_vec([2.0, 1.0, 0.0, -1.0]),
                "FeatureHasher" => FeatureHasher::new()
                    .set_input_cols(["x", "y", "category"])
                    .set_output_col("out")
                    .set_num_features(8),
                "HashingTF" => HashingTF::new()
                    .set_input_col("words")
                    .set_output_col("out")
                    .set_num_features(8),
                "IndexToString" => IndexToString::new()
                    .set_input_col("index")
                    .set_output_col("out")
                    .set_labels(["a", "b", "c"]),
                "Interaction" => Interaction::new()
                    .set_input_cols(["x", "y"])
                    .set_output_col("out"),
                "NGram" => NGram::new()
                    .set_input_col("words")
                    .set_output_col("out")
                    .set_n(2),
                "Normalizer" => Normalizer::new()
                    .set_input_col("features")
                    .set_output_col("out")
                    .set_p(1.0),
                "PolynomialExpansion" => PolynomialExpansion::new()
                    .set_input_col("features")
                    .set_output_col("out")
                    .set_degree(2),
                "RegexTokenizer" => RegexTokenizer::new()
                    .set_input_col("text")
                    .set_output_col("out")
                    .set_pattern("\\s+")
                    .set_gaps(true),
                "SQLTransformer" => SQLTransformer::new()
                    .set_statement("SELECT *, x + y AS out FROM __THIS__"),
                "StopWordsRemover" => StopWordsRemover::new()
                    .set_input_col("words")
                    .set_output_col("out")
                    .set_stop_words(["the", "gamma"]),
                "Tokenizer" => Tokenizer::new()
                    .set_input_col("text")
                    .set_output_col("out"),
                "VectorAssembler" => VectorAssembler::new()
                    .set_input_cols(["x", "y"])
                    .set_output_col("out")
                    .set_handle_invalid("error"),
                "VectorSizeHint" => VectorSizeHint::new()
                    .set_input_col("rawFeatures")
                    .set_size(4),
                "VectorSlicer" => VectorSlicer::new()
                    .set_input_col("features")
                    .set_output_col("out")
                    .set_indices([0, 2]),
                "Binarizer-multi" => Binarizer::new()
                    .set_input_cols(["x", "y"])
                    .set_output_cols(["out", "other"])
                    .set_thresholds([2.0, 4.0]),
                "Bucketizer-multi" => Bucketizer::new()
                    .set_input_cols(["x", "y"])
                    .set_output_cols(["out", "other"])
                    .set_splits_array([
                        [f64::NEG_INFINITY, 3.0, f64::INFINITY],
                        [f64::NEG_INFINITY, 5.0, f64::INFINITY],
                    ]),
                "StopWordsRemover-multi" => StopWordsRemover::new()
                    .set_input_cols(["words", "words2"])
                    .set_output_cols(["out", "other"])
                    .set_stop_words(["the", "gamma"]),
                "RegexTokenizer-variant" => RegexTokenizer::new()
                    .set_input_col("text")
                    .set_output_col("out")
                    .set_pattern("[a-z]+")
                    .set_gaps(false)
                    .set_to_lowercase(false),
            ],
            estimators: [
                "BucketedRandomProjectionLSH" => BucketedRandomProjectionLSH::new()
                    .set_input_col("features")
                    .set_output_col("out")
                    .set_num_hash_tables(2)
                    .set_bucket_length(1.5)
                    .set_seed(42),
                "ChiSqSelector" => ChiSqSelector::new()
                    .set_features_col("features")
                    .set_output_col("out")
                    .set_label_col("label")
                    .set_num_top_features(2),
                "CountVectorizer" => CountVectorizer::new()
                    .set_input_col("words")
                    .set_output_col("out")
                    .set_vocab_size(4),
                "IDF" => IDF::new()
                    .set_input_col("features")
                    .set_output_col("out"),
                "Imputer" => Imputer::new()
                    .set_input_col("missing")
                    .set_output_col("out")
                    .set_missing_value(f64::NAN),
                "MaxAbsScaler" => MaxAbsScaler::new()
                    .set_input_col("features")
                    .set_output_col("out"),
                "MinHashLSH" => MinHashLSH::new()
                    .set_input_col("features")
                    .set_output_col("out")
                    .set_num_hash_tables(2)
                    .set_seed(42),
                "MinMaxScaler" => MinMaxScaler::new()
                    .set_input_col("features")
                    .set_output_col("out")
                    .set_min(-1.0)
                    .set_max(1.0),
                "OneHotEncoder" => OneHotEncoder::new()
                    .set_input_col("index")
                    .set_output_col("out")
                    .set_drop_last(false),
                "PCA" => PCA::new()
                    .set_input_col("features")
                    .set_output_col("out")
                    .set_k(2),
                "QuantileDiscretizer" => QuantileDiscretizer::new()
                    .set_input_col("x")
                    .set_output_col("out")
                    .set_num_buckets(3)
                    .set_relative_error(0.0),
                "RobustScaler" => RobustScaler::new()
                    .set_input_col("features")
                    .set_output_col("out")
                    .set_with_centering(true)
                    .set_relative_error(0.0),
                "RFormula" => RFormula::new()
                    .set_formula("label ~ x + category")
                    .set_features_col("out")
                    .set_label_col("target")
                    .set_string_indexer_order_type("alphabetAsc"),
                "StandardScaler" => StandardScaler::new()
                    .set_input_col("features")
                    .set_output_col("out")
                    .set_with_mean(true)
                    .set_with_std(true),
                "StringIndexer" => StringIndexer::new()
                    .set_input_col("category")
                    .set_output_col("out")
                    .set_string_order_type("alphabetAsc"),
                "TargetEncoder" => TargetEncoder::new()
                    .set_input_col("index")
                    .set_output_col("out")
                    .set_label_col("label")
                    .set_smoothing(1.0),
                "UnivariateFeatureSelector" => UnivariateFeatureSelector::new()
                    .set_features_col("features")
                    .set_output_col("out")
                    .set_label_col("label")
                    .set_feature_type("categorical")
                    .set_label_type("categorical")
                    .set_selection_threshold(2.0),
                "VarianceThresholdSelector" => VarianceThresholdSelector::new()
                    .set_features_col("features")
                    .set_output_col("out")
                    .set_variance_threshold(1.0),
                "VectorIndexer" => VectorIndexer::new()
                    .set_input_col("features")
                    .set_output_col("out")
                    .set_max_categories(2),
                "Word2Vec" => Word2Vec::new()
                    .set_input_col("words")
                    .set_output_col("out")
                    .set_vector_size(3)
                    .set_min_count(1)
                    .set_seed(42)
                    .set_num_partitions(1)
                    .set_max_iter(2)
                    .set_window_size(2),
                "Imputer-multi" => Imputer::new()
                    .set_input_cols(["missing", "x"])
                    .set_output_cols(["out", "other"])
                    .set_strategy("median"),
                "OneHotEncoder-multi" => OneHotEncoder::new()
                    .set_input_cols(["index", "label"])
                    .set_output_cols(["out", "other"])
                    .set_drop_last(false),
                "StringIndexer-multi" => StringIndexer::new()
                    .set_input_cols(["category", "text"])
                    .set_output_cols(["out", "other"])
                    .set_string_order_type("alphabetAsc"),
                "TargetEncoder-multi" => TargetEncoder::new()
                    .set_input_cols(["index", "label"])
                    .set_output_cols(["out", "other"])
                    .set_label_col("x")
                    .set_target_type("continuous"),
                "QuantileDiscretizer-multi" => QuantileDiscretizer::new()
                    .set_input_cols(["x", "y"])
                    .set_output_cols(["out", "other"])
                    .set_num_buckets_array([2, 3])
                    .set_relative_error(0.0),
                "StandardScaler-variant" => StandardScaler::new()
                    .set_input_col("features")
                    .set_output_col("out")
                    .set_with_mean(false)
                    .set_with_std(false),
            ],
        }
    };
}
