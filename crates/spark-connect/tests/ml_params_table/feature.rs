//! The feature classes of the completeness test in `ml_params_table.rs`.

use spark_connect::ml::*;

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[
    ("Binarizer", Binarizer::PARAM_NAMES),
    (
        "BucketedRandomProjectionLSH",
        BucketedRandomProjectionLSH::PARAM_NAMES,
    ),
    ("Bucketizer", Bucketizer::PARAM_NAMES),
    ("ChiSqSelector", ChiSqSelector::PARAM_NAMES),
    ("CountVectorizer", CountVectorizer::PARAM_NAMES),
    ("DCT", DCT::PARAM_NAMES),
    ("ElementwiseProduct", ElementwiseProduct::PARAM_NAMES),
    ("FeatureHasher", FeatureHasher::PARAM_NAMES),
    ("HashingTF", HashingTF::PARAM_NAMES),
    ("IDF", IDF::PARAM_NAMES),
    ("Imputer", Imputer::PARAM_NAMES),
    ("IndexToString", IndexToString::PARAM_NAMES),
    ("Interaction", Interaction::PARAM_NAMES),
    ("MinHashLSH", MinHashLSH::PARAM_NAMES),
    ("MinMaxScaler", MinMaxScaler::PARAM_NAMES),
    ("NGram", NGram::PARAM_NAMES),
    ("Normalizer", Normalizer::PARAM_NAMES),
    ("OneHotEncoder", OneHotEncoder::PARAM_NAMES),
    ("PCA", PCA::PARAM_NAMES),
    ("PolynomialExpansion", PolynomialExpansion::PARAM_NAMES),
    ("QuantileDiscretizer", QuantileDiscretizer::PARAM_NAMES),
    ("RFormula", RFormula::PARAM_NAMES),
    ("RegexTokenizer", RegexTokenizer::PARAM_NAMES),
    ("RobustScaler", RobustScaler::PARAM_NAMES),
    ("SQLTransformer", SQLTransformer::PARAM_NAMES),
    ("StopWordsRemover", StopWordsRemover::PARAM_NAMES),
    ("TargetEncoder", TargetEncoder::PARAM_NAMES),
    ("Tokenizer", Tokenizer::PARAM_NAMES),
    (
        "UnivariateFeatureSelector",
        UnivariateFeatureSelector::PARAM_NAMES,
    ),
    (
        "VarianceThresholdSelector",
        VarianceThresholdSelector::PARAM_NAMES,
    ),
    ("VectorIndexer", VectorIndexer::PARAM_NAMES),
    ("VectorSizeHint", VectorSizeHint::PARAM_NAMES),
    ("VectorSlicer", VectorSlicer::PARAM_NAMES),
    ("Word2Vec", Word2Vec::PARAM_NAMES),
];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[
    ("BucketedRandomProjectionLSHModel", "pending: feature"),
    ("ChiSqSelectorModel", "pending: feature"),
    ("CountVectorizerModel", "pending: feature"),
    ("IDFModel", "pending: feature"),
    ("ImputerModel", "pending: feature"),
    ("MaxAbsScaler", "pending: feature"),
    ("MaxAbsScalerModel", "pending: feature"),
    ("MinHashLSHModel", "pending: feature"),
    ("MinMaxScalerModel", "pending: feature"),
    ("OneHotEncoderModel", "pending: feature"),
    ("PCAModel", "pending: feature"),
    ("RFormulaModel", "pending: feature"),
    ("RobustScalerModel", "pending: feature"),
    (
        "StandardScaler",
        "pending: feature (rows differ: missing inputCol, outputCol, withMean)",
    ),
    ("StandardScalerModel", "pending: feature"),
    (
        "StringIndexer",
        "pending: feature (rows differ: missing inputCol, outputCol)",
    ),
    ("StringIndexerModel", "pending: feature"),
    ("TargetEncoderModel", "pending: feature"),
    ("UnivariateFeatureSelectorModel", "pending: feature"),
    ("VarianceThresholdSelectorModel", "pending: feature"),
    (
        "VectorAssembler",
        "pending: feature (rows differ: missing inputCols, outputCol)",
    ),
    ("VectorIndexerModel", "pending: feature"),
    ("Word2VecModel", "pending: feature"),
];
