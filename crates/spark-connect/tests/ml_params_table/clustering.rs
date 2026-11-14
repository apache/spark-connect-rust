//! The clustering classes of the completeness test in `ml_params_table.rs`.

use spark_connect::ml::*;

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[
    ("BisectingKMeans", BisectingKMeans::PARAM_NAMES),
    ("BisectingKMeansModel", BisectingKMeansModel::PARAM_NAMES),
    ("DistributedLDAModel", DistributedLDAModel::PARAM_NAMES),
    ("GaussianMixture", GaussianMixture::PARAM_NAMES),
    ("GaussianMixtureModel", GaussianMixtureModel::PARAM_NAMES),
    ("KMeans", KMeans::PARAM_NAMES),
    ("KMeansModel", KMeansModel::PARAM_NAMES),
    ("LDA", LDA::PARAM_NAMES),
    ("LDAModel", LDAModel::PARAM_NAMES),
    ("LocalLDAModel", LocalLDAModel::PARAM_NAMES),
    (
        "PowerIterationClustering",
        PowerIterationClustering::PARAM_NAMES,
    ),
];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[];
