//! The clustering classes of the completeness test in `ml_params_table.rs`.

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[
    ("BisectingKMeans", "pending: clustering"),
    ("BisectingKMeansModel", "pending: clustering"),
    ("DistributedLDAModel", "pending: clustering"),
    ("GaussianMixture", "pending: clustering"),
    ("GaussianMixtureModel", "pending: clustering"),
    ("KMeans", "pending: clustering"),
    ("KMeansModel", "pending: clustering"),
    ("LDA", "pending: clustering"),
    ("LDAModel", "pending: clustering"),
    ("LocalLDAModel", "pending: clustering"),
    ("PowerIterationClustering", "pending: clustering"),
];
