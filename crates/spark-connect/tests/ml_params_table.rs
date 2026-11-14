// SPDX-License-Identifier: Apache-2.0
//! The `ml_params!` tables cover PySpark's params: every class of the param golden is either
//! converted, its `PARAM_NAMES` equal to the golden's param names, or excluded with a reason.
//!
//! The golden is `tests/golden/ml_api.json`, captured from official pyspark-client 4.3.0. The
//! test runs offline and reads the file at run time.

use serde_json::Value as Json;
use std::collections::{BTreeMap, BTreeSet};

// Each area's tables live in their own file under `ml_params_table/`, so that each area's PR edits
// only its file.
#[path = "ml_params_table/classification.rs"]
mod classification;
#[path = "ml_params_table/clustering.rs"]
mod clustering;
#[path = "ml_params_table/evaluation.rs"]
mod evaluation;
#[path = "ml_params_table/feature.rs"]
mod feature;
#[path = "ml_params_table/fpm.rs"]
mod fpm;
#[path = "ml_params_table/no_rust_type.rs"]
mod no_rust_type;
#[path = "ml_params_table/recommendation.rs"]
mod recommendation;
#[path = "ml_params_table/regression.rs"]
mod regression;
#[path = "ml_params_table/tuning_pipeline.rs"]
mod tuning_pipeline;

/// The golden classes whose params are converted, with the `PARAM_NAMES` of their Rust type.
fn converted() -> Vec<(&'static str, &'static [&'static str])> {
    [
        classification::CONVERTED,
        regression::CONVERTED,
        clustering::CONVERTED,
        feature::CONVERTED,
        fpm::CONVERTED,
        recommendation::CONVERTED,
        evaluation::CONVERTED,
        tuning_pipeline::CONVERTED,
        no_rust_type::CONVERTED,
    ]
    .concat()
}

/// The golden classes that are not converted, each with its reason.
fn excluded() -> Vec<(&'static str, &'static str)> {
    [
        classification::EXCLUDED,
        regression::EXCLUDED,
        clustering::EXCLUDED,
        feature::EXCLUDED,
        fpm::EXCLUDED,
        recommendation::EXCLUDED,
        evaluation::EXCLUDED,
        tuning_pipeline::EXCLUDED,
        no_rust_type::EXCLUDED,
    ]
    .concat()
}

fn golden() -> Json {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/ml_api.json"
    );
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read the param golden {path}: {e}"));
    serde_json::from_str(&text).expect("the param golden is valid JSON")
}

/// The golden param names of the class `class`, as a sorted set.
fn golden_names(golden: &Json, class: &str) -> BTreeSet<String> {
    golden[class]["params"]
        .as_array()
        .unwrap_or_else(|| panic!("{class} has no params array in the golden"))
        .iter()
        .map(|param| param["name"].as_str().expect("a param name").to_string())
        .collect()
}

#[test]
fn every_golden_class_is_converted_or_excluded_exactly_once() {
    let golden = golden();
    let golden_classes: BTreeSet<&str> = golden
        .as_object()
        .expect("the golden is an object")
        .keys()
        .map(String::as_str)
        .collect();

    let mut listed: BTreeMap<&str, &str> = BTreeMap::new();
    for (class, _) in converted() {
        let previous = listed.insert(class, "converted");
        assert!(previous.is_none(), "{class} is listed twice");
    }
    for (class, reason) in excluded() {
        assert!(!reason.is_empty(), "{class} is excluded without a reason");
        let previous = listed.insert(class, "excluded");
        assert!(
            previous.is_none(),
            "{class} is listed twice, as {} and as excluded",
            previous.unwrap_or("")
        );
    }

    let unknown: Vec<&str> = listed
        .keys()
        .copied()
        .filter(|class| !golden_classes.contains(class))
        .collect();
    assert!(
        unknown.is_empty(),
        "listed but not in the golden: {unknown:?}"
    );

    let unlisted: Vec<&str> = golden_classes
        .iter()
        .copied()
        .filter(|class| !listed.contains_key(class))
        .collect();
    assert!(
        unlisted.is_empty(),
        "golden classes in neither CONVERTED nor EXCLUDED: {unlisted:?}"
    );
    assert_eq!(listed.len(), golden_classes.len());
}

#[test]
fn converted_classes_have_the_golden_param_names() {
    let golden = golden();
    for (class, rust_names) in converted() {
        let rust: Vec<&str> = rust_names.to_vec();
        let rust_set: BTreeSet<&str> = rust.iter().copied().collect();
        assert_eq!(
            rust_set.len(),
            rust.len(),
            "{class}::PARAM_NAMES lists a name twice: {rust:?}"
        );
        let expected = golden_names(&golden, class);
        let expected_set: BTreeSet<&str> = expected.iter().map(String::as_str).collect();
        let missing: Vec<&&str> = expected_set.difference(&rust_set).collect();
        let extra: Vec<&&str> = rust_set.difference(&expected_set).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "{class}::PARAM_NAMES differs from the golden: missing {missing:?}, extra {extra:?}"
        );
    }
}
