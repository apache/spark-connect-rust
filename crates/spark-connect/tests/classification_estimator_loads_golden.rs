// SPDX-License-Identifier: Apache-2.0
//! The loads of the NaiveBayes, MultilayerPerceptronClassifier and FMClassifier estimators,
//! replayed against the golden of official PySpark 4.3.0 with no server: `X.load(path)` is one ML
//! Read of the estimator, of the saved uid and params, and the loaded estimator saves as PySpark's
//! `write().save(path)` does.
//!
//! The goldens are replayed by the replay server of `common/ml_golden.rs`, which answers each ML
//! command with the golden result of the same command.

#[path = "common/ml_golden.rs"]
mod ml_golden;

use std::collections::BTreeSet;

use base64::{engine::general_purpose::STANDARD, Engine};
use ml_golden::{assert_commands, ReplayServer};
use prost::Message;
use serde_json::Value;
use spark_connect::ml::{Estimator, FMClassifier, MultilayerPerceptronClassifier, NaiveBayes};
use spark_connect::SparkSession;
use spark_connect_core::error::Result;
use spark_connect_proto as proto;

const PERSISTENCE: &str = include_str!("../../../tests/golden/classification_persistence.json");

/// The uid (a placeholder, as the golden normalizes it) and the names of the params of the
/// operator info that a golden load read.
fn read_info(load: &Value) -> (Option<String>, BTreeSet<String>) {
    let result = proto::MlCommandResult::decode(
        STANDARD
            .decode(load["results"][0].as_str().unwrap())
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    match result.result_type {
        Some(proto::ml_command_result::ResultType::OperatorInfo(info)) => {
            let names = info.params.map(|p| p.params.into_keys().collect());
            (info.uid, names.unwrap_or_default())
        }
        other => panic!("not an operator info: {other:?}"),
    }
}

/// The load of the golden entry `key` with `load`, then the save of the loaded estimator: their
/// commands are the golden's, and the loaded estimator has the uid and the params read. The values
/// of the params are compared by the save, whose Write sends them as PySpark re-sends a loaded
/// object's params (an in-range long `seed` as an integer, see `read_params`).
fn check_load_and_save<E: Estimator>(key: &str, load: impl Fn(&SparkSession, &str) -> Result<E>) {
    let persistence: Value = serde_json::from_str(PERSISTENCE).unwrap();
    let entry = &persistence["persistence"][key];
    assert!(entry.is_object(), "no golden entry {key}");
    let (server, session) = ReplayServer::start(&[entry]);
    let estimator = load(&session, "/unused/path").unwrap();
    assert_commands(&server, &entry["load"], key);
    let (uid, names) = read_info(&entry["load"]);
    assert_eq!(Some(estimator.operator().uid.clone()), uid, "{key}");
    let loaded: BTreeSet<String> = estimator.params().to_proto().params.into_keys().collect();
    assert_eq!(loaded, names, "{key}");
    estimator
        .write()
        .session(&session)
        .overwrite()
        .save("/unused/path")
        .unwrap();
    assert_commands(&server, &entry["save"], key);
}

#[test]
fn naive_bayes_estimators_load_and_save_as_official_pyspark() {
    for model_type in ["multinomial", "complement", "bernoulli", "gaussian"] {
        check_load_and_save(
            &format!("NaiveBayes.{model_type}.estimator"),
            NaiveBayes::load,
        );
    }
}

#[test]
fn multilayer_perceptron_classifier_loads_and_saves_as_official_pyspark() {
    check_load_and_save(
        "MultilayerPerceptronClassifier.estimator",
        MultilayerPerceptronClassifier::load,
    );
}

#[test]
fn fm_classifier_loads_and_saves_as_official_pyspark() {
    check_load_and_save("FMClassifier.estimator", FMClassifier::load);
}
