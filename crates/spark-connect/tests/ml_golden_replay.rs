// SPDX-License-Identifier: Apache-2.0
//! The offline replay of the ML goldens (tests/common/ml_golden.rs): classification golden
//! entries are served by a loopback replay server, a direct `ml::fetch` makes their requests, and
//! the requests are checked against the commands of the entries. No Spark server is needed.

#[path = "common/ml_golden.rs"]
mod ml_golden;

use std::panic::{catch_unwind, AssertUnwindSafe};

use base64::{engine::general_purpose::STANDARD, Engine};
use ml_golden::*;
use prost::Message;
use serde_json::Value as Json;
use spark_connect::ml::{fetch, CountVectorizerModel, Imputer, MlTarget, StringIndexerModel};
use spark_connect_proto as proto;

const MODELS: &str = include_str!("../../../tests/golden/classification_models.json");
const SUMMARIES: &str = include_str!("../../../tests/golden/classification_summaries.json");
const FEATURE_MODELS: &str = include_str!("../../../tests/golden/feature_models.json");

/// The message of a panic.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a panic with a message")
}

#[test]
fn fetches_replay_the_golden_commands_and_values() {
    let models = fixture(MODELS);
    let summaries = fixture(SUMMARIES);
    let coefficients = &models["models"]["LinearSVC"]["members"]["coefficients"];
    let training = &summaries["summaries"]["LinearSVC.training"]["members"];
    let total = &training["totalIterations"];
    let history = &training["objectiveHistory"];
    let (server, session) = ReplayServer::start(&[coefficients, total, history]);

    // A model's coefficients: a fetch of the model object itself.
    let model = MlTarget::unowned(&session, "<id0>");
    let value = fetch(&model, "coefficients", vec![])
        .unwrap()
        .into_value()
        .unwrap();
    assert_commands(&server, coefficients, "coefficients");
    let coefficients_value = value.as_vector().expect("a vector");
    assert_matches(
        &vector(coefficients_value),
        &coefficients["value"],
        "coefficients",
    );

    // A member of the summary of the model: a fetch of the summary, then of the member.
    let summary = MlTarget::unowned(&session, "<id0>").child("summary");
    let value = fetch(&summary, "totalIterations", vec![])
        .unwrap()
        .into_value()
        .unwrap();
    assert_commands(&server, total, "totalIterations");
    assert_eq!(value.as_i64(), total["value"].as_i64(), "totalIterations");

    let value = fetch(&summary, "objectiveHistory", vec![])
        .unwrap()
        .into_value()
        .unwrap();
    assert_commands(&server, history, "objectiveHistory");
    let values: Vec<f64> = value
        .as_array()
        .expect("an array")
        .iter()
        .map(|v| v.as_f64().expect("a double"))
        .collect();
    assert_matches(&nums(&values), &history["value"], "objectiveHistory");
}

#[test]
fn a_request_that_differs_from_its_golden_fails_the_check() {
    let summaries = fixture(SUMMARIES);
    let total = &summaries["summaries"]["LinearSVC.training"]["members"]["totalIterations"];
    let (server, session) = ReplayServer::start(&[total]);

    // The golden has the fetch of `totalIterations`: a fetch of another member is refused by the
    // replay, and the commands that were sent do not equal the golden's.
    let summary = MlTarget::unowned(&session, "<id0>").child("summary");
    assert!(fetch(&summary, "objectiveHistory", vec![]).is_err());
    let check = catch_unwind(AssertUnwindSafe(|| {
        assert_commands(&server, total, "totalIterations")
    }));
    assert!(check.is_err(), "the differing request passed the check");
}

#[test]
fn every_golden_command_of_the_classification_goldens_is_an_ml_command() {
    fn walk(entry: &Json, commands: &mut usize) {
        let Some(object) = entry.as_object() else {
            return;
        };
        if object.contains_key("commands") {
            for command in golden_commands(entry) {
                assert!(
                    command.command.is_some(),
                    "a golden command with no command"
                );
                *commands += 1;
            }
            return;
        }
        for child in object.values() {
            walk(child, commands);
        }
    }
    let mut commands = 0;
    for text in [MODELS, SUMMARIES] {
        walk(&fixture(text), &mut commands);
    }
    assert!(commands > 0, "no golden commands were found");
}

#[test]
fn a_nan_param_of_a_fit_matches_its_golden() {
    // The Imputer's fit sets `missingValue` to NaN, which is not equal to itself: both sides of
    // the comparison hold the NaN marker.
    let features = fixture(FEATURE_MODELS);
    let fit = &features["fits"]["Imputer"]["fit"];
    let (server, session) = ReplayServer::start(&[fit]);
    // The dataset is not sent: a fit's dataset is cleared by the normalization.
    let df = session.range(1).unwrap();
    let mut imputer = Imputer::new()
        .set_input_col("missing")
        .set_output_col("out")
        .set_missing_value(f64::NAN);
    imputer.fit(&df).unwrap();
    assert_commands(&server, fit, "Imputer fit");
}

#[test]
fn a_uid_in_a_string_argument_of_a_helper_fetch_matches_its_golden() {
    // The helper fetches take the uid of the new model as a string argument, which the golden
    // has as `<uid0>`.
    let features = fixture(FEATURE_MODELS);
    let vocabulary =
        &features["class_methods"]["CountVectorizerModel.from_vocabulary.plain"]["construct"];
    let labels =
        &features["class_methods"]["StringIndexerModel.from_labels.keep-unseen"]["construct"];
    let (server, session) = ReplayServer::start(&[vocabulary, labels]);
    CountVectorizerModel::from_vocabulary(
        &session,
        ["alpha", "beta", "zeta"],
        "words",
        Some("out"),
        None,
        None,
    )
    .unwrap();
    assert_commands(&server, vocabulary, "CountVectorizerModel.from_vocabulary");
    StringIndexerModel::from_labels(&session, ["c", "a"], "category", Some("out"), Some("keep"))
        .unwrap();
    assert_commands(&server, labels, "StringIndexerModel.from_labels");
}

#[test]
fn a_dataframe_member_entry_is_not_replayed_and_the_others_are() {
    // The transform of a QuantileDiscretizer model returns a DataFrame: its golden command is an
    // MlRelation, which is not replayed. The coefficients of the LinearSVC of the classification
    // goldens are replayed beside it.
    let features = fixture(FEATURE_MODELS);
    let transform = &features["models"]["QuantileDiscretizer"]["members"]["transform"];
    let models = fixture(MODELS);
    let coefficients = &models["models"]["LinearSVC"]["members"]["coefficients"];
    let (server, session) = ReplayServer::start(&[transform, coefficients]);

    let command = STANDARD
        .decode(transform["commands"][0].as_str().unwrap())
        .unwrap();
    assert!(proto::MlRelation::decode(command.as_slice()).is_ok());

    let model = MlTarget::unowned(&session, "<id0>");
    fetch(&model, "coefficients", vec![]).unwrap();
    assert_commands(&server, coefficients, "coefficients");

    let golden = catch_unwind(AssertUnwindSafe(|| golden_commands(transform)));
    let checked = catch_unwind(AssertUnwindSafe(|| {
        assert_commands(&server, transform, "transform")
    }));
    for check in [golden.err(), checked.err()] {
        let message = panic_message(check.expect("a relation entry was checked"));
        assert!(message.contains("MlRelation"), "{message}");
        assert!(message.contains("not replayed"), "{message}");
    }
}
