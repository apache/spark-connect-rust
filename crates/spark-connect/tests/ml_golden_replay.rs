// SPDX-License-Identifier: Apache-2.0
//! The offline replay of the ML goldens (tests/common/ml_golden.rs): classification golden
//! entries are served by a loopback replay server, a direct `ml::fetch` makes their requests, and
//! the requests are checked against the commands of the entries. No Spark server is needed.

#[path = "common/ml_golden.rs"]
mod ml_golden;

use std::panic::{catch_unwind, AssertUnwindSafe};

use ml_golden::*;
use serde_json::Value as Json;
use spark_connect::ml::{fetch, MlTarget};

const MODELS: &str = include_str!("../../../tests/golden/classification_models.json");
const SUMMARIES: &str = include_str!("../../../tests/golden/classification_summaries.json");

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
