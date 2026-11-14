// SPDX-License-Identifier: Apache-2.0
//! FPGrowth, ALS and the evaluators of official PySpark 4.3.0, replayed against the typed Rust
//! accessors, with no Spark server. The goldens are tests/golden/fpm_recommendation_evaluation.json
//! (the fits, the models and the evaluators, with their members) and
//! tests/golden/fpm_recommendation_evaluation_persistence.json (their save and load). The replay
//! server of `common/ml_golden.rs` answers each ML command with the golden result of that command,
//! and the test checks that the accessor sent the golden commands and returned the golden value
//! (or an error carrying the golden message). The estimators are configured as the cases of
//! frequent_patterns.json, als.json and evaluators.json, which the golden was captured from.
//!
//! Replayed: the fit of every FPGrowth and ALS case, the evaluate and isLargerBetter of each
//! evaluator, the save and load of every FPGrowth and ALS estimator, the save of every ALS model,
//! and the save and load of every evaluator. Skipped, by [`skip_reason`]: the DataFrame-valued
//! members (their rows are Arrow, which the replay does not serve), the repr() and str() of the
//! Python objects, and the rank members (see [`ALS_RANK`]). Left out with their findings: the
//! fetches of rank in the ALS fits, and the loads of the ALS models (see [`ALS_RANK`]).
#[path = "common/als.rs"]
mod als;
#[path = "common/evaluators.rs"]
mod evaluators;
#[path = "common/ml_golden.rs"]
mod ml_golden;

use std::collections::BTreeSet;
use std::fmt::Display;
use std::panic::{catch_unwind, AssertUnwindSafe};

use base64::{engine::general_purpose::STANDARD, Engine};
use ml_golden::{
    assert_commands, assert_error, assert_matches, assert_params, fixture, num, ReplayServer,
};
use prost::Message;
use serde_json::{json, Value as Json};
use spark_connect::ml::*;
use spark_connect::{DataFrame, SparkSession};
use spark_connect_proto as proto;

const MODELS: &str = include_str!("../../../tests/golden/fpm_recommendation_evaluation.json");
const PERSISTENCE: &str =
    include_str!("../../../tests/golden/fpm_recommendation_evaluation_persistence.json");
const FREQUENT_PATTERNS: &str = include_str!("../../../tests/golden/frequent_patterns.json");

/// Finding: PySpark reads the rank of an ALS model from the server: its fit fetches `rank` twice
/// after the Fit, and each read of `rank` fetches it. The Rust `rank()` is a field (of the
/// estimator, or of the read of a load), with no command; and the Rust `ALSModel.load` fetches
/// `rank` after its read, which PySpark's load does not. The replay leaves the fetches out of the
/// golden fit and the `rank` members, checks the Fit command, and leaves the loads of the ALS
/// models out.
const ALS_RANK: &str = "PySpark fetches rank from the server; the Rust rank() sends no command";

/// The path of every save and load. The replay server does not read it: the golden commands
/// normalize it to `<path>`.
const PATH: &str = "/unused/path";

/// The configuration of an FPGrowth case of frequent_patterns.json, as the live test gives it.
fn fpgrowth(values: &Json) -> FPGrowth {
    let mut estimator = FPGrowth::new();
    for (name, value) in values.as_object().unwrap() {
        estimator = match name.as_str() {
            "minSupport" => estimator.set_min_support(value.as_f64().unwrap()),
            "minConfidence" => estimator.set_min_confidence(value.as_f64().unwrap()),
            "itemsCol" => estimator.set_items_col(value.as_str().unwrap()),
            "predictionCol" => estimator.set_prediction_col(value.as_str().unwrap()),
            "numPartitions" => {
                estimator.set_num_partitions(value.as_i64().unwrap().try_into().unwrap())
            }
            other => panic!("unexpected FPGrowth parameter: {other}"),
        };
    }
    estimator
}

fn load_evaluator(class: &str, session: &SparkSession, path: &str) -> Box<dyn Evaluator> {
    match class {
        "BinaryClassificationEvaluator" => {
            Box::new(BinaryClassificationEvaluator::load(session, path).unwrap())
        }
        "RegressionEvaluator" => Box::new(RegressionEvaluator::load(session, path).unwrap()),
        "MulticlassClassificationEvaluator" => {
            Box::new(MulticlassClassificationEvaluator::load(session, path).unwrap())
        }
        "MultilabelClassificationEvaluator" => {
            Box::new(MultilabelClassificationEvaluator::load(session, path).unwrap())
        }
        "ClusteringEvaluator" => Box::new(ClusteringEvaluator::load(session, path).unwrap()),
        "RankingEvaluator" => Box::new(RankingEvaluator::load(session, path).unwrap()),
        other => panic!("uncovered evaluator {other}"),
    }
}

/// Decodes a golden ML command.
fn decode_command(command: &Json) -> proto::MlCommand {
    let bytes = STANDARD.decode(command.as_str().unwrap()).unwrap();
    proto::MlCommand::decode(bytes.as_slice()).unwrap()
}

/// The entry without the commands that `drop` picks, and their results. The replay does not check
/// the commands that the Rust client does not send, so they are left out of the golden entry
/// that holds them.
fn without(entry: &Json, drop: impl Fn(&proto::MlCommand) -> bool) -> Json {
    let mut entry = entry.clone();
    let commands = entry["commands"].as_array().unwrap().clone();
    let kept: Vec<usize> = (0..commands.len())
        .filter(|&i| !drop(&decode_command(&commands[i])))
        .collect();
    if let Some(results) = entry["results"].as_array().cloned() {
        entry["results"] = kept
            .iter()
            .filter_map(|&i| results.get(i).cloned())
            .collect();
    }
    entry["commands"] = kept.iter().map(|&i| commands[i].clone()).collect();
    entry
}

/// The golden fit without the fetches of `rank` that PySpark sends after its Fit (see
/// [`ALS_RANK`]).
fn without_rank_fetches(fit: &Json) -> Json {
    without(fit, |command| match &command.command {
        Some(proto::ml_command::Command::Fetch(fetch)) => {
            fetch.methods.iter().any(|method| method.method == "rank")
        }
        _ => false,
    })
}

/// The outcome of a member: its value, or the message of its error.
type Outcome = std::result::Result<Option<Json>, String>;

fn outcome<T, E: Display>(
    result: std::result::Result<T, E>,
    render: impl FnOnce(T) -> Json,
) -> Outcome {
    result
        .map(|value| Some(render(value)))
        .map_err(|error| error.to_string())
}

/// Why the replay does not check a member of a golden entry, or None when it checks it.
fn skip_reason(label: &str, entry: &Json) -> Option<&'static str> {
    if label == "repr()" || label == "str()" {
        return Some("the repr() and str() of the Python object");
    }
    if label == "rank" {
        return Some(ALS_RANK);
    }
    if entry["value"]["rows"].is_array() {
        return Some("a DataFrame: its rows are Arrow, which the replay does not serve");
    }
    None
}

/// The entries of the members of a golden entry that the replay checks.
fn checked_members(members: &Json) -> Vec<&Json> {
    members
        .as_object()
        .unwrap()
        .iter()
        .filter(|(label, entry)| skip_reason(label, entry).is_none())
        .map(|(_, entry)| entry)
        .collect()
}

/// Checks each member of a golden `members` map: its value or its error, and its commands. The
/// members that `dispatch` gives no accessor for must be skipped, as [`skip_reason`] gives.
fn check_members(
    server: &ReplayServer,
    members: &Json,
    dispatch: &dyn Fn(&str) -> Option<Outcome>,
    key: &str,
) {
    for (label, entry) in members.as_object().unwrap() {
        if skip_reason(label, entry).is_some() {
            continue;
        }
        let at = format!("{key}.{label}");
        let outcome = dispatch(label).unwrap_or_else(|| panic!("{at}: no accessor"));
        match outcome {
            Ok(Some(actual)) => assert_matches(&actual, &entry["value"], &at),
            Ok(None) => panic!("{at}: no value, official {entry}"),
            Err(message) => assert_error(&message, &entry["error"], &at),
        }
        assert_commands(server, entry, &at);
    }
}

/// The member of an evaluator: its metric, and whether the metric is larger for a better model.
fn evaluator_member(evaluator: &dyn Evaluator, label: &str, input: &DataFrame) -> Option<Outcome> {
    Some(match label {
        "evaluate" => outcome(evaluator.evaluate(input), num),
        "isLargerBetter()" => Ok(Some(json!(evaluator.is_larger_better()))),
        _ => return None,
    })
}

/// The uid that a golden load read: the golden normalizes it to a placeholder, so the loaded uid
/// is the placeholder too.
fn result_uid(load: &Json) -> String {
    let result = proto::MlCommandResult::decode(
        STANDARD
            .decode(load["results"][0].as_str().unwrap())
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    match result.result_type {
        Some(proto::ml_command_result::ResultType::OperatorInfo(info)) => info.uid.unwrap(),
        other => panic!("not an operator info: {other:?}"),
    }
}

/// The save and load of a stage, against its golden record: the record's uid and params are those
/// that the golden load read, and the loaded params are those the stage was saved with.
fn assert_loaded(
    record: &Json,
    loaded: (&MlOperator, &Params),
    saved: (&MlOperator, &Params),
    key: &str,
) {
    assert_eq!(record["uid_same"], true, "{key}");
    assert_eq!(record["params_before"], record["params_after"], "{key}");
    assert_eq!(loaded.0.name, saved.0.name, "{key}");
    assert_eq!(loaded.0.uid, result_uid(&record["load"]), "{key}");
    assert_params(loaded.1, saved.1, key);
}

/// Every FPGrowth case: its fit, its model's save and load, and its estimator's save and load.
/// The members of FPGrowth models are DataFrames, all skipped.
#[test]
fn fpgrowth_models_and_estimators_match_official_pyspark() {
    let golden = fixture(MODELS);
    let records = fixture(PERSISTENCE)["persistence"].clone();
    let mut seen = BTreeSet::new();
    for case in fixture(FREQUENT_PATTERNS)["fpgrowth"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let key = format!("FPGrowth.{id}");
        let fit = &golden["fits"][&key];
        assert_eq!(
            fit["dataset"],
            format!("fpm_{}", id.replace('-', "_")),
            "{key}"
        );
        let model_key = format!("{key}.model");
        let estimator_key = format!("{key}.estimator");
        let model_record = &records[&model_key];
        let estimator_record = &records[&estimator_key];

        let mut entries = checked_members(&golden["models"][&key]["members"]);
        entries.extend(checked_members(&model_record["members"]["members"]));
        entries.extend([
            &fit["fit"],
            &model_record["save"],
            &model_record["load"],
            &estimator_record["save"],
            &estimator_record["load"],
        ]);
        let (server, session) = ReplayServer::start(&entries);
        let input = session.range(1).unwrap();

        let estimator = fpgrowth(&case["values"]);
        let fitted = fpgrowth(&case["values"]).fit(&input).unwrap();
        assert_commands(&server, &fit["fit"], &key);
        check_members(&server, &golden["models"][&key]["members"], &|_| None, &key);

        fitted.write().unwrap().overwrite().save(PATH).unwrap();
        assert_commands(&server, &model_record["save"], &model_key);
        let loaded = FPGrowthModel::load(&session, PATH).unwrap();
        assert_commands(&server, &model_record["load"], &model_key);
        assert_loaded(
            model_record,
            (Transformer::operator(&loaded), Transformer::params(&loaded)),
            (Transformer::operator(&fitted), Transformer::params(&fitted)),
            &model_key,
        );
        check_members(
            &server,
            &model_record["members"]["members"],
            &|_| None,
            &model_key,
        );
        seen.insert(model_key);

        Estimator::write(&estimator)
            .session(&session)
            .overwrite()
            .save(PATH)
            .unwrap();
        assert_commands(&server, &estimator_record["save"], &estimator_key);
        let loaded = FPGrowth::load(&session, PATH).unwrap();
        assert_commands(&server, &estimator_record["load"], &estimator_key);
        assert_loaded(
            estimator_record,
            (Estimator::operator(&loaded), Estimator::params(&loaded)),
            (
                Estimator::operator(&estimator),
                Estimator::params(&estimator),
            ),
            &estimator_key,
        );
        seen.insert(estimator_key);
    }
    let official: BTreeSet<String> = golden["fits"]
        .as_object()
        .unwrap()
        .keys()
        .filter(|key| key.starts_with("FPGrowth."))
        .flat_map(|key| [format!("{key}.model"), format!("{key}.estimator")])
        .collect();
    assert_eq!(seen, official, "the FPGrowth records of the golden");
}

/// Every ALS case: its fit, its model's save, and its estimator's save and load. The rank of the
/// model and the load of the model are left out (see [`ALS_RANK`]).
#[test]
fn als_models_and_estimators_match_official_pyspark() {
    let golden = fixture(MODELS);
    let records = fixture(PERSISTENCE)["persistence"].clone();
    let als_fixture = als::fixture();
    let mut seen = BTreeSet::new();
    for case in als_fixture["cases"].as_array().unwrap() {
        let key = format!("ALS.{}", case["name"].as_str().unwrap());
        let fit = &golden["fits"][&key];
        assert_eq!(fit["dataset"], "als_train", "{key}");
        let model_key = format!("{key}.model");
        let estimator_key = format!("{key}.estimator");
        let model_record = &records[&model_key];
        let estimator_record = &records[&estimator_key];

        let fit_entry = without_rank_fetches(&fit["fit"]);
        let mut entries = checked_members(&golden["models"][&key]["members"]);
        entries.extend(checked_members(&model_record["members"]["members"]));
        entries.extend([
            &fit_entry,
            &model_record["save"],
            &model_record["load"],
            &estimator_record["save"],
            &estimator_record["load"],
        ]);
        let (server, session) = ReplayServer::start(&entries);
        let input = session.range(1).unwrap();

        let mut estimator = als::configured(&case["values"]);
        let fitted = estimator.fit(&input).unwrap();
        assert_commands(&server, &fit_entry, &key);
        check_members(&server, &golden["models"][&key]["members"], &|_| None, &key);

        // Finding: PySpark's ALSModel.load sends the read alone, and the Rust load also fetches
        // the rank of the loaded model. The replay refuses that fetch, so the loads of the ALS
        // models are left out here; their saves are replayed.
        fitted.write().unwrap().overwrite().save(PATH).unwrap();
        assert_commands(&server, &model_record["save"], &model_key);
        check_members(
            &server,
            &model_record["members"]["members"],
            &|_| None,
            &model_key,
        );
        seen.insert(model_key);

        Estimator::write(&estimator)
            .session(&session)
            .overwrite()
            .save(PATH)
            .unwrap();
        assert_commands(&server, &estimator_record["save"], &estimator_key);
        let loaded = ALS::load(&session, PATH).unwrap();
        assert_commands(&server, &estimator_record["load"], &estimator_key);
        assert_loaded(
            estimator_record,
            (Estimator::operator(&loaded), Estimator::params(&loaded)),
            (
                Estimator::operator(&estimator),
                Estimator::params(&estimator),
            ),
            &estimator_key,
        );
        seen.insert(estimator_key);
    }
    let official: BTreeSet<String> = golden["fits"]
        .as_object()
        .unwrap()
        .keys()
        .filter(|key| key.starts_with("ALS."))
        .flat_map(|key| [format!("{key}.model"), format!("{key}.estimator")])
        .collect();
    assert_eq!(seen, official, "the ALS records of the golden");
}

/// Every evaluator of the golden: its evaluate and isLargerBetter, and the save and load of the
/// evaluator, with the metric and the flag of the loaded copy.
#[test]
fn evaluators_match_official_pyspark() {
    let golden = fixture(MODELS);
    let records = fixture(PERSISTENCE)["persistence"].clone();
    let cases = evaluators::fixture();
    let mut seen = BTreeSet::new();
    for (key, record) in records.as_object().unwrap() {
        let Some(name) = key.strip_suffix(".evaluator") else {
            continue;
        };
        let case = cases["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == name)
            .unwrap_or_else(|| panic!("no evaluator case {name}"));
        let class = case["class"].as_str().unwrap();
        let official = &golden["models"][name];
        assert_eq!(official["class"], class, "{name}");

        let mut entries = checked_members(&official["members"]);
        entries.extend(checked_members(&record["members"]["members"]));
        entries.extend([&record["save"], &record["load"]]);
        let (server, session) = ReplayServer::start(&entries);
        let input = session.range(1).unwrap();

        let evaluator = evaluators::configured(class, &case["values"]);
        check_members(
            &server,
            &official["members"],
            &|label| evaluator_member(evaluator.as_ref(), label, &input),
            name,
        );

        evaluator
            .write()
            .session(&session)
            .overwrite()
            .save(PATH)
            .unwrap();
        assert_commands(&server, &record["save"], key);
        let loaded = load_evaluator(class, &session, PATH);
        assert_commands(&server, &record["load"], key);
        assert_loaded(
            record,
            (loaded.operator(), loaded.params()),
            (evaluator.operator(), evaluator.params()),
            key,
        );
        check_members(
            &server,
            &record["members"]["members"],
            &|label| evaluator_member(loaded.as_ref(), label, &input),
            key,
        );
        seen.insert(key.clone());
    }
    let official: BTreeSet<String> = records
        .as_object()
        .unwrap()
        .keys()
        .filter(|key| key.ends_with(".evaluator"))
        .cloned()
        .collect();
    assert_eq!(seen, official, "the evaluator records of the golden");
}

/// A request that differs from its golden request fails the check: the commands that were sent
/// are the golden load's, not the golden fit's.
#[test]
fn a_request_that_differs_from_its_golden_fails_the_check() {
    let golden = fixture(MODELS);
    let fit = without_rank_fetches(&golden["fits"]["ALS.explicit"]["fit"]);
    let (server, session) = ReplayServer::start(&[&fit]);
    let input = session.range(1).unwrap();
    let mut estimator = als::configured(&als::fixture()["cases"][0]["values"]);
    let _fitted = estimator.fit(&input).unwrap();
    assert_commands(&server, &fit, "fit");

    // The golden has no load of a model at this point: the replay refuses the read, which is the
    // request that was sent, so the commands that were sent are not the golden's.
    assert!(ALSModel::load(&session, PATH).is_err());
    let check = catch_unwind(AssertUnwindSafe(|| assert_commands(&server, &fit, "fit")));
    assert!(check.is_err(), "the differing request passed the check");
}
