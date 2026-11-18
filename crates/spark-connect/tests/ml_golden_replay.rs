// SPDX-License-Identifier: Apache-2.0
//! The offline replay of the ML goldens (tests/common/ml_golden.rs): classification golden
//! entries are served by a loopback replay server, a direct `ml::fetch` makes their requests, and
//! the requests are checked against the commands of the entries. No Spark server is needed.

#[path = "common/ml_golden.rs"]
mod ml_golden;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use base64::{engine::general_purpose::STANDARD, Engine};
use ml_golden::*;
use prost::Message;
use serde_json::{json, Value as Json};
use spark_connect::ml::{
    fetch, ClassificationModel, CountVectorizerModel, DecisionTreeClassifier, Estimator,
    GBTClassifier, Imputer, LogisticRegression, MlRef, MlTarget, Model, OneVsRest, Params,
    RandomForestClassifier, StringIndexerModel, SubModel, TreeEnsembleModel,
};
use spark_connect::{DataFrame, SparkSession};
use spark_connect_core::error::SparkError;
use spark_connect_proto as proto;
use spark_connect_proto::ml_command_result::ml_operator_info::Type;
use spark_connect_proto::ml_relation::MlType;

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
    // A relation entry (a DataFrame member, whose command is an MlRelation) is recognized by its
    // value, not by decoding: it must be a transform or a fetch relation, and its `transform`
    // member must be one. Every other entry's commands must be ML commands.
    fn walk(entry: &Json, key: &str, commands: &mut usize, relations: &mut usize) {
        let Some(object) = entry.as_object() else {
            return;
        };
        if object.contains_key("commands") {
            if is_relation_entry(entry) {
                for relation in golden_relations(entry) {
                    let kind = relation.ml_type.as_ref().expect("an operator");
                    assert!(
                        matches!(kind, MlType::Transform(_) | MlType::Fetch(_)),
                        "{key}: a relation that is neither a transform nor a fetch"
                    );
                    *relations += 1;
                }
                return;
            }
            assert!(
                key != "transform",
                "a transform entry that is not a relation"
            );
            for command in golden_commands(entry) {
                assert!(
                    command.command.is_some(),
                    "a golden command with no command"
                );
                *commands += 1;
            }
            return;
        }
        for (child_key, child) in object {
            walk(child, child_key, commands, relations);
        }
    }
    let mut commands = 0;
    let mut relations = 0;
    for text in [MODELS, SUMMARIES] {
        walk(&fixture(text), "", &mut commands, &mut relations);
    }
    assert!(commands > 0, "no golden commands were found");
    assert!(relations > 0, "no golden relations were found");
}

#[test]
fn a_classification_transform_is_a_relation_entry_not_a_command() {
    // The transform of a DecisionTreeClassifier model is an MlRelation: it decodes as an MlCommand
    // Fit too, so only its entry tells them apart.
    let models = fixture(MODELS);
    let transform = &models["models"]["DecisionTreeClassifier"]["members"]["transform"];
    assert!(is_relation_entry(transform));
    let checked = catch_unwind(AssertUnwindSafe(|| golden_commands(transform)));
    let message = panic_message(checked.expect_err("a relation entry is checked"));
    assert!(message.contains("MlRelation"), "{message}");
    let relations = golden_relations(transform);
    assert!(matches!(relations[0].ml_type, Some(MlType::Transform(_))));
}

#[test]
fn a_golden_with_a_random_uid_is_refused() {
    // A golden holds the placeholders of its uids. One with a random uid would match any uid of a
    // run, so it is refused rather than renumbered.
    let command = proto::MlCommand {
        command: Some(proto::ml_command::Command::Fit(proto::ml_command::Fit {
            estimator: Some(proto::MlOperator {
                uid: "LinearSVC_0123456789ab".into(),
                ..Default::default()
            }),
            ..Default::default()
        })),
    };
    let entry = serde_json::json!({
        "commands": [STANDARD.encode(command.encode_to_vec())],
    });
    let refused = catch_unwind(AssertUnwindSafe(|| golden_commands(&entry)));
    let message = panic_message(refused.expect_err("a random uid in a golden is refused"));
    assert!(message.contains("random uid"), "{message}");
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
fn a_dropped_model_is_deleted_by_an_execution_of_the_dropping_thread_as_pyspark() {
    // As PySpark's `RemoteModelRef.release_ref`, whose `_delete_ml_cache` runs one
    // `execute_command` on the thread that drops the last reference to the model, the thread of
    // its finalizer: the request has the tags and the user context extensions of that thread, goes
    // to the hook of the executions, and is reattachable, as any execution of `execute_command`.
    let features = fixture(FEATURE_MODELS);
    let fit = &features["fits"]["Imputer"]["fit"];
    let (server, session) = ReplayServer::start(&[fit]);
    let df = session.range(1).unwrap();
    let model = Imputer::new()
        .set_input_col("missing")
        .set_output_col("out")
        .set_missing_value(f64::NAN)
        .fit(&df)
        .unwrap();
    assert_commands(&server, fit, "Imputer fit");
    session.add_tag("dropper").unwrap();
    let extension = prost_types::Any {
        type_url: "type.googleapis.com/test.dropper".into(),
        value: b"dropper".to_vec(),
    };
    session.add_threadlocal_user_context_extension(extension.clone());
    let hooked = Arc::new(Mutex::new(0));
    let calls = Arc::clone(&hooked);
    let hook: spark_connect::ExecutePlanHook = Arc::new(move |request| {
        *calls.lock().unwrap() += 1;
        Ok(request)
    });
    session.set_execute_plan_hook(Some(hook));

    drop(model);
    let started = std::time::Instant::now();
    let mut requests = server.take_delete_requests();
    while requests.is_empty() && started.elapsed() < std::time::Duration::from_secs(10) {
        std::thread::sleep(std::time::Duration::from_millis(10));
        requests = server.take_delete_requests();
    }
    let [request] = requests.as_slice() else {
        panic!("the requests of the deletes of the dropped model: {requests:?}");
    };
    assert_eq!(request.session_id, session.session_id());
    assert_eq!(request.tags, ["dropper"]);
    let context = request.user_context.as_ref().expect("a user context");
    assert_eq!(context.extensions, [extension]);
    assert!(
        request.request_options.iter().any(|option| matches!(
            option.request_option,
            Some(
                proto::execute_plan_request::request_option::RequestOption::ReattachOptions(
                    proto::ReattachOptions { reattachable: true }
                )
            )
        )),
        "the delete is not reattachable: {request:?}"
    );
    assert_eq!(*hooked.lock().unwrap(), 1, "the delete went to no hook");
}

#[test]
fn a_dropped_model_sends_a_delete_that_the_replay_records() {
    // The client releases a fitted model in the background when the model is dropped. The delete
    // is no golden command: the replay records it apart, with the id of the model's golden
    // result, and it is not among the commands that the checks take.
    let features = fixture(FEATURE_MODELS);
    let fit = &features["fits"]["Imputer"]["fit"];
    let result = proto::MlCommandResult::decode(
        STANDARD
            .decode(fit["results"][0].as_str().unwrap())
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    let Some(proto::ml_command_result::ResultType::OperatorInfo(info)) = result.result_type else {
        panic!("the golden fit result is not an operator info");
    };
    let Some(Type::ObjRef(reference)) = info.r#type else {
        panic!("the golden fit result is not a reference");
    };
    let (server, session) = ReplayServer::start(&[fit]);
    let df = session.range(1).unwrap();
    let model = Imputer::new()
        .set_input_col("missing")
        .set_output_col("out")
        .set_missing_value(f64::NAN)
        .fit(&df)
        .unwrap();
    assert_commands(&server, fit, "Imputer fit");
    assert!(server.take_deletes().is_empty(), "a delete before the drop");

    drop(model);
    let started = std::time::Instant::now();
    let mut deletes = server.take_deletes();
    while deletes.is_empty() && started.elapsed() < std::time::Duration::from_secs(10) {
        std::thread::sleep(std::time::Duration::from_millis(10));
        deletes = server.take_deletes();
    }
    assert_eq!(deletes.len(), 1, "the deletes of the dropped model");
    let ids: Vec<&str> = deletes[0].obj_refs.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, [reference.id.as_str()]);
    // PySpark's `_delete_ml_cache` sets `evict_only`, to False for a released model.
    assert_eq!(deletes[0].evict_only, Some(false));
    assert!(
        server.pending_requests().is_empty(),
        "a delete is a command"
    );
}

/// The params that the golden fit of a family sent, as its estimator is built with them.
fn golden_fit_params(fit: &Json) -> proto::MlParams {
    let command = proto::MlCommand::decode(
        STANDARD
            .decode(fit["commands"][0].as_str().unwrap())
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    let Some(proto::ml_command::Command::Fit(sent)) = command.command else {
        panic!("the golden fit is not a Fit");
    };
    sent.params.expect("the golden fit's params")
}

/// The fit entry of a tree family, replayed: an estimator with the params of the golden fit,
/// fitted on a dataset and then read as PySpark's fit reads its model, sends the golden fit's
/// commands. The golden's dataset is not compared (see `normalize_commands`).
fn check_tree_fit<E: Estimator>(
    models: &Json,
    family: &str,
    mut estimator: E,
    fit_and_read: impl FnOnce(&mut E, &DataFrame),
) {
    let fit = &models["fits"][family]["fit"];
    let dataset = models["fits"][family]["dataset"].as_str().unwrap();
    let (server, session) = ReplayServer::start(&[fit]);
    *estimator.params_mut() = Params::from_proto(&golden_fit_params(fit));
    let train = golden_dataset(&session, &models["datasets"], dataset);
    fit_and_read(&mut estimator, &train);
    assert_fit_dataset(&server, &models["datasets"], dataset);
    assert_commands(&server, fit, family);
}

#[test]
fn tree_classifier_fits_send_the_golden_fit_commands() {
    let models = fixture(MODELS);
    // A decision tree's fit sends the Fit alone.
    check_tree_fit(
        &models,
        "DecisionTreeClassifier",
        DecisionTreeClassifier::new(),
        |estimator, train| {
            estimator.fit(train).unwrap();
        },
    );
    // A boosted ensemble's fit is read for its trees, and a forest's also for its classes.
    check_tree_fit(
        &models,
        "GBTClassifier",
        GBTClassifier::new(),
        |estimator, train| {
            estimator.fit(train).unwrap().trees().unwrap();
        },
    );
    for family in [
        "RandomForestClassifier.binary",
        "RandomForestClassifier.multiclass",
    ] {
        check_tree_fit(
            &models,
            family,
            RandomForestClassifier::new(),
            |estimator, train| {
                let model = estimator.fit(train).unwrap();
                model.num_classes().unwrap();
                model.trees().unwrap();
            },
        );
    }
}

/// The golden entry `entry` whose result `index`, of a Fit, has the warning `warning`, as the
/// server sends the warning of a tree training that the memory control of its ML cache stopped
/// early (`MlOperatorInfo.warning_message`).
fn with_fit_warning(entry: &Json, index: usize, warning: &str) -> Json {
    let mut entry = entry.clone();
    let encoded = STANDARD
        .decode(entry["results"][index].as_str().unwrap())
        .unwrap();
    let mut result = proto::MlCommandResult::decode(encoded.as_slice()).unwrap();
    let Some(proto::ml_command_result::ResultType::OperatorInfo(info)) =
        result.result_type.as_mut()
    else {
        panic!("the golden result {index} is not of a fit");
    };
    info.warning_message = Some(warning.to_string());
    entry["results"][index] = json!(STANDARD.encode(result.encode_to_vec()));
    entry
}

/// The warnings that the fits of `session` hand over (`set_ml_fit_warning_handler`), each with
/// the ML commands that `server` had received then and the thread that handed it over.
type Handed = Arc<Mutex<Vec<(String, usize, std::thread::ThreadId)>>>;

fn hand_over_warnings(session: &SparkSession, server: &Arc<ReplayServer>) -> Handed {
    let handed = Handed::default();
    let (server, sink) = (server.clone(), handed.clone());
    session.set_ml_fit_warning_handler(move |warning| {
        let received = server.pending_requests().len();
        let thread = std::thread::current().id();
        sink.lock()
            .unwrap()
            .push((warning.to_string(), received, thread));
        Ok(())
    });
    handed
}

#[test]
fn a_fit_hands_its_warning_over_before_its_model_is_read_and_the_model_keeps_it() {
    // As PySpark's try_remote_fit logs the warning that the server sends with the model of a fit
    // (of a tree training that the memory control of its ML cache stopped early), once the result
    // of the Fit has arrived and before it makes the model: the session's handler gets it on the
    // thread of the fit, when the server has received the Fit alone, before the classes and the
    // trees of the forest are read. The model keeps it, and the trees that a fetch returns have
    // none, as PySpark reads it of a fit only.
    let models = fixture(MODELS);
    let family = "RandomForestClassifier.binary";
    let warning = "The random forest training stops early because the model size (35016 bytes) \
                   exceeds threshold (24000 bytes).";
    let fit = with_fit_warning(&models["fits"][family]["fit"], 0, warning);
    let (server, session) = ReplayServer::start(&[&fit]);
    let server = Arc::new(server);
    let handed = hand_over_warnings(&session, &server);
    let mut estimator = RandomForestClassifier::new();
    *estimator.params_mut() = Params::from_proto(&golden_fit_params(&fit));
    let dataset = models["fits"][family]["dataset"].as_str().unwrap();
    let train = golden_dataset(&session, &models["datasets"], dataset);
    let model = estimator.fit(&train).unwrap();
    model.num_classes().unwrap();
    let trees = model.trees().unwrap();
    assert_commands(&server, &fit, family);
    let this = std::thread::current().id();
    assert_eq!(*handed.lock().unwrap(), [(warning.to_string(), 1, this)]);
    assert_eq!(model.fit_warning(), Some(warning));
    assert_eq!(model.ml_ref().and_then(MlRef::fit_warning), Some(warning));
    assert!(!trees.is_empty() && trees.iter().all(|tree| tree.fit_warning().is_none()));
}

#[test]
fn a_fit_of_no_warning_or_of_an_empty_one_hands_none_over() {
    // As PySpark's `if warning_msg := getattr(model_info, "warning_message", None)`: a fit whose
    // result has no warning, or an empty one, logs nothing.
    let models = fixture(MODELS);
    let golden = &models["fits"]["DecisionTreeClassifier"]["fit"];
    for fit in [golden.clone(), with_fit_warning(golden, 0, "")] {
        let (server, session) = ReplayServer::start(&[&fit]);
        let server = Arc::new(server);
        let handed = hand_over_warnings(&session, &server);
        let mut estimator = DecisionTreeClassifier::new();
        *estimator.params_mut() = Params::from_proto(&golden_fit_params(&fit));
        let dataset = models["fits"]["DecisionTreeClassifier"]["dataset"]
            .as_str()
            .unwrap();
        let train = golden_dataset(&session, &models["datasets"], dataset);
        let model = estimator.fit(&train).unwrap();
        assert_commands(&server, &fit, "DecisionTreeClassifier");
        assert!(handed.lock().unwrap().is_empty());
        assert_eq!(model.fit_warning(), None);
    }
}

#[test]
fn the_error_of_the_handler_of_a_fit_warning_is_the_error_of_the_fit() {
    // As an exception of the logging of PySpark's try_remote_fit (of a filter that raises) is the
    // exception of its fit, which makes no model.
    let models = fixture(MODELS);
    let fit = with_fit_warning(
        &models["fits"]["DecisionTreeClassifier"]["fit"],
        0,
        "stopped early",
    );
    let (server, session) = ReplayServer::start(&[&fit]);
    session.set_ml_fit_warning_handler(|warning| {
        Err(SparkError::value_msg(format!(
            "the handler of '{warning}' failed"
        )))
    });
    let mut estimator = DecisionTreeClassifier::new();
    *estimator.params_mut() = Params::from_proto(&golden_fit_params(&fit));
    let dataset = models["fits"]["DecisionTreeClassifier"]["dataset"]
        .as_str()
        .unwrap();
    let train = golden_dataset(&session, &models["datasets"], dataset);
    let error = estimator.fit(&train).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("the handler of 'stopped early' failed"),
        "{error}"
    );
    assert_commands(&server, &fit, "DecisionTreeClassifier");
}

#[test]
fn one_vs_rest_class_fit_sends_the_golden_fit_command() {
    // A OneVsRest fit fits its classifier once per class: the golden's first command is the
    // first such Fit, with the classifier's params, and then its class count is read. The
    // golden's commands do not record the fit's own DataFrame query of the largest label, which
    // PySpark runs as a DataFrame aggregate too, and the replay does not answer DataFrame plans:
    // so the replay checks the class fit, the golden's first command, and no more.
    let models = fixture(MODELS);
    let fit = &models["fits"]["OneVsRest"]["fit"];
    let class_fit = json!({
        "commands": [fit["commands"][0]],
        "results": [fit["results"][0]],
    });
    let (server, session) = ReplayServer::start(&[&class_fit]);
    let mut classifier = LogisticRegression::new();
    *classifier.params_mut() = Params::from_proto(&golden_fit_params(&class_fit));
    // The class fit trains on the golden rows with a binary label of its own (a transform of the
    // dataset), so the dataset itself is not checked here.
    let train = golden_dataset(&session, &models["datasets"], "multi_train");
    classifier.fit(&train).unwrap();
    assert_commands(&server, &class_fit, "OneVsRest class fit");
}

#[test]
fn one_vs_rest_class_fits_hand_their_warnings_over_on_their_threads() {
    // The fits of the classes of PySpark's OneVsRest run `classifier.fit` in the threads of its
    // pool, whose try_remote_fit logs the warning of each: the fit of each class hands its own
    // over on the thread that fits the class, once its Fit has returned and before the next class
    // fits, and the model of each class keeps its own. The golden's class Fits are its commands 0,
    // 2 and 4; the Rust fits of the classes do not read the class counts after them, which
    // PySpark's training summaries read, so the commands are not checked against the golden.
    let models = fixture(MODELS);
    let mut fit = models["fits"]["OneVsRest"]["fit"].clone();
    for class in 0..3 {
        fit = with_fit_warning(&fit, 2 * class, &format!("the warning of class {class}"));
    }
    let (server, session) = ReplayServer::start(&[&fit]);
    let server = Arc::new(server);
    let handed = hand_over_warnings(&session, &server);
    let classifier = LogisticRegression::new()
        .set_reg_param(0.01)
        .set_max_iter(20);
    let estimator = OneVsRest::new().set_classifier(classifier);
    let train = golden_dataset(&session, &models["datasets"], "multi_train");
    let (model, _) = estimator.fit_classes(&train, 3, 1, Some("weight")).unwrap();
    let handed = handed.lock().unwrap().clone();
    let this = std::thread::current().id();
    assert!(handed.iter().all(|(_, _, thread)| *thread != this));
    let handed: Vec<(String, usize)> = handed
        .into_iter()
        .map(|(warning, received, _)| (warning, received))
        .collect();
    let expected: Vec<(String, usize)> = (0..3)
        .map(|class| (format!("the warning of class {class}"), class + 1))
        .collect();
    assert_eq!(handed, expected);
    let kept: Vec<Option<String>> = model
        .models()
        .into_iter()
        .map(|model| match model {
            SubModel::LogisticRegression(model) => model.fit_warning().map(str::to_string),
            _ => panic!("a class model of no LogisticRegressionModel"),
        })
        .collect();
    let expected: Vec<Option<String>> = (0..3)
        .map(|class| Some(format!("the warning of class {class}")))
        .collect();
    assert_eq!(kept, expected);
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
