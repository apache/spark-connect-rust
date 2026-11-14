// SPDX-License-Identifier: Apache-2.0
//! The tree classification accessors, offline, against the golden of PySpark 4.3.0. A session of
//! an unreachable server captures each request through its execute hook, which fails it, so
//! each accessor's request is compared with the golden's command. The golden's results decode,
//! with the accessor's own conversions, to the golden's values. Live checks are in
//! `tests/e2e_tree_classification_api.rs`.

use std::sync::{Arc, Mutex};

use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use serde_json::{json, Value};
use spark_connect_core::channel::ChannelBuilder;
use spark_connect_core::client::SparkConnectClient;
use spark_connect_core::error::{Result, SparkError};
use spark_connect_core::runtime::block_on;
use spark_connect_proto as proto;

use crate::ml::tree_api::{as_f64, as_f64s, as_i32, as_string, as_vector};
use crate::ml::{
    BinaryClassificationSummary, BinaryRandomForestClassificationSummary,
    BinaryRandomForestClassificationTrainingSummary, ClassificationModel, ClassificationSummary,
    DecisionTreeClassificationModel, DecisionTreeClassifier, DecisionTreeModel,
    GBTClassificationModel, GBTClassifier, MlRef, MlTarget, MlValue, MlVector, Params,
    PredictionModel, ProbabilisticClassificationModel, RandomForestClassificationModel,
    RandomForestClassificationSummary, RandomForestClassificationTrainingSummary,
    RandomForestClassifier, TrainingSummary, TreeEnsembleModel,
};
use crate::{ExecutePlanHook, SparkSession};

const CLASSIFICATION_MODELS: &str =
    include_str!("../../../../../../tests/golden/classification_models.json");
const CLASSIFICATION_SUMMARIES: &str =
    include_str!("../../../../../../tests/golden/classification_summaries.json");
const CLASSIFICATION_PERSISTENCE: &str =
    include_str!("../../../../../../tests/golden/classification_persistence.json");

type Requests = Arc<Mutex<Vec<proto::ExecutePlanRequest>>>;

fn golden(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

/// A session of an unreachable server, without retries, so that a captured request fails at once.
fn offline_session() -> SparkSession {
    let builder = ChannelBuilder::parse("sc://127.0.0.1:1").expect("a valid URL");
    let mut client = block_on(SparkConnectClient::connect(&builder)).expect("a client");
    client.set_retry_policy(spark_connect_core::retries::RetryPolicy::no_retries());
    SparkSession::new(client)
}

/// Record each request of `session` and fail it, as the hook sees the request before it is sent.
fn capture(session: &SparkSession) -> Requests {
    let requests: Requests = Arc::default();
    let seen = Arc::clone(&requests);
    let hook: ExecutePlanHook = Arc::new(move |request| {
        seen.lock().unwrap().push(request);
        Err(SparkError::value_msg("captured by the test"))
    });
    session.set_execute_plan_hook(Some(hook));
    requests
}

fn ml_command(request: &proto::ExecutePlanRequest) -> proto::MlCommand {
    let plan = request.plan.as_ref().expect("a plan");
    let Some(proto::plan::OpType::Command(command)) = &plan.op_type else {
        panic!("a command request, got {plan:?}");
    };
    let Some(proto::command::CommandType::MlCommand(ml)) = &command.command_type else {
        panic!("an ML command, got {command:?}");
    };
    ml.clone()
}

fn golden_command(b64: &str) -> proto::MlCommand {
    proto::MlCommand::decode(STANDARD.decode(b64).unwrap().as_slice()).unwrap()
}

/// The value of a golden result, which is the literal of a `Param` result.
fn golden_value(b64: &str) -> MlValue {
    let result = proto::MlCommandResult::decode(STANDARD.decode(b64).unwrap().as_slice()).unwrap();
    match result.result_type {
        Some(proto::ml_command_result::ResultType::Param(literal)) => {
            MlValue::from_literal(&literal).unwrap()
        }
        other => panic!("not a param result: {other:?}"),
    }
}

fn assert_request(requests: &Requests, golden_entry: &Value) {
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1, "one request for {golden_entry}");
    let expected = golden_command(golden_entry["commands"][0].as_str().unwrap());
    assert_eq!(ml_command(&requests[0]), expected);
}

fn assert_value(golden_entry: &Value, decode: impl FnOnce(MlValue) -> Result<Value>, what: &str) {
    let decoded = decode(golden_value(golden_entry["results"][0].as_str().unwrap())).unwrap();
    assert_json_close(&decoded, &golden_entry["value"], what);
}

/// Check one accessor: its request is the golden's command, and its result decodes to the
/// golden's value. The accessor's own error is the capture's.
fn check(
    requests: &Requests,
    golden_entry: &Value,
    call: impl FnOnce(),
    decode: impl FnOnce(MlValue) -> Result<Value>,
    what: &str,
) {
    requests.lock().unwrap().clear();
    call();
    assert_request(requests, golden_entry);
    assert_value(golden_entry, decode, what);
}

fn assert_json_close(actual: &Value, expected: &Value, what: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!(
                (a - e).abs() <= 1e-9 * e.abs().max(1.0),
                "{what}: {a} is not {e}"
            );
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "{what}: length");
            for (i, (a, e)) in a.iter().zip(e).enumerate() {
                assert_json_close(a, e, &format!("{what}[{i}]"));
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            assert_eq!(a.len(), e.len(), "{what}: keys");
            for (key, e) in e {
                assert_json_close(&a[key], e, &format!("{what}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{what}"),
    }
}

fn vector_json(vector: &MlVector) -> Value {
    match vector {
        MlVector::Dense(values) => json!({"size": values.len(), "type": "dense", "values": values}),
        MlVector::Sparse {
            size,
            indices,
            values,
        } => json!({"indices": indices, "size": size, "type": "sparse", "values": values}),
    }
}

fn int(v: MlValue) -> Result<Value> {
    Ok(json!(v.as_i64().unwrap()))
}

fn probes() -> Vec<(&'static str, MlVector)> {
    vec![
        ("dense", MlVector::Dense(vec![0.5, 1.5])),
        ("dense2", MlVector::Dense(vec![3.0, -0.25])),
        (
            "sparse",
            MlVector::Sparse {
                size: 2,
                indices: vec![1],
                values: vec![2.0],
            },
        ),
    ]
}

fn member<'a>(golden: &'a Value, section: &str, key: &str, name: &str) -> &'a Value {
    let entry = &golden[section][key]["members"][name];
    assert!(!entry.is_null(), "no golden member {section}.{key}.{name}");
    entry
}

#[test]
fn decision_tree_model_accessors_send_the_golden_requests() {
    let models = golden(CLASSIFICATION_MODELS);
    let session = offline_session();
    let requests = capture(&session);
    let model = DecisionTreeClassificationModel::from_ml_ref(
        MlRef::from_id(&session, "<id0>"),
        Params::new(),
    );
    let key = "DecisionTreeClassifier";
    let g = |name: &str| member(&models, "models", key, name);
    let s = |name: &str| format!("{key}.{name}");

    check(
        &requests,
        g("depth"),
        || {
            let _ = model.depth();
        },
        |v| Ok(json!(as_i32("depth", v)?)),
        &s("depth"),
    );
    check(
        &requests,
        g("numNodes"),
        || {
            let _ = model.num_nodes();
        },
        |v| Ok(json!(as_i32("numNodes", v)?)),
        &s("numNodes"),
    );
    check(
        &requests,
        g("numClasses"),
        || {
            let _ = model.num_classes();
        },
        |v| Ok(json!(as_i32("numClasses", v)?)),
        &s("numClasses"),
    );
    check(
        &requests,
        g("numFeatures"),
        || {
            let _ = model.num_features();
        },
        int,
        &s("numFeatures"),
    );
    check(
        &requests,
        g("featureImportances"),
        || {
            let _ = model.feature_importances();
        },
        |v| Ok(vector_json(&as_vector("featureImportances", v)?)),
        &s("featureImportances"),
    );
    check(
        &requests,
        g("toDebugString"),
        || {
            let _ = model.to_debug_string();
        },
        |v| Ok(json!(as_string("toDebugString", v)?)),
        &s("toDebugString"),
    );
    for (name, features) in probes() {
        check(
            &requests,
            g(&format!("predict-{name}")),
            || {
                let _ = model.predict(&features);
            },
            |v| Ok(json!(as_f64("predict", v)?)),
            &s(&format!("predict-{name}")),
        );
        check(
            &requests,
            g(&format!("predictLeaf-{name}")),
            || {
                let _ = model.predict_leaf(&features);
            },
            |v| Ok(json!(as_f64("predictLeaf", v)?)),
            &s(&format!("predictLeaf-{name}")),
        );
        check(
            &requests,
            g(&format!("predictRaw-{name}")),
            || {
                let _ = model.predict_raw(&features);
            },
            |v| Ok(vector_json(&as_vector("predictRaw", v)?)),
            &s(&format!("predictRaw-{name}")),
        );
        check(
            &requests,
            g(&format!("predictProbability-{name}")),
            || {
                let _ = model.predict_probability(&features);
            },
            |v| Ok(vector_json(&as_vector("predictProbability", v)?)),
            &s(&format!("predictProbability-{name}")),
        );
    }
    assert_eq!(
        model.supported_impurities(),
        ["entropy", "gini"],
        "supportedImpurities"
    );
}

#[test]
fn random_forest_model_accessors_send_the_golden_requests() {
    let models = golden(CLASSIFICATION_MODELS);
    let session = offline_session();
    let requests = capture(&session);
    let model = RandomForestClassificationModel::from_ml_ref(
        MlRef::from_id(&session, "<id0>"),
        Params::new(),
    );
    let key = "RandomForestClassifier.binary";
    let g = |name: &str| member(&models, "models", key, name);
    let s = |name: &str| format!("{key}.{name}");

    check(
        &requests,
        g("numClasses"),
        || {
            let _ = model.num_classes();
        },
        |v| Ok(json!(as_i32("numClasses", v)?)),
        &s("numClasses"),
    );
    check(
        &requests,
        g("getNumTrees"),
        || {
            let _ = model.get_num_trees();
        },
        |v| Ok(json!(as_i32("getNumTrees", v)?)),
        &s("getNumTrees"),
    );
    check(
        &requests,
        g("totalNumNodes"),
        || {
            let _ = model.total_num_nodes();
        },
        |v| Ok(json!(as_i32("totalNumNodes", v)?)),
        &s("totalNumNodes"),
    );
    check(
        &requests,
        g("treeWeights"),
        || {
            let _ = model.tree_weights();
        },
        |v| Ok(json!(as_f64s("treeWeights", v)?)),
        &s("treeWeights"),
    );
    check(
        &requests,
        g("featureImportances"),
        || {
            let _ = model.feature_importances();
        },
        |v| Ok(vector_json(&as_vector("featureImportances", v)?)),
        &s("featureImportances"),
    );
    for (name, features) in probes() {
        check(
            &requests,
            g(&format!("predict-{name}")),
            || {
                let _ = model.predict(&features);
            },
            |v| Ok(json!(as_f64("predict", v)?)),
            &s(&format!("predict-{name}")),
        );
        check(
            &requests,
            g(&format!("predictLeaf-{name}")),
            || {
                let _ = model.predict_leaf(&features);
            },
            |v| Ok(vector_json(&as_vector("predictLeaf", v)?)),
            &s(&format!("predictLeaf-{name}")),
        );
        check(
            &requests,
            g(&format!("predictProbability-{name}")),
            || {
                let _ = model.predict_probability(&features);
            },
            |v| Ok(vector_json(&as_vector("predictProbability", v)?)),
            &s(&format!("predictProbability-{name}")),
        );
    }
    assert!(
        !model.has_summary(),
        "a model from a reference has no summary"
    );
    assert_eq!(
        model.supported_feature_subset_strategies(),
        ["auto", "all", "onethird", "sqrt", "log2"]
    );
}

#[test]
fn gbt_model_accessors_send_the_golden_requests() {
    let models = golden(CLASSIFICATION_MODELS);
    let session = offline_session();
    let requests = capture(&session);
    let model =
        GBTClassificationModel::from_ml_ref(MlRef::from_id(&session, "<id0>"), Params::new());
    let key = "GBTClassifier";
    let g = |name: &str| member(&models, "models", key, name);
    let s = |name: &str| format!("{key}.{name}");

    check(
        &requests,
        g("numClasses"),
        || {
            let _ = model.num_classes();
        },
        |v| Ok(json!(as_i32("numClasses", v)?)),
        &s("numClasses"),
    );
    check(
        &requests,
        g("numFeatures"),
        || {
            let _ = model.num_features();
        },
        int,
        &s("numFeatures"),
    );
    check(
        &requests,
        g("predict-dense"),
        || {
            let _ = model.predict(&MlVector::Dense(vec![0.5, 1.5]));
        },
        |v| Ok(json!(as_f64("predict", v)?)),
        &s("predict-dense"),
    );
    check(
        &requests,
        g("predictProbability-dense"),
        || {
            let _ = model.predict_probability(&MlVector::Dense(vec![0.5, 1.5]));
        },
        |v| Ok(vector_json(&as_vector("predictProbability", v)?)),
        &s("predictProbability-dense"),
    );
    check(
        &requests,
        g("predictRaw-dense"),
        || {
            let _ = model.predict_raw(&MlVector::Dense(vec![0.5, 1.5]));
        },
        |v| Ok(vector_json(&as_vector("predictRaw", v)?)),
        &s("predictRaw-dense"),
    );
    check(
        &requests,
        g("getNumTrees"),
        || {
            let _ = model.get_num_trees();
        },
        |v| Ok(json!(as_i32("getNumTrees", v)?)),
        &s("getNumTrees"),
    );
    check(
        &requests,
        g("totalNumNodes"),
        || {
            let _ = model.total_num_nodes();
        },
        |v| Ok(json!(as_i32("totalNumNodes", v)?)),
        &s("totalNumNodes"),
    );
    check(
        &requests,
        g("treeWeights"),
        || {
            let _ = model.tree_weights();
        },
        |v| Ok(json!(as_f64s("treeWeights", v)?)),
        &s("treeWeights"),
    );
    assert_eq!(model.supported_loss_types(), ["logistic"]);
    assert_eq!(model.supported_impurities(), ["variance"]);
}

#[test]
fn random_forest_summary_accessors_send_the_golden_requests() {
    let summaries = golden(CLASSIFICATION_SUMMARIES);
    let session = offline_session();
    let requests = capture(&session);
    let model = MlRef::from_id(&session, "<id0>");

    // The training summary: the summary of the model's `<model>.summary` object.
    let training = BinaryRandomForestClassificationTrainingSummary::new(
        MlTarget::object(&model).child("summary"),
    );
    let key = "RandomForestClassifier.binary.training";
    let g = |name: &str| member(&summaries, "summaries", key, name);
    let s = |name: &str| format!("{key}.{name}");
    check(
        &requests,
        g("accuracy"),
        || {
            let _ = training.accuracy();
        },
        |v| Ok(json!(as_f64("accuracy", v)?)),
        &s("accuracy"),
    );
    check(
        &requests,
        g("labels"),
        || {
            let _ = training.labels();
        },
        |v| Ok(json!(as_f64s("labels", v)?)),
        &s("labels"),
    );
    check(
        &requests,
        g("fMeasureByLabel-0.5"),
        || {
            let _ = training.f_measure_by_label(0.5);
        },
        |v| Ok(json!(as_f64s("fMeasureByLabel", v)?)),
        &s("fMeasureByLabel-0.5"),
    );
    check(
        &requests,
        g("weightedFMeasure-1.0"),
        || {
            let _ = training.weighted_f_measure(1.0);
        },
        |v| Ok(json!(as_f64("weightedFMeasure", v)?)),
        &s("weightedFMeasure-1.0"),
    );
    check(
        &requests,
        g("areaUnderROC"),
        || {
            let _ = training.area_under_roc();
        },
        |v| Ok(json!(as_f64("areaUnderROC", v)?)),
        &s("areaUnderROC"),
    );
    check(
        &requests,
        g("scoreCol"),
        || {
            let _ = training.score_col();
        },
        |v| Ok(json!(as_string("scoreCol", v)?)),
        &s("scoreCol"),
    );
    check(
        &requests,
        g("objectiveHistory"),
        || {
            let _ = training.objective_history();
        },
        |v| Ok(json!(as_f64s("objectiveHistory", v)?)),
        &s("objectiveHistory"),
    );
    check(
        &requests,
        g("totalIterations"),
        || {
            let _ = training.total_iterations();
        },
        |v| Ok(json!(as_i32("totalIterations", v)?)),
        &s("totalIterations"),
    );

    // An evaluation: the summary of the id that `evaluate` returns, which is unowned (the golden
    // numbers the summary's own object as <id0> in its entry).
    let evaluated =
        BinaryRandomForestClassificationSummary::new(MlTarget::unowned(&session, "<id0>"));
    let key = "RandomForestClassifier.binary.evaluate";
    let g = |name: &str| member(&summaries, "summaries", key, name);
    let s = |name: &str| format!("{key}.{name}");
    check(
        &requests,
        g("accuracy"),
        || {
            let _ = evaluated.accuracy();
        },
        |v| Ok(json!(as_f64("accuracy", v)?)),
        &s("accuracy"),
    );
    check(
        &requests,
        g("recallByLabel"),
        || {
            let _ = evaluated.recall_by_label();
        },
        |v| Ok(json!(as_f64s("recallByLabel", v)?)),
        &s("recallByLabel"),
    );
    check(
        &requests,
        g("weightedFalsePositiveRate"),
        || {
            let _ = evaluated.weighted_false_positive_rate();
        },
        |v| Ok(json!(as_f64("weightedFalsePositiveRate", v)?)),
        &s("weightedFalsePositiveRate"),
    );

    // The multiclass summaries carry no binary members.
    let multi =
        RandomForestClassificationTrainingSummary::new(MlTarget::object(&model).child("summary"));
    let key = "RandomForestClassifier.multiclass.training";
    let g = |name: &str| member(&summaries, "summaries", key, name);
    let s = |name: &str| format!("{key}.{name}");
    check(
        &requests,
        g("precisionByLabel"),
        || {
            let _ = multi.precision_by_label();
        },
        |v| Ok(json!(as_f64s("precisionByLabel", v)?)),
        &s("precisionByLabel"),
    );
    let multi_evaluated =
        RandomForestClassificationSummary::new(MlTarget::unowned(&session, "<id0>"));
    let key = "RandomForestClassifier.multiclass.evaluate";
    let g = |name: &str| member(&summaries, "summaries", key, name);
    let s = |name: &str| format!("{key}.{name}");
    check(
        &requests,
        g("truePositiveRateByLabel"),
        || {
            let _ = multi_evaluated.true_positive_rate_by_label();
        },
        |v| Ok(json!(as_f64s("truePositiveRateByLabel", v)?)),
        &s("truePositiveRateByLabel"),
    );
}

#[test]
fn tree_loads_send_the_golden_reads() {
    let persistence = golden(CLASSIFICATION_PERSISTENCE);
    let session = offline_session();
    let requests = capture(&session);
    let path = "<path>";
    for key in [
        "DecisionTreeClassifier.estimator",
        "DecisionTreeClassifier.model",
        "GBTClassifier.estimator",
        "GBTClassifier.model",
        "RandomForestClassifier.binary.estimator",
        "RandomForestClassifier.binary.model",
    ] {
        requests.lock().unwrap().clear();
        let _ = match key {
            "DecisionTreeClassifier.estimator" => DecisionTreeClassifier::load(&session, path)
                .map(|_| ())
                .map_err(|e| e.to_string()),
            "GBTClassifier.estimator" => GBTClassifier::load(&session, path)
                .map(|_| ())
                .map_err(|e| e.to_string()),
            "RandomForestClassifier.binary.estimator" => {
                RandomForestClassifier::load(&session, path)
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            }
            "DecisionTreeClassifier.model" => DecisionTreeClassificationModel::load(&session, path)
                .map(|_| ())
                .map_err(|e| e.to_string()),
            "GBTClassifier.model" => GBTClassificationModel::load(&session, path)
                .map(|_| ())
                .map_err(|e| e.to_string()),
            _ => RandomForestClassificationModel::load(&session, path)
                .map(|_| ())
                .map_err(|e| e.to_string()),
        };
        let entry = &persistence["persistence"][key]["load"];
        assert!(!entry.is_null(), "no golden load of {key}");
        assert_request(&requests, entry);
    }
}
