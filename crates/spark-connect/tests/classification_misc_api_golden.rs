// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, summaries and persistence of NaiveBayesModel,
//! MultilayerPerceptronClassificationModel and FMClassificationModel, replayed against the golden
//! of official PySpark 4.3.0 with no server.
//!
//! A loopback gRPC service answers each ML command with the golden result of the same command. A
//! command is matched by its bytes, normalized as the golden is (object ids are placeholders, and
//! the input relation of an `evaluate` is cleared), so every request that a call sends must be one
//! of the golden's commands. Each value that the call returns is compared with the golden's value.

use std::collections::BTreeSet;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use base64::{engine::general_purpose::STANDARD, Engine};
use futures::stream;
use prost::Message;
use serde_json::{json, Value};
use spark_connect::ml::{
    BinaryClassificationSummary, ClassificationSummary, FMClassificationModel, FMClassifier,
    MlMatrix, MlSummary, MlTarget, MlVector, Model, MultilayerPerceptronClassificationModel,
    MultilayerPerceptronClassifier, NaiveBayes, NaiveBayesModel, PredictionModel,
    ProbabilisticClassificationModel, TrainingSummary,
};
use spark_connect::SparkSession;
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_core::error::Result;
use spark_connect_proto as proto;
use tokio::net::TcpListener;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::server::{NamedService, ServerStreamingService, UnaryService};
use tonic::{Request, Response, Status};

const MODELS: &str = include_str!("../../../tests/golden/classification_models.json");
const SUMMARIES: &str = include_str!("../../../tests/golden/classification_summaries.json");
const PERSISTENCE: &str = include_str!("../../../tests/golden/classification_persistence.json");

const SERVICE: &str = "/spark.connect.SparkConnectService/";

/// The reply to a command: the bytes of its `MlCommandResult` (none for a command with no
/// result), or the message of its error.
#[derive(Clone)]
enum Reply {
    Result(Vec<u8>),
    Error(String),
}

#[derive(Default)]
struct Replay {
    /// The `MlCommandResult` of the fit of the test's estimator.
    fit: Option<Vec<u8>>,
    /// The reply to each normalized command of the golden.
    replies: Vec<(proto::MlCommand, Reply)>,
    /// The normalized commands received, in order.
    seen: Vec<proto::MlCommand>,
}

type Shared = Arc<Mutex<Replay>>;

/// An ML command, normalized as the golden is. The golden numbers the object ids of each member
/// entry afresh, so the object of a command is its placeholder `<id0>`, and the input relation of
/// a fetch is cleared. The commands compare as messages, since the map fields of params are
/// encoded in hash order.
fn normalized(mut command: proto::MlCommand) -> proto::MlCommand {
    match &mut command.command {
        Some(proto::ml_command::Command::Fetch(fetch)) => {
            if let Some(object) = &mut fetch.obj_ref {
                object.id = "<id0>".to_string();
            }
        }
        Some(proto::ml_command::Command::Write(write)) => {
            if let Some(proto::ml_command::write::Type::ObjRef(object)) = &mut write.r#type {
                object.id = "<id0>".to_string();
            }
        }
        _ => {}
    }
    if let Some(proto::ml_command::Command::Fetch(fetch)) = &mut command.command {
        for method in &mut fetch.methods {
            for arg in &mut method.args {
                if let Some(proto::fetch::method::args::ArgsType::Input(relation)) =
                    &mut arg.args_type
                {
                    *relation = proto::Relation::default();
                }
            }
        }
    }
    command
}

/// The normalized command of a golden base64 `commands` entry.
fn golden_key(b64: &str) -> proto::MlCommand {
    let bytes = STANDARD.decode(b64).expect("base64 of a golden command");
    normalized(proto::MlCommand::decode(bytes.as_slice()).expect("a golden ML command"))
}

fn golden_bytes(b64: &str) -> Vec<u8> {
    STANDARD.decode(b64).expect("base64 of a golden result")
}

/// The golden of an entry `{commands, results, error?}` and its nested entries: every entry that
/// has commands is registered as the replies of its commands, in order.
fn register(replay: &mut Replay, entry: &Value) {
    let Some(object) = entry.as_object() else {
        return;
    };
    if let Some(commands) = object.get("commands").and_then(Value::as_array) {
        let results = object.get("results").and_then(Value::as_array);
        for (index, command) in commands.iter().enumerate() {
            let key = golden_key(command.as_str().unwrap());
            match results.and_then(|r| r.get(index)).and_then(Value::as_str) {
                Some(result) => {
                    replay
                        .replies
                        .push((key, Reply::Result(golden_bytes(result))));
                }
                None => {
                    if let Some(error) = object.get("error") {
                        let message = error["message"].as_str().unwrap().to_string();
                        replay.replies.push((key, Reply::Error(message)));
                    }
                }
            }
        }
        return;
    }
    for child in object.values() {
        register(replay, child);
    }
}

/// The normalized commands of a golden entry, in order.
fn golden_commands(entry: &Value) -> Vec<proto::MlCommand> {
    entry["commands"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| golden_key(c.as_str().unwrap()))
        .collect()
}

/// A loopback service, answering the ExecutePlan requests of a replay of the golden.
#[derive(Clone)]
struct TestService(Shared);

impl NamedService for TestService {
    const NAME: &'static str = "spark.connect.SparkConnectService";
}

impl Service<http::Request<Body>> for TestService {
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<std::result::Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<Body>) -> Self::Future {
        let method = request.uri().path().trim_start_matches(SERVICE).to_string();
        let shared = self.0.clone();
        Box::pin(async move {
            let mut grpc = tonic::server::Grpc::new(BytesCodec);
            Ok(match method.as_str() {
                "ExecutePlan" => grpc.server_streaming(Execute(shared), request).await,
                "Config" => grpc.unary(Config, request).await,
                "ReleaseExecute" => grpc.unary(ReleaseExecute, request).await,
                other => {
                    eprintln!("unexpected RPC of the golden replay: {other}");
                    grpc.unary(Unsupported, request).await
                }
            })
        })
    }
}

/// ExecutePlan: an ML command is answered from the replay; anything else gets an empty result.
struct Execute(Shared);

impl ServerStreamingService<Vec<u8>> for Execute {
    type Response = Vec<u8>;
    type ResponseStream =
        futures::stream::Iter<std::vec::IntoIter<std::result::Result<Vec<u8>, Status>>>;
    type Future = BoxFuture<Response<Self::ResponseStream>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let request = proto::ExecutePlanRequest::decode(request.into_inner().as_slice()).unwrap();
        let session_id = request.session_id.clone();
        let operation_id = request.operation_id.clone().unwrap_or_default();
        let command = match request.plan.as_ref().and_then(|p| p.op_type.as_ref()) {
            Some(proto::plan::OpType::Command(command)) => match &command.command_type {
                Some(proto::command::CommandType::MlCommand(ml)) => Some(ml.clone()),
                _ => None,
            },
            _ => None,
        };
        let mut frames: Vec<std::result::Result<Vec<u8>, Status>> = Vec::new();
        if let Some(ml) = command {
            // The client releases the objects that it drops, which the golden has no command for.
            let released = matches!(ml.command, Some(proto::ml_command::Command::Delete(_)));
            let is_fit = matches!(ml.command, Some(proto::ml_command::Command::Fit(_)));
            let key = normalized(ml);
            let mut replay = self.0.lock().unwrap();
            let reply = if released {
                Some(Reply::Result(Vec::new()))
            } else {
                replay.seen.push(key.clone());
                if is_fit {
                    Some(Reply::Result(replay.fit.clone().expect("a fit result")))
                } else {
                    replay
                        .replies
                        .iter()
                        .find(|(command, _)| *command == key)
                        .map(|(_, reply)| reply.clone())
                }
            };
            match reply {
                Some(Reply::Result(bytes)) => {
                    if !bytes.is_empty() {
                        let result = proto::MlCommandResult::decode(bytes.as_slice()).unwrap();
                        frames.push(Ok(response(
                            &session_id,
                            &operation_id,
                            proto::execute_plan_response::ResponseType::MlCommandResult(result),
                        )
                        .encode_to_vec()));
                    }
                }
                Some(Reply::Error(message)) => {
                    frames.push(Err(Status::internal(message)));
                }
                None => {
                    eprintln!("no golden for {key:?}");
                    frames.push(Err(Status::invalid_argument(
                        "an ML command that the golden does not have",
                    )));
                }
            }
        }
        if frames.iter().all(|f| f.is_ok()) {
            frames.push(Ok(response(
                &session_id,
                &operation_id,
                proto::execute_plan_response::ResponseType::ResultComplete(
                    proto::execute_plan_response::ResultComplete {},
                ),
            )
            .encode_to_vec()));
        }
        Box::pin(async { Ok(Response::new(stream::iter(frames))) })
    }
}

fn response(
    session_id: &str,
    operation_id: &str,
    response_type: proto::execute_plan_response::ResponseType,
) -> proto::ExecutePlanResponse {
    proto::ExecutePlanResponse {
        session_id: session_id.to_string(),
        operation_id: operation_id.to_string(),
        response_id: "golden".into(),
        response_type: Some(response_type),
        ..Default::default()
    }
}

/// Config: answers with no pairs, which leaves plan compression off.
struct Config;

impl UnaryService<Vec<u8>> for Config {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let session_id = proto::ConfigRequest::decode(request.into_inner().as_slice())
            .unwrap()
            .session_id;
        let body = proto::ConfigResponse {
            session_id,
            ..Default::default()
        }
        .encode_to_vec();
        Box::pin(async { Ok(Response::new(body)) })
    }
}

/// ReleaseExecute: the client releases an execution after it has read its responses.
struct ReleaseExecute;

impl UnaryService<Vec<u8>> for ReleaseExecute {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let request =
            proto::ReleaseExecuteRequest::decode(request.into_inner().as_slice()).unwrap();
        let body = proto::ReleaseExecuteResponse {
            session_id: request.session_id,
            operation_id: Some(request.operation_id),
            ..Default::default()
        }
        .encode_to_vec();
        Box::pin(async { Ok(Response::new(body)) })
    }
}

/// An RPC that the golden does not replay: refused.
struct Unsupported;

impl UnaryService<Vec<u8>> for Unsupported {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, _request: Request<Vec<u8>>) -> Self::Future {
        Box::pin(async { Err(Status::unimplemented("not replayed by the golden")) })
    }
}

/// Serves the replay on a thread of its own, and returns a session of it.
fn serve(replay: Replay) -> (SparkSession, Shared) {
    let shared = Arc::new(Mutex::new(replay));
    let service = TestService(shared.clone());
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            sender.send(listener.local_addr().unwrap()).unwrap();
            let incoming = stream::unfold(listener, |listener| async {
                Some((listener.accept().await.map(|(socket, _)| socket), listener))
            });
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
    });
    let address = receiver.recv().unwrap();
    let session = SparkSession::builder()
        .remote(&format!("sc://{address}"))
        .get_or_create()
        .unwrap();
    (session, shared)
}

/// The replay of the fit `key` of the golden, with the golden entries `entries`.
fn replay_of(models: &Value, key: &str, entries: &[&Value]) -> Replay {
    let mut replay = Replay {
        fit: Some(golden_bytes(
            models["fits"][key]["fit"]["results"][0].as_str().unwrap(),
        )),
        ..Default::default()
    };
    for entry in entries {
        register(&mut replay, entry);
    }
    replay
}

/// A JSON form of a value, as the golden writes it: floats are numbers, NaN and infinities are
/// strings, and vectors and matrices are objects of their type, shape and values.
trait Golden {
    fn golden(&self) -> Value;
}

fn number(x: f64) -> Value {
    if x.is_nan() {
        json!("NaN")
    } else if x.is_infinite() {
        json!(if x > 0.0 { "Infinity" } else { "-Infinity" })
    } else {
        json!(x)
    }
}

impl Golden for f64 {
    fn golden(&self) -> Value {
        number(*self)
    }
}

impl Golden for i32 {
    fn golden(&self) -> Value {
        json!(self)
    }
}

impl Golden for bool {
    fn golden(&self) -> Value {
        json!(self)
    }
}

impl Golden for String {
    fn golden(&self) -> Value {
        json!(self)
    }
}

impl Golden for Vec<f64> {
    fn golden(&self) -> Value {
        Value::Array(self.iter().map(|x| number(*x)).collect())
    }
}

impl Golden for MlVector {
    fn golden(&self) -> Value {
        match self {
            MlVector::Dense(values) => json!({
                "type": "dense",
                "size": values.len(),
                "values": values.iter().map(|x| number(*x)).collect::<Vec<_>>(),
            }),
            MlVector::Sparse {
                size,
                indices,
                values,
            } => json!({
                "type": "sparse",
                "size": size,
                "indices": indices,
                "values": values.iter().map(|x| number(*x)).collect::<Vec<_>>(),
            }),
        }
    }
}

impl Golden for MlMatrix {
    fn golden(&self) -> Value {
        match self {
            MlMatrix::Dense {
                num_rows,
                num_cols,
                values,
                is_transposed,
            } => json!({
                "type": "dense",
                "numRows": num_rows,
                "numCols": num_cols,
                "isTransposed": is_transposed,
                "values": values.iter().map(|x| number(*x)).collect::<Vec<_>>(),
            }),
            MlMatrix::Sparse {
                num_rows,
                num_cols,
                col_ptrs,
                row_indices,
                values,
                is_transposed,
            } => json!({
                "type": "sparse",
                "numRows": num_rows,
                "numCols": num_cols,
                "isTransposed": is_transposed,
                "colPtrs": col_ptrs,
                "rowIndices": row_indices,
                "values": values.iter().map(|x| number(*x)).collect::<Vec<_>>(),
            }),
        }
    }
}

impl Golden for Value {
    fn golden(&self) -> Value {
        self.clone()
    }
}

/// A summary or model object, as the golden writes an object: its class and its reference.
fn reference(class: &str, target: &MlTarget) -> Value {
    json!({"type": class, "ref": target.id()})
}

/// Asserts that `actual` is the golden `expected`. The golden is JSON, whose decimals are read
/// within an ulp by serde_json, so numbers compare within 1e-12 relative; everything else exactly.
fn assert_golden(actual: &Value, expected: &Value, label: &str) {
    fn same(actual: &Value, expected: &Value) -> bool {
        match (actual, expected) {
            (Value::Number(a), Value::Number(b)) => match (a.as_f64(), b.as_f64()) {
                (Some(a), Some(b)) => a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs()),
                _ => a == b,
            },
            (Value::Array(a), Value::Array(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same(a, b))
            }
            (Value::Object(a), Value::Object(b)) => {
                a.len() == b.len() && a.iter().all(|(k, a)| b.get(k).is_some_and(|b| same(a, b)))
            }
            _ => actual == expected,
        }
    }
    assert!(
        same(actual, expected),
        "{label}: value {actual} is not the golden {expected}"
    );
}

/// Checks the members of one golden entry: each call's requests and value against the golden's.
struct Checker {
    shared: Shared,
    members: Value,
    checked: BTreeSet<String>,
}

impl Checker {
    fn new(shared: &Shared, members: &Value) -> Self {
        Self {
            shared: shared.clone(),
            members: members.clone(),
            checked: BTreeSet::new(),
        }
    }

    fn entry(&self, label: &str) -> Value {
        self.members
            .get(label)
            .unwrap_or_else(|| panic!("the golden has no member {label}"))
            .clone()
    }

    fn seen(&self) -> usize {
        self.shared.lock().unwrap().seen.len()
    }

    fn requests_since(&self, from: usize) -> Vec<proto::MlCommand> {
        self.shared.lock().unwrap().seen[from..].to_vec()
    }

    /// The value of `label`: the call sends the golden's commands, and returns the golden's value.
    fn value<T: Golden>(&mut self, label: &str, call: impl FnOnce() -> Result<T>) {
        let entry = self.entry(label);
        let before = self.seen();
        let actual = call().unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(
            self.requests_since(before),
            golden_commands(&entry),
            "{label}: requests"
        );
        assert_golden(&actual.golden(), &entry["value"], label);
        self.checked.insert(label.to_string());
    }

    /// The value of `label` of a call that the client makes without a request.
    fn local<T: Golden>(&mut self, label: &str, call: impl FnOnce() -> T) {
        let entry = self.entry(label);
        let before = self.seen();
        let actual = call();
        assert!(self.requests_since(before).is_empty(), "{label}: requests");
        assert_golden(&actual.golden(), &entry["value"], label);
        self.checked.insert(label.to_string());
    }

    /// The error of `label`, whose message is the golden's.
    fn error<T: Golden>(&mut self, label: &str, call: impl FnOnce() -> Result<T>) {
        let entry = self.entry(label);
        let expected = entry["error"]["message"].as_str().unwrap().to_string();
        let error = match call() {
            Ok(_) => panic!("{label}: the call succeeded"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains(&expected),
            "{label}: {error:?} does not carry {expected:?}"
        );
        self.checked.insert(label.to_string());
    }

    /// Asserts that every golden member of the entry is checked, or is one of `skipped`.
    fn finish(self, skipped: &[&str]) {
        let golden: BTreeSet<String> = self.members.as_object().unwrap().keys().cloned().collect();
        let skipped: BTreeSet<String> = skipped.iter().map(|s| s.to_string()).collect();
        let missing: Vec<&String> = golden
            .iter()
            .filter(|label| !self.checked.contains(*label) && !skipped.contains(*label))
            .collect();
        assert!(
            missing.is_empty(),
            "golden members not checked: {missing:?}"
        );
    }
}

fn dense(values: &[f64]) -> MlVector {
    MlVector::Dense(values.to_vec())
}

/// The vectors that the golden predicts, of the Naive Bayes models and of the others.
fn vectors(naive_bayes: bool) -> [(&'static str, MlVector); 3] {
    let sparse = |value: f64| MlVector::Sparse {
        size: 2,
        indices: vec![1],
        values: vec![value],
    };
    if naive_bayes {
        [
            ("dense", dense(&[1.0, 0.0])),
            ("dense2", dense(&[0.0, 1.0])),
            ("sparse", sparse(1.0)),
        ]
    } else {
        [
            ("dense", dense(&[0.5, 1.5])),
            ("dense2", dense(&[3.0, -0.25])),
            ("sparse", sparse(2.0)),
        ]
    }
}

/// The members that every classification model has (`numClasses`, `numFeatures`, and the
/// prediction of each vector), as the golden captures them.
fn classification_members<M: ProbabilisticClassificationModel>(
    checker: &mut Checker,
    model: &M,
    naive_bayes: bool,
) {
    for (name, vector) in vectors(naive_bayes) {
        checker.value(&format!("predict-{name}"), || model.predict(&vector));
        checker.value(&format!("predictRaw-{name}"), || model.predict_raw(&vector));
        checker.value(&format!("predictProbability-{name}"), || {
            model.predict_probability(&vector)
        });
    }
    checker.value("numClasses", || model.num_classes());
    checker.value("numFeatures", || model.num_features());
}

/// The members of a classification summary (`_ClassificationSummary`), except the DataFrames.
fn classification_summary_members<S: ClassificationSummary>(checker: &mut Checker, s: &S) {
    checker.value("accuracy", || s.accuracy());
    checker.value("fMeasureByLabel-0.5", || s.f_measure_by_label(0.5));
    checker.value("fMeasureByLabel-1.0", || s.f_measure_by_label(1.0));
    checker.value("falsePositiveRateByLabel", || {
        s.false_positive_rate_by_label()
    });
    checker.value("labelCol", || s.label_col());
    checker.value("labels", || s.labels());
    checker.value("precisionByLabel", || s.precision_by_label());
    checker.value("predictionCol", || s.prediction_col());
    checker.value("recallByLabel", || s.recall_by_label());
    checker.value("truePositiveRateByLabel", || {
        s.true_positive_rate_by_label()
    });
    checker.value("weightCol", || s.weight_col());
    checker.value("weightedFMeasure-0.5", || s.weighted_f_measure(0.5));
    checker.value("weightedFMeasure-1.0", || s.weighted_f_measure(1.0));
    checker.value("weightedFalsePositiveRate", || {
        s.weighted_false_positive_rate()
    });
    checker.value("weightedPrecision", || s.weighted_precision());
    checker.value("weightedRecall", || s.weighted_recall());
    checker.value("weightedTruePositiveRate", || {
        s.weighted_true_positive_rate()
    });
}

/// The scalar members of a binary classification summary (`_BinaryClassificationSummary`).
fn binary_summary_members<S: BinaryClassificationSummary>(checker: &mut Checker, s: &S) {
    checker.value("areaUnderROC", || s.area_under_roc());
    checker.value("scoreCol", || s.score_col());
}

/// The members of a training summary (`_TrainingSummary`).
fn training_summary_members<S: TrainingSummary>(checker: &mut Checker, s: &S) {
    checker.value("objectiveHistory", || s.objective_history());
    checker.value("totalIterations", || s.total_iterations());
}

/// The DataFrame members of a summary, which the golden captures in the server's own relations,
/// and the Python-only representations.
const SUMMARY_SKIPPED: [&str; 8] = [
    "predictions",
    "roc",
    "pr",
    "fMeasureByThreshold",
    "precisionByThreshold",
    "recallByThreshold",
    "repr()",
    "str()",
];

fn fit_naive_bayes(session: &SparkSession, estimator: &mut NaiveBayes) -> Result<NaiveBayesModel> {
    estimator.fit(&session.range(3)?)
}

#[test]
fn naive_bayes_members_match_official_pyspark() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    for (key, model_type) in [
        ("NaiveBayes.multinomial", "multinomial"),
        ("NaiveBayes.complement", "complement"),
        ("NaiveBayes.bernoulli", "bernoulli"),
        ("NaiveBayes.gaussian", "gaussian"),
    ] {
        let entry = &models["models"][key];
        let errors = models["models"].get(format!("{key}.errors"));
        let mut replay = replay_of(&models, key, &[entry]);
        if let Some(errors) = errors {
            register(&mut replay, errors);
        }
        let (session, shared) = serve(replay);
        let mut estimator = NaiveBayes::new()
            .set_model_type(model_type)
            .set_weight_col("weight");
        let model = fit_naive_bayes(&session, &mut estimator).unwrap();

        let mut checker = Checker::new(&shared, &entry["members"]);
        checker.value("pi", || model.pi());
        checker.value("theta", || model.theta());
        checker.value("sigma", || model.sigma());
        classification_members(&mut checker, &model, true);
        if let Some(errors) = errors {
            // The Bernoulli model takes 0 or 1 features only: the server's error, as the golden.
            let mut failures = Checker::new(&shared, &errors["members"]);
            failures.error("predict-outside-0-1", || model.predict(&dense(&[0.5, 1.5])));
            failures.finish(&[]);
        }
        checker.finish(&["repr()", "str()", "transform"]);
    }
}

#[test]
fn naive_bayes_load_and_save_match_official_pyspark() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let persistence: Value = serde_json::from_str(PERSISTENCE).unwrap();
    for (key, model_type) in [
        ("NaiveBayes.multinomial", "multinomial"),
        ("NaiveBayes.complement", "complement"),
        ("NaiveBayes.bernoulli", "bernoulli"),
        ("NaiveBayes.gaussian", "gaussian"),
    ] {
        let saved = &persistence["persistence"][format!("{key}.model")];

        // Load: a read of the saved model, whose members are the loaded model's.
        let mut replay = Replay::default();
        register(&mut replay, saved);
        let (session, shared) = serve(replay);
        let loaded = NaiveBayesModel::load(&session, "<path>").unwrap();
        let mut checker = Checker::new(&shared, &saved["members"]["members"]);
        checker.value("predict-dense", || loaded.predict(&dense(&[1.0, 0.0])));
        checker.finish(&["repr()", "str()", "transform"]);

        // Save: a write of the fitted model, with the params that its estimator sets.
        let mut replay = replay_of(&models, key, &[&models["models"][key]]);
        register(&mut replay, saved);
        let (session, shared) = serve(replay);
        let mut estimator = NaiveBayes::new()
            .set_model_type(model_type)
            .set_weight_col("weight");
        let model = fit_naive_bayes(&session, &mut estimator).unwrap();
        let before = shared.lock().unwrap().seen.len();
        model.write().unwrap().overwrite().save("<path>").unwrap();
        let seen = shared.lock().unwrap().seen[before..].to_vec();
        assert_eq!(seen, golden_commands(&saved["save"]), "{key}: save");
    }
}

#[test]
fn multilayer_perceptron_members_and_summaries_match_official_pyspark() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let summaries: Value = serde_json::from_str(SUMMARIES).unwrap();
    let key = "MultilayerPerceptronClassifier";
    let entry = &models["models"][key];
    let training = &summaries["summaries"][format!("{key}.training")];
    let evaluate = &summaries["summaries"][format!("{key}.evaluate")];
    let replay = replay_of(&models, key, &[entry, training, evaluate]);
    let (session, shared) = serve(replay);
    let mut estimator = MultilayerPerceptronClassifier::new()
        .set_layers([2, 4, 3])
        .set_max_iter(30)
        .set_seed(42)
        .set_block_size(4)
        .set_step_size(0.1);
    let model = estimator.fit(&session.range(3).unwrap()).unwrap();

    let mut checker = Checker::new(&shared, &entry["members"]);
    checker.value("weights", || model.weights());
    checker.local("hasSummary", || model.has_summary());
    classification_members(&mut checker, &model, false);
    checker.value("summary()", || {
        model.summary().map(|s| {
            reference(
                "MultilayerPerceptronClassificationTrainingSummary",
                s.summary_target(),
            )
        })
    });
    checker.value("evaluate", || {
        let df = session.range(3)?;
        model.evaluate(&df).map(|s| {
            reference(
                "MultilayerPerceptronClassificationSummary",
                s.summary_target(),
            )
        })
    });
    checker.finish(&["repr()", "str()", "transform"]);

    let summary = model.summary().unwrap();
    let mut checker = Checker::new(&shared, &training["members"]);
    classification_summary_members(&mut checker, &summary);
    training_summary_members(&mut checker, &summary);
    checker.finish(&SUMMARY_SKIPPED);

    let summary = model.evaluate(&session.range(3).unwrap()).unwrap();
    let mut checker = Checker::new(&shared, &evaluate["members"]);
    classification_summary_members(&mut checker, &summary);
    checker.finish(&SUMMARY_SKIPPED);
}

#[test]
fn multilayer_perceptron_load_and_save_match_official_pyspark() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let persistence: Value = serde_json::from_str(PERSISTENCE).unwrap();
    let key = "MultilayerPerceptronClassifier";
    let saved = &persistence["persistence"][format!("{key}.model")];

    let mut replay = Replay::default();
    register(&mut replay, saved);
    let (session, shared) = serve(replay);
    let loaded = MultilayerPerceptronClassificationModel::load(&session, "<path>").unwrap();
    let mut checker = Checker::new(&shared, &saved["members"]["members"]);
    checker.value("predict-dense", || loaded.predict(&dense(&[0.5, 1.5])));
    checker.finish(&["repr()", "str()", "transform"]);
    // A loaded model has no training summary, as PySpark's.
    assert!(!loaded.has_summary());
    assert!(loaded.summary().is_err());

    let mut replay = replay_of(&models, key, &[&models["models"][key]]);
    register(&mut replay, saved);
    let (session, shared) = serve(replay);
    let model = MultilayerPerceptronClassifier::new()
        .set_layers([2, 4, 3])
        .set_max_iter(30)
        .set_seed(42)
        .set_block_size(4)
        .set_step_size(0.1)
        .fit(&session.range(3).unwrap())
        .unwrap();
    let before = shared.lock().unwrap().seen.len();
    model.write().unwrap().overwrite().save("<path>").unwrap();
    let seen = shared.lock().unwrap().seen[before..].to_vec();
    assert_eq!(seen, golden_commands(&saved["save"]), "{key}: save");
}

#[test]
fn fm_members_and_summaries_match_official_pyspark() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let summaries: Value = serde_json::from_str(SUMMARIES).unwrap();
    let key = "FMClassifier";
    let entry = &models["models"][key];
    let training = &summaries["summaries"][format!("{key}.training")];
    let evaluate = &summaries["summaries"][format!("{key}.evaluate")];
    let replay = replay_of(&models, key, &[entry, training, evaluate]);
    let (session, shared) = serve(replay);
    let model: FMClassificationModel = FMClassifier::new()
        .set_factor_size(2)
        .set_max_iter(20)
        .set_seed(42)
        .set_reg_param(0.01)
        .set_step_size(0.1)
        .fit(&session.range(3).unwrap())
        .unwrap();

    let mut checker = Checker::new(&shared, &entry["members"]);
    checker.value("factors", || model.factors());
    checker.value("linear", || model.linear());
    checker.value("intercept", || model.intercept());
    checker.local("hasSummary", || model.has_summary());
    classification_members(&mut checker, &model, false);
    checker.value("summary()", || {
        model
            .summary()
            .map(|s| reference("FMClassificationTrainingSummary", s.summary_target()))
    });
    checker.value("evaluate", || {
        let df = session.range(3)?;
        model
            .evaluate(&df)
            .map(|s| reference("FMClassificationSummary", s.summary_target()))
    });
    checker.finish(&["repr()", "str()", "transform"]);

    let summary = model.summary().unwrap();
    let mut checker = Checker::new(&shared, &training["members"]);
    classification_summary_members(&mut checker, &summary);
    binary_summary_members(&mut checker, &summary);
    training_summary_members(&mut checker, &summary);
    checker.finish(&SUMMARY_SKIPPED);

    let summary = model.evaluate(&session.range(3).unwrap()).unwrap();
    let mut checker = Checker::new(&shared, &evaluate["members"]);
    classification_summary_members(&mut checker, &summary);
    binary_summary_members(&mut checker, &summary);
    checker.finish(&SUMMARY_SKIPPED);
}

#[test]
fn fm_load_and_save_match_official_pyspark() {
    let models: Value = serde_json::from_str(MODELS).unwrap();
    let persistence: Value = serde_json::from_str(PERSISTENCE).unwrap();
    let key = "FMClassifier";
    let saved = &persistence["persistence"][format!("{key}.model")];

    let mut replay = Replay::default();
    register(&mut replay, saved);
    let (session, shared) = serve(replay);
    let loaded = FMClassificationModel::load(&session, "<path>").unwrap();
    let mut checker = Checker::new(&shared, &saved["members"]["members"]);
    checker.value("predict-dense", || loaded.predict(&dense(&[0.5, 1.5])));
    checker.finish(&["repr()", "str()", "transform"]);
    assert!(!loaded.has_summary());
    assert!(loaded.summary().is_err());

    let mut replay = replay_of(&models, key, &[&models["models"][key]]);
    register(&mut replay, saved);
    let (session, shared) = serve(replay);
    let model = FMClassifier::new()
        .set_factor_size(2)
        .set_max_iter(20)
        .set_seed(42)
        .set_reg_param(0.01)
        .set_step_size(0.1)
        .fit(&session.range(3).unwrap())
        .unwrap();
    let before = shared.lock().unwrap().seen.len();
    model.write().unwrap().overwrite().save("<path>").unwrap();
    let seen = shared.lock().unwrap().seen[before..].to_vec();
    assert_eq!(seen, golden_commands(&saved["save"]), "{key}: save");
}
