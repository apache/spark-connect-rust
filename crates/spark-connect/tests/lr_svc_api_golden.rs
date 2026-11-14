// SPDX-License-Identifier: Apache-2.0
//! The typed accessors of LogisticRegression, LinearSVC and their models and summaries, against
//! the entries that official PySpark 4.3.0 captured (`tests/golden/classification_*.json`). No
//! Spark server is needed: a loopback Spark Connect service answers each ML command with the
//! golden result of that command, in the golden order, and the test checks that the accessor sent
//! the golden command and returned the golden value. The commands are compared after the golden's
//! normalization: the operator uids, the paths and the datasets (whose data the golden clears)
//! are cleared on both sides.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use base64::{engine::general_purpose::STANDARD, Engine};
use futures::stream;
use prost::Message;
use serde_json::{json, Value};
use spark_connect::ml::{
    BinaryClassificationSummary, BinaryLogisticRegressionSummary,
    BinaryLogisticRegressionTrainingSummary, ClassificationModel, ClassificationSummary, Estimator,
    LinearSVC, LinearSVCModel, LinearSVCSummary, LinearSVCTrainingSummary, LogisticRegression,
    LogisticRegressionEvaluation, LogisticRegressionModel, LogisticRegressionModelSummary,
    LogisticRegressionSummary, LogisticRegressionSummaryApi, LogisticRegressionTrainingSummary,
    MlMatrix, MlVector, Model, Params, PredictionModel, ProbabilisticClassificationModel,
    TrainingSummary, Transformer,
};
use spark_connect::SparkSession;
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_core::error::Result as CoreResult;
use spark_connect_proto as proto;
use tokio::net::TcpListener;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::server::{NamedService, ServerStreamingService, UnaryService};
use tonic::{Request, Response, Status};

const SERVICE: &str = "/spark.connect.SparkConnectService/";
const MODELS: &str = include_str!("../../../tests/golden/classification_models.json");
const SUMMARIES: &str = include_str!("../../../tests/golden/classification_summaries.json");
const PERSISTENCE: &str = include_str!("../../../tests/golden/classification_persistence.json");

/// The reply to one ML command: its result, an error, or none.
#[derive(Clone)]
enum Reply {
    Result(proto::MlCommandResult),
    Error(String),
    Empty,
}

/// One expected ML command, with its normalized command and its reply.
struct Exchange {
    command: proto::MlCommand,
    reply: Reply,
}

/// The commands that the test expects, in order, and the requests that were not expected.
#[derive(Default)]
struct Script {
    expected: VecDeque<Exchange>,
    unexpected: Vec<String>,
}

impl Script {
    fn answer(
        &mut self,
        command: proto::MlCommand,
    ) -> Result<Option<proto::MlCommandResult>, Status> {
        let command = normalize(command);
        let Some(exchange) = self.expected.pop_front() else {
            let message = format!("unexpected ML command {command:?}");
            self.unexpected.push(message.clone());
            return Err(Status::internal(message));
        };
        if exchange.command != command {
            let message = format!(
                "ML command differs from the golden\n golden: {:?}\n actual: {command:?}",
                exchange.command
            );
            self.unexpected.push(message.clone());
            return Err(Status::internal(message));
        }
        match exchange.reply {
            Reply::Result(result) => Ok(Some(result)),
            Reply::Empty => Ok(None),
            Reply::Error(message) => Err(Status::internal(message)),
        }
    }
}

/// The golden's normalization of an ML command: the uids of operators, the paths and the datasets
/// are cleared, as the golden clears them, and so are the inputs of fetches. The object ids are
/// the golden's placeholders, which it numbers per entry, so they are not compared.
fn normalize(mut command: proto::MlCommand) -> proto::MlCommand {
    use proto::fetch::method::args::ArgsType;
    use proto::ml_command::{write::Type as WriteType, Command};
    match command.command.as_mut() {
        Some(Command::Fit(fit)) => {
            if let Some(operator) = fit.estimator.as_mut() {
                operator.uid.clear();
            }
            fit.dataset = None;
        }
        Some(Command::Fetch(fetch)) => {
            if let Some(object) = fetch.obj_ref.as_mut() {
                object.id = "<obj>".into();
            }
            for method in &mut fetch.methods {
                for arg in &mut method.args {
                    if let Some(ArgsType::Input(relation)) = arg.args_type.as_mut() {
                        *relation = proto::Relation::default();
                    }
                }
            }
        }
        Some(Command::Read(read)) => {
            if let Some(operator) = read.operator.as_mut() {
                operator.uid.clear();
            }
            read.path.clear();
        }
        Some(Command::Write(write)) => {
            match write.r#type.as_mut() {
                Some(WriteType::Operator(operator)) => operator.uid.clear(),
                Some(WriteType::ObjRef(object)) => object.id = "<obj>".into(),
                None => {}
            }
            write.path.clear();
        }
        _ => {}
    }
    command
}

/// A loopback Spark Connect service that answers ExecutePlan with the expected replies.
#[derive(Clone)]
struct FakeService(Arc<Mutex<Script>>);

impl NamedService for FakeService {
    const NAME: &'static str = "spark.connect.SparkConnectService";
}

impl Service<http::Request<Body>> for FakeService {
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<Body>) -> Self::Future {
        let method = request.uri().path().trim_start_matches(SERVICE).to_string();
        let script = self.0.clone();
        Box::pin(async move {
            let mut grpc = tonic::server::Grpc::new(BytesCodec);
            Ok(match method.as_str() {
                "ExecutePlan" => grpc.server_streaming(Execute(script), request).await,
                "Config" => grpc.unary(Config, request).await,
                "ReleaseExecute" => grpc.unary(Empty, request).await,
                other => {
                    eprintln!("unexpected RPC {other}");
                    grpc.unary(Unimplemented, request).await
                }
            })
        })
    }
}

/// ExecutePlan: an ML command, answered from the script; the deletes of released objects are not
/// part of the script and get an empty completion.
struct Execute(Arc<Mutex<Script>>);

impl ServerStreamingService<Vec<u8>> for Execute {
    type Response = Vec<u8>;
    type ResponseStream = futures::stream::Iter<std::vec::IntoIter<Result<Vec<u8>, Status>>>;
    type Future = BoxFuture<Response<Self::ResponseStream>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let request = proto::ExecutePlanRequest::decode(request.into_inner().as_slice()).unwrap();
        let session_id = request.session_id.clone();
        let command = match request.plan.and_then(|plan| plan.op_type) {
            Some(proto::plan::OpType::Command(command)) => command.command_type,
            _ => None,
        };
        let reply = match command {
            Some(proto::command::CommandType::MlCommand(ml)) => {
                if matches!(ml.command, Some(proto::ml_command::Command::Delete(_))) {
                    Ok(None)
                } else {
                    self.0.lock().unwrap().answer(ml)
                }
            }
            other => Err(Status::internal(format!("unexpected command {other:?}"))),
        };
        Box::pin(async move {
            let result = reply?;
            let mut frames = Vec::new();
            if let Some(result) = result {
                let response = proto::ExecutePlanResponse {
                    session_id: session_id.clone(),
                    response_id: "result".into(),
                    response_type: Some(
                        proto::execute_plan_response::ResponseType::MlCommandResult(result),
                    ),
                    ..Default::default()
                };
                frames.push(Ok(response.encode_to_vec()));
            }
            let complete = proto::ExecutePlanResponse {
                session_id,
                response_id: "complete".into(),
                response_type: Some(proto::execute_plan_response::ResponseType::ResultComplete(
                    proto::execute_plan_response::ResultComplete {},
                )),
                ..Default::default()
            };
            frames.push(Ok(complete.encode_to_vec()));
            Ok(Response::new(stream::iter(frames)))
        })
    }
}

/// An RPC whose reply has no fields that the client reads: an empty message.
struct Empty;

impl UnaryService<Vec<u8>> for Empty {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, _: Request<Vec<u8>>) -> Self::Future {
        Box::pin(async { Ok(Response::new(Vec::new())) })
    }
}

/// An RPC that the golden does not need: it fails.
struct Unimplemented;

impl UnaryService<Vec<u8>> for Unimplemented {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, _: Request<Vec<u8>>) -> Self::Future {
        Box::pin(async { Err(Status::unimplemented("not part of the golden")) })
    }
}

/// Config: answers with no pairs.
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

/// A server with its script, serving on a thread of its own, since a session blocks the thread
/// that calls it, and the one session of the test.
struct Server {
    script: Arc<Mutex<Script>>,
    session: SparkSession,
}

impl Server {
    fn start() -> Self {
        let script = Arc::new(Mutex::new(Script::default()));
        let service = FakeService(script.clone());
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
        let url = format!("sc://{}", receiver.recv().unwrap());
        let session = SparkSession::builder()
            .remote(&url)
            .get_or_create()
            .unwrap();
        Self { script, session }
    }

    fn session(&self) -> SparkSession {
        self.session.clone()
    }

    /// Expect the golden entry's commands, each answered with its golden result or error.
    fn expect(&self, entry: &Value) {
        let commands = entry["commands"].as_array().unwrap();
        let results = entry["results"].as_array().unwrap();
        let mut script = self.script.lock().unwrap();
        for (index, command) in commands.iter().enumerate() {
            let command = proto::MlCommand::decode(
                STANDARD
                    .decode(command.as_str().unwrap())
                    .unwrap()
                    .as_slice(),
            )
            .unwrap();
            let reply = if let Some(result) = results.get(index) {
                let result = proto::MlCommandResult::decode(
                    STANDARD
                        .decode(result.as_str().unwrap())
                        .unwrap()
                        .as_slice(),
                )
                .unwrap();
                Reply::Result(result)
            } else if let Some(error) = entry.get("error") {
                Reply::Error(error["message"].as_str().unwrap().to_string())
            } else {
                Reply::Empty
            };
            script.expected.push_back(Exchange {
                command: normalize(command),
                reply,
            });
        }
    }

    /// Assert that every expected command was sent, and that no other command was.
    fn assert_consumed(&self, label: &str) {
        let script = self.script.lock().unwrap();
        assert!(
            script.unexpected.is_empty(),
            "{label}: {:?}",
            script.unexpected
        );
        assert!(
            script.expected.is_empty(),
            "{label}: {} golden command(s) not sent",
            script.expected.len()
        );
    }
}

fn golden(text: &str) -> Value {
    serde_json::from_str(text).unwrap()
}

/// Assert that `actual` is the golden `expected`: numbers to 1e-9 relative, the rest exactly.
fn assert_close(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!(
                (a - e).abs() <= 1e-9 * e.abs().max(1.0),
                "{path}: {a} != {e}"
            );
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "{path}: length");
            for (index, (a, e)) in a.iter().zip(e).enumerate() {
                assert_close(a, e, &format!("{path}[{index}]"));
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            let mut keys: Vec<_> = a.keys().collect();
            let mut expected_keys: Vec<_> = e.keys().collect();
            keys.sort();
            expected_keys.sort();
            assert_eq!(keys, expected_keys, "{path}: keys");
            for (key, e) in e {
                assert_close(&a[key], e, &format!("{path}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

fn vector_json(vector: &MlVector) -> Value {
    match vector {
        MlVector::Dense(values) => json!({
            "size": values.len(), "type": "dense", "values": values,
        }),
        MlVector::Sparse {
            size,
            indices,
            values,
        } => json!({
            "size": size, "type": "sparse", "indices": indices, "values": values,
        }),
    }
}

fn matrix_json(matrix: &MlMatrix) -> Value {
    match matrix {
        MlMatrix::Dense {
            num_rows,
            num_cols,
            values,
            is_transposed,
        } => json!({
            "numRows": num_rows, "numCols": num_cols, "values": values,
            "isTransposed": is_transposed, "type": "dense",
        }),
        other => panic!("no golden sparse matrix in this family: {other:?}"),
    }
}

/// The member `key` of a golden model entry, as a feature vector of its name's suffix.
fn feature_vector(key: &str) -> MlVector {
    match key.rsplit('-').next().unwrap() {
        "dense" => MlVector::Dense(vec![0.5, 1.5]),
        "dense2" => MlVector::Dense(vec![3.0, -0.25]),
        "sparse" => MlVector::Sparse {
            size: 2,
            indices: vec![1],
            values: vec![2.0],
        },
        other => panic!("unknown feature vector {other}"),
    }
}

type Member = CoreResult<Value>;

fn num<T: Into<Value>>(value: CoreResult<T>) -> Member {
    value.map(Into::into)
}

fn list(value: CoreResult<Vec<f64>>) -> Member {
    value.map(|values| json!(values))
}

/// The members of a classification summary (`_ClassificationSummary`).
fn classification<S: ClassificationSummary>(summary: &S, key: &str) -> Option<Member> {
    if let Some(beta) = key.strip_prefix("fMeasureByLabel-") {
        return Some(list(summary.f_measure_by_label(beta.parse().unwrap())));
    }
    if let Some(beta) = key.strip_prefix("weightedFMeasure-") {
        return Some(num(summary.weighted_f_measure(beta.parse().unwrap())));
    }
    Some(match key {
        "accuracy" => num(summary.accuracy()),
        "falsePositiveRateByLabel" => list(summary.false_positive_rate_by_label()),
        "labelCol" => num(summary.label_col()),
        "labels" => list(summary.labels()),
        "precisionByLabel" => list(summary.precision_by_label()),
        "predictionCol" => num(summary.prediction_col()),
        "recallByLabel" => list(summary.recall_by_label()),
        "truePositiveRateByLabel" => list(summary.true_positive_rate_by_label()),
        "weightCol" => num(summary.weight_col()),
        "weightedFalsePositiveRate" => num(summary.weighted_false_positive_rate()),
        "weightedPrecision" => num(summary.weighted_precision()),
        "weightedRecall" => num(summary.weighted_recall()),
        "weightedTruePositiveRate" => num(summary.weighted_true_positive_rate()),
        _ => return None,
    })
}

/// The members of a binary classification summary (`_BinaryClassificationSummary`).
fn binary<S: BinaryClassificationSummary>(summary: &S, key: &str) -> Option<Member> {
    Some(match key {
        "areaUnderROC" => num(summary.area_under_roc()),
        "scoreCol" => num(summary.score_col()),
        _ => return None,
    })
}

/// The members of a training summary (`_TrainingSummary`).
fn training<S: TrainingSummary>(summary: &S, key: &str) -> Option<Member> {
    Some(match key {
        "objectiveHistory" => list(summary.objective_history()),
        "totalIterations" => num(summary.total_iterations()),
        _ => return None,
    })
}

/// The members of the logistic regression summaries (`_LogisticRegressionSummary`).
fn logistic<S: LogisticRegressionSummaryApi>(summary: &S, key: &str) -> Option<Member> {
    Some(match key {
        "featuresCol" => num(summary.features_col()),
        "probabilityCol" => num(summary.probability_col()),
        _ => return None,
    })
}

/// The DataFrame-valued members, which the e2e test checks against the server's rows, and the
/// Python-only error of a multiclass summary's `roc`, which its type does not have.
const SKIPPED_MEMBERS: &[&str] = &[
    "fMeasureByThreshold",
    "pr",
    "precisionByThreshold",
    "predictions",
    "recallByThreshold",
    "roc",
    "roc-on-multiclass",
];

/// Check every golden member of a summary: its commands, then its value, or its error. The
/// dispatch returns None for a member the summary does not have, which is a mistake here.
fn check_summary_members(
    server: &Server,
    members: &Value,
    dispatch: &dyn Fn(&str) -> Option<Member>,
    label: &str,
) {
    for (key, entry) in members.as_object().unwrap() {
        if key.starts_with("repr")
            || key.starts_with("str")
            || SKIPPED_MEMBERS.contains(&key.as_str())
        {
            continue;
        }
        server.expect(entry);
        let Some(member) = dispatch(key) else {
            panic!("{label}: no accessor for {key}");
        };
        check_member(server, entry, member, &format!("{label}.{key}"));
    }
}

/// Check one member of a golden entry against its actual value or error.
fn check_member(server: &Server, entry: &Value, actual: Member, label: &str) {
    match entry.get("error") {
        Some(error) => {
            let message = error["message"].as_str().unwrap();
            let error = actual.expect_err(&format!("{label}: expected an error"));
            assert!(
                error.to_string().contains(message),
                "{label}: {error} does not carry {message}"
            );
        }
        None => assert_close(
            &actual.unwrap_or_else(|error| panic!("{label}: {error}")),
            &entry["value"],
            label,
        ),
    }
    server.assert_consumed(label);
}

/// The summary class name of a golden entry, as PySpark names it.
fn class_of(value: &Value) -> &str {
    value["type"].as_str().unwrap()
}

/// The model's members of a golden entry, for a model of the given family.
fn check_model_members(
    server: &Server,
    members: &Value,
    model: &dyn Fn(&str) -> Option<Member>,
    label: &str,
) {
    for (key, entry) in members.as_object().unwrap() {
        if key.starts_with("repr") || key.starts_with("str") || key == "transform" {
            continue;
        }
        if key == "summary" || key == "evaluate" || key.starts_with("summary") {
            continue;
        }
        server.expect(entry);
        let Some(member) = model(key) else {
            panic!("{label}: no accessor for {key}");
        };
        check_member(server, entry, member, &format!("{label}.{key}"));
    }
}

/// The LogisticRegression model's members, of the golden's names.
fn logistic_model_member(model: &LogisticRegressionModel, key: &str) -> Option<Member> {
    if key.starts_with("predictRaw-") {
        return Some(match model.predict_raw(&feature_vector(key)) {
            Ok(v) => Ok(vector_json(&v)),
            Err(e) => Err(e),
        });
    }
    if key.starts_with("predictProbability-") {
        return Some(match model.predict_probability(&feature_vector(key)) {
            Ok(v) => Ok(vector_json(&v)),
            Err(e) => Err(e),
        });
    }
    if key.starts_with("predict-") {
        return Some(num(model.predict(&feature_vector(key))));
    }
    Some(match key {
        "coefficients" => model.coefficients().map(|v| vector_json(&v)),
        "coefficientMatrix" => model.coefficient_matrix().map(|m| matrix_json(&m)),
        "intercept" => num(model.intercept()),
        "interceptVector" => model.intercept_vector().map(|v| vector_json(&v)),
        "numClasses" => num(model.num_classes()),
        "numFeatures" => num(model.num_features()),
        "hasSummary" => Ok(json!(model.has_summary())),
        _ => return None,
    })
}

/// The LinearSVC model's members, of the golden's names.
fn linear_svc_model_member(model: &LinearSVCModel, key: &str) -> Option<Member> {
    if key.starts_with("predictRaw-") {
        return Some(
            model
                .predict_raw(&feature_vector(key))
                .map(|v| vector_json(&v)),
        );
    }
    if key.starts_with("predict-") {
        return Some(num(model.predict(&feature_vector(key))));
    }
    Some(match key {
        "coefficients" => model.coefficients().map(|v| vector_json(&v)),
        "intercept" => num(model.intercept()),
        "numClasses" => num(model.num_classes()),
        "numFeatures" => num(model.num_features()),
        "hasSummary" => Ok(json!(model.has_summary())),
        _ => return None,
    })
}

/// Fit a LogisticRegression on a dataset of the golden's fit, with the fit's own params, which
/// the golden fit sent. Returns the model.
fn fit_logistic(server: &Server, fit: &Value) -> LogisticRegressionModel {
    let fit_entry = &fit["fit"];
    let command = proto::MlCommand::decode(
        STANDARD
            .decode(fit_entry["commands"][0].as_str().unwrap())
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    let Some(proto::ml_command::Command::Fit(sent)) = command.command else {
        panic!("the golden fit is not a Fit");
    };
    let mut estimator = LogisticRegression::new();
    *estimator.params_mut() = Params::from_proto(&sent.params.unwrap());
    let session = server.session();
    let train = session.range(1).unwrap();
    server.expect(fit_entry);
    let model = estimator.fit(&train).unwrap();
    server.assert_consumed("fit");
    model
}

fn fit_linear_svc(server: &Server, fit: &Value) -> LinearSVCModel {
    let fit_entry = &fit["fit"];
    let command = proto::MlCommand::decode(
        STANDARD
            .decode(fit_entry["commands"][0].as_str().unwrap())
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    let Some(proto::ml_command::Command::Fit(sent)) = command.command else {
        panic!("the golden fit is not a Fit");
    };
    let mut estimator = LinearSVC::new();
    *estimator.params_mut() = Params::from_proto(&sent.params.unwrap());
    let session = server.session();
    let train = session.range(1).unwrap();
    server.expect(fit_entry);
    let model = estimator.fit(&train).unwrap();
    server.assert_consumed("fit");
    model
}

/// The members of a LogisticRegression model's golden entry, checked on a fitted model. The
/// model's evaluate and summary are checked with their own summaries.
fn check_logistic_model(server: &Server, models: &Value, family: &str) {
    let entry = &models["models"][family];
    let model = fit_logistic(server, &models["fits"][family]);
    check_model_members(
        server,
        &entry["members"],
        &|key| logistic_model_member(&model, key),
        family,
    );
    // The summary of a fit: a Binary or a multiclass summary, as its numClasses decides.
    let summary = &entry["members"]["summary"];
    let fitted = model.summary().unwrap();
    match (&fitted, class_of(&summary["value"])) {
        (LogisticRegressionModelSummary::Binary(_), "BinaryLogisticRegressionTrainingSummary") => {}
        (LogisticRegressionModelSummary::Multiclass(_), "LogisticRegressionTrainingSummary") => {}
        (_, other) => panic!("{family}: summary is not {other}"),
    }
    let training = golden_summary_members(&models_summaries(), family, "training");
    match fitted {
        LogisticRegressionModelSummary::Binary(s) => {
            check_summary_members(
                server,
                &training,
                &|key| binary_training_member(&s, key),
                family,
            );
        }
        LogisticRegressionModelSummary::Multiclass(s) => {
            check_summary_members(
                server,
                &training,
                &|key| multiclass_training_member(&s, key),
                family,
            );
        }
    }
    // The evaluate of the model on a dataset: its own summary, binary or multiclass.
    let evaluate = &entry["members"]["evaluate"];
    let golden = golden_summary_members(&models_summaries(), family, "evaluate");
    let session = server.session();
    let eval_df = session.range(1).unwrap();
    server.expect(evaluate);
    let evaluated = model.evaluate(&eval_df).unwrap();
    server.assert_consumed("evaluate");
    match evaluated {
        LogisticRegressionEvaluation::Binary(s) => {
            assert_eq!(
                class_of(&evaluate["value"]),
                "BinaryLogisticRegressionSummary"
            );
            check_summary_members(
                server,
                &golden,
                &|key| binary_evaluate_member(&s, key),
                family,
            );
        }
        LogisticRegressionEvaluation::Multiclass(s) => {
            assert_eq!(class_of(&evaluate["value"]), "LogisticRegressionSummary");
            check_summary_members(
                server,
                &golden,
                &|key| multiclass_evaluate_member(&s, key),
                family,
            );
        }
    }
}

fn models_summaries() -> Value {
    golden(SUMMARIES)["summaries"].clone()
}

/// The golden members of a summary of a family: `training` or `evaluate`.
fn golden_summary_members(summaries: &Value, family: &str, kind: &str) -> Value {
    summaries[format!("{family}.{kind}").as_str()]["members"].clone()
}

fn binary_training_member(
    s: &BinaryLogisticRegressionTrainingSummary,
    key: &str,
) -> Option<Member> {
    classification(s, key)
        .or_else(|| binary(s, key))
        .or_else(|| logistic(s, key))
        .or_else(|| training(s, key))
}

fn multiclass_training_member(s: &LogisticRegressionTrainingSummary, key: &str) -> Option<Member> {
    classification(s, key)
        .or_else(|| logistic(s, key))
        .or_else(|| training(s, key))
}

fn binary_evaluate_member(s: &BinaryLogisticRegressionSummary, key: &str) -> Option<Member> {
    classification(s, key)
        .or_else(|| binary(s, key))
        .or_else(|| logistic(s, key))
}

fn multiclass_evaluate_member(s: &LogisticRegressionSummary, key: &str) -> Option<Member> {
    classification(s, key).or_else(|| logistic(s, key))
}

#[test]
fn logistic_regression_binomial_model_matches_official_pyspark() {
    let models = golden(MODELS);
    let server = Server::start();
    check_logistic_model(&server, &models, "LogisticRegression.binomial");
}

#[test]
fn logistic_regression_multinomial_model_matches_official_pyspark() {
    let models = golden(MODELS);
    let server = Server::start();
    check_logistic_model(&server, &models, "LogisticRegression.multinomial");
}

#[test]
fn linear_svc_model_matches_official_pyspark() {
    let models = golden(MODELS);
    let server = Server::start();
    let family = "LinearSVC";
    let entry = &models["models"][family];
    let model = fit_linear_svc(&server, &models["fits"][family]);
    check_model_members(
        &server,
        &entry["members"],
        &|key| linear_svc_model_member(&model, key),
        family,
    );
    let summaries = models_summaries();
    assert_eq!(
        class_of(&entry["members"]["summary()"]["value"]),
        "LinearSVCTrainingSummary"
    );
    let training = model.summary().unwrap();
    let golden_training = golden_summary_members(&summaries, family, "training");
    check_summary_members(
        &server,
        &golden_training,
        &|key| linear_svc_training_member(&training, key),
        family,
    );
    let evaluate = &entry["members"]["evaluate"];
    let golden_evaluate = golden_summary_members(&summaries, family, "evaluate");
    let session = server.session();
    let eval_df = session.range(1).unwrap();
    server.expect(evaluate);
    let evaluated = model.evaluate(&eval_df).unwrap();
    server.assert_consumed("evaluate");
    assert_eq!(class_of(&evaluate["value"]), "LinearSVCSummary");
    check_summary_members(
        &server,
        &golden_evaluate,
        &|key| linear_svc_evaluate_member(&evaluated, key),
        family,
    );
}

fn linear_svc_training_member(s: &LinearSVCTrainingSummary, key: &str) -> Option<Member> {
    classification(s, key)
        .or_else(|| binary(s, key))
        .or_else(|| training(s, key))
}

fn linear_svc_evaluate_member(s: &LinearSVCSummary, key: &str) -> Option<Member> {
    classification(s, key).or_else(|| binary(s, key))
}

/// The persistence of an estimator and of its model: `load`, then `save` of the loaded object,
/// as PySpark's persistence entries record them.
#[test]
fn logistic_regression_persistence_matches_official_pyspark() {
    let persistence = golden(PERSISTENCE)["persistence"].clone();
    for family in [
        "LogisticRegression.binomial",
        "LogisticRegression.multinomial",
    ] {
        let server = Server::start();
        let session = server.session();
        let estimator_entry = &persistence[format!("{family}.estimator").as_str()];
        server.expect(&estimator_entry["load"]);
        let estimator = LogisticRegression::load(&session, "/unused/path").unwrap();
        server.assert_consumed(family);
        assert_eq!(
            estimator.operator().uid,
            result_uid(&estimator_entry["load"])
        );
        assert_eq!(
            Estimator::params(&estimator).to_proto(),
            params_proto(&estimator_entry["load"])
        );
        server.expect(&estimator_entry["save"]);
        estimator
            .write()
            .session(&session)
            .overwrite()
            .save("/unused/path")
            .unwrap();
        server.assert_consumed(family);

        let model_entry = &persistence[format!("{family}.model").as_str()];
        server.expect(&model_entry["load"]);
        let model = LogisticRegressionModel::load(&session, "/unused/path").unwrap();
        server.assert_consumed(family);
        assert_eq!(model.operator().uid, result_uid(&model_entry["load"]));
        assert!(!model.has_summary());
        check_model_members(
            &server,
            &model_entry["members"]["members"],
            &|key| logistic_model_member(&model, key),
            family,
        );
        server.expect(&model_entry["save"]);
        model
            .write()
            .unwrap()
            .overwrite()
            .save("/unused/path")
            .unwrap();
        server.assert_consumed(family);
    }
}

#[test]
fn linear_svc_persistence_matches_official_pyspark() {
    let persistence = golden(PERSISTENCE)["persistence"].clone();
    let family = "LinearSVC";
    let server = Server::start();
    let session = server.session();
    let estimator_entry = &persistence[format!("{family}.estimator").as_str()];
    server.expect(&estimator_entry["load"]);
    let estimator = LinearSVC::load(&session, "/unused/path").unwrap();
    server.assert_consumed(family);
    assert_eq!(
        estimator.operator().uid,
        result_uid(&estimator_entry["load"])
    );
    server.expect(&estimator_entry["save"]);
    estimator
        .write()
        .session(&session)
        .overwrite()
        .save("/unused/path")
        .unwrap();
    server.assert_consumed(family);

    let model_entry = &persistence[format!("{family}.model").as_str()];
    server.expect(&model_entry["load"]);
    let model = LinearSVCModel::load(&session, "/unused/path").unwrap();
    server.assert_consumed(family);
    assert_eq!(model.operator().uid, result_uid(&model_entry["load"]));
    check_model_members(
        &server,
        &model_entry["members"]["members"],
        &|key| linear_svc_model_member(&model, key),
        family,
    );
    server.expect(&model_entry["save"]);
    model
        .write()
        .unwrap()
        .overwrite()
        .save("/unused/path")
        .unwrap();
    server.assert_consumed(family);
}

/// The uid that a golden load read: the golden normalizes it to a placeholder, so the loaded
/// uid is the placeholder too.
fn result_uid(load: &Value) -> String {
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

/// The params that a golden load of an operator read: the `MlParams` of its operator info.
fn params_proto(load: &Value) -> proto::MlParams {
    let result = proto::MlCommandResult::decode(
        STANDARD
            .decode(load["results"][0].as_str().unwrap())
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    match result.result_type {
        Some(proto::ml_command_result::ResultType::OperatorInfo(info)) => info.params.unwrap(),
        other => panic!("not an operator info: {other:?}"),
    }
}

/// The setters of the models set the params of their transform, as PySpark's `_set` does.
#[test]
fn model_setters_set_the_params_of_the_transform() {
    use spark_connect_proto::expression::literal::LiteralType;
    let persistence = golden(PERSISTENCE)["persistence"].clone();
    let server = Server::start();
    let session = server.session();
    let entry = &persistence["LogisticRegression.binomial.model"];
    server.expect(&entry["load"]);
    let model = LogisticRegressionModel::load(&session, "/unused/path").unwrap();
    server.assert_consumed("load");
    let model = model
        .set_features_col("x")
        .set_prediction_col("p")
        .set_probability_col("q")
        .set_raw_prediction_col("r")
        .set_threshold(0.7)
        .set_thresholds([0.4, 0.6]);
    let params = Transformer::params(&model).to_proto();
    let literal = |name: &str| params.params[name].literal_type.clone();
    assert_eq!(
        literal("featuresCol"),
        Some(LiteralType::String("x".into()))
    );
    assert_eq!(
        literal("predictionCol"),
        Some(LiteralType::String("p".into()))
    );
    assert_eq!(
        literal("probabilityCol"),
        Some(LiteralType::String("q".into()))
    );
    assert_eq!(
        literal("rawPredictionCol"),
        Some(LiteralType::String("r".into()))
    );
    assert_eq!(literal("threshold"), Some(LiteralType::Double(0.7)));
    let Some(LiteralType::Array(thresholds)) = literal("thresholds") else {
        panic!("thresholds is not an array");
    };
    assert_eq!(thresholds.elements.len(), 2);

    let entry = &persistence["LinearSVC.model"];
    server.expect(&entry["load"]);
    let svc = LinearSVCModel::load(&session, "/unused/path").unwrap();
    server.assert_consumed("load");
    let svc = svc
        .set_features_col("x")
        .set_prediction_col("p")
        .set_raw_prediction_col("r")
        .set_threshold(0.7);
    let params = Transformer::params(&svc).to_proto();
    let literal = |name: &str| params.params[name].literal_type.clone();
    assert_eq!(
        literal("featuresCol"),
        Some(LiteralType::String("x".into()))
    );
    assert_eq!(
        literal("predictionCol"),
        Some(LiteralType::String("p".into()))
    );
    assert_eq!(
        literal("rawPredictionCol"),
        Some(LiteralType::String("r".into()))
    );
    assert_eq!(literal("threshold"), Some(LiteralType::Double(0.7)));
}
