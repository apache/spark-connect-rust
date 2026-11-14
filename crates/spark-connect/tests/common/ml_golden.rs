// SPDX-License-Identifier: Apache-2.0
//! The ML goldens of scripts/ml_golden_common.py, for the live tests that compare the Rust client
//! with them: the datasets of their specs, the values of the Rust client as the helper renders
//! those of official PySpark, and their comparison. A test file includes it with
//! `#[path = "common/ml_golden.rs"] mod ml_golden;`.
//!
//! The same goldens replayed offline, with no Spark server: [`ReplayServer::start`] serves a
//! loopback Spark Connect service that answers each ML command of the golden entries it is given
//! with the entry's result (or error), in the order that the client sends them, and records the
//! ML commands that the client sends. [`assert_commands`] then checks the recorded commands of a
//! call against an entry's `commands`, after [`normalize_commands`] has numbered their object ids
//! and uids per entry, as the golden does. An entry is an object `{"commands", "results",
//! "error"?}`; an object without `commands` registers the entries among its values, so a model
//! entry registers its members. The replay serves any area's golden of that format. The entry of
//! a DataFrame member has an MlRelation as its command, which the client sends no ML command for:
//! such entries are not replayed (see [`ReplayServer::start`]).
#![allow(dead_code)] // every test binary compiles this module and uses only a part of it

use std::collections::{BTreeSet, HashMap};
use std::convert::Infallible;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use base64::{engine::general_purpose::STANDARD, Engine};
use futures::stream;
use prost::Message;
use serde_json::{json, Value as Json};
use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::ml::{MlMatrix, MlValue, MlVector, Params};
use spark_connect::row::Value;
use spark_connect::{Column, DataFrame, SparkSession};
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_proto as proto;
use tokio::net::TcpListener;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::server::{NamedService, ServerStreamingService, UnaryService};
use tonic::{Request, Response, Status};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub fn fixture(text: &str) -> Json {
    serde_json::from_str(text).unwrap()
}

/// The DataFrame of a dataset spec of `Golden.register`: its SQL; its vector columns, each of an
/// array column (`"vectors": [column, ...]` in place, or `{"column": "source", ...}`); and its
/// partitions (`"coalesce"`).
pub fn dataset(session: &SparkSession, spec: &Json) -> DataFrame {
    let mut frame = session.sql(spec["sql"].as_str().unwrap()).unwrap();
    let vectors: Vec<(String, String)> = match &spec["vectors"] {
        Json::Array(columns) => columns
            .iter()
            .map(|column| {
                let column = column.as_str().unwrap().to_string();
                (column.clone(), column)
            })
            .collect(),
        Json::Object(columns) => columns
            .iter()
            .map(|(column, source)| (column.clone(), source.as_str().unwrap().to_string()))
            .collect(),
        Json::Null => Vec::new(),
        other => panic!("the vectors of a dataset spec: {other}"),
    };
    for (column, source) in vectors {
        let source = col(&source).expression().clone();
        let vector = Column::new(Expression::function("array_to_vector", vec![source]));
        frame = frame.with_column(&column, vector);
    }
    match spec["coalesce"].as_i64() {
        Some(partitions) => frame.coalesce(partitions as i32),
        None => frame,
    }
}

// The values as scripts/ml_golden_common.py renders those of official PySpark.

pub fn num(x: f64) -> Json {
    if x.is_nan() {
        json!("NaN")
    } else if x == f64::INFINITY {
        json!("Infinity")
    } else if x == f64::NEG_INFINITY {
        json!("-Infinity")
    } else {
        json!(x)
    }
}

pub fn nums(values: &[f64]) -> Json {
    values.iter().map(|x| num(*x)).collect()
}

pub fn vector(v: &MlVector) -> Json {
    match v {
        MlVector::Dense(values) => {
            json!({"type": "dense", "size": values.len(), "values": nums(values)})
        }
        MlVector::Sparse {
            size,
            indices,
            values,
        } => json!({"type": "sparse", "size": size, "indices": indices, "values": nums(values)}),
    }
}

pub fn matrix(m: &MlMatrix) -> Json {
    match m {
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
            "values": nums(values),
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
            "values": nums(values),
        }),
    }
}

/// A collected value: a vector as `vector` renders it, a struct as the list of its values, as
/// PySpark renders a Row.
pub fn value(v: &Value) -> Json {
    match v {
        Value::Null => Json::Null,
        Value::Bool(b) => json!(b),
        Value::Byte(x) => json!(x),
        Value::Short(x) => json!(x),
        Value::Integer(x) => json!(x),
        Value::Long(x) => json!(x),
        Value::Float(x) => num(f64::from(*x)),
        Value::Double(x) => num(*x),
        Value::String(s) => json!(s),
        Value::List(items) => items.iter().map(value).collect(),
        Value::Struct(row) if row.fields() == ["type", "size", "indices", "values"] => {
            // A VectorUDT value: of type 1 dense, of only its values; of type 0 sparse.
            let field = |name| row.get_by_name(name).unwrap();
            let values = value(field("values"));
            if matches!(field("type"), Value::Byte(1)) {
                let size = values.as_array().unwrap().len();
                json!({"type": "dense", "size": size, "values": values})
            } else {
                json!({
                    "type": "sparse",
                    "size": value(field("size")),
                    "indices": value(field("indices")),
                    "values": values,
                })
            }
        }
        Value::Struct(row) => row.values().iter().map(value).collect(),
        other => panic!("no rendering of {other:?}"),
    }
}

pub fn frame(df: &DataFrame) -> Result<Json> {
    let schema = df.schema()?.simple_string();
    let rows: Vec<Json> = df
        .collect()?
        .iter()
        .map(|row| row.values().iter().map(value).collect())
        .collect();
    Ok(json!({"schema": schema, "rows": rows}))
}

/// A value to compare: the rows of a DataFrame in the order of their JSON, as the golden sorts
/// them.
pub fn normalized(mut v: Json) -> Json {
    if let Some(rows) = v.get_mut("rows").and_then(Json::as_array_mut) {
        rows.sort_by_key(|row| row.to_string());
    }
    v
}

/// Whether `actual` matches the golden's `expected`: numbers within a tolerance, arrays elementwise
/// and objects by their keys. The server computes some values in single precision (the factors of
/// ALS, the vectors and similarities of Word2Vec), whose last bits differ between the platforms of
/// the capture and of the test, and which iterations accumulate: the tolerance is that of
/// `e2e_als.rs`, which compares ALS of official PySpark across platforms.
pub fn assert_matches(actual: &Json, expected: &Json, at: &str) {
    match (actual, expected) {
        (Json::Number(a), Json::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!((a - e).abs() <= 2e-5 + 2e-4 * e.abs(), "{at}: {a} != {e}");
        }
        (Json::Array(a), Json::Array(e)) => {
            assert_eq!(a.len(), e.len(), "{at}: {actual} != {expected}");
            for (i, (a, e)) in a.iter().zip(e).enumerate() {
                assert_matches(a, e, &format!("{at}[{i}]"));
            }
        }
        (Json::Object(a), Json::Object(e)) => {
            let keys =
                |m: &serde_json::Map<String, Json>| m.keys().cloned().collect::<BTreeSet<_>>();
            assert_eq!(keys(a), keys(e), "{at}: {actual} != {expected}");
            for (key, e) in e {
                assert_matches(&a[key], e, &format!("{at}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{at}"),
    }
}

/// The error of official PySpark is in the message of the Rust one. A JVM exception that the server
/// does not classify reaches PySpark as an `UnknownException` of "(<class>) <message>".
pub fn assert_error(actual: &str, expected: &Json, at: &str) {
    let message = expected["message"].as_str().unwrap();
    let message = match message.split_once(") ") {
        Some((class, rest)) if class.starts_with("(java.") => rest,
        _ => message,
    };
    assert!(
        actual.contains(message),
        "{at}: the error {actual:?} does not carry {message:?}"
    );
}

/// The params that `loaded` has of each param that `saved` sets, by their values, as PySpark reads
/// them of their literals: an integer of any width (the server reads a seed back as a long), and
/// another value by its Debug text, which compares a NaN too.
pub fn assert_params(loaded: &Params, saved: &Params, at: &str) {
    fn canonical(value: &MlValue) -> String {
        match value {
            MlValue::Array(items) => {
                let items: Vec<String> = items.iter().map(canonical).collect();
                format!("[{}]", items.join(", "))
            }
            value => value
                .as_i64()
                .map_or_else(|| format!("{value:?}"), |i| i.to_string()),
        }
    }
    fn value(literal: &spark_connect_proto::expression::Literal) -> String {
        canonical(&MlValue::from_literal(literal).unwrap())
    }
    let loaded = loaded.to_proto().params;
    for (name, literal) in saved.to_proto().params {
        assert_eq!(
            loaded.get(&name).map(value),
            Some(value(&literal)),
            "{at}: param {name}"
        );
    }
}

// The offline replay of the goldens.

/// The name of the Spark Connect service in the paths of its RPCs.
const SERVICE: &str = "/spark.connect.SparkConnectService/";

/// The reply of the replay to a golden command.
#[derive(Clone)]
enum Reply {
    /// The golden result, as the server sends it.
    Result(proto::MlCommandResult),
    /// The golden error, by its message.
    Error(String),
    /// No result: the command completes with no result of its own.
    Empty,
}

/// A golden command, normalized on its own, with its reply.
struct Exchange {
    command: proto::MlCommand,
    reply: Reply,
    /// Whether a request has been answered with it. A request takes the first unused match.
    used: bool,
}

/// The golden exchanges of a replay, and the ML commands that the client has sent.
#[derive(Default)]
struct Replay {
    exchanges: Vec<Exchange>,
    /// The ML commands sent since they were last taken, as they were sent.
    received: Vec<proto::MlCommand>,
}

type Shared = Arc<Mutex<Replay>>;

/// A loopback Spark Connect service that replays golden entries. See the module doc.
pub struct ReplayServer(Shared);

impl ReplayServer {
    /// Serves the golden entries `entries` on a loopback port, and returns the server with a
    /// session of it. The session stays with the server's own thread for the rest of the test.
    /// The relation entries (see [`golden_commands`]) are not registered: they are not replayed.
    pub fn start(entries: &[&Json]) -> (Self, SparkSession) {
        let mut replay = Replay::default();
        for entry in entries {
            register(&mut replay, entry);
        }
        let shared = Arc::new(Mutex::new(replay));
        let service = ReplayService(shared.clone());
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
        (Self(shared), session)
    }

    /// The ML commands that the client sent since they were last taken, in order, as they were
    /// sent. Taking them clears them. A delete of a released object is not recorded.
    pub fn take_requests(&self) -> Vec<proto::MlCommand> {
        std::mem::take(&mut self.0.lock().unwrap().received)
    }
}

/// Asserts that the ML commands that `server` received since they were last taken are the
/// `commands` of the golden `entry`, normalized per entry as the golden is (see
/// [`normalize_commands`]). The commands are taken, so the next check sees only later ones.
/// `at` labels a failure.
pub fn assert_commands(server: &ReplayServer, entry: &Json, at: &str) {
    let mut actual = server.take_requests();
    let mut expected = golden_commands(entry);
    normalize_commands(&mut actual);
    normalize_commands(&mut expected);
    if actual != expected {
        panic!(
            "{at}: the ML commands differ from the golden\n{}",
            first_difference(&expected, &actual)
        );
    }
}

/// The `commands` of a golden entry, decoded as they were sent, in order. A relation entry (the
/// entry of a DataFrame member, whose command is an MlRelation) has no ML commands to check: it
/// panics with a message saying so, since such entries are not replayed.
pub fn golden_commands(entry: &Json) -> Vec<proto::MlCommand> {
    let commands = entry["commands"]
        .as_array()
        .unwrap_or_else(|| panic!("a golden entry without commands: {entry}"));
    if commands.iter().any(is_relation) {
        panic!(
            "a golden entry whose command is an MlRelation (the entry of a DataFrame member): \
             relation entries are not replayed, so its commands cannot be checked"
        );
    }
    commands.iter().map(decode_command).collect()
}

/// Normalizes the ML commands of one golden entry in place, as the golden normalizes them. The
/// object ids and the uids are renumbered `<id0>`, `<id1>`, ... and `<uid0>`, ... in order of
/// first appearance across `commands`, so a run's own ids and uids compare with the golden's
/// whatever they are. A uid inside a string value (a model's uid, given to a helper fetch) is
/// renumbered with the same counter, as the golden's Python normalizer does. A save path becomes
/// `<path>`. A NaN value becomes [`NAN_MARKER`]. The datasets of fits, evaluations and summaries,
/// and the inputs of fetches, are cleared: the replay checks the operators, the methods and the
/// params, not the data.
pub fn normalize_commands(commands: &mut [proto::MlCommand]) {
    let mut names = Placeholders::default();
    for command in commands {
        normalize_command(command, &mut names);
    }
}

/// What a NaN value is in a normalized command. NaN is not equal to itself, so a command with a
/// NaN param (an Imputer's `missingValue`) would never equal its golden's; both sides hold the
/// marker instead.
const NAN_MARKER: &str = "<NaN>";

/// The placeholders of a sequence of commands, numbered in order of first appearance.
#[derive(Default)]
struct Placeholders {
    ids: HashMap<String, String>,
    uids: HashMap<String, String>,
}

impl Placeholders {
    fn id(&mut self, raw: &mut String) {
        rename(&mut self.ids, "id", raw);
    }

    fn uid(&mut self, raw: &mut String) {
        rename(&mut self.uids, "uid", raw);
    }

    /// A string value: its random uids (`Name_<12 hex>`) become uid placeholders.
    fn text(&mut self, text: &mut String) {
        *text = scrub_random_uids(text, &mut self.uids);
    }
}

/// The text with each random uid of it, `[A-Za-z]+_[0-9a-f]{12}` as scripts/ml_golden_common.py
/// matches them, replaced by its uid placeholder, numbered in `uids` as the uid fields are. The
/// matches are found from the left, as a regex substitution finds them.
fn scrub_random_uids(text: &str, uids: &mut HashMap<String, String>) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    // The bytes of `text` before `copied` are in `out`.
    let mut copied = 0;
    let mut at = 0;
    while at < bytes.len() {
        if !bytes[at].is_ascii_alphabetic() {
            at += 1;
            continue;
        }
        // A run of letters, which only an underscore can end a uid's name with.
        let start = at;
        while at < bytes.len() && bytes[at].is_ascii_alphabetic() {
            at += 1;
        }
        let hex = bytes.get(at + 1..at + 13);
        let is_uid = bytes.get(at) == Some(&b'_')
            && hex.is_some_and(|hex| {
                hex.iter()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
            });
        if is_uid {
            let end = at + 13;
            let mut placeholder = text[start..end].to_string();
            rename(uids, "uid", &mut placeholder);
            out.push_str(&text[copied..start]);
            out.push_str(&placeholder);
            copied = end;
            at = end;
        }
    }
    out.push_str(&text[copied..]);
    out
}

/// The values of a map in the order of its keys, so that the placeholders of a map's values are
/// numbered the same whatever the order of the map's storage.
fn values_in_key_order<V>(map: &mut HashMap<String, V>) -> Vec<&mut V> {
    let mut entries: Vec<(&String, &mut V)> = map.iter_mut().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries.into_iter().map(|(_, value)| value).collect()
}

/// Replaces `raw` with the placeholder of its kind, numbering a new one at the next index. An
/// empty string is no object and stays.
fn rename(table: &mut HashMap<String, String>, kind: &str, raw: &mut String) {
    if raw.is_empty() {
        return;
    }
    let next = format!("<{kind}{}>", table.len());
    *raw = table.entry(raw.clone()).or_insert(next).clone();
}

fn clear_path(path: &mut String) {
    if !path.is_empty() {
        *path = "<path>".into();
    }
}

/// An operator: its name, then its uid.
fn normalize_operator(operator: &mut proto::MlOperator, names: &mut Placeholders) {
    names.text(&mut operator.name);
    names.uid(&mut operator.uid);
}

/// The params of a command, in the order of their keys: each literal is normalized.
fn normalize_params(params: Option<&mut proto::MlParams>, names: &mut Placeholders) {
    if let Some(params) = params {
        for literal in values_in_key_order(&mut params.params) {
            normalize_literal(literal, names);
        }
    }
}

/// A literal: its strings are normalized, and a NaN value becomes [`NAN_MARKER`]. A specialized
/// array of doubles or floats that has a NaN is first expanded to an array literal of its
/// elements, so that its NaN is an element like those of an array literal.
fn normalize_literal(literal: &mut proto::expression::Literal, names: &mut Placeholders) {
    use proto::expression::literal::{Array, LiteralType, SpecializedArray};
    if let Some(elements) = nan_array_elements(literal.literal_type.as_ref()) {
        literal.literal_type = Some(LiteralType::Array(Array {
            elements,
            ..Default::default()
        }));
    }
    if is_nan(literal.literal_type.as_ref()) {
        literal.literal_type = Some(LiteralType::String(NAN_MARKER.into()));
        return;
    }
    match literal.literal_type.as_mut() {
        Some(LiteralType::String(text)) => names.text(text),
        Some(LiteralType::Array(array)) => {
            for element in &mut array.elements {
                normalize_literal(element, names);
            }
        }
        Some(LiteralType::Map(map)) => {
            for key in &mut map.keys {
                normalize_literal(key, names);
            }
            for value in &mut map.values {
                normalize_literal(value, names);
            }
        }
        Some(LiteralType::Struct(row)) => {
            for element in &mut row.elements {
                normalize_literal(element, names);
            }
        }
        Some(LiteralType::SpecializedArray(SpecializedArray {
            value_type:
                Some(proto::expression::literal::specialized_array::ValueType::Strings(strings)),
        })) => {
            for value in &mut strings.values {
                names.text(value);
            }
        }
        _ => {}
    }
}

/// Whether a literal is a NaN double or float.
fn is_nan(literal_type: Option<&proto::expression::literal::LiteralType>) -> bool {
    use proto::expression::literal::LiteralType;
    matches!(literal_type, Some(LiteralType::Double(x)) if x.is_nan())
        || matches!(literal_type, Some(LiteralType::Float(x)) if x.is_nan())
}

/// The elements of a specialized array of doubles or floats that has a NaN, as the literals of
/// its values; None for any other literal.
fn nan_array_elements(
    literal_type: Option<&proto::expression::literal::LiteralType>,
) -> Option<Vec<proto::expression::Literal>> {
    use proto::expression::literal::{specialized_array::ValueType, LiteralType, SpecializedArray};
    let element = |literal_type| proto::expression::Literal {
        literal_type: Some(literal_type),
        ..Default::default()
    };
    let Some(LiteralType::SpecializedArray(SpecializedArray {
        value_type: Some(value_type),
    })) = literal_type
    else {
        return None;
    };
    match value_type {
        ValueType::Doubles(doubles) if doubles.values.iter().any(|x| x.is_nan()) => Some(
            doubles
                .values
                .iter()
                .map(|x| element(LiteralType::Double(*x)))
                .collect(),
        ),
        ValueType::Floats(floats) if floats.values.iter().any(|x| x.is_nan()) => Some(
            floats
                .values
                .iter()
                .map(|x| element(LiteralType::Float(*x)))
                .collect(),
        ),
        _ => None,
    }
}

fn normalize_command(command: &mut proto::MlCommand, names: &mut Placeholders) {
    use proto::fetch::method::args::ArgsType;
    use proto::ml_command::{write::Type as WriteType, Command};
    match command.command.as_mut() {
        Some(Command::Fit(fit)) => {
            if let Some(estimator) = fit.estimator.as_mut() {
                normalize_operator(estimator, names);
            }
            normalize_params(fit.params.as_mut(), names);
            fit.dataset = None;
        }
        Some(Command::Evaluate(evaluate)) => {
            if let Some(evaluator) = evaluate.evaluator.as_mut() {
                normalize_operator(evaluator, names);
            }
            normalize_params(evaluate.params.as_mut(), names);
            evaluate.dataset = None;
        }
        Some(Command::CreateSummary(summary)) => {
            if let Some(model) = summary.model_ref.as_mut() {
                names.id(&mut model.id);
            }
            summary.dataset = None;
        }
        Some(Command::GetModelSize(size)) => {
            if let Some(model) = size.model_ref.as_mut() {
                names.id(&mut model.id);
            }
        }
        Some(Command::Fetch(fetch)) => {
            if let Some(object) = fetch.obj_ref.as_mut() {
                names.id(&mut object.id);
            }
            for method in &mut fetch.methods {
                names.text(&mut method.method);
                for arg in &mut method.args {
                    match arg.args_type.as_mut() {
                        Some(ArgsType::Param(param)) => normalize_literal(param, names),
                        Some(ArgsType::Input(input)) => *input = proto::Relation::default(),
                        None => {}
                    }
                }
            }
        }
        Some(Command::Delete(delete)) => {
            for object in &mut delete.obj_refs {
                names.id(&mut object.id);
            }
        }
        Some(Command::Read(read)) => {
            if let Some(operator) = read.operator.as_mut() {
                normalize_operator(operator, names);
            }
            clear_path(&mut read.path);
        }
        Some(Command::Write(write)) => {
            match write.r#type.as_mut() {
                Some(WriteType::Operator(operator)) => normalize_operator(operator, names),
                Some(WriteType::ObjRef(object)) => names.id(&mut object.id),
                None => {}
            }
            normalize_params(write.params.as_mut(), names);
            clear_path(&mut write.path);
            for value in values_in_key_order(&mut write.options) {
                names.text(value);
            }
        }
        Some(Command::CleanCache(_) | Command::GetCacheInfo(_)) | None => {}
    }
}

/// Registers a golden entry: each of its commands with its reply. An object without `commands`
/// registers its values that are entries. A relation entry (the entry of a DataFrame member) is
/// not replayed: the client sends no ML command for it, so it registers nothing.
fn register(replay: &mut Replay, entry: &Json) {
    let Some(object) = entry.as_object() else {
        return;
    };
    let Some(commands) = object.get("commands").and_then(Json::as_array) else {
        for child in object.values() {
            register(replay, child);
        }
        return;
    };
    if commands.iter().any(is_relation) {
        return;
    }
    let results = object.get("results").and_then(Json::as_array);
    for (index, command) in commands.iter().enumerate() {
        let mut key = vec![decode_command(command)];
        normalize_commands(&mut key);
        let reply = match (results.and_then(|r| r.get(index)), object.get("error")) {
            (Some(result), _) => Reply::Result(decode_result(result)),
            (None, Some(error)) => Reply::Error(error["message"].as_str().unwrap().to_string()),
            (None, None) => Reply::Empty,
        };
        replay.exchanges.push(Exchange {
            command: key.remove(0),
            reply,
            used: false,
        });
    }
}

fn decode_bytes(b64: &Json) -> Vec<u8> {
    STANDARD
        .decode(b64.as_str().expect("the base64 of a golden message"))
        .expect("the base64 of a golden message")
}

fn decode_command(b64: &Json) -> proto::MlCommand {
    proto::MlCommand::decode(decode_bytes(b64).as_slice()).expect("a golden ML command")
}

/// Whether a golden command is an MlRelation, the plan of a DataFrame that a member returns, rather
/// than an ML command: no MlCommand decodes from it.
fn is_relation(b64: &Json) -> bool {
    proto::MlCommand::decode(decode_bytes(b64).as_slice()).is_err()
}

fn decode_result(b64: &Json) -> proto::MlCommandResult {
    proto::MlCommandResult::decode(decode_bytes(b64).as_slice()).expect("a golden result")
}

/// Answers an ML command from the golden: the first unused exchange whose normalized command
/// equals the command's, normalized on its own, or else the first that equals it. A delete of a
/// released object is not part of the golden: it completes with no result, and is not recorded.
fn answer(
    shared: &Shared,
    command: proto::MlCommand,
) -> std::result::Result<Option<proto::MlCommandResult>, Status> {
    let mut replay = shared.lock().unwrap();
    if matches!(command.command, Some(proto::ml_command::Command::Delete(_))) {
        return Ok(None);
    }
    replay.received.push(command.clone());
    let mut single = vec![command];
    normalize_commands(&mut single);
    let key = single.remove(0);
    let position = replay
        .exchanges
        .iter()
        .position(|e| !e.used && e.command == key)
        .or_else(|| replay.exchanges.iter().position(|e| e.command == key));
    let Some(position) = position else {
        return Err(Status::invalid_argument(format!(
            "the golden has no command {key:?}"
        )));
    };
    let exchange = &mut replay.exchanges[position];
    exchange.used = true;
    match &exchange.reply {
        Reply::Result(result) => Ok(Some(result.clone())),
        Reply::Empty => Ok(None),
        Reply::Error(message) => Err(Status::internal(message.clone())),
    }
}

/// The replay's service: ExecutePlan answers ML commands from the golden, and Config and
/// ReleaseExecute answer as a session needs. Any other RPC is refused.
#[derive(Clone)]
struct ReplayService(Shared);

impl NamedService for ReplayService {
    const NAME: &'static str = "spark.connect.SparkConnectService";
}

impl Service<http::Request<Body>> for ReplayService {
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
                    eprintln!("the golden replay does not answer {other}");
                    grpc.unary(Unsupported, request).await
                }
            })
        })
    }
}

/// ExecutePlan: an ML command of the golden, answered as the golden answers it.
struct Execute(Shared);

impl ServerStreamingService<Vec<u8>> for Execute {
    type Response = Vec<u8>;
    type ResponseStream = stream::Iter<std::vec::IntoIter<std::result::Result<Vec<u8>, Status>>>;
    type Future = BoxFuture<Response<Self::ResponseStream>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let request = proto::ExecutePlanRequest::decode(request.into_inner().as_slice()).unwrap();
        let session_id = request.session_id.clone();
        let operation_id = request.operation_id.clone().unwrap_or_default();
        let command = match request.plan.and_then(|plan| plan.op_type) {
            Some(proto::plan::OpType::Command(command)) => match command.command_type {
                Some(proto::command::CommandType::MlCommand(ml)) => Some(ml),
                _ => None,
            },
            _ => None,
        };
        let reply = match command {
            Some(ml) => answer(&self.0, ml),
            None => Err(Status::unimplemented(
                "the golden replay answers ML commands only",
            )),
        };
        Box::pin(async move {
            let result = reply?;
            let mut frames = Vec::new();
            if let Some(result) = result {
                frames.push(Ok(response(
                    &session_id,
                    &operation_id,
                    proto::execute_plan_response::ResponseType::MlCommandResult(result),
                )
                .encode_to_vec()));
            }
            frames.push(Ok(response(
                &session_id,
                &operation_id,
                proto::execute_plan_response::ResponseType::ResultComplete(
                    proto::execute_plan_response::ResultComplete {},
                ),
            )
            .encode_to_vec()));
            Ok(Response::new(stream::iter(frames)))
        })
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

/// The first line where the pretty-printed `actual` differs from `expected`, with the lines
/// around it.
fn first_difference(expected: &[proto::MlCommand], actual: &[proto::MlCommand]) -> String {
    let golden_text = format!("{expected:#?}");
    let actual_text = format!("{actual:#?}");
    let golden_lines: Vec<&str> = golden_text.lines().collect();
    let actual_lines: Vec<&str> = actual_text.lines().collect();
    let line = golden_lines
        .iter()
        .zip(&actual_lines)
        .position(|(g, a)| g != a)
        .unwrap_or(golden_lines.len().min(actual_lines.len()));
    let window = |lines: &[&str]| {
        lines
            .iter()
            .skip(line.saturating_sub(2))
            .take(8)
            .copied()
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "first difference at line {}\n golden:\n{}\n actual:\n{}",
        line + 1,
        window(&golden_lines),
        window(&actual_lines)
    )
}
