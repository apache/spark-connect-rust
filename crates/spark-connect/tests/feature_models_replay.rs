// SPDX-License-Identifier: Apache-2.0
//! The fitted feature models of official PySpark 4.3.0 (tests/golden/feature_models.json and
//! feature_models_persistence.json), replayed against the typed Rust accessors, with no Spark
//! server. The replay server of `common/ml_golden.rs` answers each ML command with the golden
//! result of that command, and the test checks that the accessor sent the golden commands and
//! returned the golden value (or an error carrying the golden message). Each estimator is
//! configured as the feature parity tests configure it (`common/feature_cases.rs`), so the Fit
//! command of the Rust client is compared with the golden one.
//!
//! Replayed: the fit of every case, the members of its model that are values, the class methods
//! that build a model client-side, and the save and load of every transformer, estimator and
//! model. Skipped, by [`skip_reason`]: the DataFrame-valued members (their rows are Arrow, which
//! the replay does not serve), the members that the Rust client collects from such a DataFrame,
//! and the repr() and str() of the Python objects. The `skipped` lists of the golden are its
//! client-side params and setters, which send no command and are not golden entries.
//!
//! Left out, with the finding that keeps it out (see [`LOAD_LEFT_OUT`]): the load of the
//! StopWordsRemover stages. [`LEFT_OUT`] and [`CLASS_LEFT_OUT`] list the entries and class methods
//! left out whole, of which there are none now. The Delete that PySpark sends for the temporary
//! model of a QuantileDiscretizer's fit is not compared (see [`without_deletes`]).
#[path = "common/feature_cases.rs"]
#[macro_use]
mod feature_cases;
#[path = "common/ml_golden.rs"]
mod ml_golden;

use std::collections::BTreeSet;
use std::fmt::Display;
use std::panic::{catch_unwind, AssertUnwindSafe};

use ml_golden::{
    assert_commands, assert_error, assert_matches, assert_params, fixture, matrix, nums, vector,
    ReplayServer, Result,
};
use serde_json::{json, Value as Json};
use spark_connect::ml::*;
use spark_connect::{DataFrame, SparkSession};
use spark_connect_proto as proto;

const MODELS: &str = include_str!("../../../tests/golden/feature_models.json");
const PERSISTENCE: &str = include_str!("../../../tests/golden/feature_models_persistence.json");

/// The path of every save and load. The replay server does not read it: the golden commands
/// normalize it to `<path>`.
const PATH: &str = "/unused/path";

/// The findings that keep a golden entry out of the replay, by the golden key of the entry: none.
const LEFT_OUT: &[(&str, &str)] = &[];

/// The golden entries whose load is left out of the replay, with their finding. Their saves are
/// replayed.
const LOAD_LEFT_OUT: &[(&str, &str)] = &[
    ("StopWordsRemover", STOP_WORDS),
    ("StopWordsRemover-multi", STOP_WORDS),
];

/// The class methods that are left out of the replay, with their finding: none.
const CLASS_LEFT_OUT: &[(&str, &str)] = &[];

/// Finding: PySpark's StopWordsRemover.load fetches the default stop words after its read
/// (stopWordsRemoverGetDefaultOrUS and stopWordsRemoverLoadDefaultStopWords), which the Rust
/// load does not send. The save is replayed.
const STOP_WORDS: &str =
    "PySpark's load also fetches the default stop words; the Rust load does not";

/// The finding that keeps the golden entry `key` out of the replay, if any.
fn left_out(key: &str) -> Option<&'static str> {
    LEFT_OUT
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, finding)| *finding)
}

/// The finding that keeps the class method `key` out of the replay, if any.
fn class_left_out(key: &str) -> Option<&'static str> {
    CLASS_LEFT_OUT
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, finding)| *finding)
}

/// The finding that keeps the load of the golden entry `key` out of the replay, if any.
fn load_left_out(key: &str) -> Option<&'static str> {
    LOAD_LEFT_OUT
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, finding)| *finding)
}

/// The fitted model of a case, of each model class of official PySpark that a feature estimator
/// fits. QuantileDiscretizer's is a `Bucketizer`, a transformer.
macro_rules! fitted {
    (models: [$($model:ident),* $(,)?], transformers: [$($transformer:ident),* $(,)?] $(,)?) => {
        enum Fitted {
            $($model($model),)*
            $($transformer($transformer),)*
        }
        $(impl From<$model> for Fitted {
            fn from(model: $model) -> Self {
                Self::$model(model)
            }
        })*
        $(impl From<$transformer> for Fitted {
            fn from(model: $transformer) -> Self {
                Self::$transformer(model)
            }
        })*
        impl Fitted {
            fn class(&self) -> &'static str {
                match self {
                    $(Self::$model(_) => stringify!($model),)*
                    $(Self::$transformer(_) => stringify!($transformer),)*
                }
            }

            fn operator(&self) -> &MlOperator {
                match self {
                    $(Self::$model(m) => Transformer::operator(m),)*
                    $(Self::$transformer(m) => Transformer::operator(m),)*
                }
            }

            fn params(&self) -> &Params {
                match self {
                    $(Self::$model(m) => Transformer::params(m),)*
                    $(Self::$transformer(m) => Transformer::params(m),)*
                }
            }

            fn writer(&self, session: &SparkSession) -> Result<MlWriter> {
                Ok(match self {
                    $(Self::$model(m) => Model::write(m)?,)*
                    $(Self::$transformer(m) => m.write().session(session),)*
                })
            }

            fn load(&self, session: &SparkSession, path: &str) -> Result<Self> {
                Ok(match self {
                    $(Self::$model(_) => Self::$model($model::load(session, path)?),)*
                    $(Self::$transformer(_) => {
                        Self::$transformer($transformer::load(session, path)?)
                    })*
                })
            }
        }
    };
}

fitted! {
    models: [
        BucketedRandomProjectionLSHModel,
        ChiSqSelectorModel,
        CountVectorizerModel,
        IDFModel,
        ImputerModel,
        MaxAbsScalerModel,
        MinHashLSHModel,
        MinMaxScalerModel,
        OneHotEncoderModel,
        PCAModel,
        RFormulaModel,
        RobustScalerModel,
        StandardScalerModel,
        StringIndexerModel,
        TargetEncoderModel,
        UnivariateFeatureSelectorModel,
        VarianceThresholdSelectorModel,
        VectorIndexerModel,
        Word2VecModel,
    ],
    transformers: [Bucketizer],
}

/// The fit of an estimator into the model class of official PySpark.
trait TypedFit {
    type Model: Into<Fitted>;
    fn typed_fit(self, df: &DataFrame) -> Result<Self::Model>;
}

macro_rules! typed_fit {
    ($($estimator:ident => $model:ident),* $(,)?) => {
        $(impl TypedFit for $estimator {
            type Model = $model;
            fn typed_fit(mut self, df: &DataFrame) -> Result<$model> {
                Ok(self.fit(df)?)
            }
        })*
    };
}

typed_fit! {
    BucketedRandomProjectionLSH => BucketedRandomProjectionLSHModel,
    ChiSqSelector => ChiSqSelectorModel,
    CountVectorizer => CountVectorizerModel,
    IDF => IDFModel,
    Imputer => ImputerModel,
    MaxAbsScaler => MaxAbsScalerModel,
    MinHashLSH => MinHashLSHModel,
    MinMaxScaler => MinMaxScalerModel,
    OneHotEncoder => OneHotEncoderModel,
    PCA => PCAModel,
    RFormula => RFormulaModel,
    RobustScaler => RobustScalerModel,
    StandardScaler => StandardScalerModel,
    StringIndexer => StringIndexerModel,
    TargetEncoder => TargetEncoderModel,
    UnivariateFeatureSelector => UnivariateFeatureSelectorModel,
    VarianceThresholdSelector => VarianceThresholdSelectorModel,
    VectorIndexer => VectorIndexerModel,
    Word2Vec => Word2VecModel,
}

impl TypedFit for QuantileDiscretizer {
    type Model = Bucketizer;
    fn typed_fit(mut self, df: &DataFrame) -> Result<Bucketizer> {
        Ok(self.fit_bucketizer(df)?)
    }
}

/// A stage that saves and loads: its writer, its load, and the operator and params it keeps.
trait Persisted: Sized {
    fn writer(&self) -> MlWriter;
    fn read(session: &SparkSession, path: &str) -> Result<Self>;
    fn stage_operator(&self) -> &MlOperator;
    fn stage_params(&self) -> &Params;
}

macro_rules! persisted {
    (
        transformers: [$($transformer:ident),* $(,)?],
        estimators: [$($estimator:ident),* $(,)?] $(,)?
    ) => {
        $(impl Persisted for $transformer {
            fn writer(&self) -> MlWriter {
                self.write()
            }
            fn read(session: &SparkSession, path: &str) -> Result<Self> {
                Ok($transformer::load(session, path)?)
            }
            fn stage_operator(&self) -> &MlOperator {
                Transformer::operator(self)
            }
            fn stage_params(&self) -> &Params {
                Transformer::params(self)
            }
        })*
        $(impl Persisted for $estimator {
            fn writer(&self) -> MlWriter {
                Estimator::write(self)
            }
            fn read(session: &SparkSession, path: &str) -> Result<Self> {
                Ok($estimator::load(session, path)?)
            }
            fn stage_operator(&self) -> &MlOperator {
                Estimator::operator(self)
            }
            fn stage_params(&self) -> &Params {
                Estimator::params(self)
            }
        })*
    };
}

persisted! {
    transformers: [
        Binarizer, Bucketizer, DCT, ElementwiseProduct, FeatureHasher, HashingTF, IndexToString,
        Interaction, NGram, Normalizer, PolynomialExpansion, RegexTokenizer, SQLTransformer,
        StopWordsRemover, Tokenizer, VectorAssembler, VectorSizeHint, VectorSlicer,
    ],
    estimators: [
        BucketedRandomProjectionLSH, ChiSqSelector, CountVectorizer, IDF, Imputer, MaxAbsScaler,
        MinHashLSH, MinMaxScaler, OneHotEncoder, PCA, QuantileDiscretizer, RFormula, RobustScaler,
        StandardScaler, StringIndexer, TargetEncoder, UnivariateFeatureSelector,
        VarianceThresholdSelector, VectorIndexer, Word2Vec,
    ],
}

macro_rules! cases {
    (
        transformers: [$($tid:literal => $transformer:expr),* $(,)?],
        estimators: [$($eid:literal => $estimator:expr),* $(,)?] $(,)?
    ) => {
        /// The model that the estimator of the case `id` fits on `frame`.
        fn fitted(id: &str, frame: &DataFrame) -> Fitted {
            match id {
                $($eid => $estimator.typed_fit(frame).unwrap().into(),)*
                _ => panic!("uncovered estimator case {id}"),
            }
        }

        /// Replays the save and load of every transformer case and of every estimator case.
        fn persistence_replays(persistence: &Json, seen: &mut BTreeSet<String>) {
            $(round_trip(persistence, $tid, &$transformer, seen);)*
            $(round_trip(persistence, concat!($eid, ".estimator"), &$estimator, seen);)*
        }
    };
}

feature_cases!(cases);

/// The uid that a golden load read: the golden normalizes it to a placeholder, so the loaded uid
/// is the placeholder too.
fn result_uid(load: &Json) -> String {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use prost::Message;
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

/// The outcome of a member: its value, None where PySpark's getter raises a KeyError of an unset
/// param, or the message of its error.
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
    if entry["value"]["rows"].is_array() {
        return Some("a DataFrame: its rows are Arrow, which the replay does not serve");
    }
    match label {
        // Collected from a DataFrame of the server: the command is a relation, not an ML command.
        "categoryMaps" | "synonyms-array-word" | "synonyms-array-vector" | "synonyms-unknown" => {
            Some("collected from a DataFrame of the server, which the replay does not serve")
        }
        _ => None,
    }
}

/// The member `label` of a fitted model, as the golden labels it, or None for a member that the
/// replay does not check (see [`skip_reason`]).
fn member(model: &Fitted, label: &str) -> Option<Outcome> {
    use Fitted as F;
    Some(match (model, label) {
        (F::ChiSqSelectorModel(m), "selectedFeatures") => {
            outcome(m.selected_features(), |v| json!(v))
        }
        (F::CountVectorizerModel(m), "vocabulary") => outcome(m.vocabulary(), |v| json!(v)),
        (F::IDFModel(m), "docFreq") => outcome(m.doc_freq(), |v| json!(v)),
        (F::IDFModel(m), "idf") => outcome(m.idf(), |v| vector(&v)),
        (F::IDFModel(m), "numDocs") => outcome(m.num_docs(), |v| json!(v)),
        (F::MaxAbsScalerModel(m), "maxAbs") => outcome(m.max_abs(), |v| vector(&v)),
        (F::MinMaxScalerModel(m), "originalMax") => outcome(m.original_max(), |v| vector(&v)),
        (F::MinMaxScalerModel(m), "originalMin") => outcome(m.original_min(), |v| vector(&v)),
        (F::OneHotEncoderModel(m), "categorySizes") => outcome(m.category_sizes(), |v| json!(v)),
        (F::PCAModel(m), "explainedVariance") => outcome(m.explained_variance(), |v| vector(&v)),
        (F::PCAModel(m), "pc") => outcome(m.pc(), |v| matrix(&v)),
        (F::RobustScalerModel(m), "median") => outcome(m.median(), |v| vector(&v)),
        (F::RobustScalerModel(m), "range") => outcome(m.range(), |v| vector(&v)),
        (F::StandardScalerModel(m), "mean") => outcome(m.mean(), |v| vector(&v)),
        (F::StandardScalerModel(m), "std") => outcome(m.std(), |v| vector(&v)),
        (F::StringIndexerModel(m), "labels") => outcome(m.labels(), |v| json!(v)),
        (F::StringIndexerModel(m), "labelsArray") => outcome(m.labels_array(), |v| json!(v)),
        (F::UnivariateFeatureSelectorModel(m), "selectedFeatures") => {
            outcome(m.selected_features(), |v| json!(v))
        }
        (F::VarianceThresholdSelectorModel(m), "selectedFeatures") => {
            outcome(m.selected_features(), |v| json!(v))
        }
        (F::VectorIndexerModel(m), "numFeatures") => outcome(m.num_features(), |v| json!(v)),
        // The splits are params of the Bucketizer: None when the param is unset, which is
        // PySpark's KeyError for the other of a single and a multi-column discretizer.
        (F::Bucketizer(m), "getSplits()") => Ok(m.splits().map(|s| nums(&s))),
        (F::Bucketizer(m), "getSplitsArray()") => Ok(m
            .splits_array()
            .map(|a| a.iter().map(|s| nums(s)).collect())),
        _ => return None,
    })
}

/// Checks each member of a golden model entry (`members`) of `model`: its value or its error, and
/// its commands. The skipped members are left out, as [`skip_reason`] gives.
fn check_members(server: &ReplayServer, model: &Fitted, members: &Json, key: &str) {
    for (label, entry) in members.as_object().unwrap() {
        if skip_reason(label, entry).is_some() {
            continue;
        }
        let at = format!("{key}.{label}");
        let outcome = member(model, label).unwrap_or_else(|| panic!("{at}: no accessor"));
        match outcome {
            Ok(Some(actual)) => assert_matches(&actual, &entry["value"], &at),
            Ok(None) => assert_eq!(entry["error"]["class"], "KeyError", "{at}: an unset param"),
            Err(message) => assert_error(&message, &entry["error"], &at),
        }
        assert_commands(server, entry, &at);
    }
}

/// The entry without its Delete commands, and their results. PySpark releases the temporary model
/// of a QuantileDiscretizer's fit inside the fit, which sends a Delete of it in the entry. The
/// replay does not record a delete of a released object, so the commands that it compares leave
/// the deletes out too.
fn without_deletes(entry: &Json) -> Json {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use prost::Message;
    let is_delete = |command: &Json| {
        let bytes = STANDARD.decode(command.as_str().unwrap()).unwrap();
        let command = proto::MlCommand::decode(bytes.as_slice()).unwrap();
        matches!(command.command, Some(proto::ml_command::Command::Delete(_)))
    };
    let mut entry = entry.clone();
    let commands = entry["commands"].as_array().unwrap().clone();
    let kept: Vec<usize> = (0..commands.len())
        .filter(|&i| !is_delete(&commands[i]))
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

/// The entries of the members of a golden model that the replay checks. The others are left out
/// of the server: a DataFrame's command is an MlRelation, which the replay server does not decode
/// as an ML command.
fn checked_members(members: &Json) -> Vec<&Json> {
    members
        .as_object()
        .unwrap()
        .iter()
        .filter(|(label, entry)| skip_reason(label, entry).is_none())
        .map(|(_, entry)| entry)
        .collect()
}

/// The fit of the case `key`, its members, and the save and load of its model, with the golden
/// commands and values of each.
fn check_model(golden: &Json, persistence: &Json, key: &str) {
    let fit = &golden["fits"][key];
    let official = &golden["models"][key];
    assert_eq!(fit["dataset"], "features", "{key}");
    let model_key = format!("{key}.model");
    let record = &persistence[&model_key];
    assert!(
        record.is_object(),
        "{model_key}: no record of official PySpark"
    );
    assert_eq!(record["uid_same"], true, "{model_key}");
    assert_eq!(
        record["params_before"], record["params_after"],
        "{model_key}"
    );

    let fit_entry = without_deletes(&fit["fit"]);
    let mut entries = checked_members(&official["members"]);
    entries.extend([&fit_entry, &record["save"], &record["load"]]);
    let (server, session) = ReplayServer::start(&entries);
    // The replay normalizes the dataset of a Fit away, and the golden's dataset is a SQL statement
    // that the replay cannot run: the fit is on a one-row range, which the replay never executes.
    let input = session.range(1).unwrap();
    let model = fitted(key, &input);
    assert_commands(&server, &fit_entry, key);
    assert_eq!(model.class(), official["class"], "{key}");
    check_members(&server, &model, &official["members"], key);

    model
        .writer(&session)
        .unwrap()
        .overwrite()
        .save(PATH)
        .unwrap();
    assert_commands(&server, &record["save"], &model_key);
    let loaded = model.load(&session, PATH).unwrap();
    assert_commands(&server, &record["load"], &model_key);
    assert_eq!(loaded.operator().name, model.operator().name, "{model_key}");
    assert_eq!(
        loaded.operator().uid,
        result_uid(&record["load"]),
        "{model_key}"
    );
    assert_params(loaded.params(), model.params(), &model_key);
}

/// A pipeline's fit of the QuantileDiscretizer case sends the commands of the discretizer's own
/// fit, and gives PySpark's stage: a Bucketizer of a new uid, not the discretizer's uid.
#[test]
fn a_pipeline_fit_of_a_quantile_discretizer_gives_a_bucketizer_of_a_new_uid() {
    let golden = fixture(MODELS);
    let fit = without_deletes(&golden["fits"]["QuantileDiscretizer"]["fit"]);
    let (server, session) = ReplayServer::start(&[&fit]);
    let input = session.range(1).unwrap();
    let discretizer = QuantileDiscretizer::new()
        .set_input_col("x")
        .set_output_col("out")
        .set_num_buckets(3)
        .set_relative_error(0.0);
    let discretizer_uid = discretizer.operator().uid.clone();
    let mut pipeline = Pipeline::new().set_stages(vec![PipelineStage::estimator(discretizer)]);
    let model = pipeline.fit(&input).unwrap();
    assert_commands(&server, &fit, "QuantileDiscretizer.pipeline");
    let pipeline_model = (*model)
        .as_any()
        .downcast_ref::<PipelineModel>()
        .expect("a pipeline model");
    let PipelineStage::Transformer(bucketizer) = &pipeline_model.stages()[0] else {
        panic!("a Bucketizer is a transformer stage");
    };
    assert_eq!(
        bucketizer.operator().name,
        "org.apache.spark.ml.feature.Bucketizer"
    );
    assert_ne!(bucketizer.operator().uid, discretizer_uid);
    assert!(bucketizer.operator().uid.starts_with("Bucketizer_"));
}

/// Every fit of the golden, with its model's members and the save and load of the model.
#[test]
fn feature_models_match_official_pyspark() {
    let golden = fixture(MODELS);
    let persistence = fixture(PERSISTENCE)["persistence"].clone();
    let fits: BTreeSet<&str> = golden["fits"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let models: BTreeSet<&str> = golden["models"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(fits, models, "the fits and the models of the golden");
    for key in fits.iter().filter(|key| left_out(key).is_none()) {
        check_model(&golden, &persistence, key);
    }
    let model_records: BTreeSet<String> = persistence
        .as_object()
        .unwrap()
        .keys()
        .filter(|key| key.ends_with(".model") && left_out(key).is_none())
        .cloned()
        .collect();
    let replayed: BTreeSet<String> = fits
        .iter()
        .filter(|key| left_out(key).is_none())
        .map(|key| format!("{key}.model"))
        .collect();
    assert_eq!(replayed, model_records, "the model records of the golden");
}

/// The save and load of a transformer or an estimator, with the uid and the params that the golden
/// load reads. The records in [`LEFT_OUT`] are left out, and the loads in [`LOAD_LEFT_OUT`].
fn round_trip<T: Persisted>(persistence: &Json, key: &str, stage: &T, seen: &mut BTreeSet<String>) {
    if left_out(key).is_some() {
        return;
    }
    let record = &persistence[key];
    assert!(record.is_object(), "{key}: no record of official PySpark");
    assert_eq!(record["uid_same"], true, "{key}");
    assert_eq!(record["params_before"], record["params_after"], "{key}");
    let (server, session) = ReplayServer::start(&[&record["load"], &record["save"]]);
    if load_left_out(key).is_none() {
        let loaded = T::read(&session, PATH).unwrap();
        assert_commands(&server, &record["load"], key);
        assert_eq!(
            loaded.stage_operator().uid,
            result_uid(&record["load"]),
            "{key}"
        );
        assert_params(loaded.stage_params(), stage.stage_params(), key);
    }
    stage
        .writer()
        .session(&session)
        .overwrite()
        .save(PATH)
        .unwrap();
    assert_commands(&server, &record["save"], key);
    seen.insert(key.to_string());
}

/// Every transformer and estimator record of the golden, except the ones left out.
#[test]
fn transformer_and_estimator_persistence_matches_official_pyspark() {
    let persistence = fixture(PERSISTENCE)["persistence"].clone();
    let mut seen = BTreeSet::new();
    persistence_replays(&persistence, &mut seen);
    let official: BTreeSet<String> = persistence
        .as_object()
        .unwrap()
        .keys()
        .filter(|key| !key.ends_with(".model") && left_out(key).is_none())
        .cloned()
        .collect();
    assert_eq!(
        seen, official,
        "the transformer and estimator records of the golden"
    );
}

/// The class methods that build a model client-side: the helper fetch of each, of the uid of the
/// new model, which the replay normalizes as the golden does, or the error before any request.
#[test]
fn feature_model_class_methods_match_official_pyspark() {
    let golden = fixture(MODELS);
    let mut replayed = BTreeSet::new();
    for (key, official) in golden["class_methods"].as_object().unwrap() {
        if class_left_out(key).is_some() {
            continue;
        }
        let construct = &official["construct"];
        let (server, session) = ReplayServer::start(&[construct]);
        let built: Result<Fitted> = match key.as_str() {
            "CountVectorizerModel.from_vocabulary.empty" => CountVectorizerModel::from_vocabulary(
                &session,
                Vec::<String>::new(),
                "words",
                Some("out"),
                None,
                None,
            )
            .map(Into::into)
            .map_err(Into::into),
            "CountVectorizerModel.from_vocabulary.plain" => CountVectorizerModel::from_vocabulary(
                &session,
                ["alpha", "beta", "zeta"],
                "words",
                Some("out"),
                None,
                None,
            )
            .map(Into::into)
            .map_err(Into::into),
            "CountVectorizerModel.from_vocabulary.min-tf-binary" => {
                CountVectorizerModel::from_vocabulary(
                    &session,
                    ["alpha", "beta", "zeta"],
                    "words",
                    Some("out"),
                    Some(2.0),
                    Some(true),
                )
                .map(Into::into)
                .map_err(Into::into)
            }
            "StringIndexerModel.from_labels.plain" => StringIndexerModel::from_labels(
                &session,
                ["c", "a", "b"],
                "category",
                Some("out"),
                None,
            )
            .map(Into::into)
            .map_err(Into::into),
            "StringIndexerModel.from_labels.keep-unseen" => StringIndexerModel::from_labels(
                &session,
                ["c", "a"],
                "category",
                Some("out"),
                Some("keep"),
            )
            .map(Into::into)
            .map_err(Into::into),
            "StringIndexerModel.from_arrays_of_labels.plain" => {
                StringIndexerModel::from_arrays_of_labels(
                    &session,
                    [vec!["c", "a", "b"], vec!["alpha", "alpha beta"]],
                    ["category", "text"],
                    Some(vec!["out".to_string(), "other".to_string()]),
                    Some("keep"),
                )
                .map(Into::into)
                .map_err(Into::into)
            }
            _ => panic!("uncovered class method {key}"),
        };
        // The transform of a built model is a DataFrame, whose rows the replay does not serve: the
        // construction, a helper fetch of the uid of the new model, is what is compared.
        match built {
            Ok(model) => assert!(
                construct.get("error").is_none(),
                "{key}: built a {} where official PySpark raised",
                model.class()
            ),
            Err(error) => {
                assert!(official["transform"].is_null(), "{key}");
                assert_error(&error.to_string(), &construct["error"], key);
            }
        }
        assert_commands(&server, construct, key);
        replayed.insert(key.clone());
    }
    let official: BTreeSet<String> = golden["class_methods"]
        .as_object()
        .unwrap()
        .keys()
        .filter(|key| class_left_out(key).is_none())
        .cloned()
        .collect();
    assert_eq!(replayed, official, "the class methods of the golden");
}

/// A member whose request differs from its golden request fails the check: the replay refuses
/// the request, and the commands that were sent are not the golden's.
#[test]
fn a_request_that_differs_from_its_golden_fails_the_check() {
    let golden = fixture(MODELS);
    let fit = &golden["fits"]["IDF"]["fit"];
    let doc_freq = &golden["models"]["IDF"]["members"]["docFreq"];
    let (server, session) = ReplayServer::start(&[fit, doc_freq]);
    let input = session.range(1).unwrap();
    let Fitted::IDFModel(model) = fitted("IDF", &input) else {
        panic!("IDF is not an IDFModel");
    };
    assert_commands(&server, fit, "fit");

    // The golden has the fetch of `docFreq`: a fetch of `numDocs` is refused by the replay, and
    // the commands that were sent do not equal the golden's.
    assert!(model.num_docs().is_err());
    let check = catch_unwind(AssertUnwindSafe(|| {
        assert_commands(&server, doc_freq, "docFreq")
    }));
    assert!(check.is_err(), "the differing request passed the check");
}
