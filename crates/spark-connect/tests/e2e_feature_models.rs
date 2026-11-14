// SPDX-License-Identifier: Apache-2.0
//! The fitted feature models against a live server, as official PySpark 4.3.0 gives them of the
//! cases of tests/golden/ml_features.json. Compared with tests/golden/feature_models.json:
//! - every member of each model: the attributes, the transform, the LSH nearest neighbours and
//!   similarity joins, and the Word2Vec synonyms;
//! - the class methods that build a model client-side.
//!
//! Compared with tests/golden/feature_models_persistence.json: the save and load of every
//! transformer, model and estimator.
//!
//! scripts/capture_feature_models_golden.py captures both. Set SPARK_REMOTE.
mod common;
#[path = "common/feature_cases.rs"]
#[macro_use]
mod feature_cases;
#[path = "common/ml_golden.rs"]
mod ml_golden;

use std::collections::BTreeSet;
use std::path::Path;

use common::live_session;
use ml_golden::*;
use serde_json::{json, Value as Json};
use spark_connect::column::col;
use spark_connect::ml::*;
use spark_connect::{DataFrame, SparkSession};

/// The datasets of the golden: `features`, the frame of the feature parity tests, and `pairs`, its
/// id and features, which the similarity joins take.
fn datasets(session: &SparkSession, specs: &Json) -> (DataFrame, DataFrame) {
    let frame = dataset(session, &specs["features"]);
    let spec = &specs["pairs"];
    assert_eq!(spec["of"], "features");
    let pairs = frame.select(
        spec["select"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| col(name.as_str().unwrap())),
    );
    (frame, pairs)
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

            fn transformer(&mut self) -> &mut dyn Transformer {
                match self {
                    $(Self::$model(m) => m,)*
                    $(Self::$transformer(m) => m,)*
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
    (transformers: [$($transformer:ident),* $(,)?], estimators: [$($estimator:ident),* $(,)?] $(,)?) => {
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

/// The params that `loaded` has of each param that `saved` sets, by their values, as PySpark reads
/// them of their literals: an integer of any width (the server reads a seed back as a long), and
/// another value by its Debug text, which compares a NaN too.
fn assert_params(loaded: &Params, saved: &Params, at: &str) {
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

/// The save and load of a stage, as official PySpark's record of it has them: the uid and the
/// params that it was saved with.
fn round_trip<T: Persisted>(
    session: &SparkSession,
    key: &str,
    stage: &T,
    records: &Json,
    dir: &Path,
    seen: &mut BTreeSet<String>,
) {
    let record = &records[key];
    assert!(record.is_object(), "{key}: no record of official PySpark");
    seen.insert(key.to_string());
    assert_eq!(record["uid_same"], true, "{key}");
    assert_eq!(record["params_before"], record["params_after"], "{key}");
    let path = dir.join(key.replace(['.', '-'], "_"));
    let path = path.to_str().unwrap();
    stage
        .writer()
        .session(session)
        .overwrite()
        .save(path)
        .unwrap();
    let loaded = T::read(session, path).unwrap();
    assert_eq!(loaded.stage_operator().name, stage.stage_operator().name);
    assert_eq!(
        loaded.stage_operator().uid,
        stage.stage_operator().uid,
        "{key}"
    );
    assert_params(loaded.stage_params(), stage.stage_params(), key);
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

        /// Save and load the stage of every transformer case and the estimator of every
        /// estimator case.
        fn round_trips(session: &SparkSession, records: &Json, dir: &Path, seen: &mut BTreeSet<String>) {
            $(round_trip(session, $tid, &$transformer, records, dir, seen);)*
            $(round_trip(session, concat!($eid, ".estimator"), &$estimator, records, dir, seen);)*
        }
    };
}

feature_cases!(cases);

/// The member `label` of `model`, as the golden labels it: None of an unset param of the client,
/// which PySpark's getter raises a KeyError of.
fn member(
    model: &mut Fitted,
    label: &str,
    frame_: &DataFrame,
    pairs: &DataFrame,
) -> Result<Option<Json>> {
    use Fitted as F;
    let brp_key = MlVector::Dense(vec![1.0, 4.0, 1.0, 2.0]);
    let minhash_key = MlVector::Sparse {
        size: 4,
        indices: vec![1, 3],
        values: vec![1.0, 1.0],
    };
    let synonym = MlVector::Dense(vec![0.1, -0.2, 0.3]);
    let synonyms = |found: Vec<(String, f64)>| -> Json {
        found
            .into_iter()
            .map(|(word, similarity)| json!([word, num(similarity)]))
            .collect()
    };
    Ok(Some(match (model, label) {
        (m, "transform") => frame(&m.transformer().transform(frame_)?)?,
        (F::BucketedRandomProjectionLSHModel(m), "nearest-2") => {
            frame(&m.approx_nearest_neighbors(frame_, &brp_key, 2, "distCol")?)?
        }
        (F::BucketedRandomProjectionLSHModel(m), "nearest-3-dist") => {
            frame(&m.approx_nearest_neighbors(frame_, &brp_key, 3, "dist")?)?
        }
        (F::BucketedRandomProjectionLSHModel(m), "join-3") => {
            frame(&m.approx_similarity_join(pairs, pairs, 3.0, "distCol")?)?
        }
        (F::BucketedRandomProjectionLSHModel(m), "join-3-dist") => {
            frame(&m.approx_similarity_join(pairs, pairs, 3.0, "dist")?)?
        }
        (F::MinHashLSHModel(m), "nearest-2") => {
            frame(&m.approx_nearest_neighbors(frame_, &minhash_key, 2, "distCol")?)?
        }
        (F::MinHashLSHModel(m), "nearest-3-dist") => {
            frame(&m.approx_nearest_neighbors(frame_, &minhash_key, 3, "dist")?)?
        }
        (F::MinHashLSHModel(m), "join-0.6") => {
            frame(&m.approx_similarity_join(pairs, pairs, 0.6, "distCol")?)?
        }
        (F::MinHashLSHModel(m), "join-0.6-dist") => {
            frame(&m.approx_similarity_join(pairs, pairs, 0.6, "dist")?)?
        }
        (F::Bucketizer(m), "getSplits()") => return Ok(m.splits().map(|s| nums(&s))),
        (F::Bucketizer(m), "getSplitsArray()") => {
            return Ok(m
                .splits_array()
                .map(|a| a.iter().map(|s| nums(s)).collect()))
        }
        (F::ChiSqSelectorModel(m), "selectedFeatures") => json!(m.selected_features()?),
        (F::UnivariateFeatureSelectorModel(m), "selectedFeatures") => {
            json!(m.selected_features()?)
        }
        (F::VarianceThresholdSelectorModel(m), "selectedFeatures") => {
            json!(m.selected_features()?)
        }
        (F::CountVectorizerModel(m), "vocabulary") => json!(m.vocabulary()?),
        (F::IDFModel(m), "idf") => vector(&m.idf()?),
        (F::IDFModel(m), "docFreq") => json!(m.doc_freq()?),
        (F::IDFModel(m), "numDocs") => json!(m.num_docs()?),
        (F::ImputerModel(m), "surrogateDF") => frame(&m.surrogate_df()?)?,
        (F::MaxAbsScalerModel(m), "maxAbs") => vector(&m.max_abs()?),
        (F::MinMaxScalerModel(m), "originalMin") => vector(&m.original_min()?),
        (F::MinMaxScalerModel(m), "originalMax") => vector(&m.original_max()?),
        (F::OneHotEncoderModel(m), "categorySizes") => json!(m.category_sizes()?),
        (F::PCAModel(m), "pc") => matrix(&m.pc()?),
        (F::PCAModel(m), "explainedVariance") => vector(&m.explained_variance()?),
        (F::RobustScalerModel(m), "median") => vector(&m.median()?),
        (F::RobustScalerModel(m), "range") => vector(&m.range()?),
        (F::StandardScalerModel(m), "mean") => vector(&m.mean()?),
        (F::StandardScalerModel(m), "std") => vector(&m.std()?),
        (F::StringIndexerModel(m), "labels") => json!(m.labels()?),
        (F::StringIndexerModel(m), "labelsArray") => json!(m.labels_array()?),
        (F::VectorIndexerModel(m), "numFeatures") => json!(m.num_features()?),
        (F::VectorIndexerModel(m), "categoryMaps") => {
            // PySpark's dict of dicts, whose keys the golden renders with str().
            let maps: serde_json::Map<String, Json> = m
                .category_maps()?
                .into_iter()
                .map(|(feature, map)| {
                    let map: serde_json::Map<String, Json> = map
                        .into_iter()
                        .map(|(value, index)| (format!("{value:?}"), json!(index)))
                        .collect();
                    (feature.to_string(), Json::Object(map))
                })
                .collect();
            Json::Object(maps)
        }
        (F::Word2VecModel(m), "getVectors()") => frame(&m.get_vectors()?)?,
        (F::Word2VecModel(m), "synonyms-word") => frame(&m.find_synonyms("alpha", 2)?)?,
        (F::Word2VecModel(m), "synonyms-vector") => {
            frame(&m.find_synonyms_of_vector(&synonym, 2)?)?
        }
        (F::Word2VecModel(m), "synonyms-unknown") => frame(&m.find_synonyms("zeta", 2)?)?,
        (F::Word2VecModel(m), "synonyms-array-word") => {
            synonyms(m.find_synonyms_array("alpha", 2)?)
        }
        (F::Word2VecModel(m), "synonyms-array-vector") => {
            synonyms(m.find_synonyms_array_of_vector(&synonym, 2)?)
        }
        (m, label) => panic!("{} has no member {label} in this test", m.class()),
    }))
}

/// Each member of `members` (a golden `members` map) of `model`: its value, or its error. The
/// repr() and str() of the Python objects are the Python classes' to compare.
fn assert_members(
    model: &mut Fitted,
    members: &Json,
    frame: &DataFrame,
    pairs: &DataFrame,
    at: &str,
) {
    for (label, entry) in members.as_object().unwrap() {
        if label == "repr()" || label == "str()" {
            continue;
        }
        let at = format!("{at}.{label}");
        match (member(model, label, frame, pairs), entry.get("value")) {
            (Ok(Some(actual)), Some(expected)) => {
                assert_matches(&normalized(actual), &normalized(expected.clone()), &at)
            }
            (Ok(None), None) => assert_eq!(entry["error"]["class"], "KeyError", "{at}"),
            (Err(error), None) => assert_error(&error.to_string(), &entry["error"], &at),
            (actual, _) => panic!("{at}: {actual:?}, official {entry}"),
        }
    }
}

#[test]
fn feature_models_match_official_pyspark() {
    let Some(session) = live_session() else {
        return;
    };
    let golden = fixture(include_str!("../../../tests/golden/feature_models.json"));
    let records = fixture(include_str!(
        "../../../tests/golden/feature_models_persistence.json"
    ))["persistence"]
        .clone();
    let (frame, pairs) = datasets(&session, &golden["datasets"]);
    let dir = std::env::temp_dir().join(format!("feature-models-{}", std::process::id()));
    let mut seen = BTreeSet::new();

    for (key, fit) in golden["fits"].as_object().unwrap() {
        eprintln!("feature model: {key}");
        assert_eq!(fit["dataset"], "features");
        let mut model = fitted(key, &frame);
        let official = &golden["models"][key];
        assert_eq!(model.class(), official["class"], "{key}");
        assert_members(&mut model, &official["members"], &frame, &pairs, key);

        // The model saved and loaded back: its uid, its params, and the members of the copy.
        let model_key = format!("{key}.model");
        let record = &records[&model_key];
        assert_eq!(record["uid_same"], true, "{model_key}");
        let path = dir.join(model_key.replace(['.', '-'], "_"));
        let path = path.to_str().unwrap();
        model
            .writer(&session)
            .unwrap()
            .overwrite()
            .save(path)
            .unwrap();
        let mut loaded = model.load(&session, path).unwrap();
        assert_eq!(loaded.operator().name, model.operator().name, "{model_key}");
        assert_eq!(loaded.operator().uid, model.operator().uid, "{model_key}");
        assert_params(loaded.params(), model.params(), &model_key);
        assert_members(
            &mut loaded,
            &record["members"]["members"],
            &frame,
            &pairs,
            &model_key,
        );
        seen.insert(model_key);
    }

    round_trips(&session, &records, &dir, &mut seen);
    let official: BTreeSet<String> = records.as_object().unwrap().keys().cloned().collect();
    assert_eq!(seen, official);
    let _ = std::fs::remove_dir_all(&dir);
    session.stop().unwrap();
}

#[test]
fn feature_model_class_methods_match_official_pyspark() {
    let Some(session) = live_session() else {
        return;
    };
    let golden = fixture(include_str!("../../../tests/golden/feature_models.json"));
    let (frame, pairs) = datasets(&session, &golden["datasets"]);
    let words = ["alpha", "beta", "zeta"];
    for (key, official) in golden["class_methods"].as_object().unwrap() {
        let built: Result<Fitted> = match key.as_str() {
            "CountVectorizerModel.from_vocabulary.plain" => CountVectorizerModel::from_vocabulary(
                &session,
                words,
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
                    words,
                    "words",
                    Some("out"),
                    Some(2.0),
                    Some(true),
                )
                .map(Into::into)
                .map_err(Into::into)
            }
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
                    Some(vec!["out".into(), "other".into()]),
                    Some("keep"),
                )
                .map(Into::into)
                .map_err(Into::into)
            }
            _ => panic!("uncovered class method {key}"),
        };
        let construct = &official["construct"];
        match built {
            Ok(mut model) => {
                assert_eq!(construct["value"]["type"], model.class(), "{key}");
                assert_members(
                    &mut model,
                    &json!({"transform": official["transform"]}),
                    &frame,
                    &pairs,
                    key,
                );
            }
            Err(error) => {
                assert!(official["transform"].is_null(), "{key}");
                assert_error(&error.to_string(), &construct["error"], key);
            }
        }
    }
    session.stop().unwrap();
}
